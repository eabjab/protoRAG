"""In-process Okapi BM25 lexical search engine.

Pure Python / NumPy implementation with a dictionary-backed inverted index,
per-document token lengths, and a cached IDF table. No external service or
compiled extension is required.

Score(D, Q) = sum_i IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))
IDF(qi)    = ln( (N - n(qi) + 0.5) / (n(qi) + 0.5) + 1 )
"""

from __future__ import annotations

import math
import os
from collections import Counter
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from protorag.core.exceptions import LexicalError
from protorag.lexical.tokenizer import tokenize
from protorag.serialization.serializer import read_json, write_json_atomic

#: Per-term postings as (document indices, term frequencies) int64 arrays so
#: query-time scoring is vectorized, keeping BM25 search within its 2 ms
#: mean latency budget (see tests/benchmarks/test_latency_throughput.py).
PostingArrays = Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]


class BM25Engine:
    """Okapi BM25 ranker over an in-memory inverted index."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        lowercase: bool = True,
        stopwords: Optional[Sequence[str]] = None,
    ) -> None:
        if k1 < 0:
            raise LexicalError(f"BM25 k1 must be non-negative, got {k1!r}.")
        if not 0.0 <= b <= 1.0:
            raise LexicalError(f"BM25 b must be within [0, 1], got {b!r}.")
        self.k1 = float(k1)
        self.b = float(b)
        self.lowercase = bool(lowercase)
        self._stopwords: FrozenSet[str] = frozenset(
            (s.casefold() if lowercase else s) for s in (stopwords or ())
        )
        self._doc_ids: List[str] = []
        self._doc_id_to_idx: Dict[str, int] = {}
        self._doc_lengths: np.ndarray[Any, Any] = np.zeros(0, dtype=np.int64)
        self._inverted: Dict[str, PostingArrays] = {}
        self._idf_cache: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Index mutation
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._doc_ids)

    @property
    def doc_ids(self) -> List[str]:
        """Chunk ids currently held in the index, in insertion order."""
        return list(self._doc_ids)

    def add_documents(self, chunk_ids: Sequence[str], texts: Sequence[str]) -> None:
        """Indexes ``texts`` under ``chunk_ids`` (one id per text)."""
        if len(chunk_ids) != len(texts):
            raise LexicalError(
                f"chunk_ids ({len(chunk_ids)}) and texts ({len(texts)}) length mismatch."
            )
        base = len(self._doc_ids)
        new_lengths: List[int] = []
        new_postings: Dict[str, List[Tuple[int, int]]] = {}
        for i, (chunk_id, text) in enumerate(zip(chunk_ids, texts)):
            if chunk_id in self._doc_id_to_idx:
                raise LexicalError(f"Duplicate chunk id {chunk_id!r}.")
            tokens = tokenize(text, lowercase=self.lowercase, stopwords=self._stopwords)
            doc_idx = base + i
            self._doc_ids.append(chunk_id)
            self._doc_id_to_idx[chunk_id] = doc_idx
            new_lengths.append(len(tokens))
            for term, freq in Counter(tokens).items():
                new_postings.setdefault(term, []).append((doc_idx, freq))
        if new_lengths:
            self._doc_lengths = np.concatenate(
                [self._doc_lengths, np.asarray(new_lengths, dtype=np.int64)]
            )
        for term, posting in new_postings.items():
            pairs = np.asarray(posting, dtype=np.int64)
            arrays: PostingArrays = (pairs[:, 0].copy(), pairs[:, 1].copy())
            existing = self._inverted.get(term)
            if existing is None:
                self._inverted[term] = arrays
            else:
                self._inverted[term] = (
                    np.concatenate([existing[0], arrays[0]]),
                    np.concatenate([existing[1], arrays[1]]),
                )
        self._idf_cache.clear()

    def delete(self, chunk_ids: Sequence[str]) -> None:
        """Removes chunks from the index; unknown ids are ignored."""
        to_remove = {cid for cid in chunk_ids if cid in self._doc_id_to_idx}
        if not to_remove:
            return
        keep_mask = np.fromiter(
            (cid not in to_remove for cid in self._doc_ids),
            dtype=np.bool_,
            count=len(self._doc_ids),
        )
        new_index = np.cumsum(keep_mask, dtype=np.int64) - 1
        new_inverted: Dict[str, PostingArrays] = {}
        for term, (idxs, freqs) in self._inverted.items():
            keep = keep_mask[idxs]
            if not keep.any():
                continue
            new_inverted[term] = (new_index[idxs[keep]], freqs[keep])
        self._doc_ids = [cid for cid in self._doc_ids if cid not in to_remove]
        self._doc_id_to_idx = {cid: i for i, cid in enumerate(self._doc_ids)}
        self._doc_lengths = self._doc_lengths[keep_mask]
        self._inverted = new_inverted
        self._idf_cache.clear()

    def clear(self) -> None:
        """Flushes the entire inverted index."""
        self._doc_ids = []
        self._doc_id_to_idx = {}
        self._doc_lengths = np.zeros(0, dtype=np.int64)
        self._inverted = {}
        self._idf_cache = {}

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #

    def _idf(self, term: str) -> float:
        cached = self._idf_cache.get(term)
        if cached is not None:
            return cached
        n_docs = len(self._doc_ids)
        posting = self._inverted.get(term)
        doc_freq = 0 if posting is None else posting[0].shape[0]
        if n_docs == 0 or doc_freq == 0:
            value = 0.0
        else:
            value = math.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
        self._idf_cache[term] = value
        return value

    def _avg_doc_length(self) -> float:
        count = len(self._doc_lengths)
        if count == 0:
            return 0.0
        return float(self._doc_lengths.sum()) / count

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Ranks indexed chunks against ``query``.

        Returns up to ``top_k`` ``(chunk_id, score)`` pairs, best first.
        Unknown query terms are ignored; an empty corpus or empty query
        yields ``[]``.
        """
        if not self._doc_ids or top_k <= 0:
            return []
        query_terms = set(
            tokenize(query, lowercase=self.lowercase, stopwords=self._stopwords)
        )
        if not query_terms:
            return []
        avgdl = self._avg_doc_length()
        doc_lens = self._doc_lengths
        scores = np.zeros(len(self._doc_ids), dtype=np.float64)
        for term in sorted(query_terms):
            posting = self._inverted.get(term)
            if posting is None:
                continue
            idxs, freqs = posting
            idf = self._idf(term)
            denominator = freqs + self.k1 * (1.0 - self.b + self.b * (doc_lens[idxs] / avgdl))
            scores[idxs] += idf * (freqs * (self.k1 + 1)) / denominator
        matched = np.flatnonzero(scores)
        if matched.size == 0:
            return []
        order = np.lexsort((matched, -scores[matched]))
        top = matched[order[:top_k]]
        return [(self._doc_ids[int(doc_idx)], float(scores[doc_idx])) for doc_idx in top]

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Serializes the inverted index, doc lengths, and IDF cache to JSON."""
        payload: Dict[str, Any] = {
            "k1": self.k1,
            "b": self.b,
            "lowercase": self.lowercase,
            "stopwords": sorted(self._stopwords),
            "doc_ids": self._doc_ids,
            "doc_lengths": self._doc_lengths.tolist(),
            "inverted": {
                term: np.stack((idxs, freqs), axis=1).tolist()
                for term, (idxs, freqs) in self._inverted.items()
            },
            "idf_cache": self._idf_cache,
        }
        write_json_atomic(path, payload)

    def load(self, path: str) -> None:
        """Restores index state from a JSON file produced by :meth:`save`.

        Raises ``FileNotFoundError`` when the file is missing.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"BM25 index file not found: {path!r}.")
        payload = read_json(path)
        try:
            self.k1 = float(payload["k1"])
            self.b = float(payload["b"])
            self.lowercase = bool(payload["lowercase"])
            self._stopwords = frozenset(payload.get("stopwords", ()))
            self._doc_ids = list(payload["doc_ids"])
            self._doc_lengths = np.asarray(
                [int(length) for length in payload["doc_lengths"]], dtype=np.int64
            )
            inverted: Dict[str, PostingArrays] = {}
            for term, posting in payload["inverted"].items():
                pairs = np.asarray(posting, dtype=np.int64)
                if pairs.ndim != 2 or pairs.shape[1] != 2:
                    raise ValueError(f"malformed posting list for term {term!r}")
                inverted[term] = (pairs[:, 0].copy(), pairs[:, 1].copy())
            self._inverted = inverted
            idf_cache = payload.get("idf_cache", {})
            self._idf_cache = {term: float(value) for term, value in idf_cache.items()}
        except (KeyError, TypeError, ValueError) as err:
            raise LexicalError(f"Malformed BM25 index file {path!r}: {err}") from err
        if len(self._doc_ids) != len(self._doc_lengths):
            raise LexicalError(f"BM25 index file {path!r} has inconsistent document state.")
        self._doc_id_to_idx = {cid: i for i, cid in enumerate(self._doc_ids)}
