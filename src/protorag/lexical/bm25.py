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

from protorag.core.exceptions import LexicalError
from protorag.lexical.tokenizer import tokenize
from protorag.serialization.serializer import read_json, write_json_atomic

Posting = Tuple[int, int]  # (doc_index, term_frequency)


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
        self._doc_lengths: List[int] = []
        self._inverted: Dict[str, List[Posting]] = {}
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
        for i, (chunk_id, text) in enumerate(zip(chunk_ids, texts)):
            if chunk_id in self._doc_id_to_idx:
                raise LexicalError(f"Duplicate chunk id {chunk_id!r}.")
            tokens = tokenize(text, lowercase=self.lowercase, stopwords=self._stopwords)
            doc_idx = base + i
            self._doc_ids.append(chunk_id)
            self._doc_id_to_idx[chunk_id] = doc_idx
            self._doc_lengths.append(len(tokens))
            for term, freq in Counter(tokens).items():
                self._inverted.setdefault(term, []).append((doc_idx, freq))
        self._idf_cache.clear()

    def delete(self, chunk_ids: Sequence[str]) -> None:
        """Removes chunks from the index; unknown ids are ignored."""
        to_remove = {cid for cid in chunk_ids if cid in self._doc_id_to_idx}
        if not to_remove:
            return
        new_ids: List[str] = []
        new_lengths: List[int] = []
        new_inverted: Dict[str, List[Posting]] = {}
        for doc_idx, chunk_id in enumerate(self._doc_ids):
            if chunk_id in to_remove:
                continue
            new_idx = len(new_ids)
            new_ids.append(chunk_id)
            new_lengths.append(self._doc_lengths[doc_idx])
            for term, posting in self._inverted.items():
                for stored_idx, freq in posting:
                    if stored_idx == doc_idx:
                        new_inverted.setdefault(term, []).append((new_idx, freq))
        self._doc_ids = new_ids
        self._doc_id_to_idx = {cid: i for i, cid in enumerate(new_ids)}
        self._doc_lengths = new_lengths
        self._inverted = new_inverted
        self._idf_cache.clear()

    def clear(self) -> None:
        """Flushes the entire inverted index."""
        self._doc_ids = []
        self._doc_id_to_idx = {}
        self._doc_lengths = []
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
        doc_freq = len(self._inverted.get(term, ()))
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
        return sum(self._doc_lengths) / count

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
        scores: Dict[int, float] = {}
        for term in query_terms:
            posting = self._inverted.get(term)
            if not posting:
                continue
            idf = self._idf(term)
            for doc_idx, freq in posting:
                doc_len = self._doc_lengths[doc_idx]
                denominator = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / avgdl))
                scores[doc_idx] = scores.get(doc_idx, 0.0) + idf * (freq * (self.k1 + 1)) / denominator
        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [(self._doc_ids[doc_idx], float(score)) for doc_idx, score in ranked[:top_k]]

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
            "doc_lengths": self._doc_lengths,
            "inverted": {term: [[idx, tf] for idx, tf in posting] for term, posting in self._inverted.items()},
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
            self._doc_lengths = [int(length) for length in payload["doc_lengths"]]
            self._inverted = {
                term: [(int(idx), int(tf)) for idx, tf in posting]
                for term, posting in payload["inverted"].items()
            }
            self._idf_cache = {term: float(value) for term, value in payload.get("idf_cache", {}).items()}
        except (KeyError, TypeError, ValueError) as err:
            raise LexicalError(f"Malformed BM25 index file {path!r}: {err}") from err
        if len(self._doc_ids) != len(self._doc_lengths):
            raise LexicalError(f"BM25 index file {path!r} has inconsistent document state.")
        self._doc_id_to_idx = {cid: i for i, cid in enumerate(self._doc_ids)}
