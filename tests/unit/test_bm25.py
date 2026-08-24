"""Unit tests for the BM25 lexical engine, cross-checked against an
independent pure-Python Okapi BM25 reference."""

from __future__ import annotations

import math
import random
import re
from typing import Dict, List, Sequence, Tuple

import pytest

from protorag import BM25Engine, LexicalError
from protorag.lexical.tokenizer import DEFAULT_STOPWORDS
from protorag.serialization.serializer import read_json, write_json_atomic

_WORD_RE = re.compile(r"\w+", re.UNICODE)

CORPUS: Dict[str, str] = {
    "d1": "the cat sat on the mat",
    "d2": "the cat",
    "d3": "a dog sat on a mat",
}
WORD_POOL = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]


def _tokenize(text: str) -> List[str]:
    return [token.casefold() for token in _WORD_RE.findall(text)]


def reference_bm25(
    documents: Dict[str, str],
    query: str,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[Tuple[str, float]]:
    """Independent Okapi BM25 implementation (the cross-check oracle)."""
    tokens = {chunk_id: _tokenize(text) for chunk_id, text in documents.items()}
    n_docs = len(tokens)
    if n_docs == 0:
        return []
    avgdl = sum(len(toks) for toks in tokens.values()) / n_docs
    query_terms = set(_tokenize(query))

    def idf(term: str) -> float:
        doc_freq = sum(1 for toks in tokens.values() if term in toks)
        if doc_freq == 0:
            return 0.0
        return math.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

    scores: Dict[str, float] = {}
    for term in query_terms:
        for chunk_id, toks in tokens.items():
            freq = toks.count(term)
            if freq == 0:
                continue
            doc_len = len(toks)
            denom = freq + k1 * (1.0 - b + b * (doc_len / avgdl))
            scores[chunk_id] = scores.get(chunk_id, 0.0) + idf(term) * (freq * (k1 + 1)) / denom
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _engine_with_corpus(k1: float = 1.5, b: float = 0.75) -> BM25Engine:
    engine = BM25Engine(k1=k1, b=b)
    engine.add_documents(list(CORPUS.keys()), list(CORPUS.values()))
    return engine


def test_matches_reference_on_three_doc_corpus() -> None:
    engine = _engine_with_corpus()
    results = engine.search("cat sat mat", top_k=3)
    expected = reference_bm25(CORPUS, "cat sat mat")
    assert [chunk_id for chunk_id, _score in results] == [cid for cid, _s in expected] == [
        "d1",
        "d3",
        "d2",
    ]
    np_scores = [score for _cid, score in results]
    ref_scores = [score for _cid, score in expected]
    assert len(np_scores) == 3
    for actual, reference in zip(np_scores, ref_scores):
        assert abs(actual - reference) < 1e-9


@pytest.mark.parametrize(
    ("k1", "b"),
    [(1.5, 0.75), (1.0, 0.0), (2.0, 0.5), (0.5, 1.0), (0.0, 0.3)],
    ids=["default", "b0", "k2", "b1", "k05"],
)
def test_matches_reference_on_random_corpus(k1: float, b: float) -> None:
    rng = random.Random(1234)
    documents = {
        f"r{i}": " ".join(rng.choice(WORD_POOL) for _ in range(rng.randint(4, 12)))
        for i in range(5)
    }
    engine = BM25Engine(k1=k1, b=b)
    engine.add_documents(list(documents.keys()), list(documents.values()))

    for query in ["alpha beta zeta", "gamma theta epsilon", "alpha alpha beta", "no-such-word"]:
        results = engine.search(query, top_k=10)
        expected = reference_bm25(documents, query, k1=k1, b=b)
        assert [cid for cid, _score in results] == [cid for cid, _score in expected]
        assert len(results) == len(expected)
        for actual, reference in zip(results, expected):
            assert abs(actual[1] - reference[1]) < 1e-9


def test_casefolding_is_case_insensitive() -> None:
    engine = _engine_with_corpus()
    upper = engine.search("CAT MAT", top_k=3)
    lower = engine.search("cat mat", top_k=3)
    assert _id_scores(upper) == _id_scores(lower)


def test_stopwords_removed_when_configured() -> None:
    engine = BM25Engine(stopwords=DEFAULT_STOPWORDS)
    engine.add_documents(list(CORPUS.keys()), list(CORPUS.values()))
    with_stop = engine.search("the cat", top_k=3)
    without_stop = engine.search("cat", top_k=3)
    assert _id_scores(with_stop) == _id_scores(without_stop)


def test_empty_corpus_and_invalid_top_k() -> None:
    engine = BM25Engine()
    assert engine.search("anything", top_k=5) == []
    engine.add_documents(["a"], ["hello world"])
    assert engine.search("hello", top_k=0) == []
    assert engine.search("hello", top_k=-2) == []
    assert engine.search("no-such-token-anywhere", top_k=5) == []
    assert engine.search("", top_k=5) == []


def test_add_documents_validation() -> None:
    engine = BM25Engine()
    engine.add_documents(["a"], ["hello"])
    with pytest.raises(LexicalError, match="Duplicate chunk id"):
        engine.add_documents(["a"], ["again"])
    with pytest.raises(LexicalError, match="length mismatch"):
        engine.add_documents(["b", "c"], ["only one text"])


def test_parameter_validation() -> None:
    with pytest.raises(LexicalError, match="k1"):
        BM25Engine(k1=-0.5)
    with pytest.raises(LexicalError, match="b"):
        BM25Engine(b=-0.1)
    with pytest.raises(LexicalError, match="b"):
        BM25Engine(b=1.2)


def test_delete_removes_docs_and_ignores_unknown() -> None:
    engine = _engine_with_corpus()
    engine.delete(["d2", "nope"])
    assert engine.doc_ids == ["d1", "d3"]
    assert len(engine) == 2
    results = engine.search("cat", top_k=3)
    assert _id_scores(results) == [("d1", results[0][1])] if results else True


def test_save_load_preserves_scores(tmp_path) -> None:
    engine = _engine_with_corpus()
    path = str(tmp_path / "bm25_index.json")
    engine.save(path)

    restored = BM25Engine()
    restored.load(path)
    assert restored.doc_ids == engine.doc_ids
    assert (restored.k1, restored.b, restored.lowercase) == (engine.k1, engine.b, engine.lowercase)
    for query in ("cat sat mat", "cat", "dog"):
        assert _id_scores(restored.search(query, top_k=3)) == _id_scores(
            engine.search(query, top_k=3)
        )


def test_load_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        BM25Engine().load("no/such/bm25_index.json")


def test_load_malformed_file_raises(tmp_path) -> None:
    path = str(tmp_path / "bad.json")
    write_json_atomic(path, {"foo": "bar"})
    with pytest.raises(LexicalError, match="Malformed"):
        BM25Engine().load(path)

    # Structurally valid JSON but inconsistent document state.
    engine = _engine_with_corpus()
    consistent = str(tmp_path / "ok.json")
    engine.save(consistent)
    payload = read_json(consistent)
    payload["doc_lengths"] = payload["doc_lengths"][:1]
    broken = str(tmp_path / "broken.json")
    write_json_atomic(broken, payload)
    with pytest.raises(LexicalError, match="inconsistent"):
        BM25Engine().load(broken)


def _id_scores(results: Sequence[Tuple[str, float]]) -> List[Tuple[str, float]]:
    return [(cid, score) for cid, score in results]
