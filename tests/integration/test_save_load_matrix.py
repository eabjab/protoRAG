"""Save/load round-trip matrix: every vector backend x embedding strategy
(SPEC-001 §4.1, §4.2.2).

Offline twins load with a deterministic ``override_embedder`` and assert
top-k ID identity with score delta < 1e-5. The verbatim spec test uses the
real fastembed model (``network``).
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from protorag import ProtoRAG, QueryResult, SearchMode
from tests.conftest import NGramHashingEmbedder

TEXTS = [
    "Machine learning enables systems to learn from data.",
    "Neural networks mimic human brain structures.",
]
METADATAS = [{"topic": "AI"}, {"topic": "Bio-AI"}]
QUERY = "learning from data"

BACKENDS = ["numpy", "usearch", "chromadb"]
MODES = (SearchMode.VECTOR, SearchMode.BM25, SearchMode.HYBRID)


def _offline_rag(vector_backend: str) -> ProtoRAG:
    if vector_backend == "chromadb":
        pytest.importorskip("chromadb")
    return ProtoRAG(
        vector_backend=vector_backend,
        embedding_backend="fastembed",
        embedder_instance=NGramHashingEmbedder(dimension=64),
    )


def _snapshots(rag: ProtoRAG) -> Dict[SearchMode, List[QueryResult]]:
    return {mode: rag.search(QUERY, top_k=2, mode=mode) for mode in MODES}


def _assert_same_ranking(before: List[QueryResult], after: List[QueryResult]) -> None:
    assert [r.chunk_id for r in before] == [r.chunk_id for r in after]
    assert len(before) == len(after)
    for prev, cur in zip(before, after):
        assert abs(prev.score - cur.score) < 1e-5, (
            f"score drift {prev.score} vs {cur.score} for {prev.chunk_id}"
        )


@pytest.mark.parametrize("vector_backend", BACKENDS)
def test_save_load_offline_twin(tmp_path: Any, vector_backend: str) -> None:
    rag = _offline_rag(vector_backend)
    rag.add_texts(texts=TEXTS, metadatas=METADATAS)
    assert len(rag) == 2
    before = _snapshots(rag)
    assert before[SearchMode.VECTOR][0].content.startswith("Machine learning")
    assert before[SearchMode.VECTOR][0].metadata.get("topic") == "AI"

    path = tmp_path / "index"
    rag.save(str(path))

    loaded = ProtoRAG.load(str(path), override_embedder=NGramHashingEmbedder(dimension=64))
    assert len(loaded) == 2
    results = loaded.search(QUERY, top_k=1, mode=SearchMode.VECTOR)
    assert results
    assert "Machine learning" in results[0].content
    assert results[0].metadata.get("topic") == "AI"
    # Top-k IDs identical and scores stable across the round-trip.
    for mode in MODES:
        _assert_same_ranking(before[mode], loaded.search(QUERY, top_k=2, mode=mode))


@pytest.mark.parametrize("vector_backend", BACKENDS)
@pytest.mark.network
def test_save_load_verbatim_spec(tmp_path: Any, vector_backend: str, real_embedder: Any) -> None:
    """SPEC-001 §4.2.2 verbatim: real fastembed embeddings."""
    rag = ProtoRAG(vector_backend=vector_backend, embedding_backend="fastembed")
    rag.add_texts(texts=TEXTS, metadatas=METADATAS)
    rag.save(str(tmp_path / "index"))

    loaded = ProtoRAG.load(str(tmp_path / "index"))
    assert len(loaded) == 2
    results = loaded.search(QUERY, top_k=1, mode=SearchMode.VECTOR)
    assert results
    assert "Machine learning" in results[0].content
    assert results[0].metadata.get("topic") == "AI"
