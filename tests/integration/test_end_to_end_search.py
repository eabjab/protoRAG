"""End-to-end search acceptance tests.

The offline twin runs on the deterministic hashing embedder (no network);
the real-model test runs on the actual fastembed model and is marked
``network``.
"""

from __future__ import annotations

from typing import Any

import pytest

from protorag import Document, ProtoRAG, SearchMode
from tests.conftest import NGramHashingEmbedder

DOC1 = "The Apollo 11 mission landed humans on the Moon in July 1969."
DOC2 = "Python is a high-level, general-purpose programming language."
DOC3 = "Transformers and self-attention mechanisms revolutionized natural language processing."

SAMPLE_DOCS = [
    Document(id="doc1", content=DOC1),
    Document(id="doc2", content=DOC2),
    Document(id="doc3", content=DOC3),
]


def _offline_rag(vector_backend: str = "numpy") -> ProtoRAG:
    return ProtoRAG(
        vector_backend=vector_backend,
        embedding_backend="fastembed",
        embedder_instance=NGramHashingEmbedder(dimension=64),
    )


def test_end_to_end_offline_twin() -> None:
    rag = _offline_rag()
    rag.add_documents(list(SAMPLE_DOCS))
    assert len(rag) == 3

    bm25 = rag.search("Apollo 11 Moon 1969", top_k=1, mode=SearchMode.BM25)
    assert bm25 and bm25[0].document_id == "doc1"

    # The offline embedder is character n-gram based, so the vector query
    # shares n-grams with doc2's text; the semantic query that needs real
    # embeddings is covered by the network test below.
    vector = rag.search("Python programming language", top_k=1, mode=SearchMode.VECTOR)
    assert vector and vector[0].document_id == "doc2"

    hybrid = rag.search("Apollo space mission NLP", top_k=2, mode=SearchMode.HYBRID)
    assert len(hybrid) == 2
    assert all(result.rank > 0 for result in hybrid)


def test_end_to_end_string_modes_and_filter(make_rag: Any) -> None:
    rag = make_rag(vector_backend="numpy")
    rag.add_texts(
        ["The volcano Mount Fuji rises in Japan.", "The desert Sahara sprawls across Africa."],
        metadatas=[{"region": "asia"}, {"region": "africa"}],
    )
    # String enum values are accepted everywhere.
    assert len(rag.search("Sahara desert", top_k=1, mode="bm25")) == 1
    assert len(rag.search("Sahara desert", top_k=1, mode="vector")) == 1
    assert len(rag.search("Sahara desert", top_k=2, mode="hybrid", fusion_strategy="linear")) == 2
    # Metadata filtering (document-level key inherited by chunks).
    asia = rag.search("volcano mountain", top_k=5, filter_metadata={"region": "asia"})
    assert len(asia) == 1
    assert asia[0].metadata.get("region") == "asia"
    assert rag.search("volcano mountain", top_k=5, filter_metadata={"region": "europe"}) == []
    # Invalid enum values raise an actionable error.
    with pytest.raises(Exception, match="Invalid SearchMode"):
        rag.search("x", mode="quantum")  # type: ignore[arg-type]


def test_end_to_end_replacing_document(make_rag: Any) -> None:
    rag = make_rag(vector_backend="numpy")
    first = rag.add_documents([Document(id="doc1", content="Original content about planets.")])
    assert first == ["doc1"]
    # Re-adding the same id replaces the previous chunks.
    rag.add_documents([Document(id="doc1", content="Replaced content about oceans.")])
    assert len(rag) == 1
    results = rag.search("oceans replaced content", top_k=1, mode=SearchMode.BM25)
    assert results and results[0].content == "Replaced content about oceans."


@pytest.mark.network
def test_end_to_end_verbatim_spec(real_embedder: Any) -> None:
    """All three search modes with the real fastembed model (numpy backend)."""
    rag = ProtoRAG(vector_backend="numpy", embedding_backend="fastembed")
    rag.add_documents(list(SAMPLE_DOCS))

    results = rag.search("Apollo 11 Moon 1969", top_k=1, mode=SearchMode.BM25)
    assert results and results[0].document_id == "doc1"

    results = rag.search("coding in modern scripting languages", top_k=1, mode=SearchMode.VECTOR)
    assert results and results[0].document_id == "doc2"

    results = rag.search("Apollo space mission NLP", top_k=2, mode=SearchMode.HYBRID)
    assert len(results) == 2
