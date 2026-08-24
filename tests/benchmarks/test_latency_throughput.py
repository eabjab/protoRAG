"""Latency / throughput benchmarks: 1000 chunks of 500 chars.

Targets (mean latency, CPU): vector < 5 ms, BM25 < 2 ms, hybrid < 8 ms.
"""

from __future__ import annotations

from typing import List

import pytest

from protorag import ProtoRAG, SearchMode
from tests.conftest import NGramHashingEmbedder

N_DOCS = 1000
CHUNK_CHARS = 500
TOP_K = 10
QUERY = "topic 5 supporting details number 12"

# Mean latency budgets in milliseconds.
VECTOR_BUDGET_MS = 5.0
BM25_BUDGET_MS = 2.0
HYBRID_BUDGET_MS = 8.0


def _build_corpus() -> List[str]:
    texts: List[str] = []
    for i in range(N_DOCS):
        base = f"Document {i:04d} covers topic {i % 37} with supporting details number {i}."
        texts.append((base * 9)[:CHUNK_CHARS])  # 9x the shortest base (62 chars) >= 500
    assert all(len(text) == CHUNK_CHARS for text in texts[:3])
    return texts


@pytest.fixture(scope="module")
def benchmark_rag() -> ProtoRAG:
    rag = ProtoRAG(
        vector_backend="numpy",
        embedding_backend="fastembed",
        embedder_instance=NGramHashingEmbedder(dimension=64),
    )
    rag.add_texts(_build_corpus(), chunk_size=CHUNK_CHARS, chunk_overlap=0)
    assert len(rag) == N_DOCS
    return rag


def _mean_ms(benchmark: object) -> float:
    stats = benchmark.stats  # type: ignore[attr-defined]
    assert stats is not None
    return float(stats["mean"]) * 1000.0


def test_vector_search_latency(benchmark: object, benchmark_rag: ProtoRAG) -> None:
    benchmark(  # type: ignore[operator]
        lambda: benchmark_rag.search(QUERY, top_k=TOP_K, mode=SearchMode.VECTOR)
    )
    assert _mean_ms(benchmark) < VECTOR_BUDGET_MS


def test_bm25_search_latency(benchmark: object, benchmark_rag: ProtoRAG) -> None:
    benchmark(  # type: ignore[operator]
        lambda: benchmark_rag.search(QUERY, top_k=TOP_K, mode=SearchMode.BM25)
    )
    assert _mean_ms(benchmark) < BM25_BUDGET_MS


def test_hybrid_search_latency(benchmark: object, benchmark_rag: ProtoRAG) -> None:
    benchmark(  # type: ignore[operator]
        lambda: benchmark_rag.search(QUERY, top_k=TOP_K, mode=SearchMode.HYBRID)
    )
    assert _mean_ms(benchmark) < HYBRID_BUDGET_MS
