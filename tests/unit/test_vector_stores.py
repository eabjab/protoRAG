"""Unit tests for all vector store backends."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest

from protorag import DistanceMetric, VectorStoreError, VectorStoreRegistry
from protorag.storage.numpy_backend import NumpyVectorStore

BACKENDS = ["numpy", "usearch", "chromadb"]
METRICS = [DistanceMetric.COSINE, DistanceMetric.IP, DistanceMetric.L2]
DIMENSION = 16
N_VECTORS = 50


def _random_vectors(count: int = N_VECTORS, dim: int = DIMENSION, seed: int = 42) -> np.ndarray:
    """L2-normalized random vectors, mirroring real (normalized) embeddings.

    Normalization is required for the inner-product recall test: for
    unnormalized vectors, ``dot(v_i, v_j)`` can exceed ``||v_i||^2`` when
    ``||v_j|| > ||v_i||``, so the vector itself is not guaranteed to be its
    own top-1 match.
    """
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((count, dim)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / norms).astype(np.float32)


def _ids(count: int = N_VECTORS) -> List[str]:
    return [f"id_{i:03d}" for i in range(count)]


def _make_store(backend: str, metric: DistanceMetric) -> object:
    if backend == "chromadb":
        pytest.importorskip("chromadb")
    return VectorStoreRegistry.create(backend, dimension=DIMENSION, metric=metric)


def _search_ids(results: List[Tuple[str, float]]) -> List[str]:
    return [chunk_id for chunk_id, _score in results]


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_top1_recall_at_least_99_percent(backend: str, metric: DistanceMetric) -> None:
    """Each stored vector must be its own top-1 match (>= 99% recall)."""
    store = _make_store(backend, metric)
    vectors = _random_vectors()
    ids = _ids()
    store.add(ids, vectors)
    assert len(store) == N_VECTORS

    hits = 0
    for i in range(N_VECTORS):
        results = store.search(vectors[i], top_k=1)
        if results and results[0][0] == ids[i]:
            hits += 1
    recall = hits / N_VECTORS
    assert recall >= 0.99, f"{backend}/{metric} top-1 recall {recall:.2%} below 99%"


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_empty_store(backend: str, metric: DistanceMetric) -> None:
    store = _make_store(backend, metric)
    assert len(store) == 0
    assert store.search(_random_vectors(1, dim=DIMENSION)[0], top_k=5) == []
    store.add([], np.zeros((0, DIMENSION), dtype=np.float32))
    assert len(store) == 0


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_upsert_replaces_vector_for_same_id(backend: str, metric: DistanceMetric) -> None:
    store = _make_store(backend, metric)
    vectors = _random_vectors()
    store.add(["a", "b"], vectors[:2])
    # Re-insert "a" with a different vector: must replace, not duplicate.
    store.add(["a"], vectors[2:3])
    assert len(store) == 2
    assert set(_search_ids(store.search(vectors[2], top_k=1))) == {"a"}


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_delete_removes_and_ignores_unknown(backend: str, metric: DistanceMetric) -> None:
    store = _make_store(backend, metric)
    vectors = _random_vectors(3)
    store.add(["a", "b", "c"], vectors)
    store.delete(["b", "nope"])
    assert len(store) == 2
    remaining = set(_search_ids(store.search(vectors[0], top_k=3)))
    assert remaining == {"a", "c"}
    # Deleting again is a no-op.
    store.delete(["b"])
    assert len(store) == 2


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_save_load_roundtrip_preserves_search_results(
    backend: str, metric: DistanceMetric, tmp_path
) -> None:
    store = _make_store(backend, metric)
    vectors = _random_vectors()
    ids = _ids()
    store.add(ids, vectors)

    directory = tmp_path / "vector_store"
    store.save(str(directory))

    reloaded = _make_store(backend, metric)
    reloaded.load(str(directory))
    assert len(reloaded) == N_VECTORS

    for i in (0, 17, 49):
        original = store.search(vectors[i], top_k=10)
        restored = reloaded.search(vectors[i], top_k=10)
        assert _search_ids(original) == _search_ids(restored)
        np.testing.assert_allclose(
            [score for _id, score in original],
            [score for _id, score in restored],
            atol=1e-5,
        )


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_wrong_dimension_query_raises(backend: str, metric: DistanceMetric) -> None:
    store = _make_store(backend, metric)
    store.add(["a"], _random_vectors(1))
    with pytest.raises(VectorStoreError):
        store.search(np.zeros(3, dtype=np.float32), top_k=1)


@pytest.mark.parametrize("backend", BACKENDS)
def test_wrong_dimension_add_raises(backend: str) -> None:
    store = _make_store(backend, DistanceMetric.COSINE)
    with pytest.raises(VectorStoreError):
        store.add(["a"], np.zeros((1, 3), dtype=np.float32))


def test_registry_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="Unknown vector store backend"):
        VectorStoreRegistry.create("no-such-backend", dimension=8)


def test_registry_backend_aliases() -> None:
    assert VectorStoreRegistry.create("np", dimension=4).backend == "numpy"
    if pytest.importorskip("chromadb") is not None:
        assert VectorStoreRegistry.create("chroma", dimension=4).backend == "chromadb"


def test_uninitialized_numpy_store_raises() -> None:
    store = NumpyVectorStore()
    with pytest.raises(VectorStoreError):
        store.add(["a"], np.zeros((1, 4), dtype=np.float32))
    with pytest.raises(VectorStoreError):
        store.search(np.zeros(4, dtype=np.float32), top_k=1)


def test_numpy_store_rejects_invalid_dimension() -> None:
    with pytest.raises(VectorStoreError):
        NumpyVectorStore().initialize(0)
    with pytest.raises(VectorStoreError):
        NumpyVectorStore().initialize(-4)
