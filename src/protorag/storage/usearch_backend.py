"""High-performance C++ SIMD vector store backed by ``usearch``.

Runs entirely in-process (no daemon) with native AVX-512 / NEON SIMD
acceleration and fast binary (de)serialization via a memory-mapped
``index.usearch`` file.

Persistence layout (inside the ``vector_store/`` directory):

* ``index.usearch``    - native binary index dump
* ``store_config.json``- chunk-id <-> uint64 key mapping, dimension, metric
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import usearch

from protorag.core.entities import DistanceMetric
from protorag.core.exceptions import VectorStoreError
from protorag.serialization.serializer import read_json, write_json_atomic

_CONFIG_FILE = "store_config.json"
_INDEX_FILE = "index.usearch"


def _compiled() -> Any:
    """Return the ``usearch.compiled`` module, or raise a helpful error."""
    compiled = getattr(usearch, "compiled", None)
    if compiled is None:
        raise VectorStoreError(
            "This version of 'usearch' does not expose the compiled Index API. "
            "Upgrade with 'pip install --upgrade usearch' (a release providing "
            "'usearch.compiled.Index', e.g. >= 2.2)."
        )
    return compiled


def _metric_kind(compiled: Any, metric: DistanceMetric) -> Any:
    kind = compiled.MetricKind
    if metric is DistanceMetric.COSINE:
        return kind.Cos
    if metric is DistanceMetric.IP:
        return kind.InnerProduct
    return kind.L2sq


def _build_index(dimension: int, metric: DistanceMetric) -> Any:
    compiled = _compiled()
    try:
        return compiled.Index(ndim=int(dimension), metric_kind=_metric_kind(compiled, metric))
    except Exception as err:
        raise VectorStoreError(
            f"Failed to construct usearch index (dim={dimension}, metric={metric.value}): {err}"
        ) from err


class UsearchVectorStore:
    """ANN-exact HNSW vector index backed by Qdrant's ``usearch``."""

    def __init__(self) -> None:
        self._index: Optional[Any] = None
        self._dimension: Optional[int] = None
        self._metric: DistanceMetric = DistanceMetric.COSINE
        self._id_to_key: Dict[str, int] = {}
        self._key_to_id: Dict[int, str] = {}
        self._next_key = 0

    @property
    def backend(self) -> str:
        return "usearch"

    def initialize(self, dimension: int, metric: DistanceMetric = DistanceMetric.COSINE) -> None:
        if not isinstance(dimension, int) or dimension <= 0:
            raise VectorStoreError(f"dimension must be a positive integer, got {dimension!r}.")
        self._dimension = dimension
        self._metric = metric
        self._index = _build_index(dimension, metric)
        self._id_to_key = {}
        self._key_to_id = {}
        self._next_key = 0

    def __len__(self) -> int:
        return 0 if self._index is None else len(self._index)

    def _require_initialized(self) -> Any:
        if self._index is None:
            raise VectorStoreError(
                "UsearchVectorStore is not initialized; call initialize() first."
            )
        return self._index

    def _check_query(self, query_vector: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        query = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(-1)
        if self._dimension is not None and query.shape[0] != self._dimension:
            raise VectorStoreError(
                f"Expected query vector of dimension {self._dimension}, got {query.shape[0]}."
            )
        return query

    def _distance_to_score(self, distance: float) -> float:
        if self._metric is DistanceMetric.COSINE:
            return 1.0 - distance
        if self._metric is DistanceMetric.IP:
            return -distance
        return 1.0 / (1.0 + distance)

    def add(self, chunk_ids: Sequence[str], vectors: np.ndarray[Any, Any]) -> None:
        index = self._require_initialized()
        matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2:
            raise VectorStoreError(f"vectors must be 1-D or 2-D, got shape {matrix.shape}.")
        if self._dimension is not None and matrix.shape[1] != self._dimension:
            raise VectorStoreError(
                f"Expected vectors of dimension {self._dimension}, got {matrix.shape[1]}."
            )
        ids = list(chunk_ids)
        if len(ids) != matrix.shape[0]:
            raise VectorStoreError(
                f"chunk_ids ({len(ids)}) and vectors ({matrix.shape[0]}) count mismatch."
            )
        if not ids:
            return
        to_add_ids: List[str] = []
        to_add_vectors: List[np.ndarray[Any, Any]] = []
        for offset, chunk_id in enumerate(ids):
            existing_key = self._id_to_key.get(chunk_id)
            if existing_key is not None:  # upsert: drop the previous vector first
                self._remove_key(index, existing_key)
                del self._id_to_key[chunk_id]
                self._key_to_id.pop(existing_key, None)
            to_add_ids.append(chunk_id)
            to_add_vectors.append(matrix[offset])
        keys = np.fromiter(
            (self._next_key + i for i in range(len(to_add_ids))),
            dtype=np.uint64,
            count=len(to_add_ids),
        )
        try:
            index.add_many(keys, np.ascontiguousarray(np.stack(to_add_vectors), dtype=np.float32))
        except Exception as err:
            raise VectorStoreError(f"usearch add failed: {err}") from err
        for key, chunk_id in zip(keys.tolist(), to_add_ids):
            self._id_to_key[chunk_id] = int(key)
            self._key_to_id[int(key)] = chunk_id
        self._next_key += len(to_add_ids)

    def search(self, query_vector: np.ndarray[Any, Any], top_k: int = 10) -> List[Tuple[str, float]]:
        index = self._require_initialized()
        if top_k <= 0 or len(index) == 0:
            return []
        query = self._check_query(query_vector)
        k = min(top_k, len(index))
        try:
            results = index.search_many(query.reshape(1, -1), k)
        except Exception as err:
            raise VectorStoreError(f"usearch search failed: {err}") from err
        keys = results[0][0].tolist()
        distances = results[1][0].tolist()
        return [
            (self._key_to_id[int(key)], self._distance_to_score(float(distance)))
            for key, distance in zip(keys, distances)
        ]

    def delete(self, chunk_ids: Sequence[str]) -> None:
        index = self._require_initialized()
        for chunk_id in chunk_ids:
            key = self._id_to_key.get(chunk_id)
            if key is None:
                continue
            self._remove_key(index, key)
            del self._id_to_key[chunk_id]
            self._key_to_id.pop(key, None)

    @staticmethod
    def _remove_key(index: Any, key: int) -> None:
        try:
            index.remove_one(int(key), True, 1)
        except Exception as err:
            raise VectorStoreError(f"usearch delete failed: {err}") from err

    def save(self, directory: str) -> None:
        index = self._require_initialized()
        assert self._dimension is not None
        os.makedirs(directory, exist_ok=True)
        try:
            index.save_index_to_path(os.path.join(directory, _INDEX_FILE))
        except Exception as err:
            raise VectorStoreError(f"usearch save failed: {err}") from err
        config: Dict[str, Any] = {
            "backend": self.backend,
            "dimension": self._dimension,
            "metric": self._metric.value,
            "ids": self._key_to_id,
        }
        write_json_atomic(os.path.join(directory, _CONFIG_FILE), config)

    def load(self, directory: str) -> None:
        config_path = os.path.join(directory, _CONFIG_FILE)
        index_path = os.path.join(directory, _INDEX_FILE)
        if not os.path.exists(config_path) or not os.path.exists(index_path):
            raise FileNotFoundError(f"usearch vector store artifacts missing under {directory!r}.")
        config = read_json(config_path)
        try:
            dimension = int(config["dimension"])
            metric = DistanceMetric(str(config["metric"]))
            key_to_id = {int(key): str(cid) for key, cid in config["ids"].items()}
        except (KeyError, TypeError, ValueError) as err:
            raise VectorStoreError(f"Malformed vector store config {config_path!r}: {err}") from err
        compiled = _compiled()
        index: Any = compiled.Index()
        try:
            index.load_index_from_path(index_path)
        except Exception as err:
            raise VectorStoreError(f"usearch load failed: {err}") from err
        if len(index) != len(key_to_id):
            raise VectorStoreError(
                f"usearch index holds {len(index)} vectors but config lists {len(key_to_id)} ids."
            )
        self._dimension = dimension
        self._metric = metric
        self._index = index
        self._key_to_id = key_to_id
        self._id_to_key = {cid: key for key, cid in key_to_id.items()}
        self._next_key = (max(key_to_id) + 1) if key_to_id else 0
