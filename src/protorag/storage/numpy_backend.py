"""Zero-dependency NumPy vector store.

Vectors are kept in a contiguous float32 matrix with an id <-> row map.
Search is a single BLAS batched matrix product, which makes it the fastest
option below ~50k chunks.

Persistence layout (inside the ``vector_store/`` directory):

* ``vectors.npy``      - raw ``(N, D)`` float32 matrix
* ``store_config.json``- ids, dimension, and metric
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from protorag.core.entities import DistanceMetric
from protorag.core.exceptions import VectorStoreError
from protorag.embeddings.base import l2_normalize
from protorag.serialization.serializer import read_json, write_json_atomic

_CONFIG_FILE = "store_config.json"
_VECTORS_FILE = "vectors.npy"


class NumpyVectorStore:
    """Exact brute-force vector index backed by a NumPy matrix."""

    def __init__(self) -> None:
        self._dimension: Optional[int] = None
        self._metric: DistanceMetric = DistanceMetric.COSINE
        self._ids: List[str] = []
        self._id_to_row: Dict[str, int] = {}
        self._vectors: Optional[np.ndarray[Any, Any]] = None

    @property
    def backend(self) -> str:
        return "numpy"

    def initialize(self, dimension: int, metric: DistanceMetric = DistanceMetric.COSINE) -> None:
        if not isinstance(dimension, int) or dimension <= 0:
            raise VectorStoreError(f"dimension must be a positive integer, got {dimension!r}.")
        self._dimension = dimension
        self._metric = metric
        self._ids = []
        self._id_to_row = {}
        self._vectors = np.zeros((0, dimension), dtype=np.float32)

    def __len__(self) -> int:
        return len(self._ids)

    def _require_initialized(self) -> None:
        if self._dimension is None or self._vectors is None:
            raise VectorStoreError(
                "NumpyVectorStore is not initialized; call initialize() first."
            )

    def _as_matrix(self, vectors: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2:
            raise VectorStoreError(f"vectors must be 1-D or 2-D, got shape {matrix.shape}.")
        if self._dimension is not None and matrix.shape[1] != self._dimension:
            raise VectorStoreError(
                f"Expected vectors of dimension {self._dimension}, got {matrix.shape[1]}."
            )
        return matrix

    def add(self, chunk_ids: Sequence[str], vectors: np.ndarray[Any, Any]) -> None:
        self._require_initialized()
        matrix = self._as_matrix(vectors)
        ids = list(chunk_ids)
        if len(ids) != matrix.shape[0]:
            raise VectorStoreError(
                f"chunk_ids ({len(ids)}) and vectors ({matrix.shape[0]}) count mismatch."
            )
        if not ids:
            return
        duplicates = [cid for cid in ids if cid in self._id_to_row]
        if duplicates:
            self.delete(duplicates)
        assert self._vectors is not None
        if self._vectors.shape[0] == 0:
            self._vectors = matrix
        else:
            self._vectors = np.vstack((self._vectors, matrix))
        for offset, chunk_id in enumerate(ids):
            self._ids.append(chunk_id)
            self._id_to_row[chunk_id] = len(self._ids) - 1

    def search(self, query_vector: np.ndarray[Any, Any], top_k: int = 10) -> List[Tuple[str, float]]:
        self._require_initialized()
        if top_k <= 0 or not self._ids:
            return []
        assert self._vectors is not None
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if self._dimension is not None and query.shape[0] != self._dimension:
            raise VectorStoreError(
                f"Expected query vector of dimension {self._dimension}, got {query.shape[0]}."
            )
        if self._metric is DistanceMetric.COSINE:
            scores = l2_normalize(self._vectors) @ l2_normalize(query)
        elif self._metric is DistanceMetric.IP:
            scores = self._vectors @ query
        else:  # L2
            distances = np.sum((self._vectors - query) ** 2, axis=1)
            scores = 1.0 / (1.0 + distances)
        count = len(self._ids)
        k = min(top_k, count)
        if k < count:
            partitioned = np.argpartition(-scores, k - 1)[:k]
            order = partitioned[np.argsort(-scores[partitioned])]
        else:
            order = np.argsort(-scores)
        return [(self._ids[int(row)], float(scores[int(row)])) for row in order]

    def delete(self, chunk_ids: Sequence[str]) -> None:
        self._require_initialized()
        rows = [self._id_to_row[cid] for cid in chunk_ids if cid in self._id_to_row]
        if not rows:
            return
        assert self._vectors is not None
        keep_mask = np.ones(len(self._ids), dtype=bool)
        keep_mask[rows] = False
        keep_rows = np.nonzero(keep_mask)[0]
        self._ids = [self._ids[int(row)] for row in keep_rows]
        self._id_to_row = {cid: i for i, cid in enumerate(self._ids)}
        self._vectors = self._vectors[keep_rows]

    def save(self, directory: str) -> None:
        self._require_initialized()
        assert self._dimension is not None and self._vectors is not None
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, _VECTORS_FILE), self._vectors)
        config: Dict[str, Any] = {
            "backend": self.backend,
            "dimension": self._dimension,
            "metric": self._metric.value,
            "ids": self._ids,
        }
        write_json_atomic(os.path.join(directory, _CONFIG_FILE), config)

    def load(self, directory: str) -> None:
        config_path = os.path.join(directory, _CONFIG_FILE)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Vector store config not found: {config_path!r}.")
        config = read_json(config_path)
        try:
            dimension = int(config["dimension"])
            metric = DistanceMetric(str(config["metric"]))
            ids: List[str] = [str(cid) for cid in config["ids"]]
        except (KeyError, TypeError, ValueError) as err:
            raise VectorStoreError(f"Malformed vector store config {config_path!r}: {err}") from err
        vectors = np.load(os.path.join(directory, _VECTORS_FILE))
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1) if vectors.size else vectors.reshape(0, dimension)
        if vectors.shape[0] != len(ids):
            raise VectorStoreError(
                f"Vector count ({vectors.shape[0]}) does not match id count ({len(ids)}) "
                f"in {directory!r}."
            )
        self.initialize(dimension, metric)
        self._vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self._ids = ids
        self._id_to_row = {cid: i for i, cid in enumerate(ids)}
