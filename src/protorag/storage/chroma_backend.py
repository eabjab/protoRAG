"""Feature-rich in-memory vector store backed by ``chromadb.EphemeralClient``.

A pure in-process, in-memory Chroma instance is used (no background server,
no disk WAL). Distances are translated to higher-is-better similarity
scores, matching :mod:`protorag.storage.base`.

Persistence layout (inside the ``vector_store/`` directory):

* ``chroma_backup/collection.json`` - ids, embeddings, dimension, metric
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chromadb
import numpy as np
from chromadb.config import Settings

from protorag.core.entities import DistanceMetric
from protorag.core.exceptions import VectorStoreError
from protorag.serialization.serializer import read_json, write_json_atomic

_BACKUP_DIR = "chroma_backup"
_COLLECTION_FILE = "collection.json"
_SPACE_BY_METRIC = {
    DistanceMetric.COSINE: "cosine",
    DistanceMetric.IP: "ip",
    DistanceMetric.L2: "l2",
}
_UPSERT_BATCH = 1024


def _new_client() -> Any:
    try:
        return chromadb.EphemeralClient(
            settings=Settings(anonymized_telemetry=False)
        )
    except TypeError:  # older chromadb without the settings parameter
        return chromadb.EphemeralClient()


class ChromaVectorStore:
    """HNSW vector index backed by an in-memory Chroma collection."""

    def __init__(self) -> None:
        self._client: Optional[Any] = None
        self._collection: Optional[Any] = None
        self._collection_name: Optional[str] = None
        self._dimension: Optional[int] = None
        self._metric: DistanceMetric = DistanceMetric.COSINE

    @property
    def backend(self) -> str:
        return "chromadb"

    def initialize(self, dimension: int, metric: DistanceMetric = DistanceMetric.COSINE) -> None:
        if not isinstance(dimension, int) or dimension <= 0:
            raise VectorStoreError(f"dimension must be a positive integer, got {dimension!r}.")
        self._dimension = dimension
        self._metric = metric
        self._client = _new_client()
        # Chromadb 1.x shares the collection registry across EphemeralClient
        # instances in one process, so each store gets a unique name to avoid
        # colliding with sibling stores (e.g. a live RAG next to a loaded one).
        self._collection_name = f"protorag_{uuid.uuid4().hex}"
        space = _SPACE_BY_METRIC[metric]
        try:
            collection = self._client.create_collection(
                self._collection_name, metadata={"hnsw:space": space}
            )
        except Exception as err:
            raise VectorStoreError(
                f"Failed to create chroma collection {self._collection_name!r}: {err}"
            ) from err
        self._collection = collection

    def __len__(self) -> int:
        if self._collection is None:
            return 0
        return int(self._collection.count())

    def _require_initialized(self) -> Any:
        if self._collection is None:
            raise VectorStoreError(
                "ChromaVectorStore is not initialized; call initialize() first."
            )
        return self._collection

    def _distance_to_score(self, distance: float) -> float:
        if self._metric is DistanceMetric.COSINE:
            return 1.0 - distance
        if self._metric is DistanceMetric.IP:
            return -distance
        return 1.0 / (1.0 + distance)

    def add(self, chunk_ids: Sequence[str], vectors: np.ndarray[Any, Any]) -> None:
        collection = self._require_initialized()
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
        try:
            for start in range(0, len(ids), _UPSERT_BATCH):
                stop = min(start + _UPSERT_BATCH, len(ids))
                collection.upsert(
                    ids=ids[start:stop],
                    embeddings=matrix[start:stop].tolist(),
                )
        except Exception as err:
            raise VectorStoreError(f"chroma upsert failed: {err}") from err

    def search(self, query_vector: np.ndarray[Any, Any], top_k: int = 10) -> List[Tuple[str, float]]:
        collection = self._require_initialized()
        if top_k <= 0 or collection.count() == 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if self._dimension is not None and query.shape[0] != self._dimension:
            raise VectorStoreError(
                f"Expected query vector of dimension {self._dimension}, got {query.shape[0]}."
            )
        k = min(top_k, int(collection.count()))
        try:
            result = collection.query(query_embeddings=[query.tolist()], n_results=k)
        except Exception as err:
            raise VectorStoreError(f"chroma query failed: {err}") from err
        ids = result["ids"][0]
        distances = result["distances"][0]
        return [(str(cid), self._distance_to_score(float(distance))) for cid, distance in zip(ids, distances)]

    def delete(self, chunk_ids: Sequence[str]) -> None:
        collection = self._require_initialized()
        ids = list(chunk_ids)
        if not ids:
            return
        try:
            existing = set(collection.get(ids=ids).get("ids", []))
            if existing:
                collection.delete(ids=sorted(existing))
        except Exception as err:
            raise VectorStoreError(f"chroma delete failed: {err}") from err

    def save(self, directory: str) -> None:
        collection = self._require_initialized()
        assert self._dimension is not None
        try:
            data = collection.get(include=["embeddings"])
        except Exception as err:
            raise VectorStoreError(f"chroma snapshot failed: {err}") from err
        # Chromadb 1.x returns embeddings as a single 2-D ndarray (n x d) and
        # ids as an ndarray; flatten both to JSON-safe python structures.
        raw_ids = data.get("ids")
        ids = [str(cid) for cid in raw_ids] if raw_ids is not None else []
        raw_embeddings = data.get("embeddings")
        embedding_rows: List[List[float]] = []
        if raw_embeddings is not None:
            arr = np.asarray(raw_embeddings, dtype=np.float32)
            if arr.size > 0:
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                embedding_rows = [[float(value) for value in row] for row in arr]
        payload: Dict[str, Any] = {
            "backend": self.backend,
            "dimension": self._dimension,
            "metric": self._metric.value,
            "ids": ids,
            "embeddings": embedding_rows,
        }
        write_json_atomic(
            os.path.join(directory, _BACKUP_DIR, _COLLECTION_FILE), payload
        )

    def load(self, directory: str) -> None:
        path = os.path.join(directory, _BACKUP_DIR, _COLLECTION_FILE)
        if not os.path.exists(path):
            raise FileNotFoundError(f"chroma collection backup not found: {path!r}.")
        payload = read_json(path)
        try:
            dimension = int(payload["dimension"])
            metric = DistanceMetric(str(payload["metric"]))
            ids: List[str] = [str(cid) for cid in payload.get("ids", [])]
            embeddings: List[List[float]] = [
                [float(value) for value in row] for row in payload.get("embeddings", [])
            ]
        except (KeyError, TypeError, ValueError) as err:
            raise VectorStoreError(f"Malformed chroma collection backup {path!r}: {err}") from err
        if len(ids) != len(embeddings):
            raise VectorStoreError(
                f"chroma backup {path!r} has {len(ids)} ids but {len(embeddings)} embeddings."
            )
        self.initialize(dimension, metric)
        collection = self._require_initialized()
        if ids:
            matrix = np.asarray(embeddings, dtype=np.float32)
            try:
                for start in range(0, len(ids), _UPSERT_BATCH):
                    stop = min(start + _UPSERT_BATCH, len(ids))
                    collection.upsert(ids=ids[start:stop], embeddings=matrix[start:stop].tolist())
            except Exception as err:
                raise VectorStoreError(f"chroma restore failed: {err}") from err
