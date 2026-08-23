"""Lazy-loading factory for vector store backends.

Optional heavy dependencies (usearch, chromadb) are imported inside
:meth:`VectorStoreRegistry.create` so the CPU-only install stays lean and
missing optional backends produce an actionable ``ImportError``.
"""

from __future__ import annotations

from typing import Any

from protorag.core.entities import DistanceMetric
from protorag.storage.base import BaseVectorStore

SUPPORTED_BACKENDS = ("numpy", "usearch", "chromadb")


class VectorStoreRegistry:
    """Factory that instantiates :class:`BaseVectorStore` implementations by name."""

    @staticmethod
    def create(
        backend: str,
        dimension: int,
        metric: DistanceMetric = DistanceMetric.COSINE,
        **kwargs: Any,
    ) -> BaseVectorStore:
        key = backend.lower().strip()
        if key in ("numpy", "np"):
            from protorag.storage.numpy_backend import NumpyVectorStore

            store: BaseVectorStore = NumpyVectorStore(**kwargs)
        elif key in ("usearch",):
            try:
                from protorag.storage.usearch_backend import UsearchVectorStore
            except ImportError as err:
                raise ImportError(
                    "usearch is not installed. Run 'pip install usearch' "
                    "or 'pip install protorag-cpu'."
                ) from err
            store = UsearchVectorStore(**kwargs)
        elif key in ("chroma", "chromadb"):
            try:
                from protorag.storage.chroma_backend import ChromaVectorStore
            except ImportError as err:
                raise ImportError(
                    "chromadb is not installed. Run \"pip install 'protorag[chroma]'\" "
                    "or 'pip install chromadb'."
                ) from err
            store = ChromaVectorStore(**kwargs)
        else:
            raise ValueError(
                f"Unknown vector store backend '{backend}'. "
                f"Supported: {list(SUPPORTED_BACKENDS)}"
            )
        store.initialize(dimension, metric)
        return store
