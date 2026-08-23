"""FastEmbed (ONNX Runtime) embedding backend.

The default CPU backend: fast cold start, zero PyTorch dependency, fully
multi-threaded. Wraps Qdrant's ``fastembed`` ``TextEmbedding`` engine.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, cast

import numpy as np

from protorag.core.exceptions import EmbeddingError
from protorag.embeddings.base import l2_normalize

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# BGE models are asymmetric: the BAAI paper recommends prepending this
# instruction to queries while passages are embedded as-is.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class FastEmbedEmbedder:
    """Embedder backed by Qdrant's ``fastembed`` ONNX Runtime engine."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        threads: Optional[int] = None,
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as err:  # pragma: no cover - fastembed is a core dependency
            raise EmbeddingError(
                "fastembed is not installed. Run 'pip install fastembed' "
                "or 'pip install protorag-cpu'."
            ) from err

        self._model_name = model_name or DEFAULT_MODEL
        self._is_bge = "bge" in self._model_name.lower()
        self._threads = threads
        self._cache_dir = cache_dir
        self._kwargs: Dict[str, Any] = dict(kwargs)
        try:
            self._model = TextEmbedding(
                model_name=self._model_name, cache_dir=cache_dir, threads=threads, **kwargs
            )
        except Exception as err:
            raise EmbeddingError(
                f"Failed to initialize fastembed model '{self._model_name}': {err}"
            ) from err
        # fastembed >= 0.6 exposes an ``embedding_size`` property; older
        # releases (0.3-0.5) expose a ``dim()`` method.
        dimension: Optional[int] = None
        size_attr = getattr(self._model, "embedding_size", None)
        if isinstance(size_attr, int):
            dimension = size_attr
        else:
            dim_fn = getattr(self._model, "dim", None)
            if callable(dim_fn):
                dimension = int(dim_fn())
        if dimension is None:
            raise EmbeddingError(
                f"fastembed model '{self._model_name}' does not expose a fixed dimension."
            )
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def backend(self) -> str:
        return "fastembed"

    @property
    def init_kwargs(self) -> Dict[str, Any]:
        """Serializable constructor kwargs recorded in persistence manifests."""
        out: Dict[str, Any] = {}
        if self._threads is not None:
            out["threads"] = self._threads
        if self._cache_dir is not None:
            out["cache_dir"] = self._cache_dir
        out.update(self._kwargs)
        return out

    def embed_documents(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        return self._embed(list(texts), batch_size)

    def embed_query(self, text: str) -> np.ndarray:
        # BGE models are asymmetric: queries expect the canonical BGE prefix.
        query = BGE_QUERY_PREFIX + text if self._is_bge else text
        return cast("np.ndarray", self._embed([query], 32)[0])

    def _embed(self, texts: List[str], batch_size: int) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        try:
            vectors = [
                np.asarray(vec, dtype=np.float32)
                for vec in self._model.embed(texts, batch_size=batch_size)
            ]
        except Exception as err:
            raise EmbeddingError(f"fastembed embedding failed: {err}") from err
        stacked = np.stack(vectors).astype(np.float32, copy=False)
        return l2_normalize(stacked)
