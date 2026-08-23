"""Sentence-Transformers embedding backend (full ``protorag`` install).

Supports CUDA / MPS device acceleration and arbitrary sentence-transformers
models from the Hugging Face hub.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, cast

import numpy as np

from protorag.core.exceptions import EmbeddingError
from protorag.embeddings.base import l2_normalize

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceTransformersEmbedder:
    """Embedder backed by the ``sentence-transformers`` library."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as err:
            raise ImportError(
                "sentence-transformers is not installed. Run 'pip install protorag' "
                "or 'pip install sentence-transformers'."
            ) from err

        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:  # torch missing -> sentence-transformers will also fail later
                device = "cpu"

        self._model_name = model_name or DEFAULT_MODEL
        self._device = device
        self._kwargs: Dict[str, Any] = dict(kwargs)
        try:
            self._model = SentenceTransformer(self._model_name, device=device, **kwargs)
        except Exception as err:
            raise EmbeddingError(
                f"Failed to initialize sentence-transformers model '{self._model_name}': {err}"
            ) from err
        dim = self._model.get_sentence_embedding_dimension()
        if dim is None:
            dim = int(self._model.encode(["dimension probe"]).shape[1])
        self._dimension = int(dim)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def backend(self) -> str:
        return "sentence-transformers"

    @property
    def init_kwargs(self) -> Dict[str, Any]:
        """Serializable constructor kwargs recorded in persistence manifests."""
        out: Dict[str, Any] = {"device": self._device}
        out.update(self._kwargs)
        return out

    def embed_documents(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        batched = list(texts)
        if not batched:
            return np.zeros((0, self._dimension), dtype=np.float32)
        try:
            vectors = self._model.encode(
                batched,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as err:
            raise EmbeddingError(f"sentence-transformers encoding failed: {err}") from err
        return l2_normalize(np.asarray(vectors, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return cast("np.ndarray", self.embed_documents([text], 32)[0])
