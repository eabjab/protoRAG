"""Embedding engine protocol and shared vector utilities."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, cast, runtime_checkable

import numpy as np

_EPS = 1e-12


def l2_normalize(vectors: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Returns a float32 copy of ``vectors`` with unit L2 norm per row.

    Zero-norm rows are left untouched (they remain all-zero).
    """
    out = np.asarray(vectors, dtype=np.float32)
    if out.ndim == 1:
        norm = float(np.linalg.norm(out))
        return out / norm if norm > _EPS else out
    norms = np.linalg.norm(out, axis=-1, keepdims=True)
    norms = np.where(norms > _EPS, norms, 1.0)
    return cast("np.ndarray[Any, Any]", out / norms)


@runtime_checkable
class BaseEmbedder(Protocol):
    """Structural interface for all embedding backends."""

    @property
    def dimension(self) -> int:
        """Number of components in every produced embedding vector."""
        ...

    @property
    def model_name(self) -> str:
        """Identifier (HF repo id or module name) of the backing model."""
        ...

    @property
    def backend(self) -> str:
        """Backend key recorded in persistence manifests (e.g. ``fastembed``)."""
        ...

    def embed_documents(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray[Any, Any]:
        """Computes embeddings for a batch of documents.

        Returns a float32 array of shape ``(N, D)``.
        """
        ...

    def embed_query(self, text: str) -> np.ndarray[Any, Any]:
        """Computes embedding for a single query string.

        Returns a float32 array of shape ``(D,)``.
        """
        ...
