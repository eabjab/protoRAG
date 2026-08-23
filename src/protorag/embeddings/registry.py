"""Lazy-loading factory for embedding backends.

Imports happen inside :meth:`EmbedderRegistry.create` so that heavy optional
dependencies (torch, sentence-transformers) are only touched when actually
requested, keeping the CPU-only install lightweight.
"""

from __future__ import annotations

from typing import Any, Optional

from protorag.embeddings.base import BaseEmbedder

SUPPORTED_BACKENDS = ("fastembed", "sentence-transformers", "torch")


class EmbedderRegistry:
    """Factory that instantiates :class:`BaseEmbedder` implementations by name."""

    @staticmethod
    def create(
        backend: str,
        model_name: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseEmbedder:
        backend_lower = backend.lower().strip()
        if backend_lower in ("fastembed", "fast-embed", "fast_embed"):
            from protorag.embeddings.fastembed_backend import FastEmbedEmbedder

            return FastEmbedEmbedder(model_name=model_name, **kwargs)
        if backend_lower in ("sentence-transformers", "sentence_transformers", "sentence-transformer"):
            try:
                from protorag.embeddings.sentence_tx_backend import (
                    SentenceTransformersEmbedder,
                )
            except ImportError as err:
                raise ImportError(
                    "sentence-transformers is not installed. Run 'pip install protorag' "
                    "or 'pip install sentence-transformers'."
                ) from err
            return SentenceTransformersEmbedder(
                model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2", **kwargs
            )
        if backend_lower in ("torch", "pytorch"):
            try:
                from protorag.embeddings.torch_backend import TorchCustomEmbedder
            except ImportError as err:
                raise ImportError(
                    "PyTorch is not installed. Run 'pip install protorag' or install "
                    "PyTorch for your system."
                ) from err
            return TorchCustomEmbedder(model_name=model_name, **kwargs)
        raise ValueError(
            f"Unknown embedding backend '{backend}'. "
            f"Supported: {list(SUPPORTED_BACKENDS)}"
        )

    @staticmethod
    def is_available(backend: str) -> bool:
        """Returns True when the python package behind ``backend`` is importable."""
        import importlib.util

        backend_lower = backend.lower().strip()
        package = {
            "fastembed": "fastembed",
            "fast-embed": "fastembed",
            "fast_embed": "fastembed",
            "sentence-transformers": "sentence_transformers",
            "sentence_transformers": "sentence_transformers",
            "sentence-transformer": "sentence_transformers",
            "torch": "torch",
            "pytorch": "torch",
        }.get(backend_lower)
        if package is None:
            raise ValueError(f"Unknown embedding backend '{backend}'.")
        return importlib.util.find_spec(package) is not None


__all__ = ["SUPPORTED_BACKENDS", "EmbedderRegistry"]
