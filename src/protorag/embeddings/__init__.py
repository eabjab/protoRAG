"""Embedding engine subsystem: protocol, backends, and factory registry."""

from protorag.embeddings.base import BaseEmbedder, l2_normalize
from protorag.embeddings.registry import SUPPORTED_BACKENDS, EmbedderRegistry

__all__ = [
    "SUPPORTED_BACKENDS",
    "BaseEmbedder",
    "EmbedderRegistry",
    "l2_normalize",
]
