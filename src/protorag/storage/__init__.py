"""Vector storage subsystem: protocol, backends, and factory registry."""

from protorag.storage.base import BaseVectorStore
from protorag.storage.registry import SUPPORTED_BACKENDS, VectorStoreRegistry

__all__ = [
    "SUPPORTED_BACKENDS",
    "BaseVectorStore",
    "VectorStoreRegistry",
]
