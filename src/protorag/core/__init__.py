"""Core domain entities, exceptions, and chunking for protoRAG."""

from protorag.core.chunker import (
    BaseChunker,
    RecursiveCharacterChunker,
    SimpleCharacterChunker,
)
from protorag.core.engine import ProtoRAG
from protorag.core.entities import (
    Chunk,
    DistanceMetric,
    Document,
    FusionStrategy,
    QueryResult,
    SearchMode,
)
from protorag.core.exceptions import (
    ChunkingError,
    EmbeddingError,
    IncompatibleBackendError,
    LexicalError,
    ProtoRAGException,
    SerializationError,
    VectorStoreError,
)

__all__ = [
    "BaseChunker",
    "Chunk",
    "ChunkingError",
    "DistanceMetric",
    "Document",
    "EmbeddingError",
    "FusionStrategy",
    "IncompatibleBackendError",
    "LexicalError",
    "ProtoRAG",
    "ProtoRAGException",
    "QueryResult",
    "RecursiveCharacterChunker",
    "SearchMode",
    "SerializationError",
    "SimpleCharacterChunker",
    "VectorStoreError",
]
