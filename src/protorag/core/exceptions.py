"""protoRAG exception hierarchy.

Every exception raised by protoRAG derives from :class:`ProtoRAGException` so
callers can catch a single base type when wrapping the library.
"""

from __future__ import annotations


class ProtoRAGException(Exception):
    """Base class for all protoRAG errors."""


class EmbeddingError(ProtoRAGException):
    """Raised when an embedding backend fails to initialize or compute vectors."""


class VectorStoreError(ProtoRAGException):
    """Raised on invalid vector store operations (bad shapes, missing init, ...)."""


class LexicalError(ProtoRAGException):
    """Raised on invalid BM25 / lexical index operations."""


class ChunkingError(ProtoRAGException):
    """Raised when chunker configuration or input is invalid."""


class SerializationError(ProtoRAGException):
    """Raised when an index directory is corrupted or schema-incompatible."""


class IncompatibleBackendError(ProtoRAGException):
    """Raised when a persisted index requires a backend missing in this environment.

    The message always includes an actionable remediation: either install the
    missing dependency or pass a dimension-compatible substitute via
    ``ProtoRAG.load(path, override_embedder=...)``.
    """
