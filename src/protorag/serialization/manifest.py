"""Pydantic data models for the persistence manifest (``manifest.json``).

The manifest is the single source of truth for an index's engine
configuration: schema and package versions, embedding configuration,
vector store configuration, lexical index parameters, and content
statistics used for load-time consistency verification.
"""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field

from protorag._version import __version__
from protorag.core.entities import DistanceMetric
from protorag.core.exceptions import SerializationError

SUPPORTED_SCHEMA_MAJOR = 1
DEFAULT_SCHEMA_VERSION = "1.0.0"


def verify_schema_version(schema_version: str) -> None:
    """Raise :class:`SerializationError` when *schema_version* is unsupported.

    Only the major component is significant; ``"1.2.3"`` is accepted by
    protorag ``0.1.x`` because it is forward-compatible within major 1.
    """
    try:
        major = int(str(schema_version).split(".", 1)[0])
    except (ValueError, IndexError):
        raise SerializationError(
            f"Unparseable schema_version {schema_version!r} in manifest."
        ) from None
    if major != SUPPORTED_SCHEMA_MAJOR:
        raise SerializationError(
            f"Unsupported manifest schema_version {schema_version!r}. "
            f"protorag {__version__} supports major version {SUPPORTED_SCHEMA_MAJOR} "
            f"(e.g. {DEFAULT_SCHEMA_VERSION!r})."
        )


class EmbeddingConfig(BaseModel):
    """Configuration needed to reconstruct (or replace) the embedder."""

    backend: str
    model_name: str
    dimension: int
    kwargs: Dict[str, Any] = Field(default_factory=dict)


class VectorStoreConfig(BaseModel):
    """Configuration needed to reconstruct the vector store backend."""

    backend: str
    metric: DistanceMetric = DistanceMetric.COSINE
    dimension: int
    kwargs: Dict[str, Any] = Field(default_factory=dict)


class LexicalConfig(BaseModel):
    """Parameters of the in-process BM25 lexical index."""

    backend: str = "bm25"
    k1: float = 1.5
    b: float = 0.75
    lowercase: bool = True


class IndexStats(BaseModel):
    """Content statistics recorded at save time for consistency checks."""

    total_documents: int = 0
    total_chunks: int = 0


class Manifest(BaseModel):
    """Top-level ``manifest.json`` document."""

    schema_version: str = DEFAULT_SCHEMA_VERSION
    protorag_version: str = __version__
    created_at_utc: str
    embedding_config: EmbeddingConfig
    vector_store_config: VectorStoreConfig
    lexical_config: LexicalConfig = Field(default_factory=LexicalConfig)
    stats: IndexStats = Field(default_factory=IndexStats)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dict (enums serialized to their string values)."""
        return self.model_dump(mode="json")
