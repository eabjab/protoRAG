"""Persistence subsystem: manifest models and atomic JSON / JSONL IO."""

from protorag.serialization.manifest import (
    DEFAULT_SCHEMA_VERSION,
    EmbeddingConfig,
    IndexStats,
    LexicalConfig,
    Manifest,
    VectorStoreConfig,
    verify_schema_version,
)
from protorag.serialization.serializer import (
    read_json,
    read_jsonl,
    utc_now_iso,
    write_json_atomic,
    write_jsonl_atomic,
)

__all__ = [
    "DEFAULT_SCHEMA_VERSION",
    "EmbeddingConfig",
    "IndexStats",
    "LexicalConfig",
    "Manifest",
    "VectorStoreConfig",
    "read_json",
    "read_jsonl",
    "utc_now_iso",
    "verify_schema_version",
    "write_json_atomic",
    "write_jsonl_atomic",
]
