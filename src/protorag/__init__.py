"""protoRAG - zero-infrastructure in-memory RAG prototyping framework.

Primary entry points::

    from protorag import ProtoRAG, Document, SearchMode
"""

from protorag._version import __version__
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
from protorag.embeddings.base import BaseEmbedder
from protorag.embeddings.registry import EmbedderRegistry
from protorag.hybrid.fusion import fuse, linear_fuse, rrf_fuse
from protorag.hybrid.normalizer import minmax_normalize, zscore_normalize
from protorag.lexical.bm25 import BM25Engine
from protorag.lexical.tokenizer import tokenize
from protorag.serialization.manifest import (
    EmbeddingConfig,
    IndexStats,
    LexicalConfig,
    Manifest,
    VectorStoreConfig,
)
from protorag.storage.base import BaseVectorStore
from protorag.storage.registry import VectorStoreRegistry
from protorag.tools.hf_tool import ProtoRAGTool

__all__ = [
    "BM25Engine",
    "BaseChunker",
    "BaseEmbedder",
    "BaseVectorStore",
    "Chunk",
    "ChunkingError",
    "DistanceMetric",
    "Document",
    "EmbedderRegistry",
    "EmbeddingConfig",
    "EmbeddingError",
    "FusionStrategy",
    "IncompatibleBackendError",
    "IndexStats",
    "LexicalConfig",
    "LexicalError",
    "Manifest",
    "ProtoRAG",
    "ProtoRAGException",
    "ProtoRAGTool",
    "QueryResult",
    "RecursiveCharacterChunker",
    "SearchMode",
    "SerializationError",
    "SimpleCharacterChunker",
    "VectorStoreConfig",
    "VectorStoreError",
    "VectorStoreRegistry",
    "__version__",
    "fuse",
    "linear_fuse",
    "minmax_normalize",
    "rrf_fuse",
    "tokenize",
    "zscore_normalize",
]
