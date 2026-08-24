"""Core domain entities and enumerations for protoRAG.

All public data structures are immutable dataclasses with strict typing so
they can be safely shared across threads and pickled for persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np


class SearchMode(str, Enum):
    """Retrieval mode selector for :meth:`ProtoRAG.search`."""

    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"


class DistanceMetric(str, Enum):
    """Distance / similarity metric understood by vector stores."""

    COSINE = "cosine"
    IP = "inner_product"
    L2 = "l2"


class FusionStrategy(str, Enum):
    """Strategy used to merge vector and lexical rankings in hybrid search."""

    RRF = "rrf"  # Reciprocal Rank Fusion
    LINEAR = "linear"  # Weighted Linear Score Combination


@dataclass(frozen=True)
class Document:
    """A source document ingested into the knowledge base."""

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A slice of a :class:`Document` that is embedded and indexed."""

    id: str
    document_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray[Any, Any]] = None


@dataclass(frozen=True)
class QueryResult:
    """A single search hit returned by :meth:`ProtoRAG.search`."""

    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    vector_score: Optional[float] = None
    lexical_score: Optional[float] = None
    rank: int = 1
