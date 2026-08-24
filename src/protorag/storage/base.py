"""Vector store protocol.

All backends convert their native distance semantics into a *similarity
score* where **higher is better**:

* ``cosine``          -> cosine similarity in ``[-1, 1]``
* ``inner_product``   -> raw dot product (unbounded)
* ``l2``              -> ``1 / (1 + squared_distance)`` in ``(0, 1]``
"""

from __future__ import annotations

from typing import Any, List, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

from protorag.core.entities import DistanceMetric


@runtime_checkable
class BaseVectorStore(Protocol):
    """Structural interface for all vector storage backends."""

    @property
    def backend(self) -> str:
        """Backend key recorded in persistence manifests."""
        ...

    def initialize(self, dimension: int, metric: DistanceMetric = DistanceMetric.COSINE) -> None:
        """Initializes index data structures with the given dimensionality and metric."""
        ...

    def add(self, chunk_ids: Sequence[str], vectors: np.ndarray[Any, Any]) -> None:
        """Inserts vectors corresponding to ``chunk_ids`` into the index."""
        ...

    def search(self, query_vector: np.ndarray[Any, Any], top_k: int = 10) -> List[Tuple[str, float]]:
        """Returns up to ``top_k`` ``(chunk_id, similarity_score)`` pairs, best first."""
        ...

    def delete(self, chunk_ids: Sequence[str]) -> None:
        """Removes vectors corresponding to ``chunk_ids``; unknown ids are ignored."""
        ...

    def save(self, directory: str) -> None:
        """Serializes internal index state to ``directory``."""
        ...

    def load(self, directory: str) -> None:
        """Deserializes internal index state from ``directory``."""
        ...

    def __len__(self) -> int:
        """Total number of vectors in the index."""
        ...
