"""Hybrid ranking fusion: Reciprocal Rank Fusion and weighted linear combination.

Conventions used throughout:

* A *ranking* is a ``List[Tuple[chunk_id, score]]`` ordered best-first.
* ``rankings[0]`` is the vector retriever's ranking, ``rankings[1]`` the
  lexical (BM25) retriever's ranking.
* RRF uses ranks only (scores are ignored); the default smoothing constant is
  ``k = 60`` and per-retriever weights are ``1.0``.
* Linear fusion min-max normalizes each retriever's scores, then combines
  them as ``alpha * vector + (1 - alpha) * bm25``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from protorag.core.entities import FusionStrategy
from protorag.hybrid.normalizer import minmax_normalize

Ranking = List[Tuple[str, float]]

DEFAULT_RRF_K = 60


def rrf_fuse(
    rankings: Sequence[Ranking],
    k: int = DEFAULT_RRF_K,
    weights: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Reciprocal Rank Fusion over one or more ranked lists.

    ``RRF(d) = sum_m w_m / (k + Rank_m(d))`` where missing entries
    contribute zero.
    """
    if k <= 0:
        raise ValueError(f"RRF smoothing constant k must be positive, got {k!r}.")
    fused: Dict[str, float] = defaultdict(float)
    for model_idx, ranking in enumerate(rankings):
        weight = weights[model_idx] if weights is not None and model_idx < len(weights) else 1.0
        for rank, (chunk_id, _score) in enumerate(ranking, start=1):
            fused[chunk_id] += weight / (k + rank)
    return dict(fused)


def linear_fuse(score_maps: Sequence[Dict[str, float]], alphas: Sequence[float]) -> Dict[str, float]:
    """Weighted linear combination of min-max normalized score maps.

    ``len(alphas)`` must equal ``len(score_maps)``. Weights do not need to sum
    to one; they are applied after normalization.
    """
    if len(alphas) != len(score_maps):
        raise ValueError(
            f"Expected {len(score_maps)} alpha weights, got {len(alphas)}."
        )
    normalized = [minmax_normalize(scores) for scores in score_maps]
    fused: Dict[str, float] = defaultdict(float)
    for alpha, scores in zip(alphas, normalized):
        for chunk_id, score in scores.items():
            fused[chunk_id] += alpha * score
    return dict(fused)


def fuse(
    rankings: Sequence[Ranking],
    strategy: FusionStrategy = FusionStrategy.RRF,
    alpha: float = 0.5,
    rrf_k: int = DEFAULT_RRF_K,
    weights: Optional[Sequence[float]] = None,
) -> List[Tuple[str, float]]:
    """Merges retriever rankings into a single best-first list.

    Args:
        rankings: One ranking per retriever (vector first, then lexical).
        strategy: ``RRF`` or ``LINEAR``.
        alpha: Vector weight for linear fusion (``1.0`` => pure vector).
        rrf_k: RRF smoothing constant.
        weights: Optional per-retriever weights (RRF only).

    Returns:
        ``(chunk_id, fused_score)`` pairs sorted by descending score; ties are
        broken by chunk id for deterministic output.
    """
    if strategy is FusionStrategy.RRF:
        fused = rrf_fuse(rankings, k=rrf_k, weights=weights)
    elif strategy is FusionStrategy.LINEAR:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be within [0, 1], got {alpha!r}.")
        score_maps = [dict(ranking) for ranking in rankings]
        fused = linear_fuse(score_maps, [alpha, 1.0 - alpha])
    else:  # pragma: no cover - exhaustiveness guard
        raise ValueError(f"Unsupported fusion strategy: {strategy!r}")
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))
