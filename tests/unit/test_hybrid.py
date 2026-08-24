"""Unit tests for hybrid fusion and score normalization."""

from __future__ import annotations

from typing import Dict, List, Tuple

import pytest

from protorag import FusionStrategy, fuse
from protorag.hybrid.fusion import DEFAULT_RRF_K, linear_fuse, rrf_fuse
from protorag.hybrid.normalizer import minmax_normalize, zscore_normalize

Ranking = List[Tuple[str, float]]

VECTOR_RANKING: Ranking = [("a", 0.9), ("b", 0.5)]
LEXICAL_RANKING: Ranking = [("b", 3.0), ("c", 1.0)]
# Full-map rankings (every id in every list) for the fuse() linear tests, so
# min-max normalization sees the same spread as the hand-computed scores.
VECTOR_RANKING_FULL: Ranking = [("a", 0.9), ("b", 0.5), ("c", 0.1)]
LEXICAL_RANKING_FULL: Ranking = [("a", 5.0), ("c", 4.0), ("b", 1.0)]
VECTOR_SCORES: Dict[str, float] = {"a": 0.9, "b": 0.5, "c": 0.1}
LEXICAL_SCORES: Dict[str, float] = {"a": 5.0, "b": 1.0, "c": 4.0}


def _order(pairs: List[Tuple[str, float]]) -> List[str]:
    return [chunk_id for chunk_id, _score in pairs]


# --------------------------------------------------------------------------- #
# Reciprocal Rank Fusion
# --------------------------------------------------------------------------- #


def test_rrf_hand_computed_scores() -> None:
    fused = rrf_fuse([VECTOR_RANKING, LEXICAL_RANKING], k=60)
    assert set(fused) == {"a", "b", "c"}
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["c"] == pytest.approx(1 / 62)
    assert _order(fuse([VECTOR_RANKING, LEXICAL_RANKING], FusionStrategy.RRF)) == ["b", "a", "c"]


def test_rrf_default_k() -> None:
    assert DEFAULT_RRF_K == 60
    fused = rrf_fuse([VECTOR_RANKING, LEXICAL_RANKING])
    assert fused["a"] == pytest.approx(1 / 61)


def test_rrf_weights_change_ordering() -> None:
    # Boosting the lexical list lifts "c" (rank 2 there) above "a".
    fused = fuse([VECTOR_RANKING, LEXICAL_RANKING], FusionStrategy.RRF, weights=[1.0, 2.0])
    assert _order(fused) == ["b", "c", "a"]


def test_rrf_invalid_k() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        rrf_fuse([VECTOR_RANKING], k=0)
    with pytest.raises(ValueError):
        rrf_fuse([VECTOR_RANKING], k=-5)


def test_rrf_ties_broken_by_chunk_id() -> None:
    rankings: List[Ranking] = [
        [("x", 1.0), ("z", 0.5)],
        [("w", 1.0), ("y", 0.5)],
    ]
    fused = fuse(rankings, FusionStrategy.RRF)
    # x and w tie at rank 1; y and z tie at rank 2. Ties break on ascending
    # chunk id, so "w" precedes "x" and "y" precedes "z".
    assert _order(fused) == ["w", "x", "y", "z"]


# --------------------------------------------------------------------------- #
# Linear fusion
# --------------------------------------------------------------------------- #


def test_linear_fusion_alpha_half_hand_computed() -> None:
    fused = linear_fuse([VECTOR_SCORES, LEXICAL_SCORES], [0.5, 0.5])
    # minmax: vector -> a=1, b=0.5, c=0 ; lexical -> a=1, b=0, c=0.75
    assert fused["a"] == pytest.approx(1.0, abs=1e-9)
    assert fused["b"] == pytest.approx(0.25, abs=1e-9)
    assert fused["c"] == pytest.approx(0.375, abs=1e-9)
    # fuse() normalizes each retriever's own score map: vector -> a=1, b=.5,
    # c=0; lexical -> a=1, b=0, c=.75. alpha=0.5 => a=1.0, c=0.375, b=0.25.
    assert _order(
        fuse([VECTOR_RANKING_FULL, LEXICAL_RANKING_FULL], FusionStrategy.LINEAR, alpha=0.5)
    ) == ["a", "c", "b"]


def test_linear_fusion_alpha_extremes() -> None:
    pure_vector = fuse(
        [VECTOR_RANKING_FULL, LEXICAL_RANKING_FULL], FusionStrategy.LINEAR, alpha=1.0
    )
    assert _order(pure_vector) == ["a", "b", "c"]
    pure_lexical = fuse(
        [VECTOR_RANKING_FULL, LEXICAL_RANKING_FULL], FusionStrategy.LINEAR, alpha=0.0
    )
    assert _order(pure_lexical) == ["a", "c", "b"]


def test_linear_fusion_alpha_out_of_range() -> None:
    with pytest.raises(ValueError, match="alpha"):
        fuse([VECTOR_RANKING, LEXICAL_RANKING], FusionStrategy.LINEAR, alpha=1.5)
    with pytest.raises(ValueError):
        fuse([VECTOR_RANKING, LEXICAL_RANKING], FusionStrategy.LINEAR, alpha=-0.1)


def test_linear_fuse_alpha_count_mismatch() -> None:
    with pytest.raises(ValueError, match="alpha weights"):
        linear_fuse([VECTOR_SCORES], [0.5, 0.5])


# --------------------------------------------------------------------------- #
# Normalizers
# --------------------------------------------------------------------------- #


def test_minmax_empty_and_constant() -> None:
    assert minmax_normalize({}) == {}
    assert minmax_normalize({"x": 3.0, "y": 3.0}) == {"x": 1.0, "y": 1.0}


def test_minmax_spread() -> None:
    normalized = minmax_normalize({"a": 2.0, "b": 4.0})
    assert normalized["a"] == pytest.approx(0.0, abs=1e-9)
    assert normalized["b"] == pytest.approx(1.0, abs=1e-6)
    assert all(0.0 <= value <= 1.0 for value in normalized.values())


def test_zscore_empty_and_constant() -> None:
    assert zscore_normalize({}) == {}
    assert zscore_normalize({"x": 7.0, "y": 7.0}) == {"x": 0.0, "y": 0.0}


def test_zscore_spread() -> None:
    normalized = zscore_normalize({"a": 1.0, "b": 2.0, "c": 3.0})
    values = list(normalized.values())
    assert sum(values) / 3 == pytest.approx(0.0, abs=1e-9)
    assert normalized["a"] == pytest.approx(-1.22474487, abs=1e-6)
    assert normalized["c"] == pytest.approx(1.22474487, abs=1e-6)
