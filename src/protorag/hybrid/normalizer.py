"""Score normalizers used before fusing heterogeneous retriever scores."""

from __future__ import annotations

from typing import Dict

_EPS = 1e-12


def minmax_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalizes scores into ``[0, 1]``.

    Constant (zero-spread) input maps every score to ``1.0`` so a single
    candidate does not collapse to a zero contribution.
    """
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    spread = hi - lo
    if spread < _EPS:
        return {chunk_id: 1.0 for chunk_id in scores}
    return {chunk_id: (value - lo) / spread for chunk_id, value in scores.items()}


def zscore_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Z-score normalizes scores (mean 0, unit variance); constant input -> 0.0."""
    if not scores:
        return {}
    values = list(scores.values())
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = variance ** 0.5
    if std < _EPS:
        return {chunk_id: 0.0 for chunk_id in scores}
    return {chunk_id: (value - mean) / std for chunk_id, value in scores.items()}
