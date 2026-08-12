"""normalizers.py — map raw metric values to [0, 1], scale-free.

═══════════════════════════════════════════════════════════════════════════════
SCAFFOLDING REPO DE TEST — NE PAS recopier dans le vrai repo.
Le vrai repo possède son propre normalizers.py ; ce fichier est une
reconstruction fidèle inférée des usages observés dans score.py /
weight_solver.py (fit(dict name→array), transform(name, value,
higher_is_better), transform_smooth(...)).  Si le comportement du vrai
normalizer diffère, c'est LUI la référence.
═══════════════════════════════════════════════════════════════════════════════

Contract (used by :class:`~functions.dispersion.scoring.score.ScoreFunction`):

- ``fit(reference)`` receives ``{metric_name: 1-D float array}`` containing
  only finite values, each array of size >= 2 (score.py filters beforehand).
- ``transform(name, value, higher_is_better)`` -> step-function normalisation
  in [0, 1].  Rank-based for :class:`QuantileNormalizer`: the fraction of the
  reference sample that the value beats.
- ``transform_smooth(name, value, higher_is_better)`` -> continuous,
  piecewise-linear interpolation of the same mapping (for gradient-based
  inner solvers).
- Both raise ``KeyError`` with an actionable message for a metric name that
  was never fitted — no silent fallback.
"""

from __future__ import annotations

import math
from typing import Dict, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "Normalizer",
    "QuantileNormalizer",
    "ZScoreNormalizer",
    "MinMaxNormalizer",
]


@runtime_checkable
class Normalizer(Protocol):
    """Structural protocol for normalizers (fit once, transform many)."""

    def fit(self, reference: Dict[str, np.ndarray]) -> None: ...

    def transform(self, name: str, value: float, higher_is_better: bool) -> float: ...

    def transform_smooth(self, name: str, value: float, higher_is_better: bool) -> float: ...


def _require_fitted(store: dict, name: str, cls_name: str):
    if name not in store:
        raise KeyError(
            f"{cls_name}: metric '{name}' has no fitted reference. "
            f"Fitted metrics: {sorted(store.keys())}. "
            f"Call fit() with a reference for this metric first."
        )


class QuantileNormalizer:
    """Rank-based (empirical CDF) normalisation against a fixed reference.

    ``transform`` returns the fraction of reference values the candidate
    beats (ties counted as beaten for ``higher_is_better=True``); a value of
    0.90 reads as "better than 90 % of the reference sample".  For
    ``higher_is_better=False`` the fraction of reference values that are
    >= value is returned, so lower raw values map to higher scores.

    ``transform_smooth`` linearly interpolates the ECDF between reference
    points (mid-rank plotting positions), clamped to [0, 1] outside the
    reference range — continuous and monotone, suitable for SLSQP.
    """

    #: Number of quantile knots for the smooth (interpolated) transform.
    #: Coarser than the raw reference on purpose: fewer kinks → smoother
    #: SLSQP landscape, while the step transform keeps full rank resolution.
    N_SMOOTH_KNOTS = 41

    def __init__(self) -> None:
        self._sorted: Dict[str, np.ndarray] = {}
        self._knots: Dict[str, np.ndarray] = {}

    def fit(self, reference: Dict[str, np.ndarray]) -> None:
        self._sorted = {}
        self._knots = {}
        for name, arr in reference.items():
            a = np.asarray(arr, dtype=np.float64)
            a = a[np.isfinite(a)]
            if a.size < 2:
                raise ValueError(
                    f"QuantileNormalizer.fit: reference for '{name}' needs >= 2 "
                    f"finite values, got {a.size}."
                )
            self._sorted[name] = np.sort(a)
            k = min(self.N_SMOOTH_KNOTS, a.size)
            self._knots[name] = np.quantile(self._sorted[name], np.linspace(0.0, 1.0, k))

    def transform(self, name: str, value: float, higher_is_better: bool) -> float:
        _require_fitted(self._sorted, name, "QuantileNormalizer")
        ref = self._sorted[name]
        n = ref.size
        if higher_is_better:
            frac = np.searchsorted(ref, value, side="right") / n
        else:
            frac = (n - np.searchsorted(ref, value, side="left")) / n
        return float(frac)

    def transform_smooth(self, name: str, value: float, higher_is_better: bool) -> float:
        _require_fitted(self._sorted, name, "QuantileNormalizer")
        knots = self._knots[name]
        # Piecewise-linear interpolation of the ECDF over a coarse quantile
        # grid (min → 0, max → 1): non-zero gradient across the whole
        # reference hull, exact agreement with the step transform at the
        # extremes, and far fewer kinks than raw-reference interpolation —
        # a smoother landscape for the gradient-based inner solver.
        positions = np.linspace(0.0, 1.0, knots.size)
        cdf = float(np.interp(value, knots, positions))
        return cdf if higher_is_better else 1.0 - cdf


class ZScoreNormalizer:
    """Gaussian-CDF normalisation: Φ((value − mean) / std) of the reference."""

    def __init__(self) -> None:
        self._stats: Dict[str, tuple] = {}

    def fit(self, reference: Dict[str, np.ndarray]) -> None:
        self._stats = {}
        for name, arr in reference.items():
            a = np.asarray(arr, dtype=np.float64)
            a = a[np.isfinite(a)]
            if a.size < 2:
                raise ValueError(
                    f"ZScoreNormalizer.fit: reference for '{name}' needs >= 2 "
                    f"finite values, got {a.size}."
                )
            self._stats[name] = (float(a.mean()), float(a.std(ddof=1)))

    def _cdf(self, name: str, value: float) -> float:
        mean, std = self._stats[name]
        if std <= 0.0:
            return 0.5
        z = (value - mean) / std
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def transform(self, name: str, value: float, higher_is_better: bool) -> float:
        _require_fitted(self._stats, name, "ZScoreNormalizer")
        c = self._cdf(name, value)
        return c if higher_is_better else 1.0 - c

    # Already smooth — same mapping.
    def transform_smooth(self, name: str, value: float, higher_is_better: bool) -> float:
        return self.transform(name, value, higher_is_better)


class MinMaxNormalizer:
    """Affine normalisation onto [0, 1] using the reference min/max."""

    def __init__(self) -> None:
        self._range: Dict[str, tuple] = {}

    def fit(self, reference: Dict[str, np.ndarray]) -> None:
        self._range = {}
        for name, arr in reference.items():
            a = np.asarray(arr, dtype=np.float64)
            a = a[np.isfinite(a)]
            if a.size < 2:
                raise ValueError(
                    f"MinMaxNormalizer.fit: reference for '{name}' needs >= 2 "
                    f"finite values, got {a.size}."
                )
            self._range[name] = (float(a.min()), float(a.max()))

    def transform(self, name: str, value: float, higher_is_better: bool) -> float:
        _require_fitted(self._range, name, "MinMaxNormalizer")
        lo, hi = self._range[name]
        if hi - lo <= 0.0:
            score = 0.5
        else:
            score = min(1.0, max(0.0, (value - lo) / (hi - lo)))
        return score if higher_is_better else 1.0 - score

    # Already piecewise-linear — same mapping.
    def transform_smooth(self, name: str, value: float, higher_is_better: bool) -> float:
        return self.transform(name, value, higher_is_better)
