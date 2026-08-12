"""
score.py — ScoreFunction orchestrator and MetricWeights
========================================================

Pipeline
--------
::

    fitness(net_pnl) = aggregate( normalize( metrics(net_pnl) ), weights )

Three decoupled layers:

1. **Metrics** — compute raw scalar values from the P&L series.
2. **Normalizer** — map raw values to [0, 1] (rank-based by default).
3. **Aggregator** — weighted combination of normalized scores.

Bilevel context
---------------
In the dispersion optimizer the score is evaluated **at optimal weights**:

::

    fitness(S) = max_w score(net_pnl(S, w))

where *S* is a candidate stock subset and *w* is found by a continuous
solver (see :mod:`functions.dispersion.scoring.weight_solver`).

How to add a criterion in 3 lines
----------------------------------
.. code-block:: python

    from functions.dispersion.scoring.metrics import register_metric, ScoreContext
    import numpy as np

    @register_metric()
    class MyMetric:
        name = "my_metric"
        higher_is_better = True
        is_linear = False
        def compute(self, net_pnl: np.ndarray, ctx: ScoreContext) -> float:
            return float(np.median(net_pnl))

That's it.  The metric is automatically available in ``MetricWeights`` and
scored/normalized without any other code change.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from .aggregators import Aggregator, WeightedSum
from .metrics import (
    DEFAULT_METRICS,
    Metric,
    MetricRegistry,
    ScoreContext,
)
from .normalizers import Normalizer, QuantileNormalizer

__all__ = [
    "MetricWeights",
    "ScoreFunction",
]


# ---------------------------------------------------------------------------
# MetricWeights — validated, immutable weight mapping
# ---------------------------------------------------------------------------


class MetricWeights:
    """Immutable mapping of metric names to preference weights.

    Parameters
    ----------
    weights:
        ``{metric_name: weight}`` dict.  Weights must be >= 0 and keys must
        correspond to registered metric names.  Auto-normalised to sum ≈ 1.

    Raises
    ------
    ValueError
        If any key is not a registered metric name or any value is negative.
    """

    def __init__(self, weights: Dict[str, float]) -> None:
        # Validate keys
        unknown = set(weights) - set(MetricRegistry)
        if unknown:
            raise ValueError(
                f"Unknown metric names: {sorted(unknown)}. "
                f"Registered: {sorted(MetricRegistry.keys())}"
            )

        # Validate values
        for k, v in weights.items():
            if v < 0:
                raise ValueError(f"Weight for '{k}' must be >= 0, got {v}")

        total = sum(weights.values())
        if total == 0:
            raise ValueError("At least one weight must be > 0")

        # Auto-normalize with warning
        if abs(total - 1.0) > 1e-6:
            warnings.warn(
                f"MetricWeights sum={total:.6f} ≠ 1; auto-normalising.",
                stacklevel=2,
            )
            weights = {k: v / total for k, v in weights.items()}

        self._data: Mapping[str, float] = MappingProxyType(dict(weights))

    # --- Mapping interface (read-only) ---

    def __getitem__(self, key: str) -> float:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def get(self, key: str, default: float = 0.0) -> float:
        return self._data.get(key, default)

    @property
    def active_names(self) -> List[str]:
        """Metric names with weight > 0."""
        return [k for k, v in self._data.items() if v > 0]

    def to_dict(self) -> Dict[str, float]:
        """Return a mutable copy of the weights dict."""
        return dict(self._data)

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v:.4f}" for k, v in self._data.items() if v > 0)
        return f"MetricWeights({items})"

    # --- Legacy compatibility ---

    @classmethod
    def from_legacy(
        cls,
        last_value: float = 0.3,
        avg_5y: float = 0.1,
        avg_3y: float = 0.2,
        max_drawdown: float = 0.1,
        hit_ratio: float = 0.3,
    ) -> "MetricWeights":
        """Create MetricWeights from old ScoreWeights fields.

        Mapping:
            last_value  → last_carry
            avg_5y + avg_3y → mean_payoff  (merged)
            max_drawdown → min_payoff
            hit_ratio → hit_ratio
        """
        weights: Dict[str, float] = {}
        if last_value > 0:
            weights["last_carry"] = last_value
        if avg_5y + avg_3y > 0:
            weights["mean_payoff"] = avg_5y + avg_3y
        if max_drawdown > 0:
            weights["min_payoff"] = max_drawdown
        if hit_ratio > 0:
            weights["hit_ratio"] = hit_ratio
        return cls(weights)


# ---------------------------------------------------------------------------
# ScoreFunction — three-layer orchestrator
# ---------------------------------------------------------------------------


class ScoreFunction:
    """Orchestrate metric computation, normalisation, and aggregation.

    Parameters
    ----------
    metrics:
        List of :class:`Metric` instances to evaluate.
    normalizer:
        A fitted (or to-be-fitted) :class:`Normalizer`.
    aggregator:
        An :class:`Aggregator` instance.
    weights:
        :class:`MetricWeights` specifying preference over metrics.
    """

    def __init__(
        self,
        metrics: Sequence[Metric],
        normalizer: Optional[Normalizer] = None,
        aggregator: Optional[Aggregator] = None,
        weights: Optional[MetricWeights] = None,
    ) -> None:
        self._metrics = list(metrics)
        self._normalizer = normalizer or QuantileNormalizer()
        self._aggregator = aggregator or WeightedSum()

        # Default: equal weight on all provided metrics
        if weights is None:
            n = len(self._metrics)
            weights = MetricWeights({m.name: 1.0 / n for m in self._metrics})
        self._weights = weights

        # Validate: all weighted metric names must exist in our metric list
        metric_names = {m.name for m in self._metrics}
        for name in self._weights.active_names:
            if name not in metric_names:
                raise ValueError(
                    f"Weighted metric '{name}' not in provided metrics list. "
                    f"Available: {sorted(metric_names)}"
                )

        self._fitted = False

    @property
    def metrics(self) -> List[Metric]:
        return list(self._metrics)

    @property
    def weights(self) -> MetricWeights:
        return self._weights

    @property
    def normalizer(self) -> Normalizer:
        return self._normalizer

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    # --- Build reference distribution ---

    def build_reference(
        self,
        sample_net_pnls: Iterable[np.ndarray],
        ctx: ScoreContext,
        sample_extras: Optional[Iterable[Optional[Dict[str, float]]]] = None,
    ) -> None:
        """Compute raw metrics on a reference sample and fit the normalizer.

        This MUST be called before :meth:`score` or :meth:`score_smooth`.
        The reference is **fixed for the entire optimization run** to ensure
        reproducibility and stable comparisons.

        Parameters
        ----------
        sample_net_pnls:
            Iterable of 1-D net-P&L arrays from feasible baskets.
        ctx:
            Score context (shared across all evaluations in this run).
        sample_extras:
            Optional iterable aligned with ``sample_net_pnls`` providing, per
            sample, extra raw metric values that cannot be computed from the
            P&L series alone (e.g. ``{"weighted_strike": 0.021}``).  Values
            given here override the computed ones for that sample.

        Notes
        -----
        Metrics whose reference ends up with no finite value (e.g.
        ``weighted_strike`` when no extras are supplied) are excluded from the
        normalizer fit instead of crashing it.  If such a metric carries a
        **positive weight**, a ``RuntimeError`` is raised — silently scoring
        an active metric at 0 would be a hidden behaviour change.
        """
        # Collect raw metric values across all samples
        raw_by_metric: Dict[str, List[float]] = {m.name: [] for m in self._metrics}

        extras_list = list(sample_extras) if sample_extras is not None else None

        for i, pnl in enumerate(sample_net_pnls):
            extra = extras_list[i] if (extras_list is not None and i < len(extras_list)) else None
            for m in self._metrics:
                if extra is not None and m.name in extra:
                    raw_by_metric[m.name].append(float(extra[m.name]))
                else:
                    raw_by_metric[m.name].append(m.compute(pnl, ctx))

        # Convert to arrays, drop NaN; metrics with an empty finite reference
        # are excluded from the fit (they stay usable only at weight 0).
        raw_arrays: Dict[str, np.ndarray] = {}
        self._unfitted_metrics: set = set()
        for name, vals in raw_by_metric.items():
            arr = np.array(vals, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size < 2:
                self._unfitted_metrics.add(name)
            else:
                raw_arrays[name] = arr

        active_unfitted = self._unfitted_metrics & set(self._weights.active_names)
        if active_unfitted:
            raise RuntimeError(
                f"build_reference: active metric(s) {sorted(active_unfitted)} have no "
                f"finite reference values. For 'weighted_strike', pass per-sample "
                f"values via sample_extras (or a ctx carrying weighted_strike)."
            )
        if self._unfitted_metrics:
            warnings.warn(
                f"build_reference: metrics {sorted(self._unfitted_metrics)} have no "
                f"finite reference — they stay inactive (weight 0) for this run.",
                stacklevel=2,
            )

        self._normalizer.fit(raw_arrays)
        self._fitted = True

    # --- Scoring methods ---

    def score(self, net_pnl: np.ndarray, ctx: ScoreContext) -> float:
        """Compute the full fitness score (step-function normalisation).

        Parameters
        ----------
        net_pnl:
            1-D raw net-P&L series.
        ctx:
            Score context.

        Returns
        -------
        float
            Aggregated score in approximately [0, 1].

        Raises
        ------
        RuntimeError
            If :meth:`build_reference` has not been called.
        """
        if not self._fitted:
            raise RuntimeError("ScoreFunction not fitted. Call build_reference() first.")

        raw = self._compute_raw(net_pnl, ctx)
        normalized = self._normalize(raw, smooth=False)
        return self._aggregator.aggregate(normalized, self._weights.to_dict())

    def score_smooth(self, net_pnl: np.ndarray, ctx: ScoreContext) -> float:
        """Compute score using the smoothed (interpolated) normalisation.

        Use this in the inner weight-optimisation loop where gradient-based
        solvers need a differentiable objective.

        Parameters
        ----------
        net_pnl:
            1-D raw net-P&L series.
        ctx:
            Score context.

        Returns
        -------
        float
            Aggregated smoothed score in approximately [0, 1].
        """
        if not self._fitted:
            raise RuntimeError("ScoreFunction not fitted. Call build_reference() first.")

        raw = self._compute_raw(net_pnl, ctx)
        normalized = self._normalize(raw, smooth=True)
        return self._aggregator.aggregate(normalized, self._weights.to_dict())

    def raw_metrics(self, net_pnl: np.ndarray, ctx: ScoreContext) -> Dict[str, float]:
        """Return raw metric values (before normalisation).

        Useful for diagnostics, tie-breaking, and corner-extremality checks.
        """
        return self._compute_raw(net_pnl, ctx)

    def score_smooth_from_raw(self, raw: Dict[str, float]) -> float:
        """Aggregate a caller-built raw dict through the smooth normalisation.

        Used by the inner weight solver, which substitutes differentiable
        surrogates for non-smooth raw metrics (e.g. ``soft_hit_ratio``) and
        injects portfolio-level values (e.g. ``weighted_strike``) before
        normalisation.  Keys missing from ``raw`` score 0 for that metric.

        Parameters
        ----------
        raw:
            ``{metric_name: raw_value}``.  Only metrics known to this
            ScoreFunction are considered.

        Returns
        -------
        float
            Aggregated smooth score.
        """
        if not self._fitted:
            raise RuntimeError("ScoreFunction not fitted. Call build_reference() first.")
        filled = {m.name: raw.get(m.name, float("nan")) for m in self._metrics}
        normalized = self._normalize(filled, smooth=True)
        return self._aggregator.aggregate(normalized, self._weights.to_dict())

    # --- Internals ---

    def _compute_raw(self, net_pnl: np.ndarray, ctx: ScoreContext) -> Dict[str, float]:
        """Evaluate all metrics on the given P&L series."""
        return {m.name: m.compute(net_pnl, ctx) for m in self._metrics}

    def _normalize(self, raw: Dict[str, float], smooth: bool) -> Dict[str, float]:
        """Normalize raw values to [0, 1] using the fitted normalizer."""
        normalized: Dict[str, float] = {}
        unfitted = getattr(self, "_unfitted_metrics", set())
        for m in self._metrics:
            if m.name in unfitted:
                normalized[m.name] = 0.0  # no reference — inactive by construction
                continue
            val = raw[m.name]
            if np.isnan(val):
                normalized[m.name] = 0.0  # NaN → worst possible score
                continue
            if smooth:
                normalized[m.name] = self._normalizer.transform_smooth(
                    m.name, val, m.higher_is_better
                )
            else:
                normalized[m.name] = self._normalizer.transform(
                    m.name, val, m.higher_is_better
                )
        return normalized


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def make_default_score_function(
    weights: Optional[MetricWeights] = None,
    last_carry_k: int = 1,
    mean_payoff_window: Optional[int] = None,
) -> ScoreFunction:
    """Build a ScoreFunction with the default metric set.

    Parameters
    ----------
    weights:
        Custom metric weights.  If None, uses equal weights on the 4 core
        metrics (last_carry, hit_ratio, min_payoff, mean_payoff).  Optional
        metrics (max_drawdown, cvar_5, sharpe_payoff, weighted_strike) are
        always instantiated at weight 0 and activate by receiving a weight.
    last_carry_k:
        Window for LastCarry metric.
    mean_payoff_window:
        Window for MeanPayoff metric. None = full series.

    Returns
    -------
    ScoreFunction
        Ready to be fitted via :meth:`~ScoreFunction.build_reference`.
    """
    from .metrics import (
        AxeBookCleaned,
        AxePackageRecycled,
        CVaR5,
        HitRatio,
        LastCarry,
        MaxDrawdown,
        MeanPayoff,
        MinPayoff,
        SharpePayoff,
        WeightedStrike,
    )

    # Core metrics + optional ones (weight 0 by default).  Optional metrics
    # are still part of the reference build, so giving them a weight later
    # activates them instantly without any code change.
    metrics: List[Metric] = [
        LastCarry(k=last_carry_k),
        HitRatio(),
        MinPayoff(),
        MeanPayoff(window=mean_payoff_window),
        MaxDrawdown(),
        CVaR5(),
        SharpePayoff(),
        WeightedStrike(),
        AxeBookCleaned(),
        AxePackageRecycled(),
    ]

    if weights is None:
        weights = MetricWeights(
            {
                "last_carry": 0.25,
                "hit_ratio": 0.25,
                "min_payoff": 0.25,
                "mean_payoff": 0.25,
            }
        )

    return ScoreFunction(
        metrics=metrics,
        normalizer=QuantileNormalizer(),
        aggregator=WeightedSum(),
        weights=weights,
    )
