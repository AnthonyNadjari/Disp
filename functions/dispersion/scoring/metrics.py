"""
functions/dispersion/scoring/metrics.py
========================================
Metric Protocol, ScoreContext, MetricRegistry and all default metrics
for the dispersion-scoring layer.

Design notes
------------
* Every metric operates on the **raw net P&L series** (not cumulative).
* Metrics that are *linear in portfolio weights* expose ``is_linear = True``
  (currently :class:`LastCarry` and :class:`MeanPayoff`).
* Non-linear metrics (hit-ratio, min-payoff, CVaR, Sharpe) expose
  ``is_linear = False``.
* The module-level :data:`MetricRegistry` dict maps *metric name* -> *metric
  class* and is populated automatically via the :func:`register_metric`
  decorator.
* All computations use NumPy vectorised operations; no external dependencies
  beyond NumPy.

Usage
-----
>>> import numpy as np
>>> from functions.dispersion.scoring.metrics import (
...     ScoreContext, MetricRegistry, Sharpe, MeanPayoff
... )
>>> rng = np.random.default_rng(0)
>>> pnl = rng.normal(0.001, 0.02, 252)
>>> ctx = ScoreContext(n_days=252)
>>> print(MetricRegistry["sharpe_payoff"]().compute(pnl, ctx))
>>> print(MeanPayoff(window=63).compute(pnl, ctx))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar, Dict, Optional, Protocol, Type, runtime_checkable

import numpy as np

# ---------------------------------------------------------------------------
# MetricRegistry – populated by @register_metric
# ---------------------------------------------------------------------------

#: Module-level registry mapping metric name -> metric class.
MetricRegistry: Dict[str, Type["Metric"]] = {}


def register_metric(name: Optional[str] = None):
    """Class decorator that registers a :class:`Metric` implementation.

    Parameters
    ----------
    name:
        Override the registry key.  When *None* the class attribute
        ``cls.name`` is used.

    Examples
    --------
    >>> @register_metric()
    ... class MyMetric:
    ...     name = "my_metric"
    ...     higher_is_better = True
    ...     is_linear = False
    ...     def compute(self, net_pnl, ctx): ...

    >>> @register_metric("alias")
    ... class AnotherMetric:
    ...     name = "another_metric"
    ...     ...
    """
    def decorator(cls: Type) -> Type:
        key = name if name is not None else cls.name
        MetricRegistry[key] = cls
        return cls

    # Support both ``@register_metric`` (no call) and ``@register_metric()``
    # and ``@register_metric("custom_name")``.
    if callable(name):
        # Called as bare ``@register_metric`` without parentheses
        cls, name_inner = name, None  # type: ignore[assignment]
        key = cls.name  # type: ignore[union-attr]
        MetricRegistry[key] = cls
        return cls

    return decorator


# ---------------------------------------------------------------------------
# ScoreContext
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoreContext:
    """Immutable context object passed alongside the P&L array to every metric.

    Attributes
    ----------
    n_days:
        Total number of trading days in the evaluation window.
    ann_factor:
        Annualisation factor (default 252 for daily data).
    index:
        Optional array of the index / benchmark net-P&L series with the same
        length as the strategy ``net_pnl`` array, for future relative metrics.
    weighted_strike:
        Optional weighted-average net strike of the basket being scored
        (decimal, e.g. 0.023 = 2.3%).  Required only when the
        ``weighted_strike`` metric has a positive weight; callers that do not
        use that metric can ignore this field entirely.
    """

    n_days: int
    ann_factor: int = 252
    index: Optional[np.ndarray] = field(default=None, compare=False, hash=False)
    weighted_strike: Optional[float] = None


# ---------------------------------------------------------------------------
# Metric Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Metric(Protocol):
    """Structural protocol that all metrics must satisfy.

    Attributes
    ----------
    name:
        Unique snake_case identifier used as the registry key.
    higher_is_better:
        Semantics flag: ``True`` means a larger value is preferred (e.g.
        Sharpe), ``False`` means a smaller value is preferred (e.g. max-DD).
    is_linear:
        ``True`` when ``compute(w1*pnl1 + w2*pnl2, ctx)`` equals
        ``w1*compute(pnl1, ctx) + w2*compute(pnl2, ctx)`` for non-negative
        scalar weights *w*.  Enables fast portfolio-level aggregation.
    """

    name: str
    higher_is_better: bool
    is_linear: bool
    is_tail_metric: bool

    def compute(self, net_pnl: np.ndarray, ctx: ScoreContext) -> float:
        """Compute the scalar metric value.

        Parameters
        ----------
        net_pnl:
            1-D float array of **raw** (non-cumulative) net P&L observations,
            ordered oldest -> newest.
        ctx:
            Contextual parameters (horizon, annualisation factor, …).

        Returns
        -------
        float
            Scalar metric value.  ``np.nan`` should be returned when the
            computation is not feasible (e.g. empty series, zero variance).
        """
        ...


# ---------------------------------------------------------------------------
# Default metric implementations
# ---------------------------------------------------------------------------

@register_metric()
class LastCarry:
    """Mean of the last *K* net-P&L observations.

    This is the "recent carry" of the strategy.  It is **linear in portfolio
    weights** because the mean of a weighted sum equals the weighted sum of
    means.

    Parameters
    ----------
    k:
        Number of trailing observations to average.  Defaults to ``1``
        (i.e. the most recent day's P&L).
    """

    name: ClassVar[str] = "last_carry"
    higher_is_better: ClassVar[bool] = True
    is_linear: ClassVar[bool] = True
    is_tail_metric: ClassVar[bool] = False

    def __init__(self, k: int = 1) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        object.__setattr__(self, "_k", k)

    def compute(self, net_pnl: np.ndarray, ctx: ScoreContext) -> float:
        """Mean of the last *k* observations (recent carry).

        Parameters
        ----------
        net_pnl:
            Raw net-P&L series (1-D float array), oldest -> newest.
        ctx:
            Score context (unused by this metric but required by the protocol).

        Returns
        -------
        float
        """
        arr = np.asarray(net_pnl, dtype=float)
        if arr.size == 0:
            return math.nan
        tail = arr[-self._k:]
        return float(np.mean(tail))


@register_metric()
class HitRatio:
    """Fraction of days with strictly positive net P&L.

    .. math::

        \\text{HitRatio} = \\frac{1}{N}\\sum_{t=1}^{N} \\mathbf{1}[r_t > 0]

    This is a *step function* (``escalier``) in portfolio weights and is
    therefore **not linear**.

    Parameters
    ----------
    threshold:
        P&L threshold above which a day is counted as a "hit".
        Defaults to ``0.0``.
    """

    name: ClassVar[str] = "hit_ratio"
    higher_is_better: ClassVar[bool] = True
    is_linear: ClassVar[bool] = False
    is_tail_metric: ClassVar[bool] = False

    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = threshold

    def compute(self, net_pnl: np.ndarray, ctx: ScoreContext) -> float:
        """Return ``mean(net_pnl > threshold)``.

        Parameters
        ----------
        net_pnl:
            Raw net-P&L series (1-D float array), oldest -> newest.
        ctx:
            Score context (unused).

        Returns
        -------
        float
            Value in ``[0, 1]``.
        """
        arr = np.asarray(net_pnl, dtype=float)
        if arr.size == 0:
            return math.nan
        return float(np.mean(arr > self.threshold))


@register_metric()
class MinPayoff:
    """Minimum (worst) net-P&L observation over the full series.

    .. math::

        \\text{MinPayoff} = \\min_{t} r_t

    This is **non-smooth** in portfolio weights (it is a non-differentiable
    concave function of *w*) and therefore not linear.
    """

    name: ClassVar[str] = "min_payoff"
    higher_is_better: ClassVar[bool] = True
    is_linear: ClassVar[bool] = False
    is_tail_metric: ClassVar[bool] = True

    def compute(self, net_pnl: np.ndarray, ctx: ScoreContext) -> float:
        """Return ``min(net_pnl)``.

        Parameters
        ----------
        net_pnl:
            Raw net-P&L series (1-D float array), oldest -> newest.
        ctx:
            Score context (unused).

        Returns
        -------
        float
        """
        arr = np.asarray(net_pnl, dtype=float)
        if arr.size == 0:
            return math.nan
        return float(np.min(arr))


@register_metric()
class MeanPayoff:
    """Mean net-P&L over a trailing window of *W* observations.

    .. math::

        \\text{MeanPayoff} = \\frac{1}{W}\\sum_{t=N-W+1}^{N} r_t

    This metric is **linear in portfolio weights**.

    Parameters
    ----------
    window:
        Trailing window length.  When ``None`` (default) the full series is
        used.
    """

    name: ClassVar[str] = "mean_payoff"
    higher_is_better: ClassVar[bool] = True
    is_linear: ClassVar[bool] = True
    is_tail_metric: ClassVar[bool] = False

    def __init__(self, window: Optional[int] = None) -> None:
        if window is not None and window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.window = window

    def compute(self, net_pnl: np.ndarray, ctx: ScoreContext) -> float:
        """Return ``mean(net_pnl[-W:])`` where *W* is ``self.window``.

        When ``self.window`` is *None* the full series is averaged.

        Parameters
        ----------
        net_pnl:
            Raw net-P&L series (1-D float array), oldest -> newest.
        ctx:
            Score context (used for a fallback window equal to ``ctx.n_days``
            when ``self.window`` is *None*).

        Returns
        -------
        float
        """
        arr = np.asarray(net_pnl, dtype=float)
        if arr.size == 0:
            return math.nan
        w = self.window  # may be None -> full series
        tail = arr if w is None else arr[-w:]
        return float(np.mean(tail))


@register_metric()
class CVaR5:
    """5 % Conditional Value-at-Risk (expected shortfall) of net P&L.

    Computes the mean of the bottom ``alpha`` fraction of the distribution:

    .. math::

        \\text{CVaR}_{\\alpha} = \\mathbb{E}[r \\mid r \\le q_{\\alpha}(r)]

    where :math:`q_{\\alpha}` is the :math:`\\alpha`-quantile.

    ``higher_is_better = True`` because a *less negative* (higher) CVaR is
    preferable.

    Parameters
    ----------
    alpha:
        Tail probability.  Defaults to ``0.05`` (5 %).
    """

    name: ClassVar[str] = "cvar_5"
    higher_is_better: ClassVar[bool] = True
    is_linear: ClassVar[bool] = False
    is_tail_metric: ClassVar[bool] = True

    def __init__(self, alpha: float = 0.05) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha

    def compute(self, net_pnl: np.ndarray, ctx: ScoreContext) -> float:
        """Return the mean of the bottom ``alpha`` quantile of ``net_pnl``.

        Parameters
        ----------
        net_pnl:
            Raw net-P&L series (1-D float array), oldest -> newest.
        ctx:
            Score context (unused).

        Returns
        -------
        float
            CVaR value (typically negative for a loss-making tail).
            Returns ``np.nan`` when the series is too short to estimate a tail.
        """
        arr = np.asarray(net_pnl, dtype=float)
        n = arr.size
        if n == 0:
            return math.nan
        threshold = np.quantile(arr, self.alpha)
        tail = arr[arr <= threshold]
        if tail.size == 0:
            # Degenerate: all observations are identical
            return float(threshold)
        return float(np.mean(tail))


@register_metric()
class MaxDrawdown:
    """Maximum peak-to-trough drawdown of the net P&L series.

    Computed on the **date-ordered** (chronological) series — never sorted.

    .. math::

        \\text{peak}_t = \\max_{s \\le t} r_s
        \\text{MaxDrawdown} = \\max_t (\\text{peak}_t - r_t)

    ``higher_is_better = False`` because a smaller drawdown is preferred.
    ``is_linear = False`` because running-max is non-smooth.

    Unlike min_payoff (absolute worst point), max_drawdown captures the
    magnitude of the worst peak-to-trough collapse — the "drops all at once"
    ugliness visible on the chart.
    """

    name: ClassVar[str] = "max_drawdown"
    higher_is_better: ClassVar[bool] = False
    is_linear: ClassVar[bool] = False
    is_tail_metric: ClassVar[bool] = False

    def compute(self, net_pnl: np.ndarray, ctx: ScoreContext) -> float:
        """Return the maximum peak-to-trough drop in the ordered series.

        Parameters
        ----------
        net_pnl:
            Raw net-P&L series (1-D float array), **chronologically ordered**
            (oldest → newest by maturity date).  Do NOT sort.
        ctx:
            Score context (unused).

        Returns
        -------
        float
            Non-negative magnitude of the worst peak-to-trough drop.
            Returns 0.0 if the series is monotonically increasing.
            Returns ``np.nan`` for empty input.
        """
        arr = np.asarray(net_pnl, dtype=float)
        if arr.size == 0:
            return math.nan
        peak = np.maximum.accumulate(arr)
        drawdown = peak - arr
        return float(np.max(drawdown))


@register_metric()
class SharpePayoff:
    """Annualised Sharpe ratio computed on net P&L.

    .. math::

        \\text{Sharpe} = \\frac{\\mu(r)}{\\sigma(r)} \\cdot \\sqrt{F}

    where :math:`F` is the annualisation factor from :class:`ScoreContext`.

    Parameters
    ----------
    ddof:
        Delta degrees of freedom for the standard deviation.  Defaults to
        ``1`` (sample std).
    min_obs:
        Minimum number of observations required.  Returns ``np.nan`` when the
        series is shorter.  Defaults to ``2``.
    """

    name: ClassVar[str] = "sharpe_payoff"
    higher_is_better: ClassVar[bool] = True
    is_linear: ClassVar[bool] = False
    is_tail_metric: ClassVar[bool] = False

    def __init__(self, ddof: int = 1, min_obs: int = 2) -> None:
        self.ddof = ddof
        self.min_obs = min_obs

    def compute(self, net_pnl: np.ndarray, ctx: ScoreContext) -> float:
        """Return ``mean(net_pnl) / std(net_pnl) * sqrt(ann_factor)``.

        Parameters
        ----------
        net_pnl:
            Raw net-P&L series (1-D float array), oldest -> newest.
        ctx:
            Score context – ``ann_factor`` is used for annualisation.

        Returns
        -------
        float
            Annualised Sharpe ratio, or ``np.nan`` when the series is too
            short or has zero variance.
        """
        arr = np.asarray(net_pnl, dtype=float)
        if arr.size < max(self.min_obs, 2):
            return math.nan
        mu = np.mean(arr)
        sigma = np.std(arr, ddof=self.ddof)
        if sigma == 0.0:
            return math.nan
        return float(mu / sigma * math.sqrt(ctx.ann_factor))


@register_metric()
class WeightedStrike:
    """Weighted-average net strike of the basket (objective, not constraint).

    Unlike every other metric, this value is **not computable from the net-P&L
    series** — it is a function of the weights and per-stock strikes and must
    be supplied by the caller through :attr:`ScoreContext.weighted_strike`.
    Returns ``nan`` when the context does not carry it (the scoring layer
    treats a NaN raw value as worst-possible for weighted metrics and the
    reference build skips it unless per-sample values are provided).

    ``higher_is_better = False``: a lower strike is a cheaper package.
    ``is_linear = False`` here is deliberate even though dot(w, strikes) is
    linear in *w*: the linear LP fast path derives per-stock values from P&L
    columns, which this metric cannot provide.  Configs weighting it are
    routed to the SLSQP path, where the solver injects dot(w, strikes) into
    the context at every evaluation.

    The ``max_net_strike`` hard constraint is independent and remains active
    regardless of this metric's weight (objective pushes, constraint bounds).
    """

    name: ClassVar[str] = "weighted_strike"
    higher_is_better: ClassVar[bool] = False
    is_linear: ClassVar[bool] = False
    is_tail_metric: ClassVar[bool] = False

    def compute(self, net_pnl: np.ndarray, ctx: ScoreContext) -> float:
        """Return ``ctx.weighted_strike`` or ``nan`` when absent.

        Parameters
        ----------
        net_pnl:
            Unused (present for protocol compliance).
        ctx:
            Score context — must carry ``weighted_strike`` for a finite value.

        Returns
        -------
        float
        """
        ws = ctx.weighted_strike
        return float(ws) if ws is not None else math.nan


# ---------------------------------------------------------------------------
# Convenience re-export of all default metric instances
# ---------------------------------------------------------------------------

#: Pre-built instances with default parameters, keyed by metric name.
DEFAULT_METRICS: Dict[str, Metric] = {
    name: cls() for name, cls in MetricRegistry.items()
}


# ---------------------------------------------------------------------------
# Differentiable surrogate for hit_ratio (gradient-based solvers only)
# ---------------------------------------------------------------------------

def soft_hit_ratio(net_pnl: np.ndarray, tau: float | None = None) -> float:
    """Differentiable sigmoid relaxation of hit_ratio for gradient-based solvers."""
    if tau is None:
        tau = max(0.1 * float(np.std(net_pnl)), 1e-8)
    z = np.clip(net_pnl / tau, -60.0, 60.0)   # avoid exp overflow
    return float(np.mean(1.0 / (1.0 + np.exp(-z))))


__all__ = [
    # Protocol & context
    "Metric",
    "ScoreContext",
    # Registry helpers
    "MetricRegistry",
    "register_metric",
    # Concrete metrics
    "LastCarry",
    "HitRatio",
    "MinPayoff",
    "MeanPayoff",
    "CVaR5",
    "MaxDrawdown",
    "SharpePayoff",
    "WeightedStrike",
    # Pre-built defaults
    "DEFAULT_METRICS",
    # Differentiable surrogate
    "soft_hit_ratio",
]
