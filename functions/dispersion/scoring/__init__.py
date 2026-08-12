"""
functions/dispersion/scoring — Modular dispersion portfolio scoring layer.

Pipeline::

    fitness(net_pnl) = aggregate( normalize( metrics(net_pnl) ), weights )

Three decoupled layers connected by Protocols:

1. **Metrics** (metrics.py) — compute raw scalars from P&L series.
   Add a new criterion = write a class + @register_metric.  No other changes.
2. **Normalizers** (normalizers.py) — map raw values to [0,1] scale-free.
   Default: QuantileNormalizer (rank-based, ensures equal-weight = equal-influence).
3. **Aggregators** (aggregators.py) — combine normalised scores.
   Default: WeightedSum.

Plus:
- **ScoreFunction** (score.py) — orchestrator binding the 3 layers.
- **MetricWeights** (score.py) — validated immutable weight mapping.
- **WeightSolver** (weight_solver.py) — inner-level continuous optimisation of
  portfolio weights for the bilevel architecture.
"""

from .aggregators import Aggregator, ChebyshevAggregator, WeightedSum
from .metrics import (
    DEFAULT_METRICS,
    CVaR5,
    HitRatio,
    LastCarry,
    MaxDrawdown,
    MeanPayoff,
    Metric,
    MetricRegistry,
    MinPayoff,
    ScoreContext,
    SharpePayoff,
    WeightedStrike,
    register_metric,
)
from .normalizers import (
    MinMaxNormalizer,
    Normalizer,
    QuantileNormalizer,
    ZScoreNormalizer,
)
from .score import MetricWeights, ScoreFunction, make_default_score_function
from .weight_solver import WeightConstraints, WeightSolver, WeightSolverResult

__all__ = [
    # Protocols
    "Metric",
    "Normalizer",
    "Aggregator",
    # Context
    "ScoreContext",
    # Registry
    "MetricRegistry",
    "register_metric",
    "DEFAULT_METRICS",
    # Metrics
    "LastCarry",
    "HitRatio",
    "MinPayoff",
    "MeanPayoff",
    "CVaR5",
    "SharpePayoff",
    "WeightedStrike",
    # Normalizers
    "QuantileNormalizer",
    "ZScoreNormalizer",
    "MinMaxNormalizer",
    # Aggregators
    "WeightedSum",
    "ChebyshevAggregator",
    # Orchestration
    "ScoreFunction",
    "MetricWeights",
    "make_default_score_function",
    # Weight solver
    "WeightConstraints",
    "WeightSolver",
    "WeightSolverResult",
]
