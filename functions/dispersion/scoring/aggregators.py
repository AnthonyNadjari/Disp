"""aggregators.py — combine normalised metric scores into one fitness value.

═══════════════════════════════════════════════════════════════════════════════
SCAFFOLDING REPO DE TEST — NE PAS recopier dans le vrai repo.
Le vrai repo possède son propre aggregators.py ; ce fichier est une
reconstruction inférée de l'usage observé dans score.py
(``aggregate(normalized: Dict[str, float], weights: Dict[str, float])``).
Si le comportement du vrai aggregator diffère, c'est LUI la référence.
═══════════════════════════════════════════════════════════════════════════════

Contract (used by :class:`~functions.dispersion.scoring.score.ScoreFunction`):

- ``aggregate(normalized, weights)`` receives the full normalised dict
  (every metric present, values in [0, 1]) and the full weight dict
  (already normalised to sum ≈ 1 by ``MetricWeights``); returns a scalar
  in approximately [0, 1].
"""

from __future__ import annotations

from typing import Dict, Protocol, runtime_checkable

__all__ = [
    "Aggregator",
    "WeightedSum",
    "ChebyshevAggregator",
]


@runtime_checkable
class Aggregator(Protocol):
    """Structural protocol for aggregators."""

    def aggregate(self, normalized: Dict[str, float], weights: Dict[str, float]) -> float: ...


class WeightedSum:
    """Weighted arithmetic mean of normalised scores (the default).

    ``score = Σ_i w_i · s_i`` — with weights summing to 1 and scores in
    [0, 1], the aggregate stays in [0, 1].  Metrics absent from
    ``normalized`` contribute 0 (worst) rather than crashing.
    """

    def aggregate(self, normalized: Dict[str, float], weights: Dict[str, float]) -> float:
        return float(sum(w * normalized.get(name, 0.0) for name, w in weights.items() if w > 0))


class ChebyshevAggregator:
    """Weighted-Chebyshev (worst-case) scalarisation.

    ``score = 1 − max_i w_i · (1 − s_i)`` over active metrics (w_i > 0):
    the aggregate is driven by the single worst weighted shortfall from the
    ideal point (all scores = 1).  Reduces to ``s`` for a single active
    metric at weight 1.  Useful to demand balance across criteria instead
    of allowing one strong metric to compensate a weak one.
    """

    def aggregate(self, normalized: Dict[str, float], weights: Dict[str, float]) -> float:
        shortfalls = [
            w * (1.0 - normalized.get(name, 0.0))
            for name, w in weights.items()
            if w > 0
        ]
        if not shortfalls:
            return 0.0
        return float(1.0 - max(shortfalls))
