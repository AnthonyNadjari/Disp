"""
Aggregator protocol and implementations for combining normalized criterion scores
into a single scalar score.

Each normalizer guarantees that every value in *normalized* lies in [0, 1] with
the orientation **1 = best**.  The aggregators below therefore inherit the same
orientation: a higher aggregate value is always better.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Aggregator(Protocol):
    """Protocol satisfied by any stateless aggregation strategy.

    An aggregator takes a dict of already-normalised criterion scores and their
    associated weights and collapses them into a single ``float`` in [0, 1].

    Parameters
    ----------
    normalized:
        Mapping ``{criterion_name: score}`` where every score ∈ [0, 1] and
        1 means *best*.
    weights:
        Mapping ``{criterion_name: weight}`` with non-negative weights.  Only
        criteria whose weight is **strictly positive** are considered; zero-
        weighted criteria are silently ignored so that callers may include
        disabled criteria in the dict without changing results.

    Returns
    -------
    float
        Aggregate score.  Semantics are implementation-defined but all built-in
        implementations preserve the 1 = best orientation.
    """

    def aggregate(
        self,
        normalized: dict[str, float],
        weights: dict[str, float],
    ) -> float:
        ...


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------

class WeightedSum:
    """Default aggregator: weighted linear combination of normalised scores.

    The aggregate is::

        score = Σ  weights[n] * normalized[n]   for n where weights[n] > 0

    Because each ``normalized[n] ∈ [0, 1]`` with orientation 1 = best, the
    weighted sum is maximised when **every** criterion is at its best value,
    giving exact corner extremality: a candidate that is optimal on all active
    criteria achieves the theoretical maximum (``Σ weights[n]``), and one that
    is worst on all criteria scores 0.

    The result is **not** re-normalised to [0, 1] by default so that callers
    can interpret the raw magnitude (e.g. to compare against a budget expressed
    in the same weight units).  If a [0, 1] output is required, divide by the
    sum of active weights.

    This class is stateless; the same instance may be used concurrently from
    multiple threads.
    """

    def aggregate(
        self,
        normalized: dict[str, float],
        weights: dict[str, float],
    ) -> float:
        """Return the weighted sum of active normalised scores.

        Parameters
        ----------
        normalized:
            ``{criterion_name: score}`` with scores ∈ [0, 1], 1 = best.
        weights:
            ``{criterion_name: weight}``.  Criteria with ``weight <= 0`` are
            skipped.

        Returns
        -------
        float
            ``Σ weights[n] * normalized[n]`` over all ``n`` with
            ``weights[n] > 0``.  Returns ``0.0`` if no criterion is active.

        Examples
        --------
        >>> agg = WeightedSum()
        >>> agg.aggregate({"a": 1.0, "b": 0.5}, {"a": 2.0, "b": 1.0})
        2.5
        >>> agg.aggregate({"a": 0.8}, {"a": 0.0})  # weight=0 → ignored
        0.0
        """
        return sum(
            weights[n] * normalized[n]
            for n in weights
            if weights[n] > 0
        )


class ChebyshevAggregator:
    """Optional aggregator: maximises the worst weighted criterion (minimax).

    The aggregate is::

        score = min( weights[n] * normalized[n] )   for n where weights[n] > 0

    This is the **Chebyshev** (L∞) scalarisation of the multi-criteria problem.
    Selecting the candidate with the **highest** Chebyshev aggregate maximises
    the worst-case weighted performance — useful when balance across criteria
    matters more than excelling on a few.

    Like :class:`WeightedSum`, this class is stateless and thread-safe.

    Notes
    -----
    * Because ``normalized[n] ∈ [0, 1]`` and ``weights[n] > 0``, the aggregate
      is also non-negative.
    * A criterion with a very high weight will dominate (pull the minimum down)
      unless its normalised score is correspondingly high, which is the intended
      behaviour: high weight ↔ "must perform well here".
    """

    def aggregate(
        self,
        normalized: dict[str, float],
        weights: dict[str, float],
    ) -> float:
        """Return the minimum weighted normalised score across active criteria.

        Parameters
        ----------
        normalized:
            ``{criterion_name: score}`` with scores ∈ [0, 1], 1 = best.
        weights:
            ``{criterion_name: weight}``.  Criteria with ``weight <= 0`` are
            skipped.

        Returns
        -------
        float
            ``min( weights[n] * normalized[n] )`` over all ``n`` with
            ``weights[n] > 0``.  Returns ``0.0`` if no criterion is active.

        Examples
        --------
        >>> agg = ChebyshevAggregator()
        >>> agg.aggregate({"a": 1.0, "b": 0.5}, {"a": 2.0, "b": 1.0})
        0.5
        >>> agg.aggregate({"a": 0.8}, {"a": 0.0})  # weight=0 → ignored
        0.0
        """
        active = [
            weights[n] * normalized[n]
            for n in weights
            if weights[n] > 0
        ]
        return min(active) if active else 0.0
