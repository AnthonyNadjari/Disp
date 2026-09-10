"""Warm start: a previous winner seeds the GA's initial population
(re-optimization speedup). Offline, synthetic universe."""

from __future__ import annotations

import numpy as np

from functions.dispersion.models import (
    DispersionLeg,
    MissingDataPolicy,
    OptimizationConstraints,
)
from functions.dispersion._optimizer import DispersionOptimizer
from functions.dispersion.scoring import MetricWeights

N_DAYS = 250
N = 8
LO, HI = 0.05, 0.60


def _universe(seed=31):
    rng = np.random.default_rng(seed)
    names = [f"B{i}" for i in range(N)]
    means = np.linspace(-0.2, 1.6, N)
    pnl = np.column_stack([rng.normal(means[i], 0.9, N_DAYS) for i in range(N)])
    legs = [
        DispersionLeg(variance_asset=names[i], strike_mono_var_swap=0.10 + 0.01 * i,
                      min_weight=LO, max_weight=HI)
        for i in range(N)
    ]
    col_map = {names[i]: i for i in range(N)}
    return legs, pnl, col_map


def _cons():
    return OptimizationConstraints(
        min_stocks_long=2, max_stocks_long=3,
        min_stocks_short=0, max_stocks_short=0,
        max_net_strike=10.0, population_size=40,
        max_generations=120, time_limit_seconds=20.0,
        stagnation_limit=20,
    )


def _mk(warm=None, weights=None):
    legs, pnl, col_map = _universe()
    return DispersionOptimizer(
        long_candidates=legs, short_candidates=[],
        pnl_matrix=pnl, column_map=col_map,
        constraints=_cons(),
        missing_data_policy=MissingDataPolicy.FILL_ZERO,
        metric_weights=MetricWeights(weights or {"mean_payoff": 0.5, "hit_ratio": 0.5}),
        seed=0, warm_start=warm)


def test_warm_start_seeds_and_run_delivers():
    r1 = _mk().run()
    assert r1.long_basket
    warm = ([k for k, _ in r1.long_basket], [])

    opt2 = _mk(warm=warm)
    r2 = opt2.run()
    assert r2.long_basket

    # After run() the scorer is fitted: the warm-start individual maps the
    # winner's names back to candidate indices and scores > 0.
    inds = opt2._warm_start_individuals()
    assert len(inds) == 1
    assert inds[0].fitness > 0.0
    assert len(inds[0].long_indices) == len(r1.long_basket)


def test_warm_start_unknown_names_ignored():
    opt = _mk(warm=(["NOPE", "ALSO_NOPE"], []))
    assert opt._warm_start_individuals() == []


def test_cold_start_unchanged_and_deterministic():
    """Without warm_start nothing changes: same seed → same basket."""
    r1 = _mk().run()
    r2 = _mk().run()
    assert [k for k, _ in r1.long_basket] == [k for k, _ in r2.long_basket]
    assert abs(r1.score - r2.score) < 1e-12
