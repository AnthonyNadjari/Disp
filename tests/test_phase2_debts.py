"""Phase 2 debt tests: MILP benchmark universe consistency (forced names
pinned in, non-candidate columns pinned out)."""

from __future__ import annotations

import numpy as np
import pytest

from functions.dispersion.models import (
    DispersionLeg,
    MissingDataPolicy,
    OptimizationConstraints,
)
from functions.dispersion._optimizer import DispersionOptimizer
from functions.dispersion.scoring import MetricWeights

N_DAYS = 200


def _mk_legs(names, lo=0.05, hi=0.60):
    return [
        DispersionLeg(variance_asset=n, strike_mono_var_swap=0.10 + 0.01 * i,
                      min_weight=lo, max_weight=hi)
        for i, n in enumerate(names)
    ]


def _mk_pnl(n, seed=17):
    rng = np.random.default_rng(seed)
    means = np.linspace(-0.4, 1.2, n)
    return np.column_stack([rng.normal(means[i], 0.9, N_DAYS) for i in range(n)])


def _cons():
    return OptimizationConstraints(
        min_stocks_long=2, max_stocks_long=3,
        min_stocks_short=0, max_stocks_short=0,
        max_net_strike=10.0, population_size=30,
        max_generations=60, time_limit_seconds=15.0,
        stagnation_limit=15,
    )


def _run(names, pnl, col_map, forced=None, seed=0):
    opt = DispersionOptimizer(
        long_candidates=_mk_legs(names),
        short_candidates=[],
        pnl_matrix=pnl,
        column_map=col_map,
        constraints=_cons(),
        missing_data_policy=MissingDataPolicy.FILL_ZERO,
        metric_weights=MetricWeights({"min_payoff": 1.0}),
        seed=seed,
        forced_long_indices=forced,
    )
    result = opt.run()
    assert result.long_basket
    return opt


def test_milp_forced_name_selected():
    names = [f"M{i}" for i in range(6)]
    pnl = _mk_pnl(6)
    col_map = {n: i for i, n in enumerate(names)}

    opt_free = _run(names, pnl, col_map)
    milp_free = opt_free.milp_benchmark()
    assert milp_free["status"] == "optimal"

    # Force the WORST name (lowest mean by construction: index 0)
    opt_forced = _run(names, pnl, col_map, forced=[0])
    milp_forced = opt_forced.milp_benchmark()
    assert milp_forced["status"] == "optimal"

    z = np.round(milp_forced["z"]).astype(int)
    assert z[col_map["M0"]] == 1, "forced name must be selected by the MILP certificate"
    assert milp_forced["weights"][col_map["M0"]] >= 0.05 - 1e-9, (
        "forced name must carry at least its min weight")
    # Forcing a bad name can only degrade (or match) the certified optimum
    assert milp_forced["min_payoff"] <= milp_free["min_payoff"] + 1e-9


def test_milp_never_selects_non_candidate_columns():
    """Columns present in the P&L matrix but absent from the candidate list
    (excluded / filtered names) must be unselectable by the certificate —
    mirroring how _api filters the universe while the matrix keeps all
    loaded columns."""
    names = [f"M{i}" for i in range(6)]
    pnl = _mk_pnl(6, seed=19)
    # Make the LAST column overwhelmingly attractive for a maximin objective…
    pnl[:, 5] = np.abs(pnl[:, 5]) + 2.0
    col_map = {n: i for i, n in enumerate(names)}

    # …but exclude it from the candidate universe (as _api's exclusion does)
    candidate_names = names[:5]
    opt = DispersionOptimizer(
        long_candidates=_mk_legs(candidate_names),
        short_candidates=[],
        pnl_matrix=pnl,
        column_map=col_map,          # matrix still has all 6 columns
        constraints=_cons(),
        missing_data_policy=MissingDataPolicy.FILL_ZERO,
        metric_weights=MetricWeights({"min_payoff": 1.0}),
        seed=0,
    )
    result = opt.run()
    assert "M5" not in [k for k, _ in result.long_basket]

    milp = opt.milp_benchmark()
    assert milp["status"] == "optimal"
    z = np.round(milp["z"]).astype(int)
    assert z[col_map["M5"]] == 0, (
        "MILP certificate selected a column the GA cannot pick "
        "(excluded/filtered name)")
    assert abs(milp["weights"][col_map["M5"]]) <= 1e-9
