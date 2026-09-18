"""A requested short leg is never silently dropped.

Regression: ``_long_only`` used to be inferred from ``len(short_candidates) == 0``,
so a short candidate that failed to load (or was excluded / filtered out) flipped
the whole run to long-only and ``min_stocks_short`` was silently ignored — a run
configured with min_stocks_short = max_stocks_short = 1 came back long-only.
Long-only is now an INTENT (``max_stocks_short == 0``); anything else raises.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions.dispersion._optimizer import DispersionOptimizer  # noqa: E402
from functions.dispersion.models import (  # noqa: E402
    DispersionLeg, MissingDataPolicy, OptimizationConstraints,
)
from functions.dispersion.scoring import MetricWeights  # noqa: E402


def _universe(n=8, rows=200, seed=0):
    rng = np.random.default_rng(seed)
    names = [f"S{i}" for i in range(n)]
    pnl = rng.normal(0.02, 0.5, size=(rows, n))
    legs = [DispersionLeg(variance_asset=t, strike_mono_var_swap=0.20,
                          min_weight=0.0, max_weight=1.0) for t in names]
    return legs, pnl, {t: i for i, t in enumerate(names)}


def _opt(long_legs, short_legs, pnl, col_map, **cons_kw):
    cons = OptimizationConstraints(
        min_stocks_long=2, max_stocks_long=3,
        max_net_strike=10.0, population_size=16, max_generations=6,
        time_limit_seconds=8.0, stagnation_limit=4, **cons_kw)
    return DispersionOptimizer(
        long_candidates=long_legs, short_candidates=short_legs,
        pnl_matrix=pnl, column_map=col_map, constraints=cons,
        missing_data_policy=MissingDataPolicy.FILL_ZERO,
        metric_weights=MetricWeights({"mean_payoff": 1.0}), seed=1,
    )


def test_requested_short_leg_with_no_candidates_raises():
    """min_stocks_short = max_stocks_short = 1 and zero usable short candidates
    → loud error, NOT a silent long-only basket (the reported bug)."""
    legs, pnl, col_map = _universe()
    opt = _opt(legs[:5], [], pnl, col_map, min_stocks_short=1, max_stocks_short=1)
    assert not opt._long_only                       # intent, not data
    with pytest.raises(ValueError, match="Short leg requested but not feasible"):
        opt.run()


def test_fewer_short_candidates_than_min_raises():
    legs, pnl, col_map = _universe()
    opt = _opt(legs[:5], legs[5:6], pnl, col_map, min_stocks_short=2, max_stocks_short=3)
    with pytest.raises(ValueError, match="min_stocks_short=2"):
        opt.run()


def test_optional_shorts_with_no_candidates_runs_long_only():
    """min_stocks_short = 0 (shorts optional) and no candidates: nothing is
    violated, so the run proceeds long-only — and must not divide by zero when
    drawing the reference sample."""
    legs, pnl, col_map = _universe()
    opt = _opt(legs, [], pnl, col_map, min_stocks_short=0, max_stocks_short=5)
    assert opt._long_only
    result = opt.run()
    assert result.long_basket and not result.short_basket


def test_explicit_long_only_still_runs():
    """max_stocks_short = 0 is the long-only intent — unchanged behaviour."""
    legs, pnl, col_map = _universe()
    opt = _opt(legs, [], pnl, col_map, min_stocks_short=0, max_stocks_short=0)
    assert opt._long_only
    result = opt.run()
    assert result.long_basket and not result.short_basket


@pytest.mark.parametrize("n_short", [1, 2])
def test_short_leg_is_delivered_when_feasible(n_short):
    """min = max = n_short → the delivered winner carries exactly n_short shorts."""
    legs, pnl, col_map = _universe()
    opt = _opt(legs[:5], legs[5:], pnl, col_map,
               min_stocks_short=n_short, max_stocks_short=n_short)
    result = opt.run()
    assert len(result.long_basket) >= 2
    assert len(result.short_basket) == n_short, result.short_basket


def test_api_short_requested_but_all_dropped_raises(monkeypatch):
    """_prepare_optimization_inputs raises when shorts were provided but none
    survive the universe filter (price load failure / exclusion) — instead of
    handing the optimizer an empty short list."""
    import pandas as pd
    from functions.dispersion import _api, _backtester as bt
    from functions.dispersion.models import DispersionConfig

    tickers = ["S0", "S1", "S2"]              # the short name "GONE" is never priced
    idx = pd.bdate_range("2025-01-01", periods=200)
    px = pd.DataFrame(100.0, index=idx, columns=tickers)
    pnl = np.random.default_rng(0).normal(0.02, 0.5, size=(len(idx), len(tickers)))
    col_map = {t: i for i, t in enumerate(tickers)}

    def fake_load(self, basket, **kwargs):
        return {
            "variance_px": px, "corridor_px": None,
            "legs": list(basket.long_candidates) + list(basket.short_candidates),
            "long_legs": list(basket.long_candidates),
            "short_legs": list(basket.short_candidates),
        }

    monkeypatch.setattr(bt.DispersionDataLoader, "load", fake_load)
    monkeypatch.setattr(_api, "_build_pnl_matrix", lambda *a, **k: (pnl.copy(), dict(col_map)))

    def _row(t):
        return {"Variance Asset": t, "Strike Mono Var Swap (%)": 20.0,
                "Min Weight": 0.0, "Max Weight": 100.0}

    with pytest.raises(ValueError, match="Infeasible short constraints"):
        _api._prepare_optimization_inputs(
            long_df=pd.DataFrame([_row(t) for t in tickers]),
            config=DispersionConfig(cross_corridor=False),
            constraints=OptimizationConstraints(
                min_stocks_long=2, max_stocks_long=3,
                min_stocks_short=1, max_stocks_short=1),
            short_df=pd.DataFrame([_row("GONE")]),   # never priced → dropped
            start_date=None, end_date=None, filter_zero_hr=False,
            forced_set=set(), excluded_set=set(),
        )
