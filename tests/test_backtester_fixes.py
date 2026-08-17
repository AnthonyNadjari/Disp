"""Regression tests for the post-audit backtester fixes.

Covers three confirmed defects:
  Fix 1 — cross-corridor silent fallback: a leg whose corridor stock price is
          missing must be DROPPED (not turned into a plain index variance swap).
  Fix 2 — end_date look-ahead: the backtest curve must be bounded to end_date.
  Fix 3 — stale P&L on no-trade rows: a NaN (no-trade) day must emit NaN, not a
          stale value carried from the last valid window.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from functions.dispersion._api import _build_pnl_matrix, _to_swap_config
from functions.dispersion._backtester import (
    DispersionBacktester,
    _rolling_pnl_corridor,
    _rolling_pnl_volswap,
)
from functions.dispersion.models import (
    DispersionConfig,
    DispersionLeg,
    MissingDataPolicy,
)


def _series(n=400, seed=0, start=100.0, vol=0.012):
    rng = np.random.default_rng(seed)
    return start * np.cumprod(1.0 + rng.normal(0.0, vol, n))


# ── Fix 3: kernels must emit NaN on a no-trade (NaN) row ─────────────────────

def test_volswap_kernel_nan_row_emits_nan():
    n_exp = 20
    prices = _series(200, seed=1)
    gap = 150                      # well past the n_exp warm-up
    prices[gap] = np.nan
    out = _rolling_pnl_volswap(prices, 0.10, n_exp, 2.5)
    assert np.isnan(out[gap]), "no-trade day must emit NaN, not stale P&L"
    assert not np.isnan(out[gap - 1]), "a valid post-warm-up day should have P&L"
    assert not np.isnan(out[gap + 1]), "the next valid day resumes normally"


def test_corridor_kernel_nan_row_emits_nan():
    n_exp = 20
    var = _series(200, seed=2)
    corr = _series(200, seed=3)
    gap = 150
    corr[gap] = np.nan             # corridor series missing on this day
    out = _rolling_pnl_corridor(var, corr, 0.10, 1.30, 0.70, n_exp, 2.5)
    assert np.isnan(out[gap]), "no-trade day (either series NaN) must emit NaN"
    assert not np.isnan(out[gap - 1])


# ── Fix 1: cross-corridor drops a leg with a missing corridor price ──────────

def test_cross_corridor_drops_missing_corridor_leg():
    dates = pd.bdate_range("2020-01-02", periods=400)
    price_data = pd.DataFrame({"IDX Index": _series(400, seed=4)}, index=dates)
    # Only AAA's corridor stock price is available; BBB's is missing entirely.
    index_data = pd.DataFrame({"AAA Equity": _series(400, seed=5)}, index=dates)
    legs = [
        DispersionLeg(variance_asset="IDX Index", corridor_condition_asset="AAA Equity",
                      strike_mono_var_swap=0.10, strike_cross_corridor=0.10),
        DispersionLeg(variance_asset="IDX Index", corridor_condition_asset="BBB Equity",
                      strike_mono_var_swap=0.10, strike_cross_corridor=0.10),
    ]
    cfg = DispersionConfig(cross_corridor=True, n_exp=60)
    with pytest.warns(UserWarning, match="dropped"):
        pnl, col_map = _build_pnl_matrix(price_data, index_data, legs, cfg)
    assert "AAA Equity" in col_map
    assert "BBB Equity" not in col_map, (
        "leg with a missing corridor price must be dropped, not fall back to an index swap")
    assert pnl.shape[1] == 1


# ── Fix 2: the backtest curve is bounded to end_date ─────────────────────────

def test_backtest_honors_end_date():
    dates = pd.bdate_range("2020-01-02", periods=500)
    price_data = pd.DataFrame({"IDX Index": _series(500, seed=6)}, index=dates)
    legs = [DispersionLeg(variance_asset="IDX Index", strike_mono_var_swap=0.10)]
    weights = {"IDX Index": 1.0}
    cfg = _to_swap_config(DispersionConfig(n_exp=20, missing_data_policy=MissingDataPolicy.FILL_ZERO))
    bt = DispersionBacktester(cfg)
    end = date(2021, 6, 30)

    res_capped = bt.run(price_data, legs, weights, start_date=date(2020, 6, 1), end_date=end)
    assert len(res_capped.timeseries) > 0
    assert res_capped.timeseries.index.max() <= pd.Timestamp(end), "end_date must bound the curve"

    res_open = bt.run(price_data, legs, weights, start_date=date(2020, 6, 1))
    assert res_open.timeseries.index.max() > pd.Timestamp(end), (
        "sanity: without end_date the curve extends past the cutoff")


# ── Short-leg sign: optimizer and backtester must agree in EVERY mode ────────

def test_cross_corridor_short_sign_optimizer_matches_backtester():
    """A short basket row = a SOLD leg = subtracted, in cross-corridor mode too.
    The optimizer's net P&L for a fixed (long, short) allocation must equal the
    backtester's Result curve on the same matrix (FILL_ZERO → no masking noise)."""
    from functions.dispersion._optimizer import DispersionOptimizer
    from functions.dispersion.scoring import MetricWeights
    from functions.dispersion.models import OptimizationConstraints

    rng = np.random.default_rng(11)
    n_days, names = 300, ["S1 Equity", "S2 Equity", "S3 Equity", "S4 Equity"]
    pnl = np.column_stack([rng.normal(0.3 * (i + 1), 1.0, n_days) for i in range(4)])
    col_map = {t: i for i, t in enumerate(names)}
    legs = [DispersionLeg(variance_asset="IDX Index", corridor_condition_asset=t,
                          strike_mono_var_swap=0.10, strike_cross_corridor=0.10,
                          min_weight=0.05, max_weight=0.95) for t in names]
    long_legs, short_legs = legs[:2], legs[2:]
    cons = OptimizationConstraints(
        min_stocks_long=2, max_stocks_long=2, min_stocks_short=2, max_stocks_short=2,
        max_net_strike=10.0, population_size=20, max_generations=10,
        time_limit_seconds=5.0, stagnation_limit=5)

    opt = DispersionOptimizer(
        long_candidates=long_legs, short_candidates=short_legs,
        pnl_matrix=pnl, column_map=col_map, constraints=cons,
        missing_data_policy=MissingDataPolicy.FILL_ZERO,
        metric_weights=MetricWeights({"mean_payoff": 1.0}),
        is_cross_corridor=True, seed=0)

    long_pos, long_w = [0, 1], np.array([0.5, 0.5])
    short_pos, short_w = [2, 3], np.array([0.5, 0.5])
    net_opt = opt._adaptive_net_pnl(long_pos, long_w, short_pos, short_w)

    # Backtester on the same matrix: monkey-free — feed the P&L matrix through
    # the policy layer directly via run()'s internals is heavier; instead
    # compute the reference by hand from the SAME convention: net = L − S.
    expected = pnl[:, [0, 1]] @ long_w - pnl[:, [2, 3]] @ short_w
    assert np.allclose(net_opt, expected), (
        "cross-corridor short leg must be SUBTRACTED (sold leg), matching the "
        "backtester's long_pnl + (negated) short_pnl convention")


def test_backtester_short_sign_reference():
    """Ground truth for the convention: the backtester subtracts a short row."""
    dates = pd.bdate_range("2021-01-04", periods=200)
    up = np.full(200, 1.0)      # leg that always pays +1
    dn = np.full(200, 0.25)     # leg that always pays +0.25
    pnl = np.column_stack([up, dn])

    cfg = _to_swap_config(DispersionConfig(n_exp=20, missing_data_policy=MissingDataPolicy.FILL_ZERO))
    bt = DispersionBacktester(cfg)
    legs = [DispersionLeg(variance_asset="A", strike_mono_var_swap=0.1),
            DispersionLeg(variance_asset="B", strike_mono_var_swap=0.1)]
    long_pnl, short_pnl, _, _ = bt._apply_fill_zero_policy(pnl, ["A", "B"], {"A": 1.0, "B": -1.0})
    net = long_pnl + short_pnl
    assert np.allclose(net, up - dn), "backtester net must be long − short"
