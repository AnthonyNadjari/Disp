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
