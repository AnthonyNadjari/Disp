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


# ── reweight_grace_days: live semantics, shared by both engines ──────────────

def test_active_mask_with_grace_semantics():
    from functions.dispersion.scoring.weight_solver import active_mask_with_grace
    v = np.array([[0, 1, 1, 0, 0, 1, 0, 0, 0, 1]], dtype=bool).T.repeat(1, axis=1)
    is_valid = v.reshape(-1, 1)
    # grace=0 → identical to validity (bit-identical historical behaviour)
    assert np.array_equal(active_mask_with_grace(is_valid, 0), is_valid)
    m2 = active_mask_with_grace(is_valid, 2)[:, 0]
    # not started yet (row 0 invalid, no prior valid) → inactive
    assert not m2[0]
    # 2-day gap (rows 3,4) bridged by grace=2
    assert m2[3] and m2[4]
    # 3-day gap (rows 6,7,8): first 2 in grace, 3rd beyond → inactive
    assert m2[6] and m2[7] and not m2[8]
    # trading day after the gap → active again
    assert m2[9]


def test_grace_holds_weight_backtester_and_optimizer_agree():
    """A 2-day gap with grace=3: the gapped name KEEPS its weight (basket
    under-invested those days) instead of redistributing. Backtester policy
    and optimizer adaptive P&L must produce the SAME curve."""
    from functions.dispersion._optimizer import DispersionOptimizer
    from functions.dispersion.scoring import MetricWeights
    from functions.dispersion.models import OptimizationConstraints

    n_days = 120
    pnl = np.column_stack([np.full(n_days, 1.0), np.full(n_days, 0.5)])
    pnl[60:62, 1] = np.nan                    # 2-day gap on name B
    names = ["A", "B"]
    col_map = {t: i for i, t in enumerate(names)}
    legs = [DispersionLeg(variance_asset=t, strike_mono_var_swap=0.10,
                          min_weight=0.05, max_weight=0.95) for t in names]
    w = {"A": 0.5, "B": 0.5}

    for grace, gap_expect in [(0, 1.0), (3, 0.5)]:
        # grace=0: B's weight redistributes to A on gap days → net = 1.0
        # grace=3: B keeps its slot, contributes 0 → net = 0.5·1 + 0.5·0 = 0.5
        cfg = _to_swap_config(DispersionConfig(
            n_exp=20, missing_data_policy=MissingDataPolicy.ADAPTIVE_REWEIGHT,
            reweight_grace_days=grace))
        bt = DispersionBacktester(cfg)
        long_pnl, short_pnl, active, _ = bt._apply_adaptive_policy(pnl, names, w)
        net_bt = long_pnl + short_pnl
        assert net_bt[59] == pytest.approx(0.75)          # normal day
        assert net_bt[60] == pytest.approx(gap_expect), f"grace={grace}"
        assert net_bt[62] == pytest.approx(0.75)          # back to normal

        cons = OptimizationConstraints(
            min_stocks_long=2, max_stocks_long=2, min_stocks_short=0,
            max_stocks_short=0, max_net_strike=10.0, population_size=10,
            max_generations=5, time_limit_seconds=5.0, stagnation_limit=5)
        opt = DispersionOptimizer(
            long_candidates=legs, short_candidates=[], pnl_matrix=pnl,
            column_map=col_map, constraints=cons,
            missing_data_policy=MissingDataPolicy.ADAPTIVE_REWEIGHT,
            reweight_grace_days=grace,
            metric_weights=MetricWeights({"mean_payoff": 1.0}), seed=0)
        net_opt = opt._adaptive_net_pnl([0, 1], np.array([0.5, 0.5]))
        assert np.allclose(net_opt, net_bt), (
            f"grace={grace}: optimizer and backtester adaptive curves diverge")


def test_bundle_v1_replays_with_grace_zero(tmp_path):
    """Old (v1) bundles stored grace=3 while it was a NO-OP — replay must force
    0 so their results reproduce. New (v2) bundles honour the stored value."""
    import json
    from functions.dispersion.run_bundle import load_run_bundle, save_run_bundle
    from functions.dispersion.models import OptimizationConstraints

    rng = np.random.default_rng(3)
    pnl = rng.normal(0.5, 1.0, (150, 4))
    names = [f"G{i}" for i in range(4)]
    legs = [DispersionLeg(variance_asset=t, strike_mono_var_swap=0.10,
                          min_weight=0.05, max_weight=0.95) for t in names]
    cons = OptimizationConstraints(
        min_stocks_long=2, max_stocks_long=3, min_stocks_short=0,
        max_stocks_short=0, max_net_strike=10.0, population_size=16,
        max_generations=8, time_limit_seconds=5.0, stagnation_limit=5)
    path = str(tmp_path / "b")
    save_run_bundle(
        path, pnl_matrix=pnl, column_map={t: i for i, t in enumerate(names)},
        long_candidates=legs, short_candidates=[], constraints=cons,
        score_weights={"mean_payoff": 1.0}, seed=0,
        missing_data_policy=MissingDataPolicy.ADAPTIVE_REWEIGHT,
        reweight_grace_days=4)
    # v2 bundle: stored grace honoured
    b2 = load_run_bundle(path)
    assert b2.reweight_grace_days == 4
    # simulate a v1 bundle: rewrite version + stored (inert) grace
    with open(f"{path}/bundle.json") as f:
        payload = json.load(f)
    payload["bundle_version"] = 1
    payload["optimizer"]["reweight_grace_days"] = 3
    with open(f"{path}/bundle.json", "w") as f:
        json.dump(payload, f)
    b1 = load_run_bundle(path)
    assert b1.reweight_grace_days == 0, "v1 bundles must replay with grace=0 (it was inert)"


# ── Commit-4 robustness: strike validation, loud Bloomberg, no inf ───────────

def test_blank_mono_strike_raises_at_parse():
    from functions.dispersion._api import _df_to_legs
    df = pd.DataFrame({
        "Variance Asset": ["OK Equity", "BAD Equity"],
        "Strike Mono Var Swap (%)": [21.4, np.nan],
        "Min Weight": [1.0, 1.0], "Max Weight": [60.0, 60.0],
    })
    with pytest.raises(ValueError, match="BAD Equity.*Strike Mono Var Swap"):
        _df_to_legs(df, is_cross_corridor=False)


def test_duplicate_corridor_key_raises():
    from functions.dispersion._backtester import compute_leg_pnl_columns
    dates = pd.bdate_range("2020-01-02", periods=100)
    variance_px = pd.DataFrame({"IDX Index": _series(100, seed=8)}, index=dates)
    corridor_px = pd.DataFrame({"DUP Equity": _series(100, seed=9)}, index=dates)
    legs = [DispersionLeg(variance_asset="IDX Index", corridor_condition_asset="DUP Equity",
                          strike_mono_var_swap=0.10, strike_cross_corridor=0.10)
            for _ in range(2)]
    cfg = _to_swap_config(DispersionConfig(cross_corridor=True, n_exp=20))
    with pytest.raises(ValueError, match="Duplicate candidate key 'DUP Equity'"):
        compute_leg_pnl_columns(variance_px, corridor_px, legs, cfg)


def test_zero_price_yields_nan_not_inf():
    prices = _series(120, seed=10)
    prices[100] = 0.0                       # pathological zero close
    out = _rolling_pnl_volswap(prices, 0.10, 20, 2.5)
    assert not np.isinf(out[~np.isnan(out)]).any(), "zero price must yield NaN, never inf"


# ── Review fix: grace mask must carry PRE-WINDOW history across the slice ────

def test_optimizer_honors_external_active_mask():
    """A gap already open at window start (name printed before the window)
    must stay in-grace: _api builds the mask on FULL history and passes it in;
    the optimizer must use it verbatim instead of re-deriving from the sliced
    window (where the name looks 'never started')."""
    from functions.dispersion._optimizer import DispersionOptimizer
    from functions.dispersion.scoring import MetricWeights
    from functions.dispersion.models import OptimizationConstraints

    n_days = 60
    pnl = np.column_stack([np.full(n_days, 1.0), np.full(n_days, 0.5)])
    pnl[:2, 1] = np.nan            # B gapped on window rows 0-1 (gap began pre-window)
    names = ["A", "B"]
    legs = [DispersionLeg(variance_asset=t, strike_mono_var_swap=0.10,
                          min_weight=0.05, max_weight=0.95) for t in names]
    cons = OptimizationConstraints(
        min_stocks_long=2, max_stocks_long=2, min_stocks_short=0,
        max_stocks_short=0, max_net_strike=10.0, population_size=10,
        max_generations=5, time_limit_seconds=5.0, stagnation_limit=5)

    def make(mask):
        return DispersionOptimizer(
            long_candidates=legs, short_candidates=[], pnl_matrix=pnl,
            column_map={t: i for i, t in enumerate(names)}, constraints=cons,
            missing_data_policy=MissingDataPolicy.ADAPTIVE_REWEIGHT,
            reweight_grace_days=2, active_mask=mask,
            metric_weights=MetricWeights({"mean_payoff": 1.0}), seed=0)

    w = np.array([0.5, 0.5])
    # window-derived (fallback): B 'never started' on rows 0-1 → redistribute → 1.0
    net_fallback = make(None)._adaptive_net_pnl([0, 1], w)
    assert net_fallback[0] == pytest.approx(1.0)
    # full-history mask (as _api now builds): B in-grace → holds weight → 0.5
    full_mask = np.ones((n_days, 2), dtype=bool)     # pre-window print ⇒ rows 0-1 in-grace
    net_ext = make(full_mask)._adaptive_net_pnl([0, 1], w)
    assert net_ext[0] == pytest.approx(0.5)
    assert net_ext[2] == pytest.approx(0.75)


def test_bundle_stores_and_replays_active_mask(tmp_path):
    from functions.dispersion.run_bundle import load_run_bundle, save_run_bundle
    from functions.dispersion.models import OptimizationConstraints
    import os

    rng = np.random.default_rng(6)
    pnl = rng.normal(0.5, 1.0, (120, 3))
    pnl[:2, 0] = np.nan
    names = ["M0", "M1", "M2"]
    mask = np.ones((120, 3), dtype=bool)             # full-history semantics
    legs = [DispersionLeg(variance_asset=t, strike_mono_var_swap=0.10,
                          min_weight=0.05, max_weight=0.95) for t in names]
    cons = OptimizationConstraints(
        min_stocks_long=2, max_stocks_long=3, min_stocks_short=0,
        max_stocks_short=0, max_net_strike=10.0, population_size=12,
        max_generations=6, time_limit_seconds=5.0, stagnation_limit=5)
    path = str(tmp_path / "bm")
    save_run_bundle(
        path, pnl_matrix=pnl, column_map={t: i for i, t in enumerate(names)},
        long_candidates=legs, short_candidates=[], constraints=cons,
        score_weights={"mean_payoff": 1.0}, seed=0,
        missing_data_policy=MissingDataPolicy.ADAPTIVE_REWEIGHT,
        reweight_grace_days=2, active_mask=mask)
    assert os.path.exists(f"{path}/active_mask.parquet")
    b = load_run_bundle(path)
    assert b.active_mask is not None and b.active_mask.shape == (120, 3)
    assert bool(b.active_mask[0, 0]), "carried-in grace row must survive the roundtrip"
