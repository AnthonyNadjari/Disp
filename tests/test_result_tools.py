"""Per-stock breakdown/stats, clean export, and the opt-in prep cache."""

from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd
import pytest

from functions.dispersion._api import _to_swap_config
from functions.dispersion._backtester import DispersionBacktester
from functions.dispersion.models import (
    BacktestResult,
    DispersionConfig,
    DispersionLeg,
    MissingDataPolicy,
    frames_to_zip_bytes,
)

N = 300


def _xc_result(seed=0, n_names=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=N)
    variance_px = pd.DataFrame(
        {"IDX Index": 100 * np.cumprod(1 + rng.normal(0, 0.008, N))}, index=dates)
    corridor_px = pd.DataFrame(
        {f"S{i} Equity": 100 * np.cumprod(1 + rng.normal(0, 0.02, N))
         for i in range(n_names)}, index=dates)
    legs = [DispersionLeg(variance_asset="IDX Index", corridor_condition_asset=c,
                          strike_mono_var_swap=0.20, strike_cross_corridor=0.12)
            for c in corridor_px.columns]
    cfg = _to_swap_config(DispersionConfig(cross_corridor=True, n_exp=40,
                                           missing_data_policy=MissingDataPolicy.FILL_ZERO))
    bt = DispersionBacktester(cfg)
    res = bt.run(variance_px, legs, {c: 1.0 / n_names for c in corridor_px.columns},
                 corridor_px=corridor_px, start_date=dates[0].date())
    return res


def test_per_stock_three_lines_and_identity():
    res = _xc_result()
    df = res.per_stock("S0 Equity")
    assert list(df.columns) == ["net", "mono_leg", "cross_leg"]
    m = df.dropna()
    np.testing.assert_allclose(m["net"], m["mono_leg"] - m["cross_leg"], rtol=1e-10)


def test_per_stock_stats_shape_and_values():
    res = _xc_result()
    st = res.per_stock_stats()
    assert set(st["line"]) == {"net", "mono_leg", "cross_leg"}
    assert len(st) == 3 * 3                      # 3 stocks × 3 lines
    assert ((st["hit_ratio"] >= 0) & (st["hit_ratio"] <= 100)).all()
    assert (st["max_drawdown"] <= 1e-9).all()
    one = res.per_stock_stats("S1 Equity")
    assert len(one) == 3 and set(one["stock"]) == {"S1 Equity"}


def test_per_stock_unknown_name_raises():
    res = _xc_result()
    with pytest.raises(KeyError, match="Available sample"):
        res.per_stock("GHOST Equity")


def test_export_zip_roundtrip(tmp_path):
    res = _xc_result()
    out = res.export(str(tmp_path / "bt.zip"))
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
        assert {"curve.csv", "per_leg_net.csv", "mono_cross_legs.csv",
                "per_stock_stats.csv", "summary.csv"} <= names
        with z.open("curve.csv") as f:
            curve = pd.read_csv(f, index_col=0)
    assert len(curve) == len(res.timeseries)
    assert "Result" in curve.columns
    with pytest.raises(ValueError, match="xlsx or .zip"):
        res.export(str(tmp_path / "bt.txt"))


def test_export_xlsx_needs_engine_or_works(tmp_path):
    res = _xc_result()
    try:
        import openpyxl  # noqa: F401
        have = True
    except ImportError:
        have = False
    if have:
        out = res.export(str(tmp_path / "bt.xlsx"))
        assert pd.read_excel(out, sheet_name="curve").shape[0] == len(res.timeseries)
    else:
        with pytest.raises(RuntimeError, match="openpyxl"):
            res.export(str(tmp_path / "bt.xlsx"))


def test_frames_to_zip_bytes_is_a_valid_archive():
    import io
    data = frames_to_zip_bytes({"a": pd.DataFrame({"x": [1, 2]})})
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert z.namelist() == ["a.csv"]


# ── prep cache: second run skips the load; any input change busts it ─────────

def test_cache_prep_skips_reload(monkeypatch):
    import functions.dispersion._api as api
    import functions.dispersion._backtester as bt

    rng = np.random.default_rng(9)
    tickers = [f"C{i}" for i in range(6)]
    pnl = np.column_stack([rng.normal(0.2 * (i + 1), 0.8, 250) for i in range(6)])
    col_map = {t: i for i, t in enumerate(tickers)}
    dates = pd.bdate_range("2024-01-02", periods=250)
    price_df = pd.DataFrame(100.0, index=dates, columns=tickers)
    calls = {"n": 0}

    def fake_load(self, basket):
        calls["n"] += 1
        legs = list(basket.long_candidates) + list(basket.short_candidates)
        return {"variance_px": price_df, "corridor_px": None, "legs": legs,
                "long_legs": list(basket.long_candidates),
                "short_legs": list(basket.short_candidates)}

    monkeypatch.setattr(bt.DispersionDataLoader, "load", fake_load)
    monkeypatch.setattr(api, "_build_pnl_matrix",
                        lambda vpx, cpx, legs, cfg: (pnl.copy(), dict(col_map)))
    empty_ts = pd.DataFrame({"Long Leg": [], "Short Leg": [], "Result": []})
    monkeypatch.setattr(bt.DispersionBacktester, "run_from_optimization",
                        lambda self, **kw: BacktestResult(timeseries=empty_ts.copy()))
    api._PREP_CACHE.clear()

    from functions.dispersion.models import OptimizationConstraints
    long_df = pd.DataFrame({"Variance Asset": tickers,
                            "Strike Mono Var Swap (%)": 12.0,
                            "Min Weight": 5.0, "Max Weight": 60.0})
    cfg = DispersionConfig(missing_data_policy=MissingDataPolicy.FILL_ZERO)
    cons = OptimizationConstraints(
        min_stocks_long=2, max_stocks_long=3, min_stocks_short=0,
        max_stocks_short=0, max_net_strike=10.0, population_size=30,
        max_generations=40, time_limit_seconds=10.0, stagnation_limit=10)

    r1 = api.optimize(long_df, cfg, cons, score_weights={"mean_payoff": 1.0},
                      seed=0, cache_prep=True)
    assert calls["n"] == 1
    # different weights, same data → NO reload, and a valid (different-config) run
    r2 = api.optimize(long_df, cfg, cons, score_weights={"min_payoff": 1.0},
                      seed=0, cache_prep=True)
    assert calls["n"] == 1, "second run must hit the prep cache"
    assert r2.long_basket, "cached prep must still produce a real result"
    # identical config twice → identical basket (cache must not perturb results)
    r3 = api.optimize(long_df, cfg, cons, score_weights={"mean_payoff": 1.0},
                      seed=0, cache_prep=True)
    assert calls["n"] == 1
    assert r3.long_basket == r1.long_basket and r3.score == r1.score
    # change the data → cache busted
    long_df2 = long_df.copy()
    long_df2.loc[0, "Max Weight"] = 55.0
    api.optimize(long_df2, cfg, cons, score_weights={"mean_payoff": 1.0},
                 seed=0, cache_prep=True)
    assert calls["n"] == 2, "changed input must bust the cache"
    # default (cache off) → always reloads
    api.optimize(long_df, cfg, cons, score_weights={"mean_payoff": 1.0}, seed=0)
    assert calls["n"] == 3
    api._PREP_CACHE.clear()


# ── shared calibration: reuse is bit-identical and actually shared ───────────

def test_reference_cache_shared_and_bit_identical():
    from functions.dispersion._optimizer import DispersionOptimizer
    from functions.dispersion.scoring import MetricWeights
    from functions.dispersion.models import OptimizationConstraints

    rng = np.random.default_rng(21)
    names = [f"R{i}" for i in range(8)]
    pnl = np.column_stack([rng.normal(0.15 * (i + 1), 0.9, 250) for i in range(8)])
    col_map = {t: i for i, t in enumerate(names)}
    legs = [DispersionLeg(variance_asset=t, strike_mono_var_swap=0.12,
                          min_weight=0.05, max_weight=0.60) for t in names]
    cons = OptimizationConstraints(
        min_stocks_long=2, max_stocks_long=3, min_stocks_short=0,
        max_stocks_short=0, max_net_strike=10.0, population_size=30,
        max_generations=30, time_limit_seconds=8.0, stagnation_limit=10)

    def run(weights, cache):
        opt = DispersionOptimizer(
            long_candidates=legs, short_candidates=[], pnl_matrix=pnl,
            column_map=col_map, constraints=cons,
            missing_data_policy=MissingDataPolicy.FILL_ZERO,
            metric_weights=MetricWeights(weights), seed=0,
            reference_cache=cache)
        return opt.run()

    shared = {}
    r1 = run({"mean_payoff": 1.0}, shared)
    assert len(shared) == 1, "first config must populate the calibration cache"
    r2 = run({"hit_ratio": 1.0}, shared)          # same group (no tail metric)
    assert len(shared) == 1, "same-group config must REUSE the calibration"
    r2_solo = run({"hit_ratio": 1.0}, None)       # no cache at all
    assert r2.long_basket == r2_solo.long_basket and r2.score == r2_solo.score, (
        "cached-calibration run must be bit-identical to a solo run")
    r3 = run({"min_payoff": 1.0}, shared)          # tail metric → bigger sample
    assert len(shared) == 2, "tail-metric config needs its own calibration group"
    assert r3.long_basket, "tail config must still produce a result"
