"""Offline end-to-end tests of the _api.optimize() pipeline.

The Bloomberg loader and the production backtester are faked (monkeypatched)
so the FULL public path — validation, candidate filtering, forced/excluded
plumbing, GA, result packaging — runs offline on a synthetic P&L matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from functions.dispersion.models import (
    BacktestResult,
    DispersionConfig,
    MissingDataPolicy,
    OptimizationConstraints,
)

N_DAYS = 250


def _cons(**over):
    base = dict(
        min_stocks_long=2, max_stocks_long=3,
        min_stocks_short=0, max_stocks_short=0,
        max_net_strike=10.0, population_size=40,
        max_generations=120, time_limit_seconds=30.0,
        stagnation_limit=20,
    )
    base.update(over)
    return OptimizationConstraints(**base)


@pytest.fixture
def api_offline(monkeypatch):
    """Install fakes and return the _api module driven by a synthetic matrix."""
    import functions.dispersion._api as api
    import functions.dispersion._backtester as bt

    def install(pnl: np.ndarray, col_map: dict, tickers: list):
        dates = pd.bdate_range("2024-01-02", periods=pnl.shape[0])
        price_df = pd.DataFrame(100.0, index=dates, columns=tickers)

        def fake_load(self, basket):
            legs = list(basket.long_candidates) + list(basket.short_candidates)
            return {
                "price_data": price_df,
                "index_data": None,
                "legs": legs,
                "long_legs": list(basket.long_candidates),
                "short_legs": list(basket.short_candidates),
            }

        def fake_build(price_data, index_data, legs, cfg):
            return pnl.copy(), dict(col_map)

        empty_ts = pd.DataFrame({"Long Leg": [], "Short Leg": [], "Result": []})
        monkeypatch.setattr(bt.DispersionDataLoader, "load", fake_load)
        monkeypatch.setattr(api, "_build_pnl_matrix", fake_build)
        monkeypatch.setattr(
            bt.DispersionBacktester, "run_from_optimization",
            lambda self, **kw: BacktestResult(timeseries=empty_ts.copy()))
        return api

    return install


def _universe(n=8, seed=13):
    rng = np.random.default_rng(seed)
    tickers = [f"E{i}" for i in range(n)]
    means = np.linspace(0.0, 1.5, n)          # E{n-1} is clearly the best name
    pnl = np.column_stack([rng.normal(means[i], 0.8, N_DAYS) for i in range(n)])
    col_map = {tickers[i]: i for i in range(n)}
    long_df = pd.DataFrame({
        "Variance Asset": tickers,
        "Strike Mono Var Swap (%)": [12.0 + 0.5 * i for i in range(n)],
        "Min Weight": 5.0,
        "Max Weight": 60.0,
    })
    return tickers, pnl, col_map, long_df


def test_api_excluded_by_construction(api_offline):
    """Excluded names are removed from the candidate universe before the GA:
    the best name, once excluded, can never appear in the basket."""
    tickers, pnl, col_map, long_df = _universe()
    api = api_offline(pnl, col_map, tickers)
    cfg = DispersionConfig(missing_data_policy=MissingDataPolicy.FILL_ZERO)

    best_name = tickers[-1]
    res_free = api.optimize(long_df, cfg, _cons(), score_weights={"mean_payoff": 1.0}, seed=0)
    assert best_name in [k for k, _ in res_free.long_basket], (
        "sanity: the dominant name should be picked when not excluded")

    res_excl = api.optimize(long_df, cfg, _cons(), score_weights={"mean_payoff": 1.0},
                            seed=0, excluded_tickers=[best_name.lower()])  # casefold match
    got = [k for k, _ in res_excl.long_basket]
    assert best_name not in got, f"excluded name {best_name} leaked into basket {got}"
    assert res_excl.long_basket, "exclusion must not empty the result"


def test_api_forced_present_and_bundle_saved(api_offline, tmp_path):
    """Forced name always in the basket; save_bundle_path writes a replayable
    bundle whose replay reproduces the API result exactly."""
    from functions.dispersion.run_bundle import load_run_bundle

    tickers, pnl, col_map, long_df = _universe()
    api = api_offline(pnl, col_map, tickers)
    cfg = DispersionConfig(missing_data_policy=MissingDataPolicy.FILL_ZERO)
    bundle_dir = str(tmp_path / "bundle")

    worst = tickers[0]
    res = api.optimize(long_df, cfg, _cons(), score_weights={"mean_payoff": 1.0},
                       seed=0, forced_tickers=[worst], save_bundle_path=bundle_dir)
    assert worst in [k for k, _ in res.long_basket]

    bundle = load_run_bundle(bundle_dir)
    assert bundle.result is not None and bundle.result["long_basket"], "result snapshot missing"
    replay = bundle.replay()
    assert replay.long_basket == res.long_basket
    assert replay.score == pytest.approx(res.score, abs=0)
    assert replay.scoring_signature == res.scoring_signature


def _xc_universe(n=5, seed=17):
    """Cross-corridor universe: ONE shared Variance Asset (index), n distinct
    Corridor Condition Assets (the traded single stocks) — the per-name key."""
    rng = np.random.default_rng(seed)
    corr = [f"{100 + i} XX Equity" for i in range(n)]     # the traded stocks
    means = np.linspace(0.0, 1.5, n)
    pnl = np.column_stack([rng.normal(means[i], 0.8, N_DAYS) for i in range(n)])
    col_map = {corr[i]: i for i in range(n)}              # keyed by CORRIDOR asset
    long_df = pd.DataFrame({
        "Variance Asset": ["IDX Index"] * n,             # shared index
        "Corridor Condition Asset": corr,
        "Strike Cross Corridor (%)": [20.0 + i for i in range(n)],
        "Strike Mono Var Swap (%)": [12.0 + 0.5 * i for i in range(n)],
        "Min Weight": 5.0,
        "Max Weight": 60.0,
    })
    return corr, pnl, col_map, long_df


def test_api_cross_corridor_force_corridor_asset(api_offline):
    """Issue 3: in cross-corridor mode the forcible/candidate key is the
    Corridor Condition Asset (stock), not the shared Variance Asset (index).
    Forcing the stock must work; forcing a bogus name must raise a message that
    reports the CORRIDOR sample (stocks), not the index."""
    corr, pnl, col_map, long_df = _xc_universe()
    api = api_offline(pnl, col_map, ["IDX Index"])
    cfg = DispersionConfig(cross_corridor=True, missing_data_policy=MissingDataPolicy.FILL_ZERO)

    # (a) forcing the actual corridor stock succeeds (matched on the corridor key)
    target = corr[0]
    res = api.optimize(long_df, cfg, _cons(), score_weights={"mean_payoff": 1.0},
                       seed=0, forced_tickers=[target.lower()])   # casefold match
    assert target in [k for k, _ in res.long_basket], (
        f"forced corridor asset {target} missing from basket {res.long_basket}")

    # (b) forcing a bogus name raises, and the message is corridor-aware:
    #     it names the Corridor Condition Asset key and samples the STOCKS.
    with pytest.raises(ValueError) as ei:
        api.optimize(long_df, cfg, _cons(), score_weights={"mean_payoff": 1.0},
                     seed=0, forced_tickers=["005930 kp equity"])
    msg = str(ei.value)
    assert "Corridor Condition Asset" in msg, msg
    assert "005930 kp equity" in msg, msg
    assert corr[0] in msg, f"sample should list the stocks, got: {msg}"
    assert "IDX Index" not in msg, f"sample must NOT be the index: {msg}"


def test_api_mono_force_bogus_message(api_offline):
    """Mono mode: the bogus-forced message reports the Variance Asset key."""
    tickers, pnl, col_map, long_df = _universe()
    api = api_offline(pnl, col_map, tickers)
    cfg = DispersionConfig(missing_data_policy=MissingDataPolicy.FILL_ZERO)
    with pytest.raises(ValueError) as ei:
        api.optimize(long_df, cfg, _cons(), score_weights={"mean_payoff": 1.0},
                     seed=0, forced_tickers=["nope zz equity"])
    assert "Variance Asset" in str(ei.value)
