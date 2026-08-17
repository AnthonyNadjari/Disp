"""Phase 4b — optimize_multi: one data load, N configs, comparison table.

Reuses the offline _api harness (fake loader / injected matrix / fake
backtester).
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


def _cons():
    return OptimizationConstraints(
        min_stocks_long=2, max_stocks_long=3,
        min_stocks_short=0, max_stocks_short=0,
        max_net_strike=10.0, population_size=40,
        max_generations=120, time_limit_seconds=30.0,
        stagnation_limit=20,
    )


@pytest.fixture
def api_offline_counted(monkeypatch):
    """Offline _api with a load-call counter (to prove single-load)."""
    import functions.dispersion._api as api
    import functions.dispersion._backtester as bt

    counter = {"loads": 0}

    def install(pnl, col_map, tickers):
        dates = pd.bdate_range("2024-01-02", periods=pnl.shape[0])
        price_df = pd.DataFrame(100.0, index=dates, columns=tickers)

        def fake_load(self, basket):
            counter["loads"] += 1
            legs = list(basket.long_candidates) + list(basket.short_candidates)
            return {"variance_px": price_df, "corridor_px": None, "legs": legs,
                    "long_legs": list(basket.long_candidates),
                    "short_legs": list(basket.short_candidates)}

        empty_ts = pd.DataFrame({"Long Leg": [], "Short Leg": [], "Result": []})
        monkeypatch.setattr(bt.DispersionDataLoader, "load", fake_load)
        monkeypatch.setattr(api, "_build_pnl_matrix",
                            lambda price_data, index_data, legs, cfg: (pnl.copy(), dict(col_map)))
        monkeypatch.setattr(bt.DispersionBacktester, "run_from_optimization",
                            lambda self, **kw: BacktestResult(timeseries=empty_ts.copy()))
        return api, counter

    return install


def _universe(n=7, seed=41):
    rng = np.random.default_rng(seed)
    tickers = [f"M{i}" for i in range(n)]
    means = np.linspace(-0.1, 1.4, n)
    pnl = np.column_stack([rng.normal(means[i], 0.9, N_DAYS) for i in range(n)])
    col_map = {tickers[i]: i for i in range(n)}
    long_df = pd.DataFrame({
        "Variance Asset": tickers,
        "Strike Mono Var Swap (%)": [12.0 + 0.5 * i for i in range(n)],
        "Min Weight": 5.0,
        "Max Weight": 60.0,
    })
    return tickers, pnl, col_map, long_df


CONFIGS = [
    {"mean_payoff": 1.0},
    {"mean_payoff": 0.5, "hit_ratio": 0.5},
]


def test_multi_matches_single_per_config(api_offline_counted):
    tickers, pnl, col_map, long_df = _universe()
    api, counter = api_offline_counted(pnl, col_map, tickers)
    cfg = DispersionConfig(missing_data_policy=MissingDataPolicy.FILL_ZERO)

    multi = api.optimize_multi(long_df, cfg, _cons(), configs=CONFIGS, seed=0)
    assert len(multi.results) == 2
    loads_after_multi = counter["loads"]
    assert loads_after_multi == 1, f"data loaded {loads_after_multi} times, expected once"

    for weights, multi_res in zip(CONFIGS, multi.results):
        single = api.optimize(long_df, cfg, _cons(), score_weights=dict(weights), seed=0)
        assert multi_res.long_basket == single.long_basket, (
            f"multi != single for {weights}")
        assert multi_res.score == single.score
        assert multi_res.scoring_signature == single.scoring_signature


def test_multi_comparison_table_contents(api_offline_counted):
    tickers, pnl, col_map, long_df = _universe()
    api, _ = api_offline_counted(pnl, col_map, tickers)
    cfg = DispersionConfig(missing_data_policy=MissingDataPolicy.FILL_ZERO)

    multi = api.optimize_multi(long_df, cfg, _cons(), configs=CONFIGS, seed=0)
    comp = multi.comparison
    assert len(comp) == 2
    for col in ("config", "score", "net_strike_pct", "basket", "scoring_signature"):
        assert col in comp.columns
    # Active-metric percentile columns exist and are within [0, 1]
    assert "pct_mean_payoff" in comp.columns
    pcts = comp[[c for c in comp.columns if c.startswith("pct_")]].to_numpy(dtype=float)
    finite = pcts[np.isfinite(pcts)]
    assert ((finite >= 0.0) & (finite <= 1.0)).all()
    # Raw metric values present for the winner
    assert "raw_mean_payoff" in comp.columns
    # Backtests attached by default
    assert all(r.backtest is not None for r in multi.results)


def test_multi_empty_configs_raises(api_offline_counted):
    tickers, pnl, col_map, long_df = _universe()
    api, _ = api_offline_counted(pnl, col_map, tickers)
    cfg = DispersionConfig(missing_data_policy=MissingDataPolicy.FILL_ZERO)
    with pytest.raises(ValueError, match="configs"):
        api.optimize_multi(long_df, cfg, _cons(), configs=[])


# ---------------------------------------------------------------------------
# Phase 4d — bootstrap robustness diagnostic
# ---------------------------------------------------------------------------


def test_robustness_check_attaches_deterministic_diagnostic(api_offline_counted):
    tickers, pnl, col_map, long_df = _universe()
    api, _ = api_offline_counted(pnl, col_map, tickers)
    cfg = DispersionConfig(missing_data_policy=MissingDataPolicy.FILL_ZERO)

    r1 = api.optimize(long_df, cfg, _cons(), score_weights={"mean_payoff": 0.6, "hit_ratio": 0.4},
                      seed=0, robustness_check=True)
    rb = r1.robustness
    assert rb is not None
    assert rb["n_draws"] == 300
    assert 0.0 <= rb["top1_freq"] <= rb["top3_freq"] <= 1.0
    assert rb["n_challengers"] >= 1
    assert set(rb["winner_raw_ci"]) == {"mean_payoff", "hit_ratio"}
    for ci in rb["winner_raw_ci"].values():
        assert ci["lo"] <= ci["mean"] <= ci["hi"]

    # Deterministic: identical re-run reproduces the diagnostic exactly
    r2 = api.optimize(long_df, cfg, _cons(), score_weights={"mean_payoff": 0.6, "hit_ratio": 0.4},
                      seed=0, robustness_check=True)
    assert r2.robustness == rb

    # Default OFF
    r3 = api.optimize(long_df, cfg, _cons(), score_weights={"mean_payoff": 0.6, "hit_ratio": 0.4},
                      seed=0)
    assert r3.robustness is None


def test_robustness_dominant_winner_is_stable(api_offline_counted):
    """A clearly dominant name should keep the winner on top in most draws."""
    rng = np.random.default_rng(60)
    tickers = [f"D{i}" for i in range(6)]
    pnl = np.column_stack([rng.normal(0.1, 1.0, N_DAYS) for _ in range(5)]
                          + [rng.normal(3.0, 0.3, N_DAYS)])  # D5 dominates
    col_map = {t: i for i, t in enumerate(tickers)}
    long_df = pd.DataFrame({
        "Variance Asset": tickers,
        "Strike Mono Var Swap (%)": 12.0,
        "Min Weight": 5.0,
        "Max Weight": 60.0,
    })
    api, _ = api_offline_counted(pnl, col_map, tickers)
    cfg = DispersionConfig(missing_data_policy=MissingDataPolicy.FILL_ZERO)
    res = api.optimize(long_df, cfg, _cons(), score_weights={"mean_payoff": 1.0},
                       seed=0, robustness_check=True)
    assert "D5" in [k for k, _ in res.long_basket]
    assert res.robustness["top1_freq"] >= 0.5, (
        f"dominant winner unstable: top1_freq={res.robustness['top1_freq']:.2f}")
