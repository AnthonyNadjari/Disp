"""Unit tests for the scoring-layer features added with the weighted-strike
objective and the forced/excluded names plumbing.

Fully offline: no Bloomberg, no Streamlit.
"""

from __future__ import annotations

import dataclasses
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from functions.dispersion.scoring import (
    MetricWeights,
    ScoreContext,
    WeightConstraints,
    WeightedStrike,
    make_default_score_function,
)
from functions.dispersion.scoring.weight_solver import WeightSolver

RNG = np.random.default_rng(3)
N_DAYS = 300


def _ref_samples(n=80):
    return [RNG.normal(0.8, 1.2, N_DAYS) for _ in range(n)]


def _extras_for(samples):
    # Synthetic per-sample weighted strikes in a plausible decimal range
    return [{"weighted_strike": float(0.08 + 0.10 * RNG.random())} for _ in samples]


# ---------------------------------------------------------------------------
# WeightedStrike metric + ScoreContext plumbing
# ---------------------------------------------------------------------------


def test_weighted_strike_metric_reads_context():
    m = WeightedStrike()
    pnl = RNG.normal(0, 1, 50)
    assert math.isnan(m.compute(pnl, ScoreContext(n_days=50)))
    ctx = ScoreContext(n_days=50, weighted_strike=0.123)
    assert m.compute(pnl, ctx) == pytest.approx(0.123)
    # frozen dataclass stays replace-able (used per-evaluation by the solver)
    ctx2 = dataclasses.replace(ctx, weighted_strike=0.2)
    assert m.compute(pnl, ctx2) == pytest.approx(0.2)


def test_factory_wires_all_default_metrics():
    sf = make_default_score_function()
    names = {m.name for m in sf.metrics}
    assert {"last_carry", "hit_ratio", "min_payoff", "mean_payoff",
            "max_drawdown", "cvar_5", "sharpe_payoff", "weighted_strike"} <= names


@pytest.mark.parametrize("optional_metric", ["max_drawdown", "cvar_5", "sharpe_payoff"])
def test_optional_metric_weight_does_not_crash(optional_metric):
    """Regression: weighting an optional metric used to raise
    'Weighted metric not in provided metrics list'."""
    sf = make_default_score_function(
        weights=MetricWeights({optional_metric: 0.5, "mean_payoff": 0.5})
    )
    ctx = ScoreContext(n_days=N_DAYS)
    sf.build_reference(_ref_samples(), ctx)
    s = sf.score(RNG.normal(0.8, 1.2, N_DAYS), ctx)
    assert 0.0 <= s <= 1.0 and np.isfinite(s)


# ---------------------------------------------------------------------------
# build_reference: sample_extras + unfittable-metric guard
# ---------------------------------------------------------------------------


def test_build_reference_without_extras_warns_and_stays_usable():
    sf = make_default_score_function()  # weighted_strike at weight 0
    ctx = ScoreContext(n_days=N_DAYS)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sf.build_reference(_ref_samples(), ctx)
    assert any("weighted_strike" in str(w.message) for w in rec)
    # Scoring still works; the unfitted metric contributes 0 at weight 0
    s = sf.score(RNG.normal(0.8, 1.2, N_DAYS), ctx)
    assert np.isfinite(s)


def test_build_reference_active_unfittable_raises():
    sf = make_default_score_function(
        weights=MetricWeights({"weighted_strike": 0.5, "mean_payoff": 0.5})
    )
    with pytest.raises(RuntimeError, match="weighted_strike"):
        sf.build_reference(_ref_samples(), ScoreContext(n_days=N_DAYS))


def test_build_reference_with_extras_scores_strike_direction():
    """Lower strike must score strictly better when weighted_strike is active."""
    sf = make_default_score_function(weights=MetricWeights({"weighted_strike": 1.0}))
    samples = _ref_samples()
    sf.build_reference(samples, ScoreContext(n_days=N_DAYS),
                       sample_extras=_extras_for(samples))
    pnl = RNG.normal(0.8, 1.2, N_DAYS)
    lo = sf.score(pnl, ScoreContext(n_days=N_DAYS, weighted_strike=0.09))
    hi = sf.score(pnl, ScoreContext(n_days=N_DAYS, weighted_strike=0.17))
    assert lo > hi, f"lower strike should score higher: {lo:.4f} vs {hi:.4f}"


def test_score_smooth_from_raw_missing_key_is_worst_not_crash():
    sf = make_default_score_function(
        weights=MetricWeights({"mean_payoff": 0.5, "hit_ratio": 0.5})
    )
    sf.build_reference(_ref_samples(), ScoreContext(n_days=N_DAYS))
    full = sf.score_smooth_from_raw({"mean_payoff": 0.9, "hit_ratio": 0.7})
    partial = sf.score_smooth_from_raw({"mean_payoff": 0.9})
    assert np.isfinite(full) and np.isfinite(partial)
    assert partial < full  # missing raw -> worst (0) contribution, never a crash


# ---------------------------------------------------------------------------
# Inner solver: the strike objective actually tilts the weights
# ---------------------------------------------------------------------------


def test_solver_tilts_weights_toward_low_strike():
    n = 4
    # Identical payoff quality; only strikes differ -> the tilt must come
    # from the weighted_strike objective alone.
    base = RNG.normal(0.8, 1.0, (N_DAYS, 1))
    pnl = np.repeat(base, n, axis=1) + RNG.normal(0, 1e-3, (N_DAYS, n))
    strikes = np.array([0.08, 0.12, 0.16, 0.20])

    ctx = ScoreContext(n_days=N_DAYS)
    samples = _ref_samples()
    wc = WeightConstraints(min_weight=0.05, max_weight=0.60, max_stocks=n)

    sf = make_default_score_function(
        weights=MetricWeights({"mean_payoff": 0.5, "weighted_strike": 0.5})
    )
    sf.build_reference(samples, ctx, sample_extras=_extras_for(samples))
    solver = WeightSolver(sf, ctx, wc, missing_data_policy="fill_zero")
    res = solver.solve(pnl, np.arange(n), strikes=strikes)
    assert res.feasible
    achieved = float(res.weights @ strikes)
    equal = float(np.full(n, 1 / n) @ strikes)
    assert achieved < equal - 1e-3, (
        f"strike objective inactive: achieved {achieved:.4f} vs equal-weight {equal:.4f}"
    )
    # The cheapest name must carry the largest weight
    assert int(np.argmax(res.weights)) == 0


# ---------------------------------------------------------------------------
# API pre-load validation (runs before any Bloomberg access)
# ---------------------------------------------------------------------------


def _dummy_inputs():
    from functions.dispersion.models import DispersionConfig, OptimizationConstraints
    long_df = pd.DataFrame({
        "Variance Asset": ["AAA", "BBB", "CCC"],
        "Strike Mono Var Swap (%)": [12.0, 13.0, 14.0],
    })
    cfg = DispersionConfig()
    cons = OptimizationConstraints(min_stocks_long=2, max_stocks_long=2,
                                   min_stocks_short=0, max_stocks_short=0)
    return long_df, cfg, cons


def test_api_forced_excluded_overlap_raises_before_load():
    from functions.dispersion._api import optimize
    long_df, cfg, cons = _dummy_inputs()
    with pytest.raises(ValueError, match="both forced and excluded"):
        optimize(long_df, cfg, cons,
                 forced_tickers=["AAA"], excluded_tickers=["aaa"])  # casefold match


def test_api_too_many_forced_raises_before_load():
    from functions.dispersion._api import optimize
    long_df, cfg, cons = _dummy_inputs()
    with pytest.raises(ValueError, match="max_stocks_long"):
        optimize(long_df, cfg, cons,
                 forced_tickers=["AAA", "BBB", "CCC"])  # 3 forced > max 2
