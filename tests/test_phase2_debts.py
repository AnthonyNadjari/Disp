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


# ---------------------------------------------------------------------------
# smooth_weights: metric-faithful blend (ws-aware + optional metrics)
# ---------------------------------------------------------------------------


def _fitted_solver(weights_dict, n=4, seed=3, with_extras=False):
    from functions.dispersion.scoring import (
        MetricWeights, ScoreContext, WeightConstraints, make_default_score_function)
    from functions.dispersion.scoring.weight_solver import WeightSolver
    rng = np.random.default_rng(seed)
    ctx = ScoreContext(n_days=N_DAYS)
    samples = [rng.normal(0.8, 1.2, N_DAYS) for _ in range(80)]
    extras = ([{"weighted_strike": float(0.08 + 0.10 * rng.random())} for _ in samples]
              if with_extras else None)
    sf = make_default_score_function(weights=MetricWeights(weights_dict))
    sf.build_reference(samples, ctx, sample_extras=extras)
    wc = WeightConstraints(min_weight=0.05, max_weight=0.60, max_stocks=n)
    return WeightSolver(sf, ctx, wc, missing_data_policy="fill_zero"), ctx


def test_smooth_weights_requires_strikes_when_ws_active():
    solver, _ = _fitted_solver({"mean_payoff": 0.5, "weighted_strike": 0.5},
                               with_extras=True)
    rng = np.random.default_rng(4)
    pnl = rng.normal(0.8, 1.0, (N_DAYS, 4))
    w_star = np.array([0.4, 0.3, 0.2, 0.1])
    with pytest.raises(ValueError, match="weighted_strike"):
        solver.smooth_weights(w_star, pnl, np.arange(4))


def test_smooth_weights_ws_aware_blend_floor_holds():
    solver, _ = _fitted_solver({"mean_payoff": 0.5, "weighted_strike": 0.5},
                               with_extras=True)
    rng = np.random.default_rng(5)
    base = rng.normal(0.8, 1.0, (N_DAYS, 1))
    pnl = np.repeat(base, 4, axis=1) + rng.normal(0, 1e-3, (N_DAYS, 4))
    strikes = np.array([0.08, 0.12, 0.16, 0.20])
    res = solver.solve(pnl, np.arange(4), strikes=strikes)
    assert res.feasible
    w_star = res.weights
    eps = 0.05
    w_sm = solver.smooth_weights(w_star.copy(), pnl, np.arange(4),
                                 eps_min=eps, strikes=strikes)

    def blend(w):
        p = pnl @ w
        return 0.5 * float(np.mean(p)) - 0.5 * float(np.dot(w, strikes))

    if not np.allclose(w_sm, w_star):
        # accepted smoothing: less dispersed AND the ws-aware blend floor held
        assert np.std(w_sm) < np.std(w_star)
        assert blend(w_sm) >= blend(w_star) - 4 * eps - 1e-6
        assert float(np.min(pnl @ w_sm)) >= float(np.min(pnl @ w_star)) - 4 * eps - 1e-6


def test_smooth_weights_optional_metric_blend():
    solver, ctx = _fitted_solver({"mean_payoff": 0.6, "cvar_5": 0.4})
    rng = np.random.default_rng(6)
    pnl = rng.normal(0.6, 1.0, (N_DAYS, 4))
    pnl[:, 0] += 0.8
    res = solver.solve(pnl, np.arange(4))
    assert res.feasible
    w_star = res.weights
    w_sm = solver.smooth_weights(w_star.copy(), pnl, np.arange(4), eps_min=0.05)

    from functions.dispersion.scoring.metrics import CVaR5

    def blend(w):
        p = pnl @ w
        return 0.6 * float(np.mean(p)) + 0.4 * CVaR5().compute(p, ctx)

    if not np.allclose(w_sm, w_star):
        assert np.std(w_sm) < np.std(w_star)
        assert blend(w_sm) >= blend(w_star) - 4 * 0.05 - 1e-6


# ---------------------------------------------------------------------------
# n_reference_samples: exposed parameter, defaults unchanged
# ---------------------------------------------------------------------------


def _quick_opt(n_ref=None, seed=0, weights=None):
    names = [f"R{i}" for i in range(6)]
    pnl = _mk_pnl(6, seed=23)
    col_map = {n: i for i, n in enumerate(names)}
    opt = DispersionOptimizer(
        long_candidates=_mk_legs(names),
        short_candidates=[],
        pnl_matrix=pnl,
        column_map=col_map,
        constraints=_cons(),
        missing_data_policy=MissingDataPolicy.FILL_ZERO,
        metric_weights=MetricWeights(weights or {"mean_payoff": 1.0}),
        seed=seed,
        n_reference_samples=n_ref,
    )
    return opt.run()


def test_n_reference_samples_default_unchanged():
    """Explicitly passing the adaptive default (300 for non-tail configs)
    must reproduce the default run bit-for-bit."""
    r_default = _quick_opt(n_ref=None)
    r_explicit = _quick_opt(n_ref=300)
    assert r_default.long_basket == r_explicit.long_basket
    assert r_default.score == r_explicit.score
    assert r_default.scoring_signature == r_explicit.scoring_signature


def test_n_reference_samples_changes_reference_and_signature():
    r_default = _quick_opt(n_ref=None)
    r_more = _quick_opt(n_ref=600)
    assert r_more.reference_size > r_default.reference_size
    assert r_more.scoring_signature != r_default.scoring_signature


def test_n_reference_samples_too_small_raises():
    with pytest.raises(ValueError, match="n_reference_samples"):
        _quick_opt(n_ref=50)


# ---------------------------------------------------------------------------
# _calculate_stock_quality: decoupled from legacy ScoreWeights
# ---------------------------------------------------------------------------


def _opt_for_quality(weights):
    names = [f"Q{i}" for i in range(4)]
    legs = _mk_legs(names)
    # Give one leg precomputed backtest metrics (production shape)
    legs[0].metrics = {"last_value": 2.0, "avg_5y": 0.05, "avg_3y": 0.06,
                       "hit_ratio": 80.0, "max_drawdown": -0.30}
    pnl = _mk_pnl(4, seed=29)
    return DispersionOptimizer(
        long_candidates=legs,
        short_candidates=[],
        pnl_matrix=pnl,
        column_map={n: i for i, n in enumerate(names)},
        constraints=_cons(),
        missing_data_policy=MissingDataPolicy.FILL_ZERO,
        metric_weights=MetricWeights(weights),
        seed=0,
    )


def test_quality_neutral_without_leg_metrics():
    opt = _opt_for_quality({"mean_payoff": 1.0})
    # legs[1..3] have no metrics dict -> neutral bias
    assert opt._quality_long[1] == 0.0
    assert opt._quality_long[2] == 0.0


def test_quality_follows_active_metric_weights():
    q_carry = _opt_for_quality({"last_carry": 1.0})._quality_long[0]
    q_hit = _opt_for_quality({"hit_ratio": 1.0})._quality_long[0]
    q_risk = _opt_for_quality({"min_payoff": 1.0})._quality_long[0]
    # last_value=2.0/3 vs hit=(0.8-0.5)*2 -> both positive but different
    assert q_carry == pytest.approx(2.0 / 3.0)
    assert q_hit == pytest.approx(0.6)
    # pure risk objective penalizes the drawdown proxy -> negative bias
    assert q_risk == pytest.approx(-0.30 / 1.5)
    # sharpe has no per-leg proxy -> neutral
    assert _opt_for_quality({"sharpe_payoff": 1.0})._quality_long[0] == 0.0


def test_legacy_score_weights_param_removed():
    import inspect
    sig = inspect.signature(DispersionOptimizer.__init__)
    assert "score_weights" not in sig.parameters
