"""End-to-end corner extremality tests for the dispersion optimizer.

THE requirement under test: putting weight=1 on a single metric must make the
FULL optimizer (outer subset selection + inner weight solve) return the basket
that maximizes that metric among ALL feasible baskets — verified by exhaustive
enumeration on a small synthetic universe, with a ground truth computed
independently of the solver under test.

Also covered:
- the homogeneous-universe scenario (~20 near-identical stocks), where the
  GA's proxy ranking is least reliable — the realistic production case;
- forced-name enforcement (forced tickers always present, weights within
  their input bounds);
- the 2-start weight diagnostic (equal vs greedy start converge in score).

Runs fully offline: DispersionOptimizer is driven directly with a synthetic
P&L matrix (no Bloomberg, no Streamlit).
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy.optimize import linprog

from functions.dispersion.models import (
    DispersionLeg,
    MissingDataPolicy,
    OptimizationConstraints,
)
from functions.dispersion._optimizer import DispersionOptimizer
from functions.dispersion.scoring import MetricWeights

SEED = 7
N_DAYS = 400


# ---------------------------------------------------------------------------
# Synthetic universes
# ---------------------------------------------------------------------------


def _make_universe(n_stocks: int, heterogeneous: bool, seed: int = SEED):
    """Build (legs, pnl_matrix, col_map) with reproducible synthetic payoffs."""
    rng = np.random.default_rng(seed)
    names = [f"T{i}" for i in range(n_stocks)]
    if heterogeneous:
        means = np.linspace(-0.5, 2.0, n_stocks)
        stds = np.linspace(0.5, 2.5, n_stocks)
    else:
        # Near-identical quality: tiny spread — the hard case for proxy ranking
        means = 0.8 + 0.02 * rng.standard_normal(n_stocks)
        stds = 1.0 + 0.02 * rng.standard_normal(n_stocks)
    pnl = np.column_stack([
        rng.normal(means[i], max(stds[i], 0.1), N_DAYS) for i in range(n_stocks)
    ])
    legs = [
        DispersionLeg(
            variance_asset=names[i],
            strike_mono_var_swap=0.10 + 0.01 * i,  # distinct strikes for the strike tests
            min_weight=0.05,
            max_weight=0.60,
        )
        for i in range(n_stocks)
    ]
    col_map = {names[i]: i for i in range(n_stocks)}
    return legs, pnl, col_map


def _constraints(min_l: int, max_l: int, time_s: float = 12.0) -> OptimizationConstraints:
    return OptimizationConstraints(
        min_stocks_long=min_l,
        max_stocks_long=max_l,
        min_stocks_short=0,
        max_stocks_short=0,
        max_net_strike=10.0,  # loose: strike never binds in these tests
        population_size=60,
        max_generations=300,
        time_limit_seconds=time_s,
        stagnation_limit=40,
    )


def _run_optimizer(legs, pnl, col_map, cons, weights: dict, seed: int = 0,
                   forced=None):
    opt = DispersionOptimizer(
        long_candidates=legs,
        short_candidates=[],
        pnl_matrix=pnl,
        column_map=col_map,
        constraints=cons,
        missing_data_policy=MissingDataPolicy.FILL_ZERO,
        metric_weights=MetricWeights(weights),
        seed=seed,
        forced_long_indices=forced,
    )
    result = opt.run()
    assert result.long_basket, "optimizer returned an empty basket"
    return opt, result


def _basket_arrays(result, col_map, legs):
    keys = [k for k, _ in result.long_basket]
    w = np.array([wt for _, wt in result.long_basket], dtype=float)
    cols = [col_map[k] for k in keys]
    strikes = np.array(
        [next(l.strike_mono_var_swap for l in legs if l.variance_asset == k) for k in keys]
    )
    return keys, w, cols, strikes


# ---------------------------------------------------------------------------
# Independent ground truths (never reuse WeightSolver)
# ---------------------------------------------------------------------------


def _lp_max_linear(values: np.ndarray, lo: float, hi: float, maximize: bool = True):
    """max/min of values·w  s.t. sum(w)=1, lo<=w<=hi  — exact via linprog."""
    n = len(values)
    c = -values if maximize else values
    res = linprog(c, A_eq=np.ones((1, n)), b_eq=[1.0],
                  bounds=[(lo, hi)] * n, method="highs")
    assert res.success
    return float(values @ res.x)


def _lp_maximin(sub_pnl: np.ndarray, lo: float, hi: float) -> float:
    """max_w min_t (P@w)_t  s.t. sum(w)=1, lo<=w<=hi — exact epigraph LP."""
    d, n = sub_pnl.shape
    # variables: [w (n), t (1)] ; maximize t ; P@w >= t
    c = np.zeros(n + 1)
    c[-1] = -1.0
    A_ub = np.hstack([-sub_pnl, np.ones((d, 1))])
    b_ub = np.zeros(d)
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                  bounds=[(lo, hi)] * n + [(None, None)], method="highs")
    assert res.success
    return float(res.x[-1])


def _grid_weights(n: int, lo: float, hi: float, steps: int = 6):
    """Coarse feasible simplex grid for non-smooth ground truths (hit_ratio)."""
    grid = np.linspace(lo, hi, steps)
    for combo in itertools.product(grid, repeat=n):
        w = np.array(combo)
        s = w.sum()
        if s <= 0:
            continue
        w = w / s
        if np.all(w >= lo - 1e-9) and np.all(w <= hi + 1e-9):
            yield w


def _brute_force_best(metric: str, legs, pnl, min_l: int, max_l: int):
    """Best achievable raw value of `metric` over ALL feasible (subset, weights)."""
    n = len(legs)
    lo, hi = legs[0].min_weight, legs[0].max_weight
    best = -math.inf
    for size in range(min_l, max_l + 1):
        for subset in itertools.combinations(range(n), size):
            sub = pnl[:, list(subset)]
            if metric == "last_carry":
                best = max(best, _lp_max_linear(sub[-1, :], lo, hi))
            elif metric == "mean_payoff":
                best = max(best, _lp_max_linear(sub.mean(axis=0), lo, hi))
            elif metric == "min_payoff":
                best = max(best, _lp_maximin(sub, lo, hi))
            elif metric == "weighted_strike":
                strikes = np.array([legs[i].strike_mono_var_swap for i in subset])
                # higher_is_better=False → best achievable = MIN weighted strike
                best = max(best, -_lp_max_linear(strikes, lo, hi, maximize=False))
            elif metric == "hit_ratio":
                for w in _grid_weights(size, lo, hi):
                    best = max(best, float(np.mean(sub @ w > 0)))
            else:
                raise ValueError(metric)
    return best if metric != "weighted_strike" else -best


def _achieved(metric: str, result, legs, pnl, col_map) -> float:
    _, w, cols, strikes = _basket_arrays(result, col_map, legs)
    net = pnl[:, cols] @ w
    if metric == "last_carry":
        return float(net[-1])
    if metric == "mean_payoff":
        return float(net.mean())
    if metric == "min_payoff":
        return float(net.min())
    if metric == "hit_ratio":
        return float(np.mean(net > 0))
    if metric == "weighted_strike":
        return float(w @ strikes)
    raise ValueError(metric)


# ---------------------------------------------------------------------------
# Tests — heterogeneous small universe (exact ground truths)
# ---------------------------------------------------------------------------

HET = _make_universe(n_stocks=8, heterogeneous=True)
CONS_SMALL = _constraints(min_l=2, max_l=3)

EXACT_METRICS = ["last_carry", "mean_payoff", "min_payoff"]


@pytest.mark.parametrize("metric", EXACT_METRICS)
def test_corner_extremality_exact(metric):
    legs, pnl, col_map = HET
    _, result = _run_optimizer(legs, pnl, col_map, CONS_SMALL, {metric: 1.0})
    truth = _brute_force_best(metric, legs, pnl, 2, 3)
    got = _achieved(metric, result, legs, pnl, col_map)
    assert got >= truth - 1e-4, (
        f"corner extremality violated for {metric}: achieved {got:.6f} "
        f"< brute-force optimum {truth:.6f}"
    )


def test_corner_extremality_weighted_strike():
    legs, pnl, col_map = HET
    _, result = _run_optimizer(legs, pnl, col_map, CONS_SMALL, {"weighted_strike": 1.0})
    truth = _brute_force_best("weighted_strike", legs, pnl, 2, 3)
    got = _achieved("weighted_strike", result, legs, pnl, col_map)
    # higher_is_better=False → achieved must be <= truth + tol
    assert got <= truth + 1e-4, (
        f"weighted_strike extremality violated: achieved {got:.6f} "
        f"> brute-force minimum {truth:.6f}"
    )


def test_corner_extremality_hit_ratio_grid():
    legs, pnl, col_map = HET
    _, result = _run_optimizer(legs, pnl, col_map, CONS_SMALL, {"hit_ratio": 1.0})
    truth = _brute_force_best("hit_ratio", legs, pnl, 2, 3)
    got = _achieved("hit_ratio", result, legs, pnl, col_map)
    # Grid truth is itself approximate → symmetric tolerance of 2 grid points
    assert got >= truth - 2.0 / N_DAYS - 1e-9, (
        f"hit_ratio extremality violated: achieved {got:.4f} vs grid best {truth:.4f}"
    )


# ---------------------------------------------------------------------------
# Tests — homogeneous ~20-stock universe (production scenario)
# ---------------------------------------------------------------------------

HOMO = _make_universe(n_stocks=20, heterogeneous=False, seed=11)
CONS_HOMO = _constraints(min_l=3, max_l=4, time_s=20.0)


@pytest.mark.parametrize("metric", ["last_carry", "min_payoff"])
def test_corner_extremality_homogeneous(metric):
    """Near-identical stocks: the case where a proxy ranking is least reliable."""
    legs, pnl, col_map = HOMO
    _, result = _run_optimizer(legs, pnl, col_map, CONS_HOMO, {metric: 1.0})
    truth = _brute_force_best(metric, legs, pnl, 3, 4)
    got = _achieved(metric, result, legs, pnl, col_map)
    # Small slack: C(20,3)+C(20,4) subsets — GA must land at (or within noise of)
    # the true optimum; a material gap means selection lost the argmax subset.
    slack = 0.02 * max(1.0, abs(truth))
    assert got >= truth - slack, (
        f"[homogeneous] {metric}: achieved {got:.6f} < optimum {truth:.6f} "
        f"(gap {truth - got:.6f} > slack {slack:.6f})"
    )


# ---------------------------------------------------------------------------
# Forced names
# ---------------------------------------------------------------------------


def test_forced_names_always_present():
    legs, pnl, col_map = HET
    # Force the WORST stock (T0 has the lowest mean by construction) — the
    # optimizer would never pick it on merit, which is exactly the point.
    _, result = _run_optimizer(
        legs, pnl, col_map, CONS_SMALL, {"mean_payoff": 1.0}, forced=[0]
    )
    keys = [k for k, _ in result.long_basket]
    assert "T0" in keys, f"forced ticker T0 missing from basket {keys}"
    w0 = dict(result.long_basket)["T0"]
    assert legs[0].min_weight - 1e-6 <= w0 <= legs[0].max_weight + 1e-6


def test_forced_infeasible_raises():
    legs, pnl, col_map = HET
    with pytest.raises(ValueError, match="forced"):
        DispersionOptimizer(
            long_candidates=legs,
            short_candidates=[],
            pnl_matrix=pnl,
            column_map=col_map,
            constraints=_constraints(2, 3),
            missing_data_policy=MissingDataPolicy.FILL_ZERO,
            metric_weights=MetricWeights({"mean_payoff": 1.0}),
            forced_long_indices=[0, 1, 2, 3],  # 4 forced > max_stocks_long=3
        )


# ---------------------------------------------------------------------------
# 2-start weight diagnostic (equal vs greedy start must agree on score)
# ---------------------------------------------------------------------------


def test_two_start_diagnostic_converges():
    legs, pnl, col_map = HET
    opt, result = _run_optimizer(
        legs, pnl, col_map, CONS_SMALL,
        {"mean_payoff": 0.4, "hit_ratio": 0.3, "min_payoff": 0.3},
    )
    _, _, cols, strikes = _basket_arrays(result, col_map, legs)
    diag = opt._weight_solver.diagnose_convergence(
        pnl_matrix=opt._ts_mat,
        stock_indices=np.array(cols, dtype=int),
        strikes=strikes,
        tol=0.02,
    )
    assert diag["converged"], (
        f"2-start diagnostic failed: equal-start score {diag['score_equal_start']:.4f} "
        f"vs greedy-start score {diag['score_greedy_start']:.4f} "
        f"(gap {diag['score_gap']:.4f}) — solver likely trapped on a plateau"
    )
