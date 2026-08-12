"""Phase 4c — absolute-Vega toggle / axe recycling.

OFF = strictly historical behaviour (gated globally by the golden suite; the
iso test here proves the stronger property that a P&L-only config returns the
IDENTICAL basket with the toggle ON).  ON = corner extremality of criterion A
against an independent LP ground truth written in this file.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.optimize import linprog

from functions.dispersion.models import (
    DispersionLeg,
    MissingDataPolicy,
    OptimizationConstraints,
    VegaConfig,
)
from functions.dispersion._optimizer import DispersionOptimizer
from functions.dispersion.scoring import MetricWeights

N_DAYS = 250
N = 8
LO, HI = 0.05, 0.60
V_MIN, V_MAX = 50.0, 200.0

TARGETS = np.array([100.0, 80.0, 60.0, 40.0, 20.0, 0.0, 0.0, 0.0])
CAPS = np.array([110.0, 90.0, 70.0, 50.0, 30.0, np.inf, np.inf, np.inf])
T_TOTAL = float(TARGETS.sum())


def _universe(seed=51, targets=TARGETS, caps=CAPS):
    rng = np.random.default_rng(seed)
    names = [f"V{i}" for i in range(N)]
    means = np.linspace(0.2, 1.0, N)
    pnl = np.column_stack([rng.normal(means[i], 0.9, N_DAYS) for i in range(N)])
    legs = [
        DispersionLeg(variance_asset=names[i], strike_mono_var_swap=0.10 + 0.01 * i,
                      min_weight=LO, max_weight=HI,
                      axe_target=(None if targets[i] == 0 else float(targets[i])),
                      axe_cap=(None if not np.isfinite(caps[i]) else float(caps[i])))
        for i in range(N)
    ]
    col_map = {names[i]: i for i in range(N)}
    return legs, pnl, col_map


def _cons():
    return OptimizationConstraints(
        min_stocks_long=2, max_stocks_long=3,
        min_stocks_short=0, max_stocks_short=0,
        max_net_strike=10.0, population_size=40,
        max_generations=120, time_limit_seconds=20.0,
        stagnation_limit=20,
    )


def _run(weights, vega=None, seed=0, universe=None):
    legs, pnl, col_map = universe or _universe()
    opt = DispersionOptimizer(
        long_candidates=legs, short_candidates=[],
        pnl_matrix=pnl, column_map=col_map,
        constraints=_cons(),
        missing_data_policy=MissingDataPolicy.FILL_ZERO,
        metric_weights=MetricWeights(weights),
        seed=seed,
        vega_config=vega,
    )
    result = opt.run()
    assert result.long_basket, "empty basket"
    return legs, pnl, col_map, result


# ---------------------------------------------------------------------------
# Independent ground truth: exact axe LP per subset (test-local formulation)
# ---------------------------------------------------------------------------


def _axe_lp_truth(subset):
    """Max A over (v, V) for a subset: independent linprog formulation."""
    n = len(subset)
    t = TARGETS[list(subset)]
    cp = CAPS[list(subset)]
    c = np.zeros(2 * n + 1)
    c[n:2 * n] = -1.0  # max Σ r
    bounds = ([(0.0, None if not np.isfinite(cp[i]) else float(cp[i])) for i in range(n)]
              + [(0.0, float(t[i])) for i in range(n)]
              + [(V_MIN, V_MAX)])
    rows = []
    rhs = []
    eye = np.eye(n)
    blk = np.zeros((n, 2 * n + 1))          # r - v <= 0
    blk[:, :n] = -eye
    blk[:, n:2 * n] = eye
    rows.append(blk)
    rhs.append(np.zeros(n))
    blk = np.zeros((n, 2 * n + 1))          # v - HI·V <= 0
    blk[:, :n] = eye
    blk[:, -1] = -HI
    rows.append(blk)
    rhs.append(np.zeros(n))
    blk = np.zeros((n, 2 * n + 1))          # LO·V - v <= 0
    blk[:, :n] = -eye
    blk[:, -1] = LO
    rows.append(blk)
    rhs.append(np.zeros(n))
    A_eq = np.zeros((1, 2 * n + 1))
    A_eq[0, :n] = 1.0
    A_eq[0, -1] = -1.0
    res = linprog(c, A_ub=np.vstack(rows), b_ub=np.concatenate(rhs),
                  A_eq=A_eq, b_eq=[0.0], bounds=bounds, method="highs")
    if not res.success:
        return None
    return float(res.x[n:2 * n].sum()) / T_TOTAL


def test_vega_axe_corner_vs_lp_truth():
    """weight=1 on axe_book_cleaned ⇒ the returned basket cleans as much of
    the axe book as the best (subset, v, V) found by exhaustive LP."""
    legs, pnl, col_map, result = _run(
        {"axe_book_cleaned": 1.0}, vega=VegaConfig(v_min=V_MIN, v_max=V_MAX))

    truth = max(
        v for size in (2, 3)
        for subset in itertools.combinations(range(N), size)
        for v in [_axe_lp_truth(subset)]
        if v is not None
    )
    assert result.axe_cleaned is not None
    assert result.axe_cleaned >= truth - 1e-4, (
        f"axe corner extremality violated: cleaned {result.axe_cleaned:.6f} "
        f"< LP optimum {truth:.6f}")

    # Delivered allocation honours every hard constraint
    assert result.total_vega is not None
    assert V_MIN - 1e-6 <= result.total_vega <= V_MAX + 1e-6
    vb = dict(result.vega_basket)
    assert abs(sum(vb.values()) - result.total_vega) <= 1e-6 * max(1.0, result.total_vega)
    for k, v in vb.items():
        i = col_map[k]
        assert v <= CAPS[i] + 1e-6, f"{k}: vega {v:.2f} > cap {CAPS[i]:.2f}"
        w_i = v / result.total_vega
        assert LO - 1e-6 <= w_i <= HI + 1e-6


def test_vega_pnl_only_same_basket_as_off_plus_V():
    """P&L-only config: the toggle must NOT change the basket (P&L is the
    weight series, V-independent); it only attaches the deterministic V
    (max-clean rule) and the axe fractions."""
    legs, pnl, col_map, r_off = _run({"mean_payoff": 1.0}, vega=None)
    _, _, _, r_on = _run({"mean_payoff": 1.0}, vega=VegaConfig(v_min=V_MIN, v_max=V_MAX))

    assert r_off.long_basket == r_on.long_basket, "vega toggle changed a P&L-only basket"
    assert r_off.score == r_on.score
    assert r_off.total_vega is None
    assert r_on.total_vega is not None
    # Max-clean rule: V = min(V_MAX, min cap_i / w_i)
    w = {k: wt for k, wt in r_on.long_basket}
    exp_v = min(V_MAX, min(CAPS[col_map[k]] / wt for k, wt in w.items()))
    assert r_on.total_vega == pytest.approx(max(exp_v, V_MIN), abs=1e-6)


def test_vega_package_recycled_direction():
    """weight=1 on axe_package_recycled ⇒ near-total recycling is reachable
    when every selectable name carries a big axe (small V, v_i <= target)."""
    # Varied targets: half big axes, half small — keeps the reference for
    # criterion B non-degenerate (a constant reference is refused by design)
    targets = np.where(np.arange(N) % 2 == 0, 500.0, 30.0)
    caps = np.full(N, np.inf)
    universe = _universe(seed=52, targets=targets, caps=caps)
    _, _, _, result = _run({"axe_package_recycled": 1.0},
                           vega=VegaConfig(v_min=V_MIN, v_max=V_MAX),
                           universe=universe)
    assert result.axe_recycled is not None
    assert result.axe_recycled >= 0.99, (
        f"package recycling should reach ~100%, got {result.axe_recycled:.4f}")


def test_vega_config_time_validations():
    legs, pnl, col_map = _universe()
    with pytest.raises(ValueError, match="v_max"):
        VegaConfig(v_min=100.0, v_max=50.0)
    with pytest.raises(ValueError, match="v_min"):
        VegaConfig(v_min=0.0, v_max=50.0)

    # b_min·V_min exceeding a name's cap must fail loudly at construction
    legs_bad, pnl_b, col_map_b = _universe()
    legs_bad[0].axe_cap = 1.0  # min_weight 0.05 × V_min 50 = 2.5 > cap 1.0
    with pytest.raises(ValueError, match="Min Weight"):
        DispersionOptimizer(
            long_candidates=legs_bad, short_candidates=[],
            pnl_matrix=pnl_b, column_map=col_map_b,
            constraints=_cons(),
            missing_data_policy=MissingDataPolicy.FILL_ZERO,
            metric_weights=MetricWeights({"mean_payoff": 1.0}),
            vega_config=VegaConfig(v_min=V_MIN, v_max=V_MAX),
        )

    # Axe metrics without the toggle must refuse clearly
    with pytest.raises(RuntimeError, match="Vega toggle"):
        _run({"axe_book_cleaned": 1.0}, vega=None)

    # Axe metrics with the toggle but no targets anywhere must refuse
    no_axe = _universe(seed=53, targets=np.zeros(N), caps=np.full(N, np.inf))
    with pytest.raises(RuntimeError, match="Axe Target"):
        _run({"axe_book_cleaned": 1.0},
             vega=VegaConfig(v_min=V_MIN, v_max=V_MAX), universe=no_axe)


def test_vega_bundle_roundtrip(tmp_path):
    from functions.dispersion.run_bundle import load_run_bundle, save_run_bundle

    legs, pnl, col_map, result = _run(
        {"axe_book_cleaned": 1.0}, vega=VegaConfig(v_min=V_MIN, v_max=V_MAX))
    path = str(tmp_path / "bundle_vega")
    save_run_bundle(
        path, pnl_matrix=pnl, column_map=col_map,
        long_candidates=legs, short_candidates=[],
        constraints=_cons(), score_weights={"axe_book_cleaned": 1.0}, seed=0,
        missing_data_policy=MissingDataPolicy.FILL_ZERO,
        vega_config=VegaConfig(v_min=V_MIN, v_max=V_MAX),
    )
    bundle = load_run_bundle(path)
    assert bundle.vega_config is not None
    assert bundle.long_candidates[0].axe_target == legs[0].axe_target
    replay = bundle.replay()
    assert replay.long_basket == result.long_basket
    assert replay.score == result.score
    assert replay.total_vega == result.total_vega
    assert replay.axe_cleaned == result.axe_cleaned
