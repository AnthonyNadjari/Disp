"""Phase 4a — regional/sector bucket constraints (count + weight), E2E with
independent LP ground truths.

Universe: 8 stocks, buckets US = {B0..B3}, EU = {B4..B7}.  EU names are made
systematically better so the constraints genuinely bind.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.optimize import linprog

from functions.dispersion.models import (
    BucketConstraint,
    DispersionLeg,
    MissingDataPolicy,
    OptimizationConstraints,
)
from functions.dispersion._optimizer import DispersionOptimizer
from functions.dispersion.scoring import MetricWeights

N_DAYS = 250
N = 8
LO, HI = 0.05, 0.60


def _universe(seed=31):
    rng = np.random.default_rng(seed)
    names = [f"B{i}" for i in range(N)]
    means = np.concatenate([np.linspace(-0.2, 0.4, 4),    # US: mediocre
                            np.linspace(0.8, 1.6, 4)])    # EU: strong
    pnl = np.column_stack([rng.normal(means[i], 0.9, N_DAYS) for i in range(N)])
    legs = [
        DispersionLeg(variance_asset=names[i], strike_mono_var_swap=0.10 + 0.01 * i,
                      min_weight=LO, max_weight=HI,
                      sector="US" if i < 4 else "EU")
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


BUCKETS = [
    BucketConstraint(bucket="US", min_names=1, max_weight=0.35),
    BucketConstraint(bucket="EU", min_names=1, min_weight=0.30),
]


def _bucket_of(i):
    return "US" if i < 4 else "EU"


def _counts_ok(subset):
    us = sum(1 for i in subset if _bucket_of(i) == "US")
    eu = len(subset) - us
    return us >= 1 and eu >= 1


def _group_rows(subset):
    """LP rows: US weight <= 0.35 ; EU weight >= 0.30 (as -sum <= -0.30)."""
    n = len(subset)
    rows, rhs = [], []
    us_pos = [k for k, i in enumerate(subset) if _bucket_of(i) == "US"]
    eu_pos = [k for k, i in enumerate(subset) if _bucket_of(i) == "EU"]
    if us_pos:
        r = np.zeros(n)
        r[us_pos] = 1.0
        rows.append(r)
        rhs.append(0.35)
    if eu_pos:
        r = np.zeros(n)
        r[eu_pos] = -1.0
        rows.append(r)
        rhs.append(-0.30)
    return np.vstack(rows), np.array(rhs)


def _lp_max_mean(sub_pnl, subset):
    n = len(subset)
    A_ub, b_ub = _group_rows(subset)
    res = linprog(-sub_pnl.mean(axis=0), A_ub=A_ub, b_ub=b_ub,
                  A_eq=np.ones((1, n)), b_eq=[1.0],
                  bounds=[(LO, HI)] * n, method="highs")
    if not res.success:
        return None
    return float(sub_pnl.mean(axis=0) @ res.x)


def _lp_maximin(sub_pnl, subset):
    d, n = sub_pnl.shape
    A_g, b_g = _group_rows(subset)
    A_ub = np.vstack([np.hstack([-sub_pnl, np.ones((d, 1))]),
                      np.hstack([A_g, np.zeros((A_g.shape[0], 1))])])
    b_ub = np.concatenate([np.zeros(d), b_g])
    c = np.zeros(n + 1)
    c[-1] = -1.0
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                  bounds=[(LO, HI)] * n + [(None, None)], method="highs")
    if not res.success:
        return None
    return float(res.x[-1])


def _run(weights, seed=0):
    legs, pnl, col_map = _universe()
    opt = DispersionOptimizer(
        long_candidates=legs, short_candidates=[],
        pnl_matrix=pnl, column_map=col_map,
        constraints=_cons(),
        missing_data_policy=MissingDataPolicy.FILL_ZERO,
        metric_weights=MetricWeights(weights),
        seed=seed,
        bucket_constraints=[BucketConstraint(**{
            k: getattr(bc, k) for k in
            ("bucket", "min_names", "max_names", "min_weight", "max_weight")})
            for bc in BUCKETS],
    )
    result = opt.run()
    assert result.long_basket, "empty basket under bucket constraints"
    return legs, pnl, col_map, result


def _check_bucket_satisfaction(result, col_map):
    keys = [k for k, _ in result.long_basket]
    w = np.array([wt for _, wt in result.long_basket])
    cols = [col_map[k] for k in keys]
    us_w = sum(wt for c, wt in zip(cols, w) if _bucket_of(c) == "US")
    eu_w = sum(wt for c, wt in zip(cols, w) if _bucket_of(c) == "EU")
    us_n = sum(1 for c in cols if _bucket_of(c) == "US")
    eu_n = sum(1 for c in cols if _bucket_of(c) == "EU")
    assert us_n >= 1 and eu_n >= 1, f"count bounds violated: US={us_n} EU={eu_n}"
    assert us_w <= 0.35 + 1e-4, f"US weight cap violated: {us_w:.4f} > 0.35"
    assert eu_w >= 0.30 - 1e-4, f"EU weight floor violated: {eu_w:.4f} < 0.30"
    return cols, w


def test_bucket_corner_mean_payoff_vs_lp_truth():
    legs, pnl, col_map, result = _run({"mean_payoff": 1.0})
    cols, w = _check_bucket_satisfaction(result, col_map)
    truth = max(
        v for size in (2, 3)
        for subset in itertools.combinations(range(N), size)
        if _counts_ok(subset)
        for v in [_lp_max_mean(pnl[:, list(subset)], subset)]
        if v is not None
    )
    achieved = float(pnl[:, cols].mean(axis=0) @ w)
    assert achieved >= truth - 1e-4, (
        f"bucket-constrained mean extremality violated: {achieved:.6f} < {truth:.6f}")


def test_bucket_corner_min_payoff_vs_lp_truth():
    legs, pnl, col_map, result = _run({"min_payoff": 1.0})
    cols, w = _check_bucket_satisfaction(result, col_map)
    truth = max(
        v for size in (2, 3)
        for subset in itertools.combinations(range(N), size)
        if _counts_ok(subset)
        for v in [_lp_maximin(pnl[:, list(subset)], subset)]
        if v is not None
    )
    achieved = float((pnl[:, cols] @ w).min())
    assert achieved >= truth - 1e-3, (
        f"bucket-constrained maximin extremality violated: {achieved:.6f} < {truth:.6f}")


def test_bucket_nonexact_config_satisfies_constraints():
    legs, pnl, col_map, result = _run({"mean_payoff": 0.5, "hit_ratio": 0.5})
    _check_bucket_satisfaction(result, col_map)


def test_bucket_config_time_validations():
    legs, pnl, col_map = _universe()

    def _mk(bcs, cons=None):
        return DispersionOptimizer(
            long_candidates=legs, short_candidates=[],
            pnl_matrix=pnl, column_map=col_map,
            constraints=cons or _cons(),
            missing_data_policy=MissingDataPolicy.FILL_ZERO,
            metric_weights=MetricWeights({"mean_payoff": 1.0}),
            bucket_constraints=bcs,
        )

    with pytest.raises(ValueError, match="available candidate"):
        _mk([BucketConstraint(bucket="US", min_names=5)])
    with pytest.raises(ValueError, match="no candidates"):
        _mk([BucketConstraint(bucket="ASIA", min_names=1)])
    with pytest.raises(ValueError, match="min_names >= 1"):
        _mk([BucketConstraint(bucket="US", min_weight=0.2)])
    with pytest.raises(ValueError, match="floors"):
        _mk([BucketConstraint(bucket="US", min_names=1, min_weight=0.6),
             BucketConstraint(bucket="EU", min_names=1, min_weight=0.6)])
    with pytest.raises(ValueError, match="max_stocks_long"):
        _mk([BucketConstraint(bucket="US", min_names=2),
             BucketConstraint(bucket="EU", min_names=2)])
    # BucketConstraint's own validation
    with pytest.raises(ValueError, match="max_names"):
        BucketConstraint(bucket="US", min_names=3, max_names=1)
    with pytest.raises(ValueError, match="DECIMAL"):
        BucketConstraint(bucket="US", min_weight=30.0)  # percent, not decimal


def test_bucket_bundle_roundtrip(tmp_path):
    from functions.dispersion.run_bundle import load_run_bundle, save_run_bundle

    legs, pnl, col_map, result = _run({"mean_payoff": 1.0})
    path = str(tmp_path / "bundle_buckets")
    save_run_bundle(
        path, pnl_matrix=pnl, column_map=col_map,
        long_candidates=legs, short_candidates=[],
        constraints=_cons(), score_weights={"mean_payoff": 1.0}, seed=0,
        missing_data_policy=MissingDataPolicy.FILL_ZERO,
        bucket_constraints=BUCKETS,
    )
    bundle = load_run_bundle(path)
    assert bundle.bucket_constraints is not None
    assert {bc.bucket for bc in bundle.bucket_constraints} == {"US", "EU"}
    replay = bundle.replay()
    assert replay.long_basket == result.long_basket
    assert replay.score == result.score
