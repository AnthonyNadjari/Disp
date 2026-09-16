"""Carry criterion window ("last N months"): date-index counting, plumbing
into the score function, and bundle persistence."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions.dispersion.scoring.score import carry_window_obs, make_default_score_function  # noqa: E402
from functions.dispersion.scoring import MetricWeights  # noqa: E402
from functions.dispersion.scoring.weight_solver import concave_blend_lambdas  # noqa: E402


def test_carry_window_counts_real_dates():
    d = pd.bdate_range("2025-01-01", "2026-09-15")
    k3 = carry_window_obs(d, 3)
    assert 60 <= k3 <= 68                      # ~3 months of business days
    assert carry_window_obs(d, 1) < k3 < carry_window_obs(d, 6)
    assert carry_window_obs(d, 0) == 1        # 0 months = most recent observation
    assert carry_window_obs(None, 3) == 1
    assert carry_window_obs([], 3) == 1
    assert carry_window_obs(list(d)[:5], 24) == 5   # window longer than history → all


def test_carry_window_respects_gaps():
    # weekly grid: 3 months ≈ 13 observations, not 63
    d = pd.date_range("2026-01-01", periods=52, freq="W")
    assert 12 <= carry_window_obs(d, 3) <= 14


def test_score_function_and_solver_share_the_window():
    sf = make_default_score_function(weights=MetricWeights({"last_carry": 0.5, "mean_payoff": 0.5}),
                                     last_carry_k=17)
    lc = next(m for m in sf.metrics if m.name == "last_carry")
    assert lc._k == 17
    _, _, lam_carry, k_carry = concave_blend_lambdas(sf)
    assert (lam_carry, k_carry) == (0.5, 17)      # solver reads the same k
    pnl = np.arange(40, dtype=float)
    from functions.dispersion.scoring.metrics import ScoreContext
    assert lc.compute(pnl, ScoreContext(n_days=40)) == pytest.approx(np.mean(pnl[-17:]))


def test_bundle_persists_last_carry_k(tmp_path):
    from functions.dispersion.run_bundle import save_run_bundle, load_run_bundle
    from functions.dispersion.models import DispersionLeg, OptimizationConstraints, MissingDataPolicy
    legs = [DispersionLeg(variance_asset=f"S{i}", strike_mono_var_swap=0.20, min_weight=0.0, max_weight=1.0)
            for i in range(4)]
    pnl = np.random.default_rng(0).normal(size=(30, 4))
    path = str(tmp_path / "b")
    save_run_bundle(path, pnl_matrix=pnl, column_map={f"S{i}": i for i in range(4)},
                    long_candidates=legs, short_candidates=[],
                    constraints=OptimizationConstraints(min_stocks_long=2, max_stocks_long=3),
                    score_weights={"last_carry": 1.0}, seed=1,
                    missing_data_policy=MissingDataPolicy.FILL_ZERO, last_carry_k=9)
    assert load_run_bundle(path).last_carry_k == 9
