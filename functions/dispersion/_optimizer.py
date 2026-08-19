"""
Dispersion Optimizer — Genetic Algorithm for basket selection.
Faithful port of the original Gaia_PP StreamlinedOptimizer with the same:
  - Fitness function (normalized metrics, date-based 3Y/5Y windows)
  - Quality-biased weight allocation
  - Repair-based crossover/mutation
  - Tournament selection + elitism
The optimizer uses fillna(0) on the P&L matrix (penalizes missing data).
The backtester separately handles adaptive reweight for reported results.
Usage:
    from functions.dispersion._optimizer import DispersionOptimizer
    optimizer = DispersionOptimizer(long_candidates, short_candidates,
                                    pnl_matrix, column_map, constraints,
                                    metric_weights=MetricWeights({...}))
    result = optimizer.run()
"""
from __future__ import annotations
import itertools
import math
import random
import time
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from functions.dispersion._logging import logger as _engine_log

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Engine tuning constants (single home for every GA-side magic number)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass(frozen=True)
class _TuningConstants:
    """Every tunable magic number of the GA engine, in one place.

    Defaults are STRICTLY the historical values.  Numerical tolerances
    (1e-8 / 1e-9 / 1e-12 feasibility epsilons) are correctness guards, not
    tuning knobs, and deliberately stay inline.  Solver-side numbers live in
    scoring.weight_solver._SolverTuning (the solver cannot import from here).
    """

    # ── Scoring / ranking ──
    #: Quantile score above which the GA switches to the raw tie-break.
    #: Higher → tie-break rarer (top-percentile baskets rank equal);
    #: lower → raw blend dominates earlier. Range [0.9, 0.999].
    tiebreak_threshold: float = 0.99
    #: Reference sample size, no tail metric active.  More = finer quantile
    #: resolution, slower setup. Range [100, 2000].
    n_reference_base: int = 300
    #: Reference sample size when a tail metric (min_payoff/cvar_5) is
    #: active — tails need more mass. Range [300, 3000].
    n_reference_tail: int = 800
    #: Minimum fitted reference size before the run aborts. Below this the
    #: quantile normalizer is meaningless. Range [50, 300].
    min_reference_size: int = 100
    #: Reference build gives up after n_samples × this many attempts.
    #: Higher tolerates more rejected baskets (slow data). Range [2, 10].
    reference_attempts_factor: int = 3
    #: Elite baskets injected into the reference: min(this, n_samples // 10).
    #: More = normalizer ceiling closer to the GA's elite. Range [10, 100].
    reference_elite_cap: int = 50

    # ── Data validity ──
    #: Minimum non-zero P&L observations for a basket to be scoreable, and
    #: minimum valid rows in the drop-incomplete path.  Higher = stricter
    #: history requirement, more rejections. Range [20, 150].
    min_valid_days: int = 50

    # ── Population / evolution ──
    #: Population init and per-generation fill give up after size × this
    #: many attempts. Range [2, 10].
    population_attempts_factor: int = 3
    #: Attempts to build one random individual before giving up. Range [5, 50].
    create_attempts: int = 20
    #: Random immigrants per generation: max(immigrants_min, size × frac).
    #: More = diversity, slower convergence. Ranges [0, 20] / [0.0, 0.3].
    immigrants_min: int = 4
    immigrants_frac: float = 0.10
    #: Population floor after crossover failures: max(min_abs, size × frac).
    pop_floor_abs: int = 30
    pop_floor_frac: float = 0.6
    #: Backfill attempts factor over the missing count. Range [1, 5].
    pop_backfill_factor: int = 2
    #: Consecutive infeasible exact solves before aborting with a units
    #: error (fail-fast on mis-scaled strikes). Range [5, 100].
    infeasible_streak_limit: int = 20

    # ── Post-GA refinement / local search ──
    #: Elites refined by the inner solver after the GA. More = better
    #: subsets found, slower. Range [5, 100].
    refine_top_k: int = 30
    #: Local search wall budget: min(cap_s, max(floor_s, frac × time_limit)).
    local_search_cap_s: float = 8.0
    local_search_floor_s: float = 2.0
    local_search_frac: float = 0.4
    #: Best-improvement sweeps per descent. Range [2, 20].
    local_search_sweeps: int = 6
    #: Multi-start seeds (incumbent + this many distinct elites). Range [0, 10].
    local_search_seeds: int = 3
    #: Exhaustive-search ceiling: when the feasible long-subset count is <=
    #: this, the post-GA local search enumerates ALL subsets (each an exact
    #: solve, a few ms) and returns the true argmax — corner extremality
    #: becomes a seed-INDEPENDENT GUARANTEE for universes up to this size
    #: (covers realistic small candidate pools and every golden).  The count
    #: bound is the safety: no time check, so the enumeration always completes.
    #: Sized for ~a handful of seconds post-GA; raising it extends the
    #: guarantee to bigger universes at a per-subset solve cost.  Above the
    #: ceiling, no polynomial method can guarantee the global optimum, so the
    #: heuristic descent + escape + random restarts runs instead (best-effort).
    #: Range [0, 20000].
    local_search_exhaustive_max_subsets: int = 2000
    #: Random-restart diversification budget for the heuristic regime (large
    #: universes, above the exhaustive ceiling): draw this many random
    #: feasible subsets, 1-swap-descend each, keep the best — escapes the
    #: shared basin the GA + elite starts fall into on near-homogeneous
    #: universes.  Deterministic (run RNG).  Early-stops on the time budget.
    #: 0 disables. Range [0, 2000].
    local_search_random_restarts: int = 400
    #: Safety-net score tie window — below it the raw min decides. Range
    #: [1e-6, 1e-2].
    safety_tie_window: float = 1e-4

    # ── Reference weight-strategy mix (cumulative draw thresholds) ──
    #: P(equal weights) = mix_equal; P(QP-diversified) = mix_qp − mix_equal;
    #: P(greedy-spread) = mix_greedy − mix_qp; remainder = Dirichlet.
    mix_equal: float = 0.4
    mix_qp: float = 0.7
    mix_greedy: float = 0.9
    #: Greedy reference allocator: best name gets lb + frac × (ub − lb).
    greedy_best_frac: float = 0.6
    #: Greedy remainder spread multiplier over quality share. Range [1, 3].
    greedy_spread_boost: float = 1.5
    #: QP diversification penalty λ (0 = LP corners, big = equal-weight).
    qp_diversification_penalty: float = 0.3

    # ── Bootstrap robustness diagnostic ──
    bootstrap_draws: int = 300
    bootstrap_top_k: int = 10
    #: Final-population snapshot kept as challenger source. Range [10, 200].
    bootstrap_population_snapshot: int = 60


#: Module-level defaults (import and share; construct a custom instance only
#: for experiments — the engine reads THIS object).
TUNING = _TuningConstants()

def _scoring_signature(metric_weights, seed: int, n_samples: int,
                       reference_size: int) -> str:
    """Reproducibility fingerprint: sha256[:16] over the scoring inputs.

    Two runs with the same signature scored baskets on the same objective
    (active metrics + weights), the same seed and the same reference-sample
    geometry — their scores are directly comparable.
    """
    import hashlib
    import json
    payload = {
        "metrics": {k: round(float(v), 12)
                    for k, v in sorted(metric_weights.items()) if v > 0},
        "seed": int(seed),
        "n_samples": int(n_samples),
        "reference_size": int(reference_size),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _candidate_key(leg: DispersionLeg, is_cross_corridor: bool) -> str:
    """Return the series key for PnL matrix lookup.
    For cross-corridor: use DispersionLeg.corridor_condition_asset (Corridor Condition Asset)
    For mono: use DispersionLeg.variance_asset (Variance Asset)
    """
    if is_cross_corridor and leg.corridor_condition_asset:
        return leg.corridor_condition_asset
    return leg.variance_asset
from functions.dispersion.models import (
    Basket,
    BacktestResult,
    BasketInput,
    BucketConstraint,
    DispersionLeg,
    MissingDataPolicy,
    OptimizationConstraints,
    OptimizationResult,
    ProductType,
    SwapConfig,
    VegaConfig,
)
from functions.dispersion._backtester import (
    DispersionDataLoader,
    DispersionBacktester,
    SwapCalculator,
    _rolling_pnl_volswap,
    _rolling_pnl_corridor,
)
from functions.dispersion.scoring import (
    MetricWeights, ScoreFunction, ScoreContext, WeightSolver,
    WeightConstraints, make_default_score_function,
)
from functions.dispersion.scoring.weight_solver import (
    VegaSpec,
    adaptive_pnl,
    active_mask_with_grace,
    carry_pnl_within_grace,
    concave_blend_lambdas,
    concave_blend_value,
    project_to_bounded_simplex,
)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Internal representation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _Individual:
    """Internal GA solution representation.

    The GENOME is the two index lists only.  The weight arrays are a CACHE
    of the deterministic per-subset weights derived in ``_fitness`` (exact
    inner solver for exact-path configs, bounded equal-weight projection
    otherwise) — they are never evolved by the GA operators.
    """
    __slots__ = ("long_indices", "short_indices", "long_weights", "short_weights", "fitness")

    def __init__(self, long_indices, short_indices, long_weights=None, short_weights=None):
        self.long_indices = list(long_indices)
        self.short_indices = list(short_indices)
        self.long_weights = (np.asarray(long_weights, dtype=np.float64)
                             if long_weights is not None else np.zeros(0, dtype=np.float64))
        self.short_weights = (np.asarray(short_weights, dtype=np.float64)
                              if short_weights is not None else np.zeros(0, dtype=np.float64))
        self.fitness = -np.inf
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Weight optimization helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _qp_diversified_weights(col_means: np.ndarray, lb: np.ndarray, ub: np.ndarray,
                            diversification_penalty: float = TUNING.qp_diversification_penalty) -> np.ndarray:
    """
    Find weights that maximize expected return MINUS a diversification penalty.

    Objective: max  Σ(mean_i * w_i) - λ * Σ(w_i²)
    Subject to: Σ w_i = 1, lb_i ≤ w_i ≤ ub_i

    The λ penalty discourages corner solutions (all-in on one stock).
    With λ=0 this degenerates to LP (always corners). With large λ it approaches equal-weight.
    """
    from scipy.optimize import minimize as _minimize
    n = len(col_means)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Normalize means to [0,1] range to make penalty scale-independent
    mean_range = col_means.max() - col_means.min()
    if mean_range > 1e-12:
        norm_means = (col_means - col_means.min()) / mean_range
    else:
        norm_means = np.ones(n) / n

    def objective(w):
        # Negative because minimize
        return -(norm_means @ w - diversification_penalty * np.sum(w ** 2))

    def jac(w):
        return -(norm_means - 2 * diversification_penalty * w)

    x0 = np.full(n, 1.0 / n)
    x0 = np.clip(x0, lb, ub)
    s = x0.sum()
    if s > 0:
        x0 = x0 / s

    bounds = [(float(lb[i]), float(ub[i])) for i in range(n)]
    constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1.0, 'jac': lambda w: np.ones(n)}]

    res = _minimize(objective, x0, jac=jac, method='SLSQP', bounds=bounds, constraints=constraints,
                    options={'maxiter': 100, 'ftol': 1e-10})
    if res.success:
        w = np.clip(res.x, lb, ub)
        s = w.sum()
        if s > 0:
            w = w / s
        return w
    else:
        # Fallback: equal weight within bounds
        w = np.clip(np.full(n, 1.0 / n), lb, ub)
        s = w.sum()
        if s > 0:
            w = w / s
        return w

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Genetic Optimizer (port of original StreamlinedOptimizer)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DispersionOptimizer:
    """
    Genetic optimizer for long/short basket construction.
    Direct port of the original Gaia_PP StreamlinedOptimizer.
    Maximizes weighted fitness (hit ratio, returns, drawdown) subject to:
      - Min/max stocks per leg
      - Min/max weight per stock
      - Max net strike
    """
    def __init__(
        self,
        long_candidates: List[DispersionLeg],
        short_candidates: List[DispersionLeg],
        pnl_matrix: np.ndarray,
        column_map: Dict[str, int],
        constraints: Optional[OptimizationConstraints] = None,
        logger: Optional[Callable[[str, str], None]] = None,
        missing_data_policy=None,
        adj_divs: bool = False,
        reweight_grace_days: int = 0,
        active_mask: Optional[np.ndarray] = None,
        is_cross_corridor: bool = False,
        seed: int = 0,
        global_cap: float = 9999999.0,
        global_floor: float = -9999999.0,
        metric_weights: Optional[MetricWeights] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        bisect_in_ga: bool = False,
        smooth_weights: bool = False,
        smooth_eps: float = 0.05,
        forced_long_indices: Optional[List[int]] = None,
        n_reference_samples: Optional[int] = None,
        bucket_constraints: Optional[List[BucketConstraint]] = None,
        vega_config: Optional[VegaConfig] = None,
        reference_cache: Optional[dict] = None,
    ):
        self.long_candidates = long_candidates
        self.short_candidates = short_candidates
        self.c = constraints or OptimizationConstraints()
        self._use_exact_in_ga: bool = bisect_in_ga
        self.log = logger or (lambda l, m: None)
        self._progress_cb = progress_callback
        self.missing_data_policy = missing_data_policy
        self._smooth_weights = smooth_weights
        self._smooth_eps = smooth_eps
        # Reference-sample size override (None = adaptive default 300/800).
        # The normalizer needs >= 100 fitted baskets to be meaningful.
        if n_reference_samples is not None and int(n_reference_samples) < 100:
            raise ValueError(
                f"n_reference_samples must be >= 100 (quantile normalizer needs "
                f"a meaningful reference), got {n_reference_samples}."
            )
        self._n_reference_samples = (int(n_reference_samples)
                                     if n_reference_samples is not None else None)
        # Shared-calibration cache (optimize_multi / cache_prep re-runs): maps
        # (seed, n_samples, extras signature, vega on) → the generated reference
        # sample + post-generation RNG states. Same universe is the caller's
        # contract (one prep = one cache). None = no sharing.
        self._reference_cache = reference_cache
        self.adj_divs = adj_divs
        self.reweight_grace_days = reweight_grace_days
        self.is_cross_corridor = is_cross_corridor
        # Create a backtester instance with matching config for fitness evaluation
        self.global_cap = global_cap
        self.global_floor = global_floor
        # New scoring system (bilevel)
        self._metric_weights = metric_weights
        self._use_new_scoring = metric_weights is not None
        self._score_fn: Optional[ScoreFunction] = None
        self._weight_solver: Optional[WeightSolver] = None
        self._scoring_mode: str = "legacy"  # tracks which scorer actually ran
        # _use_exact_in_ga already set at L176 from bisect_in_ga param
        self._solver_cumulative_time: float = 0.0  # track solver overhead
        _bt_config = SwapConfig(
            cross_corridor=is_cross_corridor,
            missing_data_policy=missing_data_policy or MissingDataPolicy.ADAPTIVE_REWEIGHT,
            reweight_grace_days=reweight_grace_days,
            global_cap=global_cap,
            global_floor=global_floor,
        )
        self._backtester = DispersionBacktester(_bt_config)
        # Deterministic RNG (original behavior)
        self.seed = int(seed)
        self._rng = random.Random(self.seed)
        self._np_rng = np.random.default_rng(self.seed)
        # Dedicated stream for vega reference V draws — keeps the main GA
        # stream identical whether the vega toggle is ON or OFF
        self._vega_rng = random.Random(self.seed * 1_000_003 + 77)
        # Store original P&L matrix (with NaN) for adaptive reweighting in fitness
        self._orig_ts_mat = pnl_matrix.astype(np.float64).copy()
        # Also store filled version for backward compatibility (e.g., GA initialization)
        self._ts_mat = np.nan_to_num(self._orig_ts_mat, nan=0.0)
        self._col_pos = dict(column_map)
        self._n_rows = self._ts_mat.shape[0]
        # Precompute the adaptive participation mask: raw validity, widened by
        # the reweight grace (a name keeps its weight through gaps <= grace
        # days). grace=0 returns the validity mask itself — historical
        # behaviour, bit-identical.
        self._valid_mask = ~np.isnan(self._orig_ts_mat)  # [n_rows x n_cols]
        if active_mask is not None:
            # Externally built (on FULL history, sliced to this window by the
            # caller) — a gap already open at window start stays in-grace.
            if active_mask.shape != self._orig_ts_mat.shape:
                raise ValueError(
                    f"active_mask shape {active_mask.shape} != pnl_matrix "
                    f"shape {self._orig_ts_mat.shape}")
            self._active_mask = active_mask.astype(bool)
        elif (self.missing_data_policy == MissingDataPolicy.ADAPTIVE_REWEIGHT
                and self.reweight_grace_days > 0):
            # Fallback (headless/replay without a stored mask): derive from
            # this window only — pre-window history is not visible here.
            self._active_mask = active_mask_with_grace(
                self._valid_mask, self.reweight_grace_days)
            # In-grace gap cells carry the name's last mark so the adaptive
            # numerator matches the backtester. (External masks come with a
            # matrix already carried on FULL history by _api's prep — never
            # re-carry a carried matrix.)
            self._ts_mat = np.nan_to_num(carry_pnl_within_grace(
                self._orig_ts_mat, self._valid_mask,
                self.reweight_grace_days), nan=0.0)
        else:
            self._active_mask = self._valid_mask
        self._long_only = len(self.short_candidates) == 0
        # Rejection tracking
        self._rejection_reasons = {
            "fitness<=0": 0,
            "no_weights": 0,
            "strike_invalid": 0,
            "invalid_score": 0,
            "last_carry_zero": 0,
            "len_valid_lt_50": 0,
            "bucket_counts": 0,
            "bucket_weights": 0,
            "vega_infeasible": 0,
        }
        self._strike_logged = False
        self._lp_infeasible_streak = 0
        # ── Forced inclusion (long leg): these candidate indices must be in every basket ──
        self._forced_long: List[int] = sorted(set(int(i) for i in (forced_long_indices or [])))
        if self._forced_long:
            bad = [i for i in self._forced_long if i < 0 or i >= len(self.long_candidates)]
            if bad:
                raise ValueError(f"forced_long_indices out of range: {bad} "
                                 f"(universe size = {len(self.long_candidates)})")
            k = len(self._forced_long)
            if k > self.c.max_stocks_long:
                raise ValueError(
                    f"Infeasible forced set: {k} forced names > max_stocks_long="
                    f"{self.c.max_stocks_long}. Raise max_stocks_long or force fewer names."
                )
            forced_min_sum = sum(self.long_candidates[i].min_weight for i in self._forced_long)
            if forced_min_sum > 1.0 + 1e-9:
                names = [self.long_candidates[i].variance_asset for i in self._forced_long]
                raise ValueError(
                    f"Infeasible forced set: sum of min weights of forced names = "
                    f"{forced_min_sum:.4f} > 1.0 ({names}). Lower their Min Weight inputs."
                )
        # ── Absolute-Vega mode (Phase 4c) ──
        self._vega: Optional[VegaConfig] = (
            vega_config if (vega_config is not None and vega_config.enabled) else None)
        if self._vega is not None:
            targets = []
            caps = []
            for s in self.long_candidates:
                t = float(s.axe_target) if s.axe_target is not None else 0.0
                if t < 0:
                    raise ValueError(
                        f"Vega mode: '{s.variance_asset}' has a negative Axe Target "
                        f"({t}). Axes are one-directional — targets must be >= 0.")
                cp = float(s.axe_cap) if s.axe_cap is not None else float("inf")
                if cp < 0:
                    raise ValueError(
                        f"Vega mode: '{s.variance_asset}' has a negative Axe Cap ({cp}).")
                # Symbiosis check: a selected name needs v_i >= min_weight·V_min,
                # which must fit under its hard cap.
                if s.min_weight * self._vega.v_min > cp + 1e-9:
                    raise ValueError(
                        f"Vega mode: '{s.variance_asset}' cannot be selected — its "
                        f"Min Weight ({s.min_weight:.4f} of V) at V_min="
                        f"{self._vega.v_min:g} needs {s.min_weight * self._vega.v_min:g} "
                        f"Vega but Axe Cap is {cp:g}. Lower Min Weight / V_min or "
                        f"raise the cap.")
                targets.append(t)
                caps.append(cp)
            self._vega_targets = np.asarray(targets, dtype=np.float64)
            self._vega_caps = np.asarray(caps, dtype=np.float64)
            self._vega_t_total = float(self._vega_targets.sum())
            capacity = float(np.minimum(
                np.array([s.max_weight for s in self.long_candidates]) * self._vega.v_min,
                self._vega_caps).sum())
            if capacity < self._vega.v_min - 1e-9:
                raise ValueError(
                    f"Vega mode: even the FULL candidate universe can only hold "
                    f"{capacity:g} Vega at V_min={self._vega.v_min:g} "
                    f"(per-name caps × Max Weights too tight). Lower V_min or "
                    f"relax caps/Max Weights.")
        # ── Regional/sector bucket constraints (long leg) ──
        self._bucket_constraints: List[BucketConstraint] = list(bucket_constraints or [])
        self._bucket_of: List[Optional[str]] = [
            (s.sector if s.sector else None) for s in self.long_candidates
        ]
        self._bucket_members: Dict[str, List[int]] = {}
        for i, b in enumerate(self._bucket_of):
            if b is not None:
                self._bucket_members.setdefault(b, []).append(i)
        if self._bucket_constraints:
            self._validate_bucket_constraints()
        # Precomputed count bounds per constrained bucket: {bucket: (lo, hi)}
        self._bucket_count_bounds: Dict[str, Tuple[int, Optional[int]]] = {
            bc.bucket: (bc.min_names, bc.max_names) for bc in self._bucket_constraints
        }
        # Print all config at init
        if long_candidates:
            s = long_candidates[0]
    # ══════════════════════════════════════════════════════════════════════════
    # BUCKET CONSTRAINTS (regional/sector, long leg)
    # ══════════════════════════════════════════════════════════════════════════

    def _validate_bucket_constraints(self) -> None:
        """Config-time feasibility of the bucket constraints. Raises with an
        actionable message on the first violation found."""
        c = self.c
        seen = set()
        for bc in self._bucket_constraints:
            if bc.bucket in seen:
                raise ValueError(f"Duplicate BucketConstraint for bucket '{bc.bucket}'.")
            seen.add(bc.bucket)
            members = self._bucket_members.get(bc.bucket, [])
            if not members and (bc.min_names > 0 or bc.min_weight > 0):
                available = sorted(self._bucket_members.keys())
                raise ValueError(
                    f"Bucket '{bc.bucket}' has no candidates (check the 'Sector' "
                    f"column of the input). Available buckets: {available or 'none'}."
                )
            if bc.min_names > len(members):
                raise ValueError(
                    f"Bucket '{bc.bucket}': min_names={bc.min_names} > "
                    f"{len(members)} available candidate(s) in that bucket."
                )
            if bc.min_weight > 0 and bc.min_names < 1:
                raise ValueError(
                    f"Bucket '{bc.bucket}': min_weight={bc.min_weight:.4f} requires "
                    f"min_names >= 1 (a basket without the bucket can never reach "
                    f"the weight floor). Set min_names accordingly."
                )
            if bc.min_weight > 0 and members:
                max_reachable = sum(self.long_candidates[i].max_weight for i in members)
                if bc.min_weight > max_reachable + 1e-9:
                    raise ValueError(
                        f"Bucket '{bc.bucket}': min_weight={bc.min_weight:.4f} exceeds "
                        f"the sum of Max Weight over its {len(members)} candidate(s) "
                        f"({max_reachable:.4f}). Raise their Max Weight or lower the floor."
                    )
            # Forced names inside the bucket must fit under max_names
            forced_in = sum(1 for i in self._forced_long if self._bucket_of[i] == bc.bucket)
            if bc.max_names is not None and forced_in > bc.max_names:
                raise ValueError(
                    f"Bucket '{bc.bucket}': {forced_in} forced name(s) exceed "
                    f"max_names={bc.max_names}."
                )
        total_min_names = sum(bc.min_names for bc in self._bucket_constraints)
        if total_min_names > c.max_stocks_long:
            raise ValueError(
                f"Sum of bucket min_names ({total_min_names}) > max_stocks_long "
                f"({c.max_stocks_long}). Relax the bucket minima or raise max_stocks_long."
            )
        # Effective floor with forced names: forced inside a constrained bucket
        # count toward (or exceed) its minimum; forced outside add on top.
        eff_min = 0
        for bc in self._bucket_constraints:
            forced_in = sum(1 for i in self._forced_long if self._bucket_of[i] == bc.bucket)
            eff_min += max(bc.min_names, forced_in)
        constrained = {bc.bucket for bc in self._bucket_constraints}
        eff_min += sum(1 for i in self._forced_long
                       if self._bucket_of[i] is None or self._bucket_of[i] not in constrained)
        if eff_min > c.max_stocks_long:
            raise ValueError(
                f"Bucket minima + forced names require at least {eff_min} names > "
                f"max_stocks_long ({c.max_stocks_long})."
            )
        total_min_weight = sum(bc.min_weight for bc in self._bucket_constraints)
        if total_min_weight > 1.0 + 1e-9:
            raise ValueError(
                f"Sum of bucket min_weight floors ({total_min_weight:.4f}) > 1.0 — "
                f"the long weights cannot satisfy all floors simultaneously."
            )
        # Overall capacity: can any subset of allowed size exist?
        capped = 0
        for b, members in self._bucket_members.items():
            lo_hi = next(((bc.min_names, bc.max_names) for bc in self._bucket_constraints
                          if bc.bucket == b), None)
            cap = len(members)
            if lo_hi is not None and lo_hi[1] is not None:
                cap = min(cap, lo_hi[1])
            capped += cap
        unbucketed = sum(1 for b in self._bucket_of if b is None)
        if capped + unbucketed < max(c.min_stocks_long, total_min_names, len(self._forced_long)):
            raise ValueError(
                f"Bucket caps leave at most {capped + unbucketed} selectable names — "
                f"fewer than the required basket size "
                f"({max(c.min_stocks_long, total_min_names, len(self._forced_long))})."
            )

    def _bucket_count(self, indices, bucket: str) -> int:
        return sum(1 for i in indices if self._bucket_of[i] == bucket)

    def _bucket_counts_ok(self, indices) -> bool:
        """Cardinality bounds per bucket on a candidate subset."""
        for bucket, (lo, hi) in self._bucket_count_bounds.items():
            cnt = self._bucket_count(indices, bucket)
            if cnt < lo:
                return False
            if hi is not None and cnt > hi:
                return False
        return True

    def _bucket_groups_for(self, indices) -> Optional[list]:
        """Per-bucket WEIGHT bounds for the inner solver, as
        (positions-within-subset, lo, hi) tuples. None when inactive."""
        if not self._bucket_constraints:
            return None
        groups = []
        for bc in self._bucket_constraints:
            if not bc.has_weight_bounds:
                continue
            pos = [k for k, i in enumerate(indices) if self._bucket_of[i] == bc.bucket]
            if not pos:
                continue  # min_weight>0 implies min_names>=1, enforced upstream
            groups.append((np.asarray(pos, dtype=int),
                           bc.min_weight if bc.min_weight > 0 else None,
                           bc.max_weight))
        return groups or None

    # ══════════════════════════════════════════════════════════════════════════
    # ABSOLUTE-VEGA MODE (Phase 4c)
    # ══════════════════════════════════════════════════════════════════════════

    def _vega_spec_for(self, indices) -> Optional[VegaSpec]:
        """Per-subset VegaSpec for the inner solver (None when OFF)."""
        if self._vega is None:
            return None
        idx = list(indices)
        return VegaSpec(
            targets=self._vega_targets[idx],
            caps=self._vega_caps[idx],
            v_min=self._vega.v_min,
            v_max=self._vega.v_max,
            t_total=self._vega_t_total,
        )

    def _axe_grid(self, indices, w) -> Optional[Tuple[float, float, float]]:
        """(V, A, B) for arbitrary weights via the deterministic V-grid rule
        (same rule the solver uses per evaluation).  None = caps leave no
        feasible V for these weights, or vega mode is OFF."""
        if self._vega is None or self._weight_solver is None:
            return None
        lam_a = self._metric_weights.get("axe_book_cleaned", 0.0)
        lam_b = self._metric_weights.get("axe_package_recycled", 0.0)
        # P&L-only configs: the max-clean rule (lam_a=1) picks the largest
        # feasible V — most book cleaned at zero cost to the active metrics.
        if lam_a <= 0 and lam_b <= 0:
            lam_a = 1.0
        return self._weight_solver._vega_choose_V(
            np.asarray(w, dtype=np.float64), self._vega_spec_for(indices), lam_a, lam_b)

    def _axe_ctx_values(self, indices, w, extra: Optional[Dict]) -> Tuple[Optional[float], Optional[float]]:
        """(axe_cleaned, axe_recycled) for scoring context: exact solver
        extras when available, else the deterministic grid rule."""
        if self._vega is None:
            return None, None
        if extra and extra.get("axe_cleaned") is not None:
            return extra.get("axe_cleaned"), extra.get("axe_recycled")
        picked = self._axe_grid(indices, w)
        if picked is None:
            return None, None
        return picked[1], picked[2]

    def _reference_axe_extras(self, long_indices, long_w) -> Dict[str, float]:
        """Per-sample extras for the reference build: A/B of the sampled
        basket at a uniformly drawn V (spans the [v_min, v_max] range so the
        normalizer sees the whole criterion distribution).  Empty when OFF."""
        if self._vega is None:
            return {}
        v_total = self._vega_rng.uniform(self._vega.v_min, self._vega.v_max)
        idx = list(long_indices)
        v_abs = np.minimum(np.asarray(long_w, dtype=np.float64) * v_total,
                           self._vega_caps[idx])
        cleaned = float(np.minimum(v_abs, self._vega_targets[idx]).sum())
        out = {}
        if self._vega_t_total > 0:
            out["axe_book_cleaned"] = cleaned / self._vega_t_total
        s = float(v_abs.sum())
        if s > 0:
            out["axe_package_recycled"] = cleaned / s
        return out

    def _subset_arrays(self, indices, candidates=None):
        """Resolve a candidate subset to its matrix/solver inputs (single
        implementation of a pattern that used to be copied ~6 times).

        Returns (keys, cols, strikes, bounds):
          keys    — per-leg series keys (:func:`_candidate_key`)
          cols    — int array of P&L matrix columns, or None when ANY key is
                    absent from the matrix (callers treat that as invalid)
          strikes — per-leg solver strike vector (XC convention mono − cross)
          bounds  — (n, 2) per-name [min_weight, max_weight]
        """
        cands = self.long_candidates if candidates is None else candidates
        keys = [_candidate_key(cands[i], self.is_cross_corridor) for i in indices]
        pos = [self._col_pos.get(k) for k in keys]
        if len(pos) == 0 or any(c is None for c in pos):
            return keys, None, None, None
        strikes = self._solver_strikes(indices, cands)
        bounds = np.array([[cands[i].min_weight, cands[i].max_weight] for i in indices],
                          dtype=np.float64)
        return keys, np.array(pos, dtype=int), strikes, bounds

    def _print_candidate_table(self):
        """Print a visible table of long/short candidates with PnL data quality."""
        lines = []
        lines.append("\n" + "=" * 90)
        lines.append("📋 CANDIDATE DATA QUALITY TABLE (after reference sample)")
        lines.append("=" * 90)
        lines.append(f"{'#':<4}{'Ticker':<18}{'Key':<22}{'InMatrix':<10}{'NonZero':<8}{'ValidRows':<10}{'MinW':<7}{'MaxW':<7}{'Strike':<8}")
        lines.append("-" * 90)
        for i, s in enumerate(self.long_candidates[:30]):
            key = _candidate_key(s, self.is_cross_corridor)
            in_matrix = key in self._col_pos
            if in_matrix:
                col_idx = self._col_pos[key]
                col_data = self._orig_ts_mat[:, col_idx]
                nonzero = int(np.count_nonzero(~np.isnan(col_data) & (col_data != 0)))
                valid = int(np.count_nonzero(~np.isnan(col_data)))
            else:
                nonzero = 0
                valid = 0
            lines.append(f"{i:<4}{s.variance_asset:<18}{key:<22}{'✓' if in_matrix else '✗':<10}{nonzero:<8}{valid:<10}{s.min_weight:<7.4f}{s.max_weight:<7.4f}{s.strike_mono_var_swap:<8.4f}")
        if len(self.long_candidates) > 30:
            lines.append(f"  ... and {len(self.long_candidates) - 30} more")
        # Summary
        total_in = sum(1 for s in self.long_candidates if _candidate_key(s, self.is_cross_corridor) in self._col_pos)
        lines.append("-" * 90)
        lines.append(f"TOTAL: {len(self.long_candidates)} candidates | {total_in} in matrix | {len(self.long_candidates) - total_in} MISSING")
        lines.append(f"Matrix columns: {len(self._col_pos)} | Matrix shape: {self._orig_ts_mat.shape}")
        if total_in == 0:
            lines.append("⚠️  CRITICAL: ZERO candidates found in PnL matrix! All baskets will fail.")
            lines.append(f"  Candidate keys sample: {[_candidate_key(s, self.is_cross_corridor) for s in self.long_candidates[:5]]}")
            lines.append(f"  Matrix col keys sample: {list(self._col_pos.keys())[:5]}")
        lines.append("=" * 90)
        self._debug_table = "\n".join(lines)
    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════════════════

    def run(self) -> OptimizationResult:
        """Execute the genetic algorithm. Returns best solution."""
        setup_start = time.time()
        if len(self.long_candidates) < self.c.min_stocks_long:
            return self._empty_result()
        if not self._long_only and len(self.short_candidates) < self.c.min_stocks_short:
            return self._empty_result()
        # ── Setup new scoring system (bilevel) if metric_weights provided ──
        if self._use_new_scoring:
            # Vega-mode / axe-criteria coherence (checked BEFORE the reference
            # build so the error is the actionable one, not a fit failure)
            _axe_names = (set(self._metric_weights.active_names)
                          & {"axe_book_cleaned", "axe_package_recycled"})
            if _axe_names and self._vega is None:
                raise RuntimeError(
                    f"Metric(s) {sorted(_axe_names)} are weighted but the absolute-"
                    f"Vega toggle is OFF. Pass vega=VegaConfig(v_min=..., v_max=...) "
                    f"(optimize(vega=...)) to activate axe recycling.")
            if _axe_names and self._vega_t_total <= 0:
                raise RuntimeError(
                    f"Metric(s) {sorted(_axe_names)} are weighted but no candidate "
                    f"carries an Axe Target — add the 'Axe Target' column to the "
                    f"long input (absolute Vega units).")
            self._score_fn = make_default_score_function(weights=self._metric_weights)
            ctx = ScoreContext(n_days=self._n_rows)
            # Weight bounds from stock objects (already decimal: 0.015 = 1.5%)
            all_min_w = [s.min_weight for s in self.long_candidates]
            all_max_w = [s.max_weight for s in self.long_candidates]
            wc_min = min(all_min_w) if all_min_w else 0.05
            wc_max = max(all_max_w) if all_max_w else 0.60
            wc = WeightConstraints(
                min_weight=wc_min,
                max_weight=wc_max,
                max_net_strike=self.c.max_net_strike if self.c.max_net_strike > 0 else None,
                max_stocks=self.c.max_stocks_long,
            )
            self._wc = wc
            active_metrics = [m for m in self._score_fn._metrics
                             if m.name in self._metric_weights.active_names]
            # P5: Adaptive n_samples — reduced for speed; tail metrics get more.
            # An explicit n_reference_samples overrides the adaptive default.
            has_tail = any(getattr(m, 'is_tail_metric', False) for m in active_metrics)
            n_samples = (self._n_reference_samples if self._n_reference_samples
                         else (TUNING.n_reference_tail if has_tail else TUNING.n_reference_base))
            self._weight_solver = None  # not available during reference build
            self._build_reference_sample(n_samples=n_samples)
            # ── TABLE: Long candidates + PnL data quality ──
            self._print_candidate_table()
            if not self._score_fn.is_fitted:
                raise RuntimeError(
                    "Scoring system failed to initialise: build_reference did not fit. "
                    "Check that _compute_net_pnl produces valid P&L for at least 100 random baskets."
                )
            self._scoring_mode = "v2"
            # Reproducibility fingerprint of this run's scoring setup
            self._scoring_signature = _scoring_signature(
                self._metric_weights, self.seed,
                getattr(self, "_ref_n_samples", n_samples),
                getattr(self, "_reference_size", 0),
            )
            # Construct weight solver AFTER fitting so it sees the fitted score_fn
            # Determine weight solver policy: _use_exact_in_ga=True uses exact adaptive bisection,
            # False uses fill_zero proxy in GA (safety-net always uses exact).
            if self._use_exact_in_ga:
                _ws_policy = "adaptive_reweight" if self.missing_data_policy == MissingDataPolicy.ADAPTIVE_REWEIGHT else (
                    "drop_incomplete" if self.missing_data_policy == MissingDataPolicy.DROP_INCOMPLETE_DAYS else "fill_zero"
                )
            else:
                _ws_policy = "fill_zero"  # fast proxy for GA; safety-net overrides
            self._weight_solver = WeightSolver(
                score_fn=self._score_fn, ctx=ctx, constraints=wc,
                n_restarts=3, seed=self.seed,
                missing_data_policy=_ws_policy,
            )
            _active_names = set(self._metric_weights.active_names)
            # Genome = subset only: fitness derives weights deterministically.
            # Exact-path configs solve the exact inner problem per subset
            # (memoised — fitness is a pure function of the subset); all
            # other configs use the bounded equal-weight projection during
            # the GA, with SLSQP reserved for the post-GA refinement.
            # In vega mode, the pure axe_book_cleaned objective is ALSO an
            # exact path (dedicated LP in absolute-Vega space).
            self._exact_fitness = (self._weight_solver.has_exact_path()
                                   or (self._vega is not None
                                       and _active_names == {"axe_book_cleaned"}))
        # Create initial population
        population = []
        attempts = 0
        while len(population) < self.c.population_size and attempts < self.c.population_size * TUNING.population_attempts_factor:
            ind = self._create_random_individual()
            if ind is not None:
                population.append(ind)
            attempts += 1
        if not population:
            return self._empty_result()
        # Start the evolution timer AFTER reference sample + population init
        # so the GA gets the full time budget for actual evolution
        start_time = time.time()
        setup_elapsed = start_time - setup_start
        self.log("INFO", f"⚙️ Setup took {setup_elapsed:.1f}s — GA evolution starts now (budget: {self.c.time_limit_seconds}s)")
        _engine_log.debug(f"[TIMING] setup={setup_elapsed:.1f}s | GA budget={self.c.time_limit_seconds}s")
        init_weights = np.concatenate([ind.long_weights for ind in population])
        best_individual = max(population, key=lambda x: x.fitness)
        best_score = best_individual.fitness
        generations_run = 0
        stagnation = 0
        # Evolution loop
        for generation in range(self.c.max_generations):
            if time.time() - start_time > self.c.time_limit_seconds:
                self.log("INFO", f"⏰ Time limit at gen {generation} (elapsed={time.time()-start_time:.1f}s)")
                break
            generations_run = generation
            # Sort by fitness
            population.sort(key=lambda x: x.fitness, reverse=True)
            # Elitism
            elite_count = max(1, int(len(population) * self.c.elite_ratio))
            new_population = [self._copy(ind) for ind in population[:elite_count]]
            # Fill rest via crossover + mutation
            fill_attempts = 0
            crossover_ok = 0
            crossover_fail = 0
            mutations_applied = 0
            while len(new_population) < self.c.population_size and fill_attempts < self.c.population_size * TUNING.population_attempts_factor:
                fill_attempts += 1
                p1 = self._tournament_select(population)
                p2 = self._tournament_select(population)
                child = self._crossover(p1, p2)
                if child is None:
                    crossover_fail += 1
                    continue
                crossover_ok += 1
                if self._rng.random() < self.c.mutation_rate:
                    mutated = self._mutate(child)
                    if mutated is not None:
                        child = mutated
                        mutations_applied += 1
                new_population.append(child)
            # Random immigrants: inject fresh individuals to maintain diversity
            n_immigrants = max(TUNING.immigrants_min, int(self.c.population_size * TUNING.immigrants_frac))
            for _ in range(n_immigrants):
                immigrant = self._create_random_individual()
                if immigrant is not None:
                    # Replace worst individual if population is full, else append
                    if len(new_population) >= self.c.population_size:
                        # Replace the worst non-elite individual
                        worst_idx = min(range(elite_count, len(new_population)),
                                       key=lambda i: new_population[i].fitness,
                                       default=None)
                        if worst_idx is not None and immigrant.fitness > new_population[worst_idx].fitness:
                            new_population[worst_idx] = immigrant
                    else:
                        new_population.append(immigrant)
            population = new_population
            # Population floor: if crossover failures shrank the pop, backfill with random
            min_pop = max(TUNING.pop_floor_abs, int(self.c.population_size * TUNING.pop_floor_frac))
            if len(population) < min_pop:
                backfill_attempts = 0
                while len(population) < min_pop and backfill_attempts < min_pop * TUNING.pop_backfill_factor:
                    backfill_attempts += 1
                    ind = self._create_random_individual()
                    if ind is not None:
                        population.append(ind)
            # Track best + stagnation (AFTER immigrants injected)
            gen_best = max(population, key=lambda x: x.fitness)
            if gen_best.fitness > best_score:
                best_individual = self._copy(gen_best)
                best_score = gen_best.fitness
                stagnation = 0
            else:
                stagnation += 1
            _engine_log.debug(f"[GEN {generation}] best_fitness={best_score:.4f} | elapsed={time.time()-start_time:.1f}s | cache={self._weight_solver._cache_hits}/{self._weight_solver._cache_misses} | avg_LPs={self._weight_solver._bisect_lp_count/max(1,self._weight_solver._cache_misses):.1f}")
            if stagnation > self.c.stagnation_limit:
                self.log("INFO", f"✅ Converged at gen {generation} (stagnation={self.c.stagnation_limit})")
                break
            if generation % 25 == 0:
                fitnesses = sorted([p.fitness for p in population], reverse=True)
                self.log("PROGRESS", f"{generation}/{self.c.max_generations}|best={best_score:.4f}")
                if self._progress_cb:
                    try:
                        self._progress_cb(generation, self.c.max_generations, best_score)
                    except Exception:
                        pass
        elapsed_ga = time.time() - start_time
        if best_score == -np.inf:
            return self._empty_result()
        # ── Post-GA refinement ──
        if self._use_new_scoring and self._weight_solver is not None:
            # Refine top elites for EVERY config (subset-only genome: this is
            # the only place non-exact configs get weight optimisation).
            # Exact-path configs get the LP/bisection optimum; non-linear
            # blends — including hit_ratio-only via its soft surrogate — go
            # through the smooth-normalised SLSQP inner solver.
            pre_score = best_score
            population.sort(key=lambda x: x.fitness, reverse=True)
            top_k = min(TUNING.refine_top_k, len(population))
            best_individual, best_score = self._refine_top_individuals(
                population[:top_k], best_individual, best_score
            )
            self.log("INFO", f"Refinement: {best_score:.4f} (was {pre_score:.4f})")
        else:
            pass  # Legacy scoring or non-linear metrics — no SLSQP refinement
        # ── Exact swap local search (subset-level, exact inner solve) ──
        # The GA fitness for non-linear exact-path configs (e.g. min_payoff)
        # scores individuals at their OWN evolved weights, so the subset
        # ranking is a proxy: on large homogeneous universes the argmax subset
        # can be missed by the top-K refinement pool.  This pass hill-climbs
        # single-stock swaps around the incumbent, each neighbour evaluated at
        # its EXACT optimal weights (LP, cached) — the returned basket is at
        # least a 1-swap local optimum of the true bilevel objective.
        if (self._use_new_scoring and self._weight_solver is not None
                and getattr(self, "_exact_fitness", False)):
            pre_ls = best_score
            _ls_pool = sorted(population, key=lambda x: x.fitness, reverse=True)[:TUNING.refine_top_k]
            best_individual, best_score = self._exact_swap_local_search(
                best_individual, best_score, start_pool=_ls_pool)
            if best_score > pre_ls + 1e-12:
                self.log("INFO", f"Local search: {best_score:.4f} (was {pre_ls:.4f})")
        elapsed = time.time() - start_time
        tickers = [self.long_candidates[i].variance_asset for i in best_individual.long_indices]
        self.log("SUCCESS", f"Score: {best_score:.4f} | {generations_run + 1} gens | {elapsed:.1f}s (GA={elapsed_ga:.1f}s)")
        self._last_best = best_individual
        # Snapshot the final population (best-first) — challenger source for
        # the bootstrap diagnostic when the refinement pool lacks distinct subsets
        self._final_population = [
            (list(ind.long_indices), np.array(ind.long_weights, dtype=np.float64),
             list(ind.short_indices), np.array(ind.short_weights, dtype=np.float64),
             float(ind.fitness))
            for ind in sorted(population, key=lambda x: x.fitness, reverse=True)[:TUNING.bootstrap_population_snapshot]
        ]
        return self._to_result(best_individual, generations_run + 1)

    def _refine_top_individuals(
        self,
        candidates: list,
        current_best: "_Individual",
        current_best_score: float,
    ) -> tuple:
        """Refine top-K GA elites with full SLSQP weight optimisation.
        This is where the bilevel inner solver actually runs — only on the
        best subsets found by the GA, not on every individual in every gen.
        Typically K=10, and each solve takes ~60-120ms → total ~1s.
        """
        active = self._score_fn.weights.active_names
        best = current_best
        best_score = current_best_score
        ctx = self._make_ctx()  # ws-agnostic fallback; per-basket ctx built below
        _pass_strikes = (self.c.max_net_strike > 0) or self._ws_active()

        def _ctx_for(ind_obj, long_w, solve_extra=None) -> ScoreContext:
            _ws_v = (self._net_strike(
                ind_obj.long_indices[:len(long_w)], long_w,
                ind_obj.short_indices, ind_obj.short_weights)
                if self._ws_active() else None)
            _ac, _ar = self._axe_ctx_values(
                ind_obj.long_indices[:len(long_w)], long_w, solve_extra)
            if _ws_v is None and _ac is None and _ar is None:
                return ctx
            return self._make_ctx(_ws_v, _ac, _ar)
        # Ensure weight solver uses the SAME fitted score function (not a stale ref)
        self._weight_solver._score_fn = self._score_fn
        n_infeasible = 0
        n_no_improve = 0
        n_improved = 0
        # Challenger pool for the bootstrap diagnostic (subset, solved weights,
        # short side, step score) — every feasible refinement candidate lands here
        self._refine_candidates: List[Tuple[list, np.ndarray, list, np.ndarray, float]] = []

        # CHANGE 8b: Determine acceptance mode
        active_set = set(active)
        is_single_metric = len(active) == 1
        is_concave_blend = active_set.issubset({"min_payoff", "mean_payoff", "last_carry"}) and len(active_set) >= 2
        has_hit_ratio = "hit_ratio" in active_set

        # For single-metric or concave-blend: track raw incumbent value
        if is_single_metric and not has_hit_ratio:
            metric_name = active[0]
            metric_obj = next(m for m in self._score_fn.metrics if m.name == metric_name)
            # Compute incumbent raw value from current best
            _ik, inc_long_pos, _is, _ib = self._subset_arrays(best.long_indices)
            if inc_long_pos is not None:
                inc_pnl = self._adaptive_net_pnl(inc_long_pos, best.long_weights)
                inc_raw = self._score_fn.raw_metrics(
                    inc_pnl, _ctx_for(best, best.long_weights)
                ).get(metric_name, -np.inf if metric_obj.higher_is_better else np.inf)
            else:
                inc_raw = -np.inf if metric_obj.higher_is_better else np.inf
        elif is_concave_blend and not has_hit_ratio:
            # Scalarized raw objective for the incumbent — centralized blend
            _lam_min, _lam_mean, _lam_carry, _k_carry = concave_blend_lambdas(self._score_fn)

            def _raw_blend_obj(pnl_series):
                return concave_blend_value(pnl_series, _lam_min, _lam_mean,
                                           _lam_carry, _k_carry)

            _ik, inc_long_pos, _is, _ib = self._subset_arrays(best.long_indices)
            if inc_long_pos is not None:
                inc_pnl = self._adaptive_net_pnl(inc_long_pos, best.long_weights)
                inc_raw = _raw_blend_obj(inc_pnl)
            else:
                inc_raw = -np.inf

        for ind in candidates:
            _k, long_pos, strikes, per_stock_bounds = self._subset_arrays(ind.long_indices)
            if long_pos is None:
                continue
            n_long = len(long_pos)
            result = self._weight_solver.solve(
                pnl_matrix=self._ts_mat,
                stock_indices=long_pos,
                strikes=strikes if _pass_strikes else None,
                per_stock_bounds=per_stock_bounds,
                active_mask=self._active_mask,
                group_bounds=self._bucket_groups_for(ind.long_indices[:n_long]),
                vega=self._vega_spec_for(ind.long_indices[:n_long]),
            )
            if not result.feasible:
                n_infeasible += 1
                continue
            # Compute net PnL using adaptive formula
            _short_pos_r = None
            _short_w_r = None
            if len(ind.short_indices) > 0:
                short_ids = [_candidate_key(self.short_candidates[i], self.is_cross_corridor) for i in ind.short_indices]
                short_pos_arr = np.array([self._col_pos[c] for c in short_ids if c in self._col_pos])
                if len(short_pos_arr) > 0:
                    _short_pos_r = short_pos_arr
                    _short_w_r = ind.short_weights
            net_pnl = self._adaptive_net_pnl(long_pos, result.weights, _short_pos_r, _short_w_r)
            n_nz = int(np.count_nonzero(net_pnl))
            if n_nz < TUNING.min_valid_days:
                continue
            _cand_ctx = _ctx_for(ind, result.weights, solve_extra=result.extra)
            score = self._score_fn.score(net_pnl, _cand_ctx)
            self._refine_candidates.append((
                list(ind.long_indices[:n_long]), np.array(result.weights, dtype=np.float64),
                list(ind.short_indices), np.array(ind.short_weights, dtype=np.float64),
                float(score)))

            # CHANGE 8b: Acceptance test based on config type
            if is_single_metric and not has_hit_ratio:
                # Single metric (not hit_ratio): compare raw metric value
                raw_val = self._score_fn.raw_metrics(net_pnl, _cand_ctx).get(metric_name, -np.inf)
                improved = (raw_val > inc_raw) if metric_obj.higher_is_better else (raw_val < inc_raw)
                self.log("DEBUG", f"  [REFINE] raw={raw_val:.6f} vs inc={inc_raw:.6f} -> {'ACCEPTED' if improved else 'REJECTED'}")
                if improved:
                    n_improved += 1
                    inc_raw = raw_val
                    best_score = score
                    ind.long_weights = result.weights
                    ind.fitness = score
                    best = self._copy(ind)
                else:
                    n_no_improve += 1
            elif is_concave_blend and not has_hit_ratio:
                # Concave blend: compare scalarized raw objective
                raw_val = _raw_blend_obj(net_pnl)
                improved = raw_val > inc_raw
                self.log("DEBUG", f"  [REFINE] blend_obj={raw_val:.6f} vs inc={inc_raw:.6f} -> {'ACCEPTED' if improved else 'REJECTED'}")
                if improved:
                    n_improved += 1
                    inc_raw = raw_val
                    best_score = score
                    ind.long_weights = result.weights
                    ind.fitness = score
                    best = self._copy(ind)
                else:
                    n_no_improve += 1
            else:
                # Configs containing hit_ratio: accept on the SAME tie-broken
                # scale the GA ranks on (step score saturates at the top).
                cand_fit = self._fitness_from_net(net_pnl, _cand_ctx)
                if cand_fit is not None and cand_fit > best_score:
                    n_improved += 1
                    best_score = cand_fit
                    ind.long_weights = result.weights
                    ind.fitness = cand_fit
                    best = self._copy(ind)
                else:
                    n_no_improve += 1

        self.log("DEBUG", f"Refinement stats: {n_improved} improved, {n_infeasible} infeasible, {n_no_improve} no-improve (of {len(candidates)})")
        return best, best_score
    # ══════════════════════════════════════════════════════════════════════════
    # FITNESS (exact port of original _calculate_fitness)
    # ══════════════════════════════════════════════════════════════════════════

    def _exact_swap_local_search(self, best: "_Individual", best_score: float,
                                 time_budget: Optional[float] = None,
                                 start_pool: Optional[list] = None) -> tuple:
        """Best-improvement hill-climb on stock swaps of the long leg.

        Only runs for exact-path configs (all-linear, min_payoff-only,
        concave blend), where the inner solve is an exact LP: acceptance
        compares the scalarised RAW objective (currency units, sign-adjusted)
        so it cannot be fooled by quantile-score saturation at the top.
        Robustness against GA basin misses (homogeneous universes):
          1. multi-start — descents also start from the best DISTINCT
             subsets of ``start_pool`` (refinement elite), not just the
             incumbent;
          2. 2-swap escape — when the 1-swap descent stalls with budget
             left, the pair-swap neighbourhood of the best-known subset is
             scanned in deterministic order; any improvement restarts the
             1-swap descent.
        Forced names are never swapped out; the short leg is kept fixed
        (mirrors _refine_top_individuals).  Wall-time bounded.
        """
        if time_budget is None:
            time_budget = min(TUNING.local_search_cap_s, max(TUNING.local_search_floor_s, TUNING.local_search_frac * float(self.c.time_limit_seconds)))
        t0 = time.time()
        lam = self._score_fn.weights
        metric_map = {m.name: m for m in self._score_fn.metrics}
        _pass_strikes = (self.c.max_net_strike > 0) or self._ws_active()

        # Fixed short side from the incumbent (long-leg search only)
        _short_pos = None
        _short_w = None
        if len(best.short_indices) > 0:
            _sids = [_candidate_key(self.short_candidates[i], self.is_cross_corridor)
                     for i in best.short_indices]
            _spos = np.array([self._col_pos[c] for c in _sids if c in self._col_pos])
            if len(_spos) > 0:
                _short_pos = _spos
                _short_w = best.short_weights

        def _raw_scalar(net_pnl: np.ndarray, ctx: ScoreContext) -> float:
            raw = self._score_fn.raw_metrics(net_pnl, ctx)
            total = 0.0
            for nm in lam.active_names:
                v = raw.get(nm, np.nan)
                if not np.isfinite(v):
                    return -np.inf
                total += lam[nm] * (1.0 if metric_map[nm].higher_is_better else -1.0) * v
            return float(total)

        def _eval_subset(indices: List[int]):
            """-> (raw_scalar, step_score, weights) at exact optimal weights, or None."""
            if self._bucket_count_bounds and not self._bucket_counts_ok(indices):
                return None
            _ks, pos, strikes, bounds = self._subset_arrays(indices)
            if pos is None:
                return None
            res = self._weight_solver.solve(
                pnl_matrix=self._ts_mat, stock_indices=pos,
                strikes=strikes if _pass_strikes else None,
                per_stock_bounds=bounds, active_mask=self._active_mask,
                group_bounds=self._bucket_groups_for(indices),
                vega=self._vega_spec_for(indices),
            )
            if not res.feasible:
                return None
            if not self._is_strike_valid(indices, res.weights,
                                         best.short_indices, best.short_weights):
                return None
            net = self._adaptive_net_pnl(pos, res.weights, _short_pos, _short_w)
            if int(np.count_nonzero(net)) < TUNING.min_valid_days:
                return None
            ws = (self._net_strike(indices, res.weights,
                                   best.short_indices, best.short_weights)
                  if self._ws_active() else None)
            _ac, _ar = self._axe_ctx_values(indices, res.weights, res.extra)
            ctx = self._make_ctx(ws, _ac, _ar)
            return _raw_scalar(net, ctx), self._score_fn.score(net, ctx), res.weights

        inc = _eval_subset(list(best.long_indices))
        if inc is None:
            return best, best_score
        inc_scalar, inc_score, inc_w = inc
        forced = set(self._forced_long)
        n_evals = 0
        improved_any = False

        # ── Exhaustive global search on small feasible spaces ──
        # When the number of feasible long subsets is small enough, enumerate
        # ALL of them (each inner solve is a cheap cached exact LP): this
        # returns the TRUE argmax subset, making corner extremality
        # seed-INDEPENDENT for modest universes.  Large production universes
        # (combinatorially huge) fall through to the heuristic descent+escape
        # below.  Acceptance is best-scalar, so already-optimal incumbents
        # (e.g. the exact-path goldens) are returned unchanged.
        _free_pool_ls = [j for j in range(len(self.long_candidates)) if j not in forced]
        _lo_ls, _hi_ls = self._long_size_bounds(len(_free_pool_ls))
        _n_forced = len(forced)
        _total_subsets = sum(
            math.comb(len(_free_pool_ls), s - _n_forced)
            for s in range(_lo_ls, _hi_ls + 1)
            if 0 <= s - _n_forced <= len(_free_pool_ls)
        )
        if 0 < _total_subsets <= TUNING.local_search_exhaustive_max_subsets:
            # No time check: the subset count is already bounded by the
            # ceiling, so the enumeration ALWAYS completes → the returned
            # argmax is deterministic and seed-independent (the guarantee).
            best_ex = None  # (scalar, step, weights, indices)
            for s in range(_lo_ls, _hi_ls + 1):
                k = s - _n_forced
                if k < 0 or k > len(_free_pool_ls):
                    continue
                for combo in itertools.combinations(_free_pool_ls, k):
                    trial = sorted(forced | set(combo))
                    out = _eval_subset(trial)
                    n_evals += 1
                    if out is None:
                        continue
                    if best_ex is None or out[0] > best_ex[0] + 1e-12:
                        best_ex = (out[0], out[1], out[2], trial)
            if best_ex is not None:
                _engine_log.debug(f"[LOCAL-SEARCH] exhaustive n_subsets={_total_subsets} evals={n_evals} scalar={best_ex[0]:.6f}")
                if best_ex[0] > inc_scalar + 1e-12:
                    best = self._copy(best)
                    best.long_indices = list(best_ex[3])
                    best.long_weights = best_ex[2]
                    best.fitness = best_ex[1]
                    return best, max(best_score, best_ex[1])
                best.long_weights = inc_w
                best.fitness = inc_score
                return best, max(best_score, inc_score)
            # No feasible subset at all → fall through (shouldn't happen).

        def _descend_1swap(indices, scalar, score, w):
            """Best-improvement 1-swap descent to a local optimum (or budget)."""
            nonlocal n_evals
            for _sweep in range(TUNING.local_search_sweeps):
                if time.time() - t0 > time_budget:
                    break
                best_move = None  # (scalar, score, weights, indices)
                outside = [j for j in range(len(self.long_candidates)) if j not in indices]
                for p in range(len(indices)):
                    if indices[p] in forced:
                        continue
                    for j in outside:
                        if time.time() - t0 > time_budget:
                            break
                        trial = list(indices)
                        trial[p] = j
                        out = _eval_subset(sorted(trial))
                        n_evals += 1
                        if out is None:
                            continue
                        sc, st, ww = out
                        if sc > scalar + 1e-12 and (best_move is None or sc > best_move[0]):
                            best_move = (sc, st, ww, sorted(trial))
                if best_move is None:
                    break
                scalar, score, w, indices = best_move
            return scalar, score, w, list(indices)

        def _best_2swap(indices, scalar):
            """Deterministic scan of the pair-swap neighbourhood of ``indices``.
            Returns the best improving (scalar, score, weights, indices) found
            before the time budget runs out, or None."""
            nonlocal n_evals
            free_pos = [p for p in range(len(indices)) if indices[p] not in forced]
            outside = [j for j in range(len(self.long_candidates)) if j not in indices]
            best_move = None
            for p, q in itertools.combinations(free_pos, 2):
                for j1, j2 in itertools.combinations(outside, 2):
                    if time.time() - t0 > time_budget:
                        return best_move
                    trial = list(indices)
                    trial[p] = j1
                    trial[q] = j2
                    out = _eval_subset(sorted(trial))
                    n_evals += 1
                    if out is None:
                        continue
                    sc, st, ww = out
                    if sc > scalar + 1e-12 and (best_move is None or sc > best_move[0]):
                        best_move = (sc, st, ww, sorted(trial))
            return best_move

        # Best-known solution starts at the incumbent's exact evaluation
        best_scalar, best_step, best_w = inc_scalar, inc_score, inc_w
        best_ind_list = list(best.long_indices)

        # Multi-start queue: incumbent first, then distinct elite subsets
        queue = [(inc_scalar, inc_score, inc_w, list(best.long_indices))]
        seen = {tuple(sorted(best.long_indices))}
        for cand in (start_pool or []):
            if len(queue) >= 1 + TUNING.local_search_seeds:  # incumbent + elite seeds
                break
            key = tuple(sorted(cand.long_indices))
            if key in seen:
                continue
            seen.add(key)
            out = _eval_subset(list(key))
            n_evals += 1
            if out is not None:
                queue.append((out[0], out[1], out[2], list(key)))

        for scalar, score, w, ind_list in queue:
            if time.time() - t0 > time_budget:
                break
            r_scalar, r_score, r_w, r_ind = _descend_1swap(ind_list, scalar, score, w)
            if r_scalar > best_scalar + 1e-12:
                best_scalar, best_step, best_w, best_ind_list = r_scalar, r_score, r_w, r_ind
                improved_any = True

        # 2-swap escape loop: escape, re-descend, until stalled or budget out
        while time.time() - t0 <= time_budget:
            move = _best_2swap(best_ind_list, best_scalar)
            if move is None:
                break
            best_scalar, best_step, best_w, best_ind_list = move
            improved_any = True
            r_scalar, r_score, r_w, r_ind = _descend_1swap(
                best_ind_list, best_scalar, best_step, best_w)
            if r_scalar > best_scalar + 1e-12:
                best_scalar, best_step, best_w, best_ind_list = r_scalar, r_score, r_w, r_ind

        # Random-restart diversification (large universes only — small ones
        # were already solved exactly above).  The GA + elite starts can share
        # one basin on near-homogeneous universes; independent random restarts
        # give the 1-swap descent fresh basins to fall into.  Deterministic
        # (self._rng, seeded), time-bounded, best-improvement (never regresses).
        if (_total_subsets > TUNING.local_search_exhaustive_max_subsets
                and TUNING.local_search_random_restarts > 0):
            for _r in range(TUNING.local_search_random_restarts):
                if time.time() - t0 > time_budget:
                    break
                _lo_r, _hi_r = self._long_size_bounds(len(_free_pool_ls))
                _n_r = self._rng.randint(_lo_r, _hi_r)
                trial = self._pick_long_subset(_free_pool_ls, _n_r)
                out = _eval_subset(trial)
                n_evals += 1
                if out is None:
                    continue
                r_scalar, r_score, r_w, r_ind = _descend_1swap(
                    list(trial), out[0], out[1], out[2])
                if r_scalar > best_scalar + 1e-12:
                    best_scalar, best_step, best_w, best_ind_list = r_scalar, r_score, r_w, r_ind
                    improved_any = True

        if improved_any:
            best = self._copy(best)
            best.long_indices = list(best_ind_list)
            best.long_weights = best_w
            best.fitness = best_step
            return best, max(best_score, best_step)
        # No subset move — still adopt the exact weights for the incumbent subset
        if inc_score >= best_score - 1e-12:
            best.long_weights = inc_w
            best.fitness = inc_score
            return best, inc_score
        return best, best_score

    def _adaptive_net_pnl(self, long_pos, long_w, short_pos=None, short_w=None):
        """Compute net PnL using adaptive reweight formula — mirrors _compute_net_pnl exactly.

        For ADAPTIVE_REWEIGHT policy:
            Delegates to shared `adaptive_pnl()` — single source of truth.

        For FILL_ZERO policy: plain nan_to_num @ w (original behavior).
        For DROP_INCOMPLETE: only rows where ALL selected are valid.

        Returns: net_pnl array of shape (n_rows,) [or fewer for DROP mode]
        """
        use_adaptive = (self.missing_data_policy == MissingDataPolicy.ADAPTIVE_REWEIGHT)
        use_drop = (self.missing_data_policy == MissingDataPolicy.DROP_INCOMPLETE_DAYS)

        long_pos = np.asarray(long_pos, dtype=int)
        long_w = np.asarray(long_w, dtype=np.float64)

        if use_adaptive:
            # Shared canonical function (operates on nan_to_num'd matrix + valid_mask)
            L = adaptive_pnl(self._ts_mat, long_pos, long_w, self._active_mask)

            # Short leg adaptive (if any)
            if short_pos is not None and len(short_pos) > 0:
                short_pos = np.asarray(short_pos, dtype=int)
                short_w = np.asarray(short_w, dtype=np.float64)
                S = adaptive_pnl(self._ts_mat, short_pos, short_w, self._active_mask)
            else:
                S = np.zeros(self._n_rows)

        elif use_drop:
            all_pos = np.concatenate([long_pos, short_pos]) if (short_pos is not None and len(short_pos) > 0) else long_pos
            valid_rows = ~np.isnan(self._orig_ts_mat[:, all_pos]).any(axis=1)
            if valid_rows.sum() < TUNING.min_valid_days:
                return np.zeros(0)
            L = self._orig_ts_mat[valid_rows][:, long_pos] @ long_w
            if short_pos is not None and len(short_pos) > 0:
                short_pos = np.asarray(short_pos, dtype=int)
                short_w = np.asarray(short_w, dtype=np.float64)
                S = self._orig_ts_mat[valid_rows][:, short_pos] @ short_w
            else:
                S = np.zeros(int(valid_rows.sum()))
            net = L - S  # short basket = sold legs, subtracted (same as backtester)
            if self.global_cap < 9999998 or self.global_floor > -9999998:
                net = np.clip(net, self.global_floor, self.global_cap)
            return net

        else:
            # FILL_ZERO: plain matmul (original behavior)
            L = self._ts_mat[:, long_pos] @ long_w
            if short_pos is not None and len(short_pos) > 0:
                short_pos = np.asarray(short_pos, dtype=int)
                short_w = np.asarray(short_w, dtype=np.float64)
                S = self._ts_mat[:, short_pos] @ short_w
            else:
                S = np.zeros(self._n_rows)

        # Short basket = sold legs → subtracted, in EVERY mode (cross-corridor
        # included: a short cross-corridor row is a sold (mono−cross) leg).
        # Matches the backtester (run(): result = long_pnl + short_pnl with
        # short_pnl already negated) and _net_strike (net -= short strikes).
        net = L - S
        if self.global_cap < 9999998 or self.global_floor > -9999998:
            net = np.clip(net, self.global_floor, self.global_cap)
        return net

    def _equal_projected_weights(self, indices, candidates) -> Optional[np.ndarray]:
        """Deterministic GA weights for non-exact configs (and the short leg):
        equal-weight projected onto the bounded simplex (sum=1, per-name
        Min/Max).  Returns None when the subset's bounds are infeasible."""
        n = len(indices)
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        min_w = np.array([candidates[i].min_weight for i in indices], dtype=np.float64)
        max_w = np.array([candidates[i].max_weight for i in indices], dtype=np.float64)
        if min_w.sum() > 1.0 + 1e-10 or max_w.sum() < 1.0 - 1e-10:
            return None
        w = project_to_bounded_simplex(np.full(n, 1.0 / n), min_w, max_w)
        if (
                w is None
                or not np.all(np.isfinite(w))
                or not np.isclose(w.sum(), 1.0, atol=1e-8)
                or np.any(w < min_w - 1e-8)
                or np.any(w > max_w + 1e-8)
        ):
            return None
        return w

    def _fitness(self, individual: _Individual) -> float:
        """Bilevel fitness — the genome carries ONLY the stock subsets.

        Weights are ALWAYS derived here, from one deterministic rule per
        config (single source of truth, cached on the individual):
          - exact-path configs (all-linear, min_payoff-only, concave blend):
            the exact inner solver (LP / bisection, memoised per subset) —
            fitness is a pure function of the subset;
          - any other config: bounded equal-weight projection — cheap and
            deterministic.  SLSQP runs only post-GA (_refine_top_individuals,
            local search, safety-net).
        The short leg always uses the equal-weight projection (it never had a
        solver-driven inner optimisation).
        """
        if not (self._use_new_scoring and self._score_fn is not None
                and self._score_fn.is_fitted):
            raise RuntimeError(
                "ScoreFunction not fitted — reference sample failed; cannot optimize. "
                f"use_new_scoring={self._use_new_scoring}, "
                f"score_fn={'None' if self._score_fn is None else 'present'}, "
                f"is_fitted={self._score_fn.is_fitted if self._score_fn is not None else 'N/A'}"
            )

        # ── Long leg: resolve matrix columns (all must be present) ──
        _keys, long_pos, _subset_strikes, _subset_bounds = self._subset_arrays(
            individual.long_indices)
        if long_pos is None:
            return 0.0

        # ── Bucket cardinality bounds on the subset ──
        if self._bucket_count_bounds and not self._bucket_counts_ok(individual.long_indices):
            self._rejection_reasons["bucket_counts"] += 1
            return 0.0
        _groups = self._bucket_groups_for(individual.long_indices)
        _vega_sp = self._vega_spec_for(individual.long_indices)

        # ── Long weights: deterministic function of the subset ──
        if self._exact_fitness and self._weight_solver is not None:
            strikes = _subset_strikes
            _pass_strikes = (self.c.max_net_strike > 0) or self._ws_active()
            per_stock_bounds = _subset_bounds
            # Strike visibility log — first call only
            if not self._strike_logged:
                self._strike_logged = True
                self.log("INFO", f"[STRIKE-CHECK] strikes.min={strikes.min():.4f}, strikes.max={strikes.max():.4f}, max_net_strike={self.c.max_net_strike}")
            result = self._weight_solver.solve(
                pnl_matrix=self._ts_mat,
                stock_indices=long_pos,
                strikes=strikes if _pass_strikes else None,
                per_stock_bounds=per_stock_bounds,
                active_mask=self._active_mask,
                group_bounds=_groups,
                vega=_vega_sp,
            )
            if not result.feasible:
                self._lp_infeasible_streak += 1
                # Units-mismatch fail-fast only when nothing else can explain a
                # long infeasible streak (bucket/vega constraints legitimately
                # make many subsets infeasible, e.g. size-2 under bucket caps)
                if (self._lp_infeasible_streak >= TUNING.infeasible_streak_limit
                        and not self._bucket_constraints and self._vega is None):
                    raise RuntimeError(
                        f"First {TUNING.infeasible_streak_limit} exact inner solves all infeasible — likely units mismatch. "
                        f"strikes.min={strikes.min():.4f}, strikes.max={strikes.max():.4f}, "
                        f"max_net_strike={self.c.max_net_strike}"
                    )
                return 0.0
            self._lp_infeasible_streak = 0
            long_w = result.weights
            _solve_extra = result.extra
        else:
            _solve_extra = None
            long_w = self._equal_projected_weights(individual.long_indices, self.long_candidates)
            if long_w is None:
                self._rejection_reasons["no_weights"] += 1
                return 0.0
            # Bucket WEIGHT bounds: L1-project the equal-weight point into the
            # group box (deterministic LP) when the plain projection violates it
            if _groups and not self._weight_solver._groups_satisfied(long_w, _groups):
                lb = np.array([self.long_candidates[i].min_weight
                               for i in individual.long_indices], dtype=np.float64)
                ub = np.array([self.long_candidates[i].max_weight
                               for i in individual.long_indices], dtype=np.float64)
                long_w = self._weight_solver.project_to_group_feasible(long_w, lb, ub, _groups)
                if long_w is None:
                    self._rejection_reasons["bucket_weights"] += 1
                    return 0.0
        individual.long_weights = long_w

        # ── Short leg: bounded equal-weight projection ──
        short_pos_arr = None
        short_w = np.zeros(0, dtype=np.float64)
        if individual.short_indices:
            short_ids = [_candidate_key(self.short_candidates[i], self.is_cross_corridor)
                         for i in individual.short_indices]
            spos = np.array([self._col_pos[c] for c in short_ids if c in self._col_pos])
            if len(spos) != len(individual.short_indices):
                return 0.0
            short_w = self._equal_projected_weights(individual.short_indices, self.short_candidates)
            if short_w is None:
                self._rejection_reasons["no_weights"] += 1
                return 0.0
            short_pos_arr = spos
        individual.short_weights = short_w

        # ── Hard strike constraint on the derived weights ──
        if not self._is_strike_valid(individual.long_indices, long_w,
                                     individual.short_indices, short_w):
            self._rejection_reasons["strike_invalid"] += 1
            return 0.0

        # ── Score the net P&L (adaptive formula mirrors the backtester) ──
        net_pnl_raw = self._adaptive_net_pnl(
            long_pos, long_w, short_pos_arr,
            short_w if short_pos_arr is not None else None)
        n_nonzero = int(np.count_nonzero(net_pnl_raw))
        if n_nonzero < TUNING.min_valid_days:
            self._rejection_reasons["len_valid_lt_50"] += 1
            return 0.0
        _ws = (self._net_strike(individual.long_indices, long_w,
                                individual.short_indices, short_w)
               if self._ws_active() else None)
        _ax_c, _ax_r = (None, None)
        if self._vega is not None:
            _ax_c, _ax_r = self._axe_ctx_values(individual.long_indices, long_w, _solve_extra)
            if _ax_c is None and _ax_r is None:
                # caps leave no feasible V for this subset's weights
                self._rejection_reasons["vega_infeasible"] += 1
                return 0.0
        ctx = self._make_ctx(_ws, _ax_c, _ax_r)
        fit = self._fitness_from_net(net_pnl_raw, ctx)
        if fit is None:
            self._rejection_reasons["invalid_score"] += 1
            return 0.0
        return fit

    def _fitness_from_net(self, net_pnl_raw: np.ndarray, ctx: ScoreContext) -> Optional[float]:
        """Step-quantile score with the top-saturation tie-break.

        Shared by the GA fitness and the refinement acceptance so both rank
        candidates on the SAME scale.  Tie-break: when the quantile score
        saturates at the top (>= 0.99), rank by the scalarised raw objective
        mapped into (1.0, 3.0) — beats any quantile score, monotone in the
        raw blend.  Returns None for an invalid (NaN/inf) score.
        """
        score = self._score_fn.score(net_pnl_raw, ctx)
        if np.isnan(score) or np.isinf(score):
            return None
        if score >= TUNING.tiebreak_threshold:
            raw = self._score_fn.raw_metrics(net_pnl_raw, ctx)
            lam = self._score_fn.weights
            metric_map = {m.name: m for m in self._score_fn.metrics}
            scalar = sum(
                lam.get(name, 0.0) * raw[name] * (1.0 if metric_map[name].higher_is_better else -1.0)
                for name in lam.active_names
            )
            return float(2.0 + math.tanh(scalar))
        return float(score)


    # ══════════════════════════════════════════════════════════════════════════
    # BILEVEL SCORING (new modular system)
    # ══════════════════════════════════════════════════════════════════════════

    def _compute_net_pnl(self, long_indices, long_w, short_indices, short_w) -> Optional[np.ndarray]:
        """Compute net P&L array for given stocks and weights (shared logic).
        Returns the valid (non-zero) net P&L array, or None if insufficient data.
        Uses the SAME NaN policy as _calculate_fitness to ensure invariants.
        """
        # ── Use _candidate_key for cross-corridor vs mono ──
        long_ids = [_candidate_key(self.long_candidates[i], self.is_cross_corridor) for i in long_indices]
        short_ids = [_candidate_key(self.short_candidates[i], self.is_cross_corridor) for i in short_indices] if short_indices else []
        long_pos = [self._col_pos[c] for c in long_ids if c in self._col_pos]
        short_pos = [self._col_pos[c] for c in short_ids if c in self._col_pos]
        if len(long_pos) == 0 or self._ts_mat.size == 0:
            return None
        long_w = np.asarray(long_w, dtype=np.float64)
        short_w = np.asarray(short_w, dtype=np.float64) if len(short_pos) else np.zeros(0)
        if len(long_w) != len(long_pos):
            long_w = long_w[:len(long_pos)]
            s = long_w.sum()
            if s > 0:
                long_w = long_w / s
        if len(short_w) != len(short_pos):
            short_w = short_w[:len(short_pos)]
            s = short_w.sum()
            if s > 0:
                short_w = short_w / s
        use_adaptive = (self.missing_data_policy == MissingDataPolicy.ADAPTIVE_REWEIGHT)
        use_drop = (self.missing_data_policy == MissingDataPolicy.DROP_INCOMPLETE_DAYS)
        if use_adaptive:
            L_mat = self._orig_ts_mat[:, long_pos]
            L_nan = ~self._active_mask[:, long_pos]   # inactive = out of the denominator
            L_w_matrix = np.tile(long_w, (self._n_rows, 1))
            L_w_matrix[L_nan] = 0.0
            L_w_sums = L_w_matrix.sum(axis=1, keepdims=True)
            L_w_sums[L_w_sums == 0] = 1.0
            L_w_matrix = L_w_matrix / L_w_sums
            L = (np.nan_to_num(L_mat, nan=0.0) * L_w_matrix).sum(axis=1)
            if len(short_pos) > 0:
                S_mat = self._orig_ts_mat[:, short_pos]
                S_nan = ~self._active_mask[:, short_pos]
                S_w_matrix = np.tile(short_w, (self._n_rows, 1))
                S_w_matrix[S_nan] = 0.0
                S_w_sums = S_w_matrix.sum(axis=1, keepdims=True)
                S_w_sums[S_w_sums == 0] = 1.0
                S_w_matrix = S_w_matrix / S_w_sums
                S = (np.nan_to_num(S_mat, nan=0.0) * S_w_matrix).sum(axis=1)
            else:
                S = 0.0
            valid_rows = ~L_nan.all(axis=1)
        elif use_drop:
            all_pos = long_pos + short_pos
            combined = self._orig_ts_mat[:, all_pos]
            valid_rows = ~np.isnan(combined).any(axis=1)
            if valid_rows.sum() < TUNING.min_valid_days:
                return None
            L = self._orig_ts_mat[valid_rows][:, long_pos] @ long_w
            S = (self._orig_ts_mat[valid_rows][:, short_pos] @ short_w) if len(short_pos) else 0.0
        else:
            L = self._ts_mat[:, long_pos] @ long_w
            S = (self._ts_mat[:, short_pos] @ short_w) if len(short_pos) else 0.0
            valid_rows = np.ones(self._n_rows, dtype=bool)
        if valid_rows.sum() < TUNING.min_valid_days:
            return None
        net = L - S  # short basket = sold legs, subtracted (same as backtester)
        if self.global_cap < 9999998 or self.global_floor > -9999998:
            net = np.clip(net, self.global_floor, self.global_cap)
        if not use_drop:
            net = net[valid_rows]
        nonzero_mask = net != 0.0
        n_trailing_zeros = 0
        for i in range(len(net) - 1, -1, -1):
            if net[i] == 0.0:
                n_trailing_zeros += 1
            else:
                break
        if nonzero_mask.sum() < TUNING.min_valid_days:
            return None
        # ── ALIGNED WITH BACKTESTER: return full series including zeros ──
        # Cross-corridor backtester keeps all rows. Zeros are valid observations.
        return net

    def _build_reference_sample(self, n_samples: int = 2000) -> None:
        """Build the reference sample for quantile normalisation.
        Generates random feasible baskets, computes their net P&L, and fits
        the ScoreFunction normalizer.  Called once before the GA loop.
        """
        ref_start = time.time()
        self.log("INFO", f"Building reference sample ({n_samples} baskets)...")
        ctx = ScoreContext(n_days=self._n_rows)
        # ── Shared-calibration cache: reuse the sampled baskets when another
        # config with the SAME (seed, sample size, extras, vega-mode) already
        # generated them, restoring the exact post-generation RNG states so
        # every downstream draw matches a solo run bit-for-bit. ──
        _ref_key = None
        if self._reference_cache is not None:
            _extras_sig = tuple(sorted(
                set(self._score_fn.weights.active_names)
                & {"weighted_strike", "axe_book_cleaned", "axe_package_recycled"}))
            _ref_key = (int(self.seed), int(n_samples), _extras_sig,
                        self._vega is not None)
        _hit = self._reference_cache.get(_ref_key) if _ref_key is not None else None
        if _hit is not None:
            sample_pnls = _hit["sample_pnls"]
            sample_extras = _hit["sample_extras"]
            attempts = _hit["attempts"]
            self._rng.setstate(_hit["rng_state"])
            self._np_rng.bit_generator.state = _hit["np_state"]
            self._vega_rng.setstate(_hit["vega_state"])
            self.log("INFO", f"✅ Calibration reused from shared cache ({len(sample_pnls)} baskets)")
        else:
            sample_pnls, sample_extras, attempts = self._generate_reference_sample(n_samples, ctx)
            if _ref_key is not None:
                self._reference_cache[_ref_key] = {
                    "sample_pnls": sample_pnls,
                    "sample_extras": sample_extras,
                    "attempts": attempts,
                    "rng_state": self._rng.getstate(),
                    "np_state": self._np_rng.bit_generator.state,
                    "vega_state": self._vega_rng.getstate(),
                }
        if len(sample_pnls) < TUNING.min_reference_size:
            self.log("WARN", f"⚠️ Reference sample too small ({len(sample_pnls)}/{n_samples} after {attempts} attempts), falling back to legacy scoring")
            self._use_new_scoring = False
            self._scoring_mode = "legacy"
            return
        # FIT the score function with collected samples (extras carry per-sample
        # weighted strikes so the strike objective normalises like any metric)
        self._score_fn.build_reference(sample_pnls, ctx, sample_extras=sample_extras)
        # Non-degeneracy for extras-carried ACTIVE metrics (weighted_strike,
        # axe criteria): a constant reference saturates the rank normalizer
        # (every candidate scores 0 under strictly-less tie semantics)
        _active_extras = (set(self._score_fn.weights.active_names)
                          & {"weighted_strike", "axe_book_cleaned", "axe_package_recycled"})
        for _name in sorted(_active_extras):
            _vals = np.array([e[_name] for e in sample_extras if _name in e], dtype=np.float64)
            _vals = _vals[np.isfinite(_vals)]
            if len(_vals) > 10 and np.std(_vals) < 1e-10:
                raise RuntimeError(
                    f"Reference sample non-degenerate check failed: metric '{_name}' "
                    f"has zero spread (all values ≈ {_vals[0]:.6f}) across the sampled "
                    f"baskets — the quantile normalizer would score every candidate 0. "
                    f"Vary the inputs (targets/caps/strikes) or deactivate the metric."
                )
        self._reference_size = len(sample_pnls)
        self._ref_n_samples = int(n_samples)
        # Sanity: verify max_drawdown (and all metrics) have non-degenerate spread
        # If std==0 for any active metric, the quantile normalizer is saturated
        for m in self._score_fn.metrics:
            if m.name not in self._score_fn.weights.active_names:
                continue
            vals = np.array([m.compute(p, ctx) for p in sample_pnls], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if len(vals) > 10 and np.std(vals) < 1e-10:
                # DEBUG: Print first 10 baskets for root cause analysis
                for i in range(min(10, len(sample_pnls))):
                    pnl = sample_pnls[i]
                    last_val = float(np.mean(pnl[-1:]))
                    mean_ret = float(np.mean(pnl))
                    cumsum = float(np.sum(pnl))
                raise RuntimeError(
                    f"Reference sample non-degenerate check failed: metric '{m.name}' "
                    f"has zero spread (all values ≈ {vals[0]:.6f}). The quantile normalizer "
                    f"will saturate and this metric will have no effect on scoring. "
                    f"Check that the weight strategy mix produces diverse baskets."
                )
        ref_elapsed = time.time() - ref_start
        self.log("INFO", f"✅ Reference sample fitted ({len(sample_pnls)} baskets, {attempts} attempts) → bilevel scoring active [{ref_elapsed:.1f}s]")


    def _generate_reference_sample(self, n_samples: int, ctx) -> tuple:
        """Draw the reference baskets (consumes the run RNG streams).
        Returns (sample_pnls, sample_extras, attempts)."""
        sample_pnls = []
        sample_extras = []  # aligned with sample_pnls: {"weighted_strike": ...}
        attempts = 0
        max_attempts = n_samples * TUNING.reference_attempts_factor
        _free_pool_ref = [i for i in range(len(self.long_candidates))
                          if i not in self._forced_long]

        # Track rejection reasons
        rejection_counts = {}

        while len(sample_pnls) < n_samples and attempts < max_attempts:
            attempts += 1
            # Generate random subset (always containing forced names)
            _lo, _hi = self._long_size_bounds(len(_free_pool_ref))
            n_long = self._rng.randint(_lo, _hi)
            long_indices = self._pick_long_subset(_free_pool_ref, n_long)

            # Mix weight strategies so reference spans the same distribution as GA evaluations:
            # 40% equal weights, 30% QP-diversified (penalizes corners), 20% greedy-spread, 10% Dirichlet
            r = self._rng.random()
            if r < TUNING.mix_equal:
                long_w = np.full(n_long, 1.0 / n_long)
            elif r < TUNING.mix_qp:
                # QP-optimal on column means with diversification penalty (per-stock bounds)
                long_pos = []
                for idx in long_indices:
                    t = _candidate_key(self.long_candidates[idx], self.is_cross_corridor)
                    if t in self._col_pos:
                        long_pos.append(self._col_pos[t])
                if len(long_pos) == n_long:
                    sub_pnl = self._ts_mat[:, long_pos]
                    col_means = sub_pnl.mean(axis=0)
                    # Per-stock bounds
                    lb = np.array([self.long_candidates[i].min_weight for i in long_indices])
                    ub = np.array([self.long_candidates[i].max_weight for i in long_indices])
                    long_w = _qp_diversified_weights(col_means, lb, ub)
                else:
                    long_w = np.full(n_long, 1.0 / n_long)
            elif r < TUNING.mix_greedy:
                # Greedy with diversification: bias toward best but spread remainder
                long_pos = []
                for idx in long_indices:
                    t = _candidate_key(self.long_candidates[idx], self.is_cross_corridor)
                    if t in self._col_pos:
                        long_pos.append(self._col_pos[t])
                if len(long_pos) == n_long:
                    sub_pnl = self._ts_mat[:, long_pos]
                    col_means = sub_pnl.mean(axis=0)
                    lb = np.array([self.long_candidates[i].min_weight for i in long_indices])
                    ub = np.array([self.long_candidates[i].max_weight for i in long_indices])
                    # Give best stock 50-75% of its max (not full max)
                    best_col = np.argmax(col_means)
                    long_w = lb.copy()
                    best_alloc = lb[best_col] + TUNING.greedy_best_frac * (ub[best_col] - lb[best_col])
                    long_w[best_col] = best_alloc
                    remaining = 1.0 - long_w.sum()
                    if remaining > 0:
                        # Spread remainder proportional to mean quality
                        rank = np.argsort(-col_means)
                        quality_sum = max(1e-9, sum(max(0, col_means[j]) for j in rank if j != best_col))
                        for idx2 in rank:
                            if idx2 == best_col:
                                continue
                            share = max(0, col_means[idx2]) / quality_sum
                            add = min(remaining, ub[idx2] - long_w[idx2], remaining * share * TUNING.greedy_spread_boost)
                            long_w[idx2] += add
                            remaining -= add
                            if remaining < 1e-9:
                                break
                        # Distribute any leftover evenly
                        if remaining > 1e-9:
                            for idx2 in rank:
                                if idx2 == best_col:
                                    continue
                                add = min(remaining / max(1, n_long - 1), ub[idx2] - long_w[idx2])
                                long_w[idx2] += add
                                remaining -= add
                    s = long_w.sum()
                    if s > 0:
                        long_w = long_w / s
                else:
                    long_w = np.full(n_long, 1.0 / n_long)
            else:
                # Random Dirichlet with per-stock bounds
                long_w = self._np_rng.dirichlet(np.ones(n_long))
                lb = np.array([self.long_candidates[i].min_weight for i in long_indices])
                ub = np.array([self.long_candidates[i].max_weight for i in long_indices])
                long_w = np.clip(long_w, lb, ub)
                s = long_w.sum()
                if s > 0:
                    long_w = long_w / s
            if self._long_only:
                short_indices = []
                short_w = np.zeros(0)
            else:
                n_short = self._rng.randint(
                    self.c.min_stocks_short,
                    min(self.c.max_stocks_short, len(self.short_candidates))
                )
                short_indices = self._rng.sample(range(len(self.short_candidates)), n_short)
                short_w = np.full(n_short, 1.0 / n_short)

            # Compute net P&L for this basket
            net_pnl = self._compute_net_pnl(long_indices, long_w, short_indices, short_w)

            if net_pnl is None:
                reason = "net_pnl is None"
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            elif len(net_pnl) < TUNING.min_valid_days:
                reason = "len < 50"
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            else:
                sample_pnls.append(net_pnl)
                _extra = {
                    "weighted_strike": self._net_strike(
                        long_indices, long_w, short_indices, short_w)
                }
                _extra.update(self._reference_axe_extras(long_indices, long_w))
                sample_extras.append(_extra)

        # --- ELITE SEEDING: inject baskets from top-ranked stocks so normalizer ceiling is realistic ---
        all_col_means = []
        for i, cand in enumerate(self.long_candidates):
            key = _candidate_key(cand, self.is_cross_corridor)
            if key in self._col_pos:
                col = self._col_pos[key]
                all_col_means.append((i, self._ts_mat[:, col].mean()))
        all_col_means.sort(key=lambda x: -x[1])  # descending by mean
        n_elite_baskets = min(TUNING.reference_elite_cap, n_samples // 10)
        for offset in range(n_elite_baskets):
            _lo_e, _hi_e = self._long_size_bounds(len(_free_pool_ref))
            n_long = self._rng.randint(_lo_e, _hi_e)
            top_pool = min(len(all_col_means), n_long + offset + 5)
            elite_free = [x[0] for x in all_col_means[:top_pool]
                          if x[0] not in self._forced_long]
            if len(elite_free) + len(self._forced_long) < n_long:
                continue
            long_indices_e = self._pick_long_subset(elite_free, n_long)
            long_pos = []
            for idx in long_indices_e:
                key = _candidate_key(self.long_candidates[idx], self.is_cross_corridor)
                if key in self._col_pos:
                    long_pos.append(self._col_pos[key])
            if len(long_pos) == n_long:
                sub_pnl = self._ts_mat[:, long_pos]
                col_means = sub_pnl.mean(axis=0)
                lb = np.array([self.long_candidates[i].min_weight for i in long_indices_e])
                ub = np.array([self.long_candidates[i].max_weight for i in long_indices_e])
                long_w_e = _qp_diversified_weights(col_means, lb, ub)
                s = long_w_e.sum()
                if s > 0:
                    long_w_e = long_w_e / s
            else:
                long_w_e = np.full(n_long, 1.0 / n_long)
            if self._long_only:
                short_indices_e = []
                short_w_e = np.zeros(0)
            else:
                short_indices_e = self._rng.sample(range(len(self.short_candidates)),
                    min(self.c.min_stocks_short, len(self.short_candidates)))
                short_w_e = np.full(max(len(short_indices_e), 1), 1.0 / max(len(short_indices_e), 1)) if short_indices_e else np.zeros(0)

            elite_pnl = self._compute_net_pnl(long_indices_e, long_w_e, short_indices_e, short_w_e)
            if elite_pnl is not None and len(elite_pnl) >= TUNING.min_valid_days:
                sample_pnls.append(elite_pnl)
                _extra_e = {
                    "weighted_strike": self._net_strike(
                        long_indices_e, long_w_e, short_indices_e, short_w_e)
                }
                _extra_e.update(self._reference_axe_extras(long_indices_e, long_w_e))
                sample_extras.append(_extra_e)
        return sample_pnls, sample_extras, attempts

    def _ws_active(self) -> bool:
        """True when the weighted_strike objective carries a positive weight."""
        return (self._metric_weights is not None
                and "weighted_strike" in self._metric_weights.active_names)

    def _make_ctx(self, weighted_strike: Optional[float] = None,
                  axe_cleaned: Optional[float] = None,
                  axe_recycled: Optional[float] = None) -> ScoreContext:
        """Run-level ScoreContext, optionally carrying the basket's net strike
        and (vega mode) the axe recycling criteria."""
        return ScoreContext(n_days=self._n_rows, weighted_strike=weighted_strike,
                            axe_cleaned=axe_cleaned, axe_recycled=axe_recycled)

    # ══════════════════════════════════════════════════════════════════════════
    # CONSTRAINTS
    # ══════════════════════════════════════════════════════════════════════════

    def _solver_strikes(self, indices, candidates) -> np.ndarray:
        """Build per-leg strike vector for WeightSolver constraint.
        XC: net spread = strike_mono_var_swap - strike_cross_corridor per leg.
        Standard: strike_mono_var_swap (weighted average constrained directly).
        """
        if self.is_cross_corridor:
            out = np.empty(len(indices), dtype=np.float64)
            for k, i in enumerate(indices):
                s = candidates[i]
                idx = s.strike_cross_corridor if s.strike_cross_corridor is not None else s.strike_mono_var_swap
                out[k] = s.strike_mono_var_swap - idx
                if idx is None and not getattr(self, '_xc_strike_warned', False):
                    self._xc_strike_warned = True
                    self.log("WARN", f"[STRIKE] {s.variance_asset}: strike_cross_corridor is None, using mono as fallback")
            return out
        return np.array([candidates[i].strike_mono_var_swap for i in indices], dtype=np.float64)

    def _net_strike(self, long_indices, long_w, short_indices, short_w) -> float:
        """Compute weighted-average net strike.
        Cross-corridor convention: net_strike_i = strike_mono_var_swap_i - strike_cross_corridor_i
        (spread between stock vol and index vol). Same for standard mode.
        """
        if self.is_cross_corridor:
            net = 0.0
            for i, w in zip(long_indices, long_w):
                s = self.long_candidates[i]
                idx_strike = s.strike_cross_corridor if s.strike_cross_corridor is not None else s.strike_mono_var_swap
                net += (s.strike_mono_var_swap - idx_strike) * float(w)
            if short_indices:
                for i, w in zip(short_indices, short_w):
                    s = self.short_candidates[i]
                    idx_strike = s.strike_cross_corridor if s.strike_cross_corridor is not None else s.strike_mono_var_swap
                    net -= (s.strike_mono_var_swap - idx_strike) * float(w)
            return net
        ls = np.array([self.long_candidates[i].strike_mono_var_swap for i in long_indices], dtype=np.float64)
        net = float(ls @ np.asarray(long_w, dtype=np.float64))
        if short_indices:
            ss = np.array([self.short_candidates[i].strike_mono_var_swap for i in short_indices], dtype=np.float64)
            net -= float(ss @ np.asarray(short_w, dtype=np.float64))
        return net

    def _is_strike_valid(self, long_indices, long_w, short_indices, short_w) -> bool:
        if self.c.max_net_strike <= 0:
            return True
        net = self._net_strike(long_indices, long_w, short_indices, short_w)
        return abs(net) <= self.c.max_net_strike
    # ══════════════════════════════════════════════════════════════════════════
    # GA OPERATORS (original behavior)
    # ══════════════════════════════════════════════════════════════════════════

    def _pick_long_subset(self, pool: List[int], n_long: int) -> List[int]:
        """Pick a long subset of size n_long from pool, always containing the
        forced names and honouring per-bucket count bounds.  ``pool`` must
        not contain the forced indices.  When the pool cannot satisfy a
        bucket minimum (crossover unions may lack bucket names), the deficit
        is drawn from that bucket's candidates outside the pool."""
        if self._bucket_count_bounds:
            picked = set(self._forced_long)
            # 1) satisfy each constrained bucket's minimum
            for bucket in sorted(self._bucket_count_bounds):
                lo, _hi = self._bucket_count_bounds[bucket]
                need = lo - self._bucket_count(picked, bucket)
                if need <= 0:
                    continue
                in_pool = [i for i in pool
                           if i not in picked and self._bucket_of[i] == bucket]
                if len(in_pool) < need:
                    in_pool += [i for i in self._bucket_members.get(bucket, [])
                                if i not in picked and i not in in_pool]
                take = min(need, len(in_pool))
                if take > 0:
                    picked |= set(self._rng.sample(sorted(in_pool), take))
            # 2) fill up to n_long, skipping buckets at their cap
            while len(picked) < n_long:
                avail = []
                for i in pool:
                    if i in picked:
                        continue
                    b = self._bucket_of[i]
                    if b is not None and b in self._bucket_count_bounds:
                        hi_b = self._bucket_count_bounds[b][1]
                        if hi_b is not None and self._bucket_count(picked, b) >= hi_b:
                            continue
                    avail.append(i)
                if not avail:
                    break
                picked.add(self._rng.choice(sorted(avail)))
            return sorted(picked)
        forced = self._forced_long
        n_free = n_long - len(forced)
        if n_free < 0:
            return list(forced[:n_long])  # cannot happen after __init__ validation
        if n_free > len(pool):
            n_free = len(pool)
        picked = self._rng.sample(pool, n_free) if n_free > 0 else []
        return sorted(set(picked) | set(forced))

    def _long_size_bounds(self, pool_extra: int) -> Tuple[int, int]:
        """[lo, hi] basket sizes honouring cardinality inputs, forced names
        and per-bucket count bounds.
        ``pool_extra`` = number of NON-forced candidates available."""
        k = len(self._forced_long)
        lo = max(self.c.min_stocks_long, k)
        hi = min(self.c.max_stocks_long, k + pool_extra)
        if self._bucket_count_bounds:
            eff_min = 0
            for bucket, (blo, _bhi) in self._bucket_count_bounds.items():
                forced_in = sum(1 for i in self._forced_long
                                if self._bucket_of[i] == bucket)
                eff_min += max(blo, forced_in)
            eff_min += sum(1 for i in self._forced_long
                           if (self._bucket_of[i] is None
                               or self._bucket_of[i] not in self._bucket_count_bounds))
            lo = max(lo, eff_min)
            capped_total = 0
            for b, members in self._bucket_members.items():
                cap = len(members)
                if b in self._bucket_count_bounds and self._bucket_count_bounds[b][1] is not None:
                    cap = min(cap, self._bucket_count_bounds[b][1])
                capped_total += cap
            capped_total += sum(1 for bb in self._bucket_of if bb is None)
            hi = min(hi, capped_total)
        return lo, max(lo, hi)

    def _create_random_individual(self) -> Optional[_Individual]:
        """Create an individual — the genome is ONLY the stock subsets;
        weights are derived deterministically inside _fitness."""
        max_attempts = TUNING.create_attempts
        _free_pool = [i for i in range(len(self.long_candidates)) if i not in self._forced_long]
        for _ in range(max_attempts):
            _lo, _hi = self._long_size_bounds(len(_free_pool))
            n_long = self._rng.randint(_lo, _hi)
            long_indices = self._pick_long_subset(_free_pool, n_long)
            if self._long_only:
                short_indices = []
            else:
                n_short = self._rng.randint(
                    self.c.min_stocks_short,
                    min(self.c.max_stocks_short, len(self.short_candidates))
                )
                short_indices = self._rng.sample(range(len(self.short_candidates)), n_short)
            ind = _Individual(long_indices, short_indices)
            ind.fitness = self._fitness(ind)
            if ind.fitness <= 0.0:
                self._rejection_reasons["fitness<=0"] += 1
                continue
            return ind
        return None

    def _tournament_select(self, population: List[_Individual]) -> _Individual:
        tournament = self._rng.sample(population, min(self.c.tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness)

    def _crossover(self, parent1: _Individual, parent2: _Individual) -> Optional[_Individual]:
        """Crossover on SUBSETS only: union of parent indices, then a random
        subset draw (forced names always kept).  Weights are not inherited —
        they are a deterministic function of the child's subset (_fitness)."""
        l_union_free = list((set(parent1.long_indices) | set(parent2.long_indices))
                            - set(self._forced_long))
        _lo, _hi = self._long_size_bounds(len(l_union_free))
        if _hi < _lo:
            return None
        n_long = self._rng.randint(_lo, _hi)
        long_indices = self._pick_long_subset(l_union_free, n_long)
        if self._long_only:
            short_indices = []
        else:
            s_union = list(set(parent1.short_indices) | set(parent2.short_indices))
            if len(s_union) < self.c.min_stocks_short:
                # Not enough in union, supplement from full pool
                avail = list(set(range(len(self.short_candidates))) - set(s_union))
                need = self.c.min_stocks_short - len(s_union)
                if len(avail) < need:
                    return None
                s_union += self._rng.sample(avail, need)
            n_short = self._rng.randint(
                self.c.min_stocks_short,
                min(self.c.max_stocks_short, len(s_union))
            )
            short_indices = sorted(self._rng.sample(s_union, min(n_short, len(s_union))))
        child = _Individual(long_indices, short_indices)
        child.fitness = self._fitness(child)
        return child

    def _mutate(self, individual: _Individual) -> Optional[_Individual]:
        """Mutation = stock swap.  Weight-only mutations no longer exist:
        weights are a deterministic function of the subset (_fitness)."""
        long_indices = individual.long_indices.copy()
        short_indices = individual.short_indices.copy()
        if self._long_only or self._rng.random() < 0.5:
            # Swap one stock in long leg — forced names are never swapped out
            avail = sorted(set(range(len(self.long_candidates))) - set(long_indices))
            swappable = [p for p in range(len(long_indices))
                         if long_indices[p] not in self._forced_long]
            if avail and swappable:
                pos = self._rng.choice(swappable)
                if self._bucket_count_bounds:
                    # A swap must keep every bucket inside its count bounds
                    b_rm = self._bucket_of[long_indices[pos]]

                    def _swap_ok(j):
                        b_in = self._bucket_of[j]
                        if b_in == b_rm:
                            return True
                        if b_rm is not None and b_rm in self._bucket_count_bounds:
                            lo_rm = self._bucket_count_bounds[b_rm][0]
                            if self._bucket_count(long_indices, b_rm) - 1 < lo_rm:
                                return False
                        if b_in is not None and b_in in self._bucket_count_bounds:
                            hi_in = self._bucket_count_bounds[b_in][1]
                            if hi_in is not None and self._bucket_count(long_indices, b_in) + 1 > hi_in:
                                return False
                        return True

                    avail = [j for j in avail if _swap_ok(j)]
                if avail:
                    long_indices[pos] = self._rng.choice(avail)
        else:
            # Swap one stock in short leg
            avail = sorted(set(range(len(self.short_candidates))) - set(short_indices))
            if avail and short_indices:
                pos = self._rng.randrange(len(short_indices))
                short_indices[pos] = self._rng.choice(avail)
        # Ensure uniqueness
        long_indices = list(dict.fromkeys(long_indices))
        short_indices = list(dict.fromkeys(short_indices))
        mutated = _Individual(long_indices, short_indices)
        mutated.fitness = self._fitness(mutated)
        return mutated

    # ══════════════════════════════════════════════════════════════════════════
    # UTILITIES
    # ══════════════════════════════════════════════════════════════════════════

    def _copy(self, ind: _Individual) -> _Individual:
        new = _Individual(ind.long_indices.copy(), ind.short_indices.copy(),
                          ind.long_weights.copy(), ind.short_weights.copy())
        new.fitness = ind.fitness
        return new

    def _empty_result(self) -> OptimizationResult:
        r = OptimizationResult(
            long_basket=[], short_basket=[],
            long_strike_weighted=0.0, short_strike_weighted=0.0,
            net_strike=0.0, score=-np.inf, generations_run=0, converged=False,
            scoring_mode=self._scoring_mode,
            scoring_signature=getattr(self, "_scoring_signature", None),
            seed=self.seed,
            reference_size=getattr(self, "_reference_size", None),
        )
        r._debug_info = getattr(self, '_debug_table', None) or f"Rejections: {dict(self._rejection_reasons)}"
        return r

    def _to_result(self, best: _Individual, gens: int) -> OptimizationResult:
        """Convert best individual to OptimizationResult.
        
        CRITICAL: For cross-corridor, DispersionLeg.variance_asset is shared (e.g. 'SPX Index') across
        all candidates. Must use _candidate_key (= DispersionLeg.corridor_condition_asset) as the unique basket key
        to avoid dict collapse.
        
        ORDER: Safety-net runs FIRST (may update best.long_weights/fitness),
        then ALL extractions (basket, strikes, FINAL-RAW) read from the final weights.
        """
        is_xcorr = self.is_cross_corridor
        
        # ═══ SAFETY NET (runs BEFORE any weight extraction) ═══
        # NEVER DOWNGRADE: keep GA weights if they score better on the full
        # adaptive curve.  Skipped in vega mode: the safety-net's fine
        # bisection is V-blind (the vega paths already deliver exact or
        # refined weights; the step rescore below stays vega-aware).
        if self._weight_solver.has_exact_path() and self._vega is None:
            try:
                _fk, _fp, _fs, _fb = self._subset_arrays(best.long_indices)
                if _fp is None:
                    raise ValueError("final basket has names missing from the P&L matrix")
                long_pos_final = list(_fp)
                sub_pnl_final = self._ts_mat[:, long_pos_final]
                # Use adaptive bisection for final weights (exact on full curve)
                _safety_policy = "adaptive_reweight" if self.missing_data_policy == MissingDataPolicy.ADAPTIVE_REWEIGHT else self._weight_solver._missing_data_policy
                _orig_policy = self._weight_solver._missing_data_policy
                self._weight_solver._missing_data_policy = _safety_policy
                # Clear cache to force fresh solve at fine tolerance
                self._weight_solver._cache.pop(tuple(sorted(int(i) for i in long_pos_final)), None)
                resolv = self._weight_solver.solve(
                    pnl_matrix=self._ts_mat,
                    stock_indices=np.array(long_pos_final, dtype=int),
                    active_mask=self._active_mask,
                    tol="fine",
                    group_bounds=self._bucket_groups_for(best.long_indices),
                )
                self._weight_solver._missing_data_policy = _orig_policy
                if resolv.feasible:
                    # Evaluate BOTH weight sets on the full adaptive curve with user's scoring
                    ga_pnl = self._adaptive_net_pnl(long_pos_final, best.long_weights)
                    ga_min = float(np.min(ga_pnl))
                    _ga_ws = (self._net_strike(best.long_indices, best.long_weights,
                                               best.short_indices, best.short_weights)
                              if self._ws_active() else None)
                    ga_score = self._score_fn.score(ga_pnl, self._make_ctx(_ga_ws))

                    resolv_w = project_to_bounded_simplex(
                        np.asarray(resolv.weights, dtype=np.float64),
                        np.array(
                            [self.long_candidates[i].min_weight for i in best.long_indices],
                            dtype=np.float64,
                        ),
                        np.array(
                            [self.long_candidates[i].max_weight for i in best.long_indices],
                            dtype=np.float64,
                        ),
                    )

                    if (
                            resolv_w is None
                            or not np.isclose(resolv_w.sum(), 1.0, atol=1e-8)
                            or not self._is_strike_valid(
                        best.long_indices,
                        resolv_w,
                        best.short_indices,
                        best.short_weights,
                    )
                    ):
                        raise ValueError("Final solver weights fail sum, bounds or strike validation")

                    bisect_pnl = self._adaptive_net_pnl(long_pos_final, resolv_w)
                    bisect_min = float(np.min(bisect_pnl))
                    _bs_ws = (self._net_strike(best.long_indices, resolv_w,
                                               best.short_indices, best.short_weights)
                              if self._ws_active() else None)
                    bisect_score = self._score_fn.score(bisect_pnl, self._make_ctx(_bs_ws))


                    _engine_log.debug(f"[BISECT-VS-GA] ga_min={ga_min:.4f} ga_score={ga_score:.4f} | bisect_min={bisect_min:.4f} bisect_score={bisect_score:.4f}")

                    # NEVER DOWNGRADE: pick the better weight set by user's blend objective
                    # TIE-BREAK: when scores are within 1e-4, prefer higher raw min_payoff
                    if abs(bisect_score - ga_score) < TUNING.safety_tie_window:
                        winner = "bisect" if bisect_min > ga_min else "ga"
                        _engine_log.debug(f"[SAFETY-NET] tie on score ({ga_score:.4f} vs {bisect_score:.4f}), min decides: {winner} (ga_min={ga_min:.4f} bisect_min={bisect_min:.4f})")
                        if winner == "bisect":
                            best.long_weights = resolv_w
                            best.fitness = bisect_score
                        else:
                            best.fitness = ga_score
                    elif bisect_score > ga_score:
                        best.long_weights = resolv_w
                        best.fitness = bisect_score
                        _engine_log.debug(f"[SAFETY-NET] UPGRADED: bisection wins (score {ga_score:.4f} -> {bisect_score:.4f}, min {ga_min:.4f} -> {bisect_min:.4f})")
                    else:
                        # Keep GA weights — they score equal or better
                        best.fitness = ga_score
                        _engine_log.debug(f"[SAFETY-NET] KEPT GA: ga wins (score {ga_score:.4f} >= {bisect_score:.4f}, min {ga_min:.4f} vs {bisect_min:.4f})")
                else:
                    # Bisection infeasible — keep GA weights, re-score them
                    ga_pnl = self._adaptive_net_pnl(long_pos_final, best.long_weights)
                    _ga_ws2 = (self._net_strike(best.long_indices, best.long_weights,
                                                best.short_indices, best.short_weights)
                               if self._ws_active() else None)
                    best.fitness = self._score_fn.score(ga_pnl, self._make_ctx(_ga_ws2))
                    _engine_log.debug(f"[SAFETY-NET] bisection infeasible — kept GA weights")
            except Exception as e:
                _engine_log.debug(f"[SAFETY-NET] failed: {e}")
        else:
            # ═══ NON-EXACT CONFIGS: report the STEP score ═══
            # The GA and the refinement rank on the tie-broken scale (which
            # can exceed 1.0 when the quantile score saturates); the
            # user-facing result must carry the step-quantile score of the
            # delivered basket, exactly like the exact-path safety net does.
            try:
                _fk2, _fp2, _fs2, _fb2 = self._subset_arrays(best.long_indices)
                if _fp2 is None:
                    raise ValueError("final basket has names missing from the P&L matrix")
                long_pos_final = list(_fp2)
                _sp_final = None
                _sw_final = None
                if not self._long_only and len(best.short_indices) > 0:
                    _sids_f = [self._col_pos[_candidate_key(self.short_candidates[i], is_xcorr)]
                               for i in best.short_indices
                               if _candidate_key(self.short_candidates[i], is_xcorr) in self._col_pos]
                    if _sids_f:
                        _sp_final = _sids_f
                        _sw_final = best.short_weights
                _final_pnl = self._adaptive_net_pnl(long_pos_final, best.long_weights,
                                                    _sp_final, _sw_final)
                _fin_ws = (self._net_strike(best.long_indices, best.long_weights,
                                            best.short_indices, best.short_weights)
                           if self._ws_active() else None)
                _fin_ac, _fin_ar = self._axe_ctx_values(
                    best.long_indices, best.long_weights, None)
                best.fitness = self._score_fn.score(
                    _final_pnl, self._make_ctx(_fin_ws, _fin_ac, _fin_ar))
            except Exception as e:
                self.log("WARN", f"[FINAL-SCORE] step rescore failed: {e}")

        # ═══ POST-SMOOTHING — DISABLED (smooth_weights always False from _api.py) ═══
        # Interactive smoothing is handled in the UI via _smooth_state, not here.
        assert not self._smooth_weights, (
            "In-pipeline smoothing is disabled — score/basket mismatch (audit D1). "
            "Use interactive post-smoothing in UI instead."
        )
        if self._smooth_weights:  # pragma: no cover — assertion above ensures dead
            try:
                long_pos_sm = [self._col_pos[_candidate_key(self.long_candidates[i], is_xcorr)]
                               for i in best.long_indices
                               if _candidate_key(self.long_candidates[i], is_xcorr) in self._col_pos]
                w_star = best.long_weights.copy()
                w_smooth = self._weight_solver.smooth_weights(
                    w_star=w_star,
                    pnl_matrix=self._ts_mat,
                    stock_indices=np.array(long_pos_sm, dtype=int),
                    active_mask=self._active_mask,
                    eps_min=self._smooth_eps,
                )
                # Log comparison
                star_pnl = self._adaptive_net_pnl(long_pos_sm, w_star)
                smooth_pnl = self._adaptive_net_pnl(long_pos_sm, w_smooth)
                star_min = float(np.min(star_pnl))
                smooth_min = float(np.min(smooth_pnl))
                star_mean = float(np.mean(star_pnl))
                smooth_mean = float(np.mean(smooth_pnl))
                star_std = float(np.std(w_star))
                smooth_std = float(np.std(w_smooth))
                _engine_log.debug(f"[SMOOTH] min: {star_min:.4f} -> {smooth_min:.4f} | mean: {star_mean:.4f} -> {smooth_mean:.4f} | dispersion(std): {star_std:.4f} -> {smooth_std:.4f} | eps={self._smooth_eps:.2f}")
                # Per-ticker weight comparison, sorted by |delta| descending
                _engine_log.debug("[SMOOTH-WEIGHTS] ticker | w_optimal | w_smooth | delta")
                _rows = []
                for i, (wo, ws) in enumerate(zip(w_star, w_smooth)):
                    tk = _candidate_key(self.long_candidates[best.long_indices[i]], is_xcorr)
                    _rows.append((tk, float(wo), float(ws), float(ws - wo)))
                _rows.sort(key=lambda r: abs(r[3]), reverse=True)
                for tk, wo, ws, d in _rows:
                    _engine_log.debug(f"[SMOOTH-WEIGHTS] {tk:20s} | {wo:.2f} | {ws:.2f} | {d:+.2f}")
                # Store unsmoothed basket on optimizer for API dual-backtest access
                self._last_unsmoothed_basket = [
                    (_candidate_key(self.long_candidates[i], is_xcorr), float(w))
                    for i, w in zip(best.long_indices, w_star)
                ]
                best.long_weights = w_smooth
            except Exception as e:
                _engine_log.debug(f"[SMOOTH] failed: {e}, keeping original weights")
                self._last_unsmoothed_basket = None
        else:
            self._last_unsmoothed_basket = None

        # ═══ EXTRACT BASKETS (reads best.long_weights AFTER safety-net decision) ═══
        long_basket = [
            (_candidate_key(self.long_candidates[i], is_xcorr), float(w))
            for i, w in zip(best.long_indices, best.long_weights)
        ]
        long_strikes = [
            (_candidate_key(self.long_candidates[i], is_xcorr), self.long_candidates[i].strike_mono_var_swap)
            for i in best.long_indices
        ]
        
        # Cross-corridor detail: (Variance Asset, Corridor Condition Asset, Strike Cross Corridor)
        long_xcorr = []
        if is_xcorr:
            long_xcorr = [
                (self.long_candidates[i].variance_asset,
                 self.long_candidates[i].corridor_condition_asset or "",
                 self.long_candidates[i].strike_cross_corridor or 0.0)
                for i in best.long_indices
            ]
        
        if self._long_only:
            short_basket, short_strikes, short_xcorr, ss = [], [], [], 0.0
        else:
            short_basket = [
                (_candidate_key(self.short_candidates[i], is_xcorr), float(w))
                for i, w in zip(best.short_indices, best.short_weights)
            ]
            short_strikes = [
                (_candidate_key(self.short_candidates[i], is_xcorr), self.short_candidates[i].strike_mono_var_swap)
                for i in best.short_indices
            ]
            short_xcorr = [
                (self.short_candidates[i].variance_asset,
                 self.short_candidates[i].corridor_condition_asset or "",
                 self.short_candidates[i].strike_cross_corridor or 0.0)
                for i in best.short_indices
            ] if is_xcorr else []
            ss = sum(self.short_candidates[i].strike_mono_var_swap * w
                     for i, w in zip(best.short_indices, best.short_weights))
        
        ls = sum(self.long_candidates[i].strike_mono_var_swap * w
                 for i, w in zip(best.long_indices, best.long_weights))
        
        if is_xcorr:
            net = 0.0
            for i, w in zip(best.long_indices, best.long_weights):
                s = self.long_candidates[i]
                idx_strike = s.strike_cross_corridor if s.strike_cross_corridor is not None else s.strike_mono_var_swap
                net += (s.strike_mono_var_swap - idx_strike) * w
            net_strike_val = abs(net)
        else:
            net_strike_val = abs(ls - ss)

        # ═══ BUCKET AUDIT (delivered basket must honour the constraints) ═══
        if self._bucket_constraints:
            if not self._bucket_counts_ok(best.long_indices):
                _engine_log.debug(f"⚠️ [BUCKET-WARNING] Delivered basket violates bucket COUNT bounds")
                self.log("WARN", "[BUCKET] Delivered basket violates bucket count bounds")
            _g_final = self._bucket_groups_for(best.long_indices)
            if _g_final and not self._weight_solver._groups_satisfied(
                    np.asarray(best.long_weights, dtype=np.float64), _g_final, tol=1e-4):
                _engine_log.debug(f"⚠️ [BUCKET-WARNING] Delivered basket violates bucket WEIGHT bounds")
                self.log("WARN", "[BUCKET] Delivered basket violates bucket weight bounds")

        # ═══ STRIKE AUDIT LOG ═══
        _binding = net_strike_val >= (self.c.max_net_strike - 1e-6) if self.c.max_net_strike > 0 else False
        _violation = net_strike_val > self.c.max_net_strike + 1e-4 if self.c.max_net_strike > 0 else False
        _convention = "net_spread (stock-index)" if is_xcorr else "mono_var_swap"
        _engine_log.debug(f"[STRIKE] convention={_convention} | basket={net_strike_val:.4f} | limit={self.c.max_net_strike:.4f} | binding={_binding}")
        if _violation:
            _engine_log.debug(f"⚠️ [STRIKE-WARNING] Delivered basket VIOLATES strike limit: {net_strike_val:.4f} > {self.c.max_net_strike:.4f}")
            self.log("WARN", f"[STRIKE] Violation: net_strike={net_strike_val:.4f} > limit={self.c.max_net_strike:.4f}")
        
        # ═══ FINAL-RAW (uses same best.long_weights as basket extraction) ═══
        try:
            long_pos = [self._col_pos[_candidate_key(self.long_candidates[i], is_xcorr)]
                        for i in best.long_indices
                        if _candidate_key(self.long_candidates[i], is_xcorr) in self._col_pos]
            short_pos = None
            short_w_final = None
            if not self._long_only and len(best.short_indices) > 0:
                short_pos = [self._col_pos[_candidate_key(self.short_candidates[i], is_xcorr)]
                             for i in best.short_indices
                             if _candidate_key(self.short_candidates[i], is_xcorr) in self._col_pos]
                short_w_final = best.short_weights
                if not short_pos:
                    short_pos = None
                    short_w_final = None
            final_net_pnl = self._adaptive_net_pnl(long_pos, best.long_weights, short_pos, short_w_final)
            _fin_ws = (self._net_strike(best.long_indices, best.long_weights,
                                        best.short_indices, best.short_weights)
                       if self._ws_active() else None)
            ctx = ScoreContext(n_days=len(final_net_pnl), ann_factor=252,
                               weighted_strike=_fin_ws)
            raw_final = self._score_fn.raw_metrics(final_net_pnl, ctx)
            self.log("INFO", f"[FINAL-RAW] " + " | ".join(f"{k}={v:.4f}" for k, v in raw_final.items()))
            _engine_log.debug(f"[FINAL-RAW] " + " | ".join(f"{k}={v:.4f}" for k, v in raw_final.items()))
            self._final_raw_min = float(np.min(final_net_pnl))  # For interactive assertion
        except Exception as e:
            self.log("WARN", f"[FINAL-RAW] failed: {e}")

        print(
            f"[WEIGHT-AUDIT] "
            f"best_sum={np.sum(best.long_weights):.12f} | "
            f"basket_sum={sum(w for _, w in long_basket):.12f} | "
            f"indices={len(best.long_indices)} | "
            f"weights={len(best.long_weights)} | "
            f"basket_rows={len(long_basket)} | "
            f"unique_keys={len(set(k for k, _ in long_basket))}",
            flush=True,
        )


        # ═══ ABSOLUTE-VEGA OUTPUTS (deterministic grid rule on final weights) ═══
        _vega_total = None
        _vega_basket = None
        _axe_cleaned = None
        _axe_recycled = None
        if self._vega is not None:
            _picked = self._axe_grid(best.long_indices, best.long_weights)
            if _picked is not None:
                _vega_total, _axe_cleaned, _axe_recycled = _picked
                _vega_basket = [(k, float(w) * _vega_total) for k, w in long_basket]
                print(f"[VEGA] V={_vega_total:.2f} | axe_cleaned={_axe_cleaned:.4f} | "
                      f"axe_recycled={_axe_recycled:.4f}", flush=True)
            else:
                self.log("WARN", "[VEGA] no feasible V for the delivered basket "
                                 "(caps vs V_min) — vega fields left empty")

        return OptimizationResult(
            long_basket=long_basket, short_basket=short_basket,
            long_strikes=long_strikes, short_strikes=short_strikes,
            long_strike_weighted=ls, short_strike_weighted=ss,
            net_strike=net_strike_val, score=best.fitness,
            generations_run=gens, converged=True,
            is_cross_corridor=is_xcorr,
            long_cross_corridor=long_xcorr, short_cross_corridor=short_xcorr,
            scoring_mode=self._scoring_mode,
            unsmoothed_basket=self._last_unsmoothed_basket,
            scoring_signature=getattr(self, "_scoring_signature", None),
            seed=self.seed,
            reference_size=getattr(self, "_reference_size", None),
            total_vega=_vega_total,
            vega_basket=_vega_basket,
            axe_cleaned=(float(_axe_cleaned) if _axe_cleaned is not None
                         and np.isfinite(_axe_cleaned) else None),
            axe_recycled=(float(_axe_recycled) if _axe_recycled is not None
                          and np.isfinite(_axe_recycled) else None),
        )

    @property
    def _has_unsmoothed(self):
        """Check if last run produced unsmoothed weights for dual backtest."""
        return self._smooth_weights

    def get_unsmoothed_basket(self) -> list:
        """Return basket with pre-smooth weights, or None."""
        return self._last_unsmoothed_basket

    # ── Bootstrap robustness diagnostic (called externally if robustness_check=True) ──

    def bootstrap_robustness(self, n_draws: int = TUNING.bootstrap_draws,
                             top_k: int = TUNING.bootstrap_top_k,
                             seed: Optional[int] = None) -> Optional[Dict]:
        """Day-resampling robustness of the delivered basket.

        Resamples the trading days with replacement ``n_draws`` times and, on
        each draw, re-scores the winner against its ``top_k`` best DISTINCT
        refinement challengers (each at its own solved weights, against the
        run's fixed reference).  Returns::

            {"n_draws", "n_challengers",
             "top1_freq", "top3_freq",          # winner's rank frequencies
             "winner_raw_ci": {metric: {"lo": p2.5, "mean", "hi": p97.5}}}

        Deterministic: the resampling RNG derives from the run seed.
        Returns None when no refinement pool is available (empty result).
        """
        best = getattr(self, "_last_best", None)
        if best is None or self._score_fn is None or not self._score_fn.is_fitted:
            return None

        def _contender_net_and_ctx(long_idx, long_w, short_idx, short_w):
            _bk, pos, _bs, _bb = self._subset_arrays(long_idx)
            if pos is None:
                return None
            spos = None
            sw = None
            if short_idx:
                sids = [_candidate_key(self.short_candidates[i], self.is_cross_corridor)
                        for i in short_idx]
                sp = np.array([self._col_pos[c] for c in sids if c in self._col_pos])
                if len(sp):
                    spos = sp
                    sw = short_w
            net = self._adaptive_net_pnl(pos, long_w, spos, sw)
            ws = (self._net_strike(long_idx, long_w, short_idx, short_w)
                  if self._ws_active() else None)
            ac, ar = self._axe_ctx_values(long_idx, long_w, None)
            return net, self._make_ctx(ws, ac, ar)

        winner = _contender_net_and_ctx(best.long_indices, best.long_weights,
                                        best.short_indices, best.short_weights)
        if winner is None:
            return None
        winner_key = tuple(sorted(best.long_indices))

        # Top-k distinct challengers: refinement pool first (solver-refined
        # weights), then the final population (a converged GA's refinement
        # pool can collapse to the winner's subset alone)
        challengers = []
        seen = {winner_key}
        _pool = (sorted(getattr(self, "_refine_candidates", []), key=lambda t: -t[4])
                 + list(getattr(self, "_final_population", [])))
        for long_idx, long_w, short_idx, short_w, sc in _pool:
            key = tuple(sorted(long_idx))
            if key in seen:
                continue
            seen.add(key)
            out = _contender_net_and_ctx(long_idx, long_w, short_idx, short_w)
            if out is not None:
                challengers.append(out)
            if len(challengers) >= top_k:
                break

        contenders = [winner] + challengers
        rng = np.random.default_rng([self.seed, 0xB007])
        n_rows = len(winner[0])
        active = list(self._score_fn.weights.active_names)
        top1 = 0
        top3 = 0
        raw_draws: Dict[str, List[float]] = {m: [] for m in active}
        for _ in range(int(n_draws)):
            idx = rng.integers(0, n_rows, n_rows)
            scores = []
            for net, ctx in contenders:
                fit = self._fitness_from_net(net[idx], ctx)
                scores.append(-np.inf if fit is None else fit)
            rank = 1 + sum(1 for s in scores[1:] if s > scores[0])
            if rank == 1:
                top1 += 1
            if rank <= 3:
                top3 += 1
            raw = self._score_fn.raw_metrics(winner[0][idx], winner[1])
            for m in active:
                v = raw.get(m)
                if v is not None and np.isfinite(v):
                    raw_draws[m].append(float(v))

        ci = {}
        for m in active:
            vals = np.asarray(raw_draws[m], dtype=np.float64)
            if vals.size == 0:
                continue
            ci[m] = {"lo": float(np.percentile(vals, 2.5)),
                     "mean": float(vals.mean()),
                     "hi": float(np.percentile(vals, 97.5))}
        return {
            "n_draws": int(n_draws),
            "n_challengers": len(challengers),
            "top1_freq": top1 / float(n_draws),
            "top3_freq": top3 / float(n_draws),
            "winner_raw_ci": ci,
        }

    # ── MILP benchmark (called externally if run_milp=True) ──

    def milp_benchmark(self) -> Optional[Dict[str, float]]:
        """Solve the exact MILP for min_payoff-only config. Returns None if not applicable."""
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import csr_matrix as _csr

        if not self._weight_solver._is_min_payoff_only():
            _engine_log.debug("[MILP] Not min_payoff-only config — skipping.")
            return None

        is_xcorr = self.is_cross_corridor

        # ── Use EXACTLY self._ts_mat — ALL columns, same object as _fitness ──
        P = self._ts_mat.copy()
        n_stocks = P.shape[1]  # ALL columns = full candidate universe

        if n_stocks == 0:
            _engine_log.debug("[MILP] No valid candidates.")
            return None

        # Apply same cap/floor as GA evaluation in _fitness
        if self.global_cap < 9999998 or self.global_floor > -9999998:
            P = np.clip(P, self.global_floor, self.global_cap)

        # DO NOT drop all-zero rows — GA keeps them (zeros count as non-positive days
        # in hit_ratio, and removing them changes the denominator for all metrics).
        # The GA's _fitness only strips zeros AFTER computing net_pnl
        # for a specific basket selection, not at the matrix level.
        n_days = P.shape[0]

        _engine_log.debug(f"[MILP] matrix shape={P.shape} | ga_matrix shape={self._ts_mat.shape} | id(ga_mat)={id(self._ts_mat)}")

        # Build per-column strike array (for strike constraint)
        # Invert col_pos: col_index → candidate_index
        col_to_cand = {}
        for i, leg in enumerate(self.long_candidates):
            k = _candidate_key(leg, is_xcorr)
            if k in self._col_pos:
                col_to_cand[self._col_pos[k]] = i
        strikes_per_col = np.zeros(n_stocks)
        min_w_per_col = np.zeros(n_stocks)
        max_w_per_col = np.full(n_stocks, self._weight_solver._constraints.max_weight)
        unmapped_cols: List[int] = []
        for col_idx in range(n_stocks):
            if col_idx in col_to_cand:
                cand_idx = col_to_cand[col_idx]
                leg = self.long_candidates[cand_idx]
                strikes_per_col[col_idx] = leg.strike_mono_var_swap if leg.strike_mono_var_swap is not None else 0.0
                min_w_per_col[col_idx] = leg.min_weight
                max_w_per_col[col_idx] = leg.max_weight
            else:
                # Column exists in the matrix but is NOT a selectable long
                # candidate (excluded name, 0%-HR-filtered, or a short-leg
                # column). The GA can never pick it, so the certificate must
                # not either — z_i is pinned to 0 below.
                unmapped_cols.append(col_idx)

        # ── Forced names: z_i pinned to 1 (candidate universe consistency) ──
        forced_cols: List[int] = []
        _missing_forced: List[str] = []
        for cand_idx in self._forced_long:
            k = _candidate_key(self.long_candidates[cand_idx], is_xcorr)
            col = self._col_pos.get(k)
            if col is None:
                _missing_forced.append(k)
            else:
                forced_cols.append(col)
        if _missing_forced:
            raise RuntimeError(
                f"[MILP] forced name(s) {_missing_forced} have no P&L matrix "
                f"column — cannot certify a universe the GA did not see."
            )

        c = self.c
        min_w = self._weight_solver._constraints.min_weight
        max_w = self._weight_solver._constraints.max_weight
        min_k = c.min_stocks_long
        max_k = c.max_stocks_long

        # Decision variables: [w_1..w_n, z_1..z_n, t]  total = 2n+1
        n_vars = 2 * n_stocks + 1

        # Objective: maximize t => minimize -t
        obj = np.zeros(n_vars)
        obj[-1] = -1.0  # coefficient for t

        # Variable bounds
        # w_i in [0, max_w_i], z_i in {0,1} (handled via integrality), t unbounded
        lb = np.zeros(n_vars)
        ub = np.full(n_vars, np.inf)
        ub[:n_stocks] = max_w_per_col     # per-stock max weight
        ub[n_stocks:2*n_stocks] = 1.0     # z_i <= 1
        lb[-1] = -np.inf                  # t unbounded below
        ub[-1] = np.inf
        # Universe consistency with the GA: unmapped columns unselectable,
        # forced names always selected (z fixed via bounds + integrality).
        for col_idx in unmapped_cols:
            ub[n_stocks + col_idx] = 0.0
            ub[col_idx] = 0.0             # w_i = 0 as well (belt and braces)
        for col_idx in forced_cols:
            lb[n_stocks + col_idx] = 1.0

        # Integrality: 0=continuous, 1=integer
        integrality = np.zeros(n_vars, dtype=int)
        integrality[n_stocks:2*n_stocks] = 1  # z_i are binary

        constraints = []

        # C1: sum(w) = 1
        A_eq1 = np.zeros(n_vars)
        A_eq1[:n_stocks] = 1.0
        constraints.append(LinearConstraint(A_eq1.reshape(1, -1), 1.0, 1.0))

        # C2: min_k <= sum(z) <= max_k
        A_card = np.zeros(n_vars)
        A_card[n_stocks:2*n_stocks] = 1.0
        constraints.append(LinearConstraint(A_card.reshape(1, -1), float(min_k), float(max_k)))

        # C3: w_i >= min_w_i * z_i  =>  w_i - min_w_i*z_i >= 0
        # C4: w_i <= max_w_i * z_i  =>  -w_i + max_w_i*z_i >= 0
        A_link = np.zeros((2 * n_stocks, n_vars))
        for i in range(n_stocks):
            A_link[i, i] = 1.0
            A_link[i, n_stocks + i] = -min_w_per_col[i]
            A_link[n_stocks + i, i] = -1.0
            A_link[n_stocks + i, n_stocks + i] = max_w_per_col[i]
        constraints.append(LinearConstraint(_csr(A_link), 0.0, np.inf))

        # C5: P[d]@w >= t  =>  P[d]@w - t >= 0
        A_pnl = np.zeros((n_days, n_vars))
        A_pnl[:, :n_stocks] = P
        A_pnl[:, -1] = -1.0
        constraints.append(LinearConstraint(_csr(A_pnl), 0.0, np.inf))

        # C6: strike constraint if applicable
        max_net_strike = self._weight_solver._constraints.max_net_strike
        if max_net_strike is not None and max_net_strike < 9999:
            A_strike = np.zeros(n_vars)
            A_strike[:n_stocks] = strikes_per_col
            constraints.append(LinearConstraint(A_strike.reshape(1, -1), -np.inf, max_net_strike))

        print(f"[MILP] n_candidates={n_stocks}, n_days={n_days}, min_k={min_k}, max_k={max_k} | "
              f"forced={len(forced_cols)}, unselectable={len(unmapped_cols)} | "
              f"approx=(row-level zero drop, not per-basket)", flush=True)

        res = milp(
            c=obj,
            constraints=constraints,
            integrality=integrality,
            bounds=Bounds(lb, ub),
            options={"time_limit": 300.0, "disp": False},
        )

        if not res.success:
            bound = -res.fun if hasattr(res, 'fun') and res.fun is not None else None
            _engine_log.debug(f"[MILP] Failed: {res.message} | bound={bound}")
            return {"status": "failed", "message": res.message, "bound": bound}

        milp_t = res.x[-1]
        milp_w = res.x[:n_stocks]
        milp_z = res.x[n_stocks:2*n_stocks]
        selected = int(np.round(milp_z).sum())

        _engine_log.debug(f"[MILP] Exact optimum: min_payoff={milp_t:.6f} | selected={selected} stocks")
        return {"status": "optimal", "min_payoff": milp_t, "n_selected": selected, "weights": milp_w, "z": milp_z}
