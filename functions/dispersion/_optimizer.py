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
    optimizer = DispersionOptimizer(config, constraints, score_weights)
    result = optimizer.run(pnl_matrix)
"""
from __future__ import annotations
import itertools
import math
import random
import time
import os
import numpy as np
import pandas as pd
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── File-based debug logger (always visible, even under Streamlit) ──
_OPTIM_LOG_PATH = os.path.join(os.path.dirname(__file__), "_optim_debug.log")

def _dlog(msg: str):
    """Debug log — disabled."""
    pass

def _dlog_ticker_mismatch(tickers: List[str], col_pos_keys: List[str]):
    """Visible debug for ticker format mismatch — disabled."""
    pass

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
    DispersionLeg,
    MissingDataPolicy,
    OptimizationConstraints,
    OptimizationResult,
    ProductType,
    ScoreWeights,
    SwapConfig,
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
from functions.dispersion.scoring.weight_solver import adaptive_pnl
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Internal representation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _Individual:
    """Internal GA solution representation."""
    __slots__ = ("long_indices", "short_indices", "long_weights", "short_weights", "fitness")

    def __init__(self, long_indices, short_indices, long_weights, short_weights):
        self.long_indices = list(long_indices)
        self.short_indices = list(short_indices)
        self.long_weights = np.asarray(long_weights, dtype=np.float64)
        self.short_weights = np.asarray(short_weights, dtype=np.float64)
        self.fitness = -np.inf
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Weight optimization helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _qp_diversified_weights(col_means: np.ndarray, lb: np.ndarray, ub: np.ndarray,
                            diversification_penalty: float = 0.3) -> np.ndarray:
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
        long_candidates: List[Stock],
        short_candidates: List[Stock],
        pnl_matrix: np.ndarray,
        column_map: Dict[str, int],
        constraints: Optional[OptimizationConstraints] = None,
        score_weights: Optional[ScoreWeights] = None,
        logger: Optional[Callable[[str, str], None]] = None,
        missing_data_policy=None,
        adj_divs: bool = False,
        reweight_grace_days: int = 3,
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
        self._all_metrics_linear: bool = False  # P2: adaptive dispatch
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
        # Normalize score weights (original behavior)
        sw = score_weights or ScoreWeights()
        self._score_weights_arr = sw.as_array()  # [last_value, avg_5y, avg_3y, max_drawdown, hit_ratio]
        # Store original P&L matrix (with NaN) for adaptive reweighting in fitness
        self._orig_ts_mat = pnl_matrix.astype(np.float64).copy()
        # Also store filled version for backward compatibility (e.g., GA initialization)
        self._ts_mat = np.nan_to_num(self._orig_ts_mat, nan=0.0)
        self._col_pos = dict(column_map)
        self._n_rows = self._ts_mat.shape[0]
        # Precompute validity mask for adaptive reweight (True where data exists)
        self._valid_mask = ~np.isnan(self._orig_ts_mat)  # [n_rows x n_cols]
        # Build date index for 3Y/5Y window calculation
        # Use row count as proxy: 252 trading days/year
        self._n_3y = min(self._n_rows, 252 * 3)
        self._n_5y = min(self._n_rows, 252 * 5)
        # Pre-compute per-stock quality for weight allocation
        self._quality_long = np.array([
            self._calculate_stock_quality(s) for s in self.long_candidates
        ], dtype=np.float64)
        self._quality_short = np.array([
            self._calculate_stock_quality(s) for s in self.short_candidates
        ], dtype=np.float64) if self.short_candidates else np.array([])
        self._long_only = len(self.short_candidates) == 0
        # Rejection tracking
        self._rejection_reasons = {
            "fitness<=0": 0,
            "no_weights": 0,
            "strike_invalid": 0,
            "invalid_score": 0,
            "last_carry_zero": 0,
            "len_valid_lt_50": 0,
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
        # Print all config at init
        _dlog(f"🔬 [OPTIM-INIT] {len(long_candidates)} long, {len(short_candidates)} short | "
              f"pop={self.c.population_size} gen={self.c.max_generations} time={self.c.time_limit_seconds}s")
        if long_candidates:
            s = long_candidates[0]
            _dlog(f"🔬 [OPTIM-INIT] bounds sample: min_w={s.min_weight:.4f} max_w={s.max_weight:.4f}")
    # ══════════════════════════════════════════════════════════════════════════
    # STOCK QUALITY (used by weight allocator)
    # ══════════════════════════════════════════════════════════════════════════

    def _calculate_stock_quality(self, leg: DispersionLeg) -> float:
        """Per-stock quality metric for weight allocation bias (original Gaia_PP logic)."""
        m = leg.metrics if leg.metrics else {}
        sw = self._score_weights_arr  # [last_value, avg_5y, avg_3y, max_drawdown, hit_ratio]
        # Apply same linear scaling as _calculate_fitness for consistent weighting
        last_val = m.get('last_value', 0.0) / 3.0
        avg_5y = m.get('avg_5y', 0.0) / 0.06
        avg_3y = m.get('avg_3y', 0.0) / 0.08
        # hit_ratio from backtest is in [0, 100] range → center to [-1, 1]
        hit_ratio = (m.get('hit_ratio', 50.0) / 100.0 - 0.5) * 2.0
        # max_drawdown from backtest is negative (e.g., -0.15) — use absolute value, scale
        max_dd = abs(m.get('max_drawdown', 0.0)) / 1.5
        return (
            last_val * sw[0] +
            avg_5y * sw[1] +
            avg_3y * sw[2] +
            hit_ratio * sw[4] -
            max_dd * sw[3]
        )

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
            _dlog(f"NOT ENOUGH SHORT: {len(self.short_candidates)} < {self.c.min_stocks_short}")
            return self._empty_result()
        # ── Setup new scoring system (bilevel) if metric_weights provided ──
        if self._use_new_scoring:
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
            # P2: Check if all active metrics are linear (enables inline LP during GA)
            active_metrics = [m for m in self._score_fn._metrics
                             if m.name in self._metric_weights.active_names]
            self._all_metrics_linear = all(m.is_linear for m in active_metrics)
            # P5: Adaptive n_samples — reduced for speed; tail metrics get more
            has_tail = any(getattr(m, 'is_tail_metric', False) for m in active_metrics)
            n_samples = 800 if has_tail else 300
            self._weight_solver = None  # not available during reference build
            _dlog(f"BUILD REFERENCE SAMPLE: n_samples={n_samples}")
            self._build_reference_sample(n_samples=n_samples)
            # ── TABLE: Long candidates + PnL data quality ──
            self._print_candidate_table()
            _dlog(f"SCORE FN IS FITTED: {self._score_fn.is_fitted}")
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
        # Create initial population
        population = []
        attempts = 0
        while len(population) < self.c.population_size and attempts < self.c.population_size * 3:
            ind = self._create_random_individual()
            if ind is not None:
                population.append(ind)
            attempts += 1
        if not population:
            _dlog("FAILED TO CREATE INITIAL POPULATION")
            return self._empty_result()
        _dlog(f"POPULATION: {len(population)} individuals")
        # Start the evolution timer AFTER reference sample + population init
        # so the GA gets the full time budget for actual evolution
        start_time = time.time()
        setup_elapsed = start_time - setup_start
        self.log("INFO", f"⚙️ Setup took {setup_elapsed:.1f}s — GA evolution starts now (budget: {self.c.time_limit_seconds}s)")
        print(f"[TIMING] setup={setup_elapsed:.1f}s | GA budget={self.c.time_limit_seconds}s", flush=True)
        init_weights = np.concatenate([ind.long_weights for ind in population])
        _dlog(f"INIT POP: {len(init_weights)} weights, mean={init_weights.mean():.4f}, std={init_weights.std():.4f}")
        best_individual = max(population, key=lambda x: x.fitness)
        best_score = best_individual.fitness
        generations_run = 0
        stagnation = 0
        # Evolution loop
        for generation in range(self.c.max_generations):
            if time.time() - start_time > self.c.time_limit_seconds:
                self.log("INFO", f"⏰ Time limit at gen {generation} (elapsed={time.time()-start_time:.1f}s)")
                _dlog(f"⏰ [TIME] gen {generation}/{self.c.max_generations} | {time.time()-start_time:.1f}s / {self.c.time_limit_seconds}s budget | {(time.time()-start_time)/max(1,generation):.2f}s/gen")
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
            while len(new_population) < self.c.population_size and fill_attempts < self.c.population_size * 3:
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
            n_immigrants = max(4, int(self.c.population_size * 0.10))
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
            min_pop = max(30, int(self.c.population_size * 0.6))
            if len(population) < min_pop:
                backfill_attempts = 0
                while len(population) < min_pop and backfill_attempts < min_pop * 2:
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
            print(f"[GEN {generation}] best_fitness={best_score:.4f} | elapsed={time.time()-start_time:.1f}s | cache={self._weight_solver._cache_hits}/{self._weight_solver._cache_misses} | avg_LPs={self._weight_solver._bisect_lp_count/max(1,self._weight_solver._cache_misses):.1f}", flush=True)
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
        _dlog(f"GA FINISHED: {generations_run} gens in {elapsed_ga:.1f}s | best={best_score:.4f}")
        if best_score == -np.inf:
            return self._empty_result()
        # ── Post-GA refinement ──
        _active_now = (set(self._score_fn.weights.active_names)
                       if (self._use_new_scoring and self._score_fn is not None) else set())
        if (self._use_new_scoring and self._weight_solver is not None
                and _active_now != {"hit_ratio"}):
            # Refine top elites for every config except hit_ratio-only (zero
            # gradient).  Exact-path configs get the LP/bisection optimum;
            # non-linear blends (max_drawdown, cvar_5, weighted_strike, ...)
            # go through the smooth-normalised SLSQP inner solver.
            pre_score = best_score
            population.sort(key=lambda x: x.fitness, reverse=True)
            top_k = min(30, len(population))
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
        if self._use_new_scoring and self._weight_solver is not None and self._weight_solver.has_exact_path():
            pre_ls = best_score
            _ls_pool = sorted(population, key=lambda x: x.fitness, reverse=True)[:30]
            best_individual, best_score = self._exact_swap_local_search(
                best_individual, best_score, start_pool=_ls_pool)
            if best_score > pre_ls + 1e-12:
                self.log("INFO", f"Local search: {best_score:.4f} (was {pre_ls:.4f})")
        elapsed = time.time() - start_time
        tickers = [self.long_candidates[i].variance_asset for i in best_individual.long_indices]
        self.log("SUCCESS", f"Score: {best_score:.4f} | {generations_run + 1} gens | {elapsed:.1f}s (GA={elapsed_ga:.1f}s)")
        self._last_best = best_individual
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
        # CHANGE 8a: Early exit for hit_ratio-only — SLSQP has zero gradient on step function
        active = self._score_fn.weights.active_names
        if active == ["hit_ratio"] or set(active) == {"hit_ratio"}:
            self.log("DEBUG", "Refinement skipped: hit_ratio-only (no gradient for SLSQP)")
            return (current_best, current_best_score)

        best = current_best
        best_score = current_best_score
        ctx = self._make_ctx()  # ws-agnostic fallback; per-basket ctx built below
        _pass_strikes = (self.c.max_net_strike > 0) or self._ws_active()

        def _ctx_for(ind_obj, long_w) -> ScoreContext:
            if not self._ws_active():
                return ctx
            return self._make_ctx(self._net_strike(
                ind_obj.long_indices[:len(long_w)], long_w,
                ind_obj.short_indices, ind_obj.short_weights))
        # Ensure weight solver uses the SAME fitted score function (not a stale ref)
        self._weight_solver._score_fn = self._score_fn
        n_infeasible = 0
        n_no_improve = 0
        n_improved = 0

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
            inc_long_ids = [_candidate_key(self.long_candidates[i], self.is_cross_corridor) for i in best.long_indices]
            inc_long_pos = np.array([self._col_pos[c] for c in inc_long_ids if c in self._col_pos])
            if len(inc_long_pos) > 0:
                inc_pnl = self._adaptive_net_pnl(inc_long_pos, best.long_weights)
                inc_raw = self._score_fn.raw_metrics(
                    inc_pnl, _ctx_for(best, best.long_weights)
                ).get(metric_name, -np.inf if metric_obj.higher_is_better else np.inf)
            else:
                inc_raw = -np.inf if metric_obj.higher_is_better else np.inf
        elif is_concave_blend and not has_hit_ratio:
            # Compute scalarized raw objective for incumbent
            w_dict = self._score_fn.weights
            lam_min = w_dict.get("min_payoff", 0.0)
            lam_mean = w_dict.get("mean_payoff", 0.0)
            lam_carry = w_dict.get("last_carry", 0.0)
            K_carry = 1
            for m in self._score_fn.metrics:
                if m.name == "last_carry" and hasattr(m, "_k"):
                    K_carry = m._k
                    break

            def _raw_blend_obj(pnl_series):
                if len(pnl_series) == 0:
                    return -np.inf
                val = 0.0
                if lam_min > 1e-12:
                    val += lam_min * float(np.min(pnl_series))
                if lam_mean > 1e-12:
                    val += lam_mean * float(np.mean(pnl_series))
                if lam_carry > 1e-12:
                    tail = pnl_series[-K_carry:] if K_carry <= len(pnl_series) else pnl_series
                    val += lam_carry * float(np.mean(tail))
                return val

            inc_long_ids = [_candidate_key(self.long_candidates[i], self.is_cross_corridor) for i in best.long_indices]
            inc_long_pos = np.array([self._col_pos[c] for c in inc_long_ids if c in self._col_pos])
            if len(inc_long_pos) > 0:
                inc_pnl = self._adaptive_net_pnl(inc_long_pos, best.long_weights)
                inc_raw = _raw_blend_obj(inc_pnl)
            else:
                inc_raw = -np.inf

        for ind in candidates:
            long_ids = [_candidate_key(self.long_candidates[i], self.is_cross_corridor) for i in ind.long_indices]
            long_pos = np.array([self._col_pos[c] for c in long_ids if c in self._col_pos])
            if len(long_pos) == 0:
                continue
            n_long = len(long_pos)
            strikes = self._solver_strikes(ind.long_indices[:n_long], self.long_candidates)
            per_stock_bounds = np.array([
                [self.long_candidates[i].min_weight, self.long_candidates[i].max_weight]
                for i in ind.long_indices[:n_long]
            ], dtype=np.float64)
            result = self._weight_solver.solve(
                pnl_matrix=self._ts_mat,
                stock_indices=long_pos,
                strikes=strikes if _pass_strikes else None,
                per_stock_bounds=per_stock_bounds,
                active_mask=self._valid_mask,
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
            if n_nz < 50:
                continue
            _cand_ctx = _ctx_for(ind, result.weights)
            score = self._score_fn.score(net_pnl, _cand_ctx)

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
                # Any config containing hit_ratio with other metrics: keep step-score behavior
                if score > best_score:
                    n_improved += 1
                    best_score = score
                    ind.long_weights = result.weights
                    ind.fitness = score
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
            time_budget = min(8.0, max(2.0, 0.4 * float(self.c.time_limit_seconds)))
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
            ids = [_candidate_key(self.long_candidates[i], self.is_cross_corridor) for i in indices]
            pos = np.array([self._col_pos[c] for c in ids if c in self._col_pos])
            if len(pos) != len(indices):
                return None
            strikes = self._solver_strikes(indices, self.long_candidates)
            bounds = np.array([
                [self.long_candidates[i].min_weight, self.long_candidates[i].max_weight]
                for i in indices], dtype=np.float64)
            res = self._weight_solver.solve(
                pnl_matrix=self._ts_mat, stock_indices=pos,
                strikes=strikes if _pass_strikes else None,
                per_stock_bounds=bounds, active_mask=self._valid_mask,
            )
            if not res.feasible:
                return None
            if not self._is_strike_valid(indices, res.weights,
                                         best.short_indices, best.short_weights):
                return None
            net = self._adaptive_net_pnl(pos, res.weights, _short_pos, _short_w)
            if int(np.count_nonzero(net)) < 50:
                return None
            ws = (self._net_strike(indices, res.weights,
                                   best.short_indices, best.short_weights)
                  if self._ws_active() else None)
            ctx = self._make_ctx(ws)
            return _raw_scalar(net, ctx), self._score_fn.score(net, ctx), res.weights

        inc = _eval_subset(list(best.long_indices))
        if inc is None:
            return best, best_score
        inc_scalar, inc_score, inc_w = inc
        forced = set(self._forced_long)
        n_evals = 0
        improved_any = False

        def _descend_1swap(indices, scalar, score, w):
            """Best-improvement 1-swap descent to a local optimum (or budget)."""
            nonlocal n_evals
            for _sweep in range(6):
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
            if len(queue) >= 4:  # incumbent + up to 3 elite seeds
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

        _dlog(f"[LOCAL-SEARCH] evals={n_evals} improved={improved_any} scalar={best_scalar:.6f}")
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
            L = adaptive_pnl(self._ts_mat, long_pos, long_w, self._valid_mask)

            # Short leg adaptive (if any)
            if short_pos is not None and len(short_pos) > 0:
                short_pos = np.asarray(short_pos, dtype=int)
                short_w = np.asarray(short_w, dtype=np.float64)
                S = adaptive_pnl(self._ts_mat, short_pos, short_w, self._valid_mask)
            else:
                S = np.zeros(self._n_rows)

        elif use_drop:
            all_pos = np.concatenate([long_pos, short_pos]) if (short_pos is not None and len(short_pos) > 0) else long_pos
            valid_rows = ~np.isnan(self._orig_ts_mat[:, all_pos]).any(axis=1)
            if valid_rows.sum() < 50:
                return np.zeros(0)
            L = self._orig_ts_mat[valid_rows][:, long_pos] @ long_w
            if short_pos is not None and len(short_pos) > 0:
                short_pos = np.asarray(short_pos, dtype=int)
                short_w = np.asarray(short_w, dtype=np.float64)
                S = self._orig_ts_mat[valid_rows][:, short_pos] @ short_w
            else:
                S = np.zeros(int(valid_rows.sum()))
            net = (L + S) if self.is_cross_corridor else (L - S)
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

        # For cross-corridor: net = L + S (short_weights applied to net-of-legs PnL)
        # For standard: net = L - S
        net = (L + S) if self.is_cross_corridor else (L - S)
        if self.global_cap < 9999998 or self.global_floor > -9999998:
            net = np.clip(net, self.global_floor, self.global_cap)
        return net

    def _fitness(self, individual: _Individual) -> float:
        """
        Validate every individual before scoring, regardless of whether it came
        from random generation, crossover, mutation or a solver.
        """

        long_w = np.asarray(individual.long_weights, dtype=np.float64)

        if len(long_w) != len(individual.long_indices):
            return 0.0

        long_min = np.array(
            [self.long_candidates[i].min_weight for i in individual.long_indices],
            dtype=np.float64,
        )
        long_max = np.array(
            [self.long_candidates[i].max_weight for i in individual.long_indices],
            dtype=np.float64,
        )

        # Try to repair onto the bounded simplex.
        if long_min.sum() > 1.0 + 1e-10 or long_max.sum() < 1.0 - 1e-10:
            return 0.0

        long_w = self._project_to_simplex_bounded(
            long_w,
            long_min,
            long_max,
        )

        if (
                long_w is None
                or not np.all(np.isfinite(long_w))
                or not np.isclose(long_w.sum(), 1.0, atol=1e-8)
                or np.any(long_w < long_min - 1e-8)
                or np.any(long_w > long_max + 1e-8)
        ):
            return 0.0

        individual.long_weights = long_w

        if individual.short_indices:
            short_w = np.asarray(individual.short_weights, dtype=np.float64)

            if len(short_w) != len(individual.short_indices):
                return 0.0

            short_min = np.array(
                [self.short_candidates[i].min_weight for i in individual.short_indices],
                dtype=np.float64,
            )
            short_max = np.array(
                [self.short_candidates[i].max_weight for i in individual.short_indices],
                dtype=np.float64,
            )

            if short_min.sum() > 1.0 + 1e-10 or short_max.sum() < 1.0 - 1e-10:
                return 0.0

            short_w = self._project_to_simplex_bounded(
                short_w,
                short_min,
                short_max,
            )

            if (
                    short_w is None
                    or not np.all(np.isfinite(short_w))
                    or not np.isclose(short_w.sum(), 1.0, atol=1e-8)
                    or np.any(short_w < short_min - 1e-8)
                    or np.any(short_w > short_max + 1e-8)
            ):
                return 0.0

            individual.short_weights = short_w

        if not self._is_strike_valid(
                individual.long_indices,
                individual.long_weights,
                individual.short_indices,
                individual.short_weights,
        ):
            return 0.0

        if (
                self._use_new_scoring
                and self._score_fn is not None
                and self._score_fn.is_fitted
        ):
            return self._calculate_fitness_v2(individual)

        raise RuntimeError(
            "ScoreFunction not fitted — reference sample failed; cannot optimize. "
            f"use_new_scoring={self._use_new_scoring}, "
            f"score_fn={'None' if self._score_fn is None else 'present'}, "
            f"is_fitted={self._score_fn.is_fitted if self._score_fn is not None else 'N/A'}"
        )


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
        _dlog(f"  [COMPUTE-NET] use_adaptive={use_adaptive}, use_drop={use_drop}, is_cross_corridor={self.is_cross_corridor}")
        if use_adaptive:
            L_mat = self._orig_ts_mat[:, long_pos]
            L_nan = np.isnan(L_mat)
            L_w_matrix = np.tile(long_w, (self._n_rows, 1))
            L_w_matrix[L_nan] = 0.0
            L_w_sums = L_w_matrix.sum(axis=1, keepdims=True)
            L_w_sums[L_w_sums == 0] = 1.0
            L_w_matrix = L_w_matrix / L_w_sums
            L = (np.nan_to_num(L_mat, nan=0.0) * L_w_matrix).sum(axis=1)
            if len(short_pos) > 0:
                S_mat = self._orig_ts_mat[:, short_pos]
                S_nan = np.isnan(S_mat)
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
            if valid_rows.sum() < 50:
                _dlog(f"  [COMPUTE-NET] use_drop: valid_rows.sum()={valid_rows.sum()} < 50")
                return None
            L = self._orig_ts_mat[valid_rows][:, long_pos] @ long_w
            S = (self._orig_ts_mat[valid_rows][:, short_pos] @ short_w) if len(short_pos) else 0.0
        else:
            L = self._ts_mat[:, long_pos] @ long_w
            S = (self._ts_mat[:, short_pos] @ short_w) if len(short_pos) else 0.0
            valid_rows = np.ones(self._n_rows, dtype=bool)
            _dlog(f"  [COMPUTE-NET] default: L.shape={L.shape}, S={S}, valid_rows.sum()={valid_rows.sum()}")
        if valid_rows.sum() < 50:
            _dlog(f"  [NET-PNL] valid_rows < 50: {valid_rows.sum()}")
            return None
        net = (L + S) if self.is_cross_corridor else (L - S)
        _dlog(f"  [COMPUTE-NET] net.shape={net.shape}, net[:5]={net[:5]}")
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
        _dlog(f"  [NET-PNL] len={len(net)}, nonzero={nonzero_mask.sum()}, trailing_zeros={n_trailing_zeros}")
        if nonzero_mask.sum() < 50:
            _dlog(f"  [NET-PNL] NONZERO < 50 - returning None")
            return None
        # ── ALIGNED WITH BACKTESTER: return full series including zeros ──
        # Cross-corridor backtester keeps all rows. Zeros are valid observations.
        _dlog(f"  [NET-PNL] FINAL: len={len(net)}, last_val={net[-1]:.8f}, last5={[f'{v:.6f}' for v in net[-5:]]}")
        return net

    def _build_reference_sample(self, n_samples: int = 2000) -> None:
        """Build the reference sample for quantile normalisation.
        Generates random feasible baskets, computes their net P&L, and fits
        the ScoreFunction normalizer.  Called once before the GA loop.
        """
        ref_start = time.time()
        self.log("INFO", f"Building reference sample ({n_samples} baskets)...")
        ctx = ScoreContext(n_days=self._n_rows)
        sample_pnls = []
        sample_extras = []  # aligned with sample_pnls: {"weighted_strike": ...}
        attempts = 0
        max_attempts = n_samples * 3
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
            if r < 0.4:
                long_w = np.full(n_long, 1.0 / n_long)
            elif r < 0.7:
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
            elif r < 0.9:
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
                    best_alloc = lb[best_col] + 0.6 * (ub[best_col] - lb[best_col])
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
                            add = min(remaining, ub[idx2] - long_w[idx2], remaining * share * 1.5)
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
            elif len(net_pnl) < 50:
                reason = "len < 50"
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            else:
                sample_pnls.append(net_pnl)
                sample_extras.append({
                    "weighted_strike": self._net_strike(
                        long_indices, long_w, short_indices, short_w)
                })

        # --- ELITE SEEDING: inject baskets from top-ranked stocks so normalizer ceiling is realistic ---
        all_col_means = []
        for i, cand in enumerate(self.long_candidates):
            key = _candidate_key(cand, self.is_cross_corridor)
            if key in self._col_pos:
                col = self._col_pos[key]
                all_col_means.append((i, self._ts_mat[:, col].mean()))
        all_col_means.sort(key=lambda x: -x[1])  # descending by mean
        n_elite_baskets = min(50, n_samples // 10)
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
            if elite_pnl is not None and len(elite_pnl) >= 50:
                sample_pnls.append(elite_pnl)
                sample_extras.append({
                    "weighted_strike": self._net_strike(
                        long_indices_e, long_w_e, short_indices_e, short_w_e)
                })

        if len(sample_pnls) < 100:
            self.log("WARN", f"⚠️ Reference sample too small ({len(sample_pnls)}/{n_samples} after {attempts} attempts), falling back to legacy scoring")
            self._use_new_scoring = False
            self._scoring_mode = "legacy"
            return
        # FIT the score function with collected samples (extras carry per-sample
        # weighted strikes so the strike objective normalises like any metric)
        self._score_fn.build_reference(sample_pnls, ctx, sample_extras=sample_extras)
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
                _dlog(f"\n❌ [METRIC-REJECT] metric '{m.name}' has zero spread (std={np.std(vals):.2e})")
                _dlog(f"   All {len(vals)} values ≈ {vals[0]:.6f}")
                _dlog(f"   First 10 sample PnL stats:")
                for i in range(min(10, len(sample_pnls))):
                    pnl = sample_pnls[i]
                    last_val = float(np.mean(pnl[-1:]))
                    mean_ret = float(np.mean(pnl))
                    cumsum = float(np.sum(pnl))
                    _dlog(f"     Basket-{i}: last_carry={last_val:.6f}, mean={mean_ret:.6f}, cumsum={cumsum:.6f}, len={len(pnl)}")
                raise RuntimeError(
                    f"Reference sample non-degenerate check failed: metric '{m.name}' "
                    f"has zero spread (all values ≈ {vals[0]:.6f}). The quantile normalizer "
                    f"will saturate and this metric will have no effect on scoring. "
                    f"Check that the weight strategy mix produces diverse baskets."
                )
        ref_elapsed = time.time() - ref_start
        self.log("INFO", f"✅ Reference sample fitted ({len(sample_pnls)} baskets, {attempts} attempts) → bilevel scoring active [{ref_elapsed:.1f}s]")
        _dlog(f"🔬 [REF] {len(sample_pnls)} baskets in {ref_elapsed:.1f}s")

    def _ws_active(self) -> bool:
        """True when the weighted_strike objective carries a positive weight."""
        return (self._metric_weights is not None
                and "weighted_strike" in self._metric_weights.active_names)

    def _make_ctx(self, weighted_strike: Optional[float] = None) -> ScoreContext:
        """Run-level ScoreContext, optionally carrying the basket's net strike."""
        return ScoreContext(n_days=self._n_rows, weighted_strike=weighted_strike)

    def _calculate_fitness_v2(self, individual: _Individual) -> float:
        """Bilevel fitness — adaptive path for the GA loop.
        - If ALL active metrics are linear: use full WeightSolver (LP path, ~0.01ms
          with cache hits). This gives exact optimal weights during GA.
        - If any metric is non-linear: use the INDIVIDUAL's OWN weights for scoring.
          The GA evolves both stock selection AND weights through crossover/mutation.
        Full SLSQP refinement on top elites happens post-GA (see _refine_top_individuals).
        """
        self.log("DEBUG", f"[FITNESS-v2] all_linear={self._all_metrics_linear} | path={'LP' if self._all_metrics_linear else 'OWN-WEIGHTS'}")
        # ── Use _candidate_key for cross-corridor vs mono ──
        long_ids = [_candidate_key(self.long_candidates[i], self.is_cross_corridor) for i in individual.long_indices]
        long_pos = np.array([self._col_pos[c] for c in long_ids if c in self._col_pos])
        if len(long_pos) == 0:
            return 0.0
        n_long = len(long_pos)
        # P2: Adaptive dispatch — inline exact solve when all active metrics are
        # linear (LP is microseconds + cached), or when the user forces exact-in-GA.
        if ((self._all_metrics_linear or self._use_exact_in_ga)
                and self._weight_solver is not None and self._weight_solver.has_exact_path()):
            strikes = self._solver_strikes(individual.long_indices[:n_long], self.long_candidates)
            _pass_strikes = (self.c.max_net_strike > 0) or self._ws_active()
            # Per-stock bounds from Stock objects
            per_stock_bounds = np.array([
                [self.long_candidates[i].min_weight, self.long_candidates[i].max_weight]
                for i in individual.long_indices[:n_long]
            ], dtype=np.float64)
            # Strike visibility log — first call only
            if not self._strike_logged:
                self._strike_logged = True
                self.log("INFO", f"[STRIKE-CHECK] strikes.min={strikes.min():.4f}, strikes.max={strikes.max():.4f}, max_net_strike={self.c.max_net_strike}")
            result = self._weight_solver.solve(
                pnl_matrix=self._ts_mat,
                stock_indices=long_pos,
                strikes=strikes if _pass_strikes else None,
                per_stock_bounds=per_stock_bounds,
                active_mask=self._valid_mask,
            )
            if not result.feasible:
                self._lp_infeasible_streak += 1
                if self._lp_infeasible_streak >= 20:
                    raise RuntimeError(
                        f"First 20 LP solves all infeasible — likely units mismatch. "
                        f"strikes.min={strikes.min():.4f}, strikes.max={strikes.max():.4f}, "
                        f"max_net_strike={self.c.max_net_strike}"
                    )
                        # LP infeasible
                return 0.0
            self._lp_infeasible_streak = 0
            w = result.weights
            # LP path: update individual's weights since LP gives optimal solution
            individual.long_weights = w
        else:
            # Use individual's OWN weights (evolved by GA operators)
            w = individual.long_weights
            _dlog(f"  GA weights: {w} (len={len(w) if w is not None else 0})")
            if w is None or len(w) != n_long:
                # Fallback: if weights missing or mismatched, generate fresh ones
                _dlog("  GA weights missing/mismatch - calling _optimize_weights_for_basket")
                w = self._optimize_weights_for_basket(
                    individual.long_indices[:n_long], self.long_candidates, self._quality_long)
                individual.long_weights = w
                _dlog(f"  GA weights after opt: {w}")
        # Compute net P&L using adaptive formula (mirrors backtester exactly)
        short_indices = individual.short_indices
        short_pos_arr = None
        short_w = None
        if len(short_indices) > 0:
            short_ids = [_candidate_key(self.short_candidates[i], self.is_cross_corridor) for i in short_indices]
            short_pos_arr = np.array([self._col_pos[c] for c in short_ids if c in self._col_pos])
            short_w = individual.short_weights
            if len(short_pos_arr) == 0:
                short_pos_arr = None
                short_w = None
        net_pnl_raw = self._adaptive_net_pnl(long_pos, w, short_pos_arr, short_w)
        _ws = (self._net_strike(individual.long_indices[:n_long], w,
                                short_indices, short_w)
               if self._ws_active() else None)
        ctx = self._make_ctx(_ws)
        # ── ALIGNED WITH BACKTESTER: adaptive reweight handles NaN, no strip ──
        _dlog(f"  net_pnl_raw shape={net_pnl_raw.shape}, mean={net_pnl_raw.mean():.6f}, std={net_pnl_raw.std():.6f}")
        n_nonzero = int(np.count_nonzero(net_pnl_raw))
        _dlog(f"  nonzero={n_nonzero} / {len(net_pnl_raw)}")
        if n_nonzero < 50:
            self._rejection_reasons["len_valid_lt_50"] += 1
            return 0.0
        score = self._score_fn.score(net_pnl_raw, ctx)
        if np.isnan(score) or np.isinf(score):
            tickers = [self.long_candidates[i].variance_asset for i in individual.long_indices[:n_long]]
            _dlog(f"❌ [SCORE-REJECT] tickers={tickers}, reason='invalid score ({score})'")
            self._rejection_reasons["invalid_score"] += 1
            return 0.0
        # Tie-break: when quantile score saturates at top, use raw-based rank
        # so tournament selection distinguishes between top-percentile baskets.
        if score >= 0.99:
            raw = self._score_fn.raw_metrics(net_pnl_raw, ctx)
            lam = self._score_fn.weights
            metric_map = {m.name: m for m in self._score_fn.metrics}
            scalar = sum(
                lam.get(name, 0.0) * raw[name] * (1.0 if metric_map[name].higher_is_better else -1.0)
                for name in lam.active_names
            )
            return float(2.0 + math.tanh(scalar))   # always in (1.0, 3.0): beats any quantile score, monotone in scalar
        return float(score)
    # ══════════════════════════════════════════════════════════════════════════
    # WEIGHT ALLOCATION (original quality-biased bounded allocator)
    # ══════════════════════════════════════════════════════════════════════════

    def _optimize_weights_for_basket(self, indices, candidates, quality_arr):
        """
        Generate quality-biased weights satisfying:
          - sum(weights) == 1
          - min_weight <= weight <= max_weight
        Returns None when the selected basket is infeasible.
        """
        n = len(indices)
        if n == 0:
            return np.array([], dtype=np.float64)

        min_w = np.array(
            [candidates[i].min_weight for i in indices],
            dtype=np.float64,
        )
        max_w = np.array(
            [candidates[i].max_weight for i in indices],
            dtype=np.float64,
        )

        # The selected basket cannot reach 100% within its individual bounds.
        if (
                min_w.sum() > 1.0 + 1e-10
                or max_w.sum() < 1.0 - 1e-10
        ):
            return None

        q = quality_arr[indices]

        # Start from the midpoint of each permitted range.
        w = (min_w + max_w) / 2.0

        # Apply the existing quality bias.
        q_range = float(q.max() - q.min())
        if q_range > 1e-12:
            q_norm = (q - q.mean()) / q_range
            w = w + q_norm * (max_w - min_w) * 0.4

        # Project onto sum=1 while respecting individual bounds.
        w = self._project_to_simplex_bounded(w, min_w, max_w)

        # Never allow an invalid basket into the GA.
        if (
                w is None
                or len(w) != n
                or not np.all(np.isfinite(w))
                or not np.isclose(w.sum(), 1.0, atol=1e-8)
                or np.any(w < min_w - 1e-8)
                or np.any(w > max_w + 1e-8)
        ):
            return None

        return w
    def _project_to_simplex_bounded(self, weights: np.ndarray, min_weights: np.ndarray, max_weights: np.ndarray) -> np.ndarray:
        """Project weights onto the bounded simplex {w: sum=1, min<=w<=max}.
        Uses iterative proportional fitting instead of clip-normalize cycles
        which collapse weights to boundaries.
        """
        n = len(weights)
        if n == 0:
            return weights
        w = np.clip(weights, min_weights, max_weights)
        _dlog(f"  _project_to_simplex_bounded: initial sum={w.sum():.6f}")
        # Iterative adjustment: redistribute excess/deficit proportionally
        for iteration in range(15):
            s = w.sum()
            _dlog(f"  iter {iteration}: sum={s:.6f}")
            if abs(s - 1.0) < 1e-10:
                break
            if s > 1.0:
                # Need to reduce — take from stocks proportional to their room above min
                excess = s - 1.0
                room = w - min_weights
                room_total = room.sum()
                if room_total < 1e-12:
                    break
                reduction = room * (excess / room_total)
                w = w - reduction
                w = np.maximum(w, min_weights)
            else:
                # Need to increase — give to stocks proportional to their room below max
                deficit = 1.0 - s
                room = max_weights - w
                room_total = room.sum()
                if room_total < 1e-12:
                    break
                addition = room * (deficit / room_total)
                w = w + addition
                w = np.minimum(w, max_weights)
        else:
            # Did not converge in 15 iterations
            _dlog(f"⚠️ [SIMPLEX] did NOT converge in 15 iters! sum={w.sum():.6f} n={n} "
                  f"sum_min={min_weights.sum():.4f} sum_max={max_weights.sum():.4f}")
        _dlog(f"  final w={w}, sum={w.sum():.6f}")
        return w

    def _perturb_weights(self, weights: np.ndarray, min_weights: np.ndarray, max_weights: np.ndarray) -> np.ndarray:
        """Add random noise to weights and project back onto bounded simplex.
        Uses proportional redistribution instead of clip-normalize cycles
        to preserve intermediate weight values.
        """
        range_w = max_weights - min_weights
        base_sigma = 0.08  # stronger perturbation to explore mid-range
        noise = self._np_rng.normal(0, base_sigma, size=len(weights)) * range_w
        w = weights + noise
        result = self._project_to_simplex_bounded(w, min_weights, max_weights)
        return result

    def _random_weights_for_basket(self, indices, candidates) -> Optional[np.ndarray]:
        """Generate random weights within bounds, normalized to sum=1.
        Unlike _optimize_weights_for_basket, this ignores quality scores
        to give the GA freedom to explore the weight landscape.
        Uses Dirichlet-based sampling projected onto bounded simplex to
        produce genuinely intermediate weights (not just min/max)."""
        n = len(indices)
        if n == 0:
            return np.array([], dtype=np.float64)
        min_w = np.array([candidates[i].min_weight for i in indices], dtype=np.float64)
        max_w = np.array([candidates[i].max_weight for i in indices], dtype=np.float64)
        # Scale mins if infeasible
        sum_min = float(min_w.sum())
        if sum_min > 1.0 + 1e-12:
            min_w = min_w / sum_min
            max_w = np.maximum(max_w, min_w)
        # Check feasibility: can the bounded simplex even exist?
        sum_max = float(max_w.sum())
        if sum_max < 1.0 - 1e-12:
            _dlog(f"⚠️ [RANDOM-W] INFEASIBLE: sum_max={sum_max:.4f} < 1.0 for n={n} stocks! "
                  f"min_w={min_w[:5]} max_w={max_w[:5]}")
            # Fallback: equal weights
            w = np.full(n, 1.0 / n)
            return np.clip(w, min_w, max_w)
        # Dirichlet sampling with alpha>1 biases toward center of simplex
        # alpha=2 gives bell-shaped distribution away from boundaries
        alpha = np.full(n, 2.0)
        w = self._np_rng.dirichlet(alpha)
        # Scale from unit simplex into [min_w, max_w] range
        range_w = max_w - min_w
        w = min_w + w * range_w * n  # scale up since dirichlet sums to 1
        # Project onto bounded simplex (sum=1, within bounds)
        w = self._project_to_simplex_bounded(w, min_w, max_w)
        return w
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
        forced names.  ``pool`` must not contain the forced indices."""
        forced = self._forced_long
        n_free = n_long - len(forced)
        if n_free < 0:
            return list(forced[:n_long])  # cannot happen after __init__ validation
        if n_free > len(pool):
            n_free = len(pool)
        picked = self._rng.sample(pool, n_free) if n_free > 0 else []
        return sorted(set(picked) | set(forced))

    def _long_size_bounds(self, pool_extra: int) -> Tuple[int, int]:
        """[lo, hi] basket sizes honouring cardinality inputs and forced names.
        ``pool_extra`` = number of NON-forced candidates available."""
        k = len(self._forced_long)
        lo = max(self.c.min_stocks_long, k)
        hi = min(self.c.max_stocks_long, k + pool_extra)
        return lo, max(lo, hi)

    def _create_random_individual(self) -> Optional[_Individual]:
        """Create individual with optimized weights (original logic)."""
        max_attempts = 20
        _free_pool = [i for i in range(len(self.long_candidates)) if i not in self._forced_long]
        for attempt in range(max_attempts):
            _lo, _hi = self._long_size_bounds(len(_free_pool))
            n_long = self._rng.randint(_lo, _hi)
            if self._long_only:
                long_indices = self._pick_long_subset(_free_pool, n_long)
                # 50% quality-biased, 50% random weights for diversity
                if self._rng.random() < 0.5:
                    long_w = self._optimize_weights_for_basket(
                        long_indices, self.long_candidates, self._quality_long)
                else:
                    long_w = self._random_weights_for_basket(long_indices, self.long_candidates)
                if long_w is not None and len(long_w) > 0:
                    ind = _Individual(long_indices, [], long_w, np.array([]))
                    ind.fitness = self._fitness(ind)
                    # DEBUG: Log first 10 candidates with full hit_ratio details
                    if attempt < 10:
                        tickers = [self.long_candidates[i].variance_asset for i in long_indices]
                        _dlog(f"🔍 [CANDIDATE-{attempt}] tickers={tickers}")
                        _dlog(f"  weights={ind.long_weights}, sum={ind.long_weights.sum():.4f}, "
                              f"min={ind.long_weights.min():.4f}, max={ind.long_weights.max():.4f}")
                        _dlog(f"  fitness={ind.fitness:.4f}")
                        # ── VISIBLE DEBUG: print to stdout ──
                    # Track rejection reasons
                    if ind.fitness < 0:
                        self._rejection_reasons["fitness<=0"] += 1
                        continue
                    return ind
                else:
                    self._rejection_reasons["no_weights"] += 1
            else:
                n_short = self._rng.randint(
                    self.c.min_stocks_short,
                    min(self.c.max_stocks_short, len(self.short_candidates))
                )
                long_indices = self._pick_long_subset(_free_pool, n_long)
                short_indices = self._rng.sample(range(len(self.short_candidates)), n_short)
                # 50% quality-biased, 50% random weights for diversity
                if self._rng.random() < 0.5:
                    long_w = self._optimize_weights_for_basket(
                        long_indices, self.long_candidates, self._quality_long)
                    short_w = self._optimize_weights_for_basket(
                        short_indices, self.short_candidates, self._quality_short)
                else:
                    long_w = self._random_weights_for_basket(long_indices, self.long_candidates)
                    short_w = self._random_weights_for_basket(short_indices, self.short_candidates)
                if long_w is not None and short_w is not None and len(long_w) > 0:
                    # Check strike constraint
                    if self._is_strike_valid(long_indices, long_w, short_indices, short_w):
                        ind = _Individual(long_indices, short_indices, long_w, short_w)
                        ind.fitness = self._fitness(ind)
                        return ind
                    else:
                        self._rejection_reasons["strike_invalid"] += 1
                else:
                    self._rejection_reasons["no_weights"] += 1
        return None

    def _tournament_select(self, population: List[_Individual]) -> _Individual:
        tournament = self._rng.sample(population, min(self.c.tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness)

    def _crossover(self, parent1: _Individual, parent2: _Individual) -> Optional[_Individual]:
        """
        Crossover: union of parent indices, then random subset.
        IMPROVED: Inherit parent weights for shared stocks instead of recomputing from scratch.
        """
        # Union of parent long indices, then pick random subset.
        # Forced names are always included (parents already contain them, but
        # the random subset draw must never drop them).
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
        # ── INHERIT WEIGHTS from parents for shared stocks ──
        # Build index→weight maps from both parents
        p1_long_map = dict(zip(parent1.long_indices, parent1.long_weights))
        p2_long_map = dict(zip(parent2.long_indices, parent2.long_weights))
        long_min = np.array([self.long_candidates[i].min_weight for i in long_indices])
        long_max = np.array([self.long_candidates[i].max_weight for i in long_indices])
        # For each child stock: inherit from parent (avg if in both), else random
        long_w = np.zeros(len(long_indices), dtype=np.float64)
        for k, idx in enumerate(long_indices):
            w1 = p1_long_map.get(idx)
            w2 = p2_long_map.get(idx)
            if w1 is not None and w2 is not None:
                long_w[k] = (w1 + w2) / 2.0  # average of both parents
            elif w1 is not None:
                long_w[k] = w1
            elif w2 is not None:
                long_w[k] = w2
            else:
                long_w[k] = long_min[k] + self._rng.random() * (long_max[k] - long_min[k])
        # Project onto bounded simplex (sum=1, within bounds) without clip-normalize collapse
        long_w = self._project_to_simplex_bounded(long_w, long_min, long_max)
        if len(long_w) == 0:
            return None
        # Perturb long weights — occasionally fully randomize for diversity
        if self._rng.random() < 0.3:
            long_w = self._random_weights_for_basket(long_indices, self.long_candidates)
            if long_w is None:
                return None
        else:
            long_w = self._perturb_weights(long_w, long_min, long_max)
        if short_indices:
            p1_short_map = dict(zip(parent1.short_indices, parent1.short_weights))
            p2_short_map = dict(zip(parent2.short_indices, parent2.short_weights))
            short_min = np.array([self.short_candidates[i].min_weight for i in short_indices])
            short_max = np.array([self.short_candidates[i].max_weight for i in short_indices])
            short_w = np.zeros(len(short_indices), dtype=np.float64)
            for k, idx in enumerate(short_indices):
                w1 = p1_short_map.get(idx)
                w2 = p2_short_map.get(idx)
                if w1 is not None and w2 is not None:
                    short_w[k] = (w1 + w2) / 2.0
                elif w1 is not None:
                    short_w[k] = w1
                elif w2 is not None:
                    short_w[k] = w2
                else:
                    short_w[k] = short_min[k] + self._rng.random() * (short_max[k] - short_min[k])
            short_w = self._project_to_simplex_bounded(short_w, short_min, short_max)
            # Perturb short weights — occasionally fully randomize for diversity
            if self._rng.random() < 0.3:
                short_w = self._random_weights_for_basket(short_indices, self.short_candidates)
                if short_w is None:
                    return None
            else:
                short_w = self._perturb_weights(short_w, short_min, short_max)
        else:
            short_w = np.zeros(0, dtype=np.float64)
        # Hard strike constraint
        if not self._is_strike_valid(long_indices, long_w, short_indices, short_w):
            return None
        child = _Individual(long_indices, short_indices, long_w, short_w)
        child.fitness = self._fitness(child)
        return child

    def _mutate(self, individual: _Individual) -> Optional[_Individual]:
        """Mutate: either swap 1 stock OR re-randomize weights only."""
        long_indices = individual.long_indices.copy()
        short_indices = individual.short_indices.copy()
        # 40% chance: weight-only mutation (keep same stocks, randomize weights)
        if self._rng.random() < 0.4:
            long_w = self._random_weights_for_basket(long_indices, self.long_candidates)
            if long_w is None or len(long_w) == 0:
                return None
            if short_indices:
                short_w = self._random_weights_for_basket(short_indices, self.short_candidates)
                if short_w is None or len(short_w) == 0:
                    return None
            else:
                short_w = np.zeros(0, dtype=np.float64)
            if not self._is_strike_valid(long_indices, long_w, short_indices, short_w):
                return None
            mutated = _Individual(long_indices, short_indices, long_w, short_w)
            mutated.fitness = self._fitness(mutated)
            return mutated
        # 60% chance: stock swap mutation (original behavior)
        if self._long_only or self._rng.random() < 0.5:
            # Swap one stock in long leg — forced names are never swapped out
            avail = sorted(set(range(len(self.long_candidates))) - set(long_indices))
            swappable = [p for p in range(len(long_indices))
                         if long_indices[p] not in self._forced_long]
            if avail and swappable:
                pos = self._rng.choice(swappable)
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
        # Recompute bounded weights
        long_w = self._optimize_weights_for_basket(
            long_indices, self.long_candidates, self._quality_long)
        if long_w is None or len(long_w) == 0:
            return None
        # Perturb long weights
        long_min = np.array([self.long_candidates[i].min_weight for i in long_indices])
        long_max = np.array([self.long_candidates[i].max_weight for i in long_indices])
        long_w = self._perturb_weights(long_w, long_min, long_max)
        if short_indices:
            short_w = self._optimize_weights_for_basket(
                short_indices, self.short_candidates, self._quality_short)
            if short_w is None or len(short_w) == 0:
                return None
            # Perturb short weights
            short_min = np.array([self.short_candidates[i].min_weight for i in short_indices])
            short_max = np.array([self.short_candidates[i].max_weight for i in short_indices])
            short_w = self._perturb_weights(short_w, short_min, short_max)
        else:
            short_w = np.zeros(0, dtype=np.float64)
        # Hard strike constraint
        if not self._is_strike_valid(long_indices, long_w, short_indices, short_w):
            return None
        mutated = _Individual(long_indices, short_indices, long_w, short_w)
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
        _dlog(f"EMPTY RESULT: long_only={self._long_only}, is_cross_corridor={self.is_cross_corridor}, "
              f"n_long={len(self.long_candidates)}, n_short={len(self.short_candidates)}")
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
        
        CRITICAL: For cross-corridor, Stock.variance_asset is shared (e.g. 'SPX Index') across
        all candidates. Must use _candidate_key (= Stock.corridor_condition_asset) as the unique basket key
        to avoid dict collapse.
        
        ORDER: Safety-net runs FIRST (may update best.long_weights/fitness),
        then ALL extractions (basket, strikes, FINAL-RAW) read from the final weights.
        """
        is_xcorr = self.is_cross_corridor
        
        # ═══ SAFETY NET (runs BEFORE any weight extraction) ═══
        # NEVER DOWNGRADE: keep GA weights if they score better on the full adaptive curve.
        if self._weight_solver.has_exact_path():
            try:
                long_pos_final = [self._col_pos[_candidate_key(self.long_candidates[i], is_xcorr)]
                                  for i in best.long_indices
                                  if _candidate_key(self.long_candidates[i], is_xcorr) in self._col_pos]
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
                    active_mask=self._valid_mask,
                    tol="fine",
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

                    # bisect_pnl = self._adaptive_net_pnl(long_pos_final, resolv.weights)
                    # bisect_min = float(np.min(bisect_pnl))
                    # bisect_score = resolv.score

                    resolv_w = self._project_to_simplex_bounded(
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


                    print(f"[BISECT-VS-GA] ga_min={ga_min:.4f} ga_score={ga_score:.4f} | bisect_min={bisect_min:.4f} bisect_score={bisect_score:.4f}", flush=True)

                    # NEVER DOWNGRADE: pick the better weight set by user's blend objective
                    # TIE-BREAK: when scores are within 1e-4, prefer higher raw min_payoff
                    if abs(bisect_score - ga_score) < 1e-4:
                        winner = "bisect" if bisect_min > ga_min else "ga"
                        print(f"[SAFETY-NET] tie on score ({ga_score:.4f} vs {bisect_score:.4f}), min decides: {winner} (ga_min={ga_min:.4f} bisect_min={bisect_min:.4f})", flush=True)
                        if winner == "bisect":
                            best.long_weights = resolv_w
                            best.fitness = bisect_score
                        else:
                            best.fitness = ga_score
                    elif bisect_score > ga_score:
                        best.long_weights = resolv_w
                        best.fitness = bisect_score
                        print(f"[SAFETY-NET] UPGRADED: bisection wins (score {ga_score:.4f} -> {bisect_score:.4f}, min {ga_min:.4f} -> {bisect_min:.4f})", flush=True)
                    else:
                        # Keep GA weights — they score equal or better
                        best.fitness = ga_score
                        print(f"[SAFETY-NET] KEPT GA: ga wins (score {ga_score:.4f} >= {bisect_score:.4f}, min {ga_min:.4f} vs {bisect_min:.4f})", flush=True)
                else:
                    # Bisection infeasible — keep GA weights, re-score them
                    ga_pnl = self._adaptive_net_pnl(long_pos_final, best.long_weights)
                    _ga_ws2 = (self._net_strike(best.long_indices, best.long_weights,
                                                best.short_indices, best.short_weights)
                               if self._ws_active() else None)
                    best.fitness = self._score_fn.score(ga_pnl, self._make_ctx(_ga_ws2))
                    print(f"[SAFETY-NET] bisection infeasible — kept GA weights", flush=True)
            except Exception as e:
                print(f"[SAFETY-NET] failed: {e}", flush=True)

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
                    active_mask=self._valid_mask,
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
                print(f"[SMOOTH] min: {star_min:.4f} -> {smooth_min:.4f} | mean: {star_mean:.4f} -> {smooth_mean:.4f} | dispersion(std): {star_std:.4f} -> {smooth_std:.4f} | eps={self._smooth_eps:.2f}", flush=True)
                # Per-ticker weight comparison, sorted by |delta| descending
                print("[SMOOTH-WEIGHTS] ticker | w_optimal | w_smooth | delta", flush=True)
                _rows = []
                for i, (wo, ws) in enumerate(zip(w_star, w_smooth)):
                    tk = _candidate_key(self.long_candidates[best.long_indices[i]], is_xcorr)
                    _rows.append((tk, float(wo), float(ws), float(ws - wo)))
                _rows.sort(key=lambda r: abs(r[3]), reverse=True)
                for tk, wo, ws, d in _rows:
                    print(f"[SMOOTH-WEIGHTS] {tk:20s} | {wo:.2f} | {ws:.2f} | {d:+.2f}", flush=True)
                # Store unsmoothed basket on optimizer for API dual-backtest access
                self._last_unsmoothed_basket = [
                    (_candidate_key(self.long_candidates[i], is_xcorr), float(w))
                    for i, w in zip(best.long_indices, w_star)
                ]
                best.long_weights = w_smooth
            except Exception as e:
                print(f"[SMOOTH] failed: {e}, keeping original weights", flush=True)
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

        # ═══ STRIKE AUDIT LOG ═══
        _binding = net_strike_val >= (self.c.max_net_strike - 1e-6) if self.c.max_net_strike > 0 else False
        _violation = net_strike_val > self.c.max_net_strike + 1e-4 if self.c.max_net_strike > 0 else False
        _convention = "net_spread (stock-index)" if is_xcorr else "mono_var_swap"
        print(f"[STRIKE] convention={_convention} | basket={net_strike_val:.4f} | limit={self.c.max_net_strike:.4f} | binding={_binding}", flush=True)
        if _violation:
            print(f"⚠️ [STRIKE-WARNING] Delivered basket VIOLATES strike limit: {net_strike_val:.4f} > {self.c.max_net_strike:.4f}", flush=True)
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
            print(f"[FINAL-RAW] " + " | ".join(f"{k}={v:.4f}" for k, v in raw_final.items()), flush=True)
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
        )

    @property
    def _has_unsmoothed(self):
        """Check if last run produced unsmoothed weights for dual backtest."""
        return self._smooth_weights

    def get_unsmoothed_basket(self) -> list:
        """Return basket with pre-smooth weights, or None."""
        return self._last_unsmoothed_basket

    # ── MILP benchmark (called externally if run_milp=True) ──

    def milp_benchmark(self) -> Optional[Dict[str, float]]:
        """Solve the exact MILP for min_payoff-only config. Returns None if not applicable."""
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import csr_matrix as _csr

        if not self._weight_solver._is_min_payoff_only():
            print("[MILP] Not min_payoff-only config — skipping.", flush=True)
            return None

        is_xcorr = self.is_cross_corridor

        # ── Use EXACTLY self._ts_mat — ALL columns, same object as _calculate_fitness_v2 ──
        P = self._ts_mat.copy()
        n_stocks = P.shape[1]  # ALL columns = full candidate universe

        if n_stocks == 0:
            print("[MILP] No valid candidates.", flush=True)
            return None

        # Apply same cap/floor as GA evaluation (line ~1183 of _calculate_fitness_v2)
        if self.global_cap < 9999998 or self.global_floor > -9999998:
            P = np.clip(P, self.global_floor, self.global_cap)

        # DO NOT drop all-zero rows — GA keeps them (zeros count as non-positive days
        # in hit_ratio, and removing them changes the denominator for all metrics).
        # The GA's _calculate_fitness_v2 only strips zeros AFTER computing net_pnl
        # for a specific basket selection, not at the matrix level.
        n_days = P.shape[0]

        print(f"[MILP] matrix shape={P.shape} | ga_matrix shape={self._ts_mat.shape} | id(ga_mat)={id(self._ts_mat)}", flush=True)

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
        for col_idx in range(n_stocks):
            if col_idx in col_to_cand:
                cand_idx = col_to_cand[col_idx]
                leg = self.long_candidates[cand_idx]
                strikes_per_col[col_idx] = leg.strike_mono_var_swap if leg.strike_mono_var_swap is not None else 0.0
                min_w_per_col[col_idx] = leg.min_weight
                max_w_per_col[col_idx] = leg.max_weight
            else:
                # Column exists in matrix but has no candidate mapping — allow MILP to select it
                min_w_per_col[col_idx] = self._weight_solver._constraints.min_weight
                max_w_per_col[col_idx] = self._weight_solver._constraints.max_weight

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

        print(f"[MILP] n_candidates={n_stocks}, n_days={n_days}, min_k={min_k}, max_k={max_k} | approx=(row-level zero drop, not per-basket)", flush=True)

        res = milp(
            c=obj,
            constraints=constraints,
            integrality=integrality,
            bounds=Bounds(lb, ub),
            options={"time_limit": 300.0, "disp": False},
        )

        if not res.success:
            bound = -res.fun if hasattr(res, 'fun') and res.fun is not None else None
            print(f"[MILP] Failed: {res.message} | bound={bound}", flush=True)
            return {"status": "failed", "message": res.message, "bound": bound}

        milp_t = res.x[-1]
        milp_w = res.x[:n_stocks]
        milp_z = res.x[n_stocks:2*n_stocks]
        selected = int(np.round(milp_z).sum())

        print(f"[MILP] Exact optimum: min_payoff={milp_t:.6f} | selected={selected} stocks", flush=True)
        return {"status": "optimal", "min_payoff": milp_t, "n_selected": selected, "weights": milp_w, "z": milp_z}
