"""
weight_solver.py — Inner-level weight optimisation for the bilevel optimizer
=============================================================================

Given a *fixed* subset of stocks S, find the weight vector w* that maximises
the scoring function, subject to portfolio constraints:

::

    w* = argmax_w  score_smooth(net_pnl(w), ctx)
    s.t.  sum(w) == 1
          min_weight <= w_i <= max_weight  for all i
          weighted_mean_strike(w) <= max_net_strike   (optional)

For **linear metrics** (last_carry, mean_payoff) the net P&L is a linear
function of w, so when ALL active metrics are linear, the problem reduces to
a Linear Program (solved via ``scipy.optimize.linprog`` in microseconds).

For mixed/non-linear metrics, SLSQP with multi-start is used.  The problem
dimension = |S| (typically 3–8), so it converges in milliseconds.

Cache key = ``tuple(sorted(stock_indices))`` so that the same stock subset
always hits the same cache entry regardless of input ordering, and cached
weights are returned in the canonical sorted order.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Set, Tuple

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import csr_matrix


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED: Canonical adaptive PnL evaluation (single source of truth)
# ═══════════════════════════════════════════════════════════════════════════════

def active_mask_with_grace(is_valid: np.ndarray, grace: int) -> np.ndarray:
    """Canonical ADAPTIVE_REWEIGHT participation mask (both engines use this).

    A name is ACTIVE on day t once it has printed its first valid observation,
    and stays active through a data gap of <= ``grace`` consecutive days — an
    in-grace day keeps the name's weight allocated and contributes its last
    valid payoff (see ``carry_pnl_within_grace``) instead of being
    redistributed immediately.

    grace <= 0 returns ``is_valid`` itself: exactly the historical behaviour
    (every gap day redistributes at once), bit-identical by construction.

    Parameters
    ----------
    is_valid : (T, C) bool — True where the P&L observation exists (not NaN)
    grace : int — max gap length (days) a name keeps its slot

    Returns
    -------
    (T, C) bool — the mask to feed ``adaptive_pnl`` / the weight solver
    """
    if grace <= 0:
        return is_valid
    T = is_valid.shape[0]
    pos = np.arange(T)[:, np.newaxis]
    last_valid = np.where(is_valid, pos, -1)
    last_valid = np.maximum.accumulate(last_valid, axis=0)
    started = last_valid >= 0
    return started & ((pos - last_valid) <= grace)


def carry_pnl_within_grace(pnl_matrix: np.ndarray, is_valid: np.ndarray,
                           grace: int) -> np.ndarray:
    """Fill in-grace gap days with the name's LAST VALID payoff.

    A rolling swap's window does not move on a day its name does not print
    (exchange holiday), so the carried mark is the exact payoff for that
    day — not an approximation. Cells beyond the grace horizon (gap > grace
    days) and cells before a name's first print stay NaN; the mask from
    ``active_mask_with_grace`` redistributes those weights instead.

    grace <= 0 returns the matrix unchanged: bit-identical historical
    behaviour (every gap day redistributes at once). Both engines apply this
    at the same point they build the participation mask, from the RAW
    validity — never re-carry an already-carried matrix.
    """
    if grace <= 0:
        return pnl_matrix
    T = pnl_matrix.shape[0]
    pos = np.arange(T)[:, np.newaxis]
    last_valid = np.maximum.accumulate(np.where(is_valid, pos, -1), axis=0)
    in_grace = (last_valid >= 0) & ((pos - last_valid) <= grace) & ~is_valid
    carried = np.take_along_axis(pnl_matrix, np.clip(last_valid, 0, None), axis=0)
    out = pnl_matrix.copy()
    out[in_grace] = carried[in_grace]
    return out


def adaptive_pnl(pnl_matrix: np.ndarray, stock_indices: np.ndarray, w: np.ndarray,
                 active_mask: np.ndarray = None) -> np.ndarray:
    """Compute adaptive-renormalized net PnL for a long-only weight vector.

    This is THE canonical implementation used by:
      - DispersionOptimizer._adaptive_net_pnl (long-only path)
      - WeightSolver.smooth_weights (floor evaluation)
      - Interactive post-smoothing (UI)

    Parameters
    ----------
    pnl_matrix : (T, C) array — nan_to_num'd (zeros where NaN)
    stock_indices : (n,) int array — column indices into pnl_matrix
    w : (n,) float array — weight vector (sums to 1)
    active_mask : (T, C) bool array — participation mask: the plain validity
                  mask, or `active_mask_with_grace(...)` when a reweight grace
                  is set (an active-but-gap day keeps its weight in the
                  denominator; feed a matrix pre-filled by
                  ``carry_pnl_within_grace`` so an in-grace day contributes
                  the name's last mark).
                  If None, plain matmul (no adaptive renorm).

    Returns
    -------
    (T,) array — daily net PnL
    """
    sub_pnl = pnl_matrix[:, stock_indices]  # (T, n) — zeros where NaN
    if active_mask is not None:
        subset_valid = active_mask[:, stock_indices].astype(np.float64)
        # Per-row: sum(pnl_i * w_i) / sum(valid_i * w_i) — renormalized
        w_active = subset_valid * w[np.newaxis, :]
        denom = w_active.sum(axis=1)
        denom[denom < 1e-12] = 1.0
        return (sub_pnl * w[np.newaxis, :]).sum(axis=1) / denom
    else:
        return sub_pnl @ w

from functions.dispersion._logging import logger as _engine_log
from .metrics import ScoreContext, soft_hit_ratio
from .score import ScoreFunction

__all__ = [
    "WeightConstraints",
    "WeightSolverResult",
    "WeightSolver",
    "VegaSpec",
    "project_to_bounded_simplex",
    "concave_blend_lambdas",
    "concave_blend_value",
]


# ---------------------------------------------------------------------------
# Solver tuning constants (single home for every solver-side magic number)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _SolverTuning:
    """Every tunable magic number of the inner weight solver.

    Defaults are STRICTLY the historical values.  Pure numerical tolerances
    (1e-9/1e-12 feasibility epsilons) stay inline by convention.  GA-side
    numbers live in _optimizer._TuningConstants (this module cannot import
    from the optimizer).
    """

    #: Tiny linear pull toward high-mean names in the SLSQP objective —
    #: breaks flat plateaus. Bigger distorts the objective. Range [0, 1e-2].
    reg_strength: float = 1e-4
    #: Quantile score above which the sweep ranks on the raw tie-break
    #: (must match the GA's tiebreak_threshold). Range [0.9, 0.999].
    tiebreak_threshold: float = 0.99
    #: Minimum full-active days for the linear-exact LP restrictions, and
    #: minimum active days for the bisection. Ranges [20, 150] / [5, 50].
    min_lp_days: int = 50
    min_bisect_days: int = 10
    #: Bisection tolerances (GA proxy vs fine/safety-net) and LP-iteration
    #: caps for each. Coarser GA tol = fewer LPs per subset.
    bisect_tol_ga: float = 0.01
    bisect_tol_fine: float = 1e-6
    bisect_iters_ga: int = 6
    bisect_iters_fine: int = 45
    #: Warm-start windows around cached / best-known f*: |f|·rel + abs.
    warm_rel_self: float = 0.2
    warm_abs_self: float = 0.01
    warm_rel_global: float = 0.3
    warm_abs_global: float = 0.05
    #: SLSQP iteration caps (solve vs 2-start diagnostic).
    slsqp_maxiter: int = 30
    diagnose_maxiter: int = 50
    #: Step-sweep size: Dirichlet draws + full simplex grid when n <= grid_max_n.
    sweep_dirichlet: int = 300
    sweep_grid_steps: int = 5
    sweep_grid_max_n: int = 4
    #: LP projections of group-violating sweep candidates (HiGHS each).
    sweep_projection_budget: int = 12
    #: V-grid resolution for the per-evaluation vega choice. Range [9, 101].
    vega_grid_points: int = 33
    #: last_carry proxy window (days) in the bisection step-2 LP objective.
    bisect_carry_window: int = 63
    #: Post-smoothing eps ladder multipliers.
    smooth_eps_ladder: Tuple[int, ...] = (1, 2, 4)


#: Module-level defaults — the solver reads THIS object.
SOLVER_TUNING = _SolverTuning()


def project_to_bounded_simplex(w: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """THE canonical projection onto {w : Σw = 1, lb <= w <= ub}.

    Iterative clipping: clip to the box, then redistribute the excess/deficit
    proportionally among the variables that can still move.  Converges in
    <= 2n iterations.  Single implementation for the whole engine (GA weight
    derivation, LP numerical cleanup, sweep candidates, smoothing starts,
    safety-net) — the historical optimizer-side duplicate is gone.
    """
    w = np.clip(np.asarray(w, dtype=np.float64), lb, ub)
    for _ in range(len(w) * 2):
        s = w.sum()
        if abs(s - 1.0) < 1e-10:
            break
        excess = s - 1.0
        if excess > 0:
            # Need to decrease — only variables above their lb can decrease
            free = w > lb + 1e-12
        else:
            # Need to increase — only variables below their ub can increase
            free = w < ub - 1e-12
        if not free.any():
            # Infeasible — just normalize (shouldn't happen with valid bounds)
            w = w / s
            break
        # Distribute proportionally among free variables
        adjustment = excess * (w[free] / w[free].sum())
        w[free] -= adjustment
        w = np.clip(w, lb, ub)
    return w


def concave_blend_lambdas(score_fn) -> Tuple[float, float, float, int]:
    """(lam_min, lam_mean, lam_carry, k_carry) of the concave-blend metrics.

    Single extraction point for every consumer of the {min_payoff,
    mean_payoff, last_carry} raw blend (concave LP objective, bisection
    step 2, refinement acceptance).
    """
    w = score_fn.weights
    lam_min = w.get("min_payoff", 0.0)
    lam_mean = w.get("mean_payoff", 0.0)
    lam_carry = w.get("last_carry", 0.0)
    k_carry = 1
    for m in score_fn.metrics:
        if m.name == "last_carry" and hasattr(m, "_k"):
            k_carry = m._k
            break
    return lam_min, lam_mean, lam_carry, k_carry


def concave_blend_value(pnl: np.ndarray, lam_min: float, lam_mean: float,
                        lam_carry: float, k_carry: int) -> float:
    """Scalar raw value of the concave blend on a P&L series (currency units)."""
    if len(pnl) == 0:
        return -np.inf
    val = 0.0
    if lam_min > 1e-12:
        val += lam_min * float(np.min(pnl))
    if lam_mean > 1e-12:
        val += lam_mean * float(np.mean(pnl))
    if lam_carry > 1e-12:
        tail = pnl[-k_carry:] if k_carry <= len(pnl) else pnl
        val += lam_carry * float(np.mean(tail))
    return val


# ---------------------------------------------------------------------------
# Configuration & result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeightConstraints:
    """Bounds and constraints for the inner weight solver.

    Attributes
    ----------
    min_weight:
        Minimum weight per stock (default 0.05 = 5%).
    max_weight:
        Maximum weight per stock (default 0.60 = 60%).
    max_net_strike:
        Upper bound on weighted-average net strike (optional).
    max_stocks:
        Maximum number of stocks allowed in a basket.  Used for config-time
        feasibility validation.  Defaults to 20.
    """

    min_weight: float = 0.05
    max_weight: float = 0.60
    max_net_strike: Optional[float] = None
    max_stocks: int = 20

    def __post_init__(self) -> None:
        """Validate constraint feasibility at construction time."""
        if self.min_weight * self.max_stocks > 1.0 + 1e-9:
            raise ValueError(
                f"Infeasible constraints: min_weight={self.min_weight}, "
                f"max_weight={self.max_weight}, max_stocks={self.max_stocks}. "
                f"Need min_weight*max_stocks <= 1 <= max_weight*max_stocks. "
                f"Lower min_weight to <= {1.0/self.max_stocks:.4f}, "
                f"or lower max_stocks to <= {int(1.0/self.min_weight)}."
            )
        if self.max_weight * self.max_stocks < 1.0 - 1e-9:
            raise ValueError(
                f"Infeasible constraints: max_weight={self.max_weight}, "
                f"max_stocks={self.max_stocks}. "
                f"Need max_weight*max_stocks >= 1. "
                f"Raise max_weight to >= {1.0/self.max_stocks:.4f}."
            )


@dataclass(frozen=True)
class VegaSpec:
    """Per-solve absolute-Vega data (aligned with the caller's stock_indices).

    Attributes
    ----------
    targets:
        Axe target per name (absolute Vega units; 0 = no axe on the name).
    caps:
        Hard Vega cap per name (v_i <= cap; ``np.inf`` = uncapped).
    v_min / v_max:
        Bounds for the FREE basket total V = Σ v_i.
    t_total:
        Σ targets over the WHOLE candidate universe (denominator of the
        ``axe_book_cleaned`` criterion — constant per run so the criterion
        is comparable across subsets).
    """

    targets: np.ndarray
    caps: np.ndarray
    v_min: float
    v_max: float
    t_total: float


@dataclass(frozen=True)
class WeightSolverResult:
    """Result of the inner weight optimisation.

    Attributes
    ----------
    weights:
        Optimal weight vector (length = n_stocks in subset), ordered to match
        the *sorted* stock_indices passed to :meth:`WeightSolver.solve`.
    score:
        Score at optimal weights (step-function normalisation).
    feasible:
        Whether the constraint set was feasible.
    n_evals:
        Number of objective evaluations.
    extra:
        Optional dictionary with solver-specific metadata (e.g., achieved_min_payoff).
    """

    weights: np.ndarray
    score: float
    feasible: bool
    n_evals: int
    extra: Optional[Dict] = None


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


class WeightSolver:
    """Solve the inner-level weight optimisation for a given stock subset.

    Parameters
    ----------
    score_fn:
        A fitted :class:`ScoreFunction`.
    ctx:
        Score context for the current run.
    constraints:
        Portfolio weight constraints.
    n_restarts:
        Number of random restarts for non-linear problems.
    seed:
        RNG seed for reproducible multi-start.
    """

    def __init__(
        self,
        score_fn: ScoreFunction,
        ctx: ScoreContext,
        constraints: WeightConstraints,
        n_restarts: int = 3,
        seed: int = 42,
        logger: Optional[Callable[[str, str], None]] = None,
        missing_data_policy: str = "fill_zero",
    ) -> None:
        self._score_fn = score_fn
        self._ctx = ctx
        self._constraints = constraints
        self._n_restarts = n_restarts
        self._seed = int(seed)
        self._rng = np.random.default_rng(seed)
        self._log = logger or (lambda l, m: None)
        self._missing_data_policy = missing_data_policy  # "adaptive_reweight", "fill_zero", "drop_incomplete"
        # Cache keyed by sorted tuple of global stock indices
        self._cache: Dict[Tuple[int, ...], WeightSolverResult] = {}
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        # Warm-start cache for bisection f_star values
        self._fstar_cache: Dict[Tuple[int, ...], float] = {}
        self._bisect_lp_count: int = 0  # total LPs across all bisection calls
        # Log config once at construction
        self._log("DEBUG", f"[WEIGHT-SOLVER] MetricWeights={dict(score_fn.weights.items())} policy={missing_data_policy}")

    def clear_cache(self) -> None:
        """Clear the memoisation cache."""
        self._cache.clear()

    #: Canonical bounded-simplex projection (module-level single source of truth)
    _project_to_bounded_simplex = staticmethod(project_to_bounded_simplex)

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    @staticmethod
    def _group_rows(n_vars: int, n: int, group_bounds) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Build LP inequality rows for per-group (bucket) weight bounds.

        ``group_bounds`` = iterable of (positions, lo, hi) with positions
        indexing the weight variables 0..n-1; hi may be None (no cap).
        Returns (A_ub, b_ub) or (None, None) when nothing to add.  Rows only
        touch the first n variables (auxiliaries like t stay at 0).
        """
        if not group_bounds:
            return None, None
        rows = []
        rhs = []
        for pos, lo, hi in group_bounds:
            pos = np.asarray(pos, dtype=int)
            if pos.size == 0:
                continue
            if hi is not None:
                row = np.zeros(n_vars, dtype=np.float64)
                row[pos] = 1.0
                rows.append(row)
                rhs.append(float(hi))
            if lo is not None and lo > 0.0:
                row = np.zeros(n_vars, dtype=np.float64)
                row[pos] = -1.0
                rows.append(row)
                rhs.append(-float(lo))
        if not rows:
            return None, None
        return np.vstack(rows), np.asarray(rhs, dtype=np.float64)

    @staticmethod
    def _groups_satisfied(w: np.ndarray, group_bounds, tol: float = 1e-6) -> bool:
        """Check per-group weight bounds on a weight vector."""
        if not group_bounds:
            return True
        for pos, lo, hi in group_bounds:
            pos = np.asarray(pos, dtype=int)
            s = float(w[pos].sum()) if pos.size else 0.0
            if lo is not None and s < float(lo) - tol:
                return False
            if hi is not None and s > float(hi) + tol:
                return False
        return True

    def project_to_group_feasible(
        self,
        target: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        group_bounds,
        strikes: Optional[np.ndarray] = None,
        max_net_strike: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """L1-project ``target`` onto {sum=1, lb<=w<=ub, group bounds[, strike]}.

        Deterministic LP (HiGHS): variables [w, d], minimise Σd with
        d >= ±(w − target).  Returns None when the constraint set is
        infeasible — callers must treat that as a hard rejection.
        """
        n = len(target)
        n_vars = 2 * n
        obj = np.concatenate([np.zeros(n), np.ones(n)])
        A_eq = np.zeros((1, n_vars))
        A_eq[0, :n] = 1.0
        b_eq = np.array([1.0])
        A_rows = []
        b_rows = []
        eye = np.eye(n)
        #  w - d <= target   and   -w - d <= -target
        A_rows.append(np.hstack([eye, -eye]))
        b_rows.append(np.asarray(target, dtype=np.float64))
        A_rows.append(np.hstack([-eye, -eye]))
        b_rows.append(-np.asarray(target, dtype=np.float64))
        g_A, g_b = self._group_rows(n_vars, n, group_bounds)
        if g_A is not None:
            A_rows.append(g_A)
            b_rows.append(g_b)
        if strikes is not None and max_net_strike is not None:
            row = np.zeros(n_vars)
            row[:n] = strikes
            A_rows.append(row.reshape(1, -1))
            b_rows.append(np.array([float(max_net_strike)]))
        bounds = [(float(lb[i]), float(ub[i])) for i in range(n)] + [(0.0, None)] * n
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = linprog(
                c=obj, A_ub=csr_matrix(np.vstack(A_rows)), b_ub=np.concatenate(b_rows),
                A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs",
            )
        if not res.success:
            return None
        return res.x[:n]

    # --- Absolute-Vega helpers (Phase 4c) ---

    @staticmethod
    def _axe_fracs(v_abs: np.ndarray, targets: np.ndarray, t_total: float) -> Tuple[float, float]:
        """(A, B) of an absolute allocation: A = Σmin(v,t)/T_universe,
        B = Σmin(v,t)/V.  NaN where the denominator is empty."""
        cleaned = float(np.minimum(v_abs, targets).sum())
        a = cleaned / t_total if t_total > 0 else float("nan")
        v_tot = float(v_abs.sum())
        b = cleaned / v_tot if v_tot > 0 else float("nan")
        return a, b

    def _vega_choose_V(self, w: np.ndarray, vega: VegaSpec,
                       lam_a: float, lam_b: float) -> Optional[Tuple[float, float, float]]:
        """Deterministic 1-D choice of the basket total V for given weights w.

        V ranges over a fixed 33-point grid on [v_min, min(v_max, cap-bound)]
        where cap-bound = min_i cap_i / w_i guarantees v_i = w_i·V <= cap_i by
        construction.  Picks the V maximising lam_a·A + lam_b·B (first/lowest
        V wins ties — deterministic).  Returns (V, A, B), or None when caps
        make every V < v_min (the subset cannot hold the minimum package).
        """
        w = np.asarray(w, dtype=np.float64)
        with np.errstate(divide="ignore"):
            ratios = np.where(w > 1e-12, vega.caps / np.maximum(w, 1e-12), np.inf)
        v_cap = float(np.min(ratios)) if ratios.size else float("inf")
        v_hi = min(float(vega.v_max), v_cap)
        if v_hi < float(vega.v_min) - 1e-9:
            return None
        grid = np.linspace(float(vega.v_min), v_hi, SOLVER_TUNING.vega_grid_points)
        # Tie preference: with B active, ties break toward the LOWEST V
        # (recycled fraction favours small packages); otherwise toward the
        # LARGEST V (max-clean spirit: A is nondecreasing in V, and a basket
        # with no axes takes the largest feasible package).
        prefer_low = lam_b > 0.0
        best = None
        for V in grid:
            a, b = self._axe_fracs(w * V, vega.targets, vega.t_total)
            val = (lam_a * (a if np.isfinite(a) else 0.0)
                   + lam_b * (b if np.isfinite(b) else 0.0))
            if best is None:
                best = (val, float(V), a, b)
            elif val > best[0] + 1e-12:
                best = (val, float(V), a, b)
            elif not prefer_low and val > best[0] - 1e-12:
                best = (val, float(V), a, b)  # tie → larger V
        return best[1], best[2], best[3]

    def _solve_axe_lp(
        self,
        sub_pnl: np.ndarray,
        strikes: Optional[np.ndarray],
        n: int,
        per_stock_bounds: Optional[np.ndarray],
        vega: VegaSpec,
        group_bounds=None,
    ) -> WeightSolverResult:
        """Exact LP for the PURE axe_book_cleaned objective (vega mode).

        Variables x = [v (n), r (n), V]:
            max Σ r_i                       (A = Σr / T_universe, T constant)
            r_i <= v_i ; r_i <= target_i    (r = recycled Vega per name)
            v_i <= cap_i                    (hard axe caps, via bounds)
            Σ v_i = V ;  v_min <= V <= v_max
            b_min_i·V <= v_i <= b_max_i·V   (concentration, from Min/Max Weight)
            Σ strike_i·v_i <= max_net_strike·V   (net strike on w = v/V)
            bucket weight bounds scaled by V

        Returns weights w = v/V; extras carry (vega_total, axe_cleaned,
        axe_recycled).
        """
        c_ = self._constraints
        nv = 2 * n + 1
        i_V = 2 * n
        obj = np.zeros(nv)
        obj[n:2 * n] = -1.0  # maximise Σ r

        caps = np.asarray(vega.caps, dtype=np.float64)
        targets = np.maximum(np.asarray(vega.targets, dtype=np.float64), 0.0)
        bounds = []
        for i in range(n):  # v_i ∈ [0, cap_i]
            bounds.append((0.0, float(caps[i]) if np.isfinite(caps[i]) else None))
        for i in range(n):  # r_i ∈ [0, target_i]
            bounds.append((0.0, float(targets[i])))
        bounds.append((float(vega.v_min), float(vega.v_max)))  # V

        if per_stock_bounds is not None:
            bmin = per_stock_bounds[:, 0]
            bmax = per_stock_bounds[:, 1]
        else:
            bmin = np.full(n, c_.min_weight)
            bmax = np.full(n, c_.max_weight)

        rows = []
        rhs = []
        eye = np.eye(n)
        # r_i - v_i <= 0
        blk = np.zeros((n, nv))
        blk[:, :n] = -eye
        blk[:, n:2 * n] = eye
        rows.append(blk)
        rhs.append(np.zeros(n))
        # v_i - b_max_i·V <= 0
        blk = np.zeros((n, nv))
        blk[:, :n] = eye
        blk[:, i_V] = -bmax
        rows.append(blk)
        rhs.append(np.zeros(n))
        # b_min_i·V - v_i <= 0
        blk = np.zeros((n, nv))
        blk[:, :n] = -eye
        blk[:, i_V] = bmin
        rows.append(blk)
        rhs.append(np.zeros(n))
        # Net strike on w: Σ s_i·v_i - max_net_strike·V <= 0
        if strikes is not None and c_.max_net_strike is not None:
            row = np.zeros((1, nv))
            row[0, :n] = strikes
            row[0, i_V] = -float(c_.max_net_strike)
            rows.append(row)
            rhs.append(np.zeros(1))
        # Bucket weight bounds on w, scaled by V
        if group_bounds:
            for pos, lo_g, hi_g in group_bounds:
                pos = np.asarray(pos, dtype=int)
                if pos.size == 0:
                    continue
                if hi_g is not None:
                    row = np.zeros((1, nv))
                    row[0, pos] = 1.0
                    row[0, i_V] = -float(hi_g)
                    rows.append(row)
                    rhs.append(np.zeros(1))
                if lo_g is not None and lo_g > 0.0:
                    row = np.zeros((1, nv))
                    row[0, pos] = -1.0
                    row[0, i_V] = float(lo_g)
                    rows.append(row)
                    rhs.append(np.zeros(1))
        A_ub = csr_matrix(np.vstack(rows))
        b_ub = np.concatenate(rhs)
        A_eq = np.zeros((1, nv))
        A_eq[0, :n] = 1.0
        A_eq[0, i_V] = -1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = linprog(c=obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[0.0],
                          bounds=bounds, method="highs")
        if not res.success:
            return WeightSolverResult(weights=np.full(n, 1.0 / n), score=-np.inf,
                                      feasible=False, n_evals=1)
        v = res.x[:n]
        V = float(res.x[i_V])
        if V <= 0:
            return WeightSolverResult(weights=np.full(n, 1.0 / n), score=-np.inf,
                                      feasible=False, n_evals=1)
        w = v / V
        a, b = self._axe_fracs(v, targets, vega.t_total)
        net = sub_pnl @ w
        ctx = dataclasses.replace(self._ctx, axe_cleaned=a, axe_recycled=b)
        score = self._score_fn.score(net, ctx)
        return WeightSolverResult(
            weights=w, score=score, feasible=True, n_evals=1,
            extra={"vega_total": V, "axe_cleaned": a, "axe_recycled": b},
        )

    def solve(
        self,
        pnl_matrix: np.ndarray,
        stock_indices: np.ndarray,
        strikes: Optional[np.ndarray] = None,
        per_stock_bounds: Optional[np.ndarray] = None,
        active_mask: Optional[np.ndarray] = None,
        tol: str = "ga",
        group_bounds=None,
        vega: Optional[VegaSpec] = None,
    ) -> WeightSolverResult:
        """Find optimal weights for the given stock subset.

        Parameters
        ----------
        pnl_matrix:
            PnL matrix, shape (n_days, n_stocks).  Can be the full matrix or
            a pre-sliced sub-matrix.  ``stock_indices`` indexes into this.
        stock_indices:
            Integer indices into pnl_matrix columns for this subset.
        strikes:
            Per-stock strike values (for max_net_strike constraint).
            Same length as stock_indices.  Optional.
        per_stock_bounds:
            Per-stock (min, max) weight bounds, shape (n, 2).
            If provided, overrides the global min_weight/max_weight from constraints.
            Each row is [min_weight_i, max_weight_i].
        active_mask:
            Boolean mask shape (n_days, n_total_cols) — True where data is valid.
            Used by LP paths to restrict day-constraints to full-active rows only
            (where adaptive renormalization is identity → linear form is exact).
            If None, all rows are used (backward compatible).
        group_bounds:
            Optional iterable of ``(positions, lo, hi)`` per-bucket weight
            bounds, with positions indexing into the CALLER's stock_indices
            order (hi=None = no cap).  Must be a deterministic function of
            the subset (bucket membership) — the cache is keyed on the
            subset only.

        Returns
        -------
        WeightSolverResult
            Weights are ordered to match the input ``stock_indices`` ordering.
        """
        n = len(stock_indices)
        c = self._constraints

        # Canonical cache key: sorted tuple of indices
        sorted_indices = tuple(sorted(int(i) for i in stock_indices))
        if sorted_indices in self._cache:
            self._cache_hits += 1
            cached = self._cache[sorted_indices]
            # Remap weights from sorted order to caller's order if needed
            if tuple(int(i) for i in stock_indices) != sorted_indices:
                idx_to_sorted_pos = {v: pos for pos, v in enumerate(sorted_indices)}
                perm = [idx_to_sorted_pos[int(i)] for i in stock_indices]
                remapped_w = cached.weights[perm]
                return WeightSolverResult(
                    weights=remapped_w,
                    score=cached.score,
                    feasible=cached.feasible,
                    n_evals=0,
                    extra=cached.extra,
                )
            return cached

        # Feasibility check (weight bounds + strike constraint + group floors)
        self._cache_misses += 1
        if not self._is_feasible(n, strikes, group_bounds=group_bounds):
            result = WeightSolverResult(
                weights=np.full(n, 1.0 / n),
                score=-np.inf,
                feasible=False,
                n_evals=0,
            )
            self._cache[sorted_indices] = result
            return result

        # Compute full-active-day mask for LP paths (rows where ALL selected legs have data)
        if active_mask is not None:
            subset_valid = active_mask[:, stock_indices]
            full_active_days = subset_valid.all(axis=1)  # True where ALL subset legs valid
        else:
            full_active_days = None  # use all rows (backward compat)

        # Extract sub-matrix (n_days × n_stocks_in_subset)
        sub_pnl = pnl_matrix[:, stock_indices]

        # Reorder to sorted order for canonical caching
        if tuple(int(i) for i in stock_indices) != sorted_indices:
            sort_perm = np.argsort(stock_indices)
            sub_pnl_sorted = sub_pnl[:, sort_perm]
            strikes_sorted = strikes[sort_perm] if strikes is not None else None
            bounds_sorted = per_stock_bounds[sort_perm] if per_stock_bounds is not None else None
        else:
            sub_pnl_sorted = sub_pnl
            strikes_sorted = strikes
            bounds_sorted = per_stock_bounds
            sort_perm = None

        # Remap group positions (caller order) into the sorted order
        groups_sorted = None
        if group_bounds:
            inv_perm = np.argsort(sort_perm) if sort_perm is not None else None
            groups_sorted = []
            for pos, lo, hi in group_bounds:
                pos = np.asarray(pos, dtype=int)
                groups_sorted.append((
                    np.sort(inv_perm[pos]) if inv_perm is not None else pos,
                    lo, hi))

        # Determine bisection tolerance from caller hint
        bisect_tol = SOLVER_TUNING.bisect_tol_ga if tol == "ga" else SOLVER_TUNING.bisect_tol_fine

        # Compute active_mask_sub for bisection paths (subset columns from active_mask)
        active_mask_sub = None
        if active_mask is not None:
            if tuple(int(i) for i in stock_indices) != sorted_indices:
                active_mask_sub = active_mask[:, stock_indices][:, sort_perm]
            else:
                active_mask_sub = active_mask[:, stock_indices]

        # Remap vega arrays (caller order) into the sorted order
        vega_sorted = vega
        if vega is not None and sort_perm is not None:
            vega_sorted = dataclasses.replace(
                vega,
                targets=np.asarray(vega.targets, dtype=np.float64)[sort_perm],
                caps=np.asarray(vega.caps, dtype=np.float64)[sort_perm],
            )

        # Route by policy and metric blend
        use_bisection = (self._missing_data_policy == "adaptive_reweight" and active_mask_sub is not None)
        _active_set = set(self._score_fn.weights.active_names)
        _axe_active = bool(_active_set & {"axe_book_cleaned", "axe_package_recycled"})

        if vega_sorted is not None and _active_set == {"axe_book_cleaned"}:
            # Pure criterion A: exact LP in absolute-Vega space
            result = self._solve_axe_lp(sub_pnl_sorted, strikes_sorted, n, bounds_sorted,
                                        vega_sorted, group_bounds=groups_sorted)
            self.log_solver_path("AXE_LP")
        elif vega_sorted is not None and _axe_active:
            # Axe criteria blended with P&L metrics: SLSQP in w-space with a
            # deterministic per-evaluation 1-D choice of V (see _vega_choose_V)
            result = self._solve_nonlinear(sub_pnl_sorted, strikes_sorted, n, bounds_sorted,
                                           sweep_key=sorted_indices, group_bounds=groups_sorted,
                                           vega=vega_sorted)
            self.log_solver_path("SLSQP_VEGA")
        elif self._all_linear():
            result = self._solve_linear(sub_pnl_sorted, strikes_sorted, n, bounds_sorted,
                                        group_bounds=groups_sorted)
            self.log_solver_path("LP")
        elif self._is_min_payoff_only():
            if use_bisection:
                result = self._solve_min_payoff_bisection(
                    sub_pnl_sorted, active_mask_sub, strikes_sorted, n, bounds_sorted, bisect_tol, sorted_indices,
                    group_bounds=groups_sorted
                )
                self.log_solver_path("min_payoff_BISECT")
            else:
                result = self._solve_min_payoff_lp(sub_pnl_sorted, strikes_sorted, n, bounds_sorted, full_active_days,
                                                   group_bounds=groups_sorted)
                self.log_solver_path("min_payoff_LP")
        elif self._is_concave_blend():
            if use_bisection:
                result = self._solve_concave_blend_bisection(
                    sub_pnl_sorted, active_mask_sub, strikes_sorted, n, bounds_sorted, bisect_tol, sorted_indices,
                    group_bounds=groups_sorted
                )
                self.log_solver_path("concave_blend_BISECT")
            else:
                result = self._solve_concave_blend(sub_pnl_sorted, strikes_sorted, n, bounds_sorted, full_active_days,
                                                   group_bounds=groups_sorted)
                self.log_solver_path("concave_blend_LP")
        else:
            result = self._solve_nonlinear(sub_pnl_sorted, strikes_sorted, n, bounds_sorted,
                                           sweep_key=sorted_indices, group_bounds=groups_sorted)
            self.log_solver_path("SLSQP")

        # P&L-only configs with vega ON: the weights come from the standard
        # paths (V does not enter the P&L series); attach the deterministic V
        # via the max-clean rule (largest feasible V — cleans the most book
        # at zero cost to the active metrics).
        if (vega_sorted is not None and not _axe_active
                and _active_set != {"axe_book_cleaned"} and result.feasible):
            picked = self._vega_choose_V(result.weights, vega_sorted, 1.0, 0.0)
            if picked is None:
                result = WeightSolverResult(weights=np.full(n, 1.0 / n), score=-np.inf,
                                            feasible=False, n_evals=result.n_evals)
            else:
                _V, _A, _B = picked
                _extra = dict(result.extra or {})
                _extra.update({"vega_total": _V, "axe_cleaned": _A, "axe_recycled": _B})
                result = dataclasses.replace(result, extra=_extra)

        # Cache in sorted order
        self._cache[sorted_indices] = result

        # Return in caller's order
        if sort_perm is not None:
            inv_perm = np.argsort(sort_perm)
            return WeightSolverResult(
                weights=result.weights[inv_perm],
                score=result.score,
                feasible=result.feasible,
                n_evals=result.n_evals,
                extra=result.extra,
            )
        return result

    def diagnose_convergence(
        self,
        pnl_matrix: np.ndarray,
        stock_indices: np.ndarray,
        strikes: Optional[np.ndarray] = None,
        tol: float = 0.01,
    ) -> Dict:
        """Diagnostic: compare solutions from equal-weight vs greedy start.

        Returns a dict with:
        - 'converged': bool — whether scores from both starts agree within tol
        - 'score_equal_start': float
        - 'score_greedy_start': float
        - 'weights_equal_start': ndarray
        - 'weights_greedy_start': ndarray
        - 'score_gap': float

        If converged=False, the solver is trapped on a plateau and needs
        stronger regularisation or more iterations.
        """
        n = len(stock_indices)
        c = self._constraints
        sub_pnl = pnl_matrix[:, stock_indices]

        col_means = sub_pnl.mean(axis=0)
        ws_active = "weighted_strike" in self._score_fn.weights.active_names

        def _ctx_for(w: np.ndarray) -> ScoreContext:
            if ws_active and strikes is not None:
                return dataclasses.replace(
                    self._ctx, weighted_strike=float(np.dot(w, strikes))
                )
            return self._ctx

        # Same differentiable surrogate objective as _solve_nonlinear — the
        # diagnostic must probe the solver's ACTUAL optimisation landscape,
        # not the step-wise raw hit_ratio (two starts on a staircase land on
        # different plateaus by construction).
        objective = self._smooth_surrogate_objective(sub_pnl, strikes, col_means)

        bounds = [(c.min_weight, c.max_weight)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        if strikes is not None and c.max_net_strike is not None:
            constraints.append(
                {"type": "ineq", "fun": lambda w, s=strikes: c.max_net_strike - np.dot(w, s)}
            )

        # Start 1: equal weight
        w0_equal = np.full(n, 1.0 / n)
        res_equal = minimize(objective, w0_equal, method="SLSQP",
                             bounds=bounds, constraints=constraints,
                             options={"maxiter": SOLVER_TUNING.diagnose_maxiter, "ftol": 1e-7})
        w_equal = np.clip(res_equal.x, c.min_weight, c.max_weight)
        w_equal = w_equal / w_equal.sum()
        score_equal = self._score_fn.score(sub_pnl @ w_equal, _ctx_for(w_equal))

        # Start 2: greedy
        greedy_w = np.full(n, c.min_weight)
        best_stock = np.argmax(col_means)
        greedy_w[best_stock] = c.max_weight
        remaining = 1.0 - greedy_w.sum()
        if remaining > 0:
            for i in np.argsort(-col_means):
                if i == best_stock:
                    continue
                a = min(remaining, c.max_weight - greedy_w[i])
                greedy_w[i] += a
                remaining -= a
                if remaining < 1e-12:
                    break
        greedy_w = np.clip(greedy_w, c.min_weight, c.max_weight)
        greedy_w = greedy_w / greedy_w.sum()

        res_greedy = minimize(objective, greedy_w, method="SLSQP",
                              bounds=bounds, constraints=constraints,
                              options={"maxiter": SOLVER_TUNING.diagnose_maxiter, "ftol": 1e-7})
        w_greedy = np.clip(res_greedy.x, c.min_weight, c.max_weight)
        w_greedy = w_greedy / w_greedy.sum()
        score_greedy = self._score_fn.score(sub_pnl @ w_greedy, _ctx_for(w_greedy))

        gap = abs(score_equal - score_greedy)
        return {
            "converged": gap < tol,
            "score_equal_start": score_equal,
            "score_greedy_start": score_greedy,
            "weights_equal_start": w_equal,
            "weights_greedy_start": w_greedy,
            "score_gap": gap,
        }

    # --- Feasibility ---

    def _is_feasible(self, n: int, strikes: Optional[np.ndarray] = None,
                     group_bounds=None) -> bool:
        """Check if constraint set is feasible for n stocks.

        Validates:
        - min_weight * n <= 1 (sum=1 achievable with lower bounds)
        - max_weight * n >= 1 (sum=1 achievable with upper bounds)
        - max_net_strike achievable (if strikes provided)
        - per-bucket weight floors sum to <= 1 (quick necessary condition;
          the LP itself is the full arbiter)
        """
        c = self._constraints
        if c.min_weight * n > 1.0 + 1e-9:
            return False
        if c.max_weight * n < 1.0 - 1e-9:
            return False
        # Strike feasibility: if ALL stocks have strike > max_net_strike,
        # then weighted avg > max_net_strike for ANY w with sum=1
        if strikes is not None and c.max_net_strike is not None and len(strikes) > 0:
            if np.all(strikes > c.max_net_strike + 1e-9):
                return False
        if group_bounds:
            total_floors = sum(float(lo) for _, lo, _ in group_bounds if lo)
            if total_floors > 1.0 + 1e-9:
                return False
        return True

    # --- Linear fast path (true LP via scipy.optimize.linprog) ---

    def _all_linear(self) -> bool:
        """Check if all active (weighted) metrics are linear in weights."""
        active = self._score_fn.weights.active_names
        for m in self._score_fn.metrics:
            if m.name in active and not m.is_linear:
                return False
        return True

    def _is_min_payoff_only(self) -> bool:
        """Check if min_payoff is the ONLY active metric with weight 1.0."""
        w = self._score_fn.weights
        # Exactly one active metric
        if len(w.active_names) != 1:
            return False
        # That metric must be min_payoff
        if "min_payoff" not in w.active_names:
            return False
        # Its weight must be exactly 1.0
        if abs(w["min_payoff"] - 1.0) > 1e-9:
            return False
        # All other metrics must have weight 0.0
        for name in w.keys():
            if name != "min_payoff" and abs(w[name]) > 1e-9:
                return False
        return True

    def _is_concave_blend(self) -> bool:
        """True iff active metrics are a subset of {min_payoff, mean_payoff, last_carry} with >= 2 active."""
        active: Set[str] = set(self._score_fn.weights.active_names)
        if len(active) < 2:
            return False
        return active.issubset({"min_payoff", "mean_payoff", "last_carry"})

    def has_exact_path(self) -> bool:
        """True if the current MetricWeights config has an exact solver."""
        return self._all_linear() or self._is_min_payoff_only() or self._is_concave_blend()

    def smooth_weights(
        self,
        w_star: np.ndarray,
        pnl_matrix: np.ndarray,
        stock_indices: np.ndarray,
        active_mask: np.ndarray = None,
        eps_min: float = 0.05,
        per_stock_bounds: np.ndarray = None,
        strikes: np.ndarray = None,
        group_bounds=None,
    ) -> np.ndarray:
        """Post-optimization smoothing via QP from equal-weight start.

        Minimizes ||w - 1/n||^2 subject to:
          C1 (equality):   sum(w) = 1
          C2 (ineq, per-day): min_t(adaptive_pnl_t(w)) >= min(adaptive_pnl(w*)) - cur_eps
          C3 (ineq, blend):   raw_blend(w) >= raw_blend(w*) - cur_eps
              where raw_blend = Σ_active lam_i · sign_i · raw_metric_i, built
              from the ACTIVE metric set (core + optional metrics computed via
              their Metric objects; weighted_strike enters as −lam·dot(w,
              strikes) and REQUIRES the strikes argument when active).
          C4 (ineq, strike): max_net_strike - dot(w, strikes) >= 0
          Bounds: lb_i <= w_i <= ub_i

        Starts from equal-weight projected to feasibility (NOT from w*).
        Falls back with eps ladder: eps, 2*eps, 4*eps.

        Returns smoothed weights, or w_star unchanged if all attempts fail.
        """
        n = len(w_star)
        c = self._constraints
        target = np.full(n, 1.0 / n)
        sub_pnl = pnl_matrix[:, stock_indices]

        # ── Adaptive net PnL helper — delegates to canonical shared function ──
        def _adaptive_pnl(w):
            return adaptive_pnl(pnl_matrix, stock_indices, w, active_mask)

        # ── Raw blend from the ACTIVE metric set (metric-faithful) ──
        # Each active metric contributes lam · sign · raw_value, where sign
        # follows higher_is_better.  ``weighted_strike`` is the only metric
        # not computable from the P&L series: it needs the per-stock strike
        # vector and enters as −lam · dot(w, strikes).
        # NOTE unit caveat: the eps ladder is expressed in blend units, which
        # mix metric scales (P&L units for min/mean/carry/dd/cvar, [0,1] for
        # hit_ratio, annualised ratio for sharpe, decimals for strikes) — the
        # historical behaviour, kept as-is.
        weights_dict = dict(self._score_fn.weights.items())
        active_names = [k for k, v in weights_dict.items() if v > 0]
        metric_map = {m.name: m for m in self._score_fn.metrics}
        _unknown = [k for k in active_names
                    if k != "weighted_strike" and k not in metric_map]
        if _unknown:
            raise ValueError(
                f"smooth_weights: active metric(s) {_unknown} are not present "
                f"in the score function — cannot build a faithful raw blend."
            )
        if "weighted_strike" in active_names and strikes is None:
            raise ValueError(
                "smooth_weights: 'weighted_strike' carries a positive weight "
                "but no per-stock strikes were provided. Pass strikes= (vector "
                "aligned with stock_indices) — smoothing without the strike "
                "term would silently trade the strike objective away."
            )

        def _raw_blend(pnl, w):
            """Scalarised raw objective (sign-adjusted), no normalization."""
            total = 0.0
            for name in active_names:
                lam = weights_dict[name]
                if name == "weighted_strike":
                    total += -lam * float(np.dot(w, strikes))
                    continue
                m = metric_map[name]
                v = m.compute(pnl, self._ctx)
                if not np.isfinite(v):
                    raise ValueError(
                        f"smooth_weights: metric '{name}' returned a non-finite "
                        f"raw value on the reference basket — this configuration "
                        f"cannot be smoothed safely (series too short or "
                        f"degenerate)."
                    )
                total += lam * (1.0 if m.higher_is_better else -1.0) * float(v)
            return total

        # ── Reference values from w_star ──
        ref_pnl = _adaptive_pnl(w_star)
        ref_min = float(np.min(ref_pnl))
        ref_blend = _raw_blend(ref_pnl, w_star)
        ref_disp = float(np.std(w_star))

        # ── Bounds ──
        if per_stock_bounds is not None:
            lb = per_stock_bounds[:, 0].copy()
            ub = per_stock_bounds[:, 1].copy()
        else:
            lb = np.full(n, c.min_weight)
            ub = np.full(n, c.max_weight)
        bounds = list(zip(lb.tolist(), ub.tolist()))

        # ── Starting point: equal-weight projected to bounded simplex ──
        w0 = self._project_to_bounded_simplex(target.copy(), lb, ub)

        # ── QP objective: minimize ||w - 1/n||^2 ──
        def objective(w):
            d = w - target
            return float(np.dot(d, d))

        def jac_obj(w):
            return 2.0 * (w - target)

        # ── Fallback ladder: try eps, 2*eps, 4*eps ──
        for mult in SOLVER_TUNING.smooth_eps_ladder:
            cur_eps = eps_min * mult
            floor_min = ref_min - cur_eps
            floor_blend = ref_blend - cur_eps

            # C1: sum(w) == 1
            # C2: min_t(adaptive_pnl_t(w)) >= floor_min
            # C3: raw_blend(adaptive_pnl(w)) >= floor_blend  [raw units, no score_smooth]
            # C4: max_net_strike - dot(w, strikes) >= 0
            cons = [
                {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0),
                 "jac": lambda w: np.ones(n)},
                {"type": "ineq", "fun": lambda w, _fl=floor_min: float(np.min(_adaptive_pnl(w))) - _fl},
                {"type": "ineq", "fun": lambda w, _fb=floor_blend: _raw_blend(_adaptive_pnl(w), w) - _fb},
            ]
            if strikes is not None and c.max_net_strike is not None:
                cons.append({"type": "ineq", "fun": lambda w: float(c.max_net_strike - np.dot(w, strikes))})
            # C5: per-bucket weight bounds (smoothing must not leave the box)
            if group_bounds:
                for pos, lo_g, hi_g in group_bounds:
                    pos = np.asarray(pos, dtype=int)
                    if pos.size == 0:
                        continue
                    if hi_g is not None:
                        cons.append({"type": "ineq",
                                     "fun": lambda w, _p=pos, _hi=float(hi_g): _hi - float(np.sum(w[_p]))})
                    if lo_g is not None and lo_g > 0.0:
                        cons.append({"type": "ineq",
                                     "fun": lambda w, _p=pos, _lo=float(lo_g): float(np.sum(w[_p])) - _lo})

            try:
                res = minimize(
                    objective, w0, method="SLSQP", jac=jac_obj,
                    bounds=bounds, constraints=cons,
                    options={"maxiter": 1000, "ftol": 1e-8},
                )
                # Accept iff: dispersion improved AND floors held — regardless of res.success
                # (maxiter-reached with a feasible improving point is valid)
                w_cand = np.clip(res.x, lb, ub)
                w_cand = w_cand / w_cand.sum()
                # Cleanup: project to bounded simplex for numerical precision
                w_cand = self._project_to_bounded_simplex(w_cand, lb, ub)
                cand_pnl = _adaptive_pnl(w_cand)
                actual_min = float(np.min(cand_pnl))
                actual_blend = _raw_blend(cand_pnl, w_cand)
                actual_disp = float(np.std(w_cand))

                _engine_log.debug(f"[SMOOTH-TRY] eps={cur_eps:.2f} success={res.success} dispersion {ref_disp:.4f}->{actual_disp:.4f} min {ref_min:.4f}->{actual_min:.4f} blend {ref_blend:.4f}->{actual_blend:.4f}")

                if not self._groups_satisfied(w_cand, group_bounds):
                    continue
                if actual_disp < ref_disp - 1e-6 and actual_min >= floor_min - 1e-6 and actual_blend >= floor_blend - 1e-6:
                    _engine_log.debug(f"[SMOOTH] accepted: dispersion {ref_disp:.4f}->{actual_disp:.4f} min {ref_min:.4f}->{actual_min:.4f} blend {ref_blend:.4f}->{actual_blend:.4f}")
                    return w_cand
                else:
                    continue
            except Exception as e:
                _engine_log.debug(f"[SMOOTH-TRY] eps={cur_eps:.2f} success=False(exception: {e}) dispersion {ref_disp:.4f}->N/A min {ref_min:.4f}->N/A blend {ref_blend:.4f}->N/A")
                continue

        # All attempts failed
        _engine_log.debug(f"[SMOOTH] infeasible even at eps={eps_min * 4:.2f} — weights unchanged")
        return w_star

    def _solve_concave_blend(
        self,
        sub_pnl: np.ndarray,
        strikes: Optional[np.ndarray],
        n: int,
        per_stock_bounds: Optional[np.ndarray] = None,
        full_active_days: Optional[np.ndarray] = None,
        group_bounds=None,
    ) -> WeightSolverResult:
        """Solve concave blend LP for {min_payoff, mean_payoff, last_carry} combinations.

        All three metrics share currency units so the raw blend is dimensionally
        coherent.  The objective is a weighted sum that is linear in [w, t].
        """
        c = self._constraints
        lam_min, lam_mean, lam_carry, K = concave_blend_lambdas(self._score_fn)

        # Restrict to full-active days (where adaptive renorm is identity → linear exact)
        if full_active_days is not None:
            n_total = sub_pnl.shape[0]
            sub_pnl = sub_pnl[full_active_days]
            n_full = sub_pnl.shape[0]
            self._log("DEBUG", f"[WEIGHT-SOLVER] concave_blend LP days: full={n_full}/{n_total}")
            if n_full < SOLVER_TUNING.min_lp_days:
                self._log("WARNING", f"[WEIGHT-SOLVER] concave_blend LP: insufficient full-active days ({n_full}<{SOLVER_TUNING.min_lp_days})")
                return WeightSolverResult(weights=np.full(n, 1.0 / n), score=-np.inf, feasible=False, n_evals=0)
        else:
            # Fallback: drop all-zero rows
            row_mask = np.any(sub_pnl != 0.0, axis=1)
            sub_pnl = sub_pnl[row_mask]

        D = sub_pnl.shape[0]

        if D < 2:
            raise RuntimeError(
                f"Concave blend LP: insufficient non-zero rows ({D}) after filtering."
            )

        use_t = lam_min > 1e-12  # include auxiliary variable t only if min_payoff active
        n_vars = n + 1 if use_t else n

        # --- Objective: maximize lam_min*t + lam_mean*(1/D)*sum(P@w) + lam_carry*(1/K)*sum_tail(P@w) ---
        # linprog minimises c^T x, so negate for maximisation
        obj = np.zeros(n_vars, dtype=np.float64)

        # mean_payoff contribution: (1/D) * sum_d P[d] = (1/D) * (col_sums)
        if lam_mean > 1e-12:
            col_sums = sub_pnl.sum(axis=0)  # sum over days for each stock
            obj[:n] -= lam_mean * (1.0 / D) * col_sums  # negate for min

        # last_carry contribution: (1/K) * sum over last K rows
        if lam_carry > 1e-12:
            tail_rows = sub_pnl[-K:] if K <= D else sub_pnl
            tail_sums = tail_rows.sum(axis=0)
            actual_K = min(K, D)
            obj[:n] -= lam_carry * (1.0 / actual_K) * tail_sums

        # min_payoff contribution: lam_min * t
        if use_t:
            obj[-1] = -lam_min  # minimize -lam_min*t == maximize lam_min*t

        # --- Bounds ---
        if per_stock_bounds is not None:
            w_bounds = [(float(per_stock_bounds[i, 0]), float(per_stock_bounds[i, 1])) for i in range(n)]
        else:
            w_bounds = [(c.min_weight, c.max_weight)] * n

        if use_t:
            bounds = w_bounds + [(None, None)]  # t unbounded
        else:
            bounds = w_bounds

        # --- Equality constraint: sum(w) = 1 ---
        A_eq = np.zeros((1, n_vars), dtype=np.float64)
        A_eq[0, :n] = 1.0
        b_eq = np.array([1.0])

        # --- Inequality constraints ---
        A_ub_parts = []
        b_ub_parts = []

        # P[d] @ w >= t  <==>  -P[d] @ w + t <= 0  (only if min_payoff active)
        if use_t:
            A_ub_pnl = np.zeros((D, n_vars), dtype=np.float64)
            A_ub_pnl[:, :n] = -sub_pnl
            A_ub_pnl[:, -1] = 1.0
            A_ub_parts.append(A_ub_pnl)
            b_ub_parts.append(np.zeros(D, dtype=np.float64))

        # Strike constraint: strikes @ w <= max_net_strike
        if strikes is not None and c.max_net_strike is not None:
            A_ub_strike = np.zeros((1, n_vars), dtype=np.float64)
            A_ub_strike[0, :n] = strikes
            A_ub_parts.append(A_ub_strike)
            b_ub_parts.append(np.array([c.max_net_strike]))

        # Per-bucket weight bounds
        g_A, g_b = self._group_rows(n_vars, n, group_bounds)
        if g_A is not None:
            A_ub_parts.append(g_A)
            b_ub_parts.append(g_b)

        if A_ub_parts:
            A_ub = np.vstack(A_ub_parts)
            b_ub = np.concatenate(b_ub_parts)
            A_ub_sparse = csr_matrix(A_ub)
        else:
            A_ub_sparse = None
            b_ub = None

        res = linprog(
            c=obj,
            A_ub=A_ub_sparse,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )

        if not res.success:
            # Infeasible/unbounded — log (rate-limited) and return infeasible result
            if not hasattr(self, '_concave_infeasible_count'):
                self._concave_infeasible_count = 0
            self._concave_infeasible_count += 1
            if self._concave_infeasible_count <= 3 or self._concave_infeasible_count % 500 == 0:
                self._log("WARNING",
                    f"[WEIGHT-SOLVER] Concave blend LP infeasible (#{self._concave_infeasible_count}): n={n}, D={D}, K={K}, "
                    f"lam_min={lam_min:.4f}, lam_mean={lam_mean:.4f}, lam_carry={lam_carry:.4f}, "
                    f"min_weight={c.min_weight}, max_weight={c.max_weight}, "
                    f"max_net_strike={c.max_net_strike}, "
                    f"HiGHS status={res.status}, message={res.message}"
                )
            return WeightSolverResult(weights=np.full(n, 1.0 / n), score=-np.inf, feasible=False, n_evals=0)

        w_optimal = res.x[:n]

        # Numerical cleanup — project onto simplex (skipped with bucket
        # bounds: the iterative projection is group-blind)
        if not group_bounds:
            if per_stock_bounds is not None:
                w_optimal = self._project_to_bounded_simplex(w_optimal, per_stock_bounds[:, 0], per_stock_bounds[:, 1])
            else:
                w_optimal = self._project_to_bounded_simplex(w_optimal, np.full(n, c.min_weight), np.full(n, c.max_weight))

        # Compute final net_pnl and score
        net_pnl = sub_pnl @ w_optimal
        score = self._score_fn.score(net_pnl, self._ctx)

        # Sanity check: if this is effectively min_payoff-only, verify consistency
        if lam_min > 0.999 and lam_mean < 1e-9 and lam_carry < 1e-9:
            achieved_min = float(np.min(net_pnl))
            try:
                ref_result = self._solve_min_payoff_lp(sub_pnl, strikes, n, per_stock_bounds)
                ref_min = float(np.min(sub_pnl @ ref_result.weights))
                if abs(achieved_min - ref_min) > 1e-8:
                    warnings.warn(
                        f"[WEIGHT-SOLVER] concave_blend sanity check failed: "
                        f"achieved_min={achieved_min:.10f} vs min_payoff_LP={ref_min:.10f}"
                    )
            except Exception:
                pass  # Don't fail on sanity check

        return WeightSolverResult(
            weights=w_optimal,
            score=score,
            feasible=True,
            n_evals=0,
            extra={"lam_min": lam_min, "lam_mean": lam_mean, "lam_carry": lam_carry},
        )
    
    def log_solver_path(self, path: str) -> None:
        """Log solver path decision."""
        self._log("DEBUG", f"[WEIGHT-SOLVER] path={path}")

    def _solve_linear(
        self,
        sub_pnl: np.ndarray,
        strikes: Optional[np.ndarray],
        n: int,
        per_stock_bounds: Optional[np.ndarray] = None,
        group_bounds=None,
    ) -> WeightSolverResult:
        """Solve via scipy.optimize.linprog for metrics linear in w."""
        c = self._constraints

        # Build linear objective: c^T w (to maximise → minimise -c^T w)
        value_per_stock = np.zeros(n, dtype=np.float64)
        active_weights = self._score_fn.weights

        for m in self._score_fn.metrics:
            if m.name not in active_weights.active_names:
                continue
            if not m.is_linear:
                continue
            per_stock_vals = np.array(
                [m.compute(sub_pnl[:, i], self._ctx) for i in range(n)],
                dtype=np.float64,
            )
            value_per_stock += active_weights[m.name] * per_stock_vals

        # linprog minimises c^T x, so negate for maximisation
        obj = -value_per_stock

        # Bounds: use per-stock if provided, otherwise global
        if per_stock_bounds is not None:
            bounds = [(float(per_stock_bounds[i, 0]), float(per_stock_bounds[i, 1])) for i in range(n)]
        else:
            bounds = [(c.min_weight, c.max_weight)] * n

        # Equality constraint: sum(w) = 1
        A_eq = np.ones((1, n))
        b_eq = np.array([1.0])

        # Inequality constraints: strike cap + per-bucket weight bounds
        A_parts = []
        b_parts = []
        if strikes is not None and c.max_net_strike is not None:
            A_parts.append(strikes.reshape(1, -1))
            b_parts.append(np.array([c.max_net_strike]))
        g_A, g_b = self._group_rows(n, n, group_bounds)
        if g_A is not None:
            A_parts.append(g_A)
            b_parts.append(g_b)
        A_ub = np.vstack(A_parts) if A_parts else None
        b_ub = np.concatenate(b_parts) if A_parts else None

        res = linprog(
            c=obj,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )

        if not res.success:
            if group_bounds:
                # With bucket bounds an equal-weight fallback may violate the
                # groups — infeasible must surface, not degrade silently.
                return WeightSolverResult(weights=np.full(n, 1.0 / n), score=-np.inf,
                                          feasible=False, n_evals=n)
            w = np.full(n, 1.0 / n)
        else:
            w = res.x

        # Numerical cleanup — project onto simplex respecting per-stock bounds
        # (skip when bucket bounds are active: the iterative projection is
        # group-blind and could push the LP optimum out of the bucket box)
        if not group_bounds:
            if per_stock_bounds is not None:
                w = self._project_to_bounded_simplex(w, per_stock_bounds[:, 0], per_stock_bounds[:, 1])
            else:
                w = self._project_to_bounded_simplex(w, np.full(n, c.min_weight), np.full(n, c.max_weight))

        # Final score with STEP-function normalisation
        net_pnl = sub_pnl @ w
        score = self._score_fn.score(net_pnl, self._ctx)

        return WeightSolverResult(weights=w, score=score, feasible=True, n_evals=n)

    # --- Min-payoff bisection path (ADAPTIVE_REWEIGHT exact) ---

    def _solve_min_payoff_bisection(
        self,
        sub_pnl: np.ndarray,
        active_mask_sub: np.ndarray,
        strikes: Optional[np.ndarray],
        n: int,
        per_stock_bounds: Optional[np.ndarray] = None,
        tol: float = 0.01,
        cache_key: Optional[Tuple[int, ...]] = None,
        group_bounds=None,
    ) -> WeightSolverResult:
        """Exact maximin on the adaptive-reweighted curve via bisection.

        For fixed floor f, the constraint adaptive_pnl_d >= f rewrites as:
            sum_i active[d,i] * w_i * (pnl[d,i] - f) >= 0   (linear in w)
        Bisect on f to find the maximum feasible floor.
        """
        c = self._constraints
        n_days, n_stocks = sub_pnl.shape

        # Drop days where no leg is active (no constraint needed)
        day_has_any = active_mask_sub.any(axis=1)
        pnl_active = sub_pnl[day_has_any]
        mask_active = active_mask_sub[day_has_any]
        n_active_days = pnl_active.shape[0]

        if n_active_days < SOLVER_TUNING.min_bisect_days:
            return WeightSolverResult(weights=np.full(n, 1.0 / n), score=-np.inf, feasible=False, n_evals=0)

        # Compute initial bounds for bisection
        # lo: adaptive min at equal weights (valid lower bound)
        eq_w = np.full(n, 1.0 / n)
        adaptive_eq = self._eval_adaptive_min(pnl_active, mask_active, eq_w)
        lo = adaptive_eq

        # hi: max over all days of max per-stock pnl (upper bound on achievable floor)
        hi = float(np.nanmax(pnl_active[mask_active])) if mask_active.any() else 0.0

        # Warm-start from cache (parent's f_star or global best)
        if cache_key is not None and cache_key in self._fstar_cache:
            cached_f = self._fstar_cache[cache_key]
            margin = abs(cached_f) * SOLVER_TUNING.warm_rel_self + SOLVER_TUNING.warm_abs_self
            warm_lo = cached_f - margin
            warm_hi = cached_f + margin
            lo = max(lo, warm_lo)
            hi = min(hi, warm_hi)
            if lo > hi:
                lo = adaptive_eq
                hi = float(np.nanmax(pnl_active[mask_active])) if mask_active.any() else 0.0
        elif self._fstar_cache:
            # Seed from best known f_star across all subsets
            best_known = max(self._fstar_cache.values())
            margin = abs(best_known) * SOLVER_TUNING.warm_rel_global + SOLVER_TUNING.warm_abs_global
            lo = max(lo, best_known - margin)
            hi = min(hi, best_known + margin)
            if lo > hi:
                lo = adaptive_eq
                hi = float(np.nanmax(pnl_active[mask_active])) if mask_active.any() else 0.0

        # Bounds for w_i
        if per_stock_bounds is not None:
            w_bounds = [(float(per_stock_bounds[i, 0]), float(per_stock_bounds[i, 1])) for i in range(n)]
        else:
            w_bounds = [(c.min_weight, c.max_weight)] * n

        # Pre-build equality constraint (sum=1) and strike constraint
        A_eq = np.zeros((1, n), dtype=np.float64)
        A_eq[0, :] = 1.0
        b_eq = np.array([1.0])

        A_strike = None
        b_strike = None
        if strikes is not None and c.max_net_strike is not None:
            A_strike = np.zeros((1, n), dtype=np.float64)
            A_strike[0, :] = strikes
            b_strike = np.array([c.max_net_strike])

        # Precompute sparse structure pieces that don't change with f
        # A_ub(f) = -(mask_float * (pnl - f)) = -mask_pnl + f * mask_float
        # Store as sparse once; each iteration is a scalar-multiply-and-subtract
        mask_float = mask_active.astype(np.float64)
        mask_pnl = mask_float * pnl_active  # [n_active_days x n]

        # Build sparse versions ONCE (the expensive part)
        M1_sparse = csr_matrix(mask_pnl)    # mask * pnl (fixed)
        M2_sparse = csr_matrix(mask_float)   # mask (fixed)

        # Pre-build fixed extra rows (strike cap + bucket bounds), appended once
        _extra_A_rows = []
        _extra_b_rows = []
        if A_strike is not None:
            _extra_A_rows.append(A_strike)
            _extra_b_rows.append(b_strike)
        g_A, g_b = self._group_rows(n, n, group_bounds)
        if g_A is not None:
            _extra_A_rows.append(g_A)
            _extra_b_rows.append(g_b)
        n_extra = 0
        A_strike_sparse = None
        b_strike_val = None
        if _extra_A_rows:
            _extra_A = np.vstack(_extra_A_rows)
            n_extra = _extra_A.shape[0]
            A_strike_sparse = csr_matrix(_extra_A)
            b_strike_val = np.concatenate(_extra_b_rows)

        # Bisection
        best_w = eq_w.copy()
        n_lps = 0
        max_iters = (SOLVER_TUNING.bisect_iters_ga if tol >= SOLVER_TUNING.bisect_tol_ga / 2
                     else SOLVER_TUNING.bisect_iters_fine)

        # DIAGNOSTIC: if fine tolerance, sanity-check that incumbent weights are feasible
        if tol < SOLVER_TUNING.bisect_tol_ga / 2 and per_stock_bounds is not None:
            # Test feasibility at the incumbent's adaptive min (should always pass)
            inc_f = adaptive_eq - 1e-6
            A_test = -M1_sparse + inc_f * M2_sparse
            if A_strike_sparse is not None:
                from scipy.sparse import vstack as sparse_vstack
                A_test = sparse_vstack([A_test, A_strike_sparse], format='csr')
                b_test = np.empty(n_active_days + n_extra, dtype=np.float64)
                b_test[:n_active_days] = 0.0
                b_test[n_active_days:] = b_strike_val
            else:
                b_test = np.zeros(n_active_days, dtype=np.float64)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sanity_res = linprog(
                    c=np.zeros(n, dtype=np.float64),
                    A_ub=A_test, b_ub=b_test,
                    A_eq=A_eq, b_eq=b_eq, bounds=w_bounds, method="highs",
                )
            sanity_ok = sanity_res.success
            if not sanity_ok:
                # Find first violated constraint at equal weights
                viol_vals = (A_test @ eq_w) - b_test[:A_test.shape[0]]
                violated_days = np.where(viol_vals > 1e-9)[0]
                first_viol = int(violated_days[0]) if len(violated_days) > 0 else -1
                active_cols_on_day = np.where(mask_active[first_viol])[0] if first_viol >= 0 and first_viol < n_active_days else []
                self._log("WARN", f"[BISECT-SANITY] incumbent_f={inc_f:.4f} feasible=False first_violated_day={first_viol} active_cols={list(active_cols_on_day)}")
                _engine_log.debug(f"[BISECT-SANITY] incumbent_f={inc_f:.4f} feasible=False first_violated_day={first_viol} active_cols={list(active_cols_on_day)}")
            else:
                self._log("DEBUG", f"[BISECT-SANITY] incumbent_f={inc_f:.4f} feasible=True")

        def _check_feasibility(f: float) -> Tuple[bool, Optional[np.ndarray]]:
            nonlocal n_lps
            # A_ub = -M1 + f*M2 (sparse scalar ops — no dense rebuild)
            A_ub_sparse = -M1_sparse + f * M2_sparse

            if A_strike_sparse is not None:
                from scipy.sparse import vstack as sparse_vstack
                A_ub_sparse = sparse_vstack([A_ub_sparse, A_strike_sparse], format='csr')
                b_ub_full = np.empty(n_active_days + n_extra, dtype=np.float64)
                b_ub_full[:n_active_days] = 0.0
                b_ub_full[n_active_days:] = b_strike_val
            else:
                b_ub_full = np.zeros(n_active_days, dtype=np.float64)

            n_lps += 1

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = linprog(
                    c=np.zeros(n, dtype=np.float64),
                    A_ub=A_ub_sparse, b_ub=b_ub_full,
                    A_eq=A_eq, b_eq=b_eq, bounds=w_bounds, method="highs",
                )
            if res.success:
                return True, res.x.copy()
            return False, None

        for _ in range(max_iters):
            if hi - lo < tol:
                break
            mid = (lo + hi) / 2.0
            feasible, w_candidate = _check_feasibility(mid)
            if feasible:
                lo = mid
                best_w = w_candidate
            else:
                hi = mid

        self._bisect_lp_count += n_lps
        f_star = lo

        if cache_key is not None:
            self._fstar_cache[cache_key] = f_star

        # Project weights for numerical safety (group-blind projection is
        # skipped when bucket bounds are active — LP output already honours them)
        if not group_bounds:
            if per_stock_bounds is not None:
                best_w = self._project_to_bounded_simplex(best_w, per_stock_bounds[:, 0], per_stock_bounds[:, 1])
            else:
                best_w = self._project_to_bounded_simplex(best_w, np.full(n, c.min_weight), np.full(n, c.max_weight))

        # Recompute actual adaptive min after projection
        f_actual = self._eval_adaptive_min(pnl_active, mask_active, best_w)

        # Score using the fitted score function on full adaptive series
        adaptive_pnl = self._eval_adaptive_series(sub_pnl, active_mask_sub, best_w)
        score = self._score_fn.score(adaptive_pnl, self._ctx)

        self._log("DEBUG", f"[BISECT] n_LPs={n_lps} f_star={f_star:.4f} f_actual={f_actual:.4f} n_days={n_active_days}")

        return WeightSolverResult(
            weights=best_w, score=score, feasible=True, n_evals=n_lps,
            extra={"min_payoff_achieved": f_actual, "bisect_n_lps": n_lps, "bisect_f_star": f_star},
        )

    def _solve_concave_blend_bisection(
        self,
        sub_pnl: np.ndarray,
        active_mask_sub: np.ndarray,
        strikes: Optional[np.ndarray],
        n: int,
        per_stock_bounds: Optional[np.ndarray] = None,
        tol: float = 0.01,
        cache_key: Optional[Tuple[int, ...]] = None,
        group_bounds=None,
    ) -> WeightSolverResult:
        """Lexicographic: first maximize floor via bisection, then maximize
        linear blend (mean_payoff + last_carry) subject to floor constraint.

        Step 1: bisect to find f_star (max achievable floor).
        Step 2: LP to maximize lam_mean*mean + lam_carry*carry subject to
                adaptive_pnl_d >= f_star*(1 - 1e-6) for all d.
        """
        c = self._constraints
        n_days, n_stocks = sub_pnl.shape

        # First: find f_star via bisection (reuse the min_payoff method)
        bisect_result = self._solve_min_payoff_bisection(
            sub_pnl, active_mask_sub, strikes, n, per_stock_bounds, tol, cache_key,
            group_bounds=group_bounds
        )
        if not bisect_result.feasible:
            return bisect_result

        f_star = bisect_result.extra.get("bisect_f_star", 0.0) if bisect_result.extra else 0.0
        f_constraint = f_star * (1 - 1e-6)  # slightly relax

        # Step 2: LP maximize linear blend subject to floor constraints
        lam_min, lam_mean, lam_carry, _K2 = concave_blend_lambdas(self._score_fn)

        # If only min_payoff is active (shouldn't reach here but safety), return bisect result
        if lam_mean == 0.0 and lam_carry == 0.0:
            return bisect_result

        # Objective: maximize lam_mean * mean(sub_pnl @ w) + lam_carry * mean(sub_pnl[-n_carry:] @ w)
        # = (lam_mean * col_means + lam_carry * col_carry_means) @ w
        col_means = sub_pnl.mean(axis=0)
        n_carry = min(SOLVER_TUNING.bisect_carry_window, n_days)  # ~3 months for "last_carry"
        col_carry_means = sub_pnl[-n_carry:].mean(axis=0) if n_carry > 0 else col_means

        obj_linear = -(lam_mean * col_means + lam_carry * col_carry_means)  # negate for minimize

        # Floor constraints: active[d,i] * w_i * (pnl[d,i] - f_constraint) >= 0
        day_has_any = active_mask_sub.any(axis=1)
        pnl_active = sub_pnl[day_has_any]
        mask_active = active_mask_sub[day_has_any]
        n_active_days = pnl_active.shape[0]

        coeff = mask_active * (pnl_active - f_constraint)
        A_ub = -coeff  # negate for <= form
        b_ub = np.zeros(n_active_days)

        # Strike constraint
        if strikes is not None and c.max_net_strike is not None:
            A_strike = strikes.reshape(1, -1)
            A_ub = np.vstack([A_ub, A_strike])
            b_ub = np.append(b_ub, c.max_net_strike)

        # Per-bucket weight bounds
        g_A, g_b = self._group_rows(n, n, group_bounds)
        if g_A is not None:
            A_ub = np.vstack([A_ub, g_A])
            b_ub = np.append(b_ub, g_b)

        # Equality: sum=1
        A_eq = np.zeros((1, n), dtype=np.float64)
        A_eq[0, :] = 1.0
        b_eq = np.array([1.0])

        if per_stock_bounds is not None:
            w_bounds = [(float(per_stock_bounds[i, 0]), float(per_stock_bounds[i, 1])) for i in range(n)]
        else:
            w_bounds = [(c.min_weight, c.max_weight)] * n

        A_ub_sparse = csr_matrix(A_ub)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = linprog(
                c=obj_linear, A_ub=A_ub_sparse, b_ub=b_ub,
                A_eq=A_eq, b_eq=b_eq, bounds=w_bounds, method="highs",
            )

        if res.success:
            w_opt = res.x
            if not group_bounds:
                if per_stock_bounds is not None:
                    w_opt = self._project_to_bounded_simplex(w_opt, per_stock_bounds[:, 0], per_stock_bounds[:, 1])
                else:
                    w_opt = self._project_to_bounded_simplex(w_opt, np.full(n, c.min_weight), np.full(n, c.max_weight))

            adaptive_pnl = self._eval_adaptive_series(sub_pnl, active_mask_sub, w_opt)
            score = self._score_fn.score(adaptive_pnl, self._ctx)
            f_actual = float(np.min(adaptive_pnl[active_mask_sub.any(axis=1)]))

            # Log both raw objectives for divergence visibility
            plain_mean = float(col_means @ w_opt)
            adaptive_mean = float(np.mean(adaptive_pnl))
            self._log("DEBUG", f"[BISECT-BLEND] f_star={f_star:.4f} plain_mean={plain_mean:.4f} adaptive_mean={adaptive_mean:.4f}")

            return WeightSolverResult(
                weights=w_opt, score=score, feasible=True, n_evals=bisect_result.n_evals + 1,
                extra={"min_payoff_achieved": f_actual, "bisect_f_star": f_star},
            )
        else:
            # Step 2 infeasible (rounding made floor too tight) — return bisect-only result
            return bisect_result

    @staticmethod
    def _eval_adaptive_min(pnl: np.ndarray, mask: np.ndarray, w: np.ndarray) -> float:
        """Compute min of adaptive-reweighted PnL over active days."""
        # adaptive_pnl_d = sum(mask[d]*w*pnl[d]) / sum(mask[d]*w)
        weighted_pnl = (mask * pnl) * w[np.newaxis, :]  # [days x n]
        weighted_active = mask * w[np.newaxis, :]
        numer = weighted_pnl.sum(axis=1)
        denom = weighted_active.sum(axis=1)
        # Only consider days with non-zero denom
        valid = denom > 1e-12
        if not valid.any():
            return -np.inf
        adaptive = numer[valid] / denom[valid]
        return float(np.min(adaptive))

    @staticmethod
    def _eval_adaptive_series(pnl: np.ndarray, mask: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Compute full adaptive-reweighted PnL series."""
        weighted_pnl = (mask * pnl) * w[np.newaxis, :]
        weighted_active = mask * w[np.newaxis, :]
        numer = weighted_pnl.sum(axis=1)
        denom = weighted_active.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(denom > 1e-12, numer / denom, 0.0)
        return result

    # --- Min-payoff maximin LP path ---

    def _solve_min_payoff_lp(
        self,
        sub_pnl: np.ndarray,
        strikes: Optional[np.ndarray],
        n: int,
        per_stock_bounds: Optional[np.ndarray] = None,
        full_active_days: Optional[np.ndarray] = None,
        group_bounds=None,
    ) -> WeightSolverResult:
        """Solve maximin LP for min_payoff-only optimization.

        Variables:
            w_1 ... w_n = trade weights
            t = worst daily payoff

        Maximize: t

        Subject to:
            pnl_matrix[d] @ w >= t  for every day d
            sum(w) = 1
            min_weight_i <= w_i <= max_weight_i

        Decision vector: x = [w_1, ..., w_n, t]  (length n+1)
        """
        c = self._constraints

        # Restrict to full-active days (where adaptive renorm is identity → linear exact)
        if full_active_days is not None:
            n_total = sub_pnl.shape[0]
            sub_pnl = sub_pnl[full_active_days]
            n_full = sub_pnl.shape[0]
            self._log("DEBUG", f"[WEIGHT-SOLVER] min_payoff LP days: full={n_full}/{n_total}")
            if n_full < SOLVER_TUNING.min_lp_days:
                self._log("WARNING", f"[WEIGHT-SOLVER] min_payoff LP: insufficient full-active days ({n_full}<{SOLVER_TUNING.min_lp_days})")
                return WeightSolverResult(weights=np.full(n, 1.0 / n), score=-np.inf, feasible=False, n_evals=0)

        n_days, n_stocks = sub_pnl.shape

        # Build decision vector: [w_1, ..., w_n, t]
        # Objective: minimize -t  (i.e., [0, 0, ..., 0, -1])
        obj = np.zeros(n + 1, dtype=np.float64)
        obj[-1] = -1.0  # minimize -t == maximize t

        # Bounds for w_i
        if per_stock_bounds is not None:
            w_bounds = [(float(per_stock_bounds[i, 0]), float(per_stock_bounds[i, 1])) for i in range(n)]
        else:
            w_bounds = [(c.min_weight, c.max_weight)] * n

        # t is unbounded (we'll clamp later if needed)
        t_bound = (None, None)
        bounds = w_bounds + [t_bound]

        # Equality constraint: sum(w) = 1
        A_eq = np.zeros((1, n + 1), dtype=np.float64)
        A_eq[0, :n] = 1.0
        b_eq = np.array([1.0])

        # Inequality constraints: -pnl_matrix[d] @ w + t <= 0
        # This is equivalent to: pnl_matrix[d] @ w >= t
        # Use sparse matrix for HiGHS efficiency (n_days x n+1 can be large)
        A_ub = np.zeros((n_days, n + 1), dtype=np.float64)
        A_ub[:, :n] = -sub_pnl  # negative because linprog uses A_ub @ x <= b_ub
        A_ub[:, -1] = 1.0  # +t
        b_ub = np.zeros(n_days, dtype=np.float64)

        # Strike constraint if provided
        if strikes is not None and c.max_net_strike is not None:
            A_ub_strike = np.zeros((1, n + 1), dtype=np.float64)
            A_ub_strike[0, :n] = strikes
            A_ub = np.vstack([A_ub, A_ub_strike])
            b_ub = np.append(b_ub, c.max_net_strike)

        # Per-bucket weight bounds (rows touch w only, 0 on t)
        g_A, g_b = self._group_rows(n + 1, n, group_bounds)
        if g_A is not None:
            A_ub = np.vstack([A_ub, g_A])
            b_ub = np.append(b_ub, g_b)

        # Convert to sparse for HiGHS performance
        A_ub_sparse = csr_matrix(A_ub) if A_ub.shape[0] > 0 else None

        res = linprog(
            c=obj,
            A_ub=A_ub_sparse,
            b_ub=b_ub if A_ub.shape[0] > 0 else None,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )

        if not res.success:
            # Infeasible/unbounded — log (rate-limited) and return infeasible result
            if not hasattr(self, '_minpayoff_infeasible_count'):
                self._minpayoff_infeasible_count = 0
            self._minpayoff_infeasible_count += 1
            if self._minpayoff_infeasible_count <= 3 or self._minpayoff_infeasible_count % 500 == 0:
                self._log("WARNING",
                    f"[WEIGHT-SOLVER] Min-payoff LP infeasible (#{self._minpayoff_infeasible_count}): n_trades={n_stocks}, n_days={n_days}, "
                    f"min_weight={c.min_weight}, max_weight={c.max_weight}, "
                    f"sum_constraint=1, max_net_strike={c.max_net_strike}, "
                    f"HiGHS status={res.status}, message={res.message}"
                )
            return WeightSolverResult(weights=np.full(n, 1.0 / n), score=-np.inf, feasible=False, n_evals=0)

        # Extract solution
        w_optimal = res.x[:n]
        t_optimal = res.x[-1]

        # Numerical cleanup — project onto simplex respecting per-stock bounds
        # (group-blind projection is skipped when bucket bounds are active)
        if not group_bounds:
            if per_stock_bounds is not None:
                w_optimal = self._project_to_bounded_simplex(w_optimal, per_stock_bounds[:, 0], per_stock_bounds[:, 1])
            else:
                w_optimal = self._project_to_bounded_simplex(w_optimal, np.full(n, c.min_weight), np.full(n, c.max_weight))

        # Recompute t after projection (may be lower due to projection)
        net_pnl = sub_pnl @ w_optimal
        t_recomputed = float(np.min(net_pnl))

        # Final score with STEP-function normalisation (should be ~t_recomputed after normalization)
        score = self._score_fn.score(net_pnl, self._ctx)

        return WeightSolverResult(
            weights=w_optimal,
            score=score,
            feasible=True,
            n_evals=0,
            extra={"min_payoff_achieved": t_recomputed},
        )

    # --- Non-linear SLSQP path ---

    def _smooth_surrogate_objective(self, sub_pnl: np.ndarray,
                                    strikes: Optional[np.ndarray],
                                    col_means: np.ndarray,
                                    vega: Optional[VegaSpec] = None):
        """Build the differentiable inner objective shared by the SLSQP solve
        and the 2-start convergence diagnostic.

        Raw values are made smooth first (``soft_hit_ratio`` surrogate,
        ``dot(w, strikes)`` for weighted_strike, the deterministic V-grid
        pick for the axe criteria in vega mode), then normalised through the
        interpolated CDF so every metric contributes on the same [0, 1] scale
        with the user's weights.  Returns ``objective(w) -> float`` to
        MINIMISE (negated score + tiny linear regularisation).
        """
        lam = self._score_fn.weights
        metric_map = {m.name: m for m in self._score_fn.metrics}
        ws_active = "weighted_strike" in lam.active_names
        lam_a = lam.get("axe_book_cleaned", 0.0)
        lam_b = lam.get("axe_package_recycled", 0.0)
        reg_strength = SOLVER_TUNING.reg_strength

        def objective(w: np.ndarray) -> float:
            net_pnl = sub_pnl @ w
            raw = {}
            axe_pair = None
            for name in lam.active_names:
                if name == "hit_ratio":
                    raw[name] = soft_hit_ratio(net_pnl)
                elif name == "weighted_strike":
                    if strikes is not None:
                        raw[name] = float(np.dot(w, strikes))
                    continue
                elif name in ("axe_book_cleaned", "axe_package_recycled"):
                    if vega is None:
                        continue
                    if axe_pair is None:
                        picked = self._vega_choose_V(w, vega, lam_a, lam_b)
                        if picked is None:
                            return 1e10  # caps make every V < v_min for this w
                        axe_pair = picked
                    raw[name] = (axe_pair[1] if name == "axe_book_cleaned"
                                 else axe_pair[2])
                else:
                    raw[name] = metric_map[name].compute(net_pnl, self._ctx)
                if math.isnan(raw[name]):
                    return 1e10
            score = self._score_fn.score_smooth_from_raw(raw)
            reg = reg_strength * np.dot(col_means, w)
            return -(score + reg)

        return objective

    def _solve_nonlinear(
        self,
        sub_pnl: np.ndarray,
        strikes: Optional[np.ndarray],
        n: int,
        per_stock_bounds: Optional[np.ndarray] = None,
        sweep_key: Optional[Tuple[int, ...]] = None,
        group_bounds=None,
        vega: Optional[VegaSpec] = None,
    ) -> WeightSolverResult:
        """Solve via scipy SLSQP with multi-start + a deterministic step sweep.

        SLSQP optimises the smooth surrogate (interpolated CDF, soft hit
        ratio); the final selection then compares the SLSQP optimum against a
        deterministic Dirichlet sweep (plus a coarse simplex grid for small
        n) evaluated on the STEP score — step-dominated objectives such as
        hit_ratio live on plateaus the surrogate's gradient cannot see.  The
        sweep RNG is seeded from (solver seed, subset key) so results are
        reproducible and independent of call order.
        """
        c = self._constraints
        n_evals = 0

        # Compute per-stock mean as regularisation direction (breaks plateaus)
        col_means = sub_pnl.mean(axis=0)
        reg_strength = SOLVER_TUNING.reg_strength

        lam = self._score_fn.weights
        ws_active = "weighted_strike" in lam.active_names
        if ws_active and strikes is None:
            # Cannot evaluate the strike objective without per-stock strikes.
            self._log("WARNING",
                      "[WEIGHT-SOLVER] weighted_strike is weighted but no strikes "
                      "were passed — the strike term is ignored in this solve.")

        # Objective: MINIMISE negative smooth-normalised score + tiny linear
        # regularisation (shared with diagnose_convergence — see
        # _smooth_surrogate_objective for the units rationale).
        _base_objective = self._smooth_surrogate_objective(sub_pnl, strikes, col_means,
                                                           vega=vega)
        _lam_a = lam.get("axe_book_cleaned", 0.0)
        _lam_b = lam.get("axe_package_recycled", 0.0)
        _axe_active_nl = (vega is not None and (_lam_a > 0 or _lam_b > 0))

        def objective(w: np.ndarray) -> float:
            nonlocal n_evals
            n_evals += 1
            return _base_objective(w)

        # Bounds: use per-stock if provided, otherwise global
        if per_stock_bounds is not None:
            bounds = [(float(per_stock_bounds[i, 0]), float(per_stock_bounds[i, 1])) for i in range(n)]
        else:
            bounds = [(c.min_weight, c.max_weight)] * n

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        ]
        if strikes is not None and c.max_net_strike is not None:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w, s=strikes: c.max_net_strike - np.dot(w, s),
                }
            )
        # Per-bucket weight bounds as SLSQP inequalities
        if group_bounds:
            for pos, lo, hi in group_bounds:
                pos = np.asarray(pos, dtype=int)
                if pos.size == 0:
                    continue
                if hi is not None:
                    constraints.append(
                        {"type": "ineq",
                         "fun": lambda w, _p=pos, _hi=float(hi): _hi - float(np.sum(w[_p]))})
                if lo is not None and lo > 0.0:
                    constraints.append(
                        {"type": "ineq",
                         "fun": lambda w, _p=pos, _lo=float(lo): float(np.sum(w[_p])) - _lo})

        # Multi-start
        best_w = np.full(n, 1.0 / n)
        best_score = -np.inf

        starts = self._generate_starts(n)

        # Greedy start: max weight on the stock with best mean P&L
        greedy_w = np.full(n, c.min_weight)
        remaining = 1.0 - greedy_w.sum()
        best_stock = np.argmax(col_means)
        add = min(remaining, c.max_weight - greedy_w[best_stock])
        greedy_w[best_stock] += add
        remaining -= add
        if remaining > 1e-9:
            for i in np.argsort(-col_means):
                if i == best_stock:
                    continue
                a = min(remaining, c.max_weight - greedy_w[i])
                greedy_w[i] += a
                remaining -= a
                if remaining < 1e-12:
                    break
        # Clip greedy start into bounds to avoid scipy warning
        greedy_w = np.clip(greedy_w, c.min_weight, c.max_weight)
        greedy_w = greedy_w / greedy_w.sum()
        starts.append(greedy_w)

        for w0 in starts:
            try:
                res = minimize(
                    objective,
                    w0,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={"maxiter": SOLVER_TUNING.slsqp_maxiter, "ftol": 1e-6},
                )
                if res.success or res.fun < -best_score:
                    score_val = -res.fun
                    if score_val > best_score:
                        best_score = score_val
                        best_w = res.x.copy()
            except Exception:
                continue

        # Ensure constraints are satisfied numerically
        best_w = np.clip(best_w, c.min_weight, c.max_weight)
        best_w = best_w / best_w.sum()

        # ── Per-stock bound vectors (for projection of sweep candidates) ──
        if per_stock_bounds is not None:
            lb = per_stock_bounds[:, 0]
            ub = per_stock_bounds[:, 1]
        else:
            lb = np.full(n, c.min_weight)
            ub = np.full(n, c.max_weight)

        metric_map_all = {m.name: m for m in self._score_fn.metrics}

        def _step_eval(w: np.ndarray) -> Optional[Tuple[float, float]]:
            """(rank_value, step_score) of w, or None if strike-infeasible.

            The quantile step score saturates at 1.0 once a candidate beats
            the whole reference — ranking must then fall back to the
            scalarised raw objective (same tie-break as the GA fitness) or
            the sweep cannot distinguish top-plateau candidates.
            """
            if (strikes is not None and c.max_net_strike is not None
                    and float(np.dot(w, strikes)) > c.max_net_strike + 1e-9):
                return None
            if not self._groups_satisfied(w, group_bounds):
                return None
            _repl = {}
            if ws_active and strikes is not None:
                _repl["weighted_strike"] = float(np.dot(w, strikes))
            if _axe_active_nl:
                _picked = self._vega_choose_V(w, vega, _lam_a, _lam_b)
                if _picked is None:
                    return None  # caps leave no feasible V for this w
                _repl["axe_cleaned"] = _picked[1]
                _repl["axe_recycled"] = _picked[2]
            net = sub_pnl @ w
            ctx_w = dataclasses.replace(self._ctx, **_repl) if _repl else self._ctx
            step = self._score_fn.score(net, ctx_w)
            if math.isnan(step) or math.isinf(step):
                return None
            if step < SOLVER_TUNING.tiebreak_threshold:
                return (float(step), float(step))
            raw = self._score_fn.raw_metrics(net, ctx_w)
            scalar = 0.0
            for name in lam.active_names:
                v = raw.get(name, float("nan"))
                if not np.isfinite(v):
                    return None
                scalar += lam[name] * (1.0 if metric_map_all[name].higher_is_better else -1.0) * v
            return (float(2.0 + math.tanh(scalar)), float(step))

        # ── Deterministic sweep on the STEP score ──
        # Group-infeasible candidates are filtered by _step_eval; with bucket
        # bounds the raw SLSQP point is kept unprojected (the iterative
        # projection is group-blind) and an L1 group-feasible projection of
        # each candidate is added instead.
        if group_bounds:
            cand_ws = [best_w.copy()]
            proj = self.project_to_group_feasible(
                best_w, lb, ub, group_bounds, strikes,
                c.max_net_strike if strikes is not None else None)
            if proj is not None:
                cand_ws.append(proj)
        else:
            cand_ws = [self._project_to_bounded_simplex(best_w.copy(), lb, ub)]
        sweep_entropy = [self._seed] + ([int(i) for i in sweep_key]
                                        if sweep_key is not None else [n])
        sweep_rng = np.random.default_rng(sweep_entropy)

        # LP projections are capped: they cost one HiGHS solve each.  The
        # first few group-violating candidates get projected to a feasible
        # neighbour; later ones are simply filtered by _step_eval.
        _proj_budget = [SOLVER_TUNING.sweep_projection_budget]

        def _add_candidate(w):
            w2 = self._project_to_bounded_simplex(w, lb, ub)
            cand_ws.append(w2)
            if (group_bounds and _proj_budget[0] > 0
                    and not self._groups_satisfied(w2, group_bounds)):
                _proj_budget[0] -= 1
                proj2 = self.project_to_group_feasible(
                    w2, lb, ub, group_bounds, strikes,
                    c.max_net_strike if strikes is not None else None)
                if proj2 is not None:
                    cand_ws.append(proj2)

        for _ in range(SOLVER_TUNING.sweep_dirichlet):
            _add_candidate(sweep_rng.dirichlet(np.ones(n)))
        if n <= SOLVER_TUNING.sweep_grid_max_n:
            grid = np.linspace(0.0, 1.0, SOLVER_TUNING.sweep_grid_steps)
            for combo in itertools.product(grid, repeat=n):
                s = float(sum(combo))
                if s <= 0.0:
                    continue
                _add_candidate(np.asarray(combo, dtype=np.float64) / s)

        best_rank = -np.inf
        best_step = -np.inf
        best_step_w = None
        for w in cand_ws:
            out = _step_eval(w)
            n_evals += 1
            if out is not None and out[0] > best_rank:
                best_rank, best_step = out
                best_step_w = w

        if best_step_w is not None:
            best_w = best_step_w
            final_score = float(best_step)
            if _axe_active_nl or vega is not None:
                _picked = self._vega_choose_V(best_w, vega, _lam_a, _lam_b) if vega is not None else None
                if _picked is None and vega is not None:
                    return WeightSolverResult(weights=np.full(n, 1.0 / n), score=-np.inf,
                                              feasible=False, n_evals=n_evals)
                if _picked is not None:
                    return WeightSolverResult(
                        weights=best_w, score=final_score, feasible=True, n_evals=n_evals,
                        extra={"vega_total": _picked[0], "axe_cleaned": _picked[1],
                               "axe_recycled": _picked[2]})
        elif group_bounds:
            # No candidate satisfies the bucket bounds for this subset —
            # surface infeasibility instead of returning violating weights.
            return WeightSolverResult(weights=np.full(n, 1.0 / n), score=-np.inf,
                                      feasible=False, n_evals=n_evals)
        else:
            # All candidates strike-infeasible (should not happen: the SLSQP
            # optimum honoured the constraint) — keep SLSQP weights, step score.
            net_pnl = sub_pnl @ best_w
            final_ctx = self._ctx
            if ws_active and strikes is not None:
                final_ctx = dataclasses.replace(
                    self._ctx, weighted_strike=float(np.dot(best_w, strikes))
                )
            final_score = self._score_fn.score(net_pnl, final_ctx)

        return WeightSolverResult(
            weights=best_w,
            score=final_score,
            feasible=True,
            n_evals=n_evals,
        )

    def _generate_starts(self, n: int) -> list:
        """Generate feasible starting points for multi-start optimisation."""
        c = self._constraints
        starts = []

        # Start 1: equal weight (always feasible if _is_feasible passed)
        starts.append(np.full(n, 1.0 / n))

        # Start 2+: random feasible points via Dykstra-style projection
        for _ in range(self._n_restarts - 1):
            w = self._rng.uniform(c.min_weight, c.max_weight, size=n)
            # Project onto simplex with box constraints (iterative)
            for _ in range(20):
                w = w / w.sum()  # project onto sum=1
                w = np.clip(w, c.min_weight, c.max_weight)  # project onto box
                if abs(w.sum() - 1.0) < 1e-10:
                    break
            # Final normalization
            w = w / w.sum()
            starts.append(w)

        return starts
