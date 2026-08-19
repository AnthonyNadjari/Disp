"""
Backtest engine — swap calculation, basket backtesting, and data loading.

NAMING CONVENTION (the reference for this whole package — name by ROLE, never
by asset class, because the class is only the usual case while the role never
flips):

  Variance Asset            usually the INDEX (e.g. 'HSCEI Index').
                            Underlies the CROSS leg: index variance observed
                            inside the stock's corridor. Shared across rows.
                            Its price series is ``variance_px``.

  Corridor Condition Asset  usually the STOCK (e.g. '005930 KP Equity').
                            Underlies the MONO leg (var/vol swap on the stock),
                            defines the corridor for BOTH legs, and is the
                            per-name key (P&L column, forced tickers, vega).
                            Its price series is ``corridor_px``.

  Cross-corridor leg P&L  = ( MONO(stock) − CROSS(index) ) × 100
                            pnl_mono / pnl_cross in code. "Long/Short Leg" in
                            results refers to the BASKET sides, never to
                            mono/cross.

  Mono mode (cross_corridor=False): each leg's variance_px is the traded name
  itself; the corridor is observed on itself (corridor product) or absent
  (vol swap). The same role names hold.

Missing-data policies (how gap days are filled):
  - FILL_ZERO             gap day contributes 0 for that name (legacy Gaia_PP)
  - DROP_INCOMPLETE_DAYS  keep only days where EVERY weighted name trades
  - ADAPTIVE_REWEIGHT     redistribute a gapped name's weight to active names
                          (``reweight_grace_days`` can hold its slot open for
                          short gaps)

Uses numpy throughout for performance — no pandas row iteration.
"""

from __future__ import annotations

import warnings

import numpy as np
import numba as nb
import pandas as pd
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Optional, Tuple, Any

from functions.dispersion.scoring.weight_solver import (
    active_mask_with_grace, carry_pnl_within_grace)
from functions.dispersion.models import (
    SwapConfig,
    BacktestResult,
    MissingDataPolicy,
    DispersionLeg,
    BasketInput,
    ProductType,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Swap Calculator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Numba kernels (cached for fast re-execution)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@nb.jit(nopython=True, cache=True)
def _vol_swap_window(prices: np.ndarray, strike: float, n_exp: int, local_cap: float, capped: bool = True) -> float:
    """Vol swap P&L for a single rolling window. Skips first return to match original."""
    if np.isnan(prices).any():
        return np.nan
    n = len(prices)
    sq_logs = np.empty(n - 1)
    for i in range(1, n):
        if prices[i - 1] <= 0 or prices[i] <= 0:
            return np.nan   # non-positive price → NaN, never inf via log(0)
        sq_logs[i - 1] = np.log(prices[i] / prices[i - 1]) ** 2
    # Original Gaia_PP uses sq_logs[1:] — skips the first daily return in window
    realized = np.sqrt(np.sum(sq_logs[1:]) * 252.0 / n_exp)
    if strike != 0:
        # capped=False = uncapped OTC swap: no local_cap on the realized leg
        return min(realized, local_cap * strike) - strike if capped else realized - strike
    return realized

@nb.jit(nopython=True, cache=True)
def _corridor_varswap_window(
    prices_var: np.ndarray,
    prices_corr: np.ndarray,
    strike: float,
    ubar: float,
    dbar: float,
    n_exp: int,
    local_cap: float,
    capped: bool = True,
) -> float:
    """Corridor variance swap P&L. Corridor observed on prices_corr, variance on prices_var."""
    if np.isnan(prices_var).any() or np.isnan(prices_corr).any():
        return np.nan

    upper = prices_corr[0] * ubar
    lower = prices_corr[0] * dbar
    n = len(prices_var)
    sq_logs = np.empty(n - 1)
    for i in range(1, n):
        if prices_var[i - 1] <= 0 or prices_var[i] <= 0:
            return np.nan   # non-positive price → NaN, never inf via log(0)
        sq_logs[i - 1] = np.log(prices_var[i] / prices_var[i - 1]) ** 2
    min_len = min(len(prices_var), len(prices_corr))
    in_corridor = np.zeros(min_len - 1)
    for i in range(1, min_len):
        if lower <= prices_corr[i] <= upper and lower <= prices_corr[i - 1] <= upper:
            in_corridor[i - 1] = 1.0
    M = np.sum(in_corridor)
    if M > 0:
        corridor_sum = 0.0
        for i in range(len(in_corridor)):
            if in_corridor[i] > 0:
                corridor_sum += sq_logs[i]
        # Expected Var mode: strike=0 means return realized variance (no cap, no P&L)
        if strike == 0.0:
            return 252.0 / n_exp * corridor_sum
        # capped=False = uncapped OTC swap: no local_cap on realized variance
        cap_bound = (strike * local_cap) ** 2 if capped else np.inf
        capped_var = min(252.0 / M * corridor_sum, cap_bound)
        return ((capped_var - strike ** 2) * M / n_exp) / (2.0 * strike)
    return 0.0

@nb.jit(nopython=True, cache=True)
def _rolling_pnl_volswap(prices: np.ndarray, strike: float, n_exp: int, local_cap: float, capped: bool = True) -> np.ndarray:
    """
    Full rolling vol swap P&L array with skip-NaN windowing.
    Matches original Gaia_PP behavior: for each date, take the last n_exp+1 VALID
    observations (skipping NaN days), allowing the window to extend further back.
    Optimized: O(n) cumulative valid count instead of O(n^2) linear search.
    """
    n = len(prices)
    out = np.full(n, np.nan)
    # Pre-compute valid indices
    valid_indices = np.empty(n, dtype=np.int64)
    n_valid = 0
    for i in range(n):
        if not np.isnan(prices[i]):
            valid_indices[n_valid] = i
            n_valid += 1
    # Pre-compute cumulative valid count at each position (O(n))
    cum_valid = np.zeros(n, dtype=np.int64)
    count = 0
    for i in range(n):
        if not np.isnan(prices[i]):
            count += 1
        cum_valid[i] = count
    # For each date, use cumulative count to find window start (O(1) per date)
    for i in range(n):
        # No-trade / missing day for this name: emit NaN, never carry a stale
        # window forward onto a day the name did not actually trade.
        if np.isnan(prices[i]):
            continue
        c = cum_valid[i]
        if c < n_exp + 1:
            continue

        # Extract last n_exp+1 valid prices
        window = np.empty(n_exp + 1)
        start_v = c - (n_exp + 1)
        for k in range(n_exp + 1):
            window[k] = prices[valid_indices[start_v + k]]
        out[i] = _vol_swap_window(window, strike, n_exp, local_cap, capped)
    return out

@nb.jit(nopython=True, cache=True)
def _rolling_pnl_corridor(
    prices_var: np.ndarray,
    prices_corr: np.ndarray,
    strike: float,
    ubar: float,
    dbar: float,
    n_exp: int,
    local_cap: float,
    capped: bool = True,
) -> np.ndarray:
    """
    Full rolling corridor var swap P&L array with skip-NaN windowing.
    For each date, take last n_exp+1 observations where BOTH var and corr are valid.
    """
    n = min(len(prices_var), len(prices_corr))
    out = np.full(n, np.nan)
    # Pre-compute jointly-valid indices + cumulative count (O(n), same trick as
    # the volswap kernel — replaces the per-row rescan that made this O(n²))
    valid_indices = np.empty(n, dtype=np.int64)
    n_valid = 0
    cum_valid = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if not np.isnan(prices_var[i]) and not np.isnan(prices_corr[i]):
            valid_indices[n_valid] = i
            n_valid += 1
        cum_valid[i] = n_valid
    for i in range(n):
        # No-trade / missing day on either series: emit NaN (don't carry stale P&L).
        if np.isnan(prices_var[i]) or np.isnan(prices_corr[i]):
            continue
        count = cum_valid[i]
        if count < n_exp + 1:
            continue

        start_v = count - (n_exp + 1)
        w_var = np.empty(n_exp + 1)
        w_corr = np.empty(n_exp + 1)
        for k in range(n_exp + 1):
            idx = valid_indices[start_v + k]
            w_var[k] = prices_var[idx]
            w_corr[k] = prices_corr[idx]
        out[i] = _corridor_varswap_window(w_var, w_corr, strike, ubar, dbar, n_exp, local_cap, capped)
    return out

class SwapCalculator:
    """
    Computes rolling P&L for vol swaps, var swaps, and corridor var swaps.
    Wraps numba kernels with a clean Python API.
    Usage:
        calc = SwapCalculator(config)
        pnl_array = calc.compute(prices, strike=0.22)
        pnl_cross = calc.compute(variance_px_col, strike=0.25, corridor_prices=corridor_px_col)
    """

    def __init__(self, config: SwapConfig):
        self.config = config

    def compute(
        self,
        prices: np.ndarray,
        strike: float,
        corridor_prices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute rolling swap P&L for one leg.
        Args:
            prices: Price array (variance asset)
            strike: Swap strike as decimal (e.g. 0.22)
            corridor_prices: If provided, corridor observed on this asset (cross-corridor)
        Returns:
            np.ndarray of P&L values. NaN where insufficient data.
        """
        c = self.config
        # Expected Var mode: strike=0, compute realized variance/vol (no P&L)
        if strike == 0.0:
            if c.is_vol_swap:
                return _rolling_pnl_volswap(prices, 0.0, c.n_exp, c.local_cap, c.capped)
            elif corridor_prices is not None:
                return _rolling_pnl_corridor(prices, corridor_prices, 0.0, c.barrier_up, c.barrier_down, c.n_exp, c.local_cap, c.capped)
            else:
                return _rolling_pnl_corridor(prices, prices, 0.0, c.barrier_up, c.barrier_down, c.n_exp, c.local_cap, c.capped)

        if c.is_vol_swap:
            return _rolling_pnl_volswap(prices, strike, c.n_exp, c.local_cap, c.capped)
        elif corridor_prices is not None:
            # Cross-corridor: variance from `prices`, corridor from `corridor_prices`
            return _rolling_pnl_corridor(prices, corridor_prices, strike, c.barrier_up, c.barrier_down, c.n_exp, c.local_cap, c.capped)
        else:
            # Standard corridor: same asset for both
            return _rolling_pnl_corridor(prices, prices, strike, c.barrier_up, c.barrier_down, c.n_exp, c.local_cap, c.capped)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shared leg-P&L builders — ONE source of truth for the optimizer matrix
# (_api._build_pnl_matrix) and the backtester (_compute_all_pnl)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _leg_pnl_cross_corridor(
    variance_px_col: np.ndarray,
    corridor_px_col: np.ndarray,
    leg: DispersionLeg,
    config: SwapConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Both sub-legs of ONE cross-corridor leg (per the module convention):

        mono  = swap on the STOCK — variance AND corridor on corridor_px_col
        cross = swap on the INDEX inside the stock's corridor —
                variance on variance_px_col, corridor on corridor_px_col

    Returns (pnl_mono, pnl_cross), rolling P&L arrays in decimal (×100 is
    applied by the caller). The leg P&L is pnl_mono − pnl_cross.
    """
    strike_cross = leg.strike_cross_corridor if leg.strike_cross_corridor is not None else leg.strike_mono_var_swap
    if config.is_vol_swap:
        pnl_mono = _rolling_pnl_volswap(
            corridor_px_col, leg.strike_mono_var_swap, config.n_exp, config.local_cap, config.capped)
        pnl_cross = _rolling_pnl_volswap(
            variance_px_col, strike_cross, config.n_exp, config.local_cap, config.capped)
    else:
        pnl_mono = _rolling_pnl_corridor(
            corridor_px_col, corridor_px_col, leg.strike_mono_var_swap,
            config.barrier_up, config.barrier_down, config.n_exp, config.local_cap, config.capped)
        pnl_cross = _rolling_pnl_corridor(
            variance_px_col, corridor_px_col, strike_cross,
            config.barrier_up, config.barrier_down, config.n_exp, config.local_cap, config.capped)
    return pnl_mono, pnl_cross


def compute_leg_pnl_columns(
    variance_px: pd.DataFrame,
    corridor_px: Optional[pd.DataFrame],
    legs: List[DispersionLeg],
    config: SwapConfig,
    capture_cross_legs: bool = False,
) -> Tuple[np.ndarray, List[str], Dict[str, dict]]:
    """Per-leg P&L columns — the single builder behind both engines.

    Column key = Corridor Condition Asset (the stock) in cross-corridor mode,
    else Variance Asset. A cross-corridor leg whose corridor price column is
    missing is DROPPED with a warning — never silently degraded to a plain
    index swap. A duplicate key raises: columns are keyed by it, so a
    duplicate would silently misalign every later column.

    Returns (pnl_matrix [n_rows × n_legs], column_keys, cross_legs) where
    cross_legs maps key → {'mono_pnl','cross_pnl','corridor_asset',
    'variance_asset'} (populated only when capture_cross_legs and cross mode).
    """
    calc = SwapCalculator(config)
    if corridor_px is not None and not getattr(corridor_px, "empty", True):
        # Rows are paired positionally below — both series must share one
        # calendar. The loader union-reindexes already (no-op there); this
        # guards direct/headless callers passing misaligned frames.
        if not corridor_px.index.equals(variance_px.index):
            corridor_px = corridor_px.reindex(variance_px.index)
    leg_map: Dict[str, DispersionLeg] = {}
    column_keys: List[str] = []
    skipped_missing_corr = []
    skipped_missing_var = []
    corr_cols = set(corridor_px.columns) if (
        corridor_px is not None and not getattr(corridor_px, "empty", True)) else set()
    for leg in legs:
        if leg.variance_asset not in variance_px.columns:
            skipped_missing_var.append(leg.variance_asset)
            continue
        if config.is_cross_corridor:
            if not leg.corridor_condition_asset or leg.corridor_condition_asset not in corr_cols:
                skipped_missing_corr.append(leg.corridor_condition_asset or leg.variance_asset)
                continue
            key = leg.corridor_condition_asset
        else:
            key = leg.variance_asset
        if key in leg_map:
            key_label = ("Corridor Condition Asset" if config.is_cross_corridor
                         else "Variance Asset")
            raise ValueError(
                f"Duplicate candidate key '{key}': the same {key_label} appears on "
                f"more than one input row. P&L columns are keyed by it — duplicates "
                f"would silently misalign the matrix. Keep one row per name.")
        leg_map[key] = leg
        column_keys.append(key)
    if skipped_missing_corr:
        warnings.warn(
            f"Cross-corridor: {len(skipped_missing_corr)} leg(s) dropped — their corridor "
            f"stock price did not load from Bloomberg, so no valid cross-corridor P&L can "
            f"be computed (no silent fall-back to an index swap). "
            f"Sample: {sorted(set(map(str, skipped_missing_corr)))[:10]}",
            stacklevel=2,
        )
    if skipped_missing_var:
        # A dropped leg's weight silently vanishes from the basket — never mute.
        warnings.warn(
            f"{len(skipped_missing_var)} leg(s) dropped — their variance price did not "
            f"load from Bloomberg; any weight on them is undeployed. "
            f"Sample: {sorted(set(map(str, skipped_missing_var)))[:10]}",
            stacklevel=2,
        )

    n_rows = len(variance_px)
    pnl_matrix = np.full((n_rows, len(column_keys)), np.nan)
    cross_legs: Dict[str, dict] = {}
    for i, key in enumerate(column_keys):
        leg = leg_map[key]
        variance_px_col = variance_px[leg.variance_asset].values.astype(np.float64)
        if config.is_cross_corridor:
            corridor_px_col = corridor_px[leg.corridor_condition_asset].values.astype(np.float64)
            pnl_mono, pnl_cross = _leg_pnl_cross_corridor(
                variance_px_col, corridor_px_col, leg, config)
            n = min(len(pnl_mono), len(pnl_cross), n_rows)
            pnl_matrix[:n, i] = (pnl_mono[:n] - pnl_cross[:n]) * 100
            if capture_cross_legs:
                cross_legs[key] = {
                    'mono_pnl': pnl_mono[:n] * 100,
                    'cross_pnl': pnl_cross[:n] * 100,
                    'corridor_asset': leg.corridor_condition_asset,
                    'variance_asset': leg.variance_asset,
                }
        else:
            pnl = calc.compute(variance_px_col, leg.strike_mono_var_swap, corridor_prices=None)
            n = min(len(pnl), n_rows)
            pnl_matrix[:n, i] = pnl[:n] * 100
    return pnl_matrix, column_keys, cross_legs


def series_key(leg: DispersionLeg, is_cross_corridor: bool) -> str:
    """Canonical weights/indices key for a leg: Corridor Condition Asset in
    cross-corridor mode, else variance_asset. Single source of truth — the
    backtester method and the _api helpers both delegate here."""
    if is_cross_corridor and leg.corridor_condition_asset:
        return leg.corridor_condition_asset
    return leg.variance_asset


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Dispersion Backtester
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DispersionBacktester:
    """
    Runs basket-level backtests.
    Two missing-data strategies:
      1. DROP_INCOMPLETE_DAYS: Remove any day where ANY basket leg has no data.
      2. ADAPTIVE_REWEIGHT: Redistribute weight from missing legs to available ones.
    Usage:
        config = SwapConfig.volswap(n_exp=310, missing_data_policy=MissingDataPolicy.ADAPTIVE_REWEIGHT)
        bt = DispersionBacktester(config)
        result = bt.run(variance_px, legs, weights)
    """

    def __init__(self, config: SwapConfig):
        self.config = config
        self._calc = SwapCalculator(config)
        self._cross_legs = {}  # populated during cross-corridor _compute_all_pnl

    def _series_key(self, leg: DispersionLeg) -> str:
        """Return key for weights/indices: Corridor Condition Asset for cross-corridor, else variance_asset."""
        return series_key(leg, self.config.is_cross_corridor)

    def run(
        self,
        variance_px: pd.DataFrame,
        legs: List[DispersionLeg],
        weights: Dict[str, float],
        corridor_px: Optional[pd.DataFrame] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> BacktestResult:
        """
        Run basket backtest.
        Args:
            variance_px: Variance Asset prices (columns = tickers, index = dates)
            legs: List of DispersionLeg objects
            weights: {ticker: weight} — positive=long, negative=short
            corridor_px: Corridor Condition Asset (stock) prices — cross-corridor only
            start_date: Filter output from this date (default: 20 years ago — fetch max available history)
            end_date: Filter output up to this date inclusive (default: no upper bound).
                Must match the optimizer's scoring window so reported metrics don't
                include post-cutoff (out-of-sample) data.
        """
        if start_date is None:
            start_date = date.today() - relativedelta(years=20)  # fetch max available history
        # Step 1: Compute per-leg rolling P&L as numpy matrix
        pnl_matrix, pnl_column_keys, dates_index = self._compute_all_pnl(variance_px, legs, corridor_px)
        # Step 1b: Convert weights to use pnl_column_keys (Corridor Condition Asset for cross-corridor)
        weights_by_key = {}
        for leg in legs:
            key = self._series_key(leg)
            if key in weights:
                weights_by_key[key] = weights[key]
        # A weighted leg whose P&L column was never built (price load failed)
        # silently undeploys its weight — surface it loudly.
        _dropped_weighted = [k for k, w in weights_by_key.items()
                             if w != 0 and k not in set(pnl_column_keys)]
        if _dropped_weighted:
            warnings.warn(
                f"{len(_dropped_weighted)} weighted leg(s) have no P&L column "
                f"(price data missing) — their weight is NOT deployed in this "
                f"backtest: {sorted(map(str, _dropped_weighted))[:10]}",
                stacklevel=2,
            )
        # Step 2: Apply missing data policy (all numpy)
        policy = self.config.missing_data_policy
        if policy == MissingDataPolicy.FILL_ZERO:
            long_pnl, short_pnl, active_count, valid_mask = self._apply_fill_zero_policy(
                pnl_matrix, pnl_column_keys, weights_by_key
            )
        elif policy == MissingDataPolicy.DROP_INCOMPLETE_DAYS:
            long_pnl, short_pnl, active_count, valid_mask = self._apply_drop_policy(
                pnl_matrix, pnl_column_keys, weights_by_key
            )
        else:
            long_pnl, short_pnl, active_count, valid_mask = self._apply_adaptive_policy(
                pnl_matrix, pnl_column_keys, weights_by_key
            )
        # Step 3: Build result DataFrame
        dates = dates_index[valid_mask] if valid_mask is not None else dates_index
        result = long_pnl + short_pnl
        # Apply global cap/floor (vol swap only) — clips the final basket payout
        if self.config.is_vol_swap:
            result = np.clip(result, self.config.global_floor, self.config.global_cap)
        result_df = pd.DataFrame({
            "Long Leg": long_pnl,
            "Short Leg": short_pnl,
            "Result": result,
        }, index=dates)
        # Compute actual active count from weighted legs only (for both policies)
        # Build index arrays for long/short to identify which legs are weighted
        long_idx = [i for i, t in enumerate(pnl_column_keys) if weights_by_key.get(t, 0) > 0]
        short_idx = [i for i, t in enumerate(pnl_column_keys) if weights_by_key.get(t, 0) < 0]
        all_weighted_idx = long_idx + short_idx
        if self.config.missing_data_policy == MissingDataPolicy.DROP_INCOMPLETE_DAYS:
            # DROP_INCOMPLETE_DAYS: all weighted legs must have data, so active count is constant
            if all_weighted_idx:
                actual_active = np.full(len(dates), len(all_weighted_idx), dtype=int)
            else:
                actual_active = np.zeros(len(dates), dtype=int)
        elif self.config.missing_data_policy == MissingDataPolicy.FILL_ZERO:
            # FILL_ZERO: count legs with actual non-NaN data (but all rows are kept)
            if all_weighted_idx:
                weighted_pnl = pnl_matrix[:, all_weighted_idx]
                actual_active = np.sum(~np.isnan(weighted_pnl), axis=1)
            else:
                actual_active = np.zeros(len(dates), dtype=int)
        else:
            # ADAPTIVE_REWEIGHT: a name counts as active while it holds weight —
            # including in-grace gap days (its weight stays allocated then).
            grace = self.config.reweight_grace_days
            is_valid = ~np.isnan(pnl_matrix)
            active_mask = active_mask_with_grace(is_valid, grace)
            if valid_mask is not None:
                active_sub = active_mask[valid_mask][:, all_weighted_idx]
            else:
                active_sub = active_mask[:, all_weighted_idx]
            actual_active = active_sub.sum(axis=1)
        active_series = pd.Series(actual_active, index=dates, name="Active Stocks")

        # Build per-leg P&L DataFrame for individual contributions view
        per_leg_df = pd.DataFrame(pnl_matrix, index=dates_index, columns=pnl_column_keys)
        if valid_mask is not None:
            per_leg_df = per_leg_df[valid_mask]
        # Step 4: Filter to [start_date, end_date]
        start_date = pd.Timestamp(start_date).normalize()
        _idx_norm = pd.to_datetime(result_df.index).normalize()
        mask = _idx_norm >= start_date
        if end_date is not None:
            # Upper-bound the delivered curve/metrics to the optimizer's window —
            # no post-cutoff (out-of-sample) leak into reported results.
            mask = mask & (_idx_norm <= pd.Timestamp(end_date).normalize())
        result_df = result_df[mask]
        if active_series is not None:
            active_series = active_series[mask]
        per_leg_df = per_leg_df[per_leg_df.index.isin(result_df.index)]
        # For cross-corridor: build separate leg dataframes
        cross_leg_df = None
        if self.config.is_cross_corridor and self._cross_legs:
            leg_columns = {}
            for leg_key, leg_data in self._cross_legs.items():
                variance_asset = leg_data['variance_asset']
                # Pad to match dates_index length
                stock_pnl = np.full(len(dates_index), np.nan)  # mono leg
                index_pnl = np.full(len(dates_index), np.nan)  # cross leg
                n = len(leg_data['mono_pnl'])
                stock_pnl[:n] = leg_data['mono_pnl']
                index_pnl[:n] = leg_data['cross_pnl']
                leg_columns[f"{leg_key} (mono leg — stock var)"] = stock_pnl
                leg_columns[f"{leg_key} (cross leg — index var: {variance_asset})"] = index_pnl
            cross_leg_df = pd.DataFrame(leg_columns, index=dates_index)
            if valid_mask is not None:
                cross_leg_df = cross_leg_df[valid_mask]
            cross_leg_df = cross_leg_df[cross_leg_df.index.isin(result_df.index)]
            self._cross_legs = {}  # clear after use
        bt = BacktestResult(timeseries=result_df, active_legs_count=active_series, per_leg_pnl=per_leg_df)
        bt.cross_leg_pnl = cross_leg_df  # None for mono, DataFrame for cross
        bt.compute_metrics()
        return bt

    def run_from_optimization(
        self,
        variance_px: pd.DataFrame,
        long_basket: List[Tuple[str, float]],
        short_basket: List[Tuple[str, float]],
        legs: List[DispersionLeg],
        corridor_px: Optional[pd.DataFrame] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> BacktestResult:
        """Run backtest directly from optimizer output.
        
        long_basket/short_basket keys are already the correct series_key
        (Corridor Condition Asset for cross-corridor, variance_asset for mono)
        as produced by _candidate_key() in the optimizer.
        """
        weights = {}
        for key, w in long_basket:
            weights[key] = w
        for key, w in short_basket:
            weights[key] = -w
        
        # Guard: if optimizer returned empty/degenerate basket, return empty result
        if not weights or all(abs(v) < 1e-10 for v in weights.values()):
            empty_df = pd.DataFrame({"Long Leg": [], "Short Leg": [], "Result": []})
            bt = BacktestResult(timeseries=empty_df, active_legs_count=None, per_leg_pnl=pd.DataFrame())
            bt.metrics = {"hit_ratio": 0, "mean_return": 0, "last_value": 0, "max_drawdown": 0, "n_observations": 0}
            return bt

        return self.run(variance_px, legs, weights, corridor_px, start_date, end_date)

    # ─── Internal ────────────────────────────────────────────────────────────────

    def _compute_all_pnl(
        self,
        variance_px: pd.DataFrame,
        legs: List[DispersionLeg],
        corridor_px: Optional[pd.DataFrame],
    ) -> Tuple[np.ndarray, List[str], pd.DatetimeIndex]:
        """Per-leg P&L matrix via the shared builder `compute_leg_pnl_columns`
        (same code path as the optimizer matrix), capturing the per-leg
        mono/cross breakdown for the cross-corridor display.

        Returns: (pnl_matrix [n_rows x n_cols], column_keys, dates_index).
        NaN means "no data for this leg on this date" — the missing-data
        policy handles it downstream.
        """
        pnl_matrix, column_keys, cross_legs = compute_leg_pnl_columns(
            variance_px, corridor_px, legs, self.config, capture_cross_legs=True)
        self._cross_legs = cross_legs
        if column_keys and (~np.isnan(pnl_matrix)).sum() == 0:
            warnings.warn(
                f"[Backtester] pnl_matrix is ALL NaN! shape={pnl_matrix.shape}, "
                f"tickers={column_keys[:5]}..., n_rows={len(variance_px)}")
        return pnl_matrix, column_keys, variance_px.index

    def _apply_fill_zero_policy(
        self,
        pnl_matrix: np.ndarray,
        tickers: List[str],
        weights: Dict[str, float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Original Gaia_PP behavior: fillna(0) then weight.
        Missing legs contribute 0 P&L (no row dropping, no weight redistribution).
        Returns: (long_pnl, short_pnl, active_count, valid_mask)
        """
        long_idx = [i for i, t in enumerate(tickers) if weights.get(t, 0) > 0]
        short_idx = [i for i, t in enumerate(tickers) if weights.get(t, 0) < 0]
        if not long_idx and not short_idx:
            n = pnl_matrix.shape[0]
            return np.zeros(n), np.zeros(n), np.zeros(n), None

        # Fill NaN with 0 — exactly what Gaia_PP does
        pnl_filled = np.nan_to_num(pnl_matrix, nan=0.0)
        long_weights = np.array([abs(weights[tickers[i]]) for i in long_idx]) if long_idx else np.array([])
        short_weights = np.array([abs(weights[tickers[i]]) for i in short_idx]) if short_idx else np.array([])
        n = pnl_filled.shape[0]
        if long_idx and long_weights.sum() > 0:
            long_pnl = pnl_filled[:, long_idx] @ long_weights
        else:
            long_pnl = np.zeros(n)
        if short_idx and short_weights.sum() > 0:
            # In Gaia_PP, short_weights are already negative → dot gives negative result
            # Here short_weights are abs values, so negate
            short_pnl = -(pnl_filled[:, short_idx] @ short_weights)
        else:
            short_pnl = np.zeros(n)
        # No rows dropped — all dates kept (valid_mask = None means "keep all")
        return long_pnl, short_pnl, None, None

    def _apply_drop_policy(
        self,
        pnl_matrix: np.ndarray,
        tickers: List[str],
        weights: Dict[str, float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Drop any row where ANY weighted ticker has NaN.
        Returns: (long_pnl, short_pnl, active_count, valid_row_mask)
        """
        # Build index arrays for long/short
        long_idx = [i for i, t in enumerate(tickers) if weights.get(t, 0) > 0]
        short_idx = [i for i, t in enumerate(tickers) if weights.get(t, 0) < 0]
        all_idx = long_idx + short_idx
        if not all_idx:
            n = pnl_matrix.shape[0]
            return np.zeros(n), np.zeros(n), np.zeros(n), np.ones(n, dtype=bool)

        # Only keep rows where all weighted tickers have valid data
        sub = pnl_matrix[:, all_idx]
        valid_mask = ~np.isnan(sub).any(axis=1)
        # Weighted sums on valid rows
        long_weights = np.array([abs(weights[tickers[i]]) for i in long_idx]) if long_idx else np.array([])
        short_weights = np.array([abs(weights[tickers[i]]) for i in short_idx]) if short_idx else np.array([])
        valid_pnl = pnl_matrix[valid_mask]
        n_valid = valid_pnl.shape[0]
        if long_idx and long_weights.sum() > 0:
            long_pnl = valid_pnl[:, long_idx] @ long_weights
        else:
            long_pnl = np.zeros(n_valid)
        if short_idx and short_weights.sum() > 0:
            short_pnl = -(valid_pnl[:, short_idx] @ short_weights)
        else:
            short_pnl = np.zeros(n_valid)
        # Return None for active_count — computed fresh in run() from pnl_matrix
        return long_pnl, short_pnl, None, valid_mask

    def _apply_adaptive_policy(
        self,
        pnl_matrix: np.ndarray,
        tickers: List[str],
        weights: Dict[str, float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Adaptive reweighting: when a leg has NaN P&L, redistribute its weight
        among available legs on that day. Tracks active count.
        Vectorized numpy — no Python row iteration.
        """
        grace = self.config.reweight_grace_days
        long_idx = [i for i, t in enumerate(tickers) if weights.get(t, 0) > 0]
        short_idx = [i for i, t in enumerate(tickers) if weights.get(t, 0) < 0]
        long_abs_weights = np.array([abs(weights[tickers[i]]) for i in long_idx]) if long_idx else np.array([])
        short_abs_weights = np.array([abs(weights[tickers[i]]) for i in short_idx]) if short_idx else np.array([])
        n_rows = pnl_matrix.shape[0]
        # Precompute NaN mask: True where data is valid
        is_valid = ~np.isnan(pnl_matrix)  # [n_rows x n_tickers]
        all_weighted_idx = long_idx + short_idx
        if not all_weighted_idx:
            return np.zeros(n_rows), np.zeros(n_rows), np.zeros(n_rows), np.ones(n_rows, dtype=bool)

        # In-grace gap days carry the name's last mark: its rolling window is
        # unchanged while it doesn't print, so the carried value is exact.
        # grace=0 returns the matrix untouched (historical behaviour).
        pnl_matrix = carry_pnl_within_grace(pnl_matrix, is_valid, grace)
        # Canonical participation mask (shared with the optimizer): a name is
        # active from its first valid print, and keeps its weight through gaps
        # <= grace days (carrying its last mark on those days). grace=0 == validity.
        active_mask = active_mask_with_grace(is_valid, grace)
        # Compute long leg with adaptive reweighting (redistribution, not normalization)
        long_pnl = np.zeros(n_rows)
        short_pnl = np.zeros(n_rows)
        active_count = np.zeros(n_rows)
        # For cross-corridor: apply REAL adaptive reweight (same as standard mode).
        # Business requirement: constant-vega exposure — when a leg has NaN, redistribute
        # weight to active legs. Use FILL_ZERO policy to get old nan→0 behavior.
        if long_idx:
            long_pnl_cols = pnl_matrix[:, long_idx]
            long_active = active_mask[:, long_idx]   # holds weight (incl. in-grace gap days)
            pnl_filled = np.nan_to_num(long_pnl_cols, nan=0.0)
            total_long_weight = long_abs_weights.sum()
            # Denominator over ACTIVE names: an in-grace name keeps its weight
            # (and contributes its carried mark); only a name beyond grace —
            # or not yet started — has its weight redistributed.
            active_weight_sum = (long_abs_weights[np.newaxis, :] * long_active).sum(axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                scale = np.where(active_weight_sum > 0, total_long_weight / active_weight_sum, 0.0)
            weighted = pnl_filled * long_abs_weights[np.newaxis, :]
            weighted_masked = weighted * long_active
            long_pnl = weighted_masked.sum(axis=1) * scale
            active_count += long_active.sum(axis=1)
        if short_idx:
            short_pnl_cols = pnl_matrix[:, short_idx]
            short_active = active_mask[:, short_idx]
            pnl_filled_s = np.nan_to_num(short_pnl_cols, nan=0.0)
            total_short_weight = short_abs_weights.sum()
            active_weight_sum_s = (short_abs_weights[np.newaxis, :] * short_active).sum(axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                scale_s = np.where(active_weight_sum_s > 0, total_short_weight / active_weight_sum_s, 0.0)
            weighted_s = pnl_filled_s * short_abs_weights[np.newaxis, :]
            weighted_masked_s = weighted_s * short_active
            short_pnl = -(weighted_masked_s.sum(axis=1) * scale_s)
            active_count += short_active.sum(axis=1)
        # All rows are kept (scale=0 already produces pnl=0 for fully-inactive days).
        # Return None as valid_mask so downstream uses the full dates_index.
        return long_pnl, short_pnl, active_count, None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data Loader
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize index to pd.DatetimeIndex (handles date, str, or Timestamp)."""
    if df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df

# ── Bloomberg data caching ──────────────────────────────────────────────────
# Module-level cache: avoids re-fetching on Streamlit reruns
_bbg_cache: dict = {}
_bbg_cache_ts: dict = {}

def _cached_bdh(tickers: list, field: str, start: str, end: str):
    """Bloomberg fetch with Streamlit-aware caching. Falls back to in-memory TTL cache."""
    # Try Streamlit cache first (survives reruns, 10 min TTL)
    try:
        import streamlit as st  # noqa: F401 — presence check only
    except ImportError:
        st = None
    if st is not None:
        # Real fetch errors must PROPAGATE (fail loud, no silent double-fetch).
        return _st_cached_bdh(tuple(sorted(tickers)), field, start, end)

    # Fallback: in-memory dict cache (5 min TTL)
    import time as _time
    key = f"{sorted(tickers)}|{field}|{start}|{end}"
    now = _time.time()
    if key in _bbg_cache and (now - _bbg_cache_ts.get(key, 0)) < 300:
        return _bbg_cache[key]

    from xbbg import blp
    df = blp.bdh(tickers, field, start, end)
    _bbg_cache[key] = df
    _bbg_cache_ts[key] = now
    return df

_st_fetch = None   # cached-fetch singleton — decorate ONCE, not per call


def _st_cached_bdh(tickers_tuple: tuple, field: str, start: str, end: str):
    """Streamlit-cached Bloomberg fetch (10 min TTL, persists across reruns)."""
    global _st_fetch
    if _st_fetch is None:
        import streamlit as st

        @st.cache_data(ttl=600, show_spinner=False)
        def _fetch(tickers_key, field, start, end):
            from xbbg import blp
            return blp.bdh(list(tickers_key), field, start, end)

        _st_fetch = _fetch
    return _st_fetch(tickers_tuple, field, start, end)

class DispersionDataLoader:
    """
    Fetches price data from Bloomberg and computes per-leg swap metrics.
    Usage:
        loader = DispersionDataLoader(config)
        data = loader.load(basket)
        # data['variance_px'], data['corridor_px'], data['legs']
    """

    def __init__(self, config: SwapConfig, logger=None):
        self.config = config
        self._calc = SwapCalculator(config)
        self._logger = logger or (lambda level, msg: None)

    def load(self, basket: BasketInput, end_date: Optional[date] = None,
             compute_leg_metrics: bool = True) -> Dict:
        """
        Load all data needed for optimization and backtesting.
        Args:
            end_date: optional cutoff for the per-leg metadata stats (the
                optimizer passes its window end so leg.metrics never embeds
                post-cutoff data). Does not truncate the returned price frames.
            compute_leg_metrics: set False to skip the per-leg rolling-kernel
                metadata pass (~doubles load time for large universes) when
                leg.metrics is not consumed.
        Returns:
            {
                'variance_px': pd.DataFrame (Variance Asset prices, union calendar),
                'corridor_px': pd.DataFrame or None (Corridor Condition Asset prices, cross-corridor only),
                'legs': List[DispersionLeg] (with metrics populated),
                'n_long': int,
            }
        """
        start = self._start_date()
        all_legs = basket.all_candidates
        if self.config.is_cross_corridor:
            variance_px, corridor_px = self._fetch_cross_corridor(all_legs, start)
        else:
            variance_px = self._fetch_standard(all_legs, start)
            corridor_px = None
        # Compute per-leg metrics
        if compute_leg_metrics:
            self._compute_metrics(all_legs, variance_px, corridor_px, end_date=end_date)
        return {
            "variance_px": variance_px,
            "corridor_px": corridor_px,
            "legs": all_legs,
            "n_long": len(basket.long_candidates),
            "long_legs": all_legs[:len(basket.long_candidates)],
            "short_legs": all_legs[len(basket.long_candidates):],
        }

    # def _start_date(self) -> str:
    #     dt = date.today() - relativedelta(years=self.config.lookback_years + 2)
    #     return dt.strftime("%m/%d/%Y")
    #
    def _start_date(self) -> str:
        if self.config.start_date is not None:
            # Fetch extra history to initialise the rolling P&L at the requested date
            buffer_days = int(self.config.n_exp * 365 / 252) + 30
            dt = self.config.start_date - timedelta(days=buffer_days)
        else:
            dt = date.today() - relativedelta(
                years=self.config.lookback_years + 2
            )

        return pd.Timestamp(dt).strftime("%m/%d/%Y")


    def _fetch_standard(self, legs: List[DispersionLeg], start: str) -> pd.DataFrame:
        field = "TOT_RETURN_INDEX_GROSS_DVDS" if self.config.adj_divs else "PX_LAST"
        all_tickers = [s.variance_asset for s in legs]
        today_str = date.today().strftime("%m/%d/%Y")
        self._logger("INFO", f"Fetching {len(all_tickers)} tickers from Bloomberg ({start} → today)...")
        try:
            df = _cached_bdh(all_tickers, field, start, today_str)
        except Exception as e:
            # Fail LOUD: an infra failure must never look like an empty universe.
            raise RuntimeError(
                f"Bloomberg fetch failed for {len(all_tickers)} tickers "
                f"({field}, {start} → today): {e}") from e

        if df is None or df.empty:
            self._logger("ERROR", f"Bloomberg returned empty DataFrame for {len(all_tickers)} tickers")
            return pd.DataFrame()

        # xbbg returns MultiIndex columns (ticker, field) — flatten
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # Strip whitespace from column names
        df.columns = [str(c).strip() for c in df.columns]
        # Log diagnostic: which requested tickers are missing from result
        returned_cols = set(df.columns)
        missing = [t for t in all_tickers if t not in returned_cols]
        if missing:
            self._logger("WARNING", f"{len(missing)}/{len(all_tickers)} tickers not in Bloomberg response: {missing[:5]}...")
        df = _ensure_datetime_index(df)
        return df.sort_index()

    def _fetch_cross_corridor(self, legs: List[DispersionLeg], start: str):
        var_tickers = list({s.variance_asset for s in legs})
        corr_tickers = list({s.corridor_condition_asset for s in legs if s.corridor_condition_asset})
        field = "TOT_RETURN_INDEX_GROSS_DVDS" if self.config.adj_divs else "PX_LAST"
        today_str = date.today().strftime("%m/%d/%Y")
        # Fetch variance asset prices (batch)
        try:
            variance_px = _cached_bdh(var_tickers, field, start, today_str)
        except Exception as e:
            # Fail LOUD: an infra failure must never look like an empty universe.
            raise RuntimeError(
                f"Bloomberg fetch failed for {len(var_tickers)} Variance Asset "
                f"tickers ({field}): {e}") from e

        if variance_px is None or variance_px.empty:
            self._logger("ERROR", f"Bloomberg returned empty data for {len(var_tickers)} variance asset tickers")
            return pd.DataFrame(), pd.DataFrame()

        if isinstance(variance_px.columns, pd.MultiIndex):
            variance_px.columns = variance_px.columns.get_level_values(0)
        variance_px.columns = [str(c).strip() for c in variance_px.columns]
        variance_px = _ensure_datetime_index(variance_px)
        # Log missing variance assets
        missing_var = [t for t in var_tickers if t not in variance_px.columns]
        if missing_var:
            self._logger("WARNING", f"{len(missing_var)}/{len(var_tickers)} variance assets missing from Bloomberg: {missing_var[:5]}")
        # Fetch corridor condition asset prices (batch) — always PX_LAST
        corridor_px = pd.DataFrame()
        if corr_tickers:
            try:
                corridor_px = _cached_bdh(corr_tickers, "PX_LAST", start, today_str)
            except Exception as e:
                # Fail LOUD — a missing corridor feed would otherwise drop every leg.
                raise RuntimeError(
                    f"Bloomberg fetch failed for {len(corr_tickers)} Corridor "
                    f"Condition Asset tickers: {e}") from e
            if corridor_px is not None and not corridor_px.empty:
                if isinstance(corridor_px.columns, pd.MultiIndex):
                    corridor_px.columns = corridor_px.columns.get_level_values(0)
                corridor_px.columns = [str(c).strip() for c in corridor_px.columns]
                corridor_px = _ensure_datetime_index(corridor_px)
                missing_corr = [t for t in corr_tickers if t not in corridor_px.columns]
                if missing_corr:
                    self._logger("WARNING", f"Corridor condition assets missing from Bloomberg: {missing_corr}")
        # Align to same dates
        if not corridor_px.empty:
            all_dates = variance_px.index.union(corridor_px.index)
            variance_px = variance_px.reindex(all_dates)
            corridor_px = corridor_px.reindex(all_dates)
        return variance_px.sort_index(), corridor_px.sort_index() if not corridor_px.empty else pd.DataFrame()

    def _compute_metrics(
        self, legs: List[DispersionLeg], variance_px: pd.DataFrame, corridor_px: Optional[pd.DataFrame],
        end_date: Optional[date] = None,
    ):
        """Populate each leg's .metrics dict with backtest stats.

        Prices are sliced at ``end_date`` (when given) so stats never embed
        post-cutoff data.  Cross-corridor pairs are aligned on their SHARED
        calendar (joint dropna — independent dropna would pair the variance
        price of one date with the corridor price of another).
        """
        if end_date is not None:
            cutoff = pd.Timestamp(end_date).normalize()
            variance_px = variance_px[variance_px.index.normalize() <= cutoff]
            if corridor_px is not None:
                corridor_px = corridor_px[corridor_px.index.normalize() <= cutoff]
        for leg in legs:
            if leg.variance_asset not in variance_px.columns:
                leg.metrics = {"last_value": 0, "avg_5y": 0, "avg_3y": 0, "hit_ratio": 50, "max_drawdown": 0}
                continue

            # Determine corridor prices for cross-corridor — aligned jointly
            corridor_prices = None
            if (self.config.is_cross_corridor and leg.corridor_condition_asset
                    and corridor_px is not None
                    and leg.corridor_condition_asset in corridor_px.columns):
                pair = pd.concat(
                    [variance_px[leg.variance_asset],
                     corridor_px[leg.corridor_condition_asset]], axis=1).dropna()
                prices = pair.iloc[:, 0].values.astype(np.float64)
                corridor_prices = pair.iloc[:, 1].values.astype(np.float64)
            else:
                prices = variance_px[leg.variance_asset].dropna().values.astype(np.float64)
            if len(prices) < self.config.n_exp + 1:
                leg.metrics = {"last_value": 0, "avg_5y": 0, "avg_3y": 0, "hit_ratio": 50, "max_drawdown": 0}
                continue

            pnl = self._calc.compute(prices, leg.strike_mono_var_swap, corridor_prices)
            valid = pnl[~np.isnan(pnl)]
            if len(valid) == 0:
                leg.metrics = {"last_value": 0, "avg_5y": 0, "avg_3y": 0, "hit_ratio": 50, "max_drawdown": 0}
                continue

            n_3y = min(len(valid), 252 * 3)
            n_5y = min(len(valid), 252 * 5)
            cumsum = np.cumsum(valid)
            leg.metrics = {
                "last_value": float(valid[-1]),
                "avg_5y": float(valid[-n_5y:].mean()),
                "avg_3y": float(valid[-n_3y:].mean()),
                "hit_ratio": float((valid > 0).sum() / max(1, (valid != 0).sum()) * 100),
                "max_drawdown": float((cumsum - np.maximum.accumulate(cumsum)).min()),
            }
