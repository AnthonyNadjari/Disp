# ═══════════════════════════════════════════════════════════════════════════════
# UNIT CONVENTIONS
# ═══════════════════════════════════════════════════════════════════════════════
#
# PUBLIC API (DataFrame columns):
#   - Columns with "(%)" suffix → PERCENTAGE (21.4 = 21.4%)
#   - 'Min Weight', 'Max Weight' → PERCENTAGE (1.0 = 1%, 60.0 = 60%)
#
# INTERNAL (DispersionLeg objects):
#   - DispersionLeg.strike_mono_var_swap, .strike_cross_corridor → DECIMAL (0.214 = 21.4%)
#   - DispersionLeg.weight → DECIMAL (0.05 = 5%)
#   - DispersionLeg.min_weight, .max_weight → DECIMAL (0.01 = 1%)
#     (optimizer sniffs unit; always pass percentage to match existing behavior)
#
# INTERNAL (constraints / results):
#   - OptimizationConstraints.max_net_strike → DECIMAL (0.15 = 15%)
#   - OptimizationResult.net_strike → DECIMAL
#   - OptimizationResult.long_basket weights → DECIMAL
#
# INTERNAL (PNL matrix):
#   - Values are percentage-scaled (multiplied by 100 in builder)
#   - A value of 5.0 means 5% payoff
#
# RESULTS (user-facing):
#   - BacktestResult.hit_ratio → PERCENTAGE (65.0 = 65%)
#   - BacktestResult.max_drawdown → PERCENTAGE, NEGATIVE (-12.5 = 12.5% drop)
#   - SolveResult.results_df → pre-formatted strings ("21.40%")
#
# RULE: All conversions happen HERE in _api.py. Internal modules never convert.
# ═══════════════════════════════════════════════════════════════════════════════
"""
Public entry points for the dispersion engine.

    from functions.dispersion import solve, price, optimize, backtest
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from datetime import date
from typing import Callable, Dict, List, Optional, Tuple

from functions.dispersion.models import (
    DispersionConfig,
    OptimizationConstraints,
    OptimizationResult,
    BacktestResult,
    SolveResult,
    PriceResult,
    MissingDataPolicy,
    ProductType,
    DispersionLeg,
    BasketInput,
    SwapConfig,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG CONVERSION
# ═══════════════════════════════════════════════════════════════════════════════

def _to_swap_config(cfg: DispersionConfig, **overrides) -> SwapConfig:
    """
    Convert public DispersionConfig to internal SwapConfig.

    This is the ONLY place where DispersionConfig touches SwapConfig.
    Internal modules (_backtester, _optimizer, _data_loader) continue
    to accept SwapConfig unchanged until they are migrated.
    """
    return SwapConfig(
        product_type=cfg.product_type,
        cross_corridor=cfg.cross_corridor,
        n_exp=cfg.n_exp,
        local_cap=cfg.local_cap,
        barrier_down=cfg.barrier_down,
        barrier_up=cfg.barrier_up,
        adj_divs=cfg.adj_divs,
        missing_data_policy=cfg.missing_data_policy,
        lookback_years=cfg.lookback_years,
        global_cap=cfg.global_cap,
        global_floor=cfg.global_floor,
        capped=cfg.is_capped,
        **overrides,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_columns(df: pd.DataFrame, required: List[str], fn_name: str):
    """Raise ValueError with actionable message if columns missing."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{fn_name}() expects a pandas DataFrame, got {type(df).__name__}")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{fn_name}() requires columns {missing} in df.\n"
            f"Got: {list(df.columns)}"
        )
    if df.empty:
        raise ValueError(f"{fn_name}() received an empty DataFrame.")


def _normalize_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Validate canonical column names. Reject legacy names with clear error."""
    _LEGACY = {
        'Tickers': 'Variance Asset',
        'Index': 'Corridor Condition Asset',
        'Strike Mono': 'Strike Mono Var Swap (%)',
        'Strike Cross Corr': 'Strike Cross Corridor (%)',
        'Weights': 'Weight (%)',
        'Weights (%)': 'Weight (%)',
        'Strikes': 'Strike Mono Var Swap (%)',
    }
    found = [old for old in _LEGACY if old in df.columns]
    if found:
        hint = ", ".join(f"'{old}' → '{_LEGACY[old]}'" for old in found)
        raise ValueError(
            f"Legacy column names detected: {hint}. "
            f"Use canonical names: Variance Asset, Corridor Condition Asset, "
            f"Strike Cross Corridor (%), Strike Mono Var Swap (%), Weight (%), "
            f"Min Weight, Max Weight, Side."
        )
    return df


def _warn_if_decimal_strikes(df: pd.DataFrame, col: str, fn_name: str):
    """Emit warning if strikes appear to be decimal instead of percentage."""
    if col not in df.columns:
        return
    values = pd.to_numeric(df[col], errors='coerce').dropna()
    if len(values) > 0 and values.max() < 1.0:
        warnings.warn(
            f"{fn_name}(): '{col}' max value is {values.max():.4f}. "
            f"Expected percentage (e.g. 21.4 not 0.214). Did you pass decimals?"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# DATAFRAME → STOCK CONVERSION
# ═══════════════════════════════════════════════════════════════════════════════

def _df_to_legs(df: pd.DataFrame, is_cross_corridor: bool) -> List[DispersionLeg]:
    """Parse a tickers DataFrame into DispersionLeg objects with unit conversions."""
    legs = []
    for _, row in df.iterrows():
        var_asset = str(row['Variance Asset']).strip()

        # Strike: DataFrame is percentage → DispersionLeg is decimal
        # Long leg uses Strike Mono Var Swap (%)
        strike_pct = float(row.get('Strike Mono Var Swap (%)', 0))
        strike = strike_pct / 100.0

        # Corridor Condition Asset / Strike Cross Corridor
        corr_asset = None
        cross_strike = None
        if is_cross_corridor:
            corr_asset = str(row.get('Corridor Condition Asset', '')).strip() or None
            cross_strike_pct = float(row.get('Strike Cross Corridor (%)'))
            if pd.isna(cross_strike_pct):
                raise ValueError(f"Cross-corridor backtest requires 'Strike Cross Corridor (%)' for '{var_asset}'.")
            cross_strike = cross_strike_pct / 100.0

        # Min/Max weight: UI provides percentage points (e.g. 1.5 = 1.5%)
        # DispersionLeg expects decimal (0.015 = 1.5%)
        min_w = float(row.get('Min Weight', 1.0)) / 100.0
        max_w = float(row.get('Max Weight', 60.0)) / 100.0
        sector = row.get('Sector', None)

        legs.append(DispersionLeg(
            variance_asset=var_asset,
            strike_mono_var_swap=strike,
            corridor_condition_asset=corr_asset,
            strike_cross_corridor=cross_strike,
            sector=str(sector) if sector and str(sector).strip() else None,
            min_weight=min_w,
            max_weight=max_w,
        ))
    return legs


def _df_to_backtest_inputs(df: pd.DataFrame, is_cross_corridor: bool) -> Tuple[List[DispersionLeg], Dict[str, float]]:
    """Parse a backtest DataFrame into DispersionLeg objects + weights dict."""
    legs = []
    weights = {}
    for idx, row in df.iterrows():
        var_asset = str(row['Variance Asset']).strip()
        weight_pct = float(row['Weight (%)'])
        side = str(row.get('Side', 'long')).strip().lower()

        # Strike: percentage → decimal
        strike_pct = float(row.get('Strike Mono Var Swap (%)', 0))
        strike = strike_pct / 100.0

        # Corridor Condition Asset / Strike Cross Corridor
        corr_asset = None
        cross_strike = None
        if is_cross_corridor:
            corr_asset = str(row.get('Corridor Condition Asset', '')).strip() or None
            cross_strike_pct = float(row.get('Strike Cross Corridor (%)'))
            if pd.isna(cross_strike_pct):
                raise ValueError(f"Cross-corridor backtest requires 'Strike Cross Corridor (%)' for '{var_asset}'.")
            cross_strike = cross_strike_pct / 100.0

        # Weight: percentage → decimal, signed by side
        signed_weight = weight_pct / 100.0 if side == 'long' else -(weight_pct / 100.0)

        # For cross-corridor, weights must be keyed by Corridor Condition Asset
        # to match pnl_column_keys and avoid zero-weight P&L when Variance Asset is shared
        if is_cross_corridor and corr_asset:
            weight_key = corr_asset
        else:
            weight_key = var_asset

        weights[weight_key] = signed_weight

        leg = DispersionLeg(
            variance_asset=var_asset,
            strike_mono_var_swap=strike,
            corridor_condition_asset=corr_asset,
            strike_cross_corridor=cross_strike,
        )
        legs.append(leg)

    return legs, weights


# ═══════════════════════════════════════════════════════════════════════════════
# PNL MATRIX BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def _build_pnl_matrix(
    price_data: pd.DataFrame,
    index_data: Optional[pd.DataFrame],
    legs: List[DispersionLeg],
    cfg: DispersionConfig,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Build PNL matrix from prices using numba kernels.
    Returns (pnl_matrix, column_map).
    """
    from functions.dispersion._backtester import (
        SwapCalculator, _rolling_pnl_corridor, _rolling_pnl_volswap,
    )

    internal_cfg = _to_swap_config(cfg)

    # Build column keys: for cross-corridor, use Corridor Condition Asset to avoid duplicates
    # IMPORTANT: leg_map must be keyed by the UNIQUE per-leg key (not variance_asset, which is
    # the same "SPX Index" for all cross-corridor legs and would collapse the map to 1 entry).
    leg_map = {}
    column_keys = []
    for s in legs:
        if s.variance_asset not in price_data.columns:
            continue
        if cfg.cross_corridor and s.corridor_condition_asset:
            map_key = s.corridor_condition_asset
        else:
            map_key = s.variance_asset
        leg_map[map_key] = s
        column_keys.append(map_key)

    col_map = {t: i for i, t in enumerate(column_keys)}
    n_rows = len(price_data)
    pnl_matrix = np.full((n_rows, len(column_keys)), np.nan)

    calc = SwapCalculator(internal_cfg)
    if index_data is not None and not index_data.empty:
        index_data = index_data.loc[index_data.index.isin(price_data.index)]

    for i, key in enumerate(leg_map.keys()):
        leg = leg_map[key]
        prices = price_data[leg.variance_asset].values.astype(np.float64)

        if cfg.cross_corridor and leg.corridor_condition_asset and index_data is not None and leg.corridor_condition_asset in index_data.columns:
            index_prices = index_data[leg.corridor_condition_asset].values.astype(np.float64)
            # Cross-corridor backtest:
            #   Mono leg:   corridor=Corridor Condition Asset, variance=Corridor Condition Asset
            #   Cross leg:  corridor=Corridor Condition Asset, variance=Variance Asset
            if cfg.is_vol_swap:
                # Mono leg: realized_vol on Corridor Condition Asset
                pnl_long = _rolling_pnl_volswap(index_prices, leg.strike_mono_var_swap, cfg.n_exp, cfg.local_cap)
                # Cross leg: realized_vol on Variance Asset
                cross_strike = leg.strike_cross_corridor if leg.strike_cross_corridor else leg.strike_mono_var_swap
                pnl_short = _rolling_pnl_volswap(prices, cross_strike, cfg.n_exp, cfg.local_cap)
            else:
                # Mono leg: variance=Corridor Condition Asset, corridor=Corridor Condition Asset
                pnl_long = _rolling_pnl_corridor(
                    index_prices, index_prices, leg.strike_mono_var_swap,
                    cfg.barrier_up, cfg.barrier_down, cfg.n_exp, cfg.local_cap)
                # Cross leg: variance=Variance Asset, corridor=Corridor Condition Asset
                cross_strike = leg.strike_cross_corridor if leg.strike_cross_corridor else leg.strike_mono_var_swap
                pnl_short = _rolling_pnl_corridor(
                    prices, index_prices, cross_strike,
                    cfg.barrier_up, cfg.barrier_down, cfg.n_exp, cfg.local_cap)
            n = min(len(pnl_long), len(pnl_short), n_rows)
            pnl_matrix[:n, i] = (pnl_long[:n] - pnl_short[:n]) * 100
        else:
            pnl = calc.compute(prices, leg.strike_mono_var_swap, corridor_prices=None)
            pnl_matrix[:len(pnl), i] = pnl * 100

    return pnl_matrix, col_map


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SOLVE
# ═══════════════════════════════════════════════════════════════════════════════

def solve(
    df: pd.DataFrame,
    config: DispersionConfig,
    last_obs_date: date,
    *,
    strike_date: date = None,
    eqeq_lambda: float = 0.10,
    correl_floor: float = 0.0,
    eqfx_shift: float = -0.05,
    vol_mode: str = "ATMF",
    correl_input_method: str = "Global Parameters",
    model_name: str = None,
    use_lsv: bool = False,
    lsv_params: pd.DataFrame = None,
    lsv_correl_bump: float = 0.0,
    lsv_correl_bump_style: str = "Relative",
    use_lcm: bool = False,
    lcm_properties: dict = None,
    progress_callback: Callable[[dict], None] = None,
) -> SolveResult:
    """
    Compute fair-value strikes for a set of tickers via portal pricing.

    Parameters
    ----------
    df : DataFrame
        Required columns: Variance Asset, Corridor Condition Asset, Currency
        Optional: Correlation (percentage, e.g. 46.26)
    config : DispersionConfig
        Product definition (barriers, cap, product_type).
        Note: config.n_exp is NOT used by solve/price — the product schedule
        is fully defined by strike_date → last_obs_date + exchange calendars.
    last_obs_date : date
        Product maturity (last observation date). Together with strike_date,
        defines the observation schedule via calendar intersection.
    strike_date : date, optional
        Trade inception (when the strike is set). Defaults to today.
    eqeq_lambda, correl_floor, eqfx_shift : float
        Model parameters (defaults = desk standard).
    vol_mode : str
        "ATMF" or "ATMF+ATMS"
    correl_input_method : str
        "Global Parameters" or "Individual Correlations"
    model_name : str, optional
        Override pricing model name.
    progress_callback : callable, optional
        Called with progress dict.

    Returns
    -------
    SolveResult
        .results_df, .success, .failed_tickers
    """
    df = _normalize_df_columns(df)
    _validate_columns(df, ['Variance Asset', 'Corridor Condition Asset', 'Currency'], 'solve')
    from functions.dispersion._pricing import PricingEngine, PricingConfig

    # Internally, PricingEngine expects 'Tickers' column
    engine_df = df.rename(columns={'Variance Asset': 'Tickers'})

    pricing_cfg = PricingConfig(
        strike_date=strike_date or date.today(),
        last_obs_date=last_obs_date,
        uvar=config.barrier_up,
        dvar=config.barrier_down,
        is_solve=True,
        is_capped=config.is_capped,
        is_cross_corridor=config.cross_corridor,
        eqeq_lambda=eqeq_lambda,
        correl_floor=correl_floor,
        eqfx_shift=eqfx_shift,
        vol_mode=vol_mode,
        correl_input_method=correl_input_method,
        compute_zero_strike=True,
        model_name=model_name,
        use_lsv_cross_ev=use_lsv,
        lsv_params=lsv_params,
        lsv_correl_bump=lsv_correl_bump,
        lsv_correl_bump_style=lsv_correl_bump_style,
        lcm_params={'enabled': use_lcm, 'lcm_properties': lcm_properties} if use_lcm else None,
    )
    result = PricingEngine(pricing_cfg).run(tickers_df=engine_df, progress_callback=progress_callback)

    return SolveResult(
        results_df=result.results_df if result.results_df is not None else pd.DataFrame(),
        success=result.success,
        failed_tickers=result.failed_tickers,
        error=result.error,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PRICE
# ═══════════════════════════════════════════════════════════════════════════════

def price(
    df: pd.DataFrame,
    config: DispersionConfig,
    last_obs_date: date,
    *,
    strike_date: date = None,
    eqeq_lambda: float = 0.10,
    correl_floor: float = 0.0,
    eqfx_shift: float = -0.05,
    vol_mode: str = "ATMF",
    correl_input_method: str = "Global Parameters",
    model_name: str = None,
    use_lsv: bool = False,
    lsv_params: pd.DataFrame = None,
    lsv_correl_bump: float = 0.0,
    lsv_correl_bump_style: str = "Relative",
    use_lcm: bool = False,
    lcm_properties: dict = None,
    progress_callback: Callable[[dict], None] = None,
) -> PriceResult:
    """
    Price given strikes for a set of variance assets via portal pricing.

    Parameters
    ----------
    df : DataFrame
        Required columns: Variance Asset, Corridor Condition Asset, Currency,
                          Strike Cross Corridor (%), Strike Mono Var Swap (%)
    (other params same as solve)

    Returns
    -------
    PriceResult
        .results_df, .success, .failed_tickers
    """
    df = _normalize_df_columns(df)
    _validate_columns(df, ['Variance Asset', 'Corridor Condition Asset', 'Currency'], 'price')

    from functions.dispersion._pricing import PricingEngine, PricingConfig

    # Internally, PricingEngine expects 'Tickers' column
    engine_df = df.rename(columns={'Variance Asset': 'Tickers'})

    pricing_cfg = PricingConfig(
        strike_date=strike_date or date.today(),
        last_obs_date=last_obs_date,
        uvar=config.barrier_up,
        dvar=config.barrier_down,
        is_solve=False,
        is_capped=config.is_capped,
        is_cross_corridor=config.cross_corridor,
        eqeq_lambda=eqeq_lambda,
        correl_floor=correl_floor,
        eqfx_shift=eqfx_shift,
        vol_mode=vol_mode,
        correl_input_method=correl_input_method,
        model_name=model_name,
        use_lsv_cross_ev=use_lsv,
        lsv_params=lsv_params,
        lsv_correl_bump=lsv_correl_bump,
        lsv_correl_bump_style=lsv_correl_bump_style,
        lcm_params={'enabled': use_lcm, 'lcm_properties': lcm_properties} if use_lcm else None,
    )
    result = PricingEngine(pricing_cfg).run(tickers_df=engine_df, progress_callback=progress_callback)

    return PriceResult(
        results_df=result.results_df if result.results_df is not None else pd.DataFrame(),
        success=result.success,
        failed_tickers=result.failed_tickers,
        error=result.error,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. OPTIMIZE
# ═══════════════════════════════════════════════════════════════════════════════

def optimize(
    long_df: pd.DataFrame,
    config: DispersionConfig,
    constraints: OptimizationConstraints,
    *,
    short_df: pd.DataFrame = None,
    score_weights: Dict[str, float] = None,
    start_date: date = None,
    end_date: date = None,
    filter_zero_hr: bool = False,
    progress_callback: Callable[[int, int, float], None] = None,
    run_milp: bool = False,
    bisect_in_ga: bool = False,
    seed: int = 0,
    forced_tickers: List[str] = None,
    excluded_tickers: List[str] = None,
    save_bundle_path: str = None,
) -> OptimizationResult:
    """
    Find optimal basket via genetic algorithm.

    Internally: loads Bloomberg prices → builds PNL matrix → runs GA → backtests winner.

    Parameters
    ----------
    long_df : DataFrame
        Columns: Variance Asset, Strike Mono Var Swap (%), Corridor Condition Asset (if cross_corridor),
                 Sector (optional), Min Weight (optional), Max Weight (optional)
    config : DispersionConfig
    constraints : OptimizationConstraints
    short_df : DataFrame, optional
        Same format as long_df. None = long-only.
    score_weights : dict, optional
        e.g. {'last_carry': 40, 'mean_payoff': 30, 'hit_ratio': 30}.
        Also accepts the optional metrics 'max_drawdown', 'cvar_5',
        'sharpe_payoff' and 'weighted_strike' (strike-minimisation objective;
        the max_net_strike hard constraint stays active independently).
        None = equal weights on the 4 core metrics.
    start_date, end_date : date, optional
        Optimization window. Default: lookback_years from today.
    filter_zero_hr : bool
        Remove candidates with 0% hit ratio.  Forced tickers are never removed
        by this filter.
    progress_callback : callable, optional
    forced_tickers : list of str, optional
        Long-leg names that MUST be in the final basket (presence only —
        their weights stay governed by the usual Min/Max Weight inputs).
        Matched case-insensitively against 'Variance Asset' and, in
        cross-corridor mode, 'Corridor Condition Asset'.
    excluded_tickers : list of str, optional
        Names removed from the candidate universe (long and short) before
        optimization.  Must not overlap forced_tickers.
    save_bundle_path : str, optional
        Directory path: write a replayable run bundle (P&L matrix as parquet
        + JSON with legs/constraints/weights/seed/result) after the GA runs.
        See :mod:`functions.dispersion.run_bundle`.  Only written when the
        GA actually ran (not on empty-universe early exits).

    Returns
    -------
    OptimizationResult
        .long_basket, .short_basket, .net_strike, .score, .converged, .backtest
    """
    # ── Forced / excluded: cheap validation before any data load ──
    def _norm_ticker_set(lst):
        return {str(t).strip().casefold() for t in (lst or []) if str(t).strip()}

    _forced_set = _norm_ticker_set(forced_tickers)
    _excluded_set = _norm_ticker_set(excluded_tickers)
    _overlap = _forced_set & _excluded_set
    if _overlap:
        raise ValueError(f"Tickers cannot be both forced and excluded: {sorted(_overlap)}")
    if len(_forced_set) > constraints.max_stocks_long:
        raise ValueError(
            f"{len(_forced_set)} forced tickers > max_stocks_long="
            f"{constraints.max_stocks_long}. Raise max_stocks_long or force fewer names."
        )

    long_df = _normalize_df_columns(long_df)
    required = ['Variance Asset', 'Strike Mono Var Swap (%)']
    if config.cross_corridor:
        required.append('Strike Cross Corridor (%)')
        required.append('Corridor Condition Asset')
    _validate_columns(long_df, required, 'optimize')
    _warn_if_decimal_strikes(long_df, 'Strike Mono Var Swap (%)', 'optimize')

    from functions.dispersion._backtester import (
        DispersionDataLoader, DispersionBacktester,
        _rolling_pnl_corridor, _rolling_pnl_volswap, SwapCalculator,
    )
    from functions.dispersion._optimizer import DispersionOptimizer
    from functions.dispersion.scoring import MetricWeights

    # ── Step 1: Parse DataFrames → DispersionLeg objects ──
    long_legs = _df_to_legs(long_df, config.cross_corridor)
    if short_df is not None and not short_df.empty:
        short_df = _normalize_df_columns(short_df)
        short_legs = _df_to_legs(short_df, config.cross_corridor)
    else:
        short_legs = []

    # ── Step 2: Load Bloomberg data ──
    internal_cfg = _to_swap_config(config, start_date=start_date, end_date=end_date)
    loader = DispersionDataLoader(internal_cfg)
    basket = BasketInput(long_candidates=long_legs, short_candidates=short_legs)
    data = loader.load(basket)

    if data["price_data"].empty:
        empty_ts = pd.DataFrame({"Long Leg": [], "Short Leg": [], "Result": []})
        return OptimizationResult(
            long_basket=[], short_basket=[],
            long_strike_weighted=0.0, short_strike_weighted=0.0,
            net_strike=0.0, score=-np.inf,
            generations_run=0, converged=False,
            backtest=BacktestResult(timeseries=empty_ts),
        )

    # ── Step 3: Build PNL matrix ──
    price_data_all = data["price_data"]
    index_data_all = data.get("index_data")
    legs = data["legs"]
    long_legs_loaded = data["long_legs"]
    short_legs_loaded = data["short_legs"]

    # WARM-UP FIX: Build PnL from FULL price history (same as backtester uses),
    # then slice the resulting matrix to [start_date, end_date].
    # This ensures the rolling kernel has n_exp warm-up rows BEFORE start_date,
    # so the optimizer's matrix row 0 matches the backtest curve row 0 exactly.
    price_data_for_build = price_data_all
    index_data_for_build = index_data_all
    if not start_date and not end_date:
        # No explicit dates: use lookback window including warm-up buffer
        lookback_days = int(config.lookback_years * 252) + config.n_exp
        if len(price_data_for_build) > lookback_days:
            price_data_for_build = price_data_for_build.iloc[-lookback_days:]
            if index_data_for_build is not None and not index_data_for_build.empty:
                index_data_for_build = index_data_for_build.iloc[-lookback_days:]

    pnl_matrix_full, col_map = _build_pnl_matrix(price_data_for_build, index_data_for_build, legs, config)

    # Slice to [start_date, end_date] — rows in pnl_matrix_full correspond 1:1 to price_data_for_build.index
    _build_index = price_data_for_build.index
    if start_date:
        _start_idx = _build_index.searchsorted(pd.Timestamp(start_date), side='left')
    else:
        _start_idx = 0
    if end_date:
        _end_idx = _build_index.searchsorted(pd.Timestamp(end_date), side='right')
    else:
        _end_idx = len(_build_index)
    pnl_matrix = pnl_matrix_full[_start_idx:_end_idx]

    # Also keep the date index for the optimizer window (used later for NUMERIC-CHECK alignment)
    _optimizer_dates = _build_index[_start_idx:_end_idx]
    # price_data = sliced prices (used only for candidate filtering, not PnL computation)
    price_data = price_data_for_build.iloc[_start_idx:_end_idx]
    index_data = index_data_for_build.iloc[_start_idx:_end_idx] if (index_data_for_build is not None and not index_data_for_build.empty) else index_data_for_build

    # ── Step 4: Filter candidates ──
    valid_tickers = set(col_map.keys())
    # For cross-corridor, use index (Corridor Condition Asset) as the key
    if config.cross_corridor:
        long_valid = [s for s in long_legs_loaded if s.corridor_condition_asset in valid_tickers]
        short_valid = [s for s in short_legs_loaded if s.corridor_condition_asset in valid_tickers]
        missing = [s for s in long_legs_loaded if s.corridor_condition_asset not in valid_tickers]
    else:
        long_valid = [s for s in long_legs_loaded if s.variance_asset in valid_tickers]
        short_valid = [s for s in short_legs_loaded if s.variance_asset in valid_tickers]
        missing = [s for s in long_legs_loaded if s.variance_asset not in valid_tickers]

    def _leg_keys(s):
        keys = {str(s.variance_asset).strip().casefold()}
        if getattr(s, "corridor_condition_asset", None):
            keys.add(str(s.corridor_condition_asset).strip().casefold())
        return keys

    # ── Exclusion: drop excluded names from both legs ──
    if _excluded_set:
        long_valid = [s for s in long_valid if not (_leg_keys(s) & _excluded_set)]
        short_valid = [s for s in short_valid if not (_leg_keys(s) & _excluded_set)]

    # Snapshot before the zero-HR filter so forced names can be restored
    _pre_hr_long = list(long_valid)

    if filter_zero_hr:
        long_valid, short_valid = _filter_by_hit_ratio(
            long_valid, short_valid, pnl_matrix, col_map, constraints)

    # ── Forced: never dropped by the zero-HR filter ──
    if _forced_set:
        _present = set()
        for s in long_valid:
            _present |= _leg_keys(s)
        for s in _pre_hr_long:
            if (_leg_keys(s) & _forced_set) and not (_leg_keys(s) & _present):
                long_valid.append(s)
                _present |= _leg_keys(s)
                warnings.warn(
                    f"Forced ticker '{s.variance_asset}' was removed by the 0%-HR "
                    f"filter and has been restored (forced names bypass the filter).",
                    stacklevel=2,
                )

    if len(long_valid) < constraints.min_stocks_long:
        # For cross-corridor, use index as the key
        if config.cross_corridor:
            unique_missing = set(s.corridor_condition_asset for s in missing)
            sample_ticker = missing[0].corridor_condition_asset if missing else ""
        else:
            unique_missing = set(s.variance_asset for s in missing)
            sample_ticker = missing[0].variance_asset if missing else ""
        
        empty_ts = pd.DataFrame({"Long Leg": [], "Short Leg": [], "Result": []})
        result = OptimizationResult(
            long_basket=[], short_basket=[],
            long_strike_weighted=0.0, short_strike_weighted=0.0,
            net_strike=0.0, score=-np.inf,
            generations_run=0, converged=False,
            backtest=BacktestResult(timeseries=empty_ts),
        )
        # Use correct key for cross-corridor in debug message
        if config.cross_corridor:
            missing_keys = [s.corridor_condition_asset for s in missing[:15]]
            sample_data = "\n".join([f"    {s.corridor_condition_asset}: strike={s.strike_pct:.4f}, min_w={s.min_weight:.4f}, max_w={s.max_weight:.4f}" for s in missing[:3]])
        else:
            missing_keys = [s.variance_asset for s in missing[:15]]
            sample_data = "\n".join([f"    {s.variance_asset}: strike={s.strike_pct:.4f}, min_w={s.min_weight:.4f}, max_w={s.max_weight:.4f}" for s in missing[:3]])
        
        debug_msg = f"❌ [DEBUG FAIL] len(long_valid)={len(long_valid)} < min_stocks_long={constraints.min_stocks_long}\n" + \
                    f"  long_legs_loaded={len(long_legs_loaded)}\n" + \
                    f"  valid_tickers in matrix={len(valid_tickers)}\n" + \
                    f"  Missing tickers: {missing_keys}\n"
        if len(unique_missing) == 1 and len(missing) > 0:
            debug_msg += f"\n⚠️  WARNING: All {len(missing)} tickers are '{sample_ticker}'\n"
            debug_msg += "   This suggests you pasted the wrong column (e.g. Index column instead of Tickers column)\n"
        debug_msg += f"  Sample missing data:\n{sample_data}"
        result._debug_info = debug_msg
        return result

    # ── Step 4b: Pre-flight feasibility check ──
    # The GA picks subsets of size [min_stocks_long, max_stocks_long].
    # For weights to sum to 1.0, we need max_stocks * max_weight >= 1.0.
    max_w_long = max((s.max_weight for s in long_valid), default=0.0)
    if max_w_long > 0 and constraints.max_stocks_long * max_w_long < 1.0 - 1e-9:
        needed_stocks = int(np.ceil(1.0 / max_w_long))
        needed_weight_pct = 100.0 / constraints.max_stocks_long
        raise ValueError(
            f"Infeasible long constraints:\n"
            f"  max_stocks_long = {constraints.max_stocks_long}\n"
            f"  max_weight = {max_w_long * 100:.1f}%\n"
            f"  max possible allocation = {constraints.max_stocks_long * max_w_long * 100:.1f}%\n"
            f"Required: max_stocks_long × max_weight >= 100%\n"
            f"  → with max_weight = {max_w_long * 100:.1f}%, max_stocks_long must be at least {needed_stocks}\n"
            f"  → with max_stocks_long = {constraints.max_stocks_long}, max_weight must be at least {needed_weight_pct:.1f}%"
        )

    # max_stocks_short=0 means explicit long-only — skip short feasibility and clear shorts
    if constraints.max_stocks_short == 0:
        short_valid = []

    if short_valid:
        max_w_short = max((s.max_weight for s in short_valid), default=0.0)
        if max_w_short > 0 and constraints.max_stocks_short * max_w_short < 1.0 - 1e-9:
            needed_stocks_s = int(np.ceil(1.0 / max_w_short))
            needed_weight_pct_s = 100.0 / constraints.max_stocks_short
            raise ValueError(
                f"Infeasible short constraints:\n"
                f"  max_stocks_short = {constraints.max_stocks_short}\n"
                f"  max_weight = {max_w_short * 100:.1f}%\n"
                f"  max possible allocation = {constraints.max_stocks_short * max_w_short * 100:.1f}%\n"
                f"Required: max_stocks_short × max_weight >= 100%\n"
                f"  → with max_weight = {max_w_short * 100:.1f}%, max_stocks_short must be at least {needed_stocks_s}\n"
                f"  → with max_stocks_short = {constraints.max_stocks_short}, max_weight must be at least {needed_weight_pct_s:.1f}%"
            )

    # ── Forced tickers -> indices in the final long_valid ordering ──
    _forced_indices: List[int] = []
    if _forced_set:
        _matched = set()
        for i, s in enumerate(long_valid):
            hit = _leg_keys(s) & _forced_set
            if hit:
                _forced_indices.append(i)
                _matched |= hit
        _unmatched = _forced_set - _matched
        if _unmatched:
            _sample = sorted({s.variance_asset for s in long_valid})[:15]
            raise ValueError(
                f"Forced ticker(s) not found in the candidate universe after "
                f"filtering: {sorted(_unmatched)}. "
                f"Check spelling/exclusions. Available sample: {_sample}"
            )

    # ── Step 5: Build metric weights ──
    # Legacy scoring was removed from the optimizer, so score_weights=None now
    # defaults to equal weights on the 4 core metrics instead of crashing.
    if score_weights:
        metric_weights = MetricWeights(score_weights)
    else:
        metric_weights = MetricWeights({
            "last_carry": 0.25, "hit_ratio": 0.25,
            "min_payoff": 0.25, "mean_payoff": 0.25,
        })

    # ── Step 6: Run GA ──
    # ── PRE-GA VALIDATION (data for UI expander) ──
    _preview_data = []
    _col_keys = set(col_map.keys())
    for i, s in enumerate(long_valid):
        _key = s.corridor_condition_asset if config.cross_corridor else s.variance_asset
        _in_matrix = _key in _col_keys
        _col_idx = col_map.get(_key, -1)
        _preview_data.append({
            'row': i,
            'var_asset': s.variance_asset,
            'corr_asset': s.corridor_condition_asset if config.cross_corridor else '',
            'series_key': _key,
            'strike_xc_pct': s.strike_cross_corridor * 100 if s.strike_cross_corridor else 0,
            'strike_mono_pct': s.strike_pct,
            'min_weight': s.min_weight,
            'max_weight': s.max_weight,
            'in_pnl_matrix': _in_matrix,
            'col_idx': _col_idx,
            'valid': _in_matrix,
            'reason': '' if _in_matrix else f'key "{_key}" not in pnl_matrix columns',
        })

    optimizer = DispersionOptimizer(
        long_candidates=long_valid,
        short_candidates=short_valid,
        pnl_matrix=pnl_matrix,
        column_map=col_map,
        constraints=constraints,
        missing_data_policy=config.missing_data_policy,
        reweight_grace_days=3,
        is_cross_corridor=config.cross_corridor,
        global_cap=config.global_cap,
        global_floor=config.global_floor,
        metric_weights=metric_weights,
        progress_callback=progress_callback,
        bisect_in_ga=bisect_in_ga,
        seed=seed,
        smooth_weights=False,
        smooth_eps=0.05,
        forced_long_indices=_forced_indices if _forced_indices else None,
    )
    opt_result = optimizer.run()
    opt_result._final_raw_min = getattr(optimizer, '_final_raw_min', None)

    # ── Optional: persist a replayable run bundle (offline reproduction) ──
    if save_bundle_path:
        import dataclasses as _dc
        from functions.dispersion.run_bundle import save_run_bundle
        save_run_bundle(
            save_bundle_path,
            pnl_matrix=optimizer._orig_ts_mat,
            column_map=col_map,
            long_candidates=long_valid,
            short_candidates=short_valid,
            constraints=constraints,
            score_weights=metric_weights.to_dict(),
            seed=seed,
            missing_data_policy=config.missing_data_policy,
            adj_divs=config.adj_divs,
            reweight_grace_days=3,
            is_cross_corridor=config.cross_corridor,
            global_cap=config.global_cap,
            global_floor=config.global_floor,
            bisect_in_ga=bisect_in_ga,
            forced_long_indices=_forced_indices if _forced_indices else None,
            dates=list(_optimizer_dates),
            config=_dc.asdict(config),
            provenance={
                "forced_tickers": sorted(_forced_set) if _forced_set else [],
                "excluded_tickers": sorted(_excluded_set) if _excluded_set else [],
                "start_date": start_date,
                "end_date": end_date,
                "filter_zero_hr": bool(filter_zero_hr),
            },
            result=opt_result,
        )

    # ── Persist smoothing state for interactive post-smoothing in UI ──
    opt_result._smooth_state = {
        "ts_mat": optimizer._ts_mat,             # nan_to_num'd matrix (zeros where NaN) — for evaluation + smooth_weights
        "valid_mask": optimizer._valid_mask,      # ~isnan(orig_ts_mat) — for adaptive renorm
        "col_pos": optimizer._col_pos,           # key -> column index mapping (same as FINAL-RAW uses)
        "n_rows": optimizer._n_rows,
        "weight_solver": optimizer._weight_solver,
        "long_indices": list(optimizer._last_best.long_indices) if hasattr(optimizer, '_last_best') and optimizer._last_best else None,
        "long_candidates": long_valid,
        "is_cross_corridor": config.cross_corridor,
        "missing_data_policy": config.missing_data_policy,
        "price_data_all": price_data_all,
        "index_data_all": index_data_all,
        "legs": legs,
        "internal_cfg": internal_cfg,
        "start_date": start_date,
    }

    # ── MILP benchmark (optional) ──
    if run_milp:
        milp_result = optimizer.milp_benchmark()
        opt_result._milp_result = milp_result
    else:
        opt_result._milp_result = None

    # ── TEMP AUDIT: scoring mode used ──
    print(f"[AUDIT] scoring_mode={optimizer._scoring_mode} | use_new={optimizer._use_new_scoring} | score_fn_fitted={optimizer._score_fn.is_fitted if optimizer._score_fn else 'N/A'}")

    # ── Step 6b: Capture debug info for UI ──
    
    # Capture debug info for UI display
    debug_info = []
    debug_info.append(f"🔍 [DEBUG POST-GA] score={opt_result.score:.6f}, converged={opt_result.converged}")
    debug_info.append(f"  long_basket={len(opt_result.long_basket)}, short_basket={len(opt_result.short_basket)}")
    
    if opt_result.score <= -1e9:
        debug_info.append("\n❌ NO VALID SOLUTION FOUND")
        debug_info.append("This is likely due to:")
        debug_info.append("  - All candidates failing constraint checks")
        debug_info.append("  - Reference sample produced constant last_carry (zero spread)")
        debug_info.append("  - P&L matrix has no valid data for the optimization window")
        debug_info.append(f"  - Valid long candidates after filtering: {len(long_valid)}")
        debug_info.append(f"  - Valid short candidates after filtering: {len(short_valid)}")
        
        # Show first 5 long candidates with their bounds
        if long_valid:
            debug_info.append("\nFirst 5 long candidates:")
            for i, s in enumerate(long_valid[:5]):
                key = s.corridor_condition_asset if config.cross_corridor else s.variance_asset
                debug_info.append(f"  {i+1}. {key}: min_weight={s.min_weight:.4f}, max_weight={s.max_weight:.4f}")
        
        # Show first 5 short candidates with their bounds
        if short_valid:
            debug_info.append("\nFirst 5 short candidates:")
            for i, s in enumerate(short_valid[:5]):
                key = s.corridor_condition_asset if config.cross_corridor else s.variance_asset
                debug_info.append(f"  {i+1}. {key}: min_weight={s.min_weight:.4f}, max_weight={s.max_weight:.4f}")
    
    # ── Step 7: Backtest winner (free — data in memory) ──
    # Use FULL price history (not date-cut) so backtester has n_exp warm-up rows
    # bt.run() applies start_date filter internally after computing rolling PnL
    bt = DispersionBacktester(internal_cfg)
    bt_result = bt.run_from_optimization(
        price_data=price_data_all,
        long_basket=opt_result.long_basket,
        short_basket=opt_result.short_basket,
        legs=legs,
        index_data=index_data_all,
        start_date=start_date,
    )
    opt_result.backtest = bt_result

    # ── TEMP NUMERIC CHECK: solver vs backtest alignment ──
    try:
        _best_long = opt_result.long_basket  # [(key, weight), ...]
        if _best_long:
            _long_keys = [k for k, w in _best_long]
            _long_w = np.array([w for k, w in _best_long])
            _long_cols = [col_map[k] for k in _long_keys if k in col_map]
            if len(_long_cols) == len(_long_keys):
                # Use adaptive renormalization (same formula as FINAL-RAW and backtester)
                _solver_pnl = optimizer._adaptive_net_pnl(
                    np.array(_long_cols, dtype=int), _long_w
                )
                s = _solver_pnl
                if config.global_cap < 9999998 or config.global_floor > -9999998:
                    s = np.clip(s, config.global_floor, config.global_cap)
                _bt_ts = bt_result.timeseries if bt_result else None
                b = _bt_ts["Result"].values if (_bt_ts is not None and "Result" in _bt_ts.columns) else np.array([])
                s_nz = s[s != 0.0]
                b_nz = b[b != 0.0] if len(b) > 0 else np.array([])
                s_min_nz = float(s_nz.min()) if len(s_nz) > 0 else 0.0
                b_min_nz = float(b_nz.min()) if len(b_nz) > 0 else 0.0
                print("[NUMERIC-CHECK] solver: len=%d min=%.4f mean=%.4f min_nz=%.4f" % (len(s), s.min(), s.mean(), s_min_nz), flush=True)
                print("[NUMERIC-CHECK] backtest: len=%d min=%.4f mean=%.4f min_nz=%.4f" % (len(b), b.min(), b.mean(), b_min_nz), flush=True)
                n = min(len(s), len(b))
                if n > 0:
                    close = np.allclose(s[:n], b[:n], atol=1e-8)
                    print("[NUMERIC-CHECK] same_len=%s allclose=%s" % (len(s) == len(b), close), flush=True)
                    if not close:
                        idx = np.where(~np.isclose(s[:n], b[:n], atol=1e-8))[0][:5]
                        for i in idx:
                            print("  diff[%d]: solver=%.6f backtest=%.6f" % (i, s[i], b[i]), flush=True)
    except Exception as _e:
        print("[NUMERIC-CHECK] failed: %s" % repr(_e), flush=True)

    # Return debug info in result for UI display
    opt_result._debug_info = "\n".join(debug_info) if debug_info else ""
    opt_result._preview_data = _preview_data
    opt_result._rejection_reasons = dict(getattr(optimizer, '_rejection_reasons', {}))
    opt_result._debug_table = getattr(optimizer, '_debug_table', '')
    opt_result._ref_debug_table = getattr(optimizer, '_ref_debug_table', '')
    opt_result._collapse_diagnostic = getattr(optimizer, '_collapse_diagnostic', '')
    
    return opt_result


# ═══════════════════════════════════════════════════════════════════════════════
# 4. BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════

def backtest(
    df: pd.DataFrame,
    config: DispersionConfig,
    *,
    start_date: date = None,
) -> BacktestResult:
    """
    Run PNL backtest on a defined basket. Loads prices from Bloomberg.

    Parameters
    ----------
    df : DataFrame
        Columns: Variance Asset, Weight (%), Strike Mono Var Swap (%), Side
        If cross_corridor: also Corridor Condition Asset, Strike Cross Corridor (%)
    config : DispersionConfig
    start_date : date, optional
        Default: lookback_years from today.

    Returns
    -------
    BacktestResult
        .timeseries, .hit_ratio, .mean_return, .last_value, .max_drawdown
    """
    df = _normalize_df_columns(df)
    required = ['Variance Asset', 'Weight (%)', 'Strike Mono Var Swap (%)', 'Side']
    if config.cross_corridor:
        required.append('Strike Cross Corridor (%)')
        required.append('Corridor Condition Asset')
    _validate_columns(df, required, 'backtest')

    from functions.dispersion._backtester import DispersionDataLoader, DispersionBacktester

    # ── Step 1: Parse DataFrame ──
    legs, weights = _df_to_backtest_inputs(df, config.cross_corridor)

    # ── Helper for cross-corridor series key ──
    def _series_key(leg):
        return leg.corridor_condition_asset if config.cross_corridor and leg.corridor_condition_asset else leg.variance_asset

    # ── Step 2: Load Bloomberg data ──
    internal_cfg = _to_swap_config(config, start_date=start_date)
    loader = DispersionDataLoader(internal_cfg)
    basket = BasketInput(
        long_candidates=[s for s in legs if weights[_series_key(s)] > 0],
        short_candidates=[s for s in legs if weights[_series_key(s)] < 0],
    )
    data = loader.load(basket)

    if data["price_data"].empty:
        empty_ts = pd.DataFrame({"Long Leg": [], "Short Leg": [], "Result": []})
        return BacktestResult(timeseries=empty_ts)

    # ── Step 3: Run backtester ──
    bt = DispersionBacktester(internal_cfg)
    return bt.run(
        price_data=data["price_data"],
        legs=data["legs"],
        weights=weights,
        index_data=data.get("index_data"),
        start_date=start_date,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PRIVATE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _filter_by_hit_ratio(long_legs, short_legs, pnl_matrix, col_map, constraints):
    """Remove legs with useless hit ratios from candidate pools."""
    def _hr(ticker):
        idx = col_map.get(ticker)
        if idx is None:
            return 50.0
        col = pnl_matrix[:, idx]
        valid = col[~np.isnan(col)]
        non_zero = valid[valid != 0]
        if len(non_zero) == 0:
            return 50.0
        return float((non_zero > 0).sum()) / len(non_zero) * 100.0

    # For cross-corridor, use corridor_condition_asset to look up in col_map
    # because col_map keys are built from corridor_condition_asset in _build_pnl_matrix
    long_keys = [s.corridor_condition_asset if s.corridor_condition_asset else s.variance_asset for s in long_legs]
    short_keys = [s.corridor_condition_asset if s.corridor_condition_asset else s.variance_asset for s in short_legs]
    
    filtered_long = [s for s, k in zip(long_legs, long_keys) if _hr(k) > 0.0]
    if len(filtered_long) >= constraints.min_stocks_long:
        long_legs = filtered_long

    filtered_short = [s for s, k in zip(short_legs, short_keys) if _hr(k) < 100.0]
    if len(filtered_short) >= constraints.min_stocks_short:
        short_legs = filtered_short

    return long_legs, short_legs
