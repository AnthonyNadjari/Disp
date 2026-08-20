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
#   - BacktestResult.max_drawdown → RAW P&L UNITS, <= 0 (equity-curve
#     peak-to-trough, same convention as per_stock_stats)
#   - SolveResult.results_df → pre-formatted strings ("21.40%")
#
# RULE: All conversions happen HERE in _api.py. Internal modules never convert.
# ═══════════════════════════════════════════════════════════════════════════════
"""Public headless API for the Gaia_PP dispersion engine.

The Streamlit page is ONLY a front-end: everything it does routes through the
functions below and works identically from a plain script / notebook:

    from functions.dispersion._api import optimize, backtest, optimize_multi
    from functions.dispersion.models import DispersionConfig, OptimizationConstraints

    cfg = DispersionConfig(cross_corridor=True, n_exp=252)     # product setup
    cons = OptimizationConstraints(min_stocks_long=5, max_stocks_long=10,
                                   max_net_strike=0.29, time_limit_seconds=30)
    result = optimize(long_df, cfg, cons,
                      score_weights={"mean_payoff": 0.5, "hit_ratio": 0.5},
                      seed=0)
    result.long_basket, result.score, result.backtest.timeseries

    bt = backtest(basket_df, cfg)          # standalone basket backtest

Input DataFrames use the SAME columns the page shows (Variance Asset,
Corridor Condition Asset, strikes in %, Min/Max Weight in %). The naming
convention (variance vs corridor asset, mono vs cross leg) is documented at
the top of functions/dispersion/_backtester.py.
"""
from __future__ import annotations

import re
import warnings
import numpy as np
import pandas as pd
from datetime import date
from typing import Callable, Dict, List, Optional, Tuple

from functions.dispersion._logging import logger as _engine_log
from functions.dispersion.models import (
    BucketConstraint,
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
        reweight_grace_days=cfg.reweight_grace_days,
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

# ── RIC / BBG dual input ──
# Input tables accept both forms ("AAPL.O" / "ISP.MI" / ".STOXX50E" or
# "AAPL US Equity" / "SPX Index"); RICs are normalized to the BBG form the
# Bloomberg loader needs (desk-network conversion, cached; anything
# unrecognized is kept as-is and the loader warns loudly).

_RIC_STOCK_RE = re.compile(r"^[A-Z0-9]{1,12}\.[A-Z]{1,4}$")
_ticker_conv_cache: Dict[str, str] = {}


def _looks_like_ric(name: str) -> bool:
    name = str(name).strip()
    return name.startswith(".") or bool(_RIC_STOCK_RE.match(name))


def _to_bbg(name) -> str:
    """Normalize a ticker to BBG form if it is a RIC; pass BBG names through."""
    name = str(name).strip()
    if not name or not _looks_like_ric(name):
        return name
    if name not in _ticker_conv_cache:
        try:
            from functions.common.tickers import ric_to_bbg
            _ticker_conv_cache[name] = ric_to_bbg(name) or name
        except Exception:
            _ticker_conv_cache[name] = name
    return _ticker_conv_cache[name]


def _to_ric(name) -> str:
    """Normalize a ticker to RIC form if it is a BBG name (the pricing portal
    resolves RIC identifiers); pass RICs through. Unrecognized names are kept
    as-is (the portal error will say why)."""
    name = str(name).strip()
    if not name or _looks_like_ric(name):
        return name
    if name not in _ticker_conv_cache:
        try:
            from functions.common.tickers import bbg_to_ric
            _ticker_conv_cache[name] = bbg_to_ric(name) or name
        except Exception:
            _ticker_conv_cache[name] = name
    return _ticker_conv_cache[name]


def _df_to_legs(df: pd.DataFrame, is_cross_corridor: bool) -> List[DispersionLeg]:
    """Parse a tickers DataFrame into DispersionLeg objects with unit conversions."""
    legs = []
    for _, row in df.iterrows():
        var_asset = _to_bbg(row['Variance Asset'])

        # Strike: DataFrame is percentage → DispersionLeg is decimal
        # Long leg uses Strike Mono Var Swap (%)
        strike_pct = float(row.get('Strike Mono Var Swap (%)', 0))
        if not np.isfinite(strike_pct):
            raise ValueError(
                f"'{var_asset}': 'Strike Mono Var Swap (%)' is blank/NaN. A missing "
                f"mono strike would silently produce an all-NaN leg (under DROP it "
                f"wipes the whole backtest; otherwise the leg just vanishes).")
        strike = strike_pct / 100.0

        # Corridor Condition Asset / Strike Cross Corridor
        corr_asset = None
        cross_strike = None
        if is_cross_corridor:
            corr_asset = _to_bbg(row.get('Corridor Condition Asset', '')) or None
            cross_strike_pct = float(row.get('Strike Cross Corridor (%)'))
            if pd.isna(cross_strike_pct):
                raise ValueError(f"Cross-corridor backtest requires 'Strike Cross Corridor (%)' for '{var_asset}'.")
            cross_strike = cross_strike_pct / 100.0

        # Min/Max weight: UI provides percentage points (e.g. 1.5 = 1.5%)
        # DispersionLeg expects decimal (0.015 = 1.5%)
        min_w = float(row.get('Min Weight', 1.0)) / 100.0
        max_w = float(row.get('Max Weight', 60.0)) / 100.0
        sector = row.get('Sector', None)

        # Absolute-Vega mode inputs (optional columns, absolute Vega units)
        def _opt_float(v):
            try:
                f = float(v)
            except (TypeError, ValueError):
                return None
            return None if pd.isna(f) else f

        axe_target = _opt_float(row.get('Axe Target', None))
        axe_cap = _opt_float(row.get('Axe Cap', None))

        legs.append(DispersionLeg(
            variance_asset=var_asset,
            strike_mono_var_swap=strike,
            corridor_condition_asset=corr_asset,
            strike_cross_corridor=cross_strike,
            sector=str(sector) if sector and str(sector).strip() else None,
            min_weight=min_w,
            max_weight=max_w,
            axe_target=axe_target,
            axe_cap=axe_cap,
        ))
    return legs


def _df_to_backtest_inputs(df: pd.DataFrame, is_cross_corridor: bool) -> Tuple[List[DispersionLeg], Dict[str, float]]:
    """Parse a backtest DataFrame into DispersionLeg objects + weights dict."""
    legs = []
    weights = {}
    for idx, row in df.iterrows():
        var_asset = _to_bbg(row['Variance Asset'])
        weight_pct = float(row['Weight (%)'])
        if not np.isfinite(weight_pct):
            # A NaN weight fails both w>0 and w<0 downstream and the leg would
            # vanish silently — skip it loudly instead.
            warnings.warn(f"'{var_asset}': 'Weight (%)' is blank/NaN — leg skipped.",
                          stacklevel=2)
            continue

        # Strike: percentage → decimal
        strike_pct = float(row.get('Strike Mono Var Swap (%)', 0))
        if not np.isfinite(strike_pct):
            raise ValueError(
                f"'{var_asset}': 'Strike Mono Var Swap (%)' is blank/NaN — a missing "
                f"mono strike silently drops the leg from the backtest.")
        strike = strike_pct / 100.0

        # Corridor Condition Asset / Strike Cross Corridor
        corr_asset = None
        cross_strike = None
        if is_cross_corridor:
            corr_asset = _to_bbg(row.get('Corridor Condition Asset', '')) or None
            cross_strike_pct = float(row.get('Strike Cross Corridor (%)'))
            if pd.isna(cross_strike_pct):
                raise ValueError(f"Cross-corridor backtest requires 'Strike Cross Corridor (%)' for '{var_asset}'.")
            cross_strike = cross_strike_pct / 100.0

        # Weight: percentage → decimal. With a Side column, the sign is
        # side × abs(weight) (robust to already-signed inputs); without it,
        # the weight's own sign decides (+ = long, − = short).
        side_raw = row.get('Side', None)
        if side_raw is None or (isinstance(side_raw, float) and pd.isna(side_raw)):
            signed_weight = weight_pct / 100.0
        else:
            side = str(side_raw).strip().lower()
            signed_weight = abs(weight_pct) / 100.0 * (1.0 if side == 'long' else -1.0)

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
    variance_px: pd.DataFrame,
    corridor_px: Optional[pd.DataFrame],
    legs: List[DispersionLeg],
    cfg: DispersionConfig,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Optimizer-side P&L matrix — thin wrapper over the SHARED builder
    `functions.dispersion._backtester.compute_leg_pnl_columns` (the exact same
    code path the backtester uses), so scored and backtested P&L can never
    drift apart.

    variance_px = Variance Asset prices (the index in cross-corridor mode);
    corridor_px = Corridor Condition Asset (stock) prices, None in mono mode.
    Returns (pnl_matrix, column_map) — columns keyed by the per-name key
    (Corridor Condition Asset in cross-corridor, else Variance Asset).
    """
    from functions.dispersion._backtester import compute_leg_pnl_columns

    internal_cfg = _to_swap_config(cfg)
    pnl_matrix, column_keys, _ = compute_leg_pnl_columns(
        variance_px, corridor_px, legs, internal_cfg, capture_cross_legs=False)
    col_map = {t: i for i, t in enumerate(column_keys)}
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

    # The pricing portal resolves RIC identifiers — accept BBG names too and
    # convert them (cached desk lookup; RICs pass through unchanged).
    df['Variance Asset'] = df['Variance Asset'].map(_to_ric)
    df['Corridor Condition Asset'] = df['Corridor Condition Asset'].map(_to_ric)

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
        ticker_results=getattr(result, "ticker_results", []) or [],
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

    # The pricing portal resolves RIC identifiers — accept BBG names too.
    df['Variance Asset'] = df['Variance Asset'].map(_to_ric)
    df['Corridor Condition Asset'] = df['Corridor Condition Asset'].map(_to_ric)

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
        ticker_results=getattr(result, "ticker_results", []) or [],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. OPTIMIZE
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_vega_arg(vega):
    """None | dict | VegaConfig → Optional[VegaConfig] (validates)."""
    from functions.dispersion.models import VegaConfig
    if vega is None:
        return None
    if isinstance(vega, VegaConfig):
        return vega if vega.enabled else None
    if isinstance(vega, dict):
        d = dict(vega)
        if not d.pop("enabled", True):
            return None
        try:
            return VegaConfig(**d)
        except TypeError as te:
            raise ValueError(
                f"vega dict {vega!r} is invalid: {te}. Expected keys: v_min, "
                f"v_max (absolute Vega units) and optionally enabled.")
    raise TypeError(f"vega must be a VegaConfig or dict, got {type(vega).__name__}")


# ── Prep cache: skip Bloomberg + matrix build when nothing upstream changed ──
# Single slot, opt-in (cache_prep=True). Key = content hash of every input the
# pipeline consumes, so any change to the data/config/filters busts it.
# The calendar DATE is part of the key: identical inputs on a different day
# must reload market data — a HIT may only serve same-day data.
_PREP_CACHE: dict = {}


def _prep_cache_key(long_df, config, constraints, short_df, start_date, end_date,
                    filter_zero_hr, forced_set, excluded_set) -> str:
    import dataclasses as _dc
    import hashlib
    h = hashlib.sha256()
    # Bust the cache daily: a next-day rerun with identical inputs must not
    # silently reuse stale prices / a stale P&L matrix.
    h.update(date.today().isoformat().encode())
    for df in (long_df, short_df):
        if df is None or df.empty:
            h.update(b"~none~")
        else:
            h.update(repr(tuple(df.columns)).encode())
            h.update(pd.util.hash_pandas_object(df.astype(str), index=True).values.tobytes())
    h.update(repr(sorted((k, str(v)) for k, v in _dc.asdict(config).items())).encode())
    h.update(repr((constraints.min_stocks_long, constraints.max_stocks_long,
                   constraints.min_stocks_short, constraints.max_stocks_short,
                   constraints.max_net_strike)).encode())
    h.update(repr((str(start_date), str(end_date), bool(filter_zero_hr),
                   tuple(sorted(forced_set)), tuple(sorted(excluded_set)))).encode())
    return h.hexdigest()


def _copy_prep(prep: Dict) -> Dict:
    """Fresh leg objects / arrays so a run can never poison the cached copy."""
    import dataclasses as _dc
    out = dict(prep)
    for k in ("long_valid", "short_valid", "legs"):
        out[k] = [_dc.replace(l) for l in prep[k]]
    out["pnl_matrix"] = prep["pnl_matrix"].copy()
    out["col_map"] = dict(prep["col_map"])
    out["forced_indices"] = list(prep["forced_indices"])
    if prep.get("active_mask") is not None:
        out["active_mask"] = prep["active_mask"].copy()
    return out


def _prepare_cached(cache_prep, long_df, config, constraints, *, short_df,
                    start_date, end_date, filter_zero_hr, forced_set, excluded_set) -> Dict:
    """_prepare_optimization_inputs with the opt-in single-slot cache in front."""
    if not cache_prep:
        return _prepare_optimization_inputs(
            long_df, config, constraints, short_df=short_df,
            start_date=start_date, end_date=end_date, filter_zero_hr=filter_zero_hr,
            forced_set=forced_set, excluded_set=excluded_set)
    key = _prep_cache_key(long_df, config, constraints, short_df, start_date,
                          end_date, filter_zero_hr, forced_set, excluded_set)
    if _PREP_CACHE.get("key") == key:
        _engine_log.info("prep cache HIT — Bloomberg load + matrix build skipped")
        _PREP_CACHE["last_was_hit"] = True
        out = _copy_prep(_PREP_CACHE["prep"])
        out["reference_cache"] = _PREP_CACHE.setdefault("reference_cache", {})
        return out
    _PREP_CACHE["last_was_hit"] = False
    prep = _prepare_optimization_inputs(
        long_df, config, constraints, short_df=short_df,
        start_date=start_date, end_date=end_date, filter_zero_hr=filter_zero_hr,
        forced_set=forced_set, excluded_set=excluded_set)
    if prep["early_result"] is None:
        _PREP_CACHE["key"] = key
        _PREP_CACHE["prep"] = _copy_prep(prep)
        _PREP_CACHE["reference_cache"] = {}
        prep["reference_cache"] = _PREP_CACHE["reference_cache"]
    return prep


def _prepare_optimization_inputs(
    long_df: pd.DataFrame,
    config: DispersionConfig,
    constraints: OptimizationConstraints,
    *,
    short_df: pd.DataFrame,
    start_date,
    end_date,
    filter_zero_hr: bool,
    forced_set: set,
    excluded_set: set,
) -> Dict:
    """Shared data pipeline for optimize()/optimize_multi(): parse the input
    DataFrames, load prices, build + slice the P&L matrix, filter the
    candidate universe (exclusions, 0%-HR filter, forced restoration), run
    the pre-flight feasibility checks and resolve forced indices.

    Returns {"early_result": OptimizationResult, ...} for the historical
    early-exit paths (early_result is None on success), plus the prepared
    inputs the callers need.
    """
    from functions.dispersion._backtester import DispersionDataLoader

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
    data = loader.load(basket, end_date=end_date)

    if data["variance_px"].empty:
        empty_ts = pd.DataFrame({"Long Leg": [], "Short Leg": [], "Result": []})
        return {"early_result": OptimizationResult(
            long_basket=[], short_basket=[],
            long_strike_weighted=0.0, short_strike_weighted=0.0,
            net_strike=0.0, score=-np.inf,
            generations_run=0, converged=False,
            backtest=BacktestResult(timeseries=empty_ts),
        )}

    # ── Step 3: Build PNL matrix ──
    variance_px_all = data["variance_px"]
    corridor_px_all = data.get("corridor_px")
    legs = data["legs"]
    long_legs_loaded = data["long_legs"]
    short_legs_loaded = data["short_legs"]

    # WARM-UP FIX: Build PnL from FULL price history (same as backtester uses),
    # then slice the resulting matrix to [start_date, end_date].
    # This ensures the rolling kernel has n_exp warm-up rows BEFORE start_date,
    # so the optimizer's matrix row 0 matches the backtest curve row 0 exactly.
    variance_px_for_build = variance_px_all
    corridor_px_for_build = corridor_px_all

    pnl_matrix_full, col_map = _build_pnl_matrix(variance_px_for_build, corridor_px_for_build, legs, config)

    # Slice to [start_date, end_date] — rows in pnl_matrix_full correspond 1:1 to variance_px_for_build.index
    _build_index = variance_px_for_build.index
    if start_date:
        _start_idx = _build_index.searchsorted(pd.Timestamp(start_date), side='left')
    elif end_date:
        _start_idx = 0
    else:
        # No explicit dates: default to the last lookback window — sliced AFTER
        # the full-history build, so the window's first rows have their n_exp
        # warm-up behind them (truncating prices BEFORE the build would make
        # the first n_exp window rows NaN and diverge from the delivered
        # backtest at the boundary).
        lookback_days = int(config.lookback_years * 252) + config.n_exp
        _start_idx = max(0, len(_build_index) - lookback_days)
    if end_date:
        _end_idx = _build_index.searchsorted(pd.Timestamp(end_date), side='right')
    else:
        _end_idx = len(_build_index)
    pnl_matrix = pnl_matrix_full[_start_idx:_end_idx]

    # Adaptive grace mask: built on the FULL history (a name mid-gap at window
    # start, with pre-window prints, is still in-grace) and sliced together
    # with the matrix — the backtester sees full history too, so scored and
    # delivered P&L stay equal at the window boundary. None when inactive.
    if (config.missing_data_policy == MissingDataPolicy.ADAPTIVE_REWEIGHT
            and config.reweight_grace_days > 0):
        from functions.dispersion.scoring.weight_solver import (
            active_mask_with_grace, carry_pnl_within_grace)
        _isvalid_full = ~np.isnan(pnl_matrix_full)
        _active_mask = active_mask_with_grace(
            _isvalid_full, config.reweight_grace_days)[_start_idx:_end_idx]
        # In-grace gap days carry the name's last mark — computed on FULL
        # history then sliced, same boundary rule as the mask. The optimizer
        # receives this matrix together with the external mask and does not
        # re-carry it.
        pnl_matrix = carry_pnl_within_grace(
            pnl_matrix_full, _isvalid_full,
            config.reweight_grace_days)[_start_idx:_end_idx]
    else:
        _active_mask = None

    # Also keep the date index for the optimizer window (used later for NUMERIC-CHECK alignment)
    _optimizer_dates = _build_index[_start_idx:_end_idx]
    # variance_px = sliced prices (used only for candidate filtering, not PnL computation)
    variance_px = variance_px_for_build.iloc[_start_idx:_end_idx]
    corridor_px = corridor_px_for_build.iloc[_start_idx:_end_idx] if (corridor_px_for_build is not None and not corridor_px_for_build.empty) else corridor_px_for_build

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
    if excluded_set:
        long_valid = [s for s in long_valid if not (_leg_keys(s) & excluded_set)]
        short_valid = [s for s in short_valid if not (_leg_keys(s) & excluded_set)]

    # Snapshot before the zero-HR filter so forced names can be restored
    _pre_hr_long = list(long_valid)

    if filter_zero_hr:
        long_valid, short_valid = _filter_by_hit_ratio(
            long_valid, short_valid, pnl_matrix, col_map, constraints)

    # ── Forced: never dropped by the zero-HR filter ──
    if forced_set:
        _present = set()
        for s in long_valid:
            _present |= _leg_keys(s)
        for s in _pre_hr_long:
            if (_leg_keys(s) & forced_set) and not (_leg_keys(s) & _present):
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
        return {"early_result": result}

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
    if forced_set:
        _matched = set()
        for i, s in enumerate(long_valid):
            hit = _leg_keys(s) & forced_set
            if hit:
                _forced_indices.append(i)
                _matched |= hit
        _unmatched = forced_set - _matched
        if _unmatched:
            import difflib
            from collections import Counter
            # In cross-corridor mode the per-name key IS the Corridor Condition
            # Asset (the traded single stock), NOT the shared Variance Asset
            # (index). Report the candidates under the key the user must force,
            # otherwise the sample is just the 2-3 indices and misleads them.
            if config.cross_corridor:
                _cand_keys = sorted(
                    {s.corridor_condition_asset for s in long_valid if s.corridor_condition_asset})
                _key_label = "Corridor Condition Asset"
            else:
                _cand_keys = sorted({s.variance_asset for s in long_valid})
                _key_label = "Variance Asset"
            _cand_lower = {k.casefold(): k for k in _cand_keys}
            # Did the name appear in the RAW input but get dropped?
            # long_legs = the user's parsed rows, pre-load.
            _raw_keys = set()
            for _s in long_legs:
                _raw_keys |= _leg_keys(_s)
            # Column sets to pinpoint WHY a present name is not a candidate.
            # cross-corridor: variance_px = index prices, corridor_px = stock prices.
            _pd_cols = {str(c).strip() for c in variance_px.columns} if variance_px is not None else set()
            _id_cols = ({str(c).strip() for c in corridor_px.columns}
                        if (corridor_px is not None and not getattr(corridor_px, "empty", True)) else set())
            _cc_counts = Counter(str(s.corridor_condition_asset).strip()
                                 for s in long_legs if getattr(s, "corridor_condition_asset", None))

            def _why_dropped(u):
                rows = [s for s in long_legs if u in _leg_keys(s)]
                if not rows:
                    return "not present in the parsed input."
                s = rows[0]
                va = str(s.variance_asset).strip()
                cc = (str(s.corridor_condition_asset).strip()
                      if getattr(s, "corridor_condition_asset", None) else "")
                if config.cross_corridor:
                    if va not in _pd_cols:
                        return (f"its Variance Asset (index) '{va}' was NOT returned by Bloomberg, "
                                f"so the whole row is dropped — check that index ticker.")
                    if cc and _cc_counts.get(cc, 0) > 1:
                        return (f"'{cc}' appears on {_cc_counts[cc]} input rows (duplicate); P&L "
                                f"columns are keyed by Corridor Condition Asset so duplicates "
                                f"collapse — keep one row per stock.")
                    if cc and cc not in _id_cols:
                        return (f"its corridor stock price '{cc}' did not load from Bloomberg "
                                f"(the cross P&L cannot be computed) — check that stock ticker.")
                    return ("data loaded but no P&L column was produced (history too short for the "
                            "n_exp warm-up, or all-NaN series).")
                if va not in _pd_cols:
                    return f"its ticker '{va}' was NOT returned by Bloomberg."
                return "data loaded but no P&L column was produced (history too short, or all-NaN)."

            _lines = []
            for u in sorted(_unmatched):
                if u in _raw_keys:
                    _lines.append(f"  - '{u}': present in your input but NOT a candidate — {_why_dropped(u)}")
                else:
                    near = [_cand_lower[n] for n in
                            difflib.get_close_matches(u, list(_cand_lower.keys()), n=3, cutoff=0.6)]
                    if near:
                        _lines.append(f"  - '{u}': not found. Did you mean {near}?")
                    else:
                        _lines.append(f"  - '{u}': not found.")
            _hint = ""
            if config.cross_corridor:
                _hint = (
                    f"\nCross-corridor: force the {_key_label} (the traded single "
                    f"stock, e.g. '005930 KP Equity'), not the Variance Asset (index).")
            raise ValueError(
                f"Forced ticker(s) not found among the {len(_cand_keys)} candidate "
                f"name(s) after filtering:\n" + "\n".join(_lines) + _hint +
                f"\nAvailable {_key_label} sample: {_cand_keys[:15]}"
            )

    return {
        "early_result": None,
        "internal_cfg": internal_cfg,
        "variance_px_all": variance_px_all,
        "corridor_px_all": corridor_px_all,
        "legs": legs,
        "pnl_matrix": pnl_matrix,
        "col_map": col_map,
        "long_valid": long_valid,
        "short_valid": short_valid,
        "forced_indices": _forced_indices,
        "optimizer_dates": _optimizer_dates,
        "active_mask": _active_mask,
    }


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
    n_reference_samples: int = None,
    bucket_constraints: List = None,
    vega=None,
    robustness_check: bool = False,
    log_level: str = None,
    cache_prep: bool = False,
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
    n_reference_samples : int, optional
        Size of the random-basket reference sample used to fit the quantile
        normalizer.  Default None = adaptive (300, or 800 when a tail metric
        such as min_payoff / cvar_5 is active).  Must be >= 100.
    bucket_constraints : list, optional
        Per-bucket (region/sector) constraints on the LONG leg — list of
        :class:`~functions.dispersion.models.BucketConstraint` or plain dicts
        with the same fields ({'bucket': 'US', 'min_names': 1, 'max_names': 3,
        'min_weight': 0.2, 'max_weight': 0.6}; weights DECIMAL).  Buckets
        match the input DataFrame's 'Sector' column.  None/empty = inactive.
    vega : VegaConfig or dict, optional
        Absolute-Vega mode toggle (None = OFF, historical behaviour).  Pass
        :class:`~functions.dispersion.models.VegaConfig` or
        ``{'v_min': 50, 'v_max': 200}``.  When ON: per-name Vega v_i with
        V = Σv_i free in [v_min, v_max]; Min/Max Weight become concentration
        bounds on v_i/V; the optional 'Axe Target' / 'Axe Cap' columns of
        long_df feed the recycling objectives ('axe_book_cleaned',
        'axe_package_recycled' in score_weights) and the hard caps
        v_i <= cap.  Basket P&L stays Σ(v_i·pnl_i)/V — the weight series —
        so scores/backtests remain comparable with the toggle OFF.
    robustness_check : bool
        When True, run the day-resampling bootstrap diagnostic (300 draws
        with replacement; winner vs its 10 best refinement challengers) and
        attach it as ``result.robustness`` — {top1_freq, top3_freq,
        winner_raw_ci}.  Deterministic (derived from the run seed).
    cache_prep : bool
        Keep the prepared inputs (Bloomberg prices, P&L matrix, filtered
        universe) in memory and reuse them on the next call whose inputs are
        content-identical — re-running with different score_weights then skips
        the whole load/build phase. Off by default.
    log_level : str, optional
        Activate the engine's diagnostic logger ("DEBUG"/"INFO"/...).
        Default None = silent (historical prints now live behind this).

    Returns
    -------
    OptimizationResult
        .long_basket, .short_basket, .net_strike, .score, .converged, .backtest
    """
    # ── Optional engine diagnostics ──
    if log_level:
        from functions.dispersion._logging import configure_engine_logging
        configure_engine_logging(log_level)

    # ── Forced / excluded: cheap validation before any data load ──
    def _norm_ticker_set(lst):
        return {str(t).strip().casefold() for t in (lst or []) if str(t).strip()}

    # ── Bucket constraints: normalize dicts → BucketConstraint (validates) ──
    _bucket_cons: List[BucketConstraint] = []
    for _bc in (bucket_constraints or []):
        if isinstance(_bc, BucketConstraint):
            _bucket_cons.append(_bc)
        elif isinstance(_bc, dict):
            try:
                _bucket_cons.append(BucketConstraint(**_bc))
            except TypeError as _te:
                raise ValueError(
                    f"bucket_constraints entry {_bc!r} is invalid: {_te}. Expected "
                    f"fields: bucket, min_names, max_names, min_weight, max_weight."
                )
        else:
            raise TypeError(
                f"bucket_constraints entries must be BucketConstraint or dict, "
                f"got {type(_bc).__name__}"
            )

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

    # ── Vega toggle: normalize dict → VegaConfig (validates) ──
    _vega_cfg = _normalize_vega_arg(vega)

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

    prep = _prepare_cached(
        cache_prep, long_df, config, constraints,
        short_df=short_df, start_date=start_date, end_date=end_date,
        filter_zero_hr=filter_zero_hr,
        forced_set=_forced_set, excluded_set=_excluded_set,
    )
    if prep["early_result"] is not None:
        return prep["early_result"]
    internal_cfg = prep["internal_cfg"]
    variance_px_all = prep["variance_px_all"]
    corridor_px_all = prep["corridor_px_all"]
    legs = prep["legs"]
    pnl_matrix = prep["pnl_matrix"]
    col_map = prep["col_map"]
    long_valid = prep["long_valid"]
    short_valid = prep["short_valid"]
    _forced_indices = prep["forced_indices"]
    _optimizer_dates = prep["optimizer_dates"]

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
        reweight_grace_days=config.reweight_grace_days,
        active_mask=prep["active_mask"],
        reference_cache=prep.get("reference_cache"),
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
        n_reference_samples=n_reference_samples,
        bucket_constraints=_bucket_cons if _bucket_cons else None,
        vega_config=_vega_cfg,
    )
    opt_result = optimizer.run()
    opt_result._final_raw_min = getattr(optimizer, '_final_raw_min', None)

    # ── Optional: bootstrap robustness diagnostic ──
    if robustness_check:
        opt_result.robustness = optimizer.bootstrap_robustness()

    # ── Optional: persist a replayable run bundle (offline reproduction) ──
    if save_bundle_path:
        import dataclasses as _dc
        from functions.dispersion.run_bundle import save_run_bundle
        save_run_bundle(
            save_bundle_path,
            pnl_matrix=optimizer._orig_ts_mat,
            active_mask=prep["active_mask"],
            column_map=col_map,
            long_candidates=long_valid,
            short_candidates=short_valid,
            constraints=constraints,
            score_weights=metric_weights.to_dict(),
            seed=seed,
            missing_data_policy=config.missing_data_policy,
            adj_divs=config.adj_divs,
            reweight_grace_days=config.reweight_grace_days,
            is_cross_corridor=config.cross_corridor,
            global_cap=config.global_cap,
            global_floor=config.global_floor,
            bisect_in_ga=bisect_in_ga,
            forced_long_indices=_forced_indices if _forced_indices else None,
            n_reference_samples=n_reference_samples,
            bucket_constraints=_bucket_cons if _bucket_cons else None,
            vega_config=_vega_cfg,
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
        "active_mask": optimizer._active_mask,   # participation mask (validity ∪ grace) — for adaptive renorm
        "col_pos": optimizer._col_pos,           # key -> column index mapping (same as FINAL-RAW uses)
        "n_rows": optimizer._n_rows,
        "weight_solver": optimizer._weight_solver,
        "long_indices": list(optimizer._last_best.long_indices) if hasattr(optimizer, '_last_best') and optimizer._last_best else None,
        "long_candidates": long_valid,
        "is_cross_corridor": config.cross_corridor,
        "missing_data_policy": config.missing_data_policy,
        "variance_px_all": variance_px_all,
        "corridor_px_all": corridor_px_all,
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
    _engine_log.debug(f"[AUDIT] scoring_mode={optimizer._scoring_mode} | use_new={optimizer._use_new_scoring} | score_fn_fitted={optimizer._score_fn.is_fitted if optimizer._score_fn else 'N/A'}")

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
    # Winner backtest: only the basket's legs — non-basket candidates carried
    # weight 0 anyway, computing their rolling P&L here was pure waste.
    _basket_keys = ({k for k, _ in opt_result.long_basket}
                    | {k for k, _ in opt_result.short_basket})
    _basket_legs = [l for l in legs
                    if (l.corridor_condition_asset
                        if config.cross_corridor and l.corridor_condition_asset
                        else l.variance_asset) in _basket_keys]
    # Scored window == delivered window: with no explicit user dates the
    # optimizer scored exactly _optimizer_dates — the delivered backtest must
    # cover the same span, not the full fetched history.
    _bt_start = start_date if start_date is not None else (
        _optimizer_dates[0].date() if len(_optimizer_dates) else None)
    _bt_end = end_date if end_date is not None else (
        _optimizer_dates[-1].date() if len(_optimizer_dates) else None)
    bt_result = bt.run_from_optimization(
        variance_px=variance_px_all,
        long_basket=opt_result.long_basket,
        short_basket=opt_result.short_basket,
        legs=_basket_legs,
        corridor_px=corridor_px_all,
        start_date=_bt_start,
        end_date=_bt_end,
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
                # Align on the optimizer window dates — prefix comparison of
                # misaligned series would cry wolf.
                if _bt_ts is not None and "Result" in _bt_ts.columns:
                    b = _bt_ts["Result"].reindex(
                        pd.DatetimeIndex(_optimizer_dates)).values
                else:
                    b = np.array([])
                s_nz = s[s != 0.0]
                b_nz = b[b != 0.0] if len(b) > 0 else np.array([])
                s_min_nz = float(s_nz.min()) if len(s_nz) > 0 else 0.0
                b_min_nz = float(b_nz.min()) if len(b_nz) > 0 else 0.0
                _engine_log.debug("[NUMERIC-CHECK] solver: len=%d min=%.4f mean=%.4f min_nz=%.4f" % (len(s), s.min(), s.mean(), s_min_nz))
                _engine_log.debug("[NUMERIC-CHECK] backtest: len=%d min=%.4f mean=%.4f min_nz=%.4f" % (len(b), b.min(), b.mean(), b_min_nz))
                n = min(len(s), len(b))
                if n > 0:
                    close = np.allclose(s[:n], b[:n], atol=1e-8)
                    _engine_log.debug("[NUMERIC-CHECK] same_len=%s allclose=%s" % (len(s) == len(b), close))
                    if not close:
                        idx = np.where(~np.isclose(s[:n], b[:n], atol=1e-8))[0][:5]
                        for i in idx:
                            _engine_log.debug("  diff[%d]: solver=%.6f backtest=%.6f" % (i, s[i], b[i]))
    except Exception as _e:
        _engine_log.debug("[NUMERIC-CHECK] failed: %s" % repr(_e))

    # Return debug info in result for UI display
    opt_result._debug_info = "\n".join(debug_info) if debug_info else ""
    opt_result._preview_data = _preview_data
    opt_result._rejection_reasons = dict(getattr(optimizer, '_rejection_reasons', {}))
    opt_result._debug_table = getattr(optimizer, '_debug_table', '')
    opt_result._ref_debug_table = getattr(optimizer, '_ref_debug_table', '')
    opt_result._collapse_diagnostic = getattr(optimizer, '_collapse_diagnostic', '')

    return opt_result


# ═══════════════════════════════════════════════════════════════════════════════
# 3b. OPTIMIZE MULTI (compare several weight configs on ONE data load)
# ═══════════════════════════════════════════════════════════════════════════════

class MultiOptimizeResult:
    """Result of :func:`optimize_multi`.

    Attributes
    ----------
    results:
        One :class:`OptimizationResult` per config, in input order.
    comparison:
        One row per config: score, net strike, basket, per-metric raw values
        (``raw_<metric>``) and their percentile within THAT run's reference
        distribution (``pct_<metric>``, higher = better, in [0, 1]).
    configs:
        The weight dicts, as passed.
    """

    def __init__(self, results, comparison, configs):
        self.results = results
        self.comparison = comparison
        self.configs = configs


def _winner_raw_and_percentiles(optimizer) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Raw metric values of the delivered basket + percentile of each ACTIVE
    metric within the run's fitted reference distribution (higher=better)."""
    from functions.dispersion._optimizer import _candidate_key

    best = getattr(optimizer, "_last_best", None)
    sf = optimizer._score_fn
    if best is None or sf is None or not sf.is_fitted:
        return {}, {}
    is_xc = optimizer.is_cross_corridor
    long_pos = [optimizer._col_pos[_candidate_key(optimizer.long_candidates[i], is_xc)]
                for i in best.long_indices
                if _candidate_key(optimizer.long_candidates[i], is_xc) in optimizer._col_pos]
    short_pos = None
    short_w = None
    if best.short_indices:
        sp = [optimizer._col_pos[_candidate_key(optimizer.short_candidates[i], is_xc)]
              for i in best.short_indices
              if _candidate_key(optimizer.short_candidates[i], is_xc) in optimizer._col_pos]
        if sp:
            short_pos = sp
            short_w = best.short_weights
    net = optimizer._adaptive_net_pnl(long_pos, best.long_weights, short_pos, short_w)
    ws = (optimizer._net_strike(best.long_indices, best.long_weights,
                                best.short_indices, best.short_weights)
          if optimizer._ws_active() else None)
    ctx = optimizer._make_ctx(ws)
    raw = sf.raw_metrics(net, ctx)
    unfitted = getattr(sf, "_unfitted_metrics", set())
    metric_map = {m.name: m for m in sf.metrics}
    percentiles: Dict[str, float] = {}
    for name in sf.weights.active_names:
        v = raw.get(name)
        if v is None or not np.isfinite(v) or name in unfitted:
            continue
        percentiles[name] = float(sf.normalizer.transform(
            name, float(v), metric_map[name].higher_is_better))
    return {k: float(v) for k, v in raw.items() if np.isfinite(v)}, percentiles


def optimize_multi(
    long_df: pd.DataFrame,
    config: DispersionConfig,
    constraints: OptimizationConstraints,
    *,
    configs: List[Dict[str, float]],
    short_df: pd.DataFrame = None,
    start_date: date = None,
    end_date: date = None,
    filter_zero_hr: bool = False,
    seed: int = 0,
    forced_tickers: List[str] = None,
    excluded_tickers: List[str] = None,
    bisect_in_ga: bool = False,
    n_reference_samples: int = None,
    bucket_constraints: List = None,
    run_backtests: bool = True,
    progress_callback: Callable[[int, int, float], None] = None,
    cache_prep: bool = False,
) -> MultiOptimizeResult:
    """Run the optimizer for SEVERAL score-weight configs on ONE data load.

    Headless-friendly: Bloomberg prices are loaded once, the P&L matrix is
    built once, then each config runs a fresh (identically seeded) GA on the
    same inputs.  With the same seed, each config's result is IDENTICAL to a
    single :func:`optimize` call with that config.

    Parameters
    ----------
    configs:
        List of score_weights dicts, e.g.
        ``[{'mean_payoff': 1.0}, {'mean_payoff': 0.5, 'hit_ratio': 0.5}]``.
    run_backtests:
        Backtest each winner (True, default) — set False to skip the
        backtest step (e.g. quick comparisons).
    (other parameters: same meaning as :func:`optimize`)

    Returns
    -------
    MultiOptimizeResult
        ``.results`` (one OptimizationResult per config, ``.backtest``
        attached when run_backtests) and ``.comparison`` (one row per
        config: score, net strike, basket, raw metrics + reference
        percentiles).
    """
    if not configs:
        raise ValueError("optimize_multi: 'configs' must contain at least one "
                         "score_weights dict.")

    # ── Same cheap validation prologue as optimize() ──
    def _norm_ticker_set(lst):
        return {str(t).strip().casefold() for t in (lst or []) if str(t).strip()}

    _bucket_cons: List[BucketConstraint] = []
    for _bc in (bucket_constraints or []):
        if isinstance(_bc, BucketConstraint):
            _bucket_cons.append(_bc)
        elif isinstance(_bc, dict):
            _bucket_cons.append(BucketConstraint(**_bc))
        else:
            raise TypeError(
                f"bucket_constraints entries must be BucketConstraint or dict, "
                f"got {type(_bc).__name__}")

    _forced_set = _norm_ticker_set(forced_tickers)
    _excluded_set = _norm_ticker_set(excluded_tickers)
    _overlap = _forced_set & _excluded_set
    if _overlap:
        raise ValueError(f"Tickers cannot be both forced and excluded: {sorted(_overlap)}")
    if len(_forced_set) > constraints.max_stocks_long:
        raise ValueError(
            f"{len(_forced_set)} forced tickers > max_stocks_long="
            f"{constraints.max_stocks_long}. Raise max_stocks_long or force fewer names.")

    long_df = _normalize_df_columns(long_df)
    required = ['Variance Asset', 'Strike Mono Var Swap (%)']
    if config.cross_corridor:
        required.append('Strike Cross Corridor (%)')
        required.append('Corridor Condition Asset')
    _validate_columns(long_df, required, 'optimize_multi')
    _warn_if_decimal_strikes(long_df, 'Strike Mono Var Swap (%)', 'optimize_multi')
    if short_df is not None and not short_df.empty:
        short_df = _normalize_df_columns(short_df)

    from functions.dispersion._backtester import DispersionBacktester
    from functions.dispersion._optimizer import DispersionOptimizer
    from functions.dispersion.scoring import MetricWeights

    # ── Load + build + filter ONCE ──
    prep = _prepare_cached(
        cache_prep, long_df, config, constraints,
        short_df=short_df, start_date=start_date, end_date=end_date,
        filter_zero_hr=filter_zero_hr,
        forced_set=_forced_set, excluded_set=_excluded_set,
    )
    if prep["early_result"] is not None:
        raise RuntimeError(
            "optimize_multi: the data pipeline produced no usable universe "
            "(empty prices or too few valid candidates). Run optimize() for "
            "the detailed debug output.")

    # Shared calibration across the batch (persists across calls with cache_prep)
    _shared_ref_cache = prep.get("reference_cache")
    if _shared_ref_cache is None:
        _shared_ref_cache = {}

    results: List[OptimizationResult] = []
    rows: List[Dict] = []
    for k, weights_dict in enumerate(configs):
        metric_weights = MetricWeights(dict(weights_dict))
        optimizer = DispersionOptimizer(
            long_candidates=prep["long_valid"],
            short_candidates=prep["short_valid"],
            pnl_matrix=prep["pnl_matrix"],
            column_map=prep["col_map"],
            constraints=constraints,
            missing_data_policy=config.missing_data_policy,
            reweight_grace_days=config.reweight_grace_days,
            active_mask=prep["active_mask"],
            reference_cache=_shared_ref_cache,
            is_cross_corridor=config.cross_corridor,
            global_cap=config.global_cap,
            global_floor=config.global_floor,
            metric_weights=metric_weights,
            progress_callback=progress_callback,
            bisect_in_ga=bisect_in_ga,
            seed=seed,
            smooth_weights=False,
            smooth_eps=0.05,
            forced_long_indices=prep["forced_indices"] if prep["forced_indices"] else None,
            n_reference_samples=n_reference_samples,
            bucket_constraints=_bucket_cons if _bucket_cons else None,
        )
        result = optimizer.run()

        if run_backtests and result.long_basket:
            bt = DispersionBacktester(prep["internal_cfg"])
            _bk = ({k for k, _ in result.long_basket}
                   | {k for k, _ in result.short_basket})
            _bl = [l for l in prep["legs"]
                   if (l.corridor_condition_asset
                       if config.cross_corridor and l.corridor_condition_asset
                       else l.variance_asset) in _bk]
            result.backtest = bt.run_from_optimization(
                variance_px=prep["variance_px_all"],
                long_basket=result.long_basket,
                short_basket=result.short_basket,
                legs=_bl,
                corridor_px=prep["corridor_px_all"],
                start_date=start_date,
                end_date=end_date,
            )
        results.append(result)

        raw, pct = _winner_raw_and_percentiles(optimizer)
        row = {
            "config": ", ".join(f"{k2}={v2:g}" for k2, v2 in weights_dict.items()),
            "score": result.score,
            "net_strike_pct": result.net_strike * 100.0,
            "n_names": len(result.long_basket),
            "basket": " | ".join(f"{t} {w * 100:.1f}%" for t, w in result.long_basket),
            "scoring_signature": result.scoring_signature,
        }
        for name, v in sorted(raw.items()):
            row[f"raw_{name}"] = v
        for name, v in sorted(pct.items()):
            row[f"pct_{name}"] = v
        rows.append(row)

    comparison = pd.DataFrame(rows)
    return MultiOptimizeResult(results=results, comparison=comparison,
                               configs=[dict(c) for c in configs])


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
        Columns: Variance Asset, Weight (%), Strike Mono Var Swap (%)
        Optional: Side (long/short). Without Side, the sign of Weight (%) decides
        (+ = long, − = short).
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
    # 'Side' is optional: when absent, the SIGN of 'Weight (%)' decides
    # (+ = long, − = short). When present, Side sets the sign as before.
    required = ['Variance Asset', 'Weight (%)', 'Strike Mono Var Swap (%)']
    if config.cross_corridor:
        required.append('Strike Cross Corridor (%)')
        required.append('Corridor Condition Asset')
    _validate_columns(df, required, 'backtest')

    from functions.dispersion._backtester import DispersionDataLoader, DispersionBacktester, series_key

    # ── Step 1: Parse DataFrame ──
    legs, weights = _df_to_backtest_inputs(df, config.cross_corridor)

    # ── Cross-corridor series key (canonical helper — single source of truth) ──
    def _series_key(leg):
        return series_key(leg, config.cross_corridor)

    # ── Step 2: Load Bloomberg data ──
    internal_cfg = _to_swap_config(config, start_date=start_date)
    loader = DispersionDataLoader(internal_cfg)
    basket = BasketInput(
        long_candidates=[s for s in legs if weights[_series_key(s)] > 0],
        short_candidates=[s for s in legs if weights[_series_key(s)] < 0],
    )
    data = loader.load(basket)

    if data["variance_px"].empty:
        empty_ts = pd.DataFrame({"Long Leg": [], "Short Leg": [], "Result": []})
        return BacktestResult(timeseries=empty_ts)

    # ── Step 3: Run backtester ──
    bt = DispersionBacktester(internal_cfg)
    return bt.run(
        data["variance_px"],
        legs=data["legs"],
        weights=weights,
        corridor_px=data.get("corridor_px"),
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
