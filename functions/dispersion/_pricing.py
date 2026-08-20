from __future__ import annotations

"""
Unified Dispersion Solver
=========================
Merged from: strike_solver.py, varswap_solver.py, pricing_engine.py, pricing_instruments.py, _solver_wrapper.py
"""

# ── Public API ──────────────────────────────────────────────────────────────
# Only these names are part of the stable interface.
# Internal helpers (process_single_ticker, price_using_classes, etc.) are
# implementation details and should NOT be imported directly by consumers.
__all__ = [
    # Engine (main entry point)
    "PricingEngine",
    "PricingConfig",
    "PricingResult",
    "TickerResult",
    # FPF generation
    "build_corridor_fpf",
    # Vol swap (delegated to volswap_solver.py)
    "generate_fpf_vol",
    "solve_volswap_strike_single",
    "solve_volswap_strikes_multithreaded",
    # LSV / LCM
    "calculate_lsv_v2",
    "compute_lcm_cross_impact",
    # Classes (swap — used by non-batch fallback)
    "CrossCorridorVarianceSwap",
    # Utilities
    "calculate_payment_dates",
    "get_trading_calendar",
]

# ══════════════════════════════════════════════════════════════════════════════
# Strike Solver — core pricing functions
# ══════════════════════════════════════════════════════════════════════════════
import time
import os
import logging
import math
import threading
import numpy as np
import datetime
import pandas as pd
from typing import List, Optional, Dict, Tuple, Any, Callable
from functools import lru_cache
from dataclasses import dataclass, field

# RICs requiring extended correlation estimation period (ACEqEqPeriod=300)
SPECIAL_RICS = {"ARM.O", "3690.HK", "P911_p.DE", "700.HK"}


def _apply_special_rics_param(model_params: dict, tickers) -> dict:
    """Add ACEqEqPeriod=300 if any ticker is a special RIC."""
    if tickers and SPECIAL_RICS.intersection(tickers if isinstance(tickers, set) else set(tickers)):
        model_params["ACEqEqPeriod"] = "300"
    return model_params


from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed


# Suppress noisy library loggers

def _safe_print(*args, **kwargs):
    """Print that won't crash if Streamlit pipe is closed."""
    import builtins
    try:
        builtins.print(*args, **kwargs)
    except OSError:
        pass


for _lg in ("pricingportal.rest_client", "pricingportal.pricingportal",
            "pricingportal", "urllib3"):
    logging.getLogger(_lg).setLevel(logging.WARNING)

# ── Optional heavy imports (module loads even without portal/speq) ────────────
_PORTAL_AVAILABLE = False
BANNED_RICS = []
try:
    from speq.fpf.unified_economics_schema.fpf_schema import (
        FPFUnifiedEconomicsWrapper, mc_global_autocall_v1,
        fides_global_autocall_v1, corridorCovarianceSwap_v4,
    )
    from numpy import inf
    from fpflucid_gen.economics import ShiftOH, Just, PaymentDate
    import pricingportal as _pp_module
    from pricingportal import *  # noqa: F403
    from pricingportal import NovaIdSource
    from speq_services import avsweb
    import xlrd
    from scipy.optimize import ridder
    from fpf_builder_utils.calendar import (
        get_trading_calendar as _get_trading_calendar_raw,
        offset_date, create_schedule,
    )
    from functions.common.utils import create_mutators, compute_implied_vol

    _PORTAL_AVAILABLE = True
except (ImportError, Exception) as _import_err:
    logging.getLogger(__name__).warning(
        f"solver: optional dependencies unavailable ({_import_err}). "
        "Portal-dependent functions will fail at call time."
    )
    ridder = None
    _get_trading_calendar_raw = None
    offset_date = None
    create_schedule = None
    create_mutators = None
    avsweb = None
    compute_implied_vol = None

# ── Portal infrastructure (delegated to functions.dispersion.portal) ──────────
from functions.dispersion._portal import (
    dbg, timed, portal as _portal_fn, snap as _snap_fn,
    ensure_portal as _ensure_portal_impl,
    refresh_token as _refresh_token_impl,
    get_calendar, get_calendar_from_currency,
    get_currency_calendar, payment_dates as _payment_dates_impl,
    observation_schedule, load_instrument as _load_instrument_impl,
    preload_instruments, clear_instrument_cache,
    CURRENCY_CALENDAR_MAP, BANNED_RICS,
)


# ── Portal helpers (public API, used by pages/✅Dispersion_Optimizer.py) ──────

def get_pricing_portal():
    """Get or create the pricing portal instance."""
    _ensure_portal()
    return pricing_portal


def get_live_snap():
    """Get or create the live snap instance."""
    _ensure_portal()
    return live_snap


# ── Thin delegating helpers (public API, used by notebooks/dashboard) ─────────

@lru_cache(maxsize=256)
def get_trading_calendar(asset: str) -> str:
    """Get trading calendar for asset."""
    return get_calendar(asset)


def calculate_payment_dates(obs_date, ric_or_currency):
    """Calculate T+2 payment dates."""
    return _payment_dates_impl(obs_date, ric_or_currency)


# ── PayoutTrace RA: unfunded ZCB + expected in-corridor days ─────────────────
#
# Alternative to pricing a dedicated range-accrual instrument per corridor
# asset: the portal metric ``PayoutTraceVariableExpectation`` with
# ``DumpStateNames=var_calculatePayoff_nCorridorObs`` returns the EXPECTED
# number of in-corridor observation days directly on the EV instrument.
# Because that expectation is undiscounted (unlike the RA instrument fair
# value), the discount factor must be reintroduced explicitly:
#
#     RA = ZCB_unfunded(currency, strike_date, maturity) * E[n_corridor_obs] / n_total_obs
#
# Mono and cross legs of the same line share the corridor asset and the
# barriers, so E[n_corridor_obs] is identical for both — the metric is only
# needed once per UNIQUE corridor asset (the mono universe), cross legs reuse it.

_PAYOUT_TRACE_METRIC = "PayoutTraceVariableExpectation"
_PAYOUT_TRACE_VAR = "var_calculatePayoff_nCorridorObs"
_ZCB_CACHE: Dict[tuple, float] = {}


def _unfunded_zcb(currency: str, start_date, maturity_date, snap_name: str = None) -> float:
    """Unfunded zero-coupon bond (discount factor) at 100% reoffer.

    ``get_bullet_funding`` with ``apply_spread_adjustment=False`` and
    ``spread_override=0.0`` — i.e. the pure rates ZCB, no funding spread.
    Cached per (currency, start, maturity, snap).
    """
    key = (str(currency), str(start_date), str(maturity_date), snap_name)
    if key in _ZCB_CACHE:
        return _ZCB_CACHE[key]
    _ensure_portal()
    from fpf_builder_utils.funding import get_bullet_funding as _get_bullet_funding
    funding = _get_bullet_funding(
        pricing_portal=pricing_portal,
        start_date=start_date,
        maturity_date=maturity_date,
        currency=currency,
        product="WOBEN",
        treasury_deposit_frequency=(3, "M"),
        yield_curve_snap=snap_name or live_snap["name"],
        apply_spread_adjustment=False,
        spread_scaling_factor=1.0,   # 100% reoffer
        display_spread_adjustment=False,
        spread_override=0.0,         # unfunded
    )
    zcb = float(funding.ZCB)
    _ZCB_CACHE[key] = zcb
    return zcb


def _parse_payout_trace_value(raw) -> Optional[float]:
    """Recursively find ``var_calculatePayoff_nCorridorObs`` in a portal response
    (the metric payload is a SERIALIZED JSON string under extraResults)."""
    import json as _json
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            return _parse_payout_trace_value(_json.loads(raw))
        except (ValueError, TypeError):
            return None
    if isinstance(raw, dict):
        if _PAYOUT_TRACE_VAR in raw:
            try:
                return float(raw[_PAYOUT_TRACE_VAR])
            except (TypeError, ValueError):
                return None
        for v in raw.values():
            found = _parse_payout_trace_value(v)
            if found is not None:
                return found
        return None
    if isinstance(raw, (list, tuple)):
        for v in raw:
            found = _parse_payout_trace_value(v)
            if found is not None:
                return found
    return None


def _n_corridor_obs(instruments, valuation_date, model_context, snap_name: str,
                    price_id: str = "PayoutTraceRA") -> List[Optional[float]]:
    """Expected in-corridor observation days per instrument (aligned list).

    One pricing call for all instruments with ONLY the PayoutTrace metric —
    much cheaper than a dedicated RA instrument per corridor asset.
    """
    _ensure_portal()
    raw = pricing_portal.price(
        price_id=price_id,
        instruments=instruments,
        valuation_date=valuation_date,
        metrics=[{
            "name": _PAYOUT_TRACE_METRIC,
            "metricParameters": {"DumpStateNames": {"value": _PAYOUT_TRACE_VAR}},
        }],
        calculation_parameters={},
        model_context=model_context,
        overridden_snap_name=snap_name,
    )
    # Preferred positional extraction: results.{price_id}.PayoutTraceVariableExpectation[i]
    results = raw.get("results", {}) if isinstance(raw, dict) else {}
    entries = results.get(price_id, {}).get(_PAYOUT_TRACE_METRIC, [])
    out: List[Optional[float]] = []
    for i in range(len(instruments)):
        val = None
        if i < len(entries):
            val = _parse_payout_trace_value(entries[i])
        out.append(val)
    return out


def _ra_from_payout_trace(n_corridor_obs: float, n_total_obs: int, zcb: float) -> float:
    """RA equivalent of the legacy range-accrual instrument:
    discounted expected in-corridor day fraction."""
    return zcb * (float(n_corridor_obs) / float(n_total_obs))


# Module-level state (portal connection)
pricing_portal = None
live_snap = None


def _ensure_portal():
    """Initialize portal if not yet done. Delegates to portal module."""
    global pricing_portal, live_snap
    if pricing_portal is None:
        pricing_portal = _portal_fn()
        live_snap = _snap_fn()
    return pricing_portal, live_snap


def build_corridor_fpf(
        tickers: List[str],
        last_obs_date: datetime.date,
        strike_date: datetime.date,
        strikes: List[float],
        weights: List[float],
        low_barrier: float,
        high_barrier: float,
        is_capped: bool,
        corr_asset: str,
        currency: str,
        schedule_calendar_asset: str = None,
        use_parameters: bool = False
) -> str:
    """
    Generate an FPF string for a corridor covariance swap.
    Builds the observation schedule by intersecting two calendars:
        - asset_calendar: trading calendar of the priced asset (tickers[0])
        - corr_calendar: trading calendar of the corridor asset (corr_asset)
    The schedule uses only business days where BOTH assets trade.
    The priced asset can be a stock (ISP.MI) OR an index (.STOXX50E).
    The corridor asset can also be a stock or index.
    Parameters
    ----------
    tickers : List[str]
        Asset RIC(s) being priced. Can be stocks (e.g. ["ISP.MI"]) or indices
        (e.g. [".STOXX50E"]). First element is the primary asset used for
        calendar resolution and KO payment dates.
    last_obs_date : date
        Last observation / maturity date.
    strike_date : date
        Trade inception date (start of observation window).
    strikes : List[float]
        Strike level per asset (as decimal, e.g. 0.25).
    weights : List[float]
        Basket weight per asset (sums to 1 for single-asset).
    low_barrier / high_barrier : float
        Corridor barriers (as decimal, e.g. 0.70 / 1.30).
    is_capped : bool
        Whether legs have a cap (note-style product).
    corr_asset : str
        Corridor reference asset RIC (e.g. ".STOXX50E" for cross-corridor,
        or same as tickers[0] for mono-corridor). Can be stock or index.
    currency : str
        Pricing/payment currency (e.g. "EUR").
    schedule_calendar_asset : str, optional
        Override for which asset's calendar to use for the observation schedule.
        If None, defaults to tickers[0].
        Use cases:
          - Cross-corridor stock vs index: pass the stock RIC so schedule
            uses the stock's exchange calendar.
          - Index vs index: pass whichever index should drive the schedule.
    use_parameters : bool
        If True, uses $STOCK_STRIKE1 variables for ladder solving.
        If False, uses concrete strike values (for display/verification).
    Returns
    -------
    str
        FPF string ready for pricing portal submission.
    """
    # Sanitize all RIC inputs — strip tab-pair artifacts from AR registry
    tickers = [t.split("\t")[-1].strip() if "\t" in t else t for t in tickers]
    corr_asset = corr_asset.split("\t")[0].strip() if "\t" in corr_asset else corr_asset
    if schedule_calendar_asset and "\t" in schedule_calendar_asset:
        schedule_calendar_asset = schedule_calendar_asset.split("\t")[-1].strip()
    primary_asset = tickers[0]  # The asset being priced (stock or index)
    _t0 = time.time()
    _timings = {}
    fpf = """
    corridorCovarianceSwap_v4 (19-Jun-2026;19-Jun-2026, ([(".SPX", 1, 0.183, -1.2, 0.2745, NA, 0), ("KO.N", 1, 0.2297, 0.04, 0.34455, NA, 0), ("DIS.N", 1, 0.3262, 0.04, 0.4893, NA, 0), ("AMD.O", 1, 0.4793, 0.11, 0.71895, NA, 0), ("QCOM.O", 1, 0.4046, 0.11, 0.6069, NA, 0), ("AMZN.O", 1, 0.3606, 0.11, 0.5409, NA, 0), ("MA.N", 1, 0.2715, 0.04, 0.40725, NA, 0), ("GE.N", 1, 0.3492, 0.11, 0.5238, NA, 0), ("NOW.N", 1, 0.3987, 0.11, 0.59805, NA, 0), ("META.O", 1, 0.3968, 0.11, 0.5952, NA, 0), ("NVDA.O", 1, 0.5055, 0.11, 0.75825, NA, 0), ("WFC.N", 1, 0.3409, 0.11, 0.51135, NA, 0)], GeometricBasket, 0, NA, NA, 0, 1, 1, SumSquares, 1, False, 0.1, -0.1), 252, False, Forward, 2-Jun-2025, [(2-Jun-2025, False, 2-Jun-2025;2-Jun-2025), (3-Jun-2025, False, 3-Jun-2025;3-Jun-2025), (4-Jun-2025, False, 4-Jun-2025;4-Jun-2025), (5-Jun-2025, False, 5-Jun-2025;5-Jun-2025), (6-Jun-2025, False, 6-Jun-2025;6-Jun-2025), (9-Jun-2025, False, 9-Jun-2025;9-Jun-2025), (10-Jun-2025, False, 10-Jun-2025;10-Jun-2025), (11-Jun-2025, False, 11-Jun-2025;11-Jun-2025), (12-Jun-2025, False, 12-Jun-2025;12-Jun-2025), (13-Jun-2025, False, 13-Jun-2025;13-Jun-2025), (16-Jun-2025, False, 16-Jun-2025;16-Jun-2025), (17-Jun-2025, False, 17-Jun-2025;17-Jun-2025), (18-Jun-2025, False, 18-Jun-2025;18-Jun-2025), (19-Jun-2025, False, 19-Jun-2025;19-Jun-2025), (20-Jun-2025, False, 20-Jun-2025;20-Jun-2025), (23-Jun-2025, False, 23-Jun-2025;23-Jun-2025), (24-Jun-2025, False, 24-Jun-2025;24-Jun-2025), (25-Jun-2025, False, 25-Jun-2025;25-Jun-2025), (26-Jun-2025, False, 26-Jun-2025;26-Jun-2025), (27-Jun-2025, False, 27-Jun-2025;27-Jun-2025), (30-Jun-2025, False, 30-Jun-2025;30-Jun-2025), (1-Jul-2025, False, 1-Jul-2025;1-Jul-2025), (2-Jul-2025, False, 2-Jul-2025;2-Jul-2025), (3-Jul-2025, False, 3-Jul-2025;3-Jul-2025), (4-Jul-2025, False, 4-Jul-2025;4-Jul-2025), (7-Jul-2025, False, 7-Jul-2025;7-Jul-2025), (8-Jul-2025, False, 8-Jul-2025;8-Jul-2025), (9-Jul-2025, False, 9-Jul-2025;9-Jul-2025), (10-Jul-2025, False, 10-Jul-2025;10-Jul-2025), (11-Jul-2025, False, 11-Jul-2025;11-Jul-2025), (14-Jul-2025, False, 14-Jul-2025;14-Jul-2025), (15-Jul-2025, False, 15-Jul-2025;15-Jul-2025), (16-Jul-2025, False, 16-Jul-2025;16-Jul-2025), (17-Jul-2025, False, 17-Jul-2025;17-Jul-2025), (18-Jul-2025, False, 18-Jul-2025;18-Jul-2025), (21-Jul-2025, False, 21-Jul-2025;21-Jul-2025), (22-Jul-2025, False, 22-Jul-2025;22-Jul-2025), (23-Jul-2025, False, 23-Jul-2025;23-Jul-2025), (24-Jul-2025, False, 24-Jul-2025;24-Jul-2025), (25-Jul-2025, False, 25-Jul-2025;25-Jul-2025), (28-Jul-2025, False, 28-Jul-2025;28-Jul-2025), (29-Jul-2025, False, 29-Jul-2025;29-Jul-2025), (30-Jul-2025, False, 30-Jul-2025;30-Jul-2025), (31-Jul-2025, False, 31-Jul-2025;31-Jul-2025), (1-Aug-2025, False, 1-Aug-2025;1-Aug-2025), (4-Aug-2025, False, 4-Aug-2025;4-Aug-2025), (5-Aug-2025, False, 5-Aug-2025;5-Aug-2025), (6-Aug-2025, False, 6-Aug-2025;6-Aug-2025), (7-Aug-2025, False, 7-Aug-2025;7-Aug-2025), (8-Aug-2025, False, 8-Aug-2025;8-Aug-2025), (11-Aug-2025, False, 11-Aug-2025;11-Aug-2025), (12-Aug-2025, False, 12-Aug-2025;12-Aug-2025), (13-Aug-2025, False, 13-Aug-2025;13-Aug-2025), (14-Aug-2025, False, 14-Aug-2025;14-Aug-2025), (15-Aug-2025, False, 15-Aug-2025;15-Aug-2025), (18-Aug-2025, False, 18-Aug-2025;18-Aug-2025), (19-Aug-2025, False, 19-Aug-2025;19-Aug-2025), (20-Aug-2025, False, 20-Aug-2025;20-Aug-2025), (21-Aug-2025, False, 21-Aug-2025;21-Aug-2025), (22-Aug-2025, False, 22-Aug-2025;22-Aug-2025), (25-Aug-2025, False, 25-Aug-2025;25-Aug-2025), (26-Aug-2025, False, 26-Aug-2025;26-Aug-2025), (27-Aug-2025, False, 27-Aug-2025;27-Aug-2025), (28-Aug-2025, False, 28-Aug-2025;28-Aug-2025), (29-Aug-2025, False, 29-Aug-2025;29-Aug-2025), (1-Sep-2025, False, 1-Sep-2025;1-Sep-2025), (2-Sep-2025, False, 2-Sep-2025;2-Sep-2025), (3-Sep-2025, False, 3-Sep-2025;3-Sep-2025), (4-Sep-2025, False, 4-Sep-2025;4-Sep-2025), (5-Sep-2025, False, 5-Sep-2025;5-Sep-2025), (8-Sep-2025, False, 8-Sep-2025;8-Sep-2025), (9-Sep-2025, False, 9-Sep-2025;9-Sep-2025), (10-Sep-2025, False, 10-Sep-2025;10-Sep-2025), (11-Sep-2025, False, 11-Sep-2025;11-Sep-2025), (12-Sep-2025, False, 12-Sep-2025;12-Sep-2025), (15-Sep-2025, False, 15-Sep-2025;15-Sep-2025), (16-Sep-2025, False, 16-Sep-2025;16-Sep-2025), (17-Sep-2025, False, 17-Sep-2025;17-Sep-2025), (18-Sep-2025, False, 18-Sep-2025;18-Sep-2025), (19-Sep-2025, False, 19-Sep-2025;19-Sep-2025), (22-Sep-2025, False, 22-Sep-2025;22-Sep-2025), (23-Sep-2025, False, 23-Sep-2025;23-Sep-2025), (24-Sep-2025, False, 24-Sep-2025;24-Sep-2025), (25-Sep-2025, False, 25-Sep-2025;25-Sep-2025), (26-Sep-2025, False, 26-Sep-2025;26-Sep-2025), (29-Sep-2025, False, 29-Sep-2025;29-Sep-2025), (30-Sep-2025, False, 30-Sep-2025;30-Sep-2025), (1-Oct-2025, False, 1-Oct-2025;1-Oct-2025), (2-Oct-2025, False, 2-Oct-2025;2-Oct-2025), (3-Oct-2025, False, 3-Oct-2025;3-Oct-2025), (6-Oct-2025, False, 6-Oct-2025;6-Oct-2025), (7-Oct-2025, False, 7-Oct-2025;7-Oct-2025), (8-Oct-2025, False, 8-Oct-2025;8-Oct-2025), (9-Oct-2025, False, 9-Oct-2025;9-Oct-2025), (10-Oct-2025, False, 10-Oct-2025;10-Oct-2025), (13-Oct-2025, False, 13-Oct-2025;13-Oct-2025), (14-Oct-2025, False, 14-Oct-2025;14-Oct-2025), (15-Oct-2025, False, 15-Oct-2025;15-Oct-2025), (16-Oct-2025, False, 16-Oct-2025;16-Oct-2025), (17-Oct-2025, False, 17-Oct-2025;17-Oct-2025), (20-Oct-2025, False, 20-Oct-2025;20-Oct-2025), (21-Oct-2025, False, 21-Oct-2025;21-Oct-2025), (22-Oct-2025, False, 22-Oct-2025;22-Oct-2025), (23-Oct-2025, False, 23-Oct-2025;23-Oct-2025), (24-Oct-2025, False, 24-Oct-2025;24-Oct-2025), (27-Oct-2025, False, 27-Oct-2025;27-Oct-2025), (28-Oct-2025, False, 28-Oct-2025;28-Oct-2025), (29-Oct-2025, False, 29-Oct-2025;29-Oct-2025), (30-Oct-2025, False, 30-Oct-2025;30-Oct-2025), (31-Oct-2025, False, 31-Oct-2025;31-Oct-2025), (3-Nov-2025, False, 3-Nov-2025;3-Nov-2025), (4-Nov-2025, False, 4-Nov-2025;4-Nov-2025), (5-Nov-2025, False, 5-Nov-2025;5-Nov-2025), (6-Nov-2025, False, 6-Nov-2025;6-Nov-2025), (7-Nov-2025, False, 7-Nov-2025;7-Nov-2025), (10-Nov-2025, False, 10-Nov-2025;10-Nov-2025), (11-Nov-2025, False, 11-Nov-2025;11-Nov-2025), (12-Nov-2025, False, 12-Nov-2025;12-Nov-2025), (13-Nov-2025, False, 13-Nov-2025;13-Nov-2025), (14-Nov-2025, False, 14-Nov-2025;14-Nov-2025), (17-Nov-2025, False, 17-Nov-2025;17-Nov-2025), (18-Nov-2025, False, 18-Nov-2025;18-Nov-2025), (19-Nov-2025, False, 19-Nov-2025;19-Nov-2025), (20-Nov-2025, False, 20-Nov-2025;20-Nov-2025), (21-Nov-2025, False, 21-Nov-2025;21-Nov-2025), (24-Nov-2025, False, 24-Nov-2025;24-Nov-2025), (25-Nov-2025, False, 25-Nov-2025;25-Nov-2025), (26-Nov-2025, False, 26-Nov-2025;26-Nov-2025), (27-Nov-2025, False, 27-Nov-2025;27-Nov-2025), (28-Nov-2025, False, 28-Nov-2025;28-Nov-2025), (1-Dec-2025, False, 1-Dec-2025;1-Dec-2025), (2-Dec-2025, False, 2-Dec-2025;2-Dec-2025), (3-Dec-2025, False, 3-Dec-2025;3-Dec-2025), (4-Dec-2025, False, 4-Dec-2025;4-Dec-2025), (5-Dec-2025, False, 5-Dec-2025;5-Dec-2025), (8-Dec-2025, False, 8-Dec-2025;8-Dec-2025), (9-Dec-2025, False, 9-Dec-2025;9-Dec-2025), (10-Dec-2025, False, 10-Dec-2025;10-Dec-2025), (11-Dec-2025, False, 11-Dec-2025;11-Dec-2025), (12-Dec-2025, False, 12-Dec-2025;12-Dec-2025), (15-Dec-2025, False, 15-Dec-2025;15-Dec-2025), (16-Dec-2025, False, 16-Dec-2025;16-Dec-2025), (17-Dec-2025, False, 17-Dec-2025;17-Dec-2025), (18-Dec-2025, False, 18-Dec-2025;18-Dec-2025), (19-Dec-2025, False, 19-Dec-2025;19-Dec-2025), (22-Dec-2025, False, 22-Dec-2025;22-Dec-2025), (23-Dec-2025, False, 23-Dec-2025;23-Dec-2025), (24-Dec-2025, False, 24-Dec-2025;24-Dec-2025), (25-Dec-2025, False, 25-Dec-2025;25-Dec-2025), (26-Dec-2025, False, 26-Dec-2025;26-Dec-2025), (29-Dec-2025, False, 29-Dec-2025;29-Dec-2025), (30-Dec-2025, False, 30-Dec-2025;30-Dec-2025), (31-Dec-2025, False, 31-Dec-2025;31-Dec-2025), (1-Jan-2026, False, 1-Jan-2026;1-Jan-2026), (2-Jan-2026, False, 2-Jan-2026;2-Jan-2026), (5-Jan-2026, False, 5-Jan-2026;5-Jan-2026), (6-Jan-2026, False, 6-Jan-2026;6-Jan-2026), (7-Jan-2026, False, 7-Jan-2026;7-Jan-2026), (8-Jan-2026, False, 8-Jan-2026;8-Jan-2026), (9-Jan-2026, False, 9-Jan-2026;9-Jan-2026), (12-Jan-2026, False, 12-Jan-2026;12-Jan-2026), (13-Jan-2026, False, 13-Jan-2026;13-Jan-2026), (14-Jan-2026, False, 14-Jan-2026;14-Jan-2026), (15-Jan-2026, False, 15-Jan-2026;15-Jan-2026), (16-Jan-2026, False, 16-Jan-2026;16-Jan-2026), (19-Jan-2026, False, 19-Jan-2026;19-Jan-2026), (20-Jan-2026, False, 20-Jan-2026;20-Jan-2026), (21-Jan-2026, False, 21-Jan-2026;21-Jan-2026), (22-Jan-2026, False, 22-Jan-2026;22-Jan-2026), (23-Jan-2026, False, 23-Jan-2026;23-Jan-2026), (26-Jan-2026, False, 26-Jan-2026;26-Jan-2026), (27-Jan-2026, False, 27-Jan-2026;27-Jan-2026), (28-Jan-2026, False, 28-Jan-2026;28-Jan-2026), (29-Jan-2026, False, 29-Jan-2026;29-Jan-2026), (30-Jan-2026, False, 30-Jan-2026;30-Jan-2026), (2-Feb-2026, False, 2-Feb-2026;2-Feb-2026), (3-Feb-2026, False, 3-Feb-2026;3-Feb-2026), (4-Feb-2026, False, 4-Feb-2026;4-Feb-2026), (5-Feb-2026, False, 5-Feb-2026;5-Feb-2026), (6-Feb-2026, False, 6-Feb-2026;6-Feb-2026), (9-Feb-2026, False, 9-Feb-2026;9-Feb-2026), (10-Feb-2026, False, 10-Feb-2026;10-Feb-2026), (11-Feb-2026, False, 11-Feb-2026;11-Feb-2026), (12-Feb-2026, False, 12-Feb-2026;12-Feb-2026), (13-Feb-2026, False, 13-Feb-2026;13-Feb-2026), (16-Feb-2026, False, 16-Feb-2026;16-Feb-2026), (17-Feb-2026, False, 17-Feb-2026;17-Feb-2026), (18-Feb-2026, False, 18-Feb-2026;18-Feb-2026), (19-Feb-2026, False, 19-Feb-2026;19-Feb-2026), (20-Feb-2026, False, 20-Feb-2026;20-Feb-2026), (23-Feb-2026, False, 23-Feb-2026;23-Feb-2026), (24-Feb-2026, False, 24-Feb-2026;24-Feb-2026), (25-Feb-2026, False, 25-Feb-2026;25-Feb-2026), (26-Feb-2026, False, 26-Feb-2026;26-Feb-2026), (27-Feb-2026, False, 27-Feb-2026;27-Feb-2026), (2-Mar-2026, False, 2-Mar-2026;2-Mar-2026), (3-Mar-2026, False, 3-Mar-2026;3-Mar-2026), (4-Mar-2026, False, 4-Mar-2026;4-Mar-2026), (5-Mar-2026, False, 5-Mar-2026;5-Mar-2026), (6-Mar-2026, False, 6-Mar-2026;6-Mar-2026), (9-Mar-2026, False, 9-Mar-2026;9-Mar-2026), (10-Mar-2026, False, 10-Mar-2026;10-Mar-2026), (11-Mar-2026, False, 11-Mar-2026;11-Mar-2026), (12-Mar-2026, False, 12-Mar-2026;12-Mar-2026), (13-Mar-2026, False, 13-Mar-2026;13-Mar-2026), (16-Mar-2026, False, 16-Mar-2026;16-Mar-2026), (17-Mar-2026, False, 17-Mar-2026;17-Mar-2026), (18-Mar-2026, False, 18-Mar-2026;18-Mar-2026), (19-Mar-2026, False, 19-Mar-2026;19-Mar-2026), (20-Mar-2026, False, 20-Mar-2026;20-Mar-2026), (23-Mar-2026, False, 23-Mar-2026;23-Mar-2026), (24-Mar-2026, False, 24-Mar-2026;24-Mar-2026), (25-Mar-2026, False, 25-Mar-2026;25-Mar-2026), (26-Mar-2026, False, 26-Mar-2026;26-Mar-2026), (27-Mar-2026, False, 27-Mar-2026;27-Mar-2026), (30-Mar-2026, False, 30-Mar-2026;30-Mar-2026), (31-Mar-2026, False, 31-Mar-2026;31-Mar-2026), (1-Apr-2026, False, 1-Apr-2026;1-Apr-2026), (2-Apr-2026, False, 2-Apr-2026;2-Apr-2026), (3-Apr-2026, False, 3-Apr-2026;3-Apr-2026), (6-Apr-2026, False, 6-Apr-2026;6-Apr-2026), (7-Apr-2026, False, 7-Apr-2026;7-Apr-2026), (8-Apr-2026, False, 8-Apr-2026;8-Apr-2026), (9-Apr-2026, False, 9-Apr-2026;9-Apr-2026), (10-Apr-2026, False, 10-Apr-2026;10-Apr-2026), (13-Apr-2026, False, 13-Apr-2026;13-Apr-2026), (14-Apr-2026, False, 14-Apr-2026;14-Apr-2026), (15-Apr-2026, False, 15-Apr-2026;15-Apr-2026), (16-Apr-2026, False, 16-Apr-2026;16-Apr-2026), (17-Apr-2026, False, 17-Apr-2026;17-Apr-2026), (20-Apr-2026, False, 20-Apr-2026;20-Apr-2026), (21-Apr-2026, False, 21-Apr-2026;21-Apr-2026), (22-Apr-2026, False, 22-Apr-2026;22-Apr-2026), (23-Apr-2026, False, 23-Apr-2026;23-Apr-2026), (24-Apr-2026, False, 24-Apr-2026;24-Apr-2026), (27-Apr-2026, False, 27-Apr-2026;27-Apr-2026), (28-Apr-2026, False, 28-Apr-2026;28-Apr-2026), (29-Apr-2026, False, 29-Apr-2026;29-Apr-2026), (30-Apr-2026, False, 30-Apr-2026;30-Apr-2026), (1-May-2026, False, 1-May-2026;1-May-2026), (4-May-2026, False, 4-May-2026;4-May-2026), (5-May-2026, False, 5-May-2026;5-May-2026), (6-May-2026, False, 6-May-2026;6-May-2026), (7-May-2026, False, 7-May-2026;7-May-2026), (8-May-2026, False, 8-May-2026;8-May-2026), (11-May-2026, False, 11-May-2026;11-May-2026), (12-May-2026, False, 12-May-2026;12-May-2026), (13-May-2026, False, 13-May-2026;13-May-2026), (14-May-2026, False, 14-May-2026;14-May-2026), (15-May-2026, False, 15-May-2026;15-May-2026), (18-May-2026, False, 18-May-2026;18-May-2026), (19-May-2026, False, 19-May-2026;19-May-2026), (20-May-2026, False, 20-May-2026;20-May-2026), (21-May-2026, False, 21-May-2026;21-May-2026), (22-May-2026, False, 22-May-2026;22-May-2026), (25-May-2026, False, 25-May-2026;25-May-2026), (26-May-2026, False, 26-May-2026;26-May-2026), (27-May-2026, False, 27-May-2026;27-May-2026), (28-May-2026, False, 28-May-2026;28-May-2026), (29-May-2026, False, 29-May-2026;29-May-2026), (1-Jun-2026, False, 1-Jun-2026;1-Jun-2026), (2-Jun-2026, False, 2-Jun-2026;2-Jun-2026), (3-Jun-2026, False, 3-Jun-2026;3-Jun-2026), (4-Jun-2026, False, 4-Jun-2026;4-Jun-2026), (5-Jun-2026, False, 5-Jun-2026;5-Jun-2026), (8-Jun-2026, False, 8-Jun-2026;8-Jun-2026), (9-Jun-2026, False, 9-Jun-2026;9-Jun-2026), (10-Jun-2026, False, 10-Jun-2026;10-Jun-2026), (11-Jun-2026, False, 11-Jun-2026;11-Jun-2026), (12-Jun-2026, False, 12-Jun-2026;12-Jun-2026), (15-Jun-2026, False, 15-Jun-2026;15-Jun-2026), (16-Jun-2026, False, 16-Jun-2026;16-Jun-2026), (17-Jun-2026, False, 17-Jun-2026;17-Jun-2026), (18-Jun-2026, False, 18-Jun-2026;18-Jun-2026), (19-Jun-2026, False, 19-Jun-2026;19-Jun-2026)], (FilterOff, False, GeometricBasket, [(".SPX", 1, 0)], 1, -Infinity, StrictlyUp, 0, Infinity, StrictlyDown, 0), (False, [(".SPX", 1, 0)], GeometricBasket, CurrentDate, 1, Infinity, StrictlyUp, Spread, 0), [(0, 19-Jun-2026, 19-Jun-2026;19-Jun-2026)])
    """
    fpf_to_price = FPFUnifiedEconomicsWrapper.from_data(fpf, script_cls=...)
    _timings['parse_template'] = time.time() - _t0
    _t1 = time.time()
    condition_barrier = "FilterOnBoth"
    # matu_ex, matu_stl = calculate_payment_dates(last_obs_date, tickers[0])
    # we are using t+2 bd wrt the currency!
    matu_ex, matu_stl = calculate_payment_dates(last_obs_date, currency)
    _timings['payment_dates'] = time.time() - _t1
    _t1 = time.time()
    payment_date = PaymentDate(ex=matu_ex, stl=matu_stl)
    corr_asset_list = (
        [corridorCovarianceSwap_v4.CorridorAssets(corridorAsset=corr_asset, corridorMultiplier=1.0, corridorAssetLag=0)]
    )
    corr_def = fpf_to_price.corridorDefinition.value.clone(
        corridorConditionAssetType="GeometricBasket",
        scaleStrike=True,
        lowBarrier=low_barrier,
        highBarrier=high_barrier,
        logReturnFilterType=condition_barrier,
        corridorAssets=corr_asset_list
    )
    # ── Calendar resolution ──────────────────────────────────────────────────
    # The observation schedule is the INTERSECTION of two calendars:
    #   - asset_calendar: calendar of the priced asset (stock OR index)
    #   - corr_calendar: calendar of the corridor reference asset
    #
    # Examples:
    #   Stock vs Index:  ISP.MI (MIS) ∩ .STOXX50E (SON)
    #   Index vs Index:  .STOXX50E (SON) ∩ .SPX (NYS)
    #   Mono-corridor:   ISP.MI (MIS) ∩ ISP.MI (MIS) → just MIS
    #
    # schedule_calendar_asset overrides which asset's calendar is used for
    # the "reference" side. Useful when the caller wants to control which
    # calendar drives the schedule (e.g. use the stock's calendar in
    # cross-corridor rather than the corridor index's).
    # ─────────────────────────────────────────────────────────────────────────
    # Primary asset calendar (used for KO payment dates + one side of intersection)
    asset_calendar = get_trading_calendar(primary_asset)
    _timings['get_calendar_1'] = time.time() - _t1
    _t1 = time.time()
    # Corridor/reference calendar (other side of intersection)
    ref_asset_for_cal = schedule_calendar_asset if schedule_calendar_asset else corr_asset
    corr_calendar = get_trading_calendar(ref_asset_for_cal)
    _timings['get_calendar_2'] = time.time() - _t1
    _t1 = time.time()
    # DEBUG: Log schedule asset selection logic
    if schedule_calendar_asset:
        dbg.ok("SCHEDULE-ASSET-SELECTION",
               f"Using schedule_calendar_asset override: {schedule_calendar_asset} (corr_asset={corr_asset}, primary={primary_asset})")
    else:
        dbg.ok("SCHEDULE-ASSET-SELECTION", f"No override, using corr_asset: {corr_asset} (primary={primary_asset})")
    dbg.ok("CALENDAR-RESOLUTION",
           f"primary={primary_asset}, schedule_calendar_asset={schedule_calendar_asset}, ref_asset_for_cal={ref_asset_for_cal}, corr_asset={corr_asset}, asset_cal={asset_calendar}, corr_cal={corr_calendar}")
    dbg.ok("calendar", f"asset={primary_asset}→{asset_calendar}, corridor={corr_asset}→{corr_calendar}")
    # dbg.info("calendar", f"asset={primary_asset}→{asset_calendar}, corridor={corr_asset}→{corr_calendar}")
    # Build observation schedules.
    # The extended schedule (last_obs + 10 days) is always needed for T+2 computation.
    # If both calendars are the same, one call serves all purposes.
    _sched_start = strike_date.strftime("%Y-%m-%d")
    _sched_end = last_obs_date.strftime("%Y-%m-%d")
    _sched_end_ext = (last_obs_date + datetime.timedelta(days=10)).strftime("%Y-%m-%d")

    _extended_dates = create_schedule(_sched_start, _sched_end_ext, "1B", asset_calendar, "MF")
    # dates_asset is just the extended schedule trimmed to last_obs_date
    dates_asset = [d for d in _extended_dates if d <= _sched_end]

    if asset_calendar == corr_calendar:
        # Mono-corridor case: same calendar, no intersection needed
        dates_corr = dates_asset
    else:
        dates_corr = create_schedule(_sched_start, _sched_end, "1B", corr_calendar, "MF")
    _timings['schedules'] = time.time() - _t1
    _t1 = time.time()
    # If both calendars are the same (mono-corridor), skip intersection
    if asset_calendar == corr_calendar:

        dates_sched = dates_asset
    else:
        dates_sched = sorted(list(set(dates_asset) & set(dates_corr)))

    # DEBUG: Log schedule details
    dbg.ok("OBSERVATION-SCHEDULE",
           f"dates_sched count={len(dates_sched)}, first={dates_sched[0] if dates_sched else 'N/A'}, last={dates_sched[-1] if dates_sched else 'N/A'}, asset_cal={asset_calendar}, corr_cal={corr_calendar}")
    # Calendar intersection audit log (cross-corridor only)
    global_cap = "Nothing"
    global_floor = "Nothing"
    basket_leg_cap = 6.25 / 100 if high_barrier < 200 else "Nothing"
    if use_parameters:
        # Parameterized version for solving
        annot = fpf_to_price.varianceDetails.clone(
            varianceAssetsAndIndexLegDetails=[
                fpf_to_price.varianceDetails.varianceAssetsAndIndexLegDetails[0].clone(
                    asset=ticker,
                    basketMultiplier=1,
                    strike="$STOCK_STRIKE1",
                    legCap=Just("$CAP1") if is_capped else "Nothing",
                    legFloor="Nothing",
                    legMultiplier=weights[idx]
                ) for idx, ticker in enumerate(tickers)
            ],
            globalCap=global_cap,
            globalFloor=global_floor,
            isOptionOnVariance=True
        )
    else:
        # Concrete values version for display/verification
        # Note: strikes are already variance (squared) when passed from the solver
        annot = fpf_to_price.varianceDetails.clone(
            varianceAssetsAndIndexLegDetails=[
                fpf_to_price.varianceDetails.varianceAssetsAndIndexLegDetails[0].clone(
                    asset=ticker,
                    basketMultiplier=1,
                    strike=strikes[idx],
                    legCap=Just((2.5 ** 2 - 1) * strikes[idx]) if is_capped else "Nothing",
                    legFloor="Nothing",
                    legMultiplier=weights[idx]
                ) for idx, ticker in enumerate(tickers)
            ],
            globalCap=global_cap,
            globalFloor=global_floor,
            isOptionOnVariance=True
        )
    ko_details = fpf_to_price.koDetails.clone(
        koAssets=[corridorCovarianceSwap_v4.KoAssets(koAsset=ticker, koAssetMultiplier=1.0, koAssetLag=0) for ticker in
                  tickers]
    )
    _timings['intersection_and_annot'] = time.time() - _t1
    _t1 = time.time()
    # Compute T+2 KO payment dates locally using the schedule we already have.
    # offset_date(d, "2B", cal) = "2 business days forward on this calendar".
    # Since _extended_dates (built in parallel above) covers last_obs_date + 10 days,
    # T+2 for any observation date is simply the date 2 positions ahead.
    _asset_date_index = {d: idx for idx, d in enumerate(_extended_dates)}

    def _local_t_plus_2(date_str: str) -> str:
        """Compute T+2 business day on asset_calendar without HTTP."""
        idx = _asset_date_index.get(date_str)
        if idx is not None and idx + 2 < len(_extended_dates):
            return _extended_dates[idx + 2]
        # Fallback for dates outside our schedule range
        return offset_date(date_str, "2B", asset_calendar, "MF")

    dates_sched = [
        corridorCovarianceSwap_v4.ObservationDates(
            observationDates=datetime.date.fromisoformat(i),
            isKoDate=False,
            koPaymentDate=PaymentDate(
                ex=datetime.date.fromisoformat(_local_t_plus_2(i)),
                stl=datetime.date.fromisoformat(_local_t_plus_2(i))
            )
        ) for i in dates_sched
    ]
    _timings['obs_dates_build'] = time.time() - _t1
    _t1 = time.time()
    stream_coupons = [
        corridorCovarianceSwap_v4.StreamOfCoupons(
            amount=0.0,
            observationDate=last_obs_date,
            paymentDate=payment_date
        )
    ]
    new_fpf = fpf_to_price.clone(
        paymentDate=payment_date,
        strikeDate=strike_date,
        corridorDefinition=Just(corr_def),
        observationDates=dates_sched,
        varianceDetails=annot,
        koDetails=ko_details,
        streamOfCoupons=stream_coupons
    )
    _timings['final_clone'] = time.time() - _t1
    _timings['TOTAL'] = time.time() - _t0
    # ── Timing summary ──
    _parts = " | ".join(f"{k}={v:.3f}s" for k, v in _timings.items())
    dbg.info("build_corridor_fpf", f"[TIMING] {primary_asset} vs {corr_asset}: {_parts}")
    return new_fpf.to_fpf_string()


def calculate_lsv_v2(fpf, underlyings, premium_date, currency, df_lsv_input, eqeq_spread, eqeq_floor,
                     correl_bump_lsv_style="Relative", correl_bump_lsv=0):
    """Compute LSV impact (LSV - LV) for a single FPF. Uses shared scenario builder."""
    from functions.pricing_models.lsv import calculate_lsv_v2 as _calculate_lsv_v2
    # Reuse the shared implementation
    return _calculate_lsv_v2(
        fpf=fpf,
        underlyings=underlyings,
        premium_date=premium_date,
        currency=currency,
        df_lsv_input=df_lsv_input,
        eqeq_spread=eqeq_spread,
        eqeq_floor=eqeq_floor,
        correl_bump_lsv_style=correl_bump_lsv_style,
        correl_bump_lsv=correl_bump_lsv,
    )


def compute_lcm_cross_impact(
        fpf,
        underlyings,
        premium_date,
        currency,
        model_context,
        calculation_parameters=None,
        valuation_date=None,
        lcm_properties=None,
        price_id="lcm_cross",
        snap_name=None,
):
    """
    Price LV vs LCM on the cross-corridor leg only and return (lv, lcm, impact=lcm-lv).
    IMPORTANT: Call this ONLY on the variance-asset instrument so the corridor asset leg is untouched.
    """
    from functions.pricing_models.lcm import compute_lcm_cross_impact as _compute_lcm_cross_impact
    return _compute_lcm_cross_impact(
        fpf=fpf,
        underlyings=underlyings,
        premium_date=premium_date,
        currency=currency,
        model_context=model_context,
        calculation_parameters=calculation_parameters,
        valuation_date=valuation_date,
        lcm_properties=lcm_properties,
        price_id=price_id,
        snap_name=snap_name,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Variance Swap Solver & Batch Processing
# ══════════════════════════════════════════════════════════════════════════════


def _solve_single_ev_ra(
        ticker: str,
        corr_asset: str,
        currency: str,
        last_obs_date,
        strike_date,
        is_capped: bool,
        dvar: float,
        uvar: float,
        eqeq_lambda: float,
        correl_floor: float,
        eqfx_shift: float,
        individual_correlation: float = None,
        schedule_calendar_asset: str = None,
) -> Optional[float]:
    """
    Solve a single ticker's strike via EV/RA ratio method (1 HTTP call).
    Sends both EV-cross and RA instruments in a single price() request.
    strike = sqrt(-EV_cross / RA)
    """
    _ensure_portal()
    try:
        # Build EV-cross FPF (corridor swap at near-zero strike)
        # MUST be uncapped — cap formula is (2.5²-1)*strike, so with strike≈0
        # a cap would clamp EV to ~0 for ALL tickers, producing identical values.
        ev_fpf = build_corridor_fpf(
            tickers=[ticker], last_obs_date=last_obs_date, strike_date=strike_date,
            strikes=[0.000001], weights=[1.0], low_barrier=dvar, high_barrier=uvar,
            is_capped=False, corr_asset=corr_asset,
            schedule_calendar_asset=schedule_calendar_asset, currency=currency,
            use_parameters=False,
        )

        # Build RA FPF (range accrual variant)
        ra_fpf_base = build_corridor_fpf(
            tickers=[ticker], last_obs_date=last_obs_date, strike_date=strike_date,
            strikes=[0.000001], weights=[1.0], low_barrier=dvar, high_barrier=uvar,
            is_capped=True, corr_asset=corr_asset,
            schedule_calendar_asset=schedule_calendar_asset, currency=currency,
            use_parameters=False,
        )
        fpf_obj = FPFUnifiedEconomicsWrapper.from_data(ra_fpf_base, script_cls=corridorCovarianceSwap_v4)
        new_variance_details = fpf_obj.varianceDetails.clone(
            varianceAssetsAndIndexLegDetails=[
                fpf_obj.varianceDetails.varianceAssetsAndIndexLegDetails[0].clone(
                    asset=ticker, basketMultiplier=-1, strike=-1,
                    legCap=Just(-1), legFloor=Just(-1), legMultiplier=1.0,
                )
            ],
            isOptionOnVariance=True,
        )
        ra_fpf = fpf_obj.clone(varianceDetails=new_variance_details, optionType="Const").to_fpf_string()

        # Create instruments
        all_rics = list(set([ticker, corr_asset]))
        underlyings = [_load_instrument_impl(ric) for ric in all_rics]

        ev_instrument = pricing_portal.create_fpf(
            fpf_string=ev_fpf, instrument_ccy=currency,
            underlyings=underlyings, premium_date=datetime.datetime.now().date(),
        )
        ra_instrument = pricing_portal.create_fpf(
            fpf_string=ra_fpf, instrument_ccy=currency,
            underlyings=underlyings, premium_date=datetime.datetime.now().date(),
        )

        # Model context
        is_index = corr_asset.startswith(".") and ticker.startswith(".")
        if is_index:
            _model = "EMEA-Index-MC-LV-MultiAsset"
            model_params = {}  # Index model does NOT support ACEqFxShift
        else:
            # Skip correlation params if individual_correlation is forcing a value
            if individual_correlation is not None and individual_correlation is not True:
                model_params = {}  # Correlation forced via scenario, not model params
            else:
                model_params = {
                    'ACEqEqSpread': str(eqeq_lambda),
                    "EqEqCorrFloor": str(correl_floor),
                    "ACEqFxShift": str(eqfx_shift),
                }
            _model = "EMEA-Stocks-MC-LV-MultiAsset"
        _apply_special_rics_param(model_params, tickers)
        model_context = pricing_portal.create_model_context(_model, instrument_model_parameters=model_params)
        valuation_date = strike_date if isinstance(strike_date, datetime.datetime) else datetime.datetime.combine(
            strike_date if isinstance(strike_date, datetime.date) else datetime.datetime.strptime(str(strike_date),
                                                                                                  "%Y-%m-%d").date(),
            datetime.datetime.min.time()
        )

        # Single HTTP call with both instruments
        snap = live_snap
        metrics = [pricing_portal.create_metric("FairValue")]
        # Add correlation metric if individual_correlation is set to output mode (True)
        if individual_correlation is True:
            _excel_epoch_c = date(1899, 12, 30)
            _matu_for_corr = last_obs_date if isinstance(last_obs_date, date) else datetime.datetime.strptime(
                str(last_obs_date), "%Y-%m-%d").date()
            _corr_serial = (_matu_for_corr - _excel_epoch_c).days
            _corr_params = [
                pricing_portal.create_metric_parameter("MaturityList", str(_corr_serial)),
                pricing_portal.create_metric_parameter("MaturityType", "Absolute"),
            ]
            metrics.append(pricing_portal.create_metric("Correlation", _corr_params))

        price_kwargs = {
            "price_id": "Price",
            "instruments": [ev_instrument, ra_instrument],
            "valuation_date": valuation_date,
            "calculation_parameters": {},
            "model_context": model_context,
            "overridden_snap_name": snap["name"],
            "metrics": metrics,
        }

        # Add scenario for individual correlation override
        if individual_correlation is not None:
            scenario = pricing_portal.create_scenario_simple(
                mutator_name="GenericMutatorOverrideCorrelationEqEq",
                properties={"CorrelationLevel": individual_correlation}
            )
            price_kwargs["scenario"] = scenario

        res = pricing_portal.price(**price_kwargs)

        # Extract: [0]=EV-cross, [1]=RA
        if individual_correlation is True:
            # Output mode: extract both FairValue and correlation
            fair_values = res["results"]["Price"]["FairValue"]
            ev_cross = fair_values[0]["value"]
            ra_val = fair_values[1]["value"]
            # Extract correlation from results (assuming it's in the response)
            correlation = res["results"]["Price"].get("Correlation", [{}])[0].get("value", None)
        elif individual_correlation is not None:
            # Override mode: extract FairValue only
            ev_cross = res["results"]["Price"]["SimpleScenarioBump"][0]["FairValue"][0]["value"]
            ra_val = res["results"]["Price"]["SimpleScenarioBump"][0]["FairValue"][1]["value"]
            correlation = None
        else:
            # Normal mode: extract FairValue only
            fair_values = res["results"]["Price"]["FairValue"]
            ev_cross = fair_values[0]["value"]
            ra_val = fair_values[1]["value"]
            correlation = None

        if ra_val == 0:
            dbg.err("solve-ev-ra", f"{ticker}: RA=0, cannot compute strike")
            return None

        strike_sq = -ev_cross / ra_val
        if strike_sq < 0:
            strike_sq = abs(strike_sq)

        # Return VARIANCE (strike²) and correlation (if requested)
        # Returns variance (K²); caller computes sqrt() for strike vol.
        result = {"variance": strike_sq, "ev_cross": ev_cross, "ra": ra_val}
        if correlation is not None:
            result["correlation"] = correlation
        dbg.ok("solve-ev-ra",
               f"{ticker}: EV={ev_cross:.8f}, RA={ra_val:.8f}, strike={math.sqrt(strike_sq):.6f} ({math.sqrt(strike_sq) * 100:.2f}%)")
        return result

    except Exception as e:
        import traceback
        dbg.err("solve-ev-ra", f"{ticker}: {e}")
        traceback.print_exc()
        return None


class CrossCorridorVarianceSwap:
    # Class-level cache for EV-mono results (shared across instances in same batch)
    _ev_mono_cache: Dict[str, Any] = {}
    _ev_mono_lock = threading.Lock()

    @classmethod
    def clear_ev_mono_cache(cls):
        """Clear the EV-mono cache (call between batches)."""
        with cls._ev_mono_lock:
            cls._ev_mono_cache.clear()

    def __init__(self, ref_asset, last_obs_date, strike_date, uvar, dvar, linked_asset, is_capped,
                 strike_variance_asset=None, strike_corridor_asset=None, eqeq_lambda=0.10, correl_floor=0.0,
                 eqfx_shift=0.0,
                 individual_correlation=None, currency=None, compute_atmf=True, model_name=None):
        self.ref_asset = ref_asset
        self.last_obs_date = last_obs_date
        self.strike_date = strike_date
        self.linked_asset = linked_asset
        self.uvar = uvar
        self.dvar = dvar
        self.is_capped = is_capped
        self.compute_atmf = compute_atmf
        self.model_name = model_name
        # Initialize FPF object storage
        self.fpf_obj_cross = None
        self.fpf_obj_mono = None
        # Currency determination logic - ONLY here
        if currency is not None:
            self.currency = currency
            dbg.info("currency", f"{ref_asset}: provided={currency}")
        else:
            self.currency = self.get_currency()
            dbg.info("currency", f"{ref_asset}: autodetected={self.currency}")
        self.strike_variance_asset = strike_variance_asset
        self.strike_corridor_asset = strike_corridor_asset
        self.eqeq_lambda = eqeq_lambda
        self.correl_floor = correl_floor
        self.eqfx_shift = eqfx_shift
        # LCM/LSV fields preserved for future use - currently unused
        self.lcm_impact_ref = None
        self.lcm_lv_price_ref = None
        self.lcm_adjusted_price_ref = None
        # ATMF vol storage (populated by solve_strike or compute_atmf_volspread)
        self.ref_atmf_vol = None
        self.linked_atmf_vol = None

    def get_payment_dates(self):
        """Calculate T+2 payment dates from last observation date"""
        try:
            # Try with currency first
            matu_ex, matu_stl = calculate_payment_dates(self.last_obs_date, self.currency)
            return matu_ex, matu_stl
        except (ValueError, AttributeError) as e:
            dbg.warn("paydates", f"{self.ref_asset}: currency fail → try RIC ({e})")
            try:
                # Fallback to RIC
                matu_ex, matu_stl = calculate_payment_dates(self.last_obs_date, self.ref_asset)
                return matu_ex, matu_stl
            except (ValueError, AttributeError) as e2:
                dbg.err("paydates", f"{self.ref_asset}: both methods failed ({e2}); using last_obs_date")
                return self.last_obs_date, self.last_obs_date

    # Class-level currency cache — avoids repeated HTTP calls for same ticker across threads
    _currency_cache = {}

    def get_currency(self):
        _ensure_portal()
        if self.ref_asset in CrossCorridorVarianceSwap._currency_cache:
            return CrossCorridorVarianceSwap._currency_cache[self.ref_asset]
        underlying_information = pricing_portal.get_underlying_information(
            underlying_identifier_type="ricCode",
            underlying_identifiers=[self.ref_asset]
        )
        ccy = underlying_information.get("information").get(self.ref_asset).get("currency")
        CrossCorridorVarianceSwap._currency_cache[self.ref_asset] = ccy
        return ccy

    def solve_strike(self, lsv_params=None):
        """
        Solve strikes via EV/RA ratio (1 HTTP call per asset).
        LSV/LCM paths are disabled - use generic helper functions when needed.
        """
        dbg.step("solve", f"{self.ref_asset} (ccy={self.currency})")
        # Standard solve: EV/RA ratio method (1 HTTP call per asset)
        try:
            is_cross = self.ref_asset != self.linked_asset
            _ensure_portal()

            # Solve ref asset via EV/RA ratio
            ref_result = _solve_single_ev_ra(
                ticker=self.ref_asset, corr_asset=self.linked_asset,
                currency=self.currency, last_obs_date=self.last_obs_date,
                strike_date=self.strike_date, is_capped=self.is_capped,
                dvar=self.dvar, uvar=self.uvar, eqeq_lambda=self.eqeq_lambda,
                correl_floor=self.correl_floor, eqfx_shift=self.eqfx_shift,
                individual_correlation=self.individual_correlation,
                schedule_calendar_asset=self.linked_asset if is_cross else None,
            )
            if ref_result is None:
                dbg.err("solve", f"{self.ref_asset}: EV/RA solve failed")
                return False
            self.strike_variance_asset = ref_result["variance"]
            # Store correlation if requested
            if "correlation" in ref_result:
                self._correlation_result = ref_result["correlation"]

            # Solve linked asset (if cross-corridor)
            if is_cross:
                cache_key = (self.linked_asset, str(self.last_obs_date), str(self.strike_date),
                             self.dvar, self.uvar, self.eqeq_lambda)
                cached = self._ev_mono_cache.get(f"linked_strike_{cache_key}")
                if cached is not None:
                    self.strike_corridor_asset = cached
                else:
                    linked_strike = _solve_single_ev_ra(
                        ticker=self.linked_asset, corr_asset=self.linked_asset,
                        currency=self.currency, last_obs_date=self.last_obs_date,
                        strike_date=self.strike_date, is_capped=self.is_capped,
                        dvar=self.dvar, uvar=self.uvar, eqeq_lambda=self.eqeq_lambda,
                        correl_floor=self.correl_floor, eqfx_shift=self.eqfx_shift,
                        individual_correlation=None,
                        schedule_calendar_asset=self.ref_asset if self.ref_asset != self.linked_asset else None,
                    )
                    if linked_strike is None:
                        dbg.err("solve", f"{self.linked_asset}: linked solve failed")
                        return False
                    self.strike_corridor_asset = linked_strike["variance"]
                    with self._ev_mono_lock:
                        self._ev_mono_cache[f"linked_strike_{cache_key}"] = linked_strike
            else:
                self.strike_corridor_asset = self.strike_variance_asset
            # ATMF vols
            if self.compute_atmf:
                matu_ex, matu_stl = self.get_payment_dates()
                self.ref_atmf_vol = compute_implied_vol(self.ref_asset, matu_stl, "Forward")
                self.linked_atmf_vol = compute_implied_vol(self.linked_asset, matu_stl, "Forward")
            dbg.ok("solve",
                   f"ref={math.sqrt(self.strike_variance_asset):.4f}, linked={math.sqrt(self.strike_corridor_asset):.4f}")
            return True
        except Exception as e:
            dbg.err("solve", f"{self.ref_asset}: {e}")
            self.strike_variance_asset = None
            self.strike_corridor_asset = None
            return False

    def price_instruments(self):
        _ensure_portal()
        dbg.step("price", f"{self.ref_asset} (ccy={self.currency})")
        # Price the reference asset
        self.mid_variance_asset = price_corridor_swap(
            tickers=[self.ref_asset],
            last_obs_date=self.last_obs_date,
            strike_date=self.strike_date,
            is_capped=self.is_capped,
            dvar=self.dvar,
            uvar=self.uvar,
            strikes=[self.strike_variance_asset],
            currency=self.currency,
            weights=[1.0],
            eqeq_lambda=self.eqeq_lambda,
            correl_floor=self.correl_floor,
            eqfx_shift=self.eqfx_shift,
            corr_asset=self.linked_asset,
            individual_correlation=self.individual_correlation,
            model_name=self.model_name,
            schedule_calendar_asset=self.linked_asset if self.ref_asset != self.linked_asset else None,
        )
        dbg.ok("price", f"variance_asset: {self.mid_variance_asset}")
        dbg.note("price",
                 f"ref={self.ref_asset}, linked={self.linked_asset}, k_ref={self.strike_variance_asset}, k_linked={self.strike_corridor_asset}")
        # CHECK IF SAME TICKER - REUSE PRICE
        if self.ref_asset == self.linked_asset and self.strike_variance_asset == self.strike_corridor_asset:
            dbg.info("price", "reuse variance_asset price for corridor_asset (same asset & strike)")
            self.mid_corridor_asset = self.mid_variance_asset
        else:
            dbg.step("price", f"corridor_asset: {self.linked_asset}")
            self.mid_corridor_asset = price_corridor_swap(
                tickers=[self.linked_asset],
                last_obs_date=self.last_obs_date,
                strike_date=self.strike_date,
                is_capped=self.is_capped,
                dvar=self.dvar,
                uvar=self.uvar,
                strikes=[self.strike_corridor_asset],
                currency=self.currency,
                weights=[1.0],
                eqeq_lambda=self.eqeq_lambda,
                correl_floor=self.correl_floor,
                eqfx_shift=self.eqfx_shift,
                corr_asset=self.linked_asset,
                schedule_calendar_asset=self.ref_asset,
                individual_correlation=self.individual_correlation,
                model_name=self.model_name
            )
            dbg.ok("price", f"corridor_asset: {self.mid_corridor_asset}")

    def calculate_spread_price(self):
        if self.strike_corridor_asset is None or self.strike_variance_asset is None:
            dbg.err("spread", "cannot compute (strikes are None)")
            self.spread_price_value = None
            return None
        self.spread_price_value = self.strike_corridor_asset - self.strike_variance_asset
        return self.spread_price_value

    def price_zero_strike_uncapped(self):
        """
        Price with near-zero strike and uncapped for both legs.
        Gives 'realized variance' reference values.
        Uses strike=0.0001% (effectively zero variance, 0.01% vol squared).
        """
        _ensure_portal()
        dbg.step("zero-strike", f"{self.ref_asset}")
        ZERO_STRIKE = 0.000001  # 0.01% vol squared = 0.000001 variance (effectively zero)
        try:
            mid_va_zero = price_corridor_swap(
                tickers=[self.ref_asset],
                last_obs_date=self.last_obs_date,
                strike_date=self.strike_date,
                is_capped=False,
                dvar=self.dvar,
                uvar=self.uvar,
                strikes=[ZERO_STRIKE],
                currency=self.currency,
                weights=[1.0],
                eqeq_lambda=self.eqeq_lambda,
                correl_floor=self.correl_floor,
                eqfx_shift=self.eqfx_shift,
                corr_asset=self.linked_asset,
                individual_correlation=self.individual_correlation,
                model_name=self.model_name,
                schedule_calendar_asset=self.linked_asset if self.ref_asset != self.linked_asset else None,
            )
            if self.ref_asset == self.linked_asset:
                mid_ca_zero = mid_va_zero
            else:
                mid_ca_zero = price_corridor_swap(
                    tickers=[self.linked_asset],
                    last_obs_date=self.last_obs_date,
                    strike_date=self.strike_date,
                    is_capped=False,
                    dvar=self.dvar,
                    uvar=self.uvar,
                    strikes=[ZERO_STRIKE],
                    currency=self.currency,
                    weights=[1.0],
                    eqeq_lambda=self.eqeq_lambda,
                    correl_floor=self.correl_floor,
                    eqfx_shift=self.eqfx_shift,
                    corr_asset=self.linked_asset,
                    schedule_calendar_asset=self.ref_asset,
                    individual_correlation=self.individual_correlation,
                    model_name=self.model_name
                )
            self.zero_strike_mid_variance_asset = mid_va_zero
            self.zero_strike_mid_corridor_asset = mid_ca_zero
            dbg.ok("zero-strike", f"variance_asset={mid_va_zero}, corridor_asset={mid_ca_zero}")
            return {
                'zero_strike_mid_variance_asset': mid_va_zero,
                'zero_strike_mid_corridor_asset': mid_ca_zero
            }
        except Exception as e:
            dbg.err("zero-strike", f"failed: {e}")
            return None

    def price_fair_value(self):
        """
        Price with cap = strike = floor = 1 (100%) for both legs.
        Gives the 'fair value' of the variance swap.
        One price for mono corridor, two for cross corridor.
        """
        _ensure_portal()
        dbg.step("fair-value", f"{self.ref_asset}")
        FAIR_STRIKE = 1.0  # 100% strike = cap = floor
        try:
            mid_va_fv = price_corridor_swap(
                tickers=[self.ref_asset],
                last_obs_date=self.last_obs_date,
                strike_date=self.strike_date,
                is_capped=True,
                dvar=1.0,
                uvar=1.0,
                strikes=[FAIR_STRIKE],
                currency=self.currency,
                weights=[1.0],
                eqeq_lambda=self.eqeq_lambda,
                correl_floor=self.correl_floor,
                eqfx_shift=self.eqfx_shift,
                corr_asset=self.linked_asset,
                individual_correlation=self.individual_correlation,
                model_name=self.model_name,
                schedule_calendar_asset=self.linked_asset if self.ref_asset != self.linked_asset else None,
            )
            if self.ref_asset == self.linked_asset:
                mid_ca_fv = mid_va_fv
            else:
                mid_ca_fv = price_corridor_swap(
                    tickers=[self.linked_asset],
                    last_obs_date=self.last_obs_date,
                    strike_date=self.strike_date,
                    is_capped=True,
                    dvar=1.0,
                    uvar=1.0,
                    strikes=[FAIR_STRIKE],
                    currency=self.currency,
                    weights=[1.0],
                    eqeq_lambda=self.eqeq_lambda,
                    correl_floor=self.correl_floor,
                    eqfx_shift=self.eqfx_shift,
                    corr_asset=self.linked_asset,
                    schedule_calendar_asset=self.ref_asset,
                    individual_correlation=self.individual_correlation,
                    model_name=self.model_name
                )
            self.fair_value_mid_variance_asset = mid_va_fv
            self.fair_value_mid_corridor_asset = mid_ca_fv
            dbg.ok("fair-value", f"variance_asset={mid_va_fv}, corridor_asset={mid_ca_fv}")
            return {
                'fair_value_mid_variance_asset': mid_va_fv,
                'fair_value_mid_corridor_asset': mid_ca_fv
            }
        except Exception as e:
            dbg.err("fair-value", f"failed: {e}")
            return None

    def price_range_accrual(self):
        """
        Price as Range Accrual for both legs of the cross-corridor swap.
        Uses solved strikes from self.strike_variance_asset / self.strike_corridor_asset.
        Returns dict with 'range_accrual', 'fair_value', 'discount_factor' (from cross leg).
        Also stores both legs in self.range_accrual_ref / self.range_accrual_linked.
        """
        _ensure_portal()
        dbg.step("range-accrual", f"{self.ref_asset}")
        # Use solved strikes if available, fallback to 0.5
        strike_variance_asset = self.strike_variance_asset if self.strike_variance_asset else 0.5
        strike_corridor_asset = self.strike_corridor_asset if self.strike_corridor_asset else 0.5
        try:
            ra_ref = price_range_accrual(
                tickers=[self.ref_asset],
                last_obs_date=self.last_obs_date,
                strike_date=self.strike_date,
                dvar=self.dvar,
                uvar=self.uvar,
                currency=self.currency,
                weights=[1.0],
                eqeq_lambda=self.eqeq_lambda,
                correl_floor=self.correl_floor,
                eqfx_shift=self.eqfx_shift,
                corr_asset=self.linked_asset,
                strikes=[strike_variance_asset],
                model_name=self.model_name,
                individual_correlation=self.individual_correlation,
            )
            if self.ref_asset == self.linked_asset:
                ra_linked = ra_ref
            else:
                ra_linked = price_range_accrual(
                    tickers=[self.linked_asset],
                    last_obs_date=self.last_obs_date,
                    strike_date=self.strike_date,
                    dvar=self.dvar,
                    uvar=self.uvar,
                    currency=self.currency,
                    weights=[1.0],
                    eqeq_lambda=self.eqeq_lambda,
                    correl_floor=self.correl_floor,
                    eqfx_shift=self.eqfx_shift,
                    corr_asset=self.linked_asset,
                    strikes=[strike_corridor_asset],
                    schedule_calendar_asset=self.ref_asset,
                    model_name=self.model_name,
                    individual_correlation=self.individual_correlation,
                )
            self.range_accrual_ref = ra_ref
            self.range_accrual_linked = ra_linked
            dbg.ok("range-accrual", f"ref={ra_ref}, linked={ra_linked}")
            # Return in same format as standalone function for UI compatibility
            return {
                'range_accrual': ra_ref.get('range_accrual', 0.0),
                'fair_value': ra_ref.get('fair_value', 0.0),
                'discount_factor': ra_ref.get('discount_factor', 1.0),
            }
        except Exception as e:
            dbg.err("range-accrual", f"failed: {e}")
            return None

    def compute_atmf_volspread(self):
        if not hasattr(self, 'spread_price_value') or self.spread_price_value is None:
            dbg.err("ATMF", "cannot compute: spread price is None")
            self.vol_spread = None
            return None
        strike_spread = self.spread_price_value
        # Convert from variance to vol for spread comparison
        strike_spread_vol = math.sqrt(self.strike_corridor_asset) - math.sqrt(self.strike_variance_asset) if (
                    self.strike_corridor_asset and self.strike_variance_asset) else strike_spread
        try:
            matu_ex_t2 = None
            # Try with currency FIRST
            try:
                matu_ex_t2, matu_stl_t2 = calculate_payment_dates(self.last_obs_date, self.currency)
                dbg.ok("paydates", f"currency '{self.currency}'")
            except (ValueError, AttributeError) as e:
                dbg.warn("paydates", f"currency '{self.currency}' failed: {e}")
                matu_ex_t2 = None
            # Fallback to RIC if currency failed
            if matu_ex_t2 is None:
                try:
                    matu_ex_t2, matu_stl_t2 = calculate_payment_dates(self.last_obs_date, self.ref_asset)
                    dbg.ok("paydates", f"RIC '{self.ref_asset}'")
                except (ValueError, AttributeError) as e:
                    dbg.warn("paydates", f"RIC '{self.ref_asset}' failed: {e}")
                    matu_ex_t2 = self.last_obs_date
            # Add logging before the call
            dbg.step("ATMF", f"{self.ref_asset}: maturity {matu_ex_t2}")
            self.ref_atmf_vol = compute_implied_vol(self.ref_asset, matu_ex_t2, "Forward")
            self.linked_atmf_vol = compute_implied_vol(self.linked_asset, matu_ex_t2, "Forward")
            dbg.note("ATMF", f"ref_vol={self.ref_atmf_vol}")
            # Validate both volatilities before arithmetic
            if self.ref_atmf_vol is None or self.linked_atmf_vol is None:
                dbg.err("ATMF", f"ref_vol={self.ref_atmf_vol}, linked_vol={self.linked_atmf_vol}")
                self.vol_spread = None
                return None
            atmf_spread = self.linked_atmf_vol - self.ref_atmf_vol
            self.vol_spread = strike_spread_vol - atmf_spread
            return self.vol_spread
        except Exception as e:
            dbg.err("ATMF", f"{self.ref_asset}: {e}")
            import traceback
            dbg.note("trace", traceback.format_exc())
            self.vol_spread = None
            self.ref_atmf_vol = None
            self.linked_atmf_vol = None
            return None

    def calculate_lcm_impact(self, lcm_properties=None):
        """
        Calculate LCM impact for the cross-corridor leg only.
        Only applicable when ref_asset != linked_asset.

        NOTE: ACEqEqSpread uses LambdaPricing from lcm_properties (NOT self.eqeq_lambda).
        self.eqeq_lambda is used for solve/LSV only. For LCM, the model context
        correlation surface must be calibrated at the LambdaPricing level.

        Args:
            lcm_properties: Optional dict with LCM parameters
        Returns:
            dict with 'lv', 'lcm', 'impact' keys
        """
        if self.ref_asset == self.linked_asset:
            dbg.warn("LCM", f"{self.ref_asset}: not cross-corridor, skipping LCM")
            return None
        if self.strike_variance_asset is None:
            dbg.err("LCM", f"{self.ref_asset}: strike not computed")
            return None
        try:
            _ensure_portal()
            # Generate FPF for cross-corridor leg only
            fpf_cross = build_corridor_fpf(
                tickers=[self.ref_asset],
                last_obs_date=self.last_obs_date,
                strike_date=self.strike_date,
                strikes=[self.strike_variance_asset],
                weights=[1.0],
                low_barrier=self.dvar,
                high_barrier=self.uvar,
                is_capped=self.is_capped,
                corr_asset=self.linked_asset,
                currency=self.currency,
                use_parameters=False
            )
            # Get underlyings
            underlyings = [
                pricing_portal.load_instrument(schema=NovaIdSource.REUTERS, instrument_id=ric)
                for ric in set([self.ref_asset, self.linked_asset])
            ]
            # Create model context — use LambdaPricing for ACEqEqSpread (NOT self.eqeq_lambda)
            # LCM pricing requires the correlation surface at LambdaPricing level
            _lambda_pricing = float(lcm_properties.get('LambdaPricing', 0.36)) if lcm_properties else 0.36
            model_params = {
                'ACEqEqSpread': str(_lambda_pricing),
                "EqEqCorrFloor": str(self.correl_floor),
                "ACEqFxShift": str(self.eqfx_shift)
            }
            _apply_special_rics_param(model_params, [self.ref_asset, self.linked_asset])
            model_context = pricing_portal.create_model_context(
                "EMEA-Stocks-MC-LV-MultiAsset",
                instrument_model_parameters=model_params
            )
            # Calculate LCM impact
            lcm_result = compute_lcm_cross_impact(
                fpf=fpf_cross,
                underlyings=underlyings,
                premium_date=datetime.datetime.now().date(),
                currency=self.currency,
                model_context=model_context,
                lcm_properties=lcm_properties
            )
            # Store results
            self.lcm_lv_price_ref = lcm_result['lv']
            self.lcm_adjusted_price_ref = lcm_result['lcm']
            self.lcm_impact_ref = lcm_result['impact']
            dbg.ok("LCM",
                   f"{self.ref_asset}: LV={lcm_result['lv']:.6f}, LCM={lcm_result['lcm']:.6f}, impact={lcm_result['impact']:.6f}")
            return lcm_result
        except Exception as e:
            dbg.err("LCM", f"{self.ref_asset}: calculation failed: {e}")
            import traceback
            dbg.note("trace", traceback.format_exc())
            return None

    def count_observation_dates(self):
        """
        Count the number of observation dates using stored FPF objects.
        Returns:
        - Dictionary with observation date counts for each FPF type
        """
        if self.strike_variance_asset is None:
            return {"Cross Corridor Obs Dates": 0, "Mono Corridor Obs Dates": 0}
        try:
            # Use stored FPF objects directly
            if self.ref_asset != self.linked_asset and self.fpf_obj_mono is not None:
                # Cross-corridor case - count both FPFs
                cross_count = len(self.fpf_obj_cross.observationDates) if self.fpf_obj_cross else 0
                mono_count = len(self.fpf_obj_mono.observationDates) if self.fpf_obj_mono else 0
                return {
                    "Cross Corridor Obs Dates": cross_count - 1,
                    # fpf includes today for calcs but should not be included in the
                    "Mono Corridor Obs Dates": mono_count - 1
                }
            else:
                # Regular corridor case - single FPF
                count = len(self.fpf_obj_cross.observationDates) if self.fpf_obj_cross else 0
                return {"Obs Dates": count}
        except Exception as e:
            dbg.err("obs-dates", f"{self.ref_asset}: {e}")
            return {"Cross Corridor Obs Dates": 0, "Mono Corridor Obs Dates": 0}

    def generate_fpf_string(self):
        """Generate FPF string(s) with computed strikes and store FPF objects"""
        if self.strike_variance_asset is None:
            return "Strike not computed"
        # Generate cross-corridor FPF (ref asset with linked asset as corridor condition)
        fpf_string_cross = build_corridor_fpf(
            tickers=[self.ref_asset],
            last_obs_date=self.last_obs_date,
            strike_date=self.strike_date,
            strikes=[self.strike_variance_asset],
            weights=[1.0],
            low_barrier=self.dvar,
            high_barrier=self.uvar,
            is_capped=self.is_capped,
            corr_asset=self.linked_asset,
            currency=self.currency,
            use_parameters=False
        )
        # Parse and store the FPF object
        try:
            self.fpf_obj_cross = FPFUnifiedEconomicsWrapper.from_data(
                fpf_string_cross,
                script_cls=corridorCovarianceSwap_v4
            )
        except Exception as e:
            dbg.warn("FPF", f"cross parse failed for {self.ref_asset}: {e}")
            self.fpf_obj_cross = None
        # If cross-corridor (ref_asset != linked_asset), also generate mono-corridor FPF
        if self.ref_asset != self.linked_asset and self.strike_corridor_asset is not None:
            fpf_string_mono = build_corridor_fpf(
                tickers=[self.linked_asset],
                last_obs_date=self.last_obs_date,
                strike_date=self.strike_date,
                strikes=[self.strike_corridor_asset],
                weights=[1.0],
                low_barrier=self.dvar,
                high_barrier=self.uvar,
                is_capped=self.is_capped,
                corr_asset=self.linked_asset,
                schedule_calendar_asset=self.ref_asset,
                currency=self.currency,
                use_parameters=False
            )
            # Parse and store the mono FPF object
            try:
                self.fpf_obj_mono = FPFUnifiedEconomicsWrapper.from_data(
                    fpf_string_mono,
                    script_cls=corridorCovarianceSwap_v4
                )
            except Exception as e:
                dbg.warn("FPF", f"mono parse failed for {self.linked_asset}: {e}")
                self.fpf_obj_mono = None
            return {"Cross Corridor FPF": fpf_string_cross, "Mono Corridor FPF": fpf_string_mono}
        else:
            # Regular corridor case - only one FPF
            return fpf_string_cross

    def _build_fpf_dict(self, fpf_strings, lsv_params=None, lcm_params=None):
        """
        Build FPF dictionary with clear naming: {key: fpf_string}

        Keys follow pattern: {Type} {Mode} where:
        - Type: 'Cross' or 'Mono'
        - Mode: 'Uncapped' or 'Cap' (depending on is_capped)
        - Variants: 'LSV', 'LCM' added when enabled

        Example (capped + LSV + LCM):
        {
            'Cross Uncapped': '...',
            'Cross LSV Uncapped': '...',
            'Cross LCM Uncapped': '...',
            'Cross Cap': '...',
            'Cross LSV Cap': '...',
            'Cross LCM Cap': '...',
            'Mono Uncapped': '...',
            'Mono Cap': '...',
        }
        """
        is_capped = self.is_capped
        fpf_dict = {}

        cross_fpf = fpf_strings.get("Cross Corridor FPF", fpf_strings)
        mono_fpf = fpf_strings.get("Mono Corridor FPF", "")

        # Cross corridor FPFs
        if is_capped:
            fpf_dict['Cross Cap'] = cross_fpf
        else:
            fpf_dict['Cross Uncapped'] = cross_fpf

        # LSV variants (only for cross corridor)
        if lsv_params and lsv_params.get('enabled') and hasattr(self, 'lsv_fpf_cross') and self.lsv_fpf_cross:
            if is_capped:
                fpf_dict['Cross LSV Cap'] = self.lsv_fpf_cross
            else:
                fpf_dict['Cross LSV Uncapped'] = self.lsv_fpf_cross

        # LCM variants (only for cross corridor)
        if lcm_params and lcm_params.get('enabled') and hasattr(self, 'lcm_fpf_cross') and self.lcm_fpf_cross:
            if is_capped:
                fpf_dict['Cross LCM Cap'] = self.lcm_fpf_cross
            else:
                fpf_dict['Cross LCM Uncapped'] = self.lcm_fpf_cross

        # Mono corridor FPFs
        if mono_fpf:
            if is_capped:
                fpf_dict['Mono Cap'] = mono_fpf
            else:
                fpf_dict['Mono Uncapped'] = mono_fpf

        return fpf_dict

# ══════════════════════════════════════════════════════════════════════════════
# Re-exports from engine.py + vol swap convenience wrappers
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# PricingEngine — batch solver/pricer (formerly engine.py)
# ══════════════════════════════════════════════════════════════════════════════


class _Quiet:
    """No-op context manager. display_results=False handles stdout suppression."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


_quiet = _Quiet
# ─── Vol Swap Constants ──────────────────────────────────────────────────────
# Vol swap logic extracted to volswap_solver.py — import mixin and public functions
from functions.dispersion._volswap import (
    VolSwapMixin,
    generate_fpf_vol,
    generate_fpf_vol_with_dates,
    create_and_price_fpf,
    solve_volswap_strike_single,
    solve_volswap_strikes_multithreaded,
    _VOLSWAP_STRIKE_WINDOWS,
    _get_volswap_base_fpf,
)


# ─── Data Models ─────────────────────────────────────────────────────────────
@dataclass
class PricingConfig:
    """All parameters for a pricing run. Immutable after creation."""
    strike_date: date
    last_obs_date: date
    uvar: float = 2.5
    dvar: float = -2.5
    is_solve: bool = True
    is_capped: bool = True
    is_cross_corridor: bool = True
    eqeq_lambda: float = 0.10
    correl_floor: float = 0.0
    eqfx_shift: float = -0.05
    correl_input_method: str = "Global Parameters"
    vol_mode: str = "ATMF"  # "OFF" | "ATMF" | "ATMS" | "ATMF+ATMS"
    compute_zero_strike: bool = False
    max_workers: int = 30
    model_name: Optional[str] = None
    lsv_params: Optional[dict] = None
    lcm_params: Optional[dict] = None
    individual_correlation: Optional[float] = None
    use_lsv_cross_ev: bool = False
    lsv_correl_bump: float = 0
    lsv_correl_bump_style: str = "Relative"

@dataclass
class TickerResult:
    """Result for a single ticker."""
    ticker: str
    corridor_asset: str
    success: bool
    strike: Optional[float] = None
    strike_variance_asset: Optional[float] = None  # vol level (e.g. 0.2168), not variance
    strike_corridor_asset: Optional[float] = None  # vol level (e.g. 0.2168), not variance
    ev_cross: Optional[Union[float, str]] = None
    range_accrual: Optional[Union[float, str]] = None
    range_accrual_mono: Optional[Union[float, str]] = None
    ev_mono: Optional[Union[float, str]] = None
    mid_variance_asset: Optional[float] = None
    mid_corridor_asset: Optional[float] = None
    spread_price: Optional[float] = None
    currency: Optional[str] = None
    obs_dates_cross: Optional[int] = None
    obs_dates_mono: Optional[int] = None
    atmf_vol_variance_asset: Optional[float] = None
    atmf_vol_corridor_asset: Optional[float] = None
    atms_vol_variance_asset: Optional[float] = None
    atms_vol_corridor_asset: Optional[float] = None
    vol_spread: Optional[float] = None
    ev_cross_lsv: Optional[Union[float, str]] = None
    ev_mono_lsv_adjusted: Optional[float] = None  # EV_mono_LV + LSV_impact (raw, for debug)
    strike_lsv: Optional[float] = None  # mono LSV strike
    strike_cross_lsv: Optional[float] = None  # cross LSV strike
    fpf_string_cross: Optional[str] = None
    fpf_string_mono: Optional[str] = None
    fpf_string_lsv: Optional[str] = None
    fpf_string_lcm: Optional[str] = None
    fpf_string_cap_lv: Optional[str] = None
    fpf_string_cap_lsv: Optional[str] = None
    fpf_string_cap_lcm: Optional[str] = None
    fpf_string_cap_mono: Optional[str] = None
    fpf_string_cap_lsv_mono: Optional[str] = None  # mono LSV capped FPF (separate from cross)
    # Raw EV/RA components for LSV debug
    ev_mono_lsv: Optional[float] = None  # EV_mono under LSV bump
    ev_mono_lsv0: Optional[float] = None  # EV_mono under LSV0 bump
    zero_strike_mid_variance_asset: Optional[float] = None
    zero_strike_mid_corridor_asset: Optional[float] = None
    lsv_charge: Optional[float] = None
    lcm_impact: Optional[float] = None
    correlation: Optional[float] = None
    # Price-mode LSV fields (FV under 3 model configs, same user-provided strike)
    mid_variance_asset_lv: Optional[float] = None  # FV_LV (variance asset)
    mid_variance_asset_lsv0: Optional[float] = None  # FV_LSV0 (variance asset)
    mid_variance_asset_lsv: Optional[float] = None  # FV_LSV (variance asset)
    # LCM fields
    mid_variance_asset_lcm: Optional[float] = None  # FV_LCM (variance asset, price mode)
    strike_variance_asset_lcm: Optional[float] = None  # sqrt(-EV_LCM / RA) (solve mode)
    ev_cross_lcm: Optional[Union[float, str]] = None  # raw EV_LCM (solve mode)
    # Cap adjustment fields (solve mode)
    strike_cap_adjusted: Optional[float] = None  # analytical proxy cap-adjusted strike (vol level)
    cap_impact_bp: Optional[float] = None  # cap theoretical impact in bp
    # Priced capped strikes (Phase 2b: EV_capped / RA using theoretical cap)
    strike_cap_priced_lv: Optional[float] = None  # real priced capped LV strike (vol level)
    strike_cap_priced_lsv: Optional[float] = None  # real priced capped CROSS LSV strike (vol level)
    strike_cap_priced_lsv_mono: Optional[float] = None  # real priced capped MONO LSV strike (vol level)
    strike_cap_priced_lcm: Optional[float] = None  # real priced capped LCM strike (vol level)
    strike_cap_priced_mono: Optional[float] = None  # real priced capped Mono strike (vol level)
    # Capped EV components (for transparency)
    ev_cap_cross_lv: Optional[float] = None  # capped EV cross under LV
    ev_cap_cross_lsv0: Optional[float] = None  # capped EV cross under LSV0
    ev_cap_cross_lsv: Optional[float] = None  # capped EV cross under LSV
    ev_cap_cross_lcm: Optional[float] = None  # capped EV cross under LCM
    ev_cap_mono_lv: Optional[float] = None  # capped EV mono under LV
    ev_cap_mono_lsv0: Optional[float] = None  # capped EV mono under LSV0
    ev_cap_mono_lsv: Optional[float] = None  # capped EV mono under LSV
    error: Optional[str] = None

@dataclass
class PricingResult:
    """Aggregated result from a batch pricing run."""
    success: bool
    results_df: Optional[pd.DataFrame] = None
    ticker_results: List[TickerResult] = field(default_factory=list)
    swap_objects: List[Any] = field(default_factory=list)
    successful_tickers: List[str] = field(default_factory=list)
    failed_tickers: List[str] = field(default_factory=list)
    error: Optional[str] = None

    # Backward-compat dict access for Streamlit page

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __contains__(self, key):
        return hasattr(self, key)

    def __bool__(self):
        return True


# ─── Cap Adjustment (5x5 Factor Grid) ────────────────────────────────────────

# Model constants
_CAP_LAMBDA_EU = 0.516
_CAP_LAMBDA_US = 1.000  # provisional

# Grid axes
_CAP_CORR_CENTERS = [0.3559, 0.4987, 0.6005, 0.6742, 0.7410]
_CAP_VOL_CENTERS = [21.745, 26.630, 30.520, 36.085, 54.525]  # in vol points (%)

# Tail-adjusted 5x5 factor grid: rows=correlation, cols=corridor vol
_CAP_FACTOR_GRID = [
    [44.72, 44.31, 43.89, 43.30, 38.32],
    [33.77, 33.45, 33.13, 32.82, 28.03],
    [27.58, 27.32, 27.06, 20.82, 17.30],
    [18.29, 15.37, 15.22, 15.08, 9.98],
    [13.51, 13.38, 9.97, 9.88, 7.58],
]


def _bilinear_interpolate(x: float, y: float, x_grid: list, y_grid: list, z_grid: list) -> float:
    """Bilinear interpolation on a 2D grid with clamping to boundaries.
    x = correlation, y = corridor vol (%), z_grid[row=x_idx][col=y_idx]."""
    # Clamp to grid range
    x = max(x_grid[0], min(x, x_grid[-1]))
    y = max(y_grid[0], min(y, y_grid[-1]))

    # Find bracketing indices for x
    ix = 0
    for i in range(len(x_grid) - 1):
        if x_grid[i + 1] >= x:
            ix = i
            break
    else:
        ix = len(x_grid) - 2

    # Find bracketing indices for y
    iy = 0
    for i in range(len(y_grid) - 1):
        if y_grid[i + 1] >= y:
            iy = i
            break
    else:
        iy = len(y_grid) - 2

    # Interpolation weights
    x0, x1 = x_grid[ix], x_grid[ix + 1]
    y0, y1 = y_grid[iy], y_grid[iy + 1]
    tx = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
    ty = (y - y0) / (y1 - y0) if y1 != y0 else 0.0

    # Bilinear
    z00 = z_grid[ix][iy]
    z01 = z_grid[ix][iy + 1]
    z10 = z_grid[ix + 1][iy]
    z11 = z_grid[ix + 1][iy + 1]

    return (z00 * (1 - tx) * (1 - ty) +
            z10 * tx * (1 - ty) +
            z01 * (1 - tx) * ty +
            z11 * tx * ty)


def compute_cap_adjusted_strike(
        k_raw: float,
        ra: float,
        correlation: float,
        corridor_vol_pct: float,
        is_capped: bool,
        region: str = "EU",
) -> tuple:
    """Compute cap-adjusted strike for LV cross-corridor.

    Args:
        k_raw: raw strike in decimal (e.g. 0.22 for 22%)
        ra: RA Cross value (as decimal, e.g. 0.85)
        correlation: EQ-EQ correlation (e.g. 0.55)
        corridor_vol_pct: ATMF vol corridor asset in % (e.g. 28.5)
        is_capped: whether the product is capped
        region: "EU" or "US"

    Returns:
        (strike_cap_adjusted, cap_impact_bp)
        Both None if not capped or inputs missing.
    """
    if not is_capped:
        return k_raw, 0.0

    if k_raw is None or ra is None or correlation is None or corridor_vol_pct is None:
        return None, None
    if ra == 0 or k_raw == 0:
        return None, None

    # Bilinear interpolation on the factor grid
    factor = _bilinear_interpolate(
        correlation, corridor_vol_pct,
        _CAP_CORR_CENTERS, _CAP_VOL_CENTERS, _CAP_FACTOR_GRID,
    )

    # base_correction_bp = RA * factor (RA as decimal, e.g. 0.85)
    base_correction_bp = abs(ra) * factor

    # Regional lambda
    lambda_region = _CAP_LAMBDA_EU if region.upper() in ("EU", "EUR") else _CAP_LAMBDA_US

    # Final correction
    final_correction_bp = max(0.0, lambda_region * base_correction_bp)

    # Theoretical impact X
    X = final_correction_bp / 10000.0

    # Cap-adjusted strike: K_cap = K_raw - X / (2 * K_raw * RA)
    # RA here is the decimal value (e.g. 0.85)
    k_cap_adjusted = k_raw - X / (2.0 * k_raw * abs(ra))

    return k_cap_adjusted, final_correction_bp


# ─── Caching Layer ───────────────────────────────────────────────────────────

class _PortalCache:
    """
    Thread-safe cache for expensive portal lookups within a pricing run.
    Avoids redundant HTTP calls for currencies, instruments, linked strikes, ATMF vols.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._currencies: Dict[str, str] = {}
        self._instruments: Dict[str, Any] = {}
        self._linked_strikes: Dict[str, float] = {}
        self._atmf_vols: Dict[Tuple[str, date], Optional[float]] = {}
        self._calendars: Dict[str, List[str]] = {}  # ticker -> sorted obs dates
        self._portal = None
        self._snap = None

    @property
    def portal(self):
        if self._portal is None:
            self._portal = get_pricing_portal()
            self._snap = get_live_snap()
        return self._portal

    @property
    def snap(self):
        if self._snap is None:
            self._portal = get_pricing_portal()
            self._snap = get_live_snap()
        return self._snap

    def get_currency(self, ticker: str) -> str:
        with self._lock:
            if ticker in self._currencies:
                return self._currencies[ticker]
        # HTTP call outside lock
        with _quiet():
            info = self.portal.get_underlying_information(
                underlying_identifier_type="ricCode",
                underlying_identifiers=[ticker]
            )
        currency = info.get("information", {}).get(ticker, {}).get("currency", "EUR")
        with self._lock:
            self._currencies[ticker] = currency
        return currency

    def set_currency(self, ticker: str, currency: str):
        with self._lock:
            self._currencies[ticker] = currency

    def preload_instruments(self, tickers: List[str]):
        """Preload all instruments in parallel. Call before pricing threads.
        Uses module-level portal cache so instruments persist across PricingEngine instances.
        """
        from pricingportal import NovaIdSource
        from functions.dispersion._portal import _instrument_cache as _global_instr_cache, \
            _instrument_lock as _global_instr_lock
        # Check both local dict and global module cache
        missing = [t for t in set(tickers)
                   if t not in self._instruments and t not in _global_instr_cache]
        if not missing:
            # Sync global → local for fast access
            with _global_instr_lock:
                for t in set(tickers):
                    if t in _global_instr_cache and t not in self._instruments:
                        self._instruments[t] = _global_instr_cache[t]
            return
        dbg.step("preload", f"loading {len(missing)} instruments (parallel)")
        t0 = time.time()

        def _load_one(t):
            try:
                instr = self.portal.load_instrument(
                    schema=NovaIdSource.REUTERS, instrument_id=t
                )
                with self._lock:
                    self._instruments[t] = instr
                # Also populate global cache so future PricingEngine instances skip HTTP
                with _global_instr_lock:
                    _global_instr_cache[t] = instr
            except Exception as e:
                dbg.err("preload", f"{t}: {e}")

        # Parallel instrument loading — each is a fast local+network call
        with ThreadPoolExecutor(max_workers=min(len(missing), 20)) as ex:
            list(ex.map(_load_one, missing))
        # Sync any previously-cached ones into local dict
        with _global_instr_lock:
            for t in set(tickers):
                if t in _global_instr_cache and t not in self._instruments:
                    self._instruments[t] = _global_instr_cache[t]
        dbg.ok("preload", f"{len(missing)} instruments in {time.time() - t0:.1f}s")

    def get_instrument(self, ticker: str):
        with self._lock:
            return self._instruments.get(ticker)

    def get_atmf_vol(self, ticker: str, maturity: date, anchor: str = "Forward") -> Tuple[bool, Optional[float]]:
        key = (ticker, maturity, anchor)
        with self._lock:
            if key in self._atmf_vols:
                return True, self._atmf_vols[key]
        return False, None

    def set_atmf_vol(self, ticker: str, maturity: date, vol: Optional[float], anchor: str = "Forward"):
        with self._lock:
            self._atmf_vols[(ticker, maturity, anchor)] = vol

# ─── Pricing Engine ──────────────────────────────────────────────────────────

class PricingEngine(VolSwapMixin):
    """
    High-performance pricing engine for corridor variance swaps.
    Key optimizations:
    1. Pre-solves shared linked assets once (20 tickers on .SSMI → 1 solve, not 20)
    2. Preloads all instruments before threading (avoids per-ticker HTTP)
    3. Two-pass solver: coarse 90-scenario scan → fine 20-scenario refine
       (vs old: 446 scenarios per window, up to 2 windows = 800 scenarios)
    4. Caches currencies, calendars, ATMF vols across the run
    5. Thread-safe fd-level stdout suppression (no more corrupted output)
    """

    def __init__(self, config: PricingConfig):
        self.config = config
        self.cache = _PortalCache()

    def run(
            self,
            tickers_df: pd.DataFrame,
            progress_callback: Optional[Callable] = None,
            charge_function: Optional[Callable] = None,
    ) -> PricingResult:
        """
        Run pricing for all tickers.
        Args:
            tickers_df: DataFrame with columns: 'Tickers', optionally
                       'Corridor Condition Asset', 'Currency', 'Correlation',
                       'Strike Cross (%)', 'Strike Mono (%)', 'Strikes (%)'
            progress_callback: Optional callback(dict) for UI updates.
            charge_function: Optional charge/fee function.
        Returns:
            PricingResult with results_df, swap_objects, etc.
        """
        cfg = self.config
        t_start = time.time()
        _timers = {}  # Collect timing data

        # ── Phase 0: Ensure portal connection is alive ──
        try:
            from functions.dispersion._portal import ensure_portal, refresh_token
            ensure_portal()
            refresh_token()
            CrossCorridorVarianceSwap.clear_ev_mono_cache()
        except Exception as e:
            dbg.warn("engine", f"portal refresh failed: {e}")
        # ── Phase 1: Pre-populate caches ──
        t_phase1 = time.time()
        all_tickers = set(tickers_df['Tickers'].tolist())
        if cfg.is_cross_corridor and 'Corridor Condition Asset' in tickers_df.columns:
            all_tickers.update(tickers_df['Corridor Condition Asset'].tolist())
        # Set known currencies from input
        for _, row in tickers_df.iterrows():
            if 'Currency' in row and row.get('Currency'):
                self.cache.set_currency(row['Tickers'], row['Currency'])
                if cfg.is_cross_corridor and 'Corridor Condition Asset' in row:
                    self.cache.set_currency(row['Corridor Condition Asset'], row['Currency'])
        # Batch-fetch ALL unknown currencies in 1 HTTP call (not per-ticker)
        t_ccy = time.time()
        unknown_ccy = [t for t in all_tickers if t not in self.cache._currencies]
        if unknown_ccy:
            try:
                info = self.cache.portal.get_underlying_information(
                    underlying_identifier_type="ricCode",
                    underlying_identifiers=unknown_ccy
                )
                info_map = info.get("information", {})
                for t in unknown_ccy:
                    if t in info_map and info_map[t].get("currency"):
                        self.cache.set_currency(t, info_map[t]["currency"])
                dbg.ok("preload", f"batch currencies: {len(info_map)} in 1 call")
            except Exception as e:
                dbg.warn("preload", f"batch currency failed: {e}")
        _timers['batch_currencies'] = time.time() - t_ccy
        # Preload instruments (note: load_instrument is just local object creation, fast)
        t_instr = time.time()
        self.cache.preload_instruments(list(all_tickers))
        _timers['preload_instruments'] = time.time() - t_instr
        _timers['phase1_total'] = time.time() - t_phase1
        # ── Phase 2: (reserved — linked assets solved in Phase 3 now) ──
        dbg.step("engine", f"setup done in {time.time() - t_start:.2f}s, starting {len(tickers_df)} tickers")
        # ── Phase 2: Solve/Price ──
        task_items = list(tickers_df.iterrows())
        total = len(task_items)
        indexed_results = {}
        try:
            t_solve_start = time.time()
            if progress_callback:
                try:
                    progress_callback(
                        {'status': 'phase', 'message': 'Building FPFs...', 'completed': 0, 'total': total})
                except Exception:
                    pass
            # ── Solve: single batch call for all tickers (1 HTTP request) ──
            # Both cross-corridor AND mono corridor can use the batch solver
            if cfg.is_solve:
                indexed_results = self._solve_batch_ev_ra(tickers_df, progress_callback)
            elif len(task_items) > 1:
                # Pricing mode with multiple tickers: batch all instruments in fewer HTTP calls
                indexed_results = self._price_batch(tickers_df, progress_callback)
            else:
                # Single ticker: use legacy per-ticker path
                for idx, row in task_items:
                    try:
                        result, swap_obj = self._process_ticker(row)
                        indexed_results[idx] = (result, swap_obj)
                    except Exception as e:
                        ticker = row.get('Tickers', 'unknown')
                        indexed_results[idx] = (TickerResult(
                            ticker=ticker, corridor_asset=ticker,
                            success=False, error=str(e)
                        ), None)
                    if progress_callback:
                        try:
                            progress_callback({'ticker': row.get('Tickers', ''), 'status': 'completed',
                                               'completed': idx + 1, 'total': total, 'message': ''})
                        except Exception:
                            pass
            dbg.ok("engine", f"all {total} tickers done in {time.time() - t_solve_start:.1f}s")
            _timers['solve_phase_total'] = time.time() - t_solve_start
            # === TIMING SUMMARY ===
            _safe_print("\n" + "=" * 70)
            _safe_print("  PRICING ENGINE TIMING SUMMARY")
            _safe_print("=" * 70)
            _safe_print(f"  Phase 1 (setup):           {_timers.get('phase1_total', 0):.2f}s")
            _safe_print(f"    - Batch currencies:      {_timers.get('batch_currencies', 0):.2f}s")
            _safe_print(f"    - Preload instruments:   {_timers.get('preload_instruments', 0):.2f}s")
            _safe_print(f"  Phase 2 (solve/price):     {_timers.get('solve_phase_total', 0):.2f}s")
            bt = getattr(self, '_batch_timings', {})
            if bt:
                _safe_print(f"    - FPF build:             {bt.get('fpf_build', 0):.2f}s")
                _safe_print(f"    - Load underlyings:      {bt.get('underlyings', 0):.2f}s")
                _safe_print(f"    - Create instruments:    {bt.get('instruments', 0):.2f}s")
                _safe_print(f"    - HTTP price call:       {bt.get('http_price', 0):.2f}s")
                _safe_print(
                    f"    - Post-process:          {_timers.get('solve_phase_total', 0) - bt.get('fpf_build', 0) - bt.get('underlyings', 0) - bt.get('instruments', 0) - bt.get('http_price', 0):.2f}s")
            _safe_print(f"  TOTAL:                     {time.time() - t_start:.2f}s")
            _safe_print("=" * 70 + "\n")
        except Exception as e:
            dbg.err("engine", f"execution failed: {e}")
            import traceback
            try:
                traceback.print_exc()
            except OSError:
                pass  # Streamlit pipe closed
            return PricingResult(success=False, error=str(e))
        results = []
        swap_objects = []
        for i in sorted(indexed_results.keys()):
            tr, so = indexed_results[i]
            results.append(tr)
            swap_objects.append(so)
        # Apply charge function
        if charge_function and cfg.is_solve and cfg.is_cross_corridor and cfg.vol_mode != "OFF":
            self._apply_charges(results, swap_objects, charge_function)
        successful = [r.ticker for r in results if r.success]
        failed = [r.ticker for r in results if not r.success]
        results_df = self._build_results_df(results)
        # Final progress update at 100%
        if progress_callback:
            try:
                progress_callback({
                    'ticker': 'done',
                    'status': 'completed',
                    'completed': len(successful) + len(failed),
                    'total': len(successful) + len(failed),
                    'message': 'Pricing completed'
                })
            except Exception:
                pass  # Ignore Streamlit context errors
        dbg.ok("engine", f"done: {len(successful)} ok, {len(failed)} failed in {time.time() - t_start:.1f}s")
        # Aggregate error message when all tickers fail
        error_msg = None
        if not successful and failed:
            ticker_errors = [f"{r.ticker}: {r.error}" for r in results if r.error]
            if ticker_errors:
                error_msg = f"All {len(failed)} ticker(s) failed. " + "; ".join(ticker_errors[:5])
                if len(ticker_errors) > 5:
                    error_msg += f" ... (+{len(ticker_errors) - 5} more)"
            else:
                error_msg = f"All {len(failed)} ticker(s) failed with no specific error"
        return PricingResult(
            success=len(successful) > 0,
            results_df=results_df,
            ticker_results=results,
            swap_objects=swap_objects,
            successful_tickers=successful,
            failed_tickers=failed,
            error=error_msg,
        )

    # ─── Batch Price (for pricing mode with user-provided strikes) ─────────────────────────────

    def _price_batch(
            self,
            tickers_df: pd.DataFrame,
            progress_callback: Optional[Callable] = None,
    ) -> Dict[int, Tuple['TickerResult', Any]]:
        """
        Price all tickers in 1-2 HTTP calls instead of N individual calls.

        Each ticker already has a user-provided strike. We:
          1. Build 1 reference FPF (expensive — calendar HTTP), clone per ticker with correct strike
          2. Batch-price ALL instruments in 1 HTTP call
          3. Extract FairValue / Vol / Correlation
        """
        cfg = self.config
        total = len(tickers_df)
        indexed_results: Dict[int, Tuple[TickerResult, Any]] = {}
        self._batch_timings = {}

        _ensure_portal()

        # ── Parse per-ticker info ──
        tickers = []
        corr_assets = []
        currencies = []
        strikes_ref = []  # decimal (e.g. 0.22 for 22%)
        strikes_linked = []
        individual_correlations = []

        for _, row in tickers_df.iterrows():
            ticker = row['Tickers']
            corr_asset = row.get('Corridor Condition Asset', ticker)
            tickers.append(ticker)
            corr_assets.append(corr_asset)
            currencies.append(row.get('Currency') or self.cache.get_currency(ticker) or 'EUR')
            # Per-ticker correlation
            ind_corr = None
            if cfg.correl_input_method == "Individual Correlations" and 'Correlation' in row.index:
                try:
                    val = row.get('Correlation')
                    if val is not None and val != '':
                        ind_corr = float(val) / 100.0
                except (ValueError, TypeError):
                    pass
            individual_correlations.append(ind_corr)
            # Strikes from row (stored as vol, e.g. 0.22 for 22%)
            strike_va = None
            strike_ca = None

            # Support both legacy and current column names
            # Current: 'Strike Cross Corridor (%)', 'Strike Mono Var Swap (%)'
            # Legacy: 'Strike Cross (%)', 'Strike Mono (%)'
            if 'Strikes (%)' in row and row.get('Strikes (%)'):
                strike_va = float(row['Strikes (%)']) / 100
                strike_ca = strike_va
            elif 'Strike Cross Corridor (%)' in row and row.get('Strike Cross Corridor (%)'):
                strike_va = float(row['Strike Cross Corridor (%)']) / 100
                strike_ca = float(row.get('Strike Mono Var Swap (%)', row['Strike Cross Corridor (%)'])) / 100
            elif 'Strike Cross (%)' in row and row.get('Strike Cross (%)'):
                # Legacy column names
                strike_va = float(row['Strike Cross (%)']) / 100
                strike_ca = float(row.get('Strike Mono (%)', row['Strike Cross (%)'])) / 100
            else:
                strike_va = None
                strike_ca = None
            strikes_ref.append(strike_va)
            strikes_linked.append(strike_ca)

        ref_ticker = tickers[0]
        ref_corr_asset = corr_assets[0]
        ref_currency = currencies[0]
        is_cross = any(t != c for t, c in zip(tickers, corr_assets))

        # ── LSV validation (forbidden for mono corridor) ──
        use_lsv = cfg.use_lsv_cross_ev and cfg.lsv_params is not None
        if use_lsv:
            if not cfg.is_cross_corridor:
                raise ValueError(
                    "LSV pricing is only supported for cross-corridor mode. "
                    "Disable LSV or use distinct Variance Asset and Corridor Condition Asset."
                )
            mono_only = all(t == c for t, c in zip(tickers, corr_assets))
            if mono_only:
                raise ValueError(
                    "LSV pricing is only supported for cross-corridor mode. "
                    "All tickers have Variance Asset == Corridor Condition Asset."
                )
            dbg.info("batch-price", f"LSV enabled for price mode ({len(tickers)} tickers)")

        if progress_callback:
            try:
                progress_callback({'status': 'phase', 'message': 'Building FPFs...', 'completed': 0, 'total': total})
            except Exception:
                pass

        # ── Step 1: Build FPFs per ticker (with correct strike) ──
        t_fpf = time.time()
        cross_fpfs = []  # one per ticker
        mono_fpfs = []  # one per ticker (only if cross-corridor)

        # Build each FPF with its own ticker/corridor calendar intersection
        cross_fpfs = []
        mono_fpfs = []
        cross_obs_counts = []
        mono_obs_counts = []

        try:
            for i, ticker in enumerate(tickers):
                corr_asset = corr_assets[i]

                strike_var = (strikes_ref[i] or 0.01) ** 2

                cross_fpf = build_corridor_fpf(
                    tickers=[ticker],
                    last_obs_date=cfg.last_obs_date,
                    strike_date=cfg.strike_date,
                    strikes=[strike_var],
                    weights=[1.0],
                    low_barrier=cfg.dvar,
                    high_barrier=cfg.uvar,
                    is_capped=cfg.is_capped,
                    corr_asset=corr_asset,
                    schedule_calendar_asset=corr_asset if ticker != corr_asset else None,
                    currency=currencies[i],
                    use_parameters=False,
                )

                cross_fpfs.append(cross_fpf)

                cross_obj = FPFUnifiedEconomicsWrapper.from_data(
                    cross_fpf,
                    script_cls=corridorCovarianceSwap_v4,
                )

                cross_obs_counts.append(
                    len(cross_obj.observationDates) - 1
                )

                if ticker != corr_asset:
                    mono_strike_var = (
                                              strikes_linked[i]
                                              or strikes_ref[i]
                                              or 0.01
                                      ) ** 2

                    mono_fpf = build_corridor_fpf(
                        tickers=[corr_asset],
                        last_obs_date=cfg.last_obs_date,
                        strike_date=cfg.strike_date,
                        strikes=[mono_strike_var],
                        weights=[1.0],
                        low_barrier=cfg.dvar,
                        high_barrier=cfg.uvar,
                        is_capped=cfg.is_capped,
                        corr_asset=corr_asset,
                        schedule_calendar_asset=None,
                        currency=currencies[i],
                        use_parameters=False,
                    )

                    mono_fpfs.append(mono_fpf)

                    mono_obj = FPFUnifiedEconomicsWrapper.from_data(
                        mono_fpf,
                        script_cls=corridorCovarianceSwap_v4,
                    )

                    mono_obs_counts.append(
                        len(mono_obj.observationDates) - 1
                    )

                else:
                    mono_fpfs.append(None)
                    mono_obs_counts.append(
                        cross_obs_counts[i]
                    )

        except Exception as e:
            dbg.err("batch-price", f"FPF build failed: {e}")

            for idx, ticker in enumerate(tickers):
                indexed_results[idx] = (
                    TickerResult(
                        ticker=ticker,
                        corridor_asset=corr_assets[idx],
                        success=False,
                        error=f"FPF build failed: {e}",
                    ),
                    None,
                )

            return indexed_results



        self._batch_timings['fpf_build'] = time.time() - t_fpf
        dbg.ok("batch-price", f"FPFs built in {time.time() - t_fpf:.2f}s ({total} tickers)")

        if progress_callback:
            try:
                progress_callback({'status': 'phase', 'message': 'Pricing...', 'completed': 0, 'total': total})
            except Exception:
                pass

        # ── Step 2: Create instruments and batch-price ──
        t_price = time.time()

        # Model context
        is_index_product = ref_corr_asset.startswith(".") and all(t.startswith(".") for t in tickers)
        if cfg.model_name:
            _model = cfg.model_name
        elif is_index_product:
            _model = "EMEA-Index-MC-LV-MultiAsset"
        else:
            _model = "EMEA-Stocks-MC-LV-MultiAsset"

        is_index_model = _model == "EMEA-Index-MC-LV-MultiAsset"

        if is_index_model:
            model_params = {"ACEqFxShift": str(cfg.eqfx_shift)}
        else:
            model_params = {
                'ACEqEqSpread': str(cfg.eqeq_lambda),
                "EqEqCorrFloor": str(cfg.correl_floor),
                "ACEqFxShift": str(cfg.eqfx_shift),
            }
        _apply_special_rics_param(model_params, tickers)
        model_context = pricing_portal.create_model_context(_model, instrument_model_parameters=model_params)
        valuation_date = cfg.strike_date if isinstance(cfg.strike_date,
                                                       datetime.datetime) else datetime.datetime.combine(
            cfg.strike_date if isinstance(cfg.strike_date, datetime.date) else datetime.datetime.strptime(
                str(cfg.strike_date), "%Y-%m-%d").date(),
            datetime.datetime.min.time()
        )

        # Load underlyings
        all_rics = set(tickers + corr_assets)
        underlyings_map = {ric: (self.cache.get_instrument(ric) or _load_instrument_impl(ric)) for ric in all_rics}
        self._batch_timings['underlyings'] = time.time() - t_price

        # Create instruments: [cross_0, cross_1, ..., mono_0, mono_1, ...]
        instruments = []
        for i, ticker in enumerate(tickers):
            rics = list(set([ticker, corr_assets[i]]))
            ul = [underlyings_map[r] for r in rics]
            instruments.append(pricing_portal.create_fpf(
                fpf_string=cross_fpfs[i], instrument_ccy=currencies[i],
                underlyings=ul, premium_date=datetime.datetime.now().date(),
            ))
        n_cross = len(tickers)

        mono_indices = []  # track which tickers have mono instruments
        for i, ticker in enumerate(tickers):
            if mono_fpfs[i] is not None:
                ul = [underlyings_map[corr_assets[i]]]
                instruments.append(pricing_portal.create_fpf(
                    fpf_string=mono_fpfs[i], instrument_ccy=currencies[i],
                    underlyings=ul, premium_date=datetime.datetime.now().date(),
                ))
                mono_indices.append(i)
        n_mono = len(mono_indices)

        self._batch_timings['instruments'] = time.time() - t_price - self._batch_timings['underlyings']
        _safe_print(f"[BATCH-PRICE] Model={_model} | {n_cross} cross + {n_mono} mono = {len(instruments)} instruments")

        # ── Unified Scenario (LSV / LCM) via common helper ──
        unified_scenario = None
        use_lsv = cfg.use_lsv_cross_ev and cfg.lsv_params is not None
        use_lcm = cfg.lcm_params is not None and cfg.lcm_params.get('enabled', False)

        if use_lcm:
            # Validation: LCM requires cross-corridor (Variance Asset ≠ Corridor Asset)
            for i, ticker in enumerate(tickers):
                if ticker == corr_assets[i]:
                    raise ValueError(
                        f"LCM is only applicable to cross-corridor structures "
                        f"(Variance Asset must differ from Corridor Asset). "
                        f"Ticker '{ticker}' has Variance Asset == Corridor Asset."
                    )

        if use_lsv or use_lcm:
            from functions.common.pricing_scenarios import build_unified_scenario
            _ticker_rics = list(set(tickers + corr_assets))
            _lsv_df = None
            if use_lsv:
                _lsv_df = cfg.lsv_params if isinstance(cfg.lsv_params, pd.DataFrame) else pd.DataFrame(cfg.lsv_params)
                if 'RIC' in _lsv_df.columns and _lsv_df.index.dtype != object:
                    _lsv_df = _lsv_df.set_index('RIC')

            unified_scenario = build_unified_scenario(
                pricing_portal,
                use_lsv=use_lsv,
                use_lcm=use_lcm,
                underlying_rics=_ticker_rics,
                lsv_params=_lsv_df,
                correl_bump=cfg.lsv_correl_bump,
                correl_bump_style=cfg.lsv_correl_bump_style,
                lcm_properties=cfg.lcm_params.get('lcm_properties') if cfg.lcm_params else None,
            )
            _bump_names = []
            if use_lsv:
                _bump_names = ["LV", "LSV0", "LSV"]
            else:
                _bump_names = ["LV"]
            if use_lcm:
                _bump_names.append("LCM")
            _safe_print(
                f"[BATCH-PRICE] Scenario built: {len(_bump_names)} bumps {_bump_names} for {len(_ticker_rics)} RICs")

        # Metrics
        matu_ex_0, matu_stl_0 = calculate_payment_dates(cfg.last_obs_date, currencies[0])
        _excel_epoch = date(1899, 12, 30)
        excel_date = (matu_stl_0 - _excel_epoch).days

        atmf_params = [
            pricing_portal.create_metric_parameter("AnchorType", "Forward"),
            pricing_portal.create_metric_parameter("MaturityList", str(excel_date)),
            pricing_portal.create_metric_parameter("MaturityType", "Absolute"),
            pricing_portal.create_metric_parameter("Strikes", "1"),
        ]
        corr_params = [
            pricing_portal.create_metric_parameter("MaturityList", str(excel_date)),
            pricing_portal.create_metric_parameter("MaturityType", "Absolute"),
        ]
        metrics = [
            pricing_portal.create_metric("FairValue"),
            pricing_portal.create_metric("QueryLocalCcyVol", atmf_params),
            pricing_portal.create_metric("Correlation", corr_params),
        ]

        # ── Single batch price call ──
        MAX_PER_CALL = 100
        _n_chunks_price = (len(instruments) + MAX_PER_CALL - 1) // MAX_PER_CALL
        _total_batches_price = _n_chunks_price
        _batch_counter_price = [1]
        all_results_raw = {}
        use_scenario = unified_scenario is not None
        for start in range(0, len(instruments), MAX_PER_CALL):
            chunk = instruments[start:start + MAX_PER_CALL]
            try:
                price_kwargs = dict(
                    price_id="Price",
                    instruments=chunk,
                    valuation_date=valuation_date,
                    calculation_parameters={},
                    model_context=model_context,
                    overridden_snap_name=live_snap["name"],
                    metrics=metrics,
                )
                if unified_scenario is not None:
                    price_kwargs["scenario"] = unified_scenario
                batch_res = pricing_portal.price(**price_kwargs)
                results_dict = batch_res.get("results", {})
                all_results_raw[start] = {"raw": results_dict, "chunk_size": len(chunk)}
                _safe_print(
                    f"[BATCH-PRICE] Chunk {_batch_counter_price[0]}/{_total_batches_price}: {len(chunk)} instruments OK"
                    f"{' (with scenario)' if use_scenario else ''}")
            except Exception as e:
                dbg.err("batch-price", f"Chunk failed: {e}")
                all_results_raw[start] = {"raw": {}, "chunk_size": len(chunk)}

            # Emit batch progress event
            if progress_callback:
                progress_callback({
                    "status": "pricing_batch",
                    "batch": _batch_counter_price[0],
                    "total_batches": _total_batches_price,
                    "label": "price",
                })
            _batch_counter_price[0] += 1

        self._batch_timings['http_price'] = time.time() - t_price
        dbg.ok("batch-price", f"HTTP done in {self._batch_timings['http_price']:.1f}s")

        # ── Step 3: Extract results ──
        def _extract(global_idx, metric_name, bump_name=None):
            """Get metric value for instrument at global_idx, optionally from a scenario bump."""
            running = 0
            for chunk_start in sorted(all_results_raw.keys()):
                cd = all_results_raw[chunk_start]
                if global_idx < running + cd["chunk_size"]:
                    local_idx = global_idx - running
                    raw = cd["raw"]
                    key = "Price" if local_idx == 0 else f"Price_{local_idx}"
                    if key in raw and isinstance(raw[key], dict):
                        entry = raw[key]
                        # Scenario mode: look inside bump structure
                        if bump_name is not None and use_scenario:
                            # Format: {"SimpleScenarioBump": [{"LV": [...], "LSV0": [...], "LSV": [...]}]}
                            if "SimpleScenarioBump" in entry:
                                bumps = entry["SimpleScenarioBump"]
                                if bumps and isinstance(bumps, list) and len(bumps) > 0:
                                    bump_data = bumps[0].get(bump_name, [])
                                    if bump_data and isinstance(bump_data, list):
                                        # Find metric in bump results
                                        metric_list = bump_data[0].get(metric_name, []) if isinstance(bump_data[0],
                                                                                                      dict) else []
                                        if metric_list and isinstance(metric_list, list) and len(metric_list) > 0:
                                            return metric_list[0].get("value") if isinstance(metric_list[0],
                                                                                             dict) else None
                                return None
                            # Alternate format: bump name at top level
                            if bump_name in entry:
                                bump_data = entry[bump_name]
                                if isinstance(bump_data, list) and len(bump_data) > 0 and isinstance(bump_data[0],
                                                                                                     dict):
                                    metric_list = bump_data[0].get(metric_name, [])
                                    if metric_list and isinstance(metric_list, list) and len(metric_list) > 0:
                                        return metric_list[0].get("value") if isinstance(metric_list[0], dict) else None
                                return None
                        # Non-scenario mode: direct metric access
                        metric_list = entry.get(metric_name, [])
                        if metric_list and isinstance(metric_list, list) and len(metric_list) > 0:
                            if isinstance(metric_list[0], dict):
                                return metric_list[0].get("value")
                    # Format A fallback (non-scenario only)
                    if bump_name is None and "Price" in raw and isinstance(raw["Price"], dict):
                        metric_list = raw["Price"].get(metric_name, [])
                        if metric_list and isinstance(metric_list, list) and local_idx < len(metric_list):
                            if isinstance(metric_list[local_idx], dict):
                                return metric_list[local_idx].get("value")
                    return None
                running += cd["chunk_size"]
            return None

        # ── Step 3: Extract results ──
        # Pre-compute ATMF Vol for unique Variance Assets (deduplicated)
        # The cross FPF's QueryLocalCcyVol returns corridor asset vol, NOT variance asset vol.
        # So we must fetch variance asset vol separately.
        matu_ex_0, matu_stl_0 = calculate_payment_dates(cfg.last_obs_date, currencies[0])
        _unique_vas = set(tickers)
        _va_atmf_cache = {}
        for _va in _unique_vas:
            try:
                _va_atmf_cache[_va] = compute_implied_vol(_va, matu_stl_0, "Forward")
            except Exception:
                _va_atmf_cache[_va] = None
        if _unique_vas:
            _safe_print(f"[BATCH-PRICE] ATMF vol lookup: {len(_unique_vas)} unique Variance Asset(s)")

        # Build TickerResults
        for idx, ticker in enumerate(tickers):
            try:
                # Extract FairValue — scenario mode gets per-bump, non-scenario gets single
                if use_scenario:
                    mid_va_lv = _extract(idx, "FairValue", "LV")
                    mid_va_lsv0 = _extract(idx, "FairValue", "LSV0")
                    mid_va_lsv = _extract(idx, "FairValue", "LSV")
                    mid_va_lcm = _extract(idx, "FairValue", "LCM") if use_lcm else None
                    mid_va = mid_va_lv  # backward compat: mid = LV value
                    correlation = _extract(idx, "Correlation", "LV")
                else:
                    mid_va = _extract(idx, "FairValue")
                    mid_va_lv = None
                    mid_va_lsv0 = None
                    mid_va_lsv = None
                    mid_va_lcm = None
                    correlation = _extract(idx, "Correlation")

                # ATMF Vol Variance Asset — from deduplicated cache (not from FPF result)
                atmf_vol_va = _va_atmf_cache.get(ticker)

                # Mono result (if cross-corridor)
                mid_ca = None
                atmf_vol_ca = None
                if idx in mono_indices:
                    mono_global_idx = n_cross + mono_indices.index(idx)
                    if use_scenario:
                        mid_ca = _extract(mono_global_idx, "FairValue", "LV")
                        atmf_vol_ca = _extract(mono_global_idx, "QueryLocalCcyVol", "LV")
                    else:
                        mid_ca = _extract(mono_global_idx, "FairValue")
                        atmf_vol_ca = _extract(mono_global_idx, "QueryLocalCcyVol")
                elif ticker == corr_assets[idx]:
                    mid_ca = mid_va
                    atmf_vol_ca = atmf_vol_va

                # Obs dates — count from the ref_obj which already has the schedule
                obs_count_cross = 0
                obs_count_mono = 0
                try:

                    obs_count_cross = cross_obs_counts[idx]
                    obs_count_mono = mono_obs_counts[idx]

                except Exception:
                    pass

                indexed_results[idx] = (TickerResult(
                    ticker=ticker,
                    corridor_asset=corr_assets[idx],
                    success=True if mid_va is not None else False,
                    error=None if mid_va is not None else f"No FairValue returned for {ticker}",
                    strike_variance_asset=strikes_ref[idx],  # vol (e.g. 0.22)
                    strike_corridor_asset=strikes_linked[idx],  # vol (e.g. 0.19)
                    mid_variance_asset=mid_va,
                    mid_corridor_asset=mid_ca,
                    mid_variance_asset_lv=mid_va_lv,
                    mid_variance_asset_lsv0=mid_va_lsv0,
                    mid_variance_asset_lsv=mid_va_lsv,
                    mid_variance_asset_lcm=mid_va_lcm,
                    atmf_vol_variance_asset=atmf_vol_va,
                    atmf_vol_corridor_asset=atmf_vol_ca,
                    correlation=correlation,
                    currency=currencies[idx],
                    obs_dates_cross=obs_count_cross if obs_count_cross > 0 else None,
                    obs_dates_mono=obs_count_mono if obs_count_mono > 0 else None,
                    fpf_string_cross=cross_fpfs[idx],
                ), None)

            except Exception as e:
                indexed_results[idx] = (TickerResult(
                    ticker=ticker, corridor_asset=corr_assets[idx],
                    success=False, error=str(e)
                ), None)

            if progress_callback:
                try:
                    progress_callback(
                        {'ticker': ticker, 'status': 'completed', 'completed': idx + 1, 'total': total, 'message': ''})
                except Exception:
                    pass

        dbg.ok("batch-price", f"all {total} tickers done in {time.time() - t_fpf:.1f}s")
        return indexed_results

    # ─── Batch Solve ─────────────────────────────────────────────────────

    def _solve_batch_ev_ra(
            self,
            tickers_df: pd.DataFrame,
            progress_callback: Optional[Callable] = None,
    ) -> Dict[int, Tuple['TickerResult', Any]]:
        """
        Solve all corridor strikes via EV/RA ratio in 1 HTTP call.

        For each ticker:  strike² = -EV_cross / RA_cross   (cross-corridor strike)
        For each corr:    strike² = -EV_mono  / RA_mono    (mono-corridor strike)

        Architecture:
          1. Build 1 reference FPF per calendar group (expensive: ~250 HTTP to calendar svc)
          2. Clone FPF objects for each ticker (swap ticker/corridor fields — zero HTTP)
          3. Batch-price ALL instruments in 1 HTTP call
          4. Compute strikes locally
          5. Build solved FPFs (with is_capped + final strike) for output
        """
        cfg = self.config
        total = len(tickers_df)
        indexed_results: Dict[int, Tuple[TickerResult, Any]] = {}
        self._batch_timings = {}  # Populated as we go for the summary

        _ensure_portal()

        # ── Extract per-ticker info ──
        tickers = []
        corr_assets = []
        currencies = []
        schedule_assets = []
        individual_correlations = []  # Per-ticker correlation overrides (or None)
        for _, row in tickers_df.iterrows():
            ticker = row['Tickers']
            corr_asset = row.get('Corridor Condition Asset', ticker)
            tickers.append(ticker)
            corr_assets.append(corr_asset)
            currencies.append(row.get('Currency') or self.cache.get_currency(ticker) or 'EUR')
            is_cross = ticker != corr_asset
            # For cross-corridor: schedule uses ticker's calendar (variance asset)
            # For mono-corridor: schedule uses corridor's calendar (same as ticker)
            # schedule_assets.append(ticker if is_cross else corr_asset)

            schedule_assets.append(corr_asset)


            # Per-ticker individual correlation (from DataFrame "Correlation" column)
            ind_corr = None
            if cfg.correl_input_method == "Individual Correlations" and 'Correlation' in row.index:
                try:
                    val = row.get('Correlation')
                    if val is not None and val != '':
                        ind_corr = float(val) / 100.0  # Input is in %, convert to decimal
                except (ValueError, TypeError):
                    pass
            individual_correlations.append(ind_corr)

        ref_ticker = tickers[0]
        ref_corr_asset = corr_assets[0]
        ref_currency = currencies[0]
        ref_schedule_asset = schedule_assets[0]

        # DEBUG: Log schedule asset configuration
        for idx, (t, c, s) in enumerate(zip(tickers, corr_assets, schedule_assets)):
            is_cross = t != c
            dbg.ok("SCHEDULE-CONFIG", f"idx={idx}, ticker={t}, corr={c}, schedule={s}, cross={is_cross}")

        unique_corr_assets = list(set(corr_assets))

        # ── Compute linked_asset early (needed for all_rics and LSV) ──
        linked_asset = ref_corr_asset

        # ── all_rics for logging ──
        all_rics = set(tickers + corr_assets + [linked_asset])

        # ── Build lsv_scenario early (needed before FPF building and batch pricing) ──
        lsv_scenario = None
        if cfg.lsv_params is not None:
            from functions.common.pricing_scenarios import build_lsv_scenario
            _lsv_df = cfg.lsv_params if isinstance(cfg.lsv_params, pd.DataFrame) else pd.DataFrame(cfg.lsv_params)
            if 'RIC' in _lsv_df.columns and _lsv_df.index.dtype != object:
                _lsv_df = _lsv_df.set_index('RIC')
            lsv_scenario = build_lsv_scenario(
                pricing_portal,
                underlying_rics=list(all_rics),
                lsv_params=_lsv_df,
                correl_bump=cfg.lsv_correl_bump,
                correl_bump_style=cfg.lsv_correl_bump_style,
            )

        # ── LSV flag (must be defined before FPF building uses it) ──
        use_lsv_cross_ev = cfg.use_lsv_cross_ev and cfg.lsv_params is not None
        use_lsv_mono_ev = use_lsv_cross_ev and cfg.is_cross_corridor

        if use_lsv_cross_ev:
            # Validate: LSV only valid for cross-corridor (variance asset != corridor asset)
            if not cfg.is_cross_corridor:
                raise ValueError(
                    "LSV cross-EV pricing is only supported for cross-corridor pricing. "
                    "Disable LSV or provide a distinct Variance Asset and Corridor Condition Asset."
                )
            # Per-ticker validation: ensure at least one ticker has a different corridor asset
            mono_only = all(t == c for t, c in zip(tickers, corr_assets))
            if mono_only:
                raise ValueError(
                    "LSV cross-EV pricing is only supported for cross-corridor pricing. "
                    "Disable LSV or provide a distinct Variance Asset and Corridor Condition Asset."
                )
            dbg.info("batch", f"LSV enabled for {len(all_rics)} RICs (cross-EV)")
        if use_lsv_mono_ev:
            dbg.info("batch", f"LSV also enabled for mono-EV (same {len(unique_corr_assets)} corridor assets)")

        if progress_callback:
            try:
                progress_callback({'status': 'phase', 'message': 'Building FPFs...', 'completed': 0, 'total': total})
            except Exception:
                pass

        t_fpf = time.time()
        try:
            # ── Step 1: Build reference FPF objects (expensive — calendar HTTP calls) ──
            # Group tickers by schedule_asset to ensure correct observation dates.
            # Each unique schedule_asset requires its own reference FPF because the
            # schedule is generated during FPF construction and NOT stored in the object.
            from collections import defaultdict
            schedule_groups = defaultdict(list)
            for idx, (ticker, corr, sched) in enumerate(zip(tickers, corr_assets, schedule_assets)):
                schedule_groups[sched].append((idx, ticker, corr))

            # Build one reference FPF per schedule_asset group
            ev_cross_ref_objs = {}  # schedule_asset -> ev_ref_obj
            for sched_asset, members in schedule_groups.items():
                ref_idx, ref_ticker, ref_corr = members[0]
                ref_currency_sched = currencies[ref_idx]
                dbg.ok("SCHEDULE-ASSET-GROUP", f"schedule_asset={sched_asset}: {len(members)} tickers")

                ev_cross_ref_fpf = build_corridor_fpf(
                    tickers=[ref_ticker], last_obs_date=cfg.last_obs_date,
                    strike_date=cfg.strike_date, strikes=[0.000001], weights=[1.0],
                    low_barrier=cfg.dvar, high_barrier=cfg.uvar, is_capped=False,
                    corr_asset=ref_corr, schedule_calendar_asset=sched_asset,
                    currency=ref_currency_sched, use_parameters=False,
                )

                ev_cross_ref_obj = FPFUnifiedEconomicsWrapper.from_data(ev_cross_ref_fpf,
                                                                        script_cls=corridorCovarianceSwap_v4)
                ev_cross_ref_objs[sched_asset] = ev_cross_ref_obj
                dbg.ok("SCHEDULE-ASSET-GROUP", f"  built ref FPF for schedule_asset={sched_asset}")

            # (RA is now built only per unique corridor asset via mono_ref_objs below)

            # Mono reference: clone from cross (same schedule — no calendar HTTP needed).
            # Mono corridor: variance asset = corridor asset = the index (e.g. .STOXX50E)
            # Same observation dates, same schedule — only the ticker and corridor fields differ.
            # For LSV: build LSV and LSV0 versions of mono EV (uncapped, same as cross)

            mono_ref_objs = {}

            for corr in unique_corr_assets:
                ref_idx = corr_assets.index(corr)
                sched_asset = schedule_assets[ref_idx]
                mono_ref_obj = ev_cross_ref_objs[sched_asset]
                mono_ev_obj = mono_ref_obj.clone(
                    varianceDetails=mono_ref_obj.varianceDetails.clone(
                        varianceAssetsAndIndexLegDetails=[
                            mono_ref_obj.varianceDetails.varianceAssetsAndIndexLegDetails[0].clone(
                                asset=corr,
                                basketMultiplier=1,
                                strike=0.000001,
                                legCap="Nothing",
                                legFloor="Nothing",
                                legMultiplier=1.0,
                            )
                        ],
                        isOptionOnVariance=True,
                    ),
                    corridorDefinition=Just(
                        mono_ref_obj.corridorDefinition.value.clone(
                            corridorAssets=[
                                corridorCovarianceSwap_v4.CorridorAssets(
                                    corridorAsset=corr,
                                    corridorMultiplier=1.0,
                                    corridorAssetLag=0,
                                )
                            ]
                        )
                    ),
                    koDetails=mono_ref_obj.koDetails.clone(
                        koAssets=[
                            corridorCovarianceSwap_v4.KoAssets(
                                koAsset=corr,
                                koAssetMultiplier=1.0,
                                koAssetLag=0,
                            )
                        ]
                    ),
                )

                mono_ra_obj = mono_ev_obj.clone(
                    varianceDetails=mono_ev_obj.varianceDetails.clone(
                        varianceAssetsAndIndexLegDetails=[
                            mono_ev_obj.varianceDetails.varianceAssetsAndIndexLegDetails[0].clone(
                                asset=corr,
                                basketMultiplier=-1,
                                strike=-1,
                                legCap=Just(-1),
                                legFloor=Just(-1),
                                legMultiplier=1.0,
                            )
                        ],
                        isOptionOnVariance=True,
                    )
                )

                if use_lsv_mono_ev:
                    mono_ev_lsv_obj = mono_ev_obj.clone(
                        varianceDetails=mono_ev_obj.varianceDetails.clone(
                            varianceAssetsAndIndexLegDetails=[
                                mono_ev_obj.varianceDetails.varianceAssetsAndIndexLegDetails[0].clone(
                                    asset=corr,
                                    basketMultiplier=1,
                                    strike=0.000001,
                                    legCap="Nothing",
                                    legFloor="Nothing",
                                    legMultiplier=1.0,
                                )
                            ],
                            isOptionOnVariance=True,
                        ),
                        corridorDefinition=Just(
                            mono_ev_obj.corridorDefinition.value.clone(
                                corridorAssets=[
                                    corridorCovarianceSwap_v4.CorridorAssets(
                                        corridorAsset=corr,
                                        corridorMultiplier=1.0,
                                        corridorAssetLag=0,
                                    )
                                ]
                            )
                        ),
                        koDetails=mono_ev_obj.koDetails.clone(
                            koAssets=[
                                corridorCovarianceSwap_v4.KoAssets(
                                    koAsset=corr,
                                    koAssetMultiplier=1.0,
                                    koAssetLag=0,
                                )
                            ]
                        ),
                    )

                    mono_ev_lsv_zero_obj = mono_ev_obj.clone(
                        varianceDetails=mono_ev_obj.varianceDetails.clone(
                            varianceAssetsAndIndexLegDetails=[
                                mono_ev_obj.varianceDetails.varianceAssetsAndIndexLegDetails[0].clone(
                                    asset=corr,
                                    basketMultiplier=1,
                                    strike=0.000001,
                                    legCap="Nothing",
                                    legFloor="Nothing",
                                    legMultiplier=1.0,
                                )
                            ],
                            isOptionOnVariance=True,
                        ),
                        corridorDefinition=Just(
                            mono_ev_obj.corridorDefinition.value.clone(
                                corridorAssets=[
                                    corridorCovarianceSwap_v4.CorridorAssets(
                                        corridorAsset=corr,
                                        corridorMultiplier=1.0,
                                        corridorAssetLag=0,
                                    )
                                ]
                            )
                        ),
                        koDetails=mono_ev_obj.koDetails.clone(
                            koAssets=[
                                corridorCovarianceSwap_v4.KoAssets(
                                    koAsset=corr,
                                    koAssetMultiplier=1.0,
                                    koAssetLag=0,
                                )
                            ]
                        ),
                    )

                    mono_ref_objs[corr] = (
                        mono_ev_obj,
                        mono_ra_obj,
                        mono_ev_lsv_obj,
                        mono_ev_lsv_zero_obj,
                    )
                else:
                    mono_ref_objs[corr] = (
                        mono_ev_obj,
                        mono_ra_obj,
                        None,
                        None,
                    )
            # ── Step 2: Clone FPF objects per ticker (no HTTP — instant) ──
            def _clone_for_ticker(ref_obj, ticker, corr_asset):
                """Clone FPF with new ticker/corridor. No string replace, no HTTP."""
                return ref_obj.clone(
                    varianceDetails=ref_obj.varianceDetails.clone(
                        varianceAssetsAndIndexLegDetails=[
                            ref_obj.varianceDetails.varianceAssetsAndIndexLegDetails[0].clone(asset=ticker)
                        ]
                    ),
                    corridorDefinition=Just(ref_obj.corridorDefinition.value.clone(
                        corridorAssets=[corridorCovarianceSwap_v4.CorridorAssets(
                            corridorAsset=corr_asset, corridorMultiplier=1.0, corridorAssetLag=0
                        )]
                    )),
                    koDetails=ref_obj.koDetails.clone(
                        koAssets=[corridorCovarianceSwap_v4.KoAssets(
                            koAsset=ticker, koAssetMultiplier=1.0, koAssetLag=0
                        )]
                    ),
                ).to_fpf_string()

            # ── Parallelize FPF serialization (CPU-bound, independent) ──
            from concurrent.futures import ThreadPoolExecutor, as_completed
            t_clone = time.time()
            ev_cross_fpfs = {}
            ev_mono_fpfs = {}
            ev_mono_lsv_fpfs = {}
            ev_mono_lsv_zero_fpfs = {}
            ra_fpfs = {}  # keyed by corridor asset (one RA per unique corridor asset)
            _clone_tasks = []  # (type, key, ref_obj, ticker, corr_asset)
            for i, ticker in enumerate(tickers):
                sched_asset = schedule_assets[i]
                ev_cross_ref_obj = ev_cross_ref_objs[sched_asset]
                _clone_tasks.append(('ev', i, ev_cross_ref_obj, ticker, corr_assets[i]))
            for corr, (ev_obj, ra_obj, ev_lsv_obj, ev_lsv_zero_obj) in mono_ref_objs.items():
                _clone_tasks.append(('ev_mono', corr, ev_obj, None, None))
                if ev_lsv_obj is not None:
                    _clone_tasks.append(('ev_mono_lsv', corr, ev_lsv_obj, None, None))
                if ev_lsv_zero_obj is not None:
                    _clone_tasks.append(('ev_mono_lsv_zero', corr, ev_lsv_zero_obj, None, None))
                _clone_tasks.append(('ra', corr, ra_obj, None, None))

            def _do_clone(task):
                kind, key, obj, ticker, corr = task
                if ticker is None:
                    return (kind, key, obj.to_fpf_string())
                return (kind, key, _clone_for_ticker(obj, ticker, corr))

            with ThreadPoolExecutor(max_workers=8) as pool:
                for kind, key, fpf_str in pool.map(_do_clone, _clone_tasks):
                    if kind == 'ev':
                        ev_cross_fpfs[key] = fpf_str
                    elif kind == 'ev_mono':
                        ev_mono_fpfs[key] = fpf_str
                    elif kind == 'ev_mono_lsv':
                        ev_mono_lsv_fpfs[key] = fpf_str
                    elif kind == 'ev_mono_lsv_zero':
                        ev_mono_lsv_zero_fpfs[key] = fpf_str
                    elif kind == 'ra':
                        ra_fpfs[key] = fpf_str

            dbg.info("batch",
                     f"[TIMING] Clone+serialize: {time.time() - t_clone:.2f}s ({len(_clone_tasks)} FPFs, 8 threads)")

            dbg.ok("batch", f"FPFs built in {time.time() - t_fpf:.2f}s "
                            f"({len(tickers)} cross + {len(unique_corr_assets)} mono)")
            self._batch_timings['fpf_build'] = time.time() - t_fpf

        except Exception as e:
            import traceback
            dbg.err("batch", f"FPF build failed: {e}")
            traceback.print_exc()
            for idx, ticker in enumerate(tickers):
                indexed_results[idx] = (TickerResult(
                    ticker=ticker, corridor_asset=corr_assets[idx],
                    success=False, error=f"FPF build failed: {e}",
                ), None)
            return indexed_results

        if progress_callback:
            try:
                progress_callback({'status': 'phase', 'message': 'Pricing...', 'completed': 0, 'total': total})
            except Exception:
                pass

        # ── Step 3: Create instruments and batch-price (1 HTTP call) ──
        t_price = time.time()
        # linked_asset already defined earlier (line ~2748)

        is_index_product = linked_asset.startswith(".") and all(t.startswith(".") for t in tickers)

        # ── Model selection: use user-specified model or auto-detect ──
        if cfg.model_name:
            _model = cfg.model_name
        elif is_index_product:
            _model = "EMEA-Index-MC-LV-MultiAsset"
        else:
            _model = "EMEA-Stocks-MC-LV-MultiAsset"

        # ── Model parameters: skip correlation params when using individual correlation ──
        use_individual_corr = (cfg.individual_correlation is not None
                               and cfg.correl_input_method == "Individual Correlations")
        if use_individual_corr:
            # Individual correlation mode: don't pass global correlation params
            # The correlation is forced via scenario, not model params
            model_params = {}
        elif _model == "EMEA-Index-MC-LV-MultiAsset":
            # Index model doesn't support ACEqFxShift
            model_params = {}
        else:
            model_params = {
                'ACEqEqSpread': str(cfg.eqeq_lambda),
                "EqEqCorrFloor": str(cfg.correl_floor),
                "ACEqFxShift": str(cfg.eqfx_shift),
            }
        _apply_special_rics_param(model_params, tickers)
        model_context = pricing_portal.create_model_context(_model, instrument_model_parameters=model_params)
        valuation_date = cfg.strike_date if isinstance(cfg.strike_date,
                                                       datetime.datetime) else datetime.datetime.combine(
            cfg.strike_date if isinstance(cfg.strike_date, datetime.date) else datetime.datetime.strptime(
                str(cfg.strike_date), "%Y-%m-%d").date(),
            datetime.datetime.min.time()
        )

        # Load underlyings for instrument creation (use _PortalCache which was pre-populated)
        t_ul = time.time()
        all_rics = set(tickers + corr_assets + [linked_asset])
        underlyings_map = {ric: (self.cache.get_instrument(ric) or _load_instrument_impl(ric)) for ric in all_rics}
        dbg.info("batch", f"[TIMING] Underlyings loaded: {time.time() - t_ul:.3f}s ({len(all_rics)} RICs)")
        self._batch_timings['underlyings'] = time.time() - t_ul

        def _make_instrument(fpf_string, rics_needed):
            """Create a portal instrument from FPF string."""
            ul = [underlyings_map[r] for r in set(rics_needed)]
            return pricing_portal.create_fpf(
                fpf_string=fpf_string, instrument_ccy=ref_currency,
                underlyings=ul, premium_date=datetime.datetime.now().date(),
            )

        # Build instrument list: [EV_cross_0..N-1, EV_mono_0..M-1, RA_0..M-1, EV_mono_lsv_0..M-1, EV_mono_lsv_zero_0..M-1]
        # RA is computed once per unique corridor asset and reused for both cross and mono strikes
        t_instr = time.time()
        instruments = []
        # Cross EV instruments (1 per ticker)
        for i, ticker in enumerate(tickers):
            instruments.append(_make_instrument(ev_cross_fpfs[i], [ticker, corr_assets[i]]))
        n_ev = len(tickers)
        # Mono EV instruments (1 per unique corridor asset)
        mono_corr_order = unique_corr_assets
        for corr in mono_corr_order:
            instruments.append(_make_instrument(ev_mono_fpfs[corr], [corr]))
        n_ev_mono = len(mono_corr_order)
        # RA instruments (1 per unique corridor asset — shared between cross and mono)
        for corr in mono_corr_order:
            instruments.append(_make_instrument(ra_fpfs[corr], [corr]))
        n_ra = len(mono_corr_order)
        # LSV versions for mono EV (if enabled)
        n_ev_mono_lsv = 0
        n_ev_mono_lsv_zero = 0
        if use_lsv_mono_ev:
            for corr in mono_corr_order:
                if corr in ev_mono_lsv_fpfs:
                    instruments.append(_make_instrument(ev_mono_lsv_fpfs[corr], [corr]))
                    n_ev_mono_lsv += 1
            for corr in mono_corr_order:
                if corr in ev_mono_lsv_zero_fpfs:
                    instruments.append(_make_instrument(ev_mono_lsv_zero_fpfs[corr], [corr]))
                    n_ev_mono_lsv_zero += 1
        # Build corridor asset → RA index mapping for cross strike lookup
        _corr_to_ra_idx = {corr: i for i, corr in enumerate(mono_corr_order)}
        dbg.info("batch", f"[TIMING] Instruments created: {time.time() - t_instr:.3f}s ({len(instruments)} total)")
        self._batch_timings['instruments'] = time.time() - t_instr

        # ── Instrument layout summary ──
        _has_scenario = (lsv_scenario is not None) or (
                    cfg.lcm_params is not None and cfg.lcm_params.get('enabled', False))
        _scenario_label = " | Scenario=enabled" if _has_scenario else ""
        _lsv_label = f" | LSV_mono×{n_ev_mono_lsv}" if n_ev_mono_lsv > 0 else ""
        _safe_print(f"\n[BATCH] Model={_model} | Tickers={len(tickers)} | Instruments={len(instruments)} "
                    f"(EV_cross×{n_ev}, EV_linked×{n_ev_mono}, RA×{n_ra}{_lsv_label}) | Corr assets={len(unique_corr_assets)}{_scenario_label}")

        t_http = time.time()

        # ── Chunking: split instruments into batches of ≤100 to avoid server limit ──
        MAX_INSTRUMENTS_PER_CALL = 100

        # ── Compute total batch count for progress reporting ──
        _main_chunks = (len(instruments) + MAX_INSTRUMENTS_PER_CALL - 1) // MAX_INSTRUMENTS_PER_CALL
        _atms_chunks = _main_chunks if cfg.vol_mode == "ATMF+ATMS" else 0
        _total_batches = _main_chunks + _atms_chunks
        _batch_counter = [1]  # mutable counter for nested function

        def _price_in_batches(instruments, metrics, price_id="Price", scenario=None, batch_label="main"):
            """Price instruments in chunks, return raw portal results dict."""
            nonlocal _total_batches
            all_results = {}
            global_idx = 0
            n_chunks = (len(instruments) + MAX_INSTRUMENTS_PER_CALL - 1) // MAX_INSTRUMENTS_PER_CALL
            # Dynamically adjust total if we're about to exceed it
            if _batch_counter[0] + n_chunks - 1 > _total_batches:
                _total_batches = _batch_counter[0] + n_chunks - 1
            for start in range(0, len(instruments), MAX_INSTRUMENTS_PER_CALL):
                chunk = instruments[start:start + MAX_INSTRUMENTS_PER_CALL]
                if not chunk:
                    continue
                chunk_num = start // MAX_INSTRUMENTS_PER_CALL + 1
                try:
                    price_kwargs = {
                        "price_id": price_id,
                        "instruments": chunk,
                        "valuation_date": valuation_date,
                        "calculation_parameters": {},
                        "model_context": model_context,
                        "overridden_snap_name": live_snap["name"],
                        "metrics": metrics,
                    }
                    if scenario is not None:
                        price_kwargs["scenario"] = scenario
                    # ── Progress tracking ──
                    _safe_print(f"\r[BATCH] Sent {chunk_num} / {n_chunks} "
                                f"({len(chunk)} instruments)       ", end="", flush=True)

                    dbg.step("batch", f"HTTP call: {len(chunk)} instruments (chunk start={start})...")
                    batch_res = pricing_portal.price(**price_kwargs)
                    results_dict = batch_res.get("results", {})
                    results_keys = list(results_dict.keys())

                    # DEBUG: Log pricing response keys per chunk
                    dbg.ok("BATCH-PRICING-RESPONSE",
                           f"chunk={chunk_num}/{n_chunks}, results keys={results_keys}, FairValue count={len(results_dict.get('FairValue', []))}")

                    # Store raw results_dict for this chunk
                    all_results[start] = {"raw": results_dict, "chunk_size": len(chunk), "global_idx": global_idx}

                    # Emit batch progress event
                    if progress_callback:
                        progress_callback({
                            "status": "pricing_batch",
                            "batch": _batch_counter[0],
                            "total_batches": _total_batches,
                            "label": batch_label,
                        })

                except Exception as e:
                    # Light error message
                    _safe_print(f"\r[BATCH] Chunk {chunk_num} failed: {type(e).__name__}       ")
                    # Store empty — no slow individual fallback
                    all_results[start] = {"raw": {}, "chunk_size": len(chunk), "global_idx": global_idx}
                    if progress_callback:
                        progress_callback({
                            "status": "pricing_batch",
                            "batch": _batch_counter[0],
                            "total_batches": _total_batches,
                            "label": batch_label,
                        })

                _batch_counter[0] += 1
                global_idx += len(chunk)
            return all_results

        dbg.step("batch", f"pricing {len(instruments)} instruments in chunks of {MAX_INSTRUMENTS_PER_CALL}...")
        t_price_call = time.time()
        dbg.info("batch", f"[TIMING] Pre-price setup: {time.time() - t_http:.3f}s")

        # Maturity for vol query
        matu_ex_0, matu_stl_0 = calculate_payment_dates(cfg.last_obs_date, currencies[0])
        _excel_epoch = date(1899, 12, 30)
        excel_date = (matu_stl_0 - _excel_epoch).days

        atmf_params = [
            pricing_portal.create_metric_parameter("AnchorType", "Forward"),
            pricing_portal.create_metric_parameter("MaturityList", str(excel_date)),
            pricing_portal.create_metric_parameter("MaturityType", "Absolute"),
            pricing_portal.create_metric_parameter("Strikes", "1"),
        ]

        # Correlation metric parameters (requires maturity as excel serial date)
        _excel_epoch_corr = date(1899, 12, 30)
        corr_serial = (matu_stl_0 - _excel_epoch_corr).days
        corr_params = [
            pricing_portal.create_metric_parameter("MaturityList", str(corr_serial)),
            pricing_portal.create_metric_parameter("MaturityType", "Absolute"),
        ]

        # Build scenario for individual correlation override (if applicable)
        # Per-ticker correlations require separate batch calls (one per unique correlation level)
        batch_scenario = None
        use_scenario = False

        # Check if we have per-ticker individual correlations
        has_per_ticker_corr = (cfg.correl_input_method == "Individual Correlations"
                               and any(c is not None for c in individual_correlations))

        if has_per_ticker_corr:
            # ── Per-ticker correlation mode: group instruments by correlation value ──
            # Each unique correlation needs its own batch call with its own scenario
            use_scenario = True

            # Group ticker indices by correlation value
            from collections import defaultdict
            corr_groups = defaultdict(list)  # corr_value -> [ticker_indices]
            for i, corr_val in enumerate(individual_correlations):
                corr_groups[corr_val].append(i)

            # Lighter debug: just summary

            # Price each correlation group separately
            all_ev_cross = [None] * len(tickers)
            all_ra = [None] * len(tickers)

            def _pt_fv_at(res, pos):
                """FairValue at global instrument position ``pos`` in a chunked
                batch response (handles SimpleScenarioBump-wrapped entries)."""
                running = 0
                for cs in sorted(res.keys()):
                    cd = res[cs]
                    if pos < running + cd["chunk_size"]:
                        local_idx = pos - running
                        raw = cd["raw"]
                        key = "Price" if local_idx == 0 else f"Price_{local_idx}"
                        if key in raw and isinstance(raw[key], dict):
                            fv_list = raw[key].get("FairValue", [])
                            if fv_list:
                                val = fv_list[0].get("value") if isinstance(fv_list[0], dict) else None
                                if val is not None:
                                    return val
                            bumps = raw[key].get("SimpleScenarioBump")
                            if bumps and isinstance(bumps, list) and bumps:
                                fv_list = bumps[0].get("FairValue", [])
                                if fv_list:
                                    return fv_list[0].get("value") if isinstance(fv_list[0], dict) else None
                        return None
                    running += cd["chunk_size"]
                return None

            for corr_val, ticker_indices in corr_groups.items():
                # Build instruments for this group
                group_instruments = []
                for i in ticker_indices:
                    group_instruments.append(_make_instrument(ev_cross_fpfs[i], [tickers[i], corr_assets[i]]))
                for i in ticker_indices:
                    # ra_fpfs is keyed by corridor asset, not by ticker index
                    corr = corr_assets[i]
                    group_instruments.append(_make_instrument(ra_fpfs[corr], [tickers[i], corr]))

                # Create scenario for this correlation level
                group_scenario = None
                if corr_val is not None:
                    group_scenario = pricing_portal.create_scenario_simple(
                        mutator_name="GenericMutatorOverrideCorrelationEqEq",
                        properties={"CorrelationLevel": corr_val}
                    )

                group_res = _price_in_batches(
                    group_instruments,
                    metrics=[
                        pricing_portal.create_metric("FairValue"),
                        pricing_portal.create_metric("QueryLocalCcyVol", atmf_params),
                        pricing_portal.create_metric("Correlation", corr_params),
                    ],
                    price_id="Price",
                    scenario=group_scenario,
                )

                # Extract results back into position
                n_group = len(ticker_indices)
                for local_i, global_i in enumerate(ticker_indices):
                    # EV-cross is at position local_i, RA at n_group + local_i
                    all_ev_cross[global_i] = _pt_fv_at(group_res, local_i)
                    all_ra[global_i] = _pt_fv_at(group_res, n_group + local_i)

            # Mono instruments: price without per-ticker correlation (they use the corridor asset itself)
            mono_instruments = []
            for corr in mono_corr_order:
                mono_instruments.append(_make_instrument(ev_mono_fpfs[corr], [corr]))
            for corr in mono_corr_order:
                mono_instruments.append(_make_instrument(ra_fpfs[corr], [corr]))

            mono_res = _price_in_batches(
                mono_instruments,
                metrics=[
                    pricing_portal.create_metric("FairValue"),
                    pricing_portal.create_metric("QueryLocalCcyVol", atmf_params),
                ],
                price_id="Price",
            )

            # Build a combined results_map for the post-processing code
            # We'll override ev_cross_values / ra_values_by_corr directly below
            results_map = mono_res
            # ATMS included in main batch when ATMS-only mode
            results_map_atms = None

            # Override the extraction functions for per-ticker correlation mode
            ev_cross_values = all_ev_cross
            ev_cross_lsv_values = [None] * len(tickers)  # LSV not supported with per-ticker corr
            ev_cross_lsv_zero_values = [None] * len(tickers)
            ev_cross_lcm_values = [None] * len(tickers)  # LCM not supported with per-ticker corr

            # Extract mono values from mono_res
            ev_mono_values = [_pt_fv_at(mono_res, i) for i in range(len(mono_corr_order))]
            ra_mono_values_list = [_pt_fv_at(mono_res, len(mono_corr_order) + i)
                                   for i in range(len(mono_corr_order))]

            # Build unified ra_values_by_corr: per-ticker RA from all_ra, keyed by corridor asset
            # In per-ticker mode, all_ra[idx] is the RA for ticker idx's corridor asset
            # Deduplicate: prefer mono RA for mono corridor tickers, then cross RA
            ra_values_by_corr = {}
            # First, store mono RA values (for mono corridor tickers)
            for i, corr in enumerate(mono_corr_order):
                if corr not in ra_values_by_corr and ra_mono_values_list[i] is not None:
                    ra_values_by_corr[corr] = ra_mono_values_list[i]
            # Then, store cross RA values (for cross corridor tickers)
            for idx, corr in enumerate(corr_assets):
                if corr not in ra_values_by_corr and idx < len(all_ra) and all_ra[idx] is not None:
                    ra_values_by_corr[corr] = all_ra[idx]

            t_price_done = time.time() - t_price_call
            self._batch_timings['http_price'] = t_price_done

            # Skip the normal batch pricing below — jump to per-ticker correlation post-processing
            _per_ticker_corr_mode = True

        elif cfg.individual_correlation is not None and cfg.correl_input_method == "Individual Correlations":
            # Single global correlation override (legacy: same correlation for all)
            batch_scenario = pricing_portal.create_scenario_simple(
                mutator_name="GenericMutatorOverrideCorrelationEqEq",
                properties={"CorrelationLevel": cfg.individual_correlation}
            )
            use_scenario = True
            has_per_ticker_corr = False
            _per_ticker_corr_mode = False
            dbg.info("batch", f"Using single global correlation scenario: {cfg.individual_correlation}")
        else:
            _per_ticker_corr_mode = False

        if not _per_ticker_corr_mode:
            # ── Build unified scenario via common helper ──
            unified_scenario = None
            use_lsv_solve = lsv_scenario is not None
            use_lcm_solve = cfg.lcm_params is not None and cfg.lcm_params.get('enabled', False)

            if use_lcm_solve:
                # Validation: LCM requires cross-corridor
                for i, ticker in enumerate(tickers):
                    if ticker == corr_assets[i]:
                        raise ValueError(
                            f"LCM is only applicable to cross-corridor structures "
                            f"(Variance Asset must differ from Corridor Asset). "
                            f"Ticker '{ticker}' has Variance Asset == Corridor Asset."
                        )

            if use_lsv_solve or use_lcm_solve:
                from functions.common.pricing_scenarios import build_unified_scenario
                _ticker_rics = list(set(tickers + corr_assets))
                _lsv_df = cfg.lsv_params if isinstance(cfg.lsv_params, pd.DataFrame) else pd.DataFrame(cfg.lsv_params)
                if 'RIC' in _lsv_df.columns and _lsv_df.index.dtype != object:
                    _lsv_df = _lsv_df.set_index('RIC')

                unified_scenario = build_unified_scenario(
                    pricing_portal,
                    use_lsv=use_lsv_solve,
                    use_lcm=use_lcm_solve,
                    underlying_rics=_ticker_rics,
                    lsv_params=_lsv_df,
                    correl_bump=cfg.lsv_correl_bump,
                    correl_bump_style=cfg.lsv_correl_bump_style,
                    lcm_properties=cfg.lcm_params.get('lcm_properties') if cfg.lcm_params else None,
                )
                _bump_names = []
                if use_lsv_solve:
                    _bump_names = ["LV", "LSV0", "LSV"]
                else:
                    _bump_names = ["LV"]
                if use_lcm_solve:
                    _bump_names.append("LCM")
                dbg.info("batch",
                         f"Unified scenario: {len(_bump_names)} bumps {_bump_names} for {len(_ticker_rics)} RICs")
                use_scenario = True

            # Single pricing call with all instruments
            effective_scenario = unified_scenario if unified_scenario is not None else batch_scenario

            # Build metrics list based on vol_mode
            batch_metrics = [pricing_portal.create_metric("FairValue")]
            include_atmf = cfg.vol_mode in ("ATMF", "ATMF+ATMS")
            include_atms = cfg.vol_mode in ("ATMS", "ATMF+ATMS")
            if include_atmf:
                batch_metrics.append(pricing_portal.create_metric("QueryLocalCcyVol", atmf_params))
            elif include_atms:
                # ATMS-only mode: use ATMS as the single vol metric in main batch
                atms_params = [
                    pricing_portal.create_metric_parameter("AnchorType", "Spot"),
                    pricing_portal.create_metric_parameter("MaturityList", str(excel_date)),
                    pricing_portal.create_metric_parameter("MaturityType", "Absolute"),
                    pricing_portal.create_metric_parameter("Strikes", "1"),
                ]
                batch_metrics.append(pricing_portal.create_metric("QueryLocalCcyVol", atms_params))
            batch_metrics.append(pricing_portal.create_metric("Correlation", corr_params))

            batch_res = _price_in_batches(
                instruments,
                metrics=batch_metrics,
                price_id="Price",
                scenario=effective_scenario,
            )
            t_price_done = time.time() - t_price_call
            dbg.info("batch", f"[TIMING] HTTP price call: {t_price_done:.3f}s ({len(instruments)} instruments)")
            self._batch_timings['http_price'] = t_price_done

            # Second call for ATMS only needed when BOTH ATMF+ATMS requested
            # (portal rejects duplicate QueryLocalCcyVol metric names)
            batch_res_atms = None
            if include_atmf and include_atms:
                atms_params = [
                    pricing_portal.create_metric_parameter("AnchorType", "Spot"),
                    pricing_portal.create_metric_parameter("MaturityList", str(excel_date)),
                    pricing_portal.create_metric_parameter("MaturityType", "Absolute"),
                    pricing_portal.create_metric_parameter("Strikes", "1"),
                ]
                batch_res_atms = _price_in_batches(
                    instruments,
                    metrics=[pricing_portal.create_metric("QueryLocalCcyVol", atms_params)],
                    price_id="Price",
                    batch_label="atms",
                )
            dbg.ok("batch", f"priced in {time.time() - t_price_call:.1f}s (main: {t_price_done:.1f}s)")

            # ── Step 4: Extract results ──
            results_map = batch_res
            results_map_atms = batch_res_atms if batch_res_atms else {}


        # ── Unified metric extraction ──
        # The portal can return in different formats. We figure it out from the raw response.
        def _get_metric_for_instrument(global_idx, metric_name):
            """Get a metric value for instrument at global_idx from batch results."""
            # Find which chunk this instrument belongs to
            running_idx = 0
            for chunk_start in sorted(results_map.keys()):
                chunk_data = results_map[chunk_start]
                chunk_size = chunk_data["chunk_size"]
                if global_idx < running_idx + chunk_size:
                    # This instrument is in this chunk
                    local_idx = global_idx - running_idx
                    raw = chunk_data["raw"]

                    # Try Format B first: separate keys "Price", "Price_1", "Price_2"...
                    if local_idx == 0:
                        key = "Price"
                    else:
                        key = f"Price_{local_idx}"

                    if key in raw:
                        entry = raw[key]
                        if isinstance(entry, dict):
                            # Check SimpleScenarioBump wrapping
                            if use_scenario and "SimpleScenarioBump" in entry:
                                bumps = entry["SimpleScenarioBump"]
                                if bumps and isinstance(bumps, list) and len(bumps) > 0:
                                    metric_list = bumps[0].get(metric_name, [])
                                    if metric_list and isinstance(metric_list, list) and len(metric_list) > 0:
                                        return metric_list[0].get("value") if isinstance(metric_list[0], dict) else None
                            metric_list = entry.get(metric_name, [])
                            if metric_list and isinstance(metric_list, list) and len(metric_list) > 0:
                                return metric_list[0].get("value") if isinstance(metric_list[0], dict) else None
                        return None

                    # Try Format A: single "Price" key with arrays
                    if "Price" in raw:
                        entry = raw["Price"]
                        if isinstance(entry, dict):
                            if use_scenario and "SimpleScenarioBump" in entry:
                                bumps = entry["SimpleScenarioBump"]
                                if bumps and isinstance(bumps, list) and len(bumps) > 0:
                                    metric_list = bumps[0].get(metric_name, [])
                                    if metric_list and isinstance(metric_list, list) and local_idx < len(metric_list):
                                        return metric_list[local_idx].get("value") if isinstance(metric_list[local_idx],
                                                                                                 dict) else None
                            metric_list = entry.get(metric_name, [])
                            if metric_list and isinstance(metric_list, list) and local_idx < len(metric_list):
                                return metric_list[local_idx].get("value") if isinstance(metric_list[local_idx],
                                                                                         dict) else None

                    return None
                running_idx += chunk_size
            return None

        def _get_fv(idx):
            val = _get_metric_for_instrument(idx, "FairValue")
            if val is None:
                dbg.warn("batch", f"_get_fv({idx}): no FairValue")
            return val

        def _extract_vol(vol_list, expected_asset):
            """Return QueryLocalCcyVol matching the requested RIC."""
            if not isinstance(vol_list, list):
                return None

            for item in vol_list:
                if isinstance(item, dict) and item.get("PrimaryAssetRef") == expected_asset:
                    return item.get("value")

            return None

        def _get_atmf(idx):
            if not include_atmf:
                return None
            return _get_metric_for_instrument(idx, "QueryLocalCcyVol")

        def _get_atms(idx):
            """Extract ATMS vol from results.
            - In ATMS-only mode: ATMS is the QueryLocalCcyVol in main batch.
            - In ATMF+ATMS mode: ATMS is in the separate batch_res_atms.
            """
            if not include_atms:
                return None
            if not include_atmf:
                # ATMS-only mode: ATMS is in the main results as QueryLocalCcyVol
                return _get_metric_for_instrument(idx, "QueryLocalCcyVol")
            # ATMF+ATMS mode: ATMS is in the separate batch
            if not results_map_atms:
                return None
            running_idx = 0
            for chunk_start in sorted(results_map_atms.keys()):
                chunk_data = results_map_atms[chunk_start]
                chunk_size = chunk_data["chunk_size"]
                if idx < running_idx + chunk_size:
                    local_idx = idx - running_idx
                    raw = chunk_data["raw"]
                    key = "Price" if local_idx == 0 else f"Price_{local_idx}"
                    if key in raw:
                        entry = raw[key]
                        if isinstance(entry, dict):
                            vol_list = entry.get("QueryLocalCcyVol", [])
                            if vol_list and isinstance(vol_list, list) and len(vol_list) > 0:
                                return vol_list[0].get("value") if isinstance(vol_list[0], dict) else None
                    if "Price" in raw:
                        entry = raw["Price"]
                        if isinstance(entry, dict):
                            vol_list = entry.get("QueryLocalCcyVol", [])
                            if vol_list and isinstance(vol_list, list) and local_idx < len(vol_list):
                                return vol_list[local_idx].get("value") if isinstance(vol_list[local_idx],
                                                                                      dict) else None
                    return None
                running_idx += chunk_size
            return None

        def _get_corr(idx):
            val = _get_metric_for_instrument(idx, "Correlation")
            if val is None:
                ticker = tickers[idx] if idx < len(tickers) else f"idx_{idx}"
                dbg.warn("batch", f"_get_corr({idx}): no Correlation for {ticker}")
            return val

        # EV/RA values stored as raw decimals (e.g., 0.008396 for 0.8396%)
        # Display layer adds % sign via format_pct()

        if not _per_ticker_corr_mode:
            if unified_scenario is not None:
                # Unified scenario: extract from named bumps
                # Response Format B: raw["Price"] for idx=0, raw["Price_1"] for idx=1, etc.
                # Each key contains {"LV": [{"FairValue": [...]}], "LSV0": [...], "LSV": [...], "LCM": [...]}
                # Fires for LSV-only, LCM-only, or LSV+LCM modes
                def _get_bump_fv(global_idx, bump_name):
                    """Extract FairValue for instrument at global_idx under named bump."""
                    running_idx = 0
                    for chunk_start in sorted(results_map.keys()):
                        chunk_data = results_map[chunk_start]
                        chunk_size = chunk_data["chunk_size"]
                        if global_idx < running_idx + chunk_size:
                            local_idx = global_idx - running_idx
                            raw = chunk_data["raw"]
                            # Format B: separate keys "Price", "Price_1", "Price_2"...
                            key = "Price" if local_idx == 0 else f"Price_{local_idx}"
                            if key in raw:
                                price_data = raw[key]
                                if isinstance(price_data, dict):
                                    bump_data = price_data.get(bump_name, [])
                                    if isinstance(bump_data, list) and len(bump_data) > 0:
                                        fv_list = bump_data[0].get("FairValue", [])
                                        if fv_list and isinstance(fv_list, list) and len(fv_list) > 0:
                                            return fv_list[0].get("value") if isinstance(fv_list[0], dict) else None
                                return None
                            # Format A fallback: single "Price" key with arrays indexed by position
                            if "Price" in raw:
                                price_data = raw["Price"]
                                if isinstance(price_data, dict):
                                    bump_data = price_data.get(bump_name, [])
                                    if isinstance(bump_data, list) and len(bump_data) > 0:
                                        fv_list = bump_data[0].get("FairValue", [])
                                        if fv_list and isinstance(fv_list, list) and local_idx < len(fv_list):
                                            return fv_list[local_idx].get("value") if isinstance(fv_list[local_idx],
                                                                                                 dict) else None
                            return None
                        running_idx += chunk_size
                    return None

                def _get_bump_vol(global_idx, bump_name, expected_asset):
                    running_idx = 0

                    for chunk_start in sorted(results_map.keys()):
                        chunk_data = results_map[chunk_start]
                        chunk_size = chunk_data["chunk_size"]

                        if global_idx < running_idx + chunk_size:
                            local_idx = global_idx - running_idx
                            raw = chunk_data["raw"]
                            key = "Price" if local_idx == 0 else f"Price_{local_idx}"
                            entry = raw.get(key)

                            if not isinstance(entry, dict):
                                return None

                            return _extract_vol(
                                entry.get("QueryLocalCcyVol", []),
                                expected_asset,
                            )
                        running_idx += chunk_size

                    return None

                def _get_bump_atms(global_idx, bump_name, expected_asset):
                    if not include_atms:
                        return None

                    if not include_atmf:
                        return _get_bump_vol(global_idx, bump_name, expected_asset)

                    if not results_map_atms:
                        return None

                    running_idx = 0

                    for chunk_start in sorted(results_map_atms.keys()):
                        chunk_data = results_map_atms[chunk_start]
                        chunk_size = chunk_data["chunk_size"]

                        if global_idx < running_idx + chunk_size:
                            local_idx = global_idx - running_idx
                            raw = chunk_data["raw"]
                            key = "Price" if local_idx == 0 else f"Price_{local_idx}"
                            entry = raw.get(key)

                            if not isinstance(entry, dict):
                                return None

                            return _extract_vol(
                                entry.get("QueryLocalCcyVol", []),
                                expected_asset,
                            )

                        running_idx += chunk_size

                    return None


                def _get_bump_corr(global_idx, bump_name):
                    """Extract Correlation for instrument under named bump."""
                    running_idx = 0
                    for chunk_start in sorted(results_map.keys()):
                        chunk_data = results_map[chunk_start]
                        chunk_size = chunk_data["chunk_size"]
                        if global_idx < running_idx + chunk_size:
                            local_idx = global_idx - running_idx
                            raw = chunk_data["raw"]
                            # Format B: separate keys "Price", "Price_1", "Price_2"...
                            key = "Price" if local_idx == 0 else f"Price_{local_idx}"
                            if key in raw:
                                price_data = raw[key]
                                if isinstance(price_data, dict):
                                    bump_data = price_data.get(bump_name, [])
                                    if isinstance(bump_data, list) and len(bump_data) > 0:
                                        corr_list = bump_data[0].get("Correlation", [])
                                        if corr_list and isinstance(corr_list, list) and len(corr_list) > 0:
                                            return corr_list[0].get("value") if isinstance(corr_list[0], dict) else None
                                return None
                            # Format A fallback: single "Price" key with arrays indexed by position
                            if "Price" in raw:
                                price_data = raw["Price"]
                                if isinstance(price_data, dict):
                                    bump_data = price_data.get(bump_name, [])
                                    if isinstance(bump_data, list) and len(bump_data) > 0:
                                        corr_list = bump_data[0].get("Correlation", [])
                                        if corr_list and isinstance(corr_list, list) and local_idx < len(corr_list):
                                            return corr_list[local_idx].get("value") if isinstance(corr_list[local_idx],
                                                                                                   dict) else None
                            return None
                        running_idx += chunk_size
                    return None

                # Extract under LV bump (baseline)
                # Layout: [EV_cross×N, EV_mono×M, RA×M]
                ev_cross_values = [_get_bump_fv(i, "LV") for i in range(n_ev)]
                ev_mono_values = [_get_bump_fv(n_ev + i, "LV") for i in range(n_ev_mono)]
                ra_values_by_corr = {mono_corr_order[i]: _get_bump_fv(n_ev + n_ev_mono + i, "LV") for i in range(n_ra)}

                # Extract LSV bumps (EV-cross only)
                ev_cross_lsv_zero_values = [_get_bump_fv(i, "LSV0") for i in range(n_ev)]
                ev_cross_lsv_values = [_get_bump_fv(i, "LSV") for i in range(n_ev)]

                # Extract LSV bumps for mono corridor
                # Layout: [EV_cross×N, EV_mono×M, RA×M, EV_mono_lsv×M, EV_mono_lsv_zero×M]
                # The mono LSV instruments are the regular mono EV FPFs priced under the unified
                # scenario — so we read the LSV and LSV0 bump values from the EV_mono positions.
                _mono_base_offset = n_ev  # EV_mono starts here
                ev_mono_lsv_values = [_get_bump_fv(_mono_base_offset + i, "LSV") for i in range(n_ev_mono)]
                ev_mono_lsv_zero_values = [_get_bump_fv(_mono_base_offset + i, "LSV0") for i in range(n_ev_mono)]

                # Extract LCM bump (EV-cross only, if enabled)
                ev_cross_lcm_values = [_get_bump_fv(i, "LCM") for i in range(n_ev)] if use_lcm_solve else [None] * n_ev

                # ── LCM extraction (first ticker only) ──
                if use_lcm_solve and n_ev > 0:
                    _ra_proof = ra_values_by_corr.get(corr_assets[0])
                    if ev_cross_values[0] is not None and _ra_proof is not None and _ra_proof != 0:
                        import math as _m
                        _strike_lv = _m.sqrt(abs(-ev_cross_values[0] / _ra_proof))
                    if ev_cross_lcm_values[0] is not None and _ra_proof is not None and _ra_proof != 0:
                        import math as _m
                        _strike_lcm = _m.sqrt(abs(-ev_cross_lcm_values[0] / _ra_proof))

                # Vols and correlation from LV bump

                atmf_vols_cross = {tickers[i]: _get_bump_vol(i, "LV", tickers[i]) for i in
                                   range(n_ev)} if include_atmf else {}
                atms_vols_cross = {tickers[i]: _get_bump_atms(i, "LV", tickers[i]) for i in
                                   range(n_ev)} if include_atms else {}
                atmf_vols_mono = {mono_corr_order[i]: _get_bump_vol(n_ev + i, "LV", mono_corr_order[i]) for i in
                                  range(n_ev_mono)} if include_atmf else {}
                atms_vols_mono = {mono_corr_order[i]: _get_bump_atms(n_ev + i, "LV", mono_corr_order[i]) for i in
                                  range(n_ev_mono)} if include_atms else {}




            else:
                # Layout: [EV_cross×N, EV_mono×M, RA×M]
                ev_cross_values = [_get_fv(i) for i in range(n_ev)]
                ev_cross_lsv_zero_values = [None] * n_ev
                ev_cross_lsv_values = [None] * n_ev
                ev_cross_lcm_values = [None] * n_ev
                ev_mono_values = [_get_fv(n_ev + i) for i in range(n_ev_mono)]
                ra_values_by_corr = {mono_corr_order[i]: _get_fv(n_ev + n_ev_mono + i) for i in range(n_ra)}
                # Mono LSV values (not supported in non-scenario mode)
                ev_mono_lsv_zero_values = [None] * n_ev_mono
                ev_mono_lsv_values = [None] * n_ev_mono

                # Vols and correlation from flat (no-scenario) response
                atmf_vols_cross = {tickers[i]: _get_atmf(i) for i in range(n_ev)}
                atms_vols_cross = {tickers[i]: _get_atms(i) for i in range(n_ev)}
                atmf_vols_mono = {mono_corr_order[i]: _get_atmf(n_ev + i) for i in range(n_ev_mono)}
                atms_vols_mono = {mono_corr_order[i]: _get_atms(n_ev + i) for i in range(n_ev_mono)}
        else:
            # Per-ticker correlation mode: values already extracted in the grouped pricing loop
            # Vols: not available from per-ticker path (mono_res doesn't have per-ticker vol)
            # Use empty dicts — ATMF vol will be fetched separately if needed
            atmf_vols_cross = {}
            atms_vols_cross = {}
            atmf_vols_mono = {}
            atms_vols_mono = {}


        # Populate cache
        for t, v in {**atmf_vols_cross, **atmf_vols_mono}.items():
            self.cache.set_atmf_vol(t, matu_stl_0, v, "Forward")
        for t, v in {**atms_vols_cross, **atms_vols_mono}.items():
            self.cache.set_atmf_vol(t, matu_stl_0, v, "Spot")

        # Compute mono strikes (1 per unique corridor asset)
        mono_strike_sq = {}
        for m_idx, corr in enumerate(mono_corr_order):
            ev_m, ra_m = ev_mono_values[m_idx], ra_values_by_corr.get(corr)
            if ev_m is not None and ra_m is not None and ra_m != 0:
                sq = -ev_m / ra_m
                mono_strike_sq[corr] = abs(sq)
                dbg.info("batch", f"mono {corr}: strike={math.sqrt(abs(sq)) * 100:.2f}%")
            else:
                mono_strike_sq[corr] = None
                dbg.warn("batch", f"mono {corr}: pricing failed")

        # ── Step 5: Compute per-ticker results + build solved FPFs ──
        for idx, ticker in enumerate(tickers):
            # Mono corridor index (for LSV extraction)
            m_idx = mono_corr_order.index(corr_assets[idx]) if corr_assets[idx] in mono_corr_order else None
            try:
                ev_val = ev_cross_values[idx]
                ra_val = ra_values_by_corr.get(corr_assets[idx])
                if ev_val is None:
                    pp_errors = []
                    try:
                        # Find the raw response for this instrument
                        running = 0
                        for cs in sorted(results_map.keys()):
                            cd = results_map[cs]
                            if idx < running + cd["chunk_size"]:
                                local = idx - running
                                raw = cd["raw"]
                                key = "Price" if local == 0 else f"Price_{local}"
                                if key in raw and isinstance(raw[key], dict):
                                    pp_errors.append(f"Response keys for {key}: {list(raw[key].keys())}")
                                elif "Price" in raw and isinstance(raw["Price"], dict):
                                    pp_errors.append(f"Format A. 'Price' keys: {list(raw['Price'].keys())}")
                                    fv = raw["Price"].get("FairValue", [])
                                    pp_errors.append(f"FairValue array len={len(fv)}, local_idx={local}")
                                else:
                                    pp_errors.append(f"Raw keys: {list(raw.keys())}")
                                break
                            running += cd["chunk_size"]
                    except Exception as e:
                        pp_errors.append(f"Error extracting debug: {e}")

                    # Build comprehensive error message (lighter)
                    ev_cross_str = str(ev_cross_values[idx]) if idx < len(ev_cross_values) and ev_cross_values[
                        idx] is not None else "None"
                    ra_str = str(ra_values_by_corr.get(corr_assets[idx], "None"))
                    errors_detail = "; ".join(pp_errors) if pp_errors else "No pricing portal errors found"

                    raise ValueError(
                        f"EV-cross pricing failed for '{corr_assets[idx]}' (variance={ticker}, idx={idx}). "
                        f"Portal errors: {errors_detail}. "
                        f"EV-cross={ev_cross_str}, RA={ra_str}")
                if ra_val is None:
                    pp_errors = []
                    pp_errors.append(f"RA instrument idx={n_ev + idx}, ev_val={ev_val}")
                    try:
                        running = 0
                        for cs in sorted(results_map.keys()):
                            cd = results_map[cs]
                            ra_global = n_ev + idx
                            if ra_global < running + cd["chunk_size"]:
                                local = ra_global - running
                                raw = cd["raw"]
                                key = "Price" if local == 0 else f"Price_{local}"
                                if key in raw and isinstance(raw[key], dict):
                                    pp_errors.append(f"RA Response keys for {key}: {list(raw[key].keys())}")
                                elif "Price" in raw:
                                    fv = raw["Price"].get("FairValue", [])
                                    pp_errors.append(f"Format A. FairValue len={len(fv)}, local_idx={local}")
                                break
                            running += cd["chunk_size"]
                    except Exception as e:
                        pp_errors.append(f"Error extracting debug: {e}")

                    errors_detail = "; ".join(pp_errors) if pp_errors else "No pricing portal errors found"

                    raise ValueError(f"RA pricing failed for '{corr_assets[idx]}' (variance={ticker}, idx={idx}). "
                                     f"Portal errors: {errors_detail}. "
                                     f"EV-cross={ev_val}")

                # DEBUG: Log EV/RA values before strike computation
                dbg.ok("STRIKE-COMPUTATION",
                       f"ticker={ticker}, corr={corr_assets[idx]}, EV_cross={ev_val}, RA={ra_val}")

                # Cross strike
                strike_variance = abs(-ev_val / ra_val)
                strike_variance_asset_vol = math.sqrt(strike_variance)
                dbg.ok("STRIKE-COMPUTATION",
                       f"ticker={ticker}, strike_variance={strike_variance}, strike_variance_asset_vol={strike_variance_asset_vol * 100:.2f}%")

                # Mono strike
                linked_variance = mono_strike_sq.get(corr_assets[idx], strike_variance)
                if linked_variance is None:
                    linked_variance = strike_variance
                strike_corridor_asset_vol = math.sqrt(linked_variance)

                strike_vol = strike_variance_asset_vol
                linked_vol = strike_corridor_asset_vol

                # LSV strike (optional): LV + (LSV - LSV_zero) impact
                # Compute BOTH cross and mono LSV strikes separately
                strike_lsv_vol = None  # mono LSV strike
                strike_cross_lsv_vol = None  # cross LSV strike
                _mono_lsv_adjusted_ev = None
                ev_lsv_val = None
                ev_lsv_zero_val = None

                # Cross LSV strike: cross EV + cross LSV impact
                if ev_cross_lsv_values and ev_cross_lsv_values[idx] is not None:
                    _ev_cross_lsv = ev_cross_lsv_values[idx]
                    _ev_cross_lsv0 = ev_cross_lsv_zero_values[idx] if ev_cross_lsv_zero_values else None
                    if _ev_cross_lsv0 is not None and ra_val is not None and ra_val != 0:
                        cross_lsv_impact = _ev_cross_lsv - _ev_cross_lsv0
                        cross_adjusted_ev = ev_val + cross_lsv_impact
                        strike_cross_lsv_vol = math.sqrt(abs(-cross_adjusted_ev / ra_val))

                # Mono LSV strike: mono EV + mono LSV impact
                if m_idx is not None and ev_mono_lsv_values and ev_mono_lsv_values[m_idx] is not None:
                    ev_lsv_val = ev_mono_lsv_values[m_idx]
                    ev_lsv_zero_val = ev_mono_lsv_zero_values[m_idx] if ev_mono_lsv_zero_values else None
                    if ev_lsv_zero_val is not None and ra_val is not None and ra_val != 0:
                        mono_lsv_impact = ev_lsv_val - ev_lsv_zero_val
                        _base_ev = ev_mono_values[m_idx] if ev_mono_values[m_idx] is not None else ev_val
                        _mono_lsv_adjusted_ev = _base_ev + mono_lsv_impact
                        _ra_for_lsv = ra_values_by_corr.get(corr_assets[idx]) if ra_values_by_corr.get(
                            corr_assets[idx]) else ra_val
                        strike_lsv_vol = math.sqrt(abs(-_mono_lsv_adjusted_ev / _ra_for_lsv))

                # LCM strike (optional): sqrt(-EV_LCM / RA)
                ev_lcm_val = ev_cross_lcm_values[idx] if ev_cross_lcm_values else None
                strike_lcm_vol = None
                if ev_lcm_val is not None and ra_val is not None and ra_val != 0:
                    lcm_variance = abs(-ev_lcm_val / ra_val)
                    strike_lcm_vol = math.sqrt(lcm_variance)

                # Build solved FPFs using object clone (no HTTP)
                def _solved_fpf(ref_obj, ticker_name, corr_name, strike_variance):
                    cap = Just((2.5 ** 2 - 1) * strike_variance) if cfg.is_capped else "Nothing"
                    return ref_obj.clone(
                        varianceDetails=ref_obj.varianceDetails.clone(
                            varianceAssetsAndIndexLegDetails=[
                                ref_obj.varianceDetails.varianceAssetsAndIndexLegDetails[0].clone(
                                    asset=ticker_name, basketMultiplier=1, strike=strike_variance,
                                    legCap=cap, legFloor="Nothing", legMultiplier=1.0,
                                )
                            ],
                            isOptionOnVariance=True,
                        ),
                        corridorDefinition=Just(ref_obj.corridorDefinition.value.clone(
                            corridorAssets=[corridorCovarianceSwap_v4.CorridorAssets(
                                corridorAsset=corr_name, corridorMultiplier=1.0, corridorAssetLag=0
                            )]
                        )),
                        koDetails=ref_obj.koDetails.clone(
                            koAssets=[corridorCovarianceSwap_v4.KoAssets(
                                koAsset=ticker_name, koAssetMultiplier=1.0, koAssetLag=0
                            )]
                        ),
                    ).to_fpf_string()

                ticker_ref_obj = ev_cross_ref_objs[schedule_assets[idx]]

                fpf_cross = _solved_fpf(
                    ticker_ref_obj,
                    ticker,
                    corr_assets[idx],
                    strike_variance,
                )

                mono_ev_obj = mono_ref_objs[corr_assets[idx]][0]
                fpf_mono = _solved_fpf(mono_ev_obj, corr_assets[idx], corr_assets[idx], linked_variance)

                # LSV/LCM FPFs (uncapped, with their respective strikes)
                fpf_lsv = None
                if strike_lsv_vol is not None:
                    lsv_var = strike_lsv_vol ** 2
                    fpf_lsv = _solved_fpf(ev_cross_ref_obj, ticker, corr_assets[idx], lsv_var)
                fpf_lcm = None
                if strike_lcm_vol is not None:
                    lcm_var = strike_lcm_vol ** 2
                    fpf_lcm = _solved_fpf(ev_cross_ref_obj, ticker, corr_assets[idx], lcm_var)

                # ATMF vols
                matu_ex, matu_stl = calculate_payment_dates(cfg.last_obs_date, currencies[idx])
                atmf_ref = self._get_atmf_vol(ticker, matu_stl, "Forward")
                atmf_linked = self._get_atmf_vol(corr_assets[idx], matu_stl, "Forward")
                atms_ref = self._get_atmf_vol(ticker, matu_stl, "Spot")
                atms_linked = self._get_atmf_vol(corr_assets[idx], matu_stl, "Spot")
                vol_spread = None
                if atmf_ref is not None and atmf_linked is not None:
                    vol_spread = (linked_vol - strike_vol) - (atmf_linked - atmf_ref)

                # Extract correlation from cross-corridor instrument
                if _per_ticker_corr_mode:
                    # In per-ticker mode, the correlation is what we forced via scenario
                    corr_value = individual_correlations[idx]
                elif unified_scenario is not None:
                    corr_value = _get_bump_corr(idx, "LV")
                else:
                    corr_value = _get_corr(idx)

                # Cap adjustment — analytical proxy for all variants (LV, LSV, LCM)
                _corridor_vol_pct = None
                if atmf_vols_cross.get(corr_assets[idx]) is not None:
                    _corridor_vol_pct = atmf_vols_cross[corr_assets[idx]] * 100
                elif atmf_vols_mono.get(corr_assets[idx]) is not None:
                    _corridor_vol_pct = atmf_vols_mono[corr_assets[idx]] * 100
                _cap_corr = corr_value if corr_value is not None else 0.5
                _cap_region = "US" if currencies[idx] == "USD" else "EU"
                strike_cap_vol, cap_bp = compute_cap_adjusted_strike(
                    k_raw=strike_variance_asset_vol,
                    ra=ra_val,
                    correlation=_cap_corr,
                    corridor_vol_pct=_corridor_vol_pct if _corridor_vol_pct is not None else 30.0,
                    is_capped=cfg.is_capped,
                    region=_cap_region,
                )
                # Analytical proxy for LSV cap-adjusted strikes (cross and mono separately)
                strike_cap_vol_cross_lsv = None
                strike_cap_vol_mono_lsv = None
                if cfg.is_capped and strike_cross_lsv_vol is not None:
                    strike_cap_vol_cross_lsv, _ = compute_cap_adjusted_strike(
                        k_raw=strike_cross_lsv_vol, ra=ra_val, correlation=_cap_corr,
                        corridor_vol_pct=_corridor_vol_pct if _corridor_vol_pct is not None else 30.0,
                        is_capped=True, region=_cap_region,
                    )
                if cfg.is_capped and strike_lsv_vol is not None:
                    strike_cap_vol_mono_lsv, _ = compute_cap_adjusted_strike(
                        k_raw=strike_lsv_vol, ra=ra_val, correlation=_cap_corr,
                        corridor_vol_pct=_corridor_vol_pct if _corridor_vol_pct is not None else 30.0,
                        is_capped=True, region=_cap_region,
                    )
                # Analytical proxy for LCM cap-adjusted strike
                strike_cap_vol_lcm = None
                if cfg.is_capped and strike_lcm_vol is not None:
                    strike_cap_vol_lcm, _ = compute_cap_adjusted_strike(
                        k_raw=strike_lcm_vol, ra=ra_val, correlation=_cap_corr,
                        corridor_vol_pct=_corridor_vol_pct if _corridor_vol_pct is not None else 30.0,
                        is_capped=True, region=_cap_region,
                    )

                # Observation dates from the reference FPF object

                _obs_count = 0
                try:
                    ticker_ref_obj = ev_cross_ref_objs[schedule_assets[idx]]
                    _obs_count = (
                        len(ticker_ref_obj.observationDates) - 1
                        if getattr(ticker_ref_obj, "observationDates", None)
                        else 0
                    )
                except Exception:
                    pass

                indexed_results[idx] = (TickerResult(
                    ticker=ticker,
                    corridor_asset=corr_assets[idx],
                    success=True,
                    strike_variance_asset=strike_variance_asset_vol,
                    strike_corridor_asset=strike_corridor_asset_vol,
                    strike_cap_adjusted=strike_cap_vol,
                    cap_impact_bp=cap_bp,
                    ev_cross=ev_val * 100 if ev_val is not None else None,
                    # range_accrual=ra_val * 100 if ra_val is not None else None,
 ###
                    range_accrual=ra_val * 100 if ra_val is not None else None,
                    range_accrual_mono=ra_values_by_corr.get(corr_assets[idx]) * 100
                    if ra_values_by_corr.get(corr_assets[idx]) is not None else None,
####
                    ev_mono=ev_mono_values[mono_corr_order.index(corr_assets[idx])] * 100 if ev_mono_values[
                                                                                                 mono_corr_order.index(
                                                                                                     corr_assets[
                                                                                                         idx])] is not None else None,
                    ev_cross_lsv=(ev_val + (ev_lsv_val - ev_lsv_zero_val)) * 100 if (
                                ev_lsv_val is not None and ev_lsv_zero_val is not None and ev_val is not None) else None,
                    ev_cross_lcm=ev_lcm_val * 100 if ev_lcm_val is not None else None,
                    ev_mono_lsv_adjusted=_mono_lsv_adjusted_ev * 100 if _mono_lsv_adjusted_ev is not None else None,
                    ev_mono_lsv=ev_mono_lsv_values[m_idx] * 100 if (
                                m_idx is not None and ev_mono_lsv_values and ev_mono_lsv_values[
                            m_idx] is not None) else None,
                    ev_mono_lsv0=ev_mono_lsv_zero_values[m_idx] * 100 if (
                                m_idx is not None and ev_mono_lsv_zero_values and ev_mono_lsv_zero_values[
                            m_idx] is not None) else None,
                    strike_lsv=strike_lsv_vol,
                    strike_cross_lsv=strike_cross_lsv_vol,
                    strike_variance_asset_lcm=strike_lcm_vol,
                    strike_cap_priced_lsv=strike_cap_vol_cross_lsv,
                    strike_cap_priced_lsv_mono=strike_cap_vol_mono_lsv,
                    strike_cap_priced_lcm=strike_cap_vol_lcm,
                    currency=currencies[idx],
                    fpf_string_cross=fpf_cross,
                    fpf_string_mono=fpf_mono,
                    fpf_string_lsv=fpf_lsv,
                    fpf_string_lcm=fpf_lcm,
                    obs_dates_cross=_obs_count if _obs_count > 0 else None,
                    obs_dates_mono=_obs_count if _obs_count > 0 else None,
                    atmf_vol_variance_asset=atmf_ref,
                    atmf_vol_corridor_asset=atmf_linked,
                    atms_vol_variance_asset=atms_ref,
                    atms_vol_corridor_asset=atms_linked,
                    vol_spread=vol_spread,
                    correlation=corr_value,
                ), None)

            except Exception as e:
                dbg.err("batch", f"{corr_assets[idx]}: {e}")
                indexed_results[idx] = (TickerResult(
                    ticker=corr_assets[idx], corridor_asset=corr_assets[idx],
                    success=False, error=str(e),
                ), None)

            if progress_callback:
                try:
                    progress_callback({
                        'ticker': ticker,
                        'status': 'completed' if indexed_results[idx][0].success else 'failed',
                        'completed': idx + 1, 'total': total,
                        'message': indexed_results[idx][0].error or '',
                    })
                except Exception:
                    pass

        # ── Phase 2b: Capped re-pricing (only if is_capped) ──
        if cfg.is_capped:
            t_cap = time.time()
            dbg.step("batch", "Phase 2b: Capped re-pricing...")

            # Collect successful tickers that need capped pricing
            cap_tasks = []  # [(idx, variant, cap_value, strike_theo)]
            for idx, (result, _) in indexed_results.items():
                if not result.success or result.strike_cap_adjusted is None:
                    continue
                # LV: always if is_capped and proxy succeeded
                cap_lv = (2.5 ** 2 - 1) * (result.strike_cap_adjusted ** 2)
                cap_tasks.append((idx, 'lv', cap_lv, result.strike_cap_adjusted))
                # LSV: if cross LSV was computed and proxy succeeded
                if result.strike_cross_lsv is not None:
                    cap_lsv = (2.5 ** 2 - 1) * (
                                result.strike_cap_priced_lsv ** 2) if result.strike_cap_priced_lsv is not None else (
                                                                                                                                2.5 ** 2 - 1) * (
                                                                                                                                result.strike_cross_lsv ** 2)
                    cap_tasks.append((idx, 'lsv', cap_lsv,
                                      result.strike_cap_priced_lsv if result.strike_cap_priced_lsv is not None else result.strike_cross_lsv))
                # LCM: if LCM was computed and proxy succeeded
                if result.strike_variance_asset_lcm is not None:
                    cap_lcm = (2.5 ** 2 - 1) * (
                                result.strike_cap_priced_lcm ** 2) if result.strike_cap_priced_lcm is not None else (
                                                                                                                                2.5 ** 2 - 1) * (
                                                                                                                                result.strike_variance_asset_lcm ** 2)
                    cap_tasks.append((idx, 'lcm', cap_lcm,
                                      result.strike_cap_priced_lcm if result.strike_cap_priced_lcm is not None else result.strike_variance_asset_lcm))
                # MONO: if corridor asset differs from ticker (cross-corridor), build capped mono FPF
                if corr_assets[idx] != tickers[idx] and result.strike_corridor_asset is not None:
                    cap_mono = (2.5 ** 2 - 1) * (result.strike_corridor_asset ** 2)
                    cap_tasks.append((idx, 'mono', cap_mono, result.strike_corridor_asset))
                    # LSV for mono: if LSV was computed for mono (use same cap as mono variant)
                    if result.strike_lsv is not None:
                        cap_lsv_mono = (2.5 ** 2 - 1) * (result.strike_corridor_asset ** 2)
                        cap_tasks.append((idx, 'lsv_mono', cap_lsv_mono, result.strike_corridor_asset))

            if cap_tasks:
                # Build capped FPF instruments
                cap_instruments = []
                cap_task_map = []  # [(idx, variant)] parallel to cap_instruments
                for idx, variant, cap_value, _ in cap_tasks:
                    ticker = tickers[idx]
                    corr = corr_assets[idx]
                    sched_asset = schedule_assets[idx]
                    # Use mono_ref_obj for mono variant, ev_cross_ref_obj for others
                    if variant in ('mono', 'lsv_mono'):
                        ref_pair = mono_ref_objs.get(corr)
                        if ref_pair is None:
                            continue
                        ref_obj = ref_pair[0]  # ev_obj
                        # Mono: corridor asset = variance asset = corr
                        capped_fpf_str = ref_obj.clone(
                            varianceDetails=ref_obj.varianceDetails.clone(
                                varianceAssetsAndIndexLegDetails=[
                                    ref_obj.varianceDetails.varianceAssetsAndIndexLegDetails[0].clone(
                                        asset=corr, basketMultiplier=1, strike=0.000001,
                                        legCap=Just(cap_value), legFloor="Nothing", legMultiplier=1.0,
                                    )
                                ],
                                isOptionOnVariance=True,
                            ),
                            corridorDefinition=Just(ref_obj.corridorDefinition.value.clone(
                                corridorAssets=[corridorCovarianceSwap_v4.CorridorAssets(
                                    corridorAsset=corr, corridorMultiplier=1.0, corridorAssetLag=0
                                )]
                            )),
                            koDetails=ref_obj.koDetails.clone(
                                koAssets=[corridorCovarianceSwap_v4.KoAssets(
                                    koAsset=corr, koAssetMultiplier=1.0, koAssetLag=0
                                )]
                            ),
                        ).to_fpf_string()
                    else:
                        ref_obj = ev_cross_ref_objs[sched_asset]
                        # Clone with cap set (strike=0.000001 — we only need EV, not solved FPF)
                        capped_fpf_str = ref_obj.clone(
                            varianceDetails=ref_obj.varianceDetails.clone(
                                varianceAssetsAndIndexLegDetails=[
                                    ref_obj.varianceDetails.varianceAssetsAndIndexLegDetails[0].clone(
                                        asset=ticker, basketMultiplier=1, strike=0.000001,
                                        legCap=Just(cap_value), legFloor="Nothing", legMultiplier=1.0,
                                    )
                                ],
                                isOptionOnVariance=True,
                            ),
                            corridorDefinition=Just(ref_obj.corridorDefinition.value.clone(
                                corridorAssets=[corridorCovarianceSwap_v4.CorridorAssets(
                                    corridorAsset=corr, corridorMultiplier=1.0, corridorAssetLag=0
                                )]
                            )),
                            koDetails=ref_obj.koDetails.clone(
                                koAssets=[corridorCovarianceSwap_v4.KoAssets(
                                    koAsset=ticker, koAssetMultiplier=1.0, koAssetLag=0
                                )]
                            ),
                        ).to_fpf_string()
                    cap_instruments.append(_make_instrument(capped_fpf_str,
                                                            [ticker, corr] if variant not in ('mono', 'lsv_mono') else [
                                                                corr]))
                    cap_task_map.append((idx, variant))

                # Price capped batch (reuse same scenario as Phase 1)
                effective_cap_scenario = None
                if unified_scenario is not None:
                    effective_cap_scenario = unified_scenario
                elif not _per_ticker_corr_mode:
                    effective_cap_scenario = batch_scenario

                cap_results_map = _price_in_batches(
                    cap_instruments,
                    metrics=[pricing_portal.create_metric("FairValue")],
                    price_id="Price",
                    scenario=effective_cap_scenario,
                    batch_label="capped",
                )

                # Extract capped EV values
                from functions.common.pricing_scenarios import extract_scenario_metric as _extract_metric

                def _get_cap_fv(inst_idx, bump_name=None):
                    """Get FairValue for instrument at inst_idx in cap_results_map."""
                    running = 0
                    for cs in sorted(cap_results_map.keys()):
                        cd = cap_results_map[cs]
                        if inst_idx < running + cd["chunk_size"]:
                            local_idx = inst_idx - running
                            raw = cd["raw"]
                            if bump_name:
                                # Use the proven extraction utility
                                val = _extract_metric(raw, "Price", bump_name, local_idx, "FairValue")
                                if val is not None:
                                    return val
                            # No bump or fallback: read raw FairValue
                            key = "Price" if local_idx == 0 else f"Price_{local_idx}"
                            if key in raw and isinstance(raw[key], dict):
                                fv_list = raw[key].get("FairValue", [])
                                if fv_list and isinstance(fv_list, list):
                                    val = fv_list[0]
                                    return val.get("value") if isinstance(val, dict) else val
                            # Format A: single Price key with array
                            if "Price" in raw and isinstance(raw["Price"], dict):
                                entry = raw["Price"]
                                if not bump_name:
                                    fv_list = entry.get("FairValue", [])
                                    if fv_list and isinstance(fv_list, list) and local_idx < len(fv_list):
                                        val = fv_list[local_idx]
                                        return val.get("value") if isinstance(val, dict) else val
                            return None
                        running += cd["chunk_size"]
                    return None

                # Compute real priced capped strikes and build final FPF strings
                def _build_capped_fpf_string(ref_obj, ticker, corr, strike_vol):
                    """Build solved capped FPF string with final priced strike."""
                    strike_var = strike_vol ** 2
                    cap_val = (2.5 ** 2 - 1) * strike_var
                    return ref_obj.clone(
                        varianceDetails=ref_obj.varianceDetails.clone(
                            varianceAssetsAndIndexLegDetails=[
                                ref_obj.varianceDetails.varianceAssetsAndIndexLegDetails[0].clone(
                                    asset=ticker, basketMultiplier=1, strike=strike_var,
                                    legCap=Just(cap_val), legFloor="Nothing", legMultiplier=1.0,
                                )
                            ],
                            isOptionOnVariance=True,
                        ),
                        corridorDefinition=Just(ref_obj.corridorDefinition.value.clone(
                            corridorAssets=[corridorCovarianceSwap_v4.CorridorAssets(
                                corridorAsset=corr, corridorMultiplier=1.0, corridorAssetLag=0
                            )]
                        )),
                        koDetails=ref_obj.koDetails.clone(
                            koAssets=[corridorCovarianceSwap_v4.KoAssets(
                                koAsset=ticker, koAssetMultiplier=1.0, koAssetLag=0
                            )]
                        ),
                    ).to_fpf_string()

                for inst_idx, (idx, variant) in enumerate(cap_task_map):
                    ra_val = ra_values_by_corr.get(corr_assets[idx])
                    if ra_val is None or ra_val == 0:
                        continue

                    result_obj = indexed_results[idx][0]
                    ticker = tickers[idx]
                    corr = corr_assets[idx]
                    sched_asset = schedule_assets[idx]
                    ref_obj = ev_cross_ref_objs[sched_asset]

                    if variant == 'lv':
                        ev_cap = _get_cap_fv(inst_idx, "LV") if use_scenario else _get_cap_fv(inst_idx)
                        if ev_cap is not None:
                            real_cap_strike = math.sqrt(abs(-ev_cap / ra_val))
                            result_obj.strike_cap_priced_lv = real_cap_strike
                            result_obj.ev_cap_cross_lv = ev_cap
                            result_obj.fpf_string_cap_lv = _build_capped_fpf_string(ref_obj, ticker, corr,
                                                                                    real_cap_strike)
                            dbg.ok("CAP-PRICED", f"{ticker}: LV capped strike = {real_cap_strike * 100:.2f}%")

                    elif variant == 'lsv':
                        # LSV decomposition: ev_lv_capped + (ev_lsv_capped - ev_lsv0_capped)
                        ev_cap_lv = _get_cap_fv(inst_idx, "LV")
                        ev_cap_lsv0 = _get_cap_fv(inst_idx, "LSV0")
                        ev_cap_lsv = _get_cap_fv(inst_idx, "LSV")
                        dbg.info("CAP-LSV-DEBUG",
                                 f"{ticker}: inst_idx={inst_idx}, ev_cap_lv={ev_cap_lv}, ev_cap_lsv0={ev_cap_lsv0}, ev_cap_lsv={ev_cap_lsv}, scenario={effective_cap_scenario is not None}")
                        if ev_cap_lv is not None and ev_cap_lsv0 is not None and ev_cap_lsv is not None:
                            lsv_impact = ev_cap_lsv - ev_cap_lsv0
                            adjusted_ev = ev_cap_lv + lsv_impact
                            real_cap_strike = math.sqrt(abs(-adjusted_ev / ra_val))
                            result_obj.strike_cap_priced_lsv = real_cap_strike
                            result_obj.ev_cap_cross_lsv0 = ev_cap_lsv0
                            result_obj.ev_cap_cross_lsv = ev_cap_lsv
                            result_obj.fpf_string_cap_lsv = _build_capped_fpf_string(ref_obj, ticker, corr,
                                                                                     real_cap_strike)
                            dbg.ok("CAP-PRICED", f"{ticker}: LSV capped strike = {real_cap_strike * 100:.2f}%")

                    elif variant == 'lcm':
                        ev_cap_lcm = _get_cap_fv(inst_idx, "LCM")
                        if ev_cap_lcm is not None:
                            real_cap_strike = math.sqrt(abs(-ev_cap_lcm / ra_val))
                            result_obj.strike_cap_priced_lcm = real_cap_strike
                            result_obj.ev_cap_cross_lcm = ev_cap_lcm
                            result_obj.fpf_string_cap_lcm = _build_capped_fpf_string(ref_obj, ticker, corr,
                                                                                     real_cap_strike)
                            dbg.ok("CAP-PRICED", f"{ticker}: LCM capped strike = {real_cap_strike * 100:.2f}%")

                    elif variant == 'mono':
                        # Mono: use mono_ref_obj and corr as asset
                        ref_pair = mono_ref_objs.get(corr)
                        if ref_pair is None:
                            continue
                        mono_ref_obj = ref_pair[0]  # ev_obj
                        ev_cap_mono = _get_cap_fv(inst_idx, "LV") if use_scenario else _get_cap_fv(inst_idx)
                        if ev_cap_mono is not None:
                            real_cap_strike = math.sqrt(abs(-ev_cap_mono / ra_val))
                            result_obj.strike_cap_priced_mono = real_cap_strike
                            result_obj.ev_cap_mono_lv = ev_cap_mono
                            result_obj.fpf_string_cap_mono = _build_capped_fpf_string(mono_ref_obj, corr, corr,
                                                                                      real_cap_strike)
                            dbg.ok("CAP-PRICED", f"{corr}: Mono capped strike = {real_cap_strike * 100:.2f}%")

                    elif variant == 'lsv_mono':
                        # LSV for mono: ev_cap_lv + (ev_cap_lsv - ev_cap_lsv0)
                        ref_pair = mono_ref_objs.get(corr)
                        if ref_pair is None:
                            continue
                        mono_ref_obj = ref_pair[0]  # ev_obj
                        ev_cap_lv = _get_cap_fv(inst_idx, "LV")
                        ev_cap_lsv0 = _get_cap_fv(inst_idx, "LSV0")
                        ev_cap_lsv = _get_cap_fv(inst_idx, "LSV")
                        dbg.info("CAP-LSV-MONO-DEBUG",
                                 f"{corr}: inst_idx={inst_idx}, ev_cap_lv={ev_cap_lv}, ev_cap_lsv0={ev_cap_lsv0}, ev_cap_lsv={ev_cap_lsv}, scenario={effective_cap_scenario is not None}")
                        if ev_cap_lv is not None and ev_cap_lsv0 is not None and ev_cap_lsv is not None:
                            lsv_impact = ev_cap_lsv - ev_cap_lsv0
                            adjusted_ev = ev_cap_lv + lsv_impact
                            real_cap_strike = math.sqrt(abs(-adjusted_ev / ra_val))
                            result_obj.strike_cap_priced_lsv_mono = real_cap_strike
                            result_obj.ev_cap_mono_lsv0 = ev_cap_lsv0
                            result_obj.ev_cap_mono_lsv = ev_cap_lsv
                            result_obj.fpf_string_cap_lsv_mono = _build_capped_fpf_string(mono_ref_obj, corr, corr,
                                                                                          real_cap_strike)
                            dbg.ok("CAP-PRICED", f"{corr}: LSV Mono capped strike = {real_cap_strike * 100:.2f}%")

                dbg.ok("batch",
                       f"Phase 2b capped pricing: {len(cap_instruments)} instruments in {time.time() - t_cap:.1f}s")
                self._batch_timings['capped_pricing'] = time.time() - t_cap
            else:
                dbg.info("batch", "Phase 2b: no tickers eligible for capped re-pricing")

        # ── Timing ──
        n_ok = sum(1 for r, _ in indexed_results.values() if r.success)
        dbg.ok("batch", f"done: {n_ok}/{total} in {time.time() - t_fpf:.1f}s "
                        f"(FPF={time.time() - t_fpf - t_price_done:.1f}s, price={t_price_done:.1f}s)")
        return indexed_results

    # ─── Process Single Ticker ───────────────────────────────────────────

    def _process_ticker(self, row: pd.Series) -> Tuple[TickerResult, Any]:
        """Process a single ticker. Uses CrossCorridorVarianceSwap for compatibility."""
        cfg = self.config
        ticker = row['Tickers']
        corridor_asset = row.get('Corridor Condition Asset', ticker) if cfg.is_cross_corridor else ticker
        # Resolve currency (cached)
        currency = None
        if 'Currency' in row and row.get('Currency'):
            currency = row['Currency']
        else:
            currency = self.cache.get_currency(ticker)
        # Resolve individual correlation
        individual_correl = None
        if cfg.is_cross_corridor and cfg.correl_input_method == "Individual Correlations":
            if 'Correlation' in row and row.get('Correlation'):
                try:
                    individual_correl = float(row['Correlation']) / 100
                except (ValueError, TypeError):
                    pass
        # Create swap object (needed for FPF string generation, obs date counting, etc.)
        swap = CrossCorridorVarianceSwap(
            ref_asset=ticker,
            last_obs_date=cfg.last_obs_date,
            strike_date=cfg.strike_date,
            uvar=cfg.uvar,
            dvar=cfg.dvar,
            linked_asset=corridor_asset,
            is_capped=cfg.is_capped,
            eqeq_lambda=cfg.eqeq_lambda,
            correl_floor=cfg.correl_floor,
            eqfx_shift=cfg.eqfx_shift,
            individual_correlation=individual_correl,
            currency=currency,
            compute_atmf=(cfg.vol_mode != "OFF"),
        )
        # Solve mode handled by _solve_batch_ev_ra, this only does pricing
        return self._price_ticker(swap, row)

    def _price_ticker(self, swap, row) -> Tuple[TickerResult, Any]:
        """Price a ticker with given strikes (non-solve mode)."""
        cfg = self.config
        ticker = swap.ref_asset
        corridor_asset = swap.linked_asset
        currency = swap.currency

        # Get strikes from row
        if 'Strikes (%)' in row and row.get('Strikes (%)'):
            swap.strike_variance_asset = float(row['Strikes (%)']) / 100
            swap.strike_corridor_asset = swap.strike_variance_asset
        elif 'Strike Cross Corridor (%)' in row and row.get('Strike Cross Corridor (%)'):
            swap.strike_variance_asset = float(row['Strike Cross Corridor (%)']) / 100
            swap.strike_corridor_asset = float(
                row.get('Strike Mono Var Swap (%)', row['Strike Cross Corridor (%)'])) / 100
        elif 'Strike Cross (%)' in row and row.get('Strike Cross (%)'):
            # Legacy column names
            swap.strike_variance_asset = float(row['Strike Cross (%)']) / 100
            swap.strike_corridor_asset = float(row.get('Strike Mono (%)', row['Strike Cross (%)'])) / 100
        if swap.strike_variance_asset is None:
            return TickerResult(
                ticker=ticker, corridor_asset=corridor_asset,
                success=False, error="No strike provided for pricing mode"
            ), swap

        # Expected Var mode (strike=0): skip full pricing, return zero-strike mid
        if swap.strike_variance_asset == 0.0:
            zero_mid = getattr(swap, 'compute_zero_strike_mid', lambda: None)()
            obs_counts = swap.count_observation_dates() if hasattr(swap, 'count_observation_dates') else {}
            return TickerResult(
                ticker=ticker,
                corridor_asset=corridor_asset,
                success=True,
                strike_variance_asset=0.0,
                strike_corridor_asset=0.0,
                mid_variance_asset=None,
                mid_corridor_asset=None,
                zero_strike_mid_variance_asset=zero_mid,
                currency=swap.currency,
                obs_dates_cross=obs_counts.get("Cross Corridor Obs Dates") or obs_counts.get("Obs Dates"),
                obs_dates_mono=obs_counts.get("Mono Corridor Obs Dates"),
            ), swap

        # ── Build FPF with correct strikes and price directly ──
        _ensure_portal()

        # Build corridor FPF with user-provided strikes (variance = strike²)
        strike_var_ref = swap.strike_variance_asset ** 2
        strike_var_linked = swap.strike_corridor_asset ** 2

        ev_fpf = build_corridor_fpf(
            tickers=[ticker], last_obs_date=cfg.last_obs_date,
            strike_date=cfg.strike_date, strikes=[strike_var_ref], weights=[1.0],
            low_barrier=cfg.dvar, high_barrier=cfg.uvar, is_capped=cfg.is_capped,
            corr_asset=corridor_asset, schedule_calendar_asset=corridor_asset if ticker != corridor_asset else None,
            currency=currency, use_parameters=False,
        )

        # Populate fpf_obj_cross so count_observation_dates() works
        try:
            swap.fpf_obj_cross = FPFUnifiedEconomicsWrapper.from_data(ev_fpf, script_cls=corridorCovarianceSwap_v4)
        except Exception:
            pass

        # Load underlyings
        all_rics = list(set([ticker, corridor_asset]))
        underlyings = [_load_instrument_impl(ric) for ric in all_rics]

        # Create instrument
        ev_instrument = pricing_portal.create_fpf(
            fpf_string=ev_fpf, instrument_ccy=currency,
            underlyings=underlyings, premium_date=datetime.datetime.now().date(),
        )

        # Build metrics
        matu_ex_0, matu_stl_0 = calculate_payment_dates(cfg.last_obs_date, currency)
        _excel_epoch = date(1899, 12, 30)
        excel_date = (matu_stl_0 - _excel_epoch).days

        atmf_params = [
            pricing_portal.create_metric_parameter("AnchorType", "Forward"),
            pricing_portal.create_metric_parameter("MaturityList", str(excel_date)),
            pricing_portal.create_metric_parameter("MaturityType", "Absolute"),
            pricing_portal.create_metric_parameter("Strikes", "1"),
        ]

        metrics = [
            pricing_portal.create_metric("FairValue"),
            pricing_portal.create_metric("QueryLocalCcyVol", atmf_params),
            pricing_portal.create_metric("Correlation", [
                pricing_portal.create_metric_parameter("MaturityList", str(excel_date)),
                pricing_portal.create_metric_parameter("MaturityType", "Absolute"),
            ]),
        ]

        # Build instrument list — cross + mono if different assets
        instruments = [ev_instrument]
        if ticker != corridor_asset:
            mono_fpf = build_corridor_fpf(
                tickers=[corridor_asset], last_obs_date=cfg.last_obs_date,
                strike_date=cfg.strike_date, strikes=[strike_var_linked], weights=[1.0],
                low_barrier=cfg.dvar, high_barrier=cfg.uvar, is_capped=cfg.is_capped,
                corr_asset=corridor_asset, schedule_calendar_asset=None,
                currency=currency, use_parameters=False,
            )
            mono_instrument = pricing_portal.create_fpf(
                fpf_string=mono_fpf, instrument_ccy=currency,
                underlyings=[_load_instrument_impl(corridor_asset)],
                premium_date=datetime.datetime.now().date(),
            )
            instruments.append(mono_instrument)

        # Determine model context
        is_index_product = corridor_asset.startswith(".") and ticker.startswith(".")
        if is_index_product:
            _model = "EMEA-Index-MC-LV-MultiAsset"
            model_params = {"ACEqFxShift": str(cfg.eqfx_shift)}
        else:
            _model = "EMEA-Stocks-MC-LV-MultiAsset"
            model_params = {
                'ACEqEqSpread': str(cfg.eqeq_lambda),
                "EqEqCorrFloor": str(cfg.correl_floor),
                "ACEqFxShift": str(cfg.eqfx_shift),
            }
        _apply_special_rics_param(model_params, [ticker, corridor_asset])
        model_context = pricing_portal.create_model_context(_model, instrument_model_parameters=model_params)
        valuation_date = cfg.strike_date if isinstance(cfg.strike_date,
                                                       datetime.datetime) else datetime.datetime.combine(
            cfg.strike_date if isinstance(cfg.strike_date, datetime.date) else datetime.datetime.strptime(
                str(cfg.strike_date), "%Y-%m-%d").date(),
            datetime.datetime.min.time()
        )

        # Price in single batch call
        batch_res = pricing_portal.price(
            price_id="Price",
            instruments=instruments,
            valuation_date=valuation_date,
            calculation_parameters={},
            model_context=model_context,
            overridden_snap_name=live_snap["name"],
            metrics=metrics,
        )

        # Extract results
        results_map = batch_res.get("results", {})

        def _get_fv(idx):
            key = "Price" if idx == 0 else f"Price_{idx}"
            fv_list = results_map.get(key, {}).get("FairValue", [])
            return fv_list[0].get("value") if fv_list else None

        def _get_vol(idx):
            key = "Price" if idx == 0 else f"Price_{idx}"
            vol_list = results_map.get(key, {}).get("QueryLocalCcyVol", [])
            return vol_list[0].get("value") if vol_list else None

        def _get_corr(idx):
            key = "Price" if idx == 0 else f"Price_{idx}"
            corr_list = results_map.get(key, {}).get("Correlation", [])
            if not corr_list:
                # Check if we have individual correlation input
                if hasattr(cfg, 'correl_input_method') and cfg.correl_input_method == "Individual Correlations":
                    # For individual correlations, we'll use the input value instead of pricing result
                    # This will be handled by passing the individual correlation through the result
                    dbg.info("batch", f"_get_corr({idx}): No correlation in pricing results for {ticker}. "
                                      f"Individual correlation mode detected - will use input value.")
            return corr_list[0].get("value") if corr_list else None

        # Cross is index 0, mono is index 1 (if exists)
        mid_va = _get_fv(0)
        mid_ca = _get_fv(1) if len(instruments) > 1 else mid_va
        atmf_vol_va = _get_vol(0)
        atmf_vol_ca = _get_vol(1) if len(instruments) > 1 else atmf_vol_va
        correlation = _get_corr(0)

        obs_counts = swap.count_observation_dates() if hasattr(swap, 'count_observation_dates') else {}

        return TickerResult(
            ticker=ticker,
            corridor_asset=corridor_asset,
            success=True,
            strike_variance_asset=swap.strike_variance_asset,  # already vol (set from user input)
            strike_corridor_asset=swap.strike_corridor_asset,  # already vol (set from user input)
            mid_variance_asset=mid_va,
            mid_corridor_asset=mid_ca,
            atmf_vol_variance_asset=atmf_vol_va,
            atmf_vol_corridor_asset=atmf_vol_ca,
            correlation=correlation,
            currency=currency,
            obs_dates_cross=obs_counts.get("Cross Corridor Obs Dates") or obs_counts.get("Obs Dates"),
            obs_dates_mono=obs_counts.get("Mono Corridor Obs Dates"),
            fpf_string_cross=ev_fpf,
        ), swap

    # ─── Helpers ─────────────────────────────────────────────────────────

    def _get_atmf_vol(self, ticker: str, maturity: date, anchor: str = "Forward") -> Optional[float]:
        """Get implied vol with caching. anchor='Forward' → ATMF, anchor='Spot' → ATMS.
        Returns None if not available in cache (no hidden HTTP fallback)."""
        found, vol = self.cache.get_atmf_vol(ticker, maturity, anchor)
        if found:
            return vol
        return None

    # _batch_atmf_vols removed — vols are now extracted from the main batch price call
    # via QueryLocalCcyVol metric (see _solve_batch_ev_ra, Step 3).

    def _apply_charges(self, results: List[TickerResult], swap_objects: List, charge_function: Callable):
        """Apply charge function to vol spread results."""
        for result in results:
            if not result.success or result.vol_spread is None:
                continue
            charge_input = result.vol_spread * 100
            result.lsv_charge = charge_function(charge_input) / 100

    def _build_results_df(self, results: List[TickerResult]) -> Optional[pd.DataFrame]:
        """Build display DataFrame."""
        if not results:
            return None
        cfg = self.config
        rows = []
        _sqrt = math.sqrt
        for r in results:
            if cfg.is_cross_corridor and r.ticker != r.corridor_asset:
                row = {
                    'Index Ticker': r.ticker,
                    'Corridor Asset': r.corridor_asset,
                    'Currency': r.currency,
                }
                if r.success:
                    _is_capped = cfg.is_capped
                    _uncap_label = " (Uncapped)" if not _is_capped else ""
                    # ── Cross Corridor Strikes ──
                    row[
                        f'Strike Cross Corr LV{_uncap_label} (%)'] = f"{r.strike_variance_asset * 100:.2f}%" if r.strike_variance_asset else 'FAILED'
                    if r.strike_cross_lsv is not None:
                        row[f'Strike Cross Corr LSV{_uncap_label} (%)'] = f"{r.strike_cross_lsv * 100:.2f}%"
                    if r.strike_variance_asset_lcm is not None:
                        row[f'Strike Cross Corr LCM{_uncap_label} (%)'] = f"{r.strike_variance_asset_lcm * 100:.2f}%"
                    if _is_capped:
                        if r.cap_impact_bp is not None and r.cap_impact_bp > 0:
                            row['Cap Theoretical Impact (bp)'] = f"{r.cap_impact_bp:.2f}"
                            row[
                                'Strike Cross Corr Cap Adjusted (%)'] = f"{r.strike_cap_adjusted * 100:.2f}%" if r.strike_cap_adjusted else 'N/A'
                        if r.strike_cap_priced_lv is not None:
                            row['Strike Cross Corr Cap Priced LV (%)'] = f"{r.strike_cap_priced_lv * 100:.2f}%"
                        if r.strike_cap_priced_lsv is not None:
                            row['Strike Cross Corr Cap Priced LSV (%)'] = f"{r.strike_cap_priced_lsv * 100:.2f}%"
                        if r.strike_cap_priced_lcm is not None:
                            row['Strike Cross Corr Cap Priced LCM (%)'] = f"{r.strike_cap_priced_lcm * 100:.2f}%"
                    # ── Mono Corridor Strikes ──
                    row[
                        f'Strike Mono Corr LV{_uncap_label} (%)'] = f"{r.strike_corridor_asset * 100:.2f}%" if r.strike_corridor_asset else 'FAILED'
                    if r.strike_lsv is not None:
                        row[f'Strike Mono Corr LSV{_uncap_label} (%)'] = f"{r.strike_lsv * 100:.2f}%"
                    if _is_capped:
                        if r.strike_cap_priced_mono is not None:
                            row['Strike Mono Corr Cap Priced LV (%)'] = f"{r.strike_cap_priced_mono * 100:.2f}%"
                        if r.strike_cap_priced_lsv_mono is not None:
                            row['Strike Mono Corr Cap Priced LSV (%)'] = f"{r.strike_cap_priced_lsv_mono * 100:.2f}%"
                    # ── Spread ──
                    if _is_capped:
                        # Capped: spread = Cap Priced Mono LV - Cap Priced Cross LV
                        if r.strike_cap_priced_mono is not None and r.strike_cap_priced_lv is not None:
                            row[
                                'Strike Spread LV (%)'] = f"{(r.strike_cap_priced_mono - r.strike_cap_priced_lv) * 100:.2f}%"
                        if r.strike_cap_priced_lsv_mono is not None and r.strike_cap_priced_lsv is not None:
                            row[
                                'Strike Spread LSV (%)'] = f"{(r.strike_cap_priced_lsv_mono - r.strike_cap_priced_lsv) * 100:.2f}%"
                    else:
                        # Uncapped: spread = Mono LV - Cross LV (vol space)
                        if r.strike_corridor_asset and r.strike_variance_asset:
                            row[
                                'Strike Spread LV (%)'] = f"{(r.strike_corridor_asset - r.strike_variance_asset) * 100:.2f}%"
                        if r.strike_lsv is not None and r.strike_cross_lsv is not None:
                            row['Strike Spread LSV (%)'] = f"{(r.strike_lsv - r.strike_cross_lsv) * 100:.2f}%"
                    # ── EV / RA (all in xx.xx% format) ──
                    if r.ev_cross is not None:
                        row['EV Cross LV (%)'] = f"{r.ev_cross:.2f}%"
                    if r.ev_cross_lsv is not None:
                        row['EV Cross LSV Adjusted (%)'] = f"{r.ev_cross_lsv:.2f}%"
                    if r.ev_cross_lcm is not None:
                        row['EV Cross LCM (%)'] = f"{r.ev_cross_lcm:.2f}%"

                    if r.range_accrual is not None:
                        row['RA Cross (%)'] = f"{r.range_accrual:.2f}%"
                    if r.range_accrual_mono is not None:
                        row['RA Mono (%)'] = f"{r.range_accrual_mono:.2f}%"

                    if r.ev_mono is not None:
                        row['EV Mono LV (%)'] = f"{r.ev_mono:.2f}%"
                    if r.ev_mono_lsv is not None:
                        row['EV Mono LSV (%)'] = f"{r.ev_mono_lsv:.2f}%"
                    if r.ev_mono_lsv0 is not None:
                        row['EV Mono LSV0 (%)'] = f"{r.ev_mono_lsv0:.2f}%"
                    if r.ev_mono_lsv_adjusted is not None:
                        row['EV Mono LSV Adjusted (%)'] = f"{r.ev_mono_lsv_adjusted:.2f}%"
                    # ── Capped EV components (xx.xx%) ──
                    if r.ev_cap_cross_lv is not None:
                        row['EV Cap Cross LV (%)'] = f"{r.ev_cap_cross_lv * 100:.2f}%"
                    if r.ev_cap_cross_lsv0 is not None:
                        row['EV Cap Cross LSV0 (%)'] = f"{r.ev_cap_cross_lsv0 * 100:.2f}%"
                    if r.ev_cap_cross_lsv is not None:
                        row['EV Cap Cross LSV Adjusted (%)'] = f"{r.ev_cap_cross_lsv * 100:.2f}%"
                    if r.ev_cap_cross_lcm is not None:
                        row['EV Cap Cross LCM (%)'] = f"{r.ev_cap_cross_lcm * 100:.2f}%"
                    if r.ev_cap_mono_lv is not None:
                        row['EV Cap Mono LV (%)'] = f"{r.ev_cap_mono_lv * 100:.2f}%"
                    if r.ev_cap_mono_lsv0 is not None:
                        row['EV Cap Mono LSV0 (%)'] = f"{r.ev_cap_mono_lsv0 * 100:.2f}%"
                    if r.ev_cap_mono_lsv is not None:
                        row['EV Cap Mono LSV Adjusted (%)'] = f"{r.ev_cap_mono_lsv * 100:.2f}%"
                    # ── Obs dates / Vols / Correlation ──
                    if r.obs_dates_cross is not None:
                        row['Obs Dates Cross'] = r.obs_dates_cross
                    if r.obs_dates_mono is not None:
                        row['Obs Dates Mono'] = r.obs_dates_mono
                    if r.atmf_vol_variance_asset is not None:
                        row['ATMF Vol Variance Asset (%)'] = f"{r.atmf_vol_variance_asset * 100:.2f}%"
                    if r.atmf_vol_corridor_asset is not None:
                        row['ATMF Vol Corridor Asset (%)'] = f"{r.atmf_vol_corridor_asset * 100:.2f}%"
                    if r.atms_vol_variance_asset is not None:
                        row['ATMS Vol Variance Asset (%)'] = f"{r.atms_vol_variance_asset * 100:.2f}%"
                    if r.atms_vol_corridor_asset is not None:
                        row['ATMS Vol Corridor Asset (%)'] = f"{r.atms_vol_corridor_asset * 100:.2f}%"
                    if r.vol_spread is not None:
                        row['Vol Spread (%)'] = f"{r.vol_spread * 100:.2f}%"
                    if r.correlation is not None:
                        row['Correlation'] = f"{r.correlation * 100:.2f}%"
                    if r.mid_variance_asset is not None:
                        row['Mid Variance Asset (%)'] = f"{r.mid_variance_asset * 100:.2f}%"
                    if r.mid_corridor_asset is not None:
                        row['Mid Corridor Asset (%)'] = f"{r.mid_corridor_asset * 100:.2f}%"
                    # Price-mode LSV columns (only shown when LSV enabled)
                    if r.mid_variance_asset_lv is not None:
                        row['FV Variance Asset LV (%)'] = f"{r.mid_variance_asset_lv * 100:.2f}%"
                    if r.mid_variance_asset_lsv0 is not None:
                        row['FV Variance Asset LSV0 (%)'] = f"{r.mid_variance_asset_lsv0 * 100:.2f}%"
                    if r.mid_variance_asset_lsv is not None:
                        row['FV Variance Asset LSV (%)'] = f"{r.mid_variance_asset_lsv * 100:.2f}%"
                    if r.mid_variance_asset_lsv is not None and r.mid_variance_asset_lsv0 is not None:
                        row[
                            'LSV Impact Variance Asset (%)'] = f"{(r.mid_variance_asset_lsv - r.mid_variance_asset_lsv0) * 100:.2f}%"
                    # LCM columns
                    if r.mid_variance_asset_lcm is not None:
                        row['FV Variance Asset LCM (%)'] = f"{r.mid_variance_asset_lcm * 100:.2f}%"
                    if r.mid_variance_asset_lv is not None and r.mid_variance_asset_lcm is not None:
                        row[
                            'LCM Impact Variance Asset (%)'] = f"{(r.mid_variance_asset_lcm - r.mid_variance_asset_lv) * 100:.2f}%"
                    if r.zero_strike_mid_variance_asset is not None:
                        row['Realized Var Variance Asset (%)'] = f"{r.zero_strike_mid_variance_asset * 100:.2f}%"
                    if r.zero_strike_mid_corridor_asset is not None:
                        row['Realized Var Corridor Asset (%)'] = f"{r.zero_strike_mid_corridor_asset * 100:.2f}%"
                    # ── FPF columns (only show non-empty) ──
                    if r.fpf_string_cross:
                        row['FPF Cross LV Uncapped'] = r.fpf_string_cross
                    if r.fpf_string_lsv:
                        row['FPF Cross LSV Uncapped'] = r.fpf_string_lsv
                    if r.fpf_string_lcm:
                        row['FPF Cross LCM Uncapped'] = r.fpf_string_lcm
                    if r.fpf_string_cap_lv:
                        row['FPF Cross LV Cap'] = r.fpf_string_cap_lv
                    if r.fpf_string_cap_lsv:
                        row['FPF Cross LSV Cap'] = r.fpf_string_cap_lsv
                    if r.fpf_string_cap_lcm:
                        row['FPF Cross LCM Cap'] = r.fpf_string_cap_lcm
                    if r.fpf_string_mono:
                        row['FPF Mono LV Uncapped'] = r.fpf_string_mono
                    if r.fpf_string_cap_mono:
                        row['FPF Mono LV Cap'] = r.fpf_string_cap_mono
                    if r.fpf_string_cap_lsv_mono:
                        row['FPF Mono LSV Cap'] = r.fpf_string_cap_lsv_mono
                    # ── Notional ──
                    if r.strike_variance_asset and r.strike_corridor_asset:
                        row[
                            'Sparx Notional per 1k EUR Vega (Cross)'] = f"{100 * 1000 / (2 * r.strike_variance_asset):,.2f}"
                        row[
                            'Sparx Notional per 1k EUR Vega (Mono)'] = f"{100 * 1000 / (2 * r.strike_corridor_asset):,.2f}"
                else:
                    row['Status'] = r.error or 'Failed'
            else:
                row = {
                    'Ticker': r.ticker,
                    'Currency': r.currency,
                }
                if r.success:
                    row['Maturity Date'] = cfg.last_obs_date.strftime("%d/%m/%Y")
                    row['Down Var (%)'] = f"{cfg.dvar * 100:.2f}%"
                    row['Up Var (%)'] = f"{cfg.uvar * 100:.2f}%"
                    # NOTE: strike values are stored as vol (e.g. 0.2168), not variance
                    row['Strike (%)'] = f"{r.strike_variance_asset * 100:.2f}%" if r.strike_variance_asset else 'FAILED'


                    if r.range_accrual_mono is not None:
                        row['RA Mono (%)'] = f"{r.range_accrual_mono:.2f}%"

                    if r.obs_dates_cross is not None:
                        row['Obs Dates'] = r.obs_dates_cross
                    if r.atmf_vol_variance_asset is not None:
                        row['ATMF Vol (%)'] = f"{r.atmf_vol_variance_asset * 100:.2f}%"
                        if r.strike_variance_asset:
                            row['Spread (%)'] = f"{(r.strike_variance_asset - r.atmf_vol_variance_asset) * 100:.2f}%"
                    if r.mid_variance_asset is not None:
                        row['Mid Price'] = f"{r.mid_variance_asset * 100:.2f}%"
                    if r.zero_strike_mid_variance_asset is not None:
                        row['Realized Var (%)'] = f"{r.zero_strike_mid_variance_asset * 100:.2f}%"
                    # ── FPF columns (Cross/Mono × Uncapped/Cap) ──
                    row['FPF Cross Uncapped'] = r.fpf_string_cross if r.fpf_string_cross else ''
                    row['FPF Cross LSV Uncapped'] = r.fpf_string_lsv if r.fpf_string_lsv else ''
                    row['FPF Cross LCM Uncapped'] = r.fpf_string_lcm if r.fpf_string_lcm else ''
                    row['FPF Cross Cap'] = r.fpf_string_cap_lv if r.fpf_string_cap_lv else ''
                    row['FPF Cross LSV Cap'] = r.fpf_string_cap_lsv if r.fpf_string_cap_lsv else ''
                    row['FPF Cross LCM Cap'] = r.fpf_string_cap_lcm if r.fpf_string_cap_lcm else ''
                    row['FPF Mono Uncapped'] = r.fpf_string_mono if r.fpf_string_mono else ''
                    row['FPF Mono Cap'] = r.fpf_string_cap_mono if r.fpf_string_cap_mono else ''
                    row['FPF Mono LSV Cap'] = r.fpf_string_cap_lsv_mono if r.fpf_string_cap_lsv_mono else ''
                    if r.strike_variance_asset:
                        row['Sparx Notional per 1k EUR Vega'] = f"{100 * 1000 / (2 * r.strike_variance_asset):,.2f}"
                else:
                    row['Status'] = r.error or 'Failed'
            rows.append(row)
        return pd.DataFrame(rows)

    # ─── Convenience API ─────────────────────────────────────────────────

    def solve_single(
            self,
            ticker: str,
            corridor_asset: Optional[str] = None,
            currency: Optional[str] = None,
            individual_correlation: Optional[float] = None,
    ) -> Optional[float]:
        """
        Solve strike for a single ticker via EV/RA batch (1 HTTP call).
        Quick API for scripts/notebooks. Returns: strike (float) or None.
        """
        corridor = corridor_asset or ticker
        ccy = currency or self.cache.get_currency(ticker) or 'EUR'
        df = pd.DataFrame([{
            'Tickers': ticker,
            'Corridor Condition Asset': corridor,
            'Currency': ccy,
            'Correlation': individual_correlation,
        }])
        results = self._solve_batch_ev_ra(df)
        if 0 in results and results[0][0].success:
            return results[0][0].strike
        return None

    def price_single(
            self,
            ticker: str,
            strike: float,
            corridor_asset: Optional[str] = None,
            currency: Optional[str] = None,
    ) -> Optional[float]:
        """
        Price a single ticker at a given strike. Returns mid price.
        IMPORTANT: strike is in VOL space (e.g. 0.2867 for 28.67% vol).
        It will be squared internally to variance for FPF generation.
        Do NOT pass variance (K²) from the solver — pass sqrt(K²) instead.
        """
        corridor = corridor_asset or ticker
        ccy = currency or self.cache.get_currency(ticker)
        cfg = self.config
        return price_corridor_swap(
            tickers=[ticker],
            last_obs_date=cfg.last_obs_date,
            strike_date=cfg.strike_date,
            is_capped=cfg.is_capped,
            dvar=cfg.dvar,
            uvar=cfg.uvar,
            strikes=[strike ** 2],
            currency=ccy,
            weights=[1],
            eqeq_lambda=cfg.eqeq_lambda,
            correl_floor=cfg.correl_floor,
            eqfx_shift=cfg.eqfx_shift,
            corr_asset=corridor,
            is_solving=False,
        )
    # Vol swap methods inherited from VolSwapMixin (see volswap_solver.py)


def price_corridor_swap(
        tickers: List[str],
        last_obs_date: date,
        strike_date: date,
        is_capped: bool,
        dvar: float,
        uvar: float,
        strikes: List[float],
        currency: str,
        weights: List[float],
        eqeq_lambda: float = 0.1,
        correl_floor: float = 0.0,
        eqfx_shift: float = -0.05,
        corr_asset: Optional[str] = None,
        is_solving: bool = True,
        individual_correlation: Optional[float] = None,
        model_name: Optional[str] = None,
        schedule_calendar_asset: Optional[str] = None,
) -> float:
    """
    Price a corridor variance swap with a single ticker or cross-corridor with two tickers.

    This is a standalone function that directly prices corridor swaps using the pricing portal.

    Args:
        tickers: List of ticker(s). For cross-corridor: [ref_asset, linked_asset]
        last_obs_date: Last observation date
        strike_date: Strike calculation date
        is_capped: Whether the swap is capped
        dvar: Downward variance barrier
        uvar: Upward variance barrier
        strikes: List of strike variances (K²)
        currency: Currency for pricing
        weights: List of weights for each ticker
        eqeq_lambda: Elasticity parameter for eqeq model
        correl_floor: Correlation floor
        eqfx_shift: Equity fx shift parameter
        corr_asset: Corridor asset (for cross-corridor). If None, uses first ticker
        is_solving: Whether this is for solving (True) or final pricing (False)
        individual_correlation: Override correlation (float) or output mode (True)
        model_name: Model name to use for pricing

    Returns:
        Fair value price of the corridor variance swap
    """
    corridor = corr_asset or tickers[0]

    # Determine schedule_calendar_asset
    # Caller override takes precedence; auto-detect is fallback only.
    is_cross_corridor = len(tickers) > 1 or (corr_asset and tickers[0] != corr_asset)
    if schedule_calendar_asset is not None:
        schedule_asset = schedule_calendar_asset
    else:
        schedule_asset = tickers[0] if is_cross_corridor else corridor

    # DEBUG: Log schedule asset for cross-corridor trades
    if is_cross_corridor:
        dbg.ok("CROSS-CORRIDOR", f"ticker={tickers[0]}, corr_asset={corridor}, schedule_asset={schedule_asset}")

    # Build FPF string for corridor swap
    fpf = build_corridor_fpf(
        tickers=tickers,
        last_obs_date=last_obs_date,
        strike_date=strike_date,
        strikes=strikes,
        weights=weights,
        low_barrier=dvar,
        high_barrier=uvar,
        is_capped=is_capped,
        corr_asset=corridor,
        schedule_calendar_asset=schedule_asset,
        currency=currency,
        use_parameters=False,
    )

    # Load underlyings
    all_rics = list(set(tickers + [corridor]))
    underlyings = [_load_instrument_impl(ric) for ric in all_rics]

    # Create instrument
    instrument = pricing_portal.create_fpf(
        fpf_string=fpf,
        instrument_ccy=currency,
        underlyings=underlyings,
        premium_date=datetime.datetime.now().date(),
    )

    # Model context
    is_index = corridor.startswith(".") and all(t.startswith(".") for t in tickers)
    if is_index:
        _model = "EMEA-Index-MC-LV-MultiAsset"
        model_params = {"ACEqFxShift": str(eqfx_shift)}
    else:
        _model = "EMEA-Stocks-MC-LV-MultiAsset"
        model_params = {
            'ACEqEqSpread': str(eqeq_lambda),
            "EqEqCorrFloor": str(correl_floor),
            "ACEqFxShift": str(eqfx_shift),
        }

    # Add model name if specified
    if model_name:
        _model = model_name

    _apply_special_rics_param(model_params, tickers)
    model_context = pricing_portal.create_model_context(_model, instrument_model_parameters=model_params)

    # Create metrics
    metrics = [pricing_portal.create_metric("FairValue")]

    # Add correlation metric if individual_correlation is set to output mode (True)
    if individual_correlation is True:
        _excel_epoch_c2 = date(1899, 12, 30)
        _lo_date = last_obs_date if isinstance(last_obs_date, date) else datetime.datetime.strptime(str(last_obs_date),
                                                                                                    "%Y-%m-%d").date()
        _corr_serial2 = (_lo_date - _excel_epoch_c2).days
        metrics.append(pricing_portal.create_metric("Correlation", [
            pricing_portal.create_metric_parameter("MaturityList", str(_corr_serial2)),
            pricing_portal.create_metric_parameter("MaturityType", "Absolute"),
        ]))
    elif individual_correlation is not None:
        # When using individual correlation override, still request correlation metric for output
        _excel_epoch_c2 = date(1899, 12, 30)
        _lo_date = last_obs_date if isinstance(last_obs_date, date) else datetime.datetime.strptime(str(last_obs_date),
                                                                                                    "%Y-%m-%d").date()
        _corr_serial2 = (_lo_date - _excel_epoch_c2).days
        metrics.append(pricing_portal.create_metric("Correlation", [
            pricing_portal.create_metric_parameter("MaturityList", str(_corr_serial2)),
            pricing_portal.create_metric_parameter("MaturityType", "Absolute"),
        ]))

    # Create scenario for individual correlation override
    scenario = None
    if individual_correlation is not None:
        scenario = pricing_portal.create_scenario_simple(
            mutator_name="GenericMutatorOverrideCorrelationEqEq",
            properties={"CorrelationLevel": individual_correlation}
        )

    # Price the instrument
    try:
        valuation_date = strike_date if isinstance(strike_date, datetime.datetime) else datetime.datetime.combine(
            strike_date if isinstance(strike_date, datetime.date) else datetime.datetime.strptime(str(strike_date),
                                                                                                  "%Y-%m-%d").date(),
            datetime.datetime.min.time()
        )

        result = pricing_portal.price(
            price_id="Price",
            instruments=[instrument],
            valuation_date=valuation_date,
            calculation_parameters={},
            model_context=model_context,
            overridden_snap_name=pricing_portal.get_live_snap()["name"],
            metrics=metrics,
            scenario=scenario,
        )

        # DEBUG: Log pricing response keys
        dbg.ok("PRICING-RESPONSE",
               f"result keys={list(result.keys()) if result else 'N/A'}, results keys={list(result.get('results', {}).keys()) if result else 'N/A'}")

        # Extract fair value and correlation
        if individual_correlation is not None:
            # Override mode: extract from SimpleScenarioBump
            fair_values = result["results"]["Price"]["SimpleScenarioBump"][0]["FairValue"]
            fair_value = fair_values[0].get("value", 0.0)

            # Also extract correlation if requested
            correlation = None
            if "Correlation" in result["results"]["Price"]:
                corr_values = result["results"]["Price"]["Correlation"][0].get("Correlation", [])
                if corr_values:
                    correlation = corr_values[0].get("value")

            # Log correlation mismatch for debugging
            if correlation is not None and abs(correlation - individual_correlation) > 0.001:
                dbg.warn("price_corridor_swap",
                         f"Correlation mismatch: requested={individual_correlation}, got={correlation}")

            return fair_value
        else:
            # Normal mode: extract from FairValue
            fair_values = result["results"]["Price"]["FairValue"]
            return fair_values[0].get("value", 0.0)
    except Exception as e:
        dbg.err("price_corridor_swap", f"pricing failed: {e}")
        return 0.0

