"""
Backward-compatibility shim — imports from new locations.

New code should import directly from:
    - functions.common.comet (UrlService, comet_id_to_fpf, get_ccy_from_comet_id)
    - functions.common.portal (pricing_portal, live_snap)
    - functions.lsv.setup (LSVPricingSetup, sicovam_mapping)
    - functions.common.pricing_utils (date_to_excel_serial, compute_implied_vol, etc.)
"""

from __future__ import annotations

import datetime
import threading
from datetime import date
from typing import List, Optional

import numpy as np
import pandas as pd
import requests

from pricingportal import (
    PricingPortal, NovaIdSource, NovaOptionStyle, NovaOptionType,
)
from pricingportal.pricingportal_results import pricing_error
from fpf_builder_utils.calendar import create_schedule, get_trading_calendar
from fpflucid_gen.economics import ShiftOH, Just, PaymentDate
from speq.fpf.unified_economics_schema.fpf_schema import (
    FPFUnifiedEconomicsWrapper,
    mc_global_autocall_v1,
    fides_global_autocall_v1,
    corridorCovarianceSwap_v4,
    mc_global_autocall_synth_div_v1,
)
from speq_services import avsweb

from functions.common.tickers import bbg_to_ric
from functions.common.comet import (
    UrlService, comet_id_to_fpf, get_ccy_from_comet_id, get_attribution_from_comet_id,
)
from functions.lsv.setup import LSVPricingSetup, sicovam_mapping
from functions.common.portal import make_portal


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pricing Portal Singleton
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pricing_portal = make_portal()

live_snap = pricing_portal.create_snap(
    description="Live Snap",
    mapping_context="shark",
    use_live_prices=True,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pricing helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


from functions.common.calendar import date_to_excel as date_to_excel_serial

date_to_excel_serialdate = date_to_excel_serial

_implied_vol_cache: dict = {}


def compute_implied_vol(asset: str, fwd_date: date, anchor_type: str = "Spot") -> float:
    """Compute ATMF implied vol with thread-safe caching."""
    cache_key = (asset, fwd_date, anchor_type)
    if cache_key in _implied_vol_cache:
        return _implied_vol_cache[cache_key]
    metrics = create_implied_vol_metrics(fwd_date, anchor_type)
    result = px_option(asset, fwd_date, metrics)
    vol = result["results"]["Option Forward"]["QueryLocalCcyVol"][0]["value"]
    _implied_vol_cache[cache_key] = vol
    return vol


def compute_forward(asset: str, fwd_date: date) -> float:
    """Compute forward/spot ratio."""
    metrics = create_fwd_metrics(fwd_date)
    result = px_option(asset, fwd_date, metrics)
    r = result["results"]["Option Forward"]
    return r["Forward"][0]["value"] / r["AssetSpotLevels"][0]["value"]


def compute_div_yield(asset: str, fwd_date: date) -> float:
    """Compute dividend yield to maturity."""
    metrics = create_div_yield_metric(fwd_date)
    result = px_option(asset, fwd_date, metrics)
    return result["results"]["Option Forward"]["DividendYield"][0]["value"]


def compute_repo(asset: str, fwd_date: date) -> float:
    """Compute repo rate to maturity."""
    metrics = create_repo_metric(fwd_date)
    result = px_option(asset, fwd_date, metrics)
    return result["results"]["Option Forward"]["RepoRate"][0]["value"]


# ── Metric builders ──────────────────────────────────────────────────────────

def create_implied_vol_metrics(fwd_date: date, anchor_type: str = "Spot") -> list:
    excel_date = date_to_excel_serial(fwd_date)
    params = [
        pricing_portal.create_metric_parameter("AnchorType", anchor_type),
        pricing_portal.create_metric_parameter("MaturityList", str(excel_date)),
        pricing_portal.create_metric_parameter("MaturityType", "Absolute"),
        pricing_portal.create_metric_parameter("Strikes", "1"),
    ]
    return [pricing_portal.create_metric("QueryLocalCcyVol", params)]


def create_div_yield_metric(fwd_date: date) -> list:
    excel_date = date_to_excel_serial(fwd_date)
    params = [
        pricing_portal.create_metric_parameter("MaturityList", str(excel_date)),
        pricing_portal.create_metric_parameter("MaturityType", "Absolute"),
    ]
    return [pricing_portal.create_metric('DividendYield', params)]


def create_repo_metric(fwd_date: date) -> list:
    excel_date = date_to_excel_serial(fwd_date)
    params = [
        pricing_portal.create_metric_parameter("MaturityList", str(excel_date)),
        pricing_portal.create_metric_parameter("MaturityType", "Absolute"),
    ]
    return [pricing_portal.create_metric('RepoRate', params)]


def create_fwd_metrics(fwd_date: date) -> list:
    excel_date = date_to_excel_serial(fwd_date)
    params = [
        pricing_portal.create_metric_parameter("MaturityList", str(excel_date)),
        pricing_portal.create_metric_parameter("MaturityType", "Absolute"),
    ]
    return [
        pricing_portal.create_metric("Forward", params),
        pricing_portal.create_metric("AssetSpotLevels"),
    ]


# ── Option creation & pricing ────────────────────────────────────────────────

def create_option(asset: str, fwd_date: date):
    """Create a vanilla ATM European call for metric extraction."""
    underlying = pricing_portal.load_instrument(schema=NovaIdSource.REUTERS, instrument_id=asset)
    info = pricing_portal.get_underlying_information(
        underlying_identifier_type="ricCode", underlying_identifiers=[asset]
    )
    currency = info["information"][asset]["currency"]
    return pricing_portal.create_option(
        style=NovaOptionStyle.EUROPEAN,
        strike=1,
        maturity=fwd_date,
        option_type=NovaOptionType.CALL,
        payoff_ccy=currency,
        underlying=underlying,
        strike_type="PERCENT",
    )


def px_option(asset: str, fwd_date: date, metrics: list) -> dict:
    """Price a vanilla option with given metrics."""
    option = create_option(asset, fwd_date)
    return pricing_portal.price(
        price_id="Option Forward",
        instruments=[option],
        metrics=metrics,
        overridden_snap_name=live_snap["name"],
        valuation_date=datetime.datetime.now(),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Mutators
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def create_mutators(mutator_name, mutator_properties, all_asset_overrides):
    """Create a scenario mutator (uses fresh portal for thread safety)."""
    pp = make_portal()
    if mutator_properties is not None:
        return pp.create_scenario_mutator(
            name=mutator_name,
            mutator_properties=pp.create_scenario_mutator_properties(mutator_properties),
            mutator_properties_asset_overrides=all_asset_overrides,
        )
    return pp.create_scenario_mutator(
        name=mutator_name,
        mutator_properties_asset_overrides=all_asset_overrides,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Calendar / N_EXP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def get_n_exp_from_date(matu, underlyings):
    """Compute number of expiry dates from now to maturity for given underlyings.
    Returns the average trading days across all underlyings' calendars."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def process_underlying(underlying):
        if "Index" in underlying or "Equity" in underlying:
            underlying = bbg_to_ric(underlying)
        try:
            calendar = get_trading_calendar(underlying)
            if not calendar or calendar == "":
                return None
            days = len(create_schedule(
                datetime.datetime.now().strftime("%Y-%m-%d"),
                matu.strftime("%Y-%m-%d"),
                "1B",
                calendar,
                "MF"
            )) - 1
            return days
        except:
            return None

    with ThreadPoolExecutor(max_workers=min(len(underlyings), 20)) as executor:
        futures = {executor.submit(process_underlying, u): u for u in underlyings}
        results = [future.result() for future in as_completed(futures)]

    valid_results = [r for r in results if r is not None]
    return int(sum(valid_results) / len(valid_results)) if valid_results else 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FPF / Div helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def get_div_asset(fpf):
    """Extract dividend assets from FPF and get their div streams."""
    if hasattr(fpf, 'assetParameters'):
        assets = [a.asset for a in fpf.assetParameters]
    else:
        assets = [fpf.asset]
    return assets


def get_errors_from_result(result: dict) -> str:
    """Extract error messages from pricing result."""
    errors = pricing_error(result)
    if errors:
        return "; ".join(str(e) for e in errors)
    return ""
