"""
Pricing scenario utilities — shared building blocks for scenario-based pricing.

Used by dispersion, vol feedback, correlation, or any product that needs
scenario axes (LSV, LCM, spot bumps, etc.) attached to portal price() calls.
"""
from __future__ import annotations

from typing import List, Dict, Optional, Callable, Any
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# Default parameters — single source of truth
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_LCM_PROPERTIES: Dict[str, Any] = {
    "AggregatorType": "Basket",
    "CallSkew": [-0.5],
    "PutSkew": [-0.9],
    "LambdaAtm": [0.2368839],
    "LambdaFromRho0": 0.1,
    "LambdaPricing": 0.53,
}

DEFAULT_LSV_MODEL_PROPERTIES: Dict[str, Any] = {
    "VarianceType": "ExponentialOU",
    "NumSpotSteps": 200.0,
    "NumVolatilitySteps": 300.0,
    "CorrelationBeta": 0.99,
}

DEFAULT_LSV_ZERO_PROPERTIES: Dict[str, float] = {
    "VolOfVar": 0.0001,
    "Correlation": 0.0001,
    "ReversionRate": 0.0001,
}

LSV_DEFAULTS_INDEX: Dict[str, float] = {"VolOfVar": 1.25, "Correlation": -0.7, "ReversionRate": 2.0}
LSV_DEFAULTS_STOCK: Dict[str, float] = {"VolOfVar": 0.7, "Correlation": -0.7, "ReversionRate": 2.0}

_INDEX_RICS = {".STOXX50E", ".SPX", ".FTSE", ".N225", ".HSI", ".FCHI", ".GDAXI", ".AEX", ".IBEX", ".SSMI"}


# ══════════════════════════════════════════════════════════════════════════════
# LCM — Local Correlation Model
# ══════════════════════════════════════════════════════════════════════════════

def build_lcm_bumps(
    pricing_portal,
    lcm_properties: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build LCM mutators for inclusion in a scenario axis.

    Returns a dict with:
        "lcm_mutator": The OverrideLCMWithRealisedReference mutator.
        "null_mutator": A GenericMutatorNull for padding other bumps.

    Usage in batch scenario assembly:
        parts = build_lcm_bumps(pp, lcm_properties)
        # Add parts["lcm_mutator"] to the "LCM" bump
        # Add parts["null_mutator"] to pad other bumps in the LCM slot
    """
    props = lcm_properties if lcm_properties is not None else DEFAULT_LCM_PROPERTIES

    lcm_mutator = pricing_portal.create_scenario_mutator(
        name="GenericMutatorOverrideLCMWithRealisedReference",
        mutator_properties=pricing_portal.create_scenario_mutator_properties(props),
        mutator_properties_asset_overrides=[],
    )
    null_mutator = pricing_portal.create_scenario_mutator(
        name="GenericMutatorNull",
        mutator_properties=pricing_portal.create_scenario_mutator_properties({}),
        mutator_properties_asset_overrides=[],
    )
    return {"lcm_mutator": lcm_mutator, "null_mutator": null_mutator}


def build_lcm_scenario(
    pricing_portal,
    lcm_properties: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Build a complete [LV, LCM] scenario for standalone pricing calls.

    Use this when pricing a single instrument outside the batch path
    (e.g., correlation toolbox, ad-hoc pricing).

    Response structure:
        results[price_id]['LV'][instrument_idx]['FairValue'][0]['value']
        results[price_id]['LCM'][instrument_idx]['FairValue'][0]['value']
    """
    parts = build_lcm_bumps(pricing_portal, lcm_properties)

    scenario = pricing_portal.create_scenario(axes=[
        pricing_portal.create_scenario_axis(bumps=[
            pricing_portal.create_scenario_bump(name="LV", mutators=[parts["null_mutator"]]),
            pricing_portal.create_scenario_bump(name="LCM", mutators=[parts["lcm_mutator"]]),
        ])
    ])
    return scenario


# ══════════════════════════════════════════════════════════════════════════════
# LSV — Local Stochastic Volatility
# ══════════════════════════════════════════════════════════════════════════════

def build_lsv_bumps(
    pricing_portal,
    underlying_rics: List[str],
    lsv_params: pd.DataFrame,
    correl_bump: float = 0,
    correl_bump_style: str = "Relative",
) -> Dict[str, Any]:
    """
    Build LSV mutators for inclusion in a scenario axis.

    Returns a dict with:
        "lv_mutators":   [null_mutator, correl_noop]        — for the LV bump
        "lsv0_mutators": [null_mutator, lsv0_mutator]       — for the LSV0 bump
        "lsv_mutators":  [lsv_full_mutator, correl_mutator] — for the LSV bump
        "null_mutator":  A GenericMutatorNull for padding

    Usage in batch scenario assembly:
        parts = build_lsv_bumps(pp, rics, lsv_df, correl_bump=0.05)
        # Build bumps using parts["lv_mutators"], parts["lsv0_mutators"], etc.
    """
    from pricingportal import NovaAssetType

    def _make_asset_overrides(rics, properties_fn):
        return [
            pricing_portal.create_scenario_mutator_asset_overrides(
                asset_tuple=pricing_portal.create_scenario_asset_tuple(
                    assets=[pricing_portal.create_scenario_asset(
                        asset_type=NovaAssetType.MONIKER,
                        asset_id=f"instrument.reuters/{ric}"
                    )]
                ),
                properties_overrides=pricing_portal.create_scenario_mutator_properties(
                    properties=properties_fn(ric)
                )
            )
            for ric in rics
        ]

    def _lsv_props_for_ric(ric):
        if ric in lsv_params.index:
            row = lsv_params.loc[ric]
            return {
                "VolOfVar": float(row['VolOfVar']),
                "Correlation": float(row['Eq/VolCorrel']),
                "ReversionRate": float(row['MeanReversion']),
            }
        if ric in _INDEX_RICS:
            return LSV_DEFAULTS_INDEX
        return LSV_DEFAULTS_STOCK

    # LV bump mutators
    lv_null_mutator = pricing_portal.create_scenario_mutator(
        name="GenericMutatorNull",
        mutator_properties=pricing_portal.create_scenario_mutator_properties({}),
        mutator_properties_asset_overrides=[]
    )
    lv_correl_noop = pricing_portal.create_scenario_mutator(
        name="GenericMutatorBumpCorrelationEqEq",
        mutator_properties=pricing_portal.create_scenario_mutator_properties(
            {"BumpSize": 0.0, "Style": correl_bump_style}
        ),
        mutator_properties_asset_overrides=[]
    )

    # LSV0 bump mutators
    lsv0_null_mutator = pricing_portal.create_scenario_mutator(
        name="GenericMutatorNull",
        mutator_properties=pricing_portal.create_scenario_mutator_properties({}),
        mutator_properties_asset_overrides=[]
    )
    lsv0_mutator = pricing_portal.create_scenario_mutator(
        name="GenericMutatorOverrideLSVParameters",
        mutator_properties=pricing_portal.create_scenario_mutator_properties(DEFAULT_LSV_MODEL_PROPERTIES),
        mutator_properties_asset_overrides=_make_asset_overrides(
            underlying_rics, lambda _: DEFAULT_LSV_ZERO_PROPERTIES
        )
    )

    # LSV bump mutators
    lsv_full_mutator = pricing_portal.create_scenario_mutator(
        name="GenericMutatorOverrideLSVParameters",
        mutator_properties=pricing_portal.create_scenario_mutator_properties(DEFAULT_LSV_MODEL_PROPERTIES),
        mutator_properties_asset_overrides=_make_asset_overrides(
            underlying_rics, _lsv_props_for_ric
        )
    )
    lsv_correl_mutator = pricing_portal.create_scenario_mutator(
        name="GenericMutatorBumpCorrelationEqEq",
        mutator_properties=pricing_portal.create_scenario_mutator_properties(
            {"BumpSize": correl_bump, "Style": correl_bump_style}
        ),
        mutator_properties_asset_overrides=[]
    )

    # Null for padding
    pad_null = pricing_portal.create_scenario_mutator(
        name="GenericMutatorNull",
        mutator_properties=pricing_portal.create_scenario_mutator_properties({}),
        mutator_properties_asset_overrides=[]
    )

    return {
        "lv_mutators": [lv_null_mutator, lv_correl_noop],
        "lsv0_mutators": [lsv0_null_mutator, lsv0_mutator],
        "lsv_mutators": [lsv_full_mutator, lsv_correl_mutator],
        "null_mutator": pad_null,
    }


def build_lsv_scenario(
    pricing_portal,
    underlying_rics: List[str],
    lsv_params: pd.DataFrame,
    correl_bump: float = 0,
    correl_bump_style: str = "Relative",
) -> Any:
    """
    Build a complete [LV, LSV0, LSV] scenario for standalone pricing calls.

    The scenario can be attached to any pricing_portal.price() call.
    The portal will run MC under all bumps and return results keyed by bump name.

    Args:
        pricing_portal: Active PricingPortal instance.
        underlying_rics: List of RIC strings (e.g. [".STOXX50E", "TTEF.PA"]).
        lsv_params: DataFrame indexed by RIC with columns:
                    'VolOfVar', 'Eq/VolCorrel', 'MeanReversion'.
        correl_bump: Correlation bump size for LSV bump (default 0 = no bump).
        correl_bump_style: "Relative" or "Absolute".

    Returns:
        Scenario object ready to pass to pricing_portal.price(scenario=...).

    Response structure (per price_id key):
        results[price_id]['LV'][instrument_idx]['FairValue'][0]['value']
        results[price_id]['LSV0'][instrument_idx]['FairValue'][0]['value']
        results[price_id]['LSV'][instrument_idx]['FairValue'][0]['value']
    """
    parts = build_lsv_bumps(pricing_portal, underlying_rics, lsv_params, correl_bump, correl_bump_style)

    scenario = pricing_portal.create_scenario(axes=[
        pricing_portal.create_scenario_axis(bumps=[
            pricing_portal.create_scenario_bump(name="LV", mutators=parts["lv_mutators"]),
            pricing_portal.create_scenario_bump(name="LSV0", mutators=parts["lsv0_mutators"]),
            pricing_portal.create_scenario_bump(name="LSV", mutators=parts["lsv_mutators"]),
        ])
    ])
    return scenario


# ══════════════════════════════════════════════════════════════════════════════
# Unified scenario builder — combines LSV + LCM with proper padding
# ══════════════════════════════════════════════════════════════════════════════

def build_unified_scenario(
    pricing_portal,
    use_lsv: bool = False,
    use_lcm: bool = False,
    underlying_rics: Optional[List[str]] = None,
    lsv_params: Optional[pd.DataFrame] = None,
    correl_bump: float = 0,
    correl_bump_style: str = "Relative",
    lcm_properties: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """
    Build a unified scenario axis with proper mutator padding.

    Depending on flags:
        Neither:       returns None (no scenario)
        LCM only:      [LV, LCM]           — 2 bumps × 2 mutators
        LSV only:      [LV, LSV0, LSV]     — 3 bumps × 2 mutators
        LSV + LCM:     [LV, LSV0, LSV, LCM] — 4 bumps × 3 mutators

    All bumps within an axis have equal mutator count (padded with GenericMutatorNull).

    Returns:
        Scenario object or None if neither LSV nor LCM is enabled.
    """
    if not use_lsv and not use_lcm:
        return None

    # ── LCM only ──
    if use_lcm and not use_lsv:
        lcm_parts = build_lcm_bumps(pricing_portal, lcm_properties)
        scenario = pricing_portal.create_scenario(axes=[
            pricing_portal.create_scenario_axis(bumps=[
                pricing_portal.create_scenario_bump(name="LV", mutators=[lcm_parts["null_mutator"], lcm_parts["null_mutator"]]),
                pricing_portal.create_scenario_bump(name="LCM", mutators=[lcm_parts["null_mutator"], lcm_parts["lcm_mutator"]]),
            ])
        ])
        return scenario

    # ── LSV only ──
    if use_lsv and not use_lcm:
        return build_lsv_scenario(pricing_portal, underlying_rics, lsv_params, correl_bump, correl_bump_style)

    # ── LSV + LCM ──
    lsv_parts = build_lsv_bumps(pricing_portal, underlying_rics, lsv_params, correl_bump, correl_bump_style)
    lcm_parts = build_lcm_bumps(pricing_portal, lcm_properties)

    # All bumps need 3 mutators: [LSV_slot, Correl_slot, LCM_slot]
    pad = lcm_parts["null_mutator"]

    scenario = pricing_portal.create_scenario(axes=[
        pricing_portal.create_scenario_axis(bumps=[
            pricing_portal.create_scenario_bump(name="LV", mutators=lsv_parts["lv_mutators"] + [pad]),
            pricing_portal.create_scenario_bump(name="LSV0", mutators=lsv_parts["lsv0_mutators"] + [pad]),
            pricing_portal.create_scenario_bump(name="LSV", mutators=lsv_parts["lsv_mutators"] + [pad]),
            pricing_portal.create_scenario_bump(name="LCM", mutators=[pad, lsv_parts["null_mutator"], lcm_parts["lcm_mutator"]]),
        ])
    ])
    return scenario


# ══════════════════════════════════════════════════════════════════════════════
# Extraction utilities
# ══════════════════════════════════════════════════════════════════════════════

def extract_scenario_metric(
    raw_results: dict,
    price_id: str,
    bump_name: str,
    instrument_idx: int,
    metric_name: str = "FairValue",
) -> Optional[float]:
    """
    Extract a metric value from a scenario-based pricing response.

    Handles both response formats:
      Format 1 (array): results[price_id][bump_name][instrument_idx][metric_name][0]['value']
      Format 2 (keyed): results['Price_N'][bump_name][0][metric_name][0]['value']

    Args:
        raw_results: The 'results' dict from pricing_portal.price() response.
        price_id: The price_id used in the call (e.g. "Price").
        bump_name: Named bump to extract (e.g. "LV" or "LSV").
        instrument_idx: Global instrument index in the batch.
        metric_name: Metric to extract (default "FairValue").

    Returns:
        Float value or None if extraction fails.
    """
    try:
        # Format 1: results[price_id][bump_name][instrument_idx][metric_name][0]['value']
        if price_id in raw_results:
            entry = raw_results[price_id]
            if isinstance(entry, dict) and bump_name in entry:
                bump_data = entry[bump_name]
                if isinstance(bump_data, list) and instrument_idx < len(bump_data):
                    metric_list = bump_data[instrument_idx].get(metric_name, [])
                    if metric_list and isinstance(metric_list, list):
                        val = metric_list[0]
                        return val.get("value") if isinstance(val, dict) else val

        # Format 2: keyed per instrument — results['Price_N'][bump_name][0][metric_name][0]['value']
        key = price_id if instrument_idx == 0 else f"{price_id}_{instrument_idx}"
        if key in raw_results:
            entry = raw_results[key]
            if isinstance(entry, dict) and bump_name in entry:
                bump_data = entry[bump_name]
                if isinstance(bump_data, list) and len(bump_data) > 0:
                    metric_list = bump_data[0].get(metric_name, [])
                    if metric_list and isinstance(metric_list, list):
                        val = metric_list[0]
                        return val.get("value") if isinstance(val, dict) else val

    except (KeyError, IndexError, TypeError, AttributeError):
        pass

    return None
