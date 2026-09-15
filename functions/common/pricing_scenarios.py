"""
Pricing scenario utilities — shared building blocks for scenario-based pricing.

Used by dispersion, vol feedback, correlation, or any product that needs
scenario axes (LSV, LCM, spot bumps, etc.) attached to portal price() calls.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace as _dc_replace
from typing import List, Dict, Optional, Callable, Any, Sequence, Tuple, ClassVar
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

@dataclass(frozen=True)
class LcmParamSet:
    """One LCM parameter set = one (LCM, LCM0) bump pair in a scenario axis.

    ``None`` fields mean "not specified" — the caller fills them with its own
    defaults (``filled``) before ``to_properties`` is called. Product-specific
    policies (e.g. dispersion: lambdas default to the EqEq lambda) live in the
    product, not here.

    Bump names derive from ``name``: ``""`` → ``LCM`` / ``LCM0`` (legacy single
    set); ``"A"`` → ``LCM_A`` / ``LCM0_A``. Names are restricted to
    ``[A-Za-z0-9_.-]`` so they survive as portal result keys.
    """
    name: str = ""
    lambda_pricing: Optional[float] = None
    lambda_atm: Optional[float] = None
    lambda_from_rho0: Optional[float] = None
    call_skew: Optional[float] = None
    put_skew: Optional[float] = None
    aggregator_type: Optional[str] = None

    _NAME_RE: ClassVar = re.compile(r"^[A-Za-z0-9_.\-]*$")
    _FIELDS: ClassVar = ("lambda_pricing", "lambda_atm", "lambda_from_rho0",
                         "call_skew", "put_skew", "aggregator_type")

    def __post_init__(self):
        if not isinstance(self.name, str) or not self._NAME_RE.match(self.name):
            raise ValueError(f"LcmParamSet.name {self.name!r}: use only letters, digits, '_', '.', '-'")

    # ---- bump naming ----
    @property
    def bump_lcm(self) -> str:
        return "LCM" if not self.name else f"LCM_{self.name}"

    @property
    def bump_lcm0(self) -> str:
        return "LCM0" if not self.name else f"LCM0_{self.name}"

    def lcm0_key(self, lcm0_lambda: Optional[float] = None) -> Tuple:
        """What the LCM0 bump depends on (skews are zeroed) — sets sharing this
        key share one LCM0 bump. With ``lcm0_lambda`` given, LCM0's
        LambdaPricing and LambdaAtm are both pinned to it, so only ρ0 and the
        aggregator can still split sets."""
        if lcm0_lambda is not None:
            return (float(lcm0_lambda), float(lcm0_lambda), self.lambda_from_rho0, self.aggregator_type)
        return (self.lambda_pricing, self.lambda_atm, self.lambda_from_rho0, self.aggregator_type)

    # ---- defaults / conversion ----
    def missing(self) -> List[str]:
        return [f for f in self._FIELDS if getattr(self, f) is None]

    def filled(self, **defaults) -> "LcmParamSet":
        """Copy with every ``None`` field replaced by ``defaults[field]`` when given."""
        upd = {f: defaults[f] for f in self._FIELDS if getattr(self, f) is None and f in defaults}
        return _dc_replace(self, **upd) if upd else self

    def to_properties(self) -> Dict[str, Any]:
        """Portal property bag (list form where the portal expects lists)."""
        miss = self.missing()
        if miss:
            raise ValueError(f"LcmParamSet {self.name!r}: unfilled fields {miss} — call filled(...) first")
        return {
            "AggregatorType": self.aggregator_type,
            "CallSkew": [float(self.call_skew)],
            "PutSkew": [float(self.put_skew)],
            "LambdaAtm": [float(self.lambda_atm)],
            "LambdaFromRho0": float(self.lambda_from_rho0),
            "LambdaPricing": float(self.lambda_pricing),
        }

    @classmethod
    def from_properties(cls, props: Dict[str, Any], name: str = "") -> "LcmParamSet":
        """Inverse of ``to_properties`` (accepts scalars or one-element lists)."""
        def _s(v):
            if isinstance(v, (list, tuple)):
                return float(v[0]) if v else None
            return None if v is None else (v if isinstance(v, str) else float(v))
        p = props or {}
        return cls(
            name=name,
            lambda_pricing=_s(p.get("LambdaPricing")),
            lambda_atm=_s(p.get("LambdaAtm")),
            lambda_from_rho0=_s(p.get("LambdaFromRho0")),
            call_skew=_s(p.get("CallSkew")),
            put_skew=_s(p.get("PutSkew")),
            aggregator_type=p.get("AggregatorType"),
        )

    @classmethod
    def coerce(cls, obj, name: str = "") -> "LcmParamSet":
        """``LcmParamSet`` | field-keyed dict | portal-keyed dict → ``LcmParamSet``."""
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            if any(k in obj for k in ("LambdaPricing", "LambdaAtm", "CallSkew", "PutSkew")):
                return cls.from_properties(obj, name=obj.get("name", name))
            return cls(**{**{"name": name}, **obj})
        raise TypeError(f"cannot coerce {type(obj).__name__} to LcmParamSet")


def lcm_bump_layout(lcm_sets: Sequence[LcmParamSet],
                    lcm0_lambda: Optional[float] = None) -> List[Tuple[LcmParamSet, str, str]]:
    """``[(set, lcm_bump_name, lcm0_bump_name), ...]`` in input order.

    LCM0 bumps are shared between sets with the same ``lcm0_key(lcm0_lambda)``
    (the first set owns the bump name). With ``lcm0_lambda`` every LCM0 uses
    LambdaPricing = LambdaAtm = lcm0_lambda, so all sets with the same ρ0 share
    ONE LCM0 bump. Raises on duplicate set names."""
    seen_names = set()
    lcm0_owner: Dict[Tuple, str] = {}
    out = []
    for s in lcm_sets:
        if s.name in seen_names:
            raise ValueError(f"duplicate LcmParamSet name {s.name!r}")
        seen_names.add(s.name)
        key = s.lcm0_key(lcm0_lambda)
        lcm0 = lcm0_owner.setdefault(key, s.bump_lcm0)
        out.append((s, s.bump_lcm, lcm0))
    return out


def build_lcm_mutator(pricing_portal, props: Dict[str, Any]):
    return pricing_portal.create_scenario_mutator(
        name="GenericMutatorOverrideLCMWithRealisedReference",
        mutator_properties=pricing_portal.create_scenario_mutator_properties(props),
        mutator_properties_asset_overrides=[],
    )


def build_lcm_bumps(
    pricing_portal,
    lcm_properties: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build LCM mutators for inclusion in a scenario axis.

    Returns a dict with:
        "lcm_mutator":  The OverrideLCMWithRealisedReference mutator.
        "lcm0_mutator": Same mutator, same lambdas, CallSkew = PutSkew = 0 —
                        the flat-correlation control (the LCM analogue of LSV0).
                        LCM − LCM0 isolates the correlation-skew contribution and
                        cancels the level gap + MC noise between LV and LCM.
        "null_mutator": A GenericMutatorNull for padding other bumps.

    Usage in batch scenario assembly:
        parts = build_lcm_bumps(pp, lcm_properties)
        # Add parts["lcm_mutator"] to the "LCM" bump
        # Add parts["lcm0_mutator"] to the "LCM0" bump (optional)
        # Add parts["null_mutator"] to pad other bumps in the LCM slot
    """
    props = lcm_properties if lcm_properties is not None else DEFAULT_LCM_PROPERTIES
    props0 = lcm0_properties(props)

    lcm_mutator = pricing_portal.create_scenario_mutator(
        name="GenericMutatorOverrideLCMWithRealisedReference",
        mutator_properties=pricing_portal.create_scenario_mutator_properties(props),
        mutator_properties_asset_overrides=[],
    )
    lcm0_mutator = pricing_portal.create_scenario_mutator(
        name="GenericMutatorOverrideLCMWithRealisedReference",
        mutator_properties=pricing_portal.create_scenario_mutator_properties(props0),
        mutator_properties_asset_overrides=[],
    )
    null_mutator = pricing_portal.create_scenario_mutator(
        name="GenericMutatorNull",
        mutator_properties=pricing_portal.create_scenario_mutator_properties({}),
        mutator_properties_asset_overrides=[],
    )
    return {"lcm_mutator": lcm_mutator, "lcm0_mutator": lcm0_mutator, "null_mutator": null_mutator}


def lcm0_properties(lcm_properties: Dict[str, Any], lcm0_lambda: Optional[float] = None) -> Dict[str, Any]:
    """LCM0 = the LCM property bag with CallSkew and PutSkew zeroed and, when
    ``lcm0_lambda`` is given, LambdaPricing = LambdaAtm = lcm0_lambda (the
    flat-correlation control at the run's reference lambda). List-valued keys
    stay lists (the portal expects ``[x]``)."""
    props0 = dict(lcm_properties)
    for k in ("CallSkew", "PutSkew"):
        v = lcm_properties.get(k)
        props0[k] = [0.0] if isinstance(v, (list, tuple)) else 0.0
    if lcm0_lambda is not None:
        props0["LambdaPricing"] = float(lcm0_lambda)
        props0["LambdaAtm"] = [float(lcm0_lambda)]
    return props0


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
    include_lcm0: bool = False,
    lcm_sets: Optional[Sequence[LcmParamSet]] = None,
    lcm0_lambda: Optional[float] = None,
) -> Optional[Any]:
    """
    Build a unified scenario axis with proper mutator padding.

    LCM can be given two ways:
      * ``lcm_sets`` — N fully-filled ``LcmParamSet``; each contributes an
        ``LCM_<name>`` bump plus an ``LCM0_<name>`` bump (skews = 0), LCM0
        bumps shared between sets with identical ``lcm0_key`` (``lcm_bump_layout``).
        With ``lcm0_lambda`` every LCM0 is pinned to LambdaPricing = LambdaAtm
        = lcm0_lambda (the run's reference lambda) whatever the set's lambdas,
        so sets with the same ρ0 share one LCM0. ``use_lcm`` is implied. This
        is the multi-set path: one call, LV priced once, every set priced on
        the same market and MC paths.
      * ``lcm_properties`` (+ ``include_lcm0``) — legacy single set, bump names
        ``LCM`` / ``LCM0``. Unchanged behaviour for existing consumers.

    Layouts (mutator counts are equal within the axis, padded with
    GenericMutatorNull):
        Neither:       None
        LSV only:      [LV, LSV0, LSV]                        2 mutators
        LCM only:      [LV, (LCM0…), LCM…]                    2 mutators
        LSV + LCM:     [LV, LSV0, LSV, (LCM0…), LCM…]         3 mutators
    """
    # ── Normalise the LCM input to a layout ──
    layout: List[Tuple[LcmParamSet, str, str]] = []
    if lcm_sets:
        layout = lcm_bump_layout(list(lcm_sets), lcm0_lambda)
        use_lcm = True
    elif use_lcm:
        legacy = LcmParamSet.from_properties(lcm_properties or DEFAULT_LCM_PROPERTIES, name="")
        layout = [(legacy, legacy.bump_lcm, legacy.bump_lcm0 if include_lcm0 else None)]
        lcm0_lambda = None      # legacy path: LCM0 keeps the set's own lambdas

    if not use_lsv and not use_lcm:
        return None
    if use_lsv and not use_lcm:
        return build_lsv_scenario(pricing_portal, underlying_rics, lsv_params, correl_bump, correl_bump_style)

    # ── LCM mutators: one per set, one per distinct LCM0 ──
    lcm_mutators: Dict[str, Any] = {}    # bump name -> mutator
    for s, b_lcm, b_lcm0 in layout:
        props = s.to_properties()
        lcm_mutators[b_lcm] = build_lcm_mutator(pricing_portal, props)
        if b_lcm0 is not None and b_lcm0 not in lcm_mutators:
            lcm_mutators[b_lcm0] = build_lcm_mutator(pricing_portal, lcm0_properties(props, lcm0_lambda))
    lcm0_names = [b for _, _, b in layout if b is not None]
    lcm0_names = list(dict.fromkeys(lcm0_names))          # unique, input order
    lcm_names = [b for _, b, _ in layout]

    null = pricing_portal.create_scenario_mutator(
        name="GenericMutatorNull",
        mutator_properties=pricing_portal.create_scenario_mutator_properties({}),
        mutator_properties_asset_overrides=[],
    )

    # ── LCM only: [LV, LCM0…, LCM…], 2 mutators each ──
    if not use_lsv:
        bumps = [pricing_portal.create_scenario_bump(name="LV", mutators=[null, null])]
        for b in lcm0_names + lcm_names:
            bumps.append(pricing_portal.create_scenario_bump(name=b, mutators=[null, lcm_mutators[b]]))
        return pricing_portal.create_scenario(axes=[pricing_portal.create_scenario_axis(bumps=bumps)])

    # ── LSV + LCM: [LV, LSV0, LSV, LCM0…, LCM…], 3 mutators each ──
    lsv_parts = build_lsv_bumps(pricing_portal, underlying_rics, lsv_params, correl_bump, correl_bump_style)
    pad = null
    bumps = [
        pricing_portal.create_scenario_bump(name="LV", mutators=lsv_parts["lv_mutators"] + [pad]),
        pricing_portal.create_scenario_bump(name="LSV0", mutators=lsv_parts["lsv0_mutators"] + [pad]),
        pricing_portal.create_scenario_bump(name="LSV", mutators=lsv_parts["lsv_mutators"] + [pad]),
    ]
    for b in lcm0_names + lcm_names:
        bumps.append(pricing_portal.create_scenario_bump(
            name=b, mutators=[pad, lsv_parts["null_mutator"], lcm_mutators[b]]))
    return pricing_portal.create_scenario(axes=[pricing_portal.create_scenario_axis(bumps=bumps)])


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
