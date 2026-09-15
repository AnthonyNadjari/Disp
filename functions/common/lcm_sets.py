"""
LCM parameter sets — resolution policy (portal-free, unit-testable, product-agnostic).

Lives in functions.common so any product (dispersion, correlation toolbox, vol
feedback, ...) can turn user input into filled ``LcmParamSet`` with the same
rules. ``functions.dispersion.lcm_sets`` re-exports this module.

Turns whatever the caller gave (nothing, the legacy ``lcm_params`` dict, or a
list of ``LcmParamSet`` / dicts) into a list of fully-filled ``LcmParamSet``
ready for ``functions.common.pricing_scenarios.build_unified_scenario``.

Policy
------
* ``lambda_pricing`` and ``lambda_atm`` default to the **reference lambda** of
  the run (dispersion: the EqEq lambda (the same number that is ``ACEqEqSpread`` in the model context). An
  explicit value on the set always wins — they are inputs, not overrides.
* ``lambda_from_rho0``, ``call_skew``, ``put_skew``, ``aggregator_type``
  default to ``DEFAULT_LCM_PROPERTIES`` (the CSV-style desk defaults).
* When the EqEq lambda is not usable (``<= 0``, e.g. Individual Correlations
  mode passes 0.0) the lambdas fall back to ``DEFAULT_LCM_PROPERTIES`` and the
  fact is reported through ``log``.
* **LCM0** (the control every set's impact is measured against) is ALWAYS
  LambdaPricing = LambdaAtm = the EqEq lambda, skews 0 — whatever lambdas the
  set itself uses (``lcm0_lambda_for``). One LCM0 bump therefore serves every
  set with the same ρ0. When the EqEq lambda is not usable, LCM0 falls back to
  the set's own lambdas (skews 0).
* Set names must be unique; ``""`` is allowed only for a single set (legacy
  bump names ``LCM`` / ``LCM0``, un-suffixed result columns).
"""
from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence, Union

from functions.common.pricing_scenarios import DEFAULT_LCM_PROPERTIES, LcmParamSet

LcmSetInput = Union[LcmParamSet, dict]


def _desk_defaults() -> dict:
    d = DEFAULT_LCM_PROPERTIES
    return dict(
        lambda_pricing=float(d["LambdaPricing"]),
        lambda_atm=float(d["LambdaAtm"][0]),
        lambda_from_rho0=float(d["LambdaFromRho0"]),
        call_skew=float(d["CallSkew"][0]),
        put_skew=float(d["PutSkew"][0]),
        aggregator_type=d["AggregatorType"],
    )


def resolve_lcm_sets(
    eqeq_lambda: Optional[float],
    lcm_sets: Optional[Sequence[LcmSetInput]] = None,
    legacy_lcm_params: Optional[dict] = None,
    log: Callable[[str], None] = lambda _m: None,
) -> List[LcmParamSet]:
    """Fully-filled, validated LCM sets for one pricing run. ``[]`` = LCM off.

    ``lcm_sets`` wins over ``legacy_lcm_params`` (``{'enabled': bool,
    'lcm_properties': dict}``). Dicts in ``lcm_sets`` may be field-keyed
    (``{'name': 'A', 'lambda_pricing': 0.4}``) or portal-keyed
    (``{'LambdaPricing': 0.4, ...}``).
    """
    raw: List[LcmParamSet] = []
    if lcm_sets:
        n = len(lcm_sets)
        for i, item in enumerate(lcm_sets):
            # unnamed dicts: "" when it is the only set (legacy columns), else "1", "2", …
            auto = "" if n == 1 else str(i + 1)
            raw.append(LcmParamSet.coerce(item, name=auto))
    elif legacy_lcm_params and legacy_lcm_params.get("enabled"):
        raw.append(LcmParamSet.from_properties(legacy_lcm_params.get("lcm_properties") or {}, name=""))
    if not raw:
        return []

    # ── names ──
    names = [s.name for s in raw]
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"LCM sets: duplicate names {dup}")
    if "" in names and len(raw) > 1:
        raise ValueError("LCM sets: every set needs a name when more than one set is given")

    # ── defaults ──
    defaults = _desk_defaults()
    lam = float(eqeq_lambda or 0.0)
    if lam > 0.0:
        defaults["lambda_pricing"] = lam
        defaults["lambda_atm"] = lam
        lam_src = f"EqEq lambda {lam:g}"
    else:
        lam_src = (f"desk defaults (EqEq lambda={lam:g} not usable): "
                   f"LambdaPricing={defaults['lambda_pricing']:g}, LambdaAtm={defaults['lambda_atm']:g}")

    out: List[LcmParamSet] = []
    for s in raw:
        missing = s.missing()
        f = s.filled(**defaults)
        out.append(f)
        tag = f"[LCM set {f.name or '(single)'}]"
        lam_note = ""
        if "lambda_pricing" in missing or "lambda_atm" in missing:
            which = [k for k in ("lambda_pricing", "lambda_atm") if k in missing]
            lam_note = f" ({', '.join(which)} <- {lam_src})"
        lcm0_lam = lcm0_lambda_for(eqeq_lambda)
        lcm0_note = (f"LCM0 = LambdaPricing=LambdaAtm={lcm0_lam:g} (EqEq lambda), skews 0"
                     if lcm0_lam is not None else "LCM0 = same lambdas, skews 0")
        log(f"{tag} LambdaPricing={f.lambda_pricing:g} LambdaAtm={f.lambda_atm:g} "
            f"LambdaFromRho0={f.lambda_from_rho0:g} CallSkew={f.call_skew:g} PutSkew={f.put_skew:g}"
            f"{lam_note}; {lcm0_note}")
    return out


def lcm0_lambda_for(eqeq_lambda: Optional[float]) -> Optional[float]:
    """The lambda LCM0 is pinned to: the EqEq lambda when usable, else ``None``
    (LCM0 then keeps each set's own lambdas)."""
    lam = float(eqeq_lambda or 0.0)
    return lam if lam > 0.0 else None


def lcm_column_suffix(set_name: str) -> str:
    """Result-column suffix for a set: ``''`` for the legacy unnamed set,
    ``' [A]'`` otherwise."""
    return f" [{set_name}]" if set_name else ""
