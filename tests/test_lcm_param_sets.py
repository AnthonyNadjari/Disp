"""LCM parameter sets: dataclass, bump layout, scenario shape, defaults policy.

Portal-free: a fake portal records the scenario structure. ``pricingportal``
(needed only by the LSV mutators) is stubbed in sys.modules for the LSV+LCM
layout test.
"""
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions.common.pricing_scenarios import (  # noqa: E402
    DEFAULT_LCM_PROPERTIES, LcmParamSet, lcm_bump_layout, lcm0_properties, build_unified_scenario,
)
from functions.dispersion.lcm_sets import resolve_lcm_sets, lcm_column_suffix, lcm0_lambda_for  # noqa: E402


class FakePortal:
    """Records scenario structure as plain tuples."""

    def create_scenario_mutator_properties(self, props=None, properties=None):
        return dict(props if props is not None else (properties or {}))

    def create_scenario_mutator(self, name, mutator_properties, mutator_properties_asset_overrides):
        return (name, mutator_properties)

    def create_scenario_bump(self, name, mutators):
        return (name, list(mutators))

    def create_scenario_axis(self, bumps):
        return list(bumps)

    def create_scenario(self, axes):
        return list(axes)

    # LSV side (only structure matters here)
    def create_scenario_mutator_asset_overrides(self, asset_tuple, properties_overrides):
        return ("override", asset_tuple, properties_overrides)

    def create_scenario_asset_tuple(self, assets):
        return tuple(assets)

    def create_scenario_asset(self, asset_type, asset_id):
        return (asset_type, asset_id)


@pytest.fixture
def pp():
    return FakePortal()


def _bump_names(scenario):
    return [b[0] for b in scenario[0]]


def _lcm_props_of(scenario, bump):
    for name, mutators in scenario[0]:
        if name == bump:
            return [m[1] for m in mutators if m[0] == "GenericMutatorOverrideLCMWithRealisedReference"][0]
    raise KeyError(bump)


# ── LcmParamSet ──────────────────────────────────────────────────────────────

def test_bump_names_legacy_and_named():
    assert LcmParamSet().bump_lcm == "LCM" and LcmParamSet().bump_lcm0 == "LCM0"
    s = LcmParamSet(name="A")
    assert (s.bump_lcm, s.bump_lcm0) == ("LCM_A", "LCM0_A")


def test_name_validation():
    with pytest.raises(ValueError):
        LcmParamSet(name="bad name!")
    LcmParamSet(name="ok_1.2-x")


def test_properties_roundtrip_and_shapes():
    s = LcmParamSet(name="A", lambda_pricing=0.4, lambda_atm=0.4, lambda_from_rho0=0.1,
                    call_skew=-0.5, put_skew=-0.9, aggregator_type="Basket")
    p = s.to_properties()
    assert p["CallSkew"] == [-0.5] and p["PutSkew"] == [-0.9] and p["LambdaAtm"] == [0.4]
    assert p["LambdaPricing"] == 0.4 and p["LambdaFromRho0"] == 0.1
    assert LcmParamSet.from_properties(p, name="A") == s
    # scalars accepted on the way in
    back = LcmParamSet.from_properties({"LambdaAtm": 0.3, "CallSkew": -0.1})
    assert back.lambda_atm == 0.3 and back.call_skew == -0.1 and back.lambda_pricing is None


def test_to_properties_requires_filled():
    with pytest.raises(ValueError):
        LcmParamSet(name="A").to_properties()
    filled = LcmParamSet(name="A").filled(lambda_pricing=0.4, lambda_atm=0.4, lambda_from_rho0=0.1,
                                          call_skew=-0.5, put_skew=-0.9, aggregator_type="Basket")
    assert filled.missing() == []


def test_coerce_field_dict_and_portal_dict():
    a = LcmParamSet.coerce({"name": "A", "lambda_pricing": 0.5})
    assert a.name == "A" and a.lambda_pricing == 0.5
    b = LcmParamSet.coerce({"LambdaPricing": 0.53, "LambdaAtm": [0.23]}, name="B")
    assert b.name == "B" and b.lambda_atm == 0.23
    with pytest.raises(TypeError):
        LcmParamSet.coerce(42)


def test_lcm0_properties_zero_skews_only():
    p = LcmParamSet(name="A", lambda_pricing=0.4, lambda_atm=0.4, lambda_from_rho0=0.1,
                    call_skew=-0.5, put_skew=-0.9, aggregator_type="Basket").to_properties()
    p0 = lcm0_properties(p)
    assert p0["CallSkew"] == [0.0] and p0["PutSkew"] == [0.0]
    assert {k: v for k, v in p0.items() if k not in ("CallSkew", "PutSkew")} == \
           {k: v for k, v in p.items() if k not in ("CallSkew", "PutSkew")}


# ── layout ───────────────────────────────────────────────────────────────────

def test_layout_shares_lcm0_between_same_lambdas():
    sets = resolve_lcm_sets(0.4, [{"name": "A"},
                                  {"name": "B", "call_skew": -0.2},
                                  {"name": "C", "lambda_pricing": 0.55}])
    lay = lcm_bump_layout(sets)
    assert [(s.name, b, b0) for s, b, b0 in lay] == [
        ("A", "LCM_A", "LCM0_A"), ("B", "LCM_B", "LCM0_A"), ("C", "LCM_C", "LCM0_C")]


def test_layout_pinned_lcm0_lambda_shares_one_lcm0_across_lambdas():
    """LCM0 pinned to the EqEq lambda: sets with different lambdas still share
    ONE LCM0 (only rho0 can split it)."""
    sets = resolve_lcm_sets(0.4, [{"name": "A"},
                                  {"name": "B", "lambda_pricing": 0.55, "lambda_atm": 0.3},
                                  {"name": "C", "lambda_from_rho0": 0.2}])
    lay = lcm_bump_layout(sets, lcm0_lambda=0.4)
    assert [(s.name, b, b0) for s, b, b0 in lay] == [
        ("A", "LCM_A", "LCM0_A"), ("B", "LCM_B", "LCM0_A"), ("C", "LCM_C", "LCM0_C")]


def test_lcm0_properties_pinned_lambda():
    p = LcmParamSet(name="B", lambda_pricing=0.55, lambda_atm=0.3, lambda_from_rho0=0.1,
                    call_skew=-0.5, put_skew=-0.9, aggregator_type="Basket").to_properties()
    p0 = lcm0_properties(p, lcm0_lambda=0.4)
    assert p0["LambdaPricing"] == 0.4 and p0["LambdaAtm"] == [0.4]
    assert p0["CallSkew"] == [0.0] and p0["PutSkew"] == [0.0] and p0["LambdaFromRho0"] == 0.1
    assert p["LambdaPricing"] == 0.55          # the LCM bag itself is untouched


def test_scenario_pinned_lcm0(pp):
    sets = resolve_lcm_sets(0.4, [{"name": "A"}, {"name": "B", "lambda_pricing": 0.55, "lambda_atm": 0.3}])
    sc = build_unified_scenario(pp, use_lsv=False, lcm_sets=sets, lcm0_lambda=0.4)
    assert _bump_names(sc) == ["LV", "LCM0_A", "LCM_A", "LCM_B"]       # one LCM0 for both
    assert _lcm_props_of(sc, "LCM0_A")["LambdaPricing"] == 0.4
    assert _lcm_props_of(sc, "LCM0_A")["LambdaAtm"] == [0.4]
    assert _lcm_props_of(sc, "LCM_B")["LambdaPricing"] == 0.55
    assert _lcm_props_of(sc, "LCM_B")["LambdaAtm"] == [0.3]


def test_layout_rejects_duplicate_names():
    with pytest.raises(ValueError):
        lcm_bump_layout([LcmParamSet(name="A"), LcmParamSet(name="A")])


# ── scenario shape ───────────────────────────────────────────────────────────

def test_scenario_lcm_only_multi_set(pp):
    sets = resolve_lcm_sets(0.4, [{"name": "A"}, {"name": "B", "put_skew": -0.3}, {"name": "C", "lambda_atm": 0.3}])
    sc = build_unified_scenario(pp, use_lsv=False, lcm_sets=sets)
    assert _bump_names(sc) == ["LV", "LCM0_A", "LCM0_C", "LCM_A", "LCM_B", "LCM_C"]
    assert {len(b[1]) for b in sc[0]} == {2}
    assert _lcm_props_of(sc, "LCM0_A")["CallSkew"] == [0.0]
    assert _lcm_props_of(sc, "LCM_B")["PutSkew"] == [-0.3]
    assert _lcm_props_of(sc, "LCM_C")["LambdaAtm"] == [0.3]
    assert _lcm_props_of(sc, "LCM_A")["LambdaPricing"] == 0.4


def test_scenario_legacy_kwargs_unchanged(pp):
    assert _bump_names(build_unified_scenario(pp, use_lsv=False, use_lcm=True)) == ["LV", "LCM"]
    assert _bump_names(build_unified_scenario(pp, use_lsv=False, use_lcm=True, include_lcm0=True)) == ["LV", "LCM0", "LCM"]
    assert build_unified_scenario(pp, use_lsv=False, use_lcm=False) is None


def test_scenario_single_unnamed_set_keeps_legacy_bump_names(pp):
    sets = resolve_lcm_sets(0.4, None, {"enabled": True, "lcm_properties": DEFAULT_LCM_PROPERTIES})
    assert sets[0].name == ""
    assert _bump_names(build_unified_scenario(pp, use_lsv=False, lcm_sets=sets)) == ["LV", "LCM0", "LCM"]


def test_scenario_lsv_plus_lcm_multi_set(pp, monkeypatch):
    import pandas as pd
    stub = types.ModuleType("pricingportal")
    stub.NovaAssetType = types.SimpleNamespace(MONIKER="MONIKER")
    monkeypatch.setitem(sys.modules, "pricingportal", stub)
    sets = resolve_lcm_sets(0.4, [{"name": "A"}, {"name": "B", "lambda_pricing": 0.5}])
    lsv_df = pd.DataFrame({"VolOfVar": [0.7], "Eq/VolCorrel": [-0.7], "MeanReversion": [2.0]}, index=["X.PA"])
    sc = build_unified_scenario(pp, use_lsv=True, underlying_rics=["X.PA"], lsv_params=lsv_df, lcm_sets=sets)
    assert _bump_names(sc) == ["LV", "LSV0", "LSV", "LCM0_A", "LCM0_B", "LCM_A", "LCM_B"]
    assert {len(b[1]) for b in sc[0]} == {3}


# ── defaults policy ──────────────────────────────────────────────────────────

def test_resolve_defaults_lambdas_to_eqeq():
    logs = []
    sets = resolve_lcm_sets(0.4, [{"name": "A"}, {"name": "B", "lambda_pricing": 0.55}], log=logs.append)
    a, b = sets
    assert (a.lambda_pricing, a.lambda_atm) == (0.4, 0.4)
    assert (b.lambda_pricing, b.lambda_atm) == (0.55, 0.4)          # explicit wins, missing defaults
    assert a.lambda_from_rho0 == DEFAULT_LCM_PROPERTIES["LambdaFromRho0"]
    assert a.call_skew == DEFAULT_LCM_PROPERTIES["CallSkew"][0]
    assert any("EqEq lambda 0.4" in m for m in logs)


def test_resolve_falls_back_to_desk_defaults_when_eqeq_unusable():
    logs = []
    (a,) = resolve_lcm_sets(0.0, [{"name": "A"}], log=logs.append)
    assert a.lambda_pricing == DEFAULT_LCM_PROPERTIES["LambdaPricing"]
    assert a.lambda_atm == DEFAULT_LCM_PROPERTIES["LambdaAtm"][0]
    assert any("not usable" in m for m in logs)


def test_resolve_off_and_legacy():
    assert resolve_lcm_sets(0.4, None, None) == []
    assert resolve_lcm_sets(0.4, None, {"enabled": False, "lcm_properties": {}}) == []
    (leg,) = resolve_lcm_sets(0.4, None, {"enabled": True, "lcm_properties": {"LambdaPricing": 0.53}})
    assert leg.name == "" and leg.lambda_pricing == 0.53 and leg.lambda_atm == 0.4


def test_resolve_naming_rules():
    (one,) = resolve_lcm_sets(0.4, [{"lambda_pricing": 0.5}])
    assert one.name == ""                                            # single unnamed → legacy columns
    two = resolve_lcm_sets(0.4, [{}, {}])
    assert [s.name for s in two] == ["1", "2"]                       # auto names
    with pytest.raises(ValueError):
        resolve_lcm_sets(0.4, [{"name": "A"}, {"name": "A"}])
    with pytest.raises(ValueError):
        resolve_lcm_sets(0.4, [LcmParamSet(name=""), LcmParamSet(name="B")])


def test_lcm0_lambda_for():
    assert lcm0_lambda_for(0.4) == 0.4
    assert lcm0_lambda_for(0.0) is None and lcm0_lambda_for(None) is None and lcm0_lambda_for(-1) is None


def test_resolve_log_mentions_pinned_lcm0():
    logs = []
    resolve_lcm_sets(0.4, [{"name": "B", "lambda_pricing": 0.55}], log=logs.append)
    assert "LCM0 = LambdaPricing=LambdaAtm=0.4" in logs[-1]


def test_column_suffix():
    assert lcm_column_suffix("") == ""
    assert lcm_column_suffix("A") == " [A]"
