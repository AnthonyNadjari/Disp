"""Engine-side LCM multi-set plumbing, without a portal.

``functions.dispersion._pricing`` needs the proprietary FPF stack at import
time only through ``_volswap`` (``fpf_builder_utils``); a sys.modules stub makes
the module importable so the dataclasses, the defaults resolution from a
``PricingConfig`` and the results DataFrame can be exercised offline.
"""
import datetime
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def pricing():
    if "fpf_builder_utils" not in sys.modules:
        pkg = types.ModuleType("fpf_builder_utils")
        cal = types.ModuleType("fpf_builder_utils.calendar")

        def _unavailable(*_a, **_k):
            raise RuntimeError("fpf_builder_utils stub (offline tests)")

        cal.get_trading_calendar = cal.offset_date = cal.create_schedule = _unavailable
        pkg.calendar = cal
        sys.modules["fpf_builder_utils"] = pkg
        sys.modules["fpf_builder_utils.calendar"] = cal
    import functions.dispersion._pricing as p
    return p


def _cfg(pricing, **kw):
    base = dict(strike_date=datetime.date(2026, 9, 15), last_obs_date=datetime.date(2027, 9, 15),
                is_cross_corridor=True, eqeq_lambda=0.4)
    base.update(kw)
    return pricing.PricingConfig(**base)


def _leg(pricing, s, **kw):
    return pricing.LcmLegResult(set_name=s.name, bump_lcm=s.bump_lcm, bump_lcm0=s.bump_lcm0,
                                properties=s.to_properties(), **kw)


def test_config_resolves_sets_with_eqeq_default(pricing):
    cfg = _cfg(pricing, lcm_sets=[{"name": "A"}, {"name": "B", "lambda_pricing": 0.55}])
    sets = pricing._resolve_lcm_sets_for(cfg)
    assert [s.name for s in sets] == ["A", "B"]
    assert (sets[0].lambda_pricing, sets[0].lambda_atm) == (0.4, 0.4)
    assert (sets[1].lambda_pricing, sets[1].lambda_atm) == (0.55, 0.4)


def test_config_legacy_lcm_params_is_one_unnamed_set(pricing):
    cfg = _cfg(pricing, lcm_params={"enabled": True, "lcm_properties": {"LambdaPricing": 0.53}})
    (s,) = pricing._resolve_lcm_sets_for(cfg)
    assert s.name == "" and s.lambda_pricing == 0.53 and s.lambda_atm == 0.4
    assert pricing._resolve_lcm_sets_for(_cfg(pricing)) == []


def test_sync_legacy_mirrors_first_set(pricing):
    cfg = _cfg(pricing, lcm_sets=[{"name": "A"}, {"name": "B"}])
    a, b = pricing._resolve_lcm_sets_for(cfg)
    tr = pricing.TickerResult(ticker="X.PA", corridor_asset=".STOXX50E", success=True)
    tr.lcm["A"] = _leg(pricing, a, ev_cross=-0.031, ev_cross0=-0.0305, strike=0.201, strike_raw=0.19,
                       strike_cap_proxy=0.199, fpf_string="FPF-A")
    tr.lcm["B"] = _leg(pricing, b, ev_cross=-0.02, ev_cross0=-0.02, strike=0.3, strike_raw=0.3)
    tr.sync_legacy_lcm()
    assert tr.strike_variance_asset_lcm == 0.201
    assert tr.strike_cap_priced_lcm == 0.199          # proxy until the capped batch fills it
    assert tr.fpf_string_lcm == "FPF-A"
    assert tr.lcm["A"].impact == pytest.approx(-0.0005)
    assert tr.lcm["A"].strike_raw == 0.19             # raw strike lives on the leg only


def test_sync_legacy_noop_without_lcm(pricing):
    tr = pricing.TickerResult(ticker="X.PA", corridor_asset=".STOXX50E", success=True)
    tr.sync_legacy_lcm()
    assert tr.first_lcm() is None and tr.strike_variance_asset_lcm is None and tr.fpf_string_cap_lcm is None


def test_leg_impact_none_when_one_side_missing(pricing):
    leg = pricing.LcmLegResult(set_name="A", bump_lcm="LCM_A", bump_lcm0="LCM0_A", ev_cross=-0.03)
    assert leg.impact is None and leg.mid_impact is None
    leg.mid_lcm, leg.mid_lcm0 = 0.012, 0.011
    assert leg.mid_impact == pytest.approx(0.001)


def test_results_df_cross_capped_two_sets(pricing):
    cfg = _cfg(pricing, is_capped=True, lcm_sets=[{"name": "A"}, {"name": "B", "lambda_pricing": 0.55}])
    a, b = pricing._resolve_lcm_sets_for(cfg)
    tr = pricing.TickerResult(ticker="X.PA", corridor_asset=".STOXX50E", success=True, currency="EUR",
                              strike_variance_asset=0.20, strike_corridor_asset=0.22)
    for s in (a, b):
        tr.lcm[s.name] = _leg(pricing, s, ev_cross=-0.031, ev_cross0=-0.0305, strike=0.201, strike_raw=0.19,
                              strike_cap_priced=0.198, strike_cap_priced_raw=0.187,
                              ev_cap_cross=-0.029, ev_cap_cross0=-0.0285,
                              fpf_string=f"FPF-{s.name}", fpf_string_cap=f"FPFCAP-{s.name}")
    tr.sync_legacy_lcm()
    df = pricing.PricingEngine(cfg)._build_results_df([tr])
    for col in ["Strike Cross Corr LCM [A] (%)", "Strike Cross Corr LCM [B] (%)",
                "EV Cross LCM [A] (%)", "EV Cross LCM0 [A] (%)", "LCM Impact Cross [A] (%)",
                "Strike Cross Corr Cap Priced LCM [B] (%)",
                "EV Cap Cross LCM [A] (%)", "EV Cap Cross LCM0 [A] (%)",
                "LCM Params [A]", "LCM Params [B]",
                "FPF Cross LCM Uncapped [A]", "FPF Cross LCM Cap [B]"]:
        assert col in df.columns, col
    row = df.iloc[0]
    assert row["LCM Impact Cross [A] (%)"] == "-0.05%"
    assert row["Strike Cross Corr LCM [A] (%)"] == "20.10%"
    assert "lamP=0.55" in row["LCM Params [B]"] and "lamATM=0.4" in row["LCM Params [B]"]
    assert "lamP=0.4 " in row["LCM Params [A]"]
    assert row["FPF Cross LCM Cap [B]"] == "FPFCAP-B"
    cols = list(df.columns)
    # strike block keeps the LV / LSV / LCM [A] / LCM [B] order …
    i = cols.index
    assert i("Strike Cross Corr LV (%)") < i("Strike Cross Corr LCM [A] (%)") < i("Strike Cross Corr LCM [B] (%)") \
        < i("Strike Cross Corr Cap Priced LV (%)") < i("Strike Cross Corr Cap Priced LCM [A] (%)") \
        < i("Strike Cross Corr Cap Priced LCM [B] (%)") < i("Strike Mono Corr LV (%)")
    # … and the non-strike columns are grouped per set (all [A] before any [B])
    a_cols = [c for c in cols if "[A]" in c and "Strike" not in c and not c.startswith("FPF")]
    b_cols = [c for c in cols if "[B]" in c and "Strike" not in c and not c.startswith("FPF")]
    assert max(i(c) for c in a_cols) < min(i(c) for c in b_cols)


def test_results_df_uncapped_suffix_after_label(pricing):
    cfg = _cfg(pricing, is_capped=False, lcm_sets=[{"name": "A"}])
    (s,) = pricing._resolve_lcm_sets_for(cfg)
    tr = pricing.TickerResult(ticker="X.PA", corridor_asset=".STOXX50E", success=True, currency="EUR",
                              strike_variance_asset=0.2, strike_corridor_asset=0.22)
    tr.lcm["A"] = _leg(pricing, s, ev_cross=-0.03, ev_cross0=-0.03, strike=0.2, strike_raw=0.2)
    df = pricing.PricingEngine(cfg)._build_results_df([tr])
    assert "Strike Cross Corr LCM (Uncapped) [A] (%)" in df.columns      # same shape as "… LV (Uncapped) (%)"
    assert "Strike Cross Corr LV (Uncapped) (%)" in df.columns


def test_results_df_without_lcm(pricing):
    cfg = _cfg(pricing)
    tr = pricing.TickerResult(ticker="X.PA", corridor_asset=".STOXX50E", success=True, currency="EUR",
                              strike_variance_asset=0.2, strike_corridor_asset=0.22)
    df = pricing.PricingEngine(cfg)._build_results_df([tr])
    assert not [c for c in df.columns if "LCM" in c]


def test_results_df_mono_branch(pricing):
    cfg = _cfg(pricing, is_cross_corridor=False)
    tr = pricing.TickerResult(ticker=".STOXX50E", corridor_asset=".STOXX50E", success=True, currency="EUR",
                              strike_variance_asset=0.2, range_accrual_mono=0.85, discount_factor=0.97)
    df = pricing.PricingEngine(cfg)._build_results_df([tr])
    assert df["Strike LV Uncapped (%)"].iloc[0] == "20.00%"
    assert df["RA (%)"].iloc[0] == "0.85%"
    assert "Ticker" in df.columns and "Index Ticker" not in df.columns


def test_emit_never_raises(pricing):
    def bad(_): raise RuntimeError("ui gone")
    pricing._emit(bad, {"status": "phase"})          # swallowed
    pricing._emit(None, {"status": "phase"})         # no callback


def test_validate_lcm_cross_corridor(pricing):
    pricing._validate_lcm_cross_corridor(["X.PA"], [".STOXX50E"])
    with pytest.raises(ValueError, match="cross-corridor"):
        pricing._validate_lcm_cross_corridor(["X.PA", ".SPX"], [".STOXX50E", ".SPX"])


def test_results_df_single_unnamed_set_keeps_legacy_columns(pricing):
    cfg = _cfg(pricing, is_capped=False, lcm_params={"enabled": True, "lcm_properties": {"LambdaPricing": 0.53}})
    (s,) = pricing._resolve_lcm_sets_for(cfg)
    tr = pricing.TickerResult(ticker="X.PA", corridor_asset=".STOXX50E", success=True, currency="EUR",
                              strike_variance_asset=0.2, strike_corridor_asset=0.22)
    tr.lcm[""] = _leg(pricing, s, ev_cross=-0.03, ev_cross0=-0.03, strike=0.2, strike_raw=0.2)
    df = pricing.PricingEngine(cfg)._build_results_df([tr])
    assert "Strike Cross Corr LCM (Uncapped) (%)" in df.columns
    assert "LCM Params" in df.columns
    assert not [c for c in df.columns if "[" in c]


def test_results_df_price_mode_impact_vs_lcm0(pricing):
    cfg = _cfg(pricing, is_solve=False, lcm_sets=[{"name": "A"}])
    (s,) = pricing._resolve_lcm_sets_for(cfg)
    tr = pricing.TickerResult(ticker="X.PA", corridor_asset=".STOXX50E", success=True, currency="EUR",
                              strike_variance_asset=0.2, strike_corridor_asset=0.22, mid_variance_asset=0.01)
    tr.lcm["A"] = _leg(pricing, s, mid_lcm=0.012, mid_lcm0=0.011)
    tr.sync_legacy_lcm()
    df = pricing.PricingEngine(cfg)._build_results_df([tr])
    assert df["FV Variance Asset LCM0 [A] (%)"].iloc[0] == "1.10%"
    assert df["LCM Impact Variance Asset [A] (%)"].iloc[0] == "0.10%"
    assert "LCM Params [A]" in df.columns
    assert tr.lcm["A"].mid_lcm == 0.012


def test_results_df_missing_lcm0_leaves_strike_empty_keeps_raw(pricing):
    cfg = _cfg(pricing, is_capped=False, lcm_sets=[{"name": "A"}])
    (s,) = pricing._resolve_lcm_sets_for(cfg)
    tr = pricing.TickerResult(ticker="X.PA", corridor_asset=".STOXX50E", success=True, currency="EUR",
                              strike_variance_asset=0.2, strike_corridor_asset=0.22)
    tr.lcm["A"] = _leg(pricing, s, ev_cross=-0.03, ev_cross0=None, strike=None, strike_raw=0.19)
    df = pricing.PricingEngine(cfg)._build_results_df([tr])
    assert "Strike Cross Corr LCM [A] (Uncapped) (%)" not in df.columns
    assert not [c for c in df.columns if "Raw" in c]          # raw strike not displayed
    assert tr.lcm["A"].strike_raw == 0.19                      # but kept on the object
    assert "LCM Impact Cross [A] (%)" not in df.columns


def test_params_label(pricing):
    p = {"LambdaPricing": 0.55, "LambdaAtm": [0.3], "LambdaFromRho0": 0.1, "CallSkew": [-0.5], "PutSkew": [-0.9]}
    assert pricing._lcm_params_label(p) == "lamP=0.55 lamATM=0.3 rho0=0.1 CS=-0.5 PS=-0.9"
    p0 = {"LambdaPricing": 0.4, "LambdaAtm": [0.4], "CallSkew": [0.0], "PutSkew": [0.0]}
    assert pricing._lcm_params_label(p, p0).endswith("| LCM0: lamP=0.4 lamATM=0.4 CS=PS=0")


def test_engine_layout_pins_lcm0_to_eqeq_lambda(pricing):
    cfg = _cfg(pricing, lcm_sets=[{"name": "A"}, {"name": "B", "lambda_pricing": 0.55, "lambda_atm": 0.3}])
    sets = pricing._resolve_lcm_sets_for(cfg)
    lay = pricing._lcm_layout_for(cfg, sets)
    assert [b0 for _, _, b0 in lay] == ["LCM0_A", "LCM0_A"]          # one shared LCM0
    assert pricing._lcm0_lambda_for(cfg) == 0.4
    # results carry both property bags
    tr = pricing.TickerResult(ticker="X.PA", corridor_asset=".STOXX50E", success=True, currency="EUR",
                              strike_variance_asset=0.2, strike_corridor_asset=0.22)
    from functions.common.pricing_scenarios import lcm0_properties
    s = sets[1]
    tr.lcm["B"] = pricing.LcmLegResult(set_name="B", bump_lcm="LCM_B", bump_lcm0="LCM0_A",
                                       properties=s.to_properties(),
                                       properties0=lcm0_properties(s.to_properties(), 0.4),
                                       ev_cross=-0.03, ev_cross0=-0.029, strike=0.2, strike_raw=0.19)
    df = pricing.PricingEngine(cfg)._build_results_df([tr])
    assert df["LCM Params [B]"].iloc[0] == "lamP=0.55 lamATM=0.3 rho0=0.1 CS=-0.5 PS=-0.9 | LCM0: lamP=0.4 lamATM=0.4 CS=PS=0"


def test_engine_lcm0_falls_back_when_eqeq_unusable(pricing):
    cfg = _cfg(pricing, eqeq_lambda=0.0, lcm_sets=[{"name": "A"}, {"name": "B", "lambda_pricing": 0.6}])
    sets = pricing._resolve_lcm_sets_for(cfg)
    assert pricing._lcm0_lambda_for(cfg) is None
    assert [b0 for _, _, b0 in pricing._lcm_layout_for(cfg, sets)] == ["LCM0_A", "LCM0_B"]   # own lambdas → split
