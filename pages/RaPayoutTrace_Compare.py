"""RA Comparison page — legacy range-accrual instrument vs PayoutTrace RA.

Ultra-simple: click "Run comparison", get RA_old vs RA_new side by side.

Run from the repo root:
    streamlit run pages/RaPayoutTrace_Compare.py
"""

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from functions.dispersion import solve, DispersionConfig
from functions.dispersion.models import ProductType
from functions.dispersion._pricing import (
    build_corridor_fpf,
    get_pricing_portal,
    get_live_snap,
    _n_corridor_obs,
    _unfunded_zcb,
    _ra_from_payout_trace,
)

MODEL_CONTEXT_NAME = "EMEA-Stocks-MC-LV-MultiAsset"


def _build_instrument(var_asset, corr_asset, is_capped, strike_date, last_obs_date,
                      barrier_dn, barrier_up, currency):
    fpf = build_corridor_fpf(
        tickers=[var_asset],
        last_obs_date=last_obs_date,
        strike_date=strike_date,
        strikes=[0.000001],  # placeholder — RA does not depend on the strike
        weights=[1.0],
        low_barrier=barrier_dn,
        high_barrier=barrier_up,
        is_capped=is_capped,
        corr_asset=corr_asset,
        currency=currency,
        schedule_calendar_asset=var_asset,
        use_parameters=False,
    )
    from speq.fpf.unified_economics_schema.fpf_schema import FPFUnifiedEconomicsWrapper
    from pricingportal import NovaIdSource
    wrapper = FPFUnifiedEconomicsWrapper.from_data(fpf)
    portal = get_pricing_portal()
    underlyings = [
        portal.load_instrument(schema=NovaIdSource.REUTERS, instrument_id=u)
        for u in sorted({var_asset, corr_asset})
    ]
    nova = portal.create_fpf(
        fpf_string=wrapper.to_fpf_string(),
        instrument_ccy=currency,
        underlyings=underlyings,
        premium_date=strike_date,
    )
    return nova, len(wrapper.observationDates) - 1  # engine convention: today excluded


def _solve_ra_old(df, cross, barrier_dn, barrier_up, strike_date, last_obs_date):
    cfg = DispersionConfig(
        product_type=ProductType.VAR_SWAP_CORRIDOR,
        cross_corridor=cross,
        barrier_up=barrier_up, barrier_down=barrier_dn,
        local_cap=2.5, is_capped=True,
    )
    res = solve(df=df, config=cfg,
                last_obs_date=last_obs_date, strike_date=strike_date,
                eqeq_lambda=0.10, eqfx_shift=-0.05, vol_mode="ATMF")
    if not res.success:
        raise RuntimeError(f"solve failed for {list(df['Variance Asset'])}")
    return res


st.set_page_config(page_title="RA Comparison", page_icon="⚖️", layout="wide")
st.title("⚖️ RA Comparison — legacy instrument vs PayoutTrace")
st.caption("RA_old (range-accrual instrument) vs RA_new = ZCB unfunded × E[nCorridorObs] / n_total")

with st.sidebar:
    st.header("Inputs")
    mono_txt = st.text_input("Mono tickers (comma)", value="ISP.MI, TTEF.PA")
    cross_txt = st.text_input("Cross pairs stock/index (comma)", value="ISP.MI/.STOXX50E, ASML.AS/.STOXX50E")
    currency = st.selectbox("Currency", ["EUR", "USD", "GBP", "CHF"], index=0)
    barrier_dn = st.number_input("Lower barrier", value=0.70, step=0.05)
    barrier_up = st.number_input("Upper barrier", value=1.30, step=0.05)
    strike_date = st.date_input("Strike date", value=date.today())
    last_obs_date = st.date_input("Last obs date", value=date.today() + timedelta(days=365))

run = st.button("🚀 Run comparison", type="primary", use_container_width=True)

if run:
    mono_tickers = [t.strip() for t in mono_txt.split(",") if t.strip()]
    cross_pairs = []
    for p in cross_txt.split(","):
        p = p.strip()
        if "/" in p:
            s, i = p.split("/", 1)
            cross_pairs.append((s.strip(), i.strip()))

    portal = get_pricing_portal()
    snap = get_live_snap()
    model_context = portal.create_model_context(
        name=MODEL_CONTEXT_NAME, instrument_model_parameters={})

    rows = []
    with st.spinner("Solving + pricing metric..."):
        # Mono
        if mono_tickers:
            mono_df = pd.DataFrame({
                "Variance Asset": mono_tickers,
                "Corridor Condition Asset": mono_tickers,
                "Currency": [currency] * len(mono_tickers),
            })
            res = _solve_ra_old(mono_df, False, barrier_dn, barrier_up, strike_date, last_obs_date)
            for tr in res.ticker_results:
                nova, n_total = _build_instrument(tr.ticker, tr.ticker, True,
                                                  strike_date, last_obs_date,
                                                  barrier_dn, barrier_up, currency)
                n_obs = _n_corridor_obs([nova], datetime.now(), model_context, snap["name"])[0]
                zcb = _unfunded_zcb(currency, strike_date, last_obs_date)
                rows.append({"case": "mono", "ticker": tr.ticker,
                             "RA_old": tr.range_accrual,
                             "RA_new": _ra_from_payout_trace(n_obs, n_total, zcb) if n_obs is not None else None,
                             "n_total_obs": n_total})

        # Cross
        if cross_pairs:
            cross_df = pd.DataFrame({
                "Variance Asset": [i for _, i in cross_pairs],
                "Corridor Condition Asset": [s for s, _ in cross_pairs],
                "Currency": [currency] * len(cross_pairs),
            })
            res = _solve_ra_old(cross_df, True, barrier_dn, barrier_up, strike_date, last_obs_date)
            for tr in res.ticker_results:
                # index leg
                nova, n_total = _build_instrument(tr.ticker, tr.corridor_asset, True,
                                                  strike_date, last_obs_date,
                                                  barrier_dn, barrier_up, currency)
                n_obs = _n_corridor_obs([nova], datetime.now(), model_context, snap["name"])[0]
                zcb = _unfunded_zcb(currency, strike_date, last_obs_date)
                rows.append({"case": "cross (index leg)", "ticker": f"{tr.ticker}/{tr.corridor_asset}",
                             "RA_old": tr.range_accrual,
                             "RA_new": _ra_from_payout_trace(n_obs, n_total, zcb) if n_obs is not None else None,
                             "n_total_obs": n_total})
                # mono leg — same corridor asset, same n_obs (no extra call needed in prod)
                nova_m, _ = _build_instrument(tr.corridor_asset, tr.corridor_asset, True,
                                              strike_date, last_obs_date,
                                              barrier_dn, barrier_up, currency)
                n_obs_m = _n_corridor_obs([nova_m], datetime.now(), model_context, snap["name"])[0]
                rows.append({"case": "cross (mono leg)", "ticker": tr.corridor_asset,
                             "RA_old": tr.range_accrual_mono,
                             "RA_new": _ra_from_payout_trace(n_obs_m, n_total, zcb) if n_obs_m is not None else None,
                             "n_total_obs": n_total})

    out = pd.DataFrame(rows)
    out["rel_diff_%"] = ((out["RA_new"] - out["RA_old"]).abs()
                         / out["RA_old"].abs() * 100).round(4)

    worst = out["rel_diff_%"].max()
    if worst < 0.1:
        st.success(f"✅ PASS — worst relative diff: {worst:.4f}% (threshold 0.1%)")
    else:
        st.error(f"⚠️ CHECK NEEDED — worst relative diff: {worst:.4f}% (threshold 0.1%)")

    st.dataframe(out, use_container_width=True)
