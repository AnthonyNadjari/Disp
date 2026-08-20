"""RA display page — PayoutTrace RA in the engine (single run).

The engine now computes RA exclusively via the PayoutTrace metric
(E[n_corridor_obs] × unfunded ZCB for the strike; undiscounted fraction for display).
This page runs one solve and shows the RA / strike / discount columns.

Run from the repo root:
    streamlit run pages/RaPayoutTrace_Compare.py
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from functions.dispersion import solve, DispersionConfig
from functions.dispersion.models import ProductType

st.set_page_config(page_title="RA — PayoutTrace", page_icon="⚖️", layout="wide")
st.title("⚖️ RA via PayoutTrace — engine output")
st.caption(
    "RA displayed = undiscounted day fraction E[n_corridor_obs]/n_total; "
    "strikes use ZCB(unfunded, leg currency) × that fraction. "
    "Strike Vanilla Var = sqrt(−EV/ZCB)."
)

with st.sidebar:
    st.header("Inputs")
    mono_txt = st.text_input("Mono tickers (comma)", value="ISP.MI, TTEF.PA")
    cross_txt = st.text_input("Cross pairs stock/index (comma)", value="ISP.MI/.STOXX50E, ASML.AS/.STOXX50E")
    currency = st.selectbox("Currency", ["EUR", "USD", "GBP", "CHF"], index=0)
    barrier_dn = st.number_input("Lower barrier", value=0.70, step=0.05)
    barrier_up = st.number_input("Upper barrier", value=1.30, step=0.05)
    strike_date = st.date_input("Strike date", value=date.today())
    last_obs_date = st.date_input("Last obs date", value=date.today() + timedelta(days=365))

run = st.button("🚀 Run", type="primary", use_container_width=True)

if run:
    rows = []

    def _fill(df, cross, label):
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
        for tr in res.ticker_results:
            rows.append({
                "case": label,
                "ticker": tr.ticker,
                "corr": tr.corridor_asset,
                "RA (undisc)": tr.range_accrual,
                "ZCB": tr.discount_factor,
                "strike": tr.strike_variance_asset,
                "strike mono": tr.strike_corridor_asset,
                "strike vanilla var": tr.strike_vanilla_var,
                "tenor (bd)": tr.obs_dates_cross,
            })

    mono_tickers = [t.strip() for t in mono_txt.split(",") if t.strip()]
    cross_pairs = []
    for p in cross_txt.split(","):
        p = p.strip()
        if "/" in p:
            s, i = p.split("/", 1)
            cross_pairs.append((s.strip(), i.strip()))

    with st.spinner("Solving..."):
        if mono_tickers:
            mono_df = pd.DataFrame({
                "Variance Asset": mono_tickers,
                "Corridor Condition Asset": mono_tickers,
                "Currency": [currency] * len(mono_tickers),
            })
            _fill(mono_df, False, "mono")
        if cross_pairs:
            cross_df = pd.DataFrame({
                "Variance Asset": [i for _, i in cross_pairs],
                "Corridor Condition Asset": [s for s, _ in cross_pairs],
                "Currency": [currency] * len(cross_pairs),
            })
            _fill(cross_df, True, "cross")

    st.dataframe(pd.DataFrame(rows), use_container_width=True)
