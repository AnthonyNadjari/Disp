"""RA Comparison page — legacy RA instruments vs PayoutTrace RA (engine flag).

Ultra-simple: click "Run comparison" — the SAME solve is run twice through the
engine (use_payout_trace_ra OFF vs ON) and the RA / strike columns are compared.

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

st.set_page_config(page_title="RA Comparison", page_icon="⚖️", layout="wide")
st.title("⚖️ RA — legacy instruments vs PayoutTrace (use_payout_trace_ra)")
st.caption(
    "Same solve, run twice through the engine. Expected: strikes identical (MC "
    "noise aside); RA displayed ≈ legacy/ZCB in the new path (undiscounted fraction)."
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

run = st.button("🚀 Run comparison", type="primary", use_container_width=True)

if run:
    mono_tickers = [t.strip() for t in mono_txt.split(",") if t.strip()]
    cross_pairs = []
    for p in cross_txt.split(","):
        p = p.strip()
        if "/" in p:
            s, i = p.split("/", 1)
            cross_pairs.append((s.strip(), i.strip()))

    cfg_common = dict(
        product_type=ProductType.VAR_SWAP_CORRIDOR,
        barrier_up=barrier_up, barrier_down=barrier_dn,
        local_cap=2.5, is_capped=True,
    )

    def _run(df, cross, ptr):
        cfg = DispersionConfig(cross_corridor=cross, **cfg_common)
        res = solve(df=df, config=cfg,
                    last_obs_date=last_obs_date, strike_date=strike_date,
                    eqeq_lambda=0.10, eqfx_shift=-0.05, vol_mode="ATMF",
                    use_payout_trace_ra=ptr)
        if not res.success:
            raise RuntimeError(f"solve failed (ptr={ptr}) for {list(df['Variance Asset'])}")
        return res

    rows = []

    def _fill(df, cross, label_fn):
        with st.spinner("Legacy solve (flag OFF)..."):
            res_old = _run(df, cross, False)
        with st.spinner("PayoutTrace solve (flag ON)..."):
            res_new = _run(df, cross, True)
        by_ticker_old = {tr.ticker: tr for tr in res_old.ticker_results}
        for tr_new in res_new.ticker_results:
            tr_old = by_ticker_old.get(tr_new.ticker)
            if tr_old is None:
                continue
            rows.append({
                "case": label_fn(tr_new),
                "ticker": tr_new.ticker,
                "corr": tr_new.corridor_asset,
                "RA_legacy": tr_old.range_accrual,
                "RA_ptrace(undisc)": tr_new.range_accrual,
                "ZCB": tr_new.discount_factor,
                "strike_legacy": tr_old.strike_variance_asset,
                "strike_ptrace": tr_new.strike_variance_asset,
                "strike_mono_legacy": tr_old.strike_corridor_asset,
                "strike_mono_ptrace": tr_new.strike_corridor_asset,
                "vanilla_var": tr_new.strike_vanilla_var,
            })

    if mono_tickers:
        mono_df = pd.DataFrame({
            "Variance Asset": mono_tickers,
            "Corridor Condition Asset": mono_tickers,
            "Currency": [currency] * len(mono_tickers),
        })
        _fill(mono_df, False, lambda tr: "mono")

    if cross_pairs:
        cross_df = pd.DataFrame({
            "Variance Asset": [i for _, i in cross_pairs],
            "Corridor Condition Asset": [s for s, _ in cross_pairs],
            "Currency": [currency] * len(cross_pairs),
        })
        _fill(cross_df, True, lambda tr: "cross")

    out = pd.DataFrame(rows)
    if out.empty:
        st.error("No comparable rows (both solves must succeed).")
    else:
        out["dstrike_%"] = ((out["strike_ptrace"] - out["strike_legacy"]).abs()
                            / out["strike_legacy"].abs() * 100).round(4)
        out["dstrike_mono_%"] = ((out["strike_mono_ptrace"] - out["strike_mono_legacy"]).abs()
                                 / out["strike_mono_legacy"].abs() * 100).round(4)
        out["RA_check (legacy×1/ZCB)"] = (out["RA_legacy"] / out["ZCB"]).round(4)

        st.dataframe(out, use_container_width=True)

        worst = out[["dstrike_%", "dstrike_mono_%"]].max().max()
        if worst < 0.1:
            st.success(f"✅ PASS — worst strike diff: {worst:.4f}% (threshold 0.1%)")
        else:
            st.error(f"⚠️ CHECK — worst strike diff: {worst:.4f}% (threshold 0.1%)")
        st.caption(
            "Strikes must match (MC noise aside). RA_ptrace should equal "
            "RA_check = legacy/ZCB (the undiscounted day fraction). "
            "vanilla_var = sqrt(−EV/ZCB)."
        )
