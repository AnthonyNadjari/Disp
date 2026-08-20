"""Minimal Streamlit page to test PayoutTraceVariable on a cross-corridor variance swap.

Run from the Gaia_PP root:
    streamlit run pages/PayoutTraceVariable_Test.py
"""

from datetime import date
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow execution from Gaia_PP/pages or Gaia_PP/notebooks.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from functions.dispersion._portal import ensure_portal, refresh_token
from functions.dispersion._pricing import build_corridor_fpf


METRIC = "PayoutTraceVariableExpectation"


def _extract_metric(raw, metric: str):
    """Find metric recursively without assuming one exact portal response shape."""
    if raw is None:
        return None

    if isinstance(raw, pd.DataFrame):
        if metric in raw.columns:
            return raw[metric].iloc[0]
        if metric in raw.index:
            row = raw.loc[metric]
            return row.iloc[0] if hasattr(row, "iloc") else row
        return None

    if isinstance(raw, dict):
        if metric in raw:
            return raw[metric]
        for value in raw.values():
            found = _extract_metric(value, metric)
            if found is not None:
                return found
        return None

    if isinstance(raw, (list, tuple)):
        for value in raw:
            found = _extract_metric(value, metric)
            if found is not None:
                return found
        return None

    if hasattr(raw, metric):
        return getattr(raw, metric)
    if hasattr(raw, "to_dict"):
        try:
            return _extract_metric(raw.to_dict(), metric)
        except Exception:
            pass
    return None


def price_payout_trace(
    variance_asset: str,
    corridor_asset: str,
    strike_date: date,
    maturity_date: date,
    strike_pct: float,
    lower_barrier_pct: float,
    upper_barrier_pct: float,
    currency: str,
    is_capped: bool,
):
    from datetime import datetime

    from pricingportal import NovaIdSource
    from speq.fpf.unified_economics_schema.fpf_schema import (
        FPFUnifiedEconomicsWrapper,
    )

    portal, snap = ensure_portal()
    refresh_token()

    # 1. Build the FPF string
    fpf = build_corridor_fpf(
        tickers=[variance_asset],
        last_obs_date=maturity_date,
        strike_date=strike_date,
        strikes=[strike_pct / 100.0],
        weights=[1.0],
        low_barrier=lower_barrier_pct / 100.0,
        high_barrier=upper_barrier_pct / 100.0,
        is_capped=is_capped,
        corr_asset=corridor_asset,
        currency=currency,
        schedule_calendar_asset=variance_asset,
        use_parameters=False,
    )
    fpf_unified = FPFUnifiedEconomicsWrapper.from_data(
        fpf,
        script_cls=...,
    )

    st.subheader("Unified FPF fields")

    st.write("Object type:", type(fpf_unified).__name__)
    st.write("Available fields:", list(fpf_unified.__dict__.keys()))

    st.subheader("Full unified FPF")
    st.code(fpf_unified.to_fpf_string(), language="text")

    st.subheader("Unified FPF object dump")

    try:
        st.json(fpf_unified.model_dump(mode="json"))
    except Exception:
        st.write(vars(fpf_unified))

    # 2. Parse the FPF string
    fpf_wrapper = FPFUnifiedEconomicsWrapper.from_data(
        fpf,
        script_cls=...,
    )

    # 3. Load both underlying instruments
    underlyings = [
        portal.load_instrument(
            schema=NovaIdSource.REUTERS,
            instrument_id=variance_asset,
        ),
        portal.load_instrument(
            schema=NovaIdSource.REUTERS,
            instrument_id=corridor_asset,
        ),
    ]

    # 4. Convert FPF string into a Pricing Portal instrument
    nova_fpf = portal.create_fpf(
        fpf_string=fpf_wrapper.to_fpf_string(),
        instrument_ccy=currency,
        underlyings=underlyings,
        premium_date=strike_date,
    )

    # 5. Cross-corridor requires a multi-asset model context
    model_context = portal.create_model_context(
        name="EMEA-Stocks-MC-LV-MultiAsset",
        instrument_model_parameters={},
    )
    raw = portal.price(
        price_id="PayoutTraceTest",
        instruments=[nova_fpf],

        valuation_date=datetime.now(),

        metrics=[{
            "name": METRIC,
            "metricParameters": {
                "DumpStateNames": {
                    "value": "var_calculatePayoff_nCorridorObs"
                }
            }
        }],
        calculation_parameters={},


        model_context=model_context,
        overridden_snap_name=snap["name"],
    )


    metric_value = _extract_metric(raw, METRIC)

    st.subheader("Full portal response")
    st.write(raw)
    st.json(json.loads(json.dumps(raw, default=str)))

    if metric_value is None:
        st.warning(f"{METRIC} was not returned. Check the full response below.")


    return metric_value, raw, fpf
st.set_page_config(page_title="PayoutTraceVariable Test", layout="wide")
st.title("PayoutTraceVariable Test")
st.caption("Minimal cross-corridor variance swap tester")

with st.form("payout_trace_form"):
    c1, c2 = st.columns(2)
    with c1:
        variance_asset = st.text_input("Variance asset", value="ISP.MI")
        corridor_asset = st.text_input("Corridor condition asset", value=".STOXX50E")
        currency = st.selectbox("Currency", ["EUR", "USD", "GBP", "CHF"], index=0)
        strike_pct = st.number_input("Strike (%)", min_value=0.01, value=25.00, step=0.25)
    with c2:
        strike_date = st.date_input("Strike date", value=date.today())
        maturity_date = st.date_input("Last observation date", value=date(date.today().year + 1, date.today().month, date.today().day))
        lower_barrier_pct = st.number_input("Lower barrier (% spot)", min_value=0.0, value=70.0, step=1.0)
        upper_barrier_pct = st.number_input("Upper barrier (% spot)", min_value=0.0, value=130.0, step=1.0)
        is_capped = st.checkbox("Capped", value=False)

    submitted = st.form_submit_button("Price", type="primary", use_container_width=True)

if submitted:
    if not variance_asset.strip() or not corridor_asset.strip():
        st.error("Variance asset and corridor condition asset are required.")
    elif maturity_date <= strike_date:
        st.error("Last observation date must be after strike date.")
    elif lower_barrier_pct >= upper_barrier_pct:
        st.error("Lower barrier must be below upper barrier.")
    else:
        try:
            with st.spinner("Pricing..."):
                metric_value, raw_result, fpf = price_payout_trace(
                    variance_asset=variance_asset.strip(),
                    corridor_asset=corridor_asset.strip(),
                    strike_date=strike_date,
                    maturity_date=maturity_date,
                    strike_pct=strike_pct,
                    lower_barrier_pct=lower_barrier_pct,
                    upper_barrier_pct=upper_barrier_pct,
                    currency=currency,
                    is_capped=is_capped,
                )

            if metric_value is None:
                st.warning(f"Pricing completed, but {METRIC} was not found in the returned payload.")
            else:
                st.metric(METRIC, str(metric_value))

            with st.expander("Raw portal response"):
                if isinstance(raw_result, pd.DataFrame):
                    st.dataframe(raw_result, use_container_width=True)
                else:
                    try:
                        st.json(raw_result)
                    except Exception:
                        st.code(repr(raw_result))

            with st.expander("Generated FPF"):
                st.code(fpf)

        except Exception as exc:
            st.exception(exc)
