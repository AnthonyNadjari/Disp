"""Compare legacy RA (dedicated range-accrual instrument) vs PayoutTrace-based RA.

Validates the new RA formula BEFORE the engine switches to it:

    RA_new = ZCB_unfunded(currency, strike->maturity) * E[n_corridor_obs] / n_total_obs

against the RA the engine currently prices (TickerResult.range_accrual /
range_accrual_mono, from the dedicated RA instrument in the batch).

Run from the repo root, Bloomberg session up:

    python scripts/compare_ra_payout_trace.py

PASS criterion: |RA_new - RA_old| / RA_old < ~1e-3 on every leg (MC noise aside).
"""

import os
import sys
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

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

# ── Test cases (edit me) ──────────────────────────────────────────────────────
MONO_TICKERS = ["ISP.MI", "TTEF.PA"]                    # mono corridor (asset == corridor asset)
CROSS_PAIRS = [("ISP.MI", ".STOXX50E"), ("ASML.AS", ".STOXX50E")]  # (stock, index)
CURRENCY = "EUR"
BARRIER_DN, BARRIER_UP = 0.70, 1.30
MODEL_CONTEXT_NAME = "EMEA-Stocks-MC-LV-MultiAsset"

strike_date = date.today()
last_obs_date = strike_date + timedelta(days=365)


def _build_instrument(var_asset, corr_asset, is_capped):
    """Rebuild the same corridor FPF the engine solves on + obs count."""
    fpf = build_corridor_fpf(
        tickers=[var_asset],
        last_obs_date=last_obs_date,
        strike_date=strike_date,
        strikes=[0.000001],          # placeholder — RA does not depend on the strike
        weights=[1.0],
        low_barrier=BARRIER_DN,
        high_barrier=BARRIER_UP,
        is_capped=is_capped,
        corr_asset=corr_asset,
        currency=CURRENCY,
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
        instrument_ccy=CURRENCY,
        underlyings=underlyings,
        premium_date=strike_date,
    )
    n_total = len(wrapper.observationDates) - 1  # engine convention: today excluded
    return nova, n_total


def _ra_new_for(var_asset, corr_asset, is_capped, model_context, snap_name):
    nova, n_total = _build_instrument(var_asset, corr_asset, is_capped)
    n_obs = _n_corridor_obs([nova], datetime.now(), model_context, snap_name)[0]
    if n_obs is None:
        return None, n_total
    zcb = _unfunded_zcb(CURRENCY, strike_date, last_obs_date)
    return _ra_from_payout_trace(n_obs, n_total, zcb), n_total


def _solve_ra_old(df, cross):
    cfg = DispersionConfig(
        product_type=ProductType.VAR_SWAP_CORRIDOR,
        cross_corridor=cross,
        barrier_up=BARRIER_UP, barrier_down=BARRIER_DN,
        local_cap=2.5, is_capped=True,
    )
    res = solve(df=df, config=cfg, last_obs_date=last_obs_date, strike_date=strike_date,
                eqeq_lambda=0.10, eqfx_shift=-0.05, vol_mode="ATMF")
    if not res.success:
        raise RuntimeError(f"solve failed for {list(df['Variance Asset'])}")
    return res


def main():
    portal = get_pricing_portal()
    snap = get_live_snap()
    model_context = portal.create_model_context(
        name=MODEL_CONTEXT_NAME, instrument_model_parameters={})
    rows = []

    # ── Mono corridor ──
    mono_df = pd.DataFrame({
        "Variance Asset": MONO_TICKERS,
        "Corridor Condition Asset": MONO_TICKERS,
        "Currency": [CURRENCY] * len(MONO_TICKERS),
    })
    res_mono = _solve_ra_old(mono_df, cross=False)
    for tr in res_mono.ticker_results:
        ra_new, n_total = _ra_new_for(tr.ticker, tr.ticker, True, model_context, snap["name"])
        rows.append({"case": "mono", "ticker": tr.ticker,
                     "RA_old": tr.range_accrual, "RA_new": ra_new, "n_total_obs": n_total})

    # ── Cross corridor ──
    cross_df = pd.DataFrame({
        "Variance Asset": [idx for _, idx in CROSS_PAIRS],
        "Corridor Condition Asset": [s for s, _ in CROSS_PAIRS],
        "Currency": [CURRENCY] * len(CROSS_PAIRS),
    })
    res_cross = _solve_ra_old(cross_df, cross=True)
    for tr in res_cross.ticker_results:
        # cross leg RA (index in stock corridor)
        ra_new_x, n_total = _ra_new_for(tr.ticker, tr.corridor_asset, True,
                                        model_context, snap["name"])
        rows.append({"case": "cross (index leg)", "ticker": f"{tr.ticker}/{tr.corridor_asset}",
                     "RA_old": tr.range_accrual, "RA_new": ra_new_x, "n_total_obs": n_total})
        # mono leg RA (stock in its own corridor) — same corridor asset => same n_obs,
        # so in production this needs NO extra metric call (computed once per corridor asset)
        ra_new_m, _ = _ra_new_for(tr.corridor_asset, tr.corridor_asset, True,
                                  model_context, snap["name"])
        rows.append({"case": "cross (mono leg)", "ticker": tr.corridor_asset,
                     "RA_old": tr.range_accrual_mono, "RA_new": ra_new_m, "n_total_obs": n_total})

    out = pd.DataFrame(rows)
    out["rel_diff_%"] = ((out["RA_new"] - out["RA_old"]).abs() / out["RA_old"].abs() * 100).round(4)
    pd.set_option("display.width", 160)
    print("\n" + out.to_string(index=False))
    worst = out["rel_diff_%"].max()
    print(f"\nWorst relative diff: {worst:.4f}%  ({'PASS' if worst < 0.1 else 'CHECK NEEDED'} — threshold 0.1%)")


if __name__ == "__main__":
    main()
