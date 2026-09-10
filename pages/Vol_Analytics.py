"""
Vol Analytics — implied vs subsequent realised volatility, with variance-swap economics.

Everything lives in the main page: a collapsible input panel at the top, live display
controls, then the analysis tabs. No sidebar.

What it does
------------
* Constant-maturity ATM implied vol vs the volatility that actually realised over the
  matching forward window, per name and for a weighted basket.
* Spread expressed in vol points, variance points or vega-equivalent, with percentile
  and rolling-percentile context.
* Realised vol on the **variance-swap convention** (zero-mean, Σr²/N) by default, with
  the demeaned sample-σ variant available.
* Regression of realised on implied with **Newey-West** standard errors, because the
  observations overlap by one horizon and plain OLS t-stats are meaningless there.
* Short-variance carry, optionally **capped** (the desk's 2.5× local cap), reported
  across *all* non-overlapping schedules so the result does not hinge on the arbitrary
  start date of one of them.
* Term structure on matched observation dates, not today's implied against a stale
  realised.
* Dispersion: implied vs subsequent realised correlation, the dispersion carry
  (weighted single-name variance spread minus the index one), and a single-name ranking.

Design notes
------------
* Results are latched in session state, so touching a display control never wipes the
  page — the classic "click Run again" bug of button-gated Streamlit apps.
* All Bloomberg access goes through one cached dataset builder, so display controls are
  instant and never refetch.
* Errors are scoped per section: a failure in one tab does not blank the whole page.
"""

from __future__ import annotations

import datetime as dt
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from xbbg import blp

# ────────────────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Vol Analytics", layout="wide")

TRADING_DAYS = 252

TENOR_CONFIG = {
    "1M": {"days": 21,  "field": "30DAY_IMPVOL_100.0%MNY_DF", "fb": "MATURITY_30D"},
    "3M": {"days": 63,  "field": "3MTH_IMPVOL_100.0%MNY_DF",  "fb": "MATURITY_90D"},
    "6M": {"days": 126, "field": "6MTH_IMPVOL_100.0%MNY_DF",  "fb": "MATURITY_180D"},
    "1Y": {"days": 252, "field": "12MTH_IMPVOL_100.0%MNY_DF", "fb": "MATURITY_360D"},
}

BASKET = "BASKET (avg)"

SPREAD_BASES = {
    "Variance": "variance points",
    "Vol": "vol points",
    "Vega-equivalent": "vol points (vega-equiv.)",
}

PCT_WINDOWS = {"1Y": TRADING_DAYS, "3Y": 3 * TRADING_DAYS, "5Y": 5 * TRADING_DAYS, "All": None}

BLUE, ORANGE, RED, GREY = "#00AEEF", "#F2A900", "#D62728", "#8C8C8C"


# ────────────────────────────────────────────────────────────────────────────
# INPUT PARSING
# ────────────────────────────────────────────────────────────────────────────

def parse_universe(text: str) -> tuple[list[str], Optional[np.ndarray]]:
    """Parse the underlyings box.

    One name per line, optionally followed by a weight after a comma, tab or
    semicolon — so an Excel paste works as is::

        SPX Index
        AAPL US Equity, 30
        MSFT US Equity, 70

    Returns ``(tickers, weights)`` with ``weights`` normalised to sum to 1, or
    ``None`` when no line carries one. Duplicates keep their first occurrence and
    their weights are summed.
    """
    tickers: list[str] = []
    raw_weights: list[Optional[float]] = []
    for line in text.splitlines():
        row = line.strip()
        if not row:
            continue
        parts = [p.strip() for p in row.replace("\t", ",").replace(";", ",").split(",")]
        parts = [p for p in parts if p]
        if not parts:
            continue
        name, weight = parts[0], None
        if len(parts) > 1:
            try:
                weight = float(parts[-1].replace("%", "").strip())
            except ValueError:
                raise ValueError(
                    f"Cannot read a weight from {row!r} — expected 'TICKER, weight'."
                )
        if name in tickers:
            i = tickers.index(name)
            if weight is not None:
                raw_weights[i] = (raw_weights[i] or 0.0) + weight
            continue
        tickers.append(name)
        raw_weights.append(weight)

    if not tickers:
        return [], None

    given = [w is not None for w in raw_weights]
    if not any(given):
        return tickers, None
    if not all(given):
        missing = [t for t, g in zip(tickers, given) if not g]
        raise ValueError(
            "Give a weight to every name or to none. Missing: " + ", ".join(missing)
        )

    w = np.array([float(x) for x in raw_weights], dtype=float)
    if not np.isfinite(w).all():
        raise ValueError("Weights must be finite numbers.")
    if np.isclose(w.sum(), 0.0):
        raise ValueError("Weights sum to zero.")
    if (w < 0).any():
        raise ValueError("Negative weights are not supported here — use positive basket weights.")
    return tickers, w / w.sum()


# ────────────────────────────────────────────────────────────────────────────
# BLOOMBERG
# ────────────────────────────────────────────────────────────────────────────

def _clean(raw: pd.DataFrame, tickers: Iterable[str]) -> pd.DataFrame:
    """Normalise a bdh result to a numeric frame with exactly ``tickers`` as columns."""
    tickers = list(tickers)
    if raw is None or getattr(raw, "empty", True):
        return pd.DataFrame(index=pd.DatetimeIndex([]), columns=tickers, dtype=float)
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.loc[:, ~df.columns.duplicated()]
    lookup = {c.upper(): c for c in df.columns}
    out = pd.DataFrame(index=df.index)
    for t in tickers:
        col = lookup.get(t.upper())
        out[t] = df[col] if col is not None else np.nan
    return out.sort_index()


def _bdh(tickers: Sequence[str], field: str, start: dt.date, end: dt.date, **ovr) -> pd.DataFrame:
    raw = blp.bdh(
        tickers=list(tickers),
        flds=[field],
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
        **ovr,
    )
    return _clean(raw, tickers)


def _load_prices(tickers: Sequence[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    return _bdh(tickers, "PX_LAST", start, end)


def _load_implied(tickers: Sequence[str], start: dt.date, end: dt.date,
                  tenor: str) -> tuple[pd.DataFrame, list[str]]:
    """Constant-maturity ATM implied vol, with the IVOL_MONEYNESS fallback per name."""
    cfg = TENOR_CONFIG[tenor]
    df = _bdh(tickers, cfg["field"], start, end)
    missing = [t for t in tickers if t not in df.columns or df[t].dropna().empty]
    used_fallback: list[str] = []
    if missing:
        fb = _bdh(
            missing, "IVOL_MONEYNESS", start, end,
            IVOL_MATURITY=cfg["fb"],
            IVOL_MONEYNESS_LEVEL="MONEY_LVL_100_0",
        )
        for t in missing:
            if t in fb.columns and not fb[t].dropna().empty:
                df[t] = fb[t]
                used_fallback.append(t)
    return df.reindex(columns=list(tickers)), used_fallback


# ────────────────────────────────────────────────────────────────────────────
# VOL / VARIANCE MATH
# ────────────────────────────────────────────────────────────────────────────

def forward_realised_vol(prices: pd.DataFrame, horizon: int, demean: bool = False) -> pd.DataFrame:
    """Annualised realised vol, in vol points, over the window *after* each date.

    ``demean=False`` (default) is the variance-swap convention, ``√(mean(r²))`` with no
    mean subtraction, which is what a variance swap actually settles on. ``demean=True``
    uses the sample standard deviation instead.

    The value carried at date *t* covers returns over ``(t, t + horizon]``, so it lines
    up with an implied vol observed at *t*. The last ``horizon`` rows are therefore NaN.
    """
    if prices.empty:
        return prices.copy()
    logret = np.log(prices / prices.shift(1))
    roll = logret.rolling(horizon, min_periods=horizon)
    rv = roll.std(ddof=1) if demean else np.sqrt(logret.pow(2).rolling(horizon, min_periods=horizon).mean())
    return (rv * np.sqrt(TRADING_DAYS) * 100.0).shift(-horizon)


def weighted_avg(df: pd.DataFrame, w: pd.Series) -> pd.Series:
    """Weighted cross-sectional average, renormalised on the names available each day."""
    cols = [c for c in df.columns if c in w.index]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    sub, ww = df[cols], w[cols]
    num = sub.mul(ww, axis=1).sum(axis=1, min_count=1)
    den = sub.notna().mul(ww, axis=1).sum(axis=1).replace(0.0, np.nan)
    return num / den


def implied_correlation(sigma_names: pd.DataFrame, w: pd.Series, sigma_idx: pd.Series) -> pd.Series:
    """ρ = (σ_idx² − Σ wᵢ²σᵢ²) / ((Σ wᵢσᵢ)² − Σ wᵢ²σᵢ²).

    Vols in vol points; the ratio is scale-free. ρ is left unclipped on purpose, so a
    basket that does not actually replicate the index stays visibly diagnostic.
    """
    idx = pd.to_numeric(sigma_idx, errors="coerce").reindex(sigma_names.index)
    cols = [c for c in sigma_names.columns if c in w.index]
    sub, ww = sigma_names[cols], w[cols]
    wsum = sub.mul(ww, axis=1).sum(axis=1, min_count=1)
    wsq = (sub ** 2).mul(ww ** 2, axis=1).sum(axis=1, min_count=1)
    denom = (wsum ** 2) - wsq
    return ((idx ** 2) - wsq) / denom.where(denom.abs() > 1e-12)


def basket_vol_from_rho(sigma_names: pd.DataFrame, w: pd.Series, rho: float) -> pd.Series:
    """Correlation-consistent basket vol: √(ρ(Σwσ)² + (1−ρ)Σw²σ²)."""
    cols = [c for c in sigma_names.columns if c in w.index]
    sub, ww = sigma_names[cols], w[cols]
    wsum = sub.mul(ww, axis=1).sum(axis=1, min_count=1)
    wsq = (sub ** 2).mul(ww ** 2, axis=1).sum(axis=1, min_count=1)
    return np.sqrt(rho * wsum ** 2 + (1.0 - rho) * wsq)


def spread_frame(implied: pd.DataFrame, realised: pd.DataFrame, basis: str) -> pd.DataFrame:
    """Implied − realised on the requested basis."""
    if basis == "Vol":
        return implied - realised
    var = implied ** 2 - realised ** 2
    if basis == "Variance":
        return var
    # Vega-equivalent: variance spread divided by 2K, i.e. what a var swap pays per vega
    return var / (2.0 * implied.where(implied > 0))


# ────────────────────────────────────────────────────────────────────────────
# STATS
# ────────────────────────────────────────────────────────────────────────────

def percentile_of_last(series: pd.Series, window: Optional[int]) -> float:
    """Share of observations at or below the latest one, in percent."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan
    if window is not None:
        s = s.iloc[-window:]
    return float(100.0 * (s <= s.iloc[-1]).mean())


def rolling_self_percentile(series: pd.Series, window: Optional[int]) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")

    def pct(x: np.ndarray) -> float:
        return 100.0 * float((x <= x[-1]).mean())

    if window is None:
        return s.expanding(min_periods=20).apply(pct, raw=True)
    return s.rolling(window, min_periods=min(window, 60)).apply(pct, raw=True)


def zscore_of_last(series: pd.Series, window: Optional[int]) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if window is not None:
        s = s.iloc[-window:]
    if len(s) < 5 or s.std(ddof=1) == 0:
        return np.nan
    return float((s.iloc[-1] - s.mean()) / s.std(ddof=1))


def _newey_west_se(x: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    """Bartlett-kernel HAC standard errors for the OLS of y on [1, x]."""
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    xtx_inv = np.linalg.pinv(X.T @ X)
    u = X * resid[:, None]
    S = u.T @ u
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        G = u[lag:].T @ u[:-lag]
        S = S + weight * (G + G.T)
    cov = xtx_inv @ S @ xtx_inv
    return np.sqrt(np.clip(np.diag(cov), 0.0, None))


def regression_stats(implied: pd.Series, realised: pd.Series, horizon: int) -> dict:
    """Realised on implied, with HAC inference.

    Daily observations of a ``horizon``-day forward window overlap, so residuals are
    autocorrelated by construction; plain OLS standard errors understate uncertainty by
    roughly √horizon. Newey-West with ``horizon`` lags is the standard correction, and
    the t-stat against β = 1 is the one that matters: β < 1 means implied over-predicts
    realised, which is the volatility risk premium being paid.
    """
    df = pd.concat([implied.rename("i"), realised.rename("r")], axis=1).dropna()
    if len(df) < 20:
        return {}
    x, y = df["i"].to_numpy(float), df["r"].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (intercept + slope * x)
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    se_a, se_b = _newey_west_se(x, resid, lags=horizon)
    vol_sp = df["i"] - df["r"]
    var_sp = df["i"] ** 2 - df["r"] ** 2
    return {
        "n": int(len(df)),
        "n_independent": max(1, int(len(df) / max(horizon, 1))),
        "slope": float(slope),
        "intercept": float(intercept),
        "se_slope": float(se_b),
        "t_vs_one": float((slope - 1.0) / se_b) if se_b > 0 else np.nan,
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "hit_ratio": float((df["i"] > df["r"]).mean()),
        "mean_vol_spread": float(vol_sp.mean()),
        "mean_var_spread": float(var_sp.mean()),
        "first": df.index[0],
        "last": df.index[-1],
    }


def var_swap_pnl(strike: pd.Series, realised: pd.Series, cap_mult: Optional[float]) -> pd.Series:
    """Short-variance P&L in variance points: K² − R², capped at K² − (cap·K)²."""
    pnl = strike ** 2 - realised ** 2
    if cap_mult:
        pnl = np.maximum(pnl, strike ** 2 - (cap_mult * strike) ** 2)
    return pnl


def carry_schedule(implied: pd.Series, realised: pd.Series, horizon: int,
                   cap_mult: Optional[float], offset: int = 0) -> pd.DataFrame:
    """One non-overlapping short-variance schedule starting at ``offset``."""
    df = pd.concat([implied.rename("k"), realised.rename("r")], axis=1).dropna()
    if df.empty or offset >= len(df):
        return pd.DataFrame()
    t = df.iloc[offset::horizon].copy()
    t["var_pnl"] = var_swap_pnl(t["k"], t["r"], cap_mult)
    t["vega_pnl"] = t["var_pnl"] / (2.0 * t["k"].where(t["k"] > 0))
    t["cum_var_pnl"] = t["var_pnl"].cumsum()
    t["cum_vega_pnl"] = t["vega_pnl"].cumsum()
    peak = t["cum_vega_pnl"].cummax()
    t["drawdown"] = t["cum_vega_pnl"] - peak
    return t


def carry_all_offsets(implied: pd.Series, realised: pd.Series, horizon: int,
                      cap_mult: Optional[float]) -> pd.DataFrame:
    """Summary of every non-overlapping schedule, so no single start date drives the answer."""
    rows = []
    for off in range(horizon):
        t = carry_schedule(implied, realised, horizon, cap_mult, offset=off)
        if len(t) < 2:
            continue
        vega = t["vega_pnl"]
        sd = float(vega.std(ddof=1))
        rows.append({
            "offset": off,
            "trades": int(len(t)),
            "total_vega_pnl": float(vega.sum()),
            "mean_vega_pnl": float(vega.mean()),
            "win_rate": float((t["var_pnl"] > 0).mean()),
            "sharpe": float(vega.mean() / sd * np.sqrt(TRADING_DAYS / horizon)) if sd > 0 else np.nan,
            "max_drawdown": float(t["drawdown"].min()),
        })
    return pd.DataFrame(rows)


def matched_last(implied: pd.Series, realised: pd.Series) -> tuple[float, float, object]:
    """Last date where both an implied quote and its completed realised window exist."""
    df = pd.concat([implied.rename("i"), realised.rename("r")], axis=1).dropna()
    if df.empty:
        return np.nan, np.nan, pd.NaT
    return float(df["i"].iloc[-1]), float(df["r"].iloc[-1]), df.index[-1]


# ────────────────────────────────────────────────────────────────────────────
# DATASET (single cached Bloomberg round-trip)
# ────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def build_dataset(tickers: tuple[str, ...], weights: tuple[float, ...], index_ticker: str,
                  start: dt.date, end: dt.date, tenor: str, demean: bool,
                  with_term: bool) -> dict:
    """Load everything the page needs, once, and return plain frames.

    Cached on its arguments, so every display control below is instant and no widget
    interaction ever hits Bloomberg again.
    """
    horizon = TENOR_CONFIG[tenor]["days"]

    prices = _load_prices(tickers, start, end)
    implied_raw, fallback = _load_implied(tickers, start, end, tenor)

    valid = [t for t in tickers
             if t in prices.columns and not prices[t].dropna().empty
             and t in implied_raw.columns and not implied_raw[t].dropna().empty]
    dropped = [t for t in tickers if t not in valid]
    if not valid:
        raise ValueError("No ticker has both a price history and an implied-vol history.")

    prices = prices[valid]
    implied_raw = implied_raw[valid]
    w = pd.Series(weights, index=list(tickers))[valid]
    w = w / w.sum()

    realised_raw = forward_realised_vol(prices, horizon, demean)

    implied = implied_raw.copy()
    realised = realised_raw.copy()
    if len(valid) > 1:
        implied.insert(0, BASKET, weighted_avg(implied_raw, w))
        realised.insert(0, BASKET, weighted_avg(realised_raw, w))

    out = {
        "prices": prices,
        "implied": implied,
        "realised": realised,
        "implied_names": implied_raw,
        "realised_names": realised_raw,
        "valid": valid,
        "dropped": dropped,
        "weights": w,
        "fallback": [t for t in fallback if t in valid],
        "horizon": horizon,
        "index_ticker": "",
        "index_implied": None,
        "index_realised": None,
        "term": {},
    }

    if index_ticker and len(valid) > 1:
        idx_px = _load_prices([index_ticker], start, end)
        idx_iv, _ = _load_implied([index_ticker], start, end, tenor)
        has_px = index_ticker in idx_px.columns and not idx_px[index_ticker].dropna().empty
        has_iv = index_ticker in idx_iv.columns and not idx_iv[index_ticker].dropna().empty
        if has_px and has_iv:
            out["index_ticker"] = index_ticker
            out["index_implied"] = idx_iv[index_ticker]
            out["index_realised"] = forward_realised_vol(
                idx_px[[index_ticker]], horizon, demean)[index_ticker]

    if with_term:
        for label, cfg in TENOR_CONFIG.items():
            iv_t, _ = _load_implied(valid, start, end, label)
            rv_t = forward_realised_vol(prices, cfg["days"], demean)
            iv_t = iv_t[valid]
            if len(valid) > 1:
                iv_t = iv_t.copy()
                rv_t = rv_t.copy()
                iv_t.insert(0, BASKET, weighted_avg(iv_t, w))
                rv_t.insert(0, BASKET, weighted_avg(rv_t, w))
            out["term"][label] = {"implied": iv_t, "realised": rv_t}

    return out


# ────────────────────────────────────────────────────────────────────────────
# PLOT HELPERS
# ────────────────────────────────────────────────────────────────────────────

def _style(fig: go.Figure, title: str, ylab: str = "", height: int = 520) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        margin=dict(l=20, r=20, t=70, b=20),
        yaxis_title=ylab,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#EEEEEE")
    fig.update_yaxes(showgrid=True, gridcolor="#EEEEEE")
    return fig


def _fmt(value: float, digits: int = 2, suffix: str = "") -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.{digits}f}{suffix}"


# ────────────────────────────────────────────────────────────────────────────
# INPUT PANEL  (main page, no sidebar)
# ────────────────────────────────────────────────────────────────────────────

st.title("Vol Analytics — implied vs subsequent realised")

if "va_cfg" not in st.session_state:
    st.session_state.va_cfg = None

_has_run = st.session_state.va_cfg is not None

with st.expander("⚙️  Inputs", expanded=not _has_run):
    with st.form("va_inputs"):
        c_left, c_right = st.columns([3, 2])

        with c_left:
            universe_text = st.text_area(
                "Underlyings — one per line, optional weight after a comma or tab",
                value=st.session_state.get("va_universe", "SPX Index"),
                height=170,
                help="Examples:\n\nSPX Index\n\nAAPL US Equity, 30\nMSFT US Equity, 70\n\n"
                     "Weights are renormalised to 100%. Leave them out for equal weights. "
                     "An Excel paste of two columns works as is.",
            )
            index_ticker = st.text_input(
                "Index ticker (optional) — enables the Dispersion tab",
                value=st.session_state.get("va_index", ""),
                help="With two or more underlyings, extracts implied correlation and the "
                     "dispersion carry against this index.",
            ).strip()

        with c_right:
            d1, d2 = st.columns(2)
            start_date = d1.date_input(
                "Start", value=st.session_state.get("va_start", dt.date.today() - dt.timedelta(days=5 * 365)),
                format="DD/MM/YYYY")
            end_date = d2.date_input(
                "End", value=st.session_state.get("va_end", dt.date.today()), format="DD/MM/YYYY")

            t1, t2 = st.columns(2)
            _tenors = list(TENOR_CONFIG)
            tenor = t1.selectbox(
                "Tenor", _tenors,
                index=_tenors.index(st.session_state.get("va_tenor", "3M")))
            _convs = ["Var-swap (zero mean)", "Sample σ (demeaned)"]
            rv_convention = t2.selectbox(
                "Realised vol", _convs,
                index=_convs.index(st.session_state.get("va_conv", _convs[0])),
                help="A variance swap settles on Σr²/N with no mean subtraction. The "
                     "demeaned sample σ is the statistician's estimator; on daily equity "
                     "data the two differ by a few basis points of vol.",
            )
            with_term = st.checkbox(
                "Load all tenors (term-structure tab)",
                value=st.session_state.get("va_term", True),
                help="Off makes the first run roughly four times faster.")

        submitted = st.form_submit_button("Run", type="primary", use_container_width=True)

    if submitted:
        try:
            tickers, weights = parse_universe(universe_text)
            if not tickers:
                raise ValueError("Enter at least one ticker.")
            if start_date >= end_date:
                raise ValueError("The end date must be after the start date.")
            if weights is None:
                weights = np.repeat(1.0 / len(tickers), len(tickers))
            for _stale in ("va_series", "va_ref", "va_offset"):
                st.session_state.pop(_stale, None)
            st.session_state.update(
                va_universe=universe_text, va_index=index_ticker,
                va_start=start_date, va_end=end_date, va_tenor=tenor,
                va_conv=rv_convention, va_term=bool(with_term),
                va_cfg=dict(
                    tickers=tuple(tickers), weights=tuple(float(x) for x in weights),
                    index_ticker=index_ticker, start=start_date, end=end_date,
                    tenor=tenor, demean=rv_convention.startswith("Sample"),
                    with_term=bool(with_term),
                ),
            )
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))   # keep whatever was already loaded

if st.session_state.va_cfg is None:
    st.info("Set the underlyings above and press **Run**.")
    st.stop()

cfg = st.session_state.va_cfg

try:
    with st.spinner("Loading Bloomberg data…"):
        data = build_dataset(**cfg)
except Exception as exc:  # noqa: BLE001 — surface any loader failure, keep the panel usable
    st.error(f"Data load failed: {exc}")
    st.stop()

implied: pd.DataFrame = data["implied"]
realised: pd.DataFrame = data["realised"]
valid: list[str] = data["valid"]
w_valid: pd.Series = data["weights"]
horizon: int = data["horizon"]
tenor: str = cfg["tenor"]
is_basket = len(valid) > 1

if data["dropped"]:
    st.warning("Excluded, incomplete data: " + ", ".join(data["dropped"]))
if data["fallback"]:
    st.caption("IVOL_MONEYNESS fallback used for: " + ", ".join(data["fallback"]))

# ────────────────────────────────────────────────────────────────────────────
# DISPLAY CONTROLS  (live — they never refetch or reset the page)
# ────────────────────────────────────────────────────────────────────────────

names = list(implied.columns)
# Drop any stored selection that no longer exists, before the widgets render:
# Streamlit raises when a keyed value is absent from the option list.
_stored = st.session_state.get("va_series")
if _stored is not None and any(n not in names for n in _stored):
    st.session_state.pop("va_series", None)

d1, d2, d3, d4 = st.columns([3, 2, 2, 2])
with d1:
    selected = st.multiselect("Series", names, default=names[:min(4, len(names))], key="va_series")
_ref_options = selected or names
if st.session_state.get("va_ref") not in _ref_options:
    st.session_state.pop("va_ref", None)
with d2:
    reference = st.selectbox("Reference", _ref_options, key="va_ref")
with d3:
    basis = st.radio("Spread basis", list(SPREAD_BASES), horizontal=False, key="va_basis")
with d4:
    pct_label = st.radio("Percentile window", list(PCT_WINDOWS), horizontal=False, key="va_pct")

pct_window = PCT_WINDOWS[pct_label]
unit = SPREAD_BASES[basis]
spread_df = spread_frame(implied, realised, basis)

if not selected:
    st.warning("Pick at least one series above.")
    st.stop()

ref_spread = spread_df[reference].dropna()
if ref_spread.empty:
    st.warning(
        f"{reference} has no completed observation yet: the last {horizon} trading days "
        "of implied vol have no realised window to compare against."
    )
    st.stop()

latest_iv_series = implied[reference].dropna()
iv_matched, rv_matched, matched_date = matched_last(implied[reference], realised[reference])
current_spread = float(ref_spread.iloc[-1])
pct_all = percentile_of_last(ref_spread, None)
pct_win = percentile_of_last(ref_spread, pct_window)
z_win = zscore_of_last(ref_spread, pct_window)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Reference", reference)
k2.metric(f"ATM implied ({tenor})", _fmt(latest_iv_series.iloc[-1]),
          help=f"As of {latest_iv_series.index[-1]:%d %b %Y}.")
k3.metric("Realised, last matched", _fmt(rv_matched),
          help=("no completed window yet" if pd.isna(matched_date) else
                f"Window ending {matched_date:%d %b %Y}, against an implied of {_fmt(iv_matched)}."))
k4.metric(f"Spread ({unit})", _fmt(current_spread))
k5.metric(f"Percentile ({pct_label})", _fmt(pct_win, 0, "%"),
          delta=(f"{pct_win - pct_all:+.0f} vs all history"
                 if np.isfinite(pct_win) and np.isfinite(pct_all) else None))
k6.metric(f"Z-score ({pct_label})", _fmt(z_win))

st.caption(
    f"Spread = implied − subsequent realised, {unit}. Positive means the option market "
    f"charged more than the underlying went on to deliver, so a short-variance position "
    f"made money. The last {horizon} trading days are necessarily blank."
)

tabs = st.tabs([
    "Spread & percentile", "Levels & distribution", "Regression",
    "Carry P&L", "Term structure", "Dispersion",
])

# ─── Spread & percentile ────────────────────────────────────────────────────
with tabs[0]:
    fig = go.Figure()
    for name in selected:
        fig.add_trace(go.Scatter(x=spread_df.index, y=spread_df[name], mode="lines", name=name))
    for p, dash in ((10, "dot"), (25, "dot"), (50, "dash"), (75, "dot"), (90, "dot")):
        level = float(np.percentile(ref_spread, p))
        fig.add_hline(y=level, line_dash=dash, line_width=1, line_color=GREY,
                      annotation_text=f"P{p} = {level:.1f}", annotation_position="top left")
    fig.add_hline(y=0, line_color="#444444", line_width=1)
    fig.add_trace(go.Scatter(
        x=[ref_spread.index[-1]], y=[current_spread], mode="markers", name=f"{reference} latest",
        marker=dict(size=12, color=RED, line=dict(color="white", width=1.5)),
        hovertemplate=f"{current_spread:.2f} · P{pct_all:.0f} all history<extra></extra>"))
    _style(fig, f"{tenor} implied − subsequent realised ({unit})", unit.capitalize(), 560)
    st.plotly_chart(fig, use_container_width=True)

    rp = rolling_self_percentile(spread_df[reference], pct_window)
    pfig = go.Figure()
    pfig.add_trace(go.Scatter(x=rp.index, y=rp, mode="lines", name=reference,
                              line=dict(color=BLUE, width=2)))
    for level in (10, 25, 50, 75, 90):
        pfig.add_hline(y=level, line_dash="dot", line_width=1, line_color="#DDDDDD")
    _style(pfig, f"Rolling percentile of the {reference} spread ({pct_label} window)",
           "Percentile", 340)
    pfig.update_yaxes(range=[0, 100])
    st.plotly_chart(pfig, use_container_width=True)

    st.download_button("Download spread series (CSV)", spread_df.to_csv().encode(),
                       file_name=f"spread_{tenor}_{basis.lower()}.csv", key="dl_spread")

# ─── Levels & distribution ──────────────────────────────────────────────────
with tabs[1]:
    lf = go.Figure()
    for name in selected:
        lf.add_trace(go.Scatter(x=implied.index, y=implied[name], mode="lines",
                                name=f"{name} implied"))
        lf.add_trace(go.Scatter(x=realised.index, y=realised[name], mode="lines",
                                name=f"{name} realised", line=dict(dash="dot")))
    _style(lf, f"{tenor} implied vs subsequent realised", "Annualised vol (%)", 520)
    st.plotly_chart(lf, use_container_width=True)

    hf = go.Figure()
    hf.add_trace(go.Histogram(x=ref_spread, nbinsx=40, marker_color=BLUE, name=reference))
    hf.add_vline(x=current_spread, line_color=RED, line_width=3,
                 annotation_text=f"latest {current_spread:.2f} · P{pct_all:.0f}",
                 annotation_position="top")
    hf.add_vline(x=float(ref_spread.mean()), line_color=GREY, line_dash="dash", line_width=2,
                 annotation_text=f"mean {ref_spread.mean():.2f}", annotation_position="bottom")
    _style(hf, f"{reference}: distribution of the spread", "Frequency", 380)
    hf.update_layout(showlegend=False, hovermode="closest", xaxis_title=unit)
    st.plotly_chart(hf, use_container_width=True)

    q = ref_spread.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    st.dataframe(q.to_frame(unit).T.style.format("{:.2f}"), use_container_width=True)

# ─── Regression ─────────────────────────────────────────────────────────────
with tabs[2]:
    stats = regression_stats(implied[reference], realised[reference], horizon)
    if not stats:
        st.warning("Not enough overlapping observations for a regression (20 minimum).")
    else:
        st.caption(
            f"Realised = α + β · implied, {stats['first']:%b %Y} to {stats['last']:%b %Y}. "
            f"Standard errors are Newey-West with {horizon} lags: daily observations of a "
            f"{horizon}-day forward window overlap, so plain OLS errors would be roughly "
            f"√{horizon} ≈ {np.sqrt(horizon):.1f} times too small. The effective sample is "
            f"about {stats['n_independent']} independent windows, not {stats['n']} days."
        )
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("β (realised on implied)", _fmt(stats["slope"]),
                  help=f"HAC standard error {_fmt(stats['se_slope'])}.")
        c2.metric("t-stat vs β = 1", _fmt(stats["t_vs_one"]),
                  help="Below −2: implied significantly over-predicts realised, i.e. a "
                       "statistically robust variance risk premium.")
        c3.metric("α", _fmt(stats["intercept"]))
        c4.metric("R²", _fmt(stats["r2"]))
        c5.metric("Short-vol win rate", _fmt(100 * stats["hit_ratio"], 0, "%"),
                  help="Share of days where implied ended up above subsequent realised.")

        df_reg = pd.concat([implied[reference].rename("Implied"),
                            realised[reference].rename("Realised")], axis=1).dropna()
        sc = go.Figure()
        sc.add_trace(go.Scatter(
            x=df_reg["Implied"], y=df_reg["Realised"], mode="markers", name="Observations",
            marker=dict(size=5, opacity=0.45, color=BLUE),
            text=[d.strftime("%d %b %Y") for d in df_reg.index],
            hovertemplate="%{text}<br>implied %{x:.2f} → realised %{y:.2f}<extra></extra>"))
        xs = np.linspace(df_reg["Implied"].min(), df_reg["Implied"].max(), 100)
        sc.add_trace(go.Scatter(x=xs, y=stats["intercept"] + stats["slope"] * xs, mode="lines",
                                name=f"OLS: {stats['intercept']:.1f} + {stats['slope']:.2f}·x",
                                line=dict(color=RED, width=2)))
        sc.add_trace(go.Scatter(x=xs, y=xs, mode="lines", name="y = x (no premium)",
                                line=dict(dash="dot", color=GREY)))
        sc.add_trace(go.Scatter(
            x=[df_reg["Implied"].iloc[-1]], y=[df_reg["Realised"].iloc[-1]], mode="markers",
            name="Last matched", marker=dict(size=12, color=ORANGE,
                                             line=dict(color="white", width=1.5))))
        _style(sc, f"{reference}: realised vs implied ({tenor})", "Subsequent realised (%)", 540)
        sc.update_layout(hovermode="closest", xaxis_title="Implied at trade date (%)")
        st.plotly_chart(sc, use_container_width=True)

        st.caption(
            "Points below the dotted line are windows where the option market over-charged. "
            "A slope under 1 means the premium widens as implied rises."
        )

# ─── Carry ──────────────────────────────────────────────────────────────────
with tabs[3]:
    cc1, cc2 = st.columns([1, 3])
    with cc1:
        capped = st.checkbox("Capped variance", value=True, key="va_capped")
        cap_mult = st.number_input("Cap × strike", 1.0, 10.0, 2.5, 0.25, key="va_cap",
                                   disabled=not capped)
    with cc2:
        st.caption(
            "Short a variance swap struck at the implied vol of the day, hold to expiry, "
            "collect K² − R². Trades never overlap. With the cap on, the realised leg is "
            f"limited to ({cap_mult:g}·K)², the desk's usual local cap, which truncates the "
            "left tail exactly where an uncapped short would hurt most. Vega-equivalent P&L "
            "is the variance P&L divided by 2K, i.e. what the trade pays per unit of vega."
        )

    cap_arg = float(cap_mult) if capped else None
    schedules = carry_all_offsets(implied[reference], realised[reference], horizon, cap_arg)

    if schedules.empty:
        st.warning("Not enough completed windows to build a carry schedule.")
    else:
        best_row = schedules.loc[schedules["total_vega_pnl"].idxmax()]
        worst_row = schedules.loc[schedules["total_vega_pnl"].idxmin()]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Trades per schedule", int(schedules["trades"].median()))
        m2.metric("Median total P&L (vega pts)", _fmt(schedules["total_vega_pnl"].median()))
        m3.metric("Across start dates",
                  f"{worst_row['total_vega_pnl']:.1f} … {best_row['total_vega_pnl']:.1f}",
                  help="Total P&L of the worst and best of the "
                       f"{len(schedules)} non-overlapping schedules. A wide range means the "
                       "result depends on when you happened to start.")
        m4.metric("Median win rate", _fmt(100 * schedules["win_rate"].median(), 0, "%"))
        m5.metric("Median Sharpe", _fmt(schedules["sharpe"].median()))

        _offsets = [int(o) for o in schedules["offset"]]
        if st.session_state.get("va_offset") not in _offsets:
            st.session_state.pop("va_offset", None)
        offset = st.select_slider(
            "Schedule start offset (trading days)", options=_offsets, value=_offsets[0],
            key="va_offset",
            help="Every offset is a valid non-overlapping schedule. Slide it to see how "
                 "much the equity curve depends on the entry date.")
        trades = carry_schedule(implied[reference], realised[reference], horizon, cap_arg, offset)

        cf = go.Figure()
        cf.add_trace(go.Bar(x=trades.index, y=trades["vega_pnl"], name="Per trade (vega pts)",
                            marker_color=[BLUE if v >= 0 else RED for v in trades["vega_pnl"]]))
        cf.add_trace(go.Scatter(x=trades.index, y=trades["cum_vega_pnl"], mode="lines",
                                name="Cumulative", yaxis="y2",
                                line=dict(color="#00395D", width=2)))
        _style(cf, f"Short-variance carry — {reference} ({tenor}, offset {offset})",
               "Per-trade P&L (vega points)", 520)
        cf.update_layout(yaxis2=dict(title="Cumulative (vega points)", overlaying="y",
                                     side="right", showgrid=False))
        st.plotly_chart(cf, use_container_width=True)

        ddf = go.Figure()
        ddf.add_trace(go.Scatter(x=trades.index, y=trades["drawdown"], mode="lines",
                                 fill="tozeroy", name="Drawdown",
                                 line=dict(color=RED, width=1)))
        _style(ddf, "Drawdown of the cumulative carry", "Vega points", 280)
        st.plotly_chart(ddf, use_container_width=True)

        with st.expander("All schedules"):
            st.dataframe(
                schedules.set_index("offset").style.format({
                    "trades": "{:.0f}", "total_vega_pnl": "{:.1f}", "mean_vega_pnl": "{:.2f}",
                    "win_rate": "{:.0%}", "sharpe": "{:.2f}", "max_drawdown": "{:.1f}",
                }, na_rep="—"),
                use_container_width=True)

        st.download_button("Download trades (CSV)", trades.to_csv().encode(),
                           file_name=f"carry_{reference}_{tenor}.csv", key="dl_carry")

# ─── Term structure ─────────────────────────────────────────────────────────
with tabs[4]:
    if not data["term"]:
        st.info("Term structure was not loaded. Tick *Load all tenors* in the inputs and run again.")
    else:
        rows = []
        for label, frames in data["term"].items():
            iv_t, rv_t = frames["implied"], frames["realised"]
            if reference not in iv_t.columns:
                continue
            iv_series = iv_t[reference].dropna()
            if iv_series.empty:
                continue
            iv_m, rv_m, date_m = matched_last(iv_t[reference], rv_t[reference])
            rows.append({
                "Tenor": label,
                "Days": TENOR_CONFIG[label]["days"],
                "Implied now": float(iv_series.iloc[-1]),
                "Implied at match": iv_m,
                "Realised": rv_m,
                "Vol spread": iv_m - rv_m,
                "Var spread": iv_m ** 2 - rv_m ** 2,
                "Matched on": date_m,
            })
        if not rows:
            st.warning(f"No tenor has usable data for {reference}.")
        else:
            term_df = pd.DataFrame(rows).set_index("Tenor")
            st.caption(
                "*Implied now* is today's quote. The spreads compare implied and realised on "
                "the same date — the last one whose forward window has completed, shown in "
                "*Matched on* — so they are like-for-like rather than today against a stale "
                "realised. Longer tenors match further back by construction."
            )
            st.dataframe(
                term_df.style.format({
                    "Days": "{:.0f}", "Implied now": "{:.2f}", "Implied at match": "{:.2f}",
                    "Realised": "{:.2f}", "Vol spread": "{:+.2f}", "Var spread": "{:+.1f}",
                    "Matched on": lambda d: "—" if pd.isna(d) else f"{d:%d %b %Y}",
                }, na_rep="—"),
                use_container_width=True)

            tfig = go.Figure()
            tfig.add_trace(go.Scatter(x=term_df.index, y=term_df["Implied now"],
                                      mode="lines+markers", name="Implied now",
                                      line=dict(color=BLUE, width=2)))
            tfig.add_trace(go.Scatter(x=term_df.index, y=term_df["Realised"],
                                      mode="lines+markers", name="Realised (matched)",
                                      line=dict(color=ORANGE, width=2, dash="dot")))
            _style(tfig, f"{reference}: term structure", "Annualised vol (%)", 420)
            tfig.update_layout(hovermode="x")
            st.plotly_chart(tfig, use_container_width=True)

            sfig = go.Figure()
            sfig.add_trace(go.Bar(x=term_df.index, y=term_df["Vol spread"],
                                  marker_color=[BLUE if v >= 0 else RED
                                                for v in term_df["Vol spread"]],
                                  name="Vol spread"))
            sfig.add_hline(y=0, line_color="#444444", line_width=1)
            _style(sfig, "Premium by tenor (matched dates)", "Vol points", 320)
            sfig.update_layout(showlegend=False, hovermode="x")
            st.plotly_chart(sfig, use_container_width=True)

# ─── Dispersion ─────────────────────────────────────────────────────────────
with tabs[5]:
    if not is_basket:
        st.info("Dispersion needs at least two underlyings.")
    elif not data["index_ticker"]:
        st.info(
            "Add an index ticker in the inputs to extract implied correlation. "
            + ("The one given had no usable price or implied-vol history."
               if cfg["index_ticker"] else "")
        )
    else:
        index_ticker = data["index_ticker"]
        iv_names = data["implied_names"]
        rv_names = data["realised_names"]
        idx_iv = data["index_implied"]
        idx_rv = data["index_realised"]

        rho_imp = implied_correlation(iv_names, w_valid, idx_iv)
        rho_real = implied_correlation(rv_names, w_valid, idx_rv)
        rho_df = pd.concat([rho_imp.rename("Implied ρ"), rho_real.rename("Realised ρ")], axis=1)

        rho_i_last = rho_imp.dropna().iloc[-1] if not rho_imp.dropna().empty else np.nan
        rho_i_m, rho_r_m, rho_date = matched_last(rho_imp, rho_real)

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Implied ρ now", _fmt(rho_i_last))
        r2.metric("Realised ρ (matched)", _fmt(rho_r_m),
                  help="—" if pd.isna(rho_date) else f"Window ending {rho_date:%d %b %Y}.")
        r3.metric("Correlation premium", _fmt(rho_i_m - rho_r_m, 2),
                  help="Implied minus subsequent realised on the same date — what a short "
                       "correlation, long dispersion position earned.")
        r4.metric("ρ percentile (all)", _fmt(percentile_of_last(rho_imp, None), 0, "%"))

        rf = go.Figure()
        rf.add_trace(go.Scatter(x=rho_df.index, y=rho_df["Implied ρ"], mode="lines",
                                name="Implied ρ", line=dict(color=BLUE, width=2)))
        rf.add_trace(go.Scatter(x=rho_df.index, y=rho_df["Realised ρ"], mode="lines",
                                name="Realised ρ (subsequent)",
                                line=dict(color=ORANGE, width=2, dash="dot")))
        for level in (0.0, 1.0):
            rf.add_hline(y=level, line_color=GREY, line_width=1)
        _style(rf, f"Implied vs realised correlation — {index_ticker} against the basket",
               "Correlation", 500)
        st.plotly_chart(rf, use_container_width=True)

        # Dispersion carry: long single-name variance, short index variance
        names_var_spread = weighted_avg(iv_names ** 2 - rv_names ** 2, w_valid)
        index_var_spread = idx_iv ** 2 - idx_rv ** 2
        disp = (names_var_spread - index_var_spread).dropna()
        if not disp.empty:
            st.subheader("Dispersion carry")
            st.caption(
                "Weighted single-name variance spread minus the index variance spread — the "
                "P&L of being long single-name variance and short index variance, in variance "
                "points, before cap and fees. Positive means dispersion paid."
            )
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("Latest", _fmt(disp.iloc[-1], 1))
            dc2.metric("Mean", _fmt(disp.mean(), 1))
            dc3.metric("Share positive", _fmt(100 * (disp > 0).mean(), 0, "%"))

            dfig = go.Figure()
            dfig.add_trace(go.Scatter(x=disp.index, y=disp, mode="lines", name="Dispersion carry",
                                      line=dict(color=BLUE, width=1.6), fill="tozeroy",
                                      fillcolor="rgba(0,174,239,0.15)"))
            dfig.add_hline(y=0, line_color="#444444", line_width=1)
            dfig.add_hline(y=float(disp.mean()), line_dash="dash", line_color=GREY, line_width=1,
                           annotation_text=f"mean {disp.mean():.1f}")
            _style(dfig, "Single-name minus index variance spread", "Variance points", 420)
            st.plotly_chart(dfig, use_container_width=True)

        st.subheader("Single-name ranking")
        sn_rows = []
        for t in valid:
            iv_s, rv_s = iv_names[t], rv_names[t]
            sp = (iv_s ** 2 - rv_s ** 2).dropna()
            iv_clean = iv_s.dropna()
            if sp.empty or iv_clean.empty:
                continue
            iv_m, rv_m, _ = matched_last(iv_s, rv_s)
            sn_rows.append({
                "Ticker": t,
                "Weight": float(w_valid[t]),
                "Implied now": float(iv_clean.iloc[-1]),
                "Implied at match": iv_m,
                "Realised": rv_m,
                "Var spread": float(sp.iloc[-1]),
                "Percentile": percentile_of_last(sp, None),
                "Contribution": float(w_valid[t] * sp.iloc[-1]),
            })
        if not sn_rows:
            st.info("No single name has a completed observation yet.")
        else:
            sn_df = pd.DataFrame(sn_rows).set_index("Ticker").sort_values(
                "Contribution", ascending=False)
            st.caption(
                "*Contribution* is the weight times the variance spread: what each name adds "
                "to the basket leg of the dispersion trade."
            )
            st.dataframe(
                sn_df.style.format({
                    "Weight": "{:.2%}", "Implied now": "{:.2f}", "Implied at match": "{:.2f}",
                    "Realised": "{:.2f}", "Var spread": "{:+.1f}", "Percentile": "{:.0f}%",
                    "Contribution": "{:+.1f}",
                }, na_rep="—"),
                use_container_width=True)
            st.download_button("Download single names (CSV)", sn_df.to_csv().encode(),
                               file_name=f"single_names_{tenor}.csv", key="dl_names")

# ────────────────────────────────────────────────────────────────────────────
# SUMMARY & METHODOLOGY
# ────────────────────────────────────────────────────────────────────────────

with st.expander("Summary table"):
    rows = []
    for name in names:
        sp = spread_df[name].dropna()
        iv = implied[name].dropna()
        if sp.empty or iv.empty:
            continue
        rows.append({
            "Name": name,
            "Implied now": float(iv.iloc[-1]),
            f"Spread ({unit})": float(sp.iloc[-1]),
            "Mean spread": float(sp.mean()),
            "P (all)": percentile_of_last(sp, None),
            f"P ({pct_label})": percentile_of_last(sp, pct_window),
            f"Z ({pct_label})": zscore_of_last(sp, pct_window),
            "N": int(len(sp)),
        })
    if rows:
        summary = pd.DataFrame(rows).set_index("Name")
        st.dataframe(
            summary.style.format({
                "Implied now": "{:.2f}", f"Spread ({unit})": "{:+.2f}", "Mean spread": "{:+.2f}",
                "P (all)": "{:.0f}%", f"P ({pct_label})": "{:.0f}%",
                f"Z ({pct_label})": "{:+.2f}", "N": "{:.0f}",
            }, na_rep="—"),
            use_container_width=True)
        st.download_button("Download summary (CSV)", summary.to_csv().encode(),
                           file_name=f"summary_{tenor}.csv", key="dl_summary")

if is_basket:
    with st.expander("Basket weights"):
        st.dataframe(w_valid.rename("Weight").to_frame().style.format({"Weight": "{:.2%}"}),
                     use_container_width=True)

with st.expander("Methodology & assumptions"):
    st.markdown(f"""
**Implied vol** — constant-maturity 100% moneyness Bloomberg field
(`{TENOR_CONFIG[tenor]['field']}`), with `IVOL_MONEYNESS` at 100% as an automatic
per-name fallback. Names that resolve through the fallback are listed at the top of the
page.

**Realised vol** — annualised over the **forward** {horizon}-trading-day window, so the
value carried at a date is what an option struck that day went on to face.
{'Sample standard deviation of log returns (demeaned).' if cfg['demean'] else
 'Variance-swap convention: √(mean of squared log returns), no mean subtraction, which is what a variance swap settles on.'}
Annualisation by √{TRADING_DAYS}. The last {horizon} trading days have no completed
window and are blank everywhere.

**Spread bases** — *Vol* is implied − realised in vol points, the intuitive one.
*Variance* is implied² − realised², the native unit of a variance swap. *Vega-equivalent*
is the variance spread divided by 2·K, i.e. the variance P&L expressed per unit of vega,
which is how the payoff is usually quoted.

**Basket** — `{BASKET}` is a weighted **average of single-name vols**, renormalised each
day on the names that have data. It is not a correlation-adjusted basket volatility; for
that, use the Dispersion tab with an index ticker.

**Regression** — realised on implied with Newey-West standard errors at {horizon} lags.
Daily observations of a {horizon}-day forward window overlap almost completely, so plain
OLS standard errors overstate precision by roughly √{horizon}. The reported t-stat tests
β = 1, which is the hypothesis that implied is an unbiased forecast; a significantly
negative t-stat is the variance risk premium.

**Carry** — non-overlapping short-variance trades. Because the schedule depends on which
day you start, every one of the {horizon} possible offsets is computed and the header
shows the median and the full range across them. With the cap on, the realised leg is
capped at (cap·K)², matching a capped variance swap. Vega-equivalent P&L is the variance
P&L over 2·K.

**Implied correlation** — ρ = (σ_idx² − Σ wᵢ²σᵢ²) / ((Σ wᵢσᵢ)² − Σ wᵢ²σᵢ²), left
unclipped so that a basket which does not replicate the index shows up as ρ outside
[0, 1] rather than being silently squashed. Realised ρ applies the same formula to the
subsequent realised vols.

**Percentiles** — inclusive: the share of observations at or below the latest one, over
the window chosen at the top of the page.
""")
