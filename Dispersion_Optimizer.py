import streamlit as st
# Dev hot-reload: clear cached dispersion modules so code changes apply on rerun
import sys
_stale = [k for k in sys.modules if k.startswith("functions.dispersion")]
for _k in _stale:
    del sys.modules[_k]
from functions.dispersion import optimize, backtest, DispersionConfig, OptimizationConstraints, price, solve
from functions.dispersion.models import ProductType, MissingDataPolicy, OptimizationResult, PriceResult
from functions.dispersion._portal import payment_dates as calculate_payment_dates, observation_schedule
import functions.dispersion._charts as func_graph
# ── Inlined from _optimizer.py (4 lines) ──
_log_callback = None

def set_optimization_logger(callback=None):
    global _log_callback
    _log_callback = callback

def _parse_policy(s: str) -> MissingDataPolicy:
    if s == "Adaptive Reweight":
        return MissingDataPolicy.ADAPTIVE_REWEIGHT
    elif s == "Fill Zero (Gaia_PP)":
        return MissingDataPolicy.FILL_ZERO
    return MissingDataPolicy.DROP_INCOMPLETE_DAYS

def _convert_bt_df_standard(df):
    """edited_df_bis → backtest() input format.
    
    Vol swap input schema:
        Underlying | Strike (%) | Weight (%)
    
    Corridor input schema:
        Variance Asset | Strike Mono Var Swap (%) | Weight (%)
    
    Weight can be positive or negative:
        Positive = long
        Negative = short
    
    Returns:
        Variance Asset | Strike Mono Var Swap (%) | Weight (%) | Side
    """
    df = df.copy()
    # Vol swap UI uses product-specific labels — map to API canonical names
    if 'Underlying' in df.columns:
        df.rename(columns={'Underlying': 'Variance Asset', 'Strike (%)': 'Strike Mono Var Swap (%)'}, inplace=True)
    rows = []
    for _, r in df.iterrows():
        w = float(r['Weight (%)'])
        rows.append({
            'Variance Asset': str(r['Variance Asset']).strip(),
            'Strike Mono Var Swap (%)': float(r['Strike Mono Var Swap (%)']),
            'Weight (%)': abs(w),
            'Side': 'long' if w > 0 else 'short',
        })
    return pd.DataFrame(rows)

def _convert_bt_df_cross(df):
    """edited_df_cross → backtest() input format.
    
    Input schema:
        Variance Asset | Corridor Condition Asset | Strike Cross Corridor (%) | Strike Mono Var Swap (%) | Weight (%)
    
    Weight can be positive or negative:
        Positive = long
        Negative = short
    
    Returns:
        Variance Asset | Corridor Condition Asset | Strike Cross Corridor (%) | Strike Mono Var Swap (%) | Weight (%) | Side
    """
    df = df.copy()
    rows = []
    for idx, r in df.iterrows():
        w = float(r['Weight (%)'])
        row = {
            'Variance Asset': str(r['Variance Asset']).strip(),
            'Corridor Condition Asset': str(r['Corridor Condition Asset']).strip(),
            'Strike Cross Corridor (%)': float(r['Strike Cross Corridor (%)']),
            'Strike Mono Var Swap (%)': float(r['Strike Mono Var Swap (%)']),
            'Weight (%)': abs(w),
            'Side': 'long' if w > 0 else 'short',
        }
        rows.append(row)
    result = pd.DataFrame(rows)
    return result

def _run_bt(src_df, is_vol_swap, n_exp, local_floor, local_cap,
            global_floor, global_cap, ubar, dbar, adj_divs="No",
            start_date=None, is_cross_corridor=False, missing_data_policy="Adaptive Reweight"):
    """Run backtest via new API, return (timeseries_df, metadata_dict, figure, None)."""
    cfg = DispersionConfig(
        product_type=ProductType.VOL_SWAP if is_vol_swap else ProductType.VAR_SWAP_CORRIDOR,
        cross_corridor=is_cross_corridor,
        n_exp=n_exp,
        barrier_up=ubar,
        barrier_down=dbar,
        local_cap=local_cap,
        adj_divs=(adj_divs == "Yes"),
        missing_data_policy=_parse_policy(missing_data_policy) if isinstance(missing_data_policy,
                                                                             str) else missing_data_policy,
        global_cap=global_cap,
        global_floor=global_floor,
    )
    bt_input = _convert_bt_df_cross(src_df) if is_cross_corridor else _convert_bt_df_standard(src_df)
    bt_result = backtest(bt_input, cfg, start_date=start_date)
    fig = func_graph.plot_main_backtest(bt_result.timeseries,
                                        active_count=bt_result.active_legs_count,
                                        is_cross_corridor=is_cross_corridor)
    # Build metadata dict for downstream display code
    if is_cross_corridor:
        all_tickers = src_df['Variance Asset'].astype(str).str.strip().tolist()
        all_weights = src_df['Weight (%)'].astype(float).tolist()
    else:
        all_tickers = src_df['Variance Asset'].astype(str).str.strip().tolist()
        all_weights = src_df['Weight (%)'].astype(float).tolist()
    metadata = {
        'long_tickers': [t for t, w in zip(all_tickers, all_weights) if w > 0],
        'short_tickers': [t for t, w in zip(all_tickers, all_weights) if w < 0],
        'long_weights': [abs(w) for w in all_weights if w > 0],
        'short_weights': [abs(w) for w in all_weights if w < 0],
        'raw_data': bt_result.per_leg_pnl if bt_result.per_leg_pnl is not None else pd.DataFrame(),
        'unweighted_data': bt_result.per_leg_pnl if bt_result.per_leg_pnl is not None else pd.DataFrame(),
    }
    return bt_result.timeseries, metadata, fig, None

_SHORT_DISPLAY = {'.SPX': 'SPX', '.STOXX50E': 'SX5E', '.SX7E': 'SX7E', '.SX8E': 'SX8E'}

def _get_short_leg_display(tickers):
    return ', '.join(_SHORT_DISPLAY.get(t, t) for t in tickers)

from functions.common.utils import *
from functions.dispersion.json_booking import (
    LEGS_COLS as JB_LEGS_COLS,
    EDITABLE_DEFAULTS as JB_EDITABLE_DEFAULTS,
    LOCKED_DEFAULTS as JB_LOCKED_DEFAULTS,
    df_from_paste as jb_df_from_paste,
    to_float as jb_to_float,
    parse_date as jb_parse_date,
    generate_booking_json,
)
import os
import io
import time
import cProfile
import pstats
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xbbg import blp
from datetime import date
from dateutil.relativedelta import relativedelta
import pretty_html_table as pht

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Performance: Bloomberg data caching + numba warmup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_data(ttl=300, show_spinner=False)
def _fetch_bloomberg_prices(tickers_key: str, tickers: list, field: str, start_date: str):
    """Cache Bloomberg price fetches for 5 minutes. Avoids re-fetching on every rerun."""
    today_str = date.today().strftime("%m/%d/%Y")
    df = blp.bdh(tickers, field, start_date, today_str)
    if df is not None and not df.empty and isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

@st.cache_data(ttl=600, show_spinner=False)
def _fetch_sector_data(tickers_key: str, tickers: list):
    """Cache Bloomberg sector lookups for 10 minutes."""
    df = blp.bdp(tickers, "INDUSTRY_SECTOR")
    return df

@st.cache_resource
def _warmup_numba():
    """Trigger numba JIT compilation on app load (runs once per session)."""
    try:
        dummy = np.array([100.0, 101.0, 100.5, 102.0, 101.5])
        _vol_swap_window(dummy, 0.2, 4, 2.5)
        _corridor_varswap_window(dummy, dummy, 0.2, 1.3, 0.7, 4, 2.5)
    except Exception:
        pass
    return True

def _render_optimization_result(result, is_cross, debug_info=None):
    """Render OptimizationResult directly in Streamlit UI."""
    if result.score <= -1e9:
        st.error("Optimization found no valid solution.")
        # Debug expander to show rejection reasons
        if debug_info:
            with st.expander("🔍 Debug Info (Rejection Reasons)", expanded=True):
                st.code(debug_info, language="text")
        return
    st.subheader("📊 Optimized Portfolio")
    # Build portfolio rows from OptimizationResult fields
    # For cross-corridor, long_cross_corridor = [(variance_asset, corridor_cond_asset, cross_strike), ...]
    # Key by corridor_cond_asset (idx) since that's the unique key in long_basket
    xcorr_map = {idx: (t, xcs) for t, idx, xcs in result.long_cross_corridor} if result.long_cross_corridor else {}
    xcorr_map_s = {idx: (t, xcs) for t, idx, xcs in result.short_cross_corridor} if result.short_cross_corridor else {}
    strike_map = dict(result.long_strikes + result.short_strikes) if result.long_strikes else {}
    portfolio_rows = []
    for ticker, weight in result.long_basket:
        row = {
            'Stock': ticker,
            'Side': 'Long',
            'Strike %': f"{strike_map.get(ticker, 0) * 100:.1f}",
            'Weight %': f"{abs(weight) * 100:.2f}",
        }
        if is_cross and ticker in xcorr_map:
            row['Variance Asset'] = xcorr_map[ticker][0]  # SPX Index
            row['Cross Corr Index'] = ticker  # Corridor Condition Asset IS the key
            row['Cross Corr Strike %'] = f"{xcorr_map[ticker][1] * 100:.1f}"
        portfolio_rows.append(row)
    for ticker, weight in result.short_basket:
        row = {
            'Stock': ticker,
            'Side': 'Short',
            'Strike %': f"{strike_map.get(ticker, 0) * 100:.1f}",
            'Weight %': f"{abs(weight) * 100:.2f}",
        }
        if is_cross and ticker in xcorr_map_s:
            row['Variance Asset'] = xcorr_map_s[ticker][0]
            row['Cross Corr Index'] = ticker
            row['Cross Corr Strike %'] = f"{xcorr_map_s[ticker][1] * 100:.1f}"
        portfolio_rows.append(row)
    if is_cross:
        bt_rows = []
        for row in portfolio_rows:
            w = float(row.get('Weight %', '0'))
            side = row.get('Side', 'Long')
            bt_rows.append({
                'Variance Asset': row.get('Variance Asset', row.get('Stock', '')),
                'Corridor Condition Asset': row.get('Cross Corr Index', row.get('Stock', '')),
                'Strike Cross Corridor (%)': row.get('Cross Corr Strike %', '0'),
                'Strike Mono Var Swap (%)': row.get('Strike %', '0'),
                'Weight (%)': f"{w * (-1 if side == 'Short' else 1):.4f}",
            })
        bt_df = pd.DataFrame(bt_rows)
    else:
        bt_rows = []
        for row in portfolio_rows:
            w = float(row.get('Weight %', '0'))
            side = row.get('Side', 'Long')
            bt_rows.append({
                'Variance Asset': row['Stock'],
                'Strike Mono Var Swap (%)': row['Strike %'],
                'Weight (%)': f"{w * (-1 if side == 'Short' else 1):.4f}",
            })
        bt_df = pd.DataFrame(bt_rows)
    st.dataframe(bt_df, use_container_width=True)
    # Summary metrics
    long_total = sum(abs(w) for _, w in result.long_basket) * 100
    short_total = sum(abs(w) for _, w in result.short_basket) * 100
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Score", f"{result.score:.4f}")
    col_s2.metric("Net Strike", f"{result.net_strike * 100:.2f}%")
    col_s3.metric("Long Weight", f"{long_total:.1f}%")
    col_s4.metric("Short Weight", f"{short_total:.1f}%")
    st.caption(f"Generations: {result.generations_run} | Converged: {result.converged}")
    # Scoring signature — reproducibility fingerprint of the run
    if getattr(result, 'scoring_signature', None):
        with st.expander("🧾 Scoring signature (debug)", expanded=False):
            st.markdown(
                f"- **Signature:** `{result.scoring_signature}`\n"
                f"- **Seed:** `{result.seed}`\n"
                f"- **Reference sample size:** `{result.reference_size}` baskets\n"
                f"- **Scoring mode:** `{result.scoring_mode}`")
            st.caption(
                "Same signature = same objective (active metrics + weights), same seed "
                "and same reference geometry — scores are directly comparable.")
    # Show backtest if available
    if result.backtest and result.backtest.timeseries is not None and not result.backtest.timeseries.empty:
        with st.expander("📈 Backtest", expanded=True):
            ts = result.backtest.timeseries
            fig = func_graph.plot_main_backtest(
                ts,
                active_count=result.backtest.active_legs_count,
                is_cross_corridor=is_cross,
            )
            # Overlay smoothed curve if available from interactive post-smoothing
            _sr = st.session_state.get('gaia_smooth_result')
            if _sr and _sr.get("bt_sm") and _sr["bt_sm"].timeseries is not None:
                import plotly.graph_objects as _go
                _sm_ts = _sr["bt_sm"].timeseries
                fig.data[0].name = "Optimal (w*)"
                fig.add_trace(_go.Scatter(
                    x=_sm_ts.index, y=_sm_ts["Result"],
                    name=f"Smoothed (ε={_sr['eps']:.2f})",
                    line=dict(color='#00897b', width=2.5),
                    mode='lines',
                ))
            st.plotly_chart(fig, use_container_width=True, key="plotly_opti_bt")
            # Validation summary
            _res = ts['Result']
            _valid = _res.dropna()
            if not _valid.empty:
                _first_nz = _valid[_valid != 0].index.min() if (_valid != 0).any() else None
                _ac = result.backtest.active_legs_count
                _ac_info = f"Active stocks: {int(_ac.min())}–{int(_ac.max())}" if _ac is not None and len(
                    _ac) > 0 else ""
                _hit = float((_valid[_valid != 0] > 0).mean()) if (_valid != 0).any() else 0.0
                st.caption(
                    f"**{_valid.index.min().strftime('%Y-%m-%d')} → {_valid.index.max().strftime('%Y-%m-%d')}** | "
                    f"First PnL: {_first_nz.strftime('%Y-%m-%d') if _first_nz is not None else 'N/A'} | "
                    f"Min: {_valid.min():.4f} | Max: {_valid.max():.4f} | Last: {_valid.iloc[-1]:.4f} | "
                    f"Obs: {len(_valid)} | Flat: {int((_valid == 0).sum())} | Hit: {_hit:.4f} | {_ac_info}"
                )
        # ── Runtime Validation: prove optimizer backtest matches standalone path ──
        with st.expander("🔬 Backtest Input/Output Validation", expanded=False):
            # A) Displayed portfolio table
            st.markdown("**A) Displayed Portfolio Table**")
            _a_rows = len(result.long_basket) + len(result.short_basket)
            _a_wsum = sum(abs(w) for _, w in result.long_basket) + sum(abs(w) for _, w in result.short_basket)
            st.markdown(f"- Rows: **{_a_rows}** | Weight sum: **{_a_wsum * 100:.2f}%**")
            if is_cross and result.long_cross_corridor:
                _var_assets = set(t for t, _, _ in result.long_cross_corridor)
                _corr_assets = set(idx for _, idx, _ in result.long_cross_corridor)
                st.markdown(
                    f"- Variance Assets: `{_var_assets}` | Corridor Condition Assets: **{len(_corr_assets)}** unique")
            # B) What was passed to backtest (run_from_optimization uses same long_basket/short_basket)
            st.markdown("**B) Backtest Input (run_from_optimization)**")
            _bt_keys_long = [k for k, _ in result.long_basket]
            _bt_keys_short = [k for k, _ in result.short_basket]
            _bt_weights = {k: w for k, w in result.long_basket}
            _bt_weights.update({k: -w for k, w in result.short_basket})
            st.markdown(
                f"- Long keys ({len(_bt_keys_long)}): `{_bt_keys_long[:10]}{'...' if len(_bt_keys_long) > 10 else ''}`")
            st.markdown(
                f"- Short keys ({len(_bt_keys_short)}): `{_bt_keys_short[:5]}{'...' if len(_bt_keys_short) > 5 else ''}`")
            st.markdown(
                f"- Weights dict entries: **{len(_bt_weights)}** | sum(abs): **{sum(abs(v) for v in _bt_weights.values()) * 100:.2f}%**")
            # Confirm display table matches backtest input
            _bt_df_keys = set(bt_df[
                                  'Corridor Condition Asset'].values) if is_cross and 'Corridor Condition Asset' in bt_df.columns else set(
                bt_df['Variance Asset'].values) if 'Variance Asset' in bt_df.columns else set()
            _bt_input_keys = set(_bt_keys_long + _bt_keys_short)
            _keys_match = _bt_df_keys == _bt_input_keys
            st.markdown(f"- Display table keys == Backtest input keys: **{'✅ YES' if _keys_match else '❌ NO'}**")
            if not _keys_match:
                st.warning(
                    f"Mismatch! Display: {_bt_df_keys - _bt_input_keys} | Backtest: {_bt_input_keys - _bt_df_keys}")
            # C) Backtest output
            st.markdown("**C) Backtest Result**")
            _ts = result.backtest.timeseries
            _ac = result.backtest.active_legs_count
            st.markdown(f"- timeseries.shape: **{_ts.shape}** | columns: `{list(_ts.columns)}`")
            st.markdown(f"- index: {_ts.index.min().strftime('%Y-%m-%d')} → {_ts.index.max().strftime('%Y-%m-%d')}")
            _r = _ts['Result']
            st.markdown(f"- Result: min={_r.min():.4f} | max={_r.max():.4f} | last={_r.iloc[-1]:.4f}")
            st.markdown(f"- First 5: `{[f'{v:.4f}' for v in _r.iloc[:5].values]}`")
            st.markdown(f"- Last 5: `{[f'{v:.4f}' for v in _r.iloc[-5:].values]}`")
            if _ac is not None and len(_ac) > 0:
                st.markdown(
                    f"- active_legs_count: min={int(_ac.min())} | max={int(_ac.max())} | last={int(_ac.iloc[-1])}")
            st.markdown(f"- Chart function: `func_graph.plot_main_backtest` ✅ (same as standalone)")

def streamlit_log_callback(log_entry):
    """Handle logs from optimization functions"""
    log_type, message = log_entry
    if log_type == "INFO":
        st.info(message)
    elif log_type == "WRITE":
        st.write(message)
    elif log_type == "ERROR":
        st.error(message)
    elif log_type == "SUCCESS":
        st.success(message)
    elif log_type == "WARNING":
        st.warning(message)

# Set up logger
set_optimization_logger(streamlit_log_callback)
try:
    from functions.dispersion._portal import get_trading_calendar
    from functions.dispersion._portal import observation_schedule as create_schedule
except ImportError:
    create_schedule = None
    get_trading_calendar = None
st.set_page_config(
    page_title="Gaia - Dispersion",
    page_icon="sparkles",
    layout="wide",
)
# Trigger numba warmup after set_page_config
_warmup_numba()
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if 'display_criteria' not in st.session_state:
    st.session_state['display_criteria'] = False
if 'graph_backtest' not in st.session_state:
    st.session_state["graph_backtest"] = None
if 'opt_window' not in st.session_state:
    st.session_state['opt_window'] = '5y'
if 'data_opti' not in st.session_state:
    st.session_state["data_opti"] = None
if 'input_df' not in st.session_state:
    st.session_state['input_df'] = False
st.header("♻️Risk Recycling - Genetic Algorithm Long/Short Strategy♻️")
tab1, tab3, tab4, tab5 = st.tabs([
    "Use optimizer",
    "Basket Backtesting",
    "Pricing & FPF",
    "Json Booking Generation"
])
if 'display_criteria' not in st.session_state:
    st.session_state.display_criteria = False
with tab1:
    st.info("🧬 **Genetic Algorithm**: Optimizes separate long and short baskets using evolutionary optimization")
    # Basic parameters
    c4, c5, c6, c7, c8 = st.columns(5)
    with c4:
        # Add toggle for input mode
        n_exp_mode = st.selectbox(
            'N Expected Input Mode',
            ['Days', 'Date'],
            key='n_exp_mode_tab1'
        )
        if n_exp_mode == 'Days':
            n_exp = st.number_input('Input N expected in days', value=310, key=41050)
        else:
            maturity_date = st.date_input(
                'Select Maturity Date',
                value=datetime.date.today() + datetime.timedelta(days=310),
                format="DD/MM/YYYY",
                key='maturity_date_tab1'
            )
            # Display calculated days (will be computed later when tickers are available)
            st.info("Days will be calculated based on tickers")
            n_exp = None  # Placeholder, will be calculated later
    with c5:
        local_cap = st.number_input("Input local cap", value=2.50, key=12001)
    with c6:
        local_floor = st.number_input("Input local floor", value=-9999999, key=1200)
    with c7:
        global_cap = st.number_input("Input global cap", value=999999999, key=12005)
    with c8:
        global_floor = st.number_input("Input global floor", value=-9999999, key=120045)
    st.divider()
    optimization_start = st.date_input(
        "Optimization Start Date",
        value=date.today() - relativedelta(years=5),
        key="opt_start_date",
        format="DD/MM/YYYY"
    )
    optimization_end = date.today()
    st.divider()
    # Swap parameters
    c9, c10, c11, c12 = st.columns(4)
    with c9:
        is_vol_swap = st.selectbox("Volswap or Varswap", [True, False],
                                   format_func=lambda x: ("Volswap", "Varswap")[-x + 1], key=2)
    with c10:
        if is_vol_swap == False:
            swap_type = st.selectbox("Swap Type",
                                     ["Standard Corridor", "Cross Corridor"],
                                     help="Standard: same underlying for corridor and variance. Cross: different underlyings",
                                     key=2541)
            is_corr = swap_type in ["Standard Corridor", "Cross Corridor"]
            is_cross_corridor = swap_type == "Cross Corridor"
        else:
            is_corr = False
            is_cross_corridor = False
    with c11:
        if is_vol_swap == False and is_corr == True:
            dbar = st.number_input("Input Down Var", value=70.00, step=0.01, format="%.2f", key=4545) / 100
        else:
            dbar = -9999999
    with c12:
        if is_vol_swap == False and is_corr == True:
            ubar = st.number_input("Input Up Var", value=130.00, step=0.01, format="%.2f", key=433433) / 100
        else:
            ubar = 99999999
    st.divider()
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("📈 Long Basket Candidates")
        if is_cross_corridor:
            if 'long_df_cross' not in st.session_state:
                st.session_state.long_df_cross = pd.DataFrame({
                    'Variance Asset': [],
                    'Corridor Condition Asset': [],
                    'Strike Cross Corridor (%)': [],
                    'Strike Mono Var Swap (%)': [],
                    'Min Weight': [],
                    'Max Weight': []
                })
            st.data_editor(
                st.session_state.long_df_cross[
                    ['Variance Asset', 'Corridor Condition Asset', 'Strike Cross Corridor (%)',
                     'Strike Mono Var Swap (%)', 'Min Weight', 'Max Weight']],
                num_rows="dynamic",
                use_container_width=True,
                key="long_editor_cross",
                column_order=['Variance Asset', 'Corridor Condition Asset', 'Strike Cross Corridor (%)',
                              'Strike Mono Var Swap (%)', 'Min Weight', 'Max Weight'],
            )
            long_pasted_text = st.text_area(
                "Paste long basket data here:",
                height=100,
                key="long_paste_cross",
                help="Format: Variance Asset, Corridor Condition Asset, Strike Cross Corridor (%), Strike Mono Var Swap (%), Min Weight, Max Weight"
            )
            if st.button("Fill Long Basket", key="fill_long_cross"):
                if long_pasted_text:
                    try:
                        if '\n' not in long_pasted_text:
                            values = long_pasted_text.split(
                                '\t') if '\t' in long_pasted_text else long_pasted_text.split(',')
                            row_data = {}
                            for i, col in enumerate(
                                    ['Variance Asset', 'Corridor Condition Asset', 'Strike Cross Corridor (%)',
                                     'Strike Mono Var Swap (%)', 'Min Weight', 'Max Weight']):
                                row_data[col] = [values[i] if i < len(values) else '']
                            st.session_state.long_df_cross = pd.DataFrame(row_data)
                        else:
                            df = pd.read_csv(io.StringIO(long_pasted_text),
                                             sep='\t' if '\t' in long_pasted_text else ',', header=None)
                            # Detect and skip header row if first cell looks like a column name
                            first_cell = str(df.iloc[0, 0]).strip().lower()
                            header_candidates = ['tickers', 'variance asset', 'index', 'strike', 'min weight',
                                                 'max weight']
                            if any(c in first_cell for c in header_candidates):
                                df = df.iloc[1:].reset_index(drop=True)
                                st.toast("⚠️ Header row detected and skipped", icon="ℹ️")
                            new_data = pd.DataFrame()
                            for i, col in enumerate(
                                    ['Variance Asset', 'Corridor Condition Asset', 'Strike Cross Corridor (%)',
                                     'Strike Mono Var Swap (%)', 'Min Weight', 'Max Weight']):
                                new_data[col] = df.iloc[:, i] if i < df.shape[1] else [''] * len(df)
                            st.session_state.long_df_cross = new_data
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing long basket data: {e}")
        else:
            # STANDARD INTERFACE (vol swap or corridor)
            if is_vol_swap:
                _opt_std_cols = ['Underlying', 'Strike (%)', 'Min Weight', 'Max Weight']
            else:
                _opt_std_cols = ['Variance Asset', 'Strike Mono Var Swap (%)', 'Min Weight', 'Max Weight']
            # Reset if columns changed (product type switch)
            if 'long_df' not in st.session_state or list(st.session_state.long_df.columns) != _opt_std_cols:
                st.session_state.long_df = pd.DataFrame({
                    _opt_std_cols[0]: ['SAN FP Equity', 'BNP FP Equity', 'ACA FP Equity', 'GLE FP Equity',
                                       'DBK GR Equity', 'CBK GR Equity', 'ISP IM Equity', 'UCG IM Equity',
                                       'BBVA SM Equity', 'SAN SM Equity', 'ING NA Equity', 'KBC BB Equity'],
                    _opt_std_cols[1]: [105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105],
                    'Min Weight': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    'Max Weight': [25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25]
                })
            st.data_editor(
                st.session_state.long_df,
                num_rows="dynamic",
                use_container_width=True,
                key="long_editor"
            )
            long_pasted_text = st.text_area(
                "Paste long basket data here:",
                height=100,
                key="long_paste",
                help=f"Format: {', '.join(_opt_std_cols)}"
            )
            if st.button("Fill Long Basket", key="fill_long"):
                if long_pasted_text:
                    try:
                        if '\n' not in long_pasted_text:
                            values = long_pasted_text.split(
                                '\t') if '\t' in long_pasted_text else long_pasted_text.split(',')
                            row_data = {}
                            for i, col in enumerate(_opt_std_cols):
                                row_data[col] = [values[i] if i < len(values) else '']
                            st.session_state.long_df = pd.DataFrame(row_data)
                        else:
                            df = pd.read_csv(io.StringIO(long_pasted_text),
                                             sep='\t' if '\t' in long_pasted_text else ',',
                                             header=None)
                            new_data = pd.DataFrame()
                            for i, col in enumerate(_opt_std_cols):
                                new_data[col] = df.iloc[:, i] if i < df.shape[1] else [''] * len(df)
                            st.session_state.long_df = new_data
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing long basket data: {e}")
    with col_right:
        st.subheader("📉 Short Basket Candidates")
        if is_cross_corridor:
            if 'short_df_cross' not in st.session_state:
                st.session_state.short_df_cross = pd.DataFrame({
                    'Variance Asset': [],
                    'Corridor Condition Asset': [],
                    'Strike Cross Corridor (%)': [],
                    'Strike Mono Var Swap (%)': [],
                    'Min Weight': [],
                    'Max Weight': []
                })
            st.data_editor(
                st.session_state.short_df_cross[
                    ['Variance Asset', 'Corridor Condition Asset', 'Strike Cross Corridor (%)',
                     'Strike Mono Var Swap (%)', 'Min Weight', 'Max Weight']],
                num_rows="dynamic",
                use_container_width=True,
                key="short_editor_cross",
                column_order=['Variance Asset', 'Corridor Condition Asset', 'Strike Cross Corridor (%)',
                              'Strike Mono Var Swap (%)', 'Min Weight', 'Max Weight'],
            )
            short_pasted_text = st.text_area(
                "Paste short basket data here:",
                height=100,
                key="short_paste_cross",
                help="Format: Variance Asset, Corridor Condition Asset, Strike Cross Corridor (%), Strike Mono Var Swap (%), Min Weight, Max Weight"
            )
            if st.button("Fill Short Basket", key="fill_short_cross"):
                if short_pasted_text:
                    try:
                        if '\n' not in short_pasted_text:
                            values = short_pasted_text.split(
                                '\t') if '\t' in short_pasted_text else short_pasted_text.split(',')
                            row_data = {}
                            for i, col in enumerate(
                                    ['Variance Asset', 'Corridor Condition Asset', 'Strike Cross Corridor (%)',
                                     'Strike Mono Var Swap (%)', 'Min Weight', 'Max Weight']):
                                row_data[col] = [values[i] if i < len(values) else '']
                            st.session_state.short_df_cross = pd.DataFrame(row_data)
                        else:
                            df = pd.read_csv(io.StringIO(short_pasted_text),
                                             sep='\t' if '\t' in short_pasted_text else ',',
                                             header=None)
                            new_data = pd.DataFrame()
                            for i, col in enumerate(
                                    ['Variance Asset', 'Corridor Condition Asset', 'Strike Cross Corridor (%)',
                                     'Strike Mono Var Swap (%)', 'Min Weight', 'Max Weight']):
                                new_data[col] = df.iloc[:, i] if i < df.shape[1] else [''] * len(df)
                            st.session_state.short_df_cross = new_data
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing short basket data: {e}")
        else:
            # STANDARD SHORT BASKET (vol swap or corridor)
            if 'short_df' not in st.session_state or list(st.session_state.short_df.columns) != _opt_std_cols:
                st.session_state.short_df = pd.DataFrame({
                    _opt_std_cols[0]: [],
                    _opt_std_cols[1]: [],
                    'Min Weight': [],
                    'Max Weight': []
                })
            st.data_editor(
                st.session_state.short_df,
                num_rows="dynamic",
                use_container_width=True,
                key="short_editor"
            )
            short_pasted_text = st.text_area(
                "Paste short basket data here:",
                height=100,
                key="short_paste",
                help=f"Format: {', '.join(_opt_std_cols)}"
            )
            if st.button("Fill Short Basket", key="fill_short"):
                if short_pasted_text:
                    try:
                        if '\n' not in short_pasted_text:
                            values = short_pasted_text.split(
                                '\t') if '\t' in short_pasted_text else short_pasted_text.split(',')
                            row_data = {}
                            for i, col in enumerate(_opt_std_cols):
                                row_data[col] = [values[i] if i < len(values) else '']
                            st.session_state.short_df = pd.DataFrame(row_data)
                        else:
                            df = pd.read_csv(io.StringIO(short_pasted_text),
                                             sep='\t' if '\t' in short_pasted_text else ',',
                                             header=None)
                            new_data = pd.DataFrame()
                            for i, col in enumerate(_opt_std_cols):
                                new_data[col] = df.iloc[:, i] if i < df.shape[1] else [''] * len(df)
                            st.session_state.short_df = new_data
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing short basket data: {e}")
    # Create stocks data button
    st.subheader("🧬 Backtest & Data Parameters")
    col20, col21, col22 = st.columns(3)
    with col20:
        missing_data_policy = st.selectbox(
            "Missing Data Policy",
            ["Adaptive Reweight", "Fill Zero (Gaia_PP)", "Drop Incomplete Days"],
            index=0,
            key="missing_data_policy_tab1",
            help="Adaptive Reweight: redistributes weights when stocks have missing data. Fill Zero: missing=0 contribution. Drop: only keeps days where all stocks have data."
        )
        if missing_data_policy == "Adaptive Reweight":
            reweight_grace_days = st.number_input(
                "Grace days (reweight)", min_value=0, max_value=30, value=3,
                key="grace_days_tab1",
                help="Consecutive missing days before stock excluded from reweighting"
            )
        else:
            reweight_grace_days = 3
    with col21:
        adj_divs = st.selectbox(
            "Adjust for Dividends",
            ["No", "Yes"],
            index=0,
            key="adj_divs_tab1",
            help="Adjust stock returns for dividend payments in backtest"
        )
    with col22:
        pass
    st.divider()
    if st.button("🔄 Compute Long/Short Strategy Data", type="primary"):
        # Calculate n_exp if date mode is selected
        if st.session_state.get('n_exp_mode_tab1') == 'Date':
            # Get tickers from appropriate dataframe
            if is_cross_corridor:
                long_df_check = st.session_state.get('long_df_cross', pd.DataFrame())
                # Extract stock tickers from cross corridor
                all_tickers = long_df_check['Variance Asset'].tolist() if len(long_df_check) > 0 else []
            else:
                long_df_check = st.session_state.get('long_df', pd.DataFrame())
                _ticker_col = 'Underlying' if 'Underlying' in long_df_check.columns else 'Variance Asset'
                all_tickers = long_df_check[_ticker_col].tolist() if len(long_df_check) > 0 else []
            if len(all_tickers) == 0:
                st.error("Please add stocks to calculate days from date!")
            else:
                with st.spinner("Calculating trading days..."):
                    try:
                        maturity_date = st.session_state.get('maturity_date_tab1')
                        n_exp = get_n_exp_from_date(maturity_date, all_tickers)
                        st.success(f"✅ Calculated {n_exp} trading days to {maturity_date}")
                    except Exception as e:
                        st.error(f"Failed to calculate days: {str(e)}")
                        n_exp = 310  # Fallback value
        else:
            n_exp = st.session_state.get('n_exp', 310)
        if is_cross_corridor:
            long_df_check = st.session_state.get('long_df_cross', pd.DataFrame())
            short_df_check = st.session_state.get('short_df_cross', pd.DataFrame())
            if len(long_df_check) == 0:
                st.error("Please add stocks to the long basket!")
            else:
                # Store config — optimize() handles data loading internally
                st.session_state['opt_config_ready'] = True
                st.session_state['is_vol_swap'] = is_vol_swap
                st.session_state['n_exp'] = n_exp
                st.session_state['local_cap'] = local_cap
                st.session_state['local_floor'] = local_floor
                st.session_state['global_cap'] = global_cap
                st.session_state['global_floor'] = global_floor
                st.session_state['ubar'] = ubar
                st.session_state['dbar'] = dbar
                st.session_state['adj_divs'] = adj_divs
                st.session_state['_opt_start_date_value'] = optimization_start
                st.session_state['is_cross_corridor'] = True
                st.session_state['is_long_only'] = len(short_df_check) == 0
                st.session_state['n_long_candidates'] = len(long_df_check)
                st.session_state['n_short_candidates'] = len(short_df_check)
                # ── (debug preview removed — use backtest chart for validation) ──
        else:
            long_df_check = st.session_state.get('long_df', pd.DataFrame())
            short_df_check = st.session_state.get('short_df', pd.DataFrame())
            if len(long_df_check) == 0:
                st.error("Please add stocks to the long basket!")
            else:
                # Store config — optimize() handles data loading internally
                st.session_state['opt_config_ready'] = True
                st.session_state['is_vol_swap'] = is_vol_swap
                st.session_state['n_exp'] = n_exp
                st.session_state['local_cap'] = local_cap
                st.session_state['local_floor'] = local_floor
                st.session_state['global_cap'] = global_cap
                st.session_state['global_floor'] = global_floor
                st.session_state['ubar'] = ubar
                st.session_state['dbar'] = dbar
                st.session_state['adj_divs'] = adj_divs
                st.session_state['_opt_start_date_value'] = optimization_start
                st.session_state['is_cross_corridor'] = False
                st.session_state['is_long_only'] = len(short_df_check) == 0
                st.session_state['n_long_candidates'] = len(long_df_check)
                st.session_state['n_short_candidates'] = len(short_df_check)
    # Optimization parameters section - only shows if config is ready
    if st.session_state.get('opt_config_ready'):
        st.divider()
        n_long = st.session_state.get('n_long_candidates', 0)
        n_short = st.session_state.get('n_short_candidates', 0)
        st.success(f"✅ Configuration saved: {n_long} long candidates, {n_short} short candidates")
        col44, col45 = st.columns(2)
        with col44:
            st.subheader("🎯 Basket Size Constraints")
            min_stocks_long = st.number_input("Min stocks in long basket", step=1, value=3)
            max_stocks_long = st.number_input("Max stocks in long basket", step=1, value=8)
            min_stocks_short = st.number_input("Min stocks in short basket", step=1, value=2)
            max_stocks_short = st.number_input("Max stocks in short basket", step=1, value=5)
            max_strike = st.number_input("Max net strike (%)", value=15.00)
            time_limit = st.number_input("Time Limit (seconds)", max_value=600, value=20)
            seed = st.number_input("Random seed", value=0, step=1, min_value=0)
            forced_tickers_raw = st.text_area(
                "🔒 Force names in basket (one per line or comma-separated)",
                value="", height=68,
                help="These names WILL be in the final basket. Their weights stay "
                     "governed by the Min/Max Weight inputs. Never removed by the 0%-HR filter.")
            excluded_tickers_raw = st.text_area(
                "🚫 Exclude names (one per line or comma-separated)",
                value="", height=68,
                help="Removed from the candidate universe (long and short) before optimization.")
            with st.expander("🌍 Bucket constraints (region/sector, optional)"):
                st.caption(
                    "Buckets match the 'Sector' column of the long input. "
                    "Leave the table empty to deactivate. Weights in %, "
                    "blank Max = unlimited.")
                _bucket_default = pd.DataFrame(
                    {"Bucket": pd.Series(dtype="str"),
                     "Min names": pd.Series(dtype="float"),
                     "Max names": pd.Series(dtype="float"),
                     "Min weight (%)": pd.Series(dtype="float"),
                     "Max weight (%)": pd.Series(dtype="float")})
                _bucket_df = st.data_editor(
                    st.session_state.get('_bucket_df', _bucket_default),
                    num_rows="dynamic", use_container_width=True,
                    key="_bucket_editor")
                st.session_state['_bucket_df'] = _bucket_df

            def _parse_bucket_constraints(df):
                out = []
                if df is None or df.empty:
                    return None
                for _, r in df.iterrows():
                    b = str(r.get("Bucket", "") or "").strip()
                    if not b:
                        continue
                    def _num(v):
                        try:
                            f = float(v)
                            return None if pd.isna(f) else f
                        except (TypeError, ValueError):
                            return None
                    mn = _num(r.get("Min names"))
                    mx = _num(r.get("Max names"))
                    mw = _num(r.get("Min weight (%)"))
                    xw = _num(r.get("Max weight (%)"))
                    out.append({
                        "bucket": b,
                        "min_names": int(mn) if mn is not None else 0,
                        "max_names": int(mx) if mx is not None else None,
                        "min_weight": (mw / 100.0) if mw is not None else 0.0,
                        "max_weight": (xw / 100.0) if xw is not None else None,
                    })
                return out or None

            run_milp = st.checkbox("🔬 Certify vs exact optimum (MILP, slow)", value=False)
            st.session_state['_run_milp'] = run_milp
            bisect_in_ga = st.checkbox("🎯 Exact solver in GA (slow, certification runs)", value=False)
            st.session_state['_bisect_in_ga'] = bisect_in_ga
        with col45:
            st.subheader("⚖️ Optimization Weights")
            col18, col19 = st.columns(2)
            with col18:
                last_carry_weight = st.number_input("Last carry weight", step=0.01, min_value=0.00,
                                                    max_value=1.00, value=0.30,
                                                    help="Recent payoff (last maturities)")
                mean_payoff_weight = st.number_input("Mean payoff weight", step=0.01, min_value=0.00,
                                                     max_value=1.00, value=0.30,
                                                     help="Average payoff over the backtest window")
            with col19:
                hit_ratio_weight = st.number_input("Hit ratio weight", step=0.01, min_value=0.00,
                                                   max_value=1.00, value=0.30,
                                                   help="Fraction of positive payoffs")
                min_payoff_weight = st.number_input("Min payoff weight", step=0.01, min_value=0.00,
                                                    max_value=1.00, value=0.10,
                                                    help="Worst single payoff (floor protection)")
            with st.expander("➕ Optional criteria (0 = inactive)"):
                col19b, col19c = st.columns(2)
                with col19b:
                    max_dd_weight = st.number_input("Max drawdown weight", step=0.01, min_value=0.00,
                                                    max_value=1.00, value=0.00,
                                                    help="Worst peak-to-trough drop of the payoff curve (penalized)")
                    cvar_weight = st.number_input("CVaR 5% weight", step=0.01, min_value=0.00,
                                                  max_value=1.00, value=0.00,
                                                  help="Mean of the 5% worst payoffs (tail protection)")
                with col19c:
                    sharpe_weight = st.number_input("Sharpe weight", step=0.01, min_value=0.00,
                                                    max_value=1.00, value=0.00,
                                                    help="Risk-adjusted payoff (mean/std, annualized)")
                    strike_obj_weight = st.number_input("Weighted strike weight", step=0.01, min_value=0.00,
                                                        max_value=1.00, value=0.00,
                                                        help="Minimize the basket's weighted net strike. "
                                                             "The Max net strike hard limit stays active independently.")
        # Validation
        score_weights = {
            'last_carry': last_carry_weight,
            'mean_payoff': mean_payoff_weight,
            'hit_ratio': hit_ratio_weight,
            'min_payoff': min_payoff_weight,
        }
        for _opt_name, _opt_w in [('max_drawdown', max_dd_weight), ('cvar_5', cvar_weight),
                                  ('sharpe_payoff', sharpe_weight), ('weighted_strike', strike_obj_weight)]:
            if _opt_w > 0:
                score_weights[_opt_name] = _opt_w
        total_weight = sum(score_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            st.warning(f"⚠️ Weights sum to {total_weight:.2f}, not 1.0. They will be normalized automatically.")
        st.divider()
        filter_zero_hr = st.toggle(
            "🧹 Filter useless candidates (0% HR long / 100% HR short)",
            value=True,
            help="Remove long stocks with 0% hit ratio and short stocks with 100% hit ratio before optimization. Only applies if remaining candidates >= min stocks.",
            key="filter_zero_hr"
        )
        enable_profiling = st.checkbox("📊 Enable Performance Profiling", value=False, key="enable_profiling")
        if st.button("🚀 Run Genetic Algorithm Optimization", type="primary", use_container_width=True):
            is_long_only = st.session_state.get('is_long_only', False)
            if max_strike <= 0:
                st.error("Max strike must be positive!")
            elif min_stocks_long > n_long:
                st.error(
                    f"Min long stocks ({min_stocks_long}) cannot be larger than available long candidates ({n_long})")
            elif not is_long_only and min_stocks_short > n_short:
                st.error(
                    f"Min short stocks ({min_stocks_short}) cannot be larger than available short candidates ({n_short})")
            else:
                # Pre-flight feasibility check: max_stocks × max_weight >= 100%
                _preflight_ok = True
                if is_cross_corridor:
                    _src_check = st.session_state.get('long_df_cross', pd.DataFrame())
                else:
                    _src_check = st.session_state.get('long_df', pd.DataFrame())
                if not _src_check.empty and 'Max Weight' in _src_check.columns:
                    _max_w_pct = _src_check['Max Weight'].astype(float).max()
                    _max_w_dec = _max_w_pct / 100.0
                    _max_alloc = max_stocks_long * _max_w_dec
                    if _max_alloc < 1.0 - 1e-9:
                        import math
                        _needed_stocks = math.ceil(1.0 / _max_w_dec)
                        _needed_weight = 100.0 / max_stocks_long
                        st.error(
                            f"⚠️ **Infeasible long constraints:**\n\n"
                            f"- Max stocks in long basket = **{max_stocks_long}**\n"
                            f"- Max weight per stock = **{_max_w_pct:.1f}%**\n"
                            f"- Max possible allocation = {max_stocks_long} × {_max_w_pct:.1f}% = **{_max_alloc * 100:.1f}%** < 100%\n\n"
                            f"**Required:** Max Stocks × Max Weight ≥ 100%\n\n"
                            f"Fix either:\n"
                            f"- Increase max stocks long to at least **{_needed_stocks}**\n"
                            f"- Or increase Max Weight to at least **{_needed_weight:.1f}%**"
                        )
                        _preflight_ok = False
                if _preflight_ok:
                    try:
                        if enable_profiling:
                            profiler = cProfile.Profile()
                            profiler.enable()
                        # Adjust parameters for long-only mode
                        if is_long_only:
                            st.info("🎯 Running LONG-ONLY optimization")
                            min_stocks_short_adj = 0
                            max_stocks_short_adj = 0
                        else:
                            min_stocks_short_adj = min_stocks_short
                            max_stocks_short_adj = max_stocks_short
                        # Progress bar for optimization
                        opti_progress = st.progress(0.0, text="Initializing optimizer...")
                        opti_status = st.empty()

                        def _opti_progress(gen, max_gen, best_score):
                            pct = min(gen / max(max_gen, 1), 1.0)
                            opti_progress.progress(pct, text=f"Gen {gen}/{max_gen} | Best: {best_score:.4f}")
                            opti_status.caption(f"🧬 Generation {gen}/{max_gen} — best score: {best_score:.4f}")

                        _is_xc = st.session_state.get('is_cross_corridor', False)
                        if _is_xc:
                            _long_src = st.session_state.get('long_df_cross', pd.DataFrame())
                            _short_src = st.session_state.get('short_df_cross', pd.DataFrame())
                        else:
                            _long_src = st.session_state.get('long_df', pd.DataFrame())
                            _short_src = st.session_state.get('short_df', pd.DataFrame())
                        _short_df_arg = _short_src if len(_short_src) > 0 else None
                        _cfg = DispersionConfig(
                            product_type=ProductType.VOL_SWAP if st.session_state.get('is_vol_swap',
                                                                                      False) else ProductType.VAR_SWAP_CORRIDOR,
                            cross_corridor=_is_xc,
                            n_exp=st.session_state.get('n_exp', 252),
                            barrier_up=st.session_state.get('ubar', 1.30),
                            barrier_down=st.session_state.get('dbar', 0.70),
                            local_cap=st.session_state.get('local_cap', 2.5),
                            is_capped=True,
                            missing_data_policy=_parse_policy(missing_data_policy),
                            adj_divs=(st.session_state.get('adj_divs', 'No') == "Yes"),
                            lookback_years=5,
                            global_cap=st.session_state.get('global_cap', 9999999.0),
                            global_floor=st.session_state.get('global_floor', -9999999.0),
                        )
                        _constraints = OptimizationConstraints(
                            min_stocks_long=min_stocks_long,
                            max_stocks_long=max_stocks_long,
                            min_stocks_short=min_stocks_short_adj,
                            max_stocks_short=max_stocks_short_adj,
                            max_net_strike=max_strike / 100.0,
                            time_limit_seconds=float(time_limit),
                        )
                        # Build long/short DataFrames from session state input DFs
                        if _is_xc:
                            _long_src = st.session_state.get('long_df_cross', pd.DataFrame())
                            _short_src = st.session_state.get('short_df_cross', pd.DataFrame())
                        else:
                            _long_src = st.session_state.get('long_df', pd.DataFrame()).copy()
                            _short_src = st.session_state.get('short_df', pd.DataFrame()).copy()
                            # Vol swap UI uses product-specific labels — map to API canonical names
                            if st.session_state.get('is_vol_swap', False):
                                _vs_rename = {'Underlying': 'Variance Asset', 'Strike (%)': 'Strike Mono Var Swap (%)'}
                                if 'Underlying' in _long_src.columns:
                                    _long_src.rename(columns=_vs_rename, inplace=True)
                                if 'Underlying' in _short_src.columns:
                                    _short_src.rename(columns=_vs_rename, inplace=True)
                        _short_df_arg = _short_src if len(_short_src) > 0 else None
                        with st.spinner("Loading data & running optimization..."):
                            _opt_error = None
                            result = None
                            def _parse_ticker_list(raw_text):
                                out = []
                                for chunk in raw_text.replace(',', '\n').splitlines():
                                    t = chunk.strip()
                                    if t:
                                        out.append(t)
                                return out or None

                            try:
                                result = optimize(
                                    long_df=_long_src,
                                    config=_cfg,
                                    constraints=_constraints,
                                    short_df=_short_df_arg,
                                    score_weights=score_weights,
                                    start_date=st.session_state.get('_opt_start_date_value'),
                                    filter_zero_hr=filter_zero_hr,
                                    progress_callback=_opti_progress,
                                    run_milp=st.session_state.get('_run_milp', False),
                                    bisect_in_ga=bisect_in_ga,
                                    seed=int(seed),
                                    forced_tickers=_parse_ticker_list(forced_tickers_raw),
                                    excluded_tickers=_parse_ticker_list(excluded_tickers_raw),
                                    bucket_constraints=_parse_bucket_constraints(
                                        st.session_state.get('_bucket_df')),
                                )
                            except Exception as _e:
                                _opt_error = _e
                        opti_progress.progress(1.0, text="✅ Optimization complete")
                        import time as _time
                        _time.sleep(0.3)
                        opti_progress.empty()
                        opti_status.empty()
                        # ── MILP benchmark display ──
                        if result is not None and hasattr(result, '_milp_result') and result._milp_result is not None:
                            _milp = result._milp_result
                            if _milp.get("status") == "optimal":
                                _milp_min = _milp["min_payoff"]
                                # GA min payoff from backtest series
                                _bt_ts = result.backtest.timeseries if result.backtest else None
                                if _bt_ts is not None and "Result" in _bt_ts.columns:
                                    _ga_series = _bt_ts["Result"]
                                    _ga_valid = _ga_series[_ga_series != 0.0]
                                    _ga_min = float(_ga_valid.min()) if len(_ga_valid) > 0 else 0.0
                                else:
                                    _ga_min = 0.0
                                _gap = _milp_min - _ga_min
                                _pct = abs(_gap / _milp_min) * 100 if abs(_milp_min) > 1e-9 else 0.0
                                st.info(
                                    f"🔬 GA min={_ga_min:.4f} | Exact optimum={_milp_min:.4f} | gap={_gap:.4f} ({_pct:.1f}%)")
                            elif _milp.get("status") == "failed":
                                _bound = _milp.get("bound")
                                st.warning(f"🔬 MILP hit time limit. Bound={_bound}, message={_milp.get('message')}")
                            else:
                                st.warning("🔬 MILP certificate only available for min_payoff=1.0 config")
                        # ══════════════════════════════════════════════════════════
                        # OPTIMIZER DEBUG — only expanded on error
                        # ══════════════════════════════════════════════════════════
                        _show_debug = (_opt_error is not None) or (result is None) or (
                                result and (result.score <= -1e9 or not result.long_basket))
                        if _show_debug:
                            with st.expander("🔍 Optimizer Debug", expanded=True):
                                st.markdown(
                                    f"**Input:** {len(_long_src)} long rows, {len(_short_src) if _short_src is not None else 0} short rows")
                                st.markdown(
                                    f"**Constraints:** min_stocks_long={min_stocks_long}, max_stocks_long={max_stocks_long}")
                                st.markdown(
                                    f"**Cross-corridor:** {_is_xc} | **Long-only:** {is_long_only} | **max_strike:** {max_strike}%")
                                if _opt_error:
                                    st.error(f"**Exception:** `{type(_opt_error).__name__}: {_opt_error}`")
                                    import traceback
                                    st.code("".join(traceback.format_exception(type(_opt_error), _opt_error,
                                                                               _opt_error.__traceback__)),
                                            language="python")
                                if result:
                                    st.markdown(f"**Score:** {result.score:.6f} | **Converged:** {result.converged}")
                                    st.markdown(
                                        f"**Long basket:** {len(result.long_basket)} | **Short basket:** {len(result.short_basket)}")
                                    _rejections = getattr(result, '_rejection_reasons', {})
                                    if _rejections:
                                        st.markdown("**Rejection counts:**")
                                        _rej_lines = [f"| `{k}` | **{v}** |" for k, v in sorted(_rejections.items(),
                                                                                                key=lambda x: -x[
                                                                                                    1] if isinstance(
                                                                                                    x[1], (int,
                                                                                                           float)) else 0)]
                                        st.markdown("| Reason | Count |\n|--------|-------|\n" + "\n".join(_rej_lines))
                                    _dbg_table = getattr(result, '_debug_table', '')
                                    if _dbg_table:
                                        st.code(_dbg_table, language="text")
                        if _opt_error:
                            st.stop()
                        if result:
                            st.session_state.optimization_result = result
                            st.session_state['_opti_just_ran'] = True
                            st.session_state['gaia_last_run_is_xc'] = _is_xc
                            # ── Clear stale smoothing state (new run invalidates old smooth) ──
                            st.session_state.pop('gaia_smooth_result', None)
                            # Build display from OptimizationResult directly
                            _render_optimization_result(result, _is_xc)
                            # ── Persist state for interactive post-smoothing ──
                            st.session_state['gaia_last_run'] = result
                        if enable_profiling:
                            profiler.disable()
                            with st.expander("🔍 Performance Profile - Click to Expand"):
                                s = io.StringIO()
                                ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
                                ps.print_stats(20)
                                st.code(s.getvalue(), language='text')
                    except Exception as e:
                        st.error(f"Optimization failed: {str(e)}")
                        st.exception(e)
        # ═══ 🧪 MULTI-CONFIG COMPARISON (one data load, N weight configs) ═══
        with st.expander("🧪 Compare several weight configs (one data load)"):
            st.caption(
                "One config per line, metric=weight pairs comma-separated. "
                "Same seed and constraints as the single run above — each line "
                "reproduces exactly what a single run with those weights would return.")
            _mc_text = st.text_area(
                "Configs", value="mean_payoff=0.5, hit_ratio=0.5\nmin_payoff=1.0",
                height=80, key="_mc_configs_text")
            if st.button("Run comparison", key="_mc_run_btn"):
                def _parse_ticker_list_mc(raw_text):
                    out = []
                    for chunk in (raw_text or "").replace(',', '\n').splitlines():
                        t = chunk.strip()
                        if t:
                            out.append(t)
                    return out or None

                def _parse_multi_configs(txt):
                    out = []
                    for line in txt.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        cfg_d = {}
                        for part in line.split(','):
                            if '=' not in part:
                                raise ValueError(
                                    f"Bad config entry: '{part.strip()}' (expected metric=weight)")
                            k, v = part.split('=', 1)
                            cfg_d[k.strip()] = float(v)
                        if cfg_d:
                            out.append(cfg_d)
                    return out
                try:
                    _mc_configs = _parse_multi_configs(_mc_text)
                    if not _mc_configs:
                        st.warning("No configs to run.")
                    else:
                        from functions.dispersion._api import optimize_multi
                        # Mirror of the single-run cfg/constraints construction
                        _is_xc_mc = st.session_state.get('is_cross_corridor', False)
                        if _is_xc_mc:
                            _mc_long = st.session_state.get('long_df_cross', pd.DataFrame())
                            _mc_short = st.session_state.get('short_df_cross', pd.DataFrame())
                        else:
                            _mc_long = st.session_state.get('long_df', pd.DataFrame()).copy()
                            _mc_short = st.session_state.get('short_df', pd.DataFrame()).copy()
                            if st.session_state.get('is_vol_swap', False):
                                _vs_rn = {'Underlying': 'Variance Asset',
                                          'Strike (%)': 'Strike Mono Var Swap (%)'}
                                if 'Underlying' in _mc_long.columns:
                                    _mc_long.rename(columns=_vs_rn, inplace=True)
                                if 'Underlying' in _mc_short.columns:
                                    _mc_short.rename(columns=_vs_rn, inplace=True)
                        _mc_cfg = DispersionConfig(
                            product_type=ProductType.VOL_SWAP if st.session_state.get('is_vol_swap', False)
                            else ProductType.VAR_SWAP_CORRIDOR,
                            cross_corridor=_is_xc_mc,
                            n_exp=st.session_state.get('n_exp', 252),
                            barrier_up=st.session_state.get('ubar', 1.30),
                            barrier_down=st.session_state.get('dbar', 0.70),
                            local_cap=st.session_state.get('local_cap', 2.5),
                            is_capped=True,
                            missing_data_policy=_parse_policy(missing_data_policy),
                            adj_divs=(st.session_state.get('adj_divs', 'No') == "Yes"),
                            lookback_years=5,
                            global_cap=st.session_state.get('global_cap', 9999999.0),
                            global_floor=st.session_state.get('global_floor', -9999999.0),
                        )
                        _mc_cons = OptimizationConstraints(
                            min_stocks_long=min_stocks_long,
                            max_stocks_long=max_stocks_long,
                            min_stocks_short=0 if st.session_state.get('is_long_only', False) else min_stocks_short,
                            max_stocks_short=0 if st.session_state.get('is_long_only', False) else max_stocks_short,
                            max_net_strike=max_strike / 100.0,
                            time_limit_seconds=float(time_limit),
                        )
                        with st.spinner(f"Running {len(_mc_configs)} configs on one data load..."):
                            _mc = optimize_multi(
                                _mc_long, _mc_cfg, _mc_cons,
                                configs=_mc_configs,
                                short_df=_mc_short if len(_mc_short) > 0 else None,
                                start_date=st.session_state.get('_opt_start_date_value'),
                                filter_zero_hr=filter_zero_hr,
                                seed=int(seed),
                                forced_tickers=_parse_ticker_list_mc(forced_tickers_raw),
                                excluded_tickers=_parse_ticker_list_mc(excluded_tickers_raw),
                                bucket_constraints=_parse_bucket_constraints(
                                    st.session_state.get('_bucket_df')),
                            )
                        st.session_state['_mc_result'] = _mc
                except Exception as _mc_e:
                    st.error(f"Multi-config run failed: {_mc_e}")
                    st.exception(_mc_e)
            _mc_prev = st.session_state.get('_mc_result')
            if _mc_prev is not None:
                st.dataframe(_mc_prev.comparison, use_container_width=True)
                st.caption("pct_<metric> = percentile of the winner's raw value in that "
                           "run's reference distribution (higher = better).")
    # ═══════════════════════════════════════════════════════════════════
    # PERSISTENT RESULT DISPLAY (survives slider reruns)
    # ═══════════════════════════════════════════════════════════════════
    _persisted_result = st.session_state.get('optimization_result')
    _persisted_is_xc = st.session_state.get('gaia_last_run_is_xc', False)
    if _persisted_result is not None and not st.session_state.get('_opti_just_ran'):
        # Re-render on rerun (slider interaction, etc.) — skip if we just rendered above
        _render_optimization_result(_persisted_result, _persisted_is_xc)
    # Clear the "just ran" flag after this render cycle
    st.session_state.pop('_opti_just_ran', None)
    # ═══════════════════════════════════════════════════════════════════
    # 🪄 INTERACTIVE POST-SMOOTHING (rendered in tab1, after optimization)
    # ═══════════════════════════════════════════════════════════════════
    _last_run = st.session_state.get('gaia_last_run')
    if _last_run is not None and getattr(_last_run, '_smooth_state', None) is not None:
        _ss = _last_run._smooth_state
        if 'ts_mat' not in _ss or 'valid_mask' not in _ss:
            st.info("Result from an older version — re-run the optimization to enable smoothing.")
        else:
            st.markdown("---")
            st.subheader("🪄 Post-smoothing")
            st.caption("Tune weight smoothing interactively — no re-optimization needed.")
            _ps_col1, _ps_col2 = st.columns([2, 1])
            with _ps_col1:
                _ps_eps = st.slider("Smoothing eps (floor tolerance)", 0.0, 0.30, 0.05, 0.01, key="_ps_eps_slider")
            with _ps_col2:
                _ps_apply = st.button("Apply smoothing", key="_ps_apply_btn")
            # Show hint if eps changed vs last stored result
            _prev_smooth = st.session_state.get('gaia_smooth_result')
            if _prev_smooth is not None and abs(_prev_smooth.get('eps', 0.05) - _ps_eps) > 1e-6:
                st.caption("⚠️ eps changed — click **Apply** to recompute.")
            if _ps_apply and _ss.get("weight_solver") is not None:
                import numpy as _np
                from functions.dispersion._backtester import DispersionBacktester
                from functions.dispersion.models import MissingDataPolicy
                from functions.dispersion.scoring.weight_solver import adaptive_pnl as _adaptive_pnl_fn
                from functions.dispersion._optimizer import _candidate_key as _cand_key_fn
                # Source w_star from pure optimum (unsmoothed_basket is None when pipeline smoothing=off → use long_basket)
                _unsm = _last_run.unsmoothed_basket if _last_run.unsmoothed_basket else _last_run.long_basket
                _w_star = _np.array([w for _, w in _unsm], dtype=_np.float64)
                _basket_keys = [k for k, _ in _unsm]
                _col_pos = _ss["col_pos"]
                _is_xc = _ss["is_cross_corridor"]
                _ts_mat = _ss["ts_mat"]  # nan_to_num'd (zeros where NaN)
                _valid_mask = _ss["valid_mask"]  # bool mask for adaptive renorm
                # Build stock_indices exactly as FINAL-RAW does (col_pos maps keys → column indices)
                _stock_indices = _np.array([_col_pos[k] for k in _basket_keys if k in _col_pos], dtype=int)
                if len(_stock_indices) == len(_w_star):
                    _ws = _ss["weight_solver"]
                    # ── ASSERTION: verify w_star reproduces FINAL-RAW min ──
                    _star_pnl = _adaptive_pnl_fn(_ts_mat, _stock_indices, _w_star, _valid_mask)
                    _star_min = float(_np.min(_star_pnl))
                    _expected_min = getattr(_last_run, '_final_raw_min', None)
                    if _expected_min is not None and abs(_star_min - _expected_min) > 1e-4:
                        st.error(
                            f"⚠️ Mapping mismatch: interactive min={_star_min:.6f} vs FINAL-RAW min={_expected_min:.6f}. "
                            f"Smoothing aborted — results would be wrong."
                        )
                        st.session_state.pop('gaia_smooth_result', None)
                    else:
                        # Per-stock strikes (solver convention: mono − cross in
                        # XC mode) and bounds, so the smoothed weights honour
                        # the strike cap, the weighted_strike objective (when
                        # active) and each name's Min/Max Weight inputs.
                        _legs_by_key = {_cand_key_fn(_lg, _is_xc): _lg
                                        for _lg in _ss.get("long_candidates", [])}
                        _legs_sel = [_legs_by_key.get(k) for k in _basket_keys]
                        if any(_lg is None for _lg in _legs_sel):
                            _strikes_vec = None
                            _bounds_vec = None
                        else:
                            _strikes_vec = _np.array([
                                (l.strike_mono_var_swap - (l.strike_cross_corridor
                                 if l.strike_cross_corridor is not None
                                 else l.strike_mono_var_swap))
                                if _is_xc else l.strike_mono_var_swap
                                for l in _legs_sel], dtype=_np.float64)
                            _bounds_vec = _np.array(
                                [[l.min_weight, l.max_weight] for l in _legs_sel],
                                dtype=_np.float64)
                        # Call smooth_weights — passes nan_to_num'd matrix (same as in-pipeline)
                        _w_smooth = _ws.smooth_weights(
                            w_star=_w_star.copy(),
                            pnl_matrix=_ts_mat,
                            stock_indices=_stock_indices,
                            active_mask=_valid_mask,
                            eps_min=_ps_eps,
                            per_stock_bounds=_bounds_vec,
                            strikes=_strikes_vec,
                        )
                        # Evaluate both PnL series via shared canonical function
                        _sm_pnl = _adaptive_pnl_fn(_ts_mat, _stock_indices, _w_smooth, _valid_mask)

                        # Compute stats table rows
                        def _stats_row(pnl, w):
                            nz = pnl[pnl != 0.0]
                            return {
                                "min_payoff": float(_np.min(pnl)),
                                "mean_payoff": float(_np.mean(pnl)),
                                "last_carry": float(pnl[-1]) if len(pnl) > 0 else 0.0,
                                "hit_ratio": float((nz > 0).mean()) if len(nz) > 0 else 0.0,
                                "max_payoff": float(_np.max(pnl)),
                                "dispersion": float(_np.std(w)),
                            }

                        _opt_stats = _stats_row(_star_pnl, _w_star)
                        _sm_stats = _stats_row(_sm_pnl, _w_smooth)
                        print(
                            f"[POST-SMOOTH] eps={_ps_eps:.2f} | dispersion: {_opt_stats['dispersion']:.4f} -> {_sm_stats['dispersion']:.4f} | min: {_opt_stats['min_payoff']:.4f} -> {_sm_stats['min_payoff']:.4f} | mean: {_opt_stats['mean_payoff']:.4f} -> {_sm_stats['mean_payoff']:.4f}",
                            flush=True)
                        # Backtest BOTH baskets for overlay chart
                        _optimal_basket = list(zip(_basket_keys, _w_star.tolist()))
                        _smooth_basket = list(zip(_basket_keys, _w_smooth.tolist()))
                        _bt = DispersionBacktester(_ss["internal_cfg"])
                        _bt_kwargs = dict(
                            price_data=_ss["price_data_all"],
                            short_basket=_last_run.short_basket,
                            legs=_ss["legs"],
                            index_data=_ss["index_data_all"],
                            start_date=_ss["start_date"],
                        )
                        try:
                            _bt_opt = _bt.run_from_optimization(long_basket=_optimal_basket, **_bt_kwargs)
                        except Exception as _e:
                            _bt_opt = None
                            st.error(f"Backtest (optimal) failed: {_e}")
                        try:
                            _bt_sm = _bt.run_from_optimization(long_basket=_smooth_basket, **_bt_kwargs)
                        except Exception as _e:
                            _bt_sm = None
                            st.error(f"Backtest (smoothed) failed: {_e}")
                        # Build weight comparison rows
                        _wt_rows = []
                        for i, tk in enumerate(_basket_keys):
                            wo = float(_w_star[i])
                            ws = float(_w_smooth[i])
                            _wt_rows.append({"Ticker": tk, "Optimal (%)": wo * 100, "Smoothed (%)": ws * 100,
                                             "Δ (pp)": (ws - wo) * 100})
                        # Persist to session_state
                        st.session_state['gaia_smooth_result'] = {
                            "eps": _ps_eps,
                            "w_star": _w_star.tolist(),
                            "w_smooth": _w_smooth.tolist(),
                            "basket_keys": _basket_keys,
                            "smooth_basket": _smooth_basket,
                            "opt_stats": _opt_stats,
                            "sm_stats": _sm_stats,
                            "wt_rows": _wt_rows,
                            "bt_opt": _bt_opt,
                            "bt_sm": _bt_sm,
                            "is_xc": _is_xc,
                        }
                else:
                    st.warning("Column mapping mismatch — cannot smooth.")
            # ── Render from session_state (persists across reruns) ──
            _sr = st.session_state.get('gaia_smooth_result')
            if _sr is not None:
                # Stats table (chart is overlaid on the main backtest above)
                # STATS TABLE (side-by-side)
                if "opt_stats" in _sr and "sm_stats" in _sr:
                    _os = _sr["opt_stats"]
                    _ss_stats = _sr["sm_stats"]
                    _metric_labels = [
                        ("Min payoff", "min_payoff"),
                        ("Mean payoff", "mean_payoff"),
                        ("Last carry", "last_carry"),
                        ("Hit ratio", "hit_ratio"),
                        ("Max payoff", "max_payoff"),
                        ("Dispersion (σ weights)", "dispersion"),
                    ]
                    _tbl_rows = []
                    for label, key in _metric_labels:
                        ov = _os[key]
                        sv = _ss_stats[key]
                        _tbl_rows.append({
                            "Metric": label,
                            "Optimal": f"{ov:.4f}",
                            "Smoothed": f"{sv:.4f}",
                            "Delta": f"{sv - ov:+.4f}",
                        })
                    _stats_df = pd.DataFrame(_tbl_rows)
                    st.markdown(f"**Performance comparison** (eps={_sr['eps']:.2f})")
                    st.dataframe(_stats_df, use_container_width=True, hide_index=True)
                # 3. WEIGHT TABLE
                _wt_df = pd.DataFrame(_sr["wt_rows"]).sort_values("Δ (pp)", key=abs, ascending=False).reset_index(
                    drop=True)
                st.markdown("**Weight comparison** (sorted by |Δ|)")

                def _hl_delta(row):
                    if abs(row["Δ (pp)"]) > 0.5:
                        return ["background-color: #fff3cd"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    _wt_df.style.apply(_hl_delta, axis=1).format(
                        {"Optimal (%)": "{:.2f}", "Smoothed (%)": "{:.2f}", "Δ (pp)": "{:+.2f}"}),
                    use_container_width=True
                )
                # "Use these weights" button
                if st.button("✅ Use smoothed weights", key="_ps_use_btn"):
                    _last_run.long_basket = _sr["smooth_basket"]
                    _last_run.unsmoothed_basket = list(zip(_sr["basket_keys"], _sr["w_star"]))
                    st.session_state.optimization_result = _last_run
                    st.success("Smoothed weights applied to result.")
with tab3:
    st.header("Dispersion Backtest")
    # Backtest tab uses new schema only - no migration needed
    col1, col2, col3 = st.columns(3)
    with col1:
        swap_type = st.selectbox("Product Type",
                                 ["Vol Swap", "Var Swap", "Cross Corridor"],
                                 help="Vol Swap: volatility swap, Var Swap: variance swap, Cross Corridor: variance on one asset with corridor conditions from another",
                                 key="swap_type")
        is_vol_swap = swap_type == "Vol Swap"
        is_cross_corridor = swap_type == "Cross Corridor"
        is_corr = swap_type in ["Var Swap", "Cross Corridor"]
    with col2:
        n_exp_mode = st.selectbox(
            'N Expected Input Mode',
            ['Days', 'Date'],
            key='n_exp_mode_tab3'
        )
        if n_exp_mode == 'Days':
            n_exp = st.number_input("Expiry (days)", value=315, key="expiry")
        else:
            maturity_date = st.date_input(
                'Select Maturity Date',
                value=datetime.date.today() + datetime.timedelta(days=315),
                format="DD/MM/YYYY",
                key='maturity_date_tab3'
            )
            st.info("Days will be calculated based on tickers")
            n_exp = None  # Placeholder
    with col3:
        adj_divs = st.selectbox("Adjust for Dividends", ["No", "Yes"], key="adj_divs")
        missing_data_policy = st.selectbox("Missing Data Policy",
                                           ["Adaptive Reweight", "Fill Zero (Gaia_PP)", "Drop Incomplete Days"],
                                           index=0, key="mdp_tab3",
                                           help="Adaptive Reweight: redistribute weights for missing stocks. Fill Zero: original Gaia_PP behavior.")
        if missing_data_policy == "Adaptive Reweight":
            reweight_grace_days = st.number_input(
                "Grace days (reweight)", min_value=0, max_value=30, value=3,
                key="grace_days_tab3",
                help="Number of consecutive missing days before a stock is excluded from reweighting"
            )
        else:
            reweight_grace_days = 3
    col4, col5, col6, col7 = st.columns(4)
    with col4:
        local_cap = st.number_input("Local Cap", value=2.5, key="local_cap")
    with col5:
        local_floor = st.number_input("Local Floor", value=-9999999.0, key="local_floor")
    with col6:
        global_cap = st.number_input("Global Cap", value=9999999.0, key="global_cap")
    with col7:
        global_floor = st.number_input("Global Floor", value=-9999999.0, key="global_floor")
    # Corridor parameters (only show if corridor is needed)
    if is_corr:
        st.subheader("Corridor Parameters")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            ubar = st.number_input("Upper Corridor", value=130.00, step=0.01, format="%.2f", key="ubar_pct") / 100
        with col_c2:
            dbar = st.number_input("Lower Corridor", value=70.00, step=0.01, format="%.2f", key="dbar_pct") / 100
    else:
        ubar, dbar = 1.30, 0.70
    if not is_cross_corridor:
        st.subheader("Standard Underlyings")
        # Product-specific column labels: vol swap vs corridor
        if is_vol_swap:
            _bt_std_cols = ['Underlying', 'Strike (%)', 'Weight (%)']
        else:
            _bt_std_cols = ['Variance Asset', 'Strike Mono Var Swap (%)', 'Weight (%)']
        # Reset editor if columns changed (e.g. user switched product type)
        if 'edited_df_bis' not in st.session_state or list(st.session_state.edited_df_bis.columns) != _bt_std_cols:
            st.session_state.edited_df_bis = pd.DataFrame({col: [] for col in _bt_std_cols})
        # Data editor with product-specific labels
        st.session_state.edited_df_bis = st.data_editor(
            st.session_state.edited_df_bis,
            num_rows="dynamic",
            use_container_width=True,
            key="standard_editor",
        )
        # Paste functionality
        pasted_text = st.text_area(f"Paste data here ({', '.join(_bt_std_cols)}): ", height=100, key="standard_paste")
        if st.button("Fill Standard Editor", key="fill_standard"):
            if pasted_text:
                try:
                    if '\n' not in pasted_text:
                        values = pasted_text.split('\t') if '\t' in pasted_text else pasted_text.split(',')
                        row_data = {}
                        for i, col in enumerate(_bt_std_cols):
                            row_data[col] = [values[i] if i < len(values) else '']
                        st.session_state.edited_df_bis = pd.DataFrame(row_data)
                    else:
                        df = pd.read_csv(io.StringIO(pasted_text), sep='\t' if '\t' in pasted_text else ',',
                                         header=None)
                        new_data = pd.DataFrame()
                        for i, col in enumerate(_bt_std_cols):
                            new_data[col] = df.iloc[:, i] if i < df.shape[1] else [''] * len(df)
                        st.session_state.edited_df_bis = new_data
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing data: {e}")
    else:  # Cross corridor interface
        st.subheader("Cross Corridor Underlyings")
        st.info(
            "Format: Variance Asset, Corridor Condition Asset, Strike Cross Corridor (%), Strike Mono Var Swap (%), Weight (%)")
        if 'edited_df_cross' not in st.session_state:
            st.session_state.edited_df_cross = pd.DataFrame({
                'Variance Asset': [],
                'Corridor Condition Asset': [],
                'Strike Cross Corridor (%)': [],
                'Strike Mono Var Swap (%)': [],
                'Weight (%)': [],
            })
        # Cross corridor data editor with new visible names (no internal mapping)
        st.session_state.edited_df_cross = st.data_editor(
            st.session_state.edited_df_cross,
            num_rows="dynamic",
            use_container_width=True,
            key="cross_corridor_editor",
        )
        # Cross corridor paste functionality - accept new visible names
        cross_pasted_text = st.text_area("Paste cross corridor data here:", height=100, key="cross_paste")
        if st.button("Fill Cross Corridor Editor", key="fill_cross"):
            if cross_pasted_text:
                try:
                    if '\n' not in cross_pasted_text:
                        values = cross_pasted_text.split(
                            '\t') if '\t' in cross_pasted_text else cross_pasted_text.split(',')
                        row_data = {}
                        for i, col in enumerate(
                                ['Variance Asset', 'Corridor Condition Asset', 'Strike Cross Corridor (%)',
                                 'Strike Mono Var Swap (%)', 'Weight (%)']):
                            row_data[col] = [values[i] if i < len(values) else '']
                        st.session_state.edited_df_cross = pd.DataFrame(row_data)
                    else:
                        df = pd.read_csv(io.StringIO(cross_pasted_text), sep='\t' if '\t' in cross_pasted_text else ',',
                                         header=None)
                        new_data = pd.DataFrame()
                        for i, col in enumerate(
                                ['Variance Asset', 'Corridor Condition Asset', 'Strike Cross Corridor (%)',
                                 'Strike Mono Var Swap (%)', 'Weight (%)']):
                            new_data[col] = df.iloc[:, i] if i < df.shape[1] else [''] * len(df)
                        st.session_state.edited_df_cross = new_data
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing cross corridor data: {e}")
    st.divider()
    show_individual_legs = st.toggle(
        "📊 Show Individual Legs (instead of aggregate)",
        value=False,
        key="show_individual_legs",
        help="Display each leg separately instead of weighted aggregate"
    )
    bt_start_date = st.date_input(
        "Backtest Start Date",
        value=date.today() - relativedelta(years=5),
        key="bt_start_date_tab3",
        format="DD/MM/YYYY",
        help="Start date for the backtest lookback period"
    )
    st.divider()
    # Run backtest
    if st.button("🚀 Run Backtest", key="run_backtest", type="primary", use_container_width=True):
        try:
            # Calculate n_exp if date mode is selected
            if st.session_state.get('n_exp_mode_tab3') == 'Date':
                # Get tickers from appropriate dataframe
                if is_cross_corridor:
                    df_check = st.session_state.get('edited_df_cross', pd.DataFrame())
                    all_tickers = df_check['Variance Asset'].tolist() if len(df_check) > 0 else []
                else:
                    df_check = st.session_state.get('edited_df_bis', pd.DataFrame())
                    _ticker_col = 'Underlying' if 'Underlying' in df_check.columns else 'Variance Asset'
                    all_tickers = df_check[_ticker_col].tolist() if len(df_check) > 0 else []
                if len(all_tickers) == 0:
                    st.error("Please add stocks to calculate days from date!")
                    st.stop()
                else:
                    with st.spinner("Calculating trading days..."):
                        try:
                            maturity_date = st.session_state.get('maturity_date_tab3')
                            n_exp = get_n_exp_from_date(maturity_date, all_tickers)
                            st.success(f"✅ Calculated {n_exp} trading days to {maturity_date}")
                        except Exception as e:
                            st.error(f"Failed to calculate days: {str(e)}")
                            n_exp = 315  # Fallback value
            else:
                print("oula ya pas de nexp")
                n_exp = st.session_state.get('expiry')
            st.session_state['n_exp'] = n_exp
            # Clear previous state
            st.session_state['is_dual_sectorial'] = False
            st.session_state['show_email_analysis'] = False
            st.session_state['show_60d_analysis'] = False
            st.session_state['backtest_completed'] = True
            for key in ['fig_split', 'fig_60d_split', 'fig_entry_point', 'email_60d_graph',
                        'fig_sectorial', 'fig_sectorial_long', 'fig_sectorial_short']:
                if key in st.session_state:
                    del st.session_state[key]
            # Run backtest
            if is_cross_corridor:
                df_res_basket, backtest_metadata, graph_data, cross_corridor_data = _run_bt(
                    st.session_state.edited_df_cross, is_vol_swap, n_exp, local_floor, local_cap,
                    global_floor, global_cap, ubar, dbar, adj_divs=adj_divs,
                    start_date=bt_start_date,
                    is_cross_corridor=True,
                    missing_data_policy=missing_data_policy
                )
                st.session_state['is_cross_corridor'] = True
                # Store backtest results
                st.session_state.ds_res = df_res_basket
                st.session_state.long_tickers = backtest_metadata['long_tickers']
                st.session_state.short_tickers = backtest_metadata['short_tickers']
                st.session_state.long_weights = backtest_metadata['long_weights']
                st.session_state.short_weights = backtest_metadata['short_weights']
                st.session_state.raw_data = backtest_metadata['raw_data']
                st.session_state.unweighted_data = backtest_metadata['unweighted_data']
                st.session_state.cross_leg_data = backtest_metadata.get('cross_leg_data', None)
                # Store main graph
                if graph_data:
                    st.session_state['graph_backtest'] = graph_data
                # Generate sectorial graphs (use converted dataframe with Weights (%))
                df_cross = st.session_state.edited_df_cross.copy()
                df_cross['Weight (%)'] = df_cross['Weight (%)'].astype(float)
                long_stocks = df_cross[df_cross['Weight (%)'] > 0]['Variance Asset'].tolist()
                short_stocks = df_cross[df_cross['Weight (%)'] < 0]['Variance Asset'].tolist()
                long_weights_cross = df_cross[df_cross['Weight (%)'] > 0]['Weight (%)'].astype(float).tolist()
                short_weights_cross = df_cross[df_cross['Weight (%)'] < 0]['Weight (%)'].astype(float).abs().tolist()
                sectorial_result = func_graph.graph_sectorial(
                    long_stocks,
                    short_tickers=short_stocks,
                    weights=long_weights_cross,
                    short_weights=short_weights_cross
                )
                if 'error' not in sectorial_result:
                    if sectorial_result.get('is_dual'):
                        st.session_state['is_dual_sectorial'] = True
                        st.session_state['fig_sectorial_long'] = sectorial_result['fig_long']
                        st.session_state['fig_sectorial_short'] = sectorial_result['fig_short']
                    else:
                        st.session_state['is_dual_sectorial'] = False
                        st.session_state['fig_sectorial'] = sectorial_result['fig']
            else:
                df_res_basket, backtest_metadata, graph_data, cross_corridor_data = _run_bt(
                    st.session_state.edited_df_bis, is_vol_swap, n_exp, local_floor, local_cap,
                    global_floor, global_cap, ubar, dbar, adj_divs=adj_divs,
                    start_date=bt_start_date,
                    is_cross_corridor=False,
                    missing_data_policy=missing_data_policy
                )
                st.session_state['is_cross_corridor'] = False
                # Store backtest results
                st.session_state.ds_res = df_res_basket
                st.session_state.long_tickers = backtest_metadata['long_tickers']
                st.session_state.short_tickers = backtest_metadata['short_tickers']
                st.session_state.long_weights = backtest_metadata['long_weights']
                st.session_state.short_weights = backtest_metadata['short_weights']
                st.session_state.raw_data = backtest_metadata['raw_data']
                st.session_state.unweighted_data = backtest_metadata['unweighted_data']
                st.session_state.cross_leg_data = backtest_metadata.get('cross_leg_data', None)
                # Store main graph
                if graph_data:
                    st.session_state['graph_backtest'] = graph_data
                # Generate sectorial graphs
                sectorial_result = func_graph.graph_sectorial(
                    st.session_state.get('long_tickers', []),
                    short_tickers=st.session_state.get('short_tickers', []),
                    weights=st.session_state.get('long_weights', []).tolist() if hasattr(
                        st.session_state.get('long_weights', []), 'tolist') else st.session_state.get('long_weights',
                                                                                                      []),
                    short_weights=st.session_state.get('short_weights', []).tolist() if hasattr(
                        st.session_state.get('short_weights', []), 'tolist') else st.session_state.get('short_weights',
                                                                                                       [])
                )
                if 'error' not in sectorial_result:
                    if sectorial_result.get('is_dual'):
                        st.session_state['is_dual_sectorial'] = True
                        st.session_state['fig_sectorial_long'] = sectorial_result['fig_long']
                        st.session_state['fig_sectorial_short'] = sectorial_result['fig_short']
                    else:
                        st.session_state['is_dual_sectorial'] = False
                        st.session_state['fig_sectorial'] = sectorial_result['fig']
                # Generate entry point
                entry_fig = func_graph.entry_point(
                    st.session_state.get('long_tickers', []),
                    st.session_state.get('short_tickers', []),
                    st.session_state.get('long_weights', []),
                    st.session_state.get('short_weights', []),
                    start_date=bt_start_date,
                    end_date=date.today(),
                    is_cross_corridor=False
                )
                if entry_fig is not None:
                    st.session_state['fig_entry_point'] = entry_fig
            st.success("✅ Backtest completed successfully!")
        except Exception as e:
            st.error(f"❌ Backtest failed: {str(e)}")
            st.exception(e)
    # Display results if backtest completed
    if st.session_state.get('backtest_completed', False) and 'ds_res' in st.session_state:
        is_cross_corridor = st.session_state.get('is_cross_corridor', False)
        # Guard: ensure Result series has data
        _ds_res = st.session_state.ds_res
        if _ds_res is None or _ds_res.empty or "Result" not in _ds_res.columns or _ds_res["Result"].dropna().empty:
            st.warning("⚠️ Backtest completed but produced no results. Check input data.")
        else:
            st.divider()
            st.subheader("📊 Backtest Results")
            # Results table and metrics
            col_results, col_metrics = st.columns([1, 1])
            with col_results:
                st.write("**Backtest Results:**")
                st.dataframe(_ds_res.tail(10), use_container_width=True)
            with col_metrics:
                st.write("**Performance Metrics:**")
                time_series = _ds_res["Result"]
                min_val = time_series.min()
                max_val = time_series.max()
                mean_val = time_series.mean()
                non_zero = (time_series.replace(np.nan, 0)) != 0
                hr = ((time_series.replace(np.nan, 0)) > 0).sum() / max(1, non_zero.sum()) * 100
                last_value = time_series.iloc[-1] if len(time_series) > 0 else 0.0
                p25 = float(time_series.quantile(0.25))
                p50 = float(time_series.quantile(0.50))
                p75 = float(time_series.quantile(0.75))
            metrics_data = {
                'Metric': ['Mean', 'Hit Ratio (% Positive)', '', 'Max', '75%ile', '50%ile', '25%ile', 'Min', '',
                           'Last'],
                'Backtest': [f"{mean_val:.2f}%", f"{hr:.1f}%", '', f"{max_val:.2f}%", f"{p75:.2f}%", f"{p50:.2f}%",
                             f"{p25:.2f}%", f"{min_val:.2f}%", '', f"{last_value:.2f}%"]
            }
            carry_ts = st.session_state.get('carry_result_series')
            if carry_ts is not None and len(carry_ts) > 0:
                cs = carry_ts.replace(np.nan, 0)
                c_hr = (cs > 0).sum() / max(1, (cs != 0).sum()) * 100
                carry_last = cs.iloc[-1] if len(cs) > 0 else 0.0
                metrics_data['3M Carry'] = [
                    f"{cs.mean():.2f}%", f"{c_hr:.1f}%", '',
                    f"{cs.max():.2f}%", f"{float(cs.quantile(0.75)):.2f}%",
                    f"{float(cs.quantile(0.50)):.2f}%", f"{float(cs.quantile(0.25)):.2f}%",
                    f"{cs.min():.2f}%", '', f"{carry_last:.2f}%"
                ]
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        with st.expander("Full time series"):
            st.write("**Backtest Results:**")
            st.dataframe(st.session_state.ds_res, use_container_width=True)
        # Per-stock individual time series breakdown
        _unweighted = st.session_state.get('unweighted_data', pd.DataFrame())
        if _unweighted is not None and isinstance(_unweighted, pd.DataFrame) and not _unweighted.empty:
            with st.expander("📈 Per-Stock Time Series (Raw Data)"):
                if is_cross_corridor:
                    # Cross corridor: show stock leg and index leg separately
                    _cross_legs = st.session_state.get('cross_leg_data', None)
                    if _cross_legs is not None and isinstance(_cross_legs, pd.DataFrame) and not _cross_legs.empty:
                        st.dataframe(_cross_legs, use_container_width=True)
                    else:
                        # Fallback: show net per stock
                        st.dataframe(_unweighted, use_container_width=True)
                else:
                    # Mono corridor: stock1, stock2, ...
                    st.dataframe(_unweighted, use_container_width=True)
            # Per-stock charts with individual leg breakdown
            _cross_legs_2 = st.session_state.get('cross_leg_data', None)
            with st.expander("📊 Per-Stock Charts"):
                stock_tabs = st.tabs([col.replace(' Equity', '').replace(' Index', '') for col in _unweighted.columns])
                for tab, col in zip(stock_tabs, _unweighted.columns):
                    with tab:
                        stock_series = _unweighted[col].dropna()
                        if not stock_series.empty:
                            _c1, _c2, _c3, _c4 = st.columns(4)
                            with _c1:
                                st.metric("Mean", f"{stock_series.mean():.2f}%")
                            with _c2:
                                st.metric("Last", f"{stock_series.iloc[-1]:.2f}%")
                            with _c3:
                                _nz = stock_series[stock_series != 0]
                                _hr = ((_nz > 0).sum() / max(1, len(_nz))) * 100 if len(_nz) > 0 else 0
                                st.metric("HR", f"{_hr:.0f}%")
                            with _c4:
                                st.metric("Min", f"{stock_series.min():.2f}%")
                            st.caption("**Net P&L (stock - index)**" if is_cross_corridor else "**P&L**")
                            st.line_chart(stock_series, use_container_width=True)
                            # Cross-corridor: show stock leg and index leg separately
                            if is_cross_corridor and _cross_legs_2 is not None and isinstance(_cross_legs_2,
                                                                                              pd.DataFrame):
                                stock_leg_col = f"{col} (stock leg)"
                                index_leg_col = None
                                for c in _cross_legs_2.columns:
                                    if c.startswith(f"{col} (index leg"):
                                        index_leg_col = c
                                        break
                                if stock_leg_col in _cross_legs_2.columns and index_leg_col:
                                    st.caption("**Individual Legs**")
                                    legs_df = pd.DataFrame({
                                        "Stock Leg": _cross_legs_2[stock_leg_col],
                                        "Index Leg": _cross_legs_2[index_leg_col],
                                    }).dropna()
                                    st.line_chart(legs_df, use_container_width=True)
        # ✅ NEW: Conditional plotting based on toggle
        show_individual = st.session_state.get('show_individual_legs', False)
        if show_individual:
            st.subheader("📊 Individual Leg Performance")
            # Get unweighted data from session state
            unweighted_data = st.session_state.get('unweighted_data', pd.DataFrame())
            if not unweighted_data.empty:
                # ✅ Filter by same date range as main backtest (5 years)
                start_date = bt_start_date
                nearest_index = unweighted_data.index.get_indexer([start_date], method='nearest')[0]
                unweighted_data_filtered = unweighted_data.iloc[nearest_index:]
                # Plot each leg separately
                fig = go.Figure()
                for col in unweighted_data_filtered.columns:
                    fig.add_trace(go.Scatter(
                        x=unweighted_data_filtered.index,
                        y=unweighted_data_filtered[col],
                        mode='lines',
                        name=col.replace(' Equity', '').replace(' Index', ''),
                        line=dict(width=2)
                    ))
                fig.update_layout(
                    title='Individual Leg Performance (Unweighted)',
                    xaxis_title='Date',
                    yaxis_title='Return (%)',
                    hovermode='x unified',
                    height=600,
                    showlegend=True,
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                st.plotly_chart(fig, use_container_width=True, key="plotly_individual_legs")
            else:
                st.warning("⚠️ Individual leg data not available")
        else:
            # Original aggregate plot
            if 'graph_backtest' in st.session_state and st.session_state['graph_backtest'] is not None:
                st.plotly_chart(st.session_state['graph_backtest'], use_container_width=True, key="plotly_bt_main")
        # Display sectorial analysis
        st.subheader("📊 Sectorial Analysis")
        is_dual = st.session_state.get('is_dual_sectorial', False)
        if is_dual:
            if 'fig_sectorial_long' in st.session_state and 'fig_sectorial_short' in st.session_state:
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(st.session_state.fig_sectorial_long, use_container_width=True,
                                    key="plotly_sect_long")
                with col2:
                    st.plotly_chart(st.session_state.fig_sectorial_short, use_container_width=True,
                                    key="plotly_sect_short")
        else:
            if 'fig_sectorial' in st.session_state:
                st.plotly_chart(st.session_state.fig_sectorial, use_container_width=True, key="plotly_sectorial")
        # Display entry point (only for non-cross corridor)
        if not is_cross_corridor and 'fig_entry_point' in st.session_state:
            st.subheader("📍 Entry Point Analysis")
            st.plotly_chart(st.session_state.fig_entry_point, use_container_width=True, key="plotly_entry_point")
        # Display split graph
        st.subheader("📊 Main Backtest Split")
        split_fig = func_graph.split_graph(
            st.session_state.ds_res,
            is_cross_corridor=is_cross_corridor
        )
        if split_fig is not None:
            st.session_state['fig_split'] = split_fig
            st.plotly_chart(split_fig, use_container_width=True, key="plotly_split_main")
        else:
            st.info("ℹ️ No short leg detected - split graph not applicable for long-only basket")
        st.divider()
        st.subheader("📈 Additional Analysis")
        # Email input
        recipient_email = st.text_input(
            "📧 Recipient Email (leave empty to display only)",
            value="",
            key="recipient_email_input",
            placeholder="example@company.com"
        )
        col_email, col_60d, col_clear = st.columns(3)
        with col_email:
            if st.button("📧 Generate Email Report", key="gen_email", use_container_width=True, type="primary"):
                st.session_state.show_email_analysis = True
                st.session_state.recipient_email = recipient_email
        with col_60d:
            if st.button("📅 Show 3M Carry Analysis", key="show_60d", use_container_width=True):
                st.session_state.show_60d_analysis = True
        with col_clear:
            if st.button("🗑️ Clear All", key="clear_analyses", use_container_width=True):
                for key in ['show_60d_analysis', 'show_email_analysis', 'backtest_completed',
                            'fig_split', 'fig_60d_split', 'fig_entry_point', 'fig_sectorial',
                            'fig_sectorial_long', 'fig_sectorial_short']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        st.divider()
        # 60D Analysis
        if st.session_state.get('show_60d_analysis', False):
            try:
                st.subheader("📅 3M Carry Analysis")
                df_for_60d = st.session_state.edited_df_cross if is_cross_corridor else st.session_state.edited_df_bis
                # Generate 60D backtest
                df_res_60d, _, _, _ = _run_bt(
                    df_for_60d, is_vol_swap, 60, local_floor, local_cap,
                    global_floor, global_cap, ubar, dbar, adj_divs=adj_divs,
                    start_date=bt_start_date,
                    is_cross_corridor=is_cross_corridor
                )
                st.session_state['carry_result_series'] = df_res_60d["Result"]
                # Display 60D main chart
                st.write("**3M Carry Chart:**")
                fig_60d = func_graph.plot_60d(df_res_60d, is_cross_corridor=is_cross_corridor)
                st.session_state['email_60d_graph'] = fig_60d
                st.plotly_chart(fig_60d, use_container_width=True, key="plotly_60d")
                # Display 60D split
                st.write("**3M Carry Split Chart:**")
                split_60d = func_graph.split_graph(df_res_60d, "60d", is_cross_corridor=is_cross_corridor)
                if split_60d is not None:
                    st.session_state['fig_60d_split'] = split_60d
                    st.plotly_chart(split_60d, use_container_width=True, key="plotly_60d_split")
                else:
                    st.info("ℹ️ No short leg - split not applicable")
                st.success("✅ 3M Carry Analysis completed!")
                st.divider()
            except Exception as e:
                st.error(f"❌ Error generating 3M Carry analysis: {e}")
        # Email Report Generation
        if st.session_state.get('show_email_analysis', False):
            st.session_state.show_email_analysis = False
            try:
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()

                def progress_callback(percent, message):
                    """Update Streamlit progress"""
                    progress_bar.progress(percent)
                    status_text.text(message)

                with st.expander("📧 Email Generation Progress", expanded=True):
                    status_text.text("Generating email report...")
                    # Ensure 60D graphs exist
                    if 'email_60d_graph' not in st.session_state or 'fig_60d_split' not in st.session_state:
                        df_for_60d = st.session_state.edited_df_cross if is_cross_corridor else st.session_state.edited_df_bis
                        df_res_60d, _, _, _ = _run_bt(
                            df_for_60d, is_vol_swap, 60, local_floor, local_cap,
                            global_floor, global_cap, ubar, dbar, adj_divs=adj_divs,
                            start_date=bt_start_date,
                            is_cross_corridor=is_cross_corridor
                        )
                        st.session_state['email_60d_graph'] = func_graph.plot_60d(
                            df_res_60d, is_cross_corridor=is_cross_corridor
                        )
                        st.session_state['fig_60d_split'] = func_graph.split_graph(
                            df_res_60d, "60d", is_cross_corridor=is_cross_corridor
                        )
                        st.session_state['carry_result_series'] = df_res_60d["Result"]
                    # Prepare data
                    # Prepare data
                    if is_cross_corridor:
                        data_for_email = st.session_state.edited_df_cross.copy()
                    else:
                        data_for_email = st.session_state.edited_df_bis.copy()
                    if not is_cross_corridor and (
                            'fig_entry_point' not in st.session_state or st.session_state.get('fig_entry_point') is None
                    ):
                        entry_fig = func_graph.entry_point(
                            st.session_state.get('long_tickers', []),
                            st.session_state.get('short_tickers', []),
                            st.session_state.get('long_weights', []),
                            st.session_state.get('short_weights', []),
                            start_date=bt_start_date,
                            end_date=date.today(),
                            is_cross_corridor=False
                        )
                        if entry_fig is not None:
                            st.session_state['fig_entry_point'] = entry_fig
                    # Prepare charts data for saving
                    charts_data = {
                        'main_graph': st.session_state.get('graph_backtest'),
                        'split_graph': st.session_state.get('fig_split'),
                        'graph_60d': st.session_state.get('email_60d_graph'),
                        'split_60d': st.session_state.get('fig_60d_split'),
                        'is_cross_corridor': is_cross_corridor,
                        'has_short_leg': st.session_state.ds_res.iloc[:, 1].abs().sum() > 0,
                        'is_dual_sectorial': st.session_state.get('is_dual_sectorial', False)
                    }
                    # Add sectorial graphs
                    if charts_data['is_dual_sectorial']:
                        charts_data['sectorial_long'] = st.session_state.get('fig_sectorial_long')
                        charts_data['sectorial_short'] = st.session_state.get('fig_sectorial_short')
                    else:
                        charts_data['sectorial'] = st.session_state.get('fig_sectorial')
                    # Add entry point (only if not cross corridor)
                    if not is_cross_corridor:
                        charts_data['entry_point'] = st.session_state.get('fig_entry_point')
                    # Save charts to disk
                    saved_files = func_graph.save_charts_to_disk(charts_data)
                    # Prepare email parameters
                    email_params = {
                        'data_editor': data_for_email,
                        'result_series': st.session_state.ds_res["Result"],
                        'carry_series': df_res_60d["Result"] if 'df_res_60d' in dir() else st.session_state.get(
                            'carry_result_series'),
                        'n_exp': st.session_state.get('n_exp'),
                        'is_cross_corridor': is_cross_corridor,
                        'product_type': 'Vol Swap' if is_vol_swap else 'Var Swap',
                        'short_leg_display': '' if is_cross_corridor else _get_short_leg_display(
                            st.session_state.get('short_tickers', [])
                        ),
                        'local_cap': local_cap,
                        'ubar': ubar,
                        'dbar': dbar,
                        'adj_divs': adj_divs,
                        'saved_files': saved_files,
                        'is_dual_sectorial': charts_data['is_dual_sectorial'],
                        'progress_callback': progress_callback
                    }
                    # Send email
                    email_result = func_graph.send_email_with_attachments(
                        charts_data,
                        recipient_email=st.session_state.get('recipient_email', ''),
                        **email_params
                    )
                    # Clear progress
                    time.sleep(2)
                    progress_bar.empty()
                    status_text.empty()
                    if email_result['success']:
                        st.success(f"✅ {email_result['message']}")
                    else:
                        st.error(f"❌ {email_result['message']}")
            except Exception as e:
                st.error(f"❌ Error generating email: {e}")
                st.exception(e)
with tab4:
    # ── Product & Mode ────────────────────────────────────────────────────────
    _c1, _c2, _c3, _c4 = st.columns([2, 2, 1.5, 1.5])
    with _c1:
        product_type = st.radio("Product", ["Variance Swap", "Vol Swap"],
                                horizontal=True, key="pricing_product_type")
    with _c2:
        _mode_key = "pricing_mode_unified_var" if product_type == "Variance Swap" else "pricing_mode_unified_vol"
        pricing_mode = st.radio("Mode", ["Solve", "Price", "Generate FPFs"], horizontal=True, key=_mode_key)
    with _c3:
        strike_date_p = st.date_input("Start", value=datetime.datetime.now(), format="DD/MM/YYYY", key="p_strike_date")
    with _c4:
        maturity_date_p = st.date_input("Maturity", value=datetime.date(2026, 6, 23), format="DD/MM/YYYY",
                                        key="p_maturity_date")
    st.divider()
    # ══════════════════════════════════════════════════════════════════════════
    # VARIANCE SWAP
    # ══════════════════════════════════════════════════════════════════════════
    if product_type == "Variance Swap":
        is_solve_var = pricing_mode == "Solve"
        is_generate_fpf_var = pricing_mode == "Generate FPFs"
        # ── Config row ──
        _r1, _r2, _r3, _r4, _r5, _r6, _r7_ev, _r8_fv, _r9_model = st.columns([1, 1, 1, 1, 1.2, 1.2, 1.2, 1.2, 1.5])
        with _r1:
            is_x_corr_var = st.toggle("Cross Corridor", value=True, key="p_is_x_corr")
        with _r2:
            is_corr_var = st.toggle("Corridor", value=True, key="p_is_corr")
        with _r3:
            is_capped_var = st.toggle("Capped", value=True, key="p_is_capped", help="6.25×strike²")
        with _r4:
            vol_mode_var = st.selectbox("Vol Mode", ["OFF", "ATMF", "ATMS", "ATMF+ATMS"], index=1, key="p_vol_mode")
        with _r5:
            dvar_var = st.number_input("Down Var", value=70.00, step=0.01, format="%.2f",
                                       key="p_dvar") / 100 if is_corr_var else 0.70
        with _r6:
            uvar_var = st.number_input("Up Var", value=130.00, step=0.01, format="%.2f",
                                       key="p_uvar") / 100 if is_corr_var else 1.30
        with _r7_ev:
            expected_var_mode = False  # Expected Var is now computed inside solve_strike automatically
        with _r8_fv:
            range_accrual_mode = False  # Range Accrual is now computed inside solve_strike automatically
        with _r9_model:
            model_name_var = st.selectbox("Model", [
                "Auto (detect from ticker)",
                "EMEA-Stocks-MC-LV-MultiAsset",
                "EMEA-Index-MC-LV-MultiAsset",
                "AMER-Stocks-MC-LV-MultiAsset",
                "AMER-Index-MC-LV-MultiAsset",
                "APAC-Stocks-MC-LV-MultiAsset",
                "APAC-Index-MC-LV-MultiAsset",
            ], index=0, key="p_model_name")
            model_name_param = None if "Auto" in model_name_var else model_name_var
        # ── Correlation (only if cross-corridor) ──
        if is_x_corr_var:
            _r7, _r8, _r9, _r10 = st.columns(4)
            with _r10:
                correl_input_method_var = st.selectbox("Correl Method",
                                                       ["Global Parameters", "Individual Correlations"],
                                                       key="p_correl_method")
            if correl_input_method_var == "Global Parameters":
                with _r7:
                    eqeq_lambda_var = st.number_input("EqEq λ (%)", value=40.0, key="p_eqeq") / 100
                with _r8:
                    correl_floor_var = st.number_input("Floor (%)", value=0.0, key="p_floor") / 100
                with _r9:
                    eqfx_shift_var = st.number_input("EqFx Shift (%)", value=-5.0, key="p_eqfx") / 100
            else:
                # Individual Correlations: these params are not used
                eqeq_lambda_var, correl_floor_var, eqfx_shift_var = 0.0, 0.0, 0.0
                with _r7:
                    st.caption("—")
                with _r8:
                    st.caption("—")
                with _r9:
                    st.caption("—")
        else:
            correl_input_method_var = "Global Parameters"
            eqeq_lambda_var, correl_floor_var, eqfx_shift_var = 0.1, 0.1, -0.05
        # ── Advanced: LSV / LCM ──
        if is_x_corr_var:
            with st.expander("⚙️ LSV / LCM", expanded=False):
                _adv1, _adv2 = st.columns(2)
                with _adv1:
                    apply_lsv_var = st.toggle("Enable LSV", value=False, key="p_apply_lsv")
                    if apply_lsv_var:
                        # Always sync LSV table with current tickers, pre-populated with defaults
                        # Use BOTH Variance Asset (index) and Corridor Condition Asset (stocks), deduplicated
                        _current_tickers = []
                        if 'edited_df_corr_p' in st.session_state and not st.session_state.edited_df_corr_p.empty:
                            _df_src = st.session_state.edited_df_corr_p
                            _seen = set()
                            for col in ['Variance Asset', 'Corridor Condition Asset']:
                                if col in _df_src.columns:
                                    for t in _df_src[col].astype(str).str.strip().tolist():
                                        if t and t != 'nan' and t not in _seen:
                                            _current_tickers.append(t)
                                            _seen.add(t)
                        _existing_lsv = st.session_state.get('edited_df_lsv_p', pd.DataFrame())
                        _existing_rics = set(_existing_lsv[
                                                 'RIC'].tolist()) if not _existing_lsv.empty and 'RIC' in _existing_lsv.columns else set()
                        # Index RICs get VolOfVar=1.25, stocks get 0.7
                        _INDEX_RICS = {".STOXX50E", ".SPX", ".FTSE", ".N225", ".HSI", ".FCHI", ".GDAXI", ".AEX",
                                       ".IBEX", ".SSMI"}
                        if _current_tickers and set(_current_tickers) != _existing_rics:
                            # Rebuild: keep user-edited values for existing RICs, add defaults for new ones
                            rows = []
                            for t in _current_tickers:
                                if not _existing_lsv.empty and t in _existing_rics:
                                    row = _existing_lsv[_existing_lsv['RIC'] == t].iloc[0].to_dict()
                                else:
                                    _vov = 1.25 if t in _INDEX_RICS else 0.7
                                    row = {'RIC': t, 'VolOfVar': _vov, 'Eq/VolCorrel': -0.7, 'MeanReversion': 2.0}
                                rows.append(row)
                            st.session_state.edited_df_lsv_p = pd.DataFrame(rows)
                        elif not _current_tickers and (
                                _existing_lsv.empty if isinstance(_existing_lsv, pd.DataFrame) else True):
                            st.session_state.edited_df_lsv_p = pd.DataFrame(
                                {'RIC': [], 'VolOfVar': [], 'Eq/VolCorrel': [], 'MeanReversion': []})
                        st.data_editor(st.session_state.edited_df_lsv_p, num_rows="dynamic", use_container_width=True,
                                       key="p_lsv_editor")
                        lsv_paste_var = st.text_area("Paste LSV:", height=68, key="p_lsv_paste")
                        if st.button("Fill LSV", key="p_fill_lsv") and lsv_paste_var:
                            try:
                                df = pd.read_csv(io.StringIO(lsv_paste_var), sep='\t' if '\t' in lsv_paste_var else ',',
                                                 header=None)
                                new_data = pd.DataFrame()
                                for i, col in enumerate(['RIC', 'VolOfVar', 'Eq/VolCorrel', 'MeanReversion']):
                                    if i < df.shape[1]:
                                        new_data[col] = df.iloc[:, i] if col == 'RIC' else pd.to_numeric(df.iloc[:, i],
                                                                                                         errors='coerce').fillna(
                                            0.0)
                                    else:
                                        new_data[col] = [''] * len(df) if col == 'RIC' else [0.0] * len(df)
                                st.session_state.edited_df_lsv_p = new_data
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
                        _lb1, _lb2 = st.columns(2)
                        with _lb1:
                            correl_bump_lsv_var = st.number_input("Bump (%)", value=0.0, key="p_lsv_bump") / 100
                        with _lb2:
                            correl_bump_lsv_style_var = st.selectbox("Style", ["Relative", "Absolute"], index=0,
                                                                     key="p_lsv_style")
                    else:
                        correl_bump_lsv_var, correl_bump_lsv_style_var = 0.0, "Relative"
                with _adv2:
                    apply_lcm_var = st.toggle("Enable LCM", value=False, key="p_apply_lcm")
                    if apply_lcm_var:
                        # Determine region from model name
                        _lcm_region = "AMER" if "AMER" in (model_name_var or "") else "EMEA"
                        # Load interpolated defaults based on maturity/region
                        from functions.correlation.parameters import (
                            load_lcm_params_csv, get_lcm_params_by_tenor, DEFAULT_LCM_PARAMS
                        )
                        _lcm_csv = load_lcm_params_csv(_lcm_region)
                        _lcm_defaults = get_lcm_params_by_tenor(_lcm_csv, maturity_date_p, _lcm_region,
                                                                method="nearest")
                        # ── Table 1: Currently used (nearest tenor) params ──
                        _days_to_mat = (maturity_date_p - pd.Timestamp.today().date()).days
                        _years_to_mat = _days_to_mat / 365.0
                        st.markdown(
                            f"**🎯 Active params** (nearest tenor for {_days_to_mat}d ≈ {_years_to_mat:.1f}Y, {_lcm_region})")
                        _active_df = pd.DataFrame([{
                            "λ ATM": f"{_lcm_defaults['LambdaAtm']:.4f}",
                            "Call Skew": f"{_lcm_defaults['CallSkew']:.4f}",
                            "Put Skew": f"{_lcm_defaults['PutSkew']:.4f}",
                            "λ From Rho0": f"{_lcm_defaults['LambdaFromRho0']:.4f}",
                            "λ Pricing": f"{_lcm_defaults['LambdaPricing']:.4f}",
                        }])
                        st.dataframe(_active_df, use_container_width=True, hide_index=True)
                        # ── Table 2: Full reference file params (highlight selected row) ──
                        st.markdown(f"**📖 Reference file** ({_lcm_region} all tenors)")
                        _lcm_region_defaults = DEFAULT_LCM_PARAMS[_lcm_region]
                        _lcm_ref_rows = [{"Tenor": t, **vals} for t, vals in _lcm_region_defaults.items()]
                        _ref_df = pd.DataFrame(_lcm_ref_rows)
                        # Find which tenor row is nearest
                        from functions.correlation.parameters import _parse_tenor_to_years
                        _ref_df["_years"] = _ref_df["Tenor"].apply(_parse_tenor_to_years)
                        _nearest_idx = (_ref_df["_years"] - _years_to_mat).abs().idxmin()
                        _ref_df = _ref_df.drop(columns=["_years"])
                        # Format numeric columns to 4 decimal places
                        _num_cols = [c for c in _ref_df.columns if c != "Tenor"]
                        _ref_df[_num_cols] = _ref_df[_num_cols].applymap(
                            lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)

                        def _highlight_nearest(row):
                            if row.name == _nearest_idx:
                                return ['background-color: #d4edda'] * len(row)
                            return [''] * len(row)

                        st.dataframe(_ref_df.style.apply(_highlight_nearest, axis=1), use_container_width=True,
                                     hide_index=True)
                        lcm_call_skew_var = st.number_input("Call Skew", value=float(_lcm_defaults["CallSkew"]),
                                                            step=0.05, format="%.4f", key="p_lcm_cs")
                        lcm_put_skew_var = st.number_input("Put Skew", value=float(_lcm_defaults["PutSkew"]), step=0.05,
                                                           format="%.4f", key="p_lcm_ps")
                        lcm_lambda_atm_var = st.number_input("λ ATM", value=float(_lcm_defaults["LambdaAtm"]),
                                                             step=0.01, format="%.7f", key="p_lcm_atm")
                        lcm_lambda_rho0_var = st.number_input("λ Rho0", value=float(_lcm_defaults["LambdaFromRho0"]),
                                                              step=0.01, format="%.4f", key="p_lcm_rho0")
                        lcm_lambda_pricing_var = st.number_input("λ Pricing",
                                                                 value=float(_lcm_defaults["LambdaPricing"]), step=0.01,
                                                                 format="%.4f", key="p_lcm_pr")
                        st.caption(
                            f"ℹ️ LCM uses ACEqEqSpread = λ Pricing ({lcm_lambda_pricing_var:.4f}), not EqEq λ input")
        else:
            apply_lsv_var = False
            apply_lcm_var = False
            correl_bump_lsv_var, correl_bump_lsv_style_var = 0.0, "Relative"
        # ── Underlyings editor ──
        st.divider()
        if is_solve_var:
            default_columns_var = ['Variance Asset']
        elif is_x_corr_var:
            default_columns_var = ['Variance Asset', 'Strike Cross Corridor (%)', 'Strike Mono Var Swap (%)']
        else:
            default_columns_var = ['Variance Asset', 'Strikes (%)']
        if is_x_corr_var:
            default_columns_var += ['Corridor Condition Asset', 'Currency']
            if correl_input_method_var == "Individual Correlations":
                default_columns_var.append('Correlation')
        if 'edited_df_corr_p' not in st.session_state or set(st.session_state.edited_df_corr_p.columns.tolist()) != set(
                default_columns_var):
            st.session_state.edited_df_corr_p = pd.DataFrame({col: [] for col in default_columns_var})
        st.session_state.edited_df_corr_p = st.data_editor(st.session_state.edited_df_corr_p, num_rows="dynamic",
                                                           use_container_width=True, key="p_corr_editor")
        _paste_col, _btn_col = st.columns([4, 1])
        with _paste_col:
            pasted_text_var = st.text_area("Paste data:", height=68, key="p_paste_var")
        with _btn_col:
            st.write("")
            st.write("")
            if st.button("📋 Fill", key="p_fill_var") and pasted_text_var:
                try:
                    if '\n' not in pasted_text_var:
                        values = pasted_text_var.split('\t') if '\t' in pasted_text_var else pasted_text_var.split(',')
                        st.session_state.edited_df_corr_p = pd.DataFrame(
                            {col: [values[i] if i < len(values) else ''] for i, col in enumerate(default_columns_var)})
                    else:
                        df = pd.read_csv(io.StringIO(pasted_text_var), sep='\t' if '\t' in pasted_text_var else ',',
                                         header=None)
                        new_data = pd.DataFrame()
                        # Map pasted columns to expected columns based on column count
                        # Expected columns: default_columns_var
                        pasted_cols = df.shape[1] if df is not None else 0
                        n_rows = len(df)
                        # Default mapping: column index matches expected column order
                        # Fill any missing columns with empty values
                        for i, col in enumerate(default_columns_var):
                            if i < pasted_cols:
                                new_data[col] = df.iloc[:, i]
                            else:
                                new_data[col] = [''] * n_rows
                        st.session_state.edited_df_corr_p = new_data
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        # ── Run button ──
        _btn_lbl = {"Solve": "🔍 Solve Strikes", "Price": "💰 Price Package", "Generate FPFs": "🔧 Generate FPFs"}[
            pricing_mode]
        if st.button(_btn_lbl, type="primary", use_container_width=True, key="p_run_var"):
            if st.session_state.edited_df_corr_p.empty:
                st.warning("Add data first.")
            elif is_generate_fpf_var:
                try:
                    tickers_fpf = [t for t in
                                   st.session_state.edited_df_corr_p["Variance Asset"].astype(str).str.strip().tolist()
                                   if t and t != 'nan']
                    matu_ex, matu_stl = calculate_payment_dates(maturity_date_p, tickers_fpf[0])
                    strikes_fpf = (st.session_state.edited_df_corr_p["Strikes (%)"].astype(
                        float) / 100).tolist() if 'Strikes (%)' in st.session_state.edited_df_corr_p.columns else [
                                                                                                                      0.15] * len(
                        tickers_fpf)
                    all_schedules = [
                        set(create_schedule(strike_date_p.strftime("%Y-%m-%d"), maturity_date_p.strftime("%Y-%m-%d"),
                                            "1D", get_trading_calendar(t), "MF")) for t in tickers_fpf]
                    common_dates = sorted(set.intersection(*all_schedules))
                    st.success(f"✅ {len(common_dates) - 1} obs dates")
                    fpf_results = []
                    for idx, ticker in enumerate(tickers_fpf):
                        fpf_string = vp.generate_fpf_vol_with_dates(ticker=ticker, matu_stl=matu_stl, matu_ex=matu_ex,
                                                                    strike_date=strike_date_p, strike=strikes_fpf[idx],
                                                                    weight=1.0, is_note=False,
                                                                    observation_dates=common_dates)
                        fpf_results.append(
                            {'Ticker': ticker, 'Strike': f"{strikes_fpf[idx] * 100:.2f}%", 'FPF': fpf_string})
                    for r in fpf_results:
                        with st.expander(f"{r['Ticker']} ({r['Strike']})"):
                            st.code(r['FPF'], language="xml")
                except Exception as e:
                    st.error(f"❌ {e}")
            else:
                # Price/Solve mode
                progress_bar = st.progress(0, text="Starting...")
                status_text = st.empty()

                def update_progress(progress_info):
                    if isinstance(progress_info, dict):
                        s = progress_info.get('status', '')
                        if s == 'pricing_batch':
                            batch = progress_info['batch']
                            total = progress_info['total_batches']
                            # Clamp: engine may have more batches than initially estimated
                            batch_display = min(batch, total)
                            # Pricing batches occupy 10%–70% of progress bar
                            pct = 0.10 + (min(batch, total) / max(total, 1)) * 0.60
                            progress_bar.progress(min(pct, 1.0), text=f"Pricing batch {batch_display} / {total}...")
                        elif s in ('completed', 'started', 'pricing'):
                            pct = 0.70 + (progress_info['completed'] / max(progress_info['total'], 1)) * 0.30
                            progress_bar.progress(min(pct, 1.0),
                                                  text=f"{progress_info.get('message', progress_info.get('ticker', ''))} ({progress_info['completed']}/{progress_info['total']})")
                        elif s == 'failed':
                            status_text.warning(
                                f"⚠️ {progress_info.get('ticker', '')} - {progress_info.get('message', '')}")
                        elif s == 'phase':
                            progress_bar.progress(0.05, text=progress_info.get('message', ''))

                try:
                    tickers = [t for t in
                               st.session_state.edited_df_corr_p["Variance Asset"].astype(str).str.strip().tolist() if
                               t and t != 'nan']
                    if not tickers:
                        st.warning("Add data first.")
                    else:
                        # Build product config (structure only — no business data)
                        config = DispersionConfig(
                            product_type=ProductType.VAR_SWAP_CORRIDOR,
                            cross_corridor=is_x_corr_var,
                            barrier_up=uvar_var,
                            barrier_down=dvar_var,
                            is_capped=is_capped_var,
                        )
                        # Build DataFrame with required columns for solve()/price()
                        pricing_df = st.session_state.edited_df_corr_p.copy()
                        if 'Corridor Condition Asset' not in pricing_df.columns:
                            pricing_df['Corridor Condition Asset'] = '.SPX' if is_x_corr_var else pricing_df[
                                'Variance Asset']
                        if 'Currency' not in pricing_df.columns:
                            pricing_df['Currency'] = ''  # empty → PricingEngine auto-detects from portal
                        # ── Build LCM properties dict from UI inputs ──
                        _lcm_props = {
                            "AggregatorType": "Basket",
                            "CallSkew": [lcm_call_skew_var],
                            "PutSkew": [lcm_put_skew_var],
                            "LambdaAtm": [lcm_lambda_atm_var],
                            "LambdaFromRho0": lcm_lambda_rho0_var,
                            "LambdaPricing": lcm_lambda_pricing_var,
                        } if apply_lcm_var else None
                        if is_solve_var:
                            results = solve(
                                df=pricing_df,
                                config=config,
                                last_obs_date=maturity_date_p,
                                strike_date=strike_date_p,
                                eqeq_lambda=eqeq_lambda_var,
                                correl_floor=correl_floor_var,
                                eqfx_shift=eqfx_shift_var,
                                vol_mode=vol_mode_var,
                                correl_input_method=correl_input_method_var,
                                model_name=model_name_param,
                                use_lsv=apply_lsv_var,
                                lsv_params=st.session_state.get('edited_df_lsv_p') if apply_lsv_var else None,
                                lsv_correl_bump=correl_bump_lsv_var,
                                lsv_correl_bump_style=correl_bump_lsv_style_var,
                                use_lcm=apply_lcm_var,
                                lcm_properties=_lcm_props,
                                progress_callback=update_progress,
                            )
                        else:
                            results = price(
                                df=pricing_df,
                                config=config,
                                last_obs_date=maturity_date_p,
                                strike_date=strike_date_p,
                                eqeq_lambda=eqeq_lambda_var,
                                correl_floor=correl_floor_var,
                                eqfx_shift=eqfx_shift_var,
                                vol_mode=vol_mode_var,
                                correl_input_method=correl_input_method_var,
                                model_name=model_name_param,
                                use_lsv=apply_lsv_var,
                                lsv_params=st.session_state.get('edited_df_lsv_p') if apply_lsv_var else None,
                                lsv_correl_bump=correl_bump_lsv_var,
                                lsv_correl_bump_style=correl_bump_lsv_style_var,
                                use_lcm=apply_lcm_var,
                                lcm_properties=_lcm_props,
                                progress_callback=update_progress,
                            )
                        if results and hasattr(results, 'success') and results.success:
                            st.success("✅ Pricing completed")
                            if hasattr(results, 'failed_tickers') and results.failed_tickers:
                                st.warning(f"⚠️ Failed: {', '.join(results.failed_tickers)}")
                            if hasattr(results, 'results_df') and results.results_df is not None:
                                st.session_state['pricing_results_df'] = results.results_df.copy()
                                st.session_state['original_results'] = results.results_df.copy()
                        else:
                            # ── DEBUG: show full error details ──
                            error_msg = getattr(results, 'error', None) or 'No results returned'
                            st.error(f"❌ {error_msg}")
                            if results:
                                # Show per-ticker errors if available
                                ticker_results = getattr(results, 'ticker_results', None)
                                if ticker_results:
                                    with st.expander("🔍 Per-ticker error details", expanded=True):
                                        for tr in ticker_results:
                                            status = "✅" if tr.success else "❌"
                                            st.text(f"{status} {tr.ticker}: {tr.error or 'OK'}")
                                # Show the results_df even on failure (partial results)
                                if hasattr(results,
                                           'results_df') and results.results_df is not None and not results.results_df.empty:
                                    with st.expander("📋 Partial results"):
                                        st.dataframe(results.results_df, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ {e}")
                    st.exception(e)
        # ── Persistent Result Matrix display (survives rerun from download button) ──
        if 'pricing_results_df' in st.session_state and st.session_state['pricing_results_df'] is not None:
            _highlight_rows = [
                'Strike Cross Corr Cap Priced LV (%)',
                'Strike Cross Corr Cap Priced LSV (%)',
                'Strike Cross Corr Cap Priced LCM (%)',
                'Strike Mono Corr Cap Priced LV (%)',
                'Strike Mono Corr Cap Priced LSV (%)',
                'RA (%)',
                'Correlation',
            ]
            _df_t = st.session_state['pricing_results_df'].T
            # Ensure all values are strings to avoid Arrow serialization errors (mixed int/str/None)
            _df_t = _df_t.fillna('').astype(str)

            def _highlight_key_rows(row):
                if row.name in _highlight_rows:
                    return ['background-color: #fff3cd'] * len(row)
                return [''] * len(row)

            _styled = _df_t.style.apply(_highlight_key_rows, axis=1)
            st.dataframe(_styled, use_container_width=True, height=600)
            # ── Export to Excel button (horizontal: tickers as rows, metrics as columns) ──
            from functions.dispersion.export_excel import export_result_matrix
            _excel_bytes = export_result_matrix(st.session_state['pricing_results_df'], horizontal=True)
            if _excel_bytes:
                st.download_button(
                    "📥 Download Result Matrix",
                    data=_excel_bytes,
                    file_name="Result_Matrix.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
    elif product_type == "Vol Swap":
        is_solve_vol = pricing_mode == "Solve"
        is_generate_fpf_vol = pricing_mode == "Generate FPFs"
        # ── Config row ──
        _v1, _v2, _v3, _v4, _v5, _v6 = st.columns(6)
        with _v1:
            is_note_vol = st.toggle("Note", value=True, key="p_is_note_vol", disabled=is_solve_vol,
                                    help="±10% global caps")
        with _v2:
            currency_vol = st.selectbox("Ccy", ["EUR", "USD", "CHF"], key="p_currency_vol")
        with _v3:
            eqeq_lambda_vol = st.number_input("EqEq λ", value=0.10, key="p_eqeq_vol", disabled=is_solve_vol)
        with _v4:
            correl_floor_vol = st.number_input("Floor", value=0.0, key="p_floor_vol", disabled=is_solve_vol)
        with _v5:
            eqfx_shift_vol = st.number_input("EqFx", value=-0.05, key="p_eqfx_vol", disabled=is_solve_vol)
        with _v6:
            if is_generate_fpf_vol:
                use_common_cal = st.toggle("Common Cal", value=True, key="p_common_cal",
                                           help="Intersection of calendars")
            elif is_solve_vol:
                target_value_vol = st.number_input("Target (%)", value=0.0, step=0.1, key="p_target_vol") / 100
        # ── Underlyings editor ──
        st.divider()
        if is_solve_vol:
            default_columns_vol = ['Variance Asset']
        elif is_generate_fpf_vol:
            default_columns_vol = ['Variance Asset', 'Strikes (%)', 'Weights (%)']
        else:
            default_columns_vol = ['Variance Asset', 'Strikes', 'Weights']
        if 'edited_df_volswap_p' not in st.session_state or set(
                st.session_state.edited_df_volswap_p.columns.tolist()) != set(default_columns_vol):
            st.session_state.edited_df_volswap_p = pd.DataFrame({col: [] for col in default_columns_vol})
        st.data_editor(st.session_state.edited_df_volswap_p, num_rows="dynamic", use_container_width=True,
                       key="p_volswap_editor")
        _vp_col, _vb_col = st.columns([4, 1])
        with _vp_col:
            pasted_text_vol = st.text_area("Paste data:", height=68, key="p_paste_vol")
        with _vb_col:
            st.write("")
            st.write("")
            if st.button("📋 Fill", key="p_fill_vol") and pasted_text_vol:
                try:
                    if '\n' not in pasted_text_vol:
                        values = pasted_text_vol.split('\t') if '\t' in pasted_text_vol else pasted_text_vol.split(',')
                        st.session_state.edited_df_volswap_p = pd.DataFrame(
                            {col: [values[i] if i < len(values) else ''] for i, col in enumerate(default_columns_vol)})
                    else:
                        df = pd.read_csv(io.StringIO(pasted_text_vol), sep='\t' if '\t' in pasted_text_vol else ',',
                                         header=None)
                        new_data = pd.DataFrame()
                        for i, col in enumerate(default_columns_vol):
                            new_data[col] = df.iloc[:, i] if i < df.shape[1] else [''] * len(df)
                        st.session_state.edited_df_volswap_p = new_data
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        # ── Run button ──
        _vbtn = {"Solve": "🔍 Solve Strikes", "Price": "💰 Price Vol Swap", "Generate FPFs": "🔧 Generate FPFs"}[
            pricing_mode]
        if st.button(_vbtn, type="primary", use_container_width=True, key="p_run_vol"):
            tickers_vol = [t for t in
                           st.session_state.edited_df_volswap_p["Variance Asset"].astype(str).str.strip().tolist() if
                           t and t != 'nan']
            if not tickers_vol:
                st.error("No valid tickers")
            else:
                matu_ex_vol, matu_stl_vol = calculate_payment_dates(maturity_date_p, tickers_vol[0])
                st.caption(f"📅 Ex={matu_ex_vol.strftime('%Y-%m-%d')}, Stl={matu_stl_vol.strftime('%Y-%m-%d')}")
                if is_solve_vol:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def progress_cb_vol(info):
                        progress_bar.progress(info['completed'] / info['total'])
                        status_text.text(f"{info['ticker']} ({info['completed']}/{info['total']})")

                    results_df_vol = vp.solve_volswap_strikes_multithreaded(
                        tickers=tickers_vol, maturity_date=maturity_date_p, strike_date=strike_date_p,
                        currency=currency_vol, target_value=target_value_vol,
                        eqeq_lambda=eqeq_lambda_vol, eqfx_shift=eqfx_shift_vol, progress_callback=progress_cb_vol)
                    progress_bar.progress(1.0)
                    status_text.text("✅ Done")
                    _s = len(results_df_vol[results_df_vol['Status'] == 'Success'])
                    _f = len(results_df_vol) - _s
                    st.caption(f"✅ {_s} solved | ⚠️ {_f} failed")
                    st.dataframe(results_df_vol.T, use_container_width=True, height=600)
                    st.session_state['volswap_solved_strikes'] = results_df_vol
                elif is_generate_fpf_vol:
                    try:
                        strikes_vol = (st.session_state.edited_df_volswap_p["Strikes (%)"].astype(float) / 100).tolist()
                        weights_vol = (st.session_state.edited_df_volswap_p["Weights (%)"].astype(float) / 100).tolist()
                        use_common = st.session_state.get('p_common_cal', True)
                        if use_common:
                            all_schedules = [set(create_schedule(strike_date_p.strftime("%Y-%m-%d"),
                                                                 maturity_date_p.strftime("%Y-%m-%d"), "1D",
                                                                 get_trading_calendar(t), "MF")) for t in tickers_vol]
                            common_dates = sorted(set.intersection(*all_schedules))
                            num_obs = len(common_dates) - 1
                            st.success(f"✅ Common calendar: {num_obs} obs dates")
                        else:
                            common_dates = None
                        fpf_results = []
                        for idx, ticker in enumerate(tickers_vol):
                            if not use_common:
                                obs_dates = create_schedule(strike_date_p.strftime("%Y-%m-%d"),
                                                            maturity_date_p.strftime("%Y-%m-%d"), "1D",
                                                            get_trading_calendar(ticker), "MF")
                                num_obs = len(obs_dates) - 1
                            else:
                                obs_dates = common_dates
                            fpf_string = vp.generate_fpf_vol_with_dates(
                                ticker=ticker, matu_stl=matu_stl_vol, matu_ex=matu_ex_vol,
                                strike_date=strike_date_p, strike=strikes_vol[idx],
                                weight=weights_vol[idx], is_note=is_note_vol, observation_dates=obs_dates)
                            fpf_results.append(
                                {'Ticker': ticker, 'Strike Mono Var Swap (%)': f"{strikes_vol[idx] * 100:.2f}%",
                                 'Weight (%)': f"{weights_vol[idx] * 100:.2f}%", 'Obs': num_obs, 'FPF': fpf_string})
                        results_df_fpf = pd.DataFrame(fpf_results)
                        st.dataframe(results_df_fpf[['Ticker', 'Strike Mono Var Swap (%)', 'Weight (%)', 'Obs']].T,
                                     use_container_width=True, height=600)
                        all_fpfs = "\n\n".join(
                            [f"{'=' * 60}\nTICKER: {r['Ticker']}\n{r['FPF']}" for _, r in results_df_fpf.iterrows()])
                        st.download_button("📥 Download All FPFs", data=all_fpfs,
                                           file_name=f"volswap_fpfs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                           mime="text/plain", use_container_width=True)
                        st.session_state['generated_fpfs'] = fpf_results
                    except Exception as e:
                        st.error(f"❌ {e}")
                        st.exception(e)
                else:
                    try:
                        res = vp.create_and_price_fpf(
                            tickers=tickers_vol, maturity_date=maturity_date_p, strike_date=strike_date_p,
                            strikes=(st.session_state.edited_df_volswap_p["Strikes"].astype(float) / 100).tolist(),
                            weights=(st.session_state.edited_df_volswap_p["Weights"].astype(float) / 100).tolist(),
                            is_note=is_note_vol, currency=currency_vol,
                            eqeq_lambda=eqeq_lambda_vol, eqfx_shift=eqfx_shift_vol)
                        if res['success']:
                            st.success(
                                f"✅ Fair Value: **{res['fair_value'] * 100:.4f}%** | {'Note' if res['is_note'] else 'OTC'} | {res['currency']}")
                            basket_df = pd.DataFrame({'Ticker': res['tickers'],
                                                      'Strike': [f"{s * 100:.2f}%" for s in res['strikes']],
                                                      'Weight': [f"{w * 100:.2f}%" for w in res['weights']]})
                            st.dataframe(basket_df, use_container_width=True)
                            with st.expander("📄 FPF"):
                                st.code(res['fpf_string'], language='text')
                        else:
                            st.error(f"❌ {res['error']}")
                    except Exception as e:
                        st.error(f"❌ {e}")
                        st.exception(e)
    # ══════════════════════════════════════════════════════════════════════════
    # PAIRWISE COVAR
    # ══════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Json Booking Generation
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("QRT JSON Booking Generation")
    # ── Session state init ────────────────────────────────────────────────
    if "jb_hdr_edit" not in st.session_state:
        st.session_state.jb_hdr_edit = pd.DataFrame(JB_EDITABLE_DEFAULTS, columns=["Field", "Value"])
    if "jb_hdr_locked" not in st.session_state:
        st.session_state.jb_hdr_locked = pd.DataFrame(JB_LOCKED_DEFAULTS, columns=["Field", "Value"])
    if "jb_legs_df" not in st.session_state:
        st.session_state.jb_legs_df = pd.DataFrame({c: [] for c in JB_LEGS_COLS})
    else:
        if list(st.session_state.jb_legs_df.columns) != JB_LEGS_COLS:
            st.session_state.jb_legs_df = pd.DataFrame({c: [] for c in JB_LEGS_COLS})
    if "jb_out_json" not in st.session_state:
        st.session_state.jb_out_json = None
    if "jb_out_preview" not in st.session_state:
        st.session_state.jb_out_preview = None
    # ── Header section ────────────────────────────────────────────────────
    st.subheader("Header")
    with st.form("jb_main_form"):
        jb_hdr_edited = st.data_editor(
            st.session_state.jb_hdr_edit,
            num_rows="fixed",
            use_container_width=True,
            key="jb_hdr_edit_editor",
            column_config={"Field": st.column_config.TextColumn(disabled=True)}
        )
        st.dataframe(st.session_state.jb_hdr_locked, use_container_width=True, hide_index=True)
        st.subheader("Legs")
        jb_product_type_selector = st.selectbox(
            "Product Type",
            options=["Auto-detect", "VolSwap", "CorridorVarSwap", "CrossCorridorVarSwap"],
            index=0,
            key="jb_product_type_selector"
        )
        jb_legs_edited = st.data_editor(
            st.session_state.jb_legs_df,
            num_rows="dynamic",
            use_container_width=True,
            key="jb_legs_editor"
        )
        st.divider()
        jb_generate_clicked = st.form_submit_button("Generate JSON", type="primary", use_container_width=True)
    # ── Paste helpers ─────────────────────────────────────────────────────
    with st.expander("Paste from clipboard"):
        jb_hdr_txt = st.text_area("Header: Field,Value", height=100, key="jb_hdr_paste")
        if st.button("Fill header", use_container_width=True, key="jb_fill_hdr"):
            dfp = jb_df_from_paste(jb_hdr_txt, ["Field", "Value"])
            if dfp is not None:
                base = st.session_state.jb_hdr_edit.copy()
                for _, r in dfp.iterrows():
                    f = str(r["Field"]).strip()
                    v = str(r["Value"])
                    if (base["Field"] == f).any():
                        base.loc[base["Field"] == f, "Value"] = v
                    else:
                        base = pd.concat([base, pd.DataFrame([[f, v]], columns=["Field", "Value"])], ignore_index=True)
                st.session_state.jb_hdr_edit = base
                st.rerun()
        jb_legs_txt = st.text_area(
            "Legs: RefAsset,CorridorAsset,StrikeCross,StrikeMono,WeightInput,DownBarrier,UpBarrier,Bump(%)",
            height=100, key="jb_legs_paste"
        )
        if st.button("Fill legs", use_container_width=True, key="jb_fill_legs"):
            dfp = jb_df_from_paste(jb_legs_txt, JB_LEGS_COLS)
            if dfp is not None:
                st.session_state.jb_legs_df = dfp
                st.rerun()
    # ── Generate JSON ─────────────────────────────────────────────────────
    if jb_generate_clicked:
        st.session_state.jb_hdr_edit = jb_hdr_edited
        st.session_state.jb_legs_df = jb_legs_edited
        h_edit = {str(r["Field"]).strip(): str(r["Value"]) for _, r in jb_hdr_edited.iterrows()}
        h_lock = {str(r["Field"]).strip(): str(r["Value"]) for _, r in st.session_state.jb_hdr_locked.iterrows()}
        with st.spinner("Building FPFs and generating JSON..."):
            json_str, preview_df = generate_booking_json(
                h_edit, h_lock, st.session_state.jb_legs_df, jb_product_type_selector
            )
        if json_str:
            st.session_state.jb_out_json = json_str
            st.session_state.jb_out_preview = preview_df
        else:
            st.warning("No legs to process or generation failed.")
    # ── Output ────────────────────────────────────────────────────────────
    if st.session_state.jb_out_json:
        st.divider()
        st.subheader("Output JSON")
        _jb_name = st.session_state.jb_hdr_edit.loc[
            st.session_state.jb_hdr_edit["Field"] == "Name", "Value"
        ]
        _jb_filename = (_jb_name.values[0] if len(_jb_name) else "QRT_booking").replace(" ", "_")
        st.download_button(
            label="Download JSON",
            data=st.session_state.jb_out_json,
            file_name=f"{_jb_filename}.json",
            mime="application/json",
            use_container_width=True,
            key="jb_download"
        )
        st.json(st.session_state.jb_out_json)
    if st.session_state.jb_out_preview is not None:
        st.divider()
        st.subheader("Preview")
        st.dataframe(st.session_state.jb_out_preview, use_container_width=True, hide_index=True)
