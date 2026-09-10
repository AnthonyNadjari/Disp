"""
Chart generation and email export for dispersion backtest results.

Two rendering paths:
  1. Interactive Plotly figures (for Streamlit display)
  2. Matplotlib static rendering (for email — no kaleido/orca, no disk, no subprocess)

All email functions are pure Python — no Streamlit dependency.
"""

from __future__ import annotations

import base64
import io
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UNIFIED COLOUR PALETTE — matching original Gaia_PP disp_bt_functions_graphs.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Chart line colours (original Gaia_PP)
BARCLAYS_LINE_PRIMARY = '#2980b9'   # Main line (60D, backtest) — darker blue
BARCLAYS_LINE_ENTRY = '#3498db'     # Entry point line — bright blue
BARCLAYS_NET_PNL = '#007bff'        # Net PNL in split graph

# Split graph fills (original Gaia_PP)
SPLIT_SHORT_COLOR = 'rgba(165, 165, 165, 0.6)'  # Grey — short leg
SPLIT_LONG_COLOR = 'rgba(255, 193, 7, 0.6)'     # Yellow/gold — long leg
SPLIT_NET_COLOR = 'rgba(52, 152, 219, 0.7)'     # Blue — net PNL fill

# Title/heading colour
BARCLAYS_NAVY = '#003c71'

# Additional brand colours for UI elements
BARCLAYS_TEAL = '#00897b'   # Teal — for positive outcomes
BARCLAYS_PLUM = '#7b1fa2'   # Plum — for highlights
BARCLAYS_CYAN = '#00bcd4'   # Cyan — for secondary highlights

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INTERACTIVE CHARTS (Plotly — for Streamlit display)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def standardize_chart_display(fig, title, show_legend=True, is_entry_point=False):
    """Apply consistent Barclays styling to interactive charts."""
    for trace in fig.data:
        if hasattr(trace, 'line') and trace.line:
            if hasattr(trace, 'name') and ('Net PNL' in str(trace.name) or 'Entry' in str(trace.name) or len(fig.data) == 1):
                trace.update(line_width=2.5)
            else:
                trace.update(line_width=1.5)

    # Entry point y-axis: detect vol swap vs corridor from data range
    # Also collect all_y once for both label and range calculation
    entry_all_y = []
    if is_entry_point:
        for trace in fig.data:
            if hasattr(trace, 'y') and trace.y is not None:
                vals = [v for v in trace.y if v is not None and not (isinstance(v, float) and v != v)]
                entry_all_y.extend(vals)
        if entry_all_y and max(abs(v) for v in entry_all_y) < 50:
            yaxis_title = "Vol spread"
        else:
            yaxis_title = "Strikes spread"
    else:
        yaxis_title = "Cash payout per unit of vega notional"

    # Compute y-axis range from actual data for tight, readable scaling
    yaxis_kwargs = dict(
        tickfont=dict(size=13), gridcolor='#d9d9d9', gridwidth=0.8,
        showgrid=True, showline=True, linecolor='#cccccc', linewidth=1,
        zeroline=True, zerolinecolor='#999999', zerolinewidth=1,
    )

    if is_entry_point and entry_all_y:
        # Tight axis: exactly fit the data range (no padding beyond min/max)
        y_min, y_max = min(entry_all_y), max(entry_all_y)
        spread = y_max - y_min if y_max != y_min else abs(y_max) * 0.2 or 1.0
        padding = spread * 0.02  # minimal 2% breathing room
        yaxis_kwargs['range'] = [y_min - padding, y_max + padding]

    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color=BARCLAYS_NAVY, family="Arial")),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=show_legend,
        font=dict(size=15, family="Arial", color='#333333'),
        xaxis_title="Date",
        yaxis_title=dict(text=yaxis_title, font=dict(size=14, color='#555555')),
        xaxis=dict(tickfont=dict(size=13), gridcolor='#d9d9d9', gridwidth=0.8,
                   showgrid=True, showline=True, linecolor='#cccccc', linewidth=1),
        yaxis=yaxis_kwargs,
        height=500,
        hovermode='x unified',
        legend=dict(font=dict(size=13)),
    )
    return fig

def plot_main_backtest(df_res: pd.DataFrame, active_count: Optional[pd.Series] = None,
                       is_cross_corridor: bool = False) -> go.Figure:
    """Main backtest chart with P&L line + optional active stocks on independent secondary y-axis."""
    from plotly.subplots import make_subplots

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=df_res.index, y=df_res["Result"],
        name="Net P&L", line=dict(color=BARCLAYS_LINE_PRIMARY, width=3), mode='lines'
    ), secondary_y=False)

    if active_count is not None and len(active_count) > 0:
        fig.add_trace(go.Scatter(
            x=active_count.index, y=active_count.values,
            name="Active Underlyings",
            line=dict(color='#3498db', width=2, dash='dot'),
            mode='lines',
            opacity=0.8,
        ), secondary_y=True)

    title = "Backtest Performance — Cross Corridor" if is_cross_corridor else "Backtest Performance"
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color=BARCLAYS_NAVY, family="Arial")),
        height=550, hovermode='x unified',
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(size=14, family="Arial", color='#333333'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=13)),
        xaxis=dict(tickfont=dict(size=13), gridcolor='#d9d9d9', gridwidth=0.8,
                   showgrid=True, showline=True, linecolor='#cccccc', linewidth=1),
        yaxis=dict(tickfont=dict(size=13), gridcolor='#d9d9d9', gridwidth=0.8,
                   showgrid=True, showline=True, linecolor='#cccccc', linewidth=1),
    )
    fig.update_yaxes(title_text="Cash payout per unit of vega notional", secondary_y=False,
                     title_font=dict(size=14, color='#555555'))
    if active_count is not None and len(active_count) > 0:
        max_active = int(active_count.max()) if hasattr(active_count, 'max') else int(max(active_count.values))
        fig.update_yaxes(title_text="# Active Stocks", secondary_y=True,
                         showgrid=False, range=[0, max_active + 2],
                         title_font=dict(size=14, color='#3498db'),
                         tickfont=dict(size=13, color='#3498db'))
    else:
        fig.update_yaxes(title_text="# Active Stocks", secondary_y=True,
                         showgrid=False, rangemode='tozero',
                         title_font=dict(size=14, color='#3498db'),
                         tickfont=dict(size=13, color='#3498db'))
    return fig


def plot_per_stock_contributions(pnl_matrix: pd.DataFrame, weights: Dict[str, float],
                                  is_cross_corridor: bool = False) -> go.Figure:
    """
    Plot individual stock P&L contributions as stacked area chart.
    Used when 'Show Individual Legs' is toggled for cross-corridor (no short leg split).

    Args:
        pnl_matrix: DataFrame with columns=tickers, index=dates, values=per-stock P&L
        weights: {ticker: weight} used for sorting/coloring
        is_cross_corridor: styling flag
    """
    fig = go.Figure()

    # Sort by absolute weight descending
    sorted_tickers = sorted(pnl_matrix.columns, key=lambda t: abs(weights.get(t, 0)), reverse=True)

    # Use a qualitative color palette
    colors = [
        '#00AEEF', '#003D6B', '#2ECC71', '#E74C3C', '#9B59B6',
        '#F39C12', '#1ABC9C', '#E91E63', '#3F51B5', '#FF9800',
        '#607D8B', '#795548', '#CDDC39', '#00BCD4', '#FF5722',
        '#8BC34A', '#673AB7', '#FFC107', '#009688', '#4CAF50',
    ]

    for i, ticker in enumerate(sorted_tickers):
        if ticker not in pnl_matrix.columns:
            continue
        color = colors[i % len(colors)]
        cumulative = pnl_matrix[ticker].cumsum()
        fig.add_trace(go.Scatter(
            x=pnl_matrix.index,
            y=cumulative,
            mode='lines',
            name=f"{ticker} ({weights.get(ticker, 0)*100:.0f}%)",
            line=dict(color=color, width=1.5),
        ))

    # Add total
    total = pnl_matrix.sum(axis=1).cumsum()
    fig.add_trace(go.Scatter(
        x=pnl_matrix.index,
        y=total,
        mode='lines',
        name='Total',
        line=dict(color=BARCLAYS_NAVY, width=3),
    ))

    title = "Per-Stock Cumulative P&L — Cross Corridor" if is_cross_corridor else "Per-Stock Cumulative P&L"
    fig = standardize_chart_display(fig, title, show_legend=True)
    fig.update_layout(height=600)
    return fig

def split_graph(df_res: pd.DataFrame, time_span: str = "", is_cross_corridor: bool = False,
                start_date: Optional[date] = None) -> Optional[go.Figure]:
    """Split graph with filled areas for long/short/net. Returns None if no short leg."""
    if start_date is None:
        # Use full data range — show all available history
        start_date = df_res.index.min() if len(df_res) > 0 else date.today() - relativedelta(years=5)

    # Ensure index is pd.DatetimeIndex for safe comparison
    if not isinstance(df_res.index, pd.DatetimeIndex):
        df_res.index = pd.to_datetime(df_res.index)

    nearest_idx = df_res.index.get_indexer([pd.Timestamp(start_date)], method='nearest')[0]
    df_plot = df_res.iloc[max(0, nearest_idx):].dropna()

    if len(df_plot) == 0:
        return None

    # For cross-corridor: short leg is zero (net is computed per-pair in engine).
    # Still show the split if at least the long leg has data.
    long_has_data = df_plot.iloc[:, 0].abs().max() > 0.001
    short_has_data = df_plot.iloc[:, 1].abs().max() > 0.001

    if is_cross_corridor:
        # Cross-corridor: only show split if there are actual short positions (negative weights).
        # When all weights are positive, the split is identical to the aggregate — skip it.
        if not short_has_data:
            return None
    else:
        # Standard dispersion: need both legs to show a split
        if not (long_has_data and short_has_data):
            return None

    fig = go.Figure()

    if short_has_data:
        # Standard split: show both legs + net
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot.iloc[:, 1], mode='lines',
            name=df_plot.columns[1], fill='tozeroy',
            fillcolor=SPLIT_SHORT_COLOR, line_color=SPLIT_SHORT_COLOR
        ))
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot.iloc[:, 0], mode='lines',
            name=df_plot.columns[0], fill='tozeroy',
            fillcolor=SPLIT_LONG_COLOR, line_color=SPLIT_LONG_COLOR
        ))
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["Result"], mode='lines',
            name='Net PNL', line_color=BARCLAYS_NET_PNL, line_width=2,
            fill='tozeroy', fillcolor=SPLIT_NET_COLOR
        ))
    else:
        # Cross-corridor or no short: just show the result line
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["Result"], mode='lines',
            name='Net PNL (per-pair)', line_color=BARCLAYS_NET_PNL, line_width=3,
            fill='tozeroy', fillcolor=SPLIT_NET_COLOR
        ))

    title_suffix = " — Cross Corridor" if is_cross_corridor else ""
    timespan_suffix = f" ({time_span})" if time_span else ""
    title = f'Payout at maturity — Split{title_suffix}{timespan_suffix}'
    fig = standardize_chart_display(fig, title, show_legend=True)
    return fig

def plot_60d(df_res_60d: pd.DataFrame, is_cross_corridor: bool = False,
             start_date: Optional[date] = None) -> go.Figure:
    """
    Create 3M Carry chart.
    df_res_60d has columns [Long Leg, Short Leg, Result] — we plot Result.
    """
    if start_date is None:
        # Use full data range — show all available history
        start_date = df_res_60d.index.min() if len(df_res_60d) > 0 else date.today() - relativedelta(years=5)

    # Get the Result series — this is the per-trade P&L for 60-day maturity
    result_col = df_res_60d["Result"]

    # Ensure index is pd.DatetimeIndex for safe comparison
    if not isinstance(result_col.index, pd.DatetimeIndex):
        result_col.index = pd.to_datetime(result_col.index)

    # Filter to start_date
    mask = result_col.index >= pd.Timestamp(start_date)
    if mask.sum() == 0:
        # If nothing after start_date, use all data
        plot_data = result_col.dropna()
    else:
        plot_data = result_col[mask].dropna()

    # Remove zeros at the start (before the backtest produces real values)
    first_nonzero = (plot_data != 0).idxmax() if (plot_data != 0).any() else plot_data.index[0]
    plot_data = plot_data.loc[first_nonzero:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_data.index, y=plot_data.values,
        mode='lines', name='3M Carry P&L',
        line=dict(color=BARCLAYS_LINE_PRIMARY, width=2.5)
    ))

    title = '3M Carry Analysis — Cross Corridor' if is_cross_corridor else '3M Carry Analysis'
    fig = standardize_chart_display(fig, title, show_legend=False)
    return fig

def graph_sectorial(tickers: List[str], short_tickers: Optional[List[str]] = None,
                    weights: Optional[List[float]] = None,
                    short_weights: Optional[List[float]] = None) -> Dict:
    """Create sectorial pie chart(s) via Bloomberg. Excludes index tickers from short leg pie.
    NOTE: Pie chart is NOT weight-adjusted — each stock counts equally (as stated in email disclaimer)."""

    try:
        from xbbg import blp

        # Filter out index tickers (contain "Index" in name) from short leg
        if short_tickers:
            short_tickers = [t for t in short_tickers if 'Index' not in str(t)]

        all_tickers_combined = tickers if (not short_tickers or len(short_tickers) == 0) else list(set(tickers + short_tickers))
        has_index = any('Index' in str(t) for t in all_tickers_combined)
        is_dual = short_tickers is not None and len(short_tickers) > 0 and not has_index

        # Use cached sector lookup if in Streamlit
        try:
            import streamlit as _st
            @_st.cache_data(ttl=600, show_spinner=False)
            def _sector_lookup(_key, _tickers):
                return blp.bdp(_tickers, "INDUSTRY_SECTOR")
            df = _sector_lookup(str(sorted(all_tickers_combined)), all_tickers_combined)
        except Exception:
            df = blp.bdp(all_tickers_combined, "INDUSTRY_SECTOR")
        if df is None or df.empty:
            return {'error': "No sector data available"}

        df = df.reset_index()
        if 'industry_sector' not in df.columns:
            return {'error': "Industry sector data not available"}
        df = df.dropna(subset=['industry_sector'])
        if len(df) == 0:
            return {'error': "No valid sector data found"}

        def _style_pie(fig, title):
            fig.update_traces(
                textinfo='label+percent',
                textfont_size=14,
                textposition='auto',
                marker=dict(line=dict(color='white', width=2)),
                hole=0,
                domain=dict(x=[0.05, 0.95], y=[0.05, 0.95]),
            )
            fig.update_layout(
                title=dict(text=title, font=dict(size=18, color=BARCLAYS_NAVY), x=0.5, xanchor='center', y=0.98),
                showlegend=True,
                legend=dict(font=dict(size=11), orientation='h', yanchor='top', y=-0.05, xanchor='center', x=0.5),
                paper_bgcolor='white',
                plot_bgcolor='white',
                margin=dict(t=50, b=100, l=10, r=10),
                height=600,
                width=600,
                autosize=False,
            )
            return fig

        if is_dual:
            df_long = df[df['index'].isin(tickers)].copy()
            df_short = df[df['index'].isin(short_tickers)].copy()

            # Equal weight — each stock counts as 1 (not weight-adjusted)
            df_long['weight'] = 1.0
            df_short['weight'] = 1.0

            df_long_g = df_long.groupby('industry_sector')['weight'].sum().reset_index(name='Repartition')
            df_short_g = df_short.groupby('industry_sector')['weight'].sum().reset_index(name='Repartition')

            fig_long = px.pie(df_long_g, values='Repartition', names='industry_sector',
                              color_discrete_sequence=px.colors.sequential.Blues[::-1])
            fig_short = px.pie(df_short_g, values='Repartition', names='industry_sector',
                               color_discrete_sequence=px.colors.sequential.Greys[::-1])
            _style_pie(fig_long, 'Long Basket — Sector Breakdown')
            _style_pie(fig_short, 'Short Basket — Sector Breakdown')
            return {'is_dual': True, 'fig_long': fig_long, 'fig_short': fig_short}
        else:
            # Equal weight — each stock counts as 1 (not weight-adjusted)
            df['weight'] = 1.0
            df_g = df.groupby('industry_sector')['weight'].sum().reset_index(name='Repartition')
            fig = px.pie(df_g, values='Repartition', names='industry_sector',
                         color_discrete_sequence=px.colors.sequential.Blues[::-1])
            _style_pie(fig, 'Sector Breakdown')
            return {'is_dual': False, 'fig': fig}

    except Exception as e:
        return {'error': f"Error retrieving sector data: {str(e)}"}

def entry_point(long_tickers: List[str], short_tickers: List[str],
                long_weights: List[float], short_weights: List[float],
                start_date: date, end_date: date,
                is_cross_corridor: bool = False) -> Optional[go.Figure]:
    """Generate entry point analysis using 18M implied vol data.

    Tickers should be in Bloomberg format (e.g. 'SIE GY Equity') — same as backtest input.
    Returns None with error info stored in the figure's layout.meta if generation fails.
    """
    if is_cross_corridor:
        return None
    try:
        from xbbg import blp

        if hasattr(long_tickers, 'tolist'):
            long_tickers = long_tickers.tolist()
        if hasattr(short_tickers, 'tolist'):
            short_tickers = short_tickers.tolist()
        if hasattr(long_weights, 'tolist'):
            long_weights = long_weights.tolist()
        if hasattr(short_weights, 'tolist'):
            short_weights = short_weights.tolist()

        all_tickers = list(set(long_tickers + short_tickers))
        if not all_tickers:
            return None

        data = blp.bdh(all_tickers, "18MTH_IMPVOL_90.0%MNY_DF",
                       start_date=start_date, end_date=end_date)
        if data is None or data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]

        # Use threshold dropna instead of how="any" — keep rows where at least half the tickers have data
        min_valid = max(1, len(all_tickers) // 2)
        data = data.dropna(axis=0, thresh=min_valid)
        data = data.loc[~data.index.duplicated(keep='first')]
        if len(data) == 0:
            return None

        available = data.columns.tolist()

        long_sum = pd.Series(0.0, index=data.index)
        for i, t in enumerate(long_tickers):
            if t in available:
                long_sum += data[t].fillna(0) * long_weights[i]

        short_sum = pd.Series(0.0, index=data.index)
        for i, t in enumerate(short_tickers):
            if t in available:
                short_sum += data[t].fillna(0) * short_weights[i]

        result = long_sum - short_sum
        if result.abs().sum() == 0:
            return None

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=result.index, y=result.values,
            mode='lines', name='Implied Vol Spread',
            line=dict(color=BARCLAYS_LINE_ENTRY, width=3)
        ))
        fig = standardize_chart_display(fig, 'Entry Point — 18M Implied Vol Spread', show_legend=False, is_entry_point=True)
        return fig

    except Exception as e:
        # Store error for debugging — caller can check
        import traceback
        print(f"[entry_point] Failed: {e}\n{traceback.format_exc()}")
        return None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MATPLOTLIB STATIC RENDERING (for email — no kaleido, no orca, no disk)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _color_to_rgba(color_str: str) -> tuple:
    """Convert any plotly color string to matplotlib RGBA tuple."""
    if color_str is None:
        return (0.16, 0.50, 0.73, 1.0)
    s = str(color_str).strip()
    if s.startswith('rgba('):
        p = s[5:-1].split(',')
        return (float(p[0]) / 255, float(p[1]) / 255, float(p[2]) / 255, float(p[3]))
    if s.startswith('rgb('):
        p = s[4:-1].split(',')
        return (float(p[0]) / 255, float(p[1]) / 255, float(p[2]) / 255, 1.0)
    if s.startswith('#') and len(s) == 7:
        return (int(s[1:3], 16) / 255, int(s[3:5], 16) / 255, int(s[5:7], 16) / 255, 1.0)
    return s  # let matplotlib handle named colors

def _render_line_to_png(fig: go.Figure, title: str,
                        width: int = 3200, height: int = 1800) -> bytes:
    """
    Render a plotly line/area chart to PNG bytes via matplotlib.
    Handles secondary_y axis (e.g. active stocks count vs P&L).
    Also copes with what the entry-point figure may carry: traces without an explicit
    line width (used to raise TypeError), marker-only traces, bar traces, numeric or
    categorical x axes, horizontal/vertical reference lines (layout shapes).
    100% in-memory. No subprocess. No network. No timeout risk.
    Returns b'' when nothing could be plotted (the email then omits the chart).
    """
    import re
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    def _as_list(v):
        """plotly array -> list; also decodes the {'dtype','bdata'} form of dict-built figures."""
        if v is None:
            return []
        if isinstance(v, dict) and 'bdata' in v:
            arr = np.frombuffer(base64.b64decode(v['bdata']), dtype=np.dtype(v.get('dtype', 'f8')))
            return list(arr.reshape(v['shape']) if v.get('shape') else arr)
        return list(v)

    def _x_axis(xs):
        """x values -> (array, is_date): dates -> DatetimeIndex, numbers -> floats, else categories."""
        first = next((v for v in xs if v is not None), None)
        if isinstance(first, (bool, np.bool_)):
            return [str(v) for v in xs], False
        if isinstance(first, (int, float, np.integer, np.floating)):
            return np.array([np.nan if v is None else float(v) for v in xs], dtype=float), False
        if isinstance(first, str) and not re.match(r'^\d{4}-\d{1,2}-\d{1,2}', first):
            return [str(v) for v in xs], False
        try:
            return pd.to_datetime(xs).to_numpy(), True  # numpy array: mpl chokes on a pandas Index
        except (ValueError, TypeError, OverflowError):
            return [str(v) for v in xs], False

    def _finite(ys):
        return [v for v in ys if v is not None and not (isinstance(v, float) and v != v)]

    dpi = 150
    mpl_fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax.set_facecolor('white')
    mpl_fig.patch.set_facecolor('white')

    # Detect if figure has secondary_y (plotly stores this in layout.yaxis2)
    has_secondary = (hasattr(fig.layout, 'yaxis2') and fig.layout.yaxis2 is not None
                     and getattr(fig.layout.yaxis2, 'overlaying', None) is not None)
    ax2 = None
    if has_secondary:
        ax2 = ax.twinx()

    is_entry_point = 'entry point' in title.lower() or 'implied vol' in title.lower()
    n_bars = sum(1 for t in fig.data if getattr(t, 'type', '') == 'bar')
    bar_i = 0
    plotted = 0
    any_dates = False
    cat_ticks = None

    for i, trace in enumerate(fig.data):
        ttype = getattr(trace, 'type', 'scatter')
        if ttype not in ('scatter', 'scattergl', 'bar'):
            continue
        if getattr(trace, 'visible', True) in (False, 'legendonly'):
            continue
        x = _as_list(trace.x)
        y = _as_list(trace.y)
        if len(x) == 0 or len(y) == 0:
            continue
        label = (trace.name or '') if getattr(trace, 'showlegend', True) is not False else ''

        # Determine which axis: check trace.yaxis property
        # Plotly stores 'y2' for secondary_y traces
        target_ax = ax
        if has_secondary and ax2 is not None:
            trace_yaxis = getattr(trace, 'yaxis', None)
            if trace_yaxis == 'y2':
                target_ax = ax2

        # ---- bar traces (entry-point style figures) ----
        if ttype == 'bar':
            marker = getattr(trace, 'marker', None)
            mcol = getattr(marker, 'color', None) if marker is not None else None
            color = _color_to_rgba(mcol if isinstance(mcol, str) else None)
            try:
                if getattr(trace, 'orientation', None) == 'h':
                    target_ax.barh([str(v) for v in y],
                                   np.array([np.nan if v is None else v for v in x], dtype=float),
                                   color=color, label=label, zorder=2)
                    cat_ticks = cat_ticks or []
                else:
                    y_arr = np.array([np.nan if v is None else v for v in y], dtype=float)
                    x_arr, is_date = _x_axis(x)
                    any_dates = any_dates or is_date
                    if isinstance(x_arr, list):  # categorical x: grouped bars around each category
                        pos = np.arange(len(x_arr))
                        w = 0.8 / max(n_bars, 1)
                        target_ax.bar(pos + (bar_i - (n_bars - 1) / 2) * w, y_arr, width=w,
                                      color=color, label=label, zorder=2)
                        cat_ticks = x_arr
                    else:
                        target_ax.bar(x_arr, y_arr, color=color, label=label, zorder=2, alpha=0.8)
            except (TypeError, ValueError) as e:
                print(f"[charts] Warning: bar trace '{label}' skipped: {e}")
                continue
            bar_i += 1
            plotted += 1
            continue

        # ---- scatter traces ----
        x_arr, is_date = _x_axis(x)
        any_dates = any_dates or is_date
        if isinstance(x_arr, list):
            cat_ticks = x_arr
        try:
            y_arr = np.array([np.nan if v is None else v for v in y], dtype=float)
        except (TypeError, ValueError) as e:
            print(f"[charts] Warning: trace '{label}' skipped (non-numeric y): {e}")
            continue

        # Line properties (an unset width used to give None * 0.7 -> TypeError)
        line_color = _color_to_rgba(
            getattr(trace.line, 'color', None) if trace.line else None)
        _lw = getattr(trace.line, 'width', None) if trace.line else None
        line_width = (_lw if _lw is not None else 2) * 0.7
        dash = getattr(trace.line, 'dash', None) if trace.line else None
        linestyle = '--' if dash in ('dot', 'dash', 'dashdot') else '-'
        mode = getattr(trace, 'mode', None) or 'lines'
        marker = getattr(trace, 'marker', None)
        if line_color is None and marker is not None and isinstance(getattr(marker, 'color', None), str):
            line_color = _color_to_rgba(marker.color)

        if 'lines' in mode:
            target_ax.plot(x_arr, y_arr, color=line_color, linewidth=max(line_width, 1.0),
                           label=label, zorder=3, linestyle=linestyle)
        else:
            # marker-only trace (e.g. current level): dots, a 1-point line would be invisible
            ms = getattr(marker, 'size', None) if marker is not None else None
            ms = float(ms) if isinstance(ms, (int, float)) and ms > 0 else 8.0
            target_ax.scatter(x_arr, y_arr, color=line_color, s=ms ** 2, label=label, zorder=4)
        if 'text' in mode and getattr(trace, 'text', None) is not None:
            texts = [trace.text] * len(y_arr) if isinstance(trace.text, str) else list(trace.text)
            # honour plotly textposition ('middle left', 'top center', ...) and textfont colour
            _tp = getattr(trace, 'textposition', None) or 'top center'
            _tp = str(_tp[0] if isinstance(_tp, (list, tuple)) else _tp)
            ha = 'left' if 'right' in _tp else ('right' if 'left' in _tp else 'center')
            va = 'bottom' if 'top' in _tp else ('top' if 'bottom' in _tp else 'center')
            dx = 10 if ha == 'left' else (-10 if ha == 'right' else 0)
            dy = 8 if va == 'bottom' else (-8 if va == 'top' else 0)
            _tf = getattr(trace, 'textfont', None)
            tcol = _color_to_rgba(getattr(_tf, 'color', None) if _tf is not None else None) or '#333'
            for xi, yi, txt in zip(x_arr, y_arr, texts):
                if yi == yi:
                    target_ax.annotate(str(txt), (xi, yi), fontsize=18, fontweight='bold',
                                       xytext=(dx, dy), textcoords='offset points',
                                       ha=ha, va=va, color=tcol)

        # Fill area
        if getattr(trace, 'fill', None) == 'tozeroy':
            fc = _color_to_rgba(getattr(trace, 'fillcolor', None))
            if isinstance(fc, tuple):
                target_ax.fill_between(x_arr, 0, y_arr, color=fc, zorder=2)
        plotted += 1

    if plotted == 0:
        plt.close(mpl_fig)
        return b''

    # Reference lines added with add_hline / add_vline (entry-point figures only)
    if is_entry_point:
        for sh in (getattr(fig.layout, 'shapes', None) or []):
            if getattr(sh, 'type', None) != 'line':
                continue
            sl = getattr(sh, 'line', None)
            col = _color_to_rgba(getattr(sl, 'color', None) if sl is not None else None) or '#888'
            ls = '--' if getattr(sl, 'dash', None) in ('dot', 'dash', 'dashdot') else '-'
            try:
                if sh.y0 is not None and sh.y0 == sh.y1:
                    ax.axhline(float(sh.y0), color=col, linewidth=1.2, linestyle=ls, zorder=1)
                elif sh.x0 is not None and sh.x0 == sh.x1:
                    ax.axvline(pd.to_datetime(sh.x0) if any_dates else sh.x0,
                               color=col, linewidth=1.2, linestyle=ls, zorder=1)
            except (TypeError, ValueError):
                pass
        # Annotations added with add_annotation (label box + leader line, e.g. the last-point
        # marker): first line big and bold, following lines (after <br>) smaller and grey
        from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker
        for an in (getattr(fig.layout, 'annotations', None) or []):
            try:
                if an.text is None or an.x is None or an.y is None:
                    continue
                xv = pd.to_datetime(an.x) if any_dates else an.x
                raw = re.sub(r'<br\s*/?>', '\n', str(an.text))
                parts = [re.sub(r'<[^>]+>', '', s).strip() for s in raw.split('\n')]
                parts = [s for s in parts if s] or ['']
                afont = getattr(an, 'font', None)
                col = _color_to_rgba(getattr(afont, 'color', None) if afont is not None else None) or BARCLAYS_NAVY
                bg = _color_to_rgba(getattr(an, 'bgcolor', None))
                bc = _color_to_rgba(getattr(an, 'bordercolor', None))
                lc = _color_to_rgba(getattr(an, 'arrowcolor', None)) or '#999999'
                texts = [TextArea(parts[0], textprops=dict(size=21, weight='bold', color=col))]
                texts += [TextArea(s, textprops=dict(size=14, color='#666666')) for s in parts[1:]]
                show_arrow = getattr(an, 'showarrow', True)
                axp = float(an.ax) if an.ax is not None else -60.0
                ayp = float(an.ay) if an.ay is not None else -40.0
                ab = AnnotationBbox(
                    VPacker(children=texts, align='center', pad=0, sep=5), (xv, float(an.y)),
                    xybox=(axp * 1.5, -ayp * 1.5) if show_arrow else (0, 0), boxcoords='offset points',
                    frameon=True, zorder=6,
                    bboxprops=dict(boxstyle='round,pad=0.7', fc=bg if bg is not None else 'white',
                                   ec=bc if bc is not None else 'none', lw=1.0),
                    arrowprops=dict(arrowstyle='-', color=lc, lw=1.0) if show_arrow else None)
                ax.add_artist(ab)
            except Exception:
                pass

    # Styling - primary axis
    ax.set_title(title, fontsize=32, fontweight='bold', pad=25, color=BARCLAYS_NAVY)
    ax.set_xlabel('', fontsize=22, color='#333')
    # Adapt y-axis label based on chart content
    if is_entry_point:
        # Entry point: detect if vol spread or strikes spread from data magnitude
        all_y_vals = []
        for trace in fig.data:
            if hasattr(trace, 'y') and trace.y is not None:
                all_y_vals.extend(_finite(_as_list(trace.y)))
        if all_y_vals and max(abs(v) for v in all_y_vals) < 50:
            ylabel = 'Vol spread'
        else:
            ylabel = 'Strikes spread'
    else:
        ylabel = 'Cash payout per unit of vega notional'
    ax.set_ylabel(ylabel, fontsize=22, color='#555')
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(True, alpha=0.7, linestyle='-', linewidth=0.6, color='#c0c0c0')
    ax.spines['top'].set_visible(False)
    if not has_secondary:
        ax.spines['right'].set_visible(False)

    # Entry point charts: tight y-axis to actual data range (don't force 0 into view)
    if is_entry_point:
        all_y_for_range = []
        for trace in fig.data:
            if hasattr(trace, 'y') and trace.y is not None:
                all_y_for_range.extend(_finite(_as_list(trace.y)))
        if all_y_for_range:
            y_lo, y_hi = min(all_y_for_range), max(all_y_for_range)
            spread = y_hi - y_lo if y_hi != y_lo else abs(y_hi) * 0.2 or 1.0
            pad = spread * 0.05
            ax.set_ylim(y_lo - pad, y_hi + pad)
    else:
        ax.axhline(y=0, color='#888', linewidth=0.8, linestyle='-', alpha=0.5, zorder=1)

    # Styling - secondary axis
    if ax2 is not None:
        y2_label = '# Active Stocks'
        if is_entry_point:
            _t = getattr(getattr(fig.layout.yaxis2, 'title', None), 'text', None)
            y2_label = str(_t) if _t else ''
        ax2.set_ylabel(y2_label, fontsize=20, color='#3498db')
        ax2.tick_params(axis='y', labelsize=16, colors='#3498db')
        ax2.spines['right'].set_color('#3498db')
        if not is_entry_point:
            ax2.set_ylim(bottom=0)
        ax2.grid(False)

    # Legend (combine both axes)
    handles, labels = ax.get_legend_handles_labels()
    if ax2 is not None:
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels += l2
    if len(labels) > 1:
        ax.legend(handles, labels, fontsize=20, loc='upper left', framealpha=0.9, edgecolor='#ddd')

    # x axis: dates, categories or plain numbers
    if any_dates:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=12))
        mpl_fig.autofmt_xdate(rotation=30)
    elif cat_ticks:
        ax.set_xticks(np.arange(len(cat_ticks)))
        ax.set_xticklabels(cat_ticks, rotation=30, ha='right')

    plt.tight_layout()
    buf = io.BytesIO()
    mpl_fig.savefig(buf, format='png', bbox_inches='tight', facecolor='white', dpi=dpi)
    plt.close(mpl_fig)
    buf.seek(0)
    return buf.read()
def _render_pie_to_png(fig: go.Figure, title: str,
                       width: int = 1600, height: int = 1600) -> bytes:
    """Render a plotly pie chart to PNG bytes via matplotlib — interface style.

    House Blues (long) / Greys (short) palette, white wedge borders, percent inside
    the slices (suppressed below 2.5%; navy text on light slices, white on dark),
    full labels in a right-hand legend, navy bold title. Above 20 slices the smallest
    are grouped in 'Other'."""

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # House palettes — exact same hues as the interface pies (_PIE_BLUES / _PIE_GREYS)
    _PIE_BLUES_HEX = ['#08306b', '#08519c', '#2171b5', '#4292c6', '#6baed6',
                      '#9ecae1', '#c6dbef', '#deebf7', '#f7fbff']
    _PIE_GREYS_HEX = ['#252525', '#525252', '#737373', '#969696', '#bdbdbd', '#d9d9d9', '#f0f0f0']

    dpi = 200
    mpl_fig = plt.figure(figsize=(11.5, 7.5), dpi=dpi)
    mpl_fig.patch.set_facecolor('white')
    gs = GridSpec(1, 2, width_ratios=[3, 2], figure=mpl_fig)
    ax = mpl_fig.add_subplot(gs[0])
    ax.set_aspect('equal', adjustable='box')  # keep the pie perfectly round

    trace = fig.data[0] if fig.data else None
    if trace is None:
        plt.close(mpl_fig)
        return b''

    def _as_list(v):
        if v is None:
            return []
        if isinstance(v, dict) and 'bdata' in v:
            arr = np.frombuffer(base64.b64decode(v['bdata']), dtype=np.dtype(v.get('dtype', 'f8')))
            return list(arr.reshape(v['shape']) if v.get('shape') else arr)
        return list(v)

    # Robust extraction: handle tuple, list, ndarray, or dict-based trace
    _raw_labels = getattr(trace, 'labels', None)
    _raw_values = getattr(trace, 'values', None)
    if _raw_labels is None:
        try:
            _raw_labels = trace['labels']
        except (KeyError, TypeError):
            pass
    if _raw_values is None:
        try:
            _raw_values = trace['values']
        except (KeyError, TypeError):
            pass
    labels = [str(l) for l in _as_list(_raw_labels)]
    values = []
    for v in _as_list(_raw_values):
        try:
            values.append(float(v))
        except (TypeError, ValueError):
            values.append(None)
    clean = [(l, v) for l, v in zip(labels, values) if v is not None and v == v and v > 0]
    if not clean:
        plt.close(mpl_fig)
        return b''

    # Largest slices first (matches the interface pie ordering)
    clean = sorted(clean, key=lambda lv: lv[1], reverse=True)
    max_slices = 20
    if len(clean) > max_slices:
        head, tail = clean[:max_slices - 1], clean[max_slices - 1:]
        clean = head + [(f'Other ({len(tail)})', sum(v for _, v in tail))]
    labels, values = [l for l, _ in clean], [v for _, v in clean]

    is_short_basket = 'short' in title.lower()
    palette = _PIE_GREYS_HEX if is_short_basket else _PIE_BLUES_HEX
    wedge_colors = [palette[i % len(palette)] for i in range(len(labels))]

    def _autopct(pct):
        return f'{pct:.1f}%' if pct >= 2.5 else ''

    wedges, texts, autotexts = ax.pie(
        values, autopct=_autopct, colors=wedge_colors,
        pctdistance=0.72, startangle=90, counterclock=False,
        wedgeprops={'linewidth': 2, 'edgecolor': 'white'},
    )
    for t in texts:
        t.set_visible(False)

    # % label color follows the slice luminance (white on dark, navy on light)
    def _lum(hex_color):
        h = hex_color.lstrip('#')
        r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    for at, wc in zip(autotexts, wedge_colors):
        at.set_fontsize(10)
        at.set_fontweight('bold')
        at.set_color('#00395D' if _lum(wc) > 0.55 else 'white')

    legend_title = "Stocks" if 'weight' in title.lower() else "Sectors"
    ax.legend(wedges, labels, title=legend_title, loc="center left",
              bbox_to_anchor=(1.02, 0.5), fontsize=11, title_fontsize=12, frameon=False)

    mpl_fig.suptitle(title, fontsize=20, fontweight='bold', color=BARCLAYS_NAVY, y=0.97)

    buf = io.BytesIO()
    mpl_fig.savefig(buf, format='png', facecolor='white', dpi=dpi,
                    bbox_inches='tight', pad_inches=0.35)
    plt.close(mpl_fig)
    buf.seek(0)
    return buf.read()




def render_charts_to_bytes(charts_data: Dict) -> List[Tuple[str, bytes]]:
    """
    Render all charts to PNG bytes using matplotlib.
    NO kaleido. NO orca. NO disk writes. NO subprocess. NO timeout.

    A chart that is None, or that fails to render, is reported on the console
    ("[charts] ...") and left out of the email (the HTML then drops its section).

    Returns: list of (filename, png_bytes) tuples.
    """
    import traceback as _tb

    rendered = []
    is_cross_corridor = charts_data.get('is_cross_corridor', False)
    has_short_leg = charts_data.get('has_short_leg', False)
    is_dual = charts_data.get('is_dual_sectorial', False)

    def _render(key, filename, title, renderer, *size):
        if key not in charts_data:
            return
        val = charts_data.get(key)
        if val is None:
            print(f"[charts] '{key}' is None — '{filename}' will be missing from the email")
            return
        fig = go.Figure(val) if not isinstance(val, go.Figure) else val
        try:
            png = renderer(fig, title, *size)
        except Exception as e:
            print(f"[charts] ERROR: render failed for '{filename}' — it will be missing from the email: "
                  f"{e}\n{_tb.format_exc()}")
            return
        if png and len(png) > 200:
            rendered.append((filename, png))
        else:
            print(f"[charts] ERROR: '{filename}' has nothing to plot — it will be missing from the email")

    def _line(key, filename, title, w=3200, h=1800):
        _render(key, filename, title, _render_line_to_png, w, h)

    def _pie(key, filename, title):
        _render(key, filename, title, _render_pie_to_png)

    _line('main_graph', 'line_graph.png', 'Strategy Performance')
    if has_short_leg:
        _line('split_graph', 'graph_split.png', 'Strategy Split Analysis')
    _line('graph_60d', 'graph_60d.png', '3M Carry Analysis')
    if has_short_leg:
        _line('split_60d', 'graph_60d_split.png', '3M Carry Split Analysis')
    # Sectorial chart: render for all structures except index-only
    # For cross-corridor, only long basket (stocks) is shown; short is indexes (no sector data)
    if is_dual:
        _pie('sectorial_long', 'sectorial_long.png', 'Long Basket Sectors')
        _pie('sectorial_short', 'sectorial_short.png', 'Short Basket Sectors')
    else:
        _pie('sectorial', 'sectorial.png', 'Sector Repartition')
    # Weights-by-stock pie (same figure as the UI's "Weights by stock")
    _pie('weights_pie', 'weights_pie.png', 'Weights by stock')
    # Entry point: mono = basket IV spread; cross corridor = corridor assets − index
    # IV spread (same 18M field) — the UI supplies the figure in both cases
    _line('entry_point', 'entry_point.png', 'Entry Point Analysis')

    return rendered



def _generate_email_html(charts_data: Dict, result_series: Optional[pd.Series],
                         data_editor: Optional[pd.DataFrame],
                         n_exp: int, is_cross_corridor: bool, product_type: str,
                         local_cap: float, barrier_up: float, barrier_down: float,
                         short_leg_display: str = "Index",
                         adj_divs: str = "No",
                         base64_images: Optional[Dict[str, str]] = None,
                         carry_series: Optional[pd.Series] = None) -> str:
    """
    Generate email HTML matching original Gaia_PP format exactly.
    Charts embedded as base64 data URIs — no external files needed.

    A section (and its bullet in the intro list) is emitted only when its image was
    rendered: no "Entry Point" title without a chart, no "Sector Split" without a pie.

    IMPORTANT — parameter conventions:
      barrier_up/barrier_down: Already in DECIMAL form (e.g. 1.3, 0.7) — we convert to % for display.
      local_cap: Multiplier (e.g. 2.5)
    """
    has_short_leg = charts_data.get('has_short_leg', False)
    is_dual_sectorial = charts_data.get('is_dual_sectorial', False)
    base64_images = base64_images or {}

    # Detect structure type
    if data_editor is None or len(data_editor) == 0:
        structure_type = "dispersion"
    elif is_cross_corridor:
        structure_type = "cross_corridor"
    else:
        weight_col = 'Weight (%)' if 'Weight (%)' in data_editor.columns else 'Weights'
        weights = pd.to_numeric(data_editor[weight_col], errors='coerce').tolist()
        has_negative = any(w < 0 for w in weights if not pd.isna(w))
        positive_count = sum(1 for w in weights if w > 0)
        negative_count = sum(1 for w in weights if w < 0)
        if has_negative and positive_count > 0:
            structure_type = "dispersion" if negative_count == 1 else "basket_vs_basket"
        else:
            structure_type = "long_only"

    # Dynamic content per structure type
    if structure_type in ["cross_corridor", "cross_corridor_long_only"]:
        strategy_name = "cross corridor package"
        split_description = ""
        entry_point_description = ""
        sectorial_description = "Sectorial split of the baskets components: please note this figure is not weight adjusted"
        product_label = "Cross Corridor"
    elif structure_type == "dispersion":
        strategy_name = "dispersion basket"
        split_description = "Split between the short leg and the long leg"
        entry_point_description = "This shows the historical evolution of the difference between the weighted average strikes of the stocks and the strike of the index"
        sectorial_description = "Sectorial split of the stocks: please note this figure is not weight adjusted"
        product_label = "Dispersion"
    elif structure_type == "basket_vs_basket":
        strategy_name = "basket strategy"
        split_description = "Split between the short basket and the long basket"
        entry_point_description = "This shows the historical evolution of the difference between the long basket and short basket implied volatilities"
        sectorial_description = "Sectorial split of the basket components: please note this figure is not weight adjusted"
        product_label = "Strategy"
    else:
        strategy_name = "long basket strategy"
        split_description = "Performance breakdown of the long basket components"
        entry_point_description = "This shows the historical evolution of the weighted average implied volatility of the basket components"
        sectorial_description = "Sectorial split of the basket components: please note this figure is not weight adjusted"
        product_label = "Strategy"

    # Bullet points
    backtest_split_bullet = f'''
                <li style="color: #2e75b6; font-size:14.5px; line-height:22px;">
                    <span style="color:#000000; font-size:14.5px; line-height:22px;">
                        {split_description}
                    </span>
                </li>''' if split_description else ""

    carry_split_bullet = f'''
                <li style="color: #2e75b6; font-size:14.5px; line-height:22px;">
                    <span style="color:#000000; font-size:14.5px; line-height:22px;">
                        {split_description} for the 3M period
                    </span>
                </li>''' if split_description else ""

    # Tables
    returns_table = _create_returns_table(result_series, carry_series=carry_series)
    underlyings_table = _create_underlyings_table(data_editor, n_exp)

    # Parameters
    matu = _convert_matu(n_exp)
    return_str = f"{result_series.mean():.1f}v" if result_series is not None and len(result_series) > 0 else "N/A"
    offer = _return_offer(data_editor)
    product_name = "Volswaps" if product_type == 'Vol Swap' else "Variance Swaps"

    # Clean short_leg_display for title contexts (strip "vs " prefix)
    short_name_clean = short_leg_display.replace('vs ', '').strip() if short_leg_display else ''

    # Corridor info — barrier_up/barrier_down come in as decimals (1.3, 0.7)
    # Display as percentages: 130.00/70.00
    corridor_info = ""
    if product_type in ['Var Swap', 'Cross Corridor']:
        barrier_up_display = round(barrier_up * 100, 2)
        barrier_down_display = round(barrier_down * 100, 2)
        div_text = "not dividend adjusted" if adj_divs in ["No", False] else "dividend adjusted"
        corridor_info = f"{barrier_down_display:.2f}/{barrier_up_display:.2f} barriers, T/T-1, {div_text}<br>"

    # Compute source date range from result_series
    if result_series is not None and len(result_series) > 0:
        _src_start = pd.Timestamp(result_series.index[0])
        _src_end = pd.Timestamp(result_series.index[-1])
        source_period = f"{_src_start.strftime('%b%y')} – {_src_end.strftime('%b%y')}"
    else:
        source_period = ""
    source_line = f"Source: Bloomberg and Barclays, {source_period}" if source_period else "Source: Bloomberg and Barclays"

    # Image embedding helper
    def _img(key: str, w: int = 1000, h: int = 500) -> str:
        if key in base64_images:
            return f'<img width="{w}" height="{h}" src="data:image/png;base64,{base64_images[key]}">'
        return ''

    def _img_pie(key: str) -> str:
        """Pie charts with legend — wider aspect ratio to fit legend beside pie."""
        if key not in base64_images:
            return ''
        b64 = base64_images[key]
        return f'<img width="700" height="480" src="data:image/png;base64,{b64}" style="display:block; max-width:100%;">'

    # Section title style: Aptos 14px, rgb(0,174,239), bold underline
    _title_style = 'font-family:Aptos,Calibri,sans-serif; color:rgb(0,174,239); font-size:14px;'

    # Sector section: GICS pies (long/short or single) + weights-by-stock pie,
    # whichever were rendered, two per row. Nothing rendered -> no section, no bullet.
    if is_dual_sectorial:
        _pie_keys = ["sectorial_long.png", "sectorial_short.png"]
    else:
        _pie_keys = ["sectorial.png"]
    _pie_keys.append("weights_pie.png")
    _pie_cells = [_img_pie(k) for k in _pie_keys if k in base64_images]
    if _pie_cells:
        _rows = [_pie_cells[i:i + 2] for i in range(0, len(_pie_cells), 2)]
        sectorial_html = '<table cellpadding="0" cellspacing="0">' + ''.join(
            '<tr>' + ''.join(f'<td width="700" align="center">{c}</td>' for c in row) + '</tr>'
            for row in _rows) + '</table>'
    else:
        sectorial_html = ''

    sectorial_bullet = f'''
        <li style="color: #00AEEF; font-size:14.5px; line-height:22px;">
            <span style="color:#000000; font-size:14.5px; line-height:22px;">
                {sectorial_description}
            </span>
        </li>''' if sectorial_html else ""

    sectorial_section = f'''<u><b><p style="{_title_style}">
        Sector Split
    </p></b></u>

    {sectorial_html}

    <i>
    <p style="font-family: Aptos, Calibri, sans-serif; color:#000000; font-size:12px">
        {source_line}
    </p></i>''' if sectorial_html else ''

    if has_short_leg:
        backtest_images = f'<table width="890000"><tr><td>{_img("line_graph.png")}</td><td>{_img("graph_split.png")}</td></tr></table>'
        carry_images = f'<table width="890000"><tr><td>{_img("graph_60d.png")}</td><td>{_img("graph_60d_split.png")}</td></tr></table>'
    else:
        backtest_images = f'<table width="890000"><tr><td>{_img("line_graph.png")}</td></tr></table>'
        carry_images = f'<table width="890000"><tr><td>{_img("graph_60d.png")}</td></tr></table>'

    # Entry point — title and bullet only when the image exists (mono and cross corridor)
    entry_point_bullet = ""
    entry_point_image = ""
    if 'entry_point.png' in base64_images:
        entry_point_bullet = f'''
        <li style="color: #2e75b6; font-size:14.5px; line-height:22px;">
            <span style="color:#000000; font-size:14.5px; line-height:22px;">
                Historical entry point
            </span>
            <ul type="square">
                <li style="color: #2e75b6; font-size:14.5px; line-height:22px;">
                    <span style="color:#000000; font-size:14.5px; line-height:22px;">
                        {entry_point_description}
                    </span>
                </li>
                <li style="color: #2e75b6; font-size:14.5px; line-height:22px;">
                    <span style="color:#000000; font-size:14.5px; line-height:22px;">
                        Please note these strikes are derived from Bloomberg data and may not be aligned with internal tradable prices
                    </span>
                </li>
            </ul>
        </li>'''

        entry_point_image = f'''
    <u><b><p style="font-family:Aptos,Calibri,sans-serif; color:rgb(0,174,239); font-size:14px;">
        Entry Point
    </p></b></u>
    <table width="890000"><tr><td>{_img("entry_point.png")}</td></tr></table>
    <i><p style="font-family: Aptos, Calibri, sans-serif; color:#000000; font-size:12px">
        {source_line}
    </p></i>'''

    # Full HTML assembly
    html = f"""
    <p style="font-family: Aptos, Calibri, sans-serif; color:#000000; font-size:14.5px">
        Hi, <br>
        Find below our latest {strategy_name}. Over the analysis period, average return has been {return_str}.
       <br><br>
        This email shows the below elements:

    <ul style="margin:0; margin-left: 25px; padding:0; font-family: Aptos, Calibri, sans-serif;" align="left" type="disc"; font-size:14.5px>
        <li style="color: #00AEEF; font-size:14.5px; line-height:14.5px;">
            <span style="color:#000000; font-size:14.5px; line-height:14.5px;">
                Backtest of the strategy
            </span>
            <ul type="square">
                <li style="color: #00AEEF; font-size:14.5px; line-height:22px;">
                    <span style="color:#000000; font-size:14.5px; line-height:22px;">
                        Historical payout over the analysis period for the indicated maturity
                    </span>
                </li>{backtest_split_bullet}
            </ul>
        </li>

        <li style="color: #00AEEF; font-size:14.5px; line-height:22px;">
            <span style="color:#000000; font-size:14.5px; line-height:22px;">
                3M Carry payout
            </span>
            <ul type="square">
                <li style="color: #00AEEF; font-size:14.5px; line-height:22px;">
                    <span style="color:#000000; font-size:14.5px; line-height:22px;">
                        Historical payout for a trade with a maturity of 3 months
                    </span>
                </li>{carry_split_bullet}
            </ul>
        </li>
        {sectorial_bullet}

        {entry_point_bullet}
    </ul>

    </p>
    <u><b>
    <p style="{_title_style}">
        Trade Description
    </u></b></p>
    <p style="font-family: Aptos, Calibri, sans-serif; color:#000000; font-size:14.5px">
        {product_label} {short_name_clean} {product_name} <br>
        All {product_name} capped {local_cap}<br>
        {matu} maturity<br>
        Offer @ {offer}% <br>
        {corridor_info}
    </p>
    <p style="font-family: Aptos, Calibri, sans-serif; color:red; font-size:10px; font-style:italic;">
        Levels are indicative subject to refresh
    </p>
    {underlyings_table}

    <u><b>
    <p style="{_title_style}">
        Historical Performance
    </u></b></p>

    {returns_table}

    {backtest_images}
    <i>
    <p style="font-family: Aptos, Calibri, sans-serif; color:#000000; font-size:12px">
        {source_line}
    </p></i>

    <u><b>
    <p style="{_title_style}">
        3M Carry
    </u></b></p>

    {carry_images}

    <i>
    <p style="font-family: Aptos, Calibri, sans-serif; color:#000000; font-size:12px">
        {source_line}
    </p></i>

    {sectorial_section}

    {entry_point_image}
    """
    return html


def _convert_matu(n_exp: int) -> str:
    """n_exp (business days) → maturity month label used in the email subject/body
    (e.g. 'Dec27'). ~21 business days per month."""
    return (date.today() + relativedelta(months=int(round((n_exp or 0) / 21.0)))).strftime("%b%y")


def _return_offer(data_editor) -> str:
    """'Offer @ X%' of the email trade description: weight-averaged basket strike
    in vol points (cross corridor uses the cross strike when both are present)."""
    if data_editor is None or len(data_editor) == 0:
        return "N/A"
    try:
        df = data_editor
        strike_col = ('Strike Cross Corridor (%)' if 'Strike Cross Corridor (%)' in df.columns
                      else 'Strike Mono Var Swap (%)' if 'Strike Mono Var Swap (%)' in df.columns
                      else 'Strikes (%)' if 'Strikes (%)' in df.columns else None)
        weight_col = ('Weight (%)' if 'Weight (%)' in df.columns
                      else 'Weights (%)' if 'Weights (%)' in df.columns else None)
        if strike_col is None:
            return "N/A"
        strikes = pd.to_numeric(df[strike_col], errors='coerce')
        if weight_col is not None:
            w = pd.to_numeric(df[weight_col], errors='coerce').abs()
            mask = strikes.notna() & w.notna() & (w > 0)
            if mask.sum() == 0:
                return "N/A"
            return f"{(strikes[mask] * w[mask]).sum() / w[mask].sum():.2f}"
        return f"{strikes.dropna().mean():.2f}"
    except Exception:
        return "N/A"


def _create_returns_table(result_series: Optional[pd.Series], carry_series: Optional[pd.Series] = None) -> str:
    """
    Create vertical returns table for email with two columns: Backtest and 3M Carry.
    Mean, Hit Ratio, Max, 75%ile, 50%ile, 25%ile, Min, Last.
    """
    if result_series is None or len(result_series) == 0:
        return ""
    s = result_series.replace(np.nan, 0)
    min_val = s.min()
    p25 = float(s.quantile(0.25))
    p50 = float(s.quantile(0.50))
    p75 = float(s.quantile(0.75))
    max_val = s.max()
    mean_val = s.mean()
    hr = (s > 0).sum() / max(1, (s != 0).sum()) * 100
    last_value = s.iloc[-1]

    has_carry = carry_series is not None and len(carry_series) > 0
    if has_carry:
        sc = carry_series.replace(np.nan, 0)
        c_min, c_max, c_mean = sc.min(), sc.max(), sc.mean()
        c_p25, c_p50, c_p75 = float(sc.quantile(0.25)), float(sc.quantile(0.50)), float(sc.quantile(0.75))
        c_hr = (sc > 0).sum() / max(1, (sc != 0).sum()) * 100
        c_last = sc.iloc[-1]

    n_cols = 3 if has_carry else 2

    def _header():
        carry_th = (
            '<td style="padding:5px 10px; text-align:center; font-family:Aptos,Calibri,sans-serif; font-size:12px; '
            'background-color:#00AEEF; color:white; font-weight:bold; border:0;">3M Carry</td>'
        ) if has_carry else ''
        return (
            f'<tr>'
            f'<td style="padding:5px 10px; text-align:left; font-family:Aptos,Calibri,sans-serif; font-size:12px; '
            f'background-color:white; color:white; font-weight:bold; border:0;"></td>'
            f'<td style="padding:5px 10px; text-align:center; font-family:Aptos,Calibri,sans-serif; font-size:12px; '
            f'background-color:#00AEEF; color:white; font-weight:bold; border:0;">Backtest</td>'
            f'{carry_th}</tr>'
        )

    def _row(label, value, carry_value=None):
        carry_td = (
            f'<td style="padding:4px 10px; text-align:center; font-family:Aptos,Calibri,sans-serif; font-size:12px; '
            f'background-color:#f8f9fa; border:0;">{carry_value}</td>'
        ) if has_carry else ''
        return (
            f'<tr>'
            f'<td style="padding:4px 10px; text-align:left; font-family:Aptos,Calibri,sans-serif; font-size:12px; '
            f'background-color:#00AEEF; color:white; font-weight:bold; border:0;">{label}</td>'
            f'<td style="padding:4px 10px; text-align:center; font-family:Aptos,Calibri,sans-serif; font-size:12px; '
            f'background-color:#f8f9fa; border:0;">{value}</td>'
            f'{carry_td}</tr>'
        )

    def _spacer():
        return f'<tr><td colspan="{n_cols}" style="padding:0; border:0; font-size:0; line-height:3px; height:3px; background-color:white;">&nbsp;</td></tr>'

    rows = [_header()]
    if has_carry:
        rows += [
            _row('Mean', f'{mean_val:.2f}%', f'{c_mean:.2f}%'),
            _row('Hit Ratio (% Positive)', f'{hr:.1f}%', f'{c_hr:.1f}%'),
            _spacer(),
            _row('Max', f'{max_val:.2f}%', f'{c_max:.2f}%'),
            _row('75%ile', f'{p75:.2f}%', f'{c_p75:.2f}%'),
            _row('50%ile', f'{p50:.2f}%', f'{c_p50:.2f}%'),
            _row('25%ile', f'{p25:.2f}%', f'{c_p25:.2f}%'),
            _row('Min', f'{min_val:.2f}%', f'{c_min:.2f}%'),
            _spacer(),
            _row('Last', f'{last_value:.2f}%', f'{c_last:.2f}%'),
        ]
    else:
        rows += [
            _row('Mean', f'{mean_val:.2f}%'),
            _row('Hit Ratio (% Positive)', f'{hr:.1f}%'),
            _spacer(),
            _row('Max', f'{max_val:.2f}%'),
            _row('75%ile', f'{p75:.2f}%'),
            _row('50%ile', f'{p50:.2f}%'),
            _row('25%ile', f'{p25:.2f}%'),
            _row('Min', f'{min_val:.2f}%'),
            _spacer(),
            _row('Last', f'{last_value:.2f}%'),
        ]

    return (
        '<table cellpadding="0" cellspacing="0" border="0" '
        'style="border-collapse:collapse; border-spacing:0; width:auto; margin:5px 0;">'
        f'{"".join(rows)}'
        '</table>'
    )

def _create_underlyings_table(data_editor: Optional[pd.DataFrame], n_exp: int) -> str:
    """
    Create underlyings table matching original format exactly:
    Proper columns, alternating row colors, centered numbers, left-aligned tickers.
    """
    if data_editor is None or len(data_editor) == 0:
        return ""
    try:
        df = data_editor.copy()
        is_cross_corridor = 'Corridor Condition Asset' in df.columns

        # Sort by decreasing absolute weight so heaviest positions appear first
        wt_col = 'Weight (%)' if 'Weight (%)' in df.columns else 'Weights'
        df["_abs_weight"] = pd.to_numeric(df[wt_col], errors='coerce').abs()
        df = df.sort_values("_abs_weight", ascending=False).drop(columns=["_abs_weight"]).reset_index(drop=True)

        if is_cross_corridor:
            header = """
            <table style="border-collapse: collapse; width: 90%; margin: 5px 0; font-size: 13px; border: 1px solid #ddd;">
                <tr style="background-color: #00AEEF; color: white;">
                    <th style="padding: 6px; text-align: center; font-size: 13px; border: 1px solid #ddd;">Corridor Condition Asset</th>
                    <th style="padding: 6px; text-align: center; font-size: 13px; border: 1px solid #ddd;">Variance Asset</th>
                    <th style="padding: 6px; text-align: center; font-size: 13px; border: 1px solid #ddd;">Strike Cross Corridor (%)</th>
                    <th style="padding: 6px; text-align: center; font-size: 13px; border: 1px solid #ddd;">Strike Mono Var Swap (%)</th>
                    <th style="padding: 6px; text-align: center; font-size: 13px; border: 1px solid #ddd;">Weight (%)</th>
                </tr>"""
            rows = ""
            for idx, row in df.iterrows():
                bg = "#f8f9fa" if idx % 2 == 0 else "#ffffff"
                mono_strike = float(row.get('Strike Mono Var Swap (%)', 0)) if pd.notna(row.get('Strike Mono Var Swap (%)')) else 0
                cross_strike = float(row.get('Strike Cross Corridor (%)', 0)) if pd.notna(row.get('Strike Cross Corridor (%)')) else 0
                weight = float(row.get('Weight (%)', 0)) if pd.notna(row.get('Weight (%)')) else 0
                rows += f"""
                <tr style="background-color: {bg};">
                    <td style="padding: 6px; text-align: center; font-size: 13px; border: 1px solid #ddd;">{row.get('Corridor Condition Asset', '')}</td>
                    <td style="padding: 6px; text-align: center; font-size: 13px; border: 1px solid #ddd;">{row.get('Variance Asset', '')}</td>
                    <td style="padding: 6px; text-align: center; font-size: 13px; border: 1px solid #ddd;">{cross_strike:.2f}%</td>
                    <td style="padding: 6px; text-align: center; font-size: 13px; border: 1px solid #ddd;">{mono_strike:.2f}%</td>
                    <td style="padding: 6px; text-align: center; font-size: 13px; border: 1px solid #ddd;">{weight:.2f}%</td>
                </tr>"""
            return header + rows + "</table>"
        else:
            header = """
            <table style="border-collapse: collapse; width: 70%; margin: 5px 0; font-size: 13px; border: 1px solid #ddd;">
                <tr style="background-color: #00AEEF; color: white;">
                    <th style="padding: 6px; text-align: center; font-size: 13px; border: 1px solid #ddd;">Variance Asset</th>
                    <th style="padding: 6px; text-align: center; font-size: 13px; border: 1px solid #ddd;">Strike Mono Var Swap (%)</th>
                    <th style="padding: 6px; text-align: center; font-size: 13px; border: 1px solid #ddd;">Weight (%)</th>
                    <th style="padding: 6px; text-align: center; font-size: 13px; border: 1px solid #ddd;">N Exp</th>
                </tr>"""
            rows = ""
            for idx, row in df.iterrows():
                bg = "#f8f9fa" if idx % 2 == 0 else "#ffffff"
                strike = float(row.get('Strike Mono Var Swap (%)', 0)) if pd.notna(row.get('Strike Mono Var Swap (%)')) else 0
                weight = float(row.get('Weight (%)', 0)) if pd.notna(row.get('Weight (%)')) else 0
                rows += f"""
                <tr style="background-color: {bg};">
                    <td style="padding: 6px; text-align: center; font-size: 13px; border: 1px solid #ddd;">{row.get('Variance Asset', '')}</td>
                    <td style="padding: 6px; text-align: center; font-size: 13px; border: 1px solid #ddd;">{strike:.2f}%</td>
                    <td style="padding: 6px; text-align: center; font-size: 13px; border: 1px solid #ddd;">{weight:.2f}%</td>
                    <td style="padding: 6px; text-align: center; font-size: 13px; border: 1px solid #ddd;">{n_exp}</td>
                </tr>"""
            return header + rows + "</table>"
    except Exception:
        return ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EMAIL HTML GENERATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EMAIL SEND (pure Python — works from Streamlit, Jupyter, scripts)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _render_any_to_png(fig, title: str = "", w: int = 3200, h: int = 1800) -> bytes:
    """Generic plotly Figure → PNG bytes via matplotlib (no kaleido, no disk).
    Scatter (lines/markers/text), Bar (grouped/stacked/horizontal), Histogram,
    Pie and layout shapes (hline/vline). Enough for entry-point and sector charts."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    dpi = 200
    fig_mpl, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    traces = list(fig.data)
    pies = [t for t in traces if t.type == "pie"]
    if pies:
        p = pies[0]
        labels = list(p.labels or [])
        values = [float(v) for v in (p.values or [])]
        hole = float(getattr(p, "hole", 0) or 0)
        wedges, *_ = ax.pie(values, labels=None, startangle=90, counterclock=False,
                            wedgeprops=dict(width=1 - hole) if hole else None,
                            autopct=lambda pct: f"{pct:.0f}%" if pct >= 3 else "")
        ax.legend(wedges, [f"{l} ({v:.1f})" for l, v in zip(labels, values)],
                  loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False)
        ax.set_aspect("equal")
    else:
        has_legend = False
        bar_offsets = {}
        stacked = (fig.layout.barmode == "stack")
        n_bars = sum(1 for t in traces if t.type == "bar")
        bar_i = 0
        for t in traces:
            name = t.name or ""
            if t.type in ("scatter", "scattergl"):
                x = list(t.x) if t.x is not None else list(range(len(t.y or [])))
                y = [float(v) if v is not None else np.nan for v in (t.y or [])]
                mode = t.mode or "lines"
                color = getattr(getattr(t, "line", None), "color", None) or getattr(getattr(t, "marker", None), "color", None)
                if "lines" in mode:
                    ax.plot(x, y, label=name or None, color=color if isinstance(color, str) else None,
                            linewidth=1.4)
                if "markers" in mode:
                    ax.scatter(x, y, label=None if "lines" in mode else (name or None), s=14,
                               color=color if isinstance(color, str) else None, zorder=3)
                if "text" in mode and t.text is not None:
                    for xi, yi, txt in zip(x, y, t.text):
                        ax.annotate(str(txt), (xi, yi), fontsize=7, xytext=(0, 4), textcoords="offset points")
                has_legend = has_legend or bool(name)
            elif t.type == "bar":
                x = list(t.x) if t.x is not None else list(range(len(t.y or [])))
                y = [float(v) if v is not None else 0.0 for v in (t.y or [])]
                horizontal = (t.orientation == "h")
                if horizontal:
                    ax.barh(x if t.y is None else list(t.y), [float(v) for v in t.x], label=name or None, alpha=0.85)
                else:
                    idx = np.arange(len(x))
                    if stacked or n_bars == 1:
                        bottom = [bar_offsets.get(i, 0.0) for i in idx] if stacked else None
                        ax.bar(idx, y, bottom=bottom, label=name or None, alpha=0.85)
                        if stacked:
                            for i, v in zip(idx, y):
                                bar_offsets[i] = bar_offsets.get(i, 0.0) + v
                    else:
                        width = 0.8 / n_bars
                        ax.bar(idx + (bar_i - (n_bars - 1) / 2) * width, y, width=width,
                               label=name or None, alpha=0.85)
                    ax.set_xticks(idx)
                    ax.set_xticklabels([str(v) for v in x], rotation=45, ha="right", fontsize=7)
                bar_i += 1
                has_legend = has_legend or bool(name)
            elif t.type == "histogram":
                data = [float(v) for v in (t.x if t.x is not None else t.y) if v is not None]
                ax.hist(data, bins=40, alpha=0.75, label=name or None,
                        orientation="horizontal" if t.x is None else "vertical")
                has_legend = has_legend or bool(name)
        for sh in (fig.layout.shapes or []):
            if getattr(sh, "type", None) != "line":
                continue
            ls = "--" if getattr(getattr(sh, "line", None), "dash", None) in ("dash", "dot", "dashdot") else "-"
            col = getattr(getattr(sh, "line", None), "color", None) or "grey"
            if sh.x0 == sh.x1:
                ax.axvline(sh.x0, linestyle=ls, color=col, linewidth=1)
            elif sh.y0 == sh.y1:
                ax.axhline(sh.y0, linestyle=ls, color=col, linewidth=1)
        for a in (fig.layout.annotations or []):
            if a.text and a.x is not None and a.y is not None:
                try:
                    ax.annotate(str(a.text), (a.x, a.y), fontsize=7)
                except Exception:
                    pass
        xt = getattr(getattr(fig.layout, "xaxis", None), "title", None)
        yt = getattr(getattr(fig.layout, "yaxis", None), "title", None)
        if xt is not None and getattr(xt, "text", None):
            ax.set_xlabel(xt.text, fontsize=9)
        if yt is not None and getattr(yt, "text", None):
            ax.set_ylabel(yt.text, fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=8)
        if has_legend:
            ax.legend(fontsize=8, frameon=False)
    ttl = title or (fig.layout.title.text if fig.layout.title and fig.layout.title.text else "")
    if ttl:
        ax.set_title(ttl, fontsize=11)
    fig_mpl.tight_layout()
    buf = io.BytesIO()
    fig_mpl.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig_mpl)
    return buf.getvalue()

def send_email_with_attachments(charts_data: Dict, recipient_email: str = "",
                                result_series: Optional[pd.Series] = None,
                                data_editor: Optional[pd.DataFrame] = None,
                                n_exp: int = 310,
                                is_cross_corridor: bool = False,
                                product_type: str = "Vol Swap",
                                local_cap: float = 2.5,
                                barrier_up: float = 1.3, barrier_down: float = 0.7,
                                short_leg_display: str = "Index",
                                adj_divs: str = "No",
                                progress_callback=None,
                                carry_series: Optional[pd.Series] = None,
                                **kwargs) -> Dict:
    """
    Generate and open/send an Outlook email with embedded chart images.

    This function:
    - Renders charts to PNG bytes in-memory using matplotlib (no kaleido/orca)
    - Embeds images as base64 data URIs directly in the HTML body
    - Never writes any file to disk
    - Never spawns a subprocess that can timeout
    - Works identically from Streamlit, Jupyter, or plain Python

    Args:
        charts_data: Dict with keys like 'main_graph', 'graph_60d', etc. (plotly Figures)
        recipient_email: If provided and non-empty, sends the email. Otherwise opens draft.
        result_series: The backtest Result column (pd.Series)
        data_editor: The basket DataFrame (Tickers/Strikes/Weights or cross-corridor format)
        n_exp: Number of business days to expiry
        is_cross_corridor: Whether this is a cross-corridor structure
        product_type: 'Vol Swap' or 'Var Swap' or 'Cross Corridor'
        local_cap: Local cap multiplier (e.g. 2.5)
        barrier_up: Upper corridor in decimal (e.g. 1.30)
        barrier_down: Lower corridor in decimal (e.g. 0.70)
        short_leg_display: Display name for short leg (e.g. 'SX5E Index')
        adj_divs: 'Yes' or 'No' (or bool)
        progress_callback: Optional fn(percent: int, message: str) for UI progress

    Returns:
        {'success': bool, 'message': str}
    """
    com_initialized = False

    try:
        import win32com.client as win32
        import pythoncom

        if progress_callback:
            progress_callback(10, "Rendering charts...")

        # Render charts to PNG bytes in memory — NO kaleido, NO disk
        chart_items = render_charts_to_bytes(charts_data)

        if not chart_items:
            return {'success': False, 'message': 'No charts could be rendered. Check charts_data contains valid plotly figures.'}

        if progress_callback:
            progress_callback(50, "Building email HTML...")

        # Base64-encode for inline embedding
        base64_images = {fname: base64.b64encode(png).decode('ascii') for fname, png in chart_items}

        # Normalize adj_divs to string
        if isinstance(adj_divs, bool):
            adj_divs = "Yes" if adj_divs else "No"

        # Generate HTML
        html_body = _generate_email_html(
            charts_data, result_series, data_editor,
            n_exp, is_cross_corridor, product_type,
            local_cap, barrier_up, barrier_down, short_leg_display, adj_divs,
            base64_images=base64_images,
            carry_series=carry_series,
        )

        if progress_callback:
            progress_callback(75, "Creating Outlook email...")

        pythoncom.CoInitialize()
        com_initialized = True

        outlook = win32.Dispatch('outlook.application')
        mail = outlook.CreateItem(0)

        if recipient_email and recipient_email.strip():
            mail.To = recipient_email

        # Subject line — use actual short leg name (e.g. "SPX" not generic "Index")
        matu = _convert_matu(n_exp)
        # Strip "vs " prefix — subject should read "SPX Volswap Dispersion Dec27"
        short_name = short_leg_display.replace('vs ', '').strip() if short_leg_display else ''
        if product_type == 'Vol Swap':
            mail.Subject = f'{short_name} Volswap Dispersion {matu}'
        elif is_cross_corridor:
            mail.Subject = f'{short_name} Cross Corridor Variance Swap Dispersion {matu}'
        else:
            mail.Subject = f'{short_name} Corridor Variance Swap Dispersion {matu}'

        # Set HTML body — images are base64-embedded, no attachments needed
        mail.HTMLBody = html_body

        if progress_callback:
            progress_callback(90, "Opening email...")

        if recipient_email and recipient_email.strip():
            mail.Send()
            message = f"Email sent to {recipient_email}"
        else:
            mail.Display()
            message = "Email draft opened in Outlook"

        pythoncom.CoUninitialize()
        com_initialized = False

        if progress_callback:
            progress_callback(100, "Done!")

        return {'success': True, 'message': message}

    except ImportError as e:
        return {'success': False, 'message': f'Missing dependency: {e}. Install pywin32.'}
    except Exception as e:
        if com_initialized:
            try:
                import pythoncom
                pythoncom.CoUninitialize()
            except Exception:
                pass
        return {'success': False, 'message': f'Email generation failed: {str(e)}'}

# ══════════════════════════════════════════════════════════════════════════════
# BacktestResult Extensions — monkey-patches .plot(), .email(), .rerun()
# Previously in backtest_extensions.py — now consolidated here.
# ══════════════════════════════════════════════════════════════════════════════

from functions.dispersion.models import BacktestResult as _BacktestResult

def _bt_plot(self):
    """Main backtest chart (aggregate P&L over time)."""
    return plot_main_backtest(self.timeseries)

def _bt_plot_split(self):
    """Long leg vs short leg split chart."""
    is_cross = getattr(self, "_config", {}).get("is_cross_corridor", False)
    return split_graph(self.timeseries, is_cross_corridor=is_cross)

def _bt_plot_entry_point(self):
    """Historical implied vol spread (long - short)."""
    config = getattr(self, "_config", {})
    if config.get("is_cross_corridor", False):
        return None
    metadata = getattr(self, "_metadata", {}) or {}
    long_tickers = metadata.get("long_tickers", [])
    short_tickers = metadata.get("short_tickers", [])
    long_weights = metadata.get("long_weights", [])
    short_weights = metadata.get("short_weights", [])
    if not long_tickers:
        return None
    from datetime import date as _date
    start = self.timeseries.index[0] if len(self.timeseries) > 0 else _date.today()
    end = self.timeseries.index[-1] if len(self.timeseries) > 0 else _date.today()
    return entry_point(
        long_tickers=long_tickers, short_tickers=short_tickers,
        long_weights=long_weights, short_weights=short_weights,
        start_date=start, end_date=end,
        is_cross_corridor=config.get("is_cross_corridor", False),
    )

def _bt_plot_sectorial(self):
    """Sectorial pie chart(s) via Bloomberg sector data."""
    metadata = getattr(self, "_metadata", {}) or {}
    return graph_sectorial(
        tickers=metadata.get("long_tickers", []),
        short_tickers=metadata.get("short_tickers") or None,
        weights=metadata.get("long_weights") or None,
        short_weights=metadata.get("short_weights") or None,
    )

def _bt_plot_legs(self):
    """Individual per-stock P&L traces."""
    if self.per_leg_pnl is not None:
        return plot_per_stock_contributions(self.per_leg_pnl)
    return None

def _bt_rerun(self, n_exp: int = 60):
    """Re-backtest same basket with different maturity."""
    config = getattr(self, "_config", {})
    graph_data = getattr(self, "_graph_data", None) or {}
    original_df = graph_data.get("input_df")
    if original_df is None:
        raise ValueError("Cannot rerun — original input data not stored.")
    
    from functions.dispersion import run_backtest, SwapConfig
    
    # Reconstruct basket from metadata
    metadata = getattr(self, "_metadata", {}) or {}
    long_tickers = metadata.get("long_tickers", [])
    short_tickers = metadata.get("short_tickers", [])
    is_cross = config.get("is_cross_corridor", False)
    
    # Build SwapConfig
    is_vol_swap = config.get("product_type") == "vol_swap"
    if is_cross:
        config_obj = SwapConfig.cross_corridor(n_exp=n_exp)
    elif is_vol_swap:
        config_obj = SwapConfig.vol_swap(n_exp=n_exp)
    else:
        config_obj = SwapConfig.corridor(n_exp=n_exp)
    
    config_obj.start_date = config.get("start_date", config_obj.start_date)
    config_obj.adj_divs = config.get("adj_divs", True)
    
    # Build weights dict
    weights = {}
    for t in long_tickers:
        weights[t] = 1.0 / len(long_tickers) if long_tickers else 0
    
    # Run new backtest
    result = run_backtest(
        tickers=long_tickers,
        strikes={t: 0.05 for t in long_tickers},
        weights=weights,
        config=config_obj,
    )
    
    # Copy over metadata and graph_data
    result._metadata = metadata
    result._graph_data = graph_data
    result._config = dict(config, n_exp=n_exp)
    
    return result

def _bt_email(self, recipient: str = "", adj_divs: bool = False, progress_callback=None):
    """Generate and open Outlook email with all charts embedded."""
    metadata = getattr(self, "_metadata", {}) or {}
    config = getattr(self, "_config", {})
    graph_data = getattr(self, "_graph_data", {}) or {}
    return send_email_with_attachments(
        charts_data=graph_data, recipient_email=recipient,
        result_series=self.result_series,
        data_editor=metadata.get("input_df"),
        n_exp=config.get("n_exp", 310),
        is_cross_corridor=config.get("is_cross_corridor", False),
        product_type="Corridor Var Swap" if not config.get("is_vol_swap") else "Vol Swap",
        local_cap=config.get("local_cap", 2.5),
        barrier_up=config.get("barriers", (0.7, 1.3))[1],
        barrier_down=config.get("barriers", (0.7, 1.3))[0],
        adj_divs="Yes" if adj_divs else "No",
        progress_callback=progress_callback,
    )

# ── Patch BacktestResult ──
_BacktestResult.plot = _bt_plot
_BacktestResult.plot_split = _bt_plot_split
_BacktestResult.plot_entry_point = _bt_plot_entry_point
_BacktestResult.plot_sectorial = _bt_plot_sectorial
_BacktestResult.plot_legs = _bt_plot_legs
_BacktestResult.rerun = _bt_rerun
_BacktestResult.email = _bt_email
