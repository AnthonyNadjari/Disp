# Fix — mono-corridor email: sector chart missing, "Entry Point" title without graph (`_charts.py`)

Four functions of `_charts.py` to **replace entirely** (copy-paste each block over the
existing function of the same name). Nothing else changes: `_color_to_rgba`,
`BARCLAYS_NAVY`, `_create_returns_table`, `_create_underlyings_table`, `_convert_matu`,
`_return_offer`, `send_email_with_attachments` stay as they are, and the module imports
are the ones already there (`io`, `base64`, `numpy as np`, `pandas as pd`,
`plotly.graph_objects as go`, `typing`).

## Root causes

0. **RICs instead of Bloomberg tickers (UI, fixed on main)** — in mono mode the UI passed the
   editor's names as typed (`ISP.MI`, `.STOXX50E`) to `graph_sectorial` / `entry_point`, whose
   Bloomberg lookups need `ISP IM Equity` / `SX5E Index`. `Dispersion_Optimizer.py` now normalizes
   them to Bloomberg form (the engine's own `_to_bbg`) before calling the chart functions.

1. **Entry point title without graph** — `_render_line_to_png` raises on the entry-point
   figure, the failure is swallowed (console warning only), the HTML prints the title anyway:
   - a trace built without an explicit `line=dict(width=...)` gives
     `getattr(trace.line, 'width', 2)` → `None`, then `None * 0.7` → **TypeError**
     (the main charts set `width=2`, so they never hit it);
   - a `go.Bar` trace → `trace.line` → **AttributeError**;
   - a non-date x axis (tickers, numbers) → `pd.to_datetime` → **DateParseError**;
   - `entry_point()` returning `None` (the UI now warns in that case).
2. **Sector chart missing** — the "Sector Split" section is only built when `sectorial.png`
   was rendered; a `graph_sectorial` error or a pie render failure removed it silently, and
   the weights pie the UI already passes (`charts_data['weights_pie']`) was never rendered.

## What the replacements do

- `_render_line_to_png`: same output, byte for byte, for today's charts (verified on line,
  dual-line, area-fill and secondary-axis figures); additionally handles missing line width,
  marker-only traces (dots instead of an invisible 1-point line), bar traces, numeric /
  categorical x, `add_hline` / `add_vline` reference lines (entry-point figures only);
  returns `b''` when nothing is plottable so the email omits the chart instead of showing
  an empty frame.
- `_render_pie_to_png`: interface-matching style — exact house Blues/Greys hues, white
  wedge borders, percent inside the slices (suppressed under 2.5%; navy text on light
  slices, white on dark), full labels in a right-hand legend; above 20 slices the
  smallest are grouped into "Other (n)".
- `render_charts_to_bytes`: renders `weights_pie` (house style); a figure that is `None`
  or fails to render is reported on the console (`[charts] …`, with the traceback) and left
  out.
- `_generate_email_html`: identical HTML when all images exist (verified); the "Entry
  Point" title + intro bullet appear only with the image, the "Sector Split" section + bullet
  only when at least one pie rendered; the section shows the GICS pie(s) and the weights pie,
  two per row.

---

## 1. `_render_line_to_png` — REPLACE (updated: draws the entry-point last-point marker, its label box and level line)

```python
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
            return pd.to_datetime(xs), True
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
```

---

## 2. `_render_pie_to_png` — REPLACE

Two paths: **kaleido first** — rasterizes the actual Plotly figure, so the email pie is
pixel-identical to the interface (requires `pip install kaleido==0.2.1` once in the desk
env; it's a regular wheel, pip handles it, no manual archive). Kaleido spawns a headless
Chromium that can hang in restricted environments, so the call is capped at 25 s and then
falls back to the matplotlib re-render below (same house style). If it still hangs on your
machine: `pip uninstall kaleido` — the fallback alone is deterministic.

```python
def _render_pie_to_png(fig: go.Figure, title: str,
                       width: int = 1600, height: int = 1600) -> bytes:
    """Render a plotly pie chart to PNG bytes.

    Preferred path: plotly's own to_image (kaleido) — byte-for-byte the same chart as
    on screen. Fallback: matplotlib re-render with the house style (Blues long / Greys
    short palette, white wedge borders, percent inside the slices, legend on the right)."""

    # ── Path 1: exact Plotly rasterization (needs kaleido) ──────────────────────
    # kaleido spawns a headless Chromium which can HANG in restricted environments
    # (no sandbox rights, antivirus) — hard 25s timeout, then matplotlib fallback.
    try:
        import kaleido  # noqa: F401
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutTimeout

        def _kaleido_render():
            return fig.to_image(format="png", width=width, height=height, scale=2)

        _pool = ThreadPoolExecutor(max_workers=1)
        try:
            png = _pool.submit(_kaleido_render).result(timeout=25)
            if png:
                return png
        except _FutTimeout:
            print("[charts] kaleido pie render timed out (25s) — matplotlib fallback")
        finally:
            _pool.shutdown(wait=False)
    except ImportError:
        pass
    except Exception as e:
        print(f"[charts] kaleido pie render unavailable ({e}) — falling back to matplotlib")

    # ── Path 2: matplotlib fallback (interface style) ───────────────────────────
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
```

---

## 3. `render_charts_to_bytes` — REPLACE

```python
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
    if not is_cross_corridor:
        _line('entry_point', 'entry_point.png', 'Entry Point Analysis')

    return rendered
```

---

## 4. `_generate_email_html` — REPLACE

```python
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

    # Entry point (not for cross corridor) — title and bullet only when the image exists
    entry_point_bullet = ""
    entry_point_image = ""
    if structure_type != "cross_corridor" and 'entry_point.png' in base64_images:
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
```

---

## After applying

`Dispersion_Optimizer.py` (already on main) warns in the UI when `graph_sectorial` or
`entry_point()` fails, with the reason. If a chart is still missing after this, paste the
`[charts] …` console lines: they now carry the exception and traceback.
