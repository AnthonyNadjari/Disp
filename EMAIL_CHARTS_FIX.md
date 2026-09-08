# Fix — mono-corridor email: sector chart missing, "Entry Point" title without graph (`_charts.py`)

Diagnosis from `render_charts_to_bytes` + `_generate_email_html`:

- **Sector chart**: the email's *Sector Split* section only appears when `sectorial.png`
  was rendered. For mono it isn't — either `graph_sectorial` returned `{'error': …}`
  (the UI swallowed it silently; it now warns) or `_render_pie_to_png` raised (only a
  `[charts] Warning:` console print). On top, the **weights pie** (`charts_data['weights_pie']`)
  is never rendered at all, in any mode.
- **Entry point**: `_generate_email_html` emits the "Entry Point" title for every
  non-cross structure, but the image only if `entry_point.png` was rendered — and
  `_line(...)` uses `_render_line_to_png`, which only re-draws line traces; when the
  entry-point figure has bars/markers/histogram the render raises, the image is
  skipped, the title stays.

Three edits below. `Dispersion_Optimizer.py` (already on main) now also surfaces the
underlying error (`st.warning` + a console `[charts] …` line) so the next run tells
you *why* a chart failed.

---

## Edit 1 — ADD a generic plotly → PNG renderer (next to `_render_line_to_png`)

Handles Scatter (lines / markers / text), Bar (grouped / stacked / horizontal),
Histogram, Pie, and layout shapes (h/v lines). Used as the entry-point renderer and
as fallback for the pies.

```python
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
```

---

## Edit 2 — in `render_charts_to_bytes`: fallback renderers, weights pie, entry point

**Replace** the two helpers and the tail of the function:

```python
    def _line(key, filename, title, w=3200, h=1800):
        val = charts_data.get(key)
        if val is None:
            return
        fig = go.Figure(val) if not isinstance(val, go.Figure) else val
        for renderer in (_render_line_to_png, _render_any_to_png):
            try:
                png = renderer(fig, title, w, h)
                if png and len(png) > 200:
                    rendered.append((filename, png))
                    return
            except Exception as e:
                print(f"[charts] Warning: {renderer.__name__} failed for '{filename}': {e}")
        print(f"[charts] ERROR: '{filename}' could not be rendered by any renderer — it will be missing from the email")

    def _pie(key, filename, title):
        val = charts_data.get(key)
        if val is None:
            return
        fig = go.Figure(val) if not isinstance(val, go.Figure) else val
        for renderer in (_render_pie_to_png, lambda f, t: _render_any_to_png(f, t, 2400, 1600)):
            try:
                png = renderer(fig, title)
                if png and len(png) > 200:
                    rendered.append((filename, png))
                    return
            except Exception as e:
                print(f"[charts] Warning: pie render failed for '{filename}': {e}")
        print(f"[charts] ERROR: '{filename}' could not be rendered by any renderer — it will be missing from the email")

    _line('main_graph', 'line_graph.png', 'Strategy Performance')
    if has_short_leg:
        _line('split_graph', 'graph_split.png', 'Strategy Split Analysis')
    _line('graph_60d', 'graph_60d.png', '3M Carry Analysis')
    if has_short_leg:
        _line('split_60d', 'graph_60d_split.png', '3M Carry Split Analysis')
    # Sector charts: GICS split (from graph_sectorial) + weights-by-stock pie (always)
    if is_dual:
        _pie('sectorial_long', 'sectorial_long.png', 'Long Basket Sectors')
        _pie('sectorial_short', 'sectorial_short.png', 'Short Basket Sectors')
    elif charts_data.get('sectorial') is not None:
        _pie('sectorial', 'sectorial.png', 'Sector Repartition')
    _pie('weights_pie', 'weights_pie.png', 'Weights by stock')
    if not is_cross_corridor:
        # entry-point figures carry bars / markers → generic renderer first
        val = charts_data.get('entry_point')
        if val is not None:
            fig = go.Figure(val) if not isinstance(val, go.Figure) else val
            try:
                png = _render_any_to_png(fig, 'Entry Point Analysis', 3200, 1800)
                if png and len(png) > 200:
                    rendered.append(('entry_point.png', png))
            except Exception as e:
                print(f"[charts] Warning: entry point render failed: {e}")
                _line('entry_point', 'entry_point.png', 'Entry Point Analysis')

    return rendered
```

---

## Edit 3 — in `_generate_email_html`: no orphan title, weights pie in the sector section

**Replace** the sector block:

```python
    # Image sections — sector split (GICS) + weights-by-stock pie; the section
    # appears when at least one of them was rendered
    _pie_cells = []
    if is_dual_sectorial:
        _pie_cells += [_img_pie("sectorial_long.png"), _img_pie("sectorial_short.png")]
    elif base64_images and 'sectorial.png' in base64_images:
        _pie_cells.append(_img_pie("sectorial.png"))
    if base64_images and 'weights_pie.png' in base64_images:
        _pie_cells.append(_img_pie("weights_pie.png"))
    _pie_cells = [c for c in _pie_cells if c]
    sectorial_html = (
        '<table cellpadding="0" cellspacing="0"><tr>'
        + ''.join(f'<td width="700" align="center">{c}</td>' for c in _pie_cells)
        + '</tr></table>'
    ) if _pie_cells else ''
```

**Replace** the entry-point condition (one line) so the title only appears with its image:

```python
    if structure_type != "cross_corridor" and base64_images and 'entry_point.png' in base64_images:
```

(the bullet in the intro list and the image block are both inside that `if`, so both
disappear together when the chart is unavailable).

---

## After applying

Run one mono email. The UI now prints a warning naming the failing chart and the
console shows `[charts] …` lines with the exception — paste them if anything is
still missing; that is the last piece of information needed to fix the chart
functions themselves (`graph_sectorial` / `entry_point`).
