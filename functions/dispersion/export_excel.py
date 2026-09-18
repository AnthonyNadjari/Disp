"""
Export dispersion Result Matrix to a presentation-ready Excel workbook.
Matches the compact target layout exactly.

Key design:
- Full FPF strings stay in the main sheet (not moved to a separate sheet).
- FPF strings do NOT expand column widths or spill visually.
- Exact RGB colours, no theme colours.
- Compact data columns (11-13), metric column (36).
- LCM columns carry a per-set suffix " [name]" (several LCM parameter sets
  per run): every LCM rule below matches by PREFIX, never by exact name.
- EV / FV / impact columns are exported with 4 decimals (0.0000%); strikes,
  RA, vols, correlation keep 2.
"""
import io
from typing import Optional

import pandas as pd

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False


# ─── Exact RGB hex colours ────────────────────────────────────────────────────
_DARK_BLUE = "1F4E78"
_ORANGE = "F4B183"
_LIGHT_BLUE = "D9EAF7"
_WHITE = "FFFFFF"
_BLACK = "000000"
_GREY = "D0D0D0"

# ─── Fills ─────────────────────────────────────────────────────────────────────
_FILL_DARK_BLUE = PatternFill(start_color=_DARK_BLUE, end_color=_DARK_BLUE, fill_type="solid")
_FILL_ORANGE = PatternFill(start_color=_ORANGE, end_color=_ORANGE, fill_type="solid")
_FILL_LIGHT_BLUE = PatternFill(start_color=_LIGHT_BLUE, end_color=_LIGHT_BLUE, fill_type="solid")
_FILL_WHITE = PatternFill(start_color=_WHITE, end_color=_WHITE, fill_type="solid")

# ─── Fonts ─────────────────────────────────────────────────────────────────────
_FONT_WHITE_BOLD = Font(bold=True, color=_WHITE, size=10)
_FONT_BLACK = Font(bold=False, color=_BLACK, size=10)
_FONT_BLACK_BOLD = Font(bold=True, color=_BLACK, size=10)

# ─── Borders ──────────────────────────────────────────────────────────────────
_SIDE_THIN_GREY = Side(style="thin", color=_GREY)
_SIDE_THICK_BLUE = Side(style="medium", color=_DARK_BLUE)

_BORDER_THIN = Border(
    left=_SIDE_THIN_GREY, right=_SIDE_THIN_GREY,
    top=_SIDE_THIN_GREY, bottom=_SIDE_THIN_GREY,
)
_BORDER_SECTION_TOP = Border(
    left=_SIDE_THIN_GREY, right=_SIDE_THIN_GREY,
    top=_SIDE_THICK_BLUE, bottom=_SIDE_THIN_GREY,
)

# ─── Alignment ────────────────────────────────────────────────────────────────
_ALIGN_LEFT = Alignment(horizontal="left", vertical="center", wrap_text=False)
_ALIGN_CENTER = Alignment(horizontal="center", vertical="center", wrap_text=False)

# ─── Dark blue metadata row labels ────────────────────────────────────────────
_DARK_BLUE_LABELS = {
    "Index Ticker", "Tickers", "Ticker", "Variance Asset",
    "Corridor Condition Asset", "Corridor Asset",
    "Weight", "Weight (%)",
    "Currency",
}

# ─── Orange row prefixes ──────────────────────────────────────────────────────
_ORANGE_PREFIXES = ("Strike", "Initial Strike", "Final Strike", "LSV Charge", "Cap Theoretical")

# ─── Light blue row matches ───────────────────────────────────────────────────
_LIGHT_BLUE_PREFIXES = ("EV", "ATMF", "ATMS", "Vol Spread")
_LIGHT_BLUE_EXACT = {"RA (%)", "Obs Dates Cross", "Obs Dates Mono", "Obs dates Cross",
                     "Obs dates Mono", "Correlation"}

# ─── Section separator triggers ───────────────────────────────────────────────
_SECTION_TRIGGERS = [
    "Strike Cross Corr LV",
    "Strike Cross Corr Cap Priced LV",
    "EV Cross LV",
    "EV Cap Cross LV",
    "RA (%)",
    "Obs Dates Cross",
    "Obs dates Cross",
    "ATMF Vol Variance",
    "Correlation",
]

# ─── Priority columns that get orange highlight in horizontal export ──────
# Matched by PREFIX: the LCM cap-priced columns carry a per-set suffix
# ("Strike Cross Corr Cap Priced LCM [A] (%)"), and mono-corridor pricing emits
# its own shorter names ("Strike Cap Priced LV (%)").
_PRIORITY_HIGHLIGHT_PREFIXES = (
    "Strike Cross Corr Cap Priced",     # cross mode, cross leg (LV / LSV / LCM [set])
    "Strike Mono Corr Cap Priced",      # cross mode, mono leg
    "Strike Cap Priced",                # mono-corridor mode
)
_LCM_CAP_PRICED_PREFIX = "Strike Cross Corr Cap Priced LCM"

# ─── Columns exported with 4 decimals (EVs are small numbers; 2 hide the LCM impacts)
_FOUR_DECIMAL_PREFIXES = ("EV ", "FV Variance", "LSV Impact", "LCM Impact")


def _is_priority_highlight(label: str) -> bool:
    return label.startswith(_PRIORITY_HIGHLIGHT_PREFIXES)


def _get_row_style(label: str, horizontal: bool = False):
    """Return (fill, font_data) for a metric label."""
    lbl = label.strip()
    if lbl in _DARK_BLUE_LABELS:
        return _FILL_DARK_BLUE, _FONT_WHITE_BOLD
    if horizontal:
        # In horizontal mode, only priority columns get orange
        if _is_priority_highlight(lbl):
            return _FILL_ORANGE, _FONT_BLACK
    else:
        # Vertical mode: all Strike/Cap rows get orange
        if any(lbl.startswith(p) for p in _ORANGE_PREFIXES):
            return _FILL_ORANGE, _FONT_BLACK
    if lbl in _LIGHT_BLUE_EXACT or any(lbl.startswith(p) for p in _LIGHT_BLUE_PREFIXES):
        return _FILL_LIGHT_BLUE, _FONT_BLACK
    return _FILL_WHITE, _FONT_BLACK


def _is_fpf_row(label: str) -> bool:
    lbl = label.strip().lower()
    return lbl.startswith("fpf") or lbl.startswith("ffp")


def _needs_section_border(label: str, seen_fpf: bool, seen_sparx: bool) -> bool:
    lbl = label.strip()
    for trigger in _SECTION_TRIGGERS:
        if lbl.startswith(trigger):
            return True
    if _is_fpf_row(lbl) and not seen_fpf:
        return True
    if "Sparx Notional" in lbl and not seen_sparx:
        return True
    return False


def _parse_value(val):
    """Parse display string to numeric for Excel."""
    if val is None or str(val).strip() in ("", "N/A", "FAILED", "nan", "None"):
        return ""
    s = str(val).strip()
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except (ValueError, TypeError):
            return s
    if "," in s and not any(c.isalpha() for c in s):
        try:
            return float(s.replace(",", ""))
        except (ValueError, TypeError):
            return s
    try:
        f = float(s)
        if f == int(f) and "." not in s:
            return int(f)
        return f
    except (ValueError, TypeError):
        return s


def _number_format_for(label: str) -> str:
    lbl = label.strip()
    if "Obs Date" in lbl or "Obs dates" in lbl:
        return "0"
    if "Sparx Notional" in lbl or "Notional" in lbl:
        return "#,##0.00"
    if "Impact" in lbl and "bp" in lbl.lower():
        return "0.00"
    if lbl.startswith(_FOUR_DECIMAL_PREFIXES) and "(%" in lbl:
        return "0.0000%"
    if "(%" in lbl or lbl == "Correlation" or lbl == "RA (%)":
        return "0.00%"
    return ""


def _priority_columns(metrics):
    """Priority columns first, in the fixed LV / LSV / LCM… / mono order —
    every LCM cap-priced set column slots in after the LSV one. Covers both
    cross-corridor and mono-corridor column naming."""
    order = [
        'Index Ticker',                          # cross mode
        'Ticker',                                # mono mode
        'Corridor Asset',
        'Currency',
        # cross-corridor pricing
        'Strike Cross Corr Cap Priced LV (%)',
        'Strike Cross Corr Cap Priced LSV (%)',
        '__LCM_CAP_PRICED__',
        'Strike Mono Corr Cap Priced LV (%)',
        'Strike Mono Corr Cap Priced LSV (%)',
        # mono-corridor pricing
        'Strike Cap Priced LV (%)',
        'Strike Cap Priced LSV (%)',
        'Strike Cap Priced LCM (%)',
    ]
    present = []
    for c in order:
        if c == '__LCM_CAP_PRICED__':
            present.extend(m for m in metrics if str(m).startswith(_LCM_CAP_PRICED_PREFIX))
        elif c in metrics:
            present.append(c)
    return present


def export_result_matrix(df: pd.DataFrame, horizontal: bool = False) -> Optional[bytes]:
    """
    Export the result matrix to a formatted Excel workbook.

    Args:
        df: DataFrame to export.
            - If horizontal=False (default): rows=metrics, columns=assets (transposed UI view)
            - If horizontal=True: rows=assets, columns=metrics (tickers as rows)
        horizontal: If True, exports with tickers as rows and metrics as columns.

    Preserves exact row and column order from UI.
    Full FPF strings stay in main sheet but don't expand columns.
    """
    if not HAS_OPENPYXL:
        return None

    wb = Workbook()
    ws = wb.active
    ws.title = "Result Matrix"

    if horizontal:
        return _export_horizontal(df, wb, ws)

    # ─── Vertical (default): rows=metrics, columns=assets ─────────────────────

    n_data_cols = len(df.columns)
    total_cols = n_data_cols + 1  # +1 for metric label column A

    # ─── Write all rows directly (no extra header row) ────────────────────────
    seen_fpf = False
    seen_sparx = False
    row_num = 0

    for label, row_data in df.iterrows():
        row_num += 1
        label_str = str(label).strip()
        is_fpf = _is_fpf_row(label_str)

        # Section border logic
        needs_border = _needs_section_border(label_str, seen_fpf, seen_sparx)
        if is_fpf:
            seen_fpf = True
        if "Sparx Notional" in label_str:
            seen_sparx = True

        # Style
        fill, font_data = _get_row_style(label_str)
        border = _BORDER_SECTION_TOP if needs_border else _BORDER_THIN
        num_fmt = _number_format_for(label_str)
        is_dark_blue = (fill == _FILL_DARK_BLUE)

        # Column A: metric label
        cell_a = ws.cell(row=row_num, column=1, value=label_str)
        cell_a.fill = fill
        cell_a.font = _FONT_WHITE_BOLD if is_dark_blue else _FONT_BLACK_BOLD
        cell_a.alignment = _ALIGN_LEFT
        cell_a.border = border

        # Data columns
        for col_idx, val in enumerate(row_data, start=2):
            if is_fpf:
                # Write full FPF string as text, no parsing
                cell_val = str(val) if val is not None and str(val).strip() not in ("", "nan", "None") else ""
                cell = ws.cell(row=row_num, column=col_idx, value=cell_val)
                cell.number_format = "@"  # Text format
            else:
                parsed = _parse_value(val)
                cell = ws.cell(row=row_num, column=col_idx, value=parsed)
                if isinstance(parsed, (int, float)) and num_fmt:
                    cell.number_format = num_fmt

            cell.fill = fill
            cell.font = _FONT_WHITE_BOLD if is_dark_blue else font_data
            cell.alignment = _ALIGN_CENTER
            cell.border = border

        # ── FPF anti-spill: ensure cells to the right are not empty ──
        # Write a single space in the column after the last data column
        # to prevent Excel from visually spilling long text rightward.
        if is_fpf:
            spill_col = n_data_cols + 2
            spill_cell = ws.cell(row=row_num, column=spill_col, value=" ")
            spill_cell.alignment = _ALIGN_CENTER

    # ─── Column widths: fixed, compact, FPF-proof ─────────────────────────────
    ws.column_dimensions['A'].width = 36
    for col_idx in range(2, total_cols + 1):
        ws.column_dimensions[get_column_letter(col_idx)].width = 12

    # ─── Outer border around used range ───────────────────────────────────────
    for r in range(1, row_num + 1):
        # Left edge
        c = ws.cell(row=r, column=1)
        c.border = Border(
            left=_SIDE_THICK_BLUE,
            right=c.border.right, top=c.border.top, bottom=c.border.bottom,
        )
        # Right edge
        c = ws.cell(row=r, column=total_cols)
        c.border = Border(
            right=_SIDE_THICK_BLUE,
            left=c.border.left, top=c.border.top, bottom=c.border.bottom,
        )
    for col in range(1, total_cols + 1):
        # Top edge
        c = ws.cell(row=1, column=col)
        c.border = Border(
            top=_SIDE_THICK_BLUE,
            left=c.border.left, right=c.border.right, bottom=c.border.bottom,
        )
        # Bottom edge
        c = ws.cell(row=row_num, column=col)
        c.border = Border(
            bottom=_SIDE_THICK_BLUE,
            left=c.border.left, right=c.border.right, top=c.border.top,
        )

    # ─── Freeze panes at B5 ──────────────────────────────────────────────────
    freeze_row = min(5, row_num + 1)
    ws.freeze_panes = f"B{freeze_row}"

    # ─── Simple autofilter (no table theme) ───────────────────────────────────
    ws.auto_filter.ref = f"A1:{get_column_letter(total_cols)}{row_num}"

    # ─── Save ─────────────────────────────────────────────────────────────────
    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()


def _export_horizontal(df: pd.DataFrame, wb: "Workbook", ws) -> Optional[bytes]:
    """
    Export with tickers as rows, metrics as columns (horizontal layout).
    Header row = metric names (formatted), data rows = one per ticker.
    """
    metrics = list(df.columns)
    tickers = list(df.index)
    n_tickers = len(tickers)

    # ─── Reorder columns: priority columns first ────────────────────────────
    _priority_present = _priority_columns(metrics)
    _rest = [c for c in metrics if c not in _priority_present]
    metrics = _priority_present + _rest
    n_metrics = len(metrics)
    # Reindex df columns to match new order
    df = df[metrics]

    # ─── Row 1: Header row (metric names) ────────────────────────────────────
    for col_idx, metric in enumerate(metrics, start=2):
        fill, _ = _get_row_style(str(metric), horizontal=True)
        cell = ws.cell(row=1, column=col_idx, value=str(metric))
        cell.fill = fill
        cell.font = _FONT_WHITE_BOLD if fill == _FILL_DARK_BLUE else _FONT_BLACK_BOLD
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = _BORDER_THIN

    # Column A header
    cell_a = ws.cell(row=1, column=1, value="Ticker")
    cell_a.fill = _FILL_DARK_BLUE
    cell_a.font = _FONT_WHITE_BOLD
    cell_a.alignment = _ALIGN_CENTER
    cell_a.border = _BORDER_THIN

    # ─── Data rows: one per ticker ────────────────────────────────────────────
    for row_idx, ticker in enumerate(tickers, start=2):
        # Column A: ticker name
        cell = ws.cell(row=row_idx, column=1, value=str(ticker))
        cell.fill = _FILL_DARK_BLUE
        cell.font = _FONT_WHITE_BOLD
        cell.alignment = _ALIGN_LEFT
        cell.border = _BORDER_THIN

        for col_idx, metric in enumerate(metrics, start=2):
            val = df.iloc[row_idx - 2, col_idx - 2]
            metric_str = str(metric)
            is_fpf = _is_fpf_row(metric_str)
            num_fmt = _number_format_for(metric_str)

            if is_fpf:
                cell_val = str(val) if val is not None and str(val).strip() not in ("", "nan", "None") else ""
                cell = ws.cell(row=row_idx, column=col_idx, value=cell_val)
                cell.number_format = "@"
            else:
                parsed = _parse_value(val)
                cell = ws.cell(row=row_idx, column=col_idx, value=parsed)
                if isinstance(parsed, (int, float)) and num_fmt:
                    cell.number_format = num_fmt

            cell.fill = _FILL_WHITE
            cell.font = _FONT_BLACK
            cell.alignment = _ALIGN_CENTER
            cell.border = _BORDER_THIN

    # ─── Column widths ────────────────────────────────────────────────────────
    total_cols = n_metrics + 1
    total_rows = n_tickers + 1
    ws.column_dimensions['A'].width = 16
    for col_idx in range(2, total_cols + 1):
        ws.column_dimensions[get_column_letter(col_idx)].width = 13

    # ─── Row height for header (rotated text) ────────────────────────────────
    ws.row_dimensions[1].height = 100

    # ─── Outer border ─────────────────────────────────────────────────────────
    for r in range(1, total_rows + 1):
        c = ws.cell(row=r, column=1)
        c.border = Border(left=_SIDE_THICK_BLUE, right=c.border.right, top=c.border.top, bottom=c.border.bottom)
        c = ws.cell(row=r, column=total_cols)
        c.border = Border(right=_SIDE_THICK_BLUE, left=c.border.left, top=c.border.top, bottom=c.border.bottom)
    for col in range(1, total_cols + 1):
        c = ws.cell(row=1, column=col)
        c.border = Border(top=_SIDE_THICK_BLUE, left=c.border.left, right=c.border.right, bottom=c.border.bottom)
        c = ws.cell(row=total_rows, column=col)
        c.border = Border(bottom=_SIDE_THICK_BLUE, left=c.border.left, right=c.border.right, top=c.border.top)

    # ─── Freeze panes at B2 ──────────────────────────────────────────────────
    ws.freeze_panes = "B2"

    # ─── Autofilter ───────────────────────────────────────────────────────────
    ws.auto_filter.ref = f"A1:{get_column_letter(total_cols)}{total_rows}"

    # ─── Save ─────────────────────────────────────────────────────────────────
    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()
