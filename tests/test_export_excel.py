"""Excel export: LCM per-set columns are orange + priority-ordered, EVs keep 4 decimals."""
import io
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

openpyxl = pytest.importorskip("openpyxl")
from functions.dispersion import export_excel as ex  # noqa: E402


def _df():
    return pd.DataFrame([{
        "Index Ticker": "X.PA", "Corridor Asset": ".STOXX50E", "Currency": "EUR",
        "Strike Cross Corr LV (%)": "20.00%",
        "Strike Cross Corr LCM [A] (%)": "20.10%",
        "Strike Cross Corr Cap Priced LV (%)": "19.50%",
        "Strike Cross Corr Cap Priced LSV (%)": "19.60%",
        "Strike Cross Corr Cap Priced LCM [A] (%)": "19.80%",
        "Strike Cross Corr Cap Priced LCM [B] (%)": "19.90%",
        "Strike Mono Corr Cap Priced LV (%)": "21.50%",
        "EV Cross LV (%)": "-3.1234%",
        "EV Cross LCM0 [A] (%)": "-3.0500%",
        "LCM Impact Cross [A] (%)": "-0.0734%",
        "RA (%)": "85.00%",
        "FPF Cross LV Cap": "corridorCovarianceSwap_v4 (...)",
    }], index=["X.PA"])


def _sheet(horizontal):
    # the page passes the UI frame transposed (rows = metrics) for the vertical layout
    frame = _df() if horizontal else _df().T
    data = ex.export_result_matrix(frame, horizontal=horizontal)
    return openpyxl.load_workbook(io.BytesIO(data)).active


def test_horizontal_lcm_cap_priced_orange_and_ordered():
    ws = _sheet(horizontal=True)
    headers = [c.value for c in ws[1]][1:]
    fills = {c.value: c.fill.start_color.rgb[-6:] for c in ws[1]}
    for h in ("Strike Cross Corr Cap Priced LV (%)", "Strike Cross Corr Cap Priced LCM [A] (%)",
              "Strike Cross Corr Cap Priced LCM [B] (%)", "Strike Mono Corr Cap Priced LV (%)"):
        assert fills[h] == ex._ORANGE, h
    assert fills["Strike Cross Corr LCM [A] (%)"] != ex._ORANGE      # uncapped: not orange
    # priority order: LV, LSV, LCM [A], LCM [B], then mono
    i = headers.index
    assert i("Strike Cross Corr Cap Priced LSV (%)") < i("Strike Cross Corr Cap Priced LCM [A] (%)") \
        < i("Strike Cross Corr Cap Priced LCM [B] (%)") < i("Strike Mono Corr Cap Priced LV (%)")


def test_number_formats_four_decimals_for_evs():
    ws = _sheet(horizontal=True)
    fmt = {c.value: ws.cell(row=2, column=c.column).number_format for c in ws[1] if c.value}
    assert fmt["EV Cross LV (%)"] == "0.0000%"
    assert fmt["EV Cross LCM0 [A] (%)"] == "0.0000%"
    assert fmt["LCM Impact Cross [A] (%)"] == "0.0000%"
    assert fmt["Strike Cross Corr Cap Priced LCM [A] (%)"] == "0.00%"
    assert fmt["RA (%)"] == "0.00%"
    val = {c.value: ws.cell(row=2, column=c.column).value for c in ws[1] if c.value}
    assert val["EV Cross LV (%)"] == pytest.approx(-0.031234)


def test_vertical_export_runs_and_colours_strikes():
    ws = _sheet(horizontal=False)
    rows = {ws.cell(row=r, column=1).value: ws.cell(row=r, column=1).fill.start_color.rgb[-6:]
            for r in range(1, ws.max_row + 1)}
    assert rows["Strike Cross Corr Cap Priced LCM [A] (%)"] == ex._ORANGE
    assert rows["EV Cross LCM0 [A] (%)"] == ex._LIGHT_BLUE
