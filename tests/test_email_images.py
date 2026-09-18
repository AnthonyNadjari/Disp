"""Email images must be cid: attachments, never data: URIs.

Outlook's renderer does not support ``data:`` URIs: it rewrites them into
embedded DIB pictures at send time, and the copy degrades to
"<< OLE Object: Picture (Device Independent Bitmap) >>" as soon as the message
is forwarded, replied to or pasted into a new one. Only a real attachment
referenced by Content-ID survives.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("plotly")
pytest.importorskip("matplotlib")
from functions.dispersion._charts import _generate_email_html  # noqa: E402

KEYS = ["line_graph.png", "graph_60d.png", "sectorial.png", "weights_pie.png", "entry_point.png"]


def _html(**kw):
    idx = pd.bdate_range("2025-01-01", periods=60)
    series = pd.Series(np.linspace(0, 2, 60), index=idx)
    basket = pd.DataFrame([{"Variance Asset": "AAPL UW Equity",
                            "Strike Mono Var Swap (%)": 20.0, "Weight (%)": 100.0}])
    return _generate_email_html(
        {"has_short_leg": False}, series, basket, 60, False, "Vol Swap",
        2.5, 1.3, 0.7, "SPX Index", "No", carry_series=series, **kw)


def test_cid_mode_emits_no_data_uri():
    html = _html(cid_map={k: k.split(".")[0] for k in KEYS})
    assert html.count("data:image") == 0, "a data: URI would become an OLE picture in Outlook"
    assert html.count('src="cid:') == len(KEYS)
    for k in KEYS:
        assert f'src="cid:{k.split(".")[0]}"' in html


def test_preview_mode_still_uses_data_uris():
    """The browser preview (Streamlit) has no attachments to reference."""
    html = _html(base64_images={k: "QUJD" for k in KEYS})
    assert html.count("data:image") == len(KEYS)
    assert 'src="cid:' not in html


def test_sections_render_in_cid_mode():
    """Section gating keys off the image map — it must see cid entries too,
    otherwise Entry Point / Sector Split silently vanish from the email."""
    html = _html(cid_map={k: k.split(".")[0] for k in KEYS})
    assert "Entry" in html and "Sector" in html


def test_no_images_no_img_tags():
    assert _html().count("<img") == 0


# ── Vol-swap basket columns ───────────────────────────────────────────────────
# A vol-swap basket carries 'Underlying' / 'Strike (%)'; the email reads the
# canonical 'Variance Asset' / 'Strike Mono Var Swap (%)'. Without the alias
# mapping the underlyings table printed blank tickers and zero strikes, and the
# trade description showed "Offer @ N/A".

VOLSWAP_BASKET = pd.DataFrame([
    {"Underlying": "AAPL UW Equity", "Strike (%)": 22.5, "Weight (%)": 60.0},
    {"Underlying": "MSFT UW Equity", "Strike (%)": 19.0, "Weight (%)": 40.0},
])


def _html_with(basket):
    from functions.dispersion._charts import _generate_email_html
    idx = pd.bdate_range("2025-01-01", periods=60)
    series = pd.Series(np.linspace(0, 2, 60), index=idx)
    return _generate_email_html(
        {"has_short_leg": False}, series, basket, 60, False, "Vol Swap",
        2.5, 1.3, 0.7, "SPX Index", "No", carry_series=series)


def test_volswap_basket_fills_the_underlyings_table():
    html = _html_with(VOLSWAP_BASKET)
    assert "AAPL UW Equity" in html and "MSFT UW Equity" in html
    assert "22.50" in html or "22.5" in html


def test_volswap_basket_gives_a_real_offer():
    html = _html_with(VOLSWAP_BASKET)
    assert "Offer @ N/A" not in html
    assert "Offer @ 21.10%" in html          # weight-averaged 22.5/19.0 at 60/40


def test_canonical_basket_df_is_non_destructive():
    from functions.dispersion._charts import _canonical_basket_df
    canonical = pd.DataFrame([{"Variance Asset": "X", "Strike Mono Var Swap (%)": 20.0,
                               "Weight (%)": 100.0}])
    assert _canonical_basket_df(canonical) is canonical      # already canonical: untouched
    before = list(VOLSWAP_BASKET.columns)
    out = _canonical_basket_df(VOLSWAP_BASKET)
    assert "Variance Asset" in out.columns and "Strike Mono Var Swap (%)" in out.columns
    assert list(VOLSWAP_BASKET.columns) == before            # caller's frame not mutated
    assert _canonical_basket_df(None) is None
