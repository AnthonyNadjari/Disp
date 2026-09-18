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
