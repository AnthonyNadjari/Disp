"""Vega cap/target parsing: robust formats, loud on garbage (no silent uncapped)."""

import pandas as pd
import pytest

from functions.dispersion._api import _df_to_legs


def _df(cap):
    return pd.DataFrame({
        "Variance Asset": ["A US Equity"],
        "Strike Mono Var Swap (%)": [25.0],
        "Axe Cap": [cap],
    })


@pytest.mark.parametrize("raw,expected", [
    (100.0, 100.0), ("100", 100.0), ("100 000", 100000.0), ("1,000", 1000.0),
    ("100k", 100000.0), ("100K", 100000.0), ("2M", 2000000.0),
    ("", None), ("nan", None), (None, None),
])
def test_axe_cap_formats(raw, expected):
    legs = _df_to_legs(_df(raw), is_cross_corridor=False)
    assert legs[0].axe_cap == expected


def test_axe_cap_garbage_raises_loudly():
    with pytest.raises(ValueError, match="not a number"):
        _df_to_legs(_df("abc"), is_cross_corridor=False)
