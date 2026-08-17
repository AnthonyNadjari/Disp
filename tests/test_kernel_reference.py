"""Phase E — payoff kernels vs naive reference implementations.

Each numba kernel is compared against a deliberately simple, loop-based
re-implementation of the documented payoff math on gapped random data.
Also pins the legacy Gaia_PP quirk (vol swap drops the window's first daily
return: sq_logs[1:]) so any silent change to it fails a test.
"""

from __future__ import annotations

import numpy as np

from functions.dispersion._backtester import (
    _leg_pnl_cross_corridor,
    _rolling_pnl_corridor,
    _rolling_pnl_volswap,
)
from functions.dispersion.models import DispersionConfig, DispersionLeg
from functions.dispersion._api import _to_swap_config

N, N_EXP, K, CAP = 160, 15, 0.10, 2.5
UBAR, DBAR = 1.10, 0.90


def _gapped(seed, n=N, gaps=((40, 43), (90, 91))):
    rng = np.random.default_rng(seed)
    px = 100 * np.cumprod(1 + rng.normal(0, 0.02, n))
    for a, b in gaps:
        px[a:b] = np.nan
    return px


def _naive_volswap(prices):
    """Last n_exp+1 VALID observations ending at a valid day i; realized vol
    from the window's daily log returns, SKIPPING the first one (legacy)."""
    out = np.full(len(prices), np.nan)
    valid = [j for j in range(len(prices)) if not np.isnan(prices[j])]
    for i in range(len(prices)):
        if np.isnan(prices[i]):
            continue
        upto = [j for j in valid if j <= i]
        if len(upto) < N_EXP + 1:
            continue
        w = prices[upto[-(N_EXP + 1):]]
        rets2 = [np.log(w[j + 1] / w[j]) ** 2 for j in range(len(w) - 1)]
        realized = np.sqrt(sum(rets2[1:]) * 252.0 / N_EXP)   # sq_logs[1:] quirk
        out[i] = min(realized, CAP * K) - K
    return out


def _naive_corridor(pv, pc):
    """Jointly-valid window; corridor barriers off the window's FIRST corridor
    price; a return accrues iff both its endpoints are inside the corridor."""
    n = min(len(pv), len(pc))
    out = np.full(n, np.nan)
    valid = [j for j in range(n) if not np.isnan(pv[j]) and not np.isnan(pc[j])]
    for i in range(n):
        if np.isnan(pv[i]) or np.isnan(pc[i]):
            continue
        upto = [j for j in valid if j <= i]
        if len(upto) < N_EXP + 1:
            continue
        idx = upto[-(N_EXP + 1):]
        wv, wc = pv[idx], pc[idx]
        lo, hi = wc[0] * DBAR, wc[0] * UBAR
        s, m = 0.0, 0
        for j in range(1, len(wv)):
            if lo <= wc[j] <= hi and lo <= wc[j - 1] <= hi:
                s += np.log(wv[j] / wv[j - 1]) ** 2
                m += 1
        if m > 0:
            capped = min(252.0 / m * s, (K * CAP) ** 2)
            out[i] = ((capped - K ** 2) * m / N_EXP) / (2.0 * K)
        else:
            out[i] = 0.0
    return out


def test_volswap_kernel_matches_naive_reference():
    px = _gapped(1)
    got = _rolling_pnl_volswap(px, K, N_EXP, CAP)
    want = _naive_volswap(px)
    assert np.array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_allclose(got[~np.isnan(got)], want[~np.isnan(want)], rtol=1e-12)


def test_corridor_kernel_matches_naive_reference():
    pv, pc = _gapped(2), _gapped(3, gaps=((40, 42),))
    got = _rolling_pnl_corridor(pv, pc, K, UBAR, DBAR, N_EXP, CAP)
    want = _naive_corridor(pv, pc)
    assert np.array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_allclose(got[~np.isnan(got)], want[~np.isnan(want)], rtol=1e-12)


def test_cross_leg_is_mono_minus_cross_composition():
    """The cross-corridor leg builder must be exactly: corridor kernel on the
    stock (mono) and on (index var, stock corridor) (cross) — nothing more."""
    idx_px, stk_px = _gapped(4, gaps=()), _gapped(5, gaps=())
    leg = DispersionLeg(variance_asset="I", corridor_condition_asset="S",
                        strike_mono_var_swap=K, strike_cross_corridor=K)
    cfg = _to_swap_config(DispersionConfig(cross_corridor=True, n_exp=N_EXP,
                                           barrier_up=UBAR, barrier_down=DBAR))
    mono, cross = _leg_pnl_cross_corridor(idx_px, stk_px, leg, cfg)
    np.testing.assert_allclose(
        mono, _rolling_pnl_corridor(stk_px, stk_px, K, UBAR, DBAR, N_EXP, CAP), rtol=1e-12)
    np.testing.assert_allclose(
        cross, _rolling_pnl_corridor(idx_px, stk_px, K, UBAR, DBAR, N_EXP, CAP), rtol=1e-12)


def test_volswap_first_return_skip_is_pinned():
    """Legacy quirk: the window's first daily return is EXCLUDED. If someone
    'fixes' sq_logs[1:] silently, this fails."""
    px = _gapped(6, gaps=())
    k_big = 0.30                       # cap = 0.75 — never binds here, so the
    got = _rolling_pnl_volswap(px, k_big, N_EXP, CAP)   # skip is observable
    i = np.where(~np.isnan(got))[0][0]
    w = px[i - N_EXP: i + 1]
    rets2 = np.diff(np.log(w)) ** 2
    with_skip = np.sqrt(rets2[1:].sum() * 252.0 / N_EXP) - k_big
    without_skip = np.sqrt(rets2.sum() * 252.0 / N_EXP) - k_big
    assert np.isclose(got[i], with_skip, rtol=1e-10)
    assert not np.isclose(got[i], without_skip, rtol=1e-6)
