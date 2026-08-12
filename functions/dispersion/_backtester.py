"""_backtester.py — data loading, swap P&L kernels and basket backtesting.

═══════════════════════════════════════════════════════════════════════════════
SCAFFOLDING REPO DE TEST — NE PAS recopier dans le vrai repo.
Le vrai repo possède son propre _backtester.py (Bloomberg/xbbg + kernels numba).
Ce fichier fournit uniquement les noms importés par _optimizer.py/_api.py pour
que les modules s'importent et que les tests offline tournent :

    DispersionDataLoader, DispersionBacktester, SwapCalculator,
    _rolling_pnl_volswap, _rolling_pnl_corridor

- Les kernels rolling sont des implémentations numpy plausibles (fenêtre
  n_exp, annualisation 252, cap local) pour des expériences synthétiques.
- Tout accès données (DispersionDataLoader.load) ou backtest réel
  (DispersionBacktester.run / run_from_optimization) lève une RuntimeError
  claire : pas de Bloomberg ici, pas de fallback silencieux.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from functions.dispersion.models import BacktestResult, SwapConfig

__all__ = [
    "DispersionDataLoader",
    "DispersionBacktester",
    "SwapCalculator",
    "_rolling_pnl_volswap",
    "_rolling_pnl_corridor",
]

_SCAFFOLD_MSG = (
    "test-repo scaffolding: functions/dispersion/_backtester.py is a stub. "
    "Bloomberg/xbbg data access and the production backtester are only "
    "available in the real repo. Offline paths: drive DispersionOptimizer "
    "directly with a synthetic pnl_matrix, or replay a saved run bundle "
    "(functions.dispersion.run_bundle)."
)


def _rolling_pnl_volswap(prices: np.ndarray, strike: float, n_exp: int,
                         local_cap: float) -> np.ndarray:
    """Rolling vol-swap payoff (decimal): min(realized_vol, cap·K) − K.

    Realized vol at row t = annualised std of log-returns over the trailing
    ``n_exp`` observations ending at t.  Rows before warm-up are NaN.
    """
    p = np.asarray(prices, dtype=np.float64)
    out = np.full(p.shape[0], np.nan)
    if p.shape[0] < n_exp + 1:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.diff(np.log(p))
    r2 = r * r
    csum = np.concatenate([[0.0], np.nancumsum(r2)])
    cap = local_cap * strike
    for t in range(n_exp, p.shape[0]):
        window_sum = csum[t] - csum[t - n_exp]
        rv = np.sqrt(252.0 * window_sum / n_exp)
        out[t] = min(rv, cap) - strike
    return out


def _rolling_pnl_corridor(var_prices: np.ndarray, corridor_prices: np.ndarray,
                          strike: float, barrier_up: float, barrier_down: float,
                          n_exp: int, local_cap: float) -> np.ndarray:
    """Rolling corridor-variance payoff (decimal): capped corridor vol − K.

    A day contributes its squared log-return of ``var_prices`` only when the
    T−1 corridor price lies within [barrier_down, barrier_up] × the corridor
    price at the window start.  Annualisation divides by ``n_exp`` (expected
    observation count), the corridor convention.
    """
    pv = np.asarray(var_prices, dtype=np.float64)
    pc = np.asarray(corridor_prices, dtype=np.float64)
    n = min(pv.shape[0], pc.shape[0])
    out = np.full(pv.shape[0], np.nan)
    if n < n_exp + 1:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        rv = np.diff(np.log(pv[:n]))
    rv2 = np.concatenate([[0.0], rv * rv])  # aligned: rv2[u] = return into day u
    cap_var = (local_cap * strike) ** 2
    for t in range(n_exp, n):
        s = t - n_exp
        ref = pc[s]
        if not np.isfinite(ref) or ref <= 0:
            continue
        lo, hi = barrier_down * ref, barrier_up * ref
        acc = 0.0
        for u in range(s + 1, t + 1):
            cond = pc[u - 1]  # T/T-1 corridor condition
            if np.isfinite(cond) and lo <= cond <= hi and np.isfinite(rv2[u]):
                acc += rv2[u]
        realized_var = min(252.0 * acc / n_exp, cap_var)
        out[t] = np.sqrt(realized_var) - strike
    return out


class SwapCalculator:
    """Dispatch P&L kernel by product type (scaffold)."""

    def __init__(self, config: SwapConfig) -> None:
        self.config = config

    def compute(self, prices: np.ndarray, strike: float,
                corridor_prices: Optional[np.ndarray] = None) -> np.ndarray:
        cfg = self.config
        if cfg.is_vol_swap:
            return _rolling_pnl_volswap(prices, strike, cfg.n_exp, cfg.local_cap)
        corr = corridor_prices if corridor_prices is not None else prices
        return _rolling_pnl_corridor(
            prices, corr, strike, cfg.barrier_up, cfg.barrier_down,
            cfg.n_exp, cfg.local_cap)


class DispersionDataLoader:
    """Bloomberg price loader — unavailable in the test repo (scaffold)."""

    def __init__(self, config: SwapConfig) -> None:
        self.config = config

    def load(self, basket) -> Dict:
        raise RuntimeError(_SCAFFOLD_MSG)


class DispersionBacktester:
    """Basket backtester — construction only in the test repo (scaffold).

    DispersionOptimizer.__init__ instantiates this class, so the constructor
    must succeed; any actual backtest run raises with a clear message.
    """

    def __init__(self, config: SwapConfig) -> None:
        self.config = config

    def run(self, price_data: pd.DataFrame, legs, weights,
            index_data: Optional[pd.DataFrame] = None,
            start_date=None) -> BacktestResult:
        raise RuntimeError(_SCAFFOLD_MSG)

    def run_from_optimization(self, price_data: pd.DataFrame, long_basket,
                              short_basket, legs,
                              index_data: Optional[pd.DataFrame] = None,
                              start_date=None) -> BacktestResult:
        raise RuntimeError(_SCAFFOLD_MSG)
