"""
Vol Swap Solver
===============
Extracted from solver.py — contains all vol swap pricing and solving logic.
Used by PricingEngine for vol swap operations.
"""
from __future__ import annotations

import time
import math
import re
import threading
import numpy as np
import pandas as pd
import datetime
from typing import List, Optional, Dict, Any, Callable, Tuple
from dataclasses import dataclass
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed

from functions.dispersion._portal import (
    dbg, portal as _portal_fn, snap as _snap_fn,
    ensure_portal as _ensure_portal_impl,
    get_calendar, payment_dates as _payment_dates_impl,
    load_instrument as _load_instrument_impl,
)
from fpf_builder_utils.calendar import (
    get_trading_calendar as _get_trading_calendar_raw,
    offset_date, create_schedule,
)

# ─── Constants ────────────────────────────────────────────────────────────────

_VOLSWAP_STRIKE_WINDOWS = [
    (0.05, 0.30, 0.0005),
    (0.30, 0.55, 0.0005),
    (0.55, 0.85, 0.0005),
    (0.85, 1.20, 0.001),   # high-vol window (coarser step to limit scenarios)
    (1.20, 1.80, 0.002),   # very-high-vol window (only scanned when no root below 120%)
]

# ─── Ticker normalization ────────────────────────────────────────────────────
# The pricing portal resolves RIC identifiers; BBG names pasted from a terminal
# (e.g. "NVDA UW") must be converted first — otherwise instrument loading fails
# and the solver misleadingly reports "no root found".
_VS_RIC_RE = re.compile(r"^[A-Z0-9]{1,12}\.[A-Z]{1,4}$")
_VS_RIC_CACHE: Dict[str, str] = {}


def _vs_to_ric(name) -> str:
    """Normalize to RIC form; RICs and unrecognized names pass through."""
    name = str(name).strip()
    if not name or name.startswith(".") or _VS_RIC_RE.match(name):
        return name
    if name not in _VS_RIC_CACHE:
        try:
            from functions.common.tickers import bbg_to_ric
            _VS_RIC_CACHE[name] = bbg_to_ric(name) or name
        except Exception:
            _VS_RIC_CACHE[name] = name
    return _VS_RIC_CACHE[name]

_VOLSWAP_BASE_FPF_TEMPLATE = """
corridorCovarianceSwap_v4 (19-Jun-2026;19-Jun-2026, ([("DLTR.O", 1, 0.445, 0.05, 0.6675, NA, 0)], GeometricBasket, 0, NA, NA, 0, 1, 1, SumSquares, 1, False, 0.1, -0.1), 252, False, Forward, 11-Jun-2025, [(11-Jun-2025, False, 11-Jun-2025;11-Jun-2025)], (FilterOff, False, GeometricBasket, [(".SPX", 1, 0)], 1, -Infinity, Up, 0, Infinity, Down, 0), (False, [(".SPX", 1, 0)], GeometricBasket, CurrentDate, 1, Infinity, Up, Spread, 0), [(0, 19-Jun-2026, 19-Jun-2026;19-Jun-2026)])"""

_volswap_base_fpf_cache = None


def _get_volswap_base_fpf():
    """Parse vol swap base FPF template once and cache."""
    global _volswap_base_fpf_cache
    if _volswap_base_fpf_cache is None:
        from speq.fpf.unified_economics_schema.fpf_schema import (
            FPFUnifiedEconomicsWrapper, corridorCovarianceSwap_v4,
        )
        _volswap_base_fpf_cache = FPFUnifiedEconomicsWrapper.from_data(
            _VOLSWAP_BASE_FPF_TEMPLATE, script_cls=corridorCovarianceSwap_v4
        )
    return _volswap_base_fpf_cache


def calculate_payment_dates(obs_date, ric_or_currency):
    """Calculate T+2 payment dates."""
    return _payment_dates_impl(obs_date, ric_or_currency)


# ─── FPF Generation ──────────────────────────────────────────────────────────

def generate_fpf_vol(
    tickers: List[str], last_obs_date: date, strike_date: date,
    strikes: List[float], weights: List[float], is_note: bool,
    use_parameters: bool = False, observation_dates: Optional[List[str]] = None,
) -> str:
    """Generate FPF string for a vol swap.
    When *observation_dates* is None the schedule is derived internally.
    When a list of ISO-format date strings is supplied, FPF uses those directly.
    """
    tickers = [_vs_to_ric(t) for t in tickers]
    matu_ex, matu_stl = calculate_payment_dates(last_obs_date, tickers[0])
    if observation_dates is None:
        engine = _make_engine(strike_date, last_obs_date)
        return engine._build_volswap_fpf(
            tickers=tickers, strikes=strikes, weights=weights,
            strike_date=strike_date, matu_ex=matu_ex, matu_stl=matu_stl,
            is_note=is_note, use_parameters=use_parameters,
        )
    # Path 2: explicit observation dates
    from fpflucid_gen.economics import Just, PaymentDate
    from speq.fpf.unified_economics_schema.fpf_schema import corridorCovarianceSwap_v4
    base = _get_volswap_base_fpf()
    ticker, strike, weight = tickers[0], strikes[0], weights[0]
    payment = PaymentDate(ex=matu_ex, stl=matu_stl)
    corr_assets = [corridorCovarianceSwap_v4.CorridorAssets(
        corridorAsset=ticker, corridorMultiplier=1.0, corridorAssetLag=0)]
    corr_def = base.corridorDefinition.value.clone(corridorAssets=corr_assets)
    global_cap = Just(0.10) if is_note else "Nothing"
    global_floor = Just(-0.10) if is_note else "Nothing"
    template = base.varianceDetails.varianceAssetsAndIndexLegDetails[0]
    var_assets = [template.clone(
        asset=ticker, basketMultiplier=1, strike=strike,
        legCap=Just(1.5 * strike), legFloor="Nothing", legMultiplier=weight)]
    variance = base.varianceDetails.clone(
        varianceAssetsAndIndexLegDetails=var_assets,
        globalCap=global_cap, globalFloor=global_floor, isOptionOnVariance=False)
    ko = base.koDetails.clone(koAssets=[
        corridorCovarianceSwap_v4.KoAssets(koAsset=ticker, koAssetMultiplier=1.0, koAssetLag=0)])
    obs_fpf = [
        corridorCovarianceSwap_v4.ObservationDates(
            observationDates=date.fromisoformat(d), isKoDate=False,
            koPaymentDate=PaymentDate(
                ex=date.fromisoformat(offset_date(d, "2B", "LNS", "MF")),
                stl=date.fromisoformat(offset_date(d, "2B", "LNS", "MF"))))
        for d in observation_dates]
    coupons = [corridorCovarianceSwap_v4.StreamOfCoupons(
        amount=0.0, observationDate=matu_ex, paymentDate=PaymentDate(ex=matu_ex, stl=matu_ex))]
    fpf = base.clone(
        paymentDate=payment, strikeDate=strike_date,
        corridorDefinition=Just(corr_def), observationDates=obs_fpf,
        varianceDetails=variance, koDetails=ko, streamOfCoupons=coupons)
    return fpf.to_fpf_string()


generate_fpf_vol_with_dates = generate_fpf_vol


# ─── Vol Swap Engine Mixin ────────────────────────────────────────────────────
# These methods are mixed into PricingEngine via inheritance or direct use.

class VolSwapMixin:
    """
    Vol swap solving/pricing methods. Mixed into PricingEngine.
    Requires: self.config (PricingConfig), self.cache (_PortalCache)
    """

    def solve_volswap(
        self,
        ticker: str,
        currency: Optional[str] = None,
        target_value: float = 0.0,
        return_ladder: bool = False,
    ):
        """
        Solve vol swap strike for a single ticker.
        Returns:
            strike (float) or None; or (strike, ladder_data) if return_ladder
        """
        import datetime as _dt
        ticker = _vs_to_ric(ticker)
        ccy = currency or self.cache.get_currency(ticker)
        cfg = self.config
        dbg.step("volswap-solve", f"{ticker} target={target_value*100:.4f}%")
        fail_reasons = []
        try:
            portal = self.cache.portal
            snap = self.cache.snap
            matu_ex, matu_stl = self._volswap_payment_dates(cfg.last_obs_date, ticker)
            for window_idx, (lo, hi, step) in enumerate(_VOLSWAP_STRIKE_WINDOWS):
                dbg.step("volswap-solve", f"window {window_idx+1}: {lo*100:.0f}%-{hi*100:.0f}%")
                raw_strikes = np.arange(lo, hi, step)
                n = len(raw_strikes)
                # Ladder parameters: strike + cap (1.5x)
                ladder_params = portal.create_ladder_parameters(
                    parameter_names=["VOL_STRIKE1", "CAP1"],
                    parameter_sets={
                        f"scenario{i+1:03d}": [raw_strikes[i], 1.5 * raw_strikes[i]]
                        for i in range(n)
                    },
                )
                # Parameterized FPF
                fpf = self._build_volswap_fpf(
                    tickers=[ticker],
                    strikes=[0.5],  # placeholder
                    weights=[1.0],
                    strike_date=cfg.strike_date,
                    matu_ex=matu_ex,
                    matu_stl=matu_stl,
                    is_note=False,
                    use_parameters=True,
                )
                # Load & price
                underlying = self.cache.get_instrument(ticker)
                if underlying is None:
                    self.cache.preload_instruments([ticker])
                    underlying = self.cache.get_instrument(ticker)
                    if underlying is None:
                        dbg.err("volswap-solve", f"{ticker}: instrument not found")
                        fail_reasons.append("instrument load failed (ticker not resolved)")
                        continue
                nova_fpf = portal.create_fpf(
                    fpf_string=fpf,
                    instrument_ccy=ccy,
                    underlyings=[underlying],
                    premium_date=_dt.datetime.now().date(),
                    ladder_parameters=ladder_params,
                )
                model_ctx = portal.create_model_context(
                    cfg.model_name or "EMEA-Stocks-MC-LV-MultiAsset",
                    instrument_model_parameters={
                        "ACEqEqSpread": str(cfg.eqeq_lambda),
                        "ACEqFxShift": str(cfg.eqfx_shift),
                    },
                )
                # Unique price_id per ticker+window to avoid collisions in multithreaded batch
                price_id = f"VS_{ticker}_{window_idx}"
                res = portal.price(
                    price_id=price_id,
                    instruments=[nova_fpf],
                    valuation_date=_dt.datetime.now(),
                    calculation_parameters={},
                    model_context=model_ctx,
                    overridden_snap_name=snap["name"],
                    metrics=["FairValue"],
                )
                # Extract fair values
                try:
                    fair_values = [
                        res["results"][price_id]["FairValue"][i]["value"] for i in range(n)
                    ]
                except (KeyError, IndexError, TypeError) as e:
                    dbg.err("volswap-solve", f"extraction failed: {e}")
                    fail_reasons.append(f"w{window_idx+1}: unreadable response ({e})")
                    continue
                # Find sign change (linear interpolation)
                strikes_list = raw_strikes.tolist()
                for i in range(n - 1):
                    if fair_values[i] is None or fair_values[i + 1] is None:
                        continue
                    d1 = fair_values[i] - target_value
                    d2 = fair_values[i + 1] - target_value
                    if d1 * d2 < 0:
                        x1, x2 = strikes_list[i], strikes_list[i + 1]
                        root = x1 - d1 * (x2 - x1) / (d2 - d1)
                        dbg.ok("volswap-solve", f"strike={root*100:.2f}%")
                        if return_ladder:
                            return root, {
                                "strikes": strikes_list,
                                "fair_values": fair_values,
                                "window_idx": window_idx,
                                "fpf_string": fpf,
                                "target_value": target_value,
                                "solved_strike": root,
                            }
                        return root
                _fvs = [v for v in fair_values if v is not None]
                if _fvs:
                    fail_reasons.append(
                        f"[{lo*100:.0f}-{hi*100:.0f}%] FV {min(_fvs)*100:+.3f}%..{max(_fvs)*100:+.3f}% "
                        f"vs target {target_value*100:+.3f}%")
                else:
                    fail_reasons.append(f"[{lo*100:.0f}-{hi*100:.0f}%] no fair values returned")
                dbg.warn("volswap-solve", f"no root in window {window_idx+1}")
            _reason = " | ".join(fail_reasons) or "no strike window brackets the target"
            if not hasattr(self, "_volswap_errors"):
                self._volswap_errors = {}
            self._volswap_errors[ticker] = _reason
            dbg.err("volswap-solve", f"{ticker}: no root found — {_reason}")
            return (None, None) if return_ladder else None
        except Exception as e:
            if not hasattr(self, "_volswap_errors"):
                self._volswap_errors = {}
            self._volswap_errors[ticker] = str(e)
            dbg.err("volswap-solve", f"{ticker}: {e}")
            return (None, None) if return_ladder else None

    def solve_volswap_batch(
        self,
        tickers: List[str],
        currency: Optional[str] = None,
        target_value: float = 0.0,
        progress_callback: Optional[Callable] = None,
        max_workers: int = 25,
    ) -> pd.DataFrame:
        """
        Batch-solve vol swap strikes for multiple tickers (parallel).
        Returns:
            DataFrame with Ticker, Strike (%), Target FV (%), Status, FPF
        """
        import datetime as _dt
        ccy = currency or "EUR"
        tickers = [_vs_to_ric(t) for t in tickers]
        total = len(tickers)
        results: Dict[int, dict] = {}
        completed = [0]
        dbg.step("volswap-batch", f"{total} tickers, target={target_value*100:.4f}%, workers={max_workers}")
        # ── PRE-WARM: batch instrument loading + currency fetch BEFORE threading ──
        t_prewarm = time.time()
        self.cache.preload_instruments(tickers)
        # Batch currency fetch (1 HTTP call for all unknowns)
        unknown_ccy = [t for t in tickers if t not in self.cache._currencies]
        if unknown_ccy:
            try:
                info = self.cache.portal.get_underlying_information(
                    underlying_identifier_type="ricCode",
                    underlying_identifiers=unknown_ccy
                )
                info_map = info.get("information", {})
                for t in unknown_ccy:
                    if t in info_map and info_map[t].get("currency"):
                        self.cache.set_currency(t, info_map[t]["currency"])
            except Exception:
                pass
        dbg.ok("volswap-batch", f"pre-warm: {len(tickers)} instruments + currencies in {time.time()-t_prewarm:.1f}s")

        def _solve_one(ticker: str, idx: int):
            try:
                strike = self.solve_volswap(ticker=ticker, currency=ccy, target_value=target_value)
                if strike is None:
                    return idx, {
                        "Ticker": ticker, "Strike (%)": "FAILED",
                        "Target FV (%)": f"{target_value*100:.2f}%",
                        "Status": f"Failed - {getattr(self, '_volswap_errors', {}).get(ticker, 'no root found')[:180]}",
                        "FPF": "N/A",
                    }
                # Generate concrete FPF with solved strike
                matu_ex, matu_stl = self._volswap_payment_dates(self.config.last_obs_date, ticker)
                fpf_string = self._build_volswap_fpf(
                    tickers=[ticker], strikes=[strike], weights=[1.0],
                    strike_date=self.config.strike_date,
                    matu_ex=matu_ex, matu_stl=matu_stl,
                    is_note=False, use_parameters=False,
                )
                return idx, {
                    "Ticker": ticker, "Strike (%)": f"{strike*100:.2f}%",
                    "Target FV (%)": f"{target_value*100:.2f}%",
                    "Status": "Success", "FPF": fpf_string,
                }
            except Exception as e:
                dbg.err("volswap-batch", f"{ticker}: {e}")
                return idx, {
                    "Ticker": ticker, "Strike (%)": "ERROR",
                    "Target FV (%)": f"{target_value*100:.2f}%",
                    "Status": f"Error: {e}", "FPF": "N/A",
                }

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_solve_one, t, i): (t, i) for i, t in enumerate(tickers)}
            for future in as_completed(futures):
                ticker_name, _ = futures[future]
                idx, result = future.result()
                results[idx] = result
                completed[0] += 1
                if progress_callback:
                    progress_callback({"ticker": ticker_name, "completed": completed[0], "total": total})
        ordered = [results[i] for i in range(total)]
        dbg.ok("volswap-batch", f"{sum(1 for r in ordered if r['Status']=='Success')}/{total} success")
        return pd.DataFrame(ordered)

    def price_volswap(
        self,
        tickers: List[str],
        strikes: List[float],
        weights: List[float],
        is_note: bool = False,
        currency: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Price a vol swap basket with given strikes.
        Returns:
            Dict with: success, fair_value, error, fpf_string, tickers, strikes, weights, currency, is_note
        """
        import datetime as _dt
        ccy = currency or "EUR"
        tickers = [_vs_to_ric(t) for t in tickers]
        cfg = self.config
        portal = self.cache.portal
        snap = self.cache.snap
        matu_ex, matu_stl = self._volswap_payment_dates(cfg.last_obs_date, tickers[0])
        # Generate FPF
        fpf = self._build_volswap_fpf(
            tickers=tickers, strikes=strikes, weights=weights,
            strike_date=cfg.strike_date, matu_ex=matu_ex, matu_stl=matu_stl,
            is_note=is_note, use_parameters=False,
        )
        # Load underlyings
        self.cache.preload_instruments(tickers)
        underlyings = [self.cache.get_instrument(t) for t in tickers]
        # Create and price
        nova_fpf = portal.create_fpf(
            fpf_string=fpf, instrument_ccy=ccy,
            underlyings=underlyings, premium_date=_dt.datetime.now().date(),
        )
        model_ctx = portal.create_model_context(
            cfg.model_name or "EMEA-Stocks-MC-LV-MultiAsset",
            instrument_model_parameters={
                "ACEqEqSpread": str(cfg.eqeq_lambda),
                "ACEqFxShift": str(cfg.eqfx_shift),
            },
        )
        res = portal.price(
            price_id="Price", instruments=[nova_fpf],
            valuation_date=_dt.datetime.now(), calculation_parameters={},
            model_context=model_ctx, overridden_snap_name=snap["name"],
            metrics=["FairValue"],
        )
        fair_value = res["results"]["Price"]["FairValue"][0]["value"]
        error_msg = None
        try:
            error_msg = res["rawdata"]["result"]["result"]["instrumentValuationResults"][0]["resultState"]["errorMessage"]
        except (KeyError, IndexError, TypeError):
            pass
        return {
            "success": error_msg is None, "fair_value": fair_value,
            "error": error_msg, "fpf_string": fpf,
            "tickers": tickers, "strikes": strikes, "weights": weights,
            "currency": ccy, "is_note": is_note,
        }

    # ─── Vol Swap Internals ──────────────────────────────────────────────

    def _volswap_payment_dates(self, last_obs_date: date, ticker: str):
        """Calculate payment dates for vol swap (T+2 from last obs)."""
        if not ticker:
            raise ValueError(f"_volswap_payment_dates: ticker is empty/None (last_obs={last_obs_date})")
        return calculate_payment_dates(last_obs_date, ticker)

    def _volswap_common_calendar(self, tickers: List[str]) -> List[str]:
        """Compute common trading calendar across all tickers (cached per ticker combo)."""
        cfg = self.config
        # Cache key = sorted tickers + date range
        cache_key = "|".join(sorted(tickers)) + f"|{cfg.strike_date}|{cfg.last_obs_date}"
        with self.cache._lock:
            if cache_key in self.cache._calendars:
                return self.cache._calendars[cache_key]
        all_schedules = []
        for ticker in tickers:
            cal = get_calendar(ticker)
            sched = create_schedule(
                cfg.strike_date.strftime("%Y-%m-%d"),
                cfg.last_obs_date.strftime("%Y-%m-%d"),
                "1D", cal, "MF",
            )
            all_schedules.append(set(sched))
        if not all_schedules:
            return []
        result = sorted(set.intersection(*all_schedules))
        with self.cache._lock:
            self.cache._calendars[cache_key] = result
        return result

    def _build_volswap_fpf(
        self,
        tickers: List[str],
        strikes: List[float],
        weights: List[float],
        strike_date: date,
        matu_ex: date,
        matu_stl: date,
        is_note: bool = False,
        use_parameters: bool = False,
        is_capped: bool = True,
    ) -> str:
        """Build FPF string for vol swap using common calendar."""
        from fpflucid_gen.economics import Just, PaymentDate
        from speq.fpf.unified_economics_schema.fpf_schema import (
            FPFUnifiedEconomicsWrapper, corridorCovarianceSwap_v4,
        )
        obs_dates = self._volswap_common_calendar(tickers)
        # Parse base template (cached at module level)
        base = _get_volswap_base_fpf()
        payment = PaymentDate(ex=matu_ex, stl=matu_stl)
        # Corridor definition
        corr_assets = [
            corridorCovarianceSwap_v4.CorridorAssets(
                corridorAsset=tickers[0], corridorMultiplier=1.0, corridorAssetLag=0,
            )
        ]
        corr_def = base.corridorDefinition.value.clone(corridorAssets=corr_assets)
        # Global caps
        global_cap = Just(0.10) if is_note else "Nothing"
        global_floor = Just(-0.10) if is_note else "Nothing"
        # Variance assets
        template = base.varianceDetails.varianceAssetsAndIndexLegDetails[0]
        if use_parameters:
            var_assets = [
                template.clone(
                    asset=t, basketMultiplier=1, strike="$VOL_STRIKE1",
                    legCap=Just("$CAP1"), legFloor="Nothing", legMultiplier=weights[i],
                ) for i, t in enumerate(tickers)
            ]
        else:
            var_assets = [
                template.clone(
                    asset=t, basketMultiplier=1, strike=strikes[i],
                    legCap=Just(1.5 * strikes[i]) if is_capped else "Nothing",
                    legFloor="Nothing", legMultiplier=weights[i],
                ) for i, t in enumerate(tickers)
            ]
        variance = base.varianceDetails.clone(
            varianceAssetsAndIndexLegDetails=var_assets,
            globalCap=global_cap, globalFloor=global_floor,
            isOptionOnVariance=False,
        )
        # KO details
        ko = base.koDetails.clone(
            koAssets=[
                corridorCovarianceSwap_v4.KoAssets(koAsset=t, koAssetMultiplier=1.0, koAssetLag=0)
                for t in tickers
            ]
        )
        # Observation dates
        obs_fpf = [
            corridorCovarianceSwap_v4.ObservationDates(
                observationDates=date.fromisoformat(d),
                isKoDate=False,
                koPaymentDate=PaymentDate(
                    ex=date.fromisoformat(offset_date(d, "2B", "LNS", "MF")),
                    stl=date.fromisoformat(offset_date(d, "2B", "LNS", "MF")),
                ),
            ) for d in obs_dates
        ]
        # Stream of coupons
        coupons = [
            corridorCovarianceSwap_v4.StreamOfCoupons(
                amount=0.0, observationDate=matu_ex,
                                paymentDate=PaymentDate(ex=matu_ex, stl=matu_ex),
            )
        ]
        # Assemble
        fpf = base.clone(
            paymentDate=payment, strikeDate=strike_date,
            corridorDefinition=Just(corr_def),
            observationDates=obs_fpf, varianceDetails=variance,
            koDetails=ko, streamOfCoupons=coupons,
        )
        return fpf.to_fpf_string()


# ─── Module-level convenience functions ───────────────────────────────────────

def _make_engine(strike_date: date, last_obs_date: date, eqeq_lambda: float = 0.1, eqfx_shift: float = -0.05,
                 model_name: str = None):
    """Create a PricingEngine with vol-swap-relevant config."""
    # Import here to avoid circular dependency
    from functions.dispersion._pricing import PricingEngine, PricingConfig
    return PricingEngine(PricingConfig(
        strike_date=strike_date, last_obs_date=last_obs_date,
        eqeq_lambda=eqeq_lambda, eqfx_shift=eqfx_shift,
        model_name=model_name,
    ))


def create_and_price_fpf(tickers, maturity_date, strike_date, strikes, weights, is_note, currency,
                         eqeq_lambda=0.1, eqfx_shift=-0.05, model_name=None):
    """Generate and price a vol swap FPF."""
    return _make_engine(strike_date, maturity_date, eqeq_lambda, eqfx_shift, model_name).price_volswap(
        tickers=tickers, strikes=strikes, weights=weights, is_note=is_note, currency=currency)


def solve_volswap_strike_single(ticker, maturity_date, strike_date, currency,
                                target_value=0.0, eqeq_lambda=0.1, eqfx_shift=-0.05, return_ladder=False):
    """Solve for vol swap strike that prices to target fair value."""
    return _make_engine(strike_date, maturity_date, eqeq_lambda, eqfx_shift).solve_volswap(
        ticker=ticker, currency=currency, target_value=target_value, return_ladder=return_ladder)


def solve_volswap_strikes_multithreaded(tickers, maturity_date, strike_date, currency,
                                        target_value=0.0, eqeq_lambda=0.1, eqfx_shift=-0.05,
                                        progress_callback=None, max_workers=25, model_name=None):
    """Batch-solve vol swap strikes for multiple tickers (parallel)."""
    return _make_engine(strike_date, maturity_date, eqeq_lambda, eqfx_shift, model_name).solve_volswap_batch(
        tickers=tickers, currency=currency, target_value=target_value,
        progress_callback=progress_callback, max_workers=max_workers)
