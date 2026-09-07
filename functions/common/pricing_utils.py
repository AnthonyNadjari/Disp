"""
Shared pricing utilities — retry logic, model context caching, instrument loading,
correlation-sensitivity metric helpers.
===================================================================================

Used by: correlation pricing, riversource, dispersion, vol swap, WOF put.
Import: from functions.common.pricing_utils import retry_pricing, get_model_context, load_instruments
        from functions.common.pricing_utils import correlation_sens_metric, correlation_sens_pair
"""

from __future__ import annotations

import os
import time
import threading
import logging
from functools import wraps
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


def _portal_and_snap():
    """Portal singleton + live snap — imported lazily so the pure helpers of
    this module (metric builders, response parsers) import without the
    pricing-portal stack (engine-only environments, tests)."""
    from functions.common.portal import get_portal, get_snap
    return get_portal(), get_snap()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Retry decorator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_RETRIABLE_ERRORS = (
    "Call to upstream service was unsuccessful",
    "Upstream Service Error",
    'status":"500',
    "timeout",
    "TIMEOUT",
)


def retry_pricing(max_retries: int = 3, initial_delay: float = 1.0):
    """
    Decorator: retry on transient upstream pricing errors with exponential backoff.

    Usage:
        @retry_pricing(max_retries=3)
        def price_something(...):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exc = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    err_str = str(e)
                    if any(msg in err_str for msg in _RETRIABLE_ERRORS):
                        last_exc = e
                        if attempt < max_retries - 1:
                            logger.warning(f"Retry {attempt+1}/{max_retries} for {func.__name__}: {err_str[:100]}")
                            time.sleep(delay)
                            delay *= 2
                    else:
                        raise  # non-retriable → fail immediately
            raise last_exc
        return wrapper
    return decorator


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model context cache
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_model_ctx_cache: Dict[str, Any] = {}
_ctx_lock = threading.Lock()


def get_model_context(
    model_name: str = "EMEA-Stocks-MC-LV-MultiAsset",
    eqeq_lambda: float = 0.1,
    eqfx_shift: float = -0.05,
    extra_params: Optional[Dict[str, str]] = None,
) -> Any:
    """
    Get or create a cached model context. Avoids recreating identical contexts.

    Args:
        model_name: Pricing model identifier
        eqeq_lambda: EqEq spread parameter
        eqfx_shift: EqFx shift parameter
        extra_params: Additional model parameters

    Returns:
        Model context object (cached)
    """
    params = {"ACEqEqSpread": str(eqeq_lambda), "ACEqFxShift": str(eqfx_shift)}
    if extra_params:
        params.update(extra_params)

    cache_key = f"{model_name}|{'|'.join(f'{k}={v}' for k, v in sorted(params.items()))}"

    with _ctx_lock:
        if cache_key not in _model_ctx_cache:
            portal, _ = _portal_and_snap()
            _model_ctx_cache[cache_key] = portal.create_model_context(
                model_name, instrument_model_parameters=params
            )
        return _model_ctx_cache[cache_key]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Instrument loading (batch, cached)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_instruments: Dict[str, Any] = {}
_instr_lock = threading.Lock()


def load_instruments(tickers: List[str]) -> Dict[str, Any]:
    """
    Load instruments from portal (batch, cached). Returns {ticker: instrument}.

    Only fetches tickers not already in cache. Thread-safe.
    """
    from pricingportal import NovaIdSource

    with _instr_lock:
        missing = [t for t in tickers if t not in _instruments]

    if not missing:
        return {t: _instruments[t] for t in tickers}

    portal, snap = _portal_and_snap()

    # Batch load all missing at once
    loaded = portal.load_instruments(
        identifier_type=NovaIdSource.ricCode,
        identifiers=missing,
        snap_name=snap["name"],
    )

    with _instr_lock:
        for ticker, instr in zip(missing, loaded):
            if instr is not None:
                _instruments[ticker] = instr

    return {t: _instruments.get(t) for t in tickers}


def get_instrument(ticker: str) -> Any:
    """Get single cached instrument (loads if needed)."""
    result = load_instruments([ticker])
    return result.get(ticker)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Parallel pricing helper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parallel_price(
    tasks: List[Dict[str, Any]],
    price_fn,
    max_workers: int = 25,
    progress_callback=None,
) -> List[Any]:
    """
    Run pricing tasks in parallel with optional progress callback.

    Args:
        tasks: List of dicts passed as kwargs to price_fn
        price_fn: Callable(**task) -> result
        max_workers: Thread pool size
        progress_callback: Optional fn(completed, total, ticker, result)

    Returns:
        List of results in original order
    """
    total = len(tasks)
    results = [None] * total
    completed = [0]

    def _run(idx_task):
        idx, task = idx_task
        result = price_fn(**task)
        return idx, result

    with ThreadPoolExecutor(max_workers=min(total, max_workers)) as pool:
        futures = {pool.submit(_run, (i, t)): i for i, t in enumerate(tasks)}
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
            completed[0] += 1
            if progress_callback:
                ticker = tasks[idx].get("ticker", f"#{idx}")
                progress_callback(completed[0], total, ticker, result)

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Quick price helper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@retry_pricing(max_retries=3)
def price_fpf(
    fpf_string: str,
    underlyings: List[Any],
    currency: str = "EUR",
    model_name: str = "EMEA-Stocks-MC-LV-MultiAsset",
    eqeq_lambda: float = 0.1,
    eqfx_shift: float = -0.05,
    metrics: List[str] = None,
    ladder_parameters=None,
) -> Dict[str, Any]:
    """
    Price a single FPF string. Returns raw result dict from portal.

    This is the lowest-level pricing call — everything else builds on it.
    """
    import datetime as _dt

    portal, snap = _portal_and_snap()

    nova_fpf = portal.create_fpf(
        fpf_string=fpf_string,
        instrument_ccy=currency,
        underlyings=underlyings,
        premium_date=_dt.datetime.now().date(),
        ladder_parameters=ladder_parameters,
    )

    model_ctx = get_model_context(model_name, eqeq_lambda, eqfx_shift)

    return portal.price(
        price_id="Price",
        instruments=[nova_fpf],
        valuation_date=_dt.datetime.now(),
        calculation_parameters={},
        model_context=model_ctx,
        overridden_snap_name=snap["name"],
        metrics=metrics or ["FairValue"],
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Correlation sensitivity — portal metric "CorrelationSens" (zenith CorrelationDelta)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Portal definition:
#     BumpSize        double  default 0.01
#     ScalingFactor   double  default 0.01   (alias Scaling)
#     Differencing    string  default "ForwardNormalized"
#     UseAngularBump  boolean
#
# ForwardNormalized returns (FV(ρ+Bump) − FV(ρ)) / Bump × Scaling, so sending
# Bump = Scaling = b yields exactly the FairValue change for a +b correlation
# move.  The response is a full n×n matrix over the instrument's assets — one
# entry per (PrimaryAssetRef, SecondaryAssetRef) pair, diagonal 0, symmetric.
# For a two-asset instrument the informative number is the off-diagonal
# entry (``correlation_sens_pair``).
#
# Product-agnostic: nothing here knows about corridors, strikes or RA — a
# product maps the FairValue change to its own quantity (dispersion: strike
# re-solve in functions.dispersion._pricing).
#
# Environment overrides (generic names; DISP_-prefixed legacy names accepted):
#     PRICING_CORRSENS_BUMP=0.05                  the correlation move
#     PRICING_CORRSENS_PARAMS='{"UseAngularBump":"true"}'   extra/override parameters
#     PRICING_CORRSENS_DEBUG=1                    verbose per-instrument dump

CORRELATION_SENS_METRIC = "CorrelationSens"
CORRELATION_SENS_BUMP = 0.01          # default move, absolute (0.01 = one correlation point)
_CORRSENS_BUMP_NAMES = ("LV", "LSV", "LSV0", "LCM")


def _env(name: str) -> str:
    """Generic env var, falling back to the legacy DISP_-prefixed name."""
    return (os.environ.get(name) or os.environ.get("DISP_" + name) or "").strip()


def _safe_print(*args, **kwargs) -> None:
    """Print that won't crash if the console pipe is closed (Streamlit)."""
    try:
        print(*args, **kwargs)
    except OSError:
        pass


def correlation_sens_bump() -> float:
    """The correlation move the metric refers to (absolute; 0.01 = one point).
    ``PRICING_CORRSENS_BUMP`` overrides ``CORRELATION_SENS_BUMP``."""
    raw = _env("PRICING_CORRSENS_BUMP")
    if raw:
        try:
            return float(raw)
        except ValueError:
            _safe_print(f"[CORRSENS] ignoring PRICING_CORRSENS_BUMP={raw!r} (not a number)")
    return float(CORRELATION_SENS_BUMP)


def correlation_sens_debug() -> bool:
    return _env("PRICING_CORRSENS_DEBUG").lower() in ("1", "true", "yes", "on")


def correlation_sens_metric(portal, bump: Optional[float] = None,
                            extra_params: Optional[Dict[str, str]] = None):
    """Build the CorrelationSens metric for a +``bump`` correlation move
    (default ``correlation_sens_bump()``): BumpSize = ScalingFactor = bump,
    ForwardNormalized differencing.  ``extra_params`` and the
    ``PRICING_CORRSENS_PARAMS`` JSON dict add/override parameters."""
    import json
    b = float(bump) if bump is not None else correlation_sens_bump()
    params: Dict[str, str] = {
        "BumpSize": f"{b:g}",
        "ScalingFactor": f"{b:g}",
        "Differencing": "ForwardNormalized",
    }
    raw = _env("PRICING_CORRSENS_PARAMS")
    if raw:
        try:
            params.update({str(k): str(v) for k, v in json.loads(raw).items()})
        except Exception as e:  # malformed JSON must not kill pricing — say so and keep defaults
            _safe_print(f"[CORRSENS] ignoring PRICING_CORRSENS_PARAMS ({type(e).__name__}: {e})")
    if extra_params:
        params.update({str(k): str(v) for k, v in extra_params.items()})
    return portal.create_metric(
        CORRELATION_SENS_METRIC,
        [portal.create_metric_parameter(k, v) for k, v in params.items()])


def correlation_sens_entries(entry, bump_names: Tuple[str, ...] = _CORRSENS_BUMP_NAMES) -> list:
    """All CorrelationSens entries of one instrument result, wherever the
    portal put them: top level, inside a named bump (``entry['LV'][0]``), or
    inside a ``SimpleScenarioBump`` wrapper.  ``[]`` when absent."""
    if not isinstance(entry, dict):
        return []
    lst = entry.get(CORRELATION_SENS_METRIC, [])
    if isinstance(lst, list) and lst:
        return lst
    for bn in bump_names:
        bd = entry.get(bn)
        if isinstance(bd, list) and bd and isinstance(bd[0], dict):
            lst = bd[0].get(CORRELATION_SENS_METRIC, [])
            if isinstance(lst, list) and lst:
                return lst
    bumps = entry.get("SimpleScenarioBump")
    if isinstance(bumps, list) and bumps and isinstance(bumps[0], dict):
        for bn in bump_names:
            bd = bumps[0].get(bn)
            if isinstance(bd, list) and bd and isinstance(bd[0], dict):
                lst = bd[0].get(CORRELATION_SENS_METRIC, [])
                if isinstance(lst, list) and lst:
                    return lst
    return []


def correlation_sens_matrix(entries) -> Dict[Tuple[str, str], float]:
    """{(PrimaryAssetRef, SecondaryAssetRef): value} for every entry that
    carries both refs and a numeric value."""
    out: Dict[Tuple[str, str], float] = {}
    for e in entries or []:
        if isinstance(e, dict):
            p, s, v = e.get("PrimaryAssetRef"), e.get("SecondaryAssetRef"), e.get("value")
            if p is not None and s is not None and v is not None:
                try:
                    out[(str(p), str(s))] = float(v)
                except (TypeError, ValueError):
                    pass
    return out


def correlation_sens_pair(entries, primary: Optional[str] = None,
                          secondary: Optional[str] = None) -> Optional[float]:
    """The sensitivity for one asset pair.  With ``primary``/``secondary``
    given, matches that pair in either order; otherwise returns the first
    off-diagonal entry (the only informative number of a two-asset
    instrument).  ``None`` when absent (single-asset instrument, unsupported
    metric, failed chunk)."""
    mat = correlation_sens_matrix(entries)
    if primary is not None and secondary is not None:
        return mat.get((primary, secondary), mat.get((secondary, primary)))
    for (p, s), v in mat.items():
        if p != s:
            return v
    return None


def dump_correlation_sens(results_by_chunk: dict, labels: list, header: str,
                          printer=_safe_print) -> dict:
    """Print the raw CorrelationSens response per instrument and return
    ``{global_idx: entries}``.  ``results_by_chunk`` is the batched layout
    ``{chunk_start: {"raw": results_dict, "chunk_size": n}}`` with per-
    instrument keys ``Price``, ``Price_1``, ...; ``labels[global_idx]`` is a
    human label.  When the metric key is absent the entry's own keys are
    printed, so an unexpected response shape is visible, not silently empty."""
    import pprint
    out: dict = {}
    printer(f"\n{'=' * 100}\n[CORRSENS] {header}\n{'=' * 100}")
    if not results_by_chunk:
        printer("[CORRSENS] NO results — the pricing call failed or returned nothing")
        return out
    running = 0
    for chunk_start in sorted(results_by_chunk.keys()):
        chunk = results_by_chunk[chunk_start]
        raw = chunk.get("raw") or {}
        size = int(chunk.get("chunk_size", 0))
        if not raw:
            printer(f"[CORRSENS] chunk@{chunk_start}: EMPTY raw (chunk failed) — {size} instruments skipped")
        for local_idx in range(size):
            gidx = running + local_idx
            key = "Price" if local_idx == 0 else f"Price_{local_idx}"
            label = labels[gidx] if gidx < len(labels) else f"instrument {gidx}"
            entry = raw.get(key) if raw else None
            entries = correlation_sens_entries(entry)
            out[gidx] = entries
            printer(f"\n--- [{gidx:02d}] {label} | key={key} ---")
            if entries:
                for e in entries:
                    if isinstance(e, dict):
                        pair = " / ".join(str(e.get(k)) for k in ("PrimaryAssetRef", "SecondaryAssetRef")
                                          if e.get(k) is not None)
                        flags = e.get("extraResults") or {}
                        printer(f"  value={e.get('value')!r}  pair=[{pair}]  "
                                f"maturity={e.get('Maturity')}  extraResults={sorted(flags) if flags else '{}'}")
                printer("  raw:")
                printer("  " + pprint.pformat(entries, width=110).replace("\n", "\n  "))
            elif isinstance(entry, dict):
                printer(f"  no '{CORRELATION_SENS_METRIC}' key — entry keys: {sorted(entry.keys())}")
                for bn in _CORRSENS_BUMP_NAMES + ("SimpleScenarioBump",):
                    bd = entry.get(bn)
                    if isinstance(bd, list) and bd and isinstance(bd[0], dict):
                        printer(f"    {bn}[0] keys: {sorted(bd[0].keys())}")
            else:
                printer(f"  no entry for {key} (raw keys: {sorted(raw.keys()) if raw else 'none'})")
        running += size
    n_ok = sum(1 for v in out.values() if v)
    printer(f"\n[CORRSENS] {n_ok}/{len(out)} instruments returned {CORRELATION_SENS_METRIC}\n{'=' * 100}\n")
    return out
