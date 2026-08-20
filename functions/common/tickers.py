"""
Unified ticker conversion — single source for all BBG ↔ RIC operations.
Replaces scattered conversion patterns across the codebase.

Usage:
    from functions.common.tickers import bbg_to_ric, ric_to_bbg, clean_ric
    from functions.common.tickers import bbg_to_ric_batch, ric_to_bbg_batch
    from functions.common.tickers import convert_ticker, get_currency_from_ric
"""

from __future__ import annotations

import threading
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests_negotiate_sspi import HttpNegotiateAuth

try:
    from functions.paths import LOCAL
    CERT_BUNDLE = str(LOCAL.CERTS)
except Exception:
    CERT_BUNDLE = True  # fall back to system CA store outside the desk env

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

UDX_URL = 'https://atlastrade-ldn.barcapint.com/UnderlyingManagement/api/IndexManagementService/Retrieve'
AVS_URL = 'http://dirac-pp-ldn:90/avs/instrument/search/?instrumentNameExpression='
_MAPPING_SERVER = "http://nix-liveappserver:8100/UnderlyingRestService/Service/Underyling"

# ---------------------------------------------------------------------------
# Mappings
# ---------------------------------------------------------------------------

BBG_SUFFIX_MAP = {'SW': 'SE', 'SM': 'SQ'}
MANUAL_BBG_TO_RIC = {'ABBV US Equity': 'ABBV.N', 'ABBV US': 'ABBV.N'}
BBG_TO_RIC_EXCHANGE = {
    'US': ['.N', '.O', '.A'], 'LN': ['.L'], 'FP': ['.PA'],
    'GR': ['.DE'], 'SE': ['.S'], 'IT': ['.MI'], 'SM': ['.MC'],
    'NA': ['.AS'], 'BB': ['.BR'], 'DC': ['.CO'], 'SS': ['.ST'],
    'NO': ['.OL'], 'AV': ['.VI'], 'FH': ['.HE'], 'PL': ['.LS'],
    'HK': ['.HK'], 'JP': ['.T'], 'AU': ['.AX'], 'CT': ['.TO'],
}

# Reverse: RIC suffix → BBG exchange
_RIC_TO_BBG_EXCHANGE = {}
for _exch, _suffixes in BBG_TO_RIC_EXCHANGE.items():
    for _s in _suffixes:
        _RIC_TO_BBG_EXCHANGE.setdefault(_s, _exch)

# ---------------------------------------------------------------------------
# Thread-safe conversion log
# ---------------------------------------------------------------------------

conversion_log: list[dict] = []
_log_lock = threading.Lock()


def _log_result(ticker: str, method: str, result, success: bool):
    """Thread-safe logging of conversion results."""
    with _log_lock:
        conversion_log.append({
            'Ticker': ticker,
            'Method': method,
            'Result': result,
            'Success': '✅' if success else '❌'
        })


def get_conversion_log() -> list[dict]:
    """Return a copy of the conversion log."""
    with _log_lock:
        return list(conversion_log)


def clear_conversion_log():
    """Clear the conversion log."""
    with _log_lock:
        conversion_log.clear()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def clean_ric(ric: str) -> str:
    """Clean RIC code - remove 'OLD' prefix, '-...' suffix, normalize .OQ→.O"""
    if not ric or ric == '❌ Not Found':
        return ric

    cleaned = ric.strip()
    if cleaned.upper().startswith('OLD '):
        cleaned = cleaned[4:].strip()
    if cleaned.upper().startswith('OLD'):
        cleaned = cleaned[3:]
    if '-' in cleaned:
        cleaned = cleaned.split('-')[0]
    cleaned = cleaned.replace('.OQ', '.O')
    return cleaned


def normalize_bbg(ticker: str) -> str:
    """Normalize BBG suffix (SW→SE, SM→SQ)."""
    if not ticker:
        return ticker
    has_eq = ticker.endswith(' Equity')
    parts = ticker.replace(' Equity', '').strip().split()
    if len(parts) >= 2 and parts[-1] in BBG_SUFFIX_MAP:
        parts[-1] = BBG_SUFFIX_MAP[parts[-1]]
        return ' '.join(parts) + (' Equity' if has_eq else '')
    return ticker


def validate_ric(bbg_ticker: str, ric: str) -> bool:
    """Validate RIC suffix matches BBG exchange."""
    if not ric or ric == '❌ Not Found':
        return False
    parts = bbg_ticker.replace(' Equity', '').strip().split()
    if len(parts) < 2:
        return True
    exchange = parts[-1]
    if exchange in BBG_TO_RIC_EXCHANGE:
        return any(ric.endswith(s) for s in BBG_TO_RIC_EXCHANGE[exchange])
    return True


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def udx_api(term: str, field: str = 'ric') -> Optional[dict]:
    """UDX API call — search by field (ric or bbpk)."""
    try:
        resp = requests.post(UDX_URL, json={'searchString': term},
                             auth=HttpNegotiateAuth(), verify=CERT_BUNDLE, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json().get('underlyings', [])
        for rec in data:
            if field == 'bbpk':
                clean = term.replace(' Equity', '')
                if rec.get('bbpk') == term or rec.get('bete') == clean:
                    return rec
            elif rec.get(field) == term:
                return rec
        return None
    except Exception:
        return None


def avs_api(term: str, get_bbg: bool = True) -> Optional[str]:
    """AVS API call — resolve ticker via AlternateIds."""
    try:
        resp = requests.get(f"{AVS_URL}{term}", verify=CERT_BUNDLE, timeout=10)
        if resp.status_code != 200:
            return None
        for result in resp.json():
            try:
                alt_ids = result['InstrumentDefinition']['AlternateIds']
                if get_bbg:
                    for item in alt_ids:
                        if item.get('Source') == 'REUTERS' and item.get('Id') == term:
                            for bbg in alt_ids:
                                if bbg.get('Source') == 'BLOOMBERG_PRIMARY_KEY':
                                    return bbg['Id']
                else:
                    clean = term.replace(' Equity', '').strip()
                    for item in alt_ids:
                        if item.get('Source') in ['BLOOMBERG_PRIMARY_KEY', 'BLOOMBERG_EQUITY_TICKER_WITH_EXCHANGE']:
                            if clean in item.get('Id', ''):
                                for ric in alt_ids:
                                    if ric.get('Source') == 'REUTERS':
                                        return ric['Id'].replace('.OQ', '.O')
            except Exception:
                pass
        return None
    except Exception:
        return None


def _pattern_match(bbg_ticker: str) -> Optional[str]:
    """Pattern-based conversion for US tickers."""
    if 'US' not in bbg_ticker:
        return None
    base = bbg_ticker.replace(' US Equity', '').replace(' US', '').strip()
    for suffix in ['.N', '.O', '.A']:
        candidate = f"{base}{suffix}"
        if udx_api(candidate, field='ric'):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Core conversion functions
# ---------------------------------------------------------------------------

def bbg_to_ric(ticker: str) -> Optional[str]:
    """
    Convert BBG ticker to RIC — 4 strategies:
    1. Manual override
    2. UDX with validation
    3. AVS with validation
    4. Pattern-based (US tickers)

    Examples:
        bbg_to_ric("AAPL US Equity") → "AAPL.O"
        bbg_to_ric("SAN SM Equity") → "SAN.MC"
    """
    if not ticker or not isinstance(ticker, str):
        return None

    # Method 1: Manual Override
    if ticker in MANUAL_BBG_TO_RIC:
        result = clean_ric(MANUAL_BBG_TO_RIC[ticker])
        _log_result(ticker, '1-Manual', result, True)
        return result

    normalized = normalize_bbg(ticker)

    # Method 2: UDX with validation
    udx_res = udx_api(normalized, field='bbpk')
    if udx_res and udx_res.get('ric'):
        ric = clean_ric(udx_res['ric'].replace('.OQ', '.O'))
        if validate_ric(ticker, ric):
            _log_result(ticker, '2-UDX', ric, True)
            return ric
        _log_result(ticker, '2-UDX-Invalid', ric, False)

    # Method 3: AVS with validation
    avs_res = avs_api(normalized, get_bbg=False)
    if avs_res:
        avs_res = clean_ric(avs_res)
        if validate_ric(ticker, avs_res):
            _log_result(ticker, '3-AVS', avs_res, True)
            return avs_res
        _log_result(ticker, '3-AVS-Invalid', avs_res, False)

    # Method 4: Pattern-based
    pattern_res = _pattern_match(ticker)
    if pattern_res:
        pattern_res = clean_ric(pattern_res)
        _log_result(ticker, '4-Pattern', pattern_res, True)
        return pattern_res

    _log_result(ticker, 'Failed', None, False)
    return None


def ric_to_bbg(ticker: str) -> Optional[str]:
    """
    Convert RIC to BBG — UDX then AVS fallback.

    Examples:
        ric_to_bbg("AAPL.O") → "AAPL US Equity"
        ric_to_bbg("SAN.MC") → "SAN SM Equity"
    """
    if not ticker or not isinstance(ticker, str):
        return None

    # UDX lookup
    res = udx_api(ticker, field='ric')
    if res and res.get('bbpk'):
        bbg = clean_ric(res['bbpk'])
        _log_result(ticker, 'UDX', bbg, True)
        return bbg

    # AVS fallback
    bbg = avs_api(ticker, get_bbg=True)
    if bbg:
        bbg = clean_ric(bbg)
        _log_result(ticker, 'AVS', bbg, True)
        return bbg

    _log_result(ticker, 'Failed', None, False)
    return None


# ---------------------------------------------------------------------------
# Batch operations
# ---------------------------------------------------------------------------

def bbg_to_ric_batch(tickers: list[str], mapping: dict | None = None,
                     max_workers: int = 8) -> list[str]:
    """
    Convert list of BBG tickers to RICs (parallel).

    Args:
        tickers: List of Bloomberg tickers
        mapping: Optional exchange code mapping for fallback (e.g. {'SQ': 'SM'})
        max_workers: Thread pool size
    """
    results = [None] * len(tickers)

    def _convert(idx, t):
        ric = bbg_to_ric(t)
        if not ric and mapping:
            # Retry with mapped exchange code
            parts = t.strip().split()
            if len(parts) >= 2 and parts[1] in mapping:
                retry = parts[0] + " " + mapping[parts[1]]
                if t.endswith(' Equity'):
                    retry += ' Equity'
                ric = bbg_to_ric(retry)
        return idx, ric or t

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_convert, i, t): i for i, t in enumerate(tickers)}
        for f in as_completed(futures):
            idx, result = f.result()
            results[idx] = result

    return results


def ric_to_bbg_batch(tickers: list[str], max_workers: int = 8) -> list[str]:
    """Convert list of RICs to BBG tickers (parallel)."""
    results = [None] * len(tickers)

    def _convert(idx, t):
        return idx, ric_to_bbg(t) or t

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_convert, i, t): i for i, t in enumerate(tickers)}
        for f in as_completed(futures):
            idx, result = f.result()
            results[idx] = result

    return results


# ---------------------------------------------------------------------------
# Currency / calendar helpers
# ---------------------------------------------------------------------------

def get_currency_from_ric(ric: str) -> str:
    """Infer currency from RIC suffix (offline, no API call)."""
    if not ric:
        return 'EUR'

    currency_map = {
        '.O': 'USD', '.N': 'USD', '.A': 'USD',
        '.L': 'GBP', '.S': 'CHF',
        '.ST': 'SEK', '.OL': 'NOK', '.CO': 'DKK',
        '.T': 'JPY', '.HK': 'HKD', '.AX': 'AUD', '.TO': 'CAD',
    }
    for suffix, ccy in currency_map.items():
        if ric.endswith(suffix):
            return ccy
    return 'EUR'


def get_currency(ticker: str, is_ric: bool = False) -> str:
    """Get currency for a ticker using pricing portal (online)."""
    from functions.common.portal import get_portal
    try:
        ric = ticker if is_ric else bbg_to_ric(ticker)
        if not ric:
            return "Unknown"

        portal = get_portal()
        info = portal.get_underlying_information(
            underlying_identifier_type="ricCode",
            underlying_identifiers=[ric]
        )
        currency = info.get("information", {}).get(ric, {}).get("currency")
        return currency if currency else "Unknown"
    except Exception:
        return "Error"


def convert_ticker(ticker: str, to_ric: bool = True) -> tuple[str, dict]:
    """
    Full conversion with currency and calendar.

    Returns:
        (original_ticker, {'Converted': str, 'Currency': str, 'Calendar': str})
    """
    from fpf_builder_utils.calendar import get_trading_calendar

    converted = (bbg_to_ric(ticker.strip()) if to_ric else ric_to_bbg(ticker.strip())) or '❌ Not Found'

    # Get currency
    try:
        if to_ric:
            currency = get_currency(converted, is_ric=True) if converted != '❌ Not Found' else "Unknown"
        else:
            currency = get_currency(ticker.strip(), is_ric=True)
    except Exception:
        currency = "Error"

    # Get calendar
    try:
        ric_cal = converted if (to_ric and converted != '❌ Not Found') else ticker
        calendar = get_trading_calendar(ric_cal) if ric_cal != '❌ Not Found' else "Error"
    except Exception as e:
        calendar = f"Error: {str(e)}"

    return ticker, {'Converted': converted, 'Currency': currency, 'Calendar': calendar}


def get_exchange_from_ric(ric: str) -> Optional[str]:
    """Extract exchange suffix from RIC (e.g., 'AAPL.O' → '.O')."""
    if not ric or '.' not in ric:
        return None
    idx = ric.index('.')
    return ric[idx:]


# ---------------------------------------------------------------------------
# Online lookup (used by data/sectors.py, data/maturities.py)
# Alias for backward compat — same as ric_to_bbg but explicit name
# ---------------------------------------------------------------------------

ric_to_bbg_online = ric_to_bbg
