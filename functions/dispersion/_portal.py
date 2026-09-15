# scaffolding repo de test — NE PAS recopier dans le vrai repo.
# Stubs des noms importés par functions.dispersion.__init__ / _pricing / _volswap / la page.
# Tout appel portail lève RuntimeError ; le module _pricing devient importable
# hors ligne (tests des dataclasses, du DataFrame de résultats, des scénarios).
_MSG = ("test-repo scaffolding: functions.dispersion._portal requires the real "
        "repo (portal access is unavailable offline).")


def _stub(*_a, **_k):
    raise RuntimeError(_MSG)


class _Dbg:
    """No-op logger with the real module's method names."""
    def _noop(self, *_a, **_k):
        pass
    ok = info = warn = err = step = note = _noop


dbg = _Dbg()


def timed(*_a, **_k):
    """Decorator stub: returns the function unchanged (supports @timed and @timed(...))."""
    if len(_a) == 1 and callable(_a[0]) and not _k:
        return _a[0]
    return lambda f: f


ensure_portal = refresh_token = reset_portal = _stub
portal = snap = _stub
get_calendar = get_calendar_from_currency = get_currency_calendar = _stub
payment_dates = observation_schedule = _stub
load_instrument = preload_instruments = clear_instrument_cache = _stub

CURRENCY_CALENDAR_MAP = {}
BANNED_RICS = []
_instrument_cache = {}
import threading as _threading
_instrument_lock = _threading.Lock()
