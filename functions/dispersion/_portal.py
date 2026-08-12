# scaffolding repo de test — NE PAS recopier dans le vrai repo.
# Stubs des noms importés par functions.dispersion.__init__ / la page.
_MSG = ("test-repo scaffolding: functions.dispersion._portal requires the real "
        "repo (portal access is unavailable offline).")

def _stub(*_a, **_k):
    raise RuntimeError(_MSG)

ensure_portal = refresh_token = reset_portal = _stub
get_calendar = get_currency_calendar = _stub
payment_dates = observation_schedule = _stub
load_instrument = preload_instruments = clear_instrument_cache = _stub
