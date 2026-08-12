# scaffolding repo de test — NE PAS recopier dans le vrai repo.
# Stubs des noms importés par functions.dispersion.__init__ / _api.solve/price.
_MSG = ("test-repo scaffolding: functions.dispersion._pricing requires the real "
        "repo (portal pricing engine).")

def _stub(*_a, **_k):
    raise RuntimeError(_MSG)

class PricingConfig:
    def __init__(self, *a, **k):
        raise RuntimeError(_MSG)

class PricingEngine:
    def __init__(self, *a, **k):
        raise RuntimeError(_MSG)

class CrossCorridorVarianceSwap:
    def __init__(self, *a, **k):
        raise RuntimeError(_MSG)

_load_instrument_impl = get_trading_calendar = calculate_payment_dates = _stub
