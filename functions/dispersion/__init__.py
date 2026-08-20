"""
Dispersion Trading Engine.

Public API:
    from functions.dispersion import solve, price, optimize, backtest
    from functions.dispersion import DispersionConfig, OptimizationConstraints
    from functions.dispersion import SolveResult, PriceResult, OptimizationResult, BacktestResult
"""

# Public functions
from functions.dispersion._api import solve, price, optimize, backtest

# Config classes (users construct these)
from functions.dispersion.models import (
    DispersionConfig,
    OptimizationConstraints,
    MissingDataPolicy,
    ProductType,
)

# Result types (users receive these — never construct)
from functions.dispersion.models import (
    SolveResult,
    PriceResult,
    OptimizationResult,
    BacktestResult,
)

# Internal exports (not for direct public use, but used by pages/✅Dispersion_Optimizer.py)
from functions.dispersion._optimizer import DispersionOptimizer
try:
    from functions.common.utils import get_n_exp_from_date
except Exception as _utils_import_error:  # pricing-portal stack absent (engine-only envs / tests)
    _UTILS_IMPORT_ERROR = _utils_import_error

    def get_n_exp_from_date(*args, **kwargs):
        raise RuntimeError(
            "functions.common.utils could not be imported (the pricing-portal "
            f"stack is not available in this environment): {_UTILS_IMPORT_ERROR}")
from functions.dispersion._backtester import (
    DispersionBacktester,
    DispersionDataLoader,
    SwapCalculator,
    _rolling_pnl_corridor,
)
from functions.dispersion._portal import (
    ensure_portal,
    refresh_token,
    reset_portal,
    get_calendar,
    get_currency_calendar,
    payment_dates,
    observation_schedule,
    load_instrument,
    preload_instruments,
    clear_instrument_cache,
)

# Internal exports (not for direct public use, but used by pages/✅Dispersion_Optimizer.py)
from functions.dispersion._pricing import (
    PricingEngine,
    PricingConfig,
    CrossCorridorVarianceSwap,
    _load_instrument_impl,
    get_trading_calendar,
    calculate_payment_dates,
)
