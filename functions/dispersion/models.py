"""
Domain models for the dispersion engine.
═══════════════════════════════════════════════════════════════════════════════
CONVENTIONS (single source of truth — all components follow these)
═══════════════════════════════════════════════════════════════════════════════
STRIKES:
    Always stored as DECIMALS internally: 0.22 = 22%.
    UI/email display: multiply by 100 → "22%"
    If user input > 1, it's already in percent → divide by 100 on ingestion.
WEIGHTS:
    Always stored as DECIMALS: 0.30 = 30%.
    Sign convention: POSITIVE = long, NEGATIVE = short.
    A standard dispersion basket: stocks are +0.10 each, index is -1.0.
    UI/email display: multiply by 100 → "10%", "-100%".
CORRIDOR BOUNDS (barrier_up, barrier_down):
    Always stored as DECIMALS relative to spot: 1.30 = 130% = upper, 0.70 = 70% = lower.
    Display: multiply by 100 → "130/70 barriers"
LOCAL CAP:
    Multiplier of strike. Stored as float: 2.5 means cap = 2.5 × strike.
BASKET STRUCTURE DETECTION:
    Use Basket.structure_type property:
      - "dispersion": one or more stocks long, one index short
      - "basket_vs_basket": multiple longs + multiple shorts (no index)
      - "cross_corridor": each stock paired with its own index
      - "long_only": all weights positive, no short leg
"LONG LEG" / "SHORT LEG":
    Long leg = sum of (weight_i × pnl_i) for all tickers where weight > 0
    Short leg = sum of (|weight_i| × pnl_i) for all tickers where weight < 0
    Result (net) = Long Leg − Short Leg
    In timeseries columns: ["Long Leg", "Short Leg", "Result"]
    Long Leg is always positive when longs perform well.
    Short Leg is always positive when the short (index) has HIGH vol (bad for us).
    Result = Long Leg - Short Leg → positive = we made money.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Enums
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MissingDataPolicy(Enum):
    """The 3 ways a gap day (a name with no P&L observation) is filled.

    DROP_INCOMPLETE_DAYS ("drop")
        Keep only days where EVERY weighted name trades. Cleanest curve, but
        one stale/late-listed name shrinks the whole sample to the
        intersection of histories.

    ADAPTIVE_REWEIGHT ("reweight")  — default
        Redistribute a gapped name's weight to the active names, day by day
        (constant-vega exposure). With ``reweight_grace_days`` > 0, a gap of
        <= grace days keeps the name's weight allocated (contributing 0) —
        the basket runs under-invested for those days instead of being
        recomposed. Tracks the active-name count.

    FILL_ZERO ("fill_zero")
        Legacy Gaia_PP: a gap day contributes 0 for that name — the basket is
        silently under-invested on gap days; all days are kept.

    Enum VALUES are stable (serialized in run bundles) — never rename them.
    """
    DROP_INCOMPLETE_DAYS = "drop"
    ADAPTIVE_REWEIGHT = "reweight"
    FILL_ZERO = "fill_zero"

class ProductType(Enum):
    """Swap product type — determines which P&L kernel to use."""
    VOL_SWAP = "volswap"
    VAR_SWAP_CORRIDOR = "varswap_corridor"

class StructureType(Enum):
    """Detected basket structure — drives email/display logic."""
    DISPERSION = "dispersion"           # Stocks long vs index short
    BASKET_VS_BASKET = "basket_vs_basket"  # Multiple longs vs multiple shorts
    CROSS_CORRIDOR = "cross_corridor"   # Each stock paired with own index
    LONG_ONLY = "long_only"             # All positive weights

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration (PUBLIC)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class DispersionConfig:
    """
    Product definition + evaluation preferences.
    
    Tenor is specified as EITHER:
        - n_exp (int): number of business days (e.g. 252=1Y, 126=6M)
        - last_obs_date (date): maturity date (converted to n_exp via calendar)
    
    For solve/price: pass last_obs_date directly to the function (defines product schedule).
    For optimize/backtest: n_exp defines the rolling P&L window.
    
    To convert between them:
        from functions.common.utils import get_n_exp_from_date
        n_exp = get_n_exp_from_date(maturity_date, tickers)
    
    Common usage:
        cfg = DispersionConfig(cross_corridor=True, n_exp=252)
        cfg = DispersionConfig(product_type=ProductType.VOL_SWAP, n_exp=310)
    """
    # ── Product structure ──
    product_type: ProductType = ProductType.VAR_SWAP_CORRIDOR
    cross_corridor: bool = False
    n_exp: int = 252                      # Product tenor in business days. 252=1Y, 126=6M, 63=3M.
    barrier_up: float = 1.3              # Upper corridor (1.3 = 130% spot)
    barrier_down: float = 0.7            # Lower corridor (0.7 = 70% spot)
    local_cap: float = 2.5              # Variance payoff cap: max realized = local_cap × strike vol
    is_capped: bool = True               # Capped variance legs (note-style). False = uncapped (OTC swap)
    global_cap: float = 9999999.0        # Total basket cap (vol swap only)
    global_floor: float = -9999999.0     # Total basket floor (vol swap only)

    # ── Data handling (backtest/optimize) ──
    missing_data_policy: MissingDataPolicy = MissingDataPolicy.ADAPTIVE_REWEIGHT
    reweight_grace_days: int = 0         # ADAPTIVE only: gap days a name keeps its weight (0 = redistribute at once)
    adj_divs: bool = False               # Use total return index
    lookback_years: int = 5              # Default history if no start_date given

    # ── Derived properties ──
    @property
    def is_vol_swap(self) -> bool:
        return self.product_type == ProductType.VOL_SWAP

    @property
    def is_cross_corridor(self) -> bool:
        return self.cross_corridor

    # ── Validation ──
    def __post_init__(self):
        if self.barrier_up <= 1.0:
            raise ValueError(f"barrier_up must be > 1.0, got {self.barrier_up}")
        if self.barrier_down >= 1.0:
            raise ValueError(f"barrier_down must be < 1.0, got {self.barrier_down}")
        if self.barrier_down >= self.barrier_up:
            raise ValueError(f"barrier_down ({self.barrier_down}) must be < barrier_up ({self.barrier_up})")
        if self.n_exp < 1:
            raise ValueError(f"n_exp must be >= 1, got {self.n_exp}")
        if self.local_cap <= 0:
            raise ValueError(f"local_cap must be > 0, got {self.local_cap}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration (INTERNAL - kept during transition)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class SwapConfig:
    """
    Configuration for a variance/vol swap backtest or pricing.
    All values in DECIMAL convention (see module docstring).
    cross_corridor is orthogonal to product_type — you can have a vol swap
    cross-corridor OR a var swap corridor cross-corridor.
    """
    product_type: ProductType = ProductType.VOL_SWAP
    cross_corridor: bool = False     # orthogonal flag — each stock paired with its index
    n_exp: int = 310
    local_cap: float = 2.50
    barrier_down: float = 0.70   # Lower corridor: 0.70 = 70% of spot
    barrier_up: float = 1.30    # Upper corridor: 1.30 = 130% of spot
    adj_divs: bool = False
    missing_data_policy: MissingDataPolicy = MissingDataPolicy.ADAPTIVE_REWEIGHT
    #: ADAPTIVE_REWEIGHT only — a name whose data gap lasts <= this many days
    #: KEEPS its weight (contributing 0 on gap days) instead of having it
    #: redistributed immediately. 0 (default) = historical behaviour: any gap
    #: day redistributes at once.
    reweight_grace_days: int = 0
    lookback_years: int = 5
    start_date: Optional[date] = None       # Backtest start date (optional, overrides lookback)
    end_date: Optional[date] = None         # Backtest end date (optional)
    global_cap: float = 9999999.0           # Cap on final basket P&L (vol swap only)
    global_floor: float = -9999999.0        # Floor on final basket P&L (vol swap only)

    # ── Currency ──
    currency: str = "EUR"                   # Currency for the swap

    # ── Pricing-specific fields ──
    strike_date: Optional[date] = None      # For pricing (optional for backtest)
    last_obs_date: Optional[date] = None    # For pricing (optional for backtest)
    is_solve: bool = True                   # True=solve strike, False=price at strike
    capped: bool = True                     # Apply local cap
    eq_eq_lambda: float = 0.10              # Equity-equity correlation spread
    correl_floor: float = 0.0               # Correlation floor
    eq_fx_shift: float = -0.05              # Equity-FX shift
    correl_input_method: str = "Global Parameters"  # How to get correlations
    compute_atmf: bool = True               # Compute ATMF vol
    compute_zero_strike: bool = True        # Always compute zero-strike (internal)
    max_workers: int = 30                   # Parallel workers for pricing
    # Accept old kwargs without breaking
    def __init__(self, **kwargs):
        _MAP = {'is_capped': 'capped', 'eqeq_lambda': 'eq_eq_lambda', 'eqfx_shift': 'eq_fx_shift',
                'dvar': 'barrier_down', 'uvar': 'barrier_up', 'dbar': 'barrier_down', 'ubar': 'barrier_up'}
        for old, new in _MAP.items():
            if old in kwargs and new not in kwargs:
                kwargs[new] = kwargs.pop(old)
            elif old in kwargs:
                kwargs.pop(old)
        import dataclasses
        for f in dataclasses.fields(self.__class__):
            setattr(self, f.name, kwargs.get(f.name, f.default if f.default is not dataclasses.MISSING else None))
    # Backward compat read access
    @property
    def is_capped(self): return self.capped
    @property
    def eqeq_lambda(self): return self.eq_eq_lambda
    @property
    def eqfx_shift(self): return self.eq_fx_shift
    @property
    def dbar(self): return self.barrier_down
    @property
    def ubar(self): return self.barrier_up
    @property
    def dvar(self): return self.barrier_down
    @property
    def uvar(self): return self.barrier_up

    # ── Properties ──

    @property
    def is_vol_swap(self) -> bool:
        return self.product_type == ProductType.VOL_SWAP

    @property
    def is_cross_corridor(self) -> bool:
        return self.cross_corridor

    @property
    def is_corridor(self) -> bool:
        return self.product_type == ProductType.VAR_SWAP_CORRIDOR or self.cross_corridor

    @property
    def is_index_asset(self) -> bool:
        # Mono corridor uses .SPX as the asset (index), cross corridor uses individual stocks
        return self.ref_asset and ('Index' in self.ref_asset or 'index' in self.ref_asset)

    @property
    def product_display_name(self) -> str:
        """Human-readable product name for emails/UI."""
        base = {
            ProductType.VOL_SWAP: "Volswaps",
            ProductType.VAR_SWAP_CORRIDOR: "Variance Swaps",
        }[self.product_type]
        if self.cross_corridor:
            return f"Cross Corridor {base}"
        return base

    @property
    def corridor_display(self) -> str:
        """Display corridor bounds: '70.00/130.00 barriers, T/T-1'"""
        if not self.is_corridor:
            return ""
        d = self.barrier_down * 100
        u = self.barrier_up * 100
        div = "dividend adjusted" if self.adj_divs else "not dividend adjusted"
        return f"{d:.2f}/{u:.2f} barriers, T/T-1, {div}"

    def validate(self) -> None:
        assert self.n_exp > 0, "n_exp must be positive"
        assert self.local_cap > 0, "local_cap must be positive"
        # Only validate corridor bounds when corridor logic is actually used
        # (vol swaps never use barrier_up/barrier_down even if cross_corridor is set)
        if self.is_corridor and not self.is_vol_swap:
            assert 0 < self.barrier_down < self.barrier_up, f"Need 0 < barrier_down ({self.barrier_down}) < barrier_up ({self.barrier_up})"
            assert self.barrier_down < 1.0, f"barrier_down should be decimal (e.g. 0.70), got {self.barrier_down}"
            assert self.barrier_up > 1.0, f"barrier_up should be decimal (e.g. 1.30), got {self.barrier_up}"

    # ── Factory methods (for explicit, readable construction) ──

    @classmethod
    def volswap(cls, n_exp: int = 310, local_cap: float = 2.50,
                cross_corridor: bool = False, **kwargs) -> "SwapConfig":
        return cls(product_type=ProductType.VOL_SWAP, n_exp=n_exp, local_cap=local_cap,
                   cross_corridor=cross_corridor, **kwargs)

    @classmethod
    def varswap_corridor(cls, n_exp: int = 310, barrier_down: float = 0.70, barrier_up: float = 1.30,
                         local_cap: float = 2.50, cross_corridor: bool = False, **kwargs) -> "SwapConfig":
        return cls(product_type=ProductType.VAR_SWAP_CORRIDOR, n_exp=n_exp,
                   barrier_down=barrier_down, barrier_up=barrier_up, local_cap=local_cap,
                   cross_corridor=cross_corridor, **kwargs)

    @classmethod
    def pricing(cls, strike_date: date, last_obs_date: date, is_solve: bool = True,
                capped: bool = True, is_cross_corridor: bool = True,
                barrier_down: float = 0.70, barrier_up: float = 1.30, n_exp: int = 310,
                local_cap: float = 2.50, **kwargs) -> "SwapConfig":
        """Create a pricing configuration."""
        return cls(
            product_type=ProductType.VOL_SWAP,
            cross_corridor=is_cross_corridor,
            strike_date=strike_date,
            last_obs_date=last_obs_date,
            is_solve=is_solve,
            capped=capped,
            barrier_down=barrier_down,
            barrier_up=barrier_up,
            n_exp=n_exp,
            local_cap=local_cap,
            **kwargs
        )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DispersionLeg & Basket
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class DispersionLeg:
    """
    A single leg in the dispersion basket.
    Strikes and weights are always DECIMAL (see module conventions).
    """
    variance_asset: str
    strike_mono_var_swap: float      # Decimal: 0.22 = 22%
    weight: float = 0.0             # Signed: +0.10 = long 10%, -1.0 = short 100%
    min_weight: float = 0.0
    max_weight: float = 1.0
    corridor_condition_asset: Optional[str] = None   # For cross-corridor
    strike_cross_corridor: Optional[float] = None    # Cross-corridor strike (decimal)
    sector: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    # ── Absolute-Vega mode (Phase 4c) — axe inventory per name ──
    axe_target: Optional[float] = None   # axe Vega to clean (absolute units); None/0 = no axe
    axe_cap: Optional[float] = None      # hard Vega cap v_i <= cap (absolute units); None = uncapped

    @property
    def is_long(self) -> bool:
        return self.weight > 0

    @property
    def is_short(self) -> bool:
        return self.weight < 0

    @property
    def is_index_asset(self) -> bool:
        return 'Index' in self.variance_asset or 'index' in self.variance_asset

    @property
    def strike_pct(self) -> float:
        """Strike as percentage for display: 22.0"""
        return self.strike_mono_var_swap * 100

    @property
    def weight_pct(self) -> float:
        """Weight as percentage for display: 10.0 or -100.0"""
        return self.weight * 100

@dataclass
class Basket:
    """
    Complete basket definition. Single source of truth for all leg/structure logic.
    Construct via factory methods:
        Basket.from_tickers(tickers, strikes, weights, ...)
        Basket.from_legs(legs)
    """
    legs: List[DispersionLeg] = field(default_factory=list)

    # ── Leg access ──

    @property
    def long_legs(self) -> List[DispersionLeg]:
        """All legs with positive weight."""
        return [s for s in self.legs if s.weight > 0]

    @property
    def short_legs(self) -> List[DispersionLeg]:
        """All legs with negative weight."""
        return [s for s in self.legs if s.weight < 0]

    @property
    def has_short_leg(self) -> bool:
        return len(self.short_legs) > 0

    @property
    def long_tickers(self) -> List[str]:
        return [s.variance_asset for s in self.long_legs]

    @property
    def short_tickers(self) -> List[str]:
        return [s.variance_asset for s in self.short_legs]

    @property
    def all_tickers(self) -> List[str]:
        return [s.variance_asset for s in self.legs]

    @property
    def weights_dict(self) -> Dict[str, float]:
        """Signed weight dict: {variance_asset: weight}"""
        return {s.variance_asset: s.weight for s in self.legs}

    @property
    def strikes_dict(self) -> Dict[str, float]:
        """Strike dict: {variance_asset: strike_decimal}"""
        return {s.variance_asset: s.strike_mono_var_swap for s in self.legs}

    @property
    def corridor_condition_asset_map(self) -> Optional[Dict[str, str]]:
        """For cross-corridor: {variance_asset: corridor_condition_asset}"""
        d = {s.variance_asset: s.corridor_condition_asset for s in self.legs if s.corridor_condition_asset}
        return d if d else None

    # ── Structure detection ──

    @property
    def structure_type(self) -> StructureType:
        """Detect basket structure from weight signs and ticker types."""
        if any(s.corridor_condition_asset for s in self.legs):
            return StructureType.CROSS_CORRIDOR

        longs = self.long_legs
        shorts = self.short_legs
        if not shorts:
            return StructureType.LONG_ONLY

        # Single short that looks like an index → classic dispersion
        if len(shorts) == 1 and shorts[0].is_index_asset:
            return StructureType.DISPERSION

        # Multiple shorts → basket vs basket
        if len(shorts) > 1:
            return StructureType.BASKET_VS_BASKET

        # Single short that's not an index → still dispersion-like
        return StructureType.DISPERSION

    @property
    def short_leg_display(self) -> str:
        """Display name for the short leg (e.g. 'SX5E Index')."""
        shorts = self.short_legs
        if not shorts:
            return ""
        if len(shorts) == 1:
            return shorts[0].variance_asset
        return f"{len(shorts)} stocks"

    @property
    def offer(self) -> float:
        """Weighted strike (the 'offer'): sum(weight_i × strike_i) for display."""
        if self.structure_type == StructureType.CROSS_CORRIDOR:
            # Cross-corridor: sum(weight × (strike_mono - strike_cross))
            total = 0.0
            for s in self.legs:
                strike_cross = s.strike_cross_corridor or s.strike_mono_var_swap
                total += abs(s.weight) * (s.strike_mono_var_swap - strike_cross)
            return round(total * 100, 2)  # Display as percentage
        else:
            # Standard: sum(weight × strike) × 100
            return round(sum(s.weight * s.strike_mono_var_swap for s in self.legs) * 100, 2)

    # ── Factory methods ──

    @classmethod
    def from_tickers(
        cls,
        tickers: List[str],
        strikes: Dict[str, float],
        weights: Dict[str, float],
        corridor_condition_assets: Optional[Dict[str, str]] = None,
        cross_corridor_strikes: Optional[Dict[str, float]] = None,
    ) -> "Basket":
        """Construct from dicts (the format used by the backtest engine)."""
        legs = []
        for t in tickers:
            legs.append(DispersionLeg(
                variance_asset=t,
                strike_mono_var_swap=strikes.get(t, 0),
                weight=weights.get(t, 0),
                corridor_condition_asset=corridor_condition_assets.get(t) if corridor_condition_assets else None,
                strike_cross_corridor=cross_corridor_strikes.get(t) if cross_corridor_strikes else None,
            ))
        return cls(legs=legs)

    @classmethod
    def from_legs(cls, legs: List[DispersionLeg]) -> "Basket":
        return cls(legs=legs)

    def to_dataframe(self) -> pd.DataFrame:
        """Export to DataFrame (for display or email tables)."""
        if self.structure_type == StructureType.CROSS_CORRIDOR:
            return pd.DataFrame([{
                'Variance Asset': s.variance_asset,
                'Corridor Condition Asset': s.corridor_condition_asset or '',
                'Strike Mono Var Swap (%)': s.strike_pct,
                'Strike Cross Corridor (%)': (s.strike_cross_corridor or 0) * 100,
                'Weight (%)': s.weight_pct,
            } for s in self.legs])
        else:
            return pd.DataFrame([{
                'Variance Asset': s.variance_asset,
                'Strike Mono Var Swap (%)': s.strike_pct,
                'Weight (%)': s.weight_pct,
                'Sector': s.sector or '',
            } for s in self.legs])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Optimizer Input (for BasketInput compatibility during migration)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class BasketInput:
    """
    Input for the genetic optimizer (separate long/short candidate pools).
    Standard columns: Variance Asset | Strike Mono Var Swap (%) | Min Weight | Max Weight
    Cross-corridor columns: Variance Asset | Corridor Condition Asset | Strike Cross Corridor (%) | Strike Mono Var Swap (%) | Min Weight | Max Weight
    """
    long_candidates: List[DispersionLeg] = field(default_factory=list)
    short_candidates: List[DispersionLeg] = field(default_factory=list)

    @property
    def is_long_only(self) -> bool:
        return len(self.short_candidates) == 0

    @property
    def all_candidates(self) -> List[DispersionLeg]:
        return self.long_candidates + self.short_candidates

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Optimizer Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class OptimizationConstraints:
    """Constraints for the genetic optimizer."""
    min_stocks_long: int = 3
    max_stocks_long: int = 8
    min_stocks_short: int = 2
    max_stocks_short: int = 5
    max_net_strike: float = 0.15    # Decimal: 0.15 = max 15% net strike
    time_limit_seconds: float = 30.0
    population_size: int = 250
    max_generations: int = 700
    stagnation_limit: int = 150  # Generations without improvement before stopping
    elite_ratio: float = 0.05    # Fraction of population preserved each generation
    mutation_rate: float = 0.1   # Probability of mutation per offspring
    crossover_rate: float = 0.8  # Probability of crossover per pair
    tournament_size: int = 3     # Number of individuals in tournament selection

@dataclass
class VegaConfig:
    """Absolute-Vega mode (Phase 4c toggle).

    OFF (``optimize(vega=None)``, the default) keeps the historical
    percentage-weight behaviour untouched.

    ON: each name gets an ABSOLUTE Vega v_i >= 0; the basket total
    V = Σ v_i is FREE within [v_min, v_max]; the existing per-name Min/Max
    Weight inputs are reinterpreted as concentration bounds on v_i/V; the
    per-name axe inventory (:attr:`DispersionLeg.axe_target` /
    :attr:`DispersionLeg.axe_cap`) bounds v_i <= cap and feeds the
    recycling objectives:

    - ``axe_book_cleaned``  (A) = Σ min(v_i, target_i) / Σ_universe target_i
    - ``axe_package_recycled`` (B) = Σ min(v_i, target_i) / V

    The basket P&L stays Σ(v_i · pnl_i) / V — i.e. the percentage-weight
    series with w_i = v_i / V — so scores and the backtest curve remain
    directly comparable with vega OFF.  ``max_net_strike`` also stays on
    the w_i scale.
    """
    v_min: float
    v_max: float
    enabled: bool = True

    def __post_init__(self):
        if self.v_min <= 0:
            raise ValueError(f"VegaConfig.v_min must be > 0, got {self.v_min}")
        if self.v_max < self.v_min:
            raise ValueError(
                f"VegaConfig.v_max ({self.v_max}) must be >= v_min ({self.v_min})")


@dataclass
class BucketConstraint:
    """Per-bucket (region/sector) constraints for the LONG leg.

    ``bucket`` matches :attr:`DispersionLeg.sector` (the 'Sector' column of
    the input DataFrame — free label, e.g. "US" / "EU").

    Cardinality: between ``min_names`` and ``max_names`` selected names from
    the bucket (``max_names=None`` = unlimited).
    Weight: total long weight of the bucket in [min_weight, max_weight],
    DECIMAL convention (0.30 = 30%; ``max_weight=None`` = unlimited).

    Defaults are fully inactive.  A ``min_weight > 0`` requires
    ``min_names >= 1`` (a subset without the bucket could never satisfy the
    weight floor) — validated by the optimizer at construction time.
    """
    bucket: str
    min_names: int = 0
    max_names: Optional[int] = None
    min_weight: float = 0.0
    max_weight: Optional[float] = None

    def __post_init__(self):
        if not str(self.bucket).strip():
            raise ValueError("BucketConstraint.bucket must be a non-empty label")
        if self.min_names < 0:
            raise ValueError(f"BucketConstraint({self.bucket}): min_names must be >= 0, got {self.min_names}")
        if self.max_names is not None and self.max_names < self.min_names:
            raise ValueError(
                f"BucketConstraint({self.bucket}): max_names ({self.max_names}) "
                f"< min_names ({self.min_names})")
        if not (0.0 <= self.min_weight <= 1.0):
            raise ValueError(
                f"BucketConstraint({self.bucket}): min_weight must be a DECIMAL in [0, 1] "
                f"(0.30 = 30%), got {self.min_weight}")
        if self.max_weight is not None:
            if not (0.0 < self.max_weight <= 1.0):
                raise ValueError(
                    f"BucketConstraint({self.bucket}): max_weight must be a DECIMAL in (0, 1] "
                    f"(0.30 = 30%), got {self.max_weight}")
            if self.max_weight < self.min_weight:
                raise ValueError(
                    f"BucketConstraint({self.bucket}): max_weight ({self.max_weight}) "
                    f"< min_weight ({self.min_weight})")

    @property
    def has_weight_bounds(self) -> bool:
        return self.min_weight > 0.0 or self.max_weight is not None

    @property
    def has_count_bounds(self) -> bool:
        return self.min_names > 0 or self.max_names is not None


@dataclass
class ScoreWeights:
    """
    Multi-objective fitness weights for optimization. Auto-normalized to sum=1.
    """
    last_value: float = 0.3
    avg_5y: float = 0.1
    avg_3y: float = 0.2
    max_drawdown: float = 0.1
    hit_ratio: float = 0.3

    def __post_init__(self):
        self._normalize()

    def _normalize(self):
        total = self.last_value + self.avg_5y + self.avg_3y + self.max_drawdown + self.hit_ratio
        if total > 0 and abs(total - 1.0) > 1e-6:
            self.last_value /= total
            self.avg_5y /= total
            self.avg_3y /= total
            self.max_drawdown /= total
            self.hit_ratio /= total

    def as_array(self) -> np.ndarray:
        return np.array([self.last_value, self.avg_5y, self.avg_3y, self.max_drawdown, self.hit_ratio])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Result Types (PUBLIC - returned by solve()/price())
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class SolveResult:
    """Returned by solve()."""
    results_df: pd.DataFrame    # Pre-formatted strings (see columns below)
    success: bool = True
    failed_tickers: List[str] = field(default_factory=list)
    error: Optional[str] = None

@dataclass
class PriceResult:
    """Returned by price()."""
    results_df: pd.DataFrame    # Pre-formatted strings
    success: bool = True
    failed_tickers: List[str] = field(default_factory=list)
    error: Optional[str] = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Result Types (INTERNAL - used by optimizer/backtester)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class OptimizationResult:
    """Output from the genetic optimizer."""
    long_basket: List[Tuple[str, float]]   # [(ticker, weight_decimal), ...]
    short_basket: List[Tuple[str, float]]
    long_strikes: List[Tuple[str, float]] = field(default_factory=list)  # [(ticker, strike_decimal), ...]
    short_strikes: List[Tuple[str, float]] = field(default_factory=list)
    long_strike_weighted: float = 0.0
    short_strike_weighted: float = 0.0
    net_strike: float = 0.0               # Decimal
    score: float = 0.0
    generations_run: int = 0
    converged: bool = False
    is_cross_corridor: bool = False
    # Cross-corridor data: [(variance_asset, corridor_condition_asset, cross_corridor_strike), ...]
    long_cross_corridor: List[Tuple[str, str, float]] = field(default_factory=list)
    short_cross_corridor: List[Tuple[str, str, float]] = field(default_factory=list)
    scoring_mode: str = "legacy"  # "v2" | "legacy" — which scorer actually ran
    unsmoothed_basket: list = None  # [(ticker, weight), ...] pre-smooth weights, or None
    backtest: Optional["BacktestResult"] = None  # Backtest of winning basket (computed free)
    backtest_unsmoothed: Optional["BacktestResult"] = None  # Pre-smooth backtest (only when smoothing active)
    # ── Scoring signature (reproducibility fingerprint of the run) ──
    scoring_signature: Optional[str] = None  # sha256[:16] of (active metrics+weights, seed, n_samples, reference size)
    seed: Optional[int] = None               # RNG seed the optimizer ran with
    reference_size: Optional[int] = None     # number of baskets in the fitted normalizer reference
    # ── Absolute-Vega mode outputs (None when the toggle is OFF) ──
    total_vega: Optional[float] = None       # V = Σ v_i (absolute units)
    vega_basket: list = None                 # [(ticker, v_i absolute), ...] aligned with long_basket
    axe_cleaned: Optional[float] = None      # criterion A of the delivered basket (fraction of the axe book)
    axe_recycled: Optional[float] = None     # criterion B of the delivered basket (fraction of the package)
    # ── Bootstrap robustness diagnostic (None unless robustness_check=True) ──
    robustness: Optional[Dict] = None        # {n_draws, n_challengers, top1_freq, top3_freq, winner_raw_ci}

    @property
    def is_long_only(self) -> bool:
        return len(self.short_basket) == 0

    def to_dict(self) -> Dict:
        return {
            "long_basket": self.long_basket,
            "short_basket": self.short_basket,
            "net_strike_pct": self.net_strike * 100,
            "score": self.score,
            "generations": self.generations_run,
        }

    def to_backtester_df(self) -> "pd.DataFrame":
        """Build a DataFrame matching the backtester input format.
        Cross-corridor format: Variance Asset | Corridor Condition Asset | Strike Cross Corridor (%) | Strike Mono Var Swap (%) | Weight (%)
        Standard format: Variance Asset | Strike Mono Var Swap (%) | Weight (%)
        """
        import pandas as pd
        rows = []
        if self.is_cross_corridor and self.long_cross_corridor:
            # Cross-corridor format
            xcorr_map = {t: (idx, xcs) for t, idx, xcs in self.long_cross_corridor}
            for ticker, weight in self.long_basket:
                strike_mono = dict(self.long_strikes).get(ticker, 0)
                idx_ticker, strike_cross = xcorr_map.get(ticker, ("", 0))
                rows.append({
                    "Variance Asset": idx_ticker,
                    "Corridor Condition Asset": ticker,
                    "Strike Cross Corridor (%)": round(strike_cross * 100, 2),
                    "Strike Mono Var Swap (%)": round(strike_mono * 100, 2),
                    "Weight (%)": round(weight * 100, 2),
                })
            # Short side (if any)
            xcorr_map_s = {t: (idx, xcs) for t, idx, xcs in self.short_cross_corridor}
            for ticker, weight in self.short_basket:
                strike_mono = dict(self.short_strikes).get(ticker, 0)
                idx_ticker, strike_cross = xcorr_map_s.get(ticker, ("", 0))
                rows.append({
                    "Variance Asset": idx_ticker,
                    "Corridor Condition Asset": ticker,
                    "Strike Cross Corridor (%)": round(strike_cross * 100, 2),
                    "Strike Mono Var Swap (%)": round(strike_mono * 100, 2),
                    "Weight (%)": round(-weight * 100, 2),
                })
        else:
            # Standard format (no Side column needed)
            strike_map = dict(self.long_strikes)
            for ticker, weight in self.long_basket:
                rows.append({
                    "Variance Asset": ticker,
                    "Strike Mono Var Swap (%)": round(strike_map.get(ticker, 0) * 100, 2),
                    "Weight (%)": round(weight * 100, 2),
                })
            strike_map_s = dict(self.short_strikes)
            for ticker, weight in self.short_basket:
                rows.append({
                    "Variance Asset": ticker,
                    "Strike Mono Var Swap (%)": round(strike_map_s.get(ticker, 0) * 100, 2),
                    "Weight (%)": round(-weight * 100, 2),
                })
        return pd.DataFrame(rows)

@dataclass
class BacktestResult:
    """
    Output from a backtest run.
    timeseries columns: ["Long Leg", "Short Leg", "Result"]
      - Long Leg: weighted sum of long-side P&L (positive = good)
      - Short Leg: weighted sum of short-side P&L (positive = bad for us)
      - Result: Long Leg - Short Leg (positive = we profit)
    """
    timeseries: pd.DataFrame
    active_legs_count: Optional[pd.Series] = None
    per_leg_pnl: Optional[pd.DataFrame] = None  # columns=keys, index=dates, values=daily P&L
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def result_series(self) -> pd.Series:
        """The net P&L series."""
        return self.timeseries["Result"]

    @property
    def hit_ratio(self) -> float:
        s = self.result_series.dropna()
        return float((s > 0).sum() / max(1, (s != 0).sum()) * 100) if len(s) > 0 else 0.0

    @property
    def mean_return(self) -> float:
        return float(self.result_series.mean())

    @property
    def last_value(self) -> float:
        return float(self.result_series.iloc[-1]) if len(self.timeseries) > 0 else 0.0

    @property
    def max_drawdown(self) -> float:
        """Max peak-to-trough drawdown as % of peak cumulative P&L.
        Returns a negative number (e.g., -45.0 means 45% drop from peak)."""
        cumsum = self.result_series.cumsum()
        if len(cumsum) == 0:
            return 0.0
        peak = cumsum.cummax()
        drawdown = cumsum - peak
        # Express as % of peak (avoid div by zero when peak <= 0)
        peak_at_trough = peak[drawdown.idxmin()] if len(drawdown) > 0 else 0.0
        if peak_at_trough > 1e-8:
            return float(drawdown.min() / peak_at_trough * 100)
        # Fallback: if peak never positive, return raw drawdown
        return float(drawdown.min())

    @property
    def has_short_leg(self) -> bool:
        """Whether the backtest produced a non-trivial short leg."""
        return self.timeseries["Short Leg"].abs().sum() > 0

    def compute_metrics(self) -> Dict[str, float]:
        """Compute and cache all summary metrics."""
        s = self.result_series
        self.metrics = {
            "hit_ratio": self.hit_ratio,
            "mean_return": self.mean_return,
            "last_value": self.last_value,
            "max_drawdown": self.max_drawdown,
            "min": float(s.min()),
            "max": float(s.max()),
            "p25": float(s.quantile(0.25)) if len(s) > 0 else 0.0,
            "p50": float(s.quantile(0.50)) if len(s) > 0 else 0.0,
            "p75": float(s.quantile(0.75)) if len(s) > 0 else 0.0,
            "n_observations": int((s != 0).sum()),
        }
        return self.metrics

    # ── Per-stock breakdown & clean export (headless-first; the UI calls the
    #    exact same methods, so both paths can never diverge) ─────────────────

    def per_stock(self, name: str) -> pd.DataFrame:
        """Daily P&L lines for ONE name: 'net' always; in cross-corridor mode
        also 'mono_leg' (stock var) and 'cross_leg' (index var in the stock's
        corridor), with net = mono − cross.

            bt = backtest(df, cfg)
            bt.per_stock("005930 KP Equity")        # 3-column DataFrame
        """
        if self.per_leg_pnl is None or name not in self.per_leg_pnl.columns:
            avail = list(self.per_leg_pnl.columns)[:10] if self.per_leg_pnl is not None else []
            raise KeyError(f"'{name}' has no per-leg P&L. Available sample: {avail}")
        out = pd.DataFrame({"net": self.per_leg_pnl[name]})
        clp = getattr(self, "cross_leg_pnl", None)
        if clp is not None and isinstance(clp, pd.DataFrame) and not clp.empty:
            for col in clp.columns:
                if col.startswith(f"{name} (mono leg"):
                    out["mono_leg"] = clp[col]
                elif col.startswith(f"{name} (cross leg"):
                    out["cross_leg"] = clp[col]
        return out

    def per_stock_stats(self, name: str = None) -> pd.DataFrame:
        """Tidy stats table — one row per (stock, line): mean, hit_ratio (%),
        last, min, max_drawdown, n_obs. `name=None` = every stock."""
        names = [name] if name is not None else (
            list(self.per_leg_pnl.columns) if self.per_leg_pnl is not None else [])
        rows = []
        for nm in names:
            df = self.per_stock(nm)
            for line in df.columns:
                v = df[line].dropna()
                if len(v) == 0:
                    rows.append({"stock": nm, "line": line, "mean": 0.0, "hit_ratio": 0.0,
                                 "last": 0.0, "min": 0.0, "max_drawdown": 0.0, "n_obs": 0})
                    continue
                cum = v.cumsum()
                rows.append({
                    "stock": nm, "line": line,
                    "mean": float(v.mean()),
                    "hit_ratio": float((v > 0).sum() / max(1, (v != 0).sum()) * 100),
                    "last": float(v.iloc[-1]),
                    "min": float(v.min()),
                    "max_drawdown": float((cum - cum.cummax()).min()),
                    "n_obs": int(len(v)),
                })
        return pd.DataFrame(rows)

    def to_frames(self) -> Dict[str, pd.DataFrame]:
        """Every output as tidy DataFrames: {'curve', 'per_leg_net',
        'mono_cross_legs' (cross mode), 'per_stock_stats', 'summary'}."""
        frames: Dict[str, pd.DataFrame] = {}
        curve = self.timeseries.copy()
        if self.active_legs_count is not None and len(self.active_legs_count) == len(curve):
            curve["Active Stocks"] = self.active_legs_count.values
        frames["curve"] = curve
        if self.per_leg_pnl is not None and not self.per_leg_pnl.empty:
            frames["per_leg_net"] = self.per_leg_pnl.copy()
            frames["per_stock_stats"] = self.per_stock_stats()
        clp = getattr(self, "cross_leg_pnl", None)
        if clp is not None and isinstance(clp, pd.DataFrame) and not clp.empty:
            frames["mono_cross_legs"] = clp.copy()
        m = self.metrics or self.compute_metrics()
        frames["summary"] = pd.DataFrame([m])
        return frames

    def export(self, path: str) -> str:
        """Write everything to ONE file: '.xlsx' (needs openpyxl or xlsxwriter)
        or '.zip' of CSVs (no extra dependency — always works).

            bt.export("my_backtest.zip")
        """
        frames = self.to_frames()
        if str(path).lower().endswith(".xlsx"):
            try:
                with pd.ExcelWriter(path) as xw:
                    for sheet, df in frames.items():
                        df.to_excel(xw, sheet_name=sheet[:31])
            except ImportError as e:
                raise RuntimeError(
                    f"Excel export needs 'openpyxl' (or 'xlsxwriter'): {e}. "
                    f"Use a '.zip' path instead — it has no dependency.") from e
            return path
        if str(path).lower().endswith(".zip"):
            with open(path, "wb") as f:
                f.write(frames_to_zip_bytes(frames))
            return path
        raise ValueError(f"export path must end in .xlsx or .zip, got: {path!r}")


def frames_to_zip_bytes(frames: "Dict[str, pd.DataFrame]") -> bytes:
    """{name: DataFrame} → zip archive of CSVs, as bytes (UI download button
    and BacktestResult.export('.zip') share this exact function)."""
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, df in frames.items():
            z.writestr(f"{name}.csv", df.to_csv(index=True))
    return buf.getvalue()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _to_decimal(value) -> float:
    """
    Normalize a value to decimal convention.
    If |value| > 1, assume it's in percentage → divide by 100.
    If |value| <= 1, assume it's already decimal.
    Examples: 22 → 0.22, 130 → 1.30, 0.22 → 0.22, -60 → -0.60
    """
    try:
        v = float(value)
    except (ValueError, TypeError):
        return 0.0
    if abs(v) > 1.0:
        return v / 100.0
    return v

def piecewise_function(x, a: float, b: float, c: float, d: float):
    """
    Piecewise charge function.
    Args:
        x: spread value(s)
        a: charge for high spread (x > b)
        b: threshold
        c: charge for medium spread (0 <= x <= b)
        d: charge for negative spread (x < 0)
    """
    if isinstance(x, (np.ndarray, list)):
        x = np.asarray(x, dtype=float)
        return np.where(x > b, a, np.where(x >= 0, c, d))
    if x > b:
        return a
    elif x >= 0:
        return c
    else:
        return d

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Charge Function
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ChargeFunction:
    """
    Piecewise charge function for dispersion pricing.
    The charge function applies fees based on spread vs ATMF:
      - High spread (> threshold): high_charge
      - Medium spread (0 to threshold): medium_charge
      - Negative spread (< 0): negative_charge
    Callable — pass directly to PricingEngine.run(charge_function=...).
    Usage:
        cf = ChargeFunction(high_charge=0.25, threshold=5.0, medium_charge=0.5, negative_charge=2.0)
        fee = cf(spread=6.2)   # → 0.25
        fee = cf(spread=3.0)   # → 0.5
        fee = cf(spread=-1.0)  # → 2.0
    """
    high_charge: float = 0.25       # charge if spread > threshold
    threshold: float = 5.0          # breakpoint between high and medium
    medium_charge: float = 0.5      # charge if 0 <= spread <= threshold
    negative_charge: float = 2.0    # charge if spread < 0

    def __call__(self, x) -> float:
        """Evaluate charge for a given spread value or array."""
        return piecewise_function(x, self.high_charge, self.threshold,
                                  self.medium_charge, self.negative_charge)

    def to_dict(self) -> dict:
        return {
            'high_charge': self.high_charge,
            'threshold': self.threshold,
            'medium_charge': self.medium_charge,
            'negative_charge': self.negative_charge,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'ChargeFunction':
        return cls(
            high_charge=d.get('high_charge', 0.25),
            threshold=d.get('threshold', 5.0),
            medium_charge=d.get('medium_charge', 0.5),
            negative_charge=d.get('negative_charge', 2.0),
        )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Result Types (INTERNAL - for single-ticker pricing internals)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class TickerSolveResult:
    """Result of solving a single corridor swap strike."""
    ticker: Optional[str] = None
    index: Optional[str] = None
    ticker_strike: Optional[float] = None
    index_strike: Optional[float] = None
    spread: Optional[float] = None
    atmf_vol: Optional[float] = None
    index_atmf_vol: Optional[float] = None
    expected_var_cross: Optional[float] = None
    expected_var_mono: Optional[float] = None
    range_accrual: Optional[float] = None
    currency: Optional[str] = None
    elapsed: float = 0.0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and self.ticker_strike is not None

    @property
    def strike(self) -> Optional[float]:
        """Alias - primary strike (ticker)."""
        return self.ticker_strike

@dataclass
class TickerPriceResult:
    """Result of pricing at given strikes."""
    ticker_mid: float = 0.0
    index_mid: float = 0.0
    spread: float = 0.0

@dataclass
class FpfResult:
    """Result of FPF XML generation."""
    ticker_fpf: Optional[str] = None
    index_fpf: Optional[str] = None
    n_obs: int = 0

@dataclass
class FullResult:
    """Combined result from a full solve+price+fpf run."""
    solve: Optional[TickerSolveResult] = None
    range_accrual: Optional[float] = None
    expected_variance: Optional[float] = None
    fpf: Optional[FpfResult] = None

    @property
    def success(self) -> bool:
        return self.solve is not None and self.solve.success

# Alias