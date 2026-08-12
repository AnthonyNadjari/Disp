"""Golden-bundle generator for the dispersion optimizer regression gate.

Run from the repo root:

    python tests/golden/make_goldens.py

Writes three frozen synthetic run bundles + ``expected.json``:

    bundle_a_linear/      all-linear config      {last_carry, mean_payoff}
    bundle_b_minpayoff/   min_payoff-only        {min_payoff: 1.0}  (adaptive+NaNs)
    bundle_c_mixed/       mixed SLSQP config     {mean_payoff, hit_ratio, min_payoff}

POLICY (see project invariants):
- Goldens (a) and (b) are EXACT-PATH goldens: they are UNTOUCHABLE for the
  whole project.  Any change that shifts them beyond 1e-6 is a regression.
- Golden (c) is the mixed/SLSQP golden: it may be re-baselined ONCE, during
  Phase 3 only, with a before/after justification in the report.

Regenerating goldens is therefore an exceptional action — do it only when the
policy above explicitly allows it, and record why.

Determinism notes: the synthetic universes are small enough that the GA stops
on stagnation (never on wall-clock) and the exact-path local search completes
its scan well inside its time budget, so replays are machine-independent up
to numerical noise of the LP/SLSQP stack (hence the 1e-6 gate tolerance).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from functions.dispersion.models import (  # noqa: E402
    DispersionLeg,
    MissingDataPolicy,
    OptimizationConstraints,
)
from functions.dispersion.run_bundle import (  # noqa: E402
    load_run_bundle,
    save_run_bundle,
)

N_DAYS = 300
N_STOCKS = 10


def _universe(seed: int, with_nans: bool):
    rng = np.random.default_rng(seed)
    names = [f"G{i}" for i in range(N_STOCKS)]
    means = np.linspace(-0.2, 1.2, N_STOCKS)
    stds = np.linspace(0.6, 1.8, N_STOCKS)
    pnl = np.column_stack([
        rng.normal(means[i], stds[i], N_DAYS) for i in range(N_STOCKS)
    ])
    if with_nans:
        holes = rng.random(pnl.shape) < 0.02
        holes[:, :2] = False  # keep two always-complete columns
        pnl[holes] = np.nan
    legs = [
        DispersionLeg(
            variance_asset=names[i],
            strike_mono_var_swap=0.10 + 0.006 * i,
            min_weight=0.05,
            max_weight=0.60,
        )
        for i in range(N_STOCKS)
    ]
    col_map = {names[i]: i for i in range(N_STOCKS)}
    return legs, pnl, col_map


def _constraints() -> OptimizationConstraints:
    # Small enough to converge by stagnation well before the time limit.
    return OptimizationConstraints(
        min_stocks_long=2,
        max_stocks_long=3,
        min_stocks_short=0,
        max_stocks_short=0,
        max_net_strike=10.0,
        population_size=40,
        max_generations=150,
        time_limit_seconds=60.0,
        stagnation_limit=25,
    )


CASES = {
    "bundle_a_linear": {
        "score_weights": {"last_carry": 0.5, "mean_payoff": 0.5},
        "policy": MissingDataPolicy.FILL_ZERO,
        "with_nans": False,
        "universe_seed": 21,
        "seed": 1,
    },
    "bundle_b_minpayoff": {
        "score_weights": {"min_payoff": 1.0},
        "policy": MissingDataPolicy.ADAPTIVE_REWEIGHT,
        "with_nans": True,
        "universe_seed": 22,
        "seed": 2,
    },
    "bundle_c_mixed": {
        "score_weights": {"mean_payoff": 0.4, "hit_ratio": 0.3, "min_payoff": 0.3},
        "policy": MissingDataPolicy.FILL_ZERO,
        "with_nans": False,
        "universe_seed": 23,
        "seed": 3,
    },
}


def main() -> None:
    expected = {}
    for name, spec in CASES.items():
        legs, pnl, col_map = _universe(spec["universe_seed"], spec["with_nans"])
        path = os.path.join(_HERE, name)
        save_run_bundle(
            path,
            pnl_matrix=pnl,
            column_map=col_map,
            long_candidates=legs,
            short_candidates=[],
            constraints=_constraints(),
            score_weights=spec["score_weights"],
            seed=spec["seed"],
            missing_data_policy=spec["policy"],
            provenance={"golden_case": name, "universe_seed": spec["universe_seed"]},
        )
        result = load_run_bundle(path).replay()
        if not result.long_basket:
            raise RuntimeError(f"golden {name}: optimizer returned an empty basket")
        # Sanity: a second replay must be bit-identical on this machine
        again = load_run_bundle(path).replay()
        assert again.long_basket == result.long_basket and again.score == result.score, (
            f"golden {name}: replay is not deterministic on this machine"
        )
        expected[name] = {
            "long_basket": [[k, float(w)] for k, w in result.long_basket],
            "score": float(result.score),
            "net_strike": float(result.net_strike),
            "scoring_signature": result.scoring_signature,
            "generations_run": int(result.generations_run),  # informational
        }
        print(f"{name}: basket={[k for k, _ in result.long_basket]} "
              f"score={result.score:.6f} gens={result.generations_run}")

    with open(os.path.join(_HERE, "expected.json"), "w", encoding="utf-8") as f:
        json.dump(expected, f, indent=2, sort_keys=True)
    print("expected.json written.")


if __name__ == "__main__":
    main()
