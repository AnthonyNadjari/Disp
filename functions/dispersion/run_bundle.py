"""run_bundle.py — persist and replay a full optimizer run, offline.

A *run bundle* freezes everything the genetic optimizer consumed for one run:

    <bundle_dir>/
        pnl_matrix.parquet   # the P&L matrix (NaNs preserved), columns = keys
        bundle.json          # legs, constraints, weights, seed, policy, result

Guarantee: loading a bundle and calling :meth:`RunBundle.replay` re-runs
``DispersionOptimizer`` on the exact same inputs — same bundle + same seed
⇒ same basket / score / weights (bit-for-bit on a fixed numpy/scipy stack;
within numerical tolerance across library versions).  No Bloomberg, no
Streamlit, no network.

Intended uses
-------------
- Reproduce a production run on a dev machine ("what did the GA see?").
- Golden regression tests (tests/golden/) — frozen bundles + expected outputs.
- Bug reports: attach the bundle directory.

Usage
-----
>>> from functions.dispersion.run_bundle import load_run_bundle
>>> b = load_run_bundle("runs/2026-08-12_ega")
>>> result = b.replay()
>>> result.long_basket

From the public API, ``optimize(..., save_bundle_path="runs/my_run")`` writes
the bundle automatically after the GA completes.

Notes
-----
- The wall-clock ``time_limit_seconds`` is stored as-is.  For deterministic
  replay of *converged* runs this is irrelevant (the GA stops on stagnation);
  for time-capped runs, replay on a slower machine may stop earlier — use
  ``replay(time_limit_override=...)`` to widen the budget if needed.
- The backtest of the winner is NOT part of the bundle (it needs price data);
  the bundle covers the optimization itself.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from functions.dispersion.models import (
    BucketConstraint,
    DispersionLeg,
    MissingDataPolicy,
    OptimizationConstraints,
    OptimizationResult,
    VegaConfig,
)

__all__ = [
    "RunBundle",
    "save_run_bundle",
    "load_run_bundle",
]

BUNDLE_VERSION = 2
#: Version history:
#:   1 — reweight_grace_days was stored but INERT (a no-op knob); replaying a v1
#:       bundle forces grace=0 to reproduce its true historical behaviour.
#:   2 — grace is live (ADAPTIVE_REWEIGHT holds a gapped name's weight for
#:       <= grace days); the stored value is honoured on replay.
_MATRIX_FILE = "pnl_matrix.parquet"
_MASK_FILE = "active_mask.parquet"   # optional: adaptive grace mask (grace > 0 runs)
_JSON_FILE = "bundle.json"


# ---------------------------------------------------------------------------
# (De)serialization helpers
# ---------------------------------------------------------------------------


def _leg_to_dict(leg: DispersionLeg) -> Dict:
    return {
        "variance_asset": leg.variance_asset,
        "strike_mono_var_swap": float(leg.strike_mono_var_swap),
        "weight": float(leg.weight),
        "min_weight": float(leg.min_weight),
        "max_weight": float(leg.max_weight),
        "corridor_condition_asset": leg.corridor_condition_asset,
        "strike_cross_corridor": (None if leg.strike_cross_corridor is None
                                  else float(leg.strike_cross_corridor)),
        "sector": leg.sector,
        "metrics": {k: float(v) for k, v in (leg.metrics or {}).items()},
        "axe_target": (None if leg.axe_target is None else float(leg.axe_target)),
        "axe_cap": (None if leg.axe_cap is None else float(leg.axe_cap)),
    }


def _leg_from_dict(d: Dict) -> DispersionLeg:
    return DispersionLeg(
        variance_asset=d["variance_asset"],
        strike_mono_var_swap=float(d["strike_mono_var_swap"]),
        weight=float(d.get("weight", 0.0)),
        min_weight=float(d.get("min_weight", 0.0)),
        max_weight=float(d.get("max_weight", 1.0)),
        corridor_condition_asset=d.get("corridor_condition_asset"),
        strike_cross_corridor=(None if d.get("strike_cross_corridor") is None
                               else float(d["strike_cross_corridor"])),
        sector=d.get("sector"),
        metrics=dict(d.get("metrics") or {}),
        axe_target=(None if d.get("axe_target") is None else float(d["axe_target"])),
        axe_cap=(None if d.get("axe_cap") is None else float(d["axe_cap"])),
    )


def _constraints_to_dict(c: OptimizationConstraints) -> Dict:
    return {f.name: getattr(c, f.name) for f in dataclasses.fields(c)}


def _constraints_from_dict(d: Dict) -> OptimizationConstraints:
    known = {f.name for f in dataclasses.fields(OptimizationConstraints)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(
            f"bundle.json constraints carry unknown fields {sorted(unknown)} — "
            f"bundle written by a newer engine? Known: {sorted(known)}"
        )
    return OptimizationConstraints(**d)


def _result_to_dict(r: OptimizationResult) -> Dict:
    """Snapshot of the outcome for comparison — NOT used to rebuild objects."""
    return {
        "long_basket": [[k, float(w)] for k, w in r.long_basket],
        "short_basket": [[k, float(w)] for k, w in r.short_basket],
        "score": float(r.score),
        "net_strike": float(r.net_strike),
        "generations_run": int(r.generations_run),
        "converged": bool(r.converged),
        "scoring_mode": r.scoring_mode,
        "scoring_signature": r.scoring_signature,
        "seed": r.seed,
        "reference_size": r.reference_size,
    }


def _json_default(o):
    if isinstance(o, Enum):
        return o.value
    if isinstance(o, (_dt.date, _dt.datetime)):
        return o.isoformat()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"run_bundle: cannot serialize {type(o).__name__} to JSON")


# ---------------------------------------------------------------------------
# RunBundle
# ---------------------------------------------------------------------------


@dataclass
class RunBundle:
    """In-memory representation of a saved optimizer run (see module docstring)."""

    pnl_matrix: np.ndarray
    active_mask: Optional[np.ndarray]
    column_map: Dict[str, int]
    long_candidates: List[DispersionLeg]
    short_candidates: List[DispersionLeg]
    constraints: OptimizationConstraints
    score_weights: Dict[str, float]
    seed: int
    missing_data_policy: MissingDataPolicy
    adj_divs: bool = False
    reweight_grace_days: int = 0
    is_cross_corridor: bool = False
    global_cap: float = 9999999.0
    global_floor: float = -9999999.0
    bisect_in_ga: bool = False
    forced_long_indices: Optional[List[int]] = None
    n_reference_samples: Optional[int] = None  # None = adaptive default (300/800)
    bucket_constraints: Optional[List[BucketConstraint]] = None
    vega_config: Optional[VegaConfig] = None   # absolute-Vega toggle (None = OFF)
    dates: Optional[List] = None            # optional row index of pnl_matrix
    config: Optional[Dict] = None           # DispersionConfig snapshot (informational)
    provenance: Dict = field(default_factory=dict)  # forced/excluded tickers, dates, ...
    result: Optional[Dict] = None           # outcome snapshot at save time
    meta: Dict = field(default_factory=dict)

    def replay(
        self,
        *,
        time_limit_override: Optional[float] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        logger: Optional[Callable[[str, str], None]] = None,
    ) -> OptimizationResult:
        """Re-run the genetic optimizer on the bundled inputs.

        Same bundle + same seed ⇒ same basket / score / weights.  Raises if
        the bundle is internally inconsistent (clear error, no fallback).
        """
        from functions.dispersion._optimizer import DispersionOptimizer
        from functions.dispersion.scoring import MetricWeights

        constraints = self.constraints
        if time_limit_override is not None:
            constraints = dataclasses.replace(
                constraints, time_limit_seconds=float(time_limit_override))

        optimizer = DispersionOptimizer(
            long_candidates=[dataclasses.replace(l) for l in self.long_candidates],
            short_candidates=[dataclasses.replace(l) for l in self.short_candidates],
            pnl_matrix=self.pnl_matrix,
            column_map=dict(self.column_map),
            constraints=constraints,
            logger=logger,
            missing_data_policy=self.missing_data_policy,
            adj_divs=self.adj_divs,
            reweight_grace_days=self.reweight_grace_days,
            active_mask=self.active_mask,
            is_cross_corridor=self.is_cross_corridor,
            seed=self.seed,
            global_cap=self.global_cap,
            global_floor=self.global_floor,
            metric_weights=MetricWeights(dict(self.score_weights)),
            progress_callback=progress_callback,
            bisect_in_ga=self.bisect_in_ga,
            forced_long_indices=(list(self.forced_long_indices)
                                 if self.forced_long_indices else None),
            n_reference_samples=self.n_reference_samples,
            bucket_constraints=([dataclasses.replace(bc) for bc in self.bucket_constraints]
                                if self.bucket_constraints else None),
            vega_config=(dataclasses.replace(self.vega_config)
                         if self.vega_config is not None else None),
        )
        return optimizer.run()


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def save_run_bundle(
    path: str,
    *,
    pnl_matrix: np.ndarray,
    column_map: Dict[str, int],
    long_candidates: List[DispersionLeg],
    short_candidates: List[DispersionLeg],
    constraints: OptimizationConstraints,
    score_weights: Dict[str, float],
    seed: int,
    missing_data_policy: MissingDataPolicy,
    adj_divs: bool = False,
    reweight_grace_days: int = 0,
    active_mask: Optional[np.ndarray] = None,
    is_cross_corridor: bool = False,
    global_cap: float = 9999999.0,
    global_floor: float = -9999999.0,
    bisect_in_ga: bool = False,
    forced_long_indices: Optional[List[int]] = None,
    n_reference_samples: Optional[int] = None,
    bucket_constraints: Optional[List[BucketConstraint]] = None,
    vega_config: Optional[VegaConfig] = None,
    dates=None,
    config: Optional[Dict] = None,
    provenance: Optional[Dict] = None,
    result: Optional[OptimizationResult] = None,
) -> str:
    """Write a run bundle directory (parquet matrix + JSON). Returns ``path``.

    Raises on any inconsistency (e.g. column_map not matching the matrix) —
    a bundle that cannot replay exactly must not be written silently.
    """
    mat = np.asarray(pnl_matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"save_run_bundle: pnl_matrix must be 2-D, got shape {mat.shape}")
    n_cols = mat.shape[1]
    if sorted(column_map.values()) != list(range(n_cols)):
        raise ValueError(
            f"save_run_bundle: column_map values must be exactly 0..{n_cols - 1} "
            f"(matrix has {n_cols} columns), got {sorted(column_map.values())}"
        )

    os.makedirs(path, exist_ok=True)

    # ── Matrix → parquet, columns ordered by index so load rebuilds the map ──
    cols_by_idx: List[str] = [""] * n_cols
    for key, idx in column_map.items():
        cols_by_idx[idx] = str(key)
    index = pd.Index(dates) if dates is not None else pd.RangeIndex(mat.shape[0])
    df = pd.DataFrame(mat, columns=cols_by_idx, index=index)
    df.to_parquet(os.path.join(path, _MATRIX_FILE))
    if active_mask is not None:
        # Grace > 0: the mask was built on FULL history before window slicing —
        # not derivable from the stored window alone, so persist it for exact replay.
        pd.DataFrame(np.asarray(active_mask, dtype=bool), columns=cols_by_idx,
                     index=index).to_parquet(os.path.join(path, _MASK_FILE))

    # ── Everything else → JSON ──
    payload = {
        "bundle_version": BUNDLE_VERSION,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "seed": int(seed),
        "score_weights": {str(k): float(v) for k, v in score_weights.items()},
        "constraints": _constraints_to_dict(constraints),
        "optimizer": {
            "missing_data_policy": missing_data_policy.value,
            "adj_divs": bool(adj_divs),
            "reweight_grace_days": int(reweight_grace_days),
            "is_cross_corridor": bool(is_cross_corridor),
            "global_cap": float(global_cap),
            "global_floor": float(global_floor),
            "bisect_in_ga": bool(bisect_in_ga),
            "forced_long_indices": (sorted(int(i) for i in forced_long_indices)
                                    if forced_long_indices else None),
            "n_reference_samples": (int(n_reference_samples)
                                    if n_reference_samples is not None else None),
            "bucket_constraints": ([dataclasses.asdict(bc) for bc in bucket_constraints]
                                   if bucket_constraints else None),
            "vega_config": (dataclasses.asdict(vega_config)
                            if vega_config is not None else None),
        },
        "long_candidates": [_leg_to_dict(l) for l in long_candidates],
        "short_candidates": [_leg_to_dict(l) for l in short_candidates],
        "config": config,
        "provenance": dict(provenance or {}),
        "result": _result_to_dict(result) if result is not None else None,
        "meta": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    with open(os.path.join(path, _JSON_FILE), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)
    return path


def load_run_bundle(path: str) -> RunBundle:
    """Load a bundle directory written by :func:`save_run_bundle`."""
    json_path = os.path.join(path, _JSON_FILE)
    mat_path = os.path.join(path, _MATRIX_FILE)
    if not os.path.isfile(json_path) or not os.path.isfile(mat_path):
        raise FileNotFoundError(
            f"load_run_bundle: '{path}' is not a bundle directory "
            f"(expected {_JSON_FILE} + {_MATRIX_FILE})"
        )
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    version = payload.get("bundle_version")
    if version not in (1, BUNDLE_VERSION):
        raise ValueError(
            f"load_run_bundle: unsupported bundle_version={version!r} "
            f"(this engine reads versions 1 and {BUNDLE_VERSION})"
        )

    df = pd.read_parquet(mat_path)
    pnl_matrix = df.to_numpy(dtype=np.float64)
    mask_path = os.path.join(path, _MASK_FILE)
    active_mask = (pd.read_parquet(mask_path).to_numpy(dtype=bool)
                   if os.path.exists(mask_path) else None)
    column_map = {str(c): i for i, c in enumerate(df.columns)}
    dates = None if isinstance(df.index, pd.RangeIndex) else list(df.index)

    opt = payload["optimizer"]
    return RunBundle(
        pnl_matrix=pnl_matrix,
        active_mask=active_mask,
        column_map=column_map,
        long_candidates=[_leg_from_dict(d) for d in payload["long_candidates"]],
        short_candidates=[_leg_from_dict(d) for d in payload["short_candidates"]],
        constraints=_constraints_from_dict(payload["constraints"]),
        score_weights=dict(payload["score_weights"]),
        seed=int(payload["seed"]),
        missing_data_policy=MissingDataPolicy(opt["missing_data_policy"]),
        adj_divs=bool(opt.get("adj_divs", False)),
        # v1 bundles: grace was an inert knob when they were written — force 0
        # so replay reproduces their true behaviour. v2+: honour the stored value.
        reweight_grace_days=(int(opt.get("reweight_grace_days", 0)) if version >= 2 else 0),
        is_cross_corridor=bool(opt.get("is_cross_corridor", False)),
        global_cap=float(opt.get("global_cap", 9999999.0)),
        global_floor=float(opt.get("global_floor", -9999999.0)),
        bisect_in_ga=bool(opt.get("bisect_in_ga", False)),
        forced_long_indices=opt.get("forced_long_indices"),
        n_reference_samples=opt.get("n_reference_samples"),
        bucket_constraints=([BucketConstraint(**d) for d in opt["bucket_constraints"]]
                            if opt.get("bucket_constraints") else None),
        vega_config=(VegaConfig(**opt["vega_config"])
                     if opt.get("vega_config") else None),
        dates=dates,
        config=payload.get("config"),
        provenance=dict(payload.get("provenance") or {}),
        result=payload.get("result"),
        meta=dict(payload.get("meta") or {}),
    )
