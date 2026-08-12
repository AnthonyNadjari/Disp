"""Golden regression gate: frozen run bundles must replay to frozen outputs.

MANDATORY GATE for every phase of the project (see tests/golden/make_goldens.py
for the freeze/re-baseline policy):

- ``bundle_a_linear`` and ``bundle_b_minpayoff`` are EXACT-PATH goldens —
  untouchable for the whole project.
- ``bundle_c_mixed`` (SLSQP) may be re-baselined ONCE, in Phase 3 only,
  with a numbered justification.

Tolerance 1e-6 absorbs numerical drift across scipy/HiGHS/numba versions;
same-machine replays are expected to be bit-identical.
"""

from __future__ import annotations

import json
import os

import pytest

from functions.dispersion.run_bundle import load_run_bundle

_GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
_TOL = 1e-6

with open(os.path.join(_GOLDEN_DIR, "expected.json"), "r", encoding="utf-8") as _f:
    _EXPECTED = json.load(_f)

_CASES = sorted(_EXPECTED.keys())


@pytest.mark.parametrize("case", _CASES)
def test_golden_replay_matches_expected(case):
    bundle = load_run_bundle(os.path.join(_GOLDEN_DIR, case))
    result = bundle.replay()
    exp = _EXPECTED[case]

    got_keys = [k for k, _ in result.long_basket]
    exp_keys = [k for k, _ in exp["long_basket"]]
    assert got_keys == exp_keys, (
        f"[{case}] basket changed: got {got_keys}, expected {exp_keys}"
    )

    got_w = {k: w for k, w in result.long_basket}
    for k, w_exp in exp["long_basket"]:
        assert abs(got_w[k] - w_exp) <= _TOL, (
            f"[{case}] weight drift on {k}: got {got_w[k]:.10f}, "
            f"expected {w_exp:.10f} (tol {_TOL})"
        )

    assert abs(result.score - exp["score"]) <= _TOL, (
        f"[{case}] score drift: got {result.score:.10f}, "
        f"expected {exp['score']:.10f} (tol {_TOL})"
    )
    assert abs(result.net_strike - exp["net_strike"]) <= _TOL, (
        f"[{case}] net_strike drift: got {result.net_strike:.10f}, "
        f"expected {exp['net_strike']:.10f} (tol {_TOL})"
    )
    assert result.scoring_signature == exp["scoring_signature"], (
        f"[{case}] scoring signature changed: got {result.scoring_signature}, "
        f"expected {exp['scoring_signature']} — the scoring inputs "
        f"(metrics/weights/seed/reference geometry) are no longer identical"
    )
