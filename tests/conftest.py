"""Pytest bootstrap: make the repo root importable regardless of invocation cwd.

The engine is imported as ``functions.dispersion...`` from the repo root; this
keeps ``pytest tests/`` working from any directory.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
