"""_logging.py — single leveled logger for the dispersion engine.

Silent by default (no handler, propagates to root at WARNING).  Activate
with::

    from functions.dispersion._logging import configure_engine_logging
    configure_engine_logging("DEBUG")      # or "INFO", "WARNING", ...

or headless via ``optimize(..., log_level="DEBUG")``.
"""

from __future__ import annotations

import logging
from typing import Optional

#: The engine-wide logger — every module logs through (a child of) this.
logger = logging.getLogger("functions.dispersion")


def configure_engine_logging(level: Optional[str]) -> None:
    """Attach a stderr handler at ``level`` (idempotent). None = leave silent."""
    if not level:
        return
    lvl = getattr(logging, str(level).upper(), None)
    if lvl is None:
        raise ValueError(
            f"Unknown log level {level!r} — use DEBUG/INFO/WARNING/ERROR.")
    if not any(getattr(h, "_dispersion_engine", False) for h in logger.handlers):
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        h._dispersion_engine = True
        logger.addHandler(h)
    for h in logger.handlers:
        if getattr(h, "_dispersion_engine", False):
            h.setLevel(lvl)
    logger.setLevel(lvl)
