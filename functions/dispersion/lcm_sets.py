"""
Dispersion re-export of the LCM parameter-set policy.

The implementation is product-agnostic and lives in
``functions.common.lcm_sets`` so other products can reuse it; this module
keeps the historical import path for the dispersion engine, the page and
the tests.
"""
from functions.common.lcm_sets import (  # noqa: F401
    LcmSetInput,
    lcm0_lambda_for,
    lcm_column_suffix,
    resolve_lcm_sets,
)

__all__ = ["LcmSetInput", "lcm0_lambda_for", "lcm_column_suffix", "resolve_lcm_sets"]
