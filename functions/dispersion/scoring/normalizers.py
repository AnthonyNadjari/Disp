"""
normalizers.py
==============

Normalizer protocol and three concrete implementations used by the scoring
pipeline to map raw metric values onto a common [0, 1] scale where **1 is
always best**.

Public surface
--------------
- :class:`Normalizer`         – structural Protocol (type-checking only)
- :class:`QuantileNormalizer` – empirical CDF / rank-based  (DEFAULT)
- :class:`ZScoreNormalizer`   – Gaussian CDF
- :class:`MinMaxNormalizer`   – linear rescaling

Design notes
------------
* ``fit`` must be called before any ``transform*`` call.
* ``transform`` methods clamp output to [0, 1].  QuantileNormalizer's
  ``transform_smooth`` linearly extrapolates beyond the reference support to
  preserve gradient signal, clamped to [-0.05, 1.05].
* ``transform`` is intended for final scoring (can be step-wise / piecewise).
* ``transform_smooth`` is intended for scipy-based weight optimisation and must
  be continuously differentiable wherever the optimiser queries it.
"""

from __future__ import annotations

import bisect
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.stats import norm as _norm  # used only by ZScoreNormalizer


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Normalizer(Protocol):
    """Structural protocol for all normalizer implementations.

    A ``Normalizer`` converts raw metric values into a *goodness* score in
    ``[0, 1]`` where **1 = best possible** within the reference sample.

    Two transform variants are provided so that the same normalizer can serve
    both human-readable final scores (``transform``) and differentiable
    optimisation landscapes (``transform_smooth``).
    """

    def fit(self, raw_by_metric: dict[str, np.ndarray]) -> None:
        """Store reference distributions for every metric.

        Parameters
        ----------
        raw_by_metric:
            Mapping of *metric name* → 1-D array of raw reference values
            (e.g. drawn from the historical universe of candidates).
            All arrays must be finite and non-empty.
        """
        ...

    def transform(self, name: str, raw: float, higher_is_better: bool) -> float:
        """Map a single raw observation to a goodness score in ``[0, 1]``.

        The method is guaranteed to be **monotone** with respect to
        ``higher_is_better``:

        * ``higher_is_better=True``  → score increases as *raw* increases.
        * ``higher_is_better=False`` → score increases as *raw* decreases.

        Parameters
        ----------
        name:
            Metric identifier; must have been seen during :meth:`fit`.
        raw:
            Observed metric value to score.
        higher_is_better:
            Direction of the metric.

        Returns
        -------
        float
            Score in ``[0, 1]``.
        """
        ...

    def transform_smooth(
        self, name: str, raw: float, higher_is_better: bool
    ) -> float:
        """Differentiable variant of :meth:`transform`.

        Produces the same ordinal ranking as :meth:`transform` but avoids
        discrete jumps so that automatic-differentiation / finite-difference
        gradient estimators used by ``scipy.optimize`` receive a smooth loss
        surface.

        Parameters
        ----------
        name:
            Metric identifier; must have been seen during :meth:`fit`.
        raw:
            Observed metric value to score.
        higher_is_better:
            Direction of the metric.

        Returns
        -------
        float
            Smooth score in ``[0, 1]``.
        """
        ...


# ---------------------------------------------------------------------------
# QuantileNormalizer (DEFAULT)
# ---------------------------------------------------------------------------


class QuantileNormalizer:
    """Empirical-CDF normalizer — the recommended default.

    Scoring is based on the *rank* of the observation within a reference
    sample, which makes it robust to outliers and scale-free.

    ``transform``
        Uses binary search on the sorted reference array to count how many
        reference values are beaten.  The result is therefore a *step
        function* with resolution ``1 / n_reference``.

    ``transform_smooth``
        Linearly interpolates between consecutive order statistics to produce
        a piecewise-linear CDF that is differentiable almost everywhere
        (discontinuity only at the exact sample points, which have measure
        zero).  Suitable for finite-difference gradient estimation used by
        ``scipy.optimize``.

    Both variants are strictly monotone on the support of the reference
    sample.  ``transform`` clamps to ``[0, 1]`` outside it, while
    ``transform_smooth`` linearly extrapolates (clamped to
    ``[-0.05, 1.05]``) to preserve gradient signal.

    Examples
    --------
    >>> import numpy as np
    >>> norm = QuantileNormalizer()
    >>> rng = np.random.default_rng(0)
    >>> norm.fit({"vol": rng.normal(0.15, 0.05, 1000)})
    >>> norm.transform("vol", 0.10, higher_is_better=False)  # low vol is good
    0.841...
    """

    def __init__(self) -> None:
        # metric_name → sorted 1-D ndarray of reference values
        self._sorted: dict[str, np.ndarray] = {}
        # metric_name → cached empirical-CDF grid linspace(0, 1, n)
        self._cdf_grid: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, raw_by_metric: dict[str, np.ndarray]) -> None:
        """Sort and store one reference array per metric.

        Parameters
        ----------
        raw_by_metric:
            Mapping of metric name → 1-D array of raw reference values.
            NaN / inf values are silently dropped before sorting.

        Raises
        ------
        ValueError
            If the finite sample is empty or contains fewer than 2
            *distinct* values (the empirical-CDF grid would be degenerate).
        """
        self._sorted.clear()
        self._cdf_grid.clear()
        for name, arr in raw_by_metric.items():
            a = np.asarray(arr, dtype=float).ravel()
            a = a[np.isfinite(a)]
            if a.size == 0:
                raise ValueError(
                    f"QuantileNormalizer.fit: metric '{name}' produced an "
                    "empty finite array."
                )
            n_distinct = int(np.unique(a).size)
            if n_distinct < 2:
                raise ValueError(
                    f"QuantileNormalizer.fit: metric '{name}' needs at least 2 "
                    f"distinct reference values, got {n_distinct}."
                )
            self._sorted[name] = np.sort(a)
            self._cdf_grid[name] = np.linspace(0.0, 1.0, a.size)

    # ------------------------------------------------------------------
    # transform  (step-function empirical CDF)
    # ------------------------------------------------------------------

    def transform(self, name: str, raw: float, higher_is_better: bool) -> float:
        """Score via empirical CDF (binary-search rank).

        The fraction of reference values *beaten* by *raw*:

        * ``higher_is_better=True``  → ``fraction of ref values < raw``
        * ``higher_is_better=False`` → ``fraction of ref values > raw``

        Clamped to ``[0, 1]``.  Values above the reference maximum map to
        ``1.0``; values below the minimum map to ``0.0``.

        Parameters
        ----------
        name:
            Metric name (must be present in the fitted data).
        raw:
            Raw observed value.
        higher_is_better:
            Monotonicity direction.

        Returns
        -------
        float
            Goodness score in ``[0, 1]``.
        """
        self._check_fitted(name)
        arr = self._sorted[name]
        n = len(arr)

        if higher_is_better:
            # Number of reference values strictly less than raw
            rank = bisect.bisect_left(arr, raw)
        else:
            # Number of reference values strictly greater than raw
            rank = n - bisect.bisect_right(arr, raw)

        score = rank / n
        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # transform_smooth  (piecewise-linear interpolated CDF)
    # ------------------------------------------------------------------

    def transform_smooth(
        self, name: str, raw: float, higher_is_better: bool
    ) -> float:
        """Score via linearly-interpolated empirical CDF.

        Between consecutive order statistics ``x[i]`` and ``x[i+1]`` the CDF
        is linearly interpolated, producing a piecewise-linear, differentiable
        (almost everywhere) function.  Outside the reference support the CDF
        is linearly extrapolated to preserve gradient signal:

        * ``raw < x[0]``   → extrapolated below 0.0
        * ``raw > x[-1]``  → extrapolated above 1.0

        The extrapolated value is clamped to ``[-0.05, 1.05]`` so that a
        wildly out-of-support raw value cannot dominate the aggregate
        (interior interpolation is untouched and stays in ``[0, 1]``).

        Parameters
        ----------
        name:
            Metric name (must be present in the fitted data).
        raw:
            Raw observed value.
        higher_is_better:
            Monotonicity direction.

        Returns
        -------
        float
            Smooth goodness score in ``[0, 1]``.
        """
        self._check_fitted(name)
        arr = self._sorted[name]
        n = len(arr)

        # Use numpy's linear interpolation on the empirical CDF grid.
        # xp: order statistics  yp: CDF values at those points [0/(n-1) … 1]
        xp = arr
        yp = self._cdf_grid[name]

        # Linear extrapolation beyond reference range to preserve gradient.
        # Boundary segments may have zero width when the reference sample
        # contains ties; fall back to the first non-zero-width segment slope
        # (fit guarantees at least 2 distinct values, so one always exists).
        if n >= 2:
            if raw < arr[0]:
                # Below min: use slope from first non-zero-width segment
                i = 0
                while arr[i + 1] == arr[i]:
                    i += 1
                slope = (yp[i + 1] - yp[i]) / (arr[i + 1] - arr[i])
                cdf_value = float(
                    np.clip(yp[0] + (raw - arr[0]) * slope, -0.05, 1.05)
                )
            elif raw > arr[-1]:
                # Above max: use slope from last non-zero-width segment
                j = n - 2
                while arr[j] == arr[j + 1]:
                    j -= 1
                slope = (yp[j + 1] - yp[j]) / (arr[j + 1] - arr[j])
                cdf_value = float(
                    np.clip(yp[-1] + (raw - arr[-1]) * slope, -0.05, 1.05)
                )
            else:
                # Inside range: use linear interpolation
                cdf_value = float(np.interp(raw, xp, yp))
        else:
            # Edge case: single reference value
            cdf_value = 0.0 if raw < arr[0] else 1.0

        if not higher_is_better:
            cdf_value = 1.0 - cdf_value

        # Extrapolated values are already clamped to [-0.05, 1.05]; interior
        # interpolation stays within [0, 1] — no final clip.
        return float(cdf_value)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self, name: str) -> None:
        if not self._sorted:
            raise RuntimeError(
                "QuantileNormalizer has not been fitted yet. Call fit() first."
            )
        if name not in self._sorted:
            raise KeyError(
                f"QuantileNormalizer: metric '{name}' was not seen during fit. "
                f"Available metrics: {sorted(self._sorted)}"
            )

    def __repr__(self) -> str:  # pragma: no cover
        n = len(self._sorted)
        fitted = f"fitted on {n} metric(s)" if n else "not fitted"
        return f"QuantileNormalizer({fitted})"


# ---------------------------------------------------------------------------
# ZScoreNormalizer
# ---------------------------------------------------------------------------


class ZScoreNormalizer:
    """Gaussian-CDF normalizer.

    Assumes each metric is approximately normally distributed in the reference
    sample.  The raw value is standardised and then passed through the normal
    CDF ``Φ``, which maps ℝ → (0, 1).

    ``transform`` and ``transform_smooth`` are identical because the Gaussian
    CDF is already infinitely differentiable.

    Examples
    --------
    >>> import numpy as np
    >>> norm = ZScoreNormalizer()
    >>> rng = np.random.default_rng(42)
    >>> norm.fit({"ret": rng.normal(0.08, 0.02, 500)})
    >>> norm.transform("ret", 0.08, higher_is_better=True)  # exactly mean → 0.5
    0.5
    """

    def __init__(self) -> None:
        self._mean: dict[str, float] = {}
        self._std: dict[str, float] = {}

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, raw_by_metric: dict[str, np.ndarray]) -> None:
        """Compute and store mean and standard deviation per metric.

        Parameters
        ----------
        raw_by_metric:
            Mapping of metric name → 1-D array of raw reference values.
            NaN / inf values are silently dropped before statistics are
            computed.

        Raises
        ------
        ValueError
            If the finite sample has fewer than 2 observations (std undefined)
            or all values are identical (std = 0).
        """
        self._mean.clear()
        self._std.clear()
        for name, arr in raw_by_metric.items():
            a = np.asarray(arr, dtype=float).ravel()
            a = a[np.isfinite(a)]
            if a.size < 2:
                raise ValueError(
                    f"ZScoreNormalizer.fit: metric '{name}' needs at least 2 "
                    f"finite values; got {a.size}."
                )
            std = float(np.std(a, ddof=1))
            if std == 0.0:
                raise ValueError(
                    f"ZScoreNormalizer.fit: metric '{name}' has zero variance; "
                    "z-score normalisation is undefined."
                )
            self._mean[name] = float(np.mean(a))
            self._std[name] = std

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------

    def transform(self, name: str, raw: float, higher_is_better: bool) -> float:
        """Map raw value to ``Φ(z)`` goodness score.

        Computes ``z = (raw - mean) / std`` then applies the standard-normal
        CDF.  If ``higher_is_better=False`` the score is flipped via
        ``1 - Φ(z)``.

        Parameters
        ----------
        name:
            Metric name (must be present in the fitted data).
        raw:
            Raw observed value.
        higher_is_better:
            Monotonicity direction.

        Returns
        -------
        float
            Goodness score in ``[0, 1]``.
        """
        self._check_fitted(name)
        z = (raw - self._mean[name]) / self._std[name]
        score = float(_norm.cdf(z))
        if not higher_is_better:
            score = 1.0 - score
        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # transform_smooth  (identical — Gaussian CDF is C-infinity)
    # ------------------------------------------------------------------

    def transform_smooth(
        self, name: str, raw: float, higher_is_better: bool
    ) -> float:
        """Smooth score — identical to :meth:`transform` for ``ZScoreNormalizer``.

        The Gaussian CDF is already infinitely differentiable, so no
        additional smoothing is required.

        Parameters
        ----------
        name:
            Metric name (must be present in the fitted data).
        raw:
            Raw observed value.
        higher_is_better:
            Monotonicity direction.

        Returns
        -------
        float
            Smooth goodness score in ``[0, 1]``.
        """
        return self.transform(name, raw, higher_is_better)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self, name: str) -> None:
        if not self._mean:
            raise RuntimeError(
                "ZScoreNormalizer has not been fitted yet. Call fit() first."
            )
        if name not in self._mean:
            raise KeyError(
                f"ZScoreNormalizer: metric '{name}' was not seen during fit. "
                f"Available metrics: {sorted(self._mean)}"
            )

    def __repr__(self) -> str:  # pragma: no cover
        n = len(self._mean)
        fitted = f"fitted on {n} metric(s)" if n else "not fitted"
        return f"ZScoreNormalizer({fitted})"


# ---------------------------------------------------------------------------
# MinMaxNormalizer
# ---------------------------------------------------------------------------


class MinMaxNormalizer:
    """Linear min–max rescaling normalizer.

    Maps raw values linearly onto ``[0, 1]`` using the minimum and maximum
    of the reference sample.  Values outside the training range are clamped.

    ``transform`` and ``transform_smooth`` are identical because the linear
    mapping is already smooth.

    .. warning::
        Sensitive to outliers in the reference sample because a single extreme
        value can dominate the range.  Consider using :class:`QuantileNormalizer`
        when outliers are a concern.

    Examples
    --------
    >>> import numpy as np
    >>> norm = MinMaxNormalizer()
    >>> norm.fit({"price": np.array([10.0, 20.0, 30.0, 40.0, 50.0])})
    >>> norm.transform("price", 30.0, higher_is_better=True)
    0.5
    >>> norm.transform("price", 30.0, higher_is_better=False)
    0.5
    """

    def __init__(self) -> None:
        self._min: dict[str, float] = {}
        self._max: dict[str, float] = {}

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, raw_by_metric: dict[str, np.ndarray]) -> None:
        """Compute and store per-metric minimum and maximum.

        Parameters
        ----------
        raw_by_metric:
            Mapping of metric name → 1-D array of raw reference values.
            NaN / inf values are silently dropped.

        Raises
        ------
        ValueError
            If all values for a metric are identical (range = 0) or the
            finite sample is empty.
        """
        self._min.clear()
        self._max.clear()
        for name, arr in raw_by_metric.items():
            a = np.asarray(arr, dtype=float).ravel()
            a = a[np.isfinite(a)]
            if a.size == 0:
                raise ValueError(
                    f"MinMaxNormalizer.fit: metric '{name}' produced an empty "
                    "finite array."
                )
            lo, hi = float(np.min(a)), float(np.max(a))
            if lo == hi:
                raise ValueError(
                    f"MinMaxNormalizer.fit: metric '{name}' has zero range "
                    f"(all values = {lo}); min–max normalisation is undefined."
                )
            self._min[name] = lo
            self._max[name] = hi

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------

    def transform(self, name: str, raw: float, higher_is_better: bool) -> float:
        """Map raw value linearly to ``[0, 1]``.

        Applies ``(raw - min) / (max - min)`` and flips if
        ``higher_is_better=False``.  Values outside ``[min, max]`` are
        clamped to ``[0, 1]``.

        Parameters
        ----------
        name:
            Metric name (must be present in the fitted data).
        raw:
            Raw observed value.
        higher_is_better:
            Monotonicity direction.

        Returns
        -------
        float
            Goodness score in ``[0, 1]``.
        """
        self._check_fitted(name)
        lo = self._min[name]
        hi = self._max[name]
        score = (raw - lo) / (hi - lo)
        if not higher_is_better:
            score = 1.0 - score
        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # transform_smooth  (identical — linear map is C-infinity)
    # ------------------------------------------------------------------

    def transform_smooth(
        self, name: str, raw: float, higher_is_better: bool
    ) -> float:
        """Smooth score — identical to :meth:`transform` for ``MinMaxNormalizer``.

        The linear map is already infinitely differentiable, so no additional
        smoothing is required.

        Parameters
        ----------
        name:
            Metric name (must be present in the fitted data).
        raw:
            Raw observed value.
        higher_is_better:
            Monotonicity direction.

        Returns
        -------
        float
            Smooth goodness score in ``[0, 1]``.
        """
        return self.transform(name, raw, higher_is_better)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self, name: str) -> None:
        if not self._min:
            raise RuntimeError(
                "MinMaxNormalizer has not been fitted yet. Call fit() first."
            )
        if name not in self._min:
            raise KeyError(
                f"MinMaxNormalizer: metric '{name}' was not seen during fit. "
                f"Available metrics: {sorted(self._min)}"
            )

    def __repr__(self) -> str:  # pragma: no cover
        n = len(self._min)
        fitted = f"fitted on {n} metric(s)" if n else "not fitted"
        return f"MinMaxNormalizer({fitted})"
