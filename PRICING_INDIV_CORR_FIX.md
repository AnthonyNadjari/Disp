# PRICING_INDIV_CORR_FIX — "ev_mono_lsv_values referenced before assignment" in Individual Correlations mode

## Problem

Running `solve()` with `correl_input_method="Individual Correlations"` crashes with:

```
UnboundLocalError: local variable 'ev_mono_lsv_values' referenced before assignment
```

Root cause: the per-ticker correlation branch initializes the cross LSV/LCM lists
(`ev_cross_lsv_values`, `ev_cross_lsv_zero_values`, `ev_cross_lcm_values`) but never
initializes the **mono** LSV lists — while the shared post-processing loop reads
`ev_mono_lsv_values[m_idx]` (and `ev_mono_lsv_zero_values[m_idx]`) unconditionally.

## File

`functions/dispersion/_pricing.py` (or `_pricing.py`), in the per-ticker correlation
branch (the `if` block that ends with `_per_ticker_corr_mode = True`, right after the
`ev_cross_lsv_values = [None] * len(tickers)` lines).

## Replace

```python
            ev_cross_values = all_ev_cross
            ev_cross_lsv_values = [None] * len(tickers)  # LSV not supported with per-ticker corr
            ev_cross_lsv_zero_values = [None] * len(tickers)
            ev_cross_lcm_values = [None] * len(tickers)  # LCM not supported with per-ticker corr
```

## By

```python
            ev_cross_values = all_ev_cross
            ev_cross_lsv_values = [None] * len(tickers)  # LSV not supported with per-ticker corr
            ev_cross_lsv_zero_values = [None] * len(tickers)
            ev_cross_lcm_values = [None] * len(tickers)  # LCM not supported with per-ticker corr
            # Mono LSV lists: also not supported with per-ticker corr, but the
            # post-processing loop reads them unconditionally — initialize to None
            # (sized to the MONO universe, indexed by m_idx, not len(tickers)).
            ev_mono_lsv_values = [None] * len(mono_corr_order)
            ev_mono_lsv_zero_values = [None] * len(mono_corr_order)
```

## Note — expected behavior after the fix

Individual Correlations mode does **not** compute LSV/LCM adjustments (by design —
the per-ticker path has no bump scenarios): the `Strike Cross Corr LSV/LCM (%)` and
`EV Mono LSV (%)` columns will simply be absent from the results. If you need LSV/LCM
strikes, run `Global Parameters` mode (or with individual per-name `Correlation`
column values only where you want to override the model).

## Verify

1. `solve(..., correl_input_method="Individual Correlations")` with a `Correlation`
   column → completes, no `UnboundLocalError`.
2. Same call with `use_lsv=True` → completes; LSV columns absent (not an error).
3. `Global Parameters` mode with LSV → unchanged (LSV columns present as before).
