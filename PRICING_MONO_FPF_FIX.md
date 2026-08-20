# PRICING_MONO_FPF_FIX — mono-corridor results show wrong/empty FPF columns

## Problem

In **mono-corridor** solves, the results DataFrame shows misleading FPF columns:

- `FPF Cross LSV Uncapped` appears **empty** (the column is created unconditionally,
  even when there is no LSV FPF — unlike the cross branch which only adds non-empty columns)
- the capped FPF lands under **`FPF Cross Cap`** — but in mono mode there is no "cross"
  structure at all (ticker == corridor asset); the Cross/Mono split is meaningless.

Root cause: the mono branch of `_build_results_df` writes all nine `FPF Cross/Mono ×
Uncapped/Cap` columns unconditionally, reusing the cross-corridor labels.

## File

`functions/dispersion/_pricing.py` (or `_pricing.py`), method **`_build_results_df`**,
in the **mono branch** (the `else:` after `if cfg.is_cross_corridor and r.ticker != r.corridor_asset:`).

## Replace

```python
                    # ── FPF columns (Cross/Mono × Uncapped/Cap) ──
                    row['FPF Cross Uncapped'] = r.fpf_string_cross if r.fpf_string_cross else ''
                    row['FPF Cross LSV Uncapped'] = r.fpf_string_lsv if r.fpf_string_lsv else ''
                    row['FPF Cross LCM Uncapped'] = r.fpf_string_lcm if r.fpf_string_lcm else ''
                    row['FPF Cross Cap'] = r.fpf_string_cap_lv if r.fpf_string_cap_lv else ''
                    row['FPF Cross LSV Cap'] = r.fpf_string_cap_lsv if r.fpf_string_cap_lsv else ''
                    row['FPF Cross LCM Cap'] = r.fpf_string_cap_lcm if r.fpf_string_cap_lcm else ''
                    row['FPF Mono Uncapped'] = r.fpf_string_mono if r.fpf_string_mono else ''
                    row['FPF Mono Cap'] = r.fpf_string_cap_mono if r.fpf_string_cap_mono else ''
                    row['FPF Mono LSV Cap'] = r.fpf_string_cap_lsv_mono if r.fpf_string_cap_lsv_mono else ''
```

## By

```python
                    # ── FPF columns (mono mode: ONE structure — no Cross/Mono split;
                    #    only non-empty columns are shown, matching the cross branch) ──
                    for _fpf_col, _fpf_val in [
                            ('FPF LV Uncapped',  r.fpf_string_mono or r.fpf_string_cross),
                            ('FPF LSV Uncapped', r.fpf_string_lsv),
                            ('FPF LCM Uncapped', r.fpf_string_lcm),
                            ('FPF LV Cap',       r.fpf_string_cap_mono or r.fpf_string_cap_lv),
                            ('FPF LSV Cap',      r.fpf_string_cap_lsv_mono or r.fpf_string_cap_lsv),
                            ('FPF LCM Cap',      r.fpf_string_cap_lcm)]:
                        if _fpf_val:
                            row[_fpf_col] = _fpf_val
```

Notes on the fallback chain (`or`): in a mono run the "cross" and "mono" FPFs are the
same instrument (ticker == corridor asset), so if the mono attribute is unset the cross
attribute is the correct value — but the displayed label is now the honest mono one.

## Verify

1. Mono solve, LV only, `is_capped=True`: results show `FPF LV Uncapped` and
   `FPF LV Cap` — and **no** empty `FPF Cross LSV Uncapped` column.
2. Mono solve with LSV: `FPF LSV Uncapped` appears (filled), plus capped variants.
3. Cross-corridor solve: unchanged (that branch already only shows non-empty columns).
