# PRICING_INDIV_LSV_LCM — enable LSV/LCM in Individual Correlations mode

## Goal

`solve(..., correl_input_method="Individual Correlations", use_lsv=True, use_lcm=True)`
currently produces **no** LSV/LCM columns (the per-ticker branch hardcodes the LSV/LCM
lists to `None`). This patch prices the LSV/LSV0/LCM variants **under each group's
forced correlation**, so the columns appear with exact values.

## How it works (design)

Per-ticker mode groups tickers by correlation value and prices each group with a
`GenericMutatorOverrideCorrelationEqEq` scenario. The unified (Global) path instead
prices everything under ONE `build_unified_scenario` whose bumps (`LV`, `LSV0`,
`LSV`, `LCM`) include LSV/LCM — and that helper already accepts a correlation
perturbation (`correl_bump` / `correl_bump_style`).

So per correlation group we build **one unified scenario with the group's correlation
level forced** and read the bump values at the same positions as today's LV batch.
Same number of batch calls as today (1 per group), more instruments variants
extracted from the same response.

> **ONE THING TO CHECK FIRST** (30 seconds): in
> `functions/common/pricing_scenarios.py`, confirm `build_unified_scenario`'s
> `correl_bump` accepts an **absolute level** with `correl_bump_style="Absolute"`
> (the UI exposes Relative/Absolute styles). If it only supports relative bumps, use
> the fallback in §4.

---

## 1. Per-group scenario with forced correlation + bumps

**File:** `functions/dispersion/_pricing.py`, per-ticker branch, in the
`for corr_val, ticker_indices in corr_groups.items():` loop.

### Replace

```python
                # Create scenario for this correlation level
                group_scenario = None
                if corr_val is not None:
                    group_scenario = pricing_portal.create_scenario_simple(
                        mutator_name="GenericMutatorOverrideCorrelationEqEq",
                        properties={"CorrelationLevel": corr_val}
                    )
```

### By

```python
                # Create scenario for this correlation level. With LSV/LCM requested,
                # build a UNIFIED bump scenario (LV/LSV0/LSV/LCM) with the group's
                # correlation forced — bumps are then read from the same response.
                group_scenario = None
                _use_lsv_pt = lsv_scenario is not None
                _use_lcm_pt = cfg.lcm_params is not None and cfg.lcm_params.get('enabled', False)
                if _use_lsv_pt or _use_lcm_pt:
                    from functions.common.pricing_scenarios import build_unified_scenario
                    _grp_rics = sorted({tickers[i] for i in ticker_indices}
                                       | {corr_assets[i] for i in ticker_indices})
                    _grp_lsv_df = cfg.lsv_params if isinstance(cfg.lsv_params, pd.DataFrame) \
                        else pd.DataFrame(cfg.lsv_params)
                    if 'RIC' in _grp_lsv_df.columns and _grp_lsv_df.index.dtype != object:
                        _grp_lsv_df = _grp_lsv_df.set_index('RIC')
                    group_scenario = build_unified_scenario(
                        pricing_portal,
                        use_lsv=_use_lsv_pt,
                        use_lcm=_use_lcm_pt,
                        underlying_rics=_grp_rics,
                        lsv_params=_grp_lsv_df,
                        correl_bump=(corr_val if corr_val is not None else 0.0),
                        correl_bump_style="Absolute",   # force the LEVEL, not a relative bump
                        lcm_properties=cfg.lcm_params.get('lcm_properties') if cfg.lcm_params else None,
                    )
                    _grp_bump_names = (["LV", "LSV0", "LSV"] if _use_lsv_pt else ["LV"])
                    if _use_lcm_pt:
                        _grp_bump_names.append("LCM")
                elif corr_val is not None:
                    group_scenario = pricing_portal.create_scenario_simple(
                        mutator_name="GenericMutatorOverrideCorrelationEqEq",
                        properties={"CorrelationLevel": corr_val}
                    )
```

## 2. Extract the bump values (cross + mono)

Right after the existing group extraction (`all_ev_cross[global_i] = ev_val` /
`all_ra[global_i] = ra_val` end of loop) and **before** the mono batch, initialize
and fill the LSV/LCM lists from the SAME group responses.

### Replace

```python
            # Override the extraction functions for per-ticker correlation mode
            ev_cross_values = all_ev_cross
            ev_cross_lsv_values = [None] * len(tickers)  # LSV not supported with per-ticker corr
            ev_cross_lsv_zero_values = [None] * len(tickers)
            ev_cross_lcm_values = [None] * len(tickers)  # LCM not supported with per-ticker corr
```

### By

```python
            # Override the extraction functions for per-ticker correlation mode
            ev_cross_values = all_ev_cross
            ev_cross_lsv_values = [None] * len(tickers)
            ev_cross_lsv_zero_values = [None] * len(tickers)
            ev_cross_lcm_values = [None] * len(tickers)
            ev_mono_lsv_values = [None] * len(mono_corr_order)
            ev_mono_lsv_zero_values = [None] * len(mono_corr_order)

            # LSV/LCM bumps: read per-bump values from each group's response at the
            # same instrument positions as the LV extraction above. Requires the
            # unified scenario of §1 (otherwise the lists stay None and the LSV/LCM
            # columns are simply absent, as before).
            if (_use_lsv_pt or _use_lcm_pt) and 'group_res' in dir():
                def _pt_bump_fv(res, pos, bump):
                    running = 0
                    for cs in sorted(res.keys()):
                        cd = res[cs]
                        if pos < running + cd["chunk_size"]:
                            local_idx = pos - running
                            raw = cd["raw"]
                            key = "Price" if local_idx == 0 else f"Price_{local_idx}"
                            if key in raw and isinstance(raw[key], dict):
                                bump_data = raw[key].get(bump, [])
                                if isinstance(bump_data, list) and bump_data:
                                    v = bump_data[0].get("value")
                                    return v if isinstance(v, (int, float)) else None
                            return None
                        running += cd["chunk_size"]
                    return None

                # NOTE: re-run the group loop storing each group's response and
                # indices, e.g. before the loop add `group_results = []` and inside
                # append `(ticker_indices, n_group, group_res)`. Then:
                for ticker_indices_b, n_group_b, group_res_b in group_results:
                    for local_i, global_i in enumerate(ticker_indices_b):
                        if _use_lsv_pt:
                            ev_cross_lsv_zero_values[global_i] = _pt_bump_fv(
                                group_res_b, local_i, "LSV0")
                            ev_cross_lsv_values[global_i] = _pt_bump_fv(
                                group_res_b, local_i, "LSV")
                        if _use_lcm_pt:
                            ev_cross_lcm_values[global_i] = _pt_bump_fv(
                                group_res_b, local_i, "LCM")

            # Mono LSV: second mono batch under a unified LSV scenario (mono assets
            # have no correlation override — the corridor asset IS the asset).
            if _use_lsv_pt:
                from functions.common.pricing_scenarios import build_unified_scenario
                _mono_lsv_scenario = build_unified_scenario(
                    pricing_portal,
                    use_lsv=True, use_lcm=False,
                    underlying_rics=list(mono_corr_order),
                    lsv_params=_grp_lsv_df,
                    correl_bump=0.0, correl_bump_style="Relative",
                )
                mono_lsv_res = _price_in_batches(
                    mono_instruments,
                    metrics=[pricing_portal.create_metric("FairValue")],
                    price_id="Price",
                    scenario=_mono_lsv_scenario,
                )
                for i in range(len(mono_corr_order)):
                    ev_mono_lsv_zero_values[i] = _pt_bump_fv(mono_lsv_res, i, "LSV0")
                    ev_mono_lsv_values[i] = _pt_bump_fv(mono_lsv_res, i, "LSV")
```

And inside the existing `for corr_val, ticker_indices in corr_groups.items():` loop,
record each response (needed by the extraction above):

### Replace

```python
            for corr_val, ticker_indices in corr_groups.items():
                # Build instruments for this group
```

### By

```python
            group_results = []  # (ticker_indices, n_group, group_res) per correlation group
            for corr_val, ticker_indices in corr_groups.items():
                # Build instruments for this group
```

plus, right after the `group_res = _price_in_batches(...)` call:

```python
                group_results.append((ticker_indices, len(ticker_indices), group_res))
```

## 3. Post-processing: nothing to change

The shared post-processing already consumes `ev_cross_lsv_values`,
`ev_cross_lsv_zero_values`, `ev_cross_lcm_values`, `ev_mono_lsv_values`,
`ev_mono_lsv_zero_values` and only emits columns for non-None values — with the
lists now populated, the `Strike Cross Corr LSV/LCM (%)`, `EV Mono LSV (%)` and
cap-priced variants appear automatically. (Keep the crash-fix from
PRICING_INDIV_CORR_FIX: the lists are now initialized in §2 above.)

## 4. Fallback if `correl_bump` cannot force an absolute level

If §1's check fails (`build_unified_scenario` only supports relative bumps), keep the
LV batch with the corr-override scenario (today's code) and run ONE additional batch
per group with the unified LSV/LCM scenario **without** corr override; document that
LSV/LCM values are then computed at the model correlation while LV uses the forced
one (approximation, usually small for bump-sized impacts). Structure: reuse §2 but
price `group_instruments` a second time with the unified scenario instead of reading
bumps from the same response.

## Verify

1. `solve(..., correl_input_method="Individual Correlations", use_lsv=True)` →
   `Strike Cross Corr LSV (%)` present and finite for every ticker.
2. Same + `use_lcm=True` → `Strike Cross Corr LCM (%)` present; mono rows (ticker ==
   corridor asset) still raise the intended LCM ValueError.
3. `Global Parameters` mode → unchanged columns/values.
4. Per-ticker mode without LSV/LCM → unchanged (no new columns, no extra batch).
