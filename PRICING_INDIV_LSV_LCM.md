# PRICING_INDIV_LSV_LCM — enable LSV/LCM in Individual Correlations mode (v2, corrected)

## Goal

`solve(..., correl_input_method="Individual Correlations", use_lsv=True, use_lcm=True)`
currently produces **no** LSV/LCM columns (the per-ticker branch hardcodes the LSV/LCM
lists to `None`). This patch prices the LSV/LSV0/LCM bumps **under each group's forced
correlation**, so the columns appear with exact values.

## Design (corrected after reviewing `pricing_scenarios.py`)

`correl_bump` in `build_unified_scenario` is a **perturbation** mutator
(`GenericMutatorBumpCorrelationEqEq`, BumpSize + Style Relative/Absolute) — it cannot
force a correlation LEVEL. Forcing a level requires
`GenericMutatorOverrideCorrelationEqEq(CorrelationLevel=X)`.

So the patch adds an optional `correl_override` to the scenario builders: when set,
the correlation mutator in **every** bump (LV, LSV0, LSV) is the override mutator at
the group's level instead of the no-op/bump correl mutators. One scenario per
correlation group, same number of batch calls as today.

---

## 1. `functions/common/pricing_scenarios.py` — add `correl_override`

### In `build_lsv_bumps` — signature: replace

```python
def build_lsv_bumps(
    pricing_portal,
    underlying_rics: List[str],
    lsv_params: pd.DataFrame,
    correl_bump: float = 0,
    correl_bump_style: str = "Relative",
) -> Dict[str, Any]:
```

### By

```python
def build_lsv_bumps(
    pricing_portal,
    underlying_rics: List[str],
    lsv_params: pd.DataFrame,
    correl_bump: float = 0,
    correl_bump_style: str = "Relative",
    correl_override: Optional[float] = None,
) -> Dict[str, Any]:
```

### Replace

```python
    lv_correl_noop = pricing_portal.create_scenario_mutator(
        name="GenericMutatorBumpCorrelationEqEq",
        mutator_properties=pricing_portal.create_scenario_mutator_properties(
            {"BumpSize": 0.0, "Style": correl_bump_style}
        ),
        mutator_properties_asset_overrides=[]
    )
```

### By

```python
    if correl_override is not None:
        # Force the correlation LEVEL in every bump (per-ticker correlation mode):
        # an override mutator replaces the no-op/bump correl mutators.
        _corr_level_mutator = pricing_portal.create_scenario_mutator(
            name="GenericMutatorOverrideCorrelationEqEq",
            mutator_properties=pricing_portal.create_scenario_mutator_properties(
                {"CorrelationLevel": float(correl_override)}
            ),
            mutator_properties_asset_overrides=[]
        )
        lv_correl_noop = _corr_level_mutator
    else:
        lv_correl_noop = pricing_portal.create_scenario_mutator(
            name="GenericMutatorBumpCorrelationEqEq",
            mutator_properties=pricing_portal.create_scenario_mutator_properties(
                {"BumpSize": 0.0, "Style": correl_bump_style}
            ),
            mutator_properties_asset_overrides=[]
        )
```

### Replace

```python
    lsv_correl_mutator = pricing_portal.create_scenario_mutator(
        name="GenericMutatorBumpCorrelationEqEq",
        mutator_properties=pricing_portal.create_scenario_mutator_properties(
            {"BumpSize": correl_bump, "Style": correl_bump_style}
        ),
        mutator_properties_asset_overrides=[]
    )
```

### By

```python
    if correl_override is not None:
        lsv_correl_mutator = _corr_level_mutator
    else:
        lsv_correl_mutator = pricing_portal.create_scenario_mutator(
            name="GenericMutatorBumpCorrelationEqEq",
            mutator_properties=pricing_portal.create_scenario_mutator_properties(
                {"BumpSize": correl_bump, "Style": correl_bump_style}
            ),
            mutator_properties_asset_overrides=[]
        )
```

### Thread the parameter through the two wrappers

In `build_lsv_scenario` and `build_unified_scenario`: add
`correl_override: Optional[float] = None` to each signature and pass
`correl_override=correl_override` into every `build_lsv_bumps(...)` /
`build_lsv_scenario(...)` call they make. (LCM-only path: no correl mutator exists in
its bumps — with `correl_override` set and LCM enabled, also append
`_corr_level_mutator`-equivalent into the LCM bump via the same override; simplest is
to always build LSV parts when `correl_override` is set and rely on the LSV+LCM path.)

## 2. `functions/dispersion/_pricing.py` — per-group unified scenario

In the per-ticker branch, `for corr_val, ticker_indices in corr_groups.items():` loop.

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
                # correlation forced to its level — bumps are read from the same
                # response, so no extra batch calls.
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
                        correl_override=corr_val,
                        lcm_properties=cfg.lcm_params.get('lcm_properties') if cfg.lcm_params else None,
                    )
                elif corr_val is not None:
                    group_scenario = pricing_portal.create_scenario_simple(
                        mutator_name="GenericMutatorOverrideCorrelationEqEq",
                        properties={"CorrelationLevel": corr_val}
                    )
```

## 3. Record group responses + extract the bumps

### Replace

```python
            for corr_val, ticker_indices in corr_groups.items():
                # Build instruments for this group
```

### By

```python
            group_results = []  # (ticker_indices, group_res) per correlation group
            for corr_val, ticker_indices in corr_groups.items():
                # Build instruments for this group
```

plus, right after the `group_res = _price_in_batches(...)` call:

```python
                group_results.append((ticker_indices, group_res))
```

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

            # LSV/LCM bumps: per bump name, the response entry holds
            # bump_data[0]["FairValue"][0]["value"] (same shape as _get_bump_fv).
            if (_use_lsv_pt or _use_lcm_pt) and group_results:
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
                                    fv_list = bump_data[0].get("FairValue", [])
                                    if fv_list:
                                        v = fv_list[0].get("value")
                                        return v if isinstance(v, (int, float)) else None
                            return None
                        running += cd["chunk_size"]
                    return None

                for ticker_indices_b, group_res_b in group_results:
                    for local_i, global_i in enumerate(ticker_indices_b):
                        if _use_lsv_pt:
                            ev_cross_lsv_zero_values[global_i] = _pt_bump_fv(group_res_b, local_i, "LSV0")
                            ev_cross_lsv_values[global_i] = _pt_bump_fv(group_res_b, local_i, "LSV")
                        if _use_lcm_pt:
                            ev_cross_lcm_values[global_i] = _pt_bump_fv(group_res_b, local_i, "LCM")

            # Mono LSV: one extra mono batch under a unified LSV scenario (mono
            # assets have no correlation override — corridor asset IS the asset).
            if _use_lsv_pt:
                from functions.common.pricing_scenarios import build_lsv_scenario
                _mono_lsv_scenario = build_lsv_scenario(
                    pricing_portal, list(mono_corr_order), _grp_lsv_df,
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

## 4. Post-processing: nothing to change

The shared post-processing only emits columns for non-None values — with the lists
populated, `Strike Cross Corr LSV/LCM (%)`, `EV Mono LSV (%)` and the cap-priced
variants appear automatically. (The lists are now initialized in §3, which also
covers the earlier `ev_mono_lsv_values` UnboundLocalError crash-fix.)

## Verify

1. `solve(..., correl_input_method="Individual Correlations", use_lsv=True)` →
   `Strike Cross Corr LSV (%)` present and finite for every ticker.
2. Same + `use_lcm=True` → `Strike Cross Corr LCM (%)` present; mono rows
   (ticker == corridor asset) still raise the intended LCM ValueError.
3. With `Correlation` column values forced (e.g. 55/60/48): the LV strikes must
   match today's per-ticker LV results (same forced level, now via the unified
   scenario) — compare one row before/after the patch.
4. `Global Parameters` mode with LSV/LCM → unchanged.
5. Per-ticker mode WITHOUT LSV/LCM → unchanged (no new columns, no extra batch).
