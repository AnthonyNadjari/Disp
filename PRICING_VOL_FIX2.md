# Fix 2 — vol columns disappear in scenario mode / ATMS empty for one asset

Follow-up to `PRICING_VOL_FIX.md`. Two distinct causes, both in `_pricing.py`:

1. **ATMF gone for every asset**: when the run goes through the
   unified-scenario branch, the response nests metrics INSIDE the bump —
   `entry["LV"][0]["QueryLocalCcyVol"]`. `_get_bump_fv` navigates that
   (which is why FairValues still print), but `_get_bump_vol` (~line 3879)
   and `_get_bump_atms` (~line 3903) read `entry["QueryLocalCcyVol"]` at the
   TOP level → always empty → every vol is `None` → the display layer drops
   the whole column.
2. **ATMS missing for one asset only**: the separate ATMS batch can come
   back partially/fully empty (a chunk that errored stores `raw={}`
   silently). You saw the all-empty version last run ("NOT FOUND"); a
   per-chunk failure produces the SPX-only hole. The patch makes both cases
   print a loud warning naming the asset instead of a silent blank.

One shared helper + two replaced functions. (If you applied Fix 1, also
apply Edit 1 below to `_get_metric_list_for_instrument` — same bump-aware
unwrapping, so the non-scenario branch is covered too.)

---

## Edit 0 — ADD once, next to `_extract_vol`

```python
        def _unwrap_vol_list(entry, bump_name=None):
            """QueryLocalCcyVol list from a result entry, wherever the portal
            put it: top level, inside a named bump (entry['LV'][0]), or inside
            a SimpleScenarioBump wrapper. Returns [] when absent."""
            if not isinstance(entry, dict):
                return []
            vol_list = entry.get("QueryLocalCcyVol", [])
            if isinstance(vol_list, list) and vol_list:
                return vol_list
            bump_names = [bump_name] if bump_name else ["LV", "LSV", "LSV0", "LCM"]
            for bn in bump_names:
                bump_data = entry.get(bn, [])
                if isinstance(bump_data, list) and bump_data and isinstance(bump_data[0], dict):
                    vol_list = bump_data[0].get("QueryLocalCcyVol", [])
                    if isinstance(vol_list, list) and vol_list:
                        return vol_list
            bumps = entry.get("SimpleScenarioBump", [])
            if isinstance(bumps, list) and bumps and isinstance(bumps[0], dict):
                for bn in bump_names:
                    bump_data = bumps[0].get(bn, [])
                    if isinstance(bump_data, list) and bump_data and isinstance(bump_data[0], dict):
                        vol_list = bump_data[0].get("QueryLocalCcyVol", [])
                        if isinstance(vol_list, list) and vol_list:
                            return vol_list
            return []
```

---

## Edit 1 — REPLACE `_get_bump_vol` (~lines 3879–3901)

```python
                def _get_bump_vol(global_idx, bump_name, expected_asset):
                    running_idx = 0
                    for chunk_start in sorted(results_map.keys()):
                        chunk_data = results_map[chunk_start]
                        chunk_size = chunk_data["chunk_size"]
                        if global_idx < running_idx + chunk_size:
                            local_idx = global_idx - running_idx
                            raw = chunk_data["raw"]
                            key = "Price" if local_idx == 0 else f"Price_{local_idx}"
                            vol_list = _unwrap_vol_list(raw.get(key), bump_name)
                            val = _extract_vol(vol_list, expected_asset)
                            if val is None:
                                found = [v.get("PrimaryAssetRef") for v in vol_list
                                         if isinstance(v, dict)] or "EMPTY"
                                dbg.warn("batch", f"_get_bump_vol({global_idx}, {bump_name}): "
                                                  f"no vol for {expected_asset}, response had {found}")
                            return val
                        running_idx += chunk_size
                    return None
```

---

## Edit 2 — REPLACE `_get_bump_atms` (~lines 3903–3935)

```python
                def _get_bump_atms(global_idx, bump_name, expected_asset):
                    if not include_atms:
                        return None
                    if not include_atmf:
                        return _get_bump_vol(global_idx, bump_name, expected_asset)
                    if not results_map_atms:
                        dbg.warn("batch", f"_get_bump_atms({global_idx}): ATMS batch is EMPTY "
                                          f"— the second pricing call failed; ATMS for "
                                          f"{expected_asset} will be blank")
                        return None
                    running_idx = 0
                    for chunk_start in sorted(results_map_atms.keys()):
                        chunk_data = results_map_atms[chunk_start]
                        chunk_size = chunk_data["chunk_size"]
                        if global_idx < running_idx + chunk_size:
                            local_idx = global_idx - running_idx
                            raw = chunk_data["raw"]
                            key = "Price" if local_idx == 0 else f"Price_{local_idx}"
                            # ATMS batch is priced WITHOUT the scenario → usually
                            # top-level, but _unwrap_vol_list covers both shapes.
                            vol_list = _unwrap_vol_list(raw.get(key))
                            val = _extract_vol(vol_list, expected_asset)
                            if val is None:
                                found = [v.get("PrimaryAssetRef") for v in vol_list
                                         if isinstance(v, dict)] or "EMPTY (chunk failed?)"
                                dbg.warn("batch", f"_get_bump_atms({global_idx}): no ATMS for "
                                                  f"{expected_asset}, response had {found}")
                            return val
                        running_idx += chunk_size
                    dbg.warn("batch", f"_get_bump_atms({global_idx}): index beyond ATMS chunks "
                                      f"(a chunk errored and was stored empty?)")
                    return None
```

---

## Edit 3 — only if you applied Fix 1: make its getter bump-aware too

In `_get_metric_list_for_instrument` (added by `PRICING_VOL_FIX.md`), replace
the Format-B body — everything between `if key in raw:` and the Format-A
fallback — with:

```python
                    if key in raw:
                        entry = raw[key]
                        if metric_name == "QueryLocalCcyVol":
                            return _unwrap_vol_list(entry)
                        if isinstance(entry, dict):
                            metric_list = entry.get(metric_name, [])
                            if isinstance(metric_list, list) and metric_list:
                                return metric_list
                        return []
```

(Requires Edit 0's helper to be defined before it.)

---

## After applying: read the warnings

The blanks now tell you WHY in the console:
- `response had EMPTY` on ATMS → the second pricing call / one of its chunks
  failed (infra, not parsing) — rerun or check the portal error for that chunk.
- `response had ['COP.N']` (corridor only, no index RIC) → the portal genuinely
  did not return the variance asset's vol for that FPF/metric — then the clean
  fallback is querying the missing vol directly (`compute_implied_vol(ticker,
  matu, "Spot"/"Forward")`), the same dedicated lookup the non-batch path uses.
