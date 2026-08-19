# Fix — ATMF/ATMS vols wrong for the 2nd Variance Asset (`_pricing.py`)

**Root cause.** For a cross FPF the portal returns `QueryLocalCcyVol` with **two
entries (variance asset + corridor asset) in an UNSTABLE order** — your run
proved it: `Price` → `[VLLP.PA, .STOXX50E]` but `Price_1` → `[.STOXX50E, RENA.PA]`.
`_get_atmf` / `_get_atms` take `metric_list[0]` (first entry, whatever the
asset), so the `.SPX` leg picked up COP.N's vol (31.56). On top of that, the
call sites build `{tickers[i]: _get_atmf(i)}` with **duplicate keys** (3×
`.STOXX50E`) — last leg wins. The existing helper `_extract_vol(vol_list,
expected_asset)` (matches by `PrimaryAssetRef`) is the correct tool; it just
was never used by these two functions.

Four edits, all in the batch-pricing section (~lines 3786–3827 and 4030–4033).
`_extract_vol` itself is unchanged.

---

## Edit 1 — ADD this function right after `_extract_vol` (~line 3785)

```python
        def _get_metric_list_for_instrument(idx, metric_name, results):
            """Full metric LIST for instrument at global idx (every entry,
            not just [0]) — needed for QueryLocalCcyVol, where a cross FPF
            carries one entry per asset in unstable order."""
            running_idx = 0
            for chunk_start in sorted(results.keys()):
                chunk_data = results[chunk_start]
                chunk_size = chunk_data["chunk_size"]
                if idx < running_idx + chunk_size:
                    local_idx = idx - running_idx
                    raw = chunk_data["raw"]
                    key = "Price" if local_idx == 0 else f"Price_{local_idx}"
                    # Format B: one "Price_N" key per instrument
                    if key in raw:
                        entry = raw[key]
                        if isinstance(entry, dict):
                            if use_scenario and "SimpleScenarioBump" in entry:
                                bumps = entry["SimpleScenarioBump"]
                                if bumps and isinstance(bumps, list) and len(bumps) > 0:
                                    metric_list = bumps[0].get(metric_name, [])
                                    if isinstance(metric_list, list) and metric_list:
                                        return metric_list
                            metric_list = entry.get(metric_name, [])
                            if isinstance(metric_list, list) and metric_list:
                                return metric_list
                        return []
                    # Format A fallback: single "Price" key, one entry per
                    # instrument — wrap it so the caller can still match by
                    # PrimaryAssetRef (returns None instead of a wrong asset).
                    if "Price" in raw:
                        entry = raw["Price"]
                        if isinstance(entry, dict):
                            metric_list = entry.get(metric_name, [])
                            if (isinstance(metric_list, list)
                                    and local_idx < len(metric_list)
                                    and isinstance(metric_list[local_idx], dict)):
                                return [metric_list[local_idx]]
                    return []
                running_idx += chunk_size
            return []
```

---

## Edit 2 — REPLACE `_get_atmf` (~lines 3786–3789)

**Old:**
```python
        def _get_atmf(idx):
            if not include_atmf:
                return None
            return _get_metric_for_instrument(idx, "QueryLocalCcyVol")
```

**New:**
```python
        def _get_atmf(idx, expected_asset):
            """ATMF vol of `expected_asset` from instrument idx — ALWAYS
            matched by PrimaryAssetRef, never by position (the portal's
            entry order is unstable per instrument)."""
            if not include_atmf:
                return None
            vol_list = _get_metric_list_for_instrument(idx, "QueryLocalCcyVol", results_map)
            val = _extract_vol(vol_list, expected_asset)
            if val is None and vol_list:
                dbg.warn("batch", f"_get_atmf({idx}): no vol entry for "
                                  f"{expected_asset} in {[v.get('PrimaryAssetRef') for v in vol_list if isinstance(v, dict)]}")
            return val
```

---

## Edit 3 — REPLACE `_get_atms` (~lines 3791–3827)

**Old:** the whole current function (it takes `vol_list[0]` /
`vol_list[local_idx]` positionally in both modes).

**New:**
```python
        def _get_atms(idx, expected_asset):
            """ATMS vol of `expected_asset` from instrument idx.
            - ATMS-only mode: QueryLocalCcyVol in the main batch.
            - ATMF+ATMS mode: QueryLocalCcyVol in the separate ATMS batch.
            Matched by PrimaryAssetRef in both modes."""
            if not include_atms:
                return None
            if not include_atmf:
                vol_list = _get_metric_list_for_instrument(idx, "QueryLocalCcyVol", results_map)
                return _extract_vol(vol_list, expected_asset)
            if not results_map_atms:
                return None
            vol_list = _get_metric_list_for_instrument(idx, "QueryLocalCcyVol", results_map_atms)
            return _extract_vol(vol_list, expected_asset)
```

*(Note: `use_scenario` stays False for the ATMS batch — it is priced without
the scenario — so the SimpleScenarioBump branch is simply never taken there.)*

---

## Edit 4 — REPLACE the call sites (~lines 4030–4033)

**Old:**
```python
                atmf_vols_cross = {tickers[i]: _get_atmf(i) for i in range(n_ev)}
                atms_vols_cross = {tickers[i]: _get_atms(i) for i in range(n_ev)}
                atmf_vols_mono = {mono_corr_order[i]: _get_atmf(n_ev + i) for i in range(n_ev_mono)}
                atms_vols_mono = {mono_corr_order[i]: _get_atms(n_ev + i) for i in range(n_ev_mono)}
```

**New:**
```python
                atmf_vols_cross = {tickers[i]: _get_atmf(i, tickers[i]) for i in range(n_ev)}
                atms_vols_cross = {tickers[i]: _get_atms(i, tickers[i]) for i in range(n_ev)}
                atmf_vols_mono = {mono_corr_order[i]: _get_atmf(n_ev + i, mono_corr_order[i]) for i in range(n_ev_mono)}
                atms_vols_mono = {mono_corr_order[i]: _get_atms(n_ev + i, mono_corr_order[i]) for i in range(n_ev_mono)}
```

With per-asset matching the duplicate dict keys become harmless: whichever
`.STOXX50E` leg writes last, the value extracted IS `.STOXX50E`'s vol.
A mis-indexed slot now yields `None` (and a warning) instead of a plausible
wrong number.

---

## Separate issue seen in the same run (not this fix)

`SPX CROSS ATMS RAW: NOT FOUND` means the **separate ATMS batch came back
empty** (`results_map_atms` had no chunks) — the second `_price_in_batches`
call failed silently. Worth surfacing loudly right after it runs:

```python
            if include_atmf and include_atms and not batch_res_atms:
                dbg.warn("batch", "ATMS batch returned NO results — "
                                  "ATMS columns will be empty (check the second pricing call)")
```
