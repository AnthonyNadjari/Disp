# Fix — 'Cap' / 'Uncapped' FPFs are labels, not real variants (`_pricing.py`)

**Root cause.** `generate_fpf_string()` (~line 1221) builds ONE cross FPF and
ONE mono FPF with `is_capped=self.is_capped` — the interface toggle. Then
`_build_fpf_dict` (~line 1278) keys that SAME string `'Cross Cap'` or
`'Cross Uncapped'` **based on the same toggle**: the two "variants" are one
FPF with two possible names. With the box ticked, the so-called uncapped FPF
is capped; unticked, the capped one doesn't exist.

**Fix.** Build BOTH variants explicitly (the strikes are already computed, so
it's just two `build_corridor_fpf` calls per leg) and emit both keys always.
The toggle keeps its role everywhere else (which variant is primary/parsed/
booked) — behaviour of every existing consumer is unchanged.

---

## Edit 1 — REPLACE `generate_fpf_string` (~lines 1221–1276)

```python
    def generate_fpf_string(self):
        """Generate FPF string(s) with computed strikes and store FPF objects.

        Builds BOTH cap variants per leg — is_capped only selects which one
        is the primary (parsed / returned / booked), it no longer decides
        which variants exist."""
        if self.strike_variance_asset is None:
            return "Strike not computed"

        def _build_cross(capped):
            return build_corridor_fpf(
                tickers=[self.ref_asset],
                last_obs_date=self.last_obs_date,
                strike_date=self.strike_date,
                strikes=[self.strike_variance_asset],
                weights=[1.0],
                low_barrier=self.dvar,
                high_barrier=self.uvar,
                is_capped=capped,
                corr_asset=self.linked_asset,
                currency=self.currency,
                use_parameters=False
            )

        self.fpf_cross_capped = _build_cross(True)
        self.fpf_cross_uncapped = _build_cross(False)
        fpf_string_cross = self.fpf_cross_capped if self.is_capped else self.fpf_cross_uncapped

        # Parse and store the FPF object (primary variant, as before)
        try:
            self.fpf_obj_cross = FPFUnifiedEconomicsWrapper.from_data(
                fpf_string_cross,
                script_cls=corridorCovarianceSwap_v4
            )
        except Exception as e:
            dbg.warn("FPF", f"cross parse failed for {self.ref_asset}: {e}")
            self.fpf_obj_cross = None

        # If cross-corridor (ref_asset != linked_asset), also generate mono-corridor FPF
        self.fpf_mono_capped = None
        self.fpf_mono_uncapped = None
        if self.ref_asset != self.linked_asset and self.strike_corridor_asset is not None:

            def _build_mono(capped):
                return build_corridor_fpf(
                    tickers=[self.linked_asset],
                    last_obs_date=self.last_obs_date,
                    strike_date=self.strike_date,
                    strikes=[self.strike_corridor_asset],
                    weights=[1.0],
                    low_barrier=self.dvar,
                    high_barrier=self.uvar,
                    is_capped=capped,
                    corr_asset=self.linked_asset,
                    schedule_calendar_asset=self.ref_asset,
                    currency=self.currency,
                    use_parameters=False
                )

            self.fpf_mono_capped = _build_mono(True)
            self.fpf_mono_uncapped = _build_mono(False)
            fpf_string_mono = self.fpf_mono_capped if self.is_capped else self.fpf_mono_uncapped

            # Parse and store the mono FPF object (primary variant)
            try:
                self.fpf_obj_mono = FPFUnifiedEconomicsWrapper.from_data(
                    fpf_string_mono,
                    script_cls=corridorCovarianceSwap_v4
                )
            except Exception as e:
                dbg.warn("FPF", f"mono parse failed for {self.linked_asset}: {e}")
                self.fpf_obj_mono = None
            return {"Cross Corridor FPF": fpf_string_cross, "Mono Corridor FPF": fpf_string_mono}
        else:
            # Regular corridor case - only one FPF
            return fpf_string_cross
```

---

## Edit 2 — REPLACE `_build_fpf_dict` (~lines 1278–1332)

```python
    def _build_fpf_dict(self, fpf_strings, lsv_params=None, lcm_params=None):
        """
        Build FPF dictionary with clear naming: {key: fpf_string}

        'Cross Cap' / 'Cross Uncapped' (and Mono) are now BOTH real,
        independently built variants — regardless of the is_capped toggle.
        LSV/LCM variants are single-build (they come from the solve) and are
        keyed by the mode they were actually built with.
        """
        is_capped = self.is_capped
        fpf_dict = {}

        cross_fpf = fpf_strings.get("Cross Corridor FPF", fpf_strings)
        mono_fpf = fpf_strings.get("Mono Corridor FPF", "")

        # Cross corridor FPFs — both variants, genuinely capped / uncapped
        fpf_dict['Cross Cap'] = getattr(self, 'fpf_cross_capped', None) or (cross_fpf if is_capped else None)
        fpf_dict['Cross Uncapped'] = getattr(self, 'fpf_cross_uncapped', None) or (cross_fpf if not is_capped else None)

        # LSV variants (only for cross corridor; built with self.is_capped)
        if lsv_params and lsv_params.get('enabled') and hasattr(self, 'lsv_fpf_cross') and self.lsv_fpf_cross:
            fpf_dict[f"Cross LSV {'Cap' if is_capped else 'Uncapped'}"] = self.lsv_fpf_cross

        # LCM variants (only for cross corridor; built with self.is_capped)
        if lcm_params and lcm_params.get('enabled') and hasattr(self, 'lcm_fpf_cross') and self.lcm_fpf_cross:
            fpf_dict[f"Cross LCM {'Cap' if is_capped else 'Uncapped'}"] = self.lcm_fpf_cross

        # Mono corridor FPFs — both variants
        if mono_fpf:
            fpf_dict['Mono Cap'] = getattr(self, 'fpf_mono_capped', None) or (mono_fpf if is_capped else None)
            fpf_dict['Mono Uncapped'] = getattr(self, 'fpf_mono_uncapped', None) or (mono_fpf if not is_capped else None)

        return {k: v for k, v in fpf_dict.items() if v}
```

---

## Scope note

This fixes the **generated FPF strings** (what you copy/book): both variants
now exist and each is built with the cap it claims. The *"Cap Priced"*
columns of the results table are a separate circuit — the pricing batch
prices the instruments it was given; if you also want it to always price
BOTH variants (e.g. capped EV even when the toggle is off), that is a batch-
construction change, not an FPF-labeling one — say the word and I'll write
that patch too. The `zero-strike` path is unaffected (it already forces
`is_capped=False` explicitly, ~line 516).
