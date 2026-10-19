# Phase 2 receipt: K/V UI, advisories, default unification

Baseline: `3cd6660f` (end of Phase 1). Branch `feat/llamacpp-source-builds`.

## Delivered

- `src/inference/kv_advisories.rs`: scoped advisory table. Gemma 4 caution for quantized K **or** V
  (matched by `gemma4*` architecture or a `gemma-4`/`gemma4` file name; other Gemma generations
  excluded). Qwen3.8-27B info for exactly `q8_0`/`q4_0` (file name only). No broad Qwen entry. No model
  facts, no advisories. Every entry carries its source and is qualitative.
- Endpoint: advisories populated; an omitted `device_id` resolves to the only accelerator (or several
  identical ones) so a confirmed pair can reach `Native`; ambiguous stays `Unverified`
  (`device_source` reports which). The snapshot is reused while valid instead of re-probing per edit.
- `kv-support.js`: one client (fetch, 30 s cache, always-revalidate, in-flight dedupe, late-response
  discard, runtime-identity cache flush, DOM rendering via `textContent`), `kv-support.css`.
- Surfaces: wizard KV card (`#hw-kv-support`, shared by Guided and Pro), preset editor
  (`#modal-kv-support`), Configure drawer (`#bundle-kv-support`), launch-card chip.
- Guided mixed tile "K q8_0 · V q4_0" with a state sub-label, hidden on Metal unless already selected;
  a stored mixed pair now highlights it.
- Start is gated in the wizard for a known refusal (button disabled with reason + fix, and a guard in
  `spawnServer`). Plain Save is never gated. The drawer no longer disables a KV option for an
  `advisory` capability reason.
- D11: new presets default to `q8_0`/`q8_0`; stored presets keep their values. Stale "f16 can be
  faster" tooltip and the old flash-attention tooltip rewritten.
- Docs: `docs/reference/spawn-wizard.md`, `docs/reference/api.md`.

## Evidence

- `tests/ui/core/kv-support.spec.js` (15 tests) plus 1 new `preset-flow.spec.js` test; 47 + 36 passing
  in the two focused runs, and 84/85 then 85/85 in the broader impacted set after fixing one regression.
- Screenshots: `docs/screenshots/artifacts/llama-build/kv-support--llamacpp-local--*.png` (4) and
  `kv-support--receipt.json`. Synthetic evidence (mocked endpoint); presentation only.

| Command | Exit |
| --- | --- |
| `cargo fmt --check` | 0 |
| `cargo clippy --all-targets -- -D warnings` | 0 |
| `cargo test --lib` (1685 passed, 15 ignored) | 0 |
| `npm run lint` / `validate-js` / `validate-wizard-groups` | 0 |
| `git diff --check` | 0 |
| `playwright core/kv-support.spec.js` + wizard/preset/parity specs | pass |

## Defects found while testing (and fixed)

1. The first Pro placement was inside the hidden legacy inline row, so it was never visible. Playwright
   `data-kv-state` assertions passed on it anyway; the screenshot caught it. Guided and Pro share the
   KV decision card, so there is now one wizard line, and the spec asserts visibility in both profiles.
2. Launch chips made every default `q8_0/q8_0` card show "? unverified" (no model/device on bundle
   cards). Unverified now earns a chip only for a mixed pair. Caught by an existing `preset-flow` test.
3. `.hw-decision-tile` `display` defeated the `hidden` attribute on the Metal tile; added a CSS rule.

## Deviations from the plan, to confirm

1. **Pro is not a separate surface.** The plan listed Guided and Pro separately; in the product they
   share the KV card, so "identical text" holds by construction.
2. **`calibration/executor.rs:2231-2232` and `setup-view.js:1087,1157` keep `f16`.** They are the
   fallback for a stored preset with an empty value (server default), not a new-preset default, and
   the executor comment documents hybrid models rejecting a quantized V. Changing them would measure
   or display something other than what launches.
3. **Rebuild link is a marker, not a link.** The hint text and `data-kv-rebuild` appear only for a
   confirmed missing buildable CUDA kernel; the action attaches when the Runtime panel exists (Phase 5).
4. **The drawer sends no model id** (bundle cards are redacted), so on CUDA it reports runtime
   evidence and stays honest that model execution is unverified. Bundles always launch `-fa on`, so
   the drawer has no Start gate.
5. **Model id is sent only for files under the primary models directory.** Models in extra model
   directories still get capability, just no model evidence or advisories.
6. **Pro ordering ("confirmed native first") is not implemented**: the selects list types, they do not
   rank them.
7. **Advisory for Qwen3.8-27B matches the file name only** (the architecture is shared with other
   Qwen3.x), which is deliberately narrow.

## Not done in Phase 2

- Metal bf16 hardware is read from the runtime (Phase 1 addendum); per-device multi-GPU bf16 is not.
- Light-theme and 430px screenshots of the new line; the CSS uses existing tokens only.
