# Phase 7 receipt — explicit Preset Editor bundle management

Phase 7 adds explicit model-variant management to the Preset Editor. Bundled
presets now show their exact artifact membership, local paths, roles, GGUF
metadata, launch selection, context/performance choices, and capability-backed
MoE controls. Artifact suggestions require explicit confirmation; duplicate
local paths and Hugging Face coordinates are rejected, and dense bundles do
not expose CPU MoE controls.

Legacy flat presets can be converted through `/api/presets/{id}/convert-to-bundle`
using the server-owned conversion defaults. Full-editor writes, copy, reset,
delete, and collection membership changes preserve revision and catalog-etag
guards. Flat-preset PUT behavior remains backward-compatible. Stale writes are
rejected, selected/only artifact removal is protected, and failed conversion
or save operations leave the original preset intact.

Documentation and UI evidence were updated in:

- `docs/reference/api.md`
- `docs/reference/setup-wizard.md`
- `tests/ui/capture/scenarios/presets/preset-bundle-editor.mjs`
- `tests/ui/capture/index.mjs`

## Verification

- `rtk cargo clippy -- -D warnings` — passed
- `rtk cargo test` — 1,445 passed, 14 ignored
- `rtk npm run validate-js` — passed
- `rtk npm run lint` — passed
- `rtk git diff --check` — passed
- `rtk cargo build --release` — passed
- Focused Playwright suite — 14 passed
- Screenshot harness — six preset-bundle-editor artifacts captured for dark,
  light, narrow, MoE, dense-without-MoE, and removal-warning states
