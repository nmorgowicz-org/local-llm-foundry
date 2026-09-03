# Phase 6 receipt — Spawn Wizard Guided/Pro parity

Baseline: `322b7d2 feat(presets): complete typed llama editor controls`

Implementation commit: the Phase 6 commit containing this receipt

## Scope completed

- Added the canonical wizard controls for idle-slot cache retention, mmproj
  offload, and llama.cpp reasoning effort, format, and preservation.
- Registered the existing cache, SWA, load, checkpoint, continuous-batching,
  and verbosity controls in the shared Guided/Pro registry with corrected Pro
  categories.
- Wired reset, template restore, DOM state, capability locks, payload
  serialization, and review-summary rows through the existing canonical path.
- Kept native reasoning preservation fail-closed while binary or template
  compatibility is unknown, with the unavailable reason visible.
- Preserved legacy `preserve_thinking` as a separate chat-template kwarg.
- Restored top-level shared sampling fields when saving a Rapid-MLX wizard
  preset; Rapid launch fields remain backend-exclusive.

## Files changed

- `static/index.html`
- `static/js/features/spawn-wizard.js`
- `static/js/features/spawn-wizard-groups.js`
- `static/js/features/spawn-wizard-llama-ia.js`
- `static/js/features/spawn-wizard-review-step.js`
- `static/js/features/spawn-wizard-spawn.js`
- `tests/ui/core/phase6-wizard-parity.spec.js`
- `tests/ui/core/fixtures/spawn-wizard-control-contract.json`
- `tests/ui/core/fixtures/llama-config-field-catalog.json`
- `tests/ui/capture/index.mjs`
- `tests/ui/capture/scenarios/wizard-llamacpp/spawn-wizard-guided-drawer.mjs`
- `tests/ui/capture/scenarios/wizard-llamacpp/spawn-wizard-pro-baseline.mjs`
- `docs/reference/spawn-wizard.md`
- `docs/reference/setup-wizard.md`

## Validation

| Command | Result |
| --- | --- |
| `rtk npm run validate-preset-bundle-contract` | passed; 64 llama.cpp rows |
| `rtk npm run validate-wizard-groups` | passed; 82 controls, 64 llama.cpp mounts, 22 Rapid-MLX mounts |
| `rtk npm run validate-js` | passed |
| `rtk npm run lint` | passed |
| `rtk cargo build --release` | passed |
| isolated Playwright Phase 5/6/wizard suite | 41 passed |
| `rtk git diff --check` | passed |

The required parallel `rtk cargo test` gate was also run on the host path. It
reported four unrelated existing failures in two Rapid-MLX compatibility
probes and two calibration executor tests. The two groups pass when isolated
(`8 passed` for Rapid-MLX compatibility and the calibration executor group),
so no Phase 6 Rust regression is indicated; Phase 6 changes are frontend-only.

Focused Playwright command:

`rtk env CI=1 LLAMA_MONITOR_USE_RELEASE=1 LLAMA_MONITOR_TEST_PORT=17778 npx playwright test core/phase6-wizard-parity.spec.js core/spawn-wizard.spec.js core/llama-config-parity.spec.js --workers=1`

## Screenshot manifest

The following fresh artifacts were produced under
`docs/screenshots/artifacts/wizard-llamacpp/` by sequential current-source
capture scenarios:

- `llamacpp-local--spawn-wizard-guided-cache-slots.png`
- `llamacpp-local--spawn-wizard-guided-mmproj-offload.png`
- `llamacpp-local--spawn-wizard-guided-reasoning.png`
- `llamacpp-local--spawn-wizard-pro-model-compatibility.png`
- `llamacpp-local--spawn-wizard-pro-generation-reasoning.png`
- existing Guided drawer, Pro baseline, and Launch Full config outputs

The Guided drawer and Pro baseline capture contracts completed successfully;
the new control captures were inspected at 1280×900 and show their target
controls at readable scale. Artifacts remain unpromoted because public docs do
not reference them.

## Known boundaries

- The shipped llama.cpp capability provider does not advertise mixed main
  `q8_0/q4_0`; that policy remains rejected and is not weakened here.
- Native reasoning preservation remains unavailable when template
  compatibility is unknown, as required by the fail-closed contract.
- Exact runtime memory observations remain deferred to Phase 9.
