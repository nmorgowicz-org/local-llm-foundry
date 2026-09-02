# Phase 8b receipt — preset bundle Configure drawer

Implemented the production Configure drawer behind the Phase 8a
`.launch-card-btn-configure` seam. The drawer owns module-local draft state;
opening copies the saved bundle selection, draft edits never mutate
`sessionState.presets`, and closing discards the draft. Preview requests are
monotonically generation-guarded and abort older requests. Start without
saving launches the normalized draft, Save persists through the revision-safe
selection PATCH, and Save & Start waits for successful persistence before
launching. Dirty close, Escape, and backdrop dismissal require confirmation;
Reset discards without prompting and focus returns to the opener.

The drawer renders visible disabled choices with backend reasons and
`aria-describedby`, derived `Custom` indicators, workload policy above the
predicted result, probe-backed MoE placement/headroom behavior, qualitative
CPU-offload slowdown guidance, fit-state results, and the full-preset escape
hatch. It includes dark/light theme, narrow bottom-sheet, reduced-motion, and
dialog/focus-trap behavior.

Workload policy is bundle-level persisted state, not part of the typed
`PresetBundleSelection`. Resolve accepts a top-level preview override and
PATCH accepts a top-level persisted value. The resolver centrally blocks
policy-ineligible `q4_0/q4_0` for agentic or unknown workloads and resolve
responses expose the disabled-option reason. The API documentation and
architecture contract were amended to freeze this boundary.

## Verification

- `rtk cargo clippy -- -D warnings` — passed
- `rtk cargo test web::api::preset_bundles` — 10 passed
- `rtk cargo test` — 1,446 passed, 14 ignored
- `rtk npm run validate-preset-bundle-contract` — passed
- `rtk npm run validate-js` — passed
- `rtk npm run lint` — passed
- `rtk git diff --check` — passed
- `rtk cargo build --release` — passed
- Isolated `core/preset-flow.spec.js` — 28 passed, 1 worker
- `preset-flow` includes Configure lifecycle, draft isolation/reset, stale
  response protection, disabled reasons, workload payload/diff, fit/headroom,
  dirty-close behavior, Start/Save ordering, and visible 409 Reload coverage.
- Screenshot harness `preset-bundle-editor` — six current-source artifacts
  captured: dark, light, narrow, MoE, dense-no-MoE, and removal-warning states.

Artifacts remain under the gitignored `docs/screenshots/artifacts/presets/`
directory and are not promoted because Phase 8c owns public screenshot
promotion.
