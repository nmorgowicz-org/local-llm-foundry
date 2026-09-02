# Phase 8c receipt — capture and register bundle screenshots

Baseline commit: `b9e53c48ba9ad7cce2723fea91d7c392f4bb336d`

## Summary

Added `tests/ui/capture/scenarios/presets/preset-bundle.mjs` and registered
`preset-bundle` (category `presets`) in `tests/ui/capture/index.mjs` with the
eleven `expectedOutputs` filenames from the architecture's screenshot gate.
The scenario mocks `/api/preset-cards`, `/api/presets`, bundle `resolve`,
`selection` PATCH (including a 409 revision-conflict response), estimate,
evidence, and spawn routes against a deterministic seeded Q4/Q5 bundle, then
walks the grid and Configure drawer through dark/light, narrow bottom-sheet,
reduced-motion, low-VRAM intent change, no-fit, evidence-exact,
evidence-related, and revision-conflict states.

Capturing the evidence states surfaced that the drawer had no rendering path
for the resolver's `EvidenceMatch` (architecture 12) — `preset-bundle-drawer.js`
now renders an evidence line (`renderEvidence`, `EVIDENCE_LABELS`) above the
result, styled per class in `preset-bundle-drawer.css`
(`.bundle-evidence--{exact,compatible,related,stale}`), verbatim from the
resolver's own class — never upgraded.

Capturing the grid state surfaced that the Phase 8a Configure button made the
launch-card action row four buttons wide, squeezing Start's label. Added
`.launch-card-actions--bundle` in `setup-view.css`: a two-row grid stacking
Configure/Edit above a full-width Start, with Trash anchored on the right
spanning both rows.

Both fixes are narrow, capture-driven defects in the Phase 8a/8b surfaces
(evidence had no renderer at all; the button row had no bundle-specific
layout), not new scope — no new behavior, routes, or state was added.

## Files changed

- `tests/ui/capture/scenarios/presets/preset-bundle.mjs` (new)
- `tests/ui/capture/index.mjs`
- `static/js/features/preset-bundle-drawer.js`
- `static/css/preset-bundle-drawer.css`
- `static/css/setup-view.css`

## Verification

- `rtk cargo build --release` — passed
- `rtk node tests/ui/capture/index.mjs --scenario preset-bundle` — all 11
  expected outputs saved
- `rtk bash scripts/check-unused-screenshots.sh` — no unreferenced screenshots
- `rtk npm run lint` — passed
- `rtk git diff --check` — passed

## Hard gate

Fresh screenshots visually match the architecture hierarchy: grid dark/light,
drawer default/light/narrow/reduced-motion, low-VRAM change, no-fit,
evidence-exact, evidence-related, and revision-conflict, all under
`docs/screenshots/artifacts/presets/` and none promoted to
`docs/screenshots/`.
