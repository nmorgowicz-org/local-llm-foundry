# Phase 5 receipt — Preset Editor llama.cpp parity

Phase 5 freezes the frontend field catalog in
`static/js/features/spawn-wizard-groups.js` and generates the review fixture
`tests/ui/core/fixtures/llama-config-field-catalog.json`. Existing hardware
fields that were missing from the registry are now represented, while native
llama.cpp projector-offload and reasoning controls are editor-complete with
explicit wizard-planned status for Phase 6.

The editor now loads and saves typed `mmproj_offload`,
`llama_reasoning_effort`, `llama_reasoning_format`,
`llama_reasoning_preserve`, and bundled `workload_policy` values. Rapid-MLX
reasoning remains isolated under `rapid_mlx`; it is never mapped to llama.cpp
argv. `ctk`/`ctv` are capability-sourced dropdowns with common-first ordering,
disabled unsupported values, and unknown-value preservation. The redundant
sampling overwrite was removed from `buildPresetPayload()`.

Focused contract commands:

- `npm run validate-preset-bundle-contract`
- `npm run validate-wizard-groups`
- `node scripts/validate-js.mjs`
- `npx playwright test core/llama-config-parity.spec.js core/phase7-presets.spec.js --workers=1`

The capability endpoint is authenticated and returns the exact bounded
llama-server snapshot used by the editor. No compatibility or VRAM formula is
implemented in the frontend catalog.
