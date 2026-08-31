# Phase 2 receipt — v6 typed fields and bundle schema

Date: 2026-08-31

## Result

Phase 2 is complete on branch `feature/preset-bundle-launch-options`.

- Baseline commit: `eda0d071730b0c154b657f6d0e3e52d1bc2dcd2c`
- Resulting commit: recorded by the phase commit containing this receipt
- Scope: schema v6 bundle persistence, typed llama.cpp launch settings,
  capability evidence, exact-selection/runtime validation, and batch-import
  preservation

## Implementation evidence

- `src/presets/mod.rs` registers the bundle/resolver modules, persists schema
  v6 and revisions, preserves v5 migration behavior, and keeps migrated
  `bundle` values absent.
- `src/presets/bundle.rs` defines bounded bundle enums, tagged artifact
  provenance, companions, contexts, K/V policies, performance choices,
  workload policy, structural validation, default projection, and the single
  server-owned `create_bundle_preset()` constructor. New bundles store
  `fit_enabled: Some(false)`; migration does not call the constructor.
- `src/presets/resolver.rs` validates exact artifact/catalog/curated-policy
  membership and runtime mixed-K/V, unified-memory, dense/unknown-MoE, and
  layer-bound conditions. Negative `n_cpu_moe` is rejected.
- `src/inference/llama_cpp.rs` adds typed mmproj offload and llama.cpp
  reasoning effort/format/preserve values, exact valueless-flag emission,
  capability-gated launch validation, and fail-closed unknown-value handling.
  Native reasoning preservation remains blocked when template compatibility is
  unknown.
- `src/inference/llama_cpp_capabilities.rs` records typed flag forms,
  accepted values, defaults, and unknown template compatibility for the exact
  probed binary.
- `src/inference/launch.rs` materializes bundle defaults and applies runtime
  validation immediately before adapter construction.
- `src/llama/batch_import.rs` preserves typed mmproj/reasoning values and
  unknown values without silently converting them to runtime defaults.
- Supporting launch/capability plumbing is updated in
  `src/calibration/candidates.rs`, `src/calibration/executor.rs`,
  `src/presets/validation.rs`, `src/web/api/doctor.rs`, and
  `src/web/api/vram.rs` so all capability snapshot constructors and launch
  paths carry the v6 typed evidence shape.
- `tests/fixtures/presets/schema-v6/` contains legacy-flat, Q4/Q5 exact-tune,
  provenance/companion, performance/context/MoE, and typed llama.cpp fixture
  coverage.

## Verification

All commands exited 0 unless stated otherwise:

| Command | Result |
| --- | --- |
| `rtk cargo test presets::` | 44 passed, 0 failed |
| `rtk cargo test inference::llama_cpp::` | 25 passed, 0 failed |
| `rtk cargo test inference::llama_cpp_capabilities::` | 11 passed, 0 failed |
| `rtk cargo test llama::batch_import::` | 11 passed, 0 failed |
| `rtk cargo clippy -- -D warnings` | passed, no warnings |
| `rtk cargo test` | 1,412 passed, 14 ignored, 0 failed |
| `rtk cargo build --release` | passed |
| `rtk cargo fmt -- --check` | passed |
| `rtk git diff --check` | passed |

The focused tests cover exact serialization of every known reasoning-effort
value, unknown-value round trips, mmproj absent/disabled argv behavior,
reasoning-format outer `None` without an `auto` argument, curated-only versus
validated-custom selection, and negative/dense/unknown/over-bound `n_cpu_moe`.

## Known boundaries and next phase

- Bundle resolver intent/proposal algorithms, revisioned selection APIs, and
  authenticated API routes belong to Phase 3.
- Native reasoning-preserve remains non-launchable until a bounded,
  source-backed template-compatibility contract is qualified.
- Mixed main `q8_0/q4_0` remains fail-closed because the current capability
  evidence cannot prove fused-kernel support.
- No UI behavior or screenshot claim is made in this phase.
