# Phase 10a receipt (in progress) — cross-surface end-to-end fixtures and round-trip tests

Baseline commit: `baa7ae4` (Phase 9 receipt reconciliation, see phase-9-receipt.md).

Working autonomously per explicit instruction: fix problems directly, make
best-judgment calls on decisions, and record them here rather than stopping
to ask. Nothing below required a design change — every fixture found to have
a coverage gap was closed against the existing resolver/API contract as
written; no validator was relaxed and no fixture was edited to match observed
behavior.

## Gap analysis method

Cross-referenced the 10 required end-to-end fixtures (execution plan
Phase 10a) against existing coverage by grepping for the concrete
mechanism each fixture exercises (not just its name), across
`src/presets/*.rs`, `src/inference/*.rs`, `src/web/api/*.rs`, and
`tests/fixtures/presets/`. Findings:

| # | Fixture | Prior coverage | Action |
|---|---|---|---|
| 1 | Legacy v4 llama preset | `tests/fixtures/presets/schema-v4/*.json` + migration tests | none needed |
| 2 | Invalid legacy mixed-KV preserved but blocked | `schema-v4/kv_conflict_preserved.json` | none needed |
| 3 | Qwen3.8 Brainwaves Q4/Q5 bundle | `schema-v6/q4-q5-exact-tune.json`, resolver golden fixture | none needed |
| 4 | Qwen3.6 35B MoE bundle, several `n_cpu_moe` choices | Functional MoE bundle fixtures exist but none literally named "Qwen3.6 35B" | deferred — cosmetic naming only, no functional gap (see Outstanding) |
| 5 | Dense bundle proving no MoE choices | `resolver.rs` default `bundle()` test fixture is Dense with `cpu_moe_options: vec![0]` | none needed |
| 6 | Vision bundle, mmproj on/off/unavailable | **zero** — `validate_typed_runtime_fields`'s `CAPABILITY_UNAVAILABLE` path for `mmproj_offload` had no test anywhere | closed, slice 3 |
| 7 | Reasoning model, effort + preserve gating, template-compatibility states | Covered by `llama_cpp.rs` phase2 argv tests; `reasoning_preserve_template` gate itself was **removed** in `035647e` (Phase 9) — the fail-closed check no longer exists, so "supported/unsupported/unknown template-compatibility fixtures" is moot post-035647e | none needed — architecturally closed, documented in phase-9 receipt |
| 8 | Unknown/degraded metadata bundle | **zero** — `N_CPU_MOE_METADATA_UNKNOWN` (Unknown `model_kind` / missing `moe_layer_count`) had no test in the repo | closed, slice 2 |
| 9 | Rapid-MLX preset, no llama field leakage | `validate_preset_backend_config` had no *direct* test (only reached indirectly via `request_from_api_payload`, a separate function with its own check) | closed, slice 3 |
| 10 | Exact/related/no-evidence memory states | Exact/compatible/related/stale covered by Phase 9 slice 5's Details-drawer work; the null (no-receipt-exists) case was never itself asserted | closed, slice 4 |

Also closed while in the neighborhood (round-trip requirements listed
alongside the fixture table, not fixtures themselves):

- "create/update/delete/reset/selection/conversion disk failures each
  return non-success and leave in-memory state unchanged" — 4/6 operations
  had zero coverage (`api_create_preset`, `api_update_preset`,
  `api_delete_preset`, `api_reset_presets`); closed across slices 1 and 2.
- "Windows-style paths and argv remain OsString safe" — zero coverage;
  closed, slice 5.

## Slice 1 — CRUD disk-failure round trips

Commit `c4c4b93`. Adds `break_presets_write()` (occupies `save_presets`'s
`.json.tmp` target with a directory — portable, works as root, no
permission-bit dependence) plus disk-failure tests for selection-patch,
convert-to-bundle, copy, delete, and reset. Deferred `api_create_preset` /
`api_update_preset` to the next slice (no harness existed yet in
`presets.rs`).

### Verification
- `cargo test --lib preset_bundles::tests` — 13/13 passed
- `cargo test --lib preset_routes::tests` — 4/4 passed
- `cargo clippy --all-targets --all-features -- -D warnings` — clean
- `cargo test --lib` — 1392 passed, 1 pre-existing unrelated flake (below)

## Slice 2 — create/update disk failures + n_cpu_moe metadata gating

Commit `96e7505`. Adds a `test_context`/`ApiCtx` harness to `presets.rs`
(none existed before this session) plus `create_disk_failure_...` and
`update_disk_failure_...` tests, closing the disk-failure round trip for
all 6 CRUD-adjacent operations.

Also closes fixture 8: `validate_runtime_selection`'s four `N_CPU_MOE_*`
issue codes (`N_CPU_MOE_DENSE_MODEL`, `N_CPU_MOE_UNIFIED_MEMORY_UNQUALIFIED`,
`N_CPU_MOE_METADATA_UNKNOWN`, `N_CPU_MOE_EXCEEDS_LAYER_COUNT`) had no test
anywhere in the repo — a pure-function gap, not surfaced by any existing
API-level test. Added one test per code plus a passing-case control so the
negative assertions aren't vacuous.

### Verification
- `cargo test --lib preset_routes::tests` — 6/6 passed
- `cargo test --lib presets::resolver::tests` — 11/11 passed
- `cargo clippy --all-targets --all-features -- -D warnings` — clean
- `cargo test --lib` — 1399 passed, 1 pre-existing unrelated flake

## Slice 3 — mmproj capability gating + Rapid-MLX field isolation

Commit `5c03e1f`. Closes fixture 6 (three `resolve_preset` tests: mmproj
on/off accepted when the binary advertises the matching polarity, rejected
with `CAPABILITY_UNAVAILABLE` when it doesn't) and fixture 9.

For fixture 9, discovered mid-slice that `validate_preset_backend_config`
already had *some* indirect coverage via `preset_backend_config_mismatches_are_rejected`
(llama-preset-with-rapid-config and rapid-preset-missing-config) that an
earlier grep pass missed (the grep filtered call sites, not just
definitions). Trimmed the new test set from 4 to 2 to avoid duplicating
that — kept only the two genuinely new cases: a fully valid Rapid-MLX
preset with all llama-only fields at default passes, and a Rapid-MLX
preset carrying a `bundle` is rejected (the one direction the existing
test didn't check).

### Verification
- `cargo test --lib presets::resolver::tests` — 14/14 passed
- `cargo test --lib inference::launch::tests` — 18/18 passed
- `cargo clippy --all-targets --all-features -- -D warnings` — clean

## Slice 4 — null launch-evidence state

Commit `c6d0d4d`. Closes fixture 10's remaining case: a bundled preset
resolved against a config whose `launch_evidence_dir()` is empty (the
default in `test_context`, since `AppConfig::for_test`'s config dir has no
calibration history) returns `evidence: null` and still resolves `ok: true`
— matching `lookup_evidence`'s documented contract that an evidence lookup
failure is evidence-absent, not an error.

### Verification
- `cargo test --lib preset_bundles::tests` — 14/14 passed
- `cargo clippy --all-targets --all-features -- -D warnings` — clean

## Slice 5 — Windows-style path / argv OsString safety

Commit `06cf4c5`. `build_launch()` never invokes a shell — `Command::arg`
passes each value as a distinct OS-level argument — so the only realistic
risk is in-process string handling (accidental backslash escaping or
forward-slash normalization) before the value reaches argv. Confirmed
`build_launch()` does not call `validate()` (which would `Path::exists()`-
check the model path and fail on a synthetic Windows path on this host), so
the existing `launch_args()` test helper could exercise a raw
`C:\Users\test\models\model.gguf` path directly. Added one test asserting
the value lands in argv byte-for-byte.

### Verification
- `cargo test --lib inference::llama_cpp::tests::windows_style_model_path_passes_through_argv_unmodified` — passed
- `cargo clippy --all-targets --all-features -- -D warnings` — clean
- `cargo test --lib` (full, after slice 5) — 1406 passed, 1 pre-existing unrelated flake

## Pre-existing flaky test (confirmed unrelated, not touched by this phase)

`calibration::executor::tests::post_apply_fake_runtime_persists_passed_validation`
fails intermittently under full-suite parallel `cargo test --lib` but passes
every time in isolation. Investigated per the systematic-debugging skill
(Phase 1: reproduce). Confirmed non-reproducible in isolation across
multiple runs both before and after this phase's changes; `src/calibration/executor.rs`
was not touched in this phase. Concluded environmental/parallelism-related,
not a logic regression. Left as-is — out of this phase's scope.

## Outstanding (not yet closed)

- **Fixture 4 naming**: no fixture in the repo is literally named
  "Qwen3.6 35B" — MoE-bundle functional coverage exists (curated
  `n_cpu_moe` choices, layer-count and metadata gating all tested per
  slice 2), so this is a cosmetic gap against the hard gate's literal
  wording, not a functional one. Judgment call: left open rather than
  renaming an existing fixture arbitrarily, since no test currently
  depends on that exact name and inventing one risks looking like
  fixture-shopping to satisfy the gate rather than proving behavior.
- **Named full-chain round-trip tests**: "wizard Guided -> save -> card ->
  Configure -> editor -> Start", "wizard Pro -> save -> editor -> wizard
  template reload", "legacy -> explicit bundle conversion -> Q4/Q5
  selection" — not yet verified against `tests/ui/core/spawn-wizard.spec.js`
  / `preset-flow.spec.js` for an exact named match to each phrase.
- **Card filtering/grouping/collections/sorting/running-badge/delete**:
  suspected coverage lives partly in `tests/ui/core/launch-grid.spec.js`,
  not yet cross-checked against the round-trip list verbatim.
- **Focused verification suite** not yet run end-to-end this phase:
  `npm run validate-preset-bundle-contract`, `npm run validate-js`,
  `npm run lint`, the 5 named Playwright specs, `git diff --check`.

Continuing autonomously into the remaining items above.
