# Preset Bundle and Launch-Option Implementation / Execution Plan

Status: ready for architecture approval, then phase-by-phase execution

Repository baseline: `9332965956ef6d3dc5399ab2967e999b1b7d4669`

Date: 2026-08-30

Binding architecture: `docs/plans/20260830-preset_bundle_architecture.md`

Target implementers: Luna-class agents or a local Qwen3.8-27B-class model,
coordinated one phase at a time.

## 1. Goal

Implement the approved Option B compact model card and Configure drawer without
turning every launch combination into a separate preset. At the same time,
repair and extend the canonical Spawn Wizard Guided/Pro and Preset Editor
configuration system for llama.cpp parameters that are missing or not properly
registered.

The completed system must:

- consolidate multiple quantizations of one exact model/tune on one card;
- resolve context, approved K/V, batch/ubatch, and MoE placement server-side;
- keep non-card launch parameters in the Spawn Wizard and Preset Editor;
- round-trip wizard -> saved preset -> editor -> start without field loss;
- reject mixed main KV `q8_0/q4_0` on every path;
- distinguish estimates from exact runtime memory evidence;
- preserve every legacy preset and invalid/unreadable entry fail-closed;
- work on macOS, Linux, and Windows without inferred platform support.

## 2. Lower-tier worker protocol

Every phase is a separate context. Do not carry implementation across a context
boundary without writing the phase receipt.

For each phase:

1. Read the architecture contract and only the phase's listed references.
2. Confirm `rtk git status --short` and record the baseline commit.
3. Write or update failing tests/fixtures before product code.
4. Implement only the phase scope.
5. Run focused gates, then the phase gate.
6. Write `docs/plans/evidence/preset-bundles/phase-N-receipt.md` with:
   - baseline and resulting commit;
   - files changed;
   - raw command names and exit status;
   - test counts and exact failures/skips;
   - produced JSON/screenshot manifests;
   - known gaps and next-phase prerequisites.
7. Commit one conventional commit for the phase.
8. Stop. A fresh Verifier context checks the receipt, diff, tests, and contract.
9. Advance only after explicit Verifier sign-off.

The Coordinator owns phase order, commits/pushes, and architecture revisions.
The Builder may not weaken a gate to make it pass. The Verifier may add tests
but may not silently redesign the architecture.

If a phase finds a binding contract error, stop and revise the architecture and
this plan before continuing.

## 3. Global anti-pattern guards

Never:

- create a second Guided or Pro state/payload path;
- implement card compatibility in JavaScript formulas;
- persist the Cartesian product as separate generated presets;
- group tunes automatically by family/name/filename heuristic;
- infer quantization from names when GGUF metadata is available;
- treat `--help` support as behavioral proof;
- enable mixed `q8_0/q4_0`, or hide the option instead of rendering it disabled
  with the backend-provided reason;
- allow `extra_args` to override safety-critical typed fields;
- silently rewrite invalid legacy presets;
- report estimator `Measured` evidence as an exact run receipt;
- use filename `27G`/`30G` labels as measured VRAM;
- ignore persistence errors after mutating in-memory preset state;
- use innerHTML with untrusted model/preset/error text;
- run bare `npm test`, which can kill the live service on port 7778;
- run screenshot scenarios in parallel;
- add the `ready-to-test` label;
- invoke Playwright without the isolation environment. Every Playwright run in
  this project is written as
  `rtk env CI=1 LLAMA_MONITOR_USE_RELEASE=1 LLAMA_MONITOR_TEST_PORT=17778
  npx playwright test <specs…> --workers=1` (focused) or
  `rtk env CI=1 LLAMA_MONITOR_USE_RELEASE=1 LLAMA_MONITOR_TEST_PORT=17778
  npm test` (full suite, ≥ 600 s timeout). Any other form is a gate
  violation in a phase receipt.

## 4. Phase 0 allowed current APIs and seams

Implementation may reuse these verified current contracts:

- `ModelPreset`, `migrate_preset`, `load_presets`, `save_presets` in
  `src/presets/mod.rs`;
- `request_from_preset()` and `validate_preset_backend_config()` in
  `src/inference/launch.rs`;
- `ServerConfig` and `LlamaCppAdapter::build_launch()` in
  `src/inference/llama_cpp.rs`;
- exact binary identity and bounded `--help` probing in
  `src/inference/llama_cpp_capabilities.rs`;
- authenticated preset CRUD in `src/web/api/presets.rs`;
- authenticated `/api/vram-estimate` in `src/web/api/vram.rs`;
- `QuantOption` / `quant_comparison_table()` in
  `src/llama/vram_estimator/estimate.rs` for comparison patterns only;
- `CalibrationFingerprint`, `CalibrationReceipt`, and receipt match classes in
  `src/calibration/`;
- `wizardState`, canonical DOM relocation, and the single runtime presentation
  registry in
  `spawn-wizard.js`, `spawn-wizard-groups.js`, and
  `spawn-wizard-llama-ia.js`;
- `buildSpawnPayload()` and `buildPresetPayload()` in
  `spawn-wizard-spawn.js` and `spawn-wizard-review-step.js`;
- `_buildLaunchCard()`, `_fetchCardVramEstimates()`, and `_renderCardVram()` in
  `setup-view.js`;
- the accessible drawer lifecycle in `evidence-drawer.js` and
  `evidence-drawer.css`.

No bundle resolver, bundle-selection endpoint, exact launch-memory observation,
typed llama.cpp mmproj-offload, or typed llama.cpp reasoning-effort API exists
today. Those are new work and must not be referenced before their phases.

## 5. Required target field-parity matrix

Phase 0 freezes a machine-readable row for every setting across:

1. Rust persisted field;
2. schema default/migration;
3. preset API read/write;
4. Preset Editor load;
5. Preset Editor save;
6. wizard template restore;
7. wizard reset/default;
8. wizard DOM/state binding;
9. `buildSpawnPayload()`;
10. `buildPresetPayload()`;
11. Guided registry placement;
12. Pro category/search/modified state;
13. backend validation;
14. argv emission;
15. estimator/calibration applicability;
16. tests and docs.
17. `.cmd` batch-import parse/diagnostic behavior.
18. normalized resolved-argv fingerprint classification.

Every cell is a source location or explicit `not_applicable` with a reason.
Blank cells fail the validator.

Current dispositions to encode:

| Setting | Current disposition | Target |
|---|---|---|
| context, canonical K/V, batch, ubatch, `n_cpu_moe` | fully wired but incompletely validated | card axes plus wizard/editor parity |
| `repeat_last_n` | persisted/launched/UI, missing canonical registry/fixture | repair registry and tests |
| `kv_unified`, `no_cont_batching` | typed and launched | audit placement/defaults/tests |
| `swa_full`, `load_mode`, `verbosity`, checkpoints, cache reuse | physical controls exist; registry drift | register and contract-test |
| `cache_idle_slots` | backend/editor only | add canonical wizard control |
| `mmproj_offload` | absent typed support | add tri-state typed support |
| llama.cpp reasoning effort/format/preserve | absent typed support | add backend-native typed controls |
| Rapid-MLX reasoning effort | separate request-default concept | retain separate; no shared field |
| mixed main `q8_0/q4_0` | currently constructible | reject everywhere |
| `cache_type_k/v` duplicate fields | persisted/inspected but not launched | deprecate and migrate to `ctk/ctv` |

---

# Phase 0 — Documentation discovery and evidence freeze

## Objective

Create a reproducible baseline and freeze exact schemas, routes, controls,
runtime capabilities, screenshots, and corpus evidence before product edits.

## Read first

- `docs/plans/20260830-preset_bundle_architecture.md`
- `docs/reference/spawn-wizard.md`
- `docs/reference/setup-wizard.md`
- `docs/reference/api.md` preset/session/VRAM sections
- `docs/reference/vram-estimator.md`
- `docs/agents/playwright.md`
- `docs/agents/security-details.md`
- every source listed in section 4 above

## Tests/fixtures first

Add:

- `docs/plans/evidence/preset-bundles/phase-0/source-inventory.json`
- `docs/plans/evidence/preset-bundles/phase-0/field-parity.json`
- `docs/plans/evidence/preset-bundles/phase-0/route-auth-inventory.json`
- `docs/plans/evidence/preset-bundles/phase-0/schema-v4-fixtures.json`
- `docs/plans/evidence/preset-bundles/phase-0/schema-v5-target.json`
- `docs/plans/evidence/preset-bundles/phase-0/schema-v6-target.json`
- `docs/plans/evidence/preset-bundles/phase-0/api-target-fixtures.json`
- `docs/plans/evidence/preset-bundles/phase-0/argv-fingerprint-classification.json`
- `docs/plans/evidence/preset-bundles/phase-0/runtime-flags-local.json`
- `docs/plans/evidence/preset-bundles/phase-0/runtime-flags-ryne.json`
- `docs/plans/evidence/preset-bundles/phase-0/ryne-corpus-redacted.json`
- `docs/plans/evidence/preset-bundles/phase-0/capture-manifest.json`
- `docs/plans/evidence/preset-bundles/phase-0/fit-probe-output-fixtures.json`
  — sanitized real `llama-fit-params` outputs with **stdout and stderr
  captured separately per run**: the two-line compact form on stdout
  (`DEVICE model context compute` + `Host`) and the full table form on
  stderr. Must cover **both backend output shapes**: the Metal shape with a
  dedicated `CPU_REPACK` row and a flat `Host` model column (M5 Max, Q4_K_XL
  35B, `--n-cpu-moe` 0/12/24/36) and the CUDA shape with no `CPU_REPACK` row
  and the growth carried in the `Host` model column (RTX 5090, Q4_K_XL 35B
  MTP, `-c 200000 -ctk/-ctv q8_0 -ctkd/-ctvd q4_0 -b 2048 -ub 256`,
  `--n-cpu-moe` 0/4/8/12/16, plus the `-ub` 512/1024 variants at `n = 0`).
  The captured sweeps step by 4 and 12 only because they were surveys;
  `n_cpu_moe` is a per-layer integer, so add unit-step points across the
  16 GiB decision boundary (`n = 17`, `18`, `19`, `20`) — see ADR §13 — and
  mark any synthesized rows as such in the fixture. Include one
  parse-failure sample. These are the
  fixtures the Phase 4b probe parser and Phase 4c fit search test against;
- `docs/plans/evidence/preset-bundles/phase-0/converted-bundle-fixture.json`
  — one legacy v4 preset converted under the ADR §5 Conversion defaults,
  fully determined;
- `scripts/validate-preset-bundle-contract.mjs`
- npm script `validate-preset-bundle-contract` in `package.json`.

The validator fails for missing files, duplicate semantic IDs, blank parity
cells, unknown source paths, missing route auth, stale baseline commit, missing
capture files, or prohibited mixed-KV entries accidentally marked allowed.

The redacted Ryne receipt records 26 source filenames, unique flags, and value
sets without secrets or complete environment dumps. It labels operator `G`
suffixes as unverified.

The v5/v6 and new API fixtures are proposed contracts, not claims that current
code accepts them. Phase 0 freezes exact field names, tagged artifact sources,
revision semantics, workload policy, curated/custom eligibility, unknown-enum
behavior, and request/response error shapes before implementation begins.

The field matrix includes `src/llama/batch_import.rs`. Add importer fixtures for
`--no-mmproj-offload`, reasoning effort/format/preserve, repeat-last-n,
no-kv-unified, no-cont-batching, and multiple flags on one line. Deferred or
unknown flags such as `--tools all -ag` and `--kv-offload` must produce bounded
diagnostics rather than silently becoming typed fields.

## Baseline captures

After `rtk cargo build --release`, run sequentially:

- `rtk node tests/ui/capture/index.mjs --scenario welcome`
- `rtk node tests/ui/capture/index.mjs --scenario rapid-preset`
- `rtk node tests/ui/capture/index.mjs --scenario preset-editor`
- `rtk node tests/ui/capture/index.mjs --scenario spawn-wizard-guided-drawer`
- `rtk node tests/ui/capture/index.mjs --scenario spawn-wizard-pro-baseline`
- `rtk node tests/ui/capture/index.mjs --scenario spawn-wizard-launch-full-config`

Record exact produced files and dimensions; do not use historical PNGs as the
baseline receipt.

## Phase gate

- `rtk npm run validate-preset-bundle-contract`
- `rtk npm run validate-wizard-groups`
- `rtk npm run validate-js`
- `rtk npm run lint`
- `rtk git diff --check`
- Fresh Verifier confirms the parity inventory against source, not only against
  the generator that produced it.

## Stop conditions

Stop if Ryne/local help cannot be captured, source and docs disagree on route
auth, the current screenshots cannot be reproduced, or any representative
field cannot be traced to argv or an explicit absence.

Suggested commit: `test(wizard): freeze preset bundle configuration contract`

---

# Phase 1a — Repair capability caching and preset persistence

## Objective

Fix two independent foundational defects that block trustworthy caching and
safe writes. This phase touches no K/V policy and no validation.

## Read first

- `src/inference/llama_cpp_capabilities.rs`
- `src/inference/rapid_mlx/capabilities.rs` (identical defect; live call sites
  in `src/web/api/rapid_mlx_runtime.rs` and `src/inference/launch.rs`)
- `src/web/api/presets.rs`
- Phase 0 runtime capability fixtures
- architecture §10 (the `help_hash` contract: stored probe evidence, never
  recreated from parsed flags at lookup)

## Root cause

Both capability caches store `help_hash = hash_help(raw --help text)` at probe
time (`llama_cpp_capabilities.rs:194`, `rapid_mlx/capabilities.rs:449`) but
`is_valid_for` re-validates it as
`help_hash == hash_help(&self.serve_flags.join(" "))`
(`llama_cpp_capabilities.rs:149`, `rapid_mlx/capabilities.rs:364`). Those two
strings differ for every real probe, so the filter at
`cached_snapshot(...).filter(|snap| snap.is_valid_for(identity))` rejects every
stored snapshot: every capability lookup re-runs the bounded probes (version +
help) even when the executable is unchanged. The existing tests at
`:585`/`:630` mask the bug by constructing `help_hash` from the same joined
`serve_flags` string they store, which makes the (wrong) comparison true.

## Tests first

Add failing tests for:

- a snapshot built from **real raw `--help` text** (not the joined flags)
  returns a cache hit on `cached_snapshot` for the unchanged identity — this
  fails against the bug and passes after the fix. It MUST build `help_hash`
  from raw help text that differs from the joined flags, or it will pass
  against the bug;
- the same test, repeated for `rapid_mlx::capabilities`;
- a changed executable (different `file_hash` or path) still misses the cache;
- failed create, update, delete, and reset each leave in-memory state unchanged
  and return non-success.

## Implementation

Modify:

- `src/inference/llama_cpp_capabilities.rs`
  - `CapabilitySnapshot::is_valid_for` (:146): validate by **exact executable
    identity only** — `executable_identity.path == current.path &&
    executable_identity.file_hash == current.file_hash`. Drop the
    `help_hash == hash_help(&self.serve_flags.join(" "))` clause. At lookup the
    only input is `ExecutableIdentity`; the raw `--help` text is not available
    there, so the raw-help hash (stored at :194) can never be recomputed at
    lookup — that mismatch is the bug. `help_hash` stays stored as probe
    evidence and keeps participating in `fingerprint()` (:156) and the spec
    verdict join key; it is never re-derived from parsed flags (this matches
    architecture §10).
  - preserve bounded probe identity/error evidence.
- `src/inference/rapid_mlx/capabilities.rs`
  - apply the identical fix: `is_valid_for` (:361) carries the same
    `help_hash == hash_help(&self.serve_flags.join(" "))` clause (raw help hash
    at :449 vs joined flags at :364). This instance is live — `cached_snapshot`
    is called from `rapid_mlx_runtime.rs:1068`, `rapid_mlx_runtime.rs:2347`, and
    `launch.rs:404` — so every rapid-mlx capability lookup currently misses.
  - keep `help_hash` in the fingerprint as evidence.
- `src/web/api/presets.rs`
  - persist successfully before mutating shared state or returning success for
    create, update, delete, and reset.

## Focused verification

- `rtk cargo test inference::llama_cpp_capabilities::`
- `rtk cargo test presets::`
- `rtk cargo test --test auth_routing`
- `rtk cargo clippy -- -D warnings`
- `rtk cargo fmt -- --check`
- `rtk git diff --check`

## Hard gate

A snapshot generated from real `--help` output validates on lookup without
regeneration for the unchanged identity (cache hit), in **both** llama.cpp and
rapid-mlx modules. A changed executable (different `file_hash` or path) misses
the cache and re-probes. `help_hash` remains stored and fingerprinted as probe
evidence; the lookup never re-derives it from `serve_flags` (architecture §10).
No preset mutation path reports success while leaving the on-disk file
unwritten.

## Stop conditions

Stop if making the cache hit requires re-running the bounded probe on every
lookup (the bug), or if the only way to reconcile generation/lookup is to
change *what* is hashed rather than *where* it is validated. The intended fix
removes the `help_hash` re-derivation from `is_valid_for`; it does not change
`help_hash`'s definition.

Suggested commit: `fix(inference): validate capability snapshots by executable identity`

---

# Phase 1b — Canonical K/V truth and central launch validation

## Objective

Establish one validation path for llama.cpp launch policy before any bundle
state or new control depends on it.

## Read first

- `src/presets/mod.rs:14-81,97-330,388-565`
- `src/inference/launch.rs:114-289`
- `src/inference/llama_cpp.rs:420-823`
- `src/web/api/doctor.rs`
- `src/calibration/candidates.rs:366-392` (K/V patch application; note `ctk`/`ctv`
  are the only cache fields this file touches)
- `src/llama/batch_import.rs:425-443,675-690,705-715` (argv importer; see the
  known defect below)
- Phase 0 schema and field-parity fixtures
- Phase 1a receipt

## Tests first

Add failing tests for:

- v4 -> v5 K/V migration for empty, equal, and conflicting field pairs;
- Doctor reads canonical `ctk/ctv`;
- unknown K/V value rejection;
- `q8_0/q4_0` rejection when the capability snapshot does not advertise mixed
  K/V support (see the K/V value reference below). The gate is `mixed_main_kv`
  on the snapshot: the architecture section 6 struct whose value includes
  `supported`, `reason`, and `source` — not a bare boolean. The `reason` is
  load-bearing, not decoration: invariant 17 requires an unavailable option to
  render disabled *with a reason*, and a boolean has nowhere to carry one. **There is no detector for it
  in this plan** — architecture §2.3 defers detecting `GGML_CUDA_FA_ALL_QUANTS`
  builds, and mixed-pair support is a fused-attention kernel property that
  `--help` cannot prove: `-ctk` and `-ctv` advertise their value lists
  independently, so both listing `q8_0` and `q4_0` says nothing about the pair.
  Therefore `mixed_main_kv` is constructed with `supported: false`, a constant
  reason string, and a `source` naming this rule, and stays that way for every
  binary this product launches. Write the gate as a capability read, not
  a hardcoded reject, because architecture §"Configure drawer binding behavior"
  requires the option to become selectable with no UI change once a detector
  lands. **Do not detect it from the version string.** `llama-server --version`
  emits only `version`, build number, commit, compiler, and target triple — no
  compile-flag information. Two custom builds in use today report
  `0.3.0-dev (build 1058, commit cc83d7b48) … Darwin arm64` and
  `0.3.0-dev (build 10666, commit 4e97ac86e) … Windows AMD64`, differing in
  build counter and upstream commit. A "was this custom-compiled?" heuristic
  returns true for the Darwin build, where `GGML_CUDA_FA_ALL_QUANTS` is a
  CUDA-only define that cannot exist — so the heuristic false-positives into a
  guaranteed load failure. Any eventual detector must establish the kernel
  property itself (operator declaration per binary path, or a bounded probe),
  never infer it from version, build number, or filesystem location; Cover the accept branch with a test that sets the field directly on a
  fixture snapshot; do not add a code path that infers it;
- `extra_args` rejection for `-ctk`, `-ctv`, long aliases, and duplicate typed
  safety flags;
- `--cache-ram` resolves to `0` on macOS regardless of stored `cache_ram_mib` or
  `cache_mode`, and passes the configured value through unchanged on Windows and
  Linux. Cover a stored `cache_ram_mib: Some(16384)` preset on both platform
  branches, and assert the macOS branch also suppresses `--cache-idle-slots`,
  which `--help` documents as requiring cache-ram;
- v4 `null`/zero batch fields retain omitted-argv/runtime-default behavior;
- explicit bundle performance options reject zero, while flat legacy
  `ubatch > batch` is rejected only when both values are nonzero;
- negative, dense-model, and over-layer-count `n_cpu_moe` rejection when
  metadata is authoritative;
- unsupported typed flags return actionable capability errors;
- conflicting legacy fields remain preserved/non-launchable, not deleted.

Use real legacy JSON fixtures under:

- `tests/fixtures/presets/schema-v4/`
- `tests/fixtures/presets/schema-v5/`

## K/V value reference

Verified against `llama-server --help` on the reference build. Main and draft
K/V accept the **same** value set; do not write two different enums:

```text
f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1     (default: f16)
```

Each option has one canonical long form plus aliases. Every one of these
spellings refers to the same option and must be recognized wherever argv is
parsed:

| Canonical (emit this) | Aliases (accept these) | Preset field |
|---|---|---|
| `--cache-type-k` | `-ctk` | `ctk` |
| `--cache-type-v` | `-ctv` | `ctv` |
| `--spec-draft-type-k` | `-ctkd`, `--cache-type-k-draft` | `spec_draft_type_k` |
| `--spec-draft-type-v` | `-ctvd`, `--cache-type-v-draft` | `spec_draft_type_v` |

Draft K/V is independent of main K/V. It defaults to `f16`, and quantizing it
to `q4_0/q4_0` is a supported, field-observed configuration with no measured
draft-acceptance loss. Main-K/V policy — including the mixed-pair rule — applies
to main K/V only. Do not extend it to the draft pair, and do not derive draft
values from main values.

## Known defect — argv importer drops K/V

`src/llama/batch_import.rs` reaches the shared K/V validator through
`POST /api/spawn-wizard/import-launch-file`, so it is bound by rule 5 of the
architecture's migration rules. It is currently broken in two ways:

- there is **no match arm for main K/V in any spelling**, and `ctk`/`ctv` are
  hardcoded to `"f16"` at `:712-713`;
- draft K/V matches only the canonical long forms, not the aliases above.

Every unmatched flag falls through `_ =>` at `:675` into `extra_args` as raw
text. Importing a real launch script therefore yields typed fields claiming
`f16` while `extra_args` carries the actual `-ctk`-family values — precisely the
override state the architecture forbids, reached without any user error.

Fix this as part of this phase: add main-K/V match arms, accept every alias in
the table above, drop the hardcoded `f16`, and route imported values through the
same validator as every other call site. Add a fixture that imports a launch
script setting all four K/V options via aliases and asserts the four typed
fields are populated and `extra_args` is empty.

### Same family: `fit_ctx` imports into a field no surface can show

`--fit-ctx` parses at `batch_import.rs:578-581` and persists to
`preset.fit_ctx`, but no wizard or preset-editor control exists for it —
`spawn-fit-target` and `modal-fit-target` cover `--fit-target` only. Emission at
`llama_cpp.rs:836-840` is `if fit_target { … } else if fit_ctx { … }`, so a
preset carrying both silently drops `fit_ctx` from argv.

An imported script using `--fit-ctx` therefore produces a stored value that is
invisible, uneditable, and conditionally ignored. Either register a control for
it alongside `spawn-fit-target`, or refuse the flag at import with a stated
reason — do not keep parsing it into a field with no surface. Whichever is
chosen, the precedence between `fit_target` and `fit_ctx` must be stated in the
resolve preview rather than left implicit in emission order.

No frontend calls this endpoint today, so neither defect is a live user-facing
regression. Do not build UI for the importer in this phase.

## Implementation

Modify:

- `src/presets/mod.rs`
  - bump to schema v5;
  - implement canonical K/V migration;
  - retain deprecated projection only for compatibility;
  - surface migration conflicts without destructive rewrite.
- Add `src/presets/validation.rs`
  - pure structural/product-policy validation with no binary lookup;
  - canonical K/V policy and base llama.cpp validation;
  - typed `ValidationIssue { field, code, message, repair }`.
- `src/inference/launch.rs`
  - llama.cpp branch calls shared validation;
  - direct and preset launch share it;
  - resolve `--cache-ram` to `0` on macOS regardless of stored `cache_ram_mib`
    or `cache_mode`, following the `cfg!(target_os = "macos")` precedent already
    at `:200`. Windows and Linux pass the configured value through unchanged.
    `--cache-idle-slots` already gates on `cache_ram_mib != Some(0)` in
    `llama_cpp.rs:806-815`, so suppressing it on macOS falls out of this and
    needs no separate branch.
- `src/inference/llama_cpp.rs`
  - reject safety-critical duplicates in `extra_args` before argv construction.
- `src/web/api/doctor.rs`
  - use canonical policy.
- `src/web/api/vram.rs`, `src/calibration/`, `src/llama/batch_import.rs`
  - call the same K/V validator where applicable.
- Update `docs/reference/api.md` and `docs/reference/inference-tuning.md` as if
  the canonical policy always existed.

## Focused verification

- `rtk cargo test presets::`
- `rtk cargo test inference::llama_cpp::`
- `rtk cargo test calibration::`
- `rtk cargo test llama::batch_import::`
- `rtk cargo test --test auth_routing`
- `rtk cargo clippy -- -D warnings`
- `rtk cargo fmt -- --check`
- `rtk git diff --check`

## Hard gate

No request accepted by normal preset save, direct session spawn, estimator,
Calibration, Doctor, or import may treat `q8_0/q4_0` as valid on a snapshot
whose `mixed_main_kv.supported` is `false` — which is every snapshot any
shipped binary produces. The only place the accept branch may be reached is the fixture
test that sets the field directly. A source grep for K/V validation call sites
must match the Phase 0 inventory, and a grep for `mixed_main_kv` must find
exactly one construction site (constant `supported: false`), one gate read, and
the fixture test. Any other write to that field is a stop condition.

Legacy flat zero/null batch fixtures must produce identical argv before and
after v5 migration. The nonzero requirement applies only to explicit new bundle
performance choices.

## Stop conditions

Stop if migration would overwrite a partially readable preset file, if direct
spawn bypasses validation, or if `extra_args` can still override K/V.

Suggested commit: `fix(presets): enforce canonical llama launch validation`

---

# Phase 2 — Add v6 typed fields and bundle schema

## Objective

Add versioned backend types for the missing llama.cpp settings and optional
preset bundle without changing welcome-screen behavior yet.

## Read first

- Architecture sections 4-10
- v5 fixtures and Phase 1b receipt
- `ModelPreset`, `ServerConfig`, `request_from_preset()`, argv tests
- `src/inference/llama_cpp_capabilities.rs` typed capability patterns
- `src/llama/gguf_meta.rs` and `src/models/mod.rs` filename/metadata parsing

## Tests first

Freeze `tests/fixtures/presets/schema-v6/` for:

- legacy one-artifact preset with `bundle: null`;
- Q4/Q5 exact-tune bundle;
- local and Hugging Face tagged artifact sources with revision, digest coverage,
  quantization provenance, safe API source view, and companion references;
- bundle with contexts 160k/200k/262144;
- explicit 2048/256 and 4096/4096 performance choices;
- MoE choices 0/6/16 with authoritative layer bound;
- missing artifact, duplicate ID, conflicting flat projection;
- mmproj offload default/on/off;
- all llama reasoning-effort enum values;
- reasoning-format `None` plus known explicit values; prove `None` emits no
  `--reasoning-format auto`;
- reasoning-preserve absent/false/true. `--reasoning-preserve` is a **valueless
  flag**: it is emitted with no argument or not at all. Render it as
  `Some(true)` -> emit `--reasoning-preserve`; `None` -> emit nothing;
  `Some(false)` -> emit nothing, unless the capability snapshot advertises a
  `--no-reasoning-preserve` counterpart, in which case emit that. Never emit
  `--reasoning-preserve true` or `--reasoning-preserve false`. Both idioms
  already exist in `src/inference/llama_cpp.rs` — paired
  (`--kv-unified`/`--no-kv-unified`, `--cache-idle-slots`/`--no-cache-idle-slots`)
  and presence-only (`--no-warmup`, `--no-cont-batching`) — so choose by
  capability evidence, not by analogy to whichever one you read first.
  `Some(false)` and `None` may render to identical argv while remaining
  distinct in the selection fingerprint; that is intended. Do not collapse
  `Some(false)` into `None` to make argv comparison simpler;
- unknown future enum values use bounded `Unknown(String)`, remain readable and
  round-trip through unrelated edits, but are non-launchable;
- persisted workload policies and curated versus validated-custom eligibility.
- runtime validation rejects/provides unavailable reason for `n_cpu_moe` on
  unified-memory systems until separate behavior is qualified.

Add round-trip tests proving JSON -> Rust -> JSON preserves exact integer and
enum values and never converts explicit values to runtime defaults.

- fit default on creation: a bundle created through the v6 constructor stores
  `fit_enabled: Some(false)`; a v5 preset migrated to v6 keeps its stored value,
  including `null`, and its argv is unchanged. Assert both in the same test so
  the forward-only boundary is visible in one place.

## Implementation

Add:

- `src/presets/bundle.rs`
  - `PresetBundleSpec`, identity, fully specified tagged artifact source,
    digest, quantization, companion, performance, selection, fit intent,
    workload policy, curated/custom combination policy, and revision types;
  - bounded unknown enum handling;
  - **the single server-owned v6 bundle constructor**, and it is the only
    function in the tree that may set `fit_enabled` on a new bundle. It sets
    `fit_enabled: Some(false)`. Every later surface — wizard, editor
    conversion, bundle copy — goes through it rather than seeding the field
    itself, which is what makes "fit off going forward" one line instead of a
    rule each caller has to remember. Migration is a separate path and does not
    call it: a v5 preset arriving at v6 keeps whatever it stored, including
    `null`. Phase 5 verifies this value over the wire and is forbidden from
    implementing it;
  - structural validation only; no hardware-dependent resolution.
- `src/presets/resolver.rs`
  - types and stub exact-selection resolver interface;
  - no intent algorithm yet.
- `src/presets/mod.rs`
  - schema v6 and optional `bundle`;
  - v5 -> v6 migration with `bundle: None`;
  - typed llama fields.
- `src/inference/llama_cpp.rs` and `src/inference/launch.rs`
  - `mmproj_offload: Option<bool>`;
  - backend-distinct llama reasoning effort/format/preserve types;
  - typed argv emission gated by capability snapshot.
- `src/inference/llama_cpp_capabilities.rs`
  - typed feature and enum/default evidence for the new flags.
- `src/llama/batch_import.rs`
  - parse every Phase 0 importer fixture into typed fields or explicit bounded
    diagnostics;
  - prohibited mixed KV imports are preserved but marked invalid.

Add an explicit runtime validator that receives `&CapabilitySnapshot` (or a
binary-scoped provider) immediately before resolve/spawn. Do not add global
capability lookup to pure schema migration or persistence validation.

Native reasoning-preserve requires exact binary support and compatible
reasoning mode. Because current code has no authoritative template capability
contract, implement bounded template inspection plus fixtures or report
template compatibility `unknown` and block launch. Do not equate it with
`preserve_thinking` template kwargs.

Do not reuse Rapid-MLX `reasoning_effort`; it is a request-default field with
different runtime meaning.

## Focused verification

- `rtk cargo test presets::`
- `rtk cargo test inference::llama_cpp::`
- `rtk cargo test inference::llama_cpp_capabilities::`
- `rtk cargo test llama::batch_import::`
- `rtk cargo clippy -- -D warnings`
- `rtk cargo fmt -- --check`
- `rtk git diff --check`

## Migration direction

Migration is forward-only. A v5 or v6 file cannot be read by an earlier build
and no downgrade is written. The current user base makes this acceptable: a
handful of operators, only one of whom is on v2. Record it in the receipt so a
later phase does not assume a rollback exists — the escape hatch for a bad
release is the Phase 8a render flag, not a schema downgrade.

## Hard gate

Old presets launch identically after v6 migration. New fields emit no argv when
absent. Bundle presence alone does not change card count or launch behavior.

A bundle built by the v6 constructor carries `fit_enabled: Some(false)`, and
`rg -n 'fit_enabled\s*:' src/presets/` shows exactly one assignment site for
new bundles. A second site is the defect this gate exists to catch, because the
duplicate will drift.

An unsupported stored value is preserved through API read/unrelated edit/save,
cannot be newly selected, and cannot launch until the exact runtime supports it.

## Stop conditions

Stop on projection ambiguity, unbounded enum/string fields, unknown artifact
identity, or any attempt to infer bundle membership automatically.

Suggested commit: `feat(presets): add typed launch bundles and reasoning flags`

---

# Phase 3 — Implement resolver, revisioned selection, and authenticated APIs

## Objective

Make one server-side resolver authoritative for preview, save, and spawn.

## Read first

- Architecture sections 5-8
- Phase 2 schema fixtures/receipt
- `src/web/api/presets.rs`
- `src/web/api/sessions.rs`
- `src/web/api/vram.rs`
- `tests/auth_routing.rs`

## Tests first

Add backend/API tests for:

- exact selection resolves to expected flat fields;
- flat projection equals saved default selection;
- conflicting submitted projection returns 400 with field paths;
- unknown artifact/performance/KV policy/context returns 400;
- an unknown-string `ctk`/`ctv` value outside the named KV policies round-trips
  verbatim through read/unrelated-edit/save and remains non-launchable until the
  capability-validated KV policy accepts it; it is never a migration "conflict";
- stale `expected_revision` returns 409;
- stale full PUT versus selection PATCH, reset, delete, copy, and conversion all
  return 409 without changing state;
- revision is assigned/incremented by the server and cannot be replaced by a
  client value;
- resolve performs no write;
- PATCH persists before changing memory;
- `POST /selection` with a client-supplied `intent_source` persists it as
  `None` and the API read returns it absent;
- Start once resolves an override without changing the saved selection;
- Save & Start cannot start if save fails or revision conflicts;
- session records the resolved `selection_hash`;
- every selection in `tests/fixtures/presets/fingerprint_golden.json` hashes to
  its committed literal `sel-v1:` value;
- a redacted `api_key` and a changed preset revision each leave `selection_hash`
  unchanged, and changing any single memory-relevant or behavior-only field
  changes it;
- one test named `same_selection_same_fingerprint_across_surfaces` drives all
  four call sites named in the hard gate and asserts a single shared
  fingerprint value. The hard gate is this test; do not satisfy it with four
  separate per-surface assertions that never compare against each other;
- all new routes require correct tokens under no-auth and form/basic-auth modes;
- Rapid-MLX and legacy presets remain unchanged/non-applicable.

## Implementation

Complete:

- `src/presets/resolver.rs`, with these exact signatures. Types are given here
  so they are transcribed, not invented; four other phases depend on them.

  ```rust
  /// Resolve a preset plus an optional one-shot selection into a launchable
  /// flat configuration. Pure with respect to the filesystem: it performs no
  /// write and no network or GGUF introspection (see the bounded-resolve rule
  /// below).
  pub fn resolve_preset(
      preset: &ModelPreset,
      selection: Option<&PresetBundleSelection>,
      capabilities: &CapabilitySnapshot,
  ) -> Result<ResolvedLaunch, Vec<ValidationIssue>>;

  /// Project a bundled preset's saved default selection into the same flat
  /// shape `resolve_preset` produces, without applying a one-shot selection.
  /// Must return byte-identical output to `resolve_preset(preset, None, caps)`.
  pub fn materialize_default_projection(
      preset: &ModelPreset,
      capabilities: &CapabilitySnapshot,
  ) -> Result<ResolvedLaunch, Vec<ValidationIssue>>;

  pub struct ResolvedLaunch {
      /// Flat, fully materialized preset the launcher consumes.
      pub preset: ModelPreset,
      /// `sel-v1:<sha256 hex>` — identifier 1 in the architecture doc, §7.
      pub selection_hash: String,
      /// `cfg-v1:<sha256 hex>` — identifier 2 in the architecture doc, §7. The
      /// drawer's `Start without saving` / `Save & Start` echo this back as
      /// `expected_resolved_config_hash`; without it the consent guard in §7
      /// cannot be implemented. The collapsed card's atomic Start does NOT send
      /// it. Do not name either field a bare `fingerprint` — §7
      /// opens by rejecting exactly that overload.
      pub config_hash: String,
      /// Every policy name that was resolved to a concrete value.
      pub changes: Vec<ResolvedChange>,
      /// Populated in Phase 4a; `None` here.
      pub estimate: Option<LaunchEstimate>,
      /// Populated in Phase 9; `None` here.
      pub evidence: Option<EvidenceMatch>,
  }
  ```

  `ResolvedChange` is the architecture section 8 type
  (`{ code, field, before, after, explanation, source_policy }`) and is reused
  unchanged. Do not restate it here and do not define a second change type: the
  drawer's diff list needs `before` to render "32k -> 16k" rather than a bare
  "16k", `source_policy` is `Option` because an explicit user choice has no
  originating policy, and Phase 4a's tests assert that exact field set.

  `ValidationIssue` is the Phase 1b type (`{ field, code, message, repair }`)
  and is reused unchanged. Do not introduce a second error type here.

### Selection and configuration hashes — where the algorithm lives

The normative algorithm is defined once, in
`docs/plans/20260830-preset_bundle_architecture.md`, section 7, under the
heading `### Three identifiers with three different jobs`: the three
identifiers, their input field sets, their exclusions, the canonical
`[path, type, value]` encoding, and the SHA-256 rendering. Implement it from
there verbatim. Do not restate it in this document, do not re-derive it, and do
not restate the digest length here — two copies of a hash definition drift, and
a drifted hash fails open by matching nothing. A `sel-v1:` computed at one
length and compared against one computed at another never matches, which turns
every Start into a permanent 412. If the architecture text and this plan appear to disagree, the
architecture text wins, and the disagreement is a defect to report rather than
a judgment call to resolve.

What stays here is the evidence that the implementation matches that
definition.

Add `tests/fixtures/presets/fingerprint_golden.json` containing at least three
saved selections and their expected fingerprint strings, committed as literal
values. At least one must be a bundled preset with a non-empty `tensor_split`
and at least one must carry a redacted `api_key`, proving the secret is
excluded. A test that recomputes an expected value instead of asserting the
committed literal does not satisfy this requirement. If an intended change to
the field set makes these fixtures fail, update the fixture and the version
prefix together in one commit and say so in the receipt; never edit a golden
value alone to make a test pass.
- Add `src/web/api/preset_bundles.rs` holding both new routes:
  - `POST /api/presets/{id}/resolve`;
  - `PATCH /api/presets/{id}/selection`.

  This location is decided, not left to the implementer. Phases 7 and 10 both
  assume bundle handlers live in one named file, and `presets.rs` is already
  335 lines of unrelated CRUD. Do not merge these handlers into `presets.rs`
  even if that looks smaller at the time.
- `src/web/api/sessions.rs`
  - optional selection/revision in spawn body;
  - always resolve and validate server-side;
  - retain `db-admin-token` requirement.
- `src/web/api/mod.rs` and route inventory/auth tests.
- `docs/reference/api.md` request/response examples.

Require `expected_revision`, phased so the current
editor keeps working until Phase 7:

- the new bundle routes (`POST /selection`, `POST /copy`,
  `POST /convert-to-bundle`, `DELETE`, `POST /reset`) require it from day one;
- the existing `POST /api/presets` and `PUT /api/presets/{id}` accept their
  current shape in this phase; a `PUT` on a preset with a non-null `bundle`
  requires `expected_revision` (no shipped UI can create bundled presets
  before Phase 7, so this breaks no real client);
- every mutation response returns the new `revision` and a fresh
  `catalog_etag`.

Define reset/delete/copy/convert revision semantics in the frozen API
fixtures.

Use `safe_json_body`, `#[serde(default)]`, bounded arrays/strings, and existing
secret redaction.

**Bounded resolve — decided, not conditional.** `resolve_preset` performs no
GGUF parsing, no filesystem read, and no network call. Every input it needs is
already in the preset or the passed `CapabilitySnapshot`. Where a check would
require GGUF-derived facts — the MoE layer-count bound from Phase 1b is the
live example — the resolver consumes a value already recorded on the preset and
returns a `ValidationIssue` when that value is absent, rather than reading the
file to find it. This makes resolve constant-time in practice and removes the
need for a size limit or timeout on this path. If a later phase genuinely needs
introspection during resolve, that is an architecture revision for the
Coordinator, not a local decision to add a timeout here.

## Focused verification

- `rtk cargo test presets::resolver::`
- `rtk cargo test web::api::presets::`
- `rtk cargo test web::api::sessions::`
- `rtk cargo test --test auth_routing`
- `rtk cargo clippy -- -D warnings`
- `rtk cargo fmt -- --check`
- `rtk git diff --check`

## Hard gate

Welcome Start, direct spawn, saved default spawn, and resolve preview produce
the same normalized flat configuration and fingerprint for the same selection,
proven by the single `same_selection_same_fingerprint_across_surfaces` test
named above, and the committed golden fingerprints still hold.
Calibration in this release operates only on the successfully saved default
selection and resolves it through the same path. Draft/Start-once Calibration
is deferred until its request carries an exact selection plus expected
revision/fingerprint.

An exact saved selection remains launchable when estimate/evidence enrichment
is unavailable. The response reports degraded/no-estimate state; only an intent
proposal that depends on sizing may return unavailable.

## Stop conditions

Stop if the client can submit a resolved flat preset as authority, if preview
and spawn disagree, if revisions are last-write-wins, or if persistence failure
still permits start.

Suggested commit: `feat(api): resolve revisioned preset bundle selections`

---

# Phase 4a — Deterministic intent proposals

## Objective

Implement Quality-first/Balanced/Low-VRAM proposals as a pure deterministic
function, and return exact estimate and reason data for the Configure drawer.
No `llama-fit-params` work in this sub-phase: the probe arrives in 4b. If a
task in front of you needs the probe, it belongs to a later sub-phase.

## Read first

- Architecture sections 6 and 11
- `src/web/api/vram.rs`
- `src/llama/vram_estimator/estimate.rs`
- `src/llama/spawn_wizard.rs`
- `static/js/features/vram-estimate.js` only to confirm request vocabulary
- GGUF metadata paths in `src/llama/gguf_meta.rs`

## Tests first

Add real metadata fixtures for dense, MoE, hybrid DeltaNet, vision, and unknown
models. Test:

- intent output is a complete exact selection;
- every change has code, field, before, after, and explanation;
- lower artifact choice precedes context reduction for intents that permit it;
- Low-VRAM never changes the user's selected context automatically; when no
  safe artifact, performance, or MoE placement change remains, it returns a
  no-change result with the manual context/model-variant tradeoff;
- ubatch never exceeds batch;
- MoE CPU placement is used only for proven MoE models and within bounds;
- agentic/tool workload policy never silently selects q4/q4;
- mixed q8/q4 is never proposed;
- model native context and image-token constraints are honored;
- workload policy is persisted, fingerprinted, and enforced after reopen;
- the Low-VRAM *intent* returns an unavailable reason for automatic
  `n_cpu_moe` on unified-memory systems;
- unknown/degraded metadata returns fewer options and an explanation, not a
  confident guess;
- local artifacts use exact size/digest; synthesized QuantOption sizes are
  limited to pre-download advice;
- curated-only bundles reject unlisted combinations; validated-custom bundles
  may recombine axes only after every resolver constraint passes and remain
  labelled unmeasured/unproven.

## Implementation

- Add `src/presets/intent.rs` with a pure deterministic proposal function.
- Name the frozen Phase-0 `/api/vram-estimate` response shape (the existing
  weights/KV/extras/VRAM/RAM breakdown in `api-target-fixtures.json`)
  `LaunchEstimate`; the resolve response embeds that exact frozen shape or an
  unavailable tag. No new estimate fields are invented for the bundle.
- Reuse `/api/vram-estimate` backend calculations. Do not duplicate formulas.
- Document intent ordering and degraded behavior in
  `docs/reference/vram-estimator.md`.

## Focused verification

- `rtk cargo test presets::intent::`
- `rtk cargo test llama::vram_estimator::`
- `rtk cargo clippy -- -D warnings`
- `rtk cargo fmt -- --check`
- `rtk git diff --check`

## Hard gate

Every proposal can be replayed as an exact selection with the identical
fingerprint and estimate inputs. Start never reruns intent automatically.

## Stop conditions

Stop on client-side formulas, inferred aggressive KV, use of operator filename
memory labels, or intent output that depends on unordered iteration.

Suggested commit: `feat(vram): resolve explicit preset fit intents`

---

# Phase 4b — Fit probe: identity, invocation, and parsers

## Objective

Get one correct reading out of `llama-fit-params` for one
`(resolved configuration, n_cpu_moe)` pair. There is no search in this
sub-phase. Success is: hand it a config and an `n`, get back a `FitReading`
whose device and host totals match the Phase 0 fixture by hand-check.

## Read first

- Architecture section 12, in full
- Phase 0 `fit-probe-output-fixtures.json`
- `src/inference/llama_cpp_capabilities.rs` for the existing binary-identity
  and bounded-probe patterns — follow them, do not invent a second style
- `src/config.rs` for where `AppConfig` fields are declared

## The seam this sub-phase must produce

Declare these in `src/presets/fit_probe.rs` and treat them as the contract
Phase 4c is written against:

```rust
/// One probe reading at a single n_cpu_moe. All figures in MiB.
pub struct FitReading {
    pub n_cpu_moe: u32,
    pub device_total_mib: u64, // sum of the device rows
    pub host_total_mib: u64,   // sum of EVERY non-device row
    pub model_mib: u64,
    pub context_mib: u64,
    pub compute_mib: u64,
}

/// The only capability Phase 4c is permitted to depend on.
pub trait FitReader {
    fn read(&mut self, n_cpu_moe: u32) -> Result<FitReading, FitProbeError>;
}
```

Two implementations ship in this sub-phase:

1. `ProcessFitReader` — spawns the binary, caches by
   `(artifact digest, resolved-config probe subset, probe SHA-256, n)`.
2. `FixtureFitReader` — reads `fit-probe-output-fixtures.json`. This is a
   first-class deliverable, not test scaffolding: it is what lets 4c be
   written and verified with no binary present on the machine.

## Tests first

- both parsers round-trip every Phase 0 fixture: the two-line compact form on
  stdout and the full table form on stderr;
- host total equals the sum of every non-device row, computed by row class,
  not by looking for a row named `CPU_REPACK`;
- both backend table shapes yield the same host budget — the Metal
  `CPU_REPACK` fixture and the CUDA `Host`-model-growth fixture at equivalent
  offload must produce equal host totals;
- device budget is taken against `total` minus the reserve, never against the
  `free` column;
- the parse-failure fixture yields a disabled-with-reason error, never a
  partial reading and never a zero;
- identity verification rejects a wrong SHA-256, a changed mtime, and a
  missing version line, each with a distinct reason;
- an absent `llama_fit_params_path` is the default and yields
  `probe_unavailable`, not an error path that fails the request;
- the per-probe timeout fires and is reported as such;
- a repeated `read` at the same `n` is served from cache without re-invoking
  the binary.

## Implementation

- Add the `AppConfig` field `llama_fit_params_path`, absent by default.
- Identity verification: canonical path, SHA-256, mtime, version line.
- Fixed invocation contract: `--fit off`, `-lm none`, `-lv 4`, `-fitp on`,
  `-m`, `-c`, `-ctk`/`-ctv`/`-ctkd`/`-ctvd`, `-b`/`-ub`, `--n-cpu-moe`.
- Bounded capture of **both** streams. The compact rows are on stdout; the
  host budget is only derivable from the stderr table. Capturing one stream
  is the most likely silent defect in this sub-phase.
- Compute the host budget as the sum of every non-device row. That row set
  differs by backend: `CPU_REPACK` exists under Metal and does not exist
  under CUDA, where the same weights appear as growth in the `Host` row's
  `model` column (ADR §12). A `CPU_REPACK`-keyed parser reads a flat host
  budget on CUDA and will accept placements the host cannot hold.
- Budget the device against `total` minus the reserve so a cached proposal
  does not change with unrelated desktop GPU usage. Surface a low `free` as a
  separate advisory.
- Bind the reserve to the product's existing per-device fit reserve (the
  `--fit-target` default of 1024 MiB from ADR §6). Do not introduce a second,
  probe-local reserve constant.
- Tag every reading `method = "fit_probe"`. It is estimate-class: it never
  enters `memory_peak_bytes` and never produces a measured match class.

## Focused verification

- `rtk cargo test presets::fit_probe::parse`
- `rtk cargo test presets::fit_probe::identity`
- `rtk cargo clippy -- -D warnings`
- `rtk cargo fmt -- --check`
- `rtk git diff --check`

## Hard gate

`FixtureFitReader` answers every Phase 0 fixture point without a binary on the
machine. Demonstrate it in the receipt with a test that runs under
`llama_fit_params_path` unset.

## Stop conditions

Stop if the host budget is being read from a single named row, if only one
output stream is captured, or if a parse failure produces a number rather than
a reason.

Suggested commit: `feat(presets): add bounded llama-fit-params probe`

---

# Phase 4c — Two-sided placement search

## Objective

Given a `FitReader`, propose the smallest `n_cpu_moe` that fits both the
device and host budgets. This sub-phase is a pure function over the 4b seam:
no process spawning, no HTTP, no estimator changes. `src/presets/fit_probe.rs`
may not gain a process dependency here, and the search must be exercised
entirely through `FixtureFitReader`.

This is the highest-risk sub-phase in the plan. The algorithm is given below;
implement it as written rather than deriving one.

## Read first

- Architecture sections 12 and 13
- The `FitReader` seam from Phase 4b
- Phase 0 `fit-probe-output-fixtures.json`, including the unit-step points at
  `n = 17, 18, 19, 20`

## The algorithm

```text
search(reader, n_max, device_budget, host_budget, reserve):

  # device total FALLS as n rises  -> "device fits" is a SUFFIX of [0, n_max]
  # host   total RISES as n rises  -> "host fits"   is a PREFIX, anchored at 0

  n_dev_min  = smallest n in [0, n_max] with
                 device_total(n) + reserve <= device_budget
               (None if no n qualifies)

  n_host_max = largest  n in [0, n_max] with
                 host_total(n) <= host_budget
               (None if not even n = 0 qualifies)

  if n_host_max is None: return Unavailable{ host_limited,   deficit at n = 0 }
  if n_dev_min  is None: return Unavailable{ device_limited, deficit at n_max }
  if n_dev_min > n_host_max:
      return Unavailable{ disjoint, naming BOTH sides' deficits }

  return Proposal{ n_cpu_moe: n_dev_min }   # the interval lower bound
```

Two separate binary searches, each over its own monotone predicate, each
anchored at its own end. Do not write one search for "the smallest n that fits
both": that combined predicate is not monotone, and on the interval fixture it
false-reports unavailable.

Searching either side from the wrong end inverts the result and can return an
infeasible proposal. If you find yourself reasoning about which end is which,
the two comment lines at the top of the block above are the answer.

## Worked trace to check your implementation against

CUDA fixture family, `16384` MiB device, `1024` MiB reserve, `-ub 256`:

```text
  n = 17 -> device 15629 + 1024 = 16653 > 16384  -> rejected
  n = 18 -> device 15165 + 1024 = 16189 <= 16384 -> accepted
  no smaller n is accepted, so n_dev_min = 18
  host at n = 18 is within budget, so n_host_max >= 18
  proposal = 18
```

An implementation that steps by 4 proposes `20`, is feasible, and passes every
other test in this sub-phase. `n = 18` is the test that catches it.

## Tests first

- the two-sided search returns the smallest `n_cpu_moe` that fits within the
  per-device and host reserves, degrades to a disabled-with-reason state when
  the probe is absent or unparseable, and never auto-applies;
- **interval-shaped feasible region**: a fixture where the device fits only
  for `n >= 12` and the host fits only for `n <= 24` proposes `12`. A single
  binary search for "smallest n fitting both" false-reports unavailable here;
- **disjoint region**: a fixture where `n_dev_min > n_host_max` returns
  unavailable with a reason naming both sides' deficits;
- **monotonicity direction per side**: a fixture that is host-limited (host
  row already over budget at `n = 0`) reports `host_limited` and does not
  propose a higher `n`;
- **the 16 GiB target case**: the CUDA fixture family against a `16384` MiB
  device with the `1024` MiB reserve proposes `n = 18` — the exact interval
  lower bound, not `n = 17` (infeasible) and not `n = 20` (feasible but
  over-offloaded). This is the regression test for unit-step search;
- **reserve monotonicity**: raising the reserve moves the proposal
  monotonically upward — the 16 GiB fixture proposes `n = 18` at a 1024 MiB
  reserve and a strictly higher `n` at a 3 GiB reserve;
- a search that exhausts the total wall-clock budget returns a partial result
  or `probe_unavailable`, never a proposal;
- the search visits each `n` at most once for a given reader.

## Implementation

- Search over every integer in `[0, moe_layer_count]`. The reference sweeps in
  ADR §13 step by 4 and 12 because they were surveys, not because the
  parameter is coarse; a coarse search over-offloads at the boundary.
- Do not fit a slope. The `n = 0 -> 1` transition carries a fixed
  enable-offload cost that later layers do not (device `compute` steps once at
  `n > 0` then stays flat, and the first step's model delta exceeds the steady
  per-layer delta). Search by monotonicity only.
- Take the reserve as a search parameter, defaulting to the `--fit-target`
  1024 MiB, so the drawer's headroom target re-runs this same search against a
  larger reserve rather than needing a second code path.
- Expose a single-point entry alongside the search: given an explicit
  `n_cpu_moe`, return that point's device total, host total, and headroom
  against the budget without running a search. This backs the drawer's
  measured custom value (ADR §3.2). It shares the 4b cache, so a point the
  search already visited costs nothing.
- A single probe returns near-instantly, so the total search (up to
  `2 · ceil(log2(N + 1))` serial probes — 12 at `N = 48`) is expected to
  complete well inside one drawer interaction. Still give the search a total
  wall-clock budget alongside the per-probe timeout, as a guard against a hung
  or pathological binary rather than as an expected path. On exhaustion,
  return the best bracket found so far as an explicitly partial result, or
  `probe_unavailable` when no feasible `n` has been confirmed — never a silent
  truncation presented as a proposal.
- Gate the placement search on `moe_layer_count` being present. Do not gate
  the probe itself on it; that distinction is Phase 4d's concern.

## Focused verification

- `rtk cargo test presets::fit_probe::search`
- `rtk cargo clippy -- -D warnings`
- `rtk cargo fmt -- --check`
- `rtk git diff --check`

## Hard gate

The whole sub-phase's test suite passes with no `llama-fit-params` binary
present. The `n = 18` case and the interval case are both named in the receipt
with their asserted values.

## Stop conditions

Stop if a single combined-predicate search appears, if either side is searched
from the wrong end, if the step is anything other than 1, or if the search
reaches for a process handle instead of the `FitReader` seam.

Suggested commit: `feat(presets): add two-sided n_cpu_moe placement search`

---

# Phase 4d — Probe-backed estimates and resolve wiring

## Objective

Connect 4b and 4c to the estimator and the resolve response, including the
dense-model path and the probe/estimator agreement check.

## Read first

- Architecture section 12, the probe-backed estimates and estimator-check
  subsections
- The 4b and 4c deliverables
- `src/web/api/vram.rs` and `src/llama/vram_estimator/estimate.rs`

## Tests first

- probe results are classified `method = "fit_probe"` and never enter
  `memory_peak_bytes` or a measured match class;
- the explicit `Fit automatically` action is available on both discrete and
  unified-memory systems wherever the probe runs, and returns
  disabled-with-reason when the probe is absent;
- **dense fixture**: produces a probe-backed estimate with no MoE placement
  row and no search invoked;
- **probe-unaccepted flag**: a configuration carrying mmproj or a draft model
  yields a floor-plus-additions estimate that names the additions, not a bare
  probe total;
- **single-point agreement**: a single-point probe at an arbitrary `n` agrees
  with the search's own reading at that `n`, and a point already visited by a
  search is served from cache without re-invoking the binary;
- **estimator regression**: for every Phase 0 probe fixture, the existing
  estimator's per-component predictions fall within tolerance of the captured
  `model`/`context`/`compute` figures. Tolerances come from the initial
  corpus; a failure is fixed in the estimator, never by widening the
  tolerance;
- rapid successive drawer changes never display a stale result.

## Implementation

- Extend the `ResolvedLaunch` response with estimate, fit status, exact change
  list, and evidence-match summary.
- Add request coalescing/cancellation semantics so rapid drawer changes do not
  display stale results.
- Wire the probe as an estimator input per ADR §12: the probe supersedes the
  formula total when the configuration uses only probe-accepted flags, and
  supplies a floor plus named estimated additions when it does not. Probe
  absence leaves today's estimator behaviour untouched.
- The probe runs for dense models too — one invocation, no search. Do not gate
  the probe on `moe_layer_count` being present; gate only the placement search
  on it.
- Add the probe/estimator agreement check: for a given resolved configuration,
  compute the signed per-component divergence (`model`, `context`, `compute`)
  between the formula and the probe, and expose it on the resolve response.
  Beyond tolerance the drawer shows the probe-backed number and notes the
  disagreement; within tolerance it is silent. Compare per component, not on
  the total: two components can diverge in opposite directions and cancel.
- When the device already fits at `n = 0`, the proposal is `n_cpu_moe = 0` and
  the emitted argv carries no `--n-cpu-moe` flag. Do not emit
  `--n-cpu-moe 0`.
- The probe never gates or selects launch; its absence disables
  `Fit automatically` with a reason.

## Focused verification

- `rtk cargo test presets::fit_probe::`
- `rtk cargo test presets::intent::`
- `rtk cargo test llama::vram_estimator::`
- `rtk cargo test web::api::vram::`
- `rtk cargo clippy -- -D warnings`
- `rtk cargo fmt -- --check`
- `rtk git diff --check`

## Hard gate

Every proposal can be replayed as an exact selection with the identical
fingerprint and estimate inputs. Start never reruns intent automatically. The
estimator regression test runs over the whole Phase 0 probe corpus.

## Stop conditions

Stop on client-side formulas, on any path that lets a probe figure reach
`memory_peak_bytes`, or on a divergence check computed against the total
instead of per component.

Suggested commit: `feat(vram): back launch estimates with fit probe`

---

# Phase 4.5 — Throwaway card and drawer prototype (UX gate)

## Objective

Validate the approved Option B card and Configure drawer against a real person
before Phases 5-8 harden a data contract that exists to serve them. Everything
built here is discarded.

This phase exists because the card is the premise of the project but currently
renders for the first time in Phase 8a, after the `PresetBundleSpec` shape, the
resolve response, the intent proposal format, and the evidence match classes
have all been frozen around it. Discovering a UX problem then is expensive;
discovering it here costs one context window.

## Read first

- Architecture sections 3.1, 3.2, 11, and 12
- `static/js/features/evidence-drawer.js` for the dialog lifecycle being copied
- one v6 bundle fixture frozen in Phase 0

## Build

- Render the collapsed card and the Configure drawer from a single
  hand-authored v6 bundle fixture.
- Stub the resolve round-trip with fixture responses. No backend call, no
  persistence, no revision handling, no spawn.
- Include the states that carry the UX risk: a disabled option with its
  reason, a Low VRAM proposal with its diff, the derived `Custom` indicator,
  the MoE CPU-offload slowdown warning, and a dirty-close confirm. Also cover
  workload-aware Low VRAM quality-floor preservation, context-preserving
  no-change results, and model-card/variant discovery guidance.

Keep it in a scratch directory or a branch that is never merged. Do not add
routes, do not touch `sessionState`, and do not extend the frontend field
catalog — Phase 5 owns that.

How it runs: a scratch HTML page plus a scratch module under
a gitignored scratch directory (or an unmerged branch), served by the normal
`cargo run` build. The drawer module is loaded directly with the fixture
bundle and stubbed resolve responses injected by a tiny local mock — no
`sessionState`, no routes, no `bootstrap.js` registration. The receipt records
how it was served so a second person can reproduce it.

## Hard gate

This gate is human-attested and cannot be self-certified. No implementer,
human or model, may record its outcome from its own use of the prototype. If no
second person is available, the phase is blocked, not passed; record it as
blocked in the receipt and stop.

A person other than the implementer opens the drawer, changes quantization and
context, applies Low VRAM, reads the diff, and reaches a launch decision
without needing the plan explained to them. The receipt names who performed the
pass.

Record the outcome in the phase receipt as one of:

- contract confirmed, proceed to Phase 5 unchanged;
- contract confirmed with amendments, listing the exact architecture sections
  to revise before Phase 5 begins;
- contract rejected, with the reason.

An amendment or rejection revises the architecture document first. Phases 5-8
do not begin against a contract this gate found wanting.

## Stop conditions

Stop if the prototype starts acquiring persistence, real endpoints, or reusable
structure. It is evidence, not a foundation. If it feels worth keeping, that is
a signal the phase has drifted.

Suggested commit: none. This phase produces a receipt, not a merge.

---

# Phase 5 — Freeze one frontend field catalog and complete Preset Editor parity

## Objective

Eliminate field-registration drift and expose the target llama.cpp parameters
in the Preset Editor before changing the card.

## Read first

- Phase 0 field-parity fixture
- `static/index.html:2617-3323`
- `static/js/features/presets.js` load/save/default/change-summary paths
- `static/js/features/spawn-wizard-groups.js`
- `static/js/features/spawn-wizard-llama-ia.js`
- `tests/ui/core/phase7-presets.spec.js`
- `tests/ui/core/fixtures/spawn-wizard-control-contract.json`

## Tests first

Add `tests/ui/core/llama-config-parity.spec.js` and a machine-readable field
catalog fixture. Assert:

- unique schema key and unique editor/wizard DOM IDs;
- load/save/template/reset/payload applicability for every field;
- typed number/boolean/enum round-trip fidelity;
- Rapid fields never leak into llama payload and vice versa;
- absent values remain absent rather than becoming UI defaults;
- deprecated `cache_type_k/v` are never used;
- capability-gated fields display unavailable reason, cannot be newly selected
  or launched, and preserve an already stored unsupported value through
  unrelated edits;
- change summaries show native values without floating-point noise;
- `workload_policy` round-trips through load/save for bundled presets and is
  absent for flat presets;
- the `ctk`/`ctv` editor controls are dropdowns, not free text: common values
  `f16`, `q8_0`, `q4_0` first, a separator, then the remaining binary-advertised
  values capability-sourced per binary (never hard-coded); a value the selected
  binary does not advertise is present but disabled with the backend reason;
- unknown-string `ctk`/`ctv` values outside the named policies round-trip
  verbatim through read/unrelated-edit/save and remain non-launchable until the
  capability-validated KV policy accepts them;
- `buildPresetPayload()` either removes its redundant sampling overwrite or is
  byte-equivalent to `buildSpawnPayload()` for every applicable field.

## Implementation

- Fit is off for every go-forward bundle. Do **not** implement that here: the
  architecture places it in "the single server-owned v6 bundle constructor",
  which sets `fit_enabled: Some(false)` so every wizard, editor conversion, and
  bundle-copy surface inherits the same `--fit off` default from one place. The
  frontend does not seed `fit_enabled` for a bundle at all; it renders what the
  server returns. Verify the value arrives as `false` on a newly created bundle
  and that an existing preset's stored value survives an unrelated edit.

  **Do not edit `static/js/features/models.js:1217` or
  `static/js/features/spawn-wizard.js:659` for this.** Line 1217 is inside
  `doQuickLoad()` (which begins at `models.js:1169`) and line 659 is the global
  `wizardState.hardware.fitEnabled` default. The architecture fences off both by
  name — "Do not change the global `wizardState.hardware.fitEnabled` default or
  `models.js::doQuickLoad`; neither is a bundle-only constructor" — because
  existing flat presets and Quick Load keep their current `null` behavior. A
  change at either site is a stop condition, not a shortcut.

Keep `CONTROLS` / derived `PRESENTATION_CONTROLS` in
`static/js/features/spawn-wizard-groups.js` as the single runtime frontend
catalog. Extend its descriptors with preset-editor ID/key/applicability and
parse/format metadata where needed. Generate/validate the machine-readable
parity fixture from that source. Do not create a second hand-maintained runtime
catalog. It contains no compatibility or VRAM formulas.

Modify:

- `static/index.html`
  - add editor controls for mmproj offload and llama reasoning
    effort/format/preserve;
  - preserve existing repetition, KV-unified, batching, cache, image, and
    observability fields;
  - correct `n_cpu_moe` copy to mean expert layers kept on CPU.
- `static/js/features/presets.js`
  - load/save the new typed fields;
  - load/save `workload_policy` for bundled presets; it is
    absent for flat presets;
  - implement the `ctk`/`ctv` editor controls as capability-sourced dropdowns
    (common-first ordering, separator, disabled-with-reason for values the
    selected binary does not advertise) — never free text;
  - use the catalog for parity validation and safe formatting;
  - wire the existing Rapid reasoning-effort control only to
    `rapid_mlx.reasoning_effort` request-default semantics; never map it to
    llama argv.
- `static/css/` only where existing editor patterns are insufficient.
- Do not change `tests/ui/core/js-module-baseline.json` in this phase unless an
  independently justified new imported module is actually added.

## Focused verification

- `rtk npm run validate-preset-bundle-contract`
- `rtk npm run validate-wizard-groups`
- `rtk npm run validate-js`
- `rtk npm run lint`
- From `tests/ui`: `rtk env CI=1 LLAMA_MONITOR_USE_RELEASE=1 LLAMA_MONITOR_TEST_PORT=17778 npx playwright test core/llama-config-parity.spec.js core/phase7-presets.spec.js --workers=1`
- `rtk git diff --check`
- `rtk cargo build --release`
- Extend and run `rtk node tests/ui/capture/index.mjs --scenario preset-editor`
  with exact outputs for Generation dark/light and the main editor at narrow
  width; the existing nested chat-template narrow capture is not sufficient.
  Add these expected outputs:
  - `preset-editor--neutral--generation-dark.png`
  - `preset-editor--neutral--generation-light.png`
  - `preset-editor--neutral--generation-narrow.png`

Inspect dark, light, and narrow editor captures manually.

## Hard gate

Every catalog row has editor and wizard status, even if the wizard UI lands in
the next phase. Editor save/reopen produces byte-equivalent semantic values.

## Stop conditions

Stop if a new field still needs a one-off serializer not represented in the
catalog, if a capability-disabled control can be newly selected/launched, if an
already stored unsupported value is deleted, or if editor load/save changes any
unrelated Rapid value.

Suggested commit: `feat(presets): complete typed llama editor controls`

---

# Phase 6 — Complete Spawn Wizard Guided/Pro parity

## Objective

Wire all target non-card parameters through the wizard's one canonical control
system and prove Guided/Pro payload equivalence.

## Read first

- Architecture section 9
- `docs/reference/spawn-wizard.md` frontend module map and current
  `critical`/`view` disclosure semantics
- `spawn-wizard.js` reset/template/read-state paths
- `spawn-wizard-groups.js`
- `spawn-wizard-llama-ia.js`
- `spawn-wizard-review-step.js`
- `spawn-wizard-spawn.js`
- Phase 5 field catalog and parity tests

## Tests first

Extend:

- `tests/ui/core/spawn-wizard.spec.js`
- `tests/ui/core/fixtures/spawn-wizard-control-contract.json`
- `scripts/validate-wizard-groups.mjs`

Cover:

- every row of the control table below, one test per row, asserting the row's
  `view`, `critical`, and resolved Pro category;
- mmproj offload tri-state;
- llama reasoning effort/format/preserve capability and template gating;
- corrected `n_cpu_moe` meaning and bounds;
- one DOM node per setting while switching Guided/Pro repeatedly;
- Guided drawer reachability for `view: 'both'` controls and decision-card
  ownership for `view: 'card'` controls;
- Pro category, search, modified-only, reset, and changed-count behavior;
- identical payload from Guided and Pro for identical values;
- template preset -> wizard -> save -> editor round trip;
- keyboard and axe coverage.

## Implementation

- `static/js/features/spawn-wizard.js:659` currently seeds `fitEnabled: null`.
  Apply the same creation-only default as Phase 5: `false` for a newly created
  bundle, untouched for an existing preset. The rule and its rationale are in
  the architecture doc, section 6, under the heading about `--fit` re-resolving
  intent. If only one of the two sites is changed, a new bundle's fit state
  depends on which surface created it — Phase 10a's cross-surface fixtures will
  surface that as a fingerprint mismatch rather than as a fit bug, so fix it
  here.

### The control table — binding

This is the whole of the per-control work. Each row is one unit: do the six
steps below for row 1, run the contract validator, then move to row 2. Do not
attempt all rows at once, and do not infer additional controls from other
sections of this document — this table is the complete list.

Verified against `static/index.html` and `spawn-wizard-groups.js` at the time
of writing: every control marked *exists* already has a DOM node and is missing
only its registry row. Confirm that is still true before starting; if a node is
gone, that is a Coordinator question, not a licence to create a second node.

| # | Control id | DOM node | `view` | `critical` | Intended Pro category | `llamaCategory()` edit needed? |
|---|---|---|---|---|---|---|
| 1 | `spawn-repeat-last-n` | exists | `both` | `false` | Generation & reasoning | no — derives correctly |
| 2 | `spawn-no-cont-batching` | exists | `both` | `false` | Performance | no — derives correctly |
| 3 | `spawn-swa-full` | exists | `both` | `false` | Memory & context | **yes** |
| 4 | `spawn-load-mode` | exists | `both` | `false` | Memory & context | **yes** |
| 5 | `spawn-verbosity` | exists | `both` | `false` | Network & observability | **yes** |
| 6 | `spawn-ctx-checkpoints` | exists | `both` | `false` | Memory & context | **yes** |
| 7 | `spawn-checkpoint-min-step` | exists | `both` | `false` | Memory & context | **yes** |
| 8 | `spawn-cache-reuse` | exists | `both` | `false` | Memory & context | **yes** |
| 9 | `spawn-cache-idle-slots` | **create** | `both` | `false` | Memory & context | **yes** |
| 10 | `spawn-mmproj-offload` | **create** | `both` | `false` | Model & compatibility | **yes** |
| 11 | `spawn-reasoning-effort` | **create** | `both` | `false` | Generation & reasoning | no — derives correctly |
| 12 | `spawn-reasoning-format` | **create** | `both` | `false` | Generation & reasoning | no — derives correctly |
| 13 | `spawn-reasoning-preserve` | **create** | `both` | `false` | Generation & reasoning | no — derives correctly |

Rows 3-8 are the controls this plan previously called "previously omitted
registry controls"; rows 9-13 are new.

Row 13 supersedes the existing `spawn-preserve-thinking` control. They are two
different mechanisms, not two nodes for one setting, which is why the old one
cannot simply be rebound to the new flag:

- `spawn-preserve-thinking` binds `preserve_thinking`, a **chat-template
  kwarg**. `src/inference/llama_cpp.rs` collects it into a JSON map with
  `enable_thinking` and `tool_call_format` and emits one
  `--chat-template-kwargs '{...}'` argument. The template consumes it; the
  server does not interpret it.
- `spawn-reasoning-preserve` binds `llama_reasoning_preserve`, a **native
  server flag** alongside `--reasoning`, `--reasoning-budget`, and
  `--reasoning-budget-message`, which `llama_cpp.rs` emits as separate `cmd.arg`
  calls immediately after the kwargs block.

Do not merge them, do not make one write the other's value, and do not gate the
native capability onto the kwarg control. The architecture contract states this
directly (`llama_reasoning_preserve` — "distinct from `preserve_thinking`
inside chat-template kwargs") and Phase 2 already forbids equating them.

**`preserve_thinking` is legacy.** `--reasoning-preserve` supersedes it.
Consequences for this phase, and no further:

- `spawn-reasoning-preserve` is the forward control. Row 13 is the one that
  gets built out, categorized, and covered by this phase's gate tests.
- `spawn-preserve-thinking` stays readable and editable so existing presets
  that set it keep round-tripping unchanged. Do not delete the control, the
  `preserve_thinking` preset field, or its emission in
  `src/inference/llama_cpp.rs`.
- Newly created bundles must not set `preserve_thinking`. Leave it `None`.
- Do not migrate existing `preserve_thinking` values onto
  `llama_reasoning_preserve`. The two are not value-compatible, and a silent
  migration would change launch behavior for presets that currently work.
- Removing `preserve_thinking` outright is a schema decision for the
  Coordinator, not a local one. It is not in scope for any phase in this plan.

Row 13 renders **disabled with a reason**, not hidden (invariant 17). Its two
gates are independent: exact-binary flag support from the capability snapshot,
and template compatibility. Until the bounded template-inspection rule and
fixture named in Phase 2 exist, template compatibility is `unknown` and the
control stays disabled with that stated as the reason. Do not infer template
support from the presence of `spawn-preserve-thinking`, from the model family
string, or from the flag being advertised in `--help`.

**Per row, in order:**

1. Add the DOM node to `static/index.html` if the table says *create*; otherwise
   confirm the existing node and change nothing about it.
2. Add the registry row to `CONTROLS` in `spawn-wizard-groups.js`, in the form
   `{ id, loaders: ['llama_cpp'], critical, view }`.
3. If the last column says **yes**, add the id to the matching branch of
   `llamaCategory()` so the derived category matches the table.
4. Confirm physical group ownership in `spawn-wizard-llama-ia.js` matches.
5. Add the payload mapping in `spawn-wizard-spawn.js` and the summary binding in
   `spawn-wizard-review-step.js`.
6. Run `rtk npm run validate-wizard-groups` before starting the next row.

**Trap — `proCategory` is derived, not authored.** `descriptorForControl()`
spreads `...control` and then overwrites `proCategory` with
`llamaCategory(control.id)`. A hand-written `proCategory:` in a registry row is
silently discarded and the control lands in whichever branch its id happens to
match. `llamaCategory()` ends in a catch-all `return 'Advanced'`, so a control
with no matching branch is not an error — it is a plausible-looking wrong
answer. Six of the eight rows that exist today hit that catch-all (rows 3-8);
counting the five new rows, eight of the thirteen need the edit. That is why
step 3 exists and why the contract fixture must pin the resolved category for
every row. The same applies to `guidedPlacement`, which is derived from `view`:
set `view`, never `guidedPlacement`.

- `static/index.html`: add only the canonical controls named in the table. Every
  control is one node used by both Guided and Pro. Do not add a second
  Guided-specific node for a setting that already has one, under any name.
- `spawn-wizard-llama-ia.js`: physical group ownership matches registry.
- `spawn-wizard.js`: reset, template restore, state read, capability locks.
- `spawn-wizard-review-step.js`: bind/summary/preset payload parity.
- `spawn-wizard-spawn.js`: canonical spawn serialization.
- `spawn-wizard-guided.js`: Guided may give a control a friendlier label than
  Pro shows. It may not own the value. Every write goes through the one
  canonical control node from the table; if a change here would let Guided hold
  its own copy of a value, it is out of scope for this phase.
- `docs/reference/spawn-wizard.md` and `docs/reference/setup-wizard.md`.

Raw llama.cpp flag names remain visible in Pro help text for every row.

## Focused verification

- `rtk npm run validate-preset-bundle-contract`
- `rtk npm run validate-wizard-groups`
- `rtk npm run validate-js`
- `rtk npm run lint`
- From `tests/ui`: `rtk env CI=1 LLAMA_MONITOR_USE_RELEASE=1 LLAMA_MONITOR_TEST_PORT=17778 npx playwright test core/spawn-wizard.spec.js core/llama-config-parity.spec.js --workers=1`
- `rtk cargo build --release`
- sequential captures:
  - `rtk node tests/ui/capture/index.mjs --scenario spawn-wizard-guided-drawer`
  - `rtk node tests/ui/capture/index.mjs --scenario spawn-wizard-pro-baseline`
  - `rtk node tests/ui/capture/index.mjs --scenario spawn-wizard-launch-full-config`
- Extend Guided capture steps to scroll to and capture cache/slot controls,
  mmproj offload, and reasoning controls at legible scale. Extend Pro Model and
  Generation category fixtures rather than relying on one full-page still.
  Register exact additional outputs:
  - `llamacpp-local--spawn-wizard-guided-cache-slots.png`
  - `llamacpp-local--spawn-wizard-guided-mmproj-offload.png`
  - `llamacpp-local--spawn-wizard-guided-reasoning.png`
  - `llamacpp-local--spawn-wizard-pro-model-compatibility.png`
  - `llamacpp-local--spawn-wizard-pro-generation-reasoning.png`
- `rtk git diff --check`

## Hard gate

The same changed values survive Guided -> Pro -> Guided, preset save, editor
reopen, and backend command preview without loss or default substitution.

## Stop conditions

Stop if a control is cloned, if switching views changes values, if unsupported
flags serialize, if reasoning-preserve ignores template capability, or if the
command preview differs from backend argv.

Suggested commit: `feat(wizard): expose canonical llama runtime controls`

---

# Phase 7 — Add explicit bundle management to the Preset Editor

## Objective

Let users explicitly create/manage one exact-tune bundle and its artifacts and
launch choices. Do not change the welcome card yet.

## Read first

- Architecture sections 4 and 5
- Phase 2 v6 fixtures
- Phase 3 selection APIs
- `static/js/features/presets.js`
- `static/index.html` Preset Editor structure
- `src/models/mod.rs` GGUF filename parsing
- `src/llama/gguf_meta.rs`
- model/file browser modules used by the editor and wizard

## Tests first

Add tests for:

- convert one legacy preset into one-artifact bundle with the ADR §5 Conversion
  defaults applied exactly: `workload_policy` is
  `CustomUnknown` (conversion never guesses workload), `allow_validated_custom`
  is `true` (conversion must not narrow the freedom a flat preset had),
  `curated_selections` is exactly one entry (the converted default selection),
  and `performance_options` is exactly one option labelled with the stored
  nonzero batch/ubatch pair when the flat preset had one;
- `cpu_moe_options` on conversion is `[0]` plus the stored value when nonzero
  and within bounds; otherwise empty;
- legacy zero/null batch conversion requires a concrete nonzero option or an
  exact capability-backed RuntimeDefault representation; never copy zero into
  an explicit bundle performance choice;
- explicitly add Q4 and Q5 artifacts of the same tune;
- refuse automatic grouping by family/name only;
- show metadata mismatch before user confirmation;
- duplicate artifact ID/path/HF coordinate rejection;
- remove non-selected artifact;
- prevent removing the selected/only artifact without selecting replacement;
- context/performance/MoE option validation;
- dense model hides/rejects MoE options;
- copy/rename/delete/collection semantics remain deterministic;
- the reset and delete flows re-fetch the catalog (a fresh `catalog_etag`)
  immediately before prompting for destructive confirmation, and send the fresh
  etag/revision; a stale value at prompt time is rejected;
- after this phase, `PUT` on a *flat* preset also requires `expected_revision`;
  before this commit flat presets keep last-write-wins;
- `revision` survives a restart (persisted with the schema, part of
  `catalog_etag`; legacy v4 receives `revision = 1` at its first v5 write);
- cancel leaves the preset unchanged;
- save failure leaves both UI and backend original intact;
- stale full-editor PUT versus newer drawer PATCH returns 409 and preserves the
  newer server state;
- legacy preset remains launchable and reversible before conversion save.

## Implementation

- Add a Model variants section to the Preset Editor.
- Reuse the existing file/HF browser and GGUF metadata endpoint.
- Suggestions show provenance and require explicit confirmation.
- Store exact batch/ubatch choices; never store “runtime default” when the
  intent is a reproducible option.
- Use existing preset CRUD for the full bundle and revisioned selection PATCH
  for drawer-only selection changes. From this phase, the Preset Editor, reset,
  and delete UI always send `expected_revision` for every preset, flat or
  bundled: before this commit flat presets keep today's
  last-write-wins behavior, and `PUT` on a flat preset does not require it. The
  server owns the next revision.
- Update collections and delete behavior in `src/web/api/models.rs` or the
  actual current collection owner so one bundle remains one collection item.
- Update `docs/reference/api.md` and `docs/reference/setup-wizard.md`.

## Focused verification

- `rtk cargo test presets::bundle::`
- `rtk npm run validate-js`
- `rtk npm run lint`
- From `tests/ui`: `rtk env CI=1 LLAMA_MONITOR_USE_RELEASE=1 LLAMA_MONITOR_TEST_PORT=17778 npx playwright test core/preset-flow.spec.js core/llama-config-parity.spec.js --workers=1`
- `rtk cargo build --release`
- Add and run a seeded `preset-bundle-editor` capture scenario; the existing
  New Preset scenario is insufficient. Require exact dark/light/narrow outputs
  for artifact membership, selected artifact, context/performance choices,
  MoE-visible, dense-hidden, and removal/replacement warning states.
  Register these exact expected outputs:
  - `llamacpp-local--preset-bundle-editor-artifacts-dark.png`
  - `llamacpp-local--preset-bundle-editor-artifacts-light.png`
  - `llamacpp-local--preset-bundle-editor-artifacts-narrow.png`
  - `llamacpp-local--preset-bundle-editor-moe-options.png`
  - `llamacpp-local--preset-bundle-editor-dense-no-moe.png`
  - `llamacpp-local--preset-bundle-editor-remove-warning.png`
  Implement it at
  `tests/ui/capture/scenarios/presets/preset-bundle-editor.mjs` and register it
  in `tests/ui/capture/index.mjs` with `category: 'presets'` and a fail-closed
  `expectedOutputs` list.
- `rtk git diff --check`

## Hard gate

Two weight artifacts exist inside one persisted preset bundle, reopening the
editor preserves exact membership/options, and no heuristic membership becomes
authoritative without confirmation.

## Stop conditions

Stop on cross-tune auto-grouping, hidden artifact replacement, destructive
conversion without successful persistence, or ambiguous delete semantics.

Suggested commit: `feat(presets): manage explicit model artifact bundles`

---

# Phase 8a — Implement the approved compact card

## Objective

Render one bundle as one Option B compact card, with the render flag that can
switch the whole card back off. The Configure drawer is not built here; the
Configure control opens nothing until Phase 8b.

## Read first

- Architecture section 3.1 card ASCII and its binding behavior
- Architecture section 12 live-availability rule and invariants 15-19
- `setup-view.js:842-1401`
- `setup-view.css:2511-3675`
- `attach-detach.js` preset start path
- Phase 3 resolve/start APIs and Phase 4a intent output
- Phase 4.5 receipt, including any amendments it recorded

## Tests first

Extend `tests/ui/core/preset-flow.spec.js` for:

- Q4/Q5 bundle renders one card;
- exact tune title and saved summary chips;
- card readiness: a bundle card whose selected artifact is missing local
  (`artifact_not_local` from resolve) renders the existing no-model
  `--configure` degraded state before any spawn attempt, with the download
  affordance from the existing download workflow (named test);
- the card's `Available` line and the fit verdict read
  `GET /api/memory-availability` live — the same endpoint the current UI
  polls — never a value frozen at save time; the verdict carries the read
  timestamp; discrete-VRAM and unified-memory systems both use the same live
  read;
- Start uses saved exact selection/revision;
- clicking `.launch-card[data-preset-id="..."] .launch-card-btn-start` sends
  the exact preset ID plus saved bundle revision/selection semantics; do not
  substitute a direct `doStart()` unit call for this click-path test;
- legacy one-artifact card behavior;
- the render flag forces legacy one-artifact rendering even when a v6 bundle is
  present, and does so through the legacy adapter rather than a second path;
- running badge and last-launched sorting remain correct;
- delete targets the bundle, not an artifact;
- card `Edit full preset` opens the correct bundle in `#preset-modal` and
  restores focus predictably on return;
- card dark/light/narrow/reduced-motion and axe checks.

## Implementation

Add a render flag that forces legacy one-artifact card rendering even when v6
bundles exist on disk (architecture invariant 16). Schema version and card
presentation are separate risks: migration is forward-only with no downgrade,
so a defective card must be switchable off without touching preset files. The
flag defaults to the new card, is read at render time rather than cached at
load, and its legacy path is the same one-artifact adapter legacy presets use —
not a second code path.

Modify:

- `static/js/features/setup-view.js`
  - render one bundle card;
  - display saved selection summary;
  - keep `_renderCardVram()` segmented bar and fit legend;
  - expose the Configure control with a single documented seam the Phase 8b
    drawer attaches to; do not inline drawer logic here;
  - start is an atomic resolve-and-launch: it sends the
    preset ID and `expected_revision` only — no one-shot selection and no
    config hash, because a collapsed card shows no preview for the user to
    have consented to. A revision mismatch is 409 (someone else edited the
    preset). A binary that no longer supports the saved selection is 422
    `capability_changed` with a fresh safe preview the card re-renders; the
    user may then open Configure to re-consent. The `cfg-v1:` consent hash is
    a drawer-only mechanism (Phase 8b `Start without saving` / `Save & Start`).
- `static/js/features/attach-detach.js`
  - accept optional selection/revision for Start once;
  - preserve token and readiness behavior.
- `static/css/setup-view.css`
  - compact card only; do not expand every card into inline selectors.
- `tests/ui/core/js-module-baseline.json`.

Build DOM with `textContent` for untrusted values.

## Focused verification

- `rtk npm run validate-preset-bundle-contract`
- `rtk npm run validate-js`
- `rtk npm run lint`
- From `tests/ui`: `rtk env CI=1 LLAMA_MONITOR_USE_RELEASE=1 LLAMA_MONITOR_TEST_PORT=17778 npx playwright test core/preset-flow.spec.js --workers=1`
- `rtk git diff --check`

## Hard gate

One card launches each exact tested selection; many bundle cards remain
readable at the existing grid width; no inline selector sprawl is introduced;
the render flag returns the grid to legacy rendering with no preset file
touched.

## Stop conditions

Stop on client-side compatibility logic, auto-save on chip click, loss of
focus, or any launch request lacking revision/fingerprint protection.

Suggested commit: `feat(ui): consolidate model variants in preset cards`

---

# Phase 8b — Implement the approved Configure drawer

## Objective

Deliver the Configure drawer exactly as specified in the architecture ASCII and
binding behavior, attached to the Phase 8a card seam.

## Read first

- Architecture section 3.2 drawer ASCII and binding behavior, including the
  disabled-not-hidden rule, dirty-close guard, derived `Custom`, footer
  ranking, and diff-from-saved rules
- Architecture section 12 live-availability rule and invariants 15-19
- `evidence-drawer.js:20-145`
- `evidence-drawer.css:1-187`
- Phase 8a card seam and receipt
- Phase 4.5 receipt, including any amendments it recorded

## Tests first

Extend `tests/ui/core/preset-flow.spec.js` for:

- Configure opens correct bundle and restores focus on close;
- Escape, backdrop, focus trap, keyboard-only selection;
- draft changes do not mutate the collapsed card;
- stale preview responses cannot overwrite newer choices;
- disabled choice displays backend reason, stays visible, and associates that
  reason via `aria-describedby`; no unavailable option is hidden from the DOM;
- dismissing a dirty drawer via Escape, backdrop, and the close control each
  prompt rather than discarding; Reset discards without prompting;
- `Custom` renders as a derived indicator and exposes no activatable control in
  the intent row or the Performance row;
- the diff block lists divergence from the saved selection with no intent
  applied, and names the intent when one produced the change;
- a proposal placing MoE experts on CPU renders the qualitative slowdown
  warning;
- the Workload row sits above `Predicted result`, exposes the
  four known `workload_policy` values as a dropdown, and a change to it is
  part of the draft and appears in the diff block;
- aggressive KV (`q4_0/q4_0`) eligibility reads the policy at resolve time:
  when the policy does not qualify it, the drawer renders the option disabled
  with the policy reason;
- `Fit automatically` issues the probe search and renders the
  result as a draft change with the diff block; it never auto-applies or
  persists;
- MoE placement changes the estimate visibly: the drawer reports probe-backed
  device VRAM total, expert-weight system-RAM impact, and headroom against the
  available device budget;
- a user-adjustable headroom/VRAM buffer passes through as `fit_target_mib` and
  reruns the same two-sided probe search, while manual `n_cpu_moe` changes use
  a single-point probe;
- unavailable or failed probe results show a backend reason and no stale or
  inferred number;
- when the probe is absent or unparseable, `Fit automatically` shows the
  backend `probe_unavailable` reason and never guesses a value;
- Start once uses draft and does not save;
- Save updates card only from server response;
- Save & Start waits for successful persistence;
- 409 conflict offers Reload, not overwrite;
- estimate sections and fit states match backend response;
- the prohibited main-KV policy ID/pair is absent from the main-KV option
  payload and cannot be submitted through typed main KV or `extra_args`;
- drawer `Edit full preset` opens the correct bundle in `#preset-modal` and
  restores focus predictably on return;
- drawer dark/light/narrow/reduced-motion and axe checks.

## Implementation

Add:

- `static/js/features/preset-bundle-drawer.js`
- `static/css/preset-bundle-drawer.css`

The drawer module owns exactly this state contract from the architecture:

```text
bundleId
serverRevision
savedSelection
draftSelection
normalizedPreview
dirty
previewRequestGeneration
previewAbortController
opener
```

`sessionState.presets` remains saved truth; drafts never mutate it. Only the
latest monotonically numbered resolve request updates preview state. Closing
discards draft. Save merges only the server-returned preset/revision through
the canonical `loadPresets` refresh path. Start once uses the normalized
resolve result; Save & Start uses the exact PATCH-returned revision.

Modify:

- `static/js/features/setup-view.js`
  - route configuration to the new drawer through the Phase 8a seam;
  - statically import/own `preset-bundle-drawer.js`; do not initialize a second
    copy through `bootstrap.js`.
- `static/index.html`
  - add the explicit `<link rel="stylesheet">` for
    `preset-bundle-drawer.css`; build.rs route generation alone does not load
    CSS.
- `tests/ui/core/js-module-baseline.json`.

The new drawer copies accessible mechanics, not evidence-drawer data code.
Build DOM with `textContent` for untrusted values.

## Focused verification

- `rtk npm run validate-preset-bundle-contract`
- `rtk npm run validate-js`
- `rtk npm run lint`
- From `tests/ui`: `rtk env CI=1 LLAMA_MONITOR_USE_RELEASE=1 LLAMA_MONITOR_TEST_PORT=17778 npx playwright test core/preset-flow.spec.js --workers=1`
- `rtk git diff --check`

## Hard gate

Every binding behavior rule in architecture section 3.2 has a passing test that
fails when the rule is removed. No unavailable option is absent from the DOM,
and no dirty drawer can be dismissed without a prompt.

## Stop conditions

Stop on client-side compatibility logic, hidden draft state, auto-save on chip
click, stale response races, loss of focus, or missing theme/motion states.

Suggested commit: `feat(ui): add preset bundle configure drawer`

---

# Phase 8c — Capture and register bundle screenshots

## Objective

Prove the delivered card and drawer visually match the architecture hierarchy,
under a fail-closed capture scenario.

## Read first

- Architecture sections 3.1, 3.2, and 12
- Phase 8a and 8b receipts
- `tests/ui/capture/index.mjs` scenario registry and `expectedOutputs` contract

## Screenshot gate

Add:

- `tests/ui/capture/scenarios/presets/preset-bundle.mjs`

Modify:

- `tests/ui/capture/index.mjs` — register `preset-bundle` in the `presets`
  group.

Run `rtk cargo build --release`, then sequentially capture:

- bundled grid dark;
- bundled grid light;
- drawer default dark;
- drawer light;
- drawer narrow bottom sheet;
- drawer reduced motion;
- Low VRAM change explanation;
- invalid/no-fit states;
- exact and related evidence states using fixtures.

Register `preset-bundle` with `category: 'presets'`, a deterministic seeded
Q4/Q5 bundle, and exact `expectedOutputs` filenames for every state above.
Fixtures mock resolve, selection PATCH, estimate/evidence, and spawn routes,
including stale-response and revision-conflict cases. The scenario fails when
any output is missing or only a legacy flat card rendered.

Exact expected outputs:

- `llamacpp-local--preset-bundle-grid-dark.png`
- `llamacpp-local--preset-bundle-grid-light.png`
- `llamacpp-local--preset-bundle-drawer-default-dark.png`
- `llamacpp-local--preset-bundle-drawer-light.png`
- `llamacpp-local--preset-bundle-drawer-narrow.png`
- `llamacpp-local--preset-bundle-drawer-reduced-motion.png`
- `llamacpp-local--preset-bundle-drawer-low-vram-changes.png`
- `llamacpp-local--preset-bundle-drawer-no-fit.png`
- `llamacpp-local--preset-bundle-drawer-evidence-exact.png`
- `llamacpp-local--preset-bundle-drawer-evidence-related.png`
- `llamacpp-local--preset-bundle-drawer-revision-conflict.png`

All files go under
`docs/screenshots/artifacts/presets/`; nothing is promoted to
`docs/screenshots/` unless referenced by public documentation.

## Focused verification

- `rtk cargo build --release`
- `rtk node tests/ui/capture/index.mjs --scenario preset-bundle`
- `rtk bash scripts/check-unused-screenshots.sh`
- `rtk npm run lint`
- `rtk git diff --check`

## Hard gate

Fresh screenshots visually match the architecture hierarchy. Removing any one
registered output causes the scenario to fail rather than silently skip.

## Stop conditions

Stop if a capture requires changing card or drawer behavior to succeed; that is
a Phase 8a or 8b defect, not a capture defect.

Suggested commit: `test(ui): capture preset bundle card and drawer states`

---

# Phase 9 — Add exact runtime memory observations

## Objective

Populate evidence-backed configuration memory without confusing system usage,
process memory, estimator calibration, or filename labels.

This phase may ship after the base bundled-card release. Until it passes, cards
display estimates and `No exact measurement` only.

## Read first

- Architecture section 12
- `src/calibration/mod.rs`
- `src/calibration/executor.rs`
- `src/calibration/server_qualification.rs`
- existing exact/compatible/related match logic
- `docs/reference/cache-benchmark-results.md`
- platform metrics/agent code that owns GPU/process observations

## Tests first

Create platform-labelled receipt fixtures for:

- Windows/WDDM total-device delta;
- CUDA/ROCm process accounting where available;
- Metal/unified physical/system observation;
- estimator-only;
- exact, compatible, related, and stale fingerprints;
- noisy background usage and negative/implausible deltas;
- incomplete sampling and server-start failure;
- changed binary, artifact, context, KV, batch, MoE, or concurrency.
- one mutation fixture for every memory-relevant normalized argv field,
  including companions/mmproj offload, image tokens, GPU layers/tensor split,
  KV unification/offload, load/fit/cache/SWA, cache RAM/idle/checkpoints, and all
  draft/speculative model/KV/placement settings;
- a fail-closed test that adding a typed argv field without fingerprint
  classification breaks the manifest validator.

Assert the UI never upgrades compatible/related evidence to exact and never
labels estimator calibration as an exact launch measurement.

## Implementation

- Extend the existing Calibration evidence index/fingerprint vocabulary with a
  launch-observation receipt kind; do not create an unrelated evidence store.
- Record method, before/peak/after, model delta where meaningful, sample count,
  interval, noise/quality flags, and exact selection fingerprint.
- Populate `memory_peak_bytes` only when the method semantics are explicit.
- Add bounded post-readiness sampling with guaranteed cleanup and no blocking
  of normal session control.
- Extend resolve/card evidence match response.
- Generate the evidence fingerprint from the canonical normalized resolved-argv
  manifest, not a hand-selected subset of preset fields.
- Add an evidence-details action from the card; reuse the existing evidence
  drawer where appropriate.

## Real-host gates

These gates require explicit Coordinator/user authorization before starting or
stopping remote models:

- Ryne Windows/CUDA: validate total-device delta with `--parallel 1 -fit off`
  and safe K/V only. The receipt must include exact artifact/runtime digest,
  normalized resolved argv, pre-existing server/port/process inventory,
  timestamped raw `nvidia-smi` before/readiness/peak/after samples, idle
  stabilization, background-process change/noise detection, repeated
  observation, guaranteed cleanup/restore, and explicit total-device labeling.
  A noisy or non-repeatable result remains related/noisy, never exact.
- Apple Silicon: validate unified-memory method separately.
- Linux CUDA/ROCm: validate per-process claims only where tooling proves them.

Do not run mixed q8/q4 in any qualification.

## Focused verification

- `rtk cargo test calibration::`
- `rtk cargo test inference::`
- `rtk cargo test presets::evidence::`
- From `tests/ui`: `rtk env CI=1 LLAMA_MONITOR_USE_RELEASE=1 LLAMA_MONITOR_TEST_PORT=17778 npx playwright test core/preset-flow.spec.js --workers=1`
  covering the exact, compatible, related, and stale evidence classes
- `rtk cargo clippy -- -D warnings`
- `rtk cargo fmt -- --check`
- `rtk git diff --check`

## Hard gate

An exact measurement can be traced from card -> receipt -> full fingerprint ->
raw method receipt. Changing any launch identity component prevents exact reuse.

No observation is recorded as exact runtime evidence unless the launch pinned
`--fit off`. `--fit` defaults to `on` and adjusts arguments left unset, so with
fit active the binary may shrink context or batch after the resolver decided,
and the receipt would describe argv the run did not use. Fit state is part of
launch identity: record it in the receipt, and treat a fit-on run as an estimate
rather than an observation.

## Stop conditions

Stop if WDDM is presented as per-process, Metal as VRAM-only, background usage
cannot be bounded, sampler cleanup is unreliable, or a historical filename is
treated as evidence.

Suggested commit: `feat(vram): record exact launch memory evidence`

---

# Phase 10a — Cross-surface end-to-end fixtures and round-trip tests

## Objective

Prove the whole feature round-trips across every surface as one authenticated,
cross-platform system. This phase builds fixtures and tests only; it ships no
documentation and runs no release qualification.

## Read first

- Architecture section 5 field-parity matrix and section 14 invariants 1-22
- Receipts for Phases 1a, 1b, 2, 3, 4a, 4b, 4c, 4d, 5, 6, 7, 8a, 8b, 8c,
  and 9
- `tests/ui/core/preset-flow.spec.js`, `phase7-presets.spec.js`,
  `spawn-wizard.spec.js`, `llama-config-parity.spec.js`, `security-auth.spec.js`
- `tests/fixtures/presets/` as frozen in Phase 0 and extended in Phase 1b

## Required end-to-end fixtures

1. Legacy v4 llama preset.
2. Invalid legacy mixed-KV preset preserved but blocked.
3. Qwen3.8 Brainwaves Q4/Q5 bundle.
4. Qwen3.6 35B MoE bundle with several `n_cpu_moe` choices.
5. Dense bundle proving no MoE choices.
6. Vision bundle with mmproj offload capability on/off/unavailable.
7. Reasoning model with effort values and binary/reasoning-mode gated preserve,
   plus supported/unsupported/unknown template-compatibility fixtures.
8. Unknown/degraded metadata bundle.
9. Rapid-MLX preset proving no llama field leakage.
10. Exact/related/no-evidence memory states.

## End-to-end tests

Prove:

- wizard Guided -> save -> card -> Configure -> editor -> Start;
- wizard Pro -> save -> editor -> wizard template reload;
- legacy -> explicit bundle conversion -> Q4/Q5 selection;
- Start, Start once, Save, and Save & Start semantics;
- direct/preset/card spawn use identical resolver/validator;
- API auth and malformed JSON 400 behavior;
- session persistence records resolved fingerprint without secrets;
- card filtering, family grouping, collections, sorting, running badge, delete;
- unsupported binary capabilities degrade controls without deleting values;
- create/update/delete/reset/selection/conversion disk failures each return
  non-success and leave in-memory state unchanged;
- Windows-style paths and argv remain `OsString` safe;
- all screenshot scenarios are current-source and sequential.

## Focused verification

- `rtk cargo test`
- `rtk npm run validate-preset-bundle-contract`
- `rtk npm run validate-js`
- `rtk npm run lint`
- From `tests/ui`: `rtk env CI=1 LLAMA_MONITOR_USE_RELEASE=1 LLAMA_MONITOR_TEST_PORT=17778 npx playwright test core/preset-flow.spec.js
  core/phase7-presets.spec.js core/spawn-wizard.spec.js
  core/llama-config-parity.spec.js core/security-auth.spec.js --workers=1`
- `rtk git diff --check`

## Hard gate

Every fixture above is exercised by at least one named test, and every listed
round trip passes without a field being dropped, defaulted, or silently
rewritten. A fixture with no test referencing it is a phase failure, not a
deferred item.

## Stop conditions

Stop if a round trip can only be made to pass by relaxing a Phase 1b validator,
by adding a client-side compatibility rule, or by editing a fixture to match
observed behavior rather than intended behavior. Any of those means an earlier
phase is wrong; revise it there, not here.

Suggested commit: `test(presets): cover bundle round trips across surfaces`

---

# Phase 10b — Documentation, security, and release qualification

## Objective

Turn a passing feature into a durable PR handoff: documentation written as if
the feature always existed, a completed security review, and a final receipt.
No product behavior changes in this phase.

## Read first

- Phase 10a receipt and every prior phase receipt
- Architecture section 14 invariants 1-22
- The repository's required pre-PR command ordering below; it is exact and not
  reorderable

## Documentation

Update in the same PR:

- `docs/reference/spawn-wizard.md`
- `docs/reference/setup-wizard.md`
- `docs/reference/api.md`
- `docs/reference/vram-estimator.md`
- `docs/reference/inference-tuning.md`
- `docs/reference/windows-support.md`
- README only if the bundled card is a documented headline feature.

Write as if the feature always existed. Promote screenshots only when directly
referenced.

## Mandatory pre-PR checks

Run in the repository-required exact order:

1. `rtk cargo clippy -- -D warnings`
2. `rtk cargo test`
3. `rtk npm run validate-js`
4. `rtk npm run lint`
5. `rtk git diff --check`
6. `rtk cargo build --release`
7. `rtk cargo fmt`
8. `rtk git status`
9. From `tests/ui`: `rtk npm run update-baseline` if required by new static JS
   imports; commit the baseline.

Then run the isolated full UI suite from `tests/ui` with at least 600 seconds:

```text
rtk env CI=1 LLAMA_MONITOR_USE_RELEASE=1 LLAMA_MONITOR_TEST_PORT=17778 npm test
```

Run sequential screenshot groups on one explicit port:

- `rtk env SCREENSHOT_PORT=17830 node tests/ui/capture/cli-group.mjs presets --no-attach`
- `rtk env SCREENSHOT_PORT=17830 node tests/ui/capture/cli-group.mjs wizard-llamacpp --no-attach`

Run:

- `rtk bash scripts/check-unused-screenshots.sh`
- `rtk rustup target add x86_64-pc-windows-gnu`
- `rtk cargo check --target x86_64-pc-windows-gnu` — unconditional:
  the Windows cross-target check is a compile-time release-qualification
  gate and runs once here, not per phase. Record in the receipt which phases
  touched `#[cfg]`-scoped or platform policy code; that per-phase platform
  coverage is the host-side unit test on the pure `resolve_*(TargetPlatform, …)`
  helpers, not an earlier cross-target build.
- `/security-review` and `/review` through the applicable review workflow.

## Final security checklist

- api-token on resolve and selection APIs;
- db-admin-token retained on spawn;
- safe JSON limits and timeouts;
- no secret included in selection fingerprints or receipts;
- no untrusted innerHTML;
- canonical file paths constrained to allowed roots;
- no predictable IDs;
- no direct SQLite operations;
- no unbounded artifact/option arrays;
- no safety-critical flag override through `extra_args`;
- unsupported stored values remain readable/persisted but cannot be newly
  selected or launched;
- auth routing tests cover every new endpoint.

## Final hard gate

The final receipt contains:

- every phase commit and independent Verifier result;
- schema and route manifests;
- full relevant test output summary;
- exact current-source screenshot manifest;
- local and real-host evidence clearly separated;
- remaining deferred items;
- conventional PR title and, when multiple user-facing changes are included, a
  proposed PR-body `BEGIN_COMMIT_OVERRIDE` block.

Do not add `ready-to-test`. Do not claim measured VRAM unless Phase 9 exact
receipts passed. Do not claim mixed-KV support.

## Stop conditions

Stop if any mandatory check fails; a failing check is fixed in the phase that
owns it, with that phase's gate re-run, not patched here. Stop if documentation
would have to describe behavior the tests do not demonstrate. Do not weaken,
reorder, or skip a pre-PR check to reach a green final receipt.

Suggested commit: `docs(presets): document bundled launch configuration workflow`

# Proposed PR title and override inventory

Use a `feat:` PR title because the card bundle is user-facing, for example:

```text
feat(presets): consolidate model variants into configurable cards
```

If all planned user-facing work ships in one PR, propose this PR-body block for
human review; never put it in a commit message:

```text
BEGIN_COMMIT_OVERRIDE
feat(presets): consolidate model quantizations into configurable cards
feat(wizard): expose canonical llama.cpp runtime controls
fix(presets): enforce safe KV and launch configuration validation
feat(vram): show fingerprinted launch memory evidence
END_COMMIT_OVERRIDE
```

Remove the final `feat(vram)` line if Phase 9 is intentionally deferred from
the first release.
