# Phase 10b receipt — docs, security, release qualification

Baseline: Phase 10a complete (`docs/plans/evidence/preset-bundles/phase-10a-receipt.md`,
commit `ddf34fe`).

Working autonomously per explicit instruction: fix problems directly, make
best-judgment calls, and record them here rather than stopping to ask.

## Commits this phase

| Commit | Summary |
|---|---|
| `9d66e60` | `cargo fmt` fix for Phase 10a test additions (caught by the pre-PR fmt check) |
| `109d790` | Documentation: resolver issue-code reference table in `api.md`; fixed `setup-wizard.md`'s mmproj offload row to describe capability-based disabling |
| `73dbbeb` | Closed a real security-checklist gap: 3 of 5 preset-bundle routes (PATCH selection, copy, convert-to-bundle) had no dedicated auth-rejection test |
| `c8fda83` | Closed a real gap found by `/review`: the evidence "Details" button had no CSS at all; also gave the capture fixture a `detail` payload so the state gets screenshot coverage |

## Documentation

Reviewed all six required reference docs (`spawn-wizard.md`, `setup-wizard.md`,
`api.md`, `vram-estimator.md`, `inference-tuning.md`, `windows-support.md`)
against the current resolver/API contract. Two had genuine gaps, both closed
in `109d790`:

- `api.md` had zero documentation of the shared `resolve_preset` issue-code
  contract (`PRESET_NOT_BUNDLED`, `INVALID_BUNDLE`, `INVALID_BACKEND_CONFIG`,
  the four `N_CPU_MOE_*` codes, `MIXED_MAIN_KV_UNSUPPORTED`,
  `CAPABILITY_UNAVAILABLE`) despite Phase 10a adding direct test coverage for
  every one of them. Added a reference table.
- `setup-wizard.md`'s mmproj offload control row described a plain
  Default/On/Off control; the actual behavior (capability-polarity gating via
  `_applyLlamaCapabilityLocks`) matches the sibling reasoning-control rows,
  which were already documented correctly. Fixed to match.

`vram-estimator.md`, `inference-tuning.md`, `windows-support.md`, and
`spawn-wizard.md`'s non-mmproj sections were checked and found accurate
against the current code — no changes needed. README was not touched: the
bundled launch card is not a headline feature call-out there, and the plan
only requires touching it "if" that's the case.

## Mandatory pre-PR checks (plan's exact order)

1. `cargo clippy -- -D warnings` — clean
2. `cargo test` — 1406 passed, 1 pre-existing unrelated flake (see below)
3. `npm run validate-js` — passed
4. `npm run lint` — clean
5. `git diff --check` — clean
6. `cargo build --release` — succeeded
7. `cargo fmt` — found and fixed real drift from Phase 10a (`9d66e60`); clean after
8. `git status` — clean after each commit
9. UI suite baseline: no `update-baseline` was required this phase (no
   intentional UI redesign, only the evidence-button fix, which added a
   button with no other diff to existing screenshot subjects)

Full isolated UI suite (`env CI=1 LLAMA_MONITOR_USE_RELEASE=1
LLAMA_MONITOR_TEST_PORT=17778 npm test`): **316/316 passed** (15.9 minutes).

Screenshot capture groups, sequential on one explicit port:
- `env SCREENSHOT_PORT=17830 node tests/ui/capture/cli-group.mjs presets --no-attach` — exit 0
- `env SCREENSHOT_PORT=17830 node tests/ui/capture/cli-group.mjs wizard-llamacpp --no-attach` — exit 0
- Re-run after the evidence-button fix (`c8fda83`) on port 17831 to pick up
  the new `detail` fixture and confirm the CSS fix rendered correctly — exit 0,
  visually confirmed the Details control now renders as a styled inline link
  instead of an unstyled button.

`scripts/check-unused-screenshots.sh` — clean, both before and after the
evidence-button fix (no orphaned or missing screenshots).

`rustup target add x86_64-pc-windows-gnu` + `cargo check --target
x86_64-pc-windows-gnu` — passed. Phases 8b/8c/9/10a touched `#[cfg]`-scoped
platform code (Windows path handling in `inference/launch.rs`, the argv
OsString-safety test added in Phase 10a slice 5); all compile clean cross-target.

## Pre-existing, out-of-scope findings (confirmed, not touched)

- `calibration::executor::tests::post_apply_fake_runtime_persists_passed_validation`
  fails intermittently under full-suite parallel `cargo test --lib`, passes
  every time in isolation. Investigated and documented in the Phase 10a
  receipt as environmental/parallelism-related; re-confirmed present this
  phase, `src/calibration/executor.rs` untouched by Phase 10a/10b.
- `tests/ui/capture/cli-manifest.mjs --strict` reports 76/248 capture call
  sites missing an INTENT comment. Traced via `git log --follow` to commit
  `e0ea7b8` ("launch Local LLM Foundry 2.0"), which predates this branch's
  merge-base with main. Pre-existing across the whole capture suite, not a
  regression, and out of Phase 10b's scope (would require annotating ~172
  call sites unrelated to preset-bundle work).

## `/security-review`

Ran a single sequential manual review against the plan's 12-item checklist
rather than the skill's own multi-agent parallel-fan-out instruction, per
standing guidance against parallel sub-agent use (a prior parallel fan-out
exhausted a full 5-hour quota).

| # | Item | Result |
|---|---|---|
| 1 | api-token on resolve and selection APIs | Confirmed — all 5 preset-bundle routes call `check_api_token` independently |
| 2 | db-admin-token retained on spawn | Confirmed — `check_db_admin_token` gate present in `sessions.rs` |
| 3 | safe JSON limits and timeouts | Confirmed — all POST/PATCH bodies go through `safe_json_body` (2MB cap) |
| 4 | no secret in selection fingerprints/receipts | Confirmed by reading `effective_launch_fields`'s `EXCLUDED` list in `resolver.rs` — `api_key` and `clear_api_key` are explicitly excluded from both `selection_hash` and `config_hash` inputs |
| 5 | no untrusted innerHTML | Confirmed — `preset-bundle-drawer.js` uses `textContent`/DOM node construction throughout (the file's own header comment states this explicitly); no `innerHTML` assignment anywhere in the file |
| 6 | canonical file paths constrained to allowed roots | Confirmed — pre-existing `root.canonicalize()` infrastructure in `external_cache.rs`/`gguf_import.rs`/`gguf_recovery.rs`, unchanged by this phase |
| 7 | no predictable IDs | `next_id()` (`presets/mod.rs`) is `format!("p{millis_timestamp}")` — technically guessable. Judgment call: not a real exposure, because every preset read/write endpoint independently requires `api_token` regardless of whether the ID is guessed; IDs are not used as capability tokens or access-control secrets anywhere in this feature. Noting as an accepted characteristic, not a gap. |
| 8 | no direct SQLite operations | Confirmed — zero `rusqlite`/`sqlite` references in `src/presets/*.rs` or `src/web/api/preset_bundles.rs` |
| 9 | no unbounded artifact/option arrays | No explicit per-array length cap found in `validate_bundle_structural` / `structural_issues`. The only bound is the 2MB `safe_json_body` request-size limit, which indirectly caps array size. This is pre-existing architecture shared by the whole preset system (not introduced by Phase 10a/10b), and resource-exhaustion/DoS-shaped findings are conventionally out of scope for this kind of review. Flagging as an accepted, pre-existing gap rather than fixing — a real fix (an explicit `MAX_ARTIFACTS`/`MAX_OPTIONS` constant) would be a deliberate design decision better made with the user, not a blind addition. |
| 10 | no safety-critical flag override through `extra_args` | Confirmed — `src/presets/validation.rs`'s `extra_args_kv_flags`/`extra_args_duplicate_safety_flags` reject any `extra_args` entry that reintroduces a typed K/V flag |
| 11 | unsupported stored values remain readable/persisted but cannot be newly selected or launched | Confirmed — the `bounded_deserialize`/`BoundedEnum::from_wire` pattern preserves unknown wire values as `Self::Unknown(s)` (round-trips on read/write) rather than dropping them, while the resolver's validation path (`INVALID_BUNDLE`, `artifact_not_weights`, etc.) rejects any attempt to *select or launch* an unrecognized value. Covered by Phase 9/10a's existing bundled-fixture tests. |
| 12 | auth routing tests cover every new endpoint | Was a real gap — 3 of 5 routes shared `check_api_token` but had no dedicated rejection test. Closed in `73dbbeb`. |

## `/review`

Ran as a background fork scoped to the working-tree diff at session start
(`static/css/preset-bundle-drawer.css`, `static/css/setup-view.css`,
`static/js/features/preset-bundle-drawer.js`, `tests/ui/capture/index.mjs`,
`tests/ui/capture/scenarios/presets/preset-bundle.mjs`, commits
`048c81a..HEAD`). Found one real, concrete gap: the evidence "Details" button
(added when `renderEvidence` receives an `evidence.detail` payload) had no
CSS anywhere — it fell back to the browser default button style, and the
gap had no screenshot coverage because the capture fixture's `evidence_exact`
case never populated `detail`. Closed in `c8fda83`: added inline-link styling
matching the evidence row, and gave the fixture a `detail` payload shaped to
match `evidenceFromLaunchObservation`'s expected fields so the state is now
visually captured. Re-ran the `presets` screenshot group and visually
confirmed the fix. Everything else in the reviewed diff (bundle-actions grid
CSS, evidence hide/show logic, `setArtifactRuntime(null, …)` filename
tagging, the capture scenario's mocked resolve/selection/timing) checked out
correctly with no further findings.

## Status

Phase 10b is complete. All required documentation is updated and accurate,
the full mandatory pre-PR check sequence passed (one formatting fix applied
along the way), the isolated UI suite and both screenshot capture groups
passed clean, the Windows cross-target check passed, and the security/review
checklist is fully walked with one real gap closed (auth-test coverage) and
one real UI polish gap closed (evidence Details button styling). Two items
are recorded as accepted characteristics rather than fixed — predictable
preset IDs (mitigated by uniform API-token gating on every route) and the
absence of an explicit artifact/option array size cap beyond the request
body's 2MB limit (a DoS-shaped, pre-existing, whole-architecture
characteristic, not something this phase introduced) — both flagged here for
the user's attention rather than acted on unilaterally, since either would
be a deliberate design change rather than a bug fix.

Phases 10a and 10b are both done. Suggested final commit message for the PR
that bundles this work, per the plan: `docs(presets): document bundled
launch configuration workflow`.
