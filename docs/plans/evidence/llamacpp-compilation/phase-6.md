# Phase 6 receipt: Doctor, docs, review hardening

Baseline: `b0ad5e15`, branch `feat/llamacpp-source-builds`.

## Delivered

- **Doctor** (`src/llama_build/doctor.rs`, wired in `src/web/api/doctor.rs`): pure findings from
  file reads only. Interrupted runtime update, unusable journal, unresolved build process
  records, installed files that no longer match recorded hashes, old pre-journal rollback
  backup folders (counted, never deleted), a failed or more-than-30-day-old prerequisite check
  (worded as relevant only to building), and a quiet "capability probe unavailable" warning.
  While a build, update or recovery is running, journals/process records are not reported as
  leftovers. A rebuild suggestion is appended to a K/V fallback finding only when the classifier
  confirms a missing buildable CUDA kernel; unknown evidence never produces one.
- **Docs**: new `docs/reference/llama-runtime.md` (builds, panel, prerequisites per OS, CLI, K/V
  support table, troubleshooting, process-record repair, rollback, file locations);
  `api.md`, `cli-flags.md`, `windows-support.md`, `README.md` and the docs index updated.
  CHANGELOG is produced by the release flow and is not edited by hand.

## Reviewer pass (static) and what it changed

A read-only reviewer audited Phases 4-6. No P0. Findings fixed:

1. **P1: a hard crash on Unix permanently blocked building, cleaning and startup recovery**
   with no sanctioned repair. Added `repair.rs`, `POST /api/llama-build/repair-processes`
   (`{"confirm": true}`), CLI `llama-build --repair-processes` and a panel banner button. It
   refuses while a recorded process is alive or any other process references the build
   directory, and never kills anything. Interrupted-update journal replay now also runs while
   process records exist; only the orphan sweep stays deferred.
2. **P1: doctor reported false Issues during a running build/update.** Fixed with a busy fact.
3. **P1: the "Update rolled back" pill could never appear** (the journal serializes outcomes as
   `{kind, ...}`). Fixed with a real-shape test and a dismissible banner.
4. **P1: a failed model restart after a rejected-and-restored update was dropped.** The
   rolled-back outcome now carries `restart_failed`; replies and the job text say so.
5. **P1: doc inaccuracies** (CUDA-only flag claim, log location, a nonexistent Retry control and
   `last build took`). Corrected.
6. **P2s**: Build and Validation now conflict and process records are written atomically with
   unique temp files; secrets (`*KEY*`, `*TOKEN*`, `*SECRET*`, `*PASSWORD*`, `*CREDENTIAL*`,
   `LOCAL_LLM_FOUNDRY_*`, `LLAMA_MONITOR_*`) are stripped from build children with a
   build-needed allowlist; journal `staged` must be one staging-prefixed path component;
   staged files and the manifest are fsynced before the journal; the runtime gate is re-checked
   inside `start_server` after taking `server_lifecycle`; cancel validates the job id; child
   output lines are capped at 64 KiB; the panel no longer makes hint actions sticky, bypass the
   disabled reason or show CUDA controls on a non-CUDA build; build polling now also starts
   offline.

Not changed (reported, not confirmed, or outside this feature): Origin-guard ordering in
`web/mod.rs` (every handler checks the bearer token itself), the `linkPrRefs` regex over HTML
(predates this work), poll serialization and per-tick re-render in the panel.

## Verification

Final macOS chain: `cargo clippy --all-targets -- -D warnings`, `cargo test`,
`npm run validate-js`, `npm run lint`, `git diff --check`, `cargo build --release`,
`cargo fmt --check`; 40 Playwright tests passed across the runtime panel, pill, K/V,
HTML integrity, performance, console-errors and security-auth specs.
`cargo check --target x86_64-pc-windows-gnu` also passed (library only).

Native Windows confirmation on ryne, normal checkout on `feat/llamacpp-source-builds`
at `7d5b7af7`, with `CARGO_PROFILE_TEST_DEBUG=0`:

- `cargo test --lib llama_build`: 292 passed.
- `cargo test --lib llama_runtime`: 78 passed.
- CLI integration: 8 passed; orchestration: 11 passed; Windows Job Object: 4 passed.
- The first attempt did not update the checkout because generated files blocked the pull;
  its old-snapshot results are not counted. After updating, one environment test assumed
  `PATH` rather than Windows `Path`. The assertion was made case-insensitive in
  `7d5b7af7`, and all five suites then passed.

## Manual QA matrix status (plan §10)

Covered by real runs: macOS and Windows CUDA prerequisite/smoke, build of the latest tag with
pinned commit and journaled promote, incremental rebuild, model smoke (Metal and CUDA mixed
q8_0/q4_0), Windows Job Object cancel / parent-death / grandchild containment, Unix supervisor
containment, panel and API behavior on macOS. Covered by automated fault-injection tests only:
promotion disk exhaustion, failure before/after every rename, recovery twice, concurrent
ownership matrix, K/V matrix, legacy cache reprobe. Not run for real: the macOS one-click
`brew install` with a genuinely missing tool, a real CUDA build driven through the panel, the
Windows locked-DLL promote, restart mid-build on both platforms, and everything on Linux (no
Linux machine). The post-review repair, restart-on-reject, secrets scrub, journal validation
and durability changes have now also passed the native Windows suites above; a real CUDA
compile/model smoke was not repeated after those changes.
