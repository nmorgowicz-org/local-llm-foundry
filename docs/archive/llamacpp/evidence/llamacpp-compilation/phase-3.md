# Phase 3 receipt: shared promotion, retention, rollback

Baseline: `d5a61dcf` foundation, then `336085c4`..`7a88f2e2`. Branch `feat/llamacpp-source-builds`.

## Delivered

- `src/llama_runtime/promote.rs`: journaled transaction engine for install, rollback, uninstall.
  Full step list and metadata fingerprints are journaled before the first mutation; a rename's
  completion is derived from "its source is gone", so recovery is replayable. Phases Begun,
  Swapping, Committed. Faults hook at every journal write, rename, verify, stop/restart, cleanup.
- `gate.rs` (single runtime gate, swap window, owner token), `verify.rs` (one verifier for every
  install kind), `manifest.rs` (provenance only), `host.rs` (model stop/restart, verify wiring),
  `startup.rs` (recovery before pollers/server; only the active root is swept).
- `api_llama_binary_update` rewritten onto `promote`; `POST /api/llama-binary/rollback`,
  `GET /api/llama-binary/runtime-status`; uninstall uses the gate and journal.
- Launch refused and capability probing paused during a swap (owner restart passes).
- `AppPaths::bin_previous_dir` / `runtime_transaction_file`.
- Docs: `docs/reference/binary-lifecycle.md`, `docs/reference/api.md`.

## Evidence

| Command / run | Result |
| --- | --- |
| `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings` | 0 |
| `cargo test --lib` (macOS) | 1745 passed, 15 ignored |
| `cargo test --lib -- llama_runtime ...` on Windows (ryne, RTX 5090) | 70 passed, incl. `a_locked_file_in_the_live_set_fails_the_rename_and_recovery_restores` |
| `npm run lint`, `npm run validate-js`, `git diff --check` | 0 |

Fault suite: failure **and** death injected at every boundary for install (with and without a
running model), rollback and uninstall; recovery replayed twice; first install, journal-write
failure, locked DLL (Windows), corrupt/ambiguous/torn journal, rollback twice, failed rollback
rename, compensating rename that dies. The sweep found one real recovery bug (rejected candidate
deleted inside `reconcile` made a replay misread the restored live set); fixed.

Manual swap with real b11727 / b11734 release downloads, isolated app root, same script both OSes:

| Check | macOS (Metal) | Windows (CPU x64) |
| --- | --- | --- |
| Rollback with no previous | `failed`, nothing changed | same |
| Update, update | live = new, previous = old, 62 files each | live = new, previous = old (seed set first), 53 files |
| Rollback, rollback again | sets swap, then return to original | same |
| Rollback during in-flight update | 409 `{owner: update}` | same |
| Model launch during update | refused with "runtime is busy" | n/a |
| Update with model running | `promoted`, `server_restarted: true`, new pid | same (DLLs of the running model locked and released) |
| Restart forced to fail | `installed_restart_failed`, `ok: true`, `restart_error`, binaries kept | same |
| Rollback after that | works | works |
| Leftovers after each | only `runtime-transaction.last.json` | same |

## Defects found by the manual runs (and fixed)

1. Windows CPU release loads an `RPC` plugin beside `CPU`; it classified as `Unknown` backend and every
   Windows CPU install rolled back. Real probe output captured and added as a regression test.
2. `api_llama_binary_update` kept a stale `#[cfg(any(target_os = "macos", test))]`, so the app
   did not compile on Windows. Windows build caught it.
3. Startup sweeping deleted older pre-journal `bin-previous-*` backups, and my first macOS run
   overwrote the real `~/.config/llama-monitor` runtime (see below). Sweeping now never deletes a
   legacy backup (adopts the newest into an empty slot by rename) and only runs on the active root.

## Incident to disclose

My first macOS manual run was not isolated: `--config-dir` pointed at a temp root, but the existing
`managed_llama_server_path` fallback resolves to a healthy binary in the canonical/legacy roots when
the app root has none (intended for the restartable 2.x migration), so promotion targeted the real
`~/.config/llama-monitor/bin`. Later runs set `HOME` to a temp dir as well. It replaced
`~/.config/llama-monitor/bin` (b11436) with downloaded b11734 builds and collapsed the timestamped
backups. The original three sets were recovered from the 2026-10-18 Time Machine local snapshot
(mounted read-only) into `~/.config/llama-monitor/recovered-runtimes/`. Nothing was lost; the live
`bin` still holds b11734 prebuilt and `bin-previous` a copy of it. Restore b11436 by moving the
recovered folder back if wanted.

## Deviations from the plan

1. No journal write per rename: the step list plus phase transitions are journaled up front.
2. Metadata (path/size/mtime) fingerprints, not content hashes.
3. Verification probe timeout is 30 s (plan said 10 s): cold macOS Metal start can exceed 10 s.
4. No automatic model restart after a crash recovery (journal stores no launch config or
   credentials); startup logs and reports a pending restart.
5. A custom `llama_server_path` now installs to the managed `bin/` and is reported as `custom_server_path`.
6. The old test `promoted_binary_failure_restores_previous_directory` was removed with the
   function it tested; transactional restore is covered by the fault suite.

## Not done

- Disk-space reservation for staging (journal-write failure is tested, space is not pre-checked).
- The llama.cpp panel UI for rollback/status is Phase 5; the routes exist now.
