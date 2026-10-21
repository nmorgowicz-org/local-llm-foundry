# Phase 5 receipt: build API and runtime UI

Baseline: `5b9da335`, branch `feat/llamacpp-source-builds`.

## Delivered

- **API** (`src/web/api/llama_build_routes.rs`, `src/llama_build/service.rs`): the twelve
  `/api/llama-build/*` routes of plan §7 plus `provenance` on `GET /api/llama-binary/version`.
  All are token-authenticated and listed in the 401 smoke table. Documented in
  `docs/reference/api.md`.
  - Single-flight build and validation. Build, validate, clean and install are mutually
    exclusive through the shared ownership table; conflicts return `409` with the owner.
  - `start` returns the pinned commit; there is no time-estimate field anywhere.
  - Cancel is refused (`409`, "cannot cancel safely") from the promote phase on.
  - Job ids are validated before any filesystem access; logs are served from the job's log
    file with an incremental line index and page caps.
  - `brew_install` is macOS-only, direct argv, allowlist `git/cmake/ninja/ccache`.
  - The app promotes through `LiveHost`, so a running model is restarted and a restart failure
    is reported as `succeeded_restart_failed`, distinct from a failed build.
- **UI**: `llama-runtime-panel.js/.css` (panel), `llama-runtime-state.js` (pure view model),
  extended `llama-updater.js` pill. The pill shows idle (with `src`/`prebuilt`/`custom`
  suffix), update, building with phase and `[n/m]` percent, red failure, amber
  installed-but-restart-failed, and rolled-back. The version modal is the popover: it gains
  "Build … from source", "Open llama.cpp Runtime…" and a conditional replacement caution that
  appears only when a source build is live and names no K/V pair list.
- **Panel**: error banner (phase, exit code, first error, last log lines, hint action, copy
  diagnostics, dismiss), current runtime with provenance-hash status and rollback, prerequisites
  with copy/link/brew fixes and details, build form (CUDA controls only when the platform has
  them), cancel with a reason once non-cancellable, live log with follow, disk and clean.
  All server text is inserted with `textContent`.
- **Error catalog**: the fourteen fixture logs from Phase 4 are unchanged; the UI renders the
  hint actions `wipe_build_dir`, `leave_headroom`, `clean_build_cache`, `revalidate`, copy fixes
  and links.

## Defect found while qualifying

A shallow source checkout embeds `build 1`, so `GET /api/llama-binary/version` reports
`build: 1` for every foundry source build. The pill would have shown `b1` and a permanent
"update available". The client now uses the recorded tag (`bNNNN`) for a source build and the
reference (for example `master`) when there is no tag; covered by Playwright. The server's
`build` field is unchanged.

## Verification

- `cargo clippy --all-targets -- -D warnings`, `cargo test` (no failures), `npm run validate-js`,
  `npm run lint`, `git diff --check`, `cargo build --release`, `cargo fmt --check`.
- `cargo test web::api`: 332 passed; `llama_build` unit tests plus the three integration suites
  passed; `tests/auth_routing` 43 passed.
- Playwright (`core/llama-runtime-panel.spec.js`, 14 tests, plus the existing pill, kv-support,
  html-integrity, performance, app-shell, console-errors and security-auth specs) against the
  release binary on a non-default port. Cases: prerequisite fail disables Build and names the
  check; a warning does not; failed job turns the pill red with full diagnostics and a hint
  action; source-build provenance and hash status; master reference with no bogus update;
  conditional prebuilt caution (absent for a prebuilt runtime); restart failure amber state with
  rollback and diagnostics; running build progress with Cancel disabled once promoting;
  clean disabled while another owner holds the tree; no CUDA controls on macOS; hostile server
  text rendered inertly. The unknown-capability no-rebuild-link and no-launch-block cases are
  covered by the Phase 2 `kv-support.spec.js`.

### Manual macOS run through the HTTP API (isolated `HOME` and root, release binary)

- Unauthenticated request: `401`.
- `POST /prereqs/validate` ran the full report including the smoke compile; all checks passed
  (`ccache` warn only).
- `GET /options` resolved the latest release (`b11740`, `6fa4f6c…`) and `master`.
- `POST /start` returned `202` with the pinned commit. A second start and a clean during the
  build each returned `409` with owner `build`.
- The job went fetch → compile (309 steps, progress visible) → promote (`cancellable: false`) →
  `succeeded`, "Installed b11740 · src · metal.", 415 log lines.
- `GET /api/llama-binary/version` reported provenance `foundry_source_build`, exact commit,
  `hashes_match: true`; `runtime-status` showed the install transaction; cache sizes were
  reported; cache-cap `1` and `brew_install rm` were rejected with `400`.

## Not exercised

- The macOS "missing tool → explicit one-click `brew install`" step was not run (the machine
  already had every tool); the action is covered by route and Playwright tests only.
- The Windows CUDA validate → compile → promote run used the CLI (Phase 4), not the HTTP routes.
  The routes drive the same orchestrator, but a real CUDA build through the panel was not run.
- Windows `stale` on the prerequisites route cannot reproduce the MSVC/SDK fingerprint parts of
  a full pass, so only a missing report is stale there (documented in `api.md`).
- Linux paths remain untested.
