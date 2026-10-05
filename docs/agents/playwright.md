# Playwright UI E2E Tests (Full Reference)

## Key Principle: Protect Live Instance

Default Playwright (`npm test` with no flags) spawns a fresh `local-llm-foundry` on 7778
and kills any existing process on that port. This can terminate active models or sessions.

**ALWAYS specify a test port locally. Never run bare `npm test`.**

## Isolation Modes

- Isolated (always use this): `run-server.mjs` starts a fresh instance on the specified port
  with a brand-new temp config directory. No user data used.
- Attached (advanced): If `LLAMA_MONITOR_UI_URL` is set, connects to existing instance
  using real config. Avoid for PR validation.

## Canonical Local Run (CI-Equivalent, Isolated) — USE THIS

```bash
cd tests/ui
CI=1 LLAMA_MONITOR_USE_RELEASE=1 LLAMA_MONITOR_TEST_PORT=17778 npm test
```

- `CI=1` → workers=1, retries=2, forbidOnly=true
- `LLAMA_MONITOR_USE_RELEASE=1` → uses release binary (faster; matches CI)
- `LLAMA_MONITOR_TEST_PORT=17778` → uses 17778 (leaves 7778 untouched)

If you haven’t built a release binary: `cargo build --release` first.

## Attaching to Live Instance (Advanced)

```bash
target/release/local-llm-foundry --headless --port 17778
cd tests/ui && LLAMA_MONITOR_UI_URL=http://127.0.0.1:17778 npm test
```

Not for PR validation.

## Running Locally (Useful Variants)

- CI-equivalent: `cd tests/ui && CI=1 LLAMA_MONITOR_USE_RELEASE=1 LLAMA_MONITOR_TEST_PORT=17778 npm test`
- Headed: `cd tests/ui && LLAMA_MONITOR_TEST_PORT=17778 npm run test:headed`
- Debug: `cd tests/ui && LLAMA_MONITOR_TEST_PORT=17778 npm run test:debug`
- SSH: `cd tests/ui && npm run test:ssh` (requires LLAMA_MONITOR_SSH_TARGET)
- AI: `cd tests/ui && LLAMA_MONITOR_HAS_AI=1 npm run test:ai`

## When to Run

- Before adding `ready-to-test` label.
- After changes affecting:
  - Chat behavior (tabs, messages, compaction, guided generation)
  - Navigation, modals, panels, settings flows
  - Remote-agent or SSH-related flows
- Near end of a feature branch.

For small, clearly isolated changes (refactor, docs-only), may skip e2e but must run Rust checks and JS linting.

## CI Behavior

- Triggered by `ui` job when:
  - PR has `ready-to-test` label or is from dependabot, OR
  - `static/**` or `tests/ui/**` files changed.
- Runs: validate-js, npm ci, playwright install, npm test.
- Skips:
  - SSH tests unless `LLAMA_MONITOR_SSH_TARGET` configured.
  - AI-dependent tests (excluded from CI by default).

AI-dependent tests must:
- Be gated: `LLAMA_MONITOR_HAS_AI=1`
- Use `test.skip(!hasAi, 'Set LLAMA_MONITOR_HAS_AI=1 to run AI-dependent tests.')`
- Never be mandatory for CI.

## Known Flakiness

CI mitigates via: workers=1, fullyParallel=false, retries=2.

Debugging:
1. Run with CI-equivalent flags: `cd tests/ui && CI=1 LLAMA_MONITOR_TEST_PORT=17778 npm test`
2. If that passes, local failure was a parallel timing race
3. Isolate: `cd tests/ui && LLAMA_MONITOR_TEST_PORT=17778 npx playwright test --workers=1 --grep "test name"`

Cargo test flakiness:
- Ensure no other cargo test/build running.
- Use PID-unique temp paths: `temp_dir().join(format!("name-{}", std::process::id()))`

## Maintenance Rules

- Update tests when:
  - UI element ID/label/structure changes
  - Flow is modified
  - Feature renamed/moved
  - New non-trivial feature added
- Prune when:
  - Feature removed/deprecated
  - Test no longer reflects intended behavior
  - Test is flaky due to transient behavior
- Add when:
  - Feature is visually/functionally significant
  - Test is deterministic, fast, and AI-independent (preferred)
- Avoid:
  - Tests relying on exact visual layout (use screenshot harness)
  - Tests depending on external AI without gating
  - Tests too tightly coupled to internals

## Performance Baseline

`tests/ui/core/performance.spec.js` tracks JS files loaded on cold page load.
Ceiling stored in `tests/ui/core/js-module-baseline.json`.

Whenever you add a new `.js` under `static/js/` imported from `bootstrap.js`:
```bash
cd tests/ui && npm run update-baseline
```
Commit both the new module AND updated baseline file.
