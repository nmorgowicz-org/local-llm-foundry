# UI end-to-end suite audit

Status: implemented on `chore/ui-e2e-suite-audit`. The findings and options below are the
original audit; the [Outcome](#outcome) section records what was done and measured, and
corrects the findings that the measurements disproved.

Source data: the `ui-reports-shard-{1,2}` artifacts from CI run 39410181437
(PR #467, commit `5db54230`), decoded from the Playwright HTML reports, plus
that run's job and step timings and the retry trace for
`core/console-errors.spec.js`.

## Findings

### 1. The suite is flat, not heavy

| Measure | Value |
| --- | --- |
| Tests executed | 599 (about 300 per shard) |
| Sum of test time | 48.2 min (shard 1: 23.9, shard 2: 24.3) |
| Median / p90 / p99 / max | 4.4 s / 7.7 s / 12.3 s / 17.8 s |
| Tests under 3 s | 97 (3.5 min) |
| Tests 3-6 s | 374 (27.1 min) |
| Tests 6-10 s | 110 (13.8 min) |
| Tests over 10 s | 18 (3.7 min) |

There are no pathological tests. 581 of 599 finish in under 10 s, and 374 land
in a 3 s band. The cost is a roughly constant per-test overhead multiplied by
the test count, which is why it grows linearly with every feature.

Largest files by total time: `rapid-download-ui` 4.5 min (47 tests),
`preset-flow` 3.2 (34), `rapid-preset-throughput` 3.0 (24), `chat-behavior`
2.6 (23), `spawn-wizard` 2.3 (37), `app-shell` 1.9 (23).

### 2. Page boot is about half of every test

The CI trace for a representative test shows `page.goto('/')` taking 1.7 s
(the `load` event over roughly 110 unbundled ES modules and 28 stylesheets) and
the `html.modules-ready` wait adding 0.5 s: about 2.2 s of a 4.4 s median test
is boot, paid again for every test because each test gets a new browser
context. At 599 tests that is on the order of 20+ minutes of the 48.

Boot cost on a developer machine is far smaller (the page loads in about
0.2 s), so the CI cost is dominated by runner CPU, not by the network.

Caveat: this comes from one trace of a failing test that also carried retry
tracing overhead. It should be confirmed with a trace of a passing test before
being used to size any change.

### 3. Pipeline wall time is dominated by serial stages, not test count

CI run 39410181437 took 96 minutes. The critical path is serial:

| Stage | Duration |
| --- | --- |
| `check` (fmt, clippy, tests) | 24 min |
| `ui-build` (`cargo build --release`) | 33 min build + 4.5 min cache save, 40 min job |
| `ui (shard N/2)` | 27 min each |

`ui-build` needs `check` and `check-windows` by design (do not spend the long
suite on a broken build). The UI tests are only the last 27 of 96 minutes.
`check` and `ui-build` each compile the project from scratch.

### 4. Retries multiply the cost of broken tests

`retries: 2` in CI means a deterministically failing test runs three times.
`console-errors` took three attempts to fail with identical output.

### 5. Fixed sleeps are a small cost

28 `waitForTimeout` calls totalling 15,050 ms across 13 files. Replacing every
one would save at most 15 s of delay in a single pass of the suite (the real
benefit is lower flakiness, not time).

### 6. Local macOS runs pay an extra 4.2 s per test

On macOS, `GET /api/app-home-migration/status` took a constant 4.2 s per
call (Linux CI: 4 ms). The page awaits it during boot, so every local test
paid it. **Cause found:** after a completed migration the app leaves a
multi-MB receipt file (14 MB here), and the status check parsed it with an
unbuffered `serde_json::from_reader(File)`, one `read` syscall per byte. It
was a product bug, not test tooling: every real user with a completed
migration waited about 4 s on every page load. Fixed; see Outcome.

### 7. Not yet analysed

Overlap between specs, and tests that would be cheaper at a lower level, have
not been reviewed. Candidate groups to review first: `rapid-*` (about 10
files), `spawn-wizard*` and `phase6/7-*`, `preset-flow`, and the 88
`@in-memory-test` and 32 `@fake-data-bypass` tests, which exercise page-side
logic that may not need a browser at all.

## Options

Estimates are rough and assume the figures above. None are measured.

| # | Option | Effort | Likely saving | Risk |
| --- | --- | --- | --- | --- |
| A | Start `ui-build` in parallel with `check` (needs only `paths-filter`); keep `ui` gated on both | Small, CI-only | about 24 min wall per run | Wastes about 33 runner-min when `check` fails |
| B | Cache or reuse the compiled binary: build once, share between `check`, `ui-build` and `release-smoke` | Medium | up to 20-30 min wall, plus runner minutes | Cache-key correctness |
| C | Reduce boot cost per test: reuse one page across read-only tests in a file; or share storage state | Medium | up to about 2 s per affected test | Loses per-test isolation; order dependence |
| D | Cut the number of modules loaded on boot (bundle, or defer non-essential features); also benefits users | Larger, product change | about 1 s per test | Touches production code and CSP/SRI |
| E | Per-worker server instances (own port and config dir) so `workers > 1` is safe | Medium | up to half of suite time per extra worker | Needs runner CPU headroom; 13 specs mutate shared state |
| F | Keep two shards but balance them by measured duration instead of file order | Small | Negligible now (shards are 23.9 vs 24.3) | None |
| G | Tier the suite: small smoke set on every push, full suite on `ready-to-test` and nightly | Medium | PR feedback in minutes | A regression found later than today |
| H | Demote tests that only check page-side logic to lower-level tests (Rust API tests, JS unit tests) | Larger, judgement per test | Removes tests from the suite entirely | Coverage must be matched |
| I | Delete duplicated UI coverage after an overlap review | Larger, judgement | Proportional to deletions | Coverage loss if misjudged |
| J | Lower CI `retries` from 2 to 1 (or 0 with a flake quarantine tag) | Tiny | Up to 2x on broken tests | Flakes fail runs |
| K | Replace the 28 fixed sleeps with condition waits | Small | 15 s plus less flakiness | None |
| L | Investigate the macOS 4.2 s `app-home-migration/status` stall | Small | About 4 s per test locally, nothing on CI | None |

Unknown that decides E: how many vCPUs the `arc-llama-monitor-fast` runner
has. If it is already CPU-bound (boot cost points that way), more workers
will not help and C or D are the better levers.

## Suggested order

1. A (and B if feasible): the largest wall-time win with no test changes.
2. L and K: cheap, remove noise.
3. Confirm item 2 with a passing-test trace, then choose between C, D and E.
4. G for the long term, so growth no longer lengthens every PR run.
5. H and I only after an overlap review of the groups in finding 7.

## Outcome

Every option was examined. Six were implemented, three were measured and
declined with data, and two were partly implemented. Numbers are from this
macOS machine (18 cores) with `--retries=0`; CI runners are slower, so use
ratios, not absolute times.

### Results

| Run | Before | After |
|---|---|---|
| Full suite, serial | 911 s, 599 tests, **2 failing** | 816 s, 551 tests, 0 failing |
| Full suite, 4 isolated processes | 294 s (same 2 failing) | 281 s, 0 failing |
| Smoke tier | n/a | 59 s, 45 tests |
| Median test | 1.40 s | 1.39 s |
| `GET /api/app-home-migration/status` | 4,200 ms | 30 ms |
| Page boot to `modules-ready` | about 5 s | 0.26 s |

The suite-run "before" figures were measured after the migration fix, so they
exclude its effect; the last two rows show that effect separately (before it,
each local test paid about 4 s more). The two failures in "before" were real
and are described below.

### Decisions, per option

| # | Option | Decision | Evidence |
|---|---|---|---|
| A | `ui-build` alongside `check` | **Done** | `check-windows` already runs this way. Expected saving is the duration of `check` (about 24 min per run); not yet observed in CI. |
| B | Faster-to-build UI binary | **Declined** | A clean `opt-level = 1` build took 230 s against 200 s for the release build, so there is nothing to gain. The 33 min CI build is a cold dependency cache on a slow runner; see follow-ups. |
| C | Reuse a page across tests | **Declined** | Hoisting setup saves nothing because every test gets a fresh context. Reusing a page would need isolation guarantees the shared `wizardState` and per-test mocks do not give. |
| D | Fewer modules on boot | **Declined** | Profiled at 4x CPU throttle: JavaScript is 136 ms of 747 ms. The rest is native parse, style and layout of a 5,600-line `index.html` and 28 stylesheets. Lazy-loading JS cannot recover it. |
| E | Parallel isolated servers | **Done, default off in CI** | `tests/ui/run-parallel.mjs` runs P Playwright processes, each with its own server, port, config dir and shard. Measured 3.1x at P=4. CI uses it only when the repository variable `UI_PROCS` is above 1, because the runner's CPU count is unknown. |
| F | Balance shards by duration | **Declined** | Shards were 23.9 and 24.3 min. Nothing to balance. |
| G | Tiered suite | **Done** | `npm run test:smoke` (17 files, 59 s). In CI, add the `ui-smoke` label next to `ready-to-test` to run only that tier. No nightly job was added. |
| H | Demote logic-only tests | **Done (9 of 13)** | Nine tests now run under `node:test` in `tests/ui/unit/` and in the `lint` job. Four more need a `window`/`localStorage` shim and were left. |
| I | Delete duplicated coverage | **Done (39 tests)** | Parameter rows hitting the same branch, tests fully covered by a longer one, status-below-500 API checks, and tests with no reachable assertion. |
| J | CI retries 2 to 1 | **Done** | A deterministic failure now costs two runs, not three. A pass-on-retry still reports as flaky. |
| K | Replace fixed sleeps | **Done (24 of 28)** | Four short negative waits remain, each commented. |
| L | macOS 4.2 s stall | **Done** | Receipt parsing fix above. |

### Defects the work surfaced

- **Migration receipt parsing (product).** Fixed in `src/app_migration.rs`;
  all migration JSON readers are now buffered, and the completed-receipt
  check decodes only the two root paths.
- **Axe `label-content-name-mismatch` (product, accessibility).** The KV
  scenario cards in the spawn wizard were `role=button` with an `aria-label`
  that did not contain their visible text. The axe-core 4.14 bump that arrived
  with `main` made this fail the Phase 11 accessibility test. Fixed by letting
  the visible content name the button. This would have failed the next CI run.
- **A test that had stopped testing (`kv-support.spec.js`).** The stale-response
  test held "call number 1", but the wizard now issues its own q8_0 lookup
  first (q8_0 became the default on this branch), so nothing was ever held. A
  300 ms sleep hid it. Fixed to hold the first q4_0 lookup.
- Four more tests were removed because they could not fail
  (`launch-grid` x3 returned early; `performance` timed a function that is
  never defined).

### Not done, and worth a follow-up

- **Cold Rust cache in `ui-build`.** The release build took 33 min in CI against
  3.3 min clean locally. The workflow saves the `check` cache only from `main`, and
  `ui-build` never runs there, so the first run of every PR appears to build
  all dependencies cold (inferred from the config and timings, not observed). Warming a release-profile cache from `main` would likely cut
  this sharply; it cannot be verified outside CI.
- **`UI_PROCS`.** Find the runner's vCPU count, then try `UI_PROCS=2` or `3`.
- **Audit findings left unactioned** (see the OverlapAuditor report in the
  session notes): 5 dead-module tests (`rapid-mlx-cards.js` is not imported by
  the app), 4 pure-logic tests that need a shim, 2 API tests better placed in
  Rust, a duplicated poll state machine in the Rapid download code that would
  let about 18 more tests move to Node, and a three-test skipped
  `phase7-command-preview` group that has never run.
- **Unverified possible bug:** `static/js/features/settings.js` sends
  `custom_cert_path` while `src/web/api/tls.rs` reads `customCertPath`.
  Worth a check before anything else here.
- **Test bug:** `app-shell.spec.js` reads `route.request().postDataJSON || {}`
  without calling the function, so the SSH-target assertions in that test
  never run. `console-errors.spec.js` has no `pageerror` listener.

### How to use what was added

| Command (from `tests/ui`) | Does |
|---|---|
| `npm run test:smoke` | 45-test smoke tier |
| `npm run test:unit` | 9 node:test checks, no browser |
| `npm run test:parallel -- --procs 4` | full suite as 4 isolated processes |
