# UI end-to-end suite audit

Status: findings and options. Nothing in this document has been implemented.

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

On macOS, `GET /api/app-home-migration/status` takes a constant 4.2 s per
call (Linux CI: 4 ms). The page awaits it during boot, so every local test
pays it and the local median is about 5 s. The handler's code is cheap
filesystem work, and a process sample during the call showed no busy thread,
so the cause is still unknown. It does not affect CI, but it makes local full
runs about as slow as CI and discourages running them.

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
