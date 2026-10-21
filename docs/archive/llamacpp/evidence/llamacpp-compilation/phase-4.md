# Phase 4 receipt: source-build backend and CLI

Baseline: `6ad3da68`, branch `feat/llamacpp-source-builds`.

Status: complete.

## Delivered

- Rust build units under `src/llama_build/`: paths, GPU discovery, prerequisite checks,
  vcvars capture, source pinning, configure argv, staging, provenance, process supervision,
  cache ownership, job state, orchestration, and standalone CLI.
- `local-llm-foundry llama-build --dry-run` checks tools/network and prints the resolved
  upstream tag/commit and lossless configure/build argv without creating application state.
  Execution supports latest `bNNNN` or `master`, Metal/CUDA/CPU, explicit jobs, headroom,
  and selected-tree wipe. No Python/PowerShell runtime scripts or zip packaging.
- Only `llama-server` and `llama-bench` targets. CUDA always requests
  `GGML_CUDA_FA_QUANTS=all`; configure evidence and actual runtime feature evidence remain
  separate. Unused CUDA cache defaults do not become Metal/CPU compiled-pair provenance.
- Source jobs pin an exact shallow-fetched commit. Completed jobs prune inactive pins and
  unreachable objects; oversized fetched dependency caches (512 MiB per tree) are removed
  under ownership and marked for reconfiguration. Managed ccache defaults to 2 GiB with
  validated 256 MiB–16 GiB settings in selected-root `llama-build.json`.
- Canonical path creation checks ancestors before creating descendants. Staging refuses
  stale contents and symlink directories, requires the intended plugin and CUDA DLL families,
  and hashes shipped regular files and in-set library symlinks.
- Build steps and discovery queries run contained with bounded discovery/smoke deadlines.
  Unix uses the hidden pipe-watching supervisor; Windows assigns suspended children to a
  non-breakaway kill-on-close Job Object before resuming. Uncertain/corrupt process records
  block reuse and cleanup. Startup processes interrupted builds before runtime recovery/sweep.
- In-process leases survive a dropped build observer until its blocking worker exits. The
  standalone CLI additionally takes an exclusive OS root lease; app/agent processes take a
  shared lease, preventing standalone promotion against a running managed foundry instance.
- Shared journaled promotion owns staged/installed verification. Build failures never delete
  journal-owned staging. Cancellation is requested through a shared flag; transaction recovery
  is not abandoned after the journal boundary. Existing live/previous retention is reused.
- Bounded log tee/ring, structured failure hints, last-three completed log/JSON pair retention,
  and active/journal/process-record exemptions. No new HTTP routes or UI surfaces in this phase.

## Observed real Metal run

Qualification used a disposable `HOME` **and** explicit application root:
`/tmp/llf-p4.3mQN8k/app` (canonical `/private/tmp/llf-p4.3mQN8k/app`). The real user
runtime and `/Users/nick/AI/llama.cpp-src` were not build/install targets.

```sh
env HOME=/tmp/llf-p4.3mQN8k target/debug/local-llm-foundry llama-build \
  --config-dir /tmp/llf-p4.3mQN8k/app --accelerator metal --ref latest --jobs 8
```

- Resolved `b11739` to `0ce8cd11416c8840498e6701bcbe39ad4b858c2b`.
- Built all 309 Ninja steps, staged, verified, and installed through the shared promoter.
- Job `18dfff056ebb3e18-169` persisted `status: succeeded`, no failure, and
  `Installed b11739 · src · metal.` The archived journal has `phase: committed`,
  `outcome: installed`; first-install live/previous were both absent before the transaction.
- Installed manifest records exact SHA, actual configure argv, tool versions and shipped hashes.
  Stable root contained `bin/`, checkout/build/logs and `runtime-transaction.last.json`, with
  no pending transaction, staging directory or process record after the run.
- Toolchain: Apple clang 21.0.0, CMake 4.4.3, Ninja 1.13.2, Git 2.55.0, Apple M5 Max.
- The shallow checkout embeds commit `0ce8cd1` and reports build number `1`; the resolved tag
  label is provenance. Verification passed on exact commit and actual loaded backend/device,
  not a build-number-only match.

### Incremental rebuild and retention

The release CLI repeated the same source build with `--headroom`. Job
`7a036944d15a9a6b67e79413db19eff6` configured, built incrementally, verified, and promoted in
13.96 seconds. Exactly one `bin-previous/` remained. Both sets contained 22 manifest-listed
files; independent SHA-256 verification found no mismatches in either set. The new Metal
manifest correctly has no CUDA compiled-pair/architecture claim. No pending journal,
staging directory, process record, or `refs/foundry/jobs/` pin remained.

### Real Windows CUDA build (RTX 5090, qualification root `D:\llf-p4-cuda`)

Run from the normal checkout at `5e7cfc76` (disposable `HOME` and app root):
`llama-build --accelerator cuda --ref latest --jobs 8`, about 13.5 minutes.

- First attempt failed in preflight: the CUDA smoke test passed a `\\?\` verbatim path to
  nvcc, which cannot identify the compiler. Fixed in `5e7cfc76` (verbatim prefix stripped).
- Resolved `b11739` (`0ce8cd11416c8840498e6701bcbe39ad4b858c2b`), built 565 steps, staged
  and installed through the journaled promoter ("Installed b11739 . src . FA all").
- Manifest: `fa_quants_spec: all`, `cuda_architectures: 120a-real`, 49 compiled K/V pairs.
  Installed set includes cudart/cublas/cublasLt 13 DLLs, `ggml-cuda.dll` and all CPU variants.
- Model run from the installed set (`llama-bench`, `-ngl 99 -fa 1 -ctk q8_0 -ctv q4_0`,
  Qwen3.8-27B IQ3_XXS, 9.72 GiB): CUDA backend on RTX 5090 (cc 12.0); pp128 1649.21 t/s,
  tg32 91.82 t/s. This exercises a mixed quantized K/V pair. One smoke, not a throughput claim.

### Model smoke (separate from installation proof)

Sanitized environment (`env -i`, system PATH, disposable HOME/TMPDIR), installed `llama-bench`,
`Qwen3-0.6B-Q8_0.gguf`, `-ngl 99 -p 16 -n 8 -r 1 -fa on -ctk q8_0 -ctv q8_0`:

- Loaded BLAS, MTL and CPU plugins from the installed `app/bin/` set.
- Device `MTL0 (Apple M5 Max)`, embedded Metal library, bf16 support observed.
- pp16: 2290.92 t/s; tg8: 202.52 t/s; command exited 0.
- This is one small-model smoke, not a throughput promise or general mixed-K/V proof.

## Defects found during qualification

1. Unix recovery originally searched the supervisor group even though the child has a distinct
   group. It now fails closed when durable child-group identity cannot be established; it does
   not signal an unrelated PID or call uncertain descendants gone. Normal cancel/parent-death
   containment is exercised separately by real process-tree tests.
2. Windows free-space matching compared a verbatim canonical path with ordinary drive mount
   paths. Both sides are normalized. Cached source paths likewise normalize verbatim prefixes;
   Unix comparisons do not indiscriminately fold case.
3. Deferred smoke failure was incorrectly counted as a real prerequisite failure, suppressing
   smoke and vcvars setup. The deferred placeholder is excluded from those gates.
4. The first real Metal build selected existing **Intel** Homebrew OpenSSL libraries from
   `/usr/local`, causing arm64 unresolved symbols. The native Homebrew OpenSSL root is now pinned,
   SSL-only cache entries cleared, and the exact resolved Ninja/SSL argv recorded in provenance.
   No live binaries were installed by that failed compile.
5. Windows unit fixtures passed after the path fixes, but integration linking rejected a
   5,176,055,558-byte debug `libllama_monitor` archive (`LNK4003`, then unresolved symbols).
   Qualification passed with `CARGO_PROFILE_TEST_DEBUG=0`, not repository profile changes.
6. A no-op source-unpin test was removed; actual success/failure/panic/cancellation pin behavior
   is asserted by orchestration tests. A dropped-observer regression protects worker ownership.
7. vcvars capture now starts with OS discovery variables only, so parent API tokens and
   credential-bearing proxy variables cannot be persisted in its environment dump/cache.

## Verification status

- macOS initial build-specific run: 213 library tests, 3 CLI tests, 9 orchestration tests passed.
- Real Unix supervisor integration: 7 passed, including grandchildren, TERM-ignoring tools,
  cancellation, normal tool exit and app death.
- Strict `cargo clippy --all-targets -- -D warnings` passed on intermediate integrated snapshots.
- Windows symbol-light qualification (final snapshot): 205 library fixtures, 3 CLI tests, 10 orchestration tests,
  and all 4 Job Object test functions passed (one is the Rust child-helper entrypoint).
  Real tree cases cover cancellation, parent death, normal exit leaving a grandchild, and a
  subsequent clean job.
- The full Rust run exposed a case-folding assumption in a fixture; the assertion now checks
  trailing separators without treating case-sensitive Unix paths as identical.
- Final macOS gates passed in order: clippy -D warnings, full `cargo test` (2072 passed, 15 ignored), validate-js, eslint, `git diff --check`, release build, `cargo fmt --check`.

## Security scope and remaining qualification boundaries

Review traced CLI enum/path/options to subprocess and filesystem sinks: direct argv (no user
shell command), fixed upstream repositories/ref choices, bounded jobs, canonical managed paths,
OS-random job identifiers with failure propagation, secret-free vcvars capture, sanitized runtime
verification, and conservative journal/process-record cleanup. No new network endpoints,
SQLite operations, UI insertion sinks, token comparisons, or authentication changes are added.
Phase 5 must independently enforce endpoint authentication, confirmations and ownership.

- Unix crash recovery may require manual process inspection/record repair when group identity
  cannot be proven. Do not delete `processes.json` merely to bypass the barrier.
- Linux CPU/CUDA code paths are explicitly untested.
- Peak disk usage is not measured. The provisional Windows 20 GiB floor remains.
- Standalone CLI is model-less; it does not stop/restart an unrelated externally launched server.
  Stop those processes yourself before using their managed root.
- CUDA compatibility uses the OS-specific minor-compatibility table, not the toolkit-bundled
  development driver. Sources: [CUDA 12.9.1 release notes, Table 2](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html#cuda-driver)
  and [current CUDA driver notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-driver).
  Windows CUDA 12.x minimum is 528.33; Linux is 525.60.13. CUDA 13.x publishes branch 580;
  PTX/feature caveats remain warnings, not universal runtime guarantees.
