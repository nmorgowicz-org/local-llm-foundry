# llama.cpp From-Source Builds + Capability-Aware Mixed K/V

**Date:** 2026-10-19
**Status:** Implementation plan. Approved design retained; review contracts revised 2026-10-19. Prior spike observations in §9; FA-all peak disk usage remains unmeasured.
**Last Verified:** 2026-10-19 against foundry `main` @ `fcf2b878`, llama.cpp `master` @ `098ce526b` (b11727-8),
`../llama-local-tooling/scripts/llamacpp/` (build_llamacpp.py, Build-LlamaCpp.ps1, Install-LlamaCpp.ps1)

> **For agentic workers:** execute phases in order. Every phase lists files, tests, and a hard gate.
> Do not start a phase until the previous gate passes. Tests come before product code (TDD).
> Paths and line numbers were verified on the date above; re-check before editing.

**Goal (two parts):**

1. **Build from source.** Let a user compile llama.cpp for their own machine from the foundry UI:
   Metal on macOS, CUDA on Windows. The new binaries replace the old ones with full lifecycle
   management and no historic files piling up. Prerequisites are checked, validated, and (on macOS)
   installed from the app.
2. **Mixed K/V that is honest, not blocked.** Replace today's hard block on mixed main K/V (for
   example `-ctk q8_0 -ctv q4_0`) with a per-pair check: **native**, **fallback**, **unverified**,
   or **won't start**. Separate compiled pair availability from expected execution on the selected
   device/model. Only a known "won't start" rule blocks (existing type/syntax validation remains).
   Model-specific guidance never blocks and does not claim broad family evidence.

---

## 1. End state

- The welcome-screen llama.cpp pill offers **Download prebuilt** or **Build from source**. It shows
  where the current binary came from (`b11727 · src · FA all`), a building state with the
  current phase, and a red error state that links to the failure.
- A **llama.cpp Runtime** panel holds:
  - prerequisite checks with a **Validate** button
  - build options
  - a live build log
  - provenance of the current binary
  - build-cache size with a clean button
  - one-click rollback
- A failed compile never touches the live binaries. Candidate verification happens before swapping;
  install failures restore usable binaries through journaled recovery. Successful installs use the
  same promotion path as prebuilt updates; model restart failure is reported separately.
- In stable state there is one live set (`bin/`) and at most one previous (`bin-previous/`).
  Journal-owned temporary sets are retained until recovery finishes. Keep one checkout and
  one build tree per accelerator for incremental rebuilds; inactive caches can be cleaned.
- Every build writes `bin/llama-build-manifest.json`. The manifest records the source commit,
  CMake flags (including the exact FA K/V pair list), toolchain versions, and SHA-256 of every
  shipped file. It is provenance only. Compiled pair evidence comes from the actual runtime
  closure (executable plus dynamically loaded backends), not from the manifest (§4.6–4.7).
- The spawn wizard (Guided + Pro), the preset editor, preset bundles, calibration, and Doctor all
  use one backend function to classify a K/V pair. Warnings use the same text everywhere.

### Non-goals

- The `first-token-logits` LoRA-eval tool from `llama-local-tooling` is not built, injected, or
  managed. We never patch the llama.cpp checkout.
- No dependency on `config/llama-tooling.json`, `ToolingConfig.psm1`, Python, or PowerShell
  scripts. The logic is ported to Rust (decision D1).
- Linux gets the code path (CPU/CUDA) but is labelled **untested** in the UI. There is no Linux
  CI qualification.
- No one-click install of Visual Studio or CUDA on Windows (decision D2).
- No arbitrary git refs. Only the latest release tag or `master` HEAD (decision D5).

---

## 2. Research findings (verified 2026-10-19)

### 2.1 Metal: mixed K/V is not supported by the Flash Attention kernels

`ggml/src/ggml-metal/ggml-metal-device.m`, `supports_op` → `case GGML_OP_FLASH_ATTN_EXT`:

```c
if (op->src[1]->type != op->src[2]->type) {
    return false;
}
```

This has been in place since `0320ac526` (2025-09-17, "metal : refactor + optimize v2") and is
unchanged at b11727. Equal-type FA pairs are supported for `f32, f16, bf16 (if hw), q8_0, q4_0,
q4_1, q5_0, q5_1`. There is no build flag that changes this. It is a kernel limitation.

What happens at runtime with K ≠ V on Metal. These results were **measured in Phase 0 (S1)** on an
M5 Max with Qwen3.8-27B IQ4_XS (head_dim 256), b11719, `llama-bench -p 512 -n 64 -d 4096 -r 2`:

| K / V | `-fa` | pp512 @4k t/s | tg64 @4k t/s | What happened |
| --- | --- | --- | --- | --- |
| f16 / f16 | on | 640 | 27.7 | Native FA |
| q8_0 / q8_0 | on | 567 | 25.5 | Native FA |
| q4_0 / q4_0 | on | 357 | 14.6 | Native FA |
| q8_0 / f16 | auto | 635 | 22.8 | FA disabled by the auto probe; GPU attention without FA |
| q8_0 / f16 | on | **80** | **15.8** | FA forced on → attention on **CPU** |
| q8_0 / q4_0 | on | **27** | **8.4** | Attention on **CPU** |
| q8_0 / q4_0 | auto (server) | — | — | **Starts.** Log: `enabling flash_attn since it is required for quantized V cache` → `flash_attn = enabled` → same CPU-attention path as `-fa on` |

My first reading of the source was wrong: `-fa auto` with a quantized V does **not** refuse to
start. `llama_init_from_model` forces FA on for a quantized V before the auto probe runs, so the
server starts and attention silently runs on CPU. Only `-fa off` with a quantized V refuses to
start (`quantized V cache requires flash_attn to be enabled`).

**Conclusion for this b11719 / M5 Max / Qwen3.8-27B experiment only:** mixed quantized K/V
prefill was 21× slower and decode 3× slower than the equal pair. This is not a universal performance
or startup guarantee for other devices, models, head dimensions, or versions. The comment in
`llama-local-tooling/scripts/llamacpp/build_llamacpp.py:839-841` ("the Metal flash-attn path
handles quantized KV at runtime") is correct for equal pairs and wrong for mixed pairs.

**Confirmed bug (S1):** the preset editor defaults to `-ctk q8_0`, `-ctv f16`
(`static/js/features/presets.js:124-125`, `static/index.html` `#modal-ctv` "f16 (default)")
and `-fa on` (`#modal-flash-attn`). On Metal that is the 80 / 15.8 t/s row: **7× slower prefill**
than q8_0/q8_0 in this experiment. Other Mac presets need contextual classification, not a universal
slowdown claim. The editor's "f16 can be faster than q8_0 on Apple Silicon with Qwen3" tooltip
overgeneralizes: the observed equal-pair f16/f16 advantage (+13% pp, +9% tg) is model/device-specific.

**S1b (2026-10-19, same b11719 / M5 Max / Qwen3.8-27B IQ4_XS, `llama-bench -fa on -d 8192`).**
Short prompts hide CPU attention (at 256 tokens q8_0/bf16 still benched 553 pp), so these are depth runs:

| K/V | pp512 | tg32 |
| --- | ---: | ---: |
| q8_0/q8_0 | 570 | 28.5 |
| bf16/bf16 | 549 | 26.5 |
| f16/f16 | 514 | 28.6 |
| q8_0/bf16 | 22 | 6.2 |
| bf16/q8_0 | 9 | 4.7 |
| bf16/f16 | 14 | 5.5 |

Every mixed pair collapses, including float/float (bf16/f16). Source explains it: the Metal
`FLASH_ATTN_EXT` support check returns false when `src[1]->type != src[2]->type`
(`ggml-metal-device.m`, rule present since the Metal refactor #15995). `-fa auto` with q8_0/bf16 logs
"Flash Attention is assigned to device CPU ... set to disabled"; `-fa auto` with bf16/bf16 logs
"Flash Attention enabled". bf16 hardware support is reported by the device at startup
(`ggml_metal_device_init: has bfloat = true`), which is the same flag the support check consults.
The classifier applies the Metal rule to builds at or after b11719 (`METAL_FA_RULE_FROM_BUILD`);
builds above the measured one say "not re-tested". Failing toward the warning is deliberate: a stale
warning is harmless, a missing one costs 20-50x prefill. If upstream ships mixed Metal kernels, set
`METAL_FA_MIXED_KERNELS_FROM_BUILD` and those builds return to `Unverified`. Builds below b11719 stay
`Unverified`. Re-check with `grep -n 'src\[1\]->type != op->src\[2\]->type'
ggml/src/ggml-metal/ggml-metal-device.m` when a release note mentions Metal or flash attention.

### 2.2 CUDA: missing compiled pairs can use a conversion fallback

Upstream PR #28079 (`5a4d0feca`, 2026-09-09) replaced `GGML_CUDA_FA_ALL_QUANTS` with:

```cmake
set(GGML_CUDA_FA_QUANTS "q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16" CACHE STRING
    "...\"all\" or a list such as \"q8_0-q8_0;q8_0-q4_0\"")
```

- `all` compiles every K×V combination of `q4_0 q4_1 q5_0 q5_1 q8_0 bf16 f16` (49 pairs).
- A list compiles exactly those pairs plus `f16-f16` (`ggml/cmake/common.cmake:53-110`).
  **Setting a list replaces the default list.** To keep the defaults you must list them again.
- `GGML_CUDA_FA_ALL_QUANTS=ON` still works but is deprecated (it maps to `all`).
- If a pair is **not** compiled, the CUDA backend does not fail. It converts K and V to f16 and
  logs once (`ggml/src/ggml-cuda/fattn.cu`):
  `no FlashAttention vector kernel compiled for K/V types X-Y, converting K and V to f16 instead
  (slow). Add "X-Y" to GGML_CUDA_FA_QUANTS to compile it.`
- The binary reports its compiled list as a backend feature (`FA_QUANTS`, `ggml-cuda.cu:6093`),
  printed by `llama_print_system_info()` at **trace** verbosity. **Verified (S3):**
  `llama-server -lv 4 -m <path that does not exist>` prints the line and exits within about a
  second, with no model loaded. On ryne (RTX 5090, b11696):
  `CUDA : ARCHS = 1200 | USE_GRAPHS = 1 | FA_QUANTS = all | BLACKWELL_NATIVE_FP4 = 1 | …`.
  On Metal: `MTL : EMBED_LIBRARY = 1 | …`. This is the **primary compiled-pair evidence**
  (§4.7), including for externally supplied binaries; it does not prove model execution.
- `iq4_nl` is not in this version's `FA_TYPES` and has no CUDA FA vec kernel. Where accepted
  by the runtime's cache type parser, CUDA uses conversion fallback under the documented FA path;
  on Metal it is not an FA type. Unknown/unsupported cache names remain separate syntax validation.

**Conclusion:** the foundry's hard block (`MixedMainKv::product_default_denied`) is stricter
than the observed upstream behavior. A CUDA runtime reporting a missing `q8_0/q4_0` or
`q8_0/f16` kernel can use the conversion fallback; a compiled pair is evidence of availability,
not universal execution. Do not infer any particular prebuilt's list from upstream CMake defaults:
probe its loaded backend and qualify device/model constraints.

### 2.3 Foundry today: KV choice audit

| Surface | Offers | K ≠ V? | Mixed handling | Default |
| --- | --- | --- | --- | --- |
| Wizard Guided tiles (`index.html` `#hw-kv-tiles`, `spawn-wizard.js:2215-2229`) | q8_0 / q4_0 / f16 | **No.** Tiles set K = V | None visible. A mixed pair from autosize leaves no tile active | q8_0/q8_0 |
| Wizard Pro selects (`spawn-wizard.js:2680-2739`) | Hardcoded 9 types + snapshot extras | Yes | Blocked only at launch | q8_0/q8_0 |
| Preset editor (`presets.js:123-139`) | Common + snapshot `cache` values | Yes | Blocked only at launch | **q8_0/f16** |
| Bundles (`presets/bundle.rs:852-856`, `resolver.rs:772,895`) | f16/f16, q8/q8, q4/q4, q8_0/q4_0 | One mixed direction | Blocked unless `mixed_main_kv.supported` | n/a |
| Launch (`inference/launch.rs:439-460`) | — | — | `validate_main_kv_policy` + an extra q8→q4-only bail when snapshot is `None` | — |
| Validator (`presets/validation.rs:68-96`) | — | — | Only **q8_0↔q4_0** counts as "mixed". Every other K≠V pair passes unchecked | — |
| Calibration (`calibration/executor.rs:2268`) | — | — | Same validator | **q8_0/f16** |
| VRAM estimator (`llama/vram_estimator/estimate.rs:20-65`) | — | — | Sizes K and V independently ✔ | q8_0/q8_0 |
| Doctor (`web/api/doctor.rs:88-99`) | — | — | Reports the validator issue | — |

Problems to fix:

1. "Mixed" is defined as one pair, not as K ≠ V or "pair not natively compiled".
2. The blocking decision ignores the platform, which is the thing that matters.
3. Default V differs by entry point (`q8_0` in the wizard and estimator, `f16` in the editor,
   calibration and setup chip).
4. `-fa` and KV choice are never checked together, although on Metal the pair + `-fa` decides
   whether the server starts.
5. Guided mode cannot express the pair the user actually runs (Qwen3.8-27B at q8_0/q4_0).
6. The only model-specific KV advice is the Qwen3/Metal tooltip, which is probably wrong (§2.1).

### 2.4 What we port from `llama-local-tooling`

From `build_llamacpp.py`, port these to Rust:

- NVIDIA detection via `nvidia-smi --query-gpu=name,compute_cap,memory.total`
- the compute-cap → `CMAKE_CUDA_ARCHITECTURES` map (lines 71-82, 75 → 121a-real)
- CUDA root discovery (`CUDA_PATH`, then `Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*`)
- stale CMake cache repair (`find_stale_cache_paths` + `cmake -U`, lines 433-530)
- the stale-artifact sweep via `ninja -t targets all` (lines 387-430)
- the CUDA runtime DLL staging (`cudart64_*`, `cublas64_*`, `cublasLt64_*`)
- the per-accelerator CMake flags (lines 791-870)

From `Build-LlamaCpp.ps1:222-325`, port the vcvarsall environment capture.

From `Install-LlamaCpp.ps1`, port the running-process detection idea only. The foundry already
owns the server process, so the port is simpler.

Drop:

- the `first-token-logits` injection (lines 296-380, 556-570)
- `config/llama-tooling.json`
- `uv`/`.venv` setup
- zip packaging
- `EXPECTED_BINARIES` beyond what the foundry runs

---

## 3. Decisions

| # | Decision | Choice |
| --- | --- | --- |
| D1 | Build engine | Port to Rust; call git/cmake/ninja directly. No Python. |
| D2 | Prerequisite install | Detect + validate (with real smoke tests) + guided fix everywhere; one-click `brew install` on macOS only. |
| D3 | Source/build retention | Keep one checkout + one build tree per accelerator for incremental rebuilds; show size; "Clean build cache" button. |
| D4 | Binary rollback | Exactly one previous set (`bin-previous/`). |
| D5 | Buildable refs | Latest release tag (`bNNNN`) and `master` HEAD. |
| D6 | UI entry | Pill + popover + dedicated llama.cpp Runtime panel. |
| D7 | Mixed K/V policy | `native` / `fallback` / nonblocking `unverified` / `wont_start`; only known `wont_start` blocks, alongside existing type/syntax validation. |
| D8 | Metal mixed K/V | S1 measured CPU attention for mixed pairs with quantized V or forced FA; retain strong context-qualified warnings, never universal execution promises. Known quantized V + FA off is `wont_start`. No mixed Guided tile on Metal; rebuilding does not add mixed kernels. |
| D9 | CUDA FA kernels | **Always `-DGGML_CUDA_FA_QUANTS=all`** (every K×V combination of `q4_0 q4_1 q5_0 q5_1 q8_0 bf16 f16`, 49 pairs). Not a user option: no preset picker in the UI or API. Accept the longer build time. |
| D10 | Model advice | Data table keyed by GGUF architecture/family. Shown as info/caution text next to KV controls. Never blocks. Entries cite where they came from. |
| D11 | Default V type | Unify on **q8_0/q8_0** everywhere (equal pair with available FA kernels on evidenced CUDA/Metal configurations). Existing presets keep stored values; availability is not universal model execution proof. |
| D12 | Build targets | `llama-server` and `llama-bench` only, with `LLAMA_BUILD_TESTS=OFF`, `LLAMA_BUILD_EXAMPLES=OFF`. |

---

## 4. Architecture

### 4.1 Module layout

```
src/llama_build/
  mod.rs            pub API used by web/api/llama_build.rs; BuildRequest/BuildPlan types
  paths.rs          on-disk layout helpers (wraps AppPaths)
  prereqs.rs        check registry; Check { id, title, status, detail, fix, evidence }
  prereqs_macos.rs  CLT, brew, cmake, ninja, ccache, git, disk, Metal device
  prereqs_windows.rs vswhere/MSVC, Windows SDK, CUDA/nvcc, driver, git, disk, smoke test
  toolchain_env.rs  Windows: vcvars64 env capture + cache; Unix: passthrough + PATH sanity
  gpu.rs            nvidia-smi parse, compute-cap → arch map, Apple chip detect
  source.rs         managed git checkout: clone/fetch/checkout/reset/clean, ref resolution
  configure.rs      pure fn: (BuildPlan) -> Vec<String> cmake args; cache-heal; FA list
  job.rs            single-flight build job, phases, cancel, log tee, structured failure
  stage.rs          collect artifacts from build tree → staging dir; CUDA runtime DLLs
  manifest.rs       BuildManifest read/write/verify (SHA-256 per file)
src/llama_runtime/
  promote.rs        shared verifier, journaled promotion/rollback and startup recovery
  probe.rs          bounded runtime-closure probing/identity (Phase 1); reused by shared verifier
src/inference/
  kv_support.rs     classify_kv_pair(): the only K/V policy function
  kv_advisories.rs  family advice table
src/web/api/
  llama_build.rs    HTTP routes (all behind API token)
static/js/features/
  llama-runtime-panel.js   new panel
  llama-updater.js         pill/popover extended
  kv-support.js            shared client rendering of classify_kv_pair results
static/css/llama-runtime-panel.css
```

Each unit has one job. `configure.rs`, `gpu.rs` parsing, `manifest.rs`, and `kv_support.rs` are
pure and unit-tested without a toolchain. `job.rs` takes a `Toolchain` trait so tests can inject a
fake that fails at any phase.

### 4.2 On-disk layout (under `AppPaths.root`)

```
bin/                          live set (unchanged location; llama_server_path default)
  llama-server[.exe], llama-bench[.exe], *.dylib|*.dll, llama-build-manifest.json
bin-previous/                 exactly one rollback set (or absent)
.bin.staging-<job>/           candidate; cleanup only after transaction recovery
.bin.txn-<id>-*/              journal-owned temporary old/failed sets during swap/recovery
runtime-transaction.json     durable transaction journal (atomic replace + flush)
llama-build/
  src/                        git checkout (github.com/ggml-org/llama.cpp)
  build-metal/ | build-cuda/ | build-cpu/   CMake build trees
  ccache/                     managed CCACHE_DIR; finite configurable cap (§4.4)
  toolchain-cache.json        captured vcvars env + toolchain fingerprint (Windows)
  prereqs-last.json           last validation result + fingerprint
logs/llama-build/<job-id>.log + <job-id>.json   last 3 completed pairs kept
```

Add `AppPaths::llama_build_dir()`, `llama_source_dir()`, `llama_build_tree_dir(accel)`,
`llama_build_logs_dir()`, `bin_previous_dir()` in `src/paths.rs`, with tests matching the
existing path tests.

**Custom server path:** if `config.llama_server_path` points outside the managed `bin/`
(see `managed_llama_server_path`, `llama_binary.rs:35-66`), builds still install to managed
`bin/`. The panel shows "Your server path is custom: `<path>`. Builds install to the managed
runtime." and a **Use managed runtime** button. We never write into a user-chosen directory.

### 4.3 Shared promotion (refactor of `api_llama_binary_update`)

Today `api_llama_binary_update` (`llama_binary.rs:817-1366`) mixes download, staging, stop,
swap, health check, restore, and backup sweep inside one warp closure, using a `fail_update!`
macro. Extract to `src/llama_runtime/promote.rs`:

```rust
pub struct PromoteRequest {
    pub staged_dir: PathBuf,          // complete candidate set, already outside bin/
    pub label: String,                // "b11727 prebuilt" | "b11727 src (FA all)"
    pub manifest: Option<BuildManifest>,
    pub verification: VerificationIntent, // server-owned expectations, never manifest authority
}
pub enum PromoteOutcome {
    Promoted { restarted: bool },
    InstalledRestartFailed { restart_error: String }, // binaries passed; model did not restart
    RolledBack { reason: PromoteError },
}
pub async fn promote(state: &AppState, cfg: &AppConfig, req: PromoteRequest)
    -> Result<PromoteOutcome, PromoteError>;
pub async fn rollback(state: &AppState, cfg: &AppConfig) -> Result<PromoteOutcome, PromoteError>;
```

`VerificationIntent` is a small internal record: pinned source commit (source builds), expected
backend, selected detected device id and install kind (source/prebuilt/rollback). The caller
constructs it from the resolved job/release operation, not client free text or manifest claims.
Source CUDA always requires observed FA-all; no API option relaxes D9. A prebuilt/rollback
uses its actual feature evidence and reports uncertainty without inferring an upstream default.

**One verification owner:** `promote` invokes a shared staged-runtime verifier for both source
and prebuilt installs; `job.rs` does not independently launch `generate_snapshot` or repeat GPU
probing. Pure hash/configure checks may happen before this handoff. Preparation (exec bits,
quarantine removal, file inventory) precedes **any** staged execution. The verifier checks
compile identity, loader closure and intended backend separately (§4.6); `--help` alone is not
sufficient. A no-model probe can initialize a GPU: if a running model could contend, durably
record restart intent in the journal **before** stopping/reaping it under the runtime/model lock.
Tell the user the validation/swap window interrupts the model. Candidate rejection leaves old
binaries intact, recovers the transaction and attempts old-model restart; report restart errors
separately. The shared verifier may be reused for ordinary bounded capability reads, without a swap.

**Recoverable transaction, not a sequence of best-effort renames:**

1. Acquire the runtime/model-lifecycle mutex; validate same-volume managed paths and recover any
   prior journal first. Prepare candidate permissions/quarantine and inventory without execution.
   Reserve space for staging and journal. Cancellation is still allowed before journal creation.
2. Before stopping a model, running staged checks, or renaming, atomically write and durably
   flush `runtime-transaction.json` under
   app root. Record transaction id, operation (install/rollback/uninstall), path roles, set
   inventories/hashes, last usable live set, old previous set, restart intent/config reference,
   phase, and intended next action. Persist intent **before**, completion **after** each mutation.
   Use file/directory flushes and platform durability primitives; a durability failure stops
   forward progress and enters recovery. Never store credentials in the journal.
   Journal creation begins the noncancellable transaction. Stop/reap a contending model and run
   the staged verifier now, updating restart/verification states durably. On rejection replay
   recovery even if no rename occurred; on crash after stop the saved intent reconciles restart.
3. Preserve existing `bin-previous` at a journal-owned temporary path, then move live `bin` to
   its journal-owned old-live path, then candidate to `bin`. Never delete the last usable set
   to make room. Every rename failure (including sharing violations) invokes recovery; no
   unchecked early return after mutation. Missing initial live/previous sets are valid cases.
4. Verify at the final path with the shared verifier (relocation/loader check, not duplicate
   unrelated probing). On binary/loader/backend failure restore the old live **and old previous**
   transactionally, recording rollback intents and completions; keep recovery material if any
   compensating rename fails. Repeated recovery reconciles hashes/path roles, not just phase flags.
5. After live verification, durably commit the installed set and arrange old live as
   `bin-previous`. Only now may superseded previous/failed sets be deleted. Restart the remembered
   model once, with recorded restart state. Success returns `Promoted`; failure returns
   `InstalledRestartFailed` with binaries still installed and rollback available. A model
   config/OOM/execution failure is **not** automatic proof of bad binaries and does not automatically
   trigger binary rollback. Offer retry/manual rollback and preserve diagnostics.
6. Persist the terminal outcome before cleanup and journal removal. Recovery of a committed
   install completes retention; recovery of an uncommitted swap restores the last usable set.
   Restart reconciliation checks owned process/session identity to avoid duplicate model starts.

`rollback` uses the same journal/verifier to swap live and previous through temporary paths,
preserving both until commit; rollback twice returns to the original. Uninstall shares journaled
mutation/retention rules and explicit user intent; it cannot race a swap.

Promotion is **noncancellable once the journal transaction begins**, including compensating
recovery. Disable Cancel and return a clear noncancellable status then. Before that boundary,
cancel reaps owned build processes; no promotion probe or model stop has begun. UI disconnect
never cancels recovery.
Crash boundaries at every intent, rename, verification, commit, restart and cleanup are recoverable.

**Startup order (before serving mutation/model-launch APIs):** acquire ownership → establish that
prior job descendants cannot write → replay/reconcile the runtime journal idempotently → reconcile
model restart and mark interrupted build jobs → invalidate/reprobe changed runtime identity →
only then remove unreferenced orphan staging/legacy backups and prune logs/caches. Never sweep
journal-referenced directories; ambiguous/corrupt journals retain sets and surface a repair error
rather than guessing or deleting. Stable completion leaves only live + at most one previous set.

The in-place `copy_all_files(tmp → dest_dir)` fallback branch (`llama_binary.rs:~1257`) applies
when the binary sits in an unmanaged dir. It is removed for promotions, per the custom-path rule
in §4.2.

### 4.3.1 Ownership and lock order

| Operation | Exclusive/conflicting ownership |
| --- | --- |
| Build (fetch through artifact collection) | One build owns the shared checkout **and all build trees**; no parallel accelerator build against this checkout. Validation uses separate temp trees but shares toolchain access. |
| Clean/wipe/cache deletion | Excludes builds using checkout/build/ccache and validation using selected cache/toolchain; return `409` with owner, never delete active files. Optional source cleanup excludes fetch/build too. |
| Toolchain install/fix | Excludes prerequisite checks, validation and builds using the changing toolchain; invalidate captured env/results on completion. No toolchain lock during unrelated runtime download. |
| Prebuilt update / source promote / rollback / uninstall | One runtime mutation mutex; same mutex gates model start/stop/restart at the validation/swap window. Download and compile do not hold it. |
| Capability/model probe | Short runtime identity read/probe lease, exclusive with swap; GPU probes coordinate with model lifecycle, not concurrent unowned processes. |

Fixed order whenever multiple guards are needed: toolchain lease → checkout/build/cache ownership
→ runtime/model-lifecycle mutex → existing AppState session locks (`sessions` then
`active_session_id`). Use short state-lock scopes; no awaits/process waits under session guards.
Never acquire an earlier lock while holding a later one. Downloads need only their own job/temp
ownership; compile needs build/toolchain ownership, not runtime ownership. Prebuilt promotion
needs no build-tree lock. Hold runtime ownership only for coordinated verification/swap/restart.

### 4.4 Build job

```
preflight → fetch → configure → compile → stage → promote (prepare/verify/transaction/restart) → done
```

| Phase | Work | Failure keeps live binaries? |
| --- | --- | --- |
| preflight | Re-run fast prereqs (reuse smoke only with unchanged fingerprint); estimate additional disk allocations; choose latest tag or master and resolve once to exact commit | yes |
| fetch | Fetch that exact object, pin with a job ref until completion; detached checkout/reset to commit; `git clean -fdx` (trees outside `src/`). Ref movement never changes this job's commit | yes |
| configure | Stale-cache heal (`cmake -U`); `cmake -S src -B build-<accel> …` | yes |
| compile | `cmake --build … --target llama-server llama-bench --parallel N` (add `--config Release` for Windows Ninja Multi-Config) | yes |
| stage | Recheck free space; collect this configure's outputs + shared libraries + selected CUDA DLLs into staging; write provenance hashes/configure evidence | yes |
| promote | Shared verifier owns staged execution, exact commit/closure/backend checks, journaled swap and restart (§4.3, §4.6) | restores on binary/swap failure; model restart failure has separate installed outcome |

- **Single flight:** one build at a time; a second `start` returns `409` with the running job id.
- **Cancel and crash containment:** all subprocesses (git, compilers, smoke tests, probes and
  installers) launch under owned supervision. Windows: assign suspended children to a
  kill-on-close Job Object before resume, forbid breakaway, terminate job and wait for all
  descendants on cancel. `taskkill /T` alone is not crash safety. Unix: an owned supervisor
  watches app liveness through a pipe, owns process groups, sends TERM then KILL after 10 s,
  and reaps descendants; durable process records include creation identity and job token.
  Recovery validates identity/ownership before signalling, never blindly kills a saved PID/PGID.
  Do not release tree/toolchain ownership until no descendant can write. If containment cannot
  be established after a crash, block reuse/cleanup and report repair needed. Cancel before the
  journal boundary removes only unreferenced staging; after that boundary it is rejected (§4.3).
  Interrupted configure/compile marks `needs_reconfigure`; a dirty tree is revalidated before reuse.
- **Job concurrency:** `--parallel` defaults to `max(1, logical_cpus - 1)`. The panel offers
  "Leave headroom for a running model" (`logical_cpus / 2`) and lowers priority (`nice 10` on unix,
  `BELOW_NORMAL_PRIORITY_CLASS` on Windows).
- **Logging:** every child process's stdout/stderr is teed line-by-line into the job log file
  and an in-memory ring buffer (last 2,000 lines) served to the UI via polling. Each phase writes
  a header line with the exact command, working dir, and a redacted env diff.
- **Structured failure:**

```rust
pub struct BuildFailure {
    pub phase: Phase,
    pub summary: String,          // one human sentence
    pub command: Option<String>,  // exact argv, shell-quoted
    pub exit_code: Option<i32>,
    pub log_tail: Vec<String>,    // last 80 lines
    pub first_error: Option<String>, // first line matching error patterns (see §8)
    pub hint: Option<Fix>,        // known-failure match, e.g. stale cache → "Wipe build dir"
    pub log_path: PathBuf,
}
```

- **Restart safety/retention:** persist job state atomically in `<id>.json`. Startup follows §4.3,
  not unconditional staging deletion. Keep the last 3 completed `.log` + `.json` pairs; active
  and journal-referenced records are exempt until resolved. Bound per-job disk logs (configurable
  64 MiB default, rotate/truncate with an explicit marker); the UI ring remains 2,000 lines.
- **Disk budget:** calculate **additional** checkout/download allocations, build-tree growth,
  bounded ccache growth, candidate staging and verification temporaries, plus conservative
  safety margin. Already occupied live/previous/build dirs are reflected in free space and must
  not be charged again as new allocations. Show assumptions, not false peak precision. Enforce
  provisional floors of 20 GB Windows / 2 GB macOS or the larger estimated additional need;
  Windows floor stays until actual FA-all peak measurement. Recheck before stage and before
  the journal; never delete usable runtime sets to recover space.
- **Managed caches:** `CCACHE_DIR=<app-root>/llama-build/ccache`, conservative finite default
  cap 2 GiB. Configure via `AppConfig.llama_build.ccache_max_bytes`, validated finite range
  256 MiB–16 GiB; apply before child launch and surface in options/panel. Report size separately.
  Clean removes inactive build trees and managed ccache (optional source checkout),
  never live/previous or active validation/job files. Retain one checkout, shallow bounded
  history where possible, prune stale remote/job refs and fetched dependency caches when
  inactive; preserve pinned build objects until jobs finish. No new general cache manager.

### 4.5 CMake configuration (pure function in `configure.rs`)

Common to all accelerators:

```
-DCMAKE_BUILD_TYPE=Release   -DLLAMA_BUILD_TESTS=OFF   -DLLAMA_BUILD_EXAMPLES=OFF
-DLLAMA_BUILD_TOOLS=ON       -DLLAMA_BUILD_SERVER=ON   -DBUILD_SHARED_LIBS=ON
-DGGML_BACKEND_DL=ON         -DGGML_NATIVE=OFF  (CUDA) / ON (Metal, CPU — local-only binaries)
-DGGML_CCACHE=ON when ccache is present
```

**Metal (macOS):** `-G Ninja -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
-DGGML_METAL_USE_BF16=<ON on Apple Silicon> -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple`.
`EMBED_LIBRARY` embeds the `.metal` source as text, and kernels compile at runtime. **Verified
(S4):** with `DEVELOPER_DIR=/Library/Developer/CommandLineTools` (no `metal` compiler available),
configure + build of `llama-server`/`llama-bench` succeeded in 22 s on an M5 Max (no ccache),
producing a 114 MB build tree. The binary ran Qwen3.8-27B at 707 pp / 33 tg t/s. Full Xcode is
**not** a prerequisite.

Two traps found during S4, both handled in `configure.rs`:

- **Canonicalize source and build directories** passed to CMake (`std::fs::canonicalize`).
  Create managed directories first, then canonicalize and check containment; never canonicalize
  uncreated output-file names. A build dir reached through a symlink (`/tmp` → `/private/tmp`)
  broke `scripts/ui-assets.cmake:218`: it builds
  relative paths and then can't `lstat` the gzip input (`Unable to read from file …bundle.*.css`).
  This is probably also the failure recorded in a previous session's build attempt.
- **The build needs network access.** `llama-server`'s web UI assets are downloaded from the
  Hugging Face bucket `ggml-org/llama-ui` during the build (verified by sha256). This is not a
  configure-time step. Preflight checks HTTPS reachability of `huggingface.co` as well as GitHub.
  Error catalog §8 gets a matching entry.

**CUDA (Windows):** `-G "Ninja Multi-Config"` (built with `--config Release`), plus the following.
The list is cross-checked against upstream's `windows-cuda` release job (S3):

- `-DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl`, set explicitly. On ryne, Strawberry Perl's
  `C:\Strawberry\c\bin` (gcc, ninja, ccache) is on the system PATH, and `cl` is only on PATH
  after vcvars. Pinning prevents CMake from picking up MinGW gcc.
- `-DGGML_CUDA_CCCL_VERSION=v3.4.3` **when nvcc < 13.5**, matching upstream release CI ("CTK ≥ 13.5
  bundles CCCL ≥ 3.5; omit for those versions"). This makes CMake `FetchContent` clone
  `github.com/nvidia/cccl` at configure time, which is another network dependency. Keep the
  version in one constant that is updated alongside upstream.

- `-DCMAKE_CUDA_COMPILER=<cuda>\bin\nvcc.exe`
- `-DCMAKE_CUDA_HOST_COMPILER=cl.exe` (keep the tooling's quoting comment)
- `-DGGML_CUDA=ON -DGGML_CUDA_NCCL=OFF -DGGML_CUDA_COMPRESSION_MODE=speed`
- `-DGGML_CUDA_FA_QUANTS=all` (D9; fixed, not configurable)
- `-DCMAKE_CUDA_ARCHITECTURES=<auto from compute cap | user override>`

Never emit `GGML_CUDA_FA_ALL_QUANTS`.

**CUDA (Linux, untested):** the same flags without the MSVC bits; generator `Ninja`.

FA set parsing has unit tests reproducing CMake semantics. Only emit `all`; parse actual configure
status for provenance and validate it against the runtime probe. No constant supplies a guessed
prebuilt capability list:

```rust
const FA_BUILD_SPEC: &str = "all";
fn compiled_fa_pairs(spec: &str) -> Result<BTreeSet<(KvType, KvType)>, ParseError>;
// Recognized version: "all" = 7×7; valid explicit list ∪ {f16-f16}; malformed/unknown = error.
```

No compile-time estimate is shown before a build. Build time depends on CPU, ccache state and
what changed, so a pre-measured number would be misleading. The panel shows the user's **own
last build duration** once one exists, and live `[n/m]` progress during a build.

### 4.6 Build manifest

`bin/llama-build-manifest.json` (schema v1):

```json
{
  "schema": 1,
  "source": "foundry_source_build",            // or "upstream_prebuilt"
  "label": "b11727 · src · FA all",
  "llama_cpp": { "ref": "b11727", "commit": "098ce526b…", "build_number": 11727 },
  "accelerator": "cuda",
  "cmake_args": ["-DGGML_CUDA=ON", "..."],
  "fa_quants_spec": "all",
  "fa_pairs_compiled": [["q8_0","q4_0"], ["q8_0","q8_0"], "..."],
  "toolchain": { "cmake": "4.1.2", "ninja": "1.13.1", "msvc": "14.44.35207", "nvcc": "13.4.48", "sdk": "10.0.26100.0" },
  "cuda_architectures": "120a-real",
  "built_at": "2026-10-19T18:02:11Z",
  "files": { "llama-server.exe": "sha256:…", "ggml-cuda.dll": "sha256:…", "...": "..." }
}
```

Prebuilt updates write provenance too: release/asset identity and whatever valid feature evidence
the probe actually reports; absent evidence is recorded as unavailable, not an upstream default.
Manifests never authorize capabilities. Hash mismatch marks provenance `custom` and forces identity
refresh. A matching manifest still does not prove loaded backend or model execution.

### 4.6.1 Installed closure and bounded capability probe

The shared verifier (§4.3) must distinguish three levels:

1. **Compile identity:** resolve chosen tag/HEAD once, fetch and pin the exact commit through
   build. Verify checkout HEAD and embedded runtime commit against it, not just `build_number`;
   master may use `git describe` with commits past a tag. Normalize abbreviated hashes only
   against the pinned full object. Check both built executables, reject mismatched/stale outputs.
2. **Loader/backend closure:** execute from staging with controlled cwd and sanitized PATH,
   explicit staged backend/library search (Windows DLL search policy; macOS rpath/loader rules).
   Exclude build-tree/toolkit PATH and inherited library overrides; allow required OS/driver
   libraries by explicit policy. Inventory resolved paths/hashes for executable, ggml/llama
   libraries, dynamic backend plugins and other relevant loaded artifacts. Prove the intended
   CUDA/Metal backend is loaded and the selected device enumerated, not merely `--help` success
   or CPU fallback. Record OS driver dependencies separately. Repeat relocation-sensitive checks
   at final live path. Source CUDA promotion additionally requires configure **and runtime**
   `FA_QUANTS=all` evidence consistent with 49 pairs; absence/malformed evidence fails source
   build verification, not ordinary launches of existing runtimes.
3. **Model execution:** loader/backend proof is not a model smoke test. When a suitable local
   model is available, manual QA loads it on the selected device and observes attention behavior.
   Otherwise mark this proof deferred to manual QA, never fabricate a pass or require a download.

Probe `llama-server -lv 4 -m <unique-nonexistent-local-path>` in an owned temporary cwd. Reserve
a unique local parent/path, verify the model file is absent, disable remote/model lookup options,
and pass a minimal allowlisted environment and explicit backend/device context. Timeout 10 s,
combined output cap 1 MiB; terminate/reap on either limit. Parse only recognized complete feature
records for the actual loaded backend. An intentional model-missing nonzero exit is acceptable
**only** with a recognized model-missing diagnostic and valid recognized evidence; arbitrary
nonzero exit, malformed fields, timeout or truncation produce unavailable evidence with reason.
Use the same limits for staged and ordinary capability probes.

**Runtime identity/cache basis:** canonical executable path + hash **and relevant dynamically
loaded backend/library paths + hashes**, selected backend/device identity, driver/device features
as relevant, and parser/basis version. Discover actual loaded paths, not a filename guess.
Hash/load identity changes invalidate backend snapshots **and UI caches**, including plugin-only
replacement with unchanged executable. Recheck identity at launch/swap and reject stale in-flight
responses; uncertain loader identity yields unavailable evidence. This also applies to custom paths.

Typed evidence is `Confirmed<T> { value, source }`, `Assumed<T> { value, reason }`, or
`Unavailable { reason }`. Confirmed pair compilation is distinct from confirmed execution.
Missing/malformed `FA_QUANTS` never silently becomes confirmed Native, never adds a launch block;
older runtimes may give explicitly assumed guidance or unverified results, not a guessed default.

### 4.7 K/V pair classification (`src/inference/kv_support.rs`)

This replaces `MixedMainKv` and `is_mixed_main_kv_pair`. It becomes the only production
function that decides K/V policy:

```rust
pub enum KvSupport {
    Native,                                    // expected fused execution with adequate evidence
    Fallback { reason: String, impact: Impact }, // expected alternate path, not a startup guarantee
    Unverified { reason: String },              // nonblocking: insufficient execution evidence
    WontStart { reason: String, fix: String },  // known upstream refusal
}
pub enum Impact { SlowerDecode, NoFlashAttention, CpuAttention }
pub struct KvContext<'a> {
    pub basis: &'a KvSupportBasis, // typed evidence, runtime identity + basis version
    pub device: Option<&'a DeviceEvidence>, // selected id, arch, Metal bf16 support
    pub model: Option<&'a ModelEvidence>,   // metadata identity, attention/head dimensions
}
pub fn classify_kv_pair(
    k: KvType, v: KvType, flash_attn: FlashAttnMode, ctx: &KvContext<'_>,
) -> KvClassification; // { compiled_pair: Evidence<bool>, expected: KvSupport, evidence, reason, fix }
```

Use small optional records populated by existing device/GGUF metadata readers, not a speculative
hardware/model framework. `KvSupportBasis` stores backend, parsed pair evidence, loaded runtime
identity and schema/parser version. Device/model constraints can be confirmed, assumed or unknown.
Return compiled availability independently even when execution is unverified.

**Normalization and validation, before the truth table:** missing new-preset K/V normalize to
q8_0/q8_0; preserve explicit stored values. Normalize aliases/case once against the runtime's
accepted cache type inventory; accepted floating = f16/bf16/f32 where supported, quantized =
accepted nonfloating cache types. Missing FA resolves to the effective preset/server default
from versioned evidence; if unknown, evaluate possible modes and return unverified if outcomes
differ (never silently choose `on`). Unknown names/unsupported syntax use existing validation;
do not infer kernel support from acceptance or invent additional `WontStart` rules.

**Complete decision table (precedence top-down; E = equal, M = mixed):**

| Backend | FA | Accepted pair classes | Compiled availability / expected execution |
| --- | --- | --- | --- |
| any | off | E/M; V quantized, any accepted K | Regardless of pair list: `WontStart`, quantized V requires FA on/auto (known refusal). |
| CUDA/Metal/CPU/Vulkan/Unknown | off | E/M; V floating, K floating or quantized | Fused FA not selected: `Fallback{NoFlashAttention}`. No Native claim; non-FA device/model execution remains qualified/unverified if unsupported by evidence. |
| CUDA | on | E/M floating/floating, quant/float, float/quant, quant/quant | Confirmed pair present → Native **only** with adequate selected device, head-dimension and model constraints; otherwise Unverified execution with confirmed pair availability. Confirmed absent pair + evidenced conversion path → `Fallback{SlowerDecode}`. Missing/malformed/assumed list or unknown conversion/device/model constraints → Unverified. |
| CUDA | auto | same four classes, E/M | Quantized V requests FA as in §2; floating V requires evidence auto selected FA. Apply CUDA on row when FA is selected; confirmed FA disabled → NoFlashAttention; unresolved auto → Unverified. |
| Metal | on/auto | E, type in Metal FA set | Pair availability subject to bf16 hardware evidence; Native only with adequate device/model/head-dimension and FA-selection evidence. Unknown bf16 hardware or auto outcome → Unverified, not universally Native. |
| Metal | on | M, all four classes; or E outside Metal FA set | No Metal fused pair. Expected CPU attention only where backend/version/device/model evidence supports that path; otherwise Unverified with strong possible CPU-attention warning. |
| Metal | auto | M or E outside FA set; V quantized | FA required; same qualified CPU-attention/Unverified rule as on. |
| Metal | auto | M or E outside FA set; V floating | Expected NoFlashAttention where auto-disable/non-FA path evidenced; otherwise Unverified. |
| CPU | on/auto | E/M, all four classes | Unverified unless CPU-specific runtime/device/model evidence proves a path; no CUDA/Metal native or slowdown inference. |
| Vulkan | on/auto | E/M, all four classes | Unverified unless Vulkan-specific evidence proves a path; no borrowed CUDA/Metal pair list. |
| Unknown | on/auto | E/M, all four classes | Unverified; backend discovery failure is nonblocking. |
| any | unknown effective default | any accepted pair | Resolve modes as above; if not uniquely evidenced, Unverified, no new block. |

For Metal bf16, a compile flag/Apple Silicon label alone is not hardware execution evidence; the
runtime's `has bfloat` device line is. On a build at or after `METAL_FA_RULE_FROM_BUILD` (before any `METAL_FA_MIXED_KERNELS_FROM_BUILD`), a mixed pair is
`Fallback{CpuAttention}` with FA on (or auto with quantized V) and `Fallback{NoFlashAttention}` with
auto and floating V; an older or unknown build is `Unverified`.
Pair lists omit shape, attention implementation, offload selection and device restrictions; absence
of required context prevents a universal Native promise. Confirmed runtime observations may
refine expected execution only for their identity/model/device context.

CUDA rebuild links appear **only** for confirmed missing buildable CUDA kernels (not iq4_nl,
not FA off, not unknown evidence, not Metal/CPU fallback). Warn "may run attention on CPU"
when prediction is uncertain. Numerical Metal slowdown text is explicitly scoped to b11719,
M5 Max, Qwen3.8-27B IQ4_XS, head_dim 256 (§2.1), not asserted for all mixed pairs.

Callers: `presets/validation.rs` maps only `WontStart` to `KV_PAIR_WONT_START`, adds
`KV_PAIR_FALLBACK` or nonblocking `KV_PAIR_UNVERIFIED`; retain existing type validation.
Delete the no-snapshot q8→q4 block in `inference/launch.rs:439-460`; unknown basis is unverified,
not a platform-default native claim. Update calibration, bundle resolver/API and Doctor to use
this same result; bundles remain offered unless known WontStart.

**Actual cache migration:** bump capability cache schema/basis version. Deserialize legacy
`MixedMainKv { supported, reason, source }` through a legacy DTO (or discard old cache on version
mismatch), ignore its support boolean, set new basis unavailable and schedule reprobe.
A serde alias alone cannot convert this incompatible shape. Test legacy true/false, absent field,
malformed new field and current round-trip; none may manufacture confirmed support.

Authenticated `GET /api/llama-binary/kv-support?k=&v=&fa=&model_id=&device_id=` accepts optional
opaque server-resolved model identity and detected-device id, **not arbitrary paths**. Validate
device ids against discovery, resolve metadata server-side; absent metadata omits advisories
while capability remains useful. Device omission uses effective configured selection or unknown.
Return `{compiled_pair, support, evidence, reason, fix, advisories[], runtime_identity,
device_identity, model_metadata_identity, basis_version}`. Response/client cache keys include
model metadata identity (including revision), k/v/effective FA, runtime closure identity,
selected device and basis version. Invalidate on any change and discard stale responses.

**Partial runtime confirmation only:** log watcher patterns:

- `no FlashAttention vector kernel compiled for K/V types` → conversion fallback observed.
- `not supported, set to disabled` with `flash_attn` → FA disable observed.

These are nonblocking, session/context-specific evidence, not comprehensive verification.
No matching log line does not prove Native, GPU attention, or successful model execution.

### 4.8 Model-family K/V advisories (`src/inference/kv_advisories.rs`)

```rust
pub struct KvAdvisory {
    pub matches: FamilyMatch,         // gguf general.architecture / family / name regex
    pub severity: Severity,           // Info | Caution
    pub applies_to: KvPredicate,      // e.g. K below q8_0, V below q8_0, any quantized
    pub text: &'static str,
    pub source: &'static str,         // provenance shown in tooltip
}
```

Seed entries:

- **Gemma 4 family**, `Caution`, when K **or** V is quantized: "User/maintainer guidance:
  Gemma 4 can be sensitive to quantizing either K or V; prefer conservative cache precision and
  evaluate quality on your model." Mark as qualitative user guidance, not a measured universal
  severity claim or advice about every Gemma generation. Preserve known specific chart provenance
  only if actually available; do not invent a reviewed community source.
- **Qwen3.8-27B**, `Info`, K=q8_0/V=q4_0: "Maintainer reports this as a daily configuration
  with acceptable quality; validate for your workload." Source: maintainer observation, 2026-10;
  not a controlled quality measurement.
- Do not extrapolate to all Qwen3.x. Match the specific model metadata, not architecture alone.
- Remove the broad Qwen3/Apple-Silicon f16 tooltip now; S1's scoped result is already in §2.1.

Advisories never block/change values. Resolve known GGUF metadata through the model registry
(architecture plus model identity where needed). No metadata → `advisories: []`, not guessed
family advice. Keep guidance distinct from compiled/runtime capability evidence.

---

## 5. Prerequisites

Each check returns:

```rust
pub struct Check {
    pub id: &'static str,
    pub title: String,
    pub status: Status,       // Pass | Warn | Fail | Skipped
    pub detail: String,       // what we found, with versions and paths
    pub fix: Option<Fix>,     // Copy { command } | Link { url, label } | Action { id, label }  (Action = macOS brew)
    pub evidence: Vec<String> // exact commands run + trimmed output, shown under "Details"
}
```

There are three tiers: **present** (found), **version** (meets minimum), and **functional**
(a smoke test actually compiles something). The result is cached in `prereqs-last.json` with a
fingerprint (tool paths + versions + `CUDA_PATH` + VS instance id). The UI shows "validated N
days ago". **Validate** always re-runs the smoke test.

### 5.1 macOS

| Check | Present / version | Functional | Fix |
| --- | --- | --- | --- |
| Command Line Tools | `xcode-select -p`; `clang --version` | covered by smoke | Copy: `xcode-select --install` (opens Apple dialog) |
| Homebrew | `brew --version` (look in `/opt/homebrew/bin`, `/usr/local/bin` too: GUI apps lack shell PATH) | — | Link: brew.sh install command (we never pipe curl to bash) |
| git | `git --version` | `git ls-remote` reachability of the llama.cpp repo (timeout 15 s) | Action: `brew install git` |
| cmake ≥ 3.21 | `cmake --version` | smoke | Action: `brew install cmake` |
| ninja | `ninja --version` | smoke | Action: `brew install ninja` |
| ccache (optional) | `ccache --version` | — | Action: `brew install ccache` (Warn, not Fail) |
| Metal device | `system_profiler SPDisplaysDataType -json` or the existing `gpu/apple.rs` | — | — |
| Disk space | ≥ 2 GB provisional floor on app-root volume **and** sufficient additional-allocation estimate (§4.4); 114 MB S4 tree was an observed snapshot, not a peak guarantee | — | Text |
| **Smoke test** | — | configure + build a 1-file C++ project and link `-framework Metal` with Ninja under a temp dir | Shows the failing log inline |

**One-click install (D2):** `POST /api/llama-build/prereqs/fix {action:"brew_install", packages:[…]}`
runs `brew install <pkgs>` as a job, using the same log/tail/failure machinery as builds. The
package list is limited to `{git, cmake, ninja, ccache}`, checked server-side and never taken
from free text. When the job finishes, prerequisites are validated again.

### 5.2 Windows

| Check | Present / version | Functional | Fix |
| --- | --- | --- | --- |
| VS 2022 Build Tools + C++ | `vswhere -products * -version [17.0,18.0) -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath` (vswhere at `%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\`); fallback to the 5 known `vcvarsall.bat` paths from `Build-LlamaCpp.ps1:308-312` | vcvars capture succeeds and `cl.exe` resolves | Copy: `winget install Microsoft.VisualStudio.2022.BuildTools --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive"` + Link to the VS download page |
| Windows SDK | `WindowsSdkDir` in captured env | smoke | same as above |
| CMake + Ninja | from the VS "C++ CMake tools" component (captured env; present on ryne at `…\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\{CMake,Ninja}`), else PATH. Record which one was used in the check detail, since PATH may contain other copies (ryne: Strawberry ninja, standalone CMake) | smoke | add component `Microsoft.VisualStudio.Component.VC.CMake.Project` |
| CUDA Toolkit 13.x | `CUDA_PATH`; `nvcc --version`; 12.x = Warn "untested", < 12.4 = Fail | smoke | Link: NVIDIA CUDA 13.4 download; text on setting `CUDA_PATH` |
| NVIDIA driver | Parse actual driver version; compare selected toolkit/runtime to NVIDIA's documented OS-specific minimum-driver compatibility matrix, including minor-version compatibility limits and PTX/JIT caveats | backend/device probe, not compiler smoke alone | Link: NVIDIA compatibility docs + driver download |
| GPU compute cap | `nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader` | arch map hit, else Warn + manual arch field | — |
| git | `git --version` (also `%ProgramFiles%\Git\cmd`) | `git ls-remote` | Copy: `winget install Git.Git` |
| Disk space | **≥ 20 GB free provisional conservative Windows floor**, or greater estimated additional need (§4.4), pending actual FA-all peak measurement. Existing 1.0–2.4 GB CUDA trees did **not** establish FA-all configuration or peak use | recheck before stage/transaction | Text |
| Long paths | `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled` = 1 (Warn only) | — | Copy: the reg command (admin) |
| **Smoke test** | — | under captured vcvars env: `cmake -G Ninja` a project with one `.cu` (`__global__ void k(){}`) built with `CMAKE_CUDA_HOST_COMPILER=cl.exe` for the detected arch | Inline log. Recognize "unsupported Microsoft Visual Studio version" → hint: install the MSVC version this CUDA supports |

Driver validation uses the [CUDA compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/)
and selected toolkit release notes, with a cited/versioned matrix and parser fixtures. The
`nvidia-smi` "CUDA Version" field is a driver capability indicator, not an installed toolkit
version or sufficient major-number comparison. Minor-version compatibility may permit newer
toolkits with older drivers but has feature/PTX JIT limits; generated PTX/architecture-specific
code may need a newer driver. Unknown matrix entries warn and require functional backend evidence;
documented incompatibility fails build prerequisites. Do not invent a hard-coded minimum driver.

**vcvars capture (`toolchain_env.rs`):** run
`cmd /d /s /c "set "PATH=<sanitized base>" && call "<vcvars64.bat>" >nul && set"`
(ported from `Build-LlamaCpp.ps1:222-235`: a clean base PATH stops repeated captures from growing
it). Parse `KEY=VALUE` lines, keep the diff against the base env, and cache it in
`toolchain-cache.json` keyed by VS instance id + vcvars mtime. Every cmake/ninja/nvcc child gets
this env. Users never need a Developer Prompt.

**CUDA runtime DLLs:** at stage time, copy `cudart64_*.dll`, `cublas64_*.dll` and `cublasLt64_*.dll`
into staging, searching `<CUDA>\bin\x64`, `<CUDA>\bin` and `<CUDA>\lib` in that order (the same
three directories upstream release CI copies from). **Verified (S2):** CUDA 13.4 on ryne puts them
**only** in `bin\x64` (`cudart64_13.dll`, `cublas64_13.dll`, `cublasLt64_13.dll`), and `bin\` has no
DLLs. Refuse `ggml-*.dll` matches (ported guard). The binaries then don't depend on the
installer's PATH edits.

**Multiple CUDA toolkits:** ryne has `CUDA_PATH` → v13.4 plus `CUDA_PATH_V12_8`, `_V13_3` and
`_V13_4`, and PATH contains both `v13.4\bin` and `v12.8\bin`. Use `CUDA_PATH` only, pass `nvcc` by
absolute path, and put `<CUDA_PATH>\bin` and `\bin\x64` **first** in build/compile child PATH.
This is **not** the staged-runtime environment: verification removes toolkit/build-tree search
paths (§4.6.1). Show every detected toolkit in prerequisite detail.

### 5.3 Linux (untested)

The same table shape as macOS, with distro-neutral fixes (Copy text only), plus nvcc/driver
checks when CUDA is selected. The UI badge reads "Linux: untested".

---

## 6. UI/UX

### 6.1 Pill (`#llama-pill`, `llama-updater.js`)

| State | Look | Text |
| --- | --- | --- |
| idle | neutral | `llama.cpp b11727 · src` (provenance suffix from manifest: `src`, `prebuilt`, `custom`) |
| update | orange (existing) | `b11730 available` |
| building | blue, animated | `Building · compile 63%` (phase + ninja `[n/m]` progress when parsable) |
| error | **red** | `Build failed: configure`. Click opens the panel scrolled to the error |
| rollback-available-after-failure | red until dismissed | `Update rolled back` |
| installed-restart-failed | amber | `Runtime installed; model restart failed` with Retry / diagnostics / manual rollback |

Popover:

- **Download prebuilt b11730** (primary unless the live set is a source build)
- **Build b11730 from source** (primary when the live set is a source build)
- Warning when replacing a source build: "Prebuilt FA kernels may differ. We will probe the
  candidate; confirmed missing CUDA kernels can use slower conversion fallback." Show a specific
  downgrade only when candidate evidence establishes one, not from the `prebuilt` label.
- **Open llama.cpp Runtime…**

### 6.2 llama.cpp Runtime panel

Opened from the pill and from Settings. Built as `llama-runtime-panel.js` + `.css`, and registered
in `tests/ui/core/js-module-baseline.json`.

```
┌ llama.cpp Runtime ───────────────────────────────────────────────┐
│ ⚠ Last build failed at CONFIGURE  (exit 1)          [Copy diag] │  ← error banner, top, never collapsed
│   CMake Error at vendor/cpp-httplib/CMakeLists.txt:151 …        │
│   Hint: stale CMake cache → [Wipe build dir & retry]            │
├──────────────────────────────────────────────────────────────────┤
│ Current: b11727 · source build · FA all · CUDA 120a              │
│          built 2026-10-19 · provenance hashes match   [Roll back to b11720 prebuilt] │
├ Prerequisites ─────────────────────────── validated 2 h ago [Validate] ┤
│ ✔ VS 2022 Build Tools 17.14 (MSVC 14.44)                         │
│ ✔ CUDA 13.4 · detected driver meets documented matrix [Details] │
│ ✖ Ninja not found     [Copy fix]                                 │
│ ✔ Smoke test: CUDA + MSVC compile                    [Details ▸] │
├ Build ───────────────────────────────────────────────────────────┤
│ Source:  (•) Latest release b11730   ( ) master HEAD (a1b2c3d)   │
│ FA K/V kernels: all pairs (fixed)                                │
│ CUDA arch: [auto: 120a-real ▾]   Jobs: [auto ▾] ☐ leave headroom │
│ ☐ Wipe build dir (full rebuild)                                  │
│ [Build & install]   last build took 13m 40s                      │
├ Log ─────────────────────────────── [Follow] [Open file] [Copy] ─┤
│ ▸ preflight ✔  fetch ✔  configure ✔  compile … [812/1430]        │
│ …live tail…                                                      │
├ Disk ────────────────────────────────────────────────────────────┤
│ Source 412 MB · build-cuda 6.1 GB · previous set 310 MB          │
│ [Clean build cache]  (next build will be a full rebuild)         │
└──────────────────────────────────────────────────────────────────┘
```

Rules:

- **Build & install** is disabled while any prerequisite is `Fail`. Its tooltip names the failing
  checks. A `Warn` does not disable it.
- Errors render at the top in full: phase, summary, first error line, last 80 lines, and the hint
  button. "Copy diagnostics" copies: failure JSON + prereq results + manifest + OS/GPU summary +
  log path.
- The log view is the real log, uncollapsed while a build runs, with a follow toggle.
- Cancel is visible with confirmation before the journal; disabled with "Installing/recovering —
  cannot cancel safely" after it begins. Clean/fix/validate controls show ownership conflicts.
- Settings that don't apply to the platform are hidden, not disabled (no CUDA controls on macOS).

### 6.3 K/V controls (wizard Guided, wizard Pro, preset editor, bundle drawer)

Shared client module `kv-support.js` calls `/api/llama-binary/kv-support` (debounced), sends the
optional registry model id and selected device context, and caches by the complete §4.7 key,
not executable hash alone. Reject old responses after model/device/runtime changes.

- **Native:** quiet badge state only for evidenced expected execution; distinguish "pair compiled"
  from "execution unverified" in details.
- **Fallback:** amber contextual reason. Rebuild link only for confirmed missing buildable
  CUDA kernels. Metal CPU attention warning says expected/possible according to evidence;
  no Metal or CPU rebuild recommendation.
- **Unverified:** neutral/info "Could not verify execution for this runtime/device/model",
  never disables launch; show confirmed compiled-pair evidence separately if present.
- **Won't start:** red line with the fix. Launch and Save & Start are disabled with the same reason.
  Plain Save stays enabled (presets are portable across machines), and the preset card shows the
  red state.
- **Advisories (§4.8):** info/caution lines beneath, with source in the tooltip.

Per surface:

- **Guided (`#hw-kv-tiles`):** add a fourth tile, **"K q8_0 · V q4_0"** (label: "saves memory,
  model-dependent"). Its sub-label shows Native / Slow fallback / Unverified. Hidden on Metal. Selecting
  it sets the selects independently. Fix `spawn-wizard-guided.js:212-216` so a mixed pair
  highlights the matching tile instead of none. Keep `applyUseCaseKvDtype` writing equal pairs
  only.
- **Pro:** same classification line; confirmed native pairs first, evidenced fallback separately,
  unknowns labelled unverified (not "slow"). No accepted types hidden.
- **Preset editor:** same line. Change the new-preset V default to `q8_0` (D11) in `index.html`
  `#modal-ctv` "(default)" label, `presets.js:124-125,1461-1463,1530-1532`,
  `calibration/executor.rs:2231-2232`, and `setup-view.js:1087,1157`. Rewrite the `#modal-flash-attn`
  tooltip to match §2.1–2.2.
- **Bundle drawer:** the `q8_0/q4_0` KV policy is shown whenever it is not `WontStart`, with the
  classification line. Remove the "prohibited" wording.
- **Launch card chip (`setup-view.js`):** `⚠` for Fallback, neutral `?` for Unverified, `✖` for
  WontStart, each with the shared reason; existing syntax validation still applies.

---

## 7. API

All routes require the API token (`check_api_token`) and follow `rapid_mlx_repair.rs` patterns.

| Method & path | Purpose |
| --- | --- |
| `GET  /api/llama-build/prereqs` | Cached result + fingerprint staleness |
| `POST /api/llama-build/prereqs/validate` | Run all checks incl. smoke test (job; returns job id) |
| `POST /api/llama-build/prereqs/fix` | macOS only: `{action:"brew_install", packages}` (allowlisted) |
| `GET  /api/llama-build/options` | Available tag/master identities (preview; job pins once), arch, additional disk assumptions, cache cap; **no timing estimate** |
| `POST /api/llama-build/start` | `{ref: "latest_release" | "master", cuda_arch?, jobs?, headroom, wipe}` → `202 {job_id, resolved_commit}` / `409 {owner}` |
| `GET  /api/llama-build/jobs/:id` | `{phase, progress, status, cancellable, outcome?, failure?, started_at, log_cursor}`; distinguishes installed/restart-failed |
| `GET  /api/llama-build/jobs/:id/log?after=N` | Incremental log lines |
| `POST /api/llama-build/jobs/:id/cancel` | Cancel/reap before journal; conflict after noncancellable boundary |
| `GET  /api/llama-build/cache` / `DELETE` | Sizes incl. managed ccache/cap / clean inactive build trees + ccache (+ optional src); `409` if owned |
| `POST /api/llama-binary/rollback` | Swap `bin` ↔ `bin-previous` |
| `GET  /api/llama-binary/kv-support` | §4.7 |
| `GET  /api/llama-binary/version` | **extended**: adds `provenance` from manifest |

Document these in `docs/reference/api.md`.

---

## 8. Error catalog (known-failure hints)

`job.rs` matches log lines against patterns and attaches a `Fix`. Each pattern needs a fixture
log in `tests/fixtures/llama_build/`:

| Pattern | Hint |
| --- | --- |
| CMake cache path that no longer exists (from `find_stale_cache_paths`) | Auto-healed; if still failing: "Wipe build dir & retry" |
| `unsupported Microsoft Visual Studio version` | Install the MSVC toolset supported by this CUDA version |
| `No CUDA toolset found` / `CUDA_PATH` unset | Re-run prerequisites; set `CUDA_PATH` |
| `nvcc fatal : Unsupported gpu architecture` | Pick a lower arch / update CUDA |
| `fatal: unable to access` (git) | Network / proxy; show the URL |
| `No space left on device` / `There is not enough space on the disk` | Clean build cache; show free space |
| `LNK1104: cannot open file` on `bin/…` | A process holds the file (should not happen: we stage outside `bin/`) |
| `xcrun: error: invalid active developer path` | `xcode-select --install` |
| `Unable to read from file` / `Can't lstat` in `ui-assets.cmake` | Build path went through a symlink; report as a foundry bug (we canonicalize). Workaround: wipe build dir |
| `UI: HF download failed` / CCCL `FetchContent` git failure | Network access to huggingface.co / github.com is required during the build |
| Killed / exit 137 / OOM in compile | Lower jobs (`headroom`) |
| Unknown | No hint; full tail + log path still shown |

---

## 9. Phased implementation

The following tests/spikes/gates are **future implementation acceptance**, not work executed by
this plan-only revision. Each implementation phase ends with touched-module `cargo test`,
`cargo clippy -- -D warnings`, relevant UI tests, and a receipt in
`docs/plans/evidence/llamacpp-compilation/phase-N.md` (baseline, files, commands + exit codes).
Track the checklist below; do not claim manual/model proof when it is deferred.

### Phase 0: Prior spikes (observations recorded 2026-10-19; peak disk budget still provisional)

| ID | Question | Result |
| --- | --- | --- |
| S1 | Metal mixed behavior / tooltip | Measurements only on b11719, M5 Max, Qwen3.8-27B IQ4_XS, head_dim 256 (§2.1). CPU attention and slowdown numbers are scoped to those rows, not universal Metal guarantees. |
| S2 | CUDA 13.4 Windows layout | DLLs only in `bin\x64` (§5.2). VS Build Tools 17.14.36717, MSVC 14.44.35207, SDK 10.0.26100, VS-bundled CMake + Ninja present, long paths enabled. Strawberry gcc/ninja/ccache on PATH (pinned out, §4.5). Multiple toolkits (§5.2). |
| S3 | Prebuilts / probe | Observed release CI did not set `GGML_CUDA_FA_QUANTS`; this alone proves no installed asset's pair list. Ryne binary reported `all` with missing-model probe. Adopt CCCL flag for CTK < 13.5; use actual loaded-backend evidence for capabilities. |
| S4 | Metal build without Xcode.app | Works with Command Line Tools only (§4.5). Found the symlink-path and build-time network traps. |
| S5 | Build time / disk | No build timing estimate. Observed Metal tree 114 MB, existing CUDA trees 1.0–2.4 GB, checkout ~0.5 GB, Windows bin set ~0.75 GB. CUDA observations established **neither FA-all configuration nor peak space**. Windows free floor is provisionally 20 GB pending peak measurement. |
| S6 | — | Dropped (not needed). |

### Phase 1: K/V classification backend (ships value without builds)

1. [x] Tests first: expand §4.7 into backend × FA off/on/auto/default × E/M × float/quant
   combinations, all accepted types, unknown syntax, normalized defaults. Assert off + floating
   V never Native; only known quantized-V/off refusal adds a policy block. Cover Metal bf16
   hardware yes/no/unknown, missing model/head/device constraints, CUDA absent/malformed/assumed
   pairs and CPU/Vulkan/Unknown. Verify compiled availability independent of expected execution.
2. [x] Add `kv_support.rs`, typed basis to `llama_cpp_capabilities.rs` / `generate_snapshot`, and (done; "wrong loaded backend" is asserted by the Phase 3 shared verifier)
   bounded runtime-closure probe/identity helper in `llama_runtime/probe.rs`. Build this helper
   in Phase 1; Phase 3's shared verifier reuses it, avoiding a phase dependency cycle. Fixtures
   cover valid `all`, explicit list + f16-f16, missing/malformed/unknown records. Test timeout,
   output flood, arbitrary exit vs recognized missing-model exit, controlled cwd/env/local path,
   wrong loaded backend, plugin-only replacement and stale in-flight results.
3. [x] Version cache migration with legacy DTO/discard + reprobe: legacy `supported` true/false,
   absent, malformed and new round-trip. No alias-only migration or inferred Native.
4. [x] Update validation, launch, calibration, resolver, bundle API and Doctor; delete special
   no-snapshot block. Keep syntax errors; assert unknown evidence never adds a launch block.
5. [x] Add authenticated model-aware endpoint; test registry id resolution, arbitrary path (done server side; client cache keys land in Phase 2)
   rejection, absent metadata (`advisories=[]`), model revision/device/basis cache keys.
6. [x] Add partial log watchers; absence of pattern never upgrades evidence. Update old tests:
   `validation.rs:382-400`, `bundle.rs:1242-1267`, `doctor.rs:296-376`, `launch.rs:816-820`,
   `llama_cpp.rs:2219`, `candidates.rs:495`.

**Gate (met 2026-10-19; receipt `docs/plans/evidence/llamacpp-compilation/phase-1.md`):** confirmed `FA_QUANTS=all` gives confirmed compiled availability; only sufficient
device/model evidence gives expected Native. Valid missing-pair CUDA fixture warns fallback;
missing/malformed evidence is nonblocking Unverified. Metal warnings retain context; no universal
slowdown. Quantized V/off is refused before spawn, floating V/off never called fused Native.
Tests green, migration reprobes, plugin-only change invalidates both backend and UI basis.

### Phase 2: K/V UI + advisories + default unification

1. [x] Implement scoped advisory table and metadata matching tests: Gemma guidance for quantized
   K **or** V, specific Qwen3.8-27B observation, no invented broad Qwen family evidence.
2. [x] Wire shared client into Guided, Pro, editor, bundle drawer and launch chip. Send resolved
   model id; empty/missing metadata still yields useful capability and no advisory.
3. [x] Unify only new defaults to q8_0/q8_0; preserve stored presets; remove stale tooltip.
4. [x] Playwright (`docs/agents/playwright.md`): CUDA Guided mixed tile and editor show quiet
   Native with adequate confirmed context, amber only for evidenced missing-kernel fallback,
   neutral Unverified for unknown execution (including compiled pair but unknown shape). Metal
   tile hidden, Pro contextual CPU-attention warning, no rebuild link. FA off + floating V is
   no-FA, quantized V blocks Start/Save & Start but not Save. Test Gemma guidance on K-only and
   V-only quantization, no-model advisory absence, model revision/device/plugin identity changes
   invalidating cached UI and discarding late responses.

**Gate (met 2026-10-19; receipt `docs/plans/evidence/llamacpp-compilation/phase-2.md`):** Guided/Pro/editor/bundles render identical evidence/text for identical context; no
always-amber CUDA assumption. Screenshots: `docs/screenshots/artifacts/llama-build/`.

### Phase 3: Shared promotion, retention, rollback

1. [ ] `promote.rs` temp-dir fault tests on macOS **and Windows**: inject failure and process
   death before/after **each** intent flush, rename (including compensating renames and previous
   preservation), staged/live health/backend check, commit, restart and cleanup boundary.
   Replay startup recovery twice; assert last usable live and old previous survive uncommitted
   failure, no journal-referenced staging swept, committed state finishes retention. Include
   first install, disk-full journal write, locked DLL, corrupt/ambiguous journal and rollback twice.
2. [ ] Extract shared verifier/promotion for prebuilt and source; preparation before execution,
   sanitized loader closure, backend proof. Fake verifier records calls: no `job.verify` duplicate
   GPU probe. Test journaled restart intent precedes model stop, including death after stop.
3. [ ] Runtime mutex across update/promote/rollback/uninstall/model lifecycle; fixed order tests,
   cancel before journal vs rejection after. Assert compile/download do not hold runtime mutex.
4. [ ] Test binary failure causes transactional restore, whereas model restart failure returns
   `InstalledRestartFailed`, keeps installed binaries and exposes manual retry/rollback.
5. [ ] Register startup recovery before model/API startup and orphan cleanup; add rollback API
   and conditional legacy retention sweep. Prebuilt manifest is provenance, never capability.

**Gate:** macOS/Windows fault suite proves idempotent recovery at every boundary, including a
failed rollback rename. Existing `llama_binary.rs:1471-1580` tests pass; manual prebuilt swap on
both platforms leaves stable live/previous sets and reports restart failure distinctly.

### Phase 4: Build backend

Order (TDD per unit):

1. [ ] `paths.rs`: journal/temp/ccache paths, managed-path containment, log/JSON pair retention.
2. [ ] `gpu.rs`: nvidia-smi fixtures, arch map and device features.
3. [ ] `configure.rs`: golden accelerator × wipe/heal argv; always CUDA `FA_QUANTS=all`,
   never deprecated flag; parse actual configure pair list.
4. [ ] `manifest.rs`: provenance round-trip/hash mismatch; never converts to capability.
5. [ ] `prereqs*.rs`: version/vswhere/vcvars fixtures; cited toolkit/driver matrix incl. minor
   compatibility/PTX cases; disk additional-growth estimator, no live-dir double counting,
   20 GB Windows floor and stage recheck; no invented minimum driver.
6. [ ] `toolchain_env.rs`: captured build env vs sanitized staged env; malicious PATH/build-tree
   DLL fixture must not satisfy closure. Missing intended backend/device fails verification.
7. [ ] `source.rs`: local bare repo tag/master movement after resolution, exact detached commit
   pinned through GC/build, master git-describe normalization; build-number-only match rejected.
8. [ ] `stage.rs`: fake trees/loader fixtures; stale executable/plugin, missing DLL, source CUDA
   configure-all but runtime-list absent/not-all rejected by shared verifier. Model proof deferred
   explicitly if no local model is available.
9. [ ] `job.rs` fake toolchain: fail every phase; disk exhaustion compile/stage/promotion preserves
   live sets. Ownership conflict matrix tests (build vs clean, validation vs clean, install vs
   check/validation/build). Windows Job Object and Unix supervisor integration fixtures spawn
   grandchildren; cancel **and parent death** reap/contain writers before ownership release.
   Test PID identity mismatch never kills unrelated process, next job cannot inherit orphan writers.
10. [ ] Cache cap/clean and last-3 log+JSON tests, active exemptions, bounded git/dependency
    retention, interruption recovery preceding staging sweep.

**Gate:** `cargo test llama_build` green on macOS and Windows platform fixtures. Real Metal
build via CLI hook (`cargo run -- llama-build --dry-run` prints plan) passes compile/loader/backend
checks and journaled promotion; record model smoke separately or mark deferred manual QA.

### Phase 5: Build API + UI

1. [ ] Routes (§7): auth, single-flight/conflict owner, cancel boundary, model-id/path tests,
   no estimated-time field, returned pinned commit, cache cap configuration with finite bounds.
2. [ ] Panel/pill/popover + macOS install, distinguish installed/restart-failed, ownership-disabled
   clean/fix/validate and noncancellable promotion.
3. [ ] Error catalog fixtures including disk exhaustion, closure/backend mismatch and recovery.
4. [ ] Playwright: prereq Fail → disabled reason; failed job → red diagnostics; source success →
   provenance; conditional prebuilt downgrade, no guessed pair list; restart failure → installed
   amber state, retry/manual rollback; unknown capability never gets a rebuild link or launch block.

**Gate:** manual macOS missing-tool → explicit install → validate → build → journaled promote;
Windows CUDA validate → FA-all compile/loader/backend evidence → promote. With a suitable model,
manual smoke verifies selected backend/device and actual attention path for q8_0/q4_0, not just
absence of fallback logs. Without a model, record deferred manual QA, not execution success.

### Phase 6: Doctor, docs, release

1. [ ] Doctor: stale prereqs (if ever built), provenance mismatch, unavailable capability,
   pending recovery and legacy backups. Rebuild link only for confirmed missing buildable CUDA
   kernels; never generic Metal/CPU fallback.
2. Docs: `docs/reference/llama-runtime.md` (new; user-facing, covering build, prerequisites,
   troubleshooting, the K/V support table) and `docs/reference/api.md`. Update
   `docs/reference/spawn-wizard.md` and `windows-support.md`, and the `README.md` feature bullet.
3. CHANGELOG via the normal release flow.

**Gate:** reviewer pass; manual QA matrix (§10) complete.

---

## 10. Manual QA matrix

These are acceptance cases to execute during implementation qualification, not results of this
documentation edit. A checkmark indicates the platform must be covered. Injected failures use
the Phase 3/4 harnesses; model smoke needs an available local model or a recorded deferral.

| Scenario | macOS (M-series) | Windows CUDA 13.4 | Linux |
| --- | --- | --- | --- |
| Prereqs all pass, smoke green | ✔ | ✔ | untested |
| Missing tool → fix shown; macOS one-click install | ✔ | Copy fix | — |
| Build latest tag, pinned commit verified, promote, then model smoke if available | ✔ | ✔ | — |
| Master advances after resolution → build still uses pinned commit; git-describe verified | ✔ | ✔ | — |
| Incremental rebuild uses cache; elapsed time recorded, no promised speedup/estimate | ✔ | ✔ | — |
| Configure failure → hint → inactive wipe → success | ✔ | ✔ | — |
| Compile failure/exhausted disk → live untouched, model still running | ✔ | ✔ | — |
| Cancel compile with grandchildren → all writers reaped before next build | Unix supervisor | Job Object | — |
| App process death during compile → containment, identity-safe recovery, no orphan writer | ✔ | ✔ | — |
| Stage/promotion disk exhaustion incl. journal write → usable sets preserved | ✔ | ✔ | — |
| Missing staged library / polluted PATH / CPU-only load → intended backend proof fails | ✔ | ✔ | — |
| Failure/death before and after every rename (forward/rollback), health check, commit, restart → recovery twice | ✔ | ✔ incl. locked DLL | — |
| Recovery completes before orphan cleanup/model launch; corrupt journal retains sets | ✔ | ✔ | — |
| Model contention → journal intent then stop/reap/probe; failed staged check restores old service | ✔ | ✔ | — |
| Installed runtime valid but model restart fails → installed amber outcome, retry/manual rollback | ✔ | ✔ | — |
| Cancel before journal works; after journal rejected through recovery | ✔ | ✔ | — |
| Concurrent build/clean/validation/toolchain fix + runtime swap/model start obey conflict matrix | ✔ | ✔ | — |
| Rollback twice, retaining old live/previous through failed compensation | ✔ | ✔ | — |
| Prebuilt after source → downgrade only if actually evidenced; unavailable list nonblocking | ✔ | ✔ | — |
| Clean cache excludes active work, clears managed ccache/builds, cap applies on next build | ✔ | ✔ | — |
| Restart mid-build: contain descendants → journal recovery → interrupted job → orphan cleanup | ✔ | ✔ | — |
| After 3 updates stable live/previous, bounded caches/git, 3 completed log+JSON pairs | ✔ | ✔ | — |
| K/V full §4.7 matrix: FA off/on/auto/default, accepted vs unknown, E/M floating/quantized | ✔ | ✔ | — |
| Confirmed compiled pair + adequate context → Native; missing CUDA kernel → fallback; unknown context → Unverified | Metal context rows | CUDA rows | — |
| Metal bf16 hardware evidence absent/false; head dimensions unknown → no universal Native | ✔ | — | — |
| Missing/malformed probe, output flood, timeout, non-model error → unavailable, not new launch block | ✔ | ✔ | — |
| Backend/plugin replaced with same executable → backend + UI caches invalidate | ✔ | ✔ | — |
| Legacy mixed_main_kv true/false cache → ignored and reprobed | ✔ | ✔ | — |
| Model change/revision/device/FA change invalidates UI; no metadata → no advisory | ✔ | ✔ | — |
| Gemma guidance for quantizing K or V visible across surfaces, no broad Qwen claims | ✔ | ✔ | — |
| Model smoke unavailable → explicitly deferred manual QA; no false model-execution pass | ✔ | ✔ | — |

---

## 11. Invariants

1. Live runtime mutation is only through journaled promote/rollback/uninstall/recovery, with one
   shared verifier preparing permissions/quarantine before execution and proving installed closure.
2. Last usable sets survive every uncommitted failure/crash; recovery is idempotent and precedes
   orphan cleanup/model launch. Only terminal recovery may prune superseded sets/journal.
3. Stable completion retains live + at most one previous; transient journal-owned sets are allowed.
   Checkout is unpatched; one build owns checkout/trees, caches/logs are bounded and active-safe.
4. Fixed lock order (§4.3.1); runtime mutation and model lifecycle serialize at swap, not downloads
   or compile. Cancel/crash descendants cannot write after ownership release; no blind PID kills.
5. Journaled promotion is noncancellable. Model restart failure is a distinct installed outcome,
   not automatic bad-binary rollback.
6. One classifier owns K/V policy. Only known WontStart adds a launch block; existing type/syntax
   validation remains. Fallback and Unverified explain uncertainty; model guidance never blocks.
7. Compiled pairs are not model execution proof. Native requires adequate context; FA off
   floating V is no FA, never fused Native. Missing/malformed probes never become confirmed Native.
8. Runtime identity includes executable + loaded backend artifacts, paths/hashes and relevant
   device context; backend and UI invalidate together. Legacy booleans are ignored and reprobed.
9. Manifests/configure logs are provenance, not capability authority. Source CUDA still requires
   observed FA-all before promotion; general launch probing failure stays nonblocking.
10. Disk admission budgets additional allocations plus margin, not occupied live dirs; Windows
    provisional floor is 20 GB. Compile/stage/promotion exhaustion preserves usable sets.
11. All failures expose phase/command/exit/tail/log (unavailable values explicit); last 3 completed
    log+JSON pairs retained. No prerequisite install without click; Windows install never automated.
12. Model id is server-resolved, no arbitrary paths. No model metadata → no advisory; cache key
    includes metadata revision/k/v/fa/runtime/device/basis version. No broad invented family claims.

---

## 12. Risks and open items

- **Upstream churn.** CMake option names change (FA_ALL_QUANTS → FA_QUANTS happened 6 weeks ago).
  Mitigation: golden tests plus parse `FlashAttention K-V type combinations:` configure status
  as provenance, cross-check source CUDA `all` with actual loaded-backend probe before promotion.
- **Build time for FA `all`** is the longest CUDA configuration (49 pair instantiations per head
  size). ccache (already present on ryne, optional on macOS) keeps rebuilds incremental; the UI
  shows live progress and the last build's duration.
- **Build-time network.** Configure fetches CCCL from GitHub (CTK < 13.5) and the build downloads
  web-UI assets from Hugging Face. Offline builds are unsupported. Preflight checks both hosts.
- **Probe-format churn.** Unknown/missing/malformed `FA_QUANTS` is unavailable evidence, not a
  guessed native default. Doctor reports it; ordinary launches stay nonblocking, source CUDA
  install cannot claim verified FA-all.
- **Antivirus on Windows** can lock staged/live DLLs. Bounded retry (3 × 500 ms) then journaled
  recovery on every rename failure; retain recovery material if compensation also fails.
- **Gatekeeper.** Apply exec permissions/quarantine preparation before any staged probe; surface
  preparation failure rather than relying on locally built binaries normally lacking quarantine.
- **Advisory accuracy.** The seed entries are maintainer observations, not measurements we ran.
  Each entry carries its source string. Replace them with calibration-derived data later.
