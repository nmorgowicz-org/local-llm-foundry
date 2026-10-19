# Local LLM Foundry Binary Lifecycle — llama.cpp Runtime Management

The canonical executable is `local-llm-foundry` (`.exe` on Windows). The
`llama-monitor` executable remains an in-place compatibility alias through 2.x.

This document describes how Foundry discovers, installs, updates, and version-tracks
the `llama-server` binary. It is the reference implementation for backend runtime management.
The Rapid-MLX integration plan (`docs/plans/20260611-rapid_mlx_integration.md`) explicitly
names this system as the quality baseline that all future backends must match.

## Source files

| File | Responsibility |
| --- | --- |
| `src/llama/llama_cpp_downloader.rs` | Release fetching, asset selection, download/extract, artifact cleanup |
| `src/web/api/llama_binary.rs` | HTTP endpoints: version, latest, releases, release, platform-info, update, restart |
| `src/web/api/llama_runtime_routes.rs` | `POST /api/llama-binary/rollback`, `GET /api/llama-binary/runtime-status` |
| `src/llama_runtime/gate.rs` | The single runtime-mutation gate and swap-window flag |
| `src/llama_runtime/promote.rs` | Journaled promote / rollback / uninstall engine and crash recovery |
| `src/llama_runtime/verify.rs` | The one verification path (prepare, probe, evaluate) shared by every install kind |
| `src/llama_runtime/manifest.rs` | `llama-build-manifest.json`: provenance only, never an authority |
| `src/llama_runtime/host.rs` | Production host: model stop/restart and verification wiring |
| `src/llama_runtime/startup.rs` | Startup recovery of interrupted transactions |
| `src/model_download.rs` | Model-file download manager: dedup guard, existing-file check, concurrency limit |

## Binary path

The path to `llama-server` is resolved at startup from `AppConfig::llama_server_path`.
The managed runtime is the whole `bin/` directory: the executable plus every co-located
shared library. Installs always target the managed `bin/`. When `llama_server_path`
points elsewhere (a custom path) the update still installs to the managed `bin/`, leaves
the custom path untouched, and reports it as `custom_server_path` so the UI can say so.

## API endpoints

All endpoints require the configured API token (`Authorization: Bearer <token>`).

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/api/llama-binary/version` | Currently installed build number |
| `GET` | `/api/llama-binary/latest` | Latest GitHub release metadata (30-min cache) |
| `GET` | `/api/llama-binary/releases` | Last 8 releases for the version picker |
| `GET` | `/api/llama-binary/release?build=N` | Metadata for one specific build |
| `GET` | `/api/llama-binary/platform-info` | Detected OS, arch, available backends |
| `POST` | `/api/llama-binary/update` | Download and install a release (journaled promotion) |
| `POST` | `/api/llama-binary/rollback` | Swap the live and previous runtime sets |
| `GET` | `/api/llama-binary/runtime-status` | Previous availability, interrupted-transaction state, gate owner |
| `POST` | `/api/llama/restart` | Restart the running llama-server with its current config |

### `GET /api/llama-binary/version`

Runs `llama-server --version` synchronously (blocking task), combines stdout + stderr,
and extracts the build number with the pattern `(?:version|build)[:\s]+(\d+)`.

Returns `{ build: N, version: "bN", path: "..." }`.
Returns `{ build: null, version: null, path: "..." }` if the binary is missing or
the output does not match.

### `GET /api/llama-binary/latest`

Calls `list_releases` and returns the first (most recent) result. Cached for 30 minutes
in a process-static `Mutex<Option<(Instant, Value)>>`. Cache is bypassed only on expiry,
not on update install.

### `GET /api/llama-binary/releases`

Returns the most recent 8 releases. Each entry includes tag name, publication date, and release
body (notes). The endpoint returns all assets for each release without filtering by the current
platform; the caller (UI) applies its own `select_assets` filter to present only relevant
downloads for the host.

### `GET /api/llama-binary/release?build=N`

Fetches a single release by build number via `get_release_by_tag("bN")`, which hits
`https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/bN`.
Used by the version picker rollback flow.

### `GET /api/llama-binary/platform-info`

Returns:

- `os`: `"macos"` / `"linux"` / `"windows"`
- `arch`: `"aarch64"` / `"x86_64"`
- `arch_label`: human-readable architecture label
- `auto_backend`: backend chosen automatically for this platform
- `label`: short human-readable platform label (e.g. "Apple Silicon Metal")
- `backends`: array of available backend choices, each with:
- `id`: backend identifier (e.g. "cuda12", "vulkan", "rocm", "sycl", "openvino")
  - `label`: display label
  - `note`: short description of requirements
  - `recommended`: whether it is the recommended option
- `multi_backend`: true when the platform offers multiple selectable backends (Linux and Windows)

The UI uses this to pre-select the correct backend and to render backend-choice UI on
multi-backend platforms. Linux and Windows default to their generic CPU runtime;
selecting an accelerator backend is an explicit request and does not probe or
install that platform's driver/runtime.

The Intel choices have different purposes:

- `openvino` selects llama.cpp's OpenVINO archive. OpenVINO can execute on an
  Intel CPU, GPU, or NPU; GPU/NPU acceleration requires the corresponding
  OpenVINO device and driver to be available to the host or container.
- `sycl` selects llama.cpp's oneAPI/SYCL archive for Intel GPU execution,
  including supported integrated GPUs.
- `vulkan` is another possible Intel iGPU path when a Vulkan driver is
  available. It is not an OpenVINO or oneAPI runtime.

For AMD systems, `rocm` is the dedicated AMD GPU backend when the GPU and
ROCm version are supported. `vulkan` is the broader cross-vendor option and
can also use many AMD integrated GPUs when the Vulkan driver is available.
The generic CPU fallback remains valid when neither runtime is available.

The generic `cpu`/`avx2` choice remains available as the portable fallback and
does not mean that all Intel-optimized CPU execution is disabled.

### `POST /api/llama-binary/update`

Body (all fields optional):

```json
{
  "backend": "metal",
  "tag": "b9479"
}
```

`backend` defaults to the platform default (`metal` / `cpu` / `avx2`).
`tag` defaults to the latest release when omitted.

Response:

```json
{
  "ok": true,
  "outcome": "promoted",
  "version": "b9512",
  "backend": "metal",
  "arch": "arm64",
  "sha256": "abc123...",
  "server_restarted": true,
  "restart_applicable": true,
  "custom_server_path": null,
  "warnings": []
}
```

- `outcome` is one of:
  - `promoted`: installed, verified, and (when a model was running) restarted.
  - `installed_restart_failed`: the new runtime is installed and verified, but the model
    could not be restarted. `ok` is still `true`; `restart_error` carries the reason,
    `server_restarted` is `false`, and nothing was rolled back. Rollback stays available.
  - `rolled_back`: verification or the swap failed; the previous runtime is live again.
    `ok` is `false` and `error` carries the reason.
  - `failed`: nothing was changed (for example the journal could not be written).
- `sha256`: SHA256 of the installed llama-server binary.
- `server_restarted`: true when the backend already restarted the server with its previous
  config; the frontend must skip its own restart call when this is true.
- `restart_applicable`: false when no llama.cpp model was running (including Rapid-MLX).
- `409` with `{"owner": "update" | "rollback" | "uninstall" | "promote"}` when another runtime
  mutation is in flight, or `409` when an earlier interrupted transaction could not be
  reconciled (nothing is modified until it is resolved).

### `POST /api/llama-binary/rollback`

Swaps the live and previous runtime sets under the same gate and journal as an update,
verifying the set that becomes live. Rolling back twice returns to the original runtime.
Same response shape as update; errors with `outcome: "failed"` when no previous set exists.

### `GET /api/llama-binary/runtime-status`

```json
{
  "ok": true,
  "busy_owner": null,
  "live": { "label": "b9512 prebuilt" },
  "previous": { "label": "b9479 prebuilt" },
  "previous_available": true,
  "transaction": { "state": "clean" },
  "last": { "id": "...", "op": "install", "label": "b9512 prebuilt", "outcome": { "kind": "installed" } },
  "custom_server_path": null
}
```

`transaction.state` is `clean`, `pending` (a journal exists), or `blocked` (the journal is
unreadable; nothing is modified until it is resolved, and `reason` explains why).

### `POST /api/llama/restart`

Restarts the running local llama-server using its saved configuration and the current
llama-server binary. Commonly used after an in-place update to pick up the new binary.

- If no local server is running, returns an error.
- Brief pause is inserted between stop and start to ensure the previous process has fully exited.

## Update pipeline

The update handler downloads and prepares a complete candidate set while the running model
keeps serving, then hands it to the journaled promotion engine in `src/llama_runtime/promote.rs`.
Nothing is stopped, renamed, or overwritten until the candidate exists in full.

```
1. Fetch the release list; select assets for (backend, arch, os)         [no lock]
2. Download + extract into a temp dir                                    [no lock]
3. Locate llama-server inside the extracted tree                         [no lock]
4. Flatten the files into a sibling staging dir `.bin.staging-<id>`      [no lock]
5. Take the runtime gate (409 {owner} if another mutation is running)
6. promote():
   a. prepare the staged set (exec bits, strip macOS quarantine), write the manifest
   b. write the journal (steps, fingerprints), phase Begun; the swap window opens
   c. if a llama.cpp model is running: journal the restart intent, then stop it
      (a loaded model would contend for the GPU with the probe)
   d. verify the staged set (one probe, see "Verification")
   e. phase Swapping, renames in order: previous -> old-previous, live -> old-live, staged -> live
      (rollback: live -> old-live, previous -> live)
   f. verify the live set, then phase Committed
   g. retention: the old live set becomes `bin-previous`, the old previous set is deleted
   h. restart the model with its previous config (inside the owner scope)
   i. finalize: the journal is archived as runtime-transaction.last.json
```

Download, extraction and staging never hold the gate, so a slow download cannot block a
rollback or a model launch. Cancellation is honored only before the journal is written.

### Layout

```
<root>/bin/                          live set
<root>/bin-previous/                 exactly one previous set
<root>/.bin.staging-<id>/            candidate; swept at startup if orphaned
<root>/runtime-transaction.json      the in-flight journal (absent when idle)
<root>/runtime-transaction.last.json last finished transaction (status surface)
```

All of these are siblings on one volume, so every swap is a rename. A candidate is never
copied into a live directory.

### Crash recovery

The journal records the full step list and a phase before the first mutation. A rename's
completion is derived from "its source is gone", so recovery never needs a per-rename write
and can be replayed any number of times:

- **Begun / Swapping (uncommitted):** undo the steps in reverse, restoring the original live
  and previous sets; the rejected candidate is discarded afterwards.
- **Committed:** finish retention (the installed set is already durable).
- **Corrupt or ambiguous journal:** `Blocked`. No directory is touched, every new mutation
  is refused with 409, and startup logs a warning with the reason.

Recovery runs at startup, synchronously, before pollers and the web server start, for the
app root and the canonical and legacy default roots. An interrupted model stop is reported
(`the model ... was not restarted; start it again`) rather than automatically relaunched,
because the journal deliberately stores no launch credentials or configuration.

Set identity in the journal is a metadata fingerprint (relative path, size, mtime), not a
content hash, so it follows a directory rename without hashing gigabytes at each boundary.

### Verification

Every install kind (prebuilt, rollback, later source builds) uses the single path in
`src/llama_runtime/verify.rs`: prepare the directory, run one bounded probe with a sanitized
environment (30 s, because a cold macOS Metal start can exceed 10 s), then evaluate the probe
against the intent (expected backend, pinned commit, flash-attention evidence). The manifest
is provenance only; a hash mismatch marks provenance `custom` and never changes what the
probe reports.

### Gatekeeper quarantine strip (macOS)

The entire extracted archive (the executable, co-located dylibs, and Metal shaders) can carry
the `com.apple.quarantine` attribute. Gatekeeper enforces it at `dlopen()` time, so stripping
only the executable is insufficient: the process launches, loads a quarantined dylib, and is
killed by a signal (no exit code). `prepare_runtime_dir` strips the attribute recursively from
the whole staged set before it is executed. `start_server` applies the same best-effort strip
to the binary's parent directory before its own `--help` check, which covers manually placed
binaries.

### Ownership and lock order

One `RuntimeGate` serializes every runtime mutation (update, rollback, uninstall, later
builds). Acquisition order is: toolchain, checkout/build/cache, **runtime gate**,
`server_lifecycle`, session locks. Promotion takes the gate first and then calls
`stop_server`/`start_server`, which take `server_lifecycle`. Launch paths never await the
gate: while a swap is in flight a llama.cpp launch is refused immediately (the promotion's own
restart passes via the owner token), and capability snapshot generation returns an error
instead of executing or hashing a half-swapped directory. The cached snapshot stays valid.

## Model-file download stability

Model-file downloads (Hugging Face, etc.) are managed in `src/model_download.rs` with three
safety guards to reduce conflicts, wasted bandwidth, and runaway concurrency.

### Duplicate guard

A single in-memory `HashMap` keyed by `(repo_id, file_path)` tracks active download tasks.
If the API receives a second request for the same file while an existing task is running,
it returns a 409 error with the message:

> Already downloading: {name}. Please wait until it completes.

### Existing-file guard

Before starting a download, the handler checks whether the target file already exists at
its final path with a reasonable size. If so, it returns a 409 error:

> File already exists at: {path}. It may be available in your library.

This prevents re-downloading large models that are already on disk.

### Concurrency limit

A hard limit of 2 simultaneous running downloads is enforced. A third request is rejected
immediately with:

> Too many downloads in progress. Please wait for one to finish.

The frontend (hf-browse.js, models.js) listens for these responses and shows toast messages
instead of silently queuing extra tasks.

## Server stop/restart around updates

When a llama.cpp model is running at update time:

1. The running `ServerConfig` is captured when the host is created, before download.
2. Download, extraction and staging run while the model keeps serving.
3. Once the journal exists, the restart intent is journaled and then `stop_server` is called, before the staged probe (a loaded model would block or time out a second `llama-server` on Metal/CUDA).
4. After the swap commits and the live set verifies, the model is restarted with the preserved config.
5. If verification or the swap fails, recovery restores the previous directories and the model is restarted on them (`rolled_back`).
6. If only the restart fails, the new runtime stays installed (`installed_restart_failed`); the user can retry or roll back.

Rapid-MLX models and attached sessions are never stopped by a llama.cpp update.
Uninstall (`DELETE /api/llama-binary/uninstall`) uses the same gate and journal and still
returns 409 while a server is running.

## Windows self-update

On Windows, a running executable cannot overwrite itself. When `llama-monitor.exe` installs an update to itself (the `install` module in `src/agent.rs`), it writes a small `.bat` helper and launches it with the `DETACHED_PROCESS` creation flag. The batch file waits for the parent process PID to exit, then copies the new binary over the old path and starts the updated executable. This is distinct from the llama-server binary update described above, which uses the journaled rename-based promotion above and runs entirely within the Rust process.

---

## Expected quality bar for future backends

The Rapid-MLX integration must provide the same set of capabilities through its own
routes under `/api/inference/backends/rapid_mlx/`:

- Installed version display (equivalent of `/api/llama-binary/version`)
- Latest available version with release notes (equivalent of `/api/llama-binary/latest`)
- Historical release list with rollback support (equivalent of `/api/llama-binary/releases`)
- Platform availability and recommended configuration (equivalent of `/api/llama-binary/platform-info`)
- Atomic install/upgrade with a pre-activation health check gate (equivalent of the update pipeline above)
- Cleanup of superseded artifacts after install

The health check gate in particular is non-negotiable: no backend should ever overwrite
a working runtime with an unverified one. The macOS quarantine strip is a llama.cpp
specific detail (binary download); Rapid-MLX installs through pip/Homebrew and will
have its own equivalent first-run probe concerns.
