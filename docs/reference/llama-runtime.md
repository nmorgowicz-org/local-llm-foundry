# llama.cpp runtime: builds, prerequisites and K/V support

The foundry manages one llama.cpp runtime (the `bin/` directory next to its
configuration) and keeps at most one previous set for rollback. You can fill it
three ways: install an upstream **prebuilt** release, **build from source**, or
point the foundry at your own `llama-server` (a *custom* runtime, which the
foundry never modifies). Every change to the managed runtime goes through the
same journaled promotion, so a failed or interrupted update leaves the last
working runtime in place.

The pill in the top bar shows the live runtime, for example
`llama.cpp · b11740 · src`. The suffix is the provenance: `src` (built here),
`prebuilt` (upstream release) or `custom`. Click it for the version list, or open
**llama.cpp Runtime…** for the full panel.

## Why build from source

- **CUDA flash-attention K/V pairs.** Upstream CUDA builds compile only the
  common pairs. A CUDA source build always passes `-DGGML_CUDA_FA_QUANTS=all`,
  which compiles every K/V pair, so asymmetric caches such as K `q8_0` / V `q4_0` use
  the fused kernel instead of a slower conversion path.
- **Newest upstream.** Build the latest release tag, or `master` for the very
  newest code.

Metal and CPU builds are supported too, but a rebuild does not add anything for
mixed K/V pairs there (see the table below).

## The runtime panel

| Section | What it does |
| --- | --- |
| Error banner | Shown after a failed or interrupted build: phase, exit code, first error line, last log lines, a fix button for known problems, **Copy diagnostics**, **Dismiss**. |
| Current | Live runtime, provenance, whether the installed files still match the recorded hashes, and **Roll back** when a previous set exists. |
| Prerequisites | Each check with its status, a fix (a command to copy, a link, or on macOS a one-click install), and **Details** with the exact commands run. **Validate** runs everything, including a real compile of a one-file project. |
| Build | Source (latest release tag or `master`), jobs, leave-headroom, wipe build directory, and on CUDA the architecture. **Build & install** is disabled while any prerequisite fails; the tooltip names the failing checks. Warnings do not disable it. |
| Log | The live build log with phase progress. |
| Disk | Source, build trees, compiler cache and the previous set, with **Clean build cache**. |

There is no build-time estimate: the panel shows live progress.

A build pins one exact upstream commit when it starts. If `master` moves while
you build, you still get the commit that was resolved. Cancel works until
installation begins; after that the update completes or rolls back on its own.

If the update installs but the model fails to restart, the pill turns amber
("Runtime installed; model restart failed"). The runtime is valid; restart the
model yourself or roll back from the panel.

## Prerequisites

Nothing is installed without your click, and on Windows nothing is installed for
you at all.

### macOS

Xcode Command Line Tools, Git, CMake (3.21+), Ninja and a Metal-capable GPU.
`ccache` is optional and makes rebuilds faster. If Homebrew is present, the panel
offers a one-click `brew install` limited to `git`, `cmake`, `ninja` and
`ccache`.

### Windows (CUDA)

Visual Studio 2022 Build Tools with the C++ workload, the CUDA Toolkit, CMake,
Ninja and Git. The panel detects them, shows the exact version it found, and
checks that your NVIDIA driver meets the minimum for the installed toolkit.
Fixes are shown as commands to copy or links; you run them yourself. Long path
support should be enabled. Plan for roughly 20 GB of free space for the first
CUDA build (a provisional figure, not a measured peak).

### Linux

The build code exists but has not been verified on Linux.

### Network

A build downloads source from GitHub and web UI assets from Hugging Face.
Offline builds are not supported; the prerequisite check tests both hosts.

## Command line

```sh
local-llm-foundry llama-build --dry-run
local-llm-foundry llama-build --accelerator metal --ref latest
```

The CLI builds into the same managed runtime. It requires the foundry and its
models to be stopped. See [cli-flags.md](cli-flags.md#build-llamacpp-from-source)
for every flag and the exit codes.

## K/V cache support

The K/V cache type pair decides which attention kernel runs. The foundry no
longer blocks a mixed pair on its own. It states what it expects and only blocks
where llama.cpp itself refuses to start.

| State | Meaning | Where you see it |
| --- | --- | --- |
| Fused attention | A fused kernel for this pair is compiled in and expected to run. | Quiet badge. |
| Slower path | The pair works, but through a conversion or CPU attention path. | Amber line with the reason. |
| Not verified | The runtime or model could not be probed. Launch is never blocked for this. | Neutral line. |
| Won't start | llama.cpp refuses this combination (a quantized V cache with flash attention off). | Red line with the fix; Start is disabled, Save is not. |

By backend:

- **CUDA.** A pair missing from the compiled list uses a slower conversion
  fallback. When that is confirmed, the panel suggests a source build, which
  compiles every pair. An unknown pair list never produces a rebuild suggestion.
- **Metal.** The fused kernels cover equal K/V pairs. From llama.cpp b11719 a
  mixed pair runs attention on the CPU. A rebuild does not change this, so none
  is suggested.
- **CPU.** No fused-kernel concept applies; no rebuild is suggested.

Compiled kernels are not proof of execution: "compiled" and "expected to run
fused" are reported separately, and a missing or malformed probe is shown as
not verified rather than guessed.

Model-family guidance (for example for Gemma) appears as an info line under the
state. It never changes the state and never blocks a launch.

## Troubleshooting

Known failures show a hint with a fix button in the error banner. Unknown
failures still show the full log tail and the log file path.

| Symptom | What to do |
| --- | --- |
| Stale CMake cache or a configure error mentioning a path that no longer exists | **Wipe build dir & retry**. The foundry heals most stale caches automatically first. |
| `unsupported Microsoft Visual Studio version` | Install the MSVC toolset your CUDA version supports. |
| `No CUDA toolset found` | Re-run **Validate**; set `CUDA_PATH`. |
| `Unsupported gpu architecture` | Pick a lower CUDA architecture or update CUDA. |
| `fatal: unable to access` | Network or proxy problem; the failing URL is shown. |
| Out of disk space | **Clean build cache**. The live runtime is never removed to make room. |
| Compiler killed / out of memory | Rebuild with **leave headroom** (fewer jobs). |
| Hugging Face or GitHub download failed during the build | Both hosts must be reachable while building. |
| `xcrun: error: invalid active developer path` | Run `xcode-select --install`. |

### Unresolved process records

If the foundry or the CLI is killed hard (a crash, power loss, `kill -9`) during
a build, it may leave `llama-build/processes.json`. While it exists, builds and
cleaning refuse to run and startup skips orphan cleanup, so nothing deletes
files a leftover compiler may still be writing. An interrupted runtime update is
still recovered at startup. On Windows a Job Object normally kills the tools
when the app dies; on macOS the next start tries to prove which processes are
yours, and when it cannot, it stops and says so.

To clear it, first stop any leftover `cmake`, `ninja` or compiler processes
yourself. Then use **Clear record…** in the panel's red banner, or:

```sh
local-llm-foundry llama-build --repair-processes
```

(`POST /api/llama-build/repair-processes` with `{"confirm": true}` is the same
action.) It refuses, changing nothing, while a recorded process is still alive or
any other process still references the build directory, and lists them. It never
kills anything. Do not delete the file by hand.

### Doctor

**Doctor** reports, without changing anything: an interrupted runtime update
that still needs recovery, an unusable journal, unresolved build process
records, installed files that no longer match their recorded hashes, a missing
capability probe, old rollback backup folders left by an earlier updater (never
deleted automatically), and a failed or more-than-30-day-old prerequisite check.

## Rollback

**Roll back** in the panel swaps the live and previous sets through the same
journaled path. The foundry keeps the live runtime plus at most one previous
set. Rolling back twice returns to where you started.

## Where things live

Under the configuration root: `bin/` and `bin-previous/` (runtime sets),
`llama-build/` (checkout, per-accelerator build trees, compiler cache) and
`logs/llama-build/` (job records and logs). Each runtime carries `llama-build-manifest.json` with the exact
commit, build arguments and file hashes; this is provenance, not a promise of
what the runtime can do. The last three completed build logs are kept.

See [api.md](api.md) for the HTTP routes behind the panel.
