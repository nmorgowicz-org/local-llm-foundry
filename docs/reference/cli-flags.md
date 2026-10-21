# CLI Flags Reference

Use `local-llm-foundry --help` to see the current flag list from the binary.
The legacy `llama-monitor` alias accepts the same flags through 2.x.

## Core dashboard flags

### `--host` (default: `127.0.0.1`)

Bind address for the dashboard web UI.

Use `0.0.0.0` only when you intentionally want LAN access.

```bash
llama-monitor --host 127.0.0.1
llama-monitor --host 0.0.0.0
```

### `--basic-auth`

Enable HTTP Basic Auth for the dashboard in `user:password` form.

This keeps browser-native Basic Auth in front of the protected dashboard routes.
It is most useful when the dashboard is reachable by other machines, usually alongside `--host 0.0.0.0`.

On first run with the newer auth system, compatible CLI auth settings are also migrated into
`auth-config.json` so users can later manage dashboard access from the Security tab instead of
retyping credentials by hand.

```bash
llama-monitor --host 0.0.0.0 --basic-auth admin:secret123
```

### `--form-auth`

Enable an in-app sign-in screen for the dashboard in `user:password` form.

When set, unauthenticated users can still load the shell page, but protected API, WebSocket,
and control routes stay locked until a valid form-auth session is created.

Like `--basic-auth`, compatible credentials are migrated into `auth-config.json` on first run
so the instance can cut over to config-managed dashboard auth once startup flags are removed.

```bash
llama-monitor --host 0.0.0.0 --form-auth admin:secret123
```

### Choosing An Auth Mode

- Default: no auth. This matches local-only use on `127.0.0.1`.
- `--basic-auth`: browser-level challenge before protected routes are usable.
- `--form-auth`: modern in-app login screen backed by an HttpOnly session cookie.
- Both flags may be configured together; protected routes accept either valid Basic credentials or a valid form session.
- When no startup flags are present, dashboard auth can be managed from `Settings → Security`
  and is persisted in `auth-config.json` with a hashed password.

### `--clear-auth-config`

Delete the persisted dashboard auth config and exit.

Use this for local password recovery when a user is locked out of the in-app form or Basic Auth
managed through `auth-config.json`.

```bash
llama-monitor --clear-auth-config
```

After clearing the file, restart llama-monitor and set a new password from
`Settings → Security → Dashboard Access`.

See also: [Security Reference](security.md).

### `--port`, `-p` (default: `7778`)

Port for the dashboard web UI.

```bash
llama-monitor --port 8000
llama-monitor -p 9999
```

### `--headless`

Run without tray or desktop UI integration.

```bash
llama-monitor --headless
```

### `--no-tray`

Disable the tray icon even on systems where one would normally be shown.

```bash
llama-monitor --no-tray
```

## Inference backend launch defaults

These flags affect how Llama Monitor discovers and launches local inference
backends (llama.cpp, Rapid-MLX, etc.).

### `--llama-server-path`, `-s`

Path to the `llama-server` binary for the llama.cpp backend.

If you are only using the Rapid-MLX backend on Apple Silicon, this flag is
optional and can be omitted.

```bash
llama-monitor --llama-server-path /usr/local/bin/llama-server
llama-monitor -s /opt/llama.cpp/llama-server
```

### `--llama-server-cwd`

Optional working directory for `llama-server` (llama.cpp backend).

Use this only when the launched process depends on relative paths.

```bash
llama-monitor --llama-server-cwd /srv/llama
```

### `--models-dir`, `-m`

Directory used for model auto-discovery across all backends.

```bash
llama-monitor --models-dir /srv/models
```

## Build llama.cpp from source

`local-llm-foundry llama-build` compiles `llama-server` and `llama-bench` into the
selected application's managed runtime. Stop the foundry, agents, and models
using that root first. A cross-process lease prevents overlap with other
builders and running foundry/agent processes that support this lease.

```bash
local-llm-foundry llama-build --dry-run --config-dir /custom/config/path
local-llm-foundry llama-build --accelerator metal --ref latest --headroom
local-llm-foundry llama-build --accelerator cuda --ref master --jobs 8
```

| Option | Meaning |
| --- | --- |
| `--dry-run` | Read-only prerequisite checks, resolved tag/commit, and lossless JSON CMake argv. Does not compile, install, or create application directories. |
| `--config-dir PATH` | Select the managed root; build settings never fall back to another root. |
| `--accelerator metal\|cuda\|cpu` | Defaults to Metal on macOS, CUDA on Windows, CPU on Linux. Linux builds are untested. |
| `--ref latest\|master` | Latest numeric `bNNNN` release tag (default), or upstream `master`. Execution resolves once and pins the exact commit; a preview does not reserve it. |
| `--jobs N` | Explicit parallelism, bounded to 1–4096 and at most twice the logical CPU count. Default: logical CPUs minus one, at least one. |
| `--headroom` | Half the logical CPUs and lower build priority; conflicts with `--jobs`. |
| `--wipe` | Reset only the selected managed build tree before configuring. Never deletes live or previous runtimes. |
| `--repair-processes` | Clear `llama-build/processes.json` after proving no recorded process is alive and no other process references the build directory. Never kills anything; excludes the other build flags. Exit `0` cleared (or nothing to clear), `1` refused (live processes are listed), `75` busy. |

CUDA builds always compile all supported Flash Attention K/V pairs
(`GGML_CUDA_FA_QUANTS=all`). Prerequisite validation includes a compile smoke
test; dry-run skips this test. Actual installation uses staged loader/backend
verification and journaled promotion. A successful installation retains at most
one previous runtime in `bin-previous/`. Compilation failure leaves installed
binaries unchanged.

`llama-build.json` in the selected root accepts `ccache_max_bytes` (default
2 GiB, allowed 256 MiB–16 GiB). Build trees, checkout, and the capped cache live
under `llama-build/`; the last three completed log/JSON pairs are retained under
`logs/llama-build/`, with active/recovery-owned diagnostics exempt.

Ctrl-C requests cancellation before the transaction boundary. Once installation
has begun journaling, recovery must finish. Unresolved or corrupt
`llama-build/processes.json` blocks new builds and cache cleanup; stop any
surviving descendants, then run `llama-build --repair-processes`. Do not delete
it by hand to bypass the safety check.

Exit codes: `0` installed (or ready dry-run), `1` failed/not ready, `2` installed
but model restart failed (also used for CLI usage errors), `75` busy or repair
required, `130` cancelled. The standalone CLI does not restart models.

## Persisted file locations

### `--presets-file`

Override the presets JSON path.

```bash
llama-monitor --presets-file ~/.config/llama-monitor/presets-custom.json
```

### `--sessions-file`

Override the sessions JSON path.

```bash
llama-monitor --sessions-file ~/.config/llama-monitor/sessions-custom.json
```

### `--config-dir`

Override the root configuration directory. This affects persisted state such as sessions, presets, and UI settings.

```bash
llama-monitor --config-dir /custom/config/path
```

## GPU monitoring flags

### `--gpu-backend` (default: `auto`)

Select the GPU telemetry backend.

Valid values:

- `auto`
- `rocm`
- `nvidia`
- `none`

```bash
llama-monitor --gpu-backend auto
llama-monitor --gpu-backend rocm
llama-monitor --gpu-backend nvidia
llama-monitor --gpu-backend none
```

### `--gpu-arch`

ROCm architecture override used for local GPU environment setup, for example `gfx906`, `gfx1100`, or `auto`.

```bash
llama-monitor --gpu-arch gfx1100
llama-monitor --gpu-arch auto
```

### `--gpu-devices`

Comma-separated local GPU device indices.

```bash
llama-monitor --gpu-devices 0,1
```

## Polling

### `--llama-poll-interval` (default: `1`)

Polling interval in seconds for backend metrics collection (llama.cpp,
Rapid-MLX, and others).

```bash
llama-monitor --llama-poll-interval 5
```

## Remote agent mode

These flags run `llama-monitor` as the lightweight remote metrics agent instead of the full dashboard.

### `--agent`

Start in remote-agent mode.

```bash
llama-monitor --agent
```

### `--agent-host` (default: `127.0.0.1`)

Bind address for the agent HTTP server.

```bash
llama-monitor --agent --agent-host 0.0.0.0
```

### `--agent-port` (default: `7779`)

Port for the agent HTTP server.

```bash
llama-monitor --agent --agent-port 7779
```

### `--agent-token`

Optional bearer token required by the agent.

```bash
llama-monitor --agent --agent-token "your-secret-token"
```

## Remote-agent integration in dashboard mode

These flags apply when you are running the full dashboard and want it to talk to or manage a remote agent.

### `--remote-agent-url`

Explicit remote-agent URL used by the dashboard for polling.

```bash
llama-monitor --remote-agent-url http://192.168.1.100:7779
```

### `--remote-agent-token`

Optional bearer token used when polling the remote agent.

```bash
llama-monitor --remote-agent-token "your-secret-token"
```

### `--remote-agent-ssh-autostart`

Allow one SSH start attempt after attaching to a remote endpoint if the agent is unreachable.

```bash
llama-monitor --remote-agent-ssh-autostart
```

### `--remote-agent-ssh-target`

Saved SSH target for remote-agent management.

```bash
llama-monitor --remote-agent-ssh-target user@192.168.1.100
```

### `--remote-agent-ssh-command`

Optional override for the remote command used to start the agent over SSH.

```bash
llama-monitor --remote-agent-ssh-command "systemctl --user start llama-monitor-agent"
```

## Supported examples

### Dashboard on the local machine (llama.cpp)

```bash
llama-monitor --llama-server-path /usr/local/bin/llama-server --models-dir ~/models
```

### Dashboard on Apple Silicon with Rapid-MLX

For Rapid-MLX, no backend-specific server path is required; use:

```bash
llama-monitor --models-dir ~/models
```

### Headless dashboard on a LAN host

```bash
llama-monitor --headless --host 0.0.0.0 --basic-auth admin:secret123
```

### Dedicated remote agent

```bash
llama-monitor --agent --agent-host 0.0.0.0 --agent-port 7779 --agent-token "shared-secret"
```

### ROCm-focused local setup

```bash
llama-monitor --gpu-backend rocm --gpu-arch gfx1100 --gpu-devices 0,1
```
