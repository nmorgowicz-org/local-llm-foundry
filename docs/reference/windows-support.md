# Local LLM Foundry Windows Support

The 2.0 bridge publishes canonical and legacy archive aliases for the same
target build. Native scheduled-task, tray/WebView2, and mixed-version checks
remain final Windows-machine qualification markers.

This is the canonical reference for how Local LLM Foundry behaves on Windows.
It supersedes scattered comments in prior plans. For sensor-bridge
implementation detail see `windows-sensor-bridge-implementation.md`.

---

## Shipped bundle

The Windows release zip (`llama-monitor-windows-x86_64.zip`) contains three files that must stay in the same directory:

| File | Role |
|------|------|
| `llama-monitor.exe` | Main application (tray icon + WebView2 popover, dashboard, API server) |
| `sensor_bridge.exe` | C# sidecar for CPU temperature via LibreHardwareMonitor |
| `WebView2Loader.dll` | Loader for the WebView2 tray popover (see below) |

`src/lhm.rs` locates `sensor_bridge.exe` relative to the running executable, so they must be co-located.

**Why `WebView2Loader.dll` ships separately.** The Windows binary is built with the **GNU**
toolchain (`x86_64-pc-windows-gnu`, see `scripts/build-single-target.sh`) with the
`native-tray,webview-popover` features. On the GNU target, `webview2-com-sys` links
`WebView2Loader.dll` *dynamically* (it only static-links `WebView2LoaderStatic.lib` when
`target_env = "msvc"`). The WebView2 *runtime* does not place this loader on the system path, so
it must ship next to the exe — without it the app fails to launch with `STATUS_DLL_NOT_FOUND`
(`0xC0000135`). `release.yml` copies it from the build output into the zip (and fails the
release if it is missing). Moving to an MSVC build would static-link the loader and reduce this
to two files — see `docs/plans/20260622-windows_build_toolchain.md`.

---

## GUI subsystem and console attachment

`llama-monitor.exe` is linked as a **Windows GUI subsystem** binary:

```rust
#![cfg_attr(windows, windows_subsystem = "windows")]
```

This means double-clicking the executable in Explorer launches the tray icon with no console window. However, CLI invocations (`--version`, `--headless`, `--agent`) still print to the terminal because the first line of `main()` calls `attach_parent_console()`, which calls kernel32 `AttachConsole(ATTACH_PARENT_PROCESS)`. When there is no parent console (Explorer launch), `AttachConsole` fails silently and the app proceeds as a pure GUI process. When launched from PowerShell or cmd.exe, the parent console is re-attached and output appears normally.

Because Explorer-launched instances have no console, errors and diagnostics from the Rust side go to the logger (not stdout). A future `--log-file` flag is planned (P2 item) for collecting support logs in this mode.

---

## Subprocess window policy

Every short-lived helper process spawned on Windows goes through `crate::platform::no_window` (for `std::process::Command`) or `crate::platform::no_window_tokio` (for `tokio::process::Command`), defined in `src/platform.rs`. These helpers set the `CREATE_NO_WINDOW` creation flag (`0x0800_0000`) on Windows and are no-ops on other platforms.

Processes routed through these helpers:

- `nvidia-smi` and `rocm-smi` (GPU polling, runs every ~500 ms)
- `where` / `which` command-existence checks at startup
- `llama-server.exe` (model server spawn)
- `taskkill` (model server stop)
- `icacls` (file-permission hardening at startup)
- `schtasks` (sensor-bridge status checks)
- `winget` (WebView2 and PawnIO auto-install)

**Intentional exceptions:** the two `powershell -Verb RunAs` calls in `src/lhm.rs` that trigger UAC elevation prompts. Those are left without `CREATE_NO_WINDOW` by design — hiding the window would suppress the UAC dialog.

---

## Config and data directory

| Platform | Config directory |
|----------|-----------------|
| Windows  | `%APPDATA%\local-llm-foundry` (legacy `%APPDATA%\llama-monitor` remains active until explicit migration) |
| macOS    | `~/.config/local-llm-foundry` (legacy root remains active until explicit migration) |
| Linux    | `~/.config/local-llm-foundry` (legacy root remains active until explicit migration) |

The application never performs a best-effort path move. It classifies canonical,
legacy, empty, and conflicting roots first; old-only installs continue from the
legacy root. An authenticated preview and explicit confirmation queue a
copy-first maintenance migration. The next controlled restart verifies a
journaled receipt before selecting the canonical root. Rollback and legacy-root
cleanup are separate authenticated, receipt-scoped operations. The Windows
target-aware compile check runs on macOS; native `%APPDATA%`, scheduled-task,
tray, and installer behavior is qualified on Windows during the final release
matrix.

The migration notification action is restored on every authenticated startup,
including when the notification was persisted by an earlier session. Until
that controlled restart completes, local binary updates intentionally target
the active legacy `bin` directory; afterward the canonical
`local-llm-foundry/bin` directory is selected.

Remote agent management applies the same compatibility rule: a retained legacy
root is preferred when it is the only managed install, even when that root
contains the rebranded executable name. A canonical root takes precedence once
the migration has completed.

---

## GPU data sources

| Metric | Source |
|--------|--------|
| GPU name | WMI `Win32_VideoController` |
| Total VRAM | Registry `HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-...}\<N>\HardwareInformation.qwMemorySize` (QWORD); falls back to `AdapterRAM` (UINT32, wraps at 4 GiB) only when the QWORD value is absent. Implemented in `src/gpu/wmi_gpu.rs` via the `winreg` crate. |
| GPU utilization | `nvidia-smi` (NVIDIA) or `rocm-smi` (AMD) |
| GPU temperature | `nvidia-smi` (NVIDIA) or `rocm-smi` (AMD) |
| CPU temperature | `sensor_bridge.exe` via LibreHardwareMonitorLib (see sensor-bridge doc) |

Intel and non-CLI GPU utilization and temperature are not currently available on Windows. NVML direct integration (replacing the `nvidia-smi` subprocess) is a planned improvement.

---

## WebView2 runtime

The tray popover requires the Microsoft Edge WebView2 runtime. WebView2 is present on Windows 11 and most updated Windows 10 installations, but may be absent on LTSC images or older systems.

If `src/tray.rs` `build_as_child` fails, the code heuristically detects a missing WebView2 runtime from the error message and:
1. Surfaces an actionable error message naming the missing runtime.
2. Attempts a best-effort silent install via winget (through `crate::platform::no_window`):
   ```
   winget install -e --id Microsoft.EdgeWebView2Runtime --silent
     --accept-package-agreements --accept-source-agreements --disable-interactivity
   ```

IPC bridging (`window.ipc.postMessage` → WebView2 `window.chrome.webview.postMessage`) still requires real-Windows verification; a `with_initialization_script` polyfill may be needed if resize/close messages do not arrive.

---

## Automated zero-touch setup model

CPU temperature on a clean Windows machine requires three things: a self-contained `sensor_bridge.exe` (no .NET on target), the PawnIO kernel driver, and the sensor-bridge scheduled task. The one-click install in **Settings → Sensor Bridge** collapses all of these into a single UAC prompt:

1. **.NET runtime** — eliminated. `sensor_bridge.exe` is self-contained (`net10.0`, `PublishSingleFile true`, `IncludeNativeLibrariesForSelfExtract true`). No runtime install needed.
2. **PawnIO driver** — installed idempotently inside the same elevated PowerShell session, guarded by `sc query PawnIO`:
   ```
   winget install -e --id namazso.PawnIO --silent
     --accept-package-agreements --accept-source-agreements --disable-interactivity
   ```
   A failed winget install is non-fatal; the task is still registered.
3. **Scheduled task** — registered as SYSTEM so the bridge survives reboots without a logged-in user. Already implemented in `src/lhm.rs::install_local_sensor_bridge`.

### Driver-detection signals

`lhm::is_pawnio_installed()` reports whether the PawnIO kernel driver service is present. The `/api/sensor-bridge/status` endpoint includes a `pawnio` boolean. The dashboard (`static/js/features/sensor-bridge.js`) uses this to show a "driver missing" message with a link to pawnio.eu when the bridge is running but PawnIO is absent, rather than silently displaying no temperature.

---

## Self-update (Windows-specific)

On Windows, a running executable cannot overwrite itself. The self-update path for `llama-monitor.exe` (implemented in the `install` module of `src/agent.rs`) uses a detached `.bat` helper launched with `DETACHED_PROCESS`: the batch file waits for the parent PID to exit, then copies the new binary over the old one and starts the updated executable. This is distinct from the llama-server binary update, which uses a rename-based approach managed by `src/web/api/llama_binary.rs`.

---

## CI note

CI runs `cargo clippy --target x86_64-pc-windows-gnu -- -D warnings` so Windows-only `#[cfg(windows)]` code paths are linted, not just host code. Cross-compilation uses the MinGW toolchain; see `docs/reference/cross-compilation.md`.

---

**Last updated:** 2026-06-22
