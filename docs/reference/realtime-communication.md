# Real-Time Communication

Llama Monitor uses a WebSocket connection to push live dashboard state from the server to the browser. Client-side network quality and browser-pressure detection can recommend or apply less aggressive dashboard cadences.

## WebSocket Connection

```text
ws://localhost:7778/ws
```

The server pushes messages on a configurable interval (default 500ms). The browser also sends lightweight `client-visibility` messages so the server can slow hidden or idle clients without changing the global setting. When no active session exists, the server waits silently until one is established before resuming pushes.

Sleep mode affects push behavior:

- **off** (default): full payload at normal interval.
- **logs-only**: reduced payload (no heavy metrics, no GPU/system), live logs included; slower interval.
- **sleep**: minimal heartbeat (no metrics, no logs); slowest interval (≥ 10s).

If chat generation is active, the backend preserves the normal interval to avoid stalling streaming.

### Polling Interval

Configurable from 200ms to 10s via the nav **Cadence** chip or **Settings > Performance**. The interval controls how often the server pushes dashboard updates. For local sessions, GPU telemetry is also kept from polling faster than the dashboard can display it; system metrics and remote-agent checks may still use their own cadence.

| Interval | Use Case |
| ---------- | ---------- |
| 200ms | Maximum responsiveness, higher CPU/network usage |
| 500ms | Default, balanced responsiveness and resource usage |
| 1000ms | Moderate updates, lower resource usage |
| 2000ms | Battery saver, lower browser and telemetry pressure |
| 5000ms | Low power or slow networks, minimal updates |

## Network Quality Detection

The browser's [Network Information API](https://developer.mozilla.org/en-US/docs/Web/API/NetworkInformation) is used to detect connection quality and suggest appropriate polling intervals.

### Auto-Detection Mapping

| Connection Type | Auto Interval | Label |
| ----------------- | --------------- | ------- |
| `slow-2g` | 5000ms | Very Slow (2G) |
| `2g` | 5000ms | Slow (2G) |
| `3g` | 2000ms | Moderate (3G) |
| `4g` | 500ms | Good (4G) |
| `saveData` enabled | 2000ms | Data Saver |
| RTT > 500ms | 5000ms | High Latency |
| RTT > 300ms | 2000ms | Moderate Latency |
| RTT > 100ms | 1000ms | Elevated Latency |
| Unknown / API unavailable | 500ms | Detected |

When a slow network is detected, the dashboard shows an action toast that can switch cadence to **Auto**. The suggestion is shown once per session.

## Browser Pressure Detection

The dashboard samples browser main-thread timer drift. If visible dashboard timers repeatedly run late while the cadence is faster than 2s, the UI shows a one-time recommendation to switch to **Battery Saver (2s)**. This is an inferred signal: browsers do not expose reliable cross-platform CPU load, but sustained timer drift is a useful indicator that dashboard rendering, GPU driver polling, or other local work is affecting responsiveness.

### Browser Support

The Network Information API is supported in Chromium-based browsers (Chrome, Edge, Brave). Firefox and Safari fall back to the default 500ms interval. The API is optional — the dashboard works normally without it.

---

## WebSocket Message Schema

### Top-Level Message Shape

```json
{
  "mode":                    "off" | "logs-only" | "sleep",
  "sleep_mode":              true | false,
  "sleep_mode_manual":       true | false,
  "llama":                   { ...LlamaMetrics },
  "gpu":                     { ...GpuMetrics },
  "system":                  { ...SystemMetrics } | null,
  "logs":                    ["string", ...],
  "server_running":          true | false,
  "local_server_running":    true | false,
  "session_mode":            "spawn" | "attach" | "",
  "active_session_id":       "session_1746...",
  "active_session_endpoint": "http://127.0.0.1:8001",
  "active_session_status":   "running" | "stopping" | "error" | "",
  "active_session_error":    "error message string or empty",
  "active_session_preset_id": "preset_name or null",
  "last_spawn_cmd":          "last spawn command string or empty",
  "local_metrics_available": true | false,
  "host_metrics_available":  true | false,
  "remote_agent_connected":  true | false,
  "remote_agent_health_reachable": true | false,
  "remote_agent_url":        "http://...",
  "remote_agent_version":    "x.y.z or empty",
  "remote_agent_protocol_version": 1,
  "remote_agent_update_available": true | false,
  "remote_agent_protocol_too_old": true | false,
  "capabilities":            { ...MetricsCapabilities },
  "endpoint_kind":           "Local" | "Remote" | "Unknown",
  "session_kind":            "spawn" | "attach" | "none",
  "availability": {
    "system":   "Available" | "RemoteEndpoint" | ...,
    "gpu":      "Available" | "BackendUnavailable" | ...,
    "cpu_temp": "Available" | "SensorUnavailable" | ...
  }
}
```

Payload is reduced depending on `mode`. In logs-only and sleep, some fields may still appear if the backend has partial or cached data; the rules below describe the intended behavior:

- **off**: full payload with `llama`, `gpu`, `system`, `logs`, and all telemetry fields.
- **logs-only**: reduced payload — heavy metrics (`llama`, `gpu`, `system`) are typically omitted, but fragments may appear; always includes `logs` plus session flags.
- **sleep**: minimal heartbeat — no `logs`, no metrics; only `mode`, `sleep_mode`, `sleep_mode_manual`, and session flags.

`system` is `null` when `host_metrics_available` is false.
`gpu` is an empty object (`{}`) when `host_metrics_available` is false.

### LlamaMetrics (`llama`)

Populated by polling the llama.cpp server's `/metrics`, `/slots`, and `/v1/models` endpoints.

| Field | Type | Description |
| ------- | ------ | ------------- |
| `prompt_tokens_per_sec` | `f64` | Live prompt throughput (0 when idle) |
| `generation_tokens_per_sec` | `f64` | Live generation throughput (0 when idle) |
| `last_prompt_tokens_per_sec` | `f64` | Retained peak prompt throughput |
| `last_generation_tokens_per_sec` | `f64` | Retained peak generation throughput |
| `last_prompt_throughput_unix_ms` | `u64` | Epoch ms when last prompt rate was recorded |
| `last_generation_throughput_unix_ms` | `u64` | Epoch ms when last generation rate was recorded |
| `prompt_throughput_active` | `bool` | True when prompt rate > 0 this tick |
| `generation_throughput_active` | `bool` | True when generation rate > 0 this tick |
| `throughput_source` | `string` | `"interval_delta"` (always) |
| `prompt_tokens_total` | `u64` | Lifetime prompt token counter from Prometheus |
| `generation_tokens_total` | `u64` | Lifetime generation token counter from Prometheus |
| `context_capacity_tokens` | `u64` | Total KV cache capacity (sum of `n_ctx` across slots) |
| `context_live_tokens` | `u64` | Tokens currently in KV cache; 0 when unavailable |
| `context_live_tokens_available` | `bool` | True when server exposes per-slot token counts |
| `context_live_tokens_source` | `string` | Slot field used: `n_tokens`, `n_past`, `n_ctx_used`, or `n_cache_tokens` |
| `context_high_water_tokens` | `u64` | Peak context usage seen (from Prometheus `n_tokens_max`) |
| `slots_idle` | `u32` | Idle slot count |
| `slots_processing` | `u32` | Active slot count |
| `active_task_id` | `u64 \| null` | Task ID on the currently active slot |
| `last_task_id` | `u64 \| null` | Most recently seen task ID |
| `slot_generation_tokens` | `u64` | Tokens decoded in the active generation |
| `slot_generation_remaining` | `u64` | Tokens remaining in output budget |
| `slot_generation_limit` | `u64` | Total output budget (`n_predict` / `max_tokens`) |
| `slot_generation_active` | `bool` | True while generation is in progress |
| `slot_generation_available` | `bool` | True when generation progress data is exposed |
| `requests_processing` | `u32` | In-flight requests (from Prometheus) |
| `status` | `string` | Currently unused; reserved |
| `model_name` | `string` | Model ID from `/v1/models` |
| `model_params` | `u64 \| null` | Parameter count from model metadata |
| `model_ctx_train` | `u64 \| null` | Training context length from model metadata |
| `slots` | `SlotSnapshot[]` | Per-slot detail (see below) |

#### SlotSnapshot

One entry per slot returned by `/slots`.

| Field | Type | Description |
| ------- | ------ | ------------- |
| `id` | `u64 \| null` | Slot index |
| `n_ctx` | `u64` | Context window size for this slot |
| `is_processing` | `bool` | Whether the slot is currently active |
| `id_task` | `u64 \| null` | Task ID assigned to this slot |
| `output_tokens` | `u64` | Tokens generated so far |
| `output_remaining` | `u64` | Tokens remaining in budget |
| `output_limit` | `u64` | Total output budget |
| `output_active` | `bool` | Generation in progress on this slot |
| `output_available` | `bool` | True when `next_token` progress data present |
| `context_live_tokens` | `u64 \| null` | Current KV cache tokens for this slot |
| `context_live_tokens_source` | `string \| null` | Source field name |
| `speculative_enabled` | `bool` | Whether speculative decoding is active |
| `speculative_type` | `string \| null` | e.g. `"ngram_map_k"` |
| `speculative_config` | `{label, value}[]` | Speculative decoding parameters |
| `sampler_stack` | `string[]` | Active sampler names in order |
| `sampler_config` | `{label, value}[]` | Key sampler parameter values |

### GpuMetrics (`gpu`)

Empty object when `host_metrics_available` is false.

| Field | Type | Description |
| ------- | ------ | ------------- |
| `temp` | `f32` | GPU temperature °C |
| `load` | `u32` | GPU utilization % |
| `power_consumption` | `f32` | Current power draw (W) |
| `power_limit` | `u32` | TDP / power limit (W) |
| `vram_used` | `u64` | VRAM used (bytes) |
| `vram_total` | `u64` | VRAM total (bytes) |
| `sclk_mhz` | `u32` | Shader clock MHz |
| `mclk_mhz` | `u32` | Memory clock MHz |

`gpu` is a `BTreeMap<String, GpuMetrics>` keyed by device name (e.g. `"Apple M3 Max"`). Most setups have a single key.

On Apple Silicon, `power_consumption` is derived from `gpu_power` (dedicated GPU sensor) rather than from a generic SoC `total_power`.

### SystemMetrics (`system`)

`null` when `host_metrics_available` is false.

| Field | Type | Description |
| ------- | ------ | ------------- |
| `cpu_name` | `string` | CPU model string |
| `cpu_temp` | `f32` | CPU temperature °C |
| `cpu_temp_available` | `bool` | False on platforms without sensor access |
| `cpu_load` | `u32` | CPU utilization % |
| `cpu_load_available` | `bool?` | `false` while no real CPU sample exists yet (`cpu_load` is a placeholder 0); absent from older senders |
| `cpu_clock_mhz` | `u32` | Current CPU clock MHz |
| `ram_total_gb` | `f64` | Total RAM (GB) |
| `ram_used_gb` | `f64` | RAM in use (GB) |
| `ram_available_gb` | `f64` | RAM reported as available (GB) |
| `memory_pressure_level` | `string` | macOS vm_stat-derived level: `"ok"`, `"warning"`, or `"critical"` |
| `memory_free_gb` | `f64` | Free memory from macOS vm_stat (GB) |
| `memory_compressor_gb` | `f64` | Memory occupied by macOS compressor (GB) |
| `memory_compressed_gb` | `f64` | Pages stored in macOS compressor (GB) |
| `swapins` | `u64` | Cumulative swap-in page count (macOS) |
| `swapouts` | `u64` | Cumulative swap-out page count (macOS) |
| `motherboard` | `string` | Board name (empty string when unavailable) |

### Availability Reasons

Used in `availability.system`, `availability.gpu`, and `availability.cpu_temp`.

| Value | Meaning |
| ------- | --------- |
| `"Available"` | Metric is live and working |
| `"RemoteEndpoint"` | No host metrics over remote connection without agent |
| `"NoDisplay"` | No graphical session (headless system) |
| `"TrayUnavailable"` | Tray not supported on this build |
| `"SensorUnavailable"` | Hardware sensor not present |
| `"BackendUnavailable"` | GPU backend not detected |
| `"CommandMissing"` | Required system utility not installed |
| `"PermissionDenied"` | Insufficient OS permissions |
| `"MetricsUnreachable"` | Metrics endpoint not responding |
| `"NotApplicable"` | Metric does not apply in this configuration |

**Note:** Enum values are PascalCase strings (e.g., `"Available"` not `"available"`).

### Notes

- **Push interval:** Configurable via Settings > Performance (200ms to 10s). Default 500ms.
- **Sleep modes:**
  - `mode: "off"` — full telemetry; all fields present.
  - `mode: "logs-only"` — reduced payload; heavy metrics (`llama`, `gpu`, `system`) are typically omitted, but fragments may appear if the backend has partial data; `logs` are included along with session flags.
  - `mode: "sleep"` — minimal heartbeat; no logs, no metrics. Fields included:
    - `mode`, `sleep_mode`, `sleep_mode_manual`
    - Session flags: `active_session_id`, `active_session_endpoint`, `active_session_status`, `active_session_error`, `active_session_preset_id`, `session_kind`, `session_mode`, `server_running`, `local_server_running`, `remote_agent_connected`
  - During active chat generation, the backend preserves the normal interval and full payload regardless of low-power mode.
- **Host metrics gating:** `gpu` and `system` are only populated when `host_metrics_available` is `true`. This is true for local spawn/attach sessions and remote sessions where the remote agent is connected.
- **Memory pressure (macOS):** Fields `memory_pressure_level`, `memory_free_gb`, `memory_compressor_gb`, `memory_compressed_gb`, `swapins`, `swapouts` are populated on macOS via `vm_stat`. On non-macOS platforms they default to zero/empty and `memory_pressure_level` is an empty string.
- **Context live tokens:** `context_live_tokens_available` is only `true` when the llama.cpp server exposes per-slot token counts. Many servers expose `n_ctx` (capacity) but not current usage.
- **kv_cache_* fields:** Internal fields on the Rust side. They are NOT serialized into the WebSocket message; only the `context_*` equivalents are.
