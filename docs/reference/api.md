# Local LLM Foundry REST API Reference

The API surface is compatibility-stable for 2.0. Existing routes, auth
headers, unbranded serialized fields, and browser storage keys remain valid;
the product rebrand does not require API-client changes.

Llama Monitor serves its REST API on the same port as the web UI, typically `http://localhost:7778`.

This page documents the HTTP endpoints exposed by llama-monitor and the data shapes returned.

Internally, the API is implemented as a modular warp router in `src/web/api/mod.rs` delegating to
25+ domain-specific modules:

- `auth` — dashboard auth status, login, logout, config
- `benchmark` — benchmark, advise, sweep, MoE tuning
- `browse` — file system browsing
- `chat/` — chat streaming, suggestions, notes, guided generation, tabs
- `common` — shared auth helpers, context, error types
- `config` — settings, GPU env, token rotation, dashboard auth config
- `db` — database admin, queries, backups, restore, repair
- `debug` — diagnostic endpoints
- `hf` — HuggingFace integration (search, files, token, etc.)
- `lhm` — LibreHardwareMonitor (Windows-only)
- `llama_binary` — llama-server binary version and updates
- `metrics` — system/metrics endpoints
- `models` — model scan, downloads, GGUF metadata
- `presets` — preset CRUD
- `rapid_mlx_runtime` — Rapid-MLX runtime install, upgrade, repair, rollback, status, releases
- `remote_agent` — remote agent install/manage via SSH
- `self_update` — in-place update of llama-monitor
- `sensor_bridge` — Windows sensor-bridge service
- `sessions` — session CRUD, spawn, attach, detach, kill-llama, capabilities
- `sleep` — sleep mode management
- `spawn_wizard` — setup wizard helpers, chat templates, import launch files
- `system_tools` — system info, top processes, Metal GPU limit
- `templates` — system prompt templates
- `tls` — TLS/ACME configuration
- `tokens` — public token bootstrap routes
- `upstream` — upstream proxy helpers
- `vram` — VRAM estimation, quant comparison, auto-size

These modules define route handlers but do not change any public endpoint paths or authentication
rules compared to previous monolithic `api.rs` implementations.

## Base URL

```text
http://localhost:7778
```

### Rapid-MLX MTP repair

All endpoints require `Authorization: Bearer <api-token>`. Repair accepts only
absolute local paths; HF download and immutable parent selection remain explicit
caller responsibilities. When `python` is omitted, the job resolves the Python
interpreter from the Rapid-MLX executable environment rather than assuming the
system interpreter.

- `POST /api/rapid-mlx/mtp-repair` starts one bounded repair job. The body is
  `{ "target": "/path/to/trunk", "source": "/path/to/parent", "sourceFormat": "mlx" }`
  `{ "target": "/path/to/trunk", "recipe": "/path/to/recipe.json" }`, or
  `{ "target": "/path/to/trunk", "bf16Source": "owner/repo", "bf16Revision": "<commit>" }`.
  BF16 extraction uses the guarded `build-mtp-head.py` wrapper and requires a
  pinned revision for HF sources.
- `GET /api/rapid-mlx/mtp-repair/:jobId` polls progress and returns the
  candidate provenance or a fail-closed error.
- `POST /api/rapid-mlx/mtp-repair/:jobId/cancel` cancels a queued/running job.
- `GET /api/rapid-mlx/mtp-repair` lists retained jobs and registered sidecars.

Requalification accepts an optional `requalificationMode`: `screen` runs a short
sampled live probe and records `screened` evidence; `full` runs the production
sampled and constrained-tool gates and is the mode that can promote the runtime
capability; `full-diagnostic` also runs the optional greedy-parity safety check.
The preset editor defaults to `screen`; advanced users can select either full
mode. Job polling exposes the current phase/message while the UI shows the
mode-specific elapsed-time estimate.

Successful repair creates a candidate sidecar under the managed Rapid-MLX
sidecar directory. It is not considered qualified until served requalification
records an acceptance/throughput verdict.

To validate a relocated or otherwise pending managed sidecar, send
`{ "operation": "validate", "target": "/path/to/trunk" }` to the start
endpoint. The app derives the sidecar path and promotes provenance only after
the MLX tensor and `pre_fc_norm_*` checks pass. Validation never overwrites
qualified provenance.

## SPA Fallback Routing

The server includes an SPA fallback for unknown non-API paths to support client-side routing.

- For GET:
  - If the path does not start with /api and its last segment contains a dot
    (e.g., .js, .css), it is treated as asset-like and returns 404 if not found.
  - Otherwise, the server returns the SPA shell so the frontend router can handle the route.
- Existing /api/* endpoints and WebSocket routes are not affected by this fallback.
- Invariant: all SPA routes must have no dot in the last segment; new routes must respect
  this or update the guard.

## Rate Limiting and DoS Protections

- A global rate limiter is applied to all non-index routes:
  - 200 requests/second base, with a burst allowance up to 700 per second.
  - Excess requests are rejected with 429 Too Many Requests.
- WebSocket connections are limited to 50 concurrent.
  - Beyond that, new connections receive 429 with { "error": "too many connections" }.
- The /api/db/query endpoint:
  - Enforces a 256 KB HTTP body limit.
  - Enforces a 16 KB SQL string limit.
- These limits are global and not per-client.

## Sessions

All session endpoints require authentication via `Authorization: Bearer <token>`.
Most require the `api-token`; several elevated operations require the `db-admin-token`.
Without a valid token, the endpoint returns 401.

Handlers live in: `src/web/api/sessions.rs`.

Summary of endpoints:

- `GET /api/sessions` — list sessions (api-token)
- `POST /api/sessions` — create session record (api-token)
- `DELETE /api/sessions/:id` — delete session (db-admin-token)
- `GET /api/sessions/recent` — recent attach-mode sessions (api-token)
- `GET /api/sessions/active` — active session summary (api-token)
- `POST /api/sessions/active` — set active session (api-token)
- `GET /api/sessions/active/readiness` — check if active session is ready (api-token)
- `GET /api/sessions/check-endpoint` — verify reachability of an endpoint (api-token)
- `POST /api/sessions/spawn` — spawn llama-server from preset (db-admin-token, 15s cooldown)
- `POST /api/attach` — attach to existing endpoint (api-token, 10s cooldown)
- `POST /api/detach` — detach from attach session (api-token)
- `POST /api/kill-llama` — emergency kill llama-server (db-admin-token, 30s cooldown, requires `{ "confirm": "kill" }`)
- `GET /api/capabilities` — current metrics capability state (api-token)

### `GET /api/sessions`
Auth: api-token.
Returns the persisted session list from `sessions.json`.

```json
[
  {
    "id": "session_1746000000000",
    "name": "Default Session",
    "mode": { "Spawn": { "port": 8001 } },
    "status": "Stopped",
    "preset_id": "",
    "created_at": 1746000000,
    "last_active": 1746000000,
    "last_connected_at": 0,
    "connect_count": 0,
    "last_error": null
  }
]
```

All `Session` fields now use `#[serde(default)]` for backward compatibility. New fields:

| Field | Type | Default |
|-------|------|---------|
| `last_connected_at` | integer (unix timestamp) | `0` |
| `connect_count` | integer | `0` |
| `last_error` | string or null | `null` |

`mode` is serde's enum shape:
- launched session: `{ "Spawn": { "port": 8001 } }`
- attached session: `{ "Attach": { "endpoint": "http://192.168.1.50:8001" } }`

`status` is one of:
- `"Stopped"`
- `"Running"`
- `"Disconnected"`
- `{ "Error": "message" }`

### `POST /api/sessions`
Auth: api-token.
Creates a session record only. It does not start a server.

Request body must be a full `Session` object:

```json
{
  "id": "session_custom",
  "name": "Remote Box",
  "mode": { "Attach": { "endpoint": "http://192.168.1.50:8001" } },
  "status": "Disconnected",
  "preset_id": "",
  "created_at": 1746000000,
  "last_active": 1746000000
}
```

Response:

```json
{ "ok": true }
```

On failure:

```json
{ "ok": false, "error": "Maximum sessions reached" }
```

### `DELETE /api/sessions/{id}`
Auth: db-admin-token.

```json
{ "ok": true }
```

### `GET /api/sessions/active`
Auth: api-token.
Returns a compact active-session summary, not the full `Session` object.

```json
{
  "id": "session_1746000000000",
  "name": "Default Session",
  "mode": "Spawn:8001",
  "status": "Running",
  "last_active": 1746001000
}
```

If there is no active session:

```json
{ "error": "No active session" }
```

### `GET /api/sessions/recent`
Auth: api-token.
Returns recent attach-mode sessions sorted by `last_connected_at` descending, limited to 10.

```json
{
  "sessions": [
    {
      "id": "session_1746000000000",
      "name": "Remote Box",
      "mode": { "Attach": { "endpoint": "http://192.168.1.50:8001" } },
      "status": "Disconnected",
      "preset_id": "",
      "created_at": 1746000000,
      "last_active": 1746001000,
      "last_connected_at": 1746002000,
      "connect_count": 5,
      "last_error": null
    }
  ],
  "active_session_id": "session_1746000000000"
}
```

### `POST /api/sessions/active`
Auth: api-token.

```json
{ "id": "session_1746000000000" }
```

Response:

```json
{ "ok": true }
```

### `POST /api/sessions/spawn`
Auth: db-admin-token.
Creates a launch session, starts `llama-server` from a saved preset, and makes it active.
Used by:
- Spawn wizard
- Tuning panel "Apply" flow (to restart llama-server with adjusted configuration)

This endpoint enforces a 15-second cooldown between calls. If called too soon, it returns 429 with:

```json
{
  "ok": false,
  "error": "too soon; please wait",
  "seconds_remaining": 10
}
```

Request:

```json
{
  "preset_id": "default-1",
  "name": "Session on port 8001",
  "port": 8001,
  "selection": null,
  "expected_revision": 1,
  "expected_resolved_config_hash": "cfg-v1:<sha256>"
}
```

`preset_id` is required. For a bundled preset, the server resolves the saved default or the optional exact `selection`; it never trusts a client-submitted flat preset. `expected_revision` is checked when supplied and is required with a one-shot selection. `expected_resolved_config_hash` protects a preview-to-start action and returns `412 preview_stale` if the effective configuration changed. `name` defaults to `Session on port {port}`. `port` defaults to `8001`.

Success response:

```json
{
  "ok": true,
  "session_id": "session_1746000000000"
}
```

### `POST /api/attach`
Auth: api-token.
Attaches to a reachable private-network or loopback endpoint.

Request:

```json
{
  "endpoint": "http://192.168.1.50:8001",
  "backend": "rapid_mlx",
  "model_identity": "served-model-name",
  "api_key": "optional-runtime-key"
}
```

`backend` defaults to `llama_cpp` for compatibility. Rapid-MLX attach uses
`/health/ready`, authenticates `/v1/models` and `/v1/status` when a key is supplied,
and discovers `model_identity` from `/v1/models` when it is omitted. API keys remain
transient and are never serialized with the session.

Success response:

```json
{
  "ok": true,
  "backend": "rapid_mlx",
  "model_identity": "served-model-name",
  "warning": null
}
```

Before attaching, llama-monitor performs the selected backend's non-mutating readiness
check. It does not guess the backend from an OpenAI-compatible response.

```json
{
  "ok": false,
  "error": "Cannot reach the selected inference runtime at <endpoint>. Is it ready?"
}
```

If the runtime is ready but its diagnostics endpoint is unavailable, attach still
succeeds and `warning` explains that diagnostics are unavailable.

### `POST /api/detach`
Auth: api-token.
Detaches only if the active session is an attach session.

```json
{ "ok": true }
```

### `GET /api/metrics/inference`
Auth: api-token.
Returns the latest backend-neutral inference sample and a privacy-safe metric
dictionary. Numeric values are aggregate-only; each dictionary entry includes
`state` (`effective` or `unavailable`), `unit`, and an explanatory `reason`
when the selected backend does not expose that metric. Missing telemetry is
never represented as zero.

```json
{
  "ok": true,
  "snapshot": {
    "backend": "rapid_mlx",
    "health": "ok",
    "metrics": {
      "generation_tokens_per_second": {
        "state": "effective",
        "value": 42.1,
        "unit": "tokens_per_second"
      },
      "ttft": {
        "state": "unavailable",
        "unit": "milliseconds",
        "reason": "backend did not report this metric"
      }
    }
  }
}
```

### `GET /api/capabilities`
Auth: api-token.
Returns the current metrics capability state.

```json
{
  "capabilities": {
    "inference": true,
    "system": true,
    "gpu": true,
    "cpu_temperature": true,
    "memory": true,
    "host_metrics": true,
    "tray": true,
    "sensor_bridge_setup_available": false
  },
  "endpoint_kind": "Local",
  "session_kind": "Spawn",
  "tray_mode": "Desktop",
  "availability": {
    "system": "Available",
    "gpu": "Available",
    "cpu_temp": "Available"
  }
}
```

## Server Control

### `POST /api/start`
Auth: api-token.
Starts `llama-server` for the active spawn session. Request body is the full server config payload used by the launcher.

### `POST /api/stop`
Auth: api-token.
Stops the managed `llama-server`.

### `POST /api/kill-llama`
Emergency process kill for `llama-server`.

## Presets

Presets are stored in `presets.json` and use the `ModelPreset` struct. Bundle
cards and selection resolution are handled separately from legacy CRUD in
`src/web/api/preset_bundles.rs`.

### `GET /api/presets`
Auth: api-token.

```json
[
  {
    "id": "default-1",
    "name": "Example: Small Model 128K context",
    "model_path": "",
    "context_size": 128000,
    "ctk": "f16",
    "ctv": "f16",
    "tensor_split": "",
    "batch_size": 2048,
    "ubatch_size": 2048,
    "no_mmap": true,
    "ngram_spec": true,
    "parallel_slots": 1,
    "temperature": null,
    "top_p": null,
    "top_k": null,
    "min_p": null,
    "repeat_penalty": null,
    "presence_penalty": null,
    "n_cpu_moe": null,
    "gpu_layers": null,
    "mlock": false,
    "flash_attn": "",
    "split_mode": "",
    "main_gpu": null,
    "threads": null,
    "threads_batch": null,
    "rope_scaling": "",
    "rope_freq_base": null,
    "rope_freq_scale": null,
    "draft_model": "",
    "draft_min": null,
    "draft_max": null,
    "spec_ngram_size": null,
    "seed": null,
    "enable_thinking": null,
    "preserve_thinking": null,
    "reasoning": null,
    "reasoning_budget": null,
    "reasoning_budget_message": null,
    "system_prompt_file": "",
    "extra_args": ""
  }
]
```

### `GET /api/preset-cards`
Auth: api-token.

Returns the redacted setup-card projection and a catalog concurrency token. Cards expose safe artifact labels and availability, saved bundle selections, and bundle identity; local paths and API keys are not included.

```json
{
  "cards": [{"id": "preset-1", "revision": 1, "name": "Qwen bundle", "bundle": {"bundle_id": "qwen38", "tune_id": "brainwaves-q4-q5", "default_selection": {"artifact_id": "weights-q4", "context_size": 160000, "kv_policy": "q4_0_q4_0", "performance_id": "balanced", "n_cpu_moe": 0}}}],
  "catalog_etag": "catalog-v1:<sha256>"
}
```

### `POST /api/presets/{id}/resolve`
Auth: api-token.

Resolves the saved bundle selection or an optional one-shot `selection` through the server-owned resolver. Resolution does not write state. The response includes `sel-v1:` and `cfg-v1:` identifiers, normalized selection, changes, capability reasons, and a tagged estimate status. It never includes a flat preset, local artifact paths, API keys, or raw launch arguments.

```json
{"selection": {"artifact_id": "weights-q4", "context_size": 160000, "kv_policy": "q4_0_q4_0", "performance_id": "balanced", "n_cpu_moe": 0}}
```

### `PATCH /api/presets/{id}/selection`
Auth: api-token.

Updates a bundled preset’s saved default selection. `expected_revision` is required; the server strips any submitted `intent_source`, validates the candidate, durably saves it, then changes in-memory state. A stale revision returns `409` and leaves state unchanged.

```json
{"expected_revision": 1, "selection": {"artifact_id": "weights-q5", "context_size": 200000, "kv_policy": "q4_0_q4_0", "performance_id": "throughput", "n_cpu_moe": 6}}
```

### `POST /api/model-defaults`
Auth: api-token.

Returns model-family sampling recommendations used by the setup wizard and preset editor.

```json
{
  "defaults": {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "enable_thinking": true,
    "preserve_thinking": true,
    "reasoning": true,
    "reasoning_budget": 16384,
    "reasoning_budget_message": "\nFinal Answer:"
  },
  "presets": [
    {
      "name": "Agentic / Coding (thinking)",
      "description": "Recommended default for coding agents and tool-heavy work."
    }
  ]
}
```

### `POST /api/presets`
Auth: api-token.
Creates a preset from a full `ModelPreset` payload. If `id` is omitted, serde supplies one.

Request body (full `ModelPreset` shape, all fields optional with defaults):

```json
{
  "id": "default-1",
  "name": "Example: Small Model 128K context",
  "model_path": "",
  "context_size": 128000,
  "ctk": "f16",
  "ctv": "f16",
  "tensor_split": "",
  "batch_size": 2048,
  "ubatch_size": 2048,
  "no_mmap": true,
  "ngram_spec": true,
  "parallel_slots": 1,
  "temperature": null,
  "top_p": null,
  "top_k": null,
  "min_p": null,
  "repeat_penalty": null,
  "presence_penalty": null,
  "n_cpu_moe": null,
  "gpu_layers": null,
  "mlock": false,
  "flash_attn": "",
  "split_mode": "",
  "main_gpu": null,
  "threads": null,
  "threads_batch": null,
  "prio": null,
  "prio_batch": null,
  "rope_scaling": "",
  "rope_freq_base": null,
  "rope_freq_scale": null,
  "draft_model": "",
  "draft_min": null,
  "draft_max": null,
  "spec_ngram_size": null,
  "spec_type": null,
  "spec_default": false,
  "seed": null,
  "enable_thinking": null,
  "preserve_thinking": null,
  "reasoning": null,
  "reasoning_budget": null,
  "reasoning_budget_message": null,
  "system_prompt_file": "",
  "extra_args": "",
  "bind_host": null,
  "port": null,
  "hf_repo": null,
  "chat_template_file": null,
  "mmproj": null,
  "grammar": null,
  "json_schema": null,
  "cache_type_k": null,
  "cache_type_v": null,
  "max_tokens": null,
  "api_key": null,
  "alias": null,
  "benchmark_mode": false,
  "fit_enabled": null,
  "fit_ctx": null,
  "fit_target": null,
  "fit_print": null,
  "kv_unified": null,
  "cache_idle_slots": null,
  "cache_ram_mib": null
}
```

All fields use `#[serde(default)]` for backward compatibility.

**KV cache fields:**

`ctk` and `ctv` (strings, default empty = llama-server default) are the canonical K/V cache-type fields (schema v5). Launch validation reads only these. `cache_type_k`/`cache_type_v` below are retained for read-compatibility only; new writes should use the canonical fields.

**Spawn V2 extended fields** (added after initial preset schema):

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `hf_repo` | Option<String> | null | HF repo identifier for model download |
| `chat_template_file` | Option<String> | null | Path to chat template file |
| `mmproj` | Option<String> | null | Multimodal projector path |
| `grammar` | Option<String> | null | Grammar constraint file |
| `json_schema` | Option<String> | null | JSON schema constraint |
| `cache_type_k` | Option<String> | null | Deprecated: use `ctk`. Retained for backward compatibility |
| `cache_type_v` | Option<String> | null | Deprecated: use `ctv`. Retained for backward compatibility |
| `max_tokens` | Option<u64> | null | Max tokens limit |
| `enable_thinking` | Option<bool> | null | Enable thinking mode |
| `preserve_thinking` | Option<bool> | null | Preserve thinking content |
| `reasoning` | Option<String> | null | Reasoning mode |
| `reasoning_budget` | Option<i32> | null | Reasoning token budget |
| `reasoning_budget_message` | Option<String> | null | Budget limit message |
| `api_key` | Option<String> | null | API key for hosted endpoint |
| `alias` | Option<String> | null | Display alias for the preset |
| `benchmark_mode` | bool | false | Run in benchmark mode |
| `fit_enabled` | Option<bool> | null | `null` leaves the server default unchanged, `true` emits `--fit on`, and `false` emits `--fit off` |
| `fit_ctx` | Option<u32> | null | Legacy minimum context option; emitted only when fit is enabled and `fit_target` is unset |
| `fit_target` | Option<String> | null | Fit memory margin in MB; the preferred fit option when fit is enabled |
| `fit_print` | Option<bool> | null | Legacy persisted field; not emitted for current llama-server builds |
| `kv_unified` | Option<bool> | null | `null` leaves the server default unchanged, `true` emits `--kv-unified`, and `false` emits `--no-kv-unified` |
| `cache_idle_slots` | Option<bool> | null | Cache idle slots |
| `cache_ram_mib` | Option<i32> | null | Max RAM for cache (MiB) |
| `prio` | Option<i32> | null | Process priority |
| `prio_batch` | Option<i32> | null | Batch process priority |
| `gguf_architecture` | Option<String> | null | GGUF architecture (e.g. "llama", "qwen3_6") |
| `param_count` | Option<u64> | null | Total parameter count from GGUF |
| `family` | Option<String> | null | Derived family slug (e.g. "qwen36", "llama3") |
| `size_class` | Option<String> | null | Size class derived from param_count |
| `architecture_kind` | Option<String> | null | "dense" / "moe" / "hybrid_moe" |
| `expert_count` | Option<u32> | null | MoE experts per layer (MoE models) |
| `expert_used_count` | Option<u32> | null | Active MoE experts per token |
| `active_params_b` | Option<f64> | null | Effective active parameters (in billions) |
| `bind_host` | Option<String> | null | Bind address override |
| `port` | Option<u16> | null | Port override |

Speculative decoding V2 fields:

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `spec_type` | Option<String> | null | Speculative decoding type |
| `spec_default` | bool | false | Use default spec settings |
| `spec_draft_n_max` | Option<u32> | null | Max draft tokens |
| `spec_draft_n_min` | Option<u32> | null | Min draft tokens |
| `spec_draft_p_split` | Option<f32> | null | Draft split threshold |
| `spec_draft_p_min` | Option<f32> | null | Draft min probability |
| `spec_draft_ngl` | Option<i32> | null | Draft NGL layers |
| `spec_draft_device` | Option<String> | null | Draft device |
| `spec_draft_cpu_moe` | bool | false | CPU MoE for draft |
| `spec_draft_n_cpu_moe` | Option<i32> | null | CPU MoE count for draft |
| `spec_draft_type_k` | Option<String> | null | Draft key type |
| `spec_draft_type_v` | Option<String> | null | Draft value type |
| `spec_ngram_mod_n_min` | Option<u32> | null | Ngram mod min |
| `spec_ngram_mod_n_max` | Option<u32> | null | Ngram mod max |
| `spec_ngram_mod_n_match` | Option<u32> | null | Ngram mod match |
| `spec_ngram_simple_size_n` | Option<u32> | null | Ngram simple size N |
| `spec_ngram_simple_size_m` | Option<u32> | null | Ngram simple size M |
| `spec_ngram_simple_min_hits` | Option<u32> | null | Ngram simple min hits |
| `spec_ngram_map_k_size_n` | Option<u32> | null | Ngram map-k size N |
| `spec_ngram_map_k_size_m` | Option<u32> | null | Ngram map-k size M |
| `spec_ngram_map_k_min_hits` | Option<u32> | null | Ngram map-k min hits |
| `spec_ngram_map_k4v_size_n` | Option<u32> | null | Ngram map-k4v size N |
| `spec_ngram_map_k4v_size_m` | Option<u32> | null | Ngram map-k4v size M |
| `spec_ngram_map_k4v_min_hits` | Option<u32> | null | Ngram map-k4v min hits |

Response:

```json
{ "ok": true, "preset": { "...": "..." } }
```

### `PUT /api/presets/{id}`
Auth: api-token.
Updates the preset matched by the path `id`. Accepts the same `ModelPreset`
shape as POST. A bundled preset must be sent as
`{"expected_revision": <revision>, "preset": <model-preset>}`; the submitted
flat projection must agree with `bundle.default_selection`. The server assigns
the next revision and returns `409` for a stale revision or `400` for a
projection conflict.

### `POST /api/presets/{id}/copy`
Auth: api-token.

Creates a new preset from the current preset after checking its revision. The
source is unchanged; the new preset receives a server-generated ID and
`revision: 1`.

```json
{"expected_revision": 2, "new_name": "Qwen copy"}
```

### `POST /api/presets/{id}/convert-to-bundle`
Auth: api-token.

Explicitly converts one legacy flat preset into a one-artifact bundle. The
conversion is transactional, uses the saved flat values and adopted companion
paths, and assigns a fresh bundle/tune identity. `conversion` is reserved for
future bounded conversion options.

```json
{"expected_revision": 1, "conversion": {}}
```

### `DELETE /api/presets/{id}`
Auth: db-admin-token. Requires the exact confirmation string and the current
preset revision.

```json
{ "expected_revision": 2, "confirmation": "DELETE PRESET" }
```

### `POST /api/presets/reset`
Auth: db-admin-token. Requires the current catalog etag and the exact
confirmation string. Replaces the in-memory and on-disk preset list with
factory defaults only after both guards pass.

```json
{ "expected_catalog_etag": "catalog-v1:<sha256>", "confirmation": "RESET PRESETS" }
```

Successful mutations return the updated `revision` where applicable and a
fresh `catalog_etag`. Persistence failures return a non-success response and
leave the live in-memory catalog unchanged.

## Templates

Templates are stored in `templates.json`. Built-in personas live in the frontend and are merged client-side; this API only returns user-stored entries.
Route handlers: `src/web/api/templates.rs`.

### `GET /api/templates`
Auth: api-token.

```json
[
  {
    "id": "t1746000000000",
    "name": "Helpful Assistant",
    "prompt": "You are a helpful assistant.",
    "explicit_policies": {
      "level1": "Soft policy text",
      "level2": "Unrestricted policy text"
    }
  }
]
```

`explicit_policies` is optional, and each level field is optional.

### `POST /api/templates`
Auth: api-token.
Creates a template from a full `SystemPromptTemplate` payload.

```json
{ "ok": true, "template": { "...": "..." } }
```

### `PUT /api/templates/{id}`
Auth: api-token.
Updates the template matched by the path `id`.

### `DELETE /api/templates/{id}`
Auth: api-token.

```json
{ "ok": true }
```

## Models

Route handlers: `src/web/api/models.rs`.

### `GET /api/models`
Auth: api-token.
Returns the unified typed inventory for the configured `models_dir` as an array. Legacy
GGUF fields remain available, while every entry also reports its format, source,
lifecycle, compatibility, supported backends, companion kind, and typed Rapid-MLX
source when applicable. GGUF entries retain `tags` and filename-derived
`classification` fields.

```json
[
  {
    "path": "/models/gguf/Qwen3.5-27B-Q4_0.gguf",
    "filename": "Qwen3.5-27B-Q4_0.gguf",
    "size_bytes": 4680000000,
    "size_display": "4.4 GB",
    "quant_type": "Q4_0",
    "model_name": "Qwen3.5-27B",
    "is_split": false,
    "format": "gguf",
    "source": "local",
    "lifecycle": "ready",
    "compatibility": "verified",
    "supported_backends": ["llama_cpp"],
    "companion_kind": null,
    "model_source": {
      "kind": "gguf_file",
      "path": "/models/gguf/Qwen3.5-27B-Q4_0.gguf"
    },
    "tags": [],
    "classification": { "is_mtp": false, "is_moe": false, "is_vision": false }
  }
]
```

Improved quant detection (branch-specific):
- Now recognizes `-IQ` and `_IQ` patterns (e.g., IQ2_XXS, IQ3_M).
- MTP models: files ending in `-mtp.gguf` (Unsloth naming) are marked as MTP and shown with a badge in the UI.

### `GET /api/models/inventory`
Auth: api-token.

Returns `{ models_dir, entries, truncated }`. `entries` uses the same typed inventory
objects as `GET /api/models`; this wrapper also reports the canonical library root and
whether the 10,000-entry safety bound was reached.

### `POST /api/models/rapid-mlx/resolve/preview`
Auth: api-token.

Validates a tagged Rapid-MLX model source without downloading, converting, or launching
it. The response state is `ready`, `conversion_required`, `unsupported_source`, or
`invalid`, with warnings and remediation. GGUF returns `unsupported_source` and retains
the llama.cpp recommendation.

### `POST /api/models/gguf/import/compatibility/preview`
Auth: api-token.

Performs a converter-free, network-free metadata inspection for experimental
GGUF-to-MLX recovery. `path` must be library-relative, and its canonical target must be a
non-symlinked `.gguf` file inside the configured `models_dir`. Absolute and root-relative
paths are rejected rather than accepted from user input.

```json
{
  "path": "gguf/model-Q6_K.gguf"
}
```

The versioned response includes bounded source identity, authoritative architecture,
tensor/quant inventory, tokenizer/config/auxiliary-asset observations, compatibility,
exact missing fields/assets, warnings, resource estimate, and remediation. A
`bounded_gguf_header_sha256` identifies metadata plus the tensor directory; it is
explicitly not a hash of all model weights. R1 produces no weights or launchable output.

Malformed JSON, traversal, symlinks, non-GGUF files, paths outside the model library,
incomplete headers, or metadata/tensor directories over 64 MiB return `400`.

### `GET /api/models/import-lab/availability`
Auth: api-token.

Returns the local Apple-Silicon execution gate, exact experimental profile, llama.cpp
fallback, and the invariant that recovered output is not launchable.

### `POST /api/models/import-lab/resource-estimate`
Auth: api-token.

Accepts the same `{ "path": "gguf/model.gguf" }` library-relative request as the
compatibility preview. It returns source and estimated recovered-FP16 bytes, required
disk space, current disk/RAM headroom, and guidance. Metadata parsing runs in a bounded
two-worker blocking pool with a 15-second route timeout. Invalid JSON and unsafe paths
return `400`; saturation fails fast with `429`; a timeout returns `408`. A timed-out
blocking task retains its worker permit until it exits, preventing hidden overcommit.

### `GET /api/models/import-lab/jobs`
Auth: api-token.

Lists up to 32 retained in-memory recovery jobs. Jobs expose a typed state, phase,
progress percentage, bounded diagnostics, cancellation availability, and an optional
terminal result. Jobs do not resume after an app restart; completed cache manifests
remain durable model-library provenance. Public failure diagnostics use stable,
path-free messages rather than raw worker errors.

### `POST /api/models/import-lab/jobs`
Auth: api-token.

Queues the exact supported experimental profile and returns `202`:

```json
{ "source_path": "gguf/SmolLM2-135M-Instruct-Q8_0.gguf" }
```

The path is canonicalized and rejected for traversal, symlink components, non-GGUF
targets, or targets outside `models_dir` before a worker slot is consumed. One recovery
worker runs at a time. The original GGUF is never modified and published results remain
`experimental` and `launchable: false`.

### `GET /api/models/import-lab/jobs/{id}`
Auth: api-token. Returns one job or `404`.

### `POST /api/models/import-lab/jobs/{id}/cancel`
Auth: api-token. Requests cooperative process-group cancellation and staging cleanup.

### `DELETE /api/models/import-lab/jobs/{id}`
Auth: api-token. Removes a terminal in-memory job record. Active jobs cannot be removed,
and this endpoint never deletes the source GGUF or a published cache.

### `POST /api/models/library/migration/preview`
Auth: api-token.

Builds a non-mutating, bounded migration plan. The optional body selects only explicit
legacy Hugging Face repositories:

```json
{ "hf_repos": ["mlx-community/Qwen3-0.6B-4bit"] }
```

### `POST /api/models/library/migration/execute`
Auth: db-admin-token.

Executes the exact previewed plan. Concurrent executions are rejected.

```json
{
  "plan_id": "<64-character preview id>",
  "confirmation": "MIGRATE_MODEL_LIBRARY",
  "hf_repos": ["mlx-community/Qwen3-0.6B-4bit"]
}
```

The repository selection must match the preview. The operation is journaled and
restartable; a stale preview, collision, path escape, or changed persistence file is
rejected before a new plan starts.

### `GET /api/models/root-relocation/status`
Auth: api-token.

Returns the legacy default model root, Foundry destination, current selection marker,
and whether an explicit choice is required. Custom/external model roots are reported
but are never eligible for implicit relocation.

### `POST /api/models/root-relocation/preview`
Auth: api-token.

Builds a read-only, deterministic plan for the default legacy model root. Choose one
of the two explicit policies:

```json
{ "choice": "keep_legacy" }
```

or:

```json
{ "choice": "move_into_foundry" }
```

Move previews include managed models, runtimes, caches, sidecars, partial downloads,
unknown entries, free-space headroom, SHA-256 identities, and absolute persistence
path rewrites.

### `POST /api/models/root-relocation/execute`
Auth: db-admin-token.

Executes the exact root-relocation preview. The choice, 64-character `plan_id`, and
matching confirmation are required:

```json
{
  "choice": "move_into_foundry",
  "plan_id": "<64-character preview id>",
  "confirmation": "MOVE_MODELS_INTO_FOUNDRY"
}
```

`KEEP_LEGACY_MODEL_ROOT` writes a receipt-backed external-root selection without
copying. `MOVE_MODELS_INTO_FOUNDRY` copies first, verifies hashes, rewrites persisted
absolute model paths once, retains the source, and returns `restart_required: true`.
Cleanup is a separate receipt-scoped operation.

### `POST /api/models/refresh`
Auth: api-token.
Rescans `models_dir`.

### Calibration routes

Calibration start requests may include optional `server_qualification` tracks;
the default is single-user `parallel_requests: 1`. Qualification receipts are
returned with the authenticated calibration receipt and never replace an
active user server.

`POST /api/calibrations/match` accepts a saved llama.cpp `preset_id`, workload,
and budget and returns tiered receipt evidence for Spawn Wizard. Exact results
match the complete artifact/runtime fingerprint. Compatible results match the
introspected family/shape, GGUF weight-quantization signature,
hardware/workload, and normalized runtime capabilities; related results may
use a different weight quantization and are review-only. Runtime build changes
are surfaced as warnings, and Rapid-MLX receipts are never translated into
this path.

Calibration is llama.cpp-only and requires a regular local GGUF inside the
configured model library. Read and job-control routes use `api-token`;
apply and rollback require `db-admin-token` plus their exact confirmation
phrases. Calibration never changes the active inference session.

- `POST /api/calibrations/preflight` — validate the selected preset, model
  library, managed sibling `llama-bench`, and bounded Quick plan.
- `POST /api/calibrations` — start a confirmed bounded Quick job.
- `GET /api/calibrations/{id}` — poll the durable job snapshot.
- `GET /api/calibrations/{id}/receipt` — read the measured receipt and apply
  history.
- `POST /api/calibrations/{id}/cancel` — cancel the owned benchmark process.
- `POST /api/calibrations/{id}/apply` — apply the measured candidate, creating
  a derived preset by default; bounded post-apply validation runs by default
  and immediately rolls back failed validation.
- `POST /api/calibrations/{id}/rollback` — restore the private pre-apply
  snapshot or remove the derived preset when its fingerprint is unchanged.

Apply requires `APPLY_CALIBRATION`; rollback requires
`ROLLBACK_CALIBRATION`. Both operations reject stale fingerprints and never
silently overwrite user edits.

Success:

```json
{ "ok": true, "count": 12 }
```

Failure when no model directory is configured:

```json
{ "ok": false, "error": "no models directory configured (use --models-dir)" }
```

### `POST /api/models/download/start`
Start a new model download (currently HuggingFace only).

- Auth: api-token (or db-admin-token).
- Route handler: `models.rs` delegating to `src/model_download.rs`.

New behaviors (branch-specific):

- Duplicate guard: if a download for the same (repo, file) is already running, the endpoint
  rejects with `Already downloading: ...`.
- Existing-file guard: if the target file already exists and is not partial, the endpoint
  rejects with `File already exists at: ...`.
- Concurrency limit: max 2 simultaneous running downloads; further starts are rejected with
  `Too many downloads in progress`.
- On failure, error messages are human-readable and may include retry hints.

Request:

```json
{
  "model": "unsloth/Qwen3.5-27B-Q4_K_M-GGUF/qwen3.5-27b-q4_k_m.gguf",
  "source": "hf"
}
```

Success:

```json
{
  "ok": true,
  "download_id": "abc123"
}
```

Failure examples:

```json
{ "ok": false, "error": "Already downloading: unsloth/Qwen3.5-27B-Q4_K_M-GGUF/qwen3.5-27b-q4_k_m.gguf. Please wait until it completes." }
```

```json
{ "ok": false, "error": "File already exists at: /models/Qwen3.5-27B-Q4_K_M.gguf. It may be available in your library." }
```

```json
{ "ok": false, "error": "Too many downloads in progress. Please wait for one to finish." }
```

### `GET /api/models/download/{id}/status`
Poll download progress for a given `download_id`.

- Auth: api-token.

Response (running):

```json
{
  "ok": true,
  "status": {
    "download_id": "abc123",
    "status": "running",
    "bytes_downloaded": 1234567890,
    "total_bytes": 2765432100,
    "speed": 5000000,
    "eta": 500,
    "message": "",
    "local_path": "/models/Qwen3.5-27B-Q4_K_M.gguf"
  }
}
```

Response (completed):

```json
{
  "ok": true,
  "status": {
    "download_id": "abc123",
    "status": "completed",
    "bytes_downloaded": 2765432100,
    "total_bytes": 2765432100,
    "speed": 0,
    "eta": 0,
    "message": "",
    "local_path": "/models/Qwen3.5-27B-Q4_K_M.gguf"
  }
}
```

Response (failed):

```json
{
  "ok": true,
  "status": {
    "download_id": "abc123",
    "status": "failed",
    "bytes_downloaded": 1000000,
    "total_bytes": 2765432100,
    "speed": 0,
    "eta": 0,
    "message": "Connection timeout; you can retry.",
    "local_path": "/models/Qwen3.5-27B-Q4_K_M.gguf"
  }
}
```

If not found:

```json
{ "ok": false, "error": "Download not found" }
```

(404 status code)

### `POST /api/models/download/{id}/cancel`
Cancel an active download.

- Auth: api-token.

Response:

```json
{ "ok": true }
```

Or:

```json
{ "ok": false, "error": "Download not found or already finished" }
```

### `POST /api/models/gguf-meta`
Read GGUF metadata directly from a local model file (no llama-server spawn).

- Auth: api-token.
- Returns architecture, param_count, block_count, head counts, etc.
- For hybrid DeltaNet models (e.g., Qwen3.5, Qwen3.6), also returns `n_attn_layers` derived from the VRAM estimator heuristic.

Request:

```json
{ "model_path": "/models/Qwen3.5-27B-Q4_K_M.gguf" }
```

Response:

```json
{
  "ok": true,
  "architecture": "qwen35",
  "param_count": 27000000000,
  "block_count": 48,
  "head_count": 48,
  "head_count_kv": 8,
  "key_length": 128,
  "context_length": 131072,
  "embedding_length": 5120,
  "feed_forward_length": 27392,
  "expert_count": null,
  "expert_used_count": null,
  "mtp_depth": null,
  "n_attn_layers": 24
}
```

## Settings

Route handlers: `src/web/api/config.rs`.

### `GET /api/settings`
Auth: api-token.
Returns the persisted `UiSettings` object from `ui-settings.json`, with sensitive fields masked.

Security:
- `remote_agent_token` is masked (e.g., `"••••••••"`) to reduce exposure when consumed by the browser.

Example:

```json
{
  "preset_id": "",
  "port": 8001,
  "llama_server_path": "",
  "llama_server_cwd": "",
  "models_dir": "",
  "extra_models_dirs": [],
  "server_endpoint": "",
  "llama_poll_interval": 1,
  "remote_agent_url": "",
  "remote_agent_token": "••••••••",
  "remote_agent_ssh_autostart": false,
  "remote_agent_ssh_target": "",
  "remote_agent_ssh_command": "",
  "explicit_mode_policy": "",
  "context_card_view": "gauge",
  "ws_push_interval_ms": 500,
  "chat_input_height": "",
  "enabled_context_notes": true,
  "enabled_suggestions": true,
  "enabled_quick_guide": true,
  "default_sidebar_width": 280,
  "suggestion_prompts": {},
  "suggestion_count": 5,
  "context_depth": 10,
  "chat_date_format": "MM/DD/YY",
  "enter_to_send": true,
  "context_notes_sidebar_expanded": false,
  "context_notes_intro_hidden": false,
  "persist_thinking_content": false,
  "custom_suggestion_categories": {}
}
```

New field:

| Field | Type | Default |
|-------|------|---------|
| `extra_models_dirs` | array of strings | `[]` |

`extra_models_dirs` is an array of additional directories to scan for models beyond the primary `models_dir`. Useful for models distributed across multiple drives or folders.

### `GET /api/settings/full`
Returns the same `UiSettings` object, but with the real `remote_agent_token` value instead of a masked placeholder.

Security:
- Requires `api-token` authentication:
  - Header: `Authorization: Bearer <api-token>`
- Intended for trusted clients (e.g., internal tools, remote-agent.js) that need the actual token.

Example:

```json
{
  "preset_id": "",
  "port": 8001,
  "llama_server_path": "",
  "llama_server_cwd": "",
  "models_dir": "",
  "extra_models_dirs": [],
  "server_endpoint": "",
  "llama_poll_interval": 1,
  "remote_agent_url": "",
  "remote_agent_token": "actual-token-value",
  "remote_agent_ssh_autostart": false,
  "remote_agent_ssh_target": "",
  "remote_agent_ssh_command": "",
  "explicit_mode_policy": "",
  "context_card_view": "gauge",
  "ws_push_interval_ms": 500,
  "chat_input_height": "",
  "enabled_context_notes": true,
  "enabled_suggestions": true,
  "enabled_quick_guide": true,
  "default_sidebar_width": 280,
  "suggestion_prompts": {},
  "suggestion_count": 5,
  "context_depth": 10,
  "chat_date_format": "MM/DD/YY",
  "enter_to_send": true,
  "context_notes_sidebar_expanded": false,
  "context_notes_intro_hidden": false,
  "custom_suggestion_categories": {}
}
```

### `PUT /api/settings`
Auth: api-token.
Saves the `UiSettings` object to disk.

Response:

```json
{ "ok": true }
```

Notes:
- The live handler expects the current full `UiSettings` shape.
- The frontend also uses a narrow partial-update path for `ws_push_interval_ms`; external clients should prefer sending the full object.
- `custom_suggestion_categories` is a map of category key to `{ "prompt": string, "explicit": bool }`.

## GPU Environment

### `GET /api/gpu-env`

```json
{
  "env": {
    "arch": "auto",
    "devices": "",
    "rocm_path": "/opt/rocm",
    "extra_env": []
  },
  "architectures": [
    { "id": "auto", "name": "Auto-detect", "hsa_version": "" }
  ],
  "detected": {
    "arch": "gfx1100",
    "count": 1,
    "names": ["gfx1100"]
  }
}
```

`detected` can be `null` if no GPU probe succeeds.

### `PUT /api/gpu-env`
Auth: api-token.
Request body is the persisted `GpuEnv` object:

```json
{
  "arch": "auto",
  "devices": "",
  "rocm_path": "/opt/rocm",
  "extra_env": [["HSA_ENABLE_SDMA", "0"]]
}
```

Response:

```json
{ "ok": true }
```

## File Browser

Route handlers: `src/web/api/browse.rs`.

### `GET /api/browse`
Auth: api-token.
Browses a local directory.

Query params:
- `path`: absolute or relative path; if omitted, starts at the current user's home directory
- `filter`: optional, currently `gguf` or `executable`

Example:

```text
/api/browse?path=/models&filter=gguf
```

Response:

```json
{
  "path": "/models",
  "parent": "/",
  "entries": [
    {
      "name": "Qwen3.5-27B-Q4_0.gguf",
      "is_dir": false,
      "size": 4680000000,
      "size_display": "4680 MB",
      "path": "/models/Qwen3.5-27B-Q4_0.gguf"
    }
  ]
}
```

On invalid input the API returns a JSON error payload such as:

```json
{ "path": "/missing", "error": "Path not found" }
```

## Chat Transport

Route handlers: `src/web/api/chat/stream.rs`, `src/web/api/chat/suggestions.rs`,
`src/web/api/chat/notes.rs`, `src/web/api/chat/guided.rs`.

### `POST /api/chat`
Auth: api-token.
Pass-through streaming proxy to the active session's `/v1/chat/completions`.

The request body is forwarded as raw bytes. The server does not validate or reshape the OpenAI-compatible payload before forwarding it upstream.

The response is an SSE stream that forwards upstream `data: ...` events.

Inference admission behavior:
- llama-monitor serializes its own inference requests per active session before forwarding them upstream.
- On single-slot llama.cpp servers, the route waits briefly for the current upstream request to finish instead of immediately issuing a competing request.
- If the upstream server stays occupied, the route returns `429 Too Many Requests` with a plain-text busy message instead of a generic `500`.
- If the upstream server cannot be reached, the route returns `502 Bad Gateway`.
- If the upstream server accepts the connection but does not respond in time, the route returns `504 Gateway Timeout`.

### `POST /api/chat/abort`
Auth: api-token.
Accepts an optional upstream request ID and attempts native cancellation only when the
active backend advertises a verified compatible contract. Browser Stop closes local
forwarding independently.

Request body (optional for backward compatibility):

```json
{ "request_id": "chatcmpl-123" }
```

Runtimes without verified native cancellation return an explicit local-only result:

```json
{
  "ok": true,
  "cancelled": false,
  "mode": "local_only",
  "reason": "native cancellation is unavailable for this runtime"
}
```

When native cancellation succeeds, `cancelled` is `true` and `mode` is `native`.

### `POST /api/chat/suggestions`
Auth: api-token.
Generates guided-generation suggestions using either supplied chat context or a fallback tab lookup.

This route uses the same monitor-side inference queue and busy/offline/timeout handling as `POST /api/chat`.

Request:

```json
{
  "tab_id": "tab_1746000000000",
  "category": "general",
  "count": 5,
  "context_depth": 10,
  "messages": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ],
  "system_prompt": "You are ...",
  "context_notes": [
    { "section": "plot", "content": "..." }
  ],
  "quick_guide_active": "",
  "prompt": null
}
```

Response:

```json
{
  "suggestions": ["..."],
  "cards": [],
  "category": "general",
  "count": 5
}
```

For `category: "director"`, `cards` can contain structured entries:

```json
{
  "type": "pressure",
  "title": "Tighten The Net",
  "effect": "More pressure now.",
  "detail": "Force the next reply into a narrower choice."
}
```

### `POST /api/keywords/generate`
Auth: api-token.

This route uses the same monitor-side inference queue and busy/offline/timeout handling as `POST /api/chat`.

```json
{ "category": "noir" }
```

Response:

```json
{ "keywords": ["rain", "diner", "tension"] }
```

### `POST /api/context-notes/analyze`
Auth: api-token.

This route uses the same monitor-side inference queue and busy/offline/timeout handling as `POST /api/chat`.

```json
{
  "messages": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ],
  "system_prompt": "...",
  "existing_notes": [
    { "section": "character", "content": "Kira is a cynical detective.", "created_at": 1746000000000 }
  ],
  "sections": ["character", "setting", "plot", "tone"]
}
```

Response:

```json
{
  "sections": [
    {
      "section": "character",
      "suggested": "Kira is now cooperating with the suspect.",
      "status": "stale",
      "reason": "Recent messages contradict the earlier note."
    }
  ]
}
```

`status` is `new`, `current`, or `stale`.

## Chat Persistence API

Route handlers: `src/web/api/chat/tabs.rs`.

The live chat persistence layer is SQLite-backed and centered on `chat.db`. Chat tabs are no longer stored as one big JSON array.

Chat tab management endpoints:

- `GET /api/chat/tabs` — list tab metadata (no messages) (api-token)
- `POST /api/chat/tabs` — create a tab (api-token)
- `GET /api/chat/tabs/:id` — full tab with messages (api-token)
- `PUT /api/chat/tabs/:id` — full tab save; replaces messages (api-token)
- `PATCH /api/chat/tabs/:id/meta` — metadata-only update; messages ignored (api-token)
- `POST /api/chat/tabs/:id/messages` — append messages (api-token)
- `PATCH /api/chat/tabs/order` — reorder tabs (api-token)
- `DELETE /api/chat/tabs/:id` — delete tab (api-token)
- `POST /api/chat/tabs/:id/archive` — archive a tab (api-token)
- `POST /api/chat/tabs/:id/hide` — hide a tab (api-token)
- `POST /api/chat/tabs/:id/restore` — restore a hidden/archived tab (api-token)
- `GET /api/chat/search` — full-text search across messages (api-token)

### `GET /api/chat/tabs`
Auth: api-token.
Returns tab metadata only, without message bodies.

```json
[
  {
    "id": "tab_1746000000000",
    "name": "Noir Scene",
    "explicit_level": 0,
    "active_template_id": null,
    "pinned": false,
    "tab_order": 0,
    "last_ctx_pct": null,
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "message_count": 12,
    "composer_draft": "",
    "created_at": 1746000000000,
    "updated_at": 1746000100000
  }
]
```

New field:

| Field | Type | Default |
|-------|------|---------|
| `composer_draft` | string | `""` |

`composer_draft` holds per-tab composer draft text. It is persisted to the backend on input, restored on tab switch or reload, and cleared on successful send.

### `POST /api/chat/tabs`
Auth: api-token.
Creates one tab. Request body is a full `ChatTabRow`. If `id` is empty, the server generates one. `created_at` and `updated_at` are overwritten server-side.

Response is the created tab object.

### `GET /api/chat/tabs/{id}`
Auth: api-token.
Returns the full tab row plus messages.

```json
{
  "id": "tab_1746000000000",
  "name": "Noir Scene",
  "system_prompt": "",
  "ai_name": null,
  "user_name": null,
  "explicit_level": 0,
  "active_template_id": null,
  "auto_compact": true,
  "auto_compact_summarize": false,
  "compact_mode": "percent",
  "compact_threshold": 0.8,
  "model_params": {},
  "context_notes": [],
  "sidebar_width": 280,
  "tab_order": 0,
  "pinned": false,
  "last_ctx_pct": null,
  "total_input_tokens": 0,
  "total_output_tokens": 0,
  "composer_draft": "",
  "created_at": 1746000000000,
  "updated_at": 1746000100000,
  "messages": [
    {
      "id": 1,
      "tab_id": "tab_1746000000000",
      "role": "user",
      "content": "Hello",
      "timestamp_ms": 1746000001000,
      "input_tokens": null,
      "output_tokens": null,
      "cumulative_input_tokens": null,
      "cumulative_output_tokens": null,
      "compaction_marker": false,
      "variants": null,
      "variant_index": null,
      "seq": 0
    }
  ]
}
```

### `PUT /api/chat/tabs/{id}`
Auth: api-token.
Full save for one tab. The handler updates tab metadata and then replaces all messages for that tab.

Response:

```json
{ "ok": true }
```

Important behavior:
- `id` is taken from the path.
- `updated_at` is overwritten server-side.
- message order is rewritten from array order
- during this replace path, only `role`, `content`, `timestamp_ms`, `input_tokens`, `output_tokens`, and `compaction_marker` are persisted for each message; cumulative token fields and variant fields are not preserved by this route

### `PATCH /api/chat/tabs/{id}/meta`
Auth: api-token.
Metadata-only update for one tab. Request body uses the same tab shape, but `messages` are ignored.

### `POST /api/chat/tabs/{id}/messages`
Auth: api-token.
Appends one or more messages.

Request:

```json
{
  "messages": [
    {
      "tab_id": "ignored-and-overwritten",
      "role": "assistant",
      "content": "Hello back",
      "timestamp_ms": 1746000002000,
      "input_tokens": 10,
      "output_tokens": 25,
      "cumulative_input_tokens": 10,
      "cumulative_output_tokens": 25,
      "compaction_marker": false,
      "variants": null,
      "variant_index": null,
      "seq": 999
    }
  ]
}
```

Response:

```json
{ "ok": true, "last_id": 42 }
```

Important behavior:
- `tab_id` in each message is overwritten from the path
- `seq` is ignored and assigned automatically
- this append route does persist `cumulative_input_tokens`, `cumulative_output_tokens`, `variants`, and `variant_index`

### `PATCH /api/chat/tabs/order`
Auth: api-token.

```json
{
  "tab_order": ["tab_a", "tab_b", "tab_c"]
}
```

Response:

```json
{ "ok": true }
```

### `DELETE /api/chat/tabs/{id}`
Auth: api-token.

```json
{ "ok": true }
```

## Chat Search

### `GET /api/chat/search`
Auth: api-token.
Full-text search over non-compaction-marker messages in `chat.db`.

Query params:
- `q`: required search string
- `limit`: optional page size, default `20`, max `100`
- `offset`: optional result offset, default `0`
- `visibility`: optional comma-separated visibility filter (e.g., `public,private`)
- `tab_id`: optional tab ID — when present, restricts results to that single conversation

Example (app-wide search):

```text
/api/chat/search?q=slow%20endpoint&limit=20&offset=0
```

Example (scoped to one tab):

```text
/api/chat/search?q=dragon&tab_id=tab_1746000000000
```

Response:

```json
{
  "results": [
    {
      "tab_id": "tab_1746000000000",
      "tab_name": "Debug Session",
      "message_id": 17,
      "role": "assistant",
      "snippet": "Check <mark>slow</mark> HTTP <mark>endpoint</mark> first.",
      "timestamp_ms": 1746000002000
    }
  ],
  "total": 37,
  "limit": 20,
  "offset": 0,
  "has_more": true
}
```

Search notes:
- punctuation is normalized before FTS lookup
- prefix matching is used internally
- empty or unparseable queries return an empty paged result object

## Sleep Mode

Route handlers: `src/web/api/sleep.rs`.

Sleep mode is a 3-state system: off, logs-only, sleep.

- `off` — normal monitoring.
- `logs-only` — live log tail and console output enabled, but heavy metrics, GPU/CPU temp probes, and most network calls are disabled. Pollers slow to sleep-like intervals.
- `sleep` — full sleep; only minimal background (log tail if configured).

All sleep-mode endpoints require `api-token`.

### `GET /api/sleep-mode`
Returns current sleep-mode status.

```json
{
  "mode": "off",
  "enabled": false,
  "config": {
    "auto_sleep_enabled": true,
    "idle_threshold_secs": 120
  }
}
```

- `mode` is one of: `"off"`, `"logs-only"`, `"sleep"`.
- `enabled` is `true` when `mode != "off"`.

### `POST /api/sleep-mode/toggle`
Cycles the mode. Behavior depends on how the current mode was set:

- If set by the user (manual), cycles: `off` → `logs-only` → `sleep` → `off`.
- If set automatically (auto-sleep), cycles: `off` ↔ `sleep` (skips `logs-only`).

Response:

```json
{
  "ok": true,
  "mode": "logs-only",
  "enabled": true,
  "sleep_mode": true
}
```

### `POST /api/sleep-mode/set`
Explicitly set the mode.

- Supports new `"mode"` field:
  - `{ "mode": "off" }`
  - `{ "mode": "logs-only" }`
  - `{ "mode": "sleep" }`
- For backward compatibility, the legacy `{ "enabled": true/false }` shape maps:
  - `true` → `"sleep"`
  - `false` → `"off"`

Response same as GET, with updated values.

## Database Admin

Route handlers: `src/web/api/db.rs`.

All `/api/db/*` routes operate on the SQLite chat database.

Authentication:
- `GET /api/auth/status`:
  - Public endpoint used by the frontend to discover whether auth is enabled.
  - Returns enabled methods, whether auth is managed by startup flags, a local recovery command,
    and whether the current request is already authenticated.
- `POST /api/auth/login`:
  - Public endpoint used by in-app form auth.
  - Expects JSON:
      { "username": "<user>", "password": "<password>" }
  - On success, sets an HttpOnly session cookie.
- `POST /api/auth/logout`:
  - Public endpoint that clears the in-app form-auth session cookie.
- `GET /api/auth/config`:
  - Protected endpoint for the Security tab.
  - Requires `api-token`.
  - Returns the current dashboard auth mode, username, whether it is CLI-managed, and local recovery metadata.
- `PUT /api/auth/config`:
  - Protected endpoint for updating dashboard auth.
  - Requires `api-token`.
  - Expects JSON:
      {
        "basic_enabled": true|false,
        "form_enabled": true|false,
        "username": "<user>",
        "current_password": "<old-password-or-empty>",
        "new_password": "<new-password-or-empty>"
      }
  - Use `new_password` when setting a password for the first time or changing it.
  - Sending both mode flags as `false` disables dashboard auth and clears `auth-config.json`.
- Most endpoints require one of two tokens:
  - `api-token`: general API token for routine admin operations.
  - `db-admin-token`: elevated token for destructive or high-risk operations.
- Tokens are provided via:
  - Header: `Authorization: Bearer <token>`
- `GET /api/db/admin-token`:
  - Returns the db-admin-token for use by the in-browser DB Admin UI.
  - Allowed whenever a request has already passed configured auth, or when the server is bound to loopback with no auth configured.
  - Example:
      { "token": "<db-admin-token>" }
- `GET /api/internal/api-token` follows the same bootstrap policy for the main UI.

### `GET /api/db/stats`
Requires `api-token`.

```json
{
  "tab_count": 3,
  "message_count": 248,
  "fts_index_count": 248
}
```

### `GET /api/db/integrity`
Requires `api-token`.

```json
{
  "status": "healthy",
  "detail": "ok"
}
```

If `detail` is not `"ok"`, `status` is `"corrupted"`.

### `POST /api/db/maintenance`
Requires `api-token`.

```json
{ "operation": "checkpoint" }
```

Supported operations:
- `checkpoint`
- `vacuum`
- `rebuild_fts`
- `analyze`

Responses:

```json
{ "backfilled": 0, "deleted": 0, "log": 0 }
```

```json
{ "status": "vacuumed" }
```

```json
{ "status": "fts_rebuilt" }
```

```json
{ "status": "analyzed" }
```

### `POST /api/db/backup`
Requires `api-token`.

Creates a manual backup in `~/.config/llama-monitor/backups/chat_<timestamp>.db`.

```json
{
  "status": "backup_created",
  "name": "manual/chat_1746000000000.db",
  "size_bytes": 40960
}
```

Manual backups are pruned to the 7 newest `chat_*.db` files.

### `DELETE /api/db/backup`
Requires `db-admin-token`.

```json
{
  "backup_name": "chat_1746000000000.db"
}
```

Response:

```json
{
  "status": "deleted",
  "backup": "chat_1746000000000.db"
}
```

### `GET /api/db/backups`
Requires `api-token`.

Lists both manual backups (`chat_*.db`) and automatic hourly backups (`chat_auto_*.db`).

```json
{
  "backups": [
    {
      "name": "chat_auto_1746000000000.db",
      "size": 40960,
      "modified": 1746000000000
    }
  ],
  "total_size": 40960,
  "truncated": false
}
```

### `POST /api/db/restore`
Requires `db-admin-token`.

Validates backup_name:
- No path traversal (no ‘..’, leading ‘/’, or backslashes).
- Resolved path must be inside the backups directory.

Before restore, creates a safety backup as pre_restore_<timestamp>.db.
After restore, runs an integrity check; returns a note if it fails.

```json
{
  "backup_name": "chat_1746000000000.db"
}
```

Success:

```json
{
  "status": "restored",
  "backup": "chat_1746000000000.db"
}
```

Before restore, the server creates `pre_restore_<timestamp>.db` in the same `backups/` directory.

### `POST /api/db/repair`
Requires `db-admin-token`.

```json
{ "operation": "repair_indexes" }
```

Supported operations:
- `repair_indexes`
- `emergency_recovery`

Responses:

```json
{ "status": "indexes_repaired" }
```

```json
{ "status": "recovery_attempted" }
```

### `GET /api/db/indexes`
Requires `api-token`.

```json
[
  {
    "name": "idx_messages_tab",
    "table": "messages",
    "sql": "CREATE INDEX IF NOT EXISTS idx_messages_tab ON messages(tab_id, seq)",
    "rebuildable": false
  }
]
```

The metadata response is bounded to 256 backup entries. `truncated: true` means additional owned files were not included; it never exposes backup contents or filesystem paths.

### `POST /api/db/query`
Requires `api-token` or `db-admin-token` (dual-token).

Safeguards:
- Max body size: 256 KB; max SQL length: 16 KB; execution timeout: 10 seconds.
- Query results are capped at 1,000 rows and 1 MiB of serialized row data; add a `LIMIT` clause or narrow selected columns when a query exceeds either bound.
- Single-statement only: a semicolon anywhere in the SQL is rejected via a simple substring
  scan (`;` present → "Multi-statement queries are not allowed"). This is a lightweight and
  fragile guard, not full parsing.
- Only allows: `SELECT`, `VACUUM`, `ANALYZE`, and a restricted PRAGMA subset (see below).
- DML (INSERT/UPDATE/DELETE/REPLACE), DDL (CREATE/DROP/ALTER), ATTACH, LOAD_EXTENSION,
  and WAL checkpoint PRAGMAs are blocked.

PRAGMA allowlist:
- The code in `chat_storage.rs` uses an explicit allowlist (e.g. `INTEGRITY_CHECK`,
  `QUICK_CHECK`, `PAGE_COUNT`, `FREELIST_COUNT`, `SCHEMA_VERSION`, `USER_VERSION`,
  `INDEX_LIST`, `INDEX_INFO`, `TABLE_INFO`, `TABLE_XINFO`, `FOREIGN_KEY_LIST`,
  `LOCK_LIST`, `DATABASE_LIST`, `JOURNAL_MODE`, `SYNCRONOUS`, `CACHE_SIZE`,
  `CACHE_SPILL`, `WAL_AUTOCHECKPOINT`, `AUTOVACUUM`, `INCREMENTAL_VACUUM`,
  `SECURE_DELETE`, `TEMP_STORE`, `MMAP_SIZE`, `QUERY_ONLY`, `EPOCHMS`, etc.).
- `writable_schema` is NOT in the allowlist, so it is blocked.
- Several write-affecting PRAGMAs (e.g., `INCREMENTAL_VACUUM`, `AUTOVACUUM`, `SECURE_DELETE`)
  are permitted; the allowlist is not strictly "read-only."

Auth behavior:
- With `api-token`:
  - `SELECT` queries are allowed but with a relaxed column filter: sensitive columns
    (message content, `system_prompt`, `context_notes`, `model_params`) are blocked unless
    you use `db-admin-token`.
  - `PRAGMA`/`VACUUM`/`ANALYZE` allowed as-is.
- With `db-admin-token`:
  - Same allowed operations, but no column restrictions on `SELECT`.

Example request:

```json
{
  "sql": "SELECT id, name FROM tabs ORDER BY updated_at DESC LIMIT 10"
}
```

Response:

```json
{
  "columns": ["id", "name"],
  "rows": [
    { "id": "tab_1", "name": "Noir Scene" }
  ],
  "row_count": 1
}
```

## TLS / ACME

Route handlers: `src/web/api/tls.rs`.

Endpoints for managing TLS and ACME-based certificate provisioning.

Authentication:
- GET /api/tls/config: requires api-token.
- PUT /api/tls/config: requires api-token.
- POST /api/tls/acme/request: requires api-token.
- POST /api/tls/acme/renew: requires api-token.
- All TLS/ACME endpoints reject requests without a valid Bearer api-token.

### `GET /api/tls/config`
Returns the current TLS mode and ACME summary (without secrets).

Header:
- `Authorization: Bearer <api-token>`

Response:

```json
{
  "mode": "acme",
  "customCertPath": null,
  "customKeyPath": null,
  "acme": {
    "enabled": true,
    "fqdn": "llama-monitor.example.com",
    "environment": "staging",
    "dnsProvider": "cloudflare",
    "validationDelay": 300,
    "lastRenewal": null,
    "certPath": "/path/to/cert.pem",
    "keyPath": "/path/to/key.pem"
  }
}
```

`mode` can be:
- `"none"`
- `"self-signed"`
- `"custom"`
- `"acme"`

### `PUT /api/tls/config`
Updates TLS configuration, including ACME settings.

Header:
- `Authorization: Bearer <api-token>`
- `Content-Type: application/json`

Example body (ACME):

```json
{
  "mode": "acme",
  "acme": {
    "enabled": true,
    "fqdn": "llama-monitor.example.com",
    "environment": "staging",
    "dnsProvider": "cloudflare",
    "dnsConfig": {
      "CF_API_TOKEN": "your-token-here"
    },
    "validationDelay": 300
  }
}
```

Response:

```json
{ "ok": true }
```

### `POST /api/tls/acme/request`
Triggers an ACME certificate request.

Header:
- `Authorization: Bearer <api-token>`

Response:

```json
{ "status": "requested" }
```

### `POST /api/tls/acme/renew`
Triggers an ACME certificate renewal.

Header:
- `Authorization: Bearer <api-token>`

Response:

```json
{ "status": "renewed" }
```

## System and Hardware

### `POST /api/system/set-metal-gpu-limit`
( macOS only ) Adjust Metal GPU wired memory limit via `sysctl iogpu.wired_limit_mb`.

- Auth: `db-admin-token` (elevated, system-level change).
- Requires: macOS with administrator privileges (triggers a password prompt via `osascript`).
- Request: `{ "limit_mb": 40960 }`
- If not on macOS, returns:
  `{ "ok": false, "error": "Metal GPU limit tuning is only available on macOS." }`

### Memory-pressure telemetry

Branch-specific addition.

`SystemMetrics` (reported via WebSocket telemetry and used internally for monitoring) includes cross-platform memory-pressure fields. macOS reads the Mach `host_statistics64` syscall plus `kern.memorystatus_vm_pressure_level`/`vm.swapusage` sysctls (no subprocess), Linux uses `/proc/pressure/memory` plus `/proc/meminfo`, and Windows uses `GlobalMemoryStatusEx` (via sysinfo). The normalized `memory_pressure_score` is band-aligned across platforms: 0-50 = ok, 50-80 = warning, 80-100 = critical.

| Field | Type | Description |
|-------|------|-------------|
| `ram_available_gb` | float | Available RAM in GB |
| `memory_pressure_level` | string | `"ok"`, `"warning"`, or `"critical"` |
| `memory_pressure_source` | string | Platform source such as `"host_statistics64"`, `"linux_psi"`, or `"windows_memstatus"` |
| `memory_pressure_score` | float | Normalized pressure score from 0-100 |
| `memory_free_gb` | float | Free pages in GB |
| `memory_reclaimable_gb` | float | Estimated reclaimable cache/inactive memory GB |
| `memory_compressor_gb` | float | Compressor in-use GB |
| `memory_compressed_gb` | float | Total compressed data GB |
| `swap_used_gb` | float | Swap/pagefile currently used when available |
| `swapins` | integer | Cumulative swap-ins |
| `swapouts` | integer | Cumulative swap-outs |
| `swapins_delta` | integer | Swap-ins since previous sample when available |
| `swapouts_delta` | integer | Swap-outs since previous sample when available |
| `memory_psi_some_avg10` | float | Linux PSI `some avg10` stall percent |
| `memory_psi_full_avg10` | float | Linux PSI `full avg10` stall percent |
| `memory_pressure_advice` | string | Short platform-aware action hint |

The UI uses this data for:
- A Memory Pressure sparkline and metric in the system card.
- A memory-pressure pill in the top navigation bar (shown when warning/critical).
- Contextual advice that distinguishes wired/pinned memory from reclaimable cache.

Unsupported platforms keep these fields present with safe defaults.

## Llama.cpp Binary Management

Route handlers: `src/web/api/llama_binary.rs`.

Endpoints to manage llama-server binary.

### `GET /api/llama-binary/platform-info`
Return current platform and backend information.

- Auth: `api-token`.
- `rapid_mlx_local_available` is true only for `macOS` on `aarch64`.
- `rapid_mlx_local_requirement` provides the user-facing platform requirement used to
  gate local configure actions. This does not prohibit attaching to a remote endpoint.

### `GET /api/llama-binary/latest`
Return the latest installable nightly (`b<digits>`) llama.cpp release for the
current platform.

- Auth: `api-token`.
- Metadata-only versioned releases (e.g. `v0.2.0`, which ships only
  `nightly-tag.txt`) are skipped.
- Successful responses are cached for 30 minutes; error responses are not
  cached, so a nightly published after an error is visible on the next poll.

### `GET /api/llama-binary/releases`
Return recent llama.cpp releases, newest first by GitHub publication timestamp. Each item includes:

- `installable` — a binary asset exists for the current platform's default backend (the backend the version picker installs with).
- `has_binaries` — the release publishes at least one binary archive, as opposed to metadata-only versioned releases. When `installable` is false and `has_binaries` is true, no build exists for this platform.

When the newest-8 window contains no versioned (stable) release, the latest
stable release is appended with `stable: true` so the UI can pin it below the
window.

- Auth: `api-token`.

### `GET /api/llama-binary/version`
Return the currently installed llama-server binary version.

- Auth: `api-token`.

### `POST /api/llama-binary/update`
Download and install a specific llama.cpp release.

- Auth: `api-token`.
- Request: `{ "tag": "b5700", "backend": "metal" }` — `tag` selects a specific
  release; omit it to auto-select the newest release that ships an asset for
  the requested `backend` (the platform default when `backend` is omitted).
- Behavior:
  - Downloads release archive to `bin/`.
  - Applies `chmod 755` on Unix.
  - Copies all files (CUDA/Vulkan/SYCL builds require co-located libs).

### `POST /api/llama/restart`
Restart a locally running llama-server with the current binary (useful after installing a new version).

- Auth: `api-token`.
- Precondition:
  - A local llama-server must be running (`local_server_running: true`).
- Behavior:
  - Captures the current `ServerConfig` from AppState.
  - Calls `stop_server()` (kills process, clears child/metrics).
  - Calls `start_server()` with the captured config.
  - The restarted server uses the current `llama_server_path` (so after a binary update, it will use the new build).
- Errors:
  - 200 with `ok: false` if:
    - No local server is running.
    - No saved server config found.
  - Stop or restart fails.

## Rapid-MLX Runtime Management

Route handlers: `src/web/api/rapid_mlx_runtime.rs`.

Rapid-MLX is a supported inference backend for macOS Apple Silicon. Its runtime can be
managed in-place: installed, upgraded, repaired, and rolled back using isolated, immutable
environments. These endpoints:

- Do not touch Homebrew, system Python, or external package managers.
- Use a single-mutation model: only one runtime operation may run at a time.
- Never modify a running environment in place; new versions are built separately and
  activated after validation.
- Always retain the previous known-good environment to allow rollback.
- Never expose local filesystem paths in public JSON payloads.

Mutations require `db-admin-token` and an explicit confirmation token.
Read endpoints (status, releases, jobs) require `api-token`.

Inventory entries expose `source_kind` as `release` or `git`. The following
fields are present only for `git` development-source entries:

- `source_repository`: GitHub `owner/repository`;
- `source_ref`: the branch, tag, PR reference, or SHA entered by the user; and
- `source_commit`: the resolved immutable 40-character commit SHA.

### `POST /api/rapid-mlx/runtime/development/resolve`

Auth: `api-token`.

Request:

```json
{
  "repository": "nmorgowicz/Rapid-MLX",
  "reference": "pr:2140",
  "extras": ["guided", "vision"]
}
```

The response contains `source.repository`, `source.requested_ref`,
`source.resolved_commit`, `source.commit_url`, and optional PR title/base
metadata. Resolution is a lookup only; it does not install anything.

### `POST /api/rapid-mlx/runtime/development/install`

Auth: `db-admin-token`; confirmation: `INSTALL_RAPID_MLX_DEVELOPMENT`.

Request must repeat the resolved source and include the exact SHA returned by
the resolve endpoint:

```json
{
  "repository": "nmorgowicz/Rapid-MLX",
  "reference": "pr:2140",
  "resolved_commit": "0123456789abcdef0123456789abcdef01234567",
  "extras": ["guided", "vision"],
  "confirm": "INSTALL_RAPID_MLX_DEVELOPMENT"
}
```

Installation rejects requests without `resolved_commit`; it never silently
re-resolves a moving branch. The response uses the same asynchronous `job_id`
contract as release installation. A new commit creates a new immutable managed
environment and must be requalified for MTP/speculative decoding.

### `GET /api/rapid-mlx/runtime/status`

Auth: `api-token`.

Returns the current managed runtime status, inventory, and recent jobs.

Response:

```json
{
  "runtime": {
    "supported": true,
    "installer_available": true,
    "mutation_in_progress": false,
    "rollback_available": true,
    "active": {
      "environment_id": "rapid-mlx-0.10.10",
      "version": "0.10.10",
      "release_channel": "stable",
      "active": true,
      "rollback_candidate": false,
      "complete": true
    },
    "inventory": [
      {
        "environment_id": "rapid-mlx-0.10.10",
        "version": "0.10.10",
        "release_channel": "stable",
        "active": true,
        "rollback_candidate": false,
        "complete": true
      },
      {
        "environment_id": "rapid-mlx-0.10.9",
        "version": "0.10.9",
        "release_channel": "stable",
        "active": false,
        "rollback_candidate": true,
        "complete": true
      }
    ]
  },
  "jobs": [
    {
      "id": "a1b2c3d4e5f6...",
      "operation": "install",
      "state": "complete",
      "message": "Runtime validated and activated",
      "version": "0.10.10",
      "result": {
        "active": {
          "environment_id": "rapid-mlx-0.10.10",
          "version": "0.10.10",
          "release_channel": "stable",
          "active": true,
          "rollback_candidate": false,
          "complete": true
        },
        "previous_environment_id": "rapid-mlx-0.10.9"
      }
    }
  ]
}
```

Fields:

- `supported`: true if the platform is Apple Silicon macOS (Rapid-MLX requirement).
- `installer_available`: true if `uv` or equivalent installer is available.
- `mutation_in_progress`: true while an install/upgrade/repair/rollback job is active.
- `rollback_available`: true if a previous known-good environment is retained.
- `active`: current managed runtime or null.
- `inventory`: retained environments (up to internal limit), each with active/rollback
  candidate flags and immutable `extras` profile.
- `extras`: optional capability bundles (`guided`, `vision`, `embeddings`, `audio`);
  core is implicit. Omit on install/upgrade requests to use the recommended
  `guided` + `vision` profile.
- `jobs`: up to 16 most recent runtime operations.

If the manager cannot initialize, the endpoint returns 500 with:

```json
{ "ok": false, "error": "Managed Rapid-MLX storage is unavailable" }
```

### `GET /api/rapid-mlx/runtime/releases`

Auth: `api-token`.

Lists recent non-draft Rapid-MLX releases discovered from the upstream GitHub release
feed. Used for selecting a version in install/upgrade. Responses are cached for 5 minutes.

Response:

```json
{
  "releases": [
    {
      "version": "0.10.10",
      "tag": "v0.10.10",
      "channel": "stable",
      "published_at": "2026-07-16T00:00:00Z"
    },
    {
      "version": "0.10.11rc1",
      "tag": "v0.10.11rc1",
      "channel": "prerelease",
      "published_at": "2026-07-17T00:00:00Z"
    }
  ]
}
```

Notes:

- Draft releases are excluded.
- Versions below the configured floor are filtered out.
- Prerelease channels must be selected explicitly and must match between request and
  release metadata.

If GitHub is unreachable or exceeds response bounds, returns 502:

```json
{ "ok": false, "error": "Rapid-MLX release discovery is temporarily unavailable" }
```

### `POST /api/rapid-mlx/recommend`

Auth: `api-token`.

Returns a deterministic backend recommendation based on artifact type, platform, and
runtime compatibility.

Request:

```json
{
  "artifact_kind": "mlx_directory"
}
```

Accepted `artifact_kind` values:

- `gguf`: always recommends `llama_cpp`.
- `mlx_directory`, `authoritative_safetensors`, `rapid_mlx_hf_repository`, `rapid_mlx_alias`:
  recommends `rapid_mlx` if platform is Apple Silicon and a compatible runtime is present;
  otherwise indicates runtime is required or platform is unavailable.
- `unknown`: returns `manual_selection` with no recommendation.

Response (ready):

```json
{
  "recommended_backend": "rapid_mlx",
  "state": "ready",
  "reason": "This source is native to the verified Rapid-MLX resolution path."
}
```

Response (GGUF):

```json
{
  "recommended_backend": "llama_cpp",
  "state": "ready",
  "reason": "GGUF runs natively with llama.cpp."
}
```

Response (platform missing):

```json
{
  "recommended_backend": null,
  "state": "platform_unavailable",
  "reason": "Local Rapid-MLX requires Apple Silicon macOS; remote attachment remains available."
}
```

Response (runtime required):

```json
{
  "recommended_backend": null,
  "state": "runtime_required",
  "reason": "This model is Rapid-MLX compatible, but a compatible local runtime is required."
}
```

Response (unknown):

```json
{
  "recommended_backend": null,
  "state": "manual_selection",
  "reason": "Choose an engine after selecting a typed model source."
}
```

### `POST /api/rapid-mlx/runtime/install`

Auth: `db-admin-token`.

Installs the specified Rapid-MLX version as a managed runtime. Requires Apple Silicon macOS;
returns 400 if not available.

Request:

```json
{
  "version": "0.10.10",
  "channel": "stable",
  "extras": ["guided", "vision"],
  "confirm": "INSTALL_RAPID_MLX_RUNTIME"
}
```

Rules:

- `version` must match a published non-draft release exactly.
- `channel` must match (`stable` or `prerelease`).
- `confirm` must be exactly `INSTALL_RAPID_MLX_RUNTIME`.
- Rejects unknown or extra fields with 400.

If another runtime operation is in progress: 429

```json
{ "ok": false, "error": "Another managed Rapid-MLX runtime operation is already in progress" }
```

If platform not supported: 400

```json
{ "ok": false, "error": "Managed Rapid-MLX runtime changes require Apple Silicon macOS" }
```

Success: 202

```json
{ "job_id": "a1b2c3d4e5f6...", "state": "queued" }
```

Track progress via `GET /api/rapid-mlx/runtime/jobs/{job_id}`.

### `POST /api/rapid-mlx/runtime/upgrade`

Auth: `db-admin-token`.

Upgrades to a newer Rapid-MLX version by installing a second environment, validating it,
then activating it. The previous environment is kept as rollback candidate; it is never
modified in place.

Request uses the same shape as install, but with its own confirmation token:

```json
{
  "version": "0.10.11",
  "channel": "stable",
  "extras": ["guided", "vision"],
  "confirm": "UPGRADE_RAPID_MLX_RUNTIME"
}
```

Notes:

- An active inference session is not restarted or rerouted automatically.
- Same concurrency, versioning, and platform rules as install.
- Returns 202 with `job_id` on success.

### `POST /api/rapid-mlx/runtime/repair`

Auth: `db-admin-token`.

Rebuilds the currently active release in a new isolated environment and activates it,
used when the managed runtime becomes corrupted or incomplete.

Request:

```json
{
  "confirm": "REPAIR_RAPID_MLX_RUNTIME"
}
```

Rules:

- The active runtime must be verifiable against published release metadata.
- If not verifiable, returns 400:
  `{ "ok": false, "error": "The active managed Rapid-MLX runtime could not be verified against published release metadata" }`
- Same concurrency rules; returns 202 with `job_id` on success.

### `POST /api/rapid-mlx/runtime/rollback`

Auth: `db-admin-token`.

Restores the previous known-good managed runtime when available.

Request:

```json
{
  "confirm": "ROLLBACK_RAPID_MLX_RUNTIME"
}
```

Rules:

- No-op if no previous known-good environment exists:
  `{ "ok": false, "error": "No previous known-good Rapid-MLX runtime is available" }`
- Revalidates that environment before activation.
- Returns 202 with `job_id` on success.

### `GET /api/rapid-mlx/runtime/jobs/{id}`

Auth: `api-token`.

Returns the state of a specific runtime operation job.

Response:

```json
{
  "id": "a1b2c3d4e5f6...",
  "operation": "install",
  "state": "complete",
  "message": "Runtime validated and activated",
  "version": "0.10.10",
  "result": {
    "active": {
      "environment_id": "rapid-mlx-0.10.10",
      "version": "0.10.10",
      "release_channel": "stable",
      "active": true,
      "rollback_candidate": false,
      "complete": true
    },
    "previous_environment_id": "rapid-mlx-0.10.9"
  }
}
```

Fields:

- `operation`: one of `install`, `upgrade`, `repair`, `rollback`.
- `state`: one of `queued`, `running`, `complete`, `failed`.
- `message`: path-safe human-readable status; never exposes internal filesystem paths.
- `version`: version associated with the operation (if applicable).
- `result`: present on success; contains `active` runtime metadata and optional
  `previous_environment_id`.

If the job is not found, returns 404:

```json
{ "ok": false, "error": "Runtime operation was not found" }
```

## Remote Agent

Route handlers: `src/web/api/remote_agent.rs`.

All `/api/remote-agent/*` endpoints require a bearer token.

- api-token: `Authorization: Bearer <api-token>`
- db-admin-token: `Authorization: Bearer <db-admin-token>`

Without a valid token, endpoints return 401 with `{ "ok": false, "error": "unauthorized; <token-type> required" }`.

### `GET /api/remote-agent/releases/latest`

- Auth: api-token.
- Returns the latest GitHub release and assets.

### `POST /api/remote-agent/detect`

- Auth: api-token.
- Body: `{ ssh_target, ssh_connection?, agent_url? }`
- Detects remote OS, architecture, installed version, and matching release asset via SSH.

### `POST /api/remote-agent/ssh/host-key`

- Auth: api-token.
- Body: `{ ssh_target, ssh_connection }`
- Scans the SSH host key for the given target.

### `POST /api/remote-agent/ssh/trust`

- Auth: api-token.
- Body: `{ ssh_target, ssh_connection, key_hex }`
- Trusts the scanned host key for future SSH operations.

### `POST /api/remote-agent/status`

- Auth: api-token.
- Body: `{ ssh_target, ssh_connection? }`
- Checks the status of a managed remote agent (installed, running, task health).

### `POST /api/remote-agent/start`

- Auth: api-token.
- Body: `{ ssh_target, ssh_connection?, install_path?, start_command? }`
- Starts the remote agent on the target host.

### `POST /api/remote-agent/update`

- Auth: api-token.
- Body: `{ ssh_target, ssh_connection?, agent_url? }`
- Stops the existing agent, installs the latest release, and restarts it.

### `POST /api/remote-agent/stop`

- Auth: api-token.
- Body: `{ ssh_target, ssh_connection? }`
- Stops the remote agent process.

### `POST /api/remote-agent/install`

- Auth: db-admin-token (elevated).
- Body: `{ ssh_target, ssh_connection?, asset, install_path? }`
- Installs the remote agent binary and writes a `remote-agent-config.json` with the api-token.

### `POST /api/remote-agent/remove`

- Auth: db-admin-token (elevated).
- Body: `{ ssh_target, ssh_connection? }`
- Removes the managed remote agent (stops process, deletes startup task, removes binary).

### `GET /api/remote-agent/tls-status`

- Auth: api-token.
- Returns mTLS certificate status (CA, server, client).

For full details on the remote agent flow, see [Remote Agent](remote-agent.md).

## Setup Wizard (V2)

Route handlers: `src/web/api/spawn_wizard.rs`, `src/web/api/vram.rs`, `src/web/api/benchmark.rs`.

Endpoints supporting the setup wizard, VRAM estimation, and model discovery.

All require `api-token` unless noted.

### VRAM Estimation

- `POST /api/vram/estimate` — quick VRAM estimate for a configuration.
- `POST /api/vram-estimate` — full VRAM breakdown:
  - Request: optional `gpu_layers` (dense GPU layer count), `available_ram_bytes` (system RAM budget).
  - Response: `weights_bytes`, `kv_cache_bytes`, `linear_attn_state_bytes`, `mmproj_bytes`, `mtp_bytes`, `overhead_bytes`, `total_bytes`, `available_bytes`, `headroom_bytes`, `ram_bytes`, `available_ram_bytes`, `ram_headroom_bytes`, `recommendation`, `note`.
  - New recommendation: `WontFit` when CPU-offloaded weights exceed available system RAM.
- `POST /api/vram/quant-compare` — pre-download quant comparison table (Quant Advisor).
- `POST /api/vram/auto-size` — compute optimal configuration (context, KV quant, MoE tuning).

All require `api-token`.

### Llama.cpp Binary Management

Route handlers: `src/web/api/llama_binary.rs`.

- `GET /api/llama-binary/version` — currently installed binary version.
- `GET /api/llama-binary/latest` — latest available llama.cpp release.
- `GET /api/llama-binary/releases` — list of recent releases.
- `GET /api/llama-binary/release?build=N` — specific release info by build ID.
- `GET /api/llama-binary/platform-info` — current platform/backend info.
- `POST /api/llama-binary/update` — download and install a release.
  - Request: `{ "version": "b5700", "backend": "metal" }`
- `POST /api/llama/restart` — restart a locally running llama-server with the current binary (useful after installing a new version).

All require `api-token` unless noted.

### System Information

- Route handlers: `src/web/api/system_tools.rs` for top-processes and purge (no `/api` prefix).
- Route handlers: `src/web/api/vram.rs` for system info and Metal GPU limit.

- `GET /system/top-processes` — top CPU/memory-consuming processes (api-token).
- `POST /system/purge` — macOS-only manual cache purge (db-admin-token, request body `{ "confirm": "purge-memory" }`). Uses the native macOS authorization dialog and has a cooldown.
- `GET /api/system/info` — system/platform information (api-token).
- `GET /api/system/metal-gpu-limit` — current Metal GPU wired memory limit (macOS only, api-token).
- `POST /api/system/set-metal-gpu-limit` — adjust Metal GPU wired memory limit.
  - Requires `db-admin-token` (system-level change).
  - Request: `{ "limit_mb": 40960 }`

### Benchmark and Tuning

Route handlers: `src/web/api/benchmark.rs`.

All require `api-token`.

- `POST /api/benchmark` — run short inference benchmark against the running llama-server.
  - Supports optional tuning mode; 15-second cooldown.
- `POST /api/advise` — performance advice based on current preset and system info.
- `POST /api/moe-tune` — suggest MoE CPU expert offload settings.
- `POST /api/tune/ncpumoe` — tune n_cpu_moe for given constraints.
- `POST /api/model-defaults` — model-family sampling presets.
- `POST /api/bench/sweep` — benchmark sweep across configurations.
- `POST /api/bench/batch-sweep` — batched benchmark sweep.
- `POST /api/bench/mtp-sweep` — MTP-depth benchmark sweep.

### Chat Templates

External client setup profiles for OpenCode, Hermes, OpenClaw, and SillyTavern
are documented in [External client profiles](external-clients.md).

All require `api-token`.

- `POST /api/chat-template/fetch` — fetch a chat template URL and return its content.
- `POST /api/chat-template/upload` — upload a local chat template file.
- `GET /api/chat-template/dir` — list installed chat templates.
- `POST /api/chat-template/install-hf` — install a chat template from HF.
- `POST /api/chat-template/install-url` — install from a raw GitHub URL (1 MiB limit, no redirects).
- `GET /api/chat-template/active` — return the currently active template for a model family.
- `GET /api/chat-template/releases` — list known upstream releases/versions for a template.
- `POST /api/chat-template/activate` — switch the active template for a model family.
- `POST /api/chat-template/check-update` — check whether an installed template has an upstream update.
- `POST /api/chat-template/smoke-test` — run a tool-call smoke test against a candidate template.
- `POST /api/chat-template/transform` — apply a transform/patch to a template (e.g. Jinja fixups).
- `GET /api/chat-template/upstream-history` — list upstream commit/version history for a template source.
- `GET /api/chat-template/read` — read the raw content of an installed template.
- `GET /api/chat-template/discussions` — list HF discussion threads relevant to a model's chat template.
- `GET /api/chat-template/discussion-markdown` — fetch a discussion thread's content as markdown.
- `GET /api/chat-template/discussion-content` — fetch a discussion thread's raw content.
- `POST /api/chat-template/install-discussion` — install a chat template found in an HF discussion thread.

These endpoints back the chat-template lifecycle modal (install, activate, update-check, smoke-test, and Discussion-sourced templates), which auto-routes to Discussions based on template family and degrades gracefully for loose/untracked templates. Discussion create-fix candidates are activated only after the bounded two-fixture tool-call smoke test passes; Version History activation is an explicit rollback of an already-installed release.

### Sleep Mode

Route handlers: `src/web/api/sleep.rs`.

- `GET /api/sleep-mode` — returns current mode (`off`/`logs-only`/`sleep`), `enabled`, and `config`.
- `POST /api/sleep-mode/toggle` — cycle mode.
- `POST /api/sleep-mode/set` — explicitly set mode.

All require `api-token`.

### TLS / ACME Configuration

Route handlers: `src/web/api/tls.rs`.

All require `api-token`.

- `GET /api/tls/config` — current TLS mode and ACME summary.
- `PUT /api/tls/config` — update TLS/ACME config.
- `POST /api/tls/acme/request` — trigger certificate request.
- `POST /api/tls/acme/renew` — trigger certificate renewal.

### Config and Token Rotation

Route handlers: `src/web/api/config.rs`.

- `GET /api/settings` — masked `UiSettings` (api-token).
- `GET /api/settings/full` — unmasked `UiSettings` (api-token).
- `PUT /api/settings` — save `UiSettings` (api-token).
- `GET /api/gpu-env` — current GPU environment (api-token).
- `PUT /api/gpu-env` — update GPU environment (api-token).
- `POST /api/rotate-agent-token` — rotate remote-agent token (api-token).
- `POST /api/rotate-api-token` — rotate api-token (api-token).
- `POST /api/rotate-db-admin-token` — rotate db-admin-token (api-token).

All token-rotation endpoints update both the on-disk file and the in-memory `AppConfig` atomically; old tokens stop working immediately.

### Setup Wizard

Route handlers: `src/web/api/spawn_wizard.rs`.

All require `api-token`.

- `POST /api/spawn-wizard/mtp-draft-check` — validate MTP draft configuration.
- `POST /api/spawn-wizard/import-launch-file` — parse third-party launch script.

### Model Introspection

- `POST /api/model/introspect` — run `llama-server --print-model-metadata` on a local GGUF file (or cache).
  - Requires `api-token`; caches in `model-cache/<sha256>.json`; 30-second timeout.
- `POST /api/third-party-models` — scan local model directories (Ollama, LM Studio, etc.).
  - Requires `api-token`.

## HuggingFace Integration

Route handlers: `src/web/api/hf.rs`.

All endpoints require `api-token` unless noted.

### `POST /api/hf/search`
Search the HuggingFace Hub for GGUF models.

- Rate limit: 10 requests per 60 seconds.
- Sort options: `downloads` (default), `likes`, `trending`, `recent`.

### `POST /api/hf/author-models`
List GGUF models for a specific author.

### `GET /api/hf/community-picks`
Return the curated community picks list (used by the wizard's discover panel).

- Read: no auth (public).
- Update: `api-token` via `PUT` (if configured).

### `GET /api/hf/quantizers`
Return the tracked list of quantizer authors.

### `PUT /api/hf/quantizers`
Update quantizer author list.

### `POST /api/hf/files`
List GGUF files in a repo with sizes and quant labels.

### `GET /api/hf/card`
Fetch raw model card markdown by `repo` param.

- Auth: no auth (public).
- Query params: `repo` (e.g., `unsloth/Qwen3.5-27B-Q4_K_M-GGUF`).
- Used for in-app model card display.

### `GET /api/hf/token`
Check if HF token is set (masked).

- Auth: `api-token`.

Response:

```json
{ "has_token": true }
```

### `PUT /api/hf/token`
Set/update HF token (written to `hf-token` with mode 600).

- Auth: `api-token`.

Request:

```json
{ "token": "hf_..." }
```

Response:

```json
{ "ok": true }
```

### `DELETE /api/hf/token`
Remove stored HF token.

- Auth: `api-token`.

Response:

```json
{ "ok": true }
```

### `GET /api/hf/download-dir`
Return effective models download directory.

- Auth: `api-token`.

Response:

```json
{ "dir": "/Users/nick/.config/llama-monitor/models" }
```

## mlock Warnings (macOS)

Branch-specific addition (UI + preset editor only; no new endpoint).

When editing or creating a preset on macOS, if:
- `mlock` is enabled,
- and the VRAM estimate indicates the model is large relative to available memory,

the preset editor and spawn wizard display a warning that mlock pins model memory instead of letting the OS reclaim it, which can cause system unresponsiveness under pressure. This is client-side behavior triggered by existing VRAM and preset endpoints.

## LibreHardwareMonitor (LHM) (Windows-only)

These endpoints manage LibreHardwareMonitor on Windows. On non-Windows platforms they return a "Not supported on this platform" error.

### `GET /api/lhm/check`
No authentication required.
Returns whether LHM is installed and currently running.

Response:
```json
{ "running": true, "installed": true, "available": true }
```

On non-Windows:
```json
{ "running": false, "installed": false, "available": false, "error": "Not supported on this platform" }
```

### `POST /api/lhm/start`
Auth: api-token.
Starts LHM if installed.

Response:
```json
{ "success": true }
```

### `GET /api/lhm/progress`
No authentication required.
Returns the current LHM installation progress string (e.g. "not_started", "downloading", "installing", "completed").

Response:
```json
{ "progress": "completed" }
```

### `GET /api/lhm/status`
No authentication required.
Returns whether LHM is disabled via the persisted flag.

Response:
```json
{ "disabled": false }
```

### `POST /api/lhm/install`
Auth: api-token.
Downloads and installs LHM.

Response:
```json
{ "success": true }
```

### `POST /api/lhm/uninstall`
Auth: api-token.
Uninstalls LHM.

Response:
```json
{ "success": true }
```

### `POST /api/lhm/disable`
Auth: api-token.
Sets the LHM disabled flag.

Request:
```json
{ "disabled": true }
```

Response:
```json
{ "ok": true }
```

## Sensor Bridge (Windows-only)

Route handlers: `src/web/api/sensor_bridge.rs`.

Manages the local sensor-bridge service on Windows. On non-Windows platforms these endpoints return a "Not supported on this platform" error.

### `GET /api/sensor-bridge/status`
Auth: api-token.
Returns whether the sensor-bridge service is installed, running, and available.

Response:
```json
{ "installed": true, "running": true, "available": true }
```

### `POST /api/sensor-bridge/install`
Auth: api-token.
Installs the sensor-bridge service via UAC prompt.

Response:
```json
{
  "started": true,
  "message": "UAC prompt launched — approve it on your desktop to install the sensor service"
}
```

### `POST /api/sensor-bridge/uninstall`
Auth: api-token.
Uninstalls the sensor-bridge service via UAC prompt.

Response:
```json
{
  "started": true,
  "message": "UAC prompt launched — approve it on your desktop to remove the sensor service"
}
```

## Self-Update

Route handlers: `src/web/api/self_update.rs`, `src/agent.rs` (`self_update_binary`).

### `POST /api/self-update`

- Auth: requires `db-admin-token` (elevated operation).
- Body: { "confirm": "update" }.
- Cooldown: 5 minutes between calls; returns 429 with `seconds_remaining` if too soon.
- Behavior:
  - Fetches the latest release from GitHub and downloads the matching platform asset.
  - Replaces the running binary in-place.
  - Schedules exit(0) after a short delay, then performs an auto-restart using:
    - On Windows: a detached batch helper that waits for the old process to exit, copies the new binary into place, and relaunches it.
    - On Unix (macOS Apple Silicon, Linux): a detached shell launcher that restarts the new binary shortly after the old process exits.
  - If auto-restart fails for any reason, the user must manually relaunch.
- No signature or integrity check is performed; trust relies on HTTPS transport and the db-admin-token.
- Supported self-update targets:
  - macOS aarch64 (Apple Silicon only)
  - Linux x86_64
  - Linux aarch64
  - Windows x86_64

- Example request:
    { "confirm": "update" }
- Example success:
    { "ok": true, "tag_name": "v0.3.0", "restart_required": true }

## Kill-Llama

POST /api/kill-llama:
- Emergency kill for llama-server.
- Requires `db-admin-token` (elevated operation).
- Requires confirmation field: `{ "confirm": "kill" }`.
- Cooldown: 30 seconds between calls; returns 429 with `seconds_remaining` if too soon.
- Example:
    Request:  { "confirm": "kill" }
    Success:  { "ok": true }
    Too soon: { "error": "too soon; please wait", "seconds_remaining": 12 }

## Token Rotation

Route handlers: `src/web/api/config.rs`.

All three endpoints require `api-token` and return 401 if missing or invalid.

### `POST /api/rotate-agent-token`
Rotates the remote-agent token stored in `UiSettings`. Updates both on-disk and in-memory state, and notifies the agent-poll loop.

Response:
```json
{ "ok": true, "message": "Agent token rotated" }
```

### `POST /api/rotate-api-token`
Generates a new api-token, writes it to disk (encrypted if configured), and updates the live in-memory value atomically. Old token stops working immediately.

Response:
```json
{ "ok": true, "message": "API token rotated successfully." }
```

### `POST /api/rotate-db-admin-token`
Generates a new db-admin-token, writes it to disk (encrypted if configured), and updates the live in-memory value atomically. Old token stops working immediately.

Response:
```json
{ "ok": true, "message": "DB admin token rotated successfully." }
```

## Internal Token Bootstrap

### `GET /api/internal/api-token`
No authentication header is required.

Returns the api-token for use by the in-browser UI. Access is governed by a bootstrap policy:

- If "No Auth" is configured:
  - Always allowed (local-first mode).
- If any auth mode (basic or form) is configured:
  - Allowed only from:
    - Same-origin browser requests (via Origin checks), or
    - Loopback clients (127.0.0.1, localhost, ::1).
  - Otherwise returns 403.

Response:
```json
{ "token": "<api-token>" }
```

## WebSocket

For realtime metrics and capability pushes, use:

```text
ws://localhost:7778/ws
```

Limits:
- Maximum 50 concurrent connections.
- Excess connections are rejected with 429 Too Many Requests.

The WebSocket payload includes the following remote-agent fields:

| Field | Type | Description |
|-------|------|-------------|
| `remote_agent_connected` | boolean | Agent process is reachable |
| `remote_agent_health_reachable` | boolean | `/metrics` HTTP call succeeds; independent from `remote_agent_connected` |
| `remote_agent_protocol_too_old` | boolean | True when agent protocol version is below minimum enforced version |
| `remote_agent_protocol_version` | string or null | Agent's reported protocol version string |

`remote_agent_health_reachable` is set to true when the `/metrics` HTTP call succeeds and reset to false on disconnect. It is used to detect the firewall-blocked state (agent connected but health unreachable).

Platform memory-pressure fields are also emitted when present:
- `memory_pressure_level`, `memory_pressure_source`, `memory_pressure_score`, `memory_free_gb`,
  `memory_reclaimable_gb`, `memory_compressor_gb`, `memory_compressed_gb`, `swap_used_gb`,
  `memory_psi_some_avg10`, `memory_psi_full_avg10`, `memory_pressure_advice`.

See `docs/reference/realtime-communication.md` and `docs/reference/capabilities.md`.

## Legacy Chat Fields

The project previously used a flat-file chat format. The live SQLite-backed API no longer persists several legacy fields that still appear in old docs, old exports, or migration code.

Not part of the live `ChatTabRow` API:
- `ai_gender`
- `role_boundary_custom`
- `quick_guide_active`
- `armed_story_beats`
- `context_custom_sections`
- `quick_guide_draft`

Not part of the live `MessageRow` API:
- `thinking_content`
- `summarized`
- `dropped_count`
- `dropped_preview`
- `tokens_freed_estimate`
- `ctx_pct_before`
- `memory_version`
- `memory_domain`
- `summary_kind`
- `compacted_at`
- `compacted_message_count_total`
- `recent_tail_kept`

Compatibility note:
- startup migration from legacy `chat-tabs.json` still reads some of those older fields
- the live REST persistence API does not return or preserve them

## Errors

Most handlers return JSON error payloads rather than relying on HTTP status alone. Common shapes are:

```json
{ "ok": false, "error": "Preset not found" }
```

```json
{ "error": "No active session" }
```
## Application-home migration

These endpoints support the Local LLM Foundry 2.0 application-home upgrade.
They never move managed models or delete the legacy root automatically.

### `GET /api/app-home-migration/status`

Requires `api-token`. Returns the detected root state and canonical, legacy,
and active paths. Paths are metadata only; no directory is created.

### `GET /api/app-home-migration/preview`

Requires `api-token`. In `legacy_active` state, returns a deterministic plan id,
copy entries, retained entries, and byte totals. Other states return `plan:
null` and require recovery or explicit configuration rather than guessing.

### `POST /api/app-home-migration/queue`

Requires `db-admin-token` and JSON:

```json
{
  "plan_id": "<preview plan id>",
  "confirmation": "MIGRATE TO LOCAL LLM FOUNDRY"
}
```

The server re-plans before writing a sibling queue marker. A changed source,
destination collision, wrong confirmation, or stale plan is rejected. Queueing
does not migrate in the live process; the next maintenance restart executes the
copy-first plan, writes a receipt, and retains the legacy root for rollback.

### `POST /api/app-home-migration/rollback/preview`

Requires `api-token`. Returns a receipt-scoped rollback plan only when a verified
Stage A receipt exists and the retained legacy root is still present.

### `POST /api/app-home-migration/rollback/queue`

Requires `db-admin-token` and the exact confirmation string `ROLL BACK TO LLAMA
MONITOR`. Queues receipt-scoped canonical-root removal for the next controlled
restart; the legacy root is never modified.

### `POST /api/app-home-migration/cleanup`

Requires `db-admin-token` and the exact confirmation string `DELETE LEGACY ROOT
AFTER VERIFIED MIGRATION`. Queues removal of only the receipt-verified legacy
root for the next controlled restart; no inferred or external path is deleted.
