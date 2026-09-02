# oMLX Backend Integration Architecture

**Date:** 2026-09-02
**Author:** Nick M (with Claude Opus 5)
**Status:** Architecture contract. Design decisions resolved (§11); nine research gaps open (§13). Execution phases deferred until the preset bundle PR ships.
**Branch:** (not yet cut) — depends on `feature/preset-bundle-launch-options` landing first
**Last Verified:** 2026-09-02 against `docs/plans/20260830-preset_bundle_architecture.md`, `src/inference/`, and upstream oMLX/MTPLX source clones

> **For agentic workers:** this is an architecture contract, not an executable plan. It defines seams, invariants, and resolved decisions. Close §13's research gaps before attempting task decomposition. The task-by-task execution doc (with TDD steps and gates, per `superpowers:writing-plans`) is written *after* the preset bundle feature ships, and it must argue from this document.

**Goal:** Add oMLX as a first-class inference backend alongside llama.cpp and Rapid-MLX — not as a replacement for either — with a launch/capability model that generalizes to MTPLX as a third MLX loader.

**Architecture:** Extend the existing `InferenceBackend` / `BackendAdapter` enum seam with an `Omlx` variant and an `OmlxConfig` sidecar on `ModelPreset`. Introduce a second capability-evidence path for Python-package loaders, because §10's binary-SHA + `--help` probe does not describe oMLX. Preset cards gain no new axes; loader-specific semantics stay Editor-owned and reach the card only as eligibility-with-reason.

**Spec:** `docs/plans/20260830-preset_bundle_architecture.md` (binding), `docs/plans/20260830-preset_bundle_execution.md` (in flight)

---

## 1. Purpose and non-goals

### 1.1 Purpose

Make oMLX a peer runtime to Rapid-MLX. The user retains and continues to evolve the Rapid-MLX integration; oMLX is added beside it, and the design must not assume a single golden-path MLX loader. MTPLX is expected as a third peer, so every seam introduced here is designed for N MLX loaders, not two.

### 1.2 Why oMLX specifically

Two verified capabilities that Rapid-MLX does not have today:

1. **Exact (non-greedy) MTP acceptance.** `omlx/patches/mlx_lm_mtp/batch_generator.py` implements Leviathan & Chen stochastic acceptance with residual correction `max(p−q, 0)/Z`, so the marginal output distribution equals the target distribution at temperature > 0. This is the capability whose absence in Rapid-MLX is the standing complaint, and whose upstream PR remains unmerged.
2. **ANE hybrid prefill.** `omlx/patches/qwen35_ane_prefill.py` splits Qwen3.5/3.6/3.8 MLP gate/up and GDN qkv projections across the Apple Neural Engine(s), CPU, and GPU. MTPLX has **zero** ANE support — a source-wide search for `neural.?engine|coreml|AppleNeuralEngine` across `*.py`, `*.md`, `*.mm`, `*.metal`, `*.h` returns no matches.

**Verification trap, recorded so it is not re-litigated:** `omlx/custom_kernels/bonsai/fast.py:475` defines `spec_decode_verify` as a *greedy* argmax verify. That kernel serves Bonsai 1-bit/2-bit models only and is not the MTP path. Grepping for "greedy" and stopping there produces exactly the wrong conclusion about oMLX.

### 1.3 Non-goals

- Not a migration. Rapid-MLX presets keep working, keep their config shape, and keep receiving feature work.
- Not a v6 schema reopening. Multi-loader support is additive v7 work under invariant 15 (forward-only migration).
- Not a change to the preset card's axis set. See §5.
- Not MTPLX. MTPLX is named throughout as the generalization test for each seam, but its adapter is separate work.
- Not a shared "MLX config" supertype. The three loaders' launch surfaces are structurally different (§3); a unifying supertype would be a lie in the schema.

---

## 2. The structural finding that shapes everything

**oMLX is not launched per-model from the command line. Rapid-MLX is.**

The oMLX `serve` subparser (`omlx/cli.py`, lines 1044–1216) exposes ~24 arguments, and none of them name a model or its decode settings:

```
--model-dir --host --port --log-level --sse-keepalive-mode
--max-audio-upload-size --max-concurrent-requests --embedding-batch-size
--memory-guard --memory-guard-gb --paged-ssd-cache-dir --paged-ssd-cache-max-size
--hot-cache-max-size --hot-cache-write-through --no-cache --initial-cache-blocks
--mcp-config --hf-endpoint --hf-cache --ms-endpoint
--http-proxy --https-proxy --no-proxy --ca-bundle --base-path --api-key
```

Per-model behavior — including `mtp_enabled`, `mtp_num_draft_tokens`, and the whole `qwen35_ane_prefill_*` family — lives in a `ModelSettings` dataclass persisted to `model_settings.json` under the server's base path (`omlx/model_settings.py:434`), and is mutated at runtime through the admin API:

```
PUT  /api/models/{model_id}/settings
GET  /api/global-settings
POST /api/global-settings
```

Models are loaded and unloaded against a *running* server:

```
POST /v1/models/{model_id}/load
POST /v1/models/{model_id}/unload
```

This inverts our launch model. Today a preset resolves to one process with one model and one flag vector. An oMLX preset resolves to a *server scope* plus a *model activation* plus a *settings document*. Section 4 defines how we absorb that without special-casing it through the codebase.

### 2.1 Consequences that must be designed for, not discovered later

| Consequence | Where it is addressed |
| --- | --- |
| Settings are written to a file the server also owns; concurrent editors can clobber | §4.3 settings ownership |
| A "launch" may attach to an already-running server rather than spawn one | §4.2 launch modes |
| Capability evidence cannot be an executable SHA + `--help` | §6 |
| Unload/reload is a first-class lifecycle step, not a process kill | §4.4 |
| `--api-key` is a server-scope secret shared by every model on it | §8 |

---

## 3. Loader comparison — the axes the abstraction must tolerate

| Axis | llama.cpp | Rapid-MLX | oMLX | MTPLX (projected) |
| --- | --- | --- | --- | --- |
| Launch unit | process per model | process per model | server + model activation | process per model |
| Config carrier | CLI flags | CLI flags + `RapidMlxConfig` | `model_settings.json` + admin API | CLI flags |
| Spec-decode / MTP acceptance | yes, polled (`speculative_acceptance_rate`) | greedy argmax | **exact (Leviathan & Chen)** | exact (`sampling.py:244`) |
| MTP depth control | n/a | fixed | `mtp_num_draft_tokens`, adaptive 1..max | per-round block size |
| ANE prefill | no | no | **yes, opt-in, Qwen3.5/3.6/3.8 only** | no |
| Artifact form | GGUF | HF repo / alias / MLX dir | MLX dir under `--model-dir` | MLX dir |
| Capability evidence | binary SHA + `--help` | binary SHA + `--help` | package version + probe endpoint | package version |

The table is the abstraction's requirements document. Any seam that cannot express every row is the wrong seam.

---

## 4. The integration seam

### 4.1 Enum extension

`src/inference/mod.rs`:

```rust
pub enum InferenceBackend {
    #[default]
    LlamaCpp,
    RapidMlx,
    Omlx,
}
```

`src/inference/backend.rs`:

```rust
pub enum BackendAdapter {
    LlamaCpp(Arc<LlamaCppAdapter>),
    RapidMlx(Arc<RapidMlxAdapter>),
    Omlx(Arc<OmlxAdapter>),
}
```

`OmlxAdapter` implements the same six methods `RapidMlxAdapter` exposes today (`src/inference/rapid_mlx/mod.rs:823–999`):

```rust
pub async fn validate(&self) -> Result<()>;
pub async fn build_launch(&self) -> Result<SupervisedLaunch>;
pub async fn await_ready(&self, port: u16, deadline: Instant) -> Result<()>;
pub async fn poll_metrics(&self, base: &str, port: u16, session_id: &str)
    -> Result<InferenceMetricsSnapshot>;
pub async fn cancel_request(&self, port: u16, request_id: &str) -> Result<()>;
pub fn capabilities(&self) -> &CapabilitySet;
```

The `poll_metrics` contract note carries over verbatim: `base` is the full resolved endpoint and must never be assumed to be localhost, because Attach sessions may point at a remote host. For oMLX this is not hypothetical — attach-to-running-server is a normal mode (§4.2).

`RecommendationArtifactKind` gains `OmlxModelDirectory`. The existing `Unknown` variant absorbs anything a future loader introduces without a schema break.

**Rejected alternative:** a `trait BackendAdapter` with `dyn` dispatch. The exhaustive enum is what makes the compiler enumerate every call site when a fourth loader lands — that is the property most worth keeping when the whole point is N loaders.

### 4.2 Launch modes

**Spawn is the primary and expected mode.** Attach exists because the user polls a remote llama-server on a Windows box; it is not the oMLX design center. Where the two modes conflict, spawn wins and attach degrades.

`build_launch()` returns one of three shapes, decided by resolver policy, not by the UI:

1. **Spawn-and-activate (primary)** — spawn an oMLX server for the preset's `--model-dir`, wait for `/health`, `POST /v1/models/{id}/load`. We own the process, its stdout, and its lifecycle.
2. **Attach-and-activate** — a compatible server is already serving that model dir; skip the spawn, activate the model on it.
3. **Attach-only** — the target model is already loaded and its effective settings already match the preset's resolved settings; bind the session and nothing more.

Because spawn is primary, **capturing the child's stdout/stderr is part of the launch contract, not an optional debug affordance.** The supervisor attaches a line-oriented reader to the process at spawn time; §9.5 depends on it for telemetry that no endpoint exposes. Attach modes have no log stream and therefore a strictly smaller metric set; the affected cards are omitted from the dashboard in those sessions rather than rendered empty (§9.6).

Mode selection is observable and recorded in the launch receipt. A user must be able to see *why* their launch did not spawn a process.

Readiness for all three is `/health` (`omlx/server.py:2472`) followed by `/v1/models/status` (line 2990) showing the target model loaded. `/api/status` (line 2522) is the memory/telemetry surface for `poll_metrics`.

### 4.3 Settings ownership

`model_settings.json` is a document the oMLX server owns and its own admin UI edits. We become a second writer. The API makes this tractable:

**`PUT /api/models/{model_id}/settings` is a genuine partial update.** The handler (`omlx/admin/routes.py:2218`) reads Pydantic's `model_fields_set` to distinguish *"sent as null"* (clear to default) from *"not sent"* (don't touch). We send only the keys a preset resolves; every other key is untouched by construction, not by our care in reconstructing a document.

**The response returns `requires_reload`** (line 2974), derived from `EnginePool._engine_runtime_signature()` (`omlx/engine_pool.py:588`). oMLX itself decides whether a settings change demands a model reload. We act on that flag and never maintain our own list of load-bearing keys — such a list would rot on every oMLX release and fail silently when it did.

Rules:

1. **Always the API, never the file — with one exception.** Direct file writes are permitted only in the pre-spawn path, the single moment no server owns the document. Everywhere else, `PUT`.
2. **Send only preset-resolved keys.** Never a full settings object, even when convenient.
3. **Verify the echo.** Compare the response's settings against what we sent. A key that did not take the value we sent is a hard launch failure naming the key, not a warning.
4. **Reload is server-directed.** If `requires_reload` is true, unload and reload the model before binding the session. If false, do not.
5. **Detect external edits, don't merge them.** Capture the effective settings read at resolve time; if a key we intend to write has changed between then and the write, abort with a conflict error naming the key. Silently overwriting a value the user set in oMLX's own admin UI is a correctness bug.

This section was drafted as the integration's highest-risk area. With partial-update and `requires_reload` confirmed, it is not — the log parser (§9.5) is.

### 4.4 Lifecycle

Supervisor semantics differ by mode. In spawn mode we own the process and stop it. In attach modes we own *nothing* — session teardown must not stop a server we did not start, and must not unload a model another session may be using. Unload is only ever issued for a model this session activated in spawn mode, and only when no other tracked session references it.

---

## 5. Preset surface — what changes and what explicitly does not

### 5.1 The card does not change

Architecture §2.1 fixes the card's axes: artifact/quantization, context size, backend-approved main K/V policy, explicit batch/ubatch pair, MoE CPU expert placement, and estimate/fit/measured-memory evidence. §2.2 places speculative decoding and MTP options in the Wizard/Editor, not on the card. **Neither is amended by this work.**

An earlier draft of this analysis argued multi-loader MTP forces MTP onto the card. That was wrong. MTP mode is preset *structure*, which the Editor owns. What the card surfaces is the *eligibility consequence* of that structure, through machinery §6 already defines: an option whose backing capability is unavailable renders **disabled with a backend-provided reason** (invariant 17), exactly as mixed K/V does today.

### 5.2 Config sidecar

`ModelPreset` gains a third backend-owned block beside `rapid_mlx`:

```rust
pub omlx: Option<OmlxConfig>,
```

`OmlxConfig` carries the server scope (`model_dir`, `host`, `port`, `base_path`, `api_key` with the same `#[serde(default, skip_serializing)]` treatment as `RapidMlxConfig::api_key`), the model identity, the MTP block (`mtp_enabled`, `mtp_num_draft_tokens`), and the ANE block (§7). It is `Option` and absent for every non-oMLX preset, so no existing preset's serialized form changes.

**Known scaling concern, deliberately accepted for now:** one `Option<TConfig>` sidecar per loader means MTPLX adds a fourth. At four backends this is still legible; at six it is not. The execution doc should revisit whether the sidecars collapse into a tagged `backend_config` union at the point MTPLX lands — but *not* before, and never in a way that rewrites stored v6/v7 presets. Forward-only, per invariant 15.

### 5.3 Bundles need no schema change

`PresetBundleSpec` (`src/presets/bundle.rs:564–591`) already represents an MLX bundle today: every axis validation is guarded by `!self.<axis>.is_empty()` (lines 732, 742, 763), so an MLX bundle with empty `kv_policy_options` and empty `cpu_moe_options` is legal. The `#[serde(flatten)] extensions: BTreeMap<String, serde_json::Value>` at line 589 is the forward-compatibility escape hatch for loader-specific axes we have not yet designed.

**This is why no pivot is needed on the in-flight branch.** The bundle work as specified is already multi-loader-shaped.

### 5.4 Cross-loader portability

The user's stated requirement: turning on ANE in an oMLX preset and then running that preset against another loader makes the ANE portion a no-op, not an error. Invariant 8 (existing invalid presets are preserved and blocked, never silently changed) and §10's "capability degradation is never permission to delete it" already say this. The addition here is only that the *reason string* must name the loader, so the disabled control reads as "ANE prefill requires the oMLX runtime" rather than a bare "unavailable".

---

### 5.5 Speculative decoding is two tiers, not one vocabulary

Both existing backends already model speculation, and oMLX slots into the established pattern rather than introducing one.

- **Rapid-MLX** — `RapidMlxSpeculativeConfig` (`src/inference/rapid_mlx/mod.rs:68`): a `method` enum (`RapidMlxSpeculativeMethod`, today only `Mtp`), an optional drafter `model: Option<String>`, `num_speculative_tokens`, `disable_auto_k`. Method-plus-optional-drafter is already the shape.
- **llama.cpp** — the same two dimensions expressed as flat fields on `ModelPreset`: `draft_model`, `spec_type`, `ngram_spec`, and ~20 `spec_draft_*` knobs, per the legacy flat-field convention.

`OmlxSpeculativeConfig` mirrors `RapidMlxSpeculativeConfig` directly. Its method enum is larger:

```rust
pub enum OmlxSpeculativeMethod {
    NativeMtp,   // mtp_enabled + mtp_num_draft_tokens; exact acceptance
    VlmMtp,      // vlm_mtp_* ; external drafter, mlx-vlm path
    DFlash,      // block-diffusion; own nested config, see below
    Unknown(String),
}
```

DFlash carries its own substructure the way the ANE family does: `dflash_draft_model`, a draft-quantization group (`weight_bits` 2/4/8, `activation_bits` 16/32, `group_size` 32/64/128), `dflash_block_size`, `dflash_draft_sink_size`, and `dflash_verify_mode` of `"dflash" | "adaptive" | "ddtree" | "off"`. All oMLX speculative methods are mutually exclusive with one another; the config enum enforces that structurally.

**The shared vocabulary is derived, not stored.** Only two properties need to be common across loaders, because only two are reasoned about by the resolver, the card, and the dashboard:

```rust
pub enum SpeculativeAcceptance { Greedy, Exact, Unknown(String) }
pub enum SpeculativeDrafter    { Native, External, Unknown(String) }
```

`acceptance` matters because it is the property that changes output text at temperature > 0. `drafter` matters because an external drafter requires a second model path in the preset and a native one does not. Both are **computed from** whichever method the preset selects — they are not a third place to store the choice. Adding a method to any loader therefore never touches the shared vocabulary.

This supersedes an earlier proposal of a single `Greedy | Exact | ExactExternalDrafter` enum, which conflated the two axes and would not have survived contact with MTPLX's `vlm_mtp` variants.

**SpecPrefill is not in this group.** Despite the name, `specprefill_*` is attention-based *sparse prefill* — it drops tokens by keep-rate (`specprefill_keep_pct`, default 0.2, above `specprefill_threshold`, default 8192 tokens). It is a prefill optimization on the same axis as ANE (§7), not a speculative decoding method, and it is modeled there.

---

## 6. Capability evidence for a Python-package loader

Architecture §10 defines capability evidence as `snapshot_for_binary(path, qualifications)` keyed on a canonical path plus the executable's full SHA-256, with `help_hash` as stored probe evidence. **oMLX has no such executable.** It is a Python package in an environment; its `--help` is generated by argparse and does not enumerate `model_settings.json` keys — which is where every capability we care about actually lives.

The second evidence path:

- **Identity** = interpreter path + resolved package version + package distribution hash, in place of the executable SHA.
- **Probe** = a live query against a running server (`/v1/models/status`, `/api/global-settings`) plus, where no server is available, a static read of the installed `ModelSettings` field set. Both are recorded; the snapshot says which produced it.
- **Qualification** = `BuildQualificationProvider` continues to cover what a probe cannot prove, and this is where the M5/NAX facts land (§7).
- **Classification** = the existing `ProfileClassification` (`Verified`, `Provisional`, `Legacy`, `Incompatible`) is unchanged. A probe-derived snapshot is `Provisional` until a fixture pins it.

`CapabilitySet` (`src/inference/capabilities.rs`) needs no new boolean for MTP — `mtp: bool` exists. What it needs is qualification detail beneath it, because "has MTP" is now three different things across three loaders (greedy / exact / exact-with-external-drafter). The typed-enum-with-`Unknown(String)` pattern §10 already mandates is the right carrier.

`recommend_backend()` (`src/inference/backend.rs`) currently returns a single `BackendRecommendation` with a `state` of `ready` / `platform_unavailable` / `runtime_required` / `manual_selection`. With three loaders eligible for the same MLX artifact, a single recommendation discards the information the user needs. It becomes a **ranked, annotated set** — every eligible backend listed with its own state and reason, ordered by policy — with the existing single-recommendation shape retained as the head of that list for callers that only want one. Invariant 12 holds: the ranking is backend policy, never a duplicated client formula.

---

## 7. ANE prefill

### 7.1 Surface

The `qwen35_ane_prefill_*` family in `ModelSettings` is large — enable, `sequence_length` (default 2048), `tail_padding_min_tokens`, `fraction` (default 0.53), `fused_down`, `max_layers`, `dual_ane`, `gdn`, `gdn_fraction`, `gdn_max_layers`, and a `cpu_*` sub-family for gate/up, down, GDN fractions, thread count, and shared-resource dispatch.

**We do not expose all of it.** The preset carries an opt-in enable plus a small named-profile selection; the long tail stays at oMLX defaults unless a measured result justifies promoting a knob. Architecture §2.3's "blindly copying personal flags" exclusion applies directly.

### 7.2 Hard constraints that must reach the user as reasons, not silent failures

- **Model-family gated.** Dense Qwen3.5/3.6/3.8 MLPs only. Unsupported layers, token counts, dtypes, and all decode/verify calls fall through unchanged.
- **Fixed-shape compilation.** The private ANE runtime accepts only fixed shapes, so a prefill backend is bound to one loaded model *and one sequence length*. Changing context on the card can invalidate an ANE configuration — that interaction must be a resolver rule, surfaced as a disabled option with a reason, not a runtime surprise.
- **Not bit-exact.** INT8 requantization means ANE-accelerated prefill does not reproduce GPU-path logits exactly. This must be stated wherever the option is presented.
- **Resource ceilings.** `_ANE_RESIDENT_PROGRAM_LIMIT = 120` and `_ANE_BANK_RETRY_MAX_BYTES = 1 << 30` are real limits with real failure modes.
- **`dual_ane` is single-die-hostile.** Pinning a bank per physical ANE assumes two.

### 7.3 NAX / M5

`omlx/custom_kernels/qwen35_prefill/fast.py` gates NAX kernels through `is_nax_available()` (line 892), which consults `_stock_mlx_has_nax()` (line 855). Its docstring records the trap: wheels built for macOS < 26.2 carry `MLX_METAL_NO_NAX` and stay on classic kernels **even on M5 hardware**. So "M5 Max" is not sufficient evidence that NAX is active; the MLX wheel's build is. This belongs in `BuildQualificationProvider`, and the user-facing reason must distinguish "your hardware can't" from "your MLX wheel wasn't built for it" — they have different fixes.

### 7.4 SpecPrefill sits here, not with speculative decoding

`specprefill_*` is attention-based sparse prefill — it drops tokens by keep-rate (`specprefill_keep_pct`, default 0.2) above `specprefill_threshold` (default 8192 tokens), for MoE models, and is marked experimental upstream. It is a prefill optimization on the same axis as ANE, not a speculative decoding method (§5.5). It is out of scope for the first integration and is recorded here so it is not mistaken for a speculative path later.

### 7.5 Benchmarking gate

Any performance claim for ANE prefill must come from a **paired A/B inside a single process**. Separate benchmark invocations on this hardware drift ~20%, which is larger than most of the effects being measured. An execution-doc gate that accepts two separate runs as evidence is an invalid gate.

---

## 8. Security

Carried forward from the architecture doc's final checklist, with oMLX-specific additions:

- `OmlxConfig::api_key` is `#[serde(default, skip_serializing)]` — accepted on launch input only, never serialized into presets, sessions, receipts, or diagnostics. Same treatment as `RapidMlxConfig::api_key`.
- **New:** the oMLX `--api-key` is **server-scoped**, not model-scoped. Every model on that server shares it. Attaching to a running server means presenting a credential whose blast radius is the whole server; this must be explicit in the attach path, and an attach must never silently reuse a key stored for a different server scope.
- **New:** `trust_remote_code` in `ModelSettings` defaults to `False` and gates execution of custom Python (`modeling_*.py`, `tokenization_*.py`) from a model repository. We never set it to `True` implicitly, never as part of an intent helper, and never as a fallback when a load fails. Enabling it is an explicit, separately-confirmed user action.
- `--model-dir`, `--hf-cache`, `--paged-ssd-cache-dir`, and `--ca-bundle` are all path inputs: canonicalized and constrained to allowed roots.
- No secret in selection fingerprints or receipts; api-token on resolve and selection APIs; db-admin-token retained on spawn; safe JSON limits and timeouts on every oMLX HTTP call; no unbounded arrays from `/v1/models`; auth routing tests cover every new endpoint.
- oMLX server responses are untrusted input. Model lists, settings documents, and status payloads are parsed defensively and never rendered via innerHTML.

---

## 9. Metrics and dashboard wiring

oMLX's telemetry is real but shallower than llama.cpp's, and its best signal — MTP acceptance — is **not on any HTTP surface today**. That asymmetry drives the whole dashboard story.

### 9.1 What `poll_metrics` reads

Two endpoints, both already identified:

- `GET /api/status` (`omlx/server.py:2522`) — `verify_api_key`-gated, explicitly documented as the "lightweight status endpoint for external tool polling (statuslines, scripts)." This is our polling target. It wraps `get_server_metrics().get_snapshot()` and adds `native_kernel_status()` plus formatted memory.
- `GET /admin/api/stats?model=&scope=session|alltime` (`omlx/admin/routes.py:5231`) — `require_admin`-gated, richer, and the source for per-model and all-time breakdowns.

The scope parameter matters: `session` counters reset with the server, `alltime` counters are persisted and periodically flushed (`_maybe_save_alltime()`). Our dashboard must not mix them in one series.

### 9.2 The counter set

`OMLXServerMetrics.record_request_complete()` (`omlx/server_metrics.py:159`) records exactly six values per request, tracked three times over — session totals, all-time totals, and per-model:

```
prompt_tokens  completion_tokens  cached_tokens
prefill_duration  generation_duration  requests
```

`get_snapshot()` (line 255) returns them plus derived rates:

```
total_prompt_tokens      total_completion_tokens   total_cached_tokens
total_requests           total_tokens_served       total_prefill_duration
total_generation_duration                          uptime_seconds
avg_prefill_tps          avg_generation_tps        cache_efficiency
admission_paused         hard_limit
per_model{<id>: {prompt_tokens, completion_tokens, cached_tokens,
                 requests, prefill_duration, generation_duration}}
session{...}  alltime{...}
```

**These are cumulative averages, not instantaneous rates.** `avg_prefill_tps` is total prefill tokens over total prefill seconds since the counter epoch. A dashboard that plots it directly draws a line that flattens over time and never shows a regression. Every rate chart must be computed by us as a **delta between consecutive polls** — `Δtokens / Δduration` — with the server's `avg_*` fields used only for the lifetime summary tile.

### 9.3 Per-response fields

Streaming responses carry `prompt_tps` and `generation_tps` on the output object (`omlx/server.py:1710–1716`), plus a diffusion-only family (`diffusion_canvas_tps`, `diffusion_work_tps`, `diffusion_denoising_steps`) that we ignore for LLM presets. These are the honest per-request PP/TG numbers and are what a "last request" card should show — they are not averaged over the server's lifetime.

`usage.cached_tokens` on the OpenAI-shaped response gives per-request prefix-cache hits.

### 9.4 What `/api/status` adds

- **Native kernel status** (`native_kernel_status()`) — this is where NAX/M5 activation becomes observable at runtime, and it closes the §7.3 gap: the dashboard can show "NAX kernels active" versus "classic kernels" as fact rather than inference from hardware.
- **Memory** — model memory used/max, plus a `memory_pressure` block (`enabled`, `current_bytes`, `soft_bytes`, `hard_bytes`, `pressure_level` of `ok`/…) driven by the `--memory-guard` / `--memory-guard-gb` flags.
- **Concurrency** — `total_active_requests`, `total_waiting_requests`, and per-model active/waiting counts, plus prefill progress from `get_prefill_tracker()`.
- **`runtime_cache`** — paged-SSD and hot-cache observability corresponding to the `--paged-ssd-cache-*` / `--hot-cache-*` flags.
- **`admission_paused` / `hard_limit`** — backpressure state, which is a status badge, not a chart.

`GET /admin/api/activity` is a lighter poll returning only `active_models` — the right target for a high-frequency live indicator when the full stats payload is too heavy.

### 9.5 MTP acceptance — log-derived

The metric that most justifies choosing oMLX is exposed **only on the log stream**. `mtp_stats`, `accept_len`, and `acceptance_rate` appear **zero** times in `server.py`, `admin/routes.py`, and `server_metrics.py` — confirmed by count. Since spawn is the primary mode (§4.2) and we own the child's stdout there, we parse it.

**Native MTP** — `_log_mtp_stats()` in `omlx/patches/mlx_lm_mtp/batch_generator.py:2819` emits one `logger.info` line per finished sequence. The format is documented in its own docstring, which is the strongest available signal that it is intended to be read:

```
MTP[<uid>] finish=<reason> tokens=<N> cycles=<C> tok/cycle=<T> accept=<A>/<D> (<rate>%)
  depth[d1=<a>/<d>,d2=<a>/<d>,...] d0=<zero_cycles>
  emits[init=<i>,draft=<d>,bonus=<b>,verify=<v>]
  timing[backbone=<X>ms mtp=<Y>ms sample=<S>ms cache=<C>ms]
```

Backed by the `_MtpStats` dataclass (line 658): `cycles`, `accepts`, `init_emits`, `draft_emits`, `bonus_emits`, `verify_emits`, per-depth `depth_drafted[]` / `depth_accepted[]`, `zero_cycles` (cycles where the adaptive depth controller parked at 0, i.e. no speculation), and four component timings — `backbone_ms`, `mtp_head_ms`, `sample_ms`, `cache_ops_ms`.

This is richer than an acceptance rate. It supports:

- **Acceptance rate** — `accepts / total_drafted`, per request and rolled up.
- **Per-depth acceptance curve** — `depth_accepted[i] / depth_drafted[i]`. This is the chart that tells you what `mtp_num_draft_tokens` should actually be: the depth where acceptance collapses is the depth worth capping at.
- **Adaptive-depth behavior** — `zero_cycles` against `cycles` shows how often the controller gave up on speculation entirely.
- **Tokens per cycle** — the honest speedup proxy, more meaningful than raw acceptance.
- **Where the time went** — the four timings exist specifically, per the source comment, to diagnose "accept rate healthy but wall-clock throughput isn't." That is a real and common MTP failure mode and it deserves its own stacked chart.

**VLM MTP** — a separate line from `_log_vlm_mtp_stats()` (`omlx/scheduler.py:8816`), computing `acceptance_rate = total_accepted / (rounds * max_per_round)` and `avg_tokens_per_round`. Different shape, different parser, same destination cards.

**Parsing contract.** The parser is a named, versioned, independently tested component with its own fixture corpus of captured log lines — not a regex inlined in the supervisor. It must: attribute each line to a session via `uid`; tolerate unknown trailing fields without dropping the line; drop a line it cannot parse and increment a visible `unparsed_log_lines` counter rather than failing the poll or guessing; and never block the supervisor's read loop. If the counter climbs, the dashboard says the parser is behind the oMLX version — silent staleness is worse than a visible gap.

**Still upstream a metrics surface.** Log parsing is the right call for shipping, and it is also a standing tax: every oMLX release can move the format, and attach mode gets nothing. Adding `mtp_stats` to `get_snapshot()` and exposing it on `/api/status` remains a small, self-contained, broadly useful PR — much easier to land than the Rapid-MLX non-greedy work. Do both: parse now, upstream in parallel, and prefer the endpoint over the parser once it exists.

### 9.6 Cross-loader dashboard contract

`InferenceMetricsSnapshot` is shared across adapters, so the dashboard must degrade per field, not per backend. Concretely:

| Card / chart | llama.cpp | Rapid-MLX | oMLX |
| --- | --- | --- | --- |
| PP tok/s (delta-derived) | yes | yes | yes (§9.2 caveat) |
| TG tok/s (delta-derived) | yes | yes | yes |
| Last-request PP/TG | yes | yes | yes (§9.3) |
| Prefix-cache hit rate | yes | yes | yes (`cache_efficiency`, `cached_tokens`) |
| Active / queued requests | yes | yes | yes |
| Memory + pressure | yes | partial | yes (incl. soft/hard guard) |
| Kernel path (NAX/classic) | n/a | no | yes (§9.4) |
| Spec-decode acceptance rate | yes, polled | no | yes, log-derived, spawn only (§9.5) |
| Per-depth accept curve | no | no | yes, log-derived, spawn only |
| Draft/verify component timings | no | no | yes, log-derived, spawn only |
| Per-model breakdown | n/a | no | yes (`per_model`) |
| All-time vs session scope | no | no | yes (`scope=`) |

Two structural consequences.

**Extension shape is already decided.** `InferenceMetricsSnapshot` (`src/inference/metrics.rs`) is `Option`-per-field throughout and already carries `speculative_acceptance_rate: Option<f64>`, which llama.cpp populates today — oMLX simply becomes a second producer of an existing field rather than introducing a parallel one. Genuinely oMLX-shaped data (per-depth accept arrays, the four component timings, `zero_cycles`) goes in `backend_details: Option<serde_json::Value>`, whose existing contract is that the **card registry** maps it — never raw JSON to the view. Kernel path and per-model breakdown follow the same route. No new top-level fields, so no other adapter is touched.

**Missing means absent, not zero — and absent means hidden.** The snapshot dictionary already implements this; `dictionary_marks_missing_values_unavailable_without_zero_filling` is an existing test. The policy this document adds is what the *view* does with an unavailable metric, and it splits by cause:

- **Capability the backend does not have** — the card does not render. Rapid-MLX has no acceptance telemetry because it has no exact-acceptance path; a greyed tile would imply the number exists somewhere.
- **Capability present but unobservable in this session** — the card does not render either. Specifically: an oMLX server we attached to rather than spawned yields no log stream and therefore no MTP telemetry, and that card is **omitted**, not disabled. A permanently-empty tile on a rarely-used path is clutter, and clutter that looks like a broken chart.
- **Metric temporarily unavailable within a working card** — a failed poll, a not-yet-populated counter — renders as unavailable inside a card that does render. This is the case invariant 34 governs.

The distinction that matters: a card is hidden when the data can never arrive in this session; a value reads "unavailable" when it could arrive on the next poll.

### 9.7 Security note on the stats endpoints

`GET /admin/api/stats` returns `api_key` in its payload (`omlx/admin/routes.py`). That value must never be stored, logged, echoed into a receipt, or rendered. The adapter's deserializer drops it at the parse boundary rather than after — the field should not survive into any struct we hold.

---

## 10. Invariants added by this document

Numbered continuing from the architecture doc's 1–24, and binding on the execution doc.

25. Adding a backend never changes the serialized form of a preset that does not use it.
26. Loader-specific preset content is preserved verbatim when the preset is opened, edited, or launched under a different loader; unsupported content is inert and reported, never deleted (extends invariant 8).
27. Every "unavailable" state names the loader and the specific missing capability. A bare "unsupported" is a contract violation (extends invariant 17).
28. Settings reach oMLX through the partial-update API, carrying only preset-resolved keys. Direct file writes occur only pre-spawn. Reload is driven by the server's `requires_reload`, never by our own list of load-bearing keys.
29. A session stops only processes it started and unloads only models it activated.
30. Capability evidence for a Python-package loader records which probe path produced it and is `Provisional` until a fixture pins it.
31. ANE performance claims require a paired in-process A/B. Separate invocations are not evidence.
32. `trust_remote_code` is never enabled implicitly, as a fallback, or by an intent helper.
33. Rate charts are derived from deltas between polls. A server-reported lifetime average is never plotted as a time series.
34. A metric that is not measured is absent, never zero-filled. A card whose data can never arrive in this session is omitted; a value that could arrive on a later poll renders as unavailable inside a card that still shows.
35. Session-scope and all-time-scope counters are never mixed within one series.
36. The `api_key` returned by oMLX's stats payload is dropped at the parse boundary and never enters a struct, log, or receipt.
37. Log-derived metrics are produced by a versioned parser with a fixture corpus. An unparsed line increments a visible counter; it is never guessed at or silently dropped.
38. Log parsing never blocks the supervisor's read loop, and a parser failure never fails a launch or a metrics poll.
39. Where an endpoint and the log stream both carry a metric, the endpoint wins.

---

## 11. Decisions

Resolved on 2026-09-02. Each is binding on the execution doc unless new evidence overturns it.

1. **Config sidecars, revisited at MTPLX.** Keep `omlx: Option<OmlxConfig>` beside `rapid_mlx: Option<RapidMlxConfig>`. Three `Option` fields stay legible and no existing preset's serialized form changes. A tagged `backend_config` union is the better long-run shape but requires migrating stored presets to solve a problem that does not yet exist; reassess when MTPLX makes it four. Forward-only whenever it happens (invariant 15), so delay costs little.

2. **Server-scope identity is `(canonical model_dir, host, port, base_path)`.** All four, canonicalized. `base_path` is in the key because it locates `model_settings.json` — two servers on one model dir with different base paths have genuinely different settings, and treating them as one would apply a preset's settings to the wrong document. The same tuple gates credential reuse (§8): same scope, same stored API key; anything else, no reuse.

3. **Two ANE profiles, no raw knobs initially.**
   - **Balanced** — oMLX's defaults verbatim (`fraction=0.53`, `gdn=true`, `gdn_fraction=0.50`, `max_layers=64`, `gdn_max_layers=48`, `cpu_enabled=false`), except `dual_ane`, which is resolved from detected hardware rather than inherited at its upstream default of `true`. That default is single-die-hostile and must not be inherited blind.
   - **ANE-heavy** — raised `fraction` and `gdn_fraction`, `cpu_enabled=true` at its default `cpu_fraction=0.135`.

   Two, not three, because there is no measured data yet and a third would be invented rather than derived. Individual knobs are promoted to the preset only when a paired in-process A/B (invariant 31) shows one matters. `sequence_length` is excluded from profiles entirely — it is a hard fixed-shape binding that interacts with the card's context axis, so it is resolved, not chosen.

4. **Speculative decoding is two tiers.** Per-loader typed method enums for structure; a derived `acceptance` / `drafter` pair for cross-loader gating. Specified in §5.5.

5. **Recommendation ranking: capability match, then user pin, then stable order.** A loader that cannot satisfy something the preset explicitly enables sorts below one that can; among equals the user's pinned default wins; ties break on a stable order. The pin is essential — Rapid-MLX is retained deliberately, and a ranking that quietly promotes oMLX on paper score would fight the user's intent. Ranking annotates and orders; it never changes a stored preset's backend.

6. **Fixture: a hashed `ModelSettings` field-set snapshot per oMLX version.** Generated by reflecting over the dataclass at probe time — field names and types, not values — and stored as the evidence `help_hash` provides for llama.cpp. Reproducible, cheap, and it fails in the right direction: oMLX adds or renames a settings key, the snapshot stops matching, and the profile drops to `Provisional` rather than silently writing a key that no longer exists. Paired with a captured-log-line corpus for the §9.5 parser, versioned the same way.

7. **MTP telemetry: parse now, upstream later, treat the PR as optional.** The `mtp_stats` endpoint PR does not sit on this work's critical path — an unmerged upstream PR has cost this project time once already. Build the parser, ship it, open the PR when convenient. If it lands, invariant 39 retires the parser with no plan change. If it never lands, nothing is lost.

8. **`InferenceMetricsSnapshot` extension shape.** Reuse the existing `speculative_acceptance_rate: Option<f64>`, which llama.cpp already populates; put oMLX-shaped detail in `backend_details` behind the card registry. No new top-level fields, no other adapter touched.

## 12. Sequencing

This work does not start until `feature/preset-bundle-launch-options` merges. It depends on the v6 bundle schema being stable and on Phase 10b's qualification gates existing, because the execution doc reuses them.

The user's earlier question — MTPLX first, or oMLX first — resolves to **oMLX first**, on capability grounds rather than repository-health grounds. oMLX has both differentiators (exact MTP, ANE); MTPLX has one (exact MTP) and no ANE at all. The "oMLX is buggier" hypothesis does not survive normalization: MTPLX receives roughly 3× more issue reports per star. oMLX's real weakness is triage capacity — a 42% close rate against MTPLX's 89% — which predicts "a bug I file may sit" rather than "I will hit more bugs." That is a maintenance-risk fact worth recording, not a reason to sequence differently.

---

## 13. Research gaps to close before the execution doc

Each of these is a known unknown. None blocks this architecture, all block confident task decomposition. Listed with what specifically must be answered.

1. **DFlash acceptance semantics.** `dflash_verify_mode` offers `"dflash" | "adaptive" | "ddtree" | "off"` and `omlx/patches/dflash_laguna.py:316` tracks `acceptance_len`, but whether any mode preserves the target distribution the way the native MTP path provably does has not been read closely. Since exactness is the reason for choosing oMLX, DFlash carries `acceptance: Unknown(String)` until this is verified in source. **Answer needed:** which verify modes, if any, classify as `Exact`.

2. **oMLX log format stability across versions.** §9.5's parser depends on `_log_mtp_stats()`'s format. Its docstring documents the format, which suggests intent, but no upstream stability guarantee has been confirmed. **Answer needed:** diff the format across the last several oMLX releases to establish whether it churns, which sets how defensive the parser must be and how often the fixture corpus needs regenerating.

3. **`model_settings.json` concurrent-write behavior under the API.** §4.3 assumes `PUT` serializes correctly against the server's own writers. `ModelSettingsManager` holds a `threading.Lock` (`omlx/model_settings.py`), which covers in-process concurrency but says nothing about a second process editing the file. **Answer needed:** confirm the failure mode when an external process writes the file while the server is running, since our pre-spawn exception (§4.3 rule 1) must be provably safe.

4. **Reload cost.** `requires_reload` tells us *whether* a reload is needed, not what it costs. If common preset edits trigger multi-second model reloads, the Editor's UX needs to warn before saving. **Answer needed:** which of our resolved keys land in `_engine_runtime_signature()`, and measured reload wall-clock for a representative model.

5. **ANE eligibility detection.** §7 states the family gate (dense Qwen3.5/3.6/3.8) and the fixed-shape binding, but not how we *predict* eligibility at resolve time rather than discovering fallthrough at runtime. **Answer needed:** what the resolver can read from model config to decide eligibility before launch, and what remains unknowable until load.

6. **NAX detection surface.** §9.4 identifies `native_kernel_status()` as the runtime observability point. **Answer needed:** its exact payload shape, and whether it distinguishes "hardware lacks NAX" from "MLX wheel built with `MLX_METAL_NO_NAX`" — §7.3 requires those to produce different user-facing reasons because they have different fixes.

7. **Metrics endpoint auth split.** `/api/status` is `verify_api_key`-gated; `/admin/api/stats` is `require_admin`-gated. **Answer needed:** whether spawn-mode launches always hold admin credentials, or whether the richer per-model and all-time breakdowns are unavailable in some configurations — which changes §9.6's matrix.

8. **oMLX version floor.** No minimum supported version is established. **Answer needed:** the earliest release carrying exact MTP acceptance, the ANE prefill patch, and the `_log_mtp_stats()` format we parse; that becomes the floor and a Global Constraint in the execution doc.

9. **MTPLX adapter shape.** Named throughout as the generalization test but never verified against. **Answer needed:** a source pass equivalent to the oMLX one, sufficient to confirm the seams here actually accommodate it — ideally before the sidecar-vs-union decision (§11.1) is revisited.

