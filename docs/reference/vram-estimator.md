# VRAM Estimator Reference

The VRAM estimator (`src/llama/vram_estimator/`) is backend-aware. It produces
estimates for both:

- **llama.cpp** (CUDA/ROCm/Metal) — based on GGUF metadata or name heuristics.
- **Rapid-MLX** (Apple Silicon, unified memory only) — based on MLX `config.json` +
  safetensors index or name heuristics.

Both backends share the same `ModelArch` struct, the same KV cache and MoE formulas,
and the same `VramBreakdown` output shape. They differ in:

- Overhead calibration:
  - llama.cpp Metal: calibrated on Apple M5 Max.
  - llama.cpp CUDA/ROCm: calibrated on RTX 5090.
  - Rapid-MLX: formula-based approximation with a 25% safety margin; not yet
    hardware-calibrated (see "Rapid-MLX overhead" section).
- How model metadata is read:
  - llama.cpp: GGUF tensor directory / header introspection (primary) or name heuristic (fallback).
  - Rapid-MLX: MLX `config.json` + `model.safetensors.index.json` (primary) or name heuristic (fallback).
- Evidence:
  - llama.cpp: `Measured` when GGUF-backed; `Degraded` when only name-based.
  - Rapid-MLX: always at best `Approximate` (never `Measured`) until calibrated;
    `Degraded` when config fields are incomplete.

All VRAM bars in the UI (Spawn Wizard, Preset Editor, Setup view, and the Models-modal
HF-browse preview) are driven by the backend `/api/vram-estimate` with the appropriate
`backend` field — there is no client-side VRAM/KV formula. Pre-download, the backend
fetches the model's real metadata (GGUF header or MLX config) from HuggingFace so even
the browse preview uses the model's real architecture.

### Native context ceilings

The estimator response carries `native_context_limit` when metadata provides one.
GGUF uses its authoritative `*.context_length`; MLX uses
`max_position_embeddings` (or `model_max_length` only when the canonical field
is absent). Spawn Wizard and the preset editor use that value to distinguish a
standard context choice from an advanced-only extension. The standard choices
are 32K, 65K, 131K, 160K, 200K, and 262K; a choice above the metadata ceiling
is not a normal-fit recommendation, even if the memory estimate fits.

RoPE/YaRN and related extension controls are deliberately not emitted or
estimated yet. They require model-specific launch mapping, tokenizer/template
headroom, and extended-context calibration before they can become a supported
configuration.

## Memory-pool and active-parameter behavior (llama.cpp GGUF)

GGUF tensor shapes are the source of truth for MoE active parameters:

```text
active = non-routed parameters
       + routed-expert parameters × active_experts / total_experts
```

This is quantization-independent and works for complete local GGUFs and
range-fetched headers. Local validation gives 3.455B active parameters for
Qwen3.6-35B-A3B, 3.875B for Qwen3-Coder-Next, and 3.823B for Gemma 4 26B-A4B.
When tensor shapes are unavailable, the structural DeltaNet fallback only
reduces attention layers if `ssm.inner_size` is also present.

Exact path (primary):

- The GGUF tensor directory provides `tensor_param_count` (total tensor elements)
  and `expert_param_count` (routed-expert tensor elements).
- Active parameters are:

  ```text
  active = (tensor_param_count − expert_param_count)
         + expert_param_count × active_experts / total_experts
  ```

- Non-routed parameters (backbone, embeddings, DeltaNet projections) are always
  active; routed experts are scaled by the expert ratio.
- This path is preferred when both counts are present and reasonable; it matches
  local GGUF validation more closely than the older structural heuristic.

When tensor counts are missing, the estimator falls back to the structural DeltaNet
heuristic or simple MoE ratio.

Unified-memory systems use one Metal-capped memory pool. Discrete systems keep
VRAM and system RAM independent: `--n-cpu-moe` moves routed experts to RAM, and
dense `--gpu-layers` moves transformer-layer weights between the two pools.
Spawn Wizard, Preset Editor, and welcome-screen preset cards use this same
backend split. Card bars use fixed machine VRAM/RAM denominators so model and
context-size differences remain directly comparable.

The VRAM estimator (`src/llama/vram_estimator/`) predicts GPU memory usage for a given model, quantization, context size, and hardware configuration. It powers:

- The auto-size wizard
- The pre-download quant advisor
- The preset-editor VRAM strip
- Launch-card VRAM estimates

All VRAM bars in the UI (Spawn Wizard, Preset Editor, Setup view, and the Models-modal HF-browse preview) are driven by the backend `/api/vram-estimate` — there is no client-side VRAM/KV formula. Pre-download, the backend range-fetches the GGUF header from HuggingFace so even the browse preview uses the model's real architecture.

Estimation is primarily based on GGUF introspection, not name matching. Filename-based heuristics are used only when GGUF metadata is unavailable (e.g., pre-download estimates or incomplete headers).

![VRAM breakdown bar in the Hardware step](../screenshots/spawn-wizard--llamacpp-local--hardware-vram.png)

---

## ModelArch

`ModelArch` is the central struct. Every estimation function takes a `&ModelArch`.

**All fields are concrete integers or floats (0 = unknown).** Upstream `ModelMetadata` uses `Option<T>` fields; `ModelMetadata::to_arch()` (in `src/llama/spawn_wizard.rs`) resolves or falls back for each field, and uses 0 when unknown.

It is populated via `to_arch()` in two primary ways:

- **From a local GGUF file (primary)**:
  `GgufMetadata::to_model_metadata()` reads the file, producing a `ModelMetadata`, which is then converted via `ModelMetadata::to_arch()` into `ModelArch`. Structural fields (layer counts, attention config, hybrid attention interval, MTP depth, etc.) come from the GGUF header and override any name-based guess. This is authoritative for all downloaded models, including finetunes with unusual names.

- **From name + param count (fallback only)**:
  `ModelArch::from_name_and_params(name, param_b)` builds a coarse heuristic when GGUF introspection is not possible (e.g., before download). Even when a GGUF file exists, `to_arch()` always runs `from_name_and_params()` first as an initial scaffold; then overrides fields from GGUF data; and — if the name heuristic produced only weak defaults and a known GGUF architecture is present — may re-run the heuristic using the GGUF-derived family name to get the correct shape. This is intentionally minimal and should not be relied on for correctness when GGUF metadata is available.

```rust
pub struct ModelArch {
    // Standard attention
    pub n_layers: u32,          // Total transformer layers
    pub n_kv_heads: u32,        // KV heads (GQA/MQA)
    pub head_dim: u32,          // Per-head key/value dimension

    // Sliding-window / alternating attention (Gemma 3/4)
    pub n_global_attn_layers: u32,  // Layers that attend full context (0 = all)
    pub local_attn_window: u32,     // Sliding window size in tokens (0 = N/A)
    pub local_kv_heads: u32,        // KV heads for local layers
    pub global_head_dim: u32,       // head_dim override for global layers (Gemma 4 = 512)

    // MoE
    pub n_experts: u32,         // Total experts per layer (0 = dense)
    pub n_experts_used: u32,    // Experts activated per token
    pub expert_fraction: f64,   // Fraction of params in expert FFNs (default 0.65)

    // Hybrid linear attention (Qwen3.5 / Qwen3.6 / DeltaNet)
    pub n_attn_layers: u32,             // Layers with a KV cache (0 = all layers)
    pub linear_attn_state_bytes: u64,   // Fixed recurrent state size (context-independent)

    // MTP
    pub mtp_depth: u32,         // MTP prediction head count (0 = none)

    // Multimodal
    pub mmproj_bytes: u64,      // Vision projector size in bytes (0 = none)

    // Sizing / overhead
    pub param_b: f64,           // Approximate param count in billions
    pub n_embd: u32,            // Hidden/embedding dimension (for CUDA overhead estimate)
                                 // 0 = unknown; set from GGUF embedding_length when available

    // Exact per-layer size (from GGUF tensor directory)
    pub bytes_per_layer: u64,   // Bytes per repeating transformer block (0 = unknown)
}
```

### Helper predicates

| Method | True when |
|--------|-----------|
| `is_moe()` | `n_experts > 1` |
| `is_hybrid_attn()` | `n_attn_layers > 0 && n_attn_layers < n_layers` |
| `has_local_attn()` | `local_attn_window > 0 && n_global_attn_layers < n_layers` |

---

## Architecture Heuristics (Name-Based Fallback)

Used only when GGUF introspection is not available, or as an initial scaffold that is then overridden by GGUF values.

`ModelArch::from_name_and_params(name, param_b)` returns a best-effort arch from the model filename and parameter count. For any GGUF file on disk, the per-family tables below are NOT directly used for VRAM estimation — the GGUF is authoritative. These heuristics shape only pre-download or missing-field estimates.

### Priority order (first match wins)

1. `"exaone-4.5"` / `"exaone4.5"` → `exaone45_heuristic(param_b)` (checked first)
2. `"coder-next"` / `"qwen3-coder-next"` → `qwen3_coder_next_arch()`
3. `"qwen3.6"` / `"qwen3-6"` / `"qwopus3.6"` / `"qwopus3-6"` / `"qwopus36"` →
   - `"35b-a3b"` → `qwen36_35b_a3b_arch()`
   - else → `qwen36_heuristic(param_b)`
   - MTP detection: if filename contains `"mtp"` or `"multi-token"`, set `mtp_depth = 1`
4. `"qwen3.5"` / `"qwen3-5"` → `qwen35_heuristic(param_b)` (plus MTP detection if present)
5. `"gemma-4"` / `"gemma4"` → `gemma4_heuristic(name, param_b)` (plus MTP detection if present)
6. `"gemma-3"` / `"gemma3"` → `gemma3_heuristic(param_b)` (plus MTP detection if present)
7. Fallback → `standard_heuristic(param_b)` with MoE suffix parsing and MTP detection if present

### Per-family heuristics

These are initial defaults for `from_name_and_params()`; GGUF introspection overrides them when present.

#### Dense / Qwen3 standard (`standard_heuristic`)

| param_b | n_layers | n_kv_heads | head_dim |
|---------|----------|------------|----------|
| < 2 | 22 | 4 | 64 |
| 2–5 | 28 | 4 | 128 — Qwen2.5-3B / Phi-3 style |
| 5–10 | 32 | 8 | 128 — Llama-3.1-8B, Mistral-7B, Qwen2.5-7B |
| 10–25 | 40 | 8 | 128 — Llama-2-13B, Qwen2.5-14B, Mistral-22B |
| 25–35 | 48 | 4 | 128 — tuned for Qwen3-30B-A3B |
| 35–75 | 80 | 8 | 128 — Llama-70B, Qwen2.5-72B |
| 75+ | 94 | 4 | 128 — Qwen3-235B |

#### Qwen3.6 (hybrid DeltaNet + dense)

3:1 DeltaNet:Attention ratio — exactly 1/4 of layers are standard softmax attention with a KV cache. The remaining 3/4 use a fixed-size recurrent state regardless of context length.

KV cache only grows for `n_attn_layers` — the DeltaNet layers contribute nothing to context scaling.

| param_b | n_layers | n_attn_layers | n_kv_heads | head_dim | n_embd | DeltaNet state |
|---------|----------|--------------|------------|----------|--------|----------------|
| ≤ 35 (27B) | 64 | 16 | 4 | 256 | 5120 | 48 layers × 48 V-heads × 128² × 2 B ≈ 75 MB |
| > 35 (davidau 40B) | 96 | 24 | 4 | 256 | 5120 | 72 layers × 48 V-heads × 128² × 2 B ≈ 113 MB |

**Calibrated on**: Qwen3.6-27B-NEO-CODE Q4_K_M GGUF (GGUF arch tag: `qwen35`). n_embd=5120 confirmed from `embedding_length` in GGUF metadata.

#### Qwen3.6-35B-A3B (exact — confirmed from model card)

40 total layers: 10 Attention + 30 DeltaNet. "A3B" = 3B **active parameters**, not 3 active experts. Active params are derived from GGUF metadata via `GgufMetadata::active_params_b()`, which:

- Estimates backbone (non-expert) params from real attention head dims, KV heads, and embedding length.
- For hybrid DeltaNet models, uses `n_attn_layers` (from `full_attention_interval`) for the standard-attention backbone, then adds always-active DeltaNet projections sized by `ssm_inner_size`.
- Treats the rest as expert weight; active ≈ backbone + (used / total) × experts.
- Falls back to a simple ratio `total / (1 + N_experts / N_used)` when GGUF fields are missing or sanity-checks fail.

This ensures "A3B"-style labels reflect the model's actual on-the-fly footprint, not a misleading name-based guess.

| Field | Value |
|-------|-------|
| n_layers | 40 |
| n_attn_layers | 10 |
| n_kv_heads | 2 |
| head_dim | 256 |
| n_embd | 4096 (estimated; overridden by GGUF) |
| n_experts | 256 |
| n_experts_used | 9 (8 routed + 1 shared) |
| expert_fraction | 0.85 |
| DeltaNet state | 30 layers × 32 V-heads × 128² × 2 B ≈ 31 MB |

#### Qwen3.5 (hybrid DeltaNet + MoE)

Same 3:1 ratio. Only confirmed for 122B-A10B; heuristics applied to smaller sizes.

| param_b | n_layers | n_attn_layers | n_kv_heads | n_experts | n_experts_used | n_embd |
|---------|----------|--------------|------------|-----------|----------------|--------|
| ≤ 80 | 40 | 10 | 2 | 256 | 9 | 4096 |
| > 80 (122B) | 48 | 12 | 2 | 256 | 9 | 7168 |

DeltaNet V-heads: 64 for 122B (confirmed), 32 assumed for smaller.

#### Qwen3-Coder-Next (exact — confirmed)

48 layers: 12 standard attention + 36 DeltaNet.

| Field | Value |
|-------|-------|
| n_layers | 48 |
| n_attn_layers | 12 |
| n_kv_heads | 2 |
| head_dim | 256 |
| n_embd | 7168 (235B-class architecture) |
| n_experts | 512 |
| n_experts_used | 11 (10 routed + 1 shared) |
| expert_fraction | 0.92 |
| DeltaNet state | 36 layers × 32 V-heads × 128² × 2 B ≈ 38 MB |

#### Gemma 3 (alternating local/global attention)

1-in-6 layers use full global attention; remaining layers use a 512-token sliding window with MQA (`local_kv_heads = 1`).

The `sliding_window_pattern` bool array in the GGUF determines each layer's role: `false` = global (full context), `true` = local (sliding window). `n_global_attn_layers` is the count of global layers.

| param_b | n_layers | global_layers | n_kv_heads (global) | head_dim | local_kv_heads | window |
|---------|----------|---------------|---------------------|----------|----------------|--------|
| < 5 (4B) | 34 | 6 | 4 | 256 | 1 | 512 |
| 5–14 (12B) | 52 | 9 | 8 | 256 | 1 | 512 |
| > 14 (27B) | 62 | 10 | 16 | 256 | 1 | 512 |

`global_layers` is computed as `round(n_layers / 6)`.

#### Gemma 4 (sliding-window alternating attention)

5:1 local:global pattern — every 6th layer attends the full context; the rest use a sliding window.

Global layers use `global_head_dim = 512`, local layers use `head_dim = 256`. Gemma4-26B-A4B has `n_experts_used = 9` (8 routed + 1 always-loaded shared expert).

| Tier | n_layers | global_layers | n_kv_heads (global) | local_kv_heads | n_experts | window | n_embd |
|------|----------|---------------|---------------------|----------------|-----------|--------|--------|
| E2B | 35 | 7 | 1 | 1 | 0 | 512 | 1152 |
| E4B | 42 | 7 | 2 | 2 | 0 | 512 | 2048 |
| 12B dense | 48 | 8 | 1 | 8 | 0 | 1024 | 3072 |
| 26B-A4B MoE | 30 | 5 | 2 | 8 | 128 | 1024 | 2048 |
| 31B dense | 60 | 10 | 4 | 16 | 0 | 1024 | 5120 |

- n_embd values for E2B/E4B/12B/26B-A4B are estimated; GGUF `embedding_length` overrides when present
- Auto-size uses `block_count ≥ 75` → Qwen3.5; `< 75` → Qwen3.6 to disambiguate the shared `"qwen35"` arch tag when the GGUF is present
- Gemma4-31B is identified via `n_layers=60`; `param_b` is overridden to 31B to ensure the correct heuristic tier

#### EXAONE 4.5

All known EXAONE 4.5 sizes unconditionally set `mtp_depth = 1`.

| param_b | n_layers | global_layers | n_kv_heads | head_dim | window | mtp_depth | mmproj |
|---------|----------|---------------|------------|----------|--------|-----------|--------|
| ≤ 15 | 32 | 8 | 8 | 128 | 4096 | 1 | 0 |
| > 20 (33B) | 64 | 16 | 8 | 128 | 4096 | 1 | 2.58 GB |

### Generic MoE suffix parsing

For names not matched by the above, `parse_moe_suffix()` scans for `"NB-AMB"` or `"NB_AMB"` patterns (e.g. `"26B-A4B"`, `"122B-A10B"`). It:

- Uses the last valid pattern in the name (rightmost match) to avoid false positives.
- Enforces `total_b >= 7.0` to reject tokens like `"llama-3-a4b"`.
- Enforces `active_b <= total_b`.

For matched suffixes:

- `n_experts` inferred from sparsity (`active_b / total_b`):
  - < 5% → 512 (extremely sparse, Coder-Next style)
  - total > 100B → 128
  - total > 50B → 64
  - total > 20B → 32
  - else → 8 (Mixtral style)
- `n_experts_used` derived from sparsity:
  - < 5% → 11
  - ≤ 15% → 9
  - else → 8
- `expert_fraction` defaults to 0.65; family-specific constructors override this (0.85 for Qwen3.5/3.6, 0.92 for Coder-Next)

---

## Estimation Functions

### `kv_cache_bytes`

```
kv_cache_bytes(arch, context_size, parallel_slots, ctk, ctv) → u64
```

Standard (dense / hybrid, no local attention):

```
effective_layers = n_attn_layers if n_attn_layers > 0 else n_layers
K = effective_layers × n_kv_heads × head_dim × context × slots × k_bpe
V = effective_layers × n_kv_heads × head_dim × context × slots × v_bpe
total = K + V
```

For DeltaNet hybrid models (Qwen3.5/3.6): `effective_layers = n_attn_layers` (e.g. 16 out of 64 for 27B). KV grows at 1/4 the rate of a standard dense model with the same layer count.

Sliding-window (for any `has_local_attn()` model, e.g. Gemma 3/4, EXAONE 4.5):

```
global_layers = min(n_global_attn_layers, effective_layers)
local_layers  = max(effective_layers - n_global_attn_layers, 0)
g_hd          = global_head_dim if > 0 else head_dim   // Gemma 4 uses 512 for global; others fall back to head_dim
effective_local_ctx = min(context, local_attn_window) × slots

global_K = global_layers × n_kv_heads × g_hd × context × slots × k_bpe
global_V = (same with v_bpe)
local_K  = local_layers  × local_kv_heads × head_dim × effective_local_ctx × k_bpe
local_V  = (same with v_bpe)
total = global_K + global_V + local_K + local_V
```

### `moe_weight_split`

```
moe_weight_split(model_size_bytes, arch, n_cpu_moe) → (vram_bytes, ram_bytes)
```

For `n_cpu_moe > 0` on a MoE model:

```
moe_layers   = max(n_layers, 1)
cpu_layers   = min(n_cpu_moe, moe_layers)
cpu_ratio    = cpu_layers / moe_layers
expert_frac  = expert_fraction.clamp(0.3, 0.99)
cpu_bytes    = model_size_bytes × expert_frac × cpu_ratio
vram_bytes   = model_size_bytes − cpu_bytes
```

`--n-cpu-moe N` treats N as the number of transformer layers whose experts are kept on CPU — not as a count of individual experts.

For dense models or `n_cpu_moe ≤ 0`: all weights in VRAM.

### `mtp_overhead_bytes`

```
mtp_overhead_bytes(model_size_bytes, mtp_depth) → u64
  = model_size_bytes × 0.015 × mtp_depth
```

1.5% of model weights per MTP depth level. Heuristic for DeepSeek-V3 / Qwen3-MTP style heads.

### Metal (unified-memory) overhead

Used on Apple Silicon. Like the discrete model it has a context-independent base plus a
context-scaling part, but it is **much lighter** and calibrated separately against real
measurements. All inputs are GGUF-derived.

- `metal_overhead_base_bytes(arch, ubatch_size)`:
  - Context-independent per-layer graph/context cost + small ubatch scratch.
  - Formula: `per_layer × n_layers + 0.035 × ubatch` MiB, where `per_layer = 4.3` (dense) or
    `8.8` (sliding-window / Gemma — extra per-layer-input embeddings + dual local/global graph).
  - 128 MiB floor; flat 200 MiB if arch unknown.

- `metal_overhead_ctx_bytes(kv_cache_bytes)`:
  - Context-scaling working buffers, measured at a very stable **~6.5% of KV bytes**
    (`METAL_KV_OVERHEAD_FRACTION = 0.065`) across dense/MoE/hybrid/SWA models. Because it's a
    fraction of `kv_cache_bytes`, it automatically tracks hybrid attention, Gemma windows and KV quant.

- `metal_overhead_bytes(arch, ubatch_size, kv_cache_bytes)` = base + ctx.

**Calibrated on Apple M5 Max (llama.cpp b9743, Metal, `--parallel 1 --kv-unified -fa on`,
q8_0 KV)** via process physical-footprint measurements, across the same four models as the
discrete model, at 4k–213k context. Fits within ~40 MiB (worst under-prediction −17 MiB).
Metal's overhead is far smaller than CUDA's and grows ~6× more gently with context. Note: the
prior flat 300 MB estimate under-reserved Gemma-4-31B@213k by ~750 MiB — a real OOM risk this fixes.

### Discrete-GPU overhead (CUDA/ROCm)

The discrete overhead model has three parts. All inputs are GGUF-derived (`embedding_length`, `expert_count`, `block_count`, sliding-window pattern), never from name parsing.

- `discrete_overhead_base_bytes(arch, ubatch_size)`:
  - Context-independent: graph compute scratch (∝ ubatch × model width), MoE expert gather/scatter buffers, and (for Gemma with sliding-window) per-layer-input embedding tables.
  - If `n_embd == 0` or `n_layers == 0` (unknown architecture): flat 256 MB fallback.
  - 200 MiB floor in `base_bytes` (CUDA context minimum).

- `discrete_overhead_ctx_bytes_per_token(arch, ubatch_size)`:
  - Context-dependent: attention mask and per-layer prefill scratch that grow linearly with context, `n_layers`, and per-head dimension.
  - Uses `max(head_dim, global_head_dim)` for Gemma-style models with wider global heads.
  - Formula: `0.46 × n_layers × max(head_dim, global_head_dim) × (0.8 + 0.2 × ubatch/1024)`.

- `discrete_overhead_bytes(arch, ubatch_size, context_size)`:
  - Total discrete overhead = base + (per_token × context_size).

**Calibrated on RTX 5090 32 GB (WDDM), llama.cpp b9728, `--parallel 1 --kv-unified -fa on`, q8_0 KV, full GPU offload**, across:

- Qwen3.6-27B (dense-hybrid)
- Qwen3.6-35B-A3B (MoE-hybrid)
- Gemma-4-31B (dense SWA)
- Gemma-4-26B-A4B (MoE SWA)

at 4k–213k context, ubatch 1024/2048. For Qwen-family models, predictions land within tens of MiB; Gemma models are over-estimated modestly (the safe direction). The overhead is roughly **independent of model depth's KV footprint** — it grows with ubatch (scratch) and context (attention mask), so the prior `n_layers × n_embd × ubatch` formula (context-independent) was wrong in both directions.

### Discrete-GPU overhead (CUDA/ROCm)

The discrete overhead model has three parts. All inputs are GGUF-derived (`embedding_length`, `expert_count`, `block_count`, sliding-window pattern), never from name parsing.

- `discrete_overhead_base_bytes(arch, ubatch_size)`:
  - Context-independent: graph compute scratch (∝ ubatch × model width), MoE expert gather/scatter buffers, and (for Gemma with sliding-window) per-layer-input embedding tables.
  - If `n_embd == 0` or `n_layers == 0` (unknown architecture): flat 256 MB fallback.
  - 200 MiB floor in `base_bytes` (CUDA context minimum).

- `discrete_overhead_ctx_bytes_per_token(arch, ubatch_size)`:
  - Context-dependent: attention mask and per-layer prefill scratch that grow linearly with context, `n_layers`, and per-head dimension.
  - Uses `max(head_dim, global_head_dim)` for Gemma-style models with wider global heads.
  - Formula: `0.46 × n_layers × max(head_dim, global_head_dim) × (0.8 + 0.2 × ubatch/1024)`.

- `discrete_overhead_bytes(arch, ubatch_size, context_size)`:
  - Total discrete overhead = base + (per_token × context_size).

**Calibrated on RTX 5090 32 GB (WDDM), llama.cpp b9728, `--parallel 1 --kv-unified -fa on`, q8_0 KV, full GPU offload**, across:

- Qwen3.6-27B (dense-hybrid)
- Qwen3.6-35B-A3B (MoE-hybrid)
- Gemma-4-31B (dense SWA)
- Gemma-4-26B-A4B (MoE SWA)

at 4k–213k context, ubatch 1024/2048. For Qwen-family models, predictions land within tens of MiB; Gemma models are over-estimated modestly (the safe direction). The overhead is roughly **independent of model depth's KV footprint** — it grows with ubatch (scratch) and context (attention mask), so the prior `n_layers × n_embd × ubatch` formula (context-independent) was wrong in both directions.

On unified memory (Metal), the discrete functions are not used — `metal_overhead_bytes` (above) handles it, plus the headroom reserve.

### `full_estimate`

Sums all components. How weights are split depends on platform and model type:

- **Unified memory** (Metal): all weights go into VRAM; `ram_bytes = 0`.
- **Discrete GPU + MoE**: uses `moe_weight_split(model_size, arch, n_cpu_moe)`.
- **Discrete GPU + dense**: uses `dense_weight_split(model_size, arch, gpu_layers)`:
  - `gpu_layers < 0`: all weights in VRAM (automatic/all-GPU).
  - `gpu_layers == 0`: all weights in CPU RAM.
  - Otherwise: VRAM = `bytes_per_layer × gpu_layers` (or proportional fallback), rest in RAM.

Inside `full_estimate`:

```
weight_vram, ram:
  if is_unified_memory:
      all weights in VRAM
  else if arch.is_moe():
      moe_weight_split(...)
  else:
      dense_weight_split(...)

kv              = kv_cache_bytes(arch, context, slots, ctk, ctv)
linear_state    = arch.linear_attn_state_bytes
mmproj          = arch.mmproj_bytes
mtp             = mtp_overhead_bytes(model_size, arch.mtp_depth)

overhead:
  if is_unified_memory:
      metal_overhead_bytes(arch, ubatch, kv)
  else:
      discrete_overhead_bytes(arch, ubatch, context)

total = weight_vram + kv + linear_state + mmproj + mtp + overhead
```

RAM headroom and WontFit:

- `ram_headroom_bytes = available_ram_bytes - ram_bytes`.
- On discrete GPUs, if `ram_headroom_bytes < 0` and there are CPU-offloaded weights, `full_estimate` returns:
  - `recommendation = WontFit`
  - `note = "CPU-offloaded weights exceed available system RAM."`

When `available_vram_bytes == 0`, recommendation is always `Risk` with note "Memory size unknown; estimate is best-effort."

Recommendation thresholds (VRAM-based):

| Result | Discrete GPU | Unified Memory |
|--------|-------------|----------------|
| `Fit` | total ≤ 82% of available VRAM | same |
| `Tight` | total ≤ 100% | same |
| `Risk` | 100–120% (CPU spill possible) | **never** — unified memory skips Risk and jumps to WontFit |
| `WontFit` | > 120% | > 100% |

Plus a separate RAM-based condition:

- On discrete GPUs, if CPU-offloaded weights exceed available system RAM,
  `full_estimate` forces `WontFit` (even if VRAM alone would allow Fit/Tight).

Rationale: on unified memory there is no graceful CPU-spill path — once you exceed available memory, macOS begins compression and paging. So Risk is only offered on non-unified-memory systems where the OS can spill to system RAM without thrashing.

On unified-memory Macs: the preset editor and spawn wizard show an mlock warning when result is Tight. mlock pins model memory so macOS cannot reclaim it; with an already tight estimate, this can push the OS into memory compression or swap.

### `max_context`

Binary search (sliding-window models) or direct formula (standard):

```
max_context(model_size, arch, ctk, ctv, parallel_slots, ubatch,
            n_cpu_moe, available_vram, fit_granularity,
            headroom_fraction, n_ctx_train, is_unified_memory) → u64

usable = available_vram × (1 − headroom_fraction)

# base overhead (context-independent), per-token slope, and KV multiplier
if is_unified_memory:
    base_overhead    = metal_overhead_base_bytes(arch, ubatch)
    overhead_slope   = 0
    kv_overhead_mult = 1 + METAL_KV_OVERHEAD_FRACTION   # Metal ctx overhead = ~6.5% of KV
else:
    base_overhead    = discrete_overhead_base_bytes(arch, ubatch)
    overhead_slope   = discrete_overhead_ctx_bytes_per_token(arch, ubatch)
    kv_overhead_mult = 1

fixed       = weight_vram + mmproj + mtp + linear_state + base_overhead
kv_budget   = (usable − fixed) / kv_overhead_mult   # reserve Metal's KV-proportional overhead

# Binary search (sliding-window) or direct solve (standard/hybrid).
# Discrete charges a per-token overhead slope alongside the KV cache; Metal instead reserves
# its KV-proportional overhead by shrinking kv_budget (mult), since it scales with the KV size.
```

Direct (standard / hybrid):

```
max_ctx = kv_budget / (kv_bytes_per_token + overhead_slope)
```

Binary search bounds: lo = 512, hi = 2,097,152. For sliding-window models, binary-search is used to find the largest context; for standard/hybrid, the direct formula is used.

Zero-guard: returns 0 when:

- `available_vram_bytes == 0`, or
- fixed costs alone (weights in VRAM + mmproj + MTP + linear-state + overhead) exceed `usable` (available VRAM after headroom is reserved), or
- for sliding-window models, `kv_cache_bytes(512) > kv_budget` (minimum context doesn't fit).

**Important**: `is_unified_memory` is threaded from the caller. For Mac/Metal: always true. For Windows/Linux CUDA: always false. The auto_size and quant advisor both pass this flag correctly.

### MTP detection

Primary: GGUF fields (`nextn_predict_layers`, `next_n_token_count`, `num_nextn_predict_layers`, `multi_token_prediction_depth`). When present, these set `mtp_depth` directly via introspection.

The filename keyword branch in `from_name_and_params()` is a legacy pre-download VRAM-estimator fallback (degraded/best-effort). Confirmed primary-model metadata remains authoritative for Spawn Wizard recommendations and launch state. Separate draft-model/MTP-head files often have no introspectable head metadata, so filename/repository hints remain a flexible provisional discovery fallback; they must be labeled inferred and must not claim a confirmed MTP depth. The resulting estimate includes MTP overhead in its breakdown and fixed-cost context budget. EXAONE 4.5 unconditionally uses MTP (all sizes).

### `find_min_cpu_moe_to_fit_weights`

Binary search over `[0, n_layers]` to find the smallest `n_cpu_moe` whose weight footprint fits in VRAM.

Uses the platform-appropriate base overhead (`metal_overhead_base_bytes` on unified memory, `discrete_overhead_base_bytes` on CUDA/ROCm) as its overhead estimate, then checks whether `moe_weight_split` yields a VRAM footprint that fits in 80% of available VRAM minus overhead, mmproj, and MTP overhead.

If even with all experts on CPU it still doesn't fit, returns `n_layers`.

### `auto_size`

Orchestrates all of the above to produce `AutoSizeResult`:

1. Determine ubatch: 1024 for Agentic/General, 512 for Roleplay
2. Compute headroom via `compute_headroom(available_vram_bytes, is_unified_memory)`:
   - **Unified memory**: 10% base, capped at 2 GB.
   - **Discrete GPU**: 5% base, capped at 1.5 GB.
   On large budgets (>>30 GB) the cap takes effect; on smaller systems the percentage applies.
3. Find minimum `n_cpu_moe` to fit model weights via `find_min_cpu_moe_to_fit_weights`
4. For each KV quant × context combination (q8_0, q4_0, f16):
   - Call `max_context(…, is_unified_memory)` to find the largest context that fits
   - Call `full_estimate` at that context to get the full breakdown
5. Pick the standard scenario (q8_0) as the recommended result
6. Emit warnings for: agentic + low KV quant, context > n_ctx_train, MoE offload speed penalty
7. Emit notes for: MoE offload ratio, MTP overhead, mmproj presence

`n_ctx_train` is a hard cap: even if more context fits in VRAM, the auto-size function will not recommend exceeding the model's training context length.

Auto-size ignores client-provided `n_ctx`, `ctk`, `ctv`, `n_cpu_moe` and chooses its own values.

For MoE models, `build_scenarios` adds an "extended" scenario with aggressive CPU offload (~75% of layers on CPU) to show a higher-context, slower option.

When `gguf_arch == "qwen35"`, auto-size disambiguates: `block_count ≥ 75` → Qwen3.5; `< 75` → Qwen3.6.

### `estimate_model_size_bytes`

```
size_bytes = param_b × 1e9 × bpw / 8
```

Used for pre-download estimates where the actual file size is not yet known.

### Quant advisor (`quant_comparison_table`)

For each candidate quantization, it:

- Estimates file size from `param_b` and the quant's BPW.
- Checks "fits" only if the model plus a minimal KV cache (8 K tokens at q8_0) is under available VRAM.
- Scores each quant as `min(max_ctx_q8, 128K) × quality_weight` if it fits.

Gemma 4 QAT:

- If the model name contains both `"gemma-4"` (or `"gemma4"`) and `"qat"`, then Q4_0 is treated as Excellent quality.
- When Q4_0 fits, it is chosen as the recommended quant instead of higher-bit defaults.

When called by the UI after introspection, the quant-compare endpoint can receive explicit arch fields (`n_layers`, `n_kv_heads`, `head_dim`, `global_head_dim`, `n_experts`, `mtp_depth`, etc.) that override the name heuristic, improving accuracy for renamed finetunes.

### Legacy `estimate_vram` wrapper

Backward-compat wrapper kept for existing `/api/vram/estimate` callers. Uses a legacy KV heuristic (`context × effective_batch × 64 × kv_bpe`) and no per-layer formula. Always builds a default `ModelArch` with all zeros; `new` code should use `full_estimate`.

If `speculative_decoding` is true, adds `model_size_bytes / 8` to the total.

---

## API: mmproj Path Support

The `/api/vram-estimate` (breakdown) endpoint accepts either:

- `mmproj_bytes` (u64): explicit byte count
- `mmproj_path` (string): filesystem path; the server stats the file to get the size

If both are provided, `mmproj_bytes` takes precedence. The path is resolved relative to the server's working directory. This allows the preset editor to pass the configured mmproj path directly without needing a separate stat request.

---

## API Endpoints

### POST /api/vram-estimate

Architecture-aware, backend-aware VRAM breakdown endpoint.

- Requires: `api-token` (Authorization header)
- Requires (body):
  - For llama.cpp: **either** `model_path` (local GGUF file) **or** `hf_repo_id` + `hf_file_path` + `model_size_bytes`.
  - For Rapid-MLX: **either** `model_path` (local MLX directory or HF-repo-style alias) **or** `hf_repo_id` + optional `hf_file_path` + `model_size_bytes`.
- Optional (body): `n_ctx`, `gpu_layers`, `parallel_slots`, `ubatch_size`, `ctk`, `ctv`, `n_cpu_moe`, `available_vram_bytes`, `available_ram_bytes`, `is_unified_memory`, `mmproj_path`, `mmproj_bytes`
  - `gpu_layers` (int): for dense models on discrete GPU (llama.cpp): how many layers on GPU; negative = all-GPU, zero = all-CPU.
  - `available_ram_bytes` (u64): system RAM available; used to check CPU-offloaded weight budget.
- Backend selection:
  - The `backend` field (preferred) or legacy `engine` alias selects the inference backend:
    - `"llama_cpp"` (default, omitted): GGUF-based llama.cpp path.
    - `"rapid_mlx"` / `"mlx"`: Rapid-MLX path (Apple Silicon / unified memory only).
- llama.cpp behavior (default):
  - **Local path**: reads GGUF metadata from `model_path` (primary); falls back to `from_name_and_params()` on parse failure.
  - **HuggingFace (no local file yet)**: `crate::hf::fetch_gguf_header_metadata()` issues an HTTP **Range** request for the first few MB of the `.gguf` (the KV header sits at the file start), parses it with `read_gguf_metadata_from_bytes()`, and uses the caller-supplied `model_size_bytes` (from the HF file listing) for weights. This gives the model's **real** architecture before downloading 16 GB. Requires HTTP 206 (partial content); on any failure (gated repo, offline, no range support) it falls back to the name heuristic so the caller still gets a rough estimate.
- Rapid-MLX behavior:
  - `is_unified_memory` is forced to `true` server-side (no discrete-GPU/CPU-spill path).
  - **Local path**: `model_path` is interpreted as an MLX model directory. Its `config.json` is parsed into the normalized `ModelMemoryProfile`, including authoritative nested `text_config` geometry, full/local/linear layer groups, recurrent state, Gemma global/local heads, MoE topology, and MTP/companion evidence.
    - `model.safetensors.index.json` for exact weight-byte accounting (`metadata.total_size` when present, otherwise the real on-disk shard file sizes).
  - **HF-repo-style alias**: if `model_path` is not a local directory but matches `"org/repo"` (e.g. `"mlx-community/Qwen3-30B-A3B-4bit"`), it is treated as an `hf_repo_id`; the server fetches `config.json` from HuggingFace.
  - **Explicit `hf_repo_id`**: for Rapid-MLX, `hf_repo_id` plus optional immutable `hf_repo_revision` fetches `config.json` directly. `hf_file_path` is never treated as a config filename. Referenced text configs are revision-pinned, bounded, cycle-checked, and limited to safe relative JSON paths. `model_size_bytes` is required when the tree lookup cannot determine the weight size.
  - **Degraded**: if required config fields (`hidden_size`, `num_hidden_layers`, `num_attention_heads`) are missing or unrecognized, the architecture is built via `ModelArch::from_name_and_params()` and `evidence` is set to `"degraded"`. This never silently presents a heuristic guess as authoritative.
  - Optional `mlx_prefix_cache_tokens` + `mlx_prefix_cache_bits` (4 or 8, default 8) reserve a **separate** stored budget for Rapid-MLX's compressed prefix cache. This is intentionally NOT a reduction of `kv_cache_bytes`: cached entries are decompressed back to the active compute dtype before reuse, so active-request KV is unaffected by how much prefix cache exists.
  - A retained-cache recommendation is optional growth, not mandatory fit. The
    estimator presents the base launch requirement separately from the chosen
    cache budget and reserve. For qualified one-user Rapid coding workloads,
    8 GiB is the baseline retained-cache choice; 16 GiB is a branch-retention
    option. Disk checkpoints are excluded because they are snapshot writes,
    not automatic RAM-cache restoration.
  - Rapid-native policy fields are `kv_cache_dtype` (`bf16`, `int8`, or `int4`), `reasoning_mode`, and `turboquant_mode` (`v4`, `k8v4`, or `none`). The estimator never accepts llama.cpp's `ctk`/`ctv` vocabulary for a Rapid result. Reasoning resolves the active KV dtype to `int8`.
   - Optional `workload_scenario` maps page-1 use-case selection to backend memory policy. Valid keys and their memory policies:
     - `interactive_coding_agent` — coding agent workload (default, 80% priority), 128K planning context, 32K retained cache, TurboQuant eligible.
     - `general_chat` — standard chat, 32K planning context, 8K retained cache.
     - `roleplay_storytelling` — long-context narrative, 64K planning context, 32K retained cache.
     - `tool_research_agent` — multi-session tool/research, 128K planning context, 48K retained cache, 2 parallel slots.
     - `batch_eval` — batch/evaluation, 8K planning context, 0 retained cache, 4 parallel slots.
   - The workload scenario affects: recommended KV dtype, TurboQuant eligibility, MTP eligibility, parallel slot recommendations, and retained-cache sizing.
  - TurboQuant is retained-prefix storage only. It never reduces model weights, recurrent state, MTP, prefill, or the transient decompression peak. A local Rapid 0.10.17 receipt showed that `--kv-cache-turboquant k8v4` owns the active cache path: the runtime reports its active compute KV as `bf16` and does not apply a simultaneously requested `--kv-cache-dtype int4`. `k8v4` is an Advanced trial: it becomes effective only after exact model/revision qualification; an unknown community finetune is estimated as Standard retained storage and receives an explicit fallback reason. The Rapid argv is `--kv-cache-turboquant {v4,k8v4,none}`.
- Output fields: `weights_bytes`, `kv_cache_bytes`, `linear_attn_state_bytes`, `mmproj_bytes`, `mtp_bytes`, `overhead_bytes`, `total_bytes`, `available_bytes`, `headroom_bytes`, `ram_bytes`, `available_ram_bytes`, `ram_headroom_bytes`, `recommendation`, `note`
- Additional output fields (both backends; zero/"measured" for GGUF llama.cpp):
  - `mlx_prefix_cache_bytes` (u64) — the separate compressed prefix-cache budget described above (non-zero only for Rapid-MLX).
  - `evidence` (`"measured"` | `"approximate"` | `"degraded"`) — how much of the breakdown is backed by real hardware calibration vs. a formula-based approximation vs. a heuristic fallback from incomplete metadata:
    - `"measured"`: llama.cpp with GGUF metadata and hardware-calibrated overhead (Metal or discrete). This is the strongest guarantee.
    - `"approximate"`: Rapid-MLX with real MLX config metadata but an uncalibrated overhead formula (see next section). Every Rapid-MLX estimate is at best `"approximate"`.
    - `"degraded"`: one or more required architecture fields were missing/unrecognized and a name/param heuristic was used instead of real model metadata (applies to both backends).
- Consumers: Spawn Wizard, Preset Editor, Setup view, **and the Models modal HF-browse preview bar** (`updateVramDisplay` in `static/js/features/models.js`) — all share this one endpoint. The Spawn Wizard's VRAM bar tooltip appends a plain-text note when `evidence` is `"approximate"` or `"degraded"`. No independent client-side VRAM math exists — this is always backend-driven only.

### GET /api/memory-availability and POST /api/memory-availability/fit

These authenticated endpoints provide the live, backend-owned availability sample used alongside an estimate on unified-memory systems. They deliberately distinguish capacity from current launch readiness:

- `GET /api/memory-availability` returns a fresh capacity snapshot. `total_unified_bytes` and `configured_ceiling_bytes` are informational and are never presented as currently available memory.
- `POST /api/memory-availability/fit` accepts `required_bytes`, `launch_intent` (`additional_generation` or `replace_existing`), and an optional `replace_runtime_bytes`. The replacement credit is valid only for a measured, app-owned runtime that the launch will stop.
- The response reports `current_safe_availability_bytes`, `after_reclaim_bytes`, `after_closing_apps_bytes`, the selected `required_bytes`, and a target-specific state: `safe_now`, `conditional_after_reclaim`, `after_closing_apps`, or `unsafe`.

Reclaim and close-app figures are conditional scenarios, not free memory. Disk-cache purge does not claim to free a model heap, Metal allocation, or wired memory.

### Rapid-MLX overhead — approximate, not yet hardware-calibrated

Unlike llama.cpp's Metal (`metal_overhead_bytes`) and discrete-GPU (`discrete_overhead_bytes`) overhead, which are calibrated against real measured hardware footprints (see "Backend-Specific Accuracy" below), **Rapid-MLX's overhead (`mlx_overhead_bytes` in `src/llama/vram_estimator/estimate.rs`) is a documented, formula-based approximation**:

- Same per-layer order of magnitude as the Metal calibration, inflated by:
  - A fixed 25% safety margin on the base per-layer/ubatch cost.
  - A conservative 8% KV overhead fraction (vs. Metal's measured 6.5%).
- It is **not** derived from real Apple Silicon Rapid-MLX measurements and must not be presented as such.
- Every estimate with `Backend::RapidMlx` sets `EstimateEvidence::Approximate` (or `Degraded` when the source config was incomplete).
- Recalibrating this against real Rapid-MLX process-footprint measurements (same methodology as "Recalibrating the discrete overhead" and "Recalibrating the Metal overhead") is an open follow-up.

### API helpers (`build_arch_from_body`)

Used by `quant-compare` and `auto-size` endpoints when GGUF is not present or for fields not in the GGUF.

- Role: merges a name-based heuristic `ModelArch` with explicit architecture fields from the request body.
- Priority: explicit body fields take precedence; fallback to heuristic defaults.
- When `gguf_arch` is present, uses `gguf_arch_to_heuristic_name()` to pick the correct family.

### VRAM bar (UI)

The VRAM bars consume `/api/vram-estimate` directly (single source of truth) — they are not independent client-side reimplementations, so their numbers always match the backend estimator. The Models-modal HF-browse preview (`updateVramDisplay` in `static/js/features/models.js`) estimates at a fixed preview context (16K) using HF range-fetch introspection; once a model is on disk, the Spawn Wizard / Preset Editor introspect the local file and estimate at the real configured context.

---

## Backend-Specific Accuracy

| Backend | Model size | KV cache | Overhead | Total accuracy |
|---------|-----------|----------|-----------------|----------------|
| llama.cpp Metal (Apple Silicon) | ✓ exact from GGUF | ✓ formula | ✓ M5 Max-calibrated (`metal_overhead_bytes`) | ±0.05 GiB |
| llama.cpp CUDA (Windows/Linux) | ✓ exact from GGUF | ✓ formula | ✓ calibrated when n_embd known | ±0.5 GiB |
| llama.cpp CUDA, n_embd unknown | ✓ exact from GGUF | ✓ formula | 256 MB fallback | ~2–3 GiB low |
| Rapid-MLX (Apple Silicon) | ✓ from safetensors / HF | ✓ formula | Approximate (25% safety margin) | Not yet calibrated |

Discrete overhead (CUDA/ROCm) is calibrated on RTX 5090 32 GB using measurement-grounded formulas. When `n_embd` is unknown (no GGUF or missing `embedding_length`), it falls back to a 256 MB flat reserve and underestimates overhead.

Rapid-MLX overhead is formula-based (see "Rapid-MLX overhead" above) and deliberately conservative.

**Mac M5 Max calibration** (Q5_K_S, 262k ctx, q8_0 KV):

- Estimated: 18.11 + 0.87 + 8.00 + 0.30 = 27.27 GiB
- Actual: ~27 GiB observed (model loaded, memory pressure spike to ~60 GB = 33 GB baseline + 27 GB model)

**Windows RTX 5090 calibration** (Q4_K_M, 212k ctx, q8_0 KV, ubatch=1024, flash_attn=on):

- Estimated: 16.12 + 0.87 + 6.50 + 0.38 (base) + 2.54 (ctx overhead) = 26.41 GiB
- Actual nvidia-smi: 28.10 GiB (includes ~1.5 GiB WDDM display apps already in available_vram)
- Net llama-only: ~26.6 GiB → within ~0.2 GiB of estimate

---

## Quantization Table

All quantizations recognized by the estimator, with bits-per-weight and KV bytes-per-element:

| Quant | BPW | KV BPE | Quality | imatrix | Large MoE only |
|-------|-----|--------|---------|---------|----------------|
| F32 | 32.0 | 4.0 | Reference | — | — |
| F16 / BF16 | 16.0 | 2.0 | Reference | — | — |
| Q8_0 | 8.5 | 1.0 | Excellent | — | — |
| Q6_K | 6.5625 | 0.75 | Very Good | — | — |
| Q5_K_M | 5.69 | 0.625 | Very Good | — | — |
| Q5_K_S | 5.52 | 0.625 | Very Good | — | — |
| Q5_0 | 5.5 | 0.625 | Very Good | — | — |
| Q4_K_M | 4.85 | 0.5 | Good | — | — |
| Q4_K_S | 4.58 | 0.5 | Good | — | — |
| Q4_0 | 4.55 | 0.5 | Acceptable | — | — |
| Q4_1 | 4.7 | 0.5 | Acceptable | — | — |
| IQ4_XS | 4.25 | 0.5 | Very Good | ✓ | — |
| IQ4_NL | 4.5 | 0.5 | Good | ✓ | — |
| Q3_K_M | 3.875 | 0.375 | Acceptable | — | — |
| Q3_K_S | 3.4375 | 0.375 | Fair | — | — |
| Q3_K_L | 4.0 | 0.375 | Acceptable | — | — |
| IQ3_M | 3.6875 | 0.375 | Acceptable | ✓ | — |
| IQ3_S | 3.5 | 0.375 | Fair | ✓ | — |
| IQ3_XS | 3.3125 | 0.375 | Fair | ✓ | — |
| IQ3_XXS | 3.0625 | 0.375 | Fair | ✓ | — |
| Q2_K | 2.625 | 0.25 | Reduced | — | — |
| IQ2_M | 2.6875 | 0.25 | Reduced | ✓ | ✓ |
| IQ2_S | 2.5 | 0.25 | Reduced | ✓ | ✓ |
| IQ2_XS | 2.3125 | 0.25 | Reduced | ✓ | ✓ |
| IQ2_XXS | 2.0625 | 0.25 | Reduced | ✓ | ✓ |
| IQ1_M | 1.75 | 0.125 | Very Low | ✓ | ✓ |
| IQ1_S | 1.5625 | 0.125 | Very Low | ✓ | ✓ |
| TQ1_0 | 1.69 | 0.125 | Very Low | ✓ | ✓ |
| TQ2_0 | 2.0 | 0.25 | Reduced | ✓ | ✓ |

`Large MoE only`: the quant advisor hides these for models that are not large MoE.

KV BPE is used only for KV cache estimation (`kv_cache_bytes`). The `ctk` / `ctv` names map to these values directly.

---

## GGUF Metadata Integration (llama.cpp)

When a GGUF file is present, `gguf_meta.rs` reads the model's real KV header and `GgufMetadata::to_model_metadata()` builds the metadata struct. `ModelMetadata::to_arch()` (in `spawn_wizard.rs`) then converts it into `ModelArch`.

**Structural fields come from the file, not from name guesses** — the name heuristic is always run first as a scaffold, but is then overridden field-by-field with GGUF data. For "weak" heuristic results (no MoE, no hybrid, no sliding-window) with a known GGUF architecture, it may re-run the heuristic using the GGUF-derived family name. The breakdown endpoint (`/api/vram-estimate`) and `auto_size` (`/api/vram/auto-size`) both build their arch through this real-data path when `model_path` points at an on-disk GGUF.

The tensor directory (shapes and element counts) is always parsed, even for range-fetched prefixes:
- **Parameter counts** (`tensor_param_count`, `expert_param_count`) are derived from tensor shapes in the GGUF header, so they are exact for both local files and range-fetched prefixes. These feed the active-parameter formula at the top of this doc.
- **Byte sizes** (`tensor_bytes_total`, `layer_bytes_total`, `expert_bytes_total`) require the full file; they are only available for complete local files. `bytes_per_layer()` (used by `dense_weight_split`) is one such derived value.

### Key mapping

| GGUF key | ModelArch field | Notes |
|----------|----------------|-------|
| `{arch}.block_count` | `n_layers` | |
| `{arch}.attention.head_count` | (used as n_head) | Used to derive head_dim = n_embd / n_head when key_length missing |
| `{arch}.attention.head_count_kv` | `n_kv_heads` | Scalar, or per-layer array on Gemma 3/4 (see below) |
| `{arch}.attention.key_length` | `head_dim` / `global_head_dim` | Global K/V dim; on Gemma it's the wide (512) global value |
| `{arch}.attention.key_length_swa` | `head_dim` (local) | Narrow (256) local sliding-window dim on Gemma 4 |
| `{arch}.attention.sliding_window` | `local_attn_window` | e.g. Gemma 4 = 1024 |
| `{arch}.attention.sliding_window_pattern` | `n_global_attn_layers` | Per-layer bool array; count of `false` = global layers |
| `{arch}.full_attention_interval` | `n_attn_layers` | Hybrid DeltaNet: `n_attn_layers = block_count / interval` |
| `{arch}.ssm.{inner_size,state_size,conv_kernel}` | `linear_attn_state_bytes` | DeltaNet recurrent-state size |
| `{arch}.embedding_length` | `n_embd` | Used for discrete overhead formula |
| `{arch}.expert_count` | `n_experts` | |
| `{arch}.expert_used_count` | `n_experts_used` | Routed experts (shared expert counted separately) |
| `{arch}.feed_forward_length` | `n_ff` | Available for downstream; not used directly in VRAM formulas |
| `{arch}.nextn_predict_layers` (and `next_n_token_count`, `num_nextn_predict_layers`) | `mtp_depth` | MTP head count |
| `{arch}.context_length` | `n_ctx_train` | Hard cap passed to `auto_size`/`max_context` |
| `general.architecture` | (selects family heuristic) | Central for choosing correct family when filename is ambiguous (e.g. Pantheon-27B as qwen35). See `gguf_arch_to_heuristic_name()` |
| `general.parameter_count` | (used for param_b) | Sanity checks and param_b(); not in VRAM formulas but important for ID/MoE |

`gguf_arch_to_heuristic_name()` maps arch tags to heuristic names (e.g., `qwen35`/`qwen35moe` → Qwen3.6 by default; `qwen3_coder_next` → Coder-Next; `gemma4` → Gemma 4).

### Hybrid attention (Qwen3-Next / DeltaNet)

Only `block_count / full_attention_interval` layers carry a KV cache; the rest hold a fixed (context-independent) DeltaNet state sized from SSM fields: `ssm_inner_size × (ssm_state_size + ssm_conv_kernel) × 2 B` per linear layer, multiplied by the linear layers count. Both are read from the file, so a finetune with a non-standard layer count (e.g. the 35B-A3B's real `block_count` is **41**, not the 40 the name heuristic assumes) is handled correctly.

### Gemma 3/4 alternating attention

`attention.head_count_kv` is a per-layer array; the global-layer KV head count (smaller, e.g. 2/4) and local-layer count (larger, e.g. 8/16) are read at the positions marked global/local by `sliding_window_pattern`. The global layers use `key_length` (512) over the full context; the local layers use `key_length_swa` (256) capped at `sliding_window` (1024).

The GGUF arch string `"qwen35"` is mapped via `gguf_arch_to_heuristic_name()` (llama.cpp's shared tag for Qwen3.5/3.6); since `n_layers` and `n_attn_layers` now come straight from the file, the Qwen3.5-vs-3.6 distinction no longer affects the KV math.

### Implementation details

- **Introspection cache**: results are cached in `~/.config/llama-monitor/model-cache/<sha256>.json`, keyed by file path + size + mtime.
- **Fallback**: if the direct GGUF read fails (corrupt/partial file), the system falls back to running `llama-server --print-model-metadata` and parsing its output.

---

## MLX Metadata Integration (Rapid-MLX)

For Rapid-MLX models, metadata comes from `config.json` + `model.safetensors.index.json`
instead of a GGUF header. See `src/inference/rapid_mlx/mlx_meta.rs`.

- **config.json** (primary): an HF-transformers-style architecture config:
  - Required for "exact" evidence: `hidden_size`, `num_hidden_layers`, `num_attention_heads`.
  - MoE: `num_experts`, `num_experts_per_tok` (and alternate field names via `#[serde(alias)]`).
  - Sliding-window: `sliding_window`, `sliding_window_pattern`.
  - Rapid-MLX `quantization` block: `bits`, `group_size`.
  - Draft/MTP: `draft_model` or `speculative_config` sub-configs.
  - Vision: presence of `vision_config` flags the model as needing an mmproj-equivalent budget.
- **model.safetensors.index.json**:
  - Used for exact weight-byte accounting:
    - `metadata.total_size` when present (HF-exported indexes).
    - Otherwise, on-disk shard file sizes are summed.
  - Shard names are validated: no absolute paths, no `..` traversal, `.safetensors` only.
- **Evidence and mapping**:
  - `ModelMemoryProfile` is the authoritative MLX geometry input. It preserves field evidence and uses nested `text_config` rather than wrapper fields when present.
  - Incomplete/unreadable config falls back to a name heuristic only with `evidence = "degraded"`; it is never presented as exact metadata.
  - The profile maps into shared `ModelArch` geometry (layers, heads, KV heads, head dimensions, full/local/linear groups, MoE, recurrent state, and companions) without importing llama.cpp runtime vocabulary.
  - Exact per-layer byte size is computed from real on-disk/HF-listed size:
    `bytes_per_layer = model_size_bytes / n_layers`.
- **HuggingFace pre-download**:
  - For Rapid-MLX, the VRAM estimator fetches `config.json` at the selected revision without range-fetching and preserves that revision in the profile evidence.
  - Weight size is resolved from the HF tree API (`crate::hf::resolve_mlx_repo_size_bytes`).
  - If `config.json` fetch fails or is missing required fields, the model is still estimable
    via the name heuristic with `evidence = "degraded"`.

---

## MLX context-fit cards

Rapid-MLX's wizard hardware step does not offer KV-quantization-based context-fit
cards the way llama.cpp does. That axis is inert for Rapid-MLX: the runtime's
reasoning profile pins active KV to int8 on every launch regardless of the requested
dtype, and TurboQuant/PFlash are withheld pending qualification — see
[rapid-mlx-runtime.md](rapid-mlx-runtime.md#reasoning-profile-and-kv-cache-dtype) for
why. A KV-quant card set would therefore be presenting three views of one fixed value.

With KV dtype, TurboQuant, and PFlash all fixed, the levers that actually move
Rapid-MLX unified-memory occupancy are context length (the dominant term at fixed int8
KV), concurrency (`max_num_seqs` × `max_concurrent_requests`, which multiply active
KV), the retained prompt-cache budget (`retained_cache_mib` + `hybrid_cache_entries`),
and `gpu_memory_utilization` as the ceiling the others are measured against. The wizard
instead offers three *workload-shaped* cards that vary concurrency and retained-cache
budget at the user's chosen context, each requesting `/api/vram-estimate` with the same
`buildEstimateBody()` plus a different MLX policy spread:

| Card | `max_num_seqs` | Retained cache | Framing |
|---|---|---|---|
| **Single interactive user** *(default/recommended)* | 1 | measured coding-agent recommendation (8 GiB / 16 entries) | One conversation at a time, warm prompts reused. |
| **Long single context** | 1 | 0 | Maximum room for one very long conversation; nothing retained between prompts. |
| **Shared / multi-client** | 4 | 8 GiB | Several clients at once; each admitted request reserves its own active KV. |

Alongside the cards, the fixed facts are rendered once, not per-card, since they do
not vary by card: `KV: int8 (pinned by reasoning profile)`,
`TurboQuant: off (awaiting receipt)`, `PFlash: off`.

---

## Known Limitations and Calibration Notes

| Issue | Scope | Status |
|-------|-------|--------|
| `mtp_overhead_bytes` uses a 1.5% heuristic | All MTP models | Estimate; actual varies by architecture |
| Gemma 4 n_embd values for E2B/E4B/12B/26B-A4B are estimated | Gemma 4 | Overridden by GGUF embedding_length when file is local |
| Qwen3.6-35B-A3B n_embd=4096 is estimated | Qwen3.6-35B-A3B | Overridden by GGUF when file is local |
| Qwen3.5-122B n_embd=7168 is estimated | Qwen3.5 > 80B | Overridden by GGUF when file is local |
| Qwen3.5 expert counts (256/9) only confirmed for 122B-A10B | Qwen3.5 < 122B | Applied to smaller sizes; update when those release |
| Generic MoE suffix `n_experts_used` is heuristic (11/9/8) based on sparsity | Non-Qwen/Gemma MoE | Approximation; exact values from GGUF introspection take precedence |
| `expert_fraction` default 0.65 is a rough average | All MoE models | Overridden per-family; calibrate when architecture is public |
| `discrete_overhead_base_bytes` uses 256 MB fallback when n_embd unknown | Any model without GGUF `embedding_length` | Conservative; may over-reserve vs actual CUDA usage |
| Name heuristics are heuristic-only fallback | All pre-download / no-GGUF estimates | Do not rely on them when a GGUF file is present — GGUF is authoritative |
| Rapid-MLX overhead is approximate | All Rapid-MLX estimates | 25% safety margin + 8% KV fraction; recalibration via process footprint is an open task |

---

## Recalibrating the discrete overhead (measurement methodology)

The `discrete_overhead_*` constants in `estimate.rs` were fit to **direct VRAM measurements on an RTX 5090 32 GB** (Windows). If you ever need to re-measure (new llama.cpp build, new arch, or a discrepancy report), this is the exact procedure — it is not obvious and has several traps.

**How to measure total VRAM for a config:**
1. Kill any running `llama-server`, then read a clean baseline: `nvidia-smi --query-gpu=memory.used --format=csv,noheader` (desktop apps only).
2. Launch the model fully on GPU with the config under test:
   `llama-server -m MODEL -c CTX -ub UB -b 4096 -fa on -ctk q8_0 -ctv q8_0 -ngl 99 -fit off --parallel 1 --kv-unified --no-warmup --no-mmap`
3. After "server is listening", read `memory.used` again.
4. `server_total = used − baseline`. Then `overhead = server_total − model_file_bytes − KV(ctx)`, where `KV(ctx)` is `kv_cache_bytes()` for the introspected arch.

**Traps (all cost real debugging time):**
- **Windows/WDDM reports per-process VRAM as `[N/A]`** in `nvidia-smi --query-compute-apps`. You MUST use the *total* `memory.used` delta against a clean baseline; there is no per-process number.
- **`--parallel` defaults to 4** ("n_parallel auto"). That inflates buffers ~2×. Always pass `--parallel 1` to match the estimator's 1-slot assumption.
- **`-fit off`** — newer builds auto-fit (`-fit on`) and may silently reduce `-ngl` or suppress the per-buffer allocation log lines. Use `-fit off` and `-ngl 99` for a deterministic full-GPU load.
- **`llama-cli` was merged into `llama-server`** (recent betas). There is no `llama-cli.exe`; use `llama-server` and read the load log / `nvidia-smi`.
- Recent builds **do not print** `CUDA0 compute buffer size` lines at all — the `nvidia-smi` delta is the only ground truth.

**Calibration dataset (parallel=1, fa on, q8_0 KV, ngl 99, no mmproj), measured overhead in MiB:**

| Model | n_layers | head_dim(max) | n_embd | MoE | SWA | ctx | ub | overhead |
|-------|----------|---------------|--------|-----|-----|-----|-----|----------|
| Qwen3.6-27B | 64 | 256 | 5120 | no | no | 4k | 1024 | 215 |
| Qwen3.6-27B | 64 | 256 | 5120 | no | no | 131k | 1024 | 1139 |
| Qwen3.6-27B | 64 | 256 | 5120 | no | no | 213k | 1024 | 1779 |
| Qwen3.6-27B | 64 | 256 | 5120 | no | no | 4k | 2048 | 467 |
| Qwen3.6-27B | 64 | 256 | 5120 | no | no | 213k | 2048 | 2355 |
| Qwen3.6-35B-A3B | 41 | 256 | 2048 | yes | no | 4k | 1024 | 425 |
| Qwen3.6-35B-A3B | 41 | 256 | 2048 | yes | no | 213k | 1024 | 1343 |
| Gemma-4-31B | 60 | 512 | 5376 | no | yes | 4k | 1024 | 1310 |
| Gemma-4-31B | 60 | 512 | 5376 | no | yes | 213k | 1024 | 3704 |
| Gemma-4-26B-A4B | 30 | 512 | 2816 | yes | yes | 4k | 1024 | 803 |
| Gemma-4-26B-A4B | 30 | 512 | 2816 | yes | yes | 213k | 1024 | 2091 |

Key findings that shaped the model: overhead is **linear in context** (the attention mask / per-layer prefill scratch), scales with `n_layers × head_dim` (not KV footprint), grows with ubatch (graph scratch), gets a fixed bump for MoE (expert gather ~260 MiB), and a large per-layer base for Gemma SWA (per-layer-input embeddings ~20 MiB/layer). The current formula reproduces Qwen within tens of MiB and over-reserves Gemma modestly (worst under-prediction across the set: −67 MiB — i.e. essentially never under-reserves).

## Recalibrating the Metal overhead (Apple Silicon)

The Metal path is calibrated separately on **Apple M5 Max** (llama.cpp b9743, Metal). The measurement method differs from CUDA because macOS has no `nvidia-smi`:

- **Use process physical footprint, not a system delta.** Launch `llama-server` (mmap on, the default) and read `footprint <pid>` / `vmmap --summary <pid>` → "Physical footprint". With mmap, the model weights are file-backed and **excluded** from the footprint, so `footprint ≈ KV + overhead` — exactly the overhead component, cleanly isolated. Then `overhead = footprint − kv_cache_bytes(ctx)`.
  - Validation: a `--no-mmap` control makes the footprint jump by the full weight size (~20 GB), confirming weights are excluded under mmap.
- Same flags as CUDA otherwise: `-fa on -ctk q8_0 -ctv q8_0 -ngl 99 -fit off --parallel 1 --kv-unified --no-warmup`.

Findings: Metal overhead = a per-layer base (4.3 MiB/layer dense, 8.8 MiB/layer Gemma SWA) + 0.035 MiB/ubatch + **~6.5% of KV bytes** (stable 0.063–0.068 across all four families). It is far lighter than CUDA and the context part rides on the KV size, so it auto-handles hybrid/windowing/quant.

**mmap is not a throughput knob on Apple Silicon.** `llama-bench` measured identical pp/tg t/s with mmap on vs off (e.g. 35B-A3B: pp512 3163 vs 3150, tg128 113.6 vs 113.8). mmap is zero-copy into Metal; disabling it only slows the initial load and commits the whole model to RAM. The wizard therefore defaults `no_mmap = false` on unified memory, and the preset editor shows a hint recommending it stay off. The real Apple-Silicon perf lever is staying under the memory budget (avoid compression/swap) — which the estimator enforces.

---

## Recalibrating the MLX overhead (Apple Silicon, Rapid-MLX)

The Rapid-MLX overhead constants (`mlx_overhead_base_bytes`, `MLX_KV_OVERHEAD_FRACTION`) in
`estimate.rs` are currently formula-based, not measurement-grounded. The current formula:

- Uses Metal's measured per-layer base (4.3/8.8 MiB) × 1.25 (25% safety margin).
- Uses 8% of KV bytes for context-scaling buffers vs. Metal's 6.5%.

To recalibrate against real Rapid-MLX process-footprint measurements:

- Use the same M5 Max physical-footprint methodology as "Recalibrating the Metal overhead":
  - Start the Rapid-MLX server with a known model and context.
  - Read `footprint <pid>` / `vmmap --summary <pid>` → "Physical footprint".
  - With mmap-style file-backed weights (where applicable), the footprint minus KV
    approximates the overhead component.
- Compare against the current `mlx_overhead_bytes` values.
- If measurements consistently show a lower fraction, reduce the 25% margin and 8% KV
  fraction accordingly; once calibrated, switch `EstimateEvidence` from `Approximate`
  to `Measured` for Rapid-MLX (and update this file).

Until that work is done, Rapid-MLX overhead must continue to be reported as `Approximate`.

### Current active-KV qualification

For Rapid-MLX v0.11.0, active KV is modeled as **BF16** even when
`--kv-cache-dtype int4` or `int8` was requested. This records the observed
behavior in upstream issue #1197 rather than claiming memory savings the
runtime did not realize. The request is still returned beside the effective
value. Revisit this only after the upstream fix is released and a fresh,
pinned receipt verifies the effective active-KV representation.

### Estimator-to-runtime calibration receipt

To recalibrate `mlx_overhead_bytes` against real Rapid-MLX evidence rather than a formula, a
separate **calibration receipt** binds one canonical `/api/vram-estimate` prediction to one
fresh-server-per-cell `model_runtime_benchmark_receipt` cell (see
`docs/reference/model-runtime-benchmarking.md`) and records the residual between them. This is
a distinct artifact from both: it never re-implements estimation or benchmarking, it only pairs
their outputs.

Create it after capturing the estimator request and response bodies, without
their HTTP headers:

```bash
node scripts/write-estimator-calibration-receipt.mjs \
  --runtime-receipt tests/fixtures/calibration/rapid-mlx-receipts/<suite>/<cell>.json \
  --cell <cell-id> --attempt cold \
  --estimator-request /private/tmp/estimator-request.json \
  --estimator-response /private/tmp/estimator-response.json \
  --dataset-role tuning \
  --out tests/fixtures/calibration/rapid-mlx-receipts/<suite>/<cell>.calibration.json
```

```json
{
  "schema_version": 1,
  "kind": "estimator_calibration_receipt",
  "captured_at": "2026-07-24T12:00:00.000Z",
  "estimator": {
    "endpoint": "/api/vram-estimate",
    "request": { "model_path": "...", "backend": "rapid_mlx", "n_ctx": 8000, "kv_cache_dtype": "int8" },
    "response": { "total_bytes": 0, "evidence": "approximate", "recommendation": "fit" }
  },
  "runtime_receipt": {
    "path": "tests/fixtures/calibration/rapid-mlx-receipts/.../01-01-context-8000-int8.json",
    "cell_id": "context-8000-int8",
    "attempt_phase": "cold",
    "fresh_server_per_cell": true,
    "sha256": "receipt file digest, for tamper-evidence"
  },
  "model": {
    "hf_repo_id": "nightmedia/Qwen3.5-9B-DS9-USS-Defiant-1M-q8-hi-mlx",
    "revision": "...",
    "config_sha256": "..."
  },
  "measurement": {
    "metric_name": "rapid_mlx_metal_peak_memory_bytes",
    "actual_peak_bytes": 0
  },
  "residual": {
    "predicted_total_bytes": 0,
    "actual_peak_bytes": 0,
    "residual_bytes": 0,
    "residual_pct": 0.0,
    "predicted_gib": 0.0,
    "actual_gib": 0.0
  },
  "dataset_role": "tuning"
}
```

Field notes:

- `estimator.request` / `estimator.response` are the exact JSON body sent to and returned from
  `/api/vram-estimate`. **The `Authorization` header and API token are never persisted** —
  the calibration writer strips them before serialization; a receipt containing a token must be
  treated as a bug, not a formatting choice.
- `runtime_receipt.fresh_server_per_cell` must be `true`, and `measurement.metric_name` must be
  a peak (not lifetime-high-water-mark) metric — otherwise the residual is diagnostic context
  only, not calibration evidence (same rule as the benchmarking doc's "Required comparison
  discipline").
- `residual.residual_bytes = actual_peak_bytes − predicted_total_bytes`; positive means the
  estimator under-predicted (unsafe direction), negative means it over-predicted (safe
  direction).
- `dataset_role` is `"tuning"` for rows used to fit `mlx_overhead_base_bytes` /
  `MLX_KV_OVERHEAD_FRACTION`, and `"holdout"` for rows reserved to validate the fit. At least
  one context/dtype row must be `"holdout"` before a calibration can be called qualified — a fit
  validated only on its own training rows is not evidence of generalization.
- Only after holdout rows confirm the tuned formula does `EstimateEvidence` change from
  `Approximate` to `Measured` for Rapid-MLX, per the "Recalibrating the MLX overhead" section
  above.

---

## Adding a New Model Family

When a new architecture is released:

1. Add a named constructor to `ModelArch` (e.g. `fn new_family_heuristic(param_b: f64) -> Self`)
2. Add a match arm in `from_name_and_params()` before the generic MoE suffix fallback to use as a heuristic for missing metadata
3. Set `n_attn_layers` + `linear_attn_state_bytes` if hybrid (DeltaNet/SSM)
4. Set `n_global_attn_layers` + `local_attn_window` + `local_kv_heads` + `global_head_dim` if sliding-window
5. Set `n_embd` from the architecture spec (GGUF `embedding_length` or model card `hidden_size`). This is required for accurate discrete overhead estimates on Windows/Linux.
6. Set `expert_fraction` from published parameter breakdown; leave at 0.65 if unknown
7. Add `"arch-tag"` → `"family-heuristic-name"` mapping to `gguf_arch_to_heuristic_name()` if the GGUF uses a non-obvious arch string
8. Add a unit test with exact arithmetic for at least one known context size
9. Update this file with the new family's heuristic table row

---

## Related Files

| File | Purpose |
|------|---------|
| `src/llama/vram_estimator/estimate.rs` | Estimation logic (`full_estimate`, `max_context`, `kv_cache_bytes`, overhead functions; both backends) |
| `src/llama/vram_estimator/arch_heuristics.rs` | `ModelArch` struct + per-family heuristics + `gguf_arch_to_heuristic_name()` |
| `src/llama/vram_estimator/quant_table.rs` | BPW and KV BPE table |
| `src/llama/vram_estimator/tests.rs` | Unit tests including calibration assertions |
| `src/llama/spawn_wizard.rs` | `ModelMetadata::to_arch()` (GGUF → ModelArch); auto_size orchestration wrapper |
| `src/llama/gguf_meta.rs` | GGUF metadata reader (llama.cpp); `GgufMetadata::to_model_metadata()` |
| `src/inference/rapid_mlx/mlx_meta.rs` | MLX metadata reader (Rapid-MLX); `MlxMetadata::to_arch()`; safetensors index; evidence |
| `src/web/api/vram.rs` | `/api/vram/*` route handlers; dual-backend routing; `build_arch_from_body()` |
| `docs/reference/setup-wizard.md` | Wizard UI and API reference; links here for estimation details |
## Preset fit intents and the optional fit probe

Preset bundles can produce three deterministic, estimate-only fit intents:
Quality-first, Balanced, and Low-VRAM. Each proposal is materialized as one
complete selection and can be replayed with the same artifact, context, K/V,
performance, and MoE-placement inputs. The proposal does not start a model or
silently rewrite the saved default.

For Balanced and Low-VRAM, the resolver considers explicit bundle choices in a
fixed order: lower local artifact, lower listed context, lower listed
batch/ubatch pair, and then bounded CPU expert placement for authoritative MoE
metadata. `ubatch` is never allowed to exceed `batch`. Local artifact
down-selection requires a local path, exact size, and non-empty full-file
digest. Curated-only bundles reject combinations that are not explicitly
listed; validated-custom bundles still pass through every resolver check.

Workload policy is a quality floor, not a hidden optimization hint. Agentic
and unknown workloads never silently select `q4/q4`; mixed `q8/q4` is not an
automatic intent choice. Unknown or incomplete architecture metadata produces
fewer choices and an explicit unavailable reason. Automatic CPU expert
placement is also unavailable for Low-VRAM intents on unified-memory systems
until that behavior has qualified evidence.

The optional `llama-fit-params` executable is configured by
`AppConfig::llama_fit_params_path` and is absent by default. Its bounded
`ProcessFitReader` captures stdout and stderr separately: compact device rows
come from stdout, while the complete memory table on stderr supplies the
device total and the sum of every non-device host row. Results are estimate
class (`fit_probe`), never measured runtime evidence. Binary identity is bound
to canonical path, SHA-256, modification time, and version line, and repeated
points are cached by the artifact/config/binary identity and `n_cpu_moe`.
Missing, changed, timed-out, oversized, or unparsable probe output remains a
disabled-with-reason result; it is never converted into zero memory.
