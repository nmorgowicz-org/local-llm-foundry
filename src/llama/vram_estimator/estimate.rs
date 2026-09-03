// ── Imports from sibling modules ──────────────────────────────────────────────

use super::arch_heuristics::*;
#[allow(unused_imports)]
use super::quant_table::*;
use super::rapid_slopes::*;

// ── KV cache formula ──────────────────────────────────────────────────────────

/// Bytes per KV element for the given quantization name.
pub fn kv_elem_bytes(quant: &str) -> f64 {
    find_quant(quant).map(|q| q.kv_bpe).unwrap_or(1.0) // default q8_0
}

/// Compute total KV cache memory in bytes.
///
/// For Gemma-style hybrid attention:
/// - Global layers store the full context.
/// - Local layers use a sliding window.
pub fn kv_cache_bytes(
    arch: &ModelArch,
    context_size: u64,
    parallel_slots: u32,
    ctk: &str,
    ctv: &str,
) -> u64 {
    let slots = parallel_slots.max(1) as f64;
    let ctx = context_size as f64;
    let k_bpe = kv_elem_bytes(ctk);
    let v_bpe = kv_elem_bytes(ctv);

    // Hybrid linear-attention: only n_attn_layers of n_layers use a KV cache.
    // The remaining layers use a fixed recurrent state (counted separately in full_estimate).
    let effective_layers = if arch.is_hybrid_attn() {
        arch.n_attn_layers
    } else {
        arch.n_layers
    };

    if arch.has_local_attn() {
        let global_layers = arch.n_global_attn_layers.min(effective_layers) as f64;
        let local_layers = (effective_layers.saturating_sub(arch.n_global_attn_layers)) as f64;
        let g_kv = arch.n_kv_heads.max(1) as f64;
        let l_kv = arch.local_kv_heads.max(1) as f64;
        // Gemma 4 uses wider heads for global layers (global_head_dim=512 vs head_dim=256 local).
        let g_hd = if arch.global_head_dim > 0 {
            arch.global_head_dim
        } else {
            arch.head_dim
        };
        let g_hd = g_hd.max(1) as f64;
        let l_hd = arch.head_dim.max(1) as f64;
        let window = arch.local_attn_window as f64;

        // Global: full context × all slots
        let global_k = global_layers * g_kv * g_hd * ctx * slots * k_bpe;
        let global_v = global_layers * g_kv * g_hd * ctx * slots * v_bpe;

        // Local: sliding window (at most window tokens, regardless of ctx)
        let effective_local_ctx = ctx.min(window) * slots;
        let local_k = local_layers * l_kv * l_hd * effective_local_ctx * k_bpe;
        let local_v = local_layers * l_kv * l_hd * effective_local_ctx * v_bpe;

        (global_k + global_v + local_k + local_v) as u64
    } else {
        let n_layers = effective_layers.max(1) as f64;
        let n_kv = arch.n_kv_heads.max(1) as f64;
        let hd = arch.head_dim.max(1) as f64;

        let k = n_layers * n_kv * hd * ctx * slots * k_bpe;
        let v = n_layers * n_kv * hd * ctx * slots * v_bpe;
        (k + v) as u64
    }
}

// ── Weight distribution (MoE) ─────────────────────────────────────────────────

/// Split model weights between VRAM and RAM for a given `--n-cpu-moe` value.
/// Returns `(vram_bytes, ram_bytes)`.
pub fn moe_weight_split(model_size_bytes: u64, arch: &ModelArch, n_cpu_moe: i32) -> (u64, u64) {
    if !arch.is_moe() || n_cpu_moe <= 0 {
        return (model_size_bytes, 0);
    }
    // `--n-cpu-moe N` keeps the experts of the first N MoE layers on the CPU.
    //
    // Exact path: when we have measured per-layer expert bytes from the GGUF tensor
    // directory, each offloaded layer moves exactly `expert_bytes_per_layer` to RAM.
    // The denominator is the number of layers that actually carry experts.
    if arch.expert_bytes_per_layer > 0 {
        let moe_layers = if arch.moe_layer_count > 0 {
            arch.moe_layer_count
        } else {
            arch.n_layers.max(1)
        };
        let cpu_layers = (n_cpu_moe as u32).min(moe_layers) as u64;
        let cpu_bytes = (arch.expert_bytes_per_layer * cpu_layers).min(model_size_bytes);
        let vram_bytes = model_size_bytes.saturating_sub(cpu_bytes);
        return (vram_bytes, cpu_bytes);
    }

    // Fallback (no measured tensor sizes, e.g. pre-download advisor): estimate the
    // offloaded fraction as N / (MoE layer count) × expert_fraction. n_layers is the
    // right denominator (≈ the MoE layer count; a few models have a handful of dense
    // layers, which this slightly over-counts).
    let moe_layers = arch.n_layers.max(1) as f64;
    let cpu_layers = (n_cpu_moe as f64).min(moe_layers);
    let cpu_ratio = cpu_layers / moe_layers;
    let expert_frac = arch.expert_fraction.clamp(0.3, 0.99);

    let cpu_bytes = (model_size_bytes as f64 * expert_frac * cpu_ratio) as u64;
    let vram_bytes = model_size_bytes.saturating_sub(cpu_bytes);
    (vram_bytes, cpu_bytes)
}

/// Split dense-model weights for a discrete GPU `--gpu-layers` setting.
/// Negative values mean automatic/all-GPU placement.
pub fn dense_weight_split(model_size_bytes: u64, arch: &ModelArch, gpu_layers: i32) -> (u64, u64) {
    if arch.is_moe() || gpu_layers < 0 || arch.n_layers == 0 {
        return (model_size_bytes, 0);
    }

    let gpu_layers = (gpu_layers as u32).min(arch.n_layers);
    if gpu_layers == arch.n_layers {
        return (model_size_bytes, 0);
    }
    if gpu_layers == 0 {
        return (0, model_size_bytes);
    }

    let gpu_bytes = if arch.bytes_per_layer > 0 {
        arch.bytes_per_layer
            .saturating_mul(gpu_layers as u64)
            .min(model_size_bytes)
    } else {
        (model_size_bytes as f64 * gpu_layers as f64 / arch.n_layers as f64).round() as u64
    };
    (gpu_bytes, model_size_bytes.saturating_sub(gpu_bytes))
}

/// MTP prediction-head VRAM overhead.
/// Each depth level adds approximately 1.5% of model weights.
pub fn mtp_overhead_bytes(model_size_bytes: u64, mtp_depth: u32) -> u64 {
    if mtp_depth == 0 {
        return 0;
    }
    (model_size_bytes as f64 * 0.015 * mtp_depth as f64) as u64
}

/// Metal (unified-memory) overhead — context-INDEPENDENT part, in bytes.
///
/// Calibrated on **Apple M5 Max** (llama.cpp b9743, Metal, `--parallel 1 --kv-unified
/// -fa on`, q8_0 KV) via process physical-footprint measurements. The footprint that does
/// not scale with context is a per-layer graph/context cost plus a small ubatch scratch
/// term. Gemma sliding-window models carry a larger per-layer cost (per-layer-input
/// embeddings + the dual local/global attention graph). Inputs are GGUF-derived.
pub fn metal_overhead_base_bytes(arch: &ModelArch, ubatch_size: u32) -> u64 {
    if arch.n_layers == 0 {
        return 200 * 1024 * 1024; // unknown arch: flat reserve
    }
    let mib = 1024.0 * 1024.0;
    let per_layer = if arch.has_local_attn() { 8.8 } else { 4.3 };
    let base = per_layer * arch.n_layers as f64 + 0.035 * ubatch_size as f64;
    (base.max(128.0) * mib) as u64
}

/// Metal context-SCALING overhead: working buffers tied to the KV cache, measured at a very
/// stable **~6.5% of KV bytes** across dense / MoE / hybrid / sliding-window models on the
/// M5 Max. Expressed as a fraction of `kv_cache_bytes` so it automatically tracks hybrid
/// attention (fewer KV layers), Gemma sliding windows (local layers don't grow), and KV quant.
pub fn metal_overhead_ctx_bytes(kv_cache_bytes: u64) -> u64 {
    (kv_cache_bytes as f64 * METAL_KV_OVERHEAD_FRACTION) as u64
}

/// Fraction of KV-cache bytes that Metal spends on context-scaling working buffers.
/// Measured 0.063–0.068 across all tested families; 0.065 is the representative value.
pub const METAL_KV_OVERHEAD_FRACTION: f64 = 0.065;

/// Total Metal (unified-memory) overhead beyond weights + KV + mmproj + MTP, in bytes.
///
/// **Calibrated against direct Apple M5 Max physical-footprint measurements** (llama.cpp
/// b9743, Metal, `--parallel 1 --kv-unified -fa on`, q8_0 KV), across Qwen3.6-27B
/// (dense-hybrid), Qwen3.6-35B-A3B (MoE-hybrid), Gemma-4-31B (dense SWA) and Gemma-4-26B-A4B
/// (MoE SWA) at 4k–213k context. Fits within ~40 MiB (worst under-prediction −17 MiB). Far
/// lighter than the discrete-GPU overhead — Metal's context-scaling buffers are ~6.5% of KV
/// vs CUDA's much larger attention buffers — and, unlike the prior flat 300 MB estimate, it
/// correctly grows with context (the old flat value under-reserved Gemma-31B@213k by ~750 MiB).
pub fn metal_overhead_bytes(arch: &ModelArch, ubatch_size: u32, kv_cache_bytes: u64) -> u64 {
    metal_overhead_base_bytes(arch, ubatch_size) + metal_overhead_ctx_bytes(kv_cache_bytes)
}

/// Per-head K/V dimension that drives the context-scaling compute buffer. Gemma's global
/// (full-context) layers use a wider dimension than local layers, so take the max. All
/// fields are read from the GGUF (`attention.key_length` / `key_length_swa`).
fn overhead_head_dim(arch: &ModelArch) -> u64 {
    (arch.head_dim.max(arch.global_head_dim)).max(1) as u64
}

/// Context-INDEPENDENT part of the discrete-GPU (CUDA/ROCm) overhead, in bytes.
///
/// Covers the graph compute scratch (∝ ubatch × model width), MoE expert gather/scatter
/// buffers, and — for Gemma — the per-layer-input embedding tables that are resident beyond
/// the reported weights. All inputs are GGUF-derived (`embedding_length`, `expert_count`,
/// `block_count`, sliding-window pattern), never name-parsed.
pub fn discrete_overhead_base_bytes(arch: &ModelArch, ubatch_size: u32) -> u64 {
    if arch.n_embd == 0 || arch.n_layers == 0 {
        return 256 * 1024 * 1024; // unknown arch: flat 256 MB safety reserve
    }
    let mib = 1024.0 * 1024.0;
    // Graph compute scratch: ~0.22 MiB per ubatch unit at n_embd=5120, scaled by width.
    let scratch = 0.22 * ubatch_size as f64 * (arch.n_embd as f64 / 5120.0);
    // MoE expert gather/scatter working buffers (~260 MiB measured on 35B-A3B / Gemma-MoE).
    let moe = if arch.is_moe() { 260.0 } else { 0.0 };
    // Gemma alternating-attention models carry sizeable per-layer-input embeddings (~20 MiB/layer).
    let gemma = if arch.has_local_attn() {
        20.0 * arch.n_layers as f64
    } else {
        0.0
    };
    let total_mib = (scratch + moe + gemma).max(200.0); // 200 MiB floor (CUDA context)
    (total_mib * mib) as u64
}

/// Context-SCALING part of the discrete overhead: bytes of compute buffer per token of
/// context. The attention mask plus per-layer prefill scratch grow linearly with context
/// and with `n_layers × per-head-dim`; a mild ubatch factor captures the larger mask at
/// higher ubatch. ≈ 0.46 bytes per (token × layer × head-dim-unit).
pub fn discrete_overhead_ctx_bytes_per_token(arch: &ModelArch, ubatch_size: u32) -> f64 {
    if arch.n_layers == 0 {
        return 0.0;
    }
    let ub_factor = 0.8 + 0.2 * (ubatch_size as f64 / 1024.0);
    0.46 * arch.n_layers as f64 * overhead_head_dim(arch) as f64 * ub_factor
}

/// Total discrete-GPU (CUDA/ROCm) overhead beyond weights + KV + mmproj + MTP, in bytes.
///
/// **Calibrated against direct RTX 5090 32 GB VRAM measurements** (llama.cpp b9728,
/// `--parallel 1 --kv-unified -fa on`, q8_0 KV, full GPU offload), across Qwen3.6-27B
/// (dense-hybrid), Qwen3.6-35B-A3B (MoE-hybrid), Gemma-4-31B (dense SWA) and Gemma-4-26B-A4B
/// (MoE SWA), at 4k–213k context and ubatch 1024/2048. Predictions land within tens of MiB
/// for the Qwen family and over-reserve Gemma modestly (the safe direction). The overhead is
/// roughly **independent of model depth's KV footprint** — it grows with ubatch (scratch) and
/// context (attention mask), so the old `n_layers × n_embd × ubatch` model (context-independent)
/// was wrong in both directions. Returns the Metal estimate's analogue only for discrete GPUs;
/// unified memory uses [`metal_overhead_bytes`] in `full_estimate`.
pub fn discrete_overhead_bytes(arch: &ModelArch, ubatch_size: u32, context_size: u64) -> u64 {
    let base = discrete_overhead_base_bytes(arch, ubatch_size);
    let ctx =
        (discrete_overhead_ctx_bytes_per_token(arch, ubatch_size) * context_size as f64) as u64;
    base + ctx
}

/// Compute the headroom fraction to reserve when sizing context or evaluating fit.
///
/// Both platforms use a sliding-window approach: a percentage base rate that is capped
/// at an absolute maximum so large-memory systems don't waste capacity on oversized reserves.
///
/// - **Unified memory (Apple Silicon)**: 10% base, capped at 2 GB.
///   Covers Metal burst compute buffers. The caller has already applied the Metal GPU
///   wired cap (~66–75% of RAM) and subtracted OS/kernel reserve, so this headroom is
///   only for transient Metal allocations during inference.
///   - 24 GB effective budget: 10% = 2.4 GB → capped to 2 GB
///   - 42 GB effective budget: 10% = 4.2 GB → capped to 2 GB
///   - 10 GB effective budget: 10% = 1 GB → not capped (appropriate for small systems)
///
/// - **Discrete GPU**: 5% base, capped at 1.5 GB.
///   Display-driver + CUDA context overhead is roughly flat regardless of VRAM size.
///   Capping at 1.5 GB lets 5% apply up to a ~30 GB card; above that the cap locks in.
pub fn compute_headroom(available_vram_bytes: u64, is_unified_memory: bool) -> f64 {
    if available_vram_bytes == 0 {
        return if is_unified_memory { 0.10 } else { 0.05 };
    }
    let (base_fraction, max_bytes) = if is_unified_memory {
        (0.10f64, 2_000_000_000u64) // 2 GB cap for Metal burst buffers
    } else {
        (0.05f64, 1_500_000_000u64) // 1.5 GB cap for driver overhead
    };
    let cap_fraction = max_bytes as f64 / available_vram_bytes as f64;
    f64::min(base_fraction, cap_fraction)
}

// ── Backend-neutral estimator input ───────────────────────────────────────────

/// Which inference backend the estimate is for. Both backends share the same `ModelArch` and
/// `VramBreakdown` shapes; this only selects the overhead/headroom calibration, since llama.cpp
/// (Metal/CUDA/ROCm) and Rapid-MLX have different runtime memory behavior even when both run on
/// the same unified-memory hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    /// llama.cpp (CUDA/ROCm/Metal via `is_unified_memory`).
    #[default]
    LlamaCpp,
    /// Rapid-MLX. Apple-Silicon/unified-memory only; never uses the discrete-GPU overhead path
    /// regardless of `is_unified_memory`.
    RapidMlx,
}

/// How much of a `VramBreakdown` is backed by real measurements versus a formula-based
/// approximation or a degraded (heuristic-fallback) architecture guess.
///
/// llama.cpp's Metal/discrete overhead constants are calibrated against real hardware
/// measurements (see `metal_overhead_bytes` / `discrete_overhead_bytes` doc comments) — those
/// paths report `Measured`. Rapid-MLX has no equivalent hardware calibration yet (see
/// `mlx_overhead_bytes`), so any estimate using it must report `Approximate`, never `Measured`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EstimateEvidence {
    /// Overhead calibration is backed by real hardware measurements and architecture fields
    /// came from real metadata (GGUF tensor directory / MLX config + safetensors index).
    #[default]
    Measured,
    /// Architecture fields are real, but the overhead model itself is a documented formula-based
    /// approximation pending real hardware calibration (currently: all Rapid-MLX estimates).
    Approximate,
    /// One or more required architecture fields were missing/unrecognized and a name/param
    /// heuristic was used instead of real model metadata.
    Degraded,
}

/// Extra, backend-specific inputs to `full_estimate` that don't apply to every backend and
/// therefore default to inert values (`Backend::LlamaCpp`, `EstimateEvidence::Measured`, zero
/// prefix-cache reservation) for all existing GGUF callers.
#[derive(Debug, Clone)]
pub struct EstimatorOptions {
    pub backend: Backend,
    pub evidence: EstimateEvidence,
    /// HuggingFace repo ID (e.g. "mlx-community/Qwen3-30B-A3B-4bit").
    /// Stored for provenance/debugging; KV slope selection now uses ModelArch fields only.
    #[allow(dead_code)]
    pub hf_repo_id: Option<String>,
    /// Rapid-MLX prefix-cache compression (int4/int8) budget, in bytes, computed by the caller
    /// via `mlx_prefix_cache_bytes`. This is a SEPARATE stored-cache budget, not a reduction of
    /// active-context KV: cached entries are decompressed before being reused as active KV, so
    /// compressing them does not shrink the active KV footprint. Always 0 for GGUF.
    pub mlx_prefix_cache_bytes: u64,
    /// Rapid-MLX TurboQuant mode (from execution_policy.rs). Controls retained-KV savings.
    /// Only applies to qualified models via capability snapshot (D31).
    pub turboquant_mode: Option<super::execution_policy::TurboQuantMode>,
    /// Planning-context tokens for Rapid-MLX estimation. This is the configurable max tokens
    /// for the scenario (e.g. "128K for coding agent"), NOT the current active token count.
    /// Active KV scales with this; retained KV is the prefix-cache budget. They are distinct.
    /// Zero means use the legacy unified `context_size` parameter for backward compatibility.
    pub rapid_planning_context_tokens: u64,
    /// Retained prefix-cache tokens (separate from active generation tokens).
    /// For Rapid-MLX: tokens of prefix cached between turns, subject to TurboQuant savings.
    /// For llama.cpp: not applicable (0).
    pub rapid_retained_cache_tokens: u64,
    /// TurboQuant eligibility for the model. Per D31: unknown finetunes do NOT inherit
    /// alias qualification. Unknown/unqualified models → TurboQuant Disabled.
    pub turboquant_eligibility: super::execution_policy::TurboQuantEligibility,
    /// Builder item 13: MTP configuration for Rapid-MLX estimation.
    /// Controls embedded vs external MTP memory accounting.
    pub mtp_config: Option<super::workload_scenarios::MtpConfig>,
    /// Builder item 14: Client type for external-client vs app_fit variants.
    pub client_type: super::workload_scenarios::ClientType,
    /// D25: Concurrency policy for MTP admission.
    /// Workload the estimate is being admitted against. `None` means the caller did
    /// not state one, and admission falls back to `WorkloadScenario::default()`
    /// (`CodingAgent`) — a defensible shape, but an assumption rather than a fact, so
    /// the resulting admission is reported as scenario-defaulted rather than measured.
    pub workload_scenario: Option<super::workload_scenarios::WorkloadScenario>,
}

impl Default for EstimatorOptions {
    /// Backward-compatible defaults: llama.cpp, Measured evidence, zero Rapid-specific fields.
    fn default() -> Self {
        Self {
            backend: Backend::LlamaCpp,
            evidence: EstimateEvidence::Measured,
            hf_repo_id: None,
            mlx_prefix_cache_bytes: 0,
            turboquant_mode: None,
            rapid_planning_context_tokens: 0,
            rapid_retained_cache_tokens: 0,
            turboquant_eligibility: Default::default(),
            mtp_config: None,
            client_type: Default::default(),
            workload_scenario: None,
        }
    }
}

/// Rapid-MLX (unified-memory) overhead — context-INDEPENDENT part, in bytes.
///
/// **Not yet calibrated against real Apple Silicon Rapid-MLX measurements.** Rapid-MLX's
/// graph-compile/buffer-pooling behavior differs from llama.cpp's Metal backend (different
/// kernel fusion, KV cache layout, and MLX's lazy-evaluation graph cache), so llama.cpp's
/// Metal constants (`metal_overhead_base_bytes`) are deliberately NOT reused here. This is a
/// documented, formula-based approximation, partially calibrated against real Rapid-MLX
/// 0.10.12 hardware measurements on an Apple M5 Max (see per-branch notes below). Any estimate
/// using this function MUST be reported with `EstimateEvidence::Approximate`, never `Measured`,
/// since only two architectures have been directly measured so far.
pub fn mlx_overhead_base_bytes(arch: &ModelArch, ubatch_size: u32) -> u64 {
    if arch.n_layers == 0 {
        return 256 * 1024 * 1024; // unknown arch: flat reserve
    }
    let mib = 1024.0 * 1024.0;
    const SAFETY_MARGIN: f64 = 1.25;
    // Dense (non-local-attn): validated against mlx-community/Qwen3-0.6B-4bit (28 layers) —
    // predicted total (596MB) matched the server's self-reported steady-state Metal `active`
    // memory (0.6GB) to within ~1% at ctx=2048. The original Metal-derived 4.3 value holds.
    //
    // Local-attn (Gemma3/Gemma4-style sliding window): the previous 8.8 value was an unvalidated
    // copy of llama.cpp's Metal local-attn coefficient. Measured against
    // mlx-community/gemma-3-1b-it-4bit (26 layers, sliding_window=512): the server's
    // self-reported Metal `active` memory stayed flat at 0.8GB across the whole generation
    // (steps 256-1792), while the old coefficient predicted a 1061MB total against a 732MB
    // weight file — a ~260MB (33%) overhead over-prediction. Lowered to 5.5, which predicts a
    // ~938MB total (still conservative, ~17% over the observed 800MB) for the same run.
    // Single-model sample — recalibrate once a larger Gemma4 local-attn model has been measured.
    let per_layer = (if arch.has_local_attn() { 5.5 } else { 4.3 }) * SAFETY_MARGIN;
    let base = per_layer * arch.n_layers as f64 + 0.035 * ubatch_size as f64 * SAFETY_MARGIN;
    (base.max(160.0) * mib) as u64
}

/// Fraction of KV-cache bytes reserved for Rapid-MLX context-scaling working buffers.
///
/// Approximation only (see `mlx_overhead_base_bytes`): derived from Metal's measured 6.5%
/// (`METAL_KV_OVERHEAD_FRACTION`), inflated to a more conservative 8% given the lack of a
/// direct Rapid-MLX measurement to validate against.
pub const MLX_KV_OVERHEAD_FRACTION: f64 = 0.08;

/// Rapid-MLX context-SCALING overhead, in bytes. See `mlx_overhead_base_bytes` for the
/// approximate-evidence caveat that applies to this function too.
pub fn mlx_overhead_ctx_bytes(kv_cache_bytes: u64) -> u64 {
    (kv_cache_bytes as f64 * MLX_KV_OVERHEAD_FRACTION) as u64
}

/// Total Rapid-MLX (unified-memory-only) overhead beyond weights + KV + mmproj + MTP, in
/// bytes. Formula-based approximation — see `mlx_overhead_base_bytes` for why llama.cpp's
/// Metal constants are not reused and why this must be reported as `EstimateEvidence::Approximate`.
pub fn mlx_overhead_bytes(arch: &ModelArch, ubatch_size: u32, kv_cache_bytes: u64) -> u64 {
    mlx_overhead_base_bytes(arch, ubatch_size) + mlx_overhead_ctx_bytes(kv_cache_bytes)
}

/// Rapid-MLX compressed prefix-cache budget, in bytes.
///
/// This models the SEPARATE stored-cache budget for previously-computed prefixes that
/// Rapid-MLX keeps compressed (int4/int8) on disk/in memory. It is intentionally NOT a
/// reduction of `kv_cache_bytes`: cached entries are decompressed back to the active compute
/// dtype before being reused as active KV, so the active-request KV footprint is unaffected by
/// how much prefix cache exists. `cached_tokens` is the number of tokens' worth of prefix the
/// caller wants budgeted (0 = no reservation, the default when the caller hasn't configured a
/// cache budget). `compression_bits` is 4 or 8; any other value is treated as 8 (uncompressed
/// int8 baseline) to fail safe (never under-reserve).
#[cfg(test)]
pub fn mlx_prefix_cache_bytes(arch: &ModelArch, cached_tokens: u64, compression_bits: u8) -> u64 {
    if cached_tokens == 0 || arch.n_layers == 0 {
        return 0;
    }
    let bytes_per_elem = match compression_bits {
        4 => 0.5,
        8 => 1.0,
        _ => 1.0,
    };
    let effective_layers = if arch.is_hybrid_attn() {
        arch.n_attn_layers
    } else {
        arch.n_layers
    } as f64;
    let kv_heads = arch.n_kv_heads.max(1) as f64;
    let head_dim = arch.head_dim.max(1) as f64;
    // K + V, one "slot" (prefix cache is shared, not per-parallel-slot).
    (effective_layers * kv_heads * head_dim * cached_tokens as f64 * 2.0 * bytes_per_elem) as u64
}

// ── Estimate model file size from param count ─────────────────────────────────

/// Default bits-per-weight for unknown quantizations.
const DEFAULT_BPW: f64 = 4.85;

/// Estimate model file size in bytes from parameter count and quantization.
pub fn estimate_model_size_bytes(param_b: f64, quant: &str) -> u64 {
    let bpw = find_quant(quant).map(|q| q.bpw).unwrap_or(DEFAULT_BPW);
    (param_b * 1e9 * bpw / 8.0) as u64
}

/// Inverse of estimate_model_size_bytes: rough param_b from file size.
///
/// Used when GGUF introspection fails and we must guess param_b from the file size.
/// Correctly converts bytes→bits with the *8 factor: params = (bytes × 8) / 1e9 / bpw.
pub fn estimate_param_b_from_size(size_bytes: u64, bpw: f64) -> f64 {
    if bpw <= 0.0 {
        return 0.0;
    }
    (size_bytes as f64 * 8.0) / 1e9 / bpw
}

fn kv_dtype_from_estimator_quant(quant: &str) -> super::execution_policy::KvCacheDtype {
    match quant {
        "f16" => super::execution_policy::KvCacheDtype::Bf16,
        "q8_0" => super::execution_policy::KvCacheDtype::Int8,
        "q4_0" => super::execution_policy::KvCacheDtype::Int4,
        _ => super::execution_policy::KvCacheDtype::Int4,
    }
}

// ── Full VRAM estimate with breakdown ─────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize)]
pub struct VramBreakdown {
    pub weights_bytes: u64,
    /// Total KV cache bytes. For llama.cpp and legacy Rapid calls: unified value.
    /// For Rapid-MLX with planning context: sum of active_kv_bytes + retained_kv_bytes.
    pub kv_cache_bytes: u64,
    /// Active KV cache for tokens in the current generation sequence (planning context).
    /// For Rapid-MLX: scales with planning_context_tokens, NOT TurboQuant-affected.
    /// For llama.cpp/legacy: 0 (uses unified kv_cache_bytes).
    #[serde(default)]
    pub active_kv_bytes: u64,
    /// Retained prefix KV cache (reusable prompt storage between turns).
    /// For Rapid-MLX: may be reduced by TurboQuant compression (D31).
    /// For llama.cpp/legacy: 0 (uses unified kv_cache_bytes).
    #[serde(default)]
    pub retained_kv_bytes: u64,
    /// Fixed recurrent state for hybrid linear-attention layers (DeltaNet / SSM).
    /// Zero for standard transformers. Does not grow with context length.
    pub linear_attn_state_bytes: u64,
    pub mmproj_bytes: u64,
    pub mtp_bytes: u64,
    /// TurboQuant transient decompression peak bytes. During decompress→decode cycle,
    /// both compressed retained buffer and decompressed working set coexist.
    /// Only nonzero when TurboQuant is active on qualified retained KV (D31).
    /// Included in total_bytes as a real (short-lived) memory cost.
    #[serde(default)]
    pub turboquant_transient_peak_bytes: u64,
    pub overhead_bytes: u64,
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub headroom_bytes: i64, // can be negative (over budget)
    pub ram_bytes: u64,      // weights placed in CPU RAM
    pub available_ram_bytes: u64,
    pub ram_headroom_bytes: i64,
    pub recommendation: VramRecommendation,
    pub note: String,
    /// Rapid-MLX compressed prefix-cache budget (bytes). Always 0 for GGUF/llama.cpp. This is
    /// a separate stored-cache budget, NOT a reduction of `kv_cache_bytes` — see
    /// `mlx_prefix_cache_bytes` for why cached (compressed) entries don't shrink active KV.
    #[serde(default)]
    pub mlx_prefix_cache_bytes: u64,
    /// How much of this breakdown is backed by real hardware measurements vs. a formula-based
    /// approximation or a degraded (heuristic-fallback) architecture guess.
    #[serde(default)]
    pub evidence: EstimateEvidence,
    /// Effective TurboQuant mode after eligibility resolution (D31). Shows what will actually
    /// be used; may differ from requested if model is unqualified.
    #[serde(default)]
    pub effective_turboquant: super::execution_policy::TurboQuantMode,
    /// Builder item 13: MTP mode used for this estimate.
    #[serde(default)]
    pub mtp_mode: super::workload_scenarios::MtpMode,
    /// Builder item 13: External drafter companion estimate (if MTP mode = External).
    /// Total = primary breakdown total_bytes + external_companion.total_bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_companion: Option<super::workload_scenarios::ExternalCompanion>,
    /// Builder item 13: D25 MTP admission result (warnings, concurrency policy).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mtp_admission: Option<super::workload_scenarios::MtpAdmissionResult>,
    /// Builder item 14: Client type (app vs external client).
    /// Used to distinguish external_client_fit vs app_fit variants.
    #[serde(default)]
    pub client_type: super::workload_scenarios::ClientType,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VramRecommendation {
    Fit,
    Tight,
    Risk,
    WontFit,
}

/// Full VRAM estimate for a configured setup.
///
/// `is_unified_memory`: true for Apple Silicon and other unified-memory architectures where
/// GPU and system RAM share the same pool. On unified memory there is no CPU spill path —
/// exceeding available memory causes OS compression/paging, not a graceful fallback.
///
/// For Rapid-MLX backend:
/// - Active KV is computed from `opts.rapid_planning_context_tokens` (scenario target, not current).
/// - Retained KV is computed from `opts.rapid_retained_cache_tokens` (prefix-cache budget).
/// - They are distinct; no component is double-counted.
/// - TurboQuant (D31) applies ONLY to retained_kv_bytes; never to active KV, weights, MTP, recurrent, or prefill.
/// - TurboQuant transient decompression peak is an explicit additive component.
/// - If `rapid_planning_context_tokens` is 0, falls back to unified `context_size` for backward compatibility.
#[allow(clippy::too_many_arguments)]
pub fn full_estimate(
    model_size_bytes: u64,
    arch: &ModelArch,
    context_size: u64,
    ctk: &str,
    ctv: &str,
    parallel_slots: u32,
    ubatch_size: u32,
    n_cpu_moe: i32,
    gpu_layers: i32,
    available_vram_bytes: u64,
    available_ram_bytes: u64,
    is_unified_memory: bool,
    opts: EstimatorOptions,
) -> VramBreakdown {
    let (weight_vram, ram) = if is_unified_memory {
        (model_size_bytes, 0)
    } else if arch.is_moe() {
        moe_weight_split(model_size_bytes, arch, n_cpu_moe)
    } else {
        dense_weight_split(model_size_bytes, arch, gpu_layers)
    };

    // ── Active vs retained KV separation (Phase 5a Part 2) ──────────────────────
    //
    // For Rapid-MLX with planning-context info: active KV and retained KV are distinct.
    // - active_kv: KV for tokens in the current generation sequence (planning context).
    // - retained_kv: KV for prefix cache between turns, subject to TurboQuant savings.
    // - They are NEVER double-counted.
    //
    // For llama.cpp or legacy Rapid calls (no planning_context_tokens): unified KV formula.

    let is_rapid_with_context =
        matches!(opts.backend, Backend::RapidMlx) && opts.rapid_planning_context_tokens > 0;

    let rapid_kv_dtype = kv_dtype_from_estimator_quant(ctk);

    // Receipt-based slope uses only ModelArch fields — no repo ID needed.
    let receipt_slope = matches!(opts.backend, Backend::RapidMlx)
        .then(|| rapid_active_kv_bytes_per_token(arch, rapid_kv_dtype));

    let (active_kv, retained_kv_uncompressed) = if is_rapid_with_context {
        let slots = parallel_slots.max(1) as f64;
        let planning_tokens = opts.rapid_planning_context_tokens as f64;

        // Active KV: use receipt-based slope when available
        let active = if let Some(slope) = receipt_slope {
            (slope * planning_tokens * slots) as u64
        } else {
            kv_cache_bytes(
                arch,
                opts.rapid_planning_context_tokens,
                parallel_slots.max(1),
                ctk,
                ctv,
            )
        };

        // An explicit --cache-memory-mb ceiling is the admission reservation.
        // Token-derived retained KV is used only when no explicit byte ceiling
        // exists; summing both would reserve the same retained state twice.
        let retained = if opts.mlx_prefix_cache_bytes > 0 {
            0
        } else if let Some(slope) = receipt_slope {
            let retained_tokens = opts.rapid_retained_cache_tokens as f64;
            (slope * retained_tokens) as u64
        } else {
            kv_cache_bytes(arch, opts.rapid_retained_cache_tokens, 1, ctk, ctv)
        };
        (active, retained)
    } else {
        let kv = kv_cache_bytes(arch, context_size, parallel_slots, ctk, ctv);
        (kv, 0)
    };

    // ── TurboQuant savings on retained KV (D31) ─────────────────────────────────
    //
    // TurboQuant applies ONLY to qualified retained KV components. It is NOT a global multiplier.
    // - active_kv_bytes: NO TurboQuant savings (active generation KV at full compute dtype).
    // - weights_bytes: NO TurboQuant savings.
    // - mtp_bytes: NO TurboQuant savings.
    // - recurrent_state_bytes: NO TurboQuant savings.
    // - prefill/transient: NO TurboQuant savings.
    // - retained_kv_bytes: YES, TurboQuant savings apply here (prefix cache compression).
    //
    // Per D31: unknown finetunes do NOT inherit alias qualification. If eligibility is
    // NotQualified, effective mode is Disabled → 0% savings.

    let effective_turboquant = opts
        .turboquant_mode
        .unwrap_or(super::execution_policy::TurboQuantMode::Disabled)
        .ensure_qualified(opts.turboquant_eligibility);

    let turboquant_savings_fraction = effective_turboquant.retained_kv_savings_fraction();
    let retained_kv_compressed =
        if turboquant_savings_fraction > 0.0 && retained_kv_uncompressed > 0 {
            let savings = (retained_kv_uncompressed as f64 * turboquant_savings_fraction) as u64;
            retained_kv_uncompressed.saturating_sub(savings)
        } else {
            retained_kv_uncompressed
        };

    // TurboQuant transient decompression peak: during decompress→decode cycle, both compressed
    // retained buffer and decompressed working set coexist. The peak is the decompressed size
    // (which equals uncompressed retained KV). This is a real, short-lived memory cost that
    // MUST be visible in the breakdown per D31.
    let turboquant_transient_peak =
        if effective_turboquant.is_enabled() && retained_kv_uncompressed > 0 {
            retained_kv_uncompressed
        } else {
            0
        };

    // Combined KV for overhead calculations (Metal/MLX overhead scales with total KV footprint).
    let kv_for_overhead = active_kv + retained_kv_compressed;

    // For hybrid linear-attention models (e.g. Qwen3-Coder-Next / DeltaNet):
    // add the fixed recurrent state. This is constant — it does NOT grow with context.
    let linear_state = arch.linear_attn_state_bytes;
    let mmproj = arch.mmproj_bytes;
    // Builder item 13: MTP memory uses explicit config when provided.
    // Embedded MTP: counted in primary breakdown via primary_breakdown_mtp_bytes.
    // External MTP: companion counted separately, primary_breakdown_mtp_bytes returns 0.
    let mtp_config = opts.mtp_config.clone();
    let mtp = match &mtp_config {
        Some(cfg) => cfg.primary_breakdown_mtp_bytes(model_size_bytes),
        None => mtp_overhead_bytes(model_size_bytes, arch.mtp_depth),
    };
    let external_companion_bytes = mtp_config
        .as_ref()
        .filter(|cfg| cfg.mode == super::workload_scenarios::MtpMode::External)
        .and_then(|cfg| cfg.external_drafter.as_ref())
        .map(|companion| companion.total_bytes)
        .unwrap_or(0);
    // Platform-specific overhead, both calibrated against real VRAM/footprint measurements:
    // discrete GPUs → RTX-5090 model (scratch + MoE/Gemma base + context-scaling attention
    // buffers); unified memory → Apple M5 Max model (per-layer base + ~6.5% of KV).
    let overhead = match opts.backend {
        Backend::RapidMlx => mlx_overhead_bytes(arch, ubatch_size, kv_for_overhead),
        Backend::LlamaCpp if is_unified_memory => {
            metal_overhead_bytes(arch, ubatch_size, kv_for_overhead)
        }
        Backend::LlamaCpp => discrete_overhead_bytes(arch, ubatch_size, context_size),
    };
    let mlx_cache = opts.mlx_prefix_cache_bytes;

    // TOTAL is additive: all components summed, none double-counted.
    // TurboQuant transient peak is INCLUDED (it's a real memory cost during decompression).
    let total = weight_vram
        + active_kv
        + retained_kv_compressed
        + linear_state
        + mmproj
        + mtp
        + external_companion_bytes
        + turboquant_transient_peak
        + overhead
        + mlx_cache;
    let headroom = available_vram_bytes as i64 - total as i64;
    let ram_headroom = available_ram_bytes as i64 - ram as i64;

    let ram_exceeded = !is_unified_memory && ram > 0 && available_ram_bytes > 0 && ram_headroom < 0;
    let (recommendation, note) = if ram_exceeded {
        (
            VramRecommendation::WontFit,
            "CPU-offloaded weights exceed available system RAM.".into(),
        )
    } else if available_vram_bytes == 0 {
        (
            VramRecommendation::Risk,
            "Memory size unknown; estimate is best-effort.".into(),
        )
    } else if total <= (available_vram_bytes * 82 / 100) {
        (
            VramRecommendation::Fit,
            if is_unified_memory {
                "Fits with good headroom within the available unified memory budget."
            } else {
                "Fits comfortably with >18% headroom."
            }
            .into(),
        )
    } else if total <= available_vram_bytes {
        (
            VramRecommendation::Tight,
            if is_unified_memory {
                "Near the memory budget — macOS may compress memory under load. Reduce context or KV quant if you notice slowdowns."
            } else {
                "Fits, but VRAM is nearly full. Reduce context or KV quant if you hit OOM."
            }
            .into(),
        )
    } else if !is_unified_memory && total <= (available_vram_bytes * 120 / 100) {
        (
            VramRecommendation::Risk,
            "Exceeds VRAM; expect CPU spill and slower generation. Lower context or use KV quantization.".into(),
        )
    } else {
        (
            VramRecommendation::WontFit,
            if is_unified_memory {
                "Exceeds unified memory budget. On Apple Silicon there is no CPU spill path — this causes memory pressure and paging. Lower context, KV quant, or model quantization."
            } else {
                "Significantly over VRAM budget. Lower the model quant, context, or offload more MoE experts to CPU."
            }
            .into(),
        )
    };

    // Backward-compatible total KV for existing consumers.
    // For Rapid-MLX with context split: active + retained. For legacy: unified value.
    let total_kv = active_kv + retained_kv_compressed;

    // Builder item 13: D25 MTP admission.
    // Only computed when MTP config is provided.
    let (mtp_mode, external_companion, mtp_admission) = if let Some(ref cfg) = mtp_config {
        let mode = cfg.mode;
        let companion = cfg.external_drafter.clone();

        // Compute admission result when MTP is enabled.
        let admission = if mode != super::workload_scenarios::MtpMode::Disabled {
            // A caller holding the real request parameters should use
            // `compute_with_shape` instead; the decode shape otherwise follows the
            // scenario via `compute`.
            let scenario = opts.workload_scenario.unwrap_or_default();
            Some(super::workload_scenarios::MtpAdmissionResult::compute(
                mode,
                &scenario,
                arch.mtp_depth,
                parallel_slots.max(1),
            ))
        } else {
            None
        };

        (mode, companion, admission)
    } else {
        (super::workload_scenarios::MtpMode::Disabled, None, None)
    };

    VramBreakdown {
        weights_bytes: weight_vram,
        kv_cache_bytes: total_kv,
        active_kv_bytes: if is_rapid_with_context { active_kv } else { 0 },
        retained_kv_bytes: if is_rapid_with_context {
            retained_kv_compressed
        } else {
            0
        },
        linear_attn_state_bytes: linear_state,
        mmproj_bytes: mmproj,
        mtp_bytes: mtp,
        turboquant_transient_peak_bytes: turboquant_transient_peak,
        overhead_bytes: overhead,
        total_bytes: total,
        available_bytes: available_vram_bytes,
        headroom_bytes: headroom,
        ram_bytes: ram,
        available_ram_bytes,
        ram_headroom_bytes: ram_headroom,
        recommendation,
        note,
        mlx_prefix_cache_bytes: mlx_cache,
        evidence: opts.evidence,
        effective_turboquant,
        mtp_mode,
        external_companion,
        mtp_admission,
        client_type: opts.client_type,
    }
}

// ── Maximum context for a given config ────────────────────────────────────────

/// Find the largest context that fits in available VRAM.
///
/// `fit_granularity` rounds the result down to a multiple (e.g. 1024 for --fit-ctx 1024).
/// `headroom_fraction` reserves a fraction of VRAM as a safety buffer (default 0.05 = 5%).
/// `n_ctx_train` is an optional hard cap: even if more context fits in VRAM, we should not
/// silently exceed the model's training context length (unless the user extends it via
/// RoPE/YaRN scaling or a manual override).
#[allow(clippy::too_many_arguments)]
pub fn max_context(
    model_size_bytes: u64,
    arch: &ModelArch,
    ctk: &str,
    ctv: &str,
    parallel_slots: u32,
    ubatch_size: u32,
    n_cpu_moe: i32,
    available_vram_bytes: u64,
    fit_granularity: u64,
    headroom_fraction: f64,
    n_ctx_train: Option<u64>,
    is_unified_memory: bool,
    backend: Backend,
) -> u64 {
    if available_vram_bytes == 0 {
        return 0;
    }
    let (weight_vram, _) = moe_weight_split(model_size_bytes, arch, n_cpu_moe);
    let mmproj = arch.mmproj_bytes;
    let mtp = mtp_overhead_bytes(model_size_bytes, arch.mtp_depth);
    let linear_state = arch.linear_attn_state_bytes; // constant; doesn't scale with context
    // Context-INDEPENDENT overhead goes into the fixed budget. The context-SCALING part is
    // charged against the KV budget: discrete GPUs add a per-token slope; Metal/MLX scale as a
    // fraction of the KV cache, so we reserve it by shrinking the KV budget by that factor.
    let (base_overhead, overhead_slope, kv_overhead_mult) = match backend {
        Backend::RapidMlx => (
            mlx_overhead_base_bytes(arch, ubatch_size),
            0.0,
            1.0 + MLX_KV_OVERHEAD_FRACTION,
        ),
        Backend::LlamaCpp if is_unified_memory => (
            metal_overhead_base_bytes(arch, ubatch_size),
            0.0,
            1.0 + METAL_KV_OVERHEAD_FRACTION,
        ),
        Backend::LlamaCpp => (
            discrete_overhead_base_bytes(arch, ubatch_size),
            discrete_overhead_ctx_bytes_per_token(arch, ubatch_size),
            1.0,
        ),
    };
    let fixed = weight_vram + mmproj + mtp + linear_state + base_overhead;

    let usable = (available_vram_bytes as f64 * (1.0 - headroom_fraction)) as u64;
    if fixed >= usable {
        return 0;
    }
    // Reserve Metal's KV-proportional overhead up front (no-op on discrete, mult = 1.0).
    let kv_budget = ((usable - fixed) as f64 / kv_overhead_mult) as u64;

    // Binary search for context such that kv_cache_bytes(ctx) + overhead(ctx) ≤ kv_budget.
    // For non-sliding-window models we can solve directly; for Gemma we binary-search.
    let mut ctx = if arch.has_local_attn() {
        binary_search_context(arch, ctk, ctv, parallel_slots, kv_budget, overhead_slope)
    } else {
        direct_max_context(arch, ctk, ctv, parallel_slots, kv_budget, overhead_slope)
    };

    // Cap at training context (unless user has extended via RoPE/YaRN).
    if let Some(cap) = n_ctx_train
        && ctx > cap
    {
        ctx = cap;
    }

    // Round down to fit_granularity
    let g = fit_granularity.max(1);
    (ctx / g) * g
}

fn direct_max_context(
    arch: &ModelArch,
    ctk: &str,
    ctv: &str,
    parallel_slots: u32,
    kv_budget: u64,
    overhead_slope: f64,
) -> u64 {
    let slots = parallel_slots.max(1) as f64;
    // Hybrid DeltaNet models (Qwen3.6, Qwen3.5): only n_attn_layers grow the KV cache.
    // Using n_layers here would underestimate max context by 4× for those models.
    let n_layers = if arch.is_hybrid_attn() {
        arch.n_attn_layers.max(1)
    } else {
        arch.n_layers.max(1)
    } as f64;
    let n_kv = arch.n_kv_heads.max(1) as f64;
    let hd = arch.head_dim.max(1) as f64;
    let k_bpe = kv_elem_bytes(ctk);
    let v_bpe = kv_elem_bytes(ctv);
    let mut bytes_per_token = n_layers * n_kv * hd * slots * (k_bpe + v_bpe);
    // Add the discrete-GPU per-token overhead (attention mask + context-scaling buffers).
    bytes_per_token += overhead_slope;
    if bytes_per_token <= 0.0 {
        return 0;
    }
    (kv_budget as f64 / bytes_per_token) as u64
}

fn binary_search_context(
    arch: &ModelArch,
    ctk: &str,
    ctv: &str,
    parallel_slots: u32,
    kv_budget: u64,
    overhead_slope: f64,
) -> u64 {
    let mut lo = 512u64;
    // If even the minimum context doesn't fit, report zero rather than returning 512 and OOMing.
    let min_cost =
        kv_cache_bytes(arch, lo, parallel_slots, ctk, ctv) + (overhead_slope * lo as f64) as u64;
    if min_cost > kv_budget {
        return 0;
    }
    let mut hi = 2_097_152u64; // 2M upper bound
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        let cost = kv_cache_bytes(arch, mid, parallel_slots, ctk, ctv)
            + (overhead_slope * mid as f64) as u64;
        if cost <= kv_budget {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

// ── Use-case type ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UseCase {
    /// Tool-calling agents, RAG pipelines: needs coherent long-context output.
    /// Minimum q8_0 KV recommended.
    Agentic,
    /// Everyday chat, summarization, coding assistance.
    #[default]
    General,
    /// Creative writing, roleplay: context beats coherence precision.
    /// q4_0 KV acceptable.
    Roleplay,
}

impl UseCase {
    /// Minimum recommended KV cache key quantization for this use case.
    pub fn min_kv_quant(self) -> &'static str {
        match self {
            UseCase::Agentic => "q8_0",
            UseCase::General => "q8_0",
            UseCase::Roleplay => "q4_0",
        }
    }

    /// Whether to warn about KV quant degradation.
    pub fn should_warn_kv(self, kv_quant: &str) -> bool {
        let kv_bpe = kv_elem_bytes(kv_quant);
        let min_bpe = kv_elem_bytes(self.min_kv_quant());
        kv_bpe < min_bpe
    }
}

// ── Auto-size recommendation ──────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize)]
pub struct AutoSizeResult {
    /// Recommended context size (rounded to fit_granularity).
    pub context_size: u64,
    /// Recommended KV key quantization.
    pub kv_quant_k: String,
    /// Recommended KV value quantization.
    pub kv_quant_v: String,
    /// Recommended --fit-ctx value.
    pub fit_ctx: u32,
    /// Recommended ubatch_size.
    pub ubatch_size: u32,
    /// Recommended n_cpu_moe (None for dense models or when all fits in VRAM).
    pub n_cpu_moe: Option<i32>,
    /// Full VRAM breakdown for the recommended config.
    pub breakdown: VramBreakdown,
    /// Alternative scenarios for the wizard scenario cards.
    pub scenarios: Vec<ContextScenario>,
    pub warnings: Vec<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ContextScenario {
    pub label: String,
    pub kv_quant_k: String,
    pub kv_quant_v: String,
    pub context_size: u64,
    pub n_cpu_moe: Option<i32>,
    pub vram_total_gb: f64,
    pub recommended: bool,
    pub warning: Option<String>,
    pub note: String,
}

/// Compute optimal settings for a given model, hardware, and use case.
///
/// `is_unified_memory`: true for Apple Silicon — tightens headroom reservation and
/// removes the CPU-spill budget zone (paging on unified memory is not a graceful fallback).
/// `n_ctx_train`: optional training context length to cap the recommended context.
#[allow(clippy::too_many_arguments)]
pub fn auto_size(
    model_size_bytes: u64,
    arch: &ModelArch,
    available_vram_bytes: u64,
    use_case: UseCase,
    requested_parallel_slots: u32,
    preferred_fit_granularity: u64,
    is_unified_memory: bool,
    n_ctx_train: Option<u64>,
    backend: Backend,
) -> AutoSizeResult {
    let fit_gran = preferred_fit_granularity.max(512);
    let parallel_slots = requested_parallel_slots.max(1);

    // Ubatch: 1024 for agentic/general (faster prefill), 512 for roleplay (save VRAM)
    let ubatch = match use_case {
        UseCase::Agentic | UseCase::General => 1024,
        UseCase::Roleplay => 512,
    };

    // ── Step 1: Find optimal n_cpu_moe for MoE models ────────────────────────
    // Start with all experts in VRAM. If the weights alone don't fit, increment
    // n_cpu_moe until they do (or until fully offloaded).
    let n_cpu_moe = if arch.is_moe() {
        find_min_cpu_moe_to_fit_weights(
            model_size_bytes,
            arch,
            available_vram_bytes,
            ubatch,
            is_unified_memory,
            backend,
        )
    } else {
        0
    };
    let n_cpu_moe_opt = if n_cpu_moe > 0 { Some(n_cpu_moe) } else { None };

    // ── Step 2: Determine KV quant based on use case ─────────────────────────
    let (kv_k, kv_v) = best_kv_quant_for_use_case(use_case);

    let headroom = compute_headroom(available_vram_bytes, is_unified_memory);

    // ── Step 3: Compute max context for recommended KV quant ─────────────────
    let ctx = max_context(
        model_size_bytes,
        arch,
        &kv_k,
        &kv_v,
        parallel_slots,
        ubatch,
        n_cpu_moe,
        available_vram_bytes,
        fit_gran,
        headroom,
        n_ctx_train,
        is_unified_memory,
        backend,
    );

    let breakdown = full_estimate(
        model_size_bytes,
        arch,
        ctx,
        &kv_k,
        &kv_v,
        parallel_slots,
        ubatch,
        n_cpu_moe,
        -1,
        available_vram_bytes,
        0,
        is_unified_memory,
        EstimatorOptions {
            backend,
            evidence: if backend == Backend::RapidMlx {
                EstimateEvidence::Approximate
            } else {
                EstimateEvidence::Measured
            },
            ..Default::default()
        },
    );

    // ── Step 4: Warnings ──────────────────────────────────────────────────────
    let mut warnings = Vec::new();
    let mut notes = Vec::new();

    if ctx == 0 {
        warnings.push("Model weights alone exceed available VRAM. Try a smaller quantization or offload more experts to CPU.".into());
    }

    if use_case.should_warn_kv(&kv_k) {
        warnings.push(format!(
            "{} KV cache quantization can reduce coherence for {} workflows.",
            kv_k.to_uppercase(),
            format!("{use_case:?}").to_lowercase()
        ));
    }
    if use_case == UseCase::Agentic && kv_elem_bytes(&kv_k) < 1.0 {
        warnings.push("⚠ Agentic workflows should use q8_0 KV minimum. Tool-calling accuracy degrades below this.".into());
    }

    if arch.is_moe() && n_cpu_moe > 0 {
        let cpu_layers = (n_cpu_moe as u32).min(arch.n_layers);
        let gpu_layers = arch.n_layers.saturating_sub(cpu_layers);
        let cpu_fraction = cpu_layers as f64 / arch.n_layers.max(1) as f64;
        notes.push(format!(
            "MoE: expert tensors for {gpu_layers} of {} layers in VRAM, {cpu_layers} layers on CPU (~{:.0}% generation speed reduction).",
            arch.n_layers,
            cpu_fraction * 60.0
        ));
    }
    if arch.mtp_depth > 0 {
        notes.push(format!(
            "MTP depth {} adds {:.1} GB overhead for speculative prediction heads.",
            arch.mtp_depth,
            mtp_overhead_bytes(model_size_bytes, arch.mtp_depth) as f64 / 1e9
        ));
    }
    if arch.mmproj_bytes > 0 {
        notes.push(format!(
            "Vision projector ({:.1} GB) stays resident in VRAM.",
            arch.mmproj_bytes as f64 / 1e9
        ));
    }

    // ── Step 5: Alternative scenarios for the scenario cards ─────────────────
    let scenarios = build_scenarios(
        model_size_bytes,
        arch,
        available_vram_bytes,
        parallel_slots,
        ubatch,
        n_cpu_moe,
        fit_gran,
        use_case,
        &kv_k,
        is_unified_memory,
        n_ctx_train,
        backend,
    );

    AutoSizeResult {
        context_size: ctx,
        kv_quant_k: kv_k,
        kv_quant_v: kv_v,
        fit_ctx: fit_gran as u32,
        ubatch_size: ubatch,
        n_cpu_moe: n_cpu_moe_opt,
        breakdown,
        scenarios,
        warnings,
        notes,
    }
}

fn best_kv_quant_for_use_case(use_case: UseCase) -> (String, String) {
    match use_case {
        UseCase::Agentic | UseCase::General => ("q8_0".into(), "q8_0".into()),
        UseCase::Roleplay => ("q4_0".into(), "q4_0".into()),
    }
}

/// Find the smallest `--n-cpu-moe` value whose weight footprint fits in VRAM.
/// Reused by the auto-size flow and the Spawn Wizard / Preset Editor auto-tuner
/// so the instant estimate always agrees with the animated VRAM bar.
/// `is_unified_memory` selects the correct overhead model: Metal for Apple Silicon,
/// discrete (RTX 5090-calibrated) for CUDA/ROCm — the gap is large for MoE models.
pub fn find_min_cpu_moe_to_fit_weights(
    model_size_bytes: u64,
    arch: &ModelArch,
    available_vram_bytes: u64,
    ubatch_size: u32,
    is_unified_memory: bool,
    backend: Backend,
) -> i32 {
    let overhead = match backend {
        Backend::RapidMlx => mlx_overhead_base_bytes(arch, ubatch_size),
        Backend::LlamaCpp if is_unified_memory => metal_overhead_base_bytes(arch, ubatch_size),
        Backend::LlamaCpp => discrete_overhead_base_bytes(arch, ubatch_size),
    };
    let target = (available_vram_bytes * 80 / 100).saturating_sub(
        overhead + arch.mmproj_bytes + mtp_overhead_bytes(model_size_bytes, arch.mtp_depth),
    );

    // `--n-cpu-moe` counts layers, so the maximum is the layer count.
    let max_cpu = arch.n_layers as i32;

    // Binary search: find the smallest n_cpu_moe where vram fits.
    let mut lo: i32 = 0;
    let mut hi: i32 = max_cpu;

    // If even with all experts on CPU it doesn't fit, return max.
    let (vram_all_cpu, _) = moe_weight_split(model_size_bytes, arch, hi);
    if vram_all_cpu > target {
        return hi;
    }

    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let (vram, _) = moe_weight_split(model_size_bytes, arch, mid);
        if vram <= target {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }

    lo
}

#[allow(clippy::too_many_arguments)]
fn build_scenarios(
    model_size_bytes: u64,
    arch: &ModelArch,
    available_vram_bytes: u64,
    parallel_slots: u32,
    ubatch: u32,
    n_cpu_moe: i32,
    fit_gran: u64,
    _use_case: UseCase,
    recommended_kv: &str,
    is_unified_memory: bool,
    n_ctx_train: Option<u64>,
    backend: Backend,
) -> Vec<ContextScenario> {
    let mut scenarios = Vec::new();
    let headroom = compute_headroom(available_vram_bytes, is_unified_memory);

    let kv_options: &[(&str, &str, &str)] = &[
        ("q8_0", "q8_0", "Max coherence (q8_0 KV)"),
        ("q4_0", "q4_0", "Max context (q4_0 KV)"),
        ("f16", "f16", "Reference quality (f16 KV)"),
    ];

    for (kk, kv, label) in kv_options {
        let ctx = max_context(
            model_size_bytes,
            arch,
            kk,
            kv,
            parallel_slots,
            ubatch,
            n_cpu_moe,
            available_vram_bytes,
            fit_gran,
            headroom,
            n_ctx_train,
            is_unified_memory,
            backend,
        );
        let bd = full_estimate(
            model_size_bytes,
            arch,
            ctx,
            kk,
            kv,
            parallel_slots,
            ubatch,
            n_cpu_moe,
            -1,
            available_vram_bytes,
            0,
            is_unified_memory,
            EstimatorOptions {
                backend,
                evidence: if backend == Backend::RapidMlx {
                    EstimateEvidence::Approximate
                } else {
                    EstimateEvidence::Measured
                },
                ..Default::default()
            },
        );
        let warn = if _use_case == UseCase::Agentic && kv_elem_bytes(kk) < 1.0 {
            Some("⚠ Below q8_0 — not recommended for agents".into())
        } else {
            None
        };
        let rec = *kk == recommended_kv;
        scenarios.push(ContextScenario {
            label: label.to_string(),
            kv_quant_k: kk.to_string(),
            kv_quant_v: kv.to_string(),
            context_size: ctx,
            n_cpu_moe: if n_cpu_moe > 0 { Some(n_cpu_moe) } else { None },
            vram_total_gb: bd.total_bytes as f64 / 1e9,
            recommended: rec,
            warning: warn,
            note: format!("{} tokens", format_ctx(ctx)),
        });
    }

    // For MoE: add a "tight-fit" scenario with more CPU offload for extended context
    if arch.is_moe() && arch.n_layers > 1 {
        let aggressive_cpu = ((arch.n_layers as f64 * 0.75) as i32).min(arch.n_layers as i32 - 1);
        let ctx = max_context(
            model_size_bytes,
            arch,
            "q8_0",
            "q8_0",
            parallel_slots,
            ubatch,
            aggressive_cpu,
            available_vram_bytes,
            fit_gran,
            headroom,
            n_ctx_train,
            is_unified_memory,
            backend,
        );
        let bd = full_estimate(
            model_size_bytes,
            arch,
            ctx,
            "q8_0",
            "q8_0",
            parallel_slots,
            ubatch,
            aggressive_cpu,
            -1,
            available_vram_bytes,
            0,
            is_unified_memory,
            EstimatorOptions {
                backend,
                evidence: if backend == Backend::RapidMlx {
                    EstimateEvidence::Approximate
                } else {
                    EstimateEvidence::Measured
                },
                ..Default::default()
            },
        );
        scenarios.push(ContextScenario {
            label: format!("Extended ({}× CPU offload)", aggressive_cpu),
            kv_quant_k: "q8_0".into(),
            kv_quant_v: "q8_0".into(),
            context_size: ctx,
            n_cpu_moe: Some(aggressive_cpu),
            vram_total_gb: bd.total_bytes as f64 / 1e9,
            recommended: false,
            warning: Some(format!(
                "Expert tensors from {aggressive_cpu} layers on CPU — slower generation"
            )),
            note: format!("{} tokens", format_ctx(ctx)),
        });
    }

    scenarios
}

fn format_ctx(ctx: u64) -> String {
    if ctx >= 1_000_000 {
        return format!("{:.1}M", ctx as f64 / 1e6);
    }
    if ctx >= 1_000 {
        return format!("{}K", ctx / 1000);
    }
    ctx.to_string()
}

// ── Pre-download quant comparison table ───────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize)]
pub struct QuantOption {
    pub quant: String,
    pub label: String,
    pub model_size_gb: f64,
    pub fits_vram: bool,
    /// Max context at q8_0 KV (agentic quality).
    pub max_ctx_q8: u64,
    /// Max context at q4_0 KV (maximum context).
    pub max_ctx_q4: u64,
    pub quality: QuantQuality,
    pub is_imatrix: bool,
    pub large_moe_only: bool,
    pub recommended: bool,
    pub quality_label: String,
    pub notes: Vec<String>,
}

/// Build a quant comparison table for the wizard's pre-download advisor.
///
/// `param_b`: approximate parameter count (from HF metadata)
/// `arch`: architecture (from introspection or `ModelArch::from_name_and_params`)
/// `available_vram_bytes`: effective available memory (caller must subtract OS overhead on unified)
/// `use_case`: affects the recommended-quant choice
/// `workload_scenario`: optional workload scenario for workload-fit quant guidance.
///   When present, replaces generic "8k context" with scenario-specific parameters.
///   Recommended badge requires scenario-policy fit (Builder item 11).
/// `is_unified_memory`: true for Apple Silicon — tightens headroom and fits check
#[allow(clippy::too_many_arguments)]
pub fn quant_comparison_table(
    param_b: f64,
    arch: &ModelArch,
    model_name: &str,
    available_vram_bytes: u64,
    use_case: UseCase,
    workload_scenario: Option<super::workload_scenarios::WorkloadScenario>,
    parallel_slots: u32,
    is_unified_memory: bool,
    backend: Backend,
) -> Vec<QuantOption> {
    // Quants we show in the advisor (sorted from highest to lowest quality)
    let show_quants = [
        "f16", "q8_0", "q6_k", "q5_k_m", "q5_k_s", "q4_k_m", "q4_k_s", "iq4_xs", "q4_0", "q3_k_m",
        "iq3_m", "iq3_xs", "q2_k", "iq2_xxs", "iq2_xs", "iq1_m",
    ];

    let ctx = QuantTableCtx::new(
        param_b,
        arch,
        model_name,
        available_vram_bytes,
        use_case,
        workload_scenario,
        parallel_slots,
        is_unified_memory,
        backend,
    );

    let mut options: Vec<QuantOption> = Vec::new();
    let mut best_quant: Option<String> = None;
    let mut best_score = 0u64;

    for &q_name in &show_quants {
        let qi = match find_quant(q_name) {
            Some(qi) => qi,
            None => continue,
        };
        // Skip large-MoE-only quants for dense or small models
        if qi.large_moe_only && param_b < 70.0 && !arch.is_moe() {
            continue;
        }
        let quality = if ctx.is_gemma4_qat && q_name == "q4_0" {
            QuantQuality::Excellent
        } else {
            qi.quality
        };
        let model_bytes = estimate_model_size_bytes(param_b, q_name);

        let (opt, score) = ctx.build_option(
            q_name.to_string(),
            qi.label.to_string(),
            model_bytes,
            quality,
            qi.is_imatrix,
            qi.large_moe_only,
            ctx.is_gemma4_qat && q_name == "q4_0",
        );
        if score > best_score {
            best_score = score;
            best_quant = Some(q_name.to_string());
        }
        options.push(opt);
    }

    // Gemma 4 QAT is explicitly trained for Q4_0. Prefer that target whenever it
    // fits instead of allowing a generic higher-bit option to win a tied score.
    if ctx.is_gemma4_qat
        && options
            .iter()
            .any(|opt| opt.quant == "q4_0" && opt.fits_vram)
    {
        best_quant = Some("q4_0".into());
    }

    mark_recommended(&mut options, best_quant);
    options
}

/// Build a quant comparison table from the repo's *actual* GGUF files instead
/// of a synthetic standard-quant list. Used when the HF repo has already been
/// resolved and its file listing (name + size) is known — this way repos that
/// use non-standard quant naming (imatrix "IQ4_XXS", custom mixed-precision
/// schemes like "APEX-MTP-Balanced", etc.) get an advisor table that matches
/// what's actually downloadable, with real sizes instead of estimates.
///
/// Quality/is_imatrix/large_moe_only can't be looked up by name for
/// non-standard labels, so they're approximated from the file's measured
/// bits-per-weight against the nearest standard quant (see
/// `nearest_quant_for_bpw`) — the actual file size is still exact and drives
/// every fit/context number, so the estimate is only ever "how big is this",
/// not "how precise is this".
#[allow(clippy::too_many_arguments)]
pub fn quant_comparison_table_from_files(
    param_b: f64,
    arch: &ModelArch,
    model_name: &str,
    files: &[(String, u64)],
    available_vram_bytes: u64,
    use_case: UseCase,
    workload_scenario: Option<super::workload_scenarios::WorkloadScenario>,
    parallel_slots: u32,
    is_unified_memory: bool,
    backend: Backend,
) -> Vec<QuantOption> {
    let ctx = QuantTableCtx::new(
        param_b,
        arch,
        model_name,
        available_vram_bytes,
        use_case,
        workload_scenario,
        parallel_slots,
        is_unified_memory,
        backend,
    );

    let mut options: Vec<QuantOption> = Vec::new();
    let mut best_quant: Option<String> = None;
    let mut best_score = 0u64;

    for (label, size_bytes) in files {
        if *size_bytes == 0 {
            continue;
        }
        let measured_bpw = (*size_bytes as f64 * 8.0) / (param_b * 1e9);
        let exact_match = find_quant(label);
        let nearest = exact_match.unwrap_or_else(|| nearest_quant_for_bpw(measured_bpw));
        let quality = if ctx.is_gemma4_qat && nearest.name == "q4_0" {
            QuantQuality::Excellent
        } else {
            nearest.quality
        };

        let (mut opt, score) = ctx.build_option(
            label.clone(),
            label.clone(),
            *size_bytes,
            quality,
            nearest.is_imatrix,
            nearest.large_moe_only,
            false,
        );
        if exact_match.is_none() {
            opt.notes.push(
                "Quality estimated from file size (non-standard quant name) — check the repo's own benchmarks for actual quality".into(),
            );
        }
        if score > best_score {
            best_score = score;
            best_quant = Some(label.clone());
        }
        options.push(opt);
    }

    mark_recommended(&mut options, best_quant);
    options
}

fn mark_recommended(options: &mut [QuantOption], best_quant: Option<String>) {
    if let Some(best) = best_quant {
        for opt in options.iter_mut() {
            if opt.quant == best {
                opt.recommended = true;
            }
        }
    }
}

/// Shared per-request context (headroom, scenario params, gemma4-qat detection)
/// used to build each row of a quant comparison table, regardless of whether
/// the candidate list comes from the synthetic standard-quant set or a repo's
/// real file listing.
struct QuantTableCtx<'a> {
    arch: &'a ModelArch,
    available_vram_bytes: u64,
    is_unified_memory: bool,
    backend: Backend,
    headroom: f64,
    is_gemma4_qat: bool,
    min_fit_context_tokens: u64,
    effective_parallel_slots: u32,
}

impl<'a> QuantTableCtx<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        _param_b: f64,
        arch: &'a ModelArch,
        model_name: &str,
        available_vram_bytes: u64,
        use_case: UseCase,
        workload_scenario: Option<super::workload_scenarios::WorkloadScenario>,
        parallel_slots: u32,
        is_unified_memory: bool,
        backend: Backend,
    ) -> Self {
        let headroom = compute_headroom(available_vram_bytes, is_unified_memory);
        let lower_name = model_name.to_ascii_lowercase();
        let is_gemma4_qat = (lower_name.contains("gemma-4") || lower_name.contains("gemma4"))
            && lower_name.contains("qat");

        // Builder item 11: use workload scenario parameters if provided, otherwise fall back to
        // generic 8k context baseline. Unsloth values are preserved for agentic/coding modes.
        let scenario_params = workload_scenario
            .as_ref()
            .map(|s| s.to_estimator_params(super::workload_scenarios::ClientType::App));

        // Minimum context tokens for fit check: scenario-based or generic 8k fallback.
        // Agentic/coding scenarios use higher minimums to ensure tool-calling coherence.
        let min_fit_context_tokens = match use_case {
            UseCase::Agentic => scenario_params
                .map(|p| p.planning_context_tokens.max(32_000))
                .unwrap_or(32_000),
            UseCase::General => scenario_params
                .map(|p| p.planning_context_tokens.max(16_000))
                .unwrap_or(8_192),
            UseCase::Roleplay => scenario_params
                .map(|p| p.planning_context_tokens.max(32_000))
                .unwrap_or(16_000),
        };

        let effective_parallel_slots = scenario_params
            .map(|p| p.parallel_slots)
            .unwrap_or(parallel_slots.max(1));

        Self {
            arch,
            available_vram_bytes,
            is_unified_memory,
            backend,
            headroom,
            is_gemma4_qat,
            min_fit_context_tokens,
            effective_parallel_slots,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_option(
        &self,
        quant_key: String,
        label: String,
        model_bytes: u64,
        quality: QuantQuality,
        is_imatrix: bool,
        large_moe_only: bool,
        is_gemma4_qat_target: bool,
    ) -> (QuantOption, u64) {
        let model_gb = model_bytes as f64 / 1e9;
        let min_kv = kv_cache_bytes(
            self.arch,
            self.min_fit_context_tokens,
            self.effective_parallel_slots,
            "q8_0",
            "q8_0",
        );
        let oh = match self.backend {
            Backend::RapidMlx => mlx_overhead_base_bytes(self.arch, 512),
            Backend::LlamaCpp if self.is_unified_memory => {
                metal_overhead_base_bytes(self.arch, 512)
            }
            Backend::LlamaCpp => discrete_overhead_base_bytes(self.arch, 512),
        };
        let fits = model_bytes + oh + min_kv < self.available_vram_bytes;

        let max_q8 = max_context(
            model_bytes,
            self.arch,
            "q8_0",
            "q8_0",
            self.effective_parallel_slots,
            512,
            0,
            self.available_vram_bytes,
            1024,
            self.headroom,
            None, // pre-download advisor: VRAM-limited maxes only
            self.is_unified_memory,
            self.backend,
        );
        let max_q4 = max_context(
            model_bytes,
            self.arch,
            "q4_0",
            "q4_0",
            self.effective_parallel_slots,
            512,
            0,
            self.available_vram_bytes,
            1024,
            self.headroom,
            None, // pre-download advisor: VRAM-limited maxes only
            self.is_unified_memory,
            self.backend,
        );

        let mut notes = Vec::new();
        if is_imatrix {
            notes.push("Requires imatrix calibration for best quality".into());
        }
        if large_moe_only {
            notes.push("Designed for large MoE models; poor for dense".into());
        }
        if is_gemma4_qat_target {
            notes.push(
                "Official Gemma 4 QAT target; preserves near-BF16 quality at 4-bit weights".into(),
            );
        }
        match quality {
            QuantQuality::Reference => notes.push("Bit-accurate reference quality".into()),
            QuantQuality::Excellent => {
                if !is_gemma4_qat_target {
                    notes
                        .push("Near-lossless; essentially equivalent to F16 for most tasks".into());
                }
            }
            QuantQuality::VeryGood => {}
            QuantQuality::Good => {}
            QuantQuality::Acceptable => {
                notes.push("Noticeable quality reduction on complex tasks".into())
            }
            QuantQuality::Fair => {
                notes.push("Significant quality loss; only for maximum context or large MoE".into())
            }
            QuantQuality::Reduced | QuantQuality::VeryLow => {
                notes.push(
                    "Heavy quality reduction; avoid for production use on dense models".into(),
                );
            }
        }

        // Score for recommendation: balance of quality × context × fits
        let score = if fits {
            max_q8.min(128_000) * quality_weight(quality)
        } else {
            0
        };

        (
            QuantOption {
                quant: quant_key,
                label,
                model_size_gb: model_gb,
                fits_vram: fits,
                max_ctx_q8: max_q8,
                max_ctx_q4: max_q4,
                quality,
                is_imatrix,
                large_moe_only,
                recommended: false, // filled in by mark_recommended
                quality_label: quality_label(quality),
                notes,
            },
            score,
        )
    }
}

fn quality_weight(q: QuantQuality) -> u64 {
    match q {
        QuantQuality::Reference => 10,
        QuantQuality::Excellent => 9,
        QuantQuality::VeryGood => 8,
        QuantQuality::Good => 7,
        QuantQuality::Acceptable => 5,
        QuantQuality::Fair => 3,
        QuantQuality::Reduced => 1,
        QuantQuality::VeryLow => 0,
    }
}

fn quality_label(q: QuantQuality) -> String {
    match q {
        QuantQuality::Reference => "Reference",
        QuantQuality::Excellent => "Excellent",
        QuantQuality::VeryGood => "Very Good",
        QuantQuality::Good => "Good",
        QuantQuality::Acceptable => "Acceptable",
        QuantQuality::Fair => "Fair",
        QuantQuality::Reduced => "Reduced",
        QuantQuality::VeryLow => "Very Low",
    }
    .into()
}

// ── Backward-compat wrapper kept for existing callers ─────────────────────────

/// Simple estimate used by the existing `/api/vram/estimate` endpoint.
/// Kept for backward compatibility; prefer `full_estimate` for new callers.
#[allow(clippy::too_many_arguments)]
pub fn estimate_vram(
    model_size_bytes: u64,
    context_size: u64,
    kv_quant: &str,
    batch_size: u32,
    ubatch_size: u32,
    speculative_decoding: bool,
    mmproj_size_bytes: u64,
    n_cpu_moe: Option<i32>,
    available_vram_bytes: u64,
) -> VramEstimate {
    // Build a minimal arch from params we have.
    let arch = ModelArch {
        // Without introspection we don't know n_layers/n_kv_heads; use zero so
        // kv_cache_bytes falls back to 0 and we keep backward-compat behaviour
        // (caller supplies model_size + rough context instead of per-token formula).
        mmproj_bytes: mmproj_size_bytes,
        ..Default::default()
    };

    // Legacy KV heuristic: context × (batch or ubatch) × bytes_per_tok
    // Keep this so existing presets with no arch info still get a number.
    let kv_bytes_per_tok = {
        let bpe = kv_elem_bytes(kv_quant);
        (bpe * 64.0) as u64 // 64 = rough bytes/token for an "average" 7-30B model
    };
    let effective_batch = batch_size.max(ubatch_size).max(1);
    let kv_est = context_size
        .saturating_mul(effective_batch as u64)
        .saturating_mul(kv_bytes_per_tok);

    let cpu_moe = n_cpu_moe.unwrap_or(0);
    let (weight_vram, ram) = if arch.is_moe() {
        moe_weight_split(model_size_bytes, &arch, cpu_moe)
    } else {
        (model_size_bytes, 0)
    };

    let mut total = weight_vram
        .saturating_add(kv_est)
        .saturating_add(mmproj_size_bytes);

    if speculative_decoding {
        total = total.saturating_add(model_size_bytes / 8);
    }
    // Legacy/coarse path (no platform flag, often no arch detail): use the Metal model as a
    // light, safe default — base reserve + ~6.5% of the rough KV estimate.
    total = total.saturating_add(metal_overhead_bytes(&arch, ubatch_size, kv_est));

    let (recommendation, note) = if available_vram_bytes == 0 {
        (VramRecommendation::Risk, "No VRAM info available.".into())
    } else if total <= available_vram_bytes * 82 / 100 {
        (
            VramRecommendation::Fit,
            "Configuration fits with headroom.".into(),
        )
    } else if total <= available_vram_bytes {
        (
            VramRecommendation::Tight,
            "Fits, but near the VRAM limit.".into(),
        )
    } else if total <= available_vram_bytes * 150 / 100 {
        (VramRecommendation::Risk, "Likely exceeds VRAM.".into())
    } else {
        (
            VramRecommendation::WontFit,
            "Significantly over VRAM budget.".into(),
        )
    };

    VramEstimate {
        estimated_vram_bytes: total,
        estimated_ram_bytes: ram,
        available_vram_bytes,
        recommendation,
        note,
    }
}

/// Legacy struct returned by `estimate_vram` for backward compatibility.
#[derive(Debug, Clone, serde::Serialize)]
pub struct VramEstimate {
    pub estimated_vram_bytes: u64,
    pub estimated_ram_bytes: u64,
    pub available_vram_bytes: u64,
    pub recommendation: VramRecommendation,
    pub note: String,
}
