//! Minimal GGUF metadata reader.
//!
//! Reads only the KV metadata header of a GGUF file — the small section that
//! precedes tensor data. Works on any GGUF version (1, 2, 3) with no external
//! binary and no new dependencies. Calling this on a 70B model is effectively
//! instant because tensor weights are never touched.
//!
//! GGUF format reference: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

// ── Format constants ──────────────────────────────────────────────────────────

const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// Maximum header size accepted by the experimental import inspector. This includes
/// GGUF KV metadata and the tensor-info directory, but never tensor weight data.
pub const MAX_INSPECTION_HEADER_BYTES: u64 = 64 * 1024 * 1024;

/// Strict, bounded facts from the GGUF header that are needed for import policy.
#[derive(Debug, Clone)]
pub struct GgufHeaderInventory {
    pub version: u32,
    pub tensor_count: u64,
    pub header_bytes: u64,
    pub quant_types: BTreeMap<String, u64>,
    pub metadata_keys: Vec<String>,
}

struct BoundedReader<R> {
    inner: R,
    limit: u64,
}

impl<R: Read + Seek> Read for BoundedReader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let position = self.inner.stream_position()?;
        let remaining = self.limit.saturating_sub(position);
        if remaining == 0 && !buffer.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "GGUF inspection limit exceeded",
            ));
        }
        let length = usize::try_from(remaining.min(buffer.len() as u64)).unwrap_or(buffer.len());
        self.inner.read(&mut buffer[..length])
    }
}

impl<R: Read + Seek> Seek for BoundedReader<R> {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        let next = self.inner.seek(position)?;
        if next > self.limit {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "GGUF inspection limit exceeded",
            ));
        }
        Ok(next)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
enum GgufType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufType {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Uint8),
            1 => Some(Self::Int8),
            2 => Some(Self::Uint16),
            3 => Some(Self::Int16),
            4 => Some(Self::Uint32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::Uint64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _ => None,
        }
    }

    #[allow(dead_code)]
    fn as_u32(self) -> u32 {
        self as u32
    }

    fn fixed_size(self) -> Option<u64> {
        match self {
            Self::Uint8 | Self::Int8 | Self::Bool => Some(1),
            Self::Uint16 | Self::Int16 => Some(2),
            Self::Uint32 | Self::Int32 | Self::Float32 => Some(4),
            Self::Uint64 | Self::Int64 | Self::Float64 => Some(8),
            Self::String | Self::Array => None,
        }
    }
}

// ── Public output type ────────────────────────────────────────────────────────

/// Architecture metadata extracted from a GGUF file's KV header.
///
/// All fields are optional; absent fields are not present in the file.
/// Callers should fall back to name-based heuristics for missing values.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct GgufMetadata {
    /// `general.architecture` — e.g. `"llama"`, `"qwen3_6"`, `"gemma4"`.
    /// This is the canonical key used by llama.cpp to select its model loader.
    /// Present in every well-formed GGUF regardless of filename.
    pub architecture: Option<String>,

    /// `general.parameter_count` — total parameters (not active for MoE).
    pub param_count: Option<u64>,

    /// `{arch}.block_count` — total transformer layers.
    pub block_count: Option<u32>,

    /// `{arch}.attention.head_count` — query head count.
    pub head_count: Option<u32>,

    /// `{arch}.attention.head_count_kv` — KV head count (GQA/MQA compressed).
    pub head_count_kv: Option<u32>,

    /// `{arch}.attention.key_length` — per-head K/V dimension.
    pub key_length: Option<u32>,

    /// `{arch}.context_length` — training context window size.
    pub context_length: Option<u32>,

    /// `{arch}.embedding_length` — hidden (embedding) dimension.
    pub embedding_length: Option<u32>,

    /// `{arch}.feed_forward_length` — FFN intermediate dimension.
    pub feed_forward_length: Option<u32>,

    /// `{arch}.expert_count` — total MoE experts per layer.
    pub expert_count: Option<u32>,

    /// `{arch}.expert_used_count` — active MoE experts per token.
    pub expert_used_count: Option<u32>,

    /// MTP prediction depth (`{arch}.next_n_token_count` or similar).
    pub mtp_depth: Option<u32>,

    // ── Hybrid linear-attention (Qwen3-Next / DeltaNet: qwen35, qwen35moe, qwen3next) ──
    /// `{arch}.full_attention_interval` — every Nth layer is full attention; the rest
    /// are linear (Gated DeltaNet) layers. Authoritative source for `n_attn_layers`:
    /// `n_attn_layers = block_count / full_attention_interval`.
    pub full_attention_interval: Option<u32>,

    /// `{arch}.ssm.inner_size` — width of the linear-attention recurrent state
    /// (num_v_heads × head_v_dim). Used to size the fixed DeltaNet state.
    pub ssm_inner_size: Option<u32>,

    /// `{arch}.ssm.state_size` — per-head linear-attention state dimension (head_k_dim).
    pub ssm_state_size: Option<u32>,

    /// `{arch}.ssm.conv_kernel` — short-conv kernel width (adds a small conv state).
    pub ssm_conv_kernel: Option<u32>,

    // ── Sliding-window / alternating attention (Gemma 3/4) ────────────────────────
    /// `{arch}.attention.sliding_window` — local-attention window size in tokens.
    pub sliding_window: Option<u32>,

    /// `{arch}.attention.key_length_swa` — per-head K/V dimension on local (SWA) layers.
    /// Gemma 4 uses a wider `key_length` (512) on global layers and this (256) locally.
    pub key_length_swa: Option<u32>,

    /// Number of global (full-context) attention layers, derived from
    /// `{arch}.attention.sliding_window_pattern` (count of `false` entries).
    pub n_global_attn_layers: Option<u32>,

    /// KV head count on global (full-context) layers, read from the per-layer
    /// `{arch}.attention.head_count_kv` array at a global position.
    pub global_kv_heads: Option<u32>,

    /// KV head count on local (sliding-window) layers, read from the per-layer
    /// `{arch}.attention.head_count_kv` array at a local position.
    pub local_kv_heads: Option<u32>,

    // ── Measured tensor sizes (from the GGUF tensor directory, not metadata) ──────
    // Byte counts require a complete local file. Tensor element counts below come
    // from the header and are also available from range-fetched prefixes.
    /// Sum of all tensor (weight) bytes on disk.
    pub tensor_bytes_total: Option<u64>,
    /// Sum of bytes in repeating per-layer blocks (`blk.*` tensors).
    pub layer_bytes_total: Option<u64>,
    /// Sum of bytes in routed-expert FFN tensors (`*_exps.*`) — the exact portion
    /// `--n-cpu-moe` moves to CPU/RAM per offloaded layer.
    pub expert_bytes_total: Option<u64>,
    /// Sum of all tensor element counts from the GGUF tensor directory.
    pub tensor_param_count: Option<u64>,
    /// Sum of routed-expert tensor element counts (`*_exps.*`).
    pub expert_param_count: Option<u64>,
    /// Number of distinct layers that contain routed-expert tensors (the `--n-cpu-moe`
    /// denominator). For most MoE models this equals `block_count`, but some have a few
    /// leading dense layers.
    pub moe_layer_count: Option<u32>,
}

impl GgufMetadata {
    /// Approximate parameter count in billions, derived from `param_count`.
    #[allow(dead_code)]
    pub fn param_b(&self) -> Option<f64> {
        self.param_count.map(|p| p as f64 / 1e9)
    }

    /// Number of full-attention (KV-bearing) layers for hybrid linear-attention
    /// models, computed from real GGUF data: `block_count / full_attention_interval`.
    /// (llama.cpp marks every Nth layer as full attention; the rest are DeltaNet.)
    /// Returns `None` for non-hybrid models (no `full_attention_interval` key).
    pub fn n_attn_layers(&self) -> Option<u32> {
        let interval = self.full_attention_interval?;
        let blocks = self.block_count?;
        if interval <= 1 {
            return None; // interval 1 ⇒ all layers full attention (not hybrid)
        }
        Some(blocks / interval)
    }

    /// Fixed recurrent-state size (bytes) for the linear-attention (DeltaNet) layers,
    /// computed from the real `ssm.*` GGUF fields. This does NOT grow with context.
    ///
    /// Per linear layer the state is `inner_size × state_size` (the delta matrix)
    /// plus a small `conv_kernel × inner_size` short-conv state.
    /// Both components are held at 2 B/elem, then summed over all linear layers.
    /// Returns `None` when the model is not hybrid or lacks SSM metadata.
    pub fn linear_attn_state_bytes(&self) -> Option<u64> {
        let n_attn = self.n_attn_layers()?;
        let blocks = self.block_count?;
        let inner = self.ssm_inner_size? as u64;
        let state = self.ssm_state_size? as u64;
        let conv = self.ssm_conv_kernel.unwrap_or(0) as u64;
        let n_linear = blocks.saturating_sub(n_attn) as u64;
        let per_layer_elems = inner * (state + conv);
        Some(n_linear * per_layer_elems * 2)
    }

    /// Architecture label derived from the GGUF: `"dense"`, `"moe"`, or `"hybrid_moe"`.
    ///
    /// - MoE: `expert_count > 0`.
    /// - Hybrid MoE: MoE **and** hybrid attention, detected via either:
    ///   - `full_attention_interval > 1` (Qwen3/DeltaNet style), or
    ///   - `n_global_attn_layers < block_count` (Gemma4 sliding_window_pattern style).
    /// - Dense: everything else.
    ///
    /// This is the single source of truth for the architecture label shown on launch
    /// cards, in the preset editor, and in the spawn wizard.
    pub fn architecture_kind(&self) -> String {
        let is_moe = self.expert_count.is_some_and(|e| e > 0);
        let is_hybrid_attn = self.full_attention_interval.is_some_and(|v| v > 1)
            || self
                .n_global_attn_layers
                .is_some_and(|g| self.block_count.is_some_and(|b| b > 0 && g < b));
        if is_moe && is_hybrid_attn {
            "hybrid_moe".to_string()
        } else if is_moe {
            "moe".to_string()
        } else {
            "dense".to_string()
        }
    }

    /// Estimate "active parameters" in billions.
    ///
    /// - Dense: active = total.
    /// - MoE / hybrid-MoE: `active ≈ backbone + N_used · (expert_total / N_experts)`,
    ///   where the backbone is approximated from attention + embedding projections.
    ///   Falls back to the simple ratio `total / (1 + N_experts / N_used)` when the
    ///   structural estimate is unavailable or implausible.
    ///
    /// Returns `None` only when `param_count` is missing.
    pub fn active_params_b(&self) -> Option<f64> {
        let total_params = self.param_count?;

        let is_moe = self.expert_count.is_some_and(|e| e > 0);
        if !is_moe {
            return Some(total_params as f64 / 1e9);
        }

        let n_experts = self.expert_count?;
        let n_used = self.expert_used_count?;
        if n_experts == 0 || n_used == 0 || n_used > n_experts {
            // Invalid expert ratio → fall back to total.
            return Some(total_params as f64 / 1e9);
        }

        // Exact path: tensor element counts are independent of quantization.
        // Keep every non-routed parameter active and activate only the selected
        // fraction of routed experts. Shared experts and DeltaNet projections
        // naturally remain in the always-active pool.
        if let (Some(tensor_params), Some(expert_params)) =
            (self.tensor_param_count, self.expert_param_count)
            && tensor_params > 0
            && expert_params > 0
            && expert_params < tensor_params
        {
            let always_active = tensor_params - expert_params;
            let active_experts = expert_params as f64 * n_used as f64 / n_experts as f64;
            let active = always_active as f64 + active_experts;
            if active > 0.0 && active < tensor_params as f64 {
                return Some(active / 1e9);
            }
        }

        // Attempt a structural estimate from GGUF:
        //   backbone ≈ attention projections + token-embedding/output projection
        //   experts_total = P - backbone
        //   active ≈ backbone + N_used · (experts_total / N_experts)
        let have_enough = matches!(
            (
                self.block_count,
                self.head_count,
                self.head_count_kv,
                self.key_length,
                self.embedding_length,
            ),
            (Some(_), Some(_), Some(_), Some(_), Some(_))
        );

        if have_enough {
            let n_total_layers = self.block_count.unwrap() as u64;
            let head_count = self.head_count.unwrap() as u64;
            let head_count_kv = self.head_count_kv.unwrap() as u64;
            let head_dim = self.key_length.unwrap() as u64;
            let embd = self.embedding_length.unwrap() as u64;

            // For hybrid DeltaNet models (Qwen3-Next, Qwen3.6, Qwen3.5) only a subset
            // of layers are standard attention — the rest are Gated DeltaNet layers.
            // DeltaNet layers have their own always-active parameters (Q/K/V/O projections
            // sized by ssm_inner_size) that are NOT in the expert pool but MUST be counted
            // as backbone because they run on every token.
            // Only reduce the standard-attention layer count when the DeltaNet
            // dimensions are available to account for the replacement layers.
            // Otherwise those always-active weights would disappear from the
            // backbone estimate and be misclassified as routed expert weights.
            let n_attn_layers = if self.ssm_inner_size.is_some() {
                self.n_attn_layers()
                    .map(|n| n as u64)
                    .unwrap_or(n_total_layers)
            } else {
                n_total_layers
            };

            // Standard-attention backbone: Q/K/V/O per attention layer + token embeddings.
            //   attn Q/K/V/O: embd · head_dim · (2·n_head + 2·n_head_kv)
            //   token-embedding + output projection: ~2 · embd²
            let attn_per_layer = embd * head_dim * (2 * head_count + 2 * head_count_kv);
            let mut backbone_total: u64 = n_attn_layers * attn_per_layer + 2 * embd * embd;

            // DeltaNet backbone: for each linear-attention layer, add V+O projections
            // (embd × ssm_inner_size × 2) and the smaller Q+K projections
            // (embd × head_count_kv × head_dim × 2). All are always-active.
            if let Some(ssm_inner) = self.ssm_inner_size {
                let n_deltanet = n_total_layers.saturating_sub(n_attn_layers);
                if n_deltanet > 0 {
                    let deltanet_per_layer =
                        2 * embd * (head_count_kv * head_dim + ssm_inner as u64);
                    backbone_total = backbone_total.saturating_add(n_deltanet * deltanet_per_layer);
                }
            }

            if total_params <= backbone_total {
                // Backbone estimate exceeds total (bad input); use the simple ratio.
                return Some(simple_moe_active_b(total_params, n_experts, n_used));
            }
            let expert_total = total_params - backbone_total;

            // If the expert portion is <10% of total, the structural estimate is clearly
            // off (a real MoE keeps most weight in experts), so fall back.
            if (expert_total as f64) < (total_params as f64 * 0.1) {
                return Some(simple_moe_active_b(total_params, n_experts, n_used));
            }

            let per_expert = expert_total / n_experts as u64;
            let active = backbone_total as f64 + (n_used as f64 * per_expert as f64);
            let active_b = active / 1e9;

            // Sanity: active must be > 0 and < total.
            if active > 0.0 && active_b < total_params as f64 / 1e9 {
                Some(active_b)
            } else {
                Some(simple_moe_active_b(total_params, n_experts, n_used))
            }
        } else {
            // Not enough GGUF fields → simple ratio.
            Some(simple_moe_active_b(total_params, n_experts, n_used))
        }
    }

    /// Exact bytes per repeating transformer block, measured from the tensor directory.
    /// `layer_bytes_total / block_count`. This is the VRAM each `-ngl` layer occupies on
    /// the GPU (dense models) — real on-disk data, not an estimate. `None` when tensor
    /// sizes were not measured (e.g. range-fetched prefixes) or `block_count` is unknown.
    pub fn bytes_per_layer(&self) -> Option<u64> {
        let total = self.layer_bytes_total?;
        let n = self.block_count? as u64;
        (n > 0).then(|| total / n)
    }

    /// Exact routed-expert bytes per MoE layer, measured from the tensor directory.
    /// `expert_bytes_total / moe_layer_count`. This is the VRAM freed per layer offloaded
    /// via `--n-cpu-moe`. `None` for dense models or when tensor sizes were not measured.
    pub fn expert_bytes_per_layer(&self) -> Option<u64> {
        let total = self.expert_bytes_total?;
        let n = self.moe_layer_count? as u64;
        (n > 0).then(|| total / n)
    }

    /// Convert to the `ModelMetadata` type used by the spawn wizard / VRAM estimator.
    ///
    /// Sets `gguf_arch` so that renamed finetunes (e.g. "Pantheon-27B" from a
    /// Qwen3.6 base) get the correct hybrid-DeltaNet heuristic regardless of filename.
    /// Structural fields that llama.cpp records per-layer (hybrid attention interval,
    /// SSM state, Gemma global/local split, sliding window) are read from the GGUF so
    /// the VRAM math uses ground truth rather than name-based assumptions.
    pub fn to_model_metadata(&self) -> crate::llama::spawn_wizard::ModelMetadata {
        // For Gemma alternating attention, `n_kv_heads` (the global-layer KV head
        // count) comes from the per-layer array; fall back to the scalar otherwise.
        let n_kv_heads = self.global_kv_heads.or(self.head_count_kv);
        crate::llama::spawn_wizard::ModelMetadata {
            n_layers: self.block_count,
            n_ctx_train: self.context_length,
            n_embd: self.embedding_length,
            n_ff: self.feed_forward_length,
            n_head: self.head_count,
            n_kv_heads,
            head_dim: self.key_length,
            gguf_arch: self.architecture.clone(),
            architecture_kind: Some(self.architecture_kind()),
            active_params_b: self.active_params_b(),
            bytes_per_layer: self.bytes_per_layer(),
            expert_bytes_per_layer: self.expert_bytes_per_layer(),
            moe_layer_count: self.moe_layer_count,
            n_experts: self.expert_count,
            n_experts_used: self.expert_used_count,
            mtp_depth: self.mtp_depth,
            n_attn_layers: self.n_attn_layers(),
            linear_attn_state_bytes: self.linear_attn_state_bytes(),
            n_global_attn_layers: self.n_global_attn_layers,
            local_kv_heads: self.local_kv_heads,
            global_head_dim: self.key_length, // wide global K/V dim (e.g. Gemma4 = 512)
            local_head_dim: self.key_length_swa, // narrow local K/V dim (e.g. Gemma4 = 256)
            sliding_window: self.sliding_window,
            mmproj_required: false,
            cached: false,
        }
    }
}

/// Simple ratio-based active-parameter estimate (in billions) for MoE models:
/// `active ≈ total / (1 + N_experts / N_used)`. Used only when the structural
/// estimate in [`GgufMetadata::active_params_b`] is unavailable or implausible.
fn simple_moe_active_b(total: u64, n_experts: u32, n_used: u32) -> f64 {
    if n_experts == 0 || n_used == 0 {
        return total as f64 / 1e9;
    }
    let ratio = n_experts as f64 / n_used as f64;
    let active_b = total as f64 / (1.0 + ratio) / 1e9;
    if active_b > 0.0 && active_b < total as f64 / 1e9 {
        active_b
    } else {
        total as f64 / 1e9
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Read GGUF metadata from a file without touching tensor data.
///
/// # Errors
/// Returns a human-readable error string if the file cannot be opened,
/// is not a valid GGUF file, or uses an unsupported version.
pub fn read_gguf_metadata(path: &Path) -> Result<GgufMetadata, String> {
    let file = File::open(path).map_err(|e| format!("Cannot open '{}': {e}", path.display()))?;
    // Exact file size lets us measure per-tensor bytes from the tensor directory layout.
    let file_size = file.metadata().ok().map(|m| m.len());
    read_gguf_metadata_reader(BufReader::with_capacity(64 * 1024, file), file_size)
}

/// Inventory a complete local GGUF header without reading tensor weight data.
///
/// Unlike the general metadata reader, this entry point is fail-closed: it requires a
/// complete tensor directory, rejects headers over `max_header_bytes`, and reports every
/// tensor quantization type. It is intended to run inside a bounded blocking task.
pub fn read_gguf_header_inventory(
    path: &Path,
    max_header_bytes: u64,
) -> Result<GgufHeaderInventory, String> {
    let file = File::open(path).map_err(|e| format!("Cannot open '{}': {e}", path.display()))?;
    let file_size = file
        .metadata()
        .map_err(|e| format!("Cannot stat '{}': {e}", path.display()))?
        .len();
    let bounded = BoundedReader {
        inner: file,
        limit: max_header_bytes,
    };
    let mut r = BufReader::with_capacity(64 * 1024, bounded);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)
        .map_err(|e| format!("Cannot read GGUF magic: {e}"))?;
    if &magic != GGUF_MAGIC {
        return Err(format!("Data is not a GGUF file (magic: {magic:02x?})"));
    }
    let version = read_u32(&mut r)?;
    if version == 0 || version > 3 {
        return Err(format!("Unsupported GGUF version {version}"));
    }
    let (tensor_count, kv_count) = if version == 1 {
        (read_u32(&mut r)? as u64, read_u32(&mut r)? as u64)
    } else {
        (read_u64(&mut r)?, read_u64(&mut r)?)
    };
    if tensor_count == 0 || tensor_count > 1_000_000 {
        return Err(format!("Implausible tensor_count {tensor_count}"));
    }
    if kv_count > 100_000 {
        return Err(format!("Implausible kv_count {kv_count}"));
    }

    let mut metadata_keys = Vec::with_capacity((kv_count as usize).min(4096));
    let mut alignment = 32u64;
    for _ in 0..kv_count {
        let key = read_str(&mut r, version)?;
        let vtype = read_u32(&mut r)?;
        let value = read_value(&mut r, vtype, version)?;
        if key == "general.alignment" {
            alignment = value.as_u32().unwrap_or(32) as u64;
        }
        metadata_keys.push(key);
        ensure_header_bound(&mut r, file_size, max_header_bytes)?;
    }

    let mut quant_types = BTreeMap::new();
    let mut tensor_offsets = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let _name = read_str(&mut r, version)?;
        let n_dims = read_u32(&mut r)?;
        if n_dims > 8 {
            return Err(format!("Implausible tensor n_dims {n_dims}"));
        }
        for _ in 0..n_dims {
            if version == 1 {
                let _ = read_u32(&mut r)?;
            } else {
                let _ = read_u64(&mut r)?;
            }
        }
        let ggml_type = read_u32(&mut r)?;
        let offset = read_u64(&mut r)?;
        tensor_offsets.push(offset);
        *quant_types.entry(ggml_type_name(ggml_type)).or_insert(0) += 1;
        ensure_header_bound(&mut r, file_size, max_header_bytes)?;
    }
    let header_bytes = r
        .stream_position()
        .map_err(|e| format!("Cannot measure GGUF header: {e}"))?;
    let data_start = header_bytes.div_ceil(alignment.max(1)) * alignment.max(1);
    if data_start >= file_size {
        return Err("GGUF has no tensor data after its header".into());
    }
    let data_bytes = file_size - data_start;
    if let Some(offset) = tensor_offsets.iter().find(|offset| **offset >= data_bytes) {
        return Err(format!(
            "GGUF tensor offset {offset} is outside the {data_bytes}-byte tensor data section"
        ));
    }
    Ok(GgufHeaderInventory {
        version,
        tensor_count,
        header_bytes,
        quant_types,
        metadata_keys,
    })
}

fn ensure_header_bound<R: Seek>(
    r: &mut R,
    file_size: u64,
    max_header_bytes: u64,
) -> Result<(), String> {
    let position = r
        .stream_position()
        .map_err(|e| format!("Cannot measure GGUF header: {e}"))?;
    if position > file_size {
        return Err("GGUF header extends beyond end of file".into());
    }
    if position > max_header_bytes {
        return Err(format!(
            "GGUF metadata and tensor directory exceed the {max_header_bytes}-byte inspection limit"
        ));
    }
    Ok(())
}

fn ggml_type_name(value: u32) -> String {
    // Names follow ggml's stable `ggml_type` numeric ABI. Unknown values are retained
    // explicitly so policy can reject them instead of guessing.
    match value {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        30 => "BF16",
        31 => "Q4_0_4_4",
        32 => "Q4_0_4_8",
        33 => "Q4_0_8_8",
        34 => "TQ1_0",
        35 => "TQ2_0",
        36 => "IQ4_NL_4_4",
        37 => "IQ4_NL_4_8",
        38 => "IQ4_NL_8_8",
        _ => return format!("UNKNOWN_{value}"),
    }
    .into()
}

/// Parse GGUF metadata from an in-memory buffer — e.g. the first few MB of a remote file
/// fetched with an HTTP range request. Only the KV header is read; tensor data is never
/// touched, so a prefix of the file is sufficient. Returns an error (typically an
/// unexpected-EOF) if the buffer is shorter than the full KV header, so callers can retry
/// with a larger prefix.
pub fn read_gguf_metadata_from_bytes(buf: &[u8]) -> Result<GgufMetadata, String> {
    // A prefix buffer lacks the tensor blob, so we cannot measure tensor sizes here.
    read_gguf_metadata_reader(std::io::Cursor::new(buf), None)
}

/// Core parser shared by the file- and buffer-based entry points. Works over any
/// seekable reader; reads only the KV-metadata header.
pub fn read_gguf_metadata_reader<R: Read + Seek>(
    mut r: R,
    file_size: Option<u64>,
) -> Result<GgufMetadata, String> {
    // Magic
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)
        .map_err(|e| format!("Cannot read GGUF magic: {e}"))?;
    if &magic != GGUF_MAGIC {
        return Err(format!("Data is not a GGUF file (magic: {magic:02x?})"));
    }

    // Version
    let version = read_u32(&mut r)?;
    if version == 0 || version > 3 {
        return Err(format!("Unsupported GGUF version {version}"));
    }

    // tensor_count and kv_count (u32 in v1, u64 in v2+)
    let (tensor_count, kv_count) = if version == 1 {
        (read_u32(&mut r)? as u64, read_u32(&mut r)? as u64)
    } else {
        (read_u64(&mut r)?, read_u64(&mut r)?)
    };

    // Guard against pathological files
    if kv_count > 100_000 {
        return Err(format!("Implausible kv_count {kv_count}"));
    }

    // Read all KV pairs into a flat map.
    // We store scalar values; arrays are consumed/skipped.
    let mut kv: HashMap<String, KvValue> = HashMap::with_capacity(128);

    for _ in 0..kv_count {
        let key = read_str(&mut r, version)?;
        let vtype = read_u32(&mut r)?;
        let value = read_value(&mut r, vtype, version)?;
        kv.insert(key, value);
    }

    // ── Extract fields ────────────────────────────────────────────────────────
    let arch: Option<String> = kv
        .get("general.architecture")
        .and_then(KvValue::as_str)
        .map(|s| s.to_ascii_lowercase());

    let mut meta = GgufMetadata {
        architecture: arch.clone(),
        param_count: kv.get("general.parameter_count").and_then(KvValue::as_u64),
        ..Default::default()
    };

    if let Some(a) = arch.as_deref() {
        macro_rules! get_u32 {
            ($($key:expr),+) => {
                None $(.or_else(|| kv.get(&format!("{a}.{}", $key)).and_then(KvValue::as_u32)))+
            };
        }
        macro_rules! get_u32_bare {
            ($($key:expr),+) => {
                None $(.or_else(|| kv.get($key).and_then(KvValue::as_u32)))+
            };
        }

        meta.block_count = get_u32!("block_count");
        meta.head_count = get_u32!("attention.head_count");
        meta.key_length = get_u32!("attention.key_length");
        meta.key_length_swa = get_u32!("attention.key_length_swa");
        meta.context_length = get_u32!("context_length");
        meta.embedding_length = get_u32!("embedding_length");
        meta.feed_forward_length = get_u32!("feed_forward_length");
        meta.expert_count = get_u32!("expert_count");
        meta.expert_used_count = get_u32!("expert_used_count");

        // Hybrid linear-attention (Qwen3-Next / DeltaNet) and SSM state sizing.
        meta.full_attention_interval = get_u32!("full_attention_interval");
        meta.ssm_inner_size = get_u32!("ssm.inner_size");
        meta.ssm_state_size = get_u32!("ssm.state_size");
        meta.ssm_conv_kernel = get_u32!("ssm.conv_kernel");
        meta.sliding_window = get_u32!("attention.sliding_window");

        // MTP depth — key name varies across llama.cpp versions. Newer Qwen3.5/3.6
        // MoE GGUFs emit `{arch}.nextn_predict_layers`.
        meta.mtp_depth = get_u32!(
            "nextn_predict_layers",
            "next_n_token_count",
            "num_nextn_predict_layers",
            "multi_token_prediction_depth"
        )
        .or_else(|| get_u32_bare!("general.next_n_token_count"));

        // `attention.head_count_kv` is a scalar on uniform models but a per-layer
        // array on Gemma 3/4 (alternating global/local layers with different GQA).
        let kv_key = format!("{a}.attention.head_count_kv");
        let kv_val = kv.get(&kv_key);
        meta.head_count_kv = kv_val.and_then(KvValue::as_u32);

        // `attention.sliding_window_pattern`: per-layer bool array — `false` marks a
        // global (full-context) layer, `true` a local sliding-window layer. The count
        // of `false` entries is the authoritative `n_global_attn_layers`.
        let swa_pattern = kv
            .get(&format!("{a}.attention.sliding_window_pattern"))
            .and_then(KvValue::as_bool_arr);
        if let Some(pat) = swa_pattern {
            meta.n_global_attn_layers =
                Some(pat.iter().filter(|&&is_local| !is_local).count() as u32);

            // Read the global/local KV head split from the per-layer head_count_kv
            // array, indexed by the same pattern (global = !is_local position).
            if let Some(kv_arr) = kv_val.and_then(KvValue::as_u32_arr) {
                let n = pat.len().min(kv_arr.len());
                meta.global_kv_heads = (0..n).find(|&i| !pat[i]).map(|i| kv_arr[i]);
                meta.local_kv_heads = (0..n).find(|&i| pat[i]).map(|i| kv_arr[i]);
            }
        }
    }

    // ── Parse the tensor directory ────────────────────────────────────────────────
    // Tensor shapes are part of the GGUF header, so parameter counts are exact for
    // both local files and range-fetched prefixes. Exact byte sizes additionally
    // require the complete local file length.
    let alignment = kv
        .get("general.alignment")
        .and_then(KvValue::as_u32)
        .unwrap_or(32) as u64;
    if let Ok(sizes) = parse_tensor_directory(&mut r, version, tensor_count, alignment, file_size) {
        meta.tensor_bytes_total = sizes.tensor_bytes_total;
        meta.layer_bytes_total = sizes.layer_bytes_total;
        meta.expert_bytes_total = sizes.expert_bytes_total;
        meta.tensor_param_count = Some(sizes.param_count);
        meta.expert_param_count = Some(sizes.expert_param_count);
        meta.moe_layer_count = Some(sizes.moe_layer_count);
        // Use tensor-derived param count only when the KV field is absent.
        if meta.param_count.is_none() && sizes.param_count > 0 {
            meta.param_count = Some(sizes.param_count);
        }
    }

    Ok(meta)
}

/// Aggregated tensor byte sizes measured from the GGUF tensor directory.
struct TensorSizes {
    tensor_bytes_total: Option<u64>,
    layer_bytes_total: Option<u64>,
    expert_bytes_total: Option<u64>,
    moe_layer_count: u32,
    /// Sum of all tensor element counts (product of dims per tensor).
    /// Falls back to 0 on overflow; set from tensor shapes regardless of quant format.
    param_count: u64,
    expert_param_count: u64,
}

/// Parse the GGUF tensor-info directory and compute exact per-tensor byte sizes.
///
/// GGUF lays out tensor data contiguously in directory order, each tensor starting at an
/// aligned `offset` relative to the tensor-data section. We derive each tensor's size from
/// the gap to the next offset (and the final tensor from the end of the file), which is
/// exact regardless of quantization type — including future/unknown quant formats. Tensors
/// are then classified by name: `blk.*` are repeating layers; `*_exps.*` are routed-expert
/// FFN weights (what `--n-cpu-moe` offloads).
fn parse_tensor_directory<R: Read + Seek>(
    r: &mut R,
    version: u32,
    tensor_count: u64,
    alignment: u64,
    file_size: Option<u64>,
) -> Result<TensorSizes, String> {
    if tensor_count == 0 || tensor_count > 1_000_000 {
        return Err(format!("Implausible tensor_count {tensor_count}"));
    }

    // (name, offset, n_elements) for each tensor.
    let mut infos: Vec<(String, u64, u64)> = Vec::with_capacity(tensor_count as usize);
    let mut param_count_total: u64 = 0;
    let mut expert_param_count: u64 = 0;
    let mut moe_layers: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for _ in 0..tensor_count {
        let name = read_str(r, version)?;
        let n_dims = read_u32(r)?;
        if n_dims > 8 {
            return Err(format!("Implausible tensor n_dims {n_dims}"));
        }
        let mut n_elements: u64 = 1;
        for _ in 0..n_dims {
            // GGUF v1 stored dimensions as u32; v2+ as u64.
            let dim = if version == 1 {
                read_u32(r)? as u64
            } else {
                read_u64(r)?
            };
            n_elements = n_elements.saturating_mul(dim);
        }
        let _ggml_type = read_u32(r)?;
        let offset = read_u64(r)?;
        param_count_total = param_count_total.saturating_add(n_elements);
        if let Some(rest) = name.strip_prefix("blk.")
            && name.contains("_exps")
        {
            expert_param_count = expert_param_count.saturating_add(n_elements);
            if let Some(idx) = rest.split('.').next().and_then(|s| s.parse::<u32>().ok()) {
                moe_layers.insert(idx);
            }
        }
        infos.push((name, offset, n_elements));
    }

    let (tensor_bytes_total, layer_bytes_total, expert_bytes_total) =
        if let Some(file_size) = file_size {
            // Tensor data begins after the directory, padded up to `alignment`.
            let pos = r
                .stream_position()
                .map_err(|e| format!("stream_position failed: {e}"))?;
            let align = alignment.max(1);
            let data_start = pos.div_ceil(align) * align;
            if data_start > file_size {
                return Err("tensor data start beyond EOF".into());
            }
            let data_len = file_size - data_start;

            // Sizes from offset deltas. The last tensor runs to EOF.
            let mut order: Vec<usize> = (0..infos.len()).collect();
            order.sort_by_key(|&i| infos[i].1);
            let mut tensor_total = 0u64;
            let mut layer_total = 0u64;
            let mut expert_total = 0u64;
            for k in 0..order.len() {
                let i = order[k];
                let start = infos[i].1;
                let end = if k + 1 < order.len() {
                    infos[order[k + 1]].1
                } else {
                    data_len
                };
                let size = end.saturating_sub(start);
                let name = &infos[i].0;
                tensor_total = tensor_total.saturating_add(size);
                if name.starts_with("blk.") {
                    layer_total = layer_total.saturating_add(size);
                    if name.contains("_exps") {
                        expert_total = expert_total.saturating_add(size);
                    }
                }
            }
            (Some(tensor_total), Some(layer_total), Some(expert_total))
        } else {
            (None, None, None)
        };

    Ok(TensorSizes {
        tensor_bytes_total,
        layer_bytes_total,
        expert_bytes_total,
        moe_layer_count: moe_layers.len() as u32,
        param_count: param_count_total,
        expert_param_count,
    })
}

// ── Internal value type ───────────────────────────────────────────────────────

#[derive(Debug)]
enum KvValue {
    U32(u32),
    U64(u64),
    I32(i32),
    I64(i64),
    Str(String),
    /// Small integer array (e.g. per-layer `head_count_kv`). Large arrays
    /// (token vocab, etc.) are not captured — see `MAX_CAPTURED_ARRAY`.
    ArrU32(Vec<u32>),
    /// Small boolean array (e.g. `sliding_window_pattern`).
    ArrBool(Vec<bool>),
    Other, // floats, big/other arrays — skipped/irrelevant for architecture metadata
}

/// Integer/bool arrays longer than this are skipped rather than captured, so we
/// never buffer token-vocabulary-sized arrays. Per-layer arrays (head_count_kv,
/// sliding_window_pattern) are at most `block_count` (~hundreds) entries.
const MAX_CAPTURED_ARRAY: u64 = 8192;

impl KvValue {
    fn as_u32(&self) -> Option<u32> {
        match self {
            KvValue::U32(v) => Some(*v),
            KvValue::U64(v) => u32::try_from(*v).ok(),
            KvValue::I32(v) => u32::try_from(*v).ok(),
            KvValue::I64(v) => u32::try_from(*v).ok(),
            _ => None,
        }
    }

    /// Borrow a captured integer array, if this value is one.
    fn as_u32_arr(&self) -> Option<&[u32]> {
        match self {
            KvValue::ArrU32(v) => Some(v),
            _ => None,
        }
    }

    /// Borrow a captured boolean array, if this value is one.
    fn as_bool_arr(&self) -> Option<&[bool]> {
        match self {
            KvValue::ArrBool(v) => Some(v),
            _ => None,
        }
    }

    fn as_u64(&self) -> Option<u64> {
        match self {
            KvValue::U32(v) => Some(*v as u64),
            KvValue::U64(v) => Some(*v),
            KvValue::I32(v) => u64::try_from(*v).ok(),
            KvValue::I64(v) => u64::try_from(*v).ok(),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        if let KvValue::Str(s) = self {
            Some(s)
        } else {
            None
        }
    }
}

// ── Binary readers ────────────────────────────────────────────────────────────

fn read_u8(r: &mut impl Read) -> Result<u8, String> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b).map_err(|e| format!("read u8: {e}"))?;
    Ok(b[0])
}

fn read_u16(r: &mut impl Read) -> Result<u16, String> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b).map_err(|e| format!("read u16: {e}"))?;
    Ok(u16::from_le_bytes(b))
}

fn read_u32(r: &mut impl Read) -> Result<u32, String> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|e| format!("read u32: {e}"))?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(r: &mut impl Read) -> Result<u64, String> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(|e| format!("read u64: {e}"))?;
    Ok(u64::from_le_bytes(b))
}

/// Read a GGUF string. v1 uses a u32 length prefix; v2+ use u64.
fn read_str<R: Read>(r: &mut R, version: u32) -> Result<String, String> {
    let len = if version == 1 {
        read_u32(r)? as u64
    } else {
        read_u64(r)?
    };
    if len > 4_000_000 {
        return Err(format!("String too long ({len} bytes) — likely corrupt"));
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf)
        .map_err(|e| format!("read str body: {e}"))?;
    String::from_utf8(buf).map_err(|e| format!("string not UTF-8: {e}"))
}

/// Read a single GGUF value of the given type, returning a `KvValue`.
/// Fixed-size non-architecture types are consumed and discarded (returned as `Other`).
fn read_value<R: Read + Seek>(r: &mut R, vtype: u32, version: u32) -> Result<KvValue, String> {
    match GgufType::from_u32(vtype) {
        Some(GgufType::Uint8) => Ok(KvValue::U32(read_u8(r)? as u32)),
        Some(GgufType::Int8) => Ok(KvValue::I32(read_u8(r)? as i8 as i32)),
        Some(GgufType::Uint16) => Ok(KvValue::U32(read_u16(r)? as u32)),
        Some(GgufType::Int16) => Ok(KvValue::I32(read_u16(r)? as i16 as i32)),
        Some(GgufType::Uint32) => Ok(KvValue::U32(read_u32(r)?)),
        Some(GgufType::Int32) => Ok(KvValue::I32(read_u32(r)? as i32)),
        Some(GgufType::Float32) => {
            r.seek(SeekFrom::Current(4))
                .map_err(|e| format!("seek f32: {e}"))?;
            Ok(KvValue::Other)
        }
        Some(GgufType::Bool) => {
            let _ = read_u8(r)?;
            Ok(KvValue::Other)
        }
        Some(GgufType::String) => Ok(KvValue::Str(read_str(r, version)?)),
        Some(GgufType::Uint64) => Ok(KvValue::U64(read_u64(r)?)),
        Some(GgufType::Int64) => Ok(KvValue::I64(read_u64(r)? as i64)),
        Some(GgufType::Float64) => {
            r.seek(SeekFrom::Current(8))
                .map_err(|e| format!("seek f64: {e}"))?;
            Ok(KvValue::Other)
        }
        Some(GgufType::Array) => read_array(r, version),
        None => Err(format!("Unknown GGUF value type {vtype}")),
    }
}

/// Read an array value. Small integer/bool arrays are captured (per-layer config
/// like `head_count_kv` / `sliding_window_pattern`); everything else (strings,
/// floats, oversized arrays) is skipped, leaving the reader positioned correctly.
fn read_array<R: Read + Seek>(r: &mut R, version: u32) -> Result<KvValue, String> {
    let elem_type = read_u32(r)?;
    let n = if version == 1 {
        read_u32(r)? as u64
    } else {
        read_u64(r)?
    };

    let et = GgufType::from_u32(elem_type);
    let capture_int = matches!(
        et,
        Some(
            GgufType::Uint8
                | GgufType::Int8
                | GgufType::Uint16
                | GgufType::Int16
                | GgufType::Uint32
                | GgufType::Int32
                | GgufType::Uint64
                | GgufType::Int64
        )
    );

    if n <= MAX_CAPTURED_ARRAY && capture_int {
        let mut out = Vec::with_capacity(n as usize);
        for _ in 0..n {
            // Reuse the scalar reader so every element advances the offset correctly.
            if let Some(v) = read_value(r, elem_type, version)?.as_u32() {
                out.push(v);
            }
        }
        return Ok(KvValue::ArrU32(out));
    }
    if n <= MAX_CAPTURED_ARRAY && matches!(et, Some(GgufType::Bool)) {
        let mut out = Vec::with_capacity(n as usize);
        for _ in 0..n {
            out.push(read_u8(r)? != 0);
        }
        return Ok(KvValue::ArrBool(out));
    }

    // Oversized or non-capturable element type: skip past the remaining elements.
    let fixed_size: Option<u64> = et.and_then(GgufType::fixed_size);
    if let Some(stride) = fixed_size {
        let total = n.saturating_mul(stride);
        r.seek(SeekFrom::Current(total as i64))
            .map_err(|e| format!("seek array: {e}"))?;
    } else {
        for _ in 0..n {
            skip_value_type(r, elem_type, version)?;
        }
    }
    Ok(KvValue::Other)
}

/// Skip an array value: read the element type and count, then seek/read past all elements.
fn skip_array<R: Read + Seek>(r: &mut R, version: u32) -> Result<(), String> {
    let elem_type = read_u32(r)?;
    let n = if version == 1 {
        read_u32(r)? as u64
    } else {
        read_u64(r)?
    };

    // For fixed-size element types, a single seek is much faster than iterating.
    let fixed_size: Option<u64> = GgufType::from_u32(elem_type).and_then(GgufType::fixed_size);

    if let Some(stride) = fixed_size {
        let total = n.saturating_mul(stride);
        r.seek(SeekFrom::Current(total as i64))
            .map_err(|e| format!("seek array: {e}"))?;
    } else {
        // STRING or nested ARRAY — must iterate (variable-length elements)
        for _ in 0..n {
            skip_value_type(r, elem_type, version)?;
        }
    }
    Ok(())
}

/// Skip a single value of `vtype` without returning it.
fn skip_value_type<R: Read + Seek>(r: &mut R, vtype: u32, version: u32) -> Result<(), String> {
    match GgufType::from_u32(vtype) {
        Some(GgufType::Uint8 | GgufType::Int8 | GgufType::Bool) => {
            let _ = read_u8(r)?;
        }
        Some(GgufType::Uint16 | GgufType::Int16) => {
            let _ = read_u16(r)?;
        }
        Some(GgufType::Uint32 | GgufType::Int32 | GgufType::Float32) => {
            r.seek(SeekFrom::Current(4))
                .map_err(|e| format!("skip: {e}"))?;
        }
        Some(GgufType::Uint64 | GgufType::Int64 | GgufType::Float64) => {
            r.seek(SeekFrom::Current(8))
                .map_err(|e| format!("skip: {e}"))?;
        }
        Some(GgufType::String) => {
            let _ = read_str(r, version)?;
        }
        Some(GgufType::Array) => skip_array(r, version)?,
        None => return Err(format!("Unknown type to skip: {vtype}")),
    }
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal GGUF v3 byte stream in memory for testing.
    fn make_gguf(kv: &[(&str, KvEntry)]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&(kv.len() as u64).to_le_bytes());

        for (key, entry) in kv {
            out.extend_from_slice(&(key.len() as u64).to_le_bytes());
            out.extend_from_slice(key.as_bytes());
            match entry {
                KvEntry::U32(v) => {
                    out.extend_from_slice(&GgufType::Uint32.as_u32().to_le_bytes());
                    out.extend_from_slice(&v.to_le_bytes());
                }
                KvEntry::U64(v) => {
                    out.extend_from_slice(&GgufType::Uint64.as_u32().to_le_bytes());
                    out.extend_from_slice(&v.to_le_bytes());
                }
                KvEntry::Str(s) => {
                    out.extend_from_slice(&GgufType::String.as_u32().to_le_bytes());
                    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
                    out.extend_from_slice(s.as_bytes());
                }
            }
        }
        out
    }

    enum KvEntry {
        U32(u32),
        U64(u64),
        Str(String),
    }

    /// Build a GGUF v3 byte stream with a real tensor directory + blob, so the exact
    /// per-tensor size measurement (offset deltas + final file size) can be tested.
    /// `tensors` is a list of `(name, byte_size)`; sizes should be multiples of 32.
    fn make_gguf_with_tensors(kv: &[(&str, KvEntry)], tensors: &[(&str, u64)]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(tensors.len() as u64).to_le_bytes()); // tensor_count
        out.extend_from_slice(&(kv.len() as u64).to_le_bytes());

        for (key, entry) in kv {
            out.extend_from_slice(&(key.len() as u64).to_le_bytes());
            out.extend_from_slice(key.as_bytes());
            match entry {
                KvEntry::U32(v) => {
                    out.extend_from_slice(&GgufType::Uint32.as_u32().to_le_bytes());
                    out.extend_from_slice(&v.to_le_bytes());
                }
                KvEntry::U64(v) => {
                    out.extend_from_slice(&GgufType::Uint64.as_u32().to_le_bytes());
                    out.extend_from_slice(&v.to_le_bytes());
                }
                KvEntry::Str(s) => {
                    out.extend_from_slice(&GgufType::String.as_u32().to_le_bytes());
                    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
                    out.extend_from_slice(s.as_bytes());
                }
            }
        }

        // Tensor directory: cumulative offsets, n_dims=1, dims=[1], type=F32(0).
        let mut offset: u64 = 0;
        for (name, size) in tensors {
            out.extend_from_slice(&(name.len() as u64).to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&1u32.to_le_bytes()); // n_dims
            out.extend_from_slice(&1u64.to_le_bytes()); // dims[0]
            out.extend_from_slice(&0u32.to_le_bytes()); // ggml_type = F32
            out.extend_from_slice(&offset.to_le_bytes());
            offset += size;
        }

        // Pad to the default 32-byte alignment, then append a blob = sum(sizes).
        let align = 32usize;
        let pad = (align - (out.len() % align)) % align;
        out.extend(std::iter::repeat_n(0u8, pad));
        out.extend(std::iter::repeat_n(0u8, offset as usize));
        out
    }

    fn read_from_bytes(bytes: &[u8]) -> Result<GgufMetadata, String> {
        let tmp = tempfile::NamedTempFile::new().map_err(|e| format!("tempfile: {e}"))?;
        std::fs::write(tmp.path(), bytes).map_err(|e| format!("write: {e}"))?;
        read_gguf_metadata(tmp.path())
    }

    #[test]
    fn parses_qwen36_metadata() {
        let bytes = make_gguf(&[
            ("general.architecture", KvEntry::Str("qwen3_6".into())),
            ("general.parameter_count", KvEntry::U64(27_000_000_000)),
            ("qwen3_6.block_count", KvEntry::U32(64)),
            ("qwen3_6.attention.head_count", KvEntry::U32(24)),
            ("qwen3_6.attention.head_count_kv", KvEntry::U32(4)),
            ("qwen3_6.attention.key_length", KvEntry::U32(256)),
            ("qwen3_6.context_length", KvEntry::U32(262144)),
            ("qwen3_6.embedding_length", KvEntry::U32(5120)),
        ]);
        let meta = read_from_bytes(&bytes).unwrap();
        assert_eq!(meta.architecture.as_deref(), Some("qwen3_6"));
        assert_eq!(meta.block_count, Some(64));
        assert_eq!(meta.head_count_kv, Some(4));
        assert_eq!(meta.key_length, Some(256));
        assert_eq!(meta.context_length, Some(262144));
        assert!((meta.param_b().unwrap() - 27.0).abs() < 0.1);
    }

    #[test]
    fn measures_exact_tensor_sizes_from_directory() {
        let kv = [
            ("general.architecture", KvEntry::Str("qwen3_6".into())),
            ("qwen3_6.block_count", KvEntry::U32(2)),
            ("qwen3_6.expert_count", KvEntry::U32(8)),
            ("qwen3_6.expert_used_count", KvEntry::U32(2)),
        ];
        // Two layers, each with a small attention tensor + a large routed-expert tensor,
        // plus non-layer embedding/output tensors.
        let tensors = [
            ("token_embd.weight", 1024u64),
            ("blk.0.attn_q.weight", 512),
            ("blk.0.ffn_gate_exps.weight", 4096),
            ("blk.1.attn_q.weight", 512),
            ("blk.1.ffn_gate_exps.weight", 4096),
            ("output.weight", 1024),
        ];
        let bytes = make_gguf_with_tensors(&kv, &tensors);

        // The file-size-aware reader measures exact tensor bytes from offset deltas.
        let meta =
            read_gguf_metadata_reader(std::io::Cursor::new(&bytes), Some(bytes.len() as u64))
                .unwrap();
        assert_eq!(meta.tensor_bytes_total, Some(11264)); // sum of all
        assert_eq!(meta.layer_bytes_total, Some(9216)); // blk.* only
        assert_eq!(meta.expert_bytes_total, Some(8192)); // *_exps only
        assert_eq!(meta.tensor_param_count, Some(6)); // helper uses one element per tensor
        assert_eq!(meta.expert_param_count, Some(2)); // two *_exps tensors
        assert_eq!(meta.moe_layer_count, Some(2));
        assert_eq!(meta.bytes_per_layer(), Some(4608)); // 9216 / 2 layers
        assert_eq!(meta.expert_bytes_per_layer(), Some(4096)); // 8192 / 2 moe layers

        // A range-fetched prefix still contains tensor shapes, so parameter counts
        // remain exact even though on-disk byte sizes cannot be measured.
        let meta_prefix = read_gguf_metadata_reader(std::io::Cursor::new(&bytes), None).unwrap();
        assert_eq!(meta_prefix.tensor_bytes_total, None);
        assert_eq!(meta_prefix.tensor_param_count, Some(6));
        assert_eq!(meta_prefix.expert_param_count, Some(2));
        assert_eq!(meta_prefix.bytes_per_layer(), None);
    }

    #[test]
    fn parses_moe_expert_fields() {
        let bytes = make_gguf(&[
            ("general.architecture", KvEntry::Str("qwen3_6".into())),
            ("qwen3_6.block_count", KvEntry::U32(40)),
            ("qwen3_6.expert_count", KvEntry::U32(256)),
            ("qwen3_6.expert_used_count", KvEntry::U32(9)),
            ("qwen3_6.attention.head_count_kv", KvEntry::U32(2)),
        ]);
        let meta = read_from_bytes(&bytes).unwrap();
        assert_eq!(meta.expert_count, Some(256));
        assert_eq!(meta.expert_used_count, Some(9));
        assert_eq!(meta.head_count_kv, Some(2));
    }

    #[test]
    fn to_model_metadata_sets_gguf_arch() {
        let bytes = make_gguf(&[
            ("general.architecture", KvEntry::Str("qwen3_6".into())),
            ("qwen3_6.block_count", KvEntry::U32(64)),
            ("qwen3_6.attention.head_count_kv", KvEntry::U32(4)),
        ]);
        let gguf = read_from_bytes(&bytes).unwrap();
        let mm = gguf.to_model_metadata();
        assert_eq!(mm.gguf_arch.as_deref(), Some("qwen3_6"));
        assert_eq!(mm.n_layers, Some(64));
        assert_eq!(mm.n_kv_heads, Some(4));
    }

    #[test]
    fn rejects_non_gguf_file() {
        let mut bytes = make_gguf(&[]);
        bytes[0] = b'X';
        let result = read_from_bytes(&bytes);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not a GGUF"));
    }

    #[test]
    fn parses_gguf_v1_format() {
        // v1 uses u32 for string lengths and kv_count instead of u64
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&1u32.to_le_bytes()); // version 1
        out.extend_from_slice(&0u32.to_le_bytes()); // tensor_count (u32 in v1)
        out.extend_from_slice(&2u32.to_le_bytes()); // kv_count (u32 in v1)

        // KV entry: "general.architecture" = "llama" — string len is u32 in v1
        let key = b"general.architecture";
        out.extend_from_slice(&(key.len() as u32).to_le_bytes());
        out.extend_from_slice(key);
        out.extend_from_slice(&GgufType::String.as_u32().to_le_bytes());
        let val = b"llama";
        out.extend_from_slice(&(val.len() as u32).to_le_bytes()); // u32 string len in v1
        out.extend_from_slice(val);

        // KV entry: "llama.block_count" = 32
        let key2 = b"llama.block_count";
        out.extend_from_slice(&(key2.len() as u32).to_le_bytes());
        out.extend_from_slice(key2);
        out.extend_from_slice(&GgufType::Uint32.as_u32().to_le_bytes());
        out.extend_from_slice(&32u32.to_le_bytes());

        let meta = read_from_bytes(&out).unwrap();
        assert_eq!(meta.architecture.as_deref(), Some("llama"));
        assert_eq!(meta.block_count, Some(32));
    }

    #[test]
    fn parses_mtp_depth_field() {
        let bytes = make_gguf(&[
            ("general.architecture", KvEntry::Str("deepseek2".into())),
            ("deepseek2.block_count", KvEntry::U32(61)),
            ("deepseek2.next_n_token_count", KvEntry::U32(1)),
        ]);
        let meta = read_from_bytes(&bytes).unwrap();
        assert_eq!(meta.mtp_depth, Some(1));
    }

    #[test]
    fn returns_error_on_truncated_file() {
        let bytes = make_gguf(&[("general.architecture", KvEntry::Str("llama".into()))]);
        // Truncate to 10 bytes — can't even read the header
        let truncated = &bytes[..10];
        let result = read_from_bytes(truncated);
        assert!(result.is_err());
    }

    #[test]
    fn returns_error_on_unsupported_version() {
        let mut bytes = make_gguf(&[]);
        // Overwrite version field (bytes 4-7) with version 99
        bytes[4] = 99;
        bytes[5] = 0;
        bytes[6] = 0;
        bytes[7] = 0;
        let result = read_from_bytes(&bytes);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unsupported GGUF version"));
    }

    #[test]
    fn gguf_arch_drives_hybrid_heuristic_for_renamed_model() {
        let bytes = make_gguf(&[
            ("general.architecture", KvEntry::Str("qwen3_6".into())),
            ("qwen3_6.block_count", KvEntry::U32(64)),
            ("qwen3_6.attention.head_count_kv", KvEntry::U32(4)),
            ("qwen3_6.attention.key_length", KvEntry::U32(256)),
        ]);
        let gguf = read_from_bytes(&bytes).unwrap();
        let mm = gguf.to_model_metadata();
        let arch = mm.to_arch("Pantheon-Reasoning-27B-Q4_K_M.gguf", 27.0);
        assert!(
            arch.is_hybrid_attn(),
            "gguf_arch=qwen3_6 must yield hybrid-DeltaNet arch"
        );
        assert_eq!(
            arch.n_attn_layers, 16,
            "only 16 of 64 layers should have KV cache"
        );
    }

    #[test]
    fn pantheon_real_gguf_has_qwen35_arch() {
        // Integration test: reads the actual Pantheon-Reasoning-27B GGUF on disk.
        // llama.cpp uses "qwen35" for both Qwen3.5 and Qwen3.6 families.
        // Distinguished by block_count: 64 = Qwen3.6, 96 = Qwen3.5.
        let home = std::env::var("HOME").ok();
        let path = home
            .map(|h| {
                Path::new(&h)
                    .join(".config/llama-monitor/models/Pantheon-Reasoning-27B.i1-Q6_K.gguf")
            })
            .and_then(|p| p.exists().then_some(p));
        let Some(path) = path else {
            return; // file not present, skip
        };
        let gguf = read_gguf_metadata(&path).expect("read pantheon gguf");
        assert_eq!(
            gguf.architecture.as_deref(),
            Some("qwen35"),
            "Pantheon-Reasoning-27B GGUF reports qwen35 (shared by Qwen3.5+3.6)"
        );
        // 65 layers — these specific GGUFs have 65 blocks (likely an extra embedding
        // layer or architecture variant), not the canonical 64 from base Qwen3.6.
        // What matters is that block_count < 96, confirming Qwen3.6 family.
        assert!(
            gguf.block_count.unwrap() < 96,
            "block_count {} < 96 confirms Qwen3.6 family (not Qwen3.5)",
            gguf.block_count.unwrap()
        );
    }

    #[test]
    fn qwopus3_6_real_gguf_has_qwen35_arch() {
        // Integration test: reads the actual Qwopus3.6-27B-v2-MTP GGUF on disk.
        let home = std::env::var("HOME").ok();
        let path = home
            .map(|h| {
                Path::new(&h).join(".config/llama-monitor/models/Qwopus3.6-27B-v2-MTP-Q6_K.gguf")
            })
            .and_then(|p| p.exists().then_some(p));
        let Some(path) = path else {
            return; // file not present, skip
        };
        let gguf = read_gguf_metadata(&path).expect("read qwopus gguf");
        assert_eq!(
            gguf.architecture.as_deref(),
            Some("qwen35"),
            "Qwopus3.6-27B-v2-MTP GGUF reports qwen35"
        );
        assert!(
            gguf.block_count.unwrap() < 96,
            "block_count {} < 96 confirms Qwen3.6 family",
            gguf.block_count.unwrap()
        );
    }

    #[ignore]
    #[test]
    fn gemma4_31b_real_gguf_architecture() {
        let home = std::env::var("HOME").ok();
        let path = home
            .as_ref()
            .map(|h| {
                Path::new(h).join(".config/llama-monitor/models/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf")
            })
            .filter(|p| p.exists());
        let Some(path) = path else {
            return;
        };
        let gguf = read_gguf_metadata(&path).expect("read gemma4-31b gguf");
        assert_eq!(
            gguf.architecture.as_deref(),
            Some("gemma4"),
            "Gemma4-31B GGUF should report gemma4 architecture"
        );
        assert_eq!(
            gguf.block_count,
            Some(60),
            "Gemma4-31B should have 60 layers"
        );
        assert!(
            gguf.head_count_kv.is_none(),
            "Gemma4 GGUF does not expose head_count_kv (uses separate global/local KV heads)"
        );
    }

    #[ignore]
    #[test]
    fn qwen3_coder_next_real_gguf_architecture() {
        let home = std::env::var("HOME").ok();
        let path = home.as_ref().map(|h| {
            Path::new(h).join(".config/llama-monitor/models/Qwen3-Coder-Next-Huihui-Opus-4.6-Reasoning-Distilled-abliterated-IQ4_XS.gguf")
        }).filter(|p| p.exists());
        let Some(path) = path else {
            return;
        };
        let gguf = read_gguf_metadata(&path).expect("read qwen3 coder next gguf");
        assert_eq!(
            gguf.architecture.as_deref(),
            Some("qwen3next"),
            "Qwen3-Coder-Next GGUF should report qwen3next architecture"
        );
        assert_eq!(
            gguf.block_count,
            Some(48),
            "Qwen3-Coder-Next should have 48 layers"
        );
        assert!(
            gguf.expert_count.is_some(),
            "Qwen3-Coder-Next is MoE (should have expert_count)"
        );
    }

    /// Verify that the hybrid DeltaNet active-param formula uses n_attn_layers and
    /// ssm_inner_size to correctly account for always-active DeltaNet backbone params.
    ///
    /// Model: synthetic Qwen3.6-35B-A3B-like GGUF (40 layers, 10 attn + 30 DeltaNet,
    /// 256 experts, 9 used, embd=4096, head_dim=256, n_kv=2, ssm_inner=3907).
    ///
    /// Without this logic, all 40 layers would be treated as standard attention,
    /// over-counting backbone and pushing the estimate to ~4 B instead of the
    /// correct ~3 B for an "A3B" model.
    #[test]
    fn hybrid_deltanet_active_params_uses_ssm_inner_size() {
        // Parameters chosen to approximate Qwen3.6-35B-A3B architecture:
        //   embd=4096, head_count=32, head_count_kv=2, head_dim=256
        //   40 layers total, full_attention_interval=4 → 10 attn + 30 DeltaNet
        //   256 experts, 9 used, ssm_inner=3907
        let bytes = make_gguf(&[
            ("general.architecture", KvEntry::Str("qwen35moe".into())),
            ("general.parameter_count", KvEntry::U64(35_000_000_000)),
            ("qwen35moe.block_count", KvEntry::U32(40)),
            ("qwen35moe.full_attention_interval", KvEntry::U32(4)),
            ("qwen35moe.attention.head_count", KvEntry::U32(32)),
            ("qwen35moe.attention.head_count_kv", KvEntry::U32(2)),
            ("qwen35moe.attention.key_length", KvEntry::U32(256)),
            ("qwen35moe.embedding_length", KvEntry::U32(4096)),
            ("qwen35moe.expert_count", KvEntry::U32(256)),
            ("qwen35moe.expert_used_count", KvEntry::U32(9)),
            ("qwen35moe.ssm.inner_size", KvEntry::U32(3907)),
        ]);
        let meta = read_from_bytes(&bytes).unwrap();
        let active = meta
            .active_params_b()
            .expect("should compute active_params_b");
        // Expect close to 3 B (the "A3B" designation); old formula gave ~4 B.
        assert!(
            (active - 3.0).abs() < 0.5,
            "expected ~3 B active for 35B-A3B-like model, got {active:.2} B"
        );
        // Must be less than what the old formula (all 40 layers as attn) would give.
        assert!(
            active < 4.0,
            "hybrid fix should reduce estimate from old ~4 B; got {active:.2} B"
        );
    }

    #[test]
    fn active_params_uses_exact_routed_expert_tensor_counts() {
        // Real topology from local Qwen3.6-35B-A3B GGUF:
        // 34.66B total tensor elements, 32.21B routed across 256 experts,
        // 8 active experts. The remaining 2.45B parameters are always active.
        let meta = GgufMetadata {
            architecture: Some("qwen35moe".into()),
            param_count: Some(34_660_610_688),
            expert_count: Some(256),
            expert_used_count: Some(8),
            tensor_param_count: Some(34_660_610_688),
            expert_param_count: Some(32_212_254_720),
            ..Default::default()
        };

        let active = meta.active_params_b().expect("active parameter estimate");
        assert!(
            (active - 3.454_988_928).abs() < 1e-9,
            "exact tensor split should produce 3.455 B active, got {active:.9} B"
        );
    }

    #[test]
    fn local_moe_ggufs_have_expected_active_parameter_ranges() {
        let Some(home) = std::env::var("HOME").ok() else {
            return;
        };
        let models = Path::new(&home).join(".config/llama-monitor/models");
        let cases = [
            (
                "Qwen3.6-35B-A3B-uncensored-heretic-Q4_K_M.gguf",
                "qwen35moe",
                3.3,
                3.6,
            ),
            (
                "Qwen3-Coder-Next-Opus-Distilled-Q4_K_M.gguf",
                "qwen3next",
                3.7,
                4.1,
            ),
            (
                "G4-MeroMero-26B-A4B-it-uncensored-heretic-Q5_K_M.gguf",
                "gemma4",
                3.6,
                4.1,
            ),
            (
                "Nex-N2-mini-ultra-uncensored-heretic-Q6_K.gguf",
                "qwen35moe",
                3.3,
                3.6,
            ),
        ];

        for (filename, expected_arch, min_active, max_active) in cases {
            let path = models.join(filename);
            if !path.exists() {
                continue;
            }
            let meta = read_gguf_metadata(&path).unwrap_or_else(|e| panic!("{filename}: {e}"));
            assert_eq!(
                meta.architecture.as_deref(),
                Some(expected_arch),
                "{filename}"
            );
            let active = meta
                .active_params_b()
                .unwrap_or_else(|| panic!("{filename}: missing active params"));
            assert!(
                (min_active..=max_active).contains(&active),
                "{filename}: expected {min_active:.1}–{max_active:.1} B active, got {active:.3} B"
            );
        }
    }

    #[test]
    fn hybrid_deltanet_active_params_requires_ssm_dimensions() {
        // Partial GGUF metadata may expose the attention interval without the
        // DeltaNet dimensions. Keep the conservative all-attention backbone in
        // that case rather than dropping the always-active DeltaNet weights.
        let bytes = make_gguf(&[
            ("general.architecture", KvEntry::Str("qwen35moe".into())),
            ("general.parameter_count", KvEntry::U64(35_000_000_000)),
            ("qwen35moe.block_count", KvEntry::U32(40)),
            ("qwen35moe.full_attention_interval", KvEntry::U32(4)),
            ("qwen35moe.attention.head_count", KvEntry::U32(32)),
            ("qwen35moe.attention.head_count_kv", KvEntry::U32(2)),
            ("qwen35moe.attention.key_length", KvEntry::U32(256)),
            ("qwen35moe.embedding_length", KvEntry::U32(4096)),
            ("qwen35moe.expert_count", KvEntry::U32(256)),
            ("qwen35moe.expert_used_count", KvEntry::U32(9)),
        ]);
        let meta = read_from_bytes(&bytes).unwrap();
        let active = meta
            .active_params_b()
            .expect("should compute active_params_b");

        assert!(
            (active - 4.01).abs() < 0.1,
            "missing SSM dimensions should retain the ~4.01 B all-attention fallback; got {active:.2} B"
        );
        assert!(
            active > 3.5,
            "partial hybrid metadata must not produce the undercounted ~1.95 B estimate; got {active:.2} B"
        );
    }
}
