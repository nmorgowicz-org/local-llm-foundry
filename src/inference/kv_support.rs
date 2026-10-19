//! The only production policy function for main K/V cache type pairs.
//!
//! Replaces the old `MixedMainKv` hard block. A pair is classified from typed
//! evidence about the selected runtime, device and model:
//!
//! * `Native`     — expected fused attention path with adequate evidence.
//! * `Fallback`   — expected alternate path (still starts; may be slower).
//! * `Unverified` — insufficient evidence; never blocks a launch.
//! * `WontStart`  — a known upstream refusal; the only blocking class.
//!
//! Compiled pair availability is reported separately from expected execution:
//! a compiled pair is evidence of availability, not proof that a given
//! device/model executes it. See `docs/archive/llamacpp/20261019-llamacpp_compilation.md`
//! §4.6.1–§4.7.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

/// Version of the K/V support evidence/parser contract. Bump when the meaning
/// of persisted evidence changes; older cached evidence is then discarded.
pub const KV_BASIS_VERSION: u32 = 1;

/// First llama.cpp build at which the Metal flash-attention rule was checked:
/// `ggml_metal_device_supports_op` returns false for `FLASH_ATTN_EXT` when the
/// K and V types differ (source read at b11719), and mixed pairs measured on an
/// M5 Max at b11719 ran attention on the CPU. Later builds are assumed to keep
/// the rule (the failure mode is a stale warning, never a block) but are
/// reported as not re-tested. Builds below this are `Unverified`.
pub const METAL_FA_RULE_FROM_BUILD: u32 = 11719;

/// The build that was actually measured and read. Builds above it carry a
/// "not re-tested" qualifier in user text.
pub const METAL_FA_RULE_MEASURED_BUILD: u32 = 11719;

/// Escape hatch: set to the first build that ships mixed K/V Metal flash
/// attention kernels once a changelog or re-test shows it. Builds from there on
/// go back to `Unverified`. Re-check with
/// `grep -n 'src\[1\]->type != op->src\[2\]->type' ggml/src/ggml-metal/ggml-metal-device.m`.
pub const METAL_FA_MIXED_KERNELS_FROM_BUILD: Option<u32> = None;

/// Head dimensions for which the fused FlashAttention paths are considered
/// adequate. Deliberately conservative: an unlisted or unknown dimension means
/// the classification stays `Unverified`, never blocked.
const FA_ADEQUATE_HEAD_DIMS: &[u32] = &[64, 80, 96, 112, 128, 256];

/// Metal measurement context quoted wherever a Metal slowdown is mentioned.
pub const METAL_MIXED_MEASUREMENT_NOTE: &str = "On llama.cpp b11719 with an M5 Max and Qwen3.8-27B IQ4_XS (head_dim 256), a mixed quantized pair measured about 21x slower prefill and 3x slower decode than the equal pair because attention ran on the CPU. This was measured only for that device, model and version.";

// ---------------------------------------------------------------------------
// Cache types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvType {
    F32,
    F16,
    Bf16,
    Q8_0,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Iq4Nl,
}

/// Cache types for which llama.cpp builds fused FlashAttention vector kernels
/// (`FA_TYPES` in `ggml/cmake/common.cmake`).
pub const FA_TYPES: [KvType; 7] = [
    KvType::Q4_0,
    KvType::Q4_1,
    KvType::Q5_0,
    KvType::Q5_1,
    KvType::Q8_0,
    KvType::Bf16,
    KvType::F16,
];

impl KvType {
    /// Parse a cache type name case-insensitively. Unknown names return
    /// `None`; syntax validation of those names stays with the existing
    /// validators.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "f32" => Some(Self::F32),
            "f16" => Some(Self::F16),
            "bf16" => Some(Self::Bf16),
            "q8_0" => Some(Self::Q8_0),
            "q4_0" => Some(Self::Q4_0),
            "q4_1" => Some(Self::Q4_1),
            "q5_0" => Some(Self::Q5_0),
            "q5_1" => Some(Self::Q5_1),
            "iq4_nl" => Some(Self::Iq4Nl),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::Q8_0 => "q8_0",
            Self::Q4_0 => "q4_0",
            Self::Q4_1 => "q4_1",
            Self::Q5_0 => "q5_0",
            Self::Q5_1 => "q5_1",
            Self::Iq4Nl => "iq4_nl",
        }
    }

    pub fn is_floating(self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::Bf16)
    }

    pub fn is_quantized(self) -> bool {
        !self.is_floating()
    }

    /// Whether CUDA can build a fused FlashAttention vector kernel for this type.
    pub fn is_fa_buildable(self) -> bool {
        FA_TYPES.contains(&self)
    }

    /// Whether the Metal FlashAttention kernels cover this type (equal pairs
    /// only). `iq4_nl` has no Metal FA kernel; f32 does.
    fn is_metal_fa_type(self) -> bool {
        self == Self::F32 || self.is_fa_buildable()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FlashAttnMode {
    Off,
    On,
    Auto,
    /// The effective mode could not be resolved (e.g. no stored value and an
    /// unknown server default).
    Unknown,
}

impl FlashAttnMode {
    /// Parse `-fa` values as accepted by llama-server (`on`/`off`/`auto`,
    /// plus legacy boolean spellings). Empty means "not specified".
    pub fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "on" | "true" | "1" | "enabled" => Self::On,
            "off" | "false" | "0" | "disabled" => Self::Off,
            "auto" => Self::Auto,
            _ => Self::Unknown,
        }
    }
}

/// The mode a product launch actually requests: `LlamaCppAdapter` passes
/// `-fa off` only when the preset says exactly `off`; everything else
/// (empty, `auto`, `on`, legacy spellings) launches with `-fa on`.
pub fn launch_flash_attn_mode(preset_value: &str) -> FlashAttnMode {
    if preset_value == "off" {
        FlashAttnMode::Off
    } else {
        FlashAttnMode::On
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    Cuda,
    Metal,
    Cpu,
    Vulkan,
    #[default]
    Unknown,
}

// ---------------------------------------------------------------------------
// Typed evidence
// ---------------------------------------------------------------------------

/// A value together with how it was learned. Missing evidence is never
/// silently promoted to a confirmed value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum Evidence<T> {
    Confirmed { value: T, source: String },
    Assumed { value: T, reason: String },
    Unavailable { reason: String },
}

impl<T> Evidence<T> {
    pub fn confirmed(value: T, source: impl Into<String>) -> Self {
        Self::Confirmed {
            value,
            source: source.into(),
        }
    }

    pub fn assumed(value: T, reason: impl Into<String>) -> Self {
        Self::Assumed {
            value,
            reason: reason.into(),
        }
    }

    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self::Unavailable {
            reason: reason.into(),
        }
    }

    pub fn confirmed_value(&self) -> Option<&T> {
        match self {
            Self::Confirmed { value, .. } => Some(value),
            _ => None,
        }
    }

    pub fn is_confirmed(&self) -> bool {
        matches!(self, Self::Confirmed { .. })
    }
}

impl<T> Default for Evidence<T> {
    fn default() -> Self {
        Self::unavailable("not probed")
    }
}

/// Runtime evidence a classification is derived from. Stored on the
/// capability snapshot; bound to the loaded runtime identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvSupportBasis {
    pub version: u32,
    pub backend: Backend,
    /// Compiled CUDA FlashAttention pairs (from the loaded backend's
    /// `FA_QUANTS` feature record). Only meaningful when `backend == Cuda`.
    #[serde(default)]
    pub fa_pairs: Evidence<BTreeSet<(KvType, KvType)>>,
    /// Hash/path identity of the executable plus loaded backend libraries the
    /// evidence was read from.
    #[serde(default)]
    pub runtime_identity: Option<String>,
    /// llama.cpp build number the evidence was read from.
    #[serde(default)]
    pub build_number: Option<u32>,
    /// Metal `has bfloat` as reported by the device at startup. This is the
    /// flag the Metal flash-attention support check consults for bf16.
    #[serde(default)]
    pub metal_bf16: Evidence<bool>,
    /// Devices the loaded runtime enumerated (`"CUDA0 : NVIDIA ..."`). Detected
    /// device ids sent by clients are validated against this list.
    #[serde(default)]
    pub devices: Vec<String>,
}

impl KvSupportBasis {
    /// Basis with no evidence at all (unknown backend, nothing probed).
    pub fn unavailable(reason: &str) -> Self {
        Self {
            version: KV_BASIS_VERSION,
            backend: Backend::Unknown,
            fa_pairs: Evidence::unavailable(reason),
            runtime_identity: None,
            build_number: None,
            metal_bf16: Evidence::unavailable(reason),
            devices: Vec::new(),
        }
    }

    /// Whether the Metal "K and V types must match" rule applies to this
    /// runtime: a Metal build at or after the checked build and before any
    /// known mixed-kernel build.
    pub fn metal_rule_applies(&self) -> bool {
        metal_rule_applies_to(
            self.backend,
            self.build_number,
            METAL_FA_MIXED_KERNELS_FROM_BUILD,
        )
    }

    /// Qualifier for user text: empty for the measured build, otherwise says the
    /// rule was not re-tested on this one.
    pub fn metal_rule_qualifier(&self) -> &'static str {
        match self.build_number {
            Some(build) if build > METAL_FA_RULE_MEASURED_BUILD => {
                " This build has not been re-tested; the rule was checked on b11719."
            }
            _ => "",
        }
    }
}

fn metal_rule_applies_to(backend: Backend, build: Option<u32>, mixed_from: Option<u32>) -> bool {
    backend == Backend::Metal
        && build.is_some_and(|build| {
            build >= METAL_FA_RULE_FROM_BUILD && mixed_from.is_none_or(|from| build < from)
        })
}

impl Default for KvSupportBasis {
    fn default() -> Self {
        Self::unavailable("not probed")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceEvidence {
    /// Detected-device id chosen by the user or configured selection.
    pub id: String,
    /// Metal bf16 hardware support. A compile flag or an Apple Silicon label
    /// alone is not hardware evidence.
    #[serde(default)]
    pub metal_bf16: Evidence<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelEvidence {
    /// Opaque model metadata identity (including revision).
    pub metadata_identity: String,
    /// Per-head attention dimensions from GGUF metadata (more than one for
    /// models with different global/local head sizes). Empty when unknown.
    pub head_dims: Vec<u32>,
}

/// Session/context-specific runtime observations (log watchers, benchmarks).
/// Absence of an observation never upgrades evidence.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct KvObservations {
    /// Whether `-fa auto` selected FlashAttention for this runtime/device/model.
    #[serde(default)]
    pub fa_auto_selected: Evidence<bool>,
    /// Whether a mixed quantized pair ran attention on the CPU on Metal.
    #[serde(default)]
    pub metal_mixed_cpu_attention: Evidence<bool>,
}

pub struct KvContext<'a> {
    pub basis: &'a KvSupportBasis,
    pub device: Option<&'a DeviceEvidence>,
    pub model: Option<&'a ModelEvidence>,
    pub observations: Option<&'a KvObservations>,
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Impact {
    SlowerDecode,
    NoFlashAttention,
    CpuAttention,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum KvSupport {
    Native,
    Fallback { reason: String, impact: Impact },
    Unverified { reason: String },
    WontStart { reason: String, fix: String },
}

impl KvSupport {
    fn outcome_key(&self) -> (u8, Option<Impact>) {
        match self {
            Self::Native => (0, None),
            Self::Fallback { impact, .. } => (1, Some(*impact)),
            Self::Unverified { .. } => (2, None),
            Self::WontStart { .. } => (3, None),
        }
    }

    pub fn is_blocking(&self) -> bool {
        matches!(self, Self::WontStart { .. })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvClassification {
    pub k: KvType,
    pub v: KvType,
    pub flash_attn: FlashAttnMode,
    /// Whether the pair has a compiled fused kernel, independent of expected execution.
    pub compiled_pair: Evidence<bool>,
    pub expected: KvSupport,
    /// True only for a confirmed missing CUDA kernel that a source build with
    /// `GGML_CUDA_FA_QUANTS=all` would add. Drives the rebuild link.
    pub buildable_missing: bool,
}

impl KvClassification {
    pub fn is_blocking(&self) -> bool {
        self.expected.is_blocking()
    }

    pub fn reason(&self) -> Option<&str> {
        match &self.expected {
            KvSupport::Native => None,
            KvSupport::Fallback { reason, .. }
            | KvSupport::Unverified { reason }
            | KvSupport::WontStart { reason, .. } => Some(reason),
        }
    }

    pub fn fix(&self) -> Option<&str> {
        match &self.expected {
            KvSupport::WontStart { fix, .. } => Some(fix),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// FA_QUANTS parsing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FaQuantsParseError(pub String);

impl std::fmt::Display for FaQuantsParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for FaQuantsParseError {}

/// The spec every foundry CUDA source build passes (decision D9).
pub const FA_BUILD_SPEC: &str = "all";

/// Parse a `GGML_CUDA_FA_QUANTS` value the way `ggml/cmake/common.cmake` does.
///
/// `all` is the 7x7 product of [`FA_TYPES`]. Otherwise the value is a list
/// (comma or semicolon separated) of `<K>-<V>` pairs of FA types, to which
/// `f16-f16` is always added. Malformed or unknown input is an error, never a
/// guessed default.
pub fn compiled_fa_pairs(spec: &str) -> Result<BTreeSet<(KvType, KvType)>, FaQuantsParseError> {
    let spec = spec.trim().to_ascii_lowercase();
    if spec.is_empty() {
        return Err(FaQuantsParseError("FA_QUANTS is empty".into()));
    }
    let mut pairs = BTreeSet::new();
    if spec == "all" {
        for k in FA_TYPES {
            for v in FA_TYPES {
                pairs.insert((k, v));
            }
        }
        return Ok(pairs);
    }
    pairs.insert((KvType::F16, KvType::F16));
    for item in spec.split([',', ';']) {
        let item = item.trim();
        let Some((k, v)) = item.split_once('-') else {
            return Err(FaQuantsParseError(format!(
                "FA_QUANTS entry {item:?} is not \"all\" or a <K>-<V> pair"
            )));
        };
        let parse = |name: &str| {
            KvType::parse(name)
                .filter(|kind| kind.is_fa_buildable())
                .ok_or_else(|| {
                    FaQuantsParseError(format!(
                        "FA_QUANTS entry {item:?} has unknown or non-FA type {name:?}"
                    ))
                })
        };
        pairs.insert((parse(k)?, parse(v)?));
    }
    Ok(pairs)
}

/// Extract the `FA_QUANTS = <value>` record from `llama_print_system_info()`
/// output. Returns `None` when no complete record is present.
pub fn parse_fa_quants_record(output: &str) -> Option<String> {
    for line in output.lines() {
        let Some(index) = line.find("FA_QUANTS") else {
            continue;
        };
        let rest = line[index + "FA_QUANTS".len()..].trim_start();
        let Some(rest) = rest.strip_prefix('=') else {
            continue;
        };
        // The record ends at the next feature separator; the list may contain
        // commas/semicolons but never spaces or `|`.
        let value: String = rest
            .trim_start()
            .chars()
            .take_while(|c| !c.is_whitespace() && *c != '|')
            .collect();
        if !value.is_empty() {
            return Some(value);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Partial runtime confirmation from server logs
// ---------------------------------------------------------------------------

/// Something a running llama-server said about K/V execution. These are
/// session-specific, nonblocking observations: the absence of a line proves
/// nothing, and a line is never promoted to evidence for another model,
/// device or pair.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum KvLogObservation {
    /// CUDA converted K and V to f16 because no vector kernel was compiled.
    CudaConversionFallback { k: String, v: String },
    /// Flash attention was requested/auto but the runtime disabled it.
    FlashAttnDisabled,
}

pub fn observe_log_line(line: &str) -> Option<KvLogObservation> {
    const CONVERT: &str = "no FlashAttention vector kernel compiled for K/V types ";
    if let Some(index) = line.find(CONVERT) {
        let rest = &line[index + CONVERT.len()..];
        let pair = rest.split([',', ' ']).next().unwrap_or("");
        if let Some((k, v)) = pair.split_once('-')
            && let (Some(k), Some(v)) = (KvType::parse(k), KvType::parse(v))
        {
            return Some(KvLogObservation::CudaConversionFallback {
                k: k.name().into(),
                v: v.name().into(),
            });
        }
        return None;
    }
    if line.contains("flash_attn") && line.contains("not supported, set to disabled") {
        return Some(KvLogObservation::FlashAttnDisabled);
    }
    None
}

static SESSION_OBSERVATIONS: std::sync::LazyLock<std::sync::Mutex<Vec<KvLogObservation>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(Vec::new()));

const MAX_SESSION_OBSERVATIONS: usize = 32;

/// Record an observation from a log line, if it carries one. Bounded and
/// de-duplicated.
pub fn record_log_line(line: &str) {
    let Some(observation) = observe_log_line(line) else {
        return;
    };
    let mut store = SESSION_OBSERVATIONS.lock().unwrap();
    if !store.contains(&observation) && store.len() < MAX_SESSION_OBSERVATIONS {
        store.push(observation);
    }
}

/// Observations since the last [`reset_session_observations`].
pub fn session_observations() -> Vec<KvLogObservation> {
    SESSION_OBSERVATIONS.lock().unwrap().clone()
}

/// Call when a new llama-server session starts.
pub fn reset_session_observations() {
    SESSION_OBSERVATIONS.lock().unwrap().clear();
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

fn compiled_pair_evidence(k: KvType, v: KvType, ctx: &KvContext<'_>) -> Evidence<bool> {
    match ctx.basis.backend {
        Backend::Cuda => match &ctx.basis.fa_pairs {
            Evidence::Confirmed { value, source } => {
                Evidence::confirmed(value.contains(&(k, v)), source.clone())
            }
            Evidence::Assumed { value, reason } => {
                Evidence::assumed(value.contains(&(k, v)), reason.clone())
            }
            Evidence::Unavailable { reason } => Evidence::unavailable(reason.clone()),
        },
        Backend::Metal => Evidence::assumed(
            k == v && k.is_metal_fa_type(),
            "Metal FlashAttention covers equal pairs only (from llama.cpp source; not probed)",
        ),
        Backend::Cpu | Backend::Vulkan | Backend::Unknown => {
            Evidence::unavailable("no compiled-pair evidence for this backend")
        }
    }
}

fn device_model_adequate(ctx: &KvContext<'_>) -> Result<(), String> {
    let Some(device) = ctx.device else {
        return Err("the selected device is not identified".into());
    };
    if device.id.trim().is_empty() {
        return Err("the selected device is not identified".into());
    }
    let Some(model) = ctx.model else {
        return Err("no model metadata was provided".into());
    };
    if model.head_dims.is_empty() {
        return Err("the model head dimension is unknown".into());
    }
    match model
        .head_dims
        .iter()
        .find(|dim| !FA_ADEQUATE_HEAD_DIMS.contains(dim))
    {
        Some(dim) => Err(format!("model head dimension {dim} is not a verified size")),
        None => Ok(()),
    }
}

fn unverified(reason: impl Into<String>) -> KvSupport {
    KvSupport::Unverified {
        reason: reason.into(),
    }
}

/// Resolve the expected execution once FlashAttention is requested
/// (`on`, or `auto` with a quantized V).
fn expected_with_fa(
    k: KvType,
    v: KvType,
    ctx: &KvContext<'_>,
    compiled: &Evidence<bool>,
) -> KvSupport {
    match ctx.basis.backend {
        Backend::Cuda => match compiled {
            Evidence::Confirmed { value: true, .. } => match device_model_adequate(ctx) {
                Ok(()) => KvSupport::Native,
                Err(why) => unverified(format!(
                    "The {}/{} FlashAttention kernel is compiled in, but execution is unverified because {why}.",
                    k.name(),
                    v.name()
                )),
            },
            Evidence::Confirmed { value: false, .. } => KvSupport::Fallback {
                reason: format!(
                    "No FlashAttention vector kernel is compiled for {}/{}. llama.cpp converts K and V to f16 instead, which is slower. A source build with every pair compiled removes this.",
                    k.name(),
                    v.name()
                ),
                impact: Impact::SlowerDecode,
            },
            _ => unverified(
                "The compiled FlashAttention pair list could not be read from this runtime.",
            ),
        },
        Backend::Metal => {
            let equal_in_set = k == v && k.is_metal_fa_type();
            if equal_in_set {
                if k == KvType::Bf16 {
                    let bf16 = ctx
                        .device
                        .and_then(|d| d.metal_bf16.confirmed_value())
                        .or_else(|| ctx.basis.metal_bf16.confirmed_value());
                    if bf16 != Some(&true) {
                        return unverified(
                            "bf16 FlashAttention on Metal needs bf16 hardware support, which the runtime did not confirm for this device.",
                        );
                    }
                }
                return match device_model_adequate(ctx) {
                    Ok(()) => KvSupport::Native,
                    Err(why) => unverified(format!(
                        "Metal has a fused kernel for {0}/{0}, but execution is unverified because {why}.",
                        k.name()
                    )),
                };
            }
            let observed = ctx
                .observations
                .and_then(|o| o.metal_mixed_cpu_attention.confirmed_value());
            if ctx.basis.metal_rule_applies() {
                KvSupport::Fallback {
                    reason: format!(
                        "Metal flash attention supports equal K/V types only, so {}/{} attention runs on the CPU. {METAL_MIXED_MEASUREMENT_NOTE}{}",
                        k.name(),
                        v.name(),
                        ctx.basis.metal_rule_qualifier()
                    ),
                    impact: Impact::CpuAttention,
                }
            } else if observed == Some(&true) {
                KvSupport::Fallback {
                    reason: format!(
                        "Metal has no fused kernel for {}/{}; attention was observed running on the CPU. {METAL_MIXED_MEASUREMENT_NOTE}",
                        k.name(),
                        v.name()
                    ),
                    impact: Impact::CpuAttention,
                }
            } else {
                unverified(format!(
                    "Metal has no fused kernel for {}/{}; attention may run on the CPU and be much slower. {METAL_MIXED_MEASUREMENT_NOTE}",
                    k.name(),
                    v.name()
                ))
            }
        }
        Backend::Cpu => unverified("No CPU-specific evidence for this K/V pair."),
        Backend::Vulkan => unverified("No Vulkan-specific evidence for this K/V pair."),
        Backend::Unknown => unverified(
            "The inference backend could not be identified, so this K/V pair is unverified.",
        ),
    }
}

fn expected_for_mode(
    k: KvType,
    v: KvType,
    mode: FlashAttnMode,
    ctx: &KvContext<'_>,
    compiled: &Evidence<bool>,
) -> KvSupport {
    match mode {
        FlashAttnMode::Off => {
            if v.is_quantized() {
                KvSupport::WontStart {
                    reason: format!(
                        "llama.cpp refuses to start with a quantized V cache ({}) when flash attention is off.",
                        v.name()
                    ),
                    fix: "Set flash attention to on or auto, or use a floating V cache type (f16, bf16 or f32).".into(),
                }
            } else {
                KvSupport::Fallback {
                    reason: "Flash attention is off, so the fused K/V kernels are not used.".into(),
                    impact: Impact::NoFlashAttention,
                }
            }
        }
        FlashAttnMode::On => expected_with_fa(k, v, ctx, compiled),
        FlashAttnMode::Auto => {
            // A quantized V forces flash attention to be requested.
            if v.is_quantized() {
                return expected_with_fa(k, v, ctx, compiled);
            }
            // Metal with a verified rule: a pair without a fused kernel makes
            // auto disable flash attention (observed on q8_0/bf16 at b11719:
            // "Flash Attention is assigned to device CPU ... set to disabled").
            if ctx.basis.metal_rule_applies() && !(k == v && k.is_metal_fa_type()) {
                return KvSupport::Fallback {
                    reason: format!(
                        "Metal has no flash attention kernel for {}/{}, so -fa auto disables flash attention for this pair.",
                        k.name(),
                        v.name()
                    ),
                    impact: Impact::NoFlashAttention,
                };
            }
            match ctx
                .observations
                .and_then(|o| o.fa_auto_selected.confirmed_value())
            {
                Some(true) => expected_with_fa(k, v, ctx, compiled),
                Some(false) => KvSupport::Fallback {
                    reason:
                        "Flash attention auto-selection disabled flash attention for this pair."
                            .into(),
                    impact: Impact::NoFlashAttention,
                },
                None => unverified(
                    "Flash attention is set to auto and the runtime's choice is not confirmed for this pair.",
                ),
            }
        }
        FlashAttnMode::Unknown => unreachable!("resolved by the caller"),
    }
}

/// Classify a main K/V cache pair. This is the only function that decides
/// K/V policy; every caller (wizard, presets, bundles, calibration, Doctor,
/// launch) must use it.
pub fn classify_kv_pair(
    k: KvType,
    v: KvType,
    flash_attn: FlashAttnMode,
    ctx: &KvContext<'_>,
) -> KvClassification {
    let compiled_pair = compiled_pair_evidence(k, v, ctx);
    let expected = if flash_attn == FlashAttnMode::Unknown {
        let outcomes = [FlashAttnMode::Off, FlashAttnMode::On, FlashAttnMode::Auto]
            .map(|mode| expected_for_mode(k, v, mode, ctx, &compiled_pair));
        let first = outcomes[0].outcome_key();
        if outcomes.iter().all(|o| o.outcome_key() == first) {
            outcomes[0].clone()
        } else {
            unverified(
                "The effective flash attention mode is unknown and the possible modes behave differently for this K/V pair.",
            )
        }
    } else {
        expected_for_mode(k, v, flash_attn, ctx, &compiled_pair)
    };
    let buildable_missing = ctx.basis.backend == Backend::Cuda
        && k.is_fa_buildable()
        && v.is_fa_buildable()
        && matches!(compiled_pair, Evidence::Confirmed { value: false, .. })
        && matches!(
            expected,
            KvSupport::Fallback {
                impact: Impact::SlowerDecode,
                ..
            }
        );
    KvClassification {
        k,
        v,
        flash_attn,
        compiled_pair,
        expected,
        buildable_missing,
    }
}

/// Classify from string values as stored in presets. Returns `None` when
/// either value is empty ("server default") or not a known cache type; those
/// cases stay with the existing syntax validators.
pub fn classify_kv_strs(
    k: &str,
    v: &str,
    flash_attn: &str,
    ctx: &KvContext<'_>,
) -> Option<KvClassification> {
    if k.trim().is_empty() || v.trim().is_empty() {
        return None;
    }
    let k = KvType::parse(k)?;
    let v = KvType::parse(v)?;
    Some(classify_kv_pair(
        k,
        v,
        FlashAttnMode::parse(flash_attn),
        ctx,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_MODES: [FlashAttnMode; 3] =
        [FlashAttnMode::Off, FlashAttnMode::On, FlashAttnMode::Auto];

    fn cuda_basis(spec: &str) -> KvSupportBasis {
        KvSupportBasis {
            version: KV_BASIS_VERSION,
            backend: Backend::Cuda,
            fa_pairs: Evidence::confirmed(compiled_fa_pairs(spec).unwrap(), "probe"),
            runtime_identity: Some("test".into()),
            ..Default::default()
        }
    }

    fn backend_basis(backend: Backend) -> KvSupportBasis {
        KvSupportBasis {
            version: KV_BASIS_VERSION,
            backend,
            fa_pairs: Evidence::unavailable("not applicable"),
            runtime_identity: Some("test".into()),
            ..Default::default()
        }
    }

    fn device(bf16: Option<bool>) -> DeviceEvidence {
        DeviceEvidence {
            id: "dev0".into(),
            metal_bf16: bf16
                .map(|value| Evidence::confirmed(value, "device"))
                .unwrap_or_default(),
        }
    }

    fn model(head_dim: Option<u32>) -> ModelEvidence {
        ModelEvidence {
            metadata_identity: "m1".into(),
            head_dims: head_dim.into_iter().collect(),
        }
    }

    fn ctx<'a>(
        basis: &'a KvSupportBasis,
        device: Option<&'a DeviceEvidence>,
        model: Option<&'a ModelEvidence>,
    ) -> KvContext<'a> {
        KvContext {
            basis,
            device,
            model,
            observations: None,
        }
    }

    fn classify(k: KvType, v: KvType, fa: FlashAttnMode, ctx: &KvContext<'_>) -> KvSupport {
        classify_kv_pair(k, v, fa, ctx).expected
    }

    // -- types and parsing --------------------------------------------------

    #[test]
    fn kv_type_parse_is_case_insensitive_and_rejects_unknown() {
        assert_eq!(KvType::parse(" Q8_0 "), Some(KvType::Q8_0));
        assert_eq!(KvType::parse("IQ4_NL"), Some(KvType::Iq4Nl));
        assert_eq!(KvType::parse("q9_9"), None);
        assert!(KvType::F32.is_floating() && KvType::Bf16.is_floating());
        assert!(KvType::Iq4Nl.is_quantized() && !KvType::Iq4Nl.is_fa_buildable());
    }

    #[test]
    fn fa_quants_all_is_forty_nine_pairs() {
        let pairs = compiled_fa_pairs("all").unwrap();
        assert_eq!(pairs.len(), 49);
        assert!(pairs.contains(&(KvType::Q8_0, KvType::Q4_0)));
        assert!(pairs.contains(&(KvType::Q4_0, KvType::Q8_0)));
    }

    #[test]
    fn fa_quants_list_always_includes_f16_pair() {
        let pairs = compiled_fa_pairs("q8_0-q4_0").unwrap();
        assert_eq!(pairs.len(), 2);
        assert!(pairs.contains(&(KvType::F16, KvType::F16)));
        assert!(pairs.contains(&(KvType::Q8_0, KvType::Q4_0)));
        // The upstream default list as printed by the define (comma separated).
        let default = compiled_fa_pairs("q4_0-q4_0,q8_0-q8_0,f16-f16,bf16-bf16").unwrap();
        assert_eq!(default.len(), 4);
        // Semicolons are accepted for the raw CMake list spelling.
        assert_eq!(compiled_fa_pairs("q4_0-q4_0;q8_0-q8_0").unwrap().len(), 3);
    }

    #[test]
    fn fa_quants_rejects_malformed_and_unknown() {
        for bad in [
            "",
            "  ",
            "q8_0",
            "q8_0-",
            "q8_0-zzz",
            "q8_0-iq4_nl",
            "all,q8_0-q8_0",
            "q8_0-q4_0,",
        ] {
            assert!(
                compiled_fa_pairs(bad).is_err(),
                "{bad:?} should be rejected"
            );
        }
    }

    #[test]
    fn fa_quants_record_extraction() {
        let line =
            "CUDA : ARCHS = 1200 | USE_GRAPHS = 1 | FA_QUANTS = all | BLACKWELL_NATIVE_FP4 = 1 |";
        assert_eq!(parse_fa_quants_record(line).as_deref(), Some("all"));
        let list = "CUDA : FA_QUANTS = q8_0-q4_0,f16-f16 | X = 1";
        assert_eq!(
            parse_fa_quants_record(list).as_deref(),
            Some("q8_0-q4_0,f16-f16")
        );
        assert_eq!(parse_fa_quants_record("MTL : EMBED_LIBRARY = 1 |"), None);
        assert_eq!(parse_fa_quants_record("CUDA : FA_QUANTS = | X = 1"), None);
        assert_eq!(parse_fa_quants_record("FA_QUANTS all"), None);
    }

    #[test]
    fn flash_attn_parse() {
        assert_eq!(FlashAttnMode::parse("on"), FlashAttnMode::On);
        assert_eq!(FlashAttnMode::parse("OFF"), FlashAttnMode::Off);
        assert_eq!(FlashAttnMode::parse("auto"), FlashAttnMode::Auto);
        assert_eq!(FlashAttnMode::parse(""), FlashAttnMode::Unknown);
        assert_eq!(FlashAttnMode::parse("bogus"), FlashAttnMode::Unknown);
    }

    // -- FA off (any backend) ----------------------------------------------

    #[test]
    fn quantized_v_with_fa_off_wont_start_on_every_backend_regardless_of_pairs() {
        for backend in [
            Backend::Cuda,
            Backend::Metal,
            Backend::Cpu,
            Backend::Vulkan,
            Backend::Unknown,
        ] {
            let basis = if backend == Backend::Cuda {
                cuda_basis("all")
            } else {
                backend_basis(backend)
            };
            let c = ctx(&basis, None, None);
            for k in [KvType::F16, KvType::Q8_0, KvType::Q4_0] {
                for v in [KvType::Q8_0, KvType::Q4_0, KvType::Iq4Nl] {
                    let out = classify_kv_pair(k, v, FlashAttnMode::Off, &c);
                    assert!(out.is_blocking(), "{backend:?} {k:?}/{v:?}");
                    assert!(out.fix().is_some());
                }
            }
        }
    }

    #[test]
    fn floating_v_with_fa_off_is_fallback_never_native() {
        for backend in [
            Backend::Cuda,
            Backend::Metal,
            Backend::Cpu,
            Backend::Vulkan,
            Backend::Unknown,
        ] {
            let basis = if backend == Backend::Cuda {
                cuda_basis("all")
            } else {
                backend_basis(backend)
            };
            let (d, m) = (device(Some(true)), model(Some(128)));
            let c = ctx(&basis, Some(&d), Some(&m));
            for k in [KvType::F16, KvType::Q8_0, KvType::Q4_0] {
                for v in [KvType::F16, KvType::Bf16, KvType::F32] {
                    let out = classify(k, v, FlashAttnMode::Off, &c);
                    assert!(
                        matches!(
                            out,
                            KvSupport::Fallback {
                                impact: Impact::NoFlashAttention,
                                ..
                            }
                        ),
                        "{backend:?} {k:?}/{v:?}: {out:?}"
                    );
                }
            }
        }
    }

    // -- CUDA ----------------------------------------------------------------

    #[test]
    fn cuda_all_pairs_native_with_device_and_model_evidence() {
        let basis = cuda_basis("all");
        let (d, m) = (device(None), model(Some(128)));
        let c = ctx(&basis, Some(&d), Some(&m));
        for (k, v) in [
            (KvType::Q8_0, KvType::Q8_0),
            (KvType::Q8_0, KvType::Q4_0),
            (KvType::Q8_0, KvType::F16),
            (KvType::F16, KvType::Q4_0),
        ] {
            for mode in [FlashAttnMode::On, FlashAttnMode::Auto] {
                if mode == FlashAttnMode::Auto && v.is_floating() {
                    continue; // auto + floating V needs an FA-selection observation
                }
                assert_eq!(
                    classify(k, v, mode, &c),
                    KvSupport::Native,
                    "{k:?}/{v:?} {mode:?}"
                );
            }
            let out = classify_kv_pair(k, v, FlashAttnMode::On, &c);
            assert_eq!(out.compiled_pair.confirmed_value(), Some(&true));
        }
    }

    #[test]
    fn cuda_confirmed_pair_without_device_or_model_is_unverified_but_compiled_is_reported() {
        let basis = cuda_basis("all");
        for (d, m) in [
            (None, Some(model(Some(128)))),
            (Some(device(None)), None),
            (Some(device(None)), Some(model(None))),
            (Some(device(None)), Some(model(Some(13)))),
        ] {
            let c = ctx(&basis, d.as_ref(), m.as_ref());
            let out = classify_kv_pair(KvType::Q8_0, KvType::Q4_0, FlashAttnMode::On, &c);
            assert!(
                matches!(out.expected, KvSupport::Unverified { .. }),
                "{out:?}"
            );
            assert!(!out.is_blocking());
            assert_eq!(out.compiled_pair.confirmed_value(), Some(&true));
            assert!(!out.buildable_missing);
        }
    }

    #[test]
    fn cuda_confirmed_missing_pair_is_conversion_fallback_and_buildable() {
        let basis = cuda_basis("q8_0-q8_0");
        let c = ctx(&basis, None, None);
        let out = classify_kv_pair(KvType::Q8_0, KvType::Q4_0, FlashAttnMode::On, &c);
        assert!(matches!(
            out.expected,
            KvSupport::Fallback {
                impact: Impact::SlowerDecode,
                ..
            }
        ));
        assert_eq!(out.compiled_pair.confirmed_value(), Some(&false));
        assert!(out.buildable_missing);
        assert!(!out.is_blocking());
    }

    #[test]
    fn cuda_missing_non_buildable_pair_is_fallback_without_rebuild_link() {
        let basis = cuda_basis("all");
        let c = ctx(&basis, None, None);
        for (k, v) in [(KvType::Iq4Nl, KvType::Iq4Nl), (KvType::F32, KvType::F32)] {
            let out = classify_kv_pair(k, v, FlashAttnMode::On, &c);
            assert!(
                matches!(out.expected, KvSupport::Fallback { .. }),
                "{out:?}"
            );
            assert!(!out.buildable_missing, "{k:?}/{v:?} cannot be built");
        }
    }

    #[test]
    fn cuda_fa_off_never_offers_rebuild() {
        let basis = cuda_basis("q8_0-q8_0");
        let c = ctx(&basis, None, None);
        let out = classify_kv_pair(KvType::Q8_0, KvType::F16, FlashAttnMode::Off, &c);
        assert!(!out.buildable_missing);
    }

    #[test]
    fn every_head_dimension_must_be_adequate() {
        let basis = cuda_basis("all");
        let d = device(None);
        let mixed = ModelEvidence {
            metadata_identity: "gemma".into(),
            head_dims: vec![256, 512],
        };
        let c = ctx(&basis, Some(&d), Some(&mixed));
        assert!(matches!(
            classify(KvType::Q8_0, KvType::Q8_0, FlashAttnMode::On, &c),
            KvSupport::Unverified { .. }
        ));
        let both_ok = ModelEvidence {
            metadata_identity: "x".into(),
            head_dims: vec![128, 256],
        };
        let c = ctx(&basis, Some(&d), Some(&both_ok));
        assert_eq!(
            classify(KvType::Q8_0, KvType::Q8_0, FlashAttnMode::On, &c),
            KvSupport::Native
        );
    }

    #[test]
    fn cuda_missing_malformed_or_assumed_pairs_are_unverified_and_nonblocking() {
        let missing = KvSupportBasis {
            fa_pairs: Evidence::unavailable("no FA_QUANTS record"),
            ..cuda_basis("all")
        };
        let assumed = KvSupportBasis {
            fa_pairs: Evidence::assumed(compiled_fa_pairs("all").unwrap(), "older runtime"),
            ..cuda_basis("all")
        };
        let (d, m) = (device(None), model(Some(128)));
        for basis in [&missing, &assumed] {
            let c = ctx(basis, Some(&d), Some(&m));
            for mode in [FlashAttnMode::On, FlashAttnMode::Auto] {
                let out = classify_kv_pair(KvType::Q8_0, KvType::Q4_0, mode, &c);
                assert!(
                    matches!(out.expected, KvSupport::Unverified { .. }),
                    "{out:?}"
                );
                assert!(!out.buildable_missing);
                assert!(!out.compiled_pair.is_confirmed());
            }
        }
    }

    #[test]
    fn cuda_auto_with_floating_v_depends_on_observation() {
        let basis = cuda_basis("all");
        let (d, m) = (device(None), model(Some(128)));
        let mut c = ctx(&basis, Some(&d), Some(&m));
        assert!(matches!(
            classify(KvType::Q8_0, KvType::F16, FlashAttnMode::Auto, &c),
            KvSupport::Unverified { .. }
        ));
        let selected = KvObservations {
            fa_auto_selected: Evidence::confirmed(true, "log"),
            ..Default::default()
        };
        c.observations = Some(&selected);
        assert_eq!(
            classify(KvType::Q8_0, KvType::F16, FlashAttnMode::Auto, &c),
            KvSupport::Native
        );
        let disabled = KvObservations {
            fa_auto_selected: Evidence::confirmed(false, "log"),
            ..Default::default()
        };
        c.observations = Some(&disabled);
        assert!(matches!(
            classify(KvType::Q8_0, KvType::F16, FlashAttnMode::Auto, &c),
            KvSupport::Fallback {
                impact: Impact::NoFlashAttention,
                ..
            }
        ));
        // An assumed observation never upgrades.
        let assumed = KvObservations {
            fa_auto_selected: Evidence::assumed(true, "guess"),
            ..Default::default()
        };
        c.observations = Some(&assumed);
        assert!(matches!(
            classify(KvType::Q8_0, KvType::F16, FlashAttnMode::Auto, &c),
            KvSupport::Unverified { .. }
        ));
    }

    // -- Metal ---------------------------------------------------------------

    #[test]
    fn metal_equal_pair_native_only_with_full_evidence() {
        let basis = backend_basis(Backend::Metal);
        let (d, m) = (device(None), model(Some(256)));
        let c = ctx(&basis, Some(&d), Some(&m));
        assert_eq!(
            classify(KvType::Q8_0, KvType::Q8_0, FlashAttnMode::On, &c),
            KvSupport::Native
        );
        // No model metadata: unverified, never Native.
        let c = ctx(&basis, Some(&d), None);
        assert!(matches!(
            classify(KvType::Q8_0, KvType::Q8_0, FlashAttnMode::On, &c),
            KvSupport::Unverified { .. }
        ));
    }

    #[test]
    fn metal_bf16_requires_hardware_evidence() {
        let basis = backend_basis(Backend::Metal);
        let m = model(Some(128));
        for (bf16, native) in [(Some(true), true), (Some(false), false), (None, false)] {
            let d = device(bf16);
            let c = ctx(&basis, Some(&d), Some(&m));
            let out = classify(KvType::Bf16, KvType::Bf16, FlashAttnMode::On, &c);
            assert_eq!(
                out == KvSupport::Native,
                native,
                "bf16 hardware {bf16:?}: {out:?}"
            );
        }
        // f16/f16 is unaffected by the bf16 flag.
        let d = device(Some(false));
        let c = ctx(&basis, Some(&d), Some(&m));
        assert_eq!(
            classify(KvType::F16, KvType::F16, FlashAttnMode::On, &c),
            KvSupport::Native
        );
    }

    #[test]
    fn metal_mixed_pairs_are_unverified_with_scoped_warning_until_observed() {
        let basis = backend_basis(Backend::Metal);
        let (d, m) = (device(Some(true)), model(Some(256)));
        let mut c = ctx(&basis, Some(&d), Some(&m));
        for (k, v) in [
            (KvType::Q8_0, KvType::F16),
            (KvType::Q8_0, KvType::Q4_0),
            (KvType::F16, KvType::Q8_0),
            (KvType::Bf16, KvType::Q8_0),
        ] {
            for mode in [FlashAttnMode::On, FlashAttnMode::Auto] {
                if mode == FlashAttnMode::Auto && v.is_floating() {
                    continue;
                }
                let out = classify_kv_pair(k, v, mode, &c);
                let KvSupport::Unverified { reason } = &out.expected else {
                    panic!("{k:?}/{v:?} {mode:?}: {:?}", out.expected);
                };
                assert!(
                    reason.contains("b11719") && reason.contains("M5 Max"),
                    "{reason}"
                );
                assert!(!out.is_blocking());
                assert!(!out.buildable_missing, "no rebuild link on Metal");
            }
        }
        let observed = KvObservations {
            metal_mixed_cpu_attention: Evidence::confirmed(true, "log"),
            ..Default::default()
        };
        c.observations = Some(&observed);
        assert!(matches!(
            classify(KvType::Q8_0, KvType::Q4_0, FlashAttnMode::On, &c),
            KvSupport::Fallback {
                impact: Impact::CpuAttention,
                ..
            }
        ));
    }

    #[test]
    fn metal_auto_floating_v_mixed_follows_fa_auto_observation() {
        let basis = backend_basis(Backend::Metal);
        let (d, m) = (device(None), model(Some(256)));
        let mut c = ctx(&basis, Some(&d), Some(&m));
        assert!(matches!(
            classify(KvType::Q8_0, KvType::F16, FlashAttnMode::Auto, &c),
            KvSupport::Unverified { .. }
        ));
        let disabled = KvObservations {
            fa_auto_selected: Evidence::confirmed(false, "log"),
            ..Default::default()
        };
        c.observations = Some(&disabled);
        assert!(matches!(
            classify(KvType::Q8_0, KvType::F16, FlashAttnMode::Auto, &c),
            KvSupport::Fallback {
                impact: Impact::NoFlashAttention,
                ..
            }
        ));
    }

    fn verified_metal_basis() -> KvSupportBasis {
        KvSupportBasis {
            build_number: Some(11719),
            metal_bf16: Evidence::confirmed(true, "probe"),
            ..backend_basis(Backend::Metal)
        }
    }

    #[test]
    fn verified_metal_build_makes_every_mixed_pair_a_cpu_attention_fallback() {
        // Measured at b11719 / M5 Max / Qwen3.8-27B, 8k depth: every mixed pair,
        // including float/float (bf16/f16), collapsed to CPU attention.
        let basis = verified_metal_basis();
        let (d, m) = (device(None), model(Some(256)));
        let c = ctx(&basis, Some(&d), Some(&m));
        for (k, v) in [
            (KvType::Q8_0, KvType::Q4_0),
            (KvType::Q8_0, KvType::Bf16),
            (KvType::Bf16, KvType::Q8_0),
            (KvType::Bf16, KvType::F16),
            (KvType::Q8_0, KvType::F16),
        ] {
            let out = classify(k, v, FlashAttnMode::On, &c);
            assert!(
                matches!(
                    out,
                    KvSupport::Fallback {
                        impact: Impact::CpuAttention,
                        ..
                    }
                ),
                "{k:?}/{v:?}: {out:?}"
            );
            assert!(!out.is_blocking());
        }
        // Quantized V with auto also requests FA, so the same CPU path applies.
        assert!(matches!(
            classify(KvType::Q8_0, KvType::Q4_0, FlashAttnMode::Auto, &c),
            KvSupport::Fallback {
                impact: Impact::CpuAttention,
                ..
            }
        ));
        // Floating V with auto: the runtime disables flash attention (GPU, unfused).
        assert!(matches!(
            classify(KvType::Q8_0, KvType::Bf16, FlashAttnMode::Auto, &c),
            KvSupport::Fallback {
                impact: Impact::NoFlashAttention,
                ..
            }
        ));
        // Equal pairs keep their fused path.
        for t in [KvType::Q8_0, KvType::F16, KvType::Bf16, KvType::F32] {
            assert_eq!(
                classify(t, t, FlashAttnMode::On, &c),
                KvSupport::Native,
                "{t:?}"
            );
        }
    }

    #[test]
    fn newer_metal_builds_keep_the_fallback_with_a_not_retested_qualifier() {
        let (d, m) = (device(None), model(Some(256)));
        for (build, qualified) in [(11719, false), (11720, true), (99999, true)] {
            let basis = KvSupportBasis {
                build_number: Some(build),
                ..backend_basis(Backend::Metal)
            };
            let c = ctx(&basis, Some(&d), Some(&m));
            let KvSupport::Fallback { reason, impact } =
                classify(KvType::Q8_0, KvType::Q4_0, FlashAttnMode::On, &c)
            else {
                panic!("build {build} should fall back");
            };
            assert_eq!(impact, Impact::CpuAttention);
            assert_eq!(
                reason.contains("not been re-tested"),
                qualified,
                "{build}: {reason}"
            );
        }
    }

    #[test]
    fn escape_hatch_returns_builds_with_mixed_kernels_to_unverified() {
        // The constant is None today; exercise the rule function directly.
        for (build, expect) in [
            (Some(11718), false),
            (Some(11719), true),
            (Some(12000), true),
            (None, false),
        ] {
            assert_eq!(
                metal_rule_applies_to(Backend::Metal, build, None),
                expect,
                "{build:?}"
            );
        }
        for (build, expect) in [
            (Some(11719), true),
            (Some(12049), true),
            (Some(12050), false),
            (Some(13000), false),
        ] {
            assert_eq!(
                metal_rule_applies_to(Backend::Metal, build, Some(12050)),
                expect,
                "{build:?}"
            );
        }
        assert!(!metal_rule_applies_to(Backend::Cuda, Some(11719), None));
        assert_eq!(
            METAL_FA_MIXED_KERNELS_FROM_BUILD, None,
            "update the docs when this is set"
        );
    }

    #[test]
    fn unverified_metal_builds_stay_unverified_for_mixed_pairs() {
        let (d, m) = (device(None), model(Some(256)));
        for build in [None, Some(11718), Some(100)] {
            let basis = KvSupportBasis {
                build_number: build,
                ..backend_basis(Backend::Metal)
            };
            let c = ctx(&basis, Some(&d), Some(&m));
            assert!(
                matches!(
                    classify(KvType::Q8_0, KvType::Q4_0, FlashAttnMode::On, &c),
                    KvSupport::Unverified { .. }
                ),
                "build {build:?}"
            );
        }
    }

    #[test]
    fn metal_bf16_uses_the_runtime_reported_device_flag() {
        let m = model(Some(128));
        for (flag, native) in [(true, true), (false, false)] {
            let basis = KvSupportBasis {
                metal_bf16: Evidence::confirmed(flag, "probe"),
                ..verified_metal_basis()
            };
            let d = device(None); // no per-device evidence: the basis flag is used
            let c = ctx(&basis, Some(&d), Some(&m));
            let out = classify(KvType::Bf16, KvType::Bf16, FlashAttnMode::On, &c);
            assert_eq!(
                out == KvSupport::Native,
                native,
                "has bfloat={flag}: {out:?}"
            );
        }
    }

    #[test]
    fn metal_unsupported_equal_type_is_not_native() {
        let basis = backend_basis(Backend::Metal);
        let (d, m) = (device(None), model(Some(128)));
        let c = ctx(&basis, Some(&d), Some(&m));
        let out = classify(KvType::Iq4Nl, KvType::Iq4Nl, FlashAttnMode::On, &c);
        assert!(matches!(out, KvSupport::Unverified { .. }), "{out:?}");
    }

    // -- CPU / Vulkan / Unknown ---------------------------------------------

    #[test]
    fn cpu_vulkan_unknown_are_unverified_with_fa_on_or_auto() {
        for backend in [Backend::Cpu, Backend::Vulkan, Backend::Unknown] {
            let basis = backend_basis(backend);
            let (d, m) = (device(Some(true)), model(Some(128)));
            let c = ctx(&basis, Some(&d), Some(&m));
            for (k, v) in [
                (KvType::Q8_0, KvType::Q8_0),
                (KvType::Q8_0, KvType::Q4_0),
                (KvType::F16, KvType::F16),
            ] {
                for mode in [FlashAttnMode::On, FlashAttnMode::Auto] {
                    if mode == FlashAttnMode::Auto && v.is_floating() {
                        continue;
                    }
                    let out = classify_kv_pair(k, v, mode, &c);
                    assert!(
                        matches!(out.expected, KvSupport::Unverified { .. }),
                        "{backend:?} {out:?}"
                    );
                    assert!(!out.compiled_pair.is_confirmed());
                    assert!(!out.buildable_missing);
                }
            }
        }
    }

    // -- unknown effective FA mode ---------------------------------------------

    #[test]
    fn unknown_fa_mode_resolves_only_when_all_modes_agree() {
        let basis = cuda_basis("all");
        let (d, m) = (device(None), model(Some(128)));
        let c = ctx(&basis, Some(&d), Some(&m));
        // Native when on, fallback when off: unverified, and never a block.
        let out = classify_kv_pair(KvType::Q8_0, KvType::Q8_0, FlashAttnMode::Unknown, &c);
        assert!(matches!(out.expected, KvSupport::Unverified { .. }));
        // Quantized V: off would refuse, on is native: outcomes differ, so no block.
        let out = classify_kv_pair(KvType::Q8_0, KvType::Q4_0, FlashAttnMode::Unknown, &c);
        assert!(matches!(out.expected, KvSupport::Unverified { .. }));
        assert!(!out.is_blocking());
        // Missing pair with unknown mode: off + floating V falls back with a different
        // impact than on, so unverified.
        let basis = cuda_basis("q8_0-q8_0");
        let c = ctx(&basis, Some(&d), Some(&m));
        let out = classify_kv_pair(KvType::Q8_0, KvType::F16, FlashAttnMode::Unknown, &c);
        assert!(matches!(out.expected, KvSupport::Unverified { .. }));
    }

    // -- compiled availability independent of expected execution --------------

    #[test]
    fn compiled_availability_is_independent_of_expected_execution() {
        let basis = cuda_basis("all");
        let c = ctx(&basis, None, None);
        for mode in ALL_MODES {
            let out = classify_kv_pair(KvType::Q8_0, KvType::F16, mode, &c);
            assert_eq!(out.compiled_pair.confirmed_value(), Some(&true), "{mode:?}");
        }
    }

    // -- string front door ------------------------------------------------------

    #[test]
    fn strs_front_door_leaves_empty_and_unknown_to_existing_validation() {
        let basis = cuda_basis("all");
        let c = ctx(&basis, None, None);
        assert!(classify_kv_strs("", "q8_0", "on", &c).is_none());
        assert!(classify_kv_strs("q8_0", " ", "on", &c).is_none());
        assert!(classify_kv_strs("q8_0", "wat", "on", &c).is_none());
        let out = classify_kv_strs("Q8_0", "Q4_0", "off", &c).unwrap();
        assert!(out.is_blocking());
    }

    #[test]
    fn the_old_mixed_pair_no_longer_blocks_with_fa_on() {
        // The regression this module exists to fix: q8_0/q4_0 must not block
        // when flash attention is on, even with no evidence at all.
        let basis = KvSupportBasis::default();
        let c = ctx(&basis, None, None);
        for mode in [
            FlashAttnMode::On,
            FlashAttnMode::Auto,
            FlashAttnMode::Unknown,
        ] {
            let out = classify_kv_strs(
                "q8_0",
                "q4_0",
                match mode {
                    FlashAttnMode::On => "on",
                    FlashAttnMode::Auto => "auto",
                    _ => "",
                },
                &c,
            )
            .unwrap();
            assert!(!out.is_blocking(), "{mode:?}: {:?}", out.expected);
        }
    }

    #[test]
    fn log_lines_yield_partial_observations_only() {
        let convert = "W ggml_cuda_flash_attn_ext: no FlashAttention vector kernel compiled for K/V types q8_0-q4_0, converting K and V to f16 instead (slow). Add \"q8_0-q4_0\" to GGML_CUDA_FA_QUANTS to compile it.";
        assert_eq!(
            observe_log_line(convert),
            Some(KvLogObservation::CudaConversionFallback {
                k: "q8_0".into(),
                v: "q4_0".into()
            })
        );
        let disabled = "W llama_context: flash_attn not supported, set to disabled";
        assert_eq!(
            observe_log_line(disabled),
            Some(KvLogObservation::FlashAttnDisabled)
        );
        // Unrelated or look-alike lines, and unknown types, are ignored.
        assert_eq!(
            observe_log_line("I srv  llama_server: initializing ..."),
            None
        );
        assert_eq!(
            observe_log_line("fused_gdn not supported, set to disabled"),
            None
        );
        assert_eq!(
            observe_log_line("no FlashAttention vector kernel compiled for K/V types zzz-q8_0, x"),
            None
        );
    }

    #[test]
    fn session_observations_are_bounded_deduplicated_and_resettable() {
        reset_session_observations();
        for _ in 0..3 {
            record_log_line("llama_context: flash_attn not supported, set to disabled");
        }
        record_log_line("unrelated");
        assert_eq!(
            session_observations(),
            vec![KvLogObservation::FlashAttnDisabled]
        );
        reset_session_observations();
        assert!(session_observations().is_empty());
    }

    #[test]
    fn evidence_serializes_with_a_state_tag() {
        let json = serde_json::to_value(Evidence::confirmed(true, "probe")).unwrap();
        assert_eq!(json["state"], "confirmed");
        let basis = cuda_basis("q8_0-q8_0");
        let round: KvSupportBasis =
            serde_json::from_value(serde_json::to_value(&basis).unwrap()).unwrap();
        assert_eq!(round, basis);
    }
}
