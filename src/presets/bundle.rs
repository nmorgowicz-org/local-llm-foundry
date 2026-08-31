//! Phase 2: typed llama.cpp preset-bundle schema (v6).
//!
//! This module defines the wire shapes for the optional bundle attached to
//! `ModelPreset`, plus the **single server-owned v6 bundle constructor** — the
//! only code path allowed to set `fit_enabled` on a new bundle.
//!
//! Design constraints (architecture §5):
//! - Every enum that can grow uses a bounded open-string serde helper:
//!   known values deserialize to typed variants; unknown strings are
//!   preserved verbatim in an `Unknown(String)` variant and round-trip
//!   through save/edit/save cycles unchanged.
//! - `#[serde(default)]` on every persisted struct plus a flattened
//!   extension map means future fields are preserved on load even when the
//!   current build doesn't know about them.
//! - Local paths appear only in the authenticated full-editor response.
//!   The card-view projection is built by the API layer.

use serde::de::Deserializer as SerdeDeserializer;
use serde::ser::Serializer as SerdeSerializer;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::ModelPreset;

// ─────────────────────────────────────────────────────────────────────────────
// Bounded open-string enum serde helpers
//
// A single generic pattern shared by every bounded enum in this module.
// Known values → typed variant. Unknown string → Unknown(String).
// Serializes the typed variant name for known values, or the raw stored string
// for Unknown. This guarantees round-trip fidelity for unsupported values.
// ─────────────────────────────────────────────────────────────────────────────

/// Trait implemented by each bounded enum.
pub trait BoundedEnum: Sized + Clone + PartialEq + std::fmt::Debug {
    /// Parse from a wire string. Returns Self::Unknown(s) for unrecognized input.
    fn from_wire(s: &str) -> Self;
    /// Serialize: known variant → variant name; Unknown(s) → s.
    fn to_wire(&self) -> &str;
}

/// Generic deserializer for bounded enums.
pub fn bounded_deserialize<'de, D, E>(d: D) -> Result<E, D::Error>
where
    D: SerdeDeserializer<'de>,
    E: BoundedEnum,
{
    let s = String::deserialize(d)?;
    Ok(E::from_wire(&s))
}

/// Generic serializer for bounded enums.
pub fn bounded_serialize<S, E>(v: &E, s: S) -> Result<S::Ok, S::Error>
where
    S: SerdeSerializer,
    E: BoundedEnum,
{
    s.serialize_str(v.to_wire())
}

/// Generic deserializer for `Option<BoundedEnum>` fields.
///
/// `null` / absent → `None`. Present → `Some(E::from_wire(..))`, same
/// unknown-preserving semantics as the scalar `bounded_deserialize`.
pub fn bounded_deserialize_opt<'de, D, E>(d: D) -> Result<Option<E>, D::Error>
where
    D: SerdeDeserializer<'de>,
    E: BoundedEnum,
{
    let s = Option::<String>::deserialize(d)?;
    Ok(s.map(|s| E::from_wire(&s)))
}

/// Generic serializer for `Option<BoundedEnum>` fields.
pub fn bounded_serialize_opt<S, E>(v: &Option<E>, s: S) -> Result<S::Ok, S::Error>
where
    S: SerdeSerializer,
    E: BoundedEnum,
{
    match v {
        Some(e) => s.serialize_some(e.to_wire()),
        None => s.serialize_none(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PresetBundleIdentity
// ─────────────────────────────────────────────────────────────────────────────

/// Stable identity for a bundle and its exact tune.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct PresetBundleIdentity {
    pub bundle_id: String,
    pub tune_id: String,
    pub display_name: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// PresetArtifactRole
// ─────────────────────────────────────────────────────────────────────────────

/// Role of an artifact in a bundle.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum PresetArtifactRole {
    #[default]
    Weights,
    Mmproj,
    Draft,
    Unknown(String),
}

impl BoundedEnum for PresetArtifactRole {
    fn from_wire(s: &str) -> Self {
        match s {
            "weights" => Self::Weights,
            "mmproj" => Self::Mmproj,
            "draft" => Self::Draft,
            _ => Self::Unknown(s.to_string()),
        }
    }
    fn to_wire(&self) -> &str {
        match self {
            Self::Weights => "weights",
            Self::Mmproj => "mmproj",
            Self::Draft => "draft",
            Self::Unknown(s) => s,
        }
    }
}

// `Unknown(String)` is not a unit variant, so `#[derive(Default)]` cannot
// mark it `#[default]`. Implemented manually instead; `Weights` is the
// primary/most common artifact role.

// ─────────────────────────────────────────────────────────────────────────────
// PresetModelKind
// ─────────────────────────────────────────────────────────────────────────────

/// GGUF architecture class.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum PresetModelKind {
    #[default]
    Dense,
    Moe,
    Unknown(String),
}

impl BoundedEnum for PresetModelKind {
    fn from_wire(s: &str) -> Self {
        match s {
            "dense" => Self::Dense,
            "moe" | "hybrid_moe" => Self::Moe,
            _ => Self::Unknown(s.to_string()),
        }
    }
    fn to_wire(&self) -> &str {
        match self {
            Self::Dense => "dense",
            Self::Moe => "moe",
            Self::Unknown(s) => s,
        }
    }
}

// `Unknown(String)` is not a unit variant, so `#[derive(Default)]` cannot
// mark it `#[default]`. Implemented manually instead; `Dense` is the
// most common architecture class.

// ─────────────────────────────────────────────────────────────────────────────
// PresetHfOriginEvidence
// ─────────────────────────────────────────────────────────────────────────────

/// How the HF origin was established.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum PresetHfOriginEvidence {
    /// Confirmed by DownloadProvenance sidecar.
    DownloadProvenance,
    /// Suggested by hf_resolve_origin scoring.
    #[default]
    ResolverSuggestion,
    /// Manually confirmed by the operator.
    UserEntered,
    Unknown(String),
}

impl BoundedEnum for PresetHfOriginEvidence {
    fn from_wire(s: &str) -> Self {
        match s {
            "download_provenance" => Self::DownloadProvenance,
            "resolver_suggestion" => Self::ResolverSuggestion,
            "user_entered" => Self::UserEntered,
            _ => Self::Unknown(s.to_string()),
        }
    }
    fn to_wire(&self) -> &str {
        match self {
            Self::DownloadProvenance => "download_provenance",
            Self::ResolverSuggestion => "resolver_suggestion",
            Self::UserEntered => "user_entered",
            Self::Unknown(s) => s,
        }
    }
}

// `Unknown(String)` is not a unit variant, so `#[derive(Default)]` cannot
// mark it `#[default]`. Implemented manually instead; `ResolverSuggestion`
// is the weakest/least-confirmed evidence and the safest default when no
// real evidence has been recorded.

/// HF origin provenance for a model artifact.
///
/// Field names are the binding architecture §5 contract.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct PresetHfOrigin {
    pub repo_id: String,
    pub remote_path: String,
    /// Pinned revision (commit SHA when known, branch name when not).
    pub revision: Option<String>,
    /// How this origin was established.
    #[serde(
        serialize_with = "bounded_serialize",
        deserialize_with = "bounded_deserialize"
    )]
    pub evidence: PresetHfOriginEvidence,
    /// Resolver confidence score (0.0..=1.0); provenance, never silent merge.
    pub confidence: Option<f64>,
    /// Human-readable resolver reason (bounded display text).
    pub reason: Option<String>,
    /// Explicit operator confirmation for bundle conversion.
    pub user_confirmed: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// PresetDigestCoverage / PresetArtifactDigest
// ─────────────────────────────────────────────────────────────────────────────

/// How the digest was computed.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum PresetDigestCoverage {
    FullFile,
    #[default]
    BoundedGgufHeader,
    Unknown(String),
}

impl BoundedEnum for PresetDigestCoverage {
    fn from_wire(s: &str) -> Self {
        match s {
            "full_file" => Self::FullFile,
            "bounded_gguf_header" => Self::BoundedGgufHeader,
            _ => Self::Unknown(s.to_string()),
        }
    }
    fn to_wire(&self) -> &str {
        match self {
            Self::FullFile => "full_file",
            Self::BoundedGgufHeader => "bounded_gguf_header",
            Self::Unknown(s) => s,
        }
    }
}

// `Unknown(String)` is not a unit variant, so `#[derive(Default)]` cannot
// mark it `#[default]`. Implemented manually instead; `BoundedGgufHeader`
// is the weaker/more common coverage — never claim `FullFile` by default.

/// Digest of the artifact content.
///
/// Field names are the binding architecture §5 contract. A bounded-header
/// digest is sufficient for metadata provenance but never for an `exact`
/// runtime observation; exact evidence requires `FullFile` coverage.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct PresetArtifactDigest {
    /// "sha256" or similar.
    pub algorithm: String,
    pub value: String,
    #[serde(
        serialize_with = "bounded_serialize",
        deserialize_with = "bounded_deserialize"
    )]
    pub coverage: PresetDigestCoverage,
    /// How/where this digest was produced (e.g. "phase-0 capture", "on-save").
    pub provenance: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// PresetQuantizationProvenance / PresetArtifactQuantization
// ─────────────────────────────────────────────────────────────────────────────

/// How the quantization value was determined.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum PresetQuantizationProvenance {
    GgufMetadata,
    #[default]
    FilenameHint,
    UserConfirmed,
    Unknown(String),
}

impl BoundedEnum for PresetQuantizationProvenance {
    fn from_wire(s: &str) -> Self {
        match s {
            "gguf_metadata" => Self::GgufMetadata,
            "filename_hint" => Self::FilenameHint,
            "user_confirmed" => Self::UserConfirmed,
            _ => Self::Unknown(s.to_string()),
        }
    }
    fn to_wire(&self) -> &str {
        match self {
            Self::GgufMetadata => "gguf_metadata",
            Self::FilenameHint => "filename_hint",
            Self::UserConfirmed => "user_confirmed",
            Self::Unknown(s) => s,
        }
    }
}

// `Unknown(String)` is not a unit variant, so `#[derive(Default)]` cannot
// mark it `#[default]`. Implemented manually instead; `FilenameHint` is the
// weakest/provisional provenance and the safest default when unconfirmed.

/// Quantization of a model artifact.
///
/// A filename quantization hint is provisional and cannot override GGUF
/// metadata: `FilenameHint` is display-only until `GgufMetadata` or
/// `UserConfirmed` backs the value.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct PresetArtifactQuantization {
    /// e.g. "q4_k_m", "q8_0"
    pub value: String,
    #[serde(
        serialize_with = "bounded_serialize",
        deserialize_with = "bounded_deserialize"
    )]
    pub provenance: PresetQuantizationProvenance,
}

// ─────────────────────────────────────────────────────────────────────────────
// PresetArtifactMetadata
// ─────────────────────────────────────────────────────────────────────────────

/// GGUF-derived architecture metadata used by the resolver and estimator.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct PresetArtifactMetadata {
    /// `general.architecture` from the GGUF header.
    pub gguf_architecture: Option<String>,
    #[serde(
        serialize_with = "bounded_serialize",
        deserialize_with = "bounded_deserialize"
    )]
    pub model_kind: PresetModelKind,
    pub block_count: Option<u32>,
    pub moe_layer_count: Option<u32>,
    pub native_context_limit: Option<u64>,
    /// Bounded metadata digest (header region).
    pub metadata_digest: Option<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// PresetModelArtifact
// ─────────────────────────────────────────────────────────────────────────────

/// One exact model artifact (weight, mmproj, or draft) inside a bundle.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct PresetModelArtifact {
    pub id: String,
    #[serde(
        serialize_with = "bounded_serialize",
        deserialize_with = "bounded_deserialize"
    )]
    pub role: PresetArtifactRole,
    pub display_name: String,
    /// Local file path (authenticated full-editor response only).
    pub local_path: Option<String>,
    pub hf_origin: Option<PresetHfOrigin>,
    pub size_bytes: Option<u64>,
    pub digest: Option<PresetArtifactDigest>,
    pub quantization: PresetArtifactQuantization,
    pub metadata: PresetArtifactMetadata,
    /// Companion link: artifact id of the mmproj this weights artifact needs.
    pub mmproj_artifact_id: Option<String>,
    /// Companion link: artifact id of the draft model.
    pub draft_artifact_id: Option<String>,
    #[serde(flatten, default)]
    pub extensions: BTreeMap<String, serde_json::Value>,
}

// ─────────────────────────────────────────────────────────────────────────────
// LlamaKvPolicyId
// ─────────────────────────────────────────────────────────────────────────────

/// Named K/V policy pair.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum LlamaKvPolicyId {
    #[default]
    F16F16,
    Q8Q8,
    Q4Q4,
    MixedQ8Q4,
    Unknown(String),
}

impl BoundedEnum for LlamaKvPolicyId {
    fn from_wire(s: &str) -> Self {
        match s {
            "f16_f16" => Self::F16F16,
            "q8_0_q8_0" => Self::Q8Q8,
            "q4_0_q4_0" => Self::Q4Q4,
            "q8_0_q4_0" => Self::MixedQ8Q4,
            _ => Self::Unknown(s.to_string()),
        }
    }
    fn to_wire(&self) -> &str {
        match self {
            Self::F16F16 => "f16_f16",
            Self::Q8Q8 => "q8_0_q8_0",
            Self::Q4Q4 => "q4_0_q4_0",
            Self::MixedQ8Q4 => "q8_0_q4_0",
            Self::Unknown(s) => s,
        }
    }
}

// `Unknown(String)` is not a unit variant, so `#[derive(Default)]` cannot
// mark it `#[default]`. Implemented manually instead; `F16F16` is
// llama.cpp's actual runtime K/V default.

// ─────────────────────────────────────────────────────────────────────────────
// PresetFitIntent
// ─────────────────────────────────────────────────────────────────────────────

/// Non-launchable fit intent recorded in a bundle selection.
/// `None` on the selection field means "exact selection, no fit intent".
#[derive(Debug, Clone, PartialEq)]
pub enum PresetFitIntent {
    QualityFirst,
    Balanced,
    LowVram,
    Unknown(String),
}

impl BoundedEnum for PresetFitIntent {
    fn from_wire(s: &str) -> Self {
        match s {
            "quality_first" => Self::QualityFirst,
            "balanced" => Self::Balanced,
            "low_vram" => Self::LowVram,
            _ => Self::Unknown(s.to_string()),
        }
    }
    fn to_wire(&self) -> &str {
        match self {
            Self::QualityFirst => "quality_first",
            Self::Balanced => "balanced",
            Self::LowVram => "low_vram",
            Self::Unknown(s) => s,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PresetWorkloadPolicy
// ─────────────────────────────────────────────────────────────────────────────

/// Backend-owned workload policy for bundle eligibility gating.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum PresetWorkloadPolicy {
    AgenticTools,
    #[default]
    GeneralChat,
    RoleplayCreative,
    CustomUnknown,
    Unknown(String),
}

impl BoundedEnum for PresetWorkloadPolicy {
    fn from_wire(s: &str) -> Self {
        match s {
            "agentic_tools" => Self::AgenticTools,
            "general_chat" => Self::GeneralChat,
            "roleplay_creative" => Self::RoleplayCreative,
            "custom_unknown" => Self::CustomUnknown,
            _ => Self::Unknown(s.to_string()),
        }
    }
    fn to_wire(&self) -> &str {
        match self {
            Self::AgenticTools => "agentic_tools",
            Self::GeneralChat => "general_chat",
            Self::RoleplayCreative => "roleplay_creative",
            Self::CustomUnknown => "custom_unknown",
            Self::Unknown(s) => s,
        }
    }
}

// `Unknown(String)` is not a unit variant, so `#[derive(Default)]` cannot
// mark it `#[default]`. Implemented manually instead; `GeneralChat` is the
// most generic/common workload policy.

// ─────────────────────────────────────────────────────────────────────────────
// PresetPerformanceOption
// ─────────────────────────────────────────────────────────────────────────────

/// A named batch/ubatch pair choice.
///
/// Explicit bundle performance choices must be nonzero: a zero/null pair is
/// the legacy flat omit/runtime-default sentinel and may not be copied into
/// an explicit bundle option and called reproducible (architecture §5).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct PresetPerformanceOption {
    pub id: String,
    pub label: String,
    pub batch_size: u32,
    pub ubatch_size: u32,
}

// ─────────────────────────────────────────────────────────────────────────────
// PresetBundleSelection
// ─────────────────────────────────────────────────────────────────────────────

/// One exact saved selection: artifact, context, KV, performance, MoE.
///
/// `PresetBundleSelection` doubles as the `PATCH /selection` request body.
/// The server strips any client-supplied `intent_source` on mutation writes:
/// it is stored as `None` and is display provenance only — never stored state
/// (architecture §5). Phase 2 persists the field shape; enforcement lands in
/// the Phase 3 resolver/API.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct PresetBundleSelection {
    pub artifact_id: String,
    pub context_size: u64,
    #[serde(
        serialize_with = "bounded_serialize",
        deserialize_with = "bounded_deserialize"
    )]
    pub kv_policy: LlamaKvPolicyId,
    pub performance_id: String,
    pub n_cpu_moe: Option<i32>,
    /// `None` means exact selection (no intent).
    #[serde(
        serialize_with = "bounded_serialize_opt",
        deserialize_with = "bounded_deserialize_opt",
        skip_serializing_if = "Option::is_none"
    )]
    pub intent_source: Option<PresetFitIntent>,
}

// ─────────────────────────────────────────────────────────────────────────────
// PresetBundleSpec
// ─────────────────────────────────────────────────────────────────────────────

/// Full optional bundle attached to a ModelPreset in schema v6.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct PresetBundleSpec {
    pub identity: PresetBundleIdentity,
    pub artifacts: Vec<PresetModelArtifact>,
    pub context_options: Vec<u64>,
    /// Allowed KV policy IDs.
    #[serde(
        serialize_with = "bounded_serialize_vec",
        deserialize_with = "bounded_deserialize_vec"
    )]
    pub kv_policy_options: Vec<LlamaKvPolicyId>,
    pub performance_options: Vec<PresetPerformanceOption>,
    /// Allowed n_cpu_moe values; 0 = all GPU.
    pub cpu_moe_options: Vec<i32>,
    /// Curated operator-authored combinations.
    pub curated_selections: Vec<PresetBundleSelection>,
    /// Explicit bundle policy: false → only curated; true → resolver may
    /// accept recombination subject to full validation.
    pub allow_validated_custom: bool,
    #[serde(
        serialize_with = "bounded_serialize",
        deserialize_with = "bounded_deserialize"
    )]
    pub workload_policy: PresetWorkloadPolicy,
    /// The saved default selection.
    pub default_selection: PresetBundleSelection,
    #[serde(flatten, default)]
    pub extensions: BTreeMap<String, serde_json::Value>,
}

impl PresetBundleSpec {
    /// Returns true if this spec is structurally valid (pure; no binary
    /// lookup). Called on load and before save.
    pub fn structural_issues(&self) -> Vec<String> {
        let mut issues = Vec::new();

        // Artifact ID uniqueness across all roles.
        {
            let mut seen = std::collections::HashSet::new();
            for a in &self.artifacts {
                if a.id.trim().is_empty() {
                    issues.push("EMPTY_ARTIFACT_ID: every artifact must have an id".to_string());
                }
                if !seen.insert(&a.id) {
                    issues.push(format!(
                        "DUPLICATE_ARTIFACT_ID: artifact '{}' appears more than once",
                        a.id
                    ));
                }
            }
        }

        // Companion references must resolve within the same bundle.
        for a in &self.artifacts {
            if let Some(mm) = &a.mmproj_artifact_id
                && !self
                    .artifacts
                    .iter()
                    .any(|x| &x.id == mm && matches!(x.role, PresetArtifactRole::Mmproj))
            {
                issues.push(format!(
                    "MISSING_MMPROJ_ARTIFACT: artifact '{}' references mmproj '{}' which does not exist in bundle",
                    a.id, mm
                ));
            }
            if let Some(d) = &a.draft_artifact_id
                && !self
                    .artifacts
                    .iter()
                    .any(|x| &x.id == d && matches!(x.role, PresetArtifactRole::Draft))
            {
                issues.push(format!(
                    "MISSING_DRAFT_ARTIFACT: artifact '{}' references draft '{}' which does not exist in bundle",
                    a.id, d
                ));
            }
        }

        // Performance options are explicit and reproducible. A zero pair is
        // the legacy flat runtime-default sentinel, not a bundle option.
        let mut performance_ids = std::collections::HashSet::new();
        for p in &self.performance_options {
            if p.id.trim().is_empty() {
                issues.push(
                    "EMPTY_PERFORMANCE_ID: every performance option must have an id".to_string(),
                );
            }
            if !performance_ids.insert(&p.id) {
                issues.push(format!(
                    "DUPLICATE_PERFORMANCE_ID: performance option '{}' appears more than once",
                    p.id
                ));
            }
            if p.batch_size == 0 || p.ubatch_size == 0 {
                issues.push(format!(
                    "ZERO_PERFORMANCE_CHOICE: performance option '{}' has zero batch or ubatch",
                    p.id
                ));
            } else if p.ubatch_size > p.batch_size {
                issues.push(format!(
                    "UBATCH_EXCEEDS_BATCH: performance option '{}' has ubatch {} above batch {}",
                    p.id, p.ubatch_size, p.batch_size
                ));
            }
        }

        // n_cpu_moe must be non-negative.
        for m in &self.cpu_moe_options {
            if *m < 0 {
                issues.push(format!(
                    "NEGATIVE_CPU_MOE: cpu_moe option {} is negative",
                    m
                ));
            }
        }

        // The default selection must point to a weights artifact and every
        // selected axis must be present in its catalog.
        if !self.artifacts.iter().any(|a| {
            a.id == self.default_selection.artifact_id
                && matches!(a.role, PresetArtifactRole::Weights)
        }) {
            issues.push(format!(
                "DEFAULT_SELECTION_MISSING_ARTIFACT: default_selection references artifact '{}' which is not a weights artifact in the bundle",
                self.default_selection.artifact_id
            ));
        }

        if !self.context_options.is_empty()
            && !self
                .context_options
                .contains(&self.default_selection.context_size)
        {
            issues.push(
                "DEFAULT_SELECTION_CONTEXT_NOT_ALLOWED: default context is not in context_options"
                    .to_string(),
            );
        }
        if !self.kv_policy_options.is_empty()
            && !self
                .kv_policy_options
                .contains(&self.default_selection.kv_policy)
        {
            issues.push(
                "DEFAULT_SELECTION_KV_NOT_ALLOWED: default K/V policy is not in kv_policy_options"
                    .to_string(),
            );
        }
        if !self
            .performance_options
            .iter()
            .any(|p| p.id == self.default_selection.performance_id)
        {
            issues.push(
                "DEFAULT_SELECTION_PERFORMANCE_NOT_ALLOWED: default performance is not in performance_options"
                    .to_string(),
            );
        }
        if let Some(moe) = self.default_selection.n_cpu_moe
            && !self.cpu_moe_options.is_empty()
            && !self.cpu_moe_options.contains(&moe)
        {
            issues.push(
                "DEFAULT_SELECTION_CPU_MOE_NOT_ALLOWED: default n_cpu_moe is not in cpu_moe_options"
                    .to_string(),
            );
        }

        // Independent axes may not synthesize a combination when the bundle
        // explicitly restricts resolution to operator-curated selections.
        if !self.allow_validated_custom
            && !self.curated_selections.contains(&self.default_selection)
        {
            issues.push(
                "DEFAULT_SELECTION_NOT_CURATED: default selection is not curated".to_string(),
            );
        }

        issues
    }

    /// Find the artifact for a given selection.
    pub fn artifact(&self, artifact_id: &str) -> Option<&PresetModelArtifact> {
        self.artifacts.iter().find(|a| a.id == artifact_id)
    }

    /// Find the weights artifact for the default selection.
    pub fn default_weights_artifact(&self) -> Option<&PresetModelArtifact> {
        let sel = &self.default_selection;
        self.artifact(&sel.artifact_id)
            .filter(|a| matches!(a.role, PresetArtifactRole::Weights))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bounded vec serde helpers
// ─────────────────────────────────────────────────────────────────────────────

pub fn bounded_serialize_vec<S, E>(v: &[E], s: S) -> Result<S::Ok, S::Error>
where
    S: SerdeSerializer,
    E: BoundedEnum,
{
    use serde::ser::SerializeSeq;
    let mut seq = s.serialize_seq(Some(v.len()))?;
    for e in v {
        seq.serialize_element(e.to_wire())?;
    }
    seq.end()
}

pub fn bounded_deserialize_vec<'de, D, E>(d: D) -> Result<Vec<E>, D::Error>
where
    D: SerdeDeserializer<'de>,
    E: BoundedEnum,
{
    let strs = Vec::<String>::deserialize(d)?;
    Ok(strs.into_iter().map(|s| E::from_wire(&s)).collect())
}

// ─────────────────────────────────────────────────────────────────────────────
// Flat compatibility projection
//
// When a bundle is present, the top-level ModelPreset fields are derived from
// default_selection. This is a read-time materialization, not a persistence
// boundary. Called by the API layer when serving GET/PUT responses.
// ─────────────────────────────────────────────────────────────────────────────

/// Materialize the flat projection fields from a bundle's default selection.
/// Updates `preset` in-place. Does not modify `preset.bundle`.
pub fn materialize_default_projection(preset: &mut ModelPreset) {
    if let Some(ref bundle) = preset.bundle {
        let sel = &bundle.default_selection;
        if let Some(weights) = bundle.default_weights_artifact()
            && let Some(path) = &weights.local_path
        {
            preset.model_path = path.clone();
        }
        preset.context_size = sel.context_size;

        // KV policy → ctk / ctv
        (preset.ctk, preset.ctv) = match &sel.kv_policy {
            LlamaKvPolicyId::F16F16 => ("f16".to_string(), "f16".to_string()),
            LlamaKvPolicyId::Q8Q8 => ("q8_0".to_string(), "q8_0".to_string()),
            LlamaKvPolicyId::Q4Q4 => ("q4_0".to_string(), "q4_0".to_string()),
            LlamaKvPolicyId::MixedQ8Q4 => ("q8_0".to_string(), "q4_0".to_string()),
            LlamaKvPolicyId::Unknown(pair) => {
                // Conversion stores unknown pairs as `k/v`, preserving the
                // exact wire value while still projecting both flat fields.
                pair.split_once('/')
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .unwrap_or_else(|| (pair.clone(), pair.clone()))
            }
        };

        // Performance option → batch_size / ubatch_size
        if let Some(perf) = bundle
            .performance_options
            .iter()
            .find(|p| p.id == sel.performance_id)
        {
            preset.batch_size = perf.batch_size;
            preset.ubatch_size = perf.ubatch_size;
        }

        // n_cpu_moe
        preset.n_cpu_moe = sel.n_cpu_moe;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The single server-owned v6 bundle constructor
//
// This is the ONLY place in the codebase that sets `fit_enabled` on a new
// bundle. All wizard, editor-conversion, and bundle-copy surfaces must call
// this function. Migration does NOT call it — migrated presets retain their
// stored fit_enabled value (which may be None).
//
// HARD GATE: `rg -n 'fit_enabled\s*:' src/presets/` must show exactly one
// assignment site for new bundles. This constructor is that site.
// ─────────────────────────────────────────────────────────────────────────────

/// Create a new ModelPreset with a v6 bundle via the single server-owned
/// constructor. All wizard surfaces, conversion endpoints, and copy endpoints
/// must call this function instead of constructing the bundle inline.
///
/// Sets `fit_enabled` to `Some(false)` — the only assignment site for new
/// bundles. Migration must NOT call this function; a migrated preset keeps
/// its stored `fit_enabled` value (which may be `None`).
pub fn create_bundle_preset(name: &str, bundle: PresetBundleSpec) -> ModelPreset {
    // First, materialize a base preset from the default selection.
    let mut preset = ModelPreset {
        id: crate::presets::next_id(),
        name: name.to_string(),
        schema_version: Some(6),
        revision: 1,
        backend: crate::inference::InferenceBackend::LlamaCpp,
        bundle: Some(bundle.clone()),
        fit_enabled: Some(false),
        ..Default::default()
    };

    materialize_default_projection(&mut preset);

    preset
}

// ─────────────────────────────────────────────────────────────────────────────
// Structural validation (pure; no binary lookup)
// ─────────────────────────────────────────────────────────────────────────────

/// Structural validation for a v6 bundle. Returns issues if the bundle is
/// not internally consistent. Called at load and before save.
pub fn validate_bundle_structural(preset: &ModelPreset) -> Vec<String> {
    match &preset.bundle {
        None => Vec::new(),
        Some(bundle) => {
            let mut issues = bundle.structural_issues();

            // Check conflicting flat projection: if bundle is present, flat
            // model_path/ctx/ctk/ctv/batch/ubatch/n_cpu_moe must be consistent
            // with default_selection. A conflict means a second configuration
            // was written and must be rejected with a 400 field error.
            let sel = &bundle.default_selection;
            let perf = bundle
                .performance_options
                .iter()
                .find(|p| p.id == sel.performance_id);

            if let Some(weights) = bundle.default_weights_artifact()
                && let Some(path) = &weights.local_path
                && preset.model_path != *path
            {
                issues.push(
                    "FLAT_PROJECTION_CONFLICT: preset.model_path does not match bundle default_selection artifact local_path"
                        .to_string(),
                );
            }

            if preset.context_size != sel.context_size {
                issues.push(
                    "FLAT_PROJECTION_CONFLICT: preset.context_size does not match bundle default_selection context_size"
                        .to_string(),
                );
            }

            if let Some(perf) = perf {
                if preset.batch_size != perf.batch_size {
                    issues.push(
                        "FLAT_PROJECTION_CONFLICT: preset.batch_size does not match bundle default_selection performance_option"
                            .to_string(),
                    );
                }
                if preset.ubatch_size != perf.ubatch_size {
                    issues.push(
                        "FLAT_PROJECTION_CONFLICT: preset.ubatch_size does not match bundle default_selection performance_option"
                            .to_string(),
                    );
                }
            }

            issues
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_bundle() -> PresetBundleSpec {
        let weights = PresetModelArtifact {
            id: "weights-q4".into(),
            role: PresetArtifactRole::Weights,
            local_path: Some("/models/q4.gguf".into()),
            ..Default::default()
        };
        let performance = PresetPerformanceOption {
            id: "balanced".into(),
            label: "2048 / 256".into(),
            batch_size: 2048,
            ubatch_size: 256,
        };
        let selection = PresetBundleSelection {
            artifact_id: "weights-q4".into(),
            context_size: 160_000,
            kv_policy: LlamaKvPolicyId::Q4Q4,
            performance_id: "balanced".into(),
            n_cpu_moe: Some(0),
            intent_source: None,
        };
        PresetBundleSpec {
            identity: PresetBundleIdentity {
                bundle_id: "bundle_test".into(),
                tune_id: "tune_test".into(),
                display_name: "Test bundle".into(),
            },
            artifacts: vec![weights],
            context_options: vec![160_000, 200_000, 262_144],
            kv_policy_options: vec![LlamaKvPolicyId::Q4Q4],
            performance_options: vec![performance],
            cpu_moe_options: vec![0],
            curated_selections: vec![selection.clone()],
            allow_validated_custom: false,
            workload_policy: PresetWorkloadPolicy::GeneralChat,
            default_selection: selection,
            ..Default::default()
        }
    }

    #[test]
    fn bundle_unknown_enums_and_extensions_round_trip() {
        let value = serde_json::json!({
            "identity": {"bundle_id": "b", "tune_id": "t", "display_name": "Future"},
            "artifacts": [{
                "id": "w", "role": "future-role", "display_name": "weights",
                "local_path": "/models/future.gguf",
                "future_artifact_field": {"preserve": true}
            }],
            "context_options": [160000, 200000, 262144],
            "kv_policy_options": ["future_kv"],
            "performance_options": [{"id": "p", "label": "2048/256", "batch_size": 2048, "ubatch_size": 256}],
            "cpu_moe_options": [0, 6, 16],
            "curated_selections": [{"artifact_id": "w", "context_size": 160000, "kv_policy": "future_kv", "performance_id": "p", "n_cpu_moe": 0}],
            "allow_validated_custom": true,
            "workload_policy": "future-workload",
            "default_selection": {"artifact_id": "w", "context_size": 160000, "kv_policy": "future_kv", "performance_id": "p", "n_cpu_moe": 0},
            "future_bundle_field": "kept"
        });
        let bundle: PresetBundleSpec = serde_json::from_value(value.clone()).unwrap();
        assert!(
            matches!(bundle.artifacts[0].role, PresetArtifactRole::Unknown(ref v) if v == "future-role")
        );
        assert!(
            matches!(bundle.kv_policy_options[0], LlamaKvPolicyId::Unknown(ref v) if v == "future_kv")
        );
        assert!(
            matches!(bundle.workload_policy, PresetWorkloadPolicy::Unknown(ref v) if v == "future-workload")
        );
        let encoded = serde_json::to_value(bundle).unwrap();
        assert_eq!(encoded["future_bundle_field"], "kept");
        assert_eq!(
            encoded["artifacts"][0]["future_artifact_field"]["preserve"],
            true
        );
        assert_eq!(encoded["artifacts"][0]["role"], "future-role");
        assert_eq!(encoded["default_selection"]["kv_policy"], "future_kv");
    }

    #[test]
    fn structural_validation_catches_duplicate_and_projection_errors() {
        let mut bundle = valid_bundle();
        bundle.artifacts.push(bundle.artifacts[0].clone());
        bundle
            .performance_options
            .push(bundle.performance_options[0].clone());
        let mut preset = create_bundle_preset("Test", bundle);
        preset.batch_size = 99;
        let issues = validate_bundle_structural(&preset);
        assert!(
            issues
                .iter()
                .any(|issue| issue.starts_with("DUPLICATE_ARTIFACT_ID"))
        );
        assert!(
            issues
                .iter()
                .any(|issue| issue.starts_with("DUPLICATE_PERFORMANCE_ID"))
        );
        assert!(
            issues
                .iter()
                .any(|issue| issue.starts_with("FLAT_PROJECTION_CONFLICT"))
        );
    }

    #[test]
    fn bundle_constructor_sets_fit_off_and_projects_default() {
        let preset = create_bundle_preset("Test", valid_bundle());
        assert_eq!(preset.schema_version, Some(6));
        assert_eq!(preset.revision, 1);
        assert_eq!(preset.fit_enabled, Some(false));
        assert_eq!(preset.model_path, "/models/q4.gguf");
        assert_eq!(preset.context_size, 160_000);
        assert_eq!(preset.ctk, "q4_0");
        assert_eq!(preset.ctv, "q4_0");
        assert_eq!(preset.batch_size, 2048);
        assert_eq!(preset.ubatch_size, 256);
        assert_eq!(preset.n_cpu_moe, Some(0));
    }

    #[test]
    fn request_from_bundle_uses_default_projection_without_mutating_preset() {
        let preset = create_bundle_preset("Test", valid_bundle());
        let request = crate::inference::launch::request_from_preset(&preset, Some(9123)).unwrap();
        let crate::inference::launch::LocalLaunchRequest::LlamaCpp(config) = request else {
            panic!("expected llama.cpp request");
        };
        assert_eq!(config.model_path, "/models/q4.gguf");
        assert_eq!(config.context_size, 160_000);
        assert_eq!(config.ctk, "q4_0");
        assert_eq!(config.ctv, "q4_0");
        assert_eq!(config.batch_size, 2048);
        assert_eq!(config.ubatch_size, 256);
        assert_eq!(config.port, 9123);
        assert!(preset.bundle.is_some());
    }
}
