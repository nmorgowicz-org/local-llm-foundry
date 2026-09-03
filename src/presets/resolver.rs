//! The single pure resolver for typed preset bundles.
//!
//! Resolution deliberately consumes only persisted preset metadata and the
//! caller-provided capability snapshot. It never reads an artifact, probes a
//! binary, or performs network I/O.

use crate::inference::llama_cpp_capabilities::CapabilitySnapshot;
use crate::llama::vram_estimator::VramBreakdown;
use crate::presets::validation::ValidationIssue;
use crate::presets::{ModelPreset, bundle};
use bundle::BoundedEnum;
use bundle::{
    LlamaKvPolicyId, PresetArtifactRole, PresetBundleSelection, PresetBundleSpec, PresetModelKind,
};
use sha2::{Digest, Sha256};

pub type OneShotSelection = PresetBundleSelection;
/// Phase 4 owns the concrete estimator response. This marker keeps the
/// internal contract typed without making Phase 3 serialize estimator state.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LaunchEstimate {
    #[serde(flatten)]
    pub breakdown: VramBreakdown,
    pub method: String,
    pub probe_device_total_mib: u64,
    pub probe_host_total_mib: u64,
    pub divergence: EstimateDivergence,
    pub additions: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct EstimateDivergence {
    pub model_mib: i64,
    pub context_mib: i64,
    pub compute_mib: i64,
    pub within_tolerance: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceMatch {
    pub class: String,
    pub summary: String,
    /// Receipt detail for a details view. `None` when the match's underlying
    /// `LaunchObservationReceipt` could not be carried (kept optional so a
    /// future caller can still build an `EvidenceMatch` from just class/summary).
    pub detail: Option<EvidenceDetail>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceDetail {
    pub method: String,
    pub before_bytes: u64,
    pub peak_bytes: u64,
    pub after_bytes: u64,
    pub model_delta_bytes: Option<u64>,
    pub sample_count: u32,
    pub interval_ms: u32,
    pub noise_flags: Vec<String>,
    pub captured_unix_ms: u128,
}

#[derive(Debug, Clone)]
pub struct ResolvedLaunch {
    /// A flat launch envelope. Bundle authority is removed so a one-shot
    /// selection cannot be re-materialized as the saved default downstream.
    pub preset: ModelPreset,
    pub selection_hash: String,
    pub config_hash: String,
    pub changes: Vec<ResolvedChange>,
    pub estimate: Option<LaunchEstimate>,
    pub estimate_status: EstimateStatus,
    pub evidence: Option<EvidenceMatch>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ResolvedChange {
    pub code: String,
    pub field: String,
    pub before: Option<String>,
    pub after: String,
    pub explanation: String,
    pub source_policy: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
#[allow(clippy::large_enum_variant)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum EstimateStatus {
    Available { estimate: LaunchEstimate },
    Unavailable { code: String, message: String },
    NotApplicable { code: String },
}

/// Resolve a preset plus an optional one-shot selection into a flat launch
/// configuration. This function is pure with respect to the filesystem.
pub fn resolve_preset(
    preset: &ModelPreset,
    selection: Option<&PresetBundleSelection>,
    capabilities: &CapabilitySnapshot,
) -> Result<ResolvedLaunch, Vec<ValidationIssue>> {
    let mut issues =
        crate::presets::validation::validate_llama_launch_policy(preset, Some(capabilities));
    if let Err(error) = crate::inference::launch::validate_preset_backend_config(preset) {
        issues.push(issue(
            "backend",
            "INVALID_BACKEND_CONFIG",
            error.to_string(),
        ));
    }

    let Some(bundle) = preset.bundle.as_ref() else {
        if selection.is_some() {
            issues.push(issue(
                "selection",
                "PRESET_NOT_BUNDLED",
                "a one-shot selection requires a bundled llama.cpp preset",
            ));
        }
        if !issues.is_empty() {
            return Err(issues);
        }
        return Ok(build_result(preset, preset.clone(), None, Vec::new()));
    };

    issues.extend(
        bundle::validate_bundle_structural(preset)
            .into_iter()
            .map(|message| issue("bundle", "INVALID_BUNDLE", message)),
    );
    let requested = selection.unwrap_or(&bundle.default_selection);

    if selection.is_none() {
        let mut projected = preset.clone();
        bundle::materialize_default_projection(&mut projected);
        projection_conflicts(preset, &projected, &mut issues);
    }

    let exact = resolve_exact_selection(bundle, requested, Some(capabilities));
    for (code, message) in exact.block_codes.iter().zip(exact.reasons.iter()) {
        issues.push(issue("selection", code, message.clone()));
    }
    issues.extend(
        validate_runtime_selection(bundle, requested, capabilities, cfg!(target_os = "macos"))
            .into_iter()
            .map(|message| {
                let (code, detail) = message
                    .split_once(": ")
                    .unwrap_or((message.as_str(), message.as_str()));
                issue("selection", code, detail.to_string())
            }),
    );
    validate_typed_runtime_fields(preset, capabilities, &mut issues);
    if !issues.is_empty() {
        return Err(issues);
    }

    let (effective, changes) = materialize_selection(preset, bundle, requested);
    Ok(build_result(preset, effective, Some(requested), changes))
}

pub fn materialize_default_projection(
    preset: &ModelPreset,
    capabilities: &CapabilitySnapshot,
) -> Result<ResolvedLaunch, Vec<ValidationIssue>> {
    resolve_preset(preset, None, capabilities)
}

fn build_result(
    source: &ModelPreset,
    mut effective: ModelPreset,
    selection: Option<&PresetBundleSelection>,
    changes: Vec<ResolvedChange>,
) -> ResolvedLaunch {
    let selection_hash = selection_hash(source, &effective, selection);
    let config_hash = config_hash(source, &effective, selection);
    effective.bundle = None;
    ResolvedLaunch {
        preset: effective,
        selection_hash,
        config_hash,
        changes,
        estimate: None,
        estimate_status: EstimateStatus::NotApplicable {
            code: "not_requested".into(),
        },
        evidence: None,
    }
}

fn materialize_selection(
    source: &ModelPreset,
    bundle: &PresetBundleSpec,
    selection: &PresetBundleSelection,
) -> (ModelPreset, Vec<ResolvedChange>) {
    let mut effective = source.clone();
    let before = effective.clone();
    if let Some(weights) = bundle
        .artifact(&selection.artifact_id)
        .filter(|artifact| artifact.role == PresetArtifactRole::Weights)
    {
        if let Some(path) = &weights.local_path {
            effective.model_path = path.clone();
        }
        effective.mmproj = weights
            .mmproj_artifact_id
            .as_deref()
            .and_then(|id| bundle.artifact(id))
            .and_then(|artifact| artifact.local_path.clone());
        effective.draft_model = weights
            .draft_artifact_id
            .as_deref()
            .and_then(|id| bundle.artifact(id))
            .and_then(|artifact| artifact.local_path.clone())
            .unwrap_or_default();
    }
    effective.context_size = selection.context_size;
    (effective.ctk, effective.ctv) = kv_pair(&selection.kv_policy);
    if let Some(performance) = bundle
        .performance_options
        .iter()
        .find(|option| option.id == selection.performance_id)
    {
        effective.batch_size = performance.batch_size;
        effective.ubatch_size = performance.ubatch_size;
    }
    effective.n_cpu_moe = selection.n_cpu_moe;

    let mut changes = Vec::new();
    change(
        &mut changes,
        "artifact_changed",
        "model_path",
        &before.model_path,
        &effective.model_path,
        "artifact selection",
        None,
    );
    change(
        &mut changes,
        "context_changed",
        "context_size",
        &before.context_size.to_string(),
        &effective.context_size.to_string(),
        "context selection",
        None,
    );
    change(
        &mut changes,
        "kv_policy_changed",
        "ctk/ctv",
        &format!("{}/{}", before.ctk, before.ctv),
        &format!("{}/{}", effective.ctk, effective.ctv),
        "K/V policy selection",
        None,
    );
    change(
        &mut changes,
        "performance_changed",
        "batch_size/ubatch_size",
        &format!("{}/{}", before.batch_size, before.ubatch_size),
        &format!("{}/{}", effective.batch_size, effective.ubatch_size),
        "performance selection",
        None,
    );
    change(
        &mut changes,
        "cpu_moe_changed",
        "n_cpu_moe",
        &format_option(before.n_cpu_moe),
        &format_option(effective.n_cpu_moe),
        "MoE placement selection",
        None,
    );
    (effective, changes)
}

fn change(
    changes: &mut Vec<ResolvedChange>,
    code: &str,
    field: &str,
    before: &str,
    after: &str,
    explanation: &str,
    source_policy: Option<String>,
) {
    if before != after {
        changes.push(ResolvedChange {
            code: code.into(),
            field: field.into(),
            before: Some(before.into()),
            after: after.into(),
            explanation: explanation.into(),
            source_policy,
        });
    }
}

fn projection_conflicts(
    original: &ModelPreset,
    projected: &ModelPreset,
    issues: &mut Vec<ValidationIssue>,
) {
    for (field, before, after) in [
        ("model_path", &original.model_path, &projected.model_path),
        ("ctk", &original.ctk, &projected.ctk),
        ("ctv", &original.ctv, &projected.ctv),
    ] {
        if before != after {
            issues.push(issue(
                field,
                "FLAT_PROJECTION_CONFLICT",
                format!("{field} does not match bundle default_selection"),
            ));
        }
    }
    if original.context_size != projected.context_size {
        issues.push(issue(
            "context_size",
            "FLAT_PROJECTION_CONFLICT",
            "context_size does not match bundle default_selection",
        ));
    }
    if original.batch_size != projected.batch_size || original.ubatch_size != projected.ubatch_size
    {
        issues.push(issue(
            "batch_size/ubatch_size",
            "FLAT_PROJECTION_CONFLICT",
            "performance fields do not match bundle default_selection",
        ));
    }
    if original.n_cpu_moe != projected.n_cpu_moe {
        issues.push(issue(
            "n_cpu_moe",
            "FLAT_PROJECTION_CONFLICT",
            "n_cpu_moe does not match bundle default_selection",
        ));
    }
}

fn validate_typed_runtime_fields(
    preset: &ModelPreset,
    snapshot: &CapabilitySnapshot,
    issues: &mut Vec<ValidationIssue>,
) {
    use crate::inference::llama_cpp_capabilities::FeatureState;
    if let Some(value) = preset.mmproj_offload {
        let capability = &snapshot.typed.mmproj_offload;
        let supported = if value {
            &capability.positive
        } else {
            &capability.negative
        };
        if matches!(supported, FeatureState::Unavailable(_)) {
            issues.push(issue(
                "mmproj_offload",
                "CAPABILITY_UNAVAILABLE",
                "the selected binary does not advertise the requested mmproj offload flag",
            ));
        }
    }
    if !matches!(
        preset.llama_reasoning_effort,
        crate::inference::llama_cpp::LlamaReasoningEffort::Default
    ) && !snapshot.supports_reasoning_effort(
        preset
            .llama_reasoning_effort
            .as_flag_value()
            .unwrap_or_default(),
    ) {
        issues.push(issue(
            "llama_reasoning_effort",
            "CAPABILITY_UNAVAILABLE",
            "the selected binary does not advertise this reasoning effort",
        ));
    }
    if let Some(format) = &preset.llama_reasoning_format
        && !snapshot.supports_reasoning_format(format.as_flag_value().unwrap_or_default())
    {
        issues.push(issue(
            "llama_reasoning_format",
            "CAPABILITY_UNAVAILABLE",
            "the selected binary does not advertise this reasoning format",
        ));
    }
    if preset.llama_reasoning_preserve == Some(true)
        && matches!(
            snapshot.typed.reasoning_preserve.positive,
            FeatureState::Unavailable(_)
        )
    {
        issues.push(issue(
            "llama_reasoning_preserve",
            "CAPABILITY_UNAVAILABLE",
            "reasoning preservation is not supported by the selected binary",
        ));
    }
}

fn issue(field: &str, code: &str, message: impl Into<String>) -> ValidationIssue {
    ValidationIssue {
        field: field.into(),
        code: code.into(),
        message: message.into(),
        repair: None,
    }
}

fn format_option(value: Option<i32>) -> String {
    value.map_or_else(|| "None".into(), |value| value.to_string())
}

fn kv_pair(policy: &LlamaKvPolicyId) -> (String, String) {
    match policy {
        LlamaKvPolicyId::F16F16 => ("f16".into(), "f16".into()),
        LlamaKvPolicyId::Q8Q8 => ("q8_0".into(), "q8_0".into()),
        LlamaKvPolicyId::Q4Q4 => ("q4_0".into(), "q4_0".into()),
        LlamaKvPolicyId::MixedQ8Q4 => ("q8_0".into(), "q4_0".into()),
        LlamaKvPolicyId::Unknown(value) => value.split_once('/').map_or_else(
            || (value.clone(), value.clone()),
            |(k, v)| (k.into(), v.into()),
        ),
    }
}

fn selection_hash(
    source: &ModelPreset,
    effective: &ModelPreset,
    selection: Option<&PresetBundleSelection>,
) -> String {
    digest("sel-v1:", selection_fields(source, effective, selection))
}

fn config_hash(
    source: &ModelPreset,
    preset: &ModelPreset,
    selection: Option<&PresetBundleSelection>,
) -> String {
    let mut fields = effective_launch_fields(preset);
    if let Some(bundle) = source.bundle.as_ref() {
        fields.push(triple(
            "/bundle/workload_policy",
            serde_json::json!(bundle.workload_policy.to_wire()),
        ));
        let selected = selection.unwrap_or(&bundle.default_selection);
        fields.push(triple(
            "/bundle/artifact_id",
            serde_json::json!(selected.artifact_id),
        ));
        fields.push(triple(
            "/bundle/performance_id",
            serde_json::json!(selected.performance_id),
        ));
        fields.push(triple(
            "/bundle/kv_policy",
            serde_json::json!(selected.kv_policy.to_wire()),
        ));
    }
    digest("cfg-v1:", fields)
}

fn selection_fields(
    source: &ModelPreset,
    effective: &ModelPreset,
    selection: Option<&PresetBundleSelection>,
) -> Vec<serde_json::Value> {
    let mut fields = effective_launch_fields(effective)
        .into_iter()
        .filter(|field| {
            // A bundle's artifact ID is the portable model identity. Its local
            // path and companion paths are machine-local implementation detail.
            !matches!(
                field[0].as_str(),
                Some("/model_path" | "/mmproj" | "/draft_model")
            )
        })
        .collect::<Vec<_>>();
    let selected = selection.or_else(|| {
        source
            .bundle
            .as_ref()
            .map(|bundle| &bundle.default_selection)
    });
    if let Some(bundle) = source.bundle.as_ref() {
        fields.push(triple(
            "/bundle_id",
            serde_json::json!(bundle.identity.bundle_id),
        ));
        fields.push(triple(
            "/tune_id",
            serde_json::json!(bundle.identity.tune_id),
        ));
        fields.push(triple(
            "/workload_policy",
            serde_json::json!(bundle.workload_policy.to_wire()),
        ));
    }
    if let Some(selection) = selected {
        fields.push(triple(
            "/artifact_id",
            serde_json::json!(selection.artifact_id),
        ));
        fields.push(triple(
            "/context_size",
            serde_json::json!(selection.context_size),
        ));
        fields.push(triple(
            "/kv_policy",
            serde_json::json!(selection.kv_policy.to_wire()),
        ));
        fields.push(triple(
            "/performance_id",
            serde_json::json!(selection.performance_id),
        ));
        if let Some(n_cpu_moe) = selection.n_cpu_moe {
            fields.push(triple("/n_cpu_moe", serde_json::json!(n_cpu_moe)));
        }
    } else {
        fields.push(triple(
            "/model_path",
            serde_json::json!(effective.model_path),
        ));
    }
    fields
}

/// Serialize the effective launch surface into the canonical typed-triple
/// representation. Keeping this list derived from `ModelPreset` makes newly
/// persisted behavior fields participate in `cfg-v1` instead of silently
/// creating a consent hash that describes only a subset of argv.
fn effective_launch_fields(preset: &ModelPreset) -> Vec<serde_json::Value> {
    const EXCLUDED: &[&str] = &[
        "id",
        "name",
        "schema_version",
        "revision",
        "api_key",
        "api_key_configured",
        "clear_api_key",
        "bundle",
        "rapid_mlx",
        "hf_repo",
        "cache_type_k",
        "cache_type_v",
        "gguf_architecture",
        "param_count",
        "family",
        "size_class",
        "architecture_kind",
        "expert_count",
        "expert_used_count",
        "active_params_b",
        "block_count",
        "bytes_per_layer",
        "expert_bytes_per_layer",
    ];
    let value = serde_json::to_value(preset).expect("model preset serializes");
    let serde_json::Value::Object(fields) = value else {
        return Vec::new();
    };
    fields
        .into_iter()
        .filter(|(name, value)| !EXCLUDED.contains(&name.as_str()) && !value.is_null())
        .map(|(name, value)| triple(&format!("/{name}"), value))
        .collect()
}

fn triple(path: &str, value: serde_json::Value) -> serde_json::Value {
    let type_name = match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    };
    serde_json::json!([path, type_name, value])
}

fn digest(prefix: &str, mut fields: Vec<serde_json::Value>) -> String {
    fields.sort_by(|left, right| left[0].as_str().cmp(&right[0].as_str()));
    let bytes = serde_json::to_vec(&fields).expect("canonical resolver fields serialize");
    let hex = Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("{prefix}{hex}")
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ResolvedSelection {
    pub selection: PresetBundleSelection,
    pub launchable: bool,
    pub block_codes: Vec<String>,
    pub reasons: Vec<String>,
}

pub fn validate_runtime_selection(
    bundle: &PresetBundleSpec,
    selection: &PresetBundleSelection,
    snapshot: &CapabilitySnapshot,
    unified_memory: bool,
) -> Vec<String> {
    let mut issues = Vec::new();
    if selection.kv_policy == LlamaKvPolicyId::MixedQ8Q4 && !snapshot.mixed_main_kv.supported {
        issues.push("MIXED_MAIN_KV_UNSUPPORTED".to_string());
    }
    if let Some(artifact) = bundle.artifact(&selection.artifact_id) {
        let metadata = &artifact.metadata;
        if let Some(n_cpu_moe) = selection.n_cpu_moe {
            if n_cpu_moe < 0 {
                issues.push("N_CPU_MOE_NEGATIVE: n_cpu_moe must be zero or positive".into());
            }
            if n_cpu_moe > 0 {
                if unified_memory {
                    issues.push("N_CPU_MOE_UNIFIED_MEMORY_UNQUALIFIED: CPU expert placement is not qualified on unified-memory systems".into());
                }
                if matches!(metadata.model_kind, PresetModelKind::Dense) {
                    issues.push(
                        "N_CPU_MOE_DENSE_MODEL: CPU expert placement requires a proven MoE model"
                            .into(),
                    );
                }
                if matches!(metadata.model_kind, PresetModelKind::Unknown(_))
                    || metadata.moe_layer_count.is_none()
                {
                    issues.push("N_CPU_MOE_METADATA_UNKNOWN: CPU expert placement requires authoritative MoE layer metadata".into());
                } else if metadata
                    .moe_layer_count
                    .is_some_and(|layers| n_cpu_moe as u32 > layers)
                {
                    issues.push(
                        "N_CPU_MOE_EXCEEDS_LAYER_COUNT: n_cpu_moe exceeds GGUF MoE layer count"
                            .into(),
                    );
                }
            }
        }
    }
    issues
}

pub fn resolve_exact_selection(
    bundle: &PresetBundleSpec,
    selection: &PresetBundleSelection,
    snapshot: Option<&CapabilitySnapshot>,
) -> ResolvedSelection {
    let mut block_codes = Vec::new();
    let mut reasons = Vec::new();
    match bundle.artifact(&selection.artifact_id) {
        None => {
            block_codes.push("artifact_not_found".into());
            reasons.push(format!(
                "artifact '{}' is not present in the bundle",
                selection.artifact_id
            ));
        }
        Some(artifact) if artifact.role != PresetArtifactRole::Weights => {
            block_codes.push("artifact_not_weights".into());
            reasons.push("only Weights artifacts may be selected".into());
        }
        Some(artifact) if artifact.local_path.is_none() => {
            block_codes.push("artifact_not_local".into());
            reasons.push(format!(
                "artifact '{}' has no adopted local path",
                selection.artifact_id
            ));
        }
        _ => {}
    }
    if !bundle.context_options.is_empty()
        && !bundle.context_options.contains(&selection.context_size)
    {
        block_codes.push("context_not_allowed".into());
        reasons.push("selected context is not in the bundle catalog".into());
    }
    if !bundle.kv_policy_options.is_empty()
        && !bundle.kv_policy_options.contains(&selection.kv_policy)
    {
        block_codes.push("kv_policy_not_allowed".into());
        reasons.push("selected K/V policy is not in the bundle catalog".into());
    }
    if matches!(selection.kv_policy, LlamaKvPolicyId::Unknown(_)) {
        block_codes.push("unknown_kv_policy".into());
        reasons.push("unknown K/V policy is preserved but not launchable".into());
    }
    if matches!(
        bundle.workload_policy,
        bundle::PresetWorkloadPolicy::AgenticTools | bundle::PresetWorkloadPolicy::Unknown(_)
    ) && matches!(selection.kv_policy, LlamaKvPolicyId::Q4Q4)
    {
        block_codes.push("kv_policy_ineligible".into());
        reasons.push("q4_0/q4_0 is not eligible for the selected workload quality floor".into());
    }

    if !bundle.performance_options.is_empty()
        && !bundle
            .performance_options
            .iter()
            .any(|option| option.id == selection.performance_id)
    {
        block_codes.push("performance_not_allowed".into());
        reasons.push("selected performance option is not in the bundle catalog".into());
    }
    if let Some(value) = selection.n_cpu_moe
        && !bundle.cpu_moe_options.is_empty()
        && !bundle.cpu_moe_options.contains(&value)
    {
        block_codes.push("cpu_moe_not_allowed".into());
        reasons.push("selected n_cpu_moe is not in the bundle catalog".into());
    }
    if !bundle.allow_validated_custom
        && !bundle
            .curated_selections
            .iter()
            .any(|curated| same_selection_axes(curated, selection))
    {
        block_codes.push("selection_not_curated".into());
        reasons.push("bundle permits only curated selections".into());
    }
    if matches!(
        bundle.workload_policy,
        bundle::PresetWorkloadPolicy::Unknown(_)
    ) {
        block_codes.push("unknown_workload_policy".into());
        reasons.push("unknown workload policy cannot authorize a launch".into());
    }
    if selection.kv_policy == LlamaKvPolicyId::MixedQ8Q4
        && !snapshot.is_some_and(|value| value.mixed_main_kv.supported)
    {
        block_codes.push("MIXED_MAIN_KV_UNSUPPORTED".into());
        reasons.push("mixed K/V requires a binary advertising mixed_main_kv support".into());
    }
    ResolvedSelection {
        selection: selection.clone(),
        launchable: block_codes.is_empty(),
        block_codes,
        reasons,
    }
}

fn same_selection_axes(left: &PresetBundleSelection, right: &PresetBundleSelection) -> bool {
    left.artifact_id == right.artifact_id
        && left.context_size == right.context_size
        && left.kv_policy == right.kv_policy
        && left.performance_id == right.performance_id
        && left.n_cpu_moe == right.n_cpu_moe
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::llama_cpp_capabilities::FeatureState;
    use crate::presets::bundle::{
        PresetArtifactMetadata, PresetArtifactQuantization, PresetPerformanceOption,
        PresetWorkloadPolicy,
    };

    #[derive(Debug, serde::Deserialize)]
    struct GoldenFingerprint {
        name: String,
        bundle_id: String,
        tune_id: String,
        workload_policy: String,
        tensor_split: String,
        api_key: Option<String>,
        selection: PresetBundleSelection,
        expected_selection_hash: String,
        expected_config_hash: String,
    }

    fn snapshot() -> CapabilitySnapshot {
        CapabilitySnapshot::product_default()
    }

    fn bundle() -> PresetBundleSpec {
        let weights = bundle::PresetModelArtifact {
            id: "weights".into(),
            role: PresetArtifactRole::Weights,
            local_path: Some("/models/q4.gguf".into()),
            quantization: PresetArtifactQuantization::default(),
            metadata: PresetArtifactMetadata {
                model_kind: PresetModelKind::Dense,
                ..Default::default()
            },
            ..Default::default()
        };
        let selection = PresetBundleSelection {
            artifact_id: "weights".into(),
            context_size: 160_000,
            kv_policy: LlamaKvPolicyId::Q4Q4,
            performance_id: "balanced".into(),
            n_cpu_moe: Some(0),
            intent_source: None,
        };
        PresetBundleSpec {
            artifacts: vec![weights],
            context_options: vec![160_000, 200_000],
            kv_policy_options: vec![LlamaKvPolicyId::Q4Q4],
            performance_options: vec![PresetPerformanceOption {
                id: "balanced".into(),
                label: "2048/256".into(),
                batch_size: 2048,
                ubatch_size: 256,
            }],
            cpu_moe_options: vec![0],
            curated_selections: vec![selection.clone()],
            default_selection: selection,
            ..Default::default()
        }
    }

    #[test]
    fn exact_selection_resolves_to_flat_fields_and_same_default_projection() {
        let mut preset = bundle::create_bundle_preset("Qwen", bundle());
        preset.bundle.as_mut().unwrap().identity.bundle_id = "qwen".into();
        let caps = snapshot();
        let selection = preset.bundle.as_ref().unwrap().default_selection.clone();
        let explicit = resolve_preset(&preset, Some(&selection), &caps).unwrap();
        let defaulted = materialize_default_projection(&preset, &caps).unwrap();
        assert_eq!(explicit.preset.model_path, "/models/q4.gguf");
        assert_eq!(explicit.preset.batch_size, 2048);
        assert_eq!(explicit.preset.bundle, None);
        assert_eq!(explicit.preset.model_path, defaulted.preset.model_path);
        assert_eq!(explicit.preset.context_size, defaulted.preset.context_size);
        assert_eq!(explicit.preset.ctk, defaulted.preset.ctk);
        assert_eq!(explicit.preset.ctv, defaulted.preset.ctv);
        assert_eq!(explicit.preset.batch_size, defaulted.preset.batch_size);
        assert_eq!(explicit.preset.ubatch_size, defaulted.preset.ubatch_size);
        assert_eq!(explicit.preset.n_cpu_moe, defaulted.preset.n_cpu_moe);
        assert_eq!(explicit.selection_hash, defaulted.selection_hash);
        assert_eq!(explicit.config_hash, defaulted.config_hash);
    }

    #[test]
    fn unknown_artifact_and_context_are_rejected() {
        let preset = bundle::create_bundle_preset("Qwen", bundle());
        let mut selection = preset.bundle.as_ref().unwrap().default_selection.clone();
        selection.artifact_id = "missing".into();
        selection.context_size = 999;
        let issues = resolve_preset(&preset, Some(&selection), &snapshot()).unwrap_err();
        assert!(issues.iter().any(|i| i.code == "artifact_not_found"));
        assert!(issues.iter().any(|i| i.code == "context_not_allowed"));
    }

    #[test]
    fn hash_excludes_api_key_and_revision_but_changes_for_behavior() {
        let mut preset = bundle::create_bundle_preset("Qwen", bundle());
        let caps = snapshot();
        let first = resolve_preset(&preset, None, &caps).unwrap();
        assert_eq!(
            first.selection_hash,
            "sel-v1:2b3082957ed8b0dd67c863817a564d19b0aa0508a4e83dd52173121d19222b58"
        );
        assert_eq!(
            first.config_hash,
            "cfg-v1:56e63cf42513fab8ecfe52dce62f524ee08bdb462f707cbc398ceb230fadf74b"
        );
        preset.api_key = Some("secret".into());
        preset.revision += 1;
        let second = resolve_preset(&preset, None, &caps).unwrap();
        assert_eq!(first.selection_hash, second.selection_hash);
        assert_eq!(first.config_hash, second.config_hash);
        preset.no_cont_batching = true;
        let third = resolve_preset(&preset, None, &caps).unwrap();
        assert_ne!(first.config_hash, third.config_hash);
        preset.bundle.as_mut().unwrap().workload_policy =
            bundle::PresetWorkloadPolicy::RoleplayCreative;
        let fourth = resolve_preset(&preset, None, &caps).unwrap();
        assert_ne!(third.config_hash, fourth.config_hash);
    }

    #[test]
    fn intent_source_does_not_change_selection_hash() {
        let preset = bundle::create_bundle_preset("Qwen", bundle());
        let mut selection = preset.bundle.as_ref().unwrap().default_selection.clone();
        let first = resolve_preset(&preset, Some(&selection), &snapshot()).unwrap();
        selection.intent_source = Some(bundle::PresetFitIntent::LowVram);
        let second = resolve_preset(&preset, Some(&selection), &snapshot()).unwrap();
        assert_eq!(first.selection_hash, second.selection_hash);
    }

    /// Fixture 6 (Phase 10a): vision bundle with mmproj offload capability
    /// on/off/unavailable. `mmproj_offload` is a flat field on ModelPreset
    /// carried through bundle resolution unchanged; capability gating comes
    /// from `TypedLlamaCapabilities::mmproj_offload`, checked directly
    /// against whichever polarity (`positive`/`negative`) was requested.
    #[test]
    fn mmproj_offload_on_is_accepted_when_the_binary_advertises_it() {
        let mut preset = bundle::create_bundle_preset("Vision", bundle());
        preset.mmproj_offload = Some(true);
        let mut caps = snapshot();
        caps.typed.mmproj_offload.positive = FeatureState::Available;
        let result = resolve_preset(&preset, None, &caps);
        assert!(result.is_ok(), "expected success, got {result:?}");
    }

    #[test]
    fn mmproj_offload_off_is_accepted_when_the_binary_advertises_it() {
        let mut preset = bundle::create_bundle_preset("Vision", bundle());
        preset.mmproj_offload = Some(false);
        let mut caps = snapshot();
        caps.typed.mmproj_offload.negative = FeatureState::Available;
        let result = resolve_preset(&preset, None, &caps);
        assert!(result.is_ok(), "expected success, got {result:?}");
    }

    #[test]
    fn mmproj_offload_is_rejected_when_the_binary_does_not_advertise_it() {
        let mut preset = bundle::create_bundle_preset("Vision", bundle());
        preset.mmproj_offload = Some(true);
        // product_default() leaves typed.mmproj_offload.positive Unavailable.
        let caps = snapshot();
        let issues = resolve_preset(&preset, None, &caps).unwrap_err();
        assert!(
            issues.iter().any(|i| i.code == "CAPABILITY_UNAVAILABLE"),
            "expected CAPABILITY_UNAVAILABLE, got {issues:?}"
        );
    }

    fn moe_bundle_with_metadata(metadata: PresetArtifactMetadata) -> PresetBundleSpec {
        let mut spec = bundle();
        spec.artifacts[0].metadata = metadata;
        spec
    }

    #[test]
    fn n_cpu_moe_is_blocked_on_a_dense_model() {
        let bundle = moe_bundle_with_metadata(PresetArtifactMetadata {
            model_kind: PresetModelKind::Dense,
            moe_layer_count: None,
            ..Default::default()
        });
        let mut selection = bundle.default_selection.clone();
        selection.n_cpu_moe = Some(4);
        let issues = validate_runtime_selection(&bundle, &selection, &snapshot(), false);
        assert!(
            issues
                .iter()
                .any(|i| i.starts_with("N_CPU_MOE_DENSE_MODEL"))
        );
    }

    #[test]
    fn n_cpu_moe_is_blocked_on_unified_memory() {
        let bundle = moe_bundle_with_metadata(PresetArtifactMetadata {
            model_kind: PresetModelKind::Moe,
            moe_layer_count: Some(16),
            ..Default::default()
        });
        let mut selection = bundle.default_selection.clone();
        selection.n_cpu_moe = Some(4);
        let issues = validate_runtime_selection(&bundle, &selection, &snapshot(), true);
        assert!(
            issues
                .iter()
                .any(|i| i.starts_with("N_CPU_MOE_UNIFIED_MEMORY_UNQUALIFIED"))
        );
    }

    /// Fixture 8 (Phase 10a): degraded/unknown GGUF metadata. An artifact
    /// whose model_kind is Unknown, or whose moe_layer_count could not be
    /// read, must block CPU expert placement with an explanation rather than
    /// silently guessing it is safe.
    #[test]
    fn n_cpu_moe_is_blocked_when_metadata_is_unknown_or_degraded() {
        let unknown_kind = moe_bundle_with_metadata(PresetArtifactMetadata {
            model_kind: PresetModelKind::Unknown("gguf_parse_failed".into()),
            moe_layer_count: Some(16),
            ..Default::default()
        });
        let mut selection = unknown_kind.default_selection.clone();
        selection.n_cpu_moe = Some(4);
        let issues = validate_runtime_selection(&unknown_kind, &selection, &snapshot(), false);
        assert!(
            issues
                .iter()
                .any(|i| i.starts_with("N_CPU_MOE_METADATA_UNKNOWN")),
            "unknown model_kind must block n_cpu_moe: {issues:?}"
        );

        let missing_layer_count = moe_bundle_with_metadata(PresetArtifactMetadata {
            model_kind: PresetModelKind::Moe,
            moe_layer_count: None,
            ..Default::default()
        });
        let mut selection = missing_layer_count.default_selection.clone();
        selection.n_cpu_moe = Some(4);
        let issues =
            validate_runtime_selection(&missing_layer_count, &selection, &snapshot(), false);
        assert!(
            issues
                .iter()
                .any(|i| i.starts_with("N_CPU_MOE_METADATA_UNKNOWN")),
            "missing moe_layer_count must block n_cpu_moe: {issues:?}"
        );
    }

    #[test]
    fn n_cpu_moe_is_blocked_when_it_exceeds_the_gguf_layer_count() {
        let bundle = moe_bundle_with_metadata(PresetArtifactMetadata {
            model_kind: PresetModelKind::Moe,
            moe_layer_count: Some(4),
            ..Default::default()
        });
        let mut selection = bundle.default_selection.clone();
        selection.n_cpu_moe = Some(8);
        let issues = validate_runtime_selection(&bundle, &selection, &snapshot(), false);
        assert!(
            issues
                .iter()
                .any(|i| i.starts_with("N_CPU_MOE_EXCEEDS_LAYER_COUNT"))
        );
    }

    #[test]
    fn n_cpu_moe_is_accepted_with_authoritative_moe_metadata() {
        let bundle = moe_bundle_with_metadata(PresetArtifactMetadata {
            model_kind: PresetModelKind::Moe,
            moe_layer_count: Some(16),
            ..Default::default()
        });
        let mut selection = bundle.default_selection.clone();
        selection.n_cpu_moe = Some(4);
        let issues = validate_runtime_selection(&bundle, &selection, &snapshot(), false);
        assert!(issues.is_empty(), "unexpected issues: {issues:?}");
    }

    fn golden_preset(fixture: &GoldenFingerprint) -> ModelPreset {
        let mut bundle = bundle();
        bundle.identity.bundle_id = fixture.bundle_id.clone();
        bundle.identity.tune_id = fixture.tune_id.clone();
        bundle.workload_policy = PresetWorkloadPolicy::from_wire(&fixture.workload_policy);
        bundle.default_selection = fixture.selection.clone();
        bundle.curated_selections = vec![fixture.selection.clone()];
        bundle.kv_policy_options = vec![fixture.selection.kv_policy.clone()];
        let mut preset = bundle::create_bundle_preset(&fixture.name, bundle);
        preset.tensor_split = fixture.tensor_split.clone();
        preset.api_key = fixture.api_key.clone();
        preset
    }

    #[test]
    fn fingerprint_golden_fixture_matches_committed_literals() {
        let fixtures: Vec<GoldenFingerprint> = serde_json::from_str(include_str!(
            "../../tests/fixtures/presets/fingerprint_golden.json"
        ))
        .expect("fingerprint golden fixture parses");
        assert!(fixtures.len() >= 3);

        for fixture in &fixtures {
            let preset = golden_preset(fixture);
            let resolved = resolve_preset(&preset, None, &snapshot()).unwrap();
            assert_eq!(
                resolved.selection_hash, fixture.expected_selection_hash,
                "selection hash drifted for {}",
                fixture.name
            );
            assert_eq!(
                resolved.config_hash, fixture.expected_config_hash,
                "config hash drifted for {}",
                fixture.name
            );
        }
    }

    #[test]
    fn same_selection_same_fingerprint_across_surfaces() {
        let fixture = GoldenFingerprint {
            name: "cross-surface".into(),
            bundle_id: "bundle-cross-surface".into(),
            tune_id: "tune-cross-surface".into(),
            workload_policy: "general_chat".into(),
            tensor_split: "1,0".into(),
            api_key: None,
            selection: bundle().default_selection,
            expected_selection_hash: String::new(),
            expected_config_hash: String::new(),
        };
        let preset = golden_preset(&fixture);
        let selection = preset.bundle.as_ref().unwrap().default_selection.clone();
        let caps = snapshot();

        // These are the four authoritative paths: preview, collapsed-card
        // saved-default projection, direct selection spawn, and saved-default
        // spawn. All must consume the same resolver contract.
        let preview = resolve_preset(&preset, Some(&selection), &caps).unwrap();
        let card_default = materialize_default_projection(&preset, &caps).unwrap();
        let direct_spawn = resolve_preset(&preset, Some(&selection), &caps).unwrap();
        let saved_default_spawn = resolve_preset(&preset, None, &caps).unwrap();
        let fingerprints = [
            (&preview.selection_hash, &preview.config_hash),
            (&card_default.selection_hash, &card_default.config_hash),
            (&direct_spawn.selection_hash, &direct_spawn.config_hash),
            (
                &saved_default_spawn.selection_hash,
                &saved_default_spawn.config_hash,
            ),
        ];
        assert!(fingerprints.windows(2).all(|pair| pair[0] == pair[1]));
    }
}
