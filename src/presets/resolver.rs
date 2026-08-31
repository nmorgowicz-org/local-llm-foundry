//! Phase 2: bundle selection resolver — types + exact-selection stub.
//!
//! Architecture §7 makes one server-side resolver authoritative for preview,
//! save, and spawn:
//!
//! ```text
//! ModelPreset + optional one-shot selection
//!     -> structural/product validation (pure; no executable required)
//!     -> validate bundle membership and revision
//!     -> resolve named policies to exact values
//!     -> materialize flat ModelPreset
//!     -> runtime validation with an explicit CapabilitySnapshot/provider
//!     -> build canonical ResolvedLaunchManifest
//!     -> optionally enrich with estimate and evidence
//!     -> return internal ResolvedLaunch plus a separately redacted API view
//! ```
//!
//! This phase introduces the **types** and the **exact-selection** resolver
//! interface only. The intent/proposal algorithm (Low VRAM, architecture §11)
//! is deliberately not implemented yet ("no intent algorithm yet"). An *exact*
//! selection is the saved default or a one-shot override — a concrete choice,
//! not an unresolved intent, so resolving it performs membership and capability
//! support checks and never a proposal search.

use crate::inference::llama_cpp_capabilities::CapabilitySnapshot;

use super::bundle::{LlamaKvPolicyId, PresetArtifactRole, PresetBundleSelection, PresetBundleSpec};

/// A one-shot selection supplied by the client (Configure-drawer draft or
/// "Start without saving"). It is structurally identical to the stored default
/// selection; the server treats both the same way at resolve time.
pub type OneShotSelection = PresetBundleSelection;

/// The result of resolving one exact selection against a bundle and an
/// (optional) capability snapshot.
///
/// This is the "materialize the choice + explain launchability" boundary from
/// architecture §7. `block_codes` are stable, machine-readable codes the UI can
/// act on (repair links, disabled-option reasons); `reasons` are the matching
/// human-readable explanations. An empty `block_codes` means launchable (subject
/// to a successful launch-time binary check in a later phase).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ResolvedSelection {
    /// The exact selection that was resolved (normalized echo).
    pub selection: PresetBundleSelection,
    /// Whether the selection is launchable under the given capability snapshot.
    pub launchable: bool,
    /// Stable machine-readable codes explaining why it is not launchable
    /// (empty when launchable).
    pub block_codes: Vec<String>,
    /// Human-readable reasons, one per entry in `block_codes`.
    pub reasons: Vec<String>,
}

/// Validate selection values that depend on the exact executable and machine
/// topology. The topology is supplied by the caller rather than discovered
/// from global process state, keeping this boundary deterministic in tests.
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

    let artifact = bundle.artifact(&selection.artifact_id);
    if let Some(artifact) = artifact {
        let metadata = &artifact.metadata;
        if let Some(n_cpu_moe) = selection.n_cpu_moe
            && n_cpu_moe < 0
        {
            issues.push(
                "N_CPU_MOE_NEGATIVE: n_cpu_moe must be zero or a positive MoE layer count"
                    .to_string(),
            );
        }
        if let Some(n_cpu_moe) = selection.n_cpu_moe
            && n_cpu_moe > 0
        {
            if unified_memory {
                issues.push(
                    "N_CPU_MOE_UNIFIED_MEMORY_UNQUALIFIED: CPU expert placement is not qualified on unified-memory systems"
                        .to_string(),
                );
            }
            if matches!(metadata.model_kind, super::bundle::PresetModelKind::Dense) {
                issues.push(
                    "N_CPU_MOE_DENSE_MODEL: CPU expert placement requires a proven MoE model"
                        .to_string(),
                );
            }
            if matches!(
                metadata.model_kind,
                super::bundle::PresetModelKind::Unknown(_)
            ) || metadata.moe_layer_count.is_none()
            {
                issues.push(
                    "N_CPU_MOE_METADATA_UNKNOWN: CPU expert placement requires authoritative MoE layer metadata"
                        .to_string(),
                );
            } else if metadata
                .moe_layer_count
                .is_some_and(|layers| n_cpu_moe as u32 > layers)
            {
                issues.push(
                    "N_CPU_MOE_EXCEEDS_LAYER_COUNT: n_cpu_moe exceeds GGUF MoE layer count"
                        .to_string(),
                );
            }
        }
    }
    issues
}

/// Resolve an **exact** selection (the saved default or a one-shot override)
/// against the bundle's catalog and the supplied capability snapshot.
///
/// Pure for this phase. It performs the membership checks that architecture
/// §7 requires of an exact selection:
///
/// - the selected `artifact_id` must exist in the bundle and be a `Weights`
///   artifact (only `Weights` artifacts may be selected; companions are not);
/// - a `MixedQ8Q4` K/V policy is non-launchable unless the exact binary
///   advertises `mixed_main_kv` support (architecture §6).
///
/// Companion-reference resolution, batch/ubatch and `n_cpu_moe` bounds, context
/// limits, and the full flat-`ModelPreset` materialization are layered on in the
/// API/resolver phase. No intent/proposal algorithm is run here.
pub fn resolve_exact_selection(
    bundle: &PresetBundleSpec,
    selection: &PresetBundleSelection,
    snapshot: Option<&CapabilitySnapshot>,
) -> ResolvedSelection {
    let mut block_codes: Vec<String> = Vec::new();
    let mut reasons: Vec<String> = Vec::new();

    // Membership: the chosen artifact must exist and be a Weights artifact.
    match bundle.artifact(&selection.artifact_id) {
        None => {
            block_codes.push("artifact_not_found".to_string());
            reasons.push(format!(
                "artifact '{}' is not present in the bundle",
                selection.artifact_id
            ));
        }
        Some(artifact) if artifact.role != PresetArtifactRole::Weights => {
            block_codes.push("artifact_not_weights".to_string());
            reasons.push("only Weights artifacts may be selected".to_string());
        }
        _ => {}
    }

    if let Some(artifact) = bundle.artifact(&selection.artifact_id)
        && artifact.local_path.is_none()
    {
        block_codes.push("artifact_not_local".to_string());
        reasons.push(format!(
            "artifact '{}' has no adopted local path",
            selection.artifact_id
        ));
    }

    if !bundle.context_options.is_empty()
        && !bundle.context_options.contains(&selection.context_size)
    {
        block_codes.push("context_not_allowed".to_string());
        reasons.push("selected context is not in the bundle catalog".to_string());
    }
    if !bundle.kv_policy_options.is_empty()
        && !bundle.kv_policy_options.contains(&selection.kv_policy)
    {
        block_codes.push("kv_policy_not_allowed".to_string());
        reasons.push("selected K/V policy is not in the bundle catalog".to_string());
    }
    if matches!(selection.kv_policy, LlamaKvPolicyId::Unknown(_)) {
        block_codes.push("unknown_kv_policy".to_string());
        reasons.push("unknown K/V policy is preserved but not launchable".to_string());
    }
    if !bundle.performance_options.is_empty()
        && !bundle
            .performance_options
            .iter()
            .any(|option| option.id == selection.performance_id)
    {
        block_codes.push("performance_not_allowed".to_string());
        reasons.push("selected performance option is not in the bundle catalog".to_string());
    }
    if let Some(moe) = selection.n_cpu_moe
        && !bundle.cpu_moe_options.is_empty()
        && !bundle.cpu_moe_options.contains(&moe)
    {
        block_codes.push("cpu_moe_not_allowed".to_string());
        reasons.push("selected n_cpu_moe is not in the bundle catalog".to_string());
    }
    if !bundle.allow_validated_custom && !bundle.curated_selections.contains(selection) {
        block_codes.push("selection_not_curated".to_string());
        reasons.push("bundle permits only curated selections".to_string());
    }
    if matches!(
        bundle.workload_policy,
        super::bundle::PresetWorkloadPolicy::Unknown(_)
    ) {
        block_codes.push("unknown_workload_policy".to_string());
        reasons.push("unknown workload policy cannot authorize a launch".to_string());
    }

    // Capability: mixed main-K/V is non-launchable until the exact binary
    // advertises support. `None` snapshot means the probe is unavailable, which
    // is a degraded (still non-launchable) state for a mixed pair — it is never
    // silently accepted.
    if selection.kv_policy == LlamaKvPolicyId::MixedQ8Q4
        && !snapshot.is_some_and(|s| s.mixed_main_kv.supported)
    {
        block_codes.push("MIXED_MAIN_KV_UNSUPPORTED".to_string());
        reasons.push(
            "mixed K/V (q8_0/q4_0) requires a binary advertising mixed_main_kv support".to_string(),
        );
    }

    ResolvedSelection {
        selection: selection.clone(),
        launchable: block_codes.is_empty(),
        block_codes,
        reasons,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presets::bundle::{
        PresetArtifactMetadata, PresetArtifactQuantization, PresetArtifactRole,
    };

    fn weights_bundle() -> PresetBundleSpec {
        let mut spec = PresetBundleSpec::default();
        let mut weights = crate::presets::bundle::PresetModelArtifact::default();
        weights.id = "art_w".into();
        weights.role = PresetArtifactRole::Weights;
        weights.local_path = Some("/models/model.gguf".into());
        weights.quantization = PresetArtifactQuantization::default();
        weights.metadata = PresetArtifactMetadata::default();
        spec.artifacts = vec![weights];
        spec.allow_validated_custom = true;
        spec
    }

    #[test]
    fn exact_selection_missing_artifact_blocks() {
        let bundle = weights_bundle();
        let mut sel = PresetBundleSelection::default();
        sel.artifact_id = "missing".into();
        let res = resolve_exact_selection(&bundle, &sel, None);
        assert!(!res.launchable);
        assert!(res.block_codes.iter().any(|c| c == "artifact_not_found"));
    }

    #[test]
    fn exact_selection_valid_weights_launchable_without_snapshot() {
        let bundle = weights_bundle();
        let mut sel = PresetBundleSelection::default();
        sel.artifact_id = "art_w".into();
        let res = resolve_exact_selection(&bundle, &sel, None);
        assert!(res.launchable);
        assert!(res.block_codes.is_empty());
    }

    #[test]
    fn runtime_validator_blocks_unqualified_cpu_moe_on_unified_memory() {
        let mut bundle = weights_bundle();
        let artifact = bundle.artifacts.first_mut().unwrap();
        artifact.metadata.model_kind = crate::presets::bundle::PresetModelKind::Moe;
        artifact.metadata.moe_layer_count = Some(16);
        let mut selection = PresetBundleSelection::default();
        selection.artifact_id = "art_w".into();
        selection.n_cpu_moe = Some(6);
        let snapshot = CapabilitySnapshot {
            executable_identity: crate::inference::llama_cpp_capabilities::ExecutableIdentity {
                path: "/tmp/llama".into(),
                file_hash: "test".into(),
                file_mtime_unix: 0,
            },
            version_text: "test".into(),
            help_hash: "test".into(),
            serve_flags: Vec::new(),
            cache: Default::default(),
            context: Default::default(),
            concurrency: Default::default(),
            endpoints: Default::default(),
            streaming: Default::default(),
            templates: Default::default(),
            tools: Default::default(),
            speculation: Default::default(),
            typed: Default::default(),
            mixed_main_kv:
                crate::inference::llama_cpp_capabilities::MixedMainKv::product_default_denied(),
            evidence_timestamp: 0,
            source:
                crate::inference::llama_cpp_capabilities::CapabilitySnapshotSource::ManualOverride,
        };
        let issues = validate_runtime_selection(&bundle, &selection, &snapshot, true);
        assert!(
            issues
                .iter()
                .any(|issue| issue.starts_with("N_CPU_MOE_UNIFIED_MEMORY"))
        );
    }

    fn runtime_snapshot() -> CapabilitySnapshot {
        CapabilitySnapshot {
            executable_identity: crate::inference::llama_cpp_capabilities::ExecutableIdentity {
                path: "/tmp/llama".into(),
                file_hash: "test".into(),
                file_mtime_unix: 0,
            },
            version_text: "test".into(),
            help_hash: "test".into(),
            serve_flags: Vec::new(),
            cache: Default::default(),
            context: Default::default(),
            concurrency: Default::default(),
            endpoints: Default::default(),
            streaming: Default::default(),
            templates: Default::default(),
            tools: Default::default(),
            speculation: Default::default(),
            typed: Default::default(),
            mixed_main_kv:
                crate::inference::llama_cpp_capabilities::MixedMainKv::product_default_denied(),
            evidence_timestamp: 0,
            source:
                crate::inference::llama_cpp_capabilities::CapabilitySnapshotSource::ManualOverride,
        }
    }

    #[test]
    fn curated_only_rejects_non_curated_selection_and_validated_custom_allows_it() {
        let mut bundle = weights_bundle();
        let curated_selection = PresetBundleSelection {
            artifact_id: "art_w".into(),
            ..Default::default()
        };
        bundle.allow_validated_custom = false;
        bundle.curated_selections = vec![curated_selection.clone()];
        let mut selection = curated_selection;
        selection.context_size = 1234;

        let curated = resolve_exact_selection(&bundle, &selection, None);
        assert!(
            curated
                .block_codes
                .iter()
                .any(|code| code == "selection_not_curated")
        );

        bundle.allow_validated_custom = true;
        let custom = resolve_exact_selection(&bundle, &selection, None);
        assert!(
            !custom
                .block_codes
                .iter()
                .any(|code| code == "selection_not_curated")
        );
    }

    #[test]
    fn runtime_validator_rejects_negative_dense_unknown_and_over_bound_cpu_moe() {
        let snapshot = runtime_snapshot();
        let mut bundle = weights_bundle();
        {
            let artifact = bundle.artifacts.first_mut().unwrap();
            artifact.metadata.model_kind = crate::presets::bundle::PresetModelKind::Dense;
            artifact.metadata.moe_layer_count = Some(16);
        }

        let mut selection = PresetBundleSelection {
            artifact_id: "art_w".into(),
            ..Default::default()
        };
        selection.n_cpu_moe = Some(-1);
        let negative = validate_runtime_selection(&bundle, &selection, &snapshot, false);
        assert!(
            negative
                .iter()
                .any(|issue| issue.starts_with("N_CPU_MOE_NEGATIVE"))
        );

        selection.n_cpu_moe = Some(1);
        let dense = validate_runtime_selection(&bundle, &selection, &snapshot, false);
        assert!(
            dense
                .iter()
                .any(|issue| issue.starts_with("N_CPU_MOE_DENSE_MODEL"))
        );

        bundle.artifacts[0].metadata.model_kind =
            crate::presets::bundle::PresetModelKind::Unknown("future".into());
        let unknown = validate_runtime_selection(&bundle, &selection, &snapshot, false);
        assert!(
            unknown
                .iter()
                .any(|issue| issue.starts_with("N_CPU_MOE_METADATA_UNKNOWN"))
        );

        bundle.artifacts[0].metadata.model_kind = crate::presets::bundle::PresetModelKind::Moe;
        bundle.artifacts[0].metadata.moe_layer_count = Some(1);
        selection.n_cpu_moe = Some(2);
        let over_bound = validate_runtime_selection(&bundle, &selection, &snapshot, false);
        assert!(
            over_bound
                .iter()
                .any(|issue| issue.starts_with("N_CPU_MOE_EXCEEDS_LAYER_COUNT"))
        );
    }
}
