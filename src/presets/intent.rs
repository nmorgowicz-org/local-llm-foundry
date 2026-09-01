//! Deterministic, estimate-only fit intents for preset bundles.
//!
//! An intent produces one exact selection. It is never persisted implicitly;
//! callers must explicitly save the returned selection or start it once.

use crate::inference::llama_cpp_capabilities::CapabilitySnapshot;
use crate::llama::vram_estimator::{
    Backend, EstimateEvidence, EstimatorOptions, ModelArch, VramBreakdown, full_estimate,
};
use crate::presets::ModelPreset;
use crate::presets::bundle::{
    BoundedEnum, LlamaKvPolicyId, PresetArtifactRole, PresetBundleSelection, PresetBundleSpec,
    PresetDigestCoverage, PresetFitIntent, PresetPerformanceOption, PresetWorkloadPolicy,
};
use crate::presets::resolver::{ResolvedChange, resolve_preset};

#[derive(Debug, Clone)]
pub struct IntentContext<'a> {
    pub preset: &'a ModelPreset,
    pub capabilities: &'a CapabilitySnapshot,
    pub arch: ModelArch,
    pub model_size_bytes: u64,
    pub available_vram_bytes: u64,
    pub available_ram_bytes: u64,
    pub is_unified_memory: bool,
    pub gpu_layers: i32,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct IntentUnavailable {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct IntentProposal {
    pub intent: PresetFitIntent,
    pub selection: PresetBundleSelection,
    pub changes: Vec<ResolvedChange>,
    pub estimate: Option<VramBreakdown>,
    pub unavailable: Option<IntentUnavailable>,
}

#[derive(Debug, Clone)]
struct Candidate {
    estimate: Option<VramBreakdown>,
}

/// Produce a deterministic proposal from persisted metadata and caller-owned
/// memory inputs. Iteration is explicitly sorted; no filesystem, network, or
/// binary probing occurs here.
pub fn propose_intent(
    context: &IntentContext<'_>,
    intent: PresetFitIntent,
) -> Result<IntentProposal, IntentUnavailable> {
    let Some(bundle) = context.preset.bundle.as_ref() else {
        return Err(unavailable(
            "preset_not_bundled",
            "fit intents require a bundled preset",
        ));
    };
    let default = bundle.default_selection.clone();
    let mut selection = default.clone();
    selection.intent_source = Some(intent.clone());
    if !exact_local_artifact(bundle, &selection.artifact_id) {
        return Err(unavailable(
            "artifact_metadata_incomplete",
            "automatic fit requires a local artifact with exact size and SHA-256 digest",
        ));
    }
    if let Some(kv_policy) = preferred_kv(bundle, &intent) {
        selection.kv_policy = kv_policy;
    }
    let mut evaluated = candidate(context, bundle, &selection);

    if !matches!(intent, PresetFitIntent::QualityFirst) {
        // The ordering is contractual: artifact, context, performance, then
        // MoE placement. Each stage is stable and only considers explicit
        // bundle options.
        if let Some(lower) = lower_artifact(bundle, &selection.artifact_id) {
            selection.artifact_id = lower;
            evaluated = candidate(context, bundle, &selection);
        }
        if !fits(context, evaluated.estimate.as_ref())
            && let Some(smaller) =
                context_option(bundle, &selection.artifact_id, selection.context_size)
        {
            selection.context_size = smaller;
            evaluated = candidate(context, bundle, &selection);
        }
        if !fits(context, evaluated.estimate.as_ref())
            && let Some(performance) = smaller_performance(bundle, &selection.performance_id)
        {
            selection.performance_id = performance.id;
            evaluated = candidate(context, bundle, &selection);
        }
    }

    let mut unavailable_reason = None;
    if !fits(context, evaluated.estimate.as_ref()) {
        match (
            moe_option(context, bundle, selection.n_cpu_moe),
            context.is_unified_memory,
        ) {
            (Some(_), true) => {
                unavailable_reason = Some(unavailable(
                    "n_cpu_moe_unified_memory_unqualified",
                    "automatic CPU expert placement is unavailable on unified memory",
                ));
            }
            (Some(moe), false) => {
                selection.n_cpu_moe = Some(moe);
                evaluated = candidate(context, bundle, &selection);
            }
            (None, _) => {}
        }
    }

    // Curated-only bundles cannot receive a synthesized Cartesian product.
    if !bundle.allow_validated_custom
        && !bundle
            .curated_selections
            .iter()
            .any(|curated| same_axes(curated, &selection))
    {
        let curated = bundle
            .curated_selections
            .iter()
            .filter(|curated| policy_allows_kv(&bundle.workload_policy, &curated.kv_policy))
            .min_by_key(|curated| selection_cost(bundle, curated))
            .cloned()
            .ok_or_else(|| {
                unavailable(
                    "no_eligible_curated_selection",
                    "no curated selection satisfies the workload policy",
                )
            })?;
        selection = curated;
        selection.intent_source = Some(intent.clone());
        evaluated = candidate(context, bundle, &selection);
    }

    let resolved = resolve_preset(context.preset, Some(&selection), context.capabilities)
        .map_err(|issues| unavailable("intent_selection_invalid", &issues[0].message))?;
    let changes = resolved
        .changes
        .into_iter()
        .map(|mut change| {
            if change.source_policy.is_none() {
                change.source_policy = Some(bundle.workload_policy.to_wire().into());
            }
            change
        })
        .collect();
    Ok(IntentProposal {
        intent,
        selection,
        changes,
        estimate: evaluated.estimate,
        unavailable: unavailable_reason,
    })
}

fn candidate(
    context: &IntentContext<'_>,
    bundle: &PresetBundleSpec,
    selection: &PresetBundleSelection,
) -> Candidate {
    let estimate = bundle
        .artifact(&selection.artifact_id)
        .and_then(|artifact| artifact.size_bytes)
        .or_else(|| (context.model_size_bytes > 0).then_some(context.model_size_bytes))
        .filter(|size| *size > 0)
        .map(|size| {
            let performance = bundle
                .performance_options
                .iter()
                .find(|option| option.id == selection.performance_id);
            full_estimate(
                size,
                &context.arch,
                selection.context_size,
                kv_name(&selection.kv_policy).0,
                kv_name(&selection.kv_policy).1,
                context.preset.parallel_slots,
                performance.map_or(context.preset.ubatch_size, |option| option.ubatch_size),
                selection.n_cpu_moe.unwrap_or(0),
                context.gpu_layers,
                context.available_vram_bytes,
                context.available_ram_bytes,
                context.is_unified_memory,
                EstimatorOptions {
                    backend: Backend::LlamaCpp,
                    evidence: EstimateEvidence::Measured,
                    ..Default::default()
                },
            )
        });
    Candidate { estimate }
}

fn fits(context: &IntentContext<'_>, estimate: Option<&VramBreakdown>) -> bool {
    estimate.is_some_and(|estimate| {
        context.available_vram_bytes > 0
            && estimate.total_bytes <= context.available_vram_bytes
            && (context.is_unified_memory
                || context.available_ram_bytes == 0
                || estimate.ram_bytes <= context.available_ram_bytes)
    })
}

fn lower_artifact(bundle: &PresetBundleSpec, current: &str) -> Option<String> {
    let current_size = bundle
        .artifact(current)
        .and_then(|artifact| artifact.size_bytes)?;
    let mut artifacts = bundle
        .artifacts
        .iter()
        .filter(|artifact| {
            matches!(artifact.role, PresetArtifactRole::Weights)
                && artifact.local_path.is_some()
                && artifact.size_bytes.is_some()
                && artifact.digest.as_ref().is_some_and(|digest| {
                    digest.algorithm.eq_ignore_ascii_case("sha256")
                        && !digest.value.is_empty()
                        && digest.coverage == PresetDigestCoverage::FullFile
                })
        })
        .filter(|artifact| artifact.size_bytes.is_some_and(|size| size < current_size))
        .collect::<Vec<_>>();
    artifacts.sort_by(|left, right| {
        left.size_bytes
            .cmp(&right.size_bytes)
            .then_with(|| left.id.cmp(&right.id))
    });
    artifacts.first().map(|artifact| artifact.id.clone())
}

fn exact_local_artifact(bundle: &PresetBundleSpec, artifact_id: &str) -> bool {
    bundle.artifact(artifact_id).is_some_and(|artifact| {
        artifact.role == PresetArtifactRole::Weights
            && artifact.local_path.is_some()
            && artifact.size_bytes.is_some_and(|size| size > 0)
            && artifact.digest.as_ref().is_some_and(|digest| {
                digest.algorithm.eq_ignore_ascii_case("sha256")
                    && !digest.value.is_empty()
                    && digest.coverage == PresetDigestCoverage::FullFile
            })
    })
}

fn context_option(bundle: &PresetBundleSpec, artifact_id: &str, current: u64) -> Option<u64> {
    let native_limit = bundle
        .artifact(artifact_id)
        .and_then(|artifact| artifact.metadata.native_context_limit);
    let mut options = bundle
        .context_options
        .iter()
        .copied()
        .filter(|value| {
            *value > 0 && *value < current && native_limit.is_none_or(|limit| *value <= limit)
        })
        .collect::<Vec<_>>();
    options.sort_unstable();
    options.pop()
}

fn smaller_performance(
    bundle: &PresetBundleSpec,
    current: &str,
) -> Option<PresetPerformanceOption> {
    let current = bundle
        .performance_options
        .iter()
        .find(|option| option.id == current)?;
    let mut options = bundle
        .performance_options
        .iter()
        .filter(|option| option.ubatch_size <= option.batch_size)
        .filter(|option| {
            (option.batch_size, option.ubatch_size, &option.id)
                < (current.batch_size, current.ubatch_size, &current.id)
        })
        .collect::<Vec<_>>();
    options.sort_by(|left, right| {
        (left.batch_size, left.ubatch_size, &left.id).cmp(&(
            right.batch_size,
            right.ubatch_size,
            &right.id,
        ))
    });
    options.pop().cloned()
}

fn moe_option(
    context: &IntentContext<'_>,
    bundle: &PresetBundleSpec,
    current: Option<i32>,
) -> Option<i32> {
    if context.arch.n_experts == 0 || context.arch.moe_layer_count == 0 {
        return None;
    }
    let current = current.unwrap_or(0);
    let mut options = bundle
        .cpu_moe_options
        .iter()
        .copied()
        .filter(|value| *value > current && *value <= context.arch.moe_layer_count as i32)
        .collect::<Vec<_>>();
    options.sort_unstable();
    options.first().copied()
}

fn selection_cost(
    bundle: &PresetBundleSpec,
    selection: &PresetBundleSelection,
) -> (u64, u64, u32, i32, String) {
    let size = bundle
        .artifact(&selection.artifact_id)
        .and_then(|artifact| artifact.size_bytes)
        .unwrap_or(u64::MAX);
    let performance = bundle
        .performance_options
        .iter()
        .find(|option| option.id == selection.performance_id);
    (
        size,
        selection.context_size,
        performance.map_or(u32::MAX, |option| option.batch_size),
        selection.n_cpu_moe.unwrap_or(0),
        selection.artifact_id.clone(),
    )
}

fn policy_allows_kv(policy: &PresetWorkloadPolicy, kv: &LlamaKvPolicyId) -> bool {
    !matches!(
        (policy, kv),
        (PresetWorkloadPolicy::AgenticTools, LlamaKvPolicyId::Q4Q4)
            | (
                PresetWorkloadPolicy::AgenticTools,
                LlamaKvPolicyId::MixedQ8Q4
            )
            | (_, LlamaKvPolicyId::MixedQ8Q4)
            | (PresetWorkloadPolicy::Unknown(_), LlamaKvPolicyId::Q4Q4)
    )
}

fn preferred_kv(bundle: &PresetBundleSpec, intent: &PresetFitIntent) -> Option<LlamaKvPolicyId> {
    let order = match intent {
        PresetFitIntent::QualityFirst => [
            LlamaKvPolicyId::F16F16,
            LlamaKvPolicyId::Q8Q8,
            LlamaKvPolicyId::Q4Q4,
            LlamaKvPolicyId::MixedQ8Q4,
        ],
        PresetFitIntent::Balanced => [
            LlamaKvPolicyId::Q8Q8,
            LlamaKvPolicyId::F16F16,
            LlamaKvPolicyId::Q4Q4,
            LlamaKvPolicyId::MixedQ8Q4,
        ],
        PresetFitIntent::LowVram | PresetFitIntent::Unknown(_) => [
            LlamaKvPolicyId::Q4Q4,
            LlamaKvPolicyId::Q8Q8,
            LlamaKvPolicyId::F16F16,
            LlamaKvPolicyId::MixedQ8Q4,
        ],
    };
    order.into_iter().find(|kv| {
        bundle.kv_policy_options.contains(kv) && policy_allows_kv(&bundle.workload_policy, kv)
    })
}

fn kv_name(policy: &LlamaKvPolicyId) -> (&str, &str) {
    match policy {
        LlamaKvPolicyId::F16F16 => ("f16", "f16"),
        LlamaKvPolicyId::Q8Q8 => ("q8_0", "q8_0"),
        LlamaKvPolicyId::Q4Q4 => ("q4_0", "q4_0"),
        LlamaKvPolicyId::MixedQ8Q4 => ("q8_0", "q4_0"),
        LlamaKvPolicyId::Unknown(value) => value
            .split_once('/')
            .unwrap_or((value.as_str(), value.as_str())),
    }
}

fn same_axes(left: &PresetBundleSelection, right: &PresetBundleSelection) -> bool {
    left.artifact_id == right.artifact_id
        && left.context_size == right.context_size
        && left.kv_policy == right.kv_policy
        && left.performance_id == right.performance_id
        && left.n_cpu_moe == right.n_cpu_moe
}

fn unavailable(code: &str, message: &str) -> IntentUnavailable {
    IntentUnavailable {
        code: code.into(),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presets::bundle::{
        PresetArtifactDigest, PresetArtifactMetadata, PresetArtifactQuantization,
        PresetDigestCoverage,
    };

    fn context(unified: bool, moe: bool) -> (ModelPreset, CapabilitySnapshot) {
        let mut artifact = crate::presets::bundle::PresetModelArtifact {
            id: "q4".into(),
            role: PresetArtifactRole::Weights,
            local_path: Some("/models/q4.gguf".into()),
            size_bytes: Some(8_000_000_000),
            digest: Some(PresetArtifactDigest {
                algorithm: "sha256".into(),
                value: "digest".into(),
                coverage: PresetDigestCoverage::FullFile,
                ..Default::default()
            }),
            metadata: PresetArtifactMetadata {
                model_kind: if moe {
                    crate::presets::bundle::PresetModelKind::Moe
                } else {
                    crate::presets::bundle::PresetModelKind::Dense
                },
                ..Default::default()
            },
            quantization: PresetArtifactQuantization::default(),
            ..Default::default()
        };
        let selection = PresetBundleSelection {
            artifact_id: "q4".into(),
            context_size: 32_000,
            kv_policy: LlamaKvPolicyId::Q8Q8,
            performance_id: "large".into(),
            n_cpu_moe: Some(0),
            intent_source: None,
        };
        let mut spec = PresetBundleSpec {
            artifacts: vec![artifact.clone()],
            context_options: vec![8_000, 32_000],
            kv_policy_options: vec![LlamaKvPolicyId::Q8Q8, LlamaKvPolicyId::Q4Q4],
            performance_options: vec![
                PresetPerformanceOption {
                    id: "small".into(),
                    batch_size: 512,
                    ubatch_size: 256,
                    ..Default::default()
                },
                PresetPerformanceOption {
                    id: "large".into(),
                    batch_size: 2_048,
                    ubatch_size: 512,
                    ..Default::default()
                },
            ],
            cpu_moe_options: if moe { vec![0, 4] } else { vec![0] },
            curated_selections: vec![selection.clone()],
            allow_validated_custom: true,
            default_selection: selection,
            ..Default::default()
        };
        if moe {
            artifact.metadata.moe_layer_count = Some(4);
            spec.artifacts = vec![artifact];
        }
        let preset = crate::presets::bundle::create_bundle_preset("intent", spec);
        (preset, CapabilitySnapshot::product_default())
    }

    #[test]
    fn low_vram_orders_context_and_performance_changes() {
        let (preset, capabilities) = context(false, false);
        let proposal = propose_intent(
            &IntentContext {
                preset: &preset,
                capabilities: &capabilities,
                arch: ModelArch::standard_heuristic(7.0),
                model_size_bytes: 8_000_000_000,
                available_vram_bytes: 1,
                available_ram_bytes: 0,
                is_unified_memory: false,
                gpu_layers: -1,
            },
            PresetFitIntent::LowVram,
        )
        .unwrap();
        assert_eq!(proposal.selection.context_size, 8_000);
        assert_eq!(proposal.selection.performance_id, "small");
        assert!(proposal.changes.iter().all(|change| {
            !change.code.is_empty()
                && !change.field.is_empty()
                && change.before.is_some()
                && !change.after.is_empty()
                && !change.explanation.is_empty()
                && change.source_policy.is_some()
        }));
    }

    #[test]
    fn agentic_policy_does_not_silently_select_q4_or_mixed_kv() {
        let (mut preset, capabilities) = context(false, false);
        preset.bundle.as_mut().unwrap().workload_policy = PresetWorkloadPolicy::AgenticTools;
        let proposal = propose_intent(
            &IntentContext {
                preset: &preset,
                capabilities: &capabilities,
                arch: ModelArch::standard_heuristic(7.0),
                model_size_bytes: 8_000_000_000,
                available_vram_bytes: 64_000_000_000,
                available_ram_bytes: 0,
                is_unified_memory: false,
                gpu_layers: -1,
            },
            PresetFitIntent::LowVram,
        )
        .unwrap();
        assert_eq!(proposal.selection.kv_policy, LlamaKvPolicyId::Q8Q8);
    }

    #[test]
    fn context_reduction_honors_native_context_limit() {
        let (mut preset, capabilities) = context(false, false);
        preset.bundle.as_mut().unwrap().artifacts[0]
            .metadata
            .native_context_limit = Some(8_000);
        let proposal = propose_intent(
            &IntentContext {
                preset: &preset,
                capabilities: &capabilities,
                arch: ModelArch::standard_heuristic(7.0),
                model_size_bytes: 8_000_000_000,
                available_vram_bytes: 1,
                available_ram_bytes: 0,
                is_unified_memory: false,
                gpu_layers: -1,
            },
            PresetFitIntent::LowVram,
        )
        .unwrap();
        assert_eq!(proposal.selection.context_size, 8_000);
    }

    #[test]
    fn unified_memory_moe_reports_unavailable_cpu_placement() {
        let (preset, capabilities) = context(true, true);
        let proposal = propose_intent(
            &IntentContext {
                preset: &preset,
                capabilities: &capabilities,
                arch: ModelArch {
                    n_experts: 8,
                    moe_layer_count: 4,
                    ..ModelArch::standard_heuristic(7.0)
                },
                model_size_bytes: 8_000_000_000,
                available_vram_bytes: 1,
                available_ram_bytes: 0,
                is_unified_memory: true,
                gpu_layers: -1,
            },
            PresetFitIntent::LowVram,
        )
        .unwrap();
        assert_eq!(
            proposal.unavailable.unwrap().code,
            "n_cpu_moe_unified_memory_unqualified"
        );
    }
}
