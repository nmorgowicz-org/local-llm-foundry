//! Deterministic, capability-gated llama.cpp candidate planning.

use super::{CalibrationCandidate, CalibrationWorkload, LlamaCppCalibrationPatch};
use crate::inference::InferenceBackend;
use crate::inference::launch::{
    LocalLaunchRequest, request_from_preset, validate_preset_backend_config,
};
use crate::inference::llama_cpp_capabilities::CapabilitySnapshot;
use crate::presets::ModelPreset;
use anyhow::{Result, anyhow, bail};

use super::design::{OrthogonalArray, generate};

pub const BALANCED_MAX_SCREEN_TRIALS: usize = 48;
pub const BALANCED_MAX_OA_ROWS: usize = 25;
/// Final confirmation includes the baseline plus the two strongest screen
/// survivors. The baseline is required at the full workload after screening.
pub const BALANCED_MAX_VERIFICATION_CANDIDATES: usize = 3;
pub const QUICK_MAX_VERIFICATION_CANDIDATES: usize = 2;
/// Dense-Qwen default screen ladder. The Phase 4 receipt found no trustworthy
/// improvement over 512/512 and marked 4096 high-noise, so 1536/4096 remain
/// explicit Thorough-mode follow-ups rather than default Balanced rows.
const BALANCED_BATCH_LEVELS: [u32; 3] = [512, 1024, 2048];

/// The deliberately small, typed factor surface used by Balanced v1.
///
/// Factors are selected from the preset and exact capability snapshot. No
/// arbitrary argv, environment variable, or filename-derived setting can
/// enter this catalog.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BalancedFactor {
    ContextSize,
    BatchSize,
    UBatchSize,
    Threads,
    FlashAttention,
}

#[derive(Debug, Clone)]
struct FactorSpec {
    factor: BalancedFactor,
    levels: Vec<LlamaCppCalibrationPatch>,
}

#[derive(Debug, Clone)]
pub struct BalancedPlan {
    pub array: OrthogonalArray,
    pub screen_trials: usize,
    pub final_rows: usize,
    pub verification_candidates: usize,
    pub candidates: Vec<CalibrationCandidate>,
}

/// Validate Balanced's bounded run-count contract. Callers must calculate the
/// design before consent; exceeding any limit is an error, never a promotion
/// to a larger design.
pub fn validate_balanced_budget(
    screen_trials: usize,
    final_rows: usize,
    verification_candidates: usize,
) -> Result<()> {
    if screen_trials > BALANCED_MAX_SCREEN_TRIALS {
        bail!(
            "Balanced screen requires {screen_trials} trials; maximum is {BALANCED_MAX_SCREEN_TRIALS}"
        );
    }
    if final_rows > BALANCED_MAX_OA_ROWS {
        bail!(
            "Balanced orthogonal design requires {final_rows} rows; maximum is {BALANCED_MAX_OA_ROWS}"
        );
    }
    if verification_candidates > BALANCED_MAX_VERIFICATION_CANDIDATES {
        bail!(
            "Balanced verification requires {verification_candidates} candidates; maximum is {BALANCED_MAX_VERIFICATION_CANDIDATES}"
        );
    }
    let total = screen_trials
        .saturating_add(final_rows)
        .saturating_add(verification_candidates)
        .saturating_add(1); // measured baseline control
    if total > BALANCED_MAX_SCREEN_TRIALS {
        bail!("Balanced plan requires {total} trials; maximum is {BALANCED_MAX_SCREEN_TRIALS}");
    }
    Ok(())
}

/// Build the bounded Quick candidate set without launching any process.
///
/// Every candidate is validated through the production preset/launch path.
/// Optional flags are only emitted when the exact runtime help snapshot proves
/// they exist; the baseline is always retained as the measured control.
pub fn quick_candidates(
    preset: &ModelPreset,
    workload: &CalibrationWorkload,
    _capabilities: Option<&CapabilitySnapshot>,
) -> Result<Vec<CalibrationCandidate>> {
    validate_common_inputs(preset, workload)?;

    let mut planned = vec![(
        "baseline".to_string(),
        LlamaCppCalibrationPatch::default(),
        vec!["baseline preset configuration".to_string()],
    )];

    let current_batch = effective_batch(preset.batch_size);
    let current_ubatch = effective_ubatch(preset.ubatch_size);
    if current_batch > 1024 && current_ubatch > 256 {
        planned.push((
            "bounded-batch".into(),
            LlamaCppCalibrationPatch {
                batch_size: Some(1024),
                ubatch_size: Some(256),
                ..Default::default()
            },
            vec!["bounded Quick batch alternative".into()],
        ));
    }

    validate_and_materialize(preset, workload, planned)
}

/// Build the deterministic Balanced orthogonal-array plan without launching
/// any process. Four numeric core factors use L9 by default; a verified
/// flash-attention runtime uses L25 while preserving the preset's flash mode.
pub fn balanced_plan(
    preset: &ModelPreset,
    workload: &CalibrationWorkload,
    capabilities: Option<&CapabilitySnapshot>,
) -> Result<BalancedPlan> {
    validate_common_inputs(preset, workload)?;
    let specs = factor_catalog(preset, workload, capabilities)?;
    let array = if capability_has(capabilities, "-fa", "--flash-attn") {
        OrthogonalArray::L25
    } else {
        OrthogonalArray::L9
    };
    let final_rows = array.rows();
    let batch_coverage_trials = BALANCED_BATCH_LEVELS.len();
    validate_balanced_budget(
        batch_coverage_trials,
        final_rows,
        BALANCED_MAX_VERIFICATION_CANDIDATES,
    )?;
    let design = generate(array, specs.len())?;

    let mut candidates = vec![materialize_candidate(
        preset,
        workload,
        "baseline".into(),
        LlamaCppCalibrationPatch::default(),
        vec!["baseline preset configuration".into()],
    )?];
    for (row_index, row) in design.iter().enumerate() {
        let mut patch = LlamaCppCalibrationPatch::default();
        let mut evidence = vec![format!("{} orthogonal-array row", array_name(array))];
        for (level, spec) in row.iter().zip(&specs) {
            let selected = &spec.levels[usize::from(*level) % spec.levels.len()];
            merge_patch(&mut patch, selected);
            evidence.push(format!("factor={:?},level={level}", spec.factor));
        }
        if candidates
            .iter()
            .any(|candidate| candidate.typed_patch == patch)
        {
            continue;
        }
        candidates.push(materialize_candidate(
            preset,
            workload,
            format!(
                "balanced-{}-r{row_index:02}",
                array_name(array).to_ascii_lowercase()
            ),
            patch,
            evidence,
        )?);
    }
    // L9 and L25 expose only three or five levels per factor. Keep explicit
    // coverage for all five product batch values so the upper end is measured
    // rather than silently omitted by the orthogonal-array mapping. Lower
    // ubatch when necessary to preserve llama.cpp's `ubatch <= batch` rule.
    let current_ubatch = effective_ubatch(preset.ubatch_size);
    for batch_size in BALANCED_BATCH_LEVELS {
        let patch = LlamaCppCalibrationPatch {
            batch_size: Some(batch_size),
            ubatch_size: Some(current_ubatch.min(batch_size)),
            ..Default::default()
        };
        if candidates
            .iter()
            .any(|candidate| candidate.typed_patch == patch)
        {
            continue;
        }
        candidates.push(materialize_candidate(
            preset,
            workload,
            format!("balanced-batch-{batch_size}"),
            patch,
            vec![format!("explicit batch coverage at {batch_size}")],
        )?);
    }

    Ok(BalancedPlan {
        array,
        screen_trials: batch_coverage_trials,
        final_rows,
        verification_candidates: BALANCED_MAX_VERIFICATION_CANDIDATES,
        candidates,
    })
}

pub fn balanced_candidates(
    preset: &ModelPreset,
    workload: &CalibrationWorkload,
    capabilities: Option<&CapabilitySnapshot>,
) -> Result<Vec<CalibrationCandidate>> {
    Ok(balanced_plan(preset, workload, capabilities)?.candidates)
}

fn validate_common_inputs(preset: &ModelPreset, workload: &CalibrationWorkload) -> Result<()> {
    if preset.backend != InferenceBackend::LlamaCpp {
        return Err(anyhow!("Calibration candidates require a llama.cpp preset"));
    }
    validate_preset_backend_config(preset)?;
    if workload.minimum_context == 0 {
        return Err(anyhow!("Calibration workload requires a positive context"));
    }
    if preset.context_size > 0 && preset.context_size < workload.minimum_context {
        return Err(anyhow!(
            "Preset context is below the Calibration workload minimum"
        ));
    }
    if effective_ubatch(preset.ubatch_size) > effective_batch(preset.batch_size) {
        return Err(anyhow!("Preset ubatch size cannot exceed batch size"));
    }
    Ok(())
}

fn factor_catalog(
    preset: &ModelPreset,
    workload: &CalibrationWorkload,
    _capabilities: Option<&CapabilitySnapshot>,
) -> Result<Vec<FactorSpec>> {
    let batch_levels: Vec<LlamaCppCalibrationPatch> = BALANCED_BATCH_LEVELS
        .into_iter()
        .map(|value| LlamaCppCalibrationPatch {
            batch_size: Some(value),
            ..Default::default()
        })
        .collect();
    let batch_floor = batch_levels
        .iter()
        .filter_map(|patch| patch.batch_size)
        .min()
        .unwrap_or_else(|| effective_batch(preset.batch_size));
    let ubatch_levels =
        numeric_u32_levels_bounded(effective_ubatch(preset.ubatch_size), batch_floor, |value| {
            LlamaCppCalibrationPatch {
                ubatch_size: Some(value),
                ..Default::default()
            }
        });
    let context_levels = numeric_u64_levels(
        preset.context_size.max(workload.minimum_context),
        workload.minimum_context,
        |value| LlamaCppCalibrationPatch {
            context_size: Some(value),
            ..Default::default()
        },
    );
    let thread_levels = numeric_i32_levels(preset.threads, |value| LlamaCppCalibrationPatch {
        threads: Some(value),
        ..Default::default()
    });

    let specs = vec![
        FactorSpec {
            factor: BalancedFactor::ContextSize,
            levels: context_levels,
        },
        FactorSpec {
            factor: BalancedFactor::BatchSize,
            levels: batch_levels,
        },
        FactorSpec {
            factor: BalancedFactor::UBatchSize,
            levels: ubatch_levels,
        },
        FactorSpec {
            factor: BalancedFactor::Threads,
            levels: thread_levels,
        },
    ];
    Ok(specs)
}

fn numeric_u32_levels_bounded<F>(
    baseline: u32,
    upper_bound: u32,
    make_patch: F,
) -> Vec<LlamaCppCalibrationPatch>
where
    F: Fn(u32) -> LlamaCppCalibrationPatch,
{
    let high = baseline.saturating_mul(2).min(upper_bound).max(1);
    let low = (baseline / 4).max(1).min(high);
    let half = (baseline / 2).max(low).min(high);
    let one_half = baseline
        .saturating_add(baseline / 2)
        .min(high)
        .max(baseline.min(high));
    vec![
        make_patch(low),
        make_patch(half),
        make_patch(baseline.min(upper_bound)),
        make_patch(one_half),
        make_patch(high),
    ]
}

fn numeric_u64_levels<F>(
    baseline: u64,
    minimum: u64,
    make_patch: F,
) -> Vec<LlamaCppCalibrationPatch>
where
    F: Fn(u64) -> LlamaCppCalibrationPatch,
{
    // The bounded Quick/Balanced planner measures context factors only through
    // 131K. A saved preset may legitimately request a larger server context;
    // keep that effective preset value in the baseline candidate while
    // clamping only the generated factor levels so the planner cannot panic on
    // an inverted `clamp(min, max)` range.
    let baseline = baseline.min(131_072).max(minimum);
    let low = minimum.min(baseline);
    let quarter = low.saturating_add(baseline.saturating_sub(low) / 4);
    let half = low.saturating_add(baseline.saturating_sub(low) / 2);
    let one_half = baseline.saturating_add(baseline.saturating_sub(low) / 2);
    let high = baseline.saturating_mul(2).clamp(baseline, 131_072);
    vec![
        make_patch(low),
        make_patch(quarter),
        make_patch(half),
        make_patch(one_half.min(high)),
        make_patch(high),
    ]
}

fn numeric_i32_levels<F>(baseline: Option<i32>, make_patch: F) -> Vec<LlamaCppCalibrationPatch>
where
    F: Fn(i32) -> LlamaCppCalibrationPatch,
{
    let effective = baseline.unwrap_or(2).max(1);
    let low = (effective / 4).max(1);
    let half = (effective / 2).max(low);
    let high = effective.saturating_mul(2).max(effective);
    let one_half = effective.saturating_add(effective / 2).max(effective);
    let middle = baseline.map_or_else(LlamaCppCalibrationPatch::default, &make_patch);
    vec![
        make_patch(low),
        make_patch(half),
        middle,
        make_patch(one_half),
        make_patch(high),
    ]
}

fn merge_patch(target: &mut LlamaCppCalibrationPatch, source: &LlamaCppCalibrationPatch) {
    if source.gpu_layers.is_some() {
        target.gpu_layers = source.gpu_layers;
    }
    if source.context_size.is_some() {
        target.context_size = source.context_size;
    }
    if source.threads.is_some() {
        target.threads = source.threads;
    }
    if source.threads_batch.is_some() {
        target.threads_batch = source.threads_batch;
    }
    if source.ctk.is_some() {
        target.ctk = source.ctk.clone();
    }
    if source.ctv.is_some() {
        target.ctv = source.ctv.clone();
    }
    if source.batch_size.is_some() {
        target.batch_size = source.batch_size;
    }
    if source.ubatch_size.is_some() {
        target.ubatch_size = source.ubatch_size;
    }
    if source.flash_attn.is_some() {
        target.flash_attn = source.flash_attn;
    }
    if source.n_cpu_moe.is_some() {
        target.n_cpu_moe = source.n_cpu_moe;
    }
}

fn materialize_candidate(
    preset: &ModelPreset,
    workload: &CalibrationWorkload,
    id: String,
    patch: LlamaCppCalibrationPatch,
    evidence: Vec<String>,
) -> Result<CalibrationCandidate> {
    let mut candidate_preset = preset.clone();
    super::executor::apply_patch_to_preset(&mut candidate_preset, &patch);
    if candidate_preset.context_size > 0 && candidate_preset.context_size < workload.minimum_context
    {
        bail!("Calibration candidate context is below the workload minimum");
    }
    if effective_ubatch(candidate_preset.ubatch_size) > effective_batch(candidate_preset.batch_size)
    {
        bail!("Calibration candidate ubatch size exceeds batch size");
    }
    validate_preset_backend_config(&candidate_preset)?;
    let launch = request_from_preset(&candidate_preset, None)?;
    if !matches!(launch, LocalLaunchRequest::LlamaCpp(_)) {
        return Err(anyhow!("Calibration candidate crossed backend boundary"));
    }
    Ok(CalibrationCandidate {
        id,
        typed_patch: patch,
        capability_evidence: evidence,
        predicted_memory_bytes: None,
    })
}

fn validate_and_materialize(
    preset: &ModelPreset,
    workload: &CalibrationWorkload,
    planned: Vec<(String, LlamaCppCalibrationPatch, Vec<String>)>,
) -> Result<Vec<CalibrationCandidate>> {
    planned
        .into_iter()
        .map(|(id, patch, evidence)| materialize_candidate(preset, workload, id, patch, evidence))
        .collect()
}

fn effective_batch(value: u32) -> u32 {
    if value == 0 { 2048 } else { value }
}

fn effective_ubatch(value: u32) -> u32 {
    if value == 0 { 512 } else { value }
}

fn array_name(array: OrthogonalArray) -> &'static str {
    match array {
        OrthogonalArray::L9 => "L9",
        OrthogonalArray::L25 => "L25",
    }
}

fn capability_has(capabilities: Option<&CapabilitySnapshot>, short: &str, long: &str) -> bool {
    capabilities.is_some_and(|snapshot| {
        snapshot
            .serve_flags
            .iter()
            .any(|flag| flag == short || flag == long)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::llama_cpp_capabilities::{
        CacheCapabilities, CapabilitySnapshot, CapabilitySnapshotSource, ConcurrencyCapabilities,
        ContextCapabilities, EndpointCapabilities, ExecutableIdentity, MixedMainKv,
        SpeculationCapabilities, StreamingCapabilities, TemplateCapabilities, ToolCapabilities,
    };

    fn snapshot_with_flash() -> CapabilitySnapshot {
        CapabilitySnapshot {
            executable_identity: ExecutableIdentity {
                path: "llama-server".into(),
                file_hash: "hash".into(),
                file_mtime_unix: 0,
            },
            version_text: "test".into(),
            help_hash: "help".into(),
            serve_flags: vec!["-fa".into()],
            cache: CacheCapabilities::default(),
            context: ContextCapabilities::default(),
            concurrency: ConcurrencyCapabilities::default(),
            endpoints: EndpointCapabilities::default(),
            streaming: StreamingCapabilities::default(),
            templates: TemplateCapabilities::default(),
            tools: ToolCapabilities::default(),
            speculation: SpeculationCapabilities::default(),
            typed: Default::default(),
            mixed_main_kv: MixedMainKv::product_default_denied(),
            evidence_timestamp: 0,
            source: CapabilitySnapshotSource::ManualOverride,
        }
    }

    #[test]
    fn quick_candidates_are_deterministic_and_capability_gated() {
        let mut preset = ModelPreset::default();
        preset.batch_size = 2048;
        preset.ubatch_size = 512;
        let workload = CalibrationWorkload::default();
        let without = quick_candidates(&preset, &workload, None).expect("baseline candidates");
        assert_eq!(
            without.iter().map(|c| c.id.as_str()).collect::<Vec<_>>(),
            ["baseline", "bounded-batch"]
        );
        let with = quick_candidates(&preset, &workload, Some(&snapshot_with_flash()))
            .expect("capability candidates");
        assert_eq!(
            with.iter()
                .map(|candidate| candidate.id.as_str())
                .collect::<Vec<_>>(),
            ["baseline", "bounded-batch"]
        );
    }

    #[test]
    fn balanced_plan_is_stable_and_within_budget() {
        let mut preset = ModelPreset::default();
        preset.batch_size = 2048;
        preset.ubatch_size = 512;
        let workload = CalibrationWorkload::default();
        let first =
            balanced_plan(&preset, &workload, Some(&snapshot_with_flash())).expect("balanced plan");
        let second =
            balanced_plan(&preset, &workload, Some(&snapshot_with_flash())).expect("balanced plan");
        assert_eq!(first.array, OrthogonalArray::L25);
        assert_eq!(first.final_rows, 25);
        assert_eq!(first.candidates, second.candidates);
        assert!(first.candidates.len() <= BALANCED_MAX_SCREEN_TRIALS);
        assert_eq!(first.candidates[0].id, "baseline");
        assert!(first.candidates.iter().all(|candidate| {
            let mut mapped = preset.clone();
            super::super::executor::apply_patch_to_preset(&mut mapped, &candidate.typed_patch);
            effective_ubatch(mapped.ubatch_size) <= effective_batch(mapped.batch_size)
        }));
    }

    #[test]
    fn balanced_plan_covers_batch_range_without_crossing_ubatch_bound() {
        let mut preset = ModelPreset::default();
        preset.batch_size = 2048;
        preset.ubatch_size = 2048;
        let plan =
            balanced_plan(&preset, &CalibrationWorkload::default(), None).expect("balanced plan");
        let mut observed = Vec::new();
        for candidate in &plan.candidates {
            let mut mapped = preset.clone();
            super::super::executor::apply_patch_to_preset(&mut mapped, &candidate.typed_patch);
            observed.push(effective_batch(mapped.batch_size));
            assert!(effective_ubatch(mapped.ubatch_size) <= effective_batch(mapped.batch_size));
        }
        observed.sort_unstable();
        observed.dedup();
        assert_eq!(observed, BALANCED_BATCH_LEVELS);
    }

    #[test]
    fn balanced_plan_without_flash_uses_l9_and_never_emits_flash() {
        let preset = ModelPreset::default();
        let plan =
            balanced_plan(&preset, &CalibrationWorkload::default(), None).expect("balanced plan");
        assert_eq!(plan.array, OrthogonalArray::L9);
        assert!(
            plan.candidates
                .iter()
                .all(|candidate| candidate.typed_patch.flash_attn.is_none())
        );
    }

    #[test]
    fn balanced_preserves_flash_setting_without_sweeping_it() {
        let plan = balanced_plan(
            &ModelPreset::default(),
            &CalibrationWorkload::default(),
            Some(&snapshot_with_flash()),
        )
        .expect("balanced plan");
        assert_eq!(plan.array, OrthogonalArray::L25);
        assert!(
            plan.candidates
                .iter()
                .all(|candidate| candidate.typed_patch.flash_attn.is_none())
        );
    }

    #[test]
    fn rapid_preset_is_rejected_before_candidate_generation() {
        let mut preset = ModelPreset::default();
        preset.backend = InferenceBackend::RapidMlx;
        assert!(balanced_candidates(&preset, &CalibrationWorkload::default(), None).is_err());
    }

    #[test]
    fn balanced_budget_fails_closed() {
        assert!(validate_balanced_budget(49, 9, 2).is_err());
        assert!(validate_balanced_budget(0, 26, 2).is_err());
        assert!(validate_balanced_budget(0, 25, 4).is_err());
        assert!(validate_balanced_budget(21, 25, 3).is_err());
    }

    #[test]
    fn oversized_preset_context_clamps_factor_levels_without_panicking() {
        let levels = numeric_u64_levels(200_000, 8_192, |context_size| LlamaCppCalibrationPatch {
            context_size: Some(context_size),
            ..Default::default()
        });

        assert_eq!(levels.len(), 5);
        assert!(
            levels
                .iter()
                .all(|patch| { patch.context_size.is_some_and(|value| value <= 131_072) })
        );
    }
}
