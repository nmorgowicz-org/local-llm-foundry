//! Probe-backed estimate enrichment for resolved preset configurations.
//!
//! The probe is an estimate-class source. It supplies the measured base
//! components for the flags it accepts; unsupported launch components remain
//! named estimator additions and never become runtime measurement evidence.

use crate::llama::vram_estimator::{VramBreakdown, VramRecommendation};
use crate::presets::ModelPreset;
use crate::presets::fit_probe::FitReading;
use crate::presets::resolver::{EstimateDivergence, LaunchEstimate};

const MIB: u64 = 1024 * 1024;
pub const COMPONENT_TOLERANCE_MIB: i64 = 512;

#[derive(Debug, Clone)]
pub struct ProbeAddition {
    pub name: String,
    pub bytes: u64,
}

/// Turn one accepted probe reading into the existing VRAM breakdown shape.
pub fn enrich(
    mut formula: VramBreakdown,
    reading: &FitReading,
    additions: &[ProbeAddition],
) -> LaunchEstimate {
    let divergence = compare(&formula, reading);
    let probe_model = reading.model_mib.saturating_mul(MIB);
    let probe_context = reading.context_mib.saturating_mul(MIB);
    let probe_compute = reading.compute_mib.saturating_mul(MIB);
    let probe_device = reading.device_total_mib.saturating_mul(MIB);
    let probe_host = reading.host_total_mib.saturating_mul(MIB);
    let addition_bytes = additions.iter().map(|addition| addition.bytes).sum::<u64>();

    formula.weights_bytes = probe_model;
    formula.kv_cache_bytes = probe_context;
    formula.overhead_bytes = probe_compute;
    formula.total_bytes = probe_device.saturating_add(addition_bytes);
    formula.headroom_bytes = signed_headroom(formula.available_bytes, formula.total_bytes);
    formula.ram_bytes = probe_host;
    formula.ram_headroom_bytes = signed_headroom(formula.available_ram_bytes, probe_host);
    formula.recommendation = recommendation(formula.total_bytes, formula.available_bytes);
    formula.note = if additions.is_empty() {
        "Probe-backed estimate; method=fit_probe.".into()
    } else {
        format!(
            "Probe-backed floor plus estimated additions: {}; method=fit_probe.",
            additions
                .iter()
                .map(|addition| addition.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )
    };

    LaunchEstimate {
        breakdown: formula,
        method: "fit_probe".into(),
        probe_device_total_mib: reading.device_total_mib,
        probe_host_total_mib: reading.host_total_mib,
        divergence,
        additions: additions
            .iter()
            .map(|addition| addition.name.clone())
            .collect(),
    }
}

/// Compare a probe reading with the formula's three corresponding components.
pub fn compare(formula: &VramBreakdown, reading: &FitReading) -> EstimateDivergence {
    let model_mib = formula_component(formula.weights_bytes);
    let context_mib = formula_component(formula.kv_cache_bytes);
    let compute_mib = formula_component(formula.overhead_bytes);
    EstimateDivergence {
        model_mib: signed_mib_delta(reading.model_mib, model_mib),
        context_mib: signed_mib_delta(reading.context_mib, context_mib),
        compute_mib: signed_mib_delta(reading.compute_mib, compute_mib),
        within_tolerance: [
            signed_mib_delta(reading.model_mib, model_mib),
            signed_mib_delta(reading.context_mib, context_mib),
            signed_mib_delta(reading.compute_mib, compute_mib),
        ]
        .into_iter()
        .all(|delta| delta.abs() <= COMPONENT_TOLERANCE_MIB),
    }
}

/// Identify launch settings that the fixed probe invocation does not accept.
pub fn unsupported_additions(preset: &ModelPreset, formula: &VramBreakdown) -> Vec<ProbeAddition> {
    let mut additions = Vec::new();
    if preset
        .mmproj
        .as_deref()
        .is_some_and(|path| !path.is_empty())
    {
        additions.push(ProbeAddition {
            name: "mmproj".into(),
            bytes: formula.mmproj_bytes,
        });
    }
    if !preset.draft_model.is_empty() {
        additions.push(ProbeAddition {
            name: "draft model".into(),
            bytes: formula.mtp_bytes,
        });
    }
    if preset.cache_ram_mib.is_some() {
        additions.push(ProbeAddition {
            name: "cache-RAM reservation".into(),
            bytes: 0,
        });
    }
    let extra_args = preset.extra_args.to_ascii_lowercase();
    for (flag, name) in [
        ("--swa", "SWA mode"),
        ("--threads", "thread count"),
        ("--tensor-split", "tensor split"),
        ("--cache-ram", "cache-RAM reservation"),
    ] {
        if extra_args.split_whitespace().any(|arg| arg == flag) {
            additions.push(ProbeAddition {
                name: name.into(),
                bytes: 0,
            });
        }
    }
    additions
}

fn formula_component(bytes: u64) -> u64 {
    bytes / MIB
}

fn signed_mib_delta(probe: u64, formula: u64) -> i64 {
    (probe as i128 - formula as i128).clamp(i64::MIN as i128, i64::MAX as i128) as i64
}

fn signed_headroom(available: u64, used: u64) -> i64 {
    (available as i128 - used as i128).clamp(i64::MIN as i128, i64::MAX as i128) as i64
}

fn recommendation(total: u64, available: u64) -> VramRecommendation {
    if available == 0 {
        VramRecommendation::Risk
    } else if total <= available.saturating_mul(82) / 100 {
        VramRecommendation::Fit
    } else if total <= available {
        VramRecommendation::Tight
    } else {
        VramRecommendation::WontFit
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama::vram_estimator::{
        Backend, EstimateEvidence, EstimatorOptions, ModelArch, full_estimate,
    };

    #[test]
    fn probe_method_and_components_are_explicit() {
        let formula = full_estimate(
            8_000_000_000,
            &ModelArch::standard_heuristic(7.0),
            8_000,
            "q8_0",
            "q8_0",
            1,
            256,
            0,
            -1,
            20_000 * MIB,
            20_000 * MIB,
            false,
            EstimatorOptions {
                backend: Backend::LlamaCpp,
                evidence: EstimateEvidence::Measured,
                ..Default::default()
            },
        );
        let reading = FitReading {
            n_cpu_moe: 0,
            device_total_mib: 1_000,
            host_total_mib: 50,
            model_mib: 800,
            context_mib: 100,
            compute_mib: 100,
        };
        let estimate = enrich(formula, &reading, &[]);
        assert_eq!(estimate.method, "fit_probe");
        assert_eq!(estimate.breakdown.total_bytes, 1_000 * MIB);
        assert!(estimate.breakdown.note.contains("fit_probe"));
    }

    #[test]
    fn unsupported_components_are_named() {
        let mut preset = ModelPreset::default();
        preset.mmproj = Some("mmproj.gguf".into());
        preset.draft_model = "draft.gguf".into();
        preset.cache_ram_mib = Some(8192);
        let mut formula = full_estimate(
            8_000_000_000,
            &ModelArch::standard_heuristic(7.0),
            8_000,
            "q8_0",
            "q8_0",
            1,
            256,
            0,
            -1,
            20_000 * MIB,
            20_000 * MIB,
            false,
            EstimatorOptions::default(),
        );
        formula.mmproj_bytes = 10;
        formula.mtp_bytes = 20;
        let names = unsupported_additions(&preset, &formula)
            .into_iter()
            .map(|addition| addition.name)
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            vec!["mmproj", "draft model", "cache-RAM reservation"]
        );
    }
}
