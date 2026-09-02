//! Typed contracts for bounded, hardware-native llama.cpp Calibration.
//!
//! This module intentionally contains no process launching or HTTP wiring yet.
//! Keeping the fingerprints, workload policy, and durable result shapes pure
//! gives later lifecycle and adapter code one backend-safe contract to share.

use crate::inference::InferenceBackend;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub mod jobs;
pub mod paths;

pub mod analysis;
pub mod argv;
pub mod candidates;
pub mod design;
pub mod executor;
pub mod launch_evidence;
pub mod server_qualification;

pub const CALIBRATION_SCHEMA_VERSION: u32 = 1;
pub const CALIBRATION_FACTOR_CATALOG_VERSION: u32 = 1;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationWorkloadKind {
    #[default]
    Interactive,
    Agents,
    MultiUser,
    Thinking,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationObjective {
    #[default]
    Balanced,
    Fastest,
    MaxContext,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationBudget {
    #[default]
    Quick,
    Balanced,
    Thorough,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum KvQualityFloor {
    F16,
    #[default]
    Q8_0,
    AnyExplicitlyLossy,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct GpuFingerprint {
    pub vendor: Option<String>,
    pub name: Option<String>,
    pub device_id: Option<String>,
    pub memory_bytes: Option<u64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct HardwareFingerprint {
    pub os: String,
    pub arch: String,
    pub cpu_identity: Option<String>,
    pub physical_cores: Option<u32>,
    pub logical_cores: u32,
    pub memory_bytes: u64,
    pub gpu_devices: Vec<GpuFingerprint>,
    pub unified_memory: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct ModelFingerprint {
    /// Library-relative identity; never expose a raw absolute path to the UI.
    pub library_relative_id: String,
    pub file_size: u64,
    pub modified_unix_ms: u128,
    pub content_fingerprint: String,
    pub gguf_arch: Option<String>,
    pub metadata_fingerprint: String,
    /// Stable architecture/shape/weight-quantization key used only for
    /// lower-confidence compatible-model evidence. It deliberately excludes
    /// the model path and file-specific content identity.
    #[serde(default)]
    pub compatibility_key: String,
    /// Architecture/shape key without weight quantization, used only for the
    /// weaker related-model evidence tier.
    #[serde(default)]
    pub family_key: String,
    /// Human-readable digest source for UI/debugging; never filename-derived.
    #[serde(default)]
    pub quantization_signature: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct RuntimeFingerprint {
    pub server_identity: String,
    pub server_sha256: String,
    pub version: Option<String>,
    pub capability_hash: String,
    pub bench_sha256: String,
    pub fit_params_sha256: Option<String>,
    /// Hash of normalized supported capability names. This may remain stable
    /// across frequent llama.cpp rebuilds even when executable hashes change.
    #[serde(default)]
    pub capability_signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct CalibrationWorkload {
    pub kind: CalibrationWorkloadKind,
    pub prompt_tokens: u32,
    pub generation_tokens: u32,
    pub parallel_requests: u32,
    pub minimum_context: u64,
    pub objective: CalibrationObjective,
    pub fixture_id: String,
}

impl Default for CalibrationWorkload {
    fn default() -> Self {
        Self {
            kind: CalibrationWorkloadKind::Interactive,
            prompt_tokens: 512,
            generation_tokens: 256,
            parallel_requests: 1,
            minimum_context: 4096,
            objective: CalibrationObjective::Balanced,
            fixture_id: "calibration-v1-interactive".into(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct CalibrationFingerprint {
    pub schema_version: u32,
    pub backend: InferenceBackend,
    pub hardware: HardwareFingerprint,
    pub model: ModelFingerprint,
    pub runtime: RuntimeFingerprint,
    pub workload: CalibrationWorkload,
    pub baseline_config_hash: String,
    pub factor_catalog_version: u32,
}

impl CalibrationFingerprint {
    pub fn current(backend: InferenceBackend, workload: CalibrationWorkload) -> Self {
        Self {
            schema_version: CALIBRATION_SCHEMA_VERSION,
            backend,
            workload,
            factor_catalog_version: CALIBRATION_FACTOR_CATALOG_VERSION,
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct StartCalibrationRequest {
    pub preset_id: String,
    pub expected_preset_fingerprint: String,
    pub workload: CalibrationWorkload,
    pub budget: CalibrationBudget,
    pub kv_quality_floor: KvQualityFloor,
    pub max_context: Option<u64>,
    pub allow_stop_active_server: bool,
    pub exact_confirmation: Option<String>,
    #[serde(default)]
    pub server_qualification: Option<server_qualification::QualificationRequest>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct LlamaCppCalibrationPatch {
    pub gpu_layers: Option<i32>,
    pub context_size: Option<u64>,
    pub threads: Option<i32>,
    pub threads_batch: Option<i32>,
    pub ctk: Option<String>,
    pub ctv: Option<String>,
    pub batch_size: Option<u32>,
    pub ubatch_size: Option<u32>,
    pub flash_attn: Option<bool>,
    pub n_cpu_moe: Option<i32>,
}

/// The control configuration actually measured by Calibration.
///
/// This is intentionally not called “llama.cpp defaults”: an omitted preset
/// field can be normalized by the product (for example, q8_0 K-cache and
/// f16 V-cache) before `llama-bench` is launched. The adjacent help-default
/// table records what the managed server advertises for comparison.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct CalibrationBaseline {
    pub effective: BTreeMap<String, CalibrationBaselineValue>,
    pub llama_server_help_defaults: BTreeMap<String, String>,
    pub llama_server_help_sha256: Option<String>,
    pub llama_server_help_exit_code: Option<i32>,
    pub llama_server_help_output_truncated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CalibrationBaselineValue {
    pub value: String,
    /// `preset`, `calibration_policy`, or `llama_server_help_default`.
    pub source: String,
}

impl CalibrationBaselineValue {
    pub fn new(value: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            source: source.into(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct CalibrationCandidate {
    pub id: String,
    pub typed_patch: LlamaCppCalibrationPatch,
    pub capability_evidence: Vec<String>,
    pub predicted_memory_bytes: Option<u64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct CalibrationReceipt {
    pub schema_version: u32,
    pub method_version: String,
    pub job_id: String,
    pub fingerprint: CalibrationFingerprint,
    pub measurement: CalibrationMeasurement,
    pub baseline: CalibrationBaseline,
    pub budget: CalibrationBudget,
    pub candidate_results: Vec<CalibrationCandidateResult>,
    #[serde(default)]
    pub analysis: analysis::CalibrationAnalysis,
    pub selected_candidate: Option<String>,
    pub preset_id: String,
    pub preset_fingerprint: String,
    pub apply_history: Vec<CalibrationApplyRecord>,
    #[serde(default)]
    pub server_qualification: Option<server_qualification::QualificationReceipt>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct CalibrationCandidateResult {
    pub candidate: CalibrationCandidate,
    pub measurement: CalibrationMeasurement,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct CalibrationApplyRecord {
    pub target_preset_id: String,
    pub candidate_id: String,
    pub derived: bool,
    pub before_fingerprint: String,
    pub after_fingerprint: String,
    pub timestamp_unix_ms: u128,
    pub validation: String,
    #[serde(default)]
    pub rollback_id: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrialStatus {
    Ok,
    Oom,
    Error,
    Timeout,
    ParseFailure,
    Implausible,
    PredictedUnsafe,
    Cancelled,
    SuspectedCrash,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct CalibrationMeasurement {
    pub trial_id: String,
    pub status: Option<TrialStatus>,
    pub pp_tps_samples: Vec<f64>,
    pub tg_tps_samples: Vec<f64>,
    pub ttft_ms_samples: Vec<f64>,
    pub effective_tps_samples: Vec<f64>,
    pub wall_time_ms: u64,
    pub memory_peak_bytes: Option<u64>,
    pub bounded_diagnostics: Vec<String>,
    #[serde(default)]
    pub launch_evidence: Option<launch_evidence::LaunchObservationReceipt>,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationJobState {
    #[default]
    Queued,
    Running,
    Cancelling,
    Cancelled,
    Complete,
    Failed,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct CalibrationJobSnapshot {
    pub id: String,
    pub state: CalibrationJobState,
    pub phase: String,
    pub completed_trials: u32,
    pub planned_trials: u32,
    pub diagnostics: Vec<String>,
    pub receipt_id: Option<String>,
}

/// Protected manifest needed to resume a job after process restart. It is
/// never returned directly by the API; the public snapshot remains redacted.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CalibrationJobManifest {
    pub schema_version: u32,
    pub preset_id: String,
    pub preset_fingerprint: String,
    pub workload: CalibrationWorkload,
    pub budget: CalibrationBudget,
    pub candidates: Vec<CalibrationCandidate>,
    pub model_path: String,
    pub bench_path: String,
    pub fingerprint: CalibrationFingerprint,
    #[serde(default)]
    pub baseline: CalibrationBaseline,
    #[serde(default)]
    pub server_qualification: Option<server_qualification::QualificationRequest>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_fingerprint_pins_schema_and_factor_versions() {
        let fingerprint = CalibrationFingerprint::current(
            InferenceBackend::LlamaCpp,
            CalibrationWorkload::default(),
        );
        assert_eq!(fingerprint.schema_version, CALIBRATION_SCHEMA_VERSION);
        assert_eq!(
            fingerprint.factor_catalog_version,
            CALIBRATION_FACTOR_CATALOG_VERSION
        );
        assert_eq!(fingerprint.workload.prompt_tokens, 512);
    }

    #[test]
    fn default_request_does_not_authorize_disruptive_server_stop() {
        let request = StartCalibrationRequest::default();
        assert!(!request.allow_stop_active_server);
        assert!(request.exact_confirmation.is_none());
    }

    #[test]
    fn contracts_round_trip_without_absolute_paths() {
        let fingerprint = CalibrationFingerprint::current(
            InferenceBackend::LlamaCpp,
            CalibrationWorkload::default(),
        );
        let encoded = serde_json::to_string(&fingerprint).expect("serialize fingerprint");
        assert!(!encoded.contains("/Users/"));
        let decoded: CalibrationFingerprint =
            serde_json::from_str(&encoded).expect("deserialize fingerprint");
        assert_eq!(decoded, fingerprint);
    }
}
