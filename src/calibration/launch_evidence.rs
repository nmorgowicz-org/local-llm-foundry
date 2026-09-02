//! Exact runtime memory evidence for preset-bundle cards (architecture 12).
//!
//! This is a distinct receipt kind layered onto the existing Calibration
//! fingerprint/evidence vocabulary (`CalibrationFingerprint`,
//! `classify_receipt_match` in `executor.rs`), not a separate evidence store.
//! Calibration measures tuning-candidate throughput; a `LaunchObservation`
//! measures the memory a single resolved launch actually used, sampled
//! post-readiness with guaranteed cleanup.

use crate::calibration::{GpuFingerprint, HardwareFingerprint, RuntimeFingerprint};
use crate::config::AppConfig;
use crate::inference::llama_cpp_capabilities::CapabilitySnapshot;
use crate::presets::ModelPreset;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

pub const LAUNCH_EVIDENCE_SCHEMA_VERSION: u32 = 1;

/// Architecture section 12 requires a freshness window but does not pin a
/// value; hardware, drivers, and background load drift enough that 30 days
/// is a reasonable default before an otherwise-exact receipt is presented as
/// stale rather than current.
pub const EVIDENCE_FRESHNESS_WINDOW_MS: u128 = 30 * 24 * 60 * 60 * 1000;

/// Every `ModelPreset` field must be classified so a newly added field can
/// never silently join or silently miss the memory-evidence fingerprint.
/// `assert_every_model_preset_field_is_classified` (below) fails the build
/// the moment a field is added without an entry here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgvFieldClass {
    /// Changes device/host memory allocation; enters the `evidence-v1`
    /// fingerprint digest.
    MemoryRelevant,
    /// Changes launch or request behavior but not allocated memory; excluded
    /// from the fingerprint digest, but still a known, accounted-for field.
    BehaviorOnly,
    /// Must never appear in any fingerprint, log, or receipt.
    Secret,
    /// Not a real launch argv input (bookkeeping or GGUF-derived metadata),
    /// or a value whose presence makes exact fingerprinting invalid
    /// (`extra_args` can smuggle unclassified flags past every other rule).
    Forbidden,
}

/// Returns `None` for any field name this vocabulary does not know about.
/// Callers must fail closed on `None`, never assume a safe default class.
pub fn classify_argv_field(field: &str) -> Option<ArgvFieldClass> {
    use ArgvFieldClass::{BehaviorOnly, Forbidden, MemoryRelevant, Secret};
    Some(match field {
        // Artifact, context, K/V policy, batch/concurrency, GPU distribution,
        // MoE placement, mmproj identity/offload, image token bounds,
        // load/fit/cache/SWA, and all draft/speculative model/KV/placement
        // settings (architecture 12).
        "model_path"
        | "context_size"
        | "ctk"
        | "ctv"
        | "tensor_split"
        | "batch_size"
        | "ubatch_size"
        | "no_mmap"
        | "load_mode"
        | "swa_full"
        | "ctx_checkpoints"
        | "checkpoint_min_step"
        | "cache_reuse"
        | "parallel_slots"
        | "n_cpu_moe"
        | "gpu_layers"
        | "mlock"
        | "flash_attn"
        | "split_mode"
        | "main_gpu"
        | "draft_model"
        | "draft_min"
        | "draft_max"
        | "spec_ngram_size"
        | "spec_type"
        | "spec_default"
        | "spec_draft_n_max"
        | "spec_draft_n_min"
        | "spec_draft_p_split"
        | "spec_draft_p_min"
        | "spec_draft_ngl"
        | "spec_draft_device"
        | "spec_draft_cpu_moe"
        | "spec_draft_n_cpu_moe"
        | "spec_draft_type_k"
        | "spec_draft_type_v"
        | "kv_unified"
        | "cache_idle_slots"
        | "cache_ram_mib"
        | "fit_enabled"
        | "fit_ctx"
        | "fit_target"
        | "mmproj"
        | "image_min_tokens"
        | "image_max_tokens"
        | "mmproj_offload" => MemoryRelevant,

        // Sampling/generation, scheduling, speculative ngram-matching
        // heuristics (tune the matcher, not model/KV/placement), templating,
        // networking, and organizational fields.
        "threads"
        | "threads_batch"
        | "prio"
        | "prio_batch"
        | "rope_scaling"
        | "rope_freq_base"
        | "rope_freq_scale"
        | "spec_ngram_mod_n_min"
        | "spec_ngram_mod_n_max"
        | "spec_ngram_mod_n_match"
        | "spec_ngram_simple_size_n"
        | "spec_ngram_simple_size_m"
        | "spec_ngram_simple_min_hits"
        | "spec_ngram_map_k_size_n"
        | "spec_ngram_map_k_size_m"
        | "spec_ngram_map_k_min_hits"
        | "spec_ngram_map_k4v_size_n"
        | "spec_ngram_map_k4v_size_m"
        | "spec_ngram_map_k4v_min_hits"
        | "fit_print"
        | "seed"
        | "system_prompt_file"
        | "bind_host"
        | "port"
        | "chat_template_file"
        | "grammar"
        | "json_schema"
        | "max_tokens"
        | "enable_thinking"
        | "preserve_thinking"
        | "tool_call_format"
        | "reasoning"
        | "reasoning_budget"
        | "reasoning_budget_message"
        | "llama_reasoning_effort"
        | "llama_reasoning_format"
        | "llama_reasoning_preserve"
        | "alias"
        | "benchmark_mode"
        | "tags"
        | "no_cont_batching"
        | "verbosity"
        | "ngram_spec"
        | "temperature"
        | "top_p"
        | "top_k"
        | "min_p"
        | "repeat_penalty"
        | "repeat_last_n"
        | "presence_penalty" => BehaviorOnly,

        "api_key" | "api_key_configured" | "clear_api_key" => Secret,

        // Bookkeeping, GGUF-derived read-only metadata (never a launch
        // input), legacy duplicate fields already superseded elsewhere
        // (`cache_type_k/v` by `ctk/ctv`, `hf_repo` by `model_path`), and
        // typed sub-objects handled by their own dedicated fingerprinting
        // (`bundle` selection fields, `rapid_mlx` backend config).
        // `extra_args` is Forbidden rather than BehaviorOnly: it is free text
        // that can carry an unclassified llama-server flag, so its presence
        // must block exact evidence rather than be silently ignored.
        "id"
        | "name"
        | "schema_version"
        | "revision"
        | "backend"
        | "rapid_mlx"
        | "bundle"
        | "hf_repo"
        | "cache_type_k"
        | "cache_type_v"
        | "gguf_architecture"
        | "param_count"
        | "family"
        | "size_class"
        | "architecture_kind"
        | "expert_count"
        | "expert_used_count"
        | "active_params_b"
        | "block_count"
        | "bytes_per_layer"
        | "expert_bytes_per_layer"
        | "extra_args" => Forbidden,

        _ => return None,
    })
}

/// Method used to obtain a memory observation. Only `WddmTotalDeviceDelta`,
/// `CudaRocmProcessDelta`, and `MetalUnifiedObservation` may power an
/// `Exact`/`Compatible`/`Related` evidence match; `FitProbe` and
/// `EstimatorOnly` are estimate-class and must never reach
/// `memory_peak_bytes` or upgrade a match class (architecture 12).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LaunchEvidenceMethod {
    WddmTotalDeviceDelta,
    CudaRocmProcessDelta,
    MetalUnifiedObservation,
    FitProbe,
    EstimatorOnly,
}

impl LaunchEvidenceMethod {
    pub fn is_direct_observation(self) -> bool {
        matches!(
            self,
            Self::WddmTotalDeviceDelta | Self::CudaRocmProcessDelta | Self::MetalUnifiedObservation
        )
    }
}

/// The canonical normalized resolved-argv manifest digest, restricted to
/// `MemoryRelevant` fields only, plus hardware/runtime/method identity.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct LaunchEvidenceFingerprint {
    pub schema_version: u32,
    pub hardware: HardwareFingerprint,
    pub runtime: RuntimeFingerprint,
    pub method: Option<LaunchEvidenceMethod>,
    /// `evidence-v1:<sha256>` over the sorted memory-relevant argv triples.
    pub manifest_digest: String,
    pub workload_concurrency: u32,
}

/// Builds the `evidence-v1:` digest over every `MemoryRelevant` field of the
/// resolved preset. Fails closed: an unclassified field aborts fingerprinting
/// instead of silently omitting it (the manifest validator's job is to make
/// that unreachable in normal operation via
/// `assert_every_model_preset_field_is_classified`).
pub fn manifest_digest(preset: &ModelPreset) -> Result<String, String> {
    let value = serde_json::to_value(preset).map_err(|e| e.to_string())?;
    let serde_json::Value::Object(fields) = value else {
        return Err("resolved preset did not serialize to an object".into());
    };
    let mut triples = Vec::new();
    for (name, field_value) in &fields {
        match classify_argv_field(name) {
            Some(ArgvFieldClass::MemoryRelevant) => {
                if field_value.is_null() {
                    continue;
                }
                triples.push(serde_json::json!([name, field_value]));
            }
            Some(_) => {}
            None => return Err(format!("unclassified argv field: {name}")),
        }
    }
    triples.sort_by(|a, b| a[0].as_str().cmp(&b[0].as_str()));
    let bytes = serde_json::to_vec(&triples).map_err(|e| e.to_string())?;
    let hex = Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Ok(format!("evidence-v1:{hex}"))
}

/// A launch pinned `--fit off` at measurement time. Fit defaults to `on` and
/// may shrink context or batch after the resolver decided, so a fit-on run
/// describes argv the binary did not actually use and must never be recorded
/// as an observation (hard gate, architecture/Phase 9).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FitState {
    Off,
    On,
}

/// A bounded post-readiness memory sample tied to one resolved launch.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct LaunchObservationReceipt {
    pub schema_version: u32,
    pub fingerprint: LaunchEvidenceFingerprint,
    pub fit_state: FitState,
    pub before_bytes: u64,
    pub peak_bytes: u64,
    pub after_bytes: u64,
    pub model_delta_bytes: Option<u64>,
    pub sample_count: u32,
    pub interval_ms: u32,
    pub noise_flags: Vec<String>,
    pub captured_unix_ms: u128,
}

impl Default for LaunchObservationReceipt {
    fn default() -> Self {
        Self {
            schema_version: LAUNCH_EVIDENCE_SCHEMA_VERSION,
            fingerprint: LaunchEvidenceFingerprint::default(),
            fit_state: FitState::On,
            before_bytes: 0,
            peak_bytes: 0,
            after_bytes: 0,
            model_delta_bytes: None,
            sample_count: 0,
            interval_ms: 0,
            noise_flags: Vec::new(),
            captured_unix_ms: 0,
        }
    }
}

/// Raw sampler output for one bounded observation window, independent of
/// preset/fingerprint identity so platform samplers can stay agnostic of the
/// evidence vocabulary.
#[derive(Debug, Clone, Default)]
pub struct LaunchSample {
    pub before_bytes: u64,
    pub peak_bytes: u64,
    pub after_bytes: u64,
    pub sample_count: u32,
    pub interval_ms: u32,
    pub noise_flags: Vec<String>,
    pub captured_unix_ms: u128,
}

/// Refuses to construct a receipt for a fit-on launch (hard gate) or a
/// launch whose resolved preset carries non-empty `extra_args` (Forbidden:
/// unclassified argv may have changed memory behavior invisibly).
pub fn build_launch_observation(
    preset: &ModelPreset,
    fingerprint_base: LaunchEvidenceFingerprint,
    fit_state: FitState,
    sample: LaunchSample,
) -> Result<LaunchObservationReceipt, String> {
    if fit_state != FitState::Off {
        return Err(
            "fit was not pinned off; this run is an estimate, not exact runtime evidence".into(),
        );
    }
    if !preset.extra_args.trim().is_empty() {
        return Err(
            "preset has non-empty extra_args; unclassified argv cannot be recorded as exact evidence"
                .into(),
        );
    }
    let manifest_digest = manifest_digest(preset)?;
    let mut fingerprint = fingerprint_base;
    fingerprint.schema_version = LAUNCH_EVIDENCE_SCHEMA_VERSION;
    fingerprint.manifest_digest = manifest_digest;
    let model_delta_bytes = sample.peak_bytes.checked_sub(sample.before_bytes);
    Ok(LaunchObservationReceipt {
        schema_version: LAUNCH_EVIDENCE_SCHEMA_VERSION,
        fingerprint,
        fit_state,
        before_bytes: sample.before_bytes,
        peak_bytes: sample.peak_bytes,
        after_bytes: sample.after_bytes,
        model_delta_bytes,
        sample_count: sample.sample_count,
        interval_ms: sample.interval_ms,
        noise_flags: sample.noise_flags,
        captured_unix_ms: sample.captured_unix_ms,
    })
}

/// Evidence match classes (architecture 12). Distinct from Calibration's
/// `ReceiptMatchKind`: this adds `Stale`, a fingerprint match whose receipt
/// has aged past the freshness window and must never present as current.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceMatchClass {
    Exact,
    Compatible,
    Related,
    Stale,
}

impl EvidenceMatchClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Compatible => "compatible",
            Self::Related => "related",
            Self::Stale => "stale",
        }
    }
}

/// Mirrors `classify_receipt_match` in `executor.rs` for the launch-evidence
/// vocabulary, then applies the freshness downgrade to `Stale`. A receipt
/// whose method is not a direct observation (`fit_probe`/`estimator_only`)
/// never matches at all — those never upgrade a match class.
pub fn classify_evidence_match(
    receipt: &LaunchEvidenceFingerprint,
    expected: &LaunchEvidenceFingerprint,
    receipt_age_ms: u128,
    freshness_window_ms: u128,
) -> Option<(EvidenceMatchClass, Vec<String>)> {
    let method = receipt.method?;
    if !method.is_direct_observation() {
        return None;
    }
    if receipt.hardware != expected.hardware
        || receipt.method != expected.method
        || receipt.workload_concurrency != expected.workload_concurrency
    {
        return None;
    }

    let exact =
        receipt.manifest_digest == expected.manifest_digest && receipt.runtime == expected.runtime;
    let compatible = !exact
        && receipt.manifest_digest == expected.manifest_digest
        && receipt.runtime.capability_signature == expected.runtime.capability_signature
        && !expected.runtime.capability_signature.is_empty();
    let related = !exact && !compatible;

    let mut kind = if exact {
        EvidenceMatchClass::Exact
    } else if compatible {
        EvidenceMatchClass::Compatible
    } else if related {
        EvidenceMatchClass::Related
    } else {
        return None;
    };

    let mut warnings = Vec::new();
    if kind != EvidenceMatchClass::Exact {
        warnings.push("Runtime build differs from this launch's binary".into());
    }
    if receipt_age_ms > freshness_window_ms {
        kind = EvidenceMatchClass::Stale;
        warnings.push("Evidence receipt has aged past the freshness window".into());
    }
    Some((kind, warnings))
}

static HARDWARE_FINGERPRINT: OnceLock<HardwareFingerprint> = OnceLock::new();

/// Hardware does not change for the life of the process, so this is computed
/// once. Unlike the Calibration preflight fingerprint, this must stay cheap
/// enough to call on every interactive `/resolve`.
fn cached_hardware_fingerprint(config: &AppConfig) -> HardwareFingerprint {
    HARDWARE_FINGERPRINT
        .get_or_init(|| {
            let system = sysinfo::System::new_all();
            let logical_cores = std::thread::available_parallelism()
                .map(|count| count.get() as u32)
                .unwrap_or_default();
            let gpu_devices = crate::gpu::detect_backend(&config.gpu_backend)
                .read_metrics()
                .unwrap_or_default()
                .into_iter()
                .map(|(name, metrics)| {
                    let lower = name.to_ascii_lowercase();
                    let vendor = if lower.contains("nvidia") {
                        Some("nvidia".into())
                    } else if lower.contains("amd") || lower.contains("radeon") {
                        Some("amd".into())
                    } else if lower.contains("apple") {
                        Some("apple".into())
                    } else {
                        None
                    };
                    GpuFingerprint {
                        vendor,
                        name: Some(name),
                        device_id: None,
                        memory_bytes: (metrics.vram_total > 0).then_some(metrics.vram_total),
                    }
                })
                .collect();
            HardwareFingerprint {
                os: std::env::consts::OS.into(),
                arch: std::env::consts::ARCH.into(),
                cpu_identity: system.cpus().first().map(|cpu| cpu.brand().to_string()),
                physical_cores: sysinfo::System::physical_core_count().map(|count| count as u32),
                logical_cores,
                memory_bytes: system.total_memory(),
                gpu_devices,
                unified_memory: cfg!(target_os = "macos"),
            }
        })
        .clone()
}

/// The observation method this host can produce, used to scope evidence
/// lookups to receipts measured the same way this host would measure them.
pub fn current_platform_method() -> LaunchEvidenceMethod {
    if cfg!(target_os = "macos") {
        LaunchEvidenceMethod::MetalUnifiedObservation
    } else if cfg!(target_os = "windows") {
        LaunchEvidenceMethod::WddmTotalDeviceDelta
    } else {
        LaunchEvidenceMethod::CudaRocmProcessDelta
    }
}

/// Builds the fingerprint used to both save a new observation on this host
/// and look one up for the resolved bundle currently shown on a card. Both
/// call sites must derive it identically or a self-measured receipt could
/// fail to match its own future lookup.
pub fn current_fingerprint(
    config: &AppConfig,
    resolved_preset: &ModelPreset,
    capabilities: &CapabilitySnapshot,
) -> Result<LaunchEvidenceFingerprint, String> {
    Ok(LaunchEvidenceFingerprint {
        schema_version: LAUNCH_EVIDENCE_SCHEMA_VERSION,
        hardware: cached_hardware_fingerprint(config),
        runtime: RuntimeFingerprint {
            server_identity: capabilities.executable_identity.path.clone(),
            server_sha256: capabilities.executable_identity.file_hash.clone(),
            version: Some(capabilities.version_text.clone()),
            capability_hash: capabilities.help_hash.clone(),
            bench_sha256: String::new(),
            fit_params_sha256: None,
            capability_signature: capabilities.help_hash.clone(),
        },
        method: Some(current_platform_method()),
        manifest_digest: manifest_digest(resolved_preset)?,
        workload_concurrency: resolved_preset.parallel_slots,
    })
}

/// On-disk persistence for launch-observation receipts. This lives beside
/// the Calibration receipts store (same `calibrations/` root, a sibling
/// `launch-evidence/` directory) rather than an unrelated store, but keeps a
/// distinct on-disk shape: a `LaunchObservationReceipt` is one bounded
/// post-readiness sample, not a full tuning sweep.
pub mod store {
    use super::{EvidenceMatchClass, LaunchEvidenceFingerprint, LaunchObservationReceipt};
    use std::fs;
    use std::path::{Path, PathBuf};

    fn receipt_path(dir: &Path, receipt: &LaunchObservationReceipt) -> PathBuf {
        let digest = &receipt.fingerprint.manifest_digest;
        let short = digest.rsplit(':').next().unwrap_or(digest);
        let short = &short[..short.len().min(16)];
        dir.join(format!(
            "{}-{}-{}.json",
            receipt.captured_unix_ms, short, receipt.sample_count
        ))
    }

    /// Persists one receipt. Directory creation and the write are the
    /// caller's cleanup boundary: a failed write must not leave a partial
    /// file mistaken for evidence, so this writes to a temp path and renames.
    pub fn save(dir: &Path, receipt: &LaunchObservationReceipt) -> Result<PathBuf, String> {
        fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        let final_path = receipt_path(dir, receipt);
        let tmp_path = final_path.with_extension("json.tmp");
        let encoded = serde_json::to_vec_pretty(receipt).map_err(|e| e.to_string())?;
        fs::write(&tmp_path, encoded).map_err(|e| e.to_string())?;
        fs::rename(&tmp_path, &final_path).map_err(|e| e.to_string())?;
        Ok(final_path)
    }

    /// Loads every receipt in `dir`. A corrupt or non-JSON entry is skipped
    /// rather than failing the whole listing — evidence lookup is advisory,
    /// never load-bearing for session control.
    pub fn list(dir: &Path) -> Vec<LaunchObservationReceipt> {
        let Ok(entries) = fs::read_dir(dir) else {
            return Vec::new();
        };
        entries
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "json"))
            .filter_map(|entry| fs::read(entry.path()).ok())
            .filter_map(|bytes| serde_json::from_slice(&bytes).ok())
            .collect()
    }

    /// Finds the best-classified match for `expected` among receipts on disk,
    /// preferring the strongest match class and, within a class, the most
    /// recently captured receipt.
    pub fn best_match(
        dir: &Path,
        expected: &LaunchEvidenceFingerprint,
        now_unix_ms: u128,
        freshness_window_ms: u128,
    ) -> Option<(EvidenceMatchClass, Vec<String>, LaunchObservationReceipt)> {
        let mut best: Option<(EvidenceMatchClass, Vec<String>, LaunchObservationReceipt)> = None;
        for receipt in list(dir) {
            let age_ms = now_unix_ms.saturating_sub(receipt.captured_unix_ms);
            let Some((class, warnings)) = super::classify_evidence_match(
                &receipt.fingerprint,
                expected,
                age_ms,
                freshness_window_ms,
            ) else {
                continue;
            };
            let better = match &best {
                None => true,
                Some((best_class, _, best_receipt)) => {
                    rank(class) > rank(*best_class)
                        || (rank(class) == rank(*best_class)
                            && receipt.captured_unix_ms > best_receipt.captured_unix_ms)
                }
            };
            if better {
                best = Some((class, warnings, receipt));
            }
        }
        best
    }

    fn rank(class: EvidenceMatchClass) -> u8 {
        match class {
            EvidenceMatchClass::Exact => 3,
            EvidenceMatchClass::Compatible => 2,
            EvidenceMatchClass::Related => 1,
            EvidenceMatchClass::Stale => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::{GpuFingerprint, HardwareFingerprint, RuntimeFingerprint};

    #[test]
    fn every_model_preset_field_has_an_argv_classification() {
        let preset = ModelPreset::default();
        let value = serde_json::to_value(&preset).expect("preset serializes");
        let serde_json::Value::Object(fields) = value else {
            panic!("preset serializes to an object");
        };
        let unclassified: Vec<&String> = fields
            .keys()
            .filter(|name| classify_argv_field(name).is_none())
            .collect();
        assert!(
            unclassified.is_empty(),
            "unclassified ModelPreset fields (add to classify_argv_field): {unclassified:?}"
        );
    }

    #[test]
    fn unknown_field_name_is_not_silently_classified() {
        assert!(classify_argv_field("a_field_nobody_added_here").is_none());
    }

    #[test]
    fn manifest_digest_ignores_behavior_only_and_secret_fields() {
        let mut base = ModelPreset {
            model_path: "/models/a.gguf".into(),
            context_size: 4096,
            ..Default::default()
        };
        let before = manifest_digest(&base).expect("digest");
        base.temperature = Some(0.9);
        base.alias = Some("renamed".into());
        base.api_key = Some("secret-value".into());
        let after = manifest_digest(&base).expect("digest");
        assert_eq!(before, after);
        assert!(!after.contains("secret-value"));
    }

    #[test]
    fn manifest_digest_changes_with_memory_relevant_fields() {
        let base = ModelPreset {
            model_path: "/models/a.gguf".into(),
            context_size: 4096,
            ..Default::default()
        };
        let mut changed = base.clone();
        changed.context_size = 8192;
        assert_ne!(
            manifest_digest(&base).unwrap(),
            manifest_digest(&changed).unwrap()
        );
    }

    #[test]
    fn build_launch_observation_rejects_fit_on() {
        let preset = ModelPreset::default();
        let fingerprint = LaunchEvidenceFingerprint::default();
        let sample = LaunchSample {
            before_bytes: 1,
            peak_bytes: 2,
            after_bytes: 2,
            sample_count: 3,
            interval_ms: 250,
            noise_flags: Vec::new(),
            captured_unix_ms: 0,
        };
        let result = build_launch_observation(&preset, fingerprint, FitState::On, sample);
        assert!(result.is_err());
    }

    #[test]
    fn build_launch_observation_rejects_extra_args() {
        let preset = ModelPreset {
            extra_args: "--some-unclassified-flag".into(),
            ..Default::default()
        };
        let fingerprint = LaunchEvidenceFingerprint::default();
        let sample = LaunchSample {
            before_bytes: 1,
            peak_bytes: 2,
            after_bytes: 2,
            sample_count: 3,
            interval_ms: 250,
            noise_flags: Vec::new(),
            captured_unix_ms: 0,
        };
        let result = build_launch_observation(&preset, fingerprint, FitState::Off, sample);
        assert!(result.is_err());
    }

    #[test]
    fn build_launch_observation_accepts_fit_off_and_records_delta() {
        let preset = ModelPreset {
            model_path: "/models/a.gguf".into(),
            context_size: 4096,
            ..Default::default()
        };
        let fingerprint = LaunchEvidenceFingerprint {
            method: Some(LaunchEvidenceMethod::MetalUnifiedObservation),
            ..Default::default()
        };
        let sample = LaunchSample {
            before_bytes: 10_000_000_000,
            peak_bytes: 26_300_000_000,
            after_bytes: 18_000_000_000,
            sample_count: 5,
            interval_ms: 250,
            noise_flags: Vec::new(),
            captured_unix_ms: 1_700_000_000_000,
        };
        let receipt =
            build_launch_observation(&preset, fingerprint, FitState::Off, sample).expect("receipt");
        assert_eq!(receipt.model_delta_bytes, Some(16_300_000_000));
        assert!(
            receipt
                .fingerprint
                .manifest_digest
                .starts_with("evidence-v1:")
        );
    }

    fn fingerprint(
        method: LaunchEvidenceMethod,
        manifest_digest: &str,
    ) -> LaunchEvidenceFingerprint {
        LaunchEvidenceFingerprint {
            schema_version: LAUNCH_EVIDENCE_SCHEMA_VERSION,
            hardware: HardwareFingerprint {
                os: "macos".into(),
                arch: "arm64".into(),
                unified_memory: true,
                gpu_devices: vec![GpuFingerprint {
                    vendor: Some("apple".into()),
                    ..Default::default()
                }],
                ..Default::default()
            },
            runtime: RuntimeFingerprint {
                server_sha256: "server-a".into(),
                capability_signature: "cap-a".into(),
                ..Default::default()
            },
            method: Some(method),
            manifest_digest: manifest_digest.into(),
            workload_concurrency: 1,
        }
    }

    #[test]
    fn evidence_match_classifies_exact_compatible_related_stale_and_rejects_probe() {
        let expected = fingerprint(
            LaunchEvidenceMethod::MetalUnifiedObservation,
            "evidence-v1:aaa",
        );

        let (kind, _) = classify_evidence_match(&expected, &expected, 0, 60_000).expect("exact");
        assert_eq!(kind, EvidenceMatchClass::Exact);

        let mut compatible = expected.clone();
        compatible.runtime.server_sha256 = "server-b".into();
        let (kind, _) =
            classify_evidence_match(&compatible, &expected, 0, 60_000).expect("compatible");
        assert_eq!(kind, EvidenceMatchClass::Compatible);

        let mut related = expected.clone();
        related.runtime.capability_signature = "cap-b".into();
        related.manifest_digest = "evidence-v1:bbb".into();
        let (kind, _) = classify_evidence_match(&related, &expected, 0, 60_000).expect("related");
        assert_eq!(kind, EvidenceMatchClass::Related);

        let (kind, warnings) =
            classify_evidence_match(&expected, &expected, 120_000, 60_000).expect("stale");
        assert_eq!(kind, EvidenceMatchClass::Stale);
        assert!(!warnings.is_empty());

        let mut probe = expected.clone();
        probe.method = Some(LaunchEvidenceMethod::FitProbe);
        assert!(classify_evidence_match(&probe, &expected, 0, 60_000).is_none());

        let mut different_hardware = expected.clone();
        different_hardware.hardware.arch = "x86_64".into();
        assert!(classify_evidence_match(&different_hardware, &expected, 0, 60_000).is_none());
    }

    fn observation(
        method: LaunchEvidenceMethod,
        manifest_digest: &str,
        captured_unix_ms: u128,
    ) -> LaunchObservationReceipt {
        LaunchObservationReceipt {
            schema_version: LAUNCH_EVIDENCE_SCHEMA_VERSION,
            fingerprint: fingerprint(method, manifest_digest),
            fit_state: FitState::Off,
            before_bytes: 10_000_000_000,
            peak_bytes: 26_000_000_000,
            after_bytes: 18_000_000_000,
            model_delta_bytes: Some(16_000_000_000),
            sample_count: 5,
            interval_ms: 250,
            noise_flags: Vec::new(),
            captured_unix_ms,
        }
    }

    #[test]
    fn store_round_trips_receipts_to_disk() {
        let temp = tempfile::tempdir().expect("tempdir");
        let receipt = observation(
            LaunchEvidenceMethod::MetalUnifiedObservation,
            "evidence-v1:aaa",
            1_000,
        );
        let path = store::save(temp.path(), &receipt).expect("save receipt");
        assert!(path.exists());
        let loaded = store::list(temp.path());
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].fingerprint.manifest_digest, "evidence-v1:aaa");
    }

    #[test]
    fn store_best_match_prefers_strongest_class_then_most_recent() {
        let temp = tempfile::tempdir().expect("tempdir");
        let expected = fingerprint(
            LaunchEvidenceMethod::MetalUnifiedObservation,
            "evidence-v1:aaa",
        );

        let older_exact = observation(
            LaunchEvidenceMethod::MetalUnifiedObservation,
            "evidence-v1:aaa",
            1_000,
        );
        let newer_exact = observation(
            LaunchEvidenceMethod::MetalUnifiedObservation,
            "evidence-v1:aaa",
            2_000,
        );
        let mut related = observation(
            LaunchEvidenceMethod::MetalUnifiedObservation,
            "evidence-v1:zzz",
            3_000,
        );
        related.fingerprint.runtime.capability_signature = String::new();
        store::save(temp.path(), &older_exact).expect("save older");
        store::save(temp.path(), &newer_exact).expect("save newer");
        store::save(temp.path(), &related).expect("save related");

        let (class, _, matched) =
            store::best_match(temp.path(), &expected, 2_000, 60_000).expect("match");
        assert_eq!(class, EvidenceMatchClass::Exact);
        assert_eq!(matched.captured_unix_ms, 2_000);
    }

    #[test]
    fn store_list_skips_corrupt_files_instead_of_failing() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp.path().join("not-json.json"), b"not json").expect("write junk");
        assert!(store::list(temp.path()).is_empty());
    }
}
