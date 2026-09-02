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

/// Bounded post-readiness macOS/Metal sampler (architecture 12).
///
/// Runs detached from session control: `start_backend` returns to its caller
/// as soon as readiness succeeds, and this task samples afterward in the
/// background. It holds no session-control lock, never blocks a stop or
/// restart, and its only externally visible effect on success is one new
/// receipt file — a failure at any step is silently dropped, since evidence
/// capture is advisory and must never affect session control.
pub mod metal_sampler {
    use super::{FitState, LaunchSample, build_launch_observation, current_fingerprint, store};
    use crate::config::AppConfig;
    use crate::presets::ModelPreset;
    use std::time::Duration;

    const SAMPLE_COUNT: u32 = 6;
    const SAMPLE_INTERVAL: Duration = Duration::from_millis(750);
    const REPEAT_CYCLES: u32 = 3;
    // wired_bytes is system-wide, not process-scoped, so it swings more
    // between samples than a discrete GPU's total-device VRAM reading does;
    // this tolerance is wider than nvidia_sampler's 16 MiB for that reason.
    const CYCLE_AGREEMENT_TOLERANCE_BYTES: u64 = 64 * 1024 * 1024;

    /// Spawns the bounded sampler if this host/launch qualifies for exact
    /// evidence: macOS, fit explicitly pinned off, and no unclassified
    /// `extra_args` (the same hard gate `build_launch_observation` enforces,
    /// checked here too so a disqualified launch never starts a poll loop).
    /// `before_bytes` must be sampled by the caller immediately before the
    /// process spawn, while this function still has exclusive access to that
    /// pre-launch instant.
    pub fn spawn(app_config: AppConfig, preset: ModelPreset, before_bytes: u64) {
        if !cfg!(target_os = "macos") {
            return;
        }
        if preset.fit_enabled != Some(false) || !preset.extra_args.trim().is_empty() {
            return;
        }
        tokio::spawn(async move {
            run(app_config, preset, before_bytes).await;
        });
    }

    /// Runs `REPEAT_CYCLES` peak/after cycles back to back (re-baselining
    /// each cycle against the previous cycle's `after` sample, exactly like
    /// `nvidia_sampler::run`) and checks the resulting deltas against
    /// `super::nvidia_sampler::cycles_agree`. A single wired-memory sample is
    /// too easily inflated by unrelated system activity (Chrome, Spotlight,
    /// background sync) to trust on its own; requiring multiple cycles to
    /// agree is the same mitigation the noisier-signal-but-fewer-safeguards
    /// asymmetry this module used to have on Windows now also gets here.
    async fn run(app_config: AppConfig, preset: ModelPreset, before_bytes: u64) {
        let mut noise_flags = vec![
            "macOS unified-memory sample is a system-wide wired-memory delta, not process-scoped"
                .to_string(),
        ];

        let mut cycle_before = before_bytes;
        let mut deltas = Vec::with_capacity(REPEAT_CYCLES as usize);
        let mut peak_bytes = before_bytes;
        let mut after_bytes = before_bytes;

        for _ in 0..REPEAT_CYCLES {
            let mut cycle_peak = cycle_before;
            for _ in 0..SAMPLE_COUNT {
                tokio::time::sleep(SAMPLE_INTERVAL).await;
                let sample = crate::memory_availability::build_snapshot();
                cycle_peak = cycle_peak.max(sample.wired_bytes);
            }
            let cycle_after = crate::memory_availability::build_snapshot().wired_bytes;
            deltas.push(cycle_peak.checked_sub(cycle_before));
            peak_bytes = peak_bytes.max(cycle_peak);
            after_bytes = cycle_after;
            cycle_before = cycle_after;
        }

        if !super::nvidia_sampler::cycles_agree(&deltas, CYCLE_AGREEMENT_TOLERANCE_BYTES) {
            noise_flags.push(format!(
                "repeated observation cycles did not agree within tolerance: {deltas:?}"
            ));
        }

        // Reuses the OnceLock capability cache `construct_adapter` already
        // populated for this exact binary during launch validation; this
        // must never itself spawn `llama-server --help`.
        let Ok(identity) = crate::inference::llama_cpp_capabilities::ExecutableIdentity::from_path(
            &app_config.llama_server_path,
        ) else {
            return;
        };
        let Some(capabilities) =
            crate::inference::llama_cpp_capabilities::cached_snapshot(&identity)
        else {
            return;
        };
        let Ok(fingerprint) = current_fingerprint(&app_config, &preset, &capabilities) else {
            return;
        };

        let sample = LaunchSample {
            before_bytes,
            peak_bytes,
            after_bytes,
            sample_count: SAMPLE_COUNT * REPEAT_CYCLES,
            interval_ms: SAMPLE_INTERVAL.as_millis() as u32,
            noise_flags,
            captured_unix_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_millis())
                .unwrap_or_default(),
        };
        let Ok(receipt) = build_launch_observation(&preset, fingerprint, FitState::Off, sample)
        else {
            return;
        };
        let _ = store::save(&app_config.app_paths.launch_evidence_dir(), &receipt);
    }
}

/// Bounded, repeated, idle-stabilized Windows/WDDM `nvidia-smi` sampler
/// (architecture 12, Phase 9 real-host gate). Unlike `metal_sampler`
/// (single in-process poll loop), this drives an external process for every
/// sample, so it only ever pays that cost on a launch that already
/// qualifies for exact evidence — see `capture_before`'s gate.
///
/// Scoped to `WddmTotalDeviceDelta` (total-device) only. `CudaRocmProcessDelta`
/// (per-process) is deliberately not attempted: a real sample taken against
/// Ryne's driver (`tests/fixtures/nvidia_smi_compute_apps_csv.txt`) shows
/// `used_memory` reported as `[N/A]` for every process under WDDM, exactly
/// the unreliable-attribution case
/// `docs/plans/evidence/preset-bundles/windows-cuda-sampler-design.md`
/// flagged as an open question — so the compute-apps query is used here only
/// for PID-presence background-noise detection, never as a memory source.
pub mod nvidia_sampler {
    use super::{
        FitState, LaunchEvidenceMethod, LaunchSample, build_launch_observation,
        current_fingerprint, store,
    };
    use crate::config::AppConfig;
    use crate::presets::ModelPreset;
    use std::time::Duration;

    const IDLE_STABILIZE_SAMPLES: u32 = 5;
    const IDLE_STABILIZE_INTERVAL: Duration = Duration::from_millis(200);
    const IDLE_STABILIZE_TOLERANCE_BYTES: u64 = 16 * 1024 * 1024;
    const PEAK_SAMPLE_COUNT: u32 = 6;
    const PEAK_SAMPLE_INTERVAL: Duration = Duration::from_millis(750);
    const REPEAT_CYCLES: u32 = 3;

    /// One row of `nvidia-smi --query-compute-apps=pid,used_memory`. Memory
    /// is `None` whenever the driver reports `[N/A]` (the observed case on
    /// WDDM) rather than a parsed value — presence/absence of the row is
    /// still meaningful for noise detection even when memory isn't.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) struct ComputeApp {
        pub pid: u32,
        pub used_memory_bytes: Option<u64>,
    }

    /// Parses `nvidia-smi --query-compute-apps=pid,used_memory
    /// --format=csv,noheader,nounits` output. A row with an unparseable pid
    /// is dropped; `[N/A]` (or any other unparseable) memory field becomes
    /// `None` rather than dropping the row, since the pid itself is still
    /// usable for background-process presence diffing.
    pub(crate) fn parse_compute_apps_csv(csv: &str) -> Vec<ComputeApp> {
        csv.lines()
            .filter_map(|line| {
                let fields: Vec<&str> = line.split(',').map(str::trim).collect();
                if fields.len() < 2 {
                    return None;
                }
                let pid = fields[0].parse::<u32>().ok()?;
                let used_memory_bytes = fields[1].parse::<u64>().ok().map(|mib| mib * 1024 * 1024);
                Some(ComputeApp {
                    pid,
                    used_memory_bytes,
                })
            })
            .collect()
    }

    /// Finds the first pair of consecutive samples that agree within
    /// `tolerance` and returns `(that_value, true)`; otherwise returns the
    /// last sample seen and `false` so the caller can flag an unstabilized
    /// baseline rather than silently trusting a still-drifting reading.
    pub(crate) fn find_stable_value(samples: &[u64], tolerance: u64) -> (u64, bool) {
        for window in samples.windows(2) {
            if window[1].abs_diff(window[0]) <= tolerance {
                return (window[1], true);
            }
        }
        (samples.last().copied().unwrap_or(0), false)
    }

    /// Diffs two `--query-compute-apps` inventories, excluding `launched_pid`
    /// itself, and returns human-readable `noise_flags` entries for any PID
    /// that appeared, disappeared, or (when both samples have a parsed
    /// memory value) moved by more than `tolerance`.
    pub(crate) fn diff_background_processes(
        before: &[ComputeApp],
        after: &[ComputeApp],
        launched_pid: u32,
        tolerance: u64,
    ) -> Vec<String> {
        let mut flags = Vec::new();
        for app in after {
            if app.pid == launched_pid {
                continue;
            }
            match before.iter().find(|p| p.pid == app.pid) {
                None => flags.push(format!(
                    "background CUDA process pid={} appeared during sampling window",
                    app.pid
                )),
                Some(prev) => {
                    if let (Some(prev_bytes), Some(now_bytes)) =
                        (prev.used_memory_bytes, app.used_memory_bytes)
                        && prev_bytes.abs_diff(now_bytes) > tolerance
                    {
                        flags.push(format!(
                            "background CUDA process pid={} changed by {} bytes during sampling window",
                            app.pid,
                            prev_bytes.abs_diff(now_bytes)
                        ));
                    }
                }
            }
        }
        for prev in before {
            if prev.pid == launched_pid {
                continue;
            }
            if !after.iter().any(|app| app.pid == prev.pid) {
                flags.push(format!(
                    "background CUDA process pid={} disappeared during sampling window",
                    prev.pid
                ));
            }
        }
        flags
    }

    /// `true` only if every cycle produced a delta (no underflow) and every
    /// pair of consecutive cycle deltas agrees within `tolerance` (Phase 9's
    /// "repeated observation" requirement).
    pub(crate) fn cycles_agree(deltas: &[Option<u64>], tolerance: u64) -> bool {
        if deltas.is_empty() {
            return false;
        }
        let mut known = Vec::with_capacity(deltas.len());
        for delta in deltas {
            match delta {
                Some(value) => known.push(*value),
                None => return false,
            }
        }
        known.windows(2).all(|w| w[0].abs_diff(w[1]) <= tolerance)
    }

    async fn query_compute_apps() -> Vec<ComputeApp> {
        let output = tokio::process::Command::new("nvidia-smi")
            .args([
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .await;
        match output {
            Ok(output) if output.status.success() => {
                parse_compute_apps_csv(&String::from_utf8_lossy(&output.stdout))
            }
            _ => Vec::new(),
        }
    }

    /// Total-device VRAM used across every reported GPU, in bytes. Reuses
    /// the same `nvidia-smi --query-gpu` parsing the live GPU-metrics panel
    /// already relies on (`crate::gpu::nvidia::parse_nvidia_csv`), off the
    /// async runtime thread since it spawns a real child process.
    async fn total_device_used_bytes(app_config: &AppConfig) -> Option<u64> {
        let backend = crate::gpu::detect_backend(&app_config.gpu_backend);
        let metrics = tokio::task::spawn_blocking(move || backend.read_metrics().ok())
            .await
            .ok()??;
        Some(
            metrics
                .values()
                .map(|metric| metric.vram_used * 1024 * 1024)
                .sum::<u64>(),
        )
    }

    /// Pre-spawn inventory and idle-stabilized `before` baseline, captured by
    /// `start_backend` immediately before `supervisor.start()` — the same
    /// timing slot as the Metal sampler's `pre_launch_wired_bytes`. Unlike
    /// that cheap in-process call, this drives real `nvidia-smi` processes,
    /// so `capture_before` gates on qualification first and returns `None`
    /// with zero I/O for every non-qualifying (i.e. most) launch.
    pub struct PreSpawnCapture {
        before_bytes: u64,
        stabilized: bool,
        pre_existing_apps: Vec<ComputeApp>,
    }

    /// Returns `None` when this host/launch does not qualify for exact
    /// evidence (mirrors `metal_sampler::spawn`'s gate, `target_os =
    /// "windows"` in place of `"macos"`) so `start_backend` can call this
    /// unconditionally and pay no cost on the common path.
    pub async fn capture_before(
        app_config: &AppConfig,
        preset: &ModelPreset,
    ) -> Option<PreSpawnCapture> {
        if !cfg!(target_os = "windows") {
            return None;
        }
        if preset.fit_enabled != Some(false) || !preset.extra_args.trim().is_empty() {
            return None;
        }

        let pre_existing_apps = query_compute_apps().await;

        let mut samples = vec![total_device_used_bytes(app_config).await?];
        for _ in 0..IDLE_STABILIZE_SAMPLES {
            tokio::time::sleep(IDLE_STABILIZE_INTERVAL).await;
            let Some(sample) = total_device_used_bytes(app_config).await else {
                break;
            };
            samples.push(sample);
            if find_stable_value(&samples, IDLE_STABILIZE_TOLERANCE_BYTES).1 {
                break;
            }
        }
        let (before_bytes, stabilized) =
            find_stable_value(&samples, IDLE_STABILIZE_TOLERANCE_BYTES);

        Some(PreSpawnCapture {
            before_bytes,
            stabilized,
            pre_existing_apps,
        })
    }

    /// Spawns the bounded post-readiness peak/after/repeat phase, detached
    /// from session control exactly like `metal_sampler::spawn` — this
    /// function only schedules the task, so it can never block a stop or
    /// restart.
    pub fn spawn(app_config: AppConfig, preset: ModelPreset, pid: u32, capture: PreSpawnCapture) {
        tokio::spawn(async move {
            run(app_config, preset, pid, capture).await;
        });
    }

    async fn run(app_config: AppConfig, preset: ModelPreset, pid: u32, capture: PreSpawnCapture) {
        let mut noise_flags = Vec::new();
        if !capture.stabilized {
            noise_flags.push(
                "idle stabilization did not converge before spawn; `before` baseline may include background drift"
                    .to_string(),
            );
        }

        let mut cycle_before = capture.before_bytes;
        let mut deltas = Vec::with_capacity(REPEAT_CYCLES as usize);
        let mut peak_bytes = capture.before_bytes;
        let mut after_bytes = capture.before_bytes;

        for _ in 0..REPEAT_CYCLES {
            let mut cycle_peak = cycle_before;
            for _ in 0..PEAK_SAMPLE_COUNT {
                tokio::time::sleep(PEAK_SAMPLE_INTERVAL).await;
                if let Some(sample) = total_device_used_bytes(&app_config).await {
                    cycle_peak = cycle_peak.max(sample);
                }
            }
            let cycle_after = total_device_used_bytes(&app_config)
                .await
                .unwrap_or(cycle_peak);
            deltas.push(cycle_peak.checked_sub(cycle_before));
            peak_bytes = peak_bytes.max(cycle_peak);
            after_bytes = cycle_after;
            cycle_before = cycle_after;
        }

        if !cycles_agree(&deltas, IDLE_STABILIZE_TOLERANCE_BYTES) {
            noise_flags.push(format!(
                "repeated observation cycles did not agree within tolerance: {deltas:?}"
            ));
        }

        let after_apps = query_compute_apps().await;
        noise_flags.extend(diff_background_processes(
            &capture.pre_existing_apps,
            &after_apps,
            pid,
            IDLE_STABILIZE_TOLERANCE_BYTES,
        ));

        let Ok(identity) = crate::inference::llama_cpp_capabilities::ExecutableIdentity::from_path(
            &app_config.llama_server_path,
        ) else {
            return;
        };
        let Some(capabilities) =
            crate::inference::llama_cpp_capabilities::cached_snapshot(&identity)
        else {
            return;
        };
        let Ok(mut fingerprint) = current_fingerprint(&app_config, &preset, &capabilities) else {
            return;
        };
        fingerprint.method = Some(LaunchEvidenceMethod::WddmTotalDeviceDelta);

        let sample = LaunchSample {
            before_bytes: capture.before_bytes,
            peak_bytes,
            after_bytes,
            sample_count: PEAK_SAMPLE_COUNT * REPEAT_CYCLES,
            interval_ms: PEAK_SAMPLE_INTERVAL.as_millis() as u32,
            noise_flags,
            captured_unix_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_millis())
                .unwrap_or_default(),
        };
        let Ok(receipt) = build_launch_observation(&preset, fingerprint, FitState::Off, sample)
        else {
            return;
        };
        let _ = store::save(&app_config.app_paths.launch_evidence_dir(), &receipt);
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

    /// One mutation per memory-relevant normalized argv field (Phase 9 fixture
    /// corpus requirement). Field coverage is asserted against
    /// `classify_argv_field` itself, so a newly added `MemoryRelevant` field
    /// with no mutation fixture here fails this test rather than silently
    /// going unverified.
    #[test]
    fn manifest_digest_changes_for_every_memory_relevant_field() {
        let baseline_json = serde_json::json!({
            "name": "fixture",
            "model_path": "/models/a.gguf",
            "context_size": 4096,
            "ctk": "f16",
            "ctv": "f16",
            "tensor_split": "",
            "batch_size": 512,
            "ubatch_size": 512,
            "no_mmap": false,
            "load_mode": "mmap",
            "swa_full": false,
            "ctx_checkpoints": 4,
            "checkpoint_min_step": 8,
            "cache_reuse": 16,
            "parallel_slots": 1,
            "n_cpu_moe": 4,
            "gpu_layers": 20,
            "mlock": false,
            "flash_attn": "on",
            "split_mode": "layer",
            "main_gpu": 1,
            "draft_model": "/models/draft.gguf",
            "draft_min": 1,
            "draft_max": 8,
            "spec_ngram_size": 4,
            "spec_type": "ngram-simple",
            "spec_default": false,
            "spec_draft_n_max": 16,
            "spec_draft_n_min": 1,
            "spec_draft_p_split": 0.5,
            "spec_draft_p_min": 0.1,
            "spec_draft_ngl": 10,
            "spec_draft_device": "cuda:0",
            "spec_draft_cpu_moe": false,
            "spec_draft_n_cpu_moe": 2,
            "spec_draft_type_k": "q8_0",
            "spec_draft_type_v": "q8_0",
            "kv_unified": true,
            "cache_idle_slots": true,
            "cache_ram_mib": 1024,
            "fit_enabled": false,
            "fit_ctx": 4096,
            "fit_target": "balanced",
            "mmproj": "/models/mmproj.gguf",
            "image_min_tokens": 64,
            "image_max_tokens": 1024,
            "mmproj_offload": true,
        });

        let mutations: &[(&str, serde_json::Value)] = &[
            ("model_path", serde_json::json!("/models/b.gguf")),
            ("context_size", serde_json::json!(8192)),
            ("ctk", serde_json::json!("q8_0")),
            ("ctv", serde_json::json!("q8_0")),
            ("tensor_split", serde_json::json!("0.5,0.5")),
            ("batch_size", serde_json::json!(1024)),
            ("ubatch_size", serde_json::json!(256)),
            ("no_mmap", serde_json::json!(true)),
            ("load_mode", serde_json::json!("dio")),
            ("swa_full", serde_json::json!(true)),
            ("ctx_checkpoints", serde_json::json!(8)),
            ("checkpoint_min_step", serde_json::json!(16)),
            ("cache_reuse", serde_json::json!(32)),
            ("parallel_slots", serde_json::json!(2)),
            ("n_cpu_moe", serde_json::json!(8)),
            ("gpu_layers", serde_json::json!(40)),
            ("mlock", serde_json::json!(true)),
            ("flash_attn", serde_json::json!("off")),
            ("split_mode", serde_json::json!("row")),
            ("main_gpu", serde_json::json!(2)),
            ("draft_model", serde_json::json!("/models/draft2.gguf")),
            ("draft_min", serde_json::json!(2)),
            ("draft_max", serde_json::json!(16)),
            ("spec_ngram_size", serde_json::json!(8)),
            ("spec_type", serde_json::json!("ngram-map-k")),
            ("spec_default", serde_json::json!(true)),
            ("spec_draft_n_max", serde_json::json!(32)),
            ("spec_draft_n_min", serde_json::json!(2)),
            ("spec_draft_p_split", serde_json::json!(0.75)),
            ("spec_draft_p_min", serde_json::json!(0.2)),
            ("spec_draft_ngl", serde_json::json!(20)),
            ("spec_draft_device", serde_json::json!("cuda:1")),
            ("spec_draft_cpu_moe", serde_json::json!(true)),
            ("spec_draft_n_cpu_moe", serde_json::json!(4)),
            ("spec_draft_type_k", serde_json::json!("q4_0")),
            ("spec_draft_type_v", serde_json::json!("q4_0")),
            ("kv_unified", serde_json::json!(false)),
            ("cache_idle_slots", serde_json::json!(false)),
            ("cache_ram_mib", serde_json::json!(2048)),
            ("fit_enabled", serde_json::json!(true)),
            ("fit_ctx", serde_json::json!(8192)),
            ("fit_target", serde_json::json!("aggressive")),
            ("mmproj", serde_json::json!("/models/mmproj2.gguf")),
            ("image_min_tokens", serde_json::json!(128)),
            ("image_max_tokens", serde_json::json!(2048)),
            ("mmproj_offload", serde_json::json!(false)),
        ];

        let covered: std::collections::BTreeSet<String> =
            mutations.iter().map(|(name, _)| name.to_string()).collect();
        let default_value = serde_json::to_value(ModelPreset::default()).expect("default preset");
        let serde_json::Value::Object(default_fields) = default_value else {
            panic!("preset serializes to an object");
        };
        let all_memory_relevant: std::collections::BTreeSet<String> = default_fields
            .keys()
            .filter(|name| classify_argv_field(name) == Some(ArgvFieldClass::MemoryRelevant))
            .cloned()
            .collect();
        assert_eq!(
            covered, all_memory_relevant,
            "mutation fixture set does not match classify_argv_field's MemoryRelevant fields"
        );

        let baseline: ModelPreset = serde_json::from_value(baseline_json).expect("baseline preset");
        let baseline_digest = manifest_digest(&baseline).expect("baseline digest");

        for (field, mutated) in mutations {
            let mut json = serde_json::to_value(&baseline).expect("serialize baseline");
            json[field] = mutated.clone();
            let mutated_preset: ModelPreset = serde_json::from_value(json).expect("mutated preset");
            let mutated_digest = manifest_digest(&mutated_preset).expect("mutated digest");
            assert_ne!(
                baseline_digest, mutated_digest,
                "mutating memory-relevant field `{field}` did not change the manifest digest"
            );
        }
    }

    /// Platform-labelled fixtures for the two non-macOS direct-observation
    /// methods (Phase 9 fixture corpus). Neither sampler is implemented on
    /// this machine (no Windows/CUDA hardware reachable from this session —
    /// see `docs/plans/evidence/preset-bundles/windows-cuda-sampler-design.md`),
    /// but the vocabulary itself — method identity, `is_direct_observation`,
    /// and match-class gating — is platform-independent and fully testable
    /// here.
    #[test]
    fn windows_wddm_and_cuda_rocm_methods_are_direct_observations_with_distinct_identity() {
        assert!(LaunchEvidenceMethod::WddmTotalDeviceDelta.is_direct_observation());
        assert!(LaunchEvidenceMethod::CudaRocmProcessDelta.is_direct_observation());

        let wddm = fingerprint(
            LaunchEvidenceMethod::WddmTotalDeviceDelta,
            "evidence-v1:same",
        );
        let mut cuda_same_digest = wddm.clone();
        cuda_same_digest.method = Some(LaunchEvidenceMethod::CudaRocmProcessDelta);

        // Method is part of launch identity: an otherwise-identical receipt
        // captured by the other platform's sampler must never match, even
        // with the same manifest digest and runtime/hardware fields.
        assert!(classify_evidence_match(&cuda_same_digest, &wddm, 0, 60_000).is_none());
    }

    #[test]
    fn estimator_only_and_fit_probe_never_match_or_power_measured_evidence() {
        let expected = fingerprint(
            LaunchEvidenceMethod::WddmTotalDeviceDelta,
            "evidence-v1:win",
        );

        assert!(!LaunchEvidenceMethod::EstimatorOnly.is_direct_observation());
        assert!(!LaunchEvidenceMethod::FitProbe.is_direct_observation());

        let mut estimator_only = expected.clone();
        estimator_only.method = Some(LaunchEvidenceMethod::EstimatorOnly);
        assert!(classify_evidence_match(&estimator_only, &expected, 0, 60_000).is_none());

        let mut fit_probe = expected.clone();
        fit_probe.method = Some(LaunchEvidenceMethod::FitProbe);
        assert!(classify_evidence_match(&fit_probe, &expected, 0, 60_000).is_none());
    }

    #[test]
    fn negative_delta_from_noisy_sampling_never_underflows_into_a_bogus_positive_number() {
        let preset = ModelPreset {
            model_path: "/models/a.gguf".into(),
            context_size: 4096,
            ..Default::default()
        };
        let fingerprint = LaunchEvidenceFingerprint {
            method: Some(LaunchEvidenceMethod::WddmTotalDeviceDelta),
            ..Default::default()
        };
        // Background GPU usage from another process fell during the sampling
        // window, so `peak` reads lower than `before` even though this
        // launch's own allocation only grew. `checked_sub` must yield `None`
        // rather than wrapping into a huge bogus delta.
        let sample = LaunchSample {
            before_bytes: 20_000_000_000,
            peak_bytes: 18_000_000_000,
            after_bytes: 19_000_000_000,
            sample_count: 6,
            interval_ms: 750,
            noise_flags: vec![
                "background GPU process pid=1234 released memory during sampling window".into(),
            ],
            captured_unix_ms: 1_700_000_000_000,
        };
        let receipt = build_launch_observation(&preset, fingerprint, FitState::Off, sample)
            .expect("receipt still builds; noise is recorded, not fatal");
        assert_eq!(receipt.model_delta_bytes, None);
        assert!(!receipt.noise_flags.is_empty());
    }

    #[test]
    fn incomplete_sampling_window_is_recorded_not_hidden() {
        let preset = ModelPreset {
            model_path: "/models/a.gguf".into(),
            context_size: 4096,
            ..Default::default()
        };
        let fingerprint = LaunchEvidenceFingerprint {
            method: Some(LaunchEvidenceMethod::MetalUnifiedObservation),
            ..Default::default()
        };
        // The process exited mid-poll (e.g. crashed after readiness): only 2
        // of the intended 6 samples were taken before it disappeared.
        let sample = LaunchSample {
            before_bytes: 10_000_000_000,
            peak_bytes: 15_000_000_000,
            after_bytes: 15_000_000_000,
            sample_count: 2,
            interval_ms: 750,
            noise_flags: vec![
                "sampling window ended early: process exited after 2 of 6 samples".into(),
            ],
            captured_unix_ms: 1_700_000_000_000,
        };
        let receipt = build_launch_observation(&preset, fingerprint, FitState::Off, sample)
            .expect("receipt still builds for a short window; incompleteness is a noise flag");
        assert_eq!(receipt.sample_count, 2);
        assert!(receipt.noise_flags[0].contains("ended early"));
    }

    fn test_app_config(config_dir: &std::path::Path) -> AppConfig {
        use clap::Parser;
        AppConfig::from_args(crate::cli::AppArgs::parse_from([
            "llama-monitor",
            "--config-dir",
            config_dir.to_str().unwrap(),
            "--llama-server-path",
            "llama-server",
            "--gpu-backend",
            "none",
        ]))
    }

    /// Real capture from Ryne (RTX 5090, driver 616.56,
    /// `nvidia-smi --query-compute-apps=pid,used_memory
    /// --format=csv,noheader,nounits`), 2026-09-02 — not a guessed format.
    /// Every `used_memory` cell is `[N/A]` under WDDM, confirming the design
    /// doc's open question: per-process attribution is not available on this
    /// real host, which is why `nvidia_sampler` uses this query only for
    /// PID-presence diffing, never as a memory source.
    #[test]
    fn parse_compute_apps_csv_handles_real_wddm_na_output() {
        let csv = include_str!("../../tests/fixtures/nvidia_smi_compute_apps_csv.txt");
        let apps = nvidia_sampler::parse_compute_apps_csv(csv);
        assert_eq!(apps.len(), 15);
        assert!(
            apps.iter().all(|app| app.used_memory_bytes.is_none()),
            "every row in the real WDDM fixture reports [N/A] for used_memory"
        );
        assert!(apps.iter().any(|app| app.pid == 47360));
    }

    #[test]
    fn parse_compute_apps_csv_parses_a_real_memory_value_when_present() {
        let apps = nvidia_sampler::parse_compute_apps_csv("1234, 2048\n5678, [N/A]\n");
        assert_eq!(apps.len(), 2);
        assert_eq!(apps[0].pid, 1234);
        assert_eq!(apps[0].used_memory_bytes, Some(2048 * 1024 * 1024));
        assert_eq!(apps[1].pid, 5678);
        assert_eq!(apps[1].used_memory_bytes, None);
    }

    #[test]
    fn find_stable_value_detects_convergence_within_tolerance() {
        let samples = vec![20_000_000_000, 20_020_000_000, 20_021_000_000];
        let (value, stabilized) = nvidia_sampler::find_stable_value(&samples, 16 * 1024 * 1024);
        assert!(stabilized);
        assert_eq!(value, 20_021_000_000);
    }

    #[test]
    fn find_stable_value_reports_unstabilized_when_still_drifting() {
        let samples = vec![20_000_000_000, 21_000_000_000, 22_000_000_000];
        let (value, stabilized) = nvidia_sampler::find_stable_value(&samples, 16 * 1024 * 1024);
        assert!(!stabilized);
        assert_eq!(value, 22_000_000_000);
    }

    #[test]
    fn diff_background_processes_flags_appearance_disappearance_and_growth() {
        use nvidia_sampler::ComputeApp;
        let before = vec![
            ComputeApp {
                pid: 100,
                used_memory_bytes: Some(1_000_000_000),
            },
            ComputeApp {
                pid: 200,
                used_memory_bytes: Some(500_000_000),
            },
        ];
        let after = vec![
            // pid 100 grew well past tolerance.
            ComputeApp {
                pid: 100,
                used_memory_bytes: Some(2_000_000_000),
            },
            // pid 200 disappeared.
            // pid 300 appeared.
            ComputeApp {
                pid: 300,
                used_memory_bytes: None,
            },
            // pid 9 is the launched process itself; must never be flagged.
            ComputeApp {
                pid: 9,
                used_memory_bytes: Some(999_000_000),
            },
        ];
        let flags = nvidia_sampler::diff_background_processes(&before, &after, 9, 16 * 1024 * 1024);
        assert!(
            flags
                .iter()
                .any(|f| f.contains("pid=100") && f.contains("changed by"))
        );
        assert!(
            flags
                .iter()
                .any(|f| f.contains("pid=200") && f.contains("disappeared"))
        );
        assert!(
            flags
                .iter()
                .any(|f| f.contains("pid=300") && f.contains("appeared"))
        );
        assert!(!flags.iter().any(|f| f.contains("pid=9")));
    }

    #[test]
    fn diff_background_processes_does_not_flag_na_memory_as_a_change() {
        use nvidia_sampler::ComputeApp;
        let before = vec![ComputeApp {
            pid: 100,
            used_memory_bytes: None,
        }];
        let after = vec![ComputeApp {
            pid: 100,
            used_memory_bytes: None,
        }];
        let flags = nvidia_sampler::diff_background_processes(&before, &after, 9, 16 * 1024 * 1024);
        assert!(flags.is_empty());
    }

    #[test]
    fn cycles_agree_requires_every_cycle_to_produce_a_delta_and_agree_within_tolerance() {
        let tolerance = 16 * 1024 * 1024;
        assert!(nvidia_sampler::cycles_agree(
            &[
                Some(1_000_000_000),
                Some(1_005_000_000),
                Some(1_002_000_000)
            ],
            tolerance
        ));
        assert!(!nvidia_sampler::cycles_agree(
            &[Some(1_000_000_000), Some(2_000_000_000)],
            tolerance
        ));
        assert!(!nvidia_sampler::cycles_agree(
            &[Some(1_000_000_000), None, Some(1_002_000_000)],
            tolerance
        ));
        assert!(!nvidia_sampler::cycles_agree(&[], tolerance));
    }

    #[tokio::test]
    async fn nvidia_capture_before_is_a_noop_off_windows_or_when_disqualified() {
        let temp = tempfile::tempdir().expect("tempdir");
        let app_config = test_app_config(temp.path());

        // On this machine (never Windows in CI/dev here), the target_os gate
        // alone must return None with zero I/O regardless of preset shape.
        let preset = ModelPreset {
            model_path: "/models/a.gguf".into(),
            fit_enabled: Some(false),
            ..Default::default()
        };
        let result = nvidia_sampler::capture_before(&app_config, &preset).await;
        assert!(result.is_none());
    }

    #[test]
    fn metal_sampler_spawn_is_a_noop_when_launch_does_not_qualify_for_exact_evidence() {
        let temp = tempfile::tempdir().expect("tempdir");
        let app_config = test_app_config(temp.path());

        // Fit not pinned off: the gate must return before ever reaching
        // `tokio::spawn`, so calling this outside a tokio runtime must not
        // panic.
        let disqualified_by_fit = ModelPreset {
            model_path: "/models/a.gguf".into(),
            fit_enabled: Some(true),
            ..Default::default()
        };
        metal_sampler::spawn(app_config.clone(), disqualified_by_fit, 0);

        // Non-empty extra_args: same no-tokio-runtime-needed guarantee.
        let disqualified_by_extra_args = ModelPreset {
            model_path: "/models/a.gguf".into(),
            fit_enabled: Some(false),
            extra_args: "--some-flag".into(),
            ..Default::default()
        };
        metal_sampler::spawn(app_config, disqualified_by_extra_args, 0);
    }
}
