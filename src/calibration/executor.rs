//! Bounded single-trial Calibration executor.
//!
//! This is deliberately narrower than the final Quick/Balanced search. It
//! proves the lifecycle boundary first: preflight, one managed sibling
//! `llama-bench` run, durable journal transitions, cancellation, and a
//! redacted receipt. It never stops an active server or mutates a preset.

use super::candidates::{
    BALANCED_MAX_VERIFICATION_CANDIDATES, QUICK_MAX_VERIFICATION_CANDIDATES, balanced_candidates,
    quick_candidates,
};
use super::jobs::{
    JournalEvent, JournalEventKind, append_event, append_trial_result, mark_recovered_crash,
    read_events, read_manifest, read_trial_results, recover_snapshot, suspected_crash_trials,
    write_manifest, write_snapshot,
};
use super::paths::is_safe_calibration_id;
use super::paths::{RegularFileError, require_regular_file};
use super::server_qualification::{
    self, QualificationCapabilities, QualificationReceipt, QualificationRequest, QualificationTrack,
};
use super::{
    CalibrationApplyRecord, CalibrationBaseline, CalibrationBaselineValue, CalibrationBudget,
    CalibrationCandidate, CalibrationCandidateResult, CalibrationFingerprint,
    CalibrationJobManifest, CalibrationJobSnapshot, CalibrationJobState, CalibrationMeasurement,
    CalibrationReceipt, CalibrationWorkload, GpuFingerprint, HardwareFingerprint,
    LlamaCppCalibrationPatch, ModelFingerprint, RuntimeFingerprint, StartCalibrationRequest,
    TrialStatus,
};
use crate::config::AppConfig;
use crate::inference::InferenceBackend;
use crate::inference::llama_cpp_capabilities::ExecutableIdentity;
use crate::inference::llama_cpp_tools::{
    LlamaCppTool, OptionalFitParams, ResolvedTool, ToolHelpEvidence, probe_help, resolve_tool,
};
use crate::llama::bench_runner::{
    BALANCED_BENCH_REPETITIONS, QUICK_BENCH_REPETITIONS, SweepPoint, llama_bench_path,
    run_sweep_with_tokens_repetitions,
};
use crate::presets::{self, ModelPreset};
use crate::state::AppState;
use anyhow::{Context, Result, anyhow, bail};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::{Notify, Semaphore};

const METHOD_VERSION_QUICK: &str = "calibration-v1-quick-single-trial";
const METHOD_VERSION_BALANCED: &str = "calibration-v1-balanced-bounded-search";
const CONFIRMATION: &str = "CALIBRATE";
const RESUME_CONFIRMATION: &str = "RESUME_CALIBRATION";

fn verification_count(budget: &CalibrationBudget) -> u32 {
    match budget {
        CalibrationBudget::Quick => QUICK_MAX_VERIFICATION_CANDIDATES as u32,
        CalibrationBudget::Balanced => BALANCED_MAX_VERIFICATION_CANDIDATES as u32,
        CalibrationBudget::Thorough => 0,
    }
}
const MAX_DIAGNOSTICS: usize = 24;

static JOBS: LazyLock<Mutex<BTreeMap<String, RuntimeJob>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));
static JOB_GATE: LazyLock<Arc<Semaphore>> = LazyLock::new(|| Arc::new(Semaphore::new(1)));

#[derive(Clone)]
struct RuntimeJob {
    snapshot: Arc<Mutex<CalibrationJobSnapshot>>,
    cancel: Arc<AtomicBool>,
    cancel_notify: Arc<Notify>,
    snapshot_path: PathBuf,
    journal_path: PathBuf,
    results_path: PathBuf,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CalibrationPreflight {
    pub preset_id: String,
    pub preset_fingerprint: String,
    pub backend: InferenceBackend,
    pub model_identity: String,
    pub server_identity: String,
    pub server_sha256: String,
    pub bench_identity: String,
    /// SHA-256 identity of the exact managed sibling used for measurement.
    pub bench_sha256: String,
    /// Optional helper is intentionally non-blocking: missing support only
    /// disables predictive pruning, never measured calibration.
    pub fit_params_sha256: Option<String>,
    /// Defaults parsed from the exact managed `llama-server --help` output.
    /// Informational only; preset values remain the calibration authority.
    pub server_help_sha256: Option<String>,
    pub server_help_defaults: BTreeMap<String, String>,
    pub server_help_exit_code: Option<i32>,
    pub server_help_output_truncated: bool,
    pub bench_help_sha256: Option<String>,
    pub bench_help_flags: Vec<String>,
    pub bench_help_exit_code: Option<i32>,
    pub fit_params_help_sha256: Option<String>,
    pub planned_trials: u32,
    pub candidate_ids: Vec<String>,
    pub requires_server_stop: bool,
    pub supported_budget: &'static str,
    pub requested_budget: CalibrationBudget,
    pub confirmation: &'static str,
}

#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(default)]
pub struct CalibrationPreflightRequest {
    pub preset_id: String,
    pub workload: CalibrationWorkload,
    pub budget: CalibrationBudget,
}

/// Public receipt projection. Keep this separate from the on-disk receipt so
/// future private fields (raw commands, paths, or diagnostics) cannot leak by
/// accidentally deriving API serialization from the protected record.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CalibrationReceiptView {
    pub schema_version: u32,
    pub method_version: String,
    pub job_id: String,
    pub fingerprint: super::CalibrationFingerprint,
    pub measurement: super::CalibrationMeasurement,
    pub baseline: super::CalibrationBaseline,
    pub budget: super::CalibrationBudget,
    pub candidate_results: Vec<super::CalibrationCandidateResult>,
    pub analysis: super::analysis::CalibrationAnalysis,
    pub selected_candidate: Option<String>,
    pub preset_id: String,
    pub preset_fingerprint: String,
    pub apply_history: Vec<super::CalibrationApplyRecord>,
    pub server_qualification: Option<QualificationReceipt>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CalibrationReceiptMatch {
    pub receipt: CalibrationReceiptView,
    /// `exact` is safe to describe as measured on this artifact/runtime;
    /// `compatible` is advisory evidence that must carry warnings in the UI.
    pub match_kind: String,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReceiptMatchKind {
    Exact,
    Compatible,
    Related,
}

impl ReceiptMatchKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Compatible => "compatible",
            Self::Related => "related",
        }
    }
}

#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(default)]
pub struct ApplyCalibrationRequest {
    pub target_preset_id: String,
    pub expected_target_fingerprint: String,
    pub candidate_id: Option<String>,
    #[serde(default = "default_create_derived")]
    pub create_derived: bool,
    pub exact_confirmation: Option<String>,
    #[serde(default = "default_validate_after_apply")]
    pub validate_after_apply: bool,
}

fn default_create_derived() -> bool {
    true
}

fn default_validate_after_apply() -> bool {
    true
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ApplyCalibrationResult {
    pub preset_id: String,
    pub derived: bool,
    pub candidate_id: String,
    pub before_fingerprint: String,
    pub after_fingerprint: String,
    pub validation: String,
}

#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(default)]
pub struct RollbackCalibrationRequest {
    pub expected_target_fingerprint: String,
    pub exact_confirmation: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RollbackCalibrationResult {
    pub preset_id: String,
    pub before_fingerprint: String,
    pub after_fingerprint: String,
}

pub fn confirmation_phrase() -> &'static str {
    CONFIRMATION
}

pub fn resume_confirmation_phrase() -> &'static str {
    RESUME_CONFIRMATION
}

pub fn preset_fingerprint(preset: &ModelPreset) -> Result<String> {
    let encoded = serde_json::to_vec(preset).context("serialize preset fingerprint")?;
    let digest = Sha256::digest(encoded);
    let encoded_digest = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Ok(format!("sha256:{encoded_digest}"))
}

pub fn preflight(
    config: &AppConfig,
    state: &AppState,
    preset_id: &str,
    workload: &CalibrationWorkload,
    budget: CalibrationBudget,
) -> Result<CalibrationPreflight> {
    let preset = find_preset(state, preset_id)?;
    if preset.backend != InferenceBackend::LlamaCpp {
        bail!("Calibration v1 supports llama.cpp presets only");
    }
    if *state.local_server_running.lock().unwrap() || *state.server_running.lock().unwrap() {
        bail!("Stop the active inference server before starting Calibration");
    }

    let model = resolve_model_path(config, &preset.model_path)?;
    let bench_tool =
        resolve_tool(&config.llama_server_path, LlamaCppTool::Bench).map_err(|error| {
            anyhow!(
                "Managed llama-bench is unavailable beside the configured llama-server: {error}"
            )
        })?;
    let bench = bench_tool.path.clone();
    let server_sha256 = sha256_file(&config.llama_server_path)?;
    let fit_params_sha256 =
        match crate::inference::llama_cpp_tools::optional_fit_params(&config.llama_server_path) {
            OptionalFitParams::Available(tool) => Some(tool.identity.file_hash),
            OptionalFitParams::Missing | OptionalFitParams::Unusable(_) => None,
        };

    let fingerprint = preset_fingerprint(&preset)?;
    validate_workload(workload)?;
    let candidates = match &budget {
        CalibrationBudget::Quick => quick_candidates(&preset, workload, None)?,
        CalibrationBudget::Balanced => balanced_candidates(&preset, workload, None)?,
        CalibrationBudget::Thorough => {
            bail!("Thorough Calibration is not available in the bounded 2.0 release")
        }
    };
    let candidate_ids = candidates
        .into_iter()
        .map(|candidate| candidate.id)
        .collect::<Vec<_>>();
    Ok(CalibrationPreflight {
        preset_id: preset.id.clone(),
        preset_fingerprint: fingerprint,
        backend: preset.backend,
        model_identity: library_relative_identity(config, &model),
        server_identity: config
            .llama_server_path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "llama-server".into()),
        server_sha256,
        bench_identity: bench
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "llama-bench".into()),
        bench_sha256: bench_tool.identity.file_hash,
        fit_params_sha256,
        server_help_sha256: None,
        server_help_defaults: BTreeMap::new(),
        server_help_exit_code: None,
        server_help_output_truncated: false,
        bench_help_sha256: None,
        bench_help_flags: Vec::new(),
        bench_help_exit_code: None,
        fit_params_help_sha256: None,
        planned_trials: candidate_ids.len() as u32 + verification_count(&budget),
        candidate_ids,
        requires_server_stop: true,
        supported_budget: match budget {
            CalibrationBudget::Quick => "quick_single_trial",
            CalibrationBudget::Balanced => "balanced_bounded_plan",
            CalibrationBudget::Thorough => unreachable!(),
        },
        requested_budget: budget,
        confirmation: CONFIRMATION,
    })
}

/// Attach bounded `--help` evidence to an already validated preflight. This
/// is kept separate so pure callers and tests do not spawn a process, while
/// the authenticated HTTP preflight can make capability eligibility explicit.
pub async fn enrich_preflight_with_help(
    config: &AppConfig,
    preflight: &mut CalibrationPreflight,
) -> Result<()> {
    let server = ResolvedTool {
        tool: LlamaCppTool::Server,
        path: config.llama_server_path.clone(),
        identity: ExecutableIdentity::from_path(&config.llama_server_path)?,
    };
    let server_evidence = probe_help(&server, &config.llama_server_cwd).await?;
    preflight.server_help_sha256 = Some(server_evidence.help_sha256.clone());
    preflight.server_help_defaults = server_evidence.defaults.clone();
    preflight.server_help_exit_code = server_evidence.exit_code;
    preflight.server_help_output_truncated = server_evidence.output_truncated;
    let bench = resolve_tool(&config.llama_server_path, LlamaCppTool::Bench)?;
    let evidence = probe_help(&bench, &config.llama_server_cwd).await?;
    apply_help_evidence(preflight, &evidence, false);
    if let OptionalFitParams::Available(fit) =
        crate::inference::llama_cpp_tools::optional_fit_params(&config.llama_server_path)
        && let Ok(evidence) = probe_help(&fit, &config.llama_server_cwd).await
    {
        apply_help_evidence(preflight, &evidence, true);
    }
    Ok(())
}

fn apply_help_evidence(
    preflight: &mut CalibrationPreflight,
    evidence: &ToolHelpEvidence,
    fit_params: bool,
) {
    if fit_params {
        preflight.fit_params_help_sha256 = Some(evidence.help_sha256.clone());
    } else {
        preflight.bench_help_sha256 = Some(evidence.help_sha256.clone());
        preflight.bench_help_flags = evidence.flags.iter().cloned().collect();
        preflight.bench_help_exit_code = evidence.exit_code;
    }
}

pub async fn start(
    config: Arc<AppConfig>,
    state: AppState,
    request: StartCalibrationRequest,
) -> Result<CalibrationJobSnapshot> {
    match request.budget {
        CalibrationBudget::Quick => {}
        CalibrationBudget::Balanced => {}
        CalibrationBudget::Thorough => {
            bail!("Thorough Calibration is not available in the bounded 2.0 release")
        }
    }
    if request.exact_confirmation.as_deref() != Some(CONFIRMATION) {
        bail!("Exact confirmation CALIBRATE is required to start the bounded trial");
    }
    if request.allow_stop_active_server {
        bail!("Calibration cannot stop or restart an active server yet");
    }
    validate_workload(&request.workload)?;

    let preset = find_preset(&state, &request.preset_id)?;
    if preset.backend != InferenceBackend::LlamaCpp {
        bail!("Calibration v1 supports llama.cpp presets only");
    }
    let preset_fingerprint = preset_fingerprint(&preset)?;
    if !request.expected_preset_fingerprint.is_empty()
        && request.expected_preset_fingerprint != preset_fingerprint
    {
        bail!("Preset changed since preflight; refresh before starting Calibration");
    }
    let mut preflight = preflight(
        &config,
        &state,
        &request.preset_id,
        &request.workload,
        request.budget.clone(),
    )?;
    enrich_preflight_with_help(&config, &mut preflight).await?;
    let candidates = match &request.budget {
        CalibrationBudget::Quick => quick_candidates(&preset, &request.workload, None)?,
        CalibrationBudget::Balanced => balanced_candidates(&preset, &request.workload, None)?,
        CalibrationBudget::Thorough => unreachable!(),
    };
    let model_path = resolve_model_path(&config, &preset.model_path)?;
    let bench_path = llama_bench_path(&config.llama_server_path);
    let calibration_fingerprint = build_calibration_fingerprint(
        &config,
        &preset,
        &request.workload,
        &preflight,
        &model_path,
        &preset_fingerprint,
    )?;
    let baseline = calibration_baseline(&preset, &preflight);
    let id = crate::config::generate_random_token()
        .chars()
        .take(24)
        .collect::<String>();
    let job_dir = config.app_paths.calibration_jobs_dir().join(&id);
    let snapshot_path = job_dir.join("snapshot.json");
    let journal_path = job_dir.join("journal.jsonl");
    let results_path = job_dir.join("trial-results.jsonl");
    let manifest_path = job_dir.join("manifest.json");
    let snapshot = CalibrationJobSnapshot {
        id: id.clone(),
        state: CalibrationJobState::Queued,
        phase: "queued".into(),
        completed_trials: 0,
        planned_trials: preflight.planned_trials,
        diagnostics: vec!["Preset and active session remain unchanged during the trial".into()],
        receipt_id: None,
    };
    write_snapshot(&snapshot_path, &snapshot)?;
    write_manifest(
        &manifest_path,
        &CalibrationJobManifest {
            schema_version: super::CALIBRATION_SCHEMA_VERSION,
            preset_id: preset.id.clone(),
            preset_fingerprint: preset_fingerprint.clone(),
            workload: request.workload.clone(),
            budget: request.budget.clone(),
            candidates: candidates.clone(),
            model_path: model_path.to_string_lossy().into_owned(),
            bench_path: bench_path.to_string_lossy().into_owned(),
            fingerprint: calibration_fingerprint.clone(),
            baseline: baseline.clone(),
            server_qualification: request.server_qualification.clone(),
        },
    )?;
    append_event(
        &journal_path,
        &JournalEvent::new(JournalEventKind::JobCreated, None),
    )?;

    let runtime = RuntimeJob {
        snapshot: Arc::new(Mutex::new(snapshot.clone())),
        cancel: Arc::new(AtomicBool::new(false)),
        cancel_notify: Arc::new(Notify::new()),
        snapshot_path,
        journal_path,
        results_path,
    };
    {
        let mut jobs = JOBS
            .lock()
            .map_err(|_| anyhow!("Calibration registry unavailable"))?;
        if jobs.values().any(|job| {
            job.snapshot.lock().ok().is_some_and(|snapshot| {
                matches!(
                    snapshot.state,
                    CalibrationJobState::Queued
                        | CalibrationJobState::Running
                        | CalibrationJobState::Cancelling
                )
            })
        }) {
            bail!("Another Calibration job is already active");
        }
        jobs.insert(id.clone(), runtime.clone());
    }

    tokio::spawn(run_job(
        id,
        config,
        state.clone(),
        preset,
        request.workload,
        candidates,
        model_path,
        bench_path,
        calibration_fingerprint,
        baseline,
        request.budget,
        request.server_qualification,
        runtime,
        Vec::new(),
    ));
    Ok(snapshot)
}

/// Explicitly resume a job recovered from a suspected crash. Finished trial
/// results are loaded from the protected journal and are never silently rerun;
/// only unfinished trials are eligible for the resumed execution.
pub fn resume(
    config: Arc<AppConfig>,
    state: AppState,
    id: &str,
    confirmation: &str,
) -> Result<Option<CalibrationJobSnapshot>> {
    validate_job_id(id)?;
    if confirmation != RESUME_CONFIRMATION {
        bail!("Exact confirmation RESUME_CALIBRATION required");
    }
    {
        let jobs = JOBS
            .lock()
            .map_err(|_| anyhow!("Calibration registry unavailable"))?;
        if jobs.values().any(|job| {
            job.snapshot.lock().ok().is_some_and(|snapshot| {
                matches!(
                    snapshot.state,
                    CalibrationJobState::Queued
                        | CalibrationJobState::Running
                        | CalibrationJobState::Cancelling
                )
            })
        }) {
            bail!("Another Calibration job is already active");
        }
    }

    let job_dir = config.app_paths.calibration_jobs_dir().join(id);
    let snapshot_path = job_dir.join("snapshot.json");
    let journal_path = job_dir.join("journal.jsonl");
    let results_path = job_dir.join("trial-results.jsonl");
    let manifest_path = job_dir.join("manifest.json");
    let Some(mut snapshot) = recover_snapshot(&snapshot_path)? else {
        return Ok(None);
    };
    if !matches!(snapshot.state, CalibrationJobState::Failed) {
        bail!("Calibration job is not resumable");
    }
    if !snapshot.phase.contains("suspected_crash") {
        bail!("Calibration job did not recover from a suspected crash");
    }
    let manifest = read_manifest(&manifest_path)?
        .ok_or_else(|| anyhow!("Calibration resume manifest not found"))?;
    let preset = find_preset(&state, &manifest.preset_id)?;
    let current_fingerprint = preset_fingerprint(&preset)?;
    if current_fingerprint != manifest.preset_fingerprint {
        bail!("Preset changed since Calibration; refresh before resuming");
    }
    let model_path = require_regular_file(Path::new(&manifest.model_path))
        .map_err(|error| anyhow!("Calibration model is no longer available: {error:?}"))?;
    let bench_path = require_regular_file(Path::new(&manifest.bench_path))
        .map_err(|error| anyhow!("Calibration bench binary is no longer available: {error:?}"))?;
    let events = read_events(&journal_path)?;
    let suspected = suspected_crash_trials(&events);
    for trial_id in suspected {
        append_event(
            &journal_path,
            &JournalEvent::new(JournalEventKind::TrialAbandoned, Some(trial_id)),
        )?;
    }
    let prior_results = read_trial_results(&results_path)?;
    snapshot.state = CalibrationJobState::Queued;
    snapshot.phase = "resuming".into();
    snapshot
        .diagnostics
        .push("Explicit resume confirmed; finished trials will not be repeated".into());
    snapshot.diagnostics.truncate(MAX_DIAGNOSTICS);
    write_snapshot(&snapshot_path, &snapshot)?;
    append_event(
        &journal_path,
        &JournalEvent::new(JournalEventKind::JobResumed, None),
    )?;
    let runtime = RuntimeJob {
        snapshot: Arc::new(Mutex::new(snapshot.clone())),
        cancel: Arc::new(AtomicBool::new(false)),
        cancel_notify: Arc::new(Notify::new()),
        snapshot_path,
        journal_path,
        results_path,
    };
    JOBS.lock()
        .map_err(|_| anyhow!("Calibration registry unavailable"))?
        .insert(id.to_string(), runtime.clone());
    tokio::spawn(run_job(
        id.to_string(),
        config,
        state.clone(),
        preset,
        manifest.workload,
        manifest.candidates,
        model_path,
        bench_path,
        manifest.fingerprint,
        manifest.baseline,
        manifest.budget,
        manifest.server_qualification,
        runtime,
        prior_results,
    ));
    Ok(Some(snapshot))
}

pub fn get(config: &AppConfig, id: &str) -> Result<Option<CalibrationJobSnapshot>> {
    validate_job_id(id)?;
    if let Some(runtime) = JOBS
        .lock()
        .map_err(|_| anyhow!("Calibration registry unavailable"))?
        .get(id)
        .cloned()
    {
        return Ok(runtime
            .snapshot
            .lock()
            .ok()
            .map(|snapshot| snapshot.clone()));
    }
    let job_dir = config.app_paths.calibration_jobs_dir().join(id);
    let path = job_dir.join("snapshot.json");
    let Some(mut snapshot) = recover_snapshot(&path)? else {
        return Ok(None);
    };
    let journal = read_events(&job_dir.join("journal.jsonl"))?;
    let suspected = suspected_crash_trials(&journal);
    if !suspected.is_empty() {
        mark_recovered_crash(&mut snapshot, suspected.len());
        write_snapshot(&path, &snapshot)?;
    }
    Ok(Some(snapshot))
}

/// List durable job snapshots from the active application home. Terminal and
/// recovered jobs are retained until the user explicitly forgets them.
pub fn list(config: &AppConfig) -> Result<Vec<CalibrationJobSnapshot>> {
    let root = config.app_paths.calibration_jobs_dir();
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut snapshots = Vec::new();
    for entry in fs::read_dir(&root).context("list Calibration job directory")? {
        let entry = entry.context("read Calibration job entry")?;
        let file_type = entry.file_type().context("inspect Calibration job entry")?;
        if !file_type.is_dir() {
            continue;
        }
        let id = entry.file_name().to_string_lossy().into_owned();
        if !is_safe_calibration_id(&id) {
            continue;
        }
        if let Some(snapshot) = get(config, &id)? {
            snapshots.push(snapshot);
        }
    }
    snapshots.sort_by(|left, right| left.id.cmp(&right.id));
    Ok(snapshots)
}

/// Forget a terminal calibration job, its receipt, and any private rollback
/// backups referenced by that receipt. Active jobs must be cancelled and
/// allowed to reach a terminal state first.
pub fn forget(config: &AppConfig, id: &str, confirmation: &str) -> Result<bool> {
    if confirmation != FORGET_CONFIRMATION {
        bail!("Exact confirmation FORGET_CALIBRATION required");
    }
    validate_job_id(id)?;
    let runtime = JOBS
        .lock()
        .map_err(|_| anyhow!("Calibration registry unavailable"))?
        .get(id)
        .cloned();
    if let Some(runtime) = runtime {
        let snapshot = runtime
            .snapshot
            .lock()
            .map_err(|_| anyhow!("Calibration job unavailable"))?
            .clone();
        if matches!(
            snapshot.state,
            CalibrationJobState::Queued
                | CalibrationJobState::Running
                | CalibrationJobState::Cancelling
        ) {
            bail!("active Calibration job must be cancelled before forgetting");
        }
    }

    let receipt = get_receipt(config, id)?;
    if let Some(receipt) = receipt {
        for record in receipt.apply_history {
            if is_safe_calibration_id(&record.rollback_id) {
                let backup = config
                    .app_paths
                    .calibration_apply_backups_dir()
                    .join(format!("{}.json", record.rollback_id));
                if backup.exists() {
                    fs::remove_file(backup).context("remove Calibration rollback backup")?;
                }
            }
        }
    }

    let job_dir = config.app_paths.calibration_jobs_dir().join(id);
    if job_dir.exists() {
        let metadata = fs::symlink_metadata(&job_dir).context("inspect Calibration job")?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            bail!("Calibration job path is not a directory");
        }
        fs::remove_dir_all(&job_dir).context("remove Calibration job")?;
    }
    let receipt_path = config
        .app_paths
        .calibration_receipts_dir()
        .join(format!("{id}.json"));
    if receipt_path.exists() {
        fs::remove_file(receipt_path).context("remove Calibration receipt")?;
    }
    JOBS.lock()
        .map_err(|_| anyhow!("Calibration registry unavailable"))?
        .remove(id);
    Ok(true)
}

pub fn get_receipt(config: &AppConfig, id: &str) -> Result<Option<CalibrationReceipt>> {
    validate_job_id(id)?;
    let path = config
        .app_paths
        .calibration_receipts_dir()
        .join(format!("{id}.json"));
    if !path.exists() {
        return Ok(None);
    }
    let encoded = fs::read(path)?;
    Ok(Some(
        serde_json::from_slice(&encoded).context("Calibration receipt is invalid")?,
    ))
}

pub fn get_receipt_view(config: &AppConfig, id: &str) -> Result<Option<CalibrationReceiptView>> {
    Ok(
        get_receipt(config, id)?.map(|receipt| CalibrationReceiptView {
            schema_version: receipt.schema_version,
            method_version: receipt.method_version,
            job_id: receipt.job_id,
            fingerprint: receipt.fingerprint,
            measurement: receipt.measurement,
            baseline: receipt.baseline,
            budget: receipt.budget,
            candidate_results: receipt.candidate_results,
            analysis: receipt.analysis,
            selected_candidate: receipt.selected_candidate,
            preset_id: receipt.preset_id,
            preset_fingerprint: receipt.preset_fingerprint,
            apply_history: receipt.apply_history,
            server_qualification: receipt.server_qualification,
        }),
    )
}

fn classify_receipt_match(
    receipt: &CalibrationFingerprint,
    expected: &CalibrationFingerprint,
) -> Option<(ReceiptMatchKind, Vec<String>)> {
    let mut expected_legacy = expected.clone();
    expected_legacy.model.compatibility_key.clear();
    expected_legacy.model.family_key.clear();
    expected_legacy.model.quantization_signature.clear();
    expected_legacy.runtime.capability_signature.clear();
    let mut receipt_legacy = receipt.clone();
    receipt_legacy.model.compatibility_key.clear();
    receipt_legacy.model.family_key.clear();
    receipt_legacy.model.quantization_signature.clear();
    receipt_legacy.runtime.capability_signature.clear();

    let legacy_exact = receipt.model.compatibility_key.is_empty()
        && receipt.model.family_key.is_empty()
        && receipt.runtime.capability_signature.is_empty()
        && receipt_legacy == expected_legacy;
    let exact = receipt == expected || legacy_exact;
    let compatible = !exact
        && receipt.backend == expected.backend
        && receipt.hardware == expected.hardware
        && receipt.workload == expected.workload
        && receipt.factor_catalog_version == expected.factor_catalog_version
        && !expected.model.family_key.is_empty()
        && receipt.model.family_key == expected.model.family_key
        && receipt.model.quantization_signature == expected.model.quantization_signature
        && !expected.runtime.capability_signature.is_empty()
        && receipt.runtime.capability_signature == expected.runtime.capability_signature;
    let related = !exact
        && !compatible
        && receipt.backend == expected.backend
        && receipt.hardware == expected.hardware
        && receipt.workload == expected.workload
        && receipt.factor_catalog_version == expected.factor_catalog_version
        && !expected.model.family_key.is_empty()
        && receipt.model.family_key == expected.model.family_key
        && !expected.runtime.capability_signature.is_empty()
        && receipt.runtime.capability_signature == expected.runtime.capability_signature;

    let kind = if exact {
        ReceiptMatchKind::Exact
    } else if compatible {
        ReceiptMatchKind::Compatible
    } else if related {
        ReceiptMatchKind::Related
    } else {
        return None;
    };

    let mut warnings = Vec::new();
    if legacy_exact {
        warnings.push("Legacy artifact/runtime metadata matched".into());
    }
    if kind != ReceiptMatchKind::Exact
        && receipt.model.content_fingerprint != expected.model.content_fingerprint
    {
        warnings.push("Different GGUF artifact; introspected structure matched".into());
    }
    if kind != ReceiptMatchKind::Exact
        && (receipt.runtime.server_sha256 != expected.runtime.server_sha256
            || receipt.runtime.bench_sha256 != expected.runtime.bench_sha256)
    {
        warnings.push("Different llama.cpp runtime build; capability signature matched".into());
    }
    if kind != ReceiptMatchKind::Exact
        && receipt.baseline_config_hash != expected.baseline_config_hash
    {
        warnings.push("Measured baseline differs from current wizard values".into());
    }
    if kind == ReceiptMatchKind::Related {
        warnings
            .push("Different GGUF weight quantization; family and structural shape matched".into());
    }
    Some((kind, warnings))
}

/// Return completed receipts whose full calibration fingerprint exactly
/// matches the current preset, workload, hardware, and managed runtime.
///
/// Spawn Wizard reuse is intentionally strict: a model-only or runtime-only
/// match is not sufficient because the receipt's baseline configuration is
/// part of the measured result. The caller must still review the candidate
/// before applying it to wizard controls.
pub async fn matching_receipts(
    config: &AppConfig,
    state: &AppState,
    preset_id: &str,
    workload: &CalibrationWorkload,
    budget: CalibrationBudget,
) -> Result<Vec<CalibrationReceiptMatch>> {
    let preflight = preflight(config, state, preset_id, workload, budget)
        .context("calibration match preflight")?;
    let mut enriched = preflight.clone();
    enrich_preflight_with_help(config, &mut enriched).await?;
    let preset = find_preset(state, preset_id)?;
    let model_path = resolve_model_path(config, &preset.model_path)?;
    let expected = build_calibration_fingerprint(
        config,
        &preset,
        workload,
        &enriched,
        &model_path,
        &enriched.preset_fingerprint,
    )?;
    let snapshots = list(config)?;
    let mut matches = Vec::new();
    for snapshot in snapshots {
        let Some(receipt_id) = snapshot.receipt_id.as_deref() else {
            continue;
        };
        let Some(receipt) = get_receipt(config, receipt_id)? else {
            continue;
        };
        let Some((kind, warnings)) = classify_receipt_match(&receipt.fingerprint, &expected) else {
            continue;
        };
        if let Some(view) = get_receipt_view(config, receipt_id)? {
            matches.push(CalibrationReceiptMatch {
                receipt: view,
                match_kind: kind.as_str().into(),
                warnings,
            });
        }
    }
    matches.sort_by(|left, right| right.receipt.job_id.cmp(&left.receipt.job_id));
    Ok(matches)
}

pub fn cancel(config: &AppConfig, id: &str) -> Result<Option<CalibrationJobSnapshot>> {
    validate_job_id(id)?;
    let runtime = JOBS
        .lock()
        .map_err(|_| anyhow!("Calibration registry unavailable"))?
        .get(id)
        .cloned();
    let Some(runtime) = runtime else {
        return get(config, id);
    };
    runtime.cancel.store(true, Ordering::Release);
    runtime.cancel_notify.notify_waiters();
    update_snapshot(&runtime, |snapshot| {
        if matches!(
            snapshot.state,
            CalibrationJobState::Queued | CalibrationJobState::Running
        ) {
            snapshot.state = CalibrationJobState::Cancelling;
            snapshot.phase = "cancelling".into();
        }
    })?;
    Ok(runtime
        .snapshot
        .lock()
        .ok()
        .map(|snapshot| snapshot.clone()))
}

#[allow(clippy::too_many_arguments)]
async fn run_job(
    id: String,
    config: Arc<AppConfig>,
    state: AppState,
    preset: ModelPreset,
    workload: super::CalibrationWorkload,
    candidates: Vec<CalibrationCandidate>,
    model_path: PathBuf,
    bench_path: PathBuf,
    fingerprint: CalibrationFingerprint,
    baseline: CalibrationBaseline,
    budget: CalibrationBudget,
    server_qualification: Option<QualificationRequest>,
    runtime: RuntimeJob,
    mut candidate_results: Vec<CalibrationCandidateResult>,
) {
    let _permit = match JOB_GATE.clone().acquire_owned().await {
        Ok(permit) => permit,
        Err(_) => {
            let _ = fail_job(&runtime, "Calibration executor gate is unavailable");
            return;
        }
    };
    if runtime.cancel.load(Ordering::Acquire) {
        let _ = finish_cancelled(&runtime);
        return;
    }
    if let Err(error) = transition(&runtime, CalibrationJobState::Running, "trial") {
        let _ = fail_job(
            &runtime,
            &format!("Calibration journal could not be prepared: {error}"),
        );
        return;
    }

    let finished_ids = candidate_results
        .iter()
        .map(|result| result.candidate.id.clone())
        .collect::<std::collections::BTreeSet<_>>();
    // Balanced screens candidates with a bounded dense-model workload, then
    // remeasures only finalists at the user's full workload below.
    let screen_workload = screening_workload(&workload, budget == CalibrationBudget::Balanced);
    for candidate in candidates {
        if finished_ids.contains(&candidate.id) {
            continue;
        }
        if runtime.cancel.load(Ordering::Acquire) {
            let _ = append_event(
                &runtime.journal_path,
                &JournalEvent::new(JournalEventKind::JobCancelled, Some(candidate.id)),
            );
            let _ = finish_cancelled(&runtime);
            return;
        }
        let candidate_id = candidate.id.clone();
        if let Err(error) = append_event(
            &runtime.journal_path,
            &JournalEvent::new(JournalEventKind::TrialPlanned, Some(candidate_id.clone())),
        )
        .and_then(|_| {
            append_event(
                &runtime.journal_path,
                &JournalEvent::new(JournalEventKind::TrialStarted, Some(candidate_id.clone())),
            )
        }) {
            let _ = fail_job(
                &runtime,
                &format!("Calibration journal could not be prepared: {error}"),
            );
            return;
        }
        let mut candidate_preset = preset.clone();
        apply_patch_to_preset(&mut candidate_preset, &candidate.typed_patch);
        let ngl = candidate_preset.gpu_layers.unwrap_or(99);
        // An unset/auto preset follows the managed runtime's Metal default,
        // which enables flash attention for quantized KV caches. Only an
        // explicit off/false value disables it for a calibration candidate.
        let flash_attn = calibration_flash_attn(&candidate_preset);
        // Match the product preset defaults: quantized K cache with an f16 V
        // cache. Forcing q8_0 for both sides can make valid models fail context
        // creation before any candidate is measured.
        let (ctk, ctv) = calibration_cache_types(&candidate_preset);
        let batch_size = if candidate_preset.batch_size == 0 {
            2048
        } else {
            candidate_preset.batch_size
        };
        let ubatch_size = if candidate_preset.ubatch_size == 0 {
            512
        } else {
            candidate_preset.ubatch_size
        };
        let depths = vec![screen_workload.minimum_context.max(1)];
        let model_path_string = model_path.to_string_lossy().into_owned();
        let snapshot = calibration_capability_snapshot(&config).await;
        let result = if let Some(error) = calibration_kv_policy_error(snapshot.as_ref(), &ctk, &ctv)
        {
            Some(Err(error))
        } else {
            let bench = run_sweep_with_tokens_repetitions(
                &bench_path,
                &config.llama_server_cwd,
                &model_path_string,
                ngl,
                flash_attn,
                &ctk,
                &ctv,
                batch_size,
                ubatch_size,
                &depths,
                candidate_preset.n_cpu_moe,
                screen_workload.prompt_tokens,
                screen_workload.generation_tokens,
                QUICK_BENCH_REPETITIONS,
            );
            tokio::select! {
                result = bench => Some(result),
                _ = runtime.cancel_notify.notified() => None,
            }
        };
        if runtime.cancel.load(Ordering::Acquire) {
            let _ = finish_cancelled(&runtime);
            return;
        }
        let measurement = match result {
            Some(Ok(points)) => {
                let mut measurement = measurement_from_points(points);
                measurement.trial_id = candidate_id.clone();
                measurement
            }
            Some(Err(error)) => CalibrationMeasurement {
                trial_id: candidate_id.clone(),
                status: Some(TrialStatus::Error),
                bounded_diagnostics: vec![sanitize_error(&error)],
                ..Default::default()
            },
            None => {
                let _ = finish_cancelled(&runtime);
                return;
            }
        };
        candidate_results.push(CalibrationCandidateResult {
            candidate,
            measurement,
        });
        if let Some(result) = candidate_results.last()
            && let Err(error) = append_trial_result(&runtime.results_path, result)
        {
            let _ = fail_job(
                &runtime,
                &format!("Calibration trial result could not be persisted: {error}"),
            );
            return;
        }
        let _ = append_event(
            &runtime.journal_path,
            &JournalEvent::new(JournalEventKind::TrialFinished, Some(candidate_id)),
        );
        let _ = update_snapshot(&runtime, |snapshot| {
            snapshot.completed_trials = snapshot.completed_trials.saturating_add(1);
        });
    }

    if matches!(
        budget,
        CalibrationBudget::Quick | CalibrationBudget::Balanced
    ) {
        let verification_candidates = select_verification_candidates(
            &candidate_results,
            verification_count(&budget) as usize,
        );
        for candidate in verification_candidates {
            if runtime.cancel.load(Ordering::Acquire) {
                let _ = finish_cancelled(&runtime);
                return;
            }
            let trial_id = format!("{}-verification", candidate.id);
            if candidate_results
                .iter()
                .any(|result| result.candidate.id == trial_id)
            {
                continue;
            }
            let journal_result = append_event(
                &runtime.journal_path,
                &JournalEvent::new(JournalEventKind::TrialPlanned, Some(trial_id.clone())),
            )
            .and_then(|_| {
                append_event(
                    &runtime.journal_path,
                    &JournalEvent::new(JournalEventKind::TrialStarted, Some(trial_id.clone())),
                )
            });
            if let Err(error) = journal_result {
                let _ = fail_job(
                    &runtime,
                    &format!("Calibration verification journal failed: {error}"),
                );
                return;
            }
            let mut candidate_preset = preset.clone();
            apply_patch_to_preset(&mut candidate_preset, &candidate.typed_patch);
            let measurement = run_candidate_measurement(
                &config,
                &candidate_preset,
                &workload,
                &model_path,
                &bench_path,
                &runtime,
                &trial_id,
                BALANCED_BENCH_REPETITIONS,
            )
            .await;
            let Some(measurement) = measurement else {
                let _ = finish_cancelled(&runtime);
                return;
            };
            candidate_results.push(CalibrationCandidateResult {
                candidate: CalibrationCandidate {
                    id: trial_id.clone(),
                    ..candidate
                },
                measurement,
            });
            if let Some(result) = candidate_results.last()
                && let Err(error) = append_trial_result(&runtime.results_path, result)
            {
                let _ = fail_job(
                    &runtime,
                    &format!("Calibration verification result could not be persisted: {error}"),
                );
                return;
            }
            if let Err(error) = append_event(
                &runtime.journal_path,
                &JournalEvent::new(JournalEventKind::TrialFinished, Some(trial_id)),
            ) {
                let _ = fail_job(
                    &runtime,
                    &format!("Calibration verification journal failed: {error}"),
                );
                return;
            }
            let _ = update_snapshot(&runtime, |snapshot| {
                snapshot.completed_trials = snapshot.completed_trials.saturating_add(1);
            });
        }
    }

    let completed_trials = candidate_results.len() as u32;
    let analysis = super::analysis::analyze(&candidate_results);
    let selected_candidate = match budget {
        CalibrationBudget::Balanced => analysis.balanced_candidate.clone(),
        _ => analysis.fastest_candidate.clone(),
    };
    let measurement = candidate_results
        .iter()
        .find(|result| result.candidate.id == "baseline")
        .map(|result| result.measurement.clone())
        .unwrap_or_default();
    let server_qualification = match server_qualification {
        Some(_)
            if *state.local_server_running.lock().unwrap()
                || *state.server_running.lock().unwrap() =>
        {
            let _ = update_snapshot(&runtime, |snapshot| {
                snapshot
                    .diagnostics
                    .push("Server qualification deferred: an active server appeared".into());
                snapshot.diagnostics.truncate(MAX_DIAGNOSTICS);
            });
            None
        }
        Some(request) => match qualify_selected_candidate(
            &config,
            &preset,
            selected_candidate.as_deref(),
            &candidate_results,
            &request,
        )
        .await
        {
            Ok(receipt) => Some(receipt),
            Err(error) => {
                let _ = update_snapshot(&runtime, |snapshot| {
                    snapshot.diagnostics.push(format!(
                        "Server qualification degraded: {}",
                        sanitize_error(&error)
                    ));
                    snapshot.diagnostics.truncate(MAX_DIAGNOSTICS);
                });
                None
            }
        },
        None => None,
    };
    let receipt = CalibrationReceipt {
        schema_version: super::CALIBRATION_SCHEMA_VERSION,
        method_version: if budget == CalibrationBudget::Balanced {
            METHOD_VERSION_BALANCED
        } else {
            METHOD_VERSION_QUICK
        }
        .into(),
        job_id: id.clone(),
        fingerprint: CalibrationFingerprint {
            workload,
            ..fingerprint.clone()
        },
        measurement,
        baseline,
        budget,
        candidate_results,
        analysis,
        selected_candidate,
        preset_id: preset.id.clone(),
        preset_fingerprint: fingerprint.baseline_config_hash.clone(),
        apply_history: Vec::new(),
        server_qualification,
    };
    let receipt_path = config
        .app_paths
        .calibration_receipts_dir()
        .join(format!("{id}.json"));
    if let Err(error) = write_receipt(&receipt_path, &receipt) {
        let _ = fail_job(
            &runtime,
            &format!(
                "Calibration receipt could not be written: {}",
                sanitize_error(&error)
            ),
        );
        return;
    }
    let _ = update_snapshot(&runtime, |snapshot| {
        snapshot.state = CalibrationJobState::Complete;
        snapshot.phase = "complete".into();
        snapshot.completed_trials = completed_trials;
        snapshot.receipt_id = Some(id);
    });
}

const APPLY_CONFIRMATION: &str = "APPLY_CALIBRATION";
const ROLLBACK_CONFIRMATION: &str = "ROLLBACK_CALIBRATION";
const FORGET_CONFIRMATION: &str = "FORGET_CALIBRATION";

pub fn apply_confirmation_phrase() -> &'static str {
    APPLY_CONFIRMATION
}

pub fn rollback_confirmation_phrase() -> &'static str {
    ROLLBACK_CONFIRMATION
}

pub fn forget_confirmation_phrase() -> &'static str {
    FORGET_CONFIRMATION
}

pub fn apply(
    config: &AppConfig,
    state: &AppState,
    job_id: &str,
    request: ApplyCalibrationRequest,
) -> Result<ApplyCalibrationResult> {
    validate_job_id(job_id)?;
    if request.exact_confirmation.as_deref() != Some(APPLY_CONFIRMATION) {
        bail!("Exact confirmation APPLY_CALIBRATION is required");
    }
    let receipt_path = config
        .app_paths
        .calibration_receipts_dir()
        .join(format!("{job_id}.json"));
    let receipt_bytes =
        fs::read(&receipt_path).map_err(|_| anyhow!("Calibration receipt not found"))?;
    let mut receipt: CalibrationReceipt =
        serde_json::from_slice(&receipt_bytes).context("Calibration receipt is invalid")?;
    let candidate_id = request
        .candidate_id
        .clone()
        .or_else(|| receipt.selected_candidate.clone())
        .ok_or_else(|| anyhow!("Calibration receipt has no measured winner"))?;
    let candidate = receipt
        .candidate_results
        .iter()
        .find(|result| result.candidate.id == candidate_id)
        .ok_or_else(|| anyhow!("Calibration candidate not found"))?;
    if candidate.measurement.status != Some(TrialStatus::Ok) {
        bail!("Only a valid measured Calibration candidate may be applied");
    }
    let source_id = if request.target_preset_id.is_empty() {
        receipt.preset_id.clone()
    } else {
        request.target_preset_id.clone()
    };
    let mut presets = state
        .presets
        .lock()
        .map_err(|_| anyhow!("Preset store unavailable"))?
        .clone();
    let original_presets = presets.clone();
    let source_index = presets
        .iter()
        .position(|preset| preset.id == source_id)
        .ok_or_else(|| anyhow!("Target preset not found"))?;
    let before = preset_fingerprint(&presets[source_index])?;
    if !request.expected_target_fingerprint.is_empty()
        && request.expected_target_fingerprint != before
    {
        bail!("Preset changed since Calibration; refresh before applying");
    }
    if source_id != receipt.preset_id && before != receipt.preset_fingerprint {
        bail!("Target preset does not match the Calibration source");
    }

    let mut updated = presets[source_index].clone();
    apply_patch_to_preset(&mut updated, &candidate.candidate.typed_patch);
    updated.backend = InferenceBackend::LlamaCpp;
    crate::inference::launch::validate_preset_backend_config(&updated)?;
    let derived = request.create_derived;
    let target_id = if derived {
        let id = crate::config::generate_random_token()
            .chars()
            .take(24)
            .collect::<String>();
        updated.id = id.clone();
        updated.name = format!("{} (Calibrated)", updated.name);
        id
    } else {
        updated.id.clone()
    };
    let rollback_id = crate::config::generate_random_token()
        .chars()
        .take(32)
        .collect::<String>();
    let rollback_path = config
        .app_paths
        .calibration_apply_backups_dir()
        .join(format!("{rollback_id}.json"));
    write_rollback_backup(&rollback_path, &presets[source_index])?;
    if derived {
        presets.push(updated.clone());
    } else {
        presets[source_index] = updated.clone();
    }
    presets::save_presets(&config.presets_file, &presets).context("save applied preset")?;
    let after = preset_fingerprint(&updated)?;
    let record = CalibrationApplyRecord {
        target_preset_id: target_id.clone(),
        candidate_id: candidate_id.clone(),
        derived,
        before_fingerprint: before.clone(),
        after_fingerprint: after.clone(),
        timestamp_unix_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_millis()),
        validation: "not_run".into(),
        rollback_id,
    };
    receipt.apply_history.push(record);
    if let Err(error) = write_receipt(&receipt_path, &receipt) {
        // Restore the previous preset file/state if the receipt audit record
        // cannot be persisted; an apply without an audit trail is unsafe.
        let _ = presets::save_presets(&config.presets_file, &original_presets);
        return Err(error.context("persist Calibration apply history"));
    }
    *state.presets.lock().unwrap() = presets;
    Ok(ApplyCalibrationResult {
        preset_id: target_id,
        derived,
        candidate_id,
        before_fingerprint: before,
        after_fingerprint: after,
        validation: "not_run".into(),
    })
}

pub fn rollback(
    config: &AppConfig,
    state: &AppState,
    job_id: &str,
    request: RollbackCalibrationRequest,
) -> Result<RollbackCalibrationResult> {
    validate_job_id(job_id)?;
    if request.exact_confirmation.as_deref() != Some(ROLLBACK_CONFIRMATION) {
        bail!("Exact confirmation ROLLBACK_CALIBRATION is required");
    }
    let receipt_path = config
        .app_paths
        .calibration_receipts_dir()
        .join(format!("{job_id}.json"));
    let bytes = fs::read(&receipt_path).map_err(|_| anyhow!("Calibration receipt not found"))?;
    let mut receipt: CalibrationReceipt =
        serde_json::from_slice(&bytes).context("Calibration receipt is invalid")?;
    let record = receipt
        .apply_history
        .last()
        .cloned()
        .ok_or_else(|| anyhow!("Calibration receipt has no applied preset"))?;
    if record.validation == "rolled_back" {
        bail!("Calibration apply has already been rolled back");
    }
    let mut presets = state
        .presets
        .lock()
        .map_err(|_| anyhow!("Preset store unavailable"))?
        .clone();
    let index = presets
        .iter()
        .position(|preset| preset.id == record.target_preset_id)
        .ok_or_else(|| anyhow!("Applied preset no longer exists"))?;
    let current = preset_fingerprint(&presets[index])?;
    if !request.expected_target_fingerprint.is_empty()
        && request.expected_target_fingerprint != current
    {
        bail!("Preset changed since Calibration apply; refresh before rolling back");
    }
    if current != record.after_fingerprint {
        bail!("Applied preset changed since Calibration apply; rollback is unsafe");
    }
    let backup_path = config
        .app_paths
        .calibration_apply_backups_dir()
        .join(format!("{}.json", record.rollback_id));
    let backup = read_rollback_backup(&backup_path)?;
    let before_fingerprint = preset_fingerprint(&backup)?;
    if record.derived {
        presets.remove(index);
    } else {
        presets[index] = backup;
    }
    presets::save_presets(&config.presets_file, &presets).context("save Calibration rollback")?;
    let restored = if record.derived {
        before_fingerprint
    } else {
        preset_fingerprint(&presets[index])?
    };
    receipt.apply_history.push(CalibrationApplyRecord {
        validation: "rolled_back".into(),
        ..record.clone()
    });
    write_receipt(&receipt_path, &receipt)?;
    *state.presets.lock().unwrap() = presets;
    Ok(RollbackCalibrationResult {
        preset_id: record.target_preset_id,
        before_fingerprint: current,
        after_fingerprint: restored,
    })
}

fn write_rollback_backup(path: &Path, preset: &ModelPreset) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("json.tmp");
    let encoded = serde_json::to_vec(preset).context("serialize Calibration rollback backup")?;
    let mut file = fs::File::create(&temporary)?;
    file.write_all(&encoded)?;
    file.sync_all()?;
    drop(file);
    crate::config::harden_file_permissions(&temporary);
    fs::rename(&temporary, path)?;
    crate::config::harden_file_permissions(path);
    Ok(())
}

fn read_rollback_backup(path: &Path) -> Result<ModelPreset> {
    let bytes = fs::read(path).map_err(|_| anyhow!("Calibration rollback backup not found"))?;
    serde_json::from_slice(&bytes).context("Calibration rollback backup is invalid")
}

/// Apply a measured candidate, then run one bounded real llama-bench check. If
/// the check is not valid, restore the exact prior source preset (or remove the
/// derived preset) before returning an error. The active inference session is
/// never touched by this validation.
pub async fn apply_with_validation(
    config: &AppConfig,
    state: &AppState,
    job_id: &str,
    request: ApplyCalibrationRequest,
) -> Result<ApplyCalibrationResult> {
    validate_job_id(job_id)?;
    let source_id = if request.target_preset_id.is_empty() {
        get_receipt(config, job_id)?
            .ok_or_else(|| anyhow!("Calibration receipt not found"))?
            .preset_id
    } else {
        request.target_preset_id.clone()
    };
    let before = find_preset(state, &source_id)?;
    let derived = request.create_derived;
    let mut result = apply(config, state, job_id, request.clone())?;
    if !request.validate_after_apply {
        return Ok(result);
    }

    let applied = find_preset(state, &result.preset_id)?;
    let receipt = get_receipt(config, job_id)?
        .ok_or_else(|| anyhow!("Calibration receipt not found after apply"))?;
    let workload = receipt.fingerprint.workload;
    let model_path = resolve_model_path(config, &applied.model_path)?;
    let bench_path = llama_bench_path(&config.llama_server_path);
    let ngl = applied.gpu_layers.unwrap_or(99);
    let flash_attn = calibration_flash_attn(&applied);
    let (ctk, ctv) = calibration_cache_types(&applied);
    let batch_size = if applied.batch_size == 0 {
        2048
    } else {
        applied.batch_size
    };
    let ubatch_size = if applied.ubatch_size == 0 {
        512
    } else {
        applied.ubatch_size
    };
    let depth = workload.minimum_context.clamp(1, 4096);
    let snapshot = calibration_capability_snapshot(config).await;
    let sweep_result =
        if let Some(error) = calibration_kv_policy_error(snapshot.as_ref(), &ctk, &ctv) {
            Err(error)
        } else {
            run_sweep_with_tokens_repetitions(
                &bench_path,
                &config.llama_server_cwd,
                &model_path.to_string_lossy(),
                ngl,
                flash_attn,
                &ctk,
                &ctv,
                batch_size,
                ubatch_size,
                &[depth],
                applied.n_cpu_moe,
                workload.prompt_tokens.min(512),
                workload.generation_tokens.min(256),
                QUICK_BENCH_REPETITIONS,
            )
            .await
        };
    let measurement = match sweep_result {
        Ok(points) => measurement_from_points(points),
        Err(error) => CalibrationMeasurement {
            trial_id: result.candidate_id.clone(),
            status: Some(TrialStatus::Error),
            bounded_diagnostics: vec![sanitize_error(&error)],
            ..Default::default()
        },
    };
    let validation = if measurement.status == Some(TrialStatus::Ok) {
        "passed"
    } else {
        "failed_rolled_back"
    };
    update_apply_validation(config, job_id, validation)?;
    if validation != "passed" {
        rollback_immediate(config, state, &result.preset_id, derived, &before)?;
        bail!("Post-apply Calibration validation failed; the preset was rolled back")
    }
    result.validation = validation.into();
    Ok(result)
}

fn update_apply_validation(config: &AppConfig, job_id: &str, validation: &str) -> Result<()> {
    let path = config
        .app_paths
        .calibration_receipts_dir()
        .join(format!("{job_id}.json"));
    let bytes = fs::read(&path)?;
    let mut receipt: CalibrationReceipt = serde_json::from_slice(&bytes)?;
    if let Some(record) = receipt.apply_history.last_mut() {
        record.validation = validation.into();
    }
    write_receipt(&path, &receipt)
}

fn rollback_immediate(
    config: &AppConfig,
    state: &AppState,
    applied_id: &str,
    derived: bool,
    before: &ModelPreset,
) -> Result<()> {
    let mut presets = state
        .presets
        .lock()
        .map_err(|_| anyhow!("Preset store unavailable"))?
        .clone();
    if derived {
        presets.retain(|preset| preset.id != applied_id);
    } else if let Some(slot) = presets.iter_mut().find(|preset| preset.id == applied_id) {
        *slot = before.clone();
    } else {
        bail!("Applied preset disappeared during validation")
    }
    presets::save_presets(&config.presets_file, &presets)
        .context("persist Calibration rollback")?;
    *state.presets.lock().unwrap() = presets;
    Ok(())
}

#[allow(dead_code)]
fn select_winner(results: &[CalibrationCandidateResult]) -> Option<String> {
    results
        .iter()
        .filter(|result| result.measurement.status == Some(TrialStatus::Ok))
        .max_by(|left, right| {
            median(&left.measurement.tg_tps_samples)
                .partial_cmp(&median(&right.measurement.tg_tps_samples))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|result| result.candidate.id.clone())
}

/// Select candidates for the independent finalist verification pass.
/// The measured baseline is always included as the control, followed by
/// the strongest successful non-baseline survivors. Ordering is
/// deterministic for equal medians, and the caller's limit bounds the
/// total number of confirmation runs.
#[allow(dead_code)]
fn select_verification_candidates(
    results: &[CalibrationCandidateResult],
    limit: usize,
) -> Vec<CalibrationCandidate> {
    let mut selected = results
        .iter()
        .find(|result| {
            result.candidate.id == "baseline"
                && result.measurement.status == Some(TrialStatus::Ok)
                && !median(&result.measurement.tg_tps_samples).is_nan()
        })
        .map(|result| vec![result.candidate.clone()])
        .unwrap_or_default();
    let mut eligible = results
        .iter()
        .filter(|result| result.candidate.id != "baseline")
        .filter(|result| result.measurement.status == Some(TrialStatus::Ok))
        .filter(|result| !median(&result.measurement.tg_tps_samples).is_nan())
        .collect::<Vec<_>>();
    eligible.sort_by(|left, right| {
        median(&right.measurement.tg_tps_samples)
            .total_cmp(&median(&left.measurement.tg_tps_samples))
            .then_with(|| left.candidate.id.cmp(&right.candidate.id))
    });
    selected.extend(
        eligible
            .into_iter()
            .take(limit.saturating_sub(selected.len()))
            .map(|result| result.candidate.clone()),
    );
    selected
}

async fn qualify_selected_candidate(
    config: &AppConfig,
    preset: &ModelPreset,
    selected_candidate: Option<&str>,
    candidate_results: &[CalibrationCandidateResult],
    request: &QualificationRequest,
) -> Result<QualificationReceipt> {
    let candidate = selected_candidate
        .and_then(|id| {
            candidate_results
                .iter()
                .find(|result| result.candidate.id == id)
        })
        .ok_or_else(|| anyhow!("server qualification has no selected candidate"))?;
    let mut candidate_preset = preset.clone();
    apply_patch_to_preset(&mut candidate_preset, &candidate.candidate.typed_patch);
    let model_path = resolve_model_path(config, &candidate_preset.model_path)?;
    let capabilities = qualification_capabilities(&model_path, &config.llama_server_path).await;
    let mut receipt =
        run_preset_qualification(config, &candidate_preset, request, &capabilities).await?;
    if request
        .tracks
        .iter()
        .any(|track| matches!(track, QualificationTrack::Mtp | QualificationTrack::Ngram))
        && (capabilities.mtp || capabilities.ngram)
    {
        let mut baseline_preset = candidate_preset.clone();
        disable_speculation(&mut baseline_preset);
        let baseline_request = QualificationRequest {
            tracks: std::collections::BTreeSet::from([QualificationTrack::LatencyMemory]),
            parallel_requests: 1,
            allow_concurrency: false,
            ..request.clone()
        };
        let baseline =
            run_preset_qualification(config, &baseline_preset, &baseline_request, &capabilities)
                .await?;
        receipt.baseline = Some(Box::new(baseline));
    }
    Ok(receipt)
}

async fn run_preset_qualification(
    config: &AppConfig,
    preset: &ModelPreset,
    request: &QualificationRequest,
    capabilities: &QualificationCapabilities,
) -> Result<QualificationReceipt> {
    let port = allocate_qualification_port().await?;
    let launch_request = crate::inference::launch::request_from_preset(preset, Some(port))?;
    let crate::inference::launch::LocalLaunchRequest::LlamaCpp(mut server_config) = launch_request
    else {
        bail!("server qualification requires a llama.cpp preset");
    };
    server_config.parallel_slots = request.parallel_requests;
    server_config.benchmark_mode = true;
    // Same launch gate as `construct_adapter`: binary-specific K/V policy must
    // be enforced here too, since calibration builds its own adapter instead
    // of going through the shared launch path.
    let snapshot = ExecutableIdentity::from_path(&config.llama_server_path)
        .ok()
        .and_then(|identity| crate::inference::llama_cpp_capabilities::cached_snapshot(&identity));
    if let Some(snap) = &snapshot
        && let Some(issue) = presets::validation::validate_main_kv_policy(
            &server_config.ctk,
            &server_config.ctv,
            snap,
        )
    {
        bail!(
            "llama.cpp calibration launch blocked: {} ({})",
            issue.code,
            issue.message
        );
    }
    let adapter = crate::inference::llama_cpp::LlamaCppAdapter::new_with_capabilities(
        config.clone(),
        *server_config,
        crate::gpu::env::GpuEnv::default(),
        snapshot,
    );
    let launch = adapter.build_launch().await?;
    server_qualification::run_managed_server_with_capabilities(launch, request, capabilities).await
}

fn disable_speculation(preset: &mut ModelPreset) {
    preset.ngram_spec = false;
    preset.draft_model.clear();
    preset.spec_type = None;
    preset.spec_default = false;
    preset.spec_draft_n_max = None;
    preset.spec_draft_n_min = None;
    preset.spec_draft_p_split = None;
    preset.spec_draft_p_min = None;
    preset.spec_draft_ngl = None;
    preset.spec_draft_device = None;
    preset.spec_draft_cpu_moe = false;
    preset.spec_draft_n_cpu_moe = None;
    preset.spec_draft_type_k = None;
    preset.spec_draft_type_v = None;
    preset.spec_ngram_mod_n_min = None;
    preset.spec_ngram_mod_n_max = None;
    preset.spec_ngram_mod_n_match = None;
    preset.spec_ngram_simple_size_n = None;
    preset.spec_ngram_simple_size_m = None;
    preset.spec_ngram_simple_min_hits = None;
    preset.spec_ngram_map_k_size_n = None;
    preset.spec_ngram_map_k_size_m = None;
    preset.spec_ngram_map_k_min_hits = None;
    preset.spec_ngram_map_k4v_size_n = None;
    preset.spec_ngram_map_k4v_size_m = None;
    preset.spec_ngram_map_k4v_min_hits = None;
}

async fn qualification_capabilities(
    model_path: &Path,
    server_path: &Path,
) -> QualificationCapabilities {
    let Ok(snapshot) =
        crate::inference::llama_cpp_capabilities::generate_snapshot(server_path).await
    else {
        return QualificationCapabilities::default();
    };
    let mtp = crate::llama::gguf_meta::read_gguf_metadata(model_path)
        .ok()
        .and_then(|metadata| metadata.mtp_depth)
        .is_some_and(|depth| depth > 0)
        && matches!(
            snapshot.speculation.draft_model,
            crate::inference::llama_cpp_capabilities::FeatureState::Available
        );
    let ngram = matches!(
        snapshot.speculation.ngram_spec,
        crate::inference::llama_cpp_capabilities::FeatureState::Available
    );
    let mut evidence = std::collections::BTreeSet::new();
    if mtp {
        evidence.insert("mtp".into());
    }
    if ngram {
        evidence.insert("ngram".into());
    }
    QualificationCapabilities {
        mtp,
        dflash: false,
        ngram,
        evidence,
    }
}

async fn allocate_qualification_port() -> Result<u16> {
    let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).await?;
    Ok(listener.local_addr()?.port())
}

#[allow(clippy::too_many_arguments)]
async fn run_candidate_measurement(
    config: &AppConfig,
    candidate_preset: &ModelPreset,
    workload: &super::CalibrationWorkload,
    model_path: &Path,
    bench_path: &Path,
    runtime: &RuntimeJob,
    trial_id: &str,
    repetitions: u32,
) -> Option<CalibrationMeasurement> {
    let ngl = candidate_preset.gpu_layers.unwrap_or(99);
    let flash_attn = calibration_flash_attn(candidate_preset);
    let (ctk, ctv) = calibration_cache_types(candidate_preset);
    let batch_size = if candidate_preset.batch_size == 0 {
        2048
    } else {
        candidate_preset.batch_size
    };
    let ubatch_size = if candidate_preset.ubatch_size == 0 {
        512
    } else {
        candidate_preset.ubatch_size
    };
    let depths = vec![workload.minimum_context.max(1)];
    let model_path_string = model_path.to_string_lossy().into_owned();
    let snapshot = calibration_capability_snapshot(config).await;
    let result = if let Some(error) = calibration_kv_policy_error(snapshot.as_ref(), &ctk, &ctv) {
        Some(Err(error))
    } else {
        let bench = run_sweep_with_tokens_repetitions(
            bench_path,
            &config.llama_server_cwd,
            &model_path_string,
            ngl,
            flash_attn,
            &ctk,
            &ctv,
            batch_size,
            ubatch_size,
            &depths,
            candidate_preset.n_cpu_moe,
            workload.prompt_tokens,
            workload.generation_tokens,
            repetitions,
        );
        tokio::select! {
            result = bench => Some(result),
            _ = runtime.cancel_notify.notified() => None,
        }
    }?;
    if runtime.cancel.load(Ordering::Acquire) {
        return None;
    }
    Some(match result {
        Ok(points) => {
            let mut measurement = measurement_from_points(points);
            measurement.trial_id = trial_id.into();
            measurement
        }
        Err(error) => CalibrationMeasurement {
            trial_id: trial_id.into(),
            status: Some(TrialStatus::Error),
            bounded_diagnostics: vec![sanitize_error(&error)],
            ..Default::default()
        },
    })
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

fn find_preset(state: &AppState, id: &str) -> Result<ModelPreset> {
    state
        .presets
        .lock()
        .map_err(|_| anyhow!("Preset store unavailable"))?
        .iter()
        .find(|preset| preset.id == id)
        .cloned()
        .ok_or_else(|| anyhow!("Preset not found"))
}

const MAX_CALIBRATION_WORKLOAD_TOKENS: u32 = 32_768;
const SCREEN_PROMPT_TOKENS: u32 = 512;
const SCREEN_GENERATION_TOKENS: u32 = 1_024;
const SCREEN_CONTEXT_TOKENS: u64 = 32_768;

fn screening_workload(
    workload: &super::CalibrationWorkload,
    bounded: bool,
) -> super::CalibrationWorkload {
    if !bounded {
        return workload.clone();
    }
    let mut screen = workload.clone();
    screen.prompt_tokens = screen.prompt_tokens.min(SCREEN_PROMPT_TOKENS);
    screen.generation_tokens = screen.generation_tokens.min(SCREEN_GENERATION_TOKENS);
    screen.minimum_context = screen.minimum_context.min(SCREEN_CONTEXT_TOKENS);
    screen
}

fn validate_workload(workload: &super::CalibrationWorkload) -> Result<()> {
    if workload.prompt_tokens == 0 || workload.prompt_tokens > MAX_CALIBRATION_WORKLOAD_TOKENS {
        bail!(
            "Calibration prompt length must be between 1 and {MAX_CALIBRATION_WORKLOAD_TOKENS} tokens"
        );
    }
    if workload.generation_tokens == 0
        || workload.generation_tokens > MAX_CALIBRATION_WORKLOAD_TOKENS
    {
        bail!(
            "Calibration generation length must be between 1 and {MAX_CALIBRATION_WORKLOAD_TOKENS} tokens"
        );
    }
    if workload.parallel_requests != 1 {
        bail!("Quick Calibration supports one request only");
    }
    if workload.minimum_context == 0 || workload.minimum_context > 131_072 {
        bail!("Quick Calibration context must be between 1 and 131072 tokens");
    }
    Ok(())
}

fn resolve_model_path(config: &AppConfig, value: &str) -> Result<PathBuf> {
    let raw = PathBuf::from(value);
    let path = if raw.is_absolute() {
        raw
    } else {
        config
            .models_dir
            .clone()
            .unwrap_or_else(|| config.app_paths.models_dir())
            .join(raw)
    };
    let canonical = require_regular_file(&path).map_err(|error| match error {
        RegularFileError::NotFound | RegularFileError::Canonicalize => {
            anyhow!("Calibration model is unavailable")
        }
        RegularFileError::Symlink | RegularFileError::NotRegular => {
            anyhow!("Calibration requires a regular, non-symlink model file")
        }
    })?;
    let root = config
        .models_dir
        .clone()
        .unwrap_or_else(|| config.app_paths.models_dir())
        .canonicalize()
        .map_err(|_| anyhow!("Calibration model library is unavailable"))?;
    if canonical.strip_prefix(root).is_err() {
        bail!("Calibration model must be inside the configured model library");
    }
    if canonical
        .extension()
        .and_then(|ext| ext.to_str())
        .is_none_or(|ext| !ext.eq_ignore_ascii_case("gguf"))
    {
        bail!("Calibration requires a GGUF model");
    }
    Ok(canonical)
}

fn library_relative_identity(config: &AppConfig, model: &Path) -> String {
    let root = config
        .models_dir
        .clone()
        .unwrap_or_else(|| config.app_paths.models_dir());
    root.canonicalize()
        .ok()
        .and_then(|root| {
            model
                .strip_prefix(root)
                .ok()
                .map(|path| path.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| {
            model
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default()
        })
}

fn build_calibration_fingerprint(
    config: &AppConfig,
    _preset: &ModelPreset,
    workload: &CalibrationWorkload,
    preflight: &CalibrationPreflight,
    model_path: &Path,
    preset_fingerprint: &str,
) -> Result<CalibrationFingerprint> {
    let metadata = fs::metadata(model_path).context("read Calibration model metadata")?;
    let header = crate::llama::gguf_meta::read_gguf_header_inventory(
        model_path,
        crate::llama::gguf_meta::MAX_INSPECTION_HEADER_BYTES,
    )
    .map_err(|error| anyhow!("read GGUF fingerprint header: {error}"))?;
    let content_fingerprint = sha256_file_prefix(model_path, header.header_bytes)?;
    let gguf = crate::llama::gguf_meta::read_gguf_metadata(model_path).ok();
    let metadata_fingerprint = gguf
        .as_ref()
        .and_then(|metadata| serde_json::to_vec(metadata).ok())
        .map(|bytes| sha256_bytes(&bytes))
        .unwrap_or_else(|| content_fingerprint.clone());
    let quantization_signature = sha256_bytes(
        &serde_json::to_vec(&header.quant_types)
            .context("serialize GGUF quantization inventory")?,
    );
    let family_descriptor = serde_json::json!({
        "architecture": gguf.as_ref().and_then(|metadata| metadata.architecture.clone()),
        "param_count": gguf.as_ref().and_then(|metadata| metadata.param_count),
        "block_count": gguf.as_ref().and_then(|metadata| metadata.block_count),
        "head_count": gguf.as_ref().and_then(|metadata| metadata.head_count),
        "head_count_kv": gguf.as_ref().and_then(|metadata| metadata.head_count_kv),
        "key_length": gguf.as_ref().and_then(|metadata| metadata.key_length),
        "embedding_length": gguf.as_ref().and_then(|metadata| metadata.embedding_length),
        "feed_forward_length": gguf.as_ref().and_then(|metadata| metadata.feed_forward_length),
        "expert_count": gguf.as_ref().and_then(|metadata| metadata.expert_count),
        "expert_used_count": gguf.as_ref().and_then(|metadata| metadata.expert_used_count),
        "mtp_depth": gguf.as_ref().and_then(|metadata| metadata.mtp_depth),
        "tensor_param_count": gguf.as_ref().and_then(|metadata| metadata.tensor_param_count),
        "expert_param_count": gguf.as_ref().and_then(|metadata| metadata.expert_param_count),
    });
    let family_key = sha256_bytes(
        &serde_json::to_vec(&family_descriptor).context("serialize GGUF family descriptor")?,
    );
    let compatibility_descriptor = serde_json::json!({
        "family": family_descriptor,
        "quant_types": &header.quant_types,
    });
    let compatibility_key = sha256_bytes(
        &serde_json::to_vec(&compatibility_descriptor)
            .context("serialize GGUF compatibility descriptor")?,
    );
    let mut capability_flags = preflight.bench_help_flags.clone();
    capability_flags.sort();
    let capability_signature = sha256_bytes(
        &serde_json::to_vec(&(
            preflight.server_help_defaults.keys().collect::<Vec<_>>(),
            capability_flags,
        ))
        .context("serialize managed capability signature")?,
    );
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
    let modified_unix_ms = metadata
        .modified()
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_millis())
        .unwrap_or_default();

    Ok(CalibrationFingerprint {
        schema_version: super::CALIBRATION_SCHEMA_VERSION,
        backend: InferenceBackend::LlamaCpp,
        hardware: HardwareFingerprint {
            os: std::env::consts::OS.into(),
            arch: std::env::consts::ARCH.into(),
            cpu_identity: system.cpus().first().map(|cpu| cpu.brand().to_string()),
            physical_cores: sysinfo::System::physical_core_count().map(|count| count as u32),
            logical_cores,
            memory_bytes: system.total_memory(),
            gpu_devices,
            unified_memory: cfg!(target_os = "macos"),
        },
        model: ModelFingerprint {
            library_relative_id: library_relative_identity(config, model_path),
            file_size: metadata.len(),
            modified_unix_ms,
            content_fingerprint,
            gguf_arch: gguf.and_then(|metadata| metadata.architecture),
            metadata_fingerprint,
            compatibility_key,
            family_key,
            quantization_signature,
        },
        runtime: RuntimeFingerprint {
            server_identity: preflight.server_identity.clone(),
            server_sha256: preflight.server_sha256.clone(),
            version: None,
            capability_hash: preflight.bench_help_sha256.clone().unwrap_or_default(),
            bench_sha256: preflight.bench_sha256.clone(),
            fit_params_sha256: preflight.fit_params_sha256.clone(),
            capability_signature,
        },
        workload: workload.clone(),
        baseline_config_hash: preset_fingerprint.into(),
        factor_catalog_version: super::CALIBRATION_FACTOR_CATALOG_VERSION,
    })
}

fn sha256_bytes(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn sha256_file(path: &Path) -> Result<String> {
    sha256_file_prefix(path, u64::MAX)
}

fn sha256_file_prefix(path: &Path, limit: u64) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut remaining = limit;
    let mut buffer = [0u8; 1024 * 1024];
    while remaining > 0 {
        let read_limit = remaining.min(buffer.len() as u64) as usize;
        let count = file.read(&mut buffer[..read_limit])?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
        remaining -= count as u64;
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn measurement_from_points(points: Vec<SweepPoint>) -> CalibrationMeasurement {
    let mut pp = Vec::new();
    let mut tg = Vec::new();
    for point in points {
        let pp_samples = if point.pp_samples.is_empty() {
            vec![point.pp_tps]
        } else {
            point.pp_samples
        };
        pp.extend(
            pp_samples
                .into_iter()
                .filter(|value| value.is_finite() && *value > 0.0),
        );
        let tg_samples = if point.tg_samples.is_empty() {
            vec![point.tg_tps]
        } else {
            point.tg_samples
        };
        tg.extend(
            tg_samples
                .into_iter()
                .filter(|value| value.is_finite() && *value > 0.0),
        );
    }
    let effective = tg.clone();
    CalibrationMeasurement {
        trial_id: "baseline".into(),
        status: Some(if pp.is_empty() && tg.is_empty() {
            TrialStatus::Implausible
        } else {
            TrialStatus::Ok
        }),
        pp_tps_samples: pp,
        tg_tps_samples: tg,
        ttft_ms_samples: Vec::new(),
        effective_tps_samples: effective,
        wall_time_ms: 0,
        memory_peak_bytes: None,
        bounded_diagnostics: Vec::new(),
        launch_evidence: None,
    }
}

fn non_empty_or(value: &str, fallback: &str) -> String {
    if value.trim().is_empty() {
        fallback.into()
    } else {
        value.into()
    }
}

fn calibration_cache_types(preset: &ModelPreset) -> (String, String) {
    // Keep the product quality floor: q8_0 for K, f16 for V. Some hybrid
    // architectures reject a quantized V cache during context creation.
    (
        non_empty_or(&preset.ctk, "q8_0"),
        non_empty_or(&preset.ctv, "f16"),
    )
}

fn calibration_flash_attn(preset: &ModelPreset) -> &'static str {
    match preset.flash_attn.trim().to_ascii_lowercase().as_str() {
        "on" | "1" | "true" => "on",
        "off" | "0" | "false" => "off",
        _ => "auto",
    }
}

/// Refresh and fetch the live capability snapshot for the configured
/// llama-server binary, mirroring the launch gate in
/// `inference::launch::construct_adapter`. Every calibration path that
/// spawns llama-server for a bench sweep must consult this before launch so
/// the K/V policy cannot fork across call sites.
async fn calibration_capability_snapshot(
    config: &AppConfig,
) -> Option<crate::inference::llama_cpp_capabilities::CapabilitySnapshot> {
    if !config.llama_server_path.is_file() {
        return None;
    }
    let _ = crate::inference::llama_cpp_capabilities::generate_snapshot(&config.llama_server_path)
        .await;
    ExecutableIdentity::from_path(&config.llama_server_path)
        .ok()
        .and_then(|identity| crate::inference::llama_cpp_capabilities::cached_snapshot(&identity))
}

/// Build the same "launch blocked" message `construct_adapter` returns when
/// the live snapshot rejects this K/V pair. `None` means the launch may
/// proceed. Returns `String` (not `anyhow::Error`) to match
/// `run_sweep_with_tokens_repetitions`'s error type, so callers can route
/// this through the same handling as a bench-sweep failure and apply-time
/// rollback still triggers.
fn calibration_kv_policy_error(
    snapshot: Option<&crate::inference::llama_cpp_capabilities::CapabilitySnapshot>,
    ctk: &str,
    ctv: &str,
) -> Option<String> {
    let snap = snapshot?;
    let issue = presets::validation::validate_main_kv_policy(ctk, ctv, snap)?;
    Some(format!(
        "llama.cpp calibration launch blocked: {} ({})",
        issue.code, issue.message
    ))
}

fn calibration_baseline(
    preset: &ModelPreset,
    preflight: &CalibrationPreflight,
) -> CalibrationBaseline {
    const HELP_KEYS: [&str; 10] = [
        "ctx_size",
        "threads",
        "threads_batch",
        "batch_size",
        "ubatch_size",
        "cache_type_k",
        "cache_type_v",
        "flash_attn",
        "gpu_layers",
        "n_cpu_moe",
    ];
    let mut effective = BTreeMap::new();
    let mut add = |name: &str, value: String, source: &str| {
        effective.insert(
            name.to_string(),
            CalibrationBaselineValue::new(value, source),
        );
    };
    let help_default = |name: &str, fallback: &str| {
        preflight
            .server_help_defaults
            .get(name)
            .cloned()
            .unwrap_or_else(|| fallback.to_string())
    };

    add(
        "context_size",
        if preset.context_size == 0 {
            help_default("ctx_size", "0 (loaded model)")
        } else {
            preset.context_size.to_string()
        },
        if preset.context_size == 0 {
            "llama_server_help_default"
        } else {
            "preset"
        },
    );
    add(
        "threads",
        preset
            .threads
            .map_or_else(|| help_default("threads", "-1"), |value| value.to_string()),
        if preset.threads.is_some() {
            "preset"
        } else {
            "llama_server_help_default"
        },
    );
    add(
        "threads_batch",
        preset.threads_batch.map_or_else(
            || help_default("threads_batch", "same as threads"),
            |value| value.to_string(),
        ),
        if preset.threads_batch.is_some() {
            "preset"
        } else {
            "llama_server_help_default"
        },
    );
    add(
        "batch_size",
        if preset.batch_size == 0 {
            "2048".into()
        } else {
            preset.batch_size.to_string()
        },
        if preset.batch_size == 0 {
            "calibration_policy"
        } else {
            "preset"
        },
    );
    add(
        "ubatch_size",
        if preset.ubatch_size == 0 {
            "512".into()
        } else {
            preset.ubatch_size.to_string()
        },
        if preset.ubatch_size == 0 {
            "calibration_policy"
        } else {
            "preset"
        },
    );
    let (ctk, ctv) = calibration_cache_types(preset);
    add(
        "cache_type_k",
        ctk,
        if preset.ctk.trim().is_empty() {
            "calibration_policy"
        } else {
            "preset"
        },
    );
    add(
        "cache_type_v",
        ctv,
        if preset.ctv.trim().is_empty() {
            "calibration_policy"
        } else {
            "preset"
        },
    );
    add(
        "flash_attn",
        calibration_flash_attn(preset).into(),
        if preset.flash_attn.trim().is_empty() {
            "calibration_policy"
        } else {
            "preset"
        },
    );
    add(
        "gpu_layers",
        preset.gpu_layers.map_or_else(
            || "all (bench sentinel 99)".into(),
            |value| value.to_string(),
        ),
        if preset.gpu_layers.is_some() {
            "preset"
        } else {
            "calibration_policy"
        },
    );
    add(
        "n_cpu_moe",
        preset
            .n_cpu_moe
            .map_or_else(|| "unset".into(), |value| value.to_string()),
        if preset.n_cpu_moe.is_some() {
            "preset"
        } else {
            "preset_omitted"
        },
    );

    let help_defaults = preflight
        .server_help_defaults
        .iter()
        .filter(|(key, _)| HELP_KEYS.contains(&key.as_str()))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    CalibrationBaseline {
        effective,
        llama_server_help_defaults: help_defaults,
        llama_server_help_sha256: preflight.server_help_sha256.clone(),
        llama_server_help_exit_code: preflight.server_help_exit_code,
        llama_server_help_output_truncated: preflight.server_help_output_truncated,
    }
}

#[cfg(test)]
mod calibration_runtime_defaults_tests {
    use super::*;

    #[test]
    fn preserves_product_q8_key_and_f16_value_defaults() {
        let preset = ModelPreset::default();
        assert_eq!(
            calibration_cache_types(&preset),
            ("q8_0".into(), "f16".into())
        );

        let mut explicit = preset.clone();
        explicit.ctk = "f16".into();
        explicit.ctv = "q8_0".into();
        assert_eq!(
            calibration_cache_types(&explicit),
            ("f16".into(), "q8_0".into())
        );
        assert_eq!(calibration_flash_attn(&preset), "auto");
        explicit.flash_attn = "off".into();
        assert_eq!(calibration_flash_attn(&explicit), "off");
    }
}

fn sanitize_error(error: &dyn std::fmt::Display) -> String {
    error
        .to_string()
        .lines()
        .take(4)
        .map(|line| {
            let line = line.rsplit_once('/').map_or(line, |(_, tail)| tail);
            line.rsplit_once('\\').map_or(line, |(_, tail)| tail)
        })
        .collect::<Vec<_>>()
        .join(" | ")
        .chars()
        .take(512)
        .collect()
}

fn validate_job_id(id: &str) -> Result<()> {
    if is_safe_calibration_id(id) {
        Ok(())
    } else {
        bail!("invalid calibration job identifier")
    }
}

fn write_receipt(path: &Path, receipt: &CalibrationReceipt) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("json.tmp");
    let encoded = serde_json::to_vec_pretty(receipt)?;
    fs::write(&temporary, encoded)?;
    fs::rename(&temporary, path)?;
    crate::config::harden_file_permissions(path);
    Ok(())
}

fn transition(runtime: &RuntimeJob, state: CalibrationJobState, phase: &str) -> Result<()> {
    update_snapshot(runtime, |snapshot| {
        snapshot.state = state;
        snapshot.phase = phase.into();
    })
}

fn fail_job(runtime: &RuntimeJob, message: &str) -> Result<()> {
    update_snapshot(runtime, |snapshot| {
        snapshot.state = CalibrationJobState::Failed;
        snapshot.phase = "failed".into();
        snapshot.diagnostics.push(message.to_string());
        snapshot.diagnostics.truncate(MAX_DIAGNOSTICS);
    })
}

fn finish_cancelled(runtime: &RuntimeJob) -> Result<()> {
    update_snapshot(runtime, |snapshot| {
        snapshot.state = CalibrationJobState::Cancelled;
        snapshot.phase = "cancelled".into();
    })
}

fn update_snapshot<F>(runtime: &RuntimeJob, update: F) -> Result<()>
where
    F: FnOnce(&mut CalibrationJobSnapshot),
{
    let mut snapshot = runtime
        .snapshot
        .lock()
        .map_err(|_| anyhow!("Calibration job unavailable"))?;
    update(&mut snapshot);
    write_snapshot(&runtime.snapshot_path, &snapshot)
}

pub fn apply_patch_to_preset(preset: &mut ModelPreset, patch: &LlamaCppCalibrationPatch) {
    if let Some(value) = patch.gpu_layers {
        preset.gpu_layers = Some(value);
    }
    if let Some(value) = patch.context_size {
        preset.context_size = value;
    }
    if let Some(value) = patch.threads {
        preset.threads = Some(value);
    }
    if let Some(value) = patch.threads_batch {
        preset.threads_batch = Some(value);
    }
    if let Some(value) = patch.ctk.as_ref() {
        preset.ctk = value.clone();
    }
    if let Some(value) = patch.ctv.as_ref() {
        preset.ctv = value.clone();
    }
    if let Some(value) = patch.batch_size {
        preset.batch_size = value;
    }
    if let Some(value) = patch.ubatch_size {
        preset.ubatch_size = value;
    }
    if let Some(value) = patch.flash_attn {
        preset.flash_attn = if value { "on" } else { "off" }.into();
    }
    if let Some(value) = patch.n_cpu_moe {
        preset.n_cpu_moe = Some(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_updates_only_typed_llama_fields() {
        let mut preset = ModelPreset::default();
        apply_patch_to_preset(
            &mut preset,
            &LlamaCppCalibrationPatch {
                context_size: Some(8192),
                flash_attn: Some(true),
                ..Default::default()
            },
        );
        assert_eq!(preset.context_size, 8192);
        assert_eq!(preset.flash_attn, "on");
    }

    #[test]
    fn measurement_rejects_empty_or_non_finite_points() {
        let measurement = measurement_from_points(vec![SweepPoint {
            depth: 1,
            pp_tps: f64::NAN,
            tg_tps: 0.0,
            pp_stddev: 0.0,
            tg_stddev: 0.0,
            repetitions: 1,
            pp_samples: Vec::new(),
            tg_samples: Vec::new(),
        }]);
        assert_eq!(measurement.status, Some(TrialStatus::Implausible));
    }

    #[test]
    fn measurement_preserves_benchmark_repetition_samples() {
        let measurement = measurement_from_points(vec![SweepPoint {
            depth: 0,
            pp_tps: 100.0,
            tg_tps: 20.0,
            pp_stddev: 1.0,
            tg_stddev: 0.5,
            repetitions: 3,
            pp_samples: vec![99.0, 100.0, 101.0],
            tg_samples: vec![19.5, 20.0, 20.5],
        }]);
        assert_eq!(measurement.pp_tps_samples, [99.0, 100.0, 101.0]);
        assert_eq!(measurement.tg_tps_samples, [19.5, 20.0, 20.5]);
    }

    #[test]
    fn verification_selection_is_bounded_and_deterministic() {
        let candidate = |id: &str, tg: f64| CalibrationCandidateResult {
            candidate: CalibrationCandidate {
                id: id.into(),
                typed_patch: LlamaCppCalibrationPatch::default(),
                capability_evidence: Vec::new(),
                predicted_memory_bytes: None,
            },
            measurement: CalibrationMeasurement {
                status: Some(TrialStatus::Ok),
                tg_tps_samples: vec![tg],
                ..Default::default()
            },
        };
        let selected = select_verification_candidates(
            &[
                candidate("baseline", 100.0),
                candidate("b", 120.0),
                candidate("a", 120.0),
                candidate("c", 130.0),
            ],
            3,
        );
        assert_eq!(
            selected
                .iter()
                .map(|candidate| candidate.id.as_str())
                .collect::<Vec<_>>(),
            ["baseline", "c", "a"]
        );
    }

    #[test]
    fn screening_workload_caps_only_balanced_screen_dimensions() {
        let workload = CalibrationWorkload {
            prompt_tokens: 4_096,
            generation_tokens: 8_192,
            minimum_context: 131_072,
            ..Default::default()
        };

        let screen = screening_workload(&workload, true);
        assert_eq!(screen.prompt_tokens, 512);
        assert_eq!(screen.generation_tokens, 1_024);
        assert_eq!(screen.minimum_context, 32_768);
        assert_eq!(screen.parallel_requests, workload.parallel_requests);
        assert_eq!(screen.fixture_id, workload.fixture_id);

        let quick = screening_workload(&workload, false);
        assert_eq!(quick, workload);
    }

    #[test]
    fn rollback_backup_round_trips_a_preset() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("apply-backups").join("one.json");
        let preset = ModelPreset {
            id: "source".into(),
            name: "Source".into(),
            api_key: Some("secret".into()),
            ..Default::default()
        };
        write_rollback_backup(&path, &preset).expect("write backup");
        let restored = read_rollback_backup(&path).expect("read backup");
        assert_eq!(restored.id, preset.id);
        assert_eq!(restored.name, preset.name);
        assert_eq!(restored.api_key, preset.api_key);
    }

    #[test]
    fn public_job_operations_reject_path_traversal() {
        let config = AppConfig::for_test(None, None);
        assert!(get(&config, "../escape").is_err());
        assert!(get_receipt_view(&config, "/absolute").is_err());
        assert!(forget(&config, "job with spaces", FORGET_CONFIRMATION).is_err());
    }

    #[test]
    fn forgetting_requires_exact_confirmation() {
        let config = AppConfig::for_test(None, None);
        let error = forget(&config, "missing-job", "FORGET").expect_err("confirmation required");
        assert!(error.to_string().contains("FORGET_CALIBRATION"));
    }

    #[test]
    fn public_receipt_projection_does_not_include_private_paths() {
        let view = CalibrationReceiptView {
            schema_version: 1,
            method_version: "test".into(),
            job_id: "job".into(),
            fingerprint: CalibrationFingerprint::current(
                InferenceBackend::LlamaCpp,
                CalibrationWorkload::default(),
            ),
            measurement: Default::default(),
            baseline: Default::default(),
            budget: CalibrationBudget::Quick,
            candidate_results: Vec::new(),
            analysis: Default::default(),
            selected_candidate: None,
            preset_id: "preset".into(),
            preset_fingerprint: "fingerprint".into(),
            apply_history: Vec::new(),
            server_qualification: None,
        };
        let encoded = serde_json::to_string(&view).expect("serialize public receipt");
        assert!(!encoded.contains("model_path"));
        assert!(!encoded.contains("bench_path"));
    }

    fn receipt_match_fixture() -> CalibrationFingerprint {
        let mut fingerprint = CalibrationFingerprint::current(
            InferenceBackend::LlamaCpp,
            CalibrationWorkload::default(),
        );
        fingerprint.hardware.os = "test-os".into();
        fingerprint.hardware.arch = "test-arch".into();
        fingerprint.model.content_fingerprint = "artifact-a".into();
        fingerprint.model.family_key = "qwen35-9b-shape".into();
        fingerprint.model.compatibility_key = "qwen35-9b-q8".into();
        fingerprint.model.quantization_signature = "q8".into();
        fingerprint.runtime.server_sha256 = "server-a".into();
        fingerprint.runtime.bench_sha256 = "bench-a".into();
        fingerprint.runtime.capability_signature = "cap-a".into();
        fingerprint.baseline_config_hash = "baseline-a".into();
        fingerprint
    }

    #[test]
    fn receipt_matching_classifies_exact_compatible_related_and_rejects_drift() {
        let expected = receipt_match_fixture();
        let (kind, warnings) = classify_receipt_match(&expected, &expected).expect("exact");
        assert_eq!(kind, ReceiptMatchKind::Exact);
        assert!(warnings.is_empty());

        let mut compatible = expected.clone();
        compatible.model.content_fingerprint = "artifact-b".into();
        compatible.runtime.server_sha256 = "server-b".into();
        compatible.runtime.bench_sha256 = "bench-b".into();
        compatible.baseline_config_hash = "baseline-b".into();
        let (kind, warnings) = classify_receipt_match(&compatible, &expected).expect("compatible");
        assert_eq!(kind, ReceiptMatchKind::Compatible);
        assert!(
            warnings
                .iter()
                .any(|warning| warning.contains("runtime build"))
        );
        assert!(warnings.iter().any(|warning| warning.contains("baseline")));

        let mut related = compatible.clone();
        related.model.quantization_signature = "q6".into();
        related.model.compatibility_key = "qwen35-9b-q6".into();
        let (kind, warnings) = classify_receipt_match(&related, &expected).expect("related");
        assert_eq!(kind, ReceiptMatchKind::Related);
        assert!(
            warnings
                .iter()
                .any(|warning| warning.contains("quantization"))
        );

        let mut wrong_family = expected.clone();
        wrong_family.model.family_key = "other-family".into();
        assert!(classify_receipt_match(&wrong_family, &expected).is_none());

        let mut wrong_hardware = expected.clone();
        wrong_hardware.hardware.arch = "other-arch".into();
        assert!(classify_receipt_match(&wrong_hardware, &expected).is_none());
    }

    #[test]
    fn receipt_matching_accepts_legacy_receipt_without_new_metadata() {
        let expected = receipt_match_fixture();
        let mut legacy = expected.clone();
        legacy.model.compatibility_key.clear();
        legacy.model.family_key.clear();
        legacy.model.quantization_signature.clear();
        legacy.runtime.capability_signature.clear();
        let (kind, warnings) = classify_receipt_match(&legacy, &expected).expect("legacy exact");
        assert_eq!(kind, ReceiptMatchKind::Exact);
        assert!(warnings.iter().any(|warning| warning.contains("Legacy")));
    }

    #[test]
    #[ignore = "requires the local Qwen3.5 Q4/Q6 GGUF fixtures"]
    fn qwen35_local_fixtures_share_introspected_family_but_not_quantization() {
        let q4 = std::env::var_os("LLAMA_MONITOR_QWEN35_Q4").expect("Q4 fixture path");
        let q6 = std::env::var_os("LLAMA_MONITOR_QWEN35_Q6").expect("Q6 fixture path");
        let q4 = std::path::PathBuf::from(q4);
        let q6 = std::path::PathBuf::from(q6);
        let q4_meta = crate::llama::gguf_meta::read_gguf_metadata(&q4).expect("read Q4 GGUF");
        let q6_meta = crate::llama::gguf_meta::read_gguf_metadata(&q6).expect("read Q6 GGUF");
        let q4_json = serde_json::to_value(q4_meta).expect("serialize Q4 metadata");
        let q6_json = serde_json::to_value(q6_meta).expect("serialize Q6 metadata");
        for key in [
            "architecture",
            "param_count",
            "block_count",
            "head_count",
            "head_count_kv",
            "key_length",
            "embedding_length",
            "feed_forward_length",
            "mtp_depth",
        ] {
            assert_eq!(q4_json.get(key), q6_json.get(key), "family field {key}");
        }
        let q4_header = crate::llama::gguf_meta::read_gguf_header_inventory(
            &q4,
            crate::llama::gguf_meta::MAX_INSPECTION_HEADER_BYTES,
        )
        .expect("read Q4 header");
        let q6_header = crate::llama::gguf_meta::read_gguf_header_inventory(
            &q6,
            crate::llama::gguf_meta::MAX_INSPECTION_HEADER_BYTES,
        )
        .expect("read Q6 header");
        assert_ne!(q4_header.quant_types, q6_header.quant_types);
    }

    #[cfg(unix)]
    fn fake_apply_fixture(valid: bool) -> (tempfile::TempDir, AppConfig, AppState, String) {
        use std::os::unix::fs::PermissionsExt;

        let temp = tempfile::tempdir().expect("fixture tempdir");
        let root = temp.path();
        let models = root.join("models");
        let bin = root.join("bin");
        std::fs::create_dir_all(&models).expect("models dir");
        std::fs::create_dir_all(&bin).expect("bin dir");
        let model = models.join("fixture.gguf");
        std::fs::write(&model, b"deterministic fake model").expect("model fixture");
        let server = bin.join("llama-server");
        std::fs::write(&server, b"fake server").expect("server fixture");
        let bench = bin.join("llama-bench");
        // avg_ts is deliberately modest (not the 100000+ tok/s a fake
        // instant-return script could "support"): validate_wall_clock_plausibility
        // compares this rate's implied duration against the real wall-clock
        // time of spawning the fixture process, and under full-suite test
        // parallelism that spawn can take hundreds of ms of scheduling
        // delay. A too-fast fake rate implies a near-zero duration that
        // false-positives as implausible once real spawn latency exceeds
        // it; 1000 tok/s keeps comfortable headroom against that.
        let output = if valid {
            r#"[
                {"n_depth":4096,"n_gen":0,"n_prompt":512,"avg_ts":1000,"stddev":0.1,"n_rep":3,"samples_ts":[999,1000,1001]},
                {"n_depth":4096,"n_gen":256,"n_prompt":0,"avg_ts":1000,"stddev":0.1,"n_rep":3,"samples_ts":[999,1000,1001]}
            ]"#
        } else {
            r#"[{"n_depth":4096,"n_gen":256,"n_prompt":0,"avg_ts":0,"stddev":0,"n_rep":3}]"#
        };
        let escaped_output = output.replace('\'', "'\\''");
        let script = format!("#!/bin/sh\nprintf '%s' '{escaped_output}'\n");
        std::fs::write(&bench, script).expect("bench fixture");
        let mut permissions = std::fs::metadata(&bench)
            .expect("bench metadata")
            .permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&bench, permissions).expect("bench executable");

        let mut config = AppConfig::for_test(None, None);
        config.app_paths = crate::paths::AppPaths::from_root(root.to_path_buf());
        config.config_dir = root.to_path_buf();
        config.models_dir = Some(models);
        config.presets_file = root.join("presets.json");
        config.llama_server_path = server;
        config.llama_server_cwd = root.to_path_buf();
        std::fs::create_dir_all(config.app_paths.calibration_receipts_dir()).expect("receipt dir");
        std::fs::create_dir_all(config.app_paths.calibration_apply_backups_dir())
            .expect("backup dir");

        let preset = ModelPreset {
            id: "source".into(),
            name: "Source".into(),
            model_path: model.to_string_lossy().into_owned(),
            batch_size: 512,
            ubatch_size: 512,
            ..Default::default()
        };
        let fingerprint = preset_fingerprint(&preset).expect("preset fingerprint");
        let candidate = CalibrationCandidate {
            id: "candidate".into(),
            typed_patch: LlamaCppCalibrationPatch {
                batch_size: Some(512),
                ubatch_size: Some(512),
                ..Default::default()
            },
            ..Default::default()
        };
        let receipt = CalibrationReceipt {
            fingerprint: CalibrationFingerprint {
                baseline_config_hash: fingerprint.clone(),
                workload: CalibrationWorkload::default(),
                ..CalibrationFingerprint::current(
                    InferenceBackend::LlamaCpp,
                    CalibrationWorkload::default(),
                )
            },
            job_id: "job-1".into(),
            preset_id: preset.id.clone(),
            preset_fingerprint: fingerprint.clone(),
            selected_candidate: Some(candidate.id.clone()),
            candidate_results: vec![CalibrationCandidateResult {
                candidate,
                measurement: CalibrationMeasurement {
                    status: Some(TrialStatus::Ok),
                    tg_tps_samples: vec![100.0],
                    ..Default::default()
                },
            }],
            ..Default::default()
        };
        let receipt_path = config
            .app_paths
            .calibration_receipts_dir()
            .join("job-1.json");
        write_receipt(&receipt_path, &receipt).expect("write receipt");
        std::fs::write(
            &config.presets_file,
            serde_json::to_vec(&vec![preset]).expect("serialize preset"),
        )
        .expect("write presets");
        let state = AppState::default();
        *state.presets.lock().expect("preset state") =
            serde_json::from_slice(&std::fs::read(&config.presets_file).expect("read presets"))
                .expect("decode presets");
        (temp, config, state, fingerprint)
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn post_apply_fake_runtime_persists_passed_validation() {
        let (_temp, config, state, fingerprint) = fake_apply_fixture(true);
        let result = apply_with_validation(
            &config,
            &state,
            "job-1",
            ApplyCalibrationRequest {
                target_preset_id: "source".into(),
                expected_target_fingerprint: fingerprint,
                candidate_id: Some("candidate".into()),
                create_derived: true,
                exact_confirmation: Some(APPLY_CONFIRMATION.into()),
                validate_after_apply: true,
            },
        )
        .await
        .expect("fake validation succeeds");
        assert_eq!(result.validation, "passed");
        let receipt = get_receipt(&config, "job-1")
            .expect("read receipt")
            .expect("receipt exists");
        assert_eq!(receipt.apply_history.last().unwrap().validation, "passed");
        assert!(
            state
                .presets
                .lock()
                .expect("preset state")
                .iter()
                .any(|preset| preset.name == "Source (Calibrated)")
        );
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn post_apply_fake_runtime_rolls_back_failed_validation() {
        let (_temp, config, state, fingerprint) = fake_apply_fixture(false);
        let error = apply_with_validation(
            &config,
            &state,
            "job-1",
            ApplyCalibrationRequest {
                target_preset_id: "source".into(),
                expected_target_fingerprint: fingerprint,
                candidate_id: Some("candidate".into()),
                create_derived: true,
                exact_confirmation: Some(APPLY_CONFIRMATION.into()),
                validate_after_apply: true,
            },
        )
        .await
        .expect_err("fake validation fails");
        assert!(
            error
                .to_string()
                .contains("Post-apply Calibration validation failed")
        );
        assert!(
            !state
                .presets
                .lock()
                .expect("preset state")
                .iter()
                .any(|preset| preset.name == "Source (Calibrated)")
        );
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn resume_fake_runtime_reuses_finished_results_once() {
        let (_temp, config, state, fingerprint) = fake_apply_fixture(true);
        let job_dir = config.app_paths.calibration_jobs_dir().join("resume-job");
        std::fs::create_dir_all(&job_dir).expect("job dir");
        let snapshot_path = job_dir.join("snapshot.json");
        let journal_path = job_dir.join("journal.jsonl");
        let results_path = job_dir.join("trial-results.jsonl");
        let manifest_path = job_dir.join("manifest.json");
        let preset = state
            .presets
            .lock()
            .expect("preset state")
            .first()
            .cloned()
            .expect("source preset");
        let baseline = CalibrationCandidate {
            id: "baseline".into(),
            ..Default::default()
        };
        let pending = CalibrationCandidate {
            id: "pending".into(),
            typed_patch: LlamaCppCalibrationPatch {
                batch_size: Some(512),
                ubatch_size: Some(512),
                ..Default::default()
            },
            ..Default::default()
        };
        let manifest = CalibrationJobManifest {
            schema_version: super::super::CALIBRATION_SCHEMA_VERSION,
            preset_id: preset.id,
            preset_fingerprint: fingerprint.clone(),
            workload: CalibrationWorkload::default(),
            budget: CalibrationBudget::Quick,
            candidates: vec![baseline.clone(), pending],
            model_path: preset.model_path,
            bench_path: config
                .llama_server_path
                .with_file_name("llama-bench")
                .to_string_lossy()
                .into_owned(),
            fingerprint: CalibrationFingerprint {
                baseline_config_hash: fingerprint,
                ..CalibrationFingerprint::current(
                    InferenceBackend::LlamaCpp,
                    CalibrationWorkload::default(),
                )
            },
            baseline: Default::default(),
            server_qualification: None,
        };
        write_manifest(&manifest_path, &manifest).expect("write manifest");
        write_snapshot(
            &snapshot_path,
            &CalibrationJobSnapshot {
                id: "resume-job".into(),
                state: CalibrationJobState::Failed,
                phase: "suspected_crash".into(),
                completed_trials: 1,
                planned_trials: 2,
                diagnostics: vec!["suspected crash".into()],
                receipt_id: None,
            },
        )
        .expect("write recovered snapshot");
        for (candidate_id, event) in [
            ("baseline", JournalEventKind::TrialPlanned),
            ("baseline", JournalEventKind::TrialStarted),
            ("baseline", JournalEventKind::TrialFinished),
            ("pending", JournalEventKind::TrialPlanned),
            ("pending", JournalEventKind::TrialStarted),
        ] {
            append_event(
                &journal_path,
                &JournalEvent::new(event, Some(candidate_id.into())),
            )
            .expect("append resume event");
        }
        append_trial_result(
            &results_path,
            &CalibrationCandidateResult {
                candidate: baseline,
                measurement: CalibrationMeasurement {
                    trial_id: "baseline".into(),
                    status: Some(TrialStatus::Ok),
                    pp_tps_samples: vec![100.0],
                    tg_tps_samples: vec![100.0],
                    ..Default::default()
                },
            },
        )
        .expect("write finished result");

        resume(
            Arc::new(config.clone()),
            state,
            "resume-job",
            RESUME_CONFIRMATION,
        )
        .expect("resume accepted");
        // Other calibration lifecycle tests may run concurrently and share the
        // executor gate. Allow a bounded but realistic window for the resumed
        // job to acquire the gate and finish its fake runtime trial.
        for _ in 0..1200 {
            if get(&config, "resume-job")
                .expect("get resumed job")
                .is_some_and(|snapshot| snapshot.state == CalibrationJobState::Complete)
            {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        assert_eq!(
            get(&config, "resume-job")
                .expect("get completed job")
                .expect("resumed snapshot")
                .state,
            CalibrationJobState::Complete
        );
        let results = read_trial_results(&results_path).expect("read resumed results");
        assert_eq!(
            results
                .iter()
                .filter(|result| result.candidate.id == "baseline")
                .count(),
            1
        );
        assert!(
            results
                .iter()
                .any(|result| result.candidate.id == "pending")
        );
    }
}
