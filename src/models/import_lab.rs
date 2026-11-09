//! App-facing orchestration for the Experimental GGUF Import Lab.
//!
//! R4 keeps conversion outputs non-launchable and accepts only the single explicit
//! profile proven by R2. Jobs are in-memory, bounded, cancellable, and never expose
//! worker stdout or arbitrary filesystem paths as diagnostics.

use super::gguf_recovery::{
    RecoveryContext, RecoveryRequest, RecoveryResult, RecoveryTier, recover,
};
use anyhow::{Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

const MAX_JOBS: usize = 32;
const MAX_DIAGNOSTICS: usize = 24;
const MAX_DIAGNOSTIC_CHARS: usize = 512;
const DEFAULT_OUTPUT_BOUND: u64 = 512 * 1024 * 1024;
const DEFAULT_DISK_MARGIN: u64 = 512 * 1024 * 1024;
const REFERENCE_DIR: &str = "experimental/import-lab/fixtures/smollm2-135m-v1/authoritative";

static JOBS: LazyLock<Mutex<BTreeMap<String, ImportJob>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));
static JOB_GATE: LazyLock<Arc<tokio::sync::Semaphore>> =
    LazyLock::new(|| Arc::new(tokio::sync::Semaphore::new(1)));

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ImportJobStartRequest {
    pub source_path: PathBuf,
    pub source_tier: Option<RecoveryTier>,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ImportJobState {
    Queued,
    Validating,
    Recovering,
    Publishing,
    Cancelling,
    Cancelled,
    Complete,
    Failed,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImportJobResult {
    pub cache_key: String,
    pub cache_path: PathBuf,
    pub output_path: PathBuf,
    pub output_bytes: u64,
    pub compatibility: &'static str,
    pub launchable: bool,
    pub lineage: ImportLineage,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImportLineage {
    pub kind: &'static str,
    pub source_path: PathBuf,
    pub source_tier: RecoveryTier,
    pub output_dtype: &'static str,
    pub profile_id: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImportResourceEstimate {
    pub source_bytes: u64,
    pub estimated_fp16_bytes: u64,
    pub required_disk_bytes: u64,
    pub available_disk_bytes: Option<u64>,
    pub available_ram_bytes: u64,
    pub disk_sufficient: Option<bool>,
    pub ram_guidance: &'static str,
}

pub fn resource_estimate(models_dir: &Path, source_path: &Path) -> Result<ImportResourceEstimate> {
    let source = resolve_library_gguf(models_dir, source_path)?;
    let source_bytes = std::fs::metadata(&source)?.len();
    let metadata =
        crate::llama::gguf_meta::read_gguf_metadata(&source).map_err(|error| anyhow!(error))?;
    let estimated_fp16_bytes = metadata
        .param_count
        .and_then(|count| count.checked_mul(2))
        .ok_or_else(|| anyhow!("GGUF parameter count is required for a safe FP16 estimate"))?;
    let required_disk_bytes = estimated_fp16_bytes
        .saturating_mul(2)
        .saturating_add(DEFAULT_DISK_MARGIN);
    let mut system = sysinfo::System::new()
        .map_err(|error| anyhow!("failed to initialize system info: {error}"))?;
    system.refresh_memory();
    let available_ram_bytes = system.available_memory();
    let disks = sysinfo::Disks::new_with_refreshed_list()
        .map_err(|error| anyhow!("failed to read disk information: {error}"))?;
    let available_disk_bytes = disks
        .iter()
        .filter(|disk| models_dir.starts_with(disk.mount_point()))
        .max_by_key(|disk| disk.mount_point().components().count())
        .map(sysinfo::Disk::available_space);
    Ok(ImportResourceEstimate {
        source_bytes,
        estimated_fp16_bytes,
        required_disk_bytes,
        available_disk_bytes,
        available_ram_bytes,
        disk_sufficient: available_disk_bytes.map(|bytes| bytes >= required_disk_bytes),
        ram_guidance: if available_ram_bytes >= 2 * 1024 * 1024 * 1024 {
            "comfortable"
        } else {
            "memory_pressure_likely"
        },
    })
}

fn resolve_library_gguf(models_dir: &Path, source_path: &Path) -> Result<PathBuf> {
    validate_library_relative_path(source_path)?;
    let root = models_dir.canonicalize()?;
    let mut lexical = root.clone();
    for component in source_path.components() {
        if matches!(component, std::path::Component::CurDir) {
            continue;
        }
        lexical.push(component);
        let metadata = std::fs::symlink_metadata(&lexical)?;
        if metadata.file_type().is_symlink() {
            bail!("Symlinked GGUF paths are not allowed");
        }
    }
    let source = root.join(source_path).canonicalize()?;
    source
        .strip_prefix(&root)
        .map_err(|_| anyhow!("GGUF must be inside the configured model library"))?;
    if !source.is_file() {
        bail!("Resource estimates require a regular .gguf file");
    }
    Ok(source)
}

pub fn validate_library_relative_path(source_path: &Path) -> Result<()> {
    if source_path.as_os_str().is_empty()
        || source_path.is_absolute()
        || source_path.as_os_str().to_string_lossy().starts_with('/')
        || source_path.as_os_str().to_string_lossy().starts_with('\\')
        || source_path
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        bail!("GGUF path must be library-relative without traversal");
    }
    if source_path
        .extension()
        .and_then(|value| value.to_str())
        .is_none_or(|value| !value.eq_ignore_ascii_case("gguf"))
    {
        bail!("Resource estimates require a regular .gguf file");
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize)]
pub struct ImportJobSnapshot {
    pub id: String,
    pub state: ImportJobState,
    pub phase: String,
    pub progress_percent: u8,
    pub message: String,
    pub created_at_unix_ms: u128,
    pub updated_at_unix_ms: u128,
    pub can_cancel: bool,
    pub diagnostics: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<ImportJobResult>,
}

struct ImportJob {
    snapshot: ImportJobSnapshot,
    cancelled: Arc<AtomicBool>,
}

pub fn local_execution_available() -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
}

pub fn tier_for_source(path: &Path) -> Option<RecoveryTier> {
    match path.file_name()?.to_str()? {
        "SmolLM2-135M-Instruct-F16.gguf" => Some(RecoveryTier::F16),
        "SmolLM2-135M-Instruct-Q8_0.gguf" => Some(RecoveryTier::Q8_0),
        "SmolLM2-135M-Instruct-Q6_K.gguf" => Some(RecoveryTier::Q6K),
        "SmolLM2-135M-Instruct-Q4_K_M.gguf" => Some(RecoveryTier::Q4KM),
        _ => None,
    }
}

pub fn start_job(
    context: RecoveryContext,
    request: ImportJobStartRequest,
) -> Result<ImportJobSnapshot> {
    if !local_execution_available() {
        bail!("Local experimental recovery requires Apple Silicon macOS");
    }
    resolve_library_gguf(&context.models_dir, &request.source_path)?;
    let inferred = tier_for_source(&request.source_path)
        .ok_or_else(|| anyhow!("This GGUF has no supported experimental recovery profile"))?;
    if request
        .source_tier
        .as_ref()
        .is_some_and(|tier| tier != &inferred)
    {
        bail!("Requested source tier does not match the explicit recovery profile");
    }
    let id = crate::config::generate_random_token()
        .chars()
        .take(24)
        .collect::<String>();
    let now = now_ms();
    let snapshot = ImportJobSnapshot {
        id: id.clone(),
        state: ImportJobState::Queued,
        phase: "queued".into(),
        progress_percent: 0,
        message: "Waiting for the bounded recovery worker".into(),
        created_at_unix_ms: now,
        updated_at_unix_ms: now,
        can_cancel: true,
        diagnostics: vec!["Original GGUF will not be modified".into()],
        result: None,
    };
    let cancelled = Arc::new(AtomicBool::new(false));
    {
        let mut jobs = JOBS
            .lock()
            .map_err(|_| anyhow!("Import job registry is unavailable"))?;
        prune_jobs(&mut jobs);
        if jobs.len() >= MAX_JOBS {
            bail!("Too many retained import jobs; clean up completed diagnostics first");
        }
        jobs.insert(
            id.clone(),
            ImportJob {
                snapshot: snapshot.clone(),
                cancelled: Arc::clone(&cancelled),
            },
        );
    }
    let source_path = request.source_path;
    tokio::spawn(async move {
        let permit = match JOB_GATE.clone().acquire_owned().await {
            Ok(permit) => permit,
            Err(_) => {
                fail_job(&id, "Recovery worker gate is unavailable");
                return;
            }
        };
        if cancelled.load(Ordering::Acquire) {
            cancel_job_finished(&id);
            return;
        }
        update_job(
            &id,
            ImportJobState::Validating,
            "validating_source",
            12,
            "Validating exact source, profile, and authoritative assets",
        );
        update_job(
            &id,
            ImportJobState::Recovering,
            "recovering_fp16",
            35,
            "Recovering tensors into an isolated non-launchable staging cache",
        );
        let recovery = RecoveryRequest {
            source_gguf: source_path.clone(),
            reference_dir: PathBuf::from(REFERENCE_DIR),
            source_tier: inferred.clone(),
            max_output_bytes: DEFAULT_OUTPUT_BOUND,
            disk_safety_margin_bytes: DEFAULT_DISK_MARGIN,
        };
        let result = recover(context, recovery, Arc::clone(&cancelled)).await;
        drop(permit);
        match result {
            Ok(result) => complete_job(&id, source_path, inferred, result),
            Err(_error) if cancelled.load(Ordering::Acquire) => cancel_job_finished(&id),
            Err(error) => fail_job(&id, public_failure_diagnostic(&error)),
        }
    });
    Ok(snapshot)
}

pub fn job(id: &str) -> Option<ImportJobSnapshot> {
    JOBS.lock().ok()?.get(id).map(|job| job.snapshot.clone())
}

pub fn jobs() -> Vec<ImportJobSnapshot> {
    JOBS.lock()
        .map(|jobs| jobs.values().map(|job| job.snapshot.clone()).collect())
        .unwrap_or_default()
}

pub fn cancel_job(id: &str) -> Result<ImportJobSnapshot> {
    let mut jobs = JOBS
        .lock()
        .map_err(|_| anyhow!("Import job registry is unavailable"))?;
    let job = jobs
        .get_mut(id)
        .ok_or_else(|| anyhow!("Import job was not found"))?;
    if !job.snapshot.can_cancel {
        bail!("Import job is no longer cancellable");
    }
    job.cancelled.store(true, Ordering::Release);
    job.snapshot.state = ImportJobState::Cancelling;
    job.snapshot.phase = "cancelling".into();
    job.snapshot.message = "Stopping recovery and cleaning staging files".into();
    job.snapshot.updated_at_unix_ms = now_ms();
    Ok(job.snapshot.clone())
}

pub fn forget_job(id: &str) -> Result<()> {
    let mut jobs = JOBS
        .lock()
        .map_err(|_| anyhow!("Import job registry is unavailable"))?;
    let job = jobs
        .get(id)
        .ok_or_else(|| anyhow!("Import job was not found"))?;
    if job.snapshot.can_cancel {
        bail!("Active import jobs cannot be removed");
    }
    jobs.remove(id);
    Ok(())
}

fn complete_job(id: &str, source_path: PathBuf, tier: RecoveryTier, result: RecoveryResult) {
    update_job(
        id,
        ImportJobState::Publishing,
        "publishing_cache",
        90,
        "Validating complete hashes and publishing the experimental cache",
    );
    if let Ok(mut jobs) = JOBS.lock()
        && let Some(job) = jobs.get_mut(id)
    {
        let cache_key = result
            .cache_dir
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
            .to_string();
        job.snapshot.state = ImportJobState::Complete;
        job.snapshot.phase = "complete".into();
        job.snapshot.progress_percent = 100;
        job.snapshot.message =
            "Experimental recovery completed; runtime launch remains disabled".into();
        job.snapshot.updated_at_unix_ms = now_ms();
        job.snapshot.can_cancel = false;
        job.snapshot.diagnostics.push(format!(
            "Validated {} tensors with zero skipped or non-finite tensors",
            result.report.output.tensor_count
        ));
        job.snapshot.result = Some(ImportJobResult {
            cache_key,
            cache_path: result.cache_dir,
            output_path: result.fp16_dir,
            output_bytes: result.report.output.actual_output_bytes,
            compatibility: "experimental",
            launchable: false,
            lineage: ImportLineage {
                kind: "gguf_recovered_fp16",
                source_path,
                source_tier: tier,
                output_dtype: "float16",
                profile_id: "smollm2-135m-instruct-llama-v1",
            },
        });
    }
}

fn update_job(id: &str, state: ImportJobState, phase: &str, progress: u8, message: &str) {
    if let Ok(mut jobs) = JOBS.lock()
        && let Some(job) = jobs.get_mut(id)
    {
        job.snapshot.state = state;
        job.snapshot.phase = phase.into();
        job.snapshot.progress_percent = progress.min(100);
        job.snapshot.message = message.into();
        job.snapshot.updated_at_unix_ms = now_ms();
    }
}

fn fail_job(id: &str, diagnostic: &str) {
    if let Ok(mut jobs) = JOBS.lock()
        && let Some(job) = jobs.get_mut(id)
    {
        job.snapshot.state = ImportJobState::Failed;
        job.snapshot.phase = "failed".into();
        job.snapshot.message =
            "Experimental recovery failed safely; the original GGUF is unchanged".into();
        job.snapshot.updated_at_unix_ms = now_ms();
        job.snapshot.can_cancel = false;
        job.snapshot
            .diagnostics
            .push(bounded_diagnostic(diagnostic));
        job.snapshot.diagnostics.truncate(MAX_DIAGNOSTICS);
    }
}

fn cancel_job_finished(id: &str) {
    if let Ok(mut jobs) = JOBS.lock()
        && let Some(job) = jobs.get_mut(id)
    {
        job.snapshot.state = ImportJobState::Cancelled;
        job.snapshot.phase = "cancelled".into();
        job.snapshot.message = "Recovery cancelled; staging files were removed".into();
        job.snapshot.updated_at_unix_ms = now_ms();
        job.snapshot.can_cancel = false;
        job.snapshot
            .diagnostics
            .push("Cancellation completed without publishing a cache".into());
    }
}

fn prune_jobs(jobs: &mut BTreeMap<String, ImportJob>) {
    while jobs.len() >= MAX_JOBS {
        let removable = jobs
            .iter()
            .find(|(_, job)| !job.snapshot.can_cancel)
            .map(|(id, _)| id.clone());
        let Some(id) = removable else {
            break;
        };
        jobs.remove(&id);
    }
}

fn bounded_diagnostic(value: &str) -> String {
    value
        .replace('\0', "")
        .chars()
        .take(MAX_DIAGNOSTIC_CHARS)
        .collect()
}

fn public_failure_diagnostic(error: &anyhow::Error) -> &'static str {
    let message = error.to_string().to_ascii_lowercase();
    if message.contains("timed out") || message.contains("timeout") {
        "Recovery timed out before a cache could be published"
    } else if message.contains("disk") || message.contains("space") {
        "Recovery stopped because safe disk headroom could not be confirmed"
    } else if message.contains("python")
        || message.contains("environment")
        || message.contains("dependency")
    {
        "The managed recovery runtime failed its integrity check"
    } else if message.contains("source")
        || message.contains("gguf")
        || message.contains("profile")
        || message.contains("reference")
    {
        "The GGUF or authoritative recovery profile failed validation"
    } else if message.contains("bound") || message.contains("output") {
        "Recovery output failed its bounded structural validation"
    } else {
        "Recovery failed safely before publishing a cache"
    }
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |value| value.as_millis())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_explicit_profile_filenames_receive_a_tier() {
        assert_eq!(
            tier_for_source(Path::new("SmolLM2-135M-Instruct-F16.gguf")),
            Some(RecoveryTier::F16)
        );
        assert_eq!(tier_for_source(Path::new("unknown-Q4_K_M.gguf")), None);
    }

    #[test]
    fn diagnostics_are_bounded_and_nul_free() {
        let value = bounded_diagnostic(&format!("{}\0secret", "x".repeat(1024)));
        assert!(value.len() <= MAX_DIAGNOSTIC_CHARS);
        assert!(!value.contains('\0'));
    }

    #[test]
    fn public_failure_diagnostics_do_not_expose_worker_paths() {
        let error =
            anyhow!("worker stderr: traceback at /Users/example/.config/llama-monitor/secret.py");
        let diagnostic = public_failure_diagnostic(&error);
        assert!(!diagnostic.contains('/'));
        assert!(!diagnostic.contains("example"));
        assert_eq!(
            diagnostic,
            "Recovery failed safely before publishing a cache"
        );
    }

    #[test]
    fn library_source_validation_rejects_traversal_and_non_gguf_files() {
        let library = tempfile::tempdir().unwrap();
        std::fs::write(library.path().join("notes.txt"), b"not a model").unwrap();
        assert!(resolve_library_gguf(library.path(), Path::new("../outside.gguf")).is_err());
        assert!(resolve_library_gguf(library.path(), Path::new("notes.txt")).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn library_source_validation_rejects_symlinked_paths() {
        use std::os::unix::fs::symlink;

        let library = tempfile::tempdir().unwrap();
        let outside = tempfile::NamedTempFile::new().unwrap();
        symlink(outside.path(), library.path().join("model.gguf")).unwrap();
        assert!(resolve_library_gguf(library.path(), Path::new("model.gguf")).is_err());
    }

    #[test]
    fn active_job_cancels_before_it_can_be_forgotten() {
        let id = format!("test-job-{}", now_ms());
        let now = now_ms();
        JOBS.lock().unwrap().insert(
            id.clone(),
            ImportJob {
                snapshot: ImportJobSnapshot {
                    id: id.clone(),
                    state: ImportJobState::Queued,
                    phase: "queued".into(),
                    progress_percent: 0,
                    message: "queued".into(),
                    created_at_unix_ms: now,
                    updated_at_unix_ms: now,
                    can_cancel: true,
                    diagnostics: Vec::new(),
                    result: None,
                },
                cancelled: Arc::new(AtomicBool::new(false)),
            },
        );

        assert!(forget_job(&id).is_err());
        let cancelling = cancel_job(&id).unwrap();
        assert_eq!(cancelling.state, ImportJobState::Cancelling);
        assert!(cancelling.can_cancel);
        cancel_job_finished(&id);
        assert_eq!(job(&id).unwrap().state, ImportJobState::Cancelled);
        forget_job(&id).unwrap();
        assert!(job(&id).is_none());
    }
}
