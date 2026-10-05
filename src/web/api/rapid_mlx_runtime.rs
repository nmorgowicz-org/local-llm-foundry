use anyhow::Context;
use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use rand::TryRng;
use rand::rngs::SysRng;
use serde::{Deserialize, Serialize};
use warp::Filter;
use warp::http::StatusCode;

use super::common::{check_api_token, check_db_admin_token};
use super::{ApiCtx, ApiReply, ApiRoute, unauthorized_api_token, unauthorized_db_admin_token};
use crate::inference::backend::{
    BackendRecommendationInput, RecommendationArtifactKind, recommend_backend,
};
use crate::inference::rapid_mlx::capabilities::{self, CapabilitySnapshot, ExecutableIdentity};
use crate::inference::rapid_mlx::changelog;
use crate::inference::rapid_mlx::compatibility;
use crate::inference::rapid_mlx::discovery::Discovery;
use crate::inference::rapid_mlx::info_query;
use crate::inference::rapid_mlx::model_resolver::{
    AuthoritativeSafetensorsSource, RapidMlxModelSource,
};
use crate::inference::rapid_mlx::updater::{
    ManagedGitSourceSelection, ManagedReleaseChannel, ManagedReleaseSelection, ManagedRuntimeExtra,
    ManagedRuntimeSourceKind, ManagedRuntimeStatus, RapidMlxRuntimeManager, RuntimeInventoryEntry,
    RuntimeMutationResult, default_runtime_extras,
};
use crate::inference::rapid_mlx::{RapidMlxConfig, check_mutual_exclusions};
use crate::state::{DoctorFinding, DoctorFindingType, DoctorSeverity, FixAction};

const RELEASES_URL: &str =
    "https://api.github.com/repos/raullenchai/Rapid-MLX/releases?per_page=30";
const RELEASE_BY_TAG_URL: &str = "https://api.github.com/repos/raullenchai/Rapid-MLX/releases/tags";
const GITHUB_API_URL: &str = "https://api.github.com/repos";
const MAX_RELEASE_RESPONSE_BYTES: usize = 512 * 1024;
const RELEASE_CACHE_TTL: Duration = Duration::from_secs(300);
const MAX_RETAINED_JOBS: usize = 16;

type ReleaseCache = Option<(Instant, Vec<PublishedRelease>)>;

#[derive(Clone)]
struct RuntimeApiState {
    manager: Result<Arc<RapidMlxRuntimeManager>, String>,
    releases: Arc<tokio::sync::Mutex<ReleaseCache>>,
    jobs: Arc<Mutex<RuntimeJobs>>,
    model_downloads: Arc<Mutex<BTreeMap<String, ModelDownloadJob>>>,
    changelog_cache: Arc<changelog::ChangelogCacheManager>,
    client: reqwest::Client,
}

/// Tracks a pre-download of a Hugging Face model repository into the app-scoped
/// model cache, so the Spawn Wizard (and spawn itself) find the weights already
/// local instead of silently downloading at launch time. Engine is carried for
/// provenance (rapid-mlx today, omlx later) — both read the same HF cache.
#[derive(Debug, Clone, Serialize)]
struct ModelDownloadJob {
    repo_id: String,
    revision: String,
    engine: String,
    state: RuntimeJobState,
    message: String,
    error: Option<String>,
    local_path: Option<String>,
    // Byte-level progress: total from the repo listing, done accumulated from
    // per-file completions plus intra-file tqdm percentages.
    bytes_total: u64,
    bytes_done: u64,
    current_file: String,
    stalled: bool,
    restarts: u32,
    /// Set by the cancel route; the worker polls it and kills the downloader.
    #[serde(skip)]
    cancel: Arc<std::sync::atomic::AtomicBool>,
}

const MODEL_DOWNLOAD_MAX_RESTARTS: u32 = 3;
/// No forwarded progress for this long counts as a stall; the downloader is
/// restarted and resumes (the hub skips files it already finished).
const MODEL_DOWNLOAD_STALL_TIMEOUT: Duration = Duration::from_secs(8 * 60);

fn validate_model_download_repo(repo_id: &str) -> bool {
    let parts: Vec<&str> = repo_id.splitn(3, '/').collect();
    parts.len() == 2
        && !parts
            .iter()
            .any(|p| p.is_empty() || p.contains("..") || p.contains('/'))
}

#[derive(Default)]
struct RuntimeJobs {
    entries: BTreeMap<String, RuntimeJobSnapshot>,
    order: VecDeque<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum RuntimeOperation {
    Install,
    InstallDevelopment,
    Upgrade,
    Repair,
    Rollback,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum RuntimeJobState {
    Queued,
    Running,
    Complete,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize)]
struct RuntimeJobSnapshot {
    id: String,
    operation: RuntimeOperation,
    state: RuntimeJobState,
    message: String,
    version: Option<String>,
    result: Option<PublicRuntimeMutationResult>,
}

#[derive(Debug, Clone)]
enum RuntimeJobSpec {
    Release(ManagedReleaseSelection),
    Git(ManagedGitSourceSelection),
}

impl Default for RuntimeJobSnapshot {
    fn default() -> Self {
        Self {
            id: String::new(),
            operation: RuntimeOperation::Install,
            state: RuntimeJobState::Queued,
            message: String::new(),
            version: None,
            result: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct PublicRuntimeInventoryEntry {
    environment_id: String,
    version: String,
    source_kind: ManagedRuntimeSourceKind,
    source_repository: Option<String>,
    source_ref: Option<String>,
    source_commit: Option<String>,
    release_channel: ManagedReleaseChannel,
    extras: Vec<ManagedRuntimeExtra>,
    active: bool,
    rollback_candidate: bool,
    complete: bool,
}

impl From<RuntimeInventoryEntry> for PublicRuntimeInventoryEntry {
    fn from(entry: RuntimeInventoryEntry) -> Self {
        Self {
            environment_id: entry.environment_id,
            version: entry.version,
            source_kind: entry.source_kind,
            source_repository: entry.source_repository,
            source_ref: entry.source_ref,
            source_commit: entry.source_commit,
            release_channel: entry.release_channel,
            extras: entry.extras,
            active: entry.active,
            rollback_candidate: entry.rollback_candidate,
            complete: entry.complete,
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct PublicRuntimeMutationResult {
    active: PublicRuntimeInventoryEntry,
    previous_environment_id: Option<String>,
}

impl From<RuntimeMutationResult> for PublicRuntimeMutationResult {
    fn from(result: RuntimeMutationResult) -> Self {
        Self {
            active: result.active.into(),
            previous_environment_id: result.previous_environment_id,
        }
    }
}

#[derive(Debug, Serialize)]
struct PublicManagedRuntimeStatus {
    supported: bool,
    installer_available: bool,
    mutation_in_progress: bool,
    rollback_available: bool,
    active: Option<PublicRuntimeInventoryEntry>,
    inventory: Vec<PublicRuntimeInventoryEntry>,
}

impl From<ManagedRuntimeStatus> for PublicManagedRuntimeStatus {
    fn from(status: ManagedRuntimeStatus) -> Self {
        Self {
            supported: status.supported,
            installer_available: status.installer_available,
            mutation_in_progress: status.mutation_in_progress,
            rollback_available: status.rollback_available,
            active: status.active.map(Into::into),
            inventory: status.inventory.into_iter().map(Into::into).collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
struct PublishedRelease {
    version: String,
    tag: String,
    channel: ManagedReleaseChannel,
    published_at: String,
    release_notes: Option<String>,
}

impl Default for PublishedRelease {
    fn default() -> Self {
        Self {
            version: String::new(),
            tag: String::new(),
            channel: ManagedReleaseChannel::Stable,
            published_at: String::new(),
            release_notes: None,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RuntimeMutationRequest {
    version: String,
    channel: ManagedReleaseChannel,
    #[serde(default = "default_runtime_extras")]
    extras: Vec<ManagedRuntimeExtra>,
    confirm: String,
}

#[derive(Debug, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct DevelopmentSourceRequest {
    repository: String,
    reference: String,
    #[serde(default)]
    resolved_commit: Option<String>,
    #[serde(default = "default_runtime_extras")]
    extras: Vec<ManagedRuntimeExtra>,
    confirm: String,
}

#[derive(Debug, Serialize)]
struct DevelopmentSourceResolution {
    repository: String,
    requested_ref: String,
    resolved_commit: String,
    commit_url: String,
    title: String,
    base_commit: Option<String>,
    extras: Vec<ManagedRuntimeExtra>,
}

impl Default for RuntimeMutationRequest {
    fn default() -> Self {
        Self {
            version: String::new(),
            channel: ManagedReleaseChannel::Stable,
            extras: default_runtime_extras(),
            confirm: String::new(),
        }
    }
}

#[derive(Debug, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct RuntimeConfirmationRequest {
    confirm: String,
}

#[derive(Debug, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct RecommendationRequest {
    artifact_kind: RecommendationArtifactKind,
}

#[derive(Debug, Deserialize)]
struct GithubRelease {
    #[serde(default)]
    tag_name: String,
    #[serde(default)]
    draft: bool,
    #[serde(default)]
    prerelease: bool,
    #[serde(default)]
    published_at: String,
    #[serde(default)]
    body: String,
}

pub(crate) fn routes(ctx: ApiCtx) -> ApiRoute {
    let manager = RapidMlxRuntimeManager::new(&ctx.config.config_dir)
        .map(Arc::new)
        .map_err(|_| "Managed Rapid-MLX storage is unavailable".to_string());
    let client = reqwest::Client::builder()
        .user_agent("llama-monitor/rapid-mlx-runtime-manager")
        .timeout(Duration::from_secs(20))
        .build()
        .expect("static Rapid-MLX release client configuration must be valid");
    let state = RuntimeApiState {
        manager,
        releases: Arc::new(tokio::sync::Mutex::new(None)),
        jobs: Arc::new(Mutex::new(RuntimeJobs::default())),
        model_downloads: Arc::new(Mutex::new(BTreeMap::new())),
        changelog_cache: Arc::new(changelog::ChangelogCacheManager::new()),
        client,
    };

    status_route(ctx.clone(), state.clone())
        .or(catalog_route(ctx.clone(), state.clone()))
        .unify()
        .or(releases_route(ctx.clone(), state.clone()))
        .unify()
        .or(changelog_route(ctx.clone(), state.clone()))
        .unify()
        .or(recommendation_route(ctx.clone(), state.clone()))
        .unify()
        .or(doctor_route(ctx.clone(), state.clone()))
        .unify()
        .or(flag_advisor_route(ctx.clone()))
        .unify()
        .or(mtp_draft_suggestion_route(ctx.clone()))
        .unify()
        .or(mutation_route(
            ctx.clone(),
            state.clone(),
            RuntimeOperation::Install,
        ))
        .unify()
        .or(development_source_route(ctx.clone(), state.clone()))
        .unify()
        .or(mutation_route(
            ctx.clone(),
            state.clone(),
            RuntimeOperation::Upgrade,
        ))
        .unify()
        .or(simple_mutation_route(
            ctx.clone(),
            state.clone(),
            RuntimeOperation::Repair,
        ))
        .unify()
        .or(simple_mutation_route(
            ctx.clone(),
            state.clone(),
            RuntimeOperation::Rollback,
        ))
        .unify()
        .or(job_route(ctx.clone(), state.clone()))
        .unify()
        .or(model_download_route(ctx.clone(), state.clone()))
        .unify()
        .or(model_download_status_route(ctx.clone(), state.clone()))
        .unify()
        .or(model_download_cancel_route(ctx.clone(), state.clone()))
        .unify()
        .or(profile_route(ctx.clone(), state.clone()))
        .unify()
        .or(unified_profile_route(ctx.clone()))
        .unify()
        .or(escape_hatch_route(ctx.clone()))
        .unify()
        // Registered before the catalog route so the longer path wins the match.
        .or(settings_validate_route(ctx.clone()))
        .unify()
        .or(settings_catalog_route(ctx.clone()))
        .unify()
        .or(command_preview_route(ctx.clone()))
        .unify()
        .or(prefix_cache_guidance_route(ctx.clone()))
        .unify()
        .or(runtime_metadata_route(ctx, state))
        .unify()
        .boxed()
}

/// Shared driver for both model-download routes. Runs `huggingface_hub`
/// per-file downloads in a Python child, streams progress lines from its
/// stdout (file boundaries, cumulative bytes, final snapshot path) plus
/// intra-file tqdm percentages from stderr, detects stalls with a watchdog
/// and restarts to resume — the hub skips files it already finished.
fn validate_model_download_engine(engine: &str) -> bool {
    matches!(engine, "rapid-mlx" | "omlx")
}

fn spawn_model_download_worker(
    state: RuntimeApiState,
    job_id: String,
    repo_id: String,
    revision: String,
    models_dir: std::path::PathBuf,
) {
    tokio::spawn(async move {
        use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};

        // Raw string on purpose: a `\` line continuation in a normal literal strips the
        // next line's indentation, which breaks the Python `for` body.
        let python_code = r#"from huggingface_hub import HfApi, hf_hub_download
import os, sys
repo_id, revision, cache = sys.argv[1], sys.argv[2], sys.argv[3]
info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
files = [(f.rfilename, f.size or 0) for f in info.siblings if (f.size or 0) > 0]
print(f'TOTAL {sum(s for _, s in files)}', flush=True)
last = ''
for name, size in files:
    print(f'FILE {name}\t{size}', flush=True)
    last = hf_hub_download(repo_id=repo_id, filename=name, revision=revision, cache_dir=cache)
print('PATH ' + os.path.dirname(last), flush=True)"#;

        // Progress is measured on disk: the hub writes `<blob>.incomplete` files under
        // models--owner--repo/blobs while downloading, so the directory size is the
        // true byte count (xet and plain HTTP alike), including files kept from an
        // earlier attempt when resuming.
        let blobs_dir = models_dir
            .join("cache/huggingface/hub")
            .join(format!("models--{}", repo_id.replace('/', "--")))
            .join("blobs");
        let dir_bytes = |dir: &std::path::Path| -> u64 {
            std::fs::read_dir(dir)
                .map(|entries| {
                    entries
                        .flatten()
                        .filter_map(|e| e.metadata().ok())
                        .filter(|m| m.is_file())
                        .map(|m| m.len())
                        .sum()
                })
                .unwrap_or(0)
        };

        let python = if cfg!(windows) {
            "python.exe"
        } else {
            "python3"
        };
        let cancel = state
            .model_downloads
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&job_id)
            .map(|job| job.cancel.clone())
            .unwrap_or_default();

        // Preflight: say exactly what is missing instead of a bare exit status.
        let preflight = tokio::process::Command::new(python)
            .args(["-c", "import huggingface_hub"])
            .stdin(std::process::Stdio::null())
            .output()
            .await;
        let preflight_error = match preflight {
            Err(_) => Some(format!(
                "Python 3 was not found ({python}). Model downloads use the Hugging Face hub client; install Python 3 and run: pip install huggingface_hub"
            )),
            Ok(out) if !out.status.success() => Some(
                "The Python package huggingface_hub is not installed. Run: pip install huggingface_hub"
                    .to_string(),
            ),
            Ok(_) => None,
        };
        if let Some(message) = preflight_error {
            let mut downloads = state
                .model_downloads
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if let Some(job) = downloads.get_mut(&job_id) {
                job.state = RuntimeJobState::Failed;
                job.message = "Download unavailable".into();
                job.error = Some(message);
            }
            return;
        }

        let mut attempt = 0u32;
        loop {
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                let mut downloads = state
                    .model_downloads
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                if let Some(job) = downloads.get_mut(&job_id) {
                    job.state = RuntimeJobState::Cancelled;
                    job.message = "Cancelled".into();
                }
                return;
            }
            let mut child = match tokio::process::Command::new(if cfg!(windows) {
                "python.exe"
            } else {
                "python3"
            })
            .args(["-c", python_code, &repo_id, &revision])
            .arg(models_dir.join("cache/huggingface/hub"))
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .stdin(std::process::Stdio::null())
            .kill_on_drop(true)
            .spawn()
            {
                Ok(c) => c,
                Err(error) => {
                    let mut downloads = state
                        .model_downloads
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    {
                        if let Some(job) = downloads.get_mut(&job_id) {
                            job.state = RuntimeJobState::Failed;
                            job.message = "Download failed".into();
                            job.error =
                                Some(format!("Could not start the hub downloader: {error}"));
                        }
                    }
                    return;
                }
            };
            attempt += 1;
            let mut stdout = BufReader::new(child.stdout.take().expect("stdout piped")).lines();
            let mut stderr = child.stderr.take().expect("stderr piped");
            let mut last_progress = tokio::time::Instant::now();
            let mut last_bytes: u64 = 0;
            let mut total: u64 = 0;
            let mut final_path: Option<String> = None;
            let mut failure: Option<String> = None;
            let mut stalled = false;
            let mut cancelled = false;
            let mut err_tail = String::new();
            let mut err_buf = [0u8; 4096];
            let mut stderr_open = true;
            let mut tick = tokio::time::interval(Duration::from_secs(1));
            loop {
                tokio::select! {
                    line = stdout.next_line() => {
                        match line {
                            Ok(Some(line)) => {
                                last_progress = tokio::time::Instant::now();
                                if let Some(rest) = line.strip_prefix("TOTAL ") {
                                    total = rest.trim().parse().unwrap_or(0);
                                } else if let Some(rest) = line.strip_prefix("FILE ") {
                                    let name = rest.rsplit_once('\t').map(|(n, _)| n).unwrap_or(rest);
                                    let mut downloads = state.model_downloads.lock().unwrap_or_else(|e| e.into_inner());
                                    if let Some(job) = downloads.get_mut(&job_id) {
                                        job.current_file = name.to_string();
                                    }
                                } else if let Some(rest) = line.strip_prefix("PATH ") {
                                    final_path = Some(rest.trim().to_string());
                                }
                            }
                            Ok(None) => break,
                            Err(error) => {
                                failure = Some(format!("Downloader stream error: {error}"));
                                break;
                            }
                        }
                    }
                    read = stderr.read(&mut err_buf), if stderr_open => {
                        match read {
                            Ok(0) | Err(_) => stderr_open = false,
                            Ok(n) => {
                                err_tail.push_str(&String::from_utf8_lossy(&err_buf[..n]));
                                if err_tail.len() > 2000 {
                                    let cut = err_tail.len() - 2000;
                                    err_tail = err_tail.split_at(err_tail.ceil_char_boundary(cut)).1.to_string();
                                }
                            }
                        }
                    }
                    _ = tick.tick() => {
                        if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                            cancelled = true;
                            let _ = child.start_kill();
                            break;
                        }
                        let bytes = dir_bytes(&blobs_dir);
                        if bytes != last_bytes {
                            last_bytes = bytes;
                            last_progress = tokio::time::Instant::now();
                        }
                        {
                            let mut downloads = state.model_downloads.lock().unwrap_or_else(|e| e.into_inner());
                            if let Some(job) = downloads.get_mut(&job_id) {
                                job.bytes_total = total.max(job.bytes_total);
                                job.bytes_done = if job.bytes_total > 0 { bytes.min(job.bytes_total) } else { bytes };
                                if last_progress.elapsed() < Duration::from_secs(30) {
                                    job.stalled = false;
                                }
                            }
                        }
                        if last_progress.elapsed() > MODEL_DOWNLOAD_STALL_TIMEOUT {
                            stalled = true;
                            break;
                        }
                    }
                }
            }

            let status = child.wait().await;
            if cancelled {
                let mut downloads = state
                    .model_downloads
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                if let Some(job) = downloads.get_mut(&job_id) {
                    job.state = RuntimeJobState::Cancelled;
                    job.message = "Cancelled".into();
                    job.stalled = false;
                }
                return;
            }
            if !stalled && failure.is_none() && final_path.is_none() {
                // Non-stall exit without a PATH line: pull the error from stderr text.
                let code = status.ok().and_then(|s| s.code()).unwrap_or(-1);
                let detail = err_tail
                    .lines()
                    .rev()
                    .find(|l| {
                        let l = l.to_ascii_lowercase();
                        l.contains("error") || l.contains("exception")
                    })
                    .unwrap_or("")
                    .trim()
                    .to_string();
                failure = Some(if detail.is_empty() {
                    format!("Hub downloader exited with status {code}")
                } else {
                    format!("Hub downloader exited with status {code}: {detail}")
                });
            }

            let mut restart = false;
            let mut downloads = state
                .model_downloads
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            {
                if let Some(job) = downloads.get_mut(&job_id) {
                    if let Some(path) = final_path.filter(|_| failure.is_none()) {
                        job.state = RuntimeJobState::Complete;
                        job.message = "Downloaded".into();
                        job.error = None;
                        job.local_path = Some(path);
                        job.stalled = false;
                        job.bytes_done = job.bytes_total;
                    } else if stalled {
                        job.stalled = true;
                        job.restarts = attempt;
                        if attempt <= MODEL_DOWNLOAD_MAX_RESTARTS {
                            job.message = "Stalled \u{2014} resuming download".into();
                            restart = true;
                        } else {
                            job.state = RuntimeJobState::Failed;
                            job.message = "Download stalled".into();
                            job.error = Some("No progress for 8 minutes across 4 attempts. Retry when your connection is stable — finished files are kept and the download resumes.".into());
                        }
                    } else {
                        job.state = RuntimeJobState::Failed;
                        job.message = "Download failed".into();
                        job.error = failure.or_else(|| Some("Download failed".into()));
                    }
                }
            }
            if !restart {
                return;
            }
        }
    });
}

/// POST /api/models/downloads — start a background snapshot download of a
/// Hugging Face model repository into the app-scoped model cache. Engine is
/// provenance ("rapid-mlx" today, "omlx" later); both read the same HF cache.
fn model_download_route(ctx: ApiCtx, state: RuntimeApiState) -> ApiRoute {
    warp::path!("api" / "models" / "downloads")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<serde_json::Value>())
        .and_then(move |auth: Option<String>, body: serde_json::Value| {
            let ctx = ctx.clone();
            let state = state.clone();
            async move {
                if !check_api_token(&auth, &ctx.config) {
                    return Ok::<ApiReply, warp::Rejection>(unauthorized_api_token());
                }
                let repo_id = body["repo_id"].as_str().unwrap_or("").trim().to_string();
                let revision = body["revision"]
                    .as_str()
                    .map(|r| r.trim().to_string())
                    .filter(|r| !r.is_empty())
                    .unwrap_or_else(|| "main".to_string());
                let engine = body["engine"]
                    .as_str()
                    .map(|e| e.trim().to_string())
                    .filter(|e| !e.is_empty())
                    .unwrap_or_else(|| "rapid-mlx".to_string());
                if !validate_model_download_repo(&repo_id) {
                    return Ok(Box::new(warp::reply::with_status(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": "Invalid repo_id format. Expected: owner/repo"
                        })),
                        StatusCode::BAD_REQUEST,
                    )) as ApiReply);
                }
                if !validate_model_download_engine(&engine) {
                    return Ok(Box::new(warp::reply::with_status(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": "Unknown engine. Expected 'rapid-mlx' or 'omlx'"
                        })),
                        StatusCode::BAD_REQUEST,
                    )) as ApiReply);
                }

                // One download at a time per repo@revision; report the in-flight job
                // instead of stacking duplicate downloads. The guard is scoped so it
                // is dropped before the insert lock below (std Mutex is not
                // reentrant — holding it across the second lock self-deadlocks).
                {
                    let downloads = state
                        .model_downloads
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    if let Some(_existing) = downloads.values().find(|job| {
                        job.repo_id == repo_id
                            && job.revision == revision
                            && matches!(
                                job.state,
                                RuntimeJobState::Queued | RuntimeJobState::Running
                            )
                    }) {
                        return Ok(Box::new(warp::reply::json(&serde_json::json!({
                            "ok": true,
                            "already_running": true,
                            "repo_id": repo_id,
                        }))) as ApiReply);
                    }
                }

                let models_dir = super::models::get_effective_models_dir(&ctx.state)
                    .unwrap_or_else(|| ctx.config.default_models_dir.clone());
                static MODEL_DL_SEQ: std::sync::atomic::AtomicU64 =
                    std::sync::atomic::AtomicU64::new(0);
                let job_id = {
                    let nanos = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_nanos() as u64)
                        .unwrap_or(0);
                    // Nanoseconds plus a process-wide counter: unique without a CSPRNG,
                    // which is overkill for an opaque job id.
                    let seq = MODEL_DL_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    format!("mdl-{}-{}", nanos, seq)
                };
                let job = ModelDownloadJob {
                    repo_id: repo_id.clone(),
                    revision: revision.clone(),
                    engine: engine.clone(),
                    state: RuntimeJobState::Running,
                    message: "Downloading from Hugging Face".into(),
                    error: None,
                    local_path: None,
                    bytes_total: 0,
                    bytes_done: 0,
                    current_file: String::new(),
                    stalled: false,
                    restarts: 0,
                    cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)),
                };
                let mut downloads = state
                    .model_downloads
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                {
                    downloads.insert(job_id.clone(), job);
                    while downloads.len() > MAX_RETAINED_JOBS {
                        if let Some(oldest) = downloads.keys().next().cloned() {
                            downloads.remove(&oldest);
                        }
                    }
                }

                spawn_model_download_worker(
                    state.clone(),
                    job_id.clone(),
                    repo_id.clone(),
                    revision.clone(),
                    models_dir,
                );

                Ok(Box::new(warp::reply::json(&serde_json::json!({
                    "ok": true,
                    "job_id": job_id,
                    "repo_id": repo_id,
                    "revision": revision,
                    "engine": engine,
                }))) as ApiReply)
            }
        })
        .boxed()
}

/// POST /api/models/downloads/:jobId/cancel — stop a running download. Finished
/// files stay in the cache, so a later download of the same repo resumes.
fn model_download_cancel_route(ctx: ApiCtx, state: RuntimeApiState) -> ApiRoute {
    warp::path!("api" / "models" / "downloads" / String / "cancel")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |id: String, auth: Option<String>| {
            let state = state.clone();
            let ctx = ctx.clone();
            async move {
                if !check_api_token(&auth, &ctx.config) {
                    return Ok::<ApiReply, warp::Rejection>(unauthorized_api_token());
                }
                let downloads = state
                    .model_downloads
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                Ok(match downloads.get(&id) {
                    Some(job) => {
                        let active = matches!(
                            job.state,
                            RuntimeJobState::Queued | RuntimeJobState::Running
                        );
                        if active {
                            job.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                        }
                        Box::new(warp::reply::json(&serde_json::json!({
                            "ok": true,
                            "cancelling": active,
                        }))) as ApiReply
                    }
                    None => Box::new(warp::reply::with_status(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": "Unknown download job"
                        })),
                        StatusCode::NOT_FOUND,
                    )) as ApiReply,
                })
            }
        })
        .boxed()
}

/// GET /api/models/downloads/:jobId — poll a model download job.
fn model_download_status_route(ctx: ApiCtx, state: RuntimeApiState) -> ApiRoute {
    warp::path!("api" / "models" / "downloads" / String)
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |id: String, auth: Option<String>| {
            let state = state.clone();
            let ctx = ctx.clone();
            async move {
                if !check_api_token(&auth, &ctx.config) {
                    return Ok::<ApiReply, warp::Rejection>(unauthorized_api_token());
                }
                let job = state
                    .model_downloads
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .get(&id)
                    .cloned();
                Ok(match job {
                    Some(job) => Box::new(warp::reply::json(&serde_json::json!({
                        "ok": true,
                        "job": job,
                    }))) as ApiReply,
                    None => Box::new(warp::reply::with_status(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": "Unknown download job"
                        })),
                        StatusCode::NOT_FOUND,
                    )) as ApiReply,
                })
            }
        })
        .boxed()
}

fn escape_hatch_route(ctx: ApiCtx) -> ApiRoute {
    let config = ctx.config;
    warp::path!("api" / "rapid-mlx" / "escape-hatch-flags")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .map(move |auth: Option<String>| {
            if !check_api_token(&auth, &config) {
                return unauthorized_api_token();
            }
            Box::new(warp::reply::json(
                &crate::inference::rapid_mlx::escape_hatch::ALLOWED_ESCAPE_FLAGS,
            )) as ApiReply
        })
        .boxed()
}

/// Serve the Rapid-MLX semantic setting catalog, resolved against a capability snapshot.
///
/// This is the endpoint `settings.rs` was written for: it is the authoritative Rust
/// definition of what each setting is, what it defaults to, and why it is unavailable — so
/// the wizard and preset editor read capability gating from one place instead of each
/// re-deriving it from flag strings.
///
/// Capability gating needs a snapshot. A caller may pass `serve_flags` to resolve against a
/// specific runtime (the wizard previewing a version it has not installed); otherwise the
/// live discovered runtime is probed. When neither is available the catalog is still served
/// with everything ungated and `snapshot_source: "none"`, because a settings list the user
/// cannot see at all is worse than one whose availability is marked unknown.
fn settings_catalog_route(ctx: ApiCtx) -> ApiRoute {
    use crate::inference::rapid_mlx::settings;

    let config = ctx.config;
    warp::path!("api" / "rapid-mlx" / "settings")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and(warp::query::<SettingsCatalogQuery>())
        .and_then(move |auth: Option<String>, query: SettingsCatalogQuery| {
            let config = config.clone();
            async move {
                if !check_api_token(&auth, &config) {
                    return Ok(unauthorized_api_token());
                }
                let (snapshot, source) = resolve_catalog_snapshot(query.serve_flags).await;
                let settings: Vec<_> = settings::all_settings()
                    .iter()
                    .map(|setting| {
                        let supported = snapshot
                            .as_ref()
                            .map(|snap| setting.capability(snap))
                            .unwrap_or(true);
                        serde_json::json!({
                            "id": setting.id(),
                            "default": setting.default_value(),
                            "supported": supported,
                            "unsupported_reason": snapshot
                                .as_ref()
                                .and_then(|snap| setting.unsupported_reason(snap)),
                        })
                    })
                    .collect();
                let rules: Vec<_> = settings::mutual_exclusion_rules()
                    .iter()
                    .map(|rule| {
                        serde_json::json!({"settings": rule.setting_ids(), "error": rule.error})
                    })
                    .collect();
                Ok::<ApiReply, warp::Rejection>(Box::new(warp::reply::json(&serde_json::json!({
                    "ok": true,
                    "snapshot_source": source,
                    "rapid_mlx_version": snapshot.as_ref().map(|s| s.rapid_mlx_version.clone()),
                    "settings": settings,
                    "mutual_exclusion_rules": rules,
                }))) as ApiReply)
            }
        })
        .boxed()
}

#[derive(Debug, Deserialize)]
pub struct SettingsCatalogQuery {
    /// Comma-separated serve flags to gate against, instead of probing the live runtime.
    pub serve_flags: Option<String>,
}

/// Pick the capability snapshot the catalog is resolved against, and say which it was.
///
/// The source is returned rather than inferred by the client, because "this flag is
/// unsupported" and "we could not tell" must not look alike in the UI.
async fn resolve_catalog_snapshot(
    serve_flags: Option<String>,
) -> (Option<CapabilitySnapshot>, &'static str) {
    if let Some(raw) = serve_flags {
        let flags: Vec<String> = raw
            .split(',')
            .map(|flag| flag.trim().to_string())
            .filter(|flag| !flag.is_empty())
            .collect();
        return (
            Some(CapabilitySnapshot {
                serve_flags: flags,
                ..Default::default()
            }),
            "caller_flags",
        );
    }
    match capabilities::generate_snapshot_from_discovery().await {
        Ok(snapshot) => (Some(snapshot), "discovered"),
        Err(_) => (None, "none"),
    }
}

/// Validate a proposed set of settings and explain what each one resolves to.
///
/// Validation and effective-policy resolution live in the catalog, so the answer here is the
/// same answer launch will act on. Returning both together matters: a value can be perfectly
/// valid and still not be what runs, and a user told only "valid" would never learn that.
fn settings_validate_route(ctx: ApiCtx) -> ApiRoute {
    use crate::inference::rapid_mlx::settings::{self, ValidationContext};
    use std::collections::BTreeMap;

    let config = ctx.config;
    warp::path!("api" / "rapid-mlx" / "settings" / "validate")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<SettingsValidateRequest>())
        .and_then(move |auth: Option<String>, req: SettingsValidateRequest| {
            let config = config.clone();
            async move {
                if !check_api_token(&auth, &config) {
                    return Ok(unauthorized_api_token());
                }
                let (snapshot, source) = resolve_catalog_snapshot(req.serve_flags).await;
                let context = ValidationContext {
                    capabilities: snapshot.as_ref(),
                    workload_scenario: None,
                    ..Default::default()
                };

                let mut errors = Vec::new();
                let mut effective = serde_json::Map::new();
                // Unknown ids are reported rather than ignored: silently dropping a setting
                // the client believes it set is how a UI ends up showing a value that never
                // reaches launch.
                let mut unknown: Vec<&str> = Vec::new();
                let mut by_id = BTreeMap::new();

                for (id, value) in &req.values {
                    let Some(setting) = settings::all_settings()
                        .iter()
                        .find(|setting| setting.id() == id)
                    else {
                        unknown.push(id.as_str());
                        continue;
                    };
                    by_id.insert(setting.id(), value.clone());
                    if let Err(error) = setting.validate(value, &context) {
                        errors.push(error);
                        continue;
                    }
                    if let Some(snap) = snapshot.as_ref() {
                        let resolved = setting.effective_policy(value, snap);
                        let reason = resolved
                            .get("reason")
                            .and_then(|reason| reason.as_str())
                            .map(String::from);
                        let explanation = settings::EffectivePolicyExplanation {
                            requested: value.clone(),
                            effective: resolved.clone(),
                            reason,
                            reason_code: None,
                        };
                        effective.insert(
                            setting.id().to_string(),
                            serde_json::json!({
                                "explanation": explanation,
                                // The argv this setting contributes at launch. Showing it is
                                // what closes the catalog's promised trace from capability
                                // through validation to launch mapping — without it the user
                                // is told a value is fine but never what it does.
                                "cli_args": setting.to_cli_args(&resolved),
                            }),
                        );
                    }
                }

                if let Err(error) = settings::check_mutual_exclusions(&by_id) {
                    errors.push(error);
                }

                Ok::<ApiReply, warp::Rejection>(Box::new(warp::reply::json(&serde_json::json!({
                    "ok": true,
                    "valid": errors.is_empty(),
                    "snapshot_source": source,
                    "errors": errors,
                    "effective": effective,
                    "unknown_settings": unknown,
                }))) as ApiReply)
            }
        })
        .boxed()
}

#[derive(Debug, Deserialize)]
pub struct SettingsValidateRequest {
    /// Setting id -> proposed value.
    #[serde(default)]
    pub values: std::collections::BTreeMap<String, serde_json::Value>,
    /// Comma-separated serve flags to gate against, instead of probing the live runtime.
    pub serve_flags: Option<String>,
}

fn command_preview_route(ctx: ApiCtx) -> ApiRoute {
    let config = ctx.config;
    let state = ctx.state;
    warp::path!("api" / "rapid-mlx" / "command-preview")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<CommandPreviewRequest>())
        .and_then(move |auth: Option<String>, req: CommandPreviewRequest| {
            let config = config.clone();
            let state = state.clone();
            async move {
                if !check_api_token(&auth, &config) {
                    return Ok(unauthorized_api_token());
                }
                let models_dir = super::models::get_effective_models_dir(&state)
                    .unwrap_or_else(|| config.default_models_dir.clone());
                let reply = build_command_preview(req, models_dir).await;
                Ok::<ApiReply, warp::Rejection>(reply)
            }
        })
        .boxed()
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CommandPreviewRequest {
    #[serde(flatten)]
    pub config: RapidMlxConfig,
    /// Path to the Rapid-MLX executable to use for command building.
    pub executable_path: Option<String>,
    /// Optional capabilities override for testing without live probing.
    pub capabilities: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub struct CommandPreviewResponse {
    pub argv: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub redacted_summary: Option<String>,
    pub redacted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effective_policy: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requested_vs_effective: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub reasons: Vec<String>,
}

async fn build_command_preview(
    req: CommandPreviewRequest,
    models_dir: std::path::PathBuf,
) -> ApiReply {
    use crate::inference::rapid_mlx::compatibility::ServeCapabilities;
    use crate::inference::rapid_mlx::model_resolver::{self, RapidMlxResolveContext};
    use std::path::PathBuf;

    let config = req.config;
    // A preview that demands the caller already know the binary path is a preview no UI can
    // call, which is why this endpoint had no consumer. Fall back to the same resolution the
    // launcher uses (explicit -> managed -> PATH) so the frontend can just post a config.
    let binary_path = match req.executable_path {
        Some(path) => {
            let path = PathBuf::from(path);
            if !path.exists() {
                return json_error(
                    StatusCode::BAD_REQUEST,
                    format!("Executable not found: {}", path.display()),
                );
            }
            path
        }
        None => {
            match crate::inference::rapid_mlx::discovery::Discovery::resolve_binary(
                config.executable_path.as_deref(),
                config.managed_runtime_path.as_deref(),
            )
            .await
            {
                Ok((path, _source)) => path,
                Err(e) => {
                    return json_error(
                        StatusCode::BAD_REQUEST,
                        format!("Could not locate the Rapid-MLX executable: {}", e),
                    );
                }
            }
        }
    };

    let model_source = match &config.model_source {
        Some(src) => src.clone(),
        None => match model_resolver::source_from_legacy_model_path(&config.model_path) {
            Ok(src) => src,
            Err(e) => {
                return json_error(
                    StatusCode::BAD_REQUEST,
                    format!("Failed to resolve model from legacy path: {}", e),
                );
            }
        },
    };

    let model = match model_resolver::resolve(
        model_source,
        &RapidMlxResolveContext {
            models_dir,
            python_executable: PathBuf::from(if cfg!(windows) {
                "python.exe"
            } else {
                "python3"
            }),
            // Only quoted in error text; see the same field in models.rs.
            runtime_version: "runtime".into(),
            hf_token: None,
            verified_aliases: Vec::new(),
            execute_conversion: false,
        },
    )
    .await
    {
        Ok(m) => m,
        Err(e) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                format!("Failed to resolve model: {}", e),
            );
        }
    };

    let capabilities: ServeCapabilities = match req.capabilities {
        Some(flags) => ServeCapabilities::from_flags(&flags),
        None => {
            match tokio::process::Command::new(&binary_path)
                .arg("serve")
                .arg("--help")
                .output()
                .await
            {
                Ok(output) if output.status.success() => {
                    // Read both streams, the way every other capability probe in the tree
                    // does. Reading only stderr made this endpoint useless against a real
                    // runtime: 0.11.1 puts the whole help text on stdout, so the parse
                    // returned an empty capability set and every flag was reported
                    // unsupported. An empty parse is a failed probe, not a runtime that
                    // supports nothing, so it falls back rather than fails closed.
                    let help = crate::inference::rapid_mlx::compatibility::output_text(
                        &output.stdout,
                        &output.stderr,
                    );
                    if help.trim().is_empty() {
                        ServeCapabilities::verified_baseline()
                    } else {
                        ServeCapabilities::from_help(&help)
                    }
                }
                _ => ServeCapabilities::verified_baseline(),
            }
        }
    };

    // Drive the supervisor's own argv mapping through a throwaway adapter rather than
    // restating it here. A preview that fills in its own defaults is worse than no preview:
    // it shows the operator a command the launcher will not run, which is exactly how this
    // endpoint had drifted before the Phase 7A2 reconciliation.
    //
    // The adapter is a settings carrier only — nothing downstream reads its runtime metadata,
    // compatibility profile, or pollers, and `capabilities` below stays the caller's.
    let adapter = crate::inference::rapid_mlx::RapidMlxAdapter::for_settings_preview(
        binary_path.clone(),
        model,
        &config,
    );

    // One correspondence the preview cannot reproduce: `build_launch` re-resolves `hybrid_mode`
    // from the model's own config, so a model whose metadata forces hybrid may launch with a
    // switch the preview does not show. The preview reports the configured value.
    let (argv_builder, overlay_warning) = crate::inference::rapid_mlx::build_launch_argv(&adapter);
    let builder = crate::inference::rapid_mlx::apply_phase7_adapter_config(argv_builder, &adapter);

    let launch = match builder.build(binary_path, &capabilities) {
        Ok(l) => l,
        Err(e) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                format!("Command build failed: {}", e),
            );
        }
    };

    let argv: Vec<String> = launch
        .args
        .iter()
        .filter_map(|s| s.to_str().map(String::from))
        .collect();

    let mut reasons = Vec::new();
    check_setting_warnings(&config, &mut reasons);
    reasons.extend(overlay_warning);

    // Build effective_policy snapshot from config fields that were actually applied
    let effective_policy = build_effective_policy(&config);

    // Build requested_vs_effective diff using the setting catalog
    let requested_vs_effective = build_requested_vs_effective(&config, &capabilities);

    Box::new(warp::reply::json(&CommandPreviewResponse {
        argv,
        redacted_summary: Some(launch.redacted_summary),
        redacted: true,
        effective_policy: Some(effective_policy),
        requested_vs_effective: Some(requested_vs_effective),
        reasons,
    })) as ApiReply
}

fn build_effective_policy(config: &RapidMlxConfig) -> serde_json::Value {
    // Keep this snapshot honest: advanced TurboQuant requests are persisted so
    // the user can see and qualify them, but launch deliberately omits them
    // until an exact model/revision receipt exists. The requested-vs-effective
    // diff below carries the corresponding reason.
    let effective_turboquant_mode = match config.turboquant_mode {
        Some(crate::inference::rapid_mlx::TurboQuantMode::V4)
        | Some(crate::inference::rapid_mlx::TurboQuantMode::K8V4) => {
            Some(crate::inference::rapid_mlx::TurboQuantMode::Off)
        }
        ref other => other.clone(),
    };
    // `--reasoning` pins the KV cache to int8 inside the runtime, whatever `--kv-cache-dtype`
    // says (rapid-mlx `serve --help`: "pins --kv-cache-dtype to int8 regardless of the dtype
    // flag"). The VRAM estimator already models this as `reasoning_mode_overrides_kv_to_int8`;
    // reporting the requested dtype here would make this snapshot the one surface that
    // disagrees, on the interaction that matters most for memory.
    let effective_kv_cache_dtype = Some(crate::inference::rapid_mlx::KvCacheConfig::Int8);
    serde_json::json!({
        "kv_cache_dtype": effective_kv_cache_dtype,
        "turboquant_mode": effective_turboquant_mode,
        "prefix_cache_enabled": config.prefix_cache_enabled,
        "retained_cache_mib": config.retained_cache_mib,
        "disk_checkpoint_interval": config.disk_checkpoint_interval,
        "hybrid_cache_entries": config.hybrid_cache_entries,
        "pflash_policy": config.pflash_policy,
        "max_num_seqs": config.max_num_seqs,
        "max_concurrent_requests": config.max_concurrent_requests,
        "prefill_batch_size": config.prefill_batch_size,
        "completion_batch_size": config.completion_batch_size,
        "reasoning_mode": "on",
        "speculative_config": config.speculative_config,
        "mllm_vision": config.mllm_vision,
        "embeddings": config.embeddings,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "sampling_mode": config.sampling_mode,
    })
}

fn build_requested_vs_effective(
    config: &RapidMlxConfig,
    capabilities: &crate::inference::rapid_mlx::compatibility::ServeCapabilities,
) -> serde_json::Value {
    let mut diff = serde_json::Map::new();

    // KV cache dtype: may be downgraded if runtime doesn't support requested dtype
    if let Some(ref dtype) = config.kv_cache_dtype {
        let requested = dtype.to_string();
        let _effective = match dtype {
            crate::inference::rapid_mlx::KvCacheConfig::Auto => "auto".to_string(),
            crate::inference::rapid_mlx::KvCacheConfig::Bf16
            | crate::inference::rapid_mlx::KvCacheConfig::Int8
            | crate::inference::rapid_mlx::KvCacheConfig::Int4 => {
                if !capabilities.contains("--kv-cache-dtype") {
                    diff.insert("kv_cache_dtype".to_string(), serde_json::json!({
                        "requested": requested,
                        "effective": "auto",
                        "reason": "KV cache dtype flag not supported by this runtime version; using runtime default"
                    }));
                    "auto".to_string()
                } else {
                    requested.clone()
                }
            }
            crate::inference::rapid_mlx::KvCacheConfig::LegacyFp16
            | crate::inference::rapid_mlx::KvCacheConfig::LegacyFp8 => {
                diff.insert("kv_cache_dtype".to_string(), serde_json::json!({
                    "requested": requested,
                    "effective": null,
                    "reason": "This preset contains a legacy unsupported KV dtype. Choose bf16, int8, or int4 before launching."
                }));
                "invalid_legacy_value".to_string()
            }
        };
    }

    // Reasoning outranks the dtype flag inside the runtime, so say so rather than let the
    // operator read a dtype the server will not use. Recorded even when the requested dtype
    // is already int8-compatible only if it actually differs.
    let requested = config
        .kv_cache_dtype
        .as_ref()
        .map(ToString::to_string)
        .unwrap_or_else(|| "auto".to_string());
    if requested != "int8" {
        diff.insert(
            "kv_cache_dtype".to_string(),
            serde_json::json!({
                "requested": requested,
                "effective": "int8",
                "reason": "the always-on Rapid reasoning quality profile pins the KV cache to int8 regardless of the requested dtype"
            }),
        );
    }
    if config.reasoning_mode.as_deref() == Some("off") {
        diff.insert(
            "reasoning_mode".to_string(),
            serde_json::json!({
                "requested": "off",
                "effective": "on",
                "reason": "the Rapid reasoning quality profile remains on; this legacy off value adds --no-thinking instead"
            }),
        );
    }
    if let Some(ref requested) = config.speculative_config
        && !capabilities.contains("--speculative-config")
    {
        diff.insert(
            "speculative_config".to_string(),
            serde_json::json!({
                "requested": requested,
                "effective": null,
                "reason": "--speculative-config is not supported by this runtime version; speculative decoding was omitted"
            }),
        );
    }

    // PFlash: may be unavailable if runtime lacks flag
    if let Some(ref policy) = config.pflash_policy
        && policy != "auto"
        // "off" against a runtime with no PFlash is already satisfied, not a downgrade.
        && policy != "off"
        && !capabilities.contains("--pflash")
    {
        diff.insert(
            "pflash_policy".to_string(),
            serde_json::json!({
                "requested": policy,
                "effective": "auto",
                "reason": "PFlash not supported by this runtime version; using runtime default"
            }),
        );
    }

    // TurboQuant requires an exact model/revision qualification receipt. The
    // current runtime snapshot establishes only that the flag exists, not that
    // this model's retained KV path is eligible. Keep the requested value
    // visible, but omit it from preview/launch until that evidence is wired.
    if let Some(ref mode) = config.turboquant_mode {
        let mode_str = mode.to_string();
        if mode_str != "auto" && mode_str != "none" {
            diff.insert(
                "turboquant_mode".to_string(),
                serde_json::json!({
                    "requested": mode_str,
                    "effective": "none",
                    "reason": if capabilities.contains("--kv-cache-turboquant") {
                        "No exact model/revision TurboQuant qualification receipt is available; disabled"
                    } else {
                        "TurboQuant is not supported by this runtime version; disabled"
                    }
                }),
            );
        }
    }

    // The three throughput tuning values below are dropped from argv on a runtime that lacks
    // the flag rather than failing the launch. Each is reported so the user sees that the
    // runtime, not their choice, is deciding scheduling and memory headroom.
    for (name, flag, requested) in [
        (
            "max_num_seqs",
            "--max-num-seqs",
            config.max_num_seqs.map(|v| v.to_string()),
        ),
        (
            "max_concurrent_requests",
            "--max-concurrent-requests",
            config.max_concurrent_requests.map(|v| v.to_string()),
        ),
        (
            "gpu_memory_utilization",
            "--gpu-memory-utilization",
            config.gpu_memory_utilization.map(|v| v.to_string()),
        ),
    ] {
        if let Some(value) = requested
            && !capabilities.contains(flag)
        {
            diff.insert(
                name.to_string(),
                serde_json::json!({
                    "requested": value,
                    "effective": "runtime default",
                    "reason": format!("{flag} is not supported by this runtime version; omitted")
                }),
            );
        }
    }

    serde_json::Value::Object(diff)
}

fn check_setting_warnings(_config: &RapidMlxConfig, warnings: &mut Vec<String>) {
    // Check mutual exclusions
    let mut settings_map = std::collections::BTreeMap::new();
    // The reasoning quality profile is always on. A legacy reasoning_mode=off value now
    // controls --no-thinking only and must not suppress quality-profile compatibility checks.
    settings_map.insert("reasoning_mode", serde_json::json!("on"));
    if let Some(sampling) = _config.sampling_mode.as_deref() {
        settings_map.insert("sampling_mode", serde_json::json!(sampling));
    }
    if let Err(e) = check_mutual_exclusions(&settings_map) {
        warnings.push(e.message);
    }
}

fn runtime_metadata_route(ctx: ApiCtx, state: RuntimeApiState) -> ApiRoute {
    let config = ctx.config;
    warp::path!("api" / "rapid-mlx" / "runtime" / "metadata")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let config = config.clone();
            let state = state.clone();
            async move {
                if !check_api_token(&auth, &config) {
                    return Ok(unauthorized_api_token());
                }
                if crate::inference::rapid_mlx::ensure_local_platform_supported().is_err() {
                    return Ok(json_error(
                        StatusCode::BAD_REQUEST,
                        "Rapid-MLX metadata queries require Apple Silicon macOS",
                    ));
                }
                let manager_result = manager(&state);
                let mut active_info: Option<(std::path::PathBuf, ManagedReleaseChannel)> = None;
                if let Ok(manager) = &manager_result {
                    let manager = manager.clone();
                    if let Ok(Ok(status)) =
                        tokio::task::spawn_blocking(move || manager.status()).await
                        && let Some(active) = status.active
                    {
                        active_info = Some((active.executable_path, active.release_channel));
                    }
                }
                let binary_path = active_info.as_ref().map(|(p, _)| p.as_path());
                let Ok((binary, source)) =
                    crate::inference::rapid_mlx::discovery::Discovery::resolve_binary(
                        None,
                        binary_path,
                    )
                    .await
                else {
                    return Ok(json_error(
                        StatusCode::NOT_FOUND,
                        "Rapid-MLX binary not found. Run rapid-mlx doctor or install via Settings.",
                    ));
                };
                let allow_prerelease = active_info
                    .as_ref()
                    .is_some_and(|(p, c)| p == &binary && *c == ManagedReleaseChannel::Prerelease);
                // Probe compatibility to trigger capability snapshot generation
                if source
                    == crate::inference::rapid_mlx::runtime::RuntimeSource::Managed
                {
                    if allow_prerelease {
                        if crate::inference::rapid_mlx::compatibility::probe_published_managed_release(
                            &binary,
                            allow_prerelease,
                        )
                        .await
                        .is_err()
                        {
                            return Ok(json_error(
                                StatusCode::SERVICE_UNAVAILABLE,
                                "Rapid-MLX runtime probe failed",
                            ));
                        }
                    } else if crate::inference::rapid_mlx::compatibility::probe(&binary, source)
                        .await
                        .is_err()
                    {
                        return Ok(json_error(
                            StatusCode::SERVICE_UNAVAILABLE,
                            "Rapid-MLX runtime probe failed",
                        ));
                    }
                } else if crate::inference::rapid_mlx::compatibility::probe(&binary, source).await
                    .is_err()
                {
                    return Ok(json_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Rapid-MLX runtime probe failed",
                    ));
                };

                // Fetch the capability snapshot for this binary
                let identity =
                    match crate::inference::rapid_mlx::capabilities::ExecutableIdentity::from_path(
                        &binary,
                    ) {
                        Ok(id) => id,
                        Err(e) => {
                            return Ok(json_error(
                                StatusCode::INTERNAL_SERVER_ERROR,
                                format!("Failed to compute executable identity: {e}"),
                            ));
                        }
                    };
                let snapshot =
                    match crate::inference::rapid_mlx::capabilities::cached_snapshot(&identity) {
                    Some(s) => s,
                    None => {
                        return Ok(json_error(
                            StatusCode::NOT_FOUND,
                            "No capability snapshot available for this runtime",
                        ));
                    }
                };

                // D27: expose sampling defaults; only supported fields are effective
                let effective_sampling_defaults = snapshot.sampling_defaults.effective_fields();

                Ok::<ApiReply, warp::Rejection>(Box::new(warp::reply::json(
                    &serde_json::json!({
                        "ok": true,
                        "version": snapshot.rapid_mlx_version,
                        "source": source,
                        "sampling_defaults": snapshot.sampling_defaults,
                        "effective_sampling_default_fields": effective_sampling_defaults,
                                "sampling_cascade": snapshot.sampling_cascade,
                                "mtp_concurrency": snapshot.mtp_concurrency,
                                "qualified_features": snapshot.qualified_features,
                                "evidence_timestamp": snapshot.evidence_timestamp,
                                "measured_spec_decode": snapshot.measured_spec_decode,
                                "superseded_spec_decode": snapshot.superseded_spec_decode,
                            }),
                )))
            }
        })
        .boxed()
}

fn prefix_cache_guidance_route(ctx: ApiCtx) -> ApiRoute {
    let config = ctx.config;
    warp::path!("api" / "rapid-mlx" / "runtime" / "prefix-cache-guidance")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<serde_json::Value>())
        .map(move |auth: Option<String>, body: serde_json::Value| {
            if !check_api_token(&auth, &config) {
                return Box::new(warp::reply::with_status(
                    warp::reply::json(&serde_json::json!({"ok": false, "error": "Unauthorized"})),
                    warp::http::StatusCode::UNAUTHORIZED,
                )) as ApiReply;
            }
            // Pure derivation from provided parameters — no runtime state needed.
            // This is called by the wizard/preset editor to compute guidance.
            let configured_ceiling_bytes =
                body["configured_ceiling_bytes"].as_u64().unwrap_or(0u64);
            let current_safe_bytes = body["current_safe_bytes"].as_u64().unwrap_or(0u64);
            let estimated_model_overhead_bytes = body["estimated_model_overhead_bytes"]
                .as_u64()
                .unwrap_or(0u64);
            let user_max_cache_blocks: Option<u32> =
                body["user_max_cache_blocks"].as_u64().map(|v| v as u32);
            let arch_n_embd = body["arch_n_embd"].as_u64().unwrap_or(0) as u32;
            let arch_n_kv_heads = body["arch_n_kv_heads"].as_u64().unwrap_or(0) as u32;
            let arch_head_dim = body["arch_head_dim"].as_u64().unwrap_or(0) as u32;
            // Accept serve_flags from caller (from actual capability snapshot) so guidance
            // reflects real runtime capabilities rather than synthetic assumptions.
            let serve_flags: Vec<String> = body["serve_flags"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            // Use a synthetic capability snapshot with caller-provided serve_flags
            let snapshot = crate::inference::rapid_mlx::capabilities::CapabilitySnapshot {
                executable_identity: Default::default(),
                rapid_mlx_version: String::new(),
                help_hash: String::new(),
                serve_flags,
                package_versions: vec![],
                installed_extras: Default::default(),
                qualified_features: Default::default(),
                mtp_concurrency: Default::default(),
                sampling_defaults: Default::default(),
                sampling_cascade: Default::default(),
                evidence_timestamp: 0,
                source: Default::default(),
                measured_spec_decode: None,
                superseded_spec_decode: None,
            };

            let guidance = crate::inference::rapid_mlx::capabilities::PrefixCacheGuidance::derive(
                &snapshot,
                configured_ceiling_bytes,
                current_safe_bytes,
                estimated_model_overhead_bytes,
                user_max_cache_blocks,
                arch_n_embd,
                arch_n_kv_heads,
                arch_head_dim,
            );

            Box::new(warp::reply::json(&serde_json::json!({
                "ok": true,
                "guidance": guidance,
            }))) as ApiReply
        })
        .boxed()
}

fn recommendation_route(ctx: ApiCtx, state: RuntimeApiState) -> ApiRoute {
    let config = ctx.config;
    warp::path!("api" / "rapid-mlx" / "recommend")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<RecommendationRequest>())
        .and_then(
            move |auth: Option<String>, request: RecommendationRequest| {
                let config = config.clone();
                let state = state.clone();
                async move {
                    if !check_api_token(&auth, &config) {
                        return Ok(unauthorized_api_token());
                    }
                    let local_available =
                        crate::inference::rapid_mlx::ensure_local_platform_supported().is_ok();
                    let runtime_compatible = if local_available {
                        compatible_runtime_available(&state).await
                    } else {
                        false
                    };
                    let recommendation = recommend_backend(&BackendRecommendationInput {
                        artifact_kind: request.artifact_kind,
                        rapid_mlx_local_available: local_available,
                        rapid_mlx_runtime_compatible: runtime_compatible,
                    });
                    Ok::<ApiReply, warp::Rejection>(Box::new(warp::reply::json(&recommendation)))
                }
            },
        )
        .boxed()
}

async fn compatible_runtime_available(state: &RuntimeApiState) -> bool {
    let managed_active = match manager(state) {
        Ok(manager) => tokio::task::spawn_blocking(move || manager.status())
            .await
            .ok()
            .and_then(Result::ok)
            .and_then(|status| {
                status
                    .active
                    .map(|active| (active.executable_path, active.release_channel))
            }),
        Err(_) => None,
    };
    let managed_path = managed_active.as_ref().map(|(path, _)| path.as_path());
    let Ok((binary, source)) = Discovery::resolve_binary(None, managed_path).await else {
        return false;
    };
    if source == crate::inference::rapid_mlx::runtime::RuntimeSource::Managed {
        let allow_prerelease = managed_prerelease_allowed(managed_active.as_ref(), &binary);
        compatibility::probe_published_managed_release(&binary, allow_prerelease)
            .await
            .is_ok()
    } else {
        compatibility::probe(&binary, source).await.is_ok()
    }
}

fn managed_prerelease_allowed(
    managed_active: Option<&(std::path::PathBuf, ManagedReleaseChannel)>,
    resolved_binary: &std::path::Path,
) -> bool {
    managed_active.is_some_and(|(path, channel)| {
        path == resolved_binary && *channel == ManagedReleaseChannel::Prerelease
    })
}

fn status_route(ctx: ApiCtx, state: RuntimeApiState) -> ApiRoute {
    let config = ctx.config;
    warp::path!("api" / "rapid-mlx" / "runtime" / "status")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let config = config.clone();
            let state = state.clone();
            async move {
                if !check_api_token(&auth, &config) {
                    return Ok(unauthorized_api_token());
                }
                let manager = match manager(&state) {
                    Ok(manager) => manager,
                    Err(message) => {
                        return Ok(json_error(StatusCode::INTERNAL_SERVER_ERROR, message));
                    }
                };
                let status = tokio::task::spawn_blocking(move || manager.status()).await;
                match status {
                    Ok(Ok(status)) => Ok::<ApiReply, warp::Rejection>(Box::new(warp::reply::json(
                        &serde_json::json!({
                            "runtime": PublicManagedRuntimeStatus::from(status),
                            "jobs": job_list(&state),
                        }),
                    ))),
                    _ => Ok(json_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Managed Rapid-MLX status is unavailable",
                    )),
                }
            }
        })
        .boxed()
}

fn releases_route(ctx: ApiCtx, state: RuntimeApiState) -> ApiRoute {
    let config = ctx.config;
    warp::path!("api" / "rapid-mlx" / "runtime" / "releases")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let config = config.clone();
            let state = state.clone();
            async move {
                if !check_api_token(&auth, &config) {
                    return Ok(unauthorized_api_token());
                }
                match published_releases(&state).await {
                    Ok(releases) => Ok::<ApiReply, warp::Rejection>(Box::new(warp::reply::json(
                        &serde_json::json!({ "releases": releases }),
                    ))),
                    Err(_) => Ok(json_error(
                        StatusCode::BAD_GATEWAY,
                        "Rapid-MLX release discovery is temporarily unavailable",
                    )),
                }
            }
        })
        .boxed()
}

/// The curated Rapid-MLX model catalog: rows of the upstream-validated
/// `rapid-mlx models` listing (alias, measured size, parser pairing, MTP
/// sidecar, hybrid marker). This list — not raw HF discovery — is the
/// Rapid-MLX model surface in the spawn wizard.
fn catalog_route(ctx: ApiCtx, state: RuntimeApiState) -> ApiRoute {
    let config = ctx.config;
    warp::path!("api" / "rapid-mlx" / "catalog")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let config = config.clone();
            let state = state.clone();
            async move {
                if !check_api_token(&auth, &config) {
                    return Ok(unauthorized_api_token());
                }
                let managed_path = managed_executable(&state).await;
                let binary = match Discovery::resolve_binary(None, managed_path.as_deref()).await {
                    Ok((binary, _source)) => binary,
                    Err(_) => {
                        return Ok::<ApiReply, warp::Rejection>(json_error(
                            StatusCode::SERVICE_UNAVAILABLE,
                            "Rapid-MLX is not installed",
                        ));
                    }
                };
                let list = tokio::time::timeout(
                    Duration::from_secs(15),
                    info_query::fetch_model_list(&binary),
                )
                .await;
                // Tier recommendations are best-effort: an absent or unknown
                // `recipe` output must not take the catalog down.
                let recommendations =
                    tokio::time::timeout(Duration::from_secs(8), info_query::fetch_recipe(&binary))
                        .await
                        .ok()
                        .and_then(Result::ok)
                        .unwrap_or_default();
                match list {
                    Ok(Ok(models)) => Ok(Box::new(warp::reply::json(&serde_json::json!({
                        "ok": true,
                        "models": models,
                        "recommendations": recommendations,
                    }))) as ApiReply),
                    Ok(Err(message)) => Ok(json_error(
                        StatusCode::BAD_GATEWAY,
                        format!("Rapid-MLX catalog query failed: {message}"),
                    )),
                    Err(_) => Ok(json_error(
                        StatusCode::GATEWAY_TIMEOUT,
                        "Rapid-MLX catalog query timed out",
                    )),
                }
            }
        })
        .boxed()
}

async fn managed_executable(state: &RuntimeApiState) -> Option<std::path::PathBuf> {
    let manager = manager(state).ok()?;
    let status = tokio::task::spawn_blocking(move || manager.status())
        .await
        .ok()?
        .ok()?;
    status.active.map(|active| active.executable_path)
}

fn development_source_route(ctx: ApiCtx, state: RuntimeApiState) -> ApiRoute {
    let config = ctx.config.clone();
    let resolve = warp::path("api")
        .and(warp::path("rapid-mlx"))
        .and(warp::path("runtime"))
        .and(warp::path("development"))
        .and(warp::path("resolve"))
        .and(warp::path::end())
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<DevelopmentSourceRequest>())
        .and_then({
            let config = config.clone();
            let state = state.clone();
            move |auth: Option<String>, request: DevelopmentSourceRequest| {
                let config = config.clone();
                let state = state.clone();
                async move {
                    if !check_api_token(&auth, &config) {
                        return Ok(unauthorized_api_token());
                    }
                    match resolve_development_source(&state, &request).await {
                        Ok((selection, resolution)) => Ok::<ApiReply, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({
                                "ok": true,
                                "source": resolution,
                                "selection": selection,
                            })),
                        )),
                        Err(error) => Ok(json_error(StatusCode::BAD_REQUEST, error.to_string())),
                    }
                }
            }
        });

    let install = warp::path("api")
        .and(warp::path("rapid-mlx"))
        .and(warp::path("runtime"))
        .and(warp::path("development"))
        .and(warp::path("install"))
        .and(warp::path::end())
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<DevelopmentSourceRequest>())
        .and_then(
            move |auth: Option<String>, request: DevelopmentSourceRequest| {
                let config = config.clone();
                let state = state.clone();
                async move {
                    if !check_db_admin_token(&auth, &config) {
                        return Ok(unauthorized_db_admin_token());
                    }
                    if request.confirm != "INSTALL_RAPID_MLX_DEVELOPMENT" {
                        return Ok(json_error(
                            StatusCode::BAD_REQUEST,
                            "Confirmation must be INSTALL_RAPID_MLX_DEVELOPMENT",
                        ));
                    }
                    if crate::inference::rapid_mlx::ensure_local_platform_supported().is_err() {
                        return Ok(json_error(
                            StatusCode::BAD_REQUEST,
                            "Managed Rapid-MLX runtime changes require Apple Silicon macOS",
                        ));
                    }
                    let selection = match development_selection_from_request(&request) {
                        Ok(selection) => selection,
                        Err(error) => {
                            return Ok(json_error(StatusCode::BAD_REQUEST, error.to_string()));
                        }
                    };
                    start_job(
                        &state,
                        RuntimeOperation::InstallDevelopment,
                        Some(RuntimeJobSpec::Git(selection)),
                    )
                    .await
                }
            },
        );

    resolve.or(install).unify().boxed()
}

#[derive(Debug, Deserialize)]
struct GitHubCommitResponse {
    sha: String,
    html_url: String,
    commit: GitHubCommitDetails,
}

#[derive(Debug, Deserialize)]
struct GitHubCommitDetails {
    message: String,
}

#[derive(Debug, Deserialize)]
struct GitHubPullResponse {
    title: String,
    html_url: String,
    head: GitHubPullRef,
    base: GitHubPullRef,
}

#[derive(Debug, Deserialize)]
struct GitHubPullRef {
    sha: String,
}

fn development_selection_from_request(
    request: &DevelopmentSourceRequest,
) -> anyhow::Result<ManagedGitSourceSelection> {
    validate_development_input(&request.repository, &request.reference)?;
    let resolved_commit = request
        .resolved_commit
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("Resolve the development source before installing it"))?;
    ManagedGitSourceSelection::from_resolved(
        request.repository.clone(),
        request.reference.clone(),
        resolved_commit.to_string(),
        request.extras.clone(),
    )
}

async fn resolve_development_source(
    state: &RuntimeApiState,
    request: &DevelopmentSourceRequest,
) -> anyhow::Result<(ManagedGitSourceSelection, DevelopmentSourceResolution)> {
    validate_development_input(&request.repository, &request.reference)?;
    let (resolved_commit, commit_url, title, base_commit) =
        if let Some(number) = request.reference.strip_prefix("pr:") {
            let number: u64 = number
                .parse()
                .context("PR reference must use pr:<number>")?;
            if number == 0 {
                anyhow::bail!("PR reference must use a positive number");
            }
            let url = format!("{GITHUB_API_URL}/{}/pulls/{number}", request.repository);
            let body = fetch_bounded_release_body(&state.client, &url).await?;
            let pull: GitHubPullResponse = serde_json::from_slice(&body)
                .context("GitHub returned invalid pull request metadata")?;
            (
                pull.head.sha,
                pull.html_url,
                pull.title,
                Some(pull.base.sha),
            )
        } else {
            let url = format!(
                "{GITHUB_API_URL}/{}/commits/{}",
                request.repository, request.reference
            );
            let body = fetch_bounded_release_body(&state.client, &url).await?;
            let commit: GitHubCommitResponse =
                serde_json::from_slice(&body).context("GitHub returned invalid commit metadata")?;
            let title = commit
                .commit
                .message
                .lines()
                .next()
                .unwrap_or("Development source")
                .to_string();
            (commit.sha, commit.html_url, title, None)
        };
    let selection = ManagedGitSourceSelection::from_resolved(
        request.repository.clone(),
        request.reference.clone(),
        resolved_commit.clone(),
        request.extras.clone(),
    )?;
    let resolution = DevelopmentSourceResolution {
        repository: request.repository.clone(),
        requested_ref: request.reference.clone(),
        resolved_commit,
        commit_url,
        title,
        base_commit,
        extras: selection.extras().to_vec(),
    };
    Ok((selection, resolution))
}

fn validate_development_input(repository: &str, reference: &str) -> anyhow::Result<()> {
    let mut parts = repository.split('/');
    let owner = parts.next().unwrap_or_default();
    let name = parts.next().unwrap_or_default();
    if owner.is_empty()
        || name.is_empty()
        || parts.next().is_some()
        || !repository
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.' | b'/'))
    {
        anyhow::bail!("Repository must be a GitHub owner/repository identifier");
    }
    if reference.is_empty()
        || reference.len() > 256
        || reference.contains("..")
        || !reference.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.' | b'/' | b':')
        })
    {
        anyhow::bail!("Reference must be a branch, tag, commit SHA, or pr:<number>");
    }
    Ok(())
}

fn changelog_route(ctx: ApiCtx, state: RuntimeApiState) -> ApiRoute {
    let config = ctx.config;
    warp::path!("api" / "rapid-mlx" / "runtime" / "changelog")
        .and(warp::get())
        .and(warp::query::<ChangelogQuery>())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |query: ChangelogQuery, auth: Option<String>| {
            let config = config.clone();
            let state = state.clone();
            async move {
                if !check_api_token(&auth, &config) {
                    return Ok(unauthorized_api_token());
                }
                let result = changelog::fetch_compare(
                    &state.client,
                    &state.changelog_cache,
                    &query.from,
                    &query.to,
                )
                .await;

                match result {
                    Ok(summary) => Ok::<ApiReply, warp::Rejection>(Box::new(warp::reply::json(
                        &serde_json::json!({ "ok": true, "changelog": summary }),
                    ))),
                    Err(err) => {
                        let status = match err.kind {
                            changelog::ChangelogErrorKind::RateLimited => {
                                StatusCode::TOO_MANY_REQUESTS
                            }
                            changelog::ChangelogErrorKind::InvalidTag => StatusCode::NOT_FOUND,
                            _ => StatusCode::BAD_GATEWAY,
                        };
                        Ok(Box::new(warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({
                                "ok": false,
                                "error": err.message,
                                "kind": &err.kind,
                            })),
                            status,
                        )))
                    }
                }
            }
        })
        .boxed()
}

#[derive(Debug, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct ChangelogQuery {
    from: String,
    to: String,
}

fn mutation_route(ctx: ApiCtx, state: RuntimeApiState, operation: RuntimeOperation) -> ApiRoute {
    let config = ctx.config;
    let action = match operation {
        RuntimeOperation::Install => "install",
        RuntimeOperation::Upgrade => "upgrade",
        _ => unreachable!("release mutation route requires install or upgrade"),
    };
    warp::path("api")
        .and(warp::path("rapid-mlx"))
        .and(warp::path("runtime"))
        .and(warp::path(action))
        .and(warp::path::end())
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<RuntimeMutationRequest>())
        .and_then(
            move |auth: Option<String>, request: RuntimeMutationRequest| {
                let config = config.clone();
                let state = state.clone();
                async move {
                    if !check_db_admin_token(&auth, &config) {
                        return Ok(unauthorized_db_admin_token());
                    }
                    let expected = format!("{}_RAPID_MLX_RUNTIME", action.to_ascii_uppercase());
                    if request.confirm != expected || request.version.is_empty() {
                        return Ok(json_error(
                            StatusCode::BAD_REQUEST,
                            format!("Confirmation must be {expected} with an exact version"),
                        ));
                    }
                    if crate::inference::rapid_mlx::ensure_local_platform_supported().is_err() {
                        return Ok(json_error(
                            StatusCode::BAD_REQUEST,
                            "Managed Rapid-MLX runtime changes require Apple Silicon macOS",
                        ));
                    }
            let release = match select_published_release(
                &state,
                &request.version,
                request.channel,
                request.extras,
            )
                    .await
                    {
                        Ok(release) => release,
                        Err(_) => {
                            return Ok(json_error(
                                StatusCode::BAD_REQUEST,
                                "The selected Rapid-MLX release was not found in published release metadata",
                            ));
                        }
                    };
                    start_job(
                        &state,
                        operation,
                        Some(RuntimeJobSpec::Release(release)),
                    )
                    .await
                }
            },
        )
        .boxed()
}

fn simple_mutation_route(
    ctx: ApiCtx,
    state: RuntimeApiState,
    operation: RuntimeOperation,
) -> ApiRoute {
    let config = ctx.config;
    let action = match operation {
        RuntimeOperation::Repair => "repair",
        RuntimeOperation::Rollback => "rollback",
        _ => unreachable!("simple mutation route requires repair or rollback"),
    };
    warp::path("api")
        .and(warp::path("rapid-mlx"))
        .and(warp::path("runtime"))
        .and(warp::path(action))
        .and(warp::path::end())
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<RuntimeConfirmationRequest>())
        .and_then(
            move |auth: Option<String>, request: RuntimeConfirmationRequest| {
                let config = config.clone();
                let state = state.clone();
                async move {
                    if !check_db_admin_token(&auth, &config) {
                        return Ok(unauthorized_db_admin_token());
                    }
                    let expected = format!("{}_RAPID_MLX_RUNTIME", action.to_ascii_uppercase());
                    if request.confirm != expected {
                        return Ok(json_error(
                            StatusCode::BAD_REQUEST,
                            format!("Confirmation must be {expected}"),
                        ));
                    }
                    if crate::inference::rapid_mlx::ensure_local_platform_supported().is_err() {
                        return Ok(json_error(
                            StatusCode::BAD_REQUEST,
                            "Managed Rapid-MLX runtime changes require Apple Silicon macOS",
                        ));
                    }
                    let spec = if operation == RuntimeOperation::Repair {
                        let active_source = match manager(&state) {
                            Ok(manager) => {
                                tokio::task::spawn_blocking(move || manager.active_git_source())
                                    .await
                                    .ok()
                                    .and_then(Result::ok)
                                    .flatten()
                            }
                            Err(_) => None,
                        };
                        if let Some(source) = active_source {
                            Some(RuntimeJobSpec::Git(source))
                        } else {
                            match published_selection_for_active_runtime(&state).await {
                                Ok(release) => Some(RuntimeJobSpec::Release(release)),
                                Err(_) => {
                                    return Ok(json_error(
                                        StatusCode::BAD_REQUEST,
                                        "The active managed Rapid-MLX runtime could not be verified against its source metadata",
                                    ));
                                }
                            }
                        }
                    } else {
                        None
                    };
                    start_job(&state, operation, spec).await
                }
            },
        )
        .boxed()
}

async fn published_selection_for_active_runtime(
    state: &RuntimeApiState,
) -> anyhow::Result<ManagedReleaseSelection> {
    let manager = manager(state).map_err(|message| anyhow::anyhow!(message))?;
    let status = tokio::task::spawn_blocking(move || manager.status()).await??;
    let active = status
        .active
        .ok_or_else(|| anyhow::anyhow!("No active managed runtime"))?;
    select_published_release(
        state,
        &active.version,
        active.release_channel,
        active.extras,
    )
    .await
}

fn job_route(ctx: ApiCtx, state: RuntimeApiState) -> ApiRoute {
    let config = ctx.config;
    warp::path!("api" / "rapid-mlx" / "runtime" / "jobs" / String)
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |id: String, auth: Option<String>| {
            let config = config.clone();
            let state = state.clone();
            async move {
                if !check_api_token(&auth, &config) {
                    return Ok(unauthorized_api_token());
                }
                match state
                    .jobs
                    .lock()
                    .ok()
                    .and_then(|jobs| jobs.entries.get(&id).cloned())
                {
                    Some(job) => Ok::<ApiReply, warp::Rejection>(Box::new(warp::reply::json(&job))),
                    None => Ok(json_error(
                        StatusCode::NOT_FOUND,
                        "Runtime operation was not found",
                    )),
                }
            }
        })
        .boxed()
}

async fn start_job(
    state: &RuntimeApiState,
    operation: RuntimeOperation,
    spec: Option<RuntimeJobSpec>,
) -> Result<ApiReply, warp::Rejection> {
    let manager = match manager(state) {
        Ok(manager) => manager,
        Err(message) => return Ok(json_error(StatusCode::INTERNAL_SERVER_ERROR, message)),
    };
    let status_manager = manager.clone();
    if tokio::task::spawn_blocking(move || status_manager.status())
        .await
        .is_ok_and(|status| status.is_ok_and(|status| status.mutation_in_progress))
    {
        return Ok(json_error(
            StatusCode::TOO_MANY_REQUESTS,
            "Another managed Rapid-MLX runtime operation is already in progress",
        ));
    }
    let id = match random_job_id() {
        Ok(id) => id,
        Err(_) => {
            return Ok(json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Secure runtime operation ID generation is unavailable",
            ));
        }
    };
    let version = spec.as_ref().map(|item| match item {
        RuntimeJobSpec::Release(release) => release.version().to_string(),
        RuntimeJobSpec::Git(source) => format!("git:{}", source.resolved_commit()),
    });
    let version_for_log = version.clone();
    if !try_insert_job(
        state,
        RuntimeJobSnapshot {
            id: id.clone(),
            operation,
            state: RuntimeJobState::Queued,
            message: "Runtime operation queued".into(),
            version,
            result: None,
        },
    ) {
        return Ok(json_error(
            StatusCode::TOO_MANY_REQUESTS,
            "Another managed Rapid-MLX runtime operation is already in progress",
        ));
    }
    let job_state = state.clone();
    let job_id = id.clone();
    tokio::spawn(async move {
        let started = Instant::now();
        eprintln!(
            "[rapid-mlx] managed runtime {:?} {} started (job {job_id})",
            operation,
            version_for_log.as_deref().unwrap_or("unknown version"),
        );
        update_job(
            &job_state,
            &job_id,
            RuntimeJobState::Running,
            "Installing and validating an isolated runtime",
            None,
        );
        let result = match (operation, spec) {
            (RuntimeOperation::Install, Some(RuntimeJobSpec::Release(release))) => {
                manager.install_release(release).await
            }
            (RuntimeOperation::InstallDevelopment, Some(RuntimeJobSpec::Git(source))) => {
                manager.install_git_source(source).await
            }
            (RuntimeOperation::Upgrade, Some(RuntimeJobSpec::Release(release))) => {
                manager.upgrade_release(release).await
            }
            (RuntimeOperation::Repair, Some(RuntimeJobSpec::Release(release))) => {
                manager.repair_release(release).await
            }
            (RuntimeOperation::Repair, Some(RuntimeJobSpec::Git(source))) => {
                manager.upgrade_git_source(source).await
            }
            (RuntimeOperation::Rollback, None) => manager.rollback().await,
            _ => Err(anyhow::anyhow!("Invalid runtime operation state")),
        };
        match result {
            Ok(result) => {
                eprintln!(
                    "[rapid-mlx] managed runtime {:?} {} completed in {:.1}s",
                    operation,
                    version_for_log.as_deref().unwrap_or("unknown version"),
                    started.elapsed().as_secs_f64(),
                );
                update_job(
                    &job_state,
                    &job_id,
                    RuntimeJobState::Complete,
                    "Runtime validated and activated",
                    Some(result),
                );
            }
            Err(error) => {
                eprintln!(
                    "[rapid-mlx] managed runtime {:?} {} failed during validation: {error:#}",
                    operation,
                    version_for_log.as_deref().unwrap_or("unknown version"),
                );
                update_job(
                    &job_state,
                    &job_id,
                    RuntimeJobState::Failed,
                    public_runtime_error(&error),
                    None,
                );
            }
        }
    });

    Ok(Box::new(warp::reply::with_status(
        warp::reply::json(&serde_json::json!({ "job_id": id, "state": "queued" })),
        StatusCode::ACCEPTED,
    )))
}

fn manager(state: &RuntimeApiState) -> Result<Arc<RapidMlxRuntimeManager>, &'static str> {
    state
        .manager
        .as_ref()
        .cloned()
        .map_err(|_| "Managed Rapid-MLX storage is unavailable")
}

async fn published_releases(state: &RuntimeApiState) -> anyhow::Result<Vec<PublishedRelease>> {
    {
        let cache = state.releases.lock().await;
        if let Some((updated, releases)) = cache.as_ref()
            && updated.elapsed() < RELEASE_CACHE_TTL
        {
            return Ok(releases.clone());
        }
    }

    let body = fetch_bounded_release_body(&state.client, RELEASES_URL).await?;
    let releases = decode_published_releases(&body)?;
    *state.releases.lock().await = Some((Instant::now(), releases.clone()));
    Ok(releases)
}

async fn fetch_bounded_release_body(
    client: &reqwest::Client,
    url: &str,
) -> anyhow::Result<Vec<u8>> {
    let response = client.get(url).send().await?;
    if !response.status().is_success() {
        anyhow::bail!("GitHub release metadata returned a non-success status");
    }
    let mut stream = response.bytes_stream();
    let mut body = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if body.len().saturating_add(chunk.len()) > MAX_RELEASE_RESPONSE_BYTES {
            anyhow::bail!("GitHub release metadata exceeded its response bound");
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

fn decode_published_releases(body: &[u8]) -> anyhow::Result<Vec<PublishedRelease>> {
    let raw: Vec<GithubRelease> = serde_json::from_slice(body)?;
    Ok(decode_github_releases(raw))
}

fn decode_github_releases(raw: Vec<GithubRelease>) -> Vec<PublishedRelease> {
    raw.into_iter()
        .filter(|item| !item.draft)
        .filter_map(|item| {
            let version = item.tag_name.strip_prefix('v')?.to_string();
            let channel = if item.prerelease {
                ManagedReleaseChannel::Prerelease
            } else {
                ManagedReleaseChannel::Stable
            };
            RapidMlxRuntimeManager::validate_published_version(&version, channel).ok()?;
            let body = if item.body.trim().is_empty() {
                None
            } else {
                Some(item.body)
            };
            Some(PublishedRelease {
                version,
                tag: item.tag_name,
                channel,
                published_at: item.published_at,
                release_notes: body,
            })
        })
        .collect()
}

async fn select_published_release(
    state: &RuntimeApiState,
    version: &str,
    channel: ManagedReleaseChannel,
    extras: Vec<ManagedRuntimeExtra>,
) -> anyhow::Result<ManagedReleaseSelection> {
    RapidMlxRuntimeManager::validate_published_version(version, channel)?;
    if let Ok(selection) = select_release_from_metadata(
        published_releases(state).await?,
        version,
        channel,
        extras.clone(),
    ) {
        return Ok(selection);
    }

    let url = format!("{RELEASE_BY_TAG_URL}/v{version}");
    let body = fetch_bounded_release_body(&state.client, &url).await?;
    let release: GithubRelease = serde_json::from_slice(&body)?;
    select_release_from_metadata(
        decode_github_releases(vec![release]),
        version,
        channel,
        extras,
    )
}

fn select_release_from_metadata(
    releases: Vec<PublishedRelease>,
    version: &str,
    channel: ManagedReleaseChannel,
    extras: Vec<ManagedRuntimeExtra>,
) -> anyhow::Result<ManagedReleaseSelection> {
    let release = releases
        .into_iter()
        .find(|release| release.version == version && release.channel == channel)
        .ok_or_else(|| anyhow::anyhow!("Release was not found"))?;
    ManagedReleaseSelection::from_published_release_with_extras(
        release.version,
        release.channel,
        extras,
    )
}

fn random_job_id() -> anyhow::Result<String> {
    let mut bytes = [0_u8; 16];
    SysRng.try_fill_bytes(&mut bytes)?;
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}

fn try_insert_job(state: &RuntimeApiState, snapshot: RuntimeJobSnapshot) -> bool {
    if let Ok(mut jobs) = state.jobs.lock() {
        if jobs.entries.values().any(|job| {
            matches!(
                job.state,
                RuntimeJobState::Queued | RuntimeJobState::Running
            )
        }) {
            return false;
        }
        while jobs.order.len() >= MAX_RETAINED_JOBS {
            if let Some(oldest) = jobs.order.pop_front() {
                jobs.entries.remove(&oldest);
            }
        }
        jobs.order.push_back(snapshot.id.clone());
        jobs.entries.insert(snapshot.id.clone(), snapshot);
        true
    } else {
        false
    }
}

fn update_job(
    state: &RuntimeApiState,
    id: &str,
    job_state: RuntimeJobState,
    message: impl Into<String>,
    result: Option<RuntimeMutationResult>,
) {
    if let Ok(mut jobs) = state.jobs.lock()
        && let Some(job) = jobs.entries.get_mut(id)
    {
        job.state = job_state;
        job.message = message.into();
        job.result = result.map(Into::into);
    }
}

fn job_list(state: &RuntimeApiState) -> Vec<RuntimeJobSnapshot> {
    state
        .jobs
        .lock()
        .map(|jobs| {
            jobs.order
                .iter()
                .rev()
                .filter_map(|id| jobs.entries.get(id).cloned())
                .collect()
        })
        .unwrap_or_default()
}

fn public_runtime_error(error: &anyhow::Error) -> &'static str {
    let text = error.to_string();
    if text.contains("already in progress") {
        "Another managed Rapid-MLX runtime operation is already in progress"
    } else if text.contains("require macOS on Apple Silicon") {
        "Managed Rapid-MLX runtime changes require Apple Silicon macOS"
    } else if text.contains("No previous known-good") {
        "No previous known-good Rapid-MLX runtime is available"
    } else if text.contains("No active managed") {
        "No active managed Rapid-MLX runtime is available"
    } else {
        "Managed Rapid-MLX validation failed safely; the active runtime was not changed"
    }
}

fn json_error(status: StatusCode, message: impl Into<String>) -> ApiReply {
    Box::new(warp::reply::with_status(
        warp::reply::json(&serde_json::json!({ "ok": false, "error": message.into() })),
        status,
    ))
}

fn profile_route(ctx: ApiCtx, state: RuntimeApiState) -> ApiRoute {
    let config = ctx.config;
    warp::path!("api" / "rapid-mlx" / "models" / String / "profile")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |model_id: String, auth: Option<String>| {
            let config = config.clone();
            let state = state.clone();
            async move {
                if !check_api_token(&auth, &config) {
                    return Ok(unauthorized_api_token());
                }
                if crate::inference::rapid_mlx::ensure_local_platform_supported().is_err() {
                    return Ok(json_error(
                        StatusCode::BAD_REQUEST,
                        "Model profile queries require Apple Silicon macOS",
                    ));
                }
                let manager_result = manager(&state);
                let mut active_info: Option<(std::path::PathBuf, ManagedReleaseChannel)> = None;
                if let Ok(manager) = &manager_result {
                    let manager = manager.clone();
                    if let Ok(Ok(status)) =
                        tokio::task::spawn_blocking(move || manager.status()).await
                        && let Some(active) = status.active
                    {
                        active_info = Some((active.executable_path, active.release_channel));
                    }
                }
                let binary_path = active_info.as_ref().map(|(p, _)| p.as_path());
                let Ok((binary, source)) = Discovery::resolve_binary(None, binary_path).await
                else {
                    return Ok(json_error(
                        StatusCode::NOT_FOUND,
                        "Rapid-MLX binary not found. Run rapid-mlx doctor or install via Settings.",
                    ));
                };
                let allow_prerelease = active_info
                    .as_ref()
                    .is_some_and(|(p, c)| p == &binary && *c == ManagedReleaseChannel::Prerelease);
                if source == crate::inference::rapid_mlx::runtime::RuntimeSource::Managed {
                    if allow_prerelease {
                        if compatibility::probe_published_managed_release(&binary, allow_prerelease)
                            .await
                            .is_err()
                        {
                            return Ok(json_error(
                                StatusCode::SERVICE_UNAVAILABLE,
                                "Rapid-MLX runtime probe failed",
                            ));
                        }
                    } else if compatibility::probe(&binary, source).await.is_err() {
                        return Ok(json_error(
                            StatusCode::SERVICE_UNAVAILABLE,
                            "Rapid-MLX runtime probe failed",
                        ));
                    }
                } else if compatibility::probe(&binary, source).await.is_err() {
                    return Ok(json_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Rapid-MLX runtime probe failed",
                    ));
                }
                match info_query::fetch_model_profile(&binary, &model_id).await {
                    Ok(Some(profile)) => Ok::<ApiReply, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({ "ok": true, "profile": profile })),
                    )),
                    Ok(None) => Ok(json_error(
                        StatusCode::NOT_FOUND,
                        format!(
                            "Model '{}' not recognized by this Rapid-MLX installation",
                            model_id
                        ),
                    )),
                    Err(error) => {
                        let msg = error.to_string();
                        if msg.contains("timed out") {
                            Ok(json_error(
                                StatusCode::REQUEST_TIMEOUT,
                                "Rapid-MLX info query timed out",
                            ))
                        } else {
                            Ok(json_error(
                                StatusCode::INTERNAL_SERVER_ERROR,
                                "Rapid-MLX info query failed",
                            ))
                        }
                    }
                }
            }
        })
        .boxed()
}

fn unified_profile_route(ctx: ApiCtx) -> ApiRoute {
    let config = ctx.config;
    warp::path!("api" / "rapid-mlx" / "models" / String / "unified-profile")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |model_id: String, auth: Option<String>| {
            let config = config.clone();
            async move {
                if !check_api_token(&auth, &config) {
                    return Ok(unauthorized_api_token());
                }

                if model_id.is_empty() {
                    return Ok(json_error(StatusCode::BAD_REQUEST, "Model ID is required"));
                }
                if model_id.contains("..") {
                    return Ok(json_error(
                        StatusCode::BAD_REQUEST,
                        "Model ID contains invalid path traversal",
                    ));
                }

                let unified =
                    match crate::inference::rapid_mlx::build_unified_profile(&model_id).await {
                        Ok(profile) => profile,
                        Err(error) => {
                            let msg = error.to_string();
                            return Ok(json_error(
                                if msg.contains("timed out") || msg.contains("timeout") {
                                    StatusCode::REQUEST_TIMEOUT
                                } else {
                                    StatusCode::INTERNAL_SERVER_ERROR
                                },
                                format!(
                                    "Unified profile build failed: {}",
                                    msg.chars().take(200).collect::<String>()
                                ),
                            ));
                        }
                    };

                Ok::<ApiReply, warp::Rejection>(Box::new(warp::reply::json(&serde_json::json!({
                    "ok": true,
                    "profile": unified,
                }))))
            }
        })
        .boxed()
}

fn doctor_route(ctx: ApiCtx, _state: RuntimeApiState) -> ApiRoute {
    let config = ctx.config;
    warp::path!("api" / "rapid-mlx" / "doctor")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let config = config.clone();
            async move {
                if !check_api_token(&auth, &config) {
                    return Ok(unauthorized_api_token());
                }
                if crate::inference::rapid_mlx::ensure_local_platform_supported().is_err() {
                    return Ok(json_error(
                        StatusCode::BAD_REQUEST,
                        "rapid-mlx doctor requires Apple Silicon macOS",
                    ));
                }
                let Ok((binary, _)) = Discovery::resolve_binary(None, None).await else {
                    return Ok(json_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "rapid-mlx binary not found on PATH",
                    ));
                };

                let version = run_rapid_mlx_version(&binary).await;
                if version.is_err() {
                    return Ok(json_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "rapid-mlx version check failed",
                    ));
                }
                let version_str = version.unwrap_or_default();

                // No `--json` output exists on any `rapid-mlx` subcommand, so
                // `parse_doctor_output` scrapes human box-drawing/glyph text whose
                // layout is only guaranteed on trusted versions. Below the trusted
                // minor, degrade to raw-output-only with no structured findings
                // rather than risk misparsing an unknown layout.
                let version_trusted = match info_query::cached_version(&binary).await {
                    Ok(Some((_, minor))) => minor >= info_query::MIN_TRUSTED_MINOR,
                    _ => false,
                };

                let doctor_output = run_rapid_mlx_doctor(&binary).await;
                match doctor_output {
                    Ok(output) => {
                        let findings = if version_trusted {
                            parse_doctor_output(&output)
                        } else {
                            Vec::new()
                        };
                        Ok::<ApiReply, warp::Rejection>(Box::new(warp::reply::json(
                            &serde_json::json!({
                                "ok": true,
                                "version": version_str,
                                "version_trusted": version_trusted,
                                "findings": findings,
                                "raw_output": output
                            }),
                        )))
                    }
                    Err(_) => Ok(json_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "rapid-mlx doctor command failed",
                    )),
                }
            }
        })
        .boxed()
}

/// Preset-flag advisor: diffs the active session's model `rapid-mlx info` profile
/// against the active preset's Rapid-MLX launch flags and emits `DoctorFinding`s
/// with `fix: Some(FixAction::…)` where the preset is missing a flag the model's
/// declared capabilities imply it needs. Findings from this route flow into the
/// same diagnostics panel as `doctor_route`'s findings (which always keep
/// `fix: None`) via the frontend's `loadDoctorFindings`.
fn flag_advisor_route(ctx: ApiCtx) -> ApiRoute {
    let config = ctx.config;
    let state = ctx.state;
    warp::path!("api" / "rapid-mlx" / "flag-advisor")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let config = config.clone();
            let state = state.clone();
            async move {
                if !check_api_token(&auth, &config) {
                    return Ok(unauthorized_api_token());
                }

                fn empty_findings() -> ApiReply {
                    Box::new(warp::reply::json(&serde_json::json!({
                        "ok": true,
                        "findings": Vec::<DoctorFinding>::new()
                    })))
                }

                let active_session_id = state.active_session_id.lock().unwrap().clone();
                if active_session_id.is_empty() {
                    return Ok::<ApiReply, warp::Rejection>(empty_findings());
                }

                let preset_id = {
                    let sessions = state.sessions.lock().unwrap();
                    sessions
                        .iter()
                        .find(|s| s.id == active_session_id)
                        .and_then(|s| {
                            if s.preset_id.is_empty() {
                                None
                            } else {
                                Some(s.preset_id.clone())
                            }
                        })
                };
                let Some(preset_id) = preset_id else {
                    return Ok(empty_findings());
                };

                let rapid_config = {
                    let presets = state.presets.lock().unwrap();
                    presets
                        .iter()
                        .find(|p| p.id == preset_id)
                        .and_then(|p| p.rapid_mlx.clone())
                };
                let Some(rapid_config) = rapid_config else {
                    return Ok(empty_findings());
                };

                if crate::inference::rapid_mlx::ensure_local_platform_supported().is_err() {
                    return Ok(empty_findings());
                }
                let Ok((binary, _)) = Discovery::resolve_binary(None, None).await else {
                    return Ok(empty_findings());
                };
                let Some(model_id) = model_id_for_info(&rapid_config) else {
                    return Ok(empty_findings());
                };

                let profile = match info_query::fetch_model_profile(&binary, &model_id).await {
                    Ok(Some(profile)) => profile,
                    _ => return Ok(empty_findings()),
                };

                let snapshot = ExecutableIdentity::from_path(&binary)
                    .ok()
                    .and_then(|identity| capabilities::cached_snapshot(&identity));

                let findings = build_flag_advisor_findings(&profile, &rapid_config, &snapshot);
                Ok::<ApiReply, warp::Rejection>(Box::new(warp::reply::json(&serde_json::json!({
                    "ok": true,
                    "findings": findings
                }))))
            }
        })
        .boxed()
}

/// Extract the identifier to pass to `rapid-mlx info <id>` from a preset's
/// Rapid-MLX model source. Only sources `rapid-mlx info` can resolve (aliases,
/// HuggingFace repos) are supported; local directory/GGUF-file sources have no
/// equivalent `info` lookup and are skipped (advisor degrades to no findings).
fn model_id_for_info(config: &RapidMlxConfig) -> Option<String> {
    match &config.model_source {
        Some(RapidMlxModelSource::Alias { value }) => Some(value.clone()),
        Some(RapidMlxModelSource::HuggingFaceRepo { repo_id, .. }) => Some(repo_id.clone()),
        Some(RapidMlxModelSource::AuthoritativeSafetensors { source, .. }) => match source {
            AuthoritativeSafetensorsSource::HuggingFaceRepo { repo_id, .. } => {
                Some(repo_id.clone())
            }
            AuthoritativeSafetensorsSource::LocalDirectory { .. } => None,
        },
        Some(
            RapidMlxModelSource::MlxDirectory { .. }
            | RapidMlxModelSource::GgufFile { .. }
            | RapidMlxModelSource::Unknown { .. },
        ) => None,
        None if !config.model_path.is_empty() => Some(config.model_path.clone()),
        None => None,
    }
}

/// Pure diff between a model's `rapid-mlx info` profile and the active preset's
/// Rapid-MLX diagnostic flags. Kept free of I/O so it is directly unit-testable.
fn build_flag_advisor_findings(
    profile: &info_query::ModelProfile,
    config: &RapidMlxConfig,
    snapshot: &Option<CapabilitySnapshot>,
) -> Vec<DoctorFinding> {
    let mut findings = Vec::new();

    let tool_format = profile
        .tool_format
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());

    if let Some(tool_format) = tool_format {
        if config.tool_call_parser.is_none() {
            findings.push(DoctorFinding {
                finding_type: DoctorFindingType::Preset,
                severity: DoctorSeverity::Warning,
                message: format!(
                    "Model declares tool format '{tool_format}' but the active preset does not pass --tool-call-parser"
                ),
                section: "Preset Flags".to_string(),
                fix: Some(FixAction::AddToolCallParser),
            });
        }
        if !config.auto_tool_choice {
            findings.push(DoctorFinding {
                finding_type: DoctorFindingType::Preset,
                severity: DoctorSeverity::Warning,
                message: format!(
                    "Model declares tool format '{tool_format}' but the active preset does not pass --enable-auto-tool-choice"
                ),
                section: "Preset Flags".to_string(),
                fix: Some(FixAction::EnableAutoToolChoice),
            });
        }
    }

    let has_reasoning_parser = profile
        .reasoning_parser
        .as_deref()
        .map(str::trim)
        .is_some_and(|value| !value.is_empty());

    // The preset has explicitly opted out of thinking by default
    // (`enable_thinking: Some(false)`) but the launch args don't pass
    // `--no-thinking`, so a reasoning-capable model may still emit thinking
    // tokens despite the preset's stated intent.
    if has_reasoning_parser && config.enable_thinking == Some(false) && !config.no_thinking {
        findings.push(DoctorFinding {
            finding_type: DoctorFindingType::Preset,
            severity: DoctorSeverity::Warning,
            message:
                "Preset disables thinking by default but does not pass --no-thinking; the model may still emit reasoning tokens"
                    .to_string(),
            section: "Preset Flags".to_string(),
            fix: Some(FixAction::AddNoThinking),
        });
    }

    // Check sampling defaults against capability snapshot
    if let Some(snapshot) = snapshot {
        // f64 fields
        for (flag, configured, supported) in [
            (
                "--default-temperature",
                &config.default_temperature,
                &snapshot.sampling_defaults.temperature,
            ),
            (
                "--default-top-p",
                &config.default_top_p,
                &snapshot.sampling_defaults.top_p,
            ),
            (
                "--default-min-p",
                &config.default_min_p,
                &snapshot.sampling_defaults.min_p,
            ),
            (
                "--default-repetition-penalty",
                &config.default_repetition_penalty,
                &snapshot.sampling_defaults.repetition_penalty,
            ),
            (
                "--default-presence-penalty",
                &config.default_presence_penalty,
                &snapshot.sampling_defaults.presence_penalty,
            ),
            (
                "--default-frequency-penalty",
                &config.default_frequency_penalty,
                &snapshot.sampling_defaults.frequency_penalty,
            ),
        ] {
            if configured.is_some() && *supported == capabilities::DefaultFieldState::Unsupported {
                findings.push(DoctorFinding {
                    finding_type: DoctorFindingType::Preset,
                    severity: DoctorSeverity::Warning,
                    message: format!("Sampling default '{}' is configured but not supported by this Rapid-MLX version", flag),
                    section: "Sampling Defaults".to_string(),
                    fix: None,
                });
            }
        }
        // u64 fields
        for (flag, configured, supported) in [
            (
                "--default-top-k",
                &config.default_top_k,
                &snapshot.sampling_defaults.top_k,
            ),
            (
                "--max-tokens",
                &config.max_tokens,
                &snapshot.sampling_defaults.max_tokens,
            ),
        ] {
            if configured.is_some() && *supported == capabilities::DefaultFieldState::Unsupported {
                findings.push(DoctorFinding {
                    finding_type: DoctorFindingType::Preset,
                    severity: DoctorSeverity::Warning,
                    message: format!("Sampling default '{}' is configured but not supported by this Rapid-MLX version", flag),
                    section: "Sampling Defaults".to_string(),
                    fix: None,
                });
            }
        }
    }

    findings
}

async fn run_rapid_mlx_version(binary: &std::path::Path) -> Result<String, std::io::Error> {
    let output = tokio::time::timeout(
        Duration::from_secs(10),
        tokio::process::Command::new(binary)
            .arg("--version")
            .output(),
    )
    .await
    .map_err(|_| std::io::Error::new(std::io::ErrorKind::TimedOut, "version timed out"))??;

    let text = String::from_utf8_lossy(&output.stdout);
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("v") && trimmed.split('.').count() >= 3 {
            return Ok(trimmed.to_string());
        }
    }
    Ok(text.lines().next().unwrap_or("").trim().to_string())
}

async fn run_rapid_mlx_doctor(binary: &std::path::Path) -> Result<String, std::io::Error> {
    let output = tokio::time::timeout(
        Duration::from_secs(30),
        tokio::process::Command::new(binary).arg("doctor").output(),
    )
    .await
    .map_err(|_| std::io::Error::new(std::io::ErrorKind::TimedOut, "doctor timed out"))??;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let combined = if stderr.trim().is_empty() {
        stdout
    } else {
        format!("{stdout}\n{stderr}")
    };
    Ok(combined)
}

fn parse_doctor_output(output: &str) -> Vec<DoctorFinding> {
    let mut findings = Vec::new();
    let mut current_section = String::from("general");

    for line in output.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if trimmed.starts_with('◆') {
            // Capture the full multi-word header (e.g. "Optional Packages",
            // "Optional Tools", "Required Packages") — do not truncate to the
            // first word, which collides sections like "Optional Packages"
            // and "Optional Tools" into a single "Optional" bucket.
            current_section = trimmed.trim_start_matches('◆').trim().to_string();
            continue;
        }

        let first_char = trimmed.chars().next();
        let glyph = match first_char {
            Some(g @ ('✓' | '⚠' | '✗' | '×' | '!')) => g,
            _ => continue,
        };

        let severity = DoctorSeverity::from_glyph(glyph);
        let message = trimmed.trim_start_matches(glyph).trim().to_string();

        findings.push(DoctorFinding {
            finding_type: DoctorFindingType::Environment,
            severity,
            message,
            section: current_section.clone(),
            fix: None,
        });
    }

    findings
}

#[cfg(test)]
mod settings_catalog_tests {
    use super::*;
    use crate::inference::rapid_mlx::settings;

    fn snapshot_with(flags: &[&str]) -> CapabilitySnapshot {
        CapabilitySnapshot {
            serve_flags: flags.iter().map(|flag| flag.to_string()).collect(),
            ..Default::default()
        }
    }

    /// `build_effective_policy` writes every setting key by hand. That hand-written list is
    /// how the catalog went stale the first time: a setting added to one and not the other
    /// silently stops appearing in the policy snapshot the UI reads. Neither is derived from
    /// the other, so pin them together here.
    #[test]
    fn effective_policy_snapshot_covers_exactly_the_catalog() {
        let config = RapidMlxConfig::default();
        let policy = build_effective_policy(&config);
        let policy_keys: std::collections::BTreeSet<&str> = policy
            .as_object()
            .expect("effective policy is an object")
            .keys()
            .map(String::as_str)
            .collect();
        let catalog_ids: std::collections::BTreeSet<&str> = settings::all_settings()
            .iter()
            .map(|setting| setting.id())
            .collect();

        // One known, deliberate disagreement. The catalog models prefix caching as a single
        // `prefix_cache_policy`; the shipped API and every frontend consumer instead use the
        // three raw config fields below. Reconciling them is Phase 7 UI work, not something
        // to change underneath presets.js, setup-view.js, spawn-wizard.js, and
        // vram-estimate.js, none of which read `prefix_cache_policy` at all. Listed as an
        // exception so the pairing still catches *new* drift.
        const CATALOG_ONLY: &[&str] = &["prefix_cache_policy"];
        const SNAPSHOT_ONLY: &[&str] = &[
            "prefix_cache_enabled",
            "retained_cache_mib",
            "disk_checkpoint_interval",
            // Compound, discriminated product configuration rather than a scalar catalog
            // setting. It is validated by RapidMlxSpeculativeConfig and rendered as one
            // exact vLLM-style JSON object in the command builder.
            "speculative_config",
        ];

        let missing: Vec<_> = catalog_ids
            .difference(&policy_keys)
            .filter(|id| !CATALOG_ONLY.contains(*id))
            .collect();
        let extra: Vec<_> = policy_keys
            .difference(&catalog_ids)
            .filter(|id| !SNAPSHOT_ONLY.contains(*id))
            .collect();
        assert!(
            missing.is_empty() && extra.is_empty(),
            "effective-policy snapshot has drifted from the catalog.\n\
             in catalog, absent from snapshot: {missing:?}\n\
             in snapshot, absent from catalog: {extra:?}"
        );
    }

    #[test]
    fn catalog_gates_settings_on_the_snapshot_and_says_why() {
        let without = snapshot_with(&[]);
        let kv = settings::all_settings()
            .iter()
            .find(|setting| setting.id() == "kv_cache_dtype")
            .expect("kv_cache_dtype is in the catalog");

        assert!(!kv.capability(&without));
        assert_eq!(
            kv.unsupported_reason(&without).as_deref(),
            Some("Current runtime does not support --kv-cache-dtype"),
        );

        let with = snapshot_with(&["--kv-cache-dtype"]);
        assert!(kv.capability(&with));
        assert_eq!(
            kv.unsupported_reason(&with),
            None,
            "a supported setting must carry no reason, or the UI shows a stale excuse"
        );
    }

    /// The gap that mattered: an unsupported setting must not silently launch. It resolves to
    /// null and contributes no argv, rather than erroring.
    #[test]
    fn unsupported_settings_resolve_away_instead_of_reaching_argv() {
        let without = snapshot_with(&[]);
        let kv = settings::all_settings()
            .iter()
            .find(|setting| setting.id() == "kv_cache_dtype")
            .unwrap();
        let requested = serde_json::json!({"effective": "int8"});

        assert!(
            kv.validate(&requested, &settings::ValidationContext::default())
                .is_ok(),
            "validation is capability-free by design; gating happens in effective_policy"
        );
        let resolved = kv.effective_policy(&requested, &without);
        assert!(resolved.is_null());
        assert!(kv.to_cli_args(&resolved).is_empty());
    }

    #[test]
    fn mutual_exclusions_are_reported_for_conflicting_pairs() {
        let mut values = std::collections::BTreeMap::new();
        values.insert("reasoning_mode", serde_json::json!("on"));
        values.insert("sampling_mode", serde_json::json!("model_default"));
        let error = settings::check_mutual_exclusions(&values)
            .expect_err("reasoning_mode=on with sampling_mode=model_default must conflict");
        assert_eq!(error.code, "mutual_exclusion");
    }

    #[test]
    fn every_catalog_default_validates_against_its_own_setting() {
        let context = settings::ValidationContext::default();
        for setting in settings::all_settings() {
            let default = setting.default_value();
            assert!(
                setting.validate(&default, &context).is_ok(),
                "{} ships a default its own validator rejects",
                setting.id()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::rapid_mlx::CacheMode;

    fn test_state() -> RuntimeApiState {
        RuntimeApiState {
            manager: Err("unused".into()),
            releases: Arc::new(tokio::sync::Mutex::new(None)),
            jobs: Arc::new(Mutex::new(RuntimeJobs::default())),
            model_downloads: Arc::new(Mutex::new(BTreeMap::new())),
            changelog_cache: Arc::new(changelog::ChangelogCacheManager::new()),
            client: reqwest::Client::new(),
        }
    }

    fn queued_job(id: &str) -> RuntimeJobSnapshot {
        RuntimeJobSnapshot {
            id: id.into(),
            operation: RuntimeOperation::Install,
            state: RuntimeJobState::Queued,
            message: String::new(),
            version: Some("0.10.10".into()),
            result: None,
        }
    }

    #[test]
    fn runtime_errors_never_disclose_worker_paths() {
        let error = anyhow::anyhow!(
            "uv failed at /Users/person/.config/llama-monitor/runtimes/rapid-mlx/secret"
        );
        assert!(!public_runtime_error(&error).contains("/Users/"));
    }

    #[test]
    fn release_request_rejects_unknown_fields() {
        assert!(
            serde_json::from_str::<RuntimeMutationRequest>(
                r#"{"version":"0.10.10","channel":"stable","confirm":"x","extra":true}"#,
            )
            .is_err()
        );
    }

    #[test]
    fn release_request_defaults_to_guided_and_vision_profile() {
        let request: RuntimeMutationRequest =
            serde_json::from_str(r#"{"version":"0.10.10","channel":"stable","confirm":"x"}"#)
                .unwrap();
        assert_eq!(request.extras, default_runtime_extras());
    }

    #[test]
    fn development_request_accepts_pinned_resolution_and_defaults_extras() {
        let request: DevelopmentSourceRequest = serde_json::from_str(
            r#"{"repository":"nmorgowicz/Rapid-MLX","reference":"main","resolved_commit":"0123456789abcdef0123456789abcdef01234567","confirm":"x"}"#,
        )
        .unwrap();
        assert_eq!(
            request.resolved_commit.as_deref(),
            Some("0123456789abcdef0123456789abcdef01234567")
        );
        assert_eq!(request.extras, default_runtime_extras());
    }

    #[test]
    fn development_install_requires_the_resolved_commit() {
        let request: DevelopmentSourceRequest = serde_json::from_str(
            r#"{"repository":"nmorgowicz/Rapid-MLX","reference":"main","confirm":"INSTALL_RAPID_MLX_DEVELOPMENT"}"#,
        )
        .unwrap();
        let error = development_selection_from_request(&request).unwrap_err();
        assert!(error.to_string().contains("Resolve the development source"));
    }

    #[test]
    fn release_request_rejects_duplicate_extras() {
        let request: RuntimeMutationRequest = serde_json::from_str(
            r#"{"version":"0.10.10","channel":"stable","extras":["vision","vision"],"confirm":"x"}"#,
        )
        .unwrap();
        assert!(
            ManagedReleaseSelection::from_published_release_with_extras(
                request.version,
                request.channel,
                request.extras,
            )
            .is_err()
        );
    }

    #[test]
    fn release_selection_requires_exact_published_version_and_channel() {
        let releases = vec![PublishedRelease {
            version: "0.10.10".into(),
            tag: "v0.10.10".into(),
            channel: ManagedReleaseChannel::Stable,
            published_at: "2026-07-16T00:00:00Z".into(),
            release_notes: None,
        }];
        assert!(
            select_release_from_metadata(
                releases.clone(),
                "0.10.10",
                ManagedReleaseChannel::Stable,
                Vec::new(),
            )
            .is_ok()
        );
        assert!(
            select_release_from_metadata(
                releases.clone(),
                "0.10.11",
                ManagedReleaseChannel::Stable,
                Vec::new(),
            )
            .is_err()
        );
        assert!(
            select_release_from_metadata(
                releases,
                "0.10.10",
                ManagedReleaseChannel::Prerelease,
                Vec::new(),
            )
            .is_err()
        );
    }

    #[test]
    fn release_discovery_filters_drafts_and_versions_below_floor() {
        let releases = decode_published_releases(
            br#"[
                {"tag_name":"v0.10.10","draft":false,"prerelease":false},
                {"tag_name":"v0.10.8","draft":false,"prerelease":false},
                {"tag_name":"v0.10.11rc1","draft":false,"prerelease":true},
                {"tag_name":"v0.10.12","draft":true,"prerelease":false}
            ]"#,
        )
        .unwrap();
        assert_eq!(releases.len(), 2);
        assert_eq!(releases[0].version, "0.10.10");
        assert_eq!(releases[1].version, "0.10.11rc1");
    }

    #[test]
    fn api_job_admission_is_atomic() {
        let state = test_state();
        assert!(try_insert_job(&state, queued_job("first")));
        assert!(!try_insert_job(&state, queued_job("second")));
        update_job(&state, "first", RuntimeJobState::Complete, "done", None);
        assert!(try_insert_job(&state, queued_job("second")));
    }

    #[test]
    fn public_runtime_payloads_never_disclose_executable_paths() {
        let internal = RuntimeInventoryEntry {
            environment_id: "rapid-mlx-0.10.10-test".into(),
            version: "0.10.10".into(),
            source_kind: ManagedRuntimeSourceKind::Release,
            source_repository: None,
            source_ref: None,
            source_commit: None,
            release_channel: ManagedReleaseChannel::Stable,
            extras: Vec::new(),
            executable_path: "/Users/person/.config/llama-monitor/private/rapid-mlx".into(),
            active: true,
            rollback_candidate: false,
            complete: true,
            resolved_receipt: None,
            last_probe_result: None,
        };
        let public = PublicRuntimeInventoryEntry::from(internal);
        let json = serde_json::to_string(&public).unwrap();
        assert!(!json.contains("/Users/person"));
        assert!(!json.contains("executable"));
    }

    #[test]
    fn managed_prerelease_probe_policy_follows_active_manifest_channel() {
        let binary = std::path::PathBuf::from("/managed/rapid-mlx");
        assert!(managed_prerelease_allowed(
            Some(&(binary.clone(), ManagedReleaseChannel::Prerelease)),
            &binary,
        ));
        assert!(!managed_prerelease_allowed(
            Some(&(binary.clone(), ManagedReleaseChannel::Stable)),
            &binary,
        ));
        assert!(!managed_prerelease_allowed(
            Some(&(
                std::path::PathBuf::from("/other/rapid-mlx"),
                ManagedReleaseChannel::Prerelease,
            )),
            &binary,
        ));
    }

    /// Real `rapid-mlx doctor` output captured on Apple Silicon (M5 Max,
    /// rapid-mlx 0.10.12) via `rapid-mlx doctor`. Not hand-mirrored — used
    /// verbatim (box-drawing header + `◆` section markers + glyphs) so the
    /// contract test exercises the real layout, per the project's fixture rule.
    const REAL_DOCTOR_OUTPUT: &str = "\
┌─────────────────────────────────────────────────────────┐
│                    🩺 Rapid-MLX Doctor                   │
└─────────────────────────────────────────────────────────┘

◆ System
  ✓ Apple Silicon (Apple M5 Max, 64 GB)
  ✓ macOS 26.5.1 (Darwin 25.5.0)
  ✓ Free disk: 514 GB
  ⚠ HF cache size: 123 GB (consider `rapid-mlx rm` for unused models)

◆ Python
  ✓ Python 3.11.15
  ✓ Install location: virtualenv (/Users/nick/.local/share/uv/python/cpython-3.11.15-macos-aarch64-none/bin/python3.11)

◆ Required Packages
  ✓ mlx 0.32.0
  ✓ mlx-lm 0.31.3
  ✓ transformers 5.12.1
  ✓ fastapi 0.139.2
  ✓ uvicorn 0.51.0
  ✓ rapid-mlx 0.10.12

◆ Optional Packages
  ⚠ mlx-vlm (vision extras) not installed (`pip install 'rapid-mlx[vision]'`)
  ⚠ mlx-audio (audio extras) not installed (`pip install 'rapid-mlx[audio]'`)
  ⚠ mlx-embeddings (embeddings extras) not installed (`pip install 'rapid-mlx[embeddings]'`)
  ⚠ mlx-vlm 0.5.0+ (dflash extras) not installed or too old (current: not installed, need: 0.5.0+)

◆ HuggingFace Cache
  ✓ /Users/nick/.cache/huggingface/hub exists, writable
  ✓ Free space: 514 GB

◆ Network
  ✓ huggingface.co reachable

◆ Shell Integration
  ✓ rapid-mlx in $PATH (/Users/nick/.local/bin/rapid-mlx)
  ⚠ argcomplete not activated — add `eval \"$(register-python-argcomplete rapid-mlx)\"` to your shell rc

◆ Optional Tools
  ✓ codex CLI (/opt/homebrew/bin/codex)

────────────────────────────────────────
Summary: 16 ok, 6 warnings, 0 issues
Run with `--verbose` for details on each check.
";

    #[test]
    fn parse_doctor_output_captures_full_multi_word_section_names() {
        let findings = parse_doctor_output(REAL_DOCTOR_OUTPUT);

        let sections: std::collections::BTreeSet<&str> =
            findings.iter().map(|f| f.section.as_str()).collect();

        // Real multi-word headers must survive intact — the historical bug
        // truncated everything after the first word, colliding "Optional
        // Packages" and "Optional Tools" into a single "Optional" bucket.
        assert!(sections.contains("System"));
        assert!(sections.contains("Python"));
        assert!(sections.contains("Required Packages"));
        assert!(sections.contains("Optional Packages"));
        assert!(sections.contains("HuggingFace Cache"));
        assert!(sections.contains("Network"));
        assert!(sections.contains("Shell Integration"));
        assert!(sections.contains("Optional Tools"));

        // No truncated collision bucket should exist.
        assert!(!sections.contains("Optional"));
        assert!(!sections.contains("Required"));
        assert!(!sections.contains("Shell"));

        // "Optional Packages" and "Optional Tools" findings must stay in their
        // own distinct sections rather than merging.
        let optional_packages_count = findings
            .iter()
            .filter(|f| f.section == "Optional Packages")
            .count();
        let optional_tools_count = findings
            .iter()
            .filter(|f| f.section == "Optional Tools")
            .count();
        assert_eq!(optional_packages_count, 4);
        assert_eq!(optional_tools_count, 1);
    }

    #[test]
    fn parse_doctor_output_maps_glyphs_to_severity_and_rollup_matches_summary_line() {
        let findings = parse_doctor_output(REAL_DOCTOR_OUTPUT);

        let ok_count = findings
            .iter()
            .filter(|f| f.severity == DoctorSeverity::Ok)
            .count();
        let warning_count = findings
            .iter()
            .filter(|f| f.severity == DoctorSeverity::Warning)
            .count();
        let issue_count = findings
            .iter()
            .filter(|f| f.severity == DoctorSeverity::Issue)
            .count();

        // Cross-check against the fixture's own "Summary: N ok, M warnings, K issues"
        // rollup line rather than a hand-picked number.
        let summary_line = REAL_DOCTOR_OUTPUT
            .lines()
            .find(|l| l.trim_start().starts_with("Summary:"))
            .expect("fixture must contain a Summary line");
        assert_eq!(summary_line.trim(), "Summary: 16 ok, 6 warnings, 0 issues");

        assert_eq!(ok_count, 16);
        assert_eq!(warning_count, 6);
        assert_eq!(issue_count, 0);

        // All doctor findings are informational only — never fixable.
        assert!(findings.iter().all(|f| f.fix.is_none()));
        assert!(
            findings
                .iter()
                .all(|f| f.finding_type == DoctorFindingType::Environment)
        );
    }

    fn sample_profile_with_tool_format(tool_format: &str) -> info_query::ModelProfile {
        info_query::ModelProfile {
            tool_format: Some(tool_format.to_string()),
            ..Default::default()
        }
    }

    #[test]
    fn flag_advisor_emits_fix_when_preset_missing_tool_call_flags() {
        let profile = sample_profile_with_tool_format("hermes");
        let mut config = RapidMlxConfig {
            model_path: String::new(),
            model_source: None,
            served_model_name: None,
            executable_path: None,
            managed_runtime_path: None,
            host: "127.0.0.1".into(),
            port: 8000,
            log_level: "INFO".into(),
            timeout: None,
            prefix_cache_enabled: false,
            retained_cache_mib: None,
            cache_mode: CacheMode::Custom,
            disk_checkpoint_interval: 0,
            api_key: None,
            enable_thinking: None,
            reasoning_effort: None,
            trust_remote_code_consent: None,
            tool_call_parser: None,
            reasoning_parser: None,
            auto_tool_choice: false,
            no_thinking: false,
            escape_hatch_flags: Vec::new(),
            model_source_view: None,
            kv_cache_dtype: None,
            turboquant_mode: None,
            hybrid_cache_entries: None,
            hybrid_mode: crate::inference::rapid_mlx::RapidMlxHybridMode::Auto,
            pflash_policy: None,
            max_num_seqs: None,
            max_concurrent_requests: None,
            prefill_batch_size: None,
            completion_batch_size: None,
            prefill_step_size: 512,
            reasoning_mode: None,
            speculative_config: None,
            mllm_vision: None,
            embeddings: None,
            gpu_memory_utilization: None,
            sampling_mode: None,
            chat_template_file: None,
            default_temperature: None,
            default_top_p: None,
            default_top_k: None,
            default_min_p: None,
            default_repetition_penalty: None,
            default_presence_penalty: None,
            default_frequency_penalty: None,
            max_tokens: None,
        };

        let findings = build_flag_advisor_findings(&profile, &config, &None);

        assert_eq!(findings.len(), 2);
        assert!(
            findings
                .iter()
                .any(|f| f.fix == Some(FixAction::AddToolCallParser))
        );
        assert!(
            findings
                .iter()
                .any(|f| f.fix == Some(FixAction::EnableAutoToolChoice))
        );
        assert!(
            findings
                .iter()
                .all(|f| f.finding_type == DoctorFindingType::Preset)
        );

        // A requested K8V4 trial must not be reported as effective until a
        // revision-bound qualification receipt is available to the launch path.
        config.turboquant_mode = Some(crate::inference::rapid_mlx::TurboQuantMode::K8V4);
        assert_eq!(
            build_effective_policy(&config)["turboquant_mode"],
            serde_json::json!("none")
        );
    }

    #[test]
    fn flag_advisor_is_silent_when_preset_already_matches_model_profile() {
        let profile = sample_profile_with_tool_format("hermes");
        let config = RapidMlxConfig {
            model_path: String::new(),
            model_source: None,
            served_model_name: None,
            executable_path: None,
            managed_runtime_path: None,
            host: "127.0.0.1".into(),
            port: 8000,
            log_level: "INFO".into(),
            timeout: None,
            prefix_cache_enabled: false,
            retained_cache_mib: None,
            cache_mode: CacheMode::Custom,
            disk_checkpoint_interval: 0,
            api_key: None,
            enable_thinking: None,
            reasoning_effort: None,
            trust_remote_code_consent: None,
            tool_call_parser: Some("openai".to_string()),
            reasoning_parser: None,
            auto_tool_choice: true,
            no_thinking: false,
            escape_hatch_flags: Vec::new(),
            model_source_view: None,
            kv_cache_dtype: None,
            turboquant_mode: None,
            hybrid_cache_entries: None,
            hybrid_mode: crate::inference::rapid_mlx::RapidMlxHybridMode::Auto,
            pflash_policy: None,
            max_num_seqs: None,
            max_concurrent_requests: None,
            prefill_batch_size: None,
            completion_batch_size: None,
            prefill_step_size: 512,
            reasoning_mode: None,
            speculative_config: None,
            mllm_vision: None,
            embeddings: None,
            gpu_memory_utilization: None,
            sampling_mode: None,
            chat_template_file: None,
            default_temperature: None,
            default_top_p: None,
            default_top_k: None,
            default_min_p: None,
            default_repetition_penalty: None,
            default_presence_penalty: None,
            default_frequency_penalty: None,
            max_tokens: None,
        };

        let findings = build_flag_advisor_findings(&profile, &config, &None);

        assert!(findings.is_empty());
    }

    #[test]
    fn flag_advisor_recommends_no_thinking_when_preset_wants_thinking_disabled() {
        let profile = info_query::ModelProfile {
            reasoning_parser: Some("qwen3".to_string()),
            ..Default::default()
        };
        let config = RapidMlxConfig {
            model_path: String::new(),
            model_source: None,
            served_model_name: None,
            executable_path: None,
            managed_runtime_path: None,
            host: "127.0.0.1".into(),
            port: 8000,
            log_level: "INFO".into(),
            timeout: None,
            prefix_cache_enabled: false,
            retained_cache_mib: None,
            cache_mode: CacheMode::Custom,
            disk_checkpoint_interval: 0,
            api_key: None,
            enable_thinking: Some(false),
            reasoning_effort: None,
            trust_remote_code_consent: None,
            tool_call_parser: None,
            reasoning_parser: None,
            auto_tool_choice: false,
            no_thinking: false,
            escape_hatch_flags: Vec::new(),
            model_source_view: None,
            kv_cache_dtype: None,
            turboquant_mode: None,
            hybrid_cache_entries: None,
            hybrid_mode: crate::inference::rapid_mlx::RapidMlxHybridMode::Auto,
            pflash_policy: None,
            max_num_seqs: None,
            max_concurrent_requests: None,
            prefill_batch_size: None,
            completion_batch_size: None,
            prefill_step_size: 512,
            reasoning_mode: None,
            speculative_config: None,
            mllm_vision: None,
            embeddings: None,
            gpu_memory_utilization: None,
            sampling_mode: None,
            chat_template_file: None,
            default_temperature: None,
            default_top_p: None,
            default_top_k: None,
            default_min_p: None,
            default_repetition_penalty: None,
            default_presence_penalty: None,
            default_frequency_penalty: None,
            max_tokens: None,
        };

        let findings = build_flag_advisor_findings(&profile, &config, &None);

        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].fix, Some(FixAction::AddNoThinking));
    }
}

#[cfg(test)]
mod command_preview_parity_tests {
    use crate::inference::rapid_mlx::compatibility::ServeCapabilities;
    use crate::inference::rapid_mlx::model_resolver::ResolvedRapidMlxLaunchModel;
    use crate::inference::rapid_mlx::{
        RapidMlxAdapter, RapidMlxConfig, RapidMlxHybridMode, apply_phase7_adapter_config,
        build_launch_argv,
    };

    /// Every flag the settings below can emit. A missing entry makes `build` fail the
    /// capability check rather than silently drop the flag, so this list is part of the
    /// assertion.
    const ALL_SERVE_FLAGS: &str = "--host --port --served-model-name --timeout --log-level \
        --api-key --tool-call-parser --reasoning-parser --enable-auto-tool-choice \
        --enable-prefix-cache --max-cache-blocks --cache-memory-mb \
        --kv-disk-checkpoint-interval \
        --kv-cache-dtype --kv-cache-turboquant --max-num-seqs --max-concurrent-requests \
        --prefill-batch-size --completion-batch-size --batching-policy --concurrency-policy \
        --prefill-step-size --force-hybrid --no-hybrid \
        --reasoning --speculative --mllm --no-mllm --gpu-memory-utilization \
        --ui --no-ui --path --ui-config --pflash --hybrid-cache-entries \
        --response-cache --disk-checkpoint --endpoint-compatibility \
        --request-safety-policy --sampling-mode --parser-policy --security-policy \
        --default-temperature --default-top-p --default-top-k --default-min-p \
        --default-repetition-penalty --default-presence-penalty \
        --default-frequency-penalty --max-tokens";

    /// Renders argv exactly the way the command-preview handler does.
    fn preview_argv(config: &RapidMlxConfig) -> Vec<String> {
        let adapter = RapidMlxAdapter::for_settings_preview(
            "rapid-mlx".into(),
            ResolvedRapidMlxLaunchModel::validated_alias("model").unwrap(),
            config,
        );
        apply_phase7_adapter_config(build_launch_argv(&adapter).0, &adapter)
            .build(
                "rapid-mlx".into(),
                &ServeCapabilities::from_help(ALL_SERVE_FLAGS),
            )
            .expect("preview argv builds")
            .args
            .iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect()
    }

    fn value_after<'a>(argv: &'a [String], flag: &str) -> Option<&'a str> {
        argv.iter()
            .position(|arg| arg == flag)
            .and_then(|index| argv.get(index + 1))
            .map(String::as_str)
    }

    /// The preview used to carry its own copy of the adapter → argv mapping, and it had
    /// drifted: these settings were configurable, were applied at launch, and were absent
    /// from the command the operator was shown. `--prefill-step-size` is the one that
    /// mattered most, since it is what keeps long-context prefill inside the attention
    /// budget and the launcher emits it on every launch.
    #[test]
    fn command_preview_shows_the_settings_the_launcher_applies() {
        let config = RapidMlxConfig {
            served_model_name: Some("preview-name".into()),
            reasoning_parser: Some("deepseek_r1".into()),
            hybrid_mode: RapidMlxHybridMode::Force,
            // The builder clamps this to 1..=2048 on purpose: a full-attention prefill
            // materializes an O(step × context × heads) score buffer, and the 2026-07-24
            // investigation crashed Metal's single-buffer cap at 32768. 512 is the standing
            // default; 2048 is the top of the supported range, so test the boundary.
            prefill_step_size: 2048,
            default_temperature: Some(0.7),
            default_top_p: Some(0.9),
            default_top_k: Some(40),
            default_min_p: Some(0.05),
            default_repetition_penalty: Some(1.05),
            default_presence_penalty: Some(0.1),
            default_frequency_penalty: Some(0.2),
            max_tokens: Some(4096),
            ..RapidMlxConfig::default()
        };

        let argv = preview_argv(&config);

        assert_eq!(
            value_after(&argv, "--served-model-name"),
            Some("preview-name")
        );
        assert_eq!(
            value_after(&argv, "--reasoning-parser"),
            Some("deepseek_r1")
        );
        assert!(argv.iter().any(|arg| arg == "--force-hybrid"), "{argv:?}");
        assert_eq!(value_after(&argv, "--prefill-step-size"), Some("2048"));
        assert_eq!(value_after(&argv, "--default-temperature"), Some("0.7"));
        assert_eq!(value_after(&argv, "--default-top-p"), Some("0.9"));
        assert_eq!(value_after(&argv, "--default-top-k"), Some("40"));
        assert_eq!(value_after(&argv, "--default-min-p"), Some("0.05"));
        assert_eq!(
            value_after(&argv, "--default-repetition-penalty"),
            Some("1.05")
        );
        assert_eq!(
            value_after(&argv, "--default-presence-penalty"),
            Some("0.1")
        );
        assert_eq!(
            value_after(&argv, "--default-frequency-penalty"),
            Some("0.2")
        );
        assert_eq!(value_after(&argv, "--max-tokens"), Some("4096"));
    }

    /// The mirror-image defect: the preview filled in defaults of its own, so it showed
    /// three flags the launcher never passes. An invented flag is as misleading as a
    /// dropped one.
    #[test]
    fn command_preview_does_not_invent_defaults_the_launcher_omits() {
        let argv = preview_argv(&RapidMlxConfig::default());

        assert!(!argv.iter().any(|arg| arg == "--log-level"), "{argv:?}");
        assert!(!argv.iter().any(|arg| arg == "--timeout"), "{argv:?}");
        assert!(!argv.iter().any(|arg| arg == "--api-key"), "{argv:?}");
    }
}

/// Fingerprint known Qwen3.5/3.8 tiers to their official upstream MTP draft repos.
///
/// Finetune names rarely carry the base family ("Scarlett-Opus-oQ4e-MLX" is a Qwen3.8-27B
/// finetune), so detection reads the trunk's own config.json — architecture, not marketing.
/// Fingerprints are from the upstream registries' checkpoints:
/// Qwen3.8/3.6-27B (hidden 5120, 64 layers, vocab 248320), Qwen3.5-9B (4096/32),
/// Qwen3.5-4B (2560/32).
fn official_mtp_draft_for_config(
    model_type: &str,
    hidden_size: u64,
    num_hidden_layers: u64,
) -> Option<(&'static str, &'static str)> {
    if model_type != "qwen3_5" {
        return None;
    }
    if hidden_size == 5120 && num_hidden_layers == 64 {
        Some((
            "rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX",
            "a bf16 variant (…-MTP-fp16-MLX) also exists",
        ))
    } else if hidden_size == 4096 && num_hidden_layers == 32 {
        Some(("mlx-community/Qwen3.5-9B-MTP-4bit", ""))
    } else if hidden_size == 2560 && num_hidden_layers == 32 {
        Some(("mlx-community/Qwen3.5-4B-MTP-4bit", ""))
    } else {
        None
    }
}

fn mtp_draft_suggestion_route(ctx: ApiCtx) -> ApiRoute {
    let config = ctx.config;
    warp::path!("api" / "rapid-mlx" / "mtp" / "draft-suggestion")
        .and(warp::get())
        .and(warp::query::<std::collections::HashMap<String, String>>())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |query: std::collections::HashMap<String, String>, auth: Option<String>| {
            let config = config.clone();
            async move {
                if !check_api_token(&auth, &config) {
                    return Ok(unauthorized_api_token());
                }
                let trunk = query
                    .get("path")
                    .map(String::as_str)
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if trunk.is_empty() || trunk.contains("..") {
                    return Ok(json_error(
                        StatusCode::BAD_REQUEST,
                        "path is required and must not contain '..'",
                    ));
                }
                let trunk_path = std::path::Path::new(&trunk);
                let config_path = trunk_path.join("config.json");
                let Ok(config_text) = std::fs::read_to_string(&config_path) else {
                    return Ok::<ApiReply, warp::Rejection>(Box::new(warp::reply::json(
                        &serde_json::json!({
                            "ok": true,
                            "suggestion": serde_json::Value::Null,
                            "reason": "no readable config.json at the trunk path"
                        }),
                    )));
                };
                let parsed: serde_json::Value =
                    serde_json::from_str(&config_text).unwrap_or(serde_json::Value::Null);
                let text = if parsed.get("text_config").is_some() {
                    parsed["text_config"].clone()
                } else {
                    parsed.clone()
                };
                let model_type = parsed
                    .get("model_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let hidden_size = text
                    .get("hidden_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let layers = text
                    .get("num_hidden_layers")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let suggestion =
                    official_mtp_draft_for_config(&model_type, hidden_size, layers)
                        .map(|(repo, note)| {
                            serde_json::json!({
                                "repo": repo,
                                "note": note,
                                "basis": "architecture fingerprint",
                                "tier": { "model_type": model_type, "hidden_size": hidden_size, "num_hidden_layers": layers }
                            })
                        });
                Ok::<ApiReply, warp::Rejection>(Box::new(warp::reply::json(&serde_json::json!({
                    "ok": true,
                    "suggestion": suggestion,
                }))))
            }
        })
        .boxed()
}
