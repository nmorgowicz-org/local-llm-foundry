//! `/api/llama-build/*`: build llama.cpp from source inside the foundry
//! (plan §7). Every route requires the API token.
//!
//! - `GET  /status`, `GET /options`, `GET /prereqs`
//! - `POST /prereqs/validate`, `POST /prereqs/fix` (macOS `brew install`)
//! - `POST /start`, `GET /jobs/:id`, `GET /jobs/:id/log`, `POST /jobs/:id/cancel`
//! - `GET /cache`, `DELETE /cache`, `PUT /cache-cap`
//!
//! Conflicts are `409 {owner}`; a build, clean, validation or install never
//! overlaps one that owns the same directories. The handlers only validate and
//! shape JSON: jobs, ownership and caches live in
//! [`crate::llama_build::service::BuildService`]. No request field carries a
//! filesystem path; job ids are checked against the exact id grammar before
//! they touch the disk.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, OnceLock};

use serde::Deserialize;
use serde_json::{Value, json};
use warp::Filter;
use warp::http::StatusCode;

use super::llama_binary::managed_bin_dir;
use super::{ApiCtx, ApiReply, ApiRoute, box_reply, check_api_token, unauthorized_api_token};
use crate::config::AppConfig;
use crate::llama_build::cache::{
    CleanError, CleanOptions, Conflict, LlamaBuildConfig, MAX_CCACHE_BYTES, MIN_CCACHE_BYTES,
    validate_ccache_cap,
};
use crate::llama_build::gpu::{CUDA_ARCH_BY_COMPUTE_CAP, validate_cuda_architectures};
use crate::llama_build::job::{JobRecord, JobStatus};
use crate::llama_build::paths::BuildPaths;
use crate::llama_build::service::{
    BuildContext, BuildService, CancelOutcome, FixError, JobView, StartError, ValidateError,
    default_accelerator, is_job_id, owner_name,
};
use crate::llama_build::{Accelerator, BuildOptions, RefChoice, TargetOs};
use crate::llama_runtime::host::LiveHost;
use crate::llama_runtime::promote::{PromoteHost, RuntimePaths, VerifyStage};
use crate::llama_runtime::verify::{VerificationIntent, VerificationReport};
use crate::state::AppState;

/// Request bodies here are tiny; anything larger is not ours.
const BODY_LIMIT: u64 = 16 * 1024;
const CUDA_ARCH_MAX_LEN: usize = 32;

// ---------------------------------------------------------------------------
// Replies
// ---------------------------------------------------------------------------

fn reply(status: StatusCode, body: Value) -> ApiReply {
    box_reply(warp::reply::with_status(warp::reply::json(&body), status))
}

fn ok_reply(mut body: Value) -> ApiReply {
    body["ok"] = Value::Bool(true);
    reply(StatusCode::OK, body)
}

fn error_reply(status: StatusCode, message: impl Into<String>) -> ApiReply {
    reply(status, json!({ "ok": false, "error": message.into() }))
}

fn bad_request(message: impl Into<String>) -> ApiReply {
    error_reply(StatusCode::BAD_REQUEST, message)
}

fn conflict_reply(conflict: &Conflict) -> ApiReply {
    reply(
        StatusCode::CONFLICT,
        json!({
            "ok": false,
            "error": conflict.to_string(),
            "owner": owner_name(conflict.holder),
            "owner_id": conflict.holder_id,
        }),
    )
}

fn recovery_reply(message: String) -> ApiReply {
    reply(
        StatusCode::CONFLICT,
        json!({ "ok": false, "error": message, "recovery_required": true }),
    )
}

// ---------------------------------------------------------------------------
// Request parsing
// ---------------------------------------------------------------------------

fn parse_accelerator(raw: &str, os: TargetOs) -> Result<Accelerator, String> {
    let accelerator = match raw {
        "metal" => Accelerator::Metal,
        "cuda" => Accelerator::Cuda,
        "cpu" => Accelerator::Cpu,
        other => {
            return Err(format!(
                "unknown accelerator `{}`; expected metal, cuda or cpu",
                other.chars().take(32).collect::<String>()
            ));
        }
    };
    if !Accelerator::available_on(os).contains(&accelerator) {
        return Err(format!("{raw} is not available on this operating system"));
    }
    Ok(accelerator)
}

fn query_accelerator(
    params: &HashMap<String, String>,
    os: TargetOs,
) -> Result<Accelerator, String> {
    match params.get("accelerator").filter(|v| !v.is_empty()) {
        Some(raw) => parse_accelerator(raw, os),
        None => Ok(default_accelerator(os)),
    }
}

fn parse_body<T: for<'de> Deserialize<'de>>(bytes: &[u8]) -> Result<T, ApiReply> {
    serde_json::from_slice(bytes).map_err(|e| bad_request(format!("invalid request body: {e}")))
}

#[derive(Deserialize)]
struct StartBody {
    accelerator: String,
    #[serde(rename = "ref")]
    reference: String,
    #[serde(default)]
    cuda_arch: Option<String>,
    #[serde(default)]
    jobs: Option<u64>,
    #[serde(default)]
    headroom: bool,
    #[serde(default)]
    wipe: bool,
}

/// Validate a start request into [`BuildOptions`]. Pure so every bound is
/// unit-tested without a server.
fn build_options(body: StartBody, os: TargetOs, cpus: usize) -> Result<BuildOptions, String> {
    let accelerator = parse_accelerator(&body.accelerator, os)?;
    let reference = match body.reference.as_str() {
        "latest_release" => RefChoice::LatestTag,
        "master" => RefChoice::Master,
        _ => return Err("ref must be `latest_release` or `master`".into()),
    };
    let max_jobs = cpus.max(1).saturating_mul(2) as u64;
    if let Some(jobs) = body.jobs {
        if jobs == 0 || jobs > max_jobs {
            return Err(format!(
                "jobs must be between 1 and {max_jobs} (twice the logical CPUs)"
            ));
        }
        if body.headroom {
            return Err("jobs and headroom cannot be combined".into());
        }
    }
    let cuda_architectures = match body.cuda_arch.as_deref().map(str::trim) {
        None | Some("") => None,
        Some(arch) => {
            if accelerator != Accelerator::Cuda {
                return Err("cuda_arch only applies to CUDA builds".into());
            }
            if arch.len() > CUDA_ARCH_MAX_LEN
                || !arch
                    .bytes()
                    .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'z' | b';' | b'.' | b',' | b'-'))
            {
                return Err(format!(
                    "cuda_arch may use only 0-9, a-z and ; . , - (at most {CUDA_ARCH_MAX_LEN} characters)"
                ));
            }
            Some(validate_cuda_architectures(arch)?)
        }
    };
    let mut options = BuildOptions::new(accelerator, reference);
    options.cuda_architectures = cuda_architectures;
    options.parallel = body.jobs.map(|j| j as usize);
    options.leave_headroom = body.headroom;
    options.wipe_build_dir = body.wipe;
    Ok(options)
}

// ---------------------------------------------------------------------------
// JSON shaping
// ---------------------------------------------------------------------------

fn ref_name(choice: RefChoice) -> &'static str {
    match choice {
        RefChoice::LatestTag => "latest_release",
        RefChoice::Master => "master",
    }
}

fn options_json(options: &BuildOptions) -> Value {
    json!({
        "accelerator": options.accelerator.as_str(),
        "ref": ref_name(options.reference),
        "cuda_arch": options.cuda_architectures,
        "jobs": options.parallel,
        "headroom": options.leave_headroom,
        "wipe": options.wipe_build_dir,
    })
}

fn job_summary(record: &JobRecord) -> Value {
    json!({
        "id": record.id,
        "status": record.status,
        "phase": record.phase,
        "outcome": record.outcome,
        "accelerator": record.options.accelerator.as_str(),
        "finished_at": record.finished_unix,
        "failure_phase": record.failure.as_ref().map(|f| f.phase.as_str()),
    })
}

fn job_json(view: &JobView, log_cursor: usize) -> Value {
    let r = &view.record;
    json!({
        "id": r.id,
        "phase": r.phase,
        "status": r.status,
        "progress": r.progress.map(|p| json!({ "done": p.done, "total": p.total })),
        "cancellable": view.cancellable,
        "outcome": r.outcome,
        "failure": r.failure,
        "started_at": r.started_unix,
        "finished_at": r.finished_unix,
        "log_cursor": log_cursor,
        "resolved": r.resolved.as_ref().map(|x| json!({ "reference": x.reference, "commit": x.commit })),
        "restart_failed": r.status == JobStatus::SucceededRestartFailed,
        "accelerator": r.options.accelerator.as_str(),
        "options": options_json(&r.options),
    })
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

fn build_context(config: &AppConfig) -> Result<BuildContext, String> {
    let (bin_dir, _custom) = managed_bin_dir(config);
    let runtime = RuntimePaths::for_live_dir(&bin_dir)
        .map_err(|e| format!("Invalid runtime location: {e}"))?;
    Ok(BuildContext {
        paths: BuildPaths::new(&config.app_paths),
        runtime,
        config_dir: config.config_dir.clone(),
        fallback_config: config.llama_build.clone(),
    })
}

struct Request {
    state: AppState,
    config: Arc<AppConfig>,
    ctx: BuildContext,
    svc: Arc<BuildService>,
}

/// Authenticate and resolve paths; the early return is the reply to send.
fn begin(ctx: &ApiCtx, auth: &Option<String>) -> Result<Request, ApiReply> {
    if !check_api_token(auth, &ctx.config) {
        return Err(unauthorized_api_token());
    }
    let build = build_context(&ctx.config)
        .map_err(|e| error_reply(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Request {
        state: ctx.state.clone(),
        config: ctx.config.clone(),
        ctx: build,
        svc: ctx.state.llama_build.clone(),
    })
}

async fn blocking<T: Send + 'static>(
    f: impl FnOnce() -> T + Send + 'static,
) -> Result<T, ApiReply> {
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| error_reply(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

// ---------------------------------------------------------------------------
// Promote host
// ---------------------------------------------------------------------------

/// A [`LiveHost`] whose view of the running model is taken when the promotion
/// starts, not when the build did. `LiveHost::new` snapshots the model at
/// construction; after a many-minute compile that snapshot would be stale in
/// both directions (a model started meanwhile would be swapped from under).
/// The promote engine asks `cancelled`/`model_running` before it stops
/// anything, so the first call is early enough. Cancellation is this host's own
/// flag, the one the cancel route flips (`LiveHost`'s is never consulted).
struct DeferredHost {
    state: AppState,
    config: Arc<AppConfig>,
    managed_model: bool,
    cancel: Arc<AtomicBool>,
    live: OnceLock<LiveHost>,
}

impl DeferredHost {
    fn new(state: AppState, config: Arc<AppConfig>, managed_model: bool) -> Self {
        Self {
            state,
            config,
            managed_model,
            cancel: Arc::new(AtomicBool::new(false)),
            live: OnceLock::new(),
        }
    }

    fn live(&self) -> &LiveHost {
        self.live.get_or_init(|| {
            if self.managed_model {
                LiveHost::new(self.state.clone(), self.config.clone())
            } else {
                LiveHost::without_model(self.state.clone(), self.config.clone())
            }
        })
    }
}

impl PromoteHost for DeferredHost {
    fn cancelled(&self) -> bool {
        self.cancel.load(std::sync::atomic::Ordering::SeqCst)
    }

    fn model_running(&self) -> bool {
        self.live().model_running()
    }

    async fn stop_model(&self) -> Result<(), String> {
        self.live().stop_model().await
    }

    async fn restart_model(&self) -> Result<(), String> {
        self.live().restart_model().await
    }

    async fn verify(
        &self,
        dir: &Path,
        exe_name: &str,
        intent: &VerificationIntent,
        stage: VerifyStage,
    ) -> Result<VerificationReport, String> {
        self.live().verify(dir, exe_name, intent, stage).await
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn status(ctx: ApiCtx, auth: Option<String>) -> ApiReply {
    let req = match begin(&ctx, &auth) {
        Ok(r) => r,
        Err(r) => return r,
    };
    let os = req.svc.env().os();
    let gate_owner = req.state.runtime_gate.current_owner();
    let owner = req
        .svc
        .owner()
        .map(|o| owner_name(o).to_string())
        .or_else(|| gate_owner.map(|o| o.label().to_string()));
    let idle = req.svc.is_idle() && gate_owner.is_none();
    let svc = req.svc.clone();
    let build = req.ctx.clone();
    let result = blocking(move || {
        (
            svc.last_job(&build),
            if idle {
                build.recovery_required()
            } else {
                None
            },
        )
    })
    .await;
    let (last, recovery) = match result {
        Ok(v) => v,
        Err(r) => return r,
    };
    ok_reply(json!({
        "os": os,
        "accelerators": Accelerator::available_on(os).iter().map(|a| a.as_str()).collect::<Vec<_>>(),
        "default_accelerator": default_accelerator(os).as_str(),
        "active_job_id": req.svc.active_job_id(),
        "last_job": last.as_ref().map(job_summary),
        "recovery_required": recovery,
        "owner": owner,
    }))
}

async fn prereqs(ctx: ApiCtx, auth: Option<String>, params: HashMap<String, String>) -> ApiReply {
    let req = match begin(&ctx, &auth) {
        Ok(r) => r,
        Err(r) => return r,
    };
    let accelerator = match query_accelerator(&params, req.svc.env().os()) {
        Ok(a) => a,
        Err(e) => return bad_request(e),
    };
    let (svc, build) = (req.svc.clone(), req.ctx.clone());
    let view = match blocking(move || svc.prereq_view(&build, accelerator)).await {
        Ok(v) => v,
        Err(r) => return r,
    };
    ok_reply(json!({
        "accelerator": accelerator.as_str(),
        "report": view.report,
        "stale": view.stale,
        "validating": view.validating,
        "validate_job_id": view.validate_job_id,
    }))
}

#[derive(Deserialize)]
struct ValidateBody {
    accelerator: String,
}

async fn validate(ctx: ApiCtx, auth: Option<String>, body: bytes::Bytes) -> ApiReply {
    let req = match begin(&ctx, &auth) {
        Ok(r) => r,
        Err(r) => return r,
    };
    let body: ValidateBody = match parse_body(&body) {
        Ok(b) => b,
        Err(r) => return r,
    };
    let accelerator = match parse_accelerator(&body.accelerator, req.svc.env().os()) {
        Ok(a) => a,
        Err(e) => return bad_request(e),
    };
    match req.svc.start_validation(&req.ctx, accelerator) {
        Ok(job_id) => reply(
            StatusCode::ACCEPTED,
            json!({ "ok": true, "job_id": job_id }),
        ),
        Err(ValidateError::Busy(c)) => conflict_reply(&c),
        Err(ValidateError::Failed(e)) => error_reply(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

#[derive(Deserialize)]
struct FixBody {
    action: String,
    #[serde(default)]
    packages: Vec<String>,
}

async fn fix(ctx: ApiCtx, auth: Option<String>, body: bytes::Bytes) -> ApiReply {
    let req = match begin(&ctx, &auth) {
        Ok(r) => r,
        Err(r) => return r,
    };
    let body: FixBody = match parse_body(&body) {
        Ok(b) => b,
        Err(r) => return r,
    };
    if body.action != "brew_install" {
        return bad_request("action must be `brew_install`");
    }
    let svc = req.svc.clone();
    let packages = body.packages;
    let result = match blocking(move || svc.brew_install(&packages)).await {
        Ok(r) => r,
        Err(r) => return r,
    };
    match result {
        Ok(done) => {
            req.state.push_log(format!(
                "[monitor] llama-build: installed {}",
                done.installed.join(", ")
            ));
            ok_reply(json!({ "installed": done.installed, "output": done.output }))
        }
        Err(FixError::Unsupported(e)) | Err(FixError::Invalid(e)) => bad_request(e),
        Err(FixError::Busy(c)) => conflict_reply(&c),
        Err(FixError::Failed { error, output }) => reply(
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({ "ok": false, "error": error, "output": output }),
        ),
    }
}

async fn options(ctx: ApiCtx, auth: Option<String>, params: HashMap<String, String>) -> ApiReply {
    let req = match begin(&ctx, &auth) {
        Ok(r) => r,
        Err(r) => return r,
    };
    let os = req.svc.env().os();
    let accelerator = match query_accelerator(&params, os) {
        Ok(a) => a,
        Err(e) => return bad_request(e),
    };
    let (svc, build) = (req.svc.clone(), req.ctx.clone());
    let result = blocking(move || {
        let refs = svc.refs(&build);
        let arch = (accelerator == Accelerator::Cuda)
            .then(|| svc.env().cuda_arch_auto())
            .flatten();
        (refs, arch, build.config().ccache_max_bytes)
    })
    .await;
    let (refs, cuda_arch_auto, cap) = match result {
        Ok(v) => v,
        Err(r) => return r,
    };
    let entry = |id: &str, label: &str, r: &Option<crate::llama_build::source::ResolvedRef>| {
        json!({
            "id": id,
            "label": label,
            "reference": r.as_ref().map(|x| x.reference.clone()),
            "commit": r.as_ref().map(|x| x.commit.clone()),
        })
    };
    ok_reply(json!({
        "accelerator": accelerator.as_str(),
        "refs": [
            entry("latest_release", "Latest release", &refs.latest_release),
            entry("master", "Master (development)", &refs.master),
        ],
        "refs_error": refs.error,
        "cuda_arch_auto": cuda_arch_auto,
        "cuda_arch_choices": CUDA_ARCH_BY_COMPUTE_CAP.iter().map(|(_, arch)| *arch).collect::<Vec<_>>(),
        "jobs_auto": req.svc.jobs_auto(accelerator),
        "ccache_cap_bytes": cap,
        "ccache_cap_min_bytes": MIN_CCACHE_BYTES,
        "ccache_cap_max_bytes": MAX_CCACHE_BYTES,
        "show_cuda": Accelerator::available_on(os).contains(&Accelerator::Cuda),
    }))
}

async fn start(ctx: ApiCtx, auth: Option<String>, body: bytes::Bytes) -> ApiReply {
    let req = match begin(&ctx, &auth) {
        Ok(r) => r,
        Err(r) => return r,
    };
    let body: StartBody = match parse_body(&body) {
        Ok(b) => b,
        Err(r) => return r,
    };
    let options = match build_options(body, req.svc.env().os(), req.svc.env().logical_cpus()) {
        Ok(o) => o,
        Err(e) => return bad_request(e),
    };
    // Name the owner first: while something else is running its process
    // records and journal are legitimate, not a reason to demand recovery.
    if let Some(conflict) = req.svc.conflict_for_build() {
        return conflict_reply(&conflict);
    }
    if req.state.runtime_gate.current_owner().is_none() {
        let build = req.ctx.clone();
        match blocking(move || build.recovery_required()).await {
            Ok(Some(why)) => return recovery_reply(why),
            Ok(None) => {}
            Err(r) => return r,
        }
    }
    let (_, custom) = managed_bin_dir(&req.config);
    let host = DeferredHost::new(req.state.clone(), req.config.clone(), custom.is_none());
    let cancel = host.cancel.clone();
    let started = req
        .svc
        .start_build(
            &req.ctx,
            host,
            cancel,
            req.state.runtime_gate.clone(),
            options,
        )
        .await;
    match started {
        Ok(s) => {
            req.state
                .push_log(format!("[monitor] llama-build: started job {}", s.job_id));
            reply(
                StatusCode::ACCEPTED,
                json!({ "ok": true, "job_id": s.job_id, "resolved_commit": s.resolved_commit }),
            )
        }
        Err(StartError::Busy(c)) => conflict_reply(&c),
        Err(StartError::RecoveryRequired(why)) => recovery_reply(why),
        Err(StartError::Failed(e)) => error_reply(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

fn checked_job_id(id: &str) -> Result<(), ApiReply> {
    if is_job_id(id) {
        Ok(())
    } else {
        Err(bad_request("invalid job id"))
    }
}

async fn job(ctx: ApiCtx, auth: Option<String>, id: String) -> ApiReply {
    let req = match begin(&ctx, &auth) {
        Ok(r) => r,
        Err(r) => return r,
    };
    if let Err(r) = checked_job_id(&id) {
        return r;
    }
    let (svc, build) = (req.svc.clone(), req.ctx.clone());
    let result = blocking(move || {
        let view = svc.job(&build, &id)?;
        let cursor = svc
            .read_log(&build, &view.record.id, usize::MAX)
            .map_or(0, |p| p.next);
        Some((view, cursor))
    })
    .await;
    match result {
        Ok(Some((view, cursor))) => ok_reply(job_json(&view, cursor)),
        Ok(None) => error_reply(StatusCode::NOT_FOUND, "job not found"),
        Err(r) => r,
    }
}

async fn job_log(
    ctx: ApiCtx,
    auth: Option<String>,
    id: String,
    params: HashMap<String, String>,
) -> ApiReply {
    let req = match begin(&ctx, &auth) {
        Ok(r) => r,
        Err(r) => return r,
    };
    if let Err(r) = checked_job_id(&id) {
        return r;
    }
    let after = match params.get("after").filter(|v| !v.is_empty()) {
        None => 0,
        Some(raw) => match raw.parse::<usize>() {
            Ok(n) => n,
            Err(_) => return bad_request("after must be a non-negative integer"),
        },
    };
    let (svc, build) = (req.svc.clone(), req.ctx.clone());
    match blocking(move || svc.read_log(&build, &id, after)).await {
        Ok(Some(page)) => ok_reply(json!({
            "lines": page.lines,
            "next": page.next,
            "truncated": page.truncated,
        })),
        Ok(None) => error_reply(StatusCode::NOT_FOUND, "job not found"),
        Err(r) => r,
    }
}

async fn cancel(ctx: ApiCtx, auth: Option<String>, id: String) -> ApiReply {
    let req = match begin(&ctx, &auth) {
        Ok(r) => r,
        Err(r) => return r,
    };
    if let Err(r) = checked_job_id(&id) {
        return r;
    }
    let (svc, build) = (req.svc.clone(), req.ctx.clone());
    match blocking(move || svc.cancel(&build, &id)).await {
        Ok(CancelOutcome::Requested) => ok_reply(json!({})),
        Ok(CancelOutcome::NotCancellable(why)) => reply(
            StatusCode::CONFLICT,
            json!({ "ok": false, "cancellable": false, "error": why }),
        ),
        Ok(CancelOutcome::Unknown) => error_reply(StatusCode::NOT_FOUND, "job not found"),
        Err(r) => r,
    }
}

async fn cache(ctx: ApiCtx, auth: Option<String>) -> ApiReply {
    let req = match begin(&ctx, &auth) {
        Ok(r) => r,
        Err(r) => return r,
    };
    let (svc, build) = (req.svc.clone(), req.ctx.clone());
    let view = match blocking(move || svc.cache_view(&build)).await {
        Ok(v) => v,
        Err(r) => return r,
    };
    ok_reply(json!({
        "source_bytes": view.source_bytes,
        "ccache_bytes": view.ccache_bytes,
        "ccache_cap_bytes": view.ccache_cap_bytes,
        "build_trees": view.build_trees.iter()
            .map(|(a, bytes)| json!({ "accelerator": a.as_str(), "bytes": bytes }))
            .collect::<Vec<_>>(),
        "previous_bytes": view.previous_bytes,
        "owner": req.svc.owner().map(owner_name),
    }))
}

async fn clean_cache(
    ctx: ApiCtx,
    auth: Option<String>,
    params: HashMap<String, String>,
) -> ApiReply {
    let req = match begin(&ctx, &auth) {
        Ok(r) => r,
        Err(r) => return r,
    };
    let include_source = match params.get("include_source").map(String::as_str) {
        None | Some("") | Some("false") | Some("0") => false,
        Some("true") | Some("1") => true,
        Some(_) => return bad_request("include_source must be true or false"),
    };
    let (svc, build) = (req.svc.clone(), req.ctx.clone());
    let result = match blocking(move || svc.clean(&build, CleanOptions { include_source })).await {
        Ok(r) => r,
        Err(r) => return r,
    };
    match result {
        Ok(report) => {
            req.state.push_log(format!(
                "[monitor] llama-build: cleaned caches, freed {} bytes",
                report.freed_bytes
            ));
            ok_reply(json!({ "removed": report.removed, "freed_bytes": report.freed_bytes }))
        }
        Err(CleanError::Busy(c)) => conflict_reply(&c),
        Err(CleanError::Io(e)) if e.contains("containment is unresolved") => recovery_reply(e),
        Err(CleanError::Io(e)) => error_reply(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

async fn cache_cap(ctx: ApiCtx, auth: Option<String>, body: bytes::Bytes) -> ApiReply {
    let req = match begin(&ctx, &auth) {
        Ok(r) => r,
        Err(r) => return r,
    };
    let value: Value = match parse_body(&body) {
        Ok(v) => v,
        Err(r) => return r,
    };
    // Only a whole, non-negative JSON integer is a byte count: floats (and so
    // NaN/Infinity, which JSON cannot carry anyway), strings and negatives are
    // all rejected rather than rounded.
    let Some(bytes) = value.get("ccache_max_bytes").and_then(Value::as_u64) else {
        return bad_request("ccache_max_bytes must be a whole number of bytes");
    };
    let bytes = match validate_ccache_cap(bytes) {
        Ok(b) => b,
        Err(e) => return bad_request(e),
    };
    let dir = req.ctx.config_dir.clone();
    let stored = blocking(move || {
        LlamaBuildConfig {
            ccache_max_bytes: bytes,
        }
        .store(&dir)
    })
    .await;
    match stored {
        Ok(Ok(())) => ok_reply(json!({ "ccache_cap_bytes": bytes })),
        Ok(Err(e)) => error_reply(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("could not save the cache size: {e}"),
        ),
        Err(r) => r,
    }
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

fn auth_header() -> impl Filter<Extract = (Option<String>,), Error = warp::Rejection> + Clone {
    warp::header::optional::<String>("authorization")
}

fn body() -> impl Filter<Extract = (bytes::Bytes,), Error = warp::Rejection> + Clone {
    warp::body::content_length_limit(BODY_LIMIT).and(warp::body::bytes())
}

fn with_ctx(
    ctx: ApiCtx,
) -> impl Filter<Extract = (ApiCtx,), Error = std::convert::Infallible> + Clone {
    warp::any().map(move || ctx.clone())
}

pub(crate) fn routes(ctx: ApiCtx) -> ApiRoute {
    let c = || with_ctx(ctx.clone());
    let base = || warp::path("api").and(warp::path("llama-build"));

    let status_r = base()
        .and(warp::path("status"))
        .and(warp::path::end())
        .and(warp::get())
        .and(c())
        .and(auth_header())
        .then(status);
    let prereqs_r = base()
        .and(warp::path("prereqs"))
        .and(warp::path::end())
        .and(warp::get())
        .and(c())
        .and(auth_header())
        .and(warp::query::<HashMap<String, String>>())
        .then(prereqs);
    let validate_r = base()
        .and(warp::path("prereqs"))
        .and(warp::path("validate"))
        .and(warp::path::end())
        .and(warp::post())
        .and(c())
        .and(auth_header())
        .and(body())
        .then(validate);
    let fix_r = base()
        .and(warp::path("prereqs"))
        .and(warp::path("fix"))
        .and(warp::path::end())
        .and(warp::post())
        .and(c())
        .and(auth_header())
        .and(body())
        .then(fix);
    let options_r = base()
        .and(warp::path("options"))
        .and(warp::path::end())
        .and(warp::get())
        .and(c())
        .and(auth_header())
        .and(warp::query::<HashMap<String, String>>())
        .then(options);
    let start_r = base()
        .and(warp::path("start"))
        .and(warp::path::end())
        .and(warp::post())
        .and(c())
        .and(auth_header())
        .and(body())
        .then(start);
    let job_r = base()
        .and(warp::path("jobs"))
        .and(warp::path::param::<String>())
        .and(warp::path::end())
        .and(warp::get())
        .and(c())
        .and(auth_header())
        .then(|id: String, ctx: ApiCtx, auth: Option<String>| job(ctx, auth, id));
    let log_r = base()
        .and(warp::path("jobs"))
        .and(warp::path::param::<String>())
        .and(warp::path("log"))
        .and(warp::path::end())
        .and(warp::get())
        .and(c())
        .and(auth_header())
        .and(warp::query::<HashMap<String, String>>())
        .then(
            |id: String, ctx: ApiCtx, auth: Option<String>, params: HashMap<String, String>| {
                job_log(ctx, auth, id, params)
            },
        );
    let cancel_r = base()
        .and(warp::path("jobs"))
        .and(warp::path::param::<String>())
        .and(warp::path("cancel"))
        .and(warp::path::end())
        .and(warp::post())
        .and(c())
        .and(auth_header())
        .then(|id: String, ctx: ApiCtx, auth: Option<String>| cancel(ctx, auth, id));
    let cache_r = base()
        .and(warp::path("cache"))
        .and(warp::path::end())
        .and(warp::get())
        .and(c())
        .and(auth_header())
        .then(cache);
    let clean_r = base()
        .and(warp::path("cache"))
        .and(warp::path::end())
        .and(warp::delete())
        .and(c())
        .and(auth_header())
        .and(warp::query::<HashMap<String, String>>())
        .then(clean_cache);
    let cap_r = base()
        .and(warp::path("cache-cap"))
        .and(warp::path::end())
        .and(warp::put())
        .and(c())
        .and(auth_header())
        .and(body())
        .then(cache_cap);

    status_r
        .or(prereqs_r)
        .unify()
        .or(validate_r)
        .unify()
        .or(fix_r)
        .unify()
        .or(options_r)
        .unify()
        .or(start_r)
        .unify()
        .or(job_r)
        .unify()
        .or(log_r)
        .unify()
        .or(cancel_r)
        .unify()
        .or(cache_r)
        .unify()
        .or(clean_r)
        .unify()
        .or(cap_r)
        .unify()
        .boxed()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TlsMode;
    use crate::llama_build::BuildPlan;
    use crate::llama_build::job::Phase;
    use crate::llama_build::job::{
        JobDeps, PlanInputs, StepIo, StepResult, Toolchain, finish, new_shared, set_phase_public,
        with_state,
    };
    use crate::llama_build::prereqs::{Check, Collector, PrereqReport};
    use crate::llama_build::service::{BuildEnv, ToolchainMode, ToolchainSpec};
    use crate::llama_build::source::ResolvedRef;
    use crate::web::auth::AuthManager;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;

    const COMMIT: &str = "098ce526b0000000000000000000000000000000";
    const TOKEN: &str = "test-token";

    #[derive(Default)]
    struct Control {
        hold_fetch: AtomicBool,
        hold_validate: AtomicBool,
        prereqs_fail: AtomicBool,
        resolve_fail: AtomicBool,
        fingerprint: Mutex<String>,
        validations: AtomicUsize,
        brew_calls: Mutex<Vec<Vec<String>>>,
    }

    struct FakeTool {
        ctl: Arc<Control>,
        os: TargetOs,
        mode: ToolchainMode,
    }

    impl Toolchain for FakeTool {
        fn os(&self) -> TargetOs {
            self.os
        }

        fn prereqs(&self, options: &BuildOptions) -> PrereqReport {
            if self.mode == ToolchainMode::Validate {
                self.ctl.validations.fetch_add(1, Ordering::SeqCst);
                while self.ctl.hold_validate.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
            let mut c = Collector::default();
            c.fingerprint_part("k", &self.ctl.fingerprint.lock().unwrap());
            if self.ctl.prereqs_fail.load(Ordering::SeqCst) {
                c.push(Check::fail("cmake", "CMake", "cmake was not found."));
            } else {
                c.push(Check::pass("git", "git", "ok"));
            }
            PrereqReport::from_collector(self.os, options.accelerator, c)
        }

        fn plan_inputs(&self, _: &BuildOptions) -> Result<PlanInputs, String> {
            Ok(PlanInputs {
                ccache: false,
                apple_silicon: true,
                cuda: None,
                cuda_root: None,
                device_id: None,
            })
        }

        fn resolve(&self, options: &BuildOptions) -> Result<ResolvedRef, String> {
            if self.ctl.resolve_fail.load(Ordering::SeqCst) {
                return Err("network unreachable".into());
            }
            Ok(ResolvedRef {
                reference: match options.reference {
                    RefChoice::LatestTag => "b11727".into(),
                    RefChoice::Master => "master".into(),
                },
                commit: COMMIT.into(),
                build_number: Some(11727),
            })
        }

        fn fetch(&self, _: &Path, _: &ResolvedRef, _: &str, io: &mut StepIo) -> StepResult {
            (io.sink)("Fetching from origin");
            while self.ctl.hold_fetch.load(Ordering::SeqCst) && !(io.cancel)() {
                std::thread::sleep(Duration::from_millis(10));
            }
            if (io.cancel)() {
                return StepResult::Cancelled;
            }
            StepResult::Failed {
                exit_code: Some(1),
                command: "fake fetch".into(),
                detail: None,
            }
        }

        fn configure(&self, _: &BuildPlan, _: &mut StepIo) -> StepResult {
            StepResult::Cancelled
        }

        fn compile(&self, _: &BuildPlan, _: usize, _: bool, _: &mut StepIo) -> StepResult {
            StepResult::Cancelled
        }

        fn ninja_targets(&self, _: &BuildPlan) -> Result<String, String> {
            Err("unused".into())
        }

        fn toolchain_versions(&self) -> std::collections::BTreeMap<String, String> {
            Default::default()
        }

        fn free_space(&self, _: &Path) -> Option<u64> {
            Some(u64::MAX)
        }

        fn logical_cpus(&self) -> usize {
            8
        }
    }

    struct FakeEnv {
        ctl: Arc<Control>,
        os: TargetOs,
    }

    impl BuildEnv for FakeEnv {
        fn os(&self) -> TargetOs {
            self.os
        }

        fn logical_cpus(&self) -> usize {
            8
        }

        fn toolchain(&self, spec: ToolchainSpec) -> Arc<dyn Toolchain> {
            Arc::new(FakeTool {
                ctl: self.ctl.clone(),
                os: self.os,
                mode: spec.mode,
            })
        }

        fn brew_install(&self, packages: &[&str]) -> Result<String, String> {
            self.ctl
                .brew_calls
                .lock()
                .unwrap()
                .push(packages.iter().map(|p| p.to_string()).collect());
            Ok("==> installed".into())
        }

        fn cuda_arch_auto(&self) -> Option<String> {
            Some("120a-real".into())
        }
    }

    struct Rig {
        _tmp: tempfile::TempDir,
        config: Arc<AppConfig>,
        state: AppState,
        ctl: Arc<Control>,
        routes: ApiRoute,
    }

    impl Rig {
        fn new(os: TargetOs) -> Self {
            let tmp = tempfile::tempdir().unwrap();
            let root = std::fs::canonicalize(tmp.path()).unwrap();
            let mut config = AppConfig::for_test(Some(TOKEN.into()), None);
            config.app_paths = crate::paths::AppPaths::from_root(root.clone());
            config.config_dir = root.clone();
            let exe = if cfg!(windows) {
                "llama-server.exe"
            } else {
                "llama-server"
            };
            config.llama_server_path = config.app_paths.bin_dir().join(exe);
            // A live runtime that every failure must leave untouched.
            std::fs::create_dir_all(config.app_paths.bin_dir()).unwrap();
            std::fs::write(&config.llama_server_path, b"live").unwrap();
            let config = Arc::new(config);

            let ctl = Arc::new(Control::default());
            let state = AppState {
                llama_build: Arc::new(
                    crate::llama_build::service::BuildService::new(Arc::new(FakeEnv {
                        ctl: ctl.clone(),
                        os,
                    }))
                    .with_pin_wait(Duration::from_millis(500)),
                ),
                ..AppState::default()
            };
            let routes = routes(ApiCtx {
                state: state.clone(),
                config: config.clone(),
                auth: AuthManager::new(None, None, &TlsMode::None),
            });
            Self {
                _tmp: tmp,
                config,
                state,
                ctl,
                routes,
            }
        }

        fn paths(&self) -> BuildPaths {
            BuildPaths::new(&self.config.app_paths)
        }

        async fn call(&self, method: &str, path: &str, body: Option<&str>) -> (u16, Value) {
            let mut req = warp::test::request()
                .method(method)
                .path(path)
                .header("authorization", format!("Bearer {TOKEN}"));
            if let Some(b) = body {
                req = req.header("content-type", "application/json").body(b);
            }
            let resp = req.reply(&self.routes).await;
            let value = serde_json::from_slice(resp.body()).unwrap_or(Value::Null);
            (resp.status().as_u16(), value)
        }

        async fn get(&self, path: &str) -> (u16, Value) {
            self.call("GET", path, None).await
        }

        async fn start(&self, body: &str) -> (u16, Value) {
            self.call("POST", "/api/llama-build/start", Some(body))
                .await
        }

        async fn wait_status(&self, id: &str, want: &str) -> Value {
            for _ in 0..300 {
                let (_, job) = self.get(&format!("/api/llama-build/jobs/{id}")).await;
                if job["status"] == want {
                    return job;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            panic!("job {id} never reached {want}");
        }

        fn deps(&self) -> JobDeps {
            let ctx = build_context(&self.config).unwrap();
            JobDeps {
                toolchain: Arc::new(FakeTool {
                    ctl: self.ctl.clone(),
                    os: TargetOs::Macos,
                    mode: ToolchainMode::Build,
                }),
                paths: ctx.paths,
                runtime: ctx.runtime,
                config: LlamaBuildConfig::default(),
            }
        }

        /// A finished job on disk that no live service state knows about.
        fn disk_job(&self, id: &str, lines: &[String], status: Option<JobStatus>) {
            let options = BuildOptions::new(Accelerator::Metal, RefChoice::LatestTag);
            let shared = new_shared(&self.deps(), &options, id, true).unwrap();
            with_state(&shared, |s| {
                for l in lines {
                    s.log.push(l);
                }
            });
            if let Some(status) = status {
                finish(&shared, status, "done");
            }
        }
    }

    const START_METAL: &str =
        r#"{"accelerator":"metal","ref":"latest_release","headroom":false,"wipe":false}"#;

    #[tokio::test]
    async fn every_route_requires_the_api_token() {
        let rig = Rig::new(TargetOs::Macos);
        let id = "a1".repeat(16);
        let cases: Vec<(&str, String, Option<&str>)> = vec![
            ("GET", "/api/llama-build/status".into(), None),
            ("GET", "/api/llama-build/prereqs".into(), None),
            (
                "POST",
                "/api/llama-build/prereqs/validate".into(),
                Some("{}"),
            ),
            ("POST", "/api/llama-build/prereqs/fix".into(), Some("{}")),
            ("GET", "/api/llama-build/options".into(), None),
            ("POST", "/api/llama-build/start".into(), Some("{}")),
            ("GET", format!("/api/llama-build/jobs/{id}"), None),
            ("GET", format!("/api/llama-build/jobs/{id}/log"), None),
            (
                "POST",
                format!("/api/llama-build/jobs/{id}/cancel"),
                Some("{}"),
            ),
            ("GET", "/api/llama-build/cache".into(), None),
            ("DELETE", "/api/llama-build/cache".into(), None),
            ("PUT", "/api/llama-build/cache-cap".into(), Some("{}")),
        ];
        for (method, path, body) in cases {
            for header in [None, Some("Bearer wrong")] {
                let mut req = warp::test::request().method(method).path(&path);
                if let Some(h) = header {
                    req = req.header("authorization", h);
                }
                if let Some(b) = body {
                    req = req.header("content-type", "application/json").body(b);
                }
                let resp = req.reply(&rig.routes).await;
                assert_eq!(resp.status(), 401, "{method} {path} header={header:?}");
            }
        }
        // Nothing ran on behalf of an unauthenticated caller.
        assert_eq!(rig.ctl.validations.load(Ordering::SeqCst), 0);
        assert!(rig.state.llama_build.is_idle());
    }

    #[tokio::test]
    async fn status_reports_the_platform_and_an_idle_service() {
        let rig = Rig::new(TargetOs::Macos);
        let (code, body) = rig.get("/api/llama-build/status").await;
        assert_eq!(code, 200);
        assert_eq!(body["ok"], true);
        assert_eq!(body["os"], "macos");
        assert_eq!(body["accelerators"], json!(["metal", "cpu"]));
        assert_eq!(body["default_accelerator"], "metal");
        assert!(body["active_job_id"].is_null());
        assert!(body["last_job"].is_null());
        assert!(body["recovery_required"].is_null());
        assert!(body["owner"].is_null());

        let win = Rig::new(TargetOs::Windows);
        let (_, body) = win.get("/api/llama-build/status").await;
        assert_eq!(body["accelerators"], json!(["cuda", "cpu"]));
        assert_eq!(body["default_accelerator"], "cuda");
    }

    #[tokio::test]
    async fn status_demands_recovery_for_unresolved_processes_or_a_bad_journal() {
        let rig = Rig::new(TargetOs::Macos);
        std::fs::create_dir_all(rig.paths().root).unwrap();
        std::fs::write(rig.paths().process_records, b"[]").unwrap();
        let (_, body) = rig.get("/api/llama-build/status").await;
        assert!(
            body["recovery_required"]
                .as_str()
                .unwrap()
                .contains("process records")
        );
        let (code, body) = rig.start(START_METAL).await;
        assert_eq!(code, 409);
        assert_eq!(body["recovery_required"], true);
        std::fs::remove_file(rig.paths().process_records).unwrap();

        std::fs::write(
            rig.config.app_paths.runtime_transaction_file(),
            b"{not json",
        )
        .unwrap();
        let (_, body) = rig.get("/api/llama-build/status").await;
        assert!(
            body["recovery_required"]
                .as_str()
                .unwrap()
                .contains("journal")
        );
        let (code, body) = rig.start(START_METAL).await;
        assert_eq!(code, 409);
        assert_eq!(body["recovery_required"], true);
        assert!(rig.state.llama_build.active_job_id().is_none());
    }

    #[tokio::test]
    async fn prereqs_validates_the_accelerator_and_reports_a_missing_report_as_stale() {
        let rig = Rig::new(TargetOs::Macos);
        let (code, _) = rig
            .get("/api/llama-build/prereqs?accelerator=quantum")
            .await;
        assert_eq!(code, 400);
        let (code, body) = rig.get("/api/llama-build/prereqs?accelerator=cuda").await;
        assert_eq!(code, 400, "cuda is not buildable on macOS: {body}");

        let (code, body) = rig.get("/api/llama-build/prereqs?accelerator=metal").await;
        assert_eq!(code, 200);
        assert_eq!(body["accelerator"], "metal");
        assert!(body["report"].is_null());
        assert_eq!(body["stale"], true);
        assert_eq!(body["validating"], false);
        assert!(body["validate_job_id"].is_null());
        // No accelerator means the platform default.
        let (_, body) = rig.get("/api/llama-build/prereqs").await;
        assert_eq!(body["accelerator"], "metal");
    }

    #[tokio::test]
    async fn prereqs_is_stale_only_when_the_fast_recheck_fingerprint_differs() {
        let rig = Rig::new(TargetOs::Macos);
        *rig.ctl.fingerprint.lock().unwrap() = "v1".into();
        let report = FakeTool {
            ctl: rig.ctl.clone(),
            os: TargetOs::Macos,
            mode: ToolchainMode::Probe,
        }
        .prereqs(&BuildOptions::new(Accelerator::Metal, RefChoice::LatestTag));
        std::fs::create_dir_all(rig.paths().root).unwrap();
        std::fs::write(
            rig.paths().prereqs_last,
            serde_json::to_vec(&report).unwrap(),
        )
        .unwrap();

        let (_, body) = rig.get("/api/llama-build/prereqs?accelerator=metal").await;
        assert_eq!(body["stale"], false);
        assert_eq!(body["report"]["fingerprint"], report.fingerprint);
        assert_eq!(body["report"]["ready"], true);

        // A report for another accelerator does not describe this one.
        let (_, body) = rig.get("/api/llama-build/prereqs?accelerator=cpu").await;
        assert!(body["report"].is_null());
        assert_eq!(body["stale"], true);

        // The toolchain changed. The re-check result is cached briefly, so a
        // fresh service sees the change immediately.
        let rig2 = Rig::new(TargetOs::Macos);
        *rig2.ctl.fingerprint.lock().unwrap() = "v2".into();
        std::fs::create_dir_all(rig2.paths().root).unwrap();
        std::fs::write(
            rig2.paths().prereqs_last,
            serde_json::to_vec(&report).unwrap(),
        )
        .unwrap();
        let (_, body) = rig2.get("/api/llama-build/prereqs?accelerator=metal").await;
        assert_eq!(body["stale"], true);
        assert!(body["report"].is_object());
    }

    #[tokio::test]
    async fn validation_is_single_flight_and_excludes_a_build() {
        let rig = Rig::new(TargetOs::Macos);
        rig.ctl.hold_validate.store(true, Ordering::SeqCst);
        let (code, body) = rig
            .call(
                "POST",
                "/api/llama-build/prereqs/validate",
                Some(r#"{"accelerator":"metal"}"#),
            )
            .await;
        assert_eq!(code, 202);
        let job_id = body["job_id"].as_str().unwrap().to_string();

        let (_, prereqs) = rig.get("/api/llama-build/prereqs?accelerator=metal").await;
        assert_eq!(prereqs["validating"], true);
        assert_eq!(prereqs["validate_job_id"], job_id);

        let (code, body) = rig
            .call(
                "POST",
                "/api/llama-build/prereqs/validate",
                Some(r#"{"accelerator":"metal"}"#),
            )
            .await;
        assert_eq!(code, 409);
        assert_eq!(body["owner"], "validation");
        let (code, body) = rig.start(START_METAL).await;
        assert_eq!(code, 409);
        assert_eq!(body["owner"], "validation");
        let (code, body) = rig.call("DELETE", "/api/llama-build/cache", None).await;
        assert_eq!(code, 409);
        assert_eq!(body["owner"], "validation");
        let (_, status) = rig.get("/api/llama-build/status").await;
        assert_eq!(status["owner"], "validation");

        rig.ctl.hold_validate.store(false, Ordering::SeqCst);
        for _ in 0..200 {
            if rig.state.llama_build.is_idle() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(rig.state.llama_build.is_idle());
        let (_, prereqs) = rig.get("/api/llama-build/prereqs?accelerator=metal").await;
        assert_eq!(prereqs["validating"], false);
        assert_eq!(rig.ctl.validations.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn validate_rejects_bad_bodies_and_unavailable_accelerators() {
        let rig = Rig::new(TargetOs::Macos);
        for body in [
            "{}",
            "not json",
            r#"{"accelerator":"cuda"}"#,
            r#"{"accelerator":"../x"}"#,
        ] {
            let (code, _) = rig
                .call("POST", "/api/llama-build/prereqs/validate", Some(body))
                .await;
            assert_eq!(code, 400, "{body}");
        }
        assert_eq!(rig.ctl.validations.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn fix_only_installs_allowlisted_packages_with_direct_argv_on_macos() {
        let rig = Rig::new(TargetOs::Macos);
        for body in [
            r#"{"action":"brew_install","packages":["cmake;rm -rf /"]}"#,
            r#"{"action":"brew_install","packages":["wget"]}"#,
            r#"{"action":"brew_install","packages":["cmake","--HEAD"]}"#,
            r#"{"action":"brew_install","packages":[]}"#,
            r#"{"action":"brew_install"}"#,
            r#"{"action":"shell","packages":["cmake"]}"#,
            "garbage",
        ] {
            let (code, _) = rig
                .call("POST", "/api/llama-build/prereqs/fix", Some(body))
                .await;
            assert_eq!(code, 400, "{body}");
        }
        assert!(rig.ctl.brew_calls.lock().unwrap().is_empty());

        let (code, body) = rig
            .call(
                "POST",
                "/api/llama-build/prereqs/fix",
                Some(r#"{"action":"brew_install","packages":["ninja","cmake","cmake"]}"#),
            )
            .await;
        assert_eq!(code, 200);
        assert_eq!(body["ok"], true);
        assert_eq!(body["installed"], json!(["cmake", "ninja"]));
        assert_eq!(body["output"], "==> installed");
        assert_eq!(
            *rig.ctl.brew_calls.lock().unwrap(),
            vec![vec!["cmake".to_string(), "ninja".into()]]
        );
        assert!(rig.state.llama_build.is_idle());
    }

    #[tokio::test]
    async fn fix_is_refused_off_macos() {
        for os in [TargetOs::Windows, TargetOs::Linux] {
            let rig = Rig::new(os);
            let (code, body) = rig
                .call(
                    "POST",
                    "/api/llama-build/prereqs/fix",
                    Some(r#"{"action":"brew_install","packages":["cmake"]}"#),
                )
                .await;
            assert_eq!(code, 400);
            assert_eq!(body["ok"], false);
            assert!(rig.ctl.brew_calls.lock().unwrap().is_empty());
        }
    }

    #[tokio::test]
    async fn fix_conflicts_with_a_running_validation() {
        let rig = Rig::new(TargetOs::Macos);
        rig.ctl.hold_validate.store(true, Ordering::SeqCst);
        let (code, _) = rig
            .call(
                "POST",
                "/api/llama-build/prereqs/validate",
                Some(r#"{"accelerator":"metal"}"#),
            )
            .await;
        assert_eq!(code, 202);
        let (code, body) = rig
            .call(
                "POST",
                "/api/llama-build/prereqs/fix",
                Some(r#"{"action":"brew_install","packages":["cmake"]}"#),
            )
            .await;
        assert_eq!(code, 409);
        assert_eq!(body["owner"], "validation");
        assert!(rig.ctl.brew_calls.lock().unwrap().is_empty());
        rig.ctl.hold_validate.store(false, Ordering::SeqCst);
    }

    #[tokio::test]
    async fn options_offer_refs_and_bounds_but_no_time_estimate() {
        let rig = Rig::new(TargetOs::Windows);
        let (code, body) = rig.get("/api/llama-build/options?accelerator=cuda").await;
        assert_eq!(code, 200);
        assert_eq!(body["accelerator"], "cuda");
        assert_eq!(body["refs"][0]["id"], "latest_release");
        assert_eq!(body["refs"][0]["reference"], "b11727");
        assert_eq!(body["refs"][0]["commit"], COMMIT);
        assert_eq!(body["refs"][1]["id"], "master");
        assert_eq!(body["refs"][1]["reference"], "master");
        assert!(body["refs_error"].is_null());
        assert_eq!(body["cuda_arch_auto"], "120a-real");
        assert!(
            body["cuda_arch_choices"]
                .as_array()
                .unwrap()
                .contains(&json!("120a-real"))
        );
        assert_eq!(body["jobs_auto"], 7);
        assert_eq!(body["ccache_cap_bytes"], 2u64 << 30);
        assert_eq!(body["ccache_cap_min_bytes"], 256u64 << 20);
        assert_eq!(body["ccache_cap_max_bytes"], 16u64 << 30);
        assert_eq!(body["show_cuda"], true);
        // The plan forbids promising a build duration.
        for key in body.as_object().unwrap().keys() {
            for word in ["estimate", "eta", "duration", "seconds", "minutes"] {
                assert!(!key.contains(word), "unexpected time field {key}");
            }
        }

        let mac = Rig::new(TargetOs::Macos);
        let (_, body) = mac.get("/api/llama-build/options?accelerator=metal").await;
        assert_eq!(body["show_cuda"], false);
        assert!(body["cuda_arch_auto"].is_null());
    }

    #[tokio::test]
    async fn options_degrade_when_upstream_cannot_be_resolved() {
        let rig = Rig::new(TargetOs::Macos);
        rig.ctl.resolve_fail.store(true, Ordering::SeqCst);
        let (code, body) = rig.get("/api/llama-build/options").await;
        assert_eq!(code, 200);
        assert_eq!(body["ok"], true);
        assert!(
            body["refs_error"]
                .as_str()
                .unwrap()
                .contains("network unreachable")
        );
        assert!(body["refs"][0]["commit"].is_null());
        assert_eq!(body["refs"][0]["id"], "latest_release");
        let (code, _) = rig.get("/api/llama-build/options?accelerator=cuda").await;
        assert_eq!(code, 400);
    }

    fn body_for(os: TargetOs, cpus: usize, json: &str) -> Result<BuildOptions, String> {
        build_options(serde_json::from_str(json).unwrap(), os, cpus)
    }

    #[test]
    fn start_options_enforce_every_bound() {
        let m = TargetOs::Macos;
        let ok = body_for(
            m,
            8,
            r#"{"accelerator":"metal","ref":"master","jobs":16,"wipe":true}"#,
        )
        .unwrap();
        assert_eq!(ok.reference, RefChoice::Master);
        assert_eq!(ok.parallel, Some(16));
        assert!(ok.wipe_build_dir && !ok.leave_headroom);

        for (json, why) in [
            (
                r#"{"accelerator":"metal","ref":"master","jobs":17}"#,
                "jobs above 2x CPUs",
            ),
            (
                r#"{"accelerator":"metal","ref":"master","jobs":0}"#,
                "zero jobs",
            ),
            (
                r#"{"accelerator":"metal","ref":"master","jobs":2,"headroom":true}"#,
                "jobs and headroom",
            ),
            (r#"{"accelerator":"cuda","ref":"master"}"#, "cuda on macOS"),
            (
                r#"{"accelerator":"vulkan","ref":"master"}"#,
                "unknown accelerator",
            ),
            (r#"{"accelerator":"metal","ref":"v1.2.3"}"#, "unknown ref"),
            (
                r#"{"accelerator":"metal","ref":"master","cuda_arch":"89"}"#,
                "cuda_arch on metal",
            ),
        ] {
            assert!(body_for(m, 8, json).is_err(), "{why}");
        }
        // Boundary: exactly twice the CPUs is allowed; a single CPU host allows 2.
        assert!(body_for(m, 1, r#"{"accelerator":"cpu","ref":"master","jobs":2}"#).is_ok());
        assert!(body_for(m, 1, r#"{"accelerator":"cpu","ref":"master","jobs":3}"#).is_err());
    }

    #[test]
    fn cuda_arch_is_restricted_to_its_charset_and_length() {
        let w = TargetOs::Windows;
        let with = |arch: &str| {
            let body = json!({ "accelerator": "cuda", "ref": "master", "cuda_arch": arch });
            build_options(serde_json::from_value(body).unwrap(), w, 8)
        };
        assert_eq!(
            with("120a-real").unwrap().cuda_architectures.as_deref(),
            Some("120a-real")
        );
        assert_eq!(
            with("89;90a").unwrap().cuda_architectures.as_deref(),
            Some("89;90a")
        );
        assert!(with("").unwrap().cuda_architectures.is_none());
        for bad in [
            "89 && calc",
            "89$(id)",
            "89|90",
            "89\\90",
            "8 9",
            "ABC",
            &"9".repeat(33),
        ] {
            assert!(with(bad).is_err(), "{bad}");
        }
        // Passes the charset but is not a CMake architecture list.
        assert!(with("8.9").is_err());
    }

    #[tokio::test]
    async fn start_rejects_malformed_and_invalid_requests_with_400() {
        let rig = Rig::new(TargetOs::Macos);
        for body in [
            "not json",
            "{}",
            r#"{"accelerator":"metal"}"#,
            r#"{"accelerator":"metal","ref":"master","jobs":-1}"#,
            r#"{"accelerator":"metal","ref":"master","jobs":1.5}"#,
            r#"{"accelerator":"metal","ref":"master","jobs":17}"#,
            r#"{"accelerator":"metal","ref":"master","cuda_arch":"89"}"#,
            r#"{"accelerator":"cuda","ref":"master"}"#,
        ] {
            let (code, resp) = rig.start(body).await;
            assert_eq!(code, 400, "{body}: {resp}");
            assert_eq!(resp["ok"], false);
        }
        assert!(rig.state.llama_build.active_job_id().is_none());
        assert!(rig.state.llama_build.is_idle());
    }

    #[tokio::test]
    async fn a_failed_preflight_returns_the_job_with_no_pinned_commit() {
        let rig = Rig::new(TargetOs::Macos);
        rig.ctl.prereqs_fail.store(true, Ordering::SeqCst);
        let (code, body) = rig.start(START_METAL).await;
        assert_eq!(code, 202);
        assert!(body["resolved_commit"].is_null());
        let id = body["job_id"].as_str().unwrap();
        let job = rig.wait_status(id, "failed").await;
        assert_eq!(job["failure"]["phase"], "preflight");
        assert_eq!(job["cancellable"], false);
        assert_eq!(job["restart_failed"], false);
        assert!(job["resolved"].is_null());
        // The live runtime was never touched.
        assert_eq!(
            std::fs::read(&rig.config.llama_server_path).unwrap(),
            b"live"
        );
        let (_, status) = rig.get("/api/llama-build/status").await;
        assert_eq!(status["last_job"]["id"], id);
        assert_eq!(status["last_job"]["failure_phase"], "preflight");
        assert!(status["active_job_id"].is_null());
    }

    #[tokio::test]
    async fn a_running_build_pins_the_commit_owns_the_tree_and_can_be_cancelled() {
        let rig = Rig::new(TargetOs::Macos);
        rig.ctl.hold_fetch.store(true, Ordering::SeqCst);
        let (code, body) = rig.start(START_METAL).await;
        assert_eq!(code, 202, "{body}");
        assert_eq!(body["ok"], true);
        assert_eq!(body["resolved_commit"], COMMIT);
        let id = body["job_id"].as_str().unwrap().to_string();
        assert_eq!(id.len(), 32);

        // Single flight: the second start names the owner.
        let (code, other) = rig.start(START_METAL).await;
        assert_eq!(code, 409);
        assert_eq!(other["owner"], "build");
        assert_eq!(other["owner_id"], id);
        assert_eq!(other["ok"], false);
        // The tree is owned: clean, validate and install refuse.
        let (code, other) = rig.call("DELETE", "/api/llama-build/cache", None).await;
        assert_eq!((code, &other["owner"]), (409, &json!("build")));
        let (code, other) = rig
            .call(
                "POST",
                "/api/llama-build/prereqs/validate",
                Some(r#"{"accelerator":"metal"}"#),
            )
            .await;
        assert_eq!((code, &other["owner"]), (409, &json!("build")));
        let (code, other) = rig
            .call(
                "POST",
                "/api/llama-build/prereqs/fix",
                Some(r#"{"action":"brew_install","packages":["cmake"]}"#),
            )
            .await;
        assert_eq!((code, &other["owner"]), (409, &json!("build")));
        let (_, status) = rig.get("/api/llama-build/status").await;
        assert_eq!(status["active_job_id"], id);
        assert_eq!(status["owner"], "build");
        let (_, cache) = rig.get("/api/llama-build/cache").await;
        assert_eq!(cache["owner"], "build");

        let (code, job) = rig.get(&format!("/api/llama-build/jobs/{id}")).await;
        assert_eq!(code, 200);
        assert_eq!(job["status"], "running");
        assert_eq!(job["phase"], "fetch");
        assert_eq!(job["cancellable"], true);
        assert_eq!(job["resolved"]["commit"], COMMIT);
        assert_eq!(job["resolved"]["reference"], "b11727");
        assert_eq!(job["accelerator"], "metal");
        assert_eq!(job["options"]["ref"], "latest_release");
        assert_eq!(job["options"]["headroom"], false);
        assert!(job["started_at"].as_u64().unwrap() > 0);
        assert!(job["finished_at"].is_null());
        assert!(job["log_cursor"].as_u64().unwrap() > 0);

        let (code, cancel) = rig
            .call("POST", &format!("/api/llama-build/jobs/{id}/cancel"), None)
            .await;
        assert_eq!((code, &cancel["ok"]), (200, &json!(true)));
        let job = rig.wait_status(&id, "cancelled").await;
        assert_eq!(job["cancellable"], false);
        assert!(
            job["outcome"]
                .as_str()
                .unwrap()
                .contains("Nothing was installed")
        );
        assert!(job["finished_at"].as_u64().is_some());
        assert_eq!(
            std::fs::read(&rig.config.llama_server_path).unwrap(),
            b"live"
        );

        // Ownership is released only after the job is terminal.
        for _ in 0..200 {
            if rig.state.llama_build.is_idle() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let (code, _) = rig.call("DELETE", "/api/llama-build/cache", None).await;
        assert_eq!(code, 200);
        // Cancelling a finished job is a conflict, not a success.
        let (code, again) = rig
            .call("POST", &format!("/api/llama-build/jobs/{id}/cancel"), None)
            .await;
        assert_eq!(code, 409);
        assert_eq!(again["cancellable"], false);
    }

    #[tokio::test]
    async fn cancel_is_refused_from_the_promote_phase_on() {
        let rig = Rig::new(TargetOs::Macos);
        let id = "b2".repeat(16);
        let options = BuildOptions::new(Accelerator::Metal, RefChoice::LatestTag);
        let shared = new_shared(&rig.deps(), &options, &id, true).unwrap();
        let cancel = Arc::new(AtomicBool::new(false));
        rig.state
            .llama_build
            .register_for_test(&id, shared.clone(), cancel.clone());

        let (_, job) = rig.get(&format!("/api/llama-build/jobs/{id}")).await;
        assert_eq!(job["cancellable"], true, "preflight is cancellable");

        set_phase_public(&shared, Phase::Promote);
        let (_, job) = rig.get(&format!("/api/llama-build/jobs/{id}")).await;
        assert_eq!(job["phase"], "promote");
        assert_eq!(job["cancellable"], false);
        let (code, body) = rig
            .call("POST", &format!("/api/llama-build/jobs/{id}/cancel"), None)
            .await;
        assert_eq!(code, 409);
        assert_eq!(body["ok"], false);
        assert_eq!(body["cancellable"], false);
        assert_eq!(
            body["error"],
            "Installing/recovering — cannot cancel safely"
        );
        assert!(
            !cancel.load(Ordering::SeqCst),
            "the flag must not be raised"
        );
    }

    #[tokio::test]
    async fn job_ids_never_reach_the_filesystem_unless_they_are_exact_ids() {
        let rig = Rig::new(TargetOs::Macos);
        let outside = rig.config.app_paths.root.join("secret.json");
        std::fs::write(&outside, b"{}").unwrap();
        for id in [
            "abc",
            "..",
            "%2e%2e%2fsecret",
            "A1A1A1A1A1A1A1A1A1A1A1A1A1A1A1A1",
            "../../secret",
        ] {
            for (method, path) in [
                ("GET", format!("/api/llama-build/jobs/{id}")),
                ("GET", format!("/api/llama-build/jobs/{id}/log")),
                ("POST", format!("/api/llama-build/jobs/{id}/cancel")),
            ] {
                let (code, body) = rig.call(method, &path, None).await;
                assert!(code == 400 || code == 404, "{method} {path} -> {code}");
                assert_ne!(body["ok"], true);
            }
        }
        let missing = "c3".repeat(16);
        let (code, _) = rig.get(&format!("/api/llama-build/jobs/{missing}")).await;
        assert_eq!(code, 404);
        let (code, _) = rig
            .get(&format!("/api/llama-build/jobs/{missing}/log"))
            .await;
        assert_eq!(code, 404);
        let (code, _) = rig
            .call(
                "POST",
                &format!("/api/llama-build/jobs/{missing}/cancel"),
                None,
            )
            .await;
        assert_eq!(code, 404);
    }

    #[tokio::test]
    async fn the_log_is_paged_by_line_count_and_survives_a_restart() {
        let rig = Rig::new(TargetOs::Macos);
        let id = "d4".repeat(16);
        let lines: Vec<String> = (0..2300).map(|i| format!("compile {i}")).collect();
        rig.disk_job(&id, &lines, Some(JobStatus::Succeeded));

        let (code, page) = rig.get(&format!("/api/llama-build/jobs/{id}/log")).await;
        assert_eq!(code, 200);
        assert_eq!(page["ok"], true);
        assert_eq!(page["lines"][0], "compile 0");
        assert_eq!(page["next"], 2000);
        assert_eq!(page["truncated"], false);

        let (_, page) = rig
            .get(&format!("/api/llama-build/jobs/{id}/log?after=2000"))
            .await;
        assert_eq!(page["lines"].as_array().unwrap().len(), 300);
        assert_eq!(page["lines"][0], "compile 2000");
        assert_eq!(page["next"], 2300);

        // Fully caught up: nothing new, cursor unchanged.
        let (_, page) = rig
            .get(&format!("/api/llama-build/jobs/{id}/log?after=2300"))
            .await;
        assert!(page["lines"].as_array().unwrap().is_empty());
        assert_eq!(page["next"], 2300);

        let (code, _) = rig
            .get(&format!("/api/llama-build/jobs/{id}/log?after=-1"))
            .await;
        assert_eq!(code, 400);
        let (code, _) = rig
            .get(&format!("/api/llama-build/jobs/{id}/log?after=abc"))
            .await;
        assert_eq!(code, 400);

        let (_, job) = rig.get(&format!("/api/llama-build/jobs/{id}")).await;
        assert_eq!(job["log_cursor"], 2300);
        assert_eq!(job["status"], "succeeded");
        assert_eq!(job["cancellable"], false);
    }

    #[tokio::test]
    async fn a_record_left_running_by_a_dead_process_reads_as_interrupted() {
        let rig = Rig::new(TargetOs::Macos);
        let id = "e5".repeat(16);
        rig.disk_job(&id, &["one".to_string()], None);
        let (_, job) = rig.get(&format!("/api/llama-build/jobs/{id}")).await;
        assert_eq!(job["status"], "interrupted");
        assert_eq!(job["cancellable"], false);
        let (_, status) = rig.get("/api/llama-build/status").await;
        assert_eq!(status["last_job"]["status"], "interrupted");
    }

    #[tokio::test]
    async fn the_cache_reports_sizes_and_clean_spares_the_runtime_and_source_by_default() {
        let rig = Rig::new(TargetOs::Macos);
        let paths = rig.paths();
        let write = |dir: std::path::PathBuf, bytes: usize| {
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("f"), vec![0u8; bytes]).unwrap();
        };
        write(paths.source.clone(), 100);
        write(paths.ccache.clone(), 200);
        write(paths.build_tree(Accelerator::Metal), 300);
        write(rig.config.app_paths.bin_previous_dir(), 50);

        let (code, cache) = rig.get("/api/llama-build/cache").await;
        assert_eq!(code, 200);
        assert_eq!(cache["source_bytes"], 100);
        assert_eq!(cache["ccache_bytes"], 200);
        assert_eq!(cache["ccache_cap_bytes"], 2u64 << 30);
        assert_eq!(
            cache["build_trees"],
            json!([{ "accelerator": "metal", "bytes": 300 }])
        );
        assert_eq!(cache["previous_bytes"], 50);
        assert!(cache["owner"].is_null());

        let (code, _) = rig
            .call(
                "DELETE",
                "/api/llama-build/cache?include_source=maybe",
                None,
            )
            .await;
        assert_eq!(code, 400);
        let (code, done) = rig.call("DELETE", "/api/llama-build/cache", None).await;
        assert_eq!(code, 200);
        assert_eq!(done["freed_bytes"], 500);
        assert_eq!(done["removed"].as_array().unwrap().len(), 2);
        assert!(paths.source.exists());
        assert!(!paths.ccache.exists());
        assert!(!paths.build_tree(Accelerator::Metal).exists());
        // Never the live or previous runtime.
        assert_eq!(
            std::fs::read(&rig.config.llama_server_path).unwrap(),
            b"live"
        );
        assert!(rig.config.app_paths.bin_previous_dir().join("f").exists());

        let (code, done) = rig
            .call("DELETE", "/api/llama-build/cache?include_source=true", None)
            .await;
        assert_eq!(code, 200);
        assert_eq!(done["freed_bytes"], 100);
        assert!(!paths.source.exists());
    }

    #[tokio::test]
    async fn the_cache_cap_is_finite_validated_and_persisted() {
        let rig = Rig::new(TargetOs::Macos);
        let put = |body: &'static str| {
            let rig = &rig;
            async move {
                rig.call("PUT", "/api/llama-build/cache-cap", Some(body))
                    .await
            }
        };
        for bad in [
            r#"{"ccache_max_bytes":0}"#,
            r#"{"ccache_max_bytes":268435455}"#,
            r#"{"ccache_max_bytes":17179869185}"#,
            r#"{"ccache_max_bytes":18446744073709551615}"#,
            r#"{"ccache_max_bytes":-5}"#,
            r#"{"ccache_max_bytes":1e9}"#,
            r#"{"ccache_max_bytes":1073741824.5}"#,
            r#"{"ccache_max_bytes":"1073741824"}"#,
            r#"{"ccache_max_bytes":null}"#,
            r#"{}"#,
            r#"nope"#,
        ] {
            let (code, body) = put(bad).await;
            assert_eq!(code, 400, "{bad}: {body}");
            assert_eq!(body["ok"], false);
        }
        assert!(!rig.config.config_dir.join("llama-build.json").exists());

        for (bytes, label) in [
            (256u64 << 20, "min"),
            (16u64 << 30, "max"),
            (5u64 << 30, "mid"),
        ] {
            let (code, body) = rig
                .call(
                    "PUT",
                    "/api/llama-build/cache-cap",
                    Some(&format!(r#"{{"ccache_max_bytes":{bytes}}}"#)),
                )
                .await;
            assert_eq!(code, 200, "{label}: {body}");
            assert_eq!(body["ccache_cap_bytes"], bytes);
            assert_eq!(
                LlamaBuildConfig::load(&rig.config.config_dir)
                    .unwrap()
                    .ccache_max_bytes,
                bytes
            );
        }
        // The new cap shows up without a restart, in the cache and options views.
        let (_, cache) = rig.get("/api/llama-build/cache").await;
        assert_eq!(cache["ccache_cap_bytes"], 5u64 << 30);
        let (_, options) = rig.get("/api/llama-build/options").await;
        assert_eq!(options["ccache_cap_bytes"], 5u64 << 30);
    }

    #[tokio::test]
    async fn the_cache_cap_can_be_changed_while_a_build_runs() {
        let rig = Rig::new(TargetOs::Macos);
        rig.ctl.hold_fetch.store(true, Ordering::SeqCst);
        let (code, body) = rig.start(START_METAL).await;
        assert_eq!(code, 202);
        let (code, _) = rig
            .call(
                "PUT",
                "/api/llama-build/cache-cap",
                Some(r#"{"ccache_max_bytes":1073741824}"#),
            )
            .await;
        assert_eq!(code, 200);
        let id = body["job_id"].as_str().unwrap();
        rig.call("POST", &format!("/api/llama-build/jobs/{id}/cancel"), None)
            .await;
        rig.wait_status(id, "cancelled").await;
    }
}
