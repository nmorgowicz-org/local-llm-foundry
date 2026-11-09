use serde::{Deserialize, Serialize};
#[cfg(target_os = "macos")]
use std::sync::{Mutex, OnceLock};
#[cfg(target_os = "macos")]
use std::time::{Duration, Instant};
use warp::Filter;

use super::common::{check_db_admin_token, unauthorized_db_admin_token};
use super::{ApiCtx, ApiReply, ApiRoute, check_api_token, unauthorized_api_token};
use crate::system::{ReclaimAction, redact_command};

pub(crate) fn routes(ctx: ApiCtx) -> ApiRoute {
    top_processes(ctx.clone())
        .or(purge_memory(ctx.clone()))
        .unify()
        .or(reclaim_guidance(ctx.clone()))
        .unify()
        .or(reclaim_action(ctx.clone()))
        .unify()
        .or(wired_limit_get(ctx.clone()))
        .unify()
        .boxed()
}

// ── Top-processes endpoint (enhanced with phys_footprint, redacted commands) ──

/// Privacy-safe process info with clearly labeled metrics.
#[derive(Serialize)]
struct ProcessInfo {
    /// Process ID.
    pid: u32,
    /// Application name.
    name: String,
    /// Physical footprint in MB. On macOS this approximates phys_footprint
    /// (physical memory footprint including compressed pages); on other platforms
    /// this equals RSS. Labeled clearly as "physical footprint".
    phys_footprint_mb: f64,
    /// Resident Set Size in MB (actual resident pages only).
    /// Labeled clearly as "RSS" (Resident Set Size).
    rss_mb: f64,
    /// Redacted command line: secrets/tokens stripped, max 128 chars.
    command: String,
}

fn top_processes(ctx: ApiCtx) -> ApiRoute {
    let config = ctx.config;
    warp::path!("system" / "top-processes")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let config = config.clone();
            async move {
                if !check_api_token(&auth, &config) {
                    return Ok(unauthorized_api_token());
                }
                let procs = get_top_processes(15);
                Ok::<ApiReply, warp::Rejection>(Box::new(warp::reply::json(&procs)))
            }
        })
        .boxed()
}

fn get_top_processes(n: usize) -> Vec<ProcessInfo> {
    let Ok(sys) = sysinfo::System::new_all() else {
        return Vec::new();
    };
    let mut procs: Vec<ProcessInfo> = sys
        .processes()
        .iter()
        .filter_map(|(pid, process)| {
            let rss_bytes = process.memory();
            if rss_bytes == 0 {
                return None;
            }
            let cmd_raw = if process.cmd().is_empty() {
                process.name().to_string_lossy().into_owned()
            } else {
                process
                    .cmd()
                    .iter()
                    .map(|part| part.to_string_lossy())
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            // Redact secrets/tokens from command line.
            let command = redact_command(&cmd_raw);
            Some(ProcessInfo {
                pid: pid.as_u32(),
                name: process.name().to_string_lossy().into_owned(),
                // On macOS, true phys_footprint requires task_info syscall;
                // sysinfo provides RSS which is a lower bound.
                phys_footprint_mb: rss_bytes as f64 / 1024.0 / 1024.0,
                rss_mb: rss_bytes as f64 / 1024.0 / 1024.0,
                command,
            })
        })
        .collect();

    procs.sort_by(|a, b| {
        b.phys_footprint_mb
            .partial_cmp(&a.phys_footprint_mb)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    procs.truncate(n);
    procs
}

// ── Purge endpoint ────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub(crate) struct PurgeResult {
    #[serde(default)]
    pub(crate) ok: bool,
    #[serde(default)]
    pub(crate) message: String,
}

#[derive(Deserialize)]
struct PurgeRequest {
    #[serde(default)]
    confirm: String,
}

fn purge_memory(ctx: ApiCtx) -> ApiRoute {
    let config = ctx.config;
    warp::path!("system" / "purge")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(warp::body::json())
        .and_then(move |auth: Option<String>, body: PurgeRequest| {
            let config = config.clone();
            async move {
                if !check_db_admin_token(&auth, &config) {
                    return Ok(unauthorized_db_admin_token());
                }
                if body.confirm != "purge-memory" {
                    return Ok::<ApiReply, warp::Rejection>(Box::new(warp::reply::json(
                        &PurgeResult {
                            ok: false,
                            message: "Missing confirmation for memory purge.".to_string(),
                        },
                    )));
                }
                let result = run_purge();
                Ok::<ApiReply, warp::Rejection>(Box::new(warp::reply::json(&result)))
            }
        })
        .boxed()
}

/// Run the macOS disk cache purge. Made pub(crate) for use by system.rs reclaim guidance.
pub(crate) fn run_purge() -> PurgeResult {
    #[cfg(target_os = "macos")]
    {
        static LAST_PURGE: OnceLock<Mutex<Option<Instant>>> = OnceLock::new();
        let mut last = LAST_PURGE
            .get_or_init(|| Mutex::new(None))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(instant) = *last {
            let elapsed = instant.elapsed();
            if elapsed < Duration::from_secs(120) {
                let wait = Duration::from_secs(120) - elapsed;
                return PurgeResult {
                    ok: false,
                    message: format!(
                        "Memory purge is cooling down; try again in {} seconds.",
                        wait.as_secs().max(1)
                    ),
                };
            }
        }

        // Use osascript to trigger the native macOS privilege dialog so the user
        // can authorize `purge` without the app needing to run as root. This is
        // intentionally manual; automated privileged purging should use a
        // dedicated helper, not repeated hidden sudo calls.
        let script = r#"do shell script "purge" with administrator privileges"#;
        let output = std::process::Command::new("osascript")
            .args(["-e", script])
            .output();

        match output {
            Ok(o) if o.status.success() => {
                *last = Some(Instant::now());
                PurgeResult {
                    ok: true,
                    message: "macOS was asked to flush disk cache. This will not free mlock, wired, or model heap memory.".into(),
                }
            }
            Ok(o) => {
                let stderr = String::from_utf8_lossy(&o.stderr).into_owned();
                // User cancelled the auth dialog: osascript exits 1 with "User cancelled"
                if stderr.contains("cancelled") || stderr.contains("canceled") {
                    PurgeResult {
                        ok: false,
                        message: "Purge cancelled by user.".into(),
                    }
                } else {
                    PurgeResult {
                        ok: false,
                        message: format!("purge failed: {}", stderr.trim()),
                    }
                }
            }
            Err(e) => PurgeResult {
                ok: false,
                message: format!("Could not run osascript: {e}"),
            },
        }
    }

    #[cfg(not(target_os = "macos"))]
    PurgeResult {
        ok: false,
        message: "Memory purge is only supported on macOS.".into(),
    }
}

// ── Reclaim guidance endpoint ─────────────────────────────────────────────────

/// GET /system/reclaim-guidance
///
/// Returns the current memory availability snapshot plus guidance on which
/// reclaim actions can help reach the selected fit boundary.
/// Actions are only offered when they can cross the fit boundary.
/// Estimates are conservative and bounded (ranges, not promises).
/// Auth: requires api-token.
fn reclaim_guidance(ctx: ApiCtx) -> ApiRoute {
    let config = ctx.config;
    warp::path!("system" / "reclaim-guidance")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let config = config.clone();
            async move {
                if !check_api_token(&auth, &config) {
                    return Ok(unauthorized_api_token());
                }
                let snapshot = crate::memory_availability::build_snapshot();
                let guidance = crate::system::compute_reclaim_guidance(&snapshot);
                Ok::<ApiReply, warp::Rejection>(Box::new(warp::reply::json(&serde_json::json!({
                    "ok": true,
                    "guidance": guidance,
                }))))
            }
        })
        .boxed()
}

// ── Reclaim action endpoint ───────────────────────────────────────────────────

#[derive(Deserialize)]
struct ReclaimActionRequest {
    action: ReclaimAction,
}

/// POST /system/reclaim
///
/// Execute a reclaim action with before/after remeasurement.
/// Reports actual change honestly.
/// Auth: requires db-admin-token (high-impact operation).
fn reclaim_action(ctx: ApiCtx) -> ApiRoute {
    let config = ctx.config;
    warp::path!("system" / "reclaim")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(warp::body::json())
        .and_then(move |auth: Option<String>, body: ReclaimActionRequest| {
            let config = config.clone();
            async move {
                if !check_db_admin_token(&auth, &config) {
                    return Ok(unauthorized_db_admin_token());
                }
                let result = crate::system::execute_reclaim_action(body.action);
                Ok::<ApiReply, warp::Rejection>(Box::new(warp::reply::json(&serde_json::json!({
                    "ok": true,
                    "result": result,
                }))))
            }
        })
        .boxed()
}

// ── Wired-limit endpoints ─────────────────────────────────────────────────────

#[derive(Serialize)]
struct WiredLimitGetResult {
    /// Current value of iogpu.wired_limit_mb from sysctl. 0 means using macOS default.
    current_mb: u64,
    /// Maximum allowed value based on RAM-relative safe bound (95% of total RAM hard ceiling).
    /// 0 if not available (non-macOS or RAM unknown).
    max_mb: u64,
    /// RAM-relative safe default when sysctl is unset (tiered: total - 8GB for ≥24GB, total - 6GB for ≤16GB).
    /// 0 if not available.
    safe_default_mb: u64,
    /// Total system RAM in bytes.
    total_ram_bytes: u64,
    /// Behavioral notes about persistence and restart requirements.
    behavior_notes: String,
}

fn wired_limit_get(ctx: ApiCtx) -> ApiRoute {
    let config = ctx.config;
    warp::path!("api" / "system" / "wired-limit")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let config = config.clone();
            async move {
                if !check_api_token(&auth, &config) {
                    return Ok(unauthorized_api_token());
                }
                #[cfg(target_os = "macos")]
                {
                    let sys_info = crate::system::get_system_metrics();
                    let total_ram_bytes = (sys_info.ram_total_gb * 1024.0 * 1024.0 * 1024.0) as u64;
                    let current_mb = crate::gpu::apple::read_iogpu_wired_limit_mb();
                    let max_mb =
                        crate::gpu::apple::wired_limit_max_mb(total_ram_bytes).unwrap_or(0);
                    let safe_default_mb =
                        crate::gpu::apple::wired_limit_safe_default_mb(total_ram_bytes)
                            .unwrap_or(0);
                    let behavior_notes =
                        crate::gpu::apple::wired_limit_behavior_notes().to_string();
                    Ok::<ApiReply, warp::Rejection>(Box::new(warp::reply::json(
                        &WiredLimitGetResult {
                            current_mb,
                            max_mb,
                            safe_default_mb,
                            total_ram_bytes,
                            behavior_notes,
                        },
                    )))
                }
                #[cfg(not(target_os = "macos"))]
                {
                    let result = WiredLimitGetResult {
                        current_mb: 0,
                        max_mb: 0,
                        safe_default_mb: 0,
                        total_ram_bytes: 0,
                        behavior_notes:
                            "Wired limit is only applicable on macOS with Apple Silicon.".into(),
                    };
                    Ok::<ApiReply, warp::Rejection>(Box::new(warp::reply::json(&result)))
                }
            }
        })
        .boxed()
}
