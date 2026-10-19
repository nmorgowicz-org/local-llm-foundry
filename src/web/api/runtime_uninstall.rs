//! Runtime uninstall and storage-size routes.
//!
//! One contract for every managed runtime loader: report the bytes a runtime
//! occupies, and remove the runtime itself on request. Downloaded models are
//! never touched — they live outside the runtime roots and are the expensive
//! part users want to keep.

use std::path::Path;

use warp::Filter;
use warp::http::StatusCode;

use super::common::{check_api_token, check_db_admin_token};
use super::{ApiCtx, ApiReply, ApiRoute, unauthorized_api_token, unauthorized_db_admin_token};
use crate::inference::rapid_mlx::updater::RapidMlxRuntimeManager;
use crate::llama_runtime::gate::RuntimeOp;
use crate::llama_runtime::host::LiveHost;
use crate::llama_runtime::promote::{RuntimePaths, uninstall};

/// Typed confirmations for the two destructive uninstall flows. Install and
/// upgrade already demand a db-admin token plus an exact confirm string;
/// uninstall removes more (whole runtimes), so it must be gated at least as
/// strongly.
pub(crate) const RAPID_MLX_UNINSTALL_CONFIRM: &str = "UNINSTALL_RAPID_MLX";
pub(crate) const LLAMA_CPP_UNINSTALL_CONFIRM: &str = "UNINSTALL_LLAMA_CPP";

#[derive(Debug, serde::Deserialize)]
pub(crate) struct UninstallRequest {
    #[serde(default)]
    pub(crate) confirm: String,
}

fn json_error(status: StatusCode, message: impl Into<String>) -> ApiReply {
    Box::new(warp::reply::with_status(
        warp::reply::json(&serde_json::json!({
            "ok": false,
            "error": message.into(),
        })),
        status,
    ))
}

/// Best-effort recursive byte size of `dir` (0 when absent).
fn dir_size(dir: &Path) -> u64 {
    fn walk(dir: &Path, acc: &mut u64) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            match entry.file_type() {
                Ok(t) if t.is_dir() => walk(&path, acc),
                Ok(_) => {
                    if let Ok(meta) = entry.metadata() {
                        *acc += meta.len();
                    }
                }
                Err(_) => {}
            }
        }
    }
    let mut total = 0;
    walk(dir, &mut total);
    total
}

fn llama_bin_paths(config: &crate::config::AppConfig) -> Vec<std::path::PathBuf> {
    let bin = config.app_paths.bin_dir();
    let mut paths = vec![bin.clone()];
    // Updater rollback backups live beside the bin dir: bin-previous, plus
    // timestamped `bin-previous-{tag}-{pid}-{stamp}` siblings from sweeps.
    if let Some(parent) = bin.parent()
        && let Ok(entries) = std::fs::read_dir(parent)
    {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with("bin-previous") {
                paths.push(entry.path());
            }
        }
    }
    paths.retain(|p| p.exists());
    paths
}

/// `GET /api/runtimes/storage` — bytes occupied by each managed runtime.
fn storage_route(ctx: ApiCtx) -> ApiRoute {
    let config = ctx.config.clone();
    warp::path!("api" / "runtimes" / "storage")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let config = config.clone();
            async move {
                if !check_api_token(&auth, &config) {
                    return Ok(unauthorized_api_token());
                }
                let rapid_bytes = match RapidMlxRuntimeManager::new(&config.config_dir) {
                    Ok(manager) => manager.storage_bytes(),
                    Err(message) => {
                        return Ok::<ApiReply, warp::Rejection>(json_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            message.to_string(),
                        ));
                    }
                };
                let llama_paths = llama_bin_paths(&config);
                let llama_bytes: u64 = llama_paths.iter().map(|p| dir_size(p)).sum();
                Ok::<ApiReply, warp::Rejection>(Box::new(warp::reply::json(&serde_json::json!({
                    "ok": true,
                    "rapid_mlx_bytes": rapid_bytes,
                    "llama_bin_bytes": llama_bytes,
                }))))
            }
        })
        .boxed()
}

/// `DELETE /api/rapid-mlx/runtime/uninstall` — remove every managed
/// Rapid-MLX environment and the active pointer. Downloaded models are kept.
fn rapid_uninstall_route(ctx: ApiCtx) -> ApiRoute {
    let config = ctx.config.clone();
    let state = ctx.state.clone();
    warp::path!("api" / "rapid-mlx" / "runtime" / "uninstall")
        .and(warp::delete())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<UninstallRequest>())
        .and_then(move |auth: Option<String>, request: UninstallRequest| {
            let config = config.clone();
            let state = state.clone();
            async move {
                if !check_db_admin_token(&auth, &config) {
                    return Ok(unauthorized_db_admin_token());
                }
                if request.confirm != RAPID_MLX_UNINSTALL_CONFIRM {
                    return Ok::<ApiReply, warp::Rejection>(json_error(
                        StatusCode::BAD_REQUEST,
                        format!(
                            "uninstall requires the exact confirmation string {RAPID_MLX_UNINSTALL_CONFIRM}"
                        ),
                    ));
                }
                if *state.local_server_running.lock().unwrap() {
                    return Ok(json_error(
                        StatusCode::CONFLICT,
                        "A server is running; stop it before uninstalling the runtime",
                    ));
                }
                // A fresh manager is fine here: the mutation gate is shared
                // process-wide per runtime root, so this uninstall still
                // cannot race a concurrent install/upgrade.
                let manager = match RapidMlxRuntimeManager::new(&config.config_dir) {
                    Ok(manager) => manager,
                    Err(message) => {
                        return Ok::<ApiReply, warp::Rejection>(json_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            message.to_string(),
                        ));
                    }
                };
                match tokio::task::spawn_blocking(move || manager.uninstall_all()).await {
                    Ok(Ok(())) => Ok(Box::new(warp::reply::json(&serde_json::json!({
                        "ok": true,
                    })))),
                    Ok(Err(message)) => Ok(json_error(StatusCode::CONFLICT, message.to_string())),
                    Err(_) => Ok(json_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Rapid-MLX uninstall task failed",
                    )),
                }
            }
        })
        .boxed()
}

/// `DELETE /api/llama-binary/uninstall` — remove the managed llama.cpp
/// binaries and rollback backups. Downloaded models are kept.
fn llama_uninstall_route(ctx: ApiCtx) -> ApiRoute {
    let config = ctx.config.clone();
    let state = ctx.state.clone();
    warp::path!("api" / "llama-binary" / "uninstall")
        .and(warp::delete())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<UninstallRequest>())
        .and_then(move |auth: Option<String>, request: UninstallRequest| {
            let config = config.clone();
            let state = state.clone();
            async move {
                if !check_db_admin_token(&auth, &config) {
                    return Ok(unauthorized_db_admin_token());
                }
                if request.confirm != LLAMA_CPP_UNINSTALL_CONFIRM {
                    return Ok::<ApiReply, warp::Rejection>(json_error(
                        StatusCode::BAD_REQUEST,
                        format!(
                            "uninstall requires the exact confirmation string {LLAMA_CPP_UNINSTALL_CONFIRM}"
                        ),
                    ));
                }
                if *state.local_server_running.lock().unwrap() {
                    return Ok(json_error(
                        StatusCode::CONFLICT,
                        "A server is running; stop it before uninstalling the binaries",
                    ));
                }
                let paths = llama_bin_paths(&config);
                if paths.is_empty() {
                    return Ok::<ApiReply, warp::Rejection>(json_error(
                        StatusCode::NOT_FOUND,
                        "No managed llama.cpp binaries are installed",
                    ));
                }
                // One gate for every runtime mutation: an update or rollback in
                // flight makes this a 409 naming the owner, never a race.
                let mut guard = match state.runtime_gate.try_begin(RuntimeOp::Uninstall) {
                    Ok(guard) => guard,
                    Err(busy) => {
                        return Ok(json_error(
                            StatusCode::CONFLICT,
                            format!(
                                "The llama.cpp runtime is busy ({} in progress)",
                                busy.owner.label()
                            ),
                        ));
                    }
                };
                let runtime = match RuntimePaths::for_live_dir(&config.app_paths.bin_dir()) {
                    Ok(p) => p,
                    Err(e) => {
                        return Ok(json_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()));
                    }
                };
                let exe = if cfg!(windows) { "llama-server.exe" } else { "llama-server" };
                let removed = usize::from(runtime.live.is_dir()) + usize::from(runtime.previous.is_dir());
                if removed > 0 {
                    let host = LiveHost::without_model(state.clone(), config.clone());
                    match uninstall(&host, &mut guard, &runtime, exe).await {
                        Ok(_) => {}
                        Err(e) => {
                            return Ok(json_error(StatusCode::CONFLICT, e.to_string()));
                        }
                    }
                }
                drop(guard);
                // Pre-journal leftovers (`bin-previous-<tag>-<pid>-<stamp>`) are
                // not part of any transaction; remove them here.
                let legacy = tokio::task::spawn_blocking(move || {
                    let mut count = 0usize;
                    for path in &paths {
                        let name = path.file_name().map(|n| n.to_string_lossy().into_owned());
                        let is_legacy = name.as_deref().is_some_and(|n| n.starts_with("bin-previous-"));
                        if is_legacy && path.exists() {
                            if path.is_dir() {
                                std::fs::remove_dir_all(path)?;
                            } else {
                                std::fs::remove_file(path)?;
                            }
                            count += 1;
                        }
                    }
                    Ok::<usize, std::io::Error>(count)
                })
                .await;
                match legacy {
                    Ok(Ok(count)) => Ok(Box::new(warp::reply::json(&serde_json::json!({
                        "ok": true,
                        "removed": removed + count,
                    })))),
                    Ok(Err(e)) => Ok(json_error(
                        StatusCode::CONFLICT,
                        format!("Cannot remove leftover runtime backups: {e}"),
                    )),
                    Err(_) => Ok(json_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "llama.cpp uninstall task failed",
                    )),
                }
            }
        })
        .boxed()
}

pub fn routes(ctx: ApiCtx) -> ApiRoute {
    storage_route(ctx.clone())
        .or(rapid_uninstall_route(ctx.clone()))
        .unify()
        .or(llama_uninstall_route(ctx))
        .unify()
        .boxed()
}
