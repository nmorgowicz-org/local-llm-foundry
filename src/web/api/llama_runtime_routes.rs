//! Runtime-state routes for the managed llama.cpp binaries.
//!
//! - `POST /api/llama-binary/rollback` swaps the live and previous sets under
//!   the same gate and journal as an update.
//! - `GET  /api/llama-binary/runtime-status` reports what the panel needs:
//!   whether a rollback is possible, what an interrupted transaction left, and
//!   who currently owns the runtime gate.
//!
//! Both are authenticated. Neither holds the gate across anything but the swap.

use std::sync::Arc;

use warp::Filter;

use super::llama_binary::{
    PromoteReplyContext, json_busy, json_failure, managed_bin_dir, promote_reply,
};
use super::{ApiCtx, ApiRoute, box_reply, check_api_token, unauthorized_api_token};
use crate::config::AppConfig;
use crate::llama_runtime::gate::RuntimeOp;
use crate::llama_runtime::host::LiveHost;
use crate::llama_runtime::manifest::RuntimeManifest;
use crate::llama_runtime::promote::{
    Journal, JournalState, RuntimePaths, last_transaction, read_journal, rollback,
};
use crate::state::AppState;

fn exe_name() -> &'static str {
    if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    }
}

fn describe(dir: &std::path::Path) -> serde_json::Value {
    if !dir.is_dir() {
        return serde_json::Value::Null;
    }
    let label = RuntimeManifest::read(dir).ok().flatten().map(|m| m.label);
    serde_json::json!({ "label": label })
}

fn journal_summary(j: &Journal) -> serde_json::Value {
    serde_json::json!({
        "id": j.id,
        "op": j.op,
        "label": j.label,
        "started_unix": j.started_unix,
        "outcome": j.outcome,
    })
}

/// Pure projection of on-disk runtime state for the status route.
pub(super) fn runtime_status(config: &AppConfig, owner: Option<RuntimeOp>) -> serde_json::Value {
    let (bin_dir, custom) = managed_bin_dir(config);
    let paths = match RuntimePaths::for_live_dir(&bin_dir) {
        Ok(p) => p,
        Err(e) => return serde_json::json!({ "ok": false, "error": e.to_string() }),
    };
    let transaction = match read_journal(&paths) {
        JournalState::None => serde_json::json!({ "state": "clean" }),
        JournalState::Present(j) => {
            serde_json::json!({ "state": "pending", "journal": journal_summary(&j) })
        }
        JournalState::Corrupt(reason) => {
            serde_json::json!({ "state": "blocked", "reason": reason })
        }
    };
    serde_json::json!({
        "ok": true,
        "busy_owner": owner,
        "live": describe(&paths.live),
        "previous": describe(&paths.previous),
        "previous_available": paths.previous.is_dir(),
        "transaction": transaction,
        "last": last_transaction(&paths).map(|j| journal_summary(&j)),
        "custom_server_path": custom.map(|p| p.display().to_string()),
    })
}

fn status_route(ctx: ApiCtx) -> ApiRoute {
    let config = ctx.config.clone();
    let state = ctx.state.clone();
    warp::path!("api" / "llama-binary" / "runtime-status")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let config = config.clone();
            let state = state.clone();
            async move {
                if !check_api_token(&auth, &config) {
                    return Ok(unauthorized_api_token());
                }
                let owner = state.runtime_gate.current_owner();
                let body = tokio::task::spawn_blocking(move || runtime_status(&config, owner))
                    .await
                    .unwrap_or_else(|e| serde_json::json!({ "ok": false, "error": e.to_string() }));
                Ok::<_, warp::Rejection>(box_reply(warp::reply::json(&body)))
            }
        })
        .boxed()
}

async fn do_rollback(state: AppState, config: Arc<AppConfig>) -> super::llama_binary::ReplyBox {
    let (bin_dir, custom) = managed_bin_dir(&config);
    let paths = match RuntimePaths::for_live_dir(&bin_dir) {
        Ok(p) => p,
        Err(e) => return json_failure(format!("Invalid runtime location: {e}")),
    };
    let mut guard = match state.runtime_gate.try_begin(RuntimeOp::Rollback) {
        Ok(guard) => guard,
        Err(busy) => return json_busy(busy),
    };
    let host = if custom.is_some() {
        LiveHost::without_model(state.clone(), config.clone())
    } else {
        LiveHost::new(state.clone(), config.clone())
    };
    let restart_applicable = host.restart_applicable();
    state.push_log("[monitor] llama-binary/rollback: swapping to the previous runtime".into());
    let result = rollback(&host, &mut guard, &paths, exe_name()).await;
    drop(guard);
    let label = RuntimeManifest::read(&paths.live)
        .ok()
        .flatten()
        .map(|m| m.label)
        .unwrap_or_else(|| "previous".to_string());
    promote_reply(
        &state,
        result,
        PromoteReplyContext {
            tag: &label,
            backend: "",
            arch: "",
            sha256: None,
            restart_applicable,
            custom_server_path: custom.map(|p| p.display().to_string()),
        },
    )
}

fn rollback_route(ctx: ApiCtx) -> ApiRoute {
    let config = ctx.config.clone();
    let state = ctx.state.clone();
    warp::path!("api" / "llama-binary" / "rollback")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let config = config.clone();
            let state = state.clone();
            async move {
                if !check_api_token(&auth, &config) {
                    return Ok(unauthorized_api_token());
                }
                Ok::<_, warp::Rejection>(do_rollback(state, config).await)
            }
        })
        .boxed()
}

pub(crate) fn routes(ctx: ApiCtx) -> ApiRoute {
    status_route(ctx.clone())
        .or(rollback_route(ctx))
        .unify()
        .boxed()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_in(root: &std::path::Path) -> AppConfig {
        use clap::Parser;
        let mut config = AppConfig::from_args_pure(crate::cli::AppArgs::parse_from([
            "llama-monitor",
            "--llama-server-path",
            "llama-server",
            "--gpu-backend",
            "none",
        ]));
        config.app_paths = crate::paths::AppPaths::from_root(root.to_path_buf());
        config.llama_server_path = config.app_paths.bin_dir().join(exe_name());
        // An existing binary keeps `managed_llama_server_path` from falling back
        // to the developer's real canonical/legacy roots.
        std::fs::create_dir_all(config.app_paths.bin_dir()).unwrap();
        std::fs::write(&config.llama_server_path, b"x").unwrap();
        config
    }

    #[test]
    fn status_reports_previous_availability_and_clean_journal() {
        let root = tempfile::tempdir().unwrap();
        let config = config_in(root.path());
        let status = runtime_status(&config, None);
        assert_eq!(status["previous_available"], false);
        assert_eq!(status["transaction"]["state"], "clean");
        assert!(status["busy_owner"].is_null());

        std::fs::create_dir_all(config.app_paths.bin_previous_dir()).unwrap();
        let status = runtime_status(&config, Some(RuntimeOp::Update));
        assert_eq!(status["previous_available"], true);
        assert_eq!(status["busy_owner"], "update");
    }

    #[test]
    fn status_reports_a_corrupt_journal_as_blocked() {
        let root = tempfile::tempdir().unwrap();
        let config = config_in(root.path());
        std::fs::write(config.app_paths.runtime_transaction_file(), b"{not json").unwrap();
        let status = runtime_status(&config, None);
        assert_eq!(status["transaction"]["state"], "blocked");
    }

    #[test]
    fn a_custom_server_path_is_reported_and_installs_to_managed_bin() {
        let root = tempfile::tempdir().unwrap();
        let mut config = config_in(root.path());
        config.llama_server_path = root.path().join("elsewhere").join("my-server");
        let (bin, custom) = managed_bin_dir(&config);
        assert_eq!(bin, config.app_paths.bin_dir());
        assert_eq!(custom, Some(config.llama_server_path.clone()));
        let status = runtime_status(&config, None);
        assert!(status["custom_server_path"].is_string());
    }
}
