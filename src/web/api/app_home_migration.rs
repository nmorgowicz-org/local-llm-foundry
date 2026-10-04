use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use warp::Filter;

use serde::Deserialize;

use crate::app_migration::{
    AppHomeRoots, default_roots, disposable_roots, inspect_application_roots,
    plan_application_home, plan_application_home_cleanup, plan_application_home_rollback,
    queue_application_home_cleanup, queue_application_home_migration,
    queue_application_home_rollback,
};

use super::common::{
    ApiCtx, ApiReply, ApiRoute, check_api_token, check_db_admin_token, unauthorized_api_token,
    unauthorized_db_admin_token,
};

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct QueueRequest {
    plan_id: String,
    confirmation: String,
}

fn migration_roots(config: &crate::config::AppConfig) -> Result<AppHomeRoots, String> {
    resolve_migration_roots(config).map_err(|error| error.to_string())
}

fn resolve_migration_roots(config: &crate::config::AppConfig) -> anyhow::Result<AppHomeRoots> {
    config
        .migration_test_root
        .as_deref()
        .map(disposable_roots)
        .transpose()
        .map(|roots| roots.unwrap_or_else(default_roots))
}

static NEXT_PREVIEW_ID: AtomicU64 = AtomicU64::new(1);

/// Request-local scope for the application's stderr logging convention.
/// IDs are process-local counters, not credentials or client-supplied values.
struct PreviewDiagnostics {
    id: u64,
    started: Instant,
}

impl PreviewDiagnostics {
    fn new() -> Self {
        Self {
            id: NEXT_PREVIEW_ID.fetch_add(1, Ordering::Relaxed),
            started: Instant::now(),
        }
    }

    fn log(&self, level: &str, stage: &str, fields: fmt::Arguments<'_>) {
        eprintln!(
            "[{level}] app_home_migration_preview preview_id={} elapsed_ms={} stage={stage} {fields}",
            self.id,
            self.started.elapsed().as_millis(),
        );
    }

    fn failure(&self, stage: &str, error: &anyhow::Error) {
        // Debug-quote the full display chain to keep paths/OS errors on one
        // log line. Never pass configuration or authorization values here.
        self.log(
            "error",
            stage,
            format_args!("outcome=failed error_chain={:?}", format!("{error:#}")),
        );
    }
}

/// Read-only migration status used by the non-blocking frontend migration
/// toast. It is authenticated because root paths and migration state are
/// private application metadata.
pub(crate) fn routes(ctx: ApiCtx) -> ApiRoute {
    let config = ctx.config.clone();
    let status = warp::path!("api" / "app-home-migration" / "status")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |authorization: Option<String>| {
            let config = config.clone();
            async move {
                if !check_api_token(&authorization, &config) {
                    return Ok::<ApiReply, warp::Rejection>(Box::new(unauthorized_api_token()));
                }
                let roots = migration_roots(&config).map_err(|error| {
                    warp::reject::custom(super::ApiError::new(
                        warp::http::StatusCode::BAD_REQUEST,
                        error,
                    ))
                })?;
                let inspection = inspect_application_roots(&roots)
                    .map_err(|error| warp::reject::custom(super::ApiError::migration(&error)))?;
                Ok(Box::new(warp::reply::json(&serde_json::json!({
                    "ok": true,
                    "state": inspection.state,
                    "canonical_root": inspection.canonical_root,
                    "legacy_root": inspection.legacy_root,
                    "active_root": inspection.active_root,
                    "migration_required": matches!(
                        inspection.state,
                        crate::app_migration::RootState::LegacyActive
                            | crate::app_migration::RootState::MigrationQueued
                    ),
                }))))
            }
        })
        .boxed();

    let preview = {
        let config = ctx.config.clone();
        warp::path!("api" / "app-home-migration" / "preview")
            .and(warp::get())
            .and(warp::header::optional::<String>("authorization"))
            .and_then(move |authorization: Option<String>| {
                let config = config.clone();
                async move {
                    if !check_api_token(&authorization, &config) {
                        return Ok::<ApiReply, warp::Rejection>(Box::new(unauthorized_api_token()));
                    }
                    let diagnostics = PreviewDiagnostics::new();
                    diagnostics.log("info", "start", format_args!("authenticated=true"));
                    let roots = resolve_migration_roots(&config).map_err(|error| {
                        diagnostics.failure("resolve_roots", &error);
                        warp::reject::custom(super::ApiError::new(
                            warp::http::StatusCode::BAD_REQUEST,
                            error.to_string(),
                        ))
                    })?;
                    diagnostics.log(
                        "info",
                        "resolve_roots",
                        format_args!(
                            "outcome=success disposable={} canonical_root={:?} legacy_root={:?}",
                            roots.disposable, roots.canonical, roots.legacy,
                        ),
                    );
                    let inspection = inspect_application_roots(&roots).map_err(|error| {
                        diagnostics.failure("inspect_roots", &error);
                        warp::reject::custom(super::ApiError::migration(&error))
                    })?;
                    diagnostics.log(
                        "info",
                        "inspect_roots",
                        format_args!("outcome=success state={:?}", inspection.state),
                    );
                    if !matches!(
                        inspection.state,
                        crate::app_migration::RootState::LegacyActive
                    ) {
                        diagnostics.log(
                            "info",
                            "complete",
                            format_args!(
                                "outcome=no_plan state={:?} reason=not_legacy_active",
                                inspection.state,
                            ),
                        );
                        return Ok(Box::new(warp::reply::json(&serde_json::json!({
                            "ok": true,
                            "state": inspection.state,
                            "plan": null,
                        }))));
                    }
                    diagnostics.log("info", "plan", format_args!("outcome=started"));
                    let plan =
                        plan_application_home(&inspection.legacy_root, &inspection.canonical_root)
                            .map_err(|error| {
                                diagnostics.failure("plan", &error);
                                warp::reject::custom(super::ApiError::migration(&error))
                            })?;
                    diagnostics.log(
                        "info",
                        "complete",
                        format_args!(
                            "outcome=planned state={:?} entry_count={} retained_entry_count={} required_copy_bytes={} total_seen_bytes={}",
                            inspection.state,
                            plan.entries.len(),
                            plan.retained_entries.len(),
                            plan.required_copy_bytes,
                            plan.total_seen_bytes,
                        ),
                    );
                    Ok(Box::new(warp::reply::json(&serde_json::json!({
                        "ok": true,
                        "state": inspection.state,
                        "plan": plan,
                    }))))
                }
            })
            .boxed()
    };

    let queue = {
        let config = ctx.config.clone();
        warp::path!("api" / "app-home-migration" / "queue")
            .and(warp::post())
            .and(warp::header::optional::<String>("authorization"))
            .and(warp::body::json())
            .and_then(move |authorization: Option<String>, body: QueueRequest| {
                let config = config.clone();
                async move {
                    let diagnostics = PreviewDiagnostics::new();
                    if !check_db_admin_token(&authorization, &config) {
                        diagnostics.log("warn", "queue_auth", format_args!("outcome=rejected"));
                        return Ok::<ApiReply, warp::Rejection>(Box::new(
                            unauthorized_db_admin_token(),
                        ));
                    }
                    diagnostics.log("info", "queue_start", format_args!("requested_plan_id={}", body.plan_id));
                    if body.confirmation != "MIGRATE TO LOCAL LLM FOUNDRY" {
                        return Ok(Box::new(warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({
                                "ok": false,
                                "error": "exact migration confirmation is required"
                            })),
                            warp::http::StatusCode::BAD_REQUEST,
                        )));
                    }
                    let roots = migration_roots(&config).map_err(|error| {
                        warp::reject::custom(super::ApiError::new(
                            warp::http::StatusCode::BAD_REQUEST,
                            error,
                        ))
                    })?;
                    let inspection = inspect_application_roots(&roots).map_err(|error| {
                        diagnostics.failure("queue_inspect_roots", &error);
                        warp::reject::custom(super::ApiError::migration(&error))
                    })?;
                    if !matches!(
                        inspection.state,
                        crate::app_migration::RootState::LegacyActive
                    ) {
                        return Ok(Box::new(warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "state": inspection.state,
                            "error": "application root is not in a migratable legacy-only state"
                        }))));
                    }
                    let plan =
                        plan_application_home(&inspection.legacy_root, &inspection.canonical_root)
                            .map_err(|error| {
                                diagnostics.failure("queue_plan", &error);
                                warp::reject::custom(super::ApiError::migration(&error))
                            })?;
                    if plan.plan_id != body.plan_id {
                        diagnostics.log(
                            "error",
                            "queue_stale_check",
                            format_args!(
                                "outcome=stale requested_plan_id={} current_plan_id={} entry_count={}",
                                body.plan_id,
                                plan.plan_id,
                                plan.entries.len(),
                            ),
                        );
                        return Ok(Box::new(warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({
                                "ok": false,
                                "error": "migration preview is stale; refresh and try again"
                            })),
                            warp::http::StatusCode::CONFLICT,
                        )));
                    }
                    diagnostics.log("info", "queue_stale_check", format_args!("outcome=match"));
                    let request = queue_application_home_migration(&plan).map_err(|error| {
                        diagnostics.failure("queue_write", &error);
                        warp::reject::custom(super::ApiError::migration(&error))
                    })?;
                    diagnostics.log("info", "queue_complete", format_args!("outcome=queued"));
                    Ok(Box::new(warp::reply::json(&serde_json::json!({
                        "ok": true,
                        "restart_required": true,
                        "request": request,
                    }))))
                }
            })
            .boxed()
    };

    let rollback_preview = {
        let config = ctx.config.clone();
        warp::path!("api" / "app-home-migration" / "rollback" / "preview")
            .and(warp::post())
            .and(warp::header::optional::<String>("authorization"))
            .and_then(move |authorization: Option<String>| {
                let config = config.clone();
                async move {
                    if !check_api_token(&authorization, &config) {
                        return Ok::<ApiReply, warp::Rejection>(Box::new(unauthorized_api_token()));
                    }
                    let roots = migration_roots(&config).map_err(|error| {
                        warp::reject::custom(super::ApiError::new(
                            warp::http::StatusCode::BAD_REQUEST,
                            error,
                        ))
                    })?;
                    let inspection = inspect_application_roots(&roots).map_err(|error| {
                        warp::reject::custom(super::ApiError::migration(&error))
                    })?;
                    let plan = plan_application_home_rollback(
                        &inspection.canonical_root,
                        &inspection.legacy_root,
                    )
                    .map_err(|error| warp::reject::custom(super::ApiError::migration(&error)))?;
                    Ok(Box::new(warp::reply::json(&serde_json::json!({
                        "ok": true,
                        "plan": plan,
                    }))))
                }
            })
            .boxed()
    };

    let rollback_queue = {
        let config = ctx.config.clone();
        warp::path!("api" / "app-home-migration" / "rollback" / "queue")
            .and(warp::post())
            .and(warp::header::optional::<String>("authorization"))
            .and(warp::body::json())
            .and_then(move |authorization: Option<String>, body: QueueRequest| {
                let config = config.clone();
                async move {
                    if !check_db_admin_token(&authorization, &config) {
                        return Ok::<ApiReply, warp::Rejection>(Box::new(
                            unauthorized_db_admin_token(),
                        ));
                    }
                    if body.confirmation != "ROLL BACK TO LLAMA MONITOR" {
                        return Ok(Box::new(warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({
                                "ok": false,
                                "error": "exact rollback confirmation is required"
                            })),
                            warp::http::StatusCode::BAD_REQUEST,
                        )));
                    }
                    let roots = migration_roots(&config).map_err(|error| {
                        warp::reject::custom(super::ApiError::new(
                            warp::http::StatusCode::BAD_REQUEST,
                            error,
                        ))
                    })?;
                    let inspection = inspect_application_roots(&roots).map_err(|error| {
                        warp::reject::custom(super::ApiError::migration(&error))
                    })?;
                    let plan = plan_application_home_rollback(
                        &inspection.canonical_root,
                        &inspection.legacy_root,
                    )
                    .map_err(|error| warp::reject::custom(super::ApiError::migration(&error)))?;
                    let request = queue_application_home_rollback(&plan).map_err(|error| {
                        warp::reject::custom(super::ApiError::migration(&error))
                    })?;
                    Ok(Box::new(warp::reply::json(&serde_json::json!({
                        "ok": true,
                        "restart_required": true,
                        "request": request,
                    }))))
                }
            })
            .boxed()
    };

    let cleanup = {
        let config = ctx.config.clone();
        warp::path!("api" / "app-home-migration" / "cleanup")
            .and(warp::post())
            .and(warp::header::optional::<String>("authorization"))
            .and(warp::body::json())
            .and_then(move |authorization: Option<String>, body: QueueRequest| {
                let config = config.clone();
                async move {
                    if !check_db_admin_token(&authorization, &config) {
                        return Ok::<ApiReply, warp::Rejection>(Box::new(
                            unauthorized_db_admin_token(),
                        ));
                    }
                    if body.confirmation != "DELETE LEGACY ROOT AFTER VERIFIED MIGRATION" {
                        return Ok(Box::new(warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({
                                "ok": false,
                                "error": "exact cleanup confirmation is required"
                            })),
                            warp::http::StatusCode::BAD_REQUEST,
                        )));
                    }
                    let roots = migration_roots(&config).map_err(|error| {
                        warp::reject::custom(super::ApiError::new(
                            warp::http::StatusCode::BAD_REQUEST,
                            error,
                        ))
                    })?;
                    let inspection = inspect_application_roots(&roots).map_err(|error| {
                        warp::reject::custom(super::ApiError::migration(&error))
                    })?;
                    let plan = plan_application_home_cleanup(
                        &inspection.canonical_root,
                        &inspection.legacy_root,
                    )
                    .map_err(|error| warp::reject::custom(super::ApiError::migration(&error)))?;
                    let request = queue_application_home_cleanup(&plan).map_err(|error| {
                        warp::reject::custom(super::ApiError::migration(&error))
                    })?;
                    Ok(Box::new(warp::reply::json(&serde_json::json!({
                        "ok": true,
                        "restart_required": true,
                        "request": request,
                    }))))
                }
            })
            .boxed()
    };

    status
        .or(preview)
        .unify()
        .or(queue)
        .unify()
        .or(rollback_preview)
        .unify()
        .or(rollback_queue)
        .unify()
        .or(cleanup)
        .unify()
        .boxed()
}
