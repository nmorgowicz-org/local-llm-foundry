use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock};

use warp::Filter;

use crate::config::AppConfig;
use crate::state::AppState;

use super::common::{
    ApiCtx, ApiRoute, check_api_token, check_db_admin_token, unauthorized_api_token,
    unauthorized_db_admin_token,
};
use crate::llama::vram_estimator::gguf_arch_to_heuristic_name;

static MODEL_LIBRARY_MIGRATION_RUNNING: AtomicBool = AtomicBool::new(false);
static MODEL_ROOT_RELOCATION_RUNNING: AtomicBool = AtomicBool::new(false);
static MODEL_INVENTORY_SCAN_RUNNING: AtomicBool = AtomicBool::new(false);
static IMPORT_RESOURCE_ESTIMATE_GATE: LazyLock<Arc<tokio::sync::Semaphore>> =
    LazyLock::new(|| Arc::new(tokio::sync::Semaphore::new(2)));

struct MigrationExecutionGuard;

struct InventoryScanGuard;

impl MigrationExecutionGuard {
    fn acquire() -> Option<Self> {
        MODEL_LIBRARY_MIGRATION_RUNNING
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .ok()
            .map(|_| Self)
    }
}

impl Drop for MigrationExecutionGuard {
    fn drop(&mut self) {
        MODEL_LIBRARY_MIGRATION_RUNNING.store(false, Ordering::Release);
    }
}

struct ModelRootRelocationGuard;

impl ModelRootRelocationGuard {
    fn acquire() -> Option<Self> {
        MODEL_ROOT_RELOCATION_RUNNING
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .ok()
            .map(|_| Self)
    }
}

impl Drop for ModelRootRelocationGuard {
    fn drop(&mut self) {
        MODEL_ROOT_RELOCATION_RUNNING.store(false, Ordering::Release);
    }
}

impl InventoryScanGuard {
    fn acquire() -> Option<Self> {
        MODEL_INVENTORY_SCAN_RUNNING
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .ok()
            .map(|_| Self)
    }
}

impl Drop for InventoryScanGuard {
    fn drop(&mut self) {
        MODEL_INVENTORY_SCAN_RUNNING.store(false, Ordering::Release);
    }
}

fn error_reply(
    status: warp::http::StatusCode,
    error: impl ToString,
) -> Box<dyn warp::reply::Reply> {
    Box::new(warp::reply::with_status(
        warp::reply::json(&serde_json::json!({ "ok": false, "error": error.to_string() })),
        status,
    ))
}

fn migration_persistence_files(state: &AppState) -> Vec<PathBuf> {
    vec![
        state.presets_path.clone(),
        state.sessions_path.clone(),
        state.model_tags_path.clone(),
        state.ui_settings_path.clone(),
    ]
}

fn migration_import_roots(state: &AppState, models_dir: &std::path::Path) -> Vec<PathBuf> {
    state
        .model_tags_path
        .parent()
        .filter(|config_root| *config_root != models_dir && config_root.is_dir())
        .map(|path| vec![path.to_path_buf()])
        .unwrap_or_default()
}

fn model_root_relocation_paths(
    state: &AppState,
    config: &AppConfig,
) -> anyhow::Result<(PathBuf, PathBuf)> {
    let canonical = crate::paths::AppPaths::canonical_default_root().join("models");
    let legacy = crate::paths::AppPaths::legacy_default_root().join("models");
    let configured =
        get_effective_models_dir(state).unwrap_or_else(|| config.default_models_dir.clone());
    let source = if configured == canonical
        && crate::models::root_relocation::load_selection(&canonical)?.is_none()
        && legacy.is_dir()
    {
        legacy.clone()
    } else {
        configured
    };
    if source == canonical {
        anyhow::bail!("the model root is already the Foundry root");
    }
    if source != legacy {
        anyhow::bail!("custom or external model roots must be managed separately");
    }
    Ok((source, canonical))
}

fn api_model_root_relocation_status(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "models" / "root-relocation" / "status")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let state = state.clone();
            let config = app_config.clone();
            async move {
                if !check_api_token(&auth, &config) {
                    return Ok(unauthorized_api_token());
                }
                let canonical = crate::paths::AppPaths::canonical_default_root().join("models");
                let legacy = crate::paths::AppPaths::legacy_default_root().join("models");
                let selected = match crate::models::root_relocation::load_selection(&canonical) {
                    Ok(selection) => selection,
                    Err(error) => {
                        return Ok(error_reply(
                            warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                            error,
                        ));
                    }
                };
                let configured = get_effective_models_dir(&state)
                    .unwrap_or_else(|| config.default_models_dir.clone());
                let source = if configured == canonical && selected.is_none() && legacy.is_dir() {
                    legacy.clone()
                } else {
                    configured
                };
                Ok::<_, warp::Rejection>(Box::new(warp::reply::json(&serde_json::json!({
                    "source": source,
                    "destination": canonical,
                    "legacy_root": legacy,
                    "source_exists": source.is_dir(),
                    "selection": selected,
                    "custom_root": source != legacy && source != canonical,
                    "relocation_required": source == legacy && source.is_dir(),
                }))) as Box<dyn warp::reply::Reply>)
            }
        })
}

#[derive(serde::Deserialize)]
struct ModelRootRelocationPreviewRequest {
    choice: crate::models::root_relocation::ModelRootChoice,
}

#[derive(serde::Deserialize)]
struct ModelRootRelocationExecuteRequest {
    #[serde(default)]
    plan_id: String,
    #[serde(default)]
    choice: Option<crate::models::root_relocation::ModelRootChoice>,
    #[serde(default)]
    confirmation: String,
}

fn api_model_root_relocation_preview(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "models" / "root-relocation" / "preview")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<
            ModelRootRelocationPreviewRequest,
        >())
        .and_then(
            move |auth: Option<String>, request: ModelRootRelocationPreviewRequest| {
                let state = state.clone();
                let config = app_config.clone();
                async move {
                    if !check_api_token(&auth, &config) {
                        return Ok(unauthorized_api_token());
                    }
                    let (source, destination) = match model_root_relocation_paths(&state, &config) {
                        Ok(paths) => paths,
                        Err(error) => {
                            return Ok(error_reply(warp::http::StatusCode::BAD_REQUEST, error));
                        }
                    };
                    let persistence = migration_persistence_files(&state);
                    let result = tokio::time::timeout(
                    std::time::Duration::from_secs(30),
                    tokio::task::spawn_blocking(move || {
                        crate::models::root_relocation::plan_model_root_relocation_with_persistence(
                            &source,
                            &destination,
                            request.choice,
                            &persistence,
                        )
                    }),
                )
                .await;
                    match result {
                        Ok(Ok(Ok(plan))) => Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                            &plan,
                        ))
                            as Box<dyn warp::reply::Reply>),
                        Ok(Ok(Err(error))) => {
                            Ok(error_reply(warp::http::StatusCode::BAD_REQUEST, error))
                        }
                        Ok(Err(error)) => Ok(error_reply(
                            warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                            error,
                        )),
                        Err(_) => Ok(error_reply(
                            warp::http::StatusCode::REQUEST_TIMEOUT,
                            "Model-root relocation preview timed out",
                        )),
                    }
                }
            },
        )
}

fn api_model_root_relocation_execute(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "models" / "root-relocation" / "execute")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<ModelRootRelocationExecuteRequest>())
        .and_then(move |auth: Option<String>, request: ModelRootRelocationExecuteRequest| {
            let state = state.clone();
            let config = app_config.clone();
            async move {
                if !check_db_admin_token(&auth, &config) {
                    return Ok(unauthorized_db_admin_token());
                }
                let Some(choice) = request.choice else {
                    return Ok(error_reply(warp::http::StatusCode::BAD_REQUEST, "model-root choice is required"));
                };
                let confirmation = match choice {
                    crate::models::root_relocation::ModelRootChoice::KeepLegacy => "KEEP_LEGACY_MODEL_ROOT",
                    crate::models::root_relocation::ModelRootChoice::CopyIntoFoundry => "MOVE_MODELS_INTO_FOUNDRY",
                };
                if request.confirmation != confirmation || request.plan_id.len() != 64 {
                    return Ok(error_reply(warp::http::StatusCode::BAD_REQUEST, "model-root relocation requires the preview plan_id and exact confirmation"));
                }
                let Some(_guard) = ModelRootRelocationGuard::acquire() else {
                    return Ok(error_reply(warp::http::StatusCode::CONFLICT, "A model-root relocation is already running"));
                };
                let (source, destination) = match model_root_relocation_paths(&state, &config) {
                    Ok(paths) => paths,
                    Err(error) => return Ok(error_reply(warp::http::StatusCode::BAD_REQUEST, error)),
                };
                let persistence = migration_persistence_files(&state);
                let expected_plan_id = request.plan_id;
                let result = tokio::time::timeout(
                    std::time::Duration::from_secs(600),
                    tokio::task::spawn_blocking(move || {
                        let plan = crate::models::root_relocation::plan_model_root_relocation_with_persistence(
                            &source,
                            &destination,
                            choice,
                            &persistence,
                        )?;
                        if plan.plan_id != expected_plan_id {
                            anyhow::bail!("model-root relocation preview is stale; refresh and try again");
                        }
                        crate::models::root_relocation::execute_model_root_relocation(&plan)
                    }),
                )
                .await;
                match result {
                    Ok(Ok(Ok(receipt))) => {
                        if receipt.destination.is_dir()
                            && let Ok(discovered) = crate::models::scan_gguf_library(&receipt.destination)
                        {
                            *state.discovered_models.lock().unwrap() = discovered;
                        }
                        Ok::<_, warp::Rejection>(Box::new(warp::reply::json(&serde_json::json!({
                            "ok": true,
                            "receipt": receipt,
                            "restart_required": true,
                        }))) as Box<dyn warp::reply::Reply>)
                    }
                    Ok(Ok(Err(error))) => Ok(error_reply(warp::http::StatusCode::BAD_REQUEST, error)),
                    Ok(Err(error)) => Ok(error_reply(warp::http::StatusCode::INTERNAL_SERVER_ERROR, error)),
                    Err(_) => Ok(error_reply(warp::http::StatusCode::REQUEST_TIMEOUT, "Model-root relocation timed out")),
                }
            }
        })
}

/// Where the user-wide Hugging Face cache lives, when it exists at all.
///
/// One definition, in the module that audits it. Migration and the audit tab must agree
/// on which directory they mean, or the tab would report a cache the importer cannot read.
fn shared_hf_hub() -> Option<PathBuf> {
    crate::models::external_cache::shared_hub()
}

fn api_model_inventory(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "models" / "inventory")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let state = state.clone();
            let config = app_config.clone();
            async move {
                if !check_api_token(&auth, &config) {
                    return Ok(unauthorized_api_token());
                }
                let models_dir = get_effective_models_dir(&state)
                    .unwrap_or_else(|| config.default_models_dir.clone());
                let Some(scan_guard) = InventoryScanGuard::acquire() else {
                    return Ok(error_reply(
                        warp::http::StatusCode::TOO_MANY_REQUESTS,
                        "A model inventory scan is already running",
                    ));
                };
                let result = tokio::time::timeout(
                    std::time::Duration::from_secs(10),
                    tokio::task::spawn_blocking(move || {
                        let _scan_guard = scan_guard;
                        crate::models::library::inventory(&models_dir)
                    }),
                )
                .await;
                match result {
                    Ok(Ok(Ok(inventory))) => Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                        &inventory,
                    ))
                        as Box<dyn warp::reply::Reply>),
                    Ok(Ok(Err(error))) => {
                        Ok(error_reply(warp::http::StatusCode::BAD_REQUEST, error))
                    }
                    Ok(Err(error)) => Ok(error_reply(
                        warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                        error,
                    )),
                    Err(_) => Ok(error_reply(
                        warp::http::StatusCode::REQUEST_TIMEOUT,
                        "Model inventory timed out",
                    )),
                }
            }
        })
}

/// Read-only audit of the Hugging Face cache the app does not manage.
///
/// Separate from `api_model_inventory` because it answers the opposite question: the
/// inventory reports what the library can serve, this reports what is taking up disk
/// *outside* it. A machine that has ever run `hf download` by hand has both, and until
/// now nothing in the app could see the second one.
///
/// Shares [`InventoryScanGuard`] with the inventory scan. Both walk large model trees on
/// the same disk, and running them at once buys nothing but contention.
fn api_external_model_cache(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "models" / "external-cache")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let state = state.clone();
            let config = app_config.clone();
            async move {
                if !check_api_token(&auth, &config) {
                    return Ok(unauthorized_api_token());
                }
                let models_dir = get_effective_models_dir(&state)
                    .unwrap_or_else(|| config.default_models_dir.clone());
                let library_hub = models_dir.join("cache/huggingface/hub");
                let Some(root) = shared_hf_hub() else {
                    // No user-wide cache at all, which is the healthy state. Reported as a
                    // fact rather than an error so the UI can say so instead of showing a
                    // failure for a machine that has nothing to clean up.
                    return Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                        &serde_json::json!({
                            "present": false,
                            "library_hub": library_hub,
                        }),
                    ))
                        as Box<dyn warp::reply::Reply>);
                };
                let Some(scan_guard) = InventoryScanGuard::acquire() else {
                    return Ok(error_reply(
                        warp::http::StatusCode::TOO_MANY_REQUESTS,
                        "A model inventory scan is already running",
                    ));
                };
                // Generous next to the inventory's 10 s: this walks every file of a cache
                // that can hold hundreds of gigabytes, and a cold one has no page cache
                // behind it. Timing out here would leave the tab permanently blank on the
                // machines that need it most.
                let result = tokio::time::timeout(
                    std::time::Duration::from_secs(120),
                    tokio::task::spawn_blocking(move || {
                        let _scan_guard = scan_guard;
                        crate::models::external_cache::audit(&root, Some(&library_hub))
                            .map(|audit| (audit, library_hub))
                    }),
                )
                .await;
                match result {
                    Ok(Ok(Ok((audit, library_hub)))) => {
                        Ok(Box::new(warp::reply::json(&serde_json::json!({
                            "present": true,
                            "library_hub": library_hub,
                            "audit": audit,
                        }))) as Box<dyn warp::reply::Reply>)
                    }
                    Ok(Ok(Err(error))) => {
                        Ok(error_reply(warp::http::StatusCode::BAD_REQUEST, error))
                    }
                    Ok(Err(error)) => Ok(error_reply(
                        warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                        error,
                    )),
                    Err(_) => Ok(error_reply(
                        warp::http::StatusCode::REQUEST_TIMEOUT,
                        "External cache audit timed out",
                    )),
                }
            }
        })
}

/// Repos the caller wants removed from the external cache, named the way the audit named
/// them. Ids, not paths, so the request cannot describe a directory the audit never offered.
#[derive(Debug, serde::Deserialize)]
struct ExternalCacheDeleteRequest {
    repo_ids: Vec<String>,
}

/// Cap per request. The UI confirms a selection before sending, and a thousand-item body is
/// a bug or an abuse rather than a user decision.
const MAX_EXTERNAL_CACHE_DELETIONS: usize = 64;

/// Delete selected repos from the external cache.
///
/// Reports every id separately and keeps going after a failure, because a partial success
/// is the normal outcome — a repo may have been removed by hand between audit and confirm,
/// and that should not abort the rest of a selection the user already approved.
fn api_external_model_cache_delete(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "models" / "external-cache" / "delete")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<ExternalCacheDeleteRequest>())
        .and_then(
            move |auth: Option<String>, request: ExternalCacheDeleteRequest| {
                let state = state.clone();
                let config = app_config.clone();
                async move {
                    if !check_api_token(&auth, &config) {
                        return Ok(unauthorized_api_token());
                    }
                    if request.repo_ids.is_empty() {
                        return Ok(error_reply(
                            warp::http::StatusCode::BAD_REQUEST,
                            "No repos selected for deletion",
                        ));
                    }
                    if request.repo_ids.len() > MAX_EXTERNAL_CACHE_DELETIONS {
                        return Ok(error_reply(
                            warp::http::StatusCode::BAD_REQUEST,
                            format!(
                                "Too many repos in one request ({} > {MAX_EXTERNAL_CACHE_DELETIONS})",
                                request.repo_ids.len()
                            ),
                        ));
                    }
                    let models_dir = get_effective_models_dir(&state)
                        .unwrap_or_else(|| config.default_models_dir.clone());
                    let library_hub = models_dir.join("cache/huggingface/hub");
                    let Some(root) = shared_hf_hub() else {
                        return Ok(error_reply(
                            warp::http::StatusCode::NOT_FOUND,
                            "There is no external Hugging Face cache on this machine",
                        ));
                    };
                    let removed = tokio::task::spawn_blocking(move || {
                        request
                            .repo_ids
                            .into_iter()
                            .map(|repo_id| {
                                match crate::models::external_cache::remove_repo(
                                    &root,
                                    &repo_id,
                                    Some(&library_hub),
                                ) {
                                    Ok(removed) => serde_json::json!({
                                        "repo_id": removed.repo_id,
                                        "ok": true,
                                        "bytes": removed.bytes,
                                        "file_count": removed.file_count,
                                    }),
                                    Err(error) => serde_json::json!({
                                        "repo_id": repo_id,
                                        "ok": false,
                                        "error": error.to_string(),
                                    }),
                                }
                            })
                            .collect::<Vec<_>>()
                    })
                    .await;
                    match removed {
                        Ok(results) => {
                            let freed: u64 = results
                                .iter()
                                .filter_map(|row| row.get("bytes").and_then(|b| b.as_u64()))
                                .sum();
                            let failed = results
                                .iter()
                                .filter(|row| row["ok"] != serde_json::Value::Bool(true))
                                .count();
                            Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                                &serde_json::json!({
                                    "ok": failed == 0,
                                    "freed_bytes": freed,
                                    "failed": failed,
                                    "results": results,
                                }),
                            )) as Box<dyn warp::reply::Reply>)
                        }
                        Err(error) => Ok(error_reply(
                            warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                            error,
                        )),
                    }
                }
            },
        )
}

#[derive(Debug, serde::Deserialize)]
struct AdoptDirectoryRequest {
    /// The directory to bring into the library.
    path: String,
    /// Optional library name. Slugged server-side; a caller cannot choose the destination
    /// path, only influence its last component.
    #[serde(default)]
    name: Option<String>,
}

/// Describe what adopting a local model directory would cost, without writing anything.
///
/// Separate from the execute call because the honest answer is sometimes "copy 27 GB", and
/// a user is entitled to see that, plus whether the files will be linked or duplicated,
/// before agreeing to it.
fn api_adopt_directory_preview(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "models" / "library" / "adopt" / "preview")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<AdoptDirectoryRequest>())
        .and_then(
            move |auth: Option<String>, request: AdoptDirectoryRequest| {
                let state = state.clone();
                let config = app_config.clone();
                async move {
                    if !check_api_token(&auth, &config) {
                        return Ok(unauthorized_api_token());
                    }
                    let models_dir = get_effective_models_dir(&state)
                        .unwrap_or_else(|| config.default_models_dir.clone());
                    let source = expand_user_path(&request.path);
                    let name = request.name.clone();
                    let planned = tokio::task::spawn_blocking(move || {
                        crate::models::local_adopt::plan_adoption(
                            &models_dir,
                            &source,
                            name.as_deref(),
                        )
                    })
                    .await;
                    match planned {
                        Ok(Ok(plan)) => Ok::<_, warp::Rejection>(
                            Box::new(warp::reply::json(&plan)) as Box<dyn warp::reply::Reply>,
                        ),
                        // A directory that is not a model, or a name already taken, is the
                        // user's answer to give — not a server fault.
                        Ok(Err(error)) => Ok(error_reply(
                            warp::http::StatusCode::BAD_REQUEST,
                            format!("{error:#}"),
                        )),
                        Err(error) => Ok(error_reply(
                            warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                            error,
                        )),
                    }
                }
            },
        )
}

/// Adopt a local model directory into `mlx/native`.
///
/// Re-plans server-side rather than accepting a plan from the client. The preview exists to
/// inform a decision, not to authorize one, and the directory may have changed since.
fn api_adopt_directory_execute(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "models" / "library" / "adopt")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<AdoptDirectoryRequest>())
        .and_then(
            move |auth: Option<String>, request: AdoptDirectoryRequest| {
                let state = state.clone();
                let config = app_config.clone();
                async move {
                    if !check_api_token(&auth, &config) {
                        return Ok(unauthorized_api_token());
                    }
                    let models_dir = get_effective_models_dir(&state)
                        .unwrap_or_else(|| config.default_models_dir.clone());
                    let source = expand_user_path(&request.path);
                    let name = request.name.clone();
                    // No timeout. Copying a 27 GB model across filesystems legitimately takes
                    // minutes, and a timeout here would abandon a half-written staging
                    // directory rather than let the cleanup path run.
                    let adopted = tokio::task::spawn_blocking(move || {
                        crate::models::local_adopt::adopt_directory(
                            &models_dir,
                            &source,
                            name.as_deref(),
                        )
                    })
                    .await;
                    match adopted {
                        Ok(Ok(model)) => Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                            &serde_json::json!({"ok": true, "model": model}),
                        ))
                            as Box<dyn warp::reply::Reply>),
                        Ok(Err(error)) => Ok(error_reply(
                            warp::http::StatusCode::BAD_REQUEST,
                            format!("{error:#}"),
                        )),
                        Err(error) => Ok(error_reply(
                            warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                            error,
                        )),
                    }
                }
            },
        )
}

#[derive(Debug, serde::Deserialize)]
struct DeleteDirectoryRequest {
    path: String,
}

/// Delete a directory-shaped model the library manages.
///
/// The pre-existing `DELETE /api/models/file` validates a `.gguf` suffix and removes a
/// single file, which left MLX and Transformers models undeletable from the UI entirely.
/// This is the directory counterpart rather than a relaxation of that suffix check: the two
/// have genuinely different safety arguments, and widening the file endpoint to accept
/// directories would have inherited its much broader containment rule, which allows
/// anything under `$HOME`.
fn api_delete_model_directory(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "models" / "library" / "directory")
        .and(warp::delete())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<DeleteDirectoryRequest>())
        .and_then(
            move |auth: Option<String>, request: DeleteDirectoryRequest| {
                let state = state.clone();
                let config = app_config.clone();
                async move {
                    if !check_api_token(&auth, &config) {
                        return Ok(unauthorized_api_token());
                    }
                    let models_dir = get_effective_models_dir(&state)
                        .unwrap_or_else(|| config.default_models_dir.clone());
                    let target = expand_user_path(&request.path);
                    let removed = tokio::task::spawn_blocking(move || {
                        crate::models::local_adopt::remove_managed_directory(&models_dir, &target)
                    })
                    .await;
                    match removed {
                        Ok(Ok(removed)) => {
                            // Drop the entry from the running discovery list so the UI does not
                            // keep offering a model that is gone.
                            if let Ok(mut models) = state.discovered_models.lock() {
                                models.retain(|m| !m.path.starts_with(&removed.path));
                            }
                            Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                                &serde_json::json!({"ok": true, "removed": removed}),
                            ))
                                as Box<dyn warp::reply::Reply>)
                        }
                        Ok(Err(error)) => Ok(error_reply(
                            warp::http::StatusCode::BAD_REQUEST,
                            format!("{error:#}"),
                        )),
                        Err(error) => Ok(error_reply(
                            warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                            error,
                        )),
                    }
                }
            },
        )
}

#[derive(Debug, serde::Deserialize)]
struct DeleteManagedCacheRequest {
    repo_id: String,
}

fn api_delete_managed_hf_repo(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "models" / "library" / "cache")
        .and(warp::delete())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<DeleteManagedCacheRequest>())
        .and_then(
            move |auth: Option<String>, request: DeleteManagedCacheRequest| {
                let state = state.clone();
                let config = app_config.clone();
                async move {
                    if !check_api_token(&auth, &config) {
                        return Ok(unauthorized_api_token());
                    }
                    let models_dir = get_effective_models_dir(&state)
                        .unwrap_or_else(|| config.default_models_dir.clone());
                    let removed = tokio::task::spawn_blocking(move || {
                        crate::models::library::delete_managed_hf_repo(
                            &models_dir,
                            &request.repo_id,
                        )
                    })
                    .await;
                    match removed {
                        Ok(Ok(bytes)) => Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                            &serde_json::json!({"ok": true, "freed_bytes": bytes}),
                        ))
                            as Box<dyn warp::reply::Reply>),
                        Ok(Err(error)) => Ok(error_reply(
                            warp::http::StatusCode::BAD_REQUEST,
                            format!("{error:#}"),
                        )),
                        Err(error) => Ok(error_reply(
                            warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                            error,
                        )),
                    }
                }
            },
        )
}

/// Expand a leading `~` so a user can type the path they see in a shell.
///
/// Only the leading component, and only `~` alone: `~other` is left untouched rather than
/// guessed at, because resolving another user's home is not something a model-library
/// import should be doing.
fn expand_user_path(raw: &str) -> PathBuf {
    let trimmed = raw.trim();
    if trimmed == "~" {
        return dirs::home_dir().unwrap_or_else(|| PathBuf::from(trimmed));
    }
    if let Some(rest) = trimmed.strip_prefix("~/")
        && let Some(home) = dirs::home_dir()
    {
        return home.join(rest);
    }
    PathBuf::from(trimmed)
}

fn api_rapid_model_resolver_preview(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "models" / "rapid-mlx" / "resolve" / "preview")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<
            crate::inference::rapid_mlx::model_resolver::RapidMlxModelSource,
        >())
        .and_then(move |auth: Option<String>, source| {
            let state = state.clone();
            let config = app_config.clone();
            async move {
                if !check_api_token(&auth, &config) {
                    return Ok(unauthorized_api_token());
                }
                let models_dir = get_effective_models_dir(&state)
                    .unwrap_or_else(|| config.default_models_dir.clone());
                let context = crate::inference::rapid_mlx::model_resolver::RapidMlxResolveContext {
                    models_dir,
                    python_executable: PathBuf::from(if cfg!(windows) {
                        "python.exe"
                    } else {
                        "python3"
                    }),
                    runtime_version:
                        crate::inference::rapid_mlx::compatibility::LATEST_QUALIFIED_VERSION_TEXT
                            .into(),
                    hf_token: None,
                    verified_aliases: Vec::new(),
                    execute_conversion: false,
                };
                let preview =
                    crate::inference::rapid_mlx::model_resolver::preview_async(source, context)
                        .await
                        .map_err(|error| {
                            warp::reject::custom(super::ApiError::internal(error.to_string()))
                        })?;
                Ok::<_, warp::Rejection>(
                    Box::new(warp::reply::json(&preview)) as Box<dyn warp::reply::Reply>
                )
            }
        })
}

fn api_gguf_import_compatibility_preview(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "models" / "gguf" / "import" / "compatibility" / "preview")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<
            crate::models::gguf_import::GgufImportPreviewRequest,
        >())
        .and_then(
            move |auth: Option<String>,
                  request: crate::models::gguf_import::GgufImportPreviewRequest| {
                let state = state.clone();
                let config = app_config.clone();
                async move {
                    if !check_api_token(&auth, &config) {
                        return Ok(unauthorized_api_token());
                    }
                    let models_dir = get_effective_models_dir(&state)
                        .unwrap_or_else(|| config.default_models_dir.clone());
                    match crate::models::gguf_import::inspect_async(request.path, models_dir).await
                    {
                        Ok(report) => Ok::<_, warp::Rejection>(
                            Box::new(warp::reply::json(&report)) as Box<dyn warp::reply::Reply>,
                        ),
                        Err(error) => Ok(error_reply(warp::http::StatusCode::BAD_REQUEST, error)),
                    }
                }
            },
        )
}

fn import_lab_context(
    state: &AppState,
    config: &AppConfig,
) -> crate::models::gguf_recovery::RecoveryContext {
    let models_dir =
        get_effective_models_dir(state).unwrap_or_else(|| config.default_models_dir.clone());
    let config_dir = state
        .model_tags_path
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(|| {
            models_dir
                .parent()
                .map(std::path::Path::to_path_buf)
                .unwrap_or_else(|| models_dir.clone())
        });
    crate::models::gguf_recovery::RecoveryContext {
        models_dir,
        config_dir,
    }
}

fn api_import_lab_availability(
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "models" / "import-lab" / "availability")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let config = app_config.clone();
            async move {
                if !check_api_token(&auth, &config) {
                    return Ok(unauthorized_api_token());
                }
                Ok::<_, warp::Rejection>(Box::new(warp::reply::json(&serde_json::json!({
                    "local_execution_available": crate::models::import_lab::local_execution_available(),
                    "platform_requirement": "Apple Silicon macOS",
                    "supported_profile": "smollm2-135m-instruct-llama-v1",
                    "compatibility": "experimental",
                    "launchable": false,
                    "fallback_engine": "llama.cpp"
                }))) as Box<dyn warp::reply::Reply>)
            }
        })
}

fn api_import_lab_resource_estimate(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "models" / "import-lab" / "resource-estimate")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<
            crate::models::gguf_import::GgufImportPreviewRequest,
        >())
        .and_then(
            move |auth: Option<String>,
                  request: crate::models::gguf_import::GgufImportPreviewRequest| {
                let state = state.clone();
                let config = app_config.clone();
                async move {
                    if !check_api_token(&auth, &config) {
                        return Ok(unauthorized_api_token());
                    }
                    let context = import_lab_context(&state, &config);
                    let models_dir = context.models_dir.clone();
                    let path = request.path;
                    if let Err(error) =
                        crate::models::import_lab::validate_library_relative_path(&path)
                    {
                        return Ok(error_reply(warp::http::StatusCode::BAD_REQUEST, error));
                    }
                    let permit = match IMPORT_RESOURCE_ESTIMATE_GATE.clone().try_acquire_owned() {
                        Ok(permit) => permit,
                        Err(_) => {
                            return Ok(error_reply(
                                warp::http::StatusCode::TOO_MANY_REQUESTS,
                                "Resource estimation is busy; retry shortly",
                            ));
                        }
                    };
                    let result = tokio::time::timeout(
                        std::time::Duration::from_secs(15),
                        tokio::task::spawn_blocking(move || {
                            // A timed-out blocking task cannot be forcibly cancelled. Keep
                            // its permit until the task really exits so request bursts can
                            // never exceed this bounded disk/metadata worker pool.
                            let _permit = permit;
                            crate::models::import_lab::resource_estimate(&models_dir, &path)
                        }),
                    )
                    .await;
                    match result {
                        Ok(Ok(Ok(estimate))) => {
                            Ok::<_, warp::Rejection>(Box::new(warp::reply::json(&estimate))
                                as Box<dyn warp::reply::Reply>)
                        }
                        Ok(Ok(Err(error))) => {
                            Ok(error_reply(warp::http::StatusCode::BAD_REQUEST, error))
                        }
                        Ok(Err(error)) => Ok(error_reply(
                            warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                            error,
                        )),
                        Err(_) => Ok(error_reply(
                            warp::http::StatusCode::REQUEST_TIMEOUT,
                            "Resource estimate timed out",
                        )),
                    }
                }
            },
        )
}

fn api_import_lab_jobs(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    let list_config = app_config.clone();
    let list = warp::path!("api" / "models" / "import-lab" / "jobs")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let config = list_config.clone();
            async move {
                if !check_api_token(&auth, &config) {
                    return Ok(unauthorized_api_token());
                }
                Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                    &crate::models::import_lab::jobs(),
                )) as Box<dyn warp::reply::Reply>)
            }
        });

    let start = warp::path!("api" / "models" / "import-lab" / "jobs")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<
            crate::models::import_lab::ImportJobStartRequest,
        >())
        .and_then(
            move |auth: Option<String>,
                  request: crate::models::import_lab::ImportJobStartRequest| {
                let state = state.clone();
                let config = app_config.clone();
                async move {
                    if !check_api_token(&auth, &config) {
                        return Ok(unauthorized_api_token());
                    }
                    match crate::models::import_lab::start_job(
                        import_lab_context(&state, &config),
                        request,
                    ) {
                        Ok(job) => Ok::<_, warp::Rejection>(Box::new(warp::reply::with_status(
                            warp::reply::json(&job),
                            warp::http::StatusCode::ACCEPTED,
                        ))
                            as Box<dyn warp::reply::Reply>),
                        Err(error) => Ok(error_reply(warp::http::StatusCode::BAD_REQUEST, error)),
                    }
                }
            },
        );
    list.or(start).unify()
}

fn api_import_lab_job_actions(
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    let status_config = app_config.clone();
    let status = warp::path!("api" / "models" / "import-lab" / "jobs" / String)
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |id: String, auth: Option<String>| {
            let config = status_config.clone();
            async move {
                if !check_api_token(&auth, &config) {
                    return Ok(unauthorized_api_token());
                }
                match crate::models::import_lab::job(&id) {
                    Some(job) => Ok::<_, warp::Rejection>(
                        Box::new(warp::reply::json(&job)) as Box<dyn warp::reply::Reply>
                    ),
                    None => Ok(error_reply(
                        warp::http::StatusCode::NOT_FOUND,
                        "Import job was not found",
                    )),
                }
            }
        });
    let cancel_config = app_config.clone();
    let cancel = warp::path!("api" / "models" / "import-lab" / "jobs" / String / "cancel")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |id: String, auth: Option<String>| {
            let config = cancel_config.clone();
            async move {
                if !check_api_token(&auth, &config) {
                    return Ok(unauthorized_api_token());
                }
                match crate::models::import_lab::cancel_job(&id) {
                    Ok(job) => Ok::<_, warp::Rejection>(
                        Box::new(warp::reply::json(&job)) as Box<dyn warp::reply::Reply>
                    ),
                    Err(error) => Ok(error_reply(warp::http::StatusCode::BAD_REQUEST, error)),
                }
            }
        });
    let forget = warp::path!("api" / "models" / "import-lab" / "jobs" / String)
        .and(warp::delete())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |id: String, auth: Option<String>| {
            let config = app_config.clone();
            async move {
                if !check_api_token(&auth, &config) {
                    return Ok(unauthorized_api_token());
                }
                match crate::models::import_lab::forget_job(&id) {
                    Ok(()) => Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                        &serde_json::json!({"ok": true}),
                    ))
                        as Box<dyn warp::reply::Reply>),
                    Err(error) => Ok(error_reply(warp::http::StatusCode::BAD_REQUEST, error)),
                }
            }
        });
    status.or(cancel).unify().or(forget).unify()
}

fn api_library_migration_preview(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "models" / "library" / "migration" / "preview")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<MigrationPreviewRequest>())
        .and_then(
            move |auth: Option<String>, request: MigrationPreviewRequest| {
                let state = state.clone();
                let config = app_config.clone();
                async move {
                    if !check_api_token(&auth, &config) {
                        return Ok(unauthorized_api_token());
                    }
                    let models_dir = get_effective_models_dir(&state)
                        .unwrap_or_else(|| config.default_models_dir.clone());
                    let persistence = migration_persistence_files(&state);
                    let imports = migration_import_roots(&state, &models_dir);
                    let shared_hf = shared_hf_hub();
                    let result = tokio::time::timeout(
                        std::time::Duration::from_secs(10),
                        tokio::task::spawn_blocking(move || {
                            crate::models::library::plan_migration_selected_hf(
                                &models_dir,
                                &persistence,
                                &imports,
                                &request.hf_repos,
                                shared_hf.as_deref(),
                            )
                        }),
                    )
                    .await;
                    match result {
                        Ok(Ok(Ok(plan))) => Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                            &plan,
                        ))
                            as Box<dyn warp::reply::Reply>),
                        Ok(Ok(Err(error))) => {
                            Ok(error_reply(warp::http::StatusCode::BAD_REQUEST, error))
                        }
                        Ok(Err(error)) => Ok(error_reply(
                            warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                            error,
                        )),
                        Err(_) => Ok(error_reply(
                            warp::http::StatusCode::REQUEST_TIMEOUT,
                            "Migration preview timed out",
                        )),
                    }
                }
            },
        )
}

#[derive(serde::Deserialize)]
struct MigrationPreviewRequest {
    #[serde(default)]
    hf_repos: Vec<String>,
}

#[derive(serde::Deserialize)]
struct ExecuteMigrationRequest {
    #[serde(default)]
    plan_id: String,
    #[serde(default)]
    confirmation: String,
    #[serde(default)]
    hf_repos: Vec<String>,
}

fn api_library_migration_execute(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "models" / "library" / "migration" / "execute")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<ExecuteMigrationRequest>())
        .and_then(move |auth: Option<String>, request: ExecuteMigrationRequest| {
            let state = state.clone();
            let config = app_config.clone();
            async move {
                if !check_db_admin_token(&auth, &config) { return Ok(unauthorized_db_admin_token()); }
                if request.confirmation != "MIGRATE_MODEL_LIBRARY" || request.plan_id.len() != 64 {
                    return Ok(error_reply(warp::http::StatusCode::BAD_REQUEST, "Migration requires the preview plan_id and confirmation MIGRATE_MODEL_LIBRARY"));
                }
                let Some(_execution_guard) = MigrationExecutionGuard::acquire() else {
                    return Ok(error_reply(
                        warp::http::StatusCode::CONFLICT,
                        "A model-library migration is already running",
                    ));
                };
                let models_dir = get_effective_models_dir(&state).unwrap_or_else(|| config.default_models_dir.clone());
                let persistence = migration_persistence_files(&state);
                let imports = migration_import_roots(&state, &models_dir);
                let shared_hf = shared_hf_hub();
                let plan_id = request.plan_id;
                let hf_repos = request.hf_repos;
                let migration_state = state.clone();
                let result = tokio::task::spawn_blocking(move || {
                    // Hold every persistence lock across disk rewrite and memory refresh so
                    // ordinary saves cannot race the journaled migration.
                    let mut presets_guard = migration_state.presets.lock().unwrap();
                    let mut sessions_guard = migration_state.sessions.lock().unwrap();
                    let mut tags_guard = migration_state.model_tags.lock().unwrap();
                    let mut settings_guard = migration_state.ui_settings.lock().unwrap();
                    let plan = crate::models::library::execute_migration_selected_hf(
                        &models_dir, &persistence, &imports, &hf_repos, shared_hf.as_deref(), &plan_id,
                    )?;
                    crate::models::library::rewrite_in_memory_paths(&mut *presets_guard, &plan)?;
                    crate::models::library::rewrite_in_memory_paths(&mut *sessions_guard, &plan)?;
                    crate::models::library::rewrite_in_memory_paths(&mut *tags_guard, &plan)?;
                    crate::models::library::rewrite_in_memory_paths(&mut *settings_guard, &plan)?;
                    Ok::<_, anyhow::Error>(plan)
                }).await;
                match result {
                    Ok(Ok(plan)) => {
                        if let Ok(discovered) =
                            crate::models::scan_gguf_library(&plan.models_dir)
                        {
                            *state.discovered_models.lock().unwrap() = discovered;
                        }
                        Ok::<_, warp::Rejection>(Box::new(warp::reply::json(&serde_json::json!({ "ok": true, "plan": plan }))) as Box<dyn warp::reply::Reply>)
                    }
                    Ok(Err(error)) => Ok(error_reply(warp::http::StatusCode::BAD_REQUEST, error)),
                    Err(error) => Ok(error_reply(warp::http::StatusCode::INTERNAL_SERVER_ERROR, error)),
                }
            }
        })
}

/// Returns the user-configured models directory, or None if not set.
pub(crate) fn get_effective_models_dir(state: &AppState) -> Option<PathBuf> {
    if let Some(ref d) = state.models_dir {
        return Some(d.clone());
    }
    let s = state.ui_settings.lock().unwrap();
    if !s.models_dir.is_empty() {
        return Some(PathBuf::from(&s.models_dir));
    }
    None
}

// ── POST /api/models/download/start ──────────────────────────────────────────

fn api_models_download_start(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "models" / "download" / "start")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<serde_json::Value>())
        .and_then(move |auth: Option<String>, body: serde_json::Value| {
            let cfg = app_config.clone();
            let state = state.clone();
            async move {
                if !check_db_admin_token(&auth, &cfg) && !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }

                let model = body["model"].as_str().unwrap_or("").to_string();
                let source = body["source"].as_str().unwrap_or("").to_string();

                if model.is_empty() {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": "Missing 'model' field"
                        })),
                    ));
                }

                if source != "hf" {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": format!("Unsupported source '{}'; only 'hf' is currently supported", source)
                        })),
                    ));
                }

                let (repo_id, file_path) = if model.contains('/') {
                    let parts: Vec<&str> = model.splitn(2, '/').collect();
                    (parts[0].to_string(), parts.get(1).unwrap_or(&"").to_string())
                } else {
                    (model.clone(), "model.gguf".to_string())
                };

                let target_dir = get_effective_models_dir(&state)
                    .unwrap_or_else(|| cfg.default_models_dir.clone())
                    .join("gguf");

                let hf_token = crate::hf::hf_load_token();

                match crate::model_download::start_download(
                    &repo_id,
                    &file_path,
                    None,
                    &target_dir,
                    hf_token,
                ) {
                    Ok(download_id) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(
                        Box::new(warp::reply::json(&serde_json::json!({
                            "ok": true,
                            "download_id": download_id
                        }))),
                    ),
                    Err(e) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": format!("Failed to start download: {}", e)
                        })),
                    )),
                }
            }
        })
}

fn api_models_download_resume(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "models" / "download" / "resume")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<serde_json::Value>())
        .and_then(move |auth: Option<String>, body: serde_json::Value| {
            let state = state.clone();
            let config = app_config.clone();
            async move {
                if !check_api_token(&auth, &config) {
                    return Ok(unauthorized_api_token());
                }
                let raw_path = body["path"].as_str().unwrap_or("");
                let path = expand_user_path(raw_path);
                let models_dir = get_effective_models_dir(&state)
                    .unwrap_or_else(|| config.default_models_dir.clone());
                let root = match models_dir.canonicalize() {
                    Ok(root) => root,
                    Err(error) => {
                        return Ok(error_reply(warp::http::StatusCode::BAD_REQUEST, error));
                    }
                };
                let canonical = match path.canonicalize() {
                    Ok(path) if path.starts_with(&root) => path,
                    _ => {
                        return Ok(error_reply(
                            warp::http::StatusCode::BAD_REQUEST,
                            "Resume path is outside the model library",
                        ));
                    }
                };
                if !canonical.to_string_lossy().ends_with(".part") {
                    return Ok(error_reply(
                        warp::http::StatusCode::BAD_REQUEST,
                        "Resume path must be a .part file",
                    ));
                }
                let metadata = crate::model_download::load_resume_metadata(&canonical);
                let (repo_id, file_path, filename) = if let Some(metadata) = metadata {
                    let filename = metadata
                        .save_as
                        .clone()
                        .or_else(|| metadata.file_path.rsplit('/').next().map(str::to_owned));
                    let Some(filename) = filename else {
                        return Ok(error_reply(
                            warp::http::StatusCode::BAD_REQUEST,
                            "Resume metadata has no destination filename",
                        ));
                    };
                    (metadata.repo_id, metadata.file_path, filename)
                } else {
                    let repo_id = body["repo_id"].as_str().unwrap_or("").trim().to_string();
                    let file_path = body["file_path"].as_str().unwrap_or("").trim().to_string();
                    let filename = body["save_as"]
                        .as_str()
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(str::to_owned)
                        .or_else(|| file_path.rsplit('/').next().map(str::to_owned));
                    let valid_repo = repo_id.split_once('/').is_some_and(|(owner, name)| {
                        !owner.is_empty()
                            && !name.is_empty()
                            && owner.bytes().all(|byte| {
                                byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-'
                            })
                            && name.bytes().all(|byte| {
                                byte.is_ascii_alphanumeric()
                                    || byte == b'_'
                                    || byte == b'-'
                                    || byte == b'.'
                            })
                    });
                    let valid_file_path = !file_path.is_empty()
                        && !file_path.starts_with('/')
                        && !file_path.contains('\\')
                        && file_path.split('/').all(|component| {
                            !component.is_empty() && component != "." && component != ".."
                        });
                    let Some(filename) = filename else {
                        return Ok(error_reply(
                            warp::http::StatusCode::BAD_REQUEST,
                            "Resume source has no destination filename",
                        ));
                    };
                    let valid_filename = filename.to_ascii_lowercase().ends_with(".gguf")
                        && !filename.contains('/')
                        && !filename.contains('\\')
                        && filename != "."
                        && filename != "..";
                    if !valid_repo || !valid_file_path || !valid_filename {
                        return Ok(error_reply(
                            warp::http::StatusCode::BAD_REQUEST,
                            "Resume source must be a valid Hugging Face GGUF file",
                        ));
                    }
                    (repo_id, file_path, filename)
                };
                let target_dir = canonical.parent().unwrap_or(&root).to_path_buf();
                if target_dir.join(&filename).with_extension("part") != canonical {
                    return Ok(error_reply(
                        warp::http::StatusCode::BAD_REQUEST,
                        "Resume source filename does not match the partial file",
                    ));
                }
                match crate::model_download::start_download(
                    &repo_id,
                    &file_path,
                    Some(&filename),
                    &target_dir,
                    crate::hf::hf_load_token(),
                ) {
                    Ok(download_id) => Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                        &serde_json::json!({"ok": true, "download_id": download_id}),
                    ))
                        as Box<dyn warp::reply::Reply>),
                    Err(error) => Ok(error_reply(warp::http::StatusCode::BAD_REQUEST, error)),
                }
            }
        })
}

fn api_models_download_status(
    _state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "models" / "download" / String / "status")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |id: String, auth: Option<String>| {
            let cfg = app_config.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }

                match crate::model_download::get_download_status(&id) {
                    Some(status) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": true,
                            "status": status
                        })),
                    )),
                    None => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({
                                "ok": false,
                                "error": "Download not found"
                            })),
                            warp::http::StatusCode::NOT_FOUND,
                        ),
                    )),
                }
            }
        })
}

fn api_models_download_cancel(
    _state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "models" / "download" / String / "cancel")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |id: String, auth: Option<String>| {
            let cfg = app_config.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }

                let ok = crate::model_download::cancel_download(&id);
                if ok {
                    Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                        &serde_json::json!({ "ok": true }),
                    )))
                } else {
                    Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                        &serde_json::json!({
                            "ok": false,
                            "error": "Download not found or already finished"
                        }),
                    )))
                }
            }
        })
}

// ── P3.2: Third-Party Models ──────────────────────────────────────────────────

fn api_third_party_models(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "third-party-models")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<serde_json::Value>())
        .and_then(move |auth: Option<String>, body: serde_json::Value| {
            let cfg = app_config.clone();
            let state = state.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }

                let _ = body["include_subdirs"].as_bool().unwrap_or(true);

                let extra_dirs = state
                    .ui_settings
                    .lock()
                    .map(|s| s.extra_models_dirs.clone())
                    .unwrap_or_default();
                let models = crate::llama::spawn_wizard::scan_third_party_models(&extra_dirs);
                let models: Vec<serde_json::Value> = models
                    .into_iter()
                    .map(|m| {
                        serde_json::json!({
                            "path": m.path,
                            "name": m.name,
                            "source_tool": m.source_tool,
                            "size": m.size,
                        })
                    })
                    .collect();

                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                    &serde_json::json!({
                        "ok": true,
                        "models": models
                    }),
                )))
            }
        })
}

// ── P3.3: Model Introspection ──────────────────────────────────────────────

fn api_model_introspect(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "model" / "introspect")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<serde_json::Value>())
        .and_then(move |auth: Option<String>, body: serde_json::Value| {
            let cfg = app_config.clone();
            let state = state.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }

                let model_path = body["model_path"].as_str().unwrap_or("").trim().to_string();
                if model_path.is_empty() {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": "Missing 'model_path' field"
                        })),
                    ));
                }

                // Security: only allow .gguf files or Ollama content-addressed blobs
                let is_gguf_ext = model_path.to_ascii_lowercase().ends_with(".gguf");
                let is_ollama_blob = model_path.contains("/blobs/sha256-")
                    || model_path.contains("\\blobs\\sha256-");
                if !is_gguf_ext && !is_ollama_blob {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": "model_path must point to a .gguf file"
                        })),
                    ));
                }

                let canon = match std::path::Path::new(&model_path).canonicalize() {
                    Ok(p) => p,
                    Err(_) => {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({
                                "ok": false,
                                "error": "Model file not found"
                            })),
                        ));
                    }
                };
                let models_dir = get_effective_models_dir(&state)
                    .unwrap_or_else(|| cfg.default_models_dir.clone());
                let in_models_dir = models_dir
                    .canonicalize()
                    .map(|d| canon.starts_with(&d))
                    .unwrap_or(false);
                let in_home = dirs::home_dir()
                    .and_then(|h| h.canonicalize().ok())
                    .map(|h| canon.starts_with(&h))
                    .unwrap_or(false);
                let in_extra = state
                    .ui_settings
                    .lock()
                    .map(|s| {
                        s.extra_models_dirs.iter().any(|d| {
                            std::path::Path::new(d)
                                .canonicalize()
                                .map(|cd| canon.starts_with(&cd))
                                .unwrap_or(false)
                        })
                    })
                    .unwrap_or(false);
                if !in_models_dir && !in_home && !in_extra {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": "model_path is outside allowed directories"
                        })),
                    ));
                }

                let llama_server_path = cfg.llama_server_path.clone();
                let result = tokio::time::timeout(
                    std::time::Duration::from_secs(30),
                    crate::llama::spawn_wizard::introspect_model(
                        &model_path,
                        llama_server_path.to_string_lossy().as_ref(),
                    ),
                )
                .await;

                let metadata = match result {
                    Ok(Ok(meta)) => meta,
                    Ok(Err(e)) => {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({
                                "ok": false,
                                "error": e
                            })),
                        ));
                    }
                    Err(_) => {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({
                                "ok": false,
                                "error": "Introspection timed out (30s)"
                            })),
                        ));
                    }
                };

                let file_size_bytes = std::fs::metadata(&model_path).map(|m| m.len()).unwrap_or(0);

                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                    &serde_json::json!({
                        "ok": true,
                        "metadata": metadata,
                        "cached": metadata.cached,
                        "file_size_bytes": file_size_bytes
                    }),
                )))
            }
        })
}

// ── POST /api/models/gguf-meta ────────────────────────────────────────────────

fn api_models_gguf_meta(
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "models" / "gguf-meta")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<serde_json::Value>())
        .and_then(move |auth: Option<String>, body: serde_json::Value| {
            let cfg = app_config.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }

                let model_path = body["model_path"].as_str().unwrap_or("").trim().to_string();
                if model_path.is_empty() {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": "Missing 'model_path' field"
                        })),
                    ));
                }

                let meta = match crate::llama::gguf_meta::read_gguf_metadata(std::path::Path::new(
                    &model_path,
                )) {
                    Ok(m) => m,
                    Err(e) => {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({
                                "ok": false,
                                "error": format!("Failed to read GGUF metadata: {}", e)
                            })),
                        ));
                    }
                };

                // Number of full-attention (KV-bearing) layers. Prefer the real GGUF
                // value (block_count / full_attention_interval); fall back to the name
                // heuristic only for older GGUFs that don't record the interval.
                let n_attn_layers = meta.n_attn_layers().or_else(|| {
                    let arch_str = meta.architecture.as_ref()?;
                    let heuristic_name = gguf_arch_to_heuristic_name(arch_str);
                    let resolved = if arch_str.eq_ignore_ascii_case("qwen35") {
                        match meta.block_count {
                            Some(bc) if bc >= 75 => "qwen3_5".to_string(),
                            _ => heuristic_name,
                        }
                    } else {
                        heuristic_name
                    };
                    let param_b = meta.param_count.map(|p| p as f64 / 1e9).unwrap_or(0.0);
                    let arch = crate::llama::vram_estimator::ModelArch::from_name_and_params(
                        &resolved, param_b,
                    );
                    (arch.n_attn_layers < arch.n_layers).then_some(arch.n_attn_layers)
                });

                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                    &serde_json::json!({
                        "ok": true,
                        "architecture": meta.architecture,
                        "param_count": meta.param_count,
                        "block_count": meta.block_count,
                        "head_count": meta.head_count,
                        "head_count_kv": meta.head_count_kv,
                        "key_length": meta.key_length,
                        "key_length_swa": meta.key_length_swa,
                        "context_length": meta.context_length,
                        "embedding_length": meta.embedding_length,
                        "feed_forward_length": meta.feed_forward_length,
                        "expert_count": meta.expert_count,
                        "expert_used_count": meta.expert_used_count,
                        "mtp_depth": meta.mtp_depth,
                        "n_attn_layers": n_attn_layers,
                        "full_attention_interval": meta.full_attention_interval,
                        "linear_attn_state_bytes": meta.linear_attn_state_bytes(),
                        "sliding_window": meta.sliding_window,
                        "n_global_attn_layers": meta.n_global_attn_layers,
                        "global_kv_heads": meta.global_kv_heads,
                        "local_kv_heads": meta.local_kv_heads,
                    }),
                )))
            }
        })
}

// ── POST /api/models/mlx-introspect (Phase 8A3) ──────────────────────────────────────────────

fn canonical_model_path_within_roots(
    model_path: &std::path::Path,
    allowed_roots: &[PathBuf],
) -> Result<PathBuf, String> {
    let canonical_model_path = model_path
        .canonicalize()
        .map_err(|_| "Model path not found or invalid".to_string())?;

    let allowed = allowed_roots.iter().any(|root| {
        root.canonicalize()
            .map(|canonical_root| canonical_model_path.starts_with(canonical_root))
            .unwrap_or(false)
    });
    if !allowed {
        return Err("model_path is outside configured model directories".to_string());
    }

    Ok(canonical_model_path)
}

/// Builds the `config` object of the mlx-introspect response envelope from a parsed
/// `MlxConfig` plus a vision verdict, shared by the local-path and remote (HF repo_id)
/// introspection branches so the two cannot silently diverge in shape (§2.3b).
fn mlx_config_to_json_obj(
    config: crate::inference::rapid_mlx::mlx_meta::MlxConfig,
    vision_evidence: Option<crate::model_vision::VisionEvidence>,
) -> serde_json::Map<String, serde_json::Value> {
    let mut config_obj = serde_json::Map::new();
    if let Some(model_type) = config.model_type {
        config_obj.insert("model_type".into(), serde_json::json!(model_type));
    }
    if let Some(hidden_size) = config.hidden_size {
        config_obj.insert("hidden_size".into(), serde_json::json!(hidden_size));
    }
    if let Some(num_layers) = config.num_layers {
        config_obj.insert("num_layers".into(), serde_json::json!(num_layers));
    }
    if let Some(num_attention_heads) = config.num_attention_heads {
        config_obj.insert(
            "num_attention_heads".into(),
            serde_json::json!(num_attention_heads),
        );
    }
    if let Some(num_key_value_heads) = config.num_key_value_heads {
        config_obj.insert(
            "num_key_value_heads".into(),
            serde_json::json!(num_key_value_heads),
        );
    }
    if let Some(head_dim) = config.head_dim {
        config_obj.insert("head_dim".into(), serde_json::json!(head_dim));
    }
    if let Some(n_ff) = config.n_ff {
        config_obj.insert("n_ff".into(), serde_json::json!(n_ff));
    }
    if let Some(num_experts) = config.num_experts {
        config_obj.insert("num_experts".into(), serde_json::json!(num_experts));
    }
    if let Some(num_experts_per_tok) = config.num_experts_per_tok {
        config_obj.insert(
            "num_experts_per_tok".into(),
            serde_json::json!(num_experts_per_tok),
        );
    }
    if let Some(sliding_window) = config.sliding_window {
        config_obj.insert("sliding_window".into(), serde_json::json!(sliding_window));
    }
    if let Some(max_position_embeddings) = config.max_position_embeddings {
        config_obj.insert(
            "max_position_embeddings".into(),
            serde_json::json!(max_position_embeddings),
        );
    }
    if config.vision_config.is_some() {
        config_obj.insert("has_vision_config".into(), serde_json::json!(true));
    }
    match vision_evidence {
        Some(evidence) => {
            config_obj.insert("vision".into(), serde_json::json!(evidence.vision));
            config_obj.insert("vision_source".into(), serde_json::json!(evidence.source));
            config_obj.insert("vision_confidence".into(), serde_json::json!("confirmed"));
        }
        None => {
            config_obj.insert(
                "vision_source".into(),
                serde_json::json!("no readable config.json or safetensors index"),
            );
            config_obj.insert(
                "vision_confidence".into(),
                serde_json::json!("undetermined"),
            );
        }
    }
    if let Some(quant) = config.quantization {
        let mut qobj = serde_json::Map::new();
        if let Some(bits) = quant.bits {
            qobj.insert("bits".into(), serde_json::json!(bits));
        }
        if let Some(group_size) = quant.group_size {
            qobj.insert("group_size".into(), serde_json::json!(group_size));
        }
        if !qobj.is_empty() {
            config_obj.insert("quantization".into(), serde_json::Value::Object(qobj));
        }
    }
    config_obj
}

fn api_models_mlx_introspect(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "models" / "mlx-introspect")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<serde_json::Value>())
        .and_then(move |auth: Option<String>, body: serde_json::Value| {
            let cfg = app_config.clone();
            let state = state.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }

                let model_path = body["model_path"].as_str().unwrap_or("").trim().to_string();
                let repo_id = body["repo_id"].as_str().unwrap_or("").trim().to_string();

                // Remote mode (§2.3b): same response envelope, sourced from HF instead of disk.
                // Kept as a separate branch rather than unifying the fetch, since the local path
                // is a cheap blocking filesystem walk and the remote path is bounded, semaphore-
                // gated HTTPS — different concurrency shapes that should not share one code path.
                if !repo_id.is_empty() {
                    if !crate::hf::validate_hf_repo_id(&repo_id) {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({
                                "ok": false,
                                "error": "Invalid 'repo_id' — expected 'owner/name'"
                            })),
                        ));
                    }
                    let revision = {
                        let r = body["revision"].as_str().unwrap_or("main").trim().to_string();
                        if r.is_empty() { "main".to_string() } else { r }
                    };

                    let _permit = match super::hf::HF_EVIDENCE_GATE.clone().try_acquire_owned() {
                        Ok(permit) => permit,
                        Err(_) => {
                            return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                warp::reply::with_status(
                                    warp::reply::json(&serde_json::json!({
                                        "ok": false,
                                        "error": "HF evidence resolution is busy; try again shortly"
                                    })),
                                    warp::http::StatusCode::TOO_MANY_REQUESTS,
                                ),
                            ));
                        }
                    };
                    let result = tokio::time::timeout(std::time::Duration::from_secs(90), async {
                        let mut errors = Vec::new();
                        let mut response = serde_json::Map::new();

                        match crate::hf::resolve_mlx_repo_size_bytes(&repo_id).await {
                            Ok(Some(size)) => {
                                response.insert("recursive_size_bytes".into(), serde_json::json!(size));
                            }
                            Ok(None) => {}
                            Err(e) => errors.push(format!("recursive_size: {e}")),
                        }

                        match crate::hf::fetch_mlx_config_revision_aware(&repo_id, &revision, "config.json").await {
                            Ok(config) => {
                                let config_value = serde_json::to_value(&config).ok();
                                let mut index_value = None;
                                match crate::hf::fetch_raw_bytes_at(
                                    &repo_id,
                                    &revision,
                                    "model.safetensors.index.json",
                                    2 * 1024 * 1024,
                                ).await {
                                    Ok(bytes) => {
                                        index_value = serde_json::from_slice::<serde_json::Value>(&bytes).ok();
                                    }
                                    Err(e) => errors.push(format!("model.safetensors.index.json: {e}")),
                                }
                                let vision_evidence = crate::model_vision::resolve_from_artifacts(
                                    config_value.as_ref(),
                                    index_value.as_ref(),
                                );
                                let has_mmproj = index_value
                                    .as_ref()
                                    .and_then(|v| serde_json::to_vec(v).ok())
                                    .and_then(|bytes| {
                                        crate::inference::rapid_mlx::info_query::has_mmproj_in_index_bytes(&bytes).ok()
                                    })
                                    .unwrap_or(false);
                                response.insert("has_vision_adapter_in_index".into(), serde_json::json!(has_mmproj));
                                let config_obj = mlx_config_to_json_obj(config, vision_evidence);
                                response.insert("config".into(), serde_json::Value::Object(config_obj));
                            }
                            Err(e) => errors.push(format!("config.json: {e}")),
                        }

                        (response, errors)
                    })
                    .await;

                    let (mut response, errors) = match result {
                        Ok((resp, errs)) => (resp, errs),
                        Err(_) => {
                            return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                warp::reply::json(&serde_json::json!({
                                    "ok": false,
                                    "error": "MLX introspection timed out (90s)"
                                })),
                            ));
                        }
                    };
                    if !errors.is_empty() {
                        response.insert("errors".into(), serde_json::json!(errors));
                    }
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": true,
                            "repo_id": repo_id,
                            "revision": revision,
                            "data": response
                        })),
                    ));
                }

                if model_path.is_empty() {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": "Missing 'model_path' or 'repo_id' field"
                        })),
                    ));
                }

                let mut allowed_roots = vec![get_effective_models_dir(&state)
                    .unwrap_or_else(|| cfg.default_models_dir.clone())];
                allowed_roots.extend(
                    state
                    .ui_settings
                    .lock()
                    .map(|s| s.extra_models_dirs.iter().map(PathBuf::from).collect::<Vec<_>>())
                    .unwrap_or_default(),
                );
                let canon = match canonical_model_path_within_roots(
                    std::path::Path::new(&model_path),
                    &allowed_roots,
                ) {
                    Ok(path) => path,
                    Err(error) => {
                        return Ok(error_reply(warp::http::StatusCode::BAD_REQUEST, error));
                    }
                };

                // Introspect using blocking task with timeout
                let result = tokio::time::timeout(
                    std::time::Duration::from_secs(10),
                    tokio::task::spawn_blocking(move || {
                        let mut errors = Vec::new();
                        let mut response = serde_json::Map::new();

                        // Recursive size
                        let recursive_size = match crate::inference::rapid_mlx::info_query::resolve_mlx_recursive_size(&canon) {
                            Ok(s) => s,
                            Err(e) => {
                                errors.push(format!("recursive_size: {e}"));
                                0
                            }
                        };
                        response.insert("recursive_size_bytes".into(), serde_json::json!(recursive_size));

                        // Local config
                        match crate::inference::rapid_mlx::info_query::read_mlx_local_config(&canon) {
                            Ok(Some(config)) => {
                                // `vision_config` is not the only way a checkpoint carries a
                                // tower: older wrappers name `mm_vision_tower`, and some ship
                                // the tensors without naming anything. Report the artifact
                                // verdict alongside, and say when the artifacts did not
                                // settle it rather than reporting a bare `false`.
                                let vision_evidence =
                                    crate::inference::rapid_mlx::mlx_meta::read_local_vision_evidence(&canon);
                                let config_obj = mlx_config_to_json_obj(config, vision_evidence);
                                response.insert("config".into(), serde_json::Value::Object(config_obj));
                            }
                            Ok(None) => {
                                errors.push("no config.json found".into());
                            }
                            Err(e) => {
                                errors.push(format!("read config: {e}"));
                            }
                        }

                        // mmproj in index (only real MLX-VLM components, per builder item 13)
                        let has_mmproj = match crate::inference::rapid_mlx::info_query::has_mmproj_in_index(&canon) {
                            Ok(v) => v,
                            Err(e) => {
                                errors.push(format!("check mmproj: {e}"));
                                false
                            }
                        };
                        response.insert("has_vision_adapter_in_index".into(), serde_json::json!(has_mmproj));

                        (response, errors)
                    }),
                )
                .await;

                let (mut response, errors) = match result {
                    Ok(Ok((resp, errs))) => (resp, errs),
                    Ok(Err(e)) => {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({
                                "ok": false,
                                "error": format!("Introspection failed: {e}")
                            })),
                        ));
                    }
                    Err(_) => {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({
                                "ok": false,
                                "error": "MLX introspection timed out (10s)"
                            })),
                        ));
                    }
                };

                if !errors.is_empty() {
                    response.insert("errors".into(), serde_json::json!(errors));
                }

                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                    &serde_json::json!({
                        "ok": true,
                        "model_path": model_path,
                        "data": response
                    }),
                )))
            }
        })
}

// ── GET /api/models ───────────────────────────────────────────────────────────

fn api_get_models(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "models")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let state = state.clone();
            let cfg = app_config.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }
                let models_dir = get_effective_models_dir(&state)
                    .unwrap_or_else(|| cfg.default_models_dir.clone());
                let Some(scan_guard) = InventoryScanGuard::acquire() else {
                    return Ok(error_reply(
                        warp::http::StatusCode::TOO_MANY_REQUESTS,
                        "A model inventory scan is already running",
                    ));
                };
                let inventory = match tokio::time::timeout(
                    std::time::Duration::from_secs(10),
                    tokio::task::spawn_blocking(move || {
                        let _scan_guard = scan_guard;
                        crate::models::library::inventory(&models_dir)
                    }),
                )
                .await
                {
                    Ok(Ok(Ok(inventory))) => inventory,
                    Ok(Ok(Err(error))) => {
                        return Ok(error_reply(warp::http::StatusCode::BAD_REQUEST, error));
                    }
                    Ok(Err(error)) => {
                        return Ok(error_reply(
                            warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                            error,
                        ));
                    }
                    Err(_) => {
                        return Ok(error_reply(
                            warp::http::StatusCode::REQUEST_TIMEOUT,
                            "Model inventory timed out",
                        ));
                    }
                };
                let tags = state.model_tags.lock().unwrap().tags.clone();
                let legacy = state.discovered_models.lock().unwrap().clone();
                let models_with_tags: Vec<serde_json::Value> = inventory
                    .entries
                    .iter()
                    .map(|entry| {
                        let model_path = entry.path.to_string_lossy().to_string();
                        let mut obj = serde_json::to_value(entry).unwrap_or_default();
                        if let Some(model_obj) = obj.as_object_mut() {
                            model_obj.insert(
                                "tags".into(),
                                serde_json::json!(
                                    tags.get(&model_path).cloned().unwrap_or_default()
                                ),
                            );
                            if let Some(model) =
                                legacy.iter().find(|model| model.path == entry.path)
                            {
                                model_obj.insert(
                                    "classification".into(),
                                    serde_json::json!(crate::models::classify_model(model)),
                                );
                            } else {
                                model_obj.insert("classification".into(), serde_json::Value::Null);
                            }
                        }
                        obj
                    })
                    .collect();
                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                    &models_with_tags,
                )))
            }
        })
}

fn api_refresh_models(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "models" / "refresh")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let cfg = app_config.clone();
            if !check_api_token(&auth, &cfg) {
                return futures_util::future::ready(Ok(unauthorized_api_token()));
            }
            if let Some(ref dir) = state.models_dir {
                match crate::models::scan_gguf_library(dir) {
                    Ok(discovered) => {
                        let count = discovered.len();
                        *state.discovered_models.lock().unwrap() = discovered;
                        futures_util::future::ready(Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(
                            Box::new(warp::reply::json(&serde_json::json!({"ok": true, "count": count}))),
                        ))
                    }
                    Err(e) => {
                        futures_util::future::ready(Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(
                            Box::new(warp::reply::json(&serde_json::json!({"ok": false, "error": e.to_string()}))),
                        ))
                    }
                }
            } else {
                futures_util::future::ready(Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(
                    Box::new(warp::reply::json(&serde_json::json!({"ok": false, "error": "no models directory configured (use --models-dir)"}))),
                ))
            }
        })
}

fn api_delete_model_file(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "models" / "file")
        .and(warp::delete())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<serde_json::Value>())
        .and_then(move |auth: Option<String>, body: serde_json::Value| {
            let cfg = app_config.clone();
            let st = state.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }

                let path_str = match body.get("path").and_then(|v| v.as_str()) {
                    Some(p) => p.to_string(),
                    None => {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(
                                &serde_json::json!({"ok": false, "error": "missing path"}),
                            ),
                        ));
                    }
                };

                let lower_path = path_str.to_ascii_lowercase();
                let is_gguf = lower_path.ends_with(".gguf");
                let is_partial = lower_path.ends_with(".part");
                if !is_gguf && !is_partial {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(
                            &serde_json::json!({"ok": false, "error": "only .gguf or .part files can be deleted"}),
                        ),
                    ));
                }

                let path = std::path::Path::new(&path_str);
                // Containment check: must be inside allowed model directories.
                let models_dir = get_effective_models_dir(&st)
                    .unwrap_or_else(|| cfg.default_models_dir.clone());
                let canon = match path.canonicalize() {
                    Ok(c) => c,
                    Err(e) => {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(
                                &serde_json::json!({"ok": false, "error": format!("invalid path: {e}")}),
                            ),
                        ));
                    }
                };
                let in_models_dir = models_dir
                    .canonicalize()
                    .map(|d| canon.starts_with(&d))
                    .unwrap_or(false);
                let in_home = dirs::home_dir()
                    .and_then(|h| h.canonicalize().ok())
                    .map(|h| canon.starts_with(&h))
                    .unwrap_or(false);
                let in_extra = st
                    .ui_settings
                    .lock()
                    .map(|s| {
                        s.extra_models_dirs.iter().any(|d| {
                            std::path::Path::new(d)
                                .canonicalize()
                                .map(|cd| canon.starts_with(&cd))
                                .unwrap_or(false)
                        })
                    })
                    .unwrap_or(false);

                if !in_models_dir && !in_home && !in_extra {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(
                            &serde_json::json!({"ok": false, "error": "file is outside allowed model directories"}),
                        ),
                    ));
                }

                match std::fs::remove_file(&canon) {
                    Ok(_) => {
                        if is_partial {
                            let sidecar = canon.with_extension("part.json");
                            if let Err(error) = std::fs::remove_file(&sidecar)
                                && error.kind() != std::io::ErrorKind::NotFound
                            {
                                eprintln!(
                                    "[warn] could not clear resume metadata for {}: {error}",
                                    sidecar.display()
                                );
                            }
                        }
                        // Drop the file's provenance with the file. Leaving it behind would
                        // let a later, unrelated download that happens to reuse the name
                        // inherit an origin it does not have.
                        if let (Some(dir), Some(name)) =
                            (canon.parent(), canon.file_name().and_then(|n| n.to_str()))
                            && let Err(e) = crate::models::provenance::forget(dir, name)
                        {
                            eprintln!("[warn] could not clear provenance for {name}: {e}");
                        }
                        let mut models = st.discovered_models.lock().unwrap();
                        models.retain(|m| m.path.to_str() != Some(&path_str));
                        Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({"ok": true})),
                        ))
                    }
                    Err(e) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(
                            &serde_json::json!({"ok": false, "error": format!("Failed to delete: {e}")}),
                        ),
                    )),
                }
            }
        })
}

fn api_get_model_tags(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "models" / "tags")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let cfg = app_config.clone();
            if !check_api_token(&auth, &cfg) {
                return futures_util::future::ready(Ok(unauthorized_api_token()));
            }
            let tags = state.model_tags.lock().unwrap().clone();
            futures_util::future::ready(Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(
                Box::new(warp::reply::json(&tags)),
            ))
        })
}

fn api_put_model_tags(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "models" / "tags")
        .and(warp::put())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<serde_json::Value>())
        .and_then(move |auth: Option<String>, body: serde_json::Value| {
            let st = state.clone();
            let cfg = app_config.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }

                let model_path = match body.get("model_path").and_then(|v| v.as_str()) {
                    Some(p) => p.to_string(),
                    None => {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(
                                &serde_json::json!({"ok": false, "error": "missing model_path"}),
                            ),
                        ));
                    }
                };

                let new_tags = match body.get("tags") {
                    Some(t) => match t.as_array() {
                        Some(arr) => arr
                            .iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect::<Vec<String>>(),
                        None => {
                            return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                warp::reply::json(&serde_json::json!({"ok": false, "error": "tags must be an array of strings"})),
                            ));
                        }
                    },
                    None => {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(
                                &serde_json::json!({"ok": false, "error": "missing tags"}),
                            ),
                        ));
                    }
                };

                let mut tags = st.model_tags.lock().unwrap();
                tags.tags.insert(model_path, new_tags);
                let tags_path = st.model_tags_path.clone();
                if let Err(e) = tags.save(&tags_path) {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": format!("Failed to save model tags: {e}")
                        })),
                    ));
                }

                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                    &serde_json::json!({"ok": true}),
                )))
            }
        })
}

fn api_get_collections(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "collections")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let cfg = app_config.clone();
            if !check_api_token(&auth, &cfg) {
                return futures_util::future::ready(Ok(unauthorized_api_token()));
            }
            let collections = state.preset_collections.lock().unwrap().clone();
            futures_util::future::ready(Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(
                Box::new(warp::reply::json(&collections)),
            ))
        })
}

fn api_create_collection(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "collections")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<serde_json::Value>())
        .and_then(move |auth: Option<String>, body: serde_json::Value| {
            let cfg = app_config.clone();
            let st = state.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }
                let name = match body.get("name").and_then(|v| v.as_str()) {
                    Some(n) if !n.is_empty() => n.to_string(),
                    _ => {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(
                                &serde_json::json!({"ok": false, "error": "missing or empty name"}),
                            ),
                        ));
                    }
                };
                let description = body
                    .get("description")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let preset_ids = body
                    .get("preset_ids")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();

                let config_dir = st
                    .model_tags_path
                    .parent()
                    .unwrap_or(&std::path::PathBuf::from("."))
                    .to_path_buf();
                let mut collections = st.preset_collections.lock().unwrap();
                let id = crate::collections::unique_id("coll", &name, &collections.collections);
                let new = crate::collections::PresetCollection {
                    id,
                    name,
                    description,
                    preset_ids,
                };
                collections.collections.push(new.clone());
                if let Err(e) = crate::collections::save_collections(&config_dir, &collections) {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": format!("Failed to save collections: {e}")
                        })),
                    ));
                }
                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                    &serde_json::json!({"ok": true, "collection": new}),
                )))
            }
        })
}

fn api_patch_collection(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "collections" / String)
        .and(warp::patch())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<serde_json::Value>())
        .and_then(
            move |id: String, auth: Option<String>, body: serde_json::Value| {
                let cfg = app_config.clone();
                let st = state.clone();
                async move {
                    if !check_api_token(&auth, &cfg) {
                        return Ok(unauthorized_api_token());
                    }
                    let config_dir = st
                        .model_tags_path
                        .parent()
                        .unwrap_or(&std::path::PathBuf::from("."))
                        .to_path_buf();
                    let mut collections = st.preset_collections.lock().unwrap();
                    let col = match collections.collections.iter_mut().find(|c| c.id == id) {
                        Some(c) => c,
                        None => {
                            return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                warp::reply::json(&serde_json::json!({
                                    "ok": false,
                                    "error": "collection not found"
                                })),
                            ));
                        }
                    };
                    if let Some(name) = body.get("name").and_then(|v| v.as_str()) {
                        col.name = name.to_string();
                    }
                    if let Some(desc) = body.get("description") {
                        col.description = if desc.is_null() {
                            None
                        } else {
                            desc.as_str().map(|s| s.to_string())
                        };
                    }
                    if let Some(ids) = body.get("preset_ids").and_then(|v| v.as_array()) {
                        col.preset_ids = ids
                            .iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect();
                    }
                    if let Err(e) = crate::collections::save_collections(&config_dir, &collections)
                    {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({
                                "ok": false,
                                "error": format!("Failed to save collections: {e}")
                            })),
                        ));
                    }
                    Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                        &serde_json::json!({"ok": true}),
                    )))
                }
            },
        )
}

fn api_delete_collection(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "collections" / String)
        .and(warp::delete())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |id: String, auth: Option<String>| {
            let cfg = app_config.clone();
            let st = state.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }
                let config_dir = st
                    .model_tags_path
                    .parent()
                    .unwrap_or(&std::path::PathBuf::from("."))
                    .to_path_buf();
                let mut collections = st.preset_collections.lock().unwrap();
                let before = collections.collections.len();
                collections.collections.retain(|c| c.id != id);
                if collections.collections.len() == before {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": "collection not found"
                        })),
                    ));
                }
                if let Err(e) = crate::collections::save_collections(&config_dir, &collections) {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": format!("Failed to save collections: {e}")
                        })),
                    ));
                }
                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                    &serde_json::json!({"ok": true}),
                )))
            }
        })
}

pub(crate) fn routes(ctx: ApiCtx) -> ApiRoute {
    let state = ctx.state.clone();
    let config = ctx.config.clone();

    let mut r = api_models_download_start(state.clone(), config.clone())
        .or(api_models_download_status(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r
        .or(api_model_root_relocation_status(
            state.clone(),
            config.clone(),
        ))
        .unify()
        .boxed();
    r = r
        .or(api_model_root_relocation_preview(
            state.clone(),
            config.clone(),
        ))
        .unify()
        .boxed();
    r = r
        .or(api_model_root_relocation_execute(
            state.clone(),
            config.clone(),
        ))
        .unify()
        .boxed();
    r = r
        .or(api_model_inventory(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r
        .or(api_external_model_cache(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r
        .or(api_external_model_cache_delete(
            state.clone(),
            config.clone(),
        ))
        .unify()
        .boxed();
    r = r
        .or(api_adopt_directory_preview(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r
        .or(api_adopt_directory_execute(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r
        .or(api_delete_model_directory(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r
        .or(api_rapid_model_resolver_preview(
            state.clone(),
            config.clone(),
        ))
        .unify()
        .boxed();
    r = r
        .or(api_gguf_import_compatibility_preview(
            state.clone(),
            config.clone(),
        ))
        .unify()
        .boxed();
    r = r
        .or(api_import_lab_availability(config.clone()))
        .unify()
        .boxed();
    r = r
        .or(api_import_lab_resource_estimate(
            state.clone(),
            config.clone(),
        ))
        .unify()
        .boxed();
    r = r
        .or(api_import_lab_jobs(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r
        .or(api_import_lab_job_actions(config.clone()))
        .unify()
        .boxed();
    r = r
        .or(api_library_migration_preview(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r
        .or(api_library_migration_execute(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r
        .or(api_models_download_resume(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r
        .or(api_models_download_cancel(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r
        .or(api_third_party_models(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r
        .or(api_model_introspect(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r.or(api_models_gguf_meta(config.clone())).unify().boxed();
    r = r
        .or(api_models_mlx_introspect(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r
        .or(api_get_models(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r
        .or(api_refresh_models(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r
        .or(api_delete_model_file(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r
        .or(api_delete_managed_hf_repo(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r
        .or(api_get_model_tags(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r
        .or(api_put_model_tags(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r
        .or(api_get_collections(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r
        .or(api_create_collection(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r
        .or(api_patch_collection(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r
        .or(api_delete_collection(state.clone(), config.clone()))
        .unify()
        .boxed();
    r
}

#[cfg(test)]
mod phase5_auth_tests {
    use super::*;
    use crate::web::auth::AuthManager;
    use warp::http::StatusCode;

    fn test_routes() -> ApiRoute {
        let config = Arc::new(AppConfig::for_test(
            Some("api-secret".to_string()),
            Some("admin-secret".to_string()),
        ));
        routes(ApiCtx {
            state: AppState::default(),
            auth: AuthManager::new(None, None, &crate::config::TLSConfig::default().mode),
            config,
        })
    }

    #[tokio::test]
    async fn phase5_read_routes_require_api_token() {
        for (method, path, body) in [
            ("GET", "/api/models/inventory", None),
            ("GET", "/api/models/import-lab/availability", None),
            ("GET", "/api/models/import-lab/jobs", None),
            ("GET", "/api/models/import-lab/jobs/missing", None),
            (
                "POST",
                "/api/models/import-lab/resource-estimate",
                Some(r#"{"path":"gguf/model.gguf"}"#),
            ),
            (
                "POST",
                "/api/models/import-lab/jobs",
                Some(r#"{"source_path":"gguf/model.gguf"}"#),
            ),
            ("POST", "/api/models/import-lab/jobs/missing/cancel", None),
            ("DELETE", "/api/models/import-lab/jobs/missing", None),
            (
                "POST",
                "/api/models/rapid-mlx/resolve/preview",
                Some(r#"{"kind":"alias","value":"model"}"#),
            ),
            (
                "POST",
                "/api/models/gguf/import/compatibility/preview",
                Some(r#"{"path":"gguf/model.gguf"}"#),
            ),
            (
                "POST",
                "/api/models/mlx-introspect",
                Some(r#"{"model_path":"/tmp/model"}"#),
            ),
            (
                "POST",
                "/api/models/library/migration/preview",
                Some(r#"{}"#),
            ),
            (
                "POST",
                "/api/models/download/resume",
                Some(r#"{"path":"gguf/model.part"}"#),
            ),
            (
                "DELETE",
                "/api/models/library/cache",
                Some(r#"{"repo_id":"org/model"}"#),
            ),
        ] {
            let mut request = warp::test::request().method(method).path(path);
            if let Some(body) = body {
                request = request
                    .header("content-type", "application/json")
                    .body(body);
            }
            let response = request.reply(&test_routes()).await;
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED, "{path}");
        }
    }

    #[tokio::test]
    async fn migration_execute_rejects_api_token_without_db_admin_token() {
        let response = warp::test::request()
            .method("POST")
            .path("/api/models/library/migration/execute")
            .header("authorization", "Bearer api-secret")
            .header("content-type", "application/json")
            .body(r#"{"plan_id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","confirmation":"MIGRATE_MODEL_LIBRARY"}"#)
            .reply(&test_routes())
            .await;
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn resolver_preview_returns_bad_request_for_malformed_json() {
        let routes = test_routes().recover(crate::web::handle_rejection);
        let response = warp::test::request()
            .method("POST")
            .path("/api/models/rapid-mlx/resolve/preview")
            .header("authorization", "Bearer api-secret")
            .header("content-type", "application/json")
            .body("{")
            .reply(&routes)
            .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn gguf_import_preview_returns_bad_request_for_malformed_json_and_traversal() {
        let routes = test_routes().recover(crate::web::handle_rejection);
        let malformed = warp::test::request()
            .method("POST")
            .path("/api/models/gguf/import/compatibility/preview")
            .header("authorization", "Bearer api-secret")
            .header("content-type", "application/json")
            .body("{")
            .reply(&routes)
            .await;
        assert_eq!(malformed.status(), StatusCode::BAD_REQUEST);

        let traversal = warp::test::request()
            .method("POST")
            .path("/api/models/gguf/import/compatibility/preview")
            .header("authorization", "Bearer api-secret")
            .header("content-type", "application/json")
            .body(r#"{"path":"../outside.gguf"}"#)
            .reply(&routes)
            .await;
        assert_eq!(traversal.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn import_lab_read_routes_accept_api_token_and_missing_job_is_not_found() {
        let routes = test_routes().recover(crate::web::handle_rejection);
        for path in [
            "/api/models/import-lab/availability",
            "/api/models/import-lab/jobs",
        ] {
            let response = warp::test::request()
                .method("GET")
                .path(path)
                .header("authorization", "Bearer api-secret")
                .reply(&routes)
                .await;
            assert_eq!(response.status(), StatusCode::OK, "{path}");
        }
        let missing = warp::test::request()
            .method("GET")
            .path("/api/models/import-lab/jobs/missing")
            .header("authorization", "Bearer api-secret")
            .reply(&routes)
            .await;
        assert_eq!(missing.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn import_lab_write_routes_return_bad_request_for_invalid_json_and_paths() {
        let routes = test_routes().recover(crate::web::handle_rejection);
        let malformed = warp::test::request()
            .method("POST")
            .path("/api/models/import-lab/jobs")
            .header("authorization", "Bearer api-secret")
            .header("content-type", "application/json")
            .body("{")
            .reply(&routes)
            .await;
        assert_eq!(malformed.status(), StatusCode::BAD_REQUEST);

        let unknown_field = warp::test::request()
            .method("POST")
            .path("/api/models/import-lab/jobs")
            .header("authorization", "Bearer api-secret")
            .header("content-type", "application/json")
            .body(r#"{"source_path":"gguf/model.gguf","unexpected":true}"#)
            .reply(&routes)
            .await;
        assert_eq!(unknown_field.status(), StatusCode::BAD_REQUEST);

        let traversal = warp::test::request()
            .method("POST")
            .path("/api/models/import-lab/resource-estimate")
            .header("authorization", "Bearer api-secret")
            .header("content-type", "application/json")
            .body(r#"{"path":"../outside.gguf"}"#)
            .reply(&routes)
            .await;
        assert_eq!(traversal.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn import_lab_resource_estimates_fail_fast_when_worker_pool_is_saturated() {
        let permits = IMPORT_RESOURCE_ESTIMATE_GATE
            .clone()
            .acquire_many_owned(2)
            .await
            .unwrap();
        let response = warp::test::request()
            .method("POST")
            .path("/api/models/import-lab/resource-estimate")
            .header("authorization", "Bearer api-secret")
            .header("content-type", "application/json")
            .body(r#"{"path":"gguf/model.gguf"}"#)
            .reply(&test_routes())
            .await;
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        drop(permits);
    }

    #[test]
    fn mlx_introspect_path_must_be_within_a_configured_model_root() {
        let models = tempfile::tempdir().expect("models root");
        let nested = models.path().join("publisher").join("model");
        std::fs::create_dir_all(&nested).expect("nested model");

        let outside = tempfile::tempdir().expect("outside root");
        assert!(canonical_model_path_within_roots(&nested, &[models.path().to_path_buf()]).is_ok());
        let error =
            canonical_model_path_within_roots(outside.path(), &[models.path().to_path_buf()])
                .expect_err("an arbitrary home or temporary path must not be accepted");
        assert!(error.contains("outside configured model directories"));
    }

    #[test]
    fn mlx_introspect_path_accepts_an_extra_model_root() {
        let primary = tempfile::tempdir().expect("primary root");
        let extra = tempfile::tempdir().expect("extra root");
        let model = extra.path().join("mlx-model");
        std::fs::create_dir(&model).expect("model directory");

        let canonical = canonical_model_path_within_roots(
            &model,
            &[primary.path().to_path_buf(), extra.path().to_path_buf()],
        )
        .expect("configured extra model root should be accepted");
        assert_eq!(canonical, model.canonicalize().unwrap());
    }

    #[tokio::test]
    async fn mlx_introspect_rejects_malformed_json_as_bad_request() {
        let routes = test_routes().recover(crate::web::handle_rejection);
        let response = warp::test::request()
            .method("POST")
            .path("/api/models/mlx-introspect")
            .header("authorization", "Bearer api-secret")
            .header("content-type", "application/json")
            .body("{")
            .reply(&routes)
            .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}
