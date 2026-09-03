use std::sync::Arc;

use warp::Filter;

use crate::config::AppConfig;
use crate::inference::launch::validate_preset_backend_config;
use crate::presets::{self, ModelPreset};
use crate::state::AppState;
use crate::web::safe_json_body;

use super::{
    ApiCtx, ApiRoute, box_reply, check_api_token, check_db_admin_token, unauthorized_api_token,
    unauthorized_db_admin_token, with_app_config,
};

#[derive(Debug, Default, serde::Deserialize)]
#[serde(default)]
struct DeletePresetRequest {
    expected_revision: Option<u64>,
    expected_catalog_etag: Option<String>,
    confirmation: Option<String>,
}

#[derive(Debug, Default, serde::Deserialize)]
#[serde(default)]
struct ResetPresetsRequest {
    expected_catalog_etag: Option<String>,
    confirmation: Option<String>,
}

pub(crate) fn routes(ctx: ApiCtx) -> ApiRoute {
    let state = ctx.state;
    let config = ctx.config;

    api_get_presets(state.clone(), config.clone())
        .map(box_reply)
        .or(api_get_preset(state.clone(), config.clone()).map(box_reply))
        .unify()
        .or(api_create_preset(state.clone(), config.clone()).map(box_reply))
        .unify()
        .or(api_update_preset(state.clone(), config.clone()).map(box_reply))
        .unify()
        .or(api_delete_preset(state.clone(), config.clone()).map(box_reply))
        .unify()
        .or(api_reset_presets(state, config).map(box_reply))
        .unify()
        .boxed()
}

fn preset_for_api(mut preset: ModelPreset) -> ModelPreset {
    preset.api_key_configured =
        preset.api_key_configured || preset.api_key.as_ref().is_some_and(|key| !key.is_empty());
    preset.api_key = None;
    preset.clear_api_key = false;
    // D5: compute model_source_view for Rapid-MLX presets (never persisted, API-only).
    if let Some(ref mut config) = preset.rapid_mlx
        && let Some(ref source) = config.model_source
    {
        config.model_source_view = Some(
            crate::inference::rapid_mlx::model_resolver::RapidMlxModelSourceView::from_source(
                source,
            ),
        );
    }
    preset
}

fn merge_preset_api_key(updated: &mut ModelPreset, existing_api_key: Option<String>) {
    if updated.clear_api_key {
        updated.api_key = None;
    } else if updated
        .api_key
        .as_deref()
        .map(str::is_empty)
        .unwrap_or(true)
    {
        updated.api_key = existing_api_key;
    }
    updated.api_key_configured = updated.api_key.as_ref().is_some_and(|key| !key.is_empty());
    updated.clear_api_key = false;
}

fn bundle_projection_conflict(preset: &ModelPreset) -> Option<String> {
    preset.bundle.as_ref()?;
    let mut projected = preset.clone();
    crate::presets::bundle::materialize_default_projection(&mut projected);
    let conflicts = [
        ("model_path", preset.model_path != projected.model_path),
        (
            "context_size",
            preset.context_size != projected.context_size,
        ),
        ("ctk", preset.ctk != projected.ctk),
        ("ctv", preset.ctv != projected.ctv),
        ("batch_size", preset.batch_size != projected.batch_size),
        ("ubatch_size", preset.ubatch_size != projected.ubatch_size),
        ("n_cpu_moe", preset.n_cpu_moe != projected.n_cpu_moe),
    ];
    conflicts
        .into_iter()
        .find_map(|(field, conflict)| conflict.then_some(field.to_string()))
}

fn api_get_presets(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "presets")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and(with_app_config(app_config))
        .and_then(move |auth: Option<String>, cfg: Arc<AppConfig>| {
            let presets: Vec<_> = state
                .presets
                .lock()
                .unwrap()
                .clone()
                .into_iter()
                .map(preset_for_api)
                .collect();
            if !check_api_token(&auth, &cfg) {
                return futures_util::future::ready(Ok(unauthorized_api_token()));
            }
            futures_util::future::ready(Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(
                Box::new(warp::reply::json(&presets)),
            ))
        })
}

fn api_get_preset(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "presets" / String)
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and(with_app_config(app_config))
        .and_then(
            move |id: String, auth: Option<String>, cfg: Arc<AppConfig>| {
                if !check_api_token(&auth, &cfg) {
                    return futures_util::future::ready(Ok(unauthorized_api_token()));
                }
                let preset = {
                    let presets = state.presets.lock().unwrap();
                    presets.iter().find(|p| p.id == id).cloned()
                };
                futures_util::future::ready(Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(
                    Box::new(warp::reply::json(&match preset {
                        Some(preset) => {
                            serde_json::json!({"ok": true, "preset": preset_for_api(preset)})
                        }
                        None => serde_json::json!({"ok": false, "error": "preset not found"}),
                    })),
                ))
            },
        )
}

fn api_create_preset(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "presets")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(safe_json_body::<ModelPreset>())
        .and_then(move |auth: Option<String>, mut preset: ModelPreset| {
            let cfg = app_config.clone();
            if !check_api_token(&auth, &cfg) {
                return futures_util::future::ready(Ok(unauthorized_api_token()));
            }
            presets::migrate_preset(&mut preset);
            if let Err(error) = validate_preset_backend_config(&preset) {
                return futures_util::future::ready(Ok::<
                    Box<dyn warp::reply::Reply>,
                    warp::Rejection,
                >(Box::new(
                    warp::reply::with_status(
                        warp::reply::json(
                            &serde_json::json!({"ok": false, "error": error.to_string()}),
                        ),
                        warp::http::StatusCode::BAD_REQUEST,
                    ),
                )));
            }
            if preset.id.trim().is_empty() {
                preset.id = presets::next_id();
            }
            preset.api_key_configured = preset.api_key.as_ref().is_some_and(|key| !key.is_empty());

            // Populate GGUF metadata if model_path is set
            presets::ensure_gguf_metadata(&mut preset);

            let mut presets = state.presets.lock().unwrap();
            let mut candidate = presets.clone();
            candidate.push(preset.clone());
            if let Err(error) = presets::save_presets(&state.presets_path, &candidate) {
                return futures_util::future::ready(Ok::<
                    Box<dyn warp::reply::Reply>,
                    warp::Rejection,
                >(Box::new(
                    warp::reply::with_status(
                        warp::reply::json(
                            &serde_json::json!({"ok": false, "error": error.to_string()}),
                        ),
                        warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                    ),
                )));
            }
            *presets = candidate;
            futures_util::future::ready(Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(
                Box::new(warp::reply::json(
                    &serde_json::json!({"ok": true, "preset": preset_for_api(preset)}),
                )),
            ))
        })
}

fn api_update_preset(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "presets" / String)
        .and(warp::put())
        .and(warp::header::optional::<String>("authorization"))
        .and(safe_json_body::<serde_json::Value>())
        .and_then(
            move |id: String, auth: Option<String>, body: serde_json::Value| {
                let cfg = app_config.clone();
                if !check_api_token(&auth, &cfg) {
                    return futures_util::future::ready(Ok(unauthorized_api_token()));
                }
                let expected_revision = body
                    .get("expected_revision")
                    .and_then(serde_json::Value::as_u64);
                let preset_value = body
                    .get("preset")
                    .cloned()
                    .unwrap_or_else(|| body.clone());
                let mut updated = match serde_json::from_value::<ModelPreset>(preset_value) {
                    Ok(updated) => updated,
                    Err(error) => {
                        return futures_util::future::ready(Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::with_status(
                                warp::reply::json(&serde_json::json!({"ok": false, "code": "invalid_request", "error": error.to_string()})),
                                warp::http::StatusCode::BAD_REQUEST,
                            ),
                        )));
                    }
                };
                updated.id = id.clone();
                presets::migrate_preset(&mut updated);
                if let Err(error) = validate_preset_backend_config(&updated) {
                    return futures_util::future::ready(Ok::<
                        Box<dyn warp::reply::Reply>,
                        warp::Rejection,
                    >(Box::new(
                        warp::reply::with_status(
                            warp::reply::json(
                                &serde_json::json!({"ok": false, "error": error.to_string()}),
                            ),
                            warp::http::StatusCode::BAD_REQUEST,
                        ),
                    )));
                }
                let catalog = state.presets.lock().unwrap().clone();
                let current = catalog.iter().find(|preset| preset.id == id).cloned();
                let Some(current) = current else {
                    return futures_util::future::ready(Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({"ok": false, "code": "not_found", "error": "preset not found"})),
                            warp::http::StatusCode::NOT_FOUND,
                        ),
                    )));
                };
                let Some(expected_revision) = expected_revision else {
                    return futures_util::future::ready(Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({"ok": false, "code": "expected_revision_required", "error": "expected_revision is required"})),
                            warp::http::StatusCode::BAD_REQUEST,
                        ),
                    )));
                };
                if current.bundle.is_some()
                    && let Some(field) = bundle_projection_conflict(&updated)
                {
                    return futures_util::future::ready(Ok::<
                        Box<dyn warp::reply::Reply>,
                        warp::Rejection,
                    >(Box::new(warp::reply::with_status(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "code": "flat_projection_conflict",
                            "field": field,
                            "error": "bundled preset flat fields must match default_selection"
                        })),
                        warp::http::StatusCode::BAD_REQUEST,
                    ))));
                }
                if expected_revision != current.revision {
                    return futures_util::future::ready(Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({"ok": false, "code": "revision_conflict", "error": "preset revision is stale", "revision": current.revision, "catalog_etag": super::preset_bundles::catalog_etag(&catalog)})),
                            warp::http::StatusCode::CONFLICT,
                        ),
                    )));
                }
                let existing_api_key = current.api_key.clone();
                merge_preset_api_key(&mut updated, existing_api_key);

                // If model_path changed, reset GGUF-derived fields so we refresh from new file.
                let previous_model_path = Some(current.model_path.clone());

                // Reset GGUF metadata if model_path changed so we refresh from new file
                if Some(updated.model_path.trim().to_string()) != previous_model_path {
                    updated.clear_gguf_metadata();
                }

                // Populate/refresh GGUF metadata if model_path is set and fields incomplete.
                presets::ensure_gguf_metadata(&mut updated);

                let mut presets = state.presets.lock().unwrap();
                if let Some(idx) = presets.iter().position(|p| p.id == id) {
                    if current.revision != presets[idx].revision {
                        return futures_util::future::ready(Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::with_status(
                                warp::reply::json(&serde_json::json!({"ok": false, "code": "revision_conflict", "error": "preset revision is stale", "revision": presets[idx].revision})),
                                warp::http::StatusCode::CONFLICT,
                            ),
                        )));
                    }
                    updated.revision = current.revision + 1;
                    let mut candidate = presets.clone();
                    candidate[idx] = updated.clone();
                    if let Err(error) = presets::save_presets(&state.presets_path, &candidate) {
                        return futures_util::future::ready(Ok::<
                            Box<dyn warp::reply::Reply>,
                            warp::Rejection,
                        >(Box::new(
                            warp::reply::with_status(
                                warp::reply::json(
                                    &serde_json::json!({"ok": false, "error": error.to_string()}),
                                ),
                                warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                            ),
                        )));
                    }
                    *presets = candidate;
                    let etag = super::preset_bundles::catalog_etag(&presets);
                    futures_util::future::ready(Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(
                        Box::new(warp::reply::json(
                            &serde_json::json!({"ok": true, "preset": preset_for_api(updated), "revision": presets[idx].revision, "catalog_etag": etag}),
                        )),
                    ))
                } else {
                    futures_util::future::ready(Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(
                        Box::new(warp::reply::json(
                            &serde_json::json!({"ok": false, "error": "preset not found"}),
                        )),
                    ))
                }
            },
        )
}

fn api_delete_preset(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "presets" / String)
        .and(warp::delete())
        .and(warp::header::optional::<String>("authorization"))
        .and(safe_json_body::<DeletePresetRequest>())
        .and(with_app_config(app_config))
        .and_then(move |id: String, auth: Option<String>, request: DeletePresetRequest, cfg: Arc<AppConfig>| {
            if !check_db_admin_token(&auth, &cfg) {
                return futures_util::future::ready(Ok(unauthorized_db_admin_token()));
            }
            let Some(expected_revision) = request.expected_revision else {
                return futures_util::future::ready(Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                    warp::reply::with_status(
                        warp::reply::json(&serde_json::json!({"ok": false, "code": "expected_revision_required", "error": "expected_revision is required"})),
                        warp::http::StatusCode::BAD_REQUEST,
                    ),
                )));
            };
                if request.confirmation.as_deref() != Some("DELETE PRESET") {
                return futures_util::future::ready(Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                    warp::reply::with_status(
                        warp::reply::json(&serde_json::json!({"ok": false, "code": "confirmation_required", "error": "confirmation must be DELETE PRESET"})),
                        warp::http::StatusCode::BAD_REQUEST,
                    ),
                    )));
                }
                let mut presets = state.presets.lock().unwrap();
                if let Some(expected_etag) = request.expected_catalog_etag.as_deref()
                    && expected_etag != super::preset_bundles::catalog_etag(&presets)
                {
                    return futures_util::future::ready(Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({"ok": false, "code": "catalog_conflict", "error": "catalog is stale", "catalog_etag": super::preset_bundles::catalog_etag(&presets)})),
                            warp::http::StatusCode::CONFLICT,
                        ),
                    )));
                }
            let Some(current) = presets.iter().find(|preset| preset.id == id) else {
                return futures_util::future::ready(Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                    warp::reply::with_status(
                        warp::reply::json(&serde_json::json!({"ok": false, "code": "not_found", "error": "preset not found"})),
                        warp::http::StatusCode::NOT_FOUND,
                    ),
                )));
            };
            if current.revision != expected_revision {
                return futures_util::future::ready(Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                    warp::reply::with_status(
                        warp::reply::json(&serde_json::json!({"ok": false, "code": "revision_conflict", "error": "preset revision is stale", "revision": current.revision, "catalog_etag": super::preset_bundles::catalog_etag(&presets)})),
                        warp::http::StatusCode::CONFLICT,
                    ),
                )));
            }
            let mut candidate = presets.clone();
            candidate.retain(|p| p.id != id);
                if let Err(error) = presets::save_presets(&state.presets_path, &candidate) {
                return futures_util::future::ready(Ok::<
                    Box<dyn warp::reply::Reply>,
                    warp::Rejection,
                >(Box::new(
                    warp::reply::with_status(
                        warp::reply::json(
                            &serde_json::json!({"ok": false, "error": error.to_string()}),
                        ),
                        warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                    ),
                )));
                }
                *presets = candidate;
                let mut collections = state.preset_collections.lock().unwrap();
                for collection in &mut collections.collections {
                    collection.preset_ids.retain(|preset_id| preset_id != &id);
                }
                if !state.model_tags_path.as_os_str().is_empty() {
                    let collection_dir = state
                        .model_tags_path
                        .parent()
                        .unwrap_or(std::path::Path::new("."))
                        .to_path_buf();
                    if let Err(error) =
                        crate::collections::save_collections(&collection_dir, &collections)
                    {
                        eprintln!(
                            "[warn] preset {id} deleted but collection cleanup could not be persisted: {error}"
                        );
                    }
                }
                let etag = super::preset_bundles::catalog_etag(&presets);
            futures_util::future::ready(Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(
                Box::new(warp::reply::json(&serde_json::json!({"ok": true, "catalog_etag": etag}))),
            ))
        })
}

fn api_reset_presets(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "presets" / "reset")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(safe_json_body::<ResetPresetsRequest>())
        .and(with_app_config(app_config))
        .and_then(move |auth: Option<String>, request: ResetPresetsRequest, cfg: Arc<AppConfig>| {
            if !check_db_admin_token(&auth, &cfg) {
                return futures_util::future::ready(Ok(unauthorized_db_admin_token()));
            }
            if request.confirmation.as_deref() != Some("RESET PRESETS") {
                return futures_util::future::ready(Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                    warp::reply::with_status(
                        warp::reply::json(&serde_json::json!({"ok": false, "code": "confirmation_required", "error": "confirmation must be RESET PRESETS"})),
                        warp::http::StatusCode::BAD_REQUEST,
                    ),
                )));
            }
            let mut presets = state.presets.lock().unwrap();
            let current_etag = super::preset_bundles::catalog_etag(&presets);
            if request.expected_catalog_etag.as_deref() != Some(current_etag.as_str()) {
                return futures_util::future::ready(Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                    warp::reply::with_status(
                        warp::reply::json(&serde_json::json!({"ok": false, "code": "catalog_conflict", "error": "catalog is stale", "catalog_etag": current_etag})),
                        warp::http::StatusCode::CONFLICT,
                    ),
                )));
            }
            let defaults = presets::default_presets();
            if let Err(error) = presets::save_presets(&state.presets_path, &defaults) {
                return futures_util::future::ready(Ok::<
                    Box<dyn warp::reply::Reply>,
                    warp::Rejection,
                >(Box::new(
                    warp::reply::with_status(
                        warp::reply::json(
                            &serde_json::json!({"ok": false, "error": error.to_string()}),
                        ),
                        warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                    ),
                )));
            }
            *presets = defaults;
            let etag = super::preset_bundles::catalog_etag(&presets);
            futures_util::future::ready(Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(
                Box::new(warp::reply::json(&serde_json::json!({"ok": true, "catalog_etag": etag}))),
            ))
        })
}

// ── Template API ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_storage::ChatStorage;
    use crate::config::{self, TLSConfig, TlsMode};
    use crate::gpu::env::GpuEnv;
    use crate::state::AppPaths;
    use crate::web::api::ApiCtx;
    use crate::web::auth::AuthManager;
    use std::path::{Path, PathBuf};

    fn test_context(presets: Vec<ModelPreset>, presets_path: PathBuf) -> ApiCtx {
        let state = AppState::new(
            presets,
            AppPaths {
                presets_path,
                templates_path: PathBuf::new(),
                models_dir: None,
                gpu_env_path: PathBuf::new(),
                ui_settings_path: PathBuf::new(),
                sessions_path: PathBuf::new(),
                model_tags_path: PathBuf::new(),
            },
            GpuEnv::default(),
            crate::state::UiSettings::default(),
            Arc::new(ChatStorage::open(&PathBuf::from(":memory:")).unwrap()),
            TLSConfig::default(),
        );
        ApiCtx {
            state,
            config: Arc::new(config::AppConfig::for_test(
                Some("test-token".into()),
                Some("test-admin".into()),
            )),
            auth: AuthManager::new(None, None, &TlsMode::None),
        }
    }

    /// Occupies `save_presets`'s `.json.tmp` write target with a directory so
    /// the write fails deterministically and portably (no permission bits,
    /// works as root), before the rename that would replace the real file.
    fn break_presets_write(path: &Path) {
        std::fs::create_dir_all(path.with_extension("json.tmp")).unwrap();
    }

    #[tokio::test]
    async fn delete_disk_failure_returns_500_and_leaves_state_unchanged() {
        let preset = ModelPreset {
            id: "preset-1".into(),
            name: "Keep me".into(),
            revision: 1,
            ..Default::default()
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("presets.json");
        break_presets_write(&path);
        let ctx = test_context(vec![preset], path);
        let routes = routes(ctx.clone());

        let response = warp::test::request()
            .method("DELETE")
            .path("/api/presets/preset-1")
            .header("authorization", "Bearer test-admin")
            .header("content-type", "application/json")
            .json(&serde_json::json!({
                "expected_revision": 1,
                "confirmation": "DELETE PRESET"
            }))
            .reply(&routes)
            .await;
        assert_eq!(
            response.status(),
            warp::http::StatusCode::INTERNAL_SERVER_ERROR
        );
        let presets = ctx.state.presets.lock().unwrap();
        assert_eq!(
            presets.len(),
            1,
            "preset must not have been removed from memory"
        );
        assert_eq!(presets[0].id, "preset-1");
    }

    #[tokio::test]
    async fn reset_disk_failure_returns_500_and_leaves_state_unchanged() {
        let preset = ModelPreset {
            id: "preset-1".into(),
            name: "Custom".into(),
            revision: 1,
            ..Default::default()
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("presets.json");
        break_presets_write(&path);
        let ctx = test_context(vec![preset], path);
        let routes = routes(ctx.clone());
        let etag = super::super::preset_bundles::catalog_etag(&ctx.state.presets.lock().unwrap());

        let response = warp::test::request()
            .method("POST")
            .path("/api/presets/reset")
            .header("authorization", "Bearer test-admin")
            .header("content-type", "application/json")
            .json(&serde_json::json!({
                "confirmation": "RESET PRESETS",
                "expected_catalog_etag": etag
            }))
            .reply(&routes)
            .await;
        assert_eq!(
            response.status(),
            warp::http::StatusCode::INTERNAL_SERVER_ERROR
        );
        let presets = ctx.state.presets.lock().unwrap();
        assert_eq!(
            presets.len(),
            1,
            "reset must not have replaced in-memory presets"
        );
        assert_eq!(
            presets[0].id, "preset-1",
            "custom preset must survive a failed reset"
        );
    }

    #[tokio::test]
    async fn create_disk_failure_returns_500_and_leaves_state_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("presets.json");
        break_presets_write(&path);
        let ctx = test_context(vec![], path);
        let routes = routes(ctx.clone());

        let response = warp::test::request()
            .method("POST")
            .path("/api/presets")
            .header("authorization", "Bearer test-token")
            .header("content-type", "application/json")
            .json(&serde_json::json!({
                "id": "",
                "name": "New preset",
                "model_path": "/models/new.gguf"
            }))
            .reply(&routes)
            .await;
        assert_eq!(
            response.status(),
            warp::http::StatusCode::INTERNAL_SERVER_ERROR
        );
        let presets = ctx.state.presets.lock().unwrap();
        assert!(
            presets.is_empty(),
            "no preset must have been added in memory"
        );
    }

    #[tokio::test]
    async fn update_disk_failure_returns_500_and_leaves_state_unchanged() {
        let preset = ModelPreset {
            id: "preset-1".into(),
            name: "Original".into(),
            revision: 1,
            model_path: "/models/original.gguf".into(),
            ..Default::default()
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("presets.json");
        break_presets_write(&path);
        let ctx = test_context(vec![preset], path);
        let routes = routes(ctx.clone());

        let response = warp::test::request()
            .method("PUT")
            .path("/api/presets/preset-1")
            .header("authorization", "Bearer test-token")
            .header("content-type", "application/json")
            .json(&serde_json::json!({
                "expected_revision": 1,
                "preset": {
                    "id": "preset-1",
                    "name": "Renamed",
                    "model_path": "/models/original.gguf"
                }
            }))
            .reply(&routes)
            .await;
        assert_eq!(
            response.status(),
            warp::http::StatusCode::INTERNAL_SERVER_ERROR
        );
        let presets = ctx.state.presets.lock().unwrap();
        assert_eq!(presets.len(), 1);
        assert_eq!(
            presets[0].name, "Original",
            "in-memory preset must be unchanged"
        );
        assert_eq!(presets[0].revision, 1);
    }

    #[test]
    fn api_redacts_key_but_reports_configured_marker() {
        let preset = preset_for_api(ModelPreset {
            api_key: Some("secret".into()),
            ..Default::default()
        });
        let json = serde_json::to_value(preset).unwrap();
        assert!(json["api_key"].is_null());
        assert_eq!(json["api_key_configured"], true);
        assert!(!json.to_string().contains("secret"));
    }

    #[test]
    fn update_preserves_replaces_or_explicitly_clears_existing_key() {
        let mut preserve = ModelPreset::default();
        merge_preset_api_key(&mut preserve, Some("existing".into()));
        assert_eq!(preserve.api_key.as_deref(), Some("existing"));

        let mut replace = ModelPreset {
            api_key: Some("replacement".into()),
            ..Default::default()
        };
        merge_preset_api_key(&mut replace, Some("existing".into()));
        assert_eq!(replace.api_key.as_deref(), Some("replacement"));
        assert!(replace.api_key_configured);

        let mut clear = ModelPreset {
            clear_api_key: true,
            ..Default::default()
        };
        merge_preset_api_key(&mut clear, Some("existing".into()));
        assert!(clear.api_key.is_none());
        assert!(!clear.api_key_configured);
    }
}
