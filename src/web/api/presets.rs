use std::sync::Arc;

use warp::Filter;

use crate::config::AppConfig;
use crate::inference::launch::validate_preset_backend_config;
use crate::presets::{self, ModelPreset};
use crate::state::AppState;
use crate::web::safe_json_body;

use super::{
    ApiCtx, ApiRoute, box_reply, check_api_token, unauthorized_api_token, with_app_config,
};

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
        .and(safe_json_body::<ModelPreset>())
        .and_then(
            move |id: String, auth: Option<String>, mut updated: ModelPreset| {
                let cfg = app_config.clone();
                if !check_api_token(&auth, &cfg) {
                    return futures_util::future::ready(Ok(unauthorized_api_token()));
                }
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
                let existing_api_key = state
                    .presets
                    .lock()
                    .unwrap()
                    .iter()
                    .find(|preset| preset.id == id)
                    .and_then(|preset| preset.api_key.clone());
                merge_preset_api_key(&mut updated, existing_api_key);

                // If model_path changed, reset GGUF-derived fields so we refresh from new file.
                let previous_model_path = {
                    let presets_guard = state.presets.lock().unwrap();
                    presets_guard
                        .iter()
                        .find(|p| p.id == id)
                        .map(|p| p.model_path.clone())
                };

                // Reset GGUF metadata if model_path changed so we refresh from new file
                if Some(updated.model_path.trim().to_string()) != previous_model_path {
                    updated.clear_gguf_metadata();
                }

                // Populate/refresh GGUF metadata if model_path is set and fields incomplete.
                presets::ensure_gguf_metadata(&mut updated);

                let mut presets = state.presets.lock().unwrap();
                if let Some(idx) = presets.iter().position(|p| p.id == id) {
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
                    futures_util::future::ready(Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(
                        Box::new(warp::reply::json(
                            &serde_json::json!({"ok": true, "preset": preset_for_api(updated)}),
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
        .and_then(move |id: String, auth: Option<String>| {
            let cfg = app_config.clone();
            if !check_api_token(&auth, &cfg) {
                return futures_util::future::ready(Ok(unauthorized_api_token()));
            }
            let mut presets = state.presets.lock().unwrap();
            let mut candidate = presets.clone();
            let before = candidate.len();
            candidate.retain(|p| p.id != id);
            if candidate.len() == before {
                return futures_util::future::ready(Ok::<
                    Box<dyn warp::reply::Reply>,
                    warp::Rejection,
                >(Box::new(
                    warp::reply::json(
                        &serde_json::json!({"ok": false, "error": "preset not found"}),
                    ),
                )));
            }
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
                Box::new(warp::reply::json(&serde_json::json!({"ok": true}))),
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
        .and(with_app_config(app_config))
        .and_then(move |auth: Option<String>, cfg: Arc<AppConfig>| {
            if !check_api_token(&auth, &cfg) {
                return futures_util::future::ready(Ok(unauthorized_api_token()));
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
            let mut presets = state.presets.lock().unwrap();
            *presets = defaults;
            futures_util::future::ready(Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(
                Box::new(warp::reply::json(&serde_json::json!({"ok": true}))),
            ))
        })
}

// ── Template API ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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
