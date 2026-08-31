//! Authenticated bundle resolution and selection mutations.
//!
//! Bundle routes live separately from the legacy preset CRUD handlers so the
//! resolver contract and its revision semantics cannot be accidentally folded
//! into the older flat-preset API.

use std::sync::Arc;

use sha2::{Digest, Sha256};
use warp::Filter;

use crate::config::AppConfig;
use crate::inference::llama_cpp_capabilities::CapabilitySnapshot;
use crate::presets::bundle::BoundedEnum;
use crate::presets::bundle::PresetBundleSelection;
use crate::presets::{self, ModelPreset};
use crate::state::AppState;
use crate::web::safe_json_body;

use super::{
    ApiCtx, ApiReply, ApiRoute, box_reply, check_api_token, unauthorized_api_token, with_app_config,
};

pub(crate) fn routes(ctx: ApiCtx) -> ApiRoute {
    let state = ctx.state;
    let config = ctx.config;
    api_get_preset_cards(state.clone(), config.clone())
        .map(box_reply)
        .or(api_resolve_bundle(state.clone(), config.clone()).map(box_reply))
        .unify()
        .or(api_patch_selection(state.clone(), config.clone()).map(box_reply))
        .unify()
        .or(api_copy_preset(state.clone(), config.clone()).map(box_reply))
        .unify()
        .or(api_convert_to_bundle(state, config).map(box_reply))
        .unify()
        .boxed()
}

#[derive(Debug, Default, serde::Deserialize)]
#[serde(default)]
struct ResolveRequest {
    selection: Option<PresetBundleSelection>,
}

#[derive(Debug, Default, serde::Deserialize)]
#[serde(default)]
struct SelectionPatch {
    expected_revision: Option<u64>,
    selection: PresetBundleSelection,
}

#[derive(Debug, Default, serde::Deserialize)]
#[serde(default)]
struct CopyRequest {
    expected_revision: Option<u64>,
    new_name: String,
}

#[derive(Debug, Default, serde::Deserialize)]
#[serde(default)]
struct ConversionRequest {
    expected_revision: Option<u64>,
    conversion: serde_json::Value,
}

fn api_get_preset_cards(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "preset-cards")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and(with_app_config(app_config))
        .and_then(move |auth: Option<String>, cfg: Arc<AppConfig>| {
            if !check_api_token(&auth, &cfg) {
                return futures_util::future::ready(Ok(unauthorized_api_token()));
            }
            let presets = state.presets.lock().unwrap().clone();
            let cards = presets.iter().map(card_view).collect::<Vec<_>>();
            let etag = catalog_etag(&presets);
            futures_util::future::ready(Ok::<ApiReply, warp::Rejection>(Box::new(
                warp::reply::json(&serde_json::json!({
                    "cards": cards,
                    "catalog_etag": etag,
                })),
            )))
        })
}

fn api_resolve_bundle(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "presets" / String / "resolve")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(safe_json_body::<serde_json::Value>())
        .and(with_app_config(app_config))
        .and_then(
            move |id: String,
                  auth: Option<String>,
                  body: serde_json::Value,
                  cfg: Arc<AppConfig>| {
                let state = state.clone();
                async move {
                    if !check_api_token(&auth, &cfg) {
                        return Ok::<ApiReply, warp::Rejection>(unauthorized_api_token());
                    }
                    let request = match parse_resolve_request(body) {
                        Ok(request) => request,
                        Err(message) => {
                            return Ok(json_error(
                                warp::http::StatusCode::BAD_REQUEST,
                                "invalid_request",
                                message,
                            ));
                        }
                    };
                    let preset = match find_preset(&state, &id) {
                        Some(preset) => preset,
                        None => {
                            return Ok(json_error(
                                warp::http::StatusCode::NOT_FOUND,
                                "not_found",
                                "preset not found",
                            ));
                        }
                    };
                    let capabilities = current_capabilities(&cfg).await;
                    match crate::presets::resolver::resolve_preset(
                        &preset,
                        request.selection.as_ref(),
                        &capabilities,
                    ) {
                        Ok(resolved) => {
                            let selection = request.selection.or_else(|| {
                                preset
                                    .bundle
                                    .as_ref()
                                    .map(|bundle| bundle.default_selection.clone())
                            });
                            Ok(Box::new(warp::reply::json(&resolve_response(
                                &resolved,
                                selection.as_ref(),
                                preset.revision,
                            ))) as ApiReply)
                        }
                        Err(issues) => Ok(json_error(
                            warp::http::StatusCode::BAD_REQUEST,
                            "selection_invalid",
                            serde_json::to_string(&issues)
                                .unwrap_or_else(|_| "selection is invalid".into()),
                        )),
                    }
                }
            },
        )
}

fn api_patch_selection(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "presets" / String / "selection")
        .and(warp::patch())
        .and(warp::header::optional::<String>("authorization"))
        .and(safe_json_body::<SelectionPatch>())
        .and(with_app_config(app_config))
        .and_then(
            move |id: String, auth: Option<String>, patch: SelectionPatch, cfg: Arc<AppConfig>| {
                let state = state.clone();
                async move {
                    if !check_api_token(&auth, &cfg) {
                        return Ok::<ApiReply, warp::Rejection>(unauthorized_api_token());
                    }
                    let capabilities = current_capabilities(&cfg).await;
                    let mut presets = state.presets.lock().unwrap();
                    let Some(index) = presets.iter().position(|preset| preset.id == id) else {
                        return Ok(json_error(
                            warp::http::StatusCode::NOT_FOUND,
                            "not_found",
                            "preset not found",
                        ));
                    };
                    let current = presets[index].clone();
                    let Some(expected_revision) = patch.expected_revision else {
                        return Ok(json_error(
                            warp::http::StatusCode::BAD_REQUEST,
                            "expected_revision_required",
                            "expected_revision is required",
                        ));
                    };
                    if current.revision != expected_revision {
                        return Ok(conflict_response(&current, &presets));
                    }
                    let Some(bundle) = current.bundle.as_ref() else {
                        return Ok(json_error(
                            warp::http::StatusCode::BAD_REQUEST,
                            "not_bundled",
                            "selection updates require a bundled preset",
                        ));
                    };

                    // intent_source is proposal provenance, never persisted by a
                    // mutation route, even when a client supplies it.
                    let mut selection = patch.selection;
                    selection.intent_source = None;
                    let mut candidate = current.clone();
                    candidate.bundle = Some(bundle.clone());
                    candidate.bundle.as_mut().unwrap().default_selection = selection;
                    presets::bundle::materialize_default_projection(&mut candidate);
                    candidate.revision = current.revision + 1;

                    if let Err(issues) =
                        crate::presets::resolver::resolve_preset(&candidate, None, &capabilities)
                    {
                        return Ok(json_error(
                            warp::http::StatusCode::BAD_REQUEST,
                            "selection_invalid",
                            serde_json::to_string(&issues)
                                .unwrap_or_else(|_| "selection is invalid".into()),
                        ));
                    }
                    let mut replacement = presets.clone();
                    replacement[index] = candidate.clone();
                    if let Err(error) = presets::save_presets(&state.presets_path, &replacement) {
                        return Ok(json_error(
                            warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                            "persistence_failed",
                            error.to_string(),
                        ));
                    }
                    *presets = replacement;
                    let etag = catalog_etag(&presets);
                    Ok(Box::new(warp::reply::json(&serde_json::json!({
                        "ok": true,
                        "preset": redacted_preset(candidate),
                        "revision": current.revision + 1,
                        "catalog_etag": etag,
                    }))) as ApiReply)
                }
            },
        )
}

fn api_copy_preset(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "presets" / String / "copy")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(safe_json_body::<CopyRequest>())
        .and(with_app_config(app_config))
        .and_then(
            move |id: String, auth: Option<String>, request: CopyRequest, cfg: Arc<AppConfig>| {
                let state = state.clone();
                async move {
                    if !check_api_token(&auth, &cfg) {
                        return Ok::<ApiReply, warp::Rejection>(unauthorized_api_token());
                    }
                    let Some(expected_revision) = request.expected_revision else {
                        return Ok(json_error(
                            warp::http::StatusCode::BAD_REQUEST,
                            "expected_revision_required",
                            "expected_revision is required",
                        ));
                    };
                    let mut presets = state.presets.lock().unwrap();
                    let Some(index) = presets.iter().position(|preset| preset.id == id) else {
                        return Ok(json_error(
                            warp::http::StatusCode::NOT_FOUND,
                            "not_found",
                            "preset not found",
                        ));
                    };
                    let current = presets[index].clone();
                    if current.revision != expected_revision {
                        return Ok(conflict_response(&current, &presets));
                    }
                    let name = if request.new_name.trim().is_empty() {
                        format!("{} (Copy)", current.name)
                    } else {
                        request.new_name
                    };
                    let mut copy = current
                        .bundle
                        .as_ref()
                        .and_then(|_| presets::bundle::copy_bundle_preset(&current, &name))
                        .unwrap_or_else(|| {
                            let mut flat = current.clone();
                            flat.id = presets::next_id();
                            flat.name = name.clone();
                            flat.revision = 1;
                            flat
                        });
                    copy.revision = 1;
                    let mut replacement = presets.clone();
                    replacement.push(copy.clone());
                    if let Err(error) = presets::save_presets(&state.presets_path, &replacement) {
                        return Ok(json_error(
                            warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                            "persistence_failed",
                            error.to_string(),
                        ));
                    }
                    *presets = replacement;
                    let etag = catalog_etag(&presets);
                    Ok(Box::new(warp::reply::json(&serde_json::json!({
                        "ok": true,
                        "preset": redacted_preset(copy),
                        "revision": 1,
                        "catalog_etag": etag,
                    }))) as ApiReply)
                }
            },
        )
}

fn api_convert_to_bundle(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "presets" / String / "convert-to-bundle")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(safe_json_body::<ConversionRequest>())
        .and(with_app_config(app_config))
        .and_then(
            move |id: String,
                  auth: Option<String>,
                  request: ConversionRequest,
                  cfg: Arc<AppConfig>| {
                let state = state.clone();
                async move {
                    if !check_api_token(&auth, &cfg) {
                        return Ok::<ApiReply, warp::Rejection>(unauthorized_api_token());
                    }
                    let Some(expected_revision) = request.expected_revision else {
                        return Ok(json_error(
                            warp::http::StatusCode::BAD_REQUEST,
                            "expected_revision_required",
                            "expected_revision is required",
                        ));
                    };
                    let mut presets = state.presets.lock().unwrap();
                    let Some(index) = presets.iter().position(|preset| preset.id == id) else {
                        return Ok(json_error(
                            warp::http::StatusCode::NOT_FOUND,
                            "not_found",
                            "preset not found",
                        ));
                    };
                    let current = presets[index].clone();
                    if current.revision != expected_revision {
                        return Ok(conflict_response(&current, &presets));
                    }
                    if current.bundle.is_some() {
                        return Ok(json_error(
                            warp::http::StatusCode::BAD_REQUEST,
                            "already_bundled",
                            "preset is already a bundle",
                        ));
                    }
                    let name = request
                        .conversion
                        .get("name")
                        .and_then(serde_json::Value::as_str)
                        .filter(|value| !value.trim().is_empty())
                        .unwrap_or(&current.name);
                    let converted = presets::bundle::convert_flat_preset(&current, name);
                    let mut replacement = presets.clone();
                    replacement[index] = converted.clone();
                    if let Err(error) = presets::save_presets(&state.presets_path, &replacement) {
                        return Ok(json_error(
                            warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                            "persistence_failed",
                            error.to_string(),
                        ));
                    }
                    *presets = replacement;
                    let etag = catalog_etag(&presets);
                    Ok(Box::new(warp::reply::json(&serde_json::json!({
                        "ok": true,
                        "preset": redacted_preset(converted),
                        "revision": 1,
                        "catalog_etag": etag,
                    }))) as ApiReply)
                }
            },
        )
}

fn parse_resolve_request(body: serde_json::Value) -> Result<ResolveRequest, String> {
    if body.get("selection").is_some() {
        serde_json::from_value(body).map_err(|error| error.to_string())
    } else if body.is_null() || body == serde_json::json!({}) {
        Ok(ResolveRequest::default())
    } else {
        serde_json::from_value(serde_json::json!({ "selection": body }))
            .map_err(|error| error.to_string())
    }
}

pub(crate) async fn current_capabilities(config: &AppConfig) -> CapabilitySnapshot {
    if config.llama_server_path.is_file()
        && let Ok(snapshot) =
            crate::inference::llama_cpp_capabilities::generate_snapshot(&config.llama_server_path)
                .await
    {
        return snapshot;
    }
    CapabilitySnapshot::product_default()
}

fn find_preset(state: &AppState, id: &str) -> Option<ModelPreset> {
    state
        .presets
        .lock()
        .unwrap()
        .iter()
        .find(|preset| preset.id == id)
        .cloned()
}

fn resolve_response(
    resolved: &crate::presets::resolver::ResolvedLaunch,
    selection: Option<&PresetBundleSelection>,
    revision: u64,
) -> serde_json::Value {
    let changes = resolved
        .changes
        .iter()
        .map(|change| {
            let mut value = serde_json::to_value(change).unwrap_or_default();
            if change.field == "model_path" {
                value["before"] = serde_json::Value::Null;
                value["after"] = serde_json::Value::String("selected artifact".into());
            }
            value
        })
        .collect::<Vec<_>>();
    serde_json::json!({
        "ok": true,
        "selection": selection,
        "changes": changes,
        "estimate": {"status": "not_applicable", "code": "phase_not_implemented"},
        "capability_reasons": [],
        "evidence": null,
        "selection_hash": resolved.selection_hash,
        "resolved_config_hash": resolved.config_hash,
        "revision": revision,
    })
}

fn card_view(preset: &ModelPreset) -> serde_json::Value {
    let artifacts = preset
        .bundle
        .as_ref()
        .map(|bundle| {
            bundle
                .artifacts
                .iter()
                .map(|artifact| {
                    serde_json::json!({
                        "id": artifact.id,
                        "role": artifact.role.to_wire(),
                        "display_name": artifact.display_name,
                        "available": artifact.local_path.is_some(),
                        "quantization": artifact.quantization.value,
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    serde_json::json!({
        "id": preset.id,
        "revision": preset.revision,
        "name": preset.name,
        "backend": preset.backend,
        "bundle": preset.bundle.as_ref().map(|bundle| serde_json::json!({
            "bundle_id": bundle.identity.bundle_id,
            "tune_id": bundle.identity.tune_id,
            "display_name": bundle.identity.display_name,
            "default_selection": bundle.default_selection,
            "workload_policy": bundle.workload_policy.to_wire(),
            "artifacts": artifacts,
        })),
    })
}

fn redacted_preset(mut preset: ModelPreset) -> serde_json::Value {
    preset.api_key = None;
    preset.clear_api_key = false;
    serde_json::to_value(preset).unwrap_or_else(|_| serde_json::json!({"id": ""}))
}

pub(crate) fn catalog_etag(presets: &[ModelPreset]) -> String {
    let mut entries = presets
        .iter()
        .map(|preset| serde_json::json!([preset.id, preset.revision]))
        .collect::<Vec<_>>();
    entries.sort_by(|left, right| left[0].as_str().cmp(&right[0].as_str()));
    let bytes = serde_json::to_vec(&entries).expect("catalog etag serializes");
    let digest = Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("catalog-v1:{digest}")
}

fn conflict_response(current: &ModelPreset, presets: &[ModelPreset]) -> ApiReply {
    Box::new(warp::reply::with_status(
        warp::reply::json(&serde_json::json!({
            "ok": false,
            "error": "revision conflict",
            "code": "revision_conflict",
            "revision": current.revision,
            "catalog_etag": catalog_etag(presets),
        })),
        warp::http::StatusCode::CONFLICT,
    ))
}

fn json_error(status: warp::http::StatusCode, code: &str, message: impl Into<String>) -> ApiReply {
    Box::new(warp::reply::with_status(
        warp::reply::json(&serde_json::json!({
            "ok": false,
            "code": code,
            "error": message.into(),
        })),
        status,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_storage::ChatStorage;
    use crate::config::{self, TLSConfig, TlsMode};
    use crate::gpu::env::GpuEnv;
    use crate::state::AppPaths;
    use crate::web::auth::AuthManager;
    use std::path::PathBuf;
    use std::sync::Arc;

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

    #[test]
    fn catalog_etag_is_order_independent_and_revision_sensitive() {
        let mut left = ModelPreset {
            id: "b".into(),
            revision: 1,
            ..Default::default()
        };
        let right = ModelPreset {
            id: "a".into(),
            revision: 2,
            ..Default::default()
        };
        let first = catalog_etag(&[left.clone(), right.clone()]);
        left.revision = 2;
        let second = catalog_etag(&[right, left]);
        assert_ne!(first, second);
    }

    #[test]
    fn resolve_request_accepts_wrapped_and_direct_selection() {
        let wrapped = parse_resolve_request(serde_json::json!({"selection": {}})).unwrap();
        let direct = parse_resolve_request(serde_json::json!({})).unwrap();
        assert!(wrapped.selection.is_some());
        assert!(direct.selection.is_none());
    }

    #[test]
    fn resolve_response_and_cards_never_expose_local_paths_or_keys() {
        let mut preset = ModelPreset::default();
        preset.id = "preset-1".into();
        preset.model_path = "/private/models/secret.gguf".into();
        preset.api_key = Some("secret-key".into());
        let card = card_view(&preset).to_string();
        assert!(!card.contains("secret.gguf"));
        assert!(!card.contains("secret-key"));

        let resolved = crate::presets::resolver::ResolvedLaunch {
            preset,
            selection_hash: "sel-v1:test".into(),
            config_hash: "cfg-v1:test".into(),
            changes: vec![crate::presets::resolver::ResolvedChange {
                code: "artifact_changed".into(),
                field: "model_path".into(),
                before: Some("/private/models/old.gguf".into()),
                after: "/private/models/secret.gguf".into(),
                explanation: "artifact selection".into(),
                source_policy: None,
            }],
            estimate: None,
            evidence: None,
        };
        let response = resolve_response(&resolved, None, 1).to_string();
        assert!(!response.contains("secret.gguf"));
        assert!(!response.contains("secret-key"));
    }

    #[tokio::test]
    async fn cards_require_api_token() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = test_context(Vec::new(), dir.path().join("presets.json"));
        let routes = routes(ctx);
        let response = warp::test::request()
            .method("GET")
            .path("/api/preset-cards")
            .reply(&routes)
            .await;
        assert_eq!(response.status(), warp::http::StatusCode::UNAUTHORIZED);

        let response = warp::test::request()
            .method("GET")
            .path("/api/preset-cards")
            .header("authorization", "Bearer test-token")
            .reply(&routes)
            .await;
        assert_eq!(response.status(), warp::http::StatusCode::OK);
    }

    #[tokio::test]
    async fn resolve_requires_auth_and_does_not_write() {
        let mut preset = ModelPreset::default();
        preset.id = "bundle-1".into();
        preset.bundle = Some(crate::presets::bundle::PresetBundleSpec::default());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("presets.json");
        let ctx = test_context(vec![preset], path.clone());
        let routes = routes(ctx);

        let unauthorized = warp::test::request()
            .method("POST")
            .path("/api/presets/bundle-1/resolve")
            .header("content-type", "application/json")
            .json(&serde_json::json!({}))
            .reply(&routes)
            .await;
        assert_eq!(unauthorized.status(), warp::http::StatusCode::UNAUTHORIZED);

        let response = warp::test::request()
            .method("POST")
            .path("/api/presets/bundle-1/resolve")
            .header("authorization", "Bearer test-token")
            .header("content-type", "application/json")
            .json(&serde_json::json!({}))
            .reply(&routes)
            .await;
        assert_eq!(response.status(), warp::http::StatusCode::BAD_REQUEST);
        assert!(!path.exists(), "resolve must not create or rewrite presets");
    }

    #[tokio::test]
    async fn copy_checks_revision_and_assigns_new_server_revision() {
        let mut preset = ModelPreset::default();
        preset.id = "bundle-1".into();
        preset.name = "Original".into();
        preset.revision = 1;
        preset.bundle = Some(crate::presets::bundle::PresetBundleSpec::default());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("presets.json");
        let ctx = test_context(vec![preset], path.clone());
        let routes = routes(ctx.clone());

        let stale = warp::test::request()
            .method("POST")
            .path("/api/presets/bundle-1/copy")
            .header("authorization", "Bearer test-token")
            .header("content-type", "application/json")
            .json(&serde_json::json!({"expected_revision": 0, "new_name": "Copy"}))
            .reply(&routes)
            .await;
        assert_eq!(stale.status(), warp::http::StatusCode::CONFLICT);
        assert_eq!(ctx.state.presets.lock().unwrap().len(), 1);

        let response = warp::test::request()
            .method("POST")
            .path("/api/presets/bundle-1/copy")
            .header("authorization", "Bearer test-token")
            .header("content-type", "application/json")
            .json(&serde_json::json!({"expected_revision": 1, "new_name": "Copy"}))
            .reply(&routes)
            .await;
        assert_eq!(response.status(), warp::http::StatusCode::OK);
        let copied = ctx.state.presets.lock().unwrap();
        assert_eq!(copied.len(), 2);
        assert_eq!(copied[1].revision, 1);
        assert_eq!(copied[1].name, "Copy");
        let disk: Vec<ModelPreset> =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(disk.len(), 2);
        assert_eq!(disk[1].revision, 1);
    }

    #[tokio::test]
    async fn conversion_checks_revision_and_uses_bundle_constructor_defaults() {
        let mut preset = ModelPreset::default();
        preset.id = "flat-1".into();
        preset.name = "Flat".into();
        preset.revision = 1;
        preset.context_size = 32_000;
        preset.ctk = "f16".into();
        preset.ctv = "f16".into();
        preset.batch_size = 2_048;
        preset.ubatch_size = 256;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("presets.json");
        let ctx = test_context(vec![preset], path.clone());
        let routes = routes(ctx.clone());

        let stale = warp::test::request()
            .method("POST")
            .path("/api/presets/flat-1/convert-to-bundle")
            .header("authorization", "Bearer test-token")
            .header("content-type", "application/json")
            .json(&serde_json::json!({"expected_revision": 0, "conversion": {}}))
            .reply(&routes)
            .await;
        assert_eq!(stale.status(), warp::http::StatusCode::CONFLICT);

        let response = warp::test::request()
            .method("POST")
            .path("/api/presets/flat-1/convert-to-bundle")
            .header("authorization", "Bearer test-token")
            .header("content-type", "application/json")
            .json(&serde_json::json!({"expected_revision": 1, "conversion": {}}))
            .reply(&routes)
            .await;
        assert_eq!(response.status(), warp::http::StatusCode::OK);
        let converted = ctx.state.presets.lock().unwrap()[0].clone();
        let bundle = converted.bundle.as_ref().unwrap();
        assert_eq!(converted.revision, 1);
        assert_eq!(converted.fit_enabled, Some(false));
        assert_eq!(bundle.workload_policy.to_wire(), "custom_unknown");
        assert_eq!(bundle.default_selection.context_size, 32_000);
        let disk: Vec<ModelPreset> =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert!(disk[0].bundle.is_some());
    }

    #[tokio::test]
    async fn bundled_put_rejects_conflicting_flat_projection() {
        let mut bundle = crate::presets::bundle::PresetBundleSpec::default();
        let mut weights = crate::presets::bundle::PresetModelArtifact::default();
        weights.id = "weights".into();
        weights.local_path = Some("/models/model.gguf".into());
        bundle.artifacts = vec![weights];
        bundle.context_options = vec![32_000];
        bundle.kv_policy_options = vec![crate::presets::bundle::LlamaKvPolicyId::F16F16];
        bundle.performance_options = vec![crate::presets::bundle::PresetPerformanceOption {
            id: "balanced".into(),
            label: "2048 / 256".into(),
            batch_size: 2_048,
            ubatch_size: 256,
        }];
        bundle.cpu_moe_options = vec![0];
        bundle.default_selection = crate::presets::bundle::PresetBundleSelection {
            artifact_id: "weights".into(),
            context_size: 32_000,
            kv_policy: crate::presets::bundle::LlamaKvPolicyId::F16F16,
            performance_id: "balanced".into(),
            n_cpu_moe: Some(0),
            intent_source: None,
        };
        bundle.curated_selections = vec![bundle.default_selection.clone()];
        bundle.allow_validated_custom = true;
        let mut preset = crate::presets::bundle::create_bundle_preset("Bundle", bundle);
        preset.id = "bundle-1".into();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("presets.json");
        let ctx = test_context(vec![preset.clone()], path);
        let routes = crate::web::api::api_routes(
            ctx.state.clone(),
            ctx.config.clone(),
            ctx.auth.clone(),
            "127.0.0.1".into(),
        );
        let mut submitted = serde_json::to_value(&preset).unwrap();
        submitted["context_size"] = serde_json::json!(64_000);
        let response = warp::test::request()
            .method("PUT")
            .path("/api/presets/bundle-1")
            .header("authorization", "Bearer test-token")
            .header("content-type", "application/json")
            .json(&serde_json::json!({
                "expected_revision": 1,
                "preset": submitted
            }))
            .reply(&routes)
            .await;
        assert_eq!(response.status(), warp::http::StatusCode::BAD_REQUEST);
        assert_eq!(response.body()[0], b'{');
        assert_eq!(ctx.state.presets.lock().unwrap()[0].revision, 1);
    }

    #[tokio::test]
    async fn selection_patch_is_revision_guarded_and_persisted_before_response() {
        let mut preset = crate::presets::bundle::create_bundle_preset("Bundle", {
            let mut bundle = crate::presets::bundle::PresetBundleSpec::default();
            let mut weights = crate::presets::bundle::PresetModelArtifact::default();
            weights.id = "weights".into();
            weights.local_path = Some("/models/model.gguf".into());
            bundle.artifacts = vec![weights];
            bundle.context_options = vec![160_000, 200_000];
            bundle.kv_policy_options = vec![crate::presets::bundle::LlamaKvPolicyId::Q4Q4];
            bundle.performance_options = vec![crate::presets::bundle::PresetPerformanceOption {
                id: "balanced".into(),
                label: "2048 / 256".into(),
                batch_size: 2048,
                ubatch_size: 256,
            }];
            bundle.cpu_moe_options = vec![0];
            bundle.allow_validated_custom = true;
            let selection = crate::presets::bundle::PresetBundleSelection {
                artifact_id: "weights".into(),
                context_size: 160_000,
                kv_policy: crate::presets::bundle::LlamaKvPolicyId::Q4Q4,
                performance_id: "balanced".into(),
                n_cpu_moe: Some(0),
                intent_source: None,
            };
            bundle.curated_selections = vec![selection.clone()];
            bundle.default_selection = selection;
            bundle
        });
        preset.id = "bundle-1".into();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("presets.json");
        let ctx = test_context(vec![preset], path.clone());
        let routes = routes(ctx.clone());
        let body = serde_json::json!({
            "expected_revision": 1,
            "selection": {
                "artifact_id": "weights",
                "context_size": 200000,
                "kv_policy": "q4_0_q4_0",
                "performance_id": "balanced",
                "n_cpu_moe": 0,
                "intent_source": "low_vram"
            }
        });
        let response = warp::test::request()
            .method("PATCH")
            .path("/api/presets/bundle-1/selection")
            .header("authorization", "Bearer test-token")
            .header("content-type", "application/json")
            .json(&body)
            .reply(&routes)
            .await;
        assert_eq!(response.status(), warp::http::StatusCode::OK);
        let saved = ctx.state.presets.lock().unwrap()[0].clone();
        assert_eq!(saved.revision, 2);
        let saved_bundle = saved.bundle.as_ref().unwrap();
        assert_eq!(saved_bundle.default_selection.context_size, 200_000);
        assert!(saved_bundle.default_selection.intent_source.is_none());
        let disk: Vec<ModelPreset> =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(disk[0].revision, 2);

        let stale = warp::test::request()
            .method("PATCH")
            .path("/api/presets/bundle-1/selection")
            .header("authorization", "Bearer test-token")
            .header("content-type", "application/json")
            .json(&body)
            .reply(&routes)
            .await;
        assert_eq!(stale.status(), warp::http::StatusCode::CONFLICT);
    }
}
