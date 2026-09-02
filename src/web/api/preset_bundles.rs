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
use crate::presets::bundle::{PresetBundleSelection, PresetWorkloadPolicy};
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
    #[serde(deserialize_with = "crate::presets::bundle::bounded_deserialize_opt")]
    workload_policy: Option<PresetWorkloadPolicy>,
    #[serde(default)]
    fit_automatically: bool,
    #[serde(default)]
    available_vram_bytes: u64,
    #[serde(default)]
    available_ram_bytes: u64,
    #[serde(default)]
    is_unified_memory: bool,
    #[serde(default)]
    gpu_layers: Option<i32>,
    #[serde(default)]
    model_size_bytes: Option<u64>,
    #[serde(default)]
    fit_target_mib: Option<u64>,
    #[serde(default)]
    arch: Option<crate::llama::vram_estimator::ModelArch>,
}

#[derive(Debug, Default, serde::Deserialize)]
#[serde(default)]
struct SelectionPatch {
    expected_revision: Option<u64>,
    selection: PresetBundleSelection,
    #[serde(deserialize_with = "crate::presets::bundle::bounded_deserialize_opt")]
    workload_policy: Option<PresetWorkloadPolicy>,
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
                    // Architecture invariant 16: the render kill-switch reaches
                    // the UI only as a closed enum, resolved server-side.
                    "preset_bundle_ui": cfg.preset_bundle_ui.to_wire(),
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
                    let mut resolve_preset = preset.clone();
                    if let Some(workload_policy) = request.workload_policy.clone() {
                        let Some(bundle) = resolve_preset.bundle.as_mut() else {
                            return Ok(json_error(
                                warp::http::StatusCode::BAD_REQUEST,
                                "not_bundled",
                                "workload_policy requires a bundled preset",
                            ));
                        };
                        bundle.workload_policy = workload_policy;
                    }
                    let capabilities = current_capabilities(&cfg).await;
                    let requested_selection = request.selection.clone().or_else(|| {
                        resolve_preset
                            .bundle
                            .as_ref()
                            .map(|bundle| bundle.default_selection.clone())
                    });
                    match crate::presets::resolver::resolve_preset(
                        &resolve_preset,
                        requested_selection.as_ref(),
                        &capabilities,
                    ) {
                        Ok(resolved) => {
                            let (resolved, selection) = if request.fit_automatically {
                                apply_fit_estimate(
                                    resolved,
                                    &resolve_preset,
                                    requested_selection
                                        .as_ref()
                                        .expect("default selection exists for bundled preset"),
                                    &request,
                                    &cfg,
                                    &capabilities,
                                )
                            } else {
                                (resolved, requested_selection)
                            };
                            Ok(Box::new(warp::reply::json(&resolve_response(
                                &resolved,
                                selection.as_ref(),
                                resolve_preset.revision,
                                resolve_preset.bundle.as_ref(),
                                &capabilities,
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
                    let mut candidate_bundle = bundle.clone();
                    if let Some(workload_policy) = patch.workload_policy {
                        candidate_bundle.workload_policy = workload_policy;
                    }
                    candidate.bundle = Some(candidate_bundle);
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
    if body.get("selection").is_some()
        || body.get("workload_policy").is_some()
        || body.get("fit_automatically").is_some()
        || body.get("arch").is_some()
    {
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
    bundle: Option<&crate::presets::bundle::PresetBundleSpec>,
    capabilities: &CapabilitySnapshot,
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
        "estimate": serde_json::to_value(&resolved.estimate_status)
            .unwrap_or_else(|_| serde_json::json!({"status": "unavailable", "code": "serialization_error"})),
        "capability_reasons": bundle
            .map(|bundle| capability_reasons(bundle, capabilities))
            .unwrap_or_default(),
        "evidence": null,
        "selection_hash": resolved.selection_hash,
        "resolved_config_hash": resolved.config_hash,
        "revision": revision,
    })
}

fn capability_reasons(
    bundle: &crate::presets::bundle::PresetBundleSpec,
    capabilities: &CapabilitySnapshot,
) -> Vec<serde_json::Value> {
    let mut reasons = Vec::new();
    if matches!(
        bundle.workload_policy,
        PresetWorkloadPolicy::AgenticTools | PresetWorkloadPolicy::Unknown(_)
    ) {
        reasons.push(serde_json::json!({
            "field": "kv_policy",
            "value": "q4_0_q4_0",
            "reason": "q4_0/q4_0 is not eligible for the selected workload quality floor"
        }));
    }
    if !capabilities.mixed_main_kv.supported {
        reasons.push(serde_json::json!({
            "field": "kv_policy",
            "value": "q8_0_q4_0",
            "reason": "mixed K/V requires a binary advertising mixed_main_kv support"
        }));
    }
    reasons
}

fn apply_fit_estimate(
    mut resolved: crate::presets::resolver::ResolvedLaunch,
    preset: &ModelPreset,
    selection: &PresetBundleSelection,
    request: &ResolveRequest,
    config: &AppConfig,
    capabilities: &CapabilitySnapshot,
) -> (
    crate::presets::resolver::ResolvedLaunch,
    Option<PresetBundleSelection>,
) {
    use crate::llama::vram_estimator::{
        Backend, EstimateEvidence, EstimatorOptions, ModelArch, full_estimate,
    };
    use crate::presets::fit_probe::{FitProbeConfig, ProcessFitReader};
    use crate::presets::fit_search::{FitPlacementResult, FitSearchConfig, search};
    use crate::presets::probe_estimate::{enrich, unsupported_additions};

    let mut disabled = |code: &str, message: String| {
        resolved.estimate_status = crate::presets::resolver::EstimateStatus::Unavailable {
            code: code.into(),
            message,
        };
        (resolved.clone(), Some(selection.clone()))
    };
    let Some(bundle) = preset.bundle.as_ref() else {
        return disabled(
            "preset_not_bundled",
            "fit automatically requires a bundle".into(),
        );
    };
    let Some(artifact) = bundle.artifact(&selection.artifact_id) else {
        return disabled(
            "artifact_not_found",
            "selected artifact is not present in the bundle".into(),
        );
    };
    let Some(model_path) = artifact.local_path.as_ref() else {
        return disabled(
            "artifact_not_local",
            "fit automatically requires a local model artifact".into(),
        );
    };
    let Some(digest) = artifact.digest.as_ref().filter(|digest| {
        digest.algorithm.eq_ignore_ascii_case("sha256")
            && !digest.value.is_empty()
            && digest.coverage == crate::presets::bundle::PresetDigestCoverage::FullFile
    }) else {
        return disabled(
            "artifact_digest_unavailable",
            "fit automatically requires a full-file SHA-256 artifact digest".into(),
        );
    };
    let model_size_bytes = request
        .model_size_bytes
        .or(artifact.size_bytes)
        .unwrap_or_default();
    if model_size_bytes == 0 {
        return disabled(
            "artifact_size_unavailable",
            "fit automatically requires the model artifact size".into(),
        );
    }
    let arch = request.arch.clone().unwrap_or_else(|| {
        ModelArch::from_name_and_params(&preset.name, model_size_bytes as f64 / 1_000_000_000.0)
    });
    let performance = bundle
        .performance_options
        .iter()
        .find(|option| option.id == selection.performance_id);
    let batch_size = performance.map_or(preset.batch_size, |option| option.batch_size);
    let ubatch_size = performance.map_or(preset.ubatch_size, |option| option.ubatch_size);
    let probe_path = match config.llama_fit_params_path.as_ref() {
        Some(path) => path,
        None => {
            return disabled(
                "probe_unavailable",
                "llama_fit_params_path is not configured".into(),
            );
        }
    };
    let mut probe_config =
        FitProbeConfig::new(probe_path.clone(), std::path::PathBuf::from(model_path));
    probe_config.artifact_digest = digest.value.clone();
    probe_config.context_size = selection.context_size;
    probe_config.ctk = preset.ctk.clone();
    probe_config.ctv = preset.ctv.clone();
    probe_config.batch_size = batch_size;
    probe_config.ubatch_size = ubatch_size;
    let probe_timeout = probe_config.timeout;
    let mut reader = match ProcessFitReader::new(probe_config) {
        Ok(reader) => reader,
        Err(error) => return disabled("probe_unavailable", error.to_string()),
    };

    let mut effective_selection = selection.clone();
    let reading = if arch.moe_layer_count > 0 {
        let device_budget_mib = request.available_vram_bytes / (1024 * 1024);
        let host_budget_mib = if request.available_ram_bytes == 0 {
            u64::MAX
        } else {
            request.available_ram_bytes / (1024 * 1024)
        };
        match search(
            &mut reader,
            FitSearchConfig {
                n_max: arch.moe_layer_count,
                device_budget_mib,
                host_budget_mib,
                reserve_mib: request
                    .fit_target_mib
                    .unwrap_or(crate::presets::fit_probe::DEFAULT_FIT_RESERVE_MIB),
                timeout: probe_timeout,
            },
        ) {
            FitPlacementResult::Proposal(proposal) => {
                effective_selection.n_cpu_moe =
                    (proposal.n_cpu_moe > 0).then_some(proposal.n_cpu_moe as i32);
                proposal.reading
            }
            FitPlacementResult::Unavailable(unavailable) => {
                return disabled("probe_unavailable", unavailable.message);
            }
        }
    } else {
        match crate::presets::fit_probe::FitReader::read(&mut reader, 0) {
            Ok(reading) => reading,
            Err(error) => return disabled("probe_unavailable", error.to_string()),
        }
    };

    if effective_selection != *selection {
        resolved = match crate::presets::resolver::resolve_preset(
            preset,
            Some(&effective_selection),
            capabilities,
        ) {
            Ok(resolved) => resolved,
            Err(issues) => {
                return disabled("fit_selection_invalid", issues[0].message.clone());
            }
        };
    }
    let formula = full_estimate(
        model_size_bytes,
        &arch,
        effective_selection.context_size,
        &preset.ctk,
        &preset.ctv,
        preset.parallel_slots,
        ubatch_size,
        effective_selection.n_cpu_moe.unwrap_or(0),
        request.gpu_layers.or(preset.gpu_layers).unwrap_or(-1),
        request.available_vram_bytes,
        request.available_ram_bytes,
        request.is_unified_memory,
        EstimatorOptions {
            backend: Backend::LlamaCpp,
            evidence: EstimateEvidence::Measured,
            ..Default::default()
        },
    );
    let estimate = enrich(
        formula.clone(),
        &reading,
        &unsupported_additions(preset, &formula),
    );
    resolved.estimate = Some(estimate.clone());
    resolved.estimate_status = crate::presets::resolver::EstimateStatus::Available { estimate };
    (resolved, Some(effective_selection))
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
            estimate_status: crate::presets::resolver::EstimateStatus::NotApplicable {
                code: "not_requested".into(),
            },
            evidence: None,
        };
        let response = resolve_response(
            &resolved,
            None,
            1,
            None,
            &CapabilitySnapshot::product_default(),
        )
        .to_string();
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
        let stale = warp::test::request()
            .method("PUT")
            .path("/api/presets/bundle-1")
            .header("authorization", "Bearer test-token")
            .header("content-type", "application/json")
            .json(&serde_json::json!({
                "expected_revision": 0,
                "preset": preset.clone()
            }))
            .reply(&routes)
            .await;
        assert_eq!(stale.status(), warp::http::StatusCode::CONFLICT);

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
    async fn destructive_routes_require_admin_confirmation_and_current_guards() {
        let mut preset = ModelPreset::default();
        preset.id = "preset-1".into();
        preset.name = "Preset".into();
        preset.revision = 1;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("presets.json");
        let ctx = test_context(vec![preset], path.clone());
        let routes = crate::web::api::api_routes(
            ctx.state.clone(),
            ctx.config.clone(),
            ctx.auth.clone(),
            "127.0.0.1".into(),
        );

        let body = serde_json::json!({
            "expected_revision": 1,
            "confirmation": "DELETE PRESET"
        });
        let unauthorized = warp::test::request()
            .method("DELETE")
            .path("/api/presets/preset-1")
            .header("content-type", "application/json")
            .json(&body)
            .reply(&routes)
            .await;
        assert_eq!(unauthorized.status(), warp::http::StatusCode::UNAUTHORIZED);

        let wrong_token = warp::test::request()
            .method("DELETE")
            .path("/api/presets/preset-1")
            .header("authorization", "Bearer test-token")
            .header("content-type", "application/json")
            .json(&body)
            .reply(&routes)
            .await;
        assert_eq!(wrong_token.status(), warp::http::StatusCode::UNAUTHORIZED);

        let stale = warp::test::request()
            .method("DELETE")
            .path("/api/presets/preset-1")
            .header("authorization", "Bearer test-admin")
            .header("content-type", "application/json")
            .json(&serde_json::json!({
                "expected_revision": 0,
                "confirmation": "DELETE PRESET"
            }))
            .reply(&routes)
            .await;
        assert_eq!(stale.status(), warp::http::StatusCode::CONFLICT);
        assert_eq!(ctx.state.presets.lock().unwrap().len(), 1);

        let deleted = warp::test::request()
            .method("DELETE")
            .path("/api/presets/preset-1")
            .header("authorization", "Bearer test-admin")
            .header("content-type", "application/json")
            .json(&body)
            .reply(&routes)
            .await;
        assert_eq!(deleted.status(), warp::http::StatusCode::OK);
        assert!(ctx.state.presets.lock().unwrap().is_empty());
        let disk: Vec<ModelPreset> =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert!(disk.is_empty());

        let reset_preset = ModelPreset {
            id: "preset-2".into(),
            revision: 1,
            ..Default::default()
        };
        *ctx.state.presets.lock().unwrap() = vec![reset_preset];
        let reset_body = serde_json::json!({
            "expected_catalog_etag": "catalog-v1:stale",
            "confirmation": "RESET PRESETS"
        });
        let reset_stale = warp::test::request()
            .method("POST")
            .path("/api/presets/reset")
            .header("authorization", "Bearer test-admin")
            .header("content-type", "application/json")
            .json(&reset_body)
            .reply(&routes)
            .await;
        assert_eq!(reset_stale.status(), warp::http::StatusCode::CONFLICT);
        assert_eq!(ctx.state.presets.lock().unwrap()[0].id, "preset-2");

        let current_etag = catalog_etag(&ctx.state.presets.lock().unwrap());
        let reset = warp::test::request()
            .method("POST")
            .path("/api/presets/reset")
            .header("authorization", "Bearer test-admin")
            .header("content-type", "application/json")
            .json(&serde_json::json!({
                "expected_catalog_etag": current_etag,
                "confirmation": "RESET PRESETS"
            }))
            .reply(&routes)
            .await;
        assert_eq!(reset.status(), warp::http::StatusCode::OK);
        assert!(!ctx.state.presets.lock().unwrap().is_empty());
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
        let blocked = warp::test::request()
            .method("POST")
            .path("/api/presets/bundle-1/resolve")
            .header("authorization", "Bearer test-token")
            .header("content-type", "application/json")
            .json(&serde_json::json!({
                "selection": {
                    "artifact_id": "weights",
                    "context_size": 160000,
                    "kv_policy": "q4_0_q4_0",
                    "performance_id": "balanced",
                    "n_cpu_moe": 0
                },
                "workload_policy": "agentic_tools"
            }))
            .reply(&routes)
            .await;
        assert_eq!(blocked.status(), warp::http::StatusCode::BAD_REQUEST);

        let body = serde_json::json!({
        "expected_revision": 1,
            "selection": {
                "artifact_id": "weights",
                "context_size": 200000,
                "kv_policy": "q4_0_q4_0",
            "performance_id": "balanced",
            "n_cpu_moe": 0,
            "intent_source": "low_vram"
        },
        "workload_policy": "roleplay_creative"
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
        assert_eq!(saved_bundle.workload_policy.to_wire(), "roleplay_creative");
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
