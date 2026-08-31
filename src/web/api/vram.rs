use std::sync::Arc;

use warp::Filter;

use crate::config::AppConfig;
use crate::state::AppState;

use super::common::{
    ApiCtx, ApiRoute, check_api_token, check_db_admin_token, unauthorized_api_token,
    unauthorized_db_admin_token,
};

fn workload_scenario_from_json(
    value: &serde_json::Value,
) -> Option<crate::llama::vram_estimator::WorkloadScenario> {
    value
        .as_str()
        .and_then(crate::llama::vram_estimator::WorkloadScenario::from_profile_or_key)
        .or_else(|| serde_json::from_value(value.clone()).ok())
}

/// Translate Rapid's resolved native cache dtype to the shared estimator's
/// byte-width labels. This is intentionally internal: Rapid request bodies
/// never accept llama.cpp `ctk` / `ctv` vocabulary.
fn rapid_estimator_kv_quants(
    dtype: crate::llama::vram_estimator::execution_policy::KvCacheDtype,
) -> (&'static str, &'static str) {
    use crate::llama::vram_estimator::execution_policy::KvCacheDtype;

    match dtype {
        // The shared estimator calls its two-byte representation `f16`; it is
        // the correct byte-width proxy for Rapid's runtime-reported bf16 cache.
        KvCacheDtype::Bf16 => ("f16", "f16"),
        KvCacheDtype::Int8 => ("q8_0", "q8_0"),
        KvCacheDtype::Int4 => ("q4_0", "q4_0"),
    }
}

/// Fetch a fresh-or-cached llama.cpp capability snapshot for K/V policy
/// checks, mirroring `calibration::executor::calibration_capability_snapshot`.
async fn llama_kv_capability_snapshot(
    config: &AppConfig,
) -> Option<crate::inference::llama_cpp_capabilities::CapabilitySnapshot> {
    if !config.llama_server_path.is_file() {
        return None;
    }
    let _ =
        crate::inference::llama_cpp_capabilities::generate_snapshot(&config.llama_server_path)
            .await;
    crate::inference::llama_cpp_capabilities::ExecutableIdentity::from_path(
        &config.llama_server_path,
    )
    .ok()
    .and_then(|identity| crate::inference::llama_cpp_capabilities::cached_snapshot(&identity))
}

// 7) POST /api/vram-estimate (architecture-aware breakdown)
fn api_vram_estimate_breakdown(
    _state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "vram-estimate")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<serde_json::Value>())
        .and_then(move |auth: Option<String>, body: serde_json::Value| {
            let cfg = app_config.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }

                let model_path = body["model_path"].as_str().unwrap_or("").to_string();
                let n_ctx = body["n_ctx"].as_u64().unwrap_or(4096);
                let gpu_layers = body["gpu_layers"].as_i64().unwrap_or(-1) as i32;
                let explicit_parallel_slots = body["parallel_slots"].as_u64().map(|v| v as u32);
                let ubatch_size = body["ubatch_size"].as_u64().unwrap_or(2048) as u32;
                let ctk = body["ctk"].as_str().unwrap_or("q8_0").to_string();
                let ctv = body["ctv"].as_str().unwrap_or("q8_0").to_string();
                let n_cpu_moe = body["n_cpu_moe"].as_i64().map(|v| v as i32).unwrap_or(0);
                let available_vram_bytes = body["available_vram_bytes"].as_u64().unwrap_or(0);
                let available_ram_bytes = body["available_ram_bytes"].as_u64().unwrap_or(0);
                let mut is_unified_memory = body["is_unified_memory"].as_bool().unwrap_or(false);
                // mmproj_path: path to the vision projector GGUF; size read from disk.
                // mmproj_bytes: explicit size override (used when path is unavailable).
                let mmproj_path = body["mmproj_path"].as_str().unwrap_or("").to_string();
                let mmproj_bytes_override = body["mmproj_bytes"].as_u64();
                // HuggingFace coordinates for pre-download introspection: when there is no
                // local file yet, the GGUF KV header (or MLX config.json) is fetched so the
                // estimate uses the model's real architecture instead of name-based guesses.
                let hf_repo_id = body["hf_repo_id"].as_str().unwrap_or("").to_string();
                let hf_file_path = body["hf_file_path"].as_str().unwrap_or("").to_string();
                let hf_repo_revision = body["hf_repo_revision"].as_str().unwrap_or("main").to_string();
                let model_size_override = body["model_size_bytes"].as_u64();

                // Backend discriminator: `backend` (preferred) or legacy `engine` alias.
                // Defaults to llama.cpp/GGUF for backward compatibility with every existing
                // caller (Spawn Wizard, preset editor, welcome-screen cards, previews).
                let backend_field = body["backend"]
                    .as_str()
                    .or_else(|| body["engine"].as_str())
                    .unwrap_or("llama_cpp");
                let is_rapid_mlx = matches!(backend_field, "rapid_mlx" | "mlx" | "rapid-mlx");
                // Rapid's prefill working width is independently qualified from llama.cpp's
                // micro-batch size. Keep the request vocabularies separate even though both
                // feed the shared estimator's scratch/overhead width internally.
                let prefill_step_size = body["prefill_step_size"]
                    .as_u64()
                    .and_then(|value| u32::try_from(value).ok())
                    .filter(|value| (1..=2048).contains(value))
                    .unwrap_or(512);
                let estimator_work_size = if is_rapid_mlx {
                    prefill_step_size
                } else {
                    ubatch_size
                };

                // ── Rapid-MLX execution policy (Phase 5a Part 5: cross-surface equality) ────
                //
                // Accept Rapid-native vocabulary only: kv_cache_dtype {bf16,int8,int4},
                // reasoning_mode, turboquant_mode {v4,k8v4,none}. Per D1/D2: no llama
                // ctk/ctv vocabulary for Rapid estimates. These map into RapidMlxExecutionPolicy
                // for requested/effective distinction and reasons.
                let rapid_kv_cache_dtype: Option<crate::llama::vram_estimator::execution_policy::KvCacheDtype> =
                    body["kv_cache_dtype"]
                        .as_str()
                        .and_then(|s| {
                            serde_json::from_str(&format!("\"{s}\"")).ok()
                        });
                // llama-monitor always launches Rapid with its qualified reasoning/KV quality
                // profile. The separate thinking-output opt-out does not change estimate math.
                let rapid_reasoning_mode = is_rapid_mlx;
                let rapid_turboquant_mode: Option<crate::llama::vram_estimator::execution_policy::TurboQuantMode> =
                    body["turboquant_mode"]
                        .as_str()
                        .and_then(|s| {
                            serde_json::from_str(&format!("\"{s}\"")).ok()
                        });

                // Construct the execution policy for Rapid-MLX (per D31 eligibility).
                // Unknown/unqualified models → TurboQuant Disabled.
                let rapid_execution_policy = if is_rapid_mlx {
                    crate::llama::vram_estimator::execution_policy::RapidMlxExecutionPolicy::new_with_eligibility(
                        rapid_kv_cache_dtype,
                        rapid_reasoning_mode,
                        rapid_turboquant_mode,
                        crate::llama::vram_estimator::execution_policy::TurboQuantEligibility::NotQualified,
                    )
                } else {
                    Default::default()
                };

                // Rapid retained-cache is an explicit optional MiB reservation.
                // It is never derived from a generic percentage of device memory.
                let retained_cache_mib = body["retained_cache_mib"].as_u64().unwrap_or(0);

                // Resolve (model_size_bytes, arch, evidence) from a local file/directory
                // (preferred) or, failing that, by fetching metadata straight from HuggingFace.
                //
                // For Rapid-MLX, we must handle three model_path shapes:
                //   - a real local directory path (e.g. "/Users/.../models/...")
                //   - an HF-repo-style alias (e.g. "mlx-community/Qwen3-30B-A3B-4bit")
                //   - an explicit hf_repo_id
                //
                // We mirror model_resolver.rs: first try as local directory;
                // if it fails and looks like an alias, treat it as an HF repo ID.
                let (model_size_bytes, mut arch, evidence, native_context_limit, estimator_hf_repo_id) =
                    if is_rapid_mlx {
                    // Rapid-MLX is Apple-Silicon/unified-memory only.
                    is_unified_memory = true;

                    // If model_path is non-empty, try to read it as a local MLX directory.
                    let local_meta = if !model_path.is_empty() {
                        if model_path.contains("..") {
                            return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                warp::reply::json(&serde_json::json!({
                                    "ok": false,
                                    "error": "model_path must not contain '..' path traversal"
                                })),
                            ));
                        }
                        crate::inference::rapid_mlx::mlx_meta::read_mlx_model_profile(
                            std::path::Path::new(&model_path),
                        )
                        .ok() // not a local dir → maybe alias
                    } else {
                        None
                    };

                    // If we have a valid local directory, use it.
                    if let Some(profile) = local_meta {
                        let dir = std::path::Path::new(&model_path);
                        let size = crate::inference::rapid_mlx::mlx_meta::read_mlx_weight_index(dir)
                        .ok()
                        .and_then(|index| crate::inference::rapid_mlx::mlx_meta::resolve_local_weight_bytes(dir, &index))
                        .or(model_size_override)
                        .unwrap_or(0);
                        if size == 0 {
                            return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                warp::reply::json(&serde_json::json!({
                                    "ok": false,
                                    "error": "Could not determine MLX model size from safetensors index or model_size_bytes"
                                })),
                            ));
                        }
                        let param_b = crate::llama::vram_estimator::estimate_param_b_from_size(size, 4.85);
                        let mut arch = crate::llama::vram_estimator::ModelArch::from(&profile);
                        arch.param_b = param_b;
                        arch.bytes_per_layer = if arch.n_layers > 0 {
                            size / arch.n_layers as u64
                        } else {
                            0
                        };
                        let ev = if profile.is_substantive() {
                            crate::llama::vram_estimator::EstimateEvidence::Approximate
                        } else {
                            crate::llama::vram_estimator::EstimateEvidence::Degraded
                        };
                        (size, arch, ev, profile.model_context_limit.map(u64::from), None)
                    } else if is_mlx_hf_repo_alias(&model_path) {
                        // model_path is not a local directory but looks like an HF-repo-style alias
                        // (e.g. "mlx-community/Qwen3-30B-A3B-4bit"). Treat it as hf_repo_id.
                        let effective_repo = model_path.clone();
                        let size = resolve_mlx_hf_size(
                            &effective_repo,
                            model_size_override,
                        ).await;
                        let (size, arch, ev, native_context_limit) = match mlx_hf_estimate_from_repo(
                            &effective_repo,
                            &hf_repo_revision,
                            size,
                        ).await {
                            Ok(res) => res,
                            Err(e) => {
                                return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                    warp::reply::json(&serde_json::json!({
                                        "ok": false,
                                        "error": e
                                    })),
                                ));
                            }
                        };
                        (size, arch, ev, native_context_limit, Some(effective_repo))
                    } else if !hf_repo_id.is_empty() {
                        // Caller provided explicit hf_repo_id
                        let size = resolve_mlx_hf_size(&hf_repo_id, model_size_override).await;
                        let (size, arch, ev, native_context_limit) = match mlx_hf_estimate_from_repo(
                            &hf_repo_id,
                            &hf_repo_revision,
                            size,
                        ).await {
                            Ok(res) => res,
                            Err(e) => {
                                return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                    warp::reply::json(&serde_json::json!({
                                        "ok": false,
                                        "error": e
                                    })),
                                ));
                            }
                        };
                        (size, arch, ev, native_context_limit, Some(hf_repo_id.clone()))
                    } else {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({
                                "ok": false,
                                "error": "model_path, or hf_repo_id (+ optional hf_file_path), is required"
                            })),
                        ));
                    }
                } else if !model_path.is_empty() {
                    let size = match std::fs::metadata(&model_path) {
                        Ok(m) => m.len(),
                        Err(e) => {
                            return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                warp::reply::json(&serde_json::json!({
                                    "ok": false,
                                    "error": format!("Cannot stat model file: {e}")
                                })),
                            ));
                        }
                    };
                    let (arch, ev, native_context_limit) = match crate::llama::gguf_meta::read_gguf_metadata(
                        std::path::Path::new(&model_path),
                    ) {
                        Ok(meta) => (
                            meta.to_model_metadata()
                                .to_arch(&model_path, meta.param_b().unwrap_or(0.0)),
                            crate::llama::vram_estimator::EstimateEvidence::Measured,
                            meta.context_length.map(u64::from),
                        ),
                        Err(_) => (
                            // A failed GGUF read is explicitly degraded. Keep only a
                            // size-tier estimate; never infer architecture/capabilities from
                            // the local filename.
                            crate::llama::vram_estimator::ModelArch::from_name_and_params(
                                "",
                                crate::llama::vram_estimator::estimate_param_b_from_size(size, 4.85),
                            ),
                            crate::llama::vram_estimator::EstimateEvidence::Degraded,
                            None,
                        ),
                    };
                    (size, arch, ev, native_context_limit, None)
                } else if !hf_repo_id.is_empty() && !hf_file_path.is_empty() {
                    // Size must be supplied by the caller (from the HF file listing).
                    let size = model_size_override.unwrap_or(0);
                    if size == 0 {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({
                                "ok": false,
                                "error": "model_size_bytes is required when introspecting a HuggingFace model"
                            })),
                        ));
                    }
                    let (arch, ev, native_context_limit) =
                        match crate::hf::fetch_gguf_header_metadata(&hf_repo_id, &hf_file_path).await
                        {
                            Ok(meta) => (
                                meta.to_model_metadata()
                                    .to_arch(&hf_file_path, meta.param_b().unwrap_or(0.0)),
                                crate::llama::vram_estimator::EstimateEvidence::Measured,
                                meta.context_length.map(u64::from),
                            ),
                            // Range-fetch failed (offline / gated / no range support): keep a
                            // size-only degraded estimate; never guess from the HF filename.
                            Err(_) => (
                                crate::llama::vram_estimator::ModelArch::from_name_and_params(
                                    "",
                                    crate::llama::vram_estimator::estimate_param_b_from_size(size, 4.85),
                                ),
                                crate::llama::vram_estimator::EstimateEvidence::Degraded,
                                None,
                            ),
                        };
                    (size, arch, ev, native_context_limit, Some(hf_repo_id.clone()))
                } else {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": "model_path, or hf_repo_id + hf_file_path, is required"
                        })),
                    ));
                };

                // Override mmproj_bytes from explicit path or body field.
                //
                // GGUF only. An MLX vision tower lives *inside* the safetensors weights, which
                // `model_size_bytes` already covers, so adding a projector on top of it would
                // count the tower twice. `mmproj` is a llama.cpp packaging concept with no MLX
                // equivalent, but the frontend sends both fields for whichever model is selected
                // regardless of backend, so a stale mmproj path on a model entry would otherwise
                // silently inflate every MLX estimate.
                if !is_rapid_mlx {
                    if let Some(explicit) = mmproj_bytes_override {
                        arch.mmproj_bytes = explicit;
                    } else if !mmproj_path.is_empty() {
                        arch.mmproj_bytes =
                            std::fs::metadata(&mmproj_path).map(|m| m.len()).unwrap_or(0);
                    }
                }

                let mlx_cache_bytes = if is_rapid_mlx {
                    retained_cache_mib.saturating_mul(1024 * 1024)
                } else { 0 };

                // Builder item 11: accept optional workload_scenario. Scenario-derived parameters
                // only apply when explicit client values are omitted (Phase 2 omission-only rule).
                let workload_scenario = workload_scenario_from_json(&body["workload_scenario"]);

                // Scenario-derived tokens only fill gaps where explicit client values are omitted.
                let scenario_params = workload_scenario.as_ref().map(|s| s.to_estimator_params(
                    crate::llama::vram_estimator::ClientType::App,
                ));
                let explicit_planning = body["rapid_planning_context_tokens"].as_u64();
                let explicit_retained = body["rapid_retained_cache_tokens"].as_u64();
                let rapid_planning_context_tokens = explicit_planning
                    .or_else(|| scenario_params.map(|p| p.planning_context_tokens))
                    .unwrap_or(0);
                let rapid_retained_cache_tokens = explicit_retained
                    .or_else(|| scenario_params.map(|p| p.retained_cache_tokens))
                    .unwrap_or(0);

                // Slot count follows the same omission-only rule as the token counts above. The
                // scenario's own slot count had never been read, so a multi-slot workload was
                // estimated -- and admitted -- as single-slot, which put the D25
                // multi-slot-conflicts-with-MTP warning out of reach of any scenario.
                let parallel_slots = explicit_parallel_slots
                    .or_else(|| scenario_params.map(|p| p.parallel_slots))
                    .unwrap_or(1)
                    .max(1);

                // Product contract: accept the same typed speculative_config used by launch.
                // Never accept caller-authored embedded depth or memory byte counts. Embedded
                // depth comes from the server-parsed model architecture; external sidecars are
                // kept truthful as unknown-memory companions until the server can resolve them.
                let speculative_config = if is_rapid_mlx && !body["speculative_config"].is_null() {
                    let parsed = match serde_json::from_value::<
                        crate::inference::rapid_mlx::RapidMlxSpeculativeConfig,
                    >(body["speculative_config"].clone()) {
                        Ok(config) => config,
                        Err(error) => {
                            return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                warp::reply::json(&serde_json::json!({
                                    "ok": false,
                                    "error": format!("Invalid speculative_config: {error}"),
                                })),
                            ));
                        }
                    };
                    if let Err(error) = parsed.validate() {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({
                                "ok": false,
                                "error": error.to_string(),
                            })),
                        ));
                    }
                    Some(parsed)
                } else {
                    None
                };
                let mtp_config = speculative_config.map(|config| {
                    use crate::llama::vram_estimator::{
                        CompanionMemoryEvidence, CompanionType, ExternalCompanion, MtpConfig,
                        MtpMode,
                    };
                match config.model {
                    Some(model) => {
                        // A managed local sidecar is a separate resident weight file. Use
                        // its actual normalized `mtp.safetensors` size when available;
                        // arbitrary or remote companion references remain unknown rather
                        // than being presented as free memory.
                        let local_bytes = model
                            .starts_with('/')
                            .then(|| {
                                crate::inference::rapid_mlx::sidecar_inventory::estimate_local_companion_vram(
                                    std::path::Path::new(&model),
                                )
                            })
                            .flatten();
                        let (total_bytes, weights_bytes, memory_evidence) =
                            match local_bytes {
                                Some(bytes) => (
                                    bytes,
                                    bytes,
                                    CompanionMemoryEvidence::Approximate,
                                ),
                                None => (0, 0, CompanionMemoryEvidence::Unknown),
                            };
                        MtpConfig {
                            mode: MtpMode::External,
                            embedded_depth: 0,
                            external_drafter: Some(ExternalCompanion {
                                label: "Rapid-MLX MTP sidecar".into(),
                                companion_type: CompanionType::Drafter,
                                total_bytes,
                                weights_bytes,
                                kv_cache_bytes: 0,
                                source: model,
                                memory_evidence,
                            }),
                        }
                    }
                        None => MtpConfig {
                            mode: MtpMode::Embedded,
                            embedded_depth: arch.mtp_depth,
                            external_drafter: None,
                        },
                    }
                });

                // Use the execution policy's effective TurboQuant mode for the estimator.
                // Per D31: effective_turboquant already has eligibility applied.
                let opts = crate::llama::vram_estimator::EstimatorOptions {
                    backend: if is_rapid_mlx {
                        crate::llama::vram_estimator::Backend::RapidMlx
                    } else {
                        crate::llama::vram_estimator::Backend::LlamaCpp
                    },
                    evidence,
                    hf_repo_id: estimator_hf_repo_id,
                    mlx_prefix_cache_bytes: mlx_cache_bytes,
                    turboquant_mode: Some(rapid_execution_policy.effective_turboquant),
                    // Planning context tokens: explicit > scenario > 0 (legacy fallback).
                    rapid_planning_context_tokens,
                    rapid_retained_cache_tokens,
                    // TurboQuant eligibility from the execution policy (D31).
                    turboquant_eligibility: rapid_execution_policy.turboquant_eligibility,
                    mtp_config,
                    client_type: body["client_type"]
                        .as_str()
                        .and_then(|s| {
                            serde_json::from_str::<crate::llama::vram_estimator::ClientType>(&format!("\"{s}\"")).ok()
                        })
                        .unwrap_or_default(),
                    // The request already states its workload; MTP admission used to
                    // ignore it and admit against the CodingAgent default instead.
                    workload_scenario,
                };

                // `ctk` / `ctv` are llama.cpp vocabulary. Rapid uses its
                // resolved native policy, so do not let a stale llama default
                // decide MLX KV bytes. `f16` is the estimator's two-byte proxy
                // for Rapid's runtime-reported bf16 active compute cache.
                let (estimate_ctk, estimate_ctv) = if is_rapid_mlx {
                    rapid_estimator_kv_quants(rapid_execution_policy.effective_kv_dtype)
                } else {
                    (ctk.as_str(), ctv.as_str())
                };

                let breakdown = crate::llama::vram_estimator::full_estimate(
                    model_size_bytes,
                    &arch,
                    n_ctx,
                    estimate_ctk,
                    estimate_ctv,
                    parallel_slots,
                    estimator_work_size,
                    n_cpu_moe,
                    gpu_layers,
                    available_vram_bytes,
                    available_ram_bytes,
                    is_unified_memory,
                    opts,
                );

                // Builder item 6: canonical serialization includes execution_policy for
                // requested/effective distinction and reasons. Cross-surface equality: every
                // JS surface displays from this same canonical response.
                let execution_policy_json = if is_rapid_mlx {
                    serde_json::to_value(&rapid_execution_policy).unwrap_or(serde_json::Value::Null)
                } else {
                    serde_json::Value::Null
                };
                let workload_scenario_json = workload_scenario
                    .as_ref()
                    .and_then(|s| serde_json::to_value(s).ok())
                    .unwrap_or(serde_json::Value::Null);
                let effective_kv_dtype_json = if is_rapid_mlx {
                    serde_json::to_value(rapid_execution_policy.effective_kv_dtype)
                        .unwrap_or(serde_json::Value::Null)
                } else {
                    serde_json::Value::Null
                };

                // Non-blocking K/V policy signal: llama.cpp vocabulary only, mirrors
                // the hard-gate check enforced at launch/save time without rejecting
                // the estimate itself. Rapid-MLX uses its own KV vocabulary (D31),
                // so no policy applies there.
                let kv_policy_json = if is_rapid_mlx {
                    serde_json::Value::Null
                } else {
                    let issue = match llama_kv_capability_snapshot(&cfg).await {
                        Some(snapshot) => {
                            crate::presets::validation::validate_main_kv_policy(
                                &ctk, &ctv, &snapshot,
                            )
                        }
                        None => None,
                    };
                    serde_json::json!({
                        "valid": issue.is_none(),
                        "issue": issue,
                    })
                };

                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(
                    Box::new(warp::reply::json(&serde_json::json!({
                        "ok": true,
                        "weights_bytes": breakdown.weights_bytes,
                        "kv_cache_bytes": breakdown.kv_cache_bytes,
                        "active_kv_bytes": breakdown.active_kv_bytes,
                        "retained_kv_bytes": breakdown.retained_kv_bytes,
                        "linear_attn_state_bytes": breakdown.linear_attn_state_bytes,
                        "mmproj_bytes": breakdown.mmproj_bytes,
                        "mtp_bytes": breakdown.mtp_bytes,
                        "turboquant_transient_peak_bytes": breakdown.turboquant_transient_peak_bytes,
                        "overhead_bytes": breakdown.overhead_bytes,
                        "total_bytes": breakdown.total_bytes,
                        "available_bytes": breakdown.available_bytes,
                        "headroom_bytes": breakdown.headroom_bytes,
                        "ram_bytes": breakdown.ram_bytes,
                        "available_ram_bytes": breakdown.available_ram_bytes,
                        "ram_headroom_bytes": breakdown.ram_headroom_bytes,
                        "recommendation": serde_json::to_value(&breakdown.recommendation).unwrap_or(serde_json::Value::Null),
                        "note": breakdown.note,
                        "mlx_prefix_cache_bytes": breakdown.mlx_prefix_cache_bytes,
                        "evidence": serde_json::to_value(breakdown.evidence).unwrap_or(serde_json::Value::Null),
                        "effective_turboquant": serde_json::to_value(breakdown.effective_turboquant).unwrap_or(serde_json::Value::Null),
                        "mtp_mode": serde_json::to_value(breakdown.mtp_mode).unwrap_or(serde_json::Value::Null),
                        "external_companion": serde_json::to_value(&breakdown.external_companion).unwrap_or(serde_json::Value::Null),
                        "mtp_admission": serde_json::to_value(&breakdown.mtp_admission).unwrap_or(serde_json::Value::Null),
                         "client_type": serde_json::to_value(breakdown.client_type).unwrap_or(serde_json::Value::Null),
                         "execution_policy": execution_policy_json,
                         "workload_scenario": workload_scenario_json,
                         "effective_kv_dtype": effective_kv_dtype_json,
                         "kv_policy": kv_policy_json,
                         "prefill_step_size": if is_rapid_mlx {
                             serde_json::json!(prefill_step_size)
                         } else {
                             serde_json::Value::Null
                         },
                        "mlx_prefix_cache_bytes": breakdown.mlx_prefix_cache_bytes,
                        "native_context_limit": native_context_limit,
                        "context_extension_required": native_context_limit.is_some_and(|limit| n_ctx > limit),
                       }))),
                )
            }
        })
}

// 4b) POST /api/vram/estimate (legacy)
fn api_vram_estimate(
    _state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "vram" / "estimate")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<serde_json::Value>())
        .and_then(move |auth: Option<String>, body: serde_json::Value| {
            let cfg = app_config.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }

                // model: local path used to determine file size (optional when
                // model_size_bytes is provided explicitly).
                let model = body["model"].as_str().unwrap_or("").to_string();
                let context_length = body["context_length"].as_u64().unwrap_or(4096);
                // n_cpu_moe: number of transformer layers whose expert tensors stay
                // on CPU (0 = all expert tensors on GPU).
                let n_cpu_moe = body["n_cpu_moe"].as_i64().map(|v| v as i32);

                // model_size_bytes can be supplied explicitly (e.g. for HF models where
                // there is no local file yet), otherwise inferred from the filesystem.
                let model_size_bytes = body["model_size_bytes"].as_u64().unwrap_or_else(|| {
                    if model.is_empty() {
                        0
                    } else {
                        std::fs::metadata(&model).map(|m| m.len()).unwrap_or(0)
                    }
                });

                if model_size_bytes == 0 {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(
                        Box::new(warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": "Could not determine model size. Provide a local model path or set 'model_size_bytes' explicitly."
                        }))),
                    );
                }

                let kv_quant = body["kv_quant"].as_str().unwrap_or("q8_0").to_string();
                let batch_size = body["batch_size"].as_u64().unwrap_or(2048) as u32;
                let ubatch_size = body["ubatch_size"].as_u64().unwrap_or(2048) as u32;
                let speculative_decoding = body["speculative_decoding"].as_bool().unwrap_or(false);
                let mmproj_size_bytes = body["mmproj_size_bytes"].as_u64().unwrap_or(0);
                let available_vram_bytes = body["available_vram_bytes"].as_u64().unwrap_or(0);

                let estimate = crate::llama::vram_estimator::estimate_vram(
                    model_size_bytes,
                    context_length,
                    &kv_quant,
                    batch_size,
                    ubatch_size,
                    speculative_decoding,
                    mmproj_size_bytes,
                    n_cpu_moe,
                    available_vram_bytes,
                );

                let estimated_vram_mb =
                    (estimate.estimated_vram_bytes as f64) / (1024.0 * 1024.0);

                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(
                    Box::new(warp::reply::json(&serde_json::json!({
                        "ok": true,
                        "estimated_vram_mb": estimated_vram_mb,
                        "estimated_vram_bytes": estimate.estimated_vram_bytes,
                        "estimated_ram_bytes": estimate.estimated_ram_bytes,
                        "available_vram_bytes": estimate.available_vram_bytes,
                        "recommendation": serde_json::to_value(&estimate.recommendation).unwrap_or(serde_json::Value::Null),
                        "note": estimate.note
                    }))),
                )
            }
        })
}

// ── POST /api/vram/quant-compare ─────────────────────────────────────────────
fn api_vram_quant_compare(
    _state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "vram" / "quant-compare")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<serde_json::Value>())
        .and_then(move |auth: Option<String>, body: serde_json::Value| {
            let cfg = app_config.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }

                let param_b = body["param_b"].as_f64().unwrap_or(0.0);
                if param_b <= 0.0 {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": "param_b must be a positive number (model parameter count in billions)"
                        })),
                    ));
                }

                let model_name = body["model_name"].as_str().unwrap_or("").to_string();
                let available_vram_bytes = body["available_vram_bytes"].as_u64().unwrap_or(0);
                let parallel_slots = body["parallel_slots"].as_u64().unwrap_or(1) as u32;
                let backend_field = body["backend"]
                    .as_str()
                    .or_else(|| body["engine"].as_str())
                    .unwrap_or("llama_cpp");
                let is_rapid_mlx = matches!(backend_field, "rapid_mlx" | "mlx" | "rapid-mlx");
                let backend = if is_rapid_mlx {
                    crate::llama::vram_estimator::Backend::RapidMlx
                } else {
                    crate::llama::vram_estimator::Backend::LlamaCpp
                };
                let is_unified_memory =
                    is_rapid_mlx || body["is_unified_memory"].as_bool().unwrap_or(false);

                let use_case = match body["use_case"].as_str().unwrap_or("general") {
                    "agentic" => crate::llama::vram_estimator::UseCase::Agentic,
                    "roleplay" => crate::llama::vram_estimator::UseCase::Roleplay,
                    _ => crate::llama::vram_estimator::UseCase::General,
                };

                // Builder item 11: accept optional workload_scenario for workload-fit quant guidance.
                // When present, replaces generic 8k context with scenario-specific parameters.
                let workload_scenario = workload_scenario_from_json(&body["workload_scenario"]);

                // Optionally accept explicit arch fields to improve accuracy when
                // called after introspection.
                let arch = build_arch_from_body(&body, &model_name, param_b);

                // When the caller has already resolved the HF repo's file listing (real
                // quant names + sizes), use those instead of the synthetic standard-quant
                // set — repos with non-standard naming (imatrix IQ variants, custom
                // mixed-precision schemes like APEX) get an advisor table that matches
                // what's actually downloadable.
                let available_files: Vec<(String, u64)> = body["available_files"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|f| {
                                let name = f["name"].as_str()?.to_string();
                                let size = f["size_bytes"].as_u64()?;
                                Some((name, size))
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                let table = if available_files.is_empty() {
                    crate::llama::vram_estimator::quant_comparison_table(
                        param_b,
                        &arch,
                        &model_name,
                        available_vram_bytes,
                        use_case,
                        workload_scenario,
                        parallel_slots,
                        is_unified_memory,
                        backend,
                    )
                } else {
                    crate::llama::vram_estimator::quant_comparison_table_from_files(
                        param_b,
                        &arch,
                        &model_name,
                        &available_files,
                        available_vram_bytes,
                        use_case,
                        workload_scenario,
                        parallel_slots,
                        is_unified_memory,
                        backend,
                    )
                };

                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                    &serde_json::json!({ "ok": true, "quants": table }),
                )))
            }
        })
}

// ── POST /api/vram/auto-size ──────────────────────────────────────────────────
fn api_vram_auto_size(
    _state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "vram" / "auto-size")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<serde_json::Value>())
        .and_then(move |auth: Option<String>, body: serde_json::Value| {
            let cfg = app_config.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }

                let model_name = body["model_name"].as_str().unwrap_or("").to_string();
                let param_b = body["param_b"].as_f64().unwrap_or(0.0);
                let available_vram_bytes = body["available_vram_bytes"].as_u64().unwrap_or(0);
                let parallel_slots = body["parallel_slots"].as_u64().unwrap_or(1).max(1) as u32;
                let fit_granularity = body["fit_granularity"].as_u64().unwrap_or(1024).max(512);
                let backend_field = body["backend"]
                    .as_str()
                    .or_else(|| body["engine"].as_str())
                    .unwrap_or("llama_cpp");
                let is_rapid_mlx = matches!(backend_field, "rapid_mlx" | "mlx" | "rapid-mlx");
                let backend = if is_rapid_mlx {
                    crate::llama::vram_estimator::Backend::RapidMlx
                } else {
                    crate::llama::vram_estimator::Backend::LlamaCpp
                };
                let is_unified_memory =
                    is_rapid_mlx || body["is_unified_memory"].as_bool().unwrap_or(false);

                let use_case = match body["use_case"].as_str().unwrap_or("general") {
                    "agentic" => crate::llama::vram_estimator::UseCase::Agentic,
                    "roleplay" => crate::llama::vram_estimator::UseCase::Roleplay,
                    _ => crate::llama::vram_estimator::UseCase::General,
                };

                // Model size: explicit bytes > local file stat > param_b heuristic
                let model_size_bytes = body["model_size_bytes"].as_u64().unwrap_or_else(|| {
                    let path = body["model_path"].as_str().unwrap_or("");
                    if !path.is_empty() {
                        std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
                    } else {
                        0
                    }
                });

                // We need *some* size info
                if model_size_bytes == 0 && param_b <= 0.0 {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": "Provide model_size_bytes, model_path, or param_b"
                        })),
                    ));
                }

                // Read GGUF metadata to get authoritative architecture family
                // (e.g. "qwen35" even if the filename says "Pantheon-Reasoning-27B")
                let gguf_read = body["model_path"].as_str().and_then(|path_str| {
                    let path = std::path::Path::new(path_str);
                    path.exists()
                        .then(|| crate::llama::gguf_meta::read_gguf_metadata(path))
                        .transpose()
                        .ok()
                        .flatten()
                });

                // Resolve gguf_arch: prefer GGUF file's general.architecture,
                // then fall back to body field, then empty string.
                // "qwen35" is shared by Qwen3.5 and Qwen3.6 — we distinguish via block_count.
                let (gguf_arch, gguf_block_count, gguf_context_length) = match &gguf_read {
                    Some(meta) => {
                        let arch = meta
                            .architecture
                            .as_deref()
                            .unwrap_or(body["gguf_arch"].as_str().unwrap_or(""))
                            .to_string();
                        let bc = meta.block_count;
                        let ctx = meta.context_length;
                        (arch, bc, ctx)
                    }
                    None => (
                        body["gguf_arch"].as_str().unwrap_or("").to_string(),
                        None,
                        None,
                    ),
                };

                // Map qwen35 to the correct heuristic name using block_count:
                // Qwen3.6 family: ~64 layers (some GGUFs report 65 from extra
                // embedding layers). Qwen3.5 family: 96 layers.
                // Threshold at 75: anything below = Qwen3.6, above = Qwen3.5.
                let resolved_arch = if gguf_arch == "qwen35" {
                    match gguf_block_count {
                        Some(bc) if bc >= 75 => "qwen3_5".to_string(),
                        _ => "qwen3_6".to_string(),
                    }
                } else {
                    gguf_arch.clone()
                };

                // Inject resolved arch into body so build_arch_from_body can use it
                let mut enriched_body = body.clone();
                enriched_body["gguf_arch"] = serde_json::json!(resolved_arch);

                // Also cap auto-size at the model's native context length. GGUF
                // supplies this directly; MLX callers forward the ceiling from
                // the same metadata-backed /api/vram-estimate response.
                let context_cap = gguf_context_length
                    .map(u64::from)
                    .or_else(|| body["native_context_limit"].as_u64());

                // When the GGUF file is present, build the arch straight from its real
                // metadata (full_attention_interval, ssm.*, per-layer head_count_kv,
                // sliding_window, …) — the authoritative source. Only fall back to the
                // body/name heuristic for the pre-download advisor where no file exists.
                let arch = match &gguf_read {
                    Some(meta) => meta.to_model_metadata().to_arch(&model_name, param_b),
                    None => build_arch_from_body(&enriched_body, &model_name, param_b),
                };

                // If model_size_bytes is not given, estimate from param_b + quant
                let quant_hint = body["quant"].as_str().unwrap_or("q4_k_m");
                let model_bytes = if model_size_bytes > 0 {
                    model_size_bytes
                } else {
                    crate::llama::vram_estimator::estimate_model_size_bytes(param_b, quant_hint)
                };

                let result = crate::llama::vram_estimator::auto_size(
                    model_bytes,
                    &arch,
                    available_vram_bytes,
                    use_case,
                    parallel_slots,
                    fit_granularity,
                    is_unified_memory,
                    context_cap, // n_ctx_train cap from GGUF metadata
                    backend,
                );

                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                    &serde_json::json!({ "ok": true, "result": result }),
                )))
            }
        })
}

use crate::llama::vram_estimator::gguf_arch_to_heuristic_name;

/// Build a `ModelArch` from a JSON request body, falling back to heuristics
/// when introspection fields are absent.
///
/// When `gguf_arch` is present in the body, it is used as the authoritative
/// source for the heuristic name instead of the filename (which can be misleading
/// for renamed finetunes like "Qwopus3.6").
pub(crate) fn build_arch_from_body(
    body: &serde_json::Value,
    _model_name: &str,
    param_b: f64,
) -> crate::llama::vram_estimator::ModelArch {
    // GGUF architecture is authoritative when supplied by introspection. With no
    // metadata, retain only a size-tier estimate and mark the caller's result degraded;
    // never infer model properties from a filename/repository label.
    let heuristic_name = body["gguf_arch"]
        .as_str()
        .filter(|value| !value.trim().is_empty())
        .map(gguf_arch_to_heuristic_name)
        .unwrap_or_default();
    let heuristic =
        crate::llama::vram_estimator::ModelArch::from_name_and_params(&heuristic_name, param_b);

    let n_layers = body["n_layers"]
        .as_u64()
        .map(|v| v as u32)
        .unwrap_or(heuristic.n_layers);
    let n_kv_heads = body["n_kv_heads"]
        .as_u64()
        .map(|v| v as u32)
        .unwrap_or(heuristic.n_kv_heads);
    let head_dim = body["head_dim"]
        .as_u64()
        .map(|v| v as u32)
        .unwrap_or(heuristic.head_dim);
    let global_head_dim = body["global_head_dim"]
        .as_u64()
        .map(|v| v as u32)
        .unwrap_or(heuristic.global_head_dim);
    let n_experts = body["n_experts"]
        .as_u64()
        .map(|v| v as u32)
        .unwrap_or(heuristic.n_experts);
    let n_exp_used = body["n_experts_used"]
        .as_u64()
        .map(|v| v as u32)
        .unwrap_or(heuristic.n_experts_used);
    let mtp_depth = body["mtp_depth"]
        .as_u64()
        .map(|v| v as u32)
        .unwrap_or(heuristic.mtp_depth);
    let mmproj_bytes = body["mmproj_bytes"]
        .as_u64()
        .unwrap_or(heuristic.mmproj_bytes);
    let expert_frac = body["expert_fraction"]
        .as_f64()
        .unwrap_or(heuristic.expert_fraction);
    // Exact measured per-layer expert bytes from the GGUF tensor directory (0 =
    // unmeasured → the estimator falls back to expert_fraction).
    let expert_bytes_per_layer = body["expert_bytes_per_layer"].as_u64().unwrap_or(0);
    let moe_layer_count = body["moe_layer_count"]
        .as_u64()
        .map(|v| v as u32)
        .unwrap_or(0);

    // Hybrid DeltaNet: override from body if provided, otherwise preserve heuristic
    let n_attn_layers = body["n_attn_layers"]
        .as_u64()
        .map(|v| v as u32)
        .unwrap_or(heuristic.n_attn_layers);
    let linear_attn_state_bytes = body["linear_attn_state_bytes"]
        .as_u64()
        .unwrap_or(heuristic.linear_attn_state_bytes);
    // Sliding-window (Gemma): override from body if provided
    let n_global_attn_layers = body["n_global_attn_layers"]
        .as_u64()
        .map(|v| v as u32)
        .unwrap_or(heuristic.n_global_attn_layers);
    let local_attn_window = body["local_attn_window"]
        .as_u64()
        .map(|v| v as u32)
        .unwrap_or(heuristic.local_attn_window);
    let local_kv_heads = body["local_kv_heads"]
        .as_u64()
        .map(|v| v as u32)
        .unwrap_or(heuristic.local_kv_heads);

    crate::llama::vram_estimator::ModelArch {
        n_layers,
        n_kv_heads,
        head_dim,
        n_global_attn_layers,
        local_attn_window,
        local_kv_heads,
        n_attn_layers,
        linear_attn_state_bytes,
        n_experts,
        n_experts_used: n_exp_used,
        bytes_per_layer: body["bytes_per_layer"].as_u64().unwrap_or(0),
        expert_fraction: expert_frac,
        expert_bytes_per_layer,
        moe_layer_count,
        global_head_dim,
        mtp_depth,
        mmproj_bytes,
        // n_embd comes from GGUF or heuristic; body can override via "n_embd" field
        n_embd: body["n_embd"]
            .as_u64()
            .map(|v| v as u32)
            .unwrap_or(heuristic.n_embd),
        param_b,
    }
}

// ── Apple Silicon: set Metal GPU wired memory limit ───────────────────────────
// Uses osascript to invoke `sysctl iogpu.wired_limit_mb=N` with administrator
// privileges via the macOS native password dialog. No password touches the app.
// Only compiled on macOS; on other platforms returns a not-supported error.

#[cfg(target_os = "macos")]
fn api_set_metal_gpu_limit(
    _state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "system" / "set-metal-gpu-limit")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<serde_json::Value>())
        .and_then(move |auth: Option<String>, body: serde_json::Value| {
            let cfg = app_config.clone();
            async move {
                // Use db-admin-token: this changes a system-level parameter (iogpu.wired_limit_mb).
                if !check_db_admin_token(&auth, &cfg) {
                    return Ok(unauthorized_db_admin_token());
                }

                let limit_mb = match body["limit_mb"].as_u64() {
                    Some(v) if v > 0 => v,
                    _ => {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({
                                "ok": false,
                                "error": "limit_mb must be a positive integer (MiB)"
                            })),
                        ));
                    }
                };

                // Single-line osascript command (AppleScript string literals cannot span
                // newlines). Use full binary paths so the restricted do-shell-script PATH
                // (/usr/bin:/bin:/usr/sbin:/sbin) is never an issue.
                // Logic: apply sysctl immediately, then upsert the line in /etc/sysctl.conf
                // for persistence across reboots. Subshell grouping avoids if/then/fi.
                let manual_cmd = format!(
                    "sudo /usr/sbin/sysctl -w iogpu.wired_limit_mb={n} && grep -q '^iogpu.wired_limit_mb=' /etc/sysctl.conf 2>/dev/null && sudo /usr/bin/sed -i '' 's/iogpu.wired_limit_mb=.*/iogpu.wired_limit_mb={n}/' /etc/sysctl.conf || echo 'iogpu.wired_limit_mb={n}' | sudo /usr/bin/tee -a /etc/sysctl.conf",
                    n = limit_mb
                );
                let shell_cmd = format!(
                    "/usr/sbin/sysctl iogpu.wired_limit_mb={n} && (/usr/bin/grep -q '^iogpu.wired_limit_mb=' /etc/sysctl.conf 2>/dev/null && /usr/bin/sed -i '' 's/iogpu.wired_limit_mb=.*/iogpu.wired_limit_mb={n}/' /etc/sysctl.conf || /bin/echo 'iogpu.wired_limit_mb={n}' >> /etc/sysctl.conf)",
                    n = limit_mb
                );
                let script = format!(
                    "do shell script \"{cmd}\" with administrator privileges",
                    cmd = shell_cmd.replace('"', "\\\"")
                );

                let run_result = tokio::task::spawn_blocking(move || {
                    std::process::Command::new("/usr/bin/osascript")
                        .args(["-e", &script])
                        .output()
                })
                .await;

                let reply = match run_result {
                    Ok(Ok(output)) if output.status.success() => {
                        let actual = crate::gpu::apple::read_iogpu_wired_limit_mb();
                        if actual >= limit_mb {
                            serde_json::json!({
                                "ok": true,
                                "limit_mb": actual,
                                "note": "Applied immediately and saved to /etc/sysctl.conf — will persist across reboots."
                            })
                        } else {
                            // osascript exited 0 but sysctl read-back shows no change.
                            // Most likely the server PATH can't find sysctl or the
                            // kernel parameter name differs on this macOS version.
                            serde_json::json!({
                                "ok": false,
                                "error": format!(
                                    "osascript exited 0 but iogpu.wired_limit_mb read back as {} MB (expected {}). The setting may not have applied.",
                                    actual, limit_mb
                                ),
                                "manual_cmd": manual_cmd
                            })
                        }
                    }
                    Ok(Ok(output)) => {
                        let stderr = String::from_utf8_lossy(&output.stderr);
                        let stdout = String::from_utf8_lossy(&output.stdout);
                        let combined = format!("{}{}", stdout.trim(), stderr.trim());
                        let msg = if combined.contains("User canceled")
                            || combined.contains("cancelled")
                            || combined.contains("(-128)")
                        {
                            "Cancelled — password dialog was dismissed.".to_string()
                        } else {
                            format!("osascript failed: {combined}")
                        };
                        serde_json::json!({ "ok": false, "error": msg, "manual_cmd": manual_cmd })
                    }
                    Ok(Err(e)) => {
                        serde_json::json!({ "ok": false, "error": format!("Failed to launch osascript: {e}"), "manual_cmd": manual_cmd })
                    }
                    Err(e) => {
                        serde_json::json!({ "ok": false, "error": format!("Task error: {e}"), "manual_cmd": manual_cmd })
                    }
                };

                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                    warp::reply::json(&reply),
                ))
            }
        })
}

fn api_get_system_info(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "system" / "info")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let cfg = app_config.clone();
            let state = state.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }
                let metrics = state.system_metrics.lock().unwrap().clone();
                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                    &serde_json::json!({
                        "ok": true,
                        "p_cores": metrics.p_cores,
                        "e_cores": metrics.e_cores,
                        "cpu_name": metrics.cpu_name,
                    }),
                )))
            }
        })
}

#[cfg(target_os = "macos")]
fn api_get_metal_gpu_limit(
    _state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "system" / "metal-gpu-limit")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let cfg = app_config.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }
                let limit_mb = crate::gpu::apple::read_iogpu_wired_limit_mb();
                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                    &serde_json::json!({
                        "ok": true,
                        "limit_mb": limit_mb,
                        "custom": limit_mb > 0,
                    }),
                )))
            }
        })
}

#[cfg(not(target_os = "macos"))]
fn api_set_metal_gpu_limit(
    _state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "system" / "set-metal-gpu-limit")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<serde_json::Value>())
        .and_then(move |auth: Option<String>, _body: serde_json::Value| {
            let cfg = app_config.clone();
            async move {
                // Use db-admin-token: this changes a system-level parameter.
                if !check_db_admin_token(&auth, &cfg) {
                    return Ok(unauthorized_db_admin_token());
                }
                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                    &serde_json::json!({
                        "ok": false,
                        "error": "Metal GPU limit tuning is only available on macOS."
                    }),
                )))
            }
        })
}

#[cfg(not(target_os = "macos"))]
fn api_get_metal_gpu_limit(
    _state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "system" / "metal-gpu-limit")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let cfg = app_config.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }
                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                    &serde_json::json!({
                        "ok": true,
                        "limit_mb": 0,
                        "custom": false,
                    }),
                )))
            }
        })
}

/// GET /api/memory-availability — returns a live MemoryAvailabilitySnapshot.
///
/// This is the single source of truth for memory availability, used by:
/// - Rapid Wizard (fresh fetch on init, never stale llama cache)
/// - Model Browser / HF preview (configured_ceiling for stable capacity)
/// - Preset editor (current_safe_availability for launch readiness)
///
/// Auth: requires api-token (data-reading endpoint).
/// Performance: sub-second, uses cached system metrics where available.
fn api_memory_availability(
    _state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "memory-availability")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let cfg = app_config.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }

                // Build a fresh snapshot — never stale cache from llama/HF paths.
                let snapshot = crate::memory_availability::build_snapshot();

                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                    &serde_json::json!({
                        "ok": true,
                        "snapshot": snapshot,
                    }),
                )))
            }
        })
}

/// POST /api/memory-availability/fit — evaluate a selected launch against a
/// fresh snapshot. The caller supplies only the selected estimate and explicit
/// launch intent; replace credit is permitted solely for measured app-owned
/// runtime memory.
fn api_memory_availability_fit(
    _state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "memory-availability" / "fit")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<
            crate::memory_availability::MemoryAvailabilityRequest,
        >())
        .and_then(move |auth: Option<String>, request| {
            let cfg = app_config.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }
                let snapshot = crate::memory_availability::build_snapshot_for(&request);
                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                    &serde_json::json!({ "ok": true, "snapshot": snapshot }),
                )))
            }
        })
}

pub(crate) fn routes(ctx: ApiCtx) -> ApiRoute {
    let state = ctx.state.clone();
    let config = ctx.config.clone();

    let mut r = api_vram_estimate_breakdown(state.clone(), config.clone())
        .or(api_vram_estimate(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r
        .or(api_vram_quant_compare(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r
        .or(api_vram_auto_size(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r
        .or(api_get_system_info(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r
        .or(api_get_metal_gpu_limit(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r
        .or(api_set_metal_gpu_limit(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r
        .or(api_memory_availability(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r
        .or(api_memory_availability_fit(state.clone(), config.clone()))
        .unify()
        .boxed();
    r
}

/// Returns true if value looks like an HF-repo-style alias for an MLX model
/// (e.g. "mlx-community/Qwen3-30B-A3B-4bit").
///
/// Criteria mirror model_resolver.rs:
///   - contains '/' (org/repo)
///   - no leading '/' or '\'
///   - no ".." segments
///   - only safe ASCII chars (alphanumeric, -, _, ., /, :)
fn is_mlx_hf_repo_alias(value: &str) -> bool {
    let t = value.trim();
    if t.is_empty() {
        return false;
    }
    if !t.contains('/') {
        return false;
    }
    if t.starts_with('/') || t.starts_with('\\') {
        return false;
    }
    if t.contains("..") {
        return false;
    }
    t.bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'/' | b':'))
}

/// For Rapid-MLX HF-repo introspection: resolve the weight size.
///
/// If model_size_override is already set, use it.
/// Otherwise, query the HF tree API to sum .safetensors sizes.
/// If that fails or returns nothing, falls back to returning 0 (caller must error).
async fn resolve_mlx_hf_size(repo_id: &str, model_size_override: Option<u64>) -> u64 {
    if let Some(s) = model_size_override {
        return s;
    }
    match crate::hf::resolve_mlx_repo_size_bytes(repo_id).await {
        Ok(Some(s)) => s,
        _ => 0,
    }
}

/// Shared HF-repo introspection for MLX: fetch config, build arch, etc.
/// Returns (size, arch, evidence, native_context_limit) or an error string.
async fn mlx_hf_estimate_from_repo(
    repo_id: &str,
    revision: &str,
    size: u64,
) -> Result<
    (
        u64,
        crate::llama::vram_estimator::ModelArch,
        crate::llama::vram_estimator::EstimateEvidence,
        Option<u64>,
    ),
    String,
> {
    if size == 0 {
        return Err(String::from(
            "model_size_bytes is required when introspecting a HuggingFace MLX model",
        ));
    }
    // CRITICAL: Always use config.json for MLX — never hf_file_path.
    // hf_file_path is the model weight file (e.g. model.safetensors), not the config.
    // This prevents the gap 3.7 defect where hf_file_path was misused as a config name.
    match crate::hf::fetch_mlx_model_profile_revision_aware(repo_id, revision).await {
        Ok(profile) => {
            let param_b = crate::llama::vram_estimator::estimate_param_b_from_size(size, 4.85);
            let ev = if profile.is_substantive() {
                crate::llama::vram_estimator::EstimateEvidence::Approximate
            } else {
                crate::llama::vram_estimator::EstimateEvidence::Degraded
            };
            let mut arch = crate::llama::vram_estimator::ModelArch::from(&profile);
            arch.param_b = param_b;
            arch.bytes_per_layer = if arch.n_layers > 0 {
                size / arch.n_layers as u64
            } else {
                0
            };
            Ok((size, arch, ev, profile.model_context_limit.map(u64::from)))
        }
        Err(_) => {
            let arch = crate::llama::vram_estimator::ModelArch::from_name_and_params(
                "",
                crate::llama::vram_estimator::estimate_param_b_from_size(size, 4.85),
            );
            Ok((
                size,
                arch,
                crate::llama::vram_estimator::EstimateEvidence::Degraded,
                None,
            ))
        }
    }
}

#[cfg(test)]
mod mlx_estimate_tests {
    use super::*;
    use crate::llama::vram_estimator::execution_policy::KvCacheDtype;
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

    #[test]
    fn rapid_kv_dtypes_map_to_shared_estimator_byte_widths() {
        assert_eq!(
            rapid_estimator_kv_quants(KvCacheDtype::Bf16),
            ("f16", "f16")
        );
        assert_eq!(
            rapid_estimator_kv_quants(KvCacheDtype::Int8),
            ("q8_0", "q8_0")
        );
        assert_eq!(
            rapid_estimator_kv_quants(KvCacheDtype::Int4),
            ("q4_0", "q4_0")
        );
    }

    /// `/api/vram-estimate` is a data-reading endpoint (it introspects local model files) and
    /// must require `api-token` regardless of which `backend` is requested.
    #[tokio::test]
    async fn vram_estimate_requires_api_token_for_both_backends() {
        for body in [
            r#"{"model_path":"/tmp/does-not-exist.gguf"}"#,
            r#"{"backend":"rapid_mlx","model_path":"/tmp/does-not-exist"}"#,
        ] {
            let response = warp::test::request()
                .method("POST")
                .path("/api/vram-estimate")
                .header("content-type", "application/json")
                .body(body)
                .reply(&test_routes())
                .await;
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED, "{body}");
        }
    }

    /// A `model_path` containing `..` traversal must be rejected for the Rapid-MLX directory
    /// path (mirrors the path-safety rules used elsewhere in this file / `model_resolver.rs`).
    #[tokio::test]
    async fn vram_estimate_rejects_path_traversal_for_mlx_backend() {
        let response = warp::test::request()
            .method("POST")
            .path("/api/vram-estimate")
            .header("authorization", "Bearer api-secret")
            .header("content-type", "application/json")
            .body(r#"{"backend":"rapid_mlx","model_path":"../../etc/passwd"}"#)
            .reply(&test_routes())
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let json: serde_json::Value = serde_json::from_slice(response.body()).unwrap();
        assert_eq!(json["ok"], serde_json::json!(false));
        assert!(json["error"].as_str().unwrap().contains(".."));
    }

    /// A malformed JSON body must return 400, never 404 (API/serialization safety rule).
    #[tokio::test]
    async fn vram_estimate_returns_bad_request_for_malformed_json() {
        let routes = test_routes().recover(crate::web::handle_rejection);
        let response = warp::test::request()
            .method("POST")
            .path("/api/vram-estimate")
            .header("authorization", "Bearer api-secret")
            .header("content-type", "application/json")
            .body("{")
            .reply(&routes)
            .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    /// Requesting the Rapid-MLX backend against a real local MLX model directory produces a
    /// normalized breakdown that carries the MLX-specific fields (`mlx_prefix_cache_bytes`,
    /// `evidence`) and forces unified-memory semantics (Apple-Silicon-only backend).
    #[tokio::test]
    async fn vram_estimate_resolves_local_mlx_directory() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{
                "model_type": "qwen3",
                "hidden_size": 1024,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "max_position_embeddings": 131072
            }"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"a":"model.safetensors"}}"#,
        )
        .unwrap();
        std::fs::write(dir.path().join("model.safetensors"), vec![0u8; 4096]).unwrap();

        let body = serde_json::json!({
            "backend": "rapid_mlx",
            "model_path": dir.path().to_string_lossy(),
            "n_ctx": 4096,
            "prefill_step_size": 1536,
            "available_vram_bytes": 32u64 * 1024 * 1024 * 1024,
        });
        let response = warp::test::request()
            .method("POST")
            .path("/api/vram-estimate")
            .header("authorization", "Bearer api-secret")
            .header("content-type", "application/json")
            .body(body.to_string())
            .reply(&test_routes())
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let json: serde_json::Value = serde_json::from_slice(response.body()).unwrap();
        assert_eq!(json["ok"], serde_json::json!(true));
        assert_eq!(json["mlx_prefix_cache_bytes"], serde_json::json!(0));
        assert_eq!(json["evidence"], serde_json::json!("approximate"));
        assert_eq!(json["native_context_limit"], serde_json::json!(131072));
        assert_eq!(json["context_extension_required"], serde_json::json!(false));
        assert_eq!(json["prefill_step_size"], serde_json::json!(1536));
        assert!(json["weights_bytes"].as_u64().unwrap() > 0);
        // Rapid-MLX has its own KV vocabulary (D31); the llama.cpp-only policy signal
        // stays null on this backend rather than evaluating stale ctk/ctv defaults.
        assert_eq!(json["kv_policy"], serde_json::Value::Null);
    }

    /// The llama.cpp (non-Rapid-MLX) backend carries a non-blocking `kv_policy` signal
    /// mirroring the hard-gate check enforced at launch/save time
    /// (`presets::validation::validate_main_kv_policy`), without rejecting the estimate
    /// itself. With no llama-server binary available in the test config, the capability
    /// snapshot lookup is skipped and the signal defaults to valid.
    #[tokio::test]
    async fn vram_estimate_reports_kv_policy_for_llama_cpp_backend() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model.gguf");
        std::fs::write(&model_path, vec![0u8; 4096]).unwrap();

        let body = serde_json::json!({
            "model_path": model_path.to_string_lossy(),
            "n_ctx": 4096,
            "ctk": "q8_0",
            "ctv": "q8_0",
            "available_vram_bytes": 32u64 * 1024 * 1024 * 1024,
        });
        let response = warp::test::request()
            .method("POST")
            .path("/api/vram-estimate")
            .header("authorization", "Bearer api-secret")
            .header("content-type", "application/json")
            .body(body.to_string())
            .reply(&test_routes())
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let json: serde_json::Value = serde_json::from_slice(response.body()).unwrap();
        assert_eq!(json["ok"], serde_json::json!(true));
        assert_eq!(json["kv_policy"]["valid"], serde_json::json!(true));
        assert_eq!(json["kv_policy"]["issue"], serde_json::Value::Null);
        assert_eq!(json["effective_kv_dtype"], serde_json::Value::Null);
    }

    /// An MLX vision tower is packed inside the safetensors weights, so `model_size_bytes`
    /// already accounts for it. `mmproj` has no MLX equivalent, but the frontend sends
    /// `mmproj_bytes`/`mmproj_path` for the selected model regardless of backend — so a stale
    /// value must not be added on top of weights that already contain the tower.
    #[tokio::test]
    async fn vram_estimate_ignores_mmproj_override_on_the_mlx_path() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{
                "model_type": "qwen3",
                "hidden_size": 1024,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "max_position_embeddings": 131072
            }"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"a":"model.safetensors"}}"#,
        )
        .unwrap();
        std::fs::write(dir.path().join("model.safetensors"), vec![0u8; 4096]).unwrap();

        let estimate_with = |mmproj: u64| {
            let body = serde_json::json!({
                "backend": "rapid_mlx",
                "model_path": dir.path().to_string_lossy(),
                "n_ctx": 4096,
                "available_vram_bytes": 32u64 * 1024 * 1024 * 1024,
                "mmproj_bytes": mmproj,
            });
            async move {
                let response = warp::test::request()
                    .method("POST")
                    .path("/api/vram-estimate")
                    .header("authorization", "Bearer api-secret")
                    .header("content-type", "application/json")
                    .body(body.to_string())
                    .reply(&test_routes())
                    .await;
                assert_eq!(response.status(), StatusCode::OK);
                serde_json::from_slice::<serde_json::Value>(response.body()).unwrap()
            }
        };

        let without = estimate_with(0).await;
        let with_stale_mmproj = estimate_with(4 * 1024 * 1024 * 1024).await;

        assert_eq!(without["ok"], serde_json::json!(true));
        assert_eq!(with_stale_mmproj["ok"], serde_json::json!(true));
        assert_eq!(with_stale_mmproj["mmproj_bytes"], serde_json::json!(0));
        assert_eq!(
            without["total_bytes"], with_stale_mmproj["total_bytes"],
            "a 4 GiB mmproj override changed an MLX estimate; the tower is already in the weights"
        );
    }

    #[tokio::test]
    async fn vram_estimate_uses_nested_mlx_profile_geometry() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{
                "model_type":"wrapper",
                "num_hidden_layers": 99,
                "text_config": {
                    "model_type":"qwen3_6",
                    "hidden_size":1024,
                    "num_hidden_layers":8,
                    "num_attention_heads":8,
                    "num_key_value_heads":2,
                    "head_dim":128,
                    "full_attention_interval":4,
                    "layer_types":["full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention"],
                    "linear_key_head_dim":64,
                    "linear_num_key_heads":2
                }
            }"#,
        )
        .unwrap();
        let body = serde_json::json!({
            "backend": "rapid_mlx",
            "model_path": dir.path().to_string_lossy(),
            "model_size_bytes": 400_000_000u64,
            "n_ctx": 8192,
            "available_vram_bytes": 32u64 * 1024 * 1024 * 1024,
        });
        let response = warp::test::request()
            .method("POST")
            .path("/api/vram-estimate")
            .header("authorization", "Bearer api-secret")
            .header("content-type", "application/json")
            .body(body.to_string())
            .reply(&test_routes())
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let json: serde_json::Value = serde_json::from_slice(response.body()).unwrap();
        assert_eq!(json["ok"], serde_json::json!(true), "{json}");
        assert!(
            json["linear_attn_state_bytes"].as_u64().unwrap() > 0,
            "{json}"
        );
        // Two full-attention layers at 8k must not be calculated as all eight
        // wrapper layers. This is the production-path Qwen3.6 hard gate.
        assert!(
            json["kv_cache_bytes"].as_u64().unwrap() < 2_000_000_000,
            "{json}"
        );
    }

    /// HF-source MLX estimation no longer requires an explicit model_size_bytes: when it is
    /// missing, the endpoint resolves the total weight size from HF's tree API.
    /// This test hits a real repo to ensure the round-trip works (requires network).
    #[tokio::test]
    async fn vram_estimate_mlx_hf_source_resolves_size_automatically() {
        let body = serde_json::json!({
            "backend": "rapid_mlx",
            "hf_repo_id": "mlx-community/Qwen3-0.6B-4bit",
            "n_ctx": 4096,
            "available_vram_bytes": 24u64 * 1024 * 1024 * 1024,
        });
        let response = warp::test::request()
            .method("POST")
            .path("/api/vram-estimate")
            .header("authorization", "Bearer api-secret")
            .header("content-type", "application/json")
            .body(body.to_string())
            .reply(&test_routes())
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let json: serde_json::Value = serde_json::from_slice(response.body()).unwrap();
        assert_eq!(json["ok"], serde_json::json!(true), "{json}");
        assert!(json["weights_bytes"].as_u64().unwrap() > 0);
    }

    /// When model_path is an HF-repo-style alias (not a local directory), and model_size_bytes
    /// is supplied, the endpoint must treat it as an HF repo and return a valid degraded,
    /// size-only estimate when config resolution is unavailable in the unit test.
    #[tokio::test]
    async fn vram_estimate_mlx_treats_hf_style_alias_in_model_path_as_repo() {
        let body = serde_json::json!({
            "backend": "rapid_mlx",
            "model_path": "mlx-community/Qwen3-30B-A3B-4bit",
            "model_size_bytes": 16u64 * 1024 * 1024 * 1024,
            "n_ctx": 4096,
            "available_vram_bytes": 48u64 * 1024 * 1024 * 1024,
        });
        let response = warp::test::request()
            .method("POST")
            .path("/api/vram-estimate")
            .header("authorization", "Bearer api-secret")
            .header("content-type", "application/json")
            .body(body.to_string())
            .reply(&test_routes())
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let json: serde_json::Value = serde_json::from_slice(response.body()).unwrap();
        assert_eq!(json["ok"], serde_json::json!(true), "{json}");
        // Evidence is "approximate" when HF config is fetched and "degraded" when
        // config fetch fails and the endpoint can only use size-tier accounting.
        match json["evidence"].as_str() {
            Some("approximate") | Some("degraded") => {}
            Some(v) => panic!("unexpected evidence: {v}: {json}"),
            None => panic!("missing evidence: {json}"),
        }
        assert!(json["weights_bytes"].as_u64().unwrap() > 0);
    }

    // ── MemoryAvailabilitySnapshot endpoint tests ─────────────────────────────

    /// GET /api/memory-availability requires api-token (data-reading endpoint).
    #[tokio::test]
    async fn memory_availability_requires_api_token() {
        let response = warp::test::request()
            .method("GET")
            .path("/api/memory-availability")
            .reply(&test_routes())
            .await;
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    /// GET /api/memory-availability returns all required fields and never calls
    /// total_unified_bytes "available_memory_bytes".
    #[tokio::test]
    async fn memory_availability_returns_valid_snapshot_shape() {
        let response = warp::test::request()
            .method("GET")
            .path("/api/memory-availability")
            .header("authorization", "Bearer api-secret")
            .reply(&test_routes())
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let json: serde_json::Value = serde_json::from_slice(response.body()).unwrap();
        assert_eq!(json["ok"], serde_json::json!(true));
        let snap = &json["snapshot"];
        assert!(snap.get("total_unified_bytes").is_some());
        assert!(snap.get("free_bytes").is_some());
        assert!(snap.get("wired_bytes").is_some());
        assert!(snap.get("active_bytes").is_some());
        assert!(snap.get("speculative_bytes").is_some());
        assert!(snap.get("pageout_bytes").is_some());
        assert!(snap.get("metal_working_set_bytes").is_some());
        assert!(snap.get("configured_ceiling_bytes").is_some());
        assert!(snap.get("current_safe_availability_bytes").is_some());
        assert!(snap.get("state").is_some());
        assert!(snap.get("backend_specific").is_some());
        assert!(snap.get("timestamp").is_some());
        let body_str = serde_json::to_string(&json).unwrap();
        assert!(
            !body_str.contains("available_memory_bytes"),
            "must NOT call total_unified_bytes 'available_memory_bytes'"
        );
    }

    /// current_safe_availability_bytes must be ≤ configured_ceiling_bytes.
    #[tokio::test]
    async fn memory_availability_current_safe_leq_configured_ceiling() {
        let response = warp::test::request()
            .method("GET")
            .path("/api/memory-availability")
            .header("authorization", "Bearer api-secret")
            .reply(&test_routes())
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let json: serde_json::Value = serde_json::from_slice(response.body()).unwrap();
        let snap = &json["snapshot"];
        let ceiling = snap["configured_ceiling_bytes"].as_u64().unwrap_or(0);
        let safe_avail = snap["current_safe_availability_bytes"]
            .as_u64()
            .unwrap_or(0);
        if ceiling > 0 {
            assert!(
                safe_avail <= ceiling,
                "current_safe_availability_bytes ({}) must be <= configured_ceiling_bytes ({})",
                safe_avail,
                ceiling
            );
        }
    }

    /// state must be one of the four defined values.
    #[tokio::test]
    async fn memory_availability_state_is_valid_enum() {
        let response = warp::test::request()
            .method("GET")
            .path("/api/memory-availability")
            .header("authorization", "Bearer api-secret")
            .reply(&test_routes())
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let json: serde_json::Value = serde_json::from_slice(response.body()).unwrap();
        let state = json["snapshot"]["state"].as_str().unwrap();
        let valid_states = [
            "safe_now",
            "conditional_after_reclaim",
            "after_closing_apps",
            "unsafe",
        ];
        assert!(
            valid_states.contains(&state),
            "state must be one of {:?}, got '{}'",
            valid_states,
            state
        );
    }

    #[tokio::test]
    async fn memory_availability_fit_accepts_target_and_launch_intent() {
        let response = warp::test::request()
            .method("POST")
            .path("/api/memory-availability/fit")
            .header("authorization", "Bearer api-secret")
            .json(&serde_json::json!({
                "required_bytes": 1234,
                "launch_intent": "replace_existing",
                "replace_runtime_bytes": 5678
            }))
            .reply(&test_routes())
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let json: serde_json::Value = serde_json::from_slice(response.body()).unwrap();
        assert_eq!(json["snapshot"]["required_bytes"], 1234);
        assert_eq!(json["snapshot"]["launch_intent"], "replace_existing");
        assert!(json["snapshot"].get("after_reclaim_bytes").is_some());
        assert!(json["snapshot"].get("after_closing_apps_bytes").is_some());
    }
}
