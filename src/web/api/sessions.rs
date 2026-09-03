use std::net::TcpStream;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use warp::Filter;

use crate::config::AppConfig;
use crate::inference::launch::{
    launch_local, launch_local_with_resolved_preset, request_from_api_payload, request_from_preset,
    request_from_session,
};
use crate::llama::server::stop_server;
use crate::presets::bundle::PresetBundleSelection;
use crate::state::{
    self as app_state, AppState, DoctorFinding, DoctorFindingType, DoctorSeverity, FixAction,
    SessionMode, SessionStatus,
};

use super::common::{ApiCtx, ApiError, ApiRoute, box_reply};
use super::common::{
    bearer_matches_api_token, bearer_matches_db_admin_token, check_api_token, check_db_admin_token,
    unauthorized_api_token, unauthorized_db_admin_token, with_app_config,
};

pub(crate) fn routes(ctx: ApiCtx) -> ApiRoute {
    let state = ctx.state;
    let config = ctx.config;

    api_restore_hint(state.clone())
        .map(box_reply)
        .or(api_get_sessions(state.clone(), config.clone()).map(box_reply))
        .unify()
        .or(api_get_recent_sessions(state.clone(), config.clone()).map(box_reply))
        .unify()
        .or(api_check_endpoint_health(config.clone()).map(box_reply))
        .unify()
        .or(api_llama_cpp_diagnostics(state.clone(), config.clone()).map(box_reply))
        .unify()
        .or(api_apply_fix(state.clone(), config.clone()).map(box_reply))
        .unify()
        .or(api_create_session(state.clone(), config.clone()).map(box_reply))
        .unify()
        .or(api_delete_session(state.clone(), config.clone()).map(box_reply))
        .unify()
        .or(api_get_active_session(state.clone(), config.clone()).map(box_reply))
        .unify()
        .or(api_get_active_session_readiness(state.clone(), config.clone()).map(box_reply))
        .unify()
        .or(api_set_active_session(state.clone(), config.clone()).map(box_reply))
        .unify()
        .or(api_get_capabilities(state.clone(), config.clone()).map(box_reply))
        .unify()
        .or(api_spawn_session_with_preset(state.clone(), config.clone()).map(box_reply))
        .unify()
        .or(api_attach(state.clone(), config.clone()).map(box_reply))
        .unify()
        .or(api_detach(state.clone(), config.clone()).map(box_reply))
        .unify()
        .or(api_kill_server(state, config).map(box_reply))
        .unify()
        .boxed()
}

// ==================== RESTORE HINT ENDPOINT ====================

fn api_restore_hint(
    state: AppState,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path("api")
        .and(warp::path("sessions"))
        .and(warp::path("restore-hint"))
        .and(warp::get())
        .and(warp::any().map(move || state.clone()))
        .map(|state: AppState| {
            let server_running = *state.server_running.lock().unwrap();
            let active_id = state.active_session_id.lock().unwrap().clone();

            let sessions = state.sessions.lock().unwrap();
            let has_active_session = sessions
                .iter()
                .any(|s| s.id == active_id && !s.id.is_empty());
            let active_session_id = if has_active_session {
                Some(active_id.clone())
            } else {
                None
            };

            let active_session_status =
                sessions
                    .iter()
                    .find(|s| s.id == active_id)
                    .map(|s| match &s.status {
                        SessionStatus::Running => "Running".to_string(),
                        SessionStatus::Stopped => "Stopped".to_string(),
                        SessionStatus::Disconnected => "Disconnected".to_string(),
                        SessionStatus::Error(_) => "Error".to_string(),
                    });

            let has_chat_tabs = sessions
                .iter()
                .any(|s| s.connect_count > 0 || s.last_connected_at > 0);
            drop(sessions);

            let suggested_action = if server_running
                && has_active_session
                && active_session_status == Some("Running".to_string())
            {
                "resume_active"
            } else if server_running && has_active_session {
                "suggest_recent_attach"
            } else {
                "none"
            };

            warp::reply::json(&serde_json::json!({
                "server_running": server_running,
                "has_active_session": has_active_session,
                "active_session_id": active_session_id,
                "active_session_status": active_session_status,
                "has_chat_tabs": has_chat_tabs,
                "suggested_action": suggested_action
            }))
        })
}

// ==================== SESSION CRUD ====================

fn api_get_sessions(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    let app_config = app_config.clone();
    warp::path!("api" / "sessions")
        .and(warp::path::end())
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and(with_app_config(app_config))
        .and_then(move |auth: Option<String>, cfg: Arc<AppConfig>| {
            let state = state.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }
                let sessions = state.get_sessions();
                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                    &sessions,
                )))
            }
        })
}

fn api_get_recent_sessions(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    let app_config = app_config.clone();
    warp::path!("api" / "sessions" / "recent")
        .and(warp::path::end())
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and(with_app_config(app_config))
        .and_then(move |auth: Option<String>, cfg: Arc<AppConfig>| {
            let state = state.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }
                let mut sessions = state.get_sessions();
                sessions.sort_by_key(|s| std::cmp::Reverse(s.last_connected_at));
                sessions.truncate(10);
                let active_id = state.active_session_id.lock().unwrap().clone();
                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                    &serde_json::json!({
                        "sessions": sessions,
                        "active_session_id": active_id
                    }),
                )))
            }
        })
}

fn api_check_endpoint_health(
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "sessions" / "check-endpoint")
        .and(warp::path::end())
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and(warp::query::<std::collections::HashMap<String, String>>())
        .and(with_app_config(app_config))
        .and_then(
            move |auth: Option<String>,
                  params: std::collections::HashMap<String, String>,
                  cfg: Arc<AppConfig>| async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }
                let url = match params.get("url") {
                    Some(u) if !u.is_empty() => u.clone(),
                    _ => {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::with_status(
                                warp::reply::json(&serde_json::json!({"error": "missing url"})),
                                warp::http::StatusCode::BAD_REQUEST,
                            ),
                        ));
                    }
                };
                let health_url = format!("{}/health", url.trim_end_matches('/'));
                let client = reqwest::Client::builder()
                    .timeout(std::time::Duration::from_secs(4))
                    .build()
                    .unwrap_or_default();
                let reachable = client
                    .get(&health_url)
                    .send()
                    .await
                    .map(|r| r.status().as_u16() < 500)
                    .unwrap_or(false);
                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                    &serde_json::json!({ "reachable": reachable }),
                )))
            },
        )
}

fn api_llama_cpp_diagnostics(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::reply::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "sessions" / "llama-cpp-diagnostics")
        .and(warp::path::end())
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and(with_app_config(app_config))
        .and_then(move |auth: Option<String>, cfg: Arc<AppConfig>| {
            let state = state.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }
                let mut findings = Vec::new();
                findings.extend(check_llama_server_binary(&cfg.llama_server_path));

                // Model/port parity checks use the active session's llama.cpp launch
                // config when available; otherwise these are skipped (nothing to
                // check against — the default model path is preset-specific, not
                // in AppConfig).
                let active_session_id = state.active_session_id.lock().unwrap().clone();
                if !active_session_id.is_empty() {
                    let launch = state
                        .sessions
                        .lock()
                        .unwrap()
                        .iter()
                        .find(|s| s.id == active_session_id)
                        .and_then(|s| s.launch.clone());
                    if let Some(crate::inference::launch::LocalLaunchRequest::LlamaCpp(config)) =
                        launch
                    {
                        if !config.model_path.is_empty() {
                            findings
                                .extend(check_gguf_path(std::path::Path::new(&config.model_path)));
                        }
                        findings.extend(check_port_available(config.port));
                    }
                }

                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                    &serde_json::json!({
                        "ok": true,
                        "findings": findings
                    }),
                )))
            }
        })
}

fn check_llama_server_binary(bin_path: &std::path::Path) -> Vec<DoctorFinding> {
    let mut findings = Vec::new();

    if bin_path.as_os_str().is_empty() {
        findings.push(DoctorFinding {
            finding_type: DoctorFindingType::LlamaCpp,
            severity: DoctorSeverity::Issue,
            message: "llama.cpp server binary path is not configured".to_string(),
            section: "binary".to_string(),
            fix: None,
        });
        return findings;
    }

    if !bin_path.exists() {
        findings.push(DoctorFinding {
            finding_type: DoctorFindingType::LlamaCpp,
            severity: DoctorSeverity::Issue,
            message: format!(
                "llama.cpp server binary not found at {}",
                bin_path.display()
            ),
            section: "binary".to_string(),
            fix: None,
        });
        return findings;
    }

    let version_output = Command::new(bin_path).arg("--help").output();

    match version_output {
        Ok(output) if output.status.success() => {
            let text = String::from_utf8_lossy(&output.stdout);
            if text.contains("llama-server") {
                findings.push(DoctorFinding {
                    finding_type: DoctorFindingType::LlamaCpp,
                    severity: DoctorSeverity::Ok,
                    message: "llama.cpp server binary is present and executable".to_string(),
                    section: "binary".to_string(),
                    fix: None,
                });
            } else {
                findings.push(DoctorFinding {
                    finding_type: DoctorFindingType::LlamaCpp,
                    severity: DoctorSeverity::Warning,
                    message: "llama.cpp server binary exists but help output looks unusual"
                        .to_string(),
                    section: "binary".to_string(),
                    fix: None,
                });
            }
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            findings.push(DoctorFinding {
                finding_type: DoctorFindingType::LlamaCpp,
                severity: DoctorSeverity::Warning,
                message: format!(
                    "llama.cpp server binary exists but returned error: {}",
                    stderr.trim().lines().next().unwrap_or("unknown")
                ),
                section: "binary".to_string(),
                fix: None,
            });
        }
        Err(e) => {
            findings.push(DoctorFinding {
                finding_type: DoctorFindingType::LlamaCpp,
                severity: DoctorSeverity::Issue,
                message: format!("Failed to execute llama.cpp server binary: {e}"),
                section: "binary".to_string(),
                fix: None,
            });
        }
    }

    findings
}

fn check_gguf_path(gguf_path: &std::path::Path) -> Vec<DoctorFinding> {
    let mut findings = Vec::new();

    if gguf_path.as_os_str().is_empty() {
        findings.push(DoctorFinding {
            finding_type: DoctorFindingType::LlamaCpp,
            severity: DoctorSeverity::Issue,
            message: "Default model (GGUF) path is not configured".to_string(),
            section: "model".to_string(),
            fix: None,
        });
        return findings;
    }

    if !gguf_path.exists() {
        findings.push(DoctorFinding {
            finding_type: DoctorFindingType::LlamaCpp,
            severity: DoctorSeverity::Issue,
            message: format!("Default model file not found at {}", gguf_path.display()),
            section: "model".to_string(),
            fix: None,
        });
        return findings;
    }

    if !matches!(gguf_path.extension(), Some(ext) if ext == "gguf") {
        findings.push(DoctorFinding {
            finding_type: DoctorFindingType::LlamaCpp,
            severity: DoctorSeverity::Warning,
            message: "Default model file does not have .gguf extension".to_string(),
            section: "model".to_string(),
            fix: None,
        });
    } else {
        findings.push(DoctorFinding {
            finding_type: DoctorFindingType::LlamaCpp,
            severity: DoctorSeverity::Ok,
            message: format!(
                "Default model file found at {}",
                gguf_path.file_name().unwrap_or_default().to_string_lossy()
            ),
            section: "model".to_string(),
            fix: None,
        });
    }

    findings
}

fn check_port_available(port: u16) -> Vec<DoctorFinding> {
    let mut findings: Vec<DoctorFinding> = Vec::new();

    let is_in_use = TcpStream::connect(("127.0.0.1", port))
        .map(|stream| {
            stream
                .set_read_timeout(Some(Duration::from_millis(100)))
                .is_ok()
        })
        .unwrap_or(false);

    if is_in_use {
        findings.push(DoctorFinding {
            finding_type: DoctorFindingType::LlamaCpp,
            severity: DoctorSeverity::Warning,
            message: format!(
                "Port {} is already in use; a new server may fail to bind",
                port
            ),
            section: "network".to_string(),
            fix: None,
        });
    } else {
        findings.push(DoctorFinding {
            finding_type: DoctorFindingType::LlamaCpp,
            severity: DoctorSeverity::Ok,
            message: format!("Port {} is available for binding", port),
            section: "network".to_string(),
            fix: None,
        });
    }

    findings
}

#[derive(Debug, serde::Deserialize)]
struct ApplyFixRequest {
    fix: FixAction,
}

fn api_apply_fix(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::reply::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "diagnostics" / "apply-fix")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<ApplyFixRequest>())
        .and(with_app_config(app_config))
        .and_then(
            move |auth: Option<String>, req: ApplyFixRequest, cfg: Arc<AppConfig>| {
                let state = state.clone();
                async move {
                    if !check_api_token(&auth, &cfg) {
                        return Ok(unauthorized_api_token());
                    }
                    let fix_action = req.fix;

                    let active_session_id = state.active_session_id.lock().unwrap().clone();
                    if active_session_id.is_empty() {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::with_status(
                                warp::reply::json(&serde_json::json!({
                                    "ok": false,
                                    "error": "No active session"
                                })),
                                warp::http::StatusCode::BAD_REQUEST,
                            ),
                        ));
                    }

                    let preset_id = {
                        let sessions = state.sessions.lock().unwrap();
                        sessions
                            .iter()
                            .find(|s| s.id == active_session_id)
                            .and_then(|s| {
                                if s.preset_id.is_empty() {
                                    None
                                } else {
                                    Some(s.preset_id.clone())
                                }
                            })
                    };

                    let Some(preset_id) = preset_id else {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::with_status(
                                warp::reply::json(&serde_json::json!({
                                    "ok": false,
                                    "error": "Active session has no preset ID"
                                })),
                                warp::http::StatusCode::BAD_REQUEST,
                            ),
                        ));
                    };

                    // Apply fix to preset's RapidMlxConfig
                    let mut presets = state.presets.lock().unwrap();
                    let preset = presets.iter_mut().find(|p| p.id == preset_id);
                    let Some(preset) = preset else {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::with_status(
                                warp::reply::json(&serde_json::json!({
                                    "ok": false,
                                    "error": "Preset not found"
                                })),
                                warp::http::StatusCode::NOT_FOUND,
                            ),
                        ));
                    };

                    let config = preset.rapid_mlx.as_mut();
                    let Some(config) = config else {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::with_status(
                                warp::reply::json(&serde_json::json!({
                                    "ok": false,
                                    "error": "Preset does not use Rapid-MLX backend"
                                })),
                                warp::http::StatusCode::BAD_REQUEST,
                            ),
                        ));
                    };

                    match fix_action {
                        FixAction::AddToolCallParser => {
                            config.tool_call_parser = Some("openai".to_string())
                        }
                        FixAction::EnableAutoToolChoice => config.auto_tool_choice = true,
                        FixAction::AddNoThinking => config.no_thinking = true,
                        // Phase 6: these fix actions write raw fields directly, so they must also
                        // force cache_mode to Custom — otherwise a preset left on Auto/Off would
                        // have CacheMode::resolve() silently override the fix at launch time.
                        FixAction::DisablePrefixCache => {
                            config.prefix_cache_enabled = false;
                            config.cache_mode = crate::inference::rapid_mlx::CacheMode::Custom;
                        }
                        FixAction::AdjustPrefixCacheBudget(bytes) => {
                            config.retained_cache_mib = Some((bytes / (1024 * 1024)) as u32);
                            config.cache_mode = crate::inference::rapid_mlx::CacheMode::Custom;
                        }
                        FixAction::DisableMaxCacheBlocks => {
                            config.retained_cache_mib = None;
                            config.cache_mode = crate::inference::rapid_mlx::CacheMode::Custom;
                        }
                        FixAction::SetPrefixCacheBudget(bytes) => {
                            config.retained_cache_mib = Some((bytes / (1024 * 1024)) as u32);
                            config.cache_mode = crate::inference::rapid_mlx::CacheMode::Custom;
                        }
                        FixAction::ReclaimBackendAllocatorCache => {
                            // Reclaim action handled via system_tools endpoint;
                            // this fix_action is diagnostic-only and not applied here.
                        }
                    }

                    let presets_path = state.presets_path.clone();
                    drop(presets);
                    if let Err(e) =
                        crate::presets::save_presets(&presets_path, &state.presets.lock().unwrap())
                    {
                        eprintln!("[warn] Failed to save presets after applying fix: {e}");
                    }

                    Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                        &serde_json::json!({
                            "ok": true,
                            "applied": req.fix
                        }),
                    )))
                }
            },
        )
}

fn api_create_session(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    let app_config = app_config.clone();
    warp::path!("api" / "sessions")
        .and(warp::path::end())
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(warp::body::json())
        .and(with_app_config(app_config))
        .and_then(move |auth: Option<String>, session: app_state::Session, cfg: Arc<AppConfig>| {
            let state = state.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }
                if state.add_session(session) {
                    Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(
                        Box::new(warp::reply::json(&serde_json::json!({"ok": true}))),
                    )
                } else {
                    Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(
                        Box::new(warp::reply::json(
                            &serde_json::json!({"ok": false, "error": "Maximum sessions reached"}),
                        )),
                    )
                }
            }
        })
}

fn api_delete_session(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    static LAST_DELETE_SESSION: AtomicU64 = AtomicU64::new(0);

    let app_config = app_config.clone();
    warp::path!("api" / "sessions" / String)
        .and(warp::path::end())
        .and(warp::delete())
        .and(warp::header::optional::<String>("authorization"))
        .and(with_app_config(app_config))
        .and_then(
            move |session_id: String, auth: Option<String>, cfg: Arc<AppConfig>| {
                let state = state.clone();
                async move {
                    if !check_db_admin_token(&auth, &cfg) {
                        return Ok(unauthorized_db_admin_token());
                    }

                    let now = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    let last = LAST_DELETE_SESSION.load(Ordering::Acquire);
                    if now - last < 5 {
                        let remaining = 5 - (now - last);
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::with_status(
                                warp::reply::json(&serde_json::json!({
                                    "ok": false,
                                    "error": "too soon; please wait",
                                    "seconds_remaining": remaining
                                })),
                                warp::http::StatusCode::TOO_MANY_REQUESTS,
                            ),
                        ));
                    }
                    LAST_DELETE_SESSION.store(now, Ordering::Release);

                    if state.remove_session(&session_id) {
                        Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({"ok": true})),
                        ))
                    } else {
                        Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(
                                &serde_json::json!({"ok": false, "error": "Session not found"}),
                            ),
                        ))
                    }
                }
            },
        )
}

fn api_get_active_session(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    let app_config = app_config.clone();
    warp::path!("api" / "sessions" / "active")
        .and(warp::path::end())
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and(with_app_config(app_config))
        .and_then(move |auth: Option<String>, cfg: Arc<AppConfig>| {
            let state = state.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }
                let session_id = state.active_session_id.lock().unwrap().clone();
                let sessions = state.sessions.lock().unwrap();
                let session = sessions.iter().find(|s| s.id == session_id).cloned();
                drop(sessions);

                match session {
                    Some(s) => {
                        let mode_str = match s.mode {
                            SessionMode::Spawn { port, .. } => {
                                format!("Spawn:{}", port)
                            }
                            SessionMode::Attach { endpoint, .. } => {
                                format!("Attach:{}", endpoint)
                            }
                        };
                        Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({
                                "id": s.id,
                                "name": s.name,
                                "mode": mode_str,
                                "status": s.status,
                                "last_active": s.last_active,
                                "preset_id": s.preset_id,
                                "backend": s.backend,
                                "model_identity": s.model_identity
                            })),
                        ))
                    }
                    None => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({"error": "No active session"})),
                    )),
                }
            }
        })
}

fn api_get_active_session_readiness(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "sessions" / "active" / "readiness")
        .and(warp::path::end())
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and(with_app_config(app_config))
        .and_then(move |auth: Option<String>, cfg: Arc<AppConfig>| {
            let state = state.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        unauthorized_api_token(),
                    ));
                }

                let Some(session) = state.get_active_session() else {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(
                            &serde_json::json!({"ok": false, "ready": false, "error": "No active session"}),
                        ),
                    ));
                };

                let (endpoint, api_key) = match session.mode {
                    SessionMode::Spawn {
                        port,
                        bind_host,
                        api_key,
                    } => {
                        let host = super::upstream::local_connect_host(bind_host.as_deref());
                        (format!("http://{host}:{port}"), api_key)
                    }
                    SessionMode::Attach { endpoint, api_key } => (endpoint, api_key),
                };

                let client = reqwest::Client::builder()
                    .timeout(std::time::Duration::from_secs(2))
                    .build()
                    .map_err(|e| warp::reject::custom(ApiError::internal(e.to_string())))?;

                let with_auth = |mut req: reqwest::RequestBuilder| {
                    if let Some(key) = &api_key {
                        req = req.header("Authorization", format!("Bearer {}", key));
                    }
                    req
                };

                let ready = match session.backend {
                    crate::inference::InferenceBackend::RapidMlx => {
                        with_auth(client.get(format!("{endpoint}/health/ready")))
                            .send()
                            .await
                            .is_ok_and(|response| response.status().is_success())
                    }
                    crate::inference::InferenceBackend::LlamaCpp => {
                        let root_ok = with_auth(client.get(&endpoint)).send().await.is_ok();
                        let health_ok = with_auth(client.get(format!("{endpoint}/health")))
                            .send()
                            .await
                            .is_ok();
                        root_ok || health_ok
                    }
                };

                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                    &serde_json::json!({
                        "ok": true,
                        "ready": ready,
                        "endpoint": endpoint,
                        "status": session.status,
                        "backend": session.backend,
                        "model_identity": session.model_identity,
                    }),
                )))
            }
        })
}

fn api_get_capabilities(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "capabilities")
        .and(warp::path::end())
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and(with_app_config(app_config))
        .and_then(move |auth: Option<String>, cfg: Arc<AppConfig>| {
            let state = state.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(
                        unauthorized_api_token(),
                    );
                }
                let capabilities = state.calculate_capabilities();
                let endpoint_kind = state.current_endpoint_kind();
                let session_kind = state.current_session_kind();
                let tray_mode = state.tray_mode.lock().unwrap().clone();
                let active_session = state.get_active_session();
                let backend = active_session.as_ref().map(|session| session.backend);
                let model_identity = active_session
                    .as_ref()
                    .and_then(|session| session.model_identity.clone());
                let inference_features = match active_session.as_ref() {
                    Some(session) if matches!(session.mode, SessionMode::Spawn { .. }) => state
                        .backend
                        .lock()
                        .ok()
                        .and_then(|backend| match (session.backend, backend.as_ref()) {
                            (
                                crate::inference::InferenceBackend::LlamaCpp,
                                Some(crate::inference::backend::BackendAdapter::LlamaCpp(adapter)),
                            ) => Some(adapter.capabilities().clone()),
                            (
                                crate::inference::InferenceBackend::RapidMlx,
                                Some(crate::inference::backend::BackendAdapter::RapidMlx(adapter)),
                            ) => Some(adapter.capabilities().clone()),
                            _ => None,
                        })
                        .unwrap_or_default(),
                    _ => crate::inference::capabilities::CapabilitySet::default(),
                };

                let (system_reason, gpu_reason, cpu_temp_reason) =
                    state.calculate_availability_reasons();

                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                    &serde_json::json!({
                        "capabilities": capabilities,
                        "endpoint_kind": endpoint_kind,
                        "session_kind": session_kind,
                        "inference_backend": backend,
                        "model_identity": model_identity,
                        "inference_features": inference_features,
                        "tray_mode": tray_mode,
                        "availability": {
                            "system": system_reason,
                            "gpu": gpu_reason,
                            "cpu_temp": cpu_temp_reason
                        }
                    }),
                )))
            }
        })
}

fn api_set_active_session(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    let app_config = app_config.clone();
    warp::path!("api" / "sessions" / "active")
        .and(warp::path::end())
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(warp::body::json())
        .and(with_app_config(app_config))
        .and_then(move |auth: Option<String>, payload: serde_json::Value, cfg: Arc<AppConfig>| {
            let state = state.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }
                let session_id = match payload.get("id") {
                    Some(v) => v.as_str().unwrap_or("").to_string(),
                    None => {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                            &serde_json::json!({"ok": false, "error": "Missing session id"}),
                        )));
                    }
                };
                if state.set_active_session(&session_id) {
                    Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(
                        Box::new(warp::reply::json(&serde_json::json!({"ok": true}))),
                    )
                } else {
                    Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(
                        Box::new(warp::reply::json(
                            &serde_json::json!({"ok": false, "error": "Session not found"}),
                        )),
                    )
                }
            }
        })
}

fn api_spawn_session_with_preset(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    static LAST_SPAWN_SESSION: AtomicU64 = AtomicU64::new(0);

    let app_config_inner = app_config.clone();
    warp::path!("api" / "sessions" / "spawn")
        .and(warp::path::end())
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(warp::body::json())
        .and(with_app_config(app_config))
        .and_then(move |auth: Option<String>, payload: serde_json::Value, cfg: Arc<AppConfig>| {
            let state = state.clone();
            let app_config = app_config_inner.clone();
            async move {
                if !check_db_admin_token(&auth, &cfg) {
                    return Ok(unauthorized_db_admin_token());
                }

                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let last = LAST_SPAWN_SESSION.load(Ordering::Acquire);
                if now - last < 15 {
                    let remaining = 15 - (now - last);
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({
                                "ok": false,
                                "error": "too soon; please wait",
                                "seconds_remaining": remaining
                            })),
                            warp::http::StatusCode::TOO_MANY_REQUESTS,
                        ),
                    ));
                }
                LAST_SPAWN_SESSION.store(now, Ordering::Release);

                let requested_port = payload
                    .get("port")
                    .and_then(|value| value.as_u64())
                    .and_then(|port| u16::try_from(port).ok());
                let port = requested_port.unwrap_or(8001);
                let name: String = match payload.get("name") {
                    Some(v) => {
                        if let Some(s) = v.as_str() {
                            s.to_string()
                        } else {
                            format!("Session on port {}", port)
                        }
                    }
                    None => format!("Session on port {}", port),
                };

                if let Some(restored_id) = payload.get("session_id").and_then(|v| v.as_str()) {
                    let session = state
                        .get_sessions()
                        .into_iter()
                        .find(|session| session.id == restored_id);
                    let Some(session) = session else {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::with_status(
                                warp::reply::json(&serde_json::json!({"ok": false, "error": "Restored session not found"})),
                                warp::http::StatusCode::BAD_REQUEST,
                            ),
                        ));
                    };
                    let presets = state.presets.lock().unwrap().clone();
                    let transient_api_key = payload.get("api_key").and_then(|v| v.as_str());
                    let request = match request_from_session(&session, &presets, transient_api_key) {
                        Ok(request) => request,
                        Err(error) => {
                            return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                warp::reply::with_status(
                                    warp::reply::json(&serde_json::json!({"ok": false, "error": error.to_string()})),
                                    warp::http::StatusCode::BAD_REQUEST,
                                ),
                            ));
                        }
                    };
                    let response_backend = request.backend();
                    let response_port = request.port();
                    let previous_active_id = state.active_session_id.lock().unwrap().clone();
                    state.set_active_session(restored_id);
                    return match launch_local(Arc::new(state.clone()), request, &app_config).await {
                        Ok(()) => {
                            state.update_session_status(restored_id, SessionStatus::Running);
                            Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                warp::reply::json(&serde_json::json!({"ok": true, "session_id": restored_id, "backend": response_backend, "port": response_port})),
                            ))
                        }
                        Err(error) => {
                            restore_active_session_after_failed_launch(
                                &state,
                                restored_id,
                                &previous_active_id,
                            );
                            Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                warp::reply::with_status(
                                    warp::reply::json(&serde_json::json!({"ok": false, "error": error.to_string()})),
                                    warp::http::StatusCode::BAD_REQUEST,
                                ),
                            ))
                        }
                    };
                }

                let Some(preset_id) = payload
                    .get("preset_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                else {
                    let request = match request_from_api_payload(&payload) {
                        Ok(request) => request,
                        Err(e) => {
                            return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                warp::reply::with_status(
                                    warp::reply::json(&serde_json::json!({"ok": false, "error": e.to_string()})),
                                    warp::http::StatusCode::BAD_REQUEST,
                                ),
                            ));
                        }
                    };

                    let session_name = if name != format!("Session on port {}", port) {
                        name.clone()
                    } else if !request.model_identity().is_empty() {
                        let identity = request.model_identity();
                        let filename = std::path::Path::new(&identity)
                            .file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or(&identity);
                        format!("Local: {}", filename)
                    } else {
                        name.clone()
                    };

                    let session_id = app_state::generate_session_id();
                    let mut session = app_state::Session::new_spawn_with_backend(
                        session_id.clone(),
                        session_name,
                        request.port(),
                        String::new(),
                        request.bind_host(),
                        request.api_key(),
                        request.backend(),
                        Some(request.model_identity()),
                    );
                    session.launch = Some(request.for_persistence());

                    if !state.add_session(session) {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(
                                &serde_json::json!({"ok": false, "error": "Failed to create session"}),
                            ),
                        ));
                    }

                    state.set_active_session(&session_id);
                    let response_backend = request.backend();
                    let response_port = request.port();

                    match launch_local(Arc::new(state.clone()), request, &app_config).await {
                        Ok(()) => {
                            state.update_session_status(&session_id, SessionStatus::Running);
                            return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                warp::reply::json(
                                    &serde_json::json!({"ok": true, "session_id": session_id, "backend": response_backend, "port": response_port}),
                                ),
                            ));
                        }
                        Err(e) => {
                            state.remove_session(&session_id);
                            return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                warp::reply::with_status(
                                    warp::reply::json(&serde_json::json!({"ok": false, "error": e.to_string()})),
                                    warp::http::StatusCode::BAD_REQUEST,
                                ),
                            ));
                        }
                    }
                };

                let preset = {
                    let presets = state.presets.lock().unwrap();
                    match presets.iter().find(|p| p.id == preset_id).cloned() {
                        Some(p) => p,
                        None => {
                            return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                                &serde_json::json!({"ok": false, "error": "Preset not found"}),
                            )));
                        }
                    }
                };

                let requested_selection = match payload.get("selection") {
                    None => None,
                    Some(value) => match serde_json::from_value::<PresetBundleSelection>(value.clone()) {
                        Ok(selection) => Some(selection),
                        Err(error) => {
                            return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                warp::reply::with_status(
                                    warp::reply::json(&serde_json::json!({"ok": false, "code": "invalid_selection", "error": error.to_string()})),
                                    warp::http::StatusCode::BAD_REQUEST,
                                ),
                            ));
                        }
                    },
                };
                let expected_revision = payload
                    .get("expected_revision")
                    .and_then(serde_json::Value::as_u64);
                if requested_selection.is_some() && expected_revision.is_none() {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({"ok": false, "code": "expected_revision_required", "error": "expected_revision is required with a selection"})),
                            warp::http::StatusCode::BAD_REQUEST,
                        ),
                    ));
                }
                if let Some(expected) = expected_revision
                    && expected != preset.revision
                {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({"ok": false, "code": "revision_conflict", "error": "preset revision is stale", "revision": preset.revision})),
                            warp::http::StatusCode::CONFLICT,
                        ),
                    ));
                }

                let (launch_preset, resolved_selection_hash) = if preset.bundle.is_some() {
                    let capabilities = super::preset_bundles::current_capabilities(&app_config).await;

                    // current_capabilities().await is the only suspension point between the
                    // initial preset snapshot above and its use below; re-fetch and re-validate
                    // here so a concurrent mutation during that await can't let a stale preset
                    // reach resolve_preset/launch.
                    let preset = {
                        let presets = state.presets.lock().unwrap();
                        match presets.iter().find(|p| p.id == preset_id).cloned() {
                            Some(p) => p,
                            None => {
                                return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                                    &serde_json::json!({"ok": false, "error": "Preset not found"}),
                                )));
                            }
                        }
                    };
                    if let Some(expected) = expected_revision
                        && expected != preset.revision
                    {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::with_status(
                                warp::reply::json(&serde_json::json!({"ok": false, "code": "revision_conflict", "error": "preset revision is stale", "revision": preset.revision})),
                                warp::http::StatusCode::CONFLICT,
                            ),
                        ));
                    }

                    let resolved = match crate::presets::resolver::resolve_preset(
                        &preset,
                        requested_selection.as_ref(),
                        &capabilities,
                    ) {
                        Ok(resolved) => resolved,
                        Err(issues) => {
                            return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                warp::reply::with_status(
                                    warp::reply::json(&serde_json::json!({"ok": false, "code": "selection_invalid", "error": issues})),
                                    warp::http::StatusCode::BAD_REQUEST,
                                ),
                            ));
                        }
                    };
                    if let Some(expected_hash) = payload.get("expected_resolved_config_hash").and_then(serde_json::Value::as_str)
                        && expected_hash != resolved.config_hash
                    {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::with_status(
                                warp::reply::json(&serde_json::json!({"ok": false, "code": "preview_stale", "error": "resolved configuration changed", "resolved_config_hash": resolved.config_hash})),
                                warp::http::StatusCode::PRECONDITION_FAILED,
                            ),
                        ));
                    }
                    (resolved.preset, Some(resolved.selection_hash))
                } else {
                    (preset.clone(), None)
                };

                // Final live-state check immediately before the resolved config is
                // frozen into a session/launch request: closes the window even if a
                // future refactor adds an await between here and the checks above.
                if let Some(expected) = expected_revision {
                    let live_revision = {
                        let presets = state.presets.lock().unwrap();
                        presets.iter().find(|p| p.id == preset_id).map(|p| p.revision)
                    };
                    if live_revision != Some(expected) {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::with_status(
                                warp::reply::json(&serde_json::json!({"ok": false, "code": "revision_conflict", "error": "preset revision is stale", "revision": live_revision})),
                                warp::http::StatusCode::CONFLICT,
                            ),
                        ));
                    }
                }

                let request = match request_from_preset(&launch_preset, requested_port) {
                    Ok(request) => request,
                    Err(error) => {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::with_status(
                                warp::reply::json(&serde_json::json!({"ok": false, "error": error.to_string()})),
                                warp::http::StatusCode::BAD_REQUEST,
                            ),
                        ));
                    }
                };
                let session_id = app_state::generate_session_id();
                let mut session = app_state::Session::new_spawn_with_backend(
                    session_id.clone(),
                    name.clone(),
                    request.port(),
                    preset_id,
                    request.bind_host(),
                    request.api_key(),
                    request.backend(),
                    Some(request.model_identity()),
                );
                session.selection_hash = resolved_selection_hash;
                session.launch = Some(request.for_persistence());

                if !state.add_session(session) {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                        &serde_json::json!({"ok": false, "error": "Failed to create session"}),
                    )));
                }

                state.set_active_session(&session_id);
                let response_backend = request.backend();
                let response_port = request.port();

                match launch_local_with_resolved_preset(
                    Arc::new(state.clone()),
                    request,
                    &app_config,
                    Some(launch_preset),
                )
                .await
                {
                    Ok(()) => {
                        state.update_session_status(&session_id, SessionStatus::Running);
                        Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                            &serde_json::json!({"ok": true, "session_id": session_id, "backend": response_backend, "port": response_port}),
                        )))
                    }
                    Err(e) => {
                        state.remove_session(&session_id);
                        Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::with_status(
                                warp::reply::json(&serde_json::json!({"ok": false, "error": e.to_string()})),
                                warp::http::StatusCode::BAD_REQUEST,
                            ),
                        ))
                    }
                }
            }
        })
}

fn restore_active_session_after_failed_launch(
    state: &AppState,
    failed_session_id: &str,
    previous_active_id: &str,
) {
    let failed_session_is_still_active =
        state.active_session_id.lock().unwrap().as_str() == failed_session_id;
    if failed_session_is_still_active {
        state.set_active_session(previous_active_id);
    }
}

fn is_private_or_loopback_ip(ip: std::net::IpAddr) -> bool {
    ip.is_loopback()
        || matches!(ip, std::net::IpAddr::V4(v4)
            if v4.octets()[0] == 10
                || (v4.octets()[0] == 172 && (16..=31).contains(&v4.octets()[1]))
                || (v4.octets()[0] == 192 && v4.octets()[1] == 168))
}

async fn discover_rapid_model_identity(
    client: &reqwest::Client,
    endpoint: &str,
    api_key: Option<&str>,
) -> Result<Option<String>, ()> {
    use futures_util::StreamExt;

    const MAX_MODELS_RESPONSE_BYTES: usize = 64 * 1024;
    let mut request = client.get(format!("{endpoint}/v1/models"));
    if let Some(key) = api_key.filter(|key| !key.is_empty()) {
        request = request.bearer_auth(key);
    }
    let response = request.send().await.map_err(|_| ())?;
    if !response.status().is_success() {
        return Err(());
    }
    let mut stream = response.bytes_stream();
    let mut body = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|_| ())?;
        if body.len().saturating_add(chunk.len()) > MAX_MODELS_RESPONSE_BYTES {
            return Err(());
        }
        body.extend_from_slice(&chunk);
    }
    let value = serde_json::from_slice::<serde_json::Value>(&body).map_err(|_| ())?;
    let identity = value
        .get("data")
        .and_then(serde_json::Value::as_array)
        .and_then(|models| models.first())
        .and_then(|model| model.get("id"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    Ok(identity)
}

async fn inference_diagnostics_available(
    client: &reqwest::Client,
    endpoint: &str,
    backend: crate::inference::InferenceBackend,
    api_key: Option<&str>,
) -> bool {
    let diagnostics_path = match backend {
        crate::inference::InferenceBackend::LlamaCpp => "/health",
        crate::inference::InferenceBackend::RapidMlx => "/v1/status",
    };
    let mut request = client.get(format!(
        "{}{diagnostics_path}",
        endpoint.trim_end_matches('/')
    ));
    if let Some(key) = api_key.filter(|key| !key.is_empty()) {
        request = request.bearer_auth(key);
    }
    match backend {
        crate::inference::InferenceBackend::RapidMlx => request
            .send()
            .await
            .is_ok_and(|response| response.status().is_success()),
        crate::inference::InferenceBackend::LlamaCpp => request.send().await.is_ok(),
    }
}

fn api_attach(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    static LAST_ATTACH: AtomicU64 = AtomicU64::new(0);

    warp::path!("api" / "attach")
        .and(warp::path::end())
        .and(warp::post())
        .and(warp::header::headers_cloned())
        .and(warp::body::content_length_limit(16 * 1024))
        .and(warp::body::json())
        .and(with_app_config(app_config))
        .and_then(move |headers: warp::http::HeaderMap,
                      payload: serde_json::Map<String, serde_json::Value>,
                      cfg: Arc<AppConfig>| {
            let state = state.clone();
            async move {
                let bearer_str = headers
                    .get("authorization")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.strip_prefix("Bearer "));
                let has_token = bearer_matches_api_token(bearer_str, &cfg);

                if !has_token {
                    return Ok::<_, warp::Rejection>(warp::reply::with_status(
                        warp::reply::json(&serde_json::json!({ "ok": false, "error": "unauthorized" })),
                        warp::http::StatusCode::UNAUTHORIZED,
                    ));
                }

                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let last = LAST_ATTACH.load(Ordering::Acquire);
                if now - last < 10 {
                    let remaining = 10 - (now - last);
                    return Ok::<_, warp::Rejection>(warp::reply::with_status(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": "too soon; please wait",
                            "seconds_remaining": remaining
                        })),
                        warp::http::StatusCode::TOO_MANY_REQUESTS,
                    ));
                }
                LAST_ATTACH.store(now, Ordering::Release);

                let endpoint: String = match payload.get("endpoint") {
                    Some(v) => {
                        if let Some(s) = v.as_str() {
                            let parsed = url::Url::parse(s).map_err(|_| warp::reject::not_found())?;
                            if !["http", "https"].contains(&parsed.scheme()) {
                                return Ok::<_, warp::Rejection>(warp::reply::with_status(
                                    warp::reply::json(&serde_json::json!({"ok": false, "error": "Endpoint must use http:// or https://"})),
                                    warp::http::StatusCode::OK,
                                ));
                            }
                            if let Some(host) = parsed.host_str()
                                && let Ok(ip) = host.parse::<std::net::IpAddr>()
                                && !is_private_or_loopback_ip(ip)
                            {
                                return Ok::<_, warp::Rejection>(warp::reply::with_status(
                                    warp::reply::json(&serde_json::json!({"ok": false, "error": "Endpoint must be on a private network"})),
                                    warp::http::StatusCode::OK,
                                ));
                            }
                            s.to_string()
                        } else {
                            return Ok::<_, warp::Rejection>(warp::reply::with_status(
                                warp::reply::json(&serde_json::json!({"ok": false, "error": "Invalid endpoint"})),
                                warp::http::StatusCode::OK,
                            ));
                        }
                    }
                    None => {
                        return Ok::<_, warp::Rejection>(warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({"ok": false, "error": "Missing endpoint"})),
                            warp::http::StatusCode::OK,
                        ));
                    }
                };

                let caller_api_key: Option<String> = payload.get("api_key")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string());
                let backend = match payload.get("backend") {
                    Some(value) => match serde_json::from_value::<
                        crate::inference::InferenceBackend,
                    >(value.clone()) {
                        Ok(backend) => backend,
                        Err(_) => {
                            return Ok::<_, warp::Rejection>(warp::reply::with_status(
                                warp::reply::json(&serde_json::json!({
                                    "ok": false,
                                    "error": "backend must be llama_cpp or rapid_mlx"
                                })),
                                warp::http::StatusCode::BAD_REQUEST,
                            ));
                        }
                    },
                    None => crate::inference::InferenceBackend::LlamaCpp,
                };
                let mut model_identity = payload
                    .get("model_identity")
                    .and_then(|value| value.as_str())
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(str::to_string);

                let (api_key, _spawn_match_id) = {
                    let mut effective_key = caller_api_key.clone();
                    let mut match_id = None;

                    if effective_key.is_none()
                        && let Ok(parsed) = url::Url::parse(&endpoint)
                        && let Some(host) = parsed.host_str()
                        && let Some(port_num) = parsed.port()
                        && (host == "127.0.0.1" || host == "localhost") {
                            let sessions = state.sessions.lock().unwrap();
                            if let Some(s) = sessions.iter().find(|sess| {
                                if let SessionMode::Spawn {
                                    port,
                                    api_key: sess_key,
                                    ..
                                } = &sess.mode {
                                    *port == port_num && sess_key.is_some()
                                } else {
                                    false
                                }
                            }) {
                                if let SessionMode::Spawn {
                                    api_key: sess_key,
                                    ..
                                } = &s.mode {
                                    effective_key = sess_key.clone();
                                }
                                match_id = Some(s.id.clone());
                            }
                    }

                    (effective_key, match_id)
                };

                let client = match reqwest::Client::builder()
                    .timeout(std::time::Duration::from_secs(15))
                    .build()
                {
                    Ok(c) => c,
                    Err(e) => {
                        return Ok::<_, warp::Rejection>(warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({
                                "ok": false,
                                "error": format!("Failed to create HTTP client: {}", e)
                            })),
                            warp::http::StatusCode::OK,
                        ));
                    }
                };

                eprintln!("[info] Health-checking inference runtime at {}", endpoint);
                let health_url = match backend {
                    crate::inference::InferenceBackend::LlamaCpp => endpoint.clone(),
                    crate::inference::InferenceBackend::RapidMlx => {
                        format!("{}/health/ready", endpoint.trim_end_matches('/'))
                    }
                };
                let mut health_req = client.get(&health_url);
                if let Some(ref key) = api_key {
                    health_req = health_req.header("Authorization", format!("Bearer {}", key));
                }
                let server_up = match health_req.send().await {
                    Ok(resp) => match backend {
                        crate::inference::InferenceBackend::LlamaCpp => true,
                        crate::inference::InferenceBackend::RapidMlx => {
                            resp.status().is_success()
                        }
                    },
                    Err(e) => {
                        eprintln!("[warn] inference runtime health check failed: {}", e);
                        false
                    }
                };
                if !server_up {
                    return Ok::<_, warp::Rejection>(warp::reply::with_status(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": format!("Cannot reach the selected inference runtime at {}. Is it ready?", endpoint)
                        })),
                        warp::http::StatusCode::OK,
                    ));
                }

                if backend == crate::inference::InferenceBackend::RapidMlx {
                    let discovered_model = match discover_rapid_model_identity(
                        &client,
                        endpoint.trim_end_matches('/'),
                        api_key.as_deref(),
                    )
                    .await
                    {
                        Ok(identity) => identity,
                        Err(()) => {
                            return Ok::<_, warp::Rejection>(warp::reply::with_status(
                                warp::reply::json(&serde_json::json!({
                                    "ok": false,
                                    "error": "Rapid-MLX model discovery failed; verify the endpoint and API key"
                                })),
                                warp::http::StatusCode::BAD_REQUEST,
                            ));
                        }
                    };
                    if model_identity.is_none() {
                        model_identity = discovered_model;
                    }
                    if model_identity.is_none() {
                        return Ok::<_, warp::Rejection>(warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({
                                "ok": false,
                                "error": "Rapid-MLX model identity could not be discovered; enter its served model name"
                            })),
                            warp::http::StatusCode::BAD_REQUEST,
                        ));
                    }
                }

                let metrics_available = inference_diagnostics_available(
                    &client,
                    &endpoint,
                    backend,
                    api_key.as_deref(),
                )
                .await;

                let existing_session_id = {
                    let sessions = state.sessions.lock().unwrap();
                    let mut id = sessions.iter().find(|s| {
                        if let SessionMode::Attach { endpoint: ep, .. } = &s.mode {
                            *ep == endpoint && s.backend == backend
                        } else {
                            false
                        }
                    }).map(|s| s.id.clone());

                    if id.is_none()
                        && let Ok(parsed) = url::Url::parse(&endpoint)
                        && let Some(host) = parsed.host_str()
                        && let Some(port_num) = parsed.port()
                        && (host == "127.0.0.1" || host == "localhost") {
                            id = sessions.iter().find(|s| {
                                if let SessionMode::Spawn {
                                    port,
                                    ..
                                } = &s.mode {
                                    *port == port_num && s.backend == backend
                                } else {
                                    false
                                }
                            }).map(|s| s.id.clone());
                    }

                    id
                };

                let session_id = if let Some(id) = existing_session_id {
                    eprintln!("[info] Reusing existing session for {}", endpoint);
                    id
                } else {
                    let session_id = crate::state::generate_session_id();
                    let mut session = crate::state::Session::new_attach(
                        session_id.clone(),
                        format!("Attached: {}", endpoint),
                        endpoint,
                        api_key.clone(),
                    );
                    session.backend = backend;
                    session.model_identity = model_identity.clone();
                    if !state.add_session(session) {
                        return Ok::<_, warp::Rejection>(warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({"ok": false, "error": "Maximum sessions reached"})),
                            warp::http::StatusCode::OK,
                        ));
                    }
                    session_id
                };

                {
                    let mut sessions = state.sessions.lock().unwrap();
                    if let Some(s) = sessions.iter_mut().find(|s| s.id == session_id) {
                        s.backend = backend;
                        s.model_identity = model_identity.clone();
                        s.launch_requires_api_key = api_key.is_some();
                        if let SessionMode::Attach {
                            api_key: session_key,
                            ..
                        } = &mut s.mode
                        {
                            *session_key = api_key.clone();
                        }
                        s.last_connected_at = now;
                        s.connect_count += 1;
                        s.last_error = None;
                    }
                }

                state.set_active_session(&session_id);
                state.llama_poll_notify.notify_waiters();
                Ok::<_, warp::Rejection>(warp::reply::with_status(
                    warp::reply::json(&serde_json::json!({
                        "ok": true,
                        "backend": backend,
                        "model_identity": model_identity,
                        "warning": if !metrics_available {
                            Some("The inference runtime is ready, but its diagnostics endpoint is unavailable.")
                        } else {
                            None
                        }
                    })),
                    warp::http::StatusCode::OK,
                ))
            }
        })
}

fn api_detach(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    let app_config = app_config.clone();
    warp::path!("api" / "detach")
        .and(warp::path::end())
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(with_app_config(app_config))
        .and_then(move |auth: Option<String>, cfg: Arc<AppConfig>| {
            let state = state.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }
                let active_id = state.active_session_id.lock().unwrap().clone();
                if active_id.is_empty() {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                        &serde_json::json!({"ok": false, "error": "No active session to detach from"}),
                    )));
                }

                let sessions = state.sessions.lock().unwrap();
                let session = sessions.iter().find(|s| s.id == active_id);

                let is_attach = session.map(|s| matches!(s.mode, SessionMode::Attach { .. }));

                if !is_attach.unwrap_or(false) {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                        &serde_json::json!({"ok": false, "error": "Active session is not an attach session"}),
                    )));
                }

                {
                    let mut sessions = state.sessions.lock().unwrap();
                    if let Some(s) = sessions.iter_mut().find(|s| s.id == active_id) {
                        s.last_active = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs();
                    }
                }

                drop(sessions);
                state.set_active_session("");
                state.llama_poll_notify.notify_waiters();

                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(&serde_json::json!({"ok": true}))))
            }
        })
}

fn api_kill_server(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    let app_config = app_config.clone();

    warp::path!("api" / "kill-server")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(warp::body::json::<serde_json::Value>())
        .and(with_app_config(app_config))
        .and_then(move |auth: Option<String>, body: serde_json::Value, cfg: Arc<AppConfig>| {
            let state = state.clone();
            async move {
                let bearer = auth.and_then(|v| v.strip_prefix("Bearer ").map(str::to_string));
                let has_admin_token =
                    bearer_matches_db_admin_token(bearer.as_deref(), &cfg);

                if !has_admin_token {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({ "error": "unauthorized; db-admin-token required" })),
                            warp::http::StatusCode::UNAUTHORIZED,
                        ),
                    ));
                }

                let confirm = body.get("confirm")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if confirm != "kill" {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({ "error": "missing confirmation; send { \"confirm\": \"kill\" }" })),
                            warp::http::StatusCode::BAD_REQUEST,
                        ),
                    ));
                }

                state.push_log("[monitor] kill-server: kill-server requested (best-effort)".into());
                let had_managed_process = state.server_child.lock().await.is_some()
                    || state.supervisor.lock().await.is_some();
                match stop_server(&state).await {
                    Ok(()) if had_managed_process => {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({ "ok": true })),
                        ));
                    }
                    Ok(()) => {
                        state.push_log(
                            "[monitor] kill-server: no supervised process; trying legacy process cleanup"
                                .into(),
                        );
                    }
                    Err(e) => {
                        state.push_log(format!("[monitor] stop_server fallback: {}", e));
                    }
                }

                #[cfg(target_os = "windows")]
                {
                    use std::process::Command;
                    match Command::new("taskkill")
                        .args(["/IM", "llama-server.exe", "/F"])
                        .output()
                    {
                        Ok(output) => {
                            if output.status.success() {
                                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                    warp::reply::json(&serde_json::json!({ "ok": true })),
                                ))
                            } else {
                                let err = String::from_utf8_lossy(&output.stderr);
                                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                    warp::reply::json(&serde_json::json!({ "ok": false, "error": err })),
                                ))
                            }
                        }
                        Err(e) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({ "ok": false, "error": e.to_string() })),
                        )),
                    }
                }
                #[cfg(target_os = "linux")]
                {
                    use std::process::Command;
                    match Command::new("pkill").args(["-f", "llama-server"]).output() {
                        Ok(output) => {
                            if output.status.success() {
                                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                    warp::reply::json(&serde_json::json!({ "ok": true })),
                                ))
                            } else {
                                let err = String::from_utf8_lossy(&output.stderr);
                                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                    warp::reply::json(&serde_json::json!({ "ok": false, "error": err })),
                                ))
                            }
                        }
                        Err(e) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({ "ok": false, "error": e.to_string() })),
                        )),
                    }
                }
                #[cfg(target_os = "macos")]
                {
                    use std::process::Command;
                    match Command::new("pkill").args(["-f", "llama-server"]).output() {
                        Ok(output) => {
                            if output.status.success() {
                                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                    warp::reply::json(&serde_json::json!({ "ok": true })),
                                ))
                            } else {
                                let err = String::from_utf8_lossy(&output.stderr);
                                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                    warp::reply::json(&serde_json::json!({ "ok": false, "error": err })),
                                ))
                            }
                        }
                        Err(e) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({ "ok": false, "error": e.to_string() })),
                        )),
                    }
                }
                #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
                {
                    Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({ "ok": false, "error": "Unsupported platform" })),
                    ))
                }
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_restore_rolls_back_active_session() {
        let state = AppState::default();
        assert!(state.add_session(app_state::Session::new_spawn(
            "previous".into(),
            "Previous".into(),
            8001,
            String::new(),
            None,
            None,
        )));
        assert!(state.add_session(app_state::Session::new_spawn(
            "failed".into(),
            "Failed".into(),
            8002,
            String::new(),
            None,
            None,
        )));
        assert!(state.set_active_session("previous"));
        assert!(state.set_active_session("failed"));

        restore_active_session_after_failed_launch(&state, "failed", "previous");

        assert_eq!(state.active_session_id.lock().unwrap().as_str(), "previous");
    }

    #[test]
    fn failed_restore_does_not_overwrite_newer_active_selection() {
        let state = AppState::default();
        for id in ["previous", "failed", "newer"] {
            assert!(state.add_session(app_state::Session::new_spawn(
                id.into(),
                id.into(),
                8001,
                String::new(),
                None,
                None,
            )));
        }
        assert!(state.set_active_session("newer"));

        restore_active_session_after_failed_launch(&state, "failed", "previous");

        assert_eq!(state.active_session_id.lock().unwrap().as_str(), "newer");
    }

    #[tokio::test]
    async fn rapid_model_discovery_supports_open_and_protected_endpoints() {
        let route = warp::path!("v1" / "models")
            .and(warp::header::optional::<String>("authorization"))
            .map(|authorization: Option<String>| {
                if authorization
                    .as_deref()
                    .is_some_and(|value| value != "Bearer correct-key")
                {
                    warp::reply::with_status(
                        warp::reply::json(&serde_json::json!({"error": "unauthorized"})),
                        warp::http::StatusCode::UNAUTHORIZED,
                    )
                } else {
                    warp::reply::with_status(
                        warp::reply::json(&serde_json::json!({
                            "data": [{"id": "served-rapid-model"}]
                        })),
                        warp::http::StatusCode::OK,
                    )
                }
            });
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        tokio::spawn(warp::serve(route).run(([127, 0, 0, 1], port)));
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        let endpoint = format!("http://127.0.0.1:{port}");
        let client = reqwest::Client::new();

        assert_eq!(
            discover_rapid_model_identity(&client, &endpoint, None)
                .await
                .unwrap()
                .as_deref(),
            Some("served-rapid-model")
        );
        assert_eq!(
            discover_rapid_model_identity(&client, &endpoint, Some("correct-key"))
                .await
                .unwrap()
                .as_deref(),
            Some("served-rapid-model")
        );
        assert!(
            discover_rapid_model_identity(&client, &endpoint, Some("wrong-key"))
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn rapid_diagnostics_require_a_successful_authenticated_status() {
        let route = warp::path!("v1" / "status")
            .and(warp::header::optional::<String>("authorization"))
            .map(|authorization: Option<String>| {
                let status = if authorization.as_deref() == Some("Bearer correct-key") {
                    warp::http::StatusCode::OK
                } else {
                    warp::http::StatusCode::UNAUTHORIZED
                };
                warp::reply::with_status(warp::reply::json(&serde_json::json!({})), status)
            });
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        tokio::spawn(warp::serve(route).run(([127, 0, 0, 1], port)));
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        let endpoint = format!("http://127.0.0.1:{port}");
        let client = reqwest::Client::new();

        assert!(
            inference_diagnostics_available(
                &client,
                &endpoint,
                crate::inference::InferenceBackend::RapidMlx,
                Some("correct-key"),
            )
            .await
        );
        assert!(
            !inference_diagnostics_available(
                &client,
                &endpoint,
                crate::inference::InferenceBackend::RapidMlx,
                Some("wrong-key"),
            )
            .await
        );
    }

    #[test]
    fn attach_ip_policy_uses_the_full_rfc1918_172_range() {
        for ip in [
            "127.0.0.1",
            "10.1.2.3",
            "172.16.0.1",
            "172.31.255.254",
            "192.168.1.2",
        ] {
            assert!(is_private_or_loopback_ip(ip.parse().unwrap()), "{ip}");
        }
        for ip in [
            "172.4.0.1",
            "172.11.0.1",
            "172.15.255.254",
            "172.32.0.1",
            "8.8.8.8",
        ] {
            assert!(!is_private_or_loopback_ip(ip.parse().unwrap()), "{ip}");
        }
    }

    #[test]
    fn doctor_check_gguf_path_nonexistent_is_an_issue() {
        let path = std::path::Path::new("/tmp/llama-monitor-test-doctor/does-not-exist.gguf");
        let findings = check_gguf_path(path);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].severity, DoctorSeverity::Issue);
        assert_eq!(findings[0].finding_type, DoctorFindingType::LlamaCpp);
        assert!(findings[0].fix.is_none());
    }

    #[test]
    fn doctor_check_gguf_path_empty_path_is_an_issue() {
        let findings = check_gguf_path(std::path::Path::new(""));
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].severity, DoctorSeverity::Issue);
    }

    #[test]
    fn doctor_check_gguf_path_existing_gguf_file_is_ok() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, b"fake gguf bytes").unwrap();

        let findings = check_gguf_path(&path);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].severity, DoctorSeverity::Ok);
    }

    #[test]
    fn doctor_check_gguf_path_wrong_extension_is_a_warning() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.bin");
        std::fs::write(&path, b"fake bytes").unwrap();

        let findings = check_gguf_path(&path);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].severity, DoctorSeverity::Warning);
    }

    #[test]
    fn doctor_check_port_available_reports_bound_port_as_in_use() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();

        let findings = check_port_available(port);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].severity, DoctorSeverity::Warning);
        assert!(findings[0].message.contains(&port.to_string()));

        drop(listener);
    }

    #[test]
    fn doctor_check_port_available_reports_free_port_as_ok() {
        // Bind then immediately drop to obtain a port that is very likely free,
        // then verify the check reports it as available.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);

        let findings = check_port_available(port);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].severity, DoctorSeverity::Ok);
    }
}
