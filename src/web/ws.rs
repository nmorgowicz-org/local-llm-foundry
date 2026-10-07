use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use warp::Filter;
use warp::ws::Message;

use crate::state::AppState;
#[cfg(test)]
use crate::state::MetricsCapabilities;

static WS_CONNECTIONS: AtomicUsize = AtomicUsize::new(0);

const WS_PUSH_INTERVAL_DEFAULT_MS: u64 = 1_000; // Strata's dashboard cadence (app.js: setTimeout(poll, 1000))
const WS_PUSH_INTERVAL_MIN_MS: u64 = 200;
const WS_PUSH_INTERVAL_MAX_MS: u64 = 10_000;
const WS_PUSH_INTERVAL_HIDDEN_MS: u64 = 5_000;
const WS_PUSH_INTERVAL_SLEEP_MS: u64 = 10_000; // T-049 / T-053: max interval when asleep
const MAX_WS_CONNECTIONS: usize = 50;

fn clamped_push_interval_ms(settings: &crate::state::UiSettings, sleep_mode: u8) -> u64 {
    let val = settings.ws_push_interval_ms;
    let base = val.clamp(WS_PUSH_INTERVAL_MIN_MS, WS_PUSH_INTERVAL_MAX_MS);

    // 0 = Off (full), 1 = LogsOnly, 2 = Sleep (full pause)
    if sleep_mode == 2 {
        // Full sleep: enforce slow interval from config
        let slow_ms = settings.sleep_mode.sleep_ws_interval_ms;
        base.max(slow_ms)
    } else if sleep_mode == 1 {
        // Logs-only: middle interval
        let logs_ms = settings.sleep_mode.logs_only_ws_interval_ms;
        base.max(logs_ms)
    } else {
        base
    }
}

pub fn ws_route(
    ws_state: AppState,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path("ws")
        .and(warp::ws())
        .map(move |ws: warp::ws::Ws| {
            // T-051: wake-on-activity: new WS connection counts as activity
            let current = WS_CONNECTIONS.fetch_add(1, Ordering::Relaxed);
            if current >= MAX_WS_CONNECTIONS {
                WS_CONNECTIONS.fetch_sub(1, Ordering::Relaxed);
                let reply: Box<dyn warp::reply::Reply> = Box::new(warp::reply::with_status(
                    warp::reply::json(&serde_json::json!({ "error": "too many connections" })),
                    warp::http::StatusCode::TOO_MANY_REQUESTS,
                ));
                return reply;
            }

            // Record activity timestamp (T-051)
            {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                ws_state.last_activity_at.store(now, Ordering::Relaxed);
            }

            let state = ws_state.clone();
            let upgrade = ws.on_upgrade(move |socket| {
                let state = state.clone();
                async move {
                    let (mut ws_tx, mut ws_rx) = socket.split();
                    let client_visible = Arc::new(AtomicBool::new(true));

                    // T-051: On open: wake auto-sleep only. Manual sleep persists across reconnects.
                    let mode_on_open = state.sleep_mode.load(Ordering::Relaxed);
                    let manual_on_open = state.sleep_mode_manual.load(Ordering::Relaxed);
                    // Auto-sleep (mode=2, not manual): wake on WS open.
                    let should_wake = mode_on_open == 2 && !manual_on_open;
                    // Only the wake is worth a line. Every page load, tab, and reconnect opens a
                    // socket, so logging the idle case put `mode=0 manual=false` between every
                    // step of an end-to-end run and told nobody anything.
                    if should_wake {
                        eprintln!(
                            "[sleep] WS open: mode={mode_on_open} manual={manual_on_open} → waking (auto-sleep)"
                        );
                        state.sleep_mode.store(0, Ordering::Relaxed);
                        state.sleep_notify.notify_waiters();
                    }

                    let update_visible = Arc::clone(&client_visible);

                    // Spawn broadcast task using its own clone (avoids move/borrow conflicts)
                    let update_task = {
                        let s = state.clone();
                        tokio::spawn(async move {
                            let mut last_interval_ms = WS_PUSH_INTERVAL_DEFAULT_MS;
                            loop {
                                if s.active_session_id.lock().unwrap().is_empty() {
                                    s.llama_poll_notify.notified().await;
                                    continue;
                                }

                                // Check if the push interval has changed in settings
                                let current_ms = {
                                    let settings = s.ui_settings.lock().unwrap();
                                    let mode = s.sleep_mode.load(Ordering::Relaxed);
                                    clamped_push_interval_ms(&settings, mode)
                                };
                                if current_ms != last_interval_ms {
                                    last_interval_ms = current_ms;
                                }

                                // T-049 / T-053: effective interval depends on visibility + sleep_mode
                                let mode = s.sleep_mode.load(Ordering::Relaxed);
                                let client_vis = update_visible.load(Ordering::Relaxed);
                                let is_any_low_power = mode >= 1;
                                let effective_interval_ms = if mode == 2 {
                                    // Full sleep: very slow
                                    last_interval_ms.max(WS_PUSH_INTERVAL_SLEEP_MS)
                                } else if is_any_low_power || !client_vis {
                                    last_interval_ms.max(WS_PUSH_INTERVAL_HIDDEN_MS)
                                } else {
                                    last_interval_ms
                                };

                                // T-056: if chat streaming is active, don't slow interval even if low-power
                                let streaming_active = {
                                    let llama = s.llama_metrics.lock().unwrap();
                                    llama.generation_tokens_per_sec > 0.0
                                };
                                let final_interval_ms = if streaming_active && is_any_low_power {
                                    last_interval_ms
                                } else {
                                    effective_interval_ms
                                };

                                tokio::time::sleep(Duration::from_millis(final_interval_ms)).await;

                                let running = *s.server_running.lock().unwrap();
                                let local_running = *s.local_server_running.lock().unwrap();
                                let mode = s.sleep_mode.load(Ordering::Relaxed);

                                // T-049: 3-way payload:
                                // mode=0 (off): full payload
                                // mode=1 (logs-only): reduced payload + logs
                                // mode=2 (sleep): minimal heartbeat (no logs)
                                let json = if mode == 2 {
                                    let active_session_id =
                                        s.active_session_id.lock().unwrap().clone();
                                    let active_status = {
                                        let sessions = s.sessions.lock().unwrap();
                                        sessions
                                            .iter()
                                            .find(|ss| ss.id == active_session_id)
                                            .map(|ss| match &ss.status {
                                                crate::state::SessionStatus::Stopped => "stopped",
                                                crate::state::SessionStatus::Running => "running",
                                                crate::state::SessionStatus::Disconnected => "disconnected",
                                                crate::state::SessionStatus::Error(_) => "error",
                                            })
                                            .unwrap_or("stopped")
                                    };

                                     let is_manual = s.sleep_mode_manual.load(Ordering::Relaxed);
                                     serde_json::json!({
                                         "mode": "sleep",
                                         "sleep_mode": true,
                                         "sleep_mode_manual": is_manual,
                                         "server_running": running,
                                         "local_server_running": local_running,
                                         "active_session_id": active_session_id,
                                         "active_session_status": active_status
                                     })
                                 } else if mode == 1 {
                                     // Logs-only mode: reduced payload with logs
                                     let logs: Vec<String> = s
                                         .server_logs
                                         .lock()
                                         .unwrap()
                                         .iter()
                                         .cloned()
                                         .collect();
                                     let active_session_id =
                                         s.active_session_id.lock().unwrap().clone();
                                     let active_status = {
                                         let sessions = s.sessions.lock().unwrap();
                                         sessions
                                             .iter()
                                             .find(|ss| ss.id == active_session_id)
                                             .map(|ss| match &ss.status {
                                                 crate::state::SessionStatus::Stopped => "stopped",
                                                 crate::state::SessionStatus::Running => "running",
                                                 crate::state::SessionStatus::Disconnected => "disconnected",
                                                 crate::state::SessionStatus::Error(_) => "error",
                                             })
                                             .unwrap_or("stopped")
                                     };
                                     let is_manual = s.sleep_mode_manual.load(Ordering::Relaxed);
                                     serde_json::json!({
                                         "mode": "logs-only",
                                         "sleep_mode": true,
                                         "sleep_mode_manual": is_manual,
                                         "logs": logs,
                                         "server_running": running,
                                         "local_server_running": local_running,
                                         "active_session_id": active_session_id,
                                         "active_session_status": active_status
                                     })
                                 } else {
                                     // Full payload (off mode)
                                    let local_metrics_available =
                                        s.active_session_uses_local_metrics();
                                    let host_metrics_available = s.host_metrics_available();
                                    let remote_agent_connected = s.remote_agent_connected();
                                    let remote_agent_health_reachable =
                                        s.remote_agent_health_reachable();
                                    let remote_agent_url =
                                        s.remote_agent_url.lock().unwrap().clone();
                                    let remote_agent_version =
                                        s.remote_agent_version.lock().unwrap().clone();
                                    let remote_agent_update_available =
                                        *s.remote_agent_update_available.lock().unwrap();

                                    let gpu = if host_metrics_available {
                                        s.gpu_metrics.lock().unwrap().clone()
                                    } else {
                                        Default::default()
                                    };
                                    let llama = s.llama_metrics.lock().unwrap().clone();
                                    let inference = s.inference_metrics.lock().unwrap().clone();
                                    let inference_session_id =
                                        s.inference_metrics_session_id.lock().unwrap().clone();
                                    let inference_poll_sequence = s.inference_poll_sequence.load(Ordering::Relaxed);
                                    let system = if host_metrics_available {
                                        Some(s.system_metrics.lock().unwrap().clone())
                                    } else {
                                        None
                                    };
                                    let logs: Vec<String> = s
                                        .server_logs
                                        .lock()
                                        .unwrap()
                                        .iter()
                                        .cloned()
                                        .collect();
                                    let active_session_id =
                                        s.active_session_id.lock().unwrap().clone();
                                    let inference_matches_session =
                                        inference_session_id == active_session_id;
                                    let inference = inference.filter(|_| inference_matches_session);
                                    let inference_poll_failed = inference_matches_session
                                        && s.inference_poll_failed.load(Ordering::Relaxed);
                                    let inference_sampled_at_unix_ms = inference.as_ref().and_then(|sample| {
                                        sample
                                            .sampled_at
                                            .duration_since(std::time::UNIX_EPOCH)
                                            .ok()
                                            .map(|duration| duration.as_millis() as u64)
                                    });

                                    let sessions = s.sessions.lock().unwrap();
                                    let session_mode = sessions
                                        .iter()
                                        .find(|ss| ss.id == active_session_id)
                                        .map(|ss| match &ss.mode {
                                            crate::state::SessionMode::Spawn { .. } => "spawn",
                                            crate::state::SessionMode::Attach { .. } => "attach",
                                        })
                                        .unwrap_or("");
                                    let active_session_endpoint = sessions
                                        .iter()
                                        .find(|ss| ss.id == active_session_id)
                                        .map(|ss| match &ss.mode {
                                            crate::state::SessionMode::Spawn { port, bind_host, .. } => {
                                                let host = crate::web::api::upstream::local_connect_host(bind_host.as_deref());
                                                format!("http://{host}:{port}")
                                            }
                                            crate::state::SessionMode::Attach {
                                                endpoint, ..
                                            } => {
                                                endpoint.clone()
                                            }
                                        })
                                        .unwrap_or_default();
                                    let active_session_endpoint_tag =
                                        crate::inference::llama_cpp::telemetry_endpoint_tag(&active_session_endpoint);
                                    let active_backend = sessions
                                        .iter()
                                        .find(|ss| ss.id == active_session_id)
                                        .map(|ss| ss.backend);
                                    drop(sessions);

                                    let capabilities = s.calculate_capabilities();
                                    let endpoint_kind = s.current_endpoint_kind();
                                    let session_kind = s.current_session_kind();
                                    let (system_reason, gpu_reason, cpu_temp_reason) =
                                        s.calculate_availability_reasons();
                                    let last_spawn_cmd =
                                        s.last_spawn_cmd.lock().unwrap().clone();

                                    let active_session_status = {
                                        let sessions = s.sessions.lock().unwrap();
                                        sessions
                                            .iter()
                                            .find(|ss| ss.id == active_session_id)
                                            .map(|ss| match &ss.status {
                                                crate::state::SessionStatus::Stopped => "stopped",
                                                crate::state::SessionStatus::Running => "running",
                                                crate::state::SessionStatus::Disconnected => "disconnected",
                                                crate::state::SessionStatus::Error(_) => "error",
                                            })
                                            .unwrap_or("stopped")
                                    };

                                    let active_session_error = {
                                        let sessions = s.sessions.lock().unwrap();
                                        sessions
                                            .iter()
                                            .find(|ss| ss.id == active_session_id)
                                            .and_then(|ss| {
                                                if let crate::state::SessionStatus::Error(msg) =
                                                    &ss.status
                                                {
                                                    Some(msg.clone())
                                                } else {
                                                    None
                                                }
                                            })
                                    };

                                    let active_session_preset_id = {
                                        let sessions = s.sessions.lock().unwrap();
                                        sessions
                                            .iter()
                                            .find(|ss| ss.id == active_session_id)
                                            .and_then(|ss| {
                                                if ss.preset_id.is_empty() {
                                                    None
                                                } else {
                                                    Some(ss.preset_id.clone())
                                                }
                                            })
                                    };

                                    let active_session_model_identity = {
                                        let sessions = s.sessions.lock().unwrap();
                                        sessions
                                            .iter()
                                            .find(|ss| ss.id == active_session_id)
                                            .and_then(|ss| ss.model_identity.clone())
                                    };

                                     let is_manual = s.sleep_mode_manual.load(Ordering::Relaxed);
                                     serde_json::json!({
                                         "mode": "off",
                                         "sleep_mode": false,
                                        "sleep_mode_manual": is_manual,
                                        "gpu": gpu,
                                        "llama": llama,
                "backend": active_backend.or_else(|| inference.as_ref().map(|sample| sample.backend)),
                "inference": inference.clone(),
                "inference_metric_dictionary": inference.as_ref().map(|sample| sample.metric_dictionary()),
                "inference_sampled_at_unix_ms": inference_sampled_at_unix_ms,
                                        "inference_poll_sequence": inference_poll_sequence,
                                        "inference_poll_failed": inference_poll_failed,
                                        "system": system,
                                        "logs": logs,
                                        "last_spawn_cmd": last_spawn_cmd,
                                        "server_running": running,
                                        "local_server_running": local_running,
                                        "session_mode": session_mode,
                                        "active_session_status": active_session_status,
                                        "active_session_error": active_session_error,
                                         "active_session_id": active_session_id,
                                         "active_session_endpoint": active_session_endpoint,
                                         "active_session_endpoint_tag": active_session_endpoint_tag,
                                         "active_session_preset_id": active_session_preset_id,
                                         "active_session_model_identity": active_session_model_identity,
                                        "local_metrics_available": local_metrics_available,
                                        "host_metrics_available": host_metrics_available,
                                        "remote_agent_connected": remote_agent_connected,
                                        "remote_agent_health_reachable": remote_agent_health_reachable,
                                        "remote_agent_url": remote_agent_url,
                                        "remote_agent_version": remote_agent_version,
                                        "remote_agent_protocol_version": *s.remote_agent_protocol_version.lock().unwrap(),
                                        "remote_agent_update_available": remote_agent_update_available,
                                        "remote_agent_protocol_too_old": *s.remote_agent_protocol_too_old.lock().unwrap(),
                                        "capabilities": capabilities,
                                        "endpoint_kind": endpoint_kind,
                                        "session_kind": session_kind,
                                        "availability": {
                                            "system": system_reason,
                                            "gpu": gpu_reason,
                                            "cpu_temp": cpu_temp_reason
                                        }
                                    })
                                };

                                if ws_tx
                                    .send(Message::text(json.to_string()))
                                    .await
                                    .is_err()
                                {
                                    break;
                                }
                            }
                        })
                    };

                    // Message receive loop (uses outer state)
                    while let Some(msg) = ws_rx.next().await {
                        let Ok(msg) = msg else { break };
                        if !msg.is_text() {
                            continue;
                        }
                        let Ok(text) = msg.to_str() else { continue };
                        let Ok(value) = serde_json::from_str::<serde_json::Value>(text) else {
                            continue
                        };

                        // T-051: record activity on all messages
                        {
                            let now = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs();
                            state.last_activity_at.store(now, Ordering::Relaxed);
                        }

                        let msg_type = value.get("type").and_then(|v| v.as_str());

                        // T-053: client-visibility with mode (active/idle/sleep)
                        if msg_type == Some("client-visibility") {
                            let visible =
                                value.get("visible").and_then(|v| v.as_bool());
                            let mode = value.get("mode").and_then(|v| v.as_str());

                            if let Some(vis) = visible {
                                client_visible.store(vis, Ordering::Relaxed);
                            }

                            // T-051: wake auto-sleep on active visibility; manual sleep is exempt
                            // (only wakes mode=2 auto-sleep, not manual)
                            if mode == Some("active") || visible == Some(true) {
                                let current_mode = state.sleep_mode.load(Ordering::Relaxed);
                                let manual = state.sleep_mode_manual.load(Ordering::Relaxed);
                                if current_mode >= 2 && !manual {
                                    state.sleep_mode.store(0, Ordering::Relaxed);
                                    state.sleep_notify.notify_waiters();
                                }
                            }
                        }

                        // T-051: explicit wake command from client — clears manual flag too
                        if msg_type == Some("wake") {
                            state.sleep_mode_manual.store(false, Ordering::Relaxed);
                            let current_mode = state.sleep_mode.load(Ordering::Relaxed);
                            if current_mode > 0 {
                                state.sleep_mode.store(0, Ordering::Relaxed);
                                state.sleep_notify.notify_waiters();
                            }
                        }
                    }

                    update_task.abort();
                    WS_CONNECTIONS.fetch_sub(1, Ordering::Relaxed);
                }
            });

            Box::new(upgrade)
        })
}

#[cfg(test)]
fn is_full_capabilities(caps: &MetricsCapabilities, _sleep_mode: bool) -> bool {
    caps.inference
        && caps.system
        && caps.gpu
        && caps.cpu_temperature
        && caps.memory
        && caps.host_metrics
        && caps.tray
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn full_ws_payload_includes_comparable_endpoint_tags() {
        for (endpoint, expected_tag) in [
            ("http://host:8001/", "http://host:8001"),
            (
                "http://user:password@host:8001",
                "sha256:5060e47947dfab9807e4d705486d2967db47b6e3deb62061de46c2e58dc85526",
            ),
            (
                "http://host:8001/proxy/?api_key=secret",
                "sha256:caf0000ec0b93a2616855d1a63581960d2ccda36947d2fb2274f7f95a27b8052",
            ),
            (
                "http://host:8001/proxy/?api_key=secret/",
                "sha256:54f24c7817c6ebaa5e4aaf5e8f16739b783fe8424fd76fff522e1dacf98c607a",
            ),
        ] {
            let state = AppState::default();
            *state.active_session_id.lock().unwrap() = "source-session".into();
            state
                .sessions
                .lock()
                .unwrap()
                .push(crate::state::Session::new_attach(
                    "source-session".into(),
                    "test".into(),
                    endpoint.into(),
                    None,
                ));
            state.ui_settings.lock().unwrap().ws_push_interval_ms = 200;
            {
                let mut metrics = state.llama_metrics.lock().unwrap();
                metrics.telemetry_session_id = Some("source-session".into());
                metrics.telemetry_endpoint = Some(expected_tag.into());
            }
            let mut client = warp::test::ws()
                .path("/ws")
                .handshake(ws_route(state))
                .await
                .unwrap();
            let message = tokio::time::timeout(Duration::from_secs(5), client.recv())
                .await
                .unwrap()
                .unwrap();
            let payload: serde_json::Value =
                serde_json::from_str(message.to_str().unwrap()).unwrap();
            assert_eq!(payload["mode"], "off");
            assert_eq!(payload["active_session_endpoint"], endpoint);
            assert_eq!(payload["active_session_endpoint_tag"], expected_tag);
            assert_eq!(payload["llama"]["telemetry_endpoint"], expected_tag);
            assert_eq!(payload["llama"]["telemetry_session_id"], "source-session");
            let tag = payload["active_session_endpoint_tag"].as_str().unwrap();
            assert!(!tag.contains("secret"));
            assert!(!tag.contains("password"));
            drop(client);
        }
    }

    #[test]
    fn is_full_capabilities_returns_true_when_full() {
        let caps = MetricsCapabilities {
            inference: true,
            system: true,
            gpu: true,
            cpu_temperature: true,
            memory: true,
            host_metrics: true,
            tray: true,
            sensor_bridge_setup_available: true,
        };
        assert!(is_full_capabilities(&caps, false));
    }

    #[test]
    fn is_full_capabilities_returns_false_when_missing_system() {
        let caps = MetricsCapabilities {
            inference: true,
            system: false,
            gpu: true,
            cpu_temperature: true,
            memory: true,
            host_metrics: true,
            tray: true,
            sensor_bridge_setup_available: true,
        };
        assert!(!is_full_capabilities(&caps, false));
    }

    #[test]
    fn ws_connections_increment_decrement() {
        let before = WS_CONNECTIONS.load(Ordering::Relaxed);
        WS_CONNECTIONS.fetch_add(1, Ordering::Relaxed);
        assert_eq!(WS_CONNECTIONS.load(Ordering::Relaxed), before + 1);
        WS_CONNECTIONS.fetch_sub(1, Ordering::Relaxed);
        assert_eq!(WS_CONNECTIONS.load(Ordering::Relaxed), before);
    }
}
