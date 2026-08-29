use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::llama::metrics::SlotSnapshot;
use crate::state::AppState;

fn spawned_base_url(port: u16, bind_host: Option<&str>) -> String {
    let host = crate::web::api::upstream::local_connect_host(bind_host);
    format!("http://{host}:{port}")
}

fn reset_inference_poll_state_if_session_changed(
    state: &AppState,
    active_id: &str,
    session_backend: crate::inference::InferenceBackend,
) {
    let mut current = state.inference_metrics.lock().unwrap();
    let mut sampled_session = state.inference_metrics_session_id.lock().unwrap();
    let session_changed = current
        .as_ref()
        .is_some_and(|snapshot| snapshot.backend != session_backend)
        || *sampled_session != active_id;
    if session_changed {
        *current = None;
        *sampled_session = active_id.to_string();
        state
            .inference_poll_failed
            .store(false, std::sync::atomic::Ordering::Relaxed);
        state
            .inference_poll_failures
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

fn record_rapid_poll_liveness(state: &AppState, succeeded: bool) {
    state
        .inference_poll_failed
        .store(!succeeded, std::sync::atomic::Ordering::Relaxed);
    if succeeded {
        state
            .inference_poll_failures
            .store(0, std::sync::atomic::Ordering::Relaxed);
        *state.server_running.lock().unwrap() = true;
        return;
    }
    let failures = state
        .inference_poll_failures
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        + 1;
    if failures >= 3 {
        *state.server_running.lock().unwrap() = false;
    }
}

pub async fn llama_metrics_poller(state: AppState, poll_interval: u64) {
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .pool_max_idle_per_host(0)
        .pool_idle_timeout(Duration::from_secs(0))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[error] Failed to build HTTP client: {:?}", e);
            return;
        }
    };

    let mut enabled = false;
    let mut rapid_poller: Option<crate::inference::rapid_mlx::poller::RapidMlxPoller> = None;
    let mut llama_previous_counters: Option<crate::inference::llama_cpp::CounterSnapshot> = None;
    let mut llama_previous_counter_session: Option<String> = None;

    loop {
        if !enabled {
            state.llama_poll_notify.notified().await;
            enabled = true;
        }

        let active_id = { state.active_session_id.lock().unwrap().clone() };
        if active_id.is_empty() {
            enabled = false;
            tokio::time::sleep(Duration::from_secs(poll_interval)).await;
            continue;
        }

        // Determine endpoint and optional API key from active session
        let (base, api_key, session_backend) = {
            let session = {
                let sessions = state.sessions.lock().unwrap();
                sessions.iter().find(|s| s.id == active_id).cloned()
            };

            if let Some(sess) = session {
                match sess.mode {
                    crate::state::SessionMode::Spawn {
                        port,
                        bind_host,
                        api_key,
                    } => (
                        spawned_base_url(port, bind_host.as_deref()),
                        api_key,
                        sess.backend,
                    ),
                    crate::state::SessionMode::Attach { endpoint, api_key } => {
                        (endpoint, api_key, sess.backend)
                    }
                }
            } else {
                enabled = false;
                tokio::time::sleep(Duration::from_secs(poll_interval)).await;
                continue;
            }
        };
        reset_inference_poll_state_if_session_changed(&state, &active_id, session_backend);

        // Helper to add auth header if API key is set
        fn with_auth(
            mut req: reqwest::RequestBuilder,
            api_key: &Option<String>,
        ) -> reqwest::RequestBuilder {
            if let Some(key) = api_key {
                req = req.header("Authorization", format!("Bearer {}", key));
            }
            req
        }

        // llama.cpp retains its historical root + /health liveness probes. Rapid-MLX
        // performs its endpoint-specific /health probe inside the normalized poll.
        let server_up = if matches!(
            session_backend,
            crate::inference::InferenceBackend::RapidMlx
        ) {
            true
        } else {
            with_auth(client.get(&base), &api_key).send().await.is_ok()
        };

        let server_reachable = if matches!(
            session_backend,
            crate::inference::InferenceBackend::RapidMlx
        ) {
            true
        } else if server_up {
            // Try /health for detailed status
            match with_auth(client.get(format!("{base}/health")), &api_key)
                .send()
                .await
            {
                Ok(resp) => match resp.text().await {
                    Ok(body) => {
                        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
                            let mut m = state.llama_metrics.lock().unwrap();
                            m.status = json
                                .get("status")
                                .and_then(|v| v.as_str())
                                .unwrap_or("running")
                                .to_string();
                        }
                        true
                    }
                    Err(_) => true, // Server up but /health returned non-JSON
                },
                Err(_) => true, // Server up but /health not available (metrics disabled)
            }
        } else {
            false
        };

        // llama.cpp retains its historical liveness update. Rapid-MLX is updated only
        // after a normalized poll so a failed poll cannot briefly flip it back online.
        if matches!(
            session_backend,
            crate::inference::InferenceBackend::LlamaCpp
        ) {
            let mut running = state.server_running.lock().unwrap();
            if server_reachable != *running {
                *running = server_reachable;
            }
        }

        if !server_reachable {
            // Don't reset metrics when server is temporarily unavailable
            // Just continue with the last known metrics
            tokio::time::sleep(Duration::from_secs(poll_interval)).await;
            continue;
        }

        // Use the backend adapter to poll normalized metrics. Attached Rapid-MLX and
        // llama.cpp sessions construct their poll directly from the resolved endpoint;
        // they do not have a spawned adapter in state.backend (Attach sessions never
        // populate it — see poll_llama_cpp_metrics for the llama.cpp case).
        {
            let port = if let Some(sess) = {
                let sessions = state.sessions.lock().unwrap();
                sessions.iter().find(|s| s.id == active_id).cloned()
            } {
                match sess.mode {
                    crate::state::SessionMode::Spawn { port, .. } => port,
                    crate::state::SessionMode::Attach { endpoint, .. } => endpoint
                        .split(':')
                        .next_back()
                        .and_then(|p| p.parse().ok())
                        .unwrap_or(0),
                }
            } else {
                0
            };

            let snapshot_result = if matches!(
                session_backend,
                crate::inference::InferenceBackend::RapidMlx
            ) {
                let matches = rapid_poller
                    .as_ref()
                    .is_some_and(|poller| poller.matches_target(&base, api_key.as_deref()));
                if !matches {
                    rapid_poller = Some(
                        crate::inference::rapid_mlx::poller::RapidMlxPoller::from_base_url(
                            base.clone(),
                            api_key.as_deref(),
                        ),
                    );
                }
                rapid_poller
                    .as_ref()
                    .expect("poller initialized")
                    .poll()
                    .await
            } else if port != 0 {
                rapid_poller = None;
                crate::inference::llama_cpp::poll_llama_cpp_metrics(
                    &base,
                    api_key.as_deref(),
                    &active_id,
                    &mut llama_previous_counters,
                    &mut llama_previous_counter_session,
                )
                .await
            } else {
                Err(anyhow::anyhow!("active backend adapter unavailable"))
            };

            state
                .inference_poll_sequence
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if matches!(
                session_backend,
                crate::inference::InferenceBackend::RapidMlx
            ) {
                record_rapid_poll_liveness(&state, snapshot_result.is_ok());
            }
            if let Ok(snapshot) = snapshot_result {
                if matches!(
                    session_backend,
                    crate::inference::InferenceBackend::LlamaCpp
                ) {
                    state
                        .inference_poll_failed
                        .store(false, std::sync::atomic::Ordering::Relaxed);
                    state
                        .inference_poll_failures
                        .store(0, std::sync::atomic::Ordering::Relaxed);
                }
                *state.inference_metrics.lock().unwrap() = Some(snapshot.clone());
                *state.inference_metrics_session_id.lock().unwrap() = active_id.clone();
                let now_ms = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64;

                let mut m = state.llama_metrics.lock().unwrap();

                if let Some(prompt_tps) = snapshot.prompt_tokens_per_second {
                    m.prompt_tokens_per_sec = prompt_tps;
                    m.prompt_throughput_active = prompt_tps > 0.0;
                    if prompt_tps > 0.0 {
                        m.last_prompt_tokens_per_sec = prompt_tps;
                        m.last_prompt_throughput_unix_ms = now_ms;
                    }
                }

                if let Some(gen_tps) = snapshot.generation_tokens_per_second {
                    m.generation_tokens_per_sec = gen_tps;
                    m.generation_throughput_active = gen_tps > 0.0;
                    if gen_tps > 0.0 {
                        m.last_generation_tokens_per_sec = gen_tps;
                        m.last_generation_throughput_unix_ms = now_ms;
                    }
                }

                m.throughput_source = "backend_poll".to_string();

                if let Some(prompt_total) = snapshot.prompt_tokens_total {
                    m.prompt_tokens_total = prompt_total;
                }
                if let Some(completion_total) = snapshot.completion_tokens_total {
                    m.predicted_tokens_total = completion_total;
                    m.generation_tokens_total = completion_total;
                }
                if let Some(running) = snapshot.running_requests {
                    m.requests_processing = running as u32;
                }
                if let Some(details) = snapshot.backend_details {
                    if let Some(idle) = details.get("slots_idle").and_then(|v| v.as_u64()) {
                        m.slots_idle = idle as u32;
                    }
                    if let Some(processing) =
                        details.get("slots_processing").and_then(|v| v.as_u64())
                    {
                        m.slots_processing = processing as u32;
                    }
                    if let Some(max) = details.get("kv_cache_max").and_then(|v| v.as_u64()) {
                        m.kv_cache_max = max;
                        m.context_capacity_tokens = max;
                    }
                    if let Some(tokens) = details.get("kv_cache_tokens").and_then(|v| v.as_u64()) {
                        m.kv_cache_tokens = tokens;
                        m.context_live_tokens = tokens;
                    }
                    if let Some(avail) = details
                        .get("kv_cache_tokens_available")
                        .and_then(|v| v.as_bool())
                    {
                        m.kv_cache_tokens_available = avail;
                        m.context_live_tokens_available = avail;
                    }
                    if let Some(source) = details
                        .get("kv_cache_tokens_source")
                        .and_then(|v| v.as_str())
                    {
                        m.kv_cache_tokens_source = source.to_string();
                        m.context_live_tokens_source = source.to_string();
                    }
                    if let Some(active) = details.get("active_task_id").and_then(|v| v.as_u64()) {
                        m.active_task_id = Some(active);
                    }
                    if let Some(last) = details.get("last_task_id").and_then(|v| v.as_u64()) {
                        m.last_task_id = Some(last);
                    }
                    if let Some(tokens_per_decode) =
                        details.get("tokens_per_decode").and_then(|v| v.as_f64())
                    {
                        m.tokens_per_decode = tokens_per_decode;
                    }
                    if let Some(busy_slots_per_decode) = details
                        .get("n_busy_slots_per_decode")
                        .and_then(|v| v.as_f64())
                    {
                        m.n_busy_slots_per_decode = busy_slots_per_decode;
                    }
                    m.speculative_acceptance_rate = details
                        .get("speculative_acceptance_rate")
                        .and_then(|v| v.as_f64());
                    if let Some(slots) = details.get("slots").and_then(|value| {
                        serde_json::from_value::<Vec<SlotSnapshot>>(value.clone()).ok()
                    }) {
                        m.slots = slots;
                    }
                }
                if let Some(model) = snapshot.model {
                    m.model_name = model;
                }
            }
        }

        // Poll /v1/models — get model name and metadata
        if matches!(
            session_backend,
            crate::inference::InferenceBackend::LlamaCpp
        ) && let Ok(resp) = with_auth(client.get(format!("{base}/v1/models")), &api_key)
            .send()
            .await
            && let Ok(body) = resp.text().await
            && let Ok(json) = serde_json::from_str::<serde_json::Value>(&body)
            && let Some(model_name) = json["data"][0]["id"].as_str()
        {
            let mut m = state.llama_metrics.lock().unwrap();
            m.model_name = model_name.to_string();
            if let Some(n_params) = json["data"][0]["meta"]["n_params"].as_u64() {
                m.model_params = Some(n_params);
            }
            if let Some(n_ctx_train) = json["data"][0]["meta"]["n_ctx_train"].as_u64() {
                m.model_ctx_train = Some(n_ctx_train);
            }
        }

        // T-047: slow poll interval when in low-power mode (session is active here)
        let mode = state.sleep_mode.load(std::sync::atomic::Ordering::Relaxed);
        let interval_secs = if mode >= 1 {
            if let Ok(cfg) = state.sleep_mode_config.lock() {
                cfg.sleep_llama_interval_secs.max(1)
            } else {
                poll_interval
            }
        } else {
            poll_interval
        };
        tokio::time::sleep(Duration::from_secs(interval_secs)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        record_rapid_poll_liveness, reset_inference_poll_state_if_session_changed, spawned_base_url,
    };
    use crate::inference::InferenceBackend;
    use crate::state::AppState;
    use std::sync::atomic::Ordering;

    #[test]
    fn spawned_polling_uses_a_connectable_bind_host() {
        assert_eq!(spawned_base_url(8080, None), "http://127.0.0.1:8080");
        assert_eq!(
            spawned_base_url(8080, Some("0.0.0.0")),
            "http://127.0.0.1:8080"
        );
        assert_eq!(spawned_base_url(8080, Some("::1")), "http://[::1]:8080");
        assert_eq!(
            spawned_base_url(8080, Some("192.168.1.5")),
            "http://192.168.1.5:8080"
        );
    }

    #[test]
    fn switching_sessions_resets_telemetry_failure_hysteresis() {
        let state = AppState::default();
        *state.inference_metrics_session_id.lock().unwrap() = "session-a".to_string();
        state.inference_poll_failed.store(true, Ordering::Relaxed);
        state.inference_poll_failures.store(2, Ordering::Relaxed);

        reset_inference_poll_state_if_session_changed(
            &state,
            "session-b",
            InferenceBackend::RapidMlx,
        );

        assert_eq!(
            state.inference_metrics_session_id.lock().unwrap().as_str(),
            "session-b"
        );
        assert!(!state.inference_poll_failed.load(Ordering::Relaxed));
        assert_eq!(state.inference_poll_failures.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn sustained_rapid_poll_failure_does_not_flip_back_to_running() {
        let state = AppState::default();
        *state.server_running.lock().unwrap() = true;

        record_rapid_poll_liveness(&state, false);
        record_rapid_poll_liveness(&state, false);
        assert!(*state.server_running.lock().unwrap());
        record_rapid_poll_liveness(&state, false);
        assert!(!*state.server_running.lock().unwrap());
        record_rapid_poll_liveness(&state, false);
        assert!(!*state.server_running.lock().unwrap());

        record_rapid_poll_liveness(&state, true);
        assert!(*state.server_running.lock().unwrap());
        assert_eq!(state.inference_poll_failures.load(Ordering::Relaxed), 0);
    }
}
