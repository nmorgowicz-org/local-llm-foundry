use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::llama::metrics::SlotSnapshot;
use crate::state::AppState;

fn spawned_base_url(port: u16, bind_host: Option<&str>) -> String {
    let host = crate::web::api::upstream::local_connect_host(bind_host);
    format!("http://{host}:{port}")
}

/// Re-check after await: an old endpoint's response must not populate a new session.
fn with_current_poll_target(
    state: &AppState,
    active_id: &str,
    backend: crate::inference::InferenceBackend,
    base: &str,
    api_key: Option<&str>,
    publish: impl FnOnce(&crate::state::Session),
) -> bool {
    // AppState session operations acquire sessions before active_session_id.
    let sessions = state.sessions.lock().unwrap();
    let current_id = state.active_session_id.lock().unwrap();
    if current_id.as_str() != active_id {
        return false;
    }
    let Some(session) = sessions.iter().find(|s| s.id == active_id) else {
        return false;
    };
    if session.backend != backend {
        return false;
    }
    let (current_base, current_key) = match &session.mode {
        crate::state::SessionMode::Spawn {
            port,
            bind_host,
            api_key,
        } => (
            spawned_base_url(*port, bind_host.as_deref()),
            api_key.as_deref(),
        ),
        crate::state::SessionMode::Attach { endpoint, api_key } => {
            (endpoint.clone(), api_key.as_deref())
        }
    };
    let matches = current_base.trim_end_matches('/') == base.trim_end_matches('/')
        && crate::inference::llama_cpp::same_api_key(current_key, api_key);
    if matches {
        // Callback is synchronous: no guards are held across network I/O or sleep.
        publish(session);
    }
    matches
}

fn active_poll_target_matches(
    state: &AppState,
    active_id: &str,
    backend: crate::inference::InferenceBackend,
    base: &str,
    api_key: Option<&str>,
) -> bool {
    with_current_poll_target(state, active_id, backend, base, api_key, |_| {})
}

fn project_optional_llama_metrics(
    metrics: &mut crate::llama::metrics::LlamaMetrics,
    snapshot: &crate::inference::metrics::InferenceMetricsSnapshot,
) {
    let details = (snapshot.backend == crate::inference::InferenceBackend::LlamaCpp)
        .then_some(snapshot.backend_details.as_ref())
        .flatten();
    let get = |key: &str| details.and_then(|v| v.get(key));
    metrics.telemetry_session_id = get("telemetry_session_id")
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    metrics.telemetry_endpoint = get("telemetry_endpoint")
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    metrics.prompt_tokens_processed_total =
        get("prompt_tokens_processed_total").and_then(|v| v.as_f64());
    metrics.prompt_tokens_cached_total = get("prompt_tokens_cached_total").and_then(|v| v.as_f64());
    metrics.speculative_draft_tokens_total =
        get("speculative_draft_tokens_total").and_then(|v| v.as_u64());
    metrics.speculative_accepted_tokens_total =
        get("speculative_accepted_tokens_total").and_then(|v| v.as_u64());
    metrics.speculative_verification_steps_total =
        get("speculative_verification_steps_total").and_then(|v| v.as_u64());
    metrics.speculative_enabled = get("speculative_enabled").and_then(|v| v.as_bool());
    metrics.runtime_facts =
        get("runtime_facts").and_then(|v| serde_json::from_value(v.clone()).ok());
    metrics.speculative_acceptance_rate = if details.is_some() {
        snapshot.speculative_acceptance_rate
    } else {
        None
    };
    metrics.model_params = metrics.runtime_facts.as_ref().and_then(|v| v.model_params);
    metrics.model_ctx_train = get("model_ctx_train").and_then(|v| v.as_u64());
}

fn clear_optional_llama_metrics(state: &AppState) {
    let mut metrics = state.llama_metrics.lock().unwrap();
    // Optional facts may expire while legacy rates remain. Keep their attribution
    // so a failed poll cannot turn old metrics into an untagged wildcard.
    let session_id = metrics.telemetry_session_id.take();
    let endpoint = metrics.telemetry_endpoint.take();
    project_optional_llama_metrics(
        &mut metrics,
        &crate::inference::metrics::InferenceMetricsSnapshot::empty(
            crate::inference::InferenceBackend::LlamaCpp,
        ),
    );
    metrics.telemetry_session_id = session_id;
    metrics.telemetry_endpoint = endpoint;
    metrics.model_name.clear();
    metrics.slots.clear();
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
        *state.llama_metrics.lock().unwrap() = Default::default();
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum LiveSlotKey {
    Id(u64),
    // Older servers may omit slot ids; conservatively use their array position.
    Position(usize),
}

#[derive(Debug)]
struct LiveSlotProgress {
    identity: (LiveSlotKey, Option<u64>),
    generation: Option<u64>,
    prompt: Option<u64>,
}

#[derive(Debug, Default)]
struct LiveRateWindow {
    baseline: Option<(u64, std::time::Instant)>,
    ema: Option<f64>,
}

impl LiveRateWindow {
    fn rebase(&mut self, tokens: u64, now: std::time::Instant) {
        self.baseline = Some((tokens, now));
        self.ema = None;
    }

    fn report(
        &mut self,
        tokens: u64,
        now: std::time::Instant,
        chunk_size: u64,
        minimum_seconds: f64,
        new_weight: f64,
    ) -> Option<f64> {
        let (base_tokens, base_at) = self.baseline?;
        let chunk = tokens.checked_sub(base_tokens)?;
        let elapsed = now.saturating_duration_since(base_at).as_secs_f64();
        if chunk < chunk_size || elapsed <= minimum_seconds {
            // Neither small polls nor a too-short window consume a chunk.
            return None;
        }
        let rate = chunk as f64 / elapsed;
        let ema = self
            .ema
            .map_or(rate, |prev| prev * (1.0 - new_weight) + rate * new_weight);
        self.ema = Some(ema);
        self.baseline = Some((tokens, now));
        Some(ema)
    }
}

/// Track last-poll per-slot progress separately from quantized report baselines.
/// A regression below the last poll can be hidden by a still-increasing aggregate
/// (or remain above the last report); either must invalidate both rate windows.
#[derive(Debug, Default)]
struct LiveRateTracker {
    slots: Vec<LiveSlotProgress>,
    totals: Option<(u64, u64)>,
    generation: LiveRateWindow,
    prompt: LiveRateWindow,
}

impl LiveRateTracker {
    fn observe(
        &mut self,
        slots: &[SlotSnapshot],
        aggregate_prompt: u64,
        now: std::time::Instant,
    ) -> (Option<f64>, Option<f64>) {
        let mut active: Vec<_> = slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.is_processing)
            .map(|(index, slot)| LiveSlotProgress {
                identity: (
                    slot.id
                        .map_or(LiveSlotKey::Position(index), LiveSlotKey::Id),
                    slot.id_task,
                ),
                generation: slot.output_available.then_some(slot.output_tokens),
                prompt: slot.prompt_tokens_processed,
            })
            .collect();
        active.sort_by_key(|slot| slot.identity);
        if active.is_empty() {
            // Idle slots retain next_token counters on some llama.cpp versions.
            // They are not live work and cannot seed the next request's timer.
            *self = Self::default();
            return (None, None);
        }
        let generation = active.iter().fold(0_u64, |total, slot| {
            total.saturating_add(slot.generation.unwrap_or(0))
        });
        let prompt = if active.iter().all(|slot| slot.prompt.is_some()) {
            active.iter().fold(0_u64, |total, slot| {
                total.saturating_add(slot.prompt.unwrap_or(0))
            })
        } else {
            // Legacy serialized SlotSnapshots have no per-slot prefill field.
            // Their aggregate remains useful, but cannot expose masked regressions.
            aggregate_prompt
        };
        let changed_or_regressed = self.slots.len() != active.len()
            || self.slots.iter().zip(&active).any(|(old, new)| {
                old.identity != new.identity
                    || counter_changed_or_regressed(old.generation, new.generation)
                    || counter_changed_or_regressed(old.prompt, new.prompt)
            })
            || self
                .totals
                .is_none_or(|(old_gen, old_prompt)| generation < old_gen || prompt < old_prompt);
        self.slots = active;
        self.totals = Some((generation, prompt));
        if changed_or_regressed {
            // Joins/leaves and identity changes rebase as one coherent sample,
            // even if another slot's growth made both aggregates increase.
            self.generation.rebase(generation, now);
            self.prompt.rebase(prompt, now);
            return (None, None);
        }
        (
            self.generation.report(generation, now, 16, 0.5, 0.5),
            self.prompt.report(prompt, now, 1024, 1.0, 0.4),
        )
    }
}

fn counter_changed_or_regressed(old: Option<u64>, new: Option<u64>) -> bool {
    old.is_some() != new.is_some() || old.zip(new).is_some_and(|(old, new)| new < old)
}

/// Slot-derived live rates own the display when slots can be decoded. Backend
/// completed-request rates are a fallback, never an overwrite of held live EMA.
/// Both clocks are injected so turnover and reporting tests need no sleeps.
fn project_llama_throughput(
    metrics: &mut crate::llama::metrics::LlamaMetrics,
    snapshot: &crate::inference::metrics::InferenceMetricsSnapshot,
    tracker: &mut LiveRateTracker,
    now: std::time::Instant,
    now_ms: u64,
) -> Option<Vec<SlotSnapshot>> {
    let slots = snapshot
        .backend_details
        .as_ref()
        .and_then(|details| details.get("slots"))
        .and_then(|value| serde_json::from_value::<Vec<SlotSnapshot>>(value.clone()).ok());
    let (new_gen, new_prompt) = if let Some(slots) = slots.as_ref() {
        let aggregate_prompt = snapshot
            .backend_details
            .as_ref()
            .and_then(|details| details.get("slot_prompt_processed"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let reports = tracker.observe(slots, aggregate_prompt, now);
        metrics.generation_tokens_per_sec = tracker.generation.ema.unwrap_or(0.0);
        metrics.prompt_tokens_per_sec = tracker.prompt.ema.unwrap_or(0.0);
        reports
    } else {
        // A telemetry gap may hide an entire request: do not bridge its timer
        // when slots return, even if the next visible task id matches.
        *tracker = LiveRateTracker::default();
        if let Some(rate) = snapshot.generation_tokens_per_second {
            metrics.generation_tokens_per_sec = rate;
        }
        if let Some(rate) = snapshot.prompt_tokens_per_second {
            metrics.prompt_tokens_per_sec = rate;
        }
        (
            snapshot.generation_tokens_per_second,
            snapshot.prompt_tokens_per_second,
        )
    };
    metrics.generation_throughput_active = metrics.generation_tokens_per_sec > 0.0;
    metrics.prompt_throughput_active = metrics.prompt_tokens_per_sec > 0.0;
    // Holding or clearing a live rate does not fabricate a new measurement.
    // Keep the last positive rates/timestamps for the idle UI fallback.
    if let Some(rate) = new_gen.filter(|rate| *rate > 0.0) {
        metrics.last_generation_tokens_per_sec = rate;
        metrics.last_generation_throughput_unix_ms = now_ms;
    }
    if let Some(rate) = new_prompt.filter(|rate| *rate > 0.0) {
        metrics.last_prompt_tokens_per_sec = rate;
        metrics.last_prompt_throughput_unix_ms = now_ms;
    }
    metrics.throughput_source = "backend_poll".to_string();
    slots
}

fn clear_failed_llama_sample(state: &AppState, tracker: &mut LiveRateTracker) {
    // The unseen interval may contain a full request, even with missing task ids.
    *tracker = LiveRateTracker::default();
    clear_optional_llama_metrics(state);
}

/// Publish one finished poll under the active-target guard. Rapid-MLX spawned
/// targets are enriched here (cheap, no IO) because only the guard holds the
/// Session proving the snapshot belongs to the current spawn.
fn publish_poll_result(
    state: &AppState,
    active_id: &str,
    session_backend: crate::inference::InferenceBackend,
    base: &str,
    api_key: Option<&str>,
    snapshot_result: anyhow::Result<crate::inference::metrics::InferenceMetricsSnapshot>,
    llama_live_rates: &mut LiveRateTracker,
) -> bool {
    with_current_poll_target(
        state,
        active_id,
        session_backend,
        base,
        api_key,
        |session| {
            state
                .inference_poll_sequence
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if matches!(
                session_backend,
                crate::inference::InferenceBackend::RapidMlx
            ) {
                record_rapid_poll_liveness(state, snapshot_result.is_ok());
            }
            if let Ok(mut snapshot) = snapshot_result {
                // Lock order: sessions -> active_session_id -> backend. The guards
                // taken by `with_current_poll_target` are still held here, so
                // `state.backend` is the third lock in the chain. Every other
                // `backend` lock site (supervisor/server teardown and setup) takes it
                // as a short temporary and never acquires `sessions` or
                // `active_session_id` while holding it, so this ordering cannot
                // invert. Never lock `backend` first and then reach for `sessions`.
                if session_backend == crate::inference::InferenceBackend::RapidMlx {
                    crate::inference::rapid_mlx::poller::enrich_launch_model_facts(
                        &mut snapshot,
                        session,
                        state.backend.lock().unwrap().as_ref(),
                        base,
                    );
                }
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
                *state.inference_metrics_session_id.lock().unwrap() = active_id.to_string();
                // Rapid-MLX stays exclusively in the normalized inference contract.
                if snapshot.backend != crate::inference::InferenceBackend::LlamaCpp {
                    return;
                }
                let now_ms = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64;

                let mut m = state.llama_metrics.lock().unwrap();
                project_optional_llama_metrics(&mut m, &snapshot);

                let slots = project_llama_throughput(
                    &mut m,
                    &snapshot,
                    llama_live_rates,
                    std::time::Instant::now(),
                    now_ms,
                );

                if let Some(details) = snapshot.backend_details.as_ref()
                    && slots.is_some()
                {
                    let gen_tokens = details
                        .get("slot_generation_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    if let Some(remaining) = details
                        .get("slot_generation_remaining")
                        .and_then(|v| v.as_u64())
                    {
                        m.slot_generation_remaining = remaining;
                    }
                    if let Some(limit) = details
                        .get("slot_generation_limit")
                        .and_then(|v| v.as_u64())
                    {
                        m.slot_generation_limit = limit;
                    }
                    if let Some(active) = details
                        .get("slot_generation_active")
                        .and_then(|v| v.as_bool())
                    {
                        m.slot_generation_active = active;
                    }
                    if let Some(available) = details
                        .get("slot_generation_available")
                        .and_then(|v| v.as_bool())
                    {
                        m.slot_generation_available = available;
                    }

                    // Live prefill progress: prompt_tokens_processed only
                    // advances while the slot is prefilling — use its delta
                    // for a live PP rate (the Prometheus counters stall).
                    let prompt_processed = details
                        .get("slot_prompt_processed")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    if let Some(total) = details.get("slot_prompt_total").and_then(|v| v.as_u64()) {
                        m.slot_prompt_total = total;
                    }
                    if let Some(progress) =
                        details.get("slot_prompt_progress").and_then(|v| v.as_f64())
                    {
                        m.slot_prompt_progress = progress;
                    }
                    m.slot_prompt_processed = prompt_processed;

                    m.slot_generation_tokens = gen_tokens;
                }

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
                    if let Some(slots) = slots {
                        m.slots = slots;
                    }
                }
                m.model_name = snapshot.model.unwrap_or_default();
            } else {
                clear_failed_llama_sample(state, llama_live_rates);
            }
        },
    )
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
    let mut llama_runtime_cache = crate::inference::llama_cpp::LlamaRuntimeCache::default();
    let mut previous_backend = None;
    // Prometheus counters advance at completion; slots provide live progress.
    let mut llama_live_rates = LiveRateTracker::default();

    loop {
        if !enabled {
            state.llama_poll_notify.notified().await;
            enabled = true;
        }

        let active_id = { state.active_session_id.lock().unwrap().clone() };
        if active_id.is_empty() {
            llama_runtime_cache = Default::default();
            previous_backend = None;
            llama_previous_counters = None;
            llama_previous_counter_session = None;
            llama_live_rates = LiveRateTracker::default();
            *state.llama_metrics.lock().unwrap() = Default::default();
            *state.inference_metrics.lock().unwrap() = None;
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
                llama_runtime_cache = Default::default();
                previous_backend = None;
                llama_live_rates = LiveRateTracker::default();
                *state.llama_metrics.lock().unwrap() = Default::default();
                *state.inference_metrics.lock().unwrap() = None;
                enabled = false;
                tokio::time::sleep(Duration::from_secs(poll_interval)).await;
                continue;
            }
        };
        // Keep the original URL for attribution; a query-value slash is data.
        // Request construction retains the historical trimmed base.
        let source_base = base;
        let base = source_base.trim_end_matches('/').to_string();
        let target_changed =
            llama_runtime_cache.select_target(&base, api_key.as_deref(), &active_id);
        if target_changed || previous_backend != Some(session_backend) {
            // Endpoint/auth changes can occur within the same session id.
            if previous_backend != Some(session_backend) {
                llama_runtime_cache = Default::default();
                llama_runtime_cache.select_target(&base, api_key.as_deref(), &active_id);
            }
            rapid_poller = None;
            llama_previous_counters = None;
            llama_previous_counter_session = None;
            llama_live_rates = LiveRateTracker::default();
            *state.llama_metrics.lock().unwrap() = Default::default();
            *state.inference_metrics.lock().unwrap() = None;
        }
        previous_backend = Some(session_backend);
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
                        if active_poll_target_matches(
                            &state,
                            &active_id,
                            session_backend,
                            &base,
                            api_key.as_deref(),
                        ) && let Ok(json) = serde_json::from_str::<serde_json::Value>(&body)
                        {
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
        if !active_poll_target_matches(
            &state,
            &active_id,
            session_backend,
            &base,
            api_key.as_deref(),
        ) {
            continue;
        }

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
            // Historical throughput may remain, but optional efficiency/facts
            // must not masquerade as current telemetry after a failed poll.
            clear_failed_llama_sample(&state, &mut llama_live_rates);
            tokio::time::sleep(Duration::from_secs(poll_interval)).await;
            continue;
        }

        // Use the backend adapter to poll normalized metrics. Attached Rapid-MLX and
        // llama.cpp sessions construct their poll directly from the resolved endpoint;
        // they do not have a spawned adapter in state.backend (Attach sessions never
        // populate it — see poll_llama_cpp_metrics for the llama.cpp case).
        {
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
            } else if matches!(
                session_backend,
                crate::inference::InferenceBackend::LlamaCpp
            ) {
                rapid_poller = None;
                crate::inference::llama_cpp::poll_llama_cpp_metrics(
                    &source_base,
                    api_key.as_deref(),
                    &active_id,
                    &mut llama_previous_counters,
                    &mut llama_previous_counter_session,
                    &mut llama_runtime_cache,
                )
                .await
            } else {
                Err(anyhow::anyhow!("active backend adapter unavailable"))
            };
            publish_poll_result(
                &state,
                &active_id,
                session_backend,
                &base,
                api_key.as_deref(),
                snapshot_result,
                &mut llama_live_rates,
            );
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

    fn rate_slot(
        id: u64,
        task: u64,
        processing: bool,
        generated: u64,
        prompt: u64,
    ) -> serde_json::Value {
        serde_json::json!({
            "id": id, "id_task": task, "is_processing": processing,
            "n_prompt_tokens_processed": prompt,
            "next_token": [{"n_decoded": generated, "n_remain": 1000, "has_next_token": processing}]
        })
    }

    // Exercise the actual backend parser, including its retained idle output
    // counters, rather than inventing a different SlotSnapshot wire contract.
    fn rate_snapshot(
        raw_slots: Vec<serde_json::Value>,
    ) -> crate::inference::metrics::InferenceMetricsSnapshot {
        let parsed =
            crate::llama::metrics::parse_slot_metrics(&serde_json::to_string(&raw_slots).unwrap())
                .unwrap();
        let mut snapshot =
            crate::inference::metrics::InferenceMetricsSnapshot::empty(InferenceBackend::LlamaCpp);
        snapshot.generation_tokens_per_second = Some(999.0);
        snapshot.prompt_tokens_per_second = Some(9999.0);
        snapshot.backend_details = Some(serde_json::json!({
            "slots": parsed.slots,
            "slots_processing": parsed.slots_processing,
            "slot_generation_tokens": parsed.slot_generation_tokens,
            "slot_prompt_processed": parsed.slot_prompt_processed,
        }));
        snapshot
    }

    fn rate_poll(
        tracker: &mut super::LiveRateTracker,
        metrics: &mut crate::llama::metrics::LlamaMetrics,
        origin: std::time::Instant,
        seconds: u64,
        snapshot: &crate::inference::metrics::InferenceMetricsSnapshot,
    ) {
        let _ = super::project_llama_throughput(
            metrics,
            snapshot,
            tracker,
            origin + std::time::Duration::from_secs(seconds),
            1000 + seconds * 1000,
        );
    }

    fn assert_live_rates(m: &crate::llama::metrics::LlamaMetrics, generated: f64, prompt: f64) {
        assert!(
            (m.generation_tokens_per_sec - generated).abs() < 1e-9,
            "{m:?}"
        );
        assert!((m.prompt_tokens_per_sec - prompt).abs() < 1e-9, "{m:?}");
        assert_eq!(m.generation_throughput_active, generated > 0.0);
        assert_eq!(m.prompt_throughput_active, prompt > 0.0);
    }

    #[test]
    fn live_rates_rebase_back_to_back_smaller_requests_without_idle_poll() {
        let origin = std::time::Instant::now();
        let mut tracker = super::LiveRateTracker::default();
        let mut m = crate::llama::metrics::LlamaMetrics::default();
        rate_poll(
            &mut tracker,
            &mut m,
            origin,
            0,
            &rate_snapshot(vec![rate_slot(0, 1, true, 5, 1024)]),
        );
        rate_poll(
            &mut tracker,
            &mut m,
            origin,
            2,
            &rate_snapshot(vec![rate_slot(0, 1, true, 37, 3072)]),
        );
        assert_live_rates(&m, 16.0, 1024.0);
        rate_poll(
            &mut tracker,
            &mut m,
            origin,
            3,
            &rate_snapshot(vec![rate_slot(0, 2, true, 1, 16)]),
        );
        assert_live_rates(&m, 0.0, 0.0);
        assert_eq!(m.last_generation_tokens_per_sec, 16.0);
        assert_eq!(m.last_prompt_tokens_per_sec, 1024.0);
        assert_eq!(m.last_generation_throughput_unix_ms, 3000);
        assert_eq!(m.last_prompt_throughput_unix_ms, 3000);
        rate_poll(
            &mut tracker,
            &mut m,
            origin,
            5,
            &rate_snapshot(vec![rate_slot(0, 2, true, 17, 1040)]),
        );
        assert_live_rates(&m, 8.0, 512.0);
    }

    #[test]
    fn changed_request_identity_rebases_even_with_equal_or_larger_counters() {
        for (generated, prompt) in [(32, 2048), (128, 8192)] {
            let origin = std::time::Instant::now();
            let mut tracker = super::LiveRateTracker::default();
            let mut m = crate::llama::metrics::LlamaMetrics::default();
            rate_poll(
                &mut tracker,
                &mut m,
                origin,
                0,
                &rate_snapshot(vec![rate_slot(0, 1, true, 0, 0)]),
            );
            rate_poll(
                &mut tracker,
                &mut m,
                origin,
                2,
                &rate_snapshot(vec![rate_slot(0, 1, true, 32, 2048)]),
            );
            rate_poll(
                &mut tracker,
                &mut m,
                origin,
                3,
                &rate_snapshot(vec![rate_slot(0, 2, true, generated, prompt)]),
            );
            assert_live_rates(&m, 0.0, 0.0);
            rate_poll(
                &mut tracker,
                &mut m,
                origin,
                5,
                &rate_snapshot(vec![rate_slot(0, 2, true, generated + 16, prompt + 1024)]),
            );
            assert_live_rates(&m, 8.0, 512.0);
        }
    }

    #[test]
    fn same_request_counter_regression_rebases_both_windows_and_emas() {
        for (generated, prompt) in [(1, 4096), (64, 1), (1, 1)] {
            let origin = std::time::Instant::now();
            let mut tracker = super::LiveRateTracker::default();
            let mut m = crate::llama::metrics::LlamaMetrics::default();
            rate_poll(
                &mut tracker,
                &mut m,
                origin,
                0,
                &rate_snapshot(vec![rate_slot(0, 1, true, 0, 0)]),
            );
            rate_poll(
                &mut tracker,
                &mut m,
                origin,
                2,
                &rate_snapshot(vec![rate_slot(0, 1, true, 32, 2048)]),
            );
            rate_poll(
                &mut tracker,
                &mut m,
                origin,
                3,
                &rate_snapshot(vec![rate_slot(0, 1, true, generated, prompt)]),
            );
            assert_live_rates(&m, 0.0, 0.0);
            rate_poll(
                &mut tracker,
                &mut m,
                origin,
                5,
                &rate_snapshot(vec![rate_slot(0, 1, true, generated + 16, prompt + 1024)]),
            );
            assert_live_rates(&m, 8.0, 512.0);
        }
    }

    #[test]
    fn multi_slot_joins_and_leaves_rebase_even_when_aggregate_increases() {
        let origin = std::time::Instant::now();
        let mut tracker = super::LiveRateTracker::default();
        let mut m = crate::llama::metrics::LlamaMetrics::default();
        rate_poll(
            &mut tracker,
            &mut m,
            origin,
            0,
            &rate_snapshot(vec![rate_slot(0, 1, true, 0, 0)]),
        );
        rate_poll(
            &mut tracker,
            &mut m,
            origin,
            2,
            &rate_snapshot(vec![rate_slot(0, 1, true, 32, 2048)]),
        );
        rate_poll(
            &mut tracker,
            &mut m,
            origin,
            3,
            &rate_snapshot(vec![
                rate_slot(0, 1, true, 48, 3072),
                rate_slot(1, 2, true, 100, 8192),
            ]),
        );
        assert_live_rates(&m, 0.0, 0.0);
        rate_poll(
            &mut tracker,
            &mut m,
            origin,
            5,
            &rate_snapshot(vec![
                rate_slot(0, 1, true, 64, 4096),
                rate_slot(1, 2, true, 116, 9216),
            ]),
        );
        assert_live_rates(&m, 16.0, 1024.0);
        rate_poll(
            &mut tracker,
            &mut m,
            origin,
            6,
            &rate_snapshot(vec![
                rate_slot(0, 1, false, 64, 4096),
                rate_slot(1, 2, true, 300, 15000),
            ]),
        );
        assert_live_rates(&m, 0.0, 0.0);
        rate_poll(
            &mut tracker,
            &mut m,
            origin,
            8,
            &rate_snapshot(vec![
                rate_slot(0, 1, false, 9999, 9999),
                rate_slot(1, 2, true, 316, 16024),
            ]),
        );
        assert_live_rates(&m, 8.0, 512.0);
    }

    #[test]
    fn per_slot_regressions_are_not_hidden_by_other_slots_increasing() {
        for (generated, prompt) in [(1, 4096), (64, 1)] {
            let origin = std::time::Instant::now();
            let mut tracker = super::LiveRateTracker::default();
            let mut m = crate::llama::metrics::LlamaMetrics::default();
            rate_poll(
                &mut tracker,
                &mut m,
                origin,
                0,
                &rate_snapshot(vec![
                    rate_slot(0, 1, true, 0, 0),
                    rate_slot(1, 2, true, 0, 0),
                ]),
            );
            rate_poll(
                &mut tracker,
                &mut m,
                origin,
                2,
                &rate_snapshot(vec![
                    rate_slot(0, 1, true, 32, 2048),
                    rate_slot(1, 2, true, 32, 2048),
                ]),
            );
            rate_poll(
                &mut tracker,
                &mut m,
                origin,
                3,
                &rate_snapshot(vec![
                    rate_slot(0, 1, true, generated, prompt),
                    rate_slot(1, 2, true, 200, 8192),
                ]),
            );
            assert_live_rates(&m, 0.0, 0.0);
            rate_poll(
                &mut tracker,
                &mut m,
                origin,
                5,
                &rate_snapshot(vec![
                    rate_slot(0, 1, true, generated + 16, prompt + 1024),
                    rate_slot(1, 2, true, 216, 9216),
                ]),
            );
            assert_live_rates(&m, 16.0, 1024.0);
        }
    }

    #[test]
    fn live_quantized_rates_hold_despite_different_backend_snapshot_rates() {
        let origin = std::time::Instant::now();
        let mut tracker = super::LiveRateTracker::default();
        let mut m = crate::llama::metrics::LlamaMetrics::default();
        rate_poll(
            &mut tracker,
            &mut m,
            origin,
            0,
            &rate_snapshot(vec![rate_slot(0, 1, true, 0, 0)]),
        );
        rate_poll(
            &mut tracker,
            &mut m,
            origin,
            2,
            &rate_snapshot(vec![rate_slot(0, 1, true, 32, 2048)]),
        );
        for (seconds, generated, prompt, fallback) in [(3, 47, 3071, 0.0), (10, 47, 3071, 700.0)] {
            let mut snapshot = rate_snapshot(vec![rate_slot(0, 1, true, generated, prompt)]);
            snapshot.generation_tokens_per_second = Some(fallback);
            snapshot.prompt_tokens_per_second = Some(fallback);
            rate_poll(&mut tracker, &mut m, origin, seconds, &snapshot);
            assert_live_rates(&m, 16.0, 1024.0);
            assert_eq!(m.last_generation_throughput_unix_ms, 3000);
            assert_eq!(m.last_prompt_throughput_unix_ms, 3000);
        }
        rate_poll(
            &mut tracker,
            &mut m,
            origin,
            12,
            &rate_snapshot(vec![rate_slot(0, 1, true, 48, 3072)]),
        );
        assert_live_rates(&m, 8.8, 655.36);
        assert_eq!(m.last_generation_throughput_unix_ms, 13000);
        assert_eq!(m.last_prompt_throughput_unix_ms, 13000);
    }

    #[test]
    fn slot_order_does_not_change_identity_or_reset_held_rates() {
        let origin = std::time::Instant::now();
        let mut tracker = super::LiveRateTracker::default();
        let mut m = crate::llama::metrics::LlamaMetrics::default();
        rate_poll(
            &mut tracker,
            &mut m,
            origin,
            0,
            &rate_snapshot(vec![
                rate_slot(0, 1, true, 0, 0),
                rate_slot(1, 2, true, 0, 0),
            ]),
        );
        rate_poll(
            &mut tracker,
            &mut m,
            origin,
            2,
            &rate_snapshot(vec![
                rate_slot(1, 2, true, 16, 1024),
                rate_slot(0, 1, true, 16, 1024),
            ]),
        );
        assert_live_rates(&m, 16.0, 1024.0);
        rate_poll(
            &mut tracker,
            &mut m,
            origin,
            3,
            &rate_snapshot(vec![
                rate_slot(0, 1, true, 17, 1025),
                rate_slot(1, 2, true, 17, 1025),
            ]),
        );
        assert_live_rates(&m, 16.0, 1024.0);
    }

    #[test]
    fn zero_prefill_and_idle_clear_live_rates_but_preserve_last_rates_and_timestamps() {
        let origin = std::time::Instant::now();
        let mut tracker = super::LiveRateTracker::default();
        let mut m = crate::llama::metrics::LlamaMetrics::default();
        rate_poll(
            &mut tracker,
            &mut m,
            origin,
            0,
            &rate_snapshot(vec![rate_slot(0, 1, true, 0, 0)]),
        );
        rate_poll(
            &mut tracker,
            &mut m,
            origin,
            2,
            &rate_snapshot(vec![rate_slot(0, 1, true, 32, 2048)]),
        );
        for (seconds, slots) in [
            (3, vec![rate_slot(0, 2, true, 0, 0)]),
            (4, vec![rate_slot(0, 2, false, 99, 4096)]),
            (5, vec![]),
        ] {
            let mut snapshot = rate_snapshot(slots);
            snapshot.generation_tokens_per_second = Some(0.0);
            snapshot.prompt_tokens_per_second = Some(0.0);
            rate_poll(&mut tracker, &mut m, origin, seconds, &snapshot);
            assert_live_rates(&m, 0.0, 0.0);
            assert_eq!(m.last_generation_tokens_per_sec, 16.0);
            assert_eq!(m.last_prompt_tokens_per_sec, 1024.0);
            assert_eq!(m.last_generation_throughput_unix_ms, 3000);
            assert_eq!(m.last_prompt_throughput_unix_ms, 3000);
        }
        rate_poll(
            &mut tracker,
            &mut m,
            origin,
            20,
            &rate_snapshot(vec![rate_slot(0, 3, true, 0, 0)]),
        );
        rate_poll(
            &mut tracker,
            &mut m,
            origin,
            22,
            &rate_snapshot(vec![rate_slot(0, 3, true, 16, 1024)]),
        );
        assert_live_rates(&m, 8.0, 512.0);
    }

    #[test]
    fn unavailable_or_invalid_slots_use_backend_rates_and_rebase_on_return() {
        for slots in [
            None,
            Some(serde_json::Value::Null),
            Some(serde_json::json!([{}])),
        ] {
            let origin = std::time::Instant::now();
            let mut tracker = super::LiveRateTracker::default();
            let mut m = crate::llama::metrics::LlamaMetrics::default();
            rate_poll(
                &mut tracker,
                &mut m,
                origin,
                0,
                &rate_snapshot(vec![rate_slot(0, 1, true, 0, 0)]),
            );
            rate_poll(
                &mut tracker,
                &mut m,
                origin,
                2,
                &rate_snapshot(vec![rate_slot(0, 1, true, 32, 2048)]),
            );
            let mut fallback = crate::inference::metrics::InferenceMetricsSnapshot::empty(
                InferenceBackend::LlamaCpp,
            );
            fallback.generation_tokens_per_second = Some(7.0);
            fallback.prompt_tokens_per_second = Some(70.0);
            fallback.backend_details = slots.map(|v| serde_json::json!({"slots": v}));
            rate_poll(&mut tracker, &mut m, origin, 3, &fallback);
            assert_live_rates(&m, 7.0, 70.0);
            assert_eq!(m.last_generation_throughput_unix_ms, 4000);
            assert_eq!(m.last_prompt_throughput_unix_ms, 4000);
            fallback.generation_tokens_per_second = Some(0.0);
            fallback.prompt_tokens_per_second = Some(0.0);
            rate_poll(&mut tracker, &mut m, origin, 4, &fallback);
            assert_live_rates(&m, 0.0, 0.0);
            assert_eq!(m.last_generation_tokens_per_sec, 7.0);
            assert_eq!(m.last_prompt_tokens_per_sec, 70.0);
            assert_eq!(m.last_generation_throughput_unix_ms, 4000);
            assert_eq!(m.last_prompt_throughput_unix_ms, 4000);
            rate_poll(
                &mut tracker,
                &mut m,
                origin,
                5,
                &rate_snapshot(vec![rate_slot(0, 1, true, 100, 8192)]),
            );
            assert_live_rates(&m, 0.0, 0.0);
            rate_poll(
                &mut tracker,
                &mut m,
                origin,
                7,
                &rate_snapshot(vec![rate_slot(0, 1, true, 116, 9216)]),
            );
            assert_live_rates(&m, 8.0, 512.0);
        }
    }

    #[test]
    fn legacy_slots_without_per_slot_prompt_counts_use_aggregate_prompt_progress() {
        let origin = std::time::Instant::now();
        let mut tracker = super::LiveRateTracker::default();
        let mut m = crate::llama::metrics::LlamaMetrics::default();
        for (seconds, generated, prompt, expected_gen, expected_prompt) in [
            (0, 0, 0, 0.0, 0.0),
            (2, 32, 2048, 16.0, 1024.0),
            (3, 64, 1, 0.0, 0.0),
            (5, 80, 1025, 8.0, 512.0),
        ] {
            let mut snapshot = rate_snapshot(vec![rate_slot(0, 1, true, generated, prompt)]);
            snapshot.backend_details.as_mut().unwrap()["slots"][0]
                .as_object_mut()
                .unwrap()
                .remove("prompt_tokens_processed");
            rate_poll(&mut tracker, &mut m, origin, seconds, &snapshot);
            assert_live_rates(&m, expected_gen, expected_prompt);
        }
    }

    #[test]
    fn live_report_minimum_elapsed_windows_do_not_consume_unreported_chunks() {
        let origin = std::time::Instant::now();
        let mut tracker = super::LiveRateTracker::default();
        let mut m = crate::llama::metrics::LlamaMetrics::default();
        let start = rate_snapshot(vec![rate_slot(0, 1, true, 0, 0)]);
        let chunk = rate_snapshot(vec![rate_slot(0, 1, true, 16, 1024)]);
        let _ = super::project_llama_throughput(&mut m, &start, &mut tracker, origin, 1000);
        let _ = super::project_llama_throughput(
            &mut m,
            &chunk,
            &mut tracker,
            origin + std::time::Duration::from_millis(500),
            1500,
        );
        assert_live_rates(&m, 0.0, 0.0);
        let _ = super::project_llama_throughput(
            &mut m,
            &chunk,
            &mut tracker,
            origin + std::time::Duration::from_secs(1),
            2000,
        );
        assert_live_rates(&m, 16.0, 0.0);
        let _ = super::project_llama_throughput(
            &mut m,
            &chunk,
            &mut tracker,
            origin + std::time::Duration::from_secs(2),
            3000,
        );
        assert_live_rates(&m, 16.0, 512.0);
    }

    #[test]
    fn repeated_positive_idle_snapshots_preserve_last_measurement_age_even_after_tiny_request() {
        let origin = std::time::Instant::now();
        let mut tracker = super::LiveRateTracker::default();
        let mut m = crate::llama::metrics::LlamaMetrics::default();
        rate_poll(
            &mut tracker,
            &mut m,
            origin,
            0,
            &rate_snapshot(vec![rate_slot(0, 1, true, 0, 0)]),
        );
        rate_poll(
            &mut tracker,
            &mut m,
            origin,
            2,
            &rate_snapshot(vec![rate_slot(0, 1, true, 32, 2048)]),
        );
        rate_poll(
            &mut tracker,
            &mut m,
            origin,
            3,
            &rate_snapshot(vec![rate_slot(0, 2, true, 5, 100)]),
        );
        assert_live_rates(&m, 0.0, 0.0);
        let mut idle = rate_snapshot(vec![rate_slot(0, 2, false, 5, 100)]);
        idle.generation_tokens_per_second = Some(7.0);
        idle.prompt_tokens_per_second = Some(70.0);
        rate_poll(&mut tracker, &mut m, origin, 4, &idle);
        assert_live_rates(&m, 0.0, 0.0);
        assert_eq!(m.last_generation_tokens_per_sec, 16.0);
        assert_eq!(m.last_prompt_tokens_per_sec, 1024.0);
        assert_eq!(m.last_generation_throughput_unix_ms, 3000);
        assert_eq!(m.last_prompt_throughput_unix_ms, 3000);
        rate_poll(&mut tracker, &mut m, origin, 5, &idle);
        assert_live_rates(&m, 0.0, 0.0);
        assert_eq!(m.last_generation_tokens_per_sec, 16.0);
        assert_eq!(m.last_prompt_tokens_per_sec, 1024.0);
        assert_eq!(m.last_generation_throughput_unix_ms, 3000);
        assert_eq!(m.last_prompt_throughput_unix_ms, 3000);
        idle.generation_tokens_per_second = Some(17.0);
        idle.prompt_tokens_per_second = Some(170.0);
        rate_poll(&mut tracker, &mut m, origin, 6, &idle);
        assert_live_rates(&m, 0.0, 0.0);
        assert_eq!(m.last_generation_tokens_per_sec, 16.0);
        assert_eq!(m.last_prompt_tokens_per_sec, 1024.0);
        assert_eq!(m.last_generation_throughput_unix_ms, 3000);
        assert_eq!(m.last_prompt_throughput_unix_ms, 3000);
        idle.generation_tokens_per_second = Some(0.0);
        idle.prompt_tokens_per_second = Some(0.0);
        rate_poll(&mut tracker, &mut m, origin, 7, &idle);
        assert_live_rates(&m, 0.0, 0.0);
        assert_eq!(m.last_generation_tokens_per_sec, 16.0);
        assert_eq!(m.last_prompt_tokens_per_sec, 1024.0);
        assert_eq!(m.last_generation_throughput_unix_ms, 3000);
        assert_eq!(m.last_prompt_throughput_unix_ms, 3000);
    }

    #[test]
    fn regression_above_report_baseline_still_rebases_from_last_poll() {
        for (generated, prompt) in [(34, 3071), (47, 2050)] {
            let origin = std::time::Instant::now();
            let mut tracker = super::LiveRateTracker::default();
            let mut m = crate::llama::metrics::LlamaMetrics::default();
            rate_poll(
                &mut tracker,
                &mut m,
                origin,
                0,
                &rate_snapshot(vec![rate_slot(0, 1, true, 0, 0)]),
            );
            rate_poll(
                &mut tracker,
                &mut m,
                origin,
                2,
                &rate_snapshot(vec![rate_slot(0, 1, true, 32, 2048)]),
            );
            rate_poll(
                &mut tracker,
                &mut m,
                origin,
                3,
                &rate_snapshot(vec![rate_slot(0, 1, true, 47, 3071)]),
            );
            assert_live_rates(&m, 16.0, 1024.0);
            rate_poll(
                &mut tracker,
                &mut m,
                origin,
                4,
                &rate_snapshot(vec![rate_slot(0, 1, true, generated, prompt)]),
            );
            assert_live_rates(&m, 0.0, 0.0);
            rate_poll(
                &mut tracker,
                &mut m,
                origin,
                6,
                &rate_snapshot(vec![rate_slot(0, 1, true, generated + 16, prompt + 1024)]),
            );
            assert_live_rates(&m, 8.0, 512.0);
        }
    }

    #[test]
    fn missing_counter_availability_changes_rebase_without_inventing_progress() {
        let origin = std::time::Instant::now();
        let mut tracker = super::LiveRateTracker::default();
        let mut m = crate::llama::metrics::LlamaMetrics::default();
        rate_poll(
            &mut tracker,
            &mut m,
            origin,
            0,
            &rate_snapshot(vec![rate_slot(0, 1, true, 0, 0)]),
        );
        rate_poll(
            &mut tracker,
            &mut m,
            origin,
            2,
            &rate_snapshot(vec![rate_slot(0, 1, true, 32, 2048)]),
        );
        let mut missing = rate_slot(0, 1, true, 64, 4096);
        missing.as_object_mut().unwrap().remove("next_token");
        missing
            .as_object_mut()
            .unwrap()
            .remove("n_prompt_tokens_processed");
        rate_poll(
            &mut tracker,
            &mut m,
            origin,
            3,
            &rate_snapshot(vec![missing]),
        );
        assert_live_rates(&m, 0.0, 0.0);
        rate_poll(
            &mut tracker,
            &mut m,
            origin,
            4,
            &rate_snapshot(vec![rate_slot(0, 1, true, 100, 8192)]),
        );
        assert_live_rates(&m, 0.0, 0.0);
        rate_poll(
            &mut tracker,
            &mut m,
            origin,
            6,
            &rate_snapshot(vec![rate_slot(0, 1, true, 116, 9216)]),
        );
        assert_live_rates(&m, 8.0, 512.0);
    }

    #[test]
    fn failed_sample_discards_timers_and_emas_without_losing_last_measurement() {
        let state = AppState::default();
        let origin = std::time::Instant::now();
        let mut tracker = super::LiveRateTracker::default();
        {
            let mut m = state.llama_metrics.lock().unwrap();
            rate_poll(
                &mut tracker,
                &mut m,
                origin,
                0,
                &rate_snapshot(vec![rate_slot(0, 1, true, 0, 0)]),
            );
            rate_poll(
                &mut tracker,
                &mut m,
                origin,
                2,
                &rate_snapshot(vec![rate_slot(0, 1, true, 32, 2048)]),
            );
        }
        super::clear_failed_llama_sample(&state, &mut tracker);
        assert!(tracker.slots.is_empty());
        assert!(tracker.totals.is_none());
        assert!(tracker.generation.baseline.is_none());
        assert!(tracker.prompt.baseline.is_none());
        assert!(tracker.generation.ema.is_none());
        assert!(tracker.prompt.ema.is_none());
        let mut m = state.llama_metrics.lock().unwrap();
        assert_eq!(m.last_generation_tokens_per_sec, 16.0);
        assert_eq!(m.last_prompt_tokens_per_sec, 1024.0);
        assert_eq!(m.last_generation_throughput_unix_ms, 3000);
        assert_eq!(m.last_prompt_throughput_unix_ms, 3000);
        // Even identical task ids with larger counters cannot bridge a gap.
        rate_poll(
            &mut tracker,
            &mut m,
            origin,
            20,
            &rate_snapshot(vec![rate_slot(0, 1, true, 128, 8192)]),
        );
        assert_live_rates(&m, 0.0, 0.0);
        rate_poll(
            &mut tracker,
            &mut m,
            origin,
            22,
            &rate_snapshot(vec![rate_slot(0, 1, true, 144, 9216)]),
        );
        assert_live_rates(&m, 8.0, 512.0);
    }

    #[test]
    fn failed_poll_clears_optional_fields_without_fabricating_zeroes() {
        let state = AppState::default();
        {
            let mut m = state.llama_metrics.lock().unwrap();
            m.telemetry_session_id = Some("source-session".into());
            m.telemetry_endpoint = Some("sha256:source-endpoint".into());
            m.prompt_tokens_processed_total = Some(200.0);
            m.speculative_draft_tokens_total = Some(20);
            m.speculative_accepted_tokens_total = Some(10);
            m.speculative_verification_steps_total = Some(5);
            m.speculative_acceptance_rate = Some(0.5);
            m.model_params = Some(7_000_000_000);
            m.model_ctx_train = Some(32768);
            m.runtime_facts = Some(Default::default());
            m.prompt_tokens_cached_total = Some(100.0);
            m.speculative_enabled = Some(true);
            m.model_name = "old".into();
            m.slots.push(Default::default());
        }
        super::clear_optional_llama_metrics(&state);
        let m = state.llama_metrics.lock().unwrap();
        assert_eq!(m.telemetry_session_id.as_deref(), Some("source-session"));
        assert_eq!(
            m.telemetry_endpoint.as_deref(),
            Some("sha256:source-endpoint")
        );
        assert!(m.prompt_tokens_processed_total.is_none());
        assert!(m.speculative_draft_tokens_total.is_none());
        assert!(m.speculative_accepted_tokens_total.is_none());
        assert!(m.speculative_verification_steps_total.is_none());
        assert!(m.speculative_acceptance_rate.is_none());
        assert!(m.model_params.is_none());
        assert!(m.model_ctx_train.is_none());
        assert!(m.runtime_facts.is_none());
        assert!(m.prompt_tokens_cached_total.is_none());
        assert!(m.speculative_enabled.is_none());
        assert!(m.model_name.is_empty());
        assert!(m.slots.is_empty());
    }

    #[test]
    fn in_flight_target_check_rejects_endpoint_auth_backend_and_session_switches() {
        let state = AppState::default();
        *state.active_session_id.lock().unwrap() = "s".into();
        state
            .sessions
            .lock()
            .unwrap()
            .push(crate::state::Session::new_attach(
                "s".into(),
                "test".into(),
                "http://a:8001/".into(),
                Some("key-a".into()),
            ));
        assert!(super::active_poll_target_matches(
            &state,
            "s",
            InferenceBackend::LlamaCpp,
            "http://a:8001",
            Some("key-a"),
        ));
        assert!(!super::active_poll_target_matches(
            &state,
            "s",
            InferenceBackend::LlamaCpp,
            "http://b:8001",
            Some("key-a"),
        ));
        assert!(!super::active_poll_target_matches(
            &state,
            "s",
            InferenceBackend::LlamaCpp,
            "http://a:8001",
            Some("key-b"),
        ));
        assert!(!super::active_poll_target_matches(
            &state,
            "s",
            InferenceBackend::RapidMlx,
            "http://a:8001",
            Some("key-a"),
        ));
        {
            let mut sessions = state.sessions.lock().unwrap();
            sessions[0].mode = crate::state::SessionMode::Attach {
                endpoint: "http://b:8001".into(),
                api_key: Some("key-b".into()),
            };
        }
        assert!(!super::active_poll_target_matches(
            &state,
            "s",
            InferenceBackend::LlamaCpp,
            "http://a:8001",
            Some("key-a"),
        ));
        assert!(super::active_poll_target_matches(
            &state,
            "s",
            InferenceBackend::LlamaCpp,
            "http://b:8001",
            Some("key-b"),
        ));
        assert!(!super::with_current_poll_target(
            &state,
            "s",
            InferenceBackend::LlamaCpp,
            "http://a:8001",
            Some("key-a"),
            |_| state
                .llama_metrics
                .lock()
                .unwrap()
                .prompt_tokens_cached_total = Some(999.0),
        ));
        assert!(
            state
                .llama_metrics
                .lock()
                .unwrap()
                .prompt_tokens_cached_total
                .is_none()
        );
        assert!(super::with_current_poll_target(
            &state,
            "s",
            InferenceBackend::LlamaCpp,
            "http://b:8001",
            Some("key-b"),
            |_| state
                .llama_metrics
                .lock()
                .unwrap()
                .prompt_tokens_cached_total = Some(1.0),
        ));
        assert_eq!(
            state
                .llama_metrics
                .lock()
                .unwrap()
                .prompt_tokens_cached_total,
            Some(1.0)
        );
        *state.active_session_id.lock().unwrap() = "new".into();
        assert!(!super::active_poll_target_matches(
            &state,
            "s",
            InferenceBackend::LlamaCpp,
            "http://a:8001",
            Some("key-a"),
        ));
    }

    fn rapid_publication_state(session: crate::state::Session, adapter_port: u16) -> AppState {
        use crate::inference::backend::BackendAdapter;
        use crate::inference::rapid_mlx::{
            RapidMlxAdapter, model_resolver::ResolvedRapidMlxLaunchModel,
        };
        let state = AppState::default();
        let mut adapter = RapidMlxAdapter::from_resolved(
            crate::inference::rapid_mlx::runtime::RuntimeMetadata {
                version: "0.10.0".into(),
                ..Default::default()
            },
            ResolvedRapidMlxLaunchModel::validated_alias("catalog-alias").unwrap(),
        );
        adapter.host = "127.0.0.1".into();
        adapter.port = adapter_port;
        *state.backend.lock().unwrap() =
            Some(BackendAdapter::RapidMlx(std::sync::Arc::new(adapter)));
        *state.active_session_id.lock().unwrap() = session.id.clone();
        state.sessions.lock().unwrap().push(session);
        state
    }

    fn publish_rapid(state: &AppState, base: &str) -> bool {
        let mut snapshot =
            crate::inference::metrics::InferenceMetricsSnapshot::empty(InferenceBackend::RapidMlx);
        snapshot.backend_details = Some(serde_json::json!({
            "runtime_facts": {"engine_type": "batched"},
            // Stale value from an earlier publication must never survive.
            "launch_facts": {"version": "stale"},
        }));
        super::publish_poll_result(
            state,
            "s",
            InferenceBackend::RapidMlx,
            base,
            None,
            Ok(snapshot),
            &mut super::LiveRateTracker::default(),
        )
    }

    #[test]
    fn published_rapid_snapshot_carries_launch_facts_only_for_the_spawned_target() {
        let spawn = |port| {
            crate::state::Session::new_spawn_with_backend(
                "s".into(),
                "rapid".into(),
                port,
                String::new(),
                Some("127.0.0.1".into()),
                None,
                InferenceBackend::RapidMlx,
                None,
            )
        };
        let state = rapid_publication_state(spawn(8123), 8123);
        assert!(publish_rapid(&state, "http://127.0.0.1:8123"));
        let published = state.inference_metrics.lock().unwrap().clone().unwrap();
        let details = published.backend_details.unwrap();
        assert_eq!(details["launch_facts"]["version"], "0.10.0");
        assert_eq!(details["launch_facts"]["source_alias"], "catalog-alias");
        assert_eq!(details["runtime_facts"]["engine_type"], "batched");

        // Adapter for a different port: stale facts are cleared, not published.
        let state = rapid_publication_state(spawn(8123), 9999);
        assert!(publish_rapid(&state, "http://127.0.0.1:8123"));
        let published = state.inference_metrics.lock().unwrap().clone().unwrap();
        assert!(
            published
                .backend_details
                .unwrap()
                .get("launch_facts")
                .is_none()
        );

        // Attached Rapid-MLX target never receives the spawned adapter's facts.
        let mut attach = crate::state::Session::new_attach(
            "s".into(),
            "attached".into(),
            "http://127.0.0.1:8123".into(),
            None,
        );
        attach.backend = InferenceBackend::RapidMlx;
        let state = rapid_publication_state(attach, 8123);
        assert!(publish_rapid(&state, "http://127.0.0.1:8123"));
        let published = state.inference_metrics.lock().unwrap().clone().unwrap();
        assert!(
            published
                .backend_details
                .unwrap()
                .get("launch_facts")
                .is_none()
        );

        // A superseded target publishes nothing at all.
        let state = rapid_publication_state(spawn(8123), 8123);
        assert!(!publish_rapid(&state, "http://127.0.0.1:1"));
        assert!(state.inference_metrics.lock().unwrap().is_none());
    }

    #[test]
    fn backend_change_with_same_session_id_clears_llama_facts() {
        let state = AppState::default();
        *state.inference_metrics_session_id.lock().unwrap() = "s".into();
        *state.inference_metrics.lock().unwrap() = Some(
            crate::inference::metrics::InferenceMetricsSnapshot::empty(InferenceBackend::LlamaCpp),
        );
        state.llama_metrics.lock().unwrap().runtime_facts = Some(Default::default());
        reset_inference_poll_state_if_session_changed(&state, "s", InferenceBackend::RapidMlx);
        assert!(state.llama_metrics.lock().unwrap().runtime_facts.is_none());
        assert!(state.inference_metrics.lock().unwrap().is_none());
    }

    #[test]
    fn session_switch_clears_llama_runtime_and_efficiency_fields() {
        let state = AppState::default();
        *state.inference_metrics_session_id.lock().unwrap() = "old".into();
        {
            let mut m = state.llama_metrics.lock().unwrap();
            m.telemetry_session_id = Some("old".into());
            m.telemetry_endpoint = Some("sha256:old-endpoint".into());
            m.prompt_tokens_cached_total = Some(123.0);
            m.speculative_verification_steps_total = Some(5);
            m.speculative_enabled = Some(true);
            m.runtime_facts = Some(Default::default());
            m.model_name = "old-model".into();
            m.slots.push(Default::default());
        }
        reset_inference_poll_state_if_session_changed(&state, "new", InferenceBackend::RapidMlx);
        let m = state.llama_metrics.lock().unwrap();
        assert!(m.telemetry_session_id.is_none());
        assert!(m.telemetry_endpoint.is_none());
        assert_eq!(m.prompt_tokens_cached_total, None);
        assert_eq!(m.speculative_verification_steps_total, None);
        assert_eq!(m.speculative_enabled, None);
        assert!(m.runtime_facts.is_none());
        assert!(m.slots.is_empty());
        assert!(m.model_name.is_empty());
    }

    #[test]
    fn default_llama_metrics_have_no_source_tags() {
        let m = crate::llama::metrics::LlamaMetrics::default();
        assert!(m.telemetry_session_id.is_none());
        assert!(m.telemetry_endpoint.is_none());
    }

    #[test]
    fn optional_llama_projection_preserves_zero_and_clears_missing_or_foreign_fields() {
        let mut m = crate::llama::metrics::LlamaMetrics::default();
        let mut snapshot =
            crate::inference::metrics::InferenceMetricsSnapshot::empty(InferenceBackend::LlamaCpp);
        snapshot.backend_details = Some(serde_json::json!({
            "prompt_tokens_processed_total": 0.0, "prompt_tokens_cached_total": 0.0,
            "speculative_draft_tokens_total": 0, "speculative_accepted_tokens_total": 0,
            "speculative_verification_steps_total": 0, "speculative_enabled": false,
            "runtime_facts": {"model_name": "actual"},
            "telemetry_session_id": "source-session", "telemetry_endpoint": "http://source:8001"
        }));
        super::project_optional_llama_metrics(&mut m, &snapshot);
        assert_eq!(m.prompt_tokens_processed_total, Some(0.0));
        assert_eq!(m.prompt_tokens_cached_total, Some(0.0));
        assert_eq!(m.speculative_draft_tokens_total, Some(0));
        assert_eq!(m.speculative_accepted_tokens_total, Some(0));
        assert_eq!(m.speculative_verification_steps_total, Some(0));
        assert_eq!(m.speculative_enabled, Some(false));
        assert_eq!(m.telemetry_session_id.as_deref(), Some("source-session"));
        assert_eq!(m.telemetry_endpoint.as_deref(), Some("http://source:8001"));
        let wire = serde_json::to_value(&m).unwrap();
        assert_eq!(wire["prompt_tokens_cached_total"], 0.0);
        assert_eq!(wire["runtime_facts"]["model_name"], "actual");
        snapshot.backend_details = None;
        super::project_optional_llama_metrics(&mut m, &snapshot);
        assert!(m.telemetry_session_id.is_none());
        assert!(m.telemetry_endpoint.is_none());
        assert_eq!(m.prompt_tokens_cached_total, None);
        assert!(m.runtime_facts.is_none());
        assert!(serde_json::to_value(&m).unwrap()["prompt_tokens_cached_total"].is_null());
        snapshot.backend = InferenceBackend::RapidMlx;
        snapshot.backend_details = Some(serde_json::json!({"prompt_tokens_cached_total": 999}));
        super::project_optional_llama_metrics(&mut m, &snapshot);
        assert_eq!(m.prompt_tokens_cached_total, None);
    }

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
