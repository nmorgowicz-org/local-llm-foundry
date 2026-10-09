use crate::inference::InferenceBackend;
use crate::inference::metrics::{HealthState, InferenceMetricsSnapshot};
use anyhow::{Context, Result, anyhow, bail};
use futures_util::StreamExt;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, SystemTime};

const GIB_BYTES: f64 = 1_073_741_824.0;
const CALL_SPACING: Duration = Duration::from_millis(200);
const STATUS_BODY_LIMIT: usize = 512 * 1024;
const CACHE_BODY_LIMIT: usize = 256 * 1024;
const METRICS_BODY_LIMIT: usize = 256 * 1024;
const HEALTH_TIMEOUT: Duration = Duration::from_secs(2);
const METRICS_TIMEOUT: Duration = Duration::from_secs(2);
const CLIENT_CONNECT_TIMEOUT: Duration = Duration::from_secs(2);
const CLIENT_TIMEOUT: Duration = Duration::from_secs(10);
const ACTIVE_REQUEST_LIMIT: usize = 64;

#[derive(Deserialize)]
struct StatusResponse {
    #[serde(default)]
    status: Option<String>,
    model: Option<String>,
    uptime_s: Option<f64>,
    steps_executed: Option<u64>,
    num_running: Option<u64>,
    num_waiting: Option<u64>,
    total_requests_processed: Option<u64>,
    total_prompt_tokens: Option<u64>,
    total_completion_tokens: Option<u64>,
    generation_tps: Option<f64>,
    prompt_tps: Option<f64>,
    metal: Option<MetalMetrics>,
    cache: Option<Value>,
    requests: Option<Vec<Value>>,
    progress: Option<Value>,
}

#[derive(Deserialize)]
struct MetalMetrics {
    active_memory_gb: Option<f64>,
    peak_memory_gb: Option<f64>,
    cache_memory_gb: Option<f64>,
}

pub struct RapidMlxPoller {
    client: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
}

impl RapidMlxPoller {
    pub fn new(host: &str, port: u16, api_key: Option<&str>) -> Self {
        let host = match host {
            "0.0.0.0" | "::" | "[::]" => "127.0.0.1",
            "::1" => "[::1]",
            host => host,
        };
        Self::from_base_url(format!("http://{host}:{port}"), api_key)
    }

    pub fn from_base_url(base_url: String, api_key: Option<&str>) -> Self {
        // The bearer key must only ever reach the configured endpoint, and a
        // redirect could point the poller at an unrelated host or path. Never
        // follow one: a 3xx surfaces as a non-success status instead. The
        // client-level timeout is a backstop; per-request timeouts and body
        // limits below stay tighter.
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(CLIENT_CONNECT_TIMEOUT)
            .timeout(CLIENT_TIMEOUT)
            .build()
            .expect("static reqwest client configuration is valid");
        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.filter(|key| !key.is_empty()).map(str::to_string),
        }
    }

    pub fn matches_target(&self, base_url: &str, api_key: Option<&str>) -> bool {
        use subtle::ConstantTimeEq;

        let candidate_key = api_key.filter(|key| !key.is_empty());
        let key_matches = match (self.api_key.as_deref(), candidate_key) {
            (Some(left), Some(right)) => left.as_bytes().ct_eq(right.as_bytes()).into(),
            (None, None) => true,
            _ => false,
        };
        self.base_url == base_url.trim_end_matches('/') && key_matches
    }

    pub async fn poll(&self) -> Result<InferenceMetricsSnapshot> {
        let (health_ok, runtime_facts) = self.probe_health().await;

        tokio::time::sleep(CALL_SPACING).await;
        let status_response = self
            .authenticated_get(&format!("{}/v1/status", self.base_url))
            .timeout(Duration::from_secs(3))
            .send()
            .await?
            .error_for_status()
            .context("Rapid-MLX /v1/status returned an error status")?;
        let mut status: StatusResponse =
            parse_json_limited(status_response, STATUS_BODY_LIMIT, "Rapid-MLX /v1/status").await?;
        status.model = status
            .model
            .filter(|model| !model.is_empty() && model.len() <= 512);
        status.requests = status.requests.map(sanitize_requests);
        validate_status(&status)?;

        tokio::time::sleep(CALL_SPACING).await;
        let endpoint_cache = match self
            .authenticated_get(&format!("{}/v1/cache/stats", self.base_url))
            .timeout(Duration::from_secs(2))
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => {
                parse_json_limited::<Value>(response, CACHE_BODY_LIMIT, "Rapid-MLX /v1/cache/stats")
                    .await
                    .ok()
                    .and_then(recognized_endpoint_cache)
            }
            Ok(_) | Err(_) => None,
        };

        tokio::time::sleep(CALL_SPACING).await;
        let telemetry = self.optional_telemetry().await;

        let status_cache = status.cache.as_ref().and_then(normalized_status_cache);
        let global_cache_hit_rate = status_cache
            .as_ref()
            .and_then(|cache| cache.get("hit_rate"))
            .and_then(Value::as_f64);
        let global_cache_entries = status_cache
            .as_ref()
            .and_then(|cache| cache.get("entry_count"))
            .and_then(Value::as_u64);
        let cache_metrics = merge_cache_metrics(status_cache, endpoint_cache);

        let health = if !health_ok {
            HealthState::Degraded
        } else {
            match status.status.as_deref() {
                Some("generating" | "idle") => HealthState::Ok,
                Some("not_loaded") => HealthState::NotLoaded,
                _ => HealthState::Degraded,
            }
        };
        let ready = match status.status.as_deref() {
            Some("not_loaded") => Some(false),
            Some("generating" | "idle") => Some(true),
            _ => None,
        };
        let metal = status.metal.as_ref();

        Ok(InferenceMetricsSnapshot {
            sampled_at: SystemTime::now(),
            backend: InferenceBackend::RapidMlx,
            health: Some(health),
            ready,
            model: status.model,
            uptime_seconds: status.uptime_s,
            generation_tokens_per_second: status.generation_tps,
            prompt_tokens_per_second: status.prompt_tps,
            running_requests: status.num_running,
            waiting_requests: status.num_waiting,
            completed_requests_total: status.total_requests_processed,
            prompt_tokens_total: status.total_prompt_tokens,
            completion_tokens_total: status.total_completion_tokens,
            steps_executed: status.steps_executed,
            global_cache_hit_rate,
            global_cache_entries,
            ttft: None,
            speculative_acceptance_rate: telemetry
                .as_ref()
                .and_then(PrometheusTelemetry::acceptance_rate),
            active_memory_bytes: metal
                .and_then(|m| m.active_memory_gb)
                .map(gib_to_bytes)
                .transpose()?,
            peak_memory_bytes: metal
                .and_then(|m| m.peak_memory_gb)
                .map(gib_to_bytes)
                .transpose()?,
            cache_memory_bytes: metal
                .and_then(|m| m.cache_memory_gb)
                .map(gib_to_bytes)
                .transpose()?,
            cache_metrics,
            active_requests: status.requests,
            backend_details: Some(json!({
                "runtime_status": status.status,
                "progress": status.progress.and_then(recognized_progress),
                "telemetry": telemetry.as_ref().map(PrometheusTelemetry::as_json),
                "runtime_facts": runtime_facts,
            })),
        })
    }

    async fn optional_telemetry(&self) -> Option<PrometheusTelemetry> {
        // Bound the entire optional operation, including a stalled/chunked body.
        tokio::time::timeout(METRICS_TIMEOUT, async {
            let response = self
                .authenticated_get(&format!("{}/metrics", self.base_url))
                .timeout(METRICS_TIMEOUT)
                .send()
                .await?
                .error_for_status()?;
            parse_prometheus_response(response).await
        })
        .await
        .ok()?
        .ok()?
    }

    /// Liveness comes from the HTTP status alone. The optional body only feeds
    /// runtime facts, so a slow, oversized, or malformed body after a 2xx
    /// status can never turn a reachable server unhealthy.
    async fn probe_health(&self) -> (bool, Option<Value>) {
        let deadline = tokio::time::Instant::now() + HEALTH_TIMEOUT;
        let url = format!("{}/health", self.base_url);
        let response =
            match tokio::time::timeout_at(deadline, self.authenticated_get(&url).send()).await {
                Ok(Ok(response)) => response,
                _ => return (false, None),
            };
        if !response.status().is_success() {
            return (false, None);
        }
        let facts = tokio::time::timeout_at(
            deadline,
            parse_json_limited::<Value>(response, CACHE_BODY_LIMIT, "Rapid-MLX /health"),
        )
        .await
        .ok()
        .and_then(Result::ok)
        .map(|value| health_runtime_facts(&value));
        (true, facts)
    }

    fn authenticated_get(&self, url: &str) -> reqwest::RequestBuilder {
        match &self.api_key {
            Some(key) => self.client.get(url).bearer_auth(key),
            None => self.client.get(url),
        }
    }
}

/// Static launch facts from the resolved adapter: no IO, subprocess, or secrets.
/// Requested settings stay separate from live facts (never an effective lane).
pub fn launch_model_facts(adapter: &super::RapidMlxAdapter) -> Value {
    use super::model_resolver::{RapidMlxModelSource, RapidMlxModelSourceView};

    let mut facts = serde_json::Map::new();
    let view = RapidMlxModelSourceView::from_source(&adapter.resolved_model.original_input);
    for (key, value) in [
        ("repo_id", view.repo_id),
        ("revision", view.revision),
        ("served_model_name", adapter.served_model_name.clone()),
        ("version", Some(adapter.runtime.version.clone())),
    ] {
        if let Some(value) = bounded_fact_text(value.as_deref()) {
            facts.insert(key.into(), Value::String(value));
        }
    }
    let argument = &adapter.resolved_model.launch_argument;
    // A free-form launch alias is not a physical path.
    if matches!(
        &adapter.resolved_model.original_input,
        RapidMlxModelSource::Alias { .. }
    ) {
        if let Some(value) = bounded_fact_text(Some(argument)) {
            facts.insert("source_alias".into(), Value::String(value));
        }
    } else if let Some(value) = bounded_fact_text(Some(argument)) {
        // The snapshot is published to every dashboard client, so a local directory
        // must never leave this process. Keep only what identifies the model.
        match redacted_model_location(&value) {
            Some(RedactedModelLocation::HubRepo(repo)) => {
                facts
                    .entry("repo_id".to_string())
                    .or_insert(Value::String(repo));
            }
            Some(RedactedModelLocation::Directory(name)) => {
                // `./name` is a redacted path: nav.js `physicalModel` treats a
                // dot-prefixed value as a local path and shows its basename.
                facts.insert("model_path".into(), Value::String(format!("./{name}")));
            }
            None => {}
        }
    }
    if let Some(context) = adapter.context_length.filter(|value| *value > 0) {
        facts.insert("context_length".into(), json!(context));
    }
    if let Some(speculative) = adapter.speculative_config.as_ref() {
        let mut spec = serde_json::Map::new();
        spec.insert("method".into(), json!(speculative.method));
        if let Some(model) = bounded_fact_text(speculative.model.as_deref()) {
            spec.insert("model".into(), Value::String(model));
        }
        spec.insert(
            "num_speculative_tokens".into(),
            json!(speculative.num_speculative_tokens),
        );
        facts.insert("speculative_config".into(), Value::Object(spec));
    }
    Value::Object(facts)
}

/// Apply cached launch facts only while publishing a checked spawned target.
/// The caller must pass the Session held by the publication target guard.
pub fn enrich_launch_model_facts(
    snapshot: &mut InferenceMetricsSnapshot,
    session: &crate::state::Session,
    backend: Option<&crate::inference::backend::BackendAdapter>,
    base: &str,
) -> bool {
    // Clear stale launch facts even when the current target is now an attachment.
    if let Some(details) = snapshot
        .backend_details
        .as_mut()
        .and_then(Value::as_object_mut)
    {
        details.remove("launch_facts");
    }
    let crate::state::SessionMode::Spawn {
        port,
        bind_host,
        api_key,
    } = &session.mode
    else {
        return false;
    };
    let Some(crate::inference::backend::BackendAdapter::RapidMlx(adapter)) = backend else {
        return false;
    };
    let host = crate::web::api::upstream::local_connect_host(Some(&adapter.host));
    let session_host = crate::web::api::upstream::local_connect_host(bind_host.as_deref());
    let adapter_base = format!("http://{host}:{}", adapter.port);
    if snapshot.backend != InferenceBackend::RapidMlx
        || session.backend != InferenceBackend::RapidMlx
        || adapter.port != *port
        || host != session_host
        || adapter_base != base.trim_end_matches('/')
        || !crate::inference::llama_cpp::same_api_key(
            adapter.api_key.as_deref(),
            api_key.as_deref(),
        )
    {
        return false;
    }
    let details = snapshot.backend_details.get_or_insert_with(|| json!({}));
    if let Some(details) = details.as_object_mut() {
        details.insert("launch_facts".into(), launch_model_facts(adapter));
        true
    } else {
        false
    }
}

enum RedactedModelLocation {
    /// A Hugging Face hub snapshot path (`.../models--owner--name/snapshots/...`).
    HubRepo(String),
    /// Any other local directory, reduced to its final component.
    Directory(String),
}

fn redacted_model_location(argument: &str) -> Option<RedactedModelLocation> {
    let path = std::path::Path::new(argument);
    for component in path.components() {
        let part = component.as_os_str().to_string_lossy();
        if let Some(rest) = part.strip_prefix("models--")
            && let Some((owner, name)) = rest.split_once("--")
            && !owner.is_empty()
            && !name.is_empty()
        {
            let repo = format!("{owner}/{name}");
            return bounded_fact_text(Some(&repo)).map(RedactedModelLocation::HubRepo);
        }
    }
    let name = path.file_name()?.to_string_lossy();
    bounded_fact_text(Some(&name)).map(RedactedModelLocation::Directory)
}

fn bounded_fact_text(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|text| !text.is_empty() && text.len() <= 512 && !text.chars().any(char::is_control))
        .map(str::to_string)
}

fn health_runtime_facts(value: &Value) -> Value {
    let mut facts = serde_json::Map::new();
    for key in ["model_type", "engine_type"] {
        if let Some(text) = bounded_fact_text(value.get(key).and_then(Value::as_str)) {
            facts.insert(key.into(), Value::String(text));
        }
    }
    Value::Object(facts)
}

/// Privacy-safe aggregates only: labels (including model names) never escape.
/// Missing counters remain unavailable, whereas explicitly reported zeroes survive.
#[derive(Default, Debug)]
struct PrometheusTelemetry {
    succeeded_requests_total: Option<u64>,
    failed_requests_total: Option<u64>,
    cancelled_requests_total: Option<u64>,
    speculative_attempts_total: Option<u64>,
    speculative_accepts_total: Option<u64>,
}

impl PrometheusTelemetry {
    fn acceptance_rate(&self) -> Option<f64> {
        let attempts = self.speculative_attempts_total?;
        let accepts = self.speculative_accepts_total?;
        (attempts > 0 && accepts <= attempts).then(|| accepts as f64 / attempts as f64)
    }

    fn as_json(&self) -> Value {
        json!({
            "succeeded_requests_total": self.succeeded_requests_total,
            "failed_requests_total": self.failed_requests_total,
            "cancelled_requests_total": self.cancelled_requests_total,
            "speculative_attempts_total": self.speculative_attempts_total,
            "speculative_accepts_total": self.speculative_accepts_total,
            "speculative_acceptance_rate": self.acceptance_rate(),
        })
    }
}

async fn parse_prometheus_response(
    response: reqwest::Response,
) -> Result<Option<PrometheusTelemetry>> {
    if response
        .content_length()
        .is_some_and(|length| length > METRICS_BODY_LIMIT as u64)
    {
        bail!("Rapid-MLX /metrics response exceeded the telemetry limit");
    }
    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Failed reading Rapid-MLX /metrics response")?;
        if body.len().saturating_add(chunk.len()) > METRICS_BODY_LIMIT {
            bail!("Rapid-MLX /metrics response exceeded the telemetry limit");
        }
        body.extend_from_slice(&chunk);
    }
    parse_prometheus(std::str::from_utf8(&body).context("Invalid Rapid-MLX /metrics encoding")?)
}

fn parse_prometheus(body: &str) -> Result<Option<PrometheusTelemetry>> {
    let mut telemetry = PrometheusTelemetry::default();
    let mut seen = BTreeSet::new();
    let mut recognized = false;
    for line in body.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let name_end = line
            .find(|c: char| c == '{' || c.is_ascii_whitespace())
            .unwrap_or(line.len());
        let name = &line[..name_end];
        if !matches!(
            name,
            "rapid_mlx_model_requests_total"
                | "rapid_mlx_spec_decode_attempts_total"
                | "rapid_mlx_spec_decode_accepts_total"
        ) {
            continue;
        }
        // Reject bad recognized samples rather than publishing partial sums or a
        // skewed ratio. Unrelated gauges/histograms and future names are ignored.
        let (labels, value) = parse_prometheus_sample(&line[name_end..])?;
        if !seen.insert((name.to_string(), labels.clone())) {
            bail!("Duplicate Rapid-MLX /metrics counter series");
        }
        let target = match name {
            "rapid_mlx_model_requests_total" => match labels.get("outcome").map(String::as_str) {
                Some("succeeded") => &mut telemetry.succeeded_requests_total,
                Some("failed") => &mut telemetry.failed_requests_total,
                Some("cancelled") => &mut telemetry.cancelled_requests_total,
                _ => continue,
            },
            "rapid_mlx_spec_decode_attempts_total" => &mut telemetry.speculative_attempts_total,
            "rapid_mlx_spec_decode_accepts_total" => &mut telemetry.speculative_accepts_total,
            _ => unreachable!(),
        };
        *target = Some(
            target
                .unwrap_or(0)
                .checked_add(value)
                .context("Rapid-MLX /metrics counter aggregate overflow")?,
        );
        recognized = true;
    }
    Ok(recognized.then_some(telemetry))
}

fn parse_prometheus_sample(mut input: &str) -> Result<(BTreeMap<String, String>, u64)> {
    let mut labels = BTreeMap::new();
    if let Some(rest) = input.strip_prefix('{') {
        input = rest.trim_start();
        loop {
            if let Some(rest) = input.strip_prefix('}') {
                input = rest;
                break;
            }
            let (key, rest) = input
                .split_once('=')
                .context("Malformed Rapid-MLX /metrics labels")?;
            let key = key.trim();
            if key.is_empty()
                || !key.bytes().enumerate().all(|(i, c)| {
                    c == b'_' || c.is_ascii_alphabetic() || (i > 0 && c.is_ascii_digit())
                })
            {
                bail!("Invalid Rapid-MLX /metrics label name");
            }
            let (value, rest) = parse_prometheus_label_value(rest.trim_start())?;
            if labels.insert(key.to_string(), value).is_some() {
                bail!("Duplicate Rapid-MLX /metrics label");
            }
            input = rest.trim_start();
            if let Some(rest) = input.strip_prefix(',') {
                input = rest.trim_start();
            } else if !input.starts_with('}') {
                bail!("Malformed Rapid-MLX /metrics label separator");
            }
        }
    }
    if !input.starts_with(char::is_whitespace) {
        bail!("Missing Rapid-MLX /metrics sample separator");
    }
    let mut fields = input.split_whitespace();
    let value = fields
        .next()
        .context("Missing Rapid-MLX /metrics counter")?;
    let counter = parse_prometheus_counter(value)?;
    if let Some(timestamp) = fields.next() {
        timestamp
            .parse::<i64>()
            .context("Invalid Rapid-MLX /metrics timestamp")?;
    }
    if fields.next().is_some() {
        bail!("Unexpected Rapid-MLX /metrics sample fields");
    }
    Ok((labels, counter))
}

fn parse_prometheus_label_value(input: &str) -> Result<(String, &str)> {
    let input = input
        .strip_prefix('"')
        .context("Unquoted Rapid-MLX /metrics label")?;
    let mut value = String::new();
    let mut escaped = false;
    for (offset, c) in input.char_indices() {
        if escaped {
            value.push(match c {
                '\\' => '\\',
                '"' => '"',
                'n' => '\n',
                _ => bail!("Invalid Rapid-MLX /metrics label escape"),
            });
            escaped = false;
        } else {
            match c {
                '\\' => escaped = true,
                '"' => return Ok((value, &input[offset + 1..])),
                c if c.is_control() => bail!("Invalid Rapid-MLX /metrics label character"),
                c => value.push(c),
            }
        }
    }
    bail!("Unterminated Rapid-MLX /metrics label")
}

fn parse_prometheus_counter(value: &str) -> Result<u64> {
    // Counters are integral, but Prometheus also permits decimal/scientific
    // notation. Parse exactly: f64 would round large integers/fractions or
    // underflow tiny nonzero/negative values into a plausible counter.
    if let Ok(value) = value.parse::<u64>() {
        return Ok(value);
    }
    let value = value.strip_prefix('+').unwrap_or(value);
    let (mantissa, exponent) = match value.find(['e', 'E']) {
        Some(offset) => (
            &value[..offset],
            value[offset + 1..]
                .parse::<i32>()
                .context("Invalid Rapid-MLX /metrics counter exponent")?,
        ),
        None => (value, 0),
    };
    let mut digits = String::new();
    let mut decimal = false;
    let mut fractional_digits = 0_i64;
    for c in mantissa.bytes() {
        if c == b'.' && !decimal {
            decimal = true;
        } else if c.is_ascii_digit() {
            digits.push(c as char);
            fractional_digits += i64::from(decimal);
        } else {
            bail!("Invalid Rapid-MLX /metrics counter");
        }
    }
    if digits.is_empty() {
        bail!("Missing Rapid-MLX /metrics counter digits");
    }
    let digits = digits.trim_start_matches('0');
    if digits.is_empty() {
        return Ok(0);
    }
    let scale = i64::from(exponent) - fractional_digits;
    let integer_digits = if scale < 0 {
        let removed = usize::try_from(-scale).context("Invalid Rapid-MLX counter scale")?;
        if removed >= digits.len() || !digits[digits.len() - removed..].bytes().all(|c| c == b'0') {
            bail!("Fractional Rapid-MLX /metrics counter");
        }
        &digits[..digits.len() - removed]
    } else {
        digits
    };
    if integer_digits.len() as i64 + scale.max(0) > 20 {
        bail!("Rapid-MLX /metrics counter overflow");
    }
    let mut counter = integer_digits
        .parse::<u64>()
        .context("Rapid-MLX /metrics counter overflow")?;
    for _ in 0..scale.max(0) {
        counter = counter
            .checked_mul(10)
            .context("Rapid-MLX /metrics counter overflow")?;
    }
    Ok(counter)
}

async fn parse_json_limited<T: DeserializeOwned>(
    response: reqwest::Response,
    limit: usize,
    endpoint: &str,
) -> Result<T> {
    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| format!("Failed reading {endpoint} response"))?;
        if body.len().saturating_add(chunk.len()) > limit {
            bail!("{endpoint} response exceeded the {limit}-byte telemetry limit");
        }
        body.extend_from_slice(&chunk);
    }
    serde_json::from_slice(&body)
        .with_context(|| format!("Failed to parse required {endpoint} telemetry"))
}

fn validate_status(status: &StatusResponse) -> Result<()> {
    if status
        .status
        .as_ref()
        .is_some_and(|value| value.trim().is_empty() || value.len() > 64)
    {
        bail!("Rapid-MLX /v1/status contains an invalid status value");
    }
    for (name, value) in [
        ("uptime_s", status.uptime_s),
        ("generation_tps", status.generation_tps),
        ("prompt_tps", status.prompt_tps),
        (
            "metal.active_memory_gb",
            status.metal.as_ref().and_then(|m| m.active_memory_gb),
        ),
        (
            "metal.peak_memory_gb",
            status.metal.as_ref().and_then(|m| m.peak_memory_gb),
        ),
        (
            "metal.cache_memory_gb",
            status.metal.as_ref().and_then(|m| m.cache_memory_gb),
        ),
    ] {
        if value.is_some_and(|number| !number.is_finite() || number < 0.0) {
            bail!("Rapid-MLX /v1/status contains invalid numeric telemetry in {name}");
        }
    }
    Ok(())
}

fn gib_to_bytes(gib: f64) -> Result<u64> {
    let bytes = gib * GIB_BYTES;
    if !bytes.is_finite() || bytes < 0.0 || bytes > u64::MAX as f64 {
        return Err(anyhow!("Rapid-MLX Metal memory telemetry overflows bytes"));
    }
    Ok(bytes as u64)
}

fn normalized_status_cache(value: &Value) -> Option<Value> {
    let object = value.as_object()?;
    if object.get("enabled") == Some(&Value::Bool(false)) {
        return None;
    }
    let mut normalized = Map::new();
    for key in ["hits", "misses", "current_memory_bytes", "entry_count"] {
        if let Some(number) = object.get(key).and_then(Value::as_u64) {
            normalized.insert(key.to_string(), Value::from(number));
        }
    }
    if let Some(rate) = object
        .get("hit_rate")
        .and_then(Value::as_f64)
        .filter(|rate| rate.is_finite() && (0.0..=1.0).contains(rate))
    {
        normalized.insert("hit_rate".to_string(), Value::from(rate));
    }
    (!normalized.is_empty()).then_some(Value::Object(normalized))
}

fn recognized_endpoint_cache(value: Value) -> Option<Value> {
    let object = value.as_object()?;
    if object.contains_key("message") && object.contains_key("model_type") {
        return None;
    }
    let recognized = [
        "multimodal_kv_cache",
        "pixel_values_cache",
        "pil_image_cache",
    ];
    let kinds: Vec<Value> = recognized
        .iter()
        .filter(|key| object.get(**key).is_some_and(Value::is_object))
        .map(|key| Value::from(*key))
        .collect();
    (!kinds.is_empty()).then_some(json!({ "multimodal_cache_kinds": kinds }))
}

fn merge_cache_metrics(status: Option<Value>, endpoint: Option<Value>) -> Option<Value> {
    let mut merged = Map::new();
    for value in [status, endpoint].into_iter().flatten() {
        if let Some(object) = value.as_object() {
            merged.extend(object.clone());
        }
    }
    (!merged.is_empty()).then_some(Value::Object(merged))
}

fn sanitize_requests(requests: Vec<Value>) -> Vec<Value> {
    requests
        .into_iter()
        .take(ACTIVE_REQUEST_LIMIT)
        .filter_map(|request| {
            let object = request.as_object()?;
            let mut sanitized = Map::new();
            for (key, max_len) in [
                ("id", 256),
                ("request_id", 256),
                ("status", 64),
                ("phase", 64),
                ("cache_hit_type", 64),
            ] {
                if let Some(value) = object
                    .get(key)
                    .and_then(Value::as_str)
                    .filter(|value| !value.is_empty() && value.len() <= max_len)
                {
                    sanitized.insert(key.to_string(), Value::from(value));
                }
            }
            for key in [
                "prompt_tokens",
                "completion_tokens",
                "max_tokens",
                "cached_tokens",
            ] {
                if let Some(value) = object.get(key).and_then(Value::as_u64) {
                    sanitized.insert(key.to_string(), Value::from(value));
                }
            }
            for key in ["tokens_per_second", "ttft_s", "elapsed_s"] {
                if let Some(value) = object
                    .get(key)
                    .and_then(Value::as_f64)
                    .filter(|value| value.is_finite() && *value >= 0.0)
                {
                    sanitized.insert(key.to_string(), Value::from(value));
                }
            }
            (!sanitized.is_empty()).then_some(Value::Object(sanitized))
        })
        .collect()
}

fn recognized_progress(value: Value) -> Option<Value> {
    if value
        .as_f64()
        .is_some_and(|number| (0.0..=1.0).contains(&number))
    {
        return Some(value);
    }
    let object = value.as_object()?;
    let current = object.get("current").and_then(Value::as_f64)?;
    let total = object.get("total").and_then(Value::as_f64)?;
    (current.is_finite() && total.is_finite() && current >= 0.0 && total > 0.0 && current <= total)
        .then(|| json!({ "current": current, "total": total }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_facts_preserve_repo_and_served_alias_without_secrets_or_effective_lane() {
        use crate::inference::rapid_mlx::model_resolver::{
            RapidMlxModelSource, ResolvedRapidMlxLaunchModel,
        };
        let mut model = ResolvedRapidMlxLaunchModel::validated_alias("catalog-alias").unwrap();
        model.launch_argument = "/private/hub/models--owner--real-model/snapshots/abc123".into();
        model.original_input = RapidMlxModelSource::HuggingFaceRepo {
            repo_id: "owner/real-model".into(),
            revision: "abc123".into(),
        };
        let mut adapter = super::super::RapidMlxAdapter::from_resolved(
            super::super::runtime::RuntimeMetadata {
                version: "0.10.0".into(),
                ..Default::default()
            },
            model,
        );
        adapter.served_model_name = Some("friendly-alias".into());
        adapter.context_length = Some(32768);
        adapter.mllm_vision = Some("auto".into());
        let facts = launch_model_facts(&adapter);
        assert_eq!(facts["repo_id"], "owner/real-model");
        assert_eq!(facts["served_model_name"], "friendly-alias");
        assert_eq!(facts["version"], "0.10.0");
        assert_eq!(facts["context_length"], 32768);
        assert!(facts.get("runtime_lane").is_none());
        assert!(facts.get("api_key").is_none());
        assert!(facts.get("speculative_config").is_none());
        // The published snapshot never carries a local directory.
        assert!(!facts.to_string().contains("/private"), "{facts}");
        assert!(facts.get("model_path").is_none());
    }

    #[test]
    fn alias_launch_facts_do_not_claim_a_physical_repo_or_default_version() {
        let adapter = super::super::RapidMlxAdapter::from_resolved(
            super::super::runtime::RuntimeMetadata::default(),
            super::super::model_resolver::ResolvedRapidMlxLaunchModel::validated_alias(
                "catalog-alias",
            )
            .unwrap(),
        );
        let facts = launch_model_facts(&adapter);
        assert_eq!(facts["source_alias"], "catalog-alias");
        assert!(facts.get("repo_id").is_none());
        assert!(facts.get("model_path").is_none());
        assert!(facts.get("version").is_none());
        assert!(facts.get("context_length").is_none());
    }

    #[tokio::test]
    async fn status_redirect_is_not_followed() {
        let mut server = mockito::Server::new_async().await;
        let _health = server
            .mock("GET", "/health")
            .with_status(200)
            .with_body("{}")
            .create_async()
            .await;
        let _status = server
            .mock("GET", "/v1/status")
            .with_status(302)
            .with_header("location", "/redirected-status")
            .create_async()
            .await;
        let target = server
            .mock("GET", "/redirected-status")
            .with_status(200)
            .with_body(r#"{"status":"idle"}"#)
            .expect(0)
            .create_async()
            .await;
        let result = RapidMlxPoller::from_base_url(server.url(), Some("secret"))
            .poll()
            .await;
        assert!(
            result.is_err(),
            "a redirected status endpoint must fail the poll"
        );
        target.assert_async().await;
    }

    fn adapter_for_source(
        original_input: super::super::model_resolver::RapidMlxModelSource,
        launch_argument: &str,
    ) -> super::super::RapidMlxAdapter {
        use crate::inference::rapid_mlx::model_resolver::ResolvedRapidMlxLaunchModel;
        let mut model = ResolvedRapidMlxLaunchModel::validated_alias("placeholder").unwrap();
        model.launch_argument = launch_argument.into();
        model.original_input = original_input;
        super::super::RapidMlxAdapter::from_resolved(
            super::super::runtime::RuntimeMetadata::default(),
            model,
        )
    }

    #[test]
    fn local_directory_launch_fact_is_a_basename_not_a_path() {
        use crate::inference::rapid_mlx::model_resolver::RapidMlxModelSource;
        let dir = "/Users/someone/private-models/My-MLX-4bit";
        let adapter =
            adapter_for_source(RapidMlxModelSource::MlxDirectory { path: dir.into() }, dir);
        let facts = launch_model_facts(&adapter);
        let text = facts.to_string();
        assert!(
            !text.contains("/Users") && !text.contains("private-models"),
            "{text}"
        );
        // `./name` is a redacted path: nav.js `physicalModel` still resolves its
        // basename, so the dashboard identity keeps working without the directory.
        assert_eq!(facts["model_path"], "./My-MLX-4bit");
    }

    #[test]
    fn hub_snapshot_launch_fact_reduces_to_repo_id() {
        use crate::inference::rapid_mlx::model_resolver::RapidMlxModelSource;
        let dir = "/Users/someone/cache/models--owner--real-model/snapshots/abc123";
        let adapter =
            adapter_for_source(RapidMlxModelSource::MlxDirectory { path: dir.into() }, dir);
        let facts = launch_model_facts(&adapter);
        let text = facts.to_string();
        assert!(
            !text.contains("/Users") && !text.contains("snapshots"),
            "{text}"
        );
        assert_eq!(facts["repo_id"], "owner/real-model");
        assert!(facts.get("model_path").is_none());
    }

    #[test]
    fn unrepresentable_launch_path_is_dropped() {
        use crate::inference::rapid_mlx::model_resolver::RapidMlxModelSource;
        let adapter =
            adapter_for_source(RapidMlxModelSource::MlxDirectory { path: "/".into() }, "/");
        let facts = launch_model_facts(&adapter);
        assert!(facts.get("model_path").is_none(), "{facts}");
    }

    #[test]
    fn health_runtime_facts_are_bounded_and_never_forward_mcp_or_secrets() {
        assert_eq!(
            health_runtime_facts(&json!({
                "model_type": "llm", "engine_type": "batched",
                "model_name": "served-alias", "api_key": "secret",
                "mcp": {"tools_available": 8},
            })),
            json!({"model_type": "llm", "engine_type": "batched"})
        );
        assert_eq!(
            health_runtime_facts(&json!({"engine_type": "x".repeat(513)})),
            json!({})
        );
    }

    #[tokio::test]
    async fn health_status_200_is_healthy_regardless_of_body() {
        use tokio::io::AsyncWriteExt;

        for body in [String::from("not json"), "x".repeat(CACHE_BODY_LIMIT + 1)] {
            let mut server = mockito::Server::new_async().await;
            let _health = server
                .mock("GET", "/health")
                .with_status(200)
                .with_body(body)
                .create_async()
                .await;
            let (ok, facts) = RapidMlxPoller::from_base_url(server.url(), None)
                .probe_health()
                .await;
            assert!(ok);
            assert!(facts.is_none());
        }

        // Headers arrive, then the body stalls past the 2s budget.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 1024];
            let _ = tokio::io::AsyncReadExt::read(&mut stream, &mut buf).await;
            stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 100\r\n\r\n{").await.unwrap();
            tokio::time::sleep(Duration::from_secs(4)).await;
        });
        let (ok, facts) = RapidMlxPoller::from_base_url(base, None)
            .probe_health()
            .await;
        server.abort();
        assert!(ok, "a 200 status must stay healthy when the body stalls");
        assert!(facts.is_none());

        let mut server = mockito::Server::new_async().await;
        let _health = server
            .mock("GET", "/health")
            .with_status(503)
            .create_async()
            .await;
        assert!(
            !RapidMlxPoller::from_base_url(server.url(), None)
                .probe_health()
                .await
                .0
        );
    }

    #[tokio::test]
    async fn snapshot_launch_enrichment_is_spawn_target_scoped_and_preserves_live_facts() {
        use crate::inference::backend::BackendAdapter;
        use crate::state::{Session, SessionMode};
        use std::sync::Arc;

        let mut server = mockito::Server::new_async().await;
        let _health = server
            .mock("GET", "/health")
            .with_status(200)
            .with_body(r#"{"model_type":"llm","engine_type":"batched"}"#)
            .create_async()
            .await;
        let _status = server
            .mock("GET", "/v1/status")
            .with_status(200)
            .with_body(STATUS_FIXTURE)
            .create_async()
            .await;
        let _cache = server
            .mock("GET", "/v1/cache/stats")
            .with_status(404)
            .create_async()
            .await;
        let _metrics = server
            .mock("GET", "/metrics")
            .with_status(404)
            .create_async()
            .await;
        let mut snapshot = RapidMlxPoller::from_base_url(server.url(), None)
            .poll()
            .await
            .unwrap();
        let endpoint = reqwest::Url::parse(&server.url()).unwrap();
        let mut adapter = super::super::RapidMlxAdapter::from_resolved(
            super::super::runtime::RuntimeMetadata {
                version: "0.10.0".into(),
                ..Default::default()
            },
            super::super::model_resolver::ResolvedRapidMlxLaunchModel::validated_alias(
                "catalog-alias",
            )
            .unwrap(),
        );
        adapter.host = endpoint.host_str().unwrap().into();
        adapter.port = endpoint.port().unwrap();
        let backend = BackendAdapter::RapidMlx(Arc::new(adapter));
        let mut session: Session =
            serde_json::from_value(json!({"id":"s", "backend":"rapid_mlx"})).unwrap();
        session.mode = SessionMode::Spawn {
            port: endpoint.port().unwrap(),
            bind_host: Some(endpoint.host_str().unwrap().into()),
            api_key: None,
        };
        assert!(enrich_launch_model_facts(
            &mut snapshot,
            &session,
            Some(&backend),
            &server.url()
        ));
        let details = snapshot.backend_details.as_ref().unwrap();
        assert_eq!(details["launch_facts"]["version"], "0.10.0");
        assert_eq!(details["runtime_facts"]["engine_type"], "batched");
        assert_eq!(details["runtime_status"], "generating");
        assert_eq!(snapshot.model.as_deref(), Some("fixture"));
        assert!(!enrich_launch_model_facts(
            &mut snapshot,
            &session,
            Some(&backend),
            "http://127.0.0.1:1"
        ));
        assert!(
            snapshot
                .backend_details
                .as_ref()
                .unwrap()
                .get("launch_facts")
                .is_none()
        );
        session.mode = SessionMode::Attach {
            endpoint: server.url(),
            api_key: None,
        };
        assert!(!enrich_launch_model_facts(
            &mut snapshot,
            &session,
            Some(&backend),
            &server.url()
        ));
        assert!(
            snapshot
                .backend_details
                .as_ref()
                .unwrap()
                .get("launch_facts")
                .is_none()
        );
    }

    const PROMETHEUS_FIXTURE: &str = r#"
# HELP rapid_mlx_model_requests_total Requests by outcome.
# TYPE rapid_mlx_model_requests_total counter
rapid_mlx_model_requests_total{model="one",outcome="succeeded"} 3
rapid_mlx_model_requests_total{outcome="succeeded",model="two } space, \"quote\" \\slash\nline"} 7.0 1700000000000
rapid_mlx_model_requests_total{model="one",outcome="failed"} 2e0
rapid_mlx_model_requests_total{model="one",outcome="cancelled"} 0
rapid_mlx_model_requests_total{model="two",outcome="cancelled"} 1
rapid_mlx_model_requests_total{model="one",outcome="queued"} 99
rapid_mlx_requests_processed_total 999
rapid_mlx_requests_cancelled_via_disconnect_total 999
rapid_mlx_spec_decode_attempts_total{family="gemma4",method="mtp"} 10
rapid_mlx_spec_decode_accepts_total{family="gemma4",method="mtp"} 9
rapid_mlx_spec_decode_attempts_total{family="other",method="draft"} 90
rapid_mlx_spec_decode_accepts_total{family="other",method="draft"} 1
rapid_mlx_spec_decode_accept_ratio{family="gemma4",method="mtp"} 0.9
rapid_mlx_spec_decode_accept_ratio{family="other",method="draft"} 0.0111
future_unknown_gauge NaN
"#;

    const STATUS_FIXTURE: &str = r#"{
        "status":"generating","model":"fixture","uptime_s":12.5,
        "steps_executed":7,"num_running":1,"num_waiting":2,
        "total_requests_processed":42,"total_prompt_tokens":100,
        "total_completion_tokens":77,"generation_tps":3.5,"prompt_tps":2.5,
        "metal":{"active_memory_gb":1.0,"peak_memory_gb":2.0,"cache_memory_gb":0.25},
        "cache":{"enabled":true,"hits":3,"misses":1,"hit_rate":0.75,
                 "entry_count":2,"current_memory_bytes":256},
        "requests":[{"id":"r1","status":"running","phase":"decode","prompt_tokens":12,
                     "completion_tokens":3,"max_tokens":64,"tokens_per_second":3.5,
                     "ttft_s":0.2,"elapsed_s":1.0,"cached_tokens":4,
                     "cache_hit_type":"prefix","prompt":"not public"}],
        "progress":{"current":3,"total":4}
    }"#;

    fn assert_status_fields_preserved(snapshot: &InferenceMetricsSnapshot) {
        assert!(matches!(snapshot.health, Some(HealthState::Ok)));
        assert_eq!(snapshot.ready, Some(true));
        assert_eq!(snapshot.model.as_deref(), Some("fixture"));
        assert_eq!(snapshot.uptime_seconds, Some(12.5));
        assert_eq!(snapshot.steps_executed, Some(7));
        assert_eq!(snapshot.generation_tokens_per_second, Some(3.5));
        assert_eq!(snapshot.prompt_tokens_per_second, Some(2.5));
        assert_eq!(snapshot.running_requests, Some(1));
        assert_eq!(snapshot.waiting_requests, Some(2));
        assert_eq!(snapshot.completed_requests_total, Some(42));
        assert_eq!(snapshot.prompt_tokens_total, Some(100));
        assert_eq!(snapshot.completion_tokens_total, Some(77));
        assert_eq!(snapshot.active_memory_bytes, Some(1_073_741_824));
        assert_eq!(snapshot.peak_memory_bytes, Some(2_147_483_648));
        assert_eq!(snapshot.cache_memory_bytes, Some(268_435_456));
        assert_eq!(snapshot.ttft, None);
        assert_eq!(snapshot.global_cache_hit_rate, Some(0.75));
        assert_eq!(snapshot.global_cache_entries, Some(2));
        assert_eq!(
            snapshot.cache_metrics,
            Some(json!({
                "hits":3,"misses":1,"hit_rate":0.75,"entry_count":2,
                "current_memory_bytes":256,"multimodal_cache_kinds":["multimodal_kv_cache"]
            }))
        );
        assert_eq!(
            snapshot.active_requests,
            Some(vec![json!({
                "id":"r1","status":"running","phase":"decode","prompt_tokens":12,
                "completion_tokens":3,"max_tokens":64,"tokens_per_second":3.5,
                "ttft_s":0.2,"elapsed_s":1.0,"cached_tokens":4,"cache_hit_type":"prefix",
            })])
        );
        let details = snapshot.backend_details.as_ref().unwrap();
        assert_eq!(details["runtime_status"], "generating");
        assert_eq!(details["progress"], json!({"current":3.0,"total":4.0}));
    }

    #[test]
    fn prometheus_aggregates_outcomes_and_weights_speculative_acceptance() {
        let telemetry = parse_prometheus(PROMETHEUS_FIXTURE).unwrap().unwrap();
        assert_eq!(
            telemetry.as_json(),
            json!({
                "succeeded_requests_total": 10,
                "failed_requests_total": 2,
                "cancelled_requests_total": 1,
                "speculative_attempts_total": 100,
                "speculative_accepts_total": 10,
                "speculative_acceptance_rate": 0.1,
            })
        );
        assert_eq!(telemetry.acceptance_rate(), Some(0.1));
        assert!(!telemetry.as_json().to_string().contains("model"));
        assert!(!telemetry.as_json().to_string().contains("gemma4"));
    }

    #[test]
    fn prometheus_preserves_missing_vs_zero_and_ignores_ratio_gauges() {
        for body in [
            "",
            "# TYPE future_metric gauge\nfuture_metric 0",
            "rapid_mlx_requests_processed_total 10",
            "rapid_mlx_spec_decode_accept_ratio{family=\"gemma4\",method=\"mtp\"} 0.8",
            "rapid_mlx_model_requests_total{outcome=\"unknown\"} 4",
        ] {
            assert!(parse_prometheus(body).unwrap().is_none(), "{body}");
        }
        let telemetry = parse_prometheus("rapid_mlx_model_requests_total{outcome=\"succeeded\"} 0")
            .unwrap()
            .unwrap();
        assert_eq!(telemetry.succeeded_requests_total, Some(0));
        assert_eq!(telemetry.failed_requests_total, None);
        assert_eq!(telemetry.acceptance_rate(), None);

        for body in [
            "rapid_mlx_spec_decode_attempts_total 0\nrapid_mlx_spec_decode_accepts_total 0",
            "rapid_mlx_spec_decode_attempts_total 10",
            "rapid_mlx_spec_decode_accepts_total 1",
            "rapid_mlx_spec_decode_attempts_total 1\nrapid_mlx_spec_decode_accepts_total 2",
        ] {
            assert_eq!(
                parse_prometheus(body).unwrap().unwrap().acceptance_rate(),
                None
            );
        }
        assert_eq!(
            parse_prometheus(
                "rapid_mlx_spec_decode_attempts_total 10\nrapid_mlx_spec_decode_accepts_total 0"
            )
            .unwrap()
            .unwrap()
            .acceptance_rate(),
            Some(0.0)
        );
    }

    #[test]
    fn prometheus_counter_notation_is_exact_and_overflow_checked() {
        for (text, expected) in [
            ("0", 0),
            ("12", 12),
            ("+12.00", 12),
            (".0", 0),
            ("1.2e1", 12),
            ("120e-1", 12),
            ("0e999", 0),
            ("9007199254740993", 9_007_199_254_740_993),
            ("9007199254740993.0", 9_007_199_254_740_993),
            ("18446744073709551615.0", u64::MAX),
        ] {
            assert_eq!(parse_prometheus_counter(text).unwrap(), expected, "{text}");
        }
        for text in [
            "",
            ".",
            "NaN",
            "+Inf",
            "-Inf",
            "-1",
            "-0.0",
            "0.1",
            "1e-400",
            "-1e-400",
            "0.99999999999999999999",
            "18446744073709551616",
            "18446744073709551616.0",
            "1e100",
            "1e999999999999",
            "1.2.0",
            "1e1e1",
        ] {
            assert!(parse_prometheus_counter(text).is_err(), "{text}");
        }
        let overflow = format!(
            "rapid_mlx_spec_decode_attempts_total{{family=\"a\"}} {}\n\
             rapid_mlx_spec_decode_attempts_total{{family=\"b\"}} 1",
            u64::MAX
        );
        assert!(parse_prometheus(&overflow).is_err());
    }

    #[test]
    fn prometheus_rejects_malformed_or_duplicate_recognized_series() {
        for sample in [
            "",
            "{outcome=succeeded} 1",
            "{outcome=\"succeeded\" 1",
            "{outcome=\"succeeded\",outcome=\"failed\"} 1",
            "{outcome=\"succeeded\",model=\"bad\\t\"} 1",
            "{outcome=\"succeeded\" model=\"two\"} 1",
            "{9bad=\"x\"} 1",
            "{outcome=\"succeeded\"}NaN",
            "{outcome=\"succeeded\"} -1",
            "{outcome=\"succeeded\"} 1 not-a-timestamp",
            "{outcome=\"succeeded\"} 1 123 extra",
        ] {
            let body = format!("rapid_mlx_model_requests_total{sample}");
            assert!(parse_prometheus(&body).is_err(), "{body}");
        }
        assert!(
            parse_prometheus(
                "rapid_mlx_spec_decode_attempts_total{family=\"a\",method=\"mtp\"} 1\n\
             rapid_mlx_spec_decode_attempts_total{method=\"mtp\",family=\"a\"} 2"
            )
            .is_err()
        );
    }

    #[test]
    fn active_requests_keep_only_bounded_valid_dashboard_fields() {
        let expected = json!({
            "id": "request-1",
            "status": "running",
            "phase": "decode",
            "prompt_tokens": 12,
            "completion_tokens": 0,
            "max_tokens": 64,
            "cached_tokens": 4,
            "tokens_per_second": 3.5,
            "ttft_s": 0.2,
            "elapsed_s": 0.0,
            "cache_hit_type": "prefix",
        });
        let mut request = expected.clone();
        request["prompt"] = json!("must not escape");
        request["future_field"] = json!({"opaque": true});
        assert_eq!(sanitize_requests(vec![request]), vec![expected]);

        let invalid = json!({
            "id": "x".repeat(257),
            "phase": "x".repeat(65),
            "cache_hit_type": "",
            "prompt_tokens": -1,
            "completion_tokens": 1.5,
            "max_tokens": "64",
            "cached_tokens": -1,
            "tokens_per_second": -0.1,
            "ttft_s": "NaN",
            "elapsed_s": -2.0,
        });
        assert!(sanitize_requests(vec![invalid, json!(null)]).is_empty());
        assert_eq!(
            sanitize_requests(vec![json!({"status": "running"}); ACTIVE_REQUEST_LIMIT + 1]).len(),
            ACTIVE_REQUEST_LIMIT
        );
    }

    #[tokio::test]
    async fn one_poller_reuses_client_across_all_authenticated_telemetry_calls() {
        let mut server = mockito::Server::new_with_opts_async(mockito::ServerOpts {
            host: "127.0.0.1",
            ..Default::default()
        })
        .await;
        let health = server
            .mock("GET", "/health")
            .match_header("authorization", "Bearer secret")
            .with_status(200)
            .expect(2)
            .create_async()
            .await;
        let status = server
            .mock("GET", "/v1/status")
            .match_header("authorization", "Bearer secret")
            .with_status(200)
            .with_body(STATUS_FIXTURE)
            .expect(2)
            .create_async()
            .await;
        let cache = server
            .mock("GET", "/v1/cache/stats")
            .match_header("authorization", "Bearer secret")
            .with_status(200)
            .with_body(r#"{"multimodal_kv_cache":{}}"#)
            .expect(2)
            .create_async()
            .await;
        let metrics = server
            .mock("GET", "/metrics")
            .match_header("authorization", "Bearer secret")
            .with_status(200)
            .with_body(PROMETHEUS_FIXTURE)
            .expect(2)
            .create_async()
            .await;

        let poller = RapidMlxPoller::from_base_url(server.url(), Some("secret"));
        assert!(poller.matches_target(&server.url(), Some("secret")));
        assert!(!poller.matches_target(&server.url(), Some("wrong")));
        for _ in 0..2 {
            let snapshot = poller.poll().await.unwrap();
            assert_status_fields_preserved(&snapshot);
            assert_eq!(snapshot.speculative_acceptance_rate, Some(0.1));
            assert_eq!(
                snapshot.backend_details.as_ref().unwrap()["telemetry"],
                parse_prometheus(PROMETHEUS_FIXTURE)
                    .unwrap()
                    .unwrap()
                    .as_json()
            );
        }
        health.assert_async().await;
        status.assert_async().await;
        cache.assert_async().await;
        metrics.assert_async().await;
    }

    #[tokio::test]
    async fn optional_metrics_failures_never_invalidate_required_status() {
        let bad_bodies = [
            (404, Vec::new()),
            (401, PROMETHEUS_FIXTURE.as_bytes().to_vec()),
            (500, PROMETHEUS_FIXTURE.as_bytes().to_vec()),
            (200, vec![b'x'; METRICS_BODY_LIMIT + 1]),
            (200, vec![0xff]),
            (200, b"rapid_mlx_spec_decode_attempts_total NaN".to_vec()),
            (
                200,
                b"rapid_mlx_spec_decode_attempts_total{broken} 1".to_vec(),
            ),
            (200, b"<html>not metrics</html>".to_vec()),
        ];
        for (code, body) in bad_bodies {
            let mut server = mockito::Server::new_with_opts_async(mockito::ServerOpts {
                host: "127.0.0.1",
                ..Default::default()
            })
            .await;
            let health = server
                .mock("GET", "/health")
                .with_status(200)
                .create_async()
                .await;
            let status = server
                .mock("GET", "/v1/status")
                .with_status(200)
                .with_body(STATUS_FIXTURE)
                .create_async()
                .await;
            let cache = server
                .mock("GET", "/v1/cache/stats")
                .with_status(200)
                .with_body(r#"{"multimodal_kv_cache":{}}"#)
                .create_async()
                .await;
            let metrics = server
                .mock("GET", "/metrics")
                .with_status(code)
                .with_body(body)
                .create_async()
                .await;
            let snapshot = RapidMlxPoller::from_base_url(server.url(), None)
                .poll()
                .await
                .unwrap();
            assert_status_fields_preserved(&snapshot);
            assert_eq!(snapshot.speculative_acceptance_rate, None);
            assert!(snapshot.backend_details.as_ref().unwrap()["telemetry"].is_null());
            health.assert_async().await;
            status.assert_async().await;
            cache.assert_async().await;
            metrics.assert_async().await;
        }
    }

    #[tokio::test]
    async fn prometheus_stream_limit_is_enforced_without_content_length() {
        for size in [METRICS_BODY_LIMIT, METRICS_BODY_LIMIT + 1] {
            let mut server = mockito::Server::new_with_opts_async(mockito::ServerOpts {
                host: "127.0.0.1",
                ..Default::default()
            })
            .await;
            let metrics = server
                .mock("GET", "/metrics")
                .with_chunked_body(move |writer| {
                    writer.write_all(PROMETHEUS_FIXTURE.as_bytes())?;
                    writer.write_all(b"#")?;
                    let padding = vec![b'x'; size - PROMETHEUS_FIXTURE.len() - 1];
                    for chunk in padding.chunks(4096) {
                        writer.write_all(chunk)?;
                    }
                    Ok(())
                })
                .create_async()
                .await;
            let response = reqwest::Client::new()
                .get(format!("{}/metrics", server.url()))
                .send()
                .await
                .unwrap();
            assert_eq!(response.content_length(), None);
            let result = parse_prometheus_response(response).await;
            if size > METRICS_BODY_LIMIT {
                assert!(result.is_err());
            } else {
                assert_eq!(result.unwrap().unwrap().acceptance_rate(), Some(0.1));
            }
            metrics.assert_async().await;
        }
    }

    #[tokio::test]
    async fn optional_metrics_timeout_covers_a_stalled_streaming_body() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let stalled_server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 1024];
            let size = socket.read(&mut request).await.unwrap();
            assert!(request[..size].starts_with(b"GET /metrics "));
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n1\r\n#\r\n")
                .await
                .unwrap();
            // Keep the socket alive but never finish the body.
            std::future::pending::<()>().await;
        });
        let poller = RapidMlxPoller::from_base_url(format!("http://{address}"), None);
        let result = tokio::time::timeout(
            METRICS_TIMEOUT + Duration::from_secs(1),
            poller.optional_telemetry(),
        )
        .await;
        stalled_server.abort();
        assert!(
            result
                .expect("optional /metrics must remain bounded")
                .is_none()
        );
    }

    #[tokio::test]
    async fn optional_metrics_transport_errors_are_unavailable() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let broken_server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            drop(socket);
        });
        let poller = RapidMlxPoller::from_base_url(format!("http://{address}"), None);
        assert!(poller.optional_telemetry().await.is_none());
        broken_server.await.unwrap();
    }
}
