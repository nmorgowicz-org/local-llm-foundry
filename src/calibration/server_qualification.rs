//! Bounded, server-realistic qualification for a saved llama.cpp preset.
//!
//! The offline calibration path measures `llama-bench`; this module owns the
//! next, deliberately smaller boundary: start one calibration-owned loopback
//! server, exercise it with deterministic requests, and always clean it up.
//! It is intentionally independent of the app's active-server supervisor so a
//! qualification cannot steal or mutate a user's running session.

use anyhow::{Context, Result, bail};
use futures_util::{StreamExt, future::join_all};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::process::Stdio;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};
use tokio::{
    io::AsyncReadExt,
    process::{Child, Command},
    task::JoinHandle,
};

use crate::inference::supervisor::SupervisedLaunch;

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
// CUDA-backed llama-server startup can exceed 20 seconds on Windows while
// loading DLLs and initializing the GPU. Keep readiness bounded, but allow a
// cold native startup enough time to reach /health.
const HEALTH_TIMEOUT: Duration = Duration::from_secs(180);
const MAX_RESPONSE_BYTES: usize = 4 * 1024 * 1024;
const MAX_SERVER_LOG_BYTES: usize = 256 * 1024;
const MAX_REQUEST_TIMEOUT_MS: u64 = 5 * 60 * 1000;
const QUALIFICATION_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum QualificationTrack {
    #[default]
    LatencyMemory,
    ToolCorrectness,
    Mtp,
    Dflash,
    Ngram,
    Concurrency,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum QualificationStatus {
    #[default]
    Completed,
    Unsupported,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct QualificationRequest {
    pub tracks: BTreeSet<QualificationTrack>,
    pub parallel_requests: u32,
    pub allow_concurrency: bool,
    pub prompt: String,
    pub generation_tokens: u32,
    pub timeout_ms: u64,
    pub capability_evidence: BTreeSet<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct QualificationCapabilities {
    pub mtp: bool,
    pub dflash: bool,
    pub ngram: bool,
    pub evidence: BTreeSet<String>,
}

impl QualificationCapabilities {
    fn from_request(request: &QualificationRequest) -> Self {
        Self {
            mtp: request.capability_evidence.contains("mtp"),
            dflash: request.capability_evidence.contains("dflash"),
            ngram: request.capability_evidence.contains("ngram"),
            evidence: request.capability_evidence.clone(),
        }
    }

    fn supports(&self, track: QualificationTrack) -> bool {
        match track {
            QualificationTrack::Mtp => self.mtp,
            QualificationTrack::Dflash => self.dflash,
            QualificationTrack::Ngram => self.ngram,
            _ => true,
        }
    }
}

impl Default for QualificationRequest {
    fn default() -> Self {
        Self {
            tracks: BTreeSet::from([QualificationTrack::LatencyMemory]),
            parallel_requests: 1,
            allow_concurrency: false,
            prompt: "Reply with one short sentence describing a calibration check.".into(),
            generation_tokens: 256,
            timeout_ms: DEFAULT_TIMEOUT.as_millis() as u64,
            capability_evidence: BTreeSet::new(),
        }
    }
}

impl QualificationRequest {
    pub fn validate(&self) -> Result<()> {
        if self.parallel_requests == 0 {
            bail!("qualification parallel_requests must be at least 1");
        }
        if self.parallel_requests > 1
            && (!self.allow_concurrency || !self.tracks.contains(&QualificationTrack::Concurrency))
        {
            bail!("parallel qualification requires explicit concurrency opt-in");
        }
        if self.tracks.contains(&QualificationTrack::Mtp) && self.parallel_requests != 1 {
            bail!("MTP qualification requires --parallel 1");
        }
        if self.prompt.trim().is_empty() {
            bail!("qualification prompt must not be empty");
        }
        if self.generation_tokens == 0 || self.generation_tokens > 8192 {
            bail!("qualification generation_tokens must be in 1..=8192");
        }
        if self.timeout_ms == 0 || self.timeout_ms > MAX_REQUEST_TIMEOUT_MS {
            bail!("qualification timeout_ms must be in 1..={MAX_REQUEST_TIMEOUT_MS}");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct LatencyMemoryResult {
    pub status: QualificationStatus,
    pub time_to_first_token_ms: Option<f64>,
    pub total_time_ms: Option<f64>,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub prompt_tokens_per_second: Option<f64>,
    pub completion_tokens_per_second: Option<f64>,
    pub requests_completed: u32,
    pub requests_failed: u32,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct ToolCorrectnessResult {
    pub status: QualificationStatus,
    pub tool_call_observed: bool,
    pub structured_output_observed: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct QualificationTrackResult {
    pub track: QualificationTrack,
    pub status: QualificationStatus,
    pub latency: Option<LatencyMemoryResult>,
    pub tool: Option<ToolCorrectnessResult>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct QualificationReceipt {
    pub schema_version: u32,
    pub endpoint: String,
    pub parallel_requests: u32,
    pub tracks: Vec<QualificationTrackResult>,
    pub server_exit_code: Option<i32>,
    pub server_log_tail: Option<String>,
    pub memory: Option<MemoryTelemetry>,
    pub capability_evidence: BTreeSet<String>,
    pub baseline: Option<Box<QualificationReceipt>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct MemoryTelemetry {
    pub process_rss_before_bytes: Option<u64>,
    pub process_rss_peak_bytes: Option<u64>,
    pub process_rss_after_bytes: Option<u64>,
}

struct ManagedServer {
    child: Child,
    endpoint: String,
    log_task: JoinHandle<String>,
}

impl ManagedServer {
    async fn start(launch: SupervisedLaunch) -> Result<Self> {
        if launch.port == 0 {
            bail!("qualification server requires a non-zero isolated port");
        }
        let endpoint = format!("http://127.0.0.1:{}", launch.port);
        let mut command = Command::new(&launch.program);
        crate::platform::no_window_tokio(&mut command);
        command
            .args(&launch.args)
            .envs(launch.env)
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        if let Some(cwd) = launch.cwd {
            command.current_dir(cwd);
        }
        let mut child = command
            .spawn()
            .with_context(|| format!("start qualification server on {}", launch.port))?;
        let stderr = child.stderr.take();
        let log_task = tokio::spawn(async move {
            let Some(stderr) = stderr else {
                return String::new();
            };
            let mut bytes = Vec::new();
            let _ = stderr
                .take((MAX_SERVER_LOG_BYTES + 1) as u64)
                .read_to_end(&mut bytes)
                .await;
            if bytes.len() > MAX_SERVER_LOG_BYTES {
                bytes.truncate(MAX_SERVER_LOG_BYTES);
            }
            String::from_utf8_lossy(&bytes).into_owned()
        });
        let server = Self {
            child,
            endpoint,
            log_task,
        };
        if let Err(error) = server.wait_ready().await {
            let (_, server_log) = server.stop().await;
            if server_log.trim().is_empty() {
                return Err(error);
            }
            return Err(error.context(format!("qualification server stderr: {server_log}")));
        }
        Ok(server)
    }

    async fn wait_ready(&self) -> Result<()> {
        let client = Client::builder().timeout(Duration::from_secs(2)).build()?;
        let deadline = Instant::now() + HEALTH_TIMEOUT;
        loop {
            if Instant::now() >= deadline {
                bail!("qualification server health check timed out");
            }
            if let Ok(response) = client.get(format!("{}/health", self.endpoint)).send().await
                && response.status().is_success()
            {
                return Ok(());
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }

    async fn stop(mut self) -> (Option<i32>, String) {
        let _ = self.child.kill().await;
        let exit_code = self
            .child
            .wait()
            .await
            .ok()
            .and_then(|status| status.code());
        let log = self.log_task.await.unwrap_or_default();
        (exit_code, log)
    }
}

/// Run qualification against an already-managed test endpoint. This seam is
/// used by unit tests and keeps fake-server validation independent of process
/// launching.
pub async fn run_against_endpoint(
    endpoint: &str,
    request: &QualificationRequest,
) -> Result<QualificationReceipt> {
    run_against_endpoint_with_capabilities(
        endpoint,
        request,
        &QualificationCapabilities::from_request(request),
    )
    .await
}

pub async fn run_against_endpoint_with_capabilities(
    endpoint: &str,
    request: &QualificationRequest,
    capabilities: &QualificationCapabilities,
) -> Result<QualificationReceipt> {
    request.validate()?;
    let endpoint = endpoint.trim_end_matches('/').to_owned();
    let client = Client::builder()
        .timeout(Duration::from_millis(request.timeout_ms.max(1)))
        .build()?;
    let mut tracks = Vec::new();

    for track in &request.tracks {
        if !capabilities.supports(*track) {
            tracks.push(QualificationTrackResult {
                track: *track,
                status: QualificationStatus::Unsupported,
                error: Some("capability evidence is absent".into()),
                ..Default::default()
            });
            continue;
        }
        let result = match track {
            QualificationTrack::LatencyMemory => {
                let latency = measure_completion(&client, &endpoint, request).await;
                let status = if latency.status == QualificationStatus::Completed {
                    QualificationStatus::Completed
                } else {
                    QualificationStatus::Failed
                };
                QualificationTrackResult {
                    track: *track,
                    status,
                    latency: Some(latency),
                    ..Default::default()
                }
            }
            QualificationTrack::Concurrency => {
                let latencies = join_all(
                    (0..request.parallel_requests)
                        .map(|_| measure_completion(&client, &endpoint, request)),
                )
                .await;
                let failed = latencies
                    .iter()
                    .filter(|result| result.status != QualificationStatus::Completed)
                    .count() as u32;
                let completed = latencies.len() as u32 - failed;
                let mut latency = latencies.into_iter().next().unwrap_or_default();
                latency.requests_completed = completed;
                latency.requests_failed = failed;
                QualificationTrackResult {
                    track: *track,
                    status: if failed == 0 {
                        QualificationStatus::Completed
                    } else {
                        QualificationStatus::Failed
                    },
                    latency: Some(latency),
                    ..Default::default()
                }
            }
            QualificationTrack::ToolCorrectness => {
                let tool = measure_tool_correctness(&client, &endpoint, request).await;
                QualificationTrackResult {
                    track: *track,
                    status: tool.status,
                    tool: Some(tool),
                    ..Default::default()
                }
            }
            QualificationTrack::Mtp if capabilities.supports(QualificationTrack::Mtp) => {
                let mut latency = measure_completion(&client, &endpoint, request).await;
                // Windows can briefly lose a loopback connection while the
                // CUDA-backed server finishes its first request. Retry once
                // against the same managed endpoint before recording failure.
                if latency.status == QualificationStatus::Failed {
                    tokio::time::sleep(Duration::from_millis(250)).await;
                    let retry = measure_completion(&client, &endpoint, request).await;
                    if retry.status == QualificationStatus::Completed {
                        latency = retry;
                    }
                }
                QualificationTrackResult {
                    track: *track,
                    status: latency.status,
                    latency: Some(latency),
                    ..Default::default()
                }
            }
            QualificationTrack::Ngram if capabilities.supports(QualificationTrack::Ngram) => {
                let latency = measure_completion(&client, &endpoint, request).await;
                QualificationTrackResult {
                    track: *track,
                    status: latency.status,
                    latency: Some(latency),
                    ..Default::default()
                }
            }
            QualificationTrack::Mtp | QualificationTrack::Dflash | QualificationTrack::Ngram => {
                QualificationTrackResult {
                    track: *track,
                    status: QualificationStatus::Unsupported,
                    error: Some(
                        "capability-qualified server execution is wired in a later track adapter"
                            .into(),
                    ),
                    ..Default::default()
                }
            }
        };
        tracks.push(result);
    }

    Ok(QualificationReceipt {
        schema_version: QUALIFICATION_SCHEMA_VERSION,
        endpoint,
        parallel_requests: request.parallel_requests,
        tracks,
        server_exit_code: None,
        server_log_tail: None,
        memory: None,
        capability_evidence: capabilities.evidence.clone(),
        baseline: None,
    })
}

pub async fn run_managed_server(
    launch: SupervisedLaunch,
    request: &QualificationRequest,
) -> Result<QualificationReceipt> {
    run_managed_server_with_capabilities(launch, request, &QualificationCapabilities::default())
        .await
}

pub async fn run_managed_server_with_capabilities(
    launch: SupervisedLaunch,
    request: &QualificationRequest,
    capabilities: &QualificationCapabilities,
) -> Result<QualificationReceipt> {
    request.validate()?;
    let server = ManagedServer::start(launch).await?;
    let endpoint = server.endpoint.clone();
    let pid = server.child.id();
    let stop_sampler = Arc::new(AtomicBool::new(false));
    let sampler_stop = stop_sampler.clone();
    let sampler = tokio::spawn(async move { sample_process_memory(pid, sampler_stop).await });
    let result = run_against_endpoint_with_capabilities(&endpoint, request, capabilities).await;
    stop_sampler.store(true, Ordering::Release);
    let memory = sampler.await.ok().flatten();
    let (exit_code, server_log_tail) = server.stop().await;
    match result {
        Ok(mut receipt) => {
            receipt.server_exit_code = exit_code;
            receipt.server_log_tail = Some(server_log_tail);
            receipt.memory = memory;
            Ok(receipt)
        }
        Err(error) => Err(error),
    }
}

async fn sample_process_memory(pid: Option<u32>, stop: Arc<AtomicBool>) -> Option<MemoryTelemetry> {
    let pid = pid?;
    let mut system = sysinfo::System::new().ok()?;
    let process_id = sysinfo::Pid::from_u32(pid);
    let mut before = None;
    let mut peak = None;
    loop {
        system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[process_id]), true);
        if let Some(memory) = system.process(process_id).map(|process| process.memory()) {
            before.get_or_insert(memory);
            peak = Some(peak.map_or(memory, |value: u64| value.max(memory)));
        }
        if stop.load(Ordering::Acquire) {
            let after = system.process(process_id).map(|process| process.memory());
            return Some(MemoryTelemetry {
                process_rss_before_bytes: before,
                process_rss_peak_bytes: peak,
                process_rss_after_bytes: after,
            });
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

async fn measure_completion(
    client: &Client,
    endpoint: &str,
    request: &QualificationRequest,
) -> LatencyMemoryResult {
    let started = Instant::now();
    let response = match client
        .post(format!("{endpoint}/v1/chat/completions"))
        .json(&json!({
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.generation_tokens,
            "temperature": 0,
            "stream": true,
            "stream_options": {"include_usage": true}
        }))
        .send()
        .await
    {
        Ok(response) => response,
        Err(error) => return failed_latency(error.to_string()),
    };
    if !response.status().is_success() {
        return failed_latency(format!("completion returned HTTP {}", response.status()));
    }

    let mut stream = response.bytes_stream();
    let mut bytes_seen = 0usize;
    let mut first_token_ms = None;
    let mut completion_tokens = 0u32;
    let mut prompt_tokens = 0u32;
    let mut buffer = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(chunk) => chunk,
            Err(error) => return failed_latency(error.to_string()),
        };
        bytes_seen = bytes_seen.saturating_add(chunk.len());
        if bytes_seen > MAX_RESPONSE_BYTES {
            return failed_latency("qualification response exceeded bounded size".into());
        }
        buffer.extend_from_slice(&chunk);
        while let Some(index) = buffer.iter().position(|byte| *byte == b'\n') {
            let line = buffer.drain(..=index).collect::<Vec<_>>();
            let line = String::from_utf8_lossy(&line);
            let Some(data) = line.strip_prefix("data: ").map(str::trim) else {
                continue;
            };
            if data == "[DONE]" {
                continue;
            }
            let Ok(value) = serde_json::from_str::<Value>(data) else {
                continue;
            };
            if first_token_ms.is_none()
                && value["choices"][0]["delta"]["content"]
                    .as_str()
                    .is_some_and(|v| !v.is_empty())
            {
                first_token_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
            }
            completion_tokens = value["usage"]["completion_tokens"]
                .as_u64()
                .map_or(completion_tokens, |v| v as u32);
            prompt_tokens = value["usage"]["prompt_tokens"]
                .as_u64()
                .map_or(prompt_tokens, |v| v as u32);
        }
    }
    if bytes_seen == 0 {
        return failed_latency("qualification response was empty".into());
    }
    let total_ms = started.elapsed().as_secs_f64() * 1000.0;
    let first_token_ms = first_token_ms.unwrap_or(total_ms);
    let decode_ms = (total_ms - first_token_ms).max(1.0);
    LatencyMemoryResult {
        status: QualificationStatus::Completed,
        time_to_first_token_ms: Some(first_token_ms),
        total_time_ms: Some(total_ms),
        prompt_tokens,
        completion_tokens,
        prompt_tokens_per_second: rate(prompt_tokens, first_token_ms),
        completion_tokens_per_second: rate(completion_tokens, decode_ms),
        requests_completed: 1,
        requests_failed: 0,
        error: None,
    }
}

fn failed_latency(error: String) -> LatencyMemoryResult {
    LatencyMemoryResult {
        status: QualificationStatus::Failed,
        error: Some(error),
        ..Default::default()
    }
}

fn rate(tokens: u32, millis: f64) -> Option<f64> {
    (tokens > 0 && millis > 0.0).then(|| f64::from(tokens) / (millis / 1000.0))
}

async fn measure_tool_correctness(
    client: &Client,
    endpoint: &str,
    request: &QualificationRequest,
) -> ToolCorrectnessResult {
    let response = match client
        .post(format!("{endpoint}/v1/chat/completions"))
        .json(&json!({
            "messages": [{"role": "user", "content": "Use the calibration tool exactly once."}],
            "max_tokens": request.generation_tokens.min(256),
            "temperature": 0,
            "tools": [{"type": "function", "function": {
                "name": "calibration_probe",
                "description": "Record a deterministic calibration probe.",
                "parameters": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
            }}],
            "tool_choice": {"type": "function", "function": {"name": "calibration_probe"}}
        }))
        .send()
        .await
    {
        Ok(response) => response,
        Err(error) => {
            return ToolCorrectnessResult {
                status: QualificationStatus::Failed,
                error: Some(error.to_string()),
                ..Default::default()
            };
        }
    };
    if !response.status().is_success() {
        return ToolCorrectnessResult {
            status: QualificationStatus::Failed,
            error: Some(format!("tool probe returned HTTP {}", response.status())),
            ..Default::default()
        };
    }
    let body = match response.bytes().await {
        Ok(body) if body.len() <= MAX_RESPONSE_BYTES => body,
        Ok(_) => {
            return ToolCorrectnessResult {
                status: QualificationStatus::Failed,
                error: Some("tool probe response exceeded bounded size".into()),
                ..Default::default()
            };
        }
        Err(error) => {
            return ToolCorrectnessResult {
                status: QualificationStatus::Failed,
                error: Some(error.to_string()),
                ..Default::default()
            };
        }
    };
    let value = match serde_json::from_slice::<Value>(&body) {
        Ok(value) => value,
        Err(error) => {
            return ToolCorrectnessResult {
                status: QualificationStatus::Failed,
                error: Some(error.to_string()),
                ..Default::default()
            };
        }
    };
    let tool_call_observed = value["choices"][0]["message"]["tool_calls"]
        .as_array()
        .is_some_and(|calls| !calls.is_empty());
    let structured_output_observed = value["choices"][0]["message"]["content"]
        .as_str()
        .is_some_and(|content| serde_json::from_str::<Value>(content).is_ok());
    ToolCorrectnessResult {
        status: if tool_call_observed {
            QualificationStatus::Completed
        } else {
            QualificationStatus::Failed
        },
        tool_call_observed,
        structured_output_observed,
        error: (!tool_call_observed).then(|| "model did not return the requested tool call".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use warp::Filter;

    #[test]
    fn defaults_are_single_user_and_latency_only() {
        let request = QualificationRequest::default();
        assert_eq!(request.parallel_requests, 1);
        assert!(!request.allow_concurrency);
        assert!(request.validate().is_ok());
    }

    #[test]
    fn concurrency_requires_explicit_opt_in() {
        let request = QualificationRequest {
            parallel_requests: 2,
            ..Default::default()
        };
        assert!(request.validate().is_err());
        let request = QualificationRequest {
            tracks: BTreeSet::from([QualificationTrack::Concurrency]),
            parallel_requests: 2,
            allow_concurrency: true,
            ..Default::default()
        };
        assert!(request.validate().is_ok());
    }

    #[test]
    fn mtp_cannot_use_concurrency() {
        let request = QualificationRequest {
            tracks: BTreeSet::from([QualificationTrack::Mtp, QualificationTrack::Concurrency]),
            parallel_requests: 2,
            allow_concurrency: true,
            ..Default::default()
        };
        assert!(request.validate().is_err());
    }

    #[tokio::test]
    async fn fake_server_covers_latency_and_tool_tracks() {
        let health = warp::path!("health").map(|| warp::reply::json(&json!({"status": "ok"})));
        let completions = warp::path!("v1" / "chat" / "completions")
            .and(warp::post())
            .and(warp::body::json())
            .map(|body: Value| {
                if body["tools"].is_array() {
                    warp::reply::json(&json!({
                        "choices": [{"message": {"tool_calls": [{"function": {"name": "calibration_probe"}}]}}]
                    }))
                } else {
                    warp::reply::json(&json!({
                        "choices": [{"message": {"content": "calibration ok"}}],
                        "usage": {"prompt_tokens": 8, "completion_tokens": 4}
                    }))
                }
            });
        let listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
            .expect("allocate fake server port");
        let port = listener.local_addr().expect("fake server address").port();
        drop(listener);
        tokio::spawn(warp::serve(health.or(completions)).run(([127, 0, 0, 1], port)));
        let probe_client = Client::new();
        for _ in 0..50 {
            if probe_client
                .get(format!("http://127.0.0.1:{port}/health"))
                .send()
                .await
                .is_ok()
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let request = QualificationRequest {
            tracks: BTreeSet::from([
                QualificationTrack::LatencyMemory,
                QualificationTrack::ToolCorrectness,
            ]),
            ..Default::default()
        };
        let receipt = run_against_endpoint(&format!("http://127.0.0.1:{port}"), &request)
            .await
            .expect("fake server qualification");
        assert_eq!(receipt.tracks.len(), 2);
        assert!(
            receipt
                .tracks
                .iter()
                .all(|track| track.status == QualificationStatus::Completed),
            "unexpected qualification tracks: {:#?}",
            receipt.tracks
        );
        assert!(
            receipt
                .tracks
                .iter()
                .find_map(|track| track.tool.as_ref())
                .is_some_and(|tool| tool.tool_call_observed)
        );
    }

    #[tokio::test]
    async fn failed_http_probe_is_recorded_without_panicking() {
        let route = warp::path!("v1" / "chat" / "completions")
            .and(warp::post())
            .map(|| {
                warp::reply::with_status(
                    "probe failed",
                    warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                )
            });
        let listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
            .expect("allocate fake failure port");
        let port = listener
            .local_addr()
            .expect("failure server address")
            .port();
        drop(listener);
        tokio::spawn(warp::serve(route).run(([127, 0, 0, 1], port)));
        let receipt = run_against_endpoint(
            &format!("http://127.0.0.1:{port}"),
            &QualificationRequest::default(),
        )
        .await
        .expect("failed probe receipt");
        assert_eq!(receipt.tracks[0].status, QualificationStatus::Failed);
    }

    #[tokio::test]
    async fn empty_http_probe_is_recorded_as_failed() {
        let route = warp::path!("v1" / "chat" / "completions")
            .and(warp::post())
            .map(|| warp::reply::with_status("", warp::http::StatusCode::OK));
        let listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
            .expect("allocate empty response port");
        let port = listener
            .local_addr()
            .expect("empty response server address")
            .port();
        drop(listener);
        tokio::spawn(warp::serve(route).run(([127, 0, 0, 1], port)));
        tokio::time::sleep(Duration::from_millis(10)).await;
        let receipt = run_against_endpoint(
            &format!("http://127.0.0.1:{port}"),
            &QualificationRequest::default(),
        )
        .await
        .expect("empty probe receipt");
        assert_eq!(receipt.tracks[0].status, QualificationStatus::Failed);
    }
}
