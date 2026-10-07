use std::sync::OnceLock;
use std::time::Duration;

use warp::http::StatusCode;

use super::ApiError;
use crate::inference::InferenceBackend;
use crate::inference::backend::BackendAdapter;
use crate::state::{AppState, SessionMode};

const MONITOR_INFERENCE_QUEUE_TIMEOUT: Duration = Duration::from_secs(330);
const UPSTREAM_BUSY_WAIT_TIMEOUT: Duration = Duration::from_secs(300);
const UPSTREAM_BUSY_POLL_INTERVAL: Duration = Duration::from_millis(500);
const UPSTREAM_SEND_RETRIES: usize = 3;
const UPSTREAM_SEND_RETRY_BACKOFF_MS: u64 = 250;
const MAX_UPSTREAM_ERROR_BYTES: usize = 64 * 1024;

const UPSTREAM_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Clone, Copy)]
struct UpstreamTimeouts {
    headers: Duration,
    first_byte: Duration,
    idle: Duration,
    helper: Duration,
    error_body: Duration,
}

// Headers and the first body byte can both follow a long, silent prefill.
// Once bytes are flowing, bound stalls without limiting healthy chat duration.
const UPSTREAM_TIMEOUTS: UpstreamTimeouts = UpstreamTimeouts {
    headers: Duration::from_secs(15 * 60),
    first_byte: Duration::from_secs(15 * 60),
    idle: Duration::from_secs(5 * 60),
    helper: Duration::from_secs(30 * 60),
    error_body: Duration::from_secs(30),
};

pub(crate) struct PreparedInferenceRequest {
    pub url: String,
    pub permit: tokio::sync::OwnedSemaphorePermit,
    pub backend: InferenceBackend,
    model_identity: Option<String>,
    api_key: Option<String>,
    adapter: Option<BackendAdapter>,
}

#[derive(Clone)]
struct ActiveInferenceTarget {
    session_id: String,
    url: String,
    backend: InferenceBackend,
    model_identity: Option<String>,
    api_key: Option<String>,
    local_spawn: bool,
}

impl PartialEq for ActiveInferenceTarget {
    fn eq(&self, other: &Self) -> bool {
        use subtle::ConstantTimeEq;

        let api_key_matches = match (&self.api_key, &other.api_key) {
            (Some(left), Some(right)) => left.as_bytes().ct_eq(right.as_bytes()).into(),
            (None, None) => true,
            _ => false,
        };
        self.session_id == other.session_id
            && self.url == other.url
            && self.backend == other.backend
            && self.model_identity == other.model_identity
            && api_key_matches
            && self.local_spawn == other.local_spawn
    }
}

impl Eq for ActiveInferenceTarget {}

pub(crate) fn adapter_matches_backend(backend: InferenceBackend, adapter: &BackendAdapter) -> bool {
    matches!(
        (backend, adapter),
        (InferenceBackend::LlamaCpp, BackendAdapter::LlamaCpp(_))
            | (InferenceBackend::RapidMlx, BackendAdapter::RapidMlx(_))
    )
}

pub(crate) fn local_connect_host(bind_host: Option<&str>) -> &str {
    match bind_host {
        None | Some("") | Some("0.0.0.0") | Some("::") | Some("[::]") => "127.0.0.1",
        Some("::1") => "[::1]",
        Some(host) => host,
    }
}

impl PreparedInferenceRequest {
    /// The served model identity for this request, or `""` when none is known
    /// (e.g. an attach session that never reported one). Used to plumb the
    /// positional `model` argument into `rapid-mlx bench`.
    pub fn model_identity(&self) -> &str {
        self.model_identity.as_deref().unwrap_or_default()
    }

    pub fn authenticate(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.api_key {
            Some(key) if !key.is_empty() => request.bearer_auth(key),
            _ => request,
        }
    }

    pub fn map_chat_body(&self, body: &[u8]) -> Result<Vec<u8>, warp::Rejection> {
        let mapped = match self.backend {
            InferenceBackend::LlamaCpp => Ok(body.to_vec()),
            InferenceBackend::RapidMlx => match &self.adapter {
                Some(BackendAdapter::RapidMlx(adapter)) => adapter.map_chat_request(body),
                None => crate::inference::rapid_mlx::map_provisional_chat_request(body),
                Some(BackendAdapter::LlamaCpp(_)) => {
                    crate::inference::rapid_mlx::map_provisional_chat_request(body)
                }
            },
        };
        let mut mapped = mapped.map_err(|error| {
            warp::reject::custom(ApiError::new(StatusCode::BAD_REQUEST, error.to_string()))
        })?;
        if self.backend == InferenceBackend::RapidMlx
            && let Some(model) = self
                .model_identity
                .as_ref()
                .filter(|model| !model.is_empty())
        {
            let mut value: serde_json::Value = serde_json::from_slice(&mapped)
                .map_err(|error| warp::reject::custom(ApiError::internal(error.to_string())))?;
            if let Some(object) = value.as_object_mut()
                && !object.contains_key("model")
            {
                object.insert(
                    "model".to_string(),
                    serde_json::Value::String(model.clone()),
                );
                mapped = serde_json::to_vec(&value)
                    .map_err(|error| warp::reject::custom(ApiError::internal(error.to_string())))?;
            }
        }
        Ok(mapped)
    }
}

fn active_inference_target(state: &AppState) -> Result<ActiveInferenceTarget, warp::Rejection> {
    let session = state
        .get_active_session()
        .ok_or(warp::reject::not_found())?;
    let (url, api_key, local_spawn) = match &session.mode {
        SessionMode::Spawn {
            port,
            bind_host,
            api_key,
        } => {
            let host = local_connect_host(bind_host.as_deref());
            (
                format!("http://{host}:{port}/v1/chat/completions"),
                api_key.clone(),
                true,
            )
        }
        SessionMode::Attach { endpoint, api_key } => (
            format!("{endpoint}/v1/chat/completions"),
            api_key.clone(),
            false,
        ),
    };
    Ok(ActiveInferenceTarget {
        session_id: session.id,
        url,
        backend: session.backend,
        model_identity: session.model_identity,
        api_key,
        local_spawn,
    })
}

fn upstream_has_capacity(
    state: &AppState,
    backend: InferenceBackend,
) -> Result<bool, warp::Rejection> {
    let server_running = *state.server_running.lock().map_err(|e| {
        warp::reject::custom(ApiError::internal(format!(
            "Failed to read server state: {e}"
        )))
    })?;
    if !server_running {
        return Err(warp::reject::custom(ApiError::gateway(
            "Cannot reach the active inference runtime.",
        )));
    }
    if backend == InferenceBackend::RapidMlx {
        return Ok(true);
    }

    let metrics = state.llama_metrics.lock().map_err(|e| {
        warp::reject::custom(ApiError::internal(format!(
            "Failed to read llama metrics: {e}"
        )))
    })?;
    let total_slots = metrics.slots_idle.saturating_add(metrics.slots_processing);
    if total_slots == 0 {
        return Ok(true);
    }
    Ok(metrics.slots_processing < total_slots)
}

async fn wait_for_upstream_capacity(
    state: &AppState,
    backend: InferenceBackend,
    deadline: tokio::time::Instant,
) -> Result<(), warp::Rejection> {
    loop {
        if upstream_has_capacity(state, backend)? {
            return Ok(());
        }

        if tokio::time::Instant::now() >= deadline {
            return Err(warp::reject::custom(ApiError::busy(
                "The active inference runtime has been busy for too long. Wait for it to finish and retry.",
            )));
        }

        tokio::time::sleep(UPSTREAM_BUSY_POLL_INTERVAL).await;
    }
}

async fn acquire_inference_permit(
    state: &AppState,
    backend: InferenceBackend,
    deadline: tokio::time::Instant,
) -> Result<tokio::sync::OwnedSemaphorePermit, warp::Rejection> {
    let gate = match backend {
        InferenceBackend::LlamaCpp => state.monitor_inference_gate.clone(),
        InferenceBackend::RapidMlx => state.rapid_mlx_inference_gate.clone(),
    };
    let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
    tokio::time::timeout(
        remaining,
        gate.acquire_owned(),
    )
    .await
    .map_err(|_| {
        warp::reject::custom(ApiError::busy(
            "Another llama-monitor inference request has been queued for too long. Retry in a moment.",
        ))
    })?
    .map_err(|_| warp::reject::custom(ApiError::internal("Inference gate was closed.")))
}

pub(crate) async fn prepare_inference_request(
    state: &AppState,
) -> Result<PreparedInferenceRequest, warp::Rejection> {
    let queue_deadline = tokio::time::Instant::now() + MONITOR_INFERENCE_QUEUE_TIMEOUT;
    loop {
        let target_before_wait = active_inference_target(state)?;
        let permit =
            acquire_inference_permit(state, target_before_wait.backend, queue_deadline).await?;
        let target = active_inference_target(state)?;
        if target != target_before_wait {
            drop(permit);
            continue;
        }

        let capacity_deadline = std::cmp::min(
            queue_deadline,
            tokio::time::Instant::now() + UPSTREAM_BUSY_WAIT_TIMEOUT,
        );
        wait_for_upstream_capacity(state, target.backend, capacity_deadline).await?;
        // Capacity polling can also outlive a session switch. Never use the permit
        // or route captured for the old target in that case.
        if active_inference_target(state)? != target {
            drop(permit);
            continue;
        }

        let adapter = state
            .backend
            .lock()
            .map_err(|error| {
                warp::reject::custom(ApiError::internal(format!(
                    "Failed to read active backend: {error}"
                )))
            })?
            .clone()
            .filter(|adapter| {
                target.local_spawn && adapter_matches_backend(target.backend, adapter)
            });
        return Ok(PreparedInferenceRequest {
            url: target.url,
            permit,
            backend: target.backend,
            model_identity: target.model_identity,
            api_key: target.api_key,
            adapter,
        });
    }
}

pub(crate) fn build_upstream_client() -> Result<&'static reqwest::Client, warp::Rejection> {
    static CLIENT: OnceLock<Result<reqwest::Client, String>> = OnceLock::new();
    CLIENT
        .get_or_init(|| {
            reqwest::Client::builder()
                .connect_timeout(UPSTREAM_CONNECT_TIMEOUT)
                .build()
                .map_err(|error| error.to_string())
        })
        .as_ref()
        .map_err(|error| {
            warp::reject::custom(ApiError::internal(format!(
                "Failed to create HTTP client: {error}"
            )))
        })
}

fn should_retry_send_error(err: &reqwest::Error) -> bool {
    err.is_connect()
        || (!err.is_timeout()
            && (err.to_string().contains("error sending request")
                || err.to_string().contains("connection reset")))
}

pub(crate) async fn send_upstream_request_with_retry<F>(
    make_request: F,
) -> Result<reqwest::Response, warp::Rejection>
where
    F: FnMut() -> reqwest::RequestBuilder,
{
    send_upstream_request_with_timeouts(make_request, UPSTREAM_TIMEOUTS, None).await
}

/// Helpers collect their output before replying and hold an inference permit.
/// Use one deadline across retries and the response body, unlike interactive
/// chat whose healthy token stream has no overall duration limit.
pub(crate) async fn send_helper_request_with_retry<F>(
    make_request: F,
) -> Result<reqwest::Response, warp::Rejection>
where
    F: FnMut() -> reqwest::RequestBuilder,
{
    let deadline = tokio::time::Instant::now() + UPSTREAM_TIMEOUTS.helper;
    send_upstream_request_with_timeouts(make_request, UPSTREAM_TIMEOUTS, Some(deadline)).await
}

async fn send_upstream_request_with_timeouts<F>(
    mut make_request: F,
    timeouts: UpstreamTimeouts,
    helper_deadline: Option<tokio::time::Instant>,
) -> Result<reqwest::Response, warp::Rejection>
where
    F: FnMut() -> reqwest::RequestBuilder,
{
    let mut attempt = 0usize;
    let mut backoff_ms = UPSTREAM_SEND_RETRY_BACKOFF_MS;

    loop {
        attempt += 1;

        let now = tokio::time::Instant::now();
        let mut request = make_request();
        let mut header_deadline = now + timeouts.headers;
        if let Some(deadline) = helper_deadline {
            header_deadline = header_deadline.min(deadline);
            // reqwest carries this remaining overall budget into body reads.
            // Recomputing it per attempt keeps retries within the same deadline.
            request = request.timeout(deadline.saturating_duration_since(now));
        }
        let response = tokio::time::timeout_at(header_deadline, request.send())
            .await
            .map_err(|_| {
                // A connected runtime may have accepted this inference. Do not
                // replay it merely because prefill never produced headers.
                warp::reject::custom(upstream_timeout("response headers"))
            })?;
        match response {
            Ok(resp) if resp.status().is_success() => return Ok(resp),
            Ok(resp) => {
                let status = resp.status();
                let err_body = read_error_body_bounded(resp, timeouts.error_body)
                    .await
                    .map_err(warp::reject::custom)?;
                let mapped = ApiError::from_upstream_status(status, err_body);
                if attempt < UPSTREAM_SEND_RETRIES && mapped.status == StatusCode::TOO_MANY_REQUESTS
                {
                    retry_backoff(backoff_ms, helper_deadline).await?;
                    backoff_ms *= 2;
                    continue;
                }
                return Err(warp::reject::custom(mapped));
            }
            Err(err) => {
                if attempt < UPSTREAM_SEND_RETRIES && should_retry_send_error(&err) {
                    retry_backoff(backoff_ms, helper_deadline).await?;
                    backoff_ms *= 2;
                    continue;
                }
                return Err(warp::reject::custom(ApiError::from_reqwest(err)));
            }
        }
    }
}

async fn retry_backoff(
    backoff_ms: u64,
    helper_deadline: Option<tokio::time::Instant>,
) -> Result<(), warp::Rejection> {
    let sleep = tokio::time::sleep(Duration::from_millis(backoff_ms));
    if let Some(deadline) = helper_deadline {
        tokio::time::timeout_at(deadline, sleep)
            .await
            .map_err(|_| warp::reject::custom(upstream_timeout("helper completion")))?;
    } else {
        sleep.await;
    }
    Ok(())
}

fn upstream_timeout(phase: &str) -> ApiError {
    ApiError::new(
        StatusCode::GATEWAY_TIMEOUT,
        format!("Timed out waiting for upstream {phase}."),
    )
}

type UpstreamBodyStream =
    std::pin::Pin<Box<dyn futures_util::Stream<Item = Result<bytes::Bytes, ApiError>> + Send>>;

pub(crate) fn upstream_body_stream(response: reqwest::Response) -> UpstreamBodyStream {
    upstream_body_stream_with_timeouts(response, UPSTREAM_TIMEOUTS)
}

fn upstream_body_stream_with_timeouts(
    response: reqwest::Response,
    timeouts: UpstreamTimeouts,
) -> UpstreamBodyStream {
    use futures_util::StreamExt;
    Box::pin(async_stream::stream! {
        let mut stream = response.bytes_stream();
        let mut wait = timeouts.first_byte;
        loop {
            match tokio::time::timeout(wait, stream.next()).await {
                Ok(Some(Ok(bytes))) => {
                    wait = timeouts.idle;
                    yield Ok(bytes);
                }
                Ok(Some(Err(error))) => {
                    yield Err(ApiError::from_reqwest(error));
                    break;
                }
                Ok(None) => break,
                Err(_) => {
                    yield Err(upstream_timeout("response body"));
                    break;
                }
            }
        }
    })
}

async fn read_error_body_bounded(
    response: reqwest::Response,
    timeout: Duration,
) -> Result<String, ApiError> {
    use futures_util::StreamExt;

    let mut stream = response.bytes_stream();
    let mut body = Vec::new();
    let deadline = tokio::time::Instant::now() + timeout;
    while let Some(chunk) = tokio::time::timeout_at(deadline, stream.next())
        .await
        .map_err(|_| upstream_timeout("error response body"))?
    {
        let Ok(chunk) = chunk else {
            break;
        };
        let remaining = MAX_UPSTREAM_ERROR_BYTES.saturating_sub(body.len());
        if chunk.len() > remaining {
            body.extend_from_slice(&chunk[..remaining]);
            break;
        }
        body.extend_from_slice(&chunk);
        if body.len() == MAX_UPSTREAM_ERROR_BYTES {
            break;
        }
    }
    Ok(String::from_utf8_lossy(&body).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::rapid_mlx::RapidMlxAdapter;
    use crate::inference::rapid_mlx::runtime::{RuntimeMetadata, RuntimeSource};
    use crate::state::Session;

    fn state_with_backend(backend: InferenceBackend, id: &str, port: u16) -> AppState {
        let state = AppState::default();
        let session = Session::new_spawn_with_backend(
            id.into(),
            id.into(),
            port,
            String::new(),
            Some("127.0.0.1".into()),
            None,
            backend,
            Some(format!("{id}-model")),
        );
        assert!(state.add_session(session));
        state.set_active_session(id);
        *state.server_running.lock().unwrap() = true;
        state
    }

    async fn prepared(
        backend: InferenceBackend,
        model_identity: Option<&str>,
        api_key: Option<&str>,
    ) -> PreparedInferenceRequest {
        let gate = std::sync::Arc::new(tokio::sync::Semaphore::new(1));
        PreparedInferenceRequest {
            url: "http://127.0.0.1:1/v1/chat/completions".into(),
            permit: gate.acquire_owned().await.unwrap(),
            backend,
            model_identity: model_identity.map(str::to_string),
            api_key: api_key.map(str::to_string),
            adapter: None,
        }
    }

    #[tokio::test]
    async fn llama_mapping_remains_byte_exact() {
        let prepared = prepared(InferenceBackend::LlamaCpp, Some("ignored"), None).await;
        let body = br#"{ "messages": [], "seed": 7, "cache_prompt": true }"#;
        assert_eq!(prepared.map_chat_body(body).unwrap(), body);
    }

    #[tokio::test]
    async fn rapid_mapping_injects_persisted_model_identity_only_when_absent() {
        let prepared = prepared(InferenceBackend::RapidMlx, Some("served-model"), None).await;
        let mapped = prepared
            .map_chat_body(br#"{"messages":[],"stream":true}"#)
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&mapped).unwrap();
        assert_eq!(value["model"], "served-model");

        let mapped = prepared
            .map_chat_body(br#"{"messages":[],"model":"explicit"}"#)
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&mapped).unwrap();
        assert_eq!(value["model"], "explicit");
    }

    #[tokio::test]
    async fn attached_rapid_maps_shared_reasoning_budget_with_provisional_fields() {
        let prepared = prepared(InferenceBackend::RapidMlx, Some("attached-model"), None).await;
        let mapped = prepared
            .map_chat_body(br#"{"messages":[],"thinking_budget_tokens":2048}"#)
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&mapped).unwrap();
        assert_eq!(value["reasoning_max_tokens"], 2048);
        assert!(value.get("thinking_budget_tokens").is_none());
    }

    #[tokio::test]
    async fn rapid_spawn_maps_shared_reasoning_budget_to_native_cap() {
        let gate = std::sync::Arc::new(tokio::sync::Semaphore::new(1));
        let adapter = RapidMlxAdapter::from_resolved(
            RuntimeMetadata {
                executable_path: "rapid-mlx".into(),
                source: RuntimeSource::Managed,
                version: "0.11.1".into(),
                capability_snapshot: None,
                resolved_receipt: None,
                last_probe_result: None,
                prefix_cache_enabled: false,
                mlx_prefix_cache_bytes: None,
            },
            crate::inference::rapid_mlx::model_resolver::ResolvedRapidMlxLaunchModel::validated_alias(
                "reasoning-cap-model",
            )
            .unwrap(),
        );
        let prepared = PreparedInferenceRequest {
            url: "http://127.0.0.1:1/v1/chat/completions".into(),
            permit: gate.acquire_owned().await.unwrap(),
            backend: InferenceBackend::RapidMlx,
            model_identity: Some("reasoning-cap-model".into()),
            api_key: None,
            adapter: Some(BackendAdapter::RapidMlx(std::sync::Arc::new(adapter))),
        };

        let mapped = prepared
            .map_chat_body(br#"{"messages":[],"thinking_budget_tokens":2048,"stream":true}"#)
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&mapped).unwrap();
        assert_eq!(value["reasoning_max_tokens"], 2048);
        assert!(value.get("thinking_budget_tokens").is_none());
    }

    #[tokio::test]
    async fn upstream_auth_uses_transient_bearer_header() {
        let with_key = prepared(InferenceBackend::RapidMlx, None, Some("transient-secret")).await;
        let request = with_key
            .authenticate(reqwest::Client::new().get("http://127.0.0.1:1"))
            .build()
            .unwrap();
        assert_eq!(
            request.headers()[reqwest::header::AUTHORIZATION],
            "Bearer transient-secret"
        );

        let no_key = prepared(InferenceBackend::RapidMlx, None, None).await;
        let request = no_key
            .authenticate(reqwest::Client::new().get("http://127.0.0.1:1"))
            .build()
            .unwrap();
        assert!(
            !request
                .headers()
                .contains_key(reqwest::header::AUTHORIZATION)
        );
    }

    #[tokio::test]
    async fn upstream_non_success_status_is_propagated() {
        let mut server = mockito::Server::new_async().await;
        let _failure = server
            .mock("POST", "/v1/chat/completions")
            .with_status(401)
            .with_body("invalid runtime key")
            .create_async()
            .await;
        let client = reqwest::Client::new();
        let url = format!("{}/v1/chat/completions", server.url());
        let rejection = send_upstream_request_with_retry(|| client.post(&url))
            .await
            .unwrap_err();
        let error = rejection.find::<ApiError>().unwrap();
        assert_eq!(error.status, StatusCode::UNAUTHORIZED);
        assert!(error.message.contains("invalid runtime key"));
    }

    #[tokio::test]
    async fn upstream_error_body_is_bounded() {
        let mut server = mockito::Server::new_async().await;
        let _failure = server
            .mock("POST", "/v1/chat/completions")
            .with_status(500)
            .with_body("x".repeat(MAX_UPSTREAM_ERROR_BYTES + 4096))
            .create_async()
            .await;
        let client = reqwest::Client::new();
        let url = format!("{}/v1/chat/completions", server.url());
        let rejection = send_upstream_request_with_retry(|| client.post(&url))
            .await
            .unwrap_err();
        let error = rejection.find::<ApiError>().unwrap();
        assert_eq!(error.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert!(error.message.len() <= MAX_UPSTREAM_ERROR_BYTES + 64);
    }

    #[test]
    fn spawned_ipv6_loopback_routes_to_a_valid_local_url() {
        let state = AppState::default();
        let session = Session::new_spawn_with_backend(
            "rapid-ipv6".into(),
            "Rapid IPv6".into(),
            8123,
            String::new(),
            Some("::1".into()),
            None,
            InferenceBackend::RapidMlx,
            Some("model".into()),
        );
        assert!(state.add_session(session));
        state.set_active_session("rapid-ipv6");
        let target = active_inference_target(&state).unwrap();
        assert_eq!(target.url, "http://[::1]:8123/v1/chat/completions");
        assert_eq!(target.backend, InferenceBackend::RapidMlx);
        assert!(target.local_spawn);
    }

    #[tokio::test]
    async fn attached_session_never_inherits_a_stale_spawn_adapter() {
        let state = AppState::default();
        let mut session = Session::new_attach(
            "attached-rapid".into(),
            "Attached Rapid".into(),
            "http://127.0.0.1:9000".into(),
            None,
        );
        session.backend = InferenceBackend::RapidMlx;
        session.model_identity = Some("attached-model".into());
        assert!(state.add_session(session));
        state.set_active_session("attached-rapid");
        *state.server_running.lock().unwrap() = true;
        *state.backend.lock().unwrap() = Some(BackendAdapter::RapidMlx(std::sync::Arc::new(
            RapidMlxAdapter::from_resolved(
                RuntimeMetadata {
                    executable_path: "rapid-mlx".into(),
                    source: RuntimeSource::Managed,
                    version: "0.10.9".into(),
                    capability_snapshot: None,
                    resolved_receipt: None,
                    last_probe_result: None,
                    prefix_cache_enabled: false,
                    mlx_prefix_cache_bytes: None,
                },
                crate::inference::rapid_mlx::model_resolver::ResolvedRapidMlxLaunchModel::validated_alias("stale-model").unwrap(),
            ),
        )));

        let prepared = prepare_inference_request(&state).await.unwrap();
        assert!(prepared.adapter.is_none());
        let body = prepared
            .map_chat_body(br#"{"messages":[],"stream":true,"tools":[]}"#)
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(value.get("tools").is_none());
    }

    #[tokio::test]
    async fn llama_is_serialized_and_rapid_mlx_uses_a_fixed_bound() {
        let llama = state_with_backend(InferenceBackend::LlamaCpp, "llama", 8101);
        let first = prepare_inference_request(&llama).await.unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(30), prepare_inference_request(&llama))
                .await
                .is_err()
        );
        drop(first);
        assert!(prepare_inference_request(&llama).await.is_ok());

        let rapid = state_with_backend(InferenceBackend::RapidMlx, "rapid", 8102);
        let mut permits = Vec::new();
        for _ in 0..4 {
            permits.push(prepare_inference_request(&rapid).await.unwrap());
        }
        assert_eq!(rapid.rapid_mlx_inference_gate.available_permits(), 0);
        assert!(
            tokio::time::timeout(Duration::from_millis(30), prepare_inference_request(&rapid))
                .await
                .is_err()
        );
        drop(permits.pop());
        assert!(prepare_inference_request(&rapid).await.is_ok());
    }

    #[tokio::test]
    async fn queued_request_reroutes_after_active_backend_switch() {
        let state = state_with_backend(InferenceBackend::LlamaCpp, "old-llama", 8111);
        let blocker = prepare_inference_request(&state).await.unwrap();
        let queued_state = state.clone();
        let queued = tokio::spawn(async move { prepare_inference_request(&queued_state).await });
        tokio::time::sleep(Duration::from_millis(20)).await;

        let replacement = Session::new_spawn_with_backend(
            "new-rapid".into(),
            "new-rapid".into(),
            8112,
            String::new(),
            Some("127.0.0.1".into()),
            None,
            InferenceBackend::RapidMlx,
            Some("new-model".into()),
        );
        assert!(state.add_session(replacement));
        state.set_active_session("new-rapid");
        drop(blocker);

        let prepared = tokio::time::timeout(Duration::from_secs(1), queued)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(prepared.backend, InferenceBackend::RapidMlx);
        assert!(prepared.url.contains(":8112/"));
    }

    #[tokio::test]
    async fn queued_request_revalidates_a_rotated_protected_key() {
        let state = state_with_backend(InferenceBackend::RapidMlx, "rapid", 8113);
        {
            let mut sessions = state.sessions.lock().unwrap();
            let session = sessions
                .iter_mut()
                .find(|session| session.id == "rapid")
                .unwrap();
            if let SessionMode::Spawn { api_key, .. } = &mut session.mode {
                *api_key = Some("old-secret".into());
            }
        }
        let held: Vec<_> = futures_util::future::join_all(
            (0..4).map(|_| state.rapid_mlx_inference_gate.clone().acquire_owned()),
        )
        .await
        .into_iter()
        .map(Result::unwrap)
        .collect();
        let queued_state = state.clone();
        let queued = tokio::spawn(async move { prepare_inference_request(&queued_state).await });
        tokio::time::sleep(Duration::from_millis(20)).await;

        {
            let mut sessions = state.sessions.lock().unwrap();
            let session = sessions
                .iter_mut()
                .find(|session| session.id == "rapid")
                .unwrap();
            if let SessionMode::Spawn { api_key, .. } = &mut session.mode {
                *api_key = Some("new-secret".into());
            }
        }
        drop(held);

        let prepared = tokio::time::timeout(Duration::from_secs(1), queued)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(prepared.api_key.as_deref(), Some("new-secret"));
        assert_eq!(state.rapid_mlx_inference_gate.available_permits(), 3);
        drop(prepared);
        assert_eq!(state.rapid_mlx_inference_gate.available_permits(), 4);
    }

    #[tokio::test]
    async fn inference_permits_release_on_routing_error_and_cancellation() {
        let state = state_with_backend(InferenceBackend::RapidMlx, "rapid", 8120);
        *state.server_running.lock().unwrap() = false;
        assert!(prepare_inference_request(&state).await.is_err());
        assert_eq!(state.rapid_mlx_inference_gate.available_permits(), 4);

        *state.server_running.lock().unwrap() = true;
        let held: Vec<_> = futures_util::future::join_all(
            (0..4).map(|_| state.rapid_mlx_inference_gate.clone().acquire_owned()),
        )
        .await
        .into_iter()
        .map(Result::unwrap)
        .collect();
        let cancelled =
            tokio::time::timeout(Duration::from_millis(20), prepare_inference_request(&state))
                .await;
        assert!(cancelled.is_err());
        drop(held);
        assert_eq!(state.rapid_mlx_inference_gate.available_permits(), 4);
    }

    #[test]
    fn upstream_http_client_is_reused() {
        let first = build_upstream_client().unwrap();
        let second = build_upstream_client().unwrap();
        assert!(std::ptr::eq(first, second));
        let request = first
            .get("http://127.0.0.1:1")
            .timeout(Duration::from_millis(25))
            .build()
            .unwrap();
        assert_eq!(request.timeout().copied(), Some(Duration::from_millis(25)));
    }

    // Short policy injected into the same transport used by routes, not a
    // request-level timeout that would truncate an otherwise healthy stream.
    fn test_timeouts() -> UpstreamTimeouts {
        UpstreamTimeouts {
            headers: Duration::from_millis(400),
            first_byte: Duration::from_millis(400),
            idle: Duration::from_millis(200),
            helper: Duration::from_millis(600),
            error_body: Duration::from_millis(80),
        }
    }

    async fn scripted_upstream(
        writes: Vec<(Duration, &'static [u8])>,
        stay_open: bool,
    ) -> (String, tokio::task::JoinHandle<()>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buf = [0; 1024];
            while !request.windows(4).any(|part| part == b"\r\n\r\n") {
                let len = socket.read(&mut buf).await.unwrap();
                if len == 0 {
                    return;
                }
                request.extend_from_slice(&buf[..len]);
            }
            for (delay, bytes) in writes {
                tokio::time::sleep(delay).await;
                if socket.write_all(bytes).await.is_err() {
                    return;
                }
            }
            if stay_open {
                // End when the transport drops the connection after timeout.
                while socket.read(&mut buf).await.unwrap_or(0) != 0 {}
            }
        });
        (format!("http://{address}"), task)
    }

    const SSE_HEADERS: &[u8] =
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n";
    const SSE_CHUNK: &[u8] = b"b\r\ndata: x\n\nxx\r\n";

    #[tokio::test]
    async fn silent_headers_timeout_without_replaying_and_release_permit() {
        let (url, server) = scripted_upstream(vec![], true).await;
        let state = state_with_backend(InferenceBackend::LlamaCpp, "silent", 8101);
        let prepared = prepare_inference_request(&state).await.unwrap();
        let mut calls = 0;
        let client = build_upstream_client().unwrap();
        let mut policy = test_timeouts();
        policy.headers = Duration::from_millis(80);
        let result = tokio::time::timeout(Duration::from_secs(2), async {
            let _permit = prepared.permit;
            send_upstream_request_with_timeouts(
                || {
                    calls += 1;
                    client.post(&url)
                },
                policy,
                None,
            )
            .await
        })
        .await
        .expect("silent upstream must be bounded");
        assert_eq!(
            result.unwrap_err().find::<ApiError>().unwrap().status,
            StatusCode::GATEWAY_TIMEOUT
        );
        assert_eq!(
            calls, 1,
            "a possibly accepted inference must not be replayed"
        );
        assert_eq!(state.monitor_inference_gate.available_permits(), 1);
        server.abort();
    }

    #[tokio::test]
    async fn delayed_helper_headers_and_first_token_allow_prefill_longer_than_idle_limit() {
        use futures_util::StreamExt;
        let (url, server) = scripted_upstream(
            vec![
                (Duration::from_millis(150), SSE_HEADERS),
                (Duration::from_millis(150), SSE_CHUNK),
                (Duration::ZERO, b"0\r\n\r\n"),
            ],
            false,
        )
        .await;
        let client = build_upstream_client().unwrap();
        let mut policy = test_timeouts();
        policy.idle = Duration::from_millis(80);
        let response = send_upstream_request_with_timeouts(
            || client.post(&url),
            policy,
            Some(tokio::time::Instant::now() + policy.helper),
        )
        .await
        .unwrap();
        let mut body = upstream_body_stream_with_timeouts(response, policy);
        assert_eq!(
            body.next().await.unwrap().unwrap().as_ref(),
            b"data: x\n\nxx"
        );
        assert!(body.next().await.is_none());
        server.await.unwrap();
    }

    #[tokio::test]
    async fn silent_first_body_and_stalled_token_stream_are_bounded() {
        use futures_util::StreamExt;
        for send_token in [false, true] {
            let mut writes = vec![(Duration::ZERO, SSE_HEADERS)];
            if send_token {
                writes.push((Duration::ZERO, SSE_CHUNK));
            }
            let (url, server) = scripted_upstream(writes, true).await;
            let client = build_upstream_client().unwrap();
            let mut policy = test_timeouts();
            policy.first_byte = Duration::from_millis(80);
            policy.idle = Duration::from_millis(80);
            let response = send_upstream_request_with_timeouts(|| client.post(&url), policy, None)
                .await
                .unwrap();
            let mut body = upstream_body_stream_with_timeouts(response, policy);
            if send_token {
                assert!(body.next().await.unwrap().is_ok());
            }
            let result = tokio::time::timeout(Duration::from_secs(2), body.next())
                .await
                .expect("body waits must be bounded");
            assert_eq!(
                result.unwrap().unwrap_err().status,
                StatusCode::GATEWAY_TIMEOUT
            );
            assert!(body.next().await.is_none());
            server.abort();
        }
    }

    #[tokio::test]
    async fn healthy_chat_has_no_overall_deadline_but_helper_releases_permit() {
        use futures_util::StreamExt;
        for helper in [false, true] {
            let mut writes = vec![(Duration::ZERO, SSE_HEADERS)];
            writes.extend((0..30).map(|_| (Duration::from_millis(40), SSE_CHUNK)));
            writes.push((Duration::ZERO, b"0\r\n\r\n"));
            let (url, server) = scripted_upstream(writes, false).await;
            let state = state_with_backend(InferenceBackend::LlamaCpp, "healthy", 8101);
            let prepared = prepare_inference_request(&state).await.unwrap();
            let client = build_upstream_client().unwrap();
            let policy = test_timeouts();
            let result = tokio::time::timeout(Duration::from_secs(3), async {
                let _permit = prepared.permit;
                let deadline = helper.then(|| tokio::time::Instant::now() + policy.helper);
                let response =
                    send_upstream_request_with_timeouts(|| client.post(&url), policy, deadline)
                        .await?;
                let mut body = upstream_body_stream_with_timeouts(response, policy);
                let mut bytes = 0;
                while let Some(chunk) = body.next().await {
                    bytes += chunk.map_err(warp::reject::custom)?.len();
                }
                Ok::<_, warp::Rejection>(bytes)
            })
            .await
            .expect("helper overall deadline must be bounded");
            if helper {
                assert_eq!(
                    result.unwrap_err().find::<ApiError>().unwrap().status,
                    StatusCode::GATEWAY_TIMEOUT
                );
            } else {
                assert_eq!(result.unwrap(), 330);
            }
            assert_eq!(state.monitor_inference_gate.available_permits(), 1);
            server.abort();
        }
    }

    #[tokio::test]
    async fn silent_error_body_cannot_hold_inference_permit() {
        let (url, server) = scripted_upstream(
            vec![(
                Duration::ZERO,
                b"HTTP/1.1 500 Internal Server Error\r\nTransfer-Encoding: chunked\r\n\r\n",
            )],
            true,
        )
        .await;
        let client = build_upstream_client().unwrap();
        let state = state_with_backend(InferenceBackend::LlamaCpp, "error-body", 8101);
        let prepared = prepare_inference_request(&state).await.unwrap();
        let result = tokio::time::timeout(Duration::from_secs(2), async {
            let _permit = prepared.permit;
            send_upstream_request_with_timeouts(|| client.post(&url), test_timeouts(), None).await
        })
        .await
        .expect("error body wait must be bounded");
        assert_eq!(
            result.unwrap_err().find::<ApiError>().unwrap().status,
            StatusCode::GATEWAY_TIMEOUT
        );
        assert_eq!(state.monitor_inference_gate.available_permits(), 1);
        server.abort();
    }

    #[tokio::test]
    async fn helper_deadline_includes_retry_backoff() {
        let (url, server) = scripted_upstream(
            vec![(
                Duration::ZERO,
                b"HTTP/1.1 429 Too Many Requests\r\nContent-Length: 4\r\n\r\nbusy",
            )],
            false,
        )
        .await;
        let client = build_upstream_client().unwrap();
        let mut calls = 0;
        let result = send_upstream_request_with_timeouts(
            || {
                calls += 1;
                client.post(&url)
            },
            test_timeouts(),
            Some(tokio::time::Instant::now() + Duration::from_millis(100)),
        )
        .await;
        assert_eq!(
            result.unwrap_err().find::<ApiError>().unwrap().status,
            StatusCode::GATEWAY_TIMEOUT
        );
        assert_eq!(calls, 1);
        server.await.unwrap();
    }
}
