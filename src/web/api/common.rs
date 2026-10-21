use std::fmt;
use std::sync::Arc;

use warp::Filter;
use warp::http::StatusCode;
use warp::reject::Reject;

use crate::chat_storage::ChatStorage;
use crate::config::AppConfig;
use crate::state::AppState;
use crate::web::auth::AuthManager;

pub(crate) type ApiReply = Box<dyn warp::reply::Reply>;
pub(crate) type ApiRoute = warp::filters::BoxedFilter<(ApiReply,)>;

#[derive(Clone)]
pub(crate) struct ApiCtx {
    pub(crate) state: AppState,
    pub(crate) config: Arc<AppConfig>,
    pub(crate) auth: AuthManager,
}

#[derive(Debug)]
pub(crate) struct ApiError {
    pub(crate) status: StatusCode,
    pub(crate) message: String,
    pub(crate) public_code: Option<String>,
}

impl ApiError {
    pub(crate) fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
            public_code: None,
        }
    }

    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, message)
    }

    pub(crate) fn busy(message: impl Into<String>) -> Self {
        Self::new(StatusCode::TOO_MANY_REQUESTS, message)
    }

    pub(crate) fn gateway(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_GATEWAY, message)
    }

    pub(crate) fn gateway_timeout(message: impl Into<String>) -> Self {
        Self::new(StatusCode::GATEWAY_TIMEOUT, message)
    }

    pub(crate) fn migration(error: &anyhow::Error) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: error.to_string(),
            public_code: Some(
                serde_json::to_string(&crate::app_migration::public_error_code(error))
                    .unwrap_or_else(|_| "\"io_failure\"".to_string())
                    .trim_matches('"')
                    .to_string(),
            ),
        }
    }

    pub(crate) fn from_reqwest(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            return Self::gateway_timeout("Timed out waiting for the active inference runtime.");
        }

        if err.is_connect() {
            return Self::gateway("Cannot connect to the active inference runtime.");
        }

        let detail = err.to_string();
        if detail.contains("error sending request") || detail.contains("connection reset") {
            return Self::gateway(
                "The active inference runtime dropped the request before streaming started.",
            );
        }

        Self::gateway(format!("Upstream request failed: {detail}"))
    }

    pub(crate) fn from_upstream_status(status: StatusCode, body: String) -> Self {
        let detail = body.trim();
        let lower = detail.to_ascii_lowercase();
        if status == StatusCode::TOO_MANY_REQUESTS
            || lower.contains("busy")
            || lower.contains("no slot")
            || lower.contains("no available slot")
        {
            let message = if detail.is_empty() {
                "The active inference runtime is busy with another request."
            } else {
                detail
            };
            return Self::busy(message.to_string());
        }

        let message = if detail.is_empty() {
            format!(
                "Upstream inference runtime returned HTTP {}.",
                status.as_u16()
            )
        } else {
            format!("Upstream HTTP {}: {detail}", status.as_u16())
        };
        Self::new(status, message)
    }
}

impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ApiError {}

impl Reject for ApiError {}

/// Extract bearer token from the Authorization header.
pub(crate) fn extract_bearer(auth: Option<String>) -> Option<String> {
    auth.and_then(|v| v.strip_prefix("Bearer ").map(str::to_string))
}

/// 401 JSON reply for a missing api-token.
pub(crate) fn unauthorized_api_token() -> Box<dyn warp::reply::Reply> {
    Box::new(warp::reply::with_status(
        warp::reply::json(&serde_json::json!({
            "ok": false,
            "error": "unauthorized; api-token required"
        })),
        StatusCode::UNAUTHORIZED,
    ))
}

/// 401 JSON reply for a missing db-admin-token.
pub(crate) fn unauthorized_db_admin_token() -> Box<dyn warp::reply::Reply> {
    Box::new(warp::reply::with_status(
        warp::reply::json(&serde_json::json!({
            "ok": false,
            "error": "unauthorized; db-admin-token required"
        })),
        StatusCode::UNAUTHORIZED,
    ))
}

/// Check if the Authorization header matches the configured api-token.
pub fn check_api_token(auth: &Option<String>, cfg: &AppConfig) -> bool {
    let bearer = auth.as_ref().and_then(|v| v.strip_prefix("Bearer "));
    bearer_matches_api_token(bearer, cfg)
}

/// Gate for elevated, high-impact endpoints. The db-admin-token is required
/// whenever one is configured; the api-token alone never suffices then. With
/// no admin token (local-first mode) it falls back to the api-token, so an
/// unset admin token cannot open a route the api-token would have closed.
pub fn check_elevated_token(auth: &Option<String>, cfg: &AppConfig) -> bool {
    let admin_configured = cfg.live_db_admin_token().is_some_and(|t| !t.is_empty());
    if admin_configured {
        check_db_admin_token(auth, cfg)
    } else {
        check_api_token(auth, cfg)
    }
}

/// Check if the Authorization header matches the configured db-admin-token.
pub(crate) fn check_db_admin_token(auth: &Option<String>, cfg: &AppConfig) -> bool {
    let bearer = auth.as_ref().and_then(|v| v.strip_prefix("Bearer "));
    bearer_matches_db_admin_token(bearer, cfg)
}

/// Compare an already-extracted bearer token against the live api-token (constant-time).
/// If no api-token is configured, allow the request (local-first mode).
/// If api-token is configured but no bearer is provided, reject.
pub(crate) fn bearer_matches_api_token(bearer: Option<&str>, cfg: &AppConfig) -> bool {
    use subtle::ConstantTimeEq;

    let live = cfg.live_api_token();
    match (bearer, live.as_deref()) {
        (Some(got), Some(expected)) if !expected.is_empty() => {
            got.as_bytes().ct_eq(expected.as_bytes()).into()
        }
        // Token is configured but no bearer provided -> reject.
        (None, Some(expected)) if !expected.is_empty() => false,
        // No token configured -> allow (local-first mode).
        _ => true,
    }
}

/// Compare an already-extracted bearer token against the live db-admin-token (constant-time).
/// If no db-admin-token is configured, allow the request (local-first mode).
/// If db-admin-token is configured but no bearer is provided, reject.
pub(crate) fn bearer_matches_db_admin_token(bearer: Option<&str>, cfg: &AppConfig) -> bool {
    use subtle::ConstantTimeEq;

    let live = cfg.live_db_admin_token();
    match (bearer, live.as_deref()) {
        (Some(got), Some(expected)) if !expected.is_empty() => {
            got.as_bytes().ct_eq(expected.as_bytes()).into()
        }
        // Token is configured but no bearer provided -> reject.
        (None, Some(expected)) if !expected.is_empty() => false,
        // No token configured -> allow (local-first mode).
        _ => true,
    }
}

/// Atomic cooldown helper. Returns `(ok, seconds_remaining)`.
/// `ok` is true if the cooldown has elapsed and the timestamp was successfully CAS'd.
pub(crate) fn try_cooldown(
    last: &std::sync::atomic::AtomicU64,
    now: u64,
    cooldown_secs: u64,
) -> (bool, u64) {
    use std::sync::atomic::Ordering;
    let prev = last.load(Ordering::Acquire);
    let elapsed = now.saturating_sub(prev);
    if elapsed < cooldown_secs {
        return (false, cooldown_secs - elapsed);
    }
    let ok = last
        .compare_exchange(prev, now, Ordering::AcqRel, Ordering::Relaxed)
        .is_ok();
    (ok, 0)
}

pub(crate) fn with_app_config(
    cfg: Arc<AppConfig>,
) -> impl Filter<Extract = (Arc<AppConfig>,), Error = std::convert::Infallible> + Clone {
    warp::any().map(move || cfg.clone())
}

/// Helper: record activity timestamp and wake from auto-sleep if needed.
/// Manual sleep (user-toggled) is not interrupted here — only auto-sleep is.
pub(crate) fn record_activity(state: &AppState) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    state
        .last_activity_at
        .store(now, std::sync::atomic::Ordering::Relaxed);

    // Wake-on-activity: only wake auto-sleep, not manual sleep (T-051)
    let is_manual = state
        .sleep_mode_manual
        .load(std::sync::atomic::Ordering::Relaxed);
    let mode = state.sleep_mode.load(std::sync::atomic::Ordering::Relaxed);
    if !is_manual && mode > 0 {
        state
            .sleep_mode
            .store(0, std::sync::atomic::Ordering::Relaxed);
        state.sleep_notify.notify_waiters();
    }
}

/// Box a reply into the API envelope type.
pub(crate) fn box_reply<R: warp::Reply + 'static>(reply: R) -> ApiReply {
    Box::new(reply)
}

/// Warp filter that provides Arc<ChatStorage> to API routes.
pub(crate) fn with_chat_storage(
    cs: Arc<ChatStorage>,
) -> impl Filter<Extract = (Arc<ChatStorage>,), Error = std::convert::Infallible> + Clone + Unpin {
    warp::any().map(move || cs.clone())
}
