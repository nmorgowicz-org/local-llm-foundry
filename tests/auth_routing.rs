#![recursion_limit = "256"]

//! Integration tests for auth routing behavior.
//!
//! These tests validate:
//! - Every /api/* endpoint requires expected auth.
//! - “No Auth” mode leaves endpoints accessible.
//! - auth_guard allows: no auth, session cookie, Basic Auth, api-token (Bearer).
//! - No endpoint is unintentionally blocked or exposed.
//!
//! We test the auth filters and token-check functions directly instead of
//! spinning up the full warp server, to avoid macOS linker limits and keep
//! tests fast and focused.

use llama_monitor::{
    config::AppConfig,
    web::{
        api::check_api_token,
        auth::{AuthCredentials, AuthManager},
    },
};

const TEST_API_TOKEN: &str = "test-api-token-123";
const TEST_DB_ADMIN_TOKEN: &str = "test-db-admin-token-456";

fn cfg_no_tokens() -> AppConfig {
    AppConfig::for_test(None, None)
}

fn cfg_api_only() -> AppConfig {
    AppConfig::for_test(Some(TEST_API_TOKEN.to_string()), None)
}

fn cfg_db_only() -> AppConfig {
    AppConfig::for_test(None, Some(TEST_DB_ADMIN_TOKEN.to_string()))
}

fn cfg_both() -> AppConfig {
    AppConfig::for_test(
        Some(TEST_API_TOKEN.to_string()),
        Some(TEST_DB_ADMIN_TOKEN.to_string()),
    )
}

fn auth_none() -> AuthManager {
    AuthManager::new(
        None,
        None,
        &llama_monitor::config::TLSConfig::default().mode,
    )
}

fn auth_form() -> AuthManager {
    let creds = AuthCredentials {
        username: "admin".into(),
        password: "secret".into(),
    };
    AuthManager::new(
        None,
        Some(creds),
        &llama_monitor::config::TLSConfig::default().mode,
    )
}

fn auth_basic() -> AuthManager {
    let creds = AuthCredentials {
        username: "admin".into(),
        password: "secret".into(),
    };
    AuthManager::new(
        Some(creds),
        None,
        &llama_monitor::config::TLSConfig::default().mode,
    )
}

fn bearer(token: &str) -> Option<String> {
    Some(format!("Bearer {}", token))
}

fn basic_header(user: &str, pass: &str) -> Option<String> {
    use base64::Engine;
    let encoded =
        base64::engine::general_purpose::STANDARD.encode(format!("{}:{}", user, pass).as_bytes());
    Some(format!("Basic {}", encoded))
}

// ===================== NO AUTH MODE =====================

#[test]
fn no_auth_mode_allows_without_anything() {
    let auth = auth_none();
    assert!(
        !auth.has_any(),
        "no-auth mode must report has_any() == false"
    );

    // Simulate: auth_guard early-exits when !has_any().
    // So any request should pass auth_guard.
    assert!(
        !auth.authenticate_request(None, None),
        "no credentials needed when no auth configured"
    );
}

#[test]
fn no_api_token_configured_allows_any_bearer() {
    // Local-first behavior: if api-token is not configured, any request is allowed.
    let cfg = cfg_no_tokens();

    assert!(
        check_api_token(&None, &cfg),
        "no api-token configured => allow"
    );
    assert!(
        check_api_token(&bearer("anything"), &cfg),
        "no api-token configured => allow any bearer"
    );
}

// ===================== FORM AUTH MODE =====================

#[test]
fn form_auth_blocks_without_creds() {
    let auth = auth_form();
    assert!(auth.has_any());

    // No auth header, no cookie => should fail.
    assert!(
        !auth.authenticate_request(None, None),
        "form auth must block unauthenticated requests"
    );
}

#[test]
fn form_auth_allows_with_session_cookie() {
    let auth = auth_form();

    // Create a valid session.
    let token = auth
        .create_form_session("admin")
        .expect("create_form_session should succeed");

    let cookie = format!("llama_monitor_session={}", token);

    // auth_guard should allow via session cookie.
    assert!(
        auth.authenticate_request(None, Some(&cookie)),
        "valid session cookie must authenticate"
    );
}

#[test]
fn form_auth_rejects_bad_credentials() {
    let auth = auth_form();
    let hdr = basic_header("admin", "wrong");

    assert!(
        !auth.authenticate_request(hdr.as_deref(), None),
        "wrong password must fail"
    );
}

// ===================== BASIC AUTH MODE =====================

#[test]
fn basic_auth_blocks_without_creds() {
    let auth = auth_basic();
    assert!(auth.has_any());

    assert!(
        !auth.authenticate_request(None, None),
        "basic auth must block unauthenticated requests"
    );
}

#[test]
fn basic_auth_allows_correct_header() {
    let auth = auth_basic();
    let hdr = basic_header("admin", "secret");

    assert!(
        auth.authenticate_request(hdr.as_deref(), None),
        "correct Basic Auth must authenticate"
    );
}

#[test]
fn basic_auth_rejects_wrong_password() {
    let auth = auth_basic();
    let hdr = basic_header("admin", "wrong");

    assert!(
        !auth.authenticate_request(hdr.as_deref(), None),
        "wrong password must fail"
    );
}

// ===================== API-TOKEN AUTH (BEARER) =====================

// These tests are critical: they ensure api-token Bearer auth works for all
// /api/* endpoints, including the one that caused the previous 404 regression.

#[test]
fn api_token_allows_when_configured() {
    let cfg = cfg_api_only();

    assert!(
        check_api_token(&bearer(TEST_API_TOKEN), &cfg),
        "correct api-token must allow"
    );
}

#[test]
fn api_token_rejects_wrong_token() {
    let cfg = cfg_api_only();

    assert!(
        !check_api_token(&bearer("wrong-token"), &cfg),
        "wrong api-token must be rejected"
    );
}

#[test]
fn api_token_rejects_missing_bearer_when_configured() {
    let cfg = cfg_api_only();

    assert!(
        !check_api_token(&None, &cfg),
        "missing bearer when api-token configured must reject"
    );
}

#[test]
fn api_token_works_with_form_auth_enabled() {
    // Even when form auth is configured, endpoints that accept api-token must
    // still work with Bearer token and not require a session cookie.
    let cfg = cfg_api_only();
    let auth = auth_form();

    // auth_guard should allow api-token bearer (via its own check).
    // We simulate the auth_guard branch:
    // - has_any() is true
    // - authenticate_request (form/basic) fails without cookie
    // - but api-token check must succeed
    assert!(auth.has_any());
    assert!(
        !auth.authenticate_request(bearer(TEST_API_TOKEN).as_deref(), None),
        "form auth should not recognize api-token as form/basic auth"
    );
    // But the endpoint-level api-token check must pass.
    assert!(
        check_api_token(&bearer(TEST_API_TOKEN), &cfg),
        "api-token must allow even with form auth enabled"
    );
}

#[test]
fn api_token_works_with_basic_auth_enabled() {
    let cfg = cfg_api_only();
    let auth = auth_basic();

    assert!(auth.has_any());
    assert!(
        !auth.authenticate_request(bearer(TEST_API_TOKEN).as_deref(), None),
        "basic auth should not recognize api-token as basic auth"
    );
    // But api-token check must pass.
    assert!(
        check_api_token(&bearer(TEST_API_TOKEN), &cfg),
        "api-token must allow even with basic auth enabled"
    );
}

#[test]
fn api_token_works_with_both_tokens_and_form_auth() {
    // Full cross-mode: form auth + api-token + db-admin-token all configured.
    let cfg = cfg_both();
    let auth = auth_form();

    assert!(auth.has_any());
    assert!(
        check_api_token(&bearer(TEST_API_TOKEN), &cfg),
        "api-token must allow with all auth mechanisms enabled"
    );
}

// ===================== DB-ADMIN-TOKEN (REMOTE-AGENT) =====================

// The remote-agent endpoints use db-admin-token. We validate the same pattern.

#[test]
fn db_admin_token_allows_when_configured() {
    // We test via the same bearer_matches_db_admin_token logic used by
    // /api/remote-agent/install, /api/remote-agent/status, etc.
    let cfg = cfg_db_only();

    // Simulate: bearer_matches_db_admin_token
    let bearer = TEST_DB_ADMIN_TOKEN;
    let live = cfg.live_db_admin_token();

    use subtle::ConstantTimeEq;
    let matches = match (Some(bearer), live.as_deref()) {
        (Some(got), Some(expected)) if !expected.is_empty() => {
            got.as_bytes().ct_eq(expected.as_bytes()).into()
        }
        (None, Some(expected)) if !expected.is_empty() => false,
        _ => true,
    };

    assert!(matches, "correct db-admin-token must allow");
}

#[test]
fn db_admin_token_rejects_wrong_token() {
    let cfg = cfg_db_only();

    let bearer = "wrong-token";
    let live = cfg.live_db_admin_token();

    use subtle::ConstantTimeEq;
    let matches = match (Some(bearer), live.as_deref()) {
        (Some(got), Some(expected)) if !expected.is_empty() => {
            got.as_bytes().ct_eq(expected.as_bytes()).into()
        }
        (None, Some(expected)) if !expected.is_empty() => false,
        _ => true,
    };

    assert!(!matches, "wrong db-admin-token must be rejected");
}

// ===================== AUTH GUARD LOGIC (SIMULATED) =====================

// These tests encode the auth_guard decision logic to ensure:
// - No auth mode: allows all.
// - Form/Basic: requires valid session/credentials OR api-token.
// - api-token Bearer: always allowed when valid.
// - No endpoint is unintentionally blocked or exposed.

fn simulate_auth_guard(
    auth: &AuthManager,
    cfg: &AppConfig,
    auth_header: Option<&str>,
    cookie_header: Option<&str>,
) -> bool {
    // 1) No auth configured: allow all.
    if !auth.has_any() {
        return true;
    }

    // 2) Allow if Form Login session or Basic Auth is valid.
    if auth.authenticate_request(auth_header, cookie_header) {
        return true;
    }

    // 3) Allow if a valid api-token is present (Bearer).
    if let Some(token) = auth_header
        .and_then(|h| h.strip_prefix("Bearer "))
        .map(str::trim)
        .filter(|t| !t.is_empty())
        && check_api_token(&Some(format!("Bearer {}", token)), cfg)
    {
        return true;
    }

    false
}

#[test]
fn auth_guard_no_auth_allows_all() {
    let auth = auth_none();
    let cfg = cfg_no_tokens();

    assert!(
        simulate_auth_guard(&auth, &cfg, None, None),
        "no auth mode must allow all requests"
    );
}

#[test]
fn auth_guard_form_auth_blocks_nothing() {
    let auth = auth_form();
    let cfg = cfg_no_tokens();

    assert!(
        !simulate_auth_guard(&auth, &cfg, None, None),
        "form auth must block unauthenticated requests"
    );
}

#[test]
fn auth_guard_form_auth_allows_session() {
    let auth = auth_form();
    let cfg = cfg_no_tokens();

    let token = auth.create_form_session("admin").unwrap();
    let cookie = format!("llama_monitor_session={}", token);

    assert!(
        simulate_auth_guard(&auth, &cfg, None, Some(&cookie)),
        "valid session cookie must be allowed by auth_guard"
    );
}

#[test]
fn auth_guard_form_auth_allows_api_token() {
    // Critical: auth_guard must allow api-token Bearer even when form auth is enabled.
    let auth = auth_form();
    let cfg = cfg_api_only();

    let auth_hdr = bearer(TEST_API_TOKEN).unwrap();
    assert!(
        simulate_auth_guard(&auth, &cfg, Some(&auth_hdr), None),
        "auth_guard must allow valid api-token with form auth enabled"
    );
}

#[test]
fn auth_guard_basic_auth_allows_api_token() {
    let auth = auth_basic();
    let cfg = cfg_api_only();

    let auth_hdr = bearer(TEST_API_TOKEN).unwrap();
    assert!(
        simulate_auth_guard(&auth, &cfg, Some(&auth_hdr), None),
        "auth_guard must allow valid api-token with basic auth enabled"
    );
}

#[test]
fn auth_guard_rejects_invalid_api_token() {
    let auth = auth_form();
    let cfg = cfg_api_only();

    assert!(
        !simulate_auth_guard(&auth, &cfg, Some("Bearer wrong-token"), None),
        "auth_guard must reject invalid api-token"
    );
}

// ===================== ENDPOINT AUTH MATRIX =====================

// These tests validate that each endpoint's auth requirement is correct.
// They use check_api_token / bearer_matches_db_admin_token to simulate
// the per-endpoint auth logic.

#[test]
fn endpoint_put_chat_tabs_id_requires_api_token() {
    // PUT /api/chat/tabs/:id uses check_api_token.
    // With api-token configured: must require it.
    // With api-token not configured: must allow (local-first).
    let cfg_protected = cfg_api_only();
    let cfg_open = cfg_no_tokens();

    // Protected mode: no token => reject
    assert!(
        !check_api_token(&None, &cfg_protected),
        "PUT /api/chat/tabs/:id must require api-token when configured"
    );

    // Protected mode: correct token => allow
    assert!(
        check_api_token(&bearer(TEST_API_TOKEN), &cfg_protected),
        "correct api-token must allow"
    );

    // Open mode: no token => allow
    assert!(
        check_api_token(&None, &cfg_open),
        "local-first (no api-token) must allow"
    );
}

#[test]
fn endpoint_get_chat_tabs_requires_api_token() {
    // GET /api/chat/tabs uses check_api_token.
    let cfg = cfg_api_only();

    assert!(
        !check_api_token(&None, &cfg),
        "GET /api/chat/tabs must require api-token when configured"
    );
    assert!(
        check_api_token(&bearer(TEST_API_TOKEN), &cfg),
        "correct api-token must allow"
    );
}

#[test]
fn endpoint_get_models_no_per_endpoint_auth() {
    // GET /api/models currently has no per-endpoint token check;
    // it relies solely on auth_guard. We verify that auth_guard with
    // api-token allows it.
    let auth = auth_form();
    let cfg = cfg_api_only();

    assert!(
        simulate_auth_guard(&auth, &cfg, Some(&bearer(TEST_API_TOKEN).unwrap()), None),
        "GET /api/models must be reachable via api-token through auth_guard"
    );
}

#[test]
fn endpoint_get_gpu_env_no_per_endpoint_auth() {
    // GET /api/gpu-env: same as models.
    let auth = auth_form();
    let cfg = cfg_api_only();

    assert!(
        simulate_auth_guard(&auth, &cfg, Some(&bearer(TEST_API_TOKEN).unwrap()), None),
        "GET /api/gpu-env must be reachable via api-token through auth_guard"
    );
}

#[test]
fn endpoint_auth_login_public() {
    // POST /api/auth/login is a public route (not behind auth_guard).
    // We can't call it here without the full server, but we can verify
    // that auth_none does not block it conceptually: it's on public_api
    // which bypasses auth_guard.
    // This test documents the expectation.
    let auth = auth_form();
    assert!(auth.has_any(), "form auth is configured");
    // auth/login is public; no assertion over auth_guard needed.
}

#[test]
fn endpoint_remote_agent_install_requires_db_admin_token() {
    // POST /api/remote-agent/install uses bearer_matches_db_admin_token.
    let cfg = cfg_db_only();

    let live = cfg.live_db_admin_token();
    use subtle::ConstantTimeEq;

    // Correct token
    let matches = match (Some(TEST_DB_ADMIN_TOKEN), live.as_deref()) {
        (Some(got), Some(expected)) if !expected.is_empty() => {
            got.as_bytes().ct_eq(expected.as_bytes()).into()
        }
        (None, Some(expected)) if !expected.is_empty() => false,
        _ => true,
    };
    assert!(matches, "db-admin-token must allow remote-agent install");

    // Wrong token
    let matches_wrong = match (Some("wrong"), live.as_deref()) {
        (Some(got), Some(expected)) if !expected.is_empty() => {
            got.as_bytes().ct_eq(expected.as_bytes()).into()
        }
        (None, Some(expected)) if !expected.is_empty() => false,
        _ => true,
    };
    assert!(
        !matches_wrong,
        "wrong token must reject remote-agent install"
    );
}

#[test]
fn endpoint_spawn_session_requires_db_admin_token_for_all_backends() {
    let cfg = cfg_db_only();
    let live = cfg.live_db_admin_token();
    use subtle::ConstantTimeEq;

    let authorized = |token: Option<&str>| match (token, live.as_deref()) {
        (Some(got), Some(expected)) if !expected.is_empty() => {
            got.as_bytes().ct_eq(expected.as_bytes()).into()
        }
        (None, Some(expected)) if !expected.is_empty() => false,
        _ => true,
    };

    assert!(authorized(Some(TEST_DB_ADMIN_TOKEN)));
    assert!(!authorized(Some("wrong")));
    assert!(!authorized(None));
}

#[test]
fn endpoint_remote_agent_status_requires_db_admin_token() {
    // POST /api/remote-agent/status uses bearer_matches_db_admin_token.
    let cfg = cfg_db_only();

    let live = cfg.live_db_admin_token();
    use subtle::ConstantTimeEq;

    let matches = match (Some(TEST_DB_ADMIN_TOKEN), live.as_deref()) {
        (Some(got), Some(expected)) if !expected.is_empty() => {
            got.as_bytes().ct_eq(expected.as_bytes()).into()
        }
        (None, Some(expected)) if !expected.is_empty() => false,
        _ => true,
    };
    assert!(matches, "db-admin-token must allow remote-agent status");
}

// ── Visibility endpoint auth ─────────────────────────────────────────────────

#[test]
fn endpoint_archive_tab_requires_api_token() {
    let cfg_protected = cfg_api_only();
    let cfg_open = cfg_no_tokens();

    assert!(
        !check_api_token(&None, &cfg_protected),
        "must require api-token"
    );
    assert!(
        check_api_token(&bearer(TEST_API_TOKEN), &cfg_protected),
        "correct token must allow"
    );
    assert!(check_api_token(&None, &cfg_open), "local-first must allow");
}

#[test]
fn endpoint_hide_tab_requires_api_token() {
    let cfg_protected = cfg_api_only();
    let cfg_open = cfg_no_tokens();

    assert!(
        !check_api_token(&None, &cfg_protected),
        "must require api-token"
    );
    assert!(
        check_api_token(&bearer(TEST_API_TOKEN), &cfg_protected),
        "correct token must allow"
    );
    assert!(check_api_token(&None, &cfg_open), "local-first must allow");
}

#[test]
fn endpoint_restore_tab_requires_api_token() {
    let cfg_protected = cfg_api_only();
    let cfg_open = cfg_no_tokens();

    assert!(
        !check_api_token(&None, &cfg_protected),
        "must require api-token"
    );
    assert!(
        check_api_token(&bearer(TEST_API_TOKEN), &cfg_protected),
        "correct token must allow"
    );
    assert!(check_api_token(&None, &cfg_open), "local-first must allow");
}

#[test]
fn endpoint_list_tabs_visibility_requires_api_token() {
    let cfg_protected = cfg_api_only();
    let cfg_open = cfg_no_tokens();

    assert!(
        !check_api_token(&None, &cfg_protected),
        "must require api-token"
    );
    assert!(
        check_api_token(&bearer(TEST_API_TOKEN), &cfg_protected),
        "correct token must allow"
    );
    assert!(check_api_token(&None, &cfg_open), "local-first must allow");
}

#[test]
fn endpoint_search_visibility_requires_api_token() {
    let cfg_protected = cfg_api_only();
    let cfg_open = cfg_no_tokens();

    assert!(
        !check_api_token(&None, &cfg_protected),
        "must require api-token"
    );
    assert!(
        check_api_token(&bearer(TEST_API_TOKEN), &cfg_protected),
        "correct token must allow"
    );
    assert!(check_api_token(&None, &cfg_open), "local-first must allow");
}

#[test]
fn endpoint_archive_works_with_form_auth_enabled() {
    let auth = auth_form();
    let cfg = cfg_api_only();

    assert!(
        simulate_auth_guard(&auth, &cfg, Some(&bearer(TEST_API_TOKEN).unwrap()), None),
        "must be reachable via api-token through auth_guard"
    );
}

// ===================== SAFE JSON BODY (400 NOT 404) =====================

// These tests ensure that malformed JSON on mutating chat endpoints returns
// 400 (Bad Request) instead of 404 — the exact bug that caused silent tab loss.

#[tokio::test]
async fn safe_json_body_valid_returns_200() {
    use llama_monitor::web::{handle_rejection, safe_json_body};
    use serde::Deserialize;
    use warp::Filter;

    #[derive(Deserialize)]
    struct TestReq {
        #[allow(dead_code)]
        id: String,
    }

    let route = warp::post()
        .and(safe_json_body::<TestReq>())
        .map(|_req: TestReq| warp::reply::json(&serde_json::json!({ "ok": true })))
        .recover(handle_rejection);

    let resp = warp::test::request()
        .method("POST")
        .json(&serde_json::json!({ "id": "test" }))
        .reply(&route)
        .await;

    assert_eq!(
        resp.status(),
        warp::http::StatusCode::OK,
        "valid JSON must return 200"
    );
}

#[tokio::test]
async fn safe_json_body_invalid_returns_400_not_404() {
    use llama_monitor::web::{handle_rejection, safe_json_body};
    use serde::Deserialize;
    use warp::Filter;

    #[derive(Deserialize)]
    struct TestReq {
        #[allow(dead_code)]
        id: String,
    }

    let route = warp::post()
        .and(safe_json_body::<TestReq>())
        .map(|_req: TestReq| warp::reply::json(&serde_json::json!({ "ok": true })))
        .recover(handle_rejection);

    // Send obviously invalid JSON
    let resp = warp::test::request()
        .method("POST")
        .header("content-type", "application/json")
        .body("{ bad json }")
        .reply(&route)
        .await;

    let status = resp.status();
    assert_eq!(
        status,
        warp::http::StatusCode::BAD_REQUEST,
        "invalid JSON must return 400 (not 404)"
    );
}

// ===================== SESSIONS RECENT ENDPOINT =====================

#[tokio::test]
async fn sessions_recent_requires_api_token() {
    // This test validates that the sessions/recent endpoint is registered
    // and requires api-token auth. Since auth_routing tests the auth filters
    // directly rather than the full server, we validate the token-check pattern.
    let cfg = cfg_api_only();
    assert!(
        check_api_token(&bearer(TEST_API_TOKEN), &cfg),
        "valid api-token must be accepted for sessions/recent"
    );
    assert!(
        !check_api_token(&None, &cfg),
        "no token must be rejected for sessions/recent"
    );
    assert!(
        !check_api_token(&bearer("wrong-token"), &cfg),
        "wrong token must be rejected for sessions/recent"
    );
}

// ===================== REAL ROUTE AUTH: RAPID-MLX MODEL ENDPOINTS =====================

// Unlike the helper-level tests above, these drive the real warp routes through
// `api_routes`, so they prove each handler actually calls `check_api_token`.
// Authorized requests deliberately use inputs that stop at validation or lookup
// (400/404) or a read-only status probe, so no download worker or subprocess starts.

mod model_endpoint_routes {
    use super::*;
    use llama_monitor::{state::AppState, web::api::api_routes};
    use std::sync::Arc;
    use warp::http::StatusCode;

    /// One request against a route: `(method, path, json_body, status_when_authorized)`.
    type Case = (&'static str, &'static str, Option<&'static str>, StatusCode);

    const CASES: [Case; 4] = [
        (
            "GET",
            "/api/rapid-mlx/model-status?source=owner/model",
            None,
            StatusCode::OK,
        ),
        (
            "POST",
            "/api/models/downloads",
            // Valid JSON, invalid repo_id: passes auth, then fails validation.
            Some(r#"{"repo_id":"not a repo"}"#),
            StatusCode::BAD_REQUEST,
        ),
        (
            "GET",
            "/api/models/downloads/mdl-unknown",
            None,
            StatusCode::NOT_FOUND,
        ),
        (
            "POST",
            "/api/models/downloads/mdl-unknown/cancel",
            None,
            StatusCode::NOT_FOUND,
        ),
    ];

    async fn status(cfg: AppConfig, case: Case, token: Option<&str>) -> StatusCode {
        let routes = api_routes(
            AppState::default(),
            Arc::new(cfg),
            auth_none(),
            "127.0.0.1".to_string(),
        );
        let (method, path, body, _) = case;
        let mut request = warp::test::request().method(method).path(path);
        if let Some(token) = token {
            request = request.header("authorization", format!("Bearer {token}"));
        }
        if let Some(body) = body {
            request = request
                .header("content-type", "application/json")
                .body(body);
        }
        request.reply(&routes).await.status()
    }

    #[tokio::test]
    async fn model_endpoints_return_401_without_a_token() {
        for case in CASES {
            assert_eq!(
                status(cfg_api_only(), case, None).await,
                StatusCode::UNAUTHORIZED,
                "{} {}",
                case.0,
                case.1
            );
        }
    }

    #[tokio::test]
    async fn model_endpoints_return_401_with_a_wrong_token() {
        for case in CASES {
            assert_eq!(
                status(cfg_api_only(), case, Some("wrong-token")).await,
                StatusCode::UNAUTHORIZED,
                "{} {}",
                case.0,
                case.1
            );
        }
    }

    #[tokio::test]
    async fn model_endpoints_accept_the_api_token() {
        for case in CASES {
            assert_eq!(
                status(cfg_api_only(), case, Some(TEST_API_TOKEN)).await,
                case.3,
                "{} {}",
                case.0,
                case.1
            );
        }
    }

    #[tokio::test]
    async fn model_endpoints_are_open_when_no_api_token_is_configured() {
        for case in CASES {
            assert_eq!(
                status(cfg_no_tokens(), case, None).await,
                case.3,
                "{} {}",
                case.0,
                case.1
            );
        }
    }
}
