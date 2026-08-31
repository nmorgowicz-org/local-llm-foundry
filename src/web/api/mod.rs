use std::sync::Arc;

use warp::Filter;

mod app_home_migration;
mod auth;
mod benchmark;
mod browse;
mod calibration;
mod chat;
pub(crate) mod system_tools;
#[cfg(test)]
pub(crate) use chat::legacy_chat_types;
mod common;
mod config;
mod db;
mod debug;
mod doctor;
mod hf;
mod lhm;
mod llama_binary;
mod metrics;
mod models;
mod preset_bundles;
#[path = "presets.rs"]
mod preset_routes;
mod rapid_mlx_repair;
mod rapid_mlx_runtime;
mod remote_agent;
mod self_update;
mod sensor_bridge;
mod sessions;
mod sleep;
mod spawn_wizard;
mod templates;
mod tls;
mod tokens;
pub(crate) mod upstream;
mod vram;

pub(crate) use common::ApiError;
pub use common::check_api_token;
pub(crate) use common::{ApiCtx, ApiReply, ApiRoute, box_reply, record_activity};
pub(crate) use common::{
    bearer_matches_api_token, bearer_matches_db_admin_token, check_db_admin_token, extract_bearer,
    unauthorized_api_token, unauthorized_db_admin_token, with_app_config,
};
pub use tokens::public_tokens_routes;
#[cfg(test)]
use tokens::token_bootstrap_allowed;

use crate::config::AppConfig;
use crate::state::AppState;
use crate::web::auth::AuthManager;

pub fn api_routes(
    state: AppState,
    app_config: Arc<AppConfig>,
    auth_manager: AuthManager,
    _bind_host: String,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    let ctx = ApiCtx {
        state: state.clone(),
        config: app_config.clone(),
        auth: auth_manager.clone(),
    };

    let preset_routes = preset_routes::routes(ctx.clone());
    let preset_bundle_routes = preset_bundles::routes(ctx.clone());
    let app_home_migration_routes = app_home_migration::routes(ctx.clone());
    let template_routes = templates::routes(ctx.clone());
    let browse_routes = browse::routes(ctx.clone());
    let chat_storage = state.chat_storage.clone();
    let chat_routes = chat::routes(ctx.clone(), chat_storage.clone());
    let db_routes = db::routes(ctx.clone(), chat_storage.clone());
    let sessions_routes = sessions::routes(ctx.clone());
    let lhm_routes = lhm::routes(ctx.clone());
    let remote_agent_routes = remote_agent::routes(ctx.clone());
    let rapid_mlx_runtime_routes = rapid_mlx_runtime::routes(ctx.clone());
    let rapid_mlx_repair_routes = rapid_mlx_repair::routes(ctx.clone());
    let sensor_bridge_routes = sensor_bridge::routes(ctx.clone());
    let doctor_routes = doctor::routes(ctx.clone());

    let metrics_routes = metrics::routes(ctx.clone());
    let tls_routes = tls::routes(ctx.clone());
    let llama_binary_routes = llama_binary::routes(ctx.clone());

    let models_routes = models::routes(ctx.clone());
    let config_routes = config::routes(ctx.clone());
    let spawn_wizard_routes = spawn_wizard::routes(ctx.clone());
    let vram_routes = vram::routes(ctx.clone());
    let benchmark_routes = benchmark::routes(ctx.clone());
    let calibration_routes = calibration::routes(ctx.clone());
    let hf_routes = hf::routes(ctx.clone());
    let system_tools_routes = system_tools::routes(ctx.clone());

    let browse_with_chat = browse_routes.or(chat_routes);

    sessions_routes
        .or(browse_with_chat)
        .or(db_routes)
        .or(preset_routes)
        .or(preset_bundle_routes)
        .or(app_home_migration_routes)
        .or(template_routes)
        .or(models_routes)
        .or(config_routes)
        .or(lhm_routes)
        .or(remote_agent_routes)
        .or(rapid_mlx_runtime_routes)
        .or(rapid_mlx_repair_routes)
        .or(sensor_bridge_routes)
        .or(doctor_routes)
        .or(metrics_routes)
        .or(tls_routes)
        .or(llama_binary_routes)
        .or(spawn_wizard_routes)
        .or(vram_routes)
        .or(benchmark_routes)
        .or(calibration_routes)
        .or(hf_routes)
        .or(system_tools_routes)
        .or(sleep::routes(ctx.clone()))
        .or(debug::routes(ctx.clone()))
        .or(self_update::routes(ctx.clone()))
}

pub fn auth_api_routes(
    auth_manager: AuthManager,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    auth::routes(auth_manager)
}

#[cfg(test)]
mod tests {
    use super::hf::resolve_hf_target_dir;
    use super::legacy_chat_types::*;
    use super::spawn_wizard::is_private_host;
    use super::token_bootstrap_allowed;

    use crate::chat_storage::ChatStorage;
    use crate::config::{self, AcmeConfig, TLSConfig, TlsMode};
    use crate::gpu::env::GpuEnv;
    use crate::state::{AppPaths, AppState};
    use crate::web::auth::AuthManager;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::Arc;
    use warp::Filter;

    fn make_test_app_state(tls_config: TLSConfig) -> (AppState, Arc<config::AppConfig>) {
        let paths = AppPaths {
            presets_path: PathBuf::new(),
            templates_path: PathBuf::new(),
            models_dir: None,
            gpu_env_path: PathBuf::new(),
            ui_settings_path: PathBuf::new(),
            sessions_path: PathBuf::new(),
            model_tags_path: PathBuf::new(),
        };
        let cs = Arc::new(
            ChatStorage::open(&PathBuf::from(":memory:")).expect("open in-memory chat storage"),
        );
        let state = AppState::new(
            vec![],
            paths,
            GpuEnv::default(),
            crate::state::UiSettings::default(),
            cs,
            tls_config,
        );
        let app_config = Arc::new(config::AppConfig::for_test(
            Some("test-token".to_string()),
            None,
        ));
        (state, app_config)
    }

    fn tls_routes_filter(
        state: AppState,
        app_config: Arc<config::AppConfig>,
    ) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
        let auth = AuthManager::new(None, None, &TlsMode::None);
        super::tls::routes(super::ApiCtx {
            state,
            config: app_config,
            auth,
        })
    }

    fn auth_routes_filter(
        auth_manager: AuthManager,
    ) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
        super::auth_api_routes(auth_manager)
    }

    #[test]
    fn token_bootstrap_allows_loopback_without_basic_auth() {
        let auth = AuthManager::new(None, None, &TlsMode::None);
        assert!(token_bootstrap_allowed(&auth, "127.0.0.1"));
        assert!(token_bootstrap_allowed(&auth, "localhost"));
    }

    #[test]
    fn resolve_hf_target_dir_rejects_path_traversal() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let models_dir = tmp.path().join("models");
        let err = resolve_hf_target_dir(&models_dir, Some("../escape")).expect_err("rejects");
        assert!(err.contains("path traversal"));
    }

    #[test]
    fn resolve_hf_target_dir_creates_and_resolves_child_dir() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let models_dir = tmp.path().join("models");
        let target = resolve_hf_target_dir(&models_dir, Some("nested/model-dir")).expect("path");
        assert!(target.starts_with(models_dir.canonicalize().expect("canonical models_dir")));
        assert!(target.exists());
    }

    #[cfg(unix)]
    #[test]
    fn resolve_hf_target_dir_rechecks_symlink_escape_after_create() {
        use std::os::unix::fs::symlink;

        let tmp = tempfile::tempdir().expect("tempdir");
        let models_dir = tmp.path().join("models");
        let outside_dir = tmp.path().join("outside");

        std::fs::create_dir_all(&outside_dir).expect("outside dir");
        std::fs::create_dir_all(&models_dir).expect("models dir");
        symlink(&outside_dir, models_dir.join("linked")).expect("symlink");

        let err =
            resolve_hf_target_dir(&models_dir, Some("linked/new-download")).expect_err("rejects");
        assert!(err.contains("escapes models_dir"));
    }

    // ===== SSRF guard tests (is_private_host) =====

    #[test]
    fn srf_blocks_localhost_variants() {
        for host in [
            "localhost",
            "LOCALHOST",
            "127.0.0.1",
            "127.0.0.255",
            "127.255.255.255",
            "::1",
            "0:0:0:0:0:0:0:1",
            "0.0.0.0",
        ] {
            assert!(is_private_host(host), "Should block '{}'", host);
        }
    }

    #[test]
    fn srf_blocks_ipv4_private_ranges() {
        assert!(is_private_host("10.0.0.1"));
        assert!(is_private_host("10.255.255.255"));
        assert!(is_private_host("172.16.0.1"));
        assert!(is_private_host("172.31.255.255"));
        assert!(is_private_host("192.168.0.1"));
        assert!(is_private_host("192.168.255.255"));
        assert!(is_private_host("169.254.100.1"));
        assert!(is_private_host("169.254.0.1"));
        assert!(is_private_host("169.254.1.1"));
    }

    #[test]
    fn srf_blocks_ipv6_private_ranges() {
        assert!(is_private_host("fc00::1"));
        assert!(is_private_host("fd00::dead:beef"));
        assert!(is_private_host("fdff:ffff:ffff:ffff:ffff:ffff:ffff:ffff"));
        assert!(is_private_host("fe80::1"));
        assert!(is_private_host("febf:ffff:ffff:ffff:ffff:ffff:ffff:ffff"));
    }

    #[test]
    fn srf_blocks_internal_tlds() {
        for host in [
            "internal.local",
            "my-host.internal",
            "api.corp",
            "db.lan",
            "service.local.svc.cluster.local",
        ] {
            assert!(is_private_host(host), "Should block '{}'", host);
        }
    }

    #[test]
    fn srf_allows_public_hosts() {
        for host in [
            "huggingface.co",
            "cdn.huggingface.co",
            "google.com",
            "8.8.8.8",
            "1.1.1.1",
            "2001:4860:4860::8888",
        ] {
            assert!(!is_private_host(host), "Should allow '{}'", host);
        }
    }

    #[test]
    fn srf_allows_non_private_ipv4() {
        assert!(!is_private_host("172.32.0.1"));
        assert!(!is_private_host("172.15.255.255"));
    }

    #[test]
    fn token_bootstrap_allows_all_when_no_auth_configured() {
        let auth = AuthManager::new(None, None, &TlsMode::None);
        assert!(token_bootstrap_allowed(&auth, "0.0.0.0"));
        assert!(token_bootstrap_allowed(&auth, "192.168.2.44"));
    }

    #[test]
    fn token_bootstrap_allows_non_loopback_host_when_no_auth() {
        let auth = AuthManager::new(None, None, &TlsMode::None);
        assert!(token_bootstrap_allowed(&auth, "0.0.0.0"));
        assert!(token_bootstrap_allowed(&auth, "192.168.2.44"));
    }

    #[test]
    fn token_bootstrap_rejects_spoofed_host_header_on_non_loopback_bind() {
        let auth = AuthManager::new(
            AuthManager::parse_credentials("admin:secret"),
            None,
            &TlsMode::None,
        );
        assert!(!token_bootstrap_allowed(&auth, "0.0.0.0"));
    }

    #[test]
    fn token_bootstrap_allows_loopback_when_basic_auth_is_configured() {
        let auth = AuthManager::new(
            AuthManager::parse_credentials("admin:secret"),
            None,
            &TlsMode::None,
        );
        assert!(token_bootstrap_allowed(&auth, "127.0.0.1"));
        assert!(!token_bootstrap_allowed(&auth, "0.0.0.0"));
    }

    #[test]
    fn token_bootstrap_allows_loopback_when_form_auth_is_configured() {
        let auth = AuthManager::new(
            None,
            AuthManager::parse_credentials("admin:secret"),
            &TlsMode::None,
        );
        assert!(token_bootstrap_allowed(&auth, "127.0.0.1"));
        assert!(!token_bootstrap_allowed(&auth, "0.0.0.0"));
    }

    #[tokio::test]
    async fn form_auth_login_sets_session_cookie_and_status_reflects_it() {
        let auth = AuthManager::new(
            None,
            AuthManager::parse_credentials("admin:secret123"),
            &TlsMode::None,
        );
        let routes = auth_routes_filter(auth);

        let login_resp = warp::test::request()
            .method("POST")
            .path("/api/auth/login")
            .json(&serde_json::json!({
                "username": "admin",
                "password": "secret123",
            }))
            .reply(&routes)
            .await;

        assert_eq!(login_resp.status(), 200);
        let set_cookie = login_resp
            .headers()
            .get("set-cookie")
            .and_then(|value| value.to_str().ok())
            .expect("set-cookie header");
        assert!(set_cookie.contains("local_llm_foundry_session="));

        let status_resp = warp::test::request()
            .method("GET")
            .path("/api/auth/status")
            .header("cookie", set_cookie)
            .reply(&routes)
            .await;

        assert_eq!(status_resp.status(), 200);
        let body: serde_json::Value =
            serde_json::from_slice(status_resp.body()).expect("valid JSON");
        assert_eq!(body["enabled"], true);
        assert_eq!(body["methods"]["form"], true);
        assert_eq!(body["authenticated"], true);
        assert_eq!(body["method"], "form");
        assert_eq!(body["username"], "admin");
    }

    #[tokio::test]
    async fn form_auth_logout_clears_session_cookie() {
        let auth = AuthManager::new(
            None,
            AuthManager::parse_credentials("admin:secret123"),
            &TlsMode::None,
        );
        let routes = auth_routes_filter(auth);

        let login_resp = warp::test::request()
            .method("POST")
            .path("/api/auth/login")
            .json(&serde_json::json!({
                "username": "admin",
                "password": "secret123",
            }))
            .reply(&routes)
            .await;

        let set_cookie = login_resp
            .headers()
            .get("set-cookie")
            .and_then(|value| value.to_str().ok())
            .expect("set-cookie header");

        let logout_resp = warp::test::request()
            .method("POST")
            .path("/api/auth/logout")
            .header("cookie", set_cookie)
            .reply(&routes)
            .await;

        assert_eq!(logout_resp.status(), 200);
        let clear_cookie = logout_resp
            .headers()
            .get("set-cookie")
            .and_then(|value| value.to_str().ok())
            .expect("set-cookie clear header");
        assert!(clear_cookie.contains("Max-Age=0"));
    }

    #[tokio::test]
    async fn tls_config_get_requires_api_token() {
        let (state, app_config) = make_test_app_state(TLSConfig::default());
        let routes = tls_routes_filter(state, app_config);

        let resp = warp::test::request()
            .method("GET")
            .path("/api/tls/config")
            .reply(&routes)
            .await;
        assert_eq!(resp.status(), 401);

        let resp = warp::test::request()
            .method("GET")
            .path("/api/tls/config")
            .header("Authorization", "Bearer test-token")
            .reply(&routes)
            .await;
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = serde_json::from_slice(resp.body()).expect("valid JSON");
        assert_eq!(body["mode"], "none");
    }

    #[tokio::test]
    async fn tls_config_get_returns_acme_fields() {
        let mut dns_config = HashMap::new();
        dns_config.insert("CF_API_TOKEN".to_string(), "redacted".to_string());

        let tls_config = TLSConfig {
            mode: TlsMode::Acme,
            custom_cert_path: None,
            custom_key_path: None,
            acme: AcmeConfig {
                enabled: true,
                fqdn: "llama-monitor.example.com".to_string(),
                email: String::new(),
                environment: "staging".to_string(),
                dns_provider: "cloudflare".to_string(),
                dns_config,
                validation_delay: 300,
                last_renewal: None,
                cert_path: None,
                key_path: None,
            },
        };

        let (state, app_config) = make_test_app_state(tls_config);
        let routes = tls_routes_filter(state, app_config);

        let resp = warp::test::request()
            .method("GET")
            .path("/api/tls/config")
            .header("Authorization", "Bearer test-token")
            .reply(&routes)
            .await;

        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = serde_json::from_slice(resp.body()).expect("valid JSON");
        assert_eq!(body["mode"], "acme");
        assert_eq!(body["acme"]["fqdn"], "llama-monitor.example.com");
        assert_eq!(body["acme"]["environment"], "staging");
        assert_eq!(body["acme"]["dnsProvider"], "cloudflare");
    }

    #[tokio::test]
    async fn tls_config_put_accepts_valid_acme() {
        let (state, app_config) = make_test_app_state(TLSConfig::default());
        let routes = tls_routes_filter(state.clone(), app_config);

        let payload = serde_json::json!({
            "mode": "acme",
            "acme": {
                "enabled": true,
                "fqdn": "llama-monitor.example.com",
                "environment": "staging",
                "dnsProvider": "cloudflare",
                "validationDelay": 300,
                "dnsConfig": {
                    "CF_API_TOKEN": "test-token"
                }
            }
        });

        let resp = warp::test::request()
            .method("PUT")
            .path("/api/tls/config")
            .header("Authorization", "Bearer test-token")
            .json(&payload)
            .reply(&routes)
            .await;

        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = serde_json::from_slice(resp.body()).expect("valid JSON");
        assert_eq!(body["ok"], true);

        let cfg = state.get_tls_config();
        assert_eq!(cfg.mode, TlsMode::Acme);
        assert_eq!(cfg.acme.fqdn, "llama-monitor.example.com");
        assert_eq!(cfg.acme.dns_provider, "cloudflare");
    }

    #[tokio::test]
    async fn tls_config_put_rejects_invalid_acme_missing_provider() {
        let (state, app_config) = make_test_app_state(TLSConfig::default());
        let routes = tls_routes_filter(state, app_config);

        let payload = serde_json::json!({
            "mode": "acme",
            "acme": {
                "enabled": true,
                "fqdn": "llama-monitor.example.com",
                "environment": "staging",
                "dnsProvider": "",
                "dnsConfig": {
                    "CF_API_TOKEN": "test-token"
                }
            }
        });

        let resp = warp::test::request()
            .method("PUT")
            .path("/api/tls/config")
            .header("Authorization", "Bearer test-token")
            .json(&payload)
            .reply(&routes)
            .await;

        assert_eq!(resp.status(), 400);
        let body: serde_json::Value = serde_json::from_slice(resp.body()).expect("valid JSON");
        assert!(
            body["error"]
                .as_str()
                .map(|s| s.contains("dnsProvider"))
                .unwrap_or(false)
        );
    }

    fn make_minimal_chat_tab() -> ChatTab {
        ChatTab {
            id: "tab-1".to_string(),
            name: "Test Tab".to_string(),
            system_prompt: "You are helpful.".to_string(),
            ai_name: None,
            user_name: None,
            explicit_level: None,
            messages: vec![],
            total_input_tokens: None,
            total_output_tokens: None,
            model_params: ChatModelParams::default(),
            created_at: 0,
            updated_at: 0,
            auto_compact: None,
            auto_compact_summarize: None,
            compact_threshold: None,
            compact_mode: None,
            last_ctx_pct: None,
            active_template_id: None,
            context_notes: vec![],
            sidebar_width: 0,
            quick_guide_active: String::new(),
            armed_story_beats: vec![],
            role_boundary_custom: None,
            ai_gender: None,
        }
    }

    #[test]
    fn chat_tab_explicit_level_serialization() {
        let mut tab = make_minimal_chat_tab();
        tab.explicit_level = Some(1);

        let json = serde_json::to_string(&tab).expect("ChatTab should serialize");

        assert!(
            json.contains("\"explicitLevel\""),
            "JSON should contain camelCase 'explicitLevel' field, got: {}",
            json
        );

        let parsed: serde_json::Value =
            serde_json::from_str(&json).expect("JSON should parse to Value");
        assert_eq!(
            parsed.get("explicitLevel").and_then(|v| v.as_u64()),
            Some(1),
            "explicitLevel should be 1"
        );

        let deserialized: ChatTab =
            serde_json::from_str(&json).expect("ChatTab should deserialize from own JSON");
        assert_eq!(
            deserialized.explicit_level,
            Some(1),
            "explicit_level should round-trip to Some(1)"
        );
    }

    #[test]
    fn chat_tab_explicit_level_default() {
        let json = r#"{
            "id": "tab-1",
            "name": "Test Tab",
            "system_prompt": "You are helpful.",
            "messages": [],
            "model_params": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "min_p": 0.01,
                "repeat_penalty": 1.0
            },
            "created_at": 0,
            "updated_at": 0
        }"#;

        let result = serde_json::from_str::<ChatTab>(json);
        assert!(
            result.is_ok(),
            "Should deserialize without explicitLevel field"
        );

        let tab = result.unwrap();
        assert!(
            tab.explicit_level.is_none(),
            "explicit_level should default to None when field is absent"
        );
    }

    #[test]
    fn chat_tab_explicit_mode_alias_migration() {
        let json = r#"{
            "id": "tab-1",
            "name": "Test Tab",
            "system_prompt": "You are helpful.",
            "explicit_mode": 2,
            "messages": [],
            "model_params": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "min_p": 0.01,
                "repeat_penalty": 1.0
            },
            "created_at": 0,
            "updated_at": 0
        }"#;

        let result = serde_json::from_str::<ChatTab>(json);
        assert!(
            result.is_ok(),
            "Should deserialize legacy 'explicit_mode' field via alias"
        );

        let tab = result.unwrap();
        assert_eq!(
            tab.explicit_level,
            Some(2),
            "explicit_mode alias should map to explicit_level"
        );
    }

    #[test]
    fn chat_tab_explicit_level_all_states() {
        for level in [0u8, 1, 2] {
            let mut tab = make_minimal_chat_tab();
            tab.explicit_level = Some(level);

            let json = serde_json::to_string(&tab)
                .unwrap_or_else(|e| panic!("ChatTab should serialize for level {}: {}", level, e));

            assert!(
                json.contains("\"explicitLevel\""),
                "JSON for level {} should contain 'explicitLevel'",
                level
            );

            let deserialized: ChatTab = serde_json::from_str(&json).unwrap_or_else(|e| {
                panic!("ChatTab should deserialize for level {}: {}", level, e)
            });
            assert_eq!(
                deserialized.explicit_level,
                Some(level),
                "explicit_level should round-trip for state {}",
                level
            );
        }
    }

    #[test]
    fn chat_message_compaction_metadata_round_trips() {
        let msg = ChatMessage {
            role: "system".to_string(),
            content: "## Persistent Facts\n- Keeps rolling memory".to_string(),
            timestamp_ms: 123,
            input_tokens: None,
            output_tokens: None,
            cumulative_input_tokens: None,
            cumulative_output_tokens: None,
            compaction_marker: Some(true),
            summarized: Some(true),
            dropped_count: Some(42),
            dropped_preview: Some(vec![CompactionPreview {
                role: "user".to_string(),
                snippet: "example".to_string(),
            }]),
            tokens_freed_estimate: Some(999),
            ctx_pct_before: Some(87.5),
            memory_version: Some(2),
            memory_domain: Some("coding".to_string()),
            summary_kind: Some("rolling-memory".to_string()),
            compacted_at: Some(456),
            compacted_message_count_total: Some(84),
            recent_tail_kept: Some(8),
            thinking_content: None,
        };

        let json = serde_json::to_string(&msg).expect("ChatMessage should serialize");
        let decoded: ChatMessage =
            serde_json::from_str(&json).expect("ChatMessage should deserialize from own JSON");

        assert_eq!(decoded.compaction_marker, Some(true));
        assert_eq!(decoded.memory_version, Some(2));
        assert_eq!(decoded.memory_domain.as_deref(), Some("coding"));
        assert_eq!(decoded.summary_kind.as_deref(), Some("rolling-memory"));
        assert_eq!(decoded.compacted_message_count_total, Some(84));
        assert_eq!(decoded.recent_tail_kept, Some(8));
        assert_eq!(
            decoded
                .dropped_preview
                .as_ref()
                .and_then(|rows| rows.first())
                .map(|row| row.snippet.as_str()),
            Some("example")
        );
    }

    // ── Route smoke tests ──────────────────────────────────────────────────────
    // Each test sends a properly-formed request (correct method + Content-Type)
    // without an API token and asserts 401, not 404.
    //
    // A 404 means the route was accidentally deleted from api_routes().
    // A 401 means the route exists and auth is working correctly.
    //
    // These tests exist specifically to catch the regression from commit ac643ab
    // where a worktree-agent silently deleted 27 handler functions.

    fn make_all_routes()
    -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
        let paths = crate::state::AppPaths {
            presets_path: PathBuf::new(),
            templates_path: PathBuf::new(),
            models_dir: None,
            gpu_env_path: PathBuf::new(),
            ui_settings_path: PathBuf::new(),
            sessions_path: PathBuf::new(),
            model_tags_path: PathBuf::new(),
        };
        let cs = Arc::new(
            crate::chat_storage::ChatStorage::open(&PathBuf::from(":memory:"))
                .expect("in-memory chat storage"),
        );
        let state = crate::state::AppState::new(
            vec![],
            paths,
            crate::gpu::env::GpuEnv::default(),
            crate::state::UiSettings::default(),
            cs,
            crate::config::TLSConfig::default(),
        );
        let app_config = Arc::new(crate::config::AppConfig::for_test(
            Some("test-token".to_string()),
            Some("db-admin-token".to_string()),
        ));
        let auth = AuthManager::new(None, None, &crate::config::TlsMode::None);
        super::api_routes(state, app_config, auth, "127.0.0.1".to_string())
    }

    macro_rules! route_smoke_tests {
        ( $( ($test_name:ident, $method:expr, $path:expr, $body:expr) ),* $(,)? ) => {
            $(
                #[tokio::test]
                async fn $test_name() {
                    let routes = make_all_routes();
                    let req = warp::test::request()
                        .method($method)
                        .path($path);
                    let body_str: Option<&str> = $body;
                    let resp = if let Some(b) = body_str {
                        req.header("Content-Type", "application/json")
                           .body(b)
                           .reply(&routes)
                           .await
                    } else {
                        req.reply(&routes).await
                    };
                    assert_ne!(
                        resp.status(), 404,
                        "Route {} {} returned 404 — it may have been deleted from api_routes()",
                        $method, $path
                    );
                    assert_eq!(
                        resp.status(), 401,
                        "Route {} {} should require auth (expected 401, got {})",
                        $method, $path, resp.status()
                    );
                }
            )*
        };
    }

    route_smoke_tests![
        (
            route_spawn_wizard_import,
            "POST",
            "/api/spawn-wizard/import-launch-file",
            Some("{}")
        ),
        (
            route_chat_template_fetch,
            "POST",
            "/api/chat-template/fetch",
            Some("{}")
        ),
        (
            route_chat_template_upload,
            "POST",
            "/api/chat-template/upload",
            Some("{}")
        ),
        (
            route_chat_template_install_hf,
            "POST",
            "/api/chat-template/install-hf",
            Some(
                "{\"repo\":\"froggeric/Qwen-Fixed-Chat-Templates\",\"file\":\"chat_template.jinja\",\"name\":\"test\"}"
            )
        ),
        (
            route_chat_template_install_url,
            "POST",
            "/api/chat-template/install-url",
            Some("{}")
        ),
        (
            route_chat_template_check_update,
            "POST",
            "/api/chat-template/check-update",
            Some("{}")
        ),
        (
            route_vram_estimate,
            "POST",
            "/api/vram/estimate",
            Some("{}")
        ),
        (
            route_vram_estimate_breakdown,
            "POST",
            "/api/vram-estimate",
            Some("{}")
        ),
        (
            route_vram_quant_compare,
            "POST",
            "/api/vram/quant-compare",
            Some("{}")
        ),
        (
            route_vram_auto_size,
            "POST",
            "/api/vram/auto-size",
            Some("{}")
        ),
        (
            route_models_download_start,
            "POST",
            "/api/models/download/start",
            Some("{}")
        ),
        (
            route_models_download_status,
            "GET",
            "/api/models/download/test-id/status",
            None
        ),
        (
            route_models_download_cancel,
            "POST",
            "/api/models/download/test-id/cancel",
            Some("{}")
        ),
        (route_benchmark, "POST", "/api/benchmark", Some("{}")),
        (
            route_model_defaults,
            "POST",
            "/api/model-defaults",
            Some("{}")
        ),
        (
            route_model_introspect,
            "POST",
            "/api/model/introspect",
            Some("{}")
        ),
        (
            route_third_party_models,
            "POST",
            "/api/third-party-models",
            Some("{}")
        ),
        (route_moe_tune, "POST", "/api/moe-tune", Some("{}")),
        (route_advise, "POST", "/api/advise", Some("{}")),
        (route_tune_ncpumoe, "POST", "/api/tune/ncpumoe", Some("{}")),
        (route_bench_sweep, "POST", "/api/bench/sweep", Some("{}")),
        (route_hf_search, "POST", "/api/hf/search", Some("{}")),
        (route_hf_files, "POST", "/api/hf/files", Some("{}")),
        (
            route_hf_community_picks,
            "GET",
            "/api/hf/community-picks",
            None
        ),
        (route_hf_quantizers_get, "GET", "/api/hf/quantizers", None),
        (
            route_hf_quantizers_put,
            "PUT",
            "/api/hf/quantizers",
            Some("[]")
        ),
        (
            route_hf_community_sources_get,
            "GET",
            "/api/hf/community-sources",
            None
        ),
        // These two need bodies that actually deserialize: warp rejects a malformed body with
        // 400 before the handler runs, which would make the 401 assertion below untestable.
        (
            route_hf_community_sources_put,
            "PUT",
            "/api/hf/community-sources",
            Some(r#"{"entries":[]}"#)
        ),
        (
            route_hf_community_sources_entry_post,
            "POST",
            "/api/hf/community-sources/entry",
            Some(r#"{"username":"x","displayName":"x","description":"x","role":"curator"}"#)
        ),
        (
            route_hf_community_sources_entry_delete,
            "DELETE",
            "/api/hf/community-sources/entry?username=x&role=curator",
            None
        ),
        (
            route_hf_community_sources_reset,
            "POST",
            "/api/hf/community-sources/reset",
            None
        ),
        (route_hf_download_dir, "GET", "/api/hf/download-dir", None),
        (route_hf_token_get, "GET", "/api/hf/token", None),
        (route_hf_token_put, "PUT", "/api/hf/token", Some("{}")),
        (route_hf_token_delete, "DELETE", "/api/hf/token", None),
        (route_hf_card, "GET", "/api/hf/card?repo=test%2Fmodel", None),
        (route_hf_download, "POST", "/api/hf/download", Some("{}")),
        (
            route_llama_binary_version,
            "GET",
            "/api/llama-binary/version",
            None
        ),
        (
            route_llama_binary_latest,
            "GET",
            "/api/llama-binary/latest",
            None
        ),
        (
            route_llama_binary_update,
            "POST",
            "/api/llama-binary/update",
            Some("{}")
        ),
        (
            route_rapid_mlx_runtime_status,
            "GET",
            "/api/rapid-mlx/runtime/status",
            None
        ),
        (
            route_rapid_mlx_runtime_releases,
            "GET",
            "/api/rapid-mlx/runtime/releases",
            None
        ),
        (
            route_rapid_mlx_recommend,
            "POST",
            "/api/rapid-mlx/recommend",
            Some("{}")
        ),
        (
            route_rapid_mlx_runtime_install,
            "POST",
            "/api/rapid-mlx/runtime/install",
            Some("{}")
        ),
        (
            route_rapid_mlx_runtime_upgrade,
            "POST",
            "/api/rapid-mlx/runtime/upgrade",
            Some("{}")
        ),
        (
            route_rapid_mlx_runtime_repair,
            "POST",
            "/api/rapid-mlx/runtime/repair",
            Some("{}")
        ),
        (
            route_rapid_mlx_runtime_rollback,
            "POST",
            "/api/rapid-mlx/runtime/rollback",
            Some("{}")
        ),
        (
            route_rapid_mlx_runtime_job,
            "GET",
            "/api/rapid-mlx/runtime/jobs/missing",
            None
        ),
        (
            route_rapid_mlx_command_preview,
            "POST",
            "/api/rapid-mlx/command-preview",
            Some("{}")
        ),
        (
            route_calibration_preflight,
            "POST",
            "/api/calibrations/preflight",
            Some("{}")
        ),
        (
            route_calibration_start,
            "POST",
            "/api/calibrations",
            Some("{}")
        ),
        (
            route_calibration_match,
            "POST",
            "/api/calibrations/match",
            Some("{}")
        ),
        (route_calibration_list, "GET", "/api/calibrations", None),
        (
            route_calibration_resume,
            "POST",
            "/api/calibrations/missing/resume",
            Some("{}")
        ),
        (
            route_calibration_get,
            "GET",
            "/api/calibrations/missing",
            None
        ),
        (
            route_calibration_receipt,
            "GET",
            "/api/calibrations/missing/receipt",
            None
        ),
        (
            route_calibration_apply,
            "POST",
            "/api/calibrations/missing/apply",
            Some("{}")
        ),
        (
            route_calibration_rollback,
            "POST",
            "/api/calibrations/missing/rollback",
            Some("{}")
        ),
        (
            route_calibration_cancel,
            "POST",
            "/api/calibrations/missing/cancel",
            None
        ),
        (
            route_calibration_forget,
            "POST",
            "/api/calibrations/missing/forget",
            Some("{}")
        ),
    ];

    #[tokio::test]
    async fn rapid_mlx_runtime_mutations_require_db_admin_token() {
        let response = warp::test::request()
            .method("POST")
            .path("/api/rapid-mlx/runtime/repair")
            .header("Content-Type", "application/json")
            .header("Authorization", "Bearer test-token")
            .body(r#"{"confirm":"REPAIR_RAPID_MLX_RUNTIME"}"#)
            .reply(&make_all_routes())
            .await;
        assert_eq!(response.status(), 401);
        assert!(String::from_utf8_lossy(response.body()).contains("db-admin-token required"));
    }

    #[tokio::test]
    async fn rapid_mlx_runtime_rejects_inexact_confirmation() {
        let response = warp::test::request()
            .method("POST")
            .path("/api/rapid-mlx/runtime/rollback")
            .header("Content-Type", "application/json")
            .header("Authorization", "Bearer db-admin-token")
            .body(r#"{"confirm":"rollback"}"#)
            .reply(&make_all_routes())
            .await;
        assert_eq!(response.status(), 400);
        assert!(String::from_utf8_lossy(response.body()).contains("ROLLBACK_RAPID_MLX_RUNTIME"));
    }

    #[tokio::test]
    async fn rapid_mlx_runtime_rejects_malformed_and_unknown_json() {
        for body in [
            r#"{"confirm": "REPAIR_RAPID_MLX_RUNTIME""#,
            r#"{"confirm":"REPAIR_RAPID_MLX_RUNTIME","extra":true}"#,
        ] {
            let routes = make_all_routes().recover(super::super::handle_rejection);
            let response = warp::test::request()
                .method("POST")
                .path("/api/rapid-mlx/runtime/repair")
                .header("Content-Type", "application/json")
                .header("Authorization", "Bearer db-admin-token")
                .body(body)
                .reply(&routes)
                .await;
            assert_eq!(
                response.status(),
                400,
                "unexpected response: {}",
                String::from_utf8_lossy(response.body())
            );
        }
    }

    #[tokio::test]
    async fn rapid_mlx_runtime_job_reads_accept_api_token() {
        let response = warp::test::request()
            .method("GET")
            .path("/api/rapid-mlx/runtime/jobs/missing")
            .header("Authorization", "Bearer test-token")
            .reply(&make_all_routes())
            .await;
        assert_eq!(response.status(), 404);
        assert!(String::from_utf8_lossy(response.body()).contains("not found"));
    }

    #[tokio::test]
    async fn app_home_migration_status_requires_api_token() {
        let response = warp::test::request()
            .method("GET")
            .path("/api/app-home-migration/status")
            .reply(&make_all_routes())
            .await;
        assert_eq!(response.status(), 401);
    }

    #[tokio::test]
    async fn app_home_migration_status_accepts_api_token() {
        let response = warp::test::request()
            .method("GET")
            .path("/api/app-home-migration/status")
            .header("Authorization", "Bearer test-token")
            .reply(&make_all_routes())
            .await;
        assert_eq!(response.status(), 200);
        assert!(String::from_utf8_lossy(response.body()).contains("migration_required"));
    }

    #[tokio::test]
    async fn app_home_migration_preview_requires_api_token() {
        let response = warp::test::request()
            .method("GET")
            .path("/api/app-home-migration/preview")
            .reply(&make_all_routes())
            .await;
        assert_eq!(response.status(), 401);
    }

    #[tokio::test]
    async fn app_home_migration_queue_requires_db_admin_token() {
        let response = warp::test::request()
            .method("POST")
            .path("/api/app-home-migration/queue")
            .header("Content-Type", "application/json")
            .header("Authorization", "Bearer test-token")
            .body(r#"{"plan_id":"test","confirmation":"MIGRATE TO LOCAL LLM FOUNDRY"}"#)
            .reply(&make_all_routes())
            .await;
        assert_eq!(response.status(), 401);
    }

    #[tokio::test]
    async fn model_root_relocation_status_requires_api_token() {
        let response = warp::test::request()
            .method("GET")
            .path("/api/models/root-relocation/status")
            .reply(&make_all_routes())
            .await;
        assert_eq!(response.status(), 401);
    }

    #[tokio::test]
    async fn model_root_relocation_preview_requires_api_token() {
        let response = warp::test::request()
            .method("POST")
            .path("/api/models/root-relocation/preview")
            .header("Content-Type", "application/json")
            .body(r#"{"choice":"keep_legacy"}"#)
            .reply(&make_all_routes())
            .await;
        assert_eq!(response.status(), 401);
    }

    #[tokio::test]
    async fn model_root_relocation_execute_requires_db_admin_token() {
        let response = warp::test::request()
            .method("POST")
            .path("/api/models/root-relocation/execute")
            .header("Content-Type", "application/json")
            .header("Authorization", "Bearer test-token")
            .body(r#"{"choice":"keep_legacy","plan_id":"0000000000000000000000000000000000000000000000000000000000000000","confirmation":"KEEP_LEGACY_MODEL_ROOT"}"#)
            .reply(&make_all_routes())
            .await;
        assert_eq!(response.status(), 401);
    }
}
