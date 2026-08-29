use std::collections::{BTreeMap, HashSet};
use std::net::SocketAddr;
use std::sync::{
    Arc, LazyLock, Mutex,
    atomic::{AtomicU64, Ordering},
};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use warp::Filter;
use warp::http::{HeaderMap, StatusCode};

use crate::config::AppConfig;
use crate::gpu::{self, GpuMetrics};
use crate::remote_ssh::{self, SshConnection};
use crate::state::{AppState, EndpointKind, SessionMode};
use crate::system::{self, SystemMetrics};

/// Shell-quotes a path for safe inclusion in a command string.
///
/// Uses platform-appropriate quoting to prevent command injection when
/// user-controlled paths are interpolated into shell commands executed over SSH.
///
/// For Windows, use `shell_quote_path_cmd` when the command runs under `cmd.exe`
/// (e.g., schtasks), and `shell_quote_path` for PowerShell contexts.
fn shell_quote_path(path: &str, os: RemoteOs) -> String {
    match os {
        RemoteOs::Unix | RemoteOs::Macos => {
            // shlex quoting: wraps in single quotes, escapes embedded quotes
            shlex::try_quote(path)
                .map(|s| s.into_owned())
                .unwrap_or_else(|_| path.to_string())
        }
        RemoteOs::Windows => {
            // PowerShell single quotes are literal; escape embedded quotes by doubling
            format!("'{}'", path.replace('\'', "''"))
        }
        RemoteOs::Unknown => {
            // Conservative: treat as Unix
            shlex::try_quote(path)
                .map(|s| s.into_owned())
                .unwrap_or_else(|_| path.to_string())
        }
    }
}

/// Quotes a path for `cmd.exe` contexts (schtasks, cmd.exe /C).
///
/// Unlike PowerShell, cmd.exe does not treat single quotes as special.
/// Use double quotes with proper escaping for embedded quotes.
#[allow(dead_code)] // kept for tests; was used before PowerShell migration
fn shell_quote_path_cmd(path: &str) -> String {
    // In cmd.exe, double quotes delimit strings; escape embedded quotes with ^
    format!("\"{}\"", path.replace('"', "\"^\""))
}

/// Validates a user-supplied shell command for remote-agent autostart/start.
///
/// Enforces a strict allowlist:
/// - Must start with a plausible llama-monitor binary path.
/// - Only known, safe flags are permitted.
/// - No shell metacharacters or arbitrary tokens.
///
/// This prevents command injection when the command is sent to
/// `remote_ssh::exec` (which runs via `channel.exec()` on the remote shell).
pub(crate) fn validate_remote_command(cmd: &str) -> bool {
    let trimmed = cmd.trim();
    if trimmed.is_empty() {
        return false;
    }

    // Shell metacharacters that could be used to chain or inject commands.
    // Note: spaces are allowed (for flags), but these are not.
    let dangerous = ";|&$`(){}[]!#<>?*~\n\r\\";
    if trimmed.chars().any(|c| dangerous.contains(c)) {
        return false;
    }

    // Split into tokens by spaces; enforce structure.
    let tokens: Vec<&str> = trimmed.split_whitespace().collect();
    if tokens.is_empty() {
        return false;
    }

    // First token must be a plausible llama-monitor binary path.
    let first = tokens[0];
    let stem = first
        .split('/')
        .next_back()
        .unwrap_or(first)
        .split('\\')
        .next_back()
        .unwrap_or(first);
    if !(matches!(
        stem,
        "llama-monitor" | "llama-monitor.exe" | "local-llm-foundry" | "local-llm-foundry.exe"
    )) {
        return false;
    }

    // Allowed flags (prefix-based) and whether they take an argument.
    let allowed_flags = [
        "--agent",
        "--agent-host",
        "--agent-port",
        "--models-dir",
        "--version",
        "--help",
    ];

    let mut i = 1usize;
    while i < tokens.len() {
        let tok = tokens[i];
        if tok.starts_with("--") {
            if allowed_flags.iter().any(|f| tok.starts_with(f)) {
                // Flags that take a value are allowed if followed by a simple value token.
                if ["--agent-host", "--agent-port", "--models-dir"]
                    .iter()
                    .any(|f| tok.starts_with(f))
                {
                    i += 1;
                    if i >= tokens.len() {
                        return false;
                    }
                    // Value token must be simple (no shell chars)
                    if tokens[i].chars().any(|c| dangerous.contains(c)) {
                        return false;
                    }
                }
                i += 1;
            } else {
                // Unknown flag → reject.
                return false;
            }
        } else {
            // Positional / unknown token → reject.
            return false;
        }
    }

    true
}

/// Validates an install path to ensure it does not contain shell metacharacters
/// or target suspicious directories.
///
/// This is the primary defense against command injection; shell quoting is
/// a secondary defense-in-depth measure.
fn validate_install_path(path: &str, target_os: RemoteOs) -> Result<(), anyhow::Error> {
    // Must not contain shell metacharacters (platform-independent check)
    // Note: ~ is excluded — it's a valid Unix home directory prefix
    let dangerous_chars = ";|&$`'\"(){}[]!#<>*?";
    if path.chars().any(|c| dangerous_chars.contains(c)) {
        return Err(anyhow::anyhow!("Install path contains invalid characters"));
    }

    match target_os {
        RemoteOs::Unix | RemoteOs::Macos => {
            // Must be absolute or tilde-expanded (~)
            if !path.starts_with('/') && !path.starts_with('~') {
                return Err(anyhow::anyhow!("Install path must be absolute"));
            }
            // Must not target suspicious directories
            let forbidden = ["/tmp", "/var", "/etc"];
            if forbidden.iter().any(|f| path.starts_with(f)) {
                return Err(anyhow::anyhow!("Install path not allowed"));
            }
        }
        RemoteOs::Windows => {
            // Windows absolute: drive letter (C:\), UNC (\\), or env var (%APPDATA%\)
            let is_windows_absolute = path.len() >= 3
                && ((path.as_bytes()[1] == b':'
                    && (path.as_bytes()[2] == b'\\' || path.as_bytes()[2] == b'/'))
                    || path.starts_with("%")
                    || path.starts_with("\\\\"));
            if !is_windows_absolute {
                return Err(anyhow::anyhow!("Install path must be absolute"));
            }
            // Must not target suspicious directories
            let forbidden = ["C:\\Windows", "C:\\WINDOWS", "C:/Windows", "C:/WINDOWS"];
            if forbidden.iter().any(|f| path.starts_with(f)) {
                return Err(anyhow::anyhow!("Install path not allowed"));
            }
        }
        RemoteOs::Unknown => {
            // Conservative: require some form of absolute path
            if !path.starts_with('/') && !path.starts_with('\\') {
                return Err(anyhow::anyhow!("Install path must be absolute"));
            }
        }
    }

    Ok(())
}

thread_local! {
    static LATEST_RELEASE_CACHE: Mutex<Option<(LatestReleaseInfo, Instant)>> = const { Mutex::new(None) };
}

static REMOTE_AGENT_AUTOSTART_SUPPRESS_UNTIL: LazyLock<Mutex<Option<Instant>>> =
    LazyLock::new(|| Mutex::new(None));

const REMOTE_AGENT_DEFAULT_PORT: u16 = 7779;
const REMOTE_AGENT_POLL_INTERVAL: Duration = Duration::from_secs(2);
/// The installed agent stays quiet until an authenticated master request arrives.
/// After this idle window, the metric workers stop reading GPU/system state again.
const AGENT_MASTER_IDLE_TIMEOUT: Duration = Duration::from_secs(180);
const AGENT_IDLE_CHECK_INTERVAL: Duration = Duration::from_secs(5);
const REMOTE_AGENT_AUTOSTART_TIMEOUT: Duration = Duration::from_secs(15);
const REMOTE_AGENT_AUTOSTART_SUPPRESS_DURATION: Duration = Duration::from_secs(120);
const GITHUB_LATEST_RELEASE_URL: &str = crate::identity::RELEASE_API_URL;

fn unix_timestamp_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn master_request_is_recent(last_request: &AtomicU64, now: u64) -> bool {
    let last = last_request.load(Ordering::Relaxed);
    last != 0 && now.saturating_sub(last) <= AGENT_MASTER_IDLE_TIMEOUT.as_secs()
}

fn mark_master_request(last_request: &AtomicU64) {
    last_request.store(unix_timestamp_seconds(), Ordering::Relaxed);
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetrics {
    pub system: SystemMetrics,
    pub gpu: BTreeMap<String, GpuMetrics>,
}

const AGENT_PROTOCOL_VERSION: &str = "1.0.0";

/// Enrollment port offset: agent listens on port N for mTLS, and on port N+1 for
/// unenrolled-client CA registration (server-TLS only, api-token auth).
const ENROLLMENT_PORT_OFFSET: u16 = 1;

/// Request body for `POST /api/agent/register-ca` on the enrollment port.
#[derive(Debug, Deserialize)]
pub struct RegisterCaRequest {
    /// PEM-encoded CA certificate to trust (max 64 KB).
    pub ca_pem: String,
    /// Must be the literal string `"register-ca"` to prevent accidental registration.
    #[serde(default)]
    pub confirm: String,
}

/// Load all allowed agent tokens from agent-tokens.json (if present).
fn load_agent_tokens(config_dir: &std::path::Path) -> Vec<String> {
    let path = config_dir.join("agent-tokens.json");
    if !path.exists() {
        return Vec::new();
    }
    let Ok(content) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    let Ok(file) = serde_json::from_str::<serde_json::Value>(&content) else {
        return Vec::new();
    };
    file.get("tokens")
        .and_then(|t| t.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

/// Write agent-tokens.json (atomic).
fn save_agent_tokens(config_dir: &std::path::Path, tokens: &[String]) {
    let path = config_dir.join("agent-tokens.json");
    let file = serde_json::json!({ "tokens": tokens });
    let Ok(json) = serde_json::to_string_pretty(&file) else {
        return;
    };
    let tmp = path.with_extension("json.tmp");
    let _ = std::fs::write(&tmp, json);
    let _ = std::fs::rename(&tmp, &path);
}

/// Ensure token exists in agent-tokens.json; return the full token list.
fn ensure_token_in_file(config_dir: &std::path::Path, token: &str) -> Vec<String> {
    let mut tokens = load_agent_tokens(config_dir);
    if !tokens.iter().any(|t| t == token) {
        tokens.push(token.to_string());
    }
    save_agent_tokens(config_dir, &tokens);
    tokens
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReleaseAssetInfo {
    pub name: String,
    pub url: String,
    pub size: u64,
    pub platform: String,
    pub arch: String,
    pub archive: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatestReleaseInfo {
    pub tag_name: String,
    pub name: Option<String>,
    #[serde(default)]
    pub html_url: Option<String>,
    #[serde(default)]
    pub body: Option<String>,
    #[serde(default)]
    pub published_at: Option<String>,
    pub assets: Vec<ReleaseAssetInfo>,
    #[serde(default)]
    pub checksums_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteAgentDetectRequest {
    pub ssh_target: String,
    #[serde(default)]
    pub ssh_connection: Option<SshConnection>,
    pub agent_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteAgentDetectResponse {
    pub ok: bool,
    pub ssh_target: String,
    pub os: String,
    pub arch: String,
    pub install_path: Option<String>,
    pub start_command: Option<String>,
    pub installed: bool,
    pub reachable: bool,
    pub managed_task_name: Option<String>,
    pub managed_task_installed: bool,
    pub managed_task_command: Option<String>,
    pub managed_task_matches: bool,
    pub installed_version: Option<String>,
    pub latest_release: Option<LatestReleaseInfo>,
    pub matching_asset: Option<ReleaseAssetInfo>,
    pub update_available: bool,
    pub agent_token: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GithubRelease {
    tag_name: String,
    name: Option<String>,
    #[serde(default)]
    html_url: Option<String>,
    #[serde(default)]
    body: Option<String>,
    #[serde(default)]
    published_at: Option<String>,
    assets: Vec<GithubAsset>,
}

#[derive(Debug, Deserialize)]
struct GithubAsset {
    name: String,
    browser_download_url: String,
    size: u64,
}

pub async fn run_agent_server(app_config: Arc<AppConfig>) -> Result<()> {
    // Install the ring CryptoProvider before any rustls usage. The dashboard
    // mode gets this for free via reqwest/hyper-rustls, but the agent runs
    // without those and would otherwise panic at ServerConfig::builder().
    let _ = rustls::crypto::ring::default_provider().install_default();

    let bind_addr = format!("{}:{}", app_config.agent_host, app_config.agent_port)
        .parse::<SocketAddr>()
        .context("invalid agent bind address")?;

    let agent_config_dir = app_config.app_paths.root.clone();

    // Use explicit token, or auto-generate and persist one
    let token = match app_config.agent_token.clone() {
        Some(t) => t,
        None => {
            let config_dir = agent_config_dir.clone();
            let token_file = config_dir.join("agent-token");

            // Try to read existing token from disk
            let existing = std::fs::read_to_string(&token_file).ok().and_then(|s| {
                let trimmed = s.trim().to_string();
                if !trimmed.is_empty() {
                    Some(trimmed)
                } else {
                    None
                }
            });

            existing.unwrap_or_else(|| {
                // Generate a token exclusively from the operating system CSPRNG.
                // Never fall back to timestamps or process IDs for credentials.
                use rand::TryRng;
                use rand::rngs::SysRng;
                let mut entropy = [0u8; 16];
                SysRng
                    .try_fill_bytes(&mut entropy)
                    .expect("system entropy unavailable; refusing to create agent token");
                let bytes = u128::from_be_bytes(entropy);
                let new_token = format!("{bytes:x}");
                // Persist it
                let _ = std::fs::create_dir_all(&config_dir);
                let _ = std::fs::write(&token_file, &new_token);
                crate::config::harden_file_permissions(&token_file);
                // Log only a redacted prefix to avoid exposing the full token.
                let redacted = if new_token.len() > 8 {
                    format!("{}••••••••", &new_token[..4])
                } else {
                    "••••••••".to_string()
                };
                eprintln!("[agent] Auto-generated token: {redacted}");
                eprintln!("[agent] Token saved to {}", token_file.display());
                new_token
            })
        }
    };

    // Write token to a user-readable temp file so the main app can read it via SSH
    // (needed on Windows where the agent runs as SYSTEM and the token file is in
    // the SYSTEM profile, inaccessible to the SSH user).
    // The temp file is cleaned up after a delay to give the main app time to read it.
    let _ = write_token_to_temp_file(&token);

    // Ensure primary token is in agent-tokens.json (for multi-client support).
    let _ = ensure_token_in_file(&agent_config_dir, &token);

    let backend = gpu::detect_backend(&app_config.gpu_backend);
    let last_master_request = Arc::new(AtomicU64::new(0));
    let gpu_metrics: Arc<Mutex<BTreeMap<String, GpuMetrics>>> =
        Arc::new(Mutex::new(BTreeMap::new()));

    {
        let gpu_metrics = Arc::clone(&gpu_metrics);
        let backend = Arc::clone(&backend);
        let last_master_request = Arc::clone(&last_master_request);
        std::thread::spawn(move || {
            loop {
                if !master_request_is_recent(&last_master_request, unix_timestamp_seconds()) {
                    std::thread::sleep(AGENT_IDLE_CHECK_INTERVAL);
                    continue;
                }
                match backend.read_metrics() {
                    Ok(metrics) => {
                        if let Ok(mut lock) = gpu_metrics.lock() {
                            *lock = metrics;
                        }
                    }
                    Err(e) => eprintln!("[agent] GPU metrics unavailable: {e}"),
                }

                std::thread::sleep(Duration::from_millis(500));
            }
        });
    }

    let system_metrics: Arc<Mutex<system::SystemMetrics>> =
        Arc::new(Mutex::new(system::SystemMetrics::default()));

    {
        let system_metrics = Arc::clone(&system_metrics);
        let last_master_request = Arc::clone(&last_master_request);
        std::thread::spawn(move || {
            loop {
                if !master_request_is_recent(&last_master_request, unix_timestamp_seconds()) {
                    std::thread::sleep(AGENT_IDLE_CHECK_INTERVAL);
                    continue;
                }
                *system_metrics.lock().unwrap() = system::get_system_metrics();
                std::thread::sleep(Duration::from_secs(5));
            }
        });
    }

    let agent_info_token = token.clone(); // for authenticated /agent/info endpoint

    // Build token set: primary_token + all from agent-tokens.json
    let agent_config_dir = app_config.app_paths.root.clone();
    let extra_tokens = load_agent_tokens(&agent_config_dir);
    let mut allowed_tokens: Vec<String> = Vec::new();
    allowed_tokens.push(token.clone());
    for t in extra_tokens {
        if !allowed_tokens.contains(&t) {
            allowed_tokens.push(t);
        }
    }
    let allowed_tokens = std::sync::Arc::new(allowed_tokens);

    let auth = {
        let allowed = allowed_tokens.clone();
        let last_master_request = Arc::clone(&last_master_request);
        warp::any()
            .and(warp::header::headers_cloned())
            .and_then(move |headers: HeaderMap| {
                let allowed = allowed.clone();
                let last_master_request = last_master_request.clone();
                async move {
                    let bearer = headers
                        .get("authorization")
                        .and_then(|value| value.to_str().ok())
                        .and_then(|v| v.strip_prefix("Bearer "));

                    if let Some(tok) = bearer
                        && allowed.iter().any(|t| t == tok)
                    {
                        mark_master_request(&last_master_request);
                        return Ok::<(), warp::Rejection>(());
                    }
                    Err(warp::reject::custom(AgentAuthError))
                }
            })
    };

    let health = warp::path("health")
        .and(warp::get())
        .map(|| warp::reply::json(&serde_json::json!({ "ok": true })));

    let info = {
        let info_bind_addr = bind_addr;
        warp::path("info")
            .and(warp::get())
            .and(auth.clone())
            .map(move |_| {
                warp::reply::json(&serde_json::json!({
                    "ok": true,
                    "version": env!("CARGO_PKG_VERSION"),
                    "protocol_version": AGENT_PROTOCOL_VERSION,
                    "mode": "agent",
                    "pid": std::process::id(),
                    "executable": std::env::current_exe()
                        .ok()
                        .map(|path| path.to_string_lossy().to_string()),
                    "bind": info_bind_addr.to_string(),
                    "platform": std::env::consts::OS,
                    "arch": std::env::consts::ARCH,
                }))
            })
    };

    // Agent info endpoint (authenticated — returns token for token-verification flows)
    let agent_info = {
        let agent_token = agent_info_token.clone();
        let info_bind_addr = bind_addr;
        warp::path("agent")
            .and(warp::path("info"))
            .and(warp::get())
            .and(auth.clone())
            .map(move |_| {
                warp::reply::json(&serde_json::json!({
                    "ok": true,
                    "version": env!("CARGO_PKG_VERSION"),
                    "protocol_version": AGENT_PROTOCOL_VERSION,
                    "mode": "agent",
                    "pid": std::process::id(),
                    "executable": std::env::current_exe()
                        .ok()
                        .map(|path| path.to_string_lossy().to_string()),
                    "bind": info_bind_addr.to_string(),
                    "platform": std::env::consts::OS,
                    "arch": std::env::consts::ARCH,
                    "agent_token": agent_token,
                }))
            })
    };

    let system_route = {
        let system_metrics = Arc::clone(&system_metrics);
        warp::path!("metrics" / "system")
            .and(warp::get())
            .and(auth.clone())
            .map(move |_| warp::reply::json(&*system_metrics.lock().unwrap()))
    };

    let gpu_route = {
        let gpu_metrics = Arc::clone(&gpu_metrics);
        warp::path!("metrics" / "gpu")
            .and(warp::get())
            .and(auth.clone())
            .map(move |_| {
                let gpu = gpu_metrics.lock().unwrap().clone();
                warp::reply::json(&gpu)
            })
    };

    let metrics_route = {
        let gpu_metrics = Arc::clone(&gpu_metrics);
        let system_metrics = Arc::clone(&system_metrics);
        warp::path("metrics")
            .and(warp::path::end())
            .and(warp::get())
            .and(auth)
            .map(move |_| {
                let metrics = AgentMetrics {
                    system: system_metrics.lock().unwrap().clone(),
                    gpu: gpu_metrics.lock().unwrap().clone(),
                };
                warp::reply::json(&metrics)
            })
    };

    let routes = health
        .or(info)
        .or(agent_info)
        .or(system_route)
        .or(gpu_route)
        .or(metrics_route)
        .recover(handle_agent_rejection);

    if app_config.agent_token.is_none() {
        eprintln!("[agent] Using auto-generated token (persisted to config dir)");
    }

    // mTLS: load all CAs (legacy ca.pem + cas/ directory) and enforce client auth.
    let mut ca_pems: Vec<String> = Vec::new();

    // Legacy single CA (for backward compatibility)
    let legacy_ca_path = crate::certs::certs_dir().join("ca.pem");
    eprintln!(
        "[agent] Searching for CA: legacy path {}",
        legacy_ca_path.display()
    );
    if let Some(ca) =
        crate::certs::Cert::load(&legacy_ca_path, &crate::certs::certs_dir().join("ca.key"))
    {
        eprintln!("[agent] Loaded legacy CA from {}", legacy_ca_path.display());
        ca_pems.push(ca.pem);
    }

    // Collect all cas/ directories to search: the config-based certs dir and,
    // for managed installs, the directory next to the running binary (where the
    // dashboard drops the CA during remote install).
    let mut cas_dirs_to_search = vec![crate::certs::agent_cas_dir()];
    if let Ok(exe) = std::env::current_exe()
        && let Some(exe_dir) = exe.parent()
    {
        let install_cas = exe_dir.join("cas");
        if install_cas != crate::certs::agent_cas_dir() {
            cas_dirs_to_search.push(install_cas);
        }
    }

    for cas_dir in &cas_dirs_to_search {
        eprintln!("[agent] Searching for CAs in {}", cas_dir.display());
        if let Ok(entries) = std::fs::read_dir(cas_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("pem")
                    && let Ok(pem) = std::fs::read_to_string(&path)
                {
                    eprintln!("[agent] Loaded CA from {}", path.display());
                    ca_pems.push(pem);
                }
            }
        } else {
            eprintln!(
                "[agent] CA directory not found or unreadable: {}",
                cas_dir.display()
            );
        }
    }

    if ca_pems.is_empty() {
        eprintln!("[agent] No CA found for mTLS; agent cannot start without trust anchors");
        return Err(anyhow::anyhow!("agent mTLS: no CA found"));
    }

    eprintln!(
        "[info] Agent mTLS: loaded {} CA(s) into trust store",
        ca_pems.len()
    );

    let agent_server_cert = if let Some(cert) = load_install_dir_agent_server_cert() {
        eprintln!("[agent] Using pre-provisioned server cert from install dir");
        cert
    } else {
        eprintln!("[agent] No pre-provisioned server cert found; generating self-signed cert");
        crate::certs::ensure_agent_server_cert(vec![
            "localhost".to_string(),
            "127.0.0.1".to_string(),
        ])
    };

    eprintln!("[agent] Building mTLS server config...");
    let tls_config = match crate::certs::build_agent_tls_config(ca_pems, agent_server_cert.clone())
    {
        Ok(cfg) => {
            eprintln!("[agent] mTLS server config built successfully");
            cfg
        }
        Err(e) => {
            eprintln!("[agent] Failed to build mTLS config: {e}; agent cannot start without mTLS");
            return Err(anyhow::anyhow!("agent mTLS config build failed: {e}"));
        }
    };

    // Watch channel allows hot-reloading the mTLS ServerConfig when new client CAs are
    // registered via the enrollment port, without restarting the agent.
    let (tls_tx, tls_rx) = tokio::sync::watch::channel(std::sync::Arc::new(tls_config));
    let tls_tx = std::sync::Arc::new(tls_tx);

    // Spawn enrollment server (agent port + 1) for unenrolled clients to register their CA.
    // Authenticated by the same api-token as the main agent API. Clients obtain the token
    // automatically via SSH during bootstrap, so no manual credential handling is required.
    let enrollment_port = bind_addr.port().saturating_add(ENROLLMENT_PORT_OFFSET);
    let enrollment_addr = SocketAddr::new(bind_addr.ip(), enrollment_port);
    let enroll_tokens = allowed_tokens.clone();
    let enroll_tx = tls_tx.clone();
    let enroll_server_cert = agent_server_cert.clone();
    tokio::spawn(async move {
        if let Err(e) = run_enrollment_server(
            enrollment_addr,
            enroll_server_cert,
            enroll_tokens,
            enroll_tx,
        )
        .await
        {
            eprintln!("[agent] Enrollment server stopped: {e}");
        }
    });

    let listener = match tokio::net::TcpListener::bind(bind_addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("[agent] Failed to bind TLS listener: {e}");
            return Err(anyhow::anyhow!("Failed to bind TLS listener: {e}"));
        }
    };

    println!("[agent] Remote metrics agent listening on https://{bind_addr} (mTLS enforced)");
    println!(
        "[agent] CA enrollment available on https://{enrollment_addr} (server-TLS, api-token auth)"
    );
    println!(
        "[agent] New clients enroll automatically via SSH bootstrap — no manual steps required"
    );

    loop {
        let (stream, peer) = match listener.accept().await {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[agent] accept error: {e}");
                continue;
            }
        };

        // Read current TLS config from watch — picks up newly registered client CAs
        // without restarting. Creating TlsAcceptor per-connection is cheap relative
        // to the TLS handshake itself.
        let current_config = tls_rx.borrow().clone();
        let acceptor = tokio_rustls::TlsAcceptor::from(current_config);
        let routes_clone = routes.clone();

        tokio::spawn(async move {
            let tls_stream = match acceptor.accept(stream).await {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("[agent] TLS handshake error from {peer}: {e}");
                    return;
                }
            };

            // Log client cert subject if present (mTLS)
            {
                let (_, conn) = tls_stream.get_ref();
                let client_certs = conn.peer_certificates();
                if let Some(certs) = client_certs
                    && !certs.is_empty()
                {
                    let subject = crate::certs::extract_cert_subject_pem(&certs[0]);
                    eprintln!(
                        "[info] Agent mTLS connection accepted from {peer}, client subject: {subject}"
                    );
                } else {
                    eprintln!(
                        "[agent] connection from {peer} without client cert; mTLS will reject"
                    );
                }
            }

            let svc = warp::service(routes_clone);
            let svc = hyper_util::service::TowerToHyperService::new(svc);
            let io = hyper_util::rt::TokioIo::new(tls_stream);

            if let Err(e) =
                hyper_util::server::conn::auto::Builder::new(hyper_util::rt::TokioExecutor::new())
                    .http1()
                    .serve_connection_with_upgrades(io, svc)
                    .await
            {
                eprintln!("[agent] connection error from {peer}: {e}");
            }
        });
    }
}

/// Runs the enrollment server on the agent's secondary port (agent port + 1).
///
/// Accepts unenrolled clients (no mTLS client cert required). Exposes a single
/// endpoint: `POST /api/agent/register-ca`. Authenticated by the agent's api-token,
/// which clients obtain automatically via SSH during bootstrap enrollment.
///
/// On successful registration, writes the new CA to `cas/<instance_id>.pem` and
/// sends an updated `ServerConfig` to the watch channel, causing the main mTLS
/// acceptor to trust the new CA for subsequent connections — no agent restart needed.
async fn run_enrollment_server(
    addr: SocketAddr,
    server_cert: crate::certs::Cert,
    allowed_tokens: std::sync::Arc<Vec<String>>,
    tls_tx: std::sync::Arc<tokio::sync::watch::Sender<std::sync::Arc<rustls::ServerConfig>>>,
) -> Result<()> {
    let enrollment_tls_config = crate::certs::build_enrollment_tls_config(&server_cert)
        .map_err(|e| anyhow::anyhow!("enrollment TLS build failed: {e}"))?;
    let enrollment_tls_acceptor =
        tokio_rustls::TlsAcceptor::from(std::sync::Arc::new(enrollment_tls_config));

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| anyhow::anyhow!("enrollment server bind failed on {addr}: {e}"))?;

    // Enrollment route: POST /api/agent/register-ca
    let route = {
        let allowed = allowed_tokens.clone();
        let tx = tls_tx.clone();
        let srv_cert = server_cert.clone();
        warp::path!("api" / "agent" / "register-ca")
            .and(warp::post())
            .and(warp::header::headers_cloned())
            .and(warp::body::content_length_limit(65_536)) // 64 KB max
            .and(warp::body::json::<RegisterCaRequest>())
            .and_then(
                move |headers: warp::http::HeaderMap, body: RegisterCaRequest| {
                    let allowed = allowed.clone();
                    let tx = tx.clone();
                    let srv_cert = srv_cert.clone();
                    async move { handle_register_ca(headers, body, allowed, tx, srv_cert).await }
                },
            )
    };

    loop {
        let (stream, peer) = match listener.accept().await {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[enroll] accept error: {e}");
                continue;
            }
        };

        let acceptor = enrollment_tls_acceptor.clone();
        let route_clone = route.clone();

        tokio::spawn(async move {
            let tls_stream = match acceptor.accept(stream).await {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("[enroll] TLS error from {peer}: {e}");
                    return;
                }
            };

            let svc = warp::service(route_clone);
            let svc = hyper_util::service::TowerToHyperService::new(svc);
            let io = hyper_util::rt::TokioIo::new(tls_stream);

            if let Err(e) =
                hyper_util::server::conn::auto::Builder::new(hyper_util::rt::TokioExecutor::new())
                    .http1()
                    .serve_connection_with_upgrades(io, svc)
                    .await
            {
                eprintln!("[enroll] connection error from {peer}: {e}");
            }
        });
    }
}

/// Handler for `POST /api/agent/register-ca`.
///
/// Validates the submitted CA PEM, writes it to `cas/<instance_id>.pem`, and
/// hot-reloads the main mTLS ServerConfig so subsequent connections from this
/// client succeed without an agent restart.
///
/// Auth: api-token bearer — clients obtain this automatically via SSH during bootstrap.
async fn handle_register_ca(
    headers: warp::http::HeaderMap,
    body: RegisterCaRequest,
    allowed_tokens: std::sync::Arc<Vec<String>>,
    tls_tx: std::sync::Arc<tokio::sync::watch::Sender<std::sync::Arc<rustls::ServerConfig>>>,
    server_cert: crate::certs::Cert,
) -> Result<impl warp::Reply, warp::Rejection> {
    use subtle::ConstantTimeEq;

    let bearer = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));

    let authed = if let Some(tok) = bearer {
        allowed_tokens
            .iter()
            .any(|t| tok.as_bytes().ct_eq(t.as_bytes()).into())
    } else {
        false
    };

    if !authed {
        eprintln!("[enroll] Rejected: invalid or missing api-token");
        return Ok(warp::reply::with_status(
            warp::reply::json(&serde_json::json!({
                "ok": false,
                "error": "unauthorized — provide api-token as Bearer"
            })),
            warp::http::StatusCode::UNAUTHORIZED,
        ));
    }

    // Confirmation field prevents accidental registration.
    if body.confirm != "register-ca" {
        return Ok(warp::reply::with_status(
            warp::reply::json(&serde_json::json!({
                "ok": false,
                "error": "missing or invalid confirm field; must be \"register-ca\""
            })),
            warp::http::StatusCode::BAD_REQUEST,
        ));
    }

    // Validate CA PEM and derive a stable instance ID.
    let instance_id = match crate::certs::validate_and_get_ca_instance_id(&body.ca_pem) {
        Ok(id) => id,
        Err(reason) => {
            return Ok(warp::reply::with_status(
                warp::reply::json(&serde_json::json!({
                    "ok": false,
                    "error": format!("invalid CA PEM: {reason}")
                })),
                warp::http::StatusCode::BAD_REQUEST,
            ));
        }
    };

    // Check if this CA is already trusted.
    let cas_dir = crate::certs::agent_cas_dir();
    let ca_path = cas_dir.join(format!("{instance_id}.pem"));
    if ca_path.exists() {
        return Ok(warp::reply::with_status(
            warp::reply::json(&serde_json::json!({
                "ok": true,
                "already_trusted": true,
                "instance_id": instance_id
            })),
            warp::http::StatusCode::OK,
        ));
    }

    // Write CA to cas/ directory with restrictive permissions.
    if let Err(e) = std::fs::write(&ca_path, &body.ca_pem) {
        eprintln!("[enroll] Failed to write CA {instance_id}: {e}");
        return Ok(warp::reply::with_status(
            warp::reply::json(&serde_json::json!({
                "ok": false,
                "error": "failed to persist CA certificate"
            })),
            warp::http::StatusCode::INTERNAL_SERVER_ERROR,
        ));
    }
    crate::config::harden_file_permissions(&ca_path);

    eprintln!(
        "[agent] Registered new client CA: instance_id={instance_id}, path={}",
        ca_path.display()
    );

    // Hot-reload mTLS trust store: rebuild ServerConfig from all CAs now on disk.
    let all_cas = crate::certs::load_all_agent_cas();
    match crate::certs::build_agent_tls_config(all_cas, server_cert) {
        Ok(new_config) => {
            let _ = tls_tx.send(std::sync::Arc::new(new_config));
            eprintln!("[agent] mTLS trust store reloaded; new client can now connect");
        }
        Err(e) => {
            // Non-fatal: CA is on disk, but in-memory reload failed.
            // The agent can be restarted to pick it up.
            eprintln!("[agent] Warning: CA written to disk but in-memory reload failed: {e}");
        }
    }

    Ok(warp::reply::with_status(
        warp::reply::json(&serde_json::json!({
            "ok": true,
            "already_trusted": false,
            "instance_id": instance_id
        })),
        warp::http::StatusCode::OK,
    ))
}

pub async fn latest_release_info() -> Result<LatestReleaseInfo> {
    if std::env::var("LLAMA_SKIP_RELEASE_CHECK").is_ok() {
        anyhow::bail!("release check disabled (LLAMA_SKIP_RELEASE_CHECK)");
    }

    let cached = LATEST_RELEASE_CACHE.with(|cache| {
        let now = Instant::now();
        let cached = cache.try_lock().ok()?;
        if let Some((ref info, ref cached_at)) = *cached
            && now.duration_since(*cached_at) < Duration::from_secs(60)
        {
            return Some(info.clone());
        }
        None
    });

    if let Some(cached) = cached {
        return Ok(cached);
    }

    let release = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs(30))
        .build()?
        .get(GITHUB_LATEST_RELEASE_URL)
        .header(reqwest::header::USER_AGENT, "local-llm-foundry")
        .send()
        .await?
        .error_for_status()?
        .json::<GithubRelease>()
        .await?;

    let checksums_url = release
        .assets
        .iter()
        .find(|asset| asset.name == "checksums.json")
        .map(|asset| asset.browser_download_url.clone());
    let release_info = LatestReleaseInfo {
        tag_name: release.tag_name,
        name: release.name,
        html_url: release.html_url,
        body: release.body,
        published_at: release.published_at,
        assets: release
            .assets
            .into_iter()
            .filter_map(asset_info_from_github_asset)
            .collect(),
        checksums_url,
    };

    LATEST_RELEASE_CACHE.with(|cache| {
        let _ = cache
            .try_lock()
            .map(|mut cached| *cached = Some((release_info.clone(), Instant::now())));
    });

    Ok(release_info)
}

pub async fn detect_remote_agent(req: RemoteAgentDetectRequest) -> RemoteAgentDetectResponse {
    let connection = req
        .ssh_connection
        .clone()
        .unwrap_or_else(|| SshConnection::from_target(&req.ssh_target));
    let ssh_target = if req.ssh_target.trim().is_empty() {
        connection.target_label()
    } else {
        req.ssh_target.trim().to_string()
    };

    if ssh_target.is_empty() || connection.host.trim().is_empty() {
        return RemoteAgentDetectResponse {
            ok: false,
            ssh_target,
            os: "unknown".to_string(),
            arch: "unknown".to_string(),
            install_path: None,
            start_command: None,
            installed: false,
            reachable: false,
            managed_task_name: None,
            managed_task_installed: false,
            managed_task_command: None,
            managed_task_matches: false,
            installed_version: None,
            latest_release: None,
            matching_asset: None,
            update_available: false,
            agent_token: None,
            error: Some("Missing SSH target".to_string()),
        };
    }

    let remote_os = detect_remote_os_with(&connection).await;
    let os = remote_os.as_str().to_string();
    let arch = detect_remote_arch_with(&connection, remote_os).await;
    let detected_install_path = preferred_remote_install_path(&connection, remote_os).await;
    let installed = remote_file_exists_with(&connection, remote_os, &detected_install_path).await;
    let install_path = Some(detected_install_path);
    let managed_task =
        install::managed_task_status(&connection, remote_os, install_path.as_deref())
            .await
            .ok()
            .flatten();
    let reachable = if let Some(agent_url) = req.agent_url.as_deref() {
        agent_health_reachable(agent_url).await
    } else {
        false
    };

    let ssh_ok = remote_os != RemoteOs::Unknown;

    let installed_version = if installed && ssh_ok {
        install::get_remote_version_with(connection.clone())
            .await
            .ok()
            .flatten()
    } else {
        None
    };

    let agent_token = if ssh_ok {
        read_remote_agent_token(&connection, remote_os, req.agent_url.as_deref()).await
    } else {
        None
    };

    let latest_release = latest_release_info().await.ok();
    let matching_asset = latest_release
        .as_ref()
        .and_then(|release| release.matching_asset(&os, &arch).cloned());

    let update_available = if let (Some(installed_ver), Some(latest)) =
        (installed_version.as_deref(), latest_release.as_ref())
    {
        normalize_version_label(installed_ver) != normalize_version_label(&latest.tag_name)
    } else {
        false
    };

    let start_command = if ssh_ok {
        default_start_command_for_os_with(
            &connection,
            remote_os,
            install_path
                .as_deref()
                .unwrap_or("~/.config/local-llm-foundry/bin/local-llm-foundry"),
        )
        .await
    } else {
        String::new()
    };

    let error = if !ssh_ok {
        Some("Could not detect remote OS over SSH. Verify SSH connectivity and that remote host allows command execution.".to_string())
    } else if matching_asset.is_none() {
        Some(remote_release_asset_error(
            latest_release.as_ref(),
            &os,
            &arch,
            install_path.as_deref(),
        ))
    } else {
        None
    };

    RemoteAgentDetectResponse {
        ok: error.is_none(),
        ssh_target,
        os,
        arch,
        install_path,
        start_command: if start_command.is_empty() {
            None
        } else {
            Some(start_command)
        },
        installed,
        reachable,
        managed_task_name: managed_task.as_ref().map(|task| task.name.clone()),
        managed_task_installed: managed_task.as_ref().is_some_and(|task| task.installed),
        managed_task_command: managed_task.as_ref().and_then(|task| task.command.clone()),
        managed_task_matches: managed_task
            .as_ref()
            .is_some_and(|task| task.matches_install_path),
        installed_version,
        latest_release,
        matching_asset,
        update_available,
        agent_token,
        error,
    }
}

fn remote_release_asset_error(
    latest_release: Option<&LatestReleaseInfo>,
    os: &str,
    arch: &str,
    install_path: Option<&str>,
) -> String {
    let install_path = install_path.unwrap_or("the managed agent install path");

    match latest_release {
        Some(release) if release.assets.is_empty() => format!(
            "Latest release {} is published but does not have any installable agent assets yet. This usually means the release build is still running or asset upload has not finished. Wait for the release artifacts to appear, then retry Install / Start. Expected asset for this host: {} {}. Target install path: {}.",
            release.tag_name, os, arch, install_path
        ),
        Some(release) => format!(
            "Latest release {} does not contain a supported agent asset for {} {}. Open the release artifacts and verify that the expected package was published for this platform, then retry. Target install path: {}.",
            release.tag_name, os, arch, install_path
        ),
        None => format!(
            "Could not determine a downloadable agent build for {} {} because release metadata was unavailable. Check GitHub release availability, then retry. Target install path: {}.",
            os, arch, install_path
        ),
    }
}

pub async fn remote_agent_poller(state: AppState, app_config: Arc<AppConfig>) {
    let https_client = build_agent_https_client(Duration::from_secs(2));
    let http_client = build_plain_http_client(Duration::from_secs(2));
    let mut autostart_attempted = false;
    let mut enabled = false;
    // Tracks which agent base URLs we have already attempted (or confirmed) enrollment
    // for in this session. Avoids re-running enrollment on every poll after success.
    let mut enrolled_urls: HashSet<String> = HashSet::new();

    loop {
        if !enabled {
            state.llama_poll_notify.notified().await;
            enabled = true;
        }

        let settings = state.ui_settings.lock().unwrap().clone();
        let configured_url = first_non_empty([
            app_config.remote_agent_url.as_deref(),
            Some(settings.remote_agent_url.as_str()),
        ]);
        let url = remote_agent_url_for_active_session(&state, configured_url.as_deref());
        let token = first_non_empty([
            app_config.remote_agent_token.as_deref(),
            Some(settings.remote_agent_token.as_str()),
        ]);

        if let Some(url) = url {
            let mut metrics_result = None;
            let mut saw_unauthorized = false;
            let mut saw_connect_error = false;

            for candidate in agent_url_candidates(&url) {
                let client = agent_client_for_url(&candidate, https_client.as_ref(), &http_client);
                let mut request = client.get(format!("{}/metrics", candidate));
                if let Some(token) = &token {
                    request = request.bearer_auth(token);
                }

                match request.send().await {
                    Ok(resp) if resp.status().is_success() => {
                        // Successful connection — mark enrollment confirmed for this base URL.
                        enrolled_urls.insert(url.trim_end_matches('/').to_string());
                        metrics_result = Some((candidate, resp));
                        break;
                    }
                    Ok(resp) => {
                        if resp.status() == reqwest::StatusCode::UNAUTHORIZED {
                            saw_unauthorized = true;
                        }
                    }
                    Err(e) if e.is_connect() || e.is_timeout() => {
                        saw_connect_error = true;
                    }
                    Err(_) => {}
                }
            }

            // Auto-enrollment: if we got a connection-level error (mTLS rejection) and SSH
            // is configured, bootstrap trust automatically via SSH — no user interaction needed.
            if metrics_result.is_none() && saw_connect_error {
                let base_url = url.trim_end_matches('/').to_string();
                if !enrolled_urls.contains(&base_url) {
                    let ssh_target = first_non_empty([
                        app_config.remote_agent_ssh_target.as_deref(),
                        Some(settings.remote_agent_ssh_target.as_str()),
                    ])
                    .or_else(|| remote_host_from_agent_url(&url));

                    if let Some(target) = ssh_target {
                        match remote_ssh::with_trusted_host_key(
                            SshConnection::from_target(&target),
                            &app_config.ssh_known_hosts_file,
                        ) {
                            Ok(connection) if connection.trusted_host_key.is_some() => {
                                enrolled_urls.insert(base_url); // prevent retry loops
                                bootstrap_client_enrollment(&connection, &url, &state, &app_config)
                                    .await;
                            }
                            _ => {
                                enrolled_urls.insert(base_url);
                                eprintln!(
                                    "[agent] SSH enrollment blocked by host key check for {target}"
                                );
                            }
                        }
                    } else {
                        enrolled_urls.insert(base_url);
                        eprintln!(
                            "[agent] mTLS connect failed for {url} and no SSH target configured; \
                             set remote_agent_ssh_target to enable automatic enrollment"
                        );
                    }
                }
            }

            // A reinstall can rotate the remote token while leaving the agent
            // process and its health endpoint running. Recover the durable
            // token over the already-trusted SSH channel instead of treating
            // HTTP 401 as a permanently disconnected agent.
            if metrics_result.is_none() && saw_unauthorized {
                refresh_remote_agent_token(&state, &app_config, &settings, &url).await;
            }

            match metrics_result {
                Some((resolved_url, resp)) => match resp.json::<AgentMetrics>().await {
                    Ok(metrics) => {
                        *state.system_metrics.lock().unwrap() = metrics.system;
                        *state.gpu_metrics.lock().unwrap() = metrics.gpu;
                        *state.remote_agent_connected.lock().unwrap() = true;
                        *state.remote_agent_health_reachable.lock().unwrap() = true;
                        *state.remote_agent_url.lock().unwrap() = Some(resolved_url.clone());
                        state.refresh_capability_state();
                        autostart_attempted = false;

                        // Fetch agent version and protocol_version from /info endpoint
                        let info_client = agent_client_for_url(
                            &resolved_url,
                            https_client.as_ref(),
                            &http_client,
                        );
                        let info_url = format!("{}/info", resolved_url.trim_end_matches('/'));
                        let mut info_request = info_client.get(&info_url);
                        if let Some(t) = &token {
                            info_request = info_request.bearer_auth(t);
                        }
                        match info_request.send().await {
                            Ok(info_resp) if info_resp.status().is_success() => {
                                match info_resp.json::<serde_json::Value>().await {
                                    Ok(json) => {
                                        // Handle version
                                        if let Some(ver) =
                                            json.get("version").and_then(|v| v.as_str())
                                        {
                                            let version_str = ver.to_string();
                                            let should_check = {
                                                let mut version =
                                                    state.remote_agent_version.lock().unwrap();
                                                let is_new = version.is_none()
                                                    || version.as_deref() != Some(&version_str);
                                                *version = Some(version_str.clone());
                                                is_new
                                            };

                                            // Only check GitHub once per session (on first discovery or version change)
                                            if should_check {
                                                match latest_release_info().await {
                                                    Ok(latest) => {
                                                        let latest_ver = latest
                                                            .tag_name
                                                            .strip_prefix('v')
                                                            .unwrap_or(&latest.tag_name);
                                                        let needs_update =
                                                            version_str != latest_ver;
                                                        let mut update_avail = state
                                                            .remote_agent_update_available
                                                            .lock()
                                                            .unwrap();
                                                        *update_avail = needs_update;
                                                        drop(update_avail);
                                                        if needs_update {
                                                            eprintln!(
                                                                "[agent] Update available: running {}, latest {}",
                                                                version_str, latest.tag_name
                                                            );
                                                        }
                                                    }
                                                    Err(e) => {
                                                        eprintln!(
                                                            "[agent] Could not check latest release: {e}"
                                                        );
                                                    }
                                                }
                                            }
                                        }

                                        // Handle protocol_version: enforce minimum with graceful degradation
                                        let agent_proto = json
                                            .get("protocol_version")
                                            .and_then(|v| v.as_str())
                                            .map(String::from);

                                        let proto_too_old = match &agent_proto {
                                            Some(v) if v.as_str() < "1.0.0" => true,
                                            Some(_) => false,
                                            None => {
                                                // Unknown protocol → treat as potentially old; enable degraded mode
                                                true
                                            }
                                        };

                                        {
                                            let mut pv =
                                                state.remote_agent_protocol_version.lock().unwrap();
                                            *pv = agent_proto.clone();
                                        }
                                        {
                                            let mut flag =
                                                state.remote_agent_protocol_too_old.lock().unwrap();
                                            *flag = proto_too_old;
                                        }

                                        if proto_too_old {
                                            eprintln!(
                                                "[agent] Agent protocol version ({:?}) is below minimum (1.0.0); running in degraded compatibility mode",
                                                agent_proto.as_deref().unwrap_or("unknown")
                                            );
                                        }
                                    }
                                    Err(e) => {
                                        eprintln!("[agent] Failed to parse /info response: {e}");
                                    }
                                }
                            }
                            Ok(info_resp) => {
                                eprintln!(
                                    "[agent] /info request failed: HTTP {}",
                                    info_resp.status()
                                );
                            }
                            Err(e) => {
                                eprintln!("[agent] /info request error: {e}");
                            }
                        }
                    }
                    Err(e) => {
                        // Don’t fully disconnect on a single parse error if HTTP succeeded.
                        // Treat as degraded instead of disconnected to preserve partial metrics.
                        eprintln!("[agent] Metrics parse failed (degraded mode): {e}");
                        // Keep connected=true but allow degraded mode flags to inform the frontend.
                    }
                },
                None => {
                    mark_disconnected(&state);
                    if saw_unauthorized && token.is_none() {
                        eprintln!("[agent] Remote agent not yet authenticated (no token set)");
                    }
                    maybe_autostart_remote_agent(
                        &state,
                        &app_config,
                        &settings,
                        &url,
                        &mut autostart_attempted,
                    )
                    .await;
                }
            }
        } else {
            mark_disconnected(&state);
            enabled = false;
        }

        // T-048: slow poll when in low-power mode
        let mode = state.sleep_mode.load(std::sync::atomic::Ordering::Relaxed);
        let interval = if mode >= 1 {
            if let Ok(cfg) = state.sleep_mode_config.lock() {
                Duration::from_secs(cfg.sleep_llama_interval_secs.max(1))
            } else {
                REMOTE_AGENT_POLL_INTERVAL
            }
        } else {
            REMOTE_AGENT_POLL_INTERVAL
        };

        tokio::select! {
            _ = tokio::time::sleep(interval) => {}
            _ = state.agent_poll_notify.notified() => {}
        }
    }
}

pub fn suppress_remote_agent_autostart() {
    if let Ok(mut until) = REMOTE_AGENT_AUTOSTART_SUPPRESS_UNTIL.lock() {
        *until = Some(Instant::now() + REMOTE_AGENT_AUTOSTART_SUPPRESS_DURATION);
    }
}

async fn maybe_autostart_remote_agent(
    state: &AppState,
    app_config: &AppConfig,
    settings: &crate::state::UiSettings,
    agent_url: &str,
    attempted: &mut bool,
) {
    let enabled = app_config.remote_agent_ssh_autostart || settings.remote_agent_ssh_autostart;
    if !enabled {
        return;
    }

    if state.current_endpoint_kind() != EndpointKind::Remote {
        return;
    }

    if REMOTE_AGENT_AUTOSTART_SUPPRESS_UNTIL
        .lock()
        .ok()
        .and_then(|until| *until)
        .is_some_and(|until| Instant::now() < until)
    {
        return;
    }

    if *attempted {
        return;
    }
    *attempted = true;

    let target = first_non_empty([
        app_config.remote_agent_ssh_target.as_deref(),
        Some(settings.remote_agent_ssh_target.as_str()),
    ])
    .or_else(|| remote_host_from_agent_url(agent_url));

    let Some(target) = target else {
        eprintln!("[agent] SSH autostart enabled but no SSH target is available");
        return;
    };

    let connection = match remote_ssh::with_trusted_host_key(
        SshConnection::from_target(&target),
        &app_config.ssh_known_hosts_file,
    ) {
        Ok(connection) => connection,
        Err(e) => {
            eprintln!("[agent] Remote agent autostart blocked by SSH trust check: {e}");
            return;
        }
    };

    if connection.trusted_host_key.is_none() {
        return;
    }

    let remote_os = detect_remote_os_with(&connection).await;
    let default_install_path = preferred_remote_install_path(&connection, remote_os).await;
    let default_command =
        default_start_command_for_os_with(&connection, remote_os, &default_install_path).await;

    let command = if let Some(command) = first_non_empty([
        app_config.remote_agent_ssh_command.as_deref(),
        Some(settings.remote_agent_ssh_command.as_str()),
    ]) {
        if remote_os == RemoteOs::Windows && command.contains('~') {
            default_command
        } else if validate_remote_command(&command) {
            command
        } else {
            eprintln!(
                "[agent] Autostart: remote_agent_ssh_command failed validation; using default command"
            );
            default_command
        }
    } else {
        default_command
    };

    eprintln!("[agent] Attempting remote agent autostart via ssh {target}");

    let started = match tokio::time::timeout(
        REMOTE_AGENT_AUTOSTART_TIMEOUT,
        remote_ssh::exec(connection.clone(), command),
    )
    .await
    {
        Ok(Ok(output)) if output.status == 0 => {
            eprintln!("[agent] Remote agent autostart command completed");
            true
        }
        Ok(Ok(output)) => {
            eprintln!(
                "[agent] Remote agent autostart command exited with status {}: {}",
                output.status,
                output.stderr.trim()
            );
            false
        }
        Ok(Err(e)) => {
            eprintln!("[agent] Remote agent autostart command failed: {e}");
            false
        }
        Err(_) => {
            eprintln!("[agent] Remote agent autostart timed out; use a detached remote command");
            false
        }
    };

    // After a successful autostart, read and persist the token so the metrics
    // poller can authenticate on its next attempt.
    if started
        && settings.remote_agent_token.is_empty()
        && let Some(token) =
            read_remote_agent_token(&connection, remote_os, Some(&settings.remote_agent_url)).await
    {
        let mut s = state.ui_settings.lock().unwrap();
        if s.remote_agent_token.is_empty() {
            s.remote_agent_token = token;
            let _ = crate::state::save_ui_settings(&state.ui_settings_path, &s);
            drop(s);
            state.agent_poll_notify.notify_waiters();
        }
    }
}

async fn refresh_remote_agent_token(
    state: &AppState,
    app_config: &AppConfig,
    settings: &crate::state::UiSettings,
    agent_url: &str,
) {
    let target = first_non_empty([
        app_config.remote_agent_ssh_target.as_deref(),
        Some(settings.remote_agent_ssh_target.as_str()),
    ])
    .or_else(|| remote_host_from_agent_url(agent_url));
    let Some(target) = target else {
        return;
    };

    let Ok(connection) = remote_ssh::with_trusted_host_key(
        SshConnection::from_target(&target),
        &app_config.ssh_known_hosts_file,
    ) else {
        return;
    };
    if connection.trusted_host_key.is_none() {
        return;
    }

    let os = detect_remote_os_with(&connection).await;
    let Some(token) = read_remote_agent_token(&connection, os, Some(agent_url)).await else {
        return;
    };

    let mut current = state.ui_settings.lock().unwrap();
    if current.remote_agent_token == token {
        return;
    }
    current.remote_agent_token = token;
    if let Err(error) = crate::state::save_ui_settings(&state.ui_settings_path, &current) {
        eprintln!("[agent] Failed to persist refreshed remote-agent token: {error}");
        return;
    }
    eprintln!("[agent] Refreshed remote-agent token after an unauthorized metrics response");
    drop(current);
    state.agent_poll_notify.notify_waiters();
}

fn mark_disconnected(state: &AppState) {
    let was_connected = {
        let mut connected = state.remote_agent_connected.lock().unwrap();
        let was_connected = *connected;
        *connected = false;
        was_connected
    };
    *state.remote_agent_health_reachable.lock().unwrap() = false;
    *state.remote_agent_version.lock().unwrap() = None;
    *state.remote_agent_update_available.lock().unwrap() = false;
    if was_connected {
        state.refresh_capability_state();
    }
}

fn remote_agent_url_for_active_session(
    state: &AppState,
    configured_url: Option<&str>,
) -> Option<String> {
    // When a local session is active (Spawn or localhost Attach), never poll the remote
    // agent for metrics — it would overwrite local gpu/system data with remote data.
    {
        let active_id = state.active_session_id.lock().unwrap().clone();
        if !active_id.is_empty() && state.active_session_uses_local_metrics() {
            return None;
        }
    }

    if let Some(url) = configured_url.filter(|url| !url.trim().is_empty()) {
        return Some(url.trim().to_string());
    }

    if state.current_endpoint_kind() != EndpointKind::Remote {
        return None;
    }

    let active_id = state.active_session_id.lock().unwrap().clone();
    let session = {
        let sessions = state.sessions.lock().unwrap();
        sessions.iter().find(|s| s.id == active_id).cloned()
    }?;

    let SessionMode::Attach { endpoint, .. } = session.mode else {
        return None;
    };

    let url = reqwest::Url::parse(&endpoint)
        .or_else(|_| reqwest::Url::parse(&format!("http://{endpoint}")))
        .ok()?;
    let host = url.host_str()?;
    Some(format!("https://{host}:{REMOTE_AGENT_DEFAULT_PORT}"))
}

fn first_non_empty(values: [Option<&str>; 2]) -> Option<String> {
    values
        .into_iter()
        .flatten()
        .map(str::trim)
        .find(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn remote_host_from_agent_url(agent_url: &str) -> Option<String> {
    let url = reqwest::Url::parse(agent_url).ok()?;
    url.host_str().map(ToOwned::to_owned)
}

fn default_start_command_for_os(os: RemoteOs, install_path: &str) -> String {
    let quoted_path = shell_quote_path(install_path, os);
    match os {
        RemoteOs::Windows => {
            // Use PowerShell to create scheduled tasks — handles path quoting
            // reliably across SSH layers without backslash consumption.
            let agent_path = install_path.replace('\'', "''");
            let bridge_dir = install_path
                .rsplit_once('\\')
                .map(|(dir, _)| dir)
                .unwrap_or("");
            let bridge_path = format!("{}\\sensor_bridge.exe", bridge_dir).replace('\'', "''");
            let config_dir = bridge_dir
                .rsplit_once('\\')
                .map(|(dir, _)| dir)
                .unwrap_or(bridge_dir)
                .replace('\'', "''");
            format!(
                "powershell.exe -NoProfile -NonInteractive -Command \"$ErrorActionPreference='Stop'; \
            Unregister-ScheduledTask -TaskName '{WINDOWS_AGENT_LEGACY_TASK_NAME}' -Confirm:$false -ErrorAction SilentlyContinue; \
            Unregister-ScheduledTask -TaskName '{WINDOWS_AGENT_TASK_NAME}' -Confirm:$false -ErrorAction SilentlyContinue; \
            Unregister-ScheduledTask -TaskName '{WINDOWS_SENSOR_BRIDGE_TASK_NAME}' -Confirm:$false -ErrorAction SilentlyContinue; \
            Register-ScheduledTask -TaskName '{WINDOWS_AGENT_TASK_NAME}' -Trigger (New-ScheduledTaskTrigger -AtStartup) -Action (New-ScheduledTaskAction -Execute '{agent_path}' -Argument '--agent --config-dir \\\"{config_dir}\\\" --agent-host 0.0.0.0 --agent-port {REMOTE_AGENT_DEFAULT_PORT}') -Settings (New-ScheduledTaskSettingsSet) -User 'SYSTEM' -RunLevel Highest -Force; \
            Register-ScheduledTask -TaskName '{WINDOWS_SENSOR_BRIDGE_TASK_NAME}' -Trigger (New-ScheduledTaskTrigger -AtStartup) -Action (New-ScheduledTaskAction -Execute '{bridge_path}' -Argument '--server') -Settings (New-ScheduledTaskSettingsSet) -User 'SYSTEM' -RunLevel Highest -Force; \
            Start-ScheduledTask -TaskName '{WINDOWS_AGENT_TASK_NAME}'; \
            Start-ScheduledTask -TaskName '{WINDOWS_SENSOR_BRIDGE_TASK_NAME}'\""
            )
        }
        RemoteOs::Unix | RemoteOs::Macos => format!(
            "nohup {quoted_path} --agent --agent-host 0.0.0.0 --agent-port {REMOTE_AGENT_DEFAULT_PORT} > ~/.config/local-llm-foundry/{log} 2>&1 &",
            log = crate::identity::AGENT_LOG_RELATIVE_PATH
        ),
        RemoteOs::Unknown => format!(
            "{quoted_path} --agent --agent-host 0.0.0.0 --agent-port {REMOTE_AGENT_DEFAULT_PORT}"
        ),
    }
}

pub(crate) async fn default_start_command_for_os_with(
    connection: &SshConnection,
    os: RemoteOs,
    install_path: &str,
) -> String {
    let resolved_path = if os == RemoteOs::Windows {
        if let Some(appdata) = resolve_windows_appdata(connection).await {
            install_path.replace("%APPDATA%", &appdata)
        } else {
            install_path.to_string()
        }
    } else {
        install_path.to_string()
    };
    default_start_command_for_os(os, &resolved_path)
}

const WINDOWS_AGENT_TASK_NAME: &str = crate::identity::CANONICAL_AGENT_TASK_NAME;
const WINDOWS_AGENT_LEGACY_TASK_NAME: &str = crate::identity::LEGACY_AGENT_TASK_NAME;
const WINDOWS_SENSOR_BRIDGE_TASK_NAME: &str = crate::identity::CANONICAL_SENSOR_TASK_NAME;
const WINDOWS_SENSOR_BRIDGE_LEGACY_TASK_NAME: &str = crate::identity::LEGACY_SENSOR_TASK_NAME;

/// Batch script placed next to the Windows agent binary after install.
/// Double-clicking it (or running from cmd) requests UAC elevation via VBScript
/// and removes both the scheduled task and legacy task name.
const WINDOWS_AGENT_UNINSTALL_BAT: &[u8] = br#"@echo off
net session >nul 2>&1
if %errorlevel% == 0 goto :elevated
echo Set UAC = CreateObject^("Shell.Application"^) > "%temp%\lm_uac.vbs"
echo UAC.ShellExecute "%~f0", "", "", "runas", 1 >> "%temp%\lm_uac.vbs"
"%temp%\lm_uac.vbs"
del "%temp%\lm_uac.vbs"
goto :eof
:elevated
schtasks /End /TN "LocalLLMFoundryAgent" >nul 2>&1
schtasks /Delete /TN "LocalLLMFoundryAgent" /F >nul 2>&1
schtasks /End /TN "llama-monitor-agent" >nul 2>&1
schtasks /Delete /TN "llama-monitor-agent" /F >nul 2>&1
echo Local LLM Foundry agent service removed.
echo You may delete this folder.
pause
"#;

/// Shell script placed next to the Unix/macOS agent binary after install.
const UNIX_AGENT_UNINSTALL_SH: &[u8] = b"#!/bin/bash\n\
pkill -x local-llm-foundry 2>/dev/null || true\n\
pkill -x llama-monitor 2>/dev/null || true\n\
SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n\
echo \"Local LLM Foundry agent stopped.\"\n\
echo \"To fully remove, delete: $SCRIPT_DIR\"\n";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteOs {
    Windows,
    Unix,
    Macos,
    Unknown,
}

async fn detect_remote_os(target: &str) -> RemoteOs {
    detect_remote_os_with(&SshConnection::from_target(target)).await
}

pub(crate) async fn detect_remote_os_with(connection: &SshConnection) -> RemoteOs {
    let windows = remote_ssh::exec(connection.clone(), "cmd.exe /C ver".to_string());

    if let Ok(Ok(output)) = tokio::time::timeout(Duration::from_secs(5), windows).await
        && output.status == 0
        && output.stdout.contains("Windows")
    {
        return RemoteOs::Windows;
    }

    let unix = remote_ssh::exec(connection.clone(), "uname -s".to_string());
    if let Ok(Ok(output)) = tokio::time::timeout(Duration::from_secs(5), unix).await
        && output.status == 0
    {
        let name = output.stdout.to_ascii_lowercase();
        if name.contains("darwin") {
            return RemoteOs::Macos;
        }
        return RemoteOs::Unix;
    }

    RemoteOs::Unknown
}

/// Resolves `%APPDATA%` to its actual expanded path on the remote Windows host.
///
/// `schtasks /TR` stores the task command literally. When the task runs in the
/// Task Scheduler service context (which differs from the SSH session), env-var
/// expansion may resolve `%APPDATA%` to a system profile path instead of the
/// user's, causing "The system cannot find the path specified" on `schtasks /Run`.
/// Expanding the path at install/start time avoids this entirely.
pub(crate) async fn resolve_windows_appdata(connection: &SshConnection) -> Option<String> {
    // Use PowerShell for reliable path resolution (preserves backslashes)
    if let Ok(Ok(out)) = tokio::time::timeout(
        Duration::from_secs(5),
        remote_ssh::exec(
            connection.clone(),
            "powershell.exe -NoProfile -NonInteractive -Command \"$env:APPDATA\"".to_string(),
        ),
    )
    .await
        && out.status == 0
    {
        let s = out.stdout.trim().to_string();
        // Verify: must contain backslashes (valid Windows path)
        if !s.is_empty() && s.contains('\\') && !s.starts_with('%') {
            return Some(s);
        }
    }

    // Fallback: cmd.exe echo
    if let Ok(Ok(out)) = tokio::time::timeout(
        Duration::from_secs(5),
        remote_ssh::exec(connection.clone(), "cmd.exe /C echo %APPDATA%".to_string()),
    )
    .await
        && out.status == 0
    {
        let s = out.stdout.trim().to_string();
        if !s.is_empty() && s.contains('\\') && !s.starts_with('%') {
            return Some(s);
        }
    }

    None
}

async fn detect_remote_temp_dir(connection: &SshConnection, os: RemoteOs) -> String {
    let temp_cmd = match os {
        RemoteOs::Windows => "cmd.exe /C echo %TEMP%".to_string(),
        RemoteOs::Unix | RemoteOs::Macos => "echo /tmp".to_string(),
        RemoteOs::Unknown => return "/tmp".to_string(),
    };

    if let Ok(Ok(output)) = tokio::time::timeout(
        Duration::from_secs(5),
        remote_ssh::exec(connection.clone(), temp_cmd),
    )
    .await
    {
        let s = output.stdout.trim().to_string();
        if output.status == 0 && !s.is_empty() && !s.starts_with('%') {
            return s;
        }
    }

    // cmd.exe failed or returned unexpanded %TEMP% — try PowerShell (same
    // approach used to resolve %APPDATA%).
    if os == RemoteOs::Windows {
        if let Ok(Ok(out)) = tokio::time::timeout(
            Duration::from_secs(5),
            remote_ssh::exec(
                connection.clone(),
                "powershell.exe -NoProfile -NonInteractive -Command \"$env:TEMP\"".to_string(),
            ),
        )
        .await
        {
            let s = out.stdout.trim().to_string();
            if out.status == 0 && !s.is_empty() && s.contains('\\') && !s.starts_with('%') {
                return s;
            }
        }
        "C:\\Windows\\Temp".to_string()
    } else {
        "/tmp".to_string()
    }
}

pub async fn detect_remote_os_for_connection(connection: SshConnection) -> RemoteOs {
    detect_remote_os_with(&connection).await
}

#[derive(Debug)]
struct AgentAuthError;

impl warp::reject::Reject for AgentAuthError {}

impl LatestReleaseInfo {
    pub fn matching_asset(&self, os: &str, arch: &str) -> Option<&ReleaseAssetInfo> {
        self.assets
            .iter()
            .filter(|asset| asset.platform == os && asset.arch == normalize_arch(arch))
            .min_by_key(|asset| {
                if asset
                    .name
                    .starts_with(crate::identity::CANONICAL_RELEASE_ASSET_PREFIX)
                {
                    0
                } else {
                    1
                }
            })
    }
}

fn asset_info_from_github_asset(asset: GithubAsset) -> Option<ReleaseAssetInfo> {
    let (platform, arch, archive) = match asset.name.as_str() {
        "local-llm-foundry-windows-x86_64.zip" | "llama-monitor-windows-x86_64.zip" => {
            ("windows", "x86_64", true)
        }
        "local-llm-foundry-linux-x86_64" | "llama-monitor-linux-x86_64" => {
            ("linux", "x86_64", false)
        }
        "local-llm-foundry-linux-aarch64" | "llama-monitor-linux-aarch64" => {
            ("linux", "aarch64", false)
        }
        "local-llm-foundry-macos-aarch64.tar.gz" | "llama-monitor-macos-aarch64.tar.gz" => {
            ("macos", "aarch64", true)
        }
        _ => return None,
    };

    Some(ReleaseAssetInfo {
        name: asset.name,
        url: asset.browser_download_url,
        size: asset.size,
        platform: platform.to_string(),
        arch: arch.to_string(),
        archive,
    })
}

impl RemoteOs {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            RemoteOs::Windows => "windows",
            RemoteOs::Unix => "linux",
            RemoteOs::Macos => "macos",
            RemoteOs::Unknown => "unknown",
        }
    }
}

fn normalize_arch(arch: &str) -> String {
    match arch.trim().to_ascii_lowercase().as_str() {
        "amd64" | "x64" | "x86_64" => "x86_64".to_string(),
        "arm64" | "aarch64" => "aarch64".to_string(),
        other => other.to_string(),
    }
}

fn normalize_version_label(version: &str) -> String {
    version
        .split_whitespace()
        .last()
        .unwrap_or(version.trim())
        .trim_start_matches('v')
        .to_string()
}

fn build_plain_http_client(timeout: Duration) -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(timeout)
        .pool_max_idle_per_host(0)
        .build()
        .unwrap_or_else(|_| reqwest::Client::new())
}

fn build_agent_https_client(timeout: Duration) -> Option<reqwest::Client> {
    // Collect all trust anchors:
    //   1. Device's own CA (ca.pem) — backward compat and legacy installs where device CA == agent CA
    //   2. Remote-agent CAs (remote-cas/*.pem) — for fresh devices with their own unique CA
    // With reqwest 0.13 + rustls, tls_certs_only replaces the deprecated add_root_certificate API.
    // danger_accept_invalid_hostnames is set because agents may be reached by IP with no SAN match.
    let mut builder = reqwest::Client::builder()
        .timeout(timeout)
        .pool_max_idle_per_host(0);

    let mut trust_anchors: Vec<reqwest::Certificate> = Vec::new();

    if let Some(ca) = crate::certs::Cert::load(
        &crate::certs::certs_dir().join("ca.pem"),
        &crate::certs::certs_dir().join("ca.key"),
    ) && let Ok(cert) = reqwest::Certificate::from_pem(ca.pem.as_bytes())
    {
        trust_anchors.push(cert);
    }
    trust_anchors.extend(crate::certs::load_remote_agent_ca_certs());

    let ca_loaded = if !trust_anchors.is_empty() {
        builder = builder
            .tls_certs_only(trust_anchors)
            .danger_accept_invalid_hostnames(true);
        true
    } else {
        eprintln!("[agent] No CA found; managed agent HTTPS connections will fail");
        false
    };

    let client_cert = crate::certs::ensure_agent_client_cert();
    let combined_pem = format!("{}{}", client_cert.pem, client_cert.key);
    match reqwest::Identity::from_pem(combined_pem.as_bytes()) {
        Ok(id) => {
            builder = builder.identity(id);
        }
        Err(e) => {
            eprintln!(
                "[agent] Failed to load agent-client identity: {e}; continuing without mTLS client cert"
            );
            if ca_loaded {
                eprintln!(
                    "[agent] Agent mTLS may reject dashboard connections until agent-client identity is repaired"
                );
            }
        }
    }

    match builder.build() {
        Ok(client) => Some(client),
        Err(e) => {
            eprintln!("[agent] Failed to build HTTPS agent client: {e}");
            None
        }
    }
}

/// Derives the enrollment port URL from an agent URL.
/// The enrollment server listens on agent port + 1 (e.g., 7779 → 7780).
fn enrollment_url_from_agent_url(agent_url: &str) -> Option<String> {
    let url = reqwest::Url::parse(agent_url.trim_end_matches('/')).ok()?;
    let port = url.port().unwrap_or(7779);
    let enroll_port = port.saturating_add(ENROLLMENT_PORT_OFFSET);
    let mut enroll_url = url.clone();
    let _ = enroll_url.set_port(Some(enroll_port));
    let _ = enroll_url.set_scheme("https");
    Some(enroll_url.to_string().trim_end_matches('/').to_string())
}

/// Builds an HTTPS client without a client certificate for use on the enrollment port.
/// The enrollment port uses server-only TLS; clients do not present a client cert.
/// We still validate the server cert using our CA so the bearer token is protected in transit.
fn build_enrollment_https_client(timeout: Duration) -> Option<reqwest::Client> {
    let mut builder = reqwest::Client::builder()
        .timeout(timeout)
        .pool_max_idle_per_host(0);

    // Collect trust anchors — same logic as build_agent_https_client.
    // remote-cas/ is checked first so a fresh device whose own ca.pem differs from the
    // agent's CA can still validate the enrollment endpoint's server cert.
    let mut trust_anchors: Vec<reqwest::Certificate> = Vec::new();

    if let Some(ca) = crate::certs::Cert::load(
        &crate::certs::certs_dir().join("ca.pem"),
        &crate::certs::certs_dir().join("ca.key"),
    ) && let Ok(cert) = reqwest::Certificate::from_pem(ca.pem.as_bytes())
    {
        trust_anchors.push(cert);
    }
    trust_anchors.extend(crate::certs::load_remote_agent_ca_certs());

    if !trust_anchors.is_empty() {
        builder = builder
            .tls_certs_only(trust_anchors)
            .danger_accept_invalid_hostnames(true);
    } else {
        // No CA available yet — this only happens on the very first bootstrap before
        // the remote-cas/ entry is saved. The api-token still provides authentication.
        builder = builder.danger_accept_invalid_certs(true);
    }

    // No client cert — intentionally not mTLS on the enrollment port.
    builder.build().ok()
}

/// Reads the remote agent's `ca.pem` via SSH.
///
/// Used during bootstrap enrollment so the client can validate the agent's TLS cert
/// on all subsequent connections. The result is saved to `remote-cas/<host>.pem`.
async fn read_remote_ca_pem(connection: &SshConnection, os: RemoteOs) -> Option<String> {
    let command = match os {
        RemoteOs::Windows => {
            // Agent config lives in the SYSTEM profile on Windows.
            r#"cmd.exe /C "type "C:\Windows\System32\config\systemprofile\AppData\Roaming\llama-monitor\certs\ca.pem" 2>NUL""#
                .to_string()
        }
        RemoteOs::Unix | RemoteOs::Macos => "cat ~/.config/llama-monitor/certs/ca.pem".to_string(),
        RemoteOs::Unknown => return None,
    };
    match tokio::time::timeout(
        Duration::from_secs(5),
        remote_ssh::exec(connection.clone(), command),
    )
    .await
    {
        Ok(Ok(output)) if output.status == 0 && !output.stdout.trim().is_empty() => {
            Some(output.stdout)
        }
        _ => None,
    }
}

/// Fully automated bootstrap enrollment for a new client device.
///
/// Called when an mTLS connection to a remote agent fails (CA not trusted yet) and
/// SSH configuration is available. Performs over SSH, with no user interaction:
///   1. Reads the agent's CA cert → saved to `remote-cas/<host>.pem` for TLS validation
///   2. Reads the agent's api-token → used to authorize the enrollment request
///   3. Ensures local device CA + client cert exist (generates if missing)
///   4. POSTs the device's CA to the enrollment endpoint (auth: api-token)
///   5. Agent hot-reloads mTLS trust — device can now connect via mTLS
///   6. Saves api-token to settings so ongoing API calls also work
///
/// Returns `true` on success (mTLS should work on the next poll iteration).
async fn bootstrap_client_enrollment(
    connection: &SshConnection,
    agent_url: &str,
    state: &AppState,
    _app_config: &Arc<AppConfig>,
) -> bool {
    eprintln!("[agent] Starting SSH bootstrap enrollment for {agent_url}");

    let os = detect_remote_os_with(connection).await;

    // 1. Fetch the agent's CA cert and store it as a remote-agent trust anchor.
    match read_remote_ca_pem(connection, os).await {
        Some(pem) => {
            crate::certs::save_remote_agent_ca(&connection.host, &pem);
            eprintln!(
                "[agent] Fetched agent CA cert via SSH; saved to remote-cas/{}.pem",
                connection.host
            );
        }
        None => {
            eprintln!(
                "[agent] Could not read agent CA cert via SSH from {}; TLS validation may fail",
                connection.host
            );
        }
    }

    // 2. Fetch the api-token from the remote agent via SSH.
    let Some(api_token) = read_remote_agent_token(connection, os, Some(agent_url)).await else {
        eprintln!("[agent] Could not read api-token from remote agent via SSH; cannot enroll");
        return false;
    };

    // 3. Ensure this device has a local CA + client cert (generated fresh if missing).
    crate::certs::ensure_ca();
    crate::certs::ensure_agent_client_cert();

    // 4. POST this device's CA to the enrollment endpoint, authenticated with the api-token.
    if !try_enroll_ca(agent_url, Some(&api_token)).await {
        eprintln!("[agent] CA enrollment request failed during bootstrap");
        return false;
    }

    // 5. Save the api-token to settings so ongoing API calls authenticate correctly.
    {
        let mut s = state.ui_settings.lock().unwrap();
        s.remote_agent_token = api_token.clone();
        let _ = crate::state::save_ui_settings(&state.ui_settings_path, &s);
    }

    eprintln!(
        "[agent] Bootstrap enrollment complete for {agent_url}; mTLS trust established. \
         Retrying connection on next poll."
    );
    true
}

/// Attempts to register our local CA with a remote agent's enrollment endpoint.
///
/// `api_token` is the agent's api-token, obtained automatically via SSH during
/// bootstrap enrollment. It is used as the Bearer token to authenticate the request.
///
/// Returns `true` if the CA is now trusted (either newly registered or already was).
/// Errors are logged but not propagated — enrollment failure is non-fatal for the
/// poller; the agent just stays disconnected until enrollment succeeds.
async fn try_enroll_ca(agent_url: &str, api_token: Option<&str>) -> bool {
    let Some(enroll_url) = enrollment_url_from_agent_url(agent_url) else {
        eprintln!("[agent] Cannot derive enrollment URL from {agent_url}");
        return false;
    };

    let Some(client) = build_enrollment_https_client(Duration::from_secs(5)) else {
        eprintln!("[agent] Could not build enrollment client");
        return false;
    };

    // Load our own CA PEM to register with the remote agent.
    let Some(ca) = crate::certs::Cert::load(
        &crate::certs::certs_dir().join("ca.pem"),
        &crate::certs::certs_dir().join("ca.key"),
    ) else {
        eprintln!("[agent] No local CA found; cannot enroll with agent");
        return false;
    };

    let enroll_endpoint = format!("{enroll_url}/api/agent/register-ca");
    eprintln!("[agent] Attempting CA enrollment at {enroll_endpoint}");

    let mut req = client.post(&enroll_endpoint).json(&serde_json::json!({
        "ca_pem": ca.pem,
        "confirm": "register-ca"
    }));

    if let Some(tok) = api_token {
        req = req.bearer_auth(tok);
    }

    match req.send().await {
        Ok(resp) => {
            let status = resp.status();
            match resp.json::<serde_json::Value>().await {
                Ok(json) if json.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) => {
                    let already = json
                        .get("already_trusted")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    if already {
                        eprintln!(
                            "[agent] CA already trusted by remote agent; mTLS should succeed"
                        );
                    } else {
                        let id = json
                            .get("instance_id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown");
                        eprintln!(
                            "[agent] CA registered with remote agent (instance_id={id}); retrying mTLS connection"
                        );
                    }
                    true
                }
                Ok(json) => {
                    let err = json
                        .get("error")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown error");
                    eprintln!("[agent] CA enrollment rejected (HTTP {status}): {err}");
                    false
                }
                Err(e) => {
                    eprintln!("[agent] CA enrollment response parse error: {e}");
                    false
                }
            }
        }
        Err(e) => {
            eprintln!(
                "[agent] CA enrollment request failed (agent may not support enrollment or port {ENROLLMENT_PORT_OFFSET} offset): {e}"
            );
            false
        }
    }
}

fn load_install_dir_agent_server_cert() -> Option<crate::certs::Cert> {
    let current_exe = std::env::current_exe().ok()?;
    let install_dir = current_exe.parent()?;
    let cert_path = install_dir.join("agent-server.pem");
    let key_path = install_dir.join("agent-server.key");
    crate::certs::Cert::load(&cert_path, &key_path)
}

fn agent_url_candidates(agent_url: &str) -> Vec<String> {
    let trimmed = agent_url.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return Vec::new();
    }

    let mut candidates = Vec::new();
    if let Ok(parsed) = reqwest::Url::parse(trimmed) {
        match parsed.scheme() {
            "http" => {
                let mut https = parsed.clone();
                let _ = https.set_scheme("https");
                candidates.push(https.to_string().trim_end_matches('/').to_string());
                candidates.push(trimmed.to_string());
            }
            "https" => {
                candidates.push(trimmed.to_string());
            }
            _ => candidates.push(trimmed.to_string()),
        }
    } else {
        candidates.push(trimmed.to_string());
    }

    candidates.dedup();
    candidates
}

fn agent_client_for_url<'a>(
    agent_url: &str,
    https_client: Option<&'a reqwest::Client>,
    http_client: &'a reqwest::Client,
) -> &'a reqwest::Client {
    if agent_url.starts_with("https://") {
        https_client.unwrap_or(http_client)
    } else {
        http_client
    }
}

fn install_path_for_os(os: RemoteOs) -> Option<&'static str> {
    match os {
        RemoteOs::Windows => Some(crate::identity::install_path(true)),
        RemoteOs::Unix | RemoteOs::Macos => Some(crate::identity::install_path(false)),
        RemoteOs::Unknown => None,
    }
}

fn legacy_install_path_for_os(os: RemoteOs) -> Option<&'static str> {
    match os {
        RemoteOs::Windows => Some(crate::identity::legacy_install_path(true)),
        RemoteOs::Unix | RemoteOs::Macos => Some(crate::identity::legacy_install_path(false)),
        RemoteOs::Unknown => None,
    }
}

pub(crate) fn default_install_path_for_os(os: RemoteOs) -> String {
    install_path_for_os(os)
        .unwrap_or("/tmp/local-llm-foundry")
        .to_string()
}

async fn detect_remote_arch_with(connection: &SshConnection, os: RemoteOs) -> String {
    let command = match os {
        RemoteOs::Windows => "cmd.exe /C echo %PROCESSOR_ARCHITECTURE%",
        RemoteOs::Unix | RemoteOs::Macos => "uname -m",
        RemoteOs::Unknown => return "unknown".to_string(),
    };

    match tokio::time::timeout(
        Duration::from_secs(5),
        remote_ssh::exec(connection.clone(), command.to_string()),
    )
    .await
    {
        Ok(Ok(output)) if output.status == 0 => normalize_arch(output.stdout.trim()),
        _ => "unknown".to_string(),
    }
}

async fn remote_file_exists_with(connection: &SshConnection, os: RemoteOs, path: &str) -> bool {
    let command = match os {
        RemoteOs::Windows => format!("cmd.exe /C if exist \"{path}\" (echo yes)"),
        RemoteOs::Unix | RemoteOs::Macos => format!("test -x {path} && echo yes"),
        RemoteOs::Unknown => return false,
    };

    matches!(
        tokio::time::timeout(
            Duration::from_secs(5),
            remote_ssh::exec(connection.clone(), command)
        )
        .await,
        Ok(Ok(output))
            if output.status == 0 && output.stdout.contains("yes")
    )
}

pub(crate) async fn resolve_remote_install_path(
    connection: Option<&SshConnection>,
    os: RemoteOs,
    requested: Option<&str>,
) -> String {
    let requested = requested.map(str::trim).filter(|path| !path.is_empty());
    let requested_is_managed = requested.is_some_and(|path| {
        managed_install_path_candidates(os)
            .iter()
            .any(|managed| managed_install_path_matches(path, managed))
    });

    if let Some(path) = requested
        && !requested_is_managed
    {
        return path.to_string();
    }

    if let Some(connection) = connection {
        return preferred_remote_install_path(connection, os).await;
    }

    requested
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| default_install_path_for_os(os))
}

fn managed_install_path_matches(requested: &str, managed: &str) -> bool {
    fn normalize(path: &str) -> String {
        path.trim_end_matches(['\\', '/'])
            .replace('/', "\\")
            .to_ascii_lowercase()
    }

    normalize(requested) == normalize(managed)
}

fn windows_path_matches_install_path(
    command: &str,
    install_path: &str,
    appdata: Option<&str>,
) -> bool {
    fn normalize(path: &str, appdata: Option<&str>) -> String {
        let mut normalized = path.replace('/', "\\").to_ascii_lowercase();

        if let Some(appdata) = appdata {
            let appdata = appdata
                .trim_end_matches(['\\', '/'])
                .replace('/', "\\")
                .to_ascii_lowercase();
            normalized = normalized.replace("%appdata%", &appdata);
        }

        normalized
            .trim_matches(['"', '\''])
            .trim_end_matches(['\\', '/'])
            .to_string()
    }

    let command = normalize(command, appdata);
    let install_path = normalize(install_path, appdata);
    !install_path.is_empty() && command.contains(&install_path)
}

pub(crate) async fn preferred_remote_install_path(
    connection: &SshConnection,
    os: RemoteOs,
) -> String {
    let candidates = managed_install_path_candidates(os);
    for path in &candidates {
        if remote_file_exists_with(connection, os, path).await {
            return path.clone();
        }
    }
    default_install_path_for_os(os)
}

fn managed_install_path_candidates(os: RemoteOs) -> Vec<String> {
    let Some(canonical) = install_path_for_os(os) else {
        return Vec::new();
    };
    let Some(legacy) = legacy_install_path_for_os(os) else {
        return vec![canonical.to_string()];
    };
    let canonical_name = if os == RemoteOs::Windows {
        crate::identity::binary_name(true)
    } else {
        crate::identity::binary_name(false)
    };
    let legacy_name = if os == RemoteOs::Windows {
        crate::identity::legacy_binary_name(true)
    } else {
        crate::identity::legacy_binary_name(false)
    };
    let mut candidates = vec![
        canonical.to_string(),
        path_with_file_name(canonical, legacy_name),
        path_with_file_name(legacy, canonical_name),
        legacy.to_string(),
    ];
    candidates.dedup();
    candidates
}

fn path_with_file_name(path: &str, file_name: &str) -> String {
    path.rfind(['\\', '/'])
        .map(|index| format!("{}{}", &path[..=index], file_name))
        .unwrap_or_else(|| file_name.to_string())
}

async fn agent_health_reachable(agent_url: &str) -> bool {
    agent_health_reachable_with_token(agent_url, None).await
}

async fn agent_health_reachable_with_token(agent_url: &str, token: Option<&str>) -> bool {
    let https_client = build_agent_https_client(Duration::from_secs(2));
    let http_client = build_plain_http_client(Duration::from_secs(2));

    for candidate in agent_url_candidates(agent_url) {
        let client = agent_client_for_url(&candidate, https_client.as_ref(), &http_client);
        let mut req = client.get(format!("{}/health", candidate));
        if let Some(t) = token {
            req = req.bearer_auth(t);
        }
        if let Ok(resp) = req.send().await
            && resp.status().is_success()
        {
            return true;
        }
    }

    false
}

async fn agent_metrics_reachable_with_token(agent_url: &str, token: Option<&str>) -> bool {
    let Some(token) = token else {
        return false;
    };
    let https_client = build_agent_https_client(Duration::from_secs(2));
    let http_client = build_plain_http_client(Duration::from_secs(2));

    for candidate in agent_url_candidates(agent_url) {
        let client = agent_client_for_url(&candidate, https_client.as_ref(), &http_client);
        if let Ok(resp) = client
            .get(format!("{candidate}/metrics"))
            .bearer_auth(token)
            .send()
            .await
            && resp.status().is_success()
        {
            return true;
        }
    }
    false
}

/// Write the agent token to a temp file in each user's home directory, so the
/// main app can read it via SSH even when the agent runs as SYSTEM (Windows) or
/// another user. The files are cleaned up after 30 seconds.
fn write_token_to_temp_file(token: &str) -> Vec<std::path::PathBuf> {
    let mut paths = Vec::new();
    use rand::TryRng;
    use rand::rngs::SysRng;
    let mut nonce = [0u8; 16];
    SysRng
        .try_fill_bytes(&mut nonce)
        .expect("system entropy unavailable; refusing temporary token transport");
    let nonce = nonce.iter().map(|b| format!("{b:02x}")).collect::<String>();
    let file_name = format!(
        "{}{nonce}.tmp",
        crate::identity::CANONICAL_AGENT_TOKEN_PREFIX
    );

    // Determine home directories to write to
    let home_dirs: Vec<std::path::PathBuf> = if cfg!(windows) {
        // On Windows, write to every user's home directory under C:\Users\
        // so the SSH user can read it. The agent runs as SYSTEM, so it can
        // write to any user's directory.
        match std::fs::read_dir("C:\\Users") {
            Ok(entries) => entries
                .filter_map(|entry| entry.ok())
                .filter(|entry| {
                    // Skip system accounts
                    let name = entry.file_name().to_string_lossy().to_string();
                    !name.starts_with("$") && name != "Default" && name != "Public"
                })
                .map(|entry| entry.path().join(".local-llm-foundry"))
                .collect(),
            Err(_) => Vec::new(),
        }
    } else {
        // On Unix, write to /tmp (accessible by all users)
        vec![std::env::temp_dir()]
    };

    for mut home_dir in home_dirs {
        let _ = std::fs::create_dir_all(&home_dir);
        home_dir.push(&file_name);

        let write_result = (|| {
            #[cfg(unix)]
            use std::os::unix::fs::OpenOptionsExt;
            let mut options = std::fs::OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            options.mode(0o600);
            let mut file = options.open(&home_dir)?;
            use std::io::Write;
            file.write_all(token.as_bytes())?;
            file.sync_all()
        })();

        if write_result.is_ok() {
            let cleanup_path = home_dir.clone();
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_secs(30));
                let _ = std::fs::remove_file(&cleanup_path);
                // Clean up the canonical compatibility directory if empty.
                if let Some(parent) = cleanup_path.parent() {
                    let _ = std::fs::remove_dir(parent);
                }
            });
            eprintln!("[agent] Token written to temp file: {}", home_dir.display());
            paths.push(home_dir);
        } else {
            eprintln!(
                "[agent] Failed to write token to temp file: {}",
                home_dir.display()
            );
        }
    }

    paths
}

async fn read_remote_agent_token(
    connection: &SshConnection,
    os: RemoteOs,
    _agent_url: Option<&str>,
) -> Option<String> {
    // Try the user-readable temp file first (written by the agent on startup,
    // cleaned up after 30 seconds). On Windows, the agent writes to each user's
    // home directory so the SSH user can read it. On Unix, it writes to /tmp.
    let temp_file_cmd = match os {
        RemoteOs::Windows => {
            // Read canonical bootstrap files first, then retain the legacy glob
            // for agents upgraded in place during the 2.x compatibility window.
            "powershell.exe -NoProfile -NonInteractive -Command \"$files=@(Get-ChildItem -Path $env:USERPROFILE+'\\.local-llm-foundry\\local-llm-foundry-agent-token-*.tmp' -ErrorAction SilentlyContinue; Get-ChildItem -Path $env:USERPROFILE+'\\.llama-monitor\\llama-monitor-agent-token-*.tmp' -ErrorAction SilentlyContinue) | Sort-Object LastWriteTime -Descending; if ($files) { Get-Content -Raw -LiteralPath $files[0].FullName }\""
                .to_string()
        }
        RemoteOs::Unix | RemoteOs::Macos => {
            // Canonical files are preferred; legacy files remain readable for
            // old agents during the 2.x transition.
            "(ls -t /tmp/local-llm-foundry-agent-token-*.tmp /tmp/llama-monitor-agent-token-*.tmp 2>/dev/null | head -1 | xargs cat 2>/dev/null)"
                .to_string()
        }
        RemoteOs::Unknown => return None,
    };
    if let Ok(Ok(output)) = tokio::time::timeout(
        Duration::from_secs(5),
        remote_ssh::exec(connection.clone(), temp_file_cmd),
    )
    .await
        && output.status == 0
    {
        let token = output.stdout.trim().to_string();
        if !token.is_empty() && token.len() >= 16 {
            return Some(token);
        }
    }

    // Fall back to reading the token from the config directory (works on Unix,
    // fails on Windows SYSTEM profile).
    let command = match os {
        RemoteOs::Windows => {
            // The scheduled task runs as SYSTEM but the bootstrap command
            // deliberately points it at the SSH user's %APPDATA% directory.
            // Probe both user and SYSTEM profiles during the migration window.
            r#"cmd.exe /C "(type "%APPDATA%\local-llm-foundry\agent-token" 2>NUL || type "%APPDATA%\llama-monitor\agent-token" 2>NUL || type "C:\Windows\System32\config\systemprofile\AppData\Roaming\local-llm-foundry\agent-token" 2>NUL || type "C:\Windows\System32\config\systemprofile\AppData\Roaming\llama-monitor\agent-token" 2>NUL)""#
                .to_string()
        }
        RemoteOs::Unix | RemoteOs::Macos => "(cat ~/.config/local-llm-foundry/agent-token 2>/dev/null || cat ~/.config/llama-monitor/agent-token 2>/dev/null)".to_string(),
        RemoteOs::Unknown => return None,
    };
    match tokio::time::timeout(
        Duration::from_secs(5),
        remote_ssh::exec(connection.clone(), command),
    )
    .await
    {
        Ok(Ok(output)) if output.status == 0 => {
            let token = output.stdout.trim().to_string();
            if token.is_empty() { None } else { Some(token) }
        }
        _ => None,
    }
}

async fn handle_agent_rejection(
    rejection: warp::Rejection,
) -> Result<impl warp::Reply, std::convert::Infallible> {
    if rejection.find::<AgentAuthError>().is_some() {
        let reply = warp::reply::with_status(
            warp::reply::json(&serde_json::json!({ "error": "unauthorized" })),
            StatusCode::UNAUTHORIZED,
        );
        return Ok(reply);
    }

    let reply = warp::reply::with_status(
        warp::reply::json(&serde_json::json!({ "error": "not found" })),
        StatusCode::NOT_FOUND,
    );
    Ok(reply)
}

pub mod install {
    use super::*;
    use std::fs;
    use std::io;

    const REMOTE_AGENT_INSTALL_TIMEOUT: Duration = Duration::from_secs(60);

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RemoteAgentInstallRequest {
        pub ssh_target: String,
        #[serde(default)]
        pub ssh_connection: Option<SshConnection>,
        pub asset: ReleaseAssetInfo,
        pub install_path: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RemoteAgentInstallResponse {
        pub ok: bool,
        pub ssh_target: String,
        pub asset_name: String,
        pub install_path: String,
        pub installed: bool,
        pub error: Option<String>,
    }

    /// Write an uninstall script next to the agent binary on the remote machine.
    /// Failure is silently ignored — the agent is already installed; the script
    /// is a convenience for users who want to remove it without the dashboard.
    async fn drop_uninstall_script(connection: &SshConnection, install_path: &str, os: RemoteOs) {
        let sep = if os == RemoteOs::Windows { '\\' } else { '/' };
        let Some(dir_end) = install_path.rfind(sep) else {
            return;
        };
        let install_dir = &install_path[..dir_end];

        match os {
            RemoteOs::Windows => {
                // Resolve %APPDATA% to an actual path so SCP can use it.
                let resolved_dir = if let Some(appdata) = resolve_windows_appdata(connection).await
                {
                    install_dir.replace("%APPDATA%", &appdata)
                } else {
                    install_dir.to_string()
                };
                let remote_script = format!("{resolved_dir}\\uninstall.bat");

                let local_tmp = tempfile::NamedTempFile::new_in(std::env::temp_dir())
                    .map(|f| f.path().to_path_buf())
                    .unwrap_or_else(|_| {
                        std::env::temp_dir().join("llama_monitor_agent_uninstall.bat")
                    });
                if std::fs::write(&local_tmp, WINDOWS_AGENT_UNINSTALL_BAT).is_err() {
                    return;
                }
                let _ = remote_ssh::copy_to_remote(
                    connection.clone(),
                    local_tmp.to_string_lossy().to_string(),
                    remote_script,
                    0o644,
                )
                .await;
                let _ = std::fs::remove_file(&local_tmp);
            }
            RemoteOs::Unix | RemoteOs::Macos => {
                // SCP to /tmp first, then move to the install dir (handles ~ in paths).
                let tmp_remote = "/tmp/llama_monitor_agent_uninstall.sh";
                let final_remote = format!("{install_dir}/uninstall.sh");

                let local_tmp = tempfile::NamedTempFile::new_in(std::env::temp_dir())
                    .map(|f| f.path().to_path_buf())
                    .unwrap_or_else(|_| {
                        std::env::temp_dir().join("llama_monitor_agent_uninstall.sh")
                    });
                if std::fs::write(&local_tmp, UNIX_AGENT_UNINSTALL_SH).is_err() {
                    return;
                }
                if remote_ssh::copy_to_remote(
                    connection.clone(),
                    local_tmp.to_string_lossy().to_string(),
                    tmp_remote.to_string(),
                    0o755,
                )
                .await
                .is_ok()
                {
                    let _ = remote_ssh::exec(
                        connection.clone(),
                        format!("mv {tmp_remote} {final_remote}"),
                    )
                    .await;
                }
                let _ = std::fs::remove_file(&local_tmp);
            }
            RemoteOs::Unknown => {}
        }
    }

    /// Ships the CA certificate to the remote host so the agent can generate a server cert.
    async fn drop_ca_certificate(connection: &SshConnection, install_path: &str, os: RemoteOs) {
        let sep = if os == RemoteOs::Windows { '\\' } else { '/' };
        let Some(dir_end) = install_path.rfind(sep) else {
            return;
        };
        let install_dir = &install_path[..dir_end];

        // Resolve %APPDATA% to a real path before using it as an SCP destination —
        // SCP does not expand Windows environment variables.
        let resolved_install_dir = if os == RemoteOs::Windows {
            if let Some(appdata) = resolve_windows_appdata(connection).await {
                eprintln!("[agent] Resolved %APPDATA% → {appdata} for CA SCP path");
                install_dir.replace("%APPDATA%", &appdata)
            } else {
                eprintln!(
                    "[agent] Could not resolve %APPDATA%; SCP may fail if path contains env vars"
                );
                install_dir.to_string()
            }
        } else {
            install_dir.to_string()
        };

        let cas_dir = format!("{}{}cas", install_dir, sep);
        let resolved_cas_dir = format!("{}{}cas", resolved_install_dir, sep);

        // Ensure cas/ directory exists on the remote. Use the fully-resolved path
        // so the mkdir works regardless of whether %APPDATA% is set in the SSH session env.
        let mkdir_cmd = if os == RemoteOs::Windows {
            let ps_dir = resolved_cas_dir.replace('\'', "''");
            format!(
                "powershell.exe -NoProfile -NonInteractive -Command \"New-Item -ItemType Directory -Path '{ps_dir}' -Force | Out-Null\""
            )
        } else {
            format!("mkdir -p '{}'", cas_dir)
        };
        if let Ok(out) = remote_ssh::exec(connection.clone(), mkdir_cmd).await
            && out.status != 0
        {
            eprintln!(
                "[warn] Failed to create cas/ directory on remote: {}",
                out.stderr.trim()
            );
        }

        // Stable instance ID: hash of this instance's CA public key.
        let instance_id = {
            let ca = crate::certs::ensure_ca();
            use sha1::Digest;
            let mut ctx = sha1::Sha1::new();
            ctx.update(ca.pem.as_bytes());
            let hash = ctx.finalize();
            let mut id = String::with_capacity(16);
            for b in hash.as_slice().iter().take(8) {
                id.push_str(&format!("{:02x}", b));
            }
            id
        };

        let ca = crate::certs::ensure_ca();
        let remote_ca_path = format!("{}{}{}.pem", resolved_cas_dir, sep, instance_id);

        // Write CA cert to temp file and copy to remote
        let local_tmp = tempfile::NamedTempFile::new_in(std::env::temp_dir())
            .map(|f| f.path().to_path_buf())
            .unwrap_or_else(|_| std::env::temp_dir().join("ca.pem"));

        if std::fs::write(&local_tmp, &ca.pem).is_ok() {
            let result = remote_ssh::copy_to_remote(
                connection.clone(),
                local_tmp.to_string_lossy().to_string(),
                remote_ca_path.clone(),
                0o644,
            )
            .await;
            if result.is_ok() {
                eprintln!("[info] Installed CA as {} on remote agent", remote_ca_path);
            } else {
                eprintln!("[warn] Failed to install CA on remote agent: {:?}", result);
            }
        }
        let _ = std::fs::remove_file(&local_tmp);
    }

    fn managed_agent_server_sans(connection: &SshConnection) -> Vec<String> {
        let mut sans = vec!["localhost".to_string(), "127.0.0.1".to_string()];
        let host = connection.host.trim();
        if !host.is_empty()
            && !sans
                .iter()
                .any(|existing| existing.eq_ignore_ascii_case(host))
        {
            sans.push(host.to_string());
        }
        sans
    }

    async fn drop_agent_server_certificate(
        connection: &SshConnection,
        install_path: &str,
        os: RemoteOs,
    ) {
        let sep = if os == RemoteOs::Windows { '\\' } else { '/' };
        let Some(dir_end) = install_path.rfind(sep) else {
            return;
        };
        let install_dir = &install_path[..dir_end];

        // Resolve %APPDATA% before using as SCP destination.
        let resolved_install_dir = if os == RemoteOs::Windows {
            if let Some(appdata) = resolve_windows_appdata(connection).await {
                install_dir.replace("%APPDATA%", &appdata)
            } else {
                install_dir.to_string()
            }
        } else {
            install_dir.to_string()
        };

        let remote_cert_path = format!("{}{}agent-server.pem", resolved_install_dir, sep);
        let remote_key_path = format!("{}{}agent-server.key", resolved_install_dir, sep);

        let cert = crate::certs::generate_agent_server_cert(managed_agent_server_sans(connection));
        let local_cert_tmp = tempfile::NamedTempFile::new_in(std::env::temp_dir())
            .map(|f| f.path().to_path_buf())
            .unwrap_or_else(|_| std::env::temp_dir().join("agent-server.pem"));
        let local_key_tmp = tempfile::NamedTempFile::new_in(std::env::temp_dir())
            .map(|f| f.path().to_path_buf())
            .unwrap_or_else(|_| std::env::temp_dir().join("agent-server.key"));

        if std::fs::write(&local_cert_tmp, &cert.pem).is_ok() {
            match remote_ssh::copy_to_remote(
                connection.clone(),
                local_cert_tmp.to_string_lossy().to_string(),
                remote_cert_path.clone(),
                0o644,
            )
            .await
            {
                Ok(()) => {
                    eprintln!("[agent] Installed agent server cert at {remote_cert_path}")
                }
                Err(e) => eprintln!("[warn] Failed to install agent server cert: {e}"),
            }
        }
        if std::fs::write(&local_key_tmp, &cert.key).is_ok()
            && let Err(e) = remote_ssh::copy_to_remote(
                connection.clone(),
                local_key_tmp.to_string_lossy().to_string(),
                remote_key_path.clone(),
                0o600,
            )
            .await
        {
            eprintln!("[warn] Failed to install agent server key: {e}");
        }

        let _ = std::fs::remove_file(&local_cert_tmp);
        let _ = std::fs::remove_file(&local_key_tmp);
    }

    /// Writes remote-agent-config.json next to the agent binary with the api-token
    /// so the agent (or dashboard SSH operations) can authenticate to
    /// /api/remote-agent/* endpoints.
    async fn drop_remote_agent_config(
        connection: &SshConnection,
        install_path: &str,
        os: RemoteOs,
        api_token: Option<&str>,
    ) {
        let sep = if os == RemoteOs::Windows { '\\' } else { '/' };
        let Some(dir_end) = install_path.rfind(sep) else {
            return;
        };
        let install_dir = &install_path[..dir_end];

        let token = match api_token {
            Some(t) if !t.is_empty() => t.to_string(),
            _ => return,
        };

        // Resolve %APPDATA% before using as SCP destination.
        let resolved_install_dir = if os == RemoteOs::Windows {
            if let Some(appdata) = resolve_windows_appdata(connection).await {
                install_dir.replace("%APPDATA%", &appdata)
            } else {
                install_dir.to_string()
            }
        } else {
            install_dir.to_string()
        };

        let config_path = if os == RemoteOs::Windows {
            format!("{}{}remote-agent-config.json", resolved_install_dir, sep)
        } else {
            format!("{}/remote-agent-config.json", resolved_install_dir)
        };

        let config_json = serde_json::json!({
            "api_token": token,
        });
        let content = serde_json::to_string(&config_json).unwrap_or_default();

        // Write to temp file locally, then SCP to remote
        let local_tmp = tempfile::NamedTempFile::new_in(std::env::temp_dir())
            .map(|f| f.path().to_path_buf())
            .unwrap_or_else(|_| std::env::temp_dir().join("remote-agent-config.json"));

        if std::fs::write(&local_tmp, content).is_err() {
            let _ = std::fs::remove_file(&local_tmp);
            return;
        }

        // On Unix/macOS, use restrictive permissions (0600) so only the agent process can read it.
        if let Err(e) = remote_ssh::copy_to_remote(
            connection.clone(),
            local_tmp.to_string_lossy().to_string(),
            config_path.clone(),
            0o600,
        )
        .await
        {
            eprintln!("[warn] Failed to install remote-agent-config.json: {e}");
        }

        let _ = std::fs::remove_file(&local_tmp);
    }

    pub async fn install_remote_agent(
        ssh_target: &str,
        ssh_connection: Option<SshConnection>,
        asset: &ReleaseAssetInfo,
        install_path: Option<String>,
        os: RemoteOs,
        api_token: Option<String>,
    ) -> Result<RemoteAgentInstallResponse> {
        let connection = ssh_connection.unwrap_or_else(|| SshConnection::from_target(ssh_target));
        let install_path =
            resolve_remote_install_path(Some(&connection), os, install_path.as_deref()).await;

        // Validate install path before any network operations
        validate_install_path(&install_path, os).context("Invalid install path")?;

        // SCP and PowerShell do not consistently expand `%APPDATA%` when the
        // command is executed by SSH or Task Scheduler. Resolve it once and
        // carry the absolute path through extraction, certificate provisioning,
        // and scheduled-task registration.
        let install_path = if os == RemoteOs::Windows {
            resolve_windows_appdata(&connection)
                .await
                .map(|appdata| install_path.replace("%APPDATA%", &appdata))
                .unwrap_or(install_path)
        } else {
            install_path
        };
        let remote_temp_dir = detect_remote_temp_dir(&connection, os).await;
        let remote_temp_name = remote_temp_name_for_asset(asset, os);
        let remote_temp_path = match os {
            RemoteOs::Windows => format!("{}\\{}", remote_temp_dir, remote_temp_name),
            _ => format!("{}/{}", remote_temp_dir, remote_temp_name),
        };

        transfer_asset_to_remote_temp(&connection, asset, os, &remote_temp_path).await?;

        if os == RemoteOs::Windows {
            prepare_windows_install_target(&connection).await?;
        }

        if os == RemoteOs::Windows && asset.archive {
            extract_windows_archive_to_install_path(&connection, &remote_temp_path, &install_path)
                .await?;
        } else {
            move_binary_to_install_path(&connection, &remote_temp_path, &install_path, os).await?;
        }

        if os == RemoteOs::Unix || os == RemoteOs::Macos {
            set_executable_bit(&connection, &install_path, os).await?;
        }

        // Drop an uninstall script next to the binary (non-fatal if it fails).
        drop_uninstall_script(&connection, &install_path, os).await;

        // Ship the CA certificate so the agent trusts this dashboard's client certs.
        drop_ca_certificate(&connection, &install_path, os).await;

        // Provision a server certificate signed by this dashboard CA so the
        // dashboard can authenticate the managed agent over HTTPS.
        drop_agent_server_certificate(&connection, &install_path, os).await;

        // Write remote-agent-config.json with api-token so the agent can authenticate
        // to the dashboard's /api/remote-agent/* endpoints.
        if api_token.is_some() {
            drop_remote_agent_config(&connection, &install_path, os, api_token.as_deref()).await;
        }

        let installed = remote_file_exists_with(&connection, os, &install_path).await;

        Ok(RemoteAgentInstallResponse {
            ok: installed,
            ssh_target: connection.target_label(),
            asset_name: asset.name.clone(),
            install_path,
            installed,
            error: if installed {
                None
            } else {
                Some("Binary not found after install".to_string())
            },
        })
    }

    async fn download_asset_locally(asset: &ReleaseAssetInfo) -> Result<String> {
        // Bounded timeouts so a stalled connection can never hang the update
        // indefinitely. `connect_timeout` covers the handshake; the overall
        // `timeout` covers the full body transfer (release binaries are tens of MB).
        let resp = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(15))
            .timeout(std::time::Duration::from_secs(300))
            .build()?
            .get(&asset.url)
            .header(
                reqwest::header::USER_AGENT,
                crate::identity::RELEASE_USER_AGENT,
            )
            .send()
            .await?
            .error_for_status()?;

        let bytes = resp.bytes().await?;
        let temp_path = tempfile::NamedTempFile::new_in(std::env::temp_dir())
            .map(|f| f.path().to_path_buf())
            .unwrap_or_else(|_| std::env::temp_dir().join(&asset.name));
        fs::write(&temp_path, &bytes)?;
        Ok(temp_path.to_string_lossy().to_string())
    }

    fn remote_temp_name_for_asset(asset: &ReleaseAssetInfo, os: RemoteOs) -> String {
        if os == RemoteOs::Windows && asset.archive {
            asset.name.clone()
        } else if asset.name.ends_with(".tar.gz") {
            asset.name.trim_end_matches(".tar.gz").to_string()
        } else if asset.name.ends_with(".zip") {
            asset.name.trim_end_matches(".zip").to_string()
        } else {
            asset.name.clone()
        }
    }

    async fn transfer_asset_to_remote_temp(
        connection: &SshConnection,
        asset: &ReleaseAssetInfo,
        os: RemoteOs,
        remote_temp_path: &str,
    ) -> Result<()> {
        if os == RemoteOs::Windows {
            match download_asset_remotely(connection, asset, os, remote_temp_path).await {
                Ok(()) => return Ok(()),
                Err(remote_error) => {
                    let local_result =
                        download_and_copy_asset(connection, asset, os, remote_temp_path).await;
                    return local_result.map_err(|local_error| {
                        io::Error::other(format!(
                            "remote curl download failed ({remote_error}); local SCP fallback also failed ({local_error})"
                        ))
                        .into()
                    });
                }
            }
        }

        match download_and_copy_asset(connection, asset, os, remote_temp_path).await {
            Ok(()) => Ok(()),
            Err(copy_error) if !asset.archive => download_asset_remotely(
                connection,
                asset,
                os,
                remote_temp_path,
            )
            .await
            .map_err(|remote_error| {
                io::Error::other(format!(
                    "local SCP upload failed ({copy_error}); remote curl fallback also failed ({remote_error})"
                ))
                .into()
            }),
            Err(copy_error) => Err(copy_error),
        }
    }

    async fn download_and_copy_asset(
        connection: &SshConnection,
        asset: &ReleaseAssetInfo,
        os: RemoteOs,
        remote_temp_path: &str,
    ) -> Result<()> {
        let temp_local_path = download_asset_locally(asset).await?;
        let temp_extracted_path = if asset.archive {
            Some(extract_archive_with_timeout(&temp_local_path, asset).await?)
        } else {
            None
        };
        let binary_local_path = temp_extracted_path
            .as_deref()
            .unwrap_or(temp_local_path.as_str());

        copy_to_remote(connection, binary_local_path, remote_temp_path, os).await
    }

    async fn download_asset_remotely(
        connection: &SshConnection,
        asset: &ReleaseAssetInfo,
        os: RemoteOs,
        remote_temp_path: &str,
    ) -> Result<()> {
        let command = match os {
            RemoteOs::Windows => format!(
                "cmd.exe /C curl.exe -fL -o \"{}\" \"{}\"",
                remote_temp_path, asset.url
            ),
            RemoteOs::Unix | RemoteOs::Macos => {
                format!("curl -fL -o '{}' '{}'", remote_temp_path, asset.url)
            }
            RemoteOs::Unknown => return Err(io::Error::other("Unknown OS").into()),
        };

        let output = remote_ssh::exec(connection.clone(), command)
            .await
            .map_err(|e| io::Error::other(e.to_string()))?;

        if output.status == 0 {
            Ok(())
        } else {
            Err(io::Error::other(format!(
                "status {}: {}",
                output.status,
                output.stderr.trim()
            ))
            .into())
        }
    }

    async fn extract_archive_with_timeout(path: &str, asset: &ReleaseAssetInfo) -> Result<String> {
        tokio::time::timeout(REMOTE_AGENT_INSTALL_TIMEOUT, extract_archive(path, asset))
            .await?
            .map_err(|e| io::Error::other(format!("Archive extraction failed: {e}")).into())
    }

    async fn extract_archive(path: &str, asset: &ReleaseAssetInfo) -> Result<String> {
        // .zip assets (Windows) need "-xf"; .tar.gz assets (macOS) need "-xzf".
        // Windows tar.exe (libarchive, built-in since Win10 1803) auto-detects zip
        // format but the -z flag forces gzip decompression and will fail on zip.
        let (binary_name, tar_flag) = if asset.name.ends_with(".zip") {
            (asset.name.trim_end_matches(".zip"), "-xf")
        } else {
            (asset.name.trim_end_matches(".tar.gz"), "-xzf")
        };
        let temp_dir = tempfile::Builder::new()
            .prefix(&format!("{binary_name}-"))
            .tempdir_in(std::env::temp_dir())
            .map_err(|e| io::Error::other(format!("Failed to create temp dir: {e}")))?;
        let temp_extracted = temp_dir.path().to_path_buf();

        let output = tokio::process::Command::new("tar")
            .args([tar_flag, path, "-C", &temp_extracted.to_string_lossy()])
            .output()
            .await?;

        if !output.status.success() {
            Err(io::Error::other("Failed to extract archive").into())
        } else {
            let binary_path = extracted_binary_path(&temp_extracted, binary_name)?;
            // Keep directory alive beyond scope — caller will move/copy the binary
            let _ = temp_dir.keep();
            Ok(binary_path)
        }
    }

    fn extracted_binary_path(dir: &std::path::Path, binary_name: &str) -> Result<String> {
        let expected = dir.join(binary_name);
        if expected.is_file() {
            return Ok(expected.to_string_lossy().to_string());
        }

        let mut files = fs::read_dir(dir)?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.is_file())
            .collect::<Vec<_>>();
        files.sort();

        files
            .into_iter()
            .next()
            .map(|path| path.to_string_lossy().to_string())
            .ok_or_else(|| io::Error::other("Archive did not contain a binary file").into())
    }

    async fn copy_to_remote(
        connection: &SshConnection,
        local_path: &str,
        remote_path: &str,
        _os: RemoteOs,
    ) -> Result<()> {
        remote_ssh::copy_to_remote(
            connection.clone(),
            local_path.to_string(),
            remote_path.to_string(),
            0o755,
        )
        .await
    }

    async fn move_binary_to_install_path(
        connection: &SshConnection,
        temp_path: &str,
        install_path: &str,
        os: RemoteOs,
    ) -> Result<()> {
        // Extract directory from install_path using string manipulation
        // (Path API doesn't handle Windows env vars like %APPDATA%)
        let install_dir = match os {
            RemoteOs::Windows => {
                // Find last backslash
                if let Some(pos) = install_path.rfind('\\') {
                    install_path[..pos].to_string()
                } else {
                    return Err(io::Error::other("no directory in install path").into());
                }
            }
            RemoteOs::Unix | RemoteOs::Macos => {
                // Find last forward slash
                if let Some(pos) = install_path.rfind('/') {
                    install_path[..pos].to_string()
                } else {
                    return Err(io::Error::other("no directory in install path").into());
                }
            }
            RemoteOs::Unknown => return Err(io::Error::other("Unknown OS").into()),
        };

        let quoted_dir = shell_quote_path(&install_dir, os);
        let mkdir_command = match os {
            RemoteOs::Windows => format!(
                "cmd.exe /C if not exist \"{}\" mkdir \"{}\"",
                install_dir, install_dir
            ),
            RemoteOs::Unix | RemoteOs::Macos => format!("mkdir -p {quoted_dir}"),
            RemoteOs::Unknown => return Err(io::Error::other("Unknown OS").into()),
        };

        let output = remote_ssh::exec(connection.clone(), mkdir_command)
            .await
            .map_err(|e| io::Error::other(e.to_string()))?;

        if output.status != 0 {
            return Err(io::Error::other(format!(
                "Failed to create install dir: {}",
                output.stderr.trim()
            ))
            .into());
        }

        let quoted_temp = shell_quote_path(temp_path, os);
        let quoted_install = shell_quote_path(install_path, os);
        let command = match os {
            RemoteOs::Windows => format!("cmd.exe /C move /Y \"{temp_path}\" \"{install_path}\""),
            RemoteOs::Unix | RemoteOs::Macos => format!("mv {quoted_temp} {quoted_install}"),
            RemoteOs::Unknown => return Err(io::Error::other("Unknown OS").into()),
        };

        let output = remote_ssh::exec(connection.clone(), command)
            .await
            .map_err(|e| io::Error::other(e.to_string()))?;

        if output.status == 0 {
            Ok(())
        } else {
            Err(io::Error::other(format!("Failed to move binary: {}", output.stderr.trim())).into())
        }
    }

    async fn extract_windows_archive_to_install_path(
        connection: &SshConnection,
        archive_path: &str,
        install_path: &str,
    ) -> Result<()> {
        let install_dir = install_path
            .rsplit_once('\\')
            .map(|(dir, _)| dir.to_string())
            .ok_or_else(|| io::Error::other("no directory in install path"))?;
        let extract_dir = format!("{install_dir}\\__local_llm_foundry_extract");

        // PowerShell single-quote escaping: double any embedded single quotes
        let ps_dir = install_dir.replace('\'', "''");
        let ps_extract = extract_dir.replace('\'', "''");
        let ps_archive = archive_path.replace('\'', "''");

        let command = format!(
            "powershell.exe -NoProfile -NonInteractive -Command \"$ErrorActionPreference = 'Stop'; \
if (!(Test-Path '{dir}')) {{ New-Item -ItemType Directory -Path '{dir}' -Force | Out-Null }}; \
if (Test-Path '{extract_dir}') {{ Remove-Item -LiteralPath '{extract_dir}' -Recurse -Force -ErrorAction SilentlyContinue }}; \
New-Item -ItemType Directory -Path '{extract_dir}' -Force | Out-Null; \
Expand-Archive -LiteralPath '{archive}' -DestinationPath '{extract_dir}' -Force; \
$targets = @('local-llm-foundry.exe', 'llama-monitor.exe', 'sensor_bridge.exe', 'WebView2Loader.dll'); \
foreach ($name in $targets) {{ \
  $src = Get-ChildItem -LiteralPath '{extract_dir}' -Recurse -File -Filter $name | Select-Object -First 1; \
  $dst = Join-Path '{dir}' $name; \
  if ($null -ne $src) {{ \
    for ($i = 0; $i -lt 10; $i++) {{ \
      if (Test-Path $dst) {{ Remove-Item -LiteralPath $dst -Force -ErrorAction SilentlyContinue }}; \
      try {{ [System.IO.File]::Copy($src.FullName, $dst, $true); Remove-Item -LiteralPath $src.FullName -Force -ErrorAction SilentlyContinue; break }} catch {{ if ($i -eq 9) {{ throw }}; Start-Sleep -Milliseconds 500 }} \
    }} \
  }} \
}}; \
if (!(Test-Path (Join-Path '{dir}' 'local-llm-foundry.exe'))) {{ throw 'Required local-llm-foundry.exe was not found in the Windows agent archive' }}; \
Remove-Item -LiteralPath '{extract_dir}' -Recurse -Force -ErrorAction SilentlyContinue; \
Remove-Item -LiteralPath '{archive}' -Force -ErrorAction SilentlyContinue\"",
            dir = ps_dir,
            extract_dir = ps_extract,
            archive = ps_archive
        );

        let output = remote_ssh::exec(connection.clone(), command)
            .await
            .map_err(|e| io::Error::other(e.to_string()))?;

        if output.status == 0 {
            Ok(())
        } else {
            Err(io::Error::other(format!(
                "Failed to extract Windows archive: {}",
                output.stderr.trim()
            ))
            .into())
        }
    }

    async fn prepare_windows_install_target(connection: &SshConnection) -> Result<()> {
        let command = format!(
            "powershell.exe -NoProfile -NonInteractive -Command \" \
Stop-ScheduledTask -TaskName '{WINDOWS_AGENT_TASK_NAME}' -ErrorAction SilentlyContinue; \
Stop-ScheduledTask -TaskName '{WINDOWS_SENSOR_BRIDGE_TASK_NAME}' -ErrorAction SilentlyContinue; \
Start-Sleep -Seconds 2; \
Stop-Process -Name local-llm-foundry -Force -ErrorAction SilentlyContinue; \
Stop-Process -Name llama-monitor -Force -ErrorAction SilentlyContinue; \
Stop-Process -Name sensor_bridge -Force -ErrorAction SilentlyContinue; \
Unregister-ScheduledTask -TaskName '{WINDOWS_AGENT_TASK_NAME}' -Confirm:$false -ErrorAction SilentlyContinue; \
Unregister-ScheduledTask -TaskName '{WINDOWS_AGENT_LEGACY_TASK_NAME}' -Confirm:$false -ErrorAction SilentlyContinue; \
Unregister-ScheduledTask -TaskName '{WINDOWS_SENSOR_BRIDGE_TASK_NAME}' -Confirm:$false -ErrorAction SilentlyContinue; \
Unregister-ScheduledTask -TaskName '{WINDOWS_SENSOR_BRIDGE_LEGACY_TASK_NAME}' -Confirm:$false -ErrorAction SilentlyContinue; \
Start-Sleep -Seconds 2\""
        );

        let output = remote_ssh::exec(connection.clone(), command)
            .await
            .map_err(|e| io::Error::other(e.to_string()))?;

        if output.status == 0 {
            Ok(())
        } else {
            Err(io::Error::other(format!(
                "Failed to stop existing Windows agent before install: {}",
                output.stderr.trim()
            ))
            .into())
        }
    }

    async fn set_executable_bit(
        connection: &SshConnection,
        path: &str,
        os: RemoteOs,
    ) -> Result<()> {
        let output = match os {
            RemoteOs::Unix | RemoteOs::Macos => {
                let quoted = shell_quote_path(path, os);
                remote_ssh::exec(connection.clone(), format!("chmod +x {quoted}"))
                    .await
                    .map_err(|e| io::Error::other(e.to_string()))?
            }
            _ => return Ok(()),
        };

        if output.status == 0 {
            Ok(())
        } else {
            Err(io::Error::other("Failed to set executable bit").into())
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RemoteAgentStartResponse {
        pub ok: bool,
        pub ssh_target: String,
        pub install_path: String,
        pub running: bool,
        pub health_reachable: bool,
        pub agent_token: Option<String>,
        pub error: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RemoteAgentUpdateResponse {
        pub ok: bool,
        pub ssh_target: String,
        pub previous_version: Option<String>,
        pub new_version: Option<String>,
        pub updated: bool,
        pub running: bool,
        pub health_reachable: bool,
        pub agent_token: Option<String>,
        pub error: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RemoteAgentStopResponse {
        pub ok: bool,
        pub ssh_target: String,
        pub stopped: bool,
        pub error: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RemoteAgentStatusResponse {
        pub ok: bool,
        pub ssh_target: String,
        pub os: String,
        pub install_path: String,
        pub installed: bool,
        pub running: bool,
        pub health_reachable: bool,
        #[serde(default)]
        pub metrics_reachable: bool,
        pub installed_version: Option<String>,
        pub managed_task_name: Option<String>,
        pub managed_task_installed: bool,
        pub managed_task_command: Option<String>,
        pub managed_task_matches: bool,
        #[serde(default)]
        pub agent_token: Option<String>,
        pub error: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RemoteAgentRemoveResponse {
        pub ok: bool,
        pub ssh_target: String,
        pub removed: bool,
        pub error: Option<String>,
    }

    #[derive(Debug, Clone)]
    pub struct ManagedTaskStatus {
        pub name: String,
        pub installed: bool,
        pub command: Option<String>,
        pub matches_install_path: bool,
    }

    /// Maps a Windows scheduled-task `LastTaskResult` code to a human-readable
    /// hint. The agent crashes in the OS loader (before `main()`) emit no agent
    /// logs, so the task exit code is the only signal we have for *why* it died.
    pub(crate) fn describe_windows_task_result(code: i64) -> Option<&'static str> {
        match code {
            0 => None,
            // 0x80070002 / ERROR_FILE_NOT_FOUND — Task Scheduler could not
            // launch the executable recorded in the task action.
            2_147_942_402 => Some("agent executable not found at the scheduled task path"),
            // 0xC0000135 STATUS_DLL_NOT_FOUND — a required DLL (e.g.
            // WebView2Loader.dll) is missing next to llama-monitor.exe.
            3_221_225_781 => {
                Some("missing DLL — WebView2Loader.dll is likely absent from the install directory")
            }
            // 0xC0000139 STATUS_ENTRYPOINT_NOT_FOUND
            3_221_225_785 => Some("missing DLL export (entry point not found)"),
            // 0x1 ERROR_INVALID_FUNCTION — task still running or generic failure.
            1 => Some("generic failure (task may still be starting)"),
            _ => Some("non-zero exit; the agent process failed to start"),
        }
    }

    /// Queries the managed agent scheduled task's `LastTaskResult` so a failed
    /// health check can report the real reason the process never came up.
    async fn windows_agent_task_diagnostic(connection: &SshConnection) -> Option<String> {
        let output = remote_ssh::exec(
            connection.clone(),
            format!(
                "powershell.exe -NoProfile -NonInteractive -Command \"(Get-ScheduledTaskInfo -TaskName '{WINDOWS_AGENT_TASK_NAME}').LastTaskResult\""
            ),
        )
        .await
        .ok()?;
        if output.status != 0 {
            return None;
        }
        let code: i64 = output.stdout.trim().parse().ok()?;
        let hint = describe_windows_task_result(code)?;
        Some(format!(
            "Scheduled task '{WINDOWS_AGENT_TASK_NAME}' last exit code {code} (0x{code:08X}): {hint}."
        ))
    }

    pub async fn start_remote_agent(
        ssh_target: &str,
        ssh_connection: Option<SshConnection>,
        install_path: &str,
        command: &str,
    ) -> Result<RemoteAgentStartResponse> {
        let connection = ssh_connection.unwrap_or_else(|| SshConnection::from_target(ssh_target));
        eprintln!(
            "[agent] Starting remote agent on {} with command: {}",
            connection.target_label(),
            command
        );
        eprintln!("[agent] Install path: {}", install_path);

        // Ensure CA and server certs are provisioned before starting. The agent
        // requires the CA to start (mTLS), and the install step may have failed
        // to copy it if the cas/ directory wasn't created.
        let os_for_certs = if install_path.contains('\\')
            || install_path.to_ascii_lowercase().contains("appdata")
        {
            RemoteOs::Windows
        } else {
            RemoteOs::Unix
        };
        drop_ca_certificate(&connection, install_path, os_for_certs).await;
        drop_agent_server_certificate(&connection, install_path, os_for_certs).await;
        let start_warning = match tokio::time::timeout(
            Duration::from_secs(15),
            remote_ssh::exec(connection.clone(), command.to_string()),
        )
        .await
        {
            Ok(Ok(output)) if output.status == 0 => None,
            Ok(Ok(output)) => {
                let error_msg = if !output.stderr.is_empty() {
                    format!("Start command failed: {}", output.stderr.trim())
                } else {
                    format!("Start command exited with status: {}", output.status)
                };
                return Ok(RemoteAgentStartResponse {
                    ok: false,
                    ssh_target: connection.target_label(),
                    install_path: install_path.to_string(),
                    running: false,
                    health_reachable: false,
                    agent_token: None,
                    error: Some(error_msg),
                });
            }
            Ok(Err(e)) => return Err(e),
            Err(_) => Some(
                "Start command did not return within 15 seconds; checking agent health".to_string(),
            ),
        };

        let os_hint = if install_path.contains('\\')
            || install_path.to_ascii_lowercase().contains("appdata")
        {
            RemoteOs::Windows
        } else {
            RemoteOs::Unix
        };

        let agent_url = connection.agent_url(REMOTE_AGENT_DEFAULT_PORT);
        eprintln!("[agent] Checking agent health at {}", agent_url);
        // /health requires no auth; token is read after startup to avoid a race
        // where a freshly-started agent hasn't written its token file yet.
        //
        // Build HTTPS/HTTP clients once and reuse them across all attempts.
        // A 2-second per-request timeout plus a 3-second gap between attempts
        // keeps 5 attempts well within the 30-second outer timeout even on slow
        // or firewalled networks.
        //
        // Wait a few seconds before the first poll: the scheduled-task startup
        // on Windows typically takes 3–5 s, so an immediate check produces
        // several noisy "error sending request" lines before the agent is ready.
        const HEALTH_CHECK_ATTEMPTS: u32 = 5;
        let health_https = build_agent_https_client(Duration::from_secs(2));
        let health_http = build_plain_http_client(Duration::from_secs(2));
        let health_reachable = tokio::time::timeout(Duration::from_secs(30), async {
            tokio::time::sleep(Duration::from_secs(3)).await;
            for i in 1..=HEALTH_CHECK_ATTEMPTS {
                eprintln!("[agent] Health check attempt {i}/{HEALTH_CHECK_ATTEMPTS}...");
                let reached = 'check: {
                    for candidate in agent_url_candidates(&agent_url) {
                        let client =
                            agent_client_for_url(&candidate, health_https.as_ref(), &health_http);
                        match client.get(format!("{}/health", candidate)).send().await {
                            Ok(resp) if resp.status().is_success() => {
                                break 'check true;
                            }
                            Ok(resp) => {
                                eprintln!(
                                    "[agent] Health check {} returned status {}",
                                    candidate,
                                    resp.status()
                                );
                            }
                            Err(e) => {
                                eprintln!("[agent] Health check {} error: {}", candidate, e);
                            }
                        }
                    }
                    false
                };
                if reached {
                    eprintln!("[agent] Agent health check passed");
                    return true;
                }
                if i < HEALTH_CHECK_ATTEMPTS {
                    tokio::time::sleep(Duration::from_secs(3)).await;
                }
            }
            false
        })
        .await;

        let running = matches!(health_reachable, Ok(true));

        let error = if !running {
            let mut msg = match health_reachable {
                Err(tokio::time::error::Elapsed { .. }) => "Agent did not start within 30 seconds. Check if the agent is listening on 0.0.0.0:7779 and if the remote firewall allows inbound connections on port 7779.".to_string(),
                Ok(_) => "Agent started but is not reachable. Check SSH access and firewall rules on port 7779.".to_string(),
            };
            // On Windows the agent can crash in the OS loader before main()
            // (e.g. a missing executable or DLL), emitting no agent logs.
            // Always query the scheduled task after a failed health check,
            // including the common Ok(false) path where the outer timeout did
            // not expire.
            if os_hint == RemoteOs::Windows
                && let Some(diag) = windows_agent_task_diagnostic(&connection).await
            {
                eprintln!("[agent] {diag}");
                msg.push(' ');
                msg.push_str(&diag);
            }
            if let Some(warning) = start_warning {
                msg.push(' ');
                msg.push_str(&warning);
            }
            Some(msg)
        } else {
            None
        };

        // Read the token after the agent is confirmed running. A newly started
        // agent writes its token file during initialization, so retry briefly to
        // handle the case where the file doesn't exist yet.
        let agent_token = if running {
            let mut token = None;
            for _ in 0..6 {
                token = read_remote_agent_token(&connection, os_hint, Some(&agent_url)).await;
                if token.is_some() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
            token
        } else {
            None
        };

        Ok(RemoteAgentStartResponse {
            ok: running,
            ssh_target: connection.target_label(),
            install_path: install_path.to_string(),
            running,
            health_reachable: running,
            agent_token,
            error,
        })
    }

    pub async fn update_remote_agent(
        ssh_target: &str,
        ssh_connection: Option<SshConnection>,
    ) -> Result<RemoteAgentUpdateResponse> {
        let connection = ssh_connection.unwrap_or_else(|| SshConnection::from_target(ssh_target));
        let latest_release = latest_release_info().await?;
        let remote_version = get_remote_version_with(connection.clone()).await?;
        let remote_os = detect_remote_os_with(&connection).await;
        let remote_arch = detect_remote_arch_with(&connection, remote_os).await;

        let matching_asset = latest_release
            .matching_asset(remote_os.as_str(), &remote_arch)
            .cloned();

        if matching_asset.is_none() {
            return Ok(RemoteAgentUpdateResponse {
                ok: false,
                ssh_target: connection.target_label(),
                previous_version: remote_version,
                new_version: Some(latest_release.tag_name),
                updated: false,
                running: false,
                health_reachable: false,
                agent_token: None,
                error: Some("No matching asset for remote platform".to_string()),
            });
        }

        let install_path = preferred_remote_install_path(&connection, remote_os).await;
        let install_path_clone = install_path.clone();
        let stop_response = stop_remote_agent(ssh_target, Some(connection.clone())).await?;

        if !stop_response.stopped {
            return Ok(RemoteAgentUpdateResponse {
                ok: false,
                ssh_target: connection.target_label(),
                previous_version: remote_version,
                new_version: Some(latest_release.tag_name),
                updated: false,
                running: false,
                health_reachable: false,
                agent_token: None,
                error: Some("Failed to stop agent before update".to_string()),
            });
        }

        let install_response = install_remote_agent(
            ssh_target,
            Some(connection.clone()),
            &matching_asset.unwrap(),
            Some(install_path_clone),
            remote_os,
            None,
        )
        .await?;

        if !install_response.installed {
            return Ok(RemoteAgentUpdateResponse {
                ok: false,
                ssh_target: connection.target_label(),
                previous_version: remote_version,
                new_version: Some(latest_release.tag_name),
                updated: false,
                running: false,
                health_reachable: false,
                agent_token: None,
                error: Some("Failed to install updated agent".to_string()),
            });
        }

        let start_response = start_remote_agent(
            ssh_target,
            Some(connection.clone()),
            &install_path,
            &default_start_command_for_os_with(&connection, remote_os, &install_path).await,
        )
        .await?;

        Ok(RemoteAgentUpdateResponse {
            ok: start_response.running,
            ssh_target: connection.target_label(),
            previous_version: remote_version,
            new_version: Some(latest_release.tag_name),
            updated: start_response.running,
            running: start_response.running,
            health_reachable: start_response.health_reachable,
            agent_token: start_response.agent_token,
            error: if start_response.running {
                None
            } else {
                Some("Agent failed to start after update".to_string())
            },
        })
    }

    pub async fn stop_remote_agent(
        ssh_target: &str,
        ssh_connection: Option<SshConnection>,
    ) -> Result<RemoteAgentStopResponse> {
        let connection = ssh_connection.unwrap_or_else(|| SshConnection::from_target(ssh_target));
        let os = detect_remote_os_with(&connection).await;
        let command = match os {
            RemoteOs::Windows => {
                "cmd.exe /C (taskkill /IM local-llm-foundry.exe /F >NUL 2>NUL & taskkill /IM llama-monitor.exe /F >NUL 2>NUL)"
            }
            RemoteOs::Unix | RemoteOs::Macos => {
                "pkill -x local-llm-foundry >/dev/null 2>&1; pkill -x llama-monitor >/dev/null 2>&1; true"
            }
            RemoteOs::Unknown => return Err(io::Error::other("Unknown OS").into()),
        };

        let output = remote_ssh::exec(connection.clone(), command.to_string()).await?;

        Ok(RemoteAgentStopResponse {
            ok: output.status == 0,
            ssh_target: connection.target_label(),
            stopped: output.status == 0,
            error: if output.status == 0 {
                None
            } else {
                Some("Failed to stop agent".to_string())
            },
        })
    }

    pub async fn status_remote_agent(
        ssh_target: &str,
        ssh_connection: Option<SshConnection>,
        agent_url: Option<&str>,
    ) -> Result<RemoteAgentStatusResponse> {
        let connection = ssh_connection.unwrap_or_else(|| SshConnection::from_target(ssh_target));
        let os = detect_remote_os_with(&connection).await;
        if os == RemoteOs::Unknown {
            return Ok(RemoteAgentStatusResponse {
                ok: false,
                ssh_target: connection.target_label(),
                os: os.as_str().to_string(),
                install_path: String::new(),
                installed: false,
                running: false,
                health_reachable: false,
                metrics_reachable: false,
                installed_version: None,
                managed_task_name: None,
                managed_task_installed: false,
                managed_task_command: None,
                managed_task_matches: false,
                agent_token: None,
                error: Some("Unknown remote OS".to_string()),
            });
        }

        let install_path = preferred_remote_install_path(&connection, os).await;
        let installed = remote_file_exists_with(&connection, os, &install_path).await;
        let agent_url = agent_url
            .map(str::trim)
            .filter(|url| !url.is_empty())
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| connection.agent_url(REMOTE_AGENT_DEFAULT_PORT));
        let health_reachable = agent_health_reachable(&agent_url).await;
        let agent_token = read_remote_agent_token(&connection, os, Some(&agent_url)).await;
        let metrics_reachable =
            agent_metrics_reachable_with_token(&agent_url, agent_token.as_deref()).await;
        let installed_version = if installed {
            get_remote_version_with(connection.clone())
                .await
                .ok()
                .flatten()
        } else {
            None
        };
        let managed_task = managed_task_status(&connection, os, Some(&install_path))
            .await
            .ok()
            .flatten();
        let managed_task_name = managed_task.as_ref().map(|task| task.name.clone());
        let managed_task_installed = managed_task.as_ref().is_some_and(|task| task.installed);
        let managed_task_command = managed_task.as_ref().and_then(|task| task.command.clone());
        let managed_task_matches = managed_task
            .as_ref()
            .is_some_and(|task| task.matches_install_path);

        Ok(RemoteAgentStatusResponse {
            ok: true,
            ssh_target: connection.target_label(),
            os: os.as_str().to_string(),
            install_path,
            installed,
            running: health_reachable,
            health_reachable,
            metrics_reachable,
            installed_version,
            managed_task_name,
            managed_task_installed,
            managed_task_command,
            managed_task_matches,
            agent_token,
            error: None,
        })
    }

    pub async fn remove_remote_agent(
        ssh_target: &str,
        ssh_connection: Option<SshConnection>,
    ) -> Result<RemoteAgentRemoveResponse> {
        let connection = ssh_connection.unwrap_or_else(|| SshConnection::from_target(ssh_target));
        let os = detect_remote_os_with(&connection).await;
        let install_path = preferred_remote_install_path(&connection, os).await;
        let command = match os {
            RemoteOs::Windows => format!(
                "cmd.exe /C (taskkill /IM local-llm-foundry.exe /F >NUL 2>NUL & taskkill /IM llama-monitor.exe /F >NUL 2>NUL & schtasks /Delete /TN \"{WINDOWS_AGENT_TASK_NAME}\" /F >NUL 2>NUL & schtasks /Delete /TN \"{WINDOWS_AGENT_LEGACY_TASK_NAME}\" /F >NUL 2>NUL & del /F /Q \"{install_path}\" >NUL 2>NUL & del /F /Q \"%APPDATA%\\llama-monitor\\bin\\llama-monitor.exe\" >NUL 2>NUL & exit /B 0)"
            ),
            RemoteOs::Unix | RemoteOs::Macos => {
                let quoted = shell_quote_path(&install_path, os);
                format!(
                    "pkill -x local-llm-foundry >/dev/null 2>&1; pkill -x llama-monitor >/dev/null 2>&1; rm -f {quoted}"
                )
            }
            RemoteOs::Unknown => return Err(io::Error::other("Unknown OS").into()),
        };

        let output = remote_ssh::exec(connection.clone(), command).await?;

        Ok(RemoteAgentRemoveResponse {
            ok: output.status == 0,
            ssh_target: connection.target_label(),
            removed: output.status == 0,
            error: if output.status == 0 {
                None
            } else {
                Some("Failed to remove managed agent".to_string())
            },
        })
    }

    pub async fn managed_task_status(
        connection: &SshConnection,
        os: RemoteOs,
        install_path: Option<&str>,
    ) -> Result<Option<ManagedTaskStatus>> {
        if os != RemoteOs::Windows {
            return Ok(None);
        }

        for task_name in [WINDOWS_AGENT_TASK_NAME, WINDOWS_AGENT_LEGACY_TASK_NAME] {
            let output = remote_ssh::exec(
                connection.clone(),
                format!("cmd.exe /C schtasks /Query /TN \"{task_name}\" /V /FO LIST"),
            )
            .await?;
            if output.status != 0 {
                continue;
            }

            let command = output
                .stdout
                .lines()
                .find_map(|line| line.trim().strip_prefix("Task To Run:").map(str::trim))
                .filter(|value| !value.is_empty() && *value != "N/A")
                .map(ToOwned::to_owned);
            let appdata = resolve_windows_appdata(connection).await;
            let matches_install_path = command.as_deref().is_some_and(|command| {
                install_path.is_some_and(|path| {
                    windows_path_matches_install_path(command, path, appdata.as_deref())
                })
            });

            return Ok(Some(ManagedTaskStatus {
                name: task_name.to_string(),
                installed: true,
                command,
                matches_install_path,
            }));
        }

        Ok(Some(ManagedTaskStatus {
            name: WINDOWS_AGENT_TASK_NAME.to_string(),
            installed: false,
            command: None,
            matches_install_path: false,
        }))
    }

    pub async fn get_remote_version_with(connection: SshConnection) -> Result<Option<String>> {
        let remote_os = detect_remote_os_with(&connection).await;
        let install_path = preferred_remote_install_path(&connection, remote_os).await;
        let command = match remote_os {
            RemoteOs::Windows => format!("cmd.exe /C \"\"{install_path}\" --version\""),
            RemoteOs::Unix | RemoteOs::Macos => format!("{install_path} --version"),
            RemoteOs::Unknown => return Ok(None),
        };
        let output = remote_ssh::exec(connection, command).await?;

        if output.status == 0 {
            Ok(Some(output.stdout.trim().to_string()))
        } else {
            Ok(None)
        }
    }

    pub async fn default_start_command_for_target(ssh_target: &str, install_path: &str) -> String {
        default_start_command_for_os(detect_remote_os(ssh_target).await, install_path)
    }

    pub async fn detect_remote_os_simple(ssh_target: &str) -> RemoteOs {
        detect_remote_os(ssh_target).await
    }

    #[derive(Debug, Serialize)]
    pub struct SelfUpdateResult {
        pub tag_name: String,
    }

    /// Replace the running binary with the latest release from GitHub.
    ///
    /// On macOS/Linux: downloads the asset, extracts if needed, copies it into
    /// the same directory as the running binary, then atomically renames it over
    /// the current executable. The running process keeps its old inode in memory,
    /// so the rename is safe.
    ///
    /// On Windows: in-place replacement of a running `.exe` is blocked by the OS.
    /// Downloads the new binary, writes a small batch helper to %TEMP%, and spawns
    /// it as a detached process. The batch file waits for this PID to exit, copies
    /// the new binary over, and relaunches. `process::exit(0)` is then called by
    /// the API handler after returning a response.
    /// Download checksums.json from the release and return its parsed JSON.
    /// Fails hard if missing or malformed.
    async fn fetch_checksums_json(
        release: &crate::agent::LatestReleaseInfo,
    ) -> Result<serde_json::Value> {
        let checksums_url = release
            .checksums_url
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("Update failed: no checksums file in release"))?;

        let resp = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(15))
            .timeout(std::time::Duration::from_secs(60))
            .build()?
            .get(checksums_url)
            .header(
                reqwest::header::USER_AGENT,
                crate::identity::RELEASE_USER_AGENT,
            )
            .send()
            .await?
            .error_for_status()?;

        let bytes = resp.bytes().await?;
        let json: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|_| anyhow::anyhow!("Update failed: invalid checksums file"))?;

        if json.get("checksums").and_then(|v| v.as_object()).is_none() {
            return Err(anyhow::anyhow!("Update failed: invalid checksums file"));
        }

        Ok(json)
    }

    /// Verify the SHA-256 checksum of the downloaded asset.
    /// Fails hard if entry is missing or hash does not match.
    async fn verify_checksum(
        checksums: &serde_json::Value,
        asset_name: &str,
        asset_path: &str,
    ) -> Result<()> {
        let expected = checksums
            .get("checksums")
            .and_then(|c| c.get(asset_name))
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Update failed: checksum mismatch for {asset_name}"))?;

        let data = fs::read(asset_path)
            .map_err(|e| anyhow::anyhow!("Update failed: cannot read downloaded file: {e}"))?;

        use sha2::Digest;
        let mut hasher = sha2::Sha256::new();
        hasher.update(&data);
        let computed = hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();

        if computed != expected {
            return Err(anyhow::anyhow!(
                "Update failed: checksum mismatch for {asset_name}"
            ));
        }

        Ok(())
    }

    pub async fn self_update_binary(web_port: u16, agent_port: u16) -> Result<SelfUpdateResult> {
        #[cfg(not(unix))]
        let _ = (web_port, agent_port);

        let os = std::env::consts::OS;
        let arch = std::env::consts::ARCH;

        let release = crate::agent::latest_release_info().await?;

        if os == "windows" {
            #[cfg(windows)]
            return self_update_binary_windows(&release, arch).await;
            #[cfg(not(windows))]
            return Err(anyhow::anyhow!(
                "Windows update path is not available in this build"
            ));
        }

        // Unix/macOS: ports used for restart-launcher wait (passed from AppConfig).
        #[cfg(unix)]
        let unix_web_port = web_port;
        #[cfg(unix)]
        let unix_agent_port = agent_port;

        let asset = release
            .matching_asset(os, arch)
            .ok_or_else(|| anyhow::anyhow!("No release asset for {os}/{arch}"))?
            .clone();

        let local_path = download_asset_locally(&asset).await?;

        // Verify checksum before using the asset
        let checksums = fetch_checksums_json(&release).await?;
        verify_checksum(&checksums, &asset.name, &local_path).await?;

        let binary_path = if asset.archive {
            extract_archive_with_timeout(&local_path, &asset).await?
        } else {
            local_path.clone()
        };

        let current_exe = std::env::current_exe()
            .map_err(|e| anyhow::anyhow!("Cannot locate current binary: {e}"))?;

        // Stage in the same directory so rename stays on one filesystem.
        let parent = current_exe
            .parent()
            .ok_or_else(|| anyhow::anyhow!("Binary path has no parent directory"))?;
        let staged = parent.join(format!(
            "{}{}",
            crate::identity::UPDATE_STAGE_PREFIX,
            std::process::id()
        ));

        fs::copy(&binary_path, &staged).map_err(|e| {
            if e.kind() == std::io::ErrorKind::PermissionDenied {
                anyhow::anyhow!(
                    "Update failed: llama-monitor does not have permission to write to its install \
                     folder. Move it to a user-writable location or run as administrator."
                )
            } else {
                anyhow::anyhow!("Update failed: Cannot stage update: {e}")
            }
        })?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&staged, fs::Permissions::from_mode(0o755))
                .map_err(|e| anyhow::anyhow!("Cannot set executable permission: {e}"))?;
        }

        // Atomic rename — safe on Unix even while this process is running.
        if let Err(e) = fs::rename(&staged, &current_exe) {
            let _ = fs::remove_file(&staged);
            if e.kind() == std::io::ErrorKind::PermissionDenied {
                return Err(anyhow::anyhow!(
                    "Update failed: llama-monitor does not have permission to write to its \
                     install folder. Move it to a user-writable location or run as \
                     administrator."
                ));
            }
            return Err(anyhow::anyhow!("Update failed: Cannot replace binary: {e}"));
        }

        // Clean up the downloaded archive and the extraction temp dir. When the
        // asset was an archive, `binary_path` lives inside a `tempfile` dir that
        // `extract_archive` deliberately leaked (`.keep()`); remove its parent so
        // we don't accumulate one stale dir in $TMPDIR per update. When the asset
        // was a bare binary, `binary_path == local_path` and the file is already
        // gone from the rename above, so only the archive cleanup applies.
        if asset.archive {
            if let Some(extract_dir) = std::path::Path::new(&binary_path).parent() {
                let _ = fs::remove_dir_all(extract_dir);
            }
            let _ = fs::remove_file(&local_path);
        } else {
            let _ = fs::remove_file(&binary_path);
        }

        // On Unix/macOS: spawn a detached launcher so the app restarts after update.
        // Features:
        //  - Port-check: waits for the web/agent listen ports to be released by the
        //    exiting process before relaunching, so the new process can bind them.
        //  - Arg forwarding: preserves the original CLI arguments across the restart.
        //  - Logging: writes to a small temp log file so failures are inspectable.
        #[cfg(unix)]
        {
            use std::time::{SystemTime, UNIX_EPOCH};

            let home = std::env::var("HOME").unwrap_or_default();
            let data_dir = if !home.is_empty() {
                &home
            } else {
                // Fallback; not ideal but keeps us safe
                "/tmp"
            };
            let log_file = format!("{data_dir}/.local-llm-foundry-restart.log");

            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            // `lsof` is present on both macOS and Linux; `ss`/`netstat` are not
            // universally available (`ss` is absent on macOS entirely). We poll
            // for the LISTEN sockets to disappear, with a bounded retry budget so a
            // stuck port never wedges the relaunch forever.
            let launcher_script = format!(
                r#"
                WEB={web_port}
                AGENT={agent_port}
                port_in_use() {{
                    lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
                }}
                for i in $(seq 1 25); do
                    if ! port_in_use "$WEB" && ! port_in_use "$AGENT"; then
                        break
                    fi
                    sleep 0.2
                done

                exec "$LLAMA_RESTART_BIN" "$@"
                "#,
                web_port = unix_web_port,
                agent_port = unix_agent_port,
            );

            let binary_path = current_exe.to_string_lossy().to_string();

            // Forward the original CLI arguments (everything after argv[0]) so a
            // non-default launch (custom port, config path, …) survives the restart.
            let forwarded_args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();

            match std::process::Command::new("sh")
                .arg("-c")
                .arg(&launcher_script)
                // `$0` placeholder for `sh -c`; forwarded args become `$1`, `$2`, …
                .arg("llama-monitor-restart")
                .args(&forwarded_args)
                .env("LLAMA_RESTART_BIN", &binary_path)
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
            {
                Ok(_) => {
                    // Best-effort log that we started the launcher
                    let _ = fs::write(&log_file, format!("{} LAUNCHER_START\n", now));
                }
                Err(e) => {
                    // Log spawn failure for debugging
                    let _ = fs::write(&log_file, format!("{} LAUNCHER_SPAWN_FAIL: {}\n", now, e));
                }
            }
        }

        Ok(SelfUpdateResult {
            tag_name: release.tag_name,
        })
    }

    /// Windows-specific self-update path.
    ///
    /// Cannot rename over a running `.exe`, so instead:
    /// 1. Downloads and extracts the new binary to a temp path.
    /// 2. Writes a batch helper to %TEMP% that polls until this PID exits,
    ///    then does `copy /Y new_exe current_exe` and relaunches.
    /// 3. Spawns the batch helper as a DETACHED_PROCESS so it outlives us.
    ///
    /// The caller (`api_self_update`) schedules `process::exit(0)` after
    /// returning the HTTP response, which unblocks the batch wait loop.
    #[cfg(windows)]
    async fn self_update_binary_windows(
        release: &crate::agent::LatestReleaseInfo,
        arch: &str,
    ) -> Result<SelfUpdateResult> {
        use std::os::windows::process::CommandExt;
        const DETACHED_PROCESS: u32 = 0x0000_0008;

        let asset = release
            .matching_asset("windows", arch)
            .ok_or_else(|| anyhow::anyhow!("No Windows release asset for arch {arch}"))?
            .clone();

        let local_path = download_asset_locally(&asset).await?;

        // Verify checksum before using the asset
        let checksums = fetch_checksums_json(release).await?;
        verify_checksum(&checksums, &asset.name, &local_path).await?;

        // extract_archive_with_timeout:
        //   - creates a temp directory,
        //   - extracts the release zip,
        //   - returns one file path (via extracted_binary_path).
        //
        // We need the full extracted directory so we can copy all release files
        // (llama-monitor.exe, sensor_bridge.exe, WebView2Loader.dll) into the
        // original install folder, not just llama-monitor.exe.
        let _binary_path = extract_archive_with_timeout(&local_path, &asset).await?;
        let extract_dir = _binary_path
            .rsplit_once('\\')
            .map(|(dir, _)| dir.to_string())
            .unwrap_or_else(|| _binary_path.clone());

        let current_exe = std::env::current_exe()
            .map_err(|e| anyhow::anyhow!("Cannot locate current binary: {e}"))?;

        // Destination directory (where llama-monitor is installed).
        let install_dir = current_exe
            .parent()
            .ok_or_else(|| anyhow::anyhow!("Current exe has no parent directory"))?
            .to_string_lossy()
            .replace('/', "\\");

        // Collect all extracted files (flat layout from CI).
        let mut files: Vec<String> = match fs::read_dir(&extract_dir) {
            Ok(rd) => rd
                .filter_map(|entry| entry.ok())
                .filter_map(|e| {
                    let p = e.path();
                    if p.is_file() {
                        Some(p.to_string_lossy().replace('/', "\\"))
                    } else {
                        None
                    }
                })
                .collect(),
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "Cannot read extracted update directory: {e}"
                ));
            }
        };
        files.sort();

        if files.is_empty() {
            return Err(anyhow::anyhow!("Extracted archive contained no files"));
        }

        let pid = std::process::id();
        let batch_path = std::env::temp_dir().join(format!("lm-update-{pid}.bat"));
        let current_exe_batch = current_exe.to_string_lossy().replace('\\', "\\\\");

        // The batch file:
        //   :check      — loop until this PID disappears from tasklist
        //   copy        — copy each release file into the install directory
        //   start       — relaunch from the same path
        //   rmdir       — clean temp extract directory
        //   del         — self-destruct
        //
        // We copy ALL files instead of only llama-monitor.exe so that
        // sensor_bridge.exe and WebView2Loader.dll are updated in-place.
        let copy_lines: String = files
            .iter()
            .map(|src| {
                let src_batch = src.replace('\\', "\\\\");
                format!(
                    "    for %%F in (\"{src_batch}\") do copy /Y \"%%F\" \"{install_dir}\\%%~nxF\"\r\n",
                    src_batch = src_batch,
                    install_dir = install_dir.replace('\\', "\\\\"),
                )
            })
            .collect();

        let batch = format!(
            "@echo off\r\n\
             :check\r\n\
             tasklist /FI \"PID eq {pid}\" 2>NUL | find /I \"exe\" >NUL\r\n\
             if not errorlevel 1 (\r\n\
                 timeout /t 1 /nobreak >NUL\r\n\
                 goto check\r\n\
             )\r\n\
             {copy_lines}\
             start \"\" \"{current_exe_batch}\"\r\n\
             rmdir /S /Q \"{extract_dir}\"\r\n\
             (goto) 2>NUL & del \"%~f0\"\r\n",
            pid = pid,
            copy_lines = copy_lines,
            current_exe_batch = current_exe_batch,
            extract_dir = extract_dir.replace('\\', "\\\\"),
        );

        fs::write(&batch_path, &batch)
            .map_err(|e| anyhow::anyhow!("Cannot write update helper to temp dir: {e}"))?;

        std::process::Command::new("cmd.exe")
            .args(["/C", &batch_path.to_string_lossy()])
            .creation_flags(DETACHED_PROCESS)
            .spawn()
            .map_err(|e| anyhow::anyhow!("Cannot launch update helper: {e}"))?;

        Ok(SelfUpdateResult {
            tag_name: release.tag_name.clone(),
        })
    }
}

pub use install::{
    RemoteAgentInstallRequest, default_start_command_for_target, detect_remote_os_simple,
    install_remote_agent, remove_remote_agent, self_update_binary, start_remote_agent,
    status_remote_agent, stop_remote_agent, update_remote_agent,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_cli_and_release_versions() {
        assert_eq!(normalize_version_label("llama-monitor 0.5.1"), "0.5.1");
        assert_eq!(normalize_version_label("other-agent 0.5.1"), "0.5.1");
        assert_eq!(normalize_version_label("v0.5.1"), "0.5.1");
    }

    #[test]
    fn matching_asset_prefers_canonical_family_over_legacy_order() {
        let release = LatestReleaseInfo {
            tag_name: "v2.0.0".into(),
            name: None,
            html_url: None,
            body: None,
            published_at: None,
            checksums_url: Some("https://example.test/checksums.json".into()),
            assets: vec![
                ReleaseAssetInfo {
                    name: "llama-monitor-linux-x86_64".into(),
                    url: String::new(),
                    size: 1,
                    platform: "linux".into(),
                    arch: "x86_64".into(),
                    archive: false,
                },
                ReleaseAssetInfo {
                    name: "local-llm-foundry-linux-x86_64".into(),
                    url: String::new(),
                    size: 1,
                    platform: "linux".into(),
                    arch: "x86_64".into(),
                    archive: false,
                },
            ],
        };
        assert_eq!(
            release.matching_asset("linux", "amd64").unwrap().name,
            "local-llm-foundry-linux-x86_64"
        );
    }

    #[test]
    fn release_checksum_url_is_independent_of_filtered_binary_assets() {
        let checksum = "https://example.test/checksums.json";
        let release = LatestReleaseInfo {
            tag_name: "v2.0.0".into(),
            name: None,
            html_url: None,
            body: None,
            published_at: None,
            checksums_url: Some(checksum.into()),
            assets: Vec::new(),
        };
        assert_eq!(release.checksums_url.as_deref(), Some(checksum));
    }

    #[test]
    fn managed_install_path_candidates_cover_rebranded_legacy_binary() {
        let candidates = managed_install_path_candidates(RemoteOs::Windows);
        assert_eq!(candidates[0], crate::identity::install_path(true));
        assert_eq!(
            candidates[2],
            "%APPDATA%\\llama-monitor\\bin\\local-llm-foundry.exe"
        );
        assert_eq!(candidates[3], crate::identity::legacy_install_path(true));
    }

    #[test]
    fn managed_install_path_matching_preserves_custom_paths() {
        assert!(managed_install_path_matches(
            "%APPDATA%\\LOCAL-LLM-FOUNDRY\\bin\\local-llm-foundry.exe",
            crate::identity::install_path(true)
        ));
        assert!(managed_install_path_matches(
            crate::identity::legacy_install_path(false),
            crate::identity::legacy_install_path(false)
        ));
        assert!(!managed_install_path_matches(
            "/opt/custom-agent/bin/agent",
            crate::identity::install_path(false)
        ));
    }

    #[test]
    fn windows_task_path_matches_expanded_appdata_install_path() {
        assert!(windows_path_matches_install_path(
            r#"C:\Users\nick\AppData\Roaming\llama-monitor\bin\local-llm-foundry.exe --agent --config-dir C:\Users\nick\AppData\Roaming\llama-monitor"#,
            r#"%APPDATA%\llama-monitor\bin\local-llm-foundry.exe"#,
            Some(r#"C:\Users\nick\AppData\Roaming"#),
        ));
    }

    #[test]
    fn windows_task_path_match_is_case_and_slash_insensitive() {
        assert!(windows_path_matches_install_path(
            r#""c:/USERS/nick/AppData/Roaming/local-llm-foundry/bin/local-llm-foundry.exe" --agent"#,
            r#"%appdata%\local-llm-foundry\bin\local-llm-foundry.exe"#,
            Some(r#"C:\Users\nick\AppData\Roaming"#),
        ));
    }

    #[test]
    fn windows_task_path_does_not_match_different_install_root() {
        assert!(!windows_path_matches_install_path(
            r#"C:\Users\nick\AppData\Roaming\local-llm-foundry\bin\local-llm-foundry.exe --agent"#,
            r#"%APPDATA%\llama-monitor\bin\local-llm-foundry.exe"#,
            Some(r#"C:\Users\nick\AppData\Roaming"#),
        ));
    }

    #[test]
    fn shell_quote_path_unix_escapes_dangerous_chars() {
        let path = "/opt/test; rm -rf /";
        let quoted = shell_quote_path(path, RemoteOs::Unix);
        // shlex wraps in single quotes; the result should be a single quoted string
        assert!(quoted.starts_with('\''));
        assert!(quoted.ends_with('\''));
        // The semicolon should be inside the quotes, not interpreted as command separator
        assert!(quoted.contains(";"));
    }

    #[test]
    fn shell_quote_path_unix_handles_single_quotes() {
        let path = "/opt/it's a test";
        let quoted = shell_quote_path(path, RemoteOs::Unix);
        // shlex uses double quotes when the string contains single quotes
        // Verify: starts and ends with a quote character, spaces are inside quotes
        assert!(
            (quoted.starts_with('\'') && quoted.ends_with('\''))
                || (quoted.starts_with('"') && quoted.ends_with('"')),
            "quoted path should be wrapped in quotes: {:?}",
            quoted
        );
    }

    #[test]
    fn shell_quote_path_windows_doubles_single_quotes() {
        let path = r#"C:\Program Files\llama-monitor"#;
        let quoted = shell_quote_path(path, RemoteOs::Windows);
        assert!(quoted.starts_with('\''));
        assert!(quoted.ends_with('\''));
    }

    #[test]
    fn shell_quote_path_windows_escapes_embedded_single_quotes() {
        let path = r#"C:\It's a test\llama"#;
        let quoted = shell_quote_path(path, RemoteOs::Windows);
        // Single quotes should be doubled inside the quoted string
        assert!(quoted.contains("''"));
        assert!(!quoted.contains(r#"\'"#));
    }

    #[test]
    fn shell_quote_path_cmd_uses_double_quotes() {
        // cmd.exe does NOT treat single quotes as special
        let path = r#"C:\Program Files\llama-monitor"#;
        let quoted = shell_quote_path_cmd(path);
        assert!(quoted.starts_with('"'));
        assert!(quoted.ends_with('"'));
        assert!(!quoted.contains('\''));
    }

    #[test]
    fn shell_quote_path_cmd_escapes_embedded_double_quotes() {
        let path = r#"C:\Pro"gram Files\llama"#;
        let quoted = shell_quote_path_cmd(path);
        // Embedded double quotes should be escaped with ^
        assert!(quoted.contains("^\""));
    }

    #[test]
    fn validate_install_path_rejects_shell_injection() {
        let malicious_paths = [
            "/opt/test; rm -rf /",
            "/opt/test|whoami",
            "/opt/test&echo hacked",
            "/opt/test`id`",
            "/opt/test$(whoami)",
            "/opt/test'break'out",
            "/opt/test\"break\"out",
            "/opt/test> /dev/null",
            "/opt/test< /etc/passwd",
            "/opt/test!command",
            "/opt/test#comment",
            "/opt/test*glob",
            "/opt/test?question",
        ];

        for path in malicious_paths {
            // Malicious paths should be rejected regardless of target OS
            let result_unix = validate_install_path(path, RemoteOs::Unix);
            let result_windows = validate_install_path(path, RemoteOs::Windows);
            assert!(
                result_unix.is_err(),
                "Expected '{}' to be rejected (Unix)",
                path
            );
            assert!(
                result_windows.is_err(),
                "Expected '{}' to be rejected (Windows)",
                path
            );
        }
    }

    #[test]
    fn validate_install_path_rejects_relative_paths() {
        assert!(validate_install_path("relative/path", RemoteOs::Unix).is_err());
        assert!(validate_install_path("./path", RemoteOs::Unix).is_err());
        assert!(validate_install_path("../path", RemoteOs::Unix).is_err());
    }

    #[test]
    fn validate_install_path_rejects_suspicious_directories() {
        assert!(validate_install_path("/tmp/llama-monitor", RemoteOs::Unix).is_err());
        assert!(validate_install_path("/var/llama-monitor", RemoteOs::Unix).is_err());
        assert!(validate_install_path("/etc/llama-monitor", RemoteOs::Unix).is_err());
        assert!(validate_install_path(r#"C:\Windows\llama-monitor"#, RemoteOs::Windows).is_err());
    }

    #[test]
    fn validate_install_path_accepts_valid_paths() {
        // Unix paths
        let unix_paths = [
            "/opt/llama-monitor",
            "/usr/local/bin/llama-monitor",
            "/home/user/.local/bin/llama-monitor",
            "/Applications/Llama Monitor/llama-monitor",
            "~/.config/llama-monitor/bin/llama-monitor", // default Unix/macOS path
        ];

        for path in unix_paths {
            let result = validate_install_path(path, RemoteOs::Unix);
            assert!(result.is_ok(), "Expected '{}' to be accepted", path);
        }

        // Windows paths — test with RemoteOs::Windows regardless of build platform
        let windows_paths = [
            r#"C:\Program Files\llama-monitor"#,
            r#"C:\Users\user\.llama-monitor"#,
            r#"%APPDATA%\llama-monitor\bin\llama-monitor.exe"#, // default Windows path
        ];
        for path in windows_paths {
            let result = validate_install_path(path, RemoteOs::Windows);
            assert!(result.is_ok(), "Expected '{}' to be accepted", path);
        }
    }

    #[test]
    fn windows_task_result_identifies_missing_agent_executable() {
        assert_eq!(
            install::describe_windows_task_result(2_147_942_402),
            Some("agent executable not found at the scheduled task path")
        );
    }

    #[test]
    fn windows_task_result_keeps_success_silent() {
        assert_eq!(install::describe_windows_task_result(0), None);
    }

    #[test]
    fn agent_url_candidates_prefer_https_for_legacy_http_urls() {
        let candidates = agent_url_candidates("http://192.168.2.16:7779");
        assert_eq!(
            candidates,
            vec![
                "https://192.168.2.16:7779".to_string(),
                "http://192.168.2.16:7779".to_string()
            ]
        );
    }

    #[test]
    fn master_request_activity_is_inactive_without_request() {
        let last_request = AtomicU64::new(0);
        assert!(!master_request_is_recent(&last_request, 1_000));
    }

    #[test]
    fn master_request_activity_is_active_at_and_before_timeout() {
        let last_request = AtomicU64::new(1_000);
        let timeout = AGENT_MASTER_IDLE_TIMEOUT.as_secs();
        assert!(master_request_is_recent(&last_request, 1_000));
        assert!(master_request_is_recent(&last_request, 1_000 + timeout));
        assert!(!master_request_is_recent(&last_request, 1_001 + timeout));
    }

    #[test]
    fn marking_master_request_activates_polling() {
        let last_request = AtomicU64::new(0);
        mark_master_request(&last_request);
        assert!(master_request_is_recent(
            &last_request,
            unix_timestamp_seconds()
        ));
    }

    #[test]
    fn master_request_activity_handles_clock_skew_safely() {
        let last_request = AtomicU64::new(2_000);
        assert!(master_request_is_recent(&last_request, 1_000));
    }
}
