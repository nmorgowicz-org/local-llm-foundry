use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use aes_gcm::aead::array::Array;
use aes_gcm::{
    Aes256Gcm,
    aead::{Aead, KeyInit},
};
use rand::TryRng;
use rand::rngs::SysRng;

use crate::cli::AppArgs;
use crate::paths::AppPaths;

/// Restrict file permissions to owner-only.
/// Unix: sets mode 0600. Windows: uses icacls to remove inherited ACEs and
/// grant only the current user Full Control.
pub(crate) fn harden_file_permissions(path: &std::path::Path) {
    if !path.exists() {
        return;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(mut perms) = std::fs::metadata(path).map(|m| m.permissions()) {
            perms.set_mode(0o600);
            let _ = std::fs::set_permissions(path, perms);
        }
    }
    #[cfg(all(windows, not(test)))]
    {
        // Build the user identity string. Prefer USERDOMAIN\USERNAME for domain
        // accounts; fall back to USERNAME alone for local accounts.
        let user = match (std::env::var("USERDOMAIN"), std::env::var("USERNAME")) {
            (Ok(domain), Ok(name))
                if !domain.is_empty()
                    && domain != std::env::var("COMPUTERNAME").unwrap_or_default() =>
            {
                format!("{domain}\\{name}")
            }
            (_, Ok(name)) if !name.is_empty() => name,
            _ => {
                eprintln!(
                    "[warn] harden_file_permissions: USERNAME not set; cannot harden {}",
                    path.display()
                );
                return;
            }
        };

        let path_str = path.to_string_lossy();
        // /inheritance:r  — remove inherited ACEs
        // /grant:r        — replace (not add) explicit grants
        // (F)             — Full Control
        let mut icacls_cmd = std::process::Command::new("icacls");
        crate::platform::no_window(&mut icacls_cmd);
        let result = icacls_cmd
            .args([
                path_str.as_ref(),
                "/inheritance:r",
                "/grant:r",
                &format!("{user}:(F)"),
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();

        if !result.is_ok_and(|s| s.success()) {
            eprintln!(
                "[warn] harden_file_permissions: icacls failed for {}; \
                permissions may be too permissive",
                path.display()
            );
        }
    }

    // Unit tests use disposable temporary files. Applying ACL rewrites to each
    // test artifact makes the temp directory non-reopenable on some Windows
    // runners; production callers still take the hardened path above.
    #[cfg(all(windows, test))]
    {
        let _ = path;
    }
}

const ENCRYPTED_PREFIX: &str = "enc:";

use once_cell::sync::OnceCell;

static ENCRYPTION_KEY_CELL: OnceCell<[u8; 32]> = OnceCell::new();

/// Derive a 256-bit key from a secret using HKDF-SHA-256.
fn derive_key(secret: &[u8]) -> [u8; 32] {
    use hkdf::Hkdf;
    use sha2::Sha256;
    let hk = Hkdf::<Sha256>::new(None, secret);
    let mut key = [0u8; 32];
    hk.expand(b"llama-monitor-encryption-key", &mut key)
        .expect("valid HKDF output length");
    key
}

/// Initialize the encryption key at startup.
///
/// Priority:
/// 1) LOCAL_LLM_FOUNDRY_ENCRYPTION_KEY (if set and non-empty), then the
///    legacy LLAMA_MONITOR_ENCRYPTION_KEY alias.
/// 2) Auto-generated key stored in config_dir/encryption-key.
///
/// This ensures encryption is always enabled and fully automatic.
pub fn init_encryption_key(config_dir: &std::path::Path) -> Result<(), &'static str> {
    if ENCRYPTION_KEY_CELL.get().is_some() {
        return Ok(());
    }

    // 1) Use env var if provided
    let canonical_secret = std::env::var("LOCAL_LLM_FOUNDRY_ENCRYPTION_KEY").ok();
    let legacy_secret = std::env::var("LLAMA_MONITOR_ENCRYPTION_KEY").ok();
    let secret = match select_encryption_env(canonical_secret.as_deref(), legacy_secret.as_deref())
    {
        Ok(secret) => secret,
        Err(()) => {
            eprintln!(
                "[error] LOCAL_LLM_FOUNDRY_ENCRYPTION_KEY and LLAMA_MONITOR_ENCRYPTION_KEY differ; unset one variable and retry"
            );
            return Err("conflicting encryption environment aliases");
        }
    };
    if let Some(secret) = secret {
        let key = derive_key(secret.as_bytes());
        let _ = ENCRYPTION_KEY_CELL.set(key);
        eprintln!("[info] Using environment-provided encryption key for at-rest encryption.");
        return Ok(());
    }

    let key_file = config_dir.join("encryption-key");

    // 2) Try to load existing auto-generated key
    if key_file.exists()
        && let Ok(raw) = std::fs::read(&key_file)
        && raw.len() == 32
    {
        let mut key = [0u8; 32];
        key.copy_from_slice(&raw);
        let _ = ENCRYPTION_KEY_CELL.set(key);
        eprintln!("[info] Loaded auto-generated encryption key from {key_file:?}.");
        return Ok(());
    }

    // 3) Generate and persist a new key
    let mut key = [0u8; 32];
    SysRng.try_fill_bytes(&mut key).expect("SysRng failed");
    let _ = std::fs::create_dir_all(config_dir);
    if std::fs::write(&key_file, key).is_ok() {
        harden_file_permissions(&key_file);
        eprintln!("[info] Generated and saved encryption key to {key_file:?}.");
    } else {
        eprintln!(
            "[warn] Failed to write encryption key to {key_file:?}; \
             continuing with in-memory key only."
        );
    }
    let _ = ENCRYPTION_KEY_CELL.set(key);
    Ok(())
}

/// Select a valid encryption environment alias without exposing its value.
/// Empty/short aliases are ignored so a valid legacy value is not shadowed by
/// an unusable canonical value. Unequal valid aliases fail closed.
fn select_encryption_env<'a>(
    canonical: Option<&'a str>,
    legacy: Option<&'a str>,
) -> Result<Option<&'a str>, ()> {
    let canonical = canonical.filter(|value| !value.is_empty() && value.len() >= 16);
    let legacy = legacy.filter(|value| !value.is_empty() && value.len() >= 16);
    if canonical.is_some() && legacy.is_some() && canonical != legacy {
        return Err(());
    }
    Ok(canonical.or(legacy))
}

/// Get the active encryption key, or None if initialization failed.
fn encryption_key() -> Option<[u8; 32]> {
    ENCRYPTION_KEY_CELL.get().copied()
}

/// Generate a random 12-byte nonce.
fn random_nonce() -> [u8; 12] {
    let mut buf = [0u8; 12];
    SysRng.try_fill_bytes(&mut buf).expect("SysRng failed");
    buf
}

/// Encrypt a plaintext value using AES-256-GCM if a key is configured.
/// Returns "enc:<base64(nonce || ciphertext)>" on success, or the original plaintext if no key.
pub(crate) fn encrypt_value(plaintext: &str) -> String {
    let key_bytes = match encryption_key() {
        Some(k) => k,
        None => return plaintext.to_string(),
    };

    let key = Array::<u8, _>::from(key_bytes);
    let cipher = Aes256Gcm::new(&key);
    let nonce = aes_gcm::Nonce::from(random_nonce());

    let ct = match cipher.encrypt(&nonce, plaintext.as_ref()) {
        Ok(c) => c,
        Err(_) => return plaintext.to_string(),
    };

    // Prepend nonce to ciphertext so we can recover it during decryption.
    let mut payload = Vec::with_capacity(12 + ct.len());
    payload.extend_from_slice(&nonce);
    payload.extend_from_slice(&ct);

    let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &payload);
    format!("{ENCRYPTED_PREFIX}{b64}")
}

/// Decrypt a value if it appears encrypted.
/// If it starts with "enc:", attempts AES-256-GCM decryption using the active key.
/// If no key or decryption fails, logs a warning and falls back to the original value.
pub(crate) fn decrypt_value(ciphertext: &str) -> String {
    if !ciphertext.starts_with(ENCRYPTED_PREFIX) {
        return ciphertext.to_string();
    }

    let key_bytes = match encryption_key() {
        Some(k) => k,
        None => {
            eprintln!(
                "[warn] Decryption requested but no encryption key available; \
                returning raw encrypted blob as-is."
            );
            return ciphertext.to_string();
        }
    };

    let b64_part = &ciphertext[ENCRYPTED_PREFIX.len()..];
    let payload = match base64::Engine::decode(&base64::engine::general_purpose::STANDARD, b64_part)
    {
        Ok(v) => v,
        Err(_) => {
            eprintln!("[warn] Failed to decode encrypted value (bad base64)");
            return ciphertext.to_string();
        }
    };

    if payload.len() < 12 {
        eprintln!("[warn] Encrypted value too short to contain nonce");
        return ciphertext.to_string();
    }

    let (nonce_bytes, ct_bytes) = payload.split_at(12);
    let nonce = match aes_gcm::Nonce::try_from(nonce_bytes) {
        Ok(n) => n,
        Err(_) => {
            eprintln!("[warn] Encrypted value has invalid nonce length");
            return ciphertext.to_string();
        }
    };
    let key = Array::<u8, _>::from(key_bytes);
    let cipher = Aes256Gcm::new(&key);

    let pt = match cipher.decrypt(&nonce, ct_bytes) {
        Ok(pt) => pt,
        Err(_) => {
            eprintln!("[warn] Decryption failed (bad key or corrupted data); returning raw value");
            return ciphertext.to_string();
        }
    };

    match String::from_utf8(pt) {
        Ok(s) => s,
        Err(_) => {
            eprintln!("[warn] Decryption produced invalid UTF-8; returning raw value");
            ciphertext.to_string()
        }
    }
}

/// TLS operating mode.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum TlsMode {
    #[default]
    None,
    SelfSigned,
    Custom,
    Acme,
}

/// ACME-specific configuration (Let's Encrypt / lego).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct AcmeConfig {
    pub enabled: bool,
    pub fqdn: String,
    pub email: String,
    pub environment: String,
    pub dns_provider: String,
    pub dns_config: HashMap<String, String>,
    pub validation_delay: u64,
    pub last_renewal: Option<String>,
    pub cert_path: Option<PathBuf>,
    pub key_path: Option<PathBuf>,
}

impl AcmeConfig {
    pub fn is_valid(&self) -> bool {
        self.enabled
            && !self.fqdn.is_empty()
            && (self.environment == "staging" || self.environment == "production")
            && !self.dns_provider.is_empty()
            && !self.dns_config.is_empty()
    }
}

/// TLS configuration persisted to tls-config.json.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct TLSConfig {
    pub mode: TlsMode,
    pub custom_cert_path: Option<PathBuf>,
    pub custom_key_path: Option<PathBuf>,
    pub acme: AcmeConfig,
}

/// Persisted dashboard auth configuration stored separately from UI settings.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct DashboardAuthConfig {
    pub basic_enabled: bool,
    pub form_enabled: bool,
    pub username: String,
    pub password_hash: String,
}

impl DashboardAuthConfig {
    pub fn is_usable(&self) -> bool {
        (self.basic_enabled || self.form_enabled)
            && !self.username.trim().is_empty()
            && !self.password_hash.trim().is_empty()
    }
}

impl Default for TLSConfig {
    fn default() -> Self {
        Self {
            mode: TlsMode::None,
            custom_cert_path: None,
            custom_key_path: None,
            acme: AcmeConfig::default(),
        }
    }
}

/// Sanitize a TLSConfig so that an Acme mode with missing/invalid fields
/// falls back gracefully instead of panicking or blocking startup.
pub fn sanitize_tls_config(cfg: TLSConfig) -> TLSConfig {
    let mut cfg = cfg;
    if cfg.mode == TlsMode::Acme && !cfg.acme.is_valid() {
        cfg.mode = TlsMode::None;
    }
    cfg
}

/// Load TLSConfig from tls-config.json; on any error, return default.
pub fn load_tls_config(config_dir: &std::path::Path) -> TLSConfig {
    let path = config_dir.join("tls-config.json");
    if !path.exists() {
        return TLSConfig::default();
    }
    let Ok(contents) = fs::read_to_string(&path) else {
        eprintln!("[warn] Failed to read tls-config.json, using defaults");
        return TLSConfig::default();
    };
    let Ok(mut cfg) = serde_json::from_str::<TLSConfig>(&contents) else {
        eprintln!("[warn] Invalid tls-config.json, using defaults");
        return TLSConfig::default();
    };

    // Decrypt ACME dns_config values
    cfg.acme.dns_config = cfg
        .acme
        .dns_config
        .into_iter()
        .map(|(k, v)| (k, decrypt_value(&v)))
        .collect();

    sanitize_tls_config(cfg)
}

/// Persist TLSConfig to tls-config.json (atomic write).
pub fn save_tls_config(config_dir: &std::path::Path, cfg: &TLSConfig) -> std::io::Result<()> {
    let path = config_dir.join("tls-config.json");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    // Encrypt ACME dns_config values before writing
    let mut to_save = cfg.clone();
    to_save.acme.dns_config = to_save
        .acme
        .dns_config
        .into_iter()
        .map(|(k, v)| (k, encrypt_value(&v)))
        .collect();

    let tmp = path.with_extension("json.tmp");
    let json = serde_json::to_string_pretty(&to_save)?;
    fs::write(&tmp, json)?;
    fs::rename(&tmp, &path)?;
    harden_file_permissions(&path);
    Ok(())
}

pub fn load_auth_config(config_dir: &std::path::Path) -> DashboardAuthConfig {
    let path = config_dir.join("auth-config.json");
    if !path.exists() {
        return DashboardAuthConfig::default();
    }
    let Ok(contents) = fs::read_to_string(&path) else {
        eprintln!("[warn] Failed to read auth-config.json, using defaults");
        return DashboardAuthConfig::default();
    };
    let Ok(cfg) = serde_json::from_str::<DashboardAuthConfig>(&contents) else {
        eprintln!("[warn] Invalid auth-config.json, using defaults");
        return DashboardAuthConfig::default();
    };
    cfg
}

pub fn save_auth_config(
    config_dir: &std::path::Path,
    cfg: &DashboardAuthConfig,
) -> std::io::Result<()> {
    let path = config_dir.join("auth-config.json");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let tmp = path.with_extension("json.tmp");
    let json = serde_json::to_string_pretty(cfg)?;
    fs::write(&tmp, json)?;
    fs::rename(&tmp, &path)?;
    harden_file_permissions(&path);
    Ok(())
}

pub fn clear_auth_config(config_dir: &std::path::Path) -> std::io::Result<bool> {
    let path = config_dir.join("auth-config.json");
    if !path.exists() {
        return Ok(false);
    }
    fs::remove_file(path)?;
    Ok(true)
}

/// Card presentation mode for preset bundles (architecture invariant 16).
///
/// The bundle schema version and the card presentation are independently
/// controllable. Migration to v6 is forward-only with no downgrade, so a
/// defective card must be switchable off without touching preset files.
/// `Legacy` forces one-artifact card rendering even when v6 bundles exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PresetBundleUiMode {
    /// Render each bundle as one compact card with a Configure drawer.
    #[default]
    Bundled,
    /// Render every preset through the legacy one-artifact adapter.
    Legacy,
}

impl PresetBundleUiMode {
    /// The only representation exposed to the UI: a closed enum, never a raw
    /// environment string.
    pub fn to_wire(self) -> &'static str {
        match self {
            PresetBundleUiMode::Bundled => "bundled",
            PresetBundleUiMode::Legacy => "legacy",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "bundled" => Some(PresetBundleUiMode::Bundled),
            "legacy" => Some(PresetBundleUiMode::Legacy),
            _ => None,
        }
    }
}

/// Read the render kill-switch from the environment. Canonical
/// `LOCAL_LLM_FOUNDRY_PRESET_BUNDLE_UI`, with the legacy
/// `LLAMA_MONITOR_PRESET_BUNDLE_UI` alias named by architecture invariant 16.
fn resolve_preset_bundle_ui() -> PresetBundleUiMode {
    let canonical = std::env::var("LOCAL_LLM_FOUNDRY_PRESET_BUNDLE_UI").ok();
    let legacy = std::env::var("LLAMA_MONITOR_PRESET_BUNDLE_UI").ok();
    select_preset_bundle_ui(canonical.as_deref(), legacy.as_deref())
}

/// Unparseable values and disagreeing aliases both fall back to the default so
/// a typo cannot leave the grid in an unintended rendering mode.
fn select_preset_bundle_ui(canonical: Option<&str>, legacy: Option<&str>) -> PresetBundleUiMode {
    let canonical = canonical.and_then(PresetBundleUiMode::parse);
    let legacy = legacy.and_then(PresetBundleUiMode::parse);
    match (canonical, legacy) {
        (Some(canonical), Some(legacy)) if canonical != legacy => {
            eprintln!(
                "[warn] LOCAL_LLM_FOUNDRY_PRESET_BUNDLE_UI and LLAMA_MONITOR_PRESET_BUNDLE_UI \
                 disagree; using the default bundled card"
            );
            PresetBundleUiMode::default()
        }
        (Some(mode), _) | (None, Some(mode)) => mode,
        (None, None) => PresetBundleUiMode::default(),
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AppConfig {
    pub app_paths: AppPaths,
    pub config_dir: PathBuf,
    /// Explicit disposable migration roots used only by native qualification.
    pub migration_test_root: Option<PathBuf>,
    pub llama_server_path: PathBuf,
    /// Optional separately-built llama-fit-params estimate probe.
    pub llama_fit_params_path: Option<PathBuf>,
    pub llama_server_cwd: PathBuf,
    pub port: u16,
    pub gpu_backend: String,
    pub llama_poll_interval: u64,
    /// Render kill-switch for the preset-bundle launch card.
    pub preset_bundle_ui: PresetBundleUiMode,
    pub models_dir: Option<PathBuf>,
    pub presets_file: PathBuf,
    pub templates_file: PathBuf,
    pub gpu_env_file: PathBuf,
    pub gpu_arch_override: Option<String>,
    pub gpu_devices_override: Option<String>,
    pub ui_settings_file: PathBuf,
    pub auth_config_file: PathBuf,
    pub sessions_file: PathBuf,
    pub ssh_known_hosts_file: PathBuf,
    pub lhm_disabled_file: PathBuf,
    pub agent_host: String,
    pub agent_port: u16,
    pub agent_token: Option<String>,
    pub remote_agent_url: Option<String>,
    pub remote_agent_token: Option<String>,
    pub remote_agent_ssh_autostart: bool,
    pub remote_agent_ssh_target: Option<String>,
    pub remote_agent_ssh_command: Option<String>,
    pub tls_config: TLSConfig,
    // Live token stores — updated in-memory on rotation without requiring a restart.
    // Arc<RwLock> so all Arc<AppConfig> clones share the same backing store.
    live_api_token_store: std::sync::Arc<std::sync::RwLock<Option<String>>>,
    live_db_admin_token_store: std::sync::Arc<std::sync::RwLock<Option<String>>>,

    // Spawn V2: additional directories
    pub binaries_dir: PathBuf,
    pub default_models_dir: PathBuf,
    pub scripts_dir: PathBuf,
    pub certs_dir: PathBuf,
}

impl AppConfig {
    pub fn from_args(args: AppArgs) -> Self {
        Self::from_args_inner(args, true)
    }

    /// Resolve paths and CLI state without reading or creating application
    /// resources. Migration/status commands use this before selecting a root.
    pub fn from_args_pure(args: AppArgs) -> Self {
        Self::from_args_inner(args, false)
    }

    fn from_args_inner(args: AppArgs, initialize_resources: bool) -> Self {
        let default_server_cwd = PathBuf::from(".");

        let app_paths = AppPaths::from_root(AppPaths::resolve_root(args.config_dir.clone()));
        let config_dir = app_paths.root.clone();

        // Default binary location: <config_dir>/bin/llama-server. During the
        // restartable root migration, reuse a healthy managed binary from the
        // other application root when the selected root has no binary yet.
        // Pure config construction must remain filesystem-free for migration
        // planning and status commands.
        let default_server_path = if initialize_resources {
            app_paths.default_llama_server_path()
        } else {
            let binary_name = if cfg!(windows) {
                "llama-server.exe"
            } else {
                "llama-server"
            };
            app_paths.bin_dir().join(binary_name)
        };

        let presets_file = args
            .presets_file
            .unwrap_or_else(|| app_paths.presets_file());

        Self {
            app_paths: app_paths.clone(),
            config_dir: config_dir.clone(),
            migration_test_root: args.migration_test_root,
            llama_server_path: args.llama_server_path.unwrap_or(default_server_path),
            llama_fit_params_path: None,
            llama_server_cwd: args.llama_server_cwd.unwrap_or(default_server_cwd),
            port: args.port,
            gpu_backend: args.gpu_backend,
            models_dir: args.models_dir,
            presets_file,
            templates_file: app_paths.templates_file(),
            gpu_env_file: app_paths.gpu_env_file(),
            gpu_arch_override: args.gpu_arch,
            gpu_devices_override: args.gpu_devices,
            ui_settings_file: app_paths.ui_settings_file(),
            auth_config_file: app_paths.auth_config_file(),
            sessions_file: args
                .sessions_file
                .unwrap_or_else(|| app_paths.sessions_file()),
            ssh_known_hosts_file: app_paths.ssh_known_hosts_file(),
            llama_poll_interval: args.llama_poll_interval,
            preset_bundle_ui: resolve_preset_bundle_ui(),
            lhm_disabled_file: app_paths.lhm_disabled_file(),
            agent_host: args.agent_host,
            agent_port: args.agent_port,
            agent_token: args.agent_token,
            remote_agent_url: args.remote_agent_url,
            remote_agent_token: args.remote_agent_token,
            remote_agent_ssh_autostart: args.remote_agent_ssh_autostart,
            remote_agent_ssh_target: args.remote_agent_ssh_target,
            remote_agent_ssh_command: args.remote_agent_ssh_command,
            tls_config: if initialize_resources {
                load_tls_config(&config_dir)
            } else {
                TLSConfig::default()
            },
            live_api_token_store: std::sync::Arc::new(std::sync::RwLock::new(
                initialize_resources
                    .then(|| ensure_api_token(&config_dir))
                    .flatten(),
            )),
            live_db_admin_token_store: std::sync::Arc::new(std::sync::RwLock::new(
                initialize_resources
                    .then(|| ensure_db_admin_token(&config_dir))
                    .flatten(),
            )),

            // Spawn V2: additional directories (backward-compatible defaults)
            binaries_dir: app_paths.binaries_dir(),
            default_models_dir: app_paths.models_dir(),
            scripts_dir: app_paths.scripts_dir(),
            certs_dir: app_paths.certs_dir(),
        }
    }

    /// Read the current live API token (updated on rotation).
    pub fn live_api_token(&self) -> Option<String> {
        self.live_api_token_store.read().unwrap().clone()
    }

    /// Read the current live DB admin token (updated on rotation).
    pub fn live_db_admin_token(&self) -> Option<String> {
        self.live_db_admin_token_store.read().unwrap().clone()
    }

    /// Update the live API token after rotation.
    pub fn update_live_api_token(&self, token: String) {
        *self.live_api_token_store.write().unwrap() = Some(token);
    }

    /// Update the live DB admin token after rotation.
    pub fn update_live_db_admin_token(&self, token: String) {
        *self.live_db_admin_token_store.write().unwrap() = Some(token);
    }

    /// Construct an `AppConfig` for unit/integration tests without reading from disk.
    #[allow(dead_code)]
    pub fn for_test(api_token: Option<String>, db_admin_token: Option<String>) -> Self {
        Self {
            app_paths: AppPaths::from_root(std::path::PathBuf::from("/tmp/llama-monitor-test")),
            config_dir: std::path::PathBuf::from("/tmp/llama-monitor-test"),
            migration_test_root: None,
            llama_server_path: std::path::PathBuf::from("llama-server"),
            llama_fit_params_path: None,
            llama_server_cwd: std::path::PathBuf::from("."),
            port: 8001,
            gpu_backend: String::new(),
            llama_poll_interval: 1,
            preset_bundle_ui: PresetBundleUiMode::default(),
            models_dir: None,
            presets_file: std::path::PathBuf::new(),
            templates_file: std::path::PathBuf::new(),
            gpu_env_file: std::path::PathBuf::new(),
            gpu_arch_override: None,
            gpu_devices_override: None,
            ui_settings_file: std::path::PathBuf::new(),
            auth_config_file: std::path::PathBuf::new(),
            sessions_file: std::path::PathBuf::new(),
            ssh_known_hosts_file: std::path::PathBuf::new(),
            lhm_disabled_file: std::path::PathBuf::new(),
            agent_host: "127.0.0.1".to_string(),
            agent_port: 7777,
            agent_token: None,
            remote_agent_url: None,
            remote_agent_token: None,
            remote_agent_ssh_autostart: false,
            remote_agent_ssh_target: None,
            remote_agent_ssh_command: None,
            tls_config: TLSConfig::default(),
            live_api_token_store: std::sync::Arc::new(std::sync::RwLock::new(api_token)),
            live_db_admin_token_store: std::sync::Arc::new(std::sync::RwLock::new(db_admin_token)),

            // Spawn V2: additional directories (test defaults)
            binaries_dir: std::path::PathBuf::from("/tmp/llama-monitor-test/binaries"),
            default_models_dir: std::path::PathBuf::from("/tmp/llama-monitor-test/models"),
            scripts_dir: std::path::PathBuf::from("/tmp/llama-monitor-test/scripts"),
            certs_dir: std::path::PathBuf::from("/tmp/llama-monitor-test/certs"),
        }
    }
}

fn ensure_db_admin_token(config_dir: &PathBuf) -> Option<String> {
    let token_file = config_dir.join("db-admin-token");

    // Try to read existing token (may be encrypted)
    if let Ok(content) = fs::read_to_string(&token_file) {
        let trimmed = content.trim().to_string();
        if !trimmed.is_empty() {
            let token = decrypt_value(&trimmed);
            return Some(token);
        }
    }

    // Generate new token
    let token = generate_random_token();
    let _ = fs::create_dir_all(config_dir);
    let stored = encrypt_value(&token);
    if fs::write(&token_file, &stored).is_ok() {
        eprintln!("[config] Generated db-admin-token");
    }
    Some(token)
}

fn ensure_api_token(config_dir: &PathBuf) -> Option<String> {
    let token_file = config_dir.join("api-token");

    if let Ok(content) = fs::read_to_string(&token_file) {
        let trimmed = content.trim().to_string();
        if !trimmed.is_empty() {
            let token = decrypt_value(&trimmed);
            return Some(token);
        }
    }

    let token = generate_random_token();
    let _ = fs::create_dir_all(config_dir);
    let stored = encrypt_value(&token);
    if fs::write(&token_file, &stored).is_ok() {
        eprintln!("[config] Generated api-token");
    }
    Some(token)
}

pub(crate) fn generate_random_token() -> String {
    let mut buf = [0u8; 16];
    SysRng.try_fill_bytes(&mut buf).expect("SysRng failed");
    let value = u128::from_be_bytes(buf);
    format!("{value:x}")
}

#[cfg(test)]
mod path_resolution_tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn pure_resolution_does_not_create_a_root_or_tokens() {
        let root = std::env::temp_dir().join(format!(
            "local-llm-foundry-pure-resolution-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let args = AppArgs::parse_from([
            "llama-monitor",
            "--config-dir",
            root.to_str().expect("temp path is UTF-8"),
        ]);
        let config = AppConfig::from_args_pure(args);
        assert_eq!(config.config_dir, root);
        assert!(!config.config_dir.exists());
        assert!(config.live_api_token().is_none());
        assert!(config.live_db_admin_token().is_none());
    }

    #[test]
    fn preset_bundle_ui_defaults_to_bundled_and_fails_closed() {
        assert_eq!(
            select_preset_bundle_ui(None, None),
            PresetBundleUiMode::Bundled
        );
        assert_eq!(
            select_preset_bundle_ui(Some("legacy"), None),
            PresetBundleUiMode::Legacy
        );
        // The legacy alias named by architecture invariant 16 still works alone.
        assert_eq!(
            select_preset_bundle_ui(None, Some("LEGACY")),
            PresetBundleUiMode::Legacy
        );
        // A typo must not silently pick the other mode.
        assert_eq!(
            select_preset_bundle_ui(Some("bundle"), None),
            PresetBundleUiMode::Bundled
        );
        // Disagreeing aliases fall back to the default rather than guessing.
        assert_eq!(
            select_preset_bundle_ui(Some("legacy"), Some("bundled")),
            PresetBundleUiMode::Bundled
        );
        assert_eq!(PresetBundleUiMode::Legacy.to_wire(), "legacy");
        assert_eq!(PresetBundleUiMode::Bundled.to_wire(), "bundled");
    }

    #[test]
    fn encryption_alias_precedence_is_secret_safe() {
        let canonical = "canonical-secret-value-1234";
        let legacy = "legacy-secret-value-12345";
        assert_eq!(
            select_encryption_env(Some(canonical), None),
            Ok(Some(canonical))
        );
        assert_eq!(select_encryption_env(None, Some(legacy)), Ok(Some(legacy)));
        assert_eq!(
            select_encryption_env(Some(canonical), Some(canonical)),
            Ok(Some(canonical))
        );
        assert_eq!(
            select_encryption_env(Some(canonical), Some(legacy)),
            Err(())
        );
        assert_eq!(
            select_encryption_env(Some("short"), Some(legacy)),
            Ok(Some(legacy))
        );
        assert_eq!(
            select_encryption_env(Some("ユニコード暗号化値123456"), None),
            Ok(Some("ユニコード暗号化値123456"))
        );
    }
}
