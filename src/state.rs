use std::collections::{BTreeMap, HashMap, VecDeque};
use std::net::{IpAddr, SocketAddr, UdpSocket};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8};
use std::sync::{Arc, Mutex};

use crate::chat_storage::ChatStorage;
use crate::config::{TLSConfig, decrypt_value, encrypt_value};
use crate::gpu::GpuMetrics;
use crate::gpu::env::GpuEnv;
use crate::inference::llama_cpp::ServerConfig;
use crate::llama::metrics::LlamaMetrics;
use crate::models::DiscoveredModel;
use crate::presets;
use crate::presets::ModelPreset;
use std::sync::atomic::AtomicU64;

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ModelTags {
    #[serde(default)]
    pub tags: BTreeMap<String, Vec<String>>,
}

// Legacy format when model-tags.json was a simple array/object with a "tags" field.
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyModelTags {
    #[serde(default)]
    tags: Vec<String>,
}

impl ModelTags {
    pub fn load(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(content) => {
                // Try legacy shape: { "tags": [...] }
                if let Ok(legacy) = serde_json::from_str::<LegacyModelTags>(&content) {
                    if !legacy.tags.is_empty() {
                        eprintln!("[info] ModelTags: migrating legacy format in {:?}", path);
                    }
                    let tags = Self::default();
                    if let Err(e) = tags.save(path) {
                        eprintln!("[warn] ModelTags: failed to migrate {:?}: {e}", path);
                    }
                    return tags;
                }

                // Older versions loaded a direct map without the "tags" wrapper.
                if let Ok(tags) = serde_json::from_str::<BTreeMap<String, Vec<String>>>(&content) {
                    let tags = Self { tags };
                    if let Err(e) = tags.save(path) {
                        eprintln!("[warn] ModelTags: failed to migrate {:?}: {e}", path);
                    }
                    return tags;
                }

                // Current format: { "tags": { "model-id": ["tag1", "tag2"] } }
                if let Ok(tags) = serde_json::from_str::<Self>(&content) {
                    return tags;
                }

                eprintln!(
                    "[warn] ModelTags: corrupted JSON in {:?}, loading empty tags",
                    path
                );
                Self::default()
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let tags = Self::default();
                if !path.as_os_str().is_empty()
                    && let Err(e) = tags.save(path)
                {
                    eprintln!(
                        "[warn] ModelTags: failed to initialize missing file {:?}: {e}",
                        path
                    );
                }
                tags
            }
            Err(e) => {
                eprintln!(
                    "[warn] ModelTags: could not read {:?}, loading empty tags (error: {})",
                    path, e
                );
                Self::default()
            }
        }
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("json.tmp");
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&tmp, json)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }
}
use crate::system::SystemMetrics;

fn sensor_bridge_setup_available() -> bool {
    #[cfg(target_os = "windows")]
    {
        crate::lhm::is_sensor_bridge_available()
    }
    #[cfg(not(target_os = "windows"))]
    {
        false
    }
}

impl crate::inference::supervisor::BackendObserver for AppState {
    fn on_log_line(&self, line: &str) {
        self.push_log(line.to_string());
    }
    fn on_crash(&self, exit_status: std::process::ExitStatus, tail: Vec<String>) {
        let reason = if exit_status.success() {
            "Local inference backend exited normally".to_string()
        } else if let Some(code) = exit_status.code() {
            match code {
                137 | 9 => format!(
                    "Local inference backend was terminated (code {}). This often means out-of-memory (OOM). Try a smaller context or a different model.",
                    code
                ),
                _ => format!(
                    "Local inference backend exited unexpectedly (code {}). Check the server logs for details.",
                    code
                ),
            }
        } else {
            "Local inference backend was killed by an external signal.".to_string()
        };

        let tail_text = if tail.is_empty() {
            "\nNo backend output captured before it was killed.".to_string()
        } else {
            format!(
                "\nLast relevant backend log lines:\n{}",
                tail.iter()
                    .map(|l| format!("  {l}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        };

        let reason_display = if exit_status.success()
            || exit_status.code() == Some(137)
            || exit_status.code() == Some(9)
        {
            reason
        } else {
            format!("{}{}", reason, tail_text)
        };

        self.push_log(format!("[monitor] {}", reason_display));
        let active_id = {
            let aid = self.active_session_id.lock().unwrap();
            aid.clone()
        };
        if !active_id.is_empty() {
            use crate::state::SessionStatus;
            self.update_session_status(&active_id, SessionStatus::Error(reason_display));
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, Default)]
pub struct MetricsCapabilities {
    pub inference: bool,
    pub system: bool,
    pub gpu: bool,
    pub cpu_temperature: bool,
    pub memory: bool,
    pub host_metrics: bool,
    pub tray: bool,
    pub sensor_bridge_setup_available: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum EndpointKind {
    Local,
    Remote,
    Unknown,
}

#[derive(Debug, Clone, serde::Serialize, PartialEq)]
pub enum SessionKind {
    Spawn,
    Attach,
    None,
}

#[derive(Debug, Clone, serde::Serialize)]
#[allow(dead_code)]
pub enum TrayMode {
    Desktop,
    Headless,
    Failed,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum AvailabilityReason {
    Available,
    RemoteEndpoint,
    NoDisplay,
    TrayUnavailable,
    SensorUnavailable,
    BackendUnavailable,
    CommandMissing,
    PermissionDenied,
    MetricsUnreachable,
    NotApplicable,
}

const MAX_LOG_LINES: usize = 500;
const MAX_SESSIONS: usize = 10;

/// Persisted UI control-bar settings (survives page reload).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UiSettings {
    #[serde(default)]
    pub preset_id: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default)]
    pub llama_server_path: String,
    #[serde(default)]
    pub llama_server_cwd: String,
    #[serde(default)]
    pub models_dir: String,
    /// Additional directories to scan for models (beyond the primary download dir).
    /// Useful for models spread across multiple drives or folders.
    #[serde(default)]
    pub extra_models_dirs: Vec<String>,
    #[serde(default)]
    pub server_endpoint: String,
    #[serde(default = "default_llama_poll_interval")]
    pub llama_poll_interval: u64,
    #[serde(default)]
    pub remote_agent_url: String,
    #[serde(default)]
    pub remote_agent_token: String,
    #[serde(default)]
    pub remote_agent_ssh_autostart: bool,
    #[serde(default)]
    pub remote_agent_ssh_target: String,
    #[serde(default)]
    pub remote_agent_ssh_command: String,
    /// Policy appended when explicit mode (🔓) is enabled.
    /// Stored in `ui-settings.json` so it persists across restarts.
    #[serde(default)]
    pub explicit_mode_policy: String,
    #[serde(default = "default_context_card_view")]
    pub context_card_view: String,
    /// Last sort mode chosen in HF discovery, so it survives a restart instead of snapping
    /// back to the default on every visit. One of the `HF_SORT` values in `hf-browse.js`.
    #[serde(default = "default_hf_discovery_sort")]
    pub hf_discovery_sort: String,
    /// WebSocket dashboard push interval in milliseconds.
    /// Presets: 500 (Normal), 1000 (Balanced), 2000 (Battery Saver), 5000 (Slow Connection).
    #[serde(default = "default_ws_push_interval_ms")]
    pub ws_push_interval_ms: u64,
    /// Persisted chat input textarea height (CSS value, e.g. "80px").
    #[serde(default)]
    pub chat_input_height: String,
    /// Guided generation: context notes sidebar enabled
    #[serde(default = "default_true")]
    pub enabled_context_notes: bool,
    /// Guided generation: suggestions dropdown enabled
    #[serde(default = "default_true")]
    pub enabled_suggestions: bool,
    /// Guided generation: quick guide input enabled
    #[serde(default = "default_true")]
    pub enabled_quick_guide: bool,
    /// Guided generation: default sidebar width in pixels
    #[serde(default = "default_sidebar_width")]
    pub default_sidebar_width: u32,
    /// Guided generation: suggestion prompts by category
    #[serde(default = "default_suggestion_prompts")]
    pub suggestion_prompts: HashMap<String, String>,
    /// Guided generation: number of suggestions to generate
    #[serde(default = "default_suggestion_count")]
    pub suggestion_count: u32,
    /// Guided generation: context depth (number of messages to include)
    #[serde(default = "default_context_depth")]
    pub context_depth: u32,
    /// Shared chat date format preference.
    #[serde(default = "default_chat_date_format")]
    pub chat_date_format: String,
    /// Shared enter-to-send preference.
    #[serde(default = "default_true")]
    pub enter_to_send: bool,
    /// Shared context notes sidebar expanded state.
    #[serde(default)]
    pub context_notes_sidebar_expanded: bool,
    /// Shared context notes intro visibility.
    #[serde(default)]
    pub context_notes_intro_hidden: bool,
    /// Whether assistant thinking blocks should be stored in chat history and restored later.
    #[serde(default)]
    pub persist_thinking_content: bool,
    /// Shared custom suggestion categories used by the suggestions workspace.
    #[serde(default)]
    pub custom_suggestion_categories: HashMap<String, CustomSuggestionCategory>,
    /// Sleep/low-power mode settings persisted in ui-settings.json (T-044).
    #[serde(default)]
    pub sleep_mode: SleepModeConfig,
}

/// Sleep/low-power mode configuration (T-043).
///
/// Controls automatic transitions and intervals while asleep.
/// Loaded from ui-settings.json or defaults (T-044).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SleepModeConfig {
    /// If true, automatically enter sleep when all browser tabs are hidden.
    #[serde(default = "default_auto_sleep_when_all_hidden")]
    pub auto_sleep_when_all_hidden: bool,
    /// If set, automatically enter sleep after this many seconds of inactivity
    /// with no connections and no generating activity.
    #[serde(default)]
    pub auto_sleep_idle_secs: Option<u64>,
    /// GPU poller interval (seconds) while asleep.
    #[serde(default = "default_sleep_gpu_interval_secs")]
    pub sleep_gpu_interval_secs: u64,
    /// System metrics poller interval (seconds) while asleep.
    #[serde(default = "default_sleep_sys_interval_secs")]
    pub sleep_sys_interval_secs: u64,
    /// Llama metrics poller interval (seconds) while asleep.
    #[serde(default = "default_sleep_llama_interval_secs")]
    pub sleep_llama_interval_secs: u64,
    /// WebSocket broadcast interval (milliseconds) while asleep.
    #[serde(default = "default_sleep_ws_interval_ms")]
    pub sleep_ws_interval_ms: u64,
    /// WebSocket broadcast interval (milliseconds) in logs-only mode.
    /// Falls between normal and sleep: faster than sleep so logs flow,
    /// slower than normal to save power.
    #[serde(default = "default_logs_only_ws_interval_ms")]
    pub logs_only_ws_interval_ms: u64,
}

impl Default for SleepModeConfig {
    fn default() -> Self {
        Self {
            auto_sleep_when_all_hidden: default_auto_sleep_when_all_hidden(),
            auto_sleep_idle_secs: default_auto_sleep_idle_secs(),
            sleep_gpu_interval_secs: default_sleep_gpu_interval_secs(),
            sleep_sys_interval_secs: default_sleep_sys_interval_secs(),
            sleep_llama_interval_secs: default_sleep_llama_interval_secs(),
            sleep_ws_interval_ms: default_sleep_ws_interval_ms(),
            logs_only_ws_interval_ms: default_logs_only_ws_interval_ms(),
        }
    }
}

fn default_auto_sleep_when_all_hidden() -> bool {
    true
}

fn default_auto_sleep_idle_secs() -> Option<u64> {
    Some(600)
}

fn default_sleep_gpu_interval_secs() -> u64 {
    15
}

fn default_sleep_sys_interval_secs() -> u64 {
    15
}

fn default_sleep_llama_interval_secs() -> u64 {
    15
}

fn default_sleep_ws_interval_ms() -> u64 {
    10_000
}

fn default_logs_only_ws_interval_ms() -> u64 {
    2_000
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct CustomSuggestionCategory {
    #[serde(default)]
    pub prompt: String,
    #[serde(default)]
    pub explicit: bool,
}

fn default_true() -> bool {
    true
}

fn default_sidebar_width() -> u32 {
    280
}

fn default_suggestion_prompts() -> HashMap<String, String> {
    HashMap::new()
}

fn default_suggestion_count() -> u32 {
    5
}

fn default_context_depth() -> u32 {
    10
}

fn default_chat_date_format() -> String {
    "MM/DD/YY".to_string()
}

fn default_ws_push_interval_ms() -> u64 {
    500
}

/// Session mode: either spawn a new server or attach to existing
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum SessionMode {
    Spawn {
        port: u16,
        #[serde(default)]
        bind_host: Option<String>,
        #[serde(default)]
        #[serde(skip_serializing)]
        api_key: Option<String>,
    },
    Attach {
        endpoint: String,
        #[serde(default, skip_serializing)]
        api_key: Option<String>,
    },
}

impl Default for SessionMode {
    fn default() -> Self {
        SessionMode::Spawn {
            port: 8001,
            bind_host: None,
            api_key: None,
        }
    }
}

/// A server session (active or inactive)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Session {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub backend: crate::inference::InferenceBackend,
    #[serde(default)]
    pub model_identity: Option<String>,
    /// Secret-scrubbed backend-owned request used to restore direct spawns.
    #[serde(default)]
    pub launch: Option<crate::inference::launch::LocalLaunchRequest>,
    #[serde(default)]
    pub launch_requires_api_key: bool,
    #[serde(default)]
    pub mode: SessionMode,
    #[serde(default)]
    pub status: SessionStatus,
    #[serde(default)]
    pub preset_id: String,
    /// Selection identity used for a bundled launch, when applicable.
    #[serde(default)]
    pub selection_hash: Option<String>,
    #[serde(default)]
    pub created_at: u64,
    #[serde(default)]
    pub last_active: u64,
    #[serde(default)]
    pub last_connected_at: u64,
    #[serde(default)]
    pub connect_count: u64,
    #[serde(default)]
    pub last_error: Option<String>,
}

impl Session {
    fn now() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    pub fn new_spawn(
        id: String,
        name: String,
        port: u16,
        preset_id: String,
        bind_host: Option<String>,
        api_key: Option<String>,
    ) -> Self {
        Self::new_spawn_with_backend(
            id,
            name,
            port,
            preset_id,
            bind_host,
            api_key,
            crate::inference::InferenceBackend::LlamaCpp,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_spawn_with_backend(
        id: String,
        name: String,
        port: u16,
        preset_id: String,
        bind_host: Option<String>,
        api_key: Option<String>,
        backend: crate::inference::InferenceBackend,
        model_identity: Option<String>,
    ) -> Self {
        let now = Self::now();
        let launch_requires_api_key = api_key.as_ref().is_some_and(|key| !key.is_empty());
        Self {
            id,
            name,
            backend,
            model_identity,
            launch: None,
            launch_requires_api_key,
            mode: SessionMode::Spawn {
                port,
                bind_host,
                api_key,
            },
            status: SessionStatus::Stopped,
            preset_id,
            selection_hash: None,
            created_at: now,
            last_active: now,
            last_connected_at: 0,
            connect_count: 0,
            last_error: None,
        }
    }

    pub fn new_attach(id: String, name: String, endpoint: String, api_key: Option<String>) -> Self {
        let now = Self::now();
        let launch_requires_api_key = api_key.as_ref().is_some_and(|key| !key.is_empty());
        Self {
            id,
            name,
            backend: crate::inference::InferenceBackend::LlamaCpp,
            model_identity: None,
            launch: None,
            launch_requires_api_key,
            mode: SessionMode::Attach { endpoint, api_key },
            status: SessionStatus::Disconnected,
            preset_id: String::new(),
            selection_hash: None,
            created_at: now,
            last_active: now,
            last_connected_at: 0,
            connect_count: 0,
            last_error: None,
        }
    }

    pub fn is_inactive(&self) -> bool {
        matches!(
            self.status,
            SessionStatus::Stopped | SessionStatus::Disconnected
        )
    }
}

/// Session status
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Default)]
pub enum SessionStatus {
    #[default]
    Stopped,
    Running,
    Disconnected,
    Error(String),
}

fn default_port() -> u16 {
    8001
}

fn default_llama_poll_interval() -> u64 {
    1
}

fn default_context_card_view() -> String {
    "gauge".to_string()
}

/// Quant repos are re-uploaded long after they are created, so recency is the signal that
/// actually tracks "is this worth looking at" — a better opening view than raw download count.
fn default_hf_discovery_sort() -> String {
    "last_updated".to_string()
}

impl Default for UiSettings {
    fn default() -> Self {
        Self {
            preset_id: String::new(),
            port: 8001,
            llama_server_path: String::new(),
            llama_server_cwd: String::new(),
            models_dir: String::new(),
            extra_models_dirs: Vec::new(),
            server_endpoint: String::new(),
            llama_poll_interval: default_llama_poll_interval(),
            remote_agent_url: String::new(),
            remote_agent_token: String::new(),
            remote_agent_ssh_autostart: false,
            remote_agent_ssh_target: String::new(),
            remote_agent_ssh_command: String::new(),
            explicit_mode_policy: String::new(),
            context_card_view: default_context_card_view(),
            hf_discovery_sort: default_hf_discovery_sort(),
            ws_push_interval_ms: default_ws_push_interval_ms(),
            chat_input_height: String::new(),
            enabled_context_notes: default_true(),
            enabled_suggestions: default_true(),
            enabled_quick_guide: default_true(),
            default_sidebar_width: default_sidebar_width(),
            suggestion_prompts: default_suggestion_prompts(),
            suggestion_count: default_suggestion_count(),
            context_depth: default_context_depth(),
            chat_date_format: default_chat_date_format(),
            enter_to_send: default_true(),
            context_notes_sidebar_expanded: false,
            context_notes_intro_hidden: false,
            persist_thinking_content: false,
            custom_suggestion_categories: HashMap::new(),
            sleep_mode: SleepModeConfig::default(),
        }
    }
}

pub fn load_ui_settings(path: &Path) -> UiSettings {
    match std::fs::read_to_string(path) {
        Ok(contents) => match serde_json::from_str::<UiSettings>(&contents) {
            Ok(mut settings) => {
                settings.remote_agent_token = decrypt_value(&settings.remote_agent_token);
                settings
            }
            Err(e) => {
                eprintln!("[warn] Invalid UI settings file {:?}: {e}", path);
                UiSettings::default()
            }
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let settings = UiSettings::default();
            if !path.as_os_str().is_empty()
                && let Err(e) = save_ui_settings(path, &settings)
            {
                eprintln!(
                    "[warn] Failed to initialize missing UI settings file {:?}: {e}",
                    path
                );
            }
            settings
        }
        Err(e) => {
            eprintln!("[warn] Failed to read UI settings file {:?}: {e}", path);
            UiSettings::default()
        }
    }
}

pub fn save_ui_settings(path: &Path, settings: &UiSettings) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut to_save = settings.clone();
    to_save.remote_agent_token = encrypt_value(&to_save.remote_agent_token);

    let tmp = path.with_extension("json.tmp");
    let json = serde_json::to_string_pretty(&to_save)?;
    std::fs::write(&tmp, json)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Load sessions from file
pub fn load_sessions(path: &Path) -> Vec<Session> {
    match std::fs::read_to_string(path) {
        Ok(contents) => match serde_json::from_str::<Vec<Session>>(&contents) {
            Ok(sessions) => return sessions,
            Err(e) => eprintln!("[warn] Invalid sessions file {:?}: {e}", path),
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let sessions = default_sessions();
            if !path.as_os_str().is_empty()
                && let Err(e) = save_sessions(path, &sessions)
            {
                eprintln!(
                    "[warn] Failed to initialize missing sessions file {:?}: {e}",
                    path
                );
            }
            return sessions;
        }
        Err(e) => eprintln!("[warn] Failed to read sessions file {:?}: {e}", path),
    }

    default_sessions()
}

fn default_sessions() -> Vec<Session> {
    vec![Session::new_spawn(
        "default".to_string(),
        "Default Session".to_string(),
        8001,
        String::new(),
        None,
        None,
    )]
}

/// Save sessions to file
pub fn save_sessions(path: &Path, sessions: &[Session]) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("json.tmp");
    let json = serde_json::to_string_pretty(&sessions)?;
    std::fs::write(&tmp, json)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Paths for app state
#[derive(Clone)]
pub struct AppPaths {
    pub presets_path: PathBuf,
    pub templates_path: PathBuf,
    pub models_dir: Option<PathBuf>,
    pub gpu_env_path: PathBuf,
    pub ui_settings_path: PathBuf,
    pub sessions_path: PathBuf,
    pub model_tags_path: PathBuf,
}

/// Generate unique session ID
pub fn generate_session_id() -> String {
    format!(
        "session_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    )
}

#[derive(Clone)]
pub struct AppState {
    pub gpu_metrics: Arc<Mutex<BTreeMap<String, GpuMetrics>>>,
    pub llama_metrics: Arc<Mutex<LlamaMetrics>>,
    /// Latest backend-neutral inference sample. Cleared when the active backend changes.
    pub inference_metrics: Arc<Mutex<Option<crate::inference::metrics::InferenceMetricsSnapshot>>>,
    pub inference_poll_sequence: Arc<std::sync::atomic::AtomicU64>,
    pub inference_poll_failed: Arc<AtomicBool>,
    pub inference_metrics_session_id: Arc<Mutex<String>>,
    pub inference_poll_failures: Arc<std::sync::atomic::AtomicU64>,
    pub server_logs: Arc<Mutex<VecDeque<String>>>,
    // Stores the PID of the running llama-server child.  The Child handle itself
    // is moved directly into the death_watcher task so it can call wait(); stop_server
    // kills by PID so it never needs to hold the handle.
    #[allow(dead_code)]
    pub server_child: Arc<tokio::sync::Mutex<Option<u32>>>,
    pub server_stopping: Arc<AtomicBool>, // True when stop_server is in progress (for death-watcher coordination)
    // Fired by the death_watcher when the child exits during an intentional stop,
    // so stop_server can unblock and know the port has been released.
    #[allow(dead_code)]
    pub server_exit_notify: Arc<tokio::sync::Notify>,
    pub server_running: Arc<Mutex<bool>>, // Whether active endpoint is reachable (for inference)
    pub local_server_running: Arc<Mutex<bool>>, // Whether a local llama-server was spawned by this app
    pub backend: Arc<Mutex<Option<crate::inference::backend::BackendAdapter>>>,
    pub supervisor: Arc<tokio::sync::Mutex<Option<Arc<crate::inference::supervisor::Supervisor>>>>,
    pub local_launch_request: Arc<Mutex<Option<crate::inference::launch::LocalLaunchRequest>>>,
    /// Serializes local inference process start/stop transitions.
    pub server_lifecycle: Arc<tokio::sync::Mutex<()>>,
    /// Monotonic ownership token for local process lifecycle operations.
    pub server_generation: Arc<std::sync::atomic::AtomicU64>,
    pub server_config: Arc<Mutex<Option<ServerConfig>>>,
    pub llama_poll_notify: Arc<tokio::sync::Notify>,
    pub agent_poll_notify: Arc<tokio::sync::Notify>,
    pub presets: Arc<Mutex<Vec<ModelPreset>>>,
    pub presets_path: PathBuf,
    pub templates: Arc<Mutex<Vec<presets::SystemPromptTemplate>>>,
    pub templates_path: PathBuf,
    pub discovered_models: Arc<Mutex<Vec<DiscoveredModel>>>,
    pub models_dir: Option<PathBuf>,
    pub model_tags: Arc<Mutex<ModelTags>>,
    pub model_tags_path: PathBuf,
    pub preset_collections: Arc<Mutex<crate::collections::PresetCollections>>,
    pub gpu_env: Arc<Mutex<GpuEnv>>,
    pub gpu_env_path: PathBuf,
    pub ui_settings: Arc<Mutex<UiSettings>>,
    pub ui_settings_path: PathBuf,
    pub system_metrics: Arc<Mutex<SystemMetrics>>,
    pub sessions: Arc<Mutex<Vec<Session>>>,
    pub active_session_id: Arc<Mutex<String>>,
    #[allow(dead_code)]
    pub sessions_path: PathBuf,
    pub capabilities: Arc<Mutex<MetricsCapabilities>>,
    pub endpoint_kind: Arc<Mutex<EndpointKind>>,
    pub session_kind: Arc<Mutex<SessionKind>>,
    pub tray_mode: Arc<Mutex<TrayMode>>,
    pub remote_agent_connected: Arc<Mutex<bool>>,
    pub remote_agent_health_reachable: Arc<Mutex<bool>>,
    pub remote_agent_url: Arc<Mutex<Option<String>>>,
    pub remote_agent_version: Arc<Mutex<Option<String>>>,
    pub remote_agent_update_available: Arc<Mutex<bool>>,
    pub remote_agent_protocol_version: Arc<Mutex<Option<String>>>,
    pub remote_agent_protocol_too_old: Arc<Mutex<bool>>,
    pub chat_storage: Arc<ChatStorage>,
    pub tls_config: Arc<Mutex<TLSConfig>>,
    /// Retains llama.cpp's single-request safety contract.
    pub monitor_inference_gate: Arc<tokio::sync::Semaphore>,
    /// Rapid-MLX supports continuous batching, but llama-monitor keeps a fixed
    /// application-side bound so one client cannot create an unbounded queue.
    pub rapid_mlx_inference_gate: Arc<tokio::sync::Semaphore>,
    pub last_spawn_cmd: Arc<Mutex<String>>,

    // Sleep/low-power mode (T-042/T-043)
    // 0 = Off (full monitoring), 1 = LogsOnly, 2 = Sleep (full pause)
    pub sleep_mode: Arc<AtomicU8>,
    pub sleep_mode_manual: Arc<AtomicBool>,
    pub sleep_mode_config: Arc<Mutex<SleepModeConfig>>,
    pub sleep_notify: Arc<tokio::sync::Notify>,
    pub last_activity_at: Arc<std::sync::atomic::AtomicU64>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            gpu_metrics: Arc::new(Mutex::new(BTreeMap::new())),
            llama_metrics: Arc::new(Mutex::new(LlamaMetrics::default())),
            inference_metrics: Arc::new(Mutex::new(None)),
            inference_poll_sequence: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            inference_poll_failed: Arc::new(AtomicBool::new(false)),
            inference_metrics_session_id: Arc::new(Mutex::new(String::new())),
            inference_poll_failures: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            server_logs: Arc::new(Mutex::new(VecDeque::new())),
            server_child: Arc::new(tokio::sync::Mutex::new(None)),
            server_stopping: Arc::new(AtomicBool::new(false)),
            server_exit_notify: Arc::new(tokio::sync::Notify::new()),
            server_running: Arc::new(Mutex::new(false)),
            local_server_running: Arc::new(Mutex::new(false)),
            server_config: Arc::new(Mutex::new(None)),
            llama_poll_notify: Arc::new(tokio::sync::Notify::new()),
            agent_poll_notify: Arc::new(tokio::sync::Notify::new()),
            presets: Arc::new(Mutex::new(Vec::new())),
            presets_path: PathBuf::new(),
            templates: Arc::new(Mutex::new(Vec::new())),
            templates_path: PathBuf::new(),
            discovered_models: Arc::new(Mutex::new(Vec::new())),
            models_dir: None,
            model_tags: Arc::new(Mutex::new(ModelTags::default())),
            model_tags_path: PathBuf::new(),
            preset_collections: Arc::new(Mutex::new(
                crate::collections::PresetCollections::default(),
            )),
            gpu_env: Arc::new(Mutex::new(GpuEnv::default())),
            gpu_env_path: PathBuf::new(),
            ui_settings: Arc::new(Mutex::new(UiSettings::default())),
            ui_settings_path: PathBuf::new(),
            system_metrics: Arc::new(Mutex::new(SystemMetrics::default())),
            sessions: Arc::new(Mutex::new(Vec::new())),
            active_session_id: Arc::new(Mutex::new(String::new())),
            sessions_path: PathBuf::new(),
            capabilities: Arc::new(Mutex::new(MetricsCapabilities::default())),
            endpoint_kind: Arc::new(Mutex::new(EndpointKind::Unknown)),
            session_kind: Arc::new(Mutex::new(SessionKind::None)),
            tray_mode: Arc::new(Mutex::new(TrayMode::Headless)),
            remote_agent_connected: Arc::new(Mutex::new(false)),
            remote_agent_health_reachable: Arc::new(Mutex::new(false)),
            remote_agent_url: Arc::new(Mutex::new(None)),
            remote_agent_version: Arc::new(Mutex::new(None)),
            remote_agent_update_available: Arc::new(Mutex::new(false)),
            remote_agent_protocol_version: Arc::new(Mutex::new(None)),
            remote_agent_protocol_too_old: Arc::new(Mutex::new(false)),
            chat_storage: Arc::new(ChatStorage::default()),
            tls_config: Arc::new(Mutex::new(TLSConfig::default())),
            monitor_inference_gate: Arc::new(tokio::sync::Semaphore::new(1)),
            rapid_mlx_inference_gate: Arc::new(tokio::sync::Semaphore::new(4)),
            last_spawn_cmd: Arc::new(Mutex::new(String::new())),
            backend: Arc::new(Mutex::new(None)),
            supervisor: Arc::new(tokio::sync::Mutex::new(None)),
            local_launch_request: Arc::new(Mutex::new(None)),
            server_lifecycle: Arc::new(tokio::sync::Mutex::new(())),
            server_generation: Arc::new(AtomicU64::new(0)),
            sleep_mode: Arc::new(AtomicU8::new(0)),
            sleep_mode_manual: Arc::new(AtomicBool::new(false)),
            sleep_mode_config: Arc::new(Mutex::new(SleepModeConfig::default())),
            sleep_notify: Arc::new(tokio::sync::Notify::new()),
            last_activity_at: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }
}

impl AppState {
    pub fn new(
        presets: Vec<ModelPreset>,
        paths: AppPaths,
        gpu_env: GpuEnv,
        ui_settings: UiSettings,
        chat_storage: Arc<ChatStorage>,
        tls_config: TLSConfig,
    ) -> Self {
        let presets_path = paths.presets_path;
        let templates_path = paths.templates_path;
        let models_dir = paths.models_dir;
        let gpu_env_path = paths.gpu_env_path;
        let ui_settings_path = paths.ui_settings_path;
        let sessions_path = paths.sessions_path;
        let model_tags_path = paths.model_tags_path;
        let discovered = models_dir
            .as_ref()
            .and_then(|dir| crate::models::scan_gguf_library(dir).ok())
            .unwrap_or_default();
        let model_tags = ModelTags::load(&model_tags_path);
        let preset_collections = {
            let default_cfg = std::path::PathBuf::from(".");
            let config_dir = model_tags_path.parent().unwrap_or(&default_cfg);
            crate::collections::load_collections(config_dir)
        };

        let sessions = load_sessions(&sessions_path);
        let templates = presets::load_templates(&templates_path);
        let active_session_id = String::new();
        let endpoint_kind = EndpointKind::Unknown;
        let session_kind = SessionKind::None;

        let initial_capabilities = MetricsCapabilities {
            inference: true,
            system: false,
            gpu: false,
            cpu_temperature: false,
            memory: false,
            host_metrics: false,
            tray: true,
            sensor_bridge_setup_available: false,
        };

        // T-042/T-043/T-044: sleep_mode and config initialization
        let sleep_cfg = ui_settings.sleep_mode.clone();

        // sleep_mode stored as AtomicBool; watch::channel was used before but send()
        // silently fails when no receivers exist, so the value never updated.

        let last_activity_ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let state = Self {
            gpu_metrics: Arc::new(Mutex::new(BTreeMap::new())),
            llama_metrics: Arc::new(Mutex::new(LlamaMetrics::default())),
            inference_metrics: Arc::new(Mutex::new(None)),
            inference_poll_sequence: Arc::new(AtomicU64::new(0)),
            inference_poll_failed: Arc::new(AtomicBool::new(false)),
            inference_metrics_session_id: Arc::new(Mutex::new(String::new())),
            inference_poll_failures: Arc::new(AtomicU64::new(0)),
            server_logs: Arc::new(Mutex::new(VecDeque::new())),
            server_child: Arc::new(tokio::sync::Mutex::new(None)),
            server_stopping: Arc::new(AtomicBool::new(false)),
            server_exit_notify: Arc::new(tokio::sync::Notify::new()),
            server_running: Arc::new(Mutex::new(false)),
            local_server_running: Arc::new(Mutex::new(false)),
            server_config: Arc::new(Mutex::new(None)),
            llama_poll_notify: Arc::new(tokio::sync::Notify::new()),
            agent_poll_notify: Arc::new(tokio::sync::Notify::new()),
            presets: Arc::new(Mutex::new(presets)),
            presets_path,
            templates: Arc::new(Mutex::new(templates)),
            templates_path,
            discovered_models: Arc::new(Mutex::new(discovered)),
            models_dir,
            model_tags: Arc::new(Mutex::new(model_tags)),
            model_tags_path,
            preset_collections: Arc::new(Mutex::new(preset_collections)),
            gpu_env: Arc::new(Mutex::new(gpu_env)),
            gpu_env_path,
            ui_settings: Arc::new(Mutex::new(ui_settings)),
            ui_settings_path,
            system_metrics: Arc::new(Mutex::new(SystemMetrics::default())),
            sessions: Arc::new(Mutex::new(sessions)),
            active_session_id: Arc::new(Mutex::new(active_session_id)),
            sessions_path,
            capabilities: Arc::new(Mutex::new(initial_capabilities)),
            endpoint_kind: Arc::new(Mutex::new(endpoint_kind)),
            session_kind: Arc::new(Mutex::new(session_kind)),
            tray_mode: Arc::new(Mutex::new(TrayMode::Headless)),
            remote_agent_connected: Arc::new(Mutex::new(false)),
            remote_agent_health_reachable: Arc::new(Mutex::new(false)),
            remote_agent_url: Arc::new(Mutex::new(None)),
            remote_agent_version: Arc::new(Mutex::new(None)),
            remote_agent_update_available: Arc::new(Mutex::new(false)),
            remote_agent_protocol_version: Arc::new(Mutex::new(None)),
            remote_agent_protocol_too_old: Arc::new(Mutex::new(false)),
            chat_storage,
            tls_config: Arc::new(Mutex::new(tls_config)),
            monitor_inference_gate: Arc::new(tokio::sync::Semaphore::new(1)),
            rapid_mlx_inference_gate: Arc::new(tokio::sync::Semaphore::new(4)),
            last_spawn_cmd: Arc::new(Mutex::new(String::new())),
            backend: Arc::new(Mutex::new(None)),
            supervisor: Arc::new(tokio::sync::Mutex::new(None)),
            local_launch_request: Arc::new(Mutex::new(None)),
            server_lifecycle: Arc::new(tokio::sync::Mutex::new(())),
            server_generation: Arc::new(AtomicU64::new(0)),
            sleep_mode: Arc::new(AtomicU8::new(0)), // 0 = Off (full monitoring)
            sleep_mode_manual: Arc::new(AtomicBool::new(false)),
            sleep_mode_config: Arc::new(Mutex::new(sleep_cfg)),
            sleep_notify: Arc::new(tokio::sync::Notify::new()),
            last_activity_at: Arc::new(AtomicU64::new(last_activity_ts)),
        };

        // Prune old inactive sessions on startup (older than 7 days)
        state.prune_old_sessions(7 * 24 * 60 * 60);

        // If there are many stale sessions but no active session, clear them so
        // the app doesn’t present an empty welcome screen while silently hitting
        // the session limit.
        state.cleanup_startup_stale_sessions();

        state
    }

    pub fn push_log(&self, line: String) {
        // Filter high-frequency poll noise that clutters the console
        if is_noise_log(&line) {
            return;
        }

        // Only echo important lines to stderr (terminal):
        // - Internal monitor messages ([monitor])
        // - Lines indicating an error/failure/health issue
        // All lines are still stored for the UI and Logs tab.
        // Use ascii_lowercase (cheaper than unicode case-folding) and any()
        // to short-circuit — all our patterns are pure ASCII.
        let is_important = line.starts_with("[monitor]") || {
            let lower = line.to_ascii_lowercase();
            [
                "error",
                "oom",
                "out of memory",
                "killed",
                "health check",
                "failed",
                "cannot update",
                "not responding",
                "corrupt",
                "incompatible",
            ]
            .iter()
            .any(|p| lower.contains(p))
        };

        if is_important {
            eprintln!("[llama-monitor] {line}");
        }

        let mut logs = self.server_logs.lock().unwrap();
        if logs.len() >= MAX_LOG_LINES {
            logs.pop_front();
        }
        logs.push_back(line);
    }

    pub fn get_sessions(&self) -> Vec<Session> {
        self.sessions.lock().unwrap().clone()
    }

    #[allow(dead_code)]
    pub fn get_active_session(&self) -> Option<Session> {
        let sessions = self.sessions.lock().unwrap();
        let active_id = self.active_session_id.lock().unwrap();
        sessions.iter().find(|s| s.id == *active_id).cloned()
    }

    pub fn set_active_session(&self, session_id: &str) -> bool {
        let exists = {
            let sessions = self.sessions.lock().unwrap();
            session_id.is_empty() || sessions.iter().any(|s| s.id == session_id)
        };
        if !exists {
            return false;
        }

        *self.active_session_id.lock().unwrap() = session_id.to_string();
        self.refresh_capability_state();
        if !session_id.is_empty() {
            self.llama_poll_notify.notify_waiters();
        }
        true
    }

    pub fn add_session(&self, mut session: Session) -> bool {
        let mut sessions = match self.sessions.lock() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[error] Failed to acquire sessions lock: {e}");
                return false;
            }
        };

        let now = Session::now();
        session.last_active = now;

        // If at capacity, try to remove old inactive sessions first
        if sessions.len() >= MAX_SESSIONS {
            // Sort by created_at (oldest first), prefer inactive sessions
            let mut indices: Vec<usize> = (0..sessions.len()).collect();
            indices.sort_by(|&a, &b| {
                let sa = &sessions[a];
                let sb = &sessions[b];
                // Inactive sessions first, then by created_at
                (sa.is_inactive() as i32, sa.created_at)
                    .cmp(&(sb.is_inactive() as i32, sb.created_at))
            });

            // Find first inactive session to remove
            for &idx in &indices {
                if sessions[idx].is_inactive() {
                    eprintln!(
                        "[info] Auto-removing old inactive session: {} ({})",
                        sessions[idx].name, sessions[idx].id
                    );
                    sessions.remove(idx);
                    break;
                }
            }

            // If still at capacity, try to remove obviously stale sessions:
            // - Spawn sessions with status=Running but no local server running.
            // - Sessions stuck in Error(...) (non-recoverable / not usable).
            if sessions.len() >= MAX_SESSIONS {
                let active_id = self.active_session_id.lock().unwrap();
                let local_running = *self.local_server_running.lock().unwrap();

                let mut to_remove: Option<usize> = None;

                for (i, s) in sessions.iter().enumerate() {
                    // Never evict the current active session.
                    if s.id == *active_id {
                        continue;
                    }

                    let stale = matches!(&s.status, SessionStatus::Error(_))
                        || (matches!(&s.status, SessionStatus::Running)
                            && matches!(&s.mode, SessionMode::Spawn { .. })
                            && !local_running);

                    if stale {
                        to_remove = Some(i);
                        break;
                    }
                }

                if let Some(idx) = to_remove {
                    eprintln!(
                        "[info] Auto-removing stale session (making room): {} ({})",
                        sessions[idx].name, sessions[idx].id
                    );
                    sessions.remove(idx);
                }
            }

            // If still at capacity and no inactive sessions, we're truly full
            if sessions.len() >= MAX_SESSIONS {
                eprintln!(
                    "[warn] Session limit reached: all {} sessions are active",
                    sessions.len()
                );
                return false;
            }
        }

        sessions.push(session);
        true
    }

    /// Remove inactive sessions older than the given duration (in seconds)
    pub fn prune_old_sessions(&self, max_age_secs: u64) -> usize {
        let now = Session::now();
        let mut removed = 0;

        let mut sessions = match self.sessions.lock() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[error] Failed to acquire sessions lock for prune: {e}");
                return 0;
            }
        };

        let len_before = sessions.len();
        sessions.retain(|s| {
            let age_secs = now.saturating_sub(s.created_at);
            let is_old = age_secs > max_age_secs;
            let is_error = matches!(&s.status, SessionStatus::Error(_));
            let is_inactive = s.is_inactive() || is_error;

            if is_old && is_inactive {
                let days = (age_secs / 86_400).max(1);
                eprintln!(
                    "[info] Pruning old inactive session: {} (~{} days old)",
                    s.name, days
                );
                removed += 1;
                false
            } else {
                true
            }
        });

        if sessions.len() < len_before {
            eprintln!("[info] Pruned {} old inactive session(s)", removed);
        }

        removed
    }

    /// On startup: if we have many stale sessions but no active session selected,
    /// wipe them and start fresh so the user doesn’t encounter “session limit reached”
    /// on an otherwise empty welcome screen.
    pub fn cleanup_startup_stale_sessions(&self) {
        let (active_id, local_running) = {
            let active_id = self.active_session_id.lock().unwrap();
            let local_running = *self.local_server_running.lock().unwrap();
            (active_id.clone(), local_running)
        };

        // Only act if there’s no meaningful active session.
        if !active_id.is_empty() {
            return;
        }

        let (total, stale_count) = {
            let sessions = match self.sessions.lock() {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("[error] Failed to acquire sessions lock for startup cleanup: {e}");
                    return;
                }
            };

            let total = sessions.len();
            let stale_count = sessions
                .iter()
                .filter(|s| {
                    // Spawn-only sessions: disposable (must re-spawn) → aggressively stale.
                    if matches!(&s.mode, SessionMode::Spawn { .. }) {
                        return matches!(
                            &s.status,
                            SessionStatus::Stopped
                                | SessionStatus::Disconnected
                                | SessionStatus::Error(_)
                                | SessionStatus::Running if !local_running
                        );
                    }

                    // Attach sessions: keep longer (user may reconnect).
                    // Only treat as stale if definitively not in use.
                    if matches!(&s.mode, SessionMode::Attach { .. }) {
                        return matches!(
                            &s.status,
                            SessionStatus::Stopped | SessionStatus::Error(_)
                        );
                    }

                    false
                })
                .count();

            (total, stale_count)
        };

        // Threshold: if there are several stale sessions and almost all are stale,
        // treat it as “ghost sessions” and reset.
        if total >= 5 && stale_count >= total {
            eprintln!(
                "[info] Startup cleanup: {} of {} sessions are stale; resetting to empty",
                stale_count, total
            );

            let mut sessions = self.sessions.lock().unwrap();
            sessions.clear();

            let mut active = self.active_session_id.lock().unwrap();
            active.clear();

            // Force a fresh sessions.json.
            if let Err(e) = save_sessions(&self.sessions_path, &sessions) {
                eprintln!("[error] Failed to persist cleared sessions list: {e}");
            }
        }
    }

    pub fn remove_session(&self, session_id: &str) -> bool {
        let (removed, active_changed) = {
            let mut sessions = self.sessions.lock().unwrap();
            let len_before = sessions.len();
            sessions.retain(|s| s.id != session_id);
            let removed = sessions.len() < len_before;

            let mut active = self.active_session_id.lock().unwrap();
            let active_changed = if *active == session_id {
                active.clear();
                true
            } else {
                false
            };

            (removed, active_changed)
        };

        if active_changed {
            self.refresh_capability_state();
        }

        removed
    }

    pub fn update_session_status(&self, session_id: &str, status: SessionStatus) -> bool {
        let mut sessions = self.sessions.lock().unwrap();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        for session in sessions.iter_mut() {
            if session.id == session_id {
                session.status = status;
                session.last_active = now;
                return true;
            }
        }
        false
    }

    pub fn current_session_kind(&self) -> SessionKind {
        let active_id = self.active_session_id.lock().unwrap().clone();
        if active_id.is_empty() {
            return SessionKind::None;
        }

        let session = {
            let sessions = self.sessions.lock().unwrap();
            sessions.iter().find(|s| s.id == active_id).cloned()
        };

        match session.map(|s| s.mode) {
            Some(SessionMode::Spawn { .. }) => SessionKind::Spawn,
            Some(SessionMode::Attach { .. }) => SessionKind::Attach,
            None => SessionKind::None,
        }
    }

    pub fn current_endpoint_kind(&self) -> EndpointKind {
        let active_id = self.active_session_id.lock().unwrap().clone();
        if active_id.is_empty() {
            return EndpointKind::Unknown;
        }

        let session = {
            let sessions = self.sessions.lock().unwrap();
            sessions.iter().find(|s| s.id == active_id).cloned()
        };

        match session.map(|s| s.mode) {
            Some(SessionMode::Spawn { .. }) => EndpointKind::Local,
            Some(SessionMode::Attach { endpoint, .. }) => endpoint_kind_from_endpoint(&endpoint),
            None => EndpointKind::Unknown,
        }
    }

    pub fn refresh_capability_state(&self) {
        let capabilities = self.calculate_capabilities();
        let endpoint_kind = self.current_endpoint_kind();
        let session_kind = self.current_session_kind();

        *self.capabilities.lock().unwrap() = capabilities;
        *self.endpoint_kind.lock().unwrap() = endpoint_kind;
        *self.session_kind.lock().unwrap() = session_kind;
    }

    pub fn active_session_uses_local_metrics(&self) -> bool {
        let active_id = self.active_session_id.lock().unwrap().clone();
        if active_id.is_empty() {
            return true;
        }

        let session = {
            let sessions = self.sessions.lock().unwrap();
            sessions.iter().find(|s| s.id == active_id).cloned()
        };

        match session.map(|s| s.mode) {
            Some(SessionMode::Spawn { .. }) => true,
            Some(SessionMode::Attach { endpoint, .. }) => endpoint_is_local(&endpoint),
            None => true,
        }
    }

    pub fn remote_agent_connected(&self) -> bool {
        *self.remote_agent_connected.lock().unwrap()
    }

    pub fn remote_agent_health_reachable(&self) -> bool {
        *self.remote_agent_health_reachable.lock().unwrap()
    }

    pub fn host_metrics_available(&self) -> bool {
        self.active_session_uses_local_metrics() || self.remote_agent_connected()
    }

    #[allow(dead_code)]
    pub fn calculate_capabilities(&self) -> MetricsCapabilities {
        let active_id = self.active_session_id.lock().unwrap().clone();
        if active_id.is_empty() {
            return MetricsCapabilities {
                inference: true,
                system: false,
                gpu: false,
                cpu_temperature: false,
                memory: false,
                host_metrics: false,
                tray: true,
                sensor_bridge_setup_available: sensor_bridge_setup_available(),
            };
        }

        let session = {
            let sessions = self.sessions.lock().unwrap();
            sessions.iter().find(|s| s.id == active_id).cloned()
        };

        let full = MetricsCapabilities {
            inference: true,
            system: true,
            gpu: true,
            cpu_temperature: true,
            memory: true,
            host_metrics: true,
            tray: true,
            sensor_bridge_setup_available: sensor_bridge_setup_available(),
        };
        let inference_only = MetricsCapabilities {
            inference: true,
            system: false,
            gpu: false,
            cpu_temperature: false,
            memory: false,
            host_metrics: false,
            tray: true,
            sensor_bridge_setup_available: false,
        };

        match session {
            Some(ref s) if matches!(s.mode, SessionMode::Spawn { .. }) => full,
            Some(ref s)
                if matches!(&s.mode, SessionMode::Attach { endpoint, .. }
                    if endpoint_is_local(endpoint)) =>
            {
                full
            }
            Some(_) if self.remote_agent_connected() => full,
            Some(_) => inference_only,
            None => inference_only,
        }
    }

    #[allow(dead_code)]
    pub fn calculate_availability_reasons(
        &self,
    ) -> (AvailabilityReason, AvailabilityReason, AvailabilityReason) {
        let capabilities = self.calculate_capabilities();

        let system_reason = if capabilities.system {
            AvailabilityReason::Available
        } else {
            AvailabilityReason::RemoteEndpoint
        };

        let gpu_reason = if capabilities.gpu {
            AvailabilityReason::Available
        } else {
            AvailabilityReason::RemoteEndpoint
        };

        let cpu_temp_reason = if capabilities.cpu_temperature {
            AvailabilityReason::Available
        } else {
            AvailabilityReason::RemoteEndpoint
        };

        (system_reason, gpu_reason, cpu_temp_reason)
    }

    pub fn get_tls_config(&self) -> TLSConfig {
        self.tls_config.lock().unwrap().clone()
    }

    pub fn set_tls_config(&self, config: TLSConfig) {
        *self.tls_config.lock().unwrap() = config;
    }

    /// Sleep mode helpers (T-043 3-way):
    /// 0 = Off (full monitoring), 1 = LogsOnly, 2 = Sleep (full pause)
    #[allow(dead_code)]
    pub fn is_sleeping(&self) -> bool {
        self.sleep_mode.load(std::sync::atomic::Ordering::Relaxed) >= 2
    }

    #[allow(dead_code)]
    pub fn is_logs_only(&self) -> bool {
        self.sleep_mode.load(std::sync::atomic::Ordering::Relaxed) == 1
    }

    #[allow(dead_code)]
    pub fn is_full(&self) -> bool {
        self.sleep_mode.load(std::sync::atomic::Ordering::Relaxed) == 0
    }

    /// Current mode as string for APIs/WS.
    pub fn sleep_mode_str(&self) -> &'static str {
        match self.sleep_mode.load(std::sync::atomic::Ordering::Relaxed) {
            0 => "off",
            1 => "logs-only",
            2 => "sleep",
            _ => "off",
        }
    }
}

/// Returns `true` if the given log line is known llama.cpp poll noise.
///
/// These patterns are safe to suppress in the push_log echo loop because
/// they represent routine server status updates that add no value to terminal output
/// but fire frequently on every poll cycle.
///
/// Patterns filtered:
/// - `update_slots` + `all slots are idle` — idle-slot poll noise on every tick
/// - `stop: cancel task` + `id_task` — task cancellation on every idle tick
fn is_noise_log(line: &str) -> bool {
    (line.contains("update_slots") && line.contains("all slots are idle"))
        || (line.contains("stop: cancel task") && line.contains("id_task"))
}

#[allow(dead_code)]
pub fn endpoint_kind_from_endpoint(endpoint: &str) -> EndpointKind {
    if endpoint_is_local(endpoint) {
        EndpointKind::Local
    } else {
        EndpointKind::Remote
    }
}

fn endpoint_is_local(endpoint: &str) -> bool {
    let url = reqwest::Url::parse(endpoint)
        .or_else(|_| reqwest::Url::parse(&format!("http://{endpoint}")));
    let Ok(url) = url else {
        return false;
    };
    let Some(host) = url.host_str() else {
        return false;
    };

    host_is_local(host)
}

fn host_is_local(host: &str) -> bool {
    let host = host.trim_matches(['[', ']']);
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }

    let Ok(ip) = host.parse::<IpAddr>() else {
        return false;
    };

    if ip.is_loopback() {
        return true;
    }

    // Probe the OS routing table directly: connect a UDP socket toward the
    // target IP (no packets are sent) and read back the local address the OS
    // selected. If the local address equals the target, the IP belongs to one
    // of our own interfaces — even on multi-interface machines where the
    // outbound-to-internet probe in local_interface_ips() would miss it.
    let bind_addr = if ip.is_ipv6() { "[::]:0" } else { "0.0.0.0:0" };
    let target: SocketAddr = (ip, 80).into();
    if let Ok(socket) = UdpSocket::bind(bind_addr)
        && socket.connect(target).is_ok()
        && let Ok(local_addr) = socket.local_addr()
        && local_addr.ip() == ip
    {
        return true;
    }

    // Fallback: check outbound IPs via probes to well-known external hosts
    local_interface_ips().contains(&ip)
}

fn local_interface_ips() -> Vec<IpAddr> {
    let mut ips = Vec::new();

    let probes = [
        ("0.0.0.0:0", "8.8.8.8:80"),
        ("[::]:0", "[2001:4860:4860::8888]:80"),
    ];

    for (bind_addr, connect_addr) in probes {
        let Ok(socket) = UdpSocket::bind(bind_addr) else {
            continue;
        };
        let Ok(remote) = connect_addr.parse::<SocketAddr>() else {
            continue;
        };
        if socket.connect(remote).is_ok()
            && let Ok(local_addr) = socket.local_addr()
        {
            ips.push(local_addr.ip());
        }
    }

    ips
}

// ── Inference diagnostics ───────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixAction {
    AddToolCallParser,
    EnableAutoToolChoice,
    AddNoThinking,
    DisablePrefixCache,
    AdjustPrefixCacheBudget(u64),
    DisableMaxCacheBlocks,
    SetPrefixCacheBudget(u64),
    ReclaimBackendAllocatorCache,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DoctorFinding {
    #[serde(rename = "type")]
    pub finding_type: DoctorFindingType,
    pub severity: DoctorSeverity,
    pub message: String,
    #[serde(default)]
    pub section: String,
    #[serde(default)]
    pub fix: Option<FixAction>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DoctorFindingType {
    Environment,
    Preset,
    LlamaCpp,
    Cache,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DoctorSeverity {
    Ok,
    Warning,
    Issue,
}

impl DoctorSeverity {
    pub fn from_glyph(glyph: char) -> Self {
        match glyph {
            '✓' => Self::Ok,
            '⚠' | '!' => Self::Warning,
            '✗' | '×' => Self::Issue,
            _ => Self::Warning,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn test_paths(sessions_path: PathBuf) -> AppPaths {
        AppPaths {
            presets_path: PathBuf::new(),
            templates_path: PathBuf::new(),
            models_dir: None,
            gpu_env_path: PathBuf::new(),
            ui_settings_path: PathBuf::new(),
            sessions_path,
            model_tags_path: PathBuf::new(),
        }
    }

    fn test_tls_config() -> TLSConfig {
        TLSConfig::default()
    }

    #[test]
    fn model_tags_missing_file_is_recreated_and_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model-tags.json");

        let mut tags = ModelTags::load(&path);
        assert!(path.exists());
        assert!(tags.tags.is_empty());

        tags.tags
            .insert("/models/example.gguf".into(), vec!["family:test".into()]);
        tags.save(&path).unwrap();

        assert_eq!(ModelTags::load(&path).tags, tags.tags);
    }

    #[test]
    fn model_tags_migrates_direct_map_format() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model-tags.json");
        fs::write(
            &path,
            r#"{"/models/example.gguf":["family:test","favorite"]}"#,
        )
        .unwrap();

        let tags = ModelTags::load(&path);
        assert_eq!(
            tags.tags.get("/models/example.gguf"),
            Some(&vec!["family:test".to_string(), "favorite".to_string()])
        );

        let migrated: ModelTags = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(migrated.tags, tags.tags);
    }

    #[test]
    fn missing_ui_settings_and_sessions_files_are_recreated() {
        let dir = tempfile::tempdir().unwrap();
        let settings_path = dir.path().join("ui-settings.json");
        let sessions_path = dir.path().join("sessions.json");

        let settings = load_ui_settings(&settings_path);
        let sessions = load_sessions(&sessions_path);

        assert!(settings_path.exists());
        assert!(sessions_path.exists());
        assert_eq!(settings.ws_push_interval_ms, default_ws_push_interval_ms());
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].id, "default");
    }

    #[test]
    fn filters_llama_server_cancel_task_noise_with_variable_prefixes() {
        assert!(is_noise_log(
            "0.43.327.046 W srv          stop: cancel task, id_task = 28"
        ));
        assert!(is_noise_log(
            "5.41.976.030 W srv          stop: cancel task, id_task = 168"
        ));
        assert!(is_noise_log(
            "\u{1b}[1;36msrv\u{1b}[0m          stop: cancel task, id_task = 66"
        ));
    }

    #[test]
    fn preserves_other_llama_server_stop_logs() {
        assert!(!is_noise_log(
            "0.43.327.046 W srv          stop: server shutdown requested"
        ));
        assert!(!is_noise_log(
            "0.43.327.046 E srv          stop: cancel task queue failed"
        ));
    }

    #[test]
    fn endpoint_detection_with_various_hosts() {
        let local_hosts = [
            "http://localhost:8001",
            "127.0.0.1:8001",
            "http://[::1]:8001",
        ];
        for host in local_hosts {
            assert!(endpoint_is_local(host));
            assert_eq!(endpoint_kind_from_endpoint(host), EndpointKind::Local);
        }

        // Use RFC 5737 documentation addresses — guaranteed never assigned to
        // real interfaces, so these are always remote regardless of test environment.
        let remote_hosts = [
            "http://203.0.113.10:8001",
            "http://198.51.100.5:8001",
            "http://example.com:8001",
        ];
        for host in remote_hosts {
            assert!(!endpoint_is_local(host));
            assert_eq!(endpoint_kind_from_endpoint(host), EndpointKind::Remote);
        }
    }

    #[test]
    fn availability_reason_roundtrip() {
        let reasons = [
            AvailabilityReason::Available,
            AvailabilityReason::RemoteEndpoint,
            AvailabilityReason::NoDisplay,
            AvailabilityReason::TrayUnavailable,
            AvailabilityReason::SensorUnavailable,
            AvailabilityReason::BackendUnavailable,
            AvailabilityReason::CommandMissing,
            AvailabilityReason::PermissionDenied,
            AvailabilityReason::MetricsUnreachable,
            AvailabilityReason::NotApplicable,
        ];

        for reason in reasons {
            let json = serde_json::to_string(&reason).unwrap();
            let deserialized: AvailabilityReason = serde_json::from_str(&json).unwrap();
            assert_eq!(reason, deserialized);
        }
    }

    #[test]
    fn capabilities_based_on_session_mode() {
        let test_cases = [
            (
                "spawn",
                true,
                true,
                true,
                AvailabilityReason::Available,
                AvailabilityReason::Available,
                AvailabilityReason::Available,
            ),
            (
                "attach",
                true,
                false,
                false,
                AvailabilityReason::RemoteEndpoint,
                AvailabilityReason::RemoteEndpoint,
                AvailabilityReason::RemoteEndpoint,
            ),
        ];

        for (
            mode,
            exp_inference,
            exp_system,
            exp_gpu,
            exp_sys_reason,
            exp_gpu_reason,
            exp_cpu_reason,
        ) in test_cases
        {
            let paths = test_paths(PathBuf::new());
            let cs = Arc::new(ChatStorage::open(&PathBuf::from(":memory:")).unwrap());
            let state = AppState::new(
                vec![],
                paths,
                GpuEnv::default(),
                UiSettings::default(),
                cs,
                test_tls_config(),
            );
            let session = if mode == "spawn" {
                Session::new_spawn(
                    "test".to_string(),
                    "Test".to_string(),
                    8001,
                    String::new(),
                    None,
                    None,
                )
            } else {
                Session::new_attach(
                    "test".to_string(),
                    "Test".to_string(),
                    "http://remote.example.com:8001".to_string(),
                    None,
                )
            };
            state.add_session(session);
            state.set_active_session("test");

            let caps = state.calculate_capabilities();
            assert!(caps.inference == exp_inference, "inference for {mode}");
            assert!(caps.system == exp_system, "system for {mode}");
            assert!(caps.gpu == exp_gpu, "gpu for {mode}");

            let (system_reason, gpu_reason, cpu_temp_reason) =
                state.calculate_availability_reasons();
            assert_eq!(system_reason, exp_sys_reason, "system_reason for {mode}");
            assert_eq!(gpu_reason, exp_gpu_reason, "gpu_reason for {mode}");
            assert_eq!(
                cpu_temp_reason, exp_cpu_reason,
                "cpu_temp_reason for {mode}"
            );
        }
    }

    #[test]
    fn startup_does_not_auto_activate_persisted_sessions() {
        let suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let sessions_path = std::env::temp_dir().join(format!(
            "llama-monitor-state-test-{}-{suffix}-sessions.json",
            std::process::id(),
        ));
        let sessions = vec![Session::new_attach(
            "saved".to_string(),
            "Saved".to_string(),
            "http://remote.example.com:8001".to_string(),
            None,
        )];
        std::fs::write(&sessions_path, serde_json::to_string(&sessions).unwrap()).unwrap();

        let cs = Arc::new(ChatStorage::open(&PathBuf::from(":memory:")).unwrap());
        let state = AppState::new(
            vec![],
            test_paths(sessions_path.clone()),
            GpuEnv::default(),
            UiSettings::default(),
            cs,
            test_tls_config(),
        );

        assert!(state.active_session_id.lock().unwrap().is_empty());
        assert_eq!(state.current_session_kind(), SessionKind::None);
        assert_eq!(state.current_endpoint_kind(), EndpointKind::Unknown);
        assert!(!state.calculate_capabilities().host_metrics);

        let _ = std::fs::remove_file(sessions_path);
    }

    #[test]
    fn legacy_spawn_session_without_backend_defaults_to_llama_cpp() {
        let session: Session = serde_json::from_value(serde_json::json!({
            "id": "legacy",
            "name": "Legacy",
            "mode": { "Spawn": { "port": 8001 } }
        }))
        .unwrap();

        assert_eq!(
            session.backend,
            crate::inference::InferenceBackend::LlamaCpp
        );
        assert!(session.model_identity.is_none());
    }

    #[test]
    fn session_serialization_never_persists_endpoint_api_keys() {
        let mut session = Session::new_spawn(
            "private".into(),
            "Private".into(),
            8001,
            String::new(),
            None,
            Some("do-not-write".into()),
        );
        session.model_identity = Some("/models/private.gguf".into());
        session.launch = Some(
            crate::inference::launch::LocalLaunchRequest::LlamaCpp(Box::new(
                crate::inference::llama_cpp::ServerConfig {
                    model_path: "/models/private.gguf".into(),
                    port: 8001,
                    api_key: Some("do-not-write".into()),
                    ..Default::default()
                },
            ))
            .for_persistence(),
        );
        let json = serde_json::to_string(&session).unwrap();
        assert!(!json.contains("do-not-write"));
        assert!(json.contains("launch_requires_api_key"));

        let restored: Session = serde_json::from_str(&json).unwrap();
        assert_eq!(
            restored.backend,
            crate::inference::InferenceBackend::LlamaCpp
        );
        assert_eq!(
            restored.model_identity.as_deref(),
            Some("/models/private.gguf")
        );
        assert!(restored.launch_requires_api_key);
        assert!(matches!(
            restored.launch,
            Some(crate::inference::launch::LocalLaunchRequest::LlamaCpp(_))
        ));
        assert!(matches!(
            restored.mode,
            SessionMode::Spawn { api_key: None, .. }
        ));

        let attached = Session::new_attach(
            "private-attach".into(),
            "Private attach".into(),
            "http://127.0.0.1:9000".into(),
            Some("attach-secret".into()),
        );
        let json = serde_json::to_string(&attached).unwrap();
        assert!(!json.contains("attach-secret"));
        assert!(attached.launch_requires_api_key);
        let restored: Session = serde_json::from_str(&json).unwrap();
        assert!(restored.launch_requires_api_key);
        assert!(matches!(
            restored.mode,
            SessionMode::Attach { api_key: None, .. }
        ));
    }

    #[test]
    fn set_active_session_rejects_missing_session_without_mutating() {
        let cs = Arc::new(ChatStorage::open(&PathBuf::from(":memory:")).unwrap());
        let state = AppState::new(
            vec![],
            test_paths(PathBuf::new()),
            GpuEnv::default(),
            UiSettings::default(),
            cs,
            test_tls_config(),
        );
        state.add_session(Session::new_spawn(
            "existing".to_string(),
            "Existing".to_string(),
            8001,
            String::new(),
            None,
            None,
        ));

        assert!(state.set_active_session("existing"));
        assert!(!state.set_active_session("missing"));
        assert_eq!(*state.active_session_id.lock().unwrap(), "existing");
    }

    #[test]
    fn removing_active_session_clears_active_state() {
        let cs = Arc::new(ChatStorage::open(&PathBuf::from(":memory:")).unwrap());
        let state = AppState::new(
            vec![],
            test_paths(PathBuf::new()),
            GpuEnv::default(),
            UiSettings::default(),
            cs,
            test_tls_config(),
        );
        state.add_session(Session::new_spawn(
            "existing".to_string(),
            "Existing".to_string(),
            8001,
            String::new(),
            None,
            None,
        ));

        assert!(state.set_active_session("existing"));
        assert!(state.remove_session("existing"));
        assert!(state.active_session_id.lock().unwrap().is_empty());
        assert_eq!(state.current_session_kind(), SessionKind::None);
        assert!(!state.calculate_capabilities().host_metrics);
    }

    #[test]
    fn metrics_capabilities_serializes_to_json() {
        let caps = MetricsCapabilities {
            inference: true,
            system: true,
            gpu: false,
            cpu_temperature: false,
            memory: false,
            host_metrics: false,
            tray: true,
            sensor_bridge_setup_available: false,
        };
        let json = serde_json::to_value(&caps).unwrap();
        assert_eq!(json["inference"], true);
        assert_eq!(json["system"], true);
        assert_eq!(json["gpu"], false);
    }
}
