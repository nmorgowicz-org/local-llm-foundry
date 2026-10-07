use anyhow::{Result, anyhow};
use reqwest::Client;
use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tokio::process::Command as TokioCommand;

use crate::config::AppConfig;
use crate::gpu::env::{GpuEnv, build_nvidia_env, build_rocm_env};
use crate::inference::InferenceBackend;
use crate::inference::capabilities::CapabilitySet;
use crate::inference::llama_cpp_capabilities::CapabilitySnapshot;
use crate::inference::metrics::{HealthState, InferenceMetricsSnapshot};
use crate::inference::supervisor::SupervisedLaunch;
use crate::llama::metrics::{
    LlamaRuntimeAdapter, LlamaRuntimeFacts, parse_prometheus_metrics, parse_slot_metrics,
};

fn describe_process_status(status: std::process::ExitStatus) -> String {
    if let Some(code) = status.code() {
        return format!("exit code {code}");
    }

    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            return format!("signal {signal}");
        }
    }

    "exit status unknown".to_string()
}

fn readiness_host(bind_host: Option<&str>) -> &str {
    match bind_host.unwrap_or("127.0.0.1") {
        "0.0.0.0" | "::" | "[::]" => "127.0.0.1",
        host => host,
    }
}

fn launch_environment(gpu_backend: &str, gpu_env: &GpuEnv, cwd: &str) -> Vec<(OsString, OsString)> {
    match gpu_backend {
        "nvidia" => build_nvidia_env(gpu_env),
        "none" => Vec::new(),
        _ => build_rocm_env(gpu_env, cwd),
    }
    .into_iter()
    .map(|(key, value)| (key.into(), value.into()))
    .collect()
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct SpecDecodeConfig {
    #[serde(default)]
    pub draft_model: String,
    #[serde(default)]
    pub draft_min: Option<u32>,
    #[serde(default)]
    pub draft_max: Option<u32>,
    #[serde(default)]
    pub spec_ngram_size: Option<u32>,
    #[serde(default)]
    pub spec_type: Option<String>,
    #[serde(default)]
    pub spec_default: bool,
    #[serde(default)]
    pub spec_draft_n_max: Option<u32>,
    #[serde(default)]
    pub spec_draft_n_min: Option<u32>,
    #[serde(default)]
    pub spec_draft_p_split: Option<f32>,
    #[serde(default)]
    pub spec_draft_p_min: Option<f32>,
    #[serde(default)]
    pub spec_draft_ngl: Option<i32>,
    #[serde(default)]
    pub spec_draft_device: Option<String>,
    #[serde(default)]
    pub spec_draft_cpu_moe: bool,
    #[serde(default)]
    pub spec_draft_n_cpu_moe: Option<i32>,
    #[serde(default)]
    pub spec_draft_type_k: Option<String>,
    #[serde(default)]
    pub spec_draft_type_v: Option<String>,
    #[serde(default)]
    pub spec_ngram_mod_n_min: Option<u32>,
    #[serde(default)]
    pub spec_ngram_mod_n_max: Option<u32>,
    #[serde(default)]
    pub spec_ngram_mod_n_match: Option<u32>,
    #[serde(default)]
    pub spec_ngram_simple_size_n: Option<u32>,
    #[serde(default)]
    pub spec_ngram_simple_size_m: Option<u32>,
    #[serde(default)]
    pub spec_ngram_simple_min_hits: Option<u32>,
    #[serde(default)]
    pub spec_ngram_map_k_size_n: Option<u32>,
    #[serde(default)]
    pub spec_ngram_map_k_size_m: Option<u32>,
    #[serde(default)]
    pub spec_ngram_map_k_min_hits: Option<u32>,
    #[serde(default)]
    pub spec_ngram_map_k4v_size_n: Option<u32>,
    #[serde(default)]
    pub spec_ngram_map_k4v_size_m: Option<u32>,
    #[serde(default)]
    pub spec_ngram_map_k4v_min_hits: Option<u32>,
}

/// Explicit llama.cpp model loading policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LoadMode {
    #[serde(rename = "mmap")]
    Mmap,
    #[serde(rename = "none")]
    None,
    #[serde(rename = "mlock")]
    Mlock,
    #[serde(rename = "mmap+mlock")]
    MmapMlock,
    #[serde(rename = "dio")]
    Dio,
}

impl LoadMode {
    pub const fn as_flag(self) -> &'static str {
        match self {
            Self::Mmap => "mmap",
            Self::None => "none",
            Self::Mlock => "mlock",
            Self::MmapMlock => "mmap+mlock",
            Self::Dio => "dio",
        }
    }

    pub const fn with_mlock(self, mlock: bool) -> Self {
        if !mlock {
            return self;
        }
        match self {
            Self::Mmap => Self::MmapMlock,
            Self::None => Self::Mlock,
            other => other,
        }
    }
}

/// Typed llama.cpp reasoning-effort value (Phase 2).
///
/// Separate from Rapid-MLX `reasoning_effort` (request-default field with
/// different runtime meaning). The `Default` variant is the runtime/template
/// default and emits **no** `--reasoning-effort` argument; only an explicit
/// non-default level is emitted. `Unknown(s)` preserves an unrecognized future
/// level verbatim (round-trips unchanged through save/edit/save) and remains
/// non-launchable. Bounded open-string serde per architecture §9/§10.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum LlamaReasoningEffort {
    #[default]
    Default,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
    /// Unknown future effort level — preserved verbatim, non-launchable.
    Unknown(String),
}

impl LlamaReasoningEffort {
    /// The exact `--reasoning-effort` flag value to emit, or `None` for the
    /// runtime/template default (and for `Unknown`, which is non-launchable).
    pub fn as_flag_value(&self) -> Option<&str> {
        match self {
            Self::Default => None,
            Self::Minimal => Some("minimal"),
            Self::Low => Some("low"),
            Self::Medium => Some("medium"),
            Self::High => Some("high"),
            Self::Xhigh => Some("xhigh"),
            Self::Max => Some("max"),
            Self::Unknown(_) => None,
        }
    }

    /// Parse a wire/stored string into the enum, preserving unknown values.
    fn from_wire(s: &str) -> Self {
        match s {
            "default" => Self::Default,
            "minimal" => Self::Minimal,
            "low" => Self::Low,
            "medium" => Self::Medium,
            "high" => Self::High,
            "xhigh" => Self::Xhigh,
            "max" => Self::Max,
            _ => Self::Unknown(s.to_string()),
        }
    }

    /// The exact wire string (known variant name, or the preserved raw value).
    fn to_wire(&self) -> &str {
        match self {
            Self::Default => "default",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
            Self::Max => "max",
            Self::Unknown(s) => s,
        }
    }
}

impl serde::Serialize for LlamaReasoningEffort {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        s.serialize_str(self.to_wire())
    }
}

impl<'de> serde::Deserialize<'de> for LlamaReasoningEffort {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(d)?;
        Ok(Self::from_wire(&s))
    }
}

/// Typed llama.cpp reasoning-format value (Phase 2).
///
/// `None` (the outer Option on the field) means runtime default/auto — no
/// argument is emitted. Explicit known variants come from exact runtime
/// capability evidence; observed explicit values are `none`, `deepseek`,
/// `deepseek-legacy`. `Unknown(s)` preserves unrecognized future values
/// verbatim but remains non-launchable.
///
/// Architecture §9: never emit `--reasoning-format auto` unless a future
/// exact binary advertises `auto` as an accepted explicit value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LlamaReasoningFormat {
    None,
    Deepseek,
    DeepseekLegacy,
    /// Unknown future format — preserved, non-launchable.
    Unknown(String),
}

impl LlamaReasoningFormat {
    /// The exact `--reasoning-format` flag value to emit for an explicit format
    /// variant. Returns `None` for `Unknown` (non-launchable).
    pub fn as_flag_value(&self) -> Option<&str> {
        match self {
            Self::None => Some("none"),
            Self::Deepseek => Some("deepseek"),
            Self::DeepseekLegacy => Some("deepseek-legacy"),
            Self::Unknown(_) => None,
        }
    }

    fn from_wire(s: &str) -> Self {
        match s {
            "none" => Self::None,
            "deepseek" => Self::Deepseek,
            "deepseek-legacy" => Self::DeepseekLegacy,
            _ => Self::Unknown(s.to_string()),
        }
    }

    fn to_wire(&self) -> &str {
        match self {
            Self::None => "none",
            Self::Deepseek => "deepseek",
            Self::DeepseekLegacy => "deepseek-legacy",
            Self::Unknown(s) => s,
        }
    }
}

impl serde::Serialize for LlamaReasoningFormat {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        s.serialize_str(self.to_wire())
    }
}

impl<'de> serde::Deserialize<'de> for LlamaReasoningFormat {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(d)?;
        Ok(Self::from_wire(&s))
    }
}

/// Phase 6: cross-backend prompt-cache mode (llama.cpp side).
///
/// `Custom` is the serde default (not `Auto`) so that configs saved before this field existed
/// keep deserializing to their exact stored `cache_ram_mib` value unchanged — adding this field
/// must not silently change already-launched configurations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheMode {
    /// No workload-scenario evidence is plumbed into this launch path yet, so `Auto` resolves
    /// to the same disabled state as `Off` rather than guessing a bounded positive cap.
    Auto,
    /// Idle-slot prompt cache off (`--cache-ram 0`).
    Off,
    /// User-supplied `cache_ram_mib` is used as configured, unchanged.
    #[default]
    Custom,
}

impl CacheMode {
    /// Resolve to the effective `cache_ram_mib` value. `Custom` returns the configured value
    /// untouched; `Auto`/`Off` both resolve to `Some(0)` (disabled) in this scoped pass.
    fn resolve(self, configured_cache_ram_mib: Option<i32>) -> Option<i32> {
        match self {
            CacheMode::Auto | CacheMode::Off => Some(0),
            CacheMode::Custom => configured_cache_ram_mib,
        }
    }
}

/// Phase 1b: macOS llama.cpp has no `--cache-ram` support, so the value is
/// forced to `Some(0)` there regardless of stored `cache_ram_mib` or
/// `cache_mode`. On other platforms the configured resolution is returned
/// unchanged. Resolving to `0` also suppresses `--cache-idle-slots`, which
/// requires cache-ram to be nonzero.
fn effective_cache_ram(configured_cache_ram_mib: Option<i32>, mode: CacheMode) -> Option<i32> {
    let resolved = mode.resolve(configured_cache_ram_mib);
    if cfg!(target_os = "macos") {
        Some(0)
    } else {
        resolved
    }
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ServerConfig {
    pub model_path: String,
    pub context_size: u64,
    pub ctk: String,
    pub ctv: String,
    pub tensor_split: String,
    pub batch_size: u32,
    pub ubatch_size: u32,
    pub no_mmap: bool,
    #[serde(default)]
    pub load_mode: Option<LoadMode>,
    pub verbosity: Option<i32>,
    pub no_cont_batching: bool,
    pub swa_full: bool,
    pub ctx_checkpoints: Option<u32>,
    pub checkpoint_min_step: Option<u32>,
    pub cache_reuse: Option<u32>,
    pub port: u16,
    pub ngram_spec: bool,
    pub parallel_slots: u32,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<i32>,
    #[serde(default)]
    pub min_p: Option<f64>,
    #[serde(default)]
    pub repeat_penalty: Option<f64>,
    pub repeat_last_n: Option<u32>,
    #[serde(default)]
    pub presence_penalty: Option<f64>,
    #[serde(default)]
    pub n_cpu_moe: Option<i32>,
    #[serde(default)]
    pub gpu_layers: Option<i32>,
    #[serde(default)]
    pub mlock: bool,
    #[serde(default)]
    pub flash_attn: String,
    #[serde(default)]
    pub split_mode: String,
    #[serde(default)]
    pub main_gpu: Option<u32>,
    #[serde(default)]
    pub threads: Option<i32>,
    #[serde(default)]
    pub threads_batch: Option<i32>,
    #[serde(default)]
    pub prio: Option<i32>,
    #[serde(default)]
    pub prio_batch: Option<i32>,
    #[serde(default)]
    pub rope_scaling: String,
    #[serde(default)]
    pub rope_freq_base: Option<f64>,
    #[serde(default)]
    pub rope_freq_scale: Option<f64>,
    #[serde(flatten, default)]
    pub spec: SpecDecodeConfig,
    #[serde(default)]
    pub kv_unified: Option<bool>,
    #[serde(default)]
    pub cache_idle_slots: Option<bool>,
    #[serde(default)]
    pub cache_ram_mib: Option<i32>,
    /// Phase 6: Auto/Off/Custom prompt-cache mode. `Custom` (the default) uses
    /// `cache_ram_mib` as configured; `Auto`/`Off` override it at launch time —
    /// see [`CacheMode::resolve`].
    #[serde(default)]
    pub cache_mode: CacheMode,
    #[serde(default)]
    pub fit_enabled: Option<bool>,
    #[serde(default)]
    pub fit_ctx: Option<u32>,
    #[serde(default)]
    pub fit_target: Option<String>,
    #[serde(default)]
    pub fit_print: Option<bool>,
    #[serde(default)]
    pub seed: Option<i64>,
    pub system_prompt_file: String,
    pub extra_args: String,
    #[serde(default)]
    pub bind_host: Option<String>,
    #[serde(default)]
    pub hf_repo: Option<String>,
    #[serde(default)]
    pub alias: Option<String>,
    #[serde(default)]
    pub chat_template_file: Option<String>,
    #[serde(default)]
    pub mmproj: Option<String>,
    #[serde(default)]
    pub grammar: Option<String>,
    #[serde(default)]
    pub json_schema: Option<String>,
    #[serde(default)]
    pub cache_type_k: Option<String>,
    #[serde(default)]
    pub cache_type_v: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<u64>,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub benchmark_mode: bool,
    #[serde(default)]
    pub enable_thinking: Option<bool>,
    #[serde(default)]
    pub preserve_thinking: Option<bool>,
    #[serde(default)]
    pub tool_call_format: Option<String>,
    #[serde(default)]
    pub reasoning: Option<String>,
    #[serde(default)]
    pub reasoning_budget: Option<i32>,
    #[serde(default)]
    pub reasoning_budget_message: Option<String>,
    #[serde(default)]
    pub image_min_tokens: Option<u32>,
    #[serde(default)]
    pub image_max_tokens: Option<u32>,
    /// Phase 2: multimodal-projector GPU offload (architecture §9).
    /// `None` = exact runtime default (no arg); `Some(true)` = emit the
    /// positive flag when supported; `Some(false)` = emit `--no-mmproj-offload`
    /// when supported. Emission is capability-gated.
    #[serde(default)]
    pub mmproj_offload: Option<bool>,
    /// Phase 2: typed llama.cpp reasoning-effort level (architecture §9).
    /// Distinct from Rapid-MLX `reasoning_effort`. `Default`/`Unknown` emit no
    /// argument; explicit levels emit `--reasoning-effort <level>` when
    /// supported.
    #[serde(default)]
    pub llama_reasoning_effort: LlamaReasoningEffort,
    /// Phase 2: typed llama.cpp reasoning format (architecture §9).
    /// `None` = runtime default/auto (no arg); explicit values emit
    /// `--reasoning-format <value>` when supported. Never emits `auto`.
    #[serde(default)]
    pub llama_reasoning_format: Option<LlamaReasoningFormat>,
    /// Phase 2: preserve the reasoning trace across the full history
    /// (architecture §9). Valueless flag — `Some(true)` emits
    /// `--reasoning-preserve`, `Some(false)` emits nothing (unless the snapshot
    /// advertises `--no-reasoning-preserve`, in which case it emits that),
    /// `None` emits nothing. Requires binary support plus a compatible
    /// reasoning mode. Distinct from `preserve_thinking`.
    #[serde(default)]
    pub llama_reasoning_preserve: Option<bool>,
    /// Internal launch envelope for Phase 2 runtime validation. It is not
    /// serialized into API/session payloads; persisted presets remain the
    /// source of truth for bundle data.
    #[serde(skip)]
    pub bundle: Option<crate::presets::bundle::PresetBundleSpec>,
}

#[derive(Debug, Clone, Default)]
pub struct CounterSnapshot {
    prompt_tokens_total: f64,
    prompt_seconds_total: f64,
    predicted_tokens_total: f64,
    predicted_seconds_total: f64,
}

/// Endpoint-scoped optional metadata. Deliberately not Debug: the key contains auth.
#[derive(Clone, Default)]
pub struct LlamaRuntimeCache {
    target: Option<(String, Option<String>, String)>,
    last_attempt: Option<Instant>,
    facts: Option<LlamaRuntimeFacts>,
    speculative_enabled: Option<bool>,
    model_ctx_train: Option<u64>,
}

pub(crate) fn same_api_key(left: Option<&str>, right: Option<&str>) -> bool {
    use subtle::ConstantTimeEq;
    match (left, right) {
        (Some(left), Some(right)) => bool::from(left.as_bytes().ct_eq(right.as_bytes())),
        (None, None) => true,
        _ => false,
    }
}

/// Comparable source identity without exposing URL credentials, queries, or fragments.
pub(crate) fn telemetry_endpoint_tag(base: &str) -> String {
    use sha2::{Digest, Sha256};

    let canonical = match url::Url::parse(base) {
        Ok(mut url) => {
            if url.username().is_empty()
                && url.password().is_none()
                && url.query().is_none()
                && url.fragment().is_none()
            {
                // Preserve the existing comparison for plain endpoints.
                return base.trim_end_matches('/').into();
            }
            // Strip only path slashes: a trailing slash in a query value is data.
            let path = url.path().trim_end_matches('/').to_owned();
            url.set_path(if path.is_empty() { "/" } else { &path });
            url.to_string()
        }
        Err(_) => base.to_owned(),
    };
    // Hash the whole canonical URL so changes to credentials or server identity
    // cannot collapse into a shared redaction marker. This is not an auth token.
    let digest: String = Sha256::digest(canonical.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    format!("sha256:{digest}")
}

impl LlamaRuntimeCache {
    pub fn select_target(&mut self, base: &str, api_key: Option<&str>, session_id: &str) -> bool {
        let base = base.trim_end_matches('/');
        let unchanged = self
            .target
            .as_ref()
            .is_some_and(|(old_base, old_key, old_session)| {
                old_base == base
                    && old_session == session_id
                    && same_api_key(old_key.as_deref(), api_key)
            });
        if unchanged {
            return false;
        }
        *self = Self {
            target: Some((base.into(), api_key.map(str::to_owned), session_id.into())),
            ..Default::default()
        };
        true
    }

    fn refresh_due(&self, now: Instant) -> bool {
        self.last_attempt
            .is_none_or(|last| now.duration_since(last) >= Duration::from_secs(60))
    }

    async fn refresh(&mut self, client: &Client, base: &str, api_key: Option<&str>) {
        let now = Instant::now();
        if !self.refresh_due(now) {
            return;
        }
        // Failed/unsupported endpoints are throttled too. Expire facts rather than
        // holding an old attached model indefinitely after a runtime reload.
        self.last_attempt = Some(now);
        let (props, models) = tokio::join!(
            optional_runtime_json(client, base, "/props", api_key),
            optional_runtime_json(client, base, "/v1/models", api_key),
        );
        self.speculative_enabled = props.as_ref().and_then(props_speculative_enabled);
        self.model_ctx_train = models
            .as_ref()
            .and_then(|v| v.pointer("/data/0/meta/n_ctx_train"))
            .and_then(|v| v.as_u64());
        self.facts = parse_runtime_facts(props.as_ref(), models.as_ref());
        // GET /lora-adapters uses the server task queue. Only probe an explicitly
        // awake llama.cpp server; do not wake sleeping servers for optional facts.
        if props
            .as_ref()
            .and_then(|v| v.get("is_sleeping"))
            .and_then(|v| v.as_bool())
            == Some(false)
            && let Some(adapters) =
                optional_runtime_json(client, base, "/lora-adapters", api_key).await
            && let Some(adapters) = parse_runtime_adapters(&adapters)
        {
            self.facts.get_or_insert_with(Default::default).adapters = Some(adapters);
        }
    }
}

async fn optional_runtime_json(
    client: &Client,
    base: &str,
    path: &str,
    api_key: Option<&str>,
) -> Option<serde_json::Value> {
    // Bound the whole exchange, including body decoding. Never log upstream
    // bodies, URLs, or credentials from these optional endpoints.
    tokio::time::timeout(Duration::from_secs(2), async {
        let mut request = client.get(format!("{}{path}", base.trim_end_matches('/')));
        if let Some(key) = api_key {
            request = request.bearer_auth(key);
        }
        const MAX_RUNTIME_BYTES: usize = 1024 * 1024;
        let mut response = request.send().await.ok()?.error_for_status().ok()?;
        if response
            .content_length()
            .is_some_and(|length| length > MAX_RUNTIME_BYTES as u64)
        {
            return None;
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response.chunk().await.ok()? {
            if chunk.len() > MAX_RUNTIME_BYTES.saturating_sub(bytes.len()) {
                return None;
            }
            bytes.extend_from_slice(&chunk);
        }
        serde_json::from_slice::<serde_json::Value>(&bytes).ok()
    })
    .await
    .ok()
    .flatten()
}

fn display_basename(value: &serde_json::Value) -> Option<String> {
    let value = value.as_str()?.trim();
    let name = value.rsplit(['/', '\\']).next()?;
    (!name.is_empty() && name != "." && name != ".." && !name.chars().any(char::is_control))
        .then(|| name.to_owned())
}

fn fact_string(value: Option<&serde_json::Value>) -> Option<String> {
    let value = value?.as_str()?.trim();
    (!value.is_empty() && !value.contains(['/', '\\']) && !value.chars().any(char::is_control))
        .then(|| value.to_owned())
}

fn parse_runtime_facts(
    props: Option<&serde_json::Value>,
    models: Option<&serde_json::Value>,
) -> Option<LlamaRuntimeFacts> {
    let model = models.and_then(|v| v.pointer("/data/0"));
    let mut facts = LlamaRuntimeFacts {
        model_name: props
            .and_then(|v| v.get("model_path"))
            .and_then(display_basename),
        model_alias: props
            .and_then(|v| v.get("model_alias"))
            .and_then(display_basename)
            .or_else(|| model.and_then(|v| v.get("id")).and_then(display_basename)),
        quantization: fact_string(props.and_then(|v| v.get("model_ftype")))
            .or_else(|| fact_string(model.and_then(|v| v.pointer("/meta/ftype")))),
        model_params: model
            .and_then(|v| v.pointer("/meta/n_params"))
            .and_then(|v| v.as_u64())
            .or_else(|| {
                props
                    .and_then(|v| v.get("n_params"))
                    .and_then(|v| v.as_u64())
            }),
        server_build: fact_string(props.and_then(|v| v.get("build_info"))),
        ..Default::default()
    };
    // Only explicit booleans under named capability containers. No inference
    // from local binary capabilities, templates, launch args, or file names.
    for (container, names) in [
        ("modalities", &["vision", "video", "audio"][..]),
        (
            "chat_template_caps",
            &[
                "supports_string_content",
                "supports_typed_content",
                "supports_tools",
                "supports_tool_calls",
                "supports_parallel_tool_calls",
                "supports_system_role",
                "supports_preserve_reasoning",
                "supports_reasoning_effort",
                "supports_object_arguments",
            ][..],
        ),
    ] {
        for name in names {
            if let Some(value) = props
                .and_then(|v| v.get(container))
                .and_then(|v| v.get(*name))
                .and_then(|v| v.as_bool())
            {
                facts.capabilities.insert((*name).into(), value);
            }
        }
    }
    (facts != LlamaRuntimeFacts::default()).then_some(facts)
}

fn props_speculative_enabled(props: &serde_json::Value) -> Option<bool> {
    let params = props.pointer("/default_generation_settings/params")?;
    crate::llama::metrics::speculative_config_enabled(params)
}

fn parse_runtime_adapters(value: &serde_json::Value) -> Option<Vec<LlamaRuntimeAdapter>> {
    Some(
        value
            .as_array()?
            .iter()
            .take(64)
            .filter_map(|value| {
                let adapter = LlamaRuntimeAdapter {
                    id: value.get("id").and_then(|v| v.as_u64()),
                    name: value
                        .get("name")
                        .or_else(|| value.get("path"))
                        .and_then(display_basename),
                    scale: value
                        .get("scale")
                        .and_then(|v| v.as_f64())
                        .filter(|v| v.is_finite()),
                };
                (adapter != LlamaRuntimeAdapter::default()).then_some(adapter)
            })
            .collect(),
    )
}

fn counter_rate(
    current_tokens: f64,
    previous_tokens: f64,
    current_seconds: f64,
    previous_seconds: f64,
) -> f64 {
    let token_delta = current_tokens - previous_tokens;
    let second_delta = current_seconds - previous_seconds;

    if token_delta > 0.0 && second_delta > 0.0 {
        token_delta / second_delta
    } else {
        0.0
    }
}

pub struct LlamaCppAdapter {
    pub app_config: AppConfig,
    pub config: ServerConfig,
    gpu_env: GpuEnv,
    capabilities: Option<CapabilitySnapshot>,
    previous_counters: Mutex<Option<CounterSnapshot>>,
    previous_counter_session: Mutex<Option<String>>,
    runtime_cache: tokio::sync::Mutex<LlamaRuntimeCache>,
}

#[allow(dead_code)]
impl LlamaCppAdapter {
    pub fn new(app_config: AppConfig, config: ServerConfig, gpu_env: GpuEnv) -> Self {
        Self::new_with_capabilities(app_config, config, gpu_env, None)
    }

    pub fn new_with_capabilities(
        app_config: AppConfig,
        config: ServerConfig,
        gpu_env: GpuEnv,
        capabilities: Option<CapabilitySnapshot>,
    ) -> Self {
        Self {
            app_config,
            config,
            gpu_env,
            capabilities,
            previous_counters: Mutex::new(None),
            previous_counter_session: Mutex::new(None),
            runtime_cache: tokio::sync::Mutex::new(LlamaRuntimeCache::default()),
        }
    }

    pub async fn validate(&self) -> Result<()> {
        let bin_path = &self.app_config.llama_server_path;
        if bin_path.components().count() > 1 && !bin_path.exists() {
            return Err(anyhow!(
                "llama-server binary not found: {}. Set it in Configuration.",
                bin_path.display()
            ));
        }

        let use_hf = self.config.hf_repo.as_ref().is_some_and(|r| !r.is_empty());
        let has_model_path = !self.config.model_path.is_empty();

        if use_hf && has_model_path {
            return Err(anyhow!(
                "Cannot use both model_path and hf_repo. Choose one."
            ));
        }

        if !use_hf && has_model_path {
            if !std::path::Path::new(&self.config.model_path).exists() {
                return Err(anyhow!("Model file not found: {}", self.config.model_path));
            }
        } else if !use_hf && !has_model_path {
            return Err(anyhow!(
                "No model source specified. Provide model_path or hf_repo."
            ));
        }

        self.validate_binary().await
    }

    async fn validate_binary(&self) -> Result<()> {
        let bin_path = &self.app_config.llama_server_path;

        #[cfg(target_os = "macos")]
        if let Some(bin_dir) = bin_path.parent() {
            let _ = std::process::Command::new("xattr")
                .args(["-rd", "com.apple.quarantine"])
                .arg(bin_dir)
                .output();
        }

        let output = tokio::time::timeout(Duration::from_secs(10), async {
            TokioCommand::new(bin_path)
                .arg("--help")
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::piped())
                .output()
                .await
        })
        .await
        .map_err(|_| anyhow!("llama-server did not respond to its health check within 10 seconds"))?
        .map_err(|error| anyhow!("Failed to execute llama-server health check: {error}"))?;

        if output.status.success() {
            return Ok(());
        }

        let detail = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let status = describe_process_status(output.status);
        if detail.is_empty() {
            Err(anyhow!(
                "llama-server health check failed ({status}). The binary may be corrupted or incompatible."
            ))
        } else {
            Err(anyhow!(
                "llama-server health check failed ({status}): {detail}"
            ))
        }
    }

    pub async fn build_launch(&self) -> Result<SupervisedLaunch> {
        self.validate_typed_capabilities()?;
        let mut cmd = TokioCommand::new(&self.app_config.llama_server_path);
        crate::platform::no_window_tokio(&mut cmd);
        cmd.current_dir(&self.app_config.llama_server_cwd);

        let use_hf = self.config.hf_repo.as_ref().is_some_and(|r| !r.is_empty());
        if use_hf {
            if let Some(ref repo) = self.config.hf_repo {
                cmd.arg("-hf").arg(repo);
            }
        } else {
            cmd.arg("-m").arg(&self.config.model_path);
        }

        let ngl_str = match self.config.gpu_layers {
            Some(-1) | None => "all".to_string(),
            Some(n) => n.to_string(),
        };
        cmd.arg("-ngl").arg(&ngl_str);
        if !self.config.ctk.is_empty() {
            cmd.arg("-ctk").arg(&self.config.ctk);
        }
        if !self.config.ctv.is_empty() {
            cmd.arg("-ctv").arg(&self.config.ctv);
        }
        cmd.arg("--host")
            .arg(self.config.bind_host.as_deref().unwrap_or("127.0.0.1"));
        cmd.arg("--port").arg(self.config.port.to_string());
        if self.config.context_size > 0 {
            cmd.arg("-c").arg(self.config.context_size.to_string());
        }
        if self.config.batch_size > 0 {
            cmd.arg("-b").arg(self.config.batch_size.to_string());
        }
        if self.config.ubatch_size > 0 {
            cmd.arg("-ub").arg(self.config.ubatch_size.to_string());
        }
        cmd.arg("--no-warmup");
        cmd.arg("--jinja");
        cmd.arg("--no-context-shift");
        cmd.arg("--ctx-checkpoints")
            .arg(self.config.ctx_checkpoints.unwrap_or(32).to_string());
        if let Some(step) = self.config.checkpoint_min_step {
            cmd.arg("--checkpoint-min-step").arg(step.to_string());
        }
        if let Some(reuse) = self.config.cache_reuse {
            cmd.arg("--cache-reuse").arg(reuse.to_string());
        }
        if self.config.no_cont_batching {
            cmd.arg("--no-cont-batching");
        }
        if self.config.swa_full {
            cmd.arg("--swa-full");
        }
        cmd.arg("--keep").arg("-1");

        if let Some(mode) = self.config.load_mode {
            cmd.arg("--load-mode").arg(mode.as_flag());
        } else if self.config.no_mmap {
            // Compatibility for pre-v4 presets and older llama.cpp binaries.
            cmd.arg("--no-mmap");
        }
        if let Some(verbosity) = self.config.verbosity {
            cmd.arg("-lv").arg(verbosity.to_string());
        }
        if self.config.load_mode.is_none() && self.config.mlock {
            cmd.arg("--mlock");
        }

        let fa_value = if self.config.flash_attn == "off" {
            "off"
        } else {
            "on"
        };
        cmd.arg("-fa").arg(fa_value);

        if !self.config.tensor_split.is_empty() {
            cmd.arg("-ts").arg(&self.config.tensor_split);
        }
        if !self.config.split_mode.is_empty() {
            cmd.arg("--split-mode").arg(&self.config.split_mode);
        }
        if let Some(mg) = self.config.main_gpu {
            cmd.arg("-mg").arg(mg.to_string());
        }

        if let Some(t) = self.config.threads
            && (t == -1 || t > 0)
        {
            cmd.arg("-t").arg(t.to_string());
        }
        if let Some(tb) = self.config.threads_batch
            && (tb == -1 || tb > 0)
        {
            cmd.arg("-tb").arg(tb.to_string());
        }

        if let Some(p) = self.config.prio {
            cmd.arg("--prio").arg(p.to_string());
        }
        if let Some(pb) = self.config.prio_batch {
            cmd.arg("--prio-batch").arg(pb.to_string());
        }

        if !self.config.rope_scaling.is_empty() {
            cmd.arg("--rope-scaling").arg(&self.config.rope_scaling);
        } else if self.config.context_size > 262144 {
            cmd.arg("--rope-scaling").arg("yarn");
        }
        if let Some(base) = self.config.rope_freq_base {
            cmd.arg("--rope-freq-base").arg(format!("{:.6}", base));
        }
        if let Some(scale) = self.config.rope_freq_scale {
            cmd.arg("--rope-freq-scale").arg(format!("{:.6}", scale));
        } else if self.config.rope_scaling.is_empty() && self.config.context_size > 262144 {
            let scale = 262144.0 / self.config.context_size as f64;
            cmd.arg("--rope-freq-scale").arg(format!("{:.6}", scale));
            cmd.arg("--yarn-ext-factor").arg("1.0");
            cmd.arg("--yarn-attn-factor").arg("1.0");
            cmd.arg("--yarn-beta-fast").arg("32");
            cmd.arg("--yarn-beta-slow").arg("1");
        }

        let s = &self.config.spec;
        let spec_type_effective = if s.spec_type.is_some() {
            s.spec_type.clone()
        } else if self.config.ngram_spec {
            Some("ngram-mod".to_string())
        } else {
            None
        };

        if let Some(ref st) = spec_type_effective {
            cmd.arg("--spec-type").arg(st);
        }
        if s.spec_default {
            cmd.arg("--spec-default");
        }
        if !s.draft_model.is_empty() {
            cmd.arg("-md").arg(&s.draft_model);
        }
        if let Some(v) = s.spec_draft_n_max {
            cmd.arg("--spec-draft-n-max").arg(v.to_string());
        }
        if let Some(v) = s.spec_draft_n_min {
            cmd.arg("--spec-draft-n-min").arg(v.to_string());
        }
        if let Some(v) = s.spec_draft_p_split {
            cmd.arg("--spec-draft-p-split").arg(format!("{:.4}", v));
        }
        if let Some(v) = s.spec_draft_p_min {
            cmd.arg("--spec-draft-p-min").arg(format!("{:.4}", v));
        }
        if let Some(v) = s.spec_draft_ngl {
            let value = if v < 0 {
                "all".to_string()
            } else {
                v.to_string()
            };
            cmd.arg("--spec-draft-ngl").arg(value);
        }
        if let Some(ref v) = s.spec_draft_device {
            // `gpu` is a legacy app-level placeholder, not a llama.cpp device
            // identifier. Leave it unset unless the user supplied an explicit
            // upstream device name such as CUDA0.
            if !v.eq_ignore_ascii_case("gpu") {
                cmd.arg("--spec-draft-device").arg(v);
            }
        }
        if s.spec_draft_cpu_moe {
            cmd.arg("--spec-draft-cpu-moe");
        }
        if let Some(v) = s.spec_draft_n_cpu_moe {
            cmd.arg("--spec-draft-n-cpu-moe").arg(v.to_string());
        }
        if let Some(ref v) = s.spec_draft_type_k {
            cmd.arg("--spec-draft-type-k").arg(v);
        }
        if let Some(ref v) = s.spec_draft_type_v {
            cmd.arg("--spec-draft-type-v").arg(v);
        }
        if let Some(v) = s.spec_ngram_mod_n_min {
            cmd.arg("--spec-ngram-mod-n-min").arg(v.to_string());
        }
        if let Some(v) = s.spec_ngram_mod_n_max {
            cmd.arg("--spec-ngram-mod-n-max").arg(v.to_string());
        }
        if let Some(v) = s.spec_ngram_mod_n_match {
            cmd.arg("--spec-ngram-mod-n-match").arg(v.to_string());
        }
        if let Some(v) = s.spec_ngram_simple_size_n {
            cmd.arg("--spec-ngram-simple-size-n").arg(v.to_string());
        }
        if let Some(v) = s.spec_ngram_simple_size_m {
            cmd.arg("--spec-ngram-simple-size-m").arg(v.to_string());
        }
        if let Some(v) = s.spec_ngram_simple_min_hits {
            cmd.arg("--spec-ngram-simple-min-hits").arg(v.to_string());
        }
        if let Some(v) = s.spec_ngram_map_k_size_n {
            cmd.arg("--spec-ngram-map-k-size-n").arg(v.to_string());
        }
        if let Some(v) = s.spec_ngram_map_k_size_m {
            cmd.arg("--spec-ngram-map-k-size-m").arg(v.to_string());
        }
        if let Some(v) = s.spec_ngram_map_k_min_hits {
            cmd.arg("--spec-ngram-map-k-min-hits").arg(v.to_string());
        }
        if let Some(v) = s.spec_ngram_map_k4v_size_n {
            cmd.arg("--spec-ngram-map-k4v-size-n").arg(v.to_string());
        }
        if let Some(v) = s.spec_ngram_map_k4v_size_m {
            cmd.arg("--spec-ngram-map-k4v-size-m").arg(v.to_string());
        }
        if let Some(v) = s.spec_ngram_map_k4v_min_hits {
            cmd.arg("--spec-ngram-map-k4v-min-hits").arg(v.to_string());
        }

        if self.config.ngram_spec {
            if let Some(v) = s.spec_ngram_size {
                cmd.arg("--spec-ngram-size-n").arg(v.to_string());
            }
            if let Some(v) = s.draft_min {
                cmd.arg("--draft-min").arg(v.to_string());
            }
            if let Some(v) = s.draft_max {
                cmd.arg("--draft-max").arg(v.to_string());
            }
        }

        if self.config.parallel_slots > 0 {
            cmd.arg("--parallel")
                .arg(self.config.parallel_slots.to_string());
        }

        if let Some(t) = self.config.temperature {
            cmd.arg("--temp").arg(format!("{:.2}", t));
        }
        if let Some(tp) = self.config.top_p {
            cmd.arg("--top-p").arg(format!("{:.4}", tp));
        }
        if let Some(tk) = self.config.top_k {
            cmd.arg("--top-k").arg(tk.to_string());
        }
        if let Some(mp) = self.config.min_p {
            cmd.arg("--min-p").arg(format!("{:.4}", mp));
        }
        if let Some(rp) = self.config.repeat_penalty {
            cmd.arg("--repeat-penalty").arg(format!("{:.2}", rp));
        }
        if let Some(last_n) = self.config.repeat_last_n {
            cmd.arg("--repeat-last-n").arg(last_n.to_string());
        }
        if let Some(pp) = self.config.presence_penalty {
            cmd.arg("--presence-penalty").arg(format!("{:.4}", pp));
        }
        if let Some(n) = self.config.n_cpu_moe.filter(|value| *value > 0) {
            cmd.arg("--n-cpu-moe").arg(n.to_string());
        }

        if let Some(seed) = self.config.seed {
            cmd.arg("--seed").arg(seed.to_string());
        }
        if !self.config.system_prompt_file.is_empty() {
            cmd.arg("--system-prompt-file")
                .arg(&self.config.system_prompt_file);
        }

        if let Some(ref ct) = self.config.chat_template_file
            && !ct.is_empty()
        {
            cmd.arg("--chat-template-file").arg(ct);
        }

        {
            let mut kwargs = serde_json::Map::new();
            if let Some(et) = self.config.enable_thinking {
                kwargs.insert("enable_thinking".into(), serde_json::json!(et));
            }
            if let Some(pt) = self.config.preserve_thinking {
                kwargs.insert("preserve_thinking".into(), serde_json::json!(pt));
            }
            if let Some(ref tcf) = self.config.tool_call_format
                && !tcf.is_empty()
            {
                kwargs.insert("tool_call_format".into(), serde_json::json!(tcf));
            }
            if !kwargs.is_empty() {
                let json = serde_json::to_string(&kwargs).unwrap_or_default();
                cmd.arg("--chat-template-kwargs").arg(json);
            }
        }
        if let Some(ref mode) = self.config.reasoning
            && !mode.is_empty()
        {
            cmd.arg("--reasoning").arg(mode);
        }
        if let Some(budget) = self.config.reasoning_budget {
            cmd.arg("--reasoning-budget").arg(budget.to_string());
        }
        if let Some(ref msg) = self.config.reasoning_budget_message
            && !msg.is_empty()
        {
            cmd.arg("--reasoning-budget-message").arg(msg);
        }

        if let Some(value) = self.config.mmproj_offload {
            cmd.arg(if value {
                "--mmproj-offload"
            } else {
                "--no-mmproj-offload"
            });
        }
        if let Some(value) = self.config.llama_reasoning_effort.as_flag_value() {
            cmd.arg("--reasoning-effort").arg(value);
        }
        if let Some(value) = self
            .config
            .llama_reasoning_format
            .as_ref()
            .and_then(LlamaReasoningFormat::as_flag_value)
        {
            cmd.arg("--reasoning-format").arg(value);
        }
        if self.config.llama_reasoning_preserve == Some(true) {
            cmd.arg("--reasoning-preserve");
        } else if self.config.llama_reasoning_preserve == Some(false)
            && self
                .capabilities
                .as_ref()
                .is_some_and(|caps| caps.supports_flag("--no-reasoning-preserve"))
        {
            cmd.arg("--no-reasoning-preserve");
        }

        cmd.arg("--metrics");

        if let Some(ref mp) = self.config.mmproj
            && !mp.is_empty()
        {
            cmd.arg("--mmproj").arg(mp);
            if let Some(min) = self.config.image_min_tokens {
                cmd.arg("--image-min-tokens").arg(min.to_string());
            }
            if let Some(max) = self.config.image_max_tokens {
                cmd.arg("--image-max-tokens").arg(max.to_string());
            }
        }

        if let Some(ref g) = self.config.grammar
            && !g.is_empty()
        {
            cmd.arg("--grammar").arg(g);
        }
        if let Some(ref js) = self.config.json_schema
            && !js.is_empty()
        {
            cmd.arg("--json-schema").arg(js);
        }
        if let Some(mt) = self.config.max_tokens {
            cmd.arg("-n").arg(mt.to_string());
        }
        if let Some(ref ak) = self.config.api_key
            && !ak.is_empty()
        {
            cmd.arg("--api-key").arg(ak);
        }
        if let Some(ref al) = self.config.alias
            && !al.is_empty()
        {
            cmd.arg("--alias").arg(al);
        }

        self.append_kv_cache_args(&mut cmd);
        self.append_fit_args(&mut cmd);

        for arg in self.config.extra_args.split_whitespace() {
            cmd.arg(arg);
        }

        let args: Vec<OsString> = cmd.as_std().get_args().map(|a| a.to_owned()).collect();
        let program = PathBuf::from(cmd.as_std().get_program());

        let cwd = self.app_config.llama_server_cwd.display().to_string();
        let env = launch_environment(&self.app_config.gpu_backend, &self.gpu_env, &cwd);

        Ok(SupervisedLaunch {
            warnings: Vec::new(),
            program,
            args,
            env,
            cwd: Some(self.app_config.llama_server_cwd.clone()),
            port: self.config.port,
            redacted_summary: format!(
                "llama-server on port={} model={}",
                self.config.port,
                if !self.config.model_path.is_empty() {
                    &self.config.model_path
                } else if let Some(ref r) = self.config.hf_repo {
                    r
                } else {
                    "<unknown>"
                }
            ),
        })
    }

    fn validate_typed_capabilities(&self) -> Result<()> {
        let has_typed_value = self.config.mmproj_offload.is_some()
            || !matches!(
                &self.config.llama_reasoning_effort,
                LlamaReasoningEffort::Default
            )
            || self.config.llama_reasoning_format.is_some()
            || self.config.llama_reasoning_preserve.is_some();
        if !has_typed_value {
            return Ok(());
        }

        let caps = self.capabilities.as_ref().ok_or_else(|| {
            anyhow!(
                "typed llama.cpp launch settings require a capability snapshot for the exact binary"
            )
        })?;

        if let Some(value) = self.config.mmproj_offload {
            let flag = if value {
                "--mmproj-offload"
            } else {
                "--no-mmproj-offload"
            };
            if !caps.supports_flag(flag) {
                anyhow::bail!("typed llama.cpp setting is unsupported: {flag}");
            }
        }

        match &self.config.llama_reasoning_effort {
            LlamaReasoningEffort::Default => {}
            LlamaReasoningEffort::Unknown(value) => anyhow::bail!(
                "unknown llama.cpp reasoning effort '{value}' is preserved but not launchable"
            ),
            value => {
                let flag_value = value
                    .as_flag_value()
                    .expect("known effort has a flag value");
                if !caps.supports_reasoning_effort(flag_value) {
                    anyhow::bail!(
                        "llama.cpp reasoning effort '{flag_value}' is unavailable in the exact binary"
                    );
                }
            }
        }

        if let Some(format) = &self.config.llama_reasoning_format {
            let value = format.as_flag_value().ok_or_else(|| {
                anyhow!("unknown llama.cpp reasoning format is preserved but not launchable")
            })?;
            if !caps.supports_reasoning_format(value) {
                anyhow::bail!(
                    "llama.cpp reasoning format '{value}' is unavailable in the exact binary"
                );
            }
        }

        if let Some(preserve) = self.config.llama_reasoning_preserve {
            let flag = if preserve {
                "--reasoning-preserve"
            } else {
                "--no-reasoning-preserve"
            };
            if !caps.supports_flag(flag) {
                anyhow::bail!("typed llama.cpp setting is unsupported: {flag}");
            }
            if preserve && self.config.reasoning.as_deref().is_none_or(str::is_empty) {
                anyhow::bail!(
                    "native reasoning preservation requires an explicit compatible reasoning mode"
                );
            }
        }

        Ok(())
    }

    fn append_kv_cache_args(&self, cmd: &mut TokioCommand) {
        // Phase 6: resolve the effective cache_ram_mib through CacheMode before it drives
        // either --cache-idle-slots eligibility or --cache-ram itself.
        // macOS has no --cache-ram support in llama.cpp; it is forced to 0
        // regardless of stored cache_ram_mib or cache_mode, which also
        // suppresses --cache-idle-slots via the eligibility gate below.
        let cache_ram_mib = effective_cache_ram(self.config.cache_ram_mib, self.config.cache_mode);

        if let Some(v) = self.config.kv_unified {
            cmd.arg(if v { "--kv-unified" } else { "--no-kv-unified" });
        }
        if let Some(v) = self.config.cache_idle_slots {
            if v {
                // llama-server uses 0 as disabled and -1 as explicitly unlimited.
                let cache_enabled = cache_ram_mib != Some(0);
                if cache_enabled {
                    if self.config.kv_unified.is_none() {
                        cmd.arg("--kv-unified");
                    }
                    cmd.arg("--cache-idle-slots");
                }
            } else {
                cmd.arg("--no-cache-idle-slots");
            }
        }
        if let Some(v) = cache_ram_mib {
            cmd.arg("--cache-ram").arg(v.to_string());
        }
    }

    fn append_fit_args(&self, cmd: &mut TokioCommand) {
        match self.config.fit_enabled {
            None => return,
            Some(false) => {
                cmd.arg("--fit").arg("off");
                return;
            }
            Some(true) => {}
        }

        cmd.arg("--fit").arg("on");
        if let Some(ref v) = self.config.fit_target {
            cmd.arg("--fit-target").arg(v);
        } else if let Some(v) = self.config.fit_ctx {
            cmd.arg("--fit-ctx").arg(v.to_string());
        }
    }

    pub async fn await_ready(&self, port: u16, deadline: Instant) -> Result<()> {
        let client = Client::builder().timeout(Duration::from_secs(5)).build()?;

        let host = readiness_host(self.config.bind_host.as_deref());
        let url = format!("http://{host}:{port}/health");
        let api_key = &self.config.api_key;

        loop {
            if Instant::now() > deadline {
                return Err(anyhow!("LlamaCppAdapter: timeout waiting for readiness"));
            }

            let req = if let Some(key) = api_key {
                client
                    .get(&url)
                    .header("Authorization", format!("Bearer {}", key))
            } else {
                client.get(&url)
            };

            if let Ok(resp) = req.send().await
                && resp.status().is_success()
            {
                return Ok(());
            }

            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }

    pub async fn poll_metrics(
        &self,
        base: &str,
        session_id: &str,
    ) -> Result<InferenceMetricsSnapshot> {
        let mut runtime_cache = self.runtime_cache.lock().await;
        let mut previous_counters = self.previous_counters.lock().unwrap().clone();
        let mut previous_counter_session = self.previous_counter_session.lock().unwrap().clone();
        let result = poll_llama_cpp_metrics(
            base,
            self.config.api_key.as_deref(),
            session_id,
            &mut previous_counters,
            &mut previous_counter_session,
            &mut runtime_cache,
        )
        .await;
        *self.previous_counters.lock().unwrap() = previous_counters;
        *self.previous_counter_session.lock().unwrap() = previous_counter_session;
        result
    }

    pub async fn cancel_request(&self, _port: u16, _request_id: &str) -> Result<()> {
        Err(anyhow!(
            "The active llama.cpp backend does not support native request cancellation"
        ))
    }

    pub fn capabilities(&self) -> &CapabilitySet {
        static CAPS: CapabilitySet = CapabilitySet {
            vision: true,
            mtp: false,
            cancellation: false,
            embeddings: true,
            guided_generation: true,
            audio: false,
            tool_parsing: true,
            automatic_tool_choice: true,
            reasoning_parser: true,
            thinking_controls: true,
            mcp: true,
            cache_telemetry: true,
            status_memory_telemetry: true,
            self_diagnostic: false,
            interpretability: false,
            one_shot_launch: false,
        };
        &CAPS
    }
}

/// Poll normalized llama.cpp metrics from `base` (a full resolved endpoint URL).
/// Does not require a `LlamaCppAdapter` — used directly by the shared poller loop for
/// both spawned sessions and Attach sessions, since only Spawn sessions populate
/// `state.backend` (attach never owns/launches a process, so there is no adapter to poll
/// through there).
pub async fn poll_llama_cpp_metrics(
    base: &str,
    api_key: Option<&str>,
    session_id: &str,
    previous_counters: &mut Option<CounterSnapshot>,
    previous_counter_session: &mut Option<String>,
    runtime_cache: &mut LlamaRuntimeCache,
) -> Result<InferenceMetricsSnapshot> {
    let endpoint_tag = telemetry_endpoint_tag(base);
    let base = base.trim_end_matches('/');
    if runtime_cache.select_target(base, api_key, session_id) {
        *previous_counters = None;
        *previous_counter_session = None;
    }
    {
        let client = Client::builder()
            .timeout(Duration::from_secs(5))
            .pool_max_idle_per_host(0)
            .pool_idle_timeout(Duration::from_secs(0))
            .build()?;

        let mut snapshot = InferenceMetricsSnapshot::empty(InferenceBackend::LlamaCpp);

        // Health check
        let health_req = if let Some(key) = api_key {
            client
                .get(format!("{base}/health"))
                .header("Authorization", format!("Bearer {}", key))
        } else {
            client.get(format!("{base}/health"))
        };

        if let Ok(resp) = health_req.send().await
            && let Ok(body) = resp.text().await
            && let Ok(json) = serde_json::from_str::<serde_json::Value>(&body)
        {
            snapshot.health = Some(match json.get("status").and_then(|v| v.as_str()) {
                Some("running") => HealthState::Ok,
                Some("degraded") => HealthState::Degraded,
                Some("not_loaded") => HealthState::NotLoaded,
                _ => HealthState::Unreachable,
            });
            snapshot.ready = json.get("ready").and_then(|v| v.as_bool());
        }

        // Prometheus metrics
        let mut tokens_per_decode = 0.0;
        let mut n_busy_slots_per_decode = 0.0;
        let mut details = serde_json::Map::new();
        let metrics_req = if let Some(key) = api_key {
            client
                .get(format!("{base}/metrics"))
                .header("Authorization", format!("Bearer {}", key))
        } else {
            client.get(format!("{base}/metrics"))
        };

        if let Ok(resp) = metrics_req.send().await
            && resp.status().is_success()
            && let Ok(body) = resp.text().await
        {
            let prom = parse_prometheus_metrics(&body);
            snapshot.prompt_tokens_total = prom.prompt_tokens_processed_total.map(|v| v as u64);
            snapshot.completion_tokens_total = Some(prom.predicted_tokens_total as u64);
            snapshot.running_requests = Some(prom.requests_processing as u64);
            snapshot.steps_executed = Some(prom.n_decode_total as u64);
            tokens_per_decode = prom.tokens_per_decode;
            n_busy_slots_per_decode = prom.n_busy_slots_per_decode;
            snapshot.speculative_acceptance_rate = prom
                .speculative_accepted_tokens_total
                .zip(prom.speculative_draft_tokens_total)
                .filter(|(_, drafted)| *drafted > 0)
                .map(|(accepted, drafted)| accepted as f64 / drafted as f64);
            details.extend(serde_json::json!({
                "prompt_tokens_processed_total": prom.prompt_tokens_processed_total,
                "prompt_tokens_cached_total": prom.prompt_tokens_cached_total,
                "speculative_draft_tokens_total": prom.speculative_draft_tokens_total,
                "speculative_accepted_tokens_total": prom.speculative_accepted_tokens_total,
                "speculative_verification_steps_total": prom.speculative_verification_steps_total,
            }).as_object().unwrap().clone());

            let current_counters = CounterSnapshot {
                prompt_tokens_total: prom.prompt_tokens_total,
                prompt_seconds_total: prom.prompt_seconds_total,
                predicted_tokens_total: prom.predicted_tokens_total,
                predicted_seconds_total: prom.predicted_seconds_total,
            };

            let (prompt_tps, gen_tps) = {
                if previous_counter_session.as_deref() == Some(session_id)
                    && previous_counters.is_some()
                {
                    let prev = previous_counters.as_ref().unwrap();
                    (
                        counter_rate(
                            current_counters.prompt_tokens_total,
                            prev.prompt_tokens_total,
                            current_counters.prompt_seconds_total,
                            prev.prompt_seconds_total,
                        ),
                        counter_rate(
                            current_counters.predicted_tokens_total,
                            prev.predicted_tokens_total,
                            current_counters.predicted_seconds_total,
                            prev.predicted_seconds_total,
                        ),
                    )
                } else {
                    (0.0, 0.0)
                }
            };

            *previous_counters = Some(current_counters);
            *previous_counter_session = Some(session_id.to_string());

            snapshot.prompt_tokens_per_second = Some(prompt_tps);
            snapshot.generation_tokens_per_second = Some(gen_tps);
        }

        // Slots metrics
        let slots_req = if let Some(key) = api_key {
            client
                .get(format!("{base}/slots"))
                .header("Authorization", format!("Bearer {}", key))
        } else {
            client.get(format!("{base}/slots"))
        };

        if let Ok(resp) = slots_req.send().await
            && resp.status().is_success()
            && let Ok(body) = resp.text().await
            && let Some(slots) = parse_slot_metrics(&body)
        {
            let slot_details = serde_json::json!({
                "slots_idle": slots.slots_idle,
                "slots_processing": slots.slots_processing,
                "kv_cache_max": slots.kv_cache_max,
                "kv_cache_tokens": slots.kv_cache_tokens,
                "kv_cache_tokens_available": slots.kv_cache_tokens_available,
                "kv_cache_tokens_source": slots.kv_cache_tokens_source,
                "active_task_id": slots.active_task_id,
                "last_task_id": slots.last_task_id,
                "slot_generation_tokens": slots.slot_generation_tokens,
                "slot_generation_remaining": slots.slot_generation_remaining,
                "slot_generation_limit": slots.slot_generation_limit,
                "slot_generation_active": slots.slot_generation_active,
                "slot_generation_available": slots.slot_generation_available,
                "slot_prompt_processed": slots.slot_prompt_processed,
                "slot_prompt_total": slots.slot_prompt_total,
                "slot_prompt_progress": slots.slot_prompt_progress,
                "slots": slots.slots,
                "speculative_enabled": slots.speculative_enabled,
                "tokens_per_decode": tokens_per_decode,
                "n_busy_slots_per_decode": n_busy_slots_per_decode,
                "speculative_acceptance_rate": snapshot.speculative_acceptance_rate,
            });
            details.extend(slot_details.as_object().unwrap().clone());
        }

        runtime_cache.refresh(&client, base, api_key).await;
        snapshot.model = runtime_cache.facts.as_ref().and_then(|facts| {
            facts
                .model_alias
                .clone()
                .or_else(|| facts.model_name.clone())
        });
        // /slots gives per-request config; /props is the runtime default fallback
        // when slots are unavailable, not evidence inferred from old counters.
        let speculative_enabled = details
            .get("speculative_enabled")
            .and_then(|v| v.as_bool())
            .or(runtime_cache.speculative_enabled);
        details.insert(
            "speculative_enabled".into(),
            serde_json::json!(speculative_enabled),
        );
        details.insert(
            "runtime_facts".into(),
            serde_json::json!(runtime_cache.facts),
        );
        details.insert(
            "model_ctx_train".into(),
            serde_json::json!(runtime_cache.model_ctx_train),
        );
        details.insert(
            "tokens_per_decode".into(),
            serde_json::json!(tokens_per_decode),
        );
        details.insert(
            "n_busy_slots_per_decode".into(),
            serde_json::json!(n_busy_slots_per_decode),
        );
        details.insert(
            "speculative_acceptance_rate".into(),
            serde_json::json!(snapshot.speculative_acceptance_rate),
        );
        details.insert("telemetry_session_id".into(), serde_json::json!(session_id));
        details.insert("telemetry_endpoint".into(), serde_json::json!(endpoint_tag));
        snapshot.backend_details = Some(serde_json::Value::Object(details));

        Ok(snapshot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn telemetry_endpoint_tags_are_opaque_and_deterministic_for_sensitive_urls() {
        for (endpoint, expected) in [
            (
                "http://user:password@host:8001",
                "sha256:5060e47947dfab9807e4d705486d2967db47b6e3deb62061de46c2e58dc85526",
            ),
            (
                "http://host:8001/proxy/?api_key=secret",
                "sha256:caf0000ec0b93a2616855d1a63581960d2ccda36947d2fb2274f7f95a27b8052",
            ),
            (
                "http://host:8001#secret",
                "sha256:9093850d104d22ed397428f769c3802d7f82bde0c5004f3bab85a542a3789156",
            ),
            (
                "not a URL password",
                "sha256:ee685e3de5b0999f52cde3c28bb6d11472a826493dc9effd0958ada0cec67419",
            ),
        ] {
            for _ in 0..2 {
                let tag = telemetry_endpoint_tag(endpoint);
                assert_eq!(tag, expected);
                assert!(!tag.contains("secret"));
                assert!(!tag.contains("password"));
                assert!(!tag.contains("user"));
            }
        }
    }

    #[test]
    fn telemetry_endpoint_tags_normalize_path_slashes_not_query_value_slashes() {
        for endpoint in [
            "http://host:8001/proxy?api_key=secret",
            "http://host:8001/proxy/?api_key=secret",
            "http://host:8001/proxy///?api_key=secret",
        ] {
            assert_eq!(
                telemetry_endpoint_tag(endpoint),
                "sha256:caf0000ec0b93a2616855d1a63581960d2ccda36947d2fb2274f7f95a27b8052"
            );
        }
        assert_eq!(
            telemetry_endpoint_tag("http://host:8001/proxy/?api_key=secret/"),
            "sha256:54f24c7817c6ebaa5e4aaf5e8f16739b783fe8424fd76fff522e1dacf98c607a"
        );
        for endpoint in [
            "http://host:8001",
            "http://host:8001/",
            "http://host:8001///",
        ] {
            assert_eq!(telemetry_endpoint_tag(endpoint), "http://host:8001");
        }
        assert_eq!(
            telemetry_endpoint_tag("http://host:8001/proxy/"),
            "http://host:8001/proxy"
        );
    }

    #[test]
    fn telemetry_endpoint_tags_distinguish_query_path_credentials_and_server() {
        let endpoints = [
            "http://user:password@host:8001/proxy?api_key=secret",
            "http://user:password@host:8001/proxy?api_key=other",
            "http://user:password@host:8001/other?api_key=secret",
            "http://user:other@host:8001/proxy?api_key=secret",
            "http://other:password@host:8001/proxy?api_key=secret",
            "http://user:password@other:8001/proxy?api_key=secret",
            "http://user:password@host:8002/proxy?api_key=secret",
        ];
        let tags: std::collections::HashSet<_> =
            endpoints.into_iter().map(telemetry_endpoint_tag).collect();
        assert_eq!(tags.len(), 7);
    }

    #[tokio::test]
    async fn nonempty_slots_use_props_only_when_speculation_config_is_unavailable() {
        for (slots_body, props_body, expected) in [
            (
                r#"[{"id":0}]"#,
                r#"{"default_generation_settings":{"params":{"speculative.types":"draft-mtp"}}}"#,
                Some(true),
            ),
            (
                r#"[{"id":0,"speculative":false}]"#,
                r#"{"default_generation_settings":{"params":{"speculative.types":"draft-mtp"}}}"#,
                Some(false),
            ),
            (
                r#"[{"id":0,"params":{"speculative.type":"none"}}]"#,
                r#"{"default_generation_settings":{"params":{"speculative.types":"draft-mtp"}}}"#,
                Some(false),
            ),
            (r#"[{"id":0}]"#, "{}", None),
        ] {
            let mut server = mockito::Server::new_async().await;
            let slots = server
                .mock("GET", "/slots")
                .with_body(slots_body)
                .create_async()
                .await;
            let props = server
                .mock("GET", "/props")
                .with_body(props_body)
                .create_async()
                .await;
            let snapshot = poll_llama_cpp_metrics(
                &server.url(),
                None,
                "s",
                &mut None,
                &mut None,
                &mut LlamaRuntimeCache::default(),
            )
            .await
            .unwrap();
            let enabled = snapshot.backend_details.unwrap()["speculative_enabled"].as_bool();
            assert_eq!(
                enabled, expected,
                "slots: {slots_body}, props: {props_body}"
            );
            slots.assert_async().await;
            props.assert_async().await;
        }
    }

    #[test]
    fn runtime_facts_are_allowlisted_and_paths_are_reduced_to_basenames() {
        let props = serde_json::json!({
            "model_path": "C:\\private\\models\\attached.gguf",
            "model_ftype": "Q4_K_M",
            "build_info": "b1234-deadbeef",
            "modalities": {"vision": true, "audio": false, "bad": "/private"},
            "chat_template_caps": {"supports_tool_calls": true, "unknown_private_flag": true},
            "default_generation_settings": {"params": {"speculative.types": "none,draft-mtp"}},
            "chat_template": "private template", "total_slots": 8
        });
        let models = serde_json::json!({"data": [{
            "id": "attached-alias", "meta": {"n_params": 27000000000_u64, "ftype": "Q4_K_M"}
        }]});
        let facts = parse_runtime_facts(Some(&props), Some(&models)).unwrap();
        assert_eq!(facts.model_name.as_deref(), Some("attached.gguf"));
        assert_eq!(facts.quantization.as_deref(), Some("Q4_K_M"));
        assert_eq!(facts.model_params, Some(27000000000));
        assert_eq!(facts.server_build.as_deref(), Some("b1234-deadbeef"));
        assert_eq!(facts.capabilities.get("vision"), Some(&true));
        assert_eq!(facts.capabilities.get("audio"), Some(&false));
        assert_eq!(facts.capabilities.get("supports_tool_calls"), Some(&true));
        assert!(!facts.capabilities.contains_key("unknown_private_flag"));
        let json = serde_json::to_string(&facts).unwrap();
        assert!(!json.contains("private"));
        assert!(!json.contains("total_slots"));
        assert!(!json.contains("default_generation_settings"));
        assert_eq!(props_speculative_enabled(&props), Some(true));
    }

    #[test]
    fn runtime_parsing_tolerates_absent_and_malformed_fields() {
        assert!(parse_runtime_facts(None, None).is_none());
        let props = serde_json::json!({
            "model_alias": 42, "model_path": "/", "model_ftype": {},
            "build_info": [], "modalities": {"vision": "yes"}
        });
        assert!(parse_runtime_facts(Some(&props), None).is_none());
        for (value, expected) in [
            (serde_json::json!("none"), Some(false)),
            (serde_json::json!("none, draft-mtp"), Some(true)),
            (serde_json::json!(null), None),
            (serde_json::json!(["none"]), Some(false)),
            (serde_json::json!(["none", "ngram"]), Some(true)),
            (serde_json::json!(""), None),
        ] {
            let props = serde_json::json!({
                "default_generation_settings": {"params": {"speculative.types": value}}
            });
            assert_eq!(props_speculative_enabled(&props), expected);
        }
        assert_eq!(props_speculative_enabled(&serde_json::json!({})), None);
        let alias =
            serde_json::json!({"model_alias": "actual-alias", "model_path": "/secret/other.gguf"});
        let facts = parse_runtime_facts(Some(&alias), None).unwrap();
        assert_eq!(facts.model_alias.as_deref(), Some("actual-alias"));
        let alias_only = serde_json::json!({"model_alias": "alias-only"});
        let facts = parse_runtime_facts(Some(&alias_only), None).unwrap();
        assert!(facts.model_name.is_none());
        assert_eq!(facts.model_alias.as_deref(), Some("alias-only"));
        assert_eq!(
            parse_runtime_facts(Some(&alias), None)
                .unwrap()
                .model_name
                .as_deref(),
            Some("other.gguf")
        );
    }

    #[test]
    fn adapters_preserve_zero_scale_without_exposing_paths() {
        let adapters = parse_runtime_adapters(&serde_json::json!([
            {"id": 0, "path": "/private/lora/adapter.gguf", "scale": 0.0},
            {"id": 1, "path": "C:\\private\\second.gguf", "scale": 0.5},
            {"id": "invalid", "scale": "bad"}
        ]))
        .unwrap();
        assert_eq!(adapters.len(), 2);
        assert_eq!(adapters[0].name.as_deref(), Some("adapter.gguf"));
        assert_eq!(adapters[0].scale, Some(0.0));
        assert_eq!(adapters[1].name.as_deref(), Some("second.gguf"));
        assert!(
            !serde_json::to_string(&adapters)
                .unwrap()
                .contains("private")
        );
        assert_eq!(parse_runtime_adapters(&serde_json::json!([])), Some(vec![]));
        assert_eq!(parse_runtime_adapters(&serde_json::json!({})), None);
    }

    #[test]
    fn runtime_cache_is_scoped_to_endpoint_session_and_auth_and_throttles_failures() {
        let mut cache = LlamaRuntimeCache::default();
        let now = Instant::now();
        assert!(cache.select_target("http://a", Some("key-a"), "session-a"));
        cache.last_attempt = Some(now);
        cache.facts = Some(crate::llama::metrics::LlamaRuntimeFacts {
            model_name: Some("model-a".into()),
            ..Default::default()
        });
        cache.speculative_enabled = Some(true);
        assert!(!cache.select_target("http://a/", Some("key-a"), "session-a"));
        assert!(!cache.refresh_due(now + Duration::from_secs(59)));
        assert!(cache.refresh_due(now + Duration::from_secs(60)));
        for (base, key, session) in [
            ("http://b", Some("key-a"), "session-a"),
            ("http://b", Some("key-b"), "session-a"),
            ("http://b", Some("key-b"), "session-b"),
        ] {
            assert!(cache.select_target(base, key, session));
            assert!(cache.facts.is_none());
            assert!(cache.speculative_enabled.is_none());
            assert!(cache.refresh_due(now));
            cache.facts = Some(Default::default());
        }
    }

    #[tokio::test]
    async fn normalized_poll_keeps_efficiency_without_slots_and_caches_runtime_facts() {
        let mut server = mockito::Server::new_async().await;
        let metrics = server
            .mock("GET", "/metrics")
            .match_header("authorization", "Bearer test-key")
            .with_body(
                "llamacpp:prompt_tokens_total 100\n\
                llamacpp:prompt_tokens_cached_total 300\n\
                llamacpp:spec_decode_num_draft_tokens_total 20\n\
                llamacpp:spec_decode_num_accepted_tokens_total 10\n\
                llamacpp:spec_decode_num_drafts_total 5\n",
            )
            .expect(2)
            .create_async()
            .await;
        let props = server.mock("GET", "/props")
            .match_header("authorization", "Bearer test-key")
            .with_body(r#"{"model_alias":"actual","model_ftype":"Q4_K_M","build_info":"b1234",
                "is_sleeping":false,"default_generation_settings":{"params":{"speculative.type":"ngram"}}}"#)
            .expect(1).create_async().await;
        let models = server
            .mock("GET", "/v1/models")
            .match_header("authorization", "Bearer test-key")
            .with_body(
                r#"{"data":[{"id":"actual","meta":{"n_params":123456,"n_ctx_train":8192}}]}"#,
            )
            .expect(1)
            .create_async()
            .await;
        let adapters = server
            .mock("GET", "/lora-adapters")
            .match_header("authorization", "Bearer test-key")
            .with_body(r#"[{"id":0,"path":"/private/adapter.gguf","scale":0}]"#)
            .expect(1)
            .create_async()
            .await;
        let mut cache = LlamaRuntimeCache::default();
        let mut counters = None;
        let mut session = None;
        for _ in 0..2 {
            let snapshot = poll_llama_cpp_metrics(
                &server.url(),
                Some("test-key"),
                "s",
                &mut counters,
                &mut session,
                &mut cache,
            )
            .await
            .unwrap();
            assert_eq!(snapshot.prompt_tokens_total, Some(100));
            assert_eq!(snapshot.speculative_acceptance_rate, Some(0.5));
            assert_eq!(snapshot.model.as_deref(), Some("actual"));
            let details = snapshot.backend_details.unwrap();
            assert_eq!(details["prompt_tokens_processed_total"], 100.0);
            assert_eq!(details["prompt_tokens_cached_total"], 300.0);
            assert_eq!(details["speculative_verification_steps_total"], 5);
            assert_eq!(details["speculative_enabled"], true);
            assert_eq!(details["runtime_facts"]["model_params"], 123456);
            assert_eq!(
                details["runtime_facts"]["adapters"][0]["name"],
                "adapter.gguf"
            );
            assert_eq!(details["runtime_facts"]["adapters"][0]["scale"], 0.0);
            assert!(!details.to_string().contains("private"));
            assert!(details.get("slots").is_none());
        }
        metrics.assert_async().await;
        props.assert_async().await;
        models.assert_async().await;
        adapters.assert_async().await;
    }

    #[tokio::test]
    async fn auth_switch_resets_facts_and_counters_before_unavailable_runtime_poll() {
        let mut server = mockito::Server::new_async().await;
        let props = server
            .mock("GET", "/props")
            .match_header("authorization", "Bearer first-key")
            .with_body(r#"{"model_alias":"old","is_sleeping":true}"#)
            .expect(1)
            .create_async()
            .await;
        let metrics = server
            .mock("GET", "/metrics")
            .match_header("authorization", "Bearer first-key")
            .with_body("llamacpp:prompt_tokens_total 42\n")
            .create_async()
            .await;
        let unavailable = server
            .mock("GET", "/props")
            .match_header("authorization", "Bearer second-key")
            .with_status(403)
            .with_body(r#"{"model_alias":"must-not-leak"}"#)
            .expect(1)
            .create_async()
            .await;
        let adapters = server
            .mock("GET", "/lora-adapters")
            .expect(0)
            .create_async()
            .await;
        let mut cache = LlamaRuntimeCache::default();
        let mut counters = None;
        let mut session = None;
        let first = poll_llama_cpp_metrics(
            &server.url(),
            Some("first-key"),
            "s",
            &mut counters,
            &mut session,
            &mut cache,
        )
        .await
        .unwrap();
        assert_eq!(first.model.as_deref(), Some("old"));
        assert!(counters.is_some());
        assert!(first.backend_details.unwrap()["runtime_facts"]["adapters"].is_null());
        for _ in 0..2 {
            let next = poll_llama_cpp_metrics(
                &server.url(),
                Some("second-key"),
                "s",
                &mut counters,
                &mut session,
                &mut cache,
            )
            .await
            .unwrap();
            assert!(next.model.is_none());
            assert!(next.prompt_tokens_total.is_none());
            assert!(next.backend_details.as_ref().unwrap()["runtime_facts"].is_null());
            assert!(counters.is_none());
        }
        props.assert_async().await;
        metrics.assert_async().await;
        unavailable.assert_async().await;
        adapters.assert_async().await;
    }

    #[tokio::test]
    async fn runtime_request_timeout_degrades_without_failing_the_poll() {
        use warp::Filter;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let route = warp::any().and_then(|| async {
            tokio::time::sleep(Duration::from_secs(10)).await;
            Ok::<_, std::convert::Infallible>(warp::reply::json(
                &serde_json::json!({"model_alias":"late"}),
            ))
        });
        let task = tokio::spawn(warp::serve(route).incoming(listener).run());
        let result = optional_runtime_json(&Client::new(), &base, "/props", None).await;
        task.abort();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn oversized_optional_runtime_body_is_omitted() {
        let mut server = mockito::Server::new_async().await;
        let body = format!(r#"{{"model_alias":"{}"}}"#, "x".repeat(1024 * 1024));
        let response = server
            .mock("GET", "/props")
            .with_body(body)
            .create_async()
            .await;
        assert!(
            optional_runtime_json(&Client::new(), &server.url(), "/props", None)
                .await
                .is_none()
        );
        response.assert_async().await;
    }

    #[test]
    fn build_facts_reject_path_bearing_strings_and_unknown_shapes() {
        for value in [
            serde_json::json!("/private/bin/llama-server b123"),
            serde_json::json!("C:\\private\\llama-server.exe"),
            serde_json::json!({"version": "unknown schema"}),
        ] {
            let facts = parse_runtime_facts(
                Some(&serde_json::json!({
                    "model_alias": "model", "build_info": value, "model_ftype": 15
                })),
                None,
            )
            .unwrap();
            assert_eq!(facts.server_build, None);
            assert_eq!(facts.quantization, None);
        }
    }

    #[tokio::test]
    async fn expired_facts_are_cleared_on_failed_refresh_and_failures_are_throttled() {
        let mut server = mockito::Server::new_async().await;
        let props = server
            .mock("GET", "/props")
            .with_status(404)
            .expect(1)
            .create_async()
            .await;
        let mut cache = LlamaRuntimeCache::default();
        cache.select_target(&server.url(), None, "s");
        cache.last_attempt = Some(Instant::now() - Duration::from_secs(60));
        cache.facts = Some(LlamaRuntimeFacts {
            model_name: Some("old".into()),
            ..Default::default()
        });
        cache.speculative_enabled = Some(true);
        cache.model_ctx_train = Some(8192);
        cache.refresh(&Client::new(), &server.url(), None).await;
        assert!(cache.facts.is_none());
        assert!(cache.speculative_enabled.is_none());
        assert!(cache.model_ctx_train.is_none());
        cache.refresh(&Client::new(), &server.url(), None).await;
        props.assert_async().await;
    }

    #[test]
    fn runtime_adapter_list_is_bounded() {
        let value =
            serde_json::Value::Array((0..65).map(|id| serde_json::json!({"id": id})).collect());
        let adapters = parse_runtime_adapters(&value).unwrap();
        assert_eq!(adapters.len(), 64);
        assert_eq!(adapters.last().unwrap().id, Some(63));
    }

    async fn launch_args(config: ServerConfig) -> Vec<String> {
        let config_dir = tempfile::tempdir().unwrap();
        let args = crate::cli::AppArgs::parse_from([
            "llama-monitor",
            "--config-dir",
            config_dir.path().to_str().unwrap(),
            "--llama-server-path",
            "llama-server",
            "--gpu-backend",
            "none",
        ]);
        let adapter = LlamaCppAdapter::new(AppConfig::from_args(args), config, GpuEnv::default());
        adapter
            .build_launch()
            .await
            .unwrap()
            .args
            .into_iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect()
    }

    fn typed_capabilities(flags: &[&str]) -> CapabilitySnapshot {
        CapabilitySnapshot {
            executable_identity: crate::inference::llama_cpp_capabilities::ExecutableIdentity {
                path: "/tmp/llama-server".into(),
                file_hash: "phase2-test".into(),
                file_mtime_unix: 0,
            },
            version_text: "test".into(),
            help_hash: "test-help".into(),
            serve_flags: flags.iter().map(|flag| (*flag).to_string()).collect(),
            cache: Default::default(),
            context: Default::default(),
            concurrency: Default::default(),
            endpoints: Default::default(),
            streaming: Default::default(),
            templates: Default::default(),
            tools: Default::default(),
            speculation: Default::default(),
            typed: Default::default(),
            mixed_main_kv:
                crate::inference::llama_cpp_capabilities::MixedMainKv::product_default_denied(),
            evidence_timestamp: 0,
            source:
                crate::inference::llama_cpp_capabilities::CapabilitySnapshotSource::ManualOverride,
        }
    }

    async fn launch_args_with_capabilities(
        config: ServerConfig,
        capabilities: CapabilitySnapshot,
    ) -> anyhow::Result<Vec<String>> {
        let config_dir = tempfile::tempdir().unwrap();
        let args = crate::cli::AppArgs::parse_from([
            "llama-monitor",
            "--config-dir",
            config_dir.path().to_str().unwrap(),
            "--llama-server-path",
            "llama-server",
            "--gpu-backend",
            "none",
        ]);
        let adapter = LlamaCppAdapter::new_with_capabilities(
            AppConfig::from_args(args),
            config,
            GpuEnv::default(),
            Some(capabilities),
        );
        Ok(adapter
            .build_launch()
            .await?
            .args
            .into_iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect())
    }

    #[test]
    fn readiness_uses_loopback_for_wildcard_bind_hosts() {
        assert_eq!(readiness_host(None), "127.0.0.1");
        assert_eq!(readiness_host(Some("0.0.0.0")), "127.0.0.1");
        assert_eq!(readiness_host(Some("::")), "127.0.0.1");
        assert_eq!(readiness_host(Some("192.168.1.10")), "192.168.1.10");
    }

    #[test]
    fn launch_environment_preserves_gpu_selection_and_custom_values() {
        let gpu_env = GpuEnv {
            devices: "1,2".into(),
            extra_env: vec![("LLAMA_TEST_ENV".into(), "present".into())],
            ..Default::default()
        };

        let nvidia = launch_environment("nvidia", &gpu_env, "/tmp/llama");
        assert!(nvidia.contains(&("CUDA_VISIBLE_DEVICES".into(), "1,2".into())));
        assert!(nvidia.contains(&("LLAMA_TEST_ENV".into(), "present".into())));
        assert!(launch_environment("none", &gpu_env, "/tmp/llama").is_empty());
    }

    #[test]
    fn server_config_keeps_spawn_v2_cache_fields() {
        let mut value = serde_json::to_value(ServerConfig::default()).unwrap();
        let object = value.as_object_mut().unwrap();
        object.insert("cache_type_k".into(), serde_json::json!("q8_0"));
        object.insert("cache_type_v".into(), serde_json::json!("q4_0"));
        let config: ServerConfig = serde_json::from_value(value).unwrap();

        assert_eq!(config.cache_type_k.as_deref(), Some("q8_0"));
        assert_eq!(config.cache_type_v.as_deref(), Some("q4_0"));
    }

    #[tokio::test]
    async fn default_launch_argv_omits_experimental_webui_mcp_proxy() {
        let args = launch_args(ServerConfig {
            model_path: "/models/test.gguf".into(),
            ctk: "q8_0".into(),
            ctv: "q8_0".into(),
            port: 8080,
            ..Default::default()
        })
        .await;

        let mut expected: Vec<&str> = vec![
            "-m",
            "/models/test.gguf",
            "-ngl",
            "all",
            "-ctk",
            "q8_0",
            "-ctv",
            "q8_0",
            "--host",
            "127.0.0.1",
            "--port",
            "8080",
            "--no-warmup",
            "--jinja",
            "--no-context-shift",
            "--ctx-checkpoints",
            "32",
            "--keep",
            "-1",
            "-fa",
            "on",
            "--metrics",
        ];
        // macOS has no --cache-ram support; it is forced to 0 always.
        if cfg!(target_os = "macos") {
            expected.extend_from_slice(&["--cache-ram", "0"]);
        }
        assert_eq!(
            args,
            expected.iter().map(|s| s.to_string()).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn draft_all_maps_to_named_llama_server_value() {
        let args = launch_args(ServerConfig {
            model_path: "/models/mtp.gguf".into(),
            port: 8080,
            spec: SpecDecodeConfig {
                spec_draft_ngl: Some(-1),
                ..Default::default()
            },
            ..Default::default()
        })
        .await;

        assert!(
            args.windows(2)
                .any(|pair| pair == ["--spec-draft-ngl", "all"])
        );
        assert!(
            !args
                .windows(2)
                .any(|pair| pair == ["--spec-draft-ngl", "-1"])
        );
    }

    #[tokio::test]
    async fn verbosity_is_emitted_for_server_launch() {
        let args = launch_args(ServerConfig {
            model_path: "/models/test.gguf".into(),
            port: 8080,
            verbosity: Some(4),
            ..Default::default()
        })
        .await;
        assert!(args.windows(2).any(|pair| pair == ["-lv", "4"]));
    }

    #[tokio::test]
    async fn repeat_last_n_is_emitted_for_server_launch() {
        let args = launch_args(ServerConfig {
            model_path: "/models/test.gguf".into(),
            port: 8080,
            repeat_last_n: Some(64),
            ..Default::default()
        })
        .await;
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--repeat-last-n", "64"])
        );
    }

    #[tokio::test]
    async fn explicit_load_mode_replaces_legacy_mmap_flags() {
        let args = launch_args(ServerConfig {
            model_path: "/models/test.gguf".into(),
            port: 8080,
            no_mmap: true,
            mlock: true,
            load_mode: Some(LoadMode::MmapMlock),
            ..Default::default()
        })
        .await;
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--load-mode", "mmap+mlock"])
        );
        assert!(!args.iter().any(|arg| arg == "--no-mmap"));
        assert!(!args.iter().any(|arg| arg == "--mlock"));
    }

    /// Phase 10a Windows-safety requirement: argv is built via `Command::arg`,
    /// which never invokes a shell, so a Windows-style path must survive into
    /// argv byte-for-byte — no backslash escaping/doubling and no forward-
    /// slash normalization, either of which would corrupt the real path.
    #[tokio::test]
    async fn windows_style_model_path_passes_through_argv_unmodified() {
        let windows_path = r"C:\Users\test\models\model.gguf";
        let args = launch_args(ServerConfig {
            model_path: windows_path.into(),
            port: 8080,
            ..Default::default()
        })
        .await;

        let m_index = args.iter().position(|a| a == "-m").expect("-m present");
        assert_eq!(args[m_index + 1], windows_path);
    }

    #[tokio::test]
    async fn empty_ctk_and_ctv_are_omitted_rather_than_passed_as_empty_strings() {
        let args = launch_args(ServerConfig {
            model_path: "/models/test.gguf".into(),
            port: 8080,
            ..Default::default()
        })
        .await;

        assert!(!args.iter().any(|a| a == "-ctk"));
        assert!(!args.iter().any(|a| a == "-ctv"));
    }

    #[tokio::test]
    async fn phase2_typed_reasoning_and_mmproj_flags_emit_exact_argv() {
        let capabilities = typed_capabilities(&[
            "--mmproj-offload",
            "--no-mmproj-offload",
            "--reasoning-effort",
            "--reasoning-format",
            "--reasoning-preserve",
            "--no-reasoning-preserve",
        ]);
        let args = launch_args_with_capabilities(
            ServerConfig {
                model_path: "/models/test.gguf".into(),
                port: 8080,
                mmproj_offload: Some(true),
                llama_reasoning_effort: LlamaReasoningEffort::High,
                llama_reasoning_format: Some(LlamaReasoningFormat::None),
                llama_reasoning_preserve: Some(false),
                ..Default::default()
            },
            capabilities,
        )
        .await
        .unwrap();
        assert!(args.iter().any(|arg| arg == "--mmproj-offload"));
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--reasoning-effort", "high"])
        );
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--reasoning-format", "none"])
        );
        assert!(args.iter().any(|arg| arg == "--no-reasoning-preserve"));
        assert!(!args.iter().any(|arg| arg == "auto"));
        assert!(!args.iter().any(|arg| arg == "true"));
        assert!(!args.iter().any(|arg| arg == "false"));
    }

    #[tokio::test]
    async fn phase2_typed_values_fail_closed_without_capability_evidence() {
        let adapter = LlamaCppAdapter::new(
            AppConfig::from_args(crate::cli::AppArgs::parse_from([
                "llama-monitor",
                "--config-dir",
                tempfile::tempdir().unwrap().path().to_str().unwrap(),
                "--llama-server-path",
                "llama-server",
                "--gpu-backend",
                "none",
            ])),
            ServerConfig {
                model_path: "/models/test.gguf".into(),
                port: 8080,
                llama_reasoning_effort: LlamaReasoningEffort::Low,
                ..Default::default()
            },
            GpuEnv::default(),
        );
        let error = adapter.build_launch().await.unwrap_err().to_string();
        assert!(error.contains("capability snapshot"));
    }

    #[tokio::test]
    async fn phase2_reasoning_preserve_launches_when_binary_advertises_the_flag() {
        // Template compatibility for --reasoning-preserve cannot be verified up
        // front (llama.cpp's own 'supports_preserve_reasoning' marker isn't
        // reliably present in real GGUF chat templates), so an unsupported
        // template is expected to honor-or-ignore the flag at runtime rather
        // than be blocked here.
        let capabilities = typed_capabilities(&["--reasoning-preserve"]);
        let args = launch_args_with_capabilities(
            ServerConfig {
                model_path: "/models/test.gguf".into(),
                port: 8080,
                reasoning: Some("on".into()),
                llama_reasoning_preserve: Some(true),
                ..Default::default()
            },
            capabilities,
        )
        .await
        .unwrap();
        assert!(
            args.iter().any(|arg| arg == "--reasoning-preserve"),
            "{args:?}"
        );
    }

    #[test]
    fn phase2_reasoning_effort_values_round_trip_through_serde() {
        for wire in [
            "default", "minimal", "low", "medium", "high", "xhigh", "max",
        ] {
            let encoded = format!("\"{wire}\"");
            let value: LlamaReasoningEffort = serde_json::from_str(&encoded).unwrap();
            assert_eq!(serde_json::to_string(&value).unwrap(), encoded);
        }
    }

    #[test]
    fn phase2_unknown_reasoning_values_round_trip_but_remain_unknown() {
        let effort: LlamaReasoningEffort = serde_json::from_str("\"future\"").unwrap();
        let format: LlamaReasoningFormat = serde_json::from_str("\"future-format\"").unwrap();
        assert!(matches!(&effort, LlamaReasoningEffort::Unknown(value) if value == "future"));
        assert!(
            matches!(&format, LlamaReasoningFormat::Unknown(value) if value == "future-format")
        );
        assert_eq!(serde_json::to_string(&effort).unwrap(), "\"future\"");
        assert_eq!(serde_json::to_string(&format).unwrap(), "\"future-format\"");
    }

    #[tokio::test]
    async fn phase2_mmproj_none_and_false_have_distinct_argv_behavior() {
        let absent = launch_args(ServerConfig {
            model_path: "/models/test.gguf".into(),
            ..Default::default()
        })
        .await;
        assert!(!absent.iter().any(|arg| arg == "--mmproj-offload"));
        assert!(!absent.iter().any(|arg| arg == "--no-mmproj-offload"));

        let disabled = launch_args_with_capabilities(
            ServerConfig {
                model_path: "/models/test.gguf".into(),
                mmproj_offload: Some(false),
                ..Default::default()
            },
            typed_capabilities(&["--no-mmproj-offload"]),
        )
        .await
        .unwrap();
        assert!(disabled.iter().any(|arg| arg == "--no-mmproj-offload"));
        assert!(!disabled.iter().any(|arg| arg == "--mmproj-offload"));
    }

    #[tokio::test]
    async fn phase2_outer_none_reasoning_format_emits_no_auto_argument() {
        let args = launch_args(ServerConfig {
            model_path: "/models/test.gguf".into(),
            llama_reasoning_format: None,
            ..Default::default()
        })
        .await;
        assert!(!args.iter().any(|arg| arg == "--reasoning-format"));
        assert!(!args.iter().any(|arg| arg == "auto"));
    }

    #[tokio::test]
    async fn phase2_unknown_reasoning_format_fails_closed_at_launch() {
        let error = launch_args_with_capabilities(
            ServerConfig {
                model_path: "/models/test.gguf".into(),
                llama_reasoning_format: Some(LlamaReasoningFormat::Unknown("future".into())),
                ..Default::default()
            },
            typed_capabilities(&["--reasoning-format"]),
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("unknown llama.cpp reasoning format"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn optional_launch_argv_preserves_order_and_values() {
        let args = launch_args(ServerConfig {
            model_path: "/models/full.gguf".into(),
            context_size: 4096,
            ctk: "q4_0".into(),
            ctv: "q5_0".into(),
            tensor_split: "3,1".into(),
            batch_size: 512,
            ubatch_size: 128,
            no_mmap: true,
            port: 9090,
            parallel_slots: 2,
            temperature: Some(0.7),
            top_p: Some(0.95),
            top_k: Some(40),
            min_p: Some(0.05),
            repeat_penalty: Some(1.1),
            presence_penalty: Some(0.2),
            n_cpu_moe: Some(4),
            gpu_layers: Some(42),
            mlock: true,
            flash_attn: "off".into(),
            split_mode: "layer".into(),
            main_gpu: Some(1),
            threads: Some(8),
            threads_batch: Some(12),
            prio: Some(2),
            prio_batch: Some(3),
            rope_scaling: "yarn".into(),
            rope_freq_base: Some(10_000.0),
            rope_freq_scale: Some(0.5),
            kv_unified: Some(true),
            cache_idle_slots: Some(true),
            cache_ram_mib: Some(2048),
            fit_enabled: Some(true),
            fit_target: Some("3072".into()),
            seed: Some(7),
            system_prompt_file: "/prompts/system.txt".into(),
            extra_args: "--verbose --log-colors off".into(),
            bind_host: Some("0.0.0.0".into()),
            alias: Some("full-model".into()),
            chat_template_file: Some("/templates/chat.jinja".into()),
            mmproj: Some("/models/mmproj.gguf".into()),
            grammar: Some("root ::= answer".into()),
            json_schema: Some("{\"type\":\"object\"}".into()),
            max_tokens: Some(256),
            api_key: Some("secret".into()),
            reasoning: Some("auto".into()),
            reasoning_budget: Some(512),
            reasoning_budget_message: Some("done".into()),
            image_min_tokens: Some(280),
            image_max_tokens: Some(560),
            ..Default::default()
        })
        .await;

        let mut expected_tail: Vec<&str> = vec![
            "--api-key",
            "secret",
            "--alias",
            "full-model",
            "--kv-unified",
            "--fit",
            "on",
            "--fit-target",
            "3072",
            "--verbose",
            "--log-colors",
            "off",
        ];
        // On macOS, --cache-ram is forced to 0 and --cache-idle-slots is
        // suppressed because it requires cache-ram to be nonzero.
        if cfg!(target_os = "macos") {
            expected_tail.splice(5..5, ["--cache-ram", "0"]);
        } else {
            expected_tail.splice(5..5, ["--cache-idle-slots", "--cache-ram", "2048"]);
        }
        assert!(
            args.ends_with(
                &expected_tail
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
            )
        );
        for required in [
            "--no-mmap",
            "--mlock",
            "--split-mode",
            "--rope-scaling",
            "--parallel",
            "--chat-template-file",
            "--reasoning-budget",
            "--mmproj",
            "--grammar",
            "--json-schema",
            "--image-min-tokens",
            "--image-max-tokens",
        ] {
            assert!(args.iter().any(|arg| arg == required), "missing {required}");
        }
    }

    #[tokio::test]
    async fn cache_ram_sentinels_preserve_llama_server_semantics() {
        for (cache_ram_mib, idle_slot_cache_expected) in [(0, false), (2048, true), (-1, true)] {
            let args = launch_args(ServerConfig {
                model_path: "/models/test.gguf".into(),
                cache_ram_mib: Some(cache_ram_mib),
                cache_idle_slots: Some(true),
                ..Default::default()
            })
            .await;

            // macOS has no --cache-ram support; the value is forced to 0.
            if cfg!(target_os = "macos") {
                assert!(
                    args.windows(2).any(|pair| pair == ["--cache-ram", "0"]),
                    "macOS must emit --cache-ram 0, got {args:?}"
                );
                assert!(
                    !args.iter().any(|arg| arg == "--cache-idle-slots"),
                    "macOS must suppress --cache-idle-slots"
                );
            } else {
                assert!(
                    args.windows(2).any(|pair| {
                        pair == ["--cache-ram", cache_ram_mib.to_string().as_str()]
                    })
                );
                assert_eq!(
                    args.iter().any(|arg| arg == "--cache-idle-slots"),
                    idle_slot_cache_expected,
                    "cache_ram_mib={cache_ram_mib}"
                );
            }
        }
    }

    /// Plan §Phase 1b: a preset storing cache_ram_mib: Some(16384) must emit
    /// `--cache-ram 0` on macOS and the configured value elsewhere, and the
    /// macOS branch must also suppress --cache-idle-slots.
    #[tokio::test]
    async fn stored_cache_ram_16384_is_zeroed_on_macos_passthrough_elsewhere() {
        let args = launch_args(ServerConfig {
            model_path: "/models/test.gguf".into(),
            cache_ram_mib: Some(16384),
            cache_mode: CacheMode::Custom,
            cache_idle_slots: Some(true),
            ..Default::default()
        })
        .await;

        if cfg!(target_os = "macos") {
            assert!(
                args.windows(2).any(|pair| pair == ["--cache-ram", "0"]),
                "macOS must emit --cache-ram 0 regardless of stored value, got {args:?}"
            );
            assert!(
                !args.iter().any(|arg| arg == "--cache-idle-slots"),
                "macOS must suppress --cache-idle-slots (it requires cache-ram)"
            );
        } else {
            assert!(
                args.windows(2).any(|pair| pair == ["--cache-ram", "16384"]),
                "non-macOS must pass the configured value through, got {args:?}"
            );
            assert!(
                args.iter().any(|arg| arg == "--cache-idle-slots"),
                "non-macOS must keep --cache-idle-slots when cache-ram is nonzero"
            );
        }
    }

    #[test]
    fn cache_mode_custom_preserves_configured_value_untouched() {
        assert_eq!(CacheMode::Custom.resolve(Some(4096)), Some(4096));
        assert_eq!(CacheMode::Custom.resolve(None), None);
    }

    #[test]
    fn effective_cache_ram_is_zero_on_macos_passthrough_elsewhere() {
        // Direct helper test: on macOS the value is always 0 regardless of
        // stored value or mode. On other platforms it passes through.
        let result = effective_cache_ram(Some(16384), CacheMode::Custom);
        if cfg!(target_os = "macos") {
            assert_eq!(result, Some(0));
            let result2 = effective_cache_ram(Some(-1), CacheMode::Off);
            assert_eq!(result2, Some(0));
        } else {
            assert_eq!(result, Some(16384));
            let result2 = effective_cache_ram(Some(4096), CacheMode::Auto);
            assert_eq!(result2, Some(0));
        }
    }

    #[test]
    fn cache_mode_auto_and_off_both_disable_in_this_scoped_pass() {
        assert_eq!(CacheMode::Auto.resolve(Some(4096)), Some(0));
        assert_eq!(CacheMode::Off.resolve(Some(4096)), Some(0));
    }

    #[test]
    fn cache_mode_serde_default_is_custom_for_backward_compatibility() {
        assert_eq!(CacheMode::default(), CacheMode::Custom);
    }

    #[tokio::test]
    async fn cache_mode_auto_overrides_configured_cache_ram_mib_at_launch() {
        let args = launch_args(ServerConfig {
            model_path: "/models/test.gguf".into(),
            cache_ram_mib: Some(4096),
            cache_mode: CacheMode::Auto,
            cache_idle_slots: Some(true),
            ..Default::default()
        })
        .await;

        assert!(args.windows(2).any(|pair| pair == ["--cache-ram", "0"]));
        assert!(!args.iter().any(|arg| arg == "--cache-idle-slots"));
    }
}
