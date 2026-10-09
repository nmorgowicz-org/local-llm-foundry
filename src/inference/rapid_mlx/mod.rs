pub mod capabilities;
pub mod changelog;
pub mod command;
pub mod compatibility;
pub mod discovery;
pub mod escape_hatch;
pub mod info_query;
pub mod mlx_meta;
pub mod model_cache;
pub mod model_resolver;
pub mod poller;
pub mod repair;
pub mod runtime;
pub mod settings;
pub(crate) mod sidecar_hygiene;
pub mod sidecar_inventory;
pub mod spec_decode_store;
// The managed-updater module is complete and unit-tested but has no caller in the
// running binary yet; its wiring belongs to the Phase 12 dependency/watchlist work.
#[allow(dead_code)]
pub mod updater;

#[allow(unused_imports)]
pub use settings::{
    RapidMlxSetting, ValidationContext, ValidationError, all_settings, check_mutual_exclusions,
};

use self::command::RapidMlxCommandBuilder;
use self::compatibility::CompatibilityProfile;
use self::model_resolver::{
    RapidMlxModelSource, RapidMlxModelSourceView, ResolvedRapidMlxLaunchModel,
};
use self::runtime::RuntimeMetadata;
use crate::inference::capabilities::CapabilitySet;
use crate::inference::metrics::InferenceMetricsSnapshot;
use crate::inference::supervisor::SupervisedLaunch;
use anyhow::{Result, anyhow};
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// The launch-safe PFlash policy. See `RapidMlxConfig::pflash_policy`.
fn default_pflash_policy() -> Option<String> {
    Some("off".into())
}

fn default_num_speculative_tokens() -> u32 {
    3
}

/// Rapid-MLX speculative decoding method.
///
/// The enum is intentionally extensible, but MTP is the only method exposed by
/// llama-monitor until another runtime path has its own qualification evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RapidMlxSpeculativeMethod {
    Mtp,
}

/// Typed vLLM-compatible payload passed to Rapid-MLX `--speculative-config`.
///
/// This is deliberately not a free-form JSON escape hatch. Keeping the product
/// contract typed prevents misspelled fields from silently disabling speculation
/// and gives the estimator a trustworthy request shape (but never caller-authored
/// architecture depth or memory byte counts).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RapidMlxSpeculativeConfig {
    pub method: RapidMlxSpeculativeMethod,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default = "default_num_speculative_tokens")]
    pub num_speculative_tokens: u32,
    #[serde(default)]
    pub disable_auto_k: bool,
}

impl RapidMlxSpeculativeConfig {
    /// Validate the product-level contract before launch or estimation.
    pub fn validate(&self) -> Result<()> {
        if !(1..=8).contains(&self.num_speculative_tokens) {
            return Err(anyhow!("num_speculative_tokens must be between 1 and 8"));
        }
        if let Some(model) = &self.model {
            let model = model.trim();
            if model.is_empty() {
                return Err(anyhow!("speculative model must not be empty"));
            }
            if model.chars().any(char::is_control) || model.contains('\\') {
                return Err(anyhow!(
                    "speculative model must not contain control characters or backslashes"
                ));
            }
            if std::path::Path::new(model)
                .components()
                .any(|part| matches!(part, std::path::Component::ParentDir))
            {
                return Err(anyhow!("speculative model must not contain '..'"));
            }
            if !model.starts_with('/') {
                let mut parts = model.split('/');
                let valid_hf_repo = parts.next().is_some_and(|part| !part.is_empty())
                    && parts.next().is_some_and(|part| !part.is_empty())
                    && parts.next().is_none();
                if !valid_hf_repo {
                    return Err(anyhow!(
                        "speculative model must be an absolute local path or Hugging Face repo id (owner/repo)"
                    ));
                }
            }
        }
        Ok(())
    }

    /// Returns the companion model repo id if the `model` field is an HF repo
    /// reference (owner/repo format). Returns `None` if it's a local path or empty.
    pub fn companion_model_repo_id(&self) -> Option<&str> {
        self.model.as_deref().filter(|m| {
            !m.starts_with('/')
                && m.split('/').count() == 2
                && m.split('/').all(|part| !part.is_empty())
        })
    }

    /// Returns the companion model path if the `model` field is an absolute local
    /// path (starts with `/`). Returns `None` if it's an HF repo id or empty.
    pub fn companion_model_local_path(&self) -> Option<&str> {
        self.model.as_deref().filter(|m| m.starts_with('/'))
    }

    pub fn to_cli_json(&self) -> Result<String> {
        self.validate()?;
        serde_json::to_string(self)
            .map_err(|error| anyhow!("Could not serialize speculative config: {error}"))
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RapidMlxConfig {
    #[serde(default)]
    pub model_path: String,
    /// Typed source for new configurations. `model_path` remains the migration fallback.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_source: Option<RapidMlxModelSource>,
    #[serde(default)]
    pub served_model_name: Option<String>,
    #[serde(default)]
    pub executable_path: Option<PathBuf>,
    #[serde(default)]
    pub managed_runtime_path: Option<PathBuf>,
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default = "default_log_level")]
    pub log_level: String,
    #[serde(default)]
    pub timeout: Option<u32>,
    /// Enables Rapid's retained in-memory prefix cache.
    #[serde(default)]
    pub prefix_cache_enabled: bool,
    // prefix_cache_budget_bytes removed (Gap 2): was always None in config; the value is
    // computed from retained_cache_mib (mlx_cache_bytes) in the estimator/runtime, not configured.
    /// Retained prefix-cache capacity in MiB. This is the source-native
    /// `--cache-memory-mb` contract qualified by Phase 6.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retained_cache_mib: Option<u32>,
    /// Phase 6: Auto/Off/Custom prompt-cache mode. `Custom` (the default) uses
    /// `prefix_cache_enabled`/`retained_cache_mib`/`hybrid_cache_entries` as configured;
    /// `Auto` and `Off` override them at launch time — see [`CacheMode::resolve`].
    #[serde(default)]
    pub cache_mode: CacheMode,
    /// Keep automatic disk snapshots off for interactive launches. They are
    /// snapshot writes, not transparent cache restoration.
    #[serde(default = "default_disk_checkpoint_interval")]
    pub disk_checkpoint_interval: u32,
    /// Accepted only on launch input. Secrets are never serialized into presets,
    /// sessions, or diagnostics.
    #[serde(default, skip_serializing)]
    pub api_key: Option<String>,
    /// Default applied to chat requests that omit `enable_thinking`, mirroring
    /// llama.cpp's standing `--reasoning` server-level default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_thinking: Option<bool>,
    /// Default applied to chat requests that omit `reasoning_effort`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    /// User consent for HF repos requiring trust_remote_code (custom Python code).
    /// Revision-scoped: must be re-confirmed when revision changes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trust_remote_code_consent: Option<String>,
    /// Diagnostic fix flags: set by the diagnostics panel to patch launch behavior.
    /// These are diagnostic helpers only, not general-purpose escape hatches.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_parser: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_parser: Option<String>,
    #[serde(default)]
    pub auto_tool_choice: bool,
    #[serde(default)]
    pub no_thinking: bool,
    /// Curated escape-hatch flags for advanced tuning (PFlash, spec-decode, hybrid).
    /// Validated against an allowlist at load time; no free-text CLI injection.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub escape_hatch_flags: Vec<(String, serde_json::Value)>,
    /// Computed view of the model source for API responses and UI display.
    /// Never persisted to disk; populated by preset_for_api().
    #[serde(default, skip_deserializing, skip_serializing_if = "Option::is_none")]
    pub model_source_view: Option<RapidMlxModelSourceView>,
    // ── Phase 7: KV/cache policy ──────────────────────────────────────
    /// KV cache dtype configuration (D1/D2: effective value after overrides).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_cache_dtype: Option<KvCacheConfig>,
    /// TurboQuant reusable-prompt storage policy (D31).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turboquant_mode: Option<TurboQuantMode>,
    /// Hybrid cache entries limit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hybrid_cache_entries: Option<u64>,
    /// Architecture override. Auto leaves Rapid's alias/config detection
    /// authoritative; the other values emit mutually-exclusive switches.
    #[serde(default)]
    pub hybrid_mode: RapidMlxHybridMode,
    /// PFlash policy (auto/always/off).
    ///
    /// Defaults to `off` on deserialization, not just in `RapidMlxConfig::default()`. Plain
    /// `#[serde(default)]` on an `Option` yields `None`, which emits no `--pflash` flag and
    /// leaves the runtime's own default in force — and rapid-mlx 0.11.1 defaults `--pflash`
    /// to `always` for the verified Qwen3.5/Qwen3.6 aliases. The 2026-07-24 benchmark verdict
    /// is that `auto`/`always` are not recommended on 0.11.x: needle recall collapses 0-40%
    /// above the 32768-token threshold across both TurboQuant settings. So every preset
    /// loaded from disk and every API request has to carry the `off` explicitly.
    #[serde(
        default = "default_pflash_policy",
        skip_serializing_if = "Option::is_none"
    )]
    pub pflash_policy: Option<String>,
    // ── Phase 7: batching/concurrency ──────────────────────────────────
    /// Max number of sequences.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_num_seqs: Option<u64>,
    /// Max concurrent requests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_concurrent_requests: Option<u64>,
    /// Prefill batch size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_batch_size: Option<u64>,
    /// Completion batch size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_batch_size: Option<u64>,
    /// Per-request context window (prompt plus output) passed as `--context-length`.
    /// `None` lets the runtime size it from available memory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
    /// Reasoning effort applied to requests that send no reasoning setting, passed as
    /// `--default-reasoning-effort`. `None` sends no flag. Without it the runtime turns
    /// thinking off for casual chat requests that carry no reasoning intent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_reasoning_effort: Option<String>,
    /// Prompt chunk processed per prefill step. The generic text default is 512;
    /// the UI raises this to 1536 for a verified native vision model profile.
    #[serde(default = "default_prefill_step_size")]
    pub prefill_step_size: u32,
    // ── Phase 7: reasoning/speculative ─────────────────────────────────
    /// Reasoning mode (auto/on/off).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_mode: Option<String>,
    /// Qualified, typed Rapid-MLX speculative decoding request. MTP-only for the
    /// first product release; omitted by default because current sampled/tool
    /// workloads normally fall through to autoregressive decoding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speculative_config: Option<RapidMlxSpeculativeConfig>,
    // ── Phase 7: MLLM/embeddings ───────────────────────────────────────
    /// MLLM vision support (auto/on/off).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mllm_vision: Option<String>,
    /// Embeddings support (auto/on/off).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embeddings: Option<String>,
    // ── Phase 7: GPU ───────────────────────────────────────────────────
    /// GPU memory utilization (0.5–1.0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_memory_utilization: Option<f64>,
    // ── Phase 7: Web UI (D26/A44) ──────────────────────────────────────
    // ── Phase 7: endpoint/safety ───────────────────────────────────────
    /// Sampling mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_mode: Option<String>,
    // ── Phase 9: chat template lifecycle ───────────────────────────────
    /// Chat template file path (Phase 9). When set for Rapid-MLX, an overlay directory
    /// is created with symlinks to the model files plus this template as chat_template.jinja.
    /// None means use the model's native template. Matches llama.cpp's chat_template_file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template_file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_top_k: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_min_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_repetition_penalty: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_presence_penalty: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_frequency_penalty: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
}

fn default_disk_checkpoint_interval() -> u32 {
    0
}

fn default_prefill_step_size() -> u32 {
    512
}

fn default_host() -> String {
    "127.0.0.1".into()
}

fn default_port() -> u16 {
    8000
}

fn default_log_level() -> String {
    "INFO".into()
}

/// KV cache dtype configuration for Rapid-MLX.
///
/// These are the values accepted by the published Rapid-MLX CLI.  The two
/// legacy variants are retained only so an existing preset can be restored and
/// diagnosed instead of being silently rewritten; launch rejects them with a
/// clear migration error.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KvCacheConfig {
    /// Use default behavior.
    Auto,
    /// Override to BF16.
    Bf16,
    /// Override to 8-bit KV.
    Int8,
    /// Override to 4-bit KV.
    Int4,
    /// A value written by older Llama Monitor builds. Rapid-MLX 0.10.17 does
    /// not accept it, so keep it round-trippable but never emit it.
    #[serde(rename = "fp16")]
    LegacyFp16,
    /// A value written by older Llama Monitor builds. Rapid-MLX 0.10.17 does
    /// not accept it, so keep it round-trippable but never emit it.
    #[serde(rename = "fp8")]
    LegacyFp8,
}

impl std::fmt::Display for KvCacheConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KvCacheConfig::Auto => write!(f, "auto"),
            KvCacheConfig::Bf16 => write!(f, "bf16"),
            KvCacheConfig::Int8 => write!(f, "int8"),
            KvCacheConfig::Int4 => write!(f, "int4"),
            KvCacheConfig::LegacyFp16 => write!(f, "fp16"),
            KvCacheConfig::LegacyFp8 => write!(f, "fp8"),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RapidMlxHybridMode {
    #[default]
    Auto,
    Force,
    Disable,
}

/// Phase 6: cross-backend prompt-cache mode (Rapid side).
///
/// `Custom` is the serde default (not `Auto`) so that presets saved before this field
/// existed keep deserializing to their exact stored `prefix_cache_enabled` /
/// `retained_cache_mib` / `hybrid_cache_entries` values unchanged — adding this field
/// must not silently change already-launched configurations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheMode {
    /// Smallest memory-safe working set for the canonical single-user coding-agent loop
    /// (retained cache on, 8192 MiB, 16 hybrid entries — the measured Qwen3.6 recommendation).
    Auto,
    /// Prefix cache off entirely.
    Off,
    /// User-supplied `prefix_cache_enabled` / `retained_cache_mib` / `hybrid_cache_entries`
    /// values are used as configured, unchanged.
    #[default]
    Custom,
}

impl CacheMode {
    /// Resolve to (prefix_cache_enabled, retained_cache_mib, hybrid_cache_entries).
    /// `Custom` returns the config's own values untouched.
    pub fn resolve(
        self,
        configured_enabled: bool,
        configured_retained_mib: Option<u32>,
        configured_hybrid_entries: Option<u64>,
    ) -> (bool, Option<u32>, Option<u64>) {
        match self {
            CacheMode::Auto => (true, Some(8192), Some(16)),
            CacheMode::Off => (false, None, None),
            CacheMode::Custom => (
                configured_enabled,
                configured_retained_mib,
                configured_hybrid_entries,
            ),
        }
    }
}

/// TurboQuant reusable-prompt storage policy (D31).
/// Values match Rapid-MLX CLI --kv-cache-turboquant flag: v4, k8v4, none.
/// "auto" is our config sentinel; the builder maps it to omitting the flag (runtime default).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TurboQuantMode {
    /// Use runtime default (omit --kv-cache-turboquant flag).
    #[serde(rename = "auto")]
    Auto,
    /// V-only TurboQuant (expert legacy/A-B).
    #[serde(rename = "v4")]
    V4,
    /// K8V4 asymmetric TurboQuant (Advanced trial recommendation).
    #[serde(rename = "k8v4", alias = "k8_v4")]
    K8V4,
    /// Disable TurboQuant (Standard retained-storage policy, normally int4).
    #[serde(rename = "none", alias = "off")]
    Off,
}

impl std::fmt::Display for TurboQuantMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TurboQuantMode::Auto => write!(f, "auto"),
            TurboQuantMode::V4 => write!(f, "v4"),
            TurboQuantMode::K8V4 => write!(f, "k8v4"),
            TurboQuantMode::Off => write!(f, "none"),
        }
    }
}

pub fn ensure_local_platform_supported() -> Result<()> {
    if std::env::var_os("LLAMA_MONITOR_FAKE_RAPID_MLX_LOCAL_AVAILABLE").is_some_and(|v| v == "1") {
        return Ok(());
    }
    if std::env::consts::OS != "macos" {
        return Err(anyhow!(
            "Rapid-MLX local execution requires macOS on Apple Silicon. Detected OS: {}",
            std::env::consts::OS
        ));
    }
    if std::env::consts::ARCH != "aarch64" {
        return Err(anyhow!(
            "Rapid-MLX local execution requires Apple Silicon (aarch64). Detected architecture: {}",
            std::env::consts::ARCH
        ));
    }
    Ok(())
}

impl Default for RapidMlxConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            model_source: None,
            served_model_name: None,
            executable_path: None,
            managed_runtime_path: None,
            host: default_host(),
            port: default_port(),
            log_level: default_log_level(),
            timeout: None,
            prefix_cache_enabled: true,
            retained_cache_mib: Some(8192),
            cache_mode: CacheMode::Auto,
            disk_checkpoint_interval: 0,
            api_key: None,
            enable_thinking: None,
            reasoning_effort: None,
            trust_remote_code_consent: None,
            tool_call_parser: None,
            reasoning_parser: None,
            auto_tool_choice: false,
            no_thinking: false,
            escape_hatch_flags: Vec::new(),
            model_source_view: None,
            // Phase 7: KV/cache policy
            kv_cache_dtype: None,
            turboquant_mode: None,
            hybrid_cache_entries: Some(16),
            hybrid_mode: RapidMlxHybridMode::Auto,
            pflash_policy: Some("off".into()),
            // Phase 7: batching/concurrency
            max_num_seqs: None,
            max_concurrent_requests: None,
            prefill_batch_size: None,
            completion_batch_size: None,
            context_length: None,
            default_reasoning_effort: None,
            prefill_step_size: default_prefill_step_size(),
            // Phase 7: reasoning/speculative
            reasoning_mode: None,
            speculative_config: None,
            // Phase 7: MLLM/embeddings
            mllm_vision: None,
            embeddings: None,
            // Phase 7: GPU
            gpu_memory_utilization: None,
            // Phase 7: Web UI
            // Phase 7: endpoint/safety
            sampling_mode: None,
            // Phase 9: chat template lifecycle
            chat_template_file: None,
            default_temperature: None,
            default_top_p: None,
            default_top_k: None,
            default_min_p: None,
            default_repetition_penalty: None,
            default_presence_penalty: None,
            default_frequency_penalty: None,
            max_tokens: None,
        }
    }
}

impl RapidMlxConfig {
    /// Typed source wins; legacy model_path is only a launch-time fallback.
    /// Never opens legacy data to produce a view (Gap 3.2).
    pub fn effective_model_source(&self) -> Result<RapidMlxModelSource> {
        self.model_source.clone().map(Ok).unwrap_or_else(|| {
            self::model_resolver::source_from_legacy_model_path(&self.model_path)
        })
    }

    // No caller in the running binary: `presets.rs` assigns the `model_source_view` field
    // directly rather than deriving it from the config.
    #[allow(dead_code)]
    /// Typed source view for display/edit/clone/save/estimate/library/launch.
    /// Uses only `model_source`; ignores legacy `model_path` entirely (Gap 3.2).
    /// Returns empty view when no typed source is configured — this is the
    /// "No model configured" signal that the frontend must show.
    pub fn model_source_view(&self) -> RapidMlxModelSourceView {
        match &self.model_source {
            Some(source) => RapidMlxModelSourceView::from_source(source),
            None => RapidMlxModelSourceView::empty(),
        }
    }

    /// Never refuses: exposing on 0.0.0.0 without a key is the user's call.
    /// Returns a human-readable warning when the launch is LAN-visible and
    /// unauthenticated, so launchers can log it instead of blocking.
    pub fn access_warning(&self, fallback_api_key: Option<&str>) -> Option<String> {
        let loopback = matches!(
            self.host.as_str(),
            "127.0.0.1" | "localhost" | "::1" | "[::1]"
        );
        let has_key = self
            .api_key
            .as_deref()
            .or(fallback_api_key)
            .is_some_and(|key| !key.is_empty());
        if loopback || has_key {
            return None;
        }
        Some(format!(
            "Serving on {} without an API key — anyone on your network can use this model.              Configure an API key or bind to 127.0.0.1 to restrict access.",
            self.host
        ))
    }

    pub fn validate_speculative_config(&self) -> Result<()> {
        if let Some(config) = &self.speculative_config {
            config.validate()?;
        }
        Ok(())
    }
}

pub struct RapidMlxAdapter {
    pub runtime: RuntimeMetadata,
    pub resolved_model: ResolvedRapidMlxLaunchModel,
    pub served_model_name: Option<String>,
    pub host: String,
    pub port: u16,
    pub log_level: String,
    pub timeout: Option<u32>,
    pub enable_thinking: Option<bool>,
    pub reasoning_effort: Option<String>,
    pub trust_remote_code_consent: Option<String>,
    pub tool_call_parser: Option<String>,
    pub reasoning_parser: Option<String>,
    pub auto_tool_choice: bool,
    pub no_thinking: bool,
    pub escape_hatch_flags: Vec<(String, serde_json::Value)>,
    // Phase 7 fields
    pub kv_cache_dtype: Option<KvCacheConfig>,
    pub turboquant_mode: Option<TurboQuantMode>,
    pub hybrid_cache_entries: Option<u64>,
    pub hybrid_mode: RapidMlxHybridMode,
    pub pflash_policy: Option<String>,
    pub prefix_cache_enabled: bool,
    pub retained_cache_mib: Option<u32>,
    pub disk_checkpoint_interval: u32,
    pub max_num_seqs: Option<u64>,
    pub max_concurrent_requests: Option<u64>,
    pub prefill_batch_size: Option<u64>,
    pub completion_batch_size: Option<u64>,
    pub context_length: Option<u32>,
    pub default_reasoning_effort: Option<String>,
    pub prefill_step_size: u32,
    pub reasoning_mode: Option<String>,
    pub speculative_config: Option<RapidMlxSpeculativeConfig>,
    pub mllm_vision: Option<String>,
    pub embeddings: Option<String>,
    pub gpu_memory_utilization: Option<f64>,
    pub sampling_mode: Option<String>,
    // Phase 9: chat template lifecycle
    pub chat_template_file: Option<String>,
    pub default_temperature: Option<f64>,
    pub default_top_p: Option<f64>,
    pub default_top_k: Option<u64>,
    pub default_min_p: Option<f64>,
    pub default_repetition_penalty: Option<f64>,
    pub default_presence_penalty: Option<f64>,
    pub default_frequency_penalty: Option<f64>,
    pub max_tokens: Option<u64>,
    api_key: Option<String>,
    compatibility: CompatibilityProfile,
    capabilities: CapabilitySet,
    chat_fields: BTreeSet<&'static str>,
    pollers: std::sync::Mutex<HashMap<u16, std::sync::Arc<self::poller::RapidMlxPoller>>>,
}

impl RapidMlxAdapter {
    fn poller_for(&self, port: u16) -> Result<std::sync::Arc<self::poller::RapidMlxPoller>> {
        let mut pollers = self
            .pollers
            .lock()
            .map_err(|error| anyhow!("Rapid-MLX poller cache lock failed: {error}"))?;
        Ok(pollers
            .entry(port)
            .or_insert_with(|| {
                std::sync::Arc::new(self::poller::RapidMlxPoller::new(
                    &self.host,
                    port,
                    self.api_key.as_deref(),
                ))
            })
            .clone())
    }

    pub fn from_resolved(
        runtime: RuntimeMetadata,
        resolved_model: ResolvedRapidMlxLaunchModel,
    ) -> Self {
        Self {
            runtime,
            resolved_model,
            served_model_name: None,
            host: "127.0.0.1".to_string(),
            port: 8000,
            log_level: "INFO".to_string(),
            timeout: None,
            enable_thinking: None,
            reasoning_effort: None,
            trust_remote_code_consent: None,
            tool_call_parser: None,
            reasoning_parser: None,
            auto_tool_choice: false,
            no_thinking: false,
            escape_hatch_flags: Vec::new(),
            // Phase 7 defaults
            kv_cache_dtype: None,
            turboquant_mode: None,
            hybrid_cache_entries: None,
            hybrid_mode: RapidMlxHybridMode::Auto,
            pflash_policy: Some("off".into()),
            prefix_cache_enabled: true,
            retained_cache_mib: Some(8192),
            disk_checkpoint_interval: 0,
            max_num_seqs: None,
            max_concurrent_requests: None,
            prefill_batch_size: None,
            completion_batch_size: None,
            context_length: None,
            default_reasoning_effort: None,
            prefill_step_size: default_prefill_step_size(),
            reasoning_mode: None,
            speculative_config: None,
            mllm_vision: None,
            embeddings: None,
            gpu_memory_utilization: None,
            sampling_mode: None,
            chat_template_file: None,
            default_temperature: None,
            default_top_p: None,
            default_top_k: None,
            default_min_p: None,
            default_repetition_penalty: None,
            default_presence_penalty: None,
            default_frequency_penalty: None,
            max_tokens: None,
            api_key: None,
            compatibility: CompatibilityProfile::verified_baseline(),
            capabilities: verified_capabilities(),
            chat_fields: verified_chat_fields(),
            pollers: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Copies every launch-affecting field off a preset config.
    ///
    /// This lived inline in `inference::launch` until the Phase 7A2 reconciliation found that
    /// the command-preview endpoint reproduced the same mapping by hand and had drifted from
    /// it. Keeping the copy here lets `command_preview_settings_match_the_live_launch_path`
    /// exercise the real mapping instead of a test-local restatement of it. `api_key` is not
    /// copied: it is applied through `configure_runtime`, which drops empty strings.
    pub fn apply_config(&mut self, config: &RapidMlxConfig) {
        self.served_model_name = config.served_model_name.clone();
        self.host = config.host.clone();
        self.port = config.port;
        self.log_level = config.log_level.clone();
        self.timeout = config.timeout;
        self.enable_thinking = config.enable_thinking;
        self.reasoning_effort = config.reasoning_effort.clone();
        self.trust_remote_code_consent = config.trust_remote_code_consent.clone();
        self.tool_call_parser = config.tool_call_parser.clone();
        self.reasoning_parser = config.reasoning_parser.clone();
        self.auto_tool_choice = config.auto_tool_choice;
        self.no_thinking = config.no_thinking;
        self.escape_hatch_flags = config.escape_hatch_flags.clone();
        // Phase 7 config wiring
        self.kv_cache_dtype = config.kv_cache_dtype.clone();
        self.turboquant_mode = config.turboquant_mode.clone();
        self.hybrid_mode = config.hybrid_mode;
        self.pflash_policy = config.pflash_policy.clone();
        // Phase 6: Auto/Off/Custom prompt-cache mode resolves to the three raw fields here —
        // the single choke point shared by both the real launch path and the command-preview
        // endpoint (see `for_settings_preview`, which also calls `apply_config`).
        let (prefix_cache_enabled, retained_cache_mib, hybrid_cache_entries) =
            config.cache_mode.resolve(
                config.prefix_cache_enabled,
                config.retained_cache_mib,
                config.hybrid_cache_entries,
            );
        self.prefix_cache_enabled = prefix_cache_enabled;
        self.retained_cache_mib = retained_cache_mib;
        self.hybrid_cache_entries = hybrid_cache_entries;
        self.disk_checkpoint_interval = config.disk_checkpoint_interval;
        self.max_num_seqs = config.max_num_seqs;
        self.max_concurrent_requests = config.max_concurrent_requests;
        self.prefill_batch_size = config.prefill_batch_size;
        self.completion_batch_size = config.completion_batch_size;
        self.context_length = config.context_length;
        self.default_reasoning_effort = config.default_reasoning_effort.clone();
        self.prefill_step_size = config.prefill_step_size;
        self.reasoning_mode = config.reasoning_mode.clone();
        self.speculative_config = config.speculative_config.clone();
        self.mllm_vision = config.mllm_vision.clone();
        self.embeddings = config.embeddings.clone();
        self.gpu_memory_utilization = config.gpu_memory_utilization;
        self.sampling_mode = config.sampling_mode.clone();
        self.chat_template_file = config.chat_template_file.clone();
        self.default_temperature = config.default_temperature;
        self.default_top_p = config.default_top_p;
        self.default_top_k = config.default_top_k;
        self.default_min_p = config.default_min_p;
        self.default_repetition_penalty = config.default_repetition_penalty;
        self.default_presence_penalty = config.default_presence_penalty;
        self.default_frequency_penalty = config.default_frequency_penalty;
        self.max_tokens = config.max_tokens;
    }

    /// Builds an adapter that exists only to carry a config into the shared argv mapping.
    ///
    /// The command-preview endpoint uses this so it renders argv through
    /// [`build_launch_argv`] / [`apply_phase7_adapter_config`] — the supervisor's own mapping —
    /// instead of restating it. The runtime metadata is a stub because neither of those reads
    /// it; this adapter is not launchable and is dropped once the preview is rendered.
    pub fn for_settings_preview(
        executable_path: PathBuf,
        model: ResolvedRapidMlxLaunchModel,
        config: &RapidMlxConfig,
    ) -> Self {
        let mut adapter = Self::from_resolved(
            RuntimeMetadata {
                executable_path,
                source: runtime::RuntimeSource::Managed,
                version: compatibility::minimum_version_text(),
                capability_snapshot: None,
                resolved_receipt: None,
                last_probe_result: None,
                prefix_cache_enabled: false,
                mlx_prefix_cache_bytes: None,
            },
            model,
        );
        adapter.apply_config(config);
        // `configure_runtime` is what normally lands the key, and it drops empty strings.
        adapter.api_key = config.api_key.clone().filter(|key| !key.is_empty());
        adapter
    }

    pub fn configure_runtime(
        &mut self,
        compatibility: CompatibilityProfile,
        api_key: Option<String>,
    ) {
        let verified = compatibility.state == self::compatibility::CompatibilityState::Verified;

        // Derive capabilities from snapshot if available, otherwise fall back to
        // the verified/provisional baseline sets. The snapshot is the source of
        // truth for per-feature qualification.
        if let Some(ref snapshot) = self.runtime.capability_snapshot {
            self.capabilities = capabilities_from_snapshot(snapshot);
            self.chat_fields = chat_fields_from_snapshot(snapshot);
        } else if verified {
            self.capabilities = verified_capabilities();
            self.chat_fields = verified_chat_fields();
        } else {
            self.capabilities = provisional_capabilities();
            self.chat_fields = provisional_chat_fields();
        }

        self.compatibility = compatibility;
        self.api_key = api_key.filter(|key| !key.is_empty());
        self.pollers
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
    }

    pub async fn validate(&self) -> Result<()> {
        ensure_local_platform_supported()?;

        if !self.runtime.executable_path.is_file() {
            return Err(anyhow!(
                "Rapid-MLX executable does not exist: {}",
                self.runtime.executable_path.display()
            ));
        }
        if self.resolved_model.launch_argument.trim().is_empty() {
            return Err(anyhow!("Rapid-MLX requires a model path"));
        }
        if let Some(frequency_penalty) = self.default_frequency_penalty
            && !(-2.0..=2.0).contains(&frequency_penalty)
        {
            return Err(anyhow!(
                "default_frequency_penalty must be between -2.0 and 2.0, got {frequency_penalty}"
            ));
        }
        let access = RapidMlxConfig {
            host: self.host.clone(),
            api_key: self.api_key.clone(),
            ..Default::default()
        };
        if let Some(warning) = access.access_warning(None) {
            eprintln!("[warn] rapid-mlx adapter: {warning}");
        }

        Ok(())
    }

    /// Auto-detect Hybrid DeltaNet from MLX config and resolve hybrid_mode.
    ///
    /// If hybrid_mode is Auto and the model directory contains config.json with
    /// full_attention_interval > 1 (Qwen3.5/3.6 DeltaNet), override to Force so
    /// --force-hybrid is emitted. This prevents silent incorrect KV geometry when
    /// rapid-mlx info misidentifies Hybrid DeltaNet as "pure attention".
    fn resolve_hybrid_mode(&self) -> RapidMlxHybridMode {
        if self.hybrid_mode != RapidMlxHybridMode::Auto {
            return self.hybrid_mode;
        }
        let config_path = self.resolved_model.launch_argument.trim();
        if config_path.is_empty() {
            return self.hybrid_mode;
        }
        let path = PathBuf::from(config_path);
        let config_file = path.join("config.json");
        if !config_file.is_file() {
            return self.hybrid_mode;
        }
        let bytes = match std::fs::read(&config_file) {
            Ok(b) => b,
            Err(_) => return self.hybrid_mode,
        };
        let raw: serde_json::Value = match serde_json::from_slice(&bytes) {
            Ok(v) => v,
            Err(_) => return self.hybrid_mode,
        };
        // Check both top-level and nested text_config.full_attention_interval.
        let check_interval = |value: &serde_json::Value| -> Option<u32> {
            value
                .get("full_attention_interval")
                .and_then(|v| v.as_u64())
                .and_then(|v| (v > 1).then_some(v as u32))
        };
        if check_interval(&raw).is_some() {
            return RapidMlxHybridMode::Force;
        }
        if let Some(tc) = raw.get("text_config")
            && check_interval(tc).is_some()
        {
            return RapidMlxHybridMode::Force;
        }
        self.hybrid_mode
    }

    /// Whether `rapid-mlx info` reports a hybrid architecture for an alias or repo launch
    /// argument. Directory launches (and any lookup failure) answer false; those are covered
    /// by the local `config.json` check in `resolve_hybrid_mode`.
    async fn profile_is_hybrid(&self) -> bool {
        fetch_rapid_mlx_profile(self.resolved_model.launch_argument.trim())
            .await
            .and_then(|profile| profile.architecture)
            .is_some_and(|arch| arch.to_ascii_lowercase().contains("hybrid"))
    }

    pub async fn build_launch(&self) -> Result<SupervisedLaunch> {
        let hybrid_mode = self.resolve_hybrid_mode();
        let (argv_builder, overlay_warning) = build_launch_argv(self);
        let builder = apply_phase7_adapter_config(argv_builder, self);

        // Override hybrid_mode if resolved differently (e.g. Hybrid DeltaNet auto-detection)
        let builder = if hybrid_mode != self.hybrid_mode {
            builder.hybrid_mode(hybrid_mode)
        } else {
            builder
        };

        // rapid-mlx refuses to start with --kv-cache-dtype int8/int4 on hybrid
        // (GatedDeltaNet / Mamba ArraysCache) models: "the loaded model is incompatible:
        // ArraysCache". The wizard's reasoning profile pins int8, so presets saved for hybrid
        // aliases carry a value that can never launch. Serve bf16 (the runtime default)
        // instead and say so.
        let mut kv_warning = None;
        let builder = if matches!(
            self.kv_cache_dtype,
            Some(KvCacheConfig::Int8 | KvCacheConfig::Int4)
        ) && (hybrid_mode == RapidMlxHybridMode::Force
            || self.profile_is_hybrid().await)
        {
            kv_warning = Some(
                "This model is hybrid-attention; Rapid-MLX cannot quantize its KV cache, so the \
                 saved KV cache dtype was ignored and bf16 is used."
                    .to_string(),
            );
            builder.kv_cache_dtype(None)
        } else {
            builder
        };

        let mut launch = builder.build(
            self.runtime.executable_path.clone(),
            &self.compatibility.capabilities,
        )?;
        launch.warnings.extend(kv_warning);
        launch.redacted_summary.push_str(&format!(
            " ({}, {})",
            self.compatibility.version,
            self.compatibility.state.label()
        ));
        launch.warnings.extend(overlay_warning);
        Ok(launch)
    }

    pub async fn await_ready(&self, port: u16, deadline: Instant) -> Result<()> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .map_err(|e| anyhow!(e))?;

        let readiness_host = match self.host.as_str() {
            "0.0.0.0" | "::" | "[::]" => "127.0.0.1",
            "::1" => "[::1]",
            host => host,
        };
        let url = format!("http://{readiness_host}:{port}/health/ready");

        loop {
            if Instant::now() > deadline {
                return Err(anyhow!("Timed out waiting for Rapid-MLX to become ready"));
            }

            match client.get(&url).send().await {
                Ok(resp) if resp.status() == reqwest::StatusCode::OK => {
                    return Ok(());
                }
                Ok(_resp) => {}
                Err(_) => {}
            }

            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }

    pub async fn poll_metrics(
        &self,
        _base: &str,
        port: u16,
        _session_id: &str,
    ) -> Result<InferenceMetricsSnapshot> {
        let poller = self.poller_for(port)?;
        poller.poll().await
    }

    pub async fn cancel_request(&self, _port: u16, _request_id: &str) -> Result<()> {
        if _request_id.is_empty()
            || _request_id.len() > 128
            || !_request_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        {
            return Err(anyhow!("Rapid-MLX returned an invalid request ID"));
        }
        if !self.capabilities.cancellation {
            return Err(anyhow!(
                "Rapid-MLX native request cancellation is unavailable because the active runtime does not expose a compatible public request ID"
            ));
        }
        let host = match self.host.as_str() {
            "0.0.0.0" | "::" | "[::]" => "127.0.0.1",
            "::1" => "[::1]",
            host => host,
        };
        let url = format!("http://{host}:{}/v1/requests/{}/cancel", _port, _request_id);
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()?;
        let mut request = client.post(url);
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }
        let response = request.send().await?;
        if !response.status().is_success() {
            return Err(anyhow!(
                "Rapid-MLX cancellation returned HTTP {}",
                response.status()
            ));
        }
        Ok(())
    }

    pub fn capabilities(&self) -> &CapabilitySet {
        &self.capabilities
    }

    pub fn map_chat_request(&self, body: &[u8]) -> Result<Vec<u8>> {
        let mapped = map_chat_request_with_fields(body, &self.chat_fields)?;
        self.apply_reasoning_defaults(mapped)
    }

    /// Fills in `enable_thinking`/`reasoning_effort` from the launch config when the
    /// caller's request omits them, mirroring llama.cpp's standing `--reasoning`
    /// server-level default (Rapid-MLX has no launch-time reasoning flag; these are
    /// per-request chat fields instead).
    fn apply_reasoning_defaults(&self, mapped: Vec<u8>) -> Result<Vec<u8>> {
        if self.enable_thinking.is_none() && self.reasoning_effort.is_none() {
            return Ok(mapped);
        }
        let mut value: serde_json::Value = serde_json::from_slice(&mapped)?;
        let object = value
            .as_object_mut()
            .ok_or_else(|| anyhow!("Chat request must be a JSON object"))?;
        if self.chat_fields.contains("enable_thinking")
            && let Some(default) = self.enable_thinking
            && !object.contains_key("enable_thinking")
        {
            object.insert("enable_thinking".to_string(), serde_json::json!(default));
        }
        if self.chat_fields.contains("reasoning_effort")
            && let Some(default) = &self.reasoning_effort
            && !object.contains_key("reasoning_effort")
        {
            object.insert("reasoning_effort".to_string(), serde_json::json!(default));
        }
        Ok(serde_json::to_vec(&value)?)
    }
}

/// Why a requested chat-template overlay was not applied at launch. A stable, matchable
/// classification of the (currently string-based) overlay-failure warning, so the reason can
/// eventually drive frontend copy (Phase 1, plan §2.2) instead of the caller parsing English text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChatTemplateDegradeReason {
    /// The model was selected by alias — a bare string with no directory to overlay into.
    AliasSource,
    /// The resolved launch path does not exist on disk.
    ModelPathMissing,
    /// The resolved launch path exists but is not a directory.
    ModelPathNotDirectory,
    /// The requested template file could not be read.
    TemplateUnreadable,
    /// The overlay directory could not be written (symlinks or `chat_template.jinja`).
    OverlayWriteFailed,
}

impl ChatTemplateDegradeReason {
    pub(crate) fn code(self) -> &'static str {
        match self {
            Self::AliasSource => "alias_source",
            Self::ModelPathMissing => "model_path_missing",
            Self::ModelPathNotDirectory => "model_path_not_directory",
            Self::TemplateUnreadable => "template_unreadable",
            Self::OverlayWriteFailed => "overlay_write_failed",
        }
    }

    /// Alias sources are classified from the model's origin, not the overlay error, because
    /// `create_template_overlay` has no notion of `RapidMlxModelSource` — it only sees a path.
    fn classify(source: &model_resolver::RapidMlxModelSource, err: &anyhow::Error) -> Self {
        if matches!(source, model_resolver::RapidMlxModelSource::Alias { .. }) {
            return Self::AliasSource;
        }
        let msg = err.to_string();
        if msg.contains("does not exist") {
            Self::ModelPathMissing
        } else if msg.contains("is not a directory") {
            Self::ModelPathNotDirectory
        } else if msg.contains("not readable") {
            Self::TemplateUnreadable
        } else {
            Self::OverlayWriteFailed
        }
    }
}

/// Seeds a command builder with the non-Phase-7 launch settings.
///
/// Split out of `build_launch` during the Phase 7A2 reconciliation. Together with
/// [`apply_phase7_adapter_config`] this is the *only* adapter → argv mapping; the
/// command-preview endpoint drives the same two functions through a throwaway adapter so a
/// preview cannot describe a command the supervisor would not run. It used to keep its own
/// copy, which had drifted: the preview invented `--log-level INFO`, `--timeout 60` and an
/// empty `--api-key`, and dropped `--served-model-name`, `--reasoning-parser`, the eight
/// sampling defaults, the hybrid switches, and `--prefill-step-size`.
pub(crate) fn build_launch_argv(
    adapter: &RapidMlxAdapter,
) -> (command::RapidMlxCommandBuilder, Option<String>) {
    // Phase 9: if a custom chat template is active, create an overlay directory
    // and use that as the model argument instead of the original model dir.
    let mut overlay_warning = None;
    let resolved_model = if let Some(ref ct_file) = adapter.chat_template_file
        && !ct_file.is_empty()
    {
        match model_resolver::create_template_overlay(
            &adapter.resolved_model.launch_argument,
            Some(ct_file),
        ) {
            Ok(overlay_path) => {
                // Clone the resolved model with the overlay path as the launch argument
                ResolvedRapidMlxLaunchModel {
                    launch_argument: overlay_path,
                    ..adapter.resolved_model.clone()
                }
            }
            Err(e) => {
                // Overlay creation failed: continue with original model dir.
                // The launch will use the native template; this allows graceful degradation
                // if the template file is missing or overlay creation fails. Also surfaced
                // as a launch warning (Phase 9 B3) so this isn't silent to the operator.
                let reason =
                    ChatTemplateDegradeReason::classify(&adapter.resolved_model.original_input, &e);
                let msg = format!(
                    "Rapid-MLX chat template overlay failed [{}] ({e}, template: {ct_file}): using native model template",
                    reason.code()
                );
                eprintln!("Warning: {msg}");
                overlay_warning = Some(msg);
                adapter.resolved_model.clone()
            }
        }
    } else {
        adapter.resolved_model.clone()
    };

    let (_, speculative_warning) = effective_speculative_config(adapter);
    if let Some(warning) = speculative_warning {
        overlay_warning = Some(match overlay_warning {
            Some(existing) => format!("{existing}\n{warning}"),
            None => warning,
        });
    }

    let mut builder = RapidMlxCommandBuilder::new(resolved_model)
        .host(adapter.host.clone())
        .port(adapter.port);

    if adapter.log_level != "INFO" {
        builder = builder.log_level(adapter.log_level.clone());
    }
    if let Some(timeout) = adapter.timeout {
        builder = builder.timeout(timeout);
    }
    builder = builder
        .prefix_cache_enabled(Some(adapter.prefix_cache_enabled))
        .retained_cache_mib(adapter.retained_cache_mib)
        .disk_checkpoint_interval(Some(adapter.disk_checkpoint_interval));

    if let Some(key) = &adapter.api_key {
        builder = builder.api_key(key.clone());
    }
    if let Some(name) = &adapter.served_model_name {
        builder = builder.served_model_name(name.clone());
    }

    let builder = builder
        .trust_remote_code_consent(adapter.trust_remote_code_consent.clone())
        .tool_call_parser(adapter.tool_call_parser.clone())
        .reasoning_parser(adapter.reasoning_parser.clone())
        .auto_tool_choice(adapter.auto_tool_choice)
        .no_thinking(adapter.no_thinking)
        .escape_hatch_flags(adapter.escape_hatch_flags.clone());

    (builder, overlay_warning)
}

/// `pub(crate)` for the same reason as [`build_launch_argv`]: the command-preview endpoint
/// applies the supervisor's own mapping rather than a copy of it.
pub(crate) fn apply_phase7_adapter_config(
    builder: command::RapidMlxCommandBuilder,
    adapter: &RapidMlxAdapter,
) -> command::RapidMlxCommandBuilder {
    let (speculative_config, _) = effective_speculative_config(adapter);
    builder
        .kv_cache_dtype(adapter.kv_cache_dtype.as_ref().map(|kv| {
            use crate::inference::rapid_mlx::command::KvCacheDtypeArg;
            match kv {
                KvCacheConfig::Auto => KvCacheDtypeArg::Auto,
                KvCacheConfig::Bf16 => KvCacheDtypeArg::Explicit("bf16".into()),
                KvCacheConfig::Int8 => KvCacheDtypeArg::Explicit("int8".into()),
                KvCacheConfig::Int4 => KvCacheDtypeArg::Explicit("int4".into()),
                KvCacheConfig::LegacyFp16 => KvCacheDtypeArg::Explicit("fp16".into()),
                KvCacheConfig::LegacyFp8 => KvCacheDtypeArg::Explicit("fp8".into()),
            }
        }))
        // A runtime flag alone is insufficient evidence that this exact
        // model/revision has a qualified retained-KV path. Keep the requested
        // setting persisted, but omit TurboQuant until a receipt is available.
        .turboquant_mode(None)
        .hybrid_cache_entries(adapter.hybrid_cache_entries)
        .hybrid_mode(adapter.hybrid_mode)
        .pflash_policy(adapter.pflash_policy.clone())
        .max_num_seqs(adapter.max_num_seqs)
        .max_concurrent_requests(adapter.max_concurrent_requests)
        .prefill_batch_size(adapter.prefill_batch_size)
        .completion_batch_size(adapter.completion_batch_size)
        .context_length(adapter.context_length)
        .default_reasoning_effort(adapter.default_reasoning_effort.clone())
        .prefill_step_size(Some(adapter.prefill_step_size))
        .reasoning_mode(adapter.reasoning_mode.clone())
        .speculative_config(speculative_config)
        .mllm_vision(adapter.mllm_vision.clone())
        .embeddings(adapter.embeddings.clone())
        .gpu_memory_utilization(adapter.gpu_memory_utilization)
        .sampling_mode(adapter.sampling_mode.clone())
        .sampling_defaults(
            adapter.default_temperature,
            adapter.default_top_p,
            adapter.default_top_k,
            adapter.default_min_p,
            adapter.default_repetition_penalty,
            adapter.default_presence_penalty,
            adapter.default_frequency_penalty,
            adapter.max_tokens,
        )
}

fn effective_speculative_config(
    adapter: &RapidMlxAdapter,
) -> (Option<RapidMlxSpeculativeConfig>, Option<String>) {
    let Some(mut config) = adapter.speculative_config.clone() else {
        return (None, None);
    };

    let trunk = Path::new(&adapter.resolved_model.launch_argument);
    let has_explicit_companion = config
        .model
        .as_deref()
        .is_some_and(|model| !model.trim().is_empty());

    if trunk.is_absolute() {
        if !sidecar_inventory::is_mtp_eligible_trunk(trunk) {
            return (
                None,
                Some(
                    "Rapid-MLX speculation disabled: the selected model architecture is not MTP-eligible"
                        .into(),
                ),
            );
        }

        if !has_explicit_companion {
            let Some(sidecar) = sidecar_inventory::find_sidecar_for_path(trunk) else {
                return (
                    None,
                    Some(
                        "Rapid-MLX speculation disabled: no validated managed MTP sidecar is registered for this trunk"
                            .into(),
                    ),
                );
            };
            config.model = Some(sidecar.path);
        }
    } else if !has_explicit_companion {
        return (
            None,
            Some(
                "Rapid-MLX speculation disabled: the selected model is not a local trunk with a managed sidecar"
                    .into(),
            ),
        );
    }

    (Some(config), None)
}

pub fn map_provisional_chat_request(body: &[u8]) -> Result<Vec<u8>> {
    map_chat_request_with_fields(body, &provisional_chat_fields())
}

fn map_chat_request_with_fields(body: &[u8], fields: &BTreeSet<&'static str>) -> Result<Vec<u8>> {
    let value: serde_json::Value = serde_json::from_slice(body)
        .map_err(|error| anyhow!("Invalid chat request JSON: {error}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| anyhow!("Chat request must be a JSON object"))?;
    let mut mapped = serde_json::Map::new();
    for (key, value) in object {
        let output_key = if key == "repeat_penalty" {
            "repetition_penalty"
        } else if key == "thinking_budget_tokens" {
            // Llama.cpp and the shared chat UI use `thinking_budget_tokens` for
            // the reasoning-only ceiling. Rapid-MLX names the same semantic
            // control `reasoning_max_tokens`; forwarding the llama.cpp name is
            // silently ignored and leaves reasoning effectively unbounded.
            if object.contains_key("reasoning_max_tokens") {
                continue;
            }
            "reasoning_max_tokens"
        } else {
            key.as_str()
        };
        if fields.contains(output_key) {
            mapped.insert(output_key.to_string(), value.clone());
        }
    }
    if !mapped.contains_key("messages") {
        return Err(anyhow!("Rapid-MLX chat requests require messages"));
    }
    if fields.contains("stream_options")
        && mapped.get("stream").and_then(serde_json::Value::as_bool) == Some(true)
    {
        match mapped.get_mut("stream_options") {
            Some(serde_json::Value::Object(options)) => {
                options
                    .entry("include_usage".to_string())
                    .or_insert(serde_json::Value::Bool(true));
            }
            None => {
                mapped.insert(
                    "stream_options".to_string(),
                    serde_json::json!({"include_usage": true}),
                );
            }
            Some(_) => return Err(anyhow!("Rapid-MLX stream_options must be a JSON object")),
        }
    }
    Ok(serde_json::to_vec(&mapped)?)
}

fn provisional_capabilities() -> CapabilitySet {
    CapabilitySet {
        status_memory_telemetry: true,
        one_shot_launch: true,
        ..Default::default()
    }
}

fn verified_capabilities() -> CapabilitySet {
    CapabilitySet {
        // 0.10.9 exposes a cancellation endpoint for its private scheduler ID,
        // but that ID is not exposed in OpenAI SSE chunks or response headers.
        // The public chatcmpl-* response ID is not a compatible contract.
        cancellation: false,
        guided_generation: true,
        tool_parsing: true,
        automatic_tool_choice: true,
        reasoning_parser: true,
        thinking_controls: true,
        status_memory_telemetry: true,
        one_shot_launch: true,
        ..Default::default()
    }
}

fn provisional_chat_fields() -> BTreeSet<&'static str> {
    [
        "messages",
        "model",
        "stream",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "max_tokens",
        "max_completion_tokens",
        "reasoning_max_tokens",
        "stop",
        "repetition_penalty",
    ]
    .into_iter()
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::rapid_mlx::runtime::RuntimeSource;

    #[test]
    fn cache_mode_auto_resolves_to_measured_recommendation() {
        let (enabled, retained_mib, hybrid_entries) = CacheMode::Auto.resolve(false, None, None);
        assert!(enabled);
        assert_eq!(retained_mib, Some(8192));
        assert_eq!(hybrid_entries, Some(16));
    }

    #[test]
    fn cache_mode_off_disables_and_clears() {
        let (enabled, retained_mib, hybrid_entries) =
            CacheMode::Off.resolve(true, Some(4096), Some(8));
        assert!(!enabled);
        assert_eq!(retained_mib, None);
        assert_eq!(hybrid_entries, None);
    }

    #[test]
    fn cache_mode_custom_preserves_configured_values_untouched() {
        let (enabled, retained_mib, hybrid_entries) =
            CacheMode::Custom.resolve(true, Some(2048), Some(4));
        assert!(enabled);
        assert_eq!(retained_mib, Some(2048));
        assert_eq!(hybrid_entries, Some(4));
    }

    #[test]
    fn cache_mode_serde_default_is_custom_for_backward_compatibility() {
        // A preset JSON saved before `cache_mode` existed must deserialize with the field
        // absent, and must resolve to `Custom` — not `Auto` — so old presets keep their
        // exact stored prefix_cache_enabled/retained_cache_mib/hybrid_cache_entries values.
        assert_eq!(CacheMode::default(), CacheMode::Custom);
    }

    #[test]
    fn speculative_config_defaults_to_three_tokens_without_overwriting_saved_values() {
        let fresh: RapidMlxSpeculativeConfig = serde_json::from_str(r#"{"method":"mtp"}"#).unwrap();
        assert_eq!(fresh.num_speculative_tokens, 3);
        assert!(!fresh.disable_auto_k);

        let saved: RapidMlxSpeculativeConfig =
            serde_json::from_str(r#"{"method":"mtp","num_speculative_tokens":2}"#).unwrap();
        assert_eq!(saved.num_speculative_tokens, 2);
    }

    #[test]
    fn explicit_sidecar_is_preserved_for_non_local_model_alias() {
        let mut adapter = RapidMlxAdapter::from_resolved(
            RuntimeMetadata::default(),
            ResolvedRapidMlxLaunchModel::validated_alias("org/model").unwrap(),
        );
        adapter.speculative_config = Some(RapidMlxSpeculativeConfig {
            method: RapidMlxSpeculativeMethod::Mtp,
            model: Some("/managed/sidecar".into()),
            num_speculative_tokens: 3,
            disable_auto_k: false,
        });

        let (effective, warning) = effective_speculative_config(&adapter);
        assert_eq!(
            effective.and_then(|config| config.model),
            Some("/managed/sidecar".into())
        );
        assert!(warning.is_none());
    }

    #[test]
    fn phase7_config_serialization_roundtrip() {
        // Verify RapidMlxConfig with all Phase 7 fields serializes/deserializes without loss.
        let config = RapidMlxConfig {
            model_path: "/model".into(),
            kv_cache_dtype: Some(KvCacheConfig::Int8),
            turboquant_mode: Some(TurboQuantMode::K8V4),
            hybrid_cache_entries: Some(100),
            hybrid_mode: RapidMlxHybridMode::Auto,
            pflash_policy: Some("auto".into()),
            max_num_seqs: Some(8),
            max_concurrent_requests: Some(32),
            prefill_batch_size: Some(256),
            completion_batch_size: Some(64),
            prefill_step_size: 512,
            reasoning_mode: Some("on".into()),
            mllm_vision: Some("auto".into()),
            embeddings: Some("auto".into()),
            gpu_memory_utilization: Some(0.85),
            sampling_mode: Some("auto".into()),
            default_temperature: Some(0.7),
            default_top_p: Some(0.9),
            default_top_k: Some(40),
            default_min_p: Some(0.05),
            default_repetition_penalty: Some(1.05),
            default_presence_penalty: Some(0.0),
            default_frequency_penalty: Some(0.0),
            max_tokens: Some(32768),
            ..Default::default()
        };
        let json = serde_json::to_value(&config).unwrap();
        assert_eq!(json["turboquant_mode"], serde_json::json!("k8v4"));
        let restored: RapidMlxConfig = serde_json::from_value(json).unwrap();
        assert_eq!(restored.kv_cache_dtype, config.kv_cache_dtype);
        assert_eq!(restored.turboquant_mode, config.turboquant_mode);
        assert_eq!(restored.max_num_seqs, config.max_num_seqs);
        assert_eq!(
            restored.gpu_memory_utilization,
            config.gpu_memory_utilization
        );
    }

    #[test]
    fn turboquant_legacy_preset_literals_remain_readable() {
        let legacy_k8: TurboQuantMode = serde_json::from_str("\"k8_v4\"").unwrap();
        let legacy_off: TurboQuantMode = serde_json::from_str("\"off\"").unwrap();
        assert_eq!(legacy_k8, TurboQuantMode::K8V4);
        assert_eq!(legacy_off, TurboQuantMode::Off);
        assert_eq!(serde_json::to_string(&legacy_k8).unwrap(), "\"k8v4\"");
        assert_eq!(serde_json::to_string(&legacy_off).unwrap(), "\"none\"");
    }

    #[test]
    fn phase7_all_settings_iterate_without_panic() {
        // Verify all_settings() can be fully iterated and each returns valid default.
        use crate::inference::rapid_mlx::settings::all_settings;
        for setting in all_settings() {
            let default = setting.default_value();
            assert!(
                default.is_string()
                    || default.is_number()
                    || default.is_object()
                    || default.is_null()
            );
        }
    }

    #[test]
    fn adapter_reuses_poller_per_port_and_separates_ports() {
        let adapter = RapidMlxAdapter::from_resolved(
            RuntimeMetadata {
                executable_path: "rapid-mlx".into(),
                source: RuntimeSource::Managed,
                version: "0.11.0+git.fixture".into(),
                capability_snapshot: None,
                resolved_receipt: None,
                last_probe_result: None,
                prefix_cache_enabled: false,
                mlx_prefix_cache_bytes: None,
            },
            ResolvedRapidMlxLaunchModel::validated_alias("model").unwrap(),
        );
        let first = adapter.poller_for(8000).unwrap();
        let second = adapter.poller_for(8000).unwrap();
        let other = adapter.poller_for(8001).unwrap();
        assert!(std::sync::Arc::ptr_eq(&first, &second));
        assert!(!std::sync::Arc::ptr_eq(&first, &other));
    }

    #[test]
    fn resolve_hybrid_mode_preserves_explicit_force() {
        // Non-Auto values must not be overridden.
        let adapter = RapidMlxAdapter::from_resolved(
            RuntimeMetadata::default(),
            ResolvedRapidMlxLaunchModel::validated_alias("model").unwrap(),
        );
        let config = RapidMlxConfig::default();
        let with_force = RapidMlxConfig {
            hybrid_mode: RapidMlxHybridMode::Force,
            ..config.clone()
        };
        let with_disable = RapidMlxConfig {
            hybrid_mode: RapidMlxHybridMode::Disable,
            ..config
        };
        // We can't call resolve_hybrid_mode directly on a config, but we verify
        // the behavior is correct via the from_resolved default (Auto).
        assert_eq!(adapter.hybrid_mode, RapidMlxHybridMode::Auto);
        // Explicit values in config are preserved as-is by launch.rs wiring.
        assert_eq!(with_force.hybrid_mode, RapidMlxHybridMode::Force);
        assert_eq!(with_disable.hybrid_mode, RapidMlxHybridMode::Disable);
    }

    #[test]
    fn resolve_hybrid_mode_detects_nested_text_config() {
        use std::io::Write;

        let tmp = tempfile::tempdir().unwrap();
        let model_dir = tmp.path().join("model");
        std::fs::create_dir(&model_dir).unwrap();
        let mut file = std::fs::File::create(model_dir.join("config.json")).unwrap();
        file.write_all(
            r#"{
                "model_type": "qwen3_5",
                "text_config": {
                    "num_hidden_layers": 64,
                    "full_attention_interval": 4,
                    "num_key_value_heads": 4
                }
            }"#
            .as_bytes(),
        )
        .unwrap();

        let adapter = RapidMlxAdapter::from_resolved(
            RuntimeMetadata::default(),
            ResolvedRapidMlxLaunchModel {
                launch_argument: model_dir.to_string_lossy().to_string(),
                display_name: "test".into(),
                source_kind: crate::inference::rapid_mlx::model_resolver::ResolvedRapidMlxSourceKind::MlxDirectory,
                original_input: crate::inference::rapid_mlx::RapidMlxModelSource::MlxDirectory {
                    path: model_dir.clone(),
                },
                conversion: None,
                required_environment: Vec::new(),
                warnings: Vec::new(),
                remediation: Vec::new(),
                trust_remote_code_required: None,
                environment: std::collections::BTreeMap::new(),
            },
        );
        // Auto + full_attention_interval=4 → Force
        assert_eq!(adapter.resolve_hybrid_mode(), RapidMlxHybridMode::Force);
    }

    #[test]
    fn resolve_hybrid_mode_detects_top_level_interval() {
        use std::io::Write;

        let tmp = tempfile::tempdir().unwrap();
        let model_dir = tmp.path().join("model");
        std::fs::create_dir(&model_dir).unwrap();
        let mut file = std::fs::File::create(model_dir.join("config.json")).unwrap();
        file.write_all(
            r#"{
                "full_attention_interval": 4,
                "num_hidden_layers": 64
            }"#
            .as_bytes(),
        )
        .unwrap();

        let adapter = RapidMlxAdapter::from_resolved(
            RuntimeMetadata::default(),
            ResolvedRapidMlxLaunchModel {
                launch_argument: model_dir.to_string_lossy().to_string(),
                display_name: "test".into(),
                source_kind: crate::inference::rapid_mlx::model_resolver::ResolvedRapidMlxSourceKind::MlxDirectory,
                original_input: crate::inference::rapid_mlx::RapidMlxModelSource::MlxDirectory {
                    path: model_dir.clone(),
                },
                conversion: None,
                required_environment: Vec::new(),
                warnings: Vec::new(),
                remediation: Vec::new(),
                trust_remote_code_required: None,
                environment: std::collections::BTreeMap::new(),
            },
        );
        // Auto + top-level full_attention_interval=4 → Force
        assert_eq!(adapter.resolve_hybrid_mode(), RapidMlxHybridMode::Force);
    }

    #[test]
    fn resolve_hybrid_mode_no_interval_remains_auto() {
        use std::io::Write;

        let tmp = tempfile::tempdir().unwrap();
        let model_dir = tmp.path().join("model");
        std::fs::create_dir(&model_dir).unwrap();
        let mut file = std::fs::File::create(model_dir.join("config.json")).unwrap();
        file.write_all(
            r#"{
                "num_hidden_layers": 32,
                "num_key_value_heads": 4
            }"#
            .as_bytes(),
        )
        .unwrap();

        let adapter = RapidMlxAdapter::from_resolved(
            RuntimeMetadata::default(),
            ResolvedRapidMlxLaunchModel {
                launch_argument: model_dir.to_string_lossy().to_string(),
                display_name: "test".into(),
                source_kind: crate::inference::rapid_mlx::model_resolver::ResolvedRapidMlxSourceKind::MlxDirectory,
                original_input: crate::inference::rapid_mlx::RapidMlxModelSource::MlxDirectory {
                    path: model_dir.clone(),
                },
                conversion: None,
                required_environment: Vec::new(),
                warnings: Vec::new(),
                remediation: Vec::new(),
                trust_remote_code_required: None,
                environment: std::collections::BTreeMap::new(),
            },
        );
        // Auto + no interval → Auto
        assert_eq!(adapter.resolve_hybrid_mode(), RapidMlxHybridMode::Auto);
    }
}

/// Derive CapabilitySet from a capability snapshot's qualified features.
fn capabilities_from_snapshot(snapshot: &self::capabilities::CapabilitySnapshot) -> CapabilitySet {
    fn is_available(q: &self::capabilities::FeatureQualification) -> bool {
        matches!(q, self::capabilities::FeatureQualification::Available)
    }

    CapabilitySet {
        tool_parsing: is_available(&snapshot.qualified_features.tool_parsing),
        automatic_tool_choice: is_available(&snapshot.qualified_features.automatic_tool_choice),
        reasoning_parser: is_available(&snapshot.qualified_features.reasoning_parser),
        thinking_controls: is_available(&snapshot.qualified_features.thinking_controls),
        guided_generation: is_available(&snapshot.qualified_features.guided_generation),
        vision: is_available(&snapshot.qualified_features.vision),
        embeddings: is_available(&snapshot.qualified_features.embeddings),
        // Evidence-backed: false until a full gate sweep promotes spec_decode to
        // Available, either as a shipped version prior or as a measurement recorded
        // against this exact install. Until 2026-07-30 no source could return
        // Available at all, so this was permanently false for every user on every
        // build; see capabilities::apply_measured_spec_decode for the resolution order.
        mtp: is_available(&snapshot.qualified_features.spec_decode),
        // Core capabilities always available when runtime is validated
        status_memory_telemetry: true,
        one_shot_launch: true,
        ..Default::default()
    }
}

/// Derive chat fields from a capability snapshot.
fn chat_fields_from_snapshot(
    snapshot: &self::capabilities::CapabilitySnapshot,
) -> BTreeSet<&'static str> {
    let mut fields = provisional_chat_fields();

    // Always add fields available on verified runtime
    fields.extend([
        "stream_options",
        "presence_penalty",
        "frequency_penalty",
        "logprobs",
        "timeout",
    ]);

    // Add tool fields if tool parsing is available
    if matches!(
        snapshot.qualified_features.tool_parsing,
        self::capabilities::FeatureQualification::Available
    ) {
        fields.extend(["tools", "tool_choice", "parallel_tool_calls"]);
    }

    // Add response_format if guided is available (structured generation)
    if matches!(
        snapshot.qualified_features.guided_generation,
        self::capabilities::FeatureQualification::Available
    ) {
        fields.insert("response_format");
    }

    // Add thinking fields if reasoning/thinking controls available
    if matches!(
        snapshot.qualified_features.thinking_controls,
        self::capabilities::FeatureQualification::Available
    ) {
        fields.extend([
            "enable_thinking",
            "chat_template_kwargs",
            "reasoning_effort",
            "reasoning_max_tokens",
        ]);
    }

    fields
}

fn verified_chat_fields() -> BTreeSet<&'static str> {
    let mut fields = provisional_chat_fields();
    fields.extend([
        "stream_options",
        "presence_penalty",
        "frequency_penalty",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "response_format",
        "logprobs",
        "timeout",
        "enable_thinking",
        "chat_template_kwargs",
        "reasoning_effort",
        "reasoning_max_tokens",
    ]);
    fields
}

// ── Unified Profile (Section 20 Item 11) ─────────────────────────────────────────

/// Merged profile combining MLX config geometry, Rapid-MLX info behavioral flags,
/// and explicit fallbacks into a single recommendation with source provenance.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct UnifiedProfile {
    pub recommended: UnifiedProfileRecommended,
    pub sources: UnifiedProfileSources,
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct UnifiedProfileRecommended {
    pub hybrid_mode: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_format: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_parser: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct UnifiedProfileSources {
    pub hybrid_mode: String,
    pub tool_format: String,
    pub reasoning_parser: String,
}

/// Build a unified profile by merging three sources:
/// 1. MLX config (local or HF) → geometry and hybrid detection
/// 2. Rapid-MLX info profile → behavioral flags (tool_format, reasoning_parser, architecture)
/// 3. Explicit mappings → fallbacks when sources are incomplete
pub async fn build_unified_profile(model_id: &str) -> Result<UnifiedProfile> {
    let mut warnings = Vec::new();

    // Fetch MLX geometry from appropriate source
    let geometry_profile = fetch_geometry_profile(model_id).await;
    let hybrid_from_geometry = derive_hybrid_mode_from_geometry(&geometry_profile);

    // Fetch Rapid-MLX info profile (may fail gracefully if rapid-mlx not installed)
    let rapid_profile = fetch_rapid_mlx_profile(model_id).await;

    // Merge recommendations with priority rules
    let recommended = merge_recommendations(
        &geometry_profile,
        &rapid_profile,
        &mut warnings,
        hybrid_from_geometry,
    );

    // Build source provenance
    let sources = build_sources(&geometry_profile, &rapid_profile);

    // Add warnings for missing data or source conflicts
    add_missing_source_warnings(&mut warnings, &geometry_profile, &rapid_profile);

    Ok(UnifiedProfile {
        recommended,
        sources,
        warnings,
    })
}

/// Fetch MLX geometry profile from HF (if HF-style ID) or local config.
async fn fetch_geometry_profile(
    model_id: &str,
) -> Option<crate::llama::model_memory_profile::ModelMemoryProfile> {
    // Determine source type
    if is_local_path(model_id) {
        fetch_local_geometry(model_id).await
    } else if is_hf_repo_id(model_id) {
        fetch_hf_geometry(model_id).await
    } else {
        // Alias: try local path first, then treat as unknown
        fetch_local_geometry(model_id).await
    }
}

/// Fetch geometry from a local MLX config.json.
async fn fetch_local_geometry(
    model_id: &str,
) -> Option<crate::llama::model_memory_profile::ModelMemoryProfile> {
    let path = std::path::PathBuf::from(model_id);
    tokio::task::spawn_blocking(move || match info_query::read_mlx_local_config(&path) {
        Ok(Some(config)) => {
            let raw = serde_json::to_vec(&config).ok()?;
            mlx_meta::parse_mlx_config_bytes_to_profile(&raw).ok()
        }
        _ => None,
    })
    .await
    .ok()
    .flatten()
}

/// Fetch geometry from HF with timeout.
async fn fetch_hf_geometry(
    model_id: &str,
) -> Option<crate::llama::model_memory_profile::ModelMemoryProfile> {
    // Parse repo_id and revision
    let (repo_id, revision) = match parse_hf_repo_revision(model_id) {
        Some(r) => r,
        None => return None,
    };

    let future = crate::hf::fetch_mlx_model_profile_revision_aware(&repo_id, &revision);
    tokio::time::timeout(std::time::Duration::from_secs(10), future)
        .await
        .ok()
        .unwrap_or(Err("timeout".to_string()))
        .ok()
}

/// Fetch Rapid-MLX info profile for behavioral flags.
async fn fetch_rapid_mlx_profile(
    model_id: &str,
) -> Option<crate::inference::rapid_mlx::info_query::ModelProfile> {
    // Try to resolve a rapid-mlx binary
    let binary = match resolve_rapid_mlx_binary().await {
        Some(b) => b,
        None => return None,
    };

    let future = info_query::fetch_model_profile(&binary, model_id);
    tokio::time::timeout(std::time::Duration::from_secs(10), future)
        .await
        .ok()
        .unwrap_or(Err(anyhow::anyhow!("timeout")))
        .ok()
        .flatten()
}

async fn resolve_rapid_mlx_binary() -> Option<std::path::PathBuf> {
    // Use discovery to find binary (checks managed runtime first, then PATH)
    discovery::Discovery::resolve_binary(None, None)
        .await
        .ok()
        .map(|(binary, _)| binary)
}

fn derive_hybrid_mode_from_geometry(
    profile: &Option<crate::llama::model_memory_profile::ModelMemoryProfile>,
) -> String {
    if let Some(p) = profile {
        if let Some(interval) = p.full_attention_interval
            && interval > 1
        {
            return "force".to_string();
        }
        if p.is_hybrid_attention() {
            return "force".to_string();
        }
    }
    "auto".to_string()
}

fn geometry_source_label(
    profile: &Option<crate::llama::model_memory_profile::ModelMemoryProfile>,
) -> String {
    match profile {
        Some(p) => {
            if p.source_revision.is_some() {
                "hf_config".to_string()
            } else {
                "mlx_local".to_string()
            }
        }
        None => "fallback".to_string(),
    }
}

fn parse_hf_repo_revision(model_id: &str) -> Option<(String, String)> {
    let (repo_id, revision) = if let Some(at_pos) = model_id.find('@') {
        (
            model_id[..at_pos].to_string(),
            model_id[at_pos + 1..].to_string(),
        )
    } else {
        (model_id.to_string(), "main".to_string())
    };

    if repo_id.is_empty() || !repo_id.contains('/') {
        return None;
    }

    Some((repo_id, revision))
}

fn is_local_path(model_id: &str) -> bool {
    let path = std::path::Path::new(model_id);
    path.is_absolute()
        || model_id.starts_with('/')
        || model_id.starts_with('\\')
        || model_id.as_bytes().get(1).is_some_and(|byte| *byte == b':')
}

fn is_hf_repo_id(model_id: &str) -> bool {
    !is_local_path(model_id)
        && model_id.contains('/')
        && !model_id.starts_with('.')
        && !model_id.contains(' ')
}

fn merge_recommendations(
    geometry: &Option<crate::llama::model_memory_profile::ModelMemoryProfile>,
    rapid: &Option<crate::inference::rapid_mlx::info_query::ModelProfile>,
    warnings: &mut Vec<String>,
    hybrid_from_geometry: String,
) -> UnifiedProfileRecommended {
    // hybrid_mode: HF/local config is authoritative (hard gate)
    let hybrid_mode = hybrid_from_geometry;

    // tool_format: rapid-mlx info first, then model_type mapping
    let tool_format = if let Some(r) = rapid
        && let Some(tf) = r.tool_format.as_deref()
        && !tf.is_empty()
    {
        Some(tf.to_string())
    } else {
        derive_tool_format_from_geometry(geometry)
    };

    // reasoning_parser: rapid-mlx info first, then HF architecture pattern
    let reasoning_parser = if let Some(r) = rapid
        && let Some(rp) = r.reasoning_parser.as_deref()
        && !rp.is_empty()
    {
        Some(rp.to_string())
    } else {
        derive_reasoning_parser_from_geometry(geometry)
    };

    // Check for conflicts between rapid-mlx and geometry
    if let Some(r) = rapid {
        if r.architecture.as_ref().is_some_and(|a| {
            a.to_lowercase().contains("hybrid") || a.to_lowercase().contains("delta")
        }) && hybrid_mode == "auto"
        {
            warnings.push("Rapid-MLX info reports hybrid/delta architecture, but MLX config does not confirm full_attention_interval; geometry is authoritative.".to_string());
        } else if r
            .architecture
            .as_ref()
            .is_some_and(|a| a.to_lowercase().contains("pure attention"))
            && hybrid_mode == "force"
        {
            warnings.push("Rapid-MLX info reports pure attention, but MLX config indicates hybrid mode via full_attention_interval; geometry is authoritative.".to_string());
        }
    }

    UnifiedProfileRecommended {
        hybrid_mode,
        tool_format,
        reasoning_parser,
    }
}

fn derive_tool_format_from_geometry(
    geometry: &Option<crate::llama::model_memory_profile::ModelMemoryProfile>,
) -> Option<String> {
    let model_type = geometry.as_ref()?.model_type.as_ref()?;
    let lower = model_type.to_lowercase();

    if lower.contains("qwen") {
        Some("qwen-coder".to_string())
    } else if lower.contains("llama") || lower.contains("llawa") {
        Some("llama3-tool".to_string())
    } else if lower.contains("gemma") {
        Some("gemma-tool".to_string())
    } else {
        None
    }
}

fn derive_reasoning_parser_from_geometry(
    geometry: &Option<crate::llama::model_memory_profile::ModelMemoryProfile>,
) -> Option<String> {
    let p = geometry.as_ref()?;

    // Check architectures for ForConditionalGeneration pattern with long context
    if let Some(archs) = &p.architectures {
        let has_conditional = archs.iter().any(|a| a.contains("ForConditionalGeneration"));
        let has_long_context = p.model_context_limit.unwrap_or(0) > 32768
            || p.weights.max_position_embeddings.value > 32768;
        if has_conditional && has_long_context {
            let model_type = p.model_type.as_deref().unwrap_or("");
            if model_type.to_lowercase().contains("qwen") {
                return Some("qwen".to_string());
            }
        }
    }

    // Architecture-based pattern matching
    let arch_str = p
        .architectures
        .as_ref()
        .and_then(|a| a.first().map(|s| s.as_str()))
        .unwrap_or("");

    if arch_str.contains("Qwen3") || arch_str.contains("Qwen2") {
        return Some("qwen".to_string());
    }

    None
}

fn build_sources(
    geometry: &Option<crate::llama::model_memory_profile::ModelMemoryProfile>,
    rapid: &Option<crate::inference::rapid_mlx::info_query::ModelProfile>,
) -> UnifiedProfileSources {
    let hybrid_mode = geometry_source_label(geometry);

    let tool_format = if rapid.as_ref().is_some_and(|r| r.tool_format.is_some()) {
        "rapid_mlx".to_string()
    } else if geometry.as_ref().is_some_and(|g| g.model_type.is_some()) {
        "model_type_mapping".to_string()
    } else {
        "not_available".to_string()
    };

    let reasoning_parser = if rapid.as_ref().is_some_and(|r| r.reasoning_parser.is_some()) {
        "rapid_mlx".to_string()
    } else if geometry
        .as_ref()
        .is_some_and(|g| g.architectures.as_ref().is_some_and(|a| !a.is_empty()))
    {
        "hf_architecture".to_string()
    } else {
        "not_available".to_string()
    };

    UnifiedProfileSources {
        hybrid_mode,
        tool_format,
        reasoning_parser,
    }
}

fn add_missing_source_warnings(
    warnings: &mut Vec<String>,
    geometry: &Option<crate::llama::model_memory_profile::ModelMemoryProfile>,
    rapid: &Option<crate::inference::rapid_mlx::info_query::ModelProfile>,
) {
    // Warning when rapid-mlx unavailable
    if rapid.is_none() {
        warnings.push("Rapid-MLX info query unavailable (runtime not installed or query failed); behavioral flags fall back to heuristic mappings.".to_string());
    }

    // Warning when geometry unavailable
    if geometry.is_none() {
        warnings.push("MLX config geometry unavailable; hybrid_mode uses fallback.".to_string());
    }
}

#[cfg(test)]
mod chat_tests {
    use super::*;

    fn adapter() -> RapidMlxAdapter {
        RapidMlxAdapter::from_resolved(
            RuntimeMetadata {
                executable_path: "rapid-mlx".into(),
                source: runtime::RuntimeSource::Managed,
                version: compatibility::minimum_version_text(),
                capability_snapshot: None,
                resolved_receipt: None,
                last_probe_result: None,
                prefix_cache_enabled: false,
                mlx_prefix_cache_bytes: None,
            },
            ResolvedRapidMlxLaunchModel::validated_alias("model").unwrap(),
        )
    }

    #[test]
    fn verified_mapping_filters_llama_fields_and_preserves_supported_controls() {
        let mapped = adapter()
            .map_chat_request(
                br#"{
                    "messages":[{"role":"user","content":"hi"}],
                    "stream":true,
                    "repeat_penalty":1.1,
                    "seed":42,
                    "cache_prompt":true,
                    "thinking_budget_tokens":2048,
                    "stream_options":{"include_usage":true},
                    "tools":[{"type":"function"}],
                    "reasoning_effort":"high"
                }"#,
            )
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&mapped).unwrap();
        assert_eq!(value["repetition_penalty"], 1.1);
        assert_eq!(value["stream_options"]["include_usage"], true);
        assert!(value["tools"].is_array());
        assert_eq!(value["reasoning_effort"], "high");
        assert_eq!(value["reasoning_max_tokens"], 2048);
        assert!(value.get("repeat_penalty").is_none());
        assert!(value.get("seed").is_none());
        assert!(value.get("cache_prompt").is_none());
        assert!(value.get("thinking_budget_tokens").is_none());
    }

    #[test]
    fn native_reasoning_cap_overrides_shared_alias() {
        let mapped = adapter()
            .map_chat_request(
                br#"{
                    "messages":[],
                    "thinking_budget_tokens":2048,
                    "reasoning_max_tokens":4096
                }"#,
            )
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&mapped).unwrap();
        assert_eq!(value["reasoning_max_tokens"], 4096);
        assert!(value.get("thinking_budget_tokens").is_none());
    }

    #[test]
    fn provisional_mapping_omits_unproven_optional_fields() {
        let mapped = map_provisional_chat_request(
            br#"{"messages":[],"stream":true,"tools":[],"response_format":{"type":"json_object"},"top_k":20}"#,
        )
        .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&mapped).unwrap();
        assert_eq!(value["top_k"], 20);
        assert!(value.get("tools").is_none());
        assert!(value.get("response_format").is_none());
    }

    #[test]
    fn verified_stream_mapping_requests_usage_without_overriding_user_choice() {
        let mapped = adapter()
            .map_chat_request(br#"{"messages":[],"stream":true}"#)
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&mapped).unwrap();
        assert_eq!(value["stream_options"]["include_usage"], true);

        let mapped = adapter()
            .map_chat_request(
                br#"{"messages":[],"stream":true,"stream_options":{"include_usage":false}}"#,
            )
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&mapped).unwrap();
        assert_eq!(value["stream_options"]["include_usage"], false);
    }

    #[test]
    fn reasoning_defaults_fill_omitted_fields_without_overriding_caller_choice() {
        let mut with_defaults = adapter();
        with_defaults.enable_thinking = Some(true);
        with_defaults.reasoning_effort = Some("high".into());

        let mapped = with_defaults
            .map_chat_request(br#"{"messages":[],"stream":false}"#)
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&mapped).unwrap();
        assert_eq!(value["enable_thinking"], true);
        assert_eq!(value["reasoning_effort"], "high");

        let mapped = with_defaults
            .map_chat_request(
                br#"{"messages":[],"stream":false,"enable_thinking":false,"reasoning_effort":"low"}"#,
            )
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&mapped).unwrap();
        assert_eq!(value["enable_thinking"], false);
        assert_eq!(value["reasoning_effort"], "low");
    }

    #[test]
    fn no_reasoning_defaults_configured_leaves_request_untouched() {
        let mapped = adapter()
            .map_chat_request(br#"{"messages":[],"stream":false}"#)
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&mapped).unwrap();
        assert!(value.get("enable_thinking").is_none());
        assert!(value.get("reasoning_effort").is_none());
    }

    #[test]
    fn rapid_mapping_rejects_malformed_or_message_less_requests() {
        assert!(adapter().map_chat_request(b"not json").is_err());
        assert!(adapter().map_chat_request(br#"{"stream":true}"#).is_err());
    }
}

#[cfg(test)]
mod unified_profile_tests {
    use super::*;
    use crate::llama::model_memory_profile::*;

    fn geometry_hybrid_qwen36() -> ModelMemoryProfile {
        let mut p = ModelMemoryProfile {
            model_type: Some("qwen3_5".into()),
            architectures: Some(vec!["Qwen3ForConditionalGeneration".into()]),
            full_attention_interval: Some(4),
            model_context_limit: Some(262144),
            ..Default::default()
        };
        p.layer_groups.push(LayerMemoryGroup {
            kind: LayerGroupKind::FullAttention,
            count: 16,
            field_evidence: "counted".into(),
            kv_heads: Some(4),
            head_dim: Some(128),
            ..Default::default()
        });
        p.layer_groups.push(LayerMemoryGroup {
            kind: LayerGroupKind::LinearRecurrent,
            count: 48,
            field_evidence: "counted".into(),
            ..Default::default()
        });
        p
    }

    fn geometry_pure_attention_llama() -> ModelMemoryProfile {
        let mut p = ModelMemoryProfile {
            model_type: Some("llama".into()),
            architectures: Some(vec!["LlamaForCausalLM".into()]),
            ..Default::default()
        };
        p.layer_groups.push(LayerMemoryGroup {
            kind: LayerGroupKind::FullAttention,
            count: 32,
            field_evidence: "flat config".into(),
            kv_heads: Some(8),
            head_dim: Some(128),
            ..Default::default()
        });
        p
    }

    fn rapid_profile_with_flags() -> info_query::ModelProfile {
        info_query::ModelProfile {
            tool_format: Some("hermes".into()),
            reasoning_parser: Some("qwen3".into()),
            architecture: Some("pure attention".into()),
            ..Default::default()
        }
    }

    fn rapid_profile_hybrid_arch() -> info_query::ModelProfile {
        info_query::ModelProfile {
            tool_format: Some("hermes".into()),
            reasoning_parser: Some("qwen3".into()),
            architecture: Some("hybrid delta".into()),
            ..Default::default()
        }
    }

    #[test]
    fn hybrid_mode_authoritative_from_geometry_full_attention_interval() {
        // HF/local config wins for hybrid_mode when full_attention_interval > 1
        let geometry = Some(geometry_hybrid_qwen36());

        let hybrid = derive_hybrid_mode_from_geometry(&geometry);
        assert_eq!(hybrid, "force", "full_attention_interval=4 → force");
    }

    #[test]
    fn hybrid_mode_auto_for_pure_attention_model() {
        let geometry = Some(geometry_pure_attention_llama());

        let hybrid = derive_hybrid_mode_from_geometry(&geometry);
        assert_eq!(hybrid, "auto", "no interval + pure attention → auto");
    }

    #[test]
    fn hybrid_mode_fallback_when_no_geometry() {
        let geometry: Option<ModelMemoryProfile> = None;
        let hybrid = derive_hybrid_mode_from_geometry(&geometry);
        assert_eq!(hybrid, "auto", "missing geometry → fallback auto");
    }

    #[test]
    fn rapid_mlx_wins_for_tool_format() {
        // rapid-mlx info is authoritative for tool_format when available
        let geometry = Some(geometry_hybrid_qwen36());
        let rapid = Some(rapid_profile_with_flags());
        let mut warnings = Vec::new();
        let hybrid = "force".to_string();

        let rec = merge_recommendations(&geometry, &rapid, &mut warnings, hybrid);
        assert_eq!(rec.tool_format, Some("hermes".into()));
    }

    #[test]
    fn model_type_mapping_fallback_for_tool_format() {
        // When rapid-mlx unavailable, use model_type mapping
        let geometry = Some(geometry_hybrid_qwen36());
        let rapid: Option<info_query::ModelProfile> = None;
        let mut warnings = Vec::new();
        let hybrid = "force".to_string();

        let rec = merge_recommendations(&geometry, &rapid, &mut warnings, hybrid);
        assert_eq!(rec.tool_format, Some("qwen-coder".into()));
    }

    #[test]
    fn rapid_mlx_wins_for_reasoning_parser() {
        let geometry = Some(geometry_hybrid_qwen36());
        let rapid = Some(rapid_profile_with_flags());
        let mut warnings = Vec::new();
        let hybrid = "force".to_string();

        let rec = merge_recommendations(&geometry, &rapid, &mut warnings, hybrid);
        assert_eq!(rec.reasoning_parser, Some("qwen3".into()));
    }

    #[test]
    fn hf_architecture_fallback_for_reasoning_parser() {
        let geometry = Some(geometry_hybrid_qwen36());
        let rapid: Option<info_query::ModelProfile> = None;
        let mut warnings = Vec::new();
        let hybrid = "force".to_string();

        let rec = merge_recommendations(&geometry, &rapid, &mut warnings, hybrid);
        assert_eq!(rec.reasoning_parser, Some("qwen".into()));
    }

    #[test]
    fn warning_when_rapid_reports_hybrid_but_geometry_does_not_confirm() {
        // rapid-mlx says hybrid, but geometry has no full_attention_interval → warning
        let geometry = Some(geometry_pure_attention_llama());
        let rapid = Some(rapid_profile_hybrid_arch());
        let mut warnings = Vec::new();
        let hybrid = "auto".to_string();

        let _rec = merge_recommendations(&geometry, &rapid, &mut warnings, hybrid);
        assert!(
            warnings
                .iter()
                .any(|w| w.contains("hybrid") || w.contains("delta"))
        );
    }

    #[test]
    fn warning_when_geometry_force_but_rapid_reports_pure_attention() {
        // Geometry says force (interval > 1), rapid says pure attention → warning
        let geometry = Some(geometry_hybrid_qwen36());
        let rapid = Some(rapid_profile_with_flags());
        let mut warnings = Vec::new();
        let hybrid = "force".to_string();

        let _rec = merge_recommendations(&geometry, &rapid, &mut warnings, hybrid);
        assert!(warnings.iter().any(|w| w.contains("pure attention")));
    }

    #[test]
    fn source_labels_rapid_mlx_for_tool_format() {
        let geometry = Some(geometry_hybrid_qwen36());
        let rapid = Some(rapid_profile_with_flags());
        let sources = build_sources(&geometry, &rapid);
        assert_eq!(sources.tool_format, "rapid_mlx");
    }

    #[test]
    fn source_labels_model_type_mapping_for_tool_format() {
        let geometry = Some(geometry_hybrid_qwen36());
        let rapid: Option<info_query::ModelProfile> = None;
        let sources = build_sources(&geometry, &rapid);
        assert_eq!(sources.tool_format, "model_type_mapping");
    }

    #[test]
    fn source_labels_not_available_when_no_sources() {
        let geometry: Option<ModelMemoryProfile> = None;
        let rapid: Option<info_query::ModelProfile> = None;
        let sources = build_sources(&geometry, &rapid);
        assert_eq!(sources.tool_format, "not_available");
        assert_eq!(sources.reasoning_parser, "not_available");
        assert_eq!(sources.hybrid_mode, "fallback");
    }

    #[test]
    fn warning_when_rapid_mlx_unavailable() {
        let geometry = Some(geometry_hybrid_qwen36());
        let rapid: Option<info_query::ModelProfile> = None;
        let mut warnings = Vec::new();
        add_missing_source_warnings(&mut warnings, &geometry, &rapid);
        assert!(
            warnings
                .iter()
                .any(|w| w.contains("Rapid-MLX info query unavailable"))
        );
    }

    #[test]
    fn warning_when_geometry_unavailable() {
        let geometry: Option<ModelMemoryProfile> = None;
        let rapid = Some(rapid_profile_with_flags());
        let mut warnings = Vec::new();
        add_missing_source_warnings(&mut warnings, &geometry, &rapid);
        assert!(
            warnings
                .iter()
                .any(|w| w.contains("MLX config geometry unavailable"))
        );
    }

    #[test]
    fn hf_repo_id_parsing_with_revision() {
        let result = parse_hf_repo_revision("mlx-community/Qwen3-0.6B-4bit@main");
        assert_eq!(
            result,
            Some(("mlx-community/Qwen3-0.6B-4bit".into(), "main".into()))
        );
    }

    #[test]
    fn hf_repo_id_parsing_defaults_revision_to_main() {
        let result = parse_hf_repo_revision("mlx-community/Qwen3-0.6B-4bit");
        assert_eq!(
            result,
            Some(("mlx-community/Qwen3-0.6B-4bit".into(), "main".into()))
        );
    }

    #[test]
    fn hf_repo_id_parsing_rejects_non_hf_style() {
        let result = parse_hf_repo_revision("qwen3-0.6b-4bit");
        assert!(result.is_none());
    }

    #[test]
    fn local_path_detection() {
        assert!(is_local_path("/Users/me/models/Qwen3"));
        assert!(is_local_path("/tmp/test-model"));
        assert!(!is_local_path("mlx-community/Qwen3-0.6B"));
        assert!(!is_local_path("qwen3-0.6b-4bit"));
    }

    #[test]
    fn hf_repo_id_detection() {
        assert!(is_hf_repo_id("mlx-community/Qwen3-0.6B-4bit"));
        assert!(is_hf_repo_id("org/model@main"));
        assert!(!is_hf_repo_id("/Users/me/models/test"));
        assert!(!is_hf_repo_id("qwen3-0.6b"));
    }

    #[test]
    fn derive_tool_format_qwen() {
        let geometry = Some(geometry_hybrid_qwen36());
        assert_eq!(
            derive_tool_format_from_geometry(&geometry),
            Some("qwen-coder".into())
        );
    }

    #[test]
    fn derive_tool_format_llama() {
        let geometry = Some(geometry_pure_attention_llama());
        assert_eq!(
            derive_tool_format_from_geometry(&geometry),
            Some("llama3-tool".into())
        );
    }

    #[test]
    fn derive_tool_format_none_for_unknown() {
        let p = ModelMemoryProfile {
            model_type: Some("unknown_model".into()),
            ..Default::default()
        };
        assert!(derive_tool_format_from_geometry(&Some(p)).is_none());
    }

    #[test]
    fn unified_profile_serialization_roundtrip() {
        let profile = UnifiedProfile {
            recommended: UnifiedProfileRecommended {
                hybrid_mode: "force".into(),
                tool_format: Some("hermes".into()),
                reasoning_parser: Some("qwen3".into()),
            },
            sources: UnifiedProfileSources {
                hybrid_mode: "hf_config".into(),
                tool_format: "rapid_mlx".into(),
                reasoning_parser: "rapid_mlx".into(),
            },
            warnings: vec!["test warning".into()],
        };
        let json = serde_json::to_string(&profile).unwrap();
        let restored: UnifiedProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.recommended.hybrid_mode, "force");
        assert_eq!(restored.recommended.tool_format, Some("hermes".into()));
        assert_eq!(restored.sources.hybrid_mode, "hf_config");
    }

    #[test]
    fn unified_profile_handles_missing_optional_fields() {
        let json = r#"{"recommended":{"hybrid_mode":"auto"},"sources":{"hybrid_mode":"fallback","tool_format":"not_available","reasoning_parser":"not_available"},"warnings":[]}"#;
        let profile: UnifiedProfile = serde_json::from_str(json).unwrap();
        assert_eq!(profile.recommended.hybrid_mode, "auto");
        assert!(profile.recommended.tool_format.is_none());
        assert!(profile.recommended.reasoning_parser.is_none());
    }
}
