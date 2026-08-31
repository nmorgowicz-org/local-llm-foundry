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
use crate::inference::metrics::{HealthState, InferenceMetricsSnapshot};
use crate::inference::supervisor::SupervisedLaunch;
use crate::llama::metrics::{parse_prometheus_metrics, parse_slot_metrics};

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
}

#[derive(Debug, Clone, Default)]
pub struct CounterSnapshot {
    prompt_tokens_total: f64,
    prompt_seconds_total: f64,
    predicted_tokens_total: f64,
    predicted_seconds_total: f64,
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
    previous_counters: Mutex<Option<CounterSnapshot>>,
    previous_counter_session: Mutex<Option<String>>,
}

#[allow(dead_code)]
impl LlamaCppAdapter {
    pub fn new(app_config: AppConfig, config: ServerConfig, gpu_env: GpuEnv) -> Self {
        Self {
            app_config,
            config,
            gpu_env,
            previous_counters: Mutex::new(None),
            previous_counter_session: Mutex::new(None),
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
        if let Some(n) = self.config.n_cpu_moe {
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
        let mut previous_counters = self.previous_counters.lock().unwrap().clone();
        let mut previous_counter_session = self.previous_counter_session.lock().unwrap().clone();
        let result = poll_llama_cpp_metrics(
            base,
            self.config.api_key.as_deref(),
            session_id,
            &mut previous_counters,
            &mut previous_counter_session,
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
) -> Result<InferenceMetricsSnapshot> {
    {
        let client = Client::builder()
            .timeout(Duration::from_secs(5))
            .pool_max_idle_per_host(0)
            .pool_idle_timeout(Duration::from_secs(0))
            .build()?;

        let mut snapshot = InferenceMetricsSnapshot {
            sampled_at: std::time::SystemTime::now(),
            backend: InferenceBackend::LlamaCpp,
            health: None,
            ready: None,
            model: None,
            uptime_seconds: None,
            generation_tokens_per_second: None,
            prompt_tokens_per_second: None,
            running_requests: None,
            waiting_requests: None,
            completed_requests_total: None,
            prompt_tokens_total: None,
            completion_tokens_total: None,
            steps_executed: None,
            global_cache_hit_rate: None,
            global_cache_entries: None,
            ttft: None,
            speculative_acceptance_rate: None,
            active_memory_bytes: None,
            peak_memory_bytes: None,
            cache_memory_bytes: None,
            cache_metrics: None,
            active_requests: None,
            backend_details: None,
        };

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
        let metrics_req = if let Some(key) = api_key {
            client
                .get(format!("{base}/metrics"))
                .header("Authorization", format!("Bearer {}", key))
        } else {
            client.get(format!("{base}/metrics"))
        };

        if let Ok(resp) = metrics_req.send().await
            && let Ok(body) = resp.text().await
        {
            let prom = parse_prometheus_metrics(&body);
            snapshot.prompt_tokens_total = Some(prom.prompt_tokens_total as u64);
            snapshot.completion_tokens_total = Some(prom.predicted_tokens_total as u64);
            snapshot.running_requests = Some(prom.requests_processing as u64);
            snapshot.steps_executed = Some(prom.n_decode_total as u64);
            tokens_per_decode = prom.tokens_per_decode;
            n_busy_slots_per_decode = prom.n_busy_slots_per_decode;
            snapshot.speculative_acceptance_rate = (prom.speculative_draft_tokens_total > 0.0)
                .then(|| {
                    prom.speculative_accepted_tokens_total / prom.speculative_draft_tokens_total
                });

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
            && let Ok(body) = resp.text().await
            && let Some(slots) = parse_slot_metrics(&body)
        {
            snapshot.backend_details = Some(serde_json::json!({
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
                "slots": slots.slots,
                "tokens_per_decode": tokens_per_decode,
                "n_busy_slots_per_decode": n_busy_slots_per_decode,
                "speculative_acceptance_rate": snapshot.speculative_acceptance_rate,
            }));
        }

        // Models metadata
        let models_req = if let Some(key) = api_key {
            client
                .get(format!("{base}/v1/models"))
                .header("Authorization", format!("Bearer {}", key))
        } else {
            client.get(format!("{base}/v1/models"))
        };

        if let Ok(resp) = models_req.send().await
            && let Ok(body) = resp.text().await
            && let Ok(json) = serde_json::from_str::<serde_json::Value>(&body)
            && let Some(model) = json["data"][0].get("id").and_then(|v| v.as_str())
        {
            snapshot.model = Some(model.to_string());
        }

        Ok(snapshot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

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
