use crate::inference::rapid_mlx::compatibility::ServeCapabilities;
use crate::inference::rapid_mlx::model_resolver::{
    RapidMlxModelSource, ResolvedRapidMlxLaunchModel,
};
use crate::inference::rapid_mlx::{RapidMlxHybridMode, RapidMlxSpeculativeConfig};
use crate::inference::supervisor::SupervisedLaunch;
use anyhow::Result;
use std::ffi::OsString;
use std::path::PathBuf;

/// Reject any argv value that could be mistaken for another option or corrupt
/// the command line. Values reach the runtime as single argv entries, so the
/// dangerous cases are option injection (a leading `-`) and whitespace or
/// control characters that split or disguise the token.
/// Values `rapid-mlx serve --default-reasoning-effort` accepts.
pub(crate) const REASONING_EFFORTS: [&str; 6] =
    ["none", "minimal", "low", "medium", "high", "xhigh"];

fn validate_launch_argument(argument: &str) -> Result<()> {
    if argument.is_empty() {
        anyhow::bail!("model launch argument must not be empty");
    }
    if argument.starts_with('-') {
        anyhow::bail!("model launch argument must not start with '-'; got {argument:?}");
    }
    Ok(())
}

fn validate_served_model_name(name: &str) -> Result<()> {
    if name.is_empty() || name.starts_with('-') {
        anyhow::bail!("served model name must be non-empty and not start with '-'; got {name:?}");
    }
    if name.chars().any(char::is_whitespace) {
        anyhow::bail!("served model name must not contain whitespace; got {name:?}");
    }
    Ok(())
}

fn validate_host(host: &str) -> Result<()> {
    if host.is_empty() {
        anyhow::bail!("host must not be empty");
    }
    if host.chars().any(|c| c.is_whitespace() || c.is_control()) {
        anyhow::bail!("host must not contain whitespace or control characters; got {host:?}");
    }
    Ok(())
}

fn validate_safe_token(label: &str, value: &str) -> Result<()> {
    if value.is_empty() || value.starts_with('-') {
        anyhow::bail!("{label} must be non-empty and not start with '-'; got {value:?}");
    }
    if value.chars().any(|c| c.is_whitespace() || c.is_control()) {
        anyhow::bail!(
            "{label} must be a single token without whitespace or control characters; got {value:?}"
        );
    }
    Ok(())
}

pub struct RapidMlxCommandBuilder {
    model: ResolvedRapidMlxLaunchModel,
    served_model_name: Option<String>,
    host: String,
    port: u16,
    log_level: Option<String>,
    timeout: Option<u32>,
    api_key: Option<String>,
    tool_call_parser: Option<String>,
    reasoning_parser: Option<String>,
    auto_tool_choice: bool,
    no_thinking: bool,
    trust_remote_code_consent: Option<String>,
    escape_hatch_flags: Vec<(String, serde_json::Value)>,
    // Phase 7: KV/cache policy
    kv_cache_dtype: Option<KvCacheDtypeArg>,
    turboquant_mode: Option<String>,
    hybrid_cache_entries: Option<u64>,
    hybrid_mode: RapidMlxHybridMode,
    pflash_policy: Option<String>,
    retained_cache_mib: Option<u32>,
    prefix_cache_enabled: Option<bool>,
    disk_checkpoint_interval: Option<u32>,
    // Phase 7: batching/concurrency
    max_num_seqs: Option<u64>,
    max_concurrent_requests: Option<u64>,
    prefill_batch_size: Option<u64>,
    completion_batch_size: Option<u64>,
    context_length: Option<u32>,
    default_reasoning_effort: Option<String>,
    prefill_step_size: Option<u32>,
    // Phase 7: reasoning/speculative
    reasoning_mode: Option<String>,
    speculative_config: Option<RapidMlxSpeculativeConfig>,
    // Phase 7: MLLM/embeddings
    mllm_vision: Option<String>,
    embeddings: Option<String>,
    // Phase 7: GPU
    gpu_memory_utilization: Option<f64>,
    // Phase 7: Web UI
    // Phase 7: endpoint/safety
    sampling_mode: Option<String>,
    default_temperature: Option<f64>,
    default_top_p: Option<f64>,
    default_top_k: Option<u64>,
    default_min_p: Option<f64>,
    default_repetition_penalty: Option<f64>,
    default_presence_penalty: Option<f64>,
    default_frequency_penalty: Option<f64>,
    max_tokens: Option<u64>,
}

/// KV cache dtype argument for CLI.
#[derive(Debug, Clone)]
pub enum KvCacheDtypeArg {
    Auto,
    Explicit(String),
}

impl RapidMlxCommandBuilder {
    pub fn new(model: ResolvedRapidMlxLaunchModel) -> Self {
        Self {
            model,
            served_model_name: None,
            host: "127.0.0.1".to_string(),
            port: 8000,
            log_level: None,
            timeout: None,
            api_key: None,
            tool_call_parser: None,
            reasoning_parser: None,
            auto_tool_choice: false,
            no_thinking: false,
            trust_remote_code_consent: None,
            escape_hatch_flags: Vec::new(),
            // Phase 7 defaults
            kv_cache_dtype: None,
            turboquant_mode: None,
            hybrid_cache_entries: None,
            hybrid_mode: RapidMlxHybridMode::Auto,
            pflash_policy: None,
            retained_cache_mib: None,
            prefix_cache_enabled: None,
            disk_checkpoint_interval: None,
            max_num_seqs: None,
            max_concurrent_requests: None,
            prefill_batch_size: None,
            completion_batch_size: None,
            context_length: None,
            default_reasoning_effort: None,
            prefill_step_size: None,
            reasoning_mode: None,
            speculative_config: None,
            mllm_vision: None,
            embeddings: None,
            gpu_memory_utilization: None,
            sampling_mode: None,
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

    pub fn served_model_name(mut self, name: String) -> Self {
        self.served_model_name = Some(name);
        self
    }

    pub fn host(mut self, host: String) -> Self {
        self.host = host;
        self
    }

    pub fn port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    pub fn log_level(mut self, level: String) -> Self {
        self.log_level = Some(level);
        self
    }

    pub fn timeout(mut self, timeout: u32) -> Self {
        self.timeout = Some(timeout);
        self
    }

    pub fn api_key(mut self, api_key: String) -> Self {
        self.api_key = Some(api_key);
        self
    }

    pub fn tool_call_parser(mut self, parser: Option<String>) -> Self {
        self.tool_call_parser = parser;
        self
    }

    pub fn reasoning_parser(mut self, parser: Option<String>) -> Self {
        self.reasoning_parser = parser;
        self
    }

    pub fn auto_tool_choice(mut self, enable: bool) -> Self {
        self.auto_tool_choice = enable;
        self
    }

    pub fn no_thinking(mut self, enable: bool) -> Self {
        self.no_thinking = enable;
        self
    }

    pub fn trust_remote_code_consent(mut self, consent: Option<String>) -> Self {
        self.trust_remote_code_consent = consent;
        self
    }

    pub fn escape_hatch_flags(mut self, flags: Vec<(String, serde_json::Value)>) -> Self {
        self.escape_hatch_flags = flags;
        self
    }
    // Phase 7 setters
    pub fn kv_cache_dtype(mut self, dtype: Option<KvCacheDtypeArg>) -> Self {
        self.kv_cache_dtype = dtype;
        self
    }
    pub fn turboquant_mode(mut self, mode: Option<String>) -> Self {
        self.turboquant_mode = mode;
        self
    }
    pub fn hybrid_cache_entries(mut self, entries: Option<u64>) -> Self {
        self.hybrid_cache_entries = entries;
        self
    }
    pub fn hybrid_mode(mut self, mode: RapidMlxHybridMode) -> Self {
        self.hybrid_mode = mode;
        self
    }
    pub fn pflash_policy(mut self, policy: Option<String>) -> Self {
        self.pflash_policy = policy;
        self
    }
    pub fn retained_cache_mib(mut self, mib: Option<u32>) -> Self {
        self.retained_cache_mib = mib;
        self
    }
    pub fn prefix_cache_enabled(mut self, enabled: Option<bool>) -> Self {
        self.prefix_cache_enabled = enabled;
        self
    }
    pub fn disk_checkpoint_interval(mut self, interval: Option<u32>) -> Self {
        self.disk_checkpoint_interval = interval;
        self
    }
    pub fn max_num_seqs(mut self, seqs: Option<u64>) -> Self {
        self.max_num_seqs = seqs;
        self
    }
    pub fn max_concurrent_requests(mut self, requests: Option<u64>) -> Self {
        self.max_concurrent_requests = requests;
        self
    }
    pub fn prefill_batch_size(mut self, size: Option<u64>) -> Self {
        self.prefill_batch_size = size;
        self
    }
    pub fn completion_batch_size(mut self, size: Option<u64>) -> Self {
        self.completion_batch_size = size;
        self
    }

    pub fn default_reasoning_effort(mut self, effort: Option<String>) -> Self {
        self.default_reasoning_effort = effort;
        self
    }

    pub fn context_length(mut self, tokens: Option<u32>) -> Self {
        self.context_length = tokens;
        self
    }
    pub fn prefill_step_size(mut self, size: Option<u32>) -> Self {
        self.prefill_step_size = size;
        self
    }
    pub fn reasoning_mode(mut self, mode: Option<String>) -> Self {
        self.reasoning_mode = mode;
        self
    }

    pub fn speculative_config(mut self, config: Option<RapidMlxSpeculativeConfig>) -> Self {
        self.speculative_config = config;
        self
    }
    pub fn mllm_vision(mut self, vision: Option<String>) -> Self {
        self.mllm_vision = vision;
        self
    }
    pub fn embeddings(mut self, emb: Option<String>) -> Self {
        self.embeddings = emb;
        self
    }
    pub fn gpu_memory_utilization(mut self, util: Option<f64>) -> Self {
        self.gpu_memory_utilization = util;
        self
    }
    pub fn sampling_mode(mut self, mode: Option<String>) -> Self {
        self.sampling_mode = mode;
        self
    }
    #[allow(clippy::too_many_arguments)]
    pub fn sampling_defaults(
        mut self,
        temperature: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<u64>,
        min_p: Option<f64>,
        repetition_penalty: Option<f64>,
        presence_penalty: Option<f64>,
        frequency_penalty: Option<f64>,
        max_tokens: Option<u64>,
    ) -> Self {
        self.default_temperature = temperature;
        self.default_top_p = top_p;
        self.default_top_k = top_k;
        self.default_min_p = min_p;
        self.default_repetition_penalty = repetition_penalty;
        self.default_presence_penalty = presence_penalty;
        self.default_frequency_penalty = frequency_penalty;
        self.max_tokens = max_tokens;
        self
    }

    /// Reject every value that could be mistaken for another option, corrupt argv, or make
    /// the runtime misbehave. Called first in `build()` so the guarantee is local to the
    /// builder rather than dependent on callers having validated beforehand.
    fn validate(&self) -> Result<()> {
        validate_launch_argument(&self.model.launch_argument)?;
        if let Some(name) = &self.served_model_name {
            validate_served_model_name(name)?;
        }
        validate_host(&self.host)?;
        for (label, value) in [
            ("log_level", &self.log_level),
            ("tool_call_parser", &self.tool_call_parser),
            ("reasoning_parser", &self.reasoning_parser),
        ] {
            if let Some(value) = value {
                validate_safe_token(label, value)?;
            }
        }
        if let Some(mode) = &self.turboquant_mode
            && !matches!(mode.as_str(), "auto" | "none" | "v4" | "k8v4")
        {
            anyhow::bail!("turboquant_mode must be one of auto, none, v4, k8v4; got {mode:?}");
        }
        if let Some(util) = self.gpu_memory_utilization
            && !(util.is_finite() && util > 0.0 && util <= 1.0)
        {
            anyhow::bail!(
                "gpu_memory_utilization must be greater than 0 and at most 1; got {util}"
            );
        }
        let in_range = |label: &str, value: Option<f64>, min: f64, max: f64| -> Result<()> {
            if let Some(v) = value
                && !(v.is_finite() && v >= min && v <= max)
            {
                anyhow::bail!("{label} must be a finite number in [{min}, {max}]; got {v}");
            }
            Ok(())
        };
        in_range("default_temperature", self.default_temperature, 0.0, 10.0)?;
        in_range("default_top_p", self.default_top_p, 0.0, 1.0)?;
        in_range("default_min_p", self.default_min_p, 0.0, 1.0)?;
        in_range(
            "default_repetition_penalty",
            self.default_repetition_penalty,
            0.0,
            10.0,
        )?;
        in_range(
            "default_presence_penalty",
            self.default_presence_penalty,
            -2.0,
            2.0,
        )?;
        in_range(
            "default_frequency_penalty",
            self.default_frequency_penalty,
            -2.0,
            2.0,
        )?;
        if self.max_tokens == Some(0) {
            anyhow::bail!("max_tokens must be at least 1");
        }
        if let Some(size) = self.prefill_step_size
            && !(1..=2048).contains(&size)
        {
            anyhow::bail!("prefill_step_size must be between 1 and 2048");
        }
        Ok(())
    }

    /// Validate the escape-hatch flags and turn them into argv. Names and values are checked
    /// against the allowlist descriptors, each flag must be probed as supported, duplicates are
    /// refused, and a flag that a typed setting already controls is refused instead of being
    /// emitted twice with an argparse-last-wins result.
    fn escape_hatch_args(&self, capabilities: &ServeCapabilities) -> Result<Vec<String>> {
        use crate::inference::rapid_mlx::escape_hatch::escape_flag_to_args;
        if self.escape_hatch_flags.is_empty() {
            return Ok(Vec::new());
        }
        crate::inference::rapid_mlx::escape_hatch::validate_escape_flag_values(
            &self.escape_hatch_flags,
        )
        .map_err(|message| anyhow::anyhow!(message))?;

        let mut seen = std::collections::HashSet::new();
        for (name, _) in &self.escape_hatch_flags {
            if !seen.insert(name.as_str()) {
                anyhow::bail!("escape flag --{name} is specified more than once");
            }
        }
        let active = |flag: &str| {
            self.escape_hatch_flags
                .iter()
                .any(|(name, value)| name == flag && value == &serde_json::Value::Bool(true))
        };
        if active("force-hybrid") && active("no-hybrid") {
            anyhow::bail!("--force-hybrid and --no-hybrid are mutually exclusive");
        }
        if self.hybrid_mode != RapidMlxHybridMode::Auto
            && (active("force-hybrid") || active("no-hybrid"))
        {
            anyhow::bail!(
                "hybrid_mode cannot be combined with legacy force-hybrid/no-hybrid escape flags"
            );
        }
        if self
            .escape_hatch_flags
            .iter()
            .any(|(name, _)| name == "pflash")
            && self
                .pflash_policy
                .as_deref()
                .is_some_and(|policy| policy != "auto")
        {
            anyhow::bail!(
                "escape flag --pflash conflicts with the typed pflash_policy setting; set pflash_policy to \"auto\" or remove the escape flag"
            );
        }

        let mut args = Vec::new();
        for (name, value) in &self.escape_hatch_flags {
            let fragment = escape_flag_to_args(name, value).map_err(|m| anyhow::anyhow!(m))?;
            if fragment.is_empty() {
                continue; // `false` switch: nothing to emit, nothing to probe.
            }
            let flag = format!("--{name}");
            if !capabilities.contains(&flag) {
                anyhow::bail!(
                    "Installed Rapid-MLX does not support escape-hatch option {flag}; select a compatible runtime or remove that option"
                );
            }
            args.extend(fragment);
        }
        Ok(args)
    }

    pub fn build(
        self,
        binary_path: PathBuf,
        capabilities: &ServeCapabilities,
    ) -> Result<SupervisedLaunch> {
        // Reject unsafe values before any argv is assembled.
        self.validate()?;
        // Escape-hatch flags go through the strict path: allowlist
        // validation, capability probing, duplicate refusal, and conflicts
        // with typed settings are all enforced before any argv is emitted.
        // Computed up-front because `self` is partially moved below.
        let escape_args = self.escape_hatch_args(capabilities)?;
        let mut args = vec!["serve".to_string()];
        args.push(self.model.launch_argument.clone());

        if let Some(name) = self.served_model_name {
            capabilities.require("--served-model-name")?;
            args.push("--served-model-name".to_string());
            args.push(name);
        }

        capabilities.require("--host")?;
        args.push("--host".to_string());
        args.push(self.host);

        capabilities.require("--port")?;
        args.push("--port".to_string());
        args.push(self.port.to_string());

        if let Some(log_level) = self.log_level {
            capabilities.require("--log-level")?;
            args.push("--log-level".to_string());
            args.push(log_level);
        }

        if let Some(timeout) = self.timeout {
            capabilities.require("--timeout")?;
            args.push("--timeout".to_string());
            args.push(timeout.to_string());
        }

        // Diagnostic fix flags — not guarded by capability checks since they are
        // only activated by the diagnostics panel, never by default.
        // The runtime exits with "--enable-auto-tool-choice requires --tool-call-parser" when
        // the parser is missing. "Auto" in the UI stores no parser, so ask the runtime to
        // detect it from the alias profile (`--tool-call-parser auto`).
        let tool_call_parser = match self.tool_call_parser {
            Some(parser) => Some(parser),
            None if self.auto_tool_choice => Some("auto".to_string()),
            None => None,
        };
        if let Some(parser) = tool_call_parser {
            capabilities.require("--tool-call-parser")?;
            args.push("--tool-call-parser".to_string());
            args.push(parser);
        }
        if let Some(parser) = self.reasoning_parser {
            capabilities.require("--reasoning-parser")?;
            args.push("--reasoning-parser".to_string());
            args.push(parser);
        }
        if self.auto_tool_choice {
            capabilities.require("--enable-auto-tool-choice")?;
            args.push("--enable-auto-tool-choice".to_string());
        }
        // Escape-hatch flags were validated and converted above.
        args.extend(escape_args);
        // Phase 7: KV/cache policy flags
        if let Some(ref dtype) = self.kv_cache_dtype {
            match dtype {
                KvCacheDtypeArg::Explicit(effective) => {
                    if !matches!(effective.as_str(), "bf16" | "int8" | "int4") {
                        return Err(anyhow::anyhow!(
                            "Unsupported legacy Rapid-MLX kv_cache_dtype '{effective}'. Choose bf16, int8, or int4 before launching."
                        ));
                    }
                    capabilities.require("--kv-cache-dtype")?;
                    args.push("--kv-cache-dtype".to_string());
                    args.push(effective.clone());
                }
                KvCacheDtypeArg::Auto => {}
            }
        }
        if let Some(ref mode) = self.turboquant_mode
            && *mode != "none"
        {
            capabilities.require("--kv-cache-turboquant")?;
            args.push("--kv-cache-turboquant".to_string());
            args.push(mode.clone());
        }
        if let Some(enabled) = self.prefix_cache_enabled {
            let flag = if enabled {
                "--enable-prefix-cache"
            } else {
                "--disable-prefix-cache"
            };
            capabilities.require(flag)?;
            args.push(flag.to_string());
        }
        if let Some(mib) = self.retained_cache_mib.filter(|mib| *mib > 0) {
            capabilities.require("--cache-memory-mb")?;
            args.push("--cache-memory-mb".to_string());
            args.push(mib.to_string());
        }
        if let Some(entries) = self.hybrid_cache_entries {
            capabilities.require("--hybrid-cache-entries")?;
            args.push("--hybrid-cache-entries".to_string());
            args.push(entries.to_string());
        }
        match self.hybrid_mode {
            RapidMlxHybridMode::Auto => {}
            RapidMlxHybridMode::Force => {
                capabilities.require("--force-hybrid")?;
                args.push("--force-hybrid".to_string());
            }
            RapidMlxHybridMode::Disable => {
                capabilities.require("--no-hybrid")?;
                args.push("--no-hybrid".to_string());
            }
        }
        if let Some(ref policy) = self.pflash_policy
            && policy != "auto"
        {
            // A runtime with no `--pflash` flag has no PFlash to switch off, so the request is
            // already satisfied. Requiring the flag there would fail launches on older builds
            // for a setting that now defaults to "off" on every config.
            if policy != "off" || capabilities.contains("--pflash") {
                capabilities.require("--pflash")?;
                args.push("--pflash".to_string());
                args.push(policy.clone());
            }
        }
        if let Some(interval) = self.disk_checkpoint_interval {
            capabilities.require("--kv-disk-checkpoint-interval")?;
            args.push("--kv-disk-checkpoint-interval".to_string());
            args.push(interval.to_string());
        }
        // Phase 7: batching/concurrency flags.
        //
        // Omit-and-report, like PFlash, TurboQuant and speculative decoding above: these are
        // throughput tuning, so a runtime that lacks the flag should fall back to its own
        // scheduling rather than fail the launch. They used to call require(), which aborts
        // the whole build -- and since neither flag is in verified_baseline(), picking a
        // value in the UI blanked the command preview outright, taking the
        // requested-vs-effective diagnostics for every unrelated setting with it.
        if let Some(seqs) = self.max_num_seqs
            && capabilities.contains("--max-num-seqs")
        {
            args.push("--max-num-seqs".to_string());
            args.push(seqs.to_string());
        }
        if let Some(requests) = self.max_concurrent_requests
            && capabilities.contains("--max-concurrent-requests")
        {
            args.push("--max-concurrent-requests".to_string());
            args.push(requests.to_string());
        }
        if let Some(size) = self.prefill_batch_size {
            capabilities.require("--prefill-batch-size")?;
            args.push("--prefill-batch-size".to_string());
            args.push(size.to_string());
        }
        if let Some(size) = self.completion_batch_size {
            capabilities.require("--completion-batch-size")?;
            args.push("--completion-batch-size".to_string());
            args.push(size.to_string());
        }
        if let Some(tokens) = self.context_length.filter(|tokens| *tokens > 0) {
            capabilities.require("--context-length")?;
            args.push("--context-length".to_string());
            args.push(tokens.to_string());
        }
        if let Some(effort) = self
            .default_reasoning_effort
            .as_deref()
            .filter(|effort| !effort.is_empty())
        {
            if !REASONING_EFFORTS.contains(&effort) {
                anyhow::bail!(
                    "default_reasoning_effort must be one of {}; got {effort:?}",
                    REASONING_EFFORTS.join(", ")
                );
            }
            capabilities.require("--default-reasoning-effort")?;
            args.push("--default-reasoning-effort".to_string());
            args.push(effort.to_string());
        }
        if let Some(size) = self.prefill_step_size {
            if !(1..=2048).contains(&size) {
                anyhow::bail!("prefill_step_size must be between 1 and 2048");
            }
            capabilities.require("--prefill-step-size")?;
            args.push("--prefill-step-size".to_string());
            args.push(size.to_string());
        }
        // Rapid's --reasoning selects the qualified reasoning/KV quality profile; it does
        // not mean "show thinking". Always request that profile. An explicit thinking
        // opt-out independently adds --no-thinking for parser/chat-template behavior.
        let thinking_disabled = match self.reasoning_mode.as_deref().unwrap_or("on") {
            "on" => self.no_thinking,
            "off" => true,
            value => anyhow::bail!("reasoning_mode must be on or off; got {value:?}"),
        };
        capabilities.require("--reasoning")?;
        args.push("--reasoning".to_string());
        if thinking_disabled {
            capabilities.require("--no-thinking")?;
            args.push("--no-thinking".to_string());
        }
        // Speculative decoding is opt-in and typed. An older runtime must not blank an
        // otherwise useful preview or suppress unrelated settings: omit the unsupported
        // throughput feature and let requested_vs_effective explain the downgrade.
        if let Some(ref config) = self.speculative_config
            && capabilities.contains("--speculative-config")
        {
            // Rapid-MLX 0.11.1 resolves an HF repo sidecar with an unpinned
            // snapshot_download(repo_id). It has no revision field in the MTP schema, so
            // forwarding owner/repo would execute a mutable artifact different from the
            // revision recorded by llama-monitor's preflight/consent flow. Only an immutable
            // local companion path may launch until the runtime grows a pinned-revision API.
            if let Some(repo) = config.companion_model_repo_id() {
                anyhow::bail!(
                    "External MTP companion {repo} cannot launch safely: Rapid-MLX 0.11.1 does \
                     not support revision-pinned Hugging Face sidecars. Build or select an \
                     immutable local sidecar directory instead."
                );
            }
            args.push("--speculative-config".to_string());
            args.push(config.to_cli_json()?);

            // Estimate companion VRAM for local paths at launch time.
            // For HF repos this is handled by the preflight + pin cache.
            if let Some(companion_path) = config.companion_model_local_path() {
                let companion = std::path::Path::new(companion_path);
                if let Some(vram) =
                    crate::inference::rapid_mlx::sidecar_inventory::estimate_local_companion_vram(
                        companion,
                    )
                {
                    let mb = vram / 1_048_576;
                    eprintln!(
                        "MTP companion VRAM estimate (local path): {mb} MB from {companion_path}"
                    );
                }
            }
        }

        // Vision has only the real Rapid-MLX tri-state: Auto omits a flag,
        // On forces MLLM, and Off forces the text lane. A model-specific smoke
        // test still owns whether Auto is actually qualified.
        match self.mllm_vision.as_deref() {
            None | Some("auto") => {}
            Some("on") => {
                capabilities.require("--mllm")?;
                args.push("--mllm".to_string());
            }
            Some("off") => {
                capabilities.require("--no-mllm")?;
                args.push("--no-mllm".to_string());
            }
            Some(value) => anyhow::bail!("mllm_vision must be auto, on, or off; got {value:?}"),
        }
        if let Some(ref emb) = self.embeddings
            && emb != "auto"
        {
            anyhow::bail!(
                "embeddings is not a Rapid-MLX on/off launch setting; configure a qualified embedding model before enabling it"
            );
        }
        // Phase 7: GPU flags. Same omit-and-report treatment as the batching flags above --
        // a memory-utilisation hint is not worth failing a launch over.
        if let Some(util) = self.gpu_memory_utilization
            && capabilities.contains("--gpu-memory-utilization")
        {
            args.push("--gpu-memory-utilization".to_string());
            args.push(util.to_string());
        }
        // `sampling_mode` is persisted selection metadata. Phase 2 deliberately
        // does not turn it into a Rapid argv flag: per-field server-default
        // support is runtime-qualified in Phase 3, and an unqualified selection
        // must never create an unsupported launch argument.
        for (flag, value) in [
            (
                "--default-temperature",
                self.default_temperature.map(|v| v.to_string()),
            ),
            ("--default-top-p", self.default_top_p.map(|v| v.to_string())),
            ("--default-top-k", self.default_top_k.map(|v| v.to_string())),
            ("--default-min-p", self.default_min_p.map(|v| v.to_string())),
            (
                "--default-repetition-penalty",
                self.default_repetition_penalty.map(|v| v.to_string()),
            ),
            (
                "--default-presence-penalty",
                self.default_presence_penalty.map(|v| v.to_string()),
            ),
            (
                "--default-frequency-penalty",
                self.default_frequency_penalty.map(|v| v.to_string()),
            ),
            ("--max-tokens", self.max_tokens.map(|v| v.to_string())),
        ] {
            if let Some(value) = value {
                capabilities.require(flag)?;
                args.push(flag.to_string());
                args.push(value);
            }
        }

        let os_args: Vec<OsString> = args.into_iter().map(OsString::from).collect();

        // Prevent Rapid-MLX's first-run telemetry question from blocking an
        // unattended app launch. The user can opt in outside this process.
        let mut env = vec![(OsString::from("RAPID_MLX_TELEMETRY"), OsString::from("0"))];
        if let Some(key) = self.api_key {
            env.push((OsString::from("RAPID_MLX_API_KEY"), OsString::from(key)));
        }
        env.extend(
            self.model
                .environment()
                .map(|(name, value)| (name.clone(), value.clone())),
        );

        // Security: enforce revision-scoped consent for repos requiring trust_remote_code.
        // When the resolved model marks trust_remote_code_required=true, launch is blocked
        // unless the user has explicitly consented for that specific repo@revision.
        let mut trust_remote_code_enabled = false;
        if self.model.trust_remote_code_required == Some(true) {
            validate_trust_consent(&self.model, &self.trust_remote_code_consent)?;
            trust_remote_code_enabled = true;
        }

        // Also validate trust consent for the MTP companion model if it's an external
        // HF repo. The companion is not resolved through model_resolver, so we check
        // its pin cache entry and validate consent against it. If no pin is cached
        // (companion was never preflighted), we cannot fully validate — but the frontend
        // should have called the preflight before setting consent. Block launch on
        // missing pin when consent was provided, since that indicates a preflight was
        // done but the pin was lost.
        if let Some(ref spec_config) = self.speculative_config
            && let Some(ref companion_repo) = spec_config.companion_model_repo_id()
        {
            let cache = crate::hf::mtp_pin_cache::pin_cache();
            if let Some(pin) = cache.get(companion_repo) {
                if pin.trust_remote_code_required {
                    validate_trust_consent_simple(
                        companion_repo,
                        &pin.revision,
                        &self.trust_remote_code_consent,
                    )?;
                    trust_remote_code_enabled = true;
                }
            } else {
                // A missing pin means the companion was never authenticated to a concrete
                // revision. Do not rely on a frontend preflight having run: direct API clients
                // and lost cache state must fail closed as well.
                anyhow::bail!(
                    "Cannot validate MTP companion model {}: no cached pin for {}. \
                         Re-run the preflight before launching.",
                    companion_repo,
                    companion_repo
                );
            }
        }

        if trust_remote_code_enabled {
            env.push((OsString::from("HF_TRUST_REMOTE_CODE"), OsString::from("1")));
        }

        Ok(SupervisedLaunch {
            program: binary_path,
            args: os_args,
            env,
            cwd: None,
            port: self.port,
            redacted_summary: format!(
                "Rapid-MLX serve: {} on port {}",
                self.model.display_name, self.port
            ),
            warnings: Vec::new(),
        })
    }
}

/// Validate trust_remote_code consent matches "repo_id@revision" format and corresponds to the
/// resolved model's HF source. Blocks launch on missing consent, format error, or mismatch.
fn validate_trust_consent(
    model: &ResolvedRapidMlxLaunchModel,
    consent: &Option<String>,
) -> Result<()> {
    let consent_str = consent
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("This model requires trust_remote_code (custom Python code execution). Consent must be granted for this specific repo and revision before launching."))?;

    if consent_str.is_empty() {
        anyhow::bail!("trust_remote_code consent must not be empty");
    }

    let (consent_repo, consent_revision) = consent_str.rsplit_once('@').ok_or_else(|| {
        anyhow::anyhow!(
            "trust_remote_code consent must be in format repo_id@revision (e.g. org/model@main)"
        )
    })?;

    match &model.original_input {
        RapidMlxModelSource::HuggingFaceRepo { repo_id, revision } => {
            if consent_repo != repo_id {
                anyhow::bail!(
                    "trust_remote_code consent repo mismatch: expected {repo_id}, got {consent_repo}"
                );
            }
            if consent_revision != revision {
                anyhow::bail!(
                    "trust_remote_code consent revision mismatch for {repo_id}: expected {revision}, got {consent_revision}"
                );
            }
        }
        RapidMlxModelSource::MlxDirectory { .. }
        | RapidMlxModelSource::GgufFile { .. }
        | RapidMlxModelSource::Alias { .. }
        | RapidMlxModelSource::AuthoritativeSafetensors { .. }
        | RapidMlxModelSource::Unknown { .. } => {
            anyhow::bail!(
                "trust_remote_code consent requires an HF repo source; model source kind does not support revision-scoped consent"
            );
        }
    }

    Ok(())
}

/// Simplified trust_remote_code consent validation for the MTP companion model,
/// which is not resolved through model_resolver. Takes the expected repo id and
/// revision directly (from the pin cache) and validates consent against them.
fn validate_trust_consent_simple(
    expected_repo: &str,
    expected_revision: &str,
    consent: &Option<String>,
) -> Result<()> {
    let consent_str = consent
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("This MTP companion model requires trust_remote_code (custom Python code execution). Consent must be granted for this specific repo and revision before launching."))?;

    if consent_str.is_empty() {
        anyhow::bail!("trust_remote_code consent must not be empty");
    }

    let (consent_repo, consent_revision) = consent_str.rsplit_once('@').ok_or_else(|| {
        anyhow::anyhow!(
            "trust_remote_code consent must be in format repo_id@revision (e.g. org/model@main)"
        )
    })?;

    if consent_repo != expected_repo {
        anyhow::bail!(
            "trust_remote_code consent repo mismatch for companion model: expected {expected_repo}, got {consent_repo}"
        );
    }
    if consent_revision != expected_revision {
        anyhow::bail!(
            "trust_remote_code consent revision mismatch for companion model {expected_repo}: expected {expected_revision}, got {consent_revision}"
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init_test_pin_cache() {
        let dir = tempfile::tempdir().unwrap();
        crate::hf::mtp_pin_cache::init_pin_cache(dir.path());
    }

    #[test]
    fn a_config_from_json_still_turns_pflash_off() {
        // rapid-mlx 0.11.1 defaults --pflash to "always" for the verified Qwen3.5/Qwen3.6
        // aliases, and the 2026-07-24 benchmark verdict is that needle recall collapses
        // 0-40% above the 32768-token threshold there. The `off` default used to live only
        // in `RapidMlxConfig::default()`, so every preset read off disk and every API
        // request deserialized to None, emitted no flag, and inherited "always".
        let config: crate::inference::rapid_mlx::RapidMlxConfig =
            serde_json::from_str(r#"{"model_path": "/models/thing"}"#).unwrap();
        assert_eq!(config.pflash_policy.as_deref(), Some("off"));
    }

    #[test]
    fn turning_pflash_off_does_not_require_a_runtime_that_has_it() {
        // Now that every config carries "off", a runtime predating --pflash must still
        // launch: it has no PFlash to disable, so the request is already satisfied.
        let capabilities =
            ServeCapabilities::from_help("--host --port --served-model-name --reasoning");
        let launch = RapidMlxCommandBuilder::new(
            ResolvedRapidMlxLaunchModel::validated_alias("model").unwrap(),
        )
        .pflash_policy(Some("off".into()))
        .build("rapid-mlx".into(), &capabilities)
        .expect("an unsupported PFlash must not block a request to disable it");
        assert!(!args(&launch).iter().any(|a| a == "--pflash"));
    }

    #[test]
    fn an_explicit_pflash_mode_still_requires_the_flag() {
        let capabilities = ServeCapabilities::from_help("--host --port --served-model-name");
        let error = RapidMlxCommandBuilder::new(
            ResolvedRapidMlxLaunchModel::validated_alias("model").unwrap(),
        )
        .pflash_policy(Some("always".into()))
        .build("rapid-mlx".into(), &capabilities)
        .expect_err("asking to enable PFlash on a runtime without it must fail clearly");
        assert!(error.to_string().contains("--pflash"), "got: {error}");
    }

    fn args(launch: &SupervisedLaunch) -> Vec<String> {
        launch
            .args
            .iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn unsupported_throughput_flags_are_omitted_without_failing_the_build() {
        // None of --max-num-seqs, --max-concurrent-requests or --gpu-memory-utilization are
        // in verified_baseline(). These are exposed in the spawn wizard and the preset
        // editor, so a require() here turned an ordinary UI choice into a launch failure.
        let launch = RapidMlxCommandBuilder::new(
            ResolvedRapidMlxLaunchModel::validated_alias("model").unwrap(),
        )
        .port(9000)
        .max_num_seqs(Some(8))
        .max_concurrent_requests(Some(32))
        .gpu_memory_utilization(Some(0.85))
        .prefill_step_size(Some(512))
        .build("rapid-mlx".into(), &ServeCapabilities::verified_baseline())
        .expect("unsupported throughput tuning must not fail the build");

        for flag in [
            "--max-num-seqs",
            "--max-concurrent-requests",
            "--gpu-memory-utilization",
        ] {
            assert!(
                !launch.args.iter().any(|a| a == flag),
                "{flag} must be omitted on a runtime that does not support it"
            );
        }
        // An unrelated supported flag still reaches argv, so one unsupported choice cannot
        // suppress everything else.
        assert!(
            launch
                .args
                .windows(2)
                .any(|p| p == ["--prefill-step-size", "512"]),
            "a supported flag must survive alongside omitted ones"
        );
    }

    #[test]
    fn unsupported_speculative_is_omitted_without_failing_the_build() {
        let config = crate::inference::rapid_mlx::RapidMlxSpeculativeConfig {
            method: crate::inference::rapid_mlx::RapidMlxSpeculativeMethod::Mtp,
            model: None,
            num_speculative_tokens: 3,
            disable_auto_k: false,
        };
        let capabilities =
            ServeCapabilities::from_help("--host --port --reasoning --prefill-step-size");
        let launch = RapidMlxCommandBuilder::new(
            ResolvedRapidMlxLaunchModel::validated_alias("model").unwrap(),
        )
        .port(9000)
        .prefill_step_size(Some(512))
        .speculative_config(Some(config))
        .build("rapid-mlx".into(), &capabilities)
        .expect("an unsupported speculative config must not fail the build");

        assert!(!launch.args.iter().any(|arg| arg == "--speculative-config"));
        assert!(
            launch
                .args
                .windows(2)
                .any(|p| p == ["--prefill-step-size", "512"]),
            "unrelated flags must survive: {:?}",
            launch.args
        );
    }

    #[test]
    fn typed_mtp_config_serializes_exact_vllm_json_into_argv() {
        init_test_pin_cache();
        let config = crate::inference::rapid_mlx::RapidMlxSpeculativeConfig {
            method: crate::inference::rapid_mlx::RapidMlxSpeculativeMethod::Mtp,
            model: Some("/models/model-mtp".into()),
            num_speculative_tokens: 4,
            disable_auto_k: true,
        };
        assert_eq!(
            config.to_cli_json().unwrap(),
            r#"{"method":"mtp","model":"/models/model-mtp","num_speculative_tokens":4,"disable_auto_k":true}"#
        );
        let launch = RapidMlxCommandBuilder::new(
            ResolvedRapidMlxLaunchModel::validated_alias("model").unwrap(),
        )
        .speculative_config(Some(config))
        .build("rapid-mlx".into(), &ServeCapabilities::verified_baseline())
        .unwrap();
        assert!(launch.args.windows(2).any(|pair| {
            pair == [
                "--speculative-config",
                r#"{"method":"mtp","model":"/models/model-mtp","num_speculative_tokens":4,"disable_auto_k":true}"#,
            ]
        }));
    }

    #[test]
    fn external_hf_mtp_companion_fails_closed_until_runtime_supports_pinning() {
        let config = crate::inference::rapid_mlx::RapidMlxSpeculativeConfig {
            method: crate::inference::rapid_mlx::RapidMlxSpeculativeMethod::Mtp,
            model: Some("org/model-mtp".into()),
            num_speculative_tokens: 4,
            disable_auto_k: false,
        };

        let error = RapidMlxCommandBuilder::new(
            ResolvedRapidMlxLaunchModel::validated_alias("model").unwrap(),
        )
        .speculative_config(Some(config))
        .build("rapid-mlx".into(), &ServeCapabilities::verified_baseline())
        .unwrap_err();

        assert!(error.to_string().contains("revision-pinned"));
    }

    #[test]
    fn secure_defaults_omit_upstream_default_tuning_flags() {
        let launch = RapidMlxCommandBuilder::new(
            ResolvedRapidMlxLaunchModel::validated_alias("model").unwrap(),
        )
        .build("rapid-mlx".into(), &ServeCapabilities::verified_baseline())
        .unwrap();
        assert_eq!(
            args(&launch),
            [
                "serve",
                "model",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
                "--reasoning"
            ]
        );
        assert!(
            launch
                .env
                .iter()
                .any(|(name, value)| { name == "RAPID_MLX_TELEMETRY" && value == "0" })
        );
    }

    #[test]
    fn current_flag_names_and_secret_environment_are_used() {
        let launch = RapidMlxCommandBuilder::new(
            ResolvedRapidMlxLaunchModel::validated_alias("model").unwrap(),
        )
        .timeout(90)
        .api_key("do-not-log".into())
        .build("rapid-mlx".into(), &ServeCapabilities::verified_baseline())
        .unwrap();
        let args = args(&launch);
        assert!(args.windows(2).any(|pair| pair == ["--timeout", "90"]));
        assert!(!args.iter().any(|arg| arg == "--max-cache-blocks"));
        assert!(!args.iter().any(|arg| arg == "--request-timeout"));
        assert!(!args.iter().any(|arg| arg == "--max-blocks"));
        assert!(
            launch
                .env
                .iter()
                .any(|(name, value)| { name == "RAPID_MLX_API_KEY" && value == "do-not-log" })
        );
        assert!(!launch.redacted_summary.contains("do-not-log"));
    }

    #[test]
    fn auto_tool_choice_without_parser_asks_runtime_to_detect_it() {
        let build = |parser: Option<&str>, auto: bool| {
            RapidMlxCommandBuilder::new(
                ResolvedRapidMlxLaunchModel::validated_alias("model").unwrap(),
            )
            .auto_tool_choice(auto)
            .tool_call_parser(parser.map(String::from))
            .build("rapid-mlx".into(), &ServeCapabilities::verified_baseline())
            .unwrap()
        };
        let has = |l: &_, pair: [&str; 2]| args(l).windows(2).any(|w| w == pair);
        assert!(has(&build(None, true), ["--tool-call-parser", "auto"]));
        assert!(has(
            &build(Some("qwen3_coder_xml"), true),
            ["--tool-call-parser", "qwen3_coder_xml"]
        ));
        assert!(
            !args(&build(None, false))
                .iter()
                .any(|a| a == "--tool-call-parser")
        );
    }

    #[test]
    fn default_reasoning_effort_is_forwarded_validated_and_fails_closed() {
        let build = |caps: &ServeCapabilities, effort: Option<&str>| {
            RapidMlxCommandBuilder::new(
                ResolvedRapidMlxLaunchModel::validated_alias("model").unwrap(),
            )
            .default_reasoning_effort(effort.map(String::from))
            .build("rapid-mlx".into(), caps)
        };
        let supported = ServeCapabilities::from_help(
            "--host --port --log-level --served-model-name --timeout --enable-prefix-cache --disable-prefix-cache --cache-memory-mb --hybrid-cache-entries --kv-disk-checkpoint-interval --tool-call-parser --reasoning-parser --enable-auto-tool-choice --no-thinking --reasoning --force-hybrid --no-hybrid --prefill-step-size --pflash --speculative-config --default-reasoning-effort",
        );
        let baseline = ServeCapabilities::verified_baseline();
        let launch = build(&supported, Some("high")).unwrap();
        assert!(
            args(&launch)
                .windows(2)
                .any(|pair| pair == ["--default-reasoning-effort", "high"])
        );
        for unset in [None, Some("")] {
            let launch = build(&baseline, unset).unwrap();
            assert!(
                !args(&launch)
                    .iter()
                    .any(|a| a == "--default-reasoning-effort")
            );
        }
        assert!(build(&baseline, Some("high")).is_err());
        assert!(build(&supported, Some("extreme")).is_err());
    }

    #[test]
    fn context_length_is_forwarded_and_fails_closed_without_runtime_support() {
        let build = |caps: &ServeCapabilities, tokens: Option<u32>| {
            RapidMlxCommandBuilder::new(
                ResolvedRapidMlxLaunchModel::validated_alias("model").unwrap(),
            )
            .context_length(tokens)
            .build("rapid-mlx".into(), caps)
        };
        // Baseline flags plus the one under test.
        let supported = ServeCapabilities::from_help(
            "--host --port --log-level --served-model-name --timeout --enable-prefix-cache --disable-prefix-cache --cache-memory-mb --hybrid-cache-entries --kv-disk-checkpoint-interval --tool-call-parser --reasoning-parser --enable-auto-tool-choice --no-thinking --reasoning --force-hybrid --no-hybrid --prefill-step-size --pflash --speculative-config --context-length",
        );
        let launch = build(&supported, Some(200_000)).unwrap();
        assert!(
            args(&launch)
                .windows(2)
                .any(|pair| pair == ["--context-length", "200000"])
        );

        // Unset or zero never emits the flag, so older runtimes keep launching.
        let baseline = ServeCapabilities::verified_baseline();
        for unset in [None, Some(0)] {
            let launch = build(&baseline, unset).unwrap();
            assert!(!args(&launch).iter().any(|arg| arg == "--context-length"));
        }
        let error = build(&baseline, Some(200_000)).unwrap_err();
        assert!(
            error.to_string().contains("--context-length"),
            "got: {error}"
        );
    }

    #[test]
    fn retained_cache_uses_source_native_flags_and_disables_disk_checkpoints() {
        let launch = RapidMlxCommandBuilder::new(
            ResolvedRapidMlxLaunchModel::validated_alias("model").unwrap(),
        )
        .prefix_cache_enabled(Some(true))
        .retained_cache_mib(Some(8192))
        .hybrid_cache_entries(Some(16))
        .disk_checkpoint_interval(Some(0))
        .build("rapid-mlx".into(), &ServeCapabilities::verified_baseline())
        .unwrap();
        let args = args(&launch);
        assert!(args.iter().any(|arg| arg == "--enable-prefix-cache"));
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--cache-memory-mb", "8192"])
        );
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--hybrid-cache-entries", "16"])
        );
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--kv-disk-checkpoint-interval", "0"])
        );
        assert!(!args.iter().any(|arg| arg == "--max-cache-blocks"));
    }

    #[test]
    fn disabled_retained_cache_does_not_emit_a_zero_byte_cap() {
        let launch = RapidMlxCommandBuilder::new(
            ResolvedRapidMlxLaunchModel::validated_alias("model").unwrap(),
        )
        .prefix_cache_enabled(Some(false))
        .retained_cache_mib(Some(0))
        .build("rapid-mlx".into(), &ServeCapabilities::verified_baseline())
        .unwrap();
        let args = args(&launch);
        assert!(args.iter().any(|arg| arg == "--disable-prefix-cache"));
        assert!(!args.iter().any(|arg| arg == "--cache-memory-mb"));
    }

    #[test]
    fn explicitly_configured_unsupported_option_fails_closed() {
        let capabilities = ServeCapabilities::from_help("--host --port");
        let error = RapidMlxCommandBuilder::new(
            ResolvedRapidMlxLaunchModel::validated_alias("model").unwrap(),
        )
        .timeout(90)
        .build("rapid-mlx".into(), &capabilities)
        .unwrap_err();
        assert!(error.to_string().contains("--timeout"));
    }

    #[test]
    fn escape_hatch_flags_are_applied_correctly() {
        let flags = vec![
            ("force-hybrid".into(), serde_json::Value::Bool(true)),
            ("no-hybrid".into(), serde_json::Value::Bool(false)),
            ("pflash".into(), serde_json::Value::String("always".into())),
            (
                "pflash-threshold".into(),
                serde_json::Value::Number(serde_json::Number::from(128)),
            ),
            (
                "pflash-keep-ratio".into(),
                serde_json::Value::Number(serde_json::Number::from_f64(0.7).unwrap()),
            ),
        ];
        let launch = RapidMlxCommandBuilder::new(
            ResolvedRapidMlxLaunchModel::validated_alias("model").unwrap(),
        )
        .escape_hatch_flags(flags)
        .build(
            "rapid-mlx".into(),
            // The verified baseline does not advertise the newer pflash
            // sub-flags; a runtime that supports them probes them into its
            // capability set.
            &ServeCapabilities::from_help(
                "--host --port --log-level --served-model-name --timeout --enable-prefix-cache --disable-prefix-cache --cache-memory-mb --hybrid-cache-entries --kv-disk-checkpoint-interval --tool-call-parser --reasoning-parser --enable-auto-tool-choice --no-thinking --reasoning --force-hybrid --no-hybrid --prefill-step-size --pflash --pflash-threshold --pflash-keep-ratio --speculative-config",
            ),
        )
        .unwrap();
        let args = args(&launch);
        assert!(args.contains(&"--force-hybrid".to_string()));
        assert!(!args.contains(&"--no-hybrid".to_string()));
        assert!(args.windows(2).any(|p| p == ["--pflash", "always"]));
        assert!(args.windows(2).any(|p| p == ["--pflash-threshold", "128"]));
        assert!(args.windows(2).any(|p| p == ["--pflash-keep-ratio", "0.7"]));
    }

    #[test]
    fn trust_consent_blocks_without_consent() {
        let model = ResolvedRapidMlxLaunchModel {
            launch_argument: "org/model".into(),
            display_name: "org/model".into(),
            source_kind: crate::inference::rapid_mlx::model_resolver::ResolvedRapidMlxSourceKind::FreeFormAlias,
            original_input: RapidMlxModelSource::HuggingFaceRepo {
                repo_id: "org/model".into(),
                revision: "main".into(),
            },
            conversion: None,
            required_environment: Vec::new(),
            warnings: Vec::new(),
            remediation: Vec::new(),
            trust_remote_code_required: Some(true),
            environment: std::collections::BTreeMap::new(),
        };
        let launch = RapidMlxCommandBuilder::new(model)
            .build("rapid-mlx".into(), &ServeCapabilities::verified_baseline());
        let err = launch.unwrap_err().to_string();
        assert!(
            err.contains("trust_remote_code"),
            "expected trust error, got: {err}"
        );
    }

    #[test]
    fn trust_consent_accepts_valid_match() {
        let model = ResolvedRapidMlxLaunchModel {
            launch_argument: "org/model".into(),
            display_name: "org/model".into(),
            source_kind: crate::inference::rapid_mlx::model_resolver::ResolvedRapidMlxSourceKind::FreeFormAlias,
            original_input: RapidMlxModelSource::HuggingFaceRepo {
                repo_id: "org/model".into(),
                revision: "main".into(),
            },
            conversion: None,
            required_environment: Vec::new(),
            warnings: Vec::new(),
            remediation: Vec::new(),
            trust_remote_code_required: Some(true),
            environment: std::collections::BTreeMap::new(),
        };
        let launch = RapidMlxCommandBuilder::new(model)
            .trust_remote_code_consent(Some("org/model@main".into()))
            .build("rapid-mlx".into(), &ServeCapabilities::verified_baseline());
        assert!(launch.is_ok(), "unexpected error: {:?}", launch);
        let envs: Vec<_> = launch.unwrap().env;
        assert!(
            envs.iter()
                .any(|(n, v)| n == "HF_TRUST_REMOTE_CODE" && v == "1")
        );
    }

    #[test]
    fn trust_consent_rejects_repo_mismatch() {
        let model = ResolvedRapidMlxLaunchModel {
            launch_argument: "org/model".into(),
            display_name: "org/model".into(),
            source_kind: crate::inference::rapid_mlx::model_resolver::ResolvedRapidMlxSourceKind::FreeFormAlias,
            original_input: RapidMlxModelSource::HuggingFaceRepo {
                repo_id: "org/model".into(),
                revision: "main".into(),
            },
            conversion: None,
            required_environment: Vec::new(),
            warnings: Vec::new(),
            remediation: Vec::new(),
            trust_remote_code_required: Some(true),
            environment: std::collections::BTreeMap::new(),
        };
        let launch = RapidMlxCommandBuilder::new(model)
            .trust_remote_code_consent(Some("other/model@main".into()))
            .build("rapid-mlx".into(), &ServeCapabilities::verified_baseline());
        let err = launch.unwrap_err().to_string();
        assert!(
            err.contains("repo mismatch"),
            "expected repo mismatch, got: {err}"
        );
    }

    #[test]
    fn trust_consent_rejects_revision_mismatch() {
        let model = ResolvedRapidMlxLaunchModel {
            launch_argument: "org/model".into(),
            display_name: "org/model".into(),
            source_kind: crate::inference::rapid_mlx::model_resolver::ResolvedRapidMlxSourceKind::FreeFormAlias,
            original_input: RapidMlxModelSource::HuggingFaceRepo {
                repo_id: "org/model".into(),
                revision: "main".into(),
            },
            conversion: None,
            required_environment: Vec::new(),
            warnings: Vec::new(),
            remediation: Vec::new(),
            trust_remote_code_required: Some(true),
            environment: std::collections::BTreeMap::new(),
        };
        let launch = RapidMlxCommandBuilder::new(model)
            .trust_remote_code_consent(Some("org/model@bad-revision".into()))
            .build("rapid-mlx".into(), &ServeCapabilities::verified_baseline());
        let err = launch.unwrap_err().to_string();
        assert!(
            err.contains("revision mismatch"),
            "expected revision mismatch, got: {err}"
        );
    }

    #[test]
    fn trust_consent_rejects_invalid_format() {
        let model = ResolvedRapidMlxLaunchModel {
            launch_argument: "org/model".into(),
            display_name: "org/model".into(),
            source_kind: crate::inference::rapid_mlx::model_resolver::ResolvedRapidMlxSourceKind::FreeFormAlias,
            original_input: RapidMlxModelSource::HuggingFaceRepo {
                repo_id: "org/model".into(),
                revision: "main".into(),
            },
            conversion: None,
            required_environment: Vec::new(),
            warnings: Vec::new(),
            remediation: Vec::new(),
            trust_remote_code_required: Some(true),
            environment: std::collections::BTreeMap::new(),
        };
        let launch = RapidMlxCommandBuilder::new(model)
            .trust_remote_code_consent(Some("just-repo".into()))
            .build("rapid-mlx".into(), &ServeCapabilities::verified_baseline());
        let err = launch.unwrap_err().to_string();
        assert!(
            err.contains("repo_id@revision"),
            "expected format error, got: {err}"
        );
    }

    #[test]
    fn trust_consent_rejects_non_hf_source() {
        let model = ResolvedRapidMlxLaunchModel {
            launch_argument: "/local/path".into(),
            display_name: "/local/path".into(),
            source_kind: crate::inference::rapid_mlx::model_resolver::ResolvedRapidMlxSourceKind::FreeFormAlias,
            original_input: RapidMlxModelSource::MlxDirectory {
                path: PathBuf::from("/local/path"),
            },
            conversion: None,
            required_environment: Vec::new(),
            warnings: Vec::new(),
            remediation: Vec::new(),
            trust_remote_code_required: Some(true),
            environment: std::collections::BTreeMap::new(),
        };
        let launch = RapidMlxCommandBuilder::new(model)
            .trust_remote_code_consent(Some("org/model@main".into()))
            .build("rapid-mlx".into(), &ServeCapabilities::verified_baseline());
        let err = launch.unwrap_err().to_string();
        assert!(
            err.contains("revision-scoped consent"),
            "expected source error, got: {err}"
        );
    }

    #[test]
    fn phase7_config_produces_valid_argv() {
        use crate::inference::rapid_mlx::compatibility::ServeCapabilities;
        // Phase 7 runtime with all flags present
        let capabilities = ServeCapabilities::from_help(
            // Every flag here must exist in rapid-mlx's real `serve --help`; see
            // settings.rs::serve_flag_literals_exist_in_the_real_runtime. This string used to
            // declare a dozen invented flags, which made the assertions below confirm the
            // builder's mistakes instead of catching them.
            "--host --port --served-model-name --timeout --max-cache-blocks \
             --kv-cache-dtype --kv-cache-turboquant --max-num-seqs --max-concurrent-requests \
             --prefill-batch-size --completion-batch-size \
             --prefill-step-size --force-hybrid --no-hybrid \
             --reasoning --mllm --no-mllm --gpu-memory-utilization \
             --pflash --hybrid-cache-entries --kv-disk-checkpoint-interval \
             --response-cache-entries \
             --default-temperature --default-top-p --default-top-k --default-min-p \
             --default-repetition-penalty --default-presence-penalty \
             --default-frequency-penalty --max-tokens",
        );
        let launch = RapidMlxCommandBuilder::new(
            ResolvedRapidMlxLaunchModel::validated_alias("model").unwrap(),
        )
        .port(9000)
        .kv_cache_dtype(Some(KvCacheDtypeArg::Explicit("int8".into())))
        .turboquant_mode(Some("k8v4".into()))
        .pflash_policy(Some("always".into()))
        .hybrid_cache_entries(Some(512))
        .max_num_seqs(Some(8))
        .max_concurrent_requests(Some(32))
        .prefill_batch_size(Some(256))
        .completion_batch_size(Some(64))
        .prefill_step_size(Some(512))
        .reasoning_mode(Some("on".into()))
        .mllm_vision(Some("on".into()))
        .gpu_memory_utilization(Some(0.85))
        .sampling_mode(Some("explicit_client".into()))
        .sampling_defaults(
            Some(0.7),
            Some(0.9),
            Some(40),
            Some(0.05),
            Some(1.05),
            Some(0.0),
            Some(0.0),
            Some(32768),
        )
        .build("rapid-mlx".into(), &capabilities)
        .unwrap();
        let args = args(&launch);
        assert!(args.windows(2).any(|p| p == ["--kv-cache-dtype", "int8"]));
        assert!(
            args.windows(2)
                .any(|p| p == ["--kv-cache-turboquant", "k8v4"])
        );
        assert!(args.windows(2).any(|p| p == ["--pflash", "always"]));
        assert!(
            args.windows(2)
                .any(|p| p == ["--hybrid-cache-entries", "512"])
        );
        assert!(args.windows(2).any(|p| p == ["--max-num-seqs", "8"]));
        assert!(
            args.windows(2)
                .any(|p| p == ["--max-concurrent-requests", "32"])
        );
        assert!(
            args.windows(2)
                .any(|p| p == ["--prefill-batch-size", "256"])
        );
        assert!(
            args.windows(2)
                .any(|p| p == ["--completion-batch-size", "64"])
        );
        assert!(args.iter().any(|p| p == "--reasoning"));
        assert!(!args.windows(2).any(|p| p == ["--reasoning", "on"]));
        assert!(args.windows(2).any(|p| p == ["--prefill-step-size", "512"]));
        assert!(
            args.windows(2)
                .any(|p| p == ["--default-temperature", "0.7"])
        );
        assert!(args.windows(2).any(|p| p == ["--default-top-p", "0.9"]));
        assert!(args.windows(2).any(|p| p == ["--default-top-k", "40"]));
        assert!(args.windows(2).any(|p| p == ["--default-min-p", "0.05"]));
        assert!(args.windows(2).any(|p| p == ["--max-tokens", "32768"]));
        assert!(args.iter().any(|arg| arg == "--mllm"));
        assert!(!args.iter().any(|arg| arg == "--vision"));
        assert!(!args.iter().any(|arg| arg == "--embeddings"));
        assert!(
            args.windows(2)
                .any(|p| p == ["--gpu-memory-utilization", "0.85"])
        );
        assert!(!args.iter().any(|arg| arg == "--sampling-mode"));
    }

    #[test]
    fn phase7_auto_defaults_are_omitted() {
        let launch = RapidMlxCommandBuilder::new(
            ResolvedRapidMlxLaunchModel::validated_alias("model").unwrap(),
        )
        .kv_cache_dtype(Some(KvCacheDtypeArg::Auto))
        .turboquant_mode(Some("none".into()))
        .reasoning_mode(Some("on".into()))
        .mllm_vision(Some("auto".into()))
        .embeddings(Some("auto".into()))
        .sampling_mode(Some("auto".into()))
        .build("rapid-mlx".into(), &ServeCapabilities::verified_baseline())
        .unwrap();
        let args = args(&launch);
        assert!(!args.iter().any(|a| a.starts_with("--kv-cache-dtype")));
        assert!(!args.iter().any(|a| a.starts_with("--kv-cache-turboquant")));
        assert!(!args.iter().any(|a| a.starts_with("--mllm")));
        assert!(!args.iter().any(|a| a.starts_with("--no-mllm")));
        assert!(!args.iter().any(|a| a.starts_with("--embeddings")));
        assert!(!args.iter().any(|a| a.starts_with("--sampling-mode")));
    }

    #[test]
    fn legacy_kv_dtype_fails_before_invalid_argv_is_spawned() {
        let error = RapidMlxCommandBuilder::new(
            ResolvedRapidMlxLaunchModel::validated_alias("model").unwrap(),
        )
        .kv_cache_dtype(Some(KvCacheDtypeArg::Explicit("fp8".into())))
        .build("rapid-mlx".into(), &ServeCapabilities::verified_baseline())
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("legacy Rapid-MLX kv_cache_dtype")
        );
    }

    #[test]
    fn mllm_off_uses_the_real_text_only_flag() {
        let capabilities = ServeCapabilities::from_help("--host --port --no-mllm --reasoning");
        let launch = RapidMlxCommandBuilder::new(
            ResolvedRapidMlxLaunchModel::validated_alias("model").unwrap(),
        )
        .mllm_vision(Some("off".into()))
        .build("rapid-mlx".into(), &capabilities)
        .unwrap();
        let args = args(&launch);
        assert!(args.iter().any(|arg| arg == "--no-mllm"));
        assert!(!args.iter().any(|arg| arg == "--mllm"));
    }

    #[test]
    fn mllm_rejects_legacy_or_unknown_values() {
        let error = RapidMlxCommandBuilder::new(
            ResolvedRapidMlxLaunchModel::validated_alias("model").unwrap(),
        )
        .mllm_vision(Some("enabled".into()))
        .build("rapid-mlx".into(), &ServeCapabilities::verified_baseline())
        .unwrap_err();
        assert!(error.to_string().contains("auto, on, or off"));
    }

    #[test]
    fn phase7_unsupported_flag_fails_closed() {
        let capabilities = ServeCapabilities::from_help("--host --port --served-model-name");
        let error = RapidMlxCommandBuilder::new(
            ResolvedRapidMlxLaunchModel::validated_alias("model").unwrap(),
        )
        .turboquant_mode(Some("k8v4".into()))
        .build("rapid-mlx".into(), &capabilities)
        .unwrap_err();
        assert!(error.to_string().contains("--kv-cache-turboquant"));
    }

    #[test]
    fn reasoning_off_emits_no_thinking_flag() {
        let launch = RapidMlxCommandBuilder::new(
            ResolvedRapidMlxLaunchModel::validated_alias("model").unwrap(),
        )
        .reasoning_mode(Some("off".into()))
        .build("rapid-mlx".into(), &ServeCapabilities::verified_baseline())
        .unwrap();
        let args = args(&launch);
        assert!(args.iter().any(|arg| arg == "--no-thinking"));
        assert!(args.iter().any(|arg| arg == "--reasoning"));
    }

    #[test]
    fn reasoning_default_none_treated_as_on() {
        let launch = RapidMlxCommandBuilder::new(
            ResolvedRapidMlxLaunchModel::validated_alias("model").unwrap(),
        )
        .build("rapid-mlx".into(), &ServeCapabilities::verified_baseline())
        .unwrap();
        let args = args(&launch);
        assert!(args.iter().any(|arg| arg == "--reasoning"));
        assert!(!args.iter().any(|arg| arg == "--no-thinking"));
    }

    #[test]
    fn reasoning_auto_rejected() {
        let error = RapidMlxCommandBuilder::new(
            ResolvedRapidMlxLaunchModel::validated_alias("model").unwrap(),
        )
        .reasoning_mode(Some("auto".into()))
        .build("rapid-mlx".into(), &ServeCapabilities::verified_baseline())
        .unwrap_err();
        assert!(error.to_string().contains("on or off"));
    }

    #[test]
    fn explicit_no_thinking_is_orthogonal_to_reasoning_profile() {
        let launch = RapidMlxCommandBuilder::new(
            ResolvedRapidMlxLaunchModel::validated_alias("model").unwrap(),
        )
        .no_thinking(true)
        .reasoning_mode(Some("on".into()))
        .build("rapid-mlx".into(), &ServeCapabilities::verified_baseline())
        .unwrap();
        let args = args(&launch);
        assert!(args.iter().any(|arg| arg == "--reasoning"));
        assert!(args.iter().any(|arg| arg == "--no-thinking"));
    }
}
