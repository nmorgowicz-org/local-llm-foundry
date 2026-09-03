use crate::config::AppConfig;
use crate::inference::InferenceBackend;
use crate::inference::backend::BackendAdapter;
use crate::inference::llama_cpp::{LlamaCppAdapter, ServerConfig};
use crate::inference::rapid_mlx::compatibility;
use crate::inference::rapid_mlx::discovery::Discovery;
use crate::inference::rapid_mlx::runtime::RuntimeMetadata;
use crate::inference::rapid_mlx::{RapidMlxAdapter, RapidMlxConfig};
use crate::presets::ModelPreset;
use crate::state::{AppState, Session, SessionMode};
use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::Arc;

/// Backend-owned local launch configurations. Adding another runtime extends
/// this enum without flattening its native controls into llama.cpp's config.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "backend", content = "config", rename_all = "snake_case")]
pub enum LocalLaunchRequest {
    LlamaCpp(Box<ServerConfig>),
    RapidMlx(Box<RapidMlxConfig>),
}

impl LocalLaunchRequest {
    pub fn backend(&self) -> InferenceBackend {
        match self {
            Self::LlamaCpp(_) => InferenceBackend::LlamaCpp,
            Self::RapidMlx(_) => InferenceBackend::RapidMlx,
        }
    }

    pub fn port(&self) -> u16 {
        match self {
            Self::LlamaCpp(config) => config.port,
            Self::RapidMlx(config) => config.port,
        }
    }

    pub fn bind_host(&self) -> Option<String> {
        match self {
            Self::LlamaCpp(config) => config.bind_host.clone(),
            Self::RapidMlx(config) => Some(config.host.clone()),
        }
    }

    pub fn api_key(&self) -> Option<String> {
        match self {
            Self::LlamaCpp(config) => config.api_key.clone(),
            Self::RapidMlx(config) => config.api_key.clone(),
        }
    }

    pub fn model_identity(&self) -> String {
        match self {
            Self::LlamaCpp(config) if !config.model_path.is_empty() => config.model_path.clone(),
            Self::LlamaCpp(config) => config.hf_repo.clone().unwrap_or_default(),
            Self::RapidMlx(config) => config.served_model_name.clone().unwrap_or_else(|| {
                config
                    .model_source
                    .as_ref()
                    .map(|source| source.display_name())
                    .unwrap_or_else(|| config.model_path.clone())
            }),
        }
    }

    /// Clone a launch request for session persistence without credentials.
    pub fn for_persistence(&self) -> Self {
        let mut persisted = self.clone();
        match &mut persisted {
            Self::LlamaCpp(config) => config.api_key = None,
            Self::RapidMlx(config) => config.api_key = None,
        }
        persisted
    }
}

#[derive(serde::Deserialize)]
struct BackendDiscriminator {
    #[serde(default)]
    backend: InferenceBackend,
}

pub fn request_from_api_payload(payload: &serde_json::Value) -> Result<LocalLaunchRequest> {
    let selected: BackendDiscriminator =
        serde_json::from_value(payload.clone()).context("Invalid inference backend selection")?;
    if selected.backend == InferenceBackend::LlamaCpp && payload.get("rapid_mlx").is_some() {
        anyhow::bail!(
            "llama_cpp launch must not include a backend-owned rapid_mlx configuration object"
        );
    }
    match selected.backend {
        InferenceBackend::LlamaCpp => serde_json::from_value(payload.clone())
            .map(|config| LocalLaunchRequest::LlamaCpp(Box::new(config)))
            .context("Invalid llama.cpp launch configuration"),
        InferenceBackend::RapidMlx => {
            let config = payload.get("rapid_mlx").ok_or_else(|| {
                anyhow::anyhow!(
                    "Rapid-MLX launch requires a backend-owned rapid_mlx configuration object"
                )
            })?;
            let config: RapidMlxConfig = serde_json::from_value(config.clone())
                .context("Invalid Rapid-MLX launch configuration")?;
            if config.port == 0 {
                anyhow::bail!("Rapid-MLX launch requires a non-zero port");
            }
            config.validate_access(None)?;
            config.validate_speculative_config()?;
            Ok(LocalLaunchRequest::RapidMlx(Box::new(config)))
        }
    }
}

pub fn validate_preset_backend_config(preset: &ModelPreset) -> Result<()> {
    if preset.bundle.is_some() && preset.backend != InferenceBackend::LlamaCpp {
        anyhow::bail!(
            "preset '{}' contains a llama.cpp bundle but is not a llama.cpp preset",
            preset.name
        );
    }
    match preset.backend {
        InferenceBackend::LlamaCpp if preset.rapid_mlx.is_some() => anyhow::bail!(
            "llama_cpp preset '{}' must not include rapid_mlx configuration",
            preset.name
        ),
        InferenceBackend::RapidMlx => {
            let rapid = preset.rapid_mlx.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "Rapid-MLX preset '{}' is missing its rapid_mlx configuration",
                    preset.name
                )
            })?;
            if rapid.api_key.is_some() {
                anyhow::bail!(
                    "Rapid-MLX preset '{}' must store its API key in the protected top-level api_key field, not rapid_mlx.api_key",
                    preset.name
                );
            }
            rapid.effective_model_source().with_context(|| {
                format!(
                    "Rapid-MLX preset '{}' requires a valid model source",
                    preset.name
                )
            })?;
            if rapid.port == 0 {
                anyhow::bail!(
                    "Rapid-MLX preset '{}' requires a non-zero port",
                    preset.name
                );
            }
            rapid.validate_access(preset.api_key.as_deref())?;
            rapid.validate_speculative_config()?;
            if let Err(invalid) = crate::inference::rapid_mlx::escape_hatch::validate_escape_flags(
                &rapid.escape_hatch_flags,
            ) {
                anyhow::bail!(
                    "Rapid-MLX preset '{}' contains non-allowlisted escape-hatch flags: {}",
                    preset.name,
                    invalid.join(", ")
                );
            }
            Ok(())
        }
        InferenceBackend::LlamaCpp => {
            let issues = crate::presets::validation::validate_llama_launch_policy(preset, None);
            if !issues.is_empty() {
                let codes: Vec<String> = issues
                    .iter()
                    .map(|i| format!("{} ({})", i.code, i.message))
                    .collect();
                anyhow::bail!(
                    "llama.cpp preset '{}' failed launch validation: {}",
                    preset.name,
                    codes.join("; ")
                );
            }
            let bundle_issues = crate::presets::bundle::validate_bundle_structural(preset);
            if !bundle_issues.is_empty() {
                anyhow::bail!(
                    "llama.cpp preset '{}' failed bundle validation: {}",
                    preset.name,
                    bundle_issues.join("; ")
                );
            }
            Ok(())
        }
    }
}

pub fn request_from_preset(
    preset: &ModelPreset,
    port_override: Option<u16>,
) -> Result<LocalLaunchRequest> {
    // Bundle defaults are an explicit compatibility projection. Keep the
    // persisted bundle authoritative while giving the existing launcher the
    // ordinary flat request it already understands.
    let mut preset = preset.clone();
    if preset.bundle.is_some() {
        crate::presets::bundle::materialize_default_projection(&mut preset);
    }
    validate_preset_backend_config(&preset)?;
    match preset.backend {
        InferenceBackend::RapidMlx => {
            let mut config = preset.rapid_mlx.clone().ok_or_else(|| {
                anyhow::anyhow!(
                    "Rapid-MLX preset '{}' is missing its rapid_mlx configuration",
                    preset.name
                )
            })?;
            config.api_key = preset.api_key.clone();
            config.default_temperature = preset.temperature;
            config.default_top_p = preset.top_p;
            config.default_top_k = preset.top_k.and_then(|value| u64::try_from(value).ok());
            config.default_min_p = preset.min_p;
            config.default_repetition_penalty = preset.repeat_penalty;
            config.default_presence_penalty = preset.presence_penalty;
            config.max_tokens = preset.max_tokens;
            if let Some(port) = port_override {
                config.port = port;
            }
            if config.port == 0 {
                anyhow::bail!("Rapid-MLX launch requires a non-zero port");
            }
            Ok(LocalLaunchRequest::RapidMlx(Box::new(config)))
        }
        InferenceBackend::LlamaCpp => Ok(LocalLaunchRequest::LlamaCpp(Box::new(ServerConfig {
            model_path: preset.model_path.clone(),
            hf_repo: preset.hf_repo.clone(),
            context_size: preset.context_size,
            ctk: preset.ctk.clone(),
            ctv: preset.ctv.clone(),
            tensor_split: preset.tensor_split.clone(),
            batch_size: preset.batch_size,
            ubatch_size: preset.ubatch_size,
            no_mmap: if cfg!(target_os = "macos") && preset.mlock {
                false
            } else {
                preset.no_mmap
            },
            load_mode: preset.load_mode.map(|mode| mode.with_mlock(preset.mlock)),
            verbosity: preset.verbosity.or(Some(4)),
            no_cont_batching: preset.no_cont_batching,
            swa_full: preset.swa_full,
            ctx_checkpoints: preset.ctx_checkpoints.or(Some(32)),
            checkpoint_min_step: preset.checkpoint_min_step,
            cache_reuse: preset.cache_reuse,
            port: port_override.or(preset.port).unwrap_or(8001),
            ngram_spec: preset.ngram_spec,
            parallel_slots: preset.parallel_slots,
            temperature: preset.temperature,
            top_p: preset.top_p,
            top_k: preset.top_k,
            min_p: preset.min_p,
            repeat_penalty: preset.repeat_penalty,
            repeat_last_n: preset.repeat_last_n,
            presence_penalty: preset.presence_penalty,
            n_cpu_moe: preset.n_cpu_moe,
            gpu_layers: preset.gpu_layers,
            mlock: preset.mlock,
            flash_attn: preset.flash_attn.clone(),
            split_mode: preset.split_mode.clone(),
            main_gpu: preset.main_gpu,
            threads: preset.threads,
            threads_batch: preset.threads_batch,
            rope_scaling: preset.rope_scaling.clone(),
            rope_freq_base: preset.rope_freq_base,
            rope_freq_scale: preset.rope_freq_scale,
            spec: crate::inference::llama_cpp::SpecDecodeConfig {
                draft_model: preset.draft_model.clone(),
                draft_min: preset.draft_min,
                draft_max: preset.draft_max,
                spec_ngram_size: preset.spec_ngram_size,
                spec_type: preset.spec_type.clone(),
                spec_default: preset.spec_default,
                spec_draft_n_max: preset.spec_draft_n_max,
                spec_draft_n_min: preset.spec_draft_n_min,
                spec_draft_p_split: preset.spec_draft_p_split,
                spec_draft_p_min: preset.spec_draft_p_min,
                spec_draft_ngl: preset.spec_draft_ngl,
                spec_draft_device: preset.spec_draft_device.clone(),
                spec_draft_cpu_moe: preset.spec_draft_cpu_moe,
                spec_draft_n_cpu_moe: preset.spec_draft_n_cpu_moe,
                spec_draft_type_k: preset.spec_draft_type_k.clone(),
                spec_draft_type_v: preset.spec_draft_type_v.clone(),
                spec_ngram_mod_n_min: preset.spec_ngram_mod_n_min,
                spec_ngram_mod_n_max: preset.spec_ngram_mod_n_max,
                spec_ngram_mod_n_match: preset.spec_ngram_mod_n_match,
                spec_ngram_simple_size_n: preset.spec_ngram_simple_size_n,
                spec_ngram_simple_size_m: preset.spec_ngram_simple_size_m,
                spec_ngram_simple_min_hits: preset.spec_ngram_simple_min_hits,
                spec_ngram_map_k_size_n: preset.spec_ngram_map_k_size_n,
                spec_ngram_map_k_size_m: preset.spec_ngram_map_k_size_m,
                spec_ngram_map_k_min_hits: preset.spec_ngram_map_k_min_hits,
                spec_ngram_map_k4v_size_n: preset.spec_ngram_map_k4v_size_n,
                spec_ngram_map_k4v_size_m: preset.spec_ngram_map_k4v_size_m,
                spec_ngram_map_k4v_min_hits: preset.spec_ngram_map_k4v_min_hits,
            },
            kv_unified: preset.kv_unified,
            cache_idle_slots: preset.cache_idle_slots,
            cache_ram_mib: preset.cache_ram_mib,
            fit_enabled: preset.fit_enabled,
            fit_ctx: preset.fit_ctx,
            fit_target: preset.fit_target.clone(),
            fit_print: preset.fit_print,
            seed: preset.seed,
            system_prompt_file: preset.system_prompt_file.clone(),
            extra_args: preset.extra_args.clone(),
            bind_host: preset.bind_host.clone(),
            chat_template_file: preset.chat_template_file.clone(),
            mmproj: preset.mmproj.clone(),
            grammar: preset.grammar.clone(),
            json_schema: preset.json_schema.clone(),
            cache_type_k: preset.cache_type_k.clone(),
            cache_type_v: preset.cache_type_v.clone(),
            max_tokens: preset.max_tokens,
            api_key: preset.api_key.clone(),
            alias: preset.alias.clone(),
            benchmark_mode: preset.benchmark_mode,
            enable_thinking: preset.enable_thinking,
            preserve_thinking: preset.preserve_thinking,
            tool_call_format: preset.tool_call_format.clone(),
            reasoning: preset.reasoning.clone(),
            reasoning_budget: preset.reasoning_budget,
            reasoning_budget_message: preset.reasoning_budget_message.clone(),
            mmproj_offload: preset.mmproj_offload,
            llama_reasoning_effort: preset.llama_reasoning_effort.clone(),
            llama_reasoning_format: preset.llama_reasoning_format.clone(),
            llama_reasoning_preserve: preset.llama_reasoning_preserve,
            bundle: preset.bundle.clone(),
            image_min_tokens: preset.image_min_tokens,
            image_max_tokens: preset.image_max_tokens,
            ..Default::default()
        }))),
    }
}

pub fn request_from_session(
    session: &Session,
    presets: &[ModelPreset],
    transient_api_key: Option<&str>,
) -> Result<LocalLaunchRequest> {
    let session_api_key = match &session.mode {
        SessionMode::Spawn { api_key, .. } => api_key.clone(),
        SessionMode::Attach { .. } => anyhow::bail!("Attach sessions cannot be locally launched"),
    };

    if let Some(mut request) = session.launch.clone() {
        if request.backend() != session.backend {
            anyhow::bail!("Persisted session backend does not match its launch envelope");
        }
        let restored_key = transient_api_key
            .filter(|key| !key.is_empty())
            .map(str::to_string)
            .or(session_api_key);
        match &mut request {
            LocalLaunchRequest::LlamaCpp(config) => config.api_key = restored_key,
            LocalLaunchRequest::RapidMlx(config) => config.api_key = restored_key,
        }
        if session.launch_requires_api_key && request.api_key().is_none() {
            anyhow::bail!(
                "This restored session requires its inference API key to be entered again"
            );
        }
        return Ok(request);
    }

    let preset = presets
        .iter()
        .find(|preset| preset.id == session.preset_id)
        .ok_or_else(|| {
            anyhow::anyhow!("Restored session has no launch envelope or matching preset")
        })?;
    let port = match &session.mode {
        SessionMode::Spawn { port, .. } => Some(*port),
        SessionMode::Attach { .. } => None,
    };
    let mut request = request_from_preset(preset, port)?;
    if request.backend() != session.backend {
        anyhow::bail!("Restored session backend no longer matches its preset backend");
    }
    if let Some(key) = transient_api_key.filter(|key| !key.is_empty()) {
        match &mut request {
            LocalLaunchRequest::LlamaCpp(config) => config.api_key = Some(key.to_string()),
            LocalLaunchRequest::RapidMlx(config) => config.api_key = Some(key.to_string()),
        }
    }
    if session.launch_requires_api_key && request.api_key().is_none() {
        anyhow::bail!("This restored session requires its inference API key to be entered again");
    }
    Ok(request)
}

/// The only production adapter construction point for local inference.
pub async fn construct_adapter(
    request: &LocalLaunchRequest,
    state: &AppState,
    app_config: &AppConfig,
) -> Result<BackendAdapter> {
    match request {
        LocalLaunchRequest::LlamaCpp(config) => {
            // Capability claims must be tied to the exact selected llama-server
            // binary rather than the build bundled during development. Snapshot
            // generation is bounded; validation below remains the launch gate.
            let snapshot = if app_config.llama_server_path.is_file() {
                let _ = crate::inference::llama_cpp_capabilities::generate_snapshot(
                    &app_config.llama_server_path,
                )
                .await;
                match crate::inference::llama_cpp_capabilities::ExecutableIdentity::from_path(
                    &app_config.llama_server_path,
                ) {
                    Ok(identity) => {
                        crate::inference::llama_cpp_capabilities::cached_snapshot(&identity)
                    }
                    Err(_) => None,
                }
            } else {
                None
            };
            // Launch gate: binary-specific policy (e.g. mixed main-K/V) is
            // enforced against the live snapshot before any process is spawned.
            // Structural checks already ran at request construction time.
            if let Some(snap) = &snapshot
                && let Some(issue) = crate::presets::validation::validate_main_kv_policy(
                    &config.ctk,
                    &config.ctv,
                    snap,
                )
            {
                anyhow::bail!(
                    "llama.cpp launch blocked: {} ({})",
                    issue.code,
                    issue.message
                );
            }
            if (config.ctk.eq_ignore_ascii_case("q8_0") && config.ctv.eq_ignore_ascii_case("q4_0"))
                && snapshot.is_none()
            {
                anyhow::bail!(
                    "llama.cpp launch blocked: capability probe unavailable for mixed main K/V"
                );
            }
            let gpu_env = state.gpu_env.lock().unwrap().clone();
            if let Some(bundle) = &config.bundle {
                match &snapshot {
                    Some(snapshot) => {
                        let issues = crate::presets::resolver::validate_runtime_selection(
                            bundle,
                            &bundle.default_selection,
                            snapshot,
                            cfg!(target_os = "macos"),
                        );
                        if !issues.is_empty() {
                            anyhow::bail!(
                                "llama.cpp bundle launch blocked by runtime validation: {}",
                                issues.join("; ")
                            );
                        }
                    }
                    None if bundle
                        .default_selection
                        .n_cpu_moe
                        .is_some_and(|value| value > 0) =>
                    {
                        anyhow::bail!(
                            "llama.cpp bundle launch blocked: capability probe unavailable for n_cpu_moe validation"
                        );
                    }
                    None => {}
                }
            }
            Ok(BackendAdapter::LlamaCpp(Arc::new(
                LlamaCppAdapter::new_with_capabilities(
                    app_config.clone(),
                    config.as_ref().clone(),
                    gpu_env,
                    snapshot,
                ),
            )))
        }
        LocalLaunchRequest::RapidMlx(config) => {
            crate::inference::rapid_mlx::ensure_local_platform_supported()?;
            let model_source = config.effective_model_source()?;
            config.validate_access(None)?;

            let (executable_path, source) = Discovery::resolve_binary(
                config.executable_path.as_deref(),
                config.managed_runtime_path.as_deref(),
            )
            .await
            .with_context(|| {
                "Rapid-MLX runtime was not found. Install rapid-mlx or configure an executable path"
            })?;
            let profile = compatibility::probe(&executable_path, source)
                .await
                .with_context(|| {
                    format!(
                        "Rapid-MLX runtime at {} is incompatible",
                        executable_path.display()
                    )
                })?;
            let version = profile.version.clone();
            let capability_snapshot =
                match crate::inference::rapid_mlx::capabilities::ExecutableIdentity::from_path(
                    &executable_path,
                ) {
                    Ok(identity) => {
                        crate::inference::rapid_mlx::capabilities::cached_snapshot(&identity)
                    }
                    Err(_) => None,
                };
            let managed_evidence =
                if source == crate::inference::rapid_mlx::runtime::RuntimeSource::Managed {
                    crate::inference::rapid_mlx::updater::RapidMlxRuntimeManager::new(
                        &app_config.config_dir,
                    )
                    .ok()
                    .and_then(|manager| manager.status().ok())
                    .and_then(|status| status.active)
                    .filter(|entry| entry.executable_path == executable_path)
                } else {
                    None
                };
            let runtime = RuntimeMetadata {
                executable_path,
                source,
                version,
                capability_snapshot,
                resolved_receipt: managed_evidence
                    .as_ref()
                    .and_then(|entry| entry.resolved_receipt.clone()),
                last_probe_result: managed_evidence
                    .as_ref()
                    .and_then(|entry| entry.last_probe_result.clone()),
                prefix_cache_enabled: config.prefix_cache_enabled,
                // mlx_prefix_cache_bytes populated from estimator breakdown, not from config.
                mlx_prefix_cache_bytes: None,
            };
            let models_dir = state
                .models_dir
                .clone()
                .or_else(|| {
                    let configured = state.ui_settings.lock().unwrap().models_dir.clone();
                    (!configured.trim().is_empty()).then(|| PathBuf::from(configured))
                })
                .or_else(|| app_config.models_dir.clone())
                .unwrap_or_else(|| app_config.default_models_dir.clone());
            std::fs::create_dir_all(&models_dir)?;
            let python_executable = runtime
                .executable_path
                .parent()
                .map(|parent| {
                    parent.join(if cfg!(windows) {
                        "python.exe"
                    } else {
                        "python3"
                    })
                })
                .filter(|path| path.is_file())
                .unwrap_or_else(|| {
                    PathBuf::from(if cfg!(windows) {
                        "python.exe"
                    } else {
                        "python3"
                    })
                });
            let resolved_model = crate::inference::rapid_mlx::model_resolver::resolve(
                model_source,
                &crate::inference::rapid_mlx::model_resolver::RapidMlxResolveContext {
                    models_dir,
                    python_executable,
                    runtime_version: runtime.version.clone(),
                    hf_token: crate::hf::hf_load_token(),
                    verified_aliases: Vec::new(),
                    execute_conversion: true,
                },
            )
            .await?;
            let resolved_display_name = resolved_model.display_name.clone();
            if let Err(invalid) = crate::inference::rapid_mlx::escape_hatch::validate_escape_flags(
                &config.escape_hatch_flags,
            ) {
                anyhow::bail!(
                    "Rapid-MLX preset contains non-allowlisted escape-hatch flags: {}",
                    invalid.join(", ")
                );
            }
            let mut adapter = RapidMlxAdapter::from_resolved(runtime, resolved_model);
            adapter.apply_config(config);
            // Only the launch path has a resolved model to fall back on, so the name default
            // stays here rather than in `apply_config`.
            if adapter.served_model_name.is_none() {
                adapter.served_model_name = Some(resolved_display_name);
            }
            adapter.configure_runtime(profile, config.api_key.clone());
            Ok(BackendAdapter::RapidMlx(Arc::new(adapter)))
        }
    }
}

pub async fn launch_local(
    state: Arc<AppState>,
    request: LocalLaunchRequest,
    app_config: &AppConfig,
) -> Result<()> {
    launch_local_with_resolved_preset(state, request, app_config, None).await
}

/// Same as `launch_local`, but also carries the fully resolved `ModelPreset`
/// behind this launch when the caller has one (bundle/preset launches do; a
/// direct `ServerConfig` payload or a restored session without a matching
/// preset does not). Only the launch-evidence sampler (architecture 12)
/// consumes it; every other code path behaves identically either way.
pub async fn launch_local_with_resolved_preset(
    state: Arc<AppState>,
    request: LocalLaunchRequest,
    app_config: &AppConfig,
    resolved_preset: Option<ModelPreset>,
) -> Result<()> {
    let adapter = construct_adapter(&request, &state, app_config).await?;
    let legacy_llama_config = match &request {
        LocalLaunchRequest::LlamaCpp(config) => Some(config.as_ref().clone()),
        LocalLaunchRequest::RapidMlx(_) => None,
    };
    let port = request.port();
    let model_identity = request.model_identity();
    crate::llama::server::start_backend(
        state,
        adapter,
        request,
        port,
        model_identity,
        legacy_llama_config,
        crate::llama::server::EvidenceContext {
            app_config: app_config.clone(),
            resolved_preset,
        },
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn test_app_config(config_dir: &std::path::Path) -> AppConfig {
        AppConfig::from_args(crate::cli::AppArgs::parse_from([
            "llama-monitor",
            "--config-dir",
            config_dir.to_str().unwrap(),
            "--llama-server-path",
            "llama-server",
            "--gpu-backend",
            "none",
        ]))
    }

    #[test]
    fn direct_payload_without_backend_routes_to_llama_cpp() {
        let mut value = serde_json::to_value(ServerConfig::default()).unwrap();
        value["model_path"] = serde_json::json!("/models/legacy.gguf");
        let request = request_from_api_payload(&value).unwrap();
        assert!(matches!(request, LocalLaunchRequest::LlamaCpp(_)));
    }

    #[test]
    fn rapid_payload_requires_backend_owned_config() {
        let error = request_from_api_payload(&serde_json::json!({
            "backend": "rapid_mlx",
            "model_path": "/models/wrong-level"
        }))
        .unwrap_err();
        assert!(error.to_string().contains("rapid_mlx configuration object"));
    }

    #[test]
    fn direct_llama_payload_rejects_rapid_mlx_config() {
        for payload in [
            serde_json::json!({
                "model_path": "/models/legacy.gguf",
                "rapid_mlx": { "model_path": "/models/rapid" }
            }),
            serde_json::json!({
                "backend": "llama_cpp",
                "model_path": "/models/legacy.gguf",
                "rapid_mlx": { "model_path": "/models/rapid" }
            }),
        ] {
            let error = request_from_api_payload(&payload).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("llama_cpp launch must not include")
            );
        }
    }

    fn valid_rapid_mlx_preset() -> ModelPreset {
        ModelPreset {
            name: "Rapid".into(),
            backend: InferenceBackend::RapidMlx,
            rapid_mlx: Some(crate::inference::rapid_mlx::RapidMlxConfig {
                model_path: "/models/rapid".into(),
                port: 8123,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    /// Fixture 9 (Phase 10a): a Rapid-MLX preset proving no llama field
    /// leakage — a valid Rapid-MLX preset (llama-only fields left at their
    /// defaults) passes, while carrying a llama.cpp bundle or an api_key
    /// nested under rapid_mlx (instead of the shared top-level field) is
    /// rejected by the same gate llama_cpp presets are held to in reverse.
    #[test]
    fn rapid_mlx_preset_with_no_llama_fields_passes() {
        let preset = valid_rapid_mlx_preset();
        assert!(preset.bundle.is_none());
        validate_preset_backend_config(&preset).unwrap();
    }

    #[test]
    fn rapid_mlx_preset_rejects_a_llama_bundle() {
        let mut preset = valid_rapid_mlx_preset();
        preset.bundle = Some(crate::presets::bundle::PresetBundleSpec::default());
        let error = validate_preset_backend_config(&preset).unwrap_err();
        assert!(error.to_string().contains("not a llama.cpp preset"));
    }

    #[tokio::test]
    async fn factory_selects_llama_cpp_adapter() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_app_config(dir.path());
        let state = AppState::default();
        let request = LocalLaunchRequest::LlamaCpp(Box::default());

        let adapter = construct_adapter(&request, &state, &config).await.unwrap();
        assert!(matches!(adapter, BackendAdapter::LlamaCpp(_)));
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[tokio::test]
    async fn factory_selects_rapid_mlx_adapter() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let runtime = dir.path().join("rapid-mlx");
        std::fs::write(
            &runtime,
            "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo 'rapid-mlx 0.11.0+git.fixture'; else echo '--host --port --log-level --served-model-name --timeout --enable-prefix-cache --disable-prefix-cache --cache-memory-mb --hybrid-cache-entries --kv-disk-checkpoint-interval'; fi\n",
        )
        .unwrap();
        std::fs::set_permissions(&runtime, std::fs::Permissions::from_mode(0o755)).unwrap();
        let model = dir.path().join("mlx-model");
        std::fs::create_dir(&model).unwrap();
        std::fs::write(model.join("config.json"), r#"{"model_type":"qwen3"}"#).unwrap();
        std::fs::write(model.join("tokenizer.json"), "{}").unwrap();
        std::fs::write(model.join("model.safetensors"), "weights").unwrap();
        let config = test_app_config(dir.path());
        let state = AppState::default();
        let request = LocalLaunchRequest::RapidMlx(Box::new(RapidMlxConfig {
            model_path: model.to_string_lossy().into_owned(),
            executable_path: Some(runtime),
            ..Default::default()
        }));

        let adapter = construct_adapter(&request, &state, &config).await.unwrap();
        let BackendAdapter::RapidMlx(adapter) = adapter else {
            panic!("expected Rapid-MLX adapter");
        };
        assert_eq!(adapter.served_model_name.as_deref(), Some("mlx-model"));
    }

    #[test]
    fn rapid_preset_port_uses_nested_value_unless_explicitly_overridden() {
        let preset = ModelPreset {
            name: "Rapid".into(),
            backend: InferenceBackend::RapidMlx,
            port: Some(8001),
            rapid_mlx: Some(RapidMlxConfig {
                model_path: "/models/rapid".into(),
                port: 9123,
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(request_from_preset(&preset, None).unwrap().port(), 9123);
        assert_eq!(
            request_from_preset(&preset, Some(9222)).unwrap().port(),
            9222
        );
        assert!(request_from_preset(&preset, Some(0)).is_err());
    }

    #[test]
    fn rapid_preset_hydrates_protected_top_level_api_key_transiently() {
        let preset = ModelPreset {
            name: "Protected Rapid".into(),
            backend: InferenceBackend::RapidMlx,
            api_key: Some("preset-secret".into()),
            rapid_mlx: Some(RapidMlxConfig {
                model_path: "/models/rapid".into(),
                ..Default::default()
            }),
            ..Default::default()
        };
        let request = request_from_preset(&preset, None).unwrap();
        assert_eq!(request.api_key().as_deref(), Some("preset-secret"));
        assert!(
            !serde_json::to_string(&request)
                .unwrap()
                .contains("preset-secret")
        );
    }

    #[test]
    fn rapid_preset_rejects_nested_api_key() {
        let preset = ModelPreset {
            name: "Invalid protected Rapid".into(),
            backend: InferenceBackend::RapidMlx,
            rapid_mlx: Some(RapidMlxConfig {
                model_path: "/models/rapid".into(),
                api_key: Some("wrong-secret-location".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let error = request_from_preset(&preset, None).unwrap_err();
        assert!(error.to_string().contains("protected top-level api_key"));
        assert!(
            !serde_json::to_string(&preset)
                .unwrap()
                .contains("wrong-secret-location")
        );
    }

    #[test]
    fn direct_rapid_payload_rejects_zero_port() {
        let error = request_from_api_payload(&serde_json::json!({
            "backend": "rapid_mlx",
            "rapid_mlx": {
                "model_path": "/models/rapid",
                "port": 0
            }
        }))
        .unwrap_err();
        assert!(error.to_string().contains("non-zero port"));
    }

    #[test]
    fn rapid_lan_bind_requires_authenticated_access() {
        let error = request_from_api_payload(&serde_json::json!({
            "backend": "rapid_mlx",
            "rapid_mlx": {
                "model_path": "/models/rapid",
                "host": "0.0.0.0"
            }
        }))
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("LAN exposure requires an API key")
        );

        assert!(
            request_from_api_payload(&serde_json::json!({
                "backend": "rapid_mlx",
                "rapid_mlx": {
                    "model_path": "/models/rapid",
                    "host": "0.0.0.0",
                    "api_key": "protected"
                }
            }))
            .is_ok()
        );
    }

    #[test]
    fn preset_backend_config_mismatches_are_rejected() {
        let llama_with_rapid = ModelPreset {
            name: "Mismatch".into(),
            rapid_mlx: Some(RapidMlxConfig {
                model_path: "/models/rapid".into(),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(validate_preset_backend_config(&llama_with_rapid).is_err());

        let rapid_without_config = ModelPreset {
            name: "Missing".into(),
            backend: InferenceBackend::RapidMlx,
            ..Default::default()
        };
        assert!(validate_preset_backend_config(&rapid_without_config).is_err());
    }

    #[test]
    fn persisted_launch_envelope_scrubs_api_key_and_restores_backend() {
        let llama = LocalLaunchRequest::LlamaCpp(Box::new(ServerConfig {
            model_path: "/models/private.gguf".into(),
            port: 8001,
            api_key: Some("do-not-persist".into()),
            ..Default::default()
        }));
        let rapid = LocalLaunchRequest::RapidMlx(Box::new(RapidMlxConfig {
            model_path: "/models/private-mlx".into(),
            api_key: Some("also-do-not-persist".into()),
            ..Default::default()
        }));
        for request in [llama, rapid] {
            let persisted = request.for_persistence();
            let json = serde_json::to_string(&persisted).unwrap();
            assert!(!json.contains("do-not-persist"));
            assert!(persisted.api_key().is_none());
        }
    }

    #[test]
    fn restored_protected_session_uses_only_transient_api_key() {
        let request = LocalLaunchRequest::LlamaCpp(Box::new(ServerConfig {
            model_path: "/models/private.gguf".into(),
            port: 8001,
            api_key: Some("original-secret".into()),
            ..Default::default()
        }));
        let mut session = Session::new_spawn_with_backend(
            "private".into(),
            "Private".into(),
            8001,
            String::new(),
            None,
            Some("original-secret".into()),
            InferenceBackend::LlamaCpp,
            Some("/models/private.gguf".into()),
        );
        session.launch = Some(request.for_persistence());

        let serialized = serde_json::to_string(&session).unwrap();
        assert!(!serialized.contains("original-secret"));
        let restored_session: Session = serde_json::from_str(&serialized).unwrap();
        assert!(request_from_session(&restored_session, &[], None).is_err());

        let restored =
            request_from_session(&restored_session, &[], Some("entered-on-restore")).unwrap();
        assert_eq!(restored.api_key().as_deref(), Some("entered-on-restore"));
        assert!(
            !serde_json::to_string(&session)
                .unwrap()
                .contains("entered-on-restore")
        );
    }

    #[test]
    fn restored_direct_rapid_session_routes_from_launch_envelope() {
        let request = LocalLaunchRequest::RapidMlx(Box::new(RapidMlxConfig {
            model_path: "/models/rapid".into(),
            port: 8123,
            api_key: Some("rapid-original-secret".into()),
            ..Default::default()
        }));
        let mut session = Session::new_spawn_with_backend(
            "s1".into(),
            "Rapid".into(),
            8123,
            String::new(),
            Some("0.0.0.0".into()),
            Some("rapid-original-secret".into()),
            InferenceBackend::RapidMlx,
            Some("/models/rapid".into()),
        );
        session.launch = Some(request.for_persistence());

        let json = serde_json::to_string(&session).unwrap();
        assert!(!json.contains("rapid-original-secret"));
        let session: Session = serde_json::from_str(&json).unwrap();
        assert!(request_from_session(&session, &[], None).is_err());
        let restored = request_from_session(&session, &[], Some("rapid-transient-key")).unwrap();
        assert!(matches!(restored, LocalLaunchRequest::RapidMlx(_)));
        assert_eq!(restored.port(), 8123);
        assert_eq!(restored.api_key().as_deref(), Some("rapid-transient-key"));
        assert!(
            !serde_json::to_string(&session)
                .unwrap()
                .contains("rapid-transient-key")
        );
    }

    #[test]
    fn preset_with_non_allowlisted_escape_hatch_flag_fails_validation() {
        let preset = ModelPreset {
            name: "BadFlags".into(),
            backend: InferenceBackend::RapidMlx,
            rapid_mlx: Some(RapidMlxConfig {
                model_path: "/models/rapid".into(),
                escape_hatch_flags: vec![
                    ("pflash".into(), serde_json::Value::String("auto".into())),
                    ("bad-unknown-flag".into(), serde_json::Value::Bool(true)),
                ],
                ..Default::default()
            }),
            ..Default::default()
        };
        let err = validate_preset_backend_config(&preset).unwrap_err();
        assert!(err.to_string().contains("bad-unknown-flag"));
    }

    #[test]
    fn preset_with_valid_escape_hatch_flags_passes_validation() {
        let preset = ModelPreset {
            name: "GoodFlags".into(),
            backend: InferenceBackend::RapidMlx,
            rapid_mlx: Some(RapidMlxConfig {
                model_path: "/models/rapid".into(),
                escape_hatch_flags: vec![
                    ("force-hybrid".into(), serde_json::Value::Bool(true)),
                    (
                        "pflash-threshold".into(),
                        serde_json::Value::Number(serde_json::Number::from(64)),
                    ),
                ],
                ..Default::default()
            }),
            ..Default::default()
        };
        if let Err(e) = validate_preset_backend_config(&preset) {
            panic!("validation failed: {}", e);
        }
    }
}
