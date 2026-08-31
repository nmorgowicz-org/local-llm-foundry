use warp::Filter;

use super::common::{ApiCtx, check_api_token, unauthorized_api_token, with_app_config};
use crate::config::AppConfig;
use crate::inference::llama_cpp::ServerConfig;
use crate::inference::rapid_mlx::RapidMlxConfig;
use crate::inference::rapid_mlx::capabilities::{CacheDiagnosticParams, ExtraState};
use crate::memory_availability::MemoryAvailabilityState;
use crate::presets::validation::validate_main_kv_policy;
use crate::state::{DoctorFinding, DoctorFindingType, DoctorSeverity, FixAction};

pub fn routes(
    ctx: ApiCtx,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    let state = ctx.state;
    let config = ctx.config;
    warp::path!("api" / "doctor" / "findings")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and(with_app_config(config))
        .and_then(
            move |auth: Option<String>, cfg: std::sync::Arc<crate::config::AppConfig>| {
                let state = state.clone();
                async move {
                    if !check_api_token(&auth, &cfg) {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            unauthorized_api_token(),
                        ));
                    }
                    let mut findings = collect_llama_findings(&state, &cfg).await;
                    findings.extend(collect_cache_findings(&state).await);
                    findings.extend(collect_reclaim_findings().await);
                    Ok(Box::new(warp::reply::json(&serde_json::json!({
                        "ok": true,
                        "findings": findings,
                    }))))
                }
            },
        )
        .boxed()
}

async fn collect_llama_findings(
    state: &crate::state::AppState,
    cfg: &AppConfig,
) -> Vec<DoctorFinding> {
    let Some(config) = state
        .server_config
        .lock()
        .ok()
        .and_then(|guard| guard.clone())
    else {
        return Vec::new();
    };
    let snapshot = doctor_capability_snapshot(cfg).await;
    llama_config_findings(&config, snapshot.as_ref())
}

/// Fetch a fresh-or-cached llama.cpp capability snapshot for the K/V policy
/// check below, mirroring `web::api::vram::llama_kv_capability_snapshot`.
async fn doctor_capability_snapshot(
    config: &AppConfig,
) -> Option<crate::inference::llama_cpp_capabilities::CapabilitySnapshot> {
    if !config.llama_server_path.is_file() {
        return None;
    }
    let _ = crate::inference::llama_cpp_capabilities::generate_snapshot(&config.llama_server_path)
        .await;
    crate::inference::llama_cpp_capabilities::ExecutableIdentity::from_path(
        &config.llama_server_path,
    )
    .ok()
    .and_then(|identity| crate::inference::llama_cpp_capabilities::cached_snapshot(&identity))
}

/// Cross-backend llama.cpp checks: the canonical mixed-K/V hard-gate policy
/// (independent of tool calling), plus the known tool-loop failure modes
/// gated on `tool_enabled`.
fn llama_config_findings(
    config: &ServerConfig,
    snapshot: Option<&crate::inference::llama_cpp_capabilities::CapabilitySnapshot>,
) -> Vec<DoctorFinding> {
    let mut findings = Vec::new();
    // Canonical K/V (schema v5): empty means llama-server default (f16).
    let k = config.ctk.trim();
    let v = config.ctv.trim();

    if let Some(snapshot) = snapshot
        && let Some(issue) = validate_main_kv_policy(k, v, snapshot)
    {
        findings.push(DoctorFinding {
            finding_type: DoctorFindingType::LlamaCpp,
            severity: DoctorSeverity::Issue,
            message: issue.message,
            section: "llama.cpp cache".into(),
            fix: None,
        });
    }

    let tool_enabled = config
        .tool_call_format
        .as_deref()
        .is_some_and(|value| !value.trim().is_empty());
    if !tool_enabled {
        return findings;
    }
    let low_k = !k.is_empty() && !is_q8_or_better(k);
    let low_v = !v.is_empty() && !is_q8_or_better(v);
    if low_k || low_v {
        findings.push(DoctorFinding {
            finding_type: DoctorFindingType::LlamaCpp,
            severity: DoctorSeverity::Warning,
            message: "Tool calls are enabled with KV cache below q8_0; quantized KV can corrupt tool-call loops. Raise cache K/V to q8_0 or disable tool calling for this preset.".into(),
            section: "llama.cpp cache".into(),
            fix: None,
        });
    }
    if config
        .extra_args
        .split_whitespace()
        .any(|arg| arg == "--no-jinja")
        || config
            .chat_template_file
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
    {
        findings.push(DoctorFinding {
            finding_type: DoctorFindingType::LlamaCpp,
            severity: DoctorSeverity::Issue,
            message: "Tool-call format is configured but the active chat template/Jinja path is disabled or empty; restore the model template and keep Jinja enabled.".into(),
            section: "llama.cpp template".into(),
            fix: None,
        });
    }
    findings
}

fn is_q8_or_better(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "q8_0" | "q8_1" | "f16" | "bf16" | "f32"
    )
}

fn rapid_runtime_findings(config: &RapidMlxConfig) -> Vec<DoctorFinding> {
    let mut findings = Vec::new();
    if let Some(parser) = config.tool_call_parser.as_deref() {
        let malformed = parser.trim().is_empty()
            || parser.chars().any(char::is_whitespace)
            || parser.starts_with('-');
        if malformed {
            findings.push(DoctorFinding {
                finding_type: DoctorFindingType::Preset,
                severity: DoctorSeverity::Issue,
                message: format!("Invalid --tool-call-parser argument {parser:?}; use a parser name, not a flag or space-separated value."),
                section: "Rapid-MLX parser".into(),
                fix: Some(FixAction::AddToolCallParser),
            });
        }
    }
    findings
}

async fn collect_cache_findings(state: &crate::state::AppState) -> Vec<DoctorFinding> {
    let active_id = state.active_session_id.lock().unwrap().clone();
    if active_id.is_empty() {
        return Vec::new();
    }
    let rapid = {
        let sessions = state.sessions.lock().unwrap();
        let Some(session) = sessions.iter().find(|session| session.id == active_id) else {
            return Vec::new();
        };
        if session.preset_id.is_empty() {
            return Vec::new();
        }
        state
            .presets
            .lock()
            .unwrap()
            .iter()
            .find(|preset| preset.id == session.preset_id)
            .and_then(|preset| preset.rapid_mlx.clone())
    };
    let Some(config) = rapid else {
        return Vec::new();
    };
    let Ok((binary, source)) = crate::inference::rapid_mlx::discovery::Discovery::resolve_binary(
        config.executable_path.as_deref(),
        config.managed_runtime_path.as_deref(),
    )
    .await
    else {
        return rapid_runtime_findings(&config);
    };
    let Ok(snapshot) =
        crate::inference::rapid_mlx::capabilities::generate_snapshot(&binary, source).await
    else {
        return rapid_runtime_findings(&config);
    };
    let memory = crate::memory_availability::build_snapshot();
    let (enabled, budget, _) = config.cache_mode.resolve(
        config.prefix_cache_enabled,
        config.retained_cache_mib,
        config.hybrid_cache_entries,
    );
    let params = CacheDiagnosticParams {
        config_prefix_cache_enabled: enabled,
        config_prefix_cache_budget_bytes: budget.map(|mib| u64::from(mib) * 1024 * 1024),
        config_max_cache_blocks: None,
        snapshot: snapshot.clone(),
        configured_ceiling_bytes: memory.configured_ceiling_bytes,
        current_safe_availability_bytes: memory.current_safe_availability_bytes,
    };
    let mut findings = snapshot
        .compute_prefix_cache_findings(&params)
        .findings
        .into_iter()
        .map(|finding| DoctorFinding {
            finding_type: DoctorFindingType::Cache,
            severity: match finding.severity.as_str() {
                "error" => DoctorSeverity::Issue,
                _ => DoctorSeverity::Warning,
            },
            message: finding.message,
            section: "cache".into(),
            fix: None,
        })
        .collect::<Vec<_>>();
    findings.extend(rapid_runtime_findings(&config));
    if [
        &snapshot.installed_extras.guided,
        &snapshot.installed_extras.vision,
        &snapshot.installed_extras.embeddings,
    ]
    .iter()
    .any(|extra| matches!(extra, ExtraState::Broken(_)))
    {
        findings.push(DoctorFinding {
            finding_type: DoctorFindingType::Environment,
            severity: DoctorSeverity::Issue,
            message: "Rapid-MLX capability probe found a broken optional extra; the runtime is stale or incompatible with its managed dependency set. Repair the managed runtime before enabling optional features.".into(),
            section: "Rapid-MLX runtime".into(),
            fix: None,
        });
    }
    findings
}

async fn collect_reclaim_findings() -> Vec<DoctorFinding> {
    let snapshot = crate::memory_availability::build_snapshot();
    if !matches!(
        snapshot.state,
        MemoryAvailabilityState::Unsafe | MemoryAvailabilityState::AfterClosingApps
    ) {
        return Vec::new();
    }
    let guidance = crate::system::compute_reclaim_guidance(&snapshot);
    if guidance.available_actions.is_empty() {
        return Vec::new();
    }
    vec![DoctorFinding {
        finding_type: DoctorFindingType::Cache,
        severity: if snapshot.state == MemoryAvailabilityState::Unsafe {
            DoctorSeverity::Issue
        } else {
            DoctorSeverity::Warning
        },
        message: format!(
            "Memory pressure detected ({:?}). {}",
            snapshot.state, guidance.conservative_estimate
        ),
        section: "memory".into(),
        fix: Some(FixAction::ReclaimBackendAllocatorCache),
    }]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(
        with_blocks: bool,
    ) -> crate::inference::rapid_mlx::capabilities::CapabilitySnapshot {
        crate::inference::rapid_mlx::capabilities::CapabilitySnapshot {
            serve_flags: if with_blocks {
                vec!["--max-cache-blocks".into()]
            } else {
                Vec::new()
            },
            ..Default::default()
        }
    }

    #[test]
    fn llama_tool_cache_check_reports_below_q8() {
        let config = ServerConfig {
            tool_call_format: Some("json".into()),
            ctk: "q4_0".into(),
            ..Default::default()
        };
        let findings = llama_config_findings(&config, None);
        assert!(
            findings
                .iter()
                .any(|finding| finding.finding_type == DoctorFindingType::LlamaCpp)
        );
    }

    fn llama_capability_snapshot(
        mixed_main_kv_supported: bool,
    ) -> crate::inference::llama_cpp_capabilities::CapabilitySnapshot {
        use crate::inference::llama_cpp_capabilities::{
            CapabilitySnapshot, CapabilitySnapshotSource, ExecutableIdentity, MixedMainKv,
        };
        CapabilitySnapshot {
            executable_identity: ExecutableIdentity {
                path: "/tmp/llama-server".into(),
                file_hash: "abc123".into(),
                file_mtime_unix: 0,
            },
            version_text: String::new(),
            help_hash: String::new(),
            serve_flags: Vec::new(),
            cache: Default::default(),
            context: Default::default(),
            concurrency: Default::default(),
            endpoints: Default::default(),
            streaming: Default::default(),
            templates: Default::default(),
            tools: Default::default(),
            speculation: Default::default(),
            typed: Default::default(),
            mixed_main_kv: if mixed_main_kv_supported {
                MixedMainKv {
                    supported: true,
                    reason: "build manifest proves support".into(),
                    source: "build_manifest".into(),
                }
            } else {
                MixedMainKv::product_default_denied()
            },
            evidence_timestamp: 0,
            source: CapabilitySnapshotSource::AutoProbed,
        }
    }

    #[test]
    fn llama_mixed_kv_check_fires_independent_of_tool_enabled() {
        let config = ServerConfig {
            tool_call_format: None,
            ctk: "q8_0".into(),
            ctv: "q4_0".into(),
            ..Default::default()
        };
        let snapshot = llama_capability_snapshot(false);
        let findings = llama_config_findings(&config, Some(&snapshot));
        assert!(findings.iter().any(
            |finding| finding.finding_type == DoctorFindingType::LlamaCpp
                && finding.severity == DoctorSeverity::Issue
        ));
    }

    #[test]
    fn llama_mixed_kv_check_silent_when_binary_supports_pair() {
        let config = ServerConfig {
            tool_call_format: None,
            ctk: "q8_0".into(),
            ctv: "q4_0".into(),
            ..Default::default()
        };
        let snapshot = llama_capability_snapshot(true);
        let findings = llama_config_findings(&config, Some(&snapshot));
        assert!(findings.is_empty());
    }

    #[test]
    fn parser_check_rejects_flag_like_values() {
        let config = RapidMlxConfig {
            tool_call_parser: Some("--bad parser".into()),
            ..Default::default()
        };
        assert_eq!(rapid_runtime_findings(&config).len(), 1);
    }

    #[test]
    fn cache_doctor_reports_unsupported_block_setting() {
        let snap = snapshot(false);
        let params = CacheDiagnosticParams {
            config_prefix_cache_enabled: true,
            config_prefix_cache_budget_bytes: Some(4_000_000_000),
            config_max_cache_blocks: Some(32),
            snapshot: snap.clone(),
            configured_ceiling_bytes: 40_000_000_000,
            current_safe_availability_bytes: 30_000_000_000,
        };
        assert!(
            snap.compute_prefix_cache_findings(&params)
                .findings
                .iter()
                .any(|finding| finding.code == "CACHE_BLOCKS_UNSUPPORTED")
        );
    }

    #[test]
    fn cache_doctor_accepts_supported_block_setting() {
        let snap = snapshot(true);
        let params = CacheDiagnosticParams {
            config_prefix_cache_enabled: true,
            config_prefix_cache_budget_bytes: Some(4_000_000_000),
            config_max_cache_blocks: Some(32),
            snapshot: snap.clone(),
            configured_ceiling_bytes: 40_000_000_000,
            current_safe_availability_bytes: 30_000_000_000,
        };
        assert!(
            snap.compute_prefix_cache_findings(&params)
                .findings
                .is_empty()
        );
    }
}
