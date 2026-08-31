use warp::Filter;

use super::common::{ApiCtx, check_api_token, unauthorized_api_token, with_app_config};
use crate::inference::llama_cpp::ServerConfig;
use crate::inference::rapid_mlx::RapidMlxConfig;
use crate::inference::rapid_mlx::capabilities::{CacheDiagnosticParams, ExtraState};
use crate::memory_availability::MemoryAvailabilityState;
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
                    let mut findings = collect_llama_findings(&state);
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

fn collect_llama_findings(state: &crate::state::AppState) -> Vec<DoctorFinding> {
    state
        .server_config
        .lock()
        .ok()
        .and_then(|guard| guard.clone())
        .map_or_else(Vec::new, |config| llama_config_findings(&config))
}

/// Cross-backend llama.cpp checks for the known tool-loop failure modes.
fn llama_config_findings(config: &ServerConfig) -> Vec<DoctorFinding> {
    let mut findings = Vec::new();
    let tool_enabled = config
        .tool_call_format
        .as_deref()
        .is_some_and(|value| !value.trim().is_empty());
    if !tool_enabled {
        return findings;
    }
    // Canonical K/V (schema v5): empty means llama-server default (f16).
    let k = config.ctk.trim();
    let v = config.ctv.trim();
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
        let findings = llama_config_findings(&config);
        assert!(
            findings
                .iter()
                .any(|finding| finding.finding_type == DoctorFindingType::LlamaCpp)
        );
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
