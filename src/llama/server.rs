use crate::config::AppConfig;
use crate::inference::backend::BackendAdapter;
use crate::inference::launch::{LocalLaunchRequest, launch_local};
use crate::inference::llama_cpp::ServerConfig;
use crate::inference::supervisor::Supervisor;
use crate::presets::ModelPreset;
use crate::state::AppState;
use anyhow::Result;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

const LLAMA_STARTUP_TIMEOUT: Duration = Duration::from_secs(300);

/// Inputs the launch-evidence sampler needs (architecture 12) that no
/// existing `start_backend` parameter already carries. Grouped into one
/// struct purely to keep `start_backend`'s argument count in check; it is
/// not a meaningful domain type on its own.
pub(crate) struct EvidenceContext {
    pub app_config: AppConfig,
    pub resolved_preset: Option<ModelPreset>,
}

pub async fn start_server(
    state: Arc<AppState>,
    config: ServerConfig,
    app_config: &AppConfig,
) -> Result<()> {
    launch_local(
        state,
        LocalLaunchRequest::LlamaCpp(Box::new(config)),
        app_config,
    )
    .await
}

pub(crate) async fn start_backend(
    state: Arc<AppState>,
    adapter: BackendAdapter,
    launch_request: LocalLaunchRequest,
    port: u16,
    model_identity: String,
    legacy_llama_config: Option<ServerConfig>,
    evidence: EvidenceContext,
) -> Result<()> {
    let lifecycle = state.server_lifecycle.lock().await;
    ensure_no_local_process(&state).await?;

    state.server_stopping.store(false, Ordering::Relaxed);
    {
        state.server_logs.lock().unwrap().clear();
    }
    state.push_log(format!(
        "[monitor] start_server: launching local inference backend on port={port} model={model_identity}"
    ));
    adapter.validate().await?;

    let launch = adapter.build_launch().await?;
    if let Ok(mut last_spawn_cmd) = state.last_spawn_cmd.lock() {
        *last_spawn_cmd = crate::inference::supervisor::redacted_spawn_command(&launch);
    }
    state.push_log(format!("[monitor] {}", launch.redacted_summary));
    for warning in &launch.warnings {
        state.push_log(format!("[monitor] warning: {warning}"));
    }
    let generation = state.server_generation.fetch_add(1, Ordering::AcqRel) + 1;
    let supervisor = Arc::new(Supervisor::new(
        launch,
        state.clone(),
        state.clone(),
        generation,
    ));

    // Register ownership before spawn so a very early process exit cannot race
    // with registration and leave stale running state behind.
    {
        *state.backend.lock().unwrap() = Some(adapter.clone());
        *state.local_launch_request.lock().unwrap() = Some(launch_request);
        *state.server_config.lock().unwrap() = legacy_llama_config;
        *state.local_server_running.lock().unwrap() = true;
        *state.server_running.lock().unwrap() = false;
        *state.supervisor.lock().await = Some(supervisor.clone());
    }

    // Sampled immediately before spawn, while this call still has exclusive
    // access to the pre-launch instant the Metal sampler needs as its
    // `before` baseline (architecture 12).
    let pre_launch_wired_bytes = crate::memory_availability::build_snapshot().wired_bytes;

    // Same pre-spawn timing slot, but for the Windows/WDDM sampler. Unlike
    // the in-process wired-memory read above, this drives real `nvidia-smi`
    // processes for idle stabilization, so `capture_before` gates on
    // qualification (Windows, fit pinned off, no extra_args) itself and
    // returns instantly with zero I/O on every other launch.
    let nvidia_pre_spawn = match (&adapter, &evidence.resolved_preset) {
        (BackendAdapter::LlamaCpp(_), Some(preset)) => {
            crate::calibration::launch_evidence::nvidia_sampler::capture_before(
                &evidence.app_config,
                preset,
            )
            .await
        }
        _ => None,
    };

    let pid = match supervisor.clone().start().await {
        Ok(pid) => pid,
        Err(error) => {
            clear_generation_if_current(&state, generation).await;
            return Err(error);
        }
    };
    state.push_log(format!(
        "[monitor] start_server: local inference process spawned (pid={pid}); waiting for readiness"
    ));
    // Readiness may take minutes. Release the transition lock so stop_server
    // can cancel this generation while it is loading.
    drop(lifecycle);

    let readiness = tokio::select! {
        result = adapter.await_ready(port, Instant::now() + LLAMA_STARTUP_TIMEOUT) => result,
        () = supervisor.wait_for_exit() => Err(anyhow::anyhow!(
            "Local inference process exited before becoming ready; check the captured server logs"
        )),
    };

    let _lifecycle = state.server_lifecycle.lock().await;
    let still_owner = owns_process(&state, generation, pid).await;
    if let Err(error) = readiness {
        if !still_owner {
            return Err(anyhow::anyhow!(
                "Local inference startup was cancelled or replaced before readiness completed"
            ));
        }
        state.push_log(format!(
            "[monitor] start_server: readiness failed for pid={pid}: {error}"
        ));
        state.server_stopping.store(true, Ordering::Relaxed);
        let stop_result = supervisor.stop().await;
        state.server_stopping.store(false, Ordering::Relaxed);
        if let Err(stop_error) = stop_result {
            state.push_log(format!(
                "[monitor] start_server: failed to clean up pid={pid} after readiness failure: {stop_error}"
            ));
            return Err(anyhow::anyhow!(
                "{error}; additionally failed to stop pid={pid}: {stop_error}"
            ));
        }
        clear_generation_if_current(&state, generation).await;
        return Err(error);
    }

    if !still_owner {
        anyhow::bail!("Local inference startup was cancelled or replaced before completion");
    }

    *state.server_running.lock().unwrap() = true;
    state.push_log(format!(
        "[monitor] start_server: local inference backend ready (pid={pid}, port={})",
        port
    ));
    state.llama_poll_notify.notify_waiters();

    // Fire-and-forget: sampling happens after this function has already
    // returned control to normal session flow, and its own gate (macOS,
    // fit pinned off, no extra_args) makes it a no-op on every other launch.
    if let (BackendAdapter::LlamaCpp(_), Some(preset)) = (&adapter, evidence.resolved_preset) {
        if let Some(capture) = nvidia_pre_spawn {
            crate::calibration::launch_evidence::nvidia_sampler::spawn(
                evidence.app_config.clone(),
                preset.clone(),
                pid,
                capture,
            );
        }
        crate::calibration::launch_evidence::metal_sampler::spawn(
            evidence.app_config,
            preset,
            pre_launch_wired_bytes,
        );
    }

    Ok(())
}

pub async fn stop_server(state: &AppState) -> Result<()> {
    let _lifecycle = state.server_lifecycle.lock().await;
    // Invalidate any readiness waiter before terminating its process. A stale
    // startup continuation must never clear a later generation.
    state.server_generation.fetch_add(1, Ordering::AcqRel);
    state.server_stopping.store(true, Ordering::Relaxed);

    let supervisor = state.supervisor.lock().await.take();
    let stop_result = if let Some(supervisor) = supervisor.as_ref() {
        if let Some(pid) = *state.server_child.lock().await {
            state.push_log(format!(
                "[monitor] stop_server: terminating supervised process pid={pid}"
            ));
        }
        supervisor.stop().await
    } else {
        match *state.server_child.lock().await {
            Some(pid) => Err(anyhow::anyhow!(
                "Cannot stop local inference server pid={pid}: process supervisor is unavailable"
            )),
            None => Ok(()),
        }
    };

    state.server_stopping.store(false, Ordering::Relaxed);
    if let Err(error) = stop_result {
        if let Some(supervisor) = supervisor {
            *state.supervisor.lock().await = Some(supervisor);
        }
        return Err(error);
    }

    clear_server_state_unchecked(state).await;
    *state.llama_metrics.lock().unwrap() = crate::llama::metrics::LlamaMetrics::default();
    state.push_log("[monitor] Server stopped.".into());
    Ok(())
}

async fn ensure_no_local_process(state: &AppState) -> Result<()> {
    let pid = *state.server_child.lock().await;
    let has_supervisor = state.supervisor.lock().await.is_some();
    if pid.is_some() || has_supervisor {
        anyhow::bail!(
            "A local inference server is already running{}; stop it before starting another model",
            pid.map(|pid| format!(" (pid={pid})")).unwrap_or_default()
        );
    }
    Ok(())
}

async fn owns_process(state: &AppState, generation: u64, pid: u32) -> bool {
    state.server_generation.load(Ordering::Acquire) == generation
        && *state.server_child.lock().await == Some(pid)
}

/// Clear startup-owned state only if its generation is still current.
/// Callers hold `server_lifecycle`, preventing a replacement during cleanup.
async fn clear_generation_if_current(state: &AppState, generation: u64) -> bool {
    if state.server_generation.load(Ordering::Acquire) != generation {
        return false;
    }
    clear_server_state_unchecked(state).await;
    true
}

async fn clear_server_state_unchecked(state: &AppState) {
    *state.server_child.lock().await = None;
    *state.server_running.lock().unwrap() = false;
    *state.local_server_running.lock().unwrap() = false;
    *state.server_config.lock().unwrap() = None;
    *state.backend.lock().unwrap() = None;
    *state.local_launch_request.lock().unwrap() = None;
    *state.supervisor.lock().await = None;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn overlapping_start_is_rejected_with_existing_pid() {
        let state = AppState::default();
        *state.server_child.lock().await = Some(4242);

        let error = ensure_no_local_process(&state).await.unwrap_err();

        assert!(error.to_string().contains("already running (pid=4242)"));
        assert!(error.to_string().contains("stop it before starting"));
    }

    #[tokio::test]
    async fn stale_startup_cleanup_does_not_clear_replacement_generation() {
        let state = AppState::default();
        state.server_generation.store(8, Ordering::Release);
        *state.server_child.lock().await = Some(8080);
        *state.server_running.lock().unwrap() = true;
        *state.local_server_running.lock().unwrap() = true;
        *state.server_config.lock().unwrap() = Some(ServerConfig::default());

        let cleared = clear_generation_if_current(&state, 7).await;

        assert!(!cleared);
        assert_eq!(*state.server_child.lock().await, Some(8080));
        assert!(*state.server_running.lock().unwrap());
        assert!(*state.local_server_running.lock().unwrap());
        assert!(state.server_config.lock().unwrap().is_some());
    }

    #[tokio::test]
    async fn process_ownership_requires_matching_generation_and_pid() {
        let state = AppState::default();
        state.server_generation.store(4, Ordering::Release);
        *state.server_child.lock().await = Some(4040);

        assert!(owns_process(&state, 4, 4040).await);
        assert!(!owns_process(&state, 3, 4040).await);
        assert!(!owns_process(&state, 4, 4041).await);
    }
}
