//! Production [`PromoteHost`]: wires the promotion engine to the app's model
//! lifecycle and the shared verifier.

use std::path::Path;
use std::sync::Arc;

use crate::config::AppConfig;
use crate::inference::llama_cpp::ServerConfig;
use crate::llama::server::{start_server, stop_server};
use crate::llama_runtime::gate::RUNTIME_OWNER;
use crate::llama_runtime::promote::{PromoteHost, VerifyStage};
use crate::llama_runtime::verify::{VerificationIntent, VerificationReport, verify_runtime};
use crate::state::AppState;

/// The local llama.cpp launch config to restart after the swap, when a
/// llama.cpp model is running. Rapid-MLX and attached sessions are not ours to
/// stop.
pub fn restart_config(state: &AppState) -> Option<ServerConfig> {
    let local_running = *state.local_server_running.lock().unwrap();
    let llama_cpp_active = matches!(
        state.local_launch_request.lock().unwrap().as_ref(),
        Some(crate::inference::launch::LocalLaunchRequest::LlamaCpp(_))
    );
    if local_running && llama_cpp_active {
        state.server_config.lock().unwrap().clone()
    } else {
        None
    }
}

pub struct LiveHost {
    state: AppState,
    config: Arc<AppConfig>,
    restart: Option<ServerConfig>,
    cancel: Arc<std::sync::atomic::AtomicBool>,
}

impl LiveHost {
    pub fn new(state: AppState, config: Arc<AppConfig>) -> Self {
        let restart = restart_config(&state);
        Self {
            state,
            config,
            restart,
            cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// A host that never stops or restarts a model: for custom server paths,
    /// which are not ours to manage.
    pub fn without_model(state: AppState, config: Arc<AppConfig>) -> Self {
        Self {
            state,
            config,
            restart: None,
            cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Whether a running llama.cpp model will be interrupted by this swap.
    pub fn restart_applicable(&self) -> bool {
        self.restart.is_some()
    }

    pub fn cancel_flag(&self) -> Arc<std::sync::atomic::AtomicBool> {
        Arc::clone(&self.cancel)
    }
}

impl PromoteHost for LiveHost {
    fn cancelled(&self) -> bool {
        self.cancel.load(std::sync::atomic::Ordering::SeqCst)
    }

    fn model_running(&self) -> bool {
        self.restart.is_some()
    }

    async fn stop_model(&self) -> Result<(), String> {
        self.state.push_log(
            "[monitor] llama-runtime: stopping the running model for the runtime swap".into(),
        );
        stop_server(&self.state).await.map_err(|e| e.to_string())
    }

    async fn restart_model(&self) -> Result<(), String> {
        let Some(config) = self.restart.clone() else {
            return Ok(());
        };
        self.state.push_log(
            "[monitor] llama-runtime: restarting the model with its previous config".into(),
        );
        let state = Arc::new(self.state.clone());
        // The restart is the owner's own: it passes the launch gate.
        RUNTIME_OWNER
            .scope(true, async move {
                start_server(state, config, &self.config).await
            })
            .await
            .map_err(|e| e.to_string())
    }

    async fn verify(
        &self,
        dir: &Path,
        exe_name: &str,
        intent: &VerificationIntent,
        _stage: VerifyStage,
    ) -> Result<VerificationReport, String> {
        verify_runtime(dir, exe_name, intent)
            .await
            .map_err(|e| e.to_string())
    }
}
