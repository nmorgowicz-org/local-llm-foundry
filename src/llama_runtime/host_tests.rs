use super::*;
use std::future::{Future, poll_fn};
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::Poll;
use std::time::Duration;

use crate::inference::kv_support::Backend;
use crate::inference::launch::LocalLaunchRequest;
use crate::llama_runtime::gate::RuntimeOp;
use crate::llama_runtime::promote::{PromoteRequest, RuntimePaths, promote, rollback, uninstall};

/// Use the real lifecycle preparation and stop path, but never execute a model
/// or runtime probe. These tests exercise the admission/registration interval.
struct LifecycleHost {
    live: LiveHost,
    stopped: AtomicBool,
}

impl PromoteHost for LifecycleHost {
    async fn prepare_model(&self) {
        self.live.prepare_model().await;
    }

    fn model_running(&self) -> bool {
        self.live.model_running()
    }

    async fn stop_model(&self) -> Result<(), String> {
        self.live.stop_model().await?;
        self.stopped.store(true, Ordering::SeqCst);
        Ok(())
    }

    async fn restart_model(&self) -> Result<(), String> {
        assert_eq!(
            self.live
                .restart
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .model_path,
            "in-flight.gguf"
        );
        Ok(())
    }

    async fn verify(
        &self,
        _dir: &Path,
        _exe_name: &str,
        _intent: &VerificationIntent,
        _stage: VerifyStage,
    ) -> Result<VerificationReport, String> {
        assert!(self.stopped.load(Ordering::SeqCst));
        Ok(VerificationReport {
            backend: Backend::Cpu,
            build_number: None,
            build_commit: None,
            identity_fingerprint: None,
            notes: Vec::new(),
            evidence_unavailable: false,
        })
    }
}

#[tokio::test]
async fn every_mutation_waits_for_an_admitted_launch_before_capturing_restart() {
    for op in [
        RuntimeOp::Promote,
        RuntimeOp::Rollback,
        RuntimeOp::Uninstall,
    ] {
        let root = tempfile::tempdir().unwrap();
        let paths = RuntimePaths::for_live_dir(&root.path().join("bin")).unwrap();
        let staged = paths.staging_dir("lifecycle-test");
        for dir in [&paths.live, &paths.previous, &staged] {
            std::fs::create_dir_all(dir).unwrap();
            std::fs::write(dir.join("llama-server"), "test runtime").unwrap();
        }
        let state = AppState::default();
        let host = LifecycleHost {
            live: LiveHost::new(state.clone(), Arc::new(AppConfig::for_test(None, None))),
            stopped: AtomicBool::new(false),
        };
        // Simulate start_backend after its gate check but before registration.
        let lifecycle = state.server_lifecycle.lock().await;
        crate::inference::launch::ensure_runtime_available(&state).unwrap();
        let mut guard = state.runtime_gate.try_begin(op).unwrap();
        let request = PromoteRequest {
            staged_dir: staged,
            label: "lifecycle test".into(),
            exe_name: "llama-server".into(),
            manifest: None,
            verification: VerificationIntent::prebuilt("cpu"),
        };
        let mut mutation = Box::pin(async {
            match op {
                RuntimeOp::Promote => promote(&host, &mut guard, &paths, request).await,
                RuntimeOp::Rollback => rollback(&host, &mut guard, &paths, "llama-server").await,
                RuntimeOp::Uninstall => uninstall(&host, &mut guard, &paths, "llama-server").await,
                _ => unreachable!(),
            }
        });
        // Poll once: the mutation must wait on lifecycle without writing its
        // journal or deciding that no model needs to be stopped.
        assert!(
            poll_fn(|cx| Poll::Ready(mutation.as_mut().poll(cx)))
                .await
                .is_pending()
        );
        assert!(!paths.journal.exists());
        assert!(!host.stopped.load(Ordering::SeqCst));
        let config = ServerConfig {
            model_path: "in-flight.gguf".into(),
            ..Default::default()
        };
        *state.server_config.lock().unwrap() = Some(config.clone());
        *state.local_launch_request.lock().unwrap() =
            Some(LocalLaunchRequest::LlamaCpp(Box::new(config)));
        *state.local_server_running.lock().unwrap() = true;
        drop(lifecycle);
        tokio::time::timeout(Duration::from_secs(5), mutation)
            .await
            .expect("mutation must complete after launch registration")
            .unwrap();
        assert!(host.stopped.load(Ordering::SeqCst), "{op:?}");
        assert!(host.live.restart_applicable(), "{op:?}");
        assert!(!*state.local_server_running.lock().unwrap());
    }
}

#[tokio::test]
async fn custom_runtime_host_does_not_adopt_the_managed_models_restart() {
    let state = AppState::default();
    *state.local_server_running.lock().unwrap() = true;
    *state.server_config.lock().unwrap() = Some(ServerConfig::default());
    *state.local_launch_request.lock().unwrap() =
        Some(LocalLaunchRequest::LlamaCpp(Box::default()));
    let _guard = state.runtime_gate.try_begin(RuntimeOp::Promote).unwrap();
    let host = LiveHost::without_model(state, Arc::new(AppConfig::for_test(None, None)));
    host.prepare_model().await;
    assert!(!host.model_running());
    assert!(!host.restart_applicable());
}
