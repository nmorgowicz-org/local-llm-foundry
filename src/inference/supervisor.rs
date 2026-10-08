use crate::state::AppState;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};

/// Everything the supervisor needs to spawn a process. Secrets are safe in `env`
/// because the supervisor only logs `redacted_summary`.
#[derive(Debug, Clone)]
pub struct SupervisedLaunch {
    pub program: PathBuf,
    pub args: Vec<OsString>,
    pub env: Vec<(OsString, OsString)>,
    pub cwd: Option<PathBuf>,
    pub port: u16,
    pub redacted_summary: String,
    /// Non-fatal issues discovered while building the launch (e.g. a chat-template
    /// overlay that failed and fell back to the model's native template). Surfaced
    /// to the UI's server log so degraded launches aren't silent.
    pub warnings: Vec<String>,
}

/// Callbacks fired for process output and unexpected exits.
pub trait BackendObserver: Send + Sync + 'static {
    fn on_log_line(&self, line: &str);
    fn on_crash(&self, exit_status: std::process::ExitStatus, tail: Vec<String>);
}

pub struct Supervisor {
    pub observer: Arc<dyn BackendObserver>,
    pub launch: SupervisedLaunch,
    pub state: Arc<AppState>,
    logs: Mutex<HashMap<u32, Arc<Mutex<Vec<String>>>>>,
    pid: Mutex<Option<u32>>,
    exited: tokio::sync::Notify,
    generation: u64,
    intentional_stop: AtomicBool,
    graceful_stop_timeout: std::time::Duration,
}

impl Supervisor {
    pub fn new(
        launch: SupervisedLaunch,
        observer: Arc<dyn BackendObserver>,
        state: Arc<AppState>,
        generation: u64,
    ) -> Self {
        Self {
            observer,
            launch,
            state,
            logs: Mutex::new(HashMap::new()),
            pid: Mutex::new(None),
            exited: tokio::sync::Notify::new(),
            generation,
            intentional_stop: AtomicBool::new(false),
            graceful_stop_timeout: std::time::Duration::from_secs(10),
        }
    }

    /// Override the graceful drain window. Production uses ten seconds; a
    /// shorter value is useful for deterministic fixture runtimes.
    #[allow(dead_code)]
    pub fn with_graceful_stop_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.graceful_stop_timeout = timeout;
        self
    }

    /// Spawn the configured process and begin log/death monitoring.
    ///
    /// The death watcher owns the `Child` so it can always reap it. `stop()`
    /// deliberately kills by PID and waits for the watcher to confirm exit.
    pub async fn start(self: Arc<Self>) -> Result<u32> {
        let mut child = self.spawn(self.launch.clone()).await?;
        let pid = child
            .id()
            .ok_or_else(|| anyhow::anyhow!("Failed to get child PID"))?;
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        *self.pid.lock().unwrap() = Some(pid);
        *self.state.server_child.lock().await = Some(pid);

        let log_buffer = Arc::new(Mutex::new(Vec::new()));
        self.logs
            .lock()
            .unwrap()
            .insert(pid, Arc::clone(&log_buffer));

        if let Some(out) = stdout {
            Self::spawn_log_reader(out, Arc::clone(&self.observer), Arc::clone(&log_buffer));
        }
        if let Some(err) = stderr {
            Self::spawn_log_reader(err, Arc::clone(&self.observer), Arc::clone(&log_buffer));
        }

        let supervisor = Arc::clone(&self);
        tokio::spawn(async move {
            supervisor.monitor_death(child, pid).await;
        });

        Ok(pid)
    }

    fn spawn_log_reader<R>(
        stream: R,
        observer: Arc<dyn BackendObserver>,
        buffer: Arc<Mutex<Vec<String>>>,
    ) where
        R: tokio::io::AsyncRead + Unpin + Send + 'static,
    {
        tokio::spawn(async move {
            let mut reader = BufReader::new(stream).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                let mut lines = buffer.lock().unwrap();
                lines.push(line.clone());
                if lines.len() > 200 {
                    lines.remove(0);
                }
                drop(lines);
                observer.on_log_line(&line);
            }
        });
    }

    async fn spawn(&self, launch: SupervisedLaunch) -> Result<Child> {
        let mut cmd = Command::new(&launch.program);
        crate::platform::no_window_tokio(&mut cmd);
        // Own process group so stop() can signal the whole tree (Python
        // workers included) instead of only the direct child.
        crate::inference::process_tree::configure_process_group(&mut cmd);
        cmd.args(&launch.args);
        cmd.envs(launch.env);
        cmd.stdin(std::process::Stdio::null()); // child has its own process group: an inherited tty stdin would SIGTTIN-stop it
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());
        if let Some(cwd) = &launch.cwd {
            cmd.current_dir(cwd);
        }

        cmd.spawn().with_context(|| {
            format!(
                "Failed to launch {} on port {}",
                self.launch.program.to_string_lossy(),
                self.launch.port
            )
        })
    }

    async fn monitor_death(&self, mut child: Child, pid: u32) {
        let wait_result = child.wait().await;
        let stopping = self.intentional_stop.load(Ordering::Acquire);

        if stopping {
            let mut current = self.state.server_child.lock().await;
            if *current == Some(pid) {
                *current = None;
            }
            drop(current);
            self.logs.lock().unwrap().remove(&pid);
            self.state.push_log(format!(
                "[monitor] process supervisor: pid={pid} exited during intentional stop"
            ));
            *self.pid.lock().unwrap() = None;
            self.exited.notify_waiters();
            return;
        }

        // Serialize unexpected-exit cleanup with start/stop. This closes the
        // gap where a replacement could register after the ownership check but
        // before the old watcher cleared state or reported its crash.
        let _lifecycle = self.state.server_lifecycle.lock().await;
        let owns_generation =
            self.state.server_generation.load(Ordering::Acquire) == self.generation;
        let is_current = owns_generation && {
            let mut current = self.state.server_child.lock().await;
            if *current == Some(pid) {
                *current = None;
                true
            } else {
                false
            }
        };

        // A replacement process may already own the active state. Never let a
        // stale watcher tear down the new backend.
        if !is_current {
            self.logs.lock().unwrap().remove(&pid);
            self.state.push_log(format!(
                "[monitor] process supervisor: pid={pid} exited after replacement; ignoring stale exit"
            ));
            *self.pid.lock().unwrap() = None;
            self.exited.notify_waiters();
            return;
        }

        {
            *self.state.server_running.lock().unwrap() = false;
            *self.state.local_server_running.lock().unwrap() = false;
            *self.state.server_config.lock().unwrap() = None;
            *self.state.backend.lock().unwrap() = None;
            *self.state.local_launch_request.lock().unwrap() = None;
        }
        *self.state.supervisor.lock().await = None;

        match wait_result {
            Ok(status) => {
                let tail = self
                    .logs
                    .lock()
                    .unwrap()
                    .remove(&pid)
                    .map(|buffer| {
                        let lines = buffer.lock().unwrap();
                        let start = lines.len().saturating_sub(20);
                        lines[start..]
                            .iter()
                            .filter(|line| !line.starts_with("[monitor]"))
                            .cloned()
                            .collect()
                    })
                    .unwrap_or_default();
                self.observer.on_crash(status, tail);
            }
            Err(error) => {
                self.logs.lock().unwrap().remove(&pid);
                let message = format!(
                    "Supervised process pid={pid} wait failed: {error}. The server has stopped."
                );
                self.state.push_log(format!("[monitor] {message}"));
                let active_id = self.state.active_session_id.lock().unwrap().clone();
                if !active_id.is_empty() {
                    self.state.update_session_status(
                        &active_id,
                        crate::state::SessionStatus::Error(message),
                    );
                }
            }
        }
        // Do not wake startup waiters until crash reporting and state cleanup
        // are complete; otherwise they can race ahead and reclassify this exit.
        *self.pid.lock().unwrap() = None;
        self.exited.notify_waiters();
    }

    pub async fn stop(&self) -> Result<()> {
        self.intentional_stop.store(true, Ordering::Release);
        let pid = *self.pid.lock().unwrap();
        let Some(pid) = pid else {
            self.intentional_stop.store(false, Ordering::Release);
            return Ok(());
        };

        let result = async {
            #[cfg(unix)]
            {
                // The child runs in its own process group, so signal the
                // group: orphaned Python workers must not keep the port or
                // accelerator memory after stop() returns.
                crate::inference::process_tree::signal_process_group(pid, true);
                if self.pid.lock().unwrap().is_none() {
                    // Already gone before the first signal.
                    return Ok(());
                }

                if tokio::time::timeout(self.graceful_stop_timeout, self.wait_for_exit())
                    .await
                    .is_err()
                {
                    self.state.push_log(format!(
                        "[monitor] process supervisor: pid={pid} did not exit after SIGTERM; sending SIGKILL"
                    ));
                    crate::inference::process_tree::signal_process_group(pid, false);
                }
            }
            #[cfg(windows)]
            {
                let mut kill_cmd = Command::new("taskkill");
                crate::platform::no_window_tokio(&mut kill_cmd);
                let status = kill_cmd
                    .args(["/F", "/T", "/PID", &pid.to_string()])
                    .status()
                    .await
                    .with_context(|| format!("Failed to terminate supervised process pid={pid}"))?;
                if !status.success() && self.pid.lock().unwrap().is_some() {
                    anyhow::bail!("Failed to terminate supervised process pid={pid}");
                }
            }

            tokio::time::timeout(std::time::Duration::from_secs(10), self.wait_for_exit())
                .await
                .with_context(|| {
                    format!("Timed out waiting for supervised process pid={pid} to exit")
                })?;
            Ok(())
        }
        .await;
        if result.is_err() {
            self.intentional_stop.store(false, Ordering::Release);
        }
        result
    }

    pub async fn wait_for_exit(&self) {
        loop {
            let notified = self.exited.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.pid.lock().unwrap().is_none() {
                return;
            }
            notified.await;
        }
    }
}

fn shell_quote(value: &std::ffi::OsStr) -> String {
    let value = value.to_string_lossy();
    if value.contains(' ') || value.contains('"') || value.contains('\'') || value.is_empty() {
        format!("\"{}\"", value.replace('"', "\\\""))
    } else {
        value.into_owned()
    }
}

fn is_sensitive_option(value: &str) -> bool {
    matches!(
        value,
        "--api-key" | "--token" | "--auth-token" | "--password" | "--secret"
    )
}

/// Render a useful spawn diagnostic without persisting credentials.
pub fn redacted_spawn_command(launch: &SupervisedLaunch) -> String {
    let mut parts = vec![shell_quote(launch.program.as_os_str())];
    let mut redact_next = false;

    for arg in &launch.args {
        let value = arg.to_string_lossy();
        if redact_next {
            parts.push("<redacted>".to_string());
            redact_next = false;
            continue;
        }

        if is_sensitive_option(&value) {
            parts.push(shell_quote(arg));
            redact_next = true;
            continue;
        }

        let mut redacted_assignment = None;
        if let Some((option, _)) = value.split_once('=')
            && is_sensitive_option(option)
        {
            redacted_assignment = Some(format!("{option}=<redacted>"));
        }
        parts.push(redacted_assignment.unwrap_or_else(|| shell_quote(arg)));
    }

    parts.join(" \\\n  ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_diagnostic_redacts_split_and_assignment_secrets() {
        let launch = SupervisedLaunch {
            program: "llama-server".into(),
            args: vec![
                "--api-key".into(),
                "split-secret".into(),
                "--api-key=assignment-secret".into(),
                "--alias".into(),
                "safe name".into(),
            ],
            env: vec![("PRIVATE_ENV".into(), "environment-secret".into())],
            cwd: None,
            port: 8080,
            redacted_summary: "test".into(),
            warnings: Vec::new(),
        };

        let diagnostic = redacted_spawn_command(&launch);
        assert!(diagnostic.contains("--api-key \\\n  <redacted>"));
        assert!(diagnostic.contains("--api-key=<redacted>"));
        assert!(diagnostic.contains("\"safe name\""));
        assert!(!diagnostic.contains("split-secret"));
        assert!(!diagnostic.contains("assignment-secret"));
        assert!(!diagnostic.contains("environment-secret"));
    }
}
