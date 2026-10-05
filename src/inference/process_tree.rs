//! Process-group helpers shared by everything that spawns long-lived or
//! untrusted child processes.
//!
//! Children are placed in their own process group (group id == child pid) so
//! that timeouts, cancellation and shutdown can signal the whole tree instead
//! of only the direct child.
//!
//! PID-reuse safety: signals are only sent while the owning
//! `tokio::process::Child` handle has not been reaped. An un-reaped child
//! (running or zombie) keeps its pid, and therefore its process-group id,
//! reserved by the kernel, so the signal cannot reach an unrelated process.
//! [`terminate_and_reap`] and [`signal_child_group`] enforce that check; the raw
//! [`signal_process_group`] / [`terminate_process_tree`] functions do not and
//! must only be used with a pid whose handle is known to be live.

use std::time::Duration;
use tokio::process::{Child, Command};

/// Grace period between SIGTERM and SIGKILL for [`graceful_stop_and_reap`].
pub const DEFAULT_GRACE: Duration = Duration::from_secs(5);

/// Put the spawned child into a fresh process group (`setpgid(0, 0)`).
#[cfg(unix)]
pub fn configure_process_group(command: &mut Command) {
    use std::os::unix::process::CommandExt;
    command.as_std_mut().process_group(0);
}

#[cfg(not(unix))]
pub fn configure_process_group(_command: &mut Command) {}

/// Same as [`configure_process_group`] for `std::process::Command`.
#[cfg(unix)]
pub fn configure_process_group_std(command: &mut std::process::Command) {
    use std::os::unix::process::CommandExt;
    command.process_group(0);
}

#[cfg(not(unix))]
pub fn configure_process_group_std(_command: &mut std::process::Command) {}

/// Send SIGTERM (`graceful`) or SIGKILL (`!graceful`) to the process group
/// whose id equals `pid`. Best-effort; errors (ESRCH, EPERM) are ignored.
///
/// Callers must make sure the child handle for `pid` has not been reaped.
#[cfg(unix)]
pub fn signal_process_group(pid: u32, graceful: bool) {
    if pid <= 1 || pid > i32::MAX as u32 {
        return;
    }
    let signal = if graceful {
        libc::SIGTERM
    } else {
        libc::SIGKILL
    };
    // SAFETY: kill is called with a validated (>1) pid negated to address the
    // process group created by `configure_process_group`, and a constant signal.
    unsafe {
        libc::kill(-(pid as libc::pid_t), signal);
    }
}

#[cfg(not(unix))]
pub fn signal_process_group(_pid: u32, _graceful: bool) {}

/// SIGKILL the whole process group whose id is `pid`.
pub fn terminate_process_tree(pid: u32) {
    signal_process_group(pid, false);
}

/// Returns true while the child handle has not been reaped.
fn child_unreaped(child: &mut Child) -> bool {
    matches!(child.try_wait(), Ok(None))
}

/// Signal the child's process group only if the child has not been reaped.
/// Returns whether a signal was sent.
pub fn signal_child_group(child: &mut Child, graceful: bool) -> bool {
    let Some(pid) = child.id() else {
        return false;
    };
    if !child_unreaped(child) {
        return false;
    }
    signal_process_group(pid, graceful);
    true
}

/// Kill the whole process tree and reap the direct child.
///
/// The group is signalled only while the child handle is still un-reaped,
/// which guarantees we own that process-group ID. After a reap the ID could
/// already belong to an unrelated group, so a stale `pid` fallback is never
/// signalled. Callers that need to clean up descendants must keep the child
/// un-reaped (and thus the group) until their output readers finish.
pub async fn terminate_and_reap(child: &mut Child, _pid: u32) {
    let Some(live_pid) = child.id() else {
        return;
    };
    signal_process_group(live_pid, false);
    let _ = child.start_kill();
    let _ = child.wait().await;
}

/// True while any process remains in the group `pid` (signal-0 probe).
#[cfg(unix)]
pub fn process_group_exists(pid: u32) -> bool {
    if pid <= 1 || pid > i32::MAX as u32 {
        return false;
    }
    // SAFETY: signal 0 performs only an existence/permission check.
    let result = unsafe { libc::kill(-(pid as libc::pid_t), 0) };
    result == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(not(unix))]
pub fn process_group_exists(_pid: u32) -> bool {
    false
}

/// SIGTERM the group, wait up to `grace` for the leader and its descendants to
/// exit, then SIGKILL anything left in the group and reap the leader.
/// Returns true if everything exited within the grace period.
pub async fn graceful_stop_and_reap(child: &mut Child, grace: Duration) -> bool {
    let Some(pid) = child.id() else {
        let _ = child.wait().await;
        return true;
    };
    if !signal_child_group(child, true) {
        // The leader was already reaped, but descendants may still live in
        // the group (e.g. a python child holding stdout). Kill anything
        // remaining before declaring success.
        if process_group_exists(pid) {
            signal_process_group(pid, false);
            while process_group_exists(pid) {
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
        }
        let _ = child.wait().await;
        return true;
    }
    let deadline = tokio::time::Instant::now() + grace;
    let leader_exited = tokio::time::timeout(grace, child.wait()).await.is_ok();
    let mut clean = leader_exited;
    if leader_exited {
        // The leader is reaped, but descendants that ignore SIGTERM may remain
        // in the group. Give them the rest of the grace period.
        while process_group_exists(pid) {
            if tokio::time::Instant::now() >= deadline {
                clean = false;
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        if !clean {
            signal_process_group(pid, false);
        }
    } else {
        signal_process_group(pid, false);
        let _ = child.start_kill();
        let _ = child.wait().await;
    }
    clean
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::process::Stdio;

    fn pid_alive(pid: i32) -> bool {
        // SAFETY: signal 0 only checks for existence.
        unsafe { libc::kill(pid, 0) == 0 }
    }

    async fn spawn_tree(dir: &std::path::Path, trap_term: bool) -> (Child, i32) {
        let pid_file = dir.join("grandchild.pid");
        let script = if trap_term {
            format!(
                "(trap '' TERM; sleep 60) & echo $! > {p}; wait",
                p = pid_file.display()
            )
        } else {
            format!("sleep 60 & echo $! > {p}; wait", p = pid_file.display())
        };
        let mut command = Command::new("/bin/sh");
        command
            .arg("-c")
            .arg(script)
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        configure_process_group(&mut command);
        let child = command.spawn().unwrap();
        let mut grandchild = 0;
        for _ in 0..200 {
            if let Ok(text) = std::fs::read_to_string(&pid_file)
                && let Ok(pid) = text.trim().parse::<i32>()
            {
                grandchild = pid;
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(grandchild > 0, "grandchild pid was not recorded");
        (child, grandchild)
    }

    async fn wait_dead(pid: i32) {
        for _ in 0..200 {
            // A killed orphan is reparented and reaped by init; allow time.
            if !pid_alive(pid) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("process {pid} still alive");
    }

    #[tokio::test]
    async fn terminate_and_reap_kills_descendants() {
        let dir = tempfile::tempdir().unwrap();
        let (mut child, grandchild) = spawn_tree(dir.path(), false).await;
        let pid = child.id().unwrap();
        terminate_and_reap(&mut child, pid).await;
        wait_dead(grandchild).await;
    }

    #[tokio::test]
    async fn graceful_stop_escalates_to_kill_for_term_ignoring_descendants() {
        let dir = tempfile::tempdir().unwrap();
        let (mut child, grandchild) = spawn_tree(dir.path(), true).await;
        let clean = graceful_stop_and_reap(&mut child, Duration::from_millis(400)).await;
        assert!(!clean, "a TERM-ignoring descendant must force escalation");
        wait_dead(grandchild).await;
    }

    #[tokio::test]
    async fn graceful_stop_is_clean_when_tree_honours_term() {
        let dir = tempfile::tempdir().unwrap();
        let (mut child, grandchild) = spawn_tree(dir.path(), false).await;
        let clean = graceful_stop_and_reap(&mut child, Duration::from_secs(3)).await;
        assert!(clean);
        wait_dead(grandchild).await;
    }

    #[tokio::test]
    async fn signal_child_group_refuses_reaped_child() {
        let mut command = Command::new("/bin/sh");
        command.arg("-c").arg("exit 0");
        configure_process_group(&mut command);
        let mut child = command.spawn().unwrap();
        let _ = child.wait().await;
        assert!(!signal_child_group(&mut child, false));
    }

    #[test]
    fn invalid_pids_are_ignored() {
        signal_process_group(0, false);
        signal_process_group(1, false);
        signal_process_group(u32::MAX, true);
    }
}
