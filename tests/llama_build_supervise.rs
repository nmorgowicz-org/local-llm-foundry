//! Process containment for the llama.cpp build (plan §4.4, Phase 4 item 9).
//!
//! These run the real binary as the Unix supervisor. Each test spawns a
//! *grandchild* (`sleep`) through a shell and proves it is gone afterwards:
//! after cancel, after the build tool exits normally, and when the app itself
//! dies.

#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use llama_monitor::llama_build::supervise::{
    Launcher, Line, SUPERVISOR_ARG, SpawnSpec, process_start_time, run_contained,
};

fn launcher() -> Launcher {
    Launcher {
        supervisor_exe: Some(PathBuf::from(env!("CARGO_BIN_EXE_local-llm-foundry"))),
    }
}

fn alive(pid: u32) -> bool {
    // A zombie counts as gone for our purposes only once reaped; kill(0) on a
    // reaped pid fails with ESRCH.
    unsafe { libc::kill(pid as i32, 0) == 0 }
}

fn wait_until(what: &str, timeout: Duration, f: impl Fn() -> bool) {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if f() {
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    panic!("timed out waiting for: {what}");
}

fn read_pid(file: &Path) -> Option<u32> {
    std::fs::read_to_string(file).ok()?.trim().parse().ok()
}

/// A shell that starts a long-lived grandchild, records its pid and waits.
fn grandchild_script(pidfile: &Path) -> SpawnSpec {
    SpawnSpec::new(
        "sh",
        &[
            "-c",
            &format!(
                "sleep 300 & echo $! > '{}'; echo started; wait",
                pidfile.display()
            ),
        ],
    )
}

/// Re-execute only this test under a watchdog. Killing a stuck runner process
/// also closes its control pipe, so the real supervisor cleans up its child.
/// Unlike a timeout around a joined thread, this remains finite on the old code.
fn under_runner_watchdog(test: &str, body: impl FnOnce()) {
    const WORKER: &str = "FOUNDRY_SUPERVISE_REGRESSION_WORKER";
    if std::env::var(WORKER).ok().as_deref() == Some(test) {
        body();
        return;
    }
    let mut worker = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test, "--nocapture"])
        .env(WORKER, test)
        .stdin(Stdio::null())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(45);
    loop {
        match worker.try_wait() {
            Ok(Some(status)) => {
                assert!(status.success(), "{test}: worker failed: {status}");
                return;
            }
            Ok(None) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(25));
            }
            result => {
                let _ = worker.kill();
                let _ = worker.wait();
                panic!("{test}: runner failed to finish: {result:?}");
            }
        }
    }
}

#[test]
fn shutdown_burst_preserves_more_than_capacity_on_both_pipes() {
    under_runner_watchdog(
        "shutdown_burst_preserves_more_than_capacity_on_both_pipes",
        || {
            let mut counts = (0, 0);
            let out = run_contained(
                &SpawnSpec::new(
                    "sh",
                    &[
                        "-c",
                        "i=0; while [ \"$i\" -lt 1024 ]; do printf 'o\\n'; printf 'e\\n' >&2; i=$((i+1)); done; exit 7",
                    ],
                ),
                &launcher(),
                &mut |line| match line {
                    Line::Stdout(text) => {
                        assert_eq!(text, "o");
                        counts.0 += 1;
                    }
                    Line::Stderr(text) => {
                        assert_eq!(text, "e");
                        counts.1 += 1;
                    }
                },
                &|| false,
            )
            .unwrap();
            assert_eq!(out.exit_code, Some(7));
            assert!(!out.cancelled && out.contained);
            assert_eq!(counts, (1024, 1024));
        },
    );
}

#[test]
fn cancellation_drains_a_term_burst_larger_than_capacity() {
    under_runner_watchdog(
        "cancellation_drains_a_term_burst_larger_than_capacity",
        || {
            let ready = std::cell::Cell::new(false);
            let mut counts = (0, 0);
            // Install the trap before announcing readiness. Output is emitted
            // only in response to cancellation, while terminate() is waiting.
            // 2048 bytes per pipe fits the Linux/macOS pipe buffers even while
            // their reader is blocked on the full bounded channel.
            let script = "trap 'i=0; while [ \"$i\" -lt 1024 ]; do printf \"o\\n\"; printf \"e\\n\" >&2; i=$((i+1)); done; exit 0' TERM; echo ready; while :; do :; done";
            let out = run_contained(
                &SpawnSpec::new("sh", &["-c", script]),
                &launcher(),
                &mut |line| match line {
                    Line::Stdout(text) if text == "ready" => ready.set(true),
                    Line::Stdout(text) => {
                        assert_eq!(text, "o");
                        counts.0 += 1;
                    }
                    Line::Stderr(text) => {
                        assert_eq!(text, "e");
                        counts.1 += 1;
                    }
                },
                &|| ready.get(),
            )
            .unwrap();
            assert!(ready.get());
            assert!(out.cancelled && out.contained);
            assert_eq!(out.exit_code, None);
            assert_eq!(counts, (1024, 1024));
        },
    );
}

#[test]
fn normal_exit_streams_lines_and_reports_the_code() {
    let mut lines = Vec::new();
    let out = run_contained(
        &SpawnSpec::new(
            "sh",
            &["-c", "echo out1; echo err1 1>&2; echo out2; exit 3"],
        ),
        &launcher(),
        &mut |l| lines.push(l),
        &|| false,
    )
    .unwrap();
    assert_eq!(out.exit_code, Some(3));
    assert!(!out.cancelled && out.contained);
    assert!(lines.contains(&Line::Stdout("out1".into())));
    assert!(lines.contains(&Line::Stdout("out2".into())));
    assert!(lines.contains(&Line::Stderr("err1".into())));
}

#[test]
fn a_missing_program_is_reported_not_hung() {
    let out = run_contained(
        &SpawnSpec::new("/definitely/not/a/program", &[]),
        &launcher(),
        &mut |_| {},
        &|| false,
    )
    .unwrap();
    assert_eq!(out.exit_code, Some(127));
}

#[test]
fn the_child_environment_and_cwd_are_applied() {
    let dir = tempfile::tempdir().unwrap();
    let mut spec = SpawnSpec::new(
        "sh",
        &["-c", "echo \"$FOUNDRY_T:$(pwd)\"; echo \"${HOME:-nohome}\""],
    );
    spec.cwd = Some(dir.path().to_path_buf());
    spec.env = Some(vec![
        ("FOUNDRY_T".into(), "yes".into()),
        ("PATH".into(), "/usr/bin:/bin".into()),
    ]);
    let mut lines = Vec::new();
    run_contained(&spec, &launcher(), &mut |l| lines.push(l), &|| false).unwrap();
    let canonical = std::fs::canonicalize(dir.path()).unwrap();
    assert!(
        lines.contains(&Line::Stdout(format!("yes:{}", canonical.display()))),
        "{lines:?}"
    );
    assert!(
        lines.contains(&Line::Stdout("nohome".into())),
        "a cleared environment must not leak HOME: {lines:?}"
    );
}

#[test]
fn a_grandchild_left_behind_by_a_finished_tool_is_reaped() {
    let dir = tempfile::tempdir().unwrap();
    let pidfile = dir.path().join("pid");
    // The shell exits immediately but leaves `sleep 300` running.
    let spec = SpawnSpec::new(
        "sh",
        &[
            "-c",
            &format!("sleep 300 & echo $! > '{}'", pidfile.display()),
        ],
    );
    let out = run_contained(&spec, &launcher(), &mut |_| {}, &|| false).unwrap();
    assert_eq!(out.exit_code, Some(0));
    assert!(out.contained);
    let pid = read_pid(&pidfile).expect("grandchild pid");
    wait_until("orphaned writer reaped", Duration::from_secs(20), || {
        !alive(pid)
    });
}

#[test]
fn cancel_terminates_the_whole_process_tree() {
    let dir = tempfile::tempdir().unwrap();
    let pidfile = dir.path().join("pid");
    let pf = pidfile.clone();
    let started = Instant::now();
    let out = run_contained(
        &grandchild_script(&pidfile),
        &launcher(),
        &mut |_| {},
        &move || read_pid(&pf).is_some() && started.elapsed() > Duration::from_millis(300),
    )
    .unwrap();
    assert!(out.cancelled);
    assert!(
        out.contained,
        "containment must hold before ownership is released"
    );
    let pid = read_pid(&pidfile).expect("grandchild pid");
    wait_until(
        "grandchild gone after cancel",
        Duration::from_secs(20),
        || !alive(pid),
    );
    assert!(started.elapsed() < Duration::from_secs(25));
}

#[test]
fn a_tool_that_ignores_term_is_killed_after_the_grace_period() {
    let dir = tempfile::tempdir().unwrap();
    let pidfile = dir.path().join("pid");
    let spec = SpawnSpec::new(
        "sh",
        &[
            "-c",
            &format!(
                "trap '' TERM; sleep 300 & echo $! > '{}'; while true; do sleep 1; done",
                pidfile.display()
            ),
        ],
    );
    let pf = pidfile.clone();
    let out = run_contained(&spec, &launcher(), &mut |_| {}, &move || {
        read_pid(&pf).is_some()
    })
    .unwrap();
    assert!(out.cancelled && out.contained);
    let pid = read_pid(&pidfile).unwrap();
    wait_until("stubborn tree gone", Duration::from_secs(30), || {
        !alive(pid)
    });
}

/// The app dying closes the control pipe. Model it faithfully: the pipe's
/// writer is a separate process that we kill with SIGKILL.
#[test]
fn parent_death_reaps_the_tree_and_the_next_job_inherits_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let pidfile = dir.path().join("pid");
    let writer_pid = dir.path().join("writer");
    let exe = env!("CARGO_BIN_EXE_local-llm-foundry");
    // The subshell holding `sleep` plays the app: it owns the write end of the
    // supervisor's stdin pipe.
    let pipeline = format!(
        "sh -c \"echo \\$\\$ > '{writer}'; exec sleep 300\" | '{exe}' {arg} -- sh -c \"sleep 300 & echo \\$! > '{pidfile}'; wait\"",
        writer = writer_pid.display(),
        arg = SUPERVISOR_ARG,
        pidfile = pidfile.display(),
    );
    let mut harness = Command::new("sh")
        .arg("-c")
        .arg(&pipeline)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();

    wait_until("grandchild started", Duration::from_secs(20), || {
        read_pid(&pidfile).is_some()
    });
    let grandchild = read_pid(&pidfile).unwrap();
    let writer = read_pid(&writer_pid).unwrap();
    assert!(alive(grandchild));
    assert!(process_start_time(writer).is_some());

    // The "app" dies abruptly.
    unsafe { libc::kill(writer as i32, libc::SIGKILL) };

    wait_until(
        "supervisor reaps the tree after parent death",
        Duration::from_secs(30),
        || !alive(grandchild),
    );
    let _ = harness.wait();
}
