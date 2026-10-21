//! Real contained installer regressions, with shell fixtures instead of brew.
#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use llama_monitor::llama_build::service::{InstallFailure, run_installer};
use llama_monitor::llama_build::supervise::{Launcher, load_records};

fn launcher() -> Launcher {
    Launcher {
        supervisor_exe: Some(PathBuf::from(env!("CARGO_BIN_EXE_local-llm-foundry"))),
    }
}

fn read_pid(path: &Path) -> Option<u32> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

fn alive(pid: u32) -> bool {
    // SAFETY: signal zero only probes the fixture's process.
    unsafe { libc::kill(pid as i32, 0) == 0 }
}

fn wait_until(what: &str, f: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if f() {
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    panic!("timed out waiting for {what}");
}

// Keep a hung regression in a detached worker, not in the test's join path.
fn bounded_install(
    dir: &Path,
    script: &'static str,
    timeout: Duration,
) -> Result<String, InstallFailure> {
    let records = dir.join("processes.json");
    let pidfile = dir.join("pid");
    let (tx, rx) = mpsc::sync_channel(1);
    std::thread::spawn(move || {
        let result = run_installer(
            Path::new("/bin/sh"),
            &["-c", script, "install-test", pidfile.to_str().unwrap()],
            timeout,
            &launcher(),
            &records,
            "fixture-install",
        );
        let _ = tx.send(result);
    });
    rx.recv_timeout(Duration::from_secs(35))
        .expect("installer hung beyond its deadline and containment grace")
}

// The shell reaps its child on TERM, so the group-empty assertion does not
// depend on an external init reaping orphan zombies (notably in Linux CI).
const DESCENDANT: &str = r#"
sleep 300 & child=$!
trap 'kill "$child" 2>/dev/null; wait "$child"; exit 0' TERM
echo "$child" > "$1"
echo ready
wait "$child"
"#;

#[test]
fn installer_refuses_to_spawn_without_a_fresh_durable_record_barrier() {
    let dir = tempfile::tempdir().unwrap();
    let records = dir.path().join("processes.json");
    std::fs::write(&records, b"unresolved").unwrap();
    let error = bounded_install(
        dir.path(),
        "echo should-not-run > \"$1\"",
        Duration::from_secs(1),
    )
    .unwrap_err();
    assert!(error.contained, "no child was started");
    assert!(
        !dir.path().join("pid").exists(),
        "installer wrote before ownership was persisted"
    );
    assert_eq!(std::fs::read(records).unwrap(), b"unresolved");
}

#[test]
fn installer_deadline_reaps_descendant_and_inherited_pipes() {
    let dir = tempfile::tempdir().unwrap();
    let started = Instant::now();
    let error = bounded_install(dir.path(), DESCENDANT, Duration::from_secs(1)).unwrap_err();
    assert!(error.contained, "{error:?}");
    assert!(error.output.contains("timed out"), "{error:?}");
    assert!(error.output.contains("ready"), "{error:?}");
    let pid = read_pid(&dir.path().join("pid")).expect("fixture descendant started");
    assert!(!alive(pid), "descendant still owns the inherited pipes");
    assert!(!dir.path().join("processes.json").exists());
    assert!(started.elapsed() < Duration::from_secs(30));
}

#[test]
fn installer_output_capture_keeps_a_bounded_utf8_tail_on_failure() {
    let dir = tempfile::tempdir().unwrap();
    let error = bounded_install(
        dir.path(),
        "i=0; while [ $i -lt 10000 ]; do printf 'é-install-output-%s\\n' $i; i=$((i+1)); done; echo final-error >&2; exit 7",
        Duration::from_secs(20),
    )
    .unwrap_err();
    assert!(error.contained);
    assert!(error.output.len() <= 16 * 1024, "capture is not bounded");
    assert!(error.output.contains("install-output-9999"));
    assert!(!dir.path().join("processes.json").exists());
}

#[test]
fn installer_success_clears_its_process_record_and_sets_brew_overrides() {
    let dir = tempfile::tempdir().unwrap();
    let output = bounded_install(
        dir.path(),
        "printf '%s:%s\\n' \"$NONINTERACTIVE\" \"$HOMEBREW_NO_ENV_HINTS\"; echo installed",
        Duration::from_secs(10),
    )
    .unwrap();
    assert!(output.contains("1:1"));
    assert!(output.contains("installed"));
    assert!(!dir.path().join("processes.json").exists());
}

#[test]
fn installer_parent_death_worker() {
    let Some(dir) = std::env::var_os("FOUNDRY_INSTALL_TEST_WORKER") else {
        return;
    };
    let dir = PathBuf::from(dir);
    let _ = run_installer(
        Path::new("/bin/sh"),
        &[
            "-c",
            DESCENDANT,
            "install-test",
            dir.join("pid").to_str().unwrap(),
        ],
        Duration::from_secs(300),
        &launcher(),
        &dir.join("processes.json"),
        "parent-death-install",
    );
}

#[test]
fn installer_parent_death_uses_supervisor_cleanup_and_leaves_recovery_record() {
    let dir = tempfile::tempdir().unwrap();
    let mut app = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "installer_parent_death_worker", "--nocapture"])
        .env("FOUNDRY_INSTALL_TEST_WORKER", dir.path())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let ready = Instant::now() + Duration::from_secs(10);
    while Instant::now() < ready {
        if read_pid(&dir.path().join("pid")).is_some()
            && load_records(&dir.path().join("processes.json"))
                .is_ok_and(|records| records.len() == 1)
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    // Always close the app's control pipe, even if fixture startup failed.
    app.kill().unwrap();
    app.wait().unwrap();
    let pid = read_pid(&dir.path().join("pid")).expect("fixture descendant started");
    wait_until("installer descendant cleanup after app death", || {
        !alive(pid)
    });
    assert_eq!(
        load_records(&dir.path().join("processes.json"))
            .unwrap()
            .len(),
        1
    );
}
