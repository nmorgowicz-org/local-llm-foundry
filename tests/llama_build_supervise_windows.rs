//! Real Windows Job Object tests. Helpers are this Rust test binary, not shells.
#![cfg(windows)]

use llama_monitor::llama_build::supervise::{
    Launcher, SpawnSpec, process_start_time, run_contained,
};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

const HELPER: &str = "windows_tree_helper";

fn spec(root: &Path, finish: bool) -> SpawnSpec {
    let mut spec = SpawnSpec::new(
        std::env::current_exe().unwrap(),
        &["--exact", HELPER, "--nocapture"],
    );
    let mut env: std::collections::BTreeMap<String, String> = std::env::vars().collect();
    env.insert("LLF_TREE_ROLE".into(), "tool".into());
    env.insert("LLF_TREE_ROOT".into(), root.to_string_lossy().into_owned());
    env.insert(
        "LLF_TREE_FINISH".into(),
        if finish { "1" } else { "0" }.into(),
    );
    spec.env = Some(env.into_iter().collect());
    spec
}

fn pid(root: &Path, role: &str) -> Option<u32> {
    std::fs::read_to_string(root.join(format!("{role}.pid")))
        .ok()?
        .trim()
        .parse()
        .ok()
}

fn wait_for(label: &str, condition: impl Fn() -> bool) {
    let until = Instant::now() + Duration::from_secs(20);
    while !condition() {
        assert!(Instant::now() < until, "timed out waiting for {label}");
        std::thread::sleep(Duration::from_millis(25));
    }
}

#[test]
fn windows_tree_helper() {
    let Some(role) = std::env::var_os("LLF_TREE_ROLE") else {
        return;
    };
    let root = PathBuf::from(std::env::var_os("LLF_TREE_ROOT").unwrap());
    let role = role.to_string_lossy();
    if role == "app" {
        let outcome = run_contained(
            &spec(&root, false),
            &Launcher::current_exe(),
            &mut |_| {},
            &|| false,
        )
        .unwrap();
        assert!(outcome.contained);
        return;
    }
    std::fs::write(
        root.join(format!("{role}.pid")),
        std::process::id().to_string(),
    )
    .unwrap();
    if role == "tool" {
        let mut leaf = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", HELPER, "--nocapture"])
            .env("LLF_TREE_ROLE", "leaf")
            .spawn()
            .unwrap();
        wait_for("leaf startup", || pid(&root, "leaf").is_some());
        if std::env::var("LLF_TREE_FINISH").as_deref() == Ok("1") {
            return;
        }
        let _ = leaf.wait();
    } else {
        loop {
            std::thread::sleep(Duration::from_secs(1));
        }
    }
}

#[test]
fn windows_job_cancel_contains_grandchildren_before_return() {
    let root = tempfile::tempdir().unwrap();
    let started = Instant::now();
    let outcome = run_contained(
        &spec(root.path(), false),
        &Launcher::current_exe(),
        &mut |_| {},
        &|| pid(root.path(), "leaf").is_some() || started.elapsed() > Duration::from_secs(20),
    )
    .unwrap();
    assert!(outcome.cancelled && outcome.contained, "{outcome:?}");
    for role in ["tool", "leaf"] {
        let id = pid(root.path(), role).expect("helper started");
        wait_for("cancelled descendant exit", || {
            process_start_time(id).is_none()
        });
    }
}

#[test]
fn windows_job_normal_exit_cannot_leave_a_writer_for_the_next_job() {
    let root = tempfile::tempdir().unwrap();
    let outcome = run_contained(
        &spec(root.path(), true),
        &Launcher::current_exe(),
        &mut |_| {},
        &|| false,
    )
    .unwrap();
    assert!(
        outcome.contained && outcome.exit_code == Some(0),
        "{outcome:?}"
    );
    let leaf = pid(root.path(), "leaf").unwrap();
    wait_for("left-behind leaf exit", || {
        process_start_time(leaf).is_none()
    });
    let next = run_contained(
        &SpawnSpec::new("cmd", &["/d", "/c", "exit", "0"]),
        &Launcher::current_exe(),
        &mut |_| {},
        &|| false,
    )
    .unwrap();
    assert!(next.contained && next.exit_code == Some(0));
}

#[test]
fn windows_job_parent_death_closes_job_and_contains_grandchildren() {
    let root = tempfile::tempdir().unwrap();
    let mut app = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", HELPER, "--nocapture"])
        .env("LLF_TREE_ROLE", "app")
        .env("LLF_TREE_ROOT", root.path())
        .spawn()
        .unwrap();
    wait_for("owned process tree", || pid(root.path(), "leaf").is_some());
    let tool = pid(root.path(), "tool").unwrap();
    let leaf = pid(root.path(), "leaf").unwrap();
    app.kill().unwrap();
    app.wait().unwrap();
    for id in [tool, leaf] {
        wait_for("parent-death descendant exit", || {
            process_start_time(id).is_none()
        });
    }
}
