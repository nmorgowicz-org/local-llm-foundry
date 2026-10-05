use std::process::Command;

fn run(binary: &str, args: &[&str]) -> std::process::Output {
    Command::new(binary)
        .args(args)
        .output()
        .unwrap_or_else(|error| panic!("failed to run {binary}: {error}"))
}

#[test]
fn canonical_version_uses_foundry_identity() {
    let output = run(env!("CARGO_BIN_EXE_local-llm-foundry"), &["--version"]);
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).starts_with("local-llm-foundry "));
    assert!(!String::from_utf8_lossy(&output.stderr).contains("[compat]"));
}

#[test]
fn canonical_help_uses_foundry_identity() {
    let output = run(env!("CARGO_BIN_EXE_local-llm-foundry"), &["--help"]);
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("Usage: local-llm-foundry"));
}

#[test]
fn canonical_invalid_arguments_fail() {
    let output = run(
        env!("CARGO_BIN_EXE_local-llm-foundry"),
        &["--not-a-real-flag"],
    );
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("--not-a-real-flag"));
}

#[test]
fn manifest_has_only_the_canonical_executable() {
    let manifest = include_str!("../Cargo.toml");
    assert_eq!(manifest.matches("[[bin]]").count(), 1);
    let executable = manifest.split("[[bin]]").nth(1).unwrap();
    assert!(executable.starts_with("\nname = \"local-llm-foundry\"\npath = \"src/main.rs\"\n"));
    assert!(manifest.contains("autobins = false"));
    let workflow = include_str!("../.github/workflows/release.yml");
    assert!(!workflow.contains("release/llama-monitor"));
    assert!(!workflow.contains("windows-bundle-legacy"));
    assert!(!workflow.contains("llama-monitor-linux-"));
    assert!(!workflow.contains("llama-monitor-windows-"));
    assert!(!workflow.contains("llama-monitor-macos-"));
}
