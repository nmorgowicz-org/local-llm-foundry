use clap::Parser;
use llama_monitor::llama_build::cli::{BuildCli, dry_plan};
use llama_monitor::llama_build::job::{PlanInputs, StepIo, StepResult, Toolchain};
use llama_monitor::llama_build::paths::BuildPaths;
use llama_monitor::llama_build::prereqs::{Check, Collector, PrereqReport};
use llama_monitor::llama_build::source::ResolvedRef;
use llama_monitor::llama_build::{Accelerator, BuildOptions, BuildPlan, TargetOs};
use llama_monitor::paths::AppPaths;
use std::collections::BTreeMap;
use std::path::Path;

#[test]
fn llama_build_cli_rejects_unbounded_or_ambiguous_options() {
    for args in [
        vec!["llama-build", "--jobs", "0"],
        vec!["llama-build", "--ref", "feature/unsafe"],
        vec!["llama-build", "--accelerator", "vulkan"],
        vec!["llama-build", "--headroom", "--jobs", "2"],
    ] {
        assert!(BuildCli::try_parse_from(args).is_err());
    }
    let args = BuildCli::try_parse_from(["llama-build", "--jobs", "1000"]).unwrap();
    assert!(args.options(TargetOs::Macos, 4).is_err());
    let args = BuildCli::try_parse_from(["llama-build", "--accelerator", "metal"]).unwrap();
    assert!(args.options(TargetOs::Windows, 4).is_err());
}

#[test]
fn llama_build_cli_maps_supported_flags() {
    let args = BuildCli::try_parse_from([
        "llama-build",
        "--accelerator",
        "cpu",
        "--ref",
        "master",
        "--headroom",
        "--wipe",
        "--dry-run",
    ])
    .unwrap();
    let options = args.options(TargetOs::Linux, 8).unwrap();
    assert_eq!(options.accelerator, Accelerator::Cpu);
    assert!(options.leave_headroom && options.wipe_build_dir && args.dry_run);
}

struct PlanOnly;
impl Toolchain for PlanOnly {
    fn os(&self) -> TargetOs {
        TargetOs::Linux
    }
    fn prereqs(&self, o: &BuildOptions) -> PrereqReport {
        let mut c = Collector::default();
        c.push(Check::pass("fake", "Prerequisite fixture", "read only"));
        PrereqReport::from_collector(self.os(), o.accelerator, c)
    }
    fn plan_inputs(&self, _: &BuildOptions) -> Result<PlanInputs, String> {
        Ok(PlanInputs {
            ccache: false,
            apple_silicon: false,
            cuda: None,
            cuda_root: None,
            device_id: None,
        })
    }
    fn resolve(&self, _: &BuildOptions) -> Result<ResolvedRef, String> {
        Ok(ResolvedRef {
            reference: "master".into(),
            commit: "0123456789012345678901234567890123456789".into(),
            build_number: None,
        })
    }
    fn fetch(&self, _: &Path, _: &ResolvedRef, _: &str, _: &mut StepIo) -> StepResult {
        panic!("dry run fetch")
    }
    fn configure(&self, _: &BuildPlan, _: &mut StepIo) -> StepResult {
        panic!("dry run configure")
    }
    fn compile(&self, _: &BuildPlan, _: usize, _: bool, _: &mut StepIo) -> StepResult {
        panic!("dry run compile")
    }
    fn ninja_targets(&self, _: &BuildPlan) -> Result<String, String> {
        panic!("dry run inventory")
    }
    fn toolchain_versions(&self) -> BTreeMap<String, String> {
        BTreeMap::new()
    }
    fn free_space(&self, _: &Path) -> Option<u64> {
        Some(u64::MAX)
    }
    fn logical_cpus(&self) -> usize {
        8
    }
}

#[test]
fn llama_build_dry_run_prints_exact_ref_and_argv_without_creating_root() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("never-created");
    let paths = BuildPaths::new(&AppPaths::from_root(root.clone()));
    let args = BuildCli::try_parse_from([
        "llama-build",
        "--accelerator",
        "cpu",
        "--ref",
        "master",
        "--jobs",
        "3",
    ])
    .unwrap();
    let (text, ready) = dry_plan(
        &PlanOnly,
        &paths,
        &args.options(TargetOs::Linux, 8).unwrap(),
    )
    .unwrap();
    assert!(ready);
    assert!(text.contains("0123456789012345678901234567890123456789"));
    assert!(text.contains("Prerequisite fixture"));
    assert!(text.contains("llama-server") && text.contains("llama-bench"));
    assert!(text.contains("--parallel") && text.contains("3"));
    assert!(text.contains("configure argv:") && text.contains("build argv:"));
    assert!(
        !root.exists(),
        "dry-run must not create config, build, log or staging directories"
    );
    assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 0);
}
