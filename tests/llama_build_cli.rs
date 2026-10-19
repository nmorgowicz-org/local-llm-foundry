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

mod repair_processes {
    use clap::Parser;
    use llama_monitor::llama_build::cache::runtime_process_lease;
    use llama_monitor::llama_build::cli::{BuildCli, repair_processes_command};
    use llama_monitor::llama_build::paths::BuildPaths;
    use llama_monitor::llama_build::repair::{ProcessInfo, ProcessLister};
    use llama_monitor::paths::AppPaths;
    use std::path::Path;

    struct Fake(Vec<ProcessInfo>);
    impl ProcessLister for Fake {
        fn list(&self) -> Vec<ProcessInfo> {
            self.0.clone()
        }
        fn current_pid(&self) -> u32 {
            1
        }
    }

    fn rig() -> (tempfile::TempDir, std::path::PathBuf, BuildPaths) {
        let tmp = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(tmp.path()).unwrap();
        let paths = BuildPaths::new(&AppPaths::from_root(root.clone()));
        std::fs::create_dir_all(&paths.root).unwrap();
        std::fs::write(
            &paths.process_records,
            r#"[{"job_id":"old","pid":4000001,"start_time":100}]"#,
        )
        .unwrap();
        (tmp, root, paths)
    }

    #[test]
    fn the_flag_parses_and_excludes_every_build_option() {
        assert!(
            BuildCli::try_parse_from(["llama-build", "--repair-processes"])
                .unwrap()
                .repair_processes
        );
        for other in ["--dry-run", "--wipe", "--headroom"] {
            assert!(
                BuildCli::try_parse_from(["llama-build", "--repair-processes", other]).is_err(),
                "{other}"
            );
        }
        assert!(
            BuildCli::try_parse_from(["llama-build", "--repair-processes", "--jobs", "2"]).is_err()
        );
        assert!(
            BuildCli::try_parse_from(["llama-build", "--repair-processes", "--accelerator", "cpu"])
                .is_err()
        );
        assert!(
            !BuildCli::try_parse_from(["llama-build"])
                .unwrap()
                .repair_processes
        );
    }

    #[test]
    fn stale_records_are_cleared_and_exit_is_zero() {
        let (_t, root, paths) = rig();
        let (code, text) = repair_processes_command(&root, &Fake(vec![]));
        assert_eq!(code, 0, "{text}");
        assert!(text.contains("Cleared 1"), "{text}");
        assert!(!paths.process_records.exists());
    }

    #[test]
    fn a_live_process_refuses_with_exit_one_and_changes_nothing() {
        let (_t, root, paths) = rig();
        let live = Fake(vec![ProcessInfo {
            pid: 4_000_001,
            start_time: 100,
            command: "cmake --build".into(),
            cwd: None,
        }]);
        let (code, text) = repair_processes_command(&root, &live);
        assert_eq!(code, 1, "{text}");
        assert!(
            text.contains("4000001") && text.contains("cmake --build"),
            "{text}"
        );
        assert!(paths.process_records.exists());
    }

    #[test]
    fn a_running_app_or_builder_is_busy_exit_75() {
        let (_t, root, paths) = rig();
        let _app = runtime_process_lease(&root, false).unwrap();
        let (code, text) = repair_processes_command(&root, &Fake(vec![]));
        assert_eq!(code, 75, "{text}");
        assert!(paths.process_records.exists());
    }

    #[test]
    fn a_missing_root_is_nothing_to_repair_and_is_not_created() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("absent");
        let (code, text) = repair_processes_command(&root, &Fake(vec![]));
        assert_eq!(code, 0, "{text}");
        assert!(!Path::new(&root).exists());
    }
}
