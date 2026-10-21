//! macOS prerequisites (plan §5.1).
//!
//! Full Xcode is **not** required: a Metal build with only the Command Line
//! Tools succeeded in the S4 spike (the `.metal` source is embedded and
//! compiled at runtime).

use std::path::Path;
use std::time::Duration;

use super::prereqs::{
    CMAKE_MINIMUM, Check, Collector, DiskInputs, SmokeTest, Status, action_fix, copy_fix,
    disk_check, estimate_disk, link_fix, network_checks, parse_version, smoke_check,
    version_string,
};
use super::sysprobe::SystemProbe;
use super::{Accelerator, gpu};

const QUERY_TIMEOUT: Duration = Duration::from_secs(20);

/// One-click installs are limited to these formulae (decision D2). The list is
/// checked server-side; free text is never passed to `brew`.
pub const BREW_ALLOWLIST: [&str; 4] = ["git", "cmake", "ninja", "ccache"];

/// Validate a requested package list against the allowlist, de-duplicated and
/// in a stable order.
pub fn validate_brew_packages(requested: &[String]) -> Result<Vec<&'static str>, String> {
    if requested.is_empty() {
        return Err("no packages were requested".into());
    }
    let mut out = Vec::new();
    for name in requested {
        match BREW_ALLOWLIST.iter().find(|allowed| **allowed == name) {
            Some(allowed) => {
                if !out.contains(allowed) {
                    out.push(*allowed);
                }
            }
            None => {
                return Err(format!(
                    "`{name}` cannot be installed from here; allowed: {}",
                    BREW_ALLOWLIST.join(", ")
                ));
            }
        }
    }
    out.sort_by_key(|p| BREW_ALLOWLIST.iter().position(|a| a == p));
    Ok(out)
}

pub const BREW_INSTALL_URL: &str = "https://brew.sh";

/// Everything a check run needs beyond the machine itself.
pub struct MacContext<'a> {
    pub accelerator: Accelerator,
    pub app_root: &'a Path,
    pub disk: DiskInputs,
    pub check_network: bool,
}

fn tool_check(
    sys: &dyn SystemProbe,
    c: &mut Collector,
    id: &str,
    title: &str,
    name: &str,
    brew_ok: bool,
    required: bool,
) -> Option<String> {
    let Some(path) = sys.which(name) else {
        let status = if required { Status::Fail } else { Status::Warn };
        let mut check = Check::new(
            id,
            title,
            status,
            if required {
                format!("{name} was not found.")
            } else {
                format!("{name} was not found (optional; it speeds up rebuilds).")
            },
        );
        check = if brew_ok {
            check.with_fix(action_fix(
                &format!("brew_install:{name}"),
                &format!("Install {name} with Homebrew"),
            ))
        } else {
            check.with_fix(link_fix(BREW_INSTALL_URL, "Install Homebrew first"))
        };
        c.push(check);
        return None;
    };
    let exe = path.to_string_lossy().into_owned();
    let out = sys.run(&exe, &["--version"], QUERY_TIMEOUT);
    let (version_text, ok) = match &out {
        Ok(o) if o.ok() => (o.combined(), true),
        Ok(o) => (o.combined(), false),
        Err(e) => (e.clone(), false),
    };
    let mut check = if ok {
        let version = parse_version(&version_text).map(version_string);
        Check::pass(
            id,
            title,
            match &version {
                Some(v) => format!("{name} {v} at {exe}"),
                None => format!("{name} at {exe}"),
            },
        )
    } else {
        Check::fail(
            id,
            title,
            format!("{exe} did not run: {}", version_text.trim()),
        )
    };
    check = check.with_evidence(format!("{exe} --version\n{}", version_text.trim()));
    c.push(check);
    ok.then_some(version_text)
}

/// All macOS checks, in the order the panel shows them.
pub fn macos_checks(sys: &dyn SystemProbe, ctx: &MacContext, smoke: &dyn SmokeTest) -> Collector {
    let mut c = Collector::default();

    // Command Line Tools.
    let clt = sys.run("xcode-select", &["-p"], QUERY_TIMEOUT);
    let clt_ok = matches!(&clt, Ok(o) if o.ok() && !o.stdout.trim().is_empty());
    if clt_ok {
        let path = clt
            .as_ref()
            .map(|o| o.stdout.trim().to_string())
            .unwrap_or_default();
        let clang = sys
            .run("clang", &["--version"], QUERY_TIMEOUT)
            .ok()
            .filter(|o| o.ok())
            .map(|o| o.stdout.lines().next().unwrap_or("").to_string());
        c.fingerprint_part("xcode_select", &path);
        if let Some(line) = &clang {
            c.fingerprint_part("clang", line);
        }
        c.push(
            Check::pass(
                "clt",
                "Command Line Tools",
                format!("Developer directory {path}"),
            )
            .with_evidence(format!("xcode-select -p\n{path}"))
            .with_evidence(clang.unwrap_or_else(|| "clang --version produced no output".into())),
        );
    } else {
        c.push(
            Check::fail(
                "clt",
                "Command Line Tools",
                "The Xcode Command Line Tools are not installed. Full Xcode is not required.",
            )
            .with_fix(copy_fix("xcode-select --install")),
        );
    }

    // Homebrew.
    let brew = sys.which("brew");
    let brew_ok = brew.is_some();
    match &brew {
        Some(path) => {
            let exe = path.to_string_lossy().into_owned();
            let version = sys
                .run(&exe, &["--version"], QUERY_TIMEOUT)
                .ok()
                .map(|o| o.combined());
            c.fingerprint_part("brew", &exe);
            c.push(
                Check::pass("brew", "Homebrew", format!("Homebrew at {exe}"))
                    .with_evidence(version.unwrap_or_default()),
            );
        }
        None => c.push(
            Check::warn(
                "brew",
                "Homebrew",
                "Homebrew was not found. It is the easiest way to install the other tools; \
                 install it from brew.sh (the installer is never piped to a shell from here).",
            )
            .with_fix(link_fix(BREW_INSTALL_URL, "Install Homebrew")),
        ),
    }

    let git = tool_check(sys, &mut c, "git", "git", "git", brew_ok, true);
    if let Some(v) = &git {
        c.fingerprint_part("git", v.trim());
    }

    // cmake with a version floor.
    let cmake = tool_check(sys, &mut c, "cmake", "CMake", "cmake", brew_ok, true);
    if let Some(text) = &cmake {
        let version = parse_version(text);
        c.fingerprint_part("cmake", text.lines().next().unwrap_or(""));
        match version {
            Some(v) if v < CMAKE_MINIMUM => {
                let at = c.checks.len() - 1;
                c.checks[at] = Check::fail(
                    "cmake",
                    "CMake",
                    format!(
                        "CMake {} is older than the required {}.",
                        version_string(v),
                        version_string(CMAKE_MINIMUM)
                    ),
                )
                .with_fix(if brew_ok {
                    action_fix("brew_install:cmake", "Upgrade CMake with Homebrew")
                } else {
                    link_fix(BREW_INSTALL_URL, "Install Homebrew first")
                });
            }
            None => {
                let at = c.checks.len() - 1;
                c.checks[at].status = Status::Warn;
                c.checks[at].detail.push_str(" (version not recognized)");
            }
            Some(_) => {}
        }
    }

    let ninja = tool_check(sys, &mut c, "ninja", "Ninja", "ninja", brew_ok, true);
    if let Some(v) = &ninja {
        c.fingerprint_part("ninja", v.trim());
    }
    let ccache = tool_check(sys, &mut c, "ccache", "ccache", "ccache", brew_ok, false);
    c.fingerprint_part(
        "ccache",
        if ccache.is_some() {
            "present"
        } else {
            "absent"
        },
    );

    // Metal device.
    c.push(metal_device_check(sys));

    // Disk.
    let estimate = estimate_disk(&ctx.disk);
    c.push(disk_check(&estimate, sys.free_space(ctx.app_root)));

    // Network.
    if ctx.check_network && git.is_some() {
        for check in network_checks(sys, false) {
            c.push(check);
        }
    }

    // Functional tier last: only meaningful when the tools exist.
    let tools_ok = clt_ok && git.is_some() && cmake.is_some() && ninja.is_some();
    c.push(if tools_ok {
        smoke_check(smoke.run(ctx.accelerator, None), true)
    } else {
        smoke_check(Err(String::new()), false)
    });
    c
}

/// Metal support from `system_profiler SPDisplaysDataType -json`.
pub fn parse_metal_device(json: &str) -> Option<(String, String)> {
    let value: serde_json::Value = serde_json::from_str(json).ok()?;
    for gpu in value.get("SPDisplaysDataType")?.as_array()? {
        let family = gpu
            .get("spdisplays_mtlgpufamilysupport")
            .and_then(|v| v.as_str())
            .filter(|f| f.to_ascii_lowercase().contains("metal"));
        if let Some(family) = family {
            let name = gpu
                .get("sppci_model")
                .or_else(|| gpu.get("_name"))
                .and_then(|v| v.as_str())
                .unwrap_or("GPU");
            return Some((name.to_string(), family.to_string()));
        }
    }
    None
}

fn metal_device_check(sys: &dyn SystemProbe) -> Check {
    const ID: &str = "metal_device";
    const TITLE: &str = "Metal device";
    let out = sys.run(
        "system_profiler",
        &["SPDisplaysDataType", "-json"],
        Duration::from_secs(30),
    );
    if let Ok(o) = &out
        && o.ok()
        && let Some((name, family)) = parse_metal_device(&o.stdout)
    {
        return Check::pass(ID, TITLE, format!("{name} ({family})"))
            .with_evidence("system_profiler SPDisplaysDataType -json");
    }
    // Fall back to the chip name: Apple silicon always has Metal.
    if let Some(chip) = gpu::detect_apple_chip(sys)
        && chip.apple_silicon
    {
        return Check::pass(ID, TITLE, format!("{} (Apple silicon)", chip.name));
    }
    Check::warn(
        ID,
        TITLE,
        "No Metal device was reported. The build will compile, but the post-build backend check \
         will fail if Metal cannot load.",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama_build::TargetOs;
    use crate::llama_build::sysprobe::FakeSystem;

    const DISPLAYS: &str = include_str!("../../tests/fixtures/llama_build/macos_displays.json");

    struct Smoke(Result<String, String>);
    impl SmokeTest for Smoke {
        fn run(
            &self,
            _a: Accelerator,
            _tc: Option<&crate::llama_build::prereqs_windows::WindowsToolchain>,
        ) -> Result<String, String> {
            self.0.clone()
        }
    }

    fn ctx(app_root: &Path) -> MacContext<'_> {
        MacContext {
            accelerator: Accelerator::Metal,
            app_root,
            disk: DiskInputs {
                os: TargetOs::Macos,
                accelerator: Accelerator::Metal,
                checkout_present: false,
                build_tree_present: false,
                ccache_enabled: true,
                ccache_current_bytes: 0,
                ccache_cap_bytes: 2 << 30,
            },
            check_network: false,
        }
    }

    fn healthy() -> FakeSystem {
        FakeSystem::new(TargetOs::Macos)
            .on("xcode-select -p", "/Library/Developer/CommandLineTools\n")
            .on(
                "clang --version",
                "Apple clang version 21.0.0 (clang-2100.3.34.2)\nTarget: arm64\n",
            )
            .with_program("brew", "/opt/homebrew/bin/brew")
            .on("/opt/homebrew/bin/brew --version", "Homebrew 7.0.9\n")
            .with_program("git", "/opt/homebrew/bin/git")
            .on("/opt/homebrew/bin/git --version", "git version 2.55.0\n")
            .with_program("cmake", "/opt/homebrew/bin/cmake")
            .on(
                "/opt/homebrew/bin/cmake --version",
                "cmake version 4.4.3\n\nCMake suite maintained\n",
            )
            .with_program("ninja", "/opt/homebrew/bin/ninja")
            .on("/opt/homebrew/bin/ninja --version", "1.13.2\n")
            .on("system_profiler SPDisplaysDataType -json", DISPLAYS)
    }

    fn by_id<'a>(c: &'a Collector, id: &str) -> &'a Check {
        c.checks
            .iter()
            .find(|ch| ch.id == id)
            .unwrap_or_else(|| panic!("no {id}"))
    }

    #[test]
    fn a_healthy_machine_passes_with_ccache_only_a_warning() {
        let sys = healthy();
        let c = macos_checks(&sys, &ctx(Path::new("/tmp")), &Smoke(Ok("built".into())));
        for id in [
            "clt",
            "brew",
            "git",
            "cmake",
            "ninja",
            "metal_device",
            "disk",
            "smoke",
        ] {
            assert_eq!(
                by_id(&c, id).status,
                Status::Pass,
                "{id}: {}",
                by_id(&c, id).detail
            );
        }
        let ccache = by_id(&c, "ccache");
        assert_eq!(ccache.status, Status::Warn);
        assert!(
            matches!(&ccache.fix, Some(crate::llama_build::prereqs::Fix::Action { id, .. }) if id == "brew_install:ccache")
        );
        assert!(!c.checks.iter().any(|ch| ch.status == Status::Fail));
        assert!(by_id(&c, "cmake").detail.contains("4.4.3"));
    }

    #[test]
    fn missing_command_line_tools_offers_the_apple_dialog_and_skips_smoke() {
        let sys = FakeSystem::new(TargetOs::Macos)
            .on_status(
                "xcode-select -p",
                2,
                "",
                "xcode-select: error: unable to get active developer directory",
            )
            .with_program("git", "/usr/bin/git")
            .on("/usr/bin/git --version", "git version 2.39.5\n")
            .with_program("cmake", "/opt/homebrew/bin/cmake")
            .on("/opt/homebrew/bin/cmake --version", "cmake version 4.4.3\n")
            .with_program("ninja", "/opt/homebrew/bin/ninja")
            .on("/opt/homebrew/bin/ninja --version", "1.13.2\n");
        let c = macos_checks(&sys, &ctx(Path::new("/tmp")), &Smoke(Ok(String::new())));
        let clt = by_id(&c, "clt");
        assert_eq!(clt.status, Status::Fail);
        assert!(
            matches!(&clt.fix, Some(crate::llama_build::prereqs::Fix::Copy { command }) if command == "xcode-select --install")
        );
        assert_eq!(by_id(&c, "smoke").status, Status::Skipped);
    }

    #[test]
    fn without_homebrew_the_tool_fixes_point_at_brew_not_a_dead_action() {
        let sys = FakeSystem::new(TargetOs::Macos)
            .on("xcode-select -p", "/Library/Developer/CommandLineTools\n");
        let c = macos_checks(&sys, &ctx(Path::new("/tmp")), &Smoke(Ok(String::new())));
        assert_eq!(by_id(&c, "brew").status, Status::Warn);
        for id in ["git", "cmake", "ninja"] {
            let check = by_id(&c, id);
            assert_eq!(check.status, Status::Fail);
            assert!(
                matches!(
                    &check.fix,
                    Some(crate::llama_build::prereqs::Fix::Link { .. })
                ),
                "{id}"
            );
        }
    }

    #[test]
    fn old_cmake_fails_with_an_upgrade_action() {
        let sys = healthy().on(
            "/opt/homebrew/bin/cmake --version",
            "cmake version 3.20.6\n",
        );
        let c = macos_checks(&sys, &ctx(Path::new("/tmp")), &Smoke(Ok(String::new())));
        let cmake = by_id(&c, "cmake");
        assert_eq!(cmake.status, Status::Fail);
        assert!(cmake.detail.contains("3.20.6") && cmake.detail.contains("3.21.0"));
    }

    #[test]
    fn a_failing_smoke_test_blocks_and_carries_its_log() {
        let sys = healthy();
        let c = macos_checks(
            &sys,
            &ctx(Path::new("/tmp")),
            &Smoke(Err("ld: framework not found Metal".into())),
        );
        let smoke = by_id(&c, "smoke");
        assert_eq!(smoke.status, Status::Fail);
        assert!(smoke.evidence[0].contains("framework not found Metal"));
    }

    #[test]
    fn low_disk_space_fails() {
        let mut sys = healthy();
        sys.free = Some(300 * 1024 * 1024);
        let c = macos_checks(&sys, &ctx(Path::new("/tmp")), &Smoke(Ok(String::new())));
        assert_eq!(by_id(&c, "disk").status, Status::Fail);
    }

    #[test]
    fn metal_device_parses_the_real_capture_and_falls_back_to_the_chip() {
        let (name, family) = parse_metal_device(DISPLAYS).unwrap();
        assert_eq!(name, "Apple M5 Max");
        assert_eq!(family, "spdisplays_metal4");
        assert!(parse_metal_device("{}").is_none());
        assert!(parse_metal_device("not json").is_none());
        let sys =
            FakeSystem::new(TargetOs::Macos).on("sysctl -n machdep.cpu.brand_string", "Apple M2\n");
        assert_eq!(metal_device_check(&sys).status, Status::Pass);
        let none = FakeSystem::new(TargetOs::Macos);
        assert_eq!(metal_device_check(&none).status, Status::Warn);
    }

    #[test]
    fn the_brew_allowlist_is_enforced_server_side() {
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        assert_eq!(
            validate_brew_packages(&s(&["ninja", "cmake", "cmake"])).unwrap(),
            ["cmake", "ninja"]
        );
        assert_eq!(
            validate_brew_packages(&s(&["ccache", "git"])).unwrap(),
            ["git", "ccache"]
        );
        for bad in [
            s(&[]),
            s(&["wget"]),
            s(&["cmake", "git; rm -rf ~"]),
            s(&["--HEAD"]),
            s(&["cmake --cask"]),
            s(&["homebrew/core/cmake"]),
            s(&[""]),
        ] {
            assert!(validate_brew_packages(&bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn network_checks_run_only_when_asked_and_after_git_exists() {
        let sys = healthy()
            .on(
                "curl -sS -I -o /dev/null --max-time 15 https://github.com/ggml-org/llama.cpp",
                "",
            )
            .on(
                "curl -sS -I -o /dev/null --max-time 15 https://huggingface.co/ggml-org/llama-ui",
                "",
            );
        let mut context = ctx(Path::new("/tmp"));
        context.check_network = true;
        let c = macos_checks(&sys, &context, &Smoke(Ok(String::new())));
        assert_eq!(by_id(&c, "net_github").status, Status::Pass);
        assert_eq!(by_id(&c, "net_huggingface").status, Status::Pass);
        assert!(!c.checks.iter().any(|ch| ch.id == "net_cccl"));
    }

    #[test]
    fn the_fingerprint_changes_when_a_tool_version_changes() {
        let a = macos_checks(
            &healthy(),
            &ctx(Path::new("/tmp")),
            &Smoke(Ok(String::new())),
        );
        let b = macos_checks(
            &healthy().on("/opt/homebrew/bin/ninja --version", "1.14.0\n"),
            &ctx(Path::new("/tmp")),
            &Smoke(Ok(String::new())),
        );
        assert_ne!(a.fingerprint(), b.fingerprint());
    }
}
