//! Windows prerequisites (plan §5.2).
//!
//! Resolution is separate from judgement: [`windows_checks`] returns the
//! [`Collector`] of checks *and* the [`WindowsToolchain`] it resolved, which the
//! job layer uses to capture the MSVC environment and to build the CMake plan.
//! Paths are built with [`win_join`] so fixtures behave identically when the
//! tests run on macOS.

use std::path::{Path, PathBuf};
use std::time::Duration;

use super::gpu::{self, NvidiaGpu};
use super::prereqs::{
    CMAKE_MINIMUM, Check, Collector, DiskInputs, SmokeTest, Status, copy_fix, disk_check,
    driver_check_for_os, estimate_disk, link_fix, network_checks, parse_version, smoke_check,
    toolkit_version_check, version_string,
};
use super::sysprobe::SystemProbe;
use super::{Accelerator, CudaVersion};
use crate::llama_build::configure::CCCL_BUNDLED_FROM;

const QUERY_TIMEOUT: Duration = Duration::from_secs(20);

pub const VS_DOWNLOAD_URL: &str = "https://visualstudio.microsoft.com/downloads/";
pub const VS_WINGET: &str = "winget install Microsoft.VisualStudio.2022.BuildTools --override \
\"--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive\"";
pub const VS_CMAKE_COMPONENT: &str = "Microsoft.VisualStudio.Component.VC.CMake.Project";
pub const LONG_PATHS_REG: &str = "reg add HKLM\\SYSTEM\\CurrentControlSet\\Control\\FileSystem \
/v LongPathsEnabled /t REG_DWORD /d 1 /f";

/// The five `vcvarsall.bat` roots from the tooling's fallback list, used only
/// when `vswhere` is absent.
const KNOWN_VS_ROOTS: [&str; 5] = [
    r"C:\Program Files\Microsoft Visual Studio\2022\Community",
    r"C:\Program Files\Microsoft Visual Studio\2022\Professional",
    r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise",
    r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools",
    r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools",
];

pub fn win_join(base: &str, rest: &str) -> String {
    format!("{}\\{}", base.trim_end_matches(['\\', '/']), rest)
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WindowsToolchain {
    pub vs_install: Option<String>,
    pub vs_instance_id: Option<String>,
    pub vs_version: Option<String>,
    pub vcvars: Option<PathBuf>,
    pub cmake: Option<PathBuf>,
    pub ninja: Option<PathBuf>,
    /// `vs` when taken from the Visual Studio component, else `path`.
    pub cmake_origin: Option<&'static str>,
    pub git: Option<PathBuf>,
    pub cuda_root: Option<PathBuf>,
    pub nvcc: Option<PathBuf>,
    pub cuda_version: Option<CudaVersion>,
    /// Every toolkit found through `CUDA_PATH*`; only `CUDA_PATH` is used.
    pub other_toolkits: Vec<String>,
    pub gpus: Vec<NvidiaGpu>,
    pub cuda_architectures: Option<String>,
}

pub struct WinContext<'a> {
    pub app_root: &'a Path,
    pub disk: DiskInputs,
    /// Validated user override for `CMAKE_CUDA_ARCHITECTURES`.
    pub arch_override: Option<String>,
    pub check_network: bool,
}

/// `Cuda compilation tools, release 13.4, V13.4.92` → `13.4.92`.
pub fn parse_nvcc_version(text: &str) -> Option<CudaVersion> {
    let after_v = text.split(", V").nth(1)?;
    let token: String = after_v
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    CudaVersion::parse(&token)
}

// ---------------------------------------------------------------------------
// Resolution
// ---------------------------------------------------------------------------

struct VsInstall {
    path: String,
    instance_id: Option<String>,
    version: Option<String>,
}

fn find_vs(sys: &dyn SystemProbe, c: &mut Collector) -> Option<VsInstall> {
    let pf86 = sys
        .env_var("ProgramFiles(x86)")
        .unwrap_or_else(|| r"C:\Program Files (x86)".into());
    let vswhere = win_join(&pf86, r"Microsoft Visual Studio\Installer\vswhere.exe");
    if sys.path_exists(Path::new(&vswhere)) {
        let query = |prop: &str| -> Option<String> {
            let out = sys
                .run(
                    &vswhere,
                    &[
                        "-products",
                        "*",
                        "-version",
                        "[17.0,18.0)",
                        "-requires",
                        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                        "-property",
                        prop,
                    ],
                    QUERY_TIMEOUT,
                )
                .ok()?;
            out.ok()
                .then(|| out.stdout.lines().next().unwrap_or("").trim().to_string())
                .filter(|s| !s.is_empty())
        };
        if let Some(path) = query("installationPath") {
            let instance_id = query("instanceId");
            let version = query("installationVersion");
            c.fingerprint_part("vs_instance", instance_id.as_deref().unwrap_or(&path));
            return Some(VsInstall {
                path,
                instance_id,
                version,
            });
        }
    }
    for root in KNOWN_VS_ROOTS {
        let vcvars = win_join(root, r"VC\Auxiliary\Build\vcvars64.bat");
        if sys.path_exists(Path::new(&vcvars)) {
            c.fingerprint_part("vs_instance", root);
            return Some(VsInstall {
                path: root.to_string(),
                instance_id: None,
                version: None,
            });
        }
    }
    None
}

fn vs_checks(sys: &dyn SystemProbe, c: &mut Collector, tc: &mut WindowsToolchain) {
    const TITLE: &str = "Visual Studio 2022 Build Tools (C++)";
    let fix_copy = copy_fix(VS_WINGET);
    let Some(vs) = find_vs(sys, c) else {
        c.push(
            Check::fail(
                "vs",
                TITLE,
                "No Visual Studio 2022 install with the C++ x64 toolset was found.",
            )
            .with_fix(fix_copy),
        );
        c.push(
            Check::new(
                "vs_download",
                "Visual Studio download",
                Status::Skipped,
                "Or download the Build Tools installer and add the C++ workload.",
            )
            .with_fix(link_fix(VS_DOWNLOAD_URL, "Visual Studio downloads")),
        );
        return;
    };
    let vcvars = win_join(&vs.path, r"VC\Auxiliary\Build\vcvars64.bat");
    let have_vcvars = sys.path_exists(Path::new(&vcvars));
    tc.vs_install = Some(vs.path.clone());
    tc.vs_instance_id = vs.instance_id.clone();
    tc.vs_version = vs.version.clone();
    if have_vcvars {
        tc.vcvars = Some(PathBuf::from(&vcvars));
        c.push(
            Check::pass(
                "vs",
                TITLE,
                format!(
                    "{}{} at {}",
                    vs.version.as_deref().unwrap_or("version unknown"),
                    vs.instance_id
                        .as_deref()
                        .map(|i| format!(" (instance {i})"))
                        .unwrap_or_default(),
                    vs.path
                ),
            )
            .with_evidence(format!("vcvars64.bat: {vcvars}")),
        );
    } else {
        c.push(
            Check::fail(
                "vs",
                TITLE,
                format!(
                    "{} has no vcvars64.bat; the C++ workload is not installed.",
                    vs.path
                ),
            )
            .with_fix(fix_copy),
        );
    }

    // CMake and Ninja: the Visual Studio component first, PATH second.
    let vs_cmake = win_join(
        &vs.path,
        r"Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
    );
    let vs_ninja = win_join(
        &vs.path,
        r"Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe",
    );
    for (id, title, name, vs_path) in [
        ("cmake", "CMake", "cmake", vs_cmake),
        ("ninja", "Ninja", "ninja", vs_ninja),
    ] {
        let (path, origin) = if sys.path_exists(Path::new(&vs_path)) {
            (Some(PathBuf::from(&vs_path)), "vs")
        } else {
            (sys.which(name), "path")
        };
        let Some(path) = path else {
            c.push(
                Check::fail(id, title, format!("{name} was not found.")).with_fix(copy_fix(
                    &format!("Add the Visual Studio component {VS_CMAKE_COMPONENT}"),
                )),
            );
            continue;
        };
        let exe = path.to_string_lossy().into_owned();
        let out = sys.run(&exe, &["--version"], QUERY_TIMEOUT);
        let text = match &out {
            Ok(o) if o.ok() => o.combined(),
            Ok(o) => {
                c.push(Check::fail(
                    id,
                    title,
                    format!("{exe} did not run: {}", o.combined().trim()),
                ));
                continue;
            }
            Err(e) => {
                c.push(Check::fail(id, title, e.clone()));
                continue;
            }
        };
        let version = parse_version(&text);
        c.fingerprint_part(id, &format!("{exe}|{}", text.lines().next().unwrap_or("")));
        let too_old = id == "cmake" && version.is_some_and(|v| v < CMAKE_MINIMUM);
        let mut check = if too_old {
            Check::fail(
                id,
                title,
                format!(
                    "CMake {} is older than the required {}.",
                    version.map(version_string).unwrap_or_default(),
                    version_string(CMAKE_MINIMUM)
                ),
            )
        } else {
            Check::pass(
                id,
                title,
                format!(
                    "{} from {} at {exe}",
                    version
                        .map(version_string)
                        .unwrap_or_else(|| "version unknown".into()),
                    if origin == "vs" {
                        "the Visual Studio component"
                    } else {
                        "PATH (other copies may exist; this one is used)"
                    }
                ),
            )
        };
        check = check.with_evidence(format!("{exe} --version\n{}", text.trim()));
        c.push(check);
        if id == "cmake" {
            tc.cmake = Some(path);
            tc.cmake_origin = Some(if origin == "vs" { "vs" } else { "path" });
        } else {
            tc.ninja = Some(path);
        }
    }
}

fn cuda_checks(
    sys: &dyn SystemProbe,
    c: &mut Collector,
    tc: &mut WindowsToolchain,
    ctx: &WinContext,
) {
    // Every toolkit the installer registered, for the detail line.
    for key in [
        "CUDA_PATH_V12_4",
        "CUDA_PATH_V12_6",
        "CUDA_PATH_V12_8",
        "CUDA_PATH_V13_0",
        "CUDA_PATH_V13_1",
        "CUDA_PATH_V13_2",
        "CUDA_PATH_V13_3",
        "CUDA_PATH_V13_4",
        "CUDA_PATH_V13_5",
    ] {
        if let Some(v) = sys.env_var(key) {
            tc.other_toolkits.push(format!("{key}={v}"));
        }
    }
    let Some(root) = sys.env_var("CUDA_PATH") else {
        c.push(
            Check::fail(
                "cuda_toolkit",
                "CUDA Toolkit",
                "CUDA_PATH is not set. Install the CUDA Toolkit (the installer sets it) or set \
                 CUDA_PATH to the toolkit you want to build with.",
            )
            .with_fix(link_fix(
                "https://developer.nvidia.com/cuda-downloads",
                "NVIDIA CUDA downloads",
            )),
        );
        return;
    };
    let nvcc = win_join(&root, r"bin\nvcc.exe");
    let version = sys
        .run(&nvcc, &["--version"], QUERY_TIMEOUT)
        .ok()
        .filter(|o| o.ok())
        .and_then(|o| parse_nvcc_version(&o.stdout));
    let mut check = toolkit_version_check(version);
    check.detail = match version {
        Some(_) => format!(
            "{} — CUDA_PATH {root}{}",
            check.detail,
            if tc.other_toolkits.is_empty() {
                String::new()
            } else {
                format!(
                    "; other toolkits detected (not used): {}",
                    tc.other_toolkits.join(", ")
                )
            }
        ),
        None => format!("{} {nvcc} did not report a version.", check.detail),
    };
    c.fingerprint_part("cuda_path", &root);
    if let Some(v) = version {
        c.fingerprint_part("nvcc", &v.to_string());
        tc.cuda_root = Some(PathBuf::from(&root));
        tc.nvcc = Some(PathBuf::from(&nvcc));
        tc.cuda_version = Some(v);
    }
    c.push(check.with_evidence(format!("{nvcc} --version")));

    // GPU, architecture, driver.
    match gpu::detect_nvidia(sys) {
        Ok(gpus) => {
            let first = gpus[0].clone();
            tc.gpus = gpus;
            let arch = ctx
                .arch_override
                .clone()
                .or_else(|| first.cuda_architecture.clone());
            tc.cuda_architectures = arch.clone();
            let label = format!(
                "{} · compute {} · {} MiB",
                first.name, first.compute_cap, first.memory_mib
            );
            c.push(match (&ctx.arch_override, &first.cuda_architecture) {
                (Some(o), _) => Check::pass(
                    "gpu_arch",
                    "GPU compute capability",
                    format!("{label}; building for the architecture you chose: {o}"),
                ),
                (None, Some(a)) => Check::pass(
                    "gpu_arch",
                    "GPU compute capability",
                    format!("{label} → CMAKE_CUDA_ARCHITECTURES={a}"),
                ),
                (None, None) => Check::warn(
                    "gpu_arch",
                    "GPU compute capability",
                    format!(
                        "{label}: compute capability {} is not in the architecture map. \
                         Enter the architecture manually.",
                        first.compute_cap
                    ),
                ),
            });
            if let Some(v) = version {
                c.fingerprint_part("driver", first.driver_version.as_deref().unwrap_or(""));
                c.push(driver_check_for_os(
                    super::TargetOs::Windows,
                    v,
                    first.driver_version.as_deref(),
                    arch.as_deref(),
                ));
            }
        }
        Err(e) => {
            if let Some(o) = &ctx.arch_override {
                tc.cuda_architectures = Some(o.clone());
            }
            c.push(Check::new(
                "gpu_arch",
                "GPU compute capability",
                if ctx.arch_override.is_some() {
                    Status::Warn
                } else {
                    Status::Fail
                },
                format!("{e}. Enter the CUDA architecture manually to build anyway."),
            ));
        }
    }
}

fn git_check(sys: &dyn SystemProbe, c: &mut Collector, tc: &mut WindowsToolchain) {
    let program_files = sys
        .env_var("ProgramFiles")
        .unwrap_or_else(|| r"C:\Program Files".into());
    let fallback = win_join(&program_files, r"Git\cmd\git.exe");
    let found = sys.which("git").or_else(|| {
        sys.path_exists(Path::new(&fallback))
            .then(|| PathBuf::from(&fallback))
    });
    let Some(path) = found else {
        c.push(
            Check::fail("git", "git", "git was not found.")
                .with_fix(copy_fix("winget install Git.Git")),
        );
        return;
    };
    let exe = path.to_string_lossy().into_owned();
    match sys.run(&exe, &["--version"], QUERY_TIMEOUT) {
        Ok(o) if o.ok() => {
            c.fingerprint_part("git", o.stdout.trim());
            c.push(
                Check::pass("git", "git", format!("{} at {exe}", o.stdout.trim()))
                    .with_evidence(format!("{exe} --version")),
            );
            tc.git = Some(path);
        }
        other => c.push(Check::fail(
            "git",
            "git",
            format!("{exe} did not run: {other:?}"),
        )),
    }
}

fn long_paths_check(sys: &dyn SystemProbe) -> Check {
    let value = sys.registry_dword(
        r"SYSTEM\CurrentControlSet\Control\FileSystem",
        "LongPathsEnabled",
    );
    match value {
        Some(1) => Check::pass(
            "long_paths",
            "Long paths",
            "Windows long paths are enabled.",
        ),
        _ => Check::warn(
            "long_paths",
            "Long paths",
            "Windows long paths are not enabled. Deep CUDA build paths can exceed 260 characters.",
        )
        .with_fix(copy_fix(LONG_PATHS_REG)),
    }
}

/// Windows checks in panel order, plus the toolchain they resolved.
pub fn windows_checks(
    sys: &dyn SystemProbe,
    ctx: &WinContext,
    smoke: &dyn SmokeTest,
) -> (Collector, WindowsToolchain) {
    let mut c = Collector::default();
    let mut tc = WindowsToolchain::default();
    vs_checks(sys, &mut c, &mut tc);
    cuda_checks(sys, &mut c, &mut tc, ctx);
    git_check(sys, &mut c, &mut tc);

    let estimate = estimate_disk(&ctx.disk);
    c.push(disk_check(&estimate, sys.free_space(ctx.app_root)));
    c.push(long_paths_check(sys));

    if ctx.check_network && tc.git.is_some() {
        let need_cccl = tc.cuda_version.is_some_and(|v| v < CCCL_BUNDLED_FROM);
        for check in network_checks(sys, need_cccl) {
            c.push(check);
        }
    }

    let toolchain_ok = tc.vcvars.is_some()
        && tc.cmake.is_some()
        && tc.ninja.is_some()
        && tc.nvcc.is_some()
        && tc.git.is_some()
        && tc.cuda_architectures.is_some();
    c.push(if toolchain_ok {
        smoke_check(smoke.run(Accelerator::Cuda, Some(&tc)), true)
    } else {
        smoke_check(Err(String::new()), false)
    });
    (c, tc)
}

/// Recognized compiler errors from the CUDA smoke test, mapped to guidance.
pub fn smoke_hint(log: &str) -> Option<&'static str> {
    if log.contains("unsupported Microsoft Visual Studio version") {
        Some(
            "This CUDA Toolkit does not support the installed MSVC version. Install the MSVC toolset this CUDA version supports.",
        )
    } else if log.contains("No CUDA toolset found") {
        Some("CMake could not find the CUDA toolset. Re-run the prerequisites and check CUDA_PATH.")
    } else if log.contains("Unsupported gpu architecture") {
        Some(
            "nvcc does not support the chosen GPU architecture. Pick a lower architecture or update CUDA.",
        )
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama_build::TargetOs;
    use crate::llama_build::prereqs::Fix;
    use crate::llama_build::sysprobe::FakeSystem;

    const VSWHERE: &str = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe";
    const VS: &str = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools";
    const CUDA: &str = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4";
    const NVCC_BANNER: &str = "nvcc: NVIDIA (R) Cuda compiler driver\nCopyright (c) 2005-2026 NVIDIA Corporation\nBuilt on Sun_Sep_01_02:23:07_Pacific_Standard_Time_2026\nCuda compilation tools, release 13.4, V13.4.92\nBuild cuda_13.4.r13.4/compiler.38855100_0\n";
    const SMI_ARGS: &str =
        "--query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader";

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

    fn vswhere_cmd(prop: &str) -> String {
        format!(
            "{VSWHERE} -products * -version [17.0,18.0) -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property {prop}"
        )
    }

    /// ryne, as captured.
    fn ryne() -> FakeSystem {
        let vs_cmake =
            format!(r"{VS}\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe");
        let vs_ninja =
            format!(r"{VS}\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe");
        FakeSystem::new(TargetOs::Windows)
            .with_path(VSWHERE)
            .on(&vswhere_cmd("installationPath"), &format!("{VS}\r\n"))
            .on(&vswhere_cmd("instanceId"), "022d1522\r\n")
            .on(&vswhere_cmd("installationVersion"), "17.14.36717.8\r\n")
            .with_path(&format!(r"{VS}\VC\Auxiliary\Build\vcvars64.bat"))
            .with_program("cmake", r"C:\Program Files\CMake\bin\cmake.exe")
            .with_program("ninja", r"C:\Strawberry\c\bin\ninja.exe")
            .with_path(&vs_cmake)
            .with_path(&vs_ninja)
            .on(
                &format!("{vs_cmake} --version"),
                "cmake version 3.31.6-msvc6\n\nCMake suite maintained and supported by Kitware\n",
            )
            .on(&format!("{vs_ninja} --version"), "1.12.1\n")
            .with_env("CUDA_PATH", CUDA)
            .with_env(
                "CUDA_PATH_V12_8",
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8",
            )
            .with_env(
                "CUDA_PATH_V13_3",
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3",
            )
            .with_env("CUDA_PATH_V13_4", CUDA)
            .on(&format!(r"{CUDA}\bin\nvcc.exe --version"), NVCC_BANNER)
            .with_program("nvidia-smi", r"C:\Windows\System32\nvidia-smi.exe")
            .on(
                &format!(r"C:\Windows\System32\nvidia-smi.exe {SMI_ARGS}"),
                "NVIDIA GeForce RTX 5090, 12.0, 32607 MiB, 617.42\n",
            )
            .with_program("git", r"C:\Program Files\Git\cmd\git.exe")
            .on(
                r"C:\Program Files\Git\cmd\git.exe --version",
                "git version 2.52.0.windows.1\n",
            )
            .with_registry(
                r"SYSTEM\CurrentControlSet\Control\FileSystem",
                "LongPathsEnabled",
                1,
            )
    }

    fn ctx(root: &Path) -> WinContext<'_> {
        WinContext {
            app_root: root,
            disk: DiskInputs {
                os: TargetOs::Windows,
                accelerator: Accelerator::Cuda,
                checkout_present: false,
                build_tree_present: false,
                ccache_enabled: false,
                ccache_current_bytes: 0,
                ccache_cap_bytes: 2 << 30,
            },
            arch_override: None,
            check_network: false,
        }
    }

    fn by_id<'a>(c: &'a Collector, id: &str) -> &'a Check {
        c.checks.iter().find(|ch| ch.id == id).unwrap_or_else(|| {
            panic!(
                "no {id}: {:?}",
                c.checks.iter().map(|c| &c.id).collect::<Vec<_>>()
            )
        })
    }

    #[test]
    fn ryne_resolves_and_passes() {
        let (c, tc) = windows_checks(&ryne(), &ctx(Path::new(r"C:\app")), &Smoke(Ok("ok".into())));
        for id in [
            "vs",
            "cmake",
            "ninja",
            "cuda_toolkit",
            "gpu_arch",
            "nvidia_driver",
            "git",
            "disk",
            "long_paths",
            "smoke",
        ] {
            assert_eq!(
                by_id(&c, id).status,
                Status::Pass,
                "{id}: {}",
                by_id(&c, id).detail
            );
        }
        assert_eq!(tc.vs_instance_id.as_deref(), Some("022d1522"));
        assert_eq!(tc.cuda_version.unwrap().to_string(), "13.4.92");
        assert_eq!(tc.cuda_architectures.as_deref(), Some("120a-real"));
        assert_eq!(tc.cmake_origin, Some("vs"));
        assert!(
            tc.cmake
                .unwrap()
                .to_string_lossy()
                .contains("Visual Studio")
        );
        assert!(
            tc.ninja
                .unwrap()
                .to_string_lossy()
                .contains("Visual Studio")
        );
        assert_eq!(
            tc.nvcc.unwrap().to_string_lossy(),
            format!(r"{CUDA}\bin\nvcc.exe")
        );
    }

    #[test]
    fn the_visual_studio_cmake_wins_over_other_copies_on_path() {
        let (c, _) = windows_checks(
            &ryne(),
            &ctx(Path::new(r"C:\app")),
            &Smoke(Ok(String::new())),
        );
        let cmake = by_id(&c, "cmake");
        assert!(
            cmake.detail.contains("Visual Studio component"),
            "{}",
            cmake.detail
        );
        assert!(!cmake.detail.contains("Strawberry"));
    }

    #[test]
    fn path_cmake_is_used_and_labelled_when_the_vs_component_is_absent() {
        let mut sys = ryne();
        let vs_cmake =
            format!(r"{VS}\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe");
        sys.existing.remove(Path::new(&vs_cmake));
        let sys = sys.on(
            r"C:\Program Files\CMake\bin\cmake.exe --version",
            "cmake version 4.1.2\n",
        );
        let (c, tc) = windows_checks(&sys, &ctx(Path::new(r"C:\app")), &Smoke(Ok(String::new())));
        assert!(by_id(&c, "cmake").detail.contains("PATH"));
        assert_eq!(tc.cmake_origin, Some("path"));
    }

    #[test]
    fn every_detected_toolkit_is_shown_but_only_cuda_path_is_used() {
        let (c, tc) = windows_checks(
            &ryne(),
            &ctx(Path::new(r"C:\app")),
            &Smoke(Ok(String::new())),
        );
        let detail = &by_id(&c, "cuda_toolkit").detail;
        assert!(
            detail.contains("CUDA_PATH_V12_8") && detail.contains("CUDA_PATH_V13_3"),
            "{detail}"
        );
        assert!(detail.contains("not used"));
        assert_eq!(tc.cuda_root.unwrap().to_string_lossy(), CUDA);
    }

    #[test]
    fn missing_visual_studio_fails_with_winget_and_a_link() {
        let sys = FakeSystem::new(TargetOs::Windows);
        let (c, tc) = windows_checks(&sys, &ctx(Path::new(r"C:\app")), &Smoke(Ok(String::new())));
        let vs = by_id(&c, "vs");
        assert_eq!(vs.status, Status::Fail);
        assert!(
            matches!(&vs.fix, Some(Fix::Copy { command }) if command.starts_with("winget install Microsoft.VisualStudio.2022.BuildTools"))
        );
        assert!(matches!(
            by_id(&c, "vs_download").fix,
            Some(Fix::Link { .. })
        ));
        assert!(tc.vcvars.is_none());
        assert_eq!(by_id(&c, "smoke").status, Status::Skipped);
    }

    #[test]
    fn without_vswhere_the_known_install_roots_are_searched() {
        let root = r"C:\Program Files\Microsoft Visual Studio\2022\Community";
        let sys = FakeSystem::new(TargetOs::Windows)
            .with_path(&format!(r"{root}\VC\Auxiliary\Build\vcvars64.bat"));
        let (_, tc) = windows_checks(&sys, &ctx(Path::new(r"C:\app")), &Smoke(Ok(String::new())));
        assert_eq!(tc.vs_install.as_deref(), Some(root));
        assert!(tc.vs_instance_id.is_none());
    }

    #[test]
    fn a_vs_install_without_the_cpp_workload_fails() {
        let mut sys = ryne();
        sys.existing
            .remove(Path::new(&format!(r"{VS}\VC\Auxiliary\Build\vcvars64.bat")));
        let (c, tc) = windows_checks(&sys, &ctx(Path::new(r"C:\app")), &Smoke(Ok(String::new())));
        assert_eq!(by_id(&c, "vs").status, Status::Fail);
        assert!(tc.vcvars.is_none());
    }

    #[test]
    fn cuda_path_unset_fails_without_guessing_a_toolkit() {
        let mut sys = ryne();
        sys.env.remove("CUDA_PATH");
        let (c, tc) = windows_checks(&sys, &ctx(Path::new(r"C:\app")), &Smoke(Ok(String::new())));
        let cuda = by_id(&c, "cuda_toolkit");
        assert_eq!(cuda.status, Status::Fail);
        assert!(cuda.detail.contains("CUDA_PATH"));
        assert!(tc.nvcc.is_none());
    }

    #[test]
    fn old_and_untested_toolkits() {
        let old = ryne().on(
            &format!(r"{CUDA}\bin\nvcc.exe --version"),
            "Cuda compilation tools, release 12.3, V12.3.107\n",
        );
        let (c, _) = windows_checks(&old, &ctx(Path::new(r"C:\app")), &Smoke(Ok(String::new())));
        assert_eq!(by_id(&c, "cuda_toolkit").status, Status::Fail);
        let twelve = ryne().on(
            &format!(r"{CUDA}\bin\nvcc.exe --version"),
            "Cuda compilation tools, release 12.8, V12.8.93\n",
        );
        let (c, _) = windows_checks(
            &twelve,
            &ctx(Path::new(r"C:\app")),
            &Smoke(Ok(String::new())),
        );
        assert_eq!(by_id(&c, "cuda_toolkit").status, Status::Warn);
    }

    #[test]
    fn nvcc_versions_parse() {
        assert_eq!(
            parse_nvcc_version(NVCC_BANNER).unwrap().to_string(),
            "13.4.92"
        );
        assert!(parse_nvcc_version("nvcc: command not found").is_none());
    }

    #[test]
    fn an_old_driver_fails_prerequisites_with_the_documented_minimum() {
        let sys = ryne().on(
            &format!(r"C:\Windows\System32\nvidia-smi.exe {SMI_ARGS}"),
            "NVIDIA GeForce RTX 5090, 12.0, 32607 MiB, 560.94\n",
        );
        let (c, _) = windows_checks(&sys, &ctx(Path::new(r"C:\app")), &Smoke(Ok(String::new())));
        let driver = by_id(&c, "nvidia_driver");
        assert_eq!(driver.status, Status::Fail);
        assert!(driver.detail.contains("580"));
    }

    #[test]
    fn an_unmapped_gpu_warns_and_needs_a_manual_architecture() {
        let sys = ryne().on(
            &format!(r"C:\Windows\System32\nvidia-smi.exe {SMI_ARGS}"),
            "Future GPU, 14.0, 65536 MiB, 700.00\n",
        );
        let (c, tc) = windows_checks(&sys, &ctx(Path::new(r"C:\app")), &Smoke(Ok(String::new())));
        assert_eq!(by_id(&c, "gpu_arch").status, Status::Warn);
        assert!(tc.cuda_architectures.is_none());
        // Without an architecture there is nothing to smoke-test against.
        assert_eq!(by_id(&c, "smoke").status, Status::Skipped);
        // A manual choice unblocks it.
        let mut with = ctx(Path::new(r"C:\app"));
        with.arch_override = Some("120a-real".into());
        let (c, tc) = windows_checks(&sys, &with, &Smoke(Ok(String::new())));
        assert_eq!(tc.cuda_architectures.as_deref(), Some("120a-real"));
        assert_eq!(by_id(&c, "smoke").status, Status::Pass);
    }

    #[test]
    fn no_gpu_fails_unless_an_architecture_was_chosen() {
        let mut sys = ryne();
        sys.programs.remove("nvidia-smi");
        let (c, _) = windows_checks(&sys, &ctx(Path::new(r"C:\app")), &Smoke(Ok(String::new())));
        assert_eq!(by_id(&c, "gpu_arch").status, Status::Fail);
        let mut with = ctx(Path::new(r"C:\app"));
        with.arch_override = Some("89".into());
        let (c, _) = windows_checks(&sys, &with, &Smoke(Ok(String::new())));
        assert_eq!(by_id(&c, "gpu_arch").status, Status::Warn);
    }

    #[test]
    fn git_falls_back_to_program_files_and_fails_cleanly() {
        let mut sys = ryne();
        sys.programs.remove("git");
        sys.existing
            .remove(Path::new(r"C:\Program Files\Git\cmd\git.exe"));
        let (c, tc) = windows_checks(&sys, &ctx(Path::new(r"C:\app")), &Smoke(Ok(String::new())));
        // Not on PATH and the Git\cmd fallback is not registered as existing.
        assert_eq!(by_id(&c, "git").status, Status::Fail);
        assert!(tc.git.is_none());
        let sys = sys.with_path(r"C:\Program Files\Git\cmd\git.exe");
        let (c, tc) = windows_checks(&sys, &ctx(Path::new(r"C:\app")), &Smoke(Ok(String::new())));
        assert_eq!(by_id(&c, "git").status, Status::Pass);
        assert!(tc.git.is_some());
    }

    #[test]
    fn long_paths_only_warn() {
        let mut sys = ryne();
        sys.registry.clear();
        let (c, _) = windows_checks(&sys, &ctx(Path::new(r"C:\app")), &Smoke(Ok(String::new())));
        let lp = by_id(&c, "long_paths");
        assert_eq!(lp.status, Status::Warn);
        assert!(
            matches!(&lp.fix, Some(Fix::Copy { command }) if command.contains("LongPathsEnabled"))
        );
        assert!(!c.checks.iter().any(|ch| ch.status == Status::Fail));
    }

    #[test]
    fn twenty_gib_floor_blocks_a_small_disk() {
        let mut sys = ryne();
        sys.free = Some(15 * (1 << 30));
        let (c, _) = windows_checks(&sys, &ctx(Path::new(r"C:\app")), &Smoke(Ok(String::new())));
        assert_eq!(by_id(&c, "disk").status, Status::Fail);
    }

    #[test]
    fn cccl_reachability_is_checked_only_for_toolkits_that_fetch_it() {
        let curl = |url: &str| format!("curl -sS -I -o NUL --max-time 15 {url}");
        let sys = ryne()
            .on(&curl("https://github.com/ggml-org/llama.cpp"), "")
            .on(&curl("https://huggingface.co/ggml-org/llama-ui"), "")
            .on(&curl("https://github.com/nvidia/cccl"), "");
        let mut context = ctx(Path::new(r"C:\app"));
        context.check_network = true;
        let (c, _) = windows_checks(&sys, &context, &Smoke(Ok(String::new())));
        // 13.4 < 13.5: CCCL is fetched at configure time.
        assert_eq!(by_id(&c, "net_cccl").status, Status::Pass);
        let bundled = sys.on(
            &format!(r"{CUDA}\bin\nvcc.exe --version"),
            "Cuda compilation tools, release 13.5, V13.5.10\n",
        );
        let (c, _) = windows_checks(&bundled, &context, &Smoke(Ok(String::new())));
        assert!(!c.checks.iter().any(|ch| ch.id == "net_cccl"));
    }

    #[test]
    fn smoke_failures_are_mapped_to_guidance() {
        assert!(
            smoke_hint("nvcc : fatal error: unsupported Microsoft Visual Studio version!")
                .unwrap()
                .contains("MSVC")
        );
        assert!(
            smoke_hint("CMake Error: No CUDA toolset found.")
                .unwrap()
                .contains("CUDA_PATH")
        );
        assert!(
            smoke_hint("nvcc fatal   : Unsupported gpu architecture 'compute_130'")
                .unwrap()
                .contains("architecture")
        );
        assert!(smoke_hint("something else").is_none());
    }

    #[test]
    fn win_join_never_doubles_separators() {
        assert_eq!(win_join(r"C:\a\", "b"), r"C:\a\b");
        assert_eq!(win_join(r"C:\a", r"b\c"), r"C:\a\b\c");
    }

    #[test]
    fn fingerprint_tracks_vs_instance_cuda_and_driver() {
        let (a, _) = windows_checks(
            &ryne(),
            &ctx(Path::new(r"C:\app")),
            &Smoke(Ok(String::new())),
        );
        let driver_bump = ryne().on(
            &format!(r"C:\Windows\System32\nvidia-smi.exe {SMI_ARGS}"),
            "NVIDIA GeForce RTX 5090, 12.0, 32607 MiB, 620.00\n",
        );
        let (b, _) = windows_checks(
            &driver_bump,
            &ctx(Path::new(r"C:\app")),
            &Smoke(Ok(String::new())),
        );
        assert_ne!(a.fingerprint(), b.fingerprint());
        let (a2, _) = windows_checks(
            &ryne(),
            &ctx(Path::new(r"C:\app")),
            &Smoke(Ok(String::new())),
        );
        assert_eq!(a.fingerprint(), a2.fingerprint());
    }
}
