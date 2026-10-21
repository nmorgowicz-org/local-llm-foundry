//! Known-failure hints (plan §8).
//!
//! The job matches the tail of a failed phase's log against this table and
//! attaches a [`Hint`]. Every pattern has a fixture log in
//! `tests/fixtures/llama_build/`; an unknown failure simply gets no hint (the
//! full tail and log path are still shown).

use serde::{Deserialize, Serialize};

use super::prereqs::{Fix, action_fix, copy_fix, link_fix};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hint {
    pub id: String,
    pub message: String,
    pub fix: Option<Fix>,
}

fn hint(id: &str, message: impl Into<String>, fix: Option<Fix>) -> Hint {
    Hint {
        id: id.into(),
        message: message.into(),
        fix,
    }
}

/// What the matcher may know beyond the log text.
#[derive(Debug, Clone, Default)]
pub struct HintContext {
    /// Cached CMake paths that no longer exist after the configure.
    pub stale_cache_paths: Vec<(String, String)>,
    pub free_space_bytes: Option<u64>,
    /// The auto-heal (`cmake -U`) already ran for this configure.
    pub healed: bool,
}

fn any(lines: &[String], needles: &[&str]) -> bool {
    lines.iter().any(|l| needles.iter().any(|n| l.contains(n)))
}

fn first_url(lines: &[String]) -> Option<String> {
    for line in lines {
        let mut rest = line.as_str();
        while let Some(at) = rest.find("http") {
            let tail = &rest[at..];
            if tail.starts_with("http://") || tail.starts_with("https://") {
                let end = tail
                    .find(|c: char| c.is_whitespace() || c == '\'' || c == '"' || c == ')')
                    .unwrap_or(tail.len());
                return Some(tail[..end].to_string());
            }
            rest = &tail[4..];
        }
    }
    None
}

/// Lines that look like the failure itself, for `BuildFailure::first_error`.
pub fn first_error(lines: &[String]) -> Option<String> {
    const MARKERS: [&str; 9] = [
        "CMake Error",
        "fatal error",
        "FAILED:",
        "error:",
        "LNK",
        "fatal:",
        "nvcc fatal",
        "No space left",
        "UI: HF download failed",
    ];
    lines
        .iter()
        .find(|l| MARKERS.iter().any(|m| l.contains(m)))
        .map(|l| l.trim().chars().take(300).collect())
}

/// The first matching hint, most specific first.
pub fn match_hint(lines: &[String], ctx: &HintContext) -> Option<Hint> {
    if any(lines, &["unsupported Microsoft Visual Studio version"]) {
        return Some(hint(
            "msvc_unsupported",
            "This CUDA Toolkit does not support the installed MSVC version. Install the MSVC \
             toolset that this CUDA version supports.",
            Some(link_fix(
                "https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/",
                "CUDA on Windows: supported compilers",
            )),
        ));
    }
    if any(lines, &["No CUDA toolset found", "Failed to find nvcc"]) {
        return Some(hint(
            "no_cuda_toolset",
            "CMake could not find the CUDA toolset. Re-run the prerequisites and check that \
             CUDA_PATH points at the toolkit you want.",
            Some(action_fix("revalidate", "Re-run prerequisites")),
        ));
    }
    if any(lines, &["Unsupported gpu architecture"]) {
        return Some(hint(
            "nvcc_arch",
            "nvcc does not support the chosen GPU architecture. Pick a lower architecture in the \
             build options, or update CUDA.",
            None,
        ));
    }
    if any(lines, &["xcrun: error: invalid active developer path"]) {
        return Some(hint(
            "xcrun_path",
            "The Xcode Command Line Tools are missing or need reinstalling.",
            Some(copy_fix("xcode-select --install")),
        ));
    }
    if any(lines, &["ui-assets.cmake"]) && any(lines, &["Unable to read from file", "Can't lstat"])
    {
        return Some(hint(
            "ui_assets_symlink",
            "The build path went through a symlink, which breaks the web UI asset step. The \
             foundry canonicalizes build paths, so this is a bug; wiping the build directory \
             works around it.",
            Some(action_fix(
                "wipe_build_dir",
                "Wipe build directory and retry",
            )),
        ));
    }
    if any(lines, &["UI: HF download failed"]) {
        return Some(hint(
            "network_huggingface",
            "The build downloads the web UI assets from huggingface.co. Check network or proxy \
             access to huggingface.co and retry.",
            Some(link_fix(
                "https://huggingface.co/ggml-org/llama-ui",
                "huggingface.co/ggml-org/llama-ui",
            )),
        ));
    }
    if any(lines, &["Failed to clone repository"]) && any(lines, &["cccl"]) {
        return Some(hint(
            "network_cccl",
            "CMake fetches NVIDIA CCCL from github.com while configuring. Check network or proxy \
             access to github.com and retry.",
            Some(link_fix(
                "https://github.com/nvidia/cccl",
                "github.com/nvidia/cccl",
            )),
        ));
    }
    if any(lines, &["fatal: unable to access"]) {
        let url = first_url(lines).unwrap_or_else(|| "the remote".into());
        return Some(hint(
            "git_network",
            format!("git could not reach {url}. Check network or proxy settings and retry."),
            None,
        ));
    }
    if any(
        lines,
        &[
            "No space left on device",
            "There is not enough space on the disk",
            "insufficient disk space",
            "errno=28",
        ],
    ) {
        let free = ctx
            .free_space_bytes
            .map(|b| format!(" {:.1} GiB are free.", b as f64 / (1u64 << 30) as f64))
            .unwrap_or_default();
        return Some(hint(
            "disk_full",
            format!(
                "The disk filled up during the build.{free} Clean the build cache to free space; \
                 installed runtimes are never removed."
            ),
            Some(action_fix("clean_build_cache", "Clean build cache")),
        ));
    }
    if any(lines, &["LNK1104"]) {
        return Some(hint(
            "lnk1104",
            "A process holds an output file open. The build writes outside the live runtime, so \
             this should not happen; close anything using the build directory and retry.",
            None,
        ));
    }
    if any(
        lines,
        &[
            "Killed: 9",
            "Killed signal terminated program",
            "Error 137",
            "exit code 137",
            "out of memory",
            "std::bad_alloc",
        ],
    ) {
        return Some(hint(
            "oom",
            "The compiler was killed, most likely for lack of memory. Retry with fewer parallel \
             jobs (\"Leave headroom\").",
            Some(action_fix("leave_headroom", "Retry with fewer jobs")),
        ));
    }
    if !ctx.stale_cache_paths.is_empty() || any(lines, &["but the target was not found"]) {
        let listed = ctx
            .stale_cache_paths
            .iter()
            .map(|(v, p)| format!("{v} = {p}"))
            .collect::<Vec<_>>()
            .join("; ");
        let tail = if ctx.healed {
            "The stale entries were already purged for this configure, so the cause may be \
             something else."
        } else {
            "A stale CMake cache is the usual cause."
        };
        return Some(hint(
            "stale_cache",
            if listed.is_empty() {
                tail.to_string()
            } else {
                format!("{tail} These cached paths no longer exist: {listed}.")
            },
            Some(action_fix(
                "wipe_build_dir",
                "Wipe build directory and retry",
            )),
        ));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(text: &str) -> Vec<String> {
        text.lines().map(str::to_string).collect()
    }

    macro_rules! fixture {
        ($name:literal) => {
            lines(include_str!(concat!(
                "../../tests/fixtures/llama_build/",
                $name
            )))
        };
    }

    fn id_of(log: &[String], ctx: &HintContext) -> Option<String> {
        match_hint(log, ctx).map(|h| h.id)
    }

    #[test]
    fn every_catalog_pattern_has_a_fixture_and_matches() {
        let none = HintContext::default();
        for (log, want) in [
            (fixture!("msvc_unsupported.log"), "msvc_unsupported"),
            (fixture!("no_cuda_toolset.log"), "no_cuda_toolset"),
            (fixture!("nvcc_arch.log"), "nvcc_arch"),
            (fixture!("git_network.log"), "git_network"),
            (fixture!("disk_full.log"), "disk_full"),
            (fixture!("disk_full_windows.log"), "disk_full"),
            (fixture!("lnk1104.log"), "lnk1104"),
            (fixture!("xcrun_path.log"), "xcrun_path"),
            (fixture!("ui_assets_symlink.log"), "ui_assets_symlink"),
            (fixture!("ui_download_failed.log"), "network_huggingface"),
            (fixture!("cccl_fetch_failed.log"), "network_cccl"),
            (fixture!("oom_killed.log"), "oom"),
            (fixture!("oom_137.log"), "oom"),
            (fixture!("stale_cache_openssl.log"), "stale_cache"),
        ] {
            assert_eq!(id_of(&log, &none).as_deref(), Some(want), "{want}");
        }
    }

    #[test]
    fn an_unknown_failure_gets_no_hint_but_still_a_first_error() {
        let log = fixture!("unknown_failure.log");
        assert_eq!(id_of(&log, &HintContext::default()), None);
        let first = first_error(&log).unwrap();
        assert!(first.starts_with("FAILED:"), "{first}");
    }

    #[test]
    fn stale_cache_paths_from_the_detector_trigger_the_hint_and_are_listed() {
        let ctx = HintContext {
            stale_cache_paths: vec![(
                "OPENSSL_ROOT_DIR".into(),
                "/opt/homebrew/Cellar/openssl@3/3.5.0".into(),
            )],
            healed: true,
            ..Default::default()
        };
        let h = match_hint(&lines("some unrelated failure"), &ctx).unwrap();
        assert_eq!(h.id, "stale_cache");
        assert!(h.message.contains("OPENSSL_ROOT_DIR") && h.message.contains("already purged"));
        assert!(matches!(h.fix, Some(Fix::Action { ref id, .. }) if id == "wipe_build_dir"));
    }

    #[test]
    fn the_symlink_bug_is_not_confused_with_a_generic_cmake_error() {
        let h = match_hint(&fixture!("ui_assets_symlink.log"), &HintContext::default()).unwrap();
        assert!(h.message.contains("symlink"));
        // The same words without ui-assets.cmake do not match.
        assert_eq!(
            id_of(
                &lines("Unable to read from file x"),
                &HintContext::default()
            ),
            None
        );
    }

    #[test]
    fn git_hints_name_the_failing_url() {
        let h = match_hint(&fixture!("git_network.log"), &HintContext::default()).unwrap();
        assert!(
            h.message.contains("https://github.com/ggml-org/llama.cpp/"),
            "{}",
            h.message
        );
    }

    #[test]
    fn disk_hints_show_free_space_and_never_offer_removing_a_runtime() {
        let ctx = HintContext {
            free_space_bytes: Some(3 << 30),
            ..Default::default()
        };
        let h = match_hint(&fixture!("disk_full.log"), &ctx).unwrap();
        assert!(h.message.contains("3.0 GiB are free"));
        assert!(h.message.contains("never removed"));
        assert!(matches!(h.fix, Some(Fix::Action { ref id, .. }) if id == "clean_build_cache"));
    }

    #[test]
    fn oom_offers_the_headroom_action() {
        let h = match_hint(&fixture!("oom_killed.log"), &HintContext::default()).unwrap();
        assert!(matches!(h.fix, Some(Fix::Action { ref id, .. }) if id == "leave_headroom"));
    }

    #[test]
    fn specific_hints_win_over_the_generic_stale_cache_one() {
        let ctx = HintContext {
            stale_cache_paths: vec![("X".into(), "/gone".into())],
            ..Default::default()
        };
        assert_eq!(
            id_of(&fixture!("disk_full.log"), &ctx).as_deref(),
            Some("disk_full")
        );
        assert_eq!(
            id_of(&fixture!("xcrun_path.log"), &ctx).as_deref(),
            Some("xcrun_path")
        );
    }

    #[test]
    fn first_error_skips_noise() {
        let log =
            lines("[1/9] Building x\n-- configuring\nCMake Error at a.cmake:1 (message):\n  bad\n");
        assert_eq!(
            first_error(&log).as_deref(),
            Some("CMake Error at a.cmake:1 (message):")
        );
        assert_eq!(first_error(&lines("all good")), None);
    }
}
