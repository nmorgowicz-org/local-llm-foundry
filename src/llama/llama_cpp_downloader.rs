use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::fs;
use tokio::io::AsyncWriteExt;

fn backend_matches(name: &str, backend: &str) -> bool {
    match backend {
        "cpu" => {
            let accelerator_qualified = [
                "cuda", "vulkan", "rocm", "hip", "sycl", "oneapi", "openvino", "opencl", "metal",
            ]
            .iter()
            .any(|marker| name.contains(marker));
            !accelerator_qualified
                && (name.contains("cpu")
                    || name.contains("base")
                    || name.contains("avx2")
                    || name.contains("ubuntu")
                    || name.contains("linux"))
        }
        "avx2" => name.contains("avx2") || name.contains("cpu") || name.contains("base"),
        "cuda" => name.contains("cuda"),
        "cuda12" => {
            name.contains("cuda12")
                || name.contains("cuda-12")
                || name.contains("cu12")
                || name.contains("cuda_12")
        }
        "cuda13" => {
            name.contains("cuda13")
                || name.contains("cuda-13")
                || name.contains("cu13")
                || name.contains("cuda_13")
        }
        "sycl" => name.contains("sycl") || name.contains("oneapi"),
        "openvino" => name.contains("openvino"),
        "vulkan" => name.contains("vulkan"),
        "rocm" | "hip" => name.contains("rocm") || name.contains("hip"),
        "metal" => name.contains("metal") || name.contains("mac"),
        _ => false,
    }
}

/// Metadata for a llama.cpp GitHub release.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaCppRelease {
    pub tag_name: String,
    pub assets: Vec<LlamaCppAsset>,
    #[serde(default)]
    pub published_at: String,
    #[serde(default)]
    pub body: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaCppAsset {
    pub name: String,
    pub browser_download_url: String,
}

/// Query parameter for the single-release endpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct ReleaseQuery {
    /// Build number (e.g. 9479 for b9479).
    pub build: u64,
}

/// Status of an in-progress llama.cpp download.
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize)]
pub struct LlamaCppDownloadStatus {
    pub download_id: String,
    pub status: String,
    pub message: String,
    pub progress: f64,
}

/// True when a GitHub release asset is a downloadable binary archive rather
/// than release metadata (`.json`/`.md`/`.txt` files such as `nightly-tag.txt`).
pub fn is_binary_asset_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    !lower.ends_with(".json") && !lower.ends_with(".md") && !lower.ends_with(".txt")
}

/// True when the release publishes at least one binary archive, as opposed to
/// metadata-only versioned releases (e.g. `v0.2.0` ships only `nightly-tag.txt`).
pub fn release_has_binary_assets(release: &LlamaCppRelease) -> bool {
    release
        .assets
        .iter()
        .any(|asset| is_binary_asset_name(&asset.name))
}

/// Fetch a specific release from ggml-org/llama.cpp by tag (e.g. "b9479").
pub async fn get_release_by_tag(client: &Client, tag: &str) -> Result<LlamaCppRelease> {
    let url = format!(
        "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{}",
        tag
    );
    let resp = client
        .get(&url)
        .send()
        .await
        .context(format!("Failed to fetch release for tag {}", tag))?;

    if !resp.status().is_success() {
        anyhow::bail!(
            "GitHub API returned {} for release tag {}",
            resp.status(),
            tag
        );
    }

    let release: LlamaCppRelease = resp
        .json()
        .await
        .context(format!("Failed to parse release JSON for tag {}", tag))?;
    Ok(release)
}

/// Fetch the latest non-prerelease (versioned/stable) llama.cpp release.
///
/// Nightly builds are published as prereleases, so GitHub's `latest` endpoint
/// skips them and returns the most recent versioned tag (e.g. `v0.2.0`).
pub async fn get_latest_stable_release(client: &Client) -> Result<LlamaCppRelease> {
    let url = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest";
    let resp = client
        .get(url)
        .send()
        .await
        .context("Failed to fetch latest stable llama.cpp release")?;

    if !resp.status().is_success() {
        anyhow::bail!(
            "GitHub API returned {} for latest stable release",
            resp.status()
        );
    }

    let release: LlamaCppRelease = resp
        .json()
        .await
        .context("Failed to parse latest stable release JSON")?;
    Ok(release)
}

/// List recent releases from ggml-org/llama.cpp.
pub async fn list_releases(client: &Client) -> Result<Vec<LlamaCppRelease>> {
    let url = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=20";
    let resp = client
        .get(url)
        .send()
        .await
        .context("Failed to fetch llama.cpp releases")?;

    if !resp.status().is_success() {
        anyhow::bail!("GitHub API returned {} for releases", resp.status());
    }

    let releases: Vec<LlamaCppRelease> =
        resp.json().await.context("Failed to parse releases JSON")?;
    Ok(releases)
}

/// Sort releases newest-first by their GitHub publication timestamp.
///
/// The GitHub releases endpoint does not make its response order a suitable
/// chronological contract, especially when stable tags and build tags are
/// published close together. Empty timestamps sort last; tag names provide a
/// deterministic tie-breaker.
pub fn sort_releases_by_published_at(releases: &mut [LlamaCppRelease]) {
    releases.sort_by(|left, right| {
        right
            .published_at
            .cmp(&left.published_at)
            .then_with(|| right.tag_name.cmp(&left.tag_name))
    });
}

/// Select appropriate assets for the given platform/backend.
///
/// This is a best-effort heuristic based on known release asset naming conventions.
pub fn select_assets<'a>(
    release: &'a LlamaCppRelease,
    backend: &str,
    arch: &str,
) -> Vec<&'a LlamaCppAsset> {
    let mut selected = Vec::new();

    let backend_lower = backend.to_lowercase();
    let arch_lower = arch.to_lowercase();

    for asset in &release.assets {
        let name = asset.name.to_lowercase();

        // Skip non-binary assets.
        if !is_binary_asset_name(&asset.name) {
            continue;
        }

        // Match by backend and architecture.
        let backend_match = backend_matches(&name, backend_lower.as_str());

        let arch_match = match arch_lower.as_str() {
            "x64" | "x86_64" => {
                name.contains("x64") || name.contains("x86_64") || name.contains("amd64")
            }
            "arm64" | "aarch64" => name.contains("arm64") || name.contains("aarch64"),
            _ => true,
        };

        if backend_match && arch_match {
            selected.push(asset);
        }
    }

    selected
}

/// Select assets for a specific host operating system, backend, and
/// architecture. This is the install-path selector: unlike `select_assets`,
/// it excludes otherwise matching archives for another operating system.
pub fn select_assets_for_os<'a>(
    release: &'a LlamaCppRelease,
    backend: &str,
    arch: &str,
    os: &str,
) -> Vec<&'a LlamaCppAsset> {
    select_assets(release, backend, arch)
        .into_iter()
        .filter(|asset| {
            let name = asset.name.to_ascii_lowercase();
            match os {
                "macos" => name.contains("macos") || name.contains("darwin"),
                "linux" => name.contains("ubuntu") || name.contains("linux"),
                "windows" => name.contains("win-") || name.contains("windows"),
                _ => true,
            }
        })
        .collect()
}

/// Download and extract llama.cpp assets into binaries_dir.
///
/// For simplicity, this downloads each selected asset into binaries_dir
/// and attempts to extract .zip/.tar.gz if applicable.
pub async fn download_and_extract(
    client: &Client,
    _release: &LlamaCppRelease,
    assets: &[&LlamaCppAsset],
    binaries_dir: &Path,
) -> Result<()> {
    fs::create_dir_all(binaries_dir)
        .await
        .context("Failed to create binaries_dir")?;

    let mut progress = HashMap::new();
    for asset in assets {
        let out_path = binaries_dir.join(&asset.name);
        let tmp_path = out_path.with_extension("part");
        let result: Result<()> = async {
            // Download
            let resp = client
                .get(&asset.browser_download_url)
                .send()
                .await
                .context(format!("Failed to start download for {}", asset.name))?;

            if !resp.status().is_success() {
                anyhow::bail!("Download failed for {}: HTTP {}", asset.name, resp.status());
            }

            let total = resp.content_length().unwrap_or(0);
            let mut downloaded: u64 = 0;
            let mut file = fs::File::create(&tmp_path)
                .await
                .context("Failed to create temp file")?;

            let mut stream = resp.bytes_stream();
            use futures_util::StreamExt;
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.context("Stream error")?;
                file.write_all(&chunk)
                    .await
                    .context("Failed to write chunk")?;
                downloaded += chunk.len() as u64;
                if total > 0 {
                    progress.insert(asset.name.clone(), (downloaded as f64) / (total as f64));
                }
            }

            // Move temp -> final
            fs::rename(&tmp_path, &out_path)
                .await
                .context("Failed to finalize download file")?;

            // Extract if archive
            if asset.name.ends_with(".zip")
                || asset.name.ends_with(".tar.gz")
                || asset.name.ends_with(".tgz")
            {
                let _ = extract_archive(&out_path, binaries_dir)
                    .await
                    .inspect_err(|e| {
                        eprintln!("[warn] Failed to extract {}: {}", asset.name, e);
                    });
            }
            Ok(())
        }
        .await;

        if let Err(error) = result {
            let _ = fs::remove_file(&tmp_path).await;
            return Err(error);
        }
    }

    // Run cleanup after each install.
    if let Err(e) = cleanup_old_binaries(binaries_dir).await {
        eprintln!("[warn] llama.cpp binary cleanup failed: {}", e);
    }

    Ok(())
}

async fn extract_archive(path: &Path, dest: &Path) -> Result<()> {
    if path.extension().is_some_and(|e| e == "zip") {
        use std::io::Read;
        use zip::ZipArchive;
        let file = std::fs::File::open(path)?;
        let mut archive = ZipArchive::new(std::io::BufReader::new(file))?;
        for i in 0..archive.len() {
            let mut file = archive.by_index(i)?;
            let outpath = dest.join(file.enclosed_name().unwrap_or_else(|| PathBuf::from("out")));
            if file.is_file() {
                fs::create_dir_all(outpath.parent().unwrap_or(dest))
                    .await
                    .ok();
                let mut out_file = fs::File::create(&outpath).await?;
                let mut buf = Vec::new();
                file.read_to_end(&mut buf)?;
                out_file.write_all(&buf).await?;
            }
        }
    } else {
        let bytes = tokio::fs::read(path).await?;
        let tar_gz = std::io::Cursor::new(bytes);
        let decompressed = flate2::read::GzDecoder::new(tar_gz);
        let mut archive = tar::Archive::new(decompressed);
        archive.unpack(dest)?;
    }
    Ok(())
}

/// Clean up old llama.cpp artifacts in the binaries directory.
///
/// This is run automatically after each new download/extract.
///
/// Behavior:
/// - Keeps current binaries (llama-server, llama-bench, etc.), symlinks, and non-versioned libs.
/// - For llama-<build>-bin-*.tar.gz: keep only the latest build; remove others.
/// - For versioned dylibs like libllama.0.0.<build>.dylib:
///   keep only the highest build number in each family.
pub async fn cleanup_old_binaries(binaries_dir: &Path) -> Result<()> {
    use std::collections::HashMap;

    let mut entries = fs::read_dir(binaries_dir).await?;
    let mut files = Vec::new();
    while let Ok(Some(entry)) = entries.next_entry().await {
        files.push(entry);
    }

    // Track tarballs by build number: llama-<digits>-bin-*.tar.gz
    let mut tarball_map: HashMap<u64, PathBuf> = HashMap::new();
    // Track versioned dylibs by family prefix
    let mut versioned_map: HashMap<String, Vec<(PathBuf, u64)>> = HashMap::new();

    for entry in &files {
        if entry.file_type().await?.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        let lower = name.to_lowercase();

        // Handle tarballs: llama-b<digits>-bin-*.tar.gz (GitHub release naming)
        if lower.starts_with("llama-") && lower.ends_with(".tar.gz") && lower.contains("-bin-") {
            // GitHub names them "llama-b9637-bin-..." so left marker is "llama-b".
            if let Ok(build) = digits_between(&lower, "llama-b", "-bin-") {
                tarball_map.insert(build, entry.path());
            }
            continue;
        }

        // Handle versioned dylibs: <prefix><digits>.dylib
        if let Some(build) = extract_trailing_build_number(&lower) {
            let prefix = &lower[..lower.len() - build.to_string().len() - 6];
            versioned_map
                .entry(prefix.to_string())
                .or_default()
                .push((entry.path(), build));
        }
    }

    let mut removed = 0usize;

    // Tarballs: keep only the highest build.
    if tarball_map.len() > 1
        && let Some(max_build) = tarball_map.keys().cloned().max()
    {
        for (&build, path) in &tarball_map {
            if build != max_build {
                if let Err(e) = fs::remove_file(path).await {
                    eprintln!("[warn] failed to remove old tarball {:?}: {}", path, e);
                } else {
                    removed += 1;
                }
            }
        }
    }

    // Versioned dylibs: for each family, keep only highest build.
    for (_prefix, variants) in versioned_map {
        if variants.len() <= 1 {
            continue;
        }
        let max_build = variants.iter().map(|(_, b)| b).max().copied().unwrap_or(0);
        let to_remove: Vec<_> = variants
            .iter()
            .filter(|(_, b)| b != &max_build)
            .map(|(p, _)| p.clone())
            .collect();
        for path in to_remove {
            if let Err(e) = fs::remove_file(&path).await {
                eprintln!("[warn] failed to remove old dylib {:?}: {}", path, e);
            } else {
                removed += 1;
            }
        }
    }

    if removed > 0 {
        eprintln!(
            "[local-llm-foundry] cleaned up {} old llama.cpp artifact(s) in {:?}",
            removed, binaries_dir
        );
    }

    Ok(())
}

fn digits_between(s: &str, left: &str, right: &str) -> Result<u64> {
    let start = s
        .find(left)
        .map(|i| i + left.len())
        .ok_or_else(|| anyhow::anyhow!("missing left marker"))?;
    let end = s[start..]
        .find(right)
        .map(|i| i + start)
        .ok_or_else(|| anyhow::anyhow!("missing right marker"))?;
    let digits = &s[start..end];
    digits
        .parse::<u64>()
        .map_err(|_| anyhow::anyhow!("bad digits"))
}

fn extract_trailing_build_number(name: &str) -> Option<u64> {
    // Match: <something>.<digits>.dylib (e.g. libllama.0.0.9559.dylib)
    if !name.ends_with(".dylib") {
        return None;
    }
    let before_ext = &name[..name.len() - 6]; // strip ".dylib"
    let parts: Vec<&str> = before_ext.rsplitn(2, '.').collect();
    if parts.len() == 2 && parts[0].chars().all(|c| c.is_ascii_digit()) {
        return parts[0].parse::<u64>().ok();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn releases_are_sorted_by_publication_time_not_api_order() {
        let mut releases = vec![
            LlamaCppRelease {
                tag_name: "b10549".into(),
                assets: Vec::new(),
                published_at: "2026-08-21T10:00:00Z".into(),
                body: String::new(),
            },
            LlamaCppRelease {
                tag_name: "v0.2.0".into(),
                assets: Vec::new(),
                published_at: "2026-08-21T11:00:00Z".into(),
                body: String::new(),
            },
            LlamaCppRelease {
                tag_name: "b10567".into(),
                assets: Vec::new(),
                published_at: "2026-08-21T11:30:00Z".into(),
                body: String::new(),
            },
        ];

        sort_releases_by_published_at(&mut releases);

        assert_eq!(
            releases
                .iter()
                .map(|release| release.tag_name.as_str())
                .collect::<Vec<_>>(),
            vec!["b10567", "v0.2.0", "b10549"]
        );
    }

    #[test]
    fn test_release_has_binary_assets_distinguishes_metadata_only_releases() {
        let stable = LlamaCppRelease {
            tag_name: "v0.2.0".into(),
            assets: vec![LlamaCppAsset {
                name: "nightly-tag.txt".into(),
                browser_download_url: "https://example.com/nightly-tag.txt".into(),
            }],
            published_at: String::new(),
            body: String::new(),
        };
        assert!(!release_has_binary_assets(&stable));

        let nightly = LlamaCppRelease {
            tag_name: "b6000".into(),
            assets: vec![LlamaCppAsset {
                name: "llama-b6000-bin-ubuntu-x64.tar.gz".into(),
                browser_download_url: "https://example.com/linux".into(),
            }],
            published_at: String::new(),
            body: String::new(),
        };
        assert!(release_has_binary_assets(&nightly));
    }

    #[test]
    fn test_select_assets_cpu_x64() {
        let release = LlamaCppRelease {
            tag_name: "b6000".into(),
            assets: vec![
                LlamaCppAsset {
                    name: "llama-server-win-x64-cpu.exe".into(),
                    browser_download_url: "https://example.com/cpu".into(),
                },
                LlamaCppAsset {
                    name: "llama-server-win-x64-cuda12.zip".into(),
                    browser_download_url: "https://example.com/cuda".into(),
                },
            ],
            published_at: String::new(),
            body: String::new(),
        };
        let selected = select_assets(&release, "cpu", "x64");
        assert_eq!(selected.len(), 1);
    }

    #[test]
    fn test_select_assets_ignores_metadata_only_release() {
        let release = LlamaCppRelease {
            tag_name: "v0.2.0".into(),
            assets: vec![LlamaCppAsset {
                name: "nightly-tag.txt".into(),
                browser_download_url: "https://example.com/nightly-tag.txt".into(),
            }],
            published_at: String::new(),
            body: String::new(),
        };

        assert!(select_assets(&release, "metal", "arm64").is_empty());
    }

    #[test]
    fn test_select_assets_cpu_linux_x64() {
        let release = LlamaCppRelease {
            tag_name: "b6000".into(),
            assets: vec![LlamaCppAsset {
                name: "llama-b6000-bin-ubuntu-x64.tar.gz".into(),
                browser_download_url: "https://example.com/linux".into(),
            }],
            published_at: String::new(),
            body: String::new(),
        };

        let selected = select_assets(&release, "cpu", "x64");
        assert_eq!(selected.len(), 1);
    }

    #[test]
    fn test_select_assets_for_os_excludes_other_platform_archives() {
        let release = LlamaCppRelease {
            tag_name: "b6000".into(),
            assets: vec![
                LlamaCppAsset {
                    name: "llama-b6000-bin-ubuntu-x64.tar.gz".into(),
                    browser_download_url: "https://example.com/linux".into(),
                },
                LlamaCppAsset {
                    name: "llama-b6000-bin-win-cpu-x64.zip".into(),
                    browser_download_url: "https://example.com/windows".into(),
                },
            ],
            published_at: String::new(),
            body: String::new(),
        };

        let selected = select_assets_for_os(&release, "cpu", "x64", "linux");
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].name, "llama-b6000-bin-ubuntu-x64.tar.gz");
    }

    #[test]
    fn test_select_assets_cpu_linux_excludes_accelerated_archives() {
        let release = LlamaCppRelease {
            tag_name: "b6000".into(),
            assets: vec![
                LlamaCppAsset {
                    name: "llama-b6000-bin-ubuntu-x64.tar.gz".into(),
                    browser_download_url: "https://example.com/cpu".into(),
                },
                LlamaCppAsset {
                    name: "llama-b6000-bin-ubuntu-vulkan-x64.tar.gz".into(),
                    browser_download_url: "https://example.com/vulkan".into(),
                },
                LlamaCppAsset {
                    name: "llama-b6000-bin-ubuntu-rocm-x64.tar.gz".into(),
                    browser_download_url: "https://example.com/rocm".into(),
                },
                LlamaCppAsset {
                    name: "llama-b6000-bin-ubuntu-sycl-x64.tar.gz".into(),
                    browser_download_url: "https://example.com/sycl".into(),
                },
                LlamaCppAsset {
                    name: "llama-b6000-bin-ubuntu-openvino-x64.tar.gz".into(),
                    browser_download_url: "https://example.com/openvino".into(),
                },
            ],
            published_at: String::new(),
            body: String::new(),
        };

        let selected = select_assets(&release, "cpu", "x64");
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].name, "llama-b6000-bin-ubuntu-x64.tar.gz");
    }

    #[test]
    fn test_select_assets_openvino_linux_x64() {
        let release = LlamaCppRelease {
            tag_name: "b6000".into(),
            assets: vec![
                LlamaCppAsset {
                    name: "llama-b6000-bin-ubuntu-openvino-2026.3-x64.tar.gz".into(),
                    browser_download_url: "https://example.com/openvino-linux".into(),
                },
                LlamaCppAsset {
                    name: "llama-b6000-bin-ubuntu-x64.tar.gz".into(),
                    browser_download_url: "https://example.com/cpu".into(),
                },
            ],
            published_at: String::new(),
            body: String::new(),
        };

        let selected = select_assets(&release, "openvino", "x64");
        assert_eq!(selected.len(), 1);
        assert_eq!(
            selected[0].name,
            "llama-b6000-bin-ubuntu-openvino-2026.3-x64.tar.gz"
        );
    }

    #[test]
    fn test_select_assets_cuda_x64() {
        let release = LlamaCppRelease {
            tag_name: "b6000".into(),
            assets: vec![
                LlamaCppAsset {
                    name: "llama-server-win-x64-cpu.exe".into(),
                    browser_download_url: "https://example.com/cpu".into(),
                },
                LlamaCppAsset {
                    name: "llama-server-win-x64-cuda12.zip".into(),
                    browser_download_url: "https://example.com/cuda".into(),
                },
            ],
            published_at: String::new(),
            body: String::new(),
        };
        let selected = select_assets(&release, "cuda", "x64");
        assert_eq!(selected.len(), 1);
    }

    #[test]
    fn test_select_assets_avx2_does_not_fall_back_to_everything() {
        let release = LlamaCppRelease {
            tag_name: "b7000".into(),
            assets: vec![
                LlamaCppAsset {
                    name: "llama-server-win-x64-avx2.zip".into(),
                    browser_download_url: "https://example.com/avx2".into(),
                },
                LlamaCppAsset {
                    name: "llama-server-win-x64-cuda12.zip".into(),
                    browser_download_url: "https://example.com/cuda12".into(),
                },
                LlamaCppAsset {
                    name: "llama-server-win-x64-sycl.zip".into(),
                    browser_download_url: "https://example.com/sycl".into(),
                },
            ],
            published_at: String::new(),
            body: String::new(),
        };
        let selected = select_assets(&release, "avx2", "x64");
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].name, "llama-server-win-x64-avx2.zip");
    }

    #[test]
    fn test_select_assets_cuda12_matches_only_cuda12() {
        let release = LlamaCppRelease {
            tag_name: "b7000".into(),
            assets: vec![
                LlamaCppAsset {
                    name: "llama-server-win-x64-cuda12.zip".into(),
                    browser_download_url: "https://example.com/cuda12".into(),
                },
                LlamaCppAsset {
                    name: "llama-server-win-x64-cuda13.zip".into(),
                    browser_download_url: "https://example.com/cuda13".into(),
                },
            ],
            published_at: String::new(),
            body: String::new(),
        };
        let selected = select_assets(&release, "cuda12", "x64");
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].name, "llama-server-win-x64-cuda12.zip");
    }

    #[test]
    fn test_select_assets_sycl_matches_only_sycl() {
        let release = LlamaCppRelease {
            tag_name: "b7000".into(),
            assets: vec![
                LlamaCppAsset {
                    name: "llama-server-win-x64-sycl.zip".into(),
                    browser_download_url: "https://example.com/sycl".into(),
                },
                LlamaCppAsset {
                    name: "llama-server-win-x64-vulkan.zip".into(),
                    browser_download_url: "https://example.com/vulkan".into(),
                },
            ],
            published_at: String::new(),
            body: String::new(),
        };
        let selected = select_assets(&release, "sycl", "x64");
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].name, "llama-server-win-x64-sycl.zip");
    }
}
