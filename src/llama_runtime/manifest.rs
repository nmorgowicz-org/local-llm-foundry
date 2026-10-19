//! `bin/llama-build-manifest.json`: provenance for the installed runtime.
//!
//! The manifest records where a runtime came from and the SHA-256 of every
//! shipped file. It is **provenance only**: it never authorizes a capability
//! (plan §4.6). Compiled K/V pairs and the loaded backend come from the
//! runtime probe, bound to the runtime closure.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const MANIFEST_FILE: &str = "llama-build-manifest.json";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ManifestSource {
    UpstreamPrebuilt,
    FoundrySourceBuild,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlamaCppRef {
    #[serde(default, rename = "ref")]
    pub reference: Option<String>,
    #[serde(default)]
    pub commit: Option<String>,
    #[serde(default)]
    pub build_number: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeManifest {
    pub schema: u32,
    pub source: ManifestSource,
    pub label: String,
    pub llama_cpp: LlamaCppRef,
    #[serde(default)]
    pub accelerator: Option<String>,
    /// Source builds only (Phase 4): exact CMake arguments and toolchain.
    #[serde(default)]
    pub cmake_args: Vec<String>,
    #[serde(default)]
    pub fa_quants_spec: Option<String>,
    /// Source builds: the pairs the configure recorded. A record of what was
    /// asked for, never read as a capability.
    #[serde(default)]
    pub fa_pairs_compiled: Vec<(String, String)>,
    #[serde(default)]
    pub cuda_architectures: Option<String>,
    #[serde(default)]
    pub toolchain: BTreeMap<String, String>,
    pub built_at_unix: u64,
    /// `file name -> sha256 hex`, every file in the set except the manifest.
    pub files: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManifestCheck {
    Missing,
    Unreadable(String),
    /// Every recorded file is present with the recorded hash and nothing extra
    /// ships beside them.
    Matches,
    /// Provenance no longer describes this directory (treated as custom).
    Mismatch(Vec<String>),
}

fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 256 * 1024];
    loop {
        let n = std::io::Read::read(&mut file, &mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

fn hash_files(dir: &Path) -> std::io::Result<BTreeMap<String, String>> {
    let mut files = BTreeMap::new();
    let canonical_dir = std::fs::canonicalize(dir)?;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name == MANIFEST_FILE {
            continue;
        }
        let kind = entry.file_type()?;
        if kind.is_symlink() {
            let target = std::fs::canonicalize(entry.path())?;
            if target.parent() != Some(canonical_dir.as_path()) || !target.is_file() {
                return Err(std::io::Error::other(
                    "manifest symlink escapes the runtime set",
                ));
            }
        } else if !kind.is_file() {
            continue;
        }
        files.insert(name, sha256_file(&entry.path())?);
    }
    Ok(files)
}

impl RuntimeManifest {
    pub fn for_directory(
        dir: &Path,
        source: ManifestSource,
        label: impl Into<String>,
        llama_cpp: LlamaCppRef,
        accelerator: Option<String>,
    ) -> std::io::Result<Self> {
        Ok(Self {
            schema: 1,
            source,
            label: label.into(),
            llama_cpp,
            accelerator,
            cmake_args: Vec::new(),
            fa_quants_spec: None,
            fa_pairs_compiled: Vec::new(),
            cuda_architectures: None,
            toolchain: BTreeMap::new(),
            built_at_unix: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            files: hash_files(dir)?,
        })
    }

    pub fn write(&self, dir: &Path) -> std::io::Result<()> {
        let json = serde_json::to_vec_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(dir.join(MANIFEST_FILE), json)
    }

    pub fn read(dir: &Path) -> Result<Option<Self>, String> {
        match std::fs::read(dir.join(MANIFEST_FILE)) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map(Some)
                .map_err(|e| e.to_string()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.to_string()),
        }
    }
}

/// Compare a directory with its manifest. A mismatch marks provenance
/// `custom`; it never changes what the runtime probe reports.
pub fn check_manifest(dir: &Path) -> ManifestCheck {
    let manifest = match RuntimeManifest::read(dir) {
        Ok(Some(manifest)) => manifest,
        Ok(None) => return ManifestCheck::Missing,
        Err(e) => return ManifestCheck::Unreadable(e),
    };
    let actual = match hash_files(dir) {
        Ok(files) => files,
        Err(e) => return ManifestCheck::Unreadable(e.to_string()),
    };
    let mut differences = Vec::new();
    for (name, hash) in &manifest.files {
        match actual.get(name) {
            None => differences.push(format!("{name}: missing")),
            Some(found) if found != hash => differences.push(format!("{name}: hash changed")),
            Some(_) => {}
        }
    }
    for name in actual.keys() {
        if !manifest.files.contains_key(name) {
            differences.push(format!("{name}: not in manifest"));
        }
    }
    if differences.is_empty() {
        ManifestCheck::Matches
    } else {
        ManifestCheck::Mismatch(differences)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(dir: &Path) -> RuntimeManifest {
        std::fs::write(dir.join("llama-server"), b"exe").unwrap();
        std::fs::write(dir.join("libggml.dylib"), b"lib").unwrap();
        RuntimeManifest::for_directory(
            dir,
            ManifestSource::UpstreamPrebuilt,
            "b11727 prebuilt",
            LlamaCppRef {
                reference: Some("b11727".into()),
                commit: None,
                build_number: Some(11727),
            },
            Some("metal".into()),
        )
        .unwrap()
    }

    #[test]
    fn round_trips_and_matches_its_own_directory() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = sample(dir.path());
        manifest.write(dir.path()).unwrap();
        assert_eq!(RuntimeManifest::read(dir.path()).unwrap(), Some(manifest));
        assert_eq!(check_manifest(dir.path()), ManifestCheck::Matches);
        // The manifest never lists itself.
        assert!(
            !RuntimeManifest::read(dir.path())
                .unwrap()
                .unwrap()
                .files
                .contains_key(MANIFEST_FILE)
        );
    }

    #[test]
    fn edited_missing_or_extra_files_mark_provenance_as_mismatched() {
        let dir = tempfile::tempdir().unwrap();
        sample(dir.path()).write(dir.path()).unwrap();
        std::fs::write(dir.path().join("llama-server"), b"patched").unwrap();
        std::fs::remove_file(dir.path().join("libggml.dylib")).unwrap();
        std::fs::write(dir.path().join("extra.so"), b"x").unwrap();
        let ManifestCheck::Mismatch(diffs) = check_manifest(dir.path()) else {
            panic!("expected mismatch");
        };
        assert!(
            diffs
                .iter()
                .any(|d| d.contains("llama-server: hash changed"))
        );
        assert!(diffs.iter().any(|d| d.contains("libggml.dylib: missing")));
        assert!(
            diffs
                .iter()
                .any(|d| d.contains("extra.so: not in manifest"))
        );
    }

    #[test]
    fn missing_and_corrupt_manifests_are_distinct_and_never_an_error_for_the_runtime() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(check_manifest(dir.path()), ManifestCheck::Missing);
        std::fs::write(dir.path().join(MANIFEST_FILE), b"{nope").unwrap();
        assert!(matches!(
            check_manifest(dir.path()),
            ManifestCheck::Unreadable(_)
        ));
    }
}

#[cfg(all(test, unix))]
mod symlink_tests {
    use super::*;

    #[test]
    fn shipped_symlinks_are_hashed_and_escaping_links_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("libggml.1.dylib"), "backend").unwrap();
        std::os::unix::fs::symlink("libggml.1.dylib", dir.path().join("libggml.dylib")).unwrap();
        let hashes = hash_files(dir.path()).unwrap();
        assert_eq!(hashes["libggml.1.dylib"], hashes["libggml.dylib"]);
        let outside = tempfile::NamedTempFile::new().unwrap();
        std::os::unix::fs::symlink(outside.path(), dir.path().join("escape.dylib")).unwrap();
        assert!(hash_files(dir.path()).is_err());
    }
}
