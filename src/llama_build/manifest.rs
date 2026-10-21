//! Provenance for a source build (plan §4.6).
//!
//! This builds a [`RuntimeManifest`] from what the job knows: the pinned
//! commit, the exact CMake arguments, the toolchain and what the configure
//! recorded. It is **provenance only**. Nothing here, and nothing that reads a
//! manifest, authorizes a capability; the runtime probe does, bound to the
//! installed closure (`llama_runtime::probe`).

use std::collections::BTreeMap;
use std::path::Path;

use crate::inference::kv_support::KvType;
use crate::llama_runtime::manifest::{LlamaCppRef, ManifestSource, RuntimeManifest};

use super::Accelerator;
use super::configure::ConfigureEvidence;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceBuildInfo {
    /// `b11727` for a tag; `master` for a branch build.
    pub reference: String,
    /// Full 40-character commit the job pinned.
    pub commit: String,
    pub build_number: Option<u32>,
    pub accelerator: Accelerator,
    pub cmake_args: Vec<String>,
    pub toolchain: BTreeMap<String, String>,
    pub evidence: ConfigureEvidence,
}

fn kv_name(t: KvType) -> String {
    // Serialized names are the lowercase wire names (`q8_0`, `f16`, ...).
    serde_json::to_value(t)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_else(|| format!("{t:?}").to_ascii_lowercase())
}

/// `b11727 · src · FA all`, `master@098ce52 · src · Metal`.
pub fn build_label(info: &SourceBuildInfo) -> String {
    let id = match info.reference.as_str() {
        "master" => format!("master@{}", &info.commit[..info.commit.len().min(7)]),
        other => other.to_string(),
    };
    let tail = match (info.accelerator, info.evidence.fa_quants_spec.as_deref()) {
        (Accelerator::Cuda, Some(spec)) => format!("FA {spec}"),
        (accel, _) => accel.as_str().to_string(),
    };
    format!("{id} · src · {tail}")
}

/// Hash the staged directory and attach the build record. Call after staging is
/// complete: the file list is the set that will ship.
pub fn source_manifest(dir: &Path, info: &SourceBuildInfo) -> std::io::Result<RuntimeManifest> {
    let mut manifest = RuntimeManifest::for_directory(
        dir,
        ManifestSource::FoundrySourceBuild,
        build_label(info),
        LlamaCppRef {
            reference: Some(info.reference.clone()),
            commit: Some(info.commit.clone()),
            build_number: info.build_number,
        },
        Some(info.accelerator.as_str().to_string()),
    )?;
    manifest.cmake_args = info.cmake_args.clone();
    // Upstream caches CUDA defaults even when GGML_CUDA is OFF. Those
    // defaults are not a compiled-pair claim for Metal/CPU builds.
    if info.accelerator == Accelerator::Cuda {
        manifest.fa_quants_spec = info.evidence.fa_quants_spec.clone();
        manifest.fa_pairs_compiled = info
            .evidence
            .fa_pairs
            .iter()
            .flatten()
            .map(|(k, v)| (kv_name(*k), kv_name(*v)))
            .collect();
        manifest.cuda_architectures = info.evidence.cuda_architectures.clone();
    }
    manifest.toolchain = info.toolchain.clone();
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama_build::configure::configure_evidence;
    use crate::llama_runtime::manifest::{ManifestCheck, check_manifest};

    fn info(accel: Accelerator, cache: &str) -> SourceBuildInfo {
        SourceBuildInfo {
            reference: "b11727".into(),
            commit: "098ce526b0000000000000000000000000000000".into(),
            build_number: Some(11727),
            accelerator: accel,
            cmake_args: vec!["-DGGML_CUDA=ON".into()],
            toolchain: BTreeMap::from([("cmake".into(), "4.1.2".into())]),
            evidence: configure_evidence(cache).unwrap(),
        }
    }

    const CUDA_CACHE: &str =
        "GGML_CUDA_FA_QUANTS:STRING=all\nCMAKE_CUDA_ARCHITECTURES:STRING=120a-real\n";

    fn staged(dir: &Path) {
        std::fs::write(dir.join("llama-server.exe"), b"exe").unwrap();
        std::fs::write(dir.join("ggml-cuda.dll"), b"dll").unwrap();
    }

    #[test]
    fn unused_cuda_cache_defaults_do_not_claim_metal_compiled_pairs() {
        let dir = tempfile::tempdir().unwrap();
        staged(dir.path());
        for accelerator in [Accelerator::Metal, Accelerator::Cpu] {
            let manifest = source_manifest(dir.path(), &info(accelerator, CUDA_CACHE)).unwrap();
            assert!(manifest.fa_quants_spec.is_none());
            assert!(manifest.fa_pairs_compiled.is_empty());
            assert!(manifest.cuda_architectures.is_none());
        }
    }

    #[test]
    fn records_the_full_build_provenance() {
        let dir = tempfile::tempdir().unwrap();
        staged(dir.path());
        let m = source_manifest(dir.path(), &info(Accelerator::Cuda, CUDA_CACHE)).unwrap();
        assert_eq!(m.source, ManifestSource::FoundrySourceBuild);
        assert_eq!(m.label, "b11727 · src · FA all");
        assert_eq!(m.llama_cpp.commit.as_deref().map(str::len), Some(40));
        assert_eq!(m.accelerator.as_deref(), Some("cuda"));
        assert_eq!(m.fa_quants_spec.as_deref(), Some("all"));
        assert_eq!(m.fa_pairs_compiled.len(), 49);
        assert!(
            m.fa_pairs_compiled
                .contains(&("q8_0".into(), "q4_0".into()))
        );
        assert_eq!(m.cuda_architectures.as_deref(), Some("120a-real"));
        assert_eq!(m.cmake_args, ["-DGGML_CUDA=ON"]);
        assert_eq!(m.files.len(), 2);
    }

    #[test]
    fn round_trips_and_hash_mismatch_marks_it_custom() {
        let dir = tempfile::tempdir().unwrap();
        staged(dir.path());
        let m = source_manifest(dir.path(), &info(Accelerator::Cuda, CUDA_CACHE)).unwrap();
        m.write(dir.path()).unwrap();
        assert_eq!(RuntimeManifest::read(dir.path()).unwrap(), Some(m));
        assert_eq!(check_manifest(dir.path()), ManifestCheck::Matches);
        std::fs::write(dir.path().join("ggml-cuda.dll"), b"tampered").unwrap();
        assert!(matches!(
            check_manifest(dir.path()),
            ManifestCheck::Mismatch(d) if d.iter().any(|l| l.contains("ggml-cuda.dll"))
        ));
    }

    #[test]
    fn master_labels_carry_the_short_commit() {
        let mut i = info(Accelerator::Metal, "");
        i.reference = "master".into();
        assert_eq!(build_label(&i), "master@098ce52 · src · metal");
    }

    #[test]
    fn a_non_cuda_manifest_carries_no_fa_claim() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("llama-server"), b"x").unwrap();
        let m = source_manifest(dir.path(), &info(Accelerator::Metal, "")).unwrap();
        assert!(m.fa_quants_spec.is_none());
        assert!(m.fa_pairs_compiled.is_empty());
    }

    /// The manifest says `all`, the probe says nothing: the verifier must still
    /// reject. A manifest is never an input to `evaluate_probe`.
    #[test]
    fn a_manifest_claiming_fa_all_does_not_satisfy_the_verifier() {
        use crate::llama_runtime::probe::{RuntimeProbe, kv_basis_from_probe, parse_probe_output};
        use crate::llama_runtime::verify::{
            InstallKind, VerificationIntent, VerifyError, evaluate_probe,
        };
        let dir = tempfile::tempdir().unwrap();
        staged(dir.path());
        let m = source_manifest(dir.path(), &info(Accelerator::Cuda, CUDA_CACHE)).unwrap();
        m.write(dir.path()).unwrap();
        assert_eq!(m.fa_quants_spec.as_deref(), Some("all"));

        let parsed = parse_probe_output(
            &format!(
                "load_backend: loaded CUDA backend from {}\n",
                dir.path().join("ggml-cuda.dll").display()
            ),
            "/nonexistent/m.gguf",
        );
        let basis = kv_basis_from_probe(&parsed, None);
        let probe = RuntimeProbe {
            parsed,
            identity: None,
            basis,
        };
        let intent = VerificationIntent {
            kind: InstallKind::Source,
            expected_backend: None,
            pinned_commit: None,
            require_fa_all: true,
            device_id: None,
        };
        assert!(matches!(
            evaluate_probe(&probe, dir.path(), &intent),
            Err(VerifyError::FaQuantsNotAll(_))
        ));
    }
}
