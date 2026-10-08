//! Cache inspection for Rapid-MLX model sources.
//!
//! The launch environment pins `HF_HUB_CACHE` to `<models_dir>/cache/huggingface/hub`
//! (see `model_resolver::hugging_face_environment`). A model is only "downloaded" for
//! launch purposes when it is complete inside that hub, regardless of copies that exist
//! in the user's global HuggingFace cache.

use std::path::{Path, PathBuf};

/// Hub directory the launch environment points `HF_HUB_CACHE` at.
pub fn app_hub_dir(models_dir: &Path) -> PathBuf {
    models_dir.join("cache/huggingface/hub")
}

/// The user's global HuggingFace hub cache (`HF_HUB_CACHE`, `HF_HOME`, or `~/.cache`).
pub fn system_hub_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("HF_HUB_CACHE").filter(|v| !v.is_empty()) {
        return Some(PathBuf::from(dir));
    }
    if let Some(home) = std::env::var_os("HF_HOME").filter(|v| !v.is_empty()) {
        return Some(PathBuf::from(home).join("hub"));
    }
    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache/huggingface/hub"))
}

/// True when `repo_id` (`owner/name`) has a complete snapshot in `hub`: a snapshot
/// directory with `config.json` and at least one weight file, and no partial blobs.
pub fn repo_cached(hub: &Path, repo_id: &str) -> bool {
    let Some((owner, name)) = repo_id.split_once('/') else {
        return false;
    };
    let repo_dir = hub.join(format!("models--{owner}--{name}"));
    if let Ok(blobs) = std::fs::read_dir(repo_dir.join("blobs")) {
        for blob in blobs.flatten() {
            if blob.file_name().to_string_lossy().ends_with(".incomplete") {
                return false;
            }
        }
    }
    let Ok(snapshots) = std::fs::read_dir(repo_dir.join("snapshots")) else {
        return false;
    };
    snapshots.flatten().any(|snapshot| {
        let path = snapshot.path();
        path.join("config.json").exists()
            && std::fs::read_dir(&path).is_ok_and(|entries| {
                entries.flatten().any(|entry| {
                    let file = entry.file_name().to_string_lossy().to_string();
                    file.ends_with(".safetensors") || file.ends_with(".npz")
                })
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(hub: &Path, files: &[&str]) {
        let snap = hub.join("models--o--m/snapshots/abc");
        std::fs::create_dir_all(&snap).unwrap();
        for f in files {
            std::fs::write(snap.join(f), b"x").unwrap();
        }
    }

    #[test]
    fn complete_snapshot_is_cached() {
        let t = tempfile::tempdir().unwrap();
        snapshot(t.path(), &["config.json", "model.safetensors"]);
        assert!(repo_cached(t.path(), "o/m"));
    }

    #[test]
    fn metadata_only_snapshot_is_not_cached() {
        let t = tempfile::tempdir().unwrap();
        snapshot(t.path(), &["config.json", "README.md"]);
        assert!(!repo_cached(t.path(), "o/m"));
    }

    #[test]
    fn incomplete_blob_is_not_cached() {
        let t = tempfile::tempdir().unwrap();
        snapshot(t.path(), &["config.json", "model.safetensors"]);
        let blobs = t.path().join("models--o--m/blobs");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::write(blobs.join("deadbeef.incomplete"), b"x").unwrap();
        assert!(!repo_cached(t.path(), "o/m"));
    }

    #[test]
    fn missing_repo_is_not_cached() {
        let t = tempfile::tempdir().unwrap();
        assert!(!repo_cached(t.path(), "o/m"));
        assert!(!repo_cached(t.path(), "no-slash"));
    }
}
