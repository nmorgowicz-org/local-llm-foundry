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

/// Hub to launch from for a catalog alias: the app hub when it holds the model, else the
/// user's global hub if it does. New downloads always land in the app hub, so this only
/// lets already-downloaded global copies be reused without a duplicate download.
pub fn alias_launch_hub(models_dir: &Path, repo_id: &str) -> Option<PathBuf> {
    let app = app_hub_dir(models_dir);
    if repo_cached(&app, repo_id) {
        return Some(app);
    }
    system_hub_dir().filter(|hub| repo_cached(hub, repo_id))
}

/// Short quantization label for an MLX model (`4-bit`, `MXFP4`, …), the MLX analogue of a
/// GGUF `Q4_K_M` tag. Prefers the `quantization` block in the cached `config.json`
/// (authoritative) and falls back to tokens in the repo/alias name.
pub fn quant_label(hubs: &[PathBuf], repo_id: &str, alias: &str) -> Option<String> {
    if let Some((owner, name)) = repo_id.split_once('/') {
        for hub in hubs {
            let snapshots = hub.join(format!("models--{owner}--{name}")).join("snapshots");
            let Ok(entries) = std::fs::read_dir(snapshots) else {
                continue;
            };
            for entry in entries.flatten() {
                let Ok(text) = std::fs::read_to_string(entry.path().join("config.json")) else {
                    continue;
                };
                if let Ok(config) = serde_json::from_str::<serde_json::Value>(&text)
                    && let Some(label) = quant_from_config(&config)
                {
                    return Some(label);
                }
            }
        }
    }
    quant_from_name(repo_id).or_else(|| quant_from_name(alias))
}

fn quant_from_config(config: &serde_json::Value) -> Option<String> {
    let q = config
        .get("quantization")
        .or_else(|| config.get("quantization_config"))?;
    if let Some(mode) = q.get("mode").and_then(|m| m.as_str())
        && matches!(mode, "mxfp4" | "nvfp4" | "mxfp8")
    {
        return Some(mode.to_ascii_uppercase());
    }
    q.get("bits")
        .and_then(|b| b.as_u64())
        .map(|bits| format!("{bits}-bit"))
}

fn quant_from_name(name: &str) -> Option<String> {
    let lower = name.to_ascii_lowercase();
    for token in ["mxfp4", "nvfp4", "mxfp8"] {
        if lower.contains(token) {
            return Some(token.to_ascii_uppercase());
        }
    }
    // `4bit`, `8-bit`, `6bit` …
    let bytes = lower.as_bytes();
    for (i, _) in lower.match_indices("bit") {
        let mut j = i;
        if j > 0 && bytes[j - 1] == b'-' {
            j -= 1;
        }
        let end = j;
        while j > 0 && bytes[j - 1].is_ascii_digit() {
            j -= 1;
        }
        if j < end {
            return Some(format!("{}-bit", &lower[j..end]));
        }
    }
    for token in ["bf16", "fp16"] {
        if lower.contains(token) {
            return Some(token.to_ascii_uppercase());
        }
    }
    None
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
    fn alias_prefers_app_hub_when_complete() {
        let models = tempfile::tempdir().unwrap();
        let hub = app_hub_dir(models.path());
        snapshot(&hub, &["config.json", "model.safetensors"]);
        assert_eq!(alias_launch_hub(models.path(), "o/m"), Some(hub));
    }

    #[test]
    fn quant_labels_from_names_and_config() {
        assert_eq!(quant_from_name("rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX").as_deref(), Some("4-bit"));
        assert_eq!(quant_from_name("x/Model-8-bit").as_deref(), Some("8-bit"));
        assert_eq!(quant_from_name("x/Model-mxfp4-mlx").as_deref(), Some("MXFP4"));
        assert_eq!(quant_from_name("x/plain-model"), None);
        let cfg = serde_json::json!({"quantization": {"group_size": 32, "bits": 4, "mode": "mxfp4"}});
        assert_eq!(quant_from_config(&cfg).as_deref(), Some("MXFP4"));
        let cfg = serde_json::json!({"quantization": {"group_size": 64, "bits": 6}});
        assert_eq!(quant_from_config(&cfg).as_deref(), Some("6-bit"));
    }

    #[test]
    fn missing_repo_is_not_cached() {
        let t = tempfile::tempdir().unwrap();
        assert!(!repo_cached(t.path(), "o/m"));
        assert!(!repo_cached(t.path(), "no-slash"));
    }
}
