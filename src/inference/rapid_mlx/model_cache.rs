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
    system_hub_dir_from(
        std::env::var_os("HF_HUB_CACHE"),
        std::env::var_os("HF_HOME"),
        std::env::var_os("HOME"),
    )
}

/// Precedence: `HF_HUB_CACHE`, then `HF_HOME/hub`, then `~/.cache/huggingface/hub`.
/// Empty values count as unset. Split out so tests need not mutate process env.
fn system_hub_dir_from(
    hub_cache: Option<std::ffi::OsString>,
    hf_home: Option<std::ffi::OsString>,
    home: Option<std::ffi::OsString>,
) -> Option<PathBuf> {
    if let Some(dir) = hub_cache.filter(|v| !v.is_empty()) {
        return Some(PathBuf::from(dir));
    }
    if let Some(dir) = hf_home.filter(|v| !v.is_empty()) {
        return Some(PathBuf::from(dir).join("hub"));
    }
    home.map(|home| PathBuf::from(home).join(".cache/huggingface/hub"))
}

/// Hub to launch from for a catalog alias: the app hub when it holds the model, else the
/// user's global hub if it does. New downloads always land in the app hub, so this only
/// lets already-downloaded global copies be reused without a duplicate download.
pub fn alias_launch_hub(models_dir: &Path, repo_id: &str) -> Option<PathBuf> {
    alias_launch_hub_in(models_dir, repo_id, system_hub_dir())
}

pub fn alias_launch_hub_in(
    models_dir: &Path,
    repo_id: &str,
    system_hub: Option<PathBuf>,
) -> Option<PathBuf> {
    let app = app_hub_dir(models_dir);
    if repo_cached(&app, repo_id) {
        return Some(app);
    }
    system_hub.filter(|hub| repo_cached(hub, repo_id))
}

/// Short quantization label for an MLX model (`4-bit`, `MXFP4`, …), the MLX analogue of a
/// GGUF `Q4_K_M` tag. Prefers the `quantization` block in the cached `config.json`
/// (authoritative) and falls back to tokens in the repo/alias name.
pub fn quant_label(hubs: &[PathBuf], repo_id: &str, alias: &str) -> Option<String> {
    if let Some((owner, name)) = safe_repo_parts(repo_id) {
        for hub in hubs {
            let snapshots = hub
                .join(format!("models--{owner}--{name}"))
                .join("snapshots");
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

/// Split `owner/name`, refusing anything that could escape the hub directory when
/// formatted into `models--{owner}--{name}`: a path separator or `..` in either part,
/// or an empty part. Defense in depth; callers validate repo ids at the API edge too.
fn safe_repo_parts(repo_id: &str) -> Option<(&str, &str)> {
    let (owner, name) = repo_id.split_once('/')?;
    let unsafe_part =
        |part: &str| part.is_empty() || part.contains(['/', '\\']) || part.contains("..");
    (!unsafe_part(owner) && !unsafe_part(name)).then_some((owner, name))
}

/// True when `repo_id` (`owner/name`) has a complete snapshot in `hub`: a snapshot
/// directory with `config.json` and its weights, and no partial blobs.
///
/// "Its weights" means every shard named by `model.safetensors.index.json` when the
/// snapshot has one (symlinked blobs are resolved, so a dangling link is a missing
/// shard), otherwise at least one resolvable weight file.
pub fn repo_cached(hub: &Path, repo_id: &str) -> bool {
    complete_snapshot(hub, repo_id).is_some()
}

/// Bytes of weight files in the complete cached snapshot of `repo_id` in `hub`, or `None`
/// when the model is not completely cached there. Only files directly inside the snapshot
/// directory are summed (symlinked blobs resolved); nothing is walked recursively.
pub fn cached_weight_bytes(hub: &Path, repo_id: &str) -> Option<u64> {
    let snapshot = complete_snapshot(hub, repo_id)?;
    let mut total = 0u64;
    for entry in std::fs::read_dir(&snapshot)
        .ok()?
        .flatten()
        .take(MAX_SNAPSHOT_ENTRIES)
    {
        if !is_weight_name(&entry.file_name().to_string_lossy()) {
            continue;
        }
        if let Ok(meta) = std::fs::metadata(entry.path())
            && meta.is_file()
        {
            total = total.saturating_add(meta.len());
        }
    }
    Some(total)
}

/// Upper bound on directory entries inspected per snapshot.
const MAX_SNAPSHOT_ENTRIES: usize = 4096;
/// Upper bound on `model.safetensors.index.json` size that will be parsed.
const MAX_INDEX_BYTES: u64 = 16 * 1024 * 1024;

fn is_weight_name(file: &str) -> bool {
    file.ends_with(".safetensors") || file.ends_with(".npz")
}

fn complete_snapshot(hub: &Path, repo_id: &str) -> Option<PathBuf> {
    let (owner, name) = safe_repo_parts(repo_id)?;
    let repo_dir = hub.join(format!("models--{owner}--{name}"));
    if let Ok(blobs) = std::fs::read_dir(repo_dir.join("blobs")) {
        for blob in blobs.flatten() {
            if blob.file_name().to_string_lossy().ends_with(".incomplete") {
                return None;
            }
        }
    }
    std::fs::read_dir(repo_dir.join("snapshots"))
        .ok()?
        .flatten()
        .map(|snapshot| snapshot.path())
        .find(|path| snapshot_complete(path))
}

fn snapshot_complete(dir: &Path) -> bool {
    if !dir.join("config.json").exists() {
        return false;
    }
    let index = dir.join("model.safetensors.index.json");
    // `symlink_metadata`: a dangling index link is still a (broken) index, not "no index".
    if std::fs::symlink_metadata(&index).is_ok() {
        return index_shards_present(dir, &index);
    }
    std::fs::read_dir(dir).is_ok_and(|entries| {
        entries.flatten().take(MAX_SNAPSHOT_ENTRIES).any(|entry| {
            is_weight_name(&entry.file_name().to_string_lossy())
                && std::fs::metadata(entry.path()).is_ok_and(|meta| meta.is_file())
        })
    })
}

/// Every shard in the index's `weight_map` exists as a file. An unreadable, oversized,
/// malformed, or empty index, or one naming a path outside the snapshot, is incomplete.
fn index_shards_present(dir: &Path, index: &Path) -> bool {
    let Ok(meta) = std::fs::metadata(index) else {
        return false;
    };
    if meta.len() > MAX_INDEX_BYTES {
        return false;
    }
    let Ok(bytes) = std::fs::read(index) else {
        return false;
    };
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return false;
    };
    let Some(weight_map) = value.get("weight_map").and_then(|map| map.as_object()) else {
        return false;
    };
    let mut shards = std::collections::BTreeSet::new();
    for shard in weight_map.values() {
        let Some(shard) = shard.as_str() else {
            return false;
        };
        shards.insert(shard);
    }
    if shards.is_empty() {
        return false;
    }
    shards.into_iter().all(|shard| {
        let relative = Path::new(shard);
        relative
            .components()
            .all(|part| matches!(part, std::path::Component::Normal(_)))
            && shard.ends_with(".safetensors")
            && std::fs::metadata(dir.join(relative)).is_ok_and(|meta| meta.is_file())
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
    fn alias_falls_back_to_complete_global_hub() {
        let models = tempfile::tempdir().unwrap();
        let global = tempfile::tempdir().unwrap();
        snapshot(global.path(), &["config.json", "model.safetensors"]);
        assert_eq!(
            alias_launch_hub_in(models.path(), "o/m", Some(global.path().to_path_buf())),
            Some(global.path().to_path_buf())
        );
    }

    #[test]
    fn alias_global_fallback_rejects_incomplete_or_missing_copies() {
        let models = tempfile::tempdir().unwrap();
        let global = tempfile::tempdir().unwrap();
        assert_eq!(alias_launch_hub_in(models.path(), "o/m", None), None);
        assert_eq!(
            alias_launch_hub_in(models.path(), "o/m", Some(global.path().to_path_buf())),
            None
        );
        snapshot(global.path(), &["config.json", "model.safetensors"]);
        let blobs = global.path().join("models--o--m/blobs");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::write(blobs.join("x.incomplete"), b"x").unwrap();
        assert_eq!(
            alias_launch_hub_in(models.path(), "o/m", Some(global.path().to_path_buf())),
            None
        );
    }

    #[test]
    fn alias_prefers_app_hub_over_global_hub() {
        let models = tempfile::tempdir().unwrap();
        let global = tempfile::tempdir().unwrap();
        let app = app_hub_dir(models.path());
        snapshot(&app, &["config.json", "model.safetensors"]);
        snapshot(global.path(), &["config.json", "model.safetensors"]);
        assert_eq!(
            alias_launch_hub_in(models.path(), "o/m", Some(global.path().to_path_buf())),
            Some(app)
        );
    }

    #[test]
    fn system_hub_env_precedence_is_hub_cache_then_hf_home_then_home() {
        use std::ffi::OsString;
        let os = |s: &str| Some(OsString::from(s));
        assert_eq!(
            system_hub_dir_from(os("/hub"), os("/hf"), os("/home/u")),
            Some(PathBuf::from("/hub"))
        );
        assert_eq!(
            system_hub_dir_from(None, os("/hf"), os("/home/u")),
            Some(PathBuf::from("/hf/hub"))
        );
        assert_eq!(
            system_hub_dir_from(None, None, os("/home/u")),
            Some(PathBuf::from("/home/u/.cache/huggingface/hub"))
        );
        // Empty values are treated as unset.
        assert_eq!(
            system_hub_dir_from(os(""), os(""), os("/home/u")),
            Some(PathBuf::from("/home/u/.cache/huggingface/hub"))
        );
        assert_eq!(system_hub_dir_from(None, None, None), None);
    }

    #[test]
    fn cached_weight_bytes_sums_only_weight_files_in_the_snapshot() {
        let t = tempfile::tempdir().unwrap();
        snapshot(t.path(), &["config.json", "tokenizer.json"]);
        let snap = t.path().join("models--o--m/snapshots/abc");
        std::fs::write(snap.join("model-00001-of-00002.safetensors"), vec![0u8; 10]).unwrap();
        std::fs::write(snap.join("model-00002-of-00002.safetensors"), vec![0u8; 5]).unwrap();
        write_index(
            t.path(),
            &[
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ],
        );
        // Non-weight files and nested directories never count.
        std::fs::create_dir_all(snap.join("nested")).unwrap();
        std::fs::write(snap.join("nested/extra.safetensors"), vec![0u8; 1000]).unwrap();
        assert_eq!(cached_weight_bytes(t.path(), "o/m"), Some(15));
    }

    #[cfg(unix)]
    #[test]
    fn cached_weight_bytes_follows_symlinked_blobs() {
        let t = tempfile::tempdir().unwrap();
        snapshot(t.path(), &["config.json"]);
        let repo = t.path().join("models--o--m");
        std::fs::create_dir_all(repo.join("blobs")).unwrap();
        std::fs::write(repo.join("blobs/aaaa"), vec![0u8; 7]).unwrap();
        std::os::unix::fs::symlink(
            "../../blobs/aaaa",
            repo.join("snapshots/abc/model.safetensors"),
        )
        .unwrap();
        assert_eq!(cached_weight_bytes(t.path(), "o/m"), Some(7));
    }

    #[test]
    fn cached_weight_bytes_is_none_when_not_cached() {
        let t = tempfile::tempdir().unwrap();
        assert_eq!(cached_weight_bytes(t.path(), "o/m"), None);
        snapshot(t.path(), &["config.json", "README.md"]);
        assert_eq!(cached_weight_bytes(t.path(), "o/m"), None);
        assert_eq!(cached_weight_bytes(t.path(), "no-slash"), None);
    }

    #[test]
    fn quant_labels_from_names_and_config() {
        assert_eq!(
            quant_from_name("rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX").as_deref(),
            Some("4-bit")
        );
        assert_eq!(quant_from_name("x/Model-8-bit").as_deref(), Some("8-bit"));
        assert_eq!(
            quant_from_name("x/Model-mxfp4-mlx").as_deref(),
            Some("MXFP4")
        );
        assert_eq!(quant_from_name("x/plain-model"), None);
        let cfg =
            serde_json::json!({"quantization": {"group_size": 32, "bits": 4, "mode": "mxfp4"}});
        assert_eq!(quant_from_config(&cfg).as_deref(), Some("MXFP4"));
        let cfg = serde_json::json!({"quantization": {"group_size": 64, "bits": 6}});
        assert_eq!(quant_from_config(&cfg).as_deref(), Some("6-bit"));
    }

    fn write_index(hub: &Path, shards: &[&str]) {
        let weight_map: serde_json::Map<String, serde_json::Value> = shards
            .iter()
            .enumerate()
            .map(|(i, s)| (format!("layer.{i}.weight"), serde_json::json!(s)))
            .collect();
        let index = serde_json::json!({"metadata": {"total_size": 2}, "weight_map": weight_map});
        std::fs::write(
            hub.join("models--o--m/snapshots/abc/model.safetensors.index.json"),
            index.to_string(),
        )
        .unwrap();
    }

    #[test]
    fn sharded_snapshot_with_missing_shard_is_not_cached() {
        let t = tempfile::tempdir().unwrap();
        snapshot(
            t.path(),
            &["config.json", "model-00001-of-00002.safetensors"],
        );
        write_index(
            t.path(),
            &[
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ],
        );
        assert!(!repo_cached(t.path(), "o/m"));
    }

    #[test]
    fn sharded_snapshot_with_all_shards_is_cached() {
        let t = tempfile::tempdir().unwrap();
        snapshot(
            t.path(),
            &[
                "config.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ],
        );
        write_index(
            t.path(),
            &[
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ],
        );
        assert!(repo_cached(t.path(), "o/m"));
    }

    #[cfg(unix)]
    #[test]
    fn sharded_snapshot_resolves_symlinked_blobs() {
        let t = tempfile::tempdir().unwrap();
        snapshot(t.path(), &["config.json"]);
        let repo = t.path().join("models--o--m");
        let blobs = repo.join("blobs");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::write(blobs.join("aaaa"), b"x").unwrap();
        let snap = repo.join("snapshots/abc");
        std::os::unix::fs::symlink(
            "../../blobs/aaaa",
            snap.join("model-00001-of-00002.safetensors"),
        )
        .unwrap();
        // Second shard's symlink dangles: its blob was never downloaded.
        std::os::unix::fs::symlink(
            "../../blobs/bbbb",
            snap.join("model-00002-of-00002.safetensors"),
        )
        .unwrap();
        write_index(
            t.path(),
            &[
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ],
        );
        assert!(!repo_cached(t.path(), "o/m"));
        std::fs::write(blobs.join("bbbb"), b"x").unwrap();
        assert!(repo_cached(t.path(), "o/m"));
    }

    #[test]
    fn unsafe_or_empty_index_is_not_cached() {
        let t = tempfile::tempdir().unwrap();
        snapshot(t.path(), &["config.json", "model.safetensors"]);
        write_index(t.path(), &["../escape.safetensors"]);
        assert!(!repo_cached(t.path(), "o/m"));
        write_index(t.path(), &[]);
        assert!(!repo_cached(t.path(), "o/m"));
        std::fs::write(
            t.path()
                .join("models--o--m/snapshots/abc/model.safetensors.index.json"),
            "not json",
        )
        .unwrap();
        assert!(!repo_cached(t.path(), "o/m"));
    }

    #[test]
    fn single_file_without_index_stays_cached() {
        let t = tempfile::tempdir().unwrap();
        snapshot(t.path(), &["config.json", "weights.npz"]);
        assert!(repo_cached(t.path(), "o/m"));
    }

    #[test]
    fn missing_repo_is_not_cached() {
        let t = tempfile::tempdir().unwrap();
        assert!(!repo_cached(t.path(), "o/m"));
        assert!(!repo_cached(t.path(), "no-slash"));
    }

    /// Plant a complete, quantized snapshot at `hub/<relative>/snapshots/abc`.
    fn decoy_snapshot(hub: &Path, relative: &str) {
        let snap = hub.join(relative).join("snapshots/abc");
        std::fs::create_dir_all(&snap).unwrap();
        std::fs::write(snap.join("config.json"), r#"{"quantization":{"bits":4}}"#).unwrap();
        std::fs::write(snap.join("model.safetensors"), b"x").unwrap();
    }

    #[test]
    fn complete_snapshot_rejects_separators_and_dot_dot_in_owner_or_name() {
        let t = tempfile::tempdir().unwrap();
        // `o/m/x` would otherwise resolve to the nested `models--o--m/x` directory.
        decoy_snapshot(t.path(), "models--o--m/x");
        decoy_snapshot(t.path(), "models--o--a..b");
        for repo in ["o/m/x", "o/m\\x", "o\\p/m", "o/a..b", "../m", "o/.."] {
            assert!(!repo_cached(t.path(), repo), "{repo} must not be cached");
            assert_eq!(cached_weight_bytes(t.path(), repo), None, "{repo}");
        }
    }

    #[test]
    fn quant_label_rejects_separators_and_dot_dot_in_owner_or_name() {
        let t = tempfile::tempdir().unwrap();
        decoy_snapshot(t.path(), "models--o--m/x");
        decoy_snapshot(t.path(), "models--o--a..b");
        let hubs = [t.path().to_path_buf()];
        for repo in ["o/m/x", "o/m\\x", "o/a..b", "../m", "o/.."] {
            assert_eq!(
                quant_label(&hubs, repo, "plain"),
                None,
                "{repo} must not read a config outside its own repo directory"
            );
        }
        // A well-formed repo still reads its config.
        decoy_snapshot(t.path(), "models--o--ok");
        assert_eq!(
            quant_label(&hubs, "o/ok", "plain").as_deref(),
            Some("4-bit")
        );
    }
}
