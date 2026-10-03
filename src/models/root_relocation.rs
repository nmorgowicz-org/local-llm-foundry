//! Pure, receipt-ready model-root relocation planning.
//!
//! The application-home migration deliberately retains model trees. This
//! module owns the later explicit “keep here” / “move into Foundry” decision
//! without guessing from filenames or touching external Hugging Face roots.

use std::fs;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelRootChoice {
    KeepLegacy,
    CopyIntoFoundry,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelResourceClass {
    ManagedModel,
    ManagedRuntime,
    PartialDownload,
    HuggingFaceCache,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRelocationEntry {
    pub relative_path: PathBuf,
    pub class: ModelResourceClass,
    pub bytes: u64,
    pub is_directory: bool,
    #[serde(default)]
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRelocationPlan {
    pub schema_version: u32,
    pub plan_id: String,
    pub choice: ModelRootChoice,
    pub source: PathBuf,
    pub destination: PathBuf,
    pub entries: Vec<ModelRelocationEntry>,
    pub required_copy_bytes: u64,
    #[serde(default)]
    pub available_destination_bytes: Option<u64>,
    #[serde(default)]
    pub persistence_rewrites: Vec<ModelRelocationRewrite>,
    pub retained_external_roots: Vec<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRelocationRewrite {
    pub file: PathBuf,
    pub replacements: usize,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRelocationReceipt {
    pub schema_version: u32,
    pub plan_id: String,
    pub source: PathBuf,
    pub destination: PathBuf,
    pub copied_entries: Vec<PathBuf>,
    pub retained_source: bool,
    #[serde(default)]
    pub rewritten_files: Vec<PathBuf>,
    #[serde(default)]
    pub retained_external_roots: Vec<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRootSelection {
    pub schema_version: u32,
    pub choice: ModelRootChoice,
    pub plan_id: String,
    pub source: PathBuf,
    pub destination: PathBuf,
    pub retained_source: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelRelocationJournal {
    schema_version: u32,
    plan_id: String,
    completed_entries: Vec<PathBuf>,
}

pub fn plan_model_root_relocation(
    source: &Path,
    destination: &Path,
    choice: ModelRootChoice,
) -> Result<ModelRelocationPlan> {
    plan_model_root_relocation_with_persistence(source, destination, choice, &[])
}

pub fn plan_model_root_relocation_with_persistence(
    source: &Path,
    destination: &Path,
    choice: ModelRootChoice,
    persistence_files: &[PathBuf],
) -> Result<ModelRelocationPlan> {
    let metadata = fs::symlink_metadata(source).context("model root is not readable")?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        bail!("model root must be a real directory");
    }
    if source == destination {
        bail!("model relocation source and destination are identical");
    }
    if destination.exists()
        && (!destination.is_dir() || fs::read_dir(destination)?.next().is_some())
    {
        bail!("model relocation destination is not empty");
    }

    let mut entries = Vec::new();
    collect(source, source, &mut entries)?;
    entries.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    let required_copy_bytes = if choice == ModelRootChoice::CopyIntoFoundry {
        entries
            .iter()
            .filter(|entry| !entry.is_directory)
            .map(|entry| entry.bytes)
            .sum()
    } else {
        0
    };
    let available_destination_bytes = available_space(destination);
    let persistence_rewrites = if choice == ModelRootChoice::CopyIntoFoundry {
        plan_persistence_rewrites(persistence_files, source, destination)?
    } else {
        Vec::new()
    };
    let retained_external_roots = if choice == ModelRootChoice::KeepLegacy {
        vec![source.to_path_buf()]
    } else {
        Vec::new()
    };
    let plan_id = hex_digest(&Sha256::digest(serde_json::to_vec(&(
        1u32,
        choice,
        source,
        destination,
        &entries,
        &persistence_rewrites,
        &retained_external_roots,
    ))?));
    Ok(ModelRelocationPlan {
        schema_version: 1,
        plan_id,
        choice,
        source: source.to_path_buf(),
        destination: destination.to_path_buf(),
        entries,
        required_copy_bytes,
        available_destination_bytes,
        persistence_rewrites,
        retained_external_roots,
    })
}

pub fn relocation_receipt_path(plan: &ModelRelocationPlan) -> PathBuf {
    plan.destination
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!(
            ".local-llm-foundry-model-relocation-{}.json",
            plan.plan_id
        ))
}

pub fn relocation_selection_path(plan: &ModelRelocationPlan) -> PathBuf {
    plan.destination
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(".local-llm-foundry-model-root.json")
}

fn relocation_journal_path(plan: &ModelRelocationPlan) -> PathBuf {
    plan.destination
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!(
            ".local-llm-foundry-model-relocation-{}.journal",
            plan.plan_id
        ))
}

/// Copy-first model relocation. The source is retained until a later,
/// receipt-scoped cleanup action; rerunning a completed plan returns its receipt.
pub fn execute_model_root_relocation(plan: &ModelRelocationPlan) -> Result<ModelRelocationReceipt> {
    if plan.choice == ModelRootChoice::KeepLegacy {
        let receipt = ModelRelocationReceipt {
            schema_version: 1,
            plan_id: plan.plan_id.clone(),
            source: plan.source.clone(),
            destination: plan.destination.clone(),
            copied_entries: Vec::new(),
            retained_source: true,
            rewritten_files: Vec::new(),
            retained_external_roots: plan.retained_external_roots.clone(),
        };
        write_receipt(plan, &receipt)?;
        write_selection(
            plan,
            &ModelRootSelection {
                schema_version: 1,
                choice: plan.choice,
                plan_id: plan.plan_id.clone(),
                source: plan.source.clone(),
                destination: plan.destination.clone(),
                retained_source: true,
            },
        )?;
        return Ok(receipt);
    }
    if let Ok(file) = fs::File::open(relocation_receipt_path(plan))
        && let Ok(receipt) = serde_json::from_reader(file)
    {
        return Ok(receipt);
    }
    let journal_path = relocation_journal_path(plan);
    let current = if journal_path.is_file() {
        let mut current = plan.clone();
        let mut entries = Vec::new();
        collect(&plan.source, &plan.source, &mut entries)?;
        entries.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
        current.entries = entries;
        current
    } else {
        let persistence_files = plan
            .persistence_rewrites
            .iter()
            .map(|rewrite| rewrite.file.clone())
            .collect::<Vec<_>>();
        let current = plan_model_root_relocation_with_persistence(
            &plan.source,
            &plan.destination,
            plan.choice,
            &persistence_files,
        )?;
        if current.plan_id != plan.plan_id {
            bail!("model relocation preview is stale");
        }
        current
    };
    // Moving doesn't duplicate data — no disk space check needed.
    if plan.choice == ModelRootChoice::CopyIntoFoundry {
        if let Some(available) = current.available_destination_bytes
            && available < current.required_copy_bytes
        {
            bail!(
                "insufficient free space for model relocation ({} bytes required)",
                current.required_copy_bytes
            );
        }
    }
    let mut journal = if journal_path.is_file() {
        let journal: ModelRelocationJournal =
            serde_json::from_reader(fs::File::open(&journal_path)?)?;
        if journal.plan_id != plan.plan_id {
            bail!("model relocation journal plan does not match preview");
        }
        journal
    } else {
        ModelRelocationJournal {
            schema_version: 1,
            plan_id: plan.plan_id.clone(),
            completed_entries: Vec::new(),
        }
    };
    fs::create_dir_all(&plan.destination)?;
    let mut copied_entries = Vec::new();
    for entry in &current.entries {
        let source = plan.source.join(&entry.relative_path);
        let destination = plan.destination.join(&entry.relative_path);
        if entry.is_directory {
            fs::create_dir_all(&destination)?;
            continue;
        }
        if journal
            .completed_entries
            .iter()
            .any(|path| path == &entry.relative_path)
        {
            copied_entries.push(entry.relative_path.clone());
            continue;
        }
        if destination.exists() {
            if plan.choice == ModelRootChoice::CopyIntoFoundry {
                // If destination already exists when moving, verify it's the
                // same file.
                let destination_hash = sha256_file(&destination)?;
                if fs::metadata(&destination)?.len() != entry.bytes
                    || entry.sha256.as_deref() != Some(destination_hash.as_str())
                {
                    bail!(
                        "model relocation refuses to overwrite {}",
                        destination.display()
                    );
                }
            } else {
                let destination_hash = sha256_file(&destination)?;
                if fs::metadata(&destination)?.len() != entry.bytes
                    || entry.sha256.as_deref() != Some(destination_hash.as_str())
                {
                    bail!(
                        "model relocation refuses to overwrite {}",
                        destination.display()
                    );
                }
            }
            journal.completed_entries.push(entry.relative_path.clone());
            copied_entries.push(entry.relative_path.clone());
            fs::write(&journal_path, serde_json::to_vec_pretty(&journal)?)?;
            continue;
        }
        if plan.choice == ModelRootChoice::CopyIntoFoundry {
            // Move files (metadata-only rename on same filesystem).
            fs::create_dir_all(
                destination
                    .parent()
                    .context("model destination has no parent")?,
            )?;
            fs::rename(&source, &destination)?;
            copied_entries.push(entry.relative_path.clone());
            journal.completed_entries.push(entry.relative_path.clone());
            fs::write(&journal_path, serde_json::to_vec_pretty(&journal)?)?;
        } else {
            // Copy files (default behavior).
            fs::create_dir_all(
                destination
                    .parent()
                    .context("model destination has no parent")?,
            )?;
            let temporary = destination.with_extension("local-llm-foundry-part");
            fs::copy(&source, &temporary)?;
            if fs::metadata(&temporary)?.len() != entry.bytes {
                let _ = fs::remove_file(&temporary);
                bail!(
                    "model relocation verification failed for {}",
                    entry.relative_path.display()
                );
            }
            fs::rename(&temporary, &destination)?;
            if entry.sha256.as_deref() != Some(sha256_file(&destination)?.as_str()) {
                bail!(
                    "model relocation hash verification failed for {}",
                    entry.relative_path.display()
                );
            }
            copied_entries.push(entry.relative_path.clone());
            journal.completed_entries.push(entry.relative_path.clone());
            fs::write(&journal_path, serde_json::to_vec_pretty(&journal)?)?;
        }
    }
    let mut rewritten_files = Vec::new();
    for rewrite in &current.persistence_rewrites {
        if !rewrite.file.is_file() {
            continue;
        }
        if sha256_file(&rewrite.file)? != rewrite.sha256 {
            bail!(
                "model-root relocation persistence file changed: {}",
                rewrite.file.display()
            );
        }
        let mut value: serde_json::Value = serde_json::from_reader(fs::File::open(&rewrite.file)?)?;
        let replacements = [(
            plan.source.to_string_lossy().into_owned(),
            plan.destination.to_string_lossy().into_owned(),
        )]
        .into_iter()
        .collect();
        replace_json_paths(&mut value, &replacements)?;
        write_json_atomic(&rewrite.file, &value)?;
        rewritten_files.push(rewrite.file.clone());
    }
    let receipt = ModelRelocationReceipt {
        schema_version: 1,
        plan_id: plan.plan_id.clone(),
        source: plan.source.clone(),
        destination: plan.destination.clone(),
        copied_entries,
        retained_source: true,
        rewritten_files,
        retained_external_roots: plan.retained_external_roots.clone(),
    };
    write_receipt(plan, &receipt)?;
    write_selection(
        plan,
        &ModelRootSelection {
            schema_version: 1,
            choice: plan.choice,
            plan_id: plan.plan_id.clone(),
            source: plan.source.clone(),
            destination: plan.destination.clone(),
            retained_source: true,
        },
    )?;
    let _ = fs::remove_file(journal_path);
    Ok(receipt)
}

fn collect(root: &Path, current: &Path, entries: &mut Vec<ModelRelocationEntry>) -> Result<()> {
    for item in fs::read_dir(current)? {
        let item = item?;
        let path = item.path();
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() {
            // Allow symlinks inside the HuggingFace cache — snapshots are symlinks
            // to blobs. Reject symlinks elsewhere as they could be crafted to
            // escape the source root.
            if !path.starts_with(root.join("cache/huggingface/hub")) {
                bail!("model relocation refuses symlink: {}", path.display());
            }
        }
        let relative_path = path
            .strip_prefix(root)
            .context("model relocation path escaped source")?
            .to_path_buf();
        let is_directory = metadata.is_dir();
        let class = classify(&relative_path);
        entries.push(ModelRelocationEntry {
            relative_path,
            class,
            bytes: if metadata.is_file() {
                metadata.len()
            } else {
                0
            },
            is_directory,
            sha256: if metadata.is_file() {
                Some(sha256_file(&path)?)
            } else {
                None
            },
        });
        if is_directory {
            collect(root, &path, entries)?;
        }
    }
    Ok(())
}

fn plan_persistence_rewrites(
    persistence_files: &[PathBuf],
    source: &Path,
    destination: &Path,
) -> Result<Vec<ModelRelocationRewrite>> {
    let replacements = [(
        source.to_string_lossy().into_owned(),
        destination.to_string_lossy().into_owned(),
    )]
    .into_iter()
    .collect();
    persistence_files
        .iter()
        .filter(|file| file.is_file())
        .map(|file| {
            let value: serde_json::Value = serde_json::from_reader(fs::File::open(file)?)?;
            let replacements_count = count_json_replacements(&value, &replacements);
            if replacements_count == 0 {
                return Ok(None);
            }
            Ok(Some(ModelRelocationRewrite {
                file: file.clone(),
                replacements: replacements_count,
                sha256: sha256_file(file)?,
            }))
        })
        .filter_map(|result| result.transpose())
        .collect()
}

fn replace_json_paths(
    value: &mut serde_json::Value,
    replacements: &std::collections::BTreeMap<String, String>,
) -> Result<()> {
    match value {
        serde_json::Value::String(text) => {
            if let Some(replacement) = replacement_for_path(text, replacements) {
                *text = replacement;
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                replace_json_paths(item, replacements)?;
            }
        }
        serde_json::Value::Object(map) => {
            let old = std::mem::take(map);
            for (key, mut child) in old {
                replace_json_paths(&mut child, replacements)?;
                let rewritten_key = replacement_for_path(&key, replacements).unwrap_or(key);
                if map.insert(rewritten_key.clone(), child).is_some() {
                    bail!("model-root persistence rewrite collided at {rewritten_key}");
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn count_json_replacements(
    value: &serde_json::Value,
    replacements: &std::collections::BTreeMap<String, String>,
) -> usize {
    match value {
        serde_json::Value::String(text) => {
            usize::from(replacement_for_path(text, replacements).is_some())
        }
        serde_json::Value::Array(items) => items
            .iter()
            .map(|item| count_json_replacements(item, replacements))
            .sum(),
        serde_json::Value::Object(map) => map
            .iter()
            .map(|(key, item)| {
                usize::from(replacement_for_path(key, replacements).is_some())
                    + count_json_replacements(item, replacements)
            })
            .sum(),
        _ => 0,
    }
}

fn replacement_for_path(
    value: &str,
    replacements: &std::collections::BTreeMap<String, String>,
) -> Option<String> {
    if let Some(exact) = replacements.get(value) {
        return Some(exact.clone());
    }
    let value_path = Path::new(value);
    if !value_path.is_absolute() {
        return None;
    }
    replacements.iter().find_map(|(source, destination)| {
        let relative = value_path.strip_prefix(Path::new(source)).ok()?;
        if relative.as_os_str().is_empty() {
            return None;
        }
        Some(
            Path::new(destination)
                .join(relative)
                .to_string_lossy()
                .into_owned(),
        )
    })
}

fn write_json_atomic(path: &Path, value: &serde_json::Value) -> Result<()> {
    let temporary = path.with_extension("json.local-llm-foundry-part");
    fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn available_space(destination: &Path) -> Option<u64> {
    let mut probe = destination.to_path_buf();
    while !probe.exists() {
        if !probe.pop() {
            return None;
        }
    }
    sysinfo::Disks::new_with_refreshed_list()
        .list()
        .iter()
        .filter(|disk| probe.starts_with(disk.mount_point()))
        .max_by_key(|disk| disk.mount_point().as_os_str().len())
        .map(sysinfo::Disk::available_space)
}

fn sha256_file(path: &Path) -> Result<String> {
    let file = fs::File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 128 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(hex_digest(&digest.finalize()))
}

fn write_receipt(plan: &ModelRelocationPlan, receipt: &ModelRelocationReceipt) -> Result<()> {
    let path = relocation_receipt_path(plan);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("json.local-llm-foundry-part");
    fs::write(&temporary, serde_json::to_vec_pretty(receipt)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn write_selection(plan: &ModelRelocationPlan, selection: &ModelRootSelection) -> Result<()> {
    let path = relocation_selection_path(plan);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("json.local-llm-foundry-part");
    fs::write(&temporary, serde_json::to_vec_pretty(selection)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

pub fn load_selection(destination: &Path) -> Result<Option<ModelRootSelection>> {
    let path = destination
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(".local-llm-foundry-model-root.json");
    match fs::File::open(path) {
        Ok(file) => Ok(Some(serde_json::from_reader(file)?)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn classify(path: &Path) -> ModelResourceClass {
    let lower = path.to_string_lossy().to_ascii_lowercase();
    if lower.contains("huggingface") || lower.contains("cache/hub") {
        ModelResourceClass::HuggingFaceCache
    } else if lower.contains(".staging") || lower.ends_with(".part") {
        ModelResourceClass::PartialDownload
    } else if lower.starts_with("rapid-mlx") || lower.starts_with("runtimes") {
        ModelResourceClass::ManagedRuntime
    } else if lower.starts_with("gguf")
        || lower.starts_with("mlx")
        || lower.starts_with("transformers")
    {
        ModelResourceClass::ManagedModel
    } else {
        ModelResourceClass::Unknown
    }
}

fn hex_digest(digest: &[u8]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keep_choice_is_non_mutating_and_retains_explicit_external_root() {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("legacy-models");
        let destination = root.path().join("foundry-models");
        fs::create_dir_all(source.join("gguf/.staging")).unwrap();
        fs::write(source.join("gguf/model.gguf"), b"model").unwrap();
        let plan =
            plan_model_root_relocation(&source, &destination, ModelRootChoice::KeepLegacy).unwrap();
        assert_eq!(plan.required_copy_bytes, 0);
        assert_eq!(plan.retained_external_roots, vec![source.clone()]);
        assert!(!destination.exists());
        let receipt = execute_model_root_relocation(&plan).unwrap();
        assert!(receipt.retained_source);
        assert!(relocation_receipt_path(&plan).is_file());
        let selection = load_selection(&destination).unwrap().unwrap();
        assert_eq!(selection.choice, ModelRootChoice::KeepLegacy);
        assert_eq!(selection.source, source);
    }

    #[test]
    fn move_choice_is_deterministic_and_classifies_partial_cache_runtime() {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("legacy-models");
        let destination = root.path().join("foundry-models");
        fs::create_dir_all(source.join("mlx/native")).unwrap();
        fs::create_dir_all(source.join("cache/huggingface/hub")).unwrap();
        fs::create_dir_all(source.join(".staging")).unwrap();
        fs::create_dir_all(source.join("rapid-mlx")).unwrap();
        fs::write(source.join("mlx/native/model.safetensors"), b"model").unwrap();
        fs::write(source.join(".staging/model.part"), b"part").unwrap();
        let first =
            plan_model_root_relocation(&source, &destination, ModelRootChoice::CopyIntoFoundry)
                .unwrap();
        let second =
            plan_model_root_relocation(&source, &destination, ModelRootChoice::CopyIntoFoundry)
                .unwrap();
        assert_eq!(first.plan_id, second.plan_id);
        assert!(first.required_copy_bytes > 0);
        assert!(
            first
                .entries
                .iter()
                .any(|entry| entry.class == ModelResourceClass::PartialDownload)
        );
        assert!(
            first
                .entries
                .iter()
                .any(|entry| entry.class == ModelResourceClass::HuggingFaceCache)
        );
        assert!(
            first
                .entries
                .iter()
                .any(|entry| entry.class == ModelResourceClass::ManagedRuntime)
        );
    }

    #[test]
    fn move_execution_is_copy_first_verified_and_idempotent() {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("legacy-models");
        let destination = root.path().join("foundry-models");
        fs::create_dir_all(source.join("gguf")).unwrap();
        fs::write(source.join("gguf/model.gguf"), b"model").unwrap();
        let plan =
            plan_model_root_relocation(&source, &destination, ModelRootChoice::CopyIntoFoundry)
                .unwrap();
        let receipt = execute_model_root_relocation(&plan).unwrap();
        assert!(receipt.retained_source);
        assert_eq!(
            fs::read(destination.join("gguf/model.gguf")).unwrap(),
            b"model"
        );
        assert!(source.join("gguf/model.gguf").is_file());
        let replay = execute_model_root_relocation(&plan).unwrap();
        assert_eq!(replay.plan_id, receipt.plan_id);
        assert!(load_selection(&destination).unwrap().is_some());
    }

    #[test]
    fn source_content_change_invalidates_preview_before_copy() {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("legacy-models");
        let destination = root.path().join("foundry-models");
        fs::create_dir_all(source.join("gguf")).unwrap();
        let model = source.join("gguf/model.gguf");
        fs::write(&model, b"model").unwrap();
        let plan =
            plan_model_root_relocation(&source, &destination, ModelRootChoice::CopyIntoFoundry)
                .unwrap();
        fs::write(model, b"changed").unwrap();
        assert!(execute_model_root_relocation(&plan).is_err());
        assert!(!destination.join("gguf/model.gguf").exists());
    }

    #[test]
    fn move_rewrites_persisted_absolute_model_paths_once() {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("legacy-models");
        let destination = root.path().join("foundry-models");
        let settings = root.path().join("ui-settings.json");
        fs::create_dir_all(source.join("gguf")).unwrap();
        fs::write(source.join("gguf/model.gguf"), b"model").unwrap();
        fs::write(
            &settings,
            serde_json::to_vec(&serde_json::json!({
                "models_dir": source.to_string_lossy(),
                "recent": [source.join("gguf/model.gguf")],
            }))
            .unwrap(),
        )
        .unwrap();
        let plan = plan_model_root_relocation_with_persistence(
            &source,
            &destination,
            ModelRootChoice::CopyIntoFoundry,
            std::slice::from_ref(&settings),
        )
        .unwrap();
        assert_eq!(plan.persistence_rewrites.len(), 1);
        let receipt = execute_model_root_relocation(&plan).unwrap();
        assert_eq!(receipt.rewritten_files, vec![settings.clone()]);
        let rewritten: serde_json::Value =
            serde_json::from_reader(fs::File::open(settings).unwrap()).unwrap();
        assert_eq!(
            rewritten["models_dir"],
            destination.to_string_lossy().to_string()
        );
        assert_eq!(
            rewritten["recent"][0],
            destination
                .join("gguf/model.gguf")
                .to_string_lossy()
                .to_string()
        );
    }
}
