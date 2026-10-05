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
    MoveIntoFoundry,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRelocationEntry {
    pub relative_path: PathBuf,
    pub class: ModelResourceClass,
    pub bytes: u64,
    pub is_directory: bool,
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default)]
    pub modified_unix_nanos: Option<u128>,
    #[serde(default)]
    pub symlink_target: Option<PathBuf>,
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
    pub total_move_bytes: u64,
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
    #[serde(default)]
    pub rewritten_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRelocationReceipt {
    pub schema_version: u32,
    pub plan_id: String,
    pub source: PathBuf,
    pub destination: PathBuf,
    #[serde(default, alias = "copied_entries")]
    pub moved_entries: Vec<PathBuf>,
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
    plan: ModelRelocationPlan,
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
    if let Some(pending) = load_pending_plan(destination)? {
        if pending.source != source
            || pending.destination != destination
            || pending.choice != choice
        {
            bail!("a different model relocation is pending");
        }
        return Ok(pending);
    }
    let metadata = fs::symlink_metadata(source).context("model root is not readable")?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        bail!("model root must be a real directory");
    }
    if choice == ModelRootChoice::MoveIntoFoundry {
        validate_destination(source, destination)?;
    }
    let mut entries = Vec::new();
    collect(source, source, &mut entries)?;
    entries.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    let total_move_bytes = if choice == ModelRootChoice::MoveIntoFoundry {
        entries.iter().map(|entry| entry.bytes).sum()
    } else {
        0
    };
    let persistence_rewrites = if choice == ModelRootChoice::MoveIntoFoundry {
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
        2u32,
        choice,
        source,
        destination,
        &entries,
        &persistence_rewrites,
        &retained_external_roots,
    ))?));
    Ok(ModelRelocationPlan {
        schema_version: 2,
        plan_id,
        choice,
        source: source.to_path_buf(),
        destination: destination.to_path_buf(),
        entries,
        required_copy_bytes: 0,
        total_move_bytes,
        available_destination_bytes: available_space(destination),
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
        .join(".local-llm-foundry-model-relocation.pending.json")
}

/// Move the entire model tree with one filesystem rename, never a copy.
/// The pending plan survives interruption while persisted paths are updated.
pub fn execute_model_root_relocation(plan: &ModelRelocationPlan) -> Result<ModelRelocationReceipt> {
    let retained_source = plan.choice == ModelRootChoice::KeepLegacy;
    let selection = ModelRootSelection {
        schema_version: 2,
        choice: plan.choice,
        plan_id: plan.plan_id.clone(),
        source: plan.source.clone(),
        destination: plan.destination.clone(),
        retained_source,
    };
    if retained_source {
        let receipt = ModelRelocationReceipt {
            schema_version: 2,
            plan_id: plan.plan_id.clone(),
            source: plan.source.clone(),
            destination: plan.destination.clone(),
            moved_entries: Vec::new(),
            retained_source: true,
            rewritten_files: Vec::new(),
            retained_external_roots: plan.retained_external_roots.clone(),
        };
        write_selection(plan, &selection)?;
        write_receipt(plan, &receipt)?;
        return Ok(receipt);
    }
    if plan.schema_version != 2 {
        bail!("model move requires a fresh move-only preview");
    }
    let journal_path = relocation_journal_path(plan);
    if relocation_receipt_path(plan).exists() {
        let receipt: ModelRelocationReceipt =
            serde_json::from_reader(fs::File::open(relocation_receipt_path(plan))?)?;
        if receipt.plan_id != plan.plan_id
            || receipt.source != plan.source
            || receipt.destination != plan.destination
            || receipt.retained_source
            || fs::symlink_metadata(&plan.source).is_ok()
        {
            bail!("model move receipt does not match the completed move");
        }
        validate_real_directory(&plan.destination)?;
        write_selection(plan, &selection)?;
        if journal_path.exists() {
            fs::remove_file(&journal_path)?;
        }
        return Ok(receipt);
    }
    let pending = load_pending_plan(&plan.destination)?;
    if let Some(pending) = &pending {
        if pending.plan_id != plan.plan_id
            || pending.source != plan.source
            || pending.destination != plan.destination
        {
            bail!("model relocation journal does not match preview");
        }
    } else {
        let persistence = plan
            .persistence_rewrites
            .iter()
            .map(|rewrite| rewrite.file.clone())
            .collect::<Vec<_>>();
        let current = plan_model_root_relocation_with_persistence(
            &plan.source,
            &plan.destination,
            plan.choice,
            &persistence,
        )?;
        if current.plan_id != plan.plan_id {
            bail!("model relocation preview is stale");
        }
    }
    // Validate every persisted path rewrite BEFORE moving anything. On recovery,
    // both the original and already-rewritten checksums are accepted.
    let rewrites = prepare_persistence_rewrites(plan)?;
    if pending.is_none() {
        let journal = ModelRelocationJournal {
            schema_version: 2,
            plan: plan.clone(),
        };
        write_json_atomic(&journal_path, &serde_json::to_value(journal)?)?;
    }
    match fs::symlink_metadata(&plan.source) {
        Ok(metadata) => {
            if !metadata.is_dir() || metadata.file_type().is_symlink() {
                bail!("model source must be a real directory");
            }
            validate_destination(&plan.source, &plan.destination)?;
            verify_inventory(&plan.source, &plan.entries)?;
            fs::create_dir_all(
                plan.destination
                    .parent()
                    .context("model destination has no parent")?,
            )?;
            // Only an empty, real destination directory may be removed.
            if plan.destination.exists() {
                // Re-check right before deleting: only a tree of empty
                // directories (the skeleton the app creates at startup) is
                // ever removed.
                if !is_empty_directory_tree(&plan.destination)? {
                    bail!("model relocation destination gained files; refusing to remove it");
                }
                fs::remove_dir_all(&plan.destination)?;
            }
            fs::rename(&plan.source, &plan.destination).context("model move failed; no copy fallback is permitted (source and destination must be on the same filesystem)")?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            if pending.is_none() {
                bail!("model source disappeared before move");
            }
            validate_real_directory(&plan.destination)?;
            verify_inventory(&plan.destination, &plan.entries)?;
        }
        Err(error) => return Err(error.into()),
    }
    for (file, value) in rewrites {
        write_json_atomic(&file, &value)?;
    }
    let receipt = ModelRelocationReceipt {
        schema_version: 2,
        plan_id: plan.plan_id.clone(),
        source: plan.source.clone(),
        destination: plan.destination.clone(),
        moved_entries: plan
            .entries
            .iter()
            .filter(|entry| !entry.is_directory)
            .map(|entry| entry.relative_path.clone())
            .collect(),
        retained_source: false,
        rewritten_files: plan
            .persistence_rewrites
            .iter()
            .map(|rewrite| rewrite.file.clone())
            .collect(),
        retained_external_roots: plan.retained_external_roots.clone(),
    };
    write_selection(plan, &selection)?;
    write_receipt(plan, &receipt)?;
    fs::remove_file(journal_path)?;
    Ok(receipt)
}

pub fn load_pending_plan(destination: &Path) -> Result<Option<ModelRelocationPlan>> {
    let path = destination
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(".local-llm-foundry-model-relocation.pending.json");
    let file = match fs::File::open(&path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let journal: ModelRelocationJournal =
        serde_json::from_reader(file).context("model move recovery record is invalid")?;
    if journal.schema_version != 2
        || journal.plan.schema_version != 2
        || journal.plan.destination != destination
        || journal.plan.choice != ModelRootChoice::MoveIntoFoundry
    {
        bail!("model move recovery record does not match destination");
    }
    Ok(Some(journal.plan))
}

fn validate_real_directory(root: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(root)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        bail!("model root must be a real directory");
    }
    Ok(())
}

/// True only when `root` contains nothing but real directories (no files,
/// no symlinks, no special entries) at any depth.
fn is_empty_directory_tree(root: &Path) -> Result<bool> {
    for item in fs::read_dir(root)? {
        let path = item?.path();
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink()
            || !metadata.is_dir()
            || !is_empty_directory_tree(&path)?
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn validate_destination(source: &Path, destination: &Path) -> Result<()> {
    if source == destination {
        bail!("model relocation source and destination are identical");
    }
    match fs::symlink_metadata(destination) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink()
                || !metadata.is_dir()
                || !is_empty_directory_tree(destination)?
            {
                bail!("model relocation destination must be an empty directory tree (no files)");
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let canonical_source = fs::canonicalize(source)?;
    let mut ancestor = destination.to_path_buf();
    let mut missing = Vec::new();
    while !ancestor.exists() {
        missing.push(
            ancestor
                .file_name()
                .context("invalid model destination")?
                .to_os_string(),
        );
        if !ancestor.pop() {
            bail!("model destination has no existing ancestor");
        }
    }
    let mut resolved_destination = fs::canonicalize(ancestor)?;
    for name in missing.into_iter().rev() {
        resolved_destination.push(name);
    }
    if resolved_destination.starts_with(&canonical_source)
        || canonical_source.starts_with(&resolved_destination)
    {
        bail!("model relocation roots must not overlap");
    }
    // The move is a single rename with no copy fallback, so a cross-device
    // destination can only fail after the journal exists — which blocks every
    // other choice with no cancel path. Detect it while still planning.
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let source_device = fs::metadata(&canonical_source)?.dev();
        let destination_parent = resolved_destination
            .parent()
            .ok_or_else(|| anyhow::anyhow!("model destination has no parent"))?;
        let destination_device = fs::metadata(destination_parent)?.dev();
        if source_device != destination_device {
            bail!(
                "model relocation source and destination are on different filesystems; a move would have no copy fallback"
            );
        }
    }
    #[cfg(not(unix))]
    {
        let _ = &canonical_source;
        let _ = &resolved_destination;
    }
    Ok(())
}

fn verify_inventory(root: &Path, expected: &[ModelRelocationEntry]) -> Result<()> {
    let mut actual = Vec::new();
    collect(root, root, &mut actual)?;
    actual.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    if actual != expected {
        bail!("model relocation inventory changed; refusing to move or resume");
    }
    Ok(())
}

fn prepare_persistence_rewrites(
    plan: &ModelRelocationPlan,
) -> Result<Vec<(PathBuf, serde_json::Value)>> {
    let replacements = [(
        plan.source.to_string_lossy().into_owned(),
        plan.destination.to_string_lossy().into_owned(),
    )]
    .into_iter()
    .collect();
    let mut pending = Vec::new();
    for rewrite in &plan.persistence_rewrites {
        let hash = sha256_file(&rewrite.file).with_context(|| {
            format!(
                "model move persistence file is unreadable: {}",
                rewrite.file.display()
            )
        })?;
        if hash == rewrite.rewritten_sha256 {
            continue;
        }
        if hash != rewrite.sha256 {
            bail!(
                "model move persistence file changed: {}",
                rewrite.file.display()
            );
        }
        let mut value: serde_json::Value = serde_json::from_reader(fs::File::open(&rewrite.file)?)?;
        replace_json_paths(&mut value, &replacements)?;
        let rewritten_hash = hex_digest(&Sha256::digest(serde_json::to_vec_pretty(&value)?));
        if rewritten_hash != rewrite.rewritten_sha256 {
            bail!("model move persistence rewrite does not match preview");
        }
        pending.push((rewrite.file.clone(), value));
    }
    Ok(pending)
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
        if !metadata.is_dir() && !metadata.is_file() && !metadata.file_type().is_symlink() {
            bail!("model relocation refuses special entry: {}", path.display());
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
            // Renaming preserves the actual files; fingerprint metadata,
            // rather than reading hundreds of GB just to approve a move.
            sha256: None,
            modified_unix_nanos: metadata
                .modified()?
                .duration_since(std::time::UNIX_EPOCH)
                .ok()
                .map(|time| time.as_nanos()),
            symlink_target: if metadata.file_type().is_symlink() {
                Some(fs::read_link(&path)?)
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
            let mut rewritten = value;
            replace_json_paths(&mut rewritten, &replacements)?;
            Ok(Some(ModelRelocationRewrite {
                file: file.clone(),
                replacements: replacements_count,
                sha256: sha256_file(file)?,
                rewritten_sha256: hex_digest(&Sha256::digest(serde_json::to_vec_pretty(
                    &rewritten,
                )?)),
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
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
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
            plan_model_root_relocation(&source, &destination, ModelRootChoice::MoveIntoFoundry)
                .unwrap();
        let second =
            plan_model_root_relocation(&source, &destination, ModelRootChoice::MoveIntoFoundry)
                .unwrap();
        assert_eq!(first.plan_id, second.plan_id);
        assert_eq!(first.required_copy_bytes, 0);
        assert!(first.total_move_bytes > 0);
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
    fn move_execution_renames_models_without_leaving_a_copy_and_is_idempotent() {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("legacy-models");
        let destination = root.path().join("foundry-models");
        fs::create_dir_all(source.join("gguf")).unwrap();
        fs::write(source.join("gguf/model.gguf"), b"model").unwrap();
        let plan =
            plan_model_root_relocation(&source, &destination, ModelRootChoice::MoveIntoFoundry)
                .unwrap();
        assert_eq!(plan.required_copy_bytes, 0);
        assert_eq!(plan.total_move_bytes, 5);
        let receipt = execute_model_root_relocation(&plan).unwrap();
        assert!(!receipt.retained_source);
        assert_eq!(
            fs::read(destination.join("gguf/model.gguf")).unwrap(),
            b"model"
        );
        assert!(!source.exists());
        assert!(load_pending_plan(&destination).unwrap().is_none());
        let replay = execute_model_root_relocation(&plan).unwrap();
        assert_eq!(replay.plan_id, receipt.plan_id);
        let selection = load_selection(&destination).unwrap().unwrap();
        assert!(!selection.retained_source);
    }

    #[test]
    fn source_content_change_invalidates_preview_before_move() {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("legacy-models");
        let destination = root.path().join("foundry-models");
        fs::create_dir_all(source.join("gguf")).unwrap();
        let model = source.join("gguf/model.gguf");
        fs::write(&model, b"model").unwrap();
        let plan =
            plan_model_root_relocation(&source, &destination, ModelRootChoice::MoveIntoFoundry)
                .unwrap();
        fs::write(&model, b"changed content").unwrap();
        assert!(execute_model_root_relocation(&plan).is_err());
        assert_eq!(fs::read(model).unwrap(), b"changed content");
        assert!(!destination.exists());
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
            ModelRootChoice::MoveIntoFoundry,
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

    #[test]
    fn move_refuses_nonempty_destination_without_touching_source() {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("legacy-models");
        let destination = root.path().join("foundry-models");
        fs::create_dir_all(source.join("gguf")).unwrap();
        fs::create_dir_all(&destination).unwrap();
        fs::write(source.join("gguf/model.gguf"), b"model").unwrap();
        fs::write(destination.join("existing.bin"), b"keep").unwrap();
        assert!(
            plan_model_root_relocation(&source, &destination, ModelRootChoice::MoveIntoFoundry)
                .is_err()
        );
        assert_eq!(fs::read(source.join("gguf/model.gguf")).unwrap(), b"model");
        assert_eq!(fs::read(destination.join("existing.bin")).unwrap(), b"keep");
    }

    #[test]
    fn move_replaces_empty_startup_skeleton_but_refuses_any_file() {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("legacy-models");
        let destination = root.path().join("foundry-models");
        fs::create_dir_all(source.join("gguf")).unwrap();
        fs::write(source.join("gguf/model.gguf"), b"model").unwrap();
        // The app creates this skeleton at startup.
        fs::create_dir_all(destination.join("cache/huggingface/hub")).unwrap();
        fs::create_dir_all(destination.join(".staging/downloads")).unwrap();
        let plan =
            plan_model_root_relocation(&source, &destination, ModelRootChoice::MoveIntoFoundry)
                .unwrap();
        execute_model_root_relocation(&plan).unwrap();
        assert_eq!(
            fs::read(destination.join("gguf/model.gguf")).unwrap(),
            b"model"
        );
        assert!(!source.exists());

        let source2 = root.path().join("legacy-2");
        let destination2 = root.path().join("foundry-2");
        fs::create_dir_all(&source2).unwrap();
        fs::write(source2.join("m.gguf"), b"m").unwrap();
        fs::create_dir_all(destination2.join("cache")).unwrap();
        fs::write(destination2.join("cache/stray.bin"), b"x").unwrap();
        assert!(
            plan_model_root_relocation(&source2, &destination2, ModelRootChoice::MoveIntoFoundry)
                .is_err()
        );
        assert!(source2.join("m.gguf").is_file());
    }

    #[test]
    fn interrupted_after_rename_resumes_without_replanning_or_copying() {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("legacy-models");
        let destination = root.path().join("foundry-models");
        let settings = root.path().join("ui-settings.json");
        fs::create_dir_all(source.join("gguf")).unwrap();
        fs::write(source.join("gguf/model.gguf"), b"model").unwrap();
        fs::write(
            &settings,
            serde_json::to_vec(&serde_json::json!({"models_dir": source})).unwrap(),
        )
        .unwrap();
        let plan = plan_model_root_relocation_with_persistence(
            &source,
            &destination,
            ModelRootChoice::MoveIntoFoundry,
            std::slice::from_ref(&settings),
        )
        .unwrap();
        let journal = ModelRelocationJournal {
            schema_version: 2,
            plan: plan.clone(),
        };
        write_json_atomic(
            &relocation_journal_path(&plan),
            &serde_json::to_value(journal).unwrap(),
        )
        .unwrap();
        fs::rename(&source, &destination).unwrap();
        let resumed = plan_model_root_relocation_with_persistence(
            &source,
            &destination,
            ModelRootChoice::MoveIntoFoundry,
            std::slice::from_ref(&settings),
        )
        .unwrap();
        assert_eq!(resumed.plan_id, plan.plan_id);
        let receipt = execute_model_root_relocation(&resumed).unwrap();
        assert_eq!(receipt.rewritten_files, vec![settings.clone()]);
        assert!(!source.exists());
        assert!(load_pending_plan(&destination).unwrap().is_none());
        let rewritten: serde_json::Value =
            serde_json::from_reader(fs::File::open(settings).unwrap()).unwrap();
        assert_eq!(
            rewritten["models_dir"],
            destination.to_string_lossy().to_string()
        );
    }

    #[test]
    fn move_refuses_nested_destination() {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("legacy-models");
        fs::create_dir_all(&source).unwrap();
        let destination = source.join("foundry-models");
        assert!(
            plan_model_root_relocation(&source, &destination, ModelRootChoice::MoveIntoFoundry)
                .is_err()
        );
    }
}
