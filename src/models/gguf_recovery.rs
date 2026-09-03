//! Phase 5.5 R2/R3 profile-scoped GGUF recovery and MLX re-quantization adapters.
//!
//! Recovered output remains experimental and outside production model inventory. The
//! adapter owns canonical staging, the exact worker/dependency identity, cancellation,
//! bounded diagnostics, structural closure validation, and atomic publication.

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fs;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::Semaphore;

const WORKER: &str = include_str!("../../tools/gguf_recovery/converter.py");
const PROFILE: &str = include_str!("../../tools/gguf_recovery/profiles/smollm2_135m_v1.json");
const REQUIREMENTS_LOCK: &str = include_str!("../../tools/gguf_recovery/requirements.lock");
const ENVIRONMENT_LOCK: &str = include_str!("../../tools/gguf_recovery/environment.lock.json");
const NOTICE: &str = include_str!("../../tools/gguf_recovery/THIRD_PARTY_NOTICE.md");
const REQUANT_WORKER: &str = include_str!("../../tools/mlx_requantize/quantize.py");
const REQUANT_PROFILE: &str = include_str!("../../tools/mlx_requantize/profile.json");
const REQUANT_ENVIRONMENT_LOCK: &str =
    include_str!("../../tools/mlx_requantize/environment.lock.json");
const REQUANT_NOTICE: &str = include_str!("../../tools/mlx_requantize/THIRD_PARTY_NOTICE.md");
const PROFILE_ID: &str = "smollm2-135m-instruct-llama-v1";
const WORKER_VERSION: &str = "llama-monitor-gguf-recovery-r2-v1";
const TOOLCHAIN_VERSION: &str = "r2-v1";
const REQUANT_PROFILE_ID: &str = "smollm2-135m-r3-affine-v1";
const REQUANT_WORKER_VERSION: &str = "llama-monitor-mlx-requantize-r3-v1";
const REQUANT_TOOLCHAIN_VERSION: &str = "r3-v1";
const R2_F16_CACHE_KEY: &str = "a21cca76ec236c3c71ea2bf5eb6f78716602b90fb16d78c3aef4da51e1ff4177";
const MAX_DIAGNOSTIC_BYTES: usize = 64 * 1024;
const MAX_REPORT_BYTES: u64 = 1024 * 1024;
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;
const MAX_REFERENCE_MANIFEST_BYTES: u64 = 256 * 1024;
const CONVERSION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30 * 60);
const REQUANTIZATION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2 * 60);
const TOOLCHAIN_PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
const PIPE_DRAIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);
const PROCESS_GROUP_GRACE: std::time::Duration = std::time::Duration::from_secs(2);
const MAX_TOOLCHAIN_PROBE_BYTES: usize = 16 * 1024;
static RECOVERY_BLOCKING_GATE: LazyLock<Arc<Semaphore>> =
    LazyLock::new(|| Arc::new(Semaphore::new(2)));

#[derive(Debug, Clone)]
pub struct RecoveryContext {
    pub models_dir: PathBuf,
    pub config_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryTier {
    F16,
    Q8_0,
    Q6K,
    Q4KM,
}

impl RecoveryTier {
    fn profile_name(&self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Q8_0 => "q8_0",
            Self::Q6K => "q6_k",
            Self::Q4KM => "q4_k_m",
        }
    }
}

#[derive(Debug, Clone)]
pub struct RecoveryRequest {
    /// Library-relative GGUF path. Absolute paths and traversal are rejected.
    pub source_gguf: PathBuf,
    /// Library-relative authoritative fixture directory.
    pub reference_dir: PathBuf,
    pub source_tier: RecoveryTier,
    pub max_output_bytes: u64,
    pub disk_safety_margin_bytes: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkerReport {
    pub schema_version: u32,
    pub worker_version: String,
    pub status: String,
    pub profile_id: String,
    pub source: WorkerSource,
    pub authoritative_reference: WorkerReference,
    pub output: WorkerOutput,
    pub tensor_inventory: Vec<WorkerTensor>,
    pub skipped_tensors: u64,
    pub unknown_tensors: u64,
    pub duplicate_tensors: u64,
    pub shape_mismatches: u64,
    pub non_finite_tensors: u64,
    pub error: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkerOutput {
    pub dtype: String,
    pub tensor_count: u64,
    pub files: BTreeMap<String, String>,
    pub estimated_weight_bytes: u64,
    pub actual_output_bytes: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct WorkerSource {
    pub path: PathBuf,
    pub tier: String,
    pub size_bytes: u64,
    pub sha256: String,
    pub architecture: String,
    pub quant_inventory: BTreeMap<String, u64>,
    pub tensor_count: u64,
    pub tensor_inventory_sha256: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct WorkerReference {
    pub repo_id: String,
    pub revision: String,
    pub files: BTreeMap<String, String>,
    pub tensor_count: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct WorkerTensor {
    pub index: u64,
    pub source_name: String,
    pub quant_type: String,
    pub source_shape: Vec<u64>,
    pub output_name: String,
    pub output_shape: Vec<u64>,
}

#[derive(Debug, Deserialize)]
struct RecoveryProfile {
    profile_id: String,
    architecture: String,
    authoritative_source: ProfileReference,
    gguf_source: ProfileGgufSource,
    expected_tensor_count: u64,
    output_dtype: String,
}

#[derive(Debug, Deserialize)]
struct ProfileReference {
    repo_id: String,
    revision: String,
    weight_file: String,
    weight_sha256: String,
    files: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct ProfileGgufSource {
    tiers: BTreeMap<String, ProfileTier>,
}

#[derive(Debug, Deserialize)]
struct ProfileTier {
    filename: String,
    size: u64,
    sha256: String,
    quant_inventory: BTreeMap<String, u64>,
    tensor_inventory_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RecoveryManifest {
    schema_version: u32,
    status: String,
    launchable: bool,
    profile_id: String,
    cache_key: String,
    source_tier: RecoveryTier,
    worker_sha256: String,
    profile_sha256: String,
    requirements_lock_sha256: String,
    environment_lock_sha256: String,
    third_party_notice_sha256: String,
    worker_report_sha256: String,
    files: BTreeMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct RecoveryResult {
    pub cache_dir: PathBuf,
    pub fp16_dir: PathBuf,
    pub report: WorkerReport,
    /// Always false in R2. Runtime/parity research never promotes production support.
    pub launchable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MlxQuantRecipe {
    Affine4bitG64,
    Affine6bitG64,
    Affine8bitG64,
}

impl MlxQuantRecipe {
    fn profile_name(&self) -> &'static str {
        match self {
            Self::Affine4bitG64 => "affine_4bit_g64",
            Self::Affine6bitG64 => "affine_6bit_g64",
            Self::Affine8bitG64 => "affine_8bit_g64",
        }
    }

    fn bits(&self) -> u8 {
        match self {
            Self::Affine4bitG64 => 4,
            Self::Affine6bitG64 => 6,
            Self::Affine8bitG64 => 8,
        }
    }

    fn expected_tensor_inventory_sha256(&self) -> &'static str {
        match self {
            Self::Affine4bitG64 => {
                "d1e8c7ce799b461e703366e3d3c2642620540ecbaab5410b2201335103b7702f"
            }
            Self::Affine6bitG64 => {
                "53301c8b6fd47b9ef391e686dc20b2b7be19a486954066f2b35f301d39215539"
            }
            Self::Affine8bitG64 => {
                "f8eac26255eb3a5ae55b400a313b326e5b90fe0b10590d97eff1bec68781fac5"
            }
        }
    }
}

const REQUANTIZED_TENSOR_COUNT: u64 = 694;
const REQUANTIZED_MODULE_COUNT: u64 = 211;

#[derive(Debug, Clone)]
pub struct RequantizeRequest {
    /// Library-relative path to the complete, validated R2 cache directory.
    pub recovered_cache: PathBuf,
    pub recipe: MlxQuantRecipe,
    pub max_output_bytes: u64,
    pub disk_safety_margin_bytes: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct RequantizeReport {
    pub schema_version: u32,
    pub worker_version: String,
    pub status: String,
    pub profile_id: String,
    pub labels: RequantizeLabels,
    pub source: RequantizeSource,
    pub runtime: RequantizeRuntime,
    pub recipe: RequantizeRecipeReport,
    pub output: RequantizeOutput,
    pub elapsed_seconds: f64,
    pub launchable: bool,
    pub error: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct RequantizeLabels {
    pub original_format: String,
    pub original_gguf_quant_tier: String,
    pub recovery_format: String,
    pub recovery_cache_key: String,
    pub output_format: String,
    pub output_recipe: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct RequantizeSource {
    pub path: PathBuf,
    pub cache_key: String,
    pub manifest_sha256: String,
    pub report_sha256: String,
    pub weight_sha256: String,
    pub tensor_count: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct RequantizeRuntime {
    pub mlx_lm_version: String,
    pub mlx_version: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct RequantizeRecipeReport {
    pub recipe_id: String,
    pub mode: String,
    pub bits: u8,
    pub group_size: u64,
    pub tensor_count: u64,
    pub quantized_module_count: u64,
    pub scale_tensor_count: u64,
    pub bias_tensor_count: u64,
    pub tensor_inventory_sha256: String,
    pub tensor_inventory: Vec<RequantizeTensor>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct RequantizeTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct RequantizeOutput {
    pub files: BTreeMap<String, String>,
    pub actual_output_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RequantizeManifest {
    schema_version: u32,
    status: String,
    launchable: bool,
    profile_id: String,
    cache_key: String,
    source_recovery_cache_key: String,
    recipe: MlxQuantRecipe,
    worker_sha256: String,
    profile_sha256: String,
    environment_lock_sha256: String,
    third_party_notice_sha256: String,
    worker_report_sha256: String,
    files: BTreeMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct RequantizeResult {
    pub cache_dir: PathBuf,
    pub model_dir: PathBuf,
    pub report: RequantizeReport,
    /// Always false. Inventory may display the cache as Experimental but never launch it.
    pub launchable: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentalInventoryCacheKind {
    RecoveredGguf,
    RequantizedMlx,
}

#[derive(Debug, Clone)]
pub struct ExperimentalInventorySummary {
    pub provenance: serde_json::Value,
    pub quant_type: String,
}

#[derive(Serialize)]
struct RequantizeWorkerRequest<'a> {
    schema_version: u32,
    profile_id: &'a str,
    profile_path: &'a Path,
    source_fp16: &'a Path,
    recipe_id: &'a str,
    staging_root: &'a Path,
    output_dir: &'a Path,
    cancel_path: &'a Path,
    report_path: &'a Path,
    max_output_bytes: u64,
    disk_safety_margin_bytes: u64,
}

#[derive(Serialize)]
struct WorkerRequest<'a> {
    schema_version: u32,
    profile_id: &'a str,
    profile_path: &'a Path,
    source_gguf: &'a Path,
    source_tier: &'a str,
    reference_dir: &'a Path,
    staging_root: &'a Path,
    output_dir: &'a Path,
    cancel_path: &'a Path,
    report_path: &'a Path,
    max_output_bytes: u64,
    disk_safety_margin_bytes: u64,
}

/// Run one strictly profiled recovery. Only Apple Silicon executes the worker; every
/// platform can retain/copy the resulting experimental cache as ordinary files.
pub async fn recover(
    context: RecoveryContext,
    request: RecoveryRequest,
    cancelled: Arc<AtomicBool>,
) -> Result<RecoveryResult> {
    ensure_local_execution_supported()?;
    if cancelled.load(Ordering::Acquire) {
        bail!("GGUF recovery cancelled");
    }
    let preflight_context = context.clone();
    let preflight_request = request.clone();
    let preflight =
        run_recovery_blocking(move || prepare_recovery(&preflight_context, &preflight_request))
            .await?;
    let models_root = preflight.models_root;
    let source = preflight.source;
    let reference = preflight.reference;
    let runtime_root = preflight.runtime_root;
    let worker_path = preflight.worker_path;
    let profile_path = preflight.profile_path;
    let staging_root = preflight.staging_root;
    let cache_key = preflight.cache_key;
    let final_dir = preflight.final_dir;
    if let Some(result) = preflight.cached {
        if result.report.output.actual_output_bytes > request.max_output_bytes {
            bail!("Recovered FP16 output exceeds request output bound");
        }
        return Ok(result);
    }
    let python = runtime_python(&runtime_root);
    verify_toolchain(&python, &runtime_root).await?;
    let job_id = crate::config::generate_random_token();
    let job_root = staging_root.join(format!(".recovering-{job_id}"));
    fs::create_dir(&job_root)?;
    let cleanup = CleanupDir(job_root.clone());
    let publish = job_root.join("publish");
    let output = publish.join("fp16");
    let report_path = job_root.join("worker-report.json");
    let cancel_path = job_root.join(".cancel");
    fs::create_dir(&publish)?;
    let worker_request = WorkerRequest {
        schema_version: 1,
        profile_id: PROFILE_ID,
        profile_path: &profile_path,
        source_gguf: &source,
        source_tier: request.source_tier.profile_name(),
        reference_dir: &reference,
        staging_root: &job_root,
        output_dir: &output,
        cancel_path: &cancel_path,
        report_path: &report_path,
        max_output_bytes: request.max_output_bytes,
        disk_safety_margin_bytes: request.disk_safety_margin_bytes,
    };
    atomic_json(&job_root.join("request.json"), &worker_request)?;
    let request_path = job_root.join("request.json");
    run_worker(
        &python,
        &worker_path,
        &request_path,
        &cancel_path,
        cancelled,
    )
    .await?;
    let post_source = source.clone();
    let post_reference = reference.clone();
    let post_tier = request.source_tier.clone();
    let max_output_bytes = request.max_output_bytes;
    let result = run_recovery_blocking(move || {
        let report = read_worker_report(&report_path)?;
        validate_worker_report(
            &report,
            &output,
            &post_source,
            &post_reference,
            &post_tier,
            max_output_bytes,
        )?;
        fs::copy(&report_path, publish.join("validation.json"))?;
        let files = manifest_files(&publish)?;
        let manifest = RecoveryManifest {
            schema_version: 1,
            status: "experimental_structurally_validated".into(),
            launchable: false,
            profile_id: PROFILE_ID.into(),
            cache_key: cache_key.clone(),
            source_tier: post_tier,
            worker_sha256: sha256(WORKER.as_bytes()),
            profile_sha256: sha256(PROFILE.as_bytes()),
            requirements_lock_sha256: sha256(REQUIREMENTS_LOCK.as_bytes()),
            environment_lock_sha256: sha256(ENVIRONMENT_LOCK.as_bytes()),
            third_party_notice_sha256: sha256(NOTICE.as_bytes()),
            worker_report_sha256: hash_file(&report_path)?,
            files,
        };
        atomic_json(&publish.join("manifest.json"), &manifest)?;
        fs::File::create(publish.join(".complete"))?.sync_all()?;
        reject_existing_path(&final_dir, "final cache")?;
        fs::rename(&publish, &final_dir).context("Atomic experimental cache promotion failed")?;
        drop(cleanup);
        validate_cache(&final_dir, &cache_key, max_output_bytes)
    })
    .await?;
    require_inside(&result.cache_dir, &models_root, "published cache")?;
    Ok(result)
}

/// Re-quantize the exact validated R2 FP16 cache through pinned official mlx-lm.
/// R3 output is always experimental and non-launchable.
pub async fn requantize_mlx(
    context: RecoveryContext,
    request: RequantizeRequest,
    cancelled: Arc<AtomicBool>,
) -> Result<RequantizeResult> {
    ensure_local_execution_supported()?;
    if cancelled.load(Ordering::Acquire) {
        bail!("MLX re-quantization cancelled");
    }
    let preflight_context = context.clone();
    let preflight_request = request.clone();
    let preflight =
        run_recovery_blocking(move || prepare_requantize(&preflight_context, &preflight_request))
            .await?;
    if let Some(result) = preflight.cached {
        if result.report.output.actual_output_bytes > request.max_output_bytes {
            bail!("Re-quantized MLX output exceeds request output bound");
        }
        return Ok(result);
    }
    let python = runtime_python(&preflight.runtime_root);
    verify_python_environment(
        &python,
        &preflight.runtime_root,
        REQUANT_ENVIRONMENT_LOCK,
        "MLX re-quantization Python",
    )
    .await?;
    let job_id = crate::config::generate_random_token();
    let job_root = preflight
        .staging_root
        .join(format!(".requantizing-{job_id}"));
    fs::create_dir(&job_root)?;
    let cleanup = CleanupDir(job_root.clone());
    let publish = job_root.join("publish");
    let output = publish.join("model");
    let report_path = job_root.join("worker-report.json");
    let cancel_path = job_root.join(".cancel");
    fs::create_dir(&publish)?;
    let worker_request = RequantizeWorkerRequest {
        schema_version: 1,
        profile_id: REQUANT_PROFILE_ID,
        profile_path: &preflight.profile_path,
        source_fp16: &preflight.source.fp16_dir,
        recipe_id: request.recipe.profile_name(),
        staging_root: &job_root,
        output_dir: &output,
        cancel_path: &cancel_path,
        report_path: &report_path,
        max_output_bytes: request.max_output_bytes,
        disk_safety_margin_bytes: request.disk_safety_margin_bytes,
    };
    atomic_json(&job_root.join("request.json"), &worker_request)?;
    run_bounded_worker(
        &python,
        &preflight.worker_path,
        &job_root.join("request.json"),
        &cancel_path,
        cancelled,
        "MLX re-quantization",
        REQUANTIZATION_TIMEOUT,
    )
    .await?;
    let recipe = request.recipe.clone();
    let max_output_bytes = request.max_output_bytes;
    let source_cache = preflight.source.cache_dir.clone();
    let cache_key = preflight.cache_key.clone();
    let final_dir = preflight.final_dir.clone();
    let result = run_recovery_blocking(move || {
        let report = read_requantize_report(&report_path)?;
        validate_requantize_report(&report, &output, &source_cache, &recipe, max_output_bytes)?;
        fs::copy(&report_path, publish.join("validation.json"))?;
        let files = manifest_files(&publish)?;
        let manifest = RequantizeManifest {
            schema_version: 1,
            status: "experimental_requantized_structurally_validated".into(),
            launchable: false,
            profile_id: REQUANT_PROFILE_ID.into(),
            cache_key: cache_key.clone(),
            source_recovery_cache_key: R2_F16_CACHE_KEY.into(),
            recipe,
            worker_sha256: sha256(REQUANT_WORKER.as_bytes()),
            profile_sha256: sha256(REQUANT_PROFILE.as_bytes()),
            environment_lock_sha256: sha256(REQUANT_ENVIRONMENT_LOCK.as_bytes()),
            third_party_notice_sha256: sha256(REQUANT_NOTICE.as_bytes()),
            worker_report_sha256: hash_file(&report_path)?,
            files,
        };
        atomic_json(&publish.join("manifest.json"), &manifest)?;
        fs::File::create(publish.join(".complete"))?.sync_all()?;
        reject_existing_path(&final_dir, "final re-quantized cache")?;
        fs::rename(&publish, &final_dir)
            .context("Atomic experimental re-quantized cache promotion failed")?;
        drop(cleanup);
        validate_requantize_cache(&final_dir, &cache_key, max_output_bytes)
    })
    .await?;
    Ok(result)
}

struct RequantizePreflight {
    source: RecoveryResult,
    runtime_root: PathBuf,
    worker_path: PathBuf,
    profile_path: PathBuf,
    staging_root: PathBuf,
    cache_key: String,
    final_dir: PathBuf,
    cached: Option<RequantizeResult>,
}

fn prepare_requantize(
    context: &RecoveryContext,
    request: &RequantizeRequest,
) -> Result<RequantizePreflight> {
    let models_root = canonical_directory(&context.models_dir, "models_dir")?;
    let config_root = canonical_directory(&context.config_dir, "config_dir")?;
    let source_cache =
        canonical_library_directory(&models_root, &request.recovered_cache, "recovered R2 cache")?;
    if source_cache.file_name().and_then(OsStr::to_str) != Some(R2_F16_CACHE_KEY) {
        bail!("R3 accepts only the exact remediated R2 F16 cache");
    }
    let source = validate_cache(&source_cache, R2_F16_CACHE_KEY, u64::MAX)?;
    let runtime_root = canonical_library_directory(
        &config_root,
        Path::new("runtimes/rapid-mlx/.staging/0.10.10-qualification"),
        "pinned MLX runtime",
    )?;
    let tool_root = create_managed_directory(
        &config_root,
        &Path::new("runtimes/mlx-requantize").join(REQUANT_TOOLCHAIN_VERSION),
        "re-quantization tool root",
    )?;
    let worker_relative = Path::new("worker").join(&requantize_worker_asset_key()[..16]);
    let worker_root =
        create_managed_directory(&tool_root, &worker_relative, "re-quantization worker root")?;
    let worker_path =
        materialize_exact(&worker_root.join("quantize.py"), REQUANT_WORKER.as_bytes())?;
    let profile_path = materialize_exact(
        &worker_root.join("profile.json"),
        REQUANT_PROFILE.as_bytes(),
    )?;
    materialize_exact(
        &worker_root.join("environment.lock.json"),
        REQUANT_ENVIRONMENT_LOCK.as_bytes(),
    )?;
    materialize_exact(
        &worker_root.join("THIRD_PARTY_NOTICE.md"),
        REQUANT_NOTICE.as_bytes(),
    )?;
    let cache_root = create_managed_directory(
        &models_root,
        Path::new("rapid-mlx/requantized"),
        "re-quantized cache root",
    )?;
    let staging_root = create_managed_directory(
        &cache_root,
        Path::new(".staging"),
        "re-quantized staging root",
    )?;
    let cache_key = requantize_cache_key(&request.recipe);
    let final_dir = cache_root.join(&cache_key);
    let cached = match fs::symlink_metadata(&final_dir) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                bail!("Final re-quantized cache path must be a non-symlink directory");
            }
            Some(validate_requantize_cache(
                &final_dir,
                &cache_key,
                request.max_output_bytes,
            )?)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };
    Ok(RequantizePreflight {
        source,
        runtime_root,
        worker_path,
        profile_path,
        staging_root,
        cache_key,
        final_dir,
        cached,
    })
}

struct RecoveryPreflight {
    models_root: PathBuf,
    source: PathBuf,
    reference: PathBuf,
    runtime_root: PathBuf,
    worker_path: PathBuf,
    profile_path: PathBuf,
    staging_root: PathBuf,
    cache_key: String,
    final_dir: PathBuf,
    cached: Option<RecoveryResult>,
}

fn prepare_recovery(
    context: &RecoveryContext,
    request: &RecoveryRequest,
) -> Result<RecoveryPreflight> {
    let models_root = canonical_directory(&context.models_dir, "models_dir")?;
    let config_root = canonical_directory(&context.config_dir, "config_dir")?;
    let source = canonical_library_file(&models_root, &request.source_gguf, "source GGUF")?;
    if source
        .extension()
        .and_then(OsStr::to_str)
        .is_none_or(|value| !value.eq_ignore_ascii_case("gguf"))
    {
        bail!("Recovery source must be a .gguf file");
    }
    let reference =
        canonical_library_directory(&models_root, &request.reference_dir, "reference directory")?;
    let runtime_root = create_managed_directory(
        &config_root,
        Path::new("runtimes/gguf-recovery")
            .join(TOOLCHAIN_VERSION)
            .as_path(),
        "recovery runtime",
    )?;
    let worker_relative = Path::new("worker").join(&worker_asset_key()[..16]);
    let worker_root = create_managed_directory(&runtime_root, &worker_relative, "worker root")?;
    let worker_path = materialize_exact(&worker_root.join("converter.py"), WORKER.as_bytes())?;
    let profile_path = materialize_exact(&worker_root.join("profile.json"), PROFILE.as_bytes())?;
    materialize_exact(
        &worker_root.join("requirements.lock"),
        REQUIREMENTS_LOCK.as_bytes(),
    )?;
    materialize_exact(
        &worker_root.join("environment.lock.json"),
        ENVIRONMENT_LOCK.as_bytes(),
    )?;
    materialize_exact(
        &worker_root.join("THIRD_PARTY_NOTICE.md"),
        NOTICE.as_bytes(),
    )?;
    let imports_root =
        create_managed_directory(&models_root, Path::new("rapid-mlx/imports"), "imports root")?;
    let staging_root =
        create_managed_directory(&imports_root, Path::new(".staging"), "import staging root")?;
    let source_hash = hash_file(&source)?;
    let cache_key = cache_key(request.source_tier.clone(), &source_hash);
    let final_dir = imports_root.join(&cache_key);
    let cached = match fs::symlink_metadata(&final_dir) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                bail!("Final cache path must be an app-owned non-symlink directory");
            }
            Some(validate_cache(
                &final_dir,
                &cache_key,
                request.max_output_bytes,
            )?)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };
    Ok(RecoveryPreflight {
        models_root,
        source,
        reference,
        runtime_root,
        worker_path,
        profile_path,
        staging_root,
        cache_key,
        final_dir,
        cached,
    })
}

async fn run_recovery_blocking<T, F>(operation: F) -> Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    let permit = RECOVERY_BLOCKING_GATE
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| anyhow!("Recovery blocking worker gate is unavailable"))?;
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        operation()
    })
    .await
    .context("Recovery blocking worker failed")?
}

fn ensure_local_execution_supported() -> Result<()> {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        Ok(())
    } else {
        bail!(
            "Local GGUF recovery requires Apple Silicon; cross-platform file management remains available"
        )
    }
}

fn canonical_directory(path: &Path, kind: &str) -> Result<PathBuf> {
    let canonical = path
        .canonicalize()
        .with_context(|| format!("Cannot resolve {kind} '{}'", path.display()))?;
    if !canonical.is_dir() {
        bail!("{kind} must be a directory");
    }
    Ok(canonical)
}

fn validate_relative(path: &Path, kind: &str) -> Result<()> {
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path.components().any(|part| {
            matches!(
                part,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
        || path.to_string_lossy().starts_with(['/', '\\'])
    {
        bail!("{kind} must be a traversal-free library-relative path");
    }
    Ok(())
}

fn canonical_library_file(root: &Path, relative: &Path, kind: &str) -> Result<PathBuf> {
    validate_relative(relative, kind)?;
    let canonical = reject_symlinked_child(root, relative, kind)?;
    if !canonical.is_file() {
        bail!("{kind} must be a regular file");
    }
    Ok(canonical)
}

fn canonical_library_directory(root: &Path, relative: &Path, kind: &str) -> Result<PathBuf> {
    validate_relative(relative, kind)?;
    let canonical = reject_symlinked_child(root, relative, kind)?;
    if !canonical.is_dir() {
        bail!("{kind} must be a directory");
    }
    Ok(canonical)
}

fn reject_symlinked_child(root: &Path, relative: &Path, kind: &str) -> Result<PathBuf> {
    let mut current = root.to_path_buf();
    for component in relative.components() {
        if matches!(component, Component::CurDir) {
            continue;
        }
        current.push(component);
        let metadata = fs::symlink_metadata(&current)
            .with_context(|| format!("Cannot inspect {kind} component '{}'", current.display()))?;
        if metadata.file_type().is_symlink() {
            bail!("{kind} must not contain symlinked components");
        }
    }
    let canonical = current.canonicalize()?;
    require_inside(&canonical, root, kind)?;
    Ok(canonical)
}

fn require_inside(path: &Path, root: &Path, kind: &str) -> Result<()> {
    if path == root || path.starts_with(root) {
        Ok(())
    } else {
        bail!("{kind} escapes its canonical root")
    }
}

fn create_managed_directory(root: &Path, relative: &Path, kind: &str) -> Result<PathBuf> {
    validate_relative(relative, kind)?;
    let canonical_root = root.canonicalize()?;
    let mut current = canonical_root.clone();
    for component in relative.components() {
        if matches!(component, Component::CurDir) {
            continue;
        }
        current.push(component);
        match fs::symlink_metadata(&current) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() || !metadata.is_dir() {
                    bail!("{kind} contains a symlink or non-directory component");
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                fs::create_dir(&current)
                    .with_context(|| format!("Cannot create {kind} '{}'", current.display()))?;
            }
            Err(error) => return Err(error.into()),
        }
        let canonical = current.canonicalize()?;
        require_inside(&canonical, &canonical_root, kind)?;
        current = canonical;
    }
    Ok(current)
}

fn reject_existing_path(path: &Path, kind: &str) -> Result<()> {
    match fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Ok(_) => bail!("{kind} already exists"),
        Err(error) => Err(error.into()),
    }
}

fn runtime_python(runtime_root: &Path) -> PathBuf {
    if cfg!(windows) {
        runtime_root.join("venv/Scripts/python.exe")
    } else {
        runtime_root.join("venv/bin/python")
    }
}

fn materialize_exact(path: &Path, content: &[u8]) -> Result<PathBuf> {
    if let Ok(metadata) = fs::symlink_metadata(path) {
        if metadata.file_type().is_symlink()
            || !metadata.is_file()
            || hash_file(path)? != sha256(content)
        {
            bail!(
                "Managed recovery asset differs from embedded identity: {}",
                path.display()
            );
        }
        return Ok(path.canonicalize()?);
    }
    let temporary = path.with_extension("tmp");
    reject_existing_path(&temporary, "managed recovery asset temporary path")?;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    file.write_all(content)?;
    file.sync_all()?;
    fs::rename(&temporary, path)?;
    Ok(path.canonicalize()?)
}

async fn verify_toolchain(python: &Path, runtime_root: &Path) -> Result<()> {
    verify_python_environment(python, runtime_root, ENVIRONMENT_LOCK, "recovery Python").await
}

async fn verify_python_environment(
    python: &Path,
    runtime_root: &Path,
    environment_lock: &str,
    kind: &str,
) -> Result<()> {
    if !python.is_file() {
        bail!("Pinned {kind} is not installed: {}", python.display());
    }
    // A venv's python launcher is normally a symlink to its base interpreter. The
    // execution path itself must be app-owned, and the live `sys.prefix` probe below
    // proves that the interpreter is isolated in this exact managed venv.
    require_inside(python, runtime_root, "recovery Python")?;
    let script = r#"import hashlib,importlib.metadata as m,json,sys
h=hashlib.sha256(); ds=sorted(m.distributions(),key=lambda d:d.metadata['Name'].lower()); packages={}; count=0
for d in ds:
 n=d.metadata['Name'].lower(); packages[n]=d.version
 for f in sorted(d.files or [],key=str):
  p=d.locate_file(f)
  if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':
   b=p.read_bytes(); h.update(n.encode()+b'\0'+str(f).encode()+b'\0'+hashlib.sha256(b).digest()); count+=1
print(json.dumps({'prefix':sys.prefix,'packages':packages,'file_count':count,'environment_sha256':h.hexdigest()},sort_keys=True))"#;
    let mut command = tokio::process::Command::new(python);
    command
        .args(["-I", "-c", script])
        .env_clear()
        .env("PATH", "")
        .env("PYTHONNOUSERSITE", "1")
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("HF_HUB_DISABLE_TELEMETRY", "1")
        .env("RAPID_MLX_TELEMETRY", "0")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    configure_process_group(&mut command)?;
    let mut child = command.spawn()?;
    let child_id = child.id();
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("Toolchain probe stdout unavailable"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("Toolchain probe stderr unavailable"))?;
    let overflow = Arc::new(AtomicBool::new(false));
    let out_task = tokio::spawn(drain_bounded(
        stdout,
        MAX_TOOLCHAIN_PROBE_BYTES,
        Arc::clone(&overflow),
    ));
    let err_task = tokio::spawn(drain_bounded(
        stderr,
        MAX_TOOLCHAIN_PROBE_BYTES,
        Arc::clone(&overflow),
    ));
    let status = match tokio::time::timeout(TOOLCHAIN_PROBE_TIMEOUT, child.wait()).await {
        Ok(status) => status?,
        Err(_) => {
            terminate_process_tree(&mut child, child_id).await;
            bail!("Pinned {kind} dependency probe timed out");
        }
    };
    let (output, error) =
        finish_pipe_drains(out_task, err_task, &mut child, child_id, "toolchain probe").await?;
    if !status.success() || overflow.load(Ordering::Acquire) || error.total_bytes != 0 {
        bail!("Pinned {kind} dependency probe failed");
    }
    #[derive(Deserialize)]
    struct Probe {
        prefix: PathBuf,
        packages: BTreeMap<String, String>,
        file_count: u64,
        environment_sha256: String,
    }
    #[derive(Deserialize)]
    struct EnvironmentLock {
        schema_version: u32,
        packages: BTreeMap<String, String>,
        file_count: u64,
        environment_sha256: String,
    }
    let probe: Probe = serde_json::from_slice(&output.retained)?;
    let expected: EnvironmentLock = serde_json::from_str(environment_lock)?;
    if probe.prefix.canonicalize()? != runtime_root.join("venv").canonicalize()? {
        bail!("{kind} is not isolated in the managed virtual environment");
    }
    if expected.schema_version != 1
        || probe.packages != expected.packages
        || probe.file_count != expected.file_count
        || probe.environment_sha256 != expected.environment_sha256
    {
        bail!("{kind} dependencies differ from the hash-locked toolchain");
    }
    Ok(())
}

struct BoundedDrain {
    retained: Vec<u8>,
    total_bytes: u64,
}

async fn drain_bounded(
    mut reader: impl tokio::io::AsyncRead + Unpin,
    limit: usize,
    overflow: Arc<AtomicBool>,
) -> Result<BoundedDrain> {
    use tokio::io::AsyncReadExt;
    let mut retained = Vec::new();
    let mut total_bytes = 0u64;
    let mut buffer = [0u8; 8192];
    loop {
        let count = reader.read(&mut buffer).await?;
        if count == 0 {
            break;
        }
        total_bytes = total_bytes.saturating_add(count as u64);
        if total_bytes > limit as u64 {
            overflow.store(true, Ordering::Release);
        }
        if retained.len() < limit {
            let keep = (limit - retained.len()).min(count);
            retained.extend_from_slice(&buffer[..keep]);
        }
    }
    Ok(BoundedDrain {
        retained,
        total_bytes,
    })
}

#[cfg(unix)]
fn configure_process_group(command: &mut tokio::process::Command) -> Result<()> {
    use std::os::unix::process::CommandExt;
    unsafe {
        command.as_std_mut().pre_exec(|| {
            if unix_process::set_own_process_group() == 0 {
                Ok(())
            } else {
                Err(std::io::Error::last_os_error())
            }
        });
    }
    Ok(())
}

#[cfg(not(unix))]
fn configure_process_group(_command: &mut tokio::process::Command) -> Result<()> {
    Ok(())
}

#[cfg(unix)]
mod unix_process {
    unsafe extern "C" {
        fn setpgid(pid: i32, pgid: i32) -> i32;
        fn kill(pid: i32, signal: i32) -> i32;
    }
    pub fn set_own_process_group() -> i32 {
        unsafe { setpgid(0, 0) }
    }
    pub fn signal_group(pid: u32, signal: i32) {
        unsafe {
            let _ = kill(-(pid as i32), signal);
        }
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub fn process_exists(pid: u32) -> bool {
        let result = unsafe { kill(pid as i32, 0) };
        result == 0 || std::io::Error::last_os_error().raw_os_error() != Some(3)
    }
}

async fn terminate_process_tree(child: &mut tokio::process::Child, child_id: Option<u32>) {
    #[cfg(not(unix))]
    let _ = child_id;
    #[cfg(unix)]
    if let Some(pid) = child_id {
        unix_process::signal_group(pid, 15);
    }
    tokio::time::sleep(PROCESS_GROUP_GRACE).await;
    #[cfg(unix)]
    if let Some(pid) = child_id {
        unix_process::signal_group(pid, 9);
    }
    let _ = child.kill().await;
    let _ = child.wait().await;
}

async fn finish_pipe_drains(
    mut stdout_task: tokio::task::JoinHandle<Result<BoundedDrain>>,
    mut stderr_task: tokio::task::JoinHandle<Result<BoundedDrain>>,
    child: &mut tokio::process::Child,
    child_id: Option<u32>,
    kind: &str,
) -> Result<(BoundedDrain, BoundedDrain)> {
    let join = async { tokio::try_join!(&mut stdout_task, &mut stderr_task) };
    let pair = match tokio::time::timeout(PIPE_DRAIN_TIMEOUT, join).await {
        Ok(pair) => pair.context("Subprocess pipe drain task failed")?,
        Err(_) => {
            terminate_process_tree(child, child_id).await;
            tokio::time::timeout(PIPE_DRAIN_TIMEOUT, async {
                tokio::try_join!(&mut stdout_task, &mut stderr_task)
            })
            .await
            .with_context(|| format!("{kind} pipes remained open after process-group teardown"))?
            .context("Subprocess pipe drain task failed")?
        }
    };
    Ok((pair.0?, pair.1?))
}

async fn run_worker(
    python: &Path,
    worker: &Path,
    request: &Path,
    cancel_path: &Path,
    cancelled: Arc<AtomicBool>,
) -> Result<()> {
    run_bounded_worker(
        python,
        worker,
        request,
        cancel_path,
        cancelled,
        "GGUF recovery",
        CONVERSION_TIMEOUT,
    )
    .await
}

async fn run_bounded_worker(
    python: &Path,
    worker: &Path,
    request: &Path,
    cancel_path: &Path,
    cancelled: Arc<AtomicBool>,
    kind: &str,
    timeout: std::time::Duration,
) -> Result<()> {
    let mut command = tokio::process::Command::new(python);
    command
        .arg("-I")
        .arg(worker)
        .arg("--request")
        .arg(request)
        .env_clear()
        .env("PATH", "")
        .env("PYTHONNOUSERSITE", "1")
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("HF_HUB_DISABLE_TELEMETRY", "1")
        .env("RAPID_MLX_TELEMETRY", "0")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    configure_process_group(&mut command)?;
    let mut child = command.spawn()?;
    let child_id = child.id();
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("Worker stdout unavailable"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("Worker stderr unavailable"))?;
    let overflow = Arc::new(AtomicBool::new(false));
    let stdout_task = tokio::spawn(drain_bounded(
        stdout,
        MAX_DIAGNOSTIC_BYTES,
        Arc::clone(&overflow),
    ));
    let stderr_task = tokio::spawn(drain_bounded(
        stderr,
        MAX_DIAGNOSTIC_BYTES,
        Arc::clone(&overflow),
    ));
    let deadline = tokio::time::Instant::now() + timeout;
    let status = loop {
        if cancelled.load(Ordering::Acquire) {
            fs::write(cancel_path, b"cancel")?;
            terminate_process_tree(&mut child, child_id).await;
            bail!("{kind} cancelled");
        }
        if overflow.load(Ordering::Acquire) {
            terminate_process_tree(&mut child, child_id).await;
            bail!("{kind} worker exceeded the diagnostic output bound");
        }
        if tokio::time::Instant::now() >= deadline {
            terminate_process_tree(&mut child, child_id).await;
            bail!("{kind} worker timed out");
        }
        if let Some(status) = child.try_wait()? {
            break status;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    };
    let (out, err) = finish_pipe_drains(
        stdout_task,
        stderr_task,
        &mut child,
        child_id,
        &format!("{kind} worker"),
    )
    .await?;
    if out.total_bytes > MAX_DIAGNOSTIC_BYTES as u64
        || err.total_bytes > MAX_DIAGNOSTIC_BYTES as u64
    {
        bail!("{kind} worker exceeded the diagnostic output bound");
    }
    if !status.success() {
        let detail = String::from_utf8_lossy(&err.retained);
        bail!(
            "{kind} worker failed with {status}: {}",
            detail.chars().take(1000).collect::<String>()
        );
    }
    Ok(())
}

fn read_worker_report(path: &Path) -> Result<WorkerReport> {
    require_regular_file_bounded(path, MAX_REPORT_BYTES, "worker report")?;
    Ok(serde_json::from_reader(fs::File::open(path)?)?)
}

fn validate_worker_report(
    report: &WorkerReport,
    output: &Path,
    source: &Path,
    reference: &Path,
    tier: &RecoveryTier,
    max_output_bytes: u64,
) -> Result<()> {
    let profile: RecoveryProfile = serde_json::from_str(PROFILE)?;
    let tier_name = tier.profile_name();
    let expected_tier = profile
        .gguf_source
        .tiers
        .get(tier_name)
        .ok_or_else(|| anyhow!("Recovery tier is absent from embedded profile"))?;
    if report.schema_version != 1
        || report.worker_version != WORKER_VERSION
        || report.status != "recovered"
        || report.profile_id != profile.profile_id
        || profile.profile_id != PROFILE_ID
        || report.output.dtype != profile.output_dtype
        || report.output.tensor_count != profile.expected_tensor_count
        || report.tensor_inventory.len() as u64 != report.output.tensor_count
        || !failure_counters_are_zero(report)
    {
        bail!("Worker report failed strict R2 closure validation");
    }
    let tensor_inventory_sha = sha256(&serde_json::to_vec(&serde_json::to_value(
        &report.tensor_inventory,
    )?)?);
    if report.source.path != source
        || report.source.tier != tier_name
        || report.source.size_bytes != expected_tier.size
        || report.source.sha256 != expected_tier.sha256
        || report.source.architecture != profile.architecture
        || report.source.quant_inventory != expected_tier.quant_inventory
        || report.source.tensor_count != profile.expected_tensor_count
        || report.source.tensor_inventory_sha256 != expected_tier.tensor_inventory_sha256
        || tensor_inventory_sha != expected_tier.tensor_inventory_sha256
        || source.file_name().and_then(OsStr::to_str) != Some(expected_tier.filename.as_str())
        || fs::metadata(source)?.len() != expected_tier.size
        || hash_file(source)? != expected_tier.sha256
    {
        bail!("Worker source provenance differs from the exact profile identity");
    }
    let reference_manifest_path = reject_symlinked_child(
        reference,
        Path::new("reference-manifest.json"),
        "reference manifest",
    )?;
    require_regular_file_bounded(
        &reference_manifest_path,
        MAX_REFERENCE_MANIFEST_BYTES,
        "reference manifest",
    )?;
    let reference_manifest: serde_json::Value =
        serde_json::from_reader(fs::File::open(reference_manifest_path)?)?;
    let expected_reference_files: BTreeMap<String, String> = serde_json::from_value(
        reference_manifest
            .get("files")
            .cloned()
            .ok_or_else(|| anyhow!("Reference manifest has no file closure"))?,
    )?;
    if report.authoritative_reference.repo_id != profile.authoritative_source.repo_id
        || report.authoritative_reference.revision != profile.authoritative_source.revision
        || reference_manifest
            .get("repo_id")
            .and_then(serde_json::Value::as_str)
            != Some(profile.authoritative_source.repo_id.as_str())
        || reference_manifest
            .get("revision")
            .and_then(serde_json::Value::as_str)
            != Some(profile.authoritative_source.revision.as_str())
        || report.authoritative_reference.tensor_count != profile.expected_tensor_count
        || report.authoritative_reference.files != expected_reference_files
        || expected_reference_files != profile.authoritative_source.files
        || expected_reference_files.get(&profile.authoritative_source.weight_file)
            != Some(&profile.authoritative_source.weight_sha256)
    {
        bail!("Worker authoritative-reference provenance is invalid");
    }
    for (name, expected_hash) in &expected_reference_files {
        validate_relative(Path::new(name), "reference asset")?;
        let path = reject_symlinked_child(reference, Path::new(name), "reference asset")?;
        if !path.is_file() || hash_file(&path)? != *expected_hash {
            bail!("Authoritative reference changed during recovery");
        }
    }
    let actual = flat_files(output)?;
    if actual != report.output.files {
        bail!("Recovered output differs from the worker's complete file inventory");
    }
    let actual_output_bytes = flat_directory_bytes(output)?;
    if report.output.actual_output_bytes != actual_output_bytes
        || actual_output_bytes > max_output_bytes
        || report.output.estimated_weight_bytes > actual_output_bytes
    {
        bail!("Recovered output byte closure exceeds or differs from its bound");
    }
    Ok(())
}

fn failure_counters_are_zero(report: &WorkerReport) -> bool {
    report.skipped_tensors == 0
        && report.unknown_tensors == 0
        && report.duplicate_tensors == 0
        && report.shape_mismatches == 0
        && report.non_finite_tensors == 0
}

fn cache_key(tier: RecoveryTier, source_hash: &str) -> String {
    let mut digest = Sha256::new();
    for value in [
        PROFILE_ID,
        tier.profile_name(),
        source_hash,
        &sha256(WORKER.as_bytes()),
        &sha256(PROFILE.as_bytes()),
        &sha256(REQUIREMENTS_LOCK.as_bytes()),
        &sha256(ENVIRONMENT_LOCK.as_bytes()),
        &sha256(NOTICE.as_bytes()),
    ] {
        digest.update(value.as_bytes());
        digest.update([0]);
    }
    hex_digest(digest.finalize().as_slice())
}

fn worker_asset_key() -> String {
    let mut digest = Sha256::new();
    for content in [
        WORKER.as_bytes(),
        PROFILE.as_bytes(),
        REQUIREMENTS_LOCK.as_bytes(),
        ENVIRONMENT_LOCK.as_bytes(),
        NOTICE.as_bytes(),
    ] {
        digest.update(sha256(content).as_bytes());
        digest.update([0]);
    }
    hex_digest(digest.finalize().as_slice())
}

fn requantize_worker_asset_key() -> String {
    let mut digest = Sha256::new();
    for content in [
        REQUANT_WORKER.as_bytes(),
        REQUANT_PROFILE.as_bytes(),
        REQUANT_ENVIRONMENT_LOCK.as_bytes(),
        REQUANT_NOTICE.as_bytes(),
    ] {
        digest.update(sha256(content).as_bytes());
        digest.update([0]);
    }
    hex_digest(digest.finalize().as_slice())
}

fn requantize_cache_key(recipe: &MlxQuantRecipe) -> String {
    let mut digest = Sha256::new();
    for value in [
        REQUANT_PROFILE_ID,
        R2_F16_CACHE_KEY,
        recipe.profile_name(),
        &sha256(REQUANT_WORKER.as_bytes()),
        &sha256(REQUANT_PROFILE.as_bytes()),
        &sha256(REQUANT_ENVIRONMENT_LOCK.as_bytes()),
        &sha256(REQUANT_NOTICE.as_bytes()),
    ] {
        digest.update(value.as_bytes());
        digest.update([0]);
    }
    hex_digest(digest.finalize().as_slice())
}

fn read_requantize_report(path: &Path) -> Result<RequantizeReport> {
    require_regular_file_bounded(path, MAX_REPORT_BYTES, "re-quantization worker report")?;
    Ok(serde_json::from_reader(fs::File::open(path)?)?)
}

fn validate_requantize_report(
    report: &RequantizeReport,
    output: &Path,
    source_cache: &Path,
    recipe: &MlxQuantRecipe,
    max_output_bytes: u64,
) -> Result<()> {
    let expected_labels = RequantizeLabels {
        original_format: "gguf".into(),
        original_gguf_quant_tier: "f16".into(),
        recovery_format: "mlx_fp16_safetensors".into(),
        recovery_cache_key: R2_F16_CACHE_KEY.into(),
        output_format: "mlx_quantized_safetensors".into(),
        output_recipe: recipe.profile_name().into(),
    };
    let expected_source = RequantizeSource {
        path: source_cache.join("fp16"),
        cache_key: R2_F16_CACHE_KEY.into(),
        manifest_sha256: "9861d40f1e7c5582dac23b667a223fd29adeb4732021bb3333846280a92bcb6e".into(),
        report_sha256: "d125dc3dc078a3b7e3e376ab3fe3213edecbaf0870bb374e549df13b9e402aaf".into(),
        weight_sha256: "b9f93ce41c270feb4eb9a8870f0b48b3fd2d761bd6af47f4d74fe35d991f9bf5".into(),
        tensor_count: 272,
    };
    let expected_runtime = RequantizeRuntime {
        mlx_lm_version: "0.31.3".into(),
        mlx_version: "0.32.0".into(),
    };
    if report.schema_version != 1
        || report.worker_version != REQUANT_WORKER_VERSION
        || report.status != "requantized"
        || report.profile_id != REQUANT_PROFILE_ID
        || report.launchable
        || report.labels != expected_labels
        || report.source != expected_source
        || report.runtime != expected_runtime
        || report.recipe.recipe_id != recipe.profile_name()
        || report.recipe.mode != "affine"
        || report.recipe.bits != recipe.bits()
        || report.recipe.group_size != 64
        || report.recipe.tensor_count != REQUANTIZED_TENSOR_COUNT
        || report.recipe.quantized_module_count != REQUANTIZED_MODULE_COUNT
        || report.recipe.quantized_module_count != report.recipe.scale_tensor_count
        || report.recipe.scale_tensor_count != report.recipe.bias_tensor_count
        || report.recipe.tensor_inventory.len() as u64 != report.recipe.tensor_count
        || report.recipe.tensor_inventory_sha256
            != sha256(&serde_json::to_vec(&report.recipe.tensor_inventory)?)
        || report.recipe.tensor_inventory_sha256 != recipe.expected_tensor_inventory_sha256()
        || !report.error.is_empty()
        || !report.elapsed_seconds.is_finite()
        || report.elapsed_seconds < 0.0
    {
        bail!("Re-quantization worker report failed strict R3 closure validation");
    }
    let actual = manifest_files(output)?;
    let actual_bytes = recursive_directory_bytes(output)?;
    if actual != report.output.files
        || actual_bytes != report.output.actual_output_bytes
        || actual_bytes > max_output_bytes
    {
        bail!("Re-quantized output differs from its complete bounded closure");
    }
    let config_path = output.join("config.json");
    require_regular_file_bounded(&config_path, MAX_REPORT_BYTES, "re-quantized config")?;
    let config: serde_json::Value = serde_json::from_reader(fs::File::open(config_path)?)?;
    let expected_quantization = serde_json::json!({
        "group_size": 64,
        "bits": recipe.bits(),
        "mode": "affine"
    });
    if config.get("quantization") != Some(&expected_quantization)
        || config.get("quantization_config") != Some(&expected_quantization)
    {
        bail!("Re-quantized config differs from the exact R3 recipe");
    }
    validate_cache(source_cache, R2_F16_CACHE_KEY, u64::MAX)?;
    Ok(())
}

fn validate_requantize_cache(
    path: &Path,
    expected_key: &str,
    max_output_bytes: u64,
) -> Result<RequantizeResult> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        bail!("Experimental re-quantized cache root must be a non-symlink directory");
    }
    let manifest_path = path.join("manifest.json");
    require_regular_file_bounded(
        &manifest_path,
        MAX_MANIFEST_BYTES,
        "re-quantized cache manifest",
    )?;
    let manifest: RequantizeManifest = serde_json::from_reader(fs::File::open(&manifest_path)?)?;
    let complete = path.join(".complete");
    require_regular_file_bounded(&complete, 0, "re-quantized completion marker")?;
    if manifest.schema_version != 1
        || manifest.status != "experimental_requantized_structurally_validated"
        || manifest.launchable
        || manifest.profile_id != REQUANT_PROFILE_ID
        || manifest.cache_key != expected_key
        || manifest.cache_key != requantize_cache_key(&manifest.recipe)
        || manifest.source_recovery_cache_key != R2_F16_CACHE_KEY
        || manifest.worker_sha256 != sha256(REQUANT_WORKER.as_bytes())
        || manifest.profile_sha256 != sha256(REQUANT_PROFILE.as_bytes())
        || manifest.environment_lock_sha256 != sha256(REQUANT_ENVIRONMENT_LOCK.as_bytes())
        || manifest.third_party_notice_sha256 != sha256(REQUANT_NOTICE.as_bytes())
    {
        bail!("Experimental re-quantized cache provenance is invalid");
    }
    let report_path = path.join("validation.json");
    require_regular_file_bounded(
        &report_path,
        MAX_REPORT_BYTES,
        "re-quantized validation report",
    )?;
    if hash_file(&report_path)? != manifest.worker_report_sha256 {
        bail!("Experimental re-quantized validation report was modified");
    }
    let expected_files = manifest_files(path)?;
    if expected_files != manifest.files {
        bail!("Experimental re-quantized cache contents differ from its manifest");
    }
    let report = read_requantize_report(&report_path)?;
    let models_root = path
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .ok_or_else(|| anyhow!("Re-quantized cache has no models root"))?;
    let source_cache = models_root.join("rapid-mlx/imports").join(R2_F16_CACHE_KEY);
    validate_requantize_report(
        &report,
        &path.join("model"),
        &source_cache,
        &manifest.recipe,
        max_output_bytes,
    )?;
    Ok(RequantizeResult {
        cache_dir: path.to_path_buf(),
        model_dir: path.join("model"),
        report,
        launchable: false,
    })
}

/// Validate the immutable, non-launchable cache envelope before exposing it in the
/// model inventory. This verifies the completion sentinel, typed provenance, embedded
/// worker identities, validation-report identity, and every published file hash. It
/// returns a sanitized summary rather than the user-controlled manifest JSON.
pub fn validate_experimental_inventory_cache(
    path: &Path,
    kind: ExperimentalInventoryCacheKind,
) -> Result<ExperimentalInventorySummary> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        bail!("Experimental cache root must be a non-symlink directory");
    }
    let cache_key_name = path
        .file_name()
        .and_then(OsStr::to_str)
        .ok_or_else(|| anyhow!("Experimental cache has no UTF-8 cache key"))?;
    let complete = path.join(".complete");
    require_regular_file_bounded(&complete, 0, "experimental cache completion marker")?;

    match kind {
        ExperimentalInventoryCacheKind::RecoveredGguf => {
            let manifest_path = path.join("manifest.json");
            require_regular_file_bounded(
                &manifest_path,
                MAX_MANIFEST_BYTES,
                "recovery cache manifest",
            )?;
            let manifest: RecoveryManifest =
                serde_json::from_reader(fs::File::open(manifest_path)?)?;
            let profile: RecoveryProfile = serde_json::from_str(PROFILE)?;
            let tier = profile
                .gguf_source
                .tiers
                .get(manifest.source_tier.profile_name())
                .ok_or_else(|| anyhow!("Recovery cache tier is absent from its profile"))?;
            let expected_key = cache_key(manifest.source_tier.clone(), &tier.sha256);
            if manifest.schema_version != 1
                || manifest.status != "experimental_structurally_validated"
                || manifest.launchable
                || manifest.profile_id != PROFILE_ID
                || manifest.cache_key != cache_key_name
                || manifest.cache_key != expected_key
                || manifest.worker_sha256 != sha256(WORKER.as_bytes())
                || manifest.profile_sha256 != sha256(PROFILE.as_bytes())
                || manifest.requirements_lock_sha256 != sha256(REQUIREMENTS_LOCK.as_bytes())
                || manifest.environment_lock_sha256 != sha256(ENVIRONMENT_LOCK.as_bytes())
                || manifest.third_party_notice_sha256 != sha256(NOTICE.as_bytes())
            {
                bail!("Experimental recovery cache provenance is invalid");
            }
            validate_inventory_file_closure(path, &manifest.files, &manifest.worker_report_sha256)?;
            Ok(ExperimentalInventorySummary {
                provenance: serde_json::json!({
                    "schema_version": 1,
                    "status": manifest.status,
                    "launchable": false,
                    "profile_id": PROFILE_ID,
                    "cache_key": manifest.cache_key,
                    "source_tier": manifest.source_tier,
                    "lineage_kind": "gguf_recovered_fp16",
                }),
                quant_type: "F16 recovered".into(),
            })
        }
        ExperimentalInventoryCacheKind::RequantizedMlx => {
            let manifest_path = path.join("manifest.json");
            require_regular_file_bounded(
                &manifest_path,
                MAX_MANIFEST_BYTES,
                "re-quantized cache manifest",
            )?;
            let manifest: RequantizeManifest =
                serde_json::from_reader(fs::File::open(manifest_path)?)?;
            let expected_key = requantize_cache_key(&manifest.recipe);
            if manifest.schema_version != 1
                || manifest.status != "experimental_requantized_structurally_validated"
                || manifest.launchable
                || manifest.profile_id != REQUANT_PROFILE_ID
                || manifest.cache_key != cache_key_name
                || manifest.cache_key != expected_key
                || manifest.source_recovery_cache_key != R2_F16_CACHE_KEY
                || manifest.worker_sha256 != sha256(REQUANT_WORKER.as_bytes())
                || manifest.profile_sha256 != sha256(REQUANT_PROFILE.as_bytes())
                || manifest.environment_lock_sha256 != sha256(REQUANT_ENVIRONMENT_LOCK.as_bytes())
                || manifest.third_party_notice_sha256 != sha256(REQUANT_NOTICE.as_bytes())
            {
                bail!("Experimental re-quantized cache provenance is invalid");
            }
            validate_inventory_file_closure(path, &manifest.files, &manifest.worker_report_sha256)?;
            let recipe = manifest.recipe.profile_name();
            Ok(ExperimentalInventorySummary {
                provenance: serde_json::json!({
                    "schema_version": 1,
                    "status": manifest.status,
                    "launchable": false,
                    "profile_id": REQUANT_PROFILE_ID,
                    "cache_key": manifest.cache_key,
                    "source_recovery_cache_key": manifest.source_recovery_cache_key,
                    "recipe": {
                        "id": recipe,
                        "bits": manifest.recipe.bits(),
                        "group_size": 64,
                    },
                    "lineage_kind": "mlx_requantized",
                }),
                quant_type: recipe.into(),
            })
        }
    }
}

fn validate_inventory_file_closure(
    cache: &Path,
    expected_files: &BTreeMap<String, String>,
    expected_report_sha256: &str,
) -> Result<()> {
    let actual_files = manifest_files(cache)?;
    if &actual_files != expected_files
        || actual_files.get("validation.json").map(String::as_str) != Some(expected_report_sha256)
    {
        bail!("Experimental cache differs from its complete published file closure");
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn inventory_test_manifest(
    kind: ExperimentalInventoryCacheKind,
    files: BTreeMap<String, String>,
) -> Result<(String, serde_json::Value)> {
    let report_sha = files
        .get("validation.json")
        .cloned()
        .ok_or_else(|| anyhow!("Test file closure requires validation.json"))?;
    match kind {
        ExperimentalInventoryCacheKind::RecoveredGguf => {
            let profile: RecoveryProfile = serde_json::from_str(PROFILE)?;
            let tier = RecoveryTier::F16;
            let source = profile
                .gguf_source
                .tiers
                .get(tier.profile_name())
                .ok_or_else(|| anyhow!("Test profile has no F16 tier"))?;
            let key = cache_key(tier.clone(), &source.sha256);
            let manifest = RecoveryManifest {
                schema_version: 1,
                status: "experimental_structurally_validated".into(),
                launchable: false,
                profile_id: PROFILE_ID.into(),
                cache_key: key.clone(),
                source_tier: tier,
                worker_sha256: sha256(WORKER.as_bytes()),
                profile_sha256: sha256(PROFILE.as_bytes()),
                requirements_lock_sha256: sha256(REQUIREMENTS_LOCK.as_bytes()),
                environment_lock_sha256: sha256(ENVIRONMENT_LOCK.as_bytes()),
                third_party_notice_sha256: sha256(NOTICE.as_bytes()),
                worker_report_sha256: report_sha,
                files,
            };
            Ok((key, serde_json::to_value(manifest)?))
        }
        ExperimentalInventoryCacheKind::RequantizedMlx => {
            let recipe = MlxQuantRecipe::Affine8bitG64;
            let key = requantize_cache_key(&recipe);
            let manifest = RequantizeManifest {
                schema_version: 1,
                status: "experimental_requantized_structurally_validated".into(),
                launchable: false,
                profile_id: REQUANT_PROFILE_ID.into(),
                cache_key: key.clone(),
                source_recovery_cache_key: R2_F16_CACHE_KEY.into(),
                recipe,
                worker_sha256: sha256(REQUANT_WORKER.as_bytes()),
                profile_sha256: sha256(REQUANT_PROFILE.as_bytes()),
                environment_lock_sha256: sha256(REQUANT_ENVIRONMENT_LOCK.as_bytes()),
                third_party_notice_sha256: sha256(REQUANT_NOTICE.as_bytes()),
                worker_report_sha256: report_sha,
                files,
            };
            Ok((key, serde_json::to_value(manifest)?))
        }
    }
}

fn validate_cache(
    path: &Path,
    expected_key: &str,
    max_output_bytes: u64,
) -> Result<RecoveryResult> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        bail!("Experimental recovery cache root must be a non-symlink directory");
    }
    let manifest_path = path.join("manifest.json");
    let report_path = path.join("validation.json");
    let complete_path = path.join(".complete");
    require_regular_file_bounded(&manifest_path, MAX_MANIFEST_BYTES, "cache manifest")?;
    require_regular_file_bounded(&report_path, MAX_REPORT_BYTES, "cache validation report")?;
    require_regular_file_bounded(&complete_path, 0, "cache completion sentinel")?;
    let manifest: RecoveryManifest = serde_json::from_reader(fs::File::open(manifest_path)?)?;
    if manifest.schema_version != 1
        || manifest.status != "experimental_structurally_validated"
        || manifest.launchable
        || manifest.profile_id != PROFILE_ID
        || manifest.cache_key != expected_key
        || manifest.worker_sha256 != sha256(WORKER.as_bytes())
        || manifest.profile_sha256 != sha256(PROFILE.as_bytes())
        || manifest.requirements_lock_sha256 != sha256(REQUIREMENTS_LOCK.as_bytes())
        || manifest.environment_lock_sha256 != sha256(ENVIRONMENT_LOCK.as_bytes())
        || manifest.third_party_notice_sha256 != sha256(NOTICE.as_bytes())
    {
        bail!("Experimental recovery cache provenance is invalid");
    }
    if hash_file(&report_path)? != manifest.worker_report_sha256 {
        bail!("Experimental recovery validation report was modified");
    }
    let expected_files = manifest_files(path)?;
    if expected_files != manifest.files {
        bail!("Experimental recovery cache contents differ from its manifest");
    }
    let report = read_worker_report(&report_path)?;
    let profile: RecoveryProfile = serde_json::from_str(PROFILE)?;
    let tier = manifest.source_tier.clone();
    let source_tier = profile
        .gguf_source
        .tiers
        .get(tier.profile_name())
        .ok_or_else(|| anyhow!("Cached tier is absent from embedded profile"))?;
    if report.source.sha256 != source_tier.sha256 {
        bail!("Cached source identity differs from the embedded profile");
    }
    // Cache reuse cannot trust the original mutable source path; complete cache closure,
    // typed report identity, and the current profile/source hash bind its provenance.
    validate_cached_worker_report(&report, &path.join("fp16"), &tier, max_output_bytes)?;
    Ok(RecoveryResult {
        cache_dir: path.to_path_buf(),
        fp16_dir: path.join("fp16"),
        report,
        launchable: false,
    })
}

fn require_regular_file_bounded(path: &Path, maximum: u64, kind: &str) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("Cannot inspect {kind} '{}'", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() > maximum {
        bail!("{kind} must be a non-symlink regular file within its size bound");
    }
    Ok(())
}

fn validate_cached_worker_report(
    report: &WorkerReport,
    output: &Path,
    tier: &RecoveryTier,
    max_output_bytes: u64,
) -> Result<()> {
    let profile: RecoveryProfile = serde_json::from_str(PROFILE)?;
    let expected = profile
        .gguf_source
        .tiers
        .get(tier.profile_name())
        .ok_or_else(|| anyhow!("Unknown cached tier"))?;
    if report.schema_version != 1
        || report.worker_version != WORKER_VERSION
        || report.status != "recovered"
        || report.profile_id != PROFILE_ID
        || report.output.dtype != profile.output_dtype
        || report.output.tensor_count != profile.expected_tensor_count
        || report.tensor_inventory.len() as u64 != profile.expected_tensor_count
        || report.source.tier != tier.profile_name()
        || report.source.sha256 != expected.sha256
        || report.source.size_bytes != expected.size
        || report.source.architecture != profile.architecture
        || report.source.tensor_count != profile.expected_tensor_count
        || report.source.path.file_name().and_then(OsStr::to_str)
            != Some(expected.filename.as_str())
        || report.source.quant_inventory != expected.quant_inventory
        || report.source.tensor_inventory_sha256 != expected.tensor_inventory_sha256
        || sha256(&serde_json::to_vec(&serde_json::to_value(
            &report.tensor_inventory,
        )?)?)
            != expected.tensor_inventory_sha256
        || report.skipped_tensors != 0
        || report.unknown_tensors != 0
        || report.duplicate_tensors != 0
        || report.shape_mismatches != 0
        || report.non_finite_tensors != 0
        || report.authoritative_reference.repo_id != profile.authoritative_source.repo_id
        || report.authoritative_reference.revision != profile.authoritative_source.revision
        || report.authoritative_reference.files != profile.authoritative_source.files
        || report.authoritative_reference.tensor_count != profile.expected_tensor_count
    {
        bail!("Cached worker report failed strict profile closure");
    }
    let actual = flat_files(output)?;
    let bytes = flat_directory_bytes(output)?;
    if actual != report.output.files
        || bytes != report.output.actual_output_bytes
        || bytes > max_output_bytes
    {
        bail!("Cached output differs from its complete bounded closure");
    }
    Ok(())
}

fn flat_files(root: &Path) -> Result<BTreeMap<String, String>> {
    let mut result = BTreeMap::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        if entry.file_type()?.is_symlink() || !entry.file_type()?.is_file() {
            bail!("Unexpected non-file in recovered FP16 output");
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        result.insert(name, hash_file(&entry.path())?);
    }
    Ok(result)
}

fn flat_directory_bytes(root: &Path) -> Result<u64> {
    let mut total = 0u64;
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_symlink() || !file_type.is_file() {
            bail!("Unexpected non-file in recovered FP16 output");
        }
        total = total
            .checked_add(entry.metadata()?.len())
            .ok_or_else(|| anyhow!("Recovered output size overflow"))?;
    }
    Ok(total)
}

fn manifest_files(root: &Path) -> Result<BTreeMap<String, String>> {
    fn visit(root: &Path, current: &Path, out: &mut BTreeMap<String, String>) -> Result<()> {
        for entry in fs::read_dir(current)? {
            let entry = entry?;
            let path = entry.path();
            let file_type = entry.file_type()?;
            if file_type.is_symlink() {
                bail!("Symlink found in experimental recovery cache");
            }
            let relative = path
                .strip_prefix(root)?
                .to_string_lossy()
                .replace('\\', "/");
            if relative == "manifest.json" || relative == ".complete" {
                continue;
            }
            if file_type.is_dir() {
                visit(root, &path, out)?;
            } else if file_type.is_file() {
                out.insert(relative, hash_file(&path)?);
            } else {
                bail!("Unsupported entry in experimental recovery cache");
            }
        }
        Ok(())
    }
    let mut result = BTreeMap::new();
    visit(root, root, &mut result)?;
    Ok(result)
}

fn recursive_directory_bytes(root: &Path) -> Result<u64> {
    let mut total = 0u64;
    fn visit(current: &Path, total: &mut u64) -> Result<()> {
        for entry in fs::read_dir(current)? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            if file_type.is_symlink() {
                bail!("Symlink found while measuring directory closure");
            }
            if file_type.is_dir() {
                visit(&entry.path(), total)?;
            } else if file_type.is_file() {
                *total = total
                    .checked_add(entry.metadata()?.len())
                    .ok_or_else(|| anyhow!("Directory byte count overflow"))?;
            } else {
                bail!("Unsupported entry while measuring directory closure");
            }
        }
        Ok(())
    }
    visit(root, &mut total)?;
    Ok(total)
}

fn atomic_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let temporary = path.with_extension("tmp");
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn hash_file(path: &Path) -> Result<String> {
    use std::io::Read;
    let mut file = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(hex_digest(digest.finalize().as_slice()))
}

fn sha256(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    hex_digest(digest.finalize().as_slice())
}

fn hex_digest(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

struct CleanupDir(PathBuf);

impl Drop for CleanupDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    static WORKER_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn test_python_path() -> PathBuf {
        if let Some(path) = std::env::var_os("PYTHON") {
            let path = PathBuf::from(path);
            if path.is_file() {
                return path;
            }
        }
        if let Some(path_var) = std::env::var_os("PATH") {
            for directory in std::env::split_paths(&path_var) {
                let candidate = directory.join("python3");
                if candidate.is_file() {
                    return candidate;
                }
            }
        }
        panic!("python3 interpreter not found on PATH; set PYTHON for recovery tests");
    }

    fn write_valid_cache_manifest(cache: &Path, expected_key: &str) {
        let manifest = RecoveryManifest {
            schema_version: 1,
            status: "experimental_structurally_validated".into(),
            launchable: false,
            profile_id: PROFILE_ID.into(),
            cache_key: expected_key.into(),
            source_tier: RecoveryTier::F16,
            worker_sha256: sha256(WORKER.as_bytes()),
            profile_sha256: sha256(PROFILE.as_bytes()),
            requirements_lock_sha256: sha256(REQUIREMENTS_LOCK.as_bytes()),
            environment_lock_sha256: sha256(ENVIRONMENT_LOCK.as_bytes()),
            third_party_notice_sha256: sha256(NOTICE.as_bytes()),
            worker_report_sha256: "0".repeat(64),
            files: BTreeMap::new(),
        };
        atomic_json(&cache.join("manifest.json"), &manifest).unwrap();
        fs::write(cache.join(".complete"), b"").unwrap();
    }

    fn valid_test_report(output: &Path) -> WorkerReport {
        fs::create_dir_all(output).unwrap();
        fs::write(output.join("model.safetensors"), b"test weights").unwrap();
        WorkerReport {
            schema_version: 1,
            worker_version: "test".into(),
            status: "recovered".into(),
            profile_id: PROFILE_ID.into(),
            output: WorkerOutput {
                dtype: "float16".into(),
                tensor_count: 1,
                files: flat_files(output).unwrap(),
                estimated_weight_bytes: 12,
                actual_output_bytes: 12,
            },
            tensor_inventory: vec![WorkerTensor {
                source_name: "test".into(),
                ..WorkerTensor::default()
            }],
            ..WorkerReport::default()
        }
    }

    #[test]
    fn cache_key_is_profile_tier_and_source_bound() {
        let first = cache_key(RecoveryTier::F16, &"a".repeat(64));
        let second = cache_key(RecoveryTier::Q8_0, &"a".repeat(64));
        let third = cache_key(RecoveryTier::F16, &"b".repeat(64));
        assert_eq!(first.len(), 64);
        assert_ne!(first, second);
        assert_ne!(first, third);
    }

    #[test]
    fn worker_asset_key_binds_every_materialized_asset() {
        assert_eq!(worker_asset_key().len(), 64);
        assert_ne!(worker_asset_key(), sha256(WORKER.as_bytes()));
    }

    #[test]
    fn requantize_keys_bind_recipe_and_all_assets() {
        let four = requantize_cache_key(&MlxQuantRecipe::Affine4bitG64);
        let six = requantize_cache_key(&MlxQuantRecipe::Affine6bitG64);
        let eight = requantize_cache_key(&MlxQuantRecipe::Affine8bitG64);
        assert_eq!(four.len(), 64);
        assert_ne!(four, six);
        assert_ne!(six, eight);
        assert_eq!(requantize_worker_asset_key().len(), 64);
        assert_ne!(
            requantize_worker_asset_key(),
            sha256(REQUANT_WORKER.as_bytes())
        );
        for recipe in [
            MlxQuantRecipe::Affine4bitG64,
            MlxQuantRecipe::Affine6bitG64,
            MlxQuantRecipe::Affine8bitG64,
        ] {
            assert_eq!(recipe.expected_tensor_inventory_sha256().len(), 64);
        }
    }

    #[test]
    fn strict_report_rejects_every_failure_counter_and_output_change() {
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("fp16");
        let report = valid_test_report(&output);
        assert!(failure_counters_are_zero(&report));

        for field in [
            "skipped_tensors",
            "unknown_tensors",
            "duplicate_tensors",
            "shape_mismatches",
            "non_finite_tensors",
        ] {
            let mut invalid = report.clone();
            match field {
                "skipped_tensors" => invalid.skipped_tensors = 1,
                "unknown_tensors" => invalid.unknown_tensors = 1,
                "duplicate_tensors" => invalid.duplicate_tensors = 1,
                "shape_mismatches" => invalid.shape_mismatches = 1,
                "non_finite_tensors" => invalid.non_finite_tensors = 1,
                _ => unreachable!(),
            }
            assert!(
                !failure_counters_are_zero(&invalid),
                "{field} must fail closed"
            );
        }

        fs::write(output.join("model.safetensors"), b"modified").unwrap();
        assert_ne!(flat_files(&output).unwrap(), report.output.files);
    }

    #[cfg(unix)]
    #[test]
    fn cache_root_symlink_is_rejected_before_manifest_reads() {
        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("target");
        fs::create_dir(&target).unwrap();
        let linked = temp.path().join("linked");
        std::os::unix::fs::symlink(&target, &linked).unwrap();
        assert!(validate_cache(&linked, &"a".repeat(64), u64::MAX).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn internal_cache_control_symlinks_are_rejected_before_reads() {
        for control in ["manifest.json", "validation.json", ".complete"] {
            let temp = tempfile::tempdir().unwrap();
            let cache = temp.path().join("cache");
            fs::create_dir(&cache).unwrap();
            let expected_key = "a".repeat(64);
            write_valid_cache_manifest(&cache, &expected_key);
            fs::write(cache.join("validation.json"), b"{}").unwrap();
            let outside = temp.path().join("outside");
            fs::write(&outside, vec![b'x'; 2 * 1024 * 1024]).unwrap();
            fs::remove_file(cache.join(control)).unwrap();
            std::os::unix::fs::symlink(&outside, cache.join(control)).unwrap();
            let error = validate_cache(&cache, &expected_key, u64::MAX).unwrap_err();
            assert!(
                error.to_string().contains("non-symlink regular file"),
                "{control} was not rejected before reading: {error:#}"
            );
        }
    }

    #[test]
    fn oversized_cache_manifest_and_validation_are_rejected_before_parsing_or_hashing() {
        for (control, maximum) in [
            ("manifest.json", MAX_MANIFEST_BYTES),
            ("validation.json", MAX_REPORT_BYTES),
        ] {
            let temp = tempfile::tempdir().unwrap();
            let cache = temp.path().join("cache");
            fs::create_dir(&cache).unwrap();
            let expected_key = "a".repeat(64);
            write_valid_cache_manifest(&cache, &expected_key);
            fs::write(cache.join("validation.json"), b"{}").unwrap();
            fs::OpenOptions::new()
                .write(true)
                .truncate(true)
                .open(cache.join(control))
                .unwrap()
                .set_len(maximum + 1)
                .unwrap();
            let error = validate_cache(&cache, &expected_key, u64::MAX).unwrap_err();
            assert!(
                error.to_string().contains("size bound"),
                "{control} was parsed or hashed before its size rejection: {error:#}"
            );
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[tokio::test]
    async fn active_worker_cancellation_stops_child_and_writes_sentinel() {
        let _guard = WORKER_TEST_LOCK.lock().await;
        let temp = tempfile::tempdir().unwrap();
        let python = test_python_path();
        let worker = temp.path().join("worker.py");
        let request = temp.path().join("request.json");
        let cancel_path = temp.path().join("cancel");
        fs::write(&worker, "import time\ntime.sleep(30)\n").unwrap();
        fs::write(&request, b"{}").unwrap();
        let cancelled = Arc::new(AtomicBool::new(false));
        let trigger = Arc::clone(&cancelled);
        let trigger_task = tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            trigger.store(true, Ordering::Release);
        });
        let started = std::time::Instant::now();
        let error = run_worker(&python, &worker, &request, &cancel_path, cancelled)
            .await
            .expect_err("active worker must be cancelled");
        trigger_task.await.unwrap();
        assert!(error.to_string().contains("cancelled"));
        assert!(cancel_path.is_file());
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
    }

    #[test]
    fn relative_paths_reject_absolute_traversal_and_rooted_backslash() {
        assert!(validate_relative(Path::new("gguf/model.gguf"), "source").is_ok());
        assert!(validate_relative(Path::new("../model.gguf"), "source").is_err());
        assert!(validate_relative(Path::new("/model.gguf"), "source").is_err());
        assert!(validate_relative(Path::new("\\model.gguf"), "source").is_err());
    }

    #[cfg(unix)]
    #[test]
    fn canonical_child_rejects_symlinks() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("models");
        fs::create_dir(&root).unwrap();
        let outside = temp.path().join("outside.gguf");
        fs::write(&outside, b"GGUF").unwrap();
        std::os::unix::fs::symlink(&outside, root.join("linked.gguf")).unwrap();
        assert!(canonical_library_file(&root, Path::new("linked.gguf"), "source").is_err());
    }

    #[cfg(unix)]
    #[test]
    fn managed_directory_rejects_preexisting_symlink_before_writes() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("root");
        let outside = temp.path().join("outside");
        fs::create_dir(&root).unwrap();
        fs::create_dir(&outside).unwrap();
        std::os::unix::fs::symlink(&outside, root.join("runtimes")).unwrap();
        assert!(create_managed_directory(&root, Path::new("runtimes/worker"), "runtime").is_err());
        assert!(!outside.join("worker").exists());
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[tokio::test]
    async fn worker_output_overflow_is_bounded_without_pipe_deadlock() {
        let _guard = WORKER_TEST_LOCK.lock().await;
        let temp = tempfile::tempdir().unwrap();
        let python = test_python_path();
        let worker = temp.path().join("worker.py");
        let request = temp.path().join("request.json");
        let cancel_path = temp.path().join("cancel");
        fs::write(&worker, "import sys,time\nsys.stdout.write('x' * 1048576)\nsys.stdout.flush()\ntime.sleep(30)\n").unwrap();
        fs::write(&request, b"{}").unwrap();
        let started = std::time::Instant::now();
        let error = run_worker(
            &python,
            &worker,
            &request,
            &cancel_path,
            Arc::new(AtomicBool::new(false)),
        )
        .await
        .expect_err("diagnostic overflow must terminate the worker");
        assert!(error.to_string().contains("diagnostic output bound"));
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[tokio::test]
    async fn cancellation_kills_term_ignoring_descendant_after_leader_exits() {
        let _guard = WORKER_TEST_LOCK.lock().await;
        let temp = tempfile::tempdir().unwrap();
        let python = test_python_path();
        let worker = temp.path().join("worker.py");
        let request = temp.path().join("request.json");
        let cancel_path = temp.path().join("cancel");
        let pid_path = temp.path().join("descendant.pid");
        fs::write(
            &worker,
            "import pathlib,subprocess,sys,time\np=pathlib.Path(sys.argv[-1]).with_name('descendant.pid')\nc=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)'])\np.write_text(str(c.pid))\ntime.sleep(30)\n",
        ).unwrap();
        fs::write(&request, b"{}").unwrap();
        let cancelled = Arc::new(AtomicBool::new(false));
        let trigger = Arc::clone(&cancelled);
        let wait_pid = pid_path.clone();
        let trigger_task = tokio::spawn(async move {
            for _ in 0..100 {
                if wait_pid.is_file() {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
            trigger.store(true, Ordering::Release);
        });
        let error = run_worker(&python, &worker, &request, &cancel_path, cancelled)
            .await
            .expect_err("cancellation must stop the process group");
        trigger_task.await.unwrap();
        assert!(error.to_string().contains("cancelled"));
        let pid: u32 = fs::read_to_string(pid_path).unwrap().parse().unwrap();
        for _ in 0..100 {
            if !unix_process::process_exists(pid) {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        panic!("TERM-ignoring descendant {pid} survived process-group teardown");
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[tokio::test]
    async fn nominal_exit_closes_pipe_holding_descendant_group() {
        let _guard = WORKER_TEST_LOCK.lock().await;
        let temp = tempfile::tempdir().unwrap();
        let python = test_python_path();
        let worker = temp.path().join("worker.py");
        let request = temp.path().join("request.json");
        let cancel_path = temp.path().join("cancel");
        let pid_path = temp.path().join("descendant.pid");
        fs::write(
            &worker,
            "import pathlib,subprocess,sys\np=pathlib.Path(sys.argv[-1]).with_name('descendant.pid')\nc=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)'])\np.write_text(str(c.pid))\n",
        ).unwrap();
        fs::write(&request, b"{}").unwrap();
        let started = std::time::Instant::now();
        run_worker(
            &python,
            &worker,
            &request,
            &cancel_path,
            Arc::new(AtomicBool::new(false)),
        )
        .await
        .expect("pipe holder must be closed after nominal leader exit");
        assert!(started.elapsed() < std::time::Duration::from_secs(7));
        let pid: u32 = fs::read_to_string(pid_path).unwrap().parse().unwrap();
        for _ in 0..100 {
            if !unix_process::process_exists(pid) {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        panic!("pipe-holding descendant {pid} survived process-group teardown");
    }

    #[test]
    fn embedded_profile_and_lock_have_expected_identity() {
        let profile: serde_json::Value = serde_json::from_str(PROFILE).unwrap();
        assert_eq!(profile["profile_id"], PROFILE_ID);
        assert!(REQUIREMENTS_LOCK.contains("gguf==0.19.0"));
        assert!(REQUIREMENTS_LOCK.contains("numpy==2.5.1"));
        assert!(REQUIREMENTS_LOCK.contains("safetensors==0.8.0"));
        assert!(NOTICE.contains("6a0da6529f233df79362cbf62dd96221c895351f"));
        assert!(NOTICE.contains("Permission is hereby granted"));
        assert!(ENVIRONMENT_LOCK.contains("environment_sha256"));
    }

    #[test]
    fn embedded_requantize_profile_is_explicit_and_non_launchable() {
        let profile: serde_json::Value = serde_json::from_str(REQUANT_PROFILE).unwrap();
        assert_eq!(profile["profile_id"], REQUANT_PROFILE_ID);
        assert_eq!(profile["source"]["recovery_cache_key"], R2_F16_CACHE_KEY);
        assert_eq!(profile["runtime"]["mlx_lm_version"], "0.31.3");
        assert_eq!(profile["runtime"]["mlx_version"], "0.32.0");
        assert_eq!(profile["recipes"]["affine_4bit_g64"]["bits"], 4);
        assert_eq!(profile["recipes"]["affine_6bit_g64"]["bits"], 6);
        assert_eq!(profile["recipes"]["affine_8bit_g64"]["bits"], 8);
        assert_eq!(profile["launchable"], false);
        assert!(REQUANT_ENVIRONMENT_LOCK.contains("a0a97c14483e1e24"));
        assert!(REQUANT_NOTICE.contains("Permission is hereby granted"));
    }

    #[test]
    fn requantized_manifest_cannot_claim_launchability() {
        let temp = tempfile::tempdir().unwrap();
        let cache = temp.path().join("cache");
        fs::create_dir(&cache).unwrap();
        let recipe = MlxQuantRecipe::Affine4bitG64;
        let key = requantize_cache_key(&recipe);
        let manifest = RequantizeManifest {
            schema_version: 1,
            status: "experimental_requantized_structurally_validated".into(),
            launchable: true,
            profile_id: REQUANT_PROFILE_ID.into(),
            cache_key: key.clone(),
            source_recovery_cache_key: R2_F16_CACHE_KEY.into(),
            recipe,
            worker_sha256: sha256(REQUANT_WORKER.as_bytes()),
            profile_sha256: sha256(REQUANT_PROFILE.as_bytes()),
            environment_lock_sha256: sha256(REQUANT_ENVIRONMENT_LOCK.as_bytes()),
            third_party_notice_sha256: sha256(REQUANT_NOTICE.as_bytes()),
            worker_report_sha256: "0".repeat(64),
            files: BTreeMap::new(),
        };
        atomic_json(&cache.join("manifest.json"), &manifest).unwrap();
        fs::write(cache.join(".complete"), b"").unwrap();
        assert!(validate_requantize_cache(&cache, &key, u64::MAX).is_err());
    }

    #[test]
    fn requantized_manifest_recipe_must_derive_its_cache_key() {
        let temp = tempfile::tempdir().unwrap();
        let cache = temp.path().join("cache");
        fs::create_dir(&cache).unwrap();
        let expected_key = requantize_cache_key(&MlxQuantRecipe::Affine4bitG64);
        let manifest = RequantizeManifest {
            schema_version: 1,
            status: "experimental_requantized_structurally_validated".into(),
            launchable: false,
            profile_id: REQUANT_PROFILE_ID.into(),
            cache_key: expected_key.clone(),
            source_recovery_cache_key: R2_F16_CACHE_KEY.into(),
            recipe: MlxQuantRecipe::Affine6bitG64,
            worker_sha256: sha256(REQUANT_WORKER.as_bytes()),
            profile_sha256: sha256(REQUANT_PROFILE.as_bytes()),
            environment_lock_sha256: sha256(REQUANT_ENVIRONMENT_LOCK.as_bytes()),
            third_party_notice_sha256: sha256(REQUANT_NOTICE.as_bytes()),
            worker_report_sha256: "0".repeat(64),
            files: BTreeMap::new(),
        };
        atomic_json(&cache.join("manifest.json"), &manifest).unwrap();
        fs::write(cache.join(".complete"), b"").unwrap();
        let error = validate_requantize_cache(&cache, &expected_key, u64::MAX).unwrap_err();
        assert!(error.to_string().contains("provenance"));
    }

    #[cfg(unix)]
    #[test]
    fn requantized_cache_root_symlink_is_rejected_before_manifest_reads() {
        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("target");
        fs::create_dir(&target).unwrap();
        let linked = temp.path().join("linked");
        std::os::unix::fs::symlink(&target, &linked).unwrap();
        let error = validate_requantize_cache(&linked, &"a".repeat(64), u64::MAX).unwrap_err();
        assert!(error.to_string().contains("non-symlink directory"));
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn local_fixture(tier: RecoveryTier) -> (RecoveryContext, RecoveryRequest) {
        let home = dirs::home_dir().expect("home directory");
        let models_dir = home.join(".config/llama-monitor/models");
        let filename = match tier {
            RecoveryTier::F16 => "SmolLM2-135M-Instruct-F16.gguf",
            RecoveryTier::Q8_0 => "SmolLM2-135M-Instruct-Q8_0.gguf",
            RecoveryTier::Q6K => "SmolLM2-135M-Instruct-Q6_K.gguf",
            RecoveryTier::Q4KM => "SmolLM2-135M-Instruct-Q4_K_M.gguf",
        };
        (
            RecoveryContext {
                models_dir,
                config_dir: home.join(".config/llama-monitor"),
            },
            RecoveryRequest {
                source_gguf: PathBuf::from(format!(
                    "experimental/import-lab/fixtures/smollm2-135m-v1/gguf/{filename}"
                )),
                reference_dir: PathBuf::from(
                    "experimental/import-lab/fixtures/smollm2-135m-v1/authoritative",
                ),
                source_tier: tier,
                max_output_bytes: 512 * 1024 * 1024,
                disk_safety_margin_bytes: 512 * 1024 * 1024,
            },
        )
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn local_requantize_fixture(recipe: MlxQuantRecipe) -> (RecoveryContext, RequantizeRequest) {
        let home = dirs::home_dir().expect("home directory");
        (
            RecoveryContext {
                models_dir: home.join(".config/llama-monitor/models"),
                config_dir: home.join(".config/llama-monitor"),
            },
            RequantizeRequest {
                recovered_cache: PathBuf::from(format!("rapid-mlx/imports/{R2_F16_CACHE_KEY}")),
                recipe,
                max_output_bytes: 320 * 1024 * 1024,
                disk_safety_margin_bytes: 512 * 1024 * 1024,
            },
        )
    }

    #[ignore = "requires the pinned local R3 MLX environment"]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[tokio::test]
    async fn pinned_local_requantize_environment_matches_complete_lock() {
        let home = dirs::home_dir().expect("home directory");
        let runtime_root =
            home.join(".config/llama-monitor/runtimes/rapid-mlx/.staging/0.10.10-qualification");
        verify_python_environment(
            &runtime_python(&runtime_root),
            &runtime_root,
            REQUANT_ENVIRONMENT_LOCK,
            "MLX re-quantization Python",
        )
        .await
        .expect("complete R3 environment identity");
    }

    #[ignore = "requires the pinned local R2 cache and R3 MLX environment"]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[tokio::test]
    async fn real_smollm2_r3_selected_recipe() {
        let recipe = match std::env::var("LLAMA_MONITOR_R3_RECIPE").as_deref() {
            Ok("affine_4bit_g64") => MlxQuantRecipe::Affine4bitG64,
            Ok("affine_6bit_g64") => MlxQuantRecipe::Affine6bitG64,
            Ok("affine_8bit_g64") => MlxQuantRecipe::Affine8bitG64,
            value => panic!("set LLAMA_MONITOR_R3_RECIPE to a profile recipe, got {value:?}"),
        };
        let (context, request) = local_requantize_fixture(recipe.clone());
        let result = requantize_mlx(context, request, Arc::new(AtomicBool::new(false)))
            .await
            .expect("real R3 re-quantization");
        assert!(!result.launchable);
        assert_eq!(result.report.recipe.recipe_id, recipe.profile_name());
        assert_eq!(result.report.recipe.bits, recipe.bits());
        assert!(result.report.recipe.quantized_module_count > 0);
        assert!(result.model_dir.join("model.safetensors").is_file());
        eprintln!("R3_CACHE={}", result.cache_dir.display());
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[tokio::test]
    async fn requantize_pre_cancel_fails_before_staging() {
        let (context, request) = local_requantize_fixture(MlxQuantRecipe::Affine4bitG64);
        let error = requantize_mlx(context, request, Arc::new(AtomicBool::new(true)))
            .await
            .expect_err("pre-cancelled re-quantization must fail");
        assert!(error.to_string().contains("cancelled"));
    }

    #[ignore = "requires the pinned local R2 cache and final R3 4-bit cache"]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[tokio::test]
    async fn real_requantize_cache_honors_tighter_output_bound() {
        let (context, mut request) = local_requantize_fixture(MlxQuantRecipe::Affine4bitG64);
        request.max_output_bytes = 1;
        let error = requantize_mlx(context, request, Arc::new(AtomicBool::new(false)))
            .await
            .expect_err("cached R3 output must honor the caller's tighter bound");
        assert!(error.to_string().contains("bounded closure"));
        let home = dirs::home_dir().expect("home directory");
        let staging = home.join(".config/llama-monitor/models/rapid-mlx/requantized/.staging");
        assert_eq!(fs::read_dir(staging).unwrap().count(), 0);
    }

    #[ignore = "requires the pinned local R2 corpus and hash-locked Python environment"]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[tokio::test]
    async fn pinned_local_toolchain_matches_complete_environment_lock() {
        let home = dirs::home_dir().expect("home directory");
        let runtime_root = home.join(".config/llama-monitor/runtimes/gguf-recovery/r2-v1");
        verify_toolchain(&runtime_python(&runtime_root), &runtime_root)
            .await
            .expect("complete environment identity");
    }

    #[ignore = "requires the pinned local R2 corpus and hash-locked Python environment"]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[tokio::test]
    async fn real_smollm2_r2_selected_tier() {
        let tier = match std::env::var("LLAMA_MONITOR_R2_TIER").as_deref() {
            Ok("f16") => RecoveryTier::F16,
            Ok("q8_0") => RecoveryTier::Q8_0,
            Ok("q6_k") => RecoveryTier::Q6K,
            Ok("q4_k_m") => RecoveryTier::Q4KM,
            value => panic!("set LLAMA_MONITOR_R2_TIER to a profile tier, got {value:?}"),
        };
        let (context, request) = local_fixture(tier);
        let result = recover(context, request, Arc::new(AtomicBool::new(false)))
            .await
            .expect("real R2 recovery");
        assert!(!result.launchable);
        assert_eq!(result.report.output.tensor_count, 272);
        assert_eq!(result.report.skipped_tensors, 0);
        assert!(result.fp16_dir.join("model.safetensors").is_file());
        eprintln!("R2_CACHE={}", result.cache_dir.display());
    }

    #[ignore = "requires the pinned local R2 corpus and hash-locked Python environment"]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[tokio::test]
    async fn real_smollm2_r2_disk_bound_cleans_staging() {
        let (context, mut request) = local_fixture(RecoveryTier::F16);
        request.max_output_bytes = 1;
        let error = recover(context, request, Arc::new(AtomicBool::new(false)))
            .await
            .expect_err("disk/output bound must fail");
        assert!(
            error.to_string().contains("bound"),
            "unexpected error: {error:#}"
        );
    }

    #[ignore = "requires the pinned local R2 corpus and hash-locked Python environment"]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[tokio::test]
    async fn real_smollm2_r2_pre_cancel_cleans_staging() {
        let (context, request) = local_fixture(RecoveryTier::F16);
        let cancelled = Arc::new(AtomicBool::new(true));
        let error = recover(context, request, cancelled)
            .await
            .expect_err("pre-cancelled recovery must fail");
        assert!(error.to_string().contains("cancelled"));
    }

    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    #[test]
    fn local_execution_fails_closed_off_apple_silicon() {
        assert!(ensure_local_execution_supported().is_err());
    }
}
