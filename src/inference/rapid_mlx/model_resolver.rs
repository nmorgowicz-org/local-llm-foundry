use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, OnceLock};

/// Typed, frontend-safe codec for a Rapid-MLX model source (D5 / Gap 3.2).
/// Rust owns parsing, validation, canonicalization, and safe edit semantics.
/// Frontend never flattens a typed source into a lossy string and then reconstructs it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub struct RapidMlxModelSourceView {
    /// One of: "mlx_directory", "hugging_face_repo", "alias", "authoritative_safetensors", "gguf_file", "unknown"
    pub kind: String,
    /// Human-readable display name for UI
    pub display_name: String,
    /// Canonical identity string (repo@revision, local path, or alias value)
    pub canonical_identity: String,
    /// Hugging Face repo ID (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repo_id: Option<String>,
    /// Immutable revision/commit (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    /// Local filesystem path (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub local_path: Option<String>,
    /// MLX conversion recipe for safetensors sources
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversion_recipe: Option<String>,
    /// Provenance hash (for conversion sources)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provenance_hash: Option<String>,
    /// Fields editable by the user for this source kind
    pub editable_fields: Vec<String>,
    /// Whether this source is launchable in principle
    pub launchable: bool,
    /// Warnings/diagnostics for this source
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

impl RapidMlxModelSourceView {
    /// Create a view from a typed source. Never opens legacy `model_path`.
    pub fn from_source(source: &RapidMlxModelSource) -> Self {
        match source {
            RapidMlxModelSource::MlxDirectory { path } => Self {
                kind: "mlx_directory".into(),
                display_name: path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| path.to_string_lossy().into_owned()),
                canonical_identity: path.to_string_lossy().into_owned(),
                repo_id: None,
                revision: None,
                local_path: Some(path.to_string_lossy().into_owned()),
                conversion_recipe: None,
                provenance_hash: None,
                editable_fields: vec!["local_path".into()],
                launchable: true,
                warnings: vec![],
            },
            RapidMlxModelSource::HuggingFaceRepo { repo_id, revision } => Self {
                kind: "hugging_face_repo".into(),
                display_name: repo_id.clone(),
                canonical_identity: format!("{repo_id}@{revision}"),
                repo_id: Some(repo_id.clone()),
                revision: Some(revision.clone()),
                local_path: None,
                conversion_recipe: None,
                provenance_hash: None,
                editable_fields: vec!["repo_id".into(), "revision".into()],
                launchable: true,
                warnings: vec![],
            },
            RapidMlxModelSource::Alias { value } => Self {
                kind: "alias".into(),
                display_name: value.clone(),
                canonical_identity: value.clone(),
                repo_id: None,
                revision: None,
                local_path: None,
                conversion_recipe: None,
                provenance_hash: None,
                editable_fields: vec!["canonical_identity".into()],
                launchable: true,
                warnings: vec!["Free-form alias is validated by Rapid-MLX at launch".into()],
            },
            RapidMlxModelSource::AuthoritativeSafetensors {
                source: st_source,
                revision_or_hash,
                recipe,
            } => Self {
                kind: "authoritative_safetensors".into(),
                display_name: match st_source {
                    AuthoritativeSafetensorsSource::LocalDirectory { path } => path
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_else(|| path.to_string_lossy().into_owned()),
                    AuthoritativeSafetensorsSource::HuggingFaceRepo { repo_id, revision } => {
                        format!("{repo_id}@{revision}")
                    }
                },
                canonical_identity: match st_source {
                    AuthoritativeSafetensorsSource::LocalDirectory { path } => {
                        path.to_string_lossy().into_owned()
                    }
                    AuthoritativeSafetensorsSource::HuggingFaceRepo { repo_id, revision } => {
                        format!("{repo_id}@{revision}")
                    }
                },
                repo_id: match st_source {
                    AuthoritativeSafetensorsSource::LocalDirectory { .. } => None,
                    AuthoritativeSafetensorsSource::HuggingFaceRepo { repo_id, .. } => {
                        Some(repo_id.clone())
                    }
                },
                revision: match st_source {
                    AuthoritativeSafetensorsSource::LocalDirectory { .. } => None,
                    AuthoritativeSafetensorsSource::HuggingFaceRepo { revision, .. } => {
                        Some(revision.clone())
                    }
                },
                local_path: match st_source {
                    AuthoritativeSafetensorsSource::LocalDirectory { path } => {
                        Some(path.to_string_lossy().into_owned())
                    }
                    AuthoritativeSafetensorsSource::HuggingFaceRepo { .. } => None,
                },
                conversion_recipe: Some(String::from(match recipe {
                    MlxConversionRecipe::Fp16 => "fp16",
                    MlxConversionRecipe::Bf16 => "bf16",
                    MlxConversionRecipe::Q4 => "q4",
                    MlxConversionRecipe::Q6 => "q6",
                    MlxConversionRecipe::Q8 => "q8",
                })),
                provenance_hash: Some(revision_or_hash.clone()),
                editable_fields: vec!["revision".into(), "conversion_recipe".into()],
                launchable: false,
                warnings: vec![
                    "Requires official safetensors-to-MLX conversion".into(),
                    format!("Conversion uses pinned official mlx-lm=={PINNED_MLX_LM_VERSION}"),
                ],
            },
            RapidMlxModelSource::GgufFile { path } => Self {
                kind: "gguf_file".into(),
                display_name: path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| path.to_string_lossy().into_owned()),
                canonical_identity: path.to_string_lossy().into_owned(),
                repo_id: None,
                revision: None,
                local_path: Some(path.to_string_lossy().into_owned()),
                conversion_recipe: None,
                provenance_hash: None,
                editable_fields: vec!["local_path".into()],
                launchable: false,
                warnings: vec![
                    "GGUF files must be run with llama.cpp".into(),
                    "Use the separate experimental Phase 5.5 import workflow when available".into(),
                ],
            },
            RapidMlxModelSource::Unknown { kind, raw } => Self {
                kind: "unknown".into(),
                display_name: format!("Unknown source type: {kind}"),
                canonical_identity: raw
                    .get("canonical_identity")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                repo_id: None,
                revision: None,
                local_path: None,
                conversion_recipe: None,
                provenance_hash: None,
                editable_fields: vec![],
                launchable: false,
                warnings: vec![format!(
                    "This preset uses unsupported future source kind '{kind}'. Preserve it or update llama-monitor before launching."
                )],
            },
        }
    }

    // Convenience constructors with no caller; the resolver builds views inline.
    #[allow(dead_code)]
    /// Create an error view when no typed source is configured.
    pub fn empty() -> Self {
        Self {
            kind: "unknown".into(),
            display_name: "No model configured".into(),
            canonical_identity: "".into(),
            repo_id: None,
            revision: None,
            local_path: None,
            conversion_recipe: None,
            provenance_hash: None,
            editable_fields: vec![],
            launchable: false,
            warnings: vec!["No model source configured for this Rapid-MLX preset".into()],
        }
    }

    #[allow(dead_code)] // see the note on this impl block
    /// Check if this source is valid for launch.
    pub fn is_valid(&self) -> bool {
        self.launchable && self.warnings.is_empty() && !self.canonical_identity.is_empty()
    }
}

const MAX_BLOCKING_RESOLVER_TASKS: usize = 2;

pub const PINNED_MLX_LM_VERSION: &str = "0.31.3";
const MANIFEST_NAME: &str = "llama-monitor-conversion.json";
const COMPLETE_NAME: &str = ".complete";
#[cfg(test)]
thread_local! {
    static SAFETENSORS_INDEX_SCANS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RapidMlxModelSource {
    MlxDirectory {
        path: PathBuf,
    },
    HuggingFaceRepo {
        repo_id: String,
        revision: String,
    },
    Alias {
        value: String,
    },
    AuthoritativeSafetensors {
        source: AuthoritativeSafetensorsSource,
        revision_or_hash: String,
        recipe: MlxConversionRecipe,
    },
    GgufFile {
        path: PathBuf,
    },
    /// A forward-compatible source shape from a newer llama-monitor version.
    /// It round-trips without becoming launchable until this build understands it.
    Unknown {
        kind: String,
        raw: serde_json::Value,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum KnownRapidMlxModelSource {
    MlxDirectory {
        path: PathBuf,
    },
    HuggingFaceRepo {
        repo_id: String,
        revision: String,
    },
    Alias {
        // Accept catalog sources saved by builds that used `name`; emit `value`.
        #[serde(alias = "name")]
        value: String,
    },
    AuthoritativeSafetensors {
        source: AuthoritativeSafetensorsSource,
        revision_or_hash: String,
        #[serde(default)]
        recipe: MlxConversionRecipe,
    },
    GgufFile {
        path: PathBuf,
    },
}

impl From<KnownRapidMlxModelSource> for RapidMlxModelSource {
    fn from(value: KnownRapidMlxModelSource) -> Self {
        match value {
            KnownRapidMlxModelSource::MlxDirectory { path } => Self::MlxDirectory { path },
            KnownRapidMlxModelSource::HuggingFaceRepo { repo_id, revision } => {
                Self::HuggingFaceRepo { repo_id, revision }
            }
            KnownRapidMlxModelSource::Alias { value } => Self::Alias { value },
            KnownRapidMlxModelSource::AuthoritativeSafetensors {
                source,
                revision_or_hash,
                recipe,
            } => Self::AuthoritativeSafetensors {
                source,
                revision_or_hash,
                recipe,
            },
            KnownRapidMlxModelSource::GgufFile { path } => Self::GgufFile { path },
        }
    }
}

impl Serialize for RapidMlxModelSource {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::MlxDirectory { path } => {
                KnownRapidMlxModelSource::MlxDirectory { path: path.clone() }.serialize(serializer)
            }
            Self::HuggingFaceRepo { repo_id, revision } => {
                KnownRapidMlxModelSource::HuggingFaceRepo {
                    repo_id: repo_id.clone(),
                    revision: revision.clone(),
                }
                .serialize(serializer)
            }
            Self::Alias { value } => KnownRapidMlxModelSource::Alias {
                value: value.clone(),
            }
            .serialize(serializer),
            Self::AuthoritativeSafetensors {
                source,
                revision_or_hash,
                recipe,
            } => KnownRapidMlxModelSource::AuthoritativeSafetensors {
                source: source.clone(),
                revision_or_hash: revision_or_hash.clone(),
                recipe: *recipe,
            }
            .serialize(serializer),
            Self::GgufFile { path } => {
                KnownRapidMlxModelSource::GgufFile { path: path.clone() }.serialize(serializer)
            }
            Self::Unknown { kind, raw } => {
                let mut value = raw.clone();
                let object = value.as_object_mut().ok_or_else(|| {
                    serde::ser::Error::custom("unknown Rapid-MLX source must be an object")
                })?;
                object.insert("kind".into(), serde_json::Value::String(kind.clone()));
                value.serialize(serializer)
            }
        }
    }
}

impl<'de> Deserialize<'de> for RapidMlxModelSource {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = serde_json::Value::deserialize(deserializer)?;
        let kind = raw
            .get("kind")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| serde::de::Error::custom("Rapid-MLX model source requires string kind"))?
            .to_string();
        match serde_json::from_value::<KnownRapidMlxModelSource>(raw.clone()) {
            Ok(source) => Ok(source.into()),
            Err(_) => Ok(Self::Unknown { kind, raw }),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AuthoritativeSafetensorsSource {
    LocalDirectory { path: PathBuf },
    HuggingFaceRepo { repo_id: String, revision: String },
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MlxConversionRecipe {
    #[default]
    Fp16,
    Bf16,
    Q4,
    Q6,
    Q8,
}

impl MlxConversionRecipe {
    fn cli_args(self) -> &'static [&'static str] {
        match self {
            Self::Fp16 => &["--dtype", "float16"],
            Self::Bf16 => &["--dtype", "bfloat16"],
            Self::Q4 => &["--quantize", "--q-bits", "4"],
            Self::Q6 => &["--quantize", "--q-bits", "6"],
            Self::Q8 => &["--quantize", "--q-bits", "8"],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResolvedRapidMlxSourceKind {
    MlxDirectory,
    HuggingFaceRepo,
    VerifiedAlias,
    FreeFormAlias,
    OfficialConversion,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConversionProvenance {
    pub resolver_schema_version: u32,
    pub source: AuthoritativeSafetensorsSource,
    pub revision_or_hash: String,
    pub source_content_sha256: String,
    pub mlx_lm_version: String,
    pub recipe: MlxConversionRecipe,
    pub cache_key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedRapidMlxLaunchModel {
    pub launch_argument: String,
    pub display_name: String,
    pub source_kind: ResolvedRapidMlxSourceKind,
    pub original_input: RapidMlxModelSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conversion: Option<ConversionProvenance>,
    #[serde(default)]
    pub required_environment: Vec<String>,
    #[serde(default)]
    pub warnings: Vec<String>,
    #[serde(default)]
    pub remediation: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trust_remote_code_required: Option<bool>,
    #[serde(skip)]
    pub environment: BTreeMap<OsString, OsString>,
}

impl ResolvedRapidMlxLaunchModel {
    // Alias-only launch construction has no caller: `command.rs` validates aliases through
    // the full resolver instead.
    #[allow(dead_code)]
    pub fn validated_alias(value: &str) -> Result<Self> {
        validate_alias(value)?;
        Ok(Self {
            display_name: value.to_string(),
            launch_argument: value.to_string(),
            source_kind: ResolvedRapidMlxSourceKind::FreeFormAlias,
            original_input: RapidMlxModelSource::Alias {
                value: value.to_string(),
            },
            conversion: None,
            required_environment: Vec::new(),
            warnings: vec!["Free-form alias is validated by Rapid-MLX at launch".into()],
            remediation: Vec::new(),
            trust_remote_code_required: None,
            environment: BTreeMap::new(),
        })
    }

    pub fn environment(&self) -> impl Iterator<Item = (&OsString, &OsString)> {
        self.environment.iter()
    }
}

impl RapidMlxModelSource {
    pub fn display_name(&self) -> String {
        display_name(self)
    }
}

#[derive(Debug, Clone)]
pub struct RapidMlxResolveContext {
    pub models_dir: PathBuf,
    pub python_executable: PathBuf,
    pub runtime_version: String,
    pub hf_token: Option<String>,
    pub verified_aliases: Vec<String>,
    pub execute_conversion: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolverPreviewState {
    Ready,
    ConversionRequired,
    UnsupportedSource,
    Invalid,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResolverPreview {
    pub state: ResolverPreviewState,
    pub source_kind: String,
    pub display_name: String,
    pub warnings: Vec<String>,
    pub remediation: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct ConversionManifest {
    schema_version: u32,
    provenance: ConversionProvenance,
    files: Vec<ManifestFile>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct ManifestFile {
    path: String,
    size: u64,
    sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversionCommandPlan {
    pub program: PathBuf,
    pub args: Vec<OsString>,
    pub env: BTreeMap<OsString, OsString>,
}

/// Extract the HuggingFace repo id encoded in a hub cache path component.
///
/// The hub layout is `…/hub/models--owner--repo[/snapshots/<commit>/…]`. Both the
/// container and anything below it carry the repo identity, so a model picked from
/// the cache can be resolved without an online filename search. Returns `None`
/// when no `models--` component (or a malformed one) is present.
pub fn hf_cache_repo_id(path: &Path) -> Option<String> {
    let component = path
        .components()
        .filter_map(|c| c.as_os_str().to_str())
        .find(|c| c.starts_with("models--"))?;
    let encoded = component.strip_prefix("models--")?;
    let (owner, repo) = encoded.split_once("--")?;
    if owner.is_empty() || repo.is_empty() {
        return None;
    }
    Some(format!("{owner}/{repo}"))
}

pub fn source_from_legacy_model_path(value: &str) -> Result<RapidMlxModelSource> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        bail!("Rapid-MLX requires a model source");
    }
    let path = PathBuf::from(trimmed);
    if trimmed.to_ascii_lowercase().ends_with(".gguf") {
        return Ok(RapidMlxModelSource::GgufFile { path });
    }
    if path.is_dir() {
        return Ok(RapidMlxModelSource::MlxDirectory { path });
    }
    Ok(RapidMlxModelSource::Alias {
        value: trimmed.to_string(),
    })
}

pub fn preview(source: &RapidMlxModelSource, context: &RapidMlxResolveContext) -> ResolverPreview {
    match validate_source(source, context) {
        Ok(()) => match source {
            RapidMlxModelSource::AuthoritativeSafetensors { .. } => ResolverPreview {
                state: ResolverPreviewState::ConversionRequired,
                source_kind: "authoritative_safetensors".into(),
                display_name: display_name(source),
                warnings: vec![format!(
                    "Conversion uses pinned official mlx-lm=={PINNED_MLX_LM_VERSION}"
                )],
                remediation: Vec::new(),
            },
            _ => ResolverPreview {
                state: ResolverPreviewState::Ready,
                source_kind: source_kind_name(source).into(),
                display_name: display_name(source),
                warnings: alias_warnings(source, context),
                remediation: Vec::new(),
            },
        },
        Err(error) => {
            let unsupported = matches!(source, RapidMlxModelSource::GgufFile { .. });
            ResolverPreview {
                state: if unsupported {
                    ResolverPreviewState::UnsupportedSource
                } else {
                    ResolverPreviewState::Invalid
                },
                source_kind: source_kind_name(source).into(),
                display_name: display_name(source),
                warnings: vec![error.to_string()],
                remediation: if unsupported {
                    vec![
                        "Run this GGUF with llama.cpp".into(),
                        "Use the separate experimental Phase 5.5 import workflow when available"
                            .into(),
                    ]
                } else {
                    Vec::new()
                },
            }
        }
    }
}

pub async fn preview_async(
    source: RapidMlxModelSource,
    context: RapidMlxResolveContext,
) -> Result<ResolverPreview> {
    run_resolver_blocking(move || Ok(preview(&source, &context))).await
}

async fn run_resolver_blocking<T, F>(operation: F) -> Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    static GATE: OnceLock<Arc<tokio::sync::Semaphore>> = OnceLock::new();
    let gate = GATE
        .get_or_init(|| Arc::new(tokio::sync::Semaphore::new(MAX_BLOCKING_RESOLVER_TASKS)))
        .clone();
    let permit = gate
        .acquire_owned()
        .await
        .map_err(|_| anyhow!("Model resolver blocking worker gate was closed"))?;
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        operation()
    })
    .await
    .context("Model resolver blocking worker failed")?
}

pub async fn resolve(
    source: RapidMlxModelSource,
    context: &RapidMlxResolveContext,
) -> Result<ResolvedRapidMlxLaunchModel> {
    let validated_mlx = if let RapidMlxModelSource::MlxDirectory { path } = &source {
        let path = path.clone();
        let validation_context = context.clone();
        Some(
            run_resolver_blocking(move || {
                if validation_context.models_dir.as_os_str().is_empty() {
                    bail!("A configured models_dir is required");
                }
                let canonical = canonical_model_directory(&path, &local_model_allowed_root(&path))?;
                reject_app_staging_directory(&canonical, &validation_context.models_dir)?;
                validate_if_app_conversion(&canonical, &validation_context.models_dir)?;
                crate::inference::rapid_mlx::sidecar_hygiene::inspect_and_repair_snapshot(
                    &canonical,
                )?;
                Ok(canonical)
            })
            .await?,
        )
    } else if let RapidMlxModelSource::AuthoritativeSafetensors {
        source: AuthoritativeSafetensorsSource::LocalDirectory { .. },
        revision_or_hash,
        ..
    } = &source
    {
        if revision_or_hash.trim().len() < 7 || revision_or_hash.len() > 128 {
            bail!(
                "Authoritative safetensors requires a stable 7-128 character revision or content hash"
            );
        }
        if context.models_dir.as_os_str().is_empty() {
            bail!("A configured models_dir is required");
        }
        None
    } else {
        let validation_source = source.clone();
        let validation_context = context.clone();
        run_resolver_blocking(move || validate_source(&validation_source, &validation_context))
            .await?;
        None
    };
    let mut environment = hugging_face_environment(context)?;
    let result = match &source {
        RapidMlxModelSource::MlxDirectory { path: _ } => {
            let canonical =
                validated_mlx.expect("MLX directory validation always returns a canonical path");
            resolved(
                canonical.to_string_lossy().into_owned(),
                display_name(&source),
                ResolvedRapidMlxSourceKind::MlxDirectory,
                source.clone(),
            )
        }
        RapidMlxModelSource::HuggingFaceRepo { repo_id, revision } => {
            let snapshot =
                resolve_hf_snapshot(repo_id, revision, &source, context, &environment).await?;
            crate::inference::rapid_mlx::sidecar_hygiene::inspect_and_repair_snapshot(&snapshot)?;
            let validation_path = snapshot.clone();
            let validation_root = context.models_dir.clone();
            run_resolver_blocking(move || {
                validate_model_directory(&validation_path, &validation_root)
            })
            .await?;
            let mut output = resolved(
                snapshot.to_string_lossy().into_owned(),
                repo_id.clone(),
                ResolvedRapidMlxSourceKind::HuggingFaceRepo,
                source.clone(),
            );
            output
                .required_environment
                .extend(["HF_HUB_CACHE".into(), "HF_XET_CACHE".into()]);
            if context.hf_token.is_some() {
                output.required_environment.push("HF_TOKEN".into());
            }
            // Security: detect if this HF repo requires trust_remote_code (custom Python code).
            // Data-only repos (safetensors + config.json only) launch normally.
            // Repos with custom code require revision-scoped user consent.
            output.trust_remote_code_required =
                Some(run_resolver_blocking(move || needs_trust_remote_code(&snapshot)).await?);
            output
        }
        RapidMlxModelSource::Alias { value } => {
            let verified = context.verified_aliases.iter().any(|alias| alias == value);
            let mut output = resolved(
                value.clone(),
                value.clone(),
                if verified {
                    ResolvedRapidMlxSourceKind::VerifiedAlias
                } else {
                    ResolvedRapidMlxSourceKind::FreeFormAlias
                },
                source.clone(),
            );
            output.warnings = alias_warnings(&source, context);
            output
                .required_environment
                .extend(["HF_HUB_CACHE".into(), "HF_XET_CACHE".into()]);
            if context.hf_token.is_some() {
                output.required_environment.push("HF_TOKEN".into());
            }
            output
        }
        RapidMlxModelSource::AuthoritativeSafetensors {
            source: conversion_source,
            revision_or_hash,
            recipe,
        } => {
            if !context.execute_conversion {
                bail!("Official safetensors conversion is required; preview cannot be launched");
            }
            let local_source = match conversion_source {
                AuthoritativeSafetensorsSource::LocalDirectory { path } => {
                    let path = path.clone();
                    run_resolver_blocking(move || {
                        path.canonicalize()
                            .context("Failed to canonicalize authoritative model directory")
                    })
                    .await?
                }
                AuthoritativeSafetensorsSource::HuggingFaceRepo { repo_id, revision } => {
                    resolve_hf_snapshot(repo_id, revision, &source, context, &environment).await?
                }
            };
            let (path, provenance) = convert_authoritative(
                conversion_source.clone(),
                local_source,
                revision_or_hash.clone(),
                *recipe,
                context,
                &environment,
            )
            .await?;
            let mut output = resolved(
                path.to_string_lossy().into_owned(),
                display_name(&source),
                ResolvedRapidMlxSourceKind::OfficialConversion,
                source.clone(),
            );
            output.conversion = Some(provenance);
            output
        }
        RapidMlxModelSource::GgufFile { path } => bail!(
            "Rapid-MLX {} cannot load GGUF '{}'. Use llama.cpp, or explicitly enter the separate experimental Phase 5.5 import workflow when available",
            context.runtime_version,
            path.display()
        ),
        RapidMlxModelSource::Unknown { kind, .. } => bail!(
            "Rapid-MLX source kind '{kind}' is from a newer llama-monitor version and cannot be launched safely by this build"
        ),
    };
    let mut result = result;
    result.environment.append(&mut environment);
    Ok(result)
}

fn resolved(
    argument: String,
    display_name: String,
    kind: ResolvedRapidMlxSourceKind,
    source: RapidMlxModelSource,
) -> ResolvedRapidMlxLaunchModel {
    ResolvedRapidMlxLaunchModel {
        launch_argument: argument,
        display_name,
        source_kind: kind,
        original_input: source,
        conversion: None,
        required_environment: Vec::new(),
        warnings: Vec::new(),
        remediation: Vec::new(),
        trust_remote_code_required: None,
        environment: BTreeMap::new(),
    }
}

fn validate_source(source: &RapidMlxModelSource, context: &RapidMlxResolveContext) -> Result<()> {
    match source {
        RapidMlxModelSource::MlxDirectory { path } => {
            let canonical = canonical_model_directory(path, &local_model_allowed_root(path))?;
            reject_app_staging_directory(&canonical, &context.models_dir)?;
            validate_if_app_conversion(&canonical, &context.models_dir)?;
            Ok(())
        }
        RapidMlxModelSource::HuggingFaceRepo { repo_id, revision } => {
            validate_repo(repo_id)?;
            validate_revision(revision)
        }
        RapidMlxModelSource::Alias { value } => validate_alias(value),
        RapidMlxModelSource::AuthoritativeSafetensors {
            source,
            revision_or_hash,
            ..
        } => {
            if revision_or_hash.trim().len() < 7 || revision_or_hash.len() > 128 {
                bail!(
                    "Authoritative safetensors requires a stable 7-128 character revision or content hash"
                );
            }
            match source {
                AuthoritativeSafetensorsSource::LocalDirectory { path } => {
                    validate_transformers_directory(path)
                }
                AuthoritativeSafetensorsSource::HuggingFaceRepo { repo_id, revision } => {
                    validate_repo(repo_id)?;
                    validate_revision(revision)?;
                    if revision != revision_or_hash
                        || revision.len() != 40
                        || !revision.bytes().all(|byte| byte.is_ascii_hexdigit())
                    {
                        bail!(
                            "Authoritative Hugging Face conversion requires revision_or_hash to match an immutable 40-character commit SHA"
                        );
                    }
                    Ok(())
                }
            }
        }
        RapidMlxModelSource::GgufFile { path } => {
            bail!(
                "Rapid-MLX {} cannot load GGUF '{}'; GGUF is not a native Rapid-MLX model source",
                context.runtime_version,
                path.display()
            )
        }
        RapidMlxModelSource::Unknown { kind, .. } => {
            bail!("Unsupported future Rapid-MLX source kind '{kind}'")
        }
    }?;
    if context.models_dir.as_os_str().is_empty() {
        bail!("A configured models_dir is required");
    }
    Ok(())
}

fn reject_app_staging_directory(path: &Path, models_dir: &Path) -> Result<()> {
    let Ok(library_root) = models_dir.canonicalize() else {
        return Ok(());
    };
    let staging_root = library_root.join(".staging");
    if path.starts_with(&staging_root) {
        bail!(
            "MLX models inside the app staging directory cannot be launched until promotion completes"
        );
    }
    Ok(())
}

/// Detect whether this locally downloaded HF model directory requires
/// trust_remote_code to load. A repo is considered safe (data-only) when it
/// only contains safetensors/weights and standard config files. Custom-code
/// repos are those that declare transformers main_class / auto_map entries or
/// include model loading scripts (main.py, modeling_*.py) that rapid-mlx
/// would need to execute.
fn needs_trust_remote_code(model_dir: &Path) -> Result<bool> {
    // Check config.json for transformers main_class or auto_map indicators
    let config_path = model_dir.join("config.json");
    if config_path.is_file() {
        let content = match fs::read_to_string(&config_path) {
            Ok(c) => c,
            Err(_) => return Ok(false), // safe default: assume data-only on read error
        };
        let value: serde_json::Value = match serde_json::from_str(&content) {
            Ok(v) => v,
            Err(_) => return Ok(false), // safe default: assume data-only on parse error
        };
        // main_class indicates a custom model class must be imported
        if value.get("main_class").and_then(|v| v.as_str()).is_some() {
            return Ok(true);
        }
        // auto_map with non-standard classes indicates custom code loading
        if let Some(auto_map) = value.get("auto_map").and_then(|v| v.as_object()) {
            for (_, class_value) in auto_map {
                if let Some(class_str) = class_value.as_str()
                    && !class_str.starts_with("Auto")
                    && !class_str.contains("transformers.models")
                    && !class_str.contains("transformers_modules")
                {
                    return Ok(true);
                }
            }
        }
    }
    // Check for custom Python loaders that rapid-mlx would execute
    if let Ok(entries) = fs::read_dir(model_dir) {
        for entry in entries.filter_map(Result::ok) {
            let file_name = entry.file_name();
            let name = file_name.to_string_lossy().to_lowercase();
            if name == "main.py"
                || (name.starts_with("modeling_") && name.ends_with(".py"))
                || (name.starts_with("configuration_") && name.ends_with(".py"))
            {
                return Ok(true);
            }
        }
    }
    // Default: data-only repo is safe
    Ok(false)
}

fn validate_if_app_conversion(path: &Path, models_dir: &Path) -> Result<()> {
    let Ok(library_root) = models_dir.canonicalize() else {
        return Ok(());
    };
    if path.starts_with(library_root.join("mlx/converted")) {
        validate_cached_conversion(path).context("Converted MLX cache validation failed")?;
    }
    Ok(())
}

fn canonical_model_directory(path: &Path, allowed_symlink_root: &Path) -> Result<PathBuf> {
    validate_model_directory(path, allowed_symlink_root)?;
    let config: serde_json::Value =
        serde_json::from_reader(fs::File::open(path.join("config.json"))?)
            .context("MLX config.json is not valid JSON")?;
    if config.get("model_type").and_then(|v| v.as_str()).is_none() {
        bail!("MLX config.json must identify model_type");
    }
    path.canonicalize()
        .context("Failed to canonicalize MLX model directory")
}

fn local_model_allowed_root(path: &Path) -> PathBuf {
    path.parent()
        .filter(|parent| parent.file_name().is_some_and(|name| name == "snapshots"))
        .and_then(Path::parent)
        .filter(|repo_root| repo_root.join("blobs").is_dir())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| path.to_path_buf())
}

fn validate_transformers_directory(path: &Path) -> Result<()> {
    validate_model_directory(path, path)
}

/// Tensor names in a safetensors file, read from its header only.
///
/// Layout is `u64 little-endian header length`, then that many bytes of JSON
/// whose top-level keys are tensor names (plus an optional `__metadata__`). The
/// weights themselves are never read, so this stays cheap on a 27 GB trunk.
fn safetensors_tensor_names(path: &Path) -> Result<Vec<String>> {
    let mut file = fs::File::open(path)?;
    let mut length = [0u8; 8];
    file.read_exact(&mut length)?;
    let header_len = u64::from_le_bytes(length);
    // A real header is kilobytes to low megabytes. The bound stops a corrupt or
    // hostile length field from turning a preflight check into an OOM.
    if header_len == 0 || header_len > 64 * 1024 * 1024 {
        bail!(
            "Safetensors header length is implausible ({header_len} bytes) in {}",
            path.display()
        );
    }
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header)?;
    let parsed: serde_json::Value =
        serde_json::from_slice(&header).context("Safetensors header is invalid JSON")?;
    let object = parsed
        .as_object()
        .ok_or_else(|| anyhow!("Safetensors header is not a JSON object"))?;
    Ok(object
        .keys()
        .filter(|key| key.as_str() != "__metadata__")
        .cloned()
        .collect())
}

/// Refuse a trunk directory that contains an MTP draft-head sidecar.
///
/// `mlx_lm` loads a trunk by globbing `model*.safetensors`, so a sidecar sitting
/// in the model directory is picked up as if it were a trunk shard. The `mtp.*`
/// keys are then stripped again, but their mere presence flips
/// `should_shift_norm_weights`, which adds +1.0 to every trunk RMSNorm weight.
/// That shift is correct for a raw HF checkpoint and destroys an already-
/// converted MLX one: the backbone degrades to gibberish with no error at all.
///
/// Upstream's own extractor defaults to writing `model-mtp.safetensors` into the
/// model directory, so this layout arrives by following upstream's documented
/// usage — which is why it is refused for any model the app is asked to launch,
/// not merely for sidecars the app created.
///
/// The discriminator is deliberately "every tensor in the file is `mtp.*`", not
/// the file name and not "any mtp tensor anywhere". A checkpoint that genuinely
/// ships MTP weights interleaved with its own layers is a legitimate embedded-MTP
/// model and must still launch; a file that is *only* draft-head tensors is a
/// sidecar in the wrong place, whatever it happens to be called.
fn reject_in_trunk_mtp_sidecar(dir: &Path) -> Result<()> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Ok(());
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        // Exactly mlx_lm's glob: a file the trunk loader would pick up.
        if !(name.starts_with("model") && name.ends_with(".safetensors")) {
            continue;
        }
        if !path.is_file() {
            continue;
        }
        // An unreadable or malformed candidate is not evidence of this defect;
        // the asset validators own that complaint.
        let Ok(names) = safetensors_tensor_names(&path) else {
            continue;
        };
        if names.is_empty() {
            continue;
        }
        if names.iter().all(|tensor| tensor.starts_with("mtp.")) {
            bail!(
                "Refusing to launch {}: it contains the MTP draft-head sidecar {name}, whose \
                 {} tensors are all mtp.*. mlx_lm globs model*.safetensors, so this file is \
                 loaded as a trunk shard and silently adds +1.0 to every trunk RMSNorm weight, \
                 corrupting an already-converted MLX model into gibberish with no error. \
                 Remediation: move {name} out of the model directory (a sibling directory is \
                 fine), then point the speculative config at its new path. Never store a draft \
                 head inside the trunk.",
                dir.display(),
                names.len()
            );
        }
    }
    Ok(())
}

pub(crate) fn validate_model_directory(path: &Path, allowed_symlink_root: &Path) -> Result<()> {
    validate_model_directory_assets(path, allowed_symlink_root).map(|_| ())
}

fn validate_model_directory_assets(
    path: &Path,
    allowed_symlink_root: &Path,
) -> Result<Vec<PathBuf>> {
    if !path.is_dir() {
        bail!(
            "Model source is not a readable directory: {}",
            path.display()
        );
    }
    if path.symlink_metadata()?.file_type().is_symlink() {
        bail!("Symlink model directories are not accepted for conversion");
    }
    validate_child(path, &path.join("config.json"), allowed_symlink_root)?;
    let tokenizer_json = path.join("tokenizer.json");
    let tokenizer_config = path.join("tokenizer_config.json");
    let sentencepiece = ["tokenizer.model", "sentencepiece.bpe.model"]
        .iter()
        .map(|name| path.join(name))
        .find(|candidate| candidate.exists());
    let vocab_pair = path.join("vocab.json").exists() && path.join("merges.txt").exists();
    if tokenizer_json.exists() {
        validate_child(path, &tokenizer_json, allowed_symlink_root)?;
    }
    if tokenizer_config.exists() {
        validate_child(path, &tokenizer_config, allowed_symlink_root)?;
    }
    if !tokenizer_json.is_file() && sentencepiece.is_none() && !vocab_pair {
        bail!(
            "Model directory requires tokenizer.json, a SentencePiece model, or vocab.json plus merges.txt"
        );
    }
    for asset in sentencepiece.into_iter().chain(
        [path.join("vocab.json"), path.join("merges.txt")]
            .into_iter()
            .filter(|candidate| candidate.exists()),
    ) {
        validate_child(path, &asset, allowed_symlink_root)?;
    }
    reject_in_trunk_mtp_sidecar(path)?;
    safetensors_files(path, allowed_symlink_root)
}

fn validate_child(model_root: &Path, path: &Path, allowed_symlink_root: &Path) -> Result<()> {
    if !path.is_file() {
        bail!(
            "Model directory is missing required {}",
            path.file_name().and_then(|v| v.to_str()).unwrap_or("asset")
        );
    }
    let canonical_model = model_root.canonicalize()?;
    let canonical_allowed = allowed_symlink_root.canonicalize()?;
    let canonical = path.canonicalize()?;
    if !canonical.starts_with(&canonical_model) && !canonical.starts_with(&canonical_allowed) {
        bail!(
            "Model asset escapes its approved source/cache root: {}",
            path.display()
        );
    }
    Ok(())
}

fn safetensors_files(path: &Path, allowed_symlink_root: &Path) -> Result<Vec<PathBuf>> {
    let index_path = path.join("model.safetensors.index.json");
    let mut files = if index_path.is_file() {
        #[cfg(test)]
        SAFETENSORS_INDEX_SCANS.with(|count| count.set(count.get() + 1));
        validate_child(path, &index_path, allowed_symlink_root)?;
        let index: serde_json::Value = serde_json::from_reader(fs::File::open(&index_path)?)
            .context("Safetensors index is invalid JSON")?;
        let weight_map = index
            .get("weight_map")
            .and_then(|value| value.as_object())
            .ok_or_else(|| anyhow!("Safetensors index requires a weight_map object"))?;
        let mut names: Vec<String> = weight_map
            .values()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_string)
                    .ok_or_else(|| anyhow!("Safetensors index contains a non-string shard"))
            })
            .collect::<Result<_>>()?;
        names.sort();
        names.dedup();
        if names.is_empty() || names.len() > 4096 {
            bail!("Safetensors index has an invalid shard count");
        }
        names
            .into_iter()
            .map(|name| {
                let relative = Path::new(&name);
                if relative.is_absolute()
                    || relative
                        .components()
                        .any(|part| matches!(part, std::path::Component::ParentDir))
                    || !name.ends_with(".safetensors")
                {
                    bail!("Safetensors index contains an unsafe shard path");
                }
                Ok(path.join(relative))
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        fs::read_dir(path)?
            .filter_map(Result::ok)
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .ends_with(".safetensors")
            })
            .map(|entry| entry.path())
            .collect()
    };
    files.sort();
    if files.is_empty() {
        bail!("Model directory requires complete safetensors weights");
    }
    for file in &files {
        validate_child(path, file, allowed_symlink_root)?;
    }
    Ok(files)
}

fn validate_repo(value: &str) -> Result<()> {
    let parts: Vec<_> = value.split('/').collect();
    if parts.len() != 2
        || parts.iter().any(|part| {
            part.is_empty()
                || part.len() > 96
                || !part
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
        })
    {
        bail!("Hugging Face repository must be a valid owner/name identifier");
    }
    Ok(())
}

fn validate_revision(value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
    {
        bail!("Hugging Face revision must be explicit and contain only safe revision characters");
    }
    Ok(())
}

fn validate_alias(value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 256
        || value.starts_with(['/', '\\'])
        || value.contains("..")
        || !value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'/' | b':'))
    {
        bail!("Rapid-MLX alias contains unsafe or unsupported characters");
    }
    Ok(())
}

fn hugging_face_environment(
    context: &RapidMlxResolveContext,
) -> Result<BTreeMap<OsString, OsString>> {
    let cache = context.models_dir.join("cache/huggingface");
    let hub = cache.join("hub");
    let xet = cache.join("xet");
    ensure_inside_library(&context.models_dir, &hub)?;
    ensure_inside_library(&context.models_dir, &xet)?;
    fs::create_dir_all(&hub)?;
    fs::create_dir_all(&xet)?;
    let mut env = BTreeMap::new();
    env.insert("HF_HUB_CACHE".into(), hub.into_os_string());
    env.insert("HF_XET_CACHE".into(), xet.into_os_string());
    if let Some(token) = context.hf_token.as_ref().filter(|token| !token.is_empty()) {
        env.insert("HF_TOKEN".into(), token.into());
    }
    Ok(env)
}

fn ensure_inside_library(root: &Path, candidate: &Path) -> Result<()> {
    let root = fs::canonicalize(root)
        .with_context(|| format!("models_dir does not exist: {}", root.display()))?;
    let parent = candidate
        .parent()
        .ok_or_else(|| anyhow!("Invalid library path"))?;
    fs::create_dir_all(parent)?;
    let parent = fs::canonicalize(parent)?;
    if !parent.starts_with(&root) {
        bail!("Model storage path escapes configured models_dir");
    }
    Ok(())
}

fn conversion_key(
    source: &AuthoritativeSafetensorsSource,
    revision: &str,
    content_hash: &str,
    recipe: MlxConversionRecipe,
) -> Result<String> {
    let bytes = serde_json::to_vec(&(
        1u32,
        source,
        revision,
        content_hash,
        PINNED_MLX_LM_VERSION,
        recipe,
    ))?;
    Ok(hex_digest(&Sha256::digest(bytes)))
}

pub fn conversion_command_plan(
    python: &Path,
    local_source: &Path,
    output: &Path,
    recipe: MlxConversionRecipe,
    environment: &BTreeMap<OsString, OsString>,
) -> ConversionCommandPlan {
    let mut args: Vec<OsString> = vec!["-m".into(), "mlx_lm.convert".into(), "--hf-path".into()];
    args.push(local_source.as_os_str().to_owned());
    args.extend(["--mlx-path".into(), output.as_os_str().to_owned()]);
    args.extend(recipe.cli_args().iter().map(OsString::from));
    ConversionCommandPlan {
        program: python.to_path_buf(),
        args,
        env: environment.clone(),
    }
}

async fn convert_authoritative(
    source: AuthoritativeSafetensorsSource,
    local_source: PathBuf,
    revision: String,
    recipe: MlxConversionRecipe,
    context: &RapidMlxResolveContext,
    environment: &BTreeMap<OsString, OsString>,
) -> Result<(PathBuf, ConversionProvenance)> {
    verify_pinned_mlx_lm(&context.python_executable).await?;
    let hash_source = local_source.clone();
    let hash_root = context.models_dir.clone();
    let source_content_sha256 =
        run_resolver_blocking(move || model_content_hash(&hash_source, &hash_root)).await?;
    let key = conversion_key(&source, &revision, &source_content_sha256, recipe)?;
    let converted_root = context.models_dir.join("mlx/converted");
    let staging_root = context.models_dir.join(".staging/conversions");
    fs::create_dir_all(&converted_root)?;
    fs::create_dir_all(&staging_root)?;
    ensure_inside_library(&context.models_dir, &converted_root)?;
    ensure_inside_library(&context.models_dir, &staging_root)?;
    let final_path = converted_root.join(&key);
    let provenance = ConversionProvenance {
        resolver_schema_version: 1,
        source: source.clone(),
        revision_or_hash: revision,
        source_content_sha256,
        mlx_lm_version: PINNED_MLX_LM_VERSION.into(),
        recipe,
        cache_key: key,
    };
    if final_path.exists() {
        let validation_path = final_path.clone();
        let expected = provenance.clone();
        run_resolver_blocking(move || validate_published_conversion(&validation_path, &expected))
            .await?;
        return Ok((final_path, provenance));
    }
    let staging = tempfile::Builder::new()
        .prefix(".converting-")
        .tempdir_in(&staging_root)?;
    let output = staging.path().join("model");
    let plan = conversion_command_plan(
        &context.python_executable,
        &local_source,
        &output,
        recipe,
        environment,
    );
    run_plan(&plan)
        .await
        .context("Pinned official mlx-lm conversion failed")?;
    let validation_output = output.clone();
    run_resolver_blocking(move || validate_transformers_directory(&validation_output))
        .await
        .context("Converted MLX output is incomplete")?;
    validate_mlx_load(&context.python_executable, &output, environment).await?;
    let publish_output = output.clone();
    let publish_provenance = provenance.clone();
    run_resolver_blocking(move || {
        let manifest = ConversionManifest {
            schema_version: 1,
            provenance: publish_provenance,
            files: manifest_files(&publish_output)?,
        };
        atomic_json(&publish_output.join(MANIFEST_NAME), &manifest)?;
        fs::File::create(publish_output.join(COMPLETE_NAME))?.sync_all()?;
        Ok(())
    })
    .await?;
    let kept = staging.keep();
    let kept_output = kept.join("model");
    if let Err(error) = fs::rename(&kept_output, &final_path) {
        if final_path.exists() {
            let validation_path = final_path.clone();
            let expected = provenance.clone();
            run_resolver_blocking(move || {
                validate_published_conversion(&validation_path, &expected)
            })
            .await
            .context("Concurrent conversion published a mismatched cache entry")?;
        } else {
            return Err(error).context("Atomic promotion of converted MLX model failed");
        }
    }
    let _ = fs::remove_dir(&kept);
    let validation_path = final_path.clone();
    let expected = provenance.clone();
    run_resolver_blocking(move || validate_published_conversion(&validation_path, &expected))
        .await?;
    Ok((final_path, provenance))
}

async fn verify_pinned_mlx_lm(python: &Path) -> Result<()> {
    let code = format!(
        "import importlib.metadata,sys;sys.exit(0 if importlib.metadata.version('mlx-lm') == '{PINNED_MLX_LM_VERSION}' else 23)"
    );
    let mut version = tokio::process::Command::new(python);
    version.args(["-c", &code]).stdin(Stdio::null());
    run_command_bounded(version, std::time::Duration::from_secs(15), 16 * 1024, &[])
        .await
        .with_context(|| {
            format!("Official conversion requires exact mlx-lm=={PINNED_MLX_LM_VERSION}")
        })?;

    let mut help = tokio::process::Command::new(python);
    help.args(["-m", "mlx_lm.convert", "--help"])
        .stdin(Stdio::null());
    let output = run_command_bounded(help, std::time::Duration::from_secs(30), 128 * 1024, &[])
        .await
        .context("Unable to verify the pinned mlx_lm.convert command contract")?;
    for required in [
        "--hf-path",
        "--mlx-path",
        "--dtype",
        "--quantize",
        "--q-bits",
    ] {
        if !output.contains(required) {
            bail!("Pinned mlx_lm.convert is missing required option {required}");
        }
    }
    Ok(())
}

async fn resolve_hf_snapshot(
    repo_id: &str,
    revision: &str,
    source_metadata: &RapidMlxModelSource,
    context: &RapidMlxResolveContext,
    environment: &BTreeMap<OsString, OsString>,
) -> Result<PathBuf> {
    validate_repo(repo_id)?;
    validate_revision(revision)?;
    let code = "from huggingface_hub import snapshot_download; import sys; print(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], cache_dir=sys.argv[3]))";
    let cache = context.models_dir.join("cache/huggingface/hub");
    let mut command = tokio::process::Command::new(&context.python_executable);
    command
        .args(["-c", code, repo_id, revision])
        .arg(&cache)
        .envs(environment.iter())
        .stdin(Stdio::null());
    let secrets = sensitive_environment_values(environment);
    let output = run_command_bounded(
        command,
        std::time::Duration::from_secs(30 * 60),
        64 * 1024,
        &secrets,
    )
    .await
    .context("Revision-pinned Hugging Face snapshot download failed")?;
    let path_text = output.lines().last().unwrap_or_default().trim();
    if path_text.is_empty() {
        bail!("Hugging Face snapshot downloader did not return a local snapshot path");
    }
    let path = PathBuf::from(path_text).canonicalize()?;
    let models_root = context.models_dir.canonicalize()?;
    if !path.starts_with(&models_root) {
        bail!("Hugging Face snapshot escaped the app-scoped model cache");
    }
    atomic_json(
        &path.join(crate::models::library::HF_SOURCE_METADATA_NAME),
        source_metadata,
    )?;
    Ok(path)
}

async fn validate_mlx_load(
    python: &Path,
    output: &Path,
    env: &BTreeMap<OsString, OsString>,
) -> Result<()> {
    let mut command = tokio::process::Command::new(python);
    command
        .args([
            "-c",
            "from mlx_lm import load; import sys; load(sys.argv[1])",
        ])
        .arg(output)
        .stdin(Stdio::null());
    command.envs(env.iter());
    let secrets = sensitive_environment_values(env);
    run_command_bounded(
        command,
        std::time::Duration::from_secs(120),
        64 * 1024,
        &secrets,
    )
    .await
    .context("Converted model failed pinned mlx-lm load validation")?;
    Ok(())
}

async fn run_plan(plan: &ConversionCommandPlan) -> Result<()> {
    let mut command = tokio::process::Command::new(&plan.program);
    command
        .args(&plan.args)
        .envs(plan.env.iter())
        .stdin(Stdio::null());
    run_command_bounded(
        command,
        std::time::Duration::from_secs(2 * 60 * 60),
        128 * 1024,
        &sensitive_environment_values(&plan.env),
    )
    .await?;
    Ok(())
}

async fn run_command_bounded(
    mut command: tokio::process::Command,
    timeout: std::time::Duration,
    output_limit: usize,
    secrets: &[String],
) -> Result<String> {
    use tokio::io::AsyncReadExt;
    command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let mut child = command.spawn()?;
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("Child stdout unavailable"))?;
    let mut stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("Child stderr unavailable"))?;
    async fn drain(
        reader: &mut (impl tokio::io::AsyncRead + Unpin),
        limit: usize,
    ) -> Result<Vec<u8>> {
        let mut captured = Vec::new();
        let mut buffer = [0u8; 8192];
        loop {
            let count = reader.read(&mut buffer).await?;
            if count == 0 {
                break;
            }
            if captured.len() < limit {
                let keep = (limit - captured.len()).min(count);
                captured.extend_from_slice(&buffer[..keep]);
            }
        }
        Ok(captured)
    }
    let operation = async {
        let (stdout_bytes, stderr_bytes, status) = tokio::try_join!(
            drain(&mut stdout, output_limit),
            drain(&mut stderr, output_limit),
            async { Ok::<_, anyhow::Error>(child.wait().await?) }
        )?;
        if !status.success() {
            let mut detail = String::from_utf8_lossy(&stderr_bytes).into_owned();
            for secret in secrets.iter().filter(|secret| !secret.is_empty()) {
                detail = detail.replace(secret, "[REDACTED]");
            }
            bail!(
                "command exited with {status}: {}",
                detail.chars().take(1000).collect::<String>()
            );
        }
        Ok::<String, anyhow::Error>(String::from_utf8_lossy(&stdout_bytes).into_owned())
    };
    tokio::time::timeout(timeout, operation)
        .await
        .context("Command timed out")?
}

fn sensitive_environment_values(environment: &BTreeMap<OsString, OsString>) -> Vec<String> {
    environment
        .iter()
        .filter_map(|(name, value)| {
            let name = name.to_string_lossy().to_ascii_uppercase();
            (name.contains("TOKEN") || name.contains("KEY"))
                .then(|| value.to_string_lossy().into_owned())
        })
        .collect()
}

pub(crate) fn validate_cached_conversion(path: &Path) -> Result<()> {
    let manifest: ConversionManifest =
        serde_json::from_reader(fs::File::open(path.join(MANIFEST_NAME))?)?;
    let provenance = &manifest.provenance;
    if manifest.schema_version != 1
        || provenance.resolver_schema_version != 1
        || provenance.mlx_lm_version != PINNED_MLX_LM_VERSION
    {
        bail!("Converted model does not use the supported resolver and pinned mlx-lm identity");
    }
    if provenance.source_content_sha256.len() != 64
        || !provenance
            .source_content_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        bail!("Converted model source content hash is invalid");
    }
    let expected_key = conversion_key(
        &provenance.source,
        &provenance.revision_or_hash,
        &provenance.source_content_sha256,
        provenance.recipe,
    )?;
    if provenance.cache_key != expected_key
        || path.file_name().and_then(|name| name.to_str()) != Some(expected_key.as_str())
    {
        bail!("Converted model cache key or directory identity is invalid");
    }
    if let AuthoritativeSafetensorsSource::HuggingFaceRepo { repo_id, revision } =
        &provenance.source
    {
        validate_repo(repo_id)?;
        validate_revision(revision)?;
        if revision != &provenance.revision_or_hash
            || revision.len() != 40
            || !revision.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            bail!("Converted Hugging Face provenance is not pinned to its immutable revision");
        }
    }
    validate_published_conversion(path, &manifest.provenance)
}

fn validate_published_conversion(path: &Path, expected: &ConversionProvenance) -> Result<()> {
    if !path.join(COMPLETE_NAME).is_file() {
        bail!("Converted model cache entry is not complete");
    }
    let manifest: ConversionManifest =
        serde_json::from_reader(fs::File::open(path.join(MANIFEST_NAME))?)?;
    if manifest.schema_version != 1 || manifest.provenance != *expected {
        bail!("Converted model provenance does not match the requested source and pinned tool");
    }
    validate_transformers_directory(path)?;
    let actual = manifest_files(path)?;
    if actual != manifest.files {
        bail!("Converted model cache contents do not match the signed manifest closure");
    }
    Ok(())
}

fn manifest_files(path: &Path) -> Result<Vec<ManifestFile>> {
    let mut files = Vec::new();
    let mut stack = vec![path.to_path_buf()];
    while let Some(directory) = stack.pop() {
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            if entry.file_type()?.is_symlink() {
                bail!("Converted model output may not contain symlinks");
            }
            if entry.file_type()?.is_dir() {
                stack.push(entry.path());
                continue;
            }
            if !entry.file_type()?.is_file() {
                continue;
            }
            let relative = entry
                .path()
                .strip_prefix(path)?
                .to_string_lossy()
                .into_owned();
            if relative == MANIFEST_NAME || relative == COMPLETE_NAME {
                continue;
            }
            files.push(ManifestFile {
                path: relative,
                size: entry.metadata()?.len(),
                sha256: hash_file(&entry.path())?,
            });
            if files.len() > 4096 {
                bail!("Converted model contains too many files");
            }
        }
    }
    files.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(files)
}

fn hash_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 128 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(hex_digest(&digest.finalize()))
}

pub(crate) fn model_content_hash(path: &Path, allowed_root: &Path) -> Result<String> {
    let mut assets = validate_model_directory_assets(path, allowed_root)?;
    for name in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "sentencepiece.bpe.model",
        "vocab.json",
        "merges.txt",
        "model.safetensors.index.json",
    ] {
        let candidate = path.join(name);
        if candidate.is_file() {
            assets.push(candidate);
        }
    }
    assets.sort();
    assets.dedup();
    let mut digest = Sha256::new();
    for asset in assets {
        let relative = asset
            .strip_prefix(path)
            .with_context(|| format!("Model asset is outside source: {}", asset.display()))?;
        digest.update(relative.to_string_lossy().as_bytes());
        digest.update(asset.metadata()?.len().to_le_bytes());
        digest.update(hash_file(&asset)?.as_bytes());
    }
    Ok(hex_digest(&digest.finalize()))
}

fn atomic_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let tmp = path.with_extension("json.tmp");
    let mut file = fs::File::create(&tmp)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.flush()?;
    file.sync_all()?;
    fs::rename(tmp, path)?;
    Ok(())
}

fn source_kind_name(source: &RapidMlxModelSource) -> &'static str {
    match source {
        RapidMlxModelSource::MlxDirectory { .. } => "mlx_directory",
        RapidMlxModelSource::HuggingFaceRepo { .. } => "hugging_face_repo",
        RapidMlxModelSource::Alias { .. } => "alias",
        RapidMlxModelSource::AuthoritativeSafetensors { .. } => "authoritative_safetensors",
        RapidMlxModelSource::GgufFile { .. } => "gguf_file",
        RapidMlxModelSource::Unknown { .. } => "unknown",
    }
}

fn display_name(source: &RapidMlxModelSource) -> String {
    match source {
        RapidMlxModelSource::MlxDirectory { path } | RapidMlxModelSource::GgufFile { path } => path
            .file_name()
            .and_then(|v| v.to_str())
            .unwrap_or("model")
            .to_string(),
        RapidMlxModelSource::HuggingFaceRepo { repo_id, .. } => repo_id.clone(),
        RapidMlxModelSource::Alias { value } => value.clone(),
        RapidMlxModelSource::AuthoritativeSafetensors { source, .. } => match source {
            AuthoritativeSafetensorsSource::LocalDirectory { path } => path
                .file_name()
                .and_then(|v| v.to_str())
                .unwrap_or("converted-model")
                .to_string(),
            AuthoritativeSafetensorsSource::HuggingFaceRepo { repo_id, .. } => repo_id.clone(),
        },
        RapidMlxModelSource::Unknown { kind, .. } => format!("Unknown source type: {kind}"),
    }
}

fn alias_warnings(source: &RapidMlxModelSource, context: &RapidMlxResolveContext) -> Vec<String> {
    match source {
        RapidMlxModelSource::Alias { value }
            if !context.verified_aliases.iter().any(|alias| alias == value) => vec!["Free-form alias will be validated by Rapid-MLX at launch; no unstable CLI catalog was scraped".into()],
        _ => Vec::new(),
    }
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

/// Returns the base directory where template overlay directories are stored.
/// Creates it if it does not exist. Returns `None` only if home directory cannot be determined.
static TEMPLATE_OVERLAY_ROOT: OnceLock<PathBuf> = OnceLock::new();

pub fn init_template_overlay_root(config_dir: &Path) {
    let _ = TEMPLATE_OVERLAY_ROOT.set(config_dir.join("rapid-mlx").join("template-overlays"));
}

fn template_overlay_base_dir() -> Option<PathBuf> {
    let base = TEMPLATE_OVERLAY_ROOT.get().cloned().unwrap_or_else(|| {
        crate::paths::AppPaths::default_active_root()
            .join("rapid-mlx")
            .join("template-overlays")
    });
    let _ = std::fs::create_dir_all(&base);
    Some(base)
}

/// Creates a template overlay directory for a Rapid-MLX model when a custom chat template is active.
///
/// The overlay directory contains:
/// - Symlinks to all model files from the original model directory (weights, tokenizer, config, etc.)
/// - A copy of the chat template file as `chat_template.jinja` (takes precedence over native template)
///
/// Returns the overlay directory path. If `chat_template_file` is `None` or empty, returns the original
/// model directory path unchanged (no overlay needed).
///
/// The overlay directory is named by hashing the original model path to avoid collisions and
/// keep directory names stable across launches.
///
/// This preserves the read-only cache contract: no mutation of HF cache or user model directories.
/// Switching template versions means changing the `chat_template.jinja` in the overlay.
pub(crate) fn create_template_overlay(
    model_dir: &str,
    chat_template_file: Option<&str>,
) -> Result<String> {
    // No custom template (None or empty string): use the original model directory directly
    let Some(template_path) = chat_template_file else {
        return Ok(model_dir.to_string());
    };
    if template_path.is_empty() {
        return Ok(model_dir.to_string());
    }

    let model_path = Path::new(model_dir);

    // Validate the model directory exists, and separately that it's a directory. These are
    // distinct failure modes (a bare `Alias` launch_argument hits the first; a launch_argument
    // that resolved to a file would hit the second) and callers classify the resulting launch
    // warning by matching on the message text, so keep the two reasons apart.
    if !model_path.exists() {
        bail!("Model directory does not exist: {}", model_dir);
    }
    if !model_path.is_dir() {
        bail!("Model path is not a directory: {}", model_dir);
    }

    // Validate template file exists and is readable
    let template_content = std::fs::read(template_path)
        .with_context(|| format!("Chat template file not readable: {}", template_path))?;

    // Create overlay directory named by hashing the original model path
    let base =
        template_overlay_base_dir().ok_or_else(|| anyhow!("Could not determine home directory"))?;
    let hash = hex_digest(&Sha256::digest(model_dir.as_bytes()));
    // Include the template path in the suffix so parallel callers with the
    // same model directory cannot overwrite each other's overlay contents.
    // The stable model prefix preserves the existing deterministic/debuggable
    // naming contract.
    let template_hash = hex_digest(&Sha256::digest(template_path.as_bytes()));
    let overlay_dir = base.join(format!("{}-{}", &hash[..16], &template_hash[..8]));
    let _ = std::fs::create_dir_all(&overlay_dir);

    // Symlink all files from the original model directory (skip existing symlinks to rebuild)
    if let Ok(entries) = std::fs::read_dir(model_path) {
        for entry in entries.flatten() {
            let src = entry.path();
            let dest = overlay_dir.join(src.file_name().unwrap());

            // Remove existing symlink or file to rebuild fresh
            let _ = std::fs::remove_file(&dest);

            // Only symlink files (not subdirectories) — Rapid-MLX reads flat model files
            if src.is_file() {
                let _ = create_file_symlink(&src, &dest);
            }
        }
    }

    // Write the chat template as chat_template.jinja (takes precedence per Rapid-MLX's
    // _apply_chat_template_sidecar(): it prefers chat_template.jinja over embedded tokenizer template)
    std::fs::write(overlay_dir.join("chat_template.jinja"), &template_content)
        .with_context(|| "Failed to write chat_template.jinja into overlay directory")?;

    Ok(overlay_dir.to_string_lossy().into_owned())
}

#[cfg(unix)]
fn create_file_symlink(src: &Path, dest: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(src, dest)
}

#[cfg(windows)]
fn create_file_symlink(src: &Path, dest: &Path) -> std::io::Result<()> {
    std::os::windows::fs::symlink_file(src, dest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_alias_source_round_trips_with_canonical_value() {
        for field in ["value", "name"] {
            let wire = serde_json::json!({"kind": "alias", field: "qwen3.8-27b-4bit"});
            let source: RapidMlxModelSource = serde_json::from_value(wire).unwrap();
            assert_eq!(
                source,
                RapidMlxModelSource::Alias {
                    value: "qwen3.8-27b-4bit".into()
                },
                "catalog field {field}"
            );
            assert_eq!(
                serde_json::to_value(source).unwrap(),
                serde_json::json!({"kind": "alias", "value": "qwen3.8-27b-4bit"})
            );
        }
    }

    #[test]
    fn hf_cache_repo_id_extracts_owner_and_repo() {
        let hub = Path::new("/models/cache/huggingface/hub");
        assert_eq!(
            hf_cache_repo_id(&hub.join("models--nightmedia--Qwen3.8-27B-MindMeld-AREX-mxfp4-mlx")),
            Some("nightmedia/Qwen3.8-27B-MindMeld-AREX-mxfp4-mlx".to_string())
        );
        assert_eq!(
            hf_cache_repo_id(
                &hub.join("models--owner--repo")
                    .join("snapshots")
                    .join("abc123")
            ),
            Some("owner/repo".to_string())
        );
        assert_eq!(
            hf_cache_repo_id(Path::new("/models/mlx/native/some-model")),
            None
        );
        assert_eq!(
            hf_cache_repo_id(&hub.join("models--broken")),
            None,
            "container without a -- separator is not a repo"
        );
    }

    /// `init_template_overlay_root` writes a process-global, so tests that set it
    /// must hold this lock for their whole body: another test's root (a tempdir
    /// that has already been deleted) otherwise wins the race and the overlay
    /// lands in the void. Returns the guard; keep it in scope for the test.
    static OVERLAY_ROOT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn test_overlay_root(path: &Path) -> std::sync::MutexGuard<'static, ()> {
        let guard = OVERLAY_ROOT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        init_template_overlay_root(path);
        guard
    }

    #[test]
    fn gguf_is_preserved_but_never_launchable() {
        let source = source_from_legacy_model_path("/models/finetune.gguf").unwrap();
        assert!(matches!(source, RapidMlxModelSource::GgufFile { .. }));
        let context = RapidMlxResolveContext {
            models_dir: "/models".into(),
            python_executable: "python3".into(),
            runtime_version: "0.10.9".into(),
            hf_token: None,
            verified_aliases: vec![],
            execute_conversion: false,
        };
        assert!(matches!(
            preview(&source, &context).state,
            ResolverPreviewState::UnsupportedSource
        ));
        assert!(preview(&source, &context).warnings[0].contains("Rapid-MLX 0.10.9"));
    }

    #[tokio::test]
    async fn explicitly_verified_aliases_are_runtime_version_agnostic() {
        let models = tempfile::tempdir().unwrap();
        let source = RapidMlxModelSource::Alias {
            value: "verified-model".into(),
        };
        for version in ["0.10.9", "0.10.10", "0.10.11"] {
            let context = RapidMlxResolveContext {
                models_dir: models.path().into(),
                python_executable: "python3".into(),
                runtime_version: version.into(),
                hf_token: None,
                verified_aliases: vec!["verified-model".into()],
                execute_conversion: false,
            };
            let resolved = resolve(source.clone(), &context).await.unwrap();
            assert_eq!(
                resolved.source_kind,
                ResolvedRapidMlxSourceKind::VerifiedAlias,
                "{version}"
            );
            assert!(resolved.warnings.is_empty(), "{version}");
        }

        let context = RapidMlxResolveContext {
            models_dir: models.path().into(),
            python_executable: "python3".into(),
            runtime_version: "0.10.11".into(),
            hf_token: None,
            verified_aliases: vec![],
            execute_conversion: false,
        };
        let resolved = resolve(source, &context).await.unwrap();
        assert_eq!(
            resolved.source_kind,
            ResolvedRapidMlxSourceKind::FreeFormAlias
        );
        assert_eq!(resolved.warnings.len(), 1);
    }

    #[test]
    fn gguf_rejection_names_the_selected_runtime_profile() {
        let source = RapidMlxModelSource::GgufFile {
            path: "/models/finetune.gguf".into(),
        };
        for version in ["0.10.10", "0.10.9"] {
            let context = RapidMlxResolveContext {
                models_dir: "/models".into(),
                python_executable: "python3".into(),
                runtime_version: version.into(),
                hf_token: None,
                verified_aliases: vec![],
                execute_conversion: false,
            };
            let preview = preview(&source, &context);
            assert!(matches!(
                preview.state,
                ResolverPreviewState::UnsupportedSource
            ));
            assert!(preview.warnings[0].contains(version), "{version}");
        }
    }

    #[test]
    fn official_command_is_pinned_module_invocation_and_token_is_not_argv() {
        let env = BTreeMap::from([(OsString::from("HF_TOKEN"), OsString::from("secret"))]);
        let plan = conversion_command_plan(
            Path::new("python3"),
            Path::new("/staging/pinned-snapshot"),
            Path::new("out"),
            MlxConversionRecipe::Q6,
            &env,
        );
        let args: Vec<_> = plan.args.iter().map(|v| v.to_string_lossy()).collect();
        assert_eq!(&args[..2], ["-m", "mlx_lm.convert"]);
        assert!(args.windows(2).any(|pair| pair == ["--q-bits", "6"]));
        assert!(!args.iter().any(|arg| arg.contains("secret")));
        assert!(!args.iter().any(|arg| arg == "--revision"));
    }

    #[test]
    fn indexed_safetensors_requires_every_safe_shard() {
        let temp = tempfile::tempdir().unwrap();
        fs::write(temp.path().join("config.json"), r#"{"model_type":"test"}"#).unwrap();
        fs::write(temp.path().join("tokenizer.json"), "{}").unwrap();
        fs::write(temp.path().join("model-00001-of-00002.safetensors"), b"one").unwrap();
        fs::write(temp.path().join("model.safetensors.index.json"), r#"{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}"#).unwrap();
        assert!(validate_transformers_directory(temp.path()).is_err());
        fs::write(temp.path().join("model-00002-of-00002.safetensors"), b"two").unwrap();
        assert!(validate_transformers_directory(temp.path()).is_ok());
    }

    /// Minimal safetensors file: only the header is ever parsed, so the tensor
    /// bodies are omitted entirely.
    fn write_safetensors(path: &Path, tensors: &[&str]) {
        let entries: Vec<String> = tensors
            .iter()
            .map(|name| format!(r#""{name}":{{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#))
            .collect();
        let header = format!(r#"{{{}}}"#, entries.join(","));
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        fs::write(path, bytes).unwrap();
    }

    fn trunk_directory() -> tempfile::TempDir {
        let temp = tempfile::tempdir().unwrap();
        fs::write(temp.path().join("config.json"), r#"{"model_type":"test"}"#).unwrap();
        fs::write(temp.path().join("tokenizer.json"), "{}").unwrap();
        write_safetensors(
            &temp.path().join("model.safetensors"),
            &["model.layers.0.mlp.gate_proj.weight"],
        );
        temp
    }

    #[test]
    fn clean_trunk_directory_is_accepted() {
        let temp = trunk_directory();
        reject_in_trunk_mtp_sidecar(temp.path()).unwrap();
        assert!(validate_transformers_directory(temp.path()).is_ok());
    }

    #[test]
    fn in_trunk_mtp_sidecar_is_refused_with_remediation() {
        let temp = trunk_directory();
        write_safetensors(
            &temp.path().join("model-mtp.safetensors"),
            &["mtp.pre_fc_norm_embedding.weight", "mtp.fc.weight"],
        );
        let error = reject_in_trunk_mtp_sidecar(temp.path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("model-mtp.safetensors"), "{error}");
        assert!(error.contains("Remediation"), "{error}");
        // The launch path must inherit the refusal, not just the helper.
        assert!(validate_transformers_directory(temp.path()).is_err());
    }

    #[test]
    fn embedded_mtp_checkpoint_still_launches() {
        let temp = trunk_directory();
        // A shard that mixes draft-head and trunk tensors is a legitimate
        // embedded-MTP checkpoint, not a misplaced sidecar.
        write_safetensors(
            &temp.path().join("model-00002-of-00002.safetensors"),
            &["mtp.fc.weight", "model.layers.1.mlp.gate_proj.weight"],
        );
        reject_in_trunk_mtp_sidecar(temp.path()).unwrap();
    }

    #[test]
    fn mtp_weights_outside_the_trunk_glob_are_ignored() {
        let temp = trunk_directory();
        // Not matched by mlx_lm's `model*.safetensors` glob, so the trunk loader
        // never sees it and the norm shift is never triggered.
        write_safetensors(&temp.path().join("mtp.safetensors"), &["mtp.fc.weight"]);
        reject_in_trunk_mtp_sidecar(temp.path()).unwrap();
    }

    #[test]
    fn implausible_safetensors_header_length_does_not_allocate() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("model.safetensors");
        let mut bytes = u64::MAX.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"junk");
        fs::write(&path, bytes).unwrap();
        assert!(safetensors_tensor_names(&path).is_err());
        // An unreadable candidate is not evidence of this defect.
        reject_in_trunk_mtp_sidecar(temp.path()).unwrap();
    }

    #[test]
    fn content_hash_validates_and_reads_the_safetensors_index_once() {
        let temp = tempfile::tempdir().unwrap();
        fs::write(temp.path().join("config.json"), r#"{"model_type":"test"}"#).unwrap();
        fs::write(temp.path().join("tokenizer.json"), "{}").unwrap();
        fs::write(
            temp.path().join("model-00001-of-00001.safetensors"),
            b"weights",
        )
        .unwrap();
        fs::write(
            temp.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"a":"model-00001-of-00001.safetensors"}}"#,
        )
        .unwrap();
        SAFETENSORS_INDEX_SCANS.with(|count| count.set(0));
        let hash = model_content_hash(temp.path(), temp.path()).unwrap();
        assert_eq!(hash.len(), 64);
        assert_eq!(SAFETENSORS_INDEX_SCANS.with(std::cell::Cell::get), 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn resolver_blocking_work_does_not_stall_tokio_and_is_bounded() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let runtime_thread = std::thread::current().id();
        let active = Arc::new(AtomicUsize::new(0));
        let maximum = Arc::new(AtomicUsize::new(0));
        let operations = (0..3).map(|_| {
            let active = active.clone();
            let maximum = maximum.clone();
            run_resolver_blocking(move || {
                assert_ne!(std::thread::current().id(), runtime_thread);
                let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                maximum.fetch_max(current, Ordering::SeqCst);
                std::thread::sleep(std::time::Duration::from_millis(80));
                active.fetch_sub(1, Ordering::SeqCst);
                Ok(())
            })
        });
        let heartbeat = async {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            true
        };
        let (results, heartbeat) =
            tokio::join!(futures_util::future::join_all(operations), heartbeat);
        assert!(heartbeat);
        assert!(results.into_iter().all(|result| result.is_ok()));
        assert_eq!(maximum.load(Ordering::SeqCst), MAX_BLOCKING_RESOLVER_TASKS);
    }

    #[test]
    fn app_staging_directory_is_never_launchable() {
        let library = tempfile::tempdir().unwrap();
        let staged = library
            .path()
            .join(".staging/conversions/interrupted/model");
        fs::create_dir_all(&staged).unwrap();
        fs::write(staged.join("config.json"), r#"{"model_type":"test"}"#).unwrap();
        fs::write(staged.join("tokenizer.json"), "{}").unwrap();
        fs::write(staged.join("model.safetensors"), b"partial").unwrap();
        let source = RapidMlxModelSource::MlxDirectory { path: staged };
        let context = RapidMlxResolveContext {
            models_dir: library.path().to_path_buf(),
            python_executable: "python3".into(),
            runtime_version: "0.10.9".into(),
            hf_token: None,
            verified_aliases: vec![],
            execute_conversion: false,
        };

        let result = preview(&source, &context);
        assert!(matches!(result.state, ResolverPreviewState::Invalid));
        assert!(result.warnings[0].contains("staging directory"));
    }

    #[cfg(unix)]
    #[test]
    fn local_hf_snapshot_accepts_blob_symlinks_within_its_repository() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path().join("models--owner--model");
        let blobs = repo.join("blobs");
        let snapshot = repo.join("snapshots/abcdef0");
        fs::create_dir_all(&blobs).unwrap();
        fs::create_dir_all(&snapshot).unwrap();
        fs::write(blobs.join("config"), r#"{"model_type":"test"}"#).unwrap();
        fs::write(blobs.join("tokenizer"), "{}").unwrap();
        fs::write(blobs.join("weights"), b"weights").unwrap();
        symlink("../../blobs/config", snapshot.join("config.json")).unwrap();
        symlink("../../blobs/tokenizer", snapshot.join("tokenizer.json")).unwrap();
        symlink("../../blobs/weights", snapshot.join("model.safetensors")).unwrap();
        assert!(canonical_model_directory(&snapshot, &local_model_allowed_root(&snapshot)).is_ok());
    }

    #[test]
    fn typed_source_round_trips_without_resolved_launch_argument() {
        let source = RapidMlxModelSource::HuggingFaceRepo {
            repo_id: "owner/model".into(),
            revision: "abc1234".into(),
        };
        let value = serde_json::to_value(&source).unwrap();
        assert_eq!(value["kind"], "hugging_face_repo");
        assert!(value.get("launch_argument").is_none());
        assert_eq!(
            serde_json::from_value::<RapidMlxModelSource>(value).unwrap(),
            source
        );
    }

    #[test]
    fn sensitive_environment_values_include_tokens_but_not_cache_paths() {
        let environment = BTreeMap::from([
            (OsString::from("HF_TOKEN"), OsString::from("secret")),
            (
                OsString::from("HF_HUB_CACHE"),
                OsString::from("/models/cache"),
            ),
        ]);
        assert_eq!(sensitive_environment_values(&environment), ["secret"]);
    }

    #[test]
    fn published_conversion_rejects_unmanifested_files() {
        let temp = tempfile::tempdir().unwrap();
        fs::write(temp.path().join("config.json"), r#"{"model_type":"test"}"#).unwrap();
        fs::write(temp.path().join("tokenizer.json"), "{}").unwrap();
        fs::write(temp.path().join("model.safetensors"), b"weights").unwrap();
        let provenance = ConversionProvenance {
            resolver_schema_version: 1,
            source: AuthoritativeSafetensorsSource::LocalDirectory {
                path: PathBuf::from("/source"),
            },
            revision_or_hash: "abcdef0".into(),
            source_content_sha256: "content".into(),
            mlx_lm_version: PINNED_MLX_LM_VERSION.into(),
            recipe: MlxConversionRecipe::Fp16,
            cache_key: "cache".into(),
        };
        let manifest = ConversionManifest {
            schema_version: 1,
            provenance: provenance.clone(),
            files: manifest_files(temp.path()).unwrap(),
        };
        atomic_json(&temp.path().join(MANIFEST_NAME), &manifest).unwrap();
        fs::write(temp.path().join(COMPLETE_NAME), b"").unwrap();
        validate_published_conversion(temp.path(), &provenance).unwrap();
        fs::write(temp.path().join("unexpected.bin"), b"tampered").unwrap();
        assert!(validate_published_conversion(temp.path(), &provenance).is_err());
    }

    #[test]
    fn converted_inventory_launch_path_revalidates_manifest_hashes() {
        let library = tempfile::tempdir().unwrap();
        let source_identity = AuthoritativeSafetensorsSource::LocalDirectory {
            path: PathBuf::from("/source"),
        };
        let content_hash = "a".repeat(64);
        let cache_key = conversion_key(
            &source_identity,
            "abcdef0",
            &content_hash,
            MlxConversionRecipe::Fp16,
        )
        .unwrap();
        let provenance = ConversionProvenance {
            resolver_schema_version: 1,
            source: source_identity,
            revision_or_hash: "abcdef0".into(),
            source_content_sha256: content_hash,
            mlx_lm_version: PINNED_MLX_LM_VERSION.into(),
            recipe: MlxConversionRecipe::Fp16,
            cache_key: cache_key.clone(),
        };
        let converted = library.path().join("mlx/converted").join(cache_key);
        fs::create_dir_all(&converted).unwrap();
        fs::write(converted.join("config.json"), r#"{"model_type":"test"}"#).unwrap();
        fs::write(converted.join("tokenizer.json"), "{}").unwrap();
        fs::write(converted.join("model.safetensors"), b"weights").unwrap();
        atomic_json(
            &converted.join(MANIFEST_NAME),
            &ConversionManifest {
                schema_version: 1,
                provenance: provenance.clone(),
                files: manifest_files(&converted).unwrap(),
            },
        )
        .unwrap();
        fs::write(converted.join(COMPLETE_NAME), b"").unwrap();
        let source = RapidMlxModelSource::MlxDirectory {
            path: converted.clone(),
        };
        let context = RapidMlxResolveContext {
            models_dir: library.path().to_path_buf(),
            python_executable: "python3".into(),
            runtime_version: "0.10.9".into(),
            hf_token: None,
            verified_aliases: vec![],
            execute_conversion: false,
        };
        assert!(matches!(
            preview(&source, &context).state,
            ResolverPreviewState::Ready
        ));

        let mut wrong_tool = provenance.clone();
        wrong_tool.mlx_lm_version = "99.0.0".into();
        atomic_json(
            &converted.join(MANIFEST_NAME),
            &ConversionManifest {
                schema_version: 1,
                provenance: wrong_tool,
                files: manifest_files(&converted).unwrap(),
            },
        )
        .unwrap();
        assert!(matches!(
            preview(&source, &context).state,
            ResolverPreviewState::Invalid
        ));

        atomic_json(
            &converted.join(MANIFEST_NAME),
            &ConversionManifest {
                schema_version: 1,
                provenance,
                files: manifest_files(&converted).unwrap(),
            },
        )
        .unwrap();

        fs::write(converted.join("model.safetensors"), b"tampered").unwrap();
        assert!(matches!(
            preview(&source, &context).state,
            ResolverPreviewState::Invalid
        ));
    }

    #[test]
    fn needs_trust_remote_code_data_only_safe() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("config.json"),
            r#"{"architectures":["MLXModel"]}"#,
        )
        .unwrap();
        fs::write(dir.path().join("model.safetensors"), b"weights").unwrap();
        assert!(!needs_trust_remote_code(dir.path()).unwrap());
    }

    #[test]
    fn needs_trust_remote_code_main_class_detected() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("config.json"),
            r#"{"main_class":"CustomModel"}"#,
        )
        .unwrap();
        assert!(needs_trust_remote_code(dir.path()).unwrap());
    }

    #[test]
    fn needs_trust_remote_code_auto_map_custom_detected() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("config.json"),
            r#"{"auto_map":{"AutoModelForCausalLM":"my_module.MyModel"}}"#,
        )
        .unwrap();
        assert!(needs_trust_remote_code(dir.path()).unwrap());
    }

    #[test]
    fn needs_trust_remote_code_auto_map_standard_safe() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("config.json"),
            r#"{"auto_map":{"AutoModelForCausalLM":"transformers.models.qwen2.Qwen2ForCausalLM"}}"#,
        )
        .unwrap();
        assert!(!needs_trust_remote_code(dir.path()).unwrap());
    }

    #[test]
    fn needs_trust_remote_code_modeling_py_detected() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("modeling_custom.py"), b"def run(): pass").unwrap();
        assert!(needs_trust_remote_code(dir.path()).unwrap());
    }

    #[test]
    fn needs_trust_remote_code_main_py_detected() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("main.py"), b"print('hi')").unwrap();
        assert!(needs_trust_remote_code(dir.path()).unwrap());
    }

    #[test]
    fn needs_trust_remote_code_config_parse_error_safe() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("config.json"), b"not json").unwrap();
        assert!(!needs_trust_remote_code(dir.path()).unwrap());
    }

    #[test]
    fn unknown_source_kind_roundtrips_and_is_never_launchable() {
        let source: RapidMlxModelSource = serde_json::from_value(serde_json::json!({
            "kind": "future_registry_source",
            "canonical_identity": "registry://future/model",
            "future_field": {"preserve": true}
        }))
        .unwrap();
        let view = RapidMlxModelSourceView::from_source(&source);
        assert_eq!(view.kind, "unknown");
        assert!(!view.launchable);
        assert_eq!(
            serde_json::to_value(source).unwrap()["future_field"]["preserve"],
            true
        );
    }

    #[test]
    fn template_overlay_none_returns_original_path() {
        let result = create_template_overlay("/some/model/dir", None).unwrap();
        assert_eq!(result, "/some/model/dir");
    }

    #[test]
    fn template_overlay_empty_string_returns_original_path() {
        let result = create_template_overlay("/some/model/dir", Some("")).unwrap();
        assert_eq!(result, "/some/model/dir");
    }

    #[test]
    fn template_overlay_missing_model_dir_fails() {
        let result = create_template_overlay("/nonexistent/model", Some("/some/template.jinja"));
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Model directory does not exist")
        );
    }

    #[test]
    fn template_overlay_missing_template_file_fails() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("config.json"), b"{}").unwrap();
        let result = create_template_overlay(
            dir.path().to_string_lossy().as_ref(),
            Some("/nonexistent/template.jinja"),
        );
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Chat template file not readable")
        );
    }

    #[test]
    fn template_overlay_creates_symlinks_and_template() {
        // Create a fake model directory with some files
        let model_dir = tempfile::tempdir().unwrap();
        let _overlay_guard = test_overlay_root(model_dir.path());
        fs::write(model_dir.path().join("config.json"), b"{}").unwrap();
        fs::write(model_dir.path().join("tokenizer.json"), b"{}").unwrap();
        fs::write(model_dir.path().join("model.safetensors"), b"weights").unwrap();

        // Create a fake template file
        let template_dir = tempfile::tempdir().unwrap();
        let template_path = template_dir.path().join("my-template.jinja");
        fs::write(&template_path, b"user provided template content").unwrap();

        // Create overlay
        let overlay_path = create_template_overlay(
            model_dir.path().to_string_lossy().as_ref(),
            Some(template_path.to_string_lossy().as_ref()),
        )
        .unwrap();

        let overlay = Path::new(&overlay_path);

        // Overlay directory exists
        assert!(overlay.is_dir());

        // chat_template.jinja exists with the correct content
        let template_content = fs::read(overlay.join("chat_template.jinja")).unwrap();
        assert_eq!(template_content, b"user provided template content");

        // Symlinks exist for model files
        assert!(overlay.join("config.json").exists());
        assert!(overlay.join("tokenizer.json").exists());
        assert!(overlay.join("model.safetensors").exists());

        // Symlinks point to original files (same content)
        assert_eq!(fs::read(overlay.join("config.json")).unwrap(), b"{}");
        assert_eq!(fs::read(overlay.join("tokenizer.json")).unwrap(), b"{}");
        assert_eq!(
            fs::read(overlay.join("model.safetensors")).unwrap(),
            b"weights"
        );
    }

    #[test]
    fn template_overlay_does_not_mutate_original_model_dir() {
        let model_dir = tempfile::tempdir().unwrap();
        // Production points the overlay root at the app config dir, never at a
        // model directory — the overlay base itself (`rapid-mlx/template-overlays`)
        // is created on demand and must not count as model-dir mutation.
        let config_dir = tempfile::tempdir().unwrap();
        let _overlay_guard = test_overlay_root(config_dir.path());
        fs::write(model_dir.path().join("config.json"), b"{}").unwrap();

        let template_dir = tempfile::tempdir().unwrap();
        let template_path = template_dir.path().join("my-template.jinja");
        fs::write(&template_path, b"template content").unwrap();

        // Count files in original dir before
        let files_before: Vec<_> = fs::read_dir(model_dir.path()).unwrap().collect();

        // Create overlay
        let _overlay_path = create_template_overlay(
            model_dir.path().to_string_lossy().as_ref(),
            Some(template_path.to_string_lossy().as_ref()),
        )
        .unwrap();

        // Count files in original dir after
        let files_after: Vec<_> = fs::read_dir(model_dir.path()).unwrap().collect();

        // Original dir should have exactly the same files (no chat_template.jinja
        // added, no overlay base created inside it).
        let names_before: Vec<_> = files_before
            .iter()
            .flatten()
            .map(|e| e.file_name())
            .collect();
        let names_after: Vec<_> = files_after
            .iter()
            .flatten()
            .map(|e| e.file_name())
            .collect();
        assert_eq!(
            names_before, names_after,
            "create_template_overlay mutated the original model directory"
        );
        assert!(!model_dir.path().join("chat_template.jinja").exists());
    }

    #[test]
    fn template_overlay_path_is_deterministic() {
        // Same model dir should always produce the same overlay path
        let model_dir = tempfile::tempdir().unwrap();
        let _overlay_guard = test_overlay_root(model_dir.path());
        let model_path = model_dir.path().to_string_lossy().into_owned();
        let template_dir = tempfile::tempdir().unwrap();
        fs::write(template_dir.path().join("template.jinja"), b"template").unwrap();
        let template_path = template_dir
            .path()
            .join("template.jinja")
            .to_string_lossy()
            .into_owned();

        // First call
        let path1 = create_template_overlay(&model_path, Some(&template_path)).unwrap();

        // Second call with same inputs
        let path2 = create_template_overlay(&model_path, Some(&template_path)).unwrap();

        assert_eq!(path1, path2);
        let model_hash = hex_digest(&Sha256::digest(model_path.as_bytes()));
        assert!(path1.contains(&model_hash[..16]));
    }
}
