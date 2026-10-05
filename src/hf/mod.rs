//! HuggingFace Hub integration.
//!
//! Provides:
//! - Model search (keyword + author filter + sort)
//! - Author/org model browsing
//! - GGUF file listing with real sizes (HF tree API)
//! - Quant-type classification (standard / imatrix / unsloth-dynamic)
//! - Quant provider detection (bartowski / mradermacher / unsloth / etc.)
//! - Streaming download with resume support
//! - HF token management

pub mod mtp_pin_cache;
pub mod qualify;
#[allow(unused_imports)]
pub use qualify::{
    HfConfigEvidence, HfIdentity, HfIdentityConverter, HfIdentityEntity, HfIdentityRole,
    HfQualification, HfRuntimeSnapshot, IdentityRequest, QualifyRequest,
};

use anyhow::{Context, Result};
use hf_hub::{HFClient, HFClientSync};
use percent_encoding::{NON_ALPHANUMERIC, utf8_percent_encode};
use std::path::Path;
use std::sync::LazyLock;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;

fn hf_build_client(token: Option<String>) -> anyhow::Result<HFClientSync> {
    let async_client = if let Some(tok) = token {
        HFClient::builder().token(tok).build()
    } else {
        HFClient::new()
    }
    .map_err(|e| anyhow::anyhow!("Failed to build HF async client: {e}"))?;
    HFClientSync::from_inner(async_client)
        .map_err(|e| anyhow::anyhow!("Failed to build HF sync client: {e}"))
}

pub fn hf_resolve_download_url(repo_id: &str, file_path: &str) -> String {
    hf_resolve_download_url_at(repo_id, file_path, "main")
}

pub fn hf_resolve_download_url_at(repo_id: &str, file_path: &str, revision: &str) -> String {
    let encoded_path = utf8_percent_encode(file_path, NON_ALPHANUMERIC).to_string();
    format!("https://huggingface.co/{repo_id}/resolve/{revision}/{encoded_path}")
}

fn validate_hf_revision(revision: &str) -> Result<(), String> {
    if revision.is_empty()
        || revision.len() > 128
        || !revision
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err("Hugging Face revision must contain only safe revision characters".into());
    }
    Ok(())
}

static HF_HTTP_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(|| {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new())
});

// ── Process-lifetime cache for per-repo file sizes ────────────────────────────
//
// HF tree-API lookups (fetch_file_sizes) are the main source of rate-limit
// pressure: repeated searches/pagination re-request the same repos. File
// sizes are effectively immutable for a given repo, so cache them for the
// life of the process rather than re-fetching every time.
static HF_SIZE_CACHE: LazyLock<
    std::sync::Mutex<std::collections::HashMap<String, std::collections::HashMap<String, u64>>>,
> = LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// Byte width for a safetensors dtype string (e.g. "BF16", "U32", "I8").
fn safetensors_dtype_bytes(dtype: &str) -> f64 {
    match dtype.to_ascii_uppercase().as_str() {
        "F64" | "I64" | "U64" => 8.0,
        "F32" | "I32" | "U32" => 4.0,
        "F16" | "BF16" | "I16" | "U16" => 2.0,
        "I8" | "U8" | "F8_E4M3" | "F8_E5M2" | "BOOL" => 1.0,
        "I4" | "U4" | "F4" => 0.5,
        _ => 2.0, // unknown dtype: assume 2 bytes/param (bf16-class), the common case
    }
}

/// Compute total weight bytes from a search result's `safetensors.parameters` map
/// (present when the request includes `expand[]=safetensors`). Avoids the need
/// for a separate per-repo HF tree-API call for the common case.
///
/// Returns `None` for repos holding packed integer weights (`U32`/`I32`, as MLX
/// quantized repos do): the map counts *elements*, and each 32-bit word packs several
/// low-bit weights, so elements × 4 bytes overstates the size about 7× for 4-bit. The
/// caller then falls back to the exact file sizes from the tree API.
fn safetensors_total_bytes(item: &serde_json::Value) -> Option<u64> {
    let params = item.get("safetensors")?.get("parameters")?.as_object()?;
    if params
        .keys()
        .any(|d| matches!(d.to_ascii_uppercase().as_str(), "U32" | "I32"))
    {
        return None;
    }
    let mut total = 0.0_f64;
    for (dtype, count) in params {
        total += count.as_u64()? as f64 * safetensors_dtype_bytes(dtype);
    }
    (total > 0.0).then_some(total as u64)
}

#[cfg(test)]
mod safetensors_size_tests {
    use super::safetensors_total_bytes;
    use serde_json::json;

    #[test]
    fn bf16_repo_is_sized_from_parameters() {
        let item = json!({"safetensors": {"parameters": {"BF16": 1000}}});
        assert_eq!(safetensors_total_bytes(&item), Some(2000));
    }

    #[test]
    fn packed_mlx_weights_defer_to_exact_file_sizes() {
        // nightmedia/Qwen3.8-27B-MindMeld-AREX-mxfp4-mlx: 26.9B packed U32 elements
        // is a ~15 GB repo, not 107 GB.
        let item =
            json!({"safetensors": {"parameters": {"U32": 26893352960u64, "BF16": 463375600u64}}});
        assert_eq!(safetensors_total_bytes(&item), None);
    }
}

// ── Search sort options ───────────────────────────────────────────────────────

/// Sort order for HF model search / author browse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum HfSort {
    #[default]
    Downloads, // most downloaded (best signal for quality community quants)
    Likes,        // most liked
    CreatedAt,    // newest first
    LastModified, // most recently updated — the useful one for quant repos, which are
    // re-uploaded long after they were created
    Trending,  // HF trending score
    Relevance, // whatever the search query scores highest; only meaningful with a query
}

impl HfSort {
    /// The `sort` value to send to the HF API, or `None` to send no `sort` at all.
    ///
    /// Relevance is not a sort key on HF's side: it is what you get back when you supply a
    /// `search` term and *omit* `sort`. Passing any explicit key replaces the relevance
    /// ordering, so the only way to honour "Relevance" is to send nothing.
    fn as_api_str(self) -> Option<&'static str> {
        match self {
            HfSort::Downloads => Some("downloads"),
            HfSort::Likes => Some("likes"),
            HfSort::CreatedAt => Some("createdAt"),
            HfSort::LastModified => Some("lastModified"),
            HfSort::Trending => Some("trendingScore"),
            HfSort::Relevance => None,
        }
    }
}

// ── Quant-type and provider classification ────────────────────────────────────

/// How the GGUF file was quantized / what calibration was used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "snake_case")]
#[allow(dead_code)]
pub enum QuantFileType {
    #[default]
    Standard, // standard llama.cpp quants (Q4_K_M, Q5_K_M, Q8_0, …)
    Imatrix,        // importance-matrix calibrated (typically mradermacher's i1-* naming)
    UnslothDynamic, // Unsloth UD-* dynamic quants (mixed bpw per layer)
    BnB,            // bitsandbytes (rare in GGUF land but worth flagging)
    Unknown,
}

#[allow(dead_code)]
impl QuantFileType {
    #[allow(dead_code)]
    pub fn label(self) -> &'static str {
        match self {
            QuantFileType::Standard => "Standard",
            QuantFileType::Imatrix => "imatrix",
            QuantFileType::UnslothDynamic => "UD (Unsloth)",
            QuantFileType::BnB => "BnB",
            QuantFileType::Unknown => "Unknown",
        }
    }

    #[allow(dead_code)]
    pub fn description(self) -> &'static str {
        match self {
            QuantFileType::Standard => "Standard llama.cpp quantization",
            QuantFileType::Imatrix => {
                "Importance-matrix calibrated — generally better quality at same size"
            }
            QuantFileType::UnslothDynamic => {
                "Unsloth dynamic quant — per-layer mixed bpw, excellent quality/size tradeoff"
            }
            QuantFileType::BnB => "bitsandbytes quantization",
            QuantFileType::Unknown => "Unknown quantization type",
        }
    }
}

/// Who made the quantization.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "snake_case")]
#[allow(dead_code)]
pub enum QuantProvider {
    Bartowski,       // bartowski — standard Q4_K_M/Q5_K_M/Q8_0, extremely reliable
    Mradermacher,    // mradermacher — imatrix specialist, i1-* naming
    Unsloth,         // Unsloth — UD dynamic quants, best quality/size
    Lmstudio,        // LM Studio community quants
    TheBlokeRetired, // TheBloke (retired, legacy quants still widely used)
    #[default]
    Community, // other community quantizer
    Official,        // first-party / model author's own quants
}

#[allow(dead_code)]
impl QuantProvider {
    #[allow(dead_code)]
    pub fn label(&self) -> &'static str {
        match self {
            QuantProvider::Bartowski => "bartowski",
            QuantProvider::Mradermacher => "mradermacher",
            QuantProvider::Unsloth => "unsloth",
            QuantProvider::Lmstudio => "LM Studio",
            QuantProvider::TheBlokeRetired => "TheBloke",
            QuantProvider::Community => "community",
            QuantProvider::Official => "official",
        }
    }

    #[allow(dead_code)]
    pub fn description(&self) -> &'static str {
        match self {
            QuantProvider::Bartowski => {
                "Standard quants — Q4_K_M through Q8_0. Fast, reliable, no imatrix."
            }
            QuantProvider::Mradermacher => {
                "imatrix expert. i1-* files use importance calibration for better quality at same bpw."
            }
            QuantProvider::Unsloth => {
                "UD dynamic quants — mixed bpw per layer. Excellent quality/size. Also fine-tunes."
            }
            QuantProvider::Lmstudio => "LM Studio community quants.",
            QuantProvider::TheBlokeRetired => {
                "TheBloke (retired). Legacy GGUF quants; well-tested but not the latest models."
            }
            QuantProvider::Community => "Community quantizer.",
            QuantProvider::Official => "Model author's own quantization.",
        }
    }

    /// Infer provider from HF repo owner username.
    pub fn from_username(username: &str) -> Self {
        match username.to_ascii_lowercase().as_str() {
            "bartowski" => QuantProvider::Bartowski,
            "mradermacher" => QuantProvider::Mradermacher,
            "unsloth" => QuantProvider::Unsloth,
            "lmstudio-community" | "lmstudio" => QuantProvider::Lmstudio,
            "davidau" | "davidau-hf" => QuantProvider::Community,
            "mudler" => QuantProvider::Community,
            "jackrong" => QuantProvider::Community,
            "prithivmlmods" => QuantProvider::Community,
            _ => QuantProvider::Community,
        }
    }
}

/// Detect quant file type from GGUF filename patterns.
#[allow(dead_code)]
pub fn detect_quant_type(filename: &str) -> QuantFileType {
    let lower = filename.to_ascii_lowercase();
    // Unsloth UD: "UD-" anywhere or "-UD-" — e.g. "Qwen3.6-27B-UD-Q4_K_S.gguf"
    if lower.contains("-ud-") || lower.contains("/ud-") || lower.starts_with("ud-") {
        return QuantFileType::UnslothDynamic;
    }
    // Unsloth compact/apex variants
    if lower.contains("-unsloth-") || lower.contains("_unsloth_") {
        return QuantFileType::UnslothDynamic;
    }
    // Mradermacher imatrix: i1- prefix or .i1- pattern
    // e.g. "model.i1-Q4_K_M.gguf" or "model-i1-IQ3_S.gguf"
    if lower.contains(".i1-") || lower.contains("-i1-") {
        return QuantFileType::Imatrix;
    }
    // Generic imatrix hint
    if lower.contains("imatrix") || lower.contains(".imat") {
        return QuantFileType::Imatrix;
    }
    QuantFileType::Standard
}

// ── Model info structs ────────────────────────────────────────────────────────

/// Model result from search or author browse.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SimpleModelInfo {
    pub id: String,
    /// Commit currently returned by the Hub search API for this repository.
    /// Browse selections use this immutable revision instead of silently
    /// falling back to the mutable `main` branch.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub revision: String,
    pub gated: bool,
    pub tags: Vec<String>,
    pub downloads: u64,
    pub likes: u64,
    /// ISO 8601 creation timestamp from HF.
    pub created_at: String,
    /// ISO 8601 last-modified timestamp from HF (more useful than created_at for quant repos).
    pub last_modified: String,
    /// Author/org username (the part before "/" in the repo id).
    pub author: String,
    /// Inferred from author username.
    pub quant_provider: QuantProvider,
    /// True if this is an imatrix or UD quant repo (detected from tags or name).
    pub has_imatrix: bool,
    /// Approximate parameter count inferred from model name (0 if unknown).
    pub param_b: f64,
    /// Base model the quantization derives from (from model card metadata).
    pub base_model: String,
    /// Model format from HF filter used during search: "mlx" or "gguf".
    pub format: String,
    /// Repo size on disk (from HF API), useful for MLX models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_size_bytes: Option<u64>,
    /// Quant label for MLX models (e.g. "MXFP4", "Q4"). Empty for GGUF (use file list instead).
    #[serde(skip_serializing_if = "String::is_empty")]
    pub quant_label: String,
}

/// A GGUF file in an HF repo, with real file size and quant classification.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct HfGgufFile {
    /// Repo that owns this file. Companion projectors may come from a linked
    /// static-quant repo rather than the requested imatrix repo.
    pub repo_id: String,
    /// Path within the repo (e.g. "model-Q4_K_M.gguf").
    pub path: String,
    /// File size in bytes (0 if unknown). Fetched from HF tree API.
    pub size: u64,
    /// Short quant label (e.g. "Q4_K_M").
    pub label: String,
    /// Quant type classification.
    pub quant_type: QuantFileType,
    /// True if this is an importance-matrix calibrated file.
    pub is_imatrix: bool,
    /// mmproj companion hint: true if this looks like a vision projector.
    pub is_mmproj: bool,
    /// True when this projector's quant matches the preferred format for the model family.
    pub is_recommended_mmproj: bool,
    /// Short explanation for the family-specific projector recommendation.
    pub mmproj_recommendation: String,
    /// MTP assistant / draft model file: used with --model-draft for speculative decoding.
    pub is_draft_assistant: bool,
}

/// High-level model info from HF (for the model info endpoint).
#[derive(Debug, Clone, serde::Serialize)]
#[allow(dead_code)]
pub struct HfModelInfo {
    pub repo_id: String,
    pub gated: bool,
    pub private: bool,
    pub tags: Vec<String>,
}

/// A single file in an HF repo.
#[derive(Debug, Clone, serde::Serialize)]
#[allow(dead_code)]
pub struct HfFileInfo {
    pub r#type: String,
    pub path: String,
    pub size: Option<u64>,
}

// ── Quantizer shim types (backward compat) ────────────────────────────────────
//
// These are kept solely for the /api/hf/quantizers endpoint contract:
// the frontend still expects UserQuantizer-shaped objects (username,
// display_name, description, quant_style, note). The actual source of
// truth is the CommunitySourceCatalog; `to_quantizers()` in
// `community_source_catalog` derives this shape on the fly.

/// Shape expected by frontend for quick-pick buttons and the quantizer editor.
///
/// Backward-compatible with legacy hf-quantizers.json for migration.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UserQuantizer {
    pub username: String,
    pub display_name: String,
    pub description: String,
    pub quant_style: String, // "standard" | "imatrix" | "ud" | "mlx"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// Load user-customized quantizers from `config_dir/hf-quantizers.json`.
/// Used only by the community-source-catalog migration path.
pub fn load_user_quantizers(config_dir: &std::path::Path) -> Option<Vec<UserQuantizer>> {
    let path = config_dir.join("hf-quantizers.json");
    let text = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&text).ok()
}

// ── Core API functions ────────────────────────────────────────────────────────

/// Get basic model info for a repo (async, uses configured HF token).
#[allow(dead_code)]
pub async fn hf_get_model_info(repo_id: &str) -> Result<HfModelInfo> {
    let token = hf_load_token();
    let url = format!("https://huggingface.co/api/models/{repo_id}");
    let mut req = HF_HTTP_CLIENT.get(&url);
    if let Some(ref tok) = token {
        req = req.bearer_auth(tok);
    }

    let resp = req.send().await.context("Failed to call HF models API")?;
    if !resp.status().is_success() {
        return Ok(HfModelInfo {
            repo_id: repo_id.to_string(),
            gated: false,
            private: false,
            tags: Vec::new(),
        });
    }

    let body: serde_json::Value = resp
        .json()
        .await
        .context("Failed to parse HF models API response")?;
    Ok(HfModelInfo {
        repo_id: repo_id.to_string(),
        gated: body.get("gated").and_then(|v| v.as_bool()).unwrap_or(false),
        private: body
            .get("private")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        tags: body
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|t| t.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default(),
    })
}

/// Read-only preflight for an external speculative-decoding (MTP) companion
/// model reference. Resolves `repo_id` to its immutable commit `sha` and
/// checks — without downloading model weights — whether the repo would
/// require `trust_remote_code` to load (custom loader scripts, or a
/// `config.json` declaring `main_class`/non-standard `auto_map` entries).
///
/// This only surfaces facts; it does not grant or check consent. The actual
/// launch-time gate is `validate_trust_consent` in
/// `inference::rapid_mlx::command`, which still fail-closes on any mismatch.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeculativeModelPreflight {
    pub repo_id: String,
    pub revision: String,
    pub trust_remote_code_required: bool,
    /// Rough estimate of companion memory in bytes. Derived from:
    /// 1. `mtp.safetensors` file size from the HF tree API (best), or
    /// 2. Parameter count from `mtplx_runtime.json` quantization, or
    /// 3. Conservative bf16 fallback if no provenance data.
    ///
    /// `None` if no data is available.
    pub estimated_memory_bytes: Option<u64>,
    /// The quantization of the MTP sidecar (e.g., "bf16", "q4", "q6").
    /// Populated from `mtplx_runtime.json.mtp_sidecar` if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mtp_sidecar: Option<String>,
    /// The maximum MTP depth (e.g., 3).
    /// Populated from `mtplx_runtime.json.mtp_depth_max` if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mtp_depth_max: Option<i64>,
}

pub async fn resolve_speculative_model_preflight(
    repo_id: &str,
) -> Result<SpeculativeModelPreflight, String> {
    let token = hf_load_token();
    let url = format!("https://huggingface.co/api/models/{repo_id}");
    let mut req = HF_HTTP_CLIENT.get(&url);
    if let Some(ref tok) = token {
        req = req.bearer_auth(tok);
    }
    let resp = req
        .send()
        .await
        .map_err(|e| format!("HF models API request failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("HF returned HTTP {} for {repo_id}", resp.status()));
    }
    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse HF models API response: {e}"))?;

    let revision = body
        .get("sha")
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("HF did not return a commit sha for {repo_id}"))?
        .to_string();

    let has_custom_loader = body
        .get("siblings")
        .and_then(|v| v.as_array())
        .is_some_and(|files| {
            files.iter().any(|f| {
                let name = f
                    .get("rfilename")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .rsplit('/')
                    .next()
                    .unwrap_or("")
                    .to_lowercase();
                name == "main.py"
                    || (name.starts_with("modeling_") && name.ends_with(".py"))
                    || (name.starts_with("configuration_") && name.ends_with(".py"))
            })
        });

    let trust_remote_code_required = if has_custom_loader {
        true
    } else {
        fetch_repo_config_json_at(repo_id, &revision)
            .await
            .is_ok_and(|config| config_json_needs_trust_remote_code(&config))
    };

    // 3-tier estimate for companion VRAM:
    // 1. Tree API → mtp.safetensors file size (most accurate)
    // 2. mtplx_runtime.json → quantization + param estimate
    // 3. Conservative bf16 fallback

    let (estimated_memory_bytes, mtp_sidecar, mtp_depth_max) =
        fetch_mtp_companion_info(repo_id, &revision, token.as_deref()).await;

    Ok(SpeculativeModelPreflight {
        repo_id: repo_id.to_string(),
        revision,
        trust_remote_code_required,
        estimated_memory_bytes,
        mtp_sidecar,
        mtp_depth_max,
    })
}

/// Fetches `mtplx_runtime.json` from a HF repo. Returns parsed JSON if
/// available, `None` if the file does not exist or cannot be fetched.
async fn fetch_mtplx_runtime_json(
    repo_id: &str,
    revision: &str,
    token: Option<&str>,
) -> Option<serde_json::Value> {
    let url = format!("https://huggingface.co/{repo_id}/resolve/{revision}/mtplx_runtime.json");
    let mut req = HF_HTTP_CLIENT.get(&url);
    if let Some(tok) = token {
        req = req.bearer_auth(tok);
    }
    let resp = req.send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    resp.json::<serde_json::Value>().await.ok()
}

/// Introspects the MTP companion of a HF repo to produce an accurate VRAM
/// estimate. Makes two parallel requests:
///
/// 1. **Tree API** — fetch file list, find `mtp.safetensors` file size (works
///    for any repo with an MTP sidecar, MTPLX or not).
/// 2. **mtplx_runtime.json** — read `mtp_sidecar` quantization label and
///    `mtp_depth_max` (only available for MTPLX-provenance repos).
///
/// Falls back to a conservative bf16 estimate only if neither succeeds.
async fn fetch_mtp_companion_info(
    repo_id: &str,
    revision: &str,
    token: Option<&str>,
) -> (Option<u64>, Option<String>, Option<i64>) {
    // Run both requests in parallel
    let (size_result, runtime_result) = tokio::join!(
        fetch_mtp_safetensors_size(repo_id, revision, token),
        fetch_mtplx_runtime_json(repo_id, revision, token),
    );

    // Tree API gives the most accurate estimate
    if let Some(size) = size_result {
        if let Some(runtime) = runtime_result {
            let sidecar = runtime
                .get("mtp_sidecar")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let depth = runtime.get("mtp_depth_max").and_then(|v| v.as_i64());
            return (Some(size), sidecar, depth);
        }
        return (Some(size), None, None);
    }

    // No tree API data — try mtplx_runtime.json for quantization
    if let Some(runtime) = runtime_result {
        let sidecar = runtime
            .get("mtp_sidecar")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let depth = runtime.get("mtp_depth_max").and_then(|v| v.as_i64());
        // Without the tree API we can't estimate VRAM from quantization alone
        // (MTP sidecars are small — ~400M params at bf16 ≈ 849 MB).
        return (None, sidecar, depth);
    }

    // No data — return None
    (None, None, None)
}

/// Fetches the HF tree API and returns the file size of `mtp.safetensors`
/// if present. Also handles sharded variants like `mtp-00001-of-00002.safetensors`.
async fn fetch_mtp_safetensors_size(
    repo_id: &str,
    revision: &str,
    token: Option<&str>,
) -> Option<u64> {
    let url = format!("https://huggingface.co/api/models/{repo_id}/tree/{revision}");
    let mut req = HF_HTTP_CLIENT.get(&url);
    if let Some(tok) = token {
        req = req.bearer_auth(tok);
    }
    let resp = req.send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let files: Vec<serde_json::Value> = resp.json().await.ok()?;
    let mut total: u64 = 0;
    for file in &files {
        let path = file.get("path").and_then(|v| v.as_str())?;
        if path == "mtp.safetensors"
            || (path.starts_with("mtp-") && path.ends_with(".safetensors") && path.contains("of-"))
        {
            total += file.get("size").and_then(|v| v.as_u64()).unwrap_or(0);
        }
    }
    if total > 0 { Some(total) } else { None }
}

/// Mirrors the `main_class`/`auto_map` heuristic in
/// `inference::rapid_mlx::model_resolver::needs_trust_remote_code`, applied
/// to a remotely fetched `config.json` instead of a local model directory.
fn config_json_needs_trust_remote_code(config: &serde_json::Value) -> bool {
    if config.get("main_class").and_then(|v| v.as_str()).is_some() {
        return true;
    }
    if let Some(auto_map) = config.get("auto_map").and_then(|v| v.as_object()) {
        for class_value in auto_map.values() {
            if let Some(class_str) = class_value.as_str()
                && !class_str.starts_with("Auto")
                && !class_str.contains("transformers.models")
                && !class_str.contains("transformers_modules")
            {
                return true;
            }
        }
    }
    false
}

/// Fetch a repo's `config.json` at a specific revision as raw JSON, bounded
/// by the same size cap used for MLX config fetches. Absence or malformed
/// JSON is reported as an error rather than defaulted, so callers make an
/// explicit choice about how to treat "no config" (unlike the local-disk
/// resolver, a missing remote config is not proof of a data-only repo).
async fn fetch_repo_config_json_at(
    repo_id: &str,
    revision: &str,
) -> Result<serde_json::Value, String> {
    validate_hf_revision(revision)?;
    let url = hf_resolve_download_url_at(repo_id, "config.json", revision);
    let mut req = HF_HTTP_CLIENT.get(&url);
    if let Some(token) = hf_load_token() {
        req = req.bearer_auth(token);
    }
    let resp = req
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!(
            "HF returned HTTP {} for config.json",
            resp.status()
        ));
    }
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| format!("reading config.json: {e}"))?;
    if bytes.len() as u64 > crate::inference::rapid_mlx::mlx_meta::MAX_CONFIG_BYTES {
        return Err("config.json exceeds size limit".to_string());
    }
    serde_json::from_slice(&bytes).map_err(|e| format!("config.json is not valid JSON: {e}"))
}

/// List repo files; filters for GGUF if gguf_only=true.
#[allow(dead_code)]
pub fn hf_list_repo_files(repo_id: &str, gguf_only: bool) -> Result<Vec<HfFileInfo>> {
    let (owner, name) = repo_id
        .split_once('/')
        .context("repo_id must be in owner/name format")?;
    let client = hf_build_client(hf_load_token())?;
    let info = client
        .model(owner, name)
        .info()
        .send()
        .context("Failed to list repo files")?;

    let siblings = info
        .siblings
        .context("HF API did not return file listing (siblings)")?;
    Ok(siblings
        .iter()
        .filter(|s| !gguf_only || s.rfilename.to_ascii_lowercase().ends_with(".gguf"))
        .map(|s| HfFileInfo {
            r#type: "file".into(),
            path: s.rfilename.clone(),
            size: None,
        })
        .collect())
}

/// Get info for a single file in a repo.
#[allow(dead_code)]
pub fn hf_get_file_info(repo_id: &str, path: &str) -> Result<HfFileInfo> {
    let (owner, name) = repo_id
        .split_once('/')
        .context("repo_id must be in owner/name format")?;
    let client = hf_build_client(hf_load_token())?;
    let info = client
        .model(owner, name)
        .info()
        .send()
        .context("Failed to list repo files")?;

    let siblings = info
        .siblings
        .context("HF API did not return file listing (siblings)")?;
    siblings
        .iter()
        .find(|s| s.rfilename == path)
        .map(|s| HfFileInfo {
            r#type: "file".into(),
            path: s.rfilename.clone(),
            size: None,
        })
        .ok_or_else(|| anyhow::anyhow!("File not found: {path}"))
}

/// Range-fetch only the GGUF KV-metadata header of a HuggingFace-hosted file and parse it,
/// without downloading the multi-GB tensor data. This lets the pre-download estimator use the
/// model's *real* architecture (layer counts, hybrid attention interval, sliding window,
/// expert counts, …) instead of name-based guesses.
///
/// The KV header sits at the start of the file, so we fetch progressively larger prefixes
/// until the parser succeeds (the tokenizer arrays can push real headers to several MB).
pub async fn fetch_gguf_header_metadata(
    repo_id: &str,
    file_path: &str,
) -> Result<crate::llama::gguf_meta::GgufMetadata, String> {
    let url = hf_resolve_download_url(repo_id, file_path);
    if url.is_empty() {
        return Err(format!(
            "Could not resolve HF URL for {repo_id}/{file_path}"
        ));
    }

    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .map_err(|e| format!("http client: {e}"))?;
    let token = hf_load_token();

    let mut last_err = String::from("header larger than fetch cap");
    for &end in &[4 * 1024 * 1024u64, 16 * 1024 * 1024, 48 * 1024 * 1024] {
        let mut req = client
            .get(&url)
            .header("Range", format!("bytes=0-{}", end - 1));
        if let Some(ref tok) = token {
            req = req.bearer_auth(tok);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| format!("range request failed: {e}"))?;
        // Require a 206 so a server that ignores Range can't make us buffer the whole file.
        if resp.status() != reqwest::StatusCode::PARTIAL_CONTENT {
            return Err(format!(
                "HF did not honor range request (HTTP {})",
                resp.status()
            ));
        }
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| format!("reading header bytes: {e}"))?;
        match crate::llama::gguf_meta::read_gguf_metadata_from_bytes(&bytes) {
            Ok(md) => return Ok(md),
            Err(e) => last_err = e, // likely truncated header — try a larger prefix
        }
    }
    Err(format!("could not parse GGUF header: {last_err}"))
}

/// Fetch and parse an MLX model's `config.json` directly from a HuggingFace repo.
///
/// Unlike the GGUF header (which is range-fetched because the file can be many GB), MLX's
/// `config.json` is always small JSON, so this does a plain GET of the whole file.
pub async fn fetch_mlx_config(
    repo_id: &str,
    file_path: &str,
) -> Result<crate::inference::rapid_mlx::mlx_meta::MlxConfig, String> {
    fetch_mlx_config_revision_aware(repo_id, "main", file_path).await
}

/// Fetch and parse an MLX model's config.json from a HuggingFace repo with revision support.
///
/// Revision-aware (not always main), bounded depth/size/timeout to prevent abuse.
/// CRITICAL: Always fetches config.json regardless of hf_file_path — hf_file_path is the
/// model file (e.g. model.safetensors), not the config filename. This prevents the gap 3.7
/// defect where hf_file_path became an MLX config name.
pub async fn fetch_mlx_config_revision_aware(
    repo_id: &str,
    revision: &str,
    _hf_file_path: &str,
) -> Result<crate::inference::rapid_mlx::mlx_meta::MlxConfig, String> {
    validate_hf_revision(revision)?;
    // CRITICAL: Always use config.json for MLX config — never hf_file_path.
    // hf_file_path is the model weight file (e.g. "model.safetensors"), not the config.
    let config_path = "config.json";
    fetch_mlx_config_bytes_at(repo_id, revision, config_path).await
}

/// Fetch an MLX config at its selected revision and convert its inline
/// `text_config` geometry into the normalized profile used by Rapid estimates.
/// `hf_file_path` is intentionally not accepted: a weight filename can never
/// select the architecture config.
pub async fn fetch_mlx_model_profile_revision_aware(
    repo_id: &str,
    revision: &str,
) -> Result<crate::llama::model_memory_profile::ModelMemoryProfile, String> {
    let config = fetch_mlx_config_with_text_config(repo_id, revision).await?;
    let mut raw = serde_json::to_value(&config)
        .map_err(|error| format!("serializing MLX config for profile parsing: {error}"))?;
    if let Some(inner) = config.text_config_inner.as_deref() {
        raw["text_config"] = serde_json::to_value(inner)
            .map_err(|error| format!("serializing nested MLX text config: {error}"))?;
    }
    let bytes = serde_json::to_vec(&raw)
        .map_err(|error| format!("serializing MLX profile config: {error}"))?;
    let mut profile =
        crate::inference::rapid_mlx::mlx_meta::parse_mlx_config_bytes_to_profile(&bytes)?;
    profile.source_revision = Some(revision.to_string());
    Ok(profile)
}

/// Fetch config bytes from HF with bounds enforcement.
async fn fetch_mlx_config_bytes_at(
    repo_id: &str,
    revision: &str,
    file_path: &str,
) -> Result<crate::inference::rapid_mlx::mlx_meta::MlxConfig, String> {
    let url = hf_resolve_download_url_at(repo_id, file_path, revision);
    if url.is_empty() {
        return Err(format!(
            "Could not resolve HF URL for {repo_id}/{file_path}"
        ));
    }

    let max_bytes = crate::inference::rapid_mlx::mlx_meta::MAX_CONFIG_BYTES as usize;

    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| format!("http client: {e}"))?;
    let mut req = client.get(&url);
    if let Some(token) = hf_load_token() {
        req = req.bearer_auth(token);
    }
    let resp = req
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!(
            "HF returned HTTP {} for {file_path}",
            resp.status()
        ));
    }
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| format!("reading {file_path}: {e}"))?;
    if bytes.len() > max_bytes {
        return Err(format!(
            "{file_path} exceeds size limit ({max_bytes} bytes, got {} bytes)",
            bytes.len()
        ));
    }
    crate::inference::rapid_mlx::mlx_meta::parse_mlx_config(&bytes)
}

/// Fetch an MLX config with recursive text_config resolution.
///
/// For nested configs (Qwen3.6, Gemma4, etc.), if the config contains a text_config
/// referencing a separate file (e.g. a nested config path), this will fetch and merge it.
///
/// Bounded by:
/// - max_depth: prevents infinite recursion (default 3)
/// - timeout: overall timeout for all fetches (30 seconds)
/// - size limit: each file bounded by MAX_CONFIG_BYTES
///
/// This ensures reliable config fetching even for deeply nested or large MLX repos.
pub async fn fetch_mlx_config_with_text_config(
    repo_id: &str,
    revision: &str,
) -> Result<crate::inference::rapid_mlx::mlx_meta::MlxConfig, String> {
    const MAX_DEPTH: usize = 3;
    let mut root = fetch_mlx_config_bytes_at(repo_id, revision, "config.json").await?;
    let mut seen = std::collections::BTreeSet::from([String::from("config.json")]);
    let mut current = &mut root;

    for depth in 0..MAX_DEPTH {
        let Some(reference) = current.text_config_ref.clone() else {
            return Ok(root);
        };
        validate_mlx_config_reference(&reference)?;
        if !seen.insert(reference.clone()) {
            return Err(format!("MLX text_config reference cycle at {reference}"));
        }
        let inner = fetch_mlx_config_bytes_at(repo_id, revision, &reference)
            .await
            .map_err(|error| format!("Failed to fetch text_config at depth {depth}: {error}"))?;
        current.text_config_inner = Some(Box::new(inner));
        current = current
            .text_config_inner
            .as_deref_mut()
            .expect("nested config inserted above");
    }

    if current.text_config_ref.is_some() {
        return Err(format!(
            "Config recursion exceeded max depth {MAX_DEPTH} for {repo_id}"
        ));
    }
    Ok(root)
}

fn validate_mlx_config_reference(reference: &str) -> Result<(), String> {
    let path = Path::new(reference);
    if reference.is_empty()
        || reference.len() > 512
        // `Path::is_absolute` treats a leading slash differently on Windows
        // than on Unix. References are HF-relative paths on every platform,
        // so reject both slash forms explicitly before consulting components.
        || reference.starts_with('/')
        || reference.starts_with('\\')
        || path.is_absolute()
        || path
            .components()
            .any(|part| matches!(part, std::path::Component::ParentDir))
        || !reference.ends_with(".json")
    {
        return Err("MLX text_config reference must be a safe relative JSON path".into());
    }
    Ok(())
}

/// Stream-download a file from HF with optional resume.
/// Returns total bytes written.
#[allow(dead_code)]
pub async fn hf_download_file_stream(
    repo_id: &str,
    path: &str,
    token: Option<&str>,
    local_path: &Path,
    resume_from: u64,
) -> Result<u64> {
    let url = hf_resolve_download_url(repo_id, path);
    if url.is_empty() {
        anyhow::bail!("Failed to resolve HF URL for {path}");
    }

    let mut req = HF_HTTP_CLIENT.get(&url);
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }
    if resume_from > 0 {
        req = req.header("Range", format!("bytes={}-", resume_from));
    }

    let resp = req.send().await.context("Failed to start HF download")?;
    if !resp.status().is_success() && resp.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        anyhow::bail!(
            "HF download failed: HTTP {} for {}/{}",
            resp.status(),
            repo_id,
            path
        );
    }

    if let Some(parent) = local_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .context("Failed to create download dir")?;
    }

    // No content-length: read all at once
    if resp.content_length().is_none() {
        let bytes = resp.bytes().await.context("Failed to read response body")?;
        File::create(local_path)
            .await
            .context("Failed to create file")?
            .write_all(&bytes)
            .await?;
        return Ok(bytes.len() as u64);
    }

    // Streaming with proper open mode (truncate on fresh, append on resume)
    let mut file = if resume_from > 0 {
        File::options()
            .create(true)
            .write(true)
            .append(true)
            .open(local_path)
            .await
            .context("Failed to open file for resume")?
    } else {
        File::create(local_path)
            .await
            .context("Failed to create file")?
    };

    use futures_util::StreamExt;
    let mut stream = resp.bytes_stream();
    let mut written = resume_from;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Stream error from HF")?;
        file.write_all(&chunk).await?;
        written += chunk.len() as u64;
    }
    Ok(written)
}

// ── Search and browse ─────────────────────────────────────────────────────────

/// Model format filter for HF search (GGUF vs MLX).
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HfModelFormat {
    #[default]
    Gguf,
    Mlx,
    Both,
}

impl HfModelFormat {
    // No-op: filter applied inline in hf_search_single
}

/// Full search request parameters.
#[derive(Debug, Clone, Default)]
pub struct HfSearchParams {
    /// Keyword query. Can be empty when author is set.
    pub query: String,
    /// Filter to a specific author/org (e.g. "bartowski", "unsloth").
    pub author: Option<String>,
    pub sort: HfSort,
    pub limit: usize,
    /// Opaque cursor from the HF API `Link` response header for pagination.
    pub cursor: Option<String>,
    /// Model format filter (GGUF default for backward compatibility).
    pub format: HfModelFormat,
    /// Filter to show only quantized variants (excludes base models).
    pub quants_only: bool,
}

/// Parse the `cursor=` value out of a HF API `Link: <url>; rel="next"` header.
fn parse_cursor_from_link(link: &str) -> Option<String> {
    for part in link.split(',') {
        let part = part.trim();
        if !part.contains(r#"rel="next""#) {
            continue;
        }
        if let (Some(s), Some(e)) = (part.find('<'), part.find('>')) {
            let url_str = &part[s + 1..e];
            for param in url_str.split('&') {
                if let Some(val) = param.strip_prefix("cursor=") {
                    return Some(val.to_string());
                }
            }
        }
    }
    None
}

/// Search HuggingFace for GGUF models with keyword + optional author filter.
///
/// Returns (models, next_cursor) where next_cursor is Some when more pages exist.
pub async fn hf_search_models(
    params: &HfSearchParams,
) -> Result<(Vec<SimpleModelInfo>, Option<String>), String> {
    // Both format: do two separate searches and merge
    if matches!(params.format, HfModelFormat::Both) {
        return hf_search_both(params).await;
    }

    hf_search_single(params).await
}

/// Search for a single format (GGUF or MLX) — the core HF API call.
async fn hf_search_single(
    params: &HfSearchParams,
) -> Result<(Vec<SimpleModelInfo>, Option<String>), String> {
    let limit = params.limit.clamp(1, 100);
    let token = hf_load_token();

    let mut url = reqwest::Url::parse("https://huggingface.co/api/models")
        .map_err(|e| format!("Invalid HF API URL: {e}"))?;

    {
        let mut p = url.query_pairs_mut();
        if !params.query.is_empty() {
            p.append_pair("search", &params.query);
        }
        if let Some(ref author) = params.author {
            p.append_pair("author", author);
        }
        p.append_pair("limit", &limit.to_string());
        // Relevance yields no key: HF ranks by the search term only when `sort` is absent.
        if let Some(sort_key) = params.sort.as_api_str() {
            p.append_pair("sort", sort_key);
            p.append_pair("direction", "-1");
        }
        if let Some(ref cursor) = params.cursor {
            p.append_pair("cursor", cursor);
        }
        match params.format {
            HfModelFormat::Gguf => p.append_pair("apps", "llama.cpp"),
            // `apps=mlx-lm` only matches repos whose library metadata says so, which
            // silently drops recent MLX uploads tagged `transformers` or `any-to-any`.
            // The `mlx` tag is what the Hub itself uses for MLX repositories.
            HfModelFormat::Mlx => p.append_pair("filter", "mlx"),
            HfModelFormat::Both => unreachable!(), // handled in hf_search_both
        };
        if matches!(params.format, HfModelFormat::Mlx) {
            p.append_pair("expand[]", "safetensors");
        }
    }

    let mut req = HF_HTTP_CLIENT.get(url);
    if let Some(ref tok) = token {
        req = req.bearer_auth(tok);
    }

    let resp = req
        .send()
        .await
        .map_err(|e| format!("HF search request failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("HF search failed: HTTP {}", resp.status()));
    }

    let next_cursor = resp
        .headers()
        .get("link")
        .and_then(|v| v.to_str().ok())
        .and_then(parse_cursor_from_link);

    let items: Vec<serde_json::Value> = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse HF response: {e}"))?;
    let items = if params.quants_only {
        items
            .into_iter()
            .filter(has_quantized_base_model_tag)
            .collect()
    } else {
        items
    };

    let mut models: Vec<SimpleModelInfo> = items
        .into_iter()
        .filter_map(|item| parse_model_item(item, &params.format))
        .collect();

    // For MLX models where the safetensors expand didn't yield a size, fall back to tree API
    let token = hf_load_token();
    for model in models.iter_mut() {
        if model.model_size_bytes.is_none()
            && matches!(params.format, HfModelFormat::Mlx | HfModelFormat::Both)
            && let Ok(files) = fetch_file_sizes(&model.id, token.as_deref()).await
        {
            let total: u64 = files.values().sum();
            if total > 0 {
                model.model_size_bytes = Some(total);
            }
        }
    }

    Ok((models, next_cursor))
}

/// For Both format, do two separate searches (GGUF + MLX) and merge.
async fn hf_search_both(
    params: &HfSearchParams,
) -> Result<(Vec<SimpleModelInfo>, Option<String>), String> {
    let limit = params.limit.clamp(1, 100);
    let token = hf_load_token();

    let mut url = reqwest::Url::parse("https://huggingface.co/api/models")
        .map_err(|e| format!("Invalid HF API URL: {e}"))?;

    {
        let mut p = url.query_pairs_mut();
        if !params.query.is_empty() {
            p.append_pair("search", &params.query);
        }
        if let Some(ref author) = params.author {
            p.append_pair("author", author);
        }
        p.append_pair("limit", &limit.to_string());
        // Relevance yields no key: HF ranks by the search term only when `sort` is absent.
        if let Some(sort_key) = params.sort.as_api_str() {
            p.append_pair("sort", sort_key);
            p.append_pair("direction", "-1");
        }
        if let Some(ref cursor) = params.cursor {
            p.append_pair("cursor", cursor);
        }
        // No `apps` filter: the Hub cannot OR `llama.cpp` with `mlx`, and `apps=mlx-lm`
        // hides recent MLX repos. Results are narrowed by tag after the request.
        p.append_pair("expand[]", "safetensors");
    }

    let mut req = HF_HTTP_CLIENT.get(url);
    if let Some(ref tok) = token {
        req = req.bearer_auth(tok);
    }

    let resp = req
        .send()
        .await
        .map_err(|e| format!("HF search request failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("HF search failed: HTTP {}", resp.status()));
    }

    let next_cursor = resp
        .headers()
        .get("link")
        .and_then(|v| v.to_str().ok())
        .and_then(parse_cursor_from_link);

    let items: Vec<serde_json::Value> = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse HF response: {e}"))?;
    let items = if params.quants_only {
        items
            .into_iter()
            .filter(has_quantized_base_model_tag)
            .collect()
    } else {
        items
    };

    let mut models: Vec<SimpleModelInfo> = items
        .into_iter()
        .filter(|item| {
            item.get("tags")
                .and_then(|t| t.as_array())
                .is_some_and(|tags| {
                    tags.iter()
                        .filter_map(|t| t.as_str())
                        .any(|t| t.eq_ignore_ascii_case("mlx") || t.eq_ignore_ascii_case("gguf"))
                })
        })
        .filter_map(|item| parse_model_item(item, &HfModelFormat::Both))
        .collect();

    // For MLX models where the safetensors expand didn't yield a size, fall back to tree API
    let token = hf_load_token();
    for model in models.iter_mut() {
        if model.model_size_bytes.is_none()
            && model.format == "mlx"
            && let Ok(files) = fetch_file_sizes(&model.id, token.as_deref()).await
        {
            let total: u64 = files.values().sum();
            if total > 0 {
                model.model_size_bytes = Some(total);
            }
        }
    }

    Ok((models, next_cursor))
}

/// Browse all GGUF models from a specific HF author/org (convenience wrapper).
/// `base_model_relation` is not a working Hub models-API filter. Retain only
/// repositories that explicitly declare themselves as a quantization instead.
fn has_quantized_base_model_tag(item: &serde_json::Value) -> bool {
    item.get("tags")
        .and_then(|tags| tags.as_array())
        .is_some_and(|tags| {
            tags.iter().any(|tag| {
                tag.as_str()
                    .is_some_and(|tag| tag.starts_with("base_model:quantized:"))
            })
        })
}

/// Parse a single model JSON object from the HF API into SimpleModelInfo.
fn parse_model_item(item: serde_json::Value, format: &HfModelFormat) -> Option<SimpleModelInfo> {
    let id = item
        .get("id")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())?
        .to_string();

    let author = id.split('/').next().unwrap_or("").to_string();
    let quant_provider = QuantProvider::from_username(&author);

    let tags: Vec<String> = item
        .get("tags")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|t| t.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    let has_imatrix = tags.iter().any(|t| t == "imatrix")
        || id.to_ascii_lowercase().contains("imatrix")
        || id.to_ascii_lowercase().contains("-i1-")
        || matches!(quant_provider, QuantProvider::Mradermacher);

    let base_model = item
        .get("cardData")
        .and_then(|cd| cd.get("base_model"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    // Infer parameter count from repo name
    let param_b = infer_param_b_from_name(&id);

    let model_size_bytes = item
        .get("model_size_bytes")
        .and_then(|v| v.as_u64())
        .or_else(|| safetensors_total_bytes(&item));

    // Infer quant label from HF tags (preferred) or repo name
    let quant_label = infer_mlx_quant_label(&tags, &id);

    // HF reliably tags MLX repos "mlx" and GGUF repos "gguf" — use the real
    // per-item tag rather than blanket-stamping "both" for every result.
    let tag_has_mlx = tags.iter().any(|t| t.eq_ignore_ascii_case("mlx"));
    let tag_has_gguf = tags.iter().any(|t| t.eq_ignore_ascii_case("gguf"));

    Some(SimpleModelInfo {
        id,
        revision: item
            .get("sha")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        gated: item.get("gated").and_then(|v| v.as_bool()).unwrap_or(false),
        tags,
        downloads: item.get("downloads").and_then(|v| v.as_u64()).unwrap_or(0),
        likes: item.get("likes").and_then(|v| v.as_u64()).unwrap_or(0),
        created_at: item
            .get("createdAt")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        last_modified: item
            .get("lastModified")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        author,
        quant_provider,
        has_imatrix,
        param_b,
        base_model,
        format: match format {
            HfModelFormat::Mlx => "mlx".into(),
            HfModelFormat::Gguf => "gguf".into(),
            HfModelFormat::Both => match (tag_has_mlx, tag_has_gguf) {
                (true, false) => "mlx".into(),
                (false, true) => "gguf".into(),
                _ => "both".into(),
            },
        },
        model_size_bytes,
        quant_label,
    })
}

/// Infer parameter count in billions from a model name/repo id.
fn infer_param_b_from_name(name: &str) -> f64 {
    // Match all NB patterns, take the largest (total params, not active)
    let matches: Vec<f64> = name
        .split(|c: char| !c.is_alphanumeric() && c != '.')
        .filter_map(|token| {
            let lower = token.to_ascii_lowercase();
            if lower.ends_with('b') {
                lower[..lower.len() - 1].parse::<f64>().ok()
            } else {
                None
            }
        })
        .filter(|&v| (0.5..=2000.0).contains(&v))
        .collect();
    matches.into_iter().fold(0.0_f64, f64::max)
}

/// Infer MLX quant label from HF tags only (not repo names — too many variations).
/// Repo names are unreliable: crowqwen3.5-4b-agent-heretic-mlx-oq, Bielik-PL-11B-v3.0-Instruct-heretic-mlx-4bit, etc.
/// Returns empty string if no matching tag found (common for BF16 models without tags).
fn infer_mlx_quant_label(tags: &[String], _id: &str) -> String {
    let tags_lower = tags.iter().map(|t| t.to_ascii_lowercase());
    for tag in tags_lower {
        if tag == "4-bit" {
            return "4-bit".into();
        }
        if tag == "8-bit" {
            return "8-bit".into();
        }
        if tag == "3-bit" {
            return "3-bit".into();
        }
        if tag == "2-bit" {
            return "2-bit".into();
        }
        if tag == "5-bit" {
            return "5-bit".into();
        }
        if tag == "6-bit" {
            return "6-bit".into();
        }
    }
    String::new()
}

// ── GGUF file listing with real sizes ────────────────────────────────────────

/// List GGUF files for a repo, fetching real file sizes from the HF tree API.
///
/// Falls back to 0-size entries if the tree API is unavailable (e.g. private repos
/// without a token, or rate limiting).
pub async fn hf_list_gguf_files(repo_id: &str) -> Result<Vec<HfGgufFile>, String> {
    let token = hf_load_token();
    let mut result = list_repo_gguf_files(repo_id, token.clone()).await?;

    if !result.iter().any(|file| file.is_mmproj)
        && let Some(companion_repo) = find_mmproj_companion_repo(repo_id, token.as_deref()).await
        && let Ok(companion_files) = list_repo_gguf_files(&companion_repo, token).await
    {
        result.extend(companion_files.into_iter().filter(|file| file.is_mmproj));
        sort_gguf_files(&mut result);
    }

    Ok(result)
}

/// List MLX model files for a repo.
/// MLX models are directories containing .safetensors files; each directory is treated as a "model".
pub async fn hf_list_mlx_files(repo_id: &str) -> Result<Vec<serde_json::Value>, String> {
    let token = hf_load_token();
    let sizes = fetch_file_sizes(repo_id, token.as_deref())
        .await
        .unwrap_or_default();

    let (owner, name) = repo_id
        .split_once('/')
        .ok_or_else(|| format!("Invalid repo_id format: {repo_id}"))?;
    let client = hf_build_client(token).map_err(|e| format!("Failed to build HF client: {e}"))?;
    let info = client
        .model(owner, name)
        .info()
        .send()
        .map_err(|e| format!("Failed to list repo files: {e}"))?;

    let siblings = info
        .siblings
        .ok_or_else(|| format!("HF API did not return file listing for {repo_id}"))?;

    // MLX models: list safetensors files or directories containing them
    let result: Vec<serde_json::Value> = siblings
        .iter()
        .map(|s| {
            let path = s.rfilename.as_str();
            let size = sizes.get(path).copied().unwrap_or(0);
            let lower = path.to_ascii_lowercase();
            let is_safetensors =
                lower.ends_with(".safetensors") || lower.ends_with(".safetensors.index.json");
            let is_config = lower.ends_with("config.json") || lower.ends_with("config.yaml");
            if is_safetensors || is_config {
                serde_json::json!({
                    "name": path,
                    "path": path,
                    "size": size,
                    "repo_id": repo_id,
                })
            } else {
                serde_json::json!(null)
            }
        })
        .filter_map(|v| v.as_null().map_or(Some(v), |_| None))
        .collect();

    Ok(result)
}

async fn list_repo_gguf_files(
    repo_id: &str,
    token: Option<String>,
) -> Result<Vec<HfGgufFile>, String> {
    let sizes = fetch_file_sizes(repo_id, token.as_deref())
        .await
        .unwrap_or_default();

    let (owner, name) = repo_id
        .split_once('/')
        .ok_or_else(|| format!("Invalid repo_id format: {repo_id}"))?;
    let client = hf_build_client(token).map_err(|e| format!("Failed to build HF client: {e}"))?;
    let info = client
        .model(owner, name)
        .info()
        .send()
        .map_err(|e| format!("Failed to list repo files: {e}"))?;

    let siblings = info
        .siblings
        .ok_or_else(|| format!("HF API did not return file listing (siblings) for {repo_id}"))?;

    // Infer provider from repo owner
    let repo_owner = repo_id.split('/').next().unwrap_or("");
    let _provider = QuantProvider::from_username(repo_owner);

    let family = {
        let from_repo = infer_family_from_name(repo_id);
        if from_repo.is_empty() {
            siblings
                .iter()
                .map(|s| infer_family_from_name(&s.rfilename))
                .find(|candidate| !candidate.is_empty())
                .unwrap_or_default()
        } else {
            from_repo
        }
    };
    let mmproj_preference = mmproj_preference_for_family(&family);
    let mut result: Vec<HfGgufFile> = siblings
        .iter()
        .map(|s| s.rfilename.as_str())
        .filter(|name| name.to_ascii_lowercase().ends_with(".gguf"))
        .map(|name| {
            let quant_type = detect_quant_type(name);
            let is_imatrix = matches!(quant_type, QuantFileType::Imatrix);
            let name_lower = name.to_ascii_lowercase();
            let is_mmproj = name_lower.contains("mmproj") || name_lower.contains("projector");
            let label = infer_quant_label(name);
            let is_recommended_mmproj =
                is_mmproj && mmproj_preference.is_some_and(|preference| preference.label == label);
            let is_draft_assistant =
                is_draft_assistant_hf(&name_lower, sizes.get(name).copied().unwrap_or(0));
            HfGgufFile {
                repo_id: repo_id.to_string(),
                path: name.to_string(),
                size: sizes.get(name).copied().unwrap_or(0),
                label,
                quant_type,
                is_imatrix,
                is_mmproj,
                is_recommended_mmproj,
                mmproj_recommendation: if is_recommended_mmproj {
                    mmproj_preference
                        .map(|preference| preference.reason.to_string())
                        .unwrap_or_default()
                } else {
                    String::new()
                },
                is_draft_assistant,
            }
        })
        .collect();

    sort_gguf_files(&mut result);
    Ok(result)
}

fn sort_gguf_files(result: &mut [HfGgufFile]) {
    // Group order: main models (0) → assistant/draft files (1) → mmproj (2).
    // Within each group, apply the appropriate precision/quality ranking.
    result.sort_by(|a, b| {
        let a_group: u8 = if a.is_mmproj {
            2
        } else if a.is_draft_assistant {
            1
        } else {
            0
        };
        let b_group: u8 = if b.is_mmproj {
            2
        } else if b.is_draft_assistant {
            1
        } else {
            0
        };
        a_group
            .cmp(&b_group)
            .then_with(|| b.is_recommended_mmproj.cmp(&a.is_recommended_mmproj))
            .then_with(|| {
                let a_rank = if a.is_mmproj {
                    sort_rank_mmproj_label(&a.label)
                } else {
                    sort_rank_quant_label(&a.label)
                };
                let b_rank = if b.is_mmproj {
                    sort_rank_mmproj_label(&b.label)
                } else {
                    sort_rank_quant_label(&b.label)
                };
                a_rank.cmp(&b_rank)
            })
            .then_with(|| b.size.cmp(&a.size))
    });
}

/// Conservative heuristic for HF file classification as MTP assistant / draft model.
fn is_draft_assistant_hf(name_lower: &str, size_bytes: u64) -> bool {
    // Unambiguous MTP keywords: safe to match even when size is unknown (0).
    // Includes Unsloth's `-MTP.gguf` naming convention (e.g. "gemma-4-31B-it-Q8_0-MTP.gguf")
    // and path segments like `/MTP/` in Unsloth repo trees.
    let is_unambiguous = name_lower.contains("mtp-draft")
        || name_lower.contains("mtp_small")
        || name_lower.contains("mtp-heads")
        || name_lower.starts_with("mtp-")
        || name_lower.ends_with("-mtp.gguf")
        || name_lower.ends_with("-mtp")
        || name_lower.contains("/mtp/");

    if is_unambiguous {
        // Still reject if we know the file is a large main model.
        return size_bytes == 0 || size_bytes <= 3_000_000_000;
    }

    // Broad keywords require a confirmed, non-zero size to avoid mis-tagging
    // instruct-tuned main models (e.g. "Llama-3-8B-assistant-Q8_0.gguf") when
    // the HF tree API fails and size falls back to 0.
    let is_broad = name_lower.contains("assistant") || name_lower.contains("draft-model");
    is_broad && size_bytes > 0 && size_bytes <= 3_000_000_000
}

async fn find_mmproj_companion_repo(repo_id: &str, token: Option<&str>) -> Option<String> {
    let readme = fetch_repo_readme(repo_id, token).await.unwrap_or_default();
    for candidate in extract_static_quant_repos(&readme, repo_id) {
        if repo_contains_mmproj(&candidate, token).await {
            return Some(candidate);
        }
    }

    // mradermacher pairs weighted `-i1-GGUF` repositories with static
    // `-GGUF` repositories. Verify the target before using this fallback.
    if let Some(candidate) = mradermacher_static_repo(repo_id)
        && repo_contains_mmproj(&candidate, token).await
    {
        return Some(candidate);
    }

    // Variant repos occasionally advertise a static sibling that does not
    // actually contain the shared projector. For Qwen 3.5 variants, fall back
    // to canonical same-architecture repos and verify each one.
    for candidate in qwen35_arch_companion_repos(repo_id) {
        if repo_contains_mmproj(&candidate, token).await {
            return Some(candidate);
        }
    }

    None
}

async fn fetch_repo_readme(repo_id: &str, token: Option<&str>) -> Result<String> {
    let url = format!("https://huggingface.co/{repo_id}/raw/main/README.md");
    let mut req = HF_HTTP_CLIENT.get(url);
    if let Some(token) = token {
        req = req.bearer_auth(token);
    }
    let resp = req.send().await.context("Failed to fetch HF model card")?;
    if !resp.status().is_success() {
        anyhow::bail!("HF model card returned {}", resp.status());
    }
    resp.text().await.context("Failed to read HF model card")
}

fn extract_static_quant_repos(readme: &str, current_repo: &str) -> Vec<String> {
    let mut repos = Vec::new();

    for line in readme.lines() {
        let lower = line.to_ascii_lowercase();
        if !lower.contains("static") || (!lower.contains("quant") && !lower.contains("mmproj")) {
            continue;
        }

        for suffix in line.split("huggingface.co/").skip(1) {
            let candidate: String = suffix
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '/'))
                .collect();
            let candidate = candidate.trim_end_matches('/');
            if candidate != current_repo
                && is_valid_repo_id(candidate)
                && !repos.iter().any(|repo| repo == candidate)
            {
                repos.push(candidate.to_string());
            }
        }
    }

    repos
}

fn is_valid_repo_id(repo_id: &str) -> bool {
    let mut parts = repo_id.split('/');
    matches!(
        (parts.next(), parts.next(), parts.next()),
        (Some(owner), Some(repo), None) if !owner.is_empty() && !repo.is_empty()
    )
}

fn mradermacher_static_repo(repo_id: &str) -> Option<String> {
    let (owner, name) = repo_id.split_once('/')?;
    if !owner.eq_ignore_ascii_case("mradermacher") {
        return None;
    }
    let static_name = name.strip_suffix("-i1-GGUF")?;
    Some(format!("{owner}/{static_name}-GGUF"))
}

fn qwen35_arch_companion_repos(repo_id: &str) -> Vec<String> {
    let Some((owner, name)) = repo_id.split_once('/') else {
        return Vec::new();
    };
    if !owner.eq_ignore_ascii_case("mradermacher") {
        return Vec::new();
    }

    let Some(name) = name
        .strip_suffix("-i1-GGUF")
        .or_else(|| name.strip_suffix("-GGUF"))
    else {
        return Vec::new();
    };
    let tokens: Vec<&str> = name.split('-').collect();
    if !tokens
        .first()
        .is_some_and(|token| token.eq_ignore_ascii_case("Qwen3.5"))
    {
        return Vec::new();
    }

    let Some(total_index) = tokens.iter().position(|token| {
        token
            .strip_suffix('B')
            .or_else(|| token.strip_suffix('b'))
            .is_some_and(|value| value.parse::<u32>().is_ok())
    }) else {
        return Vec::new();
    };
    let mut end = total_index + 1;
    if tokens.get(end).is_some_and(|token| {
        token
            .strip_prefix('A')
            .or_else(|| token.strip_prefix('a'))
            .and_then(|value| value.strip_suffix('B').or_else(|| value.strip_suffix('b')))
            .is_some_and(|value| value.parse::<u32>().is_ok())
    }) {
        end += 1;
    }

    let architecture = tokens[..end].join("-");
    vec![
        format!("mradermacher/{architecture}-GGUF"),
        format!("unsloth/{architecture}-GGUF"),
    ]
}

async fn repo_contains_mmproj(repo_id: &str, token: Option<&str>) -> bool {
    fetch_file_sizes(repo_id, token).await.is_ok_and(|files| {
        files.keys().any(|path| {
            let lower = path.to_ascii_lowercase();
            lower.contains("mmproj") || lower.contains("projector")
        })
    })
}

#[derive(Clone, Copy)]
struct MmprojPreference {
    label: &'static str,
    reason: &'static str,
}

fn mmproj_preference_for_family(family: &str) -> Option<MmprojPreference> {
    match family {
        "qwen3.5" => Some(MmprojPreference {
            label: "F16",
            reason: "F16 is the documented llama.cpp projector default for Qwen 3.5",
        }),
        "qwen3.6" => Some(MmprojPreference {
            label: "F16",
            reason: "F16 is the practical default; upstream publishes F16 and BF16 without an optimization claim",
        }),
        "gemma4" => Some(MmprojPreference {
            label: "F16",
            reason: "F16 is the documented llama.cpp projector default for Gemma 4",
        }),
        _ => None,
    }
}

fn sort_rank_mmproj_label(label: &str) -> u8 {
    match label.to_ascii_uppercase().as_str() {
        "F16" => 0,
        "BF16" => 1,
        "Q8_0" => 2,
        "F32" => 3,
        _ => 4,
    }
}

/// Fetch file sizes from the HF tree API.
/// Returns a map of filename → size in bytes.
async fn fetch_file_sizes(
    repo_id: &str,
    token: Option<&str>,
) -> Result<std::collections::HashMap<String, u64>> {
    if let Some(cached) = HF_SIZE_CACHE.lock().unwrap().get(repo_id) {
        return Ok(cached.clone());
    }

    let url = format!("https://huggingface.co/api/models/{repo_id}/tree/main");
    let mut req = HF_HTTP_CLIENT.get(&url);
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }

    let resp = req.send().await.context("Failed to fetch HF tree")?;
    if !resp.status().is_success() {
        anyhow::bail!("HF tree API returned {}", resp.status());
    }

    let items: Vec<serde_json::Value> = resp.json().await.context("Failed to parse HF tree")?;
    let mut map = std::collections::HashMap::new();

    for item in items {
        let path = item
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let lower = path.to_ascii_lowercase();
        if !lower.ends_with(".gguf")
            && !lower.ends_with(".safetensors")
            && !lower.ends_with(".safetensors.index.json")
        {
            continue;
        }

        // LFS files: size is in "lfs.size"; non-LFS: in "size"
        let size = item
            .get("lfs")
            .and_then(|lfs| lfs.get("size"))
            .and_then(|v| v.as_u64())
            .or_else(|| item.get("size").and_then(|v| v.as_u64()))
            .unwrap_or(0);

        if size > 0 {
            map.insert(path, size);
        }
    }

    HF_SIZE_CACHE
        .lock()
        .unwrap()
        .insert(repo_id.to_string(), map.clone());

    Ok(map)
}

/// Sum total weight size for an MLX model repo from the HF tree API.
///
/// Used by the VRAM estimator when it receives an HF-repo-style alias as
/// `model_path` (e.g. "mlx-community/Qwen3-30B-A3B-4bit") and no
/// `model_size_bytes` was supplied. The huggingface-rs client's
/// list/get-file-info helpers do not expose sizes, so this goes directly
/// to the raw tree endpoint which does include LFS sizes.
pub async fn resolve_mlx_repo_size_bytes(repo_id: &str) -> Result<Option<u64>> {
    let url = format!("https://huggingface.co/api/models/{repo_id}/tree/main");
    let mut req = HF_HTTP_CLIENT.get(&url);
    if let Some(t) = hf_load_token() {
        req = req.bearer_auth(t);
    }

    let resp = match req.send().await {
        Ok(r) => r,
        Err(_) => return Ok(None),
    };
    if !resp.status().is_success() {
        return Ok(None);
    }

    let items: Vec<serde_json::Value> = match resp.json().await {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };

    let mut total: u64 = 0;
    for item in items {
        let path = item.get("path").and_then(|v| v.as_str()).unwrap_or("");
        if !path.ends_with(".safetensors") {
            continue;
        }
        let size = item
            .get("lfs")
            .and_then(|lfs| lfs.get("size"))
            .and_then(|v| v.as_u64())
            .or_else(|| item.get("size").and_then(|v| v.as_u64()))
            .unwrap_or(0);
        if size > 0 {
            total = total.saturating_add(size);
        }
    }

    if total > 0 { Ok(Some(total)) } else { Ok(None) }
}

/// Sort rank for quant labels (lower = higher quality / bigger file = shown first).
fn sort_rank_quant_label(label: &str) -> u8 {
    let lower = label.to_ascii_lowercase();
    if lower.contains("f32") {
        return 0;
    }
    if lower.contains("f16") || lower.contains("bf16") {
        return 1;
    }
    if lower.contains("q8") {
        return 2;
    }
    if lower.contains("q6") {
        return 3;
    }
    if lower.contains("q5_k_m") || lower.contains("q5km") {
        return 4;
    }
    if lower.contains("q5") {
        return 5;
    }
    if lower.contains("q4_k_m") || lower.contains("q4km") {
        return 6;
    }
    if lower.contains("iq4") {
        return 7;
    }
    if lower.contains("q4") {
        return 8;
    }
    if lower.contains("q3") {
        return 10;
    }
    if lower.contains("iq3") {
        return 11;
    }
    if lower.contains("q2") {
        return 14;
    }
    if lower.contains("iq2") {
        return 15;
    }
    if lower.contains("iq1") || lower.contains("q1") {
        return 18;
    }
    19
}

/// Infer a human-readable quant label from a GGUF filename.
#[allow(dead_code)]
pub fn infer_quant_label(filename: &str) -> String {
    let lower = filename.to_ascii_lowercase();

    // IQ variants first (before plain Q matches)
    if lower.contains("iq4_xs") || lower.contains("iq4xs") {
        return "IQ4_XS".into();
    }
    if lower.contains("iq4_nl") || lower.contains("iq4nl") {
        return "IQ4_NL".into();
    }
    if lower.contains("iq3_xxs") {
        return "IQ3_XXS".into();
    }
    if lower.contains("iq3_xs") {
        return "IQ3_XS".into();
    }
    if lower.contains("iq3_s") {
        return "IQ3_S".into();
    }
    if lower.contains("iq3_m") {
        return "IQ3_M".into();
    }
    if lower.contains("iq2_xxs") {
        return "IQ2_XXS".into();
    }
    if lower.contains("iq2_xs") {
        return "IQ2_XS".into();
    }
    if lower.contains("iq2_s") {
        return "IQ2_S".into();
    }
    if lower.contains("iq2_m") {
        return "IQ2_M".into();
    }
    if lower.contains("iq1_m") {
        return "IQ1_M".into();
    }
    if lower.contains("iq1_s") {
        return "IQ1_S".into();
    }

    // Standard quants
    if lower.contains("q8_0") {
        return "Q8_0".into();
    }
    if lower.contains("q6_k") {
        return "Q6_K".into();
    }
    if lower.contains("q5_k_xl") {
        return "Q5_K_XL".into();
    } // Unsloth variant
    if lower.contains("q5_k_m") {
        return "Q5_K_M".into();
    }
    if lower.contains("q5_k_s") {
        return "Q5_K_S".into();
    }
    if lower.contains("q5_0") {
        return "Q5_0".into();
    }
    if lower.contains("q4_k_xl") {
        return "Q4_K_XL".into();
    } // Unsloth variant
    if lower.contains("q4_k_m") {
        return "Q4_K_M".into();
    }
    if lower.contains("q4_k_s") {
        return "Q4_K_S".into();
    }
    if lower.contains("q4_0") {
        return "Q4_0".into();
    }
    if lower.contains("q3_k_l") {
        return "Q3_K_L".into();
    }
    if lower.contains("q3_k_m") {
        return "Q3_K_M".into();
    }
    if lower.contains("q3_k_s") {
        return "Q3_K_S".into();
    }
    if lower.contains("q2_k") {
        return "Q2_K".into();
    }
    if lower.contains("bf16") {
        return "BF16".into();
    }
    if lower.contains("f16") {
        return "F16".into();
    }
    if lower.contains("f32") {
        return "F32".into();
    }

    // Community mixed-precision schemes (e.g. this repo's own MTP/APEX quants) —
    // these aren't Unsloth's UD scheme, so keep the uploader's own naming rather
    // than mislabeling them "UD (custom)". Uploaders append arbitrary suffixes to
    // these (e.g. "APEX-MTP-I-Quality", "APEX-MTP-Balanced", "APEX-Compact") that
    // we can't enumerate in advance, so capture from the first "mtp"/"apex"
    // token through the end of the filename verbatim rather than hardcoding a
    // fixed piece list — that keeps distinct variants distinguishable in the table.
    if lower.contains("apex") || lower.contains("mtp") {
        let stem_end = filename
            .rfind(".gguf")
            .or_else(|| filename.rfind(".GGUF"))
            .unwrap_or(filename.len());
        let start = lower
            .find("apex")
            .into_iter()
            .chain(lower.find("mtp"))
            .min();
        if let Some(start) = start {
            return format!("{} (custom)", &filename[start..stem_end]);
        }
    }

    // Last resort: pull out anything that looks like a quant token (e.g. a
    // non-standard K-quant variant like "Q8_K_P") rather than collapsing it
    // to "Unknown" — an unrecognized-but-real name is more useful than none.
    if let Some(token) = extract_quant_like_token(filename) {
        return token;
    }

    "Unknown".into()
}

/// Scans a filename for a `Q`/`IQ`-prefixed token (e.g. "Q8_K_P", "IQ3_XYZ")
/// that didn't match any of the known presets above, and returns it verbatim
/// (uppercased) instead of discarding it as "Unknown".
fn extract_quant_like_token(filename: &str) -> Option<String> {
    let stem = filename
        .rsplit('/')
        .next()
        .unwrap_or(filename)
        .trim_end_matches(".gguf")
        .trim_end_matches(".GGUF");
    for segment in stem.split(['-', '_', '.']) {
        let seg_lower = segment.to_ascii_lowercase();
        let is_q =
            seg_lower.starts_with('q') && seg_lower[1..].starts_with(|c: char| c.is_ascii_digit());
        let is_iq =
            seg_lower.starts_with("iq") && seg_lower[2..].starts_with(|c: char| c.is_ascii_digit());
        if is_q || is_iq {
            // Reattach any immediately-following underscore-joined suffix segments
            // (e.g. "K", "P" in "Q8_K_P") that got split apart above.
            let start = stem.to_ascii_lowercase().find(&seg_lower)?;
            let rest = &stem[start..];
            let end = rest.find(['-', '.']).unwrap_or(rest.len());
            return Some(rest[..end].to_ascii_uppercase());
        }
    }
    None
}

// ── Token management ──────────────────────────────────────────────────────────

/// Load HF token: 1) HUGGING_FACE_HUB_TOKEN env var  2) the active application
/// root's `hf-token` file. The legacy path remains readable while the explicit
/// root migration is pending.
/// If the file is encrypted, decrypt_value handles it; otherwise treated as plaintext.
pub fn hf_load_token() -> Option<String> {
    if let Ok(v) = std::env::var("HUGGING_FACE_HUB_TOKEN")
        && !v.trim().is_empty()
    {
        return Some(v.trim().to_string());
    }
    let path = crate::paths::AppPaths::default_active_root().join("hf-token");
    std::fs::read_to_string(&path)
        .ok()
        .map(|s| crate::config::decrypt_value(s.trim()))
        .filter(|s| !s.is_empty())
}

/// Save HF token to the active application root (encrypted if key available).
pub fn hf_save_token(token: &str) -> Result<()> {
    let dir = crate::paths::AppPaths::default_active_root();
    std::fs::create_dir_all(&dir).context("Failed to create config dir")?;
    let stored = crate::config::encrypt_value(token.trim());
    std::fs::write(dir.join("hf-token"), &stored).context("Failed to write HF token")?;
    Ok(())
}

/// Mask a token for safe logging: first4****last4.
#[allow(dead_code)]
pub fn mask_token(token: &str) -> String {
    let t = token.trim();
    if t.len() <= 8 {
        return "****".to_string();
    }
    format!("{}****{}", &t[..4], &t[t.len() - 4..])
}

// ── Start a managed download ──────────────────────────────────────────────────

/// Start a download via the model_download manager.
/// `save_as` overrides the local filename (e.g. to rename a companion mmproj).
pub fn hf_start_download(
    repo_id: &str,
    file_path: &str,
    save_as: Option<&str>,
    target_path: &Path,
    _resume: bool,
) -> std::result::Result<String, String> {
    crate::model_download::start_download(repo_id, file_path, save_as, target_path, hf_load_token())
        .map_err(|e| e.to_string())
}

// ── Resolve HF origin from a local GGUF filename ─────────────────────────────

/// Candidate returned by `hf_resolve_origin`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HfResolveCandidate {
    /// HF repo ID (e.g. "bartowski/Llama-3.3-70B-Instruct-GGUF")
    pub repo_id: String,
    /// Confidence score 0.0–1.0
    pub confidence: f64,
    /// Brief reason for the score (for UI tooltips)
    pub reason: String,
    /// First few tags from the HF model card (empty if fetch failed)
    pub preview_tags: Vec<String>,
    /// Model card URL (always populated when repo_id is known)
    pub card_url: String,
    /// Detected model family slug (e.g. "qwen3.6", "llama3.3", "gemma4")
    pub family: String,
}

/// Result of resolving a local model's HF origin.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HfResolveResult {
    /// Whether the top candidate is confident enough to auto-attach
    pub confident: bool,
    /// Ranked candidates
    pub candidates: Vec<HfResolveCandidate>,
    /// The model stem used for resolution (for transparency)
    pub model_stem: String,
    /// Errors encountered (empty on full success)
    pub errors: Vec<String>,
}

/// Extract the model stem from a GGUF filename, stripping quant suffix,
/// Unsloth-distilled markers ("-UD"), and version/variant suffixes.
/// E.g. "gemma-4-31B-it-qat-UD-Q4_K_XL.gguf" → "gemma-4-31B-it-qat"
fn resolve_model_stem(filename: &str) -> String {
    let mut stem = filename
        .strip_suffix(".gguf")
        .unwrap_or(filename)
        .to_string();

    // Strip quant suffix: handle two-token patterns like "-I-Quality", "-Q8-MTP"
    // where a single-char quant indicator (I/Q/B/F) precedes a descriptor token.
    // Also handles single-token suffixes like "-Q4_K_M".
    let parts: Vec<&str> = stem.split('-').collect();
    if parts.len() >= 3 {
        let second_last = parts[parts.len() - 2];
        let last = parts[parts.len() - 1];
        let is_quant_indicator = matches!(second_last.chars().next(), Some('I' | 'Q' | 'B' | 'F'));
        if is_quant_indicator
            && second_last.len() == 1
            && matches!(
                last.chars().next(),
                Some('Q' | 'I' | 'B' | 'U' | 'M' | 'F' | 'S')
            )
            && last.len() > 2
        {
            // Strip the two-token quant suffix (e.g. "-I-Quality", "-Q8-MTP")
            let cutoff = stem.len() - second_last.len() - 1 - last.len();
            stem = stem[..cutoff].to_string();
        }
    }
    if stem.rfind('-').is_some_and(|pos| {
        let suffix = &stem[pos + 1..];
        matches!(suffix.chars().next(), Some('Q' | 'I' | 'B' | 'U')) && suffix.len() > 2
    }) {
        stem = stem[..stem.rfind('-').unwrap()].to_string();
    }

    // Strip Unsloth-distilled marker "-UD" or "-ud" if present
    if let Some(pos) = stem.rfind("-UD") {
        let after = &stem[pos + 3..];
        if after.is_empty() || after.starts_with('-') {
            stem = stem[..pos].to_string();
        }
    } else if let Some(pos) = stem.rfind("-ud") {
        let after = &stem[pos + 3..];
        if after.is_empty() || after.starts_with('-') {
            stem = stem[..pos].to_string();
        }
    }

    // Strip trailing -v1, -v2, -v1.1, -MTP, etc.
    stem = stem.strip_suffix("-MTP").unwrap_or(&stem).to_string();
    if let Some(pos) = stem.rfind("-v") {
        let after = &stem[pos + 2..];
        if after.is_empty() || after.chars().next().is_some_and(|c| c.is_ascii_digit()) {
            stem = stem[..pos].to_string();
        }
    }

    stem
}

/// Attempt to resolve the HF origin of a local GGUF file from its filename.
/// Searches HF for matching repos, scores them by filename/filename/param-match,
/// and returns ranked candidates.  Sets `confident=true` only when the top
/// candidate's score is >= 0.8.
pub async fn hf_resolve_origin(filename: &str, size_bytes: u64) -> Result<HfResolveResult, String> {
    let stem = resolve_model_stem(filename);
    let mut errors = Vec::new();

    // Build queries with stem variants. For models whose names include "-MTP"
    // (e.g. Carnice-Qwen3.6-MoE-35B-A3B-APEX-MTP-GGUF), the stem has "-MTP"
    // stripped, so we also try a query that re-adds it.
    let stem_with_mtp = format!("{stem}-MTP");
    let queries: Vec<String> = [
        stem.clone(),
        format!("{stem} GGUF"),
        stem.replace("Instruct", ""), // broader variant
        stem_with_mtp.clone(),
        format!("{stem_with_mtp} GGUF"),
    ]
    .iter()
    .filter(|q| q.len() >= 4)
    .cloned()
    .collect::<std::collections::BTreeSet<_>>()
    .into_iter()
    .collect();

    // Search HF for matching repos
    let mut candidates: Vec<(String, u64, Vec<String>)> = Vec::new();
    let _token = hf_load_token();

    for query in &queries {
        if candidates.len() >= 40 {
            break; // enough candidates
        }
        let params = HfSearchParams {
            query: query.clone(),
            author: None,
            sort: HfSort::Downloads,
            limit: 15,
            cursor: None,
            format: HfModelFormat::Gguf,
            quants_only: false,
        };
        let result = hf_search_models(&params).await;
        match result {
            Ok((items, _)) => {
                for item in items {
                    if !item.id.contains("GGUF")
                        && !item.tags.iter().any(|t| t.eq_ignore_ascii_case("gguf"))
                    {
                        continue;
                    }
                    if !candidates.iter().any(|(id, _, _)| id == &item.id) {
                        candidates.push((item.id.clone(), item.downloads, item.tags));
                    }
                }
            }
            Err(e) => errors.push(e),
        }
    }

    // If no candidates at all, return early
    if candidates.is_empty() {
        return Ok(HfResolveResult {
            confident: false,
            candidates: Vec::new(),
            model_stem: stem.clone(),
            errors,
        });
    }

    // Score and rank candidates, validating that each candidate repo actually
    // hosts a GGUF file matching the local file (name + size).
    let mut scored: Vec<HfResolveCandidate> = Vec::new();
    let token_for_file_check = _token.as_deref();

    for (repo_id, downloads, raw_tags) in &candidates {
        // Fetch repo files to see if the file exists there
        let files = match fetch_file_sizes(repo_id, token_for_file_check).await {
            Ok(f) => f,
            Err(_) => {
                // If we cannot list repo files, we can’t confirm existence;
                // treat it as unknown but still score weakly.
                let score = score_resolve_candidate(
                    repo_id, &stem, filename, size_bytes, *downloads, raw_tags, false, None,
                );
                if score > 0.1 {
                    let repo_name = repo_id.split('/').next_back().unwrap_or(repo_id);
                    let (preview_tags, family) = match hf_get_model_info(repo_id).await {
                        Ok(info) => {
                            let preview_tags: Vec<String> =
                                info.tags.iter().take(6).cloned().collect();
                            let family = detect_model_family(&info, repo_name);
                            (preview_tags, family)
                        }
                        Err(_) => (Vec::new(), infer_family_from_name(repo_name)),
                    };
                    scored.push(HfResolveCandidate {
                        repo_id: repo_id.clone(),
                        confidence: (score * 100.0).round() / 100.0,
                        reason: derive_resolve_reason(&score),
                        preview_tags,
                        card_url: format!("https://huggingface.co/{repo_id}"),
                        family,
                    });
                }
                continue;
            }
        };

        // Look for a file in this repo that matches the local file.
        let (file_confirmed, matched_file) =
            find_matching_file_in_repo(&files, &stem, filename, size_bytes);

        let score = score_resolve_candidate(
            repo_id,
            &stem,
            filename,
            size_bytes,
            *downloads,
            raw_tags,
            file_confirmed,
            matched_file.as_deref(),
        );

        if score > 0.1 {
            // Fetch model info for tags and family detection (non-blocking best-effort)
            let repo_name = repo_id.split('/').next_back().unwrap_or(repo_id);
            let (preview_tags, family) = match hf_get_model_info(repo_id).await {
                Ok(info) => {
                    let preview_tags: Vec<String> = info.tags.iter().take(6).cloned().collect();
                    let family = detect_model_family(&info, repo_name);
                    (preview_tags, family)
                }
                Err(_) => (Vec::new(), infer_family_from_name(repo_name)),
            };
            scored.push(HfResolveCandidate {
                repo_id: repo_id.clone(),
                confidence: (score * 100.0).round() / 100.0, // round to 2 decimals
                reason: derive_resolve_reason(&score),
                preview_tags,
                card_url: format!("https://huggingface.co/{repo_id}"),
                family,
            });
        }
    }

    scored.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Only auto-select when:
    // - top candidate is very strong
    // - there is a clear lead over others (or no close rival)
    let confident = if let Some(top) = scored.first() {
        let top_score = top.confidence;
        if top_score < 0.90 {
            false
        } else {
            match scored.get(1) {
                Some(next) if next.confidence >= top_score - 0.06 => false, // too close
                _ => true,
            }
        }
    } else {
        false
    };

    Ok(HfResolveResult {
        confident,
        candidates: scored,
        model_stem: stem,
        errors,
    })
}

/// Try to find the best matching file in the repo that corresponds to the local GGUF.
/// Returns (confirmed_exists, matched_filename).
///
/// Uses:
/// - token coverage: how many local filename tokens appear in HF filename
/// - token order: approximate order match to distinguish similar repos
/// - quant match: required if present
/// - size match: within ±5% if size known
///
/// Only returns confirmed if the match is strong on both token coverage and (when known) size.
fn find_matching_file_in_repo(
    files: &std::collections::HashMap<String, u64>,
    _stem: &str,
    filename: &str,
    size_bytes: u64,
) -> (bool, Option<String>) {
    let filename_lower = filename.to_ascii_lowercase();

    // Normalize filename into tokens (non-empty, not purely numeric)
    let local_tokens: Vec<&str> = filename_lower
        .split(['-', ' ', '_', '.', '/'])
        .filter(|t| !t.is_empty() && !t.chars().all(|c| c.is_ascii_digit()))
        .collect();

    if local_tokens.is_empty() {
        return (false, None);
    }

    // Extract quant token
    let quant_tok: Option<String> = extract_quant_token(&filename_lower);

    let mut best_match: Option<(String, f64)> = None;
    // Higher required coverage for longer token sets (e.g., "gemma-4-31B-it-qat-UD-Q4_K_XL")
    let min_coverage = if local_tokens.len() >= 6 { 0.9 } else { 0.8 };

    for (path, file_size) in files {
        let base = path.rsplit('/').next().unwrap_or(path);
        let base_lower = base.to_ascii_lowercase();

        // Size check: within ±5% if size known
        if size_bytes > 1_000_000 && *file_size > 0 {
            let diff = (*file_size).abs_diff(size_bytes);
            let rel_diff = diff as f64 / size_bytes as f64;
            if rel_diff > 0.05 {
                continue;
            }
        }

        // Check quant token presence if we have one
        if let Some(ref q) = quant_tok
            && !base_lower.contains(q)
        {
            continue;
        }

        // Token coverage: fraction of local_tokens found in HF filename
        let mut matched = 0usize;
        for token in &local_tokens {
            if base_lower.contains(token) {
                matched += 1;
            }
        }
        let coverage = matched as f64 / local_tokens.len() as f64;
        if coverage < min_coverage {
            continue;
        }

        // Order bonus: approximate order match of tokens.
        // If multiple repos match by tokens, better order -> better match.
        let order_score = token_order_score(&local_tokens, &base_lower);

        let score = coverage * 0.7 + order_score * 0.3;

        match &best_match {
            Some((_, best_score)) if score > *best_score => {
                best_match = Some((base.to_string(), score));
            }
            None => {
                best_match = Some((base.to_string(), score));
            }
            _ => {}
        }
    }

    match best_match {
        Some((name, score)) if score >= 0.9 => (true, Some(name)),
        _ => (false, None),
    }
}

/// Compute a simple order-based score for token sequence in candidate filename.
/// 0.0 = poor, 1.0 = nearly same order.
fn token_order_score(local_tokens: &[&str], candidate: &str) -> f64 {
    if local_tokens.len() <= 1 || candidate.is_empty() {
        return 1.0;
    }

    // Normalize candidate into token sequence positions.
    let candidate_tokens: Vec<&str> = candidate
        .split(['-', ' ', '_', '.', '/'])
        .filter(|t| !t.is_empty() && !t.chars().all(|c| c.is_ascii_digit()))
        .collect();

    // For each local token, find its index in candidate_tokens or use INF
    let mut positions: Vec<usize> = Vec::new();
    for t in local_tokens {
        if let Some(pos) = candidate_tokens
            .iter()
            .position(|ct| *ct == *t || ct.contains(t))
        {
            positions.push(pos);
        } else {
            return 0.0;
        }
    }

    if positions.is_empty() {
        return 0.0;
    }

    // Measure how monotonic the positions are
    let mut increasing = 0usize;
    let mut total_pairs = 0usize;
    let len = positions.len();
    for i in 1..len {
        for j in (i + 1)..len {
            total_pairs += 1;
            if positions[i] < positions[j] {
                increasing += 1;
            }
        }
    }

    if total_pairs == 0 {
        return 1.0;
    }

    increasing as f64 / total_pairs as f64
}

/// Extract a quant-like token from the filename for matching, e.g. "q4_k_m", "q8_0", "f16", "bf16".
fn extract_quant_token(filename_lower: &str) -> Option<String> {
    let parts: Vec<&str> = filename_lower.split(['-', '_', '.']).collect();
    for (i, p) in parts.iter().enumerate() {
        // Look for Q/I/B/F as first char and adjacent alphanumeric
        if p.len() > 1
            && matches!(p.chars().next(), Some('q' | 'i' | 'b' | 'f'))
            && (p.chars().nth(1).is_some_and(|c| c.is_ascii_alphanumeric())
                || p.chars().nth(1) == Some('_'))
        {
            // If next part is also related, merge (e.g. "Q4" + "K" + "M")
            if i + 1 < parts.len()
                && parts[i + 1].len() == 1
                && parts[i + 1]
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_alphabetic())
            {
                if i + 2 < parts.len()
                    && parts[i + 2].len() == 1
                    && parts[i + 2]
                        .chars()
                        .next()
                        .is_some_and(|c| c.is_ascii_alphabetic())
                {
                    return Some(format!("{}_{}-{}", p, parts[i + 1], parts[i + 2]));
                }
                return Some(format!("{}_{}", p, parts[i + 1]));
            }
            return Some(p.to_string());
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn score_resolve_candidate(
    repo_id: &str,
    stem: &str,
    filename: &str,
    size_bytes: u64,
    downloads: u64,
    raw_tags: &[String],
    file_exists_in_repo: bool,
    _matched_file: Option<&str>,
) -> f64 {
    let filename_lower = filename.to_ascii_lowercase();
    let stem_lower = stem.to_ascii_lowercase();
    let mut score = 0.0;

    // If we can't confirm the file actually exists in this repo,
    // heavily penalize unless the score is already trivially small (so it only appears as suggestion).
    if !file_exists_in_repo {
        // This repo is unlikely to be the true origin.
        // We still allow it to appear as low-confidence suggestion (based on name similarity),
        // but cap its score so it cannot become `confident:true`.
        let base = base_resolve_score(repo_id, stem, &filename_lower, downloads, raw_tags);
        return base.min(0.35); // never confident if file not found here
    }

    // Strong base for repos that actually host the matching file.
    score += 0.45;

    // Build reusable token lists.
    let stem_parts: Vec<&str> = stem_lower.split('-').filter(|p| !p.is_empty()).collect();
    let repo_name_lower = repo_id
        .split('/')
        .next_back()
        .unwrap_or(repo_id)
        .to_ascii_lowercase();

    // 1. Stem match in repo name
    let mut matched_parts = 0;
    for part in &stem_parts {
        if part.len() > 2 && repo_name_lower.contains(part) {
            matched_parts += 1;
        }
    }
    if matched_parts > 0 {
        let ratio = matched_parts as f64 / stem_parts.len() as f64;
        score += 0.20 * ratio;
        if matched_parts == stem_parts.len() {
            score += 0.05; // perfect stem match bonus
        }
    }

    // 2. Filename token match (more important than stem for distinguishing variants)
    let file_tokens: Vec<&str> = filename_lower
        .split(['-', '.', '_'])
        .filter(|t| t.len() >= 2 && !t.chars().all(|c| c.is_ascii_digit()))
        .collect();

    // Give extra weight to "rare" tokens: short, non-numeric tokens (e.g. "ud", "qat", "it", "mtp")
    // that help differentiate similar repos.
    let rare_token_chars: usize = file_tokens
        .iter()
        .filter(|t| t.len() <= 4 && !t.chars().all(|c| c.is_ascii_digit()))
        .count()
        .max(1);
    let mut file_matches = 0;
    let mut rare_matches = 0;

    for token in &file_tokens {
        if repo_name_lower.contains(token) {
            file_matches += 1;
            // Rare token if short and non-numeric; these are strong differentiators.
            if token.len() <= 4 && !token.chars().all(|c| c.is_ascii_digit()) {
                rare_matches += 1;
            }
        }
    }

    if file_matches > 0 {
        // Base filename token match
        score += 0.10 * (file_matches as f64 / file_tokens.len().max(1) as f64);
        // Rare-token match bonus: heavily rewards repos that include key discriminators like "ud"
        if rare_matches > 0 {
            score += 0.06 * (rare_matches as f64 / rare_token_chars as f64);
        }
    }

    // 3. Parameter count match
    let param_b = infer_param_b_from_name(&filename_lower);
    if param_b > 0.0 {
        let param_str = format!("{}", param_b as u64);
        for tag in raw_tags {
            let tag_lower = tag.to_ascii_lowercase();
            if tag_lower.starts_with("parameter_size:") && tag_lower.contains(&param_str) {
                score += 0.08;
                break;
            }
        }
        if repo_name_lower.contains(&param_str) {
            score += 0.05;
        }
    }

    // 4. Size agreement bonus when file_exists is true and size is known
    if file_exists_in_repo && size_bytes > 1_000_000 {
        score += 0.04;
    }

    // 5. Popularity tiebreaker (keep very weak to avoid overshadowing semantic match)
    if downloads > 1000 && score > 0.0 {
        score += 0.02;
    }

    score.clamp(0.0, 1.0)
}

/// Minimal "name-only" score for repos that do not host the file,
/// so we can still show them as weak suggestions.
fn base_resolve_score(
    repo_id: &str,
    stem: &str,
    filename_lower: &str,
    downloads: u64,
    raw_tags: &[String],
) -> f64 {
    let stem_lower = stem.to_ascii_lowercase();
    let mut score = 0.0;

    let stem_parts: Vec<&str> = stem_lower.split('-').filter(|p| !p.is_empty()).collect();
    let repo_name_lower = repo_id
        .split('/')
        .next_back()
        .unwrap_or(repo_id)
        .to_ascii_lowercase();
    let mut matched_parts = 0;
    for part in &stem_parts {
        if part.len() > 2 && repo_name_lower.contains(part) {
            matched_parts += 1;
        }
    }
    if matched_parts > 0 {
        let ratio = matched_parts as f64 / stem_parts.len() as f64;
        score += 0.35 * ratio;
    }

    // Filename token match
    let file_tokens: Vec<&str> = filename_lower
        .split(['-', '.', '_'])
        .filter(|t| t.len() >= 2 && !t.chars().all(|c| c.is_ascii_digit()))
        .collect();
    let mut file_matches = 0;
    for token in &file_tokens {
        if repo_name_lower.contains(token) {
            file_matches += 1;
        }
    }
    if file_matches > 0 {
        score += 0.15 * (file_matches as f64 / file_tokens.len().max(1) as f64);
    }

    // Parameter match
    let param_b = infer_param_b_from_name(filename_lower);
    if param_b > 0.0 {
        let param_str = format!("{}", param_b as u64);
        for tag in raw_tags {
            let tag_lower = tag.to_ascii_lowercase();
            if tag_lower.starts_with("parameter_size:") && tag_lower.contains(&param_str) {
                score += 0.08;
                break;
            }
        }
        if repo_name_lower.contains(&param_str) {
            score += 0.05;
        }
    }

    // Popularity (very weak)
    if downloads > 2000 && score > 0.0 {
        score += 0.02;
    }

    score.clamp(0.0, 0.7)
}

fn derive_resolve_reason(score: &f64) -> String {
    if *score >= 0.9 {
        "Strong match — stem, params, and filename tokens all align".to_string()
    } else if *score >= 0.8 {
        "High confidence — stem and most tokens match".to_string()
    } else if *score >= 0.6 {
        "Likely match — stem aligns, some uncertainty".to_string()
    } else if *score >= 0.4 {
        "Possible match — partial stem alignment".to_string()
    } else {
        "Weak match — low confidence".to_string()
    }
}

/// Detect the base model family from HF model info (tags + repo name).
/// Returns a lowercase slug like "qwen3.6", "llama3.3", "gemma4", "mistral", etc.
/// Empty string if family cannot be determined.
fn detect_model_family(info: &HfModelInfo, repo_name: &str) -> String {
    // 1. Check base_model tag from cardData
    for tag in &info.tags {
        let tag_lower = tag.to_ascii_lowercase();
        if let Some(rest) = tag_lower.strip_prefix("base_model:") {
            // e.g. "base_model:Qwen/Qwen3.6-27B" or "base_model:meta-llama/Llama-3.3-70B-Instruct"
            return infer_family_from_name(rest);
        }
    }

    // 2. Fall back to repo name heuristics
    infer_family_from_name(repo_name)
}

/// Infer model family slug from a name string (repo ID, model name, etc.).
fn infer_family_from_name(name: &str) -> String {
    let lower = name.to_ascii_lowercase();

    // Check in order from most specific to most general
    if lower.contains("qwen3.6") || lower.contains("qwen3_6") || lower.contains("qwen36") {
        return "qwen3.6".to_string();
    }
    if lower.contains("qwen3.5") || lower.contains("qwen3_5") || lower.contains("qwen35") {
        return "qwen3.5".to_string();
    }
    if lower.contains("qwen3") || lower.contains("qwen-3") || lower.contains("qwen_3") {
        return "qwen3".to_string();
    }
    if lower.contains("qwen2.5") || lower.contains("qwen2_5") || lower.contains("qwen25") {
        return "qwen2.5".to_string();
    }
    if lower.contains("qwen") {
        return "qwen".to_string();
    }
    if lower.contains("gemma-4") || lower.contains("gemma_4") || lower.contains("gemma4") {
        return "gemma4".to_string();
    }
    if lower.contains("gemma-3") || lower.contains("gemma_3") || lower.contains("gemma3") {
        return "gemma3".to_string();
    }
    if lower.contains("gemma-2") || lower.contains("gemma_2") || lower.contains("gemma2") {
        return "gemma2".to_string();
    }
    if lower.contains("gemma") {
        return "gemma".to_string();
    }
    if lower.contains("llama-3.3") || lower.contains("llama3_3") || lower.contains("llama33") {
        return "llama3.3".to_string();
    }
    if lower.contains("llama-3.1") || lower.contains("llama3_1") || lower.contains("llama31") {
        return "llama3.1".to_string();
    }
    if lower.contains("llama-3") || lower.contains("llama3") {
        return "llama3".to_string();
    }
    if lower.contains("llama") {
        return "llama".to_string();
    }
    if lower.contains("mistral-large") {
        return "mistral-large".to_string();
    }
    if lower.contains("mistral-nemo") || lower.contains("nemo") {
        return "mistral-nemo".to_string();
    }
    if lower.contains("mistral") || lower.contains("mixtral") {
        return "mistral".to_string();
    }
    if lower.contains("deepseek") {
        return "deepseek".to_string();
    }
    if lower.contains("phi") {
        return "phi".to_string();
    }
    if lower.contains("yi") && lower.contains("model") {
        return "yi".to_string();
    }

    String::new()
}

// ── Gemma4 MTP Draft Resolution ──────────────────────────────────────────────

/// Resolve the Unsloth MTP draft model repo and filename for a Gemma4 model.
///
/// Given the main model's repo_id (e.g. `unsloth/gemma-4-31B-it-qat-GGUF`) and
/// quant label (e.g. `Q8_0`), constructs the HF path to the matching MTP draft
/// model and a **local** filename that avoids QAT/non-QAT collisions.
///
/// Returns `(hf_repo_id, hf_filename, local_filename)`.
///
/// Example:
/// ```text
/// Input:  unsloth/gemma-4-31B-it-qat-GGUF, Q8_0
/// HF:     unsloth/gemma-4-31B-it-qat-GGUF/MTP/gemma-4-31B-it-Q8_0-MTP.gguf
/// Local:  gemma-4-31B-it-qat-Q8_0-MTP.gguf  (note: `-qat` inserted)
/// ```
pub fn resolve_gemma4_mtp_draft(
    repo_id: &str,
    quant_label: &str,
) -> Option<(String, String, String)> {
    let lower = repo_id.to_ascii_lowercase();

    // Determine if this is a QAT repo
    let is_qat = lower.contains("-qat");

    // Extract the Gemma4 tier from the repo id
    let tier = resolve_gemma4_tier(&lower);
    let tier = tier.as_deref()?; // Returns None if not a recognized Gemma4 tier

    // Construct the HF filename: gemma-4-{tier}-it-{quant}-MTP.gguf
    let hf_filename = format!("gemma-4-{}-it-{}-MTP.gguf", tier, quant_label);

    // Construct the local filename: insert -qat if needed to avoid collision
    let local_filename = if is_qat {
        format!("gemma-4-{}-it-qat-{}-MTP.gguf", tier, quant_label)
    } else {
        format!("gemma-4-{}-it-{}-MTP.gguf", tier, quant_label)
    };

    // The HF repo for MTP files is the same as the main repo, just /MTP/ subfolder
    Some((repo_id.to_string(), hf_filename, local_filename))
}

/// Extract the Gemma4 tier slug from a repo id or model name.
/// Returns lowercase tier: "e2b", "e4b", "12b", "26b-a4b", "31b", or "2b"
pub fn resolve_gemma4_tier(name_lower: &str) -> Option<String> {
    // Order matters: check more specific patterns first (26b-a4b before 26b)
    if name_lower.contains("26b") || name_lower.contains("a4b") {
        Some("26b-a4b".to_string())
    } else if name_lower.contains("31b") {
        Some("31b".to_string())
    } else if name_lower.contains("12b") {
        Some("12b".to_string())
    } else if name_lower.contains("e4b") {
        Some("e4b".to_string())
    } else if name_lower.contains("e2b") {
        Some("e2b".to_string())
    } else if name_lower.contains("2b") {
        Some("2b".to_string())
    } else {
        None
    }
}

/// Check if a local draft model is compatible with the given Gemma4 main model tier.
/// Returns true if the draft model's tier matches the main model's tier.
fn is_draft_compatible_with_tier(draft_name_lower: &str, tier: &str) -> bool {
    // Both the draft and main model should reference the same tier
    match tier {
        "26b-a4b" => draft_name_lower.contains("26b") || draft_name_lower.contains("a4b"),
        "31b" => draft_name_lower.contains("31b"),
        "12b" => draft_name_lower.contains("12b") && !draft_name_lower.contains("e12"),
        "e4b" => draft_name_lower.contains("e4b"),
        "e2b" => draft_name_lower.contains("e2b"),
        "2b" => draft_name_lower.contains("2b") && !draft_name_lower.contains("e2b"),
        _ => false,
    }
}

/// Find a compatible local MTP draft model for a Gemma4 main model.
///
/// Scans the models directory for a `-MTP.gguf` file matching the same tier.
/// Returns the full path of the best match, or None.
pub fn find_compatible_gemma4_mtp_draft(
    models_dir: &std::path::Path,
    main_model_name: &str,
) -> Option<std::path::PathBuf> {
    let tier = resolve_gemma4_tier(&main_model_name.to_ascii_lowercase())?;

    // Walk models dir for draft candidates
    let entries = std::fs::read_dir(models_dir).ok()?;
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();

    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_lowercase();
        if is_draft_assistant_hf(&name, 0) && is_draft_compatible_with_tier(&name, &tier) {
            candidates.push(entry.path());
        }
    }

    // Prefer exact quant match (if main model is Q8_0, prefer Q8_0 draft)
    let main_lower = main_model_name.to_ascii_lowercase();
    let best = candidates.iter().find(|p| {
        let name = p
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_lowercase();
        // Check if the quant level in the draft name matches the main model
        if main_lower.contains("q8_0") && name.contains("q8_0") {
            return true;
        }
        if main_lower.contains("q4_k_m") && name.contains("q4_k_m") {
            return true;
        }
        if main_lower.contains("q4_0") && name.contains("q4_0") {
            return true;
        }
        if main_lower.contains("bf16") && name.contains("bf16") {
            return true;
        }
        if main_lower.contains("f16") && name.contains("f16") {
            return true;
        }
        false
    });

    best.cloned().or_else(|| candidates.into_iter().next())
}

/// Validate a HuggingFace repo ID format (owner/name).
pub fn validate_hf_repo_id(repo_id: &str) -> bool {
    !repo_id.is_empty() && repo_id.contains('/') && repo_id.split('/').count() == 2
}

/// Fetch raw bytes from a file at a revision from HF.
/// Returns up to max_size bytes.
#[allow(dead_code)]
pub async fn fetch_raw_bytes_at(
    repo_id: &str,
    revision: &str,
    file_path: &str,
    max_size: u64,
) -> Result<Vec<u8>, String> {
    let url = hf_resolve_download_url_at(repo_id, file_path, revision);
    let token = hf_load_token();
    let mut req = HF_HTTP_CLIENT.get(&url);
    if let Some(tok) = token {
        req = req.bearer_auth(tok);
    }
    let resp = req
        .send()
        .await
        .map_err(|e| format!("HTTP error fetching {}: {}", file_path, e))?;
    if !resp.status().is_success() {
        return Err(format!("{}: HTTP {}", file_path, resp.status()));
    }
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| format!("Read error for {}: {}", file_path, e))?;
    let bytes = bytes.as_ref();
    if bytes.len() as u64 > max_size {
        return Err(format!(
            "{}: exceeds size limit ({}/{} bytes)",
            file_path,
            bytes.len(),
            max_size
        ));
    }
    Ok(bytes.to_vec())
}

/// List all files in a HF repo using the HF tree API.
#[allow(dead_code)]
pub async fn list_repo_siblings(repo_id: &str) -> Result<Vec<String>, String> {
    let url = format!("https://huggingface.co/api/models/{}/tree/main", repo_id);
    let token = hf_load_token();
    let mut req = HF_HTTP_CLIENT.get(&url);
    if let Some(tok) = token {
        req = req.bearer_auth(tok);
    }
    let resp = req
        .send()
        .await
        .map_err(|e| format!("HTTP error listing repo {}: {}", repo_id, e))?;
    if !resp.status().is_success() {
        return Err(format!("List repo {}: HTTP {}", repo_id, resp.status()));
    }
    let items: Vec<serde_json::Value> = resp
        .json()
        .await
        .map_err(|e| format!("JSON parse error for {}: {}", repo_id, e))?;
    let paths: Vec<String> = items
        .into_iter()
        .filter_map(|v| v.get("path").and_then(|p| p.as_str().map(String::from)))
        .collect();
    Ok(paths)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Every sort the UI offers must reach HF as a key HF actually understands, or as a
    /// deliberate absence. The shipped version mapped Name to `createdAt` and Size to
    /// `downloads`, so five visible options produced two orderings and the dropdown was
    /// decorative. Nothing caught it because nothing asserted the mapping.
    #[test]
    fn every_sort_maps_to_a_real_hf_key_or_a_deliberate_absence() {
        // The keys HF's /api/models endpoint accepts for `sort`.
        let hf_accepts = [
            "downloads",
            "likes",
            "createdAt",
            "lastModified",
            "trendingScore",
        ];

        for sort in [
            HfSort::Downloads,
            HfSort::Likes,
            HfSort::CreatedAt,
            HfSort::LastModified,
            HfSort::Trending,
        ] {
            let key = sort
                .as_api_str()
                .unwrap_or_else(|| panic!("{sort:?} must send a sort key"));
            assert!(
                hf_accepts.contains(&key),
                "{sort:?} sends unknown key {key:?}"
            );
        }

        // Relevance is the one mode defined by sending nothing: HF ranks by the search term
        // only when `sort` is absent, so giving it any key would silently replace the very
        // ordering the mode exists to show.
        assert_eq!(HfSort::Relevance.as_api_str(), None);
    }

    /// Distinct sorts must stay distinct. Collapsing two modes onto one key is precisely how
    /// the previous mapping became a no-op, and it reads as harmless at the call site.
    #[test]
    fn distinct_sorts_do_not_collapse_onto_one_key() {
        let keys: Vec<_> = [
            HfSort::Downloads,
            HfSort::Likes,
            HfSort::CreatedAt,
            HfSort::LastModified,
            HfSort::Trending,
        ]
        .iter()
        .map(|s| s.as_api_str().unwrap())
        .collect();

        let unique: std::collections::BTreeSet<_> = keys.iter().collect();
        assert_eq!(
            unique.len(),
            keys.len(),
            "two sorts share an API key: {keys:?}"
        );
    }

    #[test]
    fn mlx_revision_and_text_config_references_are_bounded_and_safe() {
        for revision in [
            "main",
            "a4c7561cc890307b95b473f8564634c3d598734a",
            "v0.10.17",
        ] {
            assert!(validate_hf_revision(revision).is_ok(), "{revision}");
        }
        for revision in ["", "main/next", "../main", "main?ref=x"] {
            assert!(validate_hf_revision(revision).is_err(), "{revision}");
        }
        assert!(validate_hf_revision(&"a".repeat(129)).is_err());

        for reference in [
            "text_config.json",
            "configs/text.json",
            "nested/config.json",
        ] {
            assert!(
                validate_mlx_config_reference(reference).is_ok(),
                "{reference}"
            );
        }
        for reference in [
            "",
            "/config.json",
            "../config.json",
            "configs/../../text.json",
            "config.yaml",
        ] {
            assert!(
                validate_mlx_config_reference(reference).is_err(),
                "{reference}"
            );
        }
    }

    #[tokio::test]
    async fn test_hf_get_model_info_smoke() {
        let _ = hf_get_model_info("gpt2").await;
    }

    #[test]
    fn test_quants_only_uses_declared_quantized_base_tag() {
        assert!(has_quantized_base_model_tag(&serde_json::json!({
            "tags": ["mlx", "base_model:quantized:mistralai/Mistral-7B-Instruct-v0.3"]
        })));
        assert!(!has_quantized_base_model_tag(&serde_json::json!({
            "tags": ["mlx", "base_model:mistralai/Mistral-7B-Instruct-v0.3"]
        })));
    }

    #[test]
    fn test_infer_quant_label() {
        assert_eq!(infer_quant_label("model-Q4_K_M.gguf"), "Q4_K_M");
        assert_eq!(infer_quant_label("model-Q8_0.gguf"), "Q8_0");
        assert_eq!(infer_quant_label("model-bf16.gguf"), "BF16");
        assert_eq!(infer_quant_label("model-random.gguf"), "Unknown");
        assert_eq!(infer_quant_label("model.i1-IQ3_S.gguf"), "IQ3_S");
        assert_eq!(infer_quant_label("Qwen3.6-27B-UD-Q4_K_M.gguf"), "Q4_K_M");
        assert_eq!(infer_quant_label("model-Q5_K_XL.gguf"), "Q5_K_XL");
        assert_eq!(infer_quant_label("model-IQ2_XXS.gguf"), "IQ2_XXS");
        // Non-standard but real quant names should be preserved, not collapsed to "Unknown".
        assert_eq!(
            infer_quant_label("Hermes3.6-35B-A3B-Genesis-V7-Q8_K_P.gguf"),
            "Q8_K_P"
        );
        // APEX/MTP/Compact are this uploader's own custom scheme, not Unsloth's UD —
        // don't mislabel them "UD (custom)".
        assert_eq!(
            infer_quant_label("Hermes3.6-35B-A3B-Genesis-V7-MTP-APEX.gguf"),
            "MTP-APEX (custom)"
        );
        assert_eq!(
            infer_quant_label("Hermes3.6-35B-A3B-Genesis-V7-APEX.gguf"),
            "APEX (custom)"
        );
        assert_eq!(
            infer_quant_label("Hermes3.6-35B-A3B-Genesis-V7-MTP-APEX-Compact.gguf"),
            "MTP-APEX-Compact (custom)"
        );
        assert_eq!(
            infer_quant_label("Hermes3.6-35B-A3B-Genesis-V7-APEX-Compact.gguf"),
            "APEX-Compact (custom)"
        );
        // Arbitrary uploader suffixes (Quality, I-Quality, Balanced, ...) can't be
        // enumerated in advance — must be preserved verbatim, not collapsed.
        assert_eq!(
            infer_quant_label("Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf"),
            "APEX-MTP-I-Quality (custom)"
        );
        assert_eq!(
            infer_quant_label("model-APEX-MTP-Balanced.gguf"),
            "APEX-MTP-Balanced (custom)"
        );
    }

    #[test]
    fn test_infer_gemma4_family() {
        assert_eq!(
            infer_family_from_name("unsloth/gemma-4-31B-it-qat-GGUF"),
            "gemma4"
        );
        assert_eq!(
            infer_family_from_name("google/gemma_4_12B_it_qat_q4_0_gguf"),
            "gemma4"
        );
    }

    #[test]
    fn test_mmproj_preferences_are_family_specific() {
        let qwen = mmproj_preference_for_family("qwen3.6").expect("Qwen preference");
        assert_eq!(qwen.label, "F16");

        let gemma = mmproj_preference_for_family("gemma4").expect("Gemma preference");
        assert_eq!(gemma.label, "F16");

        assert!(mmproj_preference_for_family("llama3").is_none());
    }

    #[test]
    fn test_detect_quant_type() {
        // Mradermacher imatrix
        assert_eq!(
            detect_quant_type("model.i1-Q4_K_M.gguf"),
            QuantFileType::Imatrix
        );
        assert_eq!(
            detect_quant_type("model-i1-IQ3_S.gguf"),
            QuantFileType::Imatrix
        );
        // Unsloth UD
        assert_eq!(
            detect_quant_type("Qwen3.6-27B-UD-Q4_K_S.gguf"),
            QuantFileType::UnslothDynamic
        );
        assert_eq!(
            detect_quant_type("model-UD-mmproj-BF16.gguf"),
            QuantFileType::UnslothDynamic
        );
        // Standard
        assert_eq!(
            detect_quant_type("model-Q4_K_M.gguf"),
            QuantFileType::Standard
        );
        assert_eq!(
            detect_quant_type("bartowski-model-Q5_K_M.gguf"),
            QuantFileType::Standard
        );
    }

    #[test]
    fn test_quant_provider_from_username() {
        assert!(matches!(
            QuantProvider::from_username("bartowski"),
            QuantProvider::Bartowski
        ));
        assert!(matches!(
            QuantProvider::from_username("mradermacher"),
            QuantProvider::Mradermacher
        ));
        assert!(matches!(
            QuantProvider::from_username("unsloth"),
            QuantProvider::Unsloth
        ));
        assert!(matches!(
            QuantProvider::from_username("DavidAU"),
            QuantProvider::Community
        ));
        // llmfan46 replaced TheBloke in the known quantizers list
        assert!(matches!(
            QuantProvider::from_username("llmfan46"),
            QuantProvider::Community
        ));
    }

    #[test]
    fn test_mask_token() {
        assert_eq!(mask_token("1234567890"), "1234****7890");
        assert_eq!(mask_token("1234"), "****");
    }

    #[test]
    fn test_infer_param_b_from_name() {
        assert_eq!(infer_param_b_from_name("bartowski/Qwen3.6-27B-GGUF"), 27.0);
        assert_eq!(
            infer_param_b_from_name("mradermacher/Qwen3.6-35B-A3B-Instruct-i1-GGUF"),
            35.0
        );
        assert_eq!(
            infer_param_b_from_name("unsloth/Llama-3.3-70B-Instruct-GGUF"),
            70.0
        );
        // Should be > 0
        assert!(infer_param_b_from_name("some-27b-model") > 0.0);
    }

    #[test]
    fn test_has_imatrix_detection() {
        let item = serde_json::json!({
            "id": "mradermacher/Qwen3.6-27B-i1-GGUF",
            "gated": false,
            "tags": ["gguf", "imatrix"],
            "downloads": 1000,
            "likes": 50,
        });
        let info = parse_model_item(item, &HfModelFormat::Gguf).unwrap();
        assert!(
            info.has_imatrix,
            "mradermacher repo should be flagged as imatrix"
        );
    }

    #[test]
    fn model_search_preserves_the_hub_commit_for_pinned_selection() {
        let item = serde_json::json!({
            "id": "mlx-community/Qwen",
            "sha": "abcdef1234567890",
            "tags": ["mlx"],
        });
        let info = parse_model_item(item, &HfModelFormat::Mlx).unwrap();
        assert_eq!(info.revision, "abcdef1234567890");

        let without_sha = serde_json::json!({
            "id": "mlx-community/Qwen",
            "tags": ["mlx"],
        });
        assert!(
            parse_model_item(without_sha, &HfModelFormat::Mlx)
                .unwrap()
                .revision
                .is_empty(),
            "missing Hub metadata must stay unpinned rather than inventing main"
        );
    }

    #[test]
    fn test_sort_rank_quality_order() {
        // Higher quality should sort before lower quality
        assert!(sort_rank_quant_label("Q8_0") < sort_rank_quant_label("Q4_K_M"));
        assert!(sort_rank_quant_label("Q4_K_M") < sort_rank_quant_label("Q3_K_M"));
        assert!(sort_rank_quant_label("Q3_K_M") < sort_rank_quant_label("IQ2_XXS"));
    }

    #[test]
    fn test_mmproj_sort_prefers_practical_precision_over_f32() {
        assert!(sort_rank_mmproj_label("F16") < sort_rank_mmproj_label("BF16"));
        assert!(sort_rank_mmproj_label("BF16") < sort_rank_mmproj_label("F32"));
    }

    #[test]
    fn test_extract_static_quant_repo_from_mradermacher_card() {
        let readme = r#"
static quants are available at https://huggingface.co/mradermacher/Qwen3.5-122B-A10B-REAP-20-GGUF

This is a vision model - mmproj files (if any) will be in the static repository.
"#;
        assert_eq!(
            extract_static_quant_repos(readme, "mradermacher/Qwen3.5-122B-A10B-REAP-20-i1-GGUF"),
            vec!["mradermacher/Qwen3.5-122B-A10B-REAP-20-GGUF"]
        );
    }

    #[test]
    fn test_mradermacher_static_repo_fallback() {
        assert_eq!(
            mradermacher_static_repo("mradermacher/Qwen3.5-122B-A10B-REAP-20-i1-GGUF").as_deref(),
            Some("mradermacher/Qwen3.5-122B-A10B-REAP-20-GGUF")
        );
        assert!(mradermacher_static_repo("other/model-i1-GGUF").is_none());
    }

    #[test]
    fn test_qwen35_arch_companion_repo_fallbacks() {
        assert_eq!(
            qwen35_arch_companion_repos("mradermacher/Qwen3.5-122B-A10B-REAP-20-i1-GGUF"),
            vec![
                "mradermacher/Qwen3.5-122B-A10B-GGUF",
                "unsloth/Qwen3.5-122B-A10B-GGUF"
            ]
        );
        assert_eq!(
            qwen35_arch_companion_repos("mradermacher/Qwen3.5-27B-heretic-i1-GGUF"),
            vec!["mradermacher/Qwen3.5-27B-GGUF", "unsloth/Qwen3.5-27B-GGUF"]
        );
        assert!(qwen35_arch_companion_repos("other/Qwen3.5-27B-i1-GGUF").is_empty());
    }
}

// ── MLX Native/Conversion Discovery (Phase 8A3) ──────────────────────────────────────────

/// Discovery result for MLX derivatives of a finetune.
///
/// Finds native MLX artifacts and authoritative safetensors conversion candidates
/// (builder item 11). Original author is always preserved as a separate role from
/// converter/publisher; original author never appears as converter.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MlxDiscoveryResult {
    /// The source repo being analyzed
    pub source_repo_id: String,
    /// Whether this source is likely a finetune
    pub source_is_finetune: bool,
    /// Original author of the finetune (if identifiable)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_author: Option<String>,
    /// Original author is always preserved through conversion
    #[serde(default)]
    pub original_author_preserved: bool,
    /// Native MLX repos that are derivatives of this source
    #[serde(default)]
    pub native_mlx_derivatives: Vec<MlxDerivative>,
    /// Safetensors conversion recipes available
    #[serde(default)]
    pub conversion_recipes: Vec<MlxConversionRecipeInfo>,
    /// Errors encountered
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MlxDerivative {
    /// HF repo id of the MLX derivative
    pub repo_id: String,
    /// Revision
    pub revision: String,
    /// Converter/publisher (never the original author)
    pub converter: String,
    /// Format type ("mlx" or "safetensors")
    pub format: String,
    /// Whether this repo has been qualified for Rapid-MLX
    #[serde(default)]
    pub is_qualified: bool,
    /// Total repo size in bytes (from HF tree API, 0 if unknown)
    #[serde(default)]
    pub size: u64,
    /// Quantization info from config (bits/group_size), if present
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant: Option<crate::inference::rapid_mlx::mlx_meta::MlxQuantization>,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MlxConversionRecipeInfo {
    /// Recipe identifier
    pub recipe_id: String,
    /// Human-readable name
    pub recipe: String,
    /// Description
    pub description: String,
    /// Required source format
    pub input_format: String,
    /// Output MLX format
    pub output_format: String,
    /// Estimated additional disk space required (bytes, 0 if unknown)
    #[serde(default)]
    pub estimated_disk: u64,
    /// Estimated conversion time label ("fast" / "moderate" / "slow")
    pub estimated_time: String,
    /// Available quantization options
    #[serde(default)]
    pub quant_options: Vec<String>,
    /// Provenance/source of this recipe
    pub provenance: String,
}

/// Known MLX converter publishers (per builder item 11/D29).
fn known_mlx_publishers() -> &'static [&'static str] {
    &["mlx-community", "ml-explore", "davidau", "mlabonne"]
}

/// Infer quantization from an MLX config if available.
async fn fetch_mlx_quant_for_repo(
    repo_id: &str,
) -> Option<crate::inference::rapid_mlx::mlx_meta::MlxQuantization> {
    let cfg_result = fetch_mlx_config(repo_id, "config.json").await;
    match cfg_result {
        Ok(cfg) => cfg.quantization,
        Err(_) => None,
    }
}

/// Check if this repo is qualified for Rapid-MLX (if qualify module available).
async fn check_qualified(repo_id: &str) -> bool {
    // For now, use a simple heuristic: mlx-community repos with config.json are treated as qualified.
    // The full qualify module integration is provided by Phase 8A2.
    let url = format!("https://huggingface.co/{repo_id}/raw/main/config.json");
    let resp = match HF_HTTP_CLIENT.get(&url).send().await {
        Ok(r) => r,
        Err(_) => return false,
    };
    resp.status().is_success() && repo_id.starts_with("mlx-community/")
}

/// Query HF's declared `base_model:quantized:` tag for MLX sibling quantizations.
///
/// The Hub's `other` and `base_model_relation` query parameters are silently
/// ignored by the models API. `filter` is the supported exact tag filter. The
/// selected repo is often itself a quantized leaf, so first resolve its declared
/// base model and then ask for all MLX quantizations of that base.
async fn fetch_mlx_relation_derivatives(repo_id: &str) -> Result<Vec<SimpleModelInfo>, String> {
    let base_model = hf_get_model_info(repo_id)
        .await
        .map_err(|e| format!("Failed to resolve MLX base model: {e}"))?
        .tags
        .into_iter()
        .find_map(|tag| tag.strip_prefix("base_model:quantized:").map(str::to_owned))
        .unwrap_or_else(|| repo_id.to_string());

    let token = hf_load_token();
    let mut url = reqwest::Url::parse("https://huggingface.co/api/models")
        .map_err(|e| format!("Invalid HF API URL: {e}"))?;
    {
        let mut p = url.query_pairs_mut();
        p.append_pair("filter", &format!("base_model:quantized:{base_model}"));
        p.append_pair("filter", "mlx");
        p.append_pair("limit", "50");
        p.append_pair("expand[]", "safetensors");
    }

    let mut req = HF_HTTP_CLIENT.get(url);
    if let Some(ref tok) = token {
        req = req.bearer_auth(tok);
    }

    let resp = req
        .send()
        .await
        .map_err(|e| format!("HF relation query failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("HF relation query failed: HTTP {}", resp.status()));
    }

    let items: Vec<serde_json::Value> = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse HF relation response: {e}"))?;

    Ok(items
        .into_iter()
        .filter_map(|item| parse_model_item(item, &HfModelFormat::Mlx))
        .collect())
}

/// Discover MLX derivatives and conversion recipes for a source repo.
///
/// Searches HF for native MLX derivatives of the given finetune, using the original
/// author/converter separation required by builder item 11. The original author is
/// never listed as a converter; converter is a separate evidence-bearing role.
pub async fn hf_discover_mlx_derivatives(repo_id: &str) -> Result<MlxDiscoveryResult, String> {
    let mut errors = Vec::new();

    // Check if the source is likely a finetune (heuristic)
    let source_is_finetune = repo_id.contains("-ft")
        || repo_id.contains("-heretic")
        || repo_id.contains("finetune")
        || repo_id.contains("merge")
        || ["unsloth", "davidau", "heretic"]
            .iter()
            .any(|o| repo_id.starts_with(*o));

    // Try to find original author via identity resolution (preferred) or heuristic
    // Identity resolution uses the full qualify module from Phase 8A2; fall back to heuristic
    let original_author = if source_is_finetune {
        Some(repo_id.split('/').next().unwrap_or("").to_string())
    } else {
        None
    };

    // Derive a search stem from the source repo name (without owner prefix)
    let stem = repo_id.split('/').next_back().unwrap_or(repo_id);

    let mut native_mlx_derivatives: Vec<MlxDerivative> = Vec::new();

    // Primary path: HF's author-declared model-relation graph (`base_model:` tag).
    // This is authoritative — a repo explicitly declares what it was quantized from —
    // unlike the fuzzy stem-name search below, which is a guess.
    let relation_results = match fetch_mlx_relation_derivatives(repo_id).await {
        Ok(results) => results,
        Err(e) => {
            errors.push(format!("relation query: {e}"));
            Vec::new()
        }
    };

    // Fall back to fuzzy stem search only when the relation graph has nothing declared
    // (common for older or less-diligently-tagged repos).
    let via_relation = !relation_results.is_empty();
    let candidates = if via_relation {
        relation_results
    } else {
        match hf_search_models(&HfSearchParams {
            query: format!("{stem} mlx"),
            author: None,
            sort: HfSort::Downloads,
            limit: 15,
            cursor: None,
            format: HfModelFormat::Mlx,
            quants_only: true,
        })
        .await
        {
            Ok((results, _)) => results,
            Err(e) => {
                errors.push(format!("search: {e}"));
                Vec::new()
            }
        }
    };

    for item in candidates {
        // Must not be the source repo itself; when falling back to fuzzy search,
        // also require the stem to actually appear in the candidate's id (the
        // relation-graph path is already authoritative and needs no such check).
        if item.id == repo_id {
            continue;
        }
        if !via_relation {
            let id_lower = item.id.to_ascii_lowercase();
            let stem_lower = stem.to_ascii_lowercase();
            if !id_lower.contains(&stem_lower[..stem_lower.len().min(16)]) {
                continue;
            }
        }

        // Original author never appears as converter
        let converter = item.author.clone();
        if let Some(ref orig) = original_author
            && converter.eq_ignore_ascii_case(orig)
        {
            continue;
        }

        // Prefer known MLX publishers
        let is_known_publisher = known_mlx_publishers()
            .iter()
            .any(|p| converter.to_ascii_lowercase() == *p);

        // Skip low-signal results unless they're known publishers
        if !is_known_publisher && item.downloads < 10 {
            continue;
        }

        // Fetch size: prefer what the search response already carried (safetensors
        // expand or relation-query result), only hitting the tree API if still unknown
        let size = if let Some(bytes) = item.model_size_bytes {
            bytes
        } else {
            resolve_mlx_repo_size_bytes(&item.id)
                .await
                .unwrap_or_default()
                .unwrap_or(0)
        };

        // Fetch quant from config
        let quant = fetch_mlx_quant_for_repo(&item.id).await;

        // Check qualification
        let is_qualified = check_qualified(&item.id).await;

        native_mlx_derivatives.push(MlxDerivative {
            repo_id: item.id,
            revision: "main".into(),
            converter,
            format: "mlx".into(),
            is_qualified,
            size,
            quant,
        });
    }

    // Sort by known publisher (preferred), then qualification
    native_mlx_derivatives.sort_by(|a, b| {
        let a_known = known_mlx_publishers()
            .iter()
            .any(|p| a.converter.to_ascii_lowercase() == *p);
        let b_known = known_mlx_publishers()
            .iter()
            .any(|p| b.converter.to_ascii_lowercase() == *p);
        if a_known != b_known {
            b_known.cmp(&a_known)
        } else {
            let a_q = a.is_qualified as u8;
            let b_q = b.is_qualified as u8;
            if a_q != b_q {
                b_q.cmp(&a_q)
            } else {
                a.repo_id.cmp(&b.repo_id)
            }
        }
    });

    native_mlx_derivatives.dedup_by(|a, b| a.repo_id == b.repo_id);

    // Standard conversion recipes (app-supported MLX conversion paths)
    let recipes = vec![
        MlxConversionRecipeInfo {
            recipe_id: "mlx_lm_load_original_f16".into(),
            recipe: "MLX-LM F16".into(),
            description: "Convert safetensors to MLX F16 format using mlx-lm tools".into(),
            input_format: "transformers (safetensors)".into(),
            output_format: "mlx (F16 safetensors shards)".into(),
            estimated_disk: 0,
            estimated_time: "moderate".into(),
            quant_options: vec!["fp16".into()],
            provenance: "mlx-community / mlx-lm load-original".into(),
        },
        MlxConversionRecipeInfo {
            recipe_id: "mlx_lm_load_original_4bit".into(),
            recipe: "MLX-LM 4-bit".into(),
            description: "Convert safetensors to MLX 4-bit quantized format".into(),
            input_format: "transformers (safetensors)".into(),
            output_format: "mlx (4-bit quantized safetensors)".into(),
            estimated_disk: 0,
            estimated_time: "moderate".into(),
            quant_options: vec!["4-bit mx4_4".into()],
            provenance: "mlx-community / mlx-lm convert-to-mx".into(),
        },
    ];

    Ok(MlxDiscoveryResult {
        source_repo_id: repo_id.to_string(),
        source_is_finetune,
        original_author,
        original_author_preserved: true,
        native_mlx_derivatives,
        conversion_recipes: recipes,
        errors,
    })
}
