use anyhow::{Context, Result, anyhow};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use tokio::io::AsyncReadExt;
use tokio::process::Command;

const INFO_TIMEOUT: Duration = Duration::from_secs(10);
// The `rapid-mlx models` listing is parsed and unit-tested but has no caller; model
// discovery currently reads the filesystem and the HF API instead. Phase 8 owns the decision
// to wire this or drop it.
const MODELS_TIMEOUT: Duration = Duration::from_secs(8);
const MAX_OUTPUT_BYTES: usize = 256 * 1024;

/// Minimum minor version at which the `rapid-mlx info <model>` output layout is trusted.
/// Below this, parsing may produce incorrect fields, so we return a minimal profile
/// with no recommendation hints (fields fall back to defaults/unknowns).
pub(crate) const MIN_TRUSTED_MINOR: u64 = 10;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub struct ModelProfile {
    #[serde(default)]
    pub tool_format: Option<String>,
    #[serde(default)]
    pub reasoning_parser: Option<String>,
    #[serde(default)]
    pub architecture: Option<String>,
    #[serde(default)]
    pub spec_decode: SpecDecodeSupport,
    #[serde(default)]
    pub mtp_path: Option<MtpPathStatus>,
    #[serde(default)]
    pub kv_share: Option<bool>,
    #[serde(default)]
    pub throttle: Option<bool>,
    #[serde(default)]
    pub suffix_tier: Option<String>,
    #[serde(default)]
    pub dflash_eligibility: Eligibility,
    #[serde(default)]
    pub ddtree_eligibility: Eligibility,
    #[serde(default)]
    pub extras: ExtraCapabilities,
    #[serde(default)]
    pub is_finetune: bool,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SpecDecodeSupport {
    Supported,
    Unsupported,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum MtpPathStatus {
    Enabled,
    Disabled,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub struct Eligibility {
    pub supported: Option<bool>,
    #[serde(default)]
    pub reasons: BTreeMap<String, Option<String>>,
}

impl Eligibility {
    // The `rapid-mlx models` listing is parsed and unit-tested but has no caller; model
    // discovery currently reads the filesystem and the HF API instead. Phase 8 owns the decision
    // to wire this or drop it.
    #[allow(dead_code)]
    pub fn is_eligible(&self) -> bool {
        self.supported == Some(true)
    }

    #[allow(dead_code)] // see the note on this impl block
    pub fn is_ineligible(&self) -> bool {
        self.supported == Some(false)
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct ExtraCapabilities {
    #[serde(default)]
    pub vision: bool,
    #[serde(default)]
    pub has_vision_tower: bool,
    #[serde(default)]
    pub embeddings: bool,
    #[serde(default)]
    pub mtp_dflash: bool,
}

/// One row of the curated `rapid-mlx models` catalog. Upstream hand-validates
/// every entry (parser pairing, MTP sidecar, measured sizes), so this list —
/// not raw HF discovery — is the Rapid-MLX model surface.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelListEntry {
    pub name: String,
    pub display_name: String,
    /// Download size in bytes, when the listing reports one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
    /// Tool-call parser upstream pairs with this model (`—` → None).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parser: Option<String>,
    /// Chat template family (`—` → None).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub template: Option<String>,
    /// Upstream marks hybrid-attention (GatedDeltaNet) models explicitly; these
    /// cannot quantize KV (ArraysCache) and always serve bf16.
    pub hybrid: bool,
    pub mtp: bool,
    /// Speculative-decode sidecar repo id when upstream pairs one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mtp_sidecar: Option<String>,
}

#[derive(Debug, Clone)]
struct QueryOutput {
    stdout: String,
}

#[allow(clippy::type_complexity)]
static VERSION_CACHE: OnceLock<Arc<std::sync::RwLock<Option<(String, u64)>>>> = OnceLock::new();

/// Query (and cache) the `rapid-mlx --version` output as `(exact_string, minor)`.
/// Shared by callers that need to version-guard text-scraping of other
/// subcommands (e.g. `info`, `bench`) before trusting their output layout.
pub(crate) async fn cached_version(binary: &Path) -> Result<Option<(String, u64)>> {
    let cache = VERSION_CACHE
        .get_or_init(|| Arc::new(std::sync::RwLock::new(None)))
        .clone();
    if let Some(version) = cache.read().unwrap().as_ref().cloned() {
        return Ok(Some(version));
    }
    let output = run_query(binary, &["--version"], INFO_TIMEOUT, MAX_OUTPUT_BYTES).await?;
    let text = output.stdout.trim();
    let parsed = parse_version_number(text);
    if let Some((exact, minor)) = parsed {
        *cache.write().unwrap() = Some((exact.clone(), minor));
        Ok(Some((exact, minor)))
    } else {
        Ok(None)
    }
}

fn parse_version_number(text: &str) -> Option<(String, u64)> {
    for start in 0..text.len() {
        if !text.as_bytes()[start].is_ascii_digit() {
            continue;
        }
        let mut cursor = start;
        let bytes = text.as_bytes();
        if let Some(_major) = parse_num(bytes, &mut cursor)
            && bytes.get(cursor) == Some(&b'.')
        {
            cursor += 1;
            if let Some(minor) = parse_num(bytes, &mut cursor)
                && bytes.get(cursor) == Some(&b'.')
            {
                cursor += 1;
                if let Some(_patch) = parse_num(bytes, &mut cursor) {
                    let suffix_end = bytes[cursor..]
                        .iter()
                        .position(|b| b.is_ascii_whitespace())
                        .map_or(bytes.len(), |off| cursor + off);
                    let exact = text[start..suffix_end].to_string();
                    return Some((exact, minor));
                }
            }
        }
    }
    None
}

fn parse_num(bytes: &[u8], cursor: &mut usize) -> Option<u64> {
    let start = *cursor;
    while bytes.get(*cursor).is_some_and(u8::is_ascii_digit) {
        *cursor += 1;
    }
    (start != *cursor)
        .then(|| {
            std::str::from_utf8(&bytes[start..*cursor])
                .ok()?
                .parse()
                .ok()
        })
        .flatten()
}

pub async fn fetch_model_profile(binary: &Path, model_id: &str) -> Result<Option<ModelProfile>> {
    if model_id.is_empty()
        || model_id.contains("..")
        || model_id.starts_with('/')
        || model_id.starts_with('\\')
    {
        return Err(anyhow!("Invalid model identifier"));
    }

    let version_trusted = match cached_version(binary).await {
        Ok(Some((_, minor))) => minor >= MIN_TRUSTED_MINOR,
        Ok(None) => false,
        Err(_) => false,
    };

    let output = run_query(binary, &["info", model_id], INFO_TIMEOUT, MAX_OUTPUT_BYTES).await?;
    if !output.stdout.is_empty() && output.stdout.contains("Error:") {
        return Ok(None);
    }
    if output.stdout.trim().is_empty() {
        return Ok(None);
    }

    parse_model_profile(&output.stdout, version_trusted, model_id)
}

pub async fn fetch_model_list(binary: &Path) -> Result<Vec<ModelListEntry>> {
    let output = run_query(binary, &["models"], MODELS_TIMEOUT, MAX_OUTPUT_BYTES).await?;
    parse_model_list(&output.stdout)
}

/// Parsed `rapid-mlx models --json`: alias → `(hf_repo, size_bytes)`. An alias whose
/// first catalog row has no `hf_path` maps to `None` (known, but not resolvable).
pub(crate) type AliasCatalog = HashMap<String, Option<(String, Option<u64>)>>;

/// How long a successfully parsed catalog is reused before the subprocess runs again.
pub(crate) const ALIAS_CATALOG_TTL: Duration = Duration::from_secs(60);
/// How long a failed catalog query is remembered, so a crash-looping or hung binary is
/// not respawned by every request.
pub(crate) const ALIAS_CATALOG_ERROR_TTL: Duration = Duration::from_secs(5);

enum CatalogLoad {
    Catalog(Arc<AliasCatalog>),
    /// The loader's error text; may name the binary path, so it is for server logs only.
    Failed(String),
}

struct CatalogSlot {
    binary: PathBuf,
    stored_at: Instant,
    load: CatalogLoad,
}

/// Most distinct binaries remembered at once. Production has one or two (managed and
/// PATH); the bound only stops the list growing without limit, and is generous so
/// parallel tests using many throwaway binaries do not evict each other.
const MAX_CATALOG_SLOTS: usize = 64;

/// Server-side cache of the alias catalog with single-flight loading: the lock is
/// held across the load, so concurrent callers wait for one subprocess and then read
/// its result. One slot per binary path so a runtime switch never reuses a stale
/// catalog. The clock and the loader are injected so the policy is testable without
/// a subprocess.
pub(crate) struct AliasCatalogCache {
    slots: tokio::sync::Mutex<Vec<CatalogSlot>>,
}

impl AliasCatalogCache {
    pub(crate) const fn new() -> Self {
        Self {
            slots: tokio::sync::Mutex::const_new(Vec::new()),
        }
    }

    pub(crate) async fn lookup<N, F, Fut>(
        &self,
        binary: &Path,
        alias: &str,
        now: N,
        load: F,
    ) -> Result<Option<(String, Option<u64>)>>
    where
        N: Fn() -> Instant,
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<AliasCatalog>>,
    {
        let mut slots = self.slots.lock().await;
        let position = slots.iter().position(|cached| cached.binary == binary);
        let fresh = position.is_some_and(|index| {
            let cached = &slots[index];
            let ttl = match cached.load {
                CatalogLoad::Catalog(_) => ALIAS_CATALOG_TTL,
                CatalogLoad::Failed(_) => ALIAS_CATALOG_ERROR_TTL,
            };
            now().saturating_duration_since(cached.stored_at) < ttl
        });
        let index = if fresh {
            position.unwrap_or_default()
        } else {
            let load = match load().await {
                Ok(catalog) => CatalogLoad::Catalog(Arc::new(catalog)),
                Err(error) => CatalogLoad::Failed(format!("{error:#}")),
            };
            let slot = CatalogSlot {
                binary: binary.to_path_buf(),
                stored_at: now(),
                load,
            };
            match position {
                Some(index) => {
                    slots[index] = slot;
                    index
                }
                None => {
                    if slots.len() >= MAX_CATALOG_SLOTS {
                        let oldest = slots
                            .iter()
                            .enumerate()
                            .min_by_key(|(_, cached)| cached.stored_at)
                            .map(|(index, _)| index)
                            .unwrap_or_default();
                        slots.swap_remove(oldest);
                    }
                    slots.push(slot);
                    slots.len() - 1
                }
            }
        };
        match &slots[index].load {
            CatalogLoad::Catalog(catalog) => Ok(catalog.get(alias).cloned().flatten()),
            CatalogLoad::Failed(message) => Err(anyhow!("{message}")),
        }
    }
}

static ALIAS_CATALOG: AliasCatalogCache = AliasCatalogCache::new();

/// Resolve a catalog alias to its `(hf_repo, size_bytes)` via the stable
/// `rapid-mlx models --json` output. `None` when the alias is not in the catalog.
/// The parsed catalog is cached for [`ALIAS_CATALOG_TTL`] and shared by concurrent callers.
pub async fn resolve_alias_repo(
    binary: &Path,
    alias: &str,
) -> Result<Option<(String, Option<u64>)>> {
    ALIAS_CATALOG
        .lookup(binary, alias, Instant::now, || fetch_alias_catalog(binary))
        .await
}

async fn fetch_alias_catalog(binary: &Path) -> Result<AliasCatalog> {
    let output = run_query(
        binary,
        &["models", "--json"],
        Duration::from_secs(15),
        4 * 1024 * 1024,
    )
    .await?;
    parse_alias_catalog(&output.stdout)
}

/// First row per alias wins (groups in key order, rows in listed order).
fn parse_alias_catalog(output: &str) -> Result<AliasCatalog> {
    let value: serde_json::Value =
        serde_json::from_str(output).context("rapid-mlx models --json was not valid JSON")?;
    let mut catalog = AliasCatalog::new();
    let Some(groups) = value.as_object() else {
        return Ok(catalog);
    };
    for entries in groups.values().filter_map(|v| v.as_array()) {
        for entry in entries {
            let Some(alias) = entry.get("alias").and_then(|v| v.as_str()) else {
                continue;
            };
            let repo = entry.get("hf_path").and_then(|v| v.as_str());
            let size = entry.get("size_bytes").and_then(|v| v.as_u64());
            catalog
                .entry(alias.to_string())
                .or_insert_with(|| repo.map(|r| (r.to_string(), size)));
        }
    }
    Ok(catalog)
}

/// One entry of `rapid-mlx recipe`: upstream's tier recommendation for this
/// exact machine ("Smart"/"Fast" picks), pinned atop the catalog picker.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RecipeRecommendation {
    /// 1-based rank as printed by upstream.
    pub rank: u32,
    /// Upstream's tier label ("Smart", "Fast", …).
    pub label: String,
    /// The model alias to serve.
    pub name: String,
    /// True when the model is already in the local cache.
    pub cached: bool,
    /// Advertised footprint/speed line ("20.0 GB RAM · 92% capability · ~41 tok/s").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub specs: Option<String>,
}

pub async fn fetch_recipe(binary: &Path) -> Result<Vec<RecipeRecommendation>> {
    let output = run_query(binary, &["recipe"], MODELS_TIMEOUT, MAX_OUTPUT_BYTES).await?;
    Ok(parse_recipe_output(&output.stdout))
}

fn parse_recipe_output(output: &str) -> Vec<RecipeRecommendation> {
    let mut recommendations = Vec::new();
    let lines: Vec<&str> = output.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        // Rank line: "1. Smart — qwen3.8-27b-4bit · cached"
        let Some((rank_part, rest)) = trimmed.split_once(". ") else {
            continue;
        };
        let Ok(rank) = rank_part.parse::<u32>() else {
            continue;
        };
        let Some((label, name_part)) = rest.split_once(" — ") else {
            continue;
        };
        let mut segments = name_part.split(" · ");
        let name = segments.next().unwrap_or("").trim().to_string();
        if name.is_empty() {
            continue;
        }
        let cached = segments.any(|seg| seg.trim() == "cached");
        // The specs line follows ("20.0 GB RAM · 92% capability · ~41 tok/s").
        let specs = lines
            .get(i + 1)
            .map(|l| l.trim().to_string())
            .filter(|l| l.contains("·"));
        recommendations.push(RecipeRecommendation {
            rank,
            label: label.trim().to_string(),
            name,
            cached,
            specs,
        });
    }
    recommendations
}

async fn run_query(
    binary: &Path,
    args: &[&str],
    timeout: Duration,
    max_bytes: usize,
) -> Result<QueryOutput> {
    let mut cmd = Command::new(binary);
    cmd.args(args)
        .kill_on_drop(true)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd
        .spawn()
        .with_context(|| format!("Failed to execute rapid-mlx query: {}", binary.display()))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("Failed to capture rapid-mlx query stdout"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("Failed to capture rapid-mlx query stderr"))?;

    let capture = async move {
        let (stdout_bytes, stderr_bytes, status) = tokio::try_join!(
            read_bounded(stdout, max_bytes),
            read_bounded(stderr, max_bytes),
            async { child.wait().await.map_err(Into::<anyhow::Error>::into) }
        )?;
        if !status.success() {
            let stderr_text = String::from_utf8_lossy(&stderr_bytes).trim().to_string();
            if stderr_text.contains("Error: model not found")
                || stderr_text.contains("unknown model")
                || stderr_text.contains("unrecognized model")
            {
                return Ok(QueryOutput {
                    stdout: String::new(),
                });
            }
            return Err(anyhow!(
                "rapid-mlx query failed: {}",
                stderr_text.chars().take(500).collect::<String>()
            ));
        }
        Ok(QueryOutput {
            stdout: String::from_utf8_lossy(&stdout_bytes).into_owned(),
        })
    };

    tokio::time::timeout(timeout, capture)
        .await
        .with_context(|| "rapid-mlx query timed out")?
}

async fn read_bounded<R>(reader: R, max_bytes: usize) -> Result<Vec<u8>>
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut bytes = Vec::with_capacity(max_bytes.min(8192));
    reader
        .take((max_bytes + 1) as u64)
        .read_to_end(&mut bytes)
        .await?;
    if bytes.len() > max_bytes {
        anyhow::bail!("rapid-mlx query output exceeded {} byte limit", max_bytes);
    }
    Ok(bytes)
}

fn parse_model_profile(
    output: &str,
    version_trusted: bool,
    model_id: &str,
) -> Result<Option<ModelProfile>> {
    let mut profile = ModelProfile::default();
    let lines: Vec<&str> = output.lines().collect();

    let mut current_section = "";
    let mut eligibility: Option<Eligibility> = None;
    let mut detected_hf_repo: Option<String> = None;

    for line in &lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if trimmed.contains("## DFlash eligibility")
            || trimmed.contains("## DFlash Eligibility")
            || trimmed.contains("DFlash eligibility")
        {
            flush_eligibility_into(&mut profile, current_section, eligibility.take());
            eligibility = Some(Eligibility::default());
            current_section = "dflash";
            if trimmed.contains("✓")
                && (trimmed.contains("eligible") || trimmed.contains("supported"))
            {
                profile.dflash_eligibility.supported = Some(true);
            } else if trimmed.contains("✗")
                || trimmed.contains("ineligible")
                || trimmed.contains("not supported")
            {
                profile.dflash_eligibility.supported = Some(false);
            }
            continue;
        }
        if trimmed.contains("## DDTree eligibility")
            || trimmed.contains("## DDTree Eligibility")
            || trimmed.contains("DDTree eligibility")
        {
            flush_eligibility_into(&mut profile, current_section, eligibility.take());
            eligibility = Some(Eligibility::default());
            current_section = "ddtree";
            if trimmed.contains("✓")
                && (trimmed.contains("eligible") || trimmed.contains("supported"))
            {
                profile.ddtree_eligibility.supported = Some(true);
            } else if trimmed.contains("✗")
                || trimmed.contains("ineligible")
                || trimmed.contains("not supported")
            {
                profile.ddtree_eligibility.supported = Some(false);
            }
            continue;
        }
        if (trimmed.starts_with("##") || trimmed.starts_with("│") && trimmed.contains("│"))
            && trimmed.contains(":")
            && let Some((key, value)) = extract_pair(trimmed)
        {
            if version_trusted {
                let key_lower = key.to_ascii_lowercase().replace([' ', '_', '-'], "");
                let value_lower = value.to_ascii_lowercase();

                match key_lower.as_str() {
                    "toolformat" | "tool" => {
                        profile.tool_format = info_value_or_absent(&value);
                    }
                    "reasoningparser" | "reasoning" => {
                        profile.reasoning_parser = info_value_or_absent(&value);
                    }
                    "architecture" | "arch" => {
                        profile.architecture = info_value_or_absent(&value);
                    }
                    "specdecode" | "speculative" => {
                        profile.spec_decode = parse_bool_value(&value_lower, '✓', '✗');
                    }
                    "mtp" | "mtpath" | "mtp-path" | "mtppath" => {
                        profile.mtp_path = match value_lower.as_str() {
                            v if v.contains("enabled") || v == "yes" || v == "✓" => {
                                Some(MtpPathStatus::Enabled)
                            }
                            v if v.contains("disabled") || v == "no" || v == "✗" => {
                                Some(MtpPathStatus::Disabled)
                            }
                            _ => Some(MtpPathStatus::Unknown),
                        };
                    }
                    "kvshare" | "kv-share" => {
                        profile.kv_share = Some(parse_yes_no(&value_lower));
                    }
                    "throttle" => {
                        profile.throttle = Some(parse_yes_no(&value_lower));
                    }
                    "suffixtier" | "suffix-tier" | "suffix" => {
                        profile.suffix_tier = Some(value.to_string());
                    }
                    _ => {}
                }

                // Every key:value line printed inside a DFlash/DDTree eligibility
                // box is a per-criterion reason (e.g. "Declared support",
                // "Not MoE", "Precision ≥8-bit", "Drafter declared",
                // "mlx-vlm 0.5.0+", "Spec tokens", "Tree budget",
                // "dtree-mlx runtime") — capture all of them rather than
                // matching a brittle, incomplete keyword allowlist.
                if !current_section.is_empty()
                    && let Some(ref mut elig) = eligibility
                {
                    let reason_key = key.to_string();
                    let reason_val = Some(value.to_string());
                    elig.reasons.insert(reason_key, reason_val);
                    if value_lower.contains("supported") && !value_lower.contains("not supported") {
                        elig.supported = Some(true);
                    } else if value_lower.contains("not supported") || value_lower.contains("✗") {
                        elig.supported = Some(false);
                    }
                }

                if !current_section.is_empty()
                    && (key_lower.contains("supported") || key_lower.contains("eligible"))
                    && let Some(ref mut elig) = eligibility
                {
                    elig.supported = Some(parse_yes_no(&value_lower));
                }
            }

            if trimmed.contains("Supported") && (trimmed.contains("✓") || trimmed.contains("Yes"))
            {
                if current_section == "dflash" {
                    profile.dflash_eligibility.supported = Some(true);
                }
                if current_section == "ddtree" {
                    profile.ddtree_eligibility.supported = Some(true);
                }
            }
            if trimmed.contains("Not supported")
                || (trimmed.contains("Supported") && trimmed.contains("✗"))
            {
                if current_section == "dflash" {
                    profile.dflash_eligibility.supported = Some(false);
                }
                if current_section == "ddtree" {
                    profile.ddtree_eligibility.supported = Some(false);
                }
            }
        }

        let line_lower = line.to_ascii_lowercase();
        if line_lower.contains("vision") && line_lower.contains("tower") {
            profile.extras.has_vision_tower = true;
            profile.extras.vision = true;
        }
        if line_lower.contains("vision") && (trimmed.contains("✓") || trimmed.contains("yes")) {
            profile.extras.vision = true;
        }
        if line_lower.contains("embeddings") {
            profile.extras.embeddings = true;
        }
        if line_lower.contains("mtp-dflash") || line_lower.contains("mtp_dflash") {
            profile.extras.mtp_dflash = true;
        }

        if trimmed.starts_with("##") && !(trimmed.contains("DFlash") || trimmed.contains("DDTree"))
        {
            flush_eligibility_into(&mut profile, current_section, eligibility.take());
            current_section = "";
        }

        let unboxed = trimmed.trim_start_matches('│').trim_end_matches('│').trim();
        if (unboxed.starts_with("Model:") || unboxed.starts_with("Name:"))
            && let Some(id_part) = unboxed.split_once(':').map(|(_, rest)| rest.trim())
            && id_part.contains('/')
            && id_part.split('/').count() == 2
        {
            detected_hf_repo = Some(id_part.to_string());
        }
    }

    flush_eligibility_into(&mut profile, current_section, eligibility.take());

    if let Some(ref repo) = detected_hf_repo {
        profile.is_finetune = repo == model_id;
    }

    if !profile.extras.vision {
        profile.extras.vision = vision_keywords_match(model_id);
    }
    if !profile.extras.vision
        && let Some(ref repo) = detected_hf_repo
    {
        profile.extras.vision = vision_keywords_match(repo);
    }
    if profile.extras.vision && !profile.extras.has_vision_tower {
        profile.extras.has_vision_tower = true;
    }

    Ok(Some(profile))
}

/// Merge an in-progress `Eligibility` accumulator into the profile field for
/// `section` ("dflash"/"ddtree"). Must be called whenever the parser is
/// about to leave a section (a new eligibility header, an unrelated `##`
/// header, or end of input) — real `rapid-mlx info` output always prints
/// both the DFlash and DDTree boxes together, so without this flush the
/// first section's accumulated per-criterion `reasons` are silently
/// discarded when the second section's header resets the accumulator.
/// Only overwrites `supported` when the accumulator captured a definite
/// value, preserving whatever the section's header line already implied.
fn flush_eligibility_into(profile: &mut ModelProfile, section: &str, elig: Option<Eligibility>) {
    let Some(elig) = elig else { return };
    let target = match section {
        "dflash" => &mut profile.dflash_eligibility,
        "ddtree" => &mut profile.ddtree_eligibility,
        _ => return,
    };
    if elig.supported.is_some() {
        target.supported = elig.supported;
    }
    target.reasons.extend(elig.reasons);
}

/// Values `rapid-mlx info` prints to mean "not detected".
///
/// `info` renders an absent field as the literal `(none)`, and it only resolves
/// tool/reasoning parsers for HF repo aliases at all — a local model directory
/// reports `(none)` for both even when the family has parsers. Storing that as
/// the field value made the app claim the model "declares tool format '(none)'"
/// in its preflight warnings, and made `has_reasoning_parser` true for a model
/// with none. The same string reached the benchmark suite's argv, where
/// `--reasoning-parser '(none)'` is an argparse error that killed the server
/// before it could become healthy.
const INFO_ABSENT_VALUES: &[&str] = &["(none)", "none", "n/a", "-", "unknown", "unset"];

fn info_value_or_absent(value: &str) -> Option<String> {
    let normalized = value.trim().to_ascii_lowercase();
    if normalized.is_empty() || INFO_ABSENT_VALUES.contains(&normalized.as_str()) {
        return None;
    }
    Some(value.to_string())
}

fn extract_pair(line: &str) -> Option<(String, String)> {
    let clean = line.trim_start_matches('│').trim_end_matches('│').trim();
    let colon_pos = clean.find(':')?;
    let key = clean[..colon_pos].trim().to_string();
    let value = clean[colon_pos + 1..].trim().to_string();
    if key.is_empty() || value.is_empty() {
        return None;
    }
    Some((key, value))
}

fn parse_bool_value(value: &str, yes_char: char, no_char: char) -> SpecDecodeSupport {
    if value.contains(yes_char) || value == "yes" || value == "supported" {
        SpecDecodeSupport::Supported
    } else if value.contains(no_char) || value == "no" || value == "unsupported" {
        SpecDecodeSupport::Unsupported
    } else {
        SpecDecodeSupport::Unknown
    }
}

fn parse_yes_no(value: &str) -> bool {
    value.contains("yes") || value.contains("✓") || value == "enabled" || value == "true"
}

fn vision_keywords_match(id: &str) -> bool {
    let lower = id.to_ascii_lowercase();
    lower.contains("vl")
        || lower.contains("vision")
        || lower.contains("multimodal")
        || lower.contains("mllm")
        || lower.contains("vlm")
}

fn parse_model_list(output: &str) -> Result<Vec<ModelListEntry>> {
    let mut entries = Vec::new();
    for line in output.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with('│') {
            continue;
        }
        if let Some(mut entry) = parse_list_line(trimmed) {
            // Catalog rows always carry the measured size column; header
            // prose ("Available models (198…)") and separator rules don't.
            if entry.size_bytes.is_none() {
                continue;
            }
            if entry.display_name.is_empty() {
                entry.display_name = entry.name.clone();
            }
            entries.push(entry);
        }
    }
    Ok(entries)
}

fn parse_list_line(line: &str) -> Option<ModelListEntry> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.is_empty() {
        return None;
    }
    let name = parts[0].to_string();
    let mut idx = 1usize;
    let mut size_bytes = None;
    // Optional "N.N GiB" column directly after the name.
    if idx + 1 < parts.len() && parts[idx + 1] == "GiB" {
        size_bytes = parts[idx]
            .parse::<f64>()
            .ok()
            .map(|gib| (gib * 1024.0 * 1024.0 * 1024.0) as u64);
        idx += 2;
    }
    // The parser and template columns always occupy a slot (a `—` placeholder
    // means "absent"), so consume the column even when the value is absent.
    let dash = |p: &str| p == "—" || p == "-";
    let parser = parts.get(idx).filter(|p| !dash(p)).map(|p| p.to_string());
    if parts.get(idx).is_some() {
        idx += 1;
    }
    let template = parts.get(idx).filter(|p| !dash(p)).map(|p| p.to_string());
    if parts.get(idx).is_some() {
        idx += 1;
    }
    // MTP column: "✓ MTP" with an optional trailing sidecar id, or "✗ hybrid".
    let mut mtp = false;
    let mut hybrid = false;
    let mut mtp_sidecar = None;
    if let Some(marker) = parts.get(idx) {
        if *marker == "✓" {
            mtp = parts.get(idx + 1).is_some_and(|p| p.contains("MTP"));
            if mtp {
                idx += 2;
                // Sidecar id ("MTP@repo@revision") trails the n/a/tier columns,
                // so scan a bounded window rather than assuming one offset.
                mtp_sidecar = parts.get(idx..).and_then(|rest| {
                    rest.iter()
                        .take(6)
                        .find(|sc| **sc != "n/a" && (sc.contains('@') || sc.contains('/')))
                        .map(|sc| sc.trim_start_matches("MTP@").to_string())
                });
            }
        } else if *marker == "✗" {
            hybrid = parts.get(idx + 1).is_some_and(|p| p.contains("hybrid"));
        }
    }
    Some(ModelListEntry {
        name,
        display_name: String::new(),
        size_bytes,
        parser,
        template,
        hybrid,
        mtp,
        mtp_sidecar,
    })
}

// ── Local MLX Introspection (Phase 8A3) ───────────────────────────────────────────────────

/// Resolve the recursive disk size for an MLX model directory.
pub fn resolve_mlx_recursive_size(model_path: &Path) -> Result<u64> {
    let mut total = 0u64;
    fn walk_dir(dir: &Path, acc: &mut u64) -> std::io::Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                walk_dir(&entry.path(), acc)?;
            } else {
                *acc += entry.metadata()?.len();
            }
        }
        Ok(())
    }
    walk_dir(model_path, &mut total).context("walk model path")?;
    Ok(total)
}

/// Read local config.json from an MLX model directory.
pub fn read_mlx_local_config(
    model_path: &Path,
) -> Result<Option<crate::inference::rapid_mlx::mlx_meta::MlxConfig>> {
    let config_path = model_path.join("config.json");
    if !config_path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&config_path).context("read config.json")?;
    let config: crate::inference::rapid_mlx::mlx_meta::MlxConfig =
        serde_json::from_str(&text).context("parse config.json")?;
    Ok(Some(config))
}

/// Check whether an MLX model directory has an index.json with mmproj-like vision adapter files listed.
///
/// Rapid-MLX has no fake mmproj equivalent; only show real integrated/qualified
/// MLX-VLM components. This checks the safetensors index for actual vision tower
/// projector tensors (mmproj, vision_proj, modality_projection, etc.) rather than
/// a naive string match.
pub fn has_mmproj_in_index(model_path: &Path) -> Result<bool> {
    let index_path = model_path.join("model.safetensors.index.json");
    if !index_path.exists() {
        return Ok(false);
    }
    let text = std::fs::read_to_string(&index_path).context("read model.safetensors.index.json")?;
    has_mmproj_in_index_bytes(text.as_bytes())
}

/// Same check as `has_mmproj_in_index`, against already-fetched index bytes — shared by the
/// local-path and remote (revision-aware) introspection paths so the tensor-name heuristic
/// cannot drift between them.
pub fn has_mmproj_in_index_bytes(bytes: &[u8]) -> Result<bool> {
    let index: serde_json::Value =
        serde_json::from_slice(bytes).context("parse model.safetensors.index.json")?;
    let weight_map = index.get("weight_map").and_then(|v| v.as_object());
    if let Some(map) = weight_map {
        for tensor_name in map.keys() {
            let lower = tensor_name.to_ascii_lowercase();
            if lower.contains("mmproj")
                || lower.contains("vision_proj")
                || lower.contains("modality_projection")
                || lower.contains("vision.tower")
            {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::FutureExt;

    #[test]
    fn parses_version_numbers_from_various_formats() {
        assert_eq!(
            parse_version_number("rapid-mlx 0.10.9"),
            Some(("0.10.9".into(), 10))
        );
        assert_eq!(
            parse_version_number("Rapid-MLX version v0.11.2\n"),
            Some(("0.11.2".into(), 11))
        );
        assert_eq!(parse_version_number("development"), None);
    }

    #[test]
    fn model_profile_serde_default_is_safe() {
        let json = r#"{}"#;
        let profile: ModelProfile = serde_json::from_str(json).unwrap();
        assert!(profile.tool_format.is_none());
        assert!(!profile.extras.vision);
    }

    #[test]
    fn eligibility_is_eligible_methods_work() {
        let yes = Eligibility {
            supported: Some(true),
            reasons: BTreeMap::new(),
        };
        let no = Eligibility {
            supported: Some(false),
            reasons: BTreeMap::new(),
        };
        let unknown = Eligibility {
            supported: None,
            reasons: BTreeMap::new(),
        };
        assert!(yes.is_eligible());
        assert!(!yes.is_ineligible());
        assert!(no.is_ineligible());
        assert!(!no.is_eligible());
        assert!(!unknown.is_eligible());
        assert!(!unknown.is_ineligible());
    }

    #[test]
    fn model_id_validation_rejects_paths() {
        assert!(
            fetch_model_profile(Path::new("rapid-mlx"), "../etc/passwd")
                .now_or_never()
                .is_some_and(|r| r.is_err())
        );
        assert!(
            fetch_model_profile(Path::new("rapid-mlx"), "/etc/passwd")
                .now_or_never()
                .is_some_and(|r| r.is_err())
        );
    }

    #[test]
    fn spec_decode_support_serializes_to_snake_case() {
        let supported = SpecDecodeSupport::Supported;
        let json = serde_json::to_string(&supported).unwrap();
        assert_eq!(json, r#""supported""#);
    }

    #[test]
    fn extra_capabilities_is_empty_by_default() {
        let caps = ExtraCapabilities::default();
        assert!(!caps.vision);
        assert!(!caps.has_vision_tower);
        assert!(!caps.embeddings);
        assert!(!caps.mtp_dflash);
    }

    #[tokio::test]
    #[ignore = "requires rapid-mlx CLI installed; run manually to verify parser against real output"]
    async fn parses_real_rapid_mlx_info_output_contract() {
        let output = std::process::Command::new("rapid-mlx")
            .args(["info", "qwen3-0.6b-4bit"])
            .output()
            .expect("rapid-mlx CLI must be installed for this contract test")
            .stdout;
        let text = String::from_utf8_lossy(&output);

        let profile = parse_model_profile(&text, true, "qwen3-0.6b-4bit")
            .unwrap()
            .expect("should parse");

        assert_eq!(profile.tool_format, Some("hermes".into()));
        assert_eq!(profile.reasoning_parser, Some("qwen3".into()));
        assert_eq!(profile.architecture, Some("pure attention".into()));
        assert_eq!(profile.spec_decode, SpecDecodeSupport::Supported);
        assert_eq!(profile.mtp_path, Some(MtpPathStatus::Disabled));
        assert_eq!(profile.kv_share, Some(false));
        assert_eq!(profile.throttle, Some(false));
        assert_eq!(profile.dflash_eligibility.supported, Some(false));
        assert_eq!(profile.ddtree_eligibility.supported, Some(false));
        assert!(!profile.extras.vision);
        assert!(!profile.extras.embeddings);
        assert!(!profile.is_finetune);
    }

    /// Real `rapid-mlx info mlx-community/Qwen3-0.6B-4bit` output (rapid-mlx
    /// 0.10.x, captured 2026-07-18), box-drawn `│`-bordered header as
    /// actually printed by the CLI. The finetune-detection regex must strip
    /// the `│` border before matching `"Model:"` — a border-less fixture
    /// (as this test previously used) doesn't exercise that bug.
    const REAL_INFO_HEADER: &str = "┌──────────────────────────────────────────────────────────────┐\n\
│ Model: mlx-community/Qwen3-0.6B-4bit                          │\n\
│ ──────────────────────────────────────────────────────────── │\n\
│ Tool format      : hermes                                     │\n\
│ Reasoning parser : qwen3                                      │\n\
│ Architecture     : pure attention                             │\n\
│ Spec decode      : ✓ supported                                │\n\
│ MTP path         : disabled                                   │\n\
│ KV-share         : no                                         │\n\
│ Throttle         : ✗ not needed                                │\n\
└──────────────────────────────────────────────────────────────┘";

    #[test]
    fn finetune_detection_marks_unknown_hf_repos() {
        // Querying by the exact HF repo id (as one would for an unrecognized
        // finetune) means the printed `│ Model: <repo> │` line equals the
        // query id, so it's flagged as a finetune/unregistered repo.
        let profile = parse_model_profile(REAL_INFO_HEADER, true, "mlx-community/Qwen3-0.6B-4bit")
            .unwrap()
            .unwrap();
        assert!(profile.is_finetune);

        // Querying by a known alias means the query id differs from the
        // resolved `Model:` line, so it's not treated as a finetune.
        let profile2 = parse_model_profile(REAL_INFO_HEADER, true, "qwen3-0.6b-4bit")
            .unwrap()
            .unwrap();
        assert!(!profile2.is_finetune);
    }

    /// Real `rapid-mlx info qwen3-0.6b-4bit` DFlash/DDTree eligibility
    /// blocks (rapid-mlx 0.10.x, captured 2026-07-18). These include
    /// criterion labels the old keyword allowlist ("Declared" | "MoE" |
    /// "Precision" | "Drafter" | "Runtime" | "Supported") dropped:
    /// `mlx-vlm 0.5.0+`, `Spec tokens`, `Tree budget`, and the
    /// lowercase-`runtime` `dtree-mlx runtime`.
    const REAL_ELIGIBILITY_BLOCKS: &str = "┌──────────────────────────────────────────────────────────────┐\n\
│ DFlash eligibility: ✗ ineligible                              │\n\
│ ──────────────────────────────────────────────────────────── │\n\
│ Declared support  : ✗ no                                      │\n\
│ Not MoE           : ✓ yes (dense)                              │\n\
│ Precision ≥8-bit  : ✗ no (4-bit/mxfp4/nvfp4)                   │\n\
│ Drafter declared  : ✗ no (dflash_draft_model unset)            │\n\
│ mlx-vlm 0.5.0+    : ✗ missing (need rapid-mlx[dflash])         │\n\
└──────────────────────────────────────────────────────────────┘\n\
\n\
┌──────────────────────────────────────────────────────────────┐\n\
│ DDTree eligibility: ✗ ineligible                              │\n\
│ ──────────────────────────────────────────────────────────── │\n\
│ Declared support  : ✗ no                                      │\n\
│ Not MoE           : ✓ yes (dense)                              │\n\
│ Precision ≥8-bit  : ✗ no (4-bit/mxfp4/nvfp4)                   │\n\
│ Drafter declared  : ✗ no (ddtree_draft_model unset)            │\n\
│ Spec tokens       : ✗ missing                                 │\n\
│ Tree budget       : ✗ missing                                 │\n\
│ dtree-mlx runtime : ✗ missing/import-broken                   │\n\
└──────────────────────────────────────────────────────────────┘";

    #[test]
    fn eligibility_reasons_capture_all_real_criterion_lines_not_just_keyword_subset() {
        let profile = parse_model_profile(REAL_ELIGIBILITY_BLOCKS, true, "qwen3-0.6b-4bit")
            .unwrap()
            .unwrap();

        assert_eq!(profile.dflash_eligibility.supported, Some(false));
        for key in [
            "Declared support",
            "Not MoE",
            "Precision ≥8-bit",
            "Drafter declared",
        ] {
            assert!(
                profile.dflash_eligibility.reasons.contains_key(key),
                "expected DFlash reasons to contain {key:?}, got {:?}",
                profile.dflash_eligibility.reasons
            );
        }
        // Previously dropped by the keyword allowlist:
        assert!(
            profile
                .dflash_eligibility
                .reasons
                .contains_key("mlx-vlm 0.5.0+"),
            "expected DFlash reasons to contain 'mlx-vlm 0.5.0+', got {:?}",
            profile.dflash_eligibility.reasons
        );

        assert_eq!(profile.ddtree_eligibility.supported, Some(false));
        for key in [
            "Declared support",
            "Not MoE",
            "Precision ≥8-bit",
            "Drafter declared",
        ] {
            assert!(
                profile.ddtree_eligibility.reasons.contains_key(key),
                "expected DDTree reasons to contain {key:?}, got {:?}",
                profile.ddtree_eligibility.reasons
            );
        }
        // Previously dropped by the keyword allowlist:
        for key in ["Spec tokens", "Tree budget", "dtree-mlx runtime"] {
            assert!(
                profile.ddtree_eligibility.reasons.contains_key(key),
                "expected DDTree reasons to contain {key:?}, got {:?}",
                profile.ddtree_eligibility.reasons
            );
        }
    }

    #[test]
    fn untrusted_version_returns_minimal_profile() {
        let output = r#"┌──────────────────────────────────────────────────────────────┐
│ Model: mlx-community/Qwen3-0.6B-4bit                         │
│ ──────────────────────────────────────────────────────────── │
│ Tool format      : hermes                                    │
│ Reasoning parser : qwen3                                     │
│ Architecture     : pure attention                            │
│ Spec decode      : ✓ supported                               │
└──────────────────────────────────────────────────────────────┘"#;

        let profile = parse_model_profile(output, false, "qwen3-0.6b-4bit")
            .unwrap()
            .unwrap();
        assert!(profile.tool_format.is_none());
        assert!(profile.reasoning_parser.is_none());
        assert!(profile.architecture.is_none());
        assert_eq!(profile.spec_decode, SpecDecodeSupport::Unknown);
    }

    /// Verbatim shape of `rapid-mlx info <local-dir>` on 0.11.1: `info` resolves
    /// parsers only for HF repo aliases, so a local model directory reports the
    /// placeholder for every one of them. Storing `"(none)"` made the app warn
    /// that the model "declares tool format '(none)'", and put
    /// `--reasoning-parser '(none)'` on the benchmark suite's argv, which is an
    /// argparse error that kills the server before it becomes healthy.
    #[test]
    fn info_none_placeholder_is_absence_not_a_parser_name() {
        let output = r#"┌──────────────────────────────────────────────────────────────┐
│ Model: /Users/nick/mlx-models/nightmedia-27b-mxfp8-mlx        │
│ ──────────────────────────────────────────────────────────── │
│ Tool format      : (none)                                    │
│ Reasoning parser : (none)                                    │
│ Architecture     : pure attention                            │
└──────────────────────────────────────────────────────────────┘"#;

        let profile = parse_model_profile(output, true, "local-dir")
            .unwrap()
            .unwrap();
        assert!(profile.tool_format.is_none());
        assert!(profile.reasoning_parser.is_none());
        // A real value alongside the placeholders still parses.
        assert_eq!(profile.architecture, Some("pure attention".into()));
    }

    #[test]
    fn vision_detected_by_model_id_keyword() {
        let output = r#"┌──────────────────────────────────────────────────────────────┐
│ Model: mlx-community/Qwen3-VL-30B-4bit                       │
│ ──────────────────────────────────────────────────────────── │
│ Tool format      : hermes                                    │
└──────────────────────────────────────────────────────────────┘"#;

        for model_id in &[
            "qwen3-vl-30b-4bit",
            "Qwen2.5-VL-3B",
            "phi-3-vision",
            "llava-multimodal",
            "mllm-7b",
            "paligemma-vlm-13b",
        ] {
            let profile = parse_model_profile(output, true, model_id)
                .unwrap()
                .unwrap();
            assert!(
                profile.extras.vision,
                "vision should be true for {}",
                model_id
            );
            assert!(
                profile.extras.has_vision_tower,
                "has_vision_tower should be true for {}",
                model_id
            );
        }
    }

    #[test]
    fn vision_detected_by_hf_repo_keyword() {
        let output = r#"┌──────────────────────────────────────────────────────────────┐
│ Model: mlx-community/Qwen2.5-VL-7B-4bit                      │
└──────────────────────────────────────────────────────────────┘"#;

        let profile = parse_model_profile(output, true, "some-alias-without-vl")
            .unwrap()
            .unwrap();
        assert!(
            profile.extras.vision,
            "should detect vision from HF repo path"
        );
        assert!(
            profile.extras.has_vision_tower,
            "should set has_vision_tower from HF repo path"
        );
    }

    /// Real `rapid-mlx models` table rows (0.15.x): size, parser, template,
    /// MTP marker with sidecar, and the hybrid marker for non-MTP models.
    #[test]
    fn model_list_rows_capture_catalog_columns() {
        let rows = "  qwen3.8-27b-4bit                  15.2 GiB   qwen3_coder_xml  qwen3               \u{2713} MTP      n/a         exp     exp     MTP@rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX@3\n  qwen3.8-27b-tensorfold            15.0 GiB   \u{2014}                qwen3               \u{2717} hybrid   n/a         verified exp     \u{2014}\n";
        let entries = parse_model_list(rows).expect("parse");
        assert_eq!(entries.len(), 2);

        let first = &entries[0];
        assert_eq!(first.name, "qwen3.8-27b-4bit");
        assert_eq!(
            first.size_bytes,
            Some((15.2f64 * 1024.0 * 1024.0 * 1024.0) as u64)
        );
        assert_eq!(first.parser.as_deref(), Some("qwen3_coder_xml"));
        assert_eq!(first.template.as_deref(), Some("qwen3"));
        assert!(first.mtp);
        assert!(!first.hybrid);
        assert_eq!(
            first.mtp_sidecar.as_deref(),
            Some("rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX@3")
        );

        let second = &entries[1];
        assert_eq!(second.name, "qwen3.8-27b-tensorfold");
        assert!(second.parser.is_none());
        assert_eq!(second.template.as_deref(), Some("qwen3"));
        assert!(!second.mtp);
        assert!(second.hybrid);
        assert!(second.mtp_sidecar.is_none());
    }

    #[test]
    fn recipe_output_parses_ranked_recommendations() {
        let output = "Recommended for this 64.0 GB Mac (64 GB tier)\n\n1. Smart \u{2014} qwen3.8-27b-4bit \u{b7} cached\n   20.0 GB RAM \u{b7} 92% capability \u{b7} ~41 tok/s\n   rapid-mlx serve qwen3.8-27b-4bit\n\n2. Fast \u{2014} qwen3.6-35b-4bit\n   20.0 GB RAM \u{b7} 87% capability \u{b7} ~60 tok/s\n   rapid-mlx serve qwen3.6-35b-4bit\n";
        let recs = parse_recipe_output(output);
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].rank, 1);
        assert_eq!(recs[0].label, "Smart");
        assert_eq!(recs[0].name, "qwen3.8-27b-4bit");
        assert!(recs[0].cached);
        assert!(
            recs[0]
                .specs
                .as_deref()
                .is_some_and(|s| s.contains("41 tok/s"))
        );
        assert_eq!(recs[1].name, "qwen3.6-35b-4bit");
        assert!(!recs[1].cached);
    }

    #[test]
    fn non_vision_models_not_false_positive() {
        let output = r#"┌──────────────────────────────────────────────────────────────┐
│ Model: mlx-community/Qwen3-0.6B-4bit                         │
└──────────────────────────────────────────────────────────────┘"#;

        for model_id in &[
            "qwen3-0.6b-4bit",
            "llama-3.1-8b",
            "gemma-2-9b",
            "mistral-nemo",
        ] {
            let profile = parse_model_profile(output, true, model_id)
                .unwrap()
                .unwrap();
            assert!(
                !profile.extras.vision,
                "vision should be false for {}",
                model_id
            );
        }
    }

    // ---- alias catalog cache ----

    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    fn catalog() -> AliasCatalog {
        parse_alias_catalog(
            r#"{"qwen":[{"alias":"known","hf_path":"o/m","size_bytes":42},
                         {"alias":"other","hf_path":"o/n"}]}"#,
        )
        .unwrap()
    }

    /// A manual clock: `now()` is `base + offset`, advanced by the test.
    struct FakeClock {
        base: std::time::Instant,
        offset_secs: AtomicU64,
    }

    impl FakeClock {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                base: std::time::Instant::now(),
                offset_secs: AtomicU64::new(0),
            })
        }
        fn advance(&self, secs: u64) {
            self.offset_secs.fetch_add(secs, Ordering::SeqCst);
        }
        fn reader(self: &Arc<Self>) -> impl Fn() -> std::time::Instant + use<> {
            let clock = self.clone();
            move || clock.base + Duration::from_secs(clock.offset_secs.load(Ordering::SeqCst))
        }
    }

    async fn lookup(
        cache: &AliasCatalogCache,
        binary: &str,
        alias: &str,
        clock: &Arc<FakeClock>,
        loads: &Arc<AtomicUsize>,
    ) -> Result<Option<(String, Option<u64>)>> {
        let loads = loads.clone();
        cache
            .lookup(Path::new(binary), alias, clock.reader(), || async move {
                loads.fetch_add(1, Ordering::SeqCst);
                Ok(catalog())
            })
            .await
    }

    #[tokio::test]
    async fn alias_catalog_parse_keeps_first_entry_and_size() {
        let parsed = parse_alias_catalog(
            r#"{"a":[{"alias":"x","hf_path":"o/first","size_bytes":7}],
                "b":[{"alias":"x","hf_path":"o/second"}]}"#,
        )
        .unwrap();
        assert_eq!(parsed["x"], Some(("o/first".to_string(), Some(7))));
        assert!(parse_alias_catalog("not json").is_err());
        assert!(parse_alias_catalog("[]").unwrap().is_empty());
    }

    #[tokio::test]
    async fn alias_catalog_cache_hit_runs_the_subprocess_once() {
        let cache = AliasCatalogCache::new();
        let clock = FakeClock::new();
        let loads = Arc::new(AtomicUsize::new(0));
        let first = lookup(&cache, "/bin/a", "known", &clock, &loads).await;
        assert_eq!(first.unwrap(), Some(("o/m".to_string(), Some(42))));
        // Same alias, a different alias, and an unknown alias are all served from cache.
        lookup(&cache, "/bin/a", "known", &clock, &loads)
            .await
            .unwrap();
        let other = lookup(&cache, "/bin/a", "other", &clock, &loads).await;
        assert_eq!(other.unwrap(), Some(("o/n".to_string(), None)));
        let missing = lookup(&cache, "/bin/a", "nope", &clock, &loads).await;
        assert_eq!(missing.unwrap(), None);
        assert_eq!(loads.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn alias_catalog_cache_expires_after_the_ttl() {
        let cache = AliasCatalogCache::new();
        let clock = FakeClock::new();
        let loads = Arc::new(AtomicUsize::new(0));
        lookup(&cache, "/bin/a", "known", &clock, &loads)
            .await
            .unwrap();
        clock.advance(ALIAS_CATALOG_TTL.as_secs() - 1);
        lookup(&cache, "/bin/a", "known", &clock, &loads)
            .await
            .unwrap();
        assert_eq!(
            loads.load(Ordering::SeqCst),
            1,
            "still fresh just before the TTL"
        );
        clock.advance(1);
        lookup(&cache, "/bin/a", "known", &clock, &loads)
            .await
            .unwrap();
        assert_eq!(loads.load(Ordering::SeqCst), 2, "reloaded at the TTL");
        lookup(&cache, "/bin/a", "known", &clock, &loads)
            .await
            .unwrap();
        assert_eq!(loads.load(Ordering::SeqCst), 2, "and cached again");
    }

    #[tokio::test]
    async fn alias_catalog_cache_is_keyed_by_binary() {
        let cache = AliasCatalogCache::new();
        let clock = FakeClock::new();
        let loads = Arc::new(AtomicUsize::new(0));
        lookup(&cache, "/bin/a", "known", &clock, &loads)
            .await
            .unwrap();
        lookup(&cache, "/bin/b", "known", &clock, &loads)
            .await
            .unwrap();
        assert_eq!(
            loads.load(Ordering::SeqCst),
            2,
            "a new runtime must not reuse the old catalog"
        );
        // Switching back must not reload: one binary never evicts another's catalog.
        lookup(&cache, "/bin/a", "known", &clock, &loads)
            .await
            .unwrap();
        assert_eq!(loads.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn alias_catalog_concurrent_callers_share_one_subprocess() {
        let cache = Arc::new(AliasCatalogCache::new());
        let clock = FakeClock::new();
        let loads = Arc::new(AtomicUsize::new(0));
        let calls = (0..16).map(|_| {
            let (cache, loads, reader) = (cache.clone(), loads.clone(), clock.reader());
            async move {
                cache
                    .lookup(Path::new("/bin/a"), "known", reader, || async move {
                        loads.fetch_add(1, Ordering::SeqCst);
                        // Hold the load open so every other caller arrives mid-flight.
                        for _ in 0..8 {
                            tokio::task::yield_now().await;
                        }
                        Ok(catalog())
                    })
                    .await
            }
        });
        for result in futures_util::future::join_all(calls).await {
            assert_eq!(result.unwrap(), Some(("o/m".to_string(), Some(42))));
        }
        assert_eq!(loads.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn alias_catalog_failures_are_briefly_cached_then_retried() {
        let cache = AliasCatalogCache::new();
        let clock = FakeClock::new();
        let loads = Arc::new(AtomicUsize::new(0));
        let failing = |loads: Arc<AtomicUsize>| async move {
            loads.fetch_add(1, Ordering::SeqCst);
            Err::<AliasCatalog, _>(anyhow!("boom at /private/path/rapid-mlx"))
        };
        let reader = clock.reader();
        let first = cache
            .lookup(Path::new("/bin/a"), "known", &reader, || {
                failing(loads.clone())
            })
            .await;
        assert!(first.is_err());
        // A crash-looping binary is not respawned on every request...
        let second = cache
            .lookup(Path::new("/bin/a"), "known", &reader, || {
                failing(loads.clone())
            })
            .await;
        assert!(second.is_err());
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        // ...but is retried once the short failure window passes.
        clock.advance(ALIAS_CATALOG_ERROR_TTL.as_secs() + 1);
        let recovered = lookup(&cache, "/bin/a", "known", &clock, &loads).await;
        assert_eq!(recovered.unwrap(), Some(("o/m".to_string(), Some(42))));
        assert_eq!(loads.load(Ordering::SeqCst), 2);
    }
}
