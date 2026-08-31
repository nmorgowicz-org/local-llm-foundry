#![allow(dead_code)]

use anyhow::{Context, Result, anyhow, bail};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;
use std::process::Stdio;
use std::sync::{Arc, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::io::AsyncReadExt;
use tokio::process::Command;

pub const CAPABILITY_PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(8);
pub const CAPABILITY_PROBE_MAX_OUTPUT: usize = 256 * 1024;

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CapabilitySnapshotSource {
    AutoProbed,
    ManualOverride,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct ExecutableIdentity {
    pub path: String,
    pub file_hash: String,
    pub file_mtime_unix: u64,
}

impl ExecutableIdentity {
    pub fn from_path(path: &Path) -> Result<Self> {
        let canonical = path
            .canonicalize()
            .context("Cannot canonicalize llama-server path")?;
        let meta = std::fs::metadata(&canonical)
            .context("Cannot read llama-server executable metadata")?;
        let mtime = meta
            .modified()
            .map(|t| t.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs())
            .unwrap_or(0);
        let hash = hash_file(&canonical)?;
        Ok(Self {
            path: canonical.to_string_lossy().into_owned(),
            file_hash: hash,
            file_mtime_unix: mtime,
        })
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum FeatureState {
    Available,
    Unavailable(String),
}

impl Default for FeatureState {
    fn default() -> Self {
        Self::Unavailable("Not verified".into())
    }
}

/// Parallel bounded capability snapshot for llama.cpp.
/// Bound to exact executable identity and help hash.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CapabilitySnapshot {
    pub executable_identity: ExecutableIdentity,
    pub version_text: String,
    pub help_hash: String,
    pub serve_flags: Vec<String>,
    /// Cache capabilities
    pub cache: CacheCapabilities,
    /// Context capabilities
    pub context: ContextCapabilities,
    /// Concurrency capabilities
    pub concurrency: ConcurrencyCapabilities,
    /// API endpoint capabilities
    pub endpoints: EndpointCapabilities,
    /// Streaming capabilities
    pub streaming: StreamingCapabilities,
    /// Template capabilities
    pub templates: TemplateCapabilities,
    /// Tool capabilities
    pub tools: ToolCapabilities,
    /// Speculative decoding capabilities
    pub speculation: SpeculationCapabilities,
    /// Typed evidence for Phase 2 llama.cpp-native options. Unlike a raw
    /// substring check, this records both forms and the accepted value set.
    #[serde(default)]
    pub typed: TypedLlamaCapabilities,
    /// Mixed main-K/V support gate (`q8_0` K + `q4_0` V fused-attention pair).
    /// Product default is `supported: false` until a trusted build manifest is integrated.
    pub mixed_main_kv: MixedMainKv,
    pub evidence_timestamp: u64,
    pub source: CapabilitySnapshotSource,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct EnumCapability {
    pub supported: FeatureState,
    #[serde(default)]
    pub accepted_values: Vec<String>,
    pub default_value: Option<String>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct BooleanFlagCapability {
    pub positive: FeatureState,
    pub negative: FeatureState,
    pub default_value: Option<bool>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct TypedLlamaCapabilities {
    pub mmproj_offload: BooleanFlagCapability,
    pub reasoning_effort: EnumCapability,
    pub reasoning_format: EnumCapability,
    pub reasoning_preserve: BooleanFlagCapability,
    /// Help advertises the flag, but does not prove template support.
    pub reasoning_preserve_template: FeatureState,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct CacheCapabilities {
    pub prompt_cache: FeatureState,
    pub ram_cache: FeatureState,
    pub idle_slot_cache: FeatureState,
    pub cache_reuse: FeatureState,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct ContextCapabilities {
    pub checkpoints: FeatureState,
    pub kv_unified: FeatureState,
    pub kv_partitioned: FeatureState,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct ConcurrencyCapabilities {
    pub auto_slots: FeatureState,
    pub explicit_slots: FeatureState,
    pub continuous_batching: FeatureState,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct EndpointCapabilities {
    pub chat_completions: FeatureState,
    pub responses: FeatureState,
    pub raw_completion: FeatureState,
    pub text_completion: FeatureState,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct StreamingCapabilities {
    pub usage_in_stream: FeatureState,
    pub progress_in_stream: FeatureState,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct TemplateCapabilities {
    pub jinja: FeatureState,
    pub chat_template_file: FeatureState,
    pub chat_template_kwargs: FeatureState,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct ToolCapabilities {
    pub tool_parsing: FeatureState,
    pub parallel_tools: FeatureState,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct SpeculationCapabilities {
    pub draft_model: FeatureState,
    pub ngram_spec: FeatureState,
}

/// Backend-owned mixed main-K/V support gate.
///
/// Only a trusted build manifest with exact SHA-256 binding may set `supported: true`.
/// The only shipped provider (Phase 1b) sets `supported: false` unconditionally.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct MixedMainKv {
    pub supported: bool,
    /// Human-readable reason; load-bearing per invariant 17 — rendered in UI alongside the option.
    pub reason: String,
    /// How the value was determined: "product_default_denied", "build_manifest", etc.
    pub source: String,
}

impl MixedMainKv {
    /// Canonical product-default construction: rejected because we have no build
    /// manifest binding to prove the fused-attention kernel property.
    pub fn product_default_denied() -> Self {
        Self {
            supported: false,
            reason: "Requires a llama.cpp build compiled with GGML_CUDA_FA_ALL_QUANTS advertising mixed main-K/V support through a product build manifest. Separate -ctk and -ctv value lists do not prove this kernel property.".into(),
            source: "product_default_denied".into(),
        }
    }
}

impl CapabilitySnapshot {
    /// Whether the exact binary advertised a flag in its bounded `--help`
    /// probe. This is deliberately the only generic flag lookup used by typed
    /// launch validation; callers must still apply the option-specific policy.
    pub fn supports_flag(&self, flag: &str) -> bool {
        self.serve_flags.iter().any(|candidate| candidate == flag)
    }

    /// Current llama.cpp reasoning-effort values observed by the product
    /// contract. Help proves the flag exists; the typed enum still rejects an
    /// unknown stored value before launch.
    pub fn supports_reasoning_effort(&self, value: &str) -> bool {
        self.supports_flag("--reasoning-effort")
            && (self.typed.reasoning_effort.accepted_values.is_empty()
                || self
                    .typed
                    .reasoning_effort
                    .accepted_values
                    .iter()
                    .any(|candidate| candidate == value))
    }

    /// Current explicit reasoning-format values observed by the product
    /// contract. `auto` is intentionally absent: no argument means auto.
    pub fn supports_reasoning_format(&self, value: &str) -> bool {
        self.supports_flag("--reasoning-format")
            && (self.typed.reasoning_format.accepted_values.is_empty()
                || self
                    .typed
                    .reasoning_format
                    .accepted_values
                    .iter()
                    .any(|candidate| candidate == value))
    }

    /// The current product has no authoritative template capability contract
    /// for native reasoning preservation. Keep this fail-closed until a
    /// bounded template inspection rule is qualified with fixtures.
    pub fn reasoning_preserve_template_compatibility(&self) -> FeatureState {
        self.typed.reasoning_preserve_template.clone()
    }

    pub fn is_valid_for(&self, current: &ExecutableIdentity) -> bool {
        self.executable_identity.path == current.path
            && self.executable_identity.file_hash == current.file_hash
    }

    pub fn fingerprint(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.executable_identity.path.as_bytes());
        hasher.update(self.executable_identity.file_hash.as_bytes());
        hasher.update(self.help_hash.as_bytes());
        hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }
}

static SNAPSHOT_CACHE: OnceLock<Arc<std::sync::RwLock<BTreeMap<String, CapabilitySnapshot>>>> =
    OnceLock::new();

pub fn cached_snapshot(identity: &ExecutableIdentity) -> Option<CapabilitySnapshot> {
    let cache = SNAPSHOT_CACHE
        .get_or_init(|| Arc::new(std::sync::RwLock::new(BTreeMap::new())))
        .clone();
    let path_key = identity.path.clone();
    cache
        .read()
        .unwrap()
        .get(&path_key)
        .cloned()
        .filter(|snap| snap.is_valid_for(identity))
}

pub fn cache_snapshot(snapshot: CapabilitySnapshot) {
    let cache = SNAPSHOT_CACHE
        .get_or_init(|| Arc::new(std::sync::RwLock::new(BTreeMap::new())))
        .clone();
    let key = snapshot.executable_identity.path.clone();
    cache.write().unwrap().insert(key, snapshot);
}

pub async fn generate_snapshot(binary: &Path) -> Result<CapabilitySnapshot> {
    let identity = ExecutableIdentity::from_path(binary)?;

    let version = probe_version(binary).await?;
    let (help_text, flags) = probe_help(binary).await?;
    let help_hash = hash_help(&help_text);

    let cache = derive_cache_capabilities(&flags);
    let context = derive_context_capabilities(&flags);
    let concurrency = derive_concurrency_capabilities(&flags);
    let endpoints = derive_endpoint_capabilities(&flags);
    let streaming = derive_streaming_capabilities(&flags);
    let templates = derive_template_capabilities(&flags);
    let tools = derive_tool_capabilities(&flags);
    let speculation = derive_speculation_capabilities(&flags);
    let typed = derive_typed_capabilities(&flags, &help_text);

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let snapshot = CapabilitySnapshot {
        executable_identity: identity,
        version_text: version,
        help_hash,
        serve_flags: flags,
        cache,
        context,
        concurrency,
        endpoints,
        streaming,
        templates,
        tools,
        speculation,
        typed,
        // Phase 1b: only shipped provider — product default denied.
        mixed_main_kv: MixedMainKv::product_default_denied(),
        evidence_timestamp: now,
        source: CapabilitySnapshotSource::AutoProbed,
    };

    cache_snapshot(snapshot.clone());
    Ok(snapshot)
}

fn derive_typed_capabilities(flags: &[String], help_text: &str) -> TypedLlamaCapabilities {
    let has = |flag: &str| flag_state(flags, flag);
    let default_value = |flag: &str| help_default(help_text, flag);
    TypedLlamaCapabilities {
        mmproj_offload: BooleanFlagCapability {
            positive: has("--mmproj-offload"),
            negative: has("--no-mmproj-offload"),
            default_value: default_value("--mmproj-offload").and_then(|value| {
                match value.as_str() {
                    "enabled" | "true" => Some(true),
                    "disabled" | "false" => Some(false),
                    _ => None,
                }
            }),
        },
        reasoning_effort: EnumCapability {
            supported: has("--reasoning-effort"),
            accepted_values: if flags.iter().any(|flag| flag == "--reasoning-effort") {
                ["minimal", "low", "medium", "high", "xhigh", "max"]
                    .into_iter()
                    .map(String::from)
                    .collect()
            } else {
                Vec::new()
            },
            default_value: default_value("--reasoning-effort"),
        },
        reasoning_format: EnumCapability {
            supported: has("--reasoning-format"),
            accepted_values: if flags.iter().any(|flag| flag == "--reasoning-format") {
                ["none", "deepseek", "deepseek-legacy"]
                    .into_iter()
                    .map(String::from)
                    .collect()
            } else {
                Vec::new()
            },
            default_value: default_value("--reasoning-format"),
        },
        reasoning_preserve: BooleanFlagCapability {
            positive: has("--reasoning-preserve"),
            negative: has("--no-reasoning-preserve"),
            default_value: default_value("--reasoning-preserve").and_then(|value| {
                match value.as_str() {
                    "enabled" | "true" => Some(true),
                    "disabled" | "false" => Some(false),
                    _ => None,
                }
            }),
        },
        reasoning_preserve_template: FeatureState::Unavailable(
            "template compatibility for native reasoning preservation is not verified".into(),
        ),
    }
}

fn help_default(help_text: &str, flag: &str) -> Option<String> {
    help_text.lines().find_map(|line| {
        if !line.split([',', ' ']).any(|token| token == flag) {
            return None;
        }
        let marker = "(default:";
        let start = line.find(marker)? + marker.len();
        let value = line[start..].split(')').next()?.trim();
        (!value.is_empty()).then(|| value.to_string())
    })
}

fn recognizable_version_line(text: &str) -> Option<String> {
    text.lines().map(str::trim).find_map(|trimmed| {
        if (trimmed.starts_with("llama-server") && trimmed.contains("version"))
            || trimmed.starts_with("llama.cpp")
        {
            Some(trimmed.to_string())
        } else {
            None
        }
    })
}

async fn probe_version(binary: &Path) -> Result<String> {
    // Recent llama.cpp builds no longer print a version banner in `--help`.
    // Prefer the dedicated probe, then fall back to recognizable help text;
    // capability extraction must not be disabled merely because version text
    // moved between commands.
    if let Ok(output) = run_probe_command(binary, &["--version"]).await {
        let text = format!(
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        if let Some(line) = text.lines().map(str::trim).find(|line| !line.is_empty()) {
            return Ok(line.to_string());
        }
    }

    let output = run_probe_command(binary, &["--help"]).await?;
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    if let Some(line) = recognizable_version_line(&text) {
        return Ok(line);
    }
    Ok("llama-server version unknown".into())
}

async fn probe_help(binary: &Path) -> Result<(String, Vec<String>)> {
    let output = run_probe_command(binary, &["--help"]).await?;
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let text = text.trim().to_string();
    let flags = extract_flags(&text);
    Ok((text, flags))
}

fn hash_help(help: &str) -> String {
    Sha256::digest(help.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn extract_flags(help: &str) -> Vec<String> {
    let mut flags = BTreeMap::new();
    for line in help.lines() {
        for token in line.split_whitespace() {
            let token =
                token.trim_matches(|c: char| matches!(c, ',' | '[' | ']' | '(' | ')' | '='));
            let flag = token.split_once('=').map_or(token, |(f, _)| f);
            if flag.starts_with("--") || (flag.starts_with('-') && !flag.starts_with("-hf")) {
                flags.insert(flag.to_string(), ());
            }
        }
    }
    flags.into_keys().collect()
}

fn derive_cache_capabilities(flags: &[String]) -> CacheCapabilities {
    CacheCapabilities {
        prompt_cache: flag_state(flags, "--cache-prompt"),
        ram_cache: flag_state(flags, "--cache-ram"),
        idle_slot_cache: flag_state(flags, "--cache-idle-slots"),
        cache_reuse: flag_state(flags, "--cache-reuse"),
    }
}

fn derive_context_capabilities(flags: &[String]) -> ContextCapabilities {
    ContextCapabilities {
        checkpoints: flag_state(flags, "--ctx-checkpoints"),
        kv_unified: flag_state(flags, "--kv-unified"),
        kv_partitioned: flag_state(flags, "--kv-separated"),
    }
}

fn derive_concurrency_capabilities(flags: &[String]) -> ConcurrencyCapabilities {
    ConcurrencyCapabilities {
        auto_slots: flag_state(flags, "--parallel"),
        explicit_slots: flag_state(flags, "-c"),
        continuous_batching: flag_state(flags, "--continuous-batching"),
    }
}

fn derive_endpoint_capabilities(flags: &[String]) -> EndpointCapabilities {
    let has_chat = flags.iter().any(|f| f.contains("chat"));
    let has_responses = flags.iter().any(|f| f.contains("responses"));
    EndpointCapabilities {
        chat_completions: if has_chat {
            FeatureState::Available
        } else {
            FeatureState::Unavailable("No chat completions endpoint detected in --help".into())
        },
        responses: if has_responses {
            FeatureState::Available
        } else {
            FeatureState::Unavailable("No Responses API endpoint detected in --help".into())
        },
        raw_completion: FeatureState::Available,
        text_completion: flag_state(flags, "--in-prefix"),
    }
}

fn derive_streaming_capabilities(flags: &[String]) -> StreamingCapabilities {
    let has_usage = flags
        .iter()
        .any(|f| f.contains("usage") || f.contains("include_usage"));
    StreamingCapabilities {
        usage_in_stream: if has_usage {
            FeatureState::Available
        } else {
            FeatureState::Unavailable(
                "No stream_options/include_usage flag detected in --help".into(),
            )
        },
        progress_in_stream: FeatureState::Available,
    }
}

fn derive_template_capabilities(flags: &[String]) -> TemplateCapabilities {
    TemplateCapabilities {
        jinja: flag_state(flags, "--jinja"),
        chat_template_file: flag_state(flags, "--chat-template-file"),
        chat_template_kwargs: flag_state(flags, "--chat-template-kwargs"),
    }
}

fn derive_tool_capabilities(flags: &[String]) -> ToolCapabilities {
    ToolCapabilities {
        tool_parsing: flag_state(flags, "--tool-call-parser"),
        parallel_tools: flag_state(flags, "--parallel-tool-calls"),
    }
}

fn derive_speculation_capabilities(flags: &[String]) -> SpeculationCapabilities {
    SpeculationCapabilities {
        draft_model: flag_state(flags, "--spec-type"),
        ngram_spec: flag_state(flags, "--spec-ngram-size-n"),
    }
}

fn flag_state(flags: &[String], flag: &str) -> FeatureState {
    if flags.iter().any(|f| f == flag) {
        FeatureState::Available
    } else {
        FeatureState::Unavailable(format!("Flag {flag} not found in --help"))
    }
}

struct ProbeOutput {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

async fn run_probe_command(binary: &Path, args: &[&str]) -> Result<ProbeOutput> {
    let mut cmd = Command::new(binary);
    cmd.args(args)
        .kill_on_drop(true)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null());

    let mut child = cmd.spawn().map_err(|e| {
        anyhow!(
            "Failed to execute llama-server probe '{} {}': {}",
            binary.display(),
            args.join(" "),
            e
        )
    })?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("Failed to capture probe stdout"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("Failed to capture probe stderr"))?;

    let capture = async {
        async fn read_bound<R>(reader: R) -> Result<Vec<u8>>
        where
            R: tokio::io::AsyncRead + Unpin,
        {
            let mut out = Vec::with_capacity(4096);
            reader
                .take((CAPABILITY_PROBE_MAX_OUTPUT + 1) as u64)
                .read_to_end(&mut out)
                .await?;
            if out.len() > CAPABILITY_PROBE_MAX_OUTPUT {
                bail!(
                    "llama-server probe output exceeded {} byte limit",
                    CAPABILITY_PROBE_MAX_OUTPUT
                );
            }
            Ok(out)
        }

        let (stdout_data, stderr_data, status) =
            tokio::join!(read_bound(stdout), read_bound(stderr), child.wait());
        let _ = status?;
        Ok::<(Vec<u8>, Vec<u8>), anyhow::Error>((stdout_data?, stderr_data?))
    };

    let (stdout_data, stderr_data) = tokio::time::timeout(CAPABILITY_PROBE_TIMEOUT, capture)
        .await
        .map_err(|_| {
            anyhow!(
                "llama-server probe timed out after {:.1}s: {} {}",
                CAPABILITY_PROBE_TIMEOUT.as_secs_f64(),
                binary.display(),
                args.join(" ")
            )
        })??;

    Ok(ProbeOutput {
        stdout: stdout_data,
        stderr: stderr_data,
    })
}

fn hash_file(path: &Path) -> Result<String> {
    let mut file =
        std::fs::File::open(path).context("Cannot open llama-server executable for hashing")?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let n = std::io::Read::read(&mut file, &mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn help_hash_is_deterministic() {
        let hash1 = hash_help("--host --port --cache-prompt");
        let hash2 = hash_help("--host --port --cache-prompt");
        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash_help("--host --port"));
    }

    #[test]
    fn typed_capability_evidence_records_forms_values_and_defaults() {
        let help = "Options:\n  --mmproj-offload, --no-mmproj-offload (default: enabled)\n  --reasoning-format FORMAT\n  --reasoning-effort LEVEL\n  --reasoning-preserve, --no-reasoning-preserve (default: template default)";
        let flags = extract_flags(help);
        let typed = derive_typed_capabilities(&flags, help);
        assert!(matches!(
            typed.mmproj_offload.positive,
            FeatureState::Available
        ));
        assert!(matches!(
            typed.mmproj_offload.negative,
            FeatureState::Available
        ));
        assert_eq!(typed.mmproj_offload.default_value, Some(true));
        assert_eq!(
            typed.reasoning_effort.accepted_values,
            vec!["minimal", "low", "medium", "high", "xhigh", "max"]
        );
        assert_eq!(
            typed.reasoning_format.accepted_values,
            vec!["none", "deepseek", "deepseek-legacy"]
        );
        assert!(matches!(
            typed.reasoning_preserve_template,
            FeatureState::Unavailable(_)
        ));
    }

    #[test]
    fn version_probe_accepts_help_without_a_version_banner() {
        let help = "Usage: llama-server [OPTIONS]\nOptions:\n  --model PATH\n  --load-mode MODE";
        assert_eq!(recognizable_version_line(help), None);
        assert_eq!(
            "llama-server version unknown",
            recognizable_version_line(help)
                .unwrap_or_else(|| "llama-server version unknown".to_string())
        );
    }

    #[test]
    fn extract_flags_from_help_text() {
        let help = r#"
Usage: llama-server [OPTIONS]

Options:
  --host TEXT
  --port INTEGER
  --cache-prompt
  --cache-ram INTEGER
  --parallel INTEGER
  --continuous-batching
  --spec-type TEXT
"#;
        let flags = extract_flags(help);
        assert!(flags.contains(&"--host".into()));
        assert!(flags.contains(&"--port".into()));
        assert!(flags.contains(&"--cache-prompt".into()));
        assert!(flags.contains(&"--parallel".into()));
        assert!(flags.contains(&"--continuous-batching".into()));
    }

    #[test]
    fn cache_capabilities_derived_from_flags() {
        let flags = vec![
            "--cache-prompt".into(),
            "--cache-ram".into(),
            "--cache-idle-slots".into(),
        ];
        let caps = derive_cache_capabilities(&flags);
        assert!(matches!(caps.prompt_cache, FeatureState::Available));
        assert!(matches!(caps.ram_cache, FeatureState::Available));
        assert!(matches!(caps.idle_slot_cache, FeatureState::Available));
        match caps.cache_reuse {
            FeatureState::Unavailable(_) => {}
            other => panic!("Expected Unavailable, got {:?}", other),
        }
    }

    #[test]
    fn context_capabilities_derived_from_flags() {
        let flags = vec!["--ctx-checkpoints".into(), "--kv-unified".into()];
        let caps = derive_context_capabilities(&flags);
        assert!(matches!(caps.checkpoints, FeatureState::Available));
        assert!(matches!(caps.kv_unified, FeatureState::Available));
    }

    #[test]
    fn concurrency_capabilities_derived_from_flags() {
        let flags = vec![
            "--parallel".into(),
            "-c".into(),
            "--continuous-batching".into(),
        ];
        let caps = derive_concurrency_capabilities(&flags);
        assert!(matches!(caps.auto_slots, FeatureState::Available));
        assert!(matches!(caps.explicit_slots, FeatureState::Available));
        assert!(matches!(caps.continuous_batching, FeatureState::Available));
    }

    #[test]
    fn snapshot_invalidates_on_file_hash_change() {
        let identity1 = ExecutableIdentity {
            path: "/tmp/llama-server".into(),
            file_hash: "abc123".into(),
            file_mtime_unix: 1000,
        };
        let identity2 = ExecutableIdentity {
            path: "/tmp/llama-server".into(),
            file_hash: "def456".into(),
            file_mtime_unix: 2000,
        };
        // help_hash is deliberately built from raw --help text, not from
        // `serve_flags.join(" ")`, so this test cannot pass against the
        // help_hash-re-derivation bug by accident.
        let snap = CapabilitySnapshot {
            executable_identity: identity1.clone(),
            version_text: "b10068".into(),
            help_hash: hash_help("usage: llama-server [options]\n  --host HOST\n  --port PORT\n"),
            serve_flags: vec!["--host".into(), "--port".into()],
            cache: Default::default(),
            context: Default::default(),
            concurrency: Default::default(),
            endpoints: Default::default(),
            streaming: Default::default(),
            templates: Default::default(),
            tools: Default::default(),
            speculation: Default::default(),
            typed: Default::default(),
            mixed_main_kv: MixedMainKv::product_default_denied(),
            evidence_timestamp: 0,
            source: CapabilitySnapshotSource::AutoProbed,
        };
        assert!(snap.is_valid_for(&identity1));
        assert!(!snap.is_valid_for(&identity2));
    }

    #[test]
    fn cached_snapshot_hits_for_unchanged_executable_identity() {
        // Regression test for the help_hash re-derivation bug: is_valid_for
        // used to recompute help_hash from serve_flags.join(" ") at lookup
        // time and compare it against a hash stored from real raw --help
        // text at probe time. Those two strings differ for every real probe,
        // so every lookup missed the cache even for an unchanged binary.
        let identity = ExecutableIdentity {
            path: "/tmp/llama-server-cache-hit-test".into(),
            file_hash: "unchanged-hash".into(),
            file_mtime_unix: 1000,
        };
        let snap = CapabilitySnapshot {
            executable_identity: identity.clone(),
            version_text: "b10068".into(),
            help_hash: hash_help(
                "usage: llama-server [options]\n  --host HOST\n  --cache-reuse N\n",
            ),
            serve_flags: vec!["--host".into(), "--cache-reuse".into()],
            cache: Default::default(),
            context: Default::default(),
            concurrency: Default::default(),
            endpoints: Default::default(),
            streaming: Default::default(),
            templates: Default::default(),
            tools: Default::default(),
            speculation: Default::default(),
            typed: Default::default(),
            mixed_main_kv: MixedMainKv::product_default_denied(),
            evidence_timestamp: 0,
            source: CapabilitySnapshotSource::AutoProbed,
        };
        cache_snapshot(snap);
        assert!(
            cached_snapshot(&identity).is_some(),
            "unchanged executable identity must hit the cache without re-probing"
        );

        let changed_identity = ExecutableIdentity {
            path: "/tmp/llama-server-cache-hit-test".into(),
            file_hash: "changed-hash".into(),
            file_mtime_unix: 2000,
        };
        assert!(
            cached_snapshot(&changed_identity).is_none(),
            "a changed file_hash must still miss the cache and re-probe"
        );
    }

    #[test]
    fn older_binary_does_not_inherit_newer_features() {
        let old_flags = vec!["--host".into(), "--port".into()];
        let new_flags = vec![
            "--host".into(),
            "--port".into(),
            "--cache-reuse".into(),
            "--continuous-batching".into(),
            "--spec-type".into(),
        ];
        let old_caps = derive_cache_capabilities(&old_flags);
        let new_caps = derive_cache_capabilities(&new_flags);
        match old_caps.cache_reuse {
            FeatureState::Unavailable(_) => {}
            _ => panic!("Old binary should not have cache_reuse"),
        }
        assert!(matches!(new_caps.cache_reuse, FeatureState::Available));
    }

    #[test]
    fn fingerprint_is_unique_per_executable_and_help() {
        let mut snap1 = CapabilitySnapshot {
            executable_identity: ExecutableIdentity {
                path: "/tmp/llama-server".into(),
                file_hash: "h".into(),
                file_mtime_unix: 0,
            },
            version_text: "b10068".into(),
            help_hash: hash_help("--host"),
            serve_flags: vec!["--host".into()],
            cache: Default::default(),
            context: Default::default(),
            concurrency: Default::default(),
            endpoints: Default::default(),
            streaming: Default::default(),
            templates: Default::default(),
            tools: Default::default(),
            speculation: Default::default(),
            typed: Default::default(),
            mixed_main_kv: MixedMainKv::product_default_denied(),
            evidence_timestamp: 0,
            source: CapabilitySnapshotSource::AutoProbed,
        };
        let fp1 = snap1.fingerprint();
        snap1.executable_identity.file_hash = "h2".into();
        let fp2 = snap1.fingerprint();
        assert_ne!(fp1, fp2);
    }
}
