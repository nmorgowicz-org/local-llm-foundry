//! `GET /api/llama-binary/kv-support`
//!
//! Authenticated, model-aware K/V pair classification. The client sends the
//! pair, the preset's flash-attention setting, an opaque model id and a
//! detected-device id; it never sends a filesystem path. The server resolves
//! the model id inside the configured models directory, reads GGUF metadata
//! itself and classifies against the loaded runtime's evidence.
//!
//! `fa` is the preset's `flash_attn` value and is mapped exactly the way a
//! launch maps it (`launch_flash_attn_mode`), so the answer matches what the
//! server would be started with.

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use sha2::{Digest, Sha256};
use warp::Filter;
use warp::http::StatusCode;

use super::{ApiCtx, ApiRoute, box_reply, check_api_token, unauthorized_api_token};
use crate::config::AppConfig;
use crate::inference::kv_advisories::{ModelFacts, advisories_for};
use crate::inference::kv_support::{
    DeviceEvidence, KvContext, KvSupportBasis, KvType, ModelEvidence, classify_kv_pair,
    launch_flash_attn_mode,
};
use crate::inference::llama_cpp_capabilities::CapabilitySnapshot;
use crate::llama::gguf_meta::GgufMetadata;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum KvQueryError {
    BadRequest(String),
}

/// Resolve an opaque model id (a path relative to the models directory) to a
/// `.gguf` file that is verifiably inside that directory.
pub(crate) fn resolve_model_id(models_dir: &Path, id: &str) -> Result<PathBuf, KvQueryError> {
    let bad = |message: &str| KvQueryError::BadRequest(message.to_string());
    let id = id.trim();
    if id.is_empty() {
        return Err(bad("model_id is empty"));
    }
    let relative = Path::new(id);
    if relative.is_absolute()
        || relative
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err(bad(
            "model_id must be a relative path inside the models directory",
        ));
    }
    if !id.to_ascii_lowercase().ends_with(".gguf") {
        return Err(bad("model_id must name a .gguf file"));
    }
    let root = models_dir
        .canonicalize()
        .map_err(|_| bad("models directory is not available"))?;
    let resolved = root
        .join(relative)
        .canonicalize()
        .map_err(|_| bad("model_id does not name a model in the models directory"))?;
    if !resolved.starts_with(&root) || !resolved.is_file() {
        return Err(bad(
            "model_id does not name a model in the models directory",
        ));
    }
    Ok(resolved)
}

/// Per-head attention dimensions from GGUF metadata. Falls back to
/// `embedding_length / head_count` when `key_length` is absent.
pub(crate) fn head_dims(meta: &GgufMetadata) -> Vec<u32> {
    let mut dims = Vec::new();
    let primary = meta
        .key_length
        .or_else(|| match (meta.embedding_length, meta.head_count) {
            (Some(embedding), Some(heads)) if heads > 0 && embedding % heads == 0 => {
                Some(embedding / heads)
            }
            _ => None,
        });
    dims.extend(primary);
    if let Some(swa) = meta.key_length_swa
        && !dims.contains(&swa)
    {
        dims.push(swa);
    }
    dims.sort_unstable();
    dims
}

fn model_evidence(id: &str, path: &Path, meta: &GgufMetadata) -> ModelEvidence {
    // Identity covers the id, size and mtime (a re-downloaded or replaced file
    // is a different revision) plus the architecture fields used above.
    let (size, mtime) = std::fs::metadata(path)
        .map(|m| {
            let mtime = m
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            (m.len(), mtime)
        })
        .unwrap_or((0, 0));
    let dims = head_dims(meta);
    let mut hasher = Sha256::new();
    hasher.update(format!(
        "{id}|{size}|{mtime}|{:?}|{dims:?}",
        meta.architecture
    ));
    let identity: String = hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    ModelEvidence {
        metadata_identity: identity,
        head_dims: dims,
    }
}

/// The device id token of an enumerated device line (`"CUDA0 : NVIDIA ..."`).
fn device_token(line: &str) -> &str {
    line.split(" : ").next().unwrap_or(line).trim()
}

pub(crate) fn resolve_device(
    basis: &KvSupportBasis,
    device_id: Option<&str>,
) -> Result<Option<DeviceEvidence>, KvQueryError> {
    let Some(id) = device_id.map(str::trim).filter(|id| !id.is_empty()) else {
        return Ok(None);
    };
    if !basis.devices.iter().any(|line| device_token(line) == id) {
        return Err(KvQueryError::BadRequest(format!(
            "device_id {id:?} was not enumerated by the loaded runtime"
        )));
    }
    Ok(Some(DeviceEvidence {
        id: id.to_string(),
        // Reported by the runtime's own Metal device init; unavailable otherwise.
        metal_bf16: basis.metal_bf16.clone(),
    }))
}

/// Build the JSON response. Pure: the caller supplies the snapshot and the
/// already-resolved model.
pub(crate) fn build_response(
    snapshot: Option<&CapabilitySnapshot>,
    params: &HashMap<String, String>,
    model: Option<ModelEvidence>,
    facts: Option<ModelFacts>,
) -> Result<serde_json::Value, KvQueryError> {
    let parse = |key: &str| -> Result<KvType, KvQueryError> {
        let raw = params.get(key).map(String::as_str).unwrap_or("");
        KvType::parse(raw)
            .ok_or_else(|| KvQueryError::BadRequest(format!("{key} is not a known cache type")))
    };
    let k = parse("k")?;
    let v = parse("v")?;
    let fa = launch_flash_attn_mode(params.get("fa").map(String::as_str).unwrap_or(""));

    let unavailable = KvSupportBasis::unavailable("no llama-server runtime evidence");
    let basis = snapshot.map(|s| &s.kv_basis).unwrap_or(&unavailable);
    let device = resolve_device(basis, params.get("device_id").map(String::as_str))?;
    let context = KvContext {
        basis,
        device: device.as_ref(),
        model: model.as_ref(),
        observations: None,
    };
    let classification = classify_kv_pair(k, v, fa, &context);
    Ok(serde_json::json!({
        "ok": true,
        "k": k.name(),
        "v": v.name(),
        "effective_flash_attn": fa,
        "compiled_pair": classification.compiled_pair,
        "support": classification.expected,
        "blocking": classification.is_blocking(),
        "reason": classification.reason(),
        "fix": classification.fix(),
        "buildable_missing": classification.buildable_missing,
        "backend": basis.backend,
        // Guidance only; never changes the classification. Empty without model facts.
        "advisories": advisories_for(k, v, facts.as_ref()),
        "runtime_identity": basis.runtime_identity,
        "device_identity": device.as_ref().map(|d| d.id.clone()),
        "model_metadata_identity": model.as_ref().map(|m| m.metadata_identity.clone()),
        "basis_version": basis.version,
        // Partial, session-specific confirmation from server logs. Never feeds
        // the classification; absence proves nothing.
        "session_observations": crate::inference::kv_support::session_observations(),
    }))
}

fn error_reply(status: StatusCode, message: &str) -> super::ApiReply {
    Box::new(warp::reply::with_status(
        warp::reply::json(&serde_json::json!({ "ok": false, "error": message })),
        status,
    ))
}

async fn current_snapshot(config: &AppConfig) -> Option<CapabilitySnapshot> {
    if !config.llama_server_path.is_file() {
        return None;
    }
    let _ = crate::inference::llama_cpp_capabilities::generate_snapshot(&config.llama_server_path)
        .await;
    crate::inference::llama_cpp_capabilities::ExecutableIdentity::from_path(
        &config.llama_server_path,
    )
    .ok()
    .and_then(|identity| crate::inference::llama_cpp_capabilities::cached_snapshot(&identity))
}

pub(crate) fn routes(ctx: ApiCtx) -> ApiRoute {
    let state = ctx.state;
    let config: Arc<AppConfig> = ctx.config;
    warp::path!("api" / "llama-binary" / "kv-support")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and(warp::query::<HashMap<String, String>>())
        .and_then(
            move |auth: Option<String>, params: HashMap<String, String>| {
                let state = state.clone();
                let config = config.clone();
                async move {
                    if !check_api_token(&auth, &config) {
                        return Ok::<super::ApiReply, warp::Rejection>(box_reply(
                            unauthorized_api_token(),
                        ));
                    }
                    let model = match params
                        .get("model_id")
                        .map(String::as_str)
                        .filter(|id| !id.trim().is_empty())
                    {
                        None => None,
                        Some(id) => {
                            let models_dir = super::models::get_effective_models_dir(&state)
                                .unwrap_or_else(|| config.default_models_dir.clone());
                            let resolved = match resolve_model_id(&models_dir, id) {
                                Ok(path) => path,
                                Err(KvQueryError::BadRequest(message)) => {
                                    return Ok(error_reply(StatusCode::BAD_REQUEST, &message));
                                }
                            };
                            // Metadata is read server-side; unreadable metadata
                            // simply leaves the model evidence out.
                            let path = resolved.clone();
                            let meta = tokio::task::spawn_blocking(move || {
                                crate::llama::gguf_meta::read_gguf_metadata(&path)
                            })
                            .await
                            .ok()
                            .and_then(Result::ok);
                            meta.map(|meta| {
                                let facts = ModelFacts {
                                    architecture: meta.architecture.clone(),
                                    file_name: resolved
                                        .file_name()
                                        .map(|n| n.to_string_lossy().into_owned()),
                                };
                                (model_evidence(id, &resolved, &meta), facts)
                            })
                        }
                    };
                    let (model, facts) = match model {
                        Some((evidence, facts)) => (Some(evidence), Some(facts)),
                        None => (None, None),
                    };
                    let snapshot = current_snapshot(&config).await;
                    match build_response(snapshot.as_ref(), &params, model, facts) {
                        Ok(body) => Ok(box_reply(warp::reply::json(&body))),
                        Err(KvQueryError::BadRequest(message)) => {
                            Ok(error_reply(StatusCode::BAD_REQUEST, &message))
                        }
                    }
                }
            },
        )
        .boxed()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::kv_support::{Backend, Evidence, KV_BASIS_VERSION, compiled_fa_pairs};

    fn params(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn cuda_snapshot() -> CapabilitySnapshot {
        let mut snapshot = CapabilitySnapshot::product_default();
        snapshot.kv_basis = KvSupportBasis {
            version: KV_BASIS_VERSION,
            backend: Backend::Cuda,
            fa_pairs: Evidence::confirmed(compiled_fa_pairs("all").unwrap(), "test"),
            runtime_identity: Some("rt-1".into()),
            devices: vec!["CUDA0 : NVIDIA GeForce RTX 5090 (32606 MiB)".into()],
            ..Default::default()
        };
        snapshot
    }

    #[test]
    fn model_id_must_stay_inside_the_models_directory() {
        let dir = tempfile::tempdir().unwrap();
        let models = dir.path().join("models");
        std::fs::create_dir_all(models.join("sub")).unwrap();
        std::fs::write(models.join("sub/a.gguf"), b"x").unwrap();
        std::fs::write(dir.path().join("outside.gguf"), b"x").unwrap();
        std::fs::write(models.join("notes.txt"), b"x").unwrap();

        assert!(resolve_model_id(&models, "sub/a.gguf").is_ok());
        for bad in [
            "",
            "  ",
            "../outside.gguf",
            "sub/../../outside.gguf",
            "/etc/passwd.gguf",
            "notes.txt",
            "missing.gguf",
            "sub",
        ] {
            assert!(resolve_model_id(&models, bad).is_err(), "{bad:?}");
        }
        let absolute = dir.path().join("outside.gguf");
        assert!(resolve_model_id(&models, &absolute.to_string_lossy()).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn symlink_escaping_the_models_directory_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let models = dir.path().join("models");
        std::fs::create_dir_all(&models).unwrap();
        std::fs::write(dir.path().join("outside.gguf"), b"x").unwrap();
        std::os::unix::fs::symlink(dir.path().join("outside.gguf"), models.join("link.gguf"))
            .unwrap();
        assert!(resolve_model_id(&models, "link.gguf").is_err());
    }

    #[test]
    fn head_dims_use_key_length_then_derive_and_include_swa() {
        let meta = GgufMetadata {
            key_length: Some(512),
            key_length_swa: Some(256),
            ..Default::default()
        };
        assert_eq!(head_dims(&meta), vec![256, 512]);
        let derived = GgufMetadata {
            embedding_length: Some(4096),
            head_count: Some(32),
            ..Default::default()
        };
        assert_eq!(head_dims(&derived), vec![128]);
        assert!(head_dims(&GgufMetadata::default()).is_empty());
        let indivisible = GgufMetadata {
            embedding_length: Some(100),
            head_count: Some(7),
            ..Default::default()
        };
        assert!(head_dims(&indivisible).is_empty());
    }

    #[test]
    fn response_for_a_confirmed_pair_without_model_is_unverified_with_compiled_evidence() {
        let body = build_response(
            Some(&cuda_snapshot()),
            &params(&[("k", "q8_0"), ("v", "q4_0"), ("fa", "on")]),
            None,
            None,
        )
        .unwrap();
        assert_eq!(body["support"]["kind"], "unverified");
        assert_eq!(body["compiled_pair"]["state"], "confirmed");
        assert_eq!(body["compiled_pair"]["value"], true);
        assert_eq!(body["blocking"], false);
        assert_eq!(body["runtime_identity"], "rt-1");
        assert_eq!(body["advisories"], serde_json::json!([]));
    }

    #[test]
    fn response_is_native_with_device_and_model_evidence() {
        let model = ModelEvidence {
            metadata_identity: "m-1".into(),
            head_dims: vec![128],
        };
        let body = build_response(
            Some(&cuda_snapshot()),
            &params(&[
                ("k", "q8_0"),
                ("v", "q4_0"),
                ("fa", "on"),
                ("device_id", "CUDA0"),
            ]),
            Some(model),
            None,
        )
        .unwrap();
        assert_eq!(body["support"]["kind"], "native");
        assert_eq!(body["device_identity"], "CUDA0");
        assert_eq!(body["model_metadata_identity"], "m-1");
    }

    #[test]
    fn quantized_v_with_fa_off_is_the_blocking_answer() {
        let body = build_response(
            Some(&cuda_snapshot()),
            &params(&[("k", "q8_0"), ("v", "q4_0"), ("fa", "off")]),
            None,
            None,
        )
        .unwrap();
        assert_eq!(body["support"]["kind"], "wont_start");
        assert_eq!(body["blocking"], true);
        assert!(body["fix"].as_str().unwrap().contains("flash attention"));
    }

    #[test]
    fn missing_fa_follows_launch_semantics_and_auto_launches_as_on() {
        for fa in [None, Some(""), Some("auto"), Some("on")] {
            let mut p = params(&[("k", "q8_0"), ("v", "q4_0")]);
            if let Some(fa) = fa {
                p.insert("fa".into(), fa.into());
            }
            let body = build_response(Some(&cuda_snapshot()), &p, None, None).unwrap();
            assert_eq!(body["effective_flash_attn"], "on", "{fa:?}");
            assert_eq!(body["blocking"], false);
        }
    }

    #[test]
    fn no_runtime_evidence_is_unverified_never_blocking() {
        let body =
            build_response(None, &params(&[("k", "q8_0"), ("v", "q4_0")]), None, None).unwrap();
        assert_eq!(body["support"]["kind"], "unverified");
        assert_eq!(body["blocking"], false);
        assert_eq!(body["compiled_pair"]["state"], "unavailable");
    }

    #[test]
    fn unknown_types_and_devices_are_bad_requests() {
        let snapshot = cuda_snapshot();
        assert!(build_response(Some(&snapshot), &params(&[("k", "q8_0")]), None, None).is_err());
        assert!(
            build_response(
                Some(&snapshot),
                &params(&[("k", "wat"), ("v", "q8_0")]),
                None,
                None
            )
            .is_err()
        );
        let err = build_response(
            Some(&snapshot),
            &params(&[("k", "q8_0"), ("v", "q8_0"), ("device_id", "CUDA9")]),
            None,
            None,
        );
        assert!(err.is_err());
        // Without any enumerated devices a device id cannot be validated.
        assert!(
            build_response(
                None,
                &params(&[("k", "q8_0"), ("v", "q8_0"), ("device_id", "CUDA0")]),
                None,
                None
            )
            .is_err()
        );
    }

    #[test]
    fn device_tokens_match_enumerated_ids_exactly() {
        let basis = cuda_snapshot().kv_basis;
        assert!(resolve_device(&basis, Some("CUDA0")).unwrap().is_some());
        assert!(resolve_device(&basis, Some("CUDA")).is_err());
        assert!(resolve_device(&basis, Some("")).unwrap().is_none());
        assert!(resolve_device(&basis, None).unwrap().is_none());
    }

    #[test]
    fn advisories_need_model_facts_and_never_change_the_classification() {
        use crate::inference::kv_advisories::ModelFacts;
        let p = params(&[("k", "q8_0"), ("v", "q4_0"), ("fa", "on")]);
        let none = build_response(Some(&cuda_snapshot()), &p, None, None).unwrap();
        assert_eq!(none["advisories"], serde_json::json!([]));

        let gemma = ModelFacts {
            architecture: Some("gemma4".into()),
            file_name: Some("gemma-4-26B-A4B-it.gguf".into()),
        };
        let with = build_response(Some(&cuda_snapshot()), &p, None, Some(gemma)).unwrap();
        assert_eq!(with["advisories"][0]["id"], "gemma4-quantized-kv");
        assert_eq!(with["advisories"][0]["severity"], "caution");
        assert!(
            with["advisories"][0]["source"]
                .as_str()
                .unwrap()
                .contains("qualitative")
        );
        assert_eq!(with["support"], none["support"]);
        assert_eq!(with["blocking"], none["blocking"]);
    }
}
