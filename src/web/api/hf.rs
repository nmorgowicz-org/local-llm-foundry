use std::path::{Path, PathBuf};
use std::sync::Arc;

use once_cell::sync::Lazy;
use warp::Filter;
use warp::http::StatusCode;

use crate::config::AppConfig;
use crate::state::AppState;

use super::common::{ApiCtx, ApiRoute, check_api_token, try_cooldown, unauthorized_api_token};
use super::models::get_effective_models_dir;

static HF_REPO_RE: Lazy<regex::Regex> =
    Lazy::new(|| regex::Regex::new(r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9._-]+$").unwrap());
pub(crate) static HF_EVIDENCE_GATE: Lazy<Arc<tokio::sync::Semaphore>> =
    Lazy::new(|| Arc::new(tokio::sync::Semaphore::new(2)));
const HF_EVIDENCE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(90);

fn validate_hf_repo_id(repo_id: &str) -> bool {
    HF_REPO_RE.is_match(repo_id)
}

/// Map the `format` field of an `/api/hf/search` request body to the
/// `HfModelFormat` threaded into the outgoing HF API `filter=` query param.
/// Anything other than `"mlx"` (including absent/empty/unrecognized values)
/// falls back to `Gguf` for backward compatibility.
pub(crate) fn parse_hf_format_param(raw: &str) -> crate::hf::HfModelFormat {
    match raw.to_lowercase().as_str() {
        "mlx" => crate::hf::HfModelFormat::Mlx,
        "both" | "all" => crate::hf::HfModelFormat::Both,
        _ => crate::hf::HfModelFormat::Gguf,
    }
}

pub(crate) fn resolve_hf_target_dir(
    models_dir: &Path,
    target_path: Option<&str>,
) -> Result<PathBuf, String> {
    std::fs::create_dir_all(models_dir).map_err(|e| {
        format!(
            "Failed to create models_dir {}: {}",
            models_dir.display(),
            e
        )
    })?;
    let models_dir_canon = models_dir.canonicalize().map_err(|e| {
        format!(
            "Failed to resolve models_dir {}: {}",
            models_dir.display(),
            e
        )
    })?;

    let Some(tp) = target_path else {
        return Ok(models_dir.to_path_buf());
    };

    if tp.contains("..") || tp.starts_with('\\') || tp.starts_with('/') {
        return Err("Invalid target_path: path traversal not allowed".to_string());
    }

    let candidate = models_dir.join(tp);
    std::fs::create_dir_all(&candidate)
        .map_err(|e| format!("Failed to create target_path: {}", e))?;
    let candidate_canon = candidate
        .canonicalize()
        .map_err(|e| format!("Failed to resolve target_path: {}", e))?;

    if !candidate_canon.starts_with(&models_dir_canon) {
        return Err("target_path escapes models_dir".to_string());
    }

    Ok(candidate_canon)
}

fn api_hf_search(
    _state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    // (window_start_secs, request_count) — protected by Mutex to avoid TOCTOU races.
    static HF_SEARCH_RATE: std::sync::LazyLock<std::sync::Mutex<(u64, u64)>> =
        std::sync::LazyLock::new(|| std::sync::Mutex::new((0, 0)));
    // Local self-protection only: Hugging Face allows 1000 requests per 5 minutes
    // with a token (500 without), and one UI interaction can issue several searches.
    const HF_SEARCH_WINDOW_SECS: u64 = 60;
    const HF_SEARCH_MAX_PER_WINDOW: u64 = 60;

    warp::path!("api" / "hf" / "search")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::hf_json_body::<serde_json::Value>())
        .and_then(move |auth: Option<String>, body: serde_json::Value| {
            let cfg = app_config.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }

                let now = std::time::SystemTime::UNIX_EPOCH
                    .elapsed()
                    .unwrap_or_default()
                    .as_secs();

                // Check and update rate limit atomically under the Mutex. A poisoned
                // lock only means another request panicked; the counters stay valid.
                let retry_after = {
                    let mut guard = HF_SEARCH_RATE.lock().unwrap_or_else(|e| e.into_inner());
                    let (window_start, count) = &mut *guard;
                    if now.saturating_sub(*window_start) >= HF_SEARCH_WINDOW_SECS {
                        *window_start = now;
                        *count = 1;
                        None
                    } else if *count >= HF_SEARCH_MAX_PER_WINDOW {
                        Some(HF_SEARCH_WINDOW_SECS.saturating_sub(now.saturating_sub(*window_start)).max(1))
                    } else {
                        *count += 1;
                        None
                    }
                };

                if let Some(wait) = retry_after {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(
                        Box::new(warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({
                                "ok": false,
                                "error": format!("Search rate limit reached. Try again in {wait}s."),
                                "retry_after_secs": wait
                            })),
                            StatusCode::TOO_MANY_REQUESTS,
                        )),
                    );
                }

                let query = body["query"].as_str().unwrap_or("").trim().to_string();
                let author = body["author"].as_str().map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
                let limit: u64 = body["limit"].as_u64().unwrap_or(20).min(100);

                let sort = match body["sort"].as_str().unwrap_or("downloads") {
                    "likes"     => crate::hf::HfSort::Likes,
                    "newest"    | "createdAt" => crate::hf::HfSort::CreatedAt,
                    "lastModified" | "last_modified" => crate::hf::HfSort::LastModified,
                    "relevance" => crate::hf::HfSort::Relevance,
                    "trending"  => crate::hf::HfSort::Trending,
                    _           => crate::hf::HfSort::Downloads,
                };

                // Require at least a query or an author — unless sorting by trending or
                // downloads (in which case empty query returns a global popular/trending list).
                // Relevance is deliberately not exempt: with nothing to be relevant to, HF
                // returns an arbitrary page.
                if query.is_empty()
                    && author.is_none()
                    && sort != crate::hf::HfSort::Trending
                    && sort != crate::hf::HfSort::Downloads
                {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(
                        Box::new(warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": "Provide 'query' or 'author' (or both)"
                        }))),
                    );
                }

                let cursor = body["cursor"].as_str().map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
                let format = parse_hf_format_param(body["format"].as_str().unwrap_or("gguf"));
                let quants_only = body["quantsOnly"].as_bool().unwrap_or(false);
                let params = crate::hf::HfSearchParams { query, author, sort, limit: limit as usize, cursor, format, quants_only };

                match crate::hf::hf_search_models(&params).await {
                    Ok((models, next_cursor)) => {
                        Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(
                        Box::new(warp::reply::json(&serde_json::json!({
                            "ok": true,
                            "models": models,
                            "next_cursor": next_cursor,
                            // Whether this request carried a Hugging Face token (the same
                            // lookup the upstream call uses). Authenticated searches get
                            // twice HF's anonymous quota.
                            "hf_authenticated": crate::hf::hf_load_token().is_some()
                        }))))
                    },
                    Err(e) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(
                        Box::new(warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": e
                        }))),
                    ),
                }
            }
        })
}

fn api_hf_files(
    _state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "hf" / "files")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::hf_json_body::<serde_json::Value>())
        .and_then(move |auth: Option<String>, body: serde_json::Value| {
            let cfg = app_config.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }

                let repo_id = body["repo_id"].as_str().unwrap_or("").trim().to_string();
                let format = body["format"].as_str().unwrap_or("gguf");
                if repo_id.is_empty() {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": "Missing 'repo_id' field"
                        })),
                    ));
                }
                if !validate_hf_repo_id(&repo_id) {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({
                                "ok": false,
                                "error": "Invalid repo_id format. Expected: owner/repo"
                            })),
                            StatusCode::BAD_REQUEST,
                        ),
                    ));
                }

                let result: Result<Vec<serde_json::Value>, String> = if format == "mlx" {
                    crate::hf::hf_list_mlx_files(&repo_id).await
                } else {
                    match crate::hf::hf_list_gguf_files(&repo_id).await {
                        Ok(files) => Ok(files.into_iter().map(|f| serde_json::json!(f)).collect()),
                        Err(e) => Err(e),
                    }
                };

                match result {
                    Ok(files) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": true,
                            "files": files
                        })),
                    )),
                    Err(e) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": e
                        })),
                    )),
                }
            }
        })
}

fn api_hf_community_picks(
    _state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "hf" / "community-picks")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let cfg = app_config.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }
                let path = cfg.config_dir.join("community-picks.json");
                let body = if path.exists() {
                    match std::fs::read_to_string(&path) {
                        Ok(s) => match serde_json::from_str::<serde_json::Value>(&s) {
                            Ok(v) => serde_json::json!({ "ok": true, "data": v }),
                            Err(e) => serde_json::json!({
                                "ok": false,
                                "error": format!("community-picks.json parse error: {e}")
                            }),
                        },
                        Err(e) => serde_json::json!({
                            "ok": false,
                            "error": format!("community-picks.json read error: {e}")
                        }),
                    }
                } else {
                    serde_json::json!({ "ok": true, "data": null })
                };
                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                    &body,
                )))
            }
        })
}

fn api_hf_quantizers(
    _state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "hf" / "quantizers")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let cfg = app_config.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }
                // The quantizer list is now a derived view over the typed community-source
                // catalog rather than a separate KnownQuantizer struct. This ensures:
                // - Single source of truth for who does what
                // - User edits in the catalog are respected
                // - Existing quick-pick behavior (quant_style, etc) is preserved
                let catalog =
                    crate::models::community_source_catalog::load_catalog(&cfg.config_dir);
                let quantizers =
                    crate::models::community_source_catalog::to_quantizers(&catalog);
                // is_custom is false: the list is catalog-derived, not user-overridden
                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                    &serde_json::json!({ "ok": true, "quantizers": quantizers, "is_custom": false }),
                )))
            }
        })
}

fn api_hf_quantizers_put(
    _state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "hf" / "quantizers")
        .and(warp::put())
        .and(warp::header::optional::<String>("authorization"))
        .and(
            warp::body::content_length_limit(256 * 1024)
                .and(warp::body::json::<Vec<crate::hf::UserQuantizer>>()),
        )
        .and_then(
            move |auth: Option<String>, body: Vec<crate::hf::UserQuantizer>| {
                let cfg = app_config.clone();
                async move {
                    if !check_api_token(&auth, &cfg) {
                        return Ok(unauthorized_api_token());
                    }

                    // Empty list = reset to defaults (remove legacy file + reset catalog)
                    if body.is_empty() {
                        let _ = std::fs::remove_file(cfg.config_dir.join("hf-quantizers.json"));
                        return match crate::models::community_source_catalog::reset_catalog(
                            &cfg.config_dir,
                        ) {
                            Ok(_) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                warp::reply::json(
                                    &serde_json::json!({ "ok": true, "reset": true }),
                                ),
                            )),
                            Err(e) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                warp::reply::json(
                                    &serde_json::json!({ "ok": false, "error": format!("{e}") }),
                                ),
                            )),
                        };
                    }

                    // A non-empty legacy payload is a replacement list, not a partial update.
                    // The catalog helper preserves bundled role metadata and reports conflicts
                    // instead of silently accepting a payload it could not apply.
                    let mut catalog =
                        crate::models::community_source_catalog::load_catalog(&cfg.config_dir);
                    if let Err(e) = crate::models::community_source_catalog::replace_quantizers(
                        &mut catalog,
                        body,
                    ) {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(
                                &serde_json::json!({ "ok": false, "error": format!("{e}") }),
                            ),
                        ));
                    }

                    match crate::models::community_source_catalog::save_catalog(
                        &cfg.config_dir,
                        &catalog,
                    ) {
                        Ok(()) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({ "ok": true })),
                        )),
                        Err(e) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(
                                &serde_json::json!({ "ok": false, "error": format!("{e}") }),
                            ),
                        )),
                    }
                }
            },
        )
}

// ── Community source catalog ──────────────────────────────────────────────────
//
// The catalog module shipped with load/save/upsert/remove/reset and no caller under
// `src/web/`, so "user-editable" was a property of the code and not of the product: the
// frontend's role badges came from a hardcoded regex list instead. These are the endpoints
// that make the on-disk catalog the actual source of truth.
//
// The role gates in `upsert_entry` (an original author can never also be registered as a
// converter for the same username) are enforced there, not here, so they hold for any caller.

fn api_hf_community_sources_get(
    _state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "hf" / "community-sources")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let cfg = app_config.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }
                use crate::models::community_source_catalog::CommunitySourceRole;
                let catalog =
                    crate::models::community_source_catalog::load_catalog(&cfg.config_dir);
                // The role vocabulary ships with the catalog so badge labels and their
                // explanations have one source of truth in the enum, not a parallel list in JS.
                let roles: Vec<_> = CommunitySourceRole::ALL
                    .iter()
                    .map(|r| {
                        serde_json::json!({
                            "id": r,
                            "label": r.label(),
                            "description": r.description(),
                        })
                    })
                    .collect();
                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                    &serde_json::json!({ "ok": true, "catalog": catalog, "roles": roles }),
                )))
            }
        })
}

fn api_hf_community_sources_put(
    _state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "hf" / "community-sources")
        .and(warp::put())
        .and(warp::header::optional::<String>("authorization"))
        .and(warp::body::content_length_limit(512 * 1024).and(warp::body::json::<
            crate::models::community_source_catalog::CommunitySourceCatalog,
        >()))
        .and_then(
            move |auth: Option<String>,
                  body: crate::models::community_source_catalog::CommunitySourceCatalog| {
                let cfg = app_config.clone();
                async move {
                    if !check_api_token(&auth, &cfg) {
                        return Ok(unauthorized_api_token());
                    }
                    match crate::models::community_source_catalog::save_catalog(
                        &cfg.config_dir,
                        &body,
                    ) {
                        Ok(()) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({ "ok": true, "catalog": body })),
                        )),
                        Err(e) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(
                                &serde_json::json!({ "ok": false, "error": format!("{e}") }),
                            ),
                        )),
                    }
                }
            },
        )
}

fn api_hf_community_sources_entry_post(
    _state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "hf" / "community-sources" / "entry")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(warp::body::content_length_limit(64 * 1024).and(warp::body::json::<
            crate::models::community_source_catalog::CommunitySourceEntry,
        >()))
        .and_then(
            move |auth: Option<String>,
                  entry: crate::models::community_source_catalog::CommunitySourceEntry| {
                let cfg = app_config.clone();
                async move {
                    if !check_api_token(&auth, &cfg) {
                        return Ok(unauthorized_api_token());
                    }
                    let mut catalog =
                        crate::models::community_source_catalog::load_catalog(&cfg.config_dir);
                    // A user-added entry is never bundled, whatever the request claims --
                    // otherwise `remove_entry` would refuse to delete it later.
                    let mut entry = entry;
                    entry.bundled = false;
                    let saved = match crate::models::community_source_catalog::upsert_entry(
                        &mut catalog,
                        entry,
                    ) {
                        Ok(e) => e,
                        Err(e) => {
                            return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                warp::reply::json(
                                    &serde_json::json!({ "ok": false, "error": format!("{e}") }),
                                ),
                            ));
                        }
                    };
                    match crate::models::community_source_catalog::save_catalog(
                        &cfg.config_dir,
                        &catalog,
                    ) {
                        Ok(()) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({ "ok": true, "entry": saved })),
                        )),
                        Err(e) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(
                                &serde_json::json!({ "ok": false, "error": format!("{e}") }),
                            ),
                        )),
                    }
                }
            },
        )
}

#[derive(serde::Deserialize)]
struct CommunitySourceEntryRef {
    username: String,
    role: crate::models::community_source_catalog::CommunitySourceRole,
}

fn api_hf_community_sources_entry_delete(
    _state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "hf" / "community-sources" / "entry")
        .and(warp::delete())
        .and(warp::header::optional::<String>("authorization"))
        .and(warp::query::<CommunitySourceEntryRef>())
        .and_then(move |auth: Option<String>, q: CommunitySourceEntryRef| {
            let cfg = app_config.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }
                let mut catalog =
                    crate::models::community_source_catalog::load_catalog(&cfg.config_dir);
                // Bundled entries survive this by design; the caller gets removed:false rather
                // than an error, and `reset` is the way back to the shipped set.
                let removed = crate::models::community_source_catalog::remove_entry(
                    &mut catalog,
                    &q.username,
                    q.role,
                );
                if removed
                    && let Err(e) = crate::models::community_source_catalog::save_catalog(
                        &cfg.config_dir,
                        &catalog,
                    )
                {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(
                            &serde_json::json!({ "ok": false, "error": format!("{e}") }),
                        ),
                    ));
                }
                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                    &serde_json::json!({ "ok": true, "removed": removed }),
                )))
            }
        })
}

fn api_hf_community_sources_reset(
    _state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "hf" / "community-sources" / "reset")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let cfg = app_config.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }
                // Preferences are editorial, not catalog data, so reset keeps them.
                match crate::models::community_source_catalog::reset_catalog(&cfg.config_dir) {
                    Ok(catalog) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({ "ok": true, "catalog": catalog })),
                    )),
                    Err(e) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(
                            &serde_json::json!({ "ok": false, "error": format!("{e}") }),
                        ),
                    )),
                }
            }
        })
}

fn api_hf_download_dir(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "hf" / "download-dir")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let cfg = app_config.clone();
            let st = state.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }
                let dir =
                    get_effective_models_dir(&st).unwrap_or_else(|| cfg.default_models_dir.clone());
                let configured = get_effective_models_dir(&st).is_some();
                let external_hf_cache = crate::models::external_cache::shared_hub();
                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                    &serde_json::json!({
                        "ok": true,
                        "dir": dir.to_string_lossy(),
                        "configured": configured,
                        "locations": {
                            "library_root": dir,
                            "gguf": dir.join("gguf"),
                            "mlx_native": dir.join("mlx/native"),
                            "managed_hf_cache": dir.join("cache/huggingface/hub"),
                            "external_hf_cache": external_hf_cache,
                        }
                    }),
                )))
            }
        })
}

fn api_hf_card(
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "hf" / "card")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and(warp::query::<std::collections::HashMap<String, String>>())
        .and_then(
            move |auth: Option<String>, params: std::collections::HashMap<String, String>| {
                let cfg = app_config.clone();
                async move {
                    if !check_api_token(&auth, &cfg) {
                        return Ok(unauthorized_api_token());
                    }

                    let repo = match params.get("repo") {
                        Some(r) if !r.is_empty() => r.clone(),
                        _ => {
                            return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                warp::reply::with_status(
                                    warp::reply::json(&serde_json::json!({
                                        "error": "Missing required query param: repo"
                                    })),
                                    warp::http::StatusCode::BAD_REQUEST,
                                ),
                            ));
                        }
                    };

                    // Basic path-traversal guard: repo must be "owner/name" with no dots or slashes beyond that
                    let parts: Vec<&str> = repo.splitn(3, '/').collect();
                    if parts.len() != 2 || parts.iter().any(|p| p.is_empty() || p.contains("..")) {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::with_status(
                                warp::reply::json(
                                    &serde_json::json!({ "error": "Invalid repo id" }),
                                ),
                                warp::http::StatusCode::BAD_REQUEST,
                            ),
                        ));
                    }

                    let url = format!("https://huggingface.co/{}/raw/main/README.md", repo);

                    let builder = reqwest::Client::builder()
                        .timeout(std::time::Duration::from_secs(20))
                        .user_agent("llama-monitor");

                    let client = match builder.build() {
                        Ok(c) => c,
                        Err(e) => {
                            return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                warp::reply::json(&serde_json::json!({ "error": e.to_string() })),
                            ));
                        }
                    };

                    let mut req = client.get(&url);
                    if let Some(token) = crate::hf::hf_load_token() {
                        req = req.header("Authorization", format!("Bearer {}", token));
                    }

                    let resp = match req.send().await {
                        Ok(r) => r,
                        Err(e) => {
                            return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                warp::reply::json(&serde_json::json!({ "error": e.to_string() })),
                            ));
                        }
                    };

                    if resp.status() == reqwest::StatusCode::NOT_FOUND {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({ "markdown": "" })),
                        ));
                    }

                    if !resp.status().is_success() {
                        let status = resp.status().as_u16();
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({
                                "error": format!("HuggingFace returned HTTP {}", status)
                            })),
                        ));
                    }

                    // Cap at 256 KB — large READMEs still render, but we don't buffer unlimited data
                    let bytes = match resp.bytes().await {
                        Ok(b) => b,
                        Err(e) => {
                            return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                warp::reply::json(&serde_json::json!({ "error": e.to_string() })),
                            ));
                        }
                    };
                    let markdown =
                        String::from_utf8_lossy(&bytes[..bytes.len().min(256 * 1024)]).into_owned();

                    Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                        &serde_json::json!({ "markdown": markdown }),
                    )))
                }
            },
        )
}

fn api_hf_meta(
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "hf" / "meta")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and(warp::query::<std::collections::HashMap<String, String>>())
        .and_then(
            move |auth: Option<String>, params: std::collections::HashMap<String, String>| {
                let cfg = app_config.clone();
                async move {
                    if !check_api_token(&auth, &cfg) {
                        return Ok(unauthorized_api_token());
                    }
                    let repo = match params.get("repo") {
                        Some(r) if !r.is_empty() => r.clone(),
                        _ => {
                            return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                warp::reply::json(
                                    &serde_json::json!({"ok": false, "error": "missing repo param"}),
                                ),
                            ));
                        }
                    };
                    let parts: Vec<&str> = repo.splitn(3, '/').collect();
                    if parts.len() != 2 || parts.iter().any(|p| p.is_empty() || p.contains("..")) {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(
                                &serde_json::json!({"ok": false, "error": "invalid repo id"}),
                            ),
                        ));
                    }
                    match crate::hf::hf_get_model_info(&repo).await {
                        Ok(info) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({
                                "ok": true,
                                "tags": info.tags,
                                "gated": info.gated,
                                "private": info.private,
                            })),
                        )),
                        Err(e) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(
                                &serde_json::json!({"ok": false, "error": e.to_string()}),
                            ),
                        )),
                    }
                }
            },
        )
}

/// Read-only preflight for an external speculative-decoding (MTP) companion
/// model: resolves `repo` to its pinned commit revision and reports whether
/// it would require `trust_remote_code`. Does not download weights and does
/// not grant consent — the launch-time gate in
/// `inference::rapid_mlx::command::validate_trust_consent` still fail-closes
/// independently of this result. Uses the MTP pin cache to avoid redundant HF
/// API calls. Returns cached data if present and not stale; otherwise resolves
/// via HF API, caches the result, then returns.
fn api_hf_mtp_preflight(
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "hf" / "mtp-preflight")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and(warp::query::<std::collections::HashMap<String, String>>())
        .and_then(
            move |auth: Option<String>, params: std::collections::HashMap<String, String>| {
                let cfg = app_config.clone();
                async move {
                    if !check_api_token(&auth, &cfg) {
                        return Ok(unauthorized_api_token());
                    }
                    let repo = match params.get("repo") {
                        Some(r) if !r.is_empty() => r.clone(),
                        _ => {
                            return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                warp::reply::with_status(
                                    warp::reply::json(&serde_json::json!({
                                        "ok": false, "error": "missing repo param"
                                    })),
                                    StatusCode::BAD_REQUEST,
                                ),
                            ));
                        }
                    };
                    if !validate_hf_repo_id(&repo) {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::with_status(
                                warp::reply::json(&serde_json::json!({
                                    "ok": false,
                                    "error": "Invalid repo id format. Expected: owner/repo"
                                })),
                                StatusCode::BAD_REQUEST,
                            ),
                        ));
                    }

                    let cache = crate::hf::mtp_pin_cache::pin_cache();
                    let force_recheck = params.get("force").map(|v| v == "1").unwrap_or(false);

                    // Check cache first (unless force-recheck)
                    if !force_recheck && let Some(mut pin) = cache.get(&repo) {
                        // Refresh a stale pin before replying, so the revision, consent state,
                        // and memory metadata shown to the user describe the same cached pin
                        // that launch validation will consult.
                        let mut stale = pin.is_stale();
                        if stale
                            && let Ok(fresh) =
                                crate::hf::resolve_speculative_model_preflight(&repo).await
                        {
                            let mut new_pin = pin.clone();
                            new_pin.refresh_from_preflight(&fresh);
                            let _ = cache.insert(new_pin);
                            pin = cache.get(&repo).unwrap_or(pin);
                            stale = pin.is_stale();
                        }
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({
                                "ok": true,
                                "repoId": pin.repo_id,
                                "revision": pin.revision,
                                "trustRemoteCodeRequired": pin.trust_remote_code_required,
                                "estimatedMemoryBytes": pin.estimated_memory_bytes,
                                "mtpSidecar": pin.mtp_sidecar,
                                "mtpDepthMax": pin.mtp_depth_max,
                                "resolvedAt": pin.resolved_at,
                                "lastRecheckAt": pin.last_recheck_at,
                                "upstreamUnchanged": pin.upstream_unchanged,
                                "stale": stale,
                            })),
                        ));
                    }

                    // Resolve via HF API
                    match crate::hf::resolve_speculative_model_preflight(&repo).await {
                        Ok(preflight) => {
                            // Cache the result
                            let pin = crate::hf::mtp_pin_cache::MtpPin {
                                repo_id: preflight.repo_id.clone(),
                                revision: preflight.revision.clone(),
                                trust_remote_code_required: preflight.trust_remote_code_required,
                                resolved_at: chrono::Utc::now().to_rfc3339(),
                                last_recheck_at: String::new(),
                                upstream_unchanged: None,
                                estimated_memory_bytes: preflight.estimated_memory_bytes,
                                mtp_sidecar: preflight.mtp_sidecar.clone(),
                                mtp_depth_max: preflight.mtp_depth_max,
                            };
                            let _ = cache.insert(pin);

                            Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                warp::reply::json(&serde_json::json!({
                                    "ok": true,
                                    "repoId": preflight.repo_id,
                                    "revision": preflight.revision,
                                    "trustRemoteCodeRequired": preflight.trust_remote_code_required,
                                    "estimatedMemoryBytes": preflight.estimated_memory_bytes,
                                    "mtpSidecar": preflight.mtp_sidecar,
                                    "mtpDepthMax": preflight.mtp_depth_max,
                                    "resolvedAt": chrono::Utc::now().to_rfc3339(),
                                    "lastRecheckAt": "",
                                    "upstreamUnchanged": null,
                                    "stale": false,
                                })),
                            ))
                        }
                        Err(e) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({"ok": false, "error": e})),
                        )),
                    }
                }
            },
        )
}

/// Re-check a cached MTP companion pin against upstream: re-resolve the repo
/// via HF API and compare the sha. Updates the pin in the cache with the new
/// revision if the upstream has changed.
fn api_hf_mtp_preflight_recheck(
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "hf" / "mtp-preflight" / "recheck")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(warp::query::<std::collections::HashMap<String, String>>())
        .and_then(
            move |auth: Option<String>, params: std::collections::HashMap<String, String>| {
                let cfg = app_config.clone();
                async move {
                    if !check_api_token(&auth, &cfg) {
                        return Ok(unauthorized_api_token());
                    }
                    let repo = match params.get("repo") {
                        Some(r) if !r.is_empty() => r.clone(),
                        _ => {
                            return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                warp::reply::with_status(
                                    warp::reply::json(&serde_json::json!({
                                        "ok": false, "error": "missing repo param"
                                    })),
                                    StatusCode::BAD_REQUEST,
                                ),
                            ));
                        }
                    };
                    if !validate_hf_repo_id(&repo) {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::with_status(
                                warp::reply::json(&serde_json::json!({
                                    "ok": false,
                                    "error": "Invalid repo id format. Expected: owner/repo"
                                })),
                                StatusCode::BAD_REQUEST,
                            ),
                        ));
                    }

                    let cache = crate::hf::mtp_pin_cache::pin_cache();
                    match cache.recheck(&repo).await {
                        Ok(pin) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({
                                "ok": true,
                                "repoId": pin.repo_id,
                                "revision": pin.revision,
                                "trustRemoteCodeRequired": pin.trust_remote_code_required,
                                "estimatedMemoryBytes": pin.estimated_memory_bytes,
                                "mtpSidecar": pin.mtp_sidecar,
                                "mtpDepthMax": pin.mtp_depth_max,
                                "resolvedAt": pin.resolved_at,
                                "lastRecheckAt": pin.last_recheck_at,
                                "upstreamUnchanged": pin.upstream_unchanged,
                            })),
                        )),
                        Err(e) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(
                                &serde_json::json!({"ok": false, "error": e.to_string()}),
                            ),
                        )),
                    }
                }
            },
        )
}

/// Return all cached MTP companion model pins.
fn api_hf_mtp_pins(
    _app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "hf" / "mtp-pins")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let cfg = _app_config.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }
                let cache = crate::hf::mtp_pin_cache::pin_cache();
                let pins = cache.all();
                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                    &serde_json::json!({
                        "ok": true,
                        "pins": pins.iter().map(|p| serde_json::json!({
                            "repoId": p.repo_id,
                            "revision": p.revision,
                            "trustRemoteCodeRequired": p.trust_remote_code_required,
                            "estimatedMemoryBytes": p.estimated_memory_bytes,
                            "mtpSidecar": p.mtp_sidecar,
                            "mtpDepthMax": p.mtp_depth_max,
                            "resolvedAt": p.resolved_at,
                            "lastRecheckAt": p.last_recheck_at,
                            "upstreamUnchanged": p.upstream_unchanged,
                            "stale": p.is_stale(),
                        })).collect::<Vec<_>>(),
                    }),
                )))
            }
        })
}

/// Remove a cached MTP companion model pin.
/// List all locally-built MTP sidecars under the managed sidecar root.
/// Returns each sidecar's slug, path, whether weights exist, and parsed
/// provenance data (trunk, build date, norm check status, VRAM estimate).
fn api_hf_mtp_sidecars(
    _app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "hf" / "mtp-sidecars")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let cfg = _app_config.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }

                match crate::inference::rapid_mlx::sidecar_inventory::discover_sidecars() {
                    Ok(entries) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": true,
                            "sidecars": entries,
                        })),
                    )),
                    Err(e) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": e.to_string(),
                        })),
                    )),
                }
            }
        })
}

fn api_hf_mtp_pin_remove(
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "hf" / "mtp-pins" / String)
        .and(warp::delete())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |repo: String, auth: Option<String>| {
            let cfg = app_config.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }
                if !validate_hf_repo_id(&repo) {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": format!("Invalid repo id: {}", repo),
                        })),
                    ));
                }
                let cache = crate::hf::mtp_pin_cache::pin_cache();
                match cache.remove(&repo) {
                    Ok(()) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": true,
                            "repoId": repo,
                        })),
                    )),
                    Err(e) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": e.to_string(),
                        })),
                    )),
                }
            }
        })
}

fn api_hf_resolve_origin(
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "hf" / "resolve-origin")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<serde_json::Value>())
        .and_then(move |auth: Option<String>, body: serde_json::Value| {
            let cfg = app_config.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }

                let filename = body["filename"].as_str().unwrap_or("").trim().to_string();
                let size_bytes: u64 = body["size_bytes"].as_u64().unwrap_or(0);

                if filename.is_empty() {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(
                            &serde_json::json!({"ok": false, "error": "filename is required"}),
                        ),
                    ));
                }

                match crate::hf::hf_resolve_origin(&filename, size_bytes).await {
                    Ok(result) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": true,
                            "confident": result.confident,
                            "candidates": result.candidates,
                            "model_stem": result.model_stem,
                            "errors": result.errors,
                        })),
                    )),
                    Err(e) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": e,
                        })),
                    )),
                }
            }
        })
}

fn api_hf_token_get(
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "hf" / "token")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let cfg = app_config.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }
                let set = crate::hf::hf_load_token().is_some();
                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                    &serde_json::json!({ "set": set }),
                )))
            }
        })
}

fn api_hf_token_put(
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "hf" / "token")
        .and(warp::put())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::hf_json_body::<serde_json::Value>())
        .and_then(move |auth: Option<String>, body: serde_json::Value| {
            let cfg = app_config.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }
                let token = body["token"].as_str().unwrap_or("").trim().to_string();
                if token.is_empty() {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(
                            &serde_json::json!({ "ok": false, "error": "token is required" }),
                        ),
                    ));
                }
                match crate::hf::hf_save_token(&token) {
                    Ok(()) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({ "ok": true })),
                    )),
                    Err(e) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(
                            &serde_json::json!({ "ok": false, "error": e.to_string() }),
                        ),
                    )),
                }
            }
        })
}

fn api_hf_token_delete(
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "hf" / "token")
        .and(warp::delete())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let cfg = app_config.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }
                let token_path = cfg.config_dir.join("hf-token");
                if token_path.exists() {
                    let _ = std::fs::remove_file(&token_path);
                }
                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                    &serde_json::json!({ "ok": true }),
                )))
            }
        })
}

fn api_hf_download(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    use std::sync::atomic::AtomicU64;
    static HF_DOWNLOAD_LAST_START: AtomicU64 = AtomicU64::new(0);

    warp::path!("api" / "hf" / "download")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::hf_json_body::<serde_json::Value>())
        .and_then(move |auth: Option<String>, body: serde_json::Value| {
            let cfg = app_config.clone();
            let state = state.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }

                let repo_id = body["repo_id"].as_str().unwrap_or("").trim().to_string();
                let file_path = body["file_path"].as_str().unwrap_or("").trim().to_string();
                let target_path: Option<String> = body["target_path"]
                    .as_str()
                    .map(|s| s.trim().to_string())
                    .or_else(|| {
                        if file_path.to_ascii_lowercase().ends_with(".gguf") {
                            Some("gguf".to_string())
                        } else {
                            Some("transformers".to_string())
                        }
                    });
                let save_as: Option<String> =
                    body["save_as"].as_str().map(|s| s.trim().to_string());
                let resume: bool = body["resume"].as_bool().unwrap_or(false);
                // Companion downloads (e.g. mmproj alongside a model) bypass the
                // 10-second cooldown so both files can start simultaneously.
                let companion: bool = body["companion"].as_bool().unwrap_or(false);

                if repo_id.is_empty() || file_path.is_empty() {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": "Missing 'repo_id' or 'file_path'"
                        })),
                    ));
                }
                if !validate_hf_repo_id(&repo_id) {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({
                                "ok": false,
                                "error": "Invalid repo_id format. Expected: owner/repo"
                            })),
                            StatusCode::BAD_REQUEST,
                        ),
                    ));
                }

                // Path traversal guard: reject "..", leading "/", leading "\\"
                if file_path.contains("..")
                    || file_path.starts_with('/')
                    || file_path.starts_with("\\")
                {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": "Invalid file_path: path traversal not allowed"
                        })),
                    ));
                }
                if save_as
                    .as_ref()
                    .is_some_and(|sa| sa.contains("..") || sa.contains('/') || sa.contains('\\'))
                {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": "Invalid save_as: must be a plain filename"
                        })),
                    ));
                }

                // Determine target directory.
                let models_dir = get_effective_models_dir(&state)
                    .unwrap_or_else(|| cfg.default_models_dir.clone());
                let target_dir = match resolve_hf_target_dir(&models_dir, target_path.as_deref()) {
                    Ok(path) => path,
                    Err(error) => {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({
                                "ok": false,
                                "error": error
                            })),
                        ));
                    }
                };

                // Cooldown between starts: 10 seconds. Companion downloads (e.g.
                // mmproj alongside a model) are exempt so both can start together.
                // If the previous download failed, allow immediate retry (no cooldown).
                if !companion {
                    let now = std::time::SystemTime::UNIX_EPOCH
                        .elapsed()
                        .unwrap_or_default()
                        .as_secs();

                    // Skip cooldown if the previous download failed (same or different file).
                    use std::sync::atomic::Ordering;
                    let last_failed = crate::model_download::MODEL_DOWNLOAD_LAST_FAILED
                        .load(Ordering::Relaxed);
                    let cooldown_ok = last_failed > 0
                        && now.saturating_sub(last_failed) < 60;

                    if !cooldown_ok {
                        let (dl_ok, _) = try_cooldown(&HF_DOWNLOAD_LAST_START, now, 10);
                        if !dl_ok {
                            return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                warp::reply::with_status(
                                    warp::reply::json(&serde_json::json!({
                                        "ok": false,
                                        "error": "Too soon; please wait 10 seconds between downloads."
                                    })),
                                    StatusCode::TOO_MANY_REQUESTS,
                                ),
                            ));
                        }
                    }
                }

                let effective_filename = save_as.as_deref().unwrap_or(&file_path);
                let local_path = target_dir
                    .join(effective_filename)
                    .to_string_lossy()
                    .into_owned();
                match crate::hf::hf_start_download(
                    &repo_id,
                    &file_path,
                    save_as.as_deref(),
                    &target_dir,
                    resume,
                ) {
                    Ok(download_id) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(
                        Box::new(warp::reply::json(&serde_json::json!({
                            "ok": true,
                            "download_id": download_id,
                            "local_path": local_path
                        }))),
                    ),
                    Err(e) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": e
                        })),
                    )),
                }
            }
        })
}

/// POST /api/hf/mlx-derivatives — discover MLX derivatives for a source repo.
/// Input: repoId (string), revision (string, optional).
/// Output: native_mlx_derivatives, conversion_recipes, original_author_preserved, etc.
fn api_hf_mlx_derivatives(
    config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "hf" / "mlx-derivatives")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(
            warp::body::content_length_limit(64 * 1024)
                .and(warp::body::json::<serde_json::Value>()),
        )
        .and_then(move |auth: Option<String>, body: serde_json::Value| {
            let cfg = config.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }
                let repo_id = body
                    .get("repoId")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                // revision is accepted for future use (pinned source revision tracking)
                let _revision = body
                    .get("revision")
                    .and_then(|v| v.as_str())
                    .unwrap_or("main")
                    .to_string();

                if !crate::hf::validate_hf_repo_id(&repo_id) {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({ "error": "Invalid repoId" })),
                    ));
                }

                match crate::hf::hf_discover_mlx_derivatives(&repo_id).await {
                    Ok(result) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&result),
                    )),
                    Err(e) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({ "error": e })),
                            StatusCode::INTERNAL_SERVER_ERROR,
                        ),
                    )),
                }
            }
        })
}

/// POST /api/hf/qualify — authoritative qualification for a repo+revision.
/// Input: repoId (string), revision (string, optional), backend (string, optional).
/// Output: HfQualification with backend_hint, qualification_state, etc.
fn api_hf_qualify(
    config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "hf" / "qualify")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<crate::hf::QualifyRequest>())
        .and_then(
            move |auth: Option<String>, req: crate::hf::QualifyRequest| {
                let cfg = config.clone();
                async move {
                    if !check_api_token(&auth, &cfg) {
                        return Ok(unauthorized_api_token());
                    }
                    let _permit = match HF_EVIDENCE_GATE.clone().try_acquire_owned() {
                        Ok(permit) => permit,
                        Err(_) => {
                            return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                warp::reply::with_status(
                                    warp::reply::json(&serde_json::json!({
                                        "error": "HF evidence resolution is busy; try again shortly"
                                    })),
                                    StatusCode::TOO_MANY_REQUESTS,
                                ),
                            ));
                        }
                    };
                    match tokio::time::timeout(
                        HF_EVIDENCE_TIMEOUT,
                        crate::hf::qualify::hf_qualify_repo(req),
                    )
                    .await
                    {
                        Err(_) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::with_status(
                                warp::reply::json(&serde_json::json!({
                                    "error": "HF qualification timed out"
                                })),
                                StatusCode::GATEWAY_TIMEOUT,
                            ),
                        )),
                        Ok(Ok(result)) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(
                            Box::new(warp::reply::json(&result)),
                        ),
                        Ok(Err(e)) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::with_status(
                                warp::reply::json(&serde_json::json!({ "error": e })),
                                StatusCode::BAD_REQUEST,
                            ),
                        )),
                    }
                }
            },
        )
}

/// POST /api/hf/identity — authorship and lineage resolution.
/// Input: repoId (string), revision (string, optional).
/// Output: HfIdentity with original_author, converter_role, roles, etc.
fn api_hf_identity(
    config: Arc<AppConfig>,
) -> impl Filter<Extract = (Box<dyn warp::reply::Reply>,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "hf" / "identity")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(super::super::safe_json_body::<crate::hf::IdentityRequest>())
        .and_then(
            move |auth: Option<String>, req: crate::hf::IdentityRequest| {
                let cfg = config.clone();
                async move {
                    if !check_api_token(&auth, &cfg) {
                        return Ok(unauthorized_api_token());
                    }
                    let _permit = match HF_EVIDENCE_GATE.clone().try_acquire_owned() {
                        Ok(permit) => permit,
                        Err(_) => {
                            return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                warp::reply::with_status(
                                    warp::reply::json(&serde_json::json!({
                                        "error": "HF evidence resolution is busy; try again shortly"
                                    })),
                                    StatusCode::TOO_MANY_REQUESTS,
                                ),
                            ));
                        }
                    };
                    match tokio::time::timeout(
                        HF_EVIDENCE_TIMEOUT,
                        crate::hf::qualify::hf_resolve_identity(req, &cfg.config_dir),
                    )
                    .await
                    {
                        Err(_) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::with_status(
                                warp::reply::json(&serde_json::json!({
                                    "error": "HF identity resolution timed out"
                                })),
                                StatusCode::GATEWAY_TIMEOUT,
                            ),
                        )),
                        Ok(Ok(result)) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(
                            Box::new(warp::reply::json(&result)),
                        ),
                        Ok(Err(e)) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::with_status(
                                warp::reply::json(&serde_json::json!({ "error": e })),
                                StatusCode::BAD_REQUEST,
                            ),
                        )),
                    }
                }
            },
        )
}

pub(crate) fn routes(ctx: ApiCtx) -> ApiRoute {
    let state = ctx.state.clone();
    let config = ctx.config.clone();

    let mut r = api_hf_search(state.clone(), config.clone())
        .or(api_hf_files(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r
        .or(api_hf_community_picks(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r
        .or(api_hf_quantizers(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r
        .or(api_hf_quantizers_put(state.clone(), config.clone()))
        .unify()
        .boxed();
    // Longest paths first: `community-sources/entry` and `/reset` must be offered before the
    // bare `community-sources` filter, which would otherwise reject the sub-paths and end the
    // chain there.
    r = r
        .or(api_hf_community_sources_entry_post(
            state.clone(),
            config.clone(),
        ))
        .unify()
        .boxed();
    r = r
        .or(api_hf_community_sources_entry_delete(
            state.clone(),
            config.clone(),
        ))
        .unify()
        .boxed();
    r = r
        .or(api_hf_community_sources_reset(
            state.clone(),
            config.clone(),
        ))
        .unify()
        .boxed();
    r = r
        .or(api_hf_community_sources_get(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r
        .or(api_hf_community_sources_put(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r
        .or(api_hf_download_dir(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r.or(api_hf_card(config.clone())).unify().boxed();
    r = r.or(api_hf_meta(config.clone())).unify().boxed();
    r = r.or(api_hf_mtp_preflight(config.clone())).unify().boxed();
    r = r
        .or(api_hf_mtp_preflight_recheck(config.clone()))
        .unify()
        .boxed();
    r = r.or(api_hf_mtp_pins(config.clone())).unify().boxed();
    r = r.or(api_hf_mtp_pin_remove(config.clone())).unify().boxed();
    r = r.or(api_hf_mtp_sidecars(config.clone())).unify().boxed();
    r = r.or(api_hf_resolve_origin(config.clone())).unify().boxed();
    r = r.or(api_hf_token_get(config.clone())).unify().boxed();
    r = r.or(api_hf_token_put(config.clone())).unify().boxed();
    r = r.or(api_hf_token_delete(config.clone())).unify().boxed();
    r = r
        .or(api_hf_download(state.clone(), config.clone()))
        .unify()
        .boxed();
    r = r.or(api_hf_mlx_derivatives(config.clone())).unify().boxed();
    r = r.or(api_hf_qualify(config.clone())).unify().boxed();
    r = r.or(api_hf_identity(config.clone())).unify().boxed();
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::web::auth::AuthManager;

    fn test_routes() -> ApiRoute {
        let config = Arc::new(AppConfig::for_test(Some("api-secret".to_string()), None));
        routes(ApiCtx {
            state: AppState::default(),
            auth: AuthManager::new(None, None, &crate::config::TLSConfig::default().mode),
            config,
        })
    }

    /// Covers the `format` threading from an `/api/hf/search` request body
    /// through to the `HfModelFormat` passed into `HfSearchParams` (and, via
    /// `HfModelFormat::as_api_filter`, to the outgoing `filter=` HF API query
    /// param) without a live HF call.
    #[test]
    fn parse_hf_format_param_threads_mlx_and_gguf() {
        assert!(matches!(
            parse_hf_format_param("mlx"),
            crate::hf::HfModelFormat::Mlx
        ));
        // Case-insensitive.
        assert!(matches!(
            parse_hf_format_param("MLX"),
            crate::hf::HfModelFormat::Mlx
        ));
        assert!(matches!(
            parse_hf_format_param("gguf"),
            crate::hf::HfModelFormat::Gguf
        ));
        assert!(matches!(
            parse_hf_format_param("both"),
            crate::hf::HfModelFormat::Both
        ));
        assert!(matches!(
            parse_hf_format_param("ALL"),
            crate::hf::HfModelFormat::Both
        ));
        // Unrecognized / absent format falls back to Gguf (backward compat).
        assert!(matches!(
            parse_hf_format_param(""),
            crate::hf::HfModelFormat::Gguf
        ));
        assert!(matches!(
            parse_hf_format_param("bogus"),
            crate::hf::HfModelFormat::Gguf
        ));
    }

    #[tokio::test]
    async fn evidence_routes_require_api_token() {
        for path in ["/api/hf/qualify", "/api/hf/identity"] {
            let response = warp::test::request()
                .method("POST")
                .path(path)
                .header("content-type", "application/json")
                .body(r#"{"repoId":"owner/model"}"#)
                .reply(&test_routes())
                .await;
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED, "{path}");
        }
    }

    #[tokio::test]
    async fn evidence_routes_return_bad_request_for_malformed_json() {
        let routes = test_routes().recover(crate::web::handle_rejection);
        for path in ["/api/hf/qualify", "/api/hf/identity"] {
            let response = warp::test::request()
                .method("POST")
                .path(path)
                .header("authorization", "Bearer api-secret")
                .header("content-type", "application/json")
                .body("{")
                .reply(&routes)
                .await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{path}");
        }
    }
}
