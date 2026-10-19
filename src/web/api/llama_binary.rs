use std::path::Path;
use std::sync::Arc;

use warp::Filter;

use crate::config::AppConfig;
use crate::inference::launch::launch_local;
use crate::llama::llama_cpp_downloader::{
    LlamaCppRelease, ReleaseQuery, cleanup_old_binaries, download_and_extract,
    get_latest_stable_release, get_release_by_tag, list_releases, release_has_binary_assets,
    select_assets_for_os, sort_releases_by_published_at,
};
use crate::llama::server::stop_server;
use crate::llama_runtime::gate::{Busy, RuntimeOp};
use crate::llama_runtime::host::LiveHost;
use crate::llama_runtime::manifest::{LlamaCppRef, ManifestSource, RuntimeManifest};
use crate::llama_runtime::promote::{
    PromoteError, PromoteOutcome, PromoteRequest, PromoteResult, RuntimePaths, new_transaction_id,
    promote,
};
use crate::llama_runtime::verify::VerificationIntent;
use crate::paths::AppPaths;
use crate::state::AppState;
use crate::web::safe_json_body;

use super::{ApiCtx, ApiRoute, box_reply, check_api_token, unauthorized_api_token};

fn default_backend_for_os(os: &str) -> &'static str {
    match os {
        "macos" => "metal",
        "linux" => "cpu",
        _ => "avx2",
    }
}

// GitHub publishes release metadata before all platform archives finish
// uploading. Keep this short so a temporarily incomplete nightly cannot make
// a valid platform build look unavailable for the next half hour.
const RELEASE_LIST_CACHE_TTL: std::time::Duration = std::time::Duration::from_secs(5 * 60);

fn managed_llama_server_path(config: &AppConfig) -> std::path::PathBuf {
    let binary_name = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    let configured = config.llama_server_path.clone();

    // Explicit custom paths remain authoritative. The fallback is only for the
    // default managed path, which may be split between the canonical and legacy
    // roots while the 2.x migration is intentionally restartable.
    if configured.is_file() || configured != config.app_paths.bin_dir().join(binary_name) {
        return configured;
    }

    let mut candidates = vec![configured];
    for root in [
        config.app_paths.root.clone(),
        AppPaths::canonical_default_root(),
        AppPaths::legacy_default_root(),
    ] {
        let candidate = root.join("bin").join(binary_name);
        if !candidates.contains(&candidate) {
            candidates.push(candidate);
        }
    }

    candidates
        .into_iter()
        .find(|path| path.is_file())
        .unwrap_or_else(|| config.llama_server_path.clone())
}

pub(super) type ReplyBox = Box<dyn warp::reply::Reply>;

pub(super) fn json_failure(message: impl Into<String>) -> ReplyBox {
    Box::new(warp::reply::json(&serde_json::json!({
        "ok": false,
        "error": message.into(),
    })))
}

pub(super) fn json_busy(busy: Busy) -> ReplyBox {
    Box::new(warp::reply::with_status(
        warp::reply::json(&serde_json::json!({
            "ok": false,
            "error": format!(
                "The llama.cpp runtime is busy ({} in progress)",
                busy.owner.label()
            ),
            "owner": busy.owner,
        })),
        warp::http::StatusCode::CONFLICT,
    ))
}

/// The directory the runtime is promoted into, and the user's custom server
/// path when `llama_server_path` points outside the managed `bin/`.
///
/// Custom paths are not authoritative for promotions: builds always install to
/// the managed directory (plan: custom server path), and the response says so.
pub(super) fn managed_bin_dir(
    config: &AppConfig,
) -> (std::path::PathBuf, Option<std::path::PathBuf>) {
    let configured = managed_llama_server_path(config);
    let managed_roots = [
        config.app_paths.root.clone(),
        AppPaths::canonical_default_root(),
        AppPaths::legacy_default_root(),
    ];
    let managed_parent = configured.parent().filter(|parent| {
        parent.file_name().is_some_and(|n| n == "bin")
            && parent
                .parent()
                .is_some_and(|root| managed_roots.iter().any(|r| r == root))
    });
    match managed_parent {
        Some(parent) => (parent.to_path_buf(), None),
        None => (config.app_paths.bin_dir(), Some(configured)),
    }
}

/// Locate the server executable in an extracted release. Archives place it at
/// the root or inside a subdirectory (e.g. `llama-bXXXX-bin-...`).
fn find_binary(root: &Path, name: &str) -> Option<std::path::PathBuf> {
    let direct = root.join(name);
    if direct.is_file() {
        return Some(direct);
    }
    for entry in std::fs::read_dir(root).ok()? {
        let path = entry.ok()?.path();
        if path.is_dir()
            && let Some(found) = find_binary(&path, name)
        {
            return Some(found);
        }
    }
    None
}

/// Flatten an extracted release into a fresh staging directory.
fn stage_release_files(src: &Path, staging: &Path) -> std::io::Result<()> {
    fn copy_all(src: &Path, dest: &Path) -> std::io::Result<()> {
        for entry in std::fs::read_dir(src)?.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_dir() {
                copy_all(&path, dest)?;
            } else if let Some(name) = path.file_name() {
                std::fs::copy(&path, dest.join(name))?;
            }
        }
        Ok(())
    }
    if staging.exists() {
        std::fs::remove_dir_all(staging)?;
    }
    std::fs::create_dir_all(staging)?;
    copy_all(src, staging)
}

pub(super) struct PromoteReplyContext<'a> {
    pub tag: &'a str,
    pub backend: &'a str,
    pub arch: &'a str,
    pub sha256: Option<String>,
    pub restart_applicable: bool,
    pub custom_server_path: Option<String>,
}

pub(super) fn promote_reply(
    state: &AppState,
    result: Result<PromoteResult, PromoteError>,
    ctx: PromoteReplyContext<'_>,
) -> ReplyBox {
    let mut body = serde_json::json!({
        "version": ctx.tag,
        "backend": ctx.backend,
        "arch": ctx.arch,
        "restart_applicable": ctx.restart_applicable,
        "custom_server_path": ctx.custom_server_path,
    });
    let fields = body.as_object_mut().expect("object literal");
    let mut status = warp::http::StatusCode::OK;
    match result {
        Ok(PromoteResult { outcome, warnings }) => {
            fields.insert("warnings".into(), serde_json::json!(warnings));
            match outcome {
                PromoteOutcome::Promoted { restarted } => {
                    state.push_log(format!(
                        "[monitor] llama-binary/update: installed {} (restarted: {restarted})",
                        ctx.tag
                    ));
                    fields.insert("ok".into(), true.into());
                    fields.insert("outcome".into(), "promoted".into());
                    fields.insert("sha256".into(), serde_json::json!(ctx.sha256));
                    // True when the backend already restarted the server; the
                    // frontend must skip its own restart call.
                    fields.insert("server_restarted".into(), restarted.into());
                }
                PromoteOutcome::InstalledRestartFailed { restart_error } => {
                    state.push_log(format!(
                        "[monitor] llama-binary/update: installed {} but the model restart failed: {restart_error}",
                        ctx.tag
                    ));
                    fields.insert("ok".into(), true.into());
                    fields.insert("outcome".into(), "installed_restart_failed".into());
                    fields.insert("sha256".into(), serde_json::json!(ctx.sha256));
                    fields.insert("server_restarted".into(), false.into());
                    fields.insert("restart_error".into(), restart_error.into());
                }
                PromoteOutcome::RolledBack { reason } => {
                    state.push_log(format!(
                        "[monitor] llama-binary/update: {reason}; previous runtime restored"
                    ));
                    fields.insert("ok".into(), false.into());
                    fields.insert("outcome".into(), "rolled_back".into());
                    fields.insert("error".into(), reason.into());
                }
            }
        }
        Err(error) => {
            state.push_log(format!("[monitor] llama-binary/update: {error}"));
            if matches!(error, PromoteError::JournalBlocked(_)) {
                status = warp::http::StatusCode::CONFLICT;
            }
            fields.insert("ok".into(), false.into());
            fields.insert("outcome".into(), "failed".into());
            fields.insert("error".into(), error.to_string().into());
        }
    }
    Box::new(warp::reply::with_status(warp::reply::json(&body), status))
}

fn parse_llama_build(output: &str) -> Option<u64> {
    use regex::Regex;

    static BUILD_RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    static VERSION_RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();

    let build_re = BUILD_RE.get_or_init(|| Regex::new(r"(?i)\bbuild\s*[:=]?\s*(\d+)").unwrap());
    let version_re =
        VERSION_RE.get_or_init(|| Regex::new(r"(?i)\bversion\s*[:=]\s*(\d+)").unwrap());

    build_re
        .captures(output)
        .or_else(|| version_re.captures(output))
        .and_then(|captures| captures.get(1))
        .and_then(|match_| match_.as_str().parse::<u64>().ok())
        .filter(|build| *build > 0)
}

fn normalized_arch(arch: &str) -> &str {
    match arch {
        "aarch64" => "arm64",
        "x86_64" => "x64",
        other => other,
    }
}

/// True when the release publishes an asset for the given backend and arch
/// that also matches the given OS.
fn release_has_matching_assets(
    release: &LlamaCppRelease,
    backend: &str,
    arch: &str,
    os: &str,
) -> bool {
    !select_assets_for_os(release, backend, arch, os).is_empty()
}

/// True when the release can be installed on the current machine with the
/// platform's default backend — the backend the updater UI installs with.
fn release_is_installable(release: &LlamaCppRelease) -> bool {
    let os = std::env::consts::OS;
    release_has_matching_assets(
        release,
        default_backend_for_os(os),
        normalized_arch(std::env::consts::ARCH),
        os,
    )
}

/// True for versioned (stable) tags such as `v0.2.0`, as opposed to nightly
/// build tags like `b10576`.
fn is_versioned_tag(tag: &str) -> bool {
    tag.strip_prefix('v')
        .is_some_and(|rest| rest.chars().next().is_some_and(|c| c.is_ascii_digit()))
}

pub(crate) fn routes(ctx: ApiCtx) -> ApiRoute {
    let state = ctx.state;
    let config = ctx.config;

    api_llama_binary_version(config.clone())
        .map(box_reply)
        .or(api_llama_binary_capabilities(config.clone()).map(box_reply))
        .unify()
        .or(api_llama_binary_latest(config.clone()).map(box_reply))
        .unify()
        .or(api_llama_binary_releases(config.clone()).map(box_reply))
        .unify()
        .or(api_llama_binary_release(config.clone()).map(box_reply))
        .unify()
        .or(api_llama_binary_platform_info(config.clone()).map(box_reply))
        .unify()
        .or(api_llama_binary_update(state.clone(), config.clone()).map(box_reply))
        .unify()
        .or(api_llama_restart(state, config).map(box_reply))
        .unify()
        .boxed()
}

/// GET /api/llama-binary/capabilities
///
/// Return the authenticated, exact-binary help snapshot used by typed preset
/// controls. This is deliberately separate from the human-readable version
/// endpoint so the editor can source `-ctk`/`-ctv` values and native reasoning
/// options from the selected executable.
fn api_llama_binary_capabilities(
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "llama-binary" / "capabilities")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let cfg = app_config.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        unauthorized_api_token(),
                    ));
                }
                if !managed_llama_server_path(&cfg).is_file() {
                    return Ok(Box::new(warp::reply::json(&serde_json::json!({
                        "ok": false,
                        "error": "llama-server binary is unavailable",
                    }))));
                }
                match crate::inference::llama_cpp_capabilities::generate_snapshot(
                    &managed_llama_server_path(&cfg),
                )
                .await
                {
                    Ok(snapshot) => Ok(Box::new(warp::reply::json(
                        &serde_json::json!({ "ok": true, "snapshot": snapshot }),
                    ))),
                    Err(error) => Ok(Box::new(warp::reply::json(&serde_json::json!({
                        "ok": false,
                        "error": format!("capability probe failed: {error}"),
                    })))),
                }
            }
        })
}

/// GET /api/llama-binary/version
fn api_llama_binary_version(
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::reply::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "llama-binary" / "version")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let cfg = app_config.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }

                let binary_path = managed_llama_server_path(&cfg);
                let path_str = binary_path.display().to_string();

                let result = tokio::task::spawn_blocking(move || {
                    std::process::Command::new(&binary_path)
                        .arg("--version")
                        .output()
                })
                .await;

                let output = match result {
                    Ok(Ok(o)) => o,
                    _ => {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({
                                "build": serde_json::Value::Null,
                                "version": serde_json::Value::Null,
                                "path": path_str
                            })),
                        ));
                    }
                };

                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                let combined = format!("{}{}", stdout, stderr);

                let build_num = parse_llama_build(&combined);

                match build_num {
                    Some(n) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "build": n,
                            "version": format!("b{}", n),
                            "path": path_str
                        })),
                    )),
                    None => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "build": serde_json::Value::Null,
                            "version": serde_json::Value::Null,
                            "path": path_str
                        })),
                    )),
                }
            }
        })
}

/// GET /api/llama-binary/latest — fetches latest release from GitHub with 30-min cache
fn api_llama_binary_latest(
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::reply::Reply,), Error = warp::Rejection> + Clone {
    use std::sync::LazyLock;
    use tokio::sync::Mutex;

    static LATEST_CACHE: LazyLock<Mutex<Option<(std::time::Instant, serde_json::Value)>>> =
        LazyLock::new(|| Mutex::new(None));

    warp::path!("api" / "llama-binary" / "latest")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let cfg = app_config.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }

                // Check cache
                {
                    let guard = LATEST_CACHE.lock().await;
                    if let Some((ts, ref cached)) = *guard
                        && ts.elapsed() < std::time::Duration::from_secs(30 * 60)
                    {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(cached),
                        ));
                    }
                }
                // Fetch from GitHub
                let client = match reqwest::Client::builder()
                    .timeout(std::time::Duration::from_secs(20))
                    .user_agent(crate::identity::RELEASE_USER_AGENT)
                    .build()
                {
                    Ok(c) => c,
                    Err(e) => {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({
                                "error": format!("Failed to create HTTP client: {}", e)
                            })),
                        ));
                    }
                };

                let url = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=20";
                let resp = match client.get(url).send().await {
                    Ok(r) => r,
                    Err(e) => {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({
                                "error": format!("GitHub API request failed: {}", e)
                            })),
                        ));
                    }
                };

                if !resp.status().is_success() {
                    let status = resp.status();
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "error": format!("GitHub API returned {}", status)
                        })),
                    ));
                }

                let mut releases: Vec<LlamaCppRelease> = match resp.json().await {
                    Ok(v) => v,
                    Err(e) => {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({
                                "error": format!("Failed to parse GitHub response: {}", e)
                            })),
                        ));
                    }
                };

                let any_releases = !releases.is_empty();
                sort_releases_by_published_at(&mut releases);
                let release = releases.into_iter().find(|release| {
                    release
                        .tag_name
                        .strip_prefix('b')
                        .and_then(|value| value.parse::<u64>().ok())
                        .is_some()
                        && release_is_installable(release)
                });

                let Some(release) = release else {
                    // Deliberately not cached: a nightly published after this
                    // response should be visible on the next poll instead of
                    // being hidden for 30 minutes.
                    let error = if any_releases {
                        "No installable nightly llama.cpp release found for this platform"
                    } else {
                        "No llama.cpp releases found on GitHub"
                    };
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({ "error": error })),
                    ));
                };

                let tag = release.tag_name;
                let build_num: Option<u64> =
                    tag.strip_prefix('b').and_then(|value| value.parse().ok());
                let asset_names: Vec<String> =
                    release.assets.into_iter().map(|asset| asset.name).collect();

                let result = serde_json::json!({
                    "tag": tag,
                    "build": build_num,
                    "assets": asset_names,
                    "published_at": release.published_at,
                    "installable": true,
                });

                // Store in cache
                {
                    let mut guard = LATEST_CACHE.lock().await;
                    *guard = Some((std::time::Instant::now(), result.clone()));
                }

                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                    &result,
                )))
            }
        })
}

/// GET /api/llama-binary/releases — lists the last 8 llama.cpp releases for the version picker
fn api_llama_binary_releases(
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::reply::Reply,), Error = warp::Rejection> + Clone {
    use std::sync::LazyLock;
    use tokio::sync::Mutex;

    static RELEASES_CACHE: LazyLock<Mutex<Option<(std::time::Instant, serde_json::Value)>>> =
        LazyLock::new(|| Mutex::new(None));

    warp::path!("api" / "llama-binary" / "releases")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let cfg = app_config.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }

                // Check short-lived cache; release assets are uploaded asynchronously.
                {
                    let guard = RELEASES_CACHE.lock().await;
                    if let Some((ts, ref cached)) = *guard
                        && ts.elapsed() < RELEASE_LIST_CACHE_TTL
                    {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(cached),
                        ));
                    }
                }

                let client = match reqwest::Client::builder()
                    .timeout(std::time::Duration::from_secs(20))
                    .user_agent(crate::identity::RELEASE_USER_AGENT)
                    .build()
                {
                    Ok(c) => c,
                    Err(e) => {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({
                                "error": format!("Failed to create HTTP client: {}", e)
                            })),
                        ));
                    }
                };

                match list_releases(&client).await {
                    Ok(mut releases) => {
                        sort_releases_by_published_at(&mut releases);
                        let window: Vec<LlamaCppRelease> = releases.into_iter().take(8).collect();

                        // Keep the latest versioned (stable) release visible even once it
                        // ages out of the newest-8 window; the UI pins it below the
                        // window so its release notes stay reachable.
                        let pinned_stable = if window.iter().any(|r| is_versioned_tag(&r.tag_name))
                        {
                            None
                        } else {
                            get_latest_stable_release(&client)
                                .await
                                .ok()
                                .filter(|release| is_versioned_tag(&release.tag_name))
                        };

                        let release_item = |r: &LlamaCppRelease| {
                            let build: Option<u64> =
                                r.tag_name.trim_start_matches('b').parse().ok();
                            serde_json::json!({
                                "tag": r.tag_name,
                                "build": build,
                                "published_at": r.published_at,
                                "body": r.body,
                                "installable": release_is_installable(r),
                                "has_binaries": release_has_binary_assets(r),
                            })
                        };

                        let mut items: Vec<serde_json::Value> =
                            window.iter().map(release_item).collect();
                        if let Some(stable) = &pinned_stable {
                            let mut item = release_item(stable);
                            item["stable"] = serde_json::json!(true);
                            items.push(item);
                        }
                        let result = serde_json::json!({ "releases": items });
                        {
                            let mut guard = RELEASES_CACHE.lock().await;
                            *guard = Some((std::time::Instant::now(), result.clone()));
                        }
                        Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&result),
                        ))
                    }
                    Err(e) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "error": format!("Failed to fetch releases: {}", e)
                        })),
                    )),
                }
            }
        })
}

/// GET /api/llama-binary/release?build=XXXXX — fetches a specific release by build number
fn api_llama_binary_release(
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::reply::Reply,), Error = warp::Rejection> + Clone {
    use std::sync::LazyLock;
    use tokio::sync::Mutex;

    static RELEASE_SINGLE_CACHE: LazyLock<Mutex<Option<(std::time::Instant, serde_json::Value)>>> =
        LazyLock::new(|| Mutex::new(None));

    warp::path!("api" / "llama-binary" / "release")
        .and(warp::get())
        .and(warp::query::<ReleaseQuery>())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |query: ReleaseQuery, auth: Option<String>| {
            let cfg = app_config.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }

                let build = query.build;

                // Check per-build cache (5 min)
                {
                    let guard = RELEASE_SINGLE_CACHE.lock().await;
                    if let Some((ts, ref cached)) = *guard
                        && ts.elapsed() < std::time::Duration::from_secs(5 * 60)
                    {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(cached),
                        ));
                    }
                }

                let client = match reqwest::Client::builder()
                    .timeout(std::time::Duration::from_secs(20))
                    .user_agent(crate::identity::RELEASE_USER_AGENT)
                    .build()
                {
                    Ok(c) => c,
                    Err(e) => {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({
                                "error": format!("Failed to create HTTP client: {}", e)
                            })),
                        ));
                    }
                };

                let tag = format!("b{}", build);
                match get_release_by_tag(&client, &tag).await {
                    Ok(release) => {
                        let result = serde_json::json!({
                            "tag": release.tag_name,
                            "build": build,
                            "published_at": release.published_at,
                            "body": release.body,
                            "installable": release_is_installable(&release),
                            "has_binaries": release_has_binary_assets(&release),
                        });
                        {
                            let mut guard = RELEASE_SINGLE_CACHE.lock().await;
                            *guard = Some((std::time::Instant::now(), result.clone()));
                        }
                        Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&result),
                        ))
                    }
                    Err(e) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "error": format!("Failed to fetch release: {}", e)
                        })),
                    )),
                }
            }
        })
}

/// GET /api/llama-binary/platform-info — returns platform/backend info for the download UI
fn api_llama_binary_platform_info(
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::reply::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "llama-binary" / "platform-info")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let cfg = app_config.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }

                let os   = std::env::consts::OS;
                let arch = std::env::consts::ARCH;

                // Human-readable arch label
                let arch_label = match arch {
                    "aarch64" => "ARM64 (Apple Silicon)",
                    "x86_64"  => "x86-64",
                    other     => other,
                };

                // The backend this machine will auto-select on download
                let auto_backend = default_backend_for_os(os);

                // Human-readable label shown before the download button
                let label = match (os, arch) {
                    ("macos", "aarch64") => "Apple Silicon Metal".to_string(),
                    ("macos", _)         => "macOS Metal (x86-64)".to_string(),
                    ("linux", "aarch64") => "Linux ARM64 (CPU)".to_string(),
                    ("linux", _)         => "Linux x86-64 (CPU)".to_string(),
                    ("windows", _)       => "Windows CPU (AVX2)".to_string(),
                    _                    => format!("{} / {}", os, arch),
                };

                // For multi-backend platforms, expose all selectable backends.
                // Windows has the most variety; Linux has a few; macOS is Metal-only.
                let backends: Vec<serde_json::Value> = match os {
                    "windows" => vec![
                        serde_json::json!({
                            "id": "avx2",
                            "label": "CPU (AVX2) — no GPU driver needed",
                            "note": "Universal fallback. Works on any CPU that supports AVX2 (2013+).",
                            "recommended": false
                        }),
                        serde_json::json!({
                            "id": "vulkan",
                            "label": "Vulkan — AMD / Intel / NVIDIA",
                    "note": "Cross-vendor GPU acceleration via Vulkan; includes many AMD integrated GPUs when the Vulkan driver is available.",
                            "recommended": false
                        }),
                        serde_json::json!({
                            "id": "cuda12",
                            "label": "CUDA 12.x — NVIDIA GPU",
                            "note": "Requires a compatible CUDA 12.x runtime and NVIDIA driver.",
                            "recommended": false
                        }),
                        serde_json::json!({
                            "id": "cuda13",
                            "label": "CUDA 13.x — NVIDIA GPU (Turing and newer)",
                            "note": "Requires a compatible CUDA 13.x runtime and NVIDIA driver; Maxwell, Pascal, and Volta are not supported.",
                            "recommended": false
                        }),
                serde_json::json!({
                    "id": "sycl",
                    "label": "SYCL / oneAPI — Intel Arc & Xe GPUs",
                    "note": "Requires Intel oneAPI runtime. For Arc A-series and Xe-HPC.",
                    "recommended": false
                }),
                serde_json::json!({
                    "id": "openvino",
                    "label": "OpenVINO — Intel CPU / GPU / NPU",
                    "note": "Requires the Intel OpenVINO runtime; GPU/NPU use also requires the device to be available to this system.",
                    "recommended": false
                }),
            ],
                    "linux" => vec![
                        serde_json::json!({
                            "id": "cpu",
                            "label": "CPU — universal",
                            "note": "No GPU driver required.",
                            "recommended": false
                        }),
                serde_json::json!({
                    "id": "vulkan",
                    "label": "Vulkan — AMD / Intel / NVIDIA",
                    "note": "Cross-vendor GPU acceleration via Vulkan; includes many AMD integrated GPUs when the Vulkan driver is available.",
                    "recommended": false
                }),
                serde_json::json!({
                    "id": "rocm",
                    "label": "ROCm — AMD GPU",
                    "note": "Requires the AMD ROCm runtime and a supported AMD GPU.",
                    "recommended": false
                }),
                serde_json::json!({
                    "id": "sycl",
                    "label": "SYCL / oneAPI — Intel GPU",
                    "note": "Requires Intel oneAPI runtime.",
                    "recommended": false
                }),
                serde_json::json!({
                    "id": "openvino",
                    "label": "OpenVINO — Intel CPU / GPU / NPU",
                    "note": "Requires the Intel OpenVINO runtime; GPU/NPU use also requires the device to be available to this system.",
                    "recommended": false
                }),
            ],
                    // macOS: Metal only — no choice needed
                    _ => vec![
                        serde_json::json!({
                            "id": "metal",
                            "label": if arch == "aarch64" {
                                "Metal — Apple Silicon (recommended)"
                            } else {
                                "Metal — Intel Mac"
                            },
                            "note": "Uses the GPU via Metal. Built in to macOS.",
                            "recommended": true
                        }),
                    ],
                };

                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                    &serde_json::json!({
                        "os":           os,
                        "arch":         arch,
                        "arch_label":   arch_label,
                        "auto_backend": auto_backend,
                        "label":        label,
                        "backends":     backends,
                        "multi_backend": os == "windows" || os == "linux",
                        "rapid_mlx_local_available": crate::inference::rapid_mlx::ensure_local_platform_supported().is_ok(),
                        "rapid_mlx_local_requirement": "Rapid-MLX local execution requires macOS on Apple Silicon",
                    }),
                )))
            }
        })
}

#[cfg(any(target_os = "macos", test))]
/// POST /api/llama-binary/update — downloads latest release and overwrites llama-server binary
fn api_llama_binary_update(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::reply::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "llama-binary" / "update")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(safe_json_body::<serde_json::Value>())
        .and_then(move |auth: Option<String>, _body: serde_json::Value| {
            let state = state.clone();
            let cfg = app_config.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }

                let os = std::env::consts::OS;
                let arch = std::env::consts::ARCH;

                // Caller may override the backend (e.g. "cuda13" on Windows).
                // Fall back to the platform default if not provided.
                let default_backend = default_backend_for_os(os);
                let backend_owned: String = _body
                    .get("backend")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .unwrap_or(default_backend)
                    .to_string();
                let backend = backend_owned.as_str();

                // Caller may specify a specific tag (e.g. "b4567") to install a previous build.
                let requested_tag: Option<String> = _body
                    .get("tag")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(str::to_string);

                let arch_str = match arch {
                    "aarch64" => "arm64",
                    "x86_64" => "x86_64",
                    other => other,
                };

                let client = match reqwest::Client::builder()
                    .timeout(std::time::Duration::from_secs(300))
                    .user_agent(crate::identity::RELEASE_USER_AGENT)
                    .build()
                {
                    Ok(c) => c,
                    Err(e) => {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({
                                "ok": false,
                                "error": format!("Failed to create HTTP client: {}", e)
                            })),
                        ));
                    }
                };

                // Fetch release list; pick specific tag if requested, otherwise take latest.
                let mut releases =
                    match list_releases(&client).await {
                        Ok(r) => r,
                        Err(e) => {
                            return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                warp::reply::json(&serde_json::json!({
                                    "ok": false,
                                    "error": format!("Failed to list releases: {}", e)
                                })),
                            ));
                        }
                    };

                sort_releases_by_published_at(&mut releases);

                if releases.is_empty() {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": "No releases found on GitHub"
                        })),
                    ));
                }

                let release = if let Some(ref wanted) = requested_tag {
                    match releases.iter().position(|r| &r.tag_name == wanted) {
                        Some(idx) => releases.remove(idx),
                        None => {
                            return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                warp::reply::json(&serde_json::json!({
                                    "ok": false,
                                    "error": format!("Tag {} not found in the last {} releases", wanted, releases.len())
                                })),
                            ));
                        }
                    }
                } else {
                    // Auto-select the newest release that actually ships an
                    // asset for the backend we are about to install with, so a
                    // non-default backend never lands on a release it cannot use.
                    match releases
                        .into_iter()
                        .find(|r| release_has_matching_assets(r, backend, arch_str, os))
                    {
                        Some(release) => release,
                        None => {
                            return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                warp::reply::json(&serde_json::json!({
                                    "ok": false,
                                    "error": format!(
                                        "No llama.cpp release found with a {} asset for OS={} arch={}",
                                        backend, os, arch_str
                                    )
                                })),
                            ));
                        }
                    }
                };
                let tag = release.tag_name.clone();

                let assets =
                    select_assets_for_os(&release, backend, arch_str, os);

                if assets.is_empty() {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": format!(
                                "No matching assets for OS={} arch={} backend={}",
                                os, arch_str, backend
                            )
                        })),
                    ));
                }

                // Download + extract to a temp dir. No runtime lock is held while
                // downloading: the running model keeps serving until the swap.
                let tmp_dir = match tempfile::tempdir() {
                    Ok(d) => d,
                    Err(e) => {
                        return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(json_failure(
                            format!("Failed to create temp dir: {e}"),
                        ));
                    }
                };
                if let Err(e) = download_and_extract(&client, &release, &assets, tmp_dir.path()).await
                {
                    return Ok(json_failure(format!("Download/extract failed: {e}")));
                }

                let binary_name = if os == "windows" { "llama-server.exe" } else { "llama-server" };
                let Some(tmp_binary) = find_binary(tmp_dir.path(), binary_name) else {
                    return Ok(json_failure(format!(
                        "Could not find '{binary_name}' in extracted archive"
                    )));
                };
                let _ = tmp_binary;

                // Stage the complete candidate beside the live directory (same
                // volume) so promotion is a pair of renames, never a copy into
                // a live directory.
                let (bin_dir, custom_path) = managed_bin_dir(&cfg);
                let paths = match RuntimePaths::for_live_dir(&bin_dir) {
                    Ok(p) => p,
                    Err(e) => return Ok(json_failure(format!("Invalid runtime location: {e}"))),
                };
                let staging = paths.staging_dir(&new_transaction_id());
                let stage_result = {
                    let src = tmp_dir.path().to_path_buf();
                    let staging = staging.clone();
                    tokio::task::spawn_blocking(move || stage_release_files(&src, &staging)).await
                };
                match stage_result {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => {
                        let _ = std::fs::remove_dir_all(&staging);
                        return Ok(json_failure(format!("Failed to stage release files: {e}")));
                    }
                    Err(e) => return Ok(json_failure(format!("Staging task failed: {e}"))),
                }
                if let Err(e) = cleanup_old_binaries(&staging).await {
                    eprintln!("[warn] llama.cpp binary cleanup failed: {e}");
                }

                let manifest = RuntimeManifest::for_directory(
                    &staging,
                    ManifestSource::UpstreamPrebuilt,
                    format!("{tag} prebuilt"),
                    LlamaCppRef {
                        reference: Some(tag.clone()),
                        commit: None,
                        build_number: parse_llama_build(&tag).map(|n| n as u32),
                    },
                    Some(backend.to_string()),
                )
                .ok();

                // One runtime-mutation gate; the swap window starts only once the
                // journal exists.
                let mut guard = match state.runtime_gate.try_begin(RuntimeOp::Update) {
                    Ok(guard) => guard,
                    Err(busy) => {
                        let _ = std::fs::remove_dir_all(&staging);
                        return Ok(json_busy(busy));
                    }
                };
                let host = if custom_path.is_some() {
                    // A custom server path is not ours to stop or restart.
                    LiveHost::without_model(state.clone(), cfg.clone())
                } else {
                    LiveHost::new(state.clone(), cfg.clone())
                };
                let restart_applicable = host.restart_applicable();
                state.push_log(format!(
                    "[monitor] llama-binary/update: installing {tag} to {}",
                    bin_dir.display()
                ));
                let request = PromoteRequest {
                    staged_dir: staging.clone(),
                    label: format!("{tag} prebuilt"),
                    exe_name: binary_name.to_string(),
                    manifest,
                    verification: VerificationIntent::prebuilt(backend),
                };
                let result = promote(&host, &mut guard, &paths, request).await;
                drop(guard);
                let _ = std::fs::remove_dir_all(&staging);

                let installed_binary = bin_dir.join(binary_name);
                let sha256_hex = std::fs::read(&installed_binary).ok().map(|bytes| {
                    use sha2::Digest;
                    let mut hasher = sha2::Sha256::new();
                    hasher.update(&bytes);
                    hasher.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>()
                });
                let custom = custom_path.as_ref().map(|p| p.display().to_string());
                Ok(promote_reply(
                    &state,
                    result,
                    PromoteReplyContext {
                        tag: &tag,
                        backend,
                        arch: arch_str,
                        sha256: sha256_hex,
                        restart_applicable,
                        custom_server_path: custom,
                    },
                ))
            }
        })
}

/// POST /api/llama/restart — restart the running llama-server with the current
/// binary (useful after installing a new llama-server version).
fn api_llama_restart(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::reply::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "llama" / "restart")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and_then(move |auth: Option<String>| {
            let state = state.clone();
            let cfg = app_config.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }

                let state_clone = state.clone();
                let local_running = *state.local_server_running.lock().unwrap();

                if !local_running {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": "No local llama-server is running."
                        })),
                    ));
                }

                // Preserve the backend-owned request before stop clears it.
                let saved_request = {
                    let guard = state_clone.local_launch_request.lock().unwrap();
                    guard.clone()
                };

                let Some(request) = saved_request else {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": "No saved local launch configuration found."
                        })),
                    ));
                };

                state.push_log("[monitor] restart: stopping existing server".into());

                // Stop current server
                if let Err(e) = stop_server(&state_clone).await {
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": format!("Failed to stop server: {}", e)
                        })),
                    ));
                }

                // Brief pause to let the old process fully shut down
                let pause_start = std::time::Instant::now();
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                state.push_log(format!(
                    "[monitor] restart: re-spawning after {}ms",
                    pause_start.elapsed().as_millis()
                ));

                // Restart the same backend with its backend-owned configuration.
                if let Err(e) = launch_local(Arc::new(state_clone.clone()), request, &cfg).await {
                    state.push_log(format!("[monitor] restart: start_server failed: {}", e));
                    return Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": format!("Failed to restart server: {}", e)
                        })),
                    ));
                }

                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                    &serde_json::json!({
                        "ok": true,
                        "message": "Server restart initiated."
                    }),
                )))
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::llama_cpp::ServerConfig;
    use crate::llama_runtime::host::restart_config;

    #[test]
    fn is_versioned_tag_distinguishes_stable_and_nightly_tags() {
        assert!(is_versioned_tag("v0.2.0"));
        assert!(is_versioned_tag("v1.10.3"));
        assert!(!is_versioned_tag("b10576"));
        assert!(!is_versioned_tag("nightly-20260822"));
        assert!(!is_versioned_tag("v-next"));
    }

    #[test]
    fn release_has_matching_assets_requires_backend_arch_and_os() {
        let release = LlamaCppRelease {
            tag_name: "b6000".into(),
            assets: vec![crate::llama::llama_cpp_downloader::LlamaCppAsset {
                name: "llama-b6000-bin-ubuntu-cuda-12.4-x64.tar.gz".into(),
                browser_download_url: "https://example.com/cuda".into(),
            }],
            published_at: String::new(),
            body: String::new(),
        };
        assert!(release_has_matching_assets(
            &release, "cuda12", "x64", "linux"
        ));
        assert!(!release_has_matching_assets(
            &release, "cpu", "x64", "linux"
        ));
        assert!(!release_has_matching_assets(
            &release, "cuda12", "x64", "windows"
        ));
        assert!(!release_has_matching_assets(
            &release, "cuda12", "arm64", "linux"
        ));
    }

    #[test]
    fn macos_arm_release_with_generic_metal_archive_is_installable() {
        let release = LlamaCppRelease {
            tag_name: "b10685".into(),
            assets: vec![crate::llama::llama_cpp_downloader::LlamaCppAsset {
                name: "llama-b10685-bin-macos-arm64.tar.gz".into(),
                browser_download_url: "https://example.com/macos-arm64".into(),
            }],
            published_at: String::new(),
            body: String::new(),
        };

        assert!(release_has_matching_assets(
            &release, "metal", "arm64", "macos"
        ));
    }

    #[test]
    fn llama_update_does_not_restart_rapid_mlx_backend() {
        let state = AppState::default();
        *state.local_server_running.lock().unwrap() = true;
        *state.server_config.lock().unwrap() = Some(ServerConfig::default());
        *state.local_launch_request.lock().unwrap() =
            Some(crate::inference::launch::LocalLaunchRequest::RapidMlx(
                Box::new(crate::inference::rapid_mlx::RapidMlxConfig {
                    model_path: "/models/rapid".into(),
                    ..Default::default()
                }),
            ));

        assert!(restart_config(&state).is_none());
    }

    #[test]
    fn health_check_timeout_allows_slow_cold_macos_startup() {
        assert_eq!(
            crate::llama_runtime::verify::VERIFY_PROBE_TIMEOUT,
            std::time::Duration::from_secs(30)
        );
    }

    #[test]
    fn parse_llama_build_prefers_build_over_semver_version() {
        assert_eq!(
            parse_llama_build("version: 0.3.0-dev (build 10672, commit abc)"),
            Some(10672)
        );
    }

    #[test]
    fn parse_llama_build_accepts_legacy_numeric_formats() {
        assert_eq!(parse_llama_build("build: 10576"), Some(10576));
        assert_eq!(parse_llama_build("version: 10576"), Some(10576));
        assert_eq!(parse_llama_build("version: 0.0.0"), None);
    }
}
