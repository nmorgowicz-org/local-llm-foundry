use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use warp::Filter;

use crate::agent;
use crate::config::AppConfig;
use crate::remote_ssh::{self, SshConnection};

use super::{
    ApiCtx, ApiRoute, bearer_matches_api_token, bearer_matches_db_admin_token, extract_bearer,
    unauthorized_api_token, unauthorized_db_admin_token, with_app_config,
};

pub(crate) fn routes(ctx: ApiCtx) -> ApiRoute {
    let config = ctx.config;

    api_remote_agent_latest_release(config.clone())
        .or(api_remote_agent_detect(config.clone()))
        .unify()
        .or(api_remote_agent_ssh_host_key(config.clone()))
        .unify()
        .or(api_remote_agent_ssh_trust(config.clone()))
        .unify()
        .or(api_remote_agent_install(config.clone()))
        .unify()
        .or(api_remote_agent_status(config.clone()))
        .unify()
        .or(api_remote_agent_start(config.clone()))
        .unify()
        .or(api_remote_agent_update(config.clone()))
        .unify()
        .or(api_remote_agent_stop(config.clone()))
        .unify()
        .or(api_remote_agent_remove(config.clone()))
        .unify()
        .or(api_remote_agent_tls_status(config))
        .unify()
        .boxed()
}

fn api_remote_agent_latest_release(app_config: Arc<AppConfig>) -> ApiRoute {
    warp::path!("api" / "remote-agent" / "releases" / "latest")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and(with_app_config(app_config))
        .and_then(
            move |auth: Option<String>, cfg: Arc<AppConfig>| async move {
                let bearer = extract_bearer(auth);
                if !bearer_matches_api_token(bearer.as_deref(), &cfg) {
                    return Ok(unauthorized_api_token());
                }
                match agent::latest_release_info().await {
                    Ok(release) => Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                        &serde_json::json!({"ok": true, "release": release}),
                    ))
                        as Box<dyn warp::reply::Reply>),
                    Err(e) => Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                        &serde_json::json!({"ok": false, "error": e.to_string()}),
                    ))
                        as Box<dyn warp::reply::Reply>),
                }
            },
        )
        .boxed()
}

fn api_remote_agent_detect(app_config: Arc<AppConfig>) -> ApiRoute {
    static LAST_REMOTE_AGENT_DETECT: AtomicU64 = AtomicU64::new(0);

    warp::path!("api" / "remote-agent" / "detect")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(warp::body::json())
        .and_then(
            move |auth: Option<String>, mut request: agent::RemoteAgentDetectRequest| {
                let app_config = app_config.clone();
                async move {
                    let bearer = extract_bearer(auth);
                    if !bearer_matches_api_token(bearer.as_deref(), &app_config) {
                        return Ok(unauthorized_api_token());
                    }
                    let now = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    let last = LAST_REMOTE_AGENT_DETECT.load(Ordering::Acquire);
                    if now - last < 10 {
                        let remaining = 10 - (now - last);
                        return Ok::<_, warp::Rejection>(Box::new(warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({
                                "ok": false,
                                "error": "too soon; please wait",
                                "seconds_remaining": remaining
                            })),
                            warp::http::StatusCode::TOO_MANY_REQUESTS,
                        ))
                            as Box<dyn warp::reply::Reply>);
                    }
                    LAST_REMOTE_AGENT_DETECT.store(now, Ordering::Release);

                    match hydrate_ssh_connection(
                        request.ssh_connection.take(),
                        &request.ssh_target,
                        &app_config,
                    ) {
                        Ok(connection) => request.ssh_connection = Some(connection),
                        Err(e) => {
                            return Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                                &serde_json::json!({"ok": false, "error": e.to_string()}),
                            ))
                                as Box<dyn warp::reply::Reply>);
                        }
                    }
                    let response = agent::detect_remote_agent(request).await;
                    Ok::<_, warp::Rejection>(
                        Box::new(warp::reply::json(&response)) as Box<dyn warp::reply::Reply>
                    )
                }
            },
        )
        .boxed()
}

fn api_remote_agent_ssh_host_key(app_config: Arc<AppConfig>) -> ApiRoute {
    static LAST_REMOTE_AGENT_SSH_HOST_KEY: AtomicU64 = AtomicU64::new(0);

    warp::path!("api" / "remote-agent" / "ssh" / "host-key")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(warp::body::json())
        .and_then(
            move |auth: Option<String>, request: serde_json::Map<String, serde_json::Value>| {
                let app_config = app_config.clone();
                async move {
                    let bearer = extract_bearer(auth);
                    if !bearer_matches_api_token(bearer.as_deref(), &app_config) {
                        return Ok(unauthorized_api_token());
                    }
                    let now = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    let last = LAST_REMOTE_AGENT_SSH_HOST_KEY.load(Ordering::Acquire);
                    if now - last < 10 {
                        let remaining = 10 - (now - last);
                        return Ok::<_, warp::Rejection>(Box::new(warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({
                                "ok": false,
                                "error": "too soon; please wait",
                                "seconds_remaining": remaining
                            })),
                            warp::http::StatusCode::TOO_MANY_REQUESTS,
                        ))
                            as Box<dyn warp::reply::Reply>);
                    }
                    LAST_REMOTE_AGENT_SSH_HOST_KEY.store(now, Ordering::Release);

                    let target = request
                        .get("ssh_target")
                        .and_then(|value| value.as_str())
                        .unwrap_or("");
                    let connection = ssh_connection_from_request(&request, target);
                    match remote_ssh::scan_host_key(
                        connection,
                        app_config.ssh_known_hosts_file.clone(),
                    )
                    .await
                    {
                        Ok(info) => Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                            &serde_json::json!({"ok": true, "host_key": info}),
                        ))
                            as Box<dyn warp::reply::Reply>),
                        Err(e) => Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                            &serde_json::json!({"ok": false, "error": e.to_string()}),
                        ))
                            as Box<dyn warp::reply::Reply>),
                    }
                }
            },
        )
        .boxed()
}

fn api_remote_agent_ssh_trust(app_config: Arc<AppConfig>) -> ApiRoute {
    static LAST_REMOTE_AGENT_SSH_TRUST: AtomicU64 = AtomicU64::new(0);

    warp::path!("api" / "remote-agent" / "ssh" / "trust")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(warp::body::json())
        .and_then(move |auth: Option<String>, request: serde_json::Map<String, serde_json::Value>| {
            let app_config = app_config.clone();
            async move {
                let bearer = extract_bearer(auth);
                if !bearer_matches_api_token(bearer.as_deref(), &app_config) {
                    return Ok(unauthorized_api_token());
                }
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let last = LAST_REMOTE_AGENT_SSH_TRUST.load(Ordering::Acquire);
                if now - last < 10 {
                    let remaining = 10 - (now - last);
                    return Ok::<_, warp::Rejection>(Box::new(warp::reply::with_status(
                        warp::reply::json(&serde_json::json!({
                            "ok": false,
                            "error": "too soon; please wait",
                            "seconds_remaining": remaining
                        })),
                        warp::http::StatusCode::TOO_MANY_REQUESTS,
                    )) as Box<dyn warp::reply::Reply>);
                }
                LAST_REMOTE_AGENT_SSH_TRUST.store(now, Ordering::Release);

                let target = request
                    .get("ssh_target")
                    .and_then(|value| value.as_str())
                    .unwrap_or("");
                let key_hex = request
                    .get("key_hex")
                    .and_then(|value| value.as_str())
                    .unwrap_or("");
                let connection = ssh_connection_from_request(&request, target);
                match remote_ssh::scan_host_key(
                    connection.clone(),
                    app_config.ssh_known_hosts_file.clone(),
                )
                .await
                {
                    Ok(info) if info.key_hex == key_hex.trim().to_ascii_lowercase() => {}
                    Ok(_) => {
                        return Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                            &serde_json::json!({"ok": false, "error": "Host key changed between scan and trust confirmation"}),
                        )) as Box<dyn warp::reply::Reply>);
                    }
                    Err(e) => {
                        return Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                            &serde_json::json!({"ok": false, "error": e.to_string()}),
                        )) as Box<dyn warp::reply::Reply>);
                    }
                }
                match remote_ssh::trust_host_key(
                    &app_config.ssh_known_hosts_file,
                    &connection,
                    key_hex,
                ) {
                    Ok(()) => Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                        &serde_json::json!({"ok": true}),
                    )) as Box<dyn warp::reply::Reply>),
                    Err(e) => Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                        &serde_json::json!({"ok": false, "error": e.to_string()}),
                    )) as Box<dyn warp::reply::Reply>),
                }
            }
        })
        .boxed()
}

fn api_remote_agent_install(app_config: Arc<AppConfig>) -> ApiRoute {
    static LAST_REMOTE_AGENT_INSTALL: AtomicU64 = AtomicU64::new(0);

    warp::path!("api" / "remote-agent" / "install")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(warp::body::json())
        .and_then(
            move |auth: Option<String>, mut request: agent::RemoteAgentInstallRequest| {
                let app_config = app_config.clone();
                async move {
                    let bearer = extract_bearer(auth);
                    if !bearer_matches_db_admin_token(bearer.as_deref(), &app_config) {
                        return Ok(unauthorized_db_admin_token());
                    }
                    let now = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    let last = LAST_REMOTE_AGENT_INSTALL.load(Ordering::Acquire);
                    if now - last < 30 {
                        let remaining = 30 - (now - last);
                        return Ok::<_, warp::Rejection>(Box::new(warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({
                                "ok": false,
                                "error": "too soon; please wait",
                                "seconds_remaining": remaining
                            })),
                            warp::http::StatusCode::TOO_MANY_REQUESTS,
                        ))
                            as Box<dyn warp::reply::Reply>);
                    }
                    LAST_REMOTE_AGENT_INSTALL.store(now, Ordering::Release);

                    agent::suppress_remote_agent_autostart();
                    request.ssh_connection = match hydrate_ssh_connection(
                        request.ssh_connection.take(),
                        &request.ssh_target,
                        &app_config,
                    ) {
                        Ok(connection) => Some(connection),
                        Err(e) => {
                            return Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                                &serde_json::json!({"ok": false, "error": e.to_string()}),
                            ))
                                as Box<dyn warp::reply::Reply>);
                        }
                    };
                    let remote_os = if let Some(connection) = request.ssh_connection.clone() {
                        agent::detect_remote_os_for_connection(connection).await
                    } else {
                        agent::detect_remote_os_simple(&request.ssh_target).await
                    };
                    let api_token = app_config.live_api_token();
                    match agent::install_remote_agent(
                        request.ssh_target.trim(),
                        request.ssh_connection.clone(),
                        &request.asset,
                        request.install_path.clone(),
                        remote_os,
                        api_token,
                    )
                    .await
                    {
                        Ok(response) => {
                            Ok::<_, warp::Rejection>(Box::new(warp::reply::json(&response))
                                as Box<dyn warp::reply::Reply>)
                        }
                        Err(e) => Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                            &serde_json::json!({"ok": false, "error": e.to_string()}),
                        ))
                            as Box<dyn warp::reply::Reply>),
                    }
                }
            },
        )
        .boxed()
}

fn api_remote_agent_status(app_config: Arc<AppConfig>) -> ApiRoute {
    static LAST_REMOTE_AGENT_STATUS: AtomicU64 = AtomicU64::new(0);

    warp::path!("api" / "remote-agent" / "status")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(warp::body::json())
        .and_then(
            move |auth: Option<String>, request: serde_json::Map<String, serde_json::Value>| {
                let app_config = app_config.clone();
                async move {
                    let bearer = extract_bearer(auth);
                    if !bearer_matches_api_token(bearer.as_deref(), &app_config) {
                        return Ok(unauthorized_api_token());
                    }
                    let now = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    let last = LAST_REMOTE_AGENT_STATUS.load(Ordering::Acquire);
                    if now - last < 5 {
                        let remaining = 5 - (now - last);
                        return Ok::<_, warp::Rejection>(Box::new(warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({
                                "ok": false,
                                "error": "too soon; please wait",
                                "seconds_remaining": remaining
                            })),
                            warp::http::StatusCode::TOO_MANY_REQUESTS,
                        ))
                            as Box<dyn warp::reply::Reply>);
                    }
                    LAST_REMOTE_AGENT_STATUS.store(now, Ordering::Release);

                    let ssh_target = match request.get("ssh_target") {
                        Some(v) => v.as_str().unwrap_or("").to_string(),
                        None => {
                            return Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                                &serde_json::json!({"ok": false, "error": "Missing ssh_target"}),
                            ))
                                as Box<dyn warp::reply::Reply>);
                        }
                    };
                    let ssh_connection = match hydrate_ssh_connection(
                        request
                            .get("ssh_connection")
                            .and_then(|value| serde_json::from_value(value.clone()).ok()),
                        &ssh_target,
                        &app_config,
                    ) {
                        Ok(connection) => Some(connection),
                        Err(e) => {
                            return Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                                &serde_json::json!({"ok": false, "error": e.to_string()}),
                            ))
                                as Box<dyn warp::reply::Reply>);
                        }
                    };
                    let agent_url = request
                        .get("agent_url")
                        .and_then(|value| value.as_str())
                        .map(str::trim)
                        .filter(|value| !value.is_empty());
                    match agent::status_remote_agent(&ssh_target, ssh_connection, agent_url).await {
                        Ok(response) => {
                            Ok::<_, warp::Rejection>(Box::new(warp::reply::json(&response))
                                as Box<dyn warp::reply::Reply>)
                        }
                        Err(e) => Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                            &serde_json::json!({"ok": false, "error": e.to_string()}),
                        ))
                            as Box<dyn warp::reply::Reply>),
                    }
                }
            },
        )
        .boxed()
}

fn api_remote_agent_start(app_config: Arc<AppConfig>) -> ApiRoute {
    static LAST_REMOTE_AGENT_START: AtomicU64 = AtomicU64::new(0);

    warp::path!("api" / "remote-agent" / "start")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(warp::body::json())
        .and_then(
            move |auth: Option<String>, request: serde_json::Map<String, serde_json::Value>| {
                let app_config = app_config.clone();
                async move {
                    let bearer = extract_bearer(auth);
                    if !bearer_matches_api_token(bearer.as_deref(), &app_config) {
                        return Ok(unauthorized_api_token());
                    }
                    let now = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    let last = LAST_REMOTE_AGENT_START.load(Ordering::Acquire);
                    if now - last < 10 {
                        let remaining = 10 - (now - last);
                        return Ok::<_, warp::Rejection>(Box::new(warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({
                                "ok": false,
                                "error": "too soon; please wait",
                                "seconds_remaining": remaining
                            })),
                            warp::http::StatusCode::TOO_MANY_REQUESTS,
                        ))
                            as Box<dyn warp::reply::Reply>);
                    }
                    LAST_REMOTE_AGENT_START.store(now, Ordering::Release);

                    agent::suppress_remote_agent_autostart();
                    let ssh_target = match request.get("ssh_target") {
                        Some(v) => v.as_str().unwrap_or("").to_string(),
                        None => {
                            return Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                                &serde_json::json!({"ok": false, "error": "Missing ssh_target"}),
                            ))
                                as Box<dyn warp::reply::Reply>);
                        }
                    };
                    let ssh_connection = match hydrate_ssh_connection(
                        request
                            .get("ssh_connection")
                            .and_then(|value| serde_json::from_value(value.clone()).ok()),
                        &ssh_target,
                        &app_config,
                    ) {
                        Ok(connection) => Some(connection),
                        Err(e) => {
                            return Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                                &serde_json::json!({"ok": false, "error": e.to_string()}),
                            ))
                                as Box<dyn warp::reply::Reply>);
                        }
                    };
                    let remote_os = if let Some(ref conn) = ssh_connection {
                        agent::detect_remote_os_with(conn).await
                    } else {
                        agent::detect_remote_os_simple(&ssh_target).await
                    };
                    let requested_install_path =
                        request.get("install_path").and_then(|v| v.as_str());
                    let install_path = agent::resolve_remote_install_path(
                        ssh_connection.as_ref(),
                        remote_os,
                        requested_install_path,
                    )
                    .await;
                    let command = if let Some(ref conn) = ssh_connection {
                        agent::default_start_command_for_os_with(conn, remote_os, &install_path)
                            .await
                    } else {
                        match request.get("start_command") {
                            Some(v) => {
                                let cmd = v.as_str().unwrap_or("").to_string();
                                if agent::validate_remote_command(&cmd) {
                                    cmd
                                } else {
                                    agent::default_start_command_for_target(
                                        &ssh_target,
                                        &install_path,
                                    )
                                    .await
                                }
                            }
                            None => {
                                agent::default_start_command_for_target(&ssh_target, &install_path)
                                    .await
                            }
                        }
                    };
                    match agent::start_remote_agent(
                        &ssh_target,
                        ssh_connection,
                        &install_path,
                        &command,
                    )
                    .await
                    {
                        Ok(response) => {
                            Ok::<_, warp::Rejection>(Box::new(warp::reply::json(&response))
                                as Box<dyn warp::reply::Reply>)
                        }
                        Err(e) => Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                            &serde_json::json!({"ok": false, "error": e.to_string()}),
                        ))
                            as Box<dyn warp::reply::Reply>),
                    }
                }
            },
        )
        .boxed()
}

fn api_remote_agent_update(app_config: Arc<AppConfig>) -> ApiRoute {
    static LAST_REMOTE_AGENT_UPDATE: AtomicU64 = AtomicU64::new(0);

    warp::path!("api" / "remote-agent" / "update")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(warp::body::json())
        .and_then(
            move |auth: Option<String>, request: serde_json::Map<String, serde_json::Value>| {
                let app_config = app_config.clone();
                async move {
                    let bearer = extract_bearer(auth);
                    if !bearer_matches_api_token(bearer.as_deref(), &app_config) {
                        return Ok(unauthorized_api_token());
                    }
                    let now = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    let last = LAST_REMOTE_AGENT_UPDATE.load(Ordering::Acquire);
                    if now - last < 30 {
                        let remaining = 30 - (now - last);
                        return Ok::<_, warp::Rejection>(Box::new(warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({
                                "ok": false,
                                "error": "too soon; please wait",
                                "seconds_remaining": remaining
                            })),
                            warp::http::StatusCode::TOO_MANY_REQUESTS,
                        ))
                            as Box<dyn warp::reply::Reply>);
                    }
                    LAST_REMOTE_AGENT_UPDATE.store(now, Ordering::Release);

                    agent::suppress_remote_agent_autostart();
                    let ssh_target = match request.get("ssh_target") {
                        Some(v) => v.as_str().unwrap_or("").to_string(),
                        None => {
                            return Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                                &serde_json::json!({"ok": false, "error": "Missing ssh_target"}),
                            ))
                                as Box<dyn warp::reply::Reply>);
                        }
                    };
                    let ssh_connection = match hydrate_ssh_connection(
                        request
                            .get("ssh_connection")
                            .and_then(|value| serde_json::from_value(value.clone()).ok()),
                        &ssh_target,
                        &app_config,
                    ) {
                        Ok(connection) => Some(connection),
                        Err(e) => {
                            return Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                                &serde_json::json!({"ok": false, "error": e.to_string()}),
                            ))
                                as Box<dyn warp::reply::Reply>);
                        }
                    };
                    match agent::update_remote_agent(&ssh_target, ssh_connection).await {
                        Ok(response) => {
                            Ok::<_, warp::Rejection>(Box::new(warp::reply::json(&response))
                                as Box<dyn warp::reply::Reply>)
                        }
                        Err(e) => Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                            &serde_json::json!({"ok": false, "error": e.to_string()}),
                        ))
                            as Box<dyn warp::reply::Reply>),
                    }
                }
            },
        )
        .boxed()
}

fn api_remote_agent_stop(app_config: Arc<AppConfig>) -> ApiRoute {
    static LAST_REMOTE_AGENT_STOP: AtomicU64 = AtomicU64::new(0);

    warp::path!("api" / "remote-agent" / "stop")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(warp::body::json())
        .and_then(
            move |auth: Option<String>, request: serde_json::Map<String, serde_json::Value>| {
                let app_config = app_config.clone();
                async move {
                    let bearer = extract_bearer(auth);
                    if !bearer_matches_api_token(bearer.as_deref(), &app_config) {
                        return Ok(unauthorized_api_token());
                    }
                    let now = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    let last = LAST_REMOTE_AGENT_STOP.load(Ordering::Acquire);
                    if now - last < 10 {
                        let remaining = 10 - (now - last);
                        return Ok::<_, warp::Rejection>(Box::new(warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({
                                "ok": false,
                                "error": "too soon; please wait",
                                "seconds_remaining": remaining
                            })),
                            warp::http::StatusCode::TOO_MANY_REQUESTS,
                        ))
                            as Box<dyn warp::reply::Reply>);
                    }
                    LAST_REMOTE_AGENT_STOP.store(now, Ordering::Release);

                    agent::suppress_remote_agent_autostart();
                    let ssh_target = match request.get("ssh_target") {
                        Some(v) => v.as_str().unwrap_or("").to_string(),
                        None => {
                            return Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                                &serde_json::json!({"ok": false, "error": "Missing ssh_target"}),
                            ))
                                as Box<dyn warp::reply::Reply>);
                        }
                    };
                    let ssh_connection = match hydrate_ssh_connection(
                        request
                            .get("ssh_connection")
                            .and_then(|value| serde_json::from_value(value.clone()).ok()),
                        &ssh_target,
                        &app_config,
                    ) {
                        Ok(connection) => Some(connection),
                        Err(e) => {
                            return Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                                &serde_json::json!({"ok": false, "error": e.to_string()}),
                            ))
                                as Box<dyn warp::reply::Reply>);
                        }
                    };
                    match agent::stop_remote_agent(&ssh_target, ssh_connection).await {
                        Ok(response) => {
                            Ok::<_, warp::Rejection>(Box::new(warp::reply::json(&response))
                                as Box<dyn warp::reply::Reply>)
                        }
                        Err(e) => Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                            &serde_json::json!({"ok": false, "error": e.to_string()}),
                        ))
                            as Box<dyn warp::reply::Reply>),
                    }
                }
            },
        )
        .boxed()
}

fn api_remote_agent_remove(app_config: Arc<AppConfig>) -> ApiRoute {
    static LAST_REMOTE_AGENT_REMOVE: AtomicU64 = AtomicU64::new(0);

    warp::path!("api" / "remote-agent" / "remove")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(warp::body::json())
        .and_then(
            move |auth: Option<String>, request: serde_json::Map<String, serde_json::Value>| {
                let app_config = app_config.clone();
                async move {
                    let bearer = extract_bearer(auth);
                    if !bearer_matches_db_admin_token(bearer.as_deref(), &app_config) {
                        return Ok(unauthorized_db_admin_token());
                    }
                    let now = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    let last = LAST_REMOTE_AGENT_REMOVE.load(Ordering::Acquire);
                    if now - last < 15 {
                        let remaining = 15 - (now - last);
                        return Ok::<_, warp::Rejection>(Box::new(warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({
                                "ok": false,
                                "error": "too soon; please wait",
                                "seconds_remaining": remaining
                            })),
                            warp::http::StatusCode::TOO_MANY_REQUESTS,
                        ))
                            as Box<dyn warp::reply::Reply>);
                    }
                    LAST_REMOTE_AGENT_REMOVE.store(now, Ordering::Release);

                    agent::suppress_remote_agent_autostart();
                    let ssh_target = match request.get("ssh_target") {
                        Some(v) => v.as_str().unwrap_or("").to_string(),
                        None => {
                            return Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                                &serde_json::json!({"ok": false, "error": "Missing ssh_target"}),
                            ))
                                as Box<dyn warp::reply::Reply>);
                        }
                    };
                    let ssh_connection = match hydrate_ssh_connection(
                        request
                            .get("ssh_connection")
                            .and_then(|value| serde_json::from_value(value.clone()).ok()),
                        &ssh_target,
                        &app_config,
                    ) {
                        Ok(connection) => Some(connection),
                        Err(e) => {
                            return Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                                &serde_json::json!({"ok": false, "error": e.to_string()}),
                            ))
                                as Box<dyn warp::reply::Reply>);
                        }
                    };
                    match agent::remove_remote_agent(&ssh_target, ssh_connection).await {
                        Ok(response) => {
                            Ok::<_, warp::Rejection>(Box::new(warp::reply::json(&response))
                                as Box<dyn warp::reply::Reply>)
                        }
                        Err(e) => Ok::<_, warp::Rejection>(Box::new(warp::reply::json(
                            &serde_json::json!({"ok": false, "error": e.to_string()}),
                        ))
                            as Box<dyn warp::reply::Reply>),
                    }
                }
            },
        )
        .boxed()
}

fn api_remote_agent_tls_status(app_config: Arc<AppConfig>) -> ApiRoute {
    warp::path!("api" / "remote-agent" / "tls-status")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .map(move |auth: Option<String>| {
            let bearer = extract_bearer(auth);
            if !bearer_matches_api_token(bearer.as_deref(), &app_config) {
                return unauthorized_api_token();
            }
            let certs_dir = crate::certs::certs_path_for(&app_config.config_dir);
            let ca_present = certs_dir.join("ca.pem").exists();
            let server_present = certs_dir.join("agent-server.pem").exists();
            let client_present = certs_dir.join("agent-client.pem").exists();
            Box::new(warp::reply::json(&serde_json::json!({
                "mtls_enforced": true,
                "ca_present": ca_present,
                "server_cert_present": server_present,
                "client_cert_present": client_present,
            }))) as Box<dyn warp::reply::Reply>
        })
        .boxed()
}

fn ssh_connection_from_request(
    request: &serde_json::Map<String, serde_json::Value>,
    target: &str,
) -> SshConnection {
    request
        .get("ssh_connection")
        .and_then(|value| serde_json::from_value(value.clone()).ok())
        .unwrap_or_else(|| SshConnection::from_target(target))
}

fn hydrate_ssh_connection(
    connection: Option<SshConnection>,
    target: &str,
    app_config: &AppConfig,
) -> anyhow::Result<SshConnection> {
    let connection = connection.unwrap_or_else(|| SshConnection::from_target(target));
    remote_ssh::with_trusted_host_key(connection, &app_config.ssh_known_hosts_file)
}
