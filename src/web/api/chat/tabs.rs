use std::collections::HashMap;
use std::sync::Arc;

use sha2::{Digest, Sha256};
use warp::Filter;

use crate::chat_storage::{ChatStorage, TabVisibility};
use crate::config::AppConfig;
use crate::state::AppState;

use super::super::common::{
    ApiCtx, ApiRoute, check_api_token, unauthorized_api_token, with_chat_storage,
};
use super::suggestions::ContextNote;

#[allow(dead_code)]
pub(crate) mod legacy_chat_types {
    use super::ContextNote;

    /// Chat message structure for persistence (legacy flat-file types, used by tests)
    #[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
    pub(crate) struct CompactionPreview {
        pub(crate) role: String,
        pub(crate) snippet: String,
    }

    #[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
    pub(crate) struct ChatMessage {
        pub(crate) role: String,
        pub(crate) content: String,
        pub(crate) timestamp_ms: u64,
        #[serde(default)]
        pub(crate) input_tokens: Option<u64>,
        #[serde(default)]
        pub(crate) output_tokens: Option<u64>,
        #[serde(default, alias = "cumulativeInputTokens")]
        pub(crate) cumulative_input_tokens: Option<u64>,
        #[serde(default, alias = "cumulativeOutputTokens")]
        pub(crate) cumulative_output_tokens: Option<u64>,
        #[serde(default)]
        pub(crate) compaction_marker: Option<bool>,
        #[serde(default)]
        pub(crate) summarized: Option<bool>,
        #[serde(default)]
        pub(crate) dropped_count: Option<u64>,
        #[serde(default)]
        pub(crate) dropped_preview: Option<Vec<CompactionPreview>>,
        #[serde(default)]
        pub(crate) tokens_freed_estimate: Option<u64>,
        #[serde(default)]
        pub(crate) ctx_pct_before: Option<f32>,
        #[serde(default)]
        pub(crate) memory_version: Option<u32>,
        #[serde(default)]
        pub(crate) memory_domain: Option<String>,
        #[serde(default)]
        pub(crate) summary_kind: Option<String>,
        #[serde(default)]
        pub(crate) compacted_at: Option<u64>,
        #[serde(default)]
        pub(crate) compacted_message_count_total: Option<u64>,
        #[serde(default)]
        pub(crate) recent_tail_kept: Option<u64>,
        #[serde(default)]
        pub(crate) thinking_content: Option<String>,
    }

    #[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
    pub(crate) struct ArmedStoryBeat {
        pub(crate) id: String,
        pub(crate) kind: String,
        pub(crate) instruction: String,
        #[serde(default)]
        pub(crate) remaining_turns: u32,
        #[serde(default)]
        pub(crate) created_at: u64,
        #[serde(default = "default_true")]
        pub(crate) enabled: bool,
    }

    fn default_true() -> bool {
        true
    }

    /// Model parameters for a chat tab
    #[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
    pub(crate) struct ChatModelParams {
        pub(crate) temperature: f32,
        pub(crate) top_p: f32,
        pub(crate) top_k: u32,
        pub(crate) min_p: f32,
        pub(crate) repeat_penalty: f32,
        pub(crate) max_tokens: Option<u32>,
    }

    impl Default for ChatModelParams {
        fn default() -> Self {
            Self {
                temperature: 0.7,
                top_p: 0.9,
                top_k: 40,
                min_p: 0.01,
                repeat_penalty: 1.0,
                max_tokens: None,
            }
        }
    }

    /// Deserialize explicit_level from bool (legacy), u8, or null.
    fn deserialize_explicit_level<'de, D>(deserializer: D) -> Result<Option<u8>, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = <Option<serde_json::Value> as serde::Deserialize>::deserialize(deserializer)?;
        match value {
            Some(v) if v.is_null() => Ok(None),
            Some(v) => {
                if let Some(n) = v.as_u64() {
                    return Ok(Some(n as u8));
                }
                if let Some(b) = v.as_bool() {
                    return Ok(Some(if b { 1 } else { 0 }));
                }
                if let Some(n) = v.as_i64() {
                    return Ok(Some(n as u8));
                }
                Ok(Some(0))
            }
            None => Ok(None),
        }
    }

    /// Chat tab structure for persistence (legacy flat-file)
    #[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
    pub(crate) struct ChatTab {
        pub(crate) id: String,
        pub(crate) name: String,
        pub(crate) system_prompt: String,
        #[serde(default)]
        pub(crate) ai_name: Option<String>,
        #[serde(default)]
        pub(crate) user_name: Option<String>,
        #[serde(
            default,
            rename = "explicitLevel",
            alias = "explicit_mode",
            alias = "explicit_level",
            deserialize_with = "deserialize_explicit_level"
        )]
        pub(crate) explicit_level: Option<u8>,
        pub(crate) messages: Vec<ChatMessage>,
        #[serde(default)]
        pub(crate) total_input_tokens: Option<u64>,
        #[serde(default)]
        pub(crate) total_output_tokens: Option<u64>,
        #[serde(default)]
        pub(crate) model_params: ChatModelParams,
        #[serde(default)]
        pub(crate) created_at: u64,
        #[serde(default)]
        pub(crate) updated_at: u64,
        #[serde(default)]
        pub(crate) auto_compact: Option<bool>,
        #[serde(default)]
        pub(crate) auto_compact_summarize: Option<bool>,
        #[serde(default)]
        pub(crate) compact_threshold: Option<f32>,
        #[serde(default)]
        pub(crate) compact_mode: Option<String>,
        #[serde(rename = "lastCtxPct", default)]
        pub(crate) last_ctx_pct: Option<f32>,
        #[serde(rename = "activeTemplateId", default)]
        pub(crate) active_template_id: Option<String>,
        #[serde(default)]
        pub(crate) context_notes: Vec<ContextNote>,
        #[serde(default, rename = "sidebarWidth")]
        pub(crate) sidebar_width: u32,
        #[serde(default)]
        pub(crate) quick_guide_active: String,
        #[serde(default)]
        pub(crate) armed_story_beats: Vec<ArmedStoryBeat>,
        #[serde(default)]
        pub(crate) role_boundary_custom: Option<String>,
        #[serde(default)]
        pub(crate) ai_gender: Option<String>,
    }
}

pub(crate) fn routes(ctx: ApiCtx, chat_storage: Arc<ChatStorage>) -> ApiRoute {
    let state = ctx.state;
    let config = ctx.config;

    let list_tabs = api_list_tabs(chat_storage.clone(), config.clone());
    let create_tab = api_create_tab(chat_storage.clone(), config.clone());
    let get_tab = api_get_tab(state.clone(), chat_storage.clone(), config.clone());
    let put_tab = api_put_tab(state.clone(), chat_storage.clone(), config.clone());
    let delete_tab = api_delete_tab(chat_storage.clone(), config.clone());
    let patch_tab_meta = api_patch_tab_meta(chat_storage.clone(), config.clone());
    let append_messages = api_append_messages(state.clone(), chat_storage.clone(), config.clone());
    let reorder_tabs = api_reorder_tabs(chat_storage.clone(), config.clone());
    let search = api_chat_search(chat_storage.clone(), config.clone());
    let archive_tab = api_archive_tab(chat_storage.clone(), config.clone());
    let hide_tab = api_hide_tab(chat_storage.clone(), config.clone());
    let restore_tab = api_restore_tab(chat_storage, config);

    list_tabs
        .or(create_tab)
        .unify()
        .or(get_tab)
        .unify()
        .or(put_tab)
        .unify()
        .or(delete_tab)
        .unify()
        .or(patch_tab_meta)
        .unify()
        .or(append_messages)
        .unify()
        .or(reorder_tabs)
        .unify()
        .or(search)
        .unify()
        .or(archive_tab)
        .unify()
        .or(hide_tab)
        .unify()
        .or(restore_tab)
        .unify()
        .boxed()
}

fn parse_visibility_param(param: &str) -> Vec<TabVisibility> {
    if param.is_empty() || param == "active" {
        vec![TabVisibility::Active]
    } else if param == "all" {
        vec![
            TabVisibility::Active,
            TabVisibility::Archived,
            TabVisibility::Hidden,
        ]
    } else if param == "archived" {
        vec![TabVisibility::Archived]
    } else if param == "hidden" {
        vec![TabVisibility::Hidden]
    } else {
        param
            .split(',')
            .map(|s| s.trim().parse().unwrap_or(TabVisibility::Active))
            .collect()
    }
}

fn now_ts() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

fn new_tab_id() -> String {
    format!("tab_{}", now_ts())
}

fn compute_template_hash(prompt: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(prompt.as_bytes());
    let result = hasher.finalize();
    result
        .iter()
        .take(6)
        .map(|b| format!("{:02x}", b))
        .collect::<String>()
}

// GET /api/chat/tabs
fn api_list_tabs(storage: Arc<ChatStorage>, app_config: Arc<AppConfig>) -> ApiRoute {
    warp::path!("api" / "chat" / "tabs")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and(warp::query::<HashMap<String, String>>())
        .and(with_chat_storage(storage))
        .and_then(
            move |auth: Option<String>, query: HashMap<String, String>, store: Arc<ChatStorage>| {
                let cfg = app_config.clone();
                async move {
                    if !check_api_token(&auth, &cfg) {
                        return Ok(unauthorized_api_token());
                    }
                    let visibilities = parse_visibility_param(
                        &query.get("visibility").cloned().unwrap_or_default(),
                    );
                    match store.list_tabs(&visibilities) {
                        Ok(tabs) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&tabs),
                        )),
                        Err(e) => {
                            eprintln!("list_tabs error: {e}");
                            Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                warp::reply::json(&Vec::<crate::chat_storage::TabMeta>::new()),
                            ))
                        }
                    }
                }
            },
        )
        .boxed()
}

// POST /api/chat/tabs
fn api_create_tab(storage: Arc<ChatStorage>, app_config: Arc<AppConfig>) -> ApiRoute {
    warp::path!("api" / "chat" / "tabs")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(crate::web::safe_json_body::<crate::chat_storage::ChatTabRow>())
        .and(with_chat_storage(storage))
        .and_then(
            move |auth: Option<String>,
                  mut tab: crate::chat_storage::ChatTabRow,
                  store: Arc<ChatStorage>| {
                let cfg = app_config.clone();
                async move {
                    if !check_api_token(&auth, &cfg) {
                        return Ok(unauthorized_api_token());
                    }
                    if tab.id.is_empty() {
                        tab.id = new_tab_id();
                    }
                    tab.created_at = now_ts();
                    tab.updated_at = tab.created_at;
                    if tab.active_template_id.is_some() && tab.template_version_or_hash.is_none() {
                        tab.template_version_or_hash =
                            Some(compute_template_hash(&tab.system_prompt));
                    }
                    match store.create_tab(&tab) {
                        Ok(_) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&tab),
                        )),
                        Err(e) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(
                                &serde_json::json!({"ok":false,"error":e.to_string()}),
                            ),
                        )),
                    }
                }
            },
        )
        .boxed()
}

// GET /api/chat/tabs/:id
fn api_get_tab(state: AppState, storage: Arc<ChatStorage>, app_config: Arc<AppConfig>) -> ApiRoute {
    warp::path!("api" / "chat" / "tabs" / String)
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and(with_chat_storage(storage))
        .and_then(
            move |id: String, auth: Option<String>, store: Arc<ChatStorage>| {
                let cfg = app_config.clone();
                let state = state.clone();
                async move {
                    if !check_api_token(&auth, &cfg) {
                        return Ok(unauthorized_api_token());
                    }
                    match store.get_tab(&id) {
                        Ok(mut tab) => {
                            if !state.ui_settings.lock().unwrap().persist_thinking_content {
                                for msg in &mut tab.messages {
                                    msg.thinking_content = None;
                                }
                            }
                            Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                warp::reply::json(&tab),
                            ))
                        }
                        Err(e) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(
                                &serde_json::json!({"ok":false,"error":e.to_string()}),
                            ),
                        )),
                    }
                }
            },
        )
        .boxed()
}

// PUT /api/chat/tabs/:id
fn api_put_tab(state: AppState, storage: Arc<ChatStorage>, app_config: Arc<AppConfig>) -> ApiRoute {
    warp::path!("api" / "chat" / "tabs" / String)
        .and(warp::put())
        .and(warp::header::optional::<String>("authorization"))
        .and(crate::web::safe_json_body::<crate::chat_storage::ChatTabRow>())
        .and(with_chat_storage(storage))
        .and_then(
            move |id: String,
                  auth: Option<String>,
                  mut tab: crate::chat_storage::ChatTabRow,
                  store: Arc<ChatStorage>| {
                let cfg = app_config.clone();
                let state = state.clone();
                async move {
                    if !check_api_token(&auth, &cfg) {
                        return Ok(unauthorized_api_token());
                    }
                    tab.id = id;
                    tab.updated_at = now_ts();
                    if tab.active_template_id.is_some() && tab.template_version_or_hash.is_none() {
                        tab.template_version_or_hash =
                            Some(compute_template_hash(&tab.system_prompt));
                    }
                    let messages = std::mem::take(&mut tab.messages);
                    let persist_thinking =
                        state.ui_settings.lock().unwrap().persist_thinking_content;
                    let msg_rows: Vec<crate::chat_storage::MessageRow> = messages
                        .into_iter()
                        .enumerate()
                        .map(|(seq, mut m)| {
                            if !persist_thinking {
                                m.thinking_content = None;
                            }
                            crate::chat_storage::MessageRow {
                                seq: seq as i64,
                                tab_id: tab.id.clone(),
                                ..m
                            }
                        })
                        .collect();
                    let result = store
                        .update_tab_meta(&tab)
                        .and_then(|_| store.replace_messages(&tab.id, &msg_rows));
                    match result {
                        Ok(_) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({"ok":true})),
                        )),
                        Err(e) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(
                                &serde_json::json!({"ok":false,"error":e.to_string()}),
                            ),
                        )),
                    }
                }
            },
        )
        .boxed()
}

// DELETE /api/chat/tabs/:id
fn api_delete_tab(storage: Arc<ChatStorage>, app_config: Arc<AppConfig>) -> ApiRoute {
    warp::path!("api" / "chat" / "tabs" / String)
        .and(warp::delete())
        .and(warp::header::optional::<String>("authorization"))
        .and(with_chat_storage(storage))
        .and_then(
            move |id: String, auth: Option<String>, store: Arc<ChatStorage>| {
                let cfg = app_config.clone();
                async move {
                    if !check_api_token(&auth, &cfg) {
                        return Ok(unauthorized_api_token());
                    }
                    match store.delete_tab(&id) {
                        Ok(_) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({"ok":true})),
                        )),
                        Err(e) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(
                                &serde_json::json!({"ok":false,"error":e.to_string()}),
                            ),
                        )),
                    }
                }
            },
        )
        .boxed()
}

// PATCH /api/chat/tabs/:id/meta
fn api_patch_tab_meta(storage: Arc<ChatStorage>, app_config: Arc<AppConfig>) -> ApiRoute {
    warp::path!("api" / "chat" / "tabs" / String / "meta")
        .and(warp::patch())
        .and(warp::header::optional::<String>("authorization"))
        .and(crate::web::safe_json_body::<crate::chat_storage::ChatTabRow>())
        .and(with_chat_storage(storage))
        .and_then(
            move |id: String,
                  auth: Option<String>,
                  mut tab: crate::chat_storage::ChatTabRow,
                  store: Arc<ChatStorage>| {
                let cfg = app_config.clone();
                async move {
                    if !check_api_token(&auth, &cfg) {
                        return Ok(unauthorized_api_token());
                    }
                    tab.id = id;
                    tab.updated_at = now_ts();
                    match store.update_tab_meta(&tab) {
                        Ok(_) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({"ok":true})),
                        )),
                        Err(e) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(
                                &serde_json::json!({"ok":false,"error":e.to_string()}),
                            ),
                        )),
                    }
                }
            },
        )
        .boxed()
}

// POST /api/chat/tabs/:id/messages
fn api_append_messages(
    state: AppState,
    storage: Arc<ChatStorage>,
    app_config: Arc<AppConfig>,
) -> ApiRoute {
    warp::path!("api" / "chat" / "tabs" / String / "messages")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(warp::body::json::<serde_json::Value>())
        .and(with_chat_storage(storage))
        .and_then(
            move |id: String,
                  auth: Option<String>,
                  body: serde_json::Value,
                  store: Arc<ChatStorage>| {
                let cfg = app_config.clone();
                let state = state.clone();
                async move {
                    if !check_api_token(&auth, &cfg) {
                        return Ok(unauthorized_api_token());
                    }
                    let persist_thinking =
                        state.ui_settings.lock().unwrap().persist_thinking_content;
                    let msgs = body["messages"].as_array().cloned().unwrap_or_default();
                    let mut last_id = 0i64;
                    for msg_val in msgs {
                        let msg: crate::chat_storage::MessageRow = serde_json::from_value(msg_val)
                            .unwrap_or_else(|_| crate::chat_storage::MessageRow {
                                tab_id: id.clone(),
                                role: "user".into(),
                                content: "".into(),
                                thinking_content: None,
                                id: 0,
                                timestamp_ms: 0,
                                input_tokens: None,
                                output_tokens: None,
                                cumulative_input_tokens: None,
                                cumulative_output_tokens: None,
                                context_fingerprint: None,
                                compaction_marker: false,
                                variants: None,
                                variant_index: None,
                                seq: 0,
                            });
                        let mut m = msg;
                        if !persist_thinking {
                            m.thinking_content = None;
                        }
                        m.tab_id = id.clone();
                        match store.append_message(&m) {
                            Ok(row_id) => last_id = row_id,
                            Err(e) => eprintln!("append_message error: {e}"),
                        }
                    }
                    Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::reply::json(
                        &serde_json::json!({"ok":true,"last_id":last_id}),
                    )))
                }
            },
        )
        .boxed()
}

// PATCH /api/chat/tabs/order
fn api_reorder_tabs(storage: Arc<ChatStorage>, app_config: Arc<AppConfig>) -> ApiRoute {
    warp::path!("api" / "chat" / "tabs" / "order")
        .and(warp::patch())
        .and(warp::header::optional::<String>("authorization"))
        .and(warp::body::json::<serde_json::Value>())
        .and(with_chat_storage(storage))
        .and_then(
            move |auth: Option<String>, body: serde_json::Value, store: Arc<ChatStorage>| {
                let cfg = app_config.clone();
                async move {
                    if !check_api_token(&auth, &cfg) {
                        return Ok(unauthorized_api_token());
                    }
                    let ids: Vec<String> = body["tab_order"]
                        .as_array()
                        .map(|a| {
                            a.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        })
                        .unwrap_or_default();
                    match store.reorder_tabs(&ids) {
                        Ok(_) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&serde_json::json!({"ok":true})),
                        )),
                        Err(e) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(
                                &serde_json::json!({"ok":false,"error":e.to_string()}),
                            ),
                        )),
                    }
                }
            },
        )
        .boxed()
}

// GET /api/chat/search
fn api_chat_search(storage: Arc<ChatStorage>, app_config: Arc<AppConfig>) -> ApiRoute {
    #[derive(serde::Deserialize)]
    struct SearchParams {
        q: String,
        #[serde(default = "default_limit")]
        limit: usize,
        #[serde(default)]
        offset: usize,
        #[serde(default)]
        visibility: String,
        tab_id: Option<String>,
    }
    fn default_limit() -> usize {
        20
    }

    warp::path!("api" / "chat" / "search")
        .and(warp::get())
        .and(warp::header::optional::<String>("authorization"))
        .and(warp::query::<SearchParams>())
        .and(with_chat_storage(storage))
        .and_then(
            move |auth: Option<String>, p: SearchParams, store: Arc<ChatStorage>| {
                let cfg = app_config.clone();
                async move {
                    if !check_api_token(&auth, &cfg) {
                        return Ok(unauthorized_api_token());
                    }

                    let limit = p.limit.clamp(1, 100);
                    let visibilities = parse_visibility_param(&p.visibility);
                    let tab_id_ref = p.tab_id.as_deref();
                    match store.search(&p.q, limit, p.offset, &visibilities, tab_id_ref) {
                        Ok(results) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(
                            Box::new(warp::reply::json(&results)),
                        ),
                        Err(e) => {
                            eprintln!("search error: {e}");
                            Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                                warp::reply::json(&crate::chat_storage::SearchResultsPage {
                                    results: Vec::new(),
                                    total: 0,
                                    limit,
                                    offset: p.offset,
                                    has_more: false,
                                }),
                            ))
                        }
                    }
                }
            },
        )
        .boxed()
}

fn api_archive_tab(storage: Arc<ChatStorage>, app_config: Arc<AppConfig>) -> ApiRoute {
    warp::path!("api" / "chat" / "tabs" / String)
        .and(warp::post())
        .and(warp::path!("archive"))
        .and(warp::header::optional::<String>("authorization"))
        .and(with_chat_storage(storage))
        .and_then(
            move |id: String, auth: Option<String>, store: Arc<ChatStorage>| {
                let cfg = app_config.clone();
                async move {
                    if !check_api_token(&auth, &cfg) {
                        return Ok(unauthorized_api_token());
                    }
                    match store.set_visibility(&id, &TabVisibility::Archived) {
                        Ok(tab) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&tab),
                        )),
                        Err(_) => Ok(Box::new(warp::reply::with_status(
                            warp::reply::json(
                                &serde_json::json!({"ok": false, "error": "not_found"}),
                            ),
                            warp::http::StatusCode::NOT_FOUND,
                        ))),
                    }
                }
            },
        )
        .boxed()
}

fn api_hide_tab(storage: Arc<ChatStorage>, app_config: Arc<AppConfig>) -> ApiRoute {
    warp::path!("api" / "chat" / "tabs" / String)
        .and(warp::post())
        .and(warp::path!("hide"))
        .and(warp::header::optional::<String>("authorization"))
        .and(with_chat_storage(storage))
        .and_then(
            move |id: String, auth: Option<String>, store: Arc<ChatStorage>| {
                let cfg = app_config.clone();
                async move {
                    if !check_api_token(&auth, &cfg) {
                        return Ok(unauthorized_api_token());
                    }
                    match store.set_visibility(&id, &TabVisibility::Hidden) {
                        Ok(tab) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&tab),
                        )),
                        Err(_) => Ok(Box::new(warp::reply::with_status(
                            warp::reply::json(
                                &serde_json::json!({"ok": false, "error": "not_found"}),
                            ),
                            warp::http::StatusCode::NOT_FOUND,
                        ))),
                    }
                }
            },
        )
        .boxed()
}

fn api_restore_tab(storage: Arc<ChatStorage>, app_config: Arc<AppConfig>) -> ApiRoute {
    warp::path!("api" / "chat" / "tabs" / String)
        .and(warp::post())
        .and(warp::path!("restore"))
        .and(warp::header::optional::<String>("authorization"))
        .and(with_chat_storage(storage))
        .and_then(
            move |id: String, auth: Option<String>, store: Arc<ChatStorage>| {
                let cfg = app_config.clone();
                async move {
                    if !check_api_token(&auth, &cfg) {
                        return Ok(unauthorized_api_token());
                    }
                    match store.set_visibility(&id, &TabVisibility::Active) {
                        Ok(tab) => Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                            warp::reply::json(&tab),
                        )),
                        Err(_) => Ok(Box::new(warp::reply::with_status(
                            warp::reply::json(
                                &serde_json::json!({"ok": false, "error": "not_found"}),
                            ),
                            warp::http::StatusCode::NOT_FOUND,
                        ))),
                    }
                }
            },
        )
        .boxed()
}
