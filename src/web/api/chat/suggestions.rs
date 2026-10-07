use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use warp::Filter;

use crate::chat_storage::ChatStorage;
use crate::config::AppConfig;
use crate::state::AppState;

use super::super::common::{
    ApiCtx, ApiError, ApiRoute, box_reply, check_api_token, unauthorized_api_token,
};
use super::super::upstream::{
    build_upstream_client, prepare_inference_request, send_helper_request_with_retry,
    upstream_body_stream,
};
use super::guided::{is_json_reasoning, strip_inline_thinking};

/// Context note for guided generation (character details, setting info, etc.)
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub(crate) struct ContextNote {
    pub(crate) section: String,
    pub(crate) content: String,
    #[serde(default)]
    pub(crate) created_at: u64,
}

/// Suggestion request for guided generation
#[derive(serde::Deserialize, Debug)]
pub(crate) struct SuggestionRequest {
    pub(crate) tab_id: String,
    pub(crate) category: String,
    #[serde(default)]
    pub(crate) count: Option<u32>,
    #[serde(default)]
    pub(crate) context_depth: Option<u32>,
    #[serde(default)]
    pub(crate) messages: Option<Vec<SuggestionContextMessage>>,
    #[serde(default)]
    pub(crate) system_prompt: Option<String>,
    #[serde(default)]
    pub(crate) context_notes: Option<Vec<ContextNote>>,
    #[serde(default)]
    pub(crate) quick_guide_active: Option<String>,
    pub(crate) prompt: Option<String>,
}

#[derive(serde::Deserialize, Clone, Debug)]
pub(crate) struct SuggestionContextMessage {
    pub(crate) role: String,
    pub(crate) content: String,
}

/// Suggestion response
#[derive(serde::Serialize, Debug)]
pub(crate) struct SuggestionResponse {
    pub(crate) suggestions: Vec<String>,
    #[serde(default)]
    pub(crate) cards: Vec<SuggestionCard>,
    pub(crate) category: String,
    pub(crate) count: u32,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub(crate) struct SuggestionCard {
    #[serde(rename = "type")]
    pub(crate) suggestion_type: String,
    pub(crate) title: String,
    pub(crate) effect: String,
    #[serde(default)]
    pub(crate) detail: String,
}

/// Keyword generation request
#[derive(serde::Deserialize, Debug)]
pub(crate) struct KeywordRequest {
    pub(crate) category: String,
}

/// Keyword generation response
#[derive(serde::Serialize, Debug)]
pub(crate) struct KeywordResponse {
    pub(crate) keywords: Vec<String>,
}

pub(crate) fn routes(ctx: ApiCtx, chat_storage: Arc<ChatStorage>) -> ApiRoute {
    let chat_suggestions =
        api_chat_suggestions(ctx.state.clone(), ctx.config.clone(), chat_storage.clone());
    let generate_keywords = api_generate_keywords(ctx.state, ctx.config);

    generate_keywords
        .map(box_reply)
        .or(chat_suggestions.map(box_reply))
        .unify()
        .boxed()
}

fn default_suggestions_output_contract(count: u32) -> String {
    format!(
        "\n\nFINAL OUTPUT REQUIREMENTS:\n\
Return ONLY valid JSON. Do not include reasoning, planning, commentary, markdown, code fences, or preamble.\n\
Return exactly {count} items in this shape:\n\
{{\"suggestions\":[{{\"title\":\"Short Title\",\"description\":\"One concise actionable next beat.\"}}]}}\n\
Rules:\n\
- `title` must be plain text, under 8 words, and user-facing.\n\
- `description` must be plain text, concise, specific, and actionable.\n\
- Do not output keys other than `suggestions`, `title`, and `description`.\n\
- Do not mention your thinking process.\n\
- Do not number the suggestions.\n"
    )
}

fn director_output_contract(count: u32) -> String {
    format!(
        "\n\nFINAL OUTPUT REQUIREMENTS:\n\
Return ONLY valid JSON. Do not include reasoning, planning, commentary, markdown, code fences, or preamble.\n\
Return exactly {count} items in this shape:\n\
{{\"suggestions\":[{{\"type\":\"pressure\",\"title\":\"Short Title\",\"effect\":\"Short immediate effect line.\",\"detail\":\"One concise sentence explaining how the assistant should continue the scene.\"}}]}}\n\
Rules:\n\
- `type` must be one of: pressure, reveal, escalation, interruption, twist, tone-shift, reversal, intimacy, investigation, confrontation.\n\
- `title` must be plain text, under 8 words, and user-facing.\n\
- `effect` must be a compact summary line under 12 words.\n\
- `detail` must be one concise actionable sentence.\n\
- Do not output keys other than `suggestions`, `type`, `title`, `effect`, and `detail`.\n\
- Do not mention your thinking process.\n\
- Do not number the suggestions.\n"
    )
}

fn suggestions_output_contract(category: &str, count: u32) -> String {
    if category == "director" {
        director_output_contract(count)
    } else {
        default_suggestions_output_contract(count)
    }
}

fn suggestion_temperature(category: &str) -> f32 {
    match category {
        "director" => 0.75,
        "general" => 0.8,
        "plot-twist" => 0.95,
        "new-character" => 0.9,
        "explicit" => 0.9,
        "action" | "comedy" | "fantasy" | "horror" | "mystery" | "noir" | "romance" | "sci-fi"
        | "thriller" | "character" => 0.85,
        _ => 0.85,
    }
}

/// Load prompts from static/prompts directory
fn load_prompts_from_files() -> HashMap<String, String> {
    let mut prompts = HashMap::new();

    let prompts_dir = PathBuf::from("static/prompts");

    if let Ok(entries) = fs::read_dir(&prompts_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "md")
                && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
                && let Ok(content) = fs::read_to_string(&path)
            {
                prompts.insert(stem.to_string(), content);
            }
        }
    }

    prompts
}

fn api_generate_keywords(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "keywords" / "generate")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(warp::body::json::<KeywordRequest>())
        .and_then(move |auth: Option<String>, req: KeywordRequest| {
            let cfg = app_config.clone();
            let state = state.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }
                let prepared = prepare_inference_request(&state).await?;

                let prompt = format!(
                    "Generate 3-5 focus keywords for a story category called \"{}\". Return only the keywords, separated by commas. No explanation.",
                    req.category
                );

                let client = build_upstream_client()?;
                let payload = serde_json::json!({
                    "messages": [
                        {"role": "system", "content": "You generate focus keywords. Return only the keywords, comma-separated, with no explanation."},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": true,
                    "thinking_budget_tokens": 0,
                    "chat_template_kwargs": {"enable_thinking": false},
                    "temperature": 0.7,
                    "max_tokens": 128,
                });
                let payload = prepared.map_chat_body(
                    &serde_json::to_vec(&payload).map_err(|error| {
                        warp::reject::custom(ApiError::internal(error.to_string()))
                    })?,
                )?;
                let url = prepared.url.clone();

                let response = send_helper_request_with_retry(|| {
                    prepared.authenticate(
                        client
                            .post(&url)
                            .header("Content-Type", "application/json")
                            .body(payload.clone()),
                    )
                })
                .await?;

                use futures_util::StreamExt;
                let mut upstream = upstream_body_stream(response);
                let mut buf = String::new();
                let mut content = String::new();

                while let Some(chunk) = upstream.next().await {
                    let chunk = chunk.map_err(warp::reject::custom)?;
                    buf.push_str(&String::from_utf8_lossy(&chunk));

                    while let Some(pos) = buf.find('\n') {
                        let line = buf[..pos].trim().to_string();
                        buf = buf[pos + 1..].to_string();

                        let Some(data) = line.strip_prefix("data: ") else { continue; };
                        let data = data.trim();
                        if data.is_empty() || data == "[DONE]" { continue; }

                        let event: serde_json::Value = serde_json::from_str(data)
                            .map_err(|e| warp::reject::custom(ApiError::internal(e.to_string())))?;
                        if let Some(delta) = event.get("choices")
                            .and_then(|c| c.as_array())
                            .and_then(|c| c.first())
                            .and_then(|c| c.get("delta"))
                            .and_then(|d| d.get("content"))
                            .and_then(|c| c.as_str())
                        {
                            content.push_str(delta);
                        }
                    }
                }

                let content = content.trim().to_string();
                if content.is_empty() {
                    return Err(warp::reject::custom(ApiError::gateway(
                        "upstream returned empty keyword content".to_string(),
                    )));
                }

                let keywords: Vec<String> = content
                    .split(',')
                    .map(|k| k.trim().to_string())
                    .filter(|k| !k.is_empty())
                    .collect();

                if keywords.is_empty() {
                    return Err(warp::reject::custom(ApiError::gateway(
                        "upstream returned no parseable keywords".to_string(),
                    )));
                }

                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                    warp::reply::json(&KeywordResponse { keywords }),
                ))
            }
        })
}

fn api_chat_suggestions(
    state: AppState,
    app_config: Arc<AppConfig>,
    chat_storage: Arc<ChatStorage>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "chat" / "suggestions")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(warp::body::json::<SuggestionRequest>())
        .and_then(move |auth: Option<String>, req: SuggestionRequest| {
            let cfg = app_config.clone();
            let state = state.clone();
            let chat_storage = chat_storage.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }
                let (all_messages, system_prompt, context_notes, quick_guide_active) =
                    if let Some(messages) = req.messages.clone() {
                        (
                            messages,
                            req.system_prompt.clone().unwrap_or_default(),
                            req.context_notes.clone().unwrap_or_default(),
                            req.quick_guide_active.clone().unwrap_or_default(),
                        )
                    } else {
                        let tab = chat_storage
                            .get_tab(&req.tab_id)
                            .ok()
                            .ok_or(warp::reject::not_found())?;

                        let messages: Vec<SuggestionContextMessage> = tab
                            .messages
                            .iter()
                            .map(|msg| SuggestionContextMessage {
                                role: msg.role.clone(),
                                content: msg.content.clone(),
                            })
                            .collect();

                        let system_prompt = tab.system_prompt.clone();
                        let context_notes: Vec<ContextNote> =
                            serde_json::from_value(tab.context_notes.clone()).unwrap_or_default();

                        (messages, system_prompt, context_notes, String::new())
                    };

                let depth = req.context_depth.unwrap_or(10) as usize;
                let mut messages: Vec<SuggestionContextMessage> = all_messages
                    .iter()
                    .rev()
                    .take(depth)
                    .cloned()
                    .collect();
                messages.reverse();

                let mut context = String::new();
                let trimmed_system_prompt = system_prompt.trim();
                if !trimmed_system_prompt.is_empty() {
                    context.push_str("System Prompt:\n");
                    context.push_str(trimmed_system_prompt);
                    context.push_str("\n\n");
                }

                let filtered_notes: Vec<ContextNote> = context_notes
                    .into_iter()
                    .filter(|note| !note.content.trim().is_empty())
                    .collect();
                if !filtered_notes.is_empty() {
                    context.push_str("Context Notes:\n");
                    for note in filtered_notes {
                        context.push_str(&format!(
                            "- {}: {}\n",
                            note.section.trim(),
                            note.content.trim()
                        ));
                    }
                    context.push('\n');
                }

                let trimmed_quick_guide = quick_guide_active.trim();
                if !trimmed_quick_guide.is_empty() {
                    context.push_str("Active Quick Guide:\n");
                    context.push_str(trimmed_quick_guide);
                    context.push_str("\n\n");
                }

                for msg in &messages {
                    let role = if msg.role == "user" { "User" } else { "Assistant" };
                    context.push_str(&format!("{}: {}\n", role, msg.content));
                }

                let file_prompts = load_prompts_from_files();

                let default_prompt = req.prompt.or_else(|| {
                    file_prompts.get(&req.category).cloned().or_else(|| {
                        match req.category.as_str() {
                            "plot-twist" => Some("You are a plot twist specialist. Based on the conversation below, suggest {} unexpected, surprising events that could happen next.\n\nFormat as a numbered list. Prioritize: betrayals, revelations, power reversals, unexpected arrivals, hidden truths.\n\n[conversation context]".to_string()),
                            "new-character" => Some("You are a character introduction specialist. Based on the conversation below, suggest {} new characters that could enter the story.\n\nFormat as: [Character Name]: [Brief description and how they connect to current story]\n\n[conversation context]".to_string()),
                            _ => Some("You are a creative brainstorming partner. Based on the conversation below, suggest {} varied, actionable next steps the user could take.\n\nFormat as a numbered list. Prioritize variety: dialogue, action, investigation, social, creative approaches.\n\n[conversation context]".to_string()),
                        }
                    })
                });

                let count = req.count.unwrap_or(5);
                let prompt = default_prompt
                    .unwrap_or_default()
                    .replace("{count}", &count.to_string())
                    .replace("[STORY CONTEXT]", &context)
                    .replace("[conversation context]", &context)
                    .replace("[CONVERSATION CONTEXT]", &context)
                    + &suggestions_output_contract(&req.category, count);
                let temperature = suggestion_temperature(&req.category);

                let suggestion_messages = vec![
                    serde_json::json!({"role": "system", "content": "You are a helpful creative writing assistant. Keep all reasoning internal. Return only the final answer in the exact requested format with no preamble or analysis."}),
                    serde_json::json!({"role": "user", "content": prompt}),
                ];

                let prepared = prepare_inference_request(&state).await?;
                let client = build_upstream_client()?;
                let payload = serde_json::json!({
                    "messages": suggestion_messages,
                    "stream": true,
                    "thinking_budget_tokens": 0,
                    "chat_template_kwargs": {
                        "enable_thinking": false
                    },
                    "temperature": temperature,
                    "max_tokens": 512,
                });
                let payload = prepared.map_chat_body(
                    &serde_json::to_vec(&payload).map_err(|error| {
                        warp::reject::custom(ApiError::internal(error.to_string()))
                    })?,
                )?;
                let url = prepared.url.clone();

                let response = send_helper_request_with_retry(|| {
                    prepared.authenticate(
                        client
                            .post(&url)
                            .header("Content-Type", "application/json")
                            .body(payload.clone()),
                    )
                })
                .await?;

                use futures_util::StreamExt;

                let mut upstream = upstream_body_stream(response);
                let mut buf = String::new();
                let mut content = String::new();
                let mut reasoning_content = String::new();

                while let Some(chunk) = upstream.next().await {
                    let chunk = chunk.map_err(warp::reject::custom)?;
                    buf.push_str(&String::from_utf8_lossy(&chunk));

                    while let Some(pos) = buf.find('\n') {
                        let line = buf[..pos].trim().to_string();
                        buf = buf[pos + 1..].to_string();

                        let Some(data) = line.strip_prefix("data: ") else {
                            continue;
                        };
                        let data = data.trim();
                        if data.is_empty() || data == "[DONE]" {
                            continue;
                        }

                        let event: serde_json::Value = serde_json::from_str(data)
                            .map_err(|e| warp::reject::custom(ApiError::internal(e.to_string())))?;
                        if let Some(delta) = event.get("choices")
                            .and_then(|c| c.as_array())
                            .and_then(|c| c.first())
                            .and_then(|c| c.get("delta"))
                            .and_then(|d| d.get("content"))
                            .and_then(|c| c.as_str())
                        {
                            content.push_str(delta);
                        } else if let Some(reasoning) = event.get("choices")
                            .and_then(|c| c.as_array())
                            .and_then(|c| c.first())
                            .and_then(|c| c.get("delta"))
                            .and_then(|d| d.get("reasoning_content"))
                            .and_then(|c| c.as_str())
                        {
                            reasoning_content.push_str(reasoning);
                        } else if let Some(message) = event.get("choices")
                            .and_then(|c| c.as_array())
                            .and_then(|c| c.first())
                            .and_then(|c| c.get("message"))
                            .and_then(|m| m.get("content"))
                            .and_then(|c| c.as_str())
                        {
                            content.push_str(message);
                        }
                    }
                }

                let content = if content.trim().is_empty() {
                    reasoning_content.trim()
                } else {
                    content.trim()
                };
                if content.is_empty() {
                    return Err(warp::reject::custom(ApiError::gateway(
                        "upstream returned empty suggestion content".to_string(),
                    )));
                }

                let cards = if req.category == "director" {
                    parse_director_cards(content)
                } else {
                    Vec::new()
                };
                let suggestions = if req.category == "director" && !cards.is_empty() {
                    cards
                        .iter()
                        .map(|card| format!("{}\n{}", card.title, card.detail))
                        .collect()
                } else {
                    parse_suggestions(content)
                };
                if suggestions.is_empty() {
                    return Err(warp::reject::custom(ApiError::gateway(
                        "upstream returned no parseable suggestions".to_string(),
                    )));
                }
                let count = if req.category == "director" && !cards.is_empty() {
                    cards.len() as u32
                } else {
                    suggestions.len() as u32
                };

                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                    warp::reply::json(&SuggestionResponse {
                        suggestions,
                        cards,
                        category: req.category,
                        count,
                    }),
                ))
            }
        })
}

fn parse_suggestions(input: &str) -> Vec<String> {
    let text = strip_inline_thinking(input);
    fn clean_fragment(value: &str) -> String {
        value
            .replace("**", "")
            .replace(['*', '`'], "")
            .trim()
            .to_string()
    }

    fn extract_json_value(s: &str) -> Option<serde_json::Value> {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(s) {
            return Some(v);
        }
        if let Some(start) = s.find('{')
            && let Some(end) = s.rfind('}')
            && let Ok(v) = serde_json::from_str::<serde_json::Value>(&s[start..=end])
        {
            return Some(v);
        }
        None
    }

    if let Some(value) = extract_json_value(&text) {
        let suggestions_json = value
            .get("suggestions")
            .and_then(|v| v.as_array())
            .cloned()
            .or_else(|| value.as_array().cloned());

        if let Some(entries) = suggestions_json {
            let extracted: Vec<String> = entries
                .into_iter()
                .filter_map(|entry| {
                    if let Some(obj) = entry.as_object() {
                        let title = obj
                            .get("title")
                            .and_then(|v| v.as_str())
                            .map(clean_fragment)
                            .unwrap_or_default();
                        let description = obj
                            .get("description")
                            .and_then(|v| v.as_str())
                            .map(clean_fragment)
                            .unwrap_or_default();
                        if title.is_empty() || description.is_empty() {
                            return None;
                        }
                        return Some(format!("{}\n{}", title, description));
                    }

                    entry
                        .as_str()
                        .map(clean_fragment)
                        .filter(|value| !value.is_empty())
                })
                .collect();

            if !extracted.is_empty() {
                return extracted;
            }
        }
    }

    let blocks: Vec<&str> = text.split("---").collect();
    let mut suggestions = Vec::new();

    for block in blocks {
        let lines: Vec<&str> = block
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect();
        if lines.len() >= 2 {
            let title = clean_fragment(lines[0]);
            let description = clean_fragment(&lines[1..].join(" "));
            let title_lower = title.to_ascii_lowercase();
            let description_lower = description.to_ascii_lowercase();
            let is_meta = title_lower.contains("thinking process")
                || description_lower.contains("analyze user input")
                || description_lower.contains("output format")
                || description_lower.contains("guidelines")
                || description_lower.contains("deconstruct key elements");
            if title.len() > 2 && !description.is_empty() && !is_meta {
                suggestions.push(format!("{}\n{}", title, description));
            }
        }
    }

    if suggestions.is_empty() {
        let blocks: Vec<&str> = text.split("\n\n").collect();
        for block in &blocks {
            let lines: Vec<&str> = block
                .lines()
                .map(|l| l.trim())
                .filter(|l| !l.is_empty())
                .collect();
            if lines.len() < 2 {
                continue;
            }
            let title = clean_fragment(lines[0]);
            if title.len() <= 2 {
                continue;
            }
            let description = clean_fragment(&lines[1..].join(" "));
            let title_lower = title.to_ascii_lowercase();
            let is_meta =
                title_lower.contains("output format") || title_lower.contains("guidelines");
            if !title.is_empty() && !description.is_empty() && !is_meta {
                suggestions.push(format!("{}\n{}", title, description));
            }
        }
        if !suggestions.is_empty() {
            return suggestions;
        }
    }

    if suggestions.is_empty() {
        let normalized = text.replace('\n', " ");
        let brainstorm_slice = normalized
            .split("Brainstorming Suggestions")
            .nth(1)
            .unwrap_or(&normalized);
        let inline_ideas = regex::Regex::new(
            r"(?:Idea|Suggestion)\s*\d+\s*(?:\(([^)]+)\))?:\s*(.+?)(?:\s+-\s+\*?(?:Idea|Suggestion)\s*\d+\s*(?:\(|:)|$)",
        )
        .ok();
        if let Some(re) = inline_ideas {
            let extracted: Vec<String> = re
                .captures_iter(brainstorm_slice)
                .filter_map(|caps| {
                    let raw_description = caps.get(2)?.as_str();
                    let description = clean_fragment(raw_description);
                    if description.is_empty() {
                        return None;
                    }
                    let raw_title = caps
                        .get(1)
                        .map(|m| m.as_str())
                        .filter(|title| !title.trim().is_empty())
                        .unwrap_or("Next Beat");
                    Some(format!("{}\n{}", clean_fragment(raw_title), description))
                })
                .collect();
            if !extracted.is_empty() {
                return extracted;
            }
        }

        let idea_bullets = regex::Regex::new(
            r"(?m)^\s*[-*]\s*\*?(?:Idea|Suggestion)\s*\d+\s*(?:\(([^)]+)\))?:\*?\s*(.+)$",
        )
        .ok();
        if let Some(re) = idea_bullets {
            let extracted: Vec<String> = text
                .lines()
                .filter_map(|line| re.captures(line))
                .filter_map(|caps| {
                    let raw_description = caps.get(2)?.as_str();
                    let description = clean_fragment(raw_description);
                    if description.is_empty() {
                        return None;
                    }
                    let raw_title = caps
                        .get(1)
                        .map(|m| m.as_str())
                        .filter(|title| !title.trim().is_empty())
                        .unwrap_or("Next Beat");
                    Some(format!("{}\n{}", clean_fragment(raw_title), description))
                })
                .collect();
            if !extracted.is_empty() {
                return extracted;
            }
        }
    }

    if suggestions.is_empty() {
        let numbered = regex::Regex::new(r"^\d+[\.\)]\s*(.+)$").ok();
        if let Some(re) = numbered {
            let numbered_suggestions: Vec<String> = text
                .lines()
                .filter_map(|line| re.captures(line))
                .filter_map(|caps| caps.get(1))
                .map(|m| clean_fragment(m.as_str()))
                .filter(|s| !s.is_empty())
                .filter(|s| {
                    let lowercase = s.to_ascii_lowercase();
                    !lowercase.contains("analyze user input")
                        && !lowercase.contains("output format")
                        && !lowercase.contains("guidelines")
                        && !is_json_reasoning(s)
                })
                .collect();
            if !numbered_suggestions.is_empty() {
                return numbered_suggestions;
            }
        }
    }

    if suggestions.is_empty() {
        let bullets = regex::Regex::new(r"^[-*]\s+(.+)$").ok();
        if let Some(re) = bullets {
            let bullet_suggestions: Vec<String> = text
                .lines()
                .filter_map(|line| re.captures(line))
                .filter_map(|caps| caps.get(1))
                .map(|m| clean_fragment(m.as_str()))
                .filter(|s| !s.is_empty())
                .filter(|s| {
                    let lowercase = s.to_ascii_lowercase();
                    !lowercase.starts_with("role:")
                        && !lowercase.starts_with("task:")
                        && !lowercase.starts_with("goal:")
                        && !lowercase.starts_with("output format:")
                })
                .collect();
            if !bullet_suggestions.is_empty() {
                return bullet_suggestions;
            }
        }
    }

    if suggestions.is_empty() {
        suggestions = text
            .lines()
            .map(clean_fragment)
            .filter(|s| !s.is_empty() && s.len() > 2)
            .filter(|s| !s.starts_with('['))
            .filter(|s| {
                let lowercase = s.to_ascii_lowercase();
                !lowercase.contains("here's a thinking process")
                    && !lowercase.contains("analyze user input")
                    && !lowercase.contains("output format")
                    && !lowercase.contains("guidelines")
            })
            .collect();
    }

    fn is_assistant_meta(text: &str) -> bool {
        let t = text.to_ascii_lowercase();
        (t.contains("the assistant")
            || t.contains("assistant will")
            || t.contains("assistant prompts")
            || t.contains("assistant suggests"))
            && (t.len() > 80)
    }

    fn extract_short_title(line: &str) -> Option<String> {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.contains(']') {
            let after = trimmed.split_once(']')?.1.trim();
            let words: Vec<&str> = after.split_whitespace().take(10).collect();
            if words.len() >= 2 {
                let candidate = words.join(" ");
                let cand_lower = candidate.to_ascii_lowercase();
                if cand_lower.contains("the assistant") || cand_lower.contains("assistant will") {
                    return None;
                }
                if candidate.len() > 15 && candidate.len() < 120 {
                    return Some(candidate);
                }
            }
        }
        None
    }

    if suggestions.len() == 1 && suggestions[0].len() > 400 {
        let big = &suggestions[0];
        let mut candidates = Vec::new();

        let bullets = regex::Regex::new(r"^[-*•]\s+(.+)$").ok();
        if let Some(re) = bullets {
            for line in big.lines() {
                if let Some(m) = re.captures(line).and_then(|c| c.get(1)) {
                    let t = clean_fragment(m.as_str());
                    if t.len() > 20 && t.len() < 300 && !is_assistant_meta(&t) {
                        candidates.push(t);
                    }
                }
            }
        }

        if candidates.is_empty() {
            for line in big.lines() {
                if let Some(title) = extract_short_title(line) {
                    candidates.push(title);
                }
            }
        }

        if candidates.is_empty() {
            for line in big.lines() {
                let t = clean_fragment(line);
                if t.len() > 20 && t.len() < 250 && !is_assistant_meta(&t) {
                    candidates.push(t);
                }
            }
        }

        if candidates.is_empty() {
            let truncated = if big.len() > 250 {
                let mut end = 250;
                while end < big.len() && !big.is_char_boundary(end) {
                    end += 1;
                }
                format!("{}…", &big[..end])
            } else {
                big.clone()
            };
            suggestions = vec![truncated];
        } else {
            suggestions = candidates.into_iter().take(6).collect();
        }
    }

    let bracket_re = regex::Regex::new(r"\[").unwrap();
    let mut cleaned = Vec::new();

    for s in suggestions {
        let trimmed = s.trim().to_string();
        if trimmed.len() < 20 {
            if !cleaned.contains(&trimmed) {
                cleaned.push(trimmed);
            }
            continue;
        }

        if trimmed.len() > 300 && trimmed.matches('[').count() >= 2 {
            let mut parts: Vec<usize> = Vec::new();
            let mut last = 0;
            for m in bracket_re.find_iter(&trimmed) {
                if m.start() > last {
                    continue;
                }
                parts.push(m.start());
                last = m.start();
            }

            if parts.len() >= 2 {
                for (i, &start) in parts.iter().enumerate() {
                    let end = if i + 1 < parts.len() {
                        parts[i + 1]
                    } else {
                        trimmed.len()
                    };
                    let part = trimmed[start..end].trim().to_string();
                    if let Some(title) = extract_short_title(&part)
                        && !cleaned.contains(&title)
                    {
                        cleaned.push(title);
                    }
                }
                if !cleaned.is_empty() {
                    continue;
                }
            }
        }

        if let Some(title) = extract_short_title(&trimmed) {
            if !cleaned.contains(&title) {
                cleaned.push(title);
            }
            continue;
        }

        let lower = trimmed.to_ascii_lowercase();
        if (lower.contains("the assistant")
            || lower.contains("assistant will")
            || lower.contains("assistant prompts")
            || lower.contains("assistant suggests"))
            && trimmed.len() > 120
        {
            let words: Vec<&str> = trimmed.split_whitespace().take(10).collect();
            let candidate = words.join(" ");
            let cand_lower = candidate.to_ascii_lowercase();
            if cand_lower.contains("the assistant") {
                continue;
            }
            if candidate.len() >= 15 && candidate.len() <= 120 && !cleaned.contains(&candidate) {
                cleaned.push(candidate);
            }
            continue;
        }

        if trimmed.len() <= 150 && !cleaned.contains(&trimmed) {
            cleaned.push(trimmed);
        }
    }

    if cleaned.is_empty() {
        let short = if text.len() > 180 {
            let mut end = 180;
            while end < text.len() && !text.is_char_boundary(end) {
                end += 1;
            }
            format!("{}…", &text[..end])
        } else {
            text.clone()
        };
        return vec![short];
    }

    cleaned.into_iter().take(6).collect()
}

fn parse_director_cards(input: &str) -> Vec<SuggestionCard> {
    let text = strip_inline_thinking(input);
    fn clean_fragment(value: &str) -> String {
        value
            .replace("**", "")
            .replace(['*', '`'], "")
            .trim()
            .to_string()
    }

    fn normalize_type(value: &str) -> String {
        let normalized = value.trim().to_ascii_lowercase().replace(['_', ' '], "-");
        match normalized.as_str() {
            "revelation" => "reveal".to_string(),
            "pressure" | "reveal" | "escalation" | "interruption" | "twist" | "tone-shift"
            | "reversal" | "intimacy" | "investigation" | "confrontation" => normalized,
            _ => "pressure".to_string(),
        }
    }

    fn infer_type(title: &str, body: &str) -> String {
        let haystack = format!("{} {}", title, body).to_ascii_lowercase();
        if haystack.contains("reveal") || haystack.contains("truth") || haystack.contains("secret")
        {
            return "reveal".to_string();
        }
        if haystack.contains("interrupt")
            || haystack.contains("arrives")
            || haystack.contains("appears")
        {
            return "interruption".to_string();
        }
        if haystack.contains("twist")
            || haystack.contains("betray")
            || haystack.contains("reversal")
        {
            return "twist".to_string();
        }
        if haystack.contains("close-quarters")
            || haystack.contains("lunges")
            || haystack.contains("fight")
            || haystack.contains("draw")
            || haystack.contains("gun")
        {
            return "confrontation".to_string();
        }
        if haystack.contains("intimate") || haystack.contains("quiet") || haystack.contains("soft")
        {
            return "intimacy".to_string();
        }
        if haystack.contains("investigate")
            || haystack.contains("photo")
            || haystack.contains("clue")
        {
            return "investigation".to_string();
        }
        if haystack.contains("escalate")
            || haystack.contains("danger")
            || haystack.contains("violence")
            || haystack.contains("shot")
        {
            return "escalation".to_string();
        }
        "pressure".to_string()
    }

    fn split_effect_detail(description: &str) -> (String, String) {
        let trimmed = clean_fragment(description);
        if trimmed.is_empty() {
            return (String::new(), String::new());
        }
        let mut parts = trimmed.splitn(2, ". ");
        let first = parts.next().unwrap_or("").trim().trim_end_matches('.');
        let rest = parts.next().unwrap_or("").trim();
        let effect = first.to_string();
        let detail = if rest.is_empty() {
            first.to_string()
        } else {
            rest.to_string()
        };
        (effect, detail)
    }

    fn extract_json_value_director(s: &str) -> Option<serde_json::Value> {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(s) {
            return Some(v);
        }
        if let Some(start) = s.find('{')
            && let Some(end) = s.rfind('}')
            && let Ok(v) = serde_json::from_str::<serde_json::Value>(&s[start..=end])
        {
            return Some(v);
        }
        None
    }

    if let Some(value) = extract_json_value_director(&text) {
        let suggestions_json = value
            .get("suggestions")
            .and_then(|v| v.as_array())
            .cloned()
            .or_else(|| value.as_array().cloned());

        if let Some(entries) = suggestions_json {
            let cards: Vec<SuggestionCard> = entries
                .into_iter()
                .filter_map(|entry| {
                    let obj = entry.as_object()?;
                    let title = clean_fragment(obj.get("title")?.as_str()?);
                    if title.is_empty() {
                        return None;
                    }
                    let effect = obj
                        .get("effect")
                        .and_then(|v| v.as_str())
                        .map(clean_fragment)
                        .unwrap_or_default();
                    let detail = obj
                        .get("detail")
                        .or_else(|| obj.get("description"))
                        .and_then(|v| v.as_str())
                        .map(clean_fragment)
                        .unwrap_or_default();
                    let suggestion_type = obj
                        .get("type")
                        .and_then(|v| v.as_str())
                        .map(normalize_type)
                        .unwrap_or_else(|| infer_type(&title, &format!("{} {}", effect, detail)));
                    let fallback_effect = if effect.is_empty() {
                        title.clone()
                    } else {
                        effect
                    };

                    let looks_like_json = {
                        let d = detail.to_ascii_lowercase();
                        d.contains("suggestions")
                            || d.contains("type")
                            || d.contains("effect")
                            || d.contains("detail")
                    };
                    let fallback_detail =
                        if detail.is_empty() || (detail.len() > 200 && looks_like_json) {
                            fallback_effect.clone()
                        } else {
                            detail
                        };
                    Some(SuggestionCard {
                        suggestion_type,
                        title,
                        effect: fallback_effect,
                        detail: fallback_detail,
                    })
                })
                .collect();
            if !cards.is_empty() {
                return cards;
            }
        }
    }

    let raw_suggestions = parse_suggestions(&text);

    let suggestions = if raw_suggestions.len() == 1 && is_json_reasoning(&raw_suggestions[0]) {
        Vec::new()
    } else {
        raw_suggestions
    };

    suggestions
        .into_iter()
        .filter_map(|entry| {
            let mut parts = entry.splitn(2, '\n');
            let title = clean_fragment(parts.next().unwrap_or(""));
            if title.is_empty() {
                return None;
            }
            let description = clean_fragment(parts.next().unwrap_or(""));
            let (effect, detail) = split_effect_detail(&description);
            Some(SuggestionCard {
                suggestion_type: infer_type(&title, &description),
                title,
                effect: if effect.is_empty() {
                    description.clone()
                } else {
                    effect
                },
                detail: if detail.is_empty() {
                    description
                } else {
                    detail
                },
            })
        })
        .collect()
}
