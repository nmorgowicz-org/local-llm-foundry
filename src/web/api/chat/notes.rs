use std::sync::Arc;

use warp::Filter;

use crate::config::AppConfig;
use crate::state::AppState;

use super::super::common::{
    ApiCtx, ApiError, ApiRoute, box_reply, check_api_token, unauthorized_api_token,
};
use super::super::upstream::{
    build_upstream_client, prepare_inference_request, send_helper_request_with_retry,
    upstream_body_stream,
};
use super::suggestions::SuggestionContextMessage;

/// Context notes analysis request
#[derive(serde::Deserialize, Debug)]
pub(crate) struct ContextNotesAnalyzeRequest {
    pub(crate) messages: Vec<SuggestionContextMessage>,
    #[serde(default)]
    pub(crate) system_prompt: Option<String>,
    pub(crate) existing_notes: Vec<super::suggestions::ContextNote>,
    pub(crate) sections: Vec<String>,
}

/// Per-section analysis result
#[derive(serde::Serialize, Debug)]
pub(crate) struct SectionAnalysis {
    pub(crate) section: String,
    pub(crate) suggested: String,
    /// "new" | "current" | "stale"
    pub(crate) status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) reason: Option<String>,
}

/// Context notes analysis response
#[derive(serde::Serialize, Debug)]
pub(crate) struct ContextNotesAnalyzeResponse {
    pub(crate) sections: Vec<SectionAnalysis>,
}

pub(crate) fn routes(ctx: ApiCtx) -> ApiRoute {
    api_analyze_context_notes(ctx.state, ctx.config)
        .map(box_reply)
        .boxed()
}

fn api_analyze_context_notes(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "context-notes" / "analyze")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(warp::body::json::<ContextNotesAnalyzeRequest>())
        .and_then(move |auth: Option<String>, req: ContextNotesAnalyzeRequest| {
            let cfg = app_config.clone();
            let state = state.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }
                let prepared = prepare_inference_request(&state).await?;

                let recent: Vec<_> = req.messages.iter().rev().take(20).rev().collect();
                let convo_text = recent
                    .iter()
                    .map(|m| format!("{}: {}", m.role, m.content))
                    .collect::<Vec<_>>()
                    .join("\n\n");

                let existing_summary = if req.existing_notes.is_empty() {
                    "None".to_string()
                } else {
                    req.existing_notes
                        .iter()
                        .map(|n| format!("[{}] {}", n.section, n.content))
                        .collect::<Vec<_>>()
                        .join("\n")
                };

                let sections_list = req.sections.join(", ");

                let system_prompt_block = req
                    .system_prompt
                    .as_deref()
                    .filter(|s| !s.trim().is_empty())
                    .map(|s| {
                        format!(
                            "\nSYSTEM PROMPT (describes the overall scenario/character setup):\n{s}\n"
                        )
                    })
                    .unwrap_or_default();

                let system_msg = "You are a creative-writing assistant that analyses a conversation and produces structured context notes. \
                    You MUST return valid JSON only — no markdown fences, no explanation, no extra text. \
                    Follow the schema exactly.";

                let user_msg = format!(
                    "Analyse the conversation below and fill in context notes for each section.\n\
                    Sections to analyse: {sections_list}\n\n\
                    For each section:\n\
                    - Write a concise, high-signal note (1-3 sentences) that a language model would find useful.\n\
                    - Compare your suggestion to the EXISTING note for that section (if any).\n\
                    - Set \"status\" to:\n\
                        \"new\"     — no existing note; you are providing a first suggestion\n\
                        \"current\" — existing note still accurately reflects the conversation\n\
                        \"stale\"   — existing note is outdated or contradicted by recent events\n\
                    - If \"stale\", add a short \"reason\" explaining what changed.\n\
                    {system_prompt_block}\n\
                    EXISTING NOTES:\n{existing_summary}\n\n\
                    CONVERSATION (most recent last):\n{convo_text}\n\n\
                    Return ONLY this JSON structure:\n\
                    {{\"sections\":[{{\"section\":\"<name>\",\"suggested\":\"<note text>\",\"status\":\"new|current|stale\",\"reason\":\"<only if stale>\"}}]}}"
                );

                let client = build_upstream_client()?;
                let payload = serde_json::json!({
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    "stream": true,
                    "thinking_budget_tokens": 0,
                    "chat_template_kwargs": {"enable_thinking": false},
                    "temperature": 0.4,
                    "max_tokens": 1024,
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

                        let event: serde_json::Value = match serde_json::from_str(data) {
                            Ok(v) => v,
                            Err(_) => continue,
                        };
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
                        "upstream returned empty analysis".to_string(),
                    )));
                }

                let json_str = {
                    let s = content.trim();
                    let s = s.strip_prefix("```json").unwrap_or(s);
                    let s = s.strip_prefix("```").unwrap_or(s);
                    let s = s.strip_suffix("```").unwrap_or(s);
                    s.trim().to_string()
                };

                let parsed: serde_json::Value = serde_json::from_str(&json_str)
                    .map_err(|e| warp::reject::custom(ApiError::internal(format!(
                        "failed to parse analysis JSON: {e} — raw: {json_str}"
                    ))))?;

                let sections_arr = parsed.get("sections")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| warp::reject::custom(ApiError::internal(
                        "analysis JSON missing 'sections' array".to_string()
                    )))?;

                let mut sections: Vec<SectionAnalysis> = Vec::new();
                for entry in sections_arr {
                    let section =
                        entry.get("section").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    let suggested =
                        entry.get("suggested").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    let status =
                        entry.get("status").and_then(|v| v.as_str()).unwrap_or("new").to_string();
                    let reason =
                        entry.get("reason").and_then(|v| v.as_str()).map(|s| s.to_string());

                    if section.is_empty() || suggested.is_empty() {
                        continue;
                    }

                    sections.push(SectionAnalysis {
                        section,
                        suggested,
                        status,
                        reason,
                    });
                }

                if sections.is_empty() {
                    return Err(warp::reject::custom(ApiError::gateway(
                        "analysis returned no usable section data".to_string(),
                    )));
                }

                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(
                    warp::reply::json(&ContextNotesAnalyzeResponse { sections }),
                ))
            }
        })
}
