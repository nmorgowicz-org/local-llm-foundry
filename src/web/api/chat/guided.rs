use std::sync::Arc;

use bytes::Bytes;
use warp::Filter;

use crate::config::AppConfig;
use crate::state::AppState;

use super::super::common::{ApiCtx, ApiRoute, box_reply, check_api_token, unauthorized_api_token};
use super::super::upstream::{
    build_upstream_client, prepare_inference_request, send_upstream_request_with_retry,
    upstream_body_stream,
};

pub(crate) fn routes(ctx: ApiCtx) -> ApiRoute {
    api_chat_guided(ctx.state.clone(), ctx.config.clone())
        .map(box_reply)
        .boxed()
}

/// Strip inline thinking/reasoning blocks from a content string.
/// Used by the guided chat handler to ensure no thinking leaks into guided-gen panels.
pub(crate) fn strip_inline_thinking(input: &str) -> String {
    let mut out = input.to_string();

    let tags = ["ANTTHINKING", "THINKING", "REASONING", "THOUGHT"];
    for tag in &tags {
        let open_tag = format!("<{}", tag);
        let close_tag = format!("</{}", tag);
        let mut next = String::new();
        let mut buf: &str = &out;

        while let Some(pos) = buf.find(&open_tag) {
            next.push_str(&buf[..pos]);
            if let Some(end) = buf[pos..].find(&close_tag) {
                buf = &buf[pos + end + close_tag.len()..];
            } else {
                break;
            }
        }
        next.push_str(buf);
        out = next;
    }

    {
        let mut open_tag = String::new();
        open_tag.push('<');
        open_tag.push_str("think");
        open_tag.push('>');
        let mut close_tag = String::new();
        close_tag.push('<');
        close_tag.push('/');
        close_tag.push_str("think");
        close_tag.push('>');

        let mut next = String::new();
        let mut buf: &str = &out;

        while let Some(pos) = buf.find(&open_tag) {
            next.push_str(&buf[..pos]);
            if let Some(end) = buf[pos..].find(&close_tag) {
                buf = &buf[pos + end + close_tag.len()..];
            } else {
                break;
            }
        }
        next.push_str(buf);
        out = next;
    }

    out = out.replace("</think>", "").replace("<think>", "");

    let mut result = String::new();
    let mut prev_blank = false;
    for line in out.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if !prev_blank && !result.is_empty() {
                result.push('\n');
            }
            prev_blank = true;
        } else {
            result.push_str(trimmed);
            result.push('\n');
            prev_blank = false;
        }
    }
    result.trim().to_string()
}

/// Returns true if the text looks like a JSON reasoning/suggestion blob
/// rather than a simple human-readable suggestion.
pub(crate) fn is_json_reasoning(text: &str) -> bool {
    if text.len() < 150 {
        return false;
    }
    let lower = text.to_ascii_lowercase();
    let has_structures = lower.contains("{") && lower.contains("}");
    let has_keys = lower.contains("type")
        && (lower.contains("suggestions")
            || lower.contains("effect")
            || lower.contains("detail")
            || lower.contains("description"));
    has_structures && has_keys
}

/// Sanitize content for the guided endpoint:
/// - Strip thinking tags.
/// - If the content is a raw JSON suggestions/director blob, replace it with concise bullets
///   so screenshots and the chat bubble stay clean.
fn sanitize_guided_content(input: &str) -> String {
    let cleaned = strip_inline_thinking(input);

    if let Ok(value) = serde_json::from_str::<serde_json::Value>(&cleaned) {
        let entries = value
            .get("suggestions")
            .and_then(|v| v.as_array())
            .cloned()
            .or_else(|| value.as_array().cloned());

        if let Some(items) = entries {
            let bullets: Vec<String> = items
                .iter()
                .filter_map(|e| {
                    let obj = e.as_object()?;
                    let title = obj.get("title").and_then(|v| v.as_str()).unwrap_or("");
                    let effect = obj.get("effect").and_then(|v| v.as_str()).unwrap_or(
                        obj.get("description")
                            .and_then(|v| v.as_str())
                            .unwrap_or(""),
                    );
                    let t = title.trim();
                    let eff = effect.trim();
                    if t.is_empty() && eff.is_empty() {
                        return None;
                    }
                    let text = if eff.is_empty() || eff == t {
                        t.to_string()
                    } else if eff.len() > 120 {
                        format!("{}: {}", t, eff[..117].trim())
                    } else {
                        format!("{}: {}", t, eff)
                    };
                    Some(text)
                })
                .take(6)
                .collect();

            if !bullets.is_empty() {
                return bullets.join("\n");
            }
        }
    }

    cleaned
}

/// Guided chat handler: same as api_chat but enforces thinking disabled and
/// strips inline thinking blocks from the streamed content.
fn api_chat_guided(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("api" / "chat" / "guided")
        .and(warp::post())
        .and(warp::header::optional::<String>("authorization"))
        .and(warp::body::content_length_limit(2 * 1024 * 1024))
        .and(warp::body::bytes())
        .and_then(move |auth: Option<String>, body: Bytes| {
            let cfg = app_config.clone();
            let state = state.clone();
            async move {
                if !check_api_token(&auth, &cfg) {
                    return Ok(unauthorized_api_token());
                }
                let prepared = prepare_inference_request(&state).await?;
                let client = build_upstream_client()?;
                let mut request_body = body.to_vec();

                if let Ok(mut val) = serde_json::from_slice::<serde_json::Value>(&request_body) {
                    if let Some(obj) = val.as_object_mut() {
                        obj.insert("enable_thinking".into(), serde_json::Value::Bool(false));
                        obj.insert(
                            "thinking_budget_tokens".into(),
                            serde_json::Value::Number(serde_json::Number::from(0u64)),
                        );
                    }
                    request_body =
                        serde_json::to_vec(&val).unwrap_or_else(|_| request_body.clone());
                }
                request_body = prepared.map_chat_body(&request_body)?;
                let url = prepared.url.clone();

                let resp = send_upstream_request_with_retry(|| {
                    prepared.authenticate(
                        client
                            .post(&url)
                            // Like chat, bound stalls but not healthy stream duration.
                            .header("Content-Type", "application/json")
                            .body(request_body.clone()),
                    )
                })
                .await?;
                let permit = prepared.permit;

                let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
                let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);

                tokio::spawn(async move {
                    use futures_util::StreamExt;
                    let _permit = permit;
                    let mut stream = upstream_body_stream(resp);
                    let mut buf = String::new();

                    loop {
                        let chunk = tokio::select! {
                            () = tx.closed() => return,
                            chunk = stream.next() => chunk,
                        };
                        let Some(chunk) = chunk else {
                            break;
                        };
                        match chunk {
                            Ok(bytes) => {
                                if tx.is_closed() {
                                    return;
                                }
                                buf.push_str(&String::from_utf8_lossy(&bytes));

                                while let Some(pos) = buf.find('\n') {
                                    let line = buf[..pos].to_string();
                                    buf = buf[pos + 1..].to_string();

                                    if let Some(data) = line.strip_prefix("data: ")
                                        && !data.trim().is_empty()
                                    {
                                        let cleaned = if let Ok(mut v) =
                                            serde_json::from_str::<serde_json::Value>(data)
                                        {
                                            if let Some(obj) = v.as_object_mut()
                                                && let Some(c) =
                                                    obj.get("content").and_then(|x| x.as_str())
                                            {
                                                let s = sanitize_guided_content(c);
                                                if s != c {
                                                    obj.insert(
                                                        "content".into(),
                                                        serde_json::Value::String(s),
                                                    );
                                                }
                                            }
                                            serde_json::to_string(&v)
                                                .unwrap_or_else(|_| data.to_string())
                                        } else {
                                            data.to_string()
                                        };
                                        let _ = tx.send(Ok::<_, warp::Error>(
                                            warp::sse::Event::default().data(cleaned),
                                        ));
                                    }
                                }
                            }
                            Err(e) => {
                                let _ = tx.send(Ok::<_, warp::Error>(
                                    warp::sse::Event::default().data(format!(
                                        "{{\"error\":\"{}\"}}",
                                        e.to_string().replace('"', "'")
                                    )),
                                ));
                                break;
                            }
                        }
                    }
                });

                Ok::<Box<dyn warp::reply::Reply>, warp::Rejection>(Box::new(warp::sse::reply(
                    stream,
                )))
            }
        })
}
