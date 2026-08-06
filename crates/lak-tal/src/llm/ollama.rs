//! Ollama Local LLM Driver with streaming support

use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};

use super::stream::LineStream;
use super::traits::*;

/// Ollama local model driver with real streaming
#[derive(Debug)]
pub struct OllamaDriver {
    base_url: String,
    model: String,
    client: reqwest::Client,
}

impl OllamaDriver {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            base_url: "http://localhost:11434".into(),
            model: model.into(),
            client: reqwest::Client::new(),
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
}

#[async_trait]
impl LLMDriver for OllamaDriver {
    fn name(&self) -> &str {
        "ollama"
    }

    async fn generate_stream(
        &self,
        request: LLMRequest,
    ) -> Result<BoxStream<'static, Result<LLMStreamEvent, LLMError>>, LLMError> {
        // Convert messages to Ollama format
        let messages: Vec<serde_json::Value> = request
            .messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    ChatRole::System => "system",
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                    ChatRole::Tool => "tool",
                };
                serde_json::json!({
                    "role": role,
                    "content": m.content,
                })
            })
            .collect();

        let mut body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "stream": true,
        });

        if let Some(max_tokens) = request.max_tokens {
            body["options"] = serde_json::json!({
                "num_predict": max_tokens,
            });
        }

        if let Some(temp) = request.temperature {
            if let Some(opts) = body.get_mut("options") {
                opts["temperature"] = serde_json::json!(temp);
            } else {
                body["options"] = serde_json::json!({ "temperature": temp });
            }
        }

        // Attach tools if supported (Ollama 0.3+)
        if let Some(tools) = &request.tools {
            let tool_defs: Vec<serde_json::Value> = tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                    })
                })
                .collect();
            body["tools"] = serde_json::json!(tool_defs);
        }

        let response = self
            .client
            .post(format!("{}/api/chat", self.base_url))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| LLMError::NetworkError(e.to_string()))?;

        let status = response.status().as_u16();
        if !response.status().is_success() {
            let err_body = response.text().await.unwrap_or_default();
            return Err(LLMError::APIError(status, err_body));
        }

        // Ollama returns newline-delimited JSON (NDJSON), not SSE.
        // LineStream reassembles lines split across network chunks.
        let stream = LineStream::new(response.bytes_stream())
            .map(|line| line.map_err(|e| LLMError::NetworkError(e.to_string())))
            .flat_map(|line| {
                let events: Vec<Result<LLMStreamEvent, LLMError>> = match line {
                    Err(e) => vec![Err(e)],
                    Ok(l) => parse_ollama_line(&l),
                };
                futures::stream::iter(events)
            });

        Ok(Box::pin(stream))
    }

    async fn count_tokens(&self, text: &str) -> Result<usize, LLMError> {
        Ok(text.len() / 4)
    }

    async fn health_check(&self) -> Result<bool, LLMError> {
        let resp = self
            .client
            .get(&self.base_url)
            .send()
            .await
            .map_err(|e| LLMError::NetworkError(e.to_string()))?;
        Ok(resp.status().is_success())
    }

    fn cost_per_1k_tokens(&self, _is_input: bool) -> f64 {
        0.0 // Local models have no API cost
    }
}

/// Parse one Ollama NDJSON line (one JSON object per line).
fn parse_ollama_line(line: &str) -> Vec<Result<LLMStreamEvent, LLMError>> {
    let line = line.trim();
    if line.is_empty() {
        return Vec::new();
    }

    let parsed: serde_json::Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };

    let mut events = Vec::new();

    // Check for error
    if let Some(error) = parsed.get("error") {
        events.push(Err(LLMError::APIError(
            500,
            error.as_str().unwrap_or("unknown").to_string(),
        )));
        return events;
    }

    // Content token — message.content is the next piece of text
    if let Some(message) = parsed.get("message") {
        if let Some(content) = message.get("content").and_then(|c| c.as_str()) {
            if !content.is_empty() {
                events.push(Ok(LLMStreamEvent::Token(content.to_string())));
            }
        }

        // Tool calls arrive complete (not fragmented) in Ollama
        if let Some(tool_calls) = message.get("tool_calls").and_then(|t| t.as_array()) {
            for (i, tc) in tool_calls.iter().enumerate() {
                let func = &tc["function"];
                events.push(Ok(LLMStreamEvent::ToolCall(ToolCallRequest {
                    id: format!("ollama_call_{i}"),
                    name: func["name"].as_str().unwrap_or("").to_string(),
                    arguments: func["arguments"].clone(),
                })));
            }
        }
    }

    // Completion signal
    if parsed
        .get("done")
        .and_then(|d| d.as_bool())
        .unwrap_or(false)
    {
        let tokens = parsed
            .get("eval_count")
            .and_then(|t| t.as_u64())
            .unwrap_or(0);

        events.push(Ok(LLMStreamEvent::Done(LLMResponse {
            content: String::new(),
            tool_calls: vec![],
            tokens_used: tokens,
            finish_reason: "stop".into(),
        })));
    }

    events
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_token_line() {
        let events =
            parse_ollama_line(r#"{"message":{"role":"assistant","content":"Hi"},"done":false}"#);
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], Ok(LLMStreamEvent::Token(_))));
    }

    #[test]
    fn test_parse_done_line() {
        let events = parse_ollama_line(
            r#"{"message":{"role":"assistant","content":""},"done":true,"eval_count":99}"#,
        );
        let done = events.iter().find_map(|e| match e {
            Ok(LLMStreamEvent::Done(r)) => Some(r),
            _ => None,
        });
        assert_eq!(done.unwrap().tokens_used, 99);
    }

    #[test]
    fn test_parse_tool_call_line() {
        let events = parse_ollama_line(
            r#"{"message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"search","arguments":{"q":"rust"}}}]},"done":false}"#,
        );
        let tc = events.iter().find_map(|e| match e {
            Ok(LLMStreamEvent::ToolCall(tc)) => Some(tc),
            _ => None,
        });
        assert!(tc.is_some());
        let tc = tc.unwrap();
        assert_eq!(tc.name, "search");
        assert_eq!(tc.arguments["q"], "rust");
    }

    #[test]
    fn test_parse_error_line() {
        let events = parse_ollama_line(r#"{"error":"model not found"}"#);
        assert!(matches!(
            events.first(),
            Some(Err(LLMError::APIError(500, _)))
        ));
    }
}
