//! OpenAI LLM Driver with SSE streaming support
//!
//! The stream parser is stateful:
//! - lines split across network chunks are reassembled by [`LineStream`]
//! - tool-call argument fragments (streamed across many deltas) are
//!   accumulated per call id and emitted as one complete `ToolCall`
//! - a single terminal `Done` event is emitted carrying usage stats

use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};

use super::stream::LineStream;
use super::traits::*;

/// OpenAI API driver with real SSE streaming
#[derive(Debug)]
pub struct OpenAIDriver {
    api_key: String,
    base_url: String,
    model: String,
    client: reqwest::Client,
}

impl OpenAIDriver {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1".into(),
            model: model.into(),
            client: reqwest::Client::new(),
        }
    }

    /// For Azure OpenAI or compatible APIs
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
}

#[async_trait]
impl LLMDriver for OpenAIDriver {
    fn name(&self) -> &str {
        "openai"
    }

    async fn generate_stream(
        &self,
        request: LLMRequest,
    ) -> Result<BoxStream<'static, Result<LLMStreamEvent, LLMError>>, LLMError> {
        let messages: Vec<serde_json::Value> = request
            .messages
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": m.role.as_str(),
                    "content": m.content,
                })
            })
            .collect();

        let mut body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "stream": true,
            "stream_options": { "include_usage": true },
            "max_tokens": request.max_tokens.unwrap_or(4096),
            "temperature": request.temperature.unwrap_or(0.7),
        });

        // Attach tool definitions if provided
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
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| LLMError::NetworkError(e.to_string()))?;

        let status = response.status().as_u16();
        if !response.status().is_success() {
            let err_body = response.text().await.unwrap_or_default();
            if status == 429 {
                return Err(LLMError::RateLimited);
            }
            if status == 400 && err_body.contains("context_length") {
                return Err(LLMError::ContextExceeded);
            }
            return Err(LLMError::APIError(status, err_body));
        }

        let mut parser = OpenAISseParser::default();
        let stream = LineStream::new(response.bytes_stream())
            .map(|line| line.map_err(|e| LLMError::NetworkError(e.to_string())))
            .flat_map(move |line| futures::stream::iter(parser.feed_line_result(line)));

        Ok(Box::pin(stream))
    }

    async fn count_tokens(&self, text: &str) -> Result<usize, LLMError> {
        Ok(estimate_tokens(text))
    }

    async fn health_check(&self) -> Result<bool, LLMError> {
        let resp = self
            .client
            .get(format!("{}/models", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
            .map_err(|e| LLMError::NetworkError(e.to_string()))?;
        Ok(resp.status().is_success())
    }

    fn cost_per_1k_tokens(&self, is_input: bool) -> f64 {
        match (self.model.as_str(), is_input) {
            ("gpt-4o", true) => 0.0025,
            ("gpt-4o", false) => 0.01,
            ("gpt-4o-mini", true) => 0.000_15,
            ("gpt-4o-mini", false) => 0.0006,
            ("gpt-4-turbo", true) => 0.01,
            ("gpt-4-turbo", false) => 0.03,
            ("gpt-3.5-turbo", true) => 0.0005,
            ("gpt-3.5-turbo", false) => 0.0015,
            _ => 0.0,
        }
    }
}

/// Rough token estimate: ~4 chars per token for Latin text, dense CJK costs more.
pub(crate) fn estimate_tokens(text: &str) -> usize {
    let char_count = text.chars().count();
    let cjk_count = text
        .chars()
        .filter(|&c| ('\u{4E00}'..='\u{9FFF}').contains(&c))
        .count();
    let non_cjk = char_count - cjk_count;
    (non_cjk / 4) + (cjk_count * 2 / 3)
}

/// Stateful OpenAI SSE parser.
///
/// Accumulates streamed tool-call fragments keyed by call id (falling back
/// to `call_{index}`), and emits exactly one terminal `Done` event — either
/// when the `[DONE]` marker arrives or when the underlying stream ends.
/// One accumulating tool call. Keyed by the stream `index`, which is the
/// only field present in *every* delta (the id only appears in the first).
#[derive(Default)]
struct ToolAccumulator {
    id: String,
    name: String,
    args: String,
}

#[derive(Default)]
pub(crate) struct OpenAISseParser {
    /// Accumulating tool calls, keyed by stream index; order preserved.
    tool_order: Vec<u64>,
    /// stream index → accumulator
    tool_parts: std::collections::HashMap<u64, ToolAccumulator>,
    tokens_used: u64,
    finish_reason: String,
    done_emitted: bool,
}

impl OpenAISseParser {
    /// Feed one line (or a stream error) and get zero or more events out.
    fn feed_line_result(
        &mut self,
        line: Result<String, LLMError>,
    ) -> Vec<Result<LLMStreamEvent, LLMError>> {
        match line {
            Err(e) => vec![Err(e)],
            Ok(l) => self.feed_line(&l),
        }
    }

    fn feed_line(&mut self, line: &str) -> Vec<Result<LLMStreamEvent, LLMError>> {
        let line = line.trim();
        if line.is_empty() || !line.starts_with("data:") {
            return Vec::new();
        }
        let payload = line.trim_start_matches("data:").trim();

        if payload == "[DONE]" {
            return self.finish("stop");
        }

        let parsed: serde_json::Value = match serde_json::from_str(payload) {
            Ok(v) => v,
            Err(_) => return Vec::new(), // skip malformed lines gracefully
        };

        let mut events = Vec::new();

        // Top-level error object (non-streaming error body)
        if let Some(error) = parsed.get("error") {
            let code = error
                .get("code")
                .and_then(|c| c.as_str())
                .and_then(|s| s.parse().ok())
                .unwrap_or(500);
            let message = error
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown")
                .to_string();
            events.push(Err(LLMError::APIError(code, message)));
            return events;
        }

        // Usage chunk (final chunk when stream_options.include_usage=true)
        if let Some(usage) = parsed.get("usage") {
            self.tokens_used = usage
                .get("total_tokens")
                .and_then(|t| t.as_u64())
                .unwrap_or(0);
        }

        if let Some(choices) = parsed.get("choices").and_then(|c| c.as_array()) {
            for choice in choices {
                if let Some(reason) = choice.get("finish_reason").and_then(|f| f.as_str()) {
                    if !reason.is_empty() {
                        self.finish_reason = reason.to_string();
                    }
                }

                let Some(delta) = choice.get("delta") else {
                    continue;
                };

                if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                    if !content.is_empty() {
                        events.push(Ok(LLMStreamEvent::Token(content.to_string())));
                    }
                }

                if let Some(reasoning) = delta.get("reasoning_content").and_then(|r| r.as_str()) {
                    if !reasoning.is_empty() {
                        events.push(Ok(LLMStreamEvent::Thinking(reasoning.to_string())));
                    }
                }

                if let Some(tool_calls) = delta.get("tool_calls").and_then(|t| t.as_array()) {
                    for tc in tool_calls {
                        self.accumulate_tool_delta(tc);
                    }
                }
            }
        }

        events
    }

    /// Merge one streamed tool-call delta into the accumulator.
    fn accumulate_tool_delta(&mut self, tc: &serde_json::Value) {
        let index = tc.get("index").and_then(|i| i.as_u64()).unwrap_or(0);

        let entry = self.tool_parts.entry(index).or_default();
        if !self.tool_order.contains(&index) {
            self.tool_order.push(index);
        }

        // The real call id arrives only in the first delta; keep it.
        if entry.id.is_empty() {
            if let Some(id) = tc
                .get("id")
                .and_then(|i| i.as_str())
                .filter(|s| !s.is_empty())
            {
                entry.id = id.to_string();
            }
        }

        if let Some(func) = tc.get("function") {
            if let Some(name) = func.get("name").and_then(|n| n.as_str()) {
                entry.name.push_str(name);
            }
            if let Some(args) = func.get("arguments").and_then(|a| a.as_str()) {
                entry.args.push_str(args);
            }
        }
    }

    /// Emit all accumulated tool calls followed by the terminal Done event.
    fn finish(&mut self, default_reason: &str) -> Vec<Result<LLMStreamEvent, LLMError>> {
        let mut events = Vec::new();
        if self.done_emitted {
            return events;
        }
        self.done_emitted = true;

        for index in std::mem::take(&mut self.tool_order) {
            if let Some(acc) = self.tool_parts.remove(&index) {
                let arguments: serde_json::Value =
                    serde_json::from_str(&acc.args).unwrap_or_else(|_| serde_json::json!({}));
                let id = if acc.id.is_empty() {
                    format!("call_{index}")
                } else {
                    acc.id
                };
                events.push(Ok(LLMStreamEvent::ToolCall(ToolCallRequest {
                    id,
                    name: acc.name,
                    arguments,
                })));
            }
        }

        let finish_reason = if self.finish_reason.is_empty() {
            default_reason.to_string()
        } else {
            std::mem::take(&mut self.finish_reason)
        };

        events.push(Ok(LLMStreamEvent::Done(LLMResponse {
            content: String::new(),
            tool_calls: vec![],
            tokens_used: self.tokens_used,
            finish_reason,
        })));
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed_lines(parser: &mut OpenAISseParser, lines: &[&str]) -> Vec<LLMStreamEvent> {
        lines
            .iter()
            .flat_map(|l| parser.feed_line(l))
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    }

    #[test]
    fn test_tokens_and_done() {
        let mut parser = OpenAISseParser::default();
        let events = feed_lines(
            &mut parser,
            &[
                r#"data: {"choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}"#,
                r#"data: {"choices":[{"delta":{"content":"lo"},"finish_reason":null}]}"#,
                r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
                r#"data: {"choices":[],"usage":{"total_tokens":42}}"#,
                "data: [DONE]",
            ],
        );

        let tokens: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                LLMStreamEvent::Token(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        // Collect owned strings separately to avoid borrow issues
        assert_eq!(tokens.len(), 2);

        let done = events.iter().find_map(|e| match e {
            LLMStreamEvent::Done(r) => Some(r),
            _ => None,
        });
        assert!(done.is_some());
        let done = done.unwrap();
        assert_eq!(done.tokens_used, 42);
        assert_eq!(done.finish_reason, "stop");
    }

    #[test]
    fn test_tool_call_fragments_are_merged() {
        let mut parser = OpenAISseParser::default();
        let events = feed_lines(
            &mut parser,
            &[
                // First fragment carries id + name
                r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","function":{"name":"get_weather","arguments":""}}]}}]}"#,
                // Subsequent fragments carry only argument pieces
                r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"ci"}}]}}]}"#,
                r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ty\": \"SF\"}"}}]}}]}"#,
                r#"data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
                "data: [DONE]",
            ],
        );

        let tool_calls: Vec<&ToolCallRequest> = events
            .iter()
            .filter_map(|e| match e {
                LLMStreamEvent::ToolCall(tc) => Some(tc),
                _ => None,
            })
            .collect();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "get_weather");
        assert_eq!(tool_calls[0].id, "call_abc");
        assert_eq!(tool_calls[0].arguments["city"], "SF");
    }

    #[test]
    fn test_two_parallel_tool_calls() {
        let mut parser = OpenAISseParser::default();
        let events = feed_lines(
            &mut parser,
            &[
                r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"a","arguments":""}},{"index":1,"id":"call_2","function":{"name":"b","arguments":""}}]}}]}"#,
                r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"x\":1}"}},{"index":1,"function":{"arguments":"{\"y\":2}"}}]}}]}"#,
                "data: [DONE]",
            ],
        );

        let tool_calls: Vec<&ToolCallRequest> = events
            .iter()
            .filter_map(|e| match e {
                LLMStreamEvent::ToolCall(tc) => Some(tc),
                _ => None,
            })
            .collect();
        assert_eq!(tool_calls.len(), 2);
        assert_eq!(tool_calls[0].arguments["x"], 1);
        assert_eq!(tool_calls[1].arguments["y"], 2);
    }

    #[test]
    fn test_error_chunk() {
        let mut parser = OpenAISseParser::default();
        let events = parser.feed_line(r#"data: {"error":{"code":"429","message":"rate limited"}}"#);
        assert!(matches!(
            events.first(),
            Some(Err(LLMError::APIError(429, _)))
        ));
    }

    #[test]
    fn test_estimate_tokens() {
        assert!(estimate_tokens("hello world") > 0);
        assert!(estimate_tokens("你好世界") > 0);
    }
}
