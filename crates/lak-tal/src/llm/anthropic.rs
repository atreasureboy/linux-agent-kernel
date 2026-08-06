//! Anthropic Claude LLM Driver with SSE streaming
//!
//! The stream parser is stateful:
//! - lines split across network chunks are reassembled by [`LineStream`]
//! - `content_block_start`/`content_block_delta`/`content_block_stop` events
//!   for `tool_use` blocks are accumulated and emitted as one complete
//!   `ToolCall` per block
//! - usage + stop reason from `message_delta` are folded into the terminal
//!   `Done` event emitted on `message_stop`

use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};

use super::openai::estimate_tokens;
use super::stream::LineStream;
use super::traits::*;

/// Anthropic API driver with real SSE streaming
#[derive(Debug)]
pub struct AnthropicDriver {
    api_key: String,
    model: String,
    client: reqwest::Client,
    anthropic_version: String,
}

impl AnthropicDriver {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            client: reqwest::Client::new(),
            anthropic_version: "2023-06-01".into(),
        }
    }

    /// Override the Anthropic API version header
    #[allow(dead_code)]
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.anthropic_version = version.into();
        self
    }
}

#[async_trait]
impl LLMDriver for AnthropicDriver {
    fn name(&self) -> &str {
        "anthropic"
    }

    async fn generate_stream(
        &self,
        request: LLMRequest,
    ) -> Result<BoxStream<'static, Result<LLMStreamEvent, LLMError>>, LLMError> {
        // Convert ChatMessages to Anthropic format (system is a top-level param)
        let mut system_prompt = String::new();
        let mut anthropic_messages: Vec<serde_json::Value> = Vec::new();

        for msg in &request.messages {
            match msg.role {
                ChatRole::System => {
                    if !system_prompt.is_empty() {
                        system_prompt.push('\n');
                    }
                    system_prompt.push_str(&msg.content);
                }
                ChatRole::Assistant => {
                    anthropic_messages.push(serde_json::json!({
                        "role": "assistant",
                        "content": msg.content,
                    }));
                }
                // User + Tool results are both "user" role in the Anthropic API
                _ => {
                    anthropic_messages.push(serde_json::json!({
                        "role": "user",
                        "content": msg.content,
                    }));
                }
            }
        }

        let mut body = serde_json::json!({
            "model": self.model,
            "max_tokens": request.max_tokens.unwrap_or(4096),
            "messages": anthropic_messages,
            "stream": true,
        });

        if !system_prompt.is_empty() {
            body["system"] = serde_json::json!(system_prompt);
        }

        if let Some(temp) = request.temperature {
            body["temperature"] = serde_json::json!(temp);
        }

        // Attach tool definitions
        if let Some(tools) = &request.tools {
            let tool_defs: Vec<serde_json::Value> = tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.parameters,
                    })
                })
                .collect();
            body["tools"] = serde_json::json!(tool_defs);
        }

        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.anthropic_version)
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
            return Err(LLMError::APIError(status, err_body));
        }

        let mut parser = AnthropicSseParser::default();
        let stream = LineStream::new(response.bytes_stream())
            .map(|line| line.map_err(|e| LLMError::NetworkError(e.to_string())))
            .flat_map(move |line| futures::stream::iter(parser.feed_line_result(line)));

        Ok(Box::pin(stream))
    }

    async fn count_tokens(&self, text: &str) -> Result<usize, LLMError> {
        Ok(estimate_tokens(text))
    }

    async fn health_check(&self) -> Result<bool, LLMError> {
        // Anthropic has no unauthenticated health endpoint; key validity is
        // checked implicitly by the first real request.
        Ok(!self.api_key.is_empty())
    }

    fn cost_per_1k_tokens(&self, is_input: bool) -> f64 {
        match (self.model.as_str(), is_input) {
            ("claude-sonnet-5", true) => 0.003,
            ("claude-sonnet-5", false) => 0.015,
            ("claude-haiku-4-5", true) => 0.0008,
            ("claude-haiku-4-5", false) => 0.004,
            ("claude-opus-5", true) => 0.015,
            ("claude-opus-5", false) => 0.075,
            _ => 0.0,
        }
    }
}

/// Stateful Anthropic SSE parser.
///
/// Tracks the currently open content block so that `input_json_delta`
/// fragments can be attached to the correct `tool_use` block, and emits a
/// single complete `ToolCall` when the block closes.
#[derive(Default)]
pub(crate) struct AnthropicSseParser {
    /// Currently open tool_use block: (id, name, accumulated raw arguments)
    current_tool: Option<(String, String, String)>,
    tokens_used: u64,
    stop_reason: Option<String>,
    done_emitted: bool,
}

impl AnthropicSseParser {
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

        let parsed: serde_json::Value = match serde_json::from_str(payload) {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };

        let mut events = Vec::new();

        if let Some(error) = parsed.get("error") {
            events.push(Err(LLMError::APIError(
                400,
                error
                    .get("message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
            )));
            return events;
        }

        let event_type = parsed.get("type").and_then(|t| t.as_str()).unwrap_or("");

        match event_type {
            "content_block_start" => {
                if let Some(block) = parsed.get("content_block") {
                    if block.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
                        // Close any dangling block before opening a new one
                        events.extend(self.close_current_tool());
                        let id = block
                            .get("id")
                            .and_then(|i| i.as_str())
                            .unwrap_or_default()
                            .to_string();
                        let name = block
                            .get("name")
                            .and_then(|n| n.as_str())
                            .unwrap_or_default()
                            .to_string();
                        self.current_tool = Some((id, name, String::new()));
                    }
                }
            }
            "content_block_delta" => {
                if let Some(delta) = parsed.get("delta") {
                    let delta_type = delta.get("type").and_then(|t| t.as_str()).unwrap_or("");
                    match delta_type {
                        "text_delta" => {
                            if let Some(text) = delta.get("text").and_then(|t| t.as_str()) {
                                events.push(Ok(LLMStreamEvent::Token(text.to_string())));
                            }
                        }
                        "input_json_delta" => {
                            if let (Some(tool), Some(partial)) = (
                                self.current_tool.as_mut(),
                                delta.get("partial_json").and_then(|p| p.as_str()),
                            ) {
                                tool.2.push_str(partial);
                            }
                        }
                        "thinking_delta" => {
                            if let Some(thinking) = delta.get("thinking").and_then(|t| t.as_str()) {
                                events.push(Ok(LLMStreamEvent::Thinking(thinking.to_string())));
                            }
                        }
                        _ => {}
                    }
                }
            }
            "content_block_stop" => {
                events.extend(self.close_current_tool());
            }
            "message_delta" => {
                if let Some(stop) = parsed
                    .get("delta")
                    .and_then(|d| d.get("stop_reason"))
                    .and_then(|s| s.as_str())
                {
                    self.stop_reason = Some(stop.to_string());
                }
                if let Some(tokens) = parsed
                    .get("usage")
                    .and_then(|u| u.get("output_tokens"))
                    .and_then(|t| t.as_u64())
                {
                    self.tokens_used = tokens;
                }
            }
            "message_stop" => {
                events.extend(self.emit_done());
            }
            "error" => {
                events.push(Err(LLMError::APIError(
                    400,
                    parsed
                        .get("error")
                        .and_then(|e| e.get("message"))
                        .and_then(|m| m.as_str())
                        .unwrap_or("unknown")
                        .to_string(),
                )));
            }
            _ => {}
        }

        events
    }

    /// Close the open tool_use block (if any), emitting the complete ToolCall.
    fn close_current_tool(&mut self) -> Vec<Result<LLMStreamEvent, LLMError>> {
        match self.current_tool.take() {
            Some((id, name, args)) => {
                let arguments: serde_json::Value = serde_json::from_str(&args)
                    .unwrap_or_else(|_| serde_json::Value::Object(Default::default()));
                vec![Ok(LLMStreamEvent::ToolCall(ToolCallRequest {
                    id,
                    name,
                    arguments,
                }))]
            }
            None => Vec::new(),
        }
    }

    fn emit_done(&mut self) -> Vec<Result<LLMStreamEvent, LLMError>> {
        if self.done_emitted {
            return Vec::new();
        }
        self.done_emitted = true;

        let mut events = self.close_current_tool();
        events.push(Ok(LLMStreamEvent::Done(LLMResponse {
            content: String::new(),
            tool_calls: vec![],
            tokens_used: self.tokens_used,
            finish_reason: self.stop_reason.take().unwrap_or_else(|| "stop".into()),
        })));
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed_lines(parser: &mut AnthropicSseParser, lines: &[&str]) -> Vec<LLMStreamEvent> {
        lines
            .iter()
            .flat_map(|l| parser.feed_line(l))
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    }

    #[test]
    fn test_text_stream_and_done() {
        let mut parser = AnthropicSseParser::default();
        let events = feed_lines(
            &mut parser,
            &[
                r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
                r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}"#,
                r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" there"}}"#,
                r#"data: {"type":"content_block_stop","index":0}"#,
                r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}"#,
                r#"data: {"type":"message_stop"}"#,
            ],
        );

        let n_tokens = events
            .iter()
            .filter(|e| matches!(e, LLMStreamEvent::Token(_)))
            .count();
        assert_eq!(n_tokens, 2);

        let done = events.iter().find_map(|e| match e {
            LLMStreamEvent::Done(r) => Some(r),
            _ => None,
        });
        assert!(done.is_some());
        let done = done.unwrap();
        assert_eq!(done.tokens_used, 7);
        assert_eq!(done.finish_reason, "end_turn");
    }

    #[test]
    fn test_tool_use_block_is_assembled() {
        let mut parser = AnthropicSseParser::default();
        let events = feed_lines(
            &mut parser,
            &[
                r#"data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01","name":"get_weather"}}"#,
                // partial JSON split across deltas
                r#"data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"ci"}}"#,
                r#"data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"ty\": \"Berlin\"}"}}"#,
                r#"data: {"type":"content_block_stop","index":1}"#,
                r#"data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":12}}"#,
                r#"data: {"type":"message_stop"}"#,
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
        assert_eq!(tool_calls[0].id, "toolu_01");
        assert_eq!(tool_calls[0].name, "get_weather");
        assert_eq!(tool_calls[0].arguments["city"], "Berlin");
    }
}
