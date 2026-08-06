//! Anthropic Claude LLM Driver

use async_trait::async_trait;
use futures::stream::BoxStream;

use super::traits::*;

/// Anthropic API 驱动
#[derive(Debug)]
#[allow(dead_code)] // Fields used when streaming is implemented
pub struct AnthropicDriver {
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl AnthropicDriver {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl LLMDriver for AnthropicDriver {
    fn name(&self) -> &str {
        "anthropic"
    }

    async fn generate_stream(
        &self,
        _request: LLMRequest,
    ) -> Result<BoxStream<'static, Result<LLMStreamEvent, LLMError>>, LLMError> {
        Err(LLMError::UnsupportedModel(
            "Anthropic driver streaming not yet implemented".into(),
        ))
    }

    async fn count_tokens(&self, text: &str) -> Result<usize, LLMError> {
        Ok(text.len() / 4)
    }

    async fn health_check(&self) -> Result<bool, LLMError> {
        Ok(true) // Placeholder
    }

    fn cost_per_1k_tokens(&self, is_input: bool) -> f64 {
        match (self.model.as_str(), is_input) {
            ("claude-sonnet-5", true) => 0.003,
            ("claude-sonnet-5", false) => 0.015,
            ("claude-haiku-4-5", true) => 0.0008,
            ("claude-haiku-4-5", false) => 0.004,
            _ => 0.0,
        }
    }
}
