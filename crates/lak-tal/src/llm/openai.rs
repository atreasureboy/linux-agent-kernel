//! OpenAI LLM Driver

use async_trait::async_trait;
use futures::stream::BoxStream;

use super::traits::*;

/// OpenAI API 驱动
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

    /// 用于 Azure OpenAI 或其他兼容 API
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
        _request: LLMRequest,
    ) -> Result<BoxStream<'static, Result<LLMStreamEvent, LLMError>>, LLMError> {
        // MVP: placeholder — full SSE parsing to be implemented
        Err(LLMError::UnsupportedModel(
            "OpenAI driver streaming not yet implemented".into(),
        ))
    }

    async fn count_tokens(&self, text: &str) -> Result<usize, LLMError> {
        // Rough estimate: ~4 chars per token
        Ok(text.len() / 4)
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
            ("gpt-4o-mini", true) => 0.00015,
            ("gpt-4o-mini", false) => 0.0006,
            _ => 0.0,
        }
    }
}
