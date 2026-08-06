//! Ollama Local LLM Driver

use async_trait::async_trait;
use futures::stream::BoxStream;

use super::traits::*;

/// Ollama 本地模型驱动
#[derive(Debug)]
#[allow(dead_code)] // Fields used when streaming is implemented
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
        _request: LLMRequest,
    ) -> Result<BoxStream<'static, Result<LLMStreamEvent, LLMError>>, LLMError> {
        Err(LLMError::UnsupportedModel(
            "Ollama driver streaming not yet implemented".into(),
        ))
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
        0.0 // 本地模型无 API 成本
    }
}
