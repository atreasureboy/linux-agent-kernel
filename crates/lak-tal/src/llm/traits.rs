//! Unified LLM Driver trait

use async_trait::async_trait;
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};

/// LLM 请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMRequest {
    pub messages: Vec<ChatMessage>,
    pub tools: Option<Vec<ToolDefinition>>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
}

/// 聊天消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

impl ChatRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

/// 工具定义（用于传递给 LLM）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// LLM 流式响应事件
#[derive(Debug, Clone)]
pub enum LLMStreamEvent {
    /// 生成的 token
    Token(String),
    /// 工具调用请求
    ToolCall(ToolCallRequest),
    /// 模型的思考过程（部分模型支持）
    Thinking(String),
    /// 生成完成
    Done(LLMResponse),
    /// 错误
    Error(LLMError),
}

/// LLM 的最终响应
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LLMResponse {
    pub content: String,
    pub tool_calls: Vec<ToolCallRequest>,
    pub tokens_used: u64,
    pub finish_reason: String,
}

/// LLM 请求的工具调用
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRequest {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// LLM 驱动可能发生的错误
#[derive(Debug, Clone, thiserror::Error)]
pub enum LLMError {
    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("API error ({0}): {1}")]
    APIError(u16, String),

    #[error("Rate limited")]
    RateLimited,

    #[error("Context length exceeded")]
    ContextExceeded,

    #[error("Content filtered")]
    ContentFiltered,

    #[error("Timeout")]
    Timeout,

    #[error("Stream interrupted")]
    StreamInterrupted,

    #[error("Malformed response: {0}")]
    MalformedResponse(String),

    #[error("Unsupported model: {0}")]
    UnsupportedModel(String),

    #[error("All backends exhausted")]
    AllBackendsExhausted,
}

/// LLM 驱动 trait
#[async_trait]
pub trait LLMDriver: Send + Sync + std::fmt::Debug {
    /// 驱动名称
    fn name(&self) -> &str;

    /// 流式生成
    async fn generate_stream(
        &self,
        request: LLMRequest,
    ) -> Result<BoxStream<'static, Result<LLMStreamEvent, LLMError>>, LLMError>;

    /// Token 计数（估计）
    async fn count_tokens(&self, text: &str) -> Result<usize, LLMError>;

    /// 健康检查
    async fn health_check(&self) -> Result<bool, LLMError>;

    /// 成本估算（每 1000 token 的美元成本）
    fn cost_per_1k_tokens(&self, is_input: bool) -> f64;
}
