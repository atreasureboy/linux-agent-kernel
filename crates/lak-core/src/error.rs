//! Agent Kernel 的统一错误类型

use crate::types::ids::{AgentId, TaskId, IntentId};
use crate::types::capability::{Capability, CapabilityRequirement};

/// Agent Kernel 的所有可能错误
#[derive(Debug, thiserror::Error)]
pub enum KernelError {
    #[error("Agent '{0}' not found")]
    AgentNotFound(AgentId),

    #[error("Task '{0}' not found")]
    TaskNotFound(TaskId),

    #[error("Intent '{0}' not found")]
    IntentNotFound(IntentId),

    #[error("Insufficient capability: required {required:?}, have {have:?}")]
    InsufficientCapability {
        required: Vec<CapabilityRequirement>,
        have: Vec<Capability>,
    },

    #[error("Context window overflow: {used}/{max} tokens")]
    ContextOverflow { used: usize, max: usize },

    #[error("Cognitive resource exhausted: {resource}")]
    ResourceExhausted { resource: String },

    #[error("Token budget exceeded: used {used}/{limit}")]
    TokenBudgetExceeded { used: u64, limit: u64 },

    #[error("Agent limit reached: {current}/{max}")]
    AgentLimitReached { current: u32, max: u32 },

    #[error("Scheduler error: {0}")]
    SchedulerError(String),

    #[error("LLM error: {0}")]
    LLMError(String),

    #[error("Tool error: {tool} — {message}")]
    ToolError { tool: String, message: String },

    #[error("Sandbox error: {0}")]
    SandboxError(String),

    #[error("Invalid state transition: from {from:?} to {to:?}")]
    InvalidStateTransition { from: String, to: String },

    #[error("Prompt injection detected: {0}")]
    PromptInjection(String),

    #[error("Capability delegation error: {0}")]
    DelegationError(String),

    #[error("Timeout: operation exceeded {duration_ms}ms")]
    Timeout { duration_ms: u64 },

    #[error("Not implemented: {0}")]
    NotImplemented(String),

    #[error("Internal error: {0}")]
    Internal(#[from] anyhow::Error),
}

impl KernelError {
    /// 是否可重试
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::LLMError(_)
                | Self::ToolError { .. }
                | Self::Timeout { .. }
                | Self::ResourceExhausted { .. }
        )
    }

    /// 获取错误的简短代码（用于 gRPC 状态码映射）
    pub fn error_code(&self) -> &'static str {
        match self {
            Self::AgentNotFound(_) => "AGENT_NOT_FOUND",
            Self::TaskNotFound(_) => "TASK_NOT_FOUND",
            Self::IntentNotFound(_) => "INTENT_NOT_FOUND",
            Self::InsufficientCapability { .. } => "CAPABILITY_DENIED",
            Self::ContextOverflow { .. } => "CONTEXT_OVERFLOW",
            Self::ResourceExhausted { .. } => "RESOURCE_EXHAUSTED",
            Self::TokenBudgetExceeded { .. } => "TOKEN_BUDGET_EXCEEDED",
            Self::AgentLimitReached { .. } => "AGENT_LIMIT_REACHED",
            Self::SchedulerError(_) => "SCHEDULER_ERROR",
            Self::LLMError(_) => "LLM_ERROR",
            Self::ToolError { .. } => "TOOL_ERROR",
            Self::SandboxError(_) => "SANDBOX_ERROR",
            Self::InvalidStateTransition { .. } => "INVALID_STATE",
            Self::PromptInjection(_) => "PROMPT_INJECTION",
            Self::DelegationError(_) => "DELEGATION_ERROR",
            Self::Timeout { .. } => "TIMEOUT",
            Self::NotImplemented(_) => "NOT_IMPLEMENTED",
            Self::Internal(_) => "INTERNAL",
        }
    }
}
