//! Agent 规格、状态和统计

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::capability::Capability;
use super::ids::AgentId;

/// 创建 Agent 时所需的规格
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSpec {
    /// Agent 名称（人类可读）
    pub name: String,
    /// 系统提示词（Agent 的"宪法"）
    pub system_prompt: String,
    /// 首选模型
    pub model: String,
    /// 上下文窗口大小（最大 token 数）
    pub max_context_tokens: usize,
    /// 初始能力
    pub initial_capabilities: Vec<Capability>,
    /// 记忆配额（字节）
    pub memory_quota_bytes: u64,
    /// 高级配置
    pub config: AgentConfig,
}

impl Default for AgentSpec {
    fn default() -> Self {
        Self {
            name: "UnnamedAgent".into(),
            system_prompt: "You are a helpful assistant.".into(),
            model: "claude-sonnet-5".into(),
            max_context_tokens: 32768,
            initial_capabilities: vec![],
            memory_quota_bytes: 1024 * 1024 * 1024, // 1 GB
            config: AgentConfig::default(),
        }
    }
}

/// Agent 的高级配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// LLM temperature (0.0 - 2.0)
    pub temperature: f32,
    /// 每次任务最大工具调用次数
    pub max_tool_calls_per_task: u32,
    /// 推理超时（秒）
    pub reasoning_timeout_seconds: u64,
    /// 是否允许自我反思
    pub allow_self_reflection: bool,
    /// 写操作是否需要审批
    pub require_approval_for_write: bool,
    /// 自定义元数据
    pub metadata: HashMap<String, String>,
    /// 标签（用于调度亲和性）
    pub tags: Vec<String>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            max_tool_calls_per_task: 20,
            reasoning_timeout_seconds: 120,
            allow_self_reflection: true,
            require_approval_for_write: false,
            metadata: HashMap::new(),
            tags: vec![],
        }
    }
}

/// Agent 的状态机
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentState {
    /// Agent 刚创建，尚未初始化
    Created,
    /// 正在加载 system prompt、记忆、能力
    Initializing,
    /// 正在执行认知任务
    Running,
    /// 无待处理任务，上下文保持活跃
    Idle,
    /// 等待外部事件（工具返回 / 意图响应）
    Blocked,
    /// 被 Supervisor 暂停
    Suspended,
    /// 上下文已卸载到长期记忆，仅保留唤醒钩子
    Sleeping,
    /// Agent 被销毁
    Terminated,
}

impl AgentState {
    /// 是否处于活跃状态（可接收新任务）
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Running | Self::Idle | Self::Blocked)
    }

    /// 是否已终结
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Terminated)
    }
}

/// Agent 的运行时统计
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentStats {
    pub total_tasks_completed: u64,
    pub total_tasks_failed: u64,
    pub total_tokens_consumed: u64,
    pub total_tool_calls: u64,
    pub total_tool_failures: u64,
    /// 认知机会指数 (0.0 - 1.0)
    pub coi: f32,
    /// 幻觉率（被检测到的幻觉 / 总声明）
    pub hallucination_rate: f32,
    /// 推理循环被检测到的次数
    pub reasoning_loop_count: u64,
    /// 平均端到端响应延迟（秒）
    pub avg_response_latency_seconds: f64,
    /// 能力违规次数
    pub capability_violations: u64,
}

/// 包含完整信息的 Agent 表示
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    pub id: AgentId,
    pub name: String,
    pub spec: AgentSpec,
    pub state: AgentState,
    pub stats: AgentStats,
    pub created_at: DateTime<Utc>,
    pub last_active_at: DateTime<Utc>,
    pub terminated_at: Option<DateTime<Utc>>,
}
