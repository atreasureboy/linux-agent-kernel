//! CognitiveTask — Agent Kernel 的"线程"
//!
//! 认知任务是调度的基本单元，等同于传统 OS 中的线程。
//! 但调度的是"思考"而非"计算"。

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::ids::{AgentId, MemoryChunkId, TaskId};

/// 认知任务：Agent 需要完成的一项认知工作
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveTask {
    pub task_id: TaskId,
    pub agent_id: AgentId,
    pub task_type: TaskType,
    pub priority: CognitivePriority,
    pub state: TaskState,
    pub content: TaskContent,
    pub deadline: Option<DateTime<Utc>>,
    pub dependencies: Vec<TaskId>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
    pub stats: TaskStats,
}

/// 任务类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskType {
    /// 需要 LLM 推理（最昂贵的操作）
    Reasoning,
    /// 执行工具
    ToolExecution,
    /// 记忆检索
    MemoryRetrieval,
    /// 意图处理
    IntentProcessing,
    /// 空闲反思（仅在资源空闲时运行）
    IdleReflection,
    /// 系统管理任务
    SystemTask,
}

/// 认知优先级
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CognitivePriority {
    /// 时间敏感度 (0-100)
    pub urgency: u8,
    /// 任务重要性 (0-100)
    pub importance: u8,
    /// 与当前上下文的关联度 (0-100)
    pub context_affinity: u8,
}

impl CognitivePriority {
    /// 计算优先级分数 (0.0 - 100.0)
    pub fn score(&self) -> f64 {
        (f64::from(self.urgency) * 0.4)
            + (f64::from(self.importance) * 0.4)
            + (f64::from(self.context_affinity) * 0.2)
    }

    pub fn low() -> Self {
        Self {
            urgency: 10,
            importance: 20,
            context_affinity: 10,
        }
    }

    pub fn normal() -> Self {
        Self {
            urgency: 40,
            importance: 50,
            context_affinity: 50,
        }
    }

    pub fn high() -> Self {
        Self {
            urgency: 80,
            importance: 80,
            context_affinity: 70,
        }
    }

    pub fn critical() -> Self {
        Self {
            urgency: 100,
            importance: 100,
            context_affinity: 90,
        }
    }
}

/// 任务包含的内容
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskContent {
    /// 自然语言描述
    pub natural_language: String,
    /// 可选的结构化约束
    pub structured_schema: Option<serde_json::Value>,
    /// 关联的记忆 ID（上下文锚点）
    pub memory_references: Vec<MemoryChunkId>,
}

/// 任务状态机
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskState {
    /// 已创建，等待调度
    Pending,
    /// 在 Ready Queue 中
    Ready,
    /// 正在执行
    Running,
    /// 被阻塞（附带原因）
    Blocked(String),
    /// 等待 LLM 响应
    AwaitingLLM,
    /// 等待工具完成
    AwaitingTool,
    /// 等待意图响应
    AwaitingIntent,
    /// 被挂起
    Suspended,
    /// 执行完成
    Completed,
    /// 执行失败
    Failed(TaskError),
    /// 被取消
    Cancelled,
}

/// 任务失败的错误信息
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskError {
    pub code: String,
    pub message: String,
    pub retryable: bool,
}

/// 任务执行统计
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TaskStats {
    pub tokens_consumed: u64,
    pub tool_calls_made: u32,
    pub reasoning_steps: u32,
    pub memory_retrievals: u32,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub llm_calls: u32,
    pub total_wall_time_ms: u64,
}
