//! MemoryChunk — Agent Kernel 的"内存页"
//!
//! 记忆是语义寻址的，而非按物理地址。
//! MemoryChunk 是语义记忆的基本单元。

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::ids::{AgentId, MemoryChunkId};

/// 语义记忆的基本单元
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryChunk {
    pub chunk_id: MemoryChunkId,
    pub agent_id: AgentId,

    /// 记忆内容
    pub content: MemoryContent,

    /// 元数据
    pub metadata: MemoryMetadata,

    /// 与其他记忆的关联
    pub relations: Vec<MemoryRelation>,

    /// 记忆层级
    pub tier: MemoryTier,
}

/// 记忆的实际内容
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContent {
    /// 原始文本
    pub raw_text: String,
    /// 可选的结构化数据
    pub structured_data: Option<serde_json::Value>,
    /// 语义向量（用于向量检索；存储时填充）
    /// MVP: Optional, 后续版本中向量数据库提供
    pub embedding: Option<Vec<f32>>,
}

/// 记忆的元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetadata {
    pub created_at: DateTime<Utc>,
    pub last_accessed_at: DateTime<Utc>,
    pub access_count: u64,
    /// 重要性评分 (0.0 - 1.0)，影响置换决策
    pub importance_score: f32,
    /// 衰减速率（每时间单位衰减的比例）
    pub decay_rate: f32,
    /// 来源
    pub source: MemorySource,
    /// 事实性标记
    pub factuality: Factuality,
}

/// 记忆来源
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemorySource {
    /// 从用户输入直接获得
    UserInput,
    /// Agent 自己的推理产出
    AgentReasoning,
    /// 从工具输出获得
    ToolOutput,
    /// 从其他 Agent 获得
    OtherAgent(AgentId),
    /// 外部数据源
    ExternalSource,
}

/// 事实性标记
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Factuality {
    /// 经多个独立来源验证，不可被 Agent 自己修改
    Fact,
    /// Agent 的推理结果，可被质疑和修正
    Belief(f32), // 置信度 0.0-1.0
    /// 从不确定来源获得，低信任度
    Hearsay,
}

/// 记忆间的关联
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRelation {
    pub target_chunk_id: MemoryChunkId,
    pub relation_type: MemoryRelationType,
    /// 关联强度 (0.0 - 1.0)
    pub strength: f32,
}

/// 记忆关联类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryRelationType {
    /// A 导致 B
    Causes,
    /// A 与 B 矛盾
    Contradicts,
    /// A 支持 B
    Supports,
    /// A 在时序上先于 B
    Follows,
    /// A 是 B 的例子
    IsExampleOf,
    /// A 是 B 的部分
    IsPartOf,
    /// A 引用了 B
    References,
}

/// 记忆层级（四层记忆模型）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryTier {
    /// 工作记忆 —— 当前上下文窗口中
    Working,
    /// 短期记忆 —— 当前会话相关，时间衰减
    ShortTerm,
    /// 长期记忆 —— 经过整理的重要记忆
    LongTerm,
    /// 归档记忆 —— 压缩的历史记忆
    Archival,
}
