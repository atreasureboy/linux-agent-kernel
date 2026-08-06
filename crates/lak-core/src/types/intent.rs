//! IntentMessage — Agent Kernel 的"IPC"
//!
//! Intent 是 Agent 间通信的基本单元。
//! 与传统消息不同，Intent 可以是模糊的、需要用语义理解的。

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::ids::{AgentId, IntentId};
use super::capability::CapabilityType;

/// 意图消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentMessage {
    pub intent_id: IntentId,
    pub source_agent_id: AgentId,
    pub target: IntentTarget,
    pub intent_type: IntentType,
    pub content: IntentContent,
    pub priority: super::task::CognitivePriority,
    /// 消息生存时间（毫秒），过期丢弃
    pub ttl_ms: u64,
    /// 关联 ID，用于追踪完整的意图链
    pub correlation_id: Option<IntentId>,
    pub created_at: DateTime<Utc>,
}

/// 意图的目标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntentTarget {
    /// 广播给所有 Agent
    Broadcast,
    /// 发送给特定 Agent
    Unicast(AgentId),
    /// 发送给一组 Agent
    Multicast(Vec<AgentId>),
    /// 按能力匹配路由（语义路由）
    ByCapability {
        cap_type: CapabilityType,
        /// 可选：路由时的语义意图描述，用于匹配最合适的 Agent
        semantic_hint: Option<String>,
    },
    /// 发布-订阅模式：匹配订阅此模式的 Agent
    PublishSubscribe {
        pattern: String,
    },
}

/// 意图类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntentType {
    /// "我需要知道 X"
    Query,
    /// "我委托你完成 X"
    Delegate,
    /// "告诉你 X 发生了"
    Inform,
    /// "请求批准做 X"
    RequestApproval,
    /// "这是你要的结果"
    Respond,
    /// "我们应该协商 X"
    Negotiate,
    /// "帮我监视 X"
    Monitor,
}

/// 意图内容
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentContent {
    /// 自然语言表达（Agent 间通信的主要载体）
    pub natural_language: String,
    /// 可选的结构化数据
    pub structured_data: Option<serde_json::Value>,
    /// 相关记忆引用
    pub memory_references: Vec<super::ids::MemoryChunkId>,
}

/// 意图路由的订阅
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentSubscription {
    pub agent_id: AgentId,
    /// 订阅的意图类型
    pub intent_types: Option<Vec<IntentType>>,
    /// 订阅的主题模式
    pub topic_pattern: Option<String>,
    /// 仅接收有特定能力要求的意图
    pub capability_filter: Option<CapabilityType>,
}
