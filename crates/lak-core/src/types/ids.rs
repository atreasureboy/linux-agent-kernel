//! 强类型 ID 系统 — Agent Kernel 的基础词汇
//!
//! 每个 ID 类型都是 newtype wrapper over UUID，防止类型混淆。
//! 例如：不能把 AgentId 当作 TaskId 传参。

use serde::{Deserialize, Serialize};
use uuid::Uuid;

macro_rules! define_id {
    ($name:ident, $doc:expr) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub Uuid);

        impl $name {
            /// 生成一个新的唯一 ID
            pub fn new() -> Self {
                Self(Uuid::new_v4())
            }

            /// 从 UUID 直接构造（用于从存储中恢复）
            pub fn from_uuid(uuid: Uuid) -> Self {
                Self(uuid)
            }

            /// 获取内部 UUID
            pub fn as_uuid(&self) -> &Uuid {
                &self.0
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

define_id!(AgentId, "智能体的唯一标识符——等同于操作系统的 PID");
define_id!(TaskId, "认知任务的唯一标识符——等同于线程的 TID");
define_id!(IntentId, "意图消息的唯一标识符——等同于网络包序列号");
define_id!(MemoryChunkId, "记忆片段的唯一标识符——等同于内存页帧号");
define_id!(CapabilityCertId, "能力证书的唯一标识符");
define_id!(CognodeId, "认知文件节点的唯一标识符");
define_id!(ReasoningChainId, "推理链的唯一标识符");

// ── 系统保留的 Agent ID ──

impl AgentId {
    /// Supervisor Agent 的固定 ID（系统内置的第一个 Agent）
    pub const SUPERVISOR: Self = Self(Uuid::from_u128(1));
    /// 系统自身的 Agent ID（由内核使用）
    pub const SYSTEM: Self = Self(Uuid::from_u128(2));
}
