//! AgentKernel — 核心 trait
//!
//! 这是 Agent Kernel 的主接口，所有操作都通过此 trait 进行。

use async_trait::async_trait;

use crate::error::KernelError;
use crate::types::agent::{Agent, AgentSpec};
use crate::types::capability::{
    Capability, CapabilityCertificate, CapabilityRequirement,
};
use crate::types::ids::*;
use crate::types::intent::{IntentMessage, IntentSubscription};
use crate::types::memory::MemoryChunk;
use crate::types::task::CognitiveTask;

/// Agent Kernel 主接口
///
/// 所有 Agent 操作都通过此 trait 进行，包括：
/// - Agent 生命周期管理
/// - 认知任务调度
/// - 意图路由
/// - 记忆管理
/// - 能力管理
#[async_trait]
pub trait AgentKernel: Send + Sync {
    // ── Agent 生命周期 ──

    /// 创建新的 Agent
    async fn create_agent(&self, spec: AgentSpec) -> Result<AgentId, KernelError>;

    /// 销毁 Agent
    async fn destroy_agent(&self, agent_id: AgentId) -> Result<(), KernelError>;

    /// 获取 Agent 信息
    async fn get_agent(&self, agent_id: AgentId) -> Result<Agent, KernelError>;

    /// 列出所有 Agent
    async fn list_agents(&self) -> Result<Vec<Agent>, KernelError>;

    /// 暂停 Agent
    async fn pause_agent(&self, agent_id: AgentId) -> Result<(), KernelError>;

    /// 恢复 Agent
    async fn resume_agent(&self, agent_id: AgentId) -> Result<(), KernelError>;

    // ── 认知任务 ──

    /// 提交认知任务
    async fn submit_task(&self, task: CognitiveTask) -> Result<TaskId, KernelError>;

    /// 取消任务
    async fn cancel_task(&self, task_id: TaskId) -> Result<(), KernelError>;

    /// 查询任务状态
    async fn get_task(&self, task_id: TaskId) -> Result<CognitiveTask, KernelError>;

    // ── 意图路由 ──

    /// 发送意图
    async fn send_intent(
        &self,
        intent: IntentMessage,
    ) -> Result<IntentId, KernelError>;

    /// 等待意图（阻塞直到收到匹配的意图）
    async fn await_intent(
        &self,
        agent_id: AgentId,
        subscription: IntentSubscription,
    ) -> Result<IntentMessage, KernelError>;

    // ── 语义记忆 ──

    /// 存储记忆
    async fn store_memory(
        &self,
        agent_id: AgentId,
        chunk: MemoryChunk,
    ) -> Result<(), KernelError>;

    /// 查询记忆
    async fn query_memory(
        &self,
        agent_id: AgentId,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<MemoryChunk>, KernelError>;

    /// 遗忘记忆
    async fn forget_memory(
        &self,
        agent_id: AgentId,
        chunk_id: MemoryChunkId,
    ) -> Result<(), KernelError>;

    // ── 能力管理 ──

    /// 授予能力
    async fn grant_capability(
        &self,
        from_agent: AgentId,
        to_agent: AgentId,
        capability: Capability,
    ) -> Result<CapabilityCertId, KernelError>;

    /// 撤销能力
    async fn revoke_capability(
        &self,
        cert_id: CapabilityCertId,
    ) -> Result<(), KernelError>;

    /// 委派能力（带衰减）
    async fn delegate_capability(
        &self,
        from_agent: AgentId,
        to_agent: AgentId,
        requirement: CapabilityRequirement,
        new_scope: Option<String>,
        new_permissions: Option<u32>,
    ) -> Result<CapabilityCertId, KernelError>;

    /// 获取 Agent 的能力证书
    async fn get_capabilities(
        &self,
        agent_id: AgentId,
    ) -> Result<CapabilityCertificate, KernelError>;

    // ── 系统 ──

    /// 获取系统状态
    async fn get_system_status(&self) -> Result<SystemStatus, KernelError>;

    /// 关闭 Agent Kernel
    async fn shutdown(&self) -> Result<(), KernelError>;
}

/// 系统状态快照
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SystemStatus {
    pub active_agents: u32,
    pub max_agents: u32,
    pub pending_tasks: u32,
    pub completed_tasks_total: u64,
    pub total_tokens_consumed: u64,
    pub average_coi: f32,
    pub scheduler_load: f32,
    pub uptime_seconds: u64,
}
