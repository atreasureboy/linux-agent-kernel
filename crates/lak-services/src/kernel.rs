//! KernelService — concrete implementation of the AgentKernel trait
//!
//! This is the operational core of LAK: it wires together memory,
//! reasoning, tool registry, and manages agent/task/intent/capability state.

use std::collections::HashMap;

use async_trait::async_trait;
use chrono::Utc;
use tokio::sync::RwLock;

use lak_core::error::KernelError;
use lak_core::traits::{AgentKernel, SystemStatus};
use lak_core::types::agent::{Agent, AgentSpec, AgentState, AgentStats};
use lak_core::types::capability::{
    Capability, CapabilityCertificate, CapabilityPermission, CapabilityRequirement, CapabilityScope,
};
use lak_core::types::ids::{AgentId, CapabilityCertId, IntentId, MemoryChunkId, TaskId};
use lak_core::types::intent::{IntentMessage, IntentSubscription};
use lak_core::types::memory::MemoryChunk;
use lak_core::types::task::{CognitiveTask, TaskState};

use super::memory::service::MemoryService;
use super::reasoning::service::ReasoningService;
use super::tool_registry::ToolRegistry;

/// Internal state protected by a read-write lock
struct KernelState {
    agents: HashMap<AgentId, Agent>,
    tasks: HashMap<TaskId, CognitiveTask>,
    intents: Vec<IntentMessage>,
    certificates: HashMap<CapabilityCertId, CapabilityCertificate>,
    memory: MemoryService,
    #[allow(dead_code)] // Will be wired into phase 2 scheduler
    reasoning: ReasoningService,
    #[allow(dead_code)] // Tool execution via Phase 2 pipeline
    tools: ToolRegistry,
    started_at: chrono::DateTime<chrono::Utc>,
    completed_tasks_total: u64,
    total_tokens_consumed: u64,
}

/// The concrete AgentKernel implementation — the heart of LAK
pub struct KernelService {
    state: RwLock<KernelState>,
    max_agents: u32,
}

impl KernelService {
    /// Create a new KernelService with default configuration
    pub fn new() -> Self {
        Self {
            state: RwLock::new(KernelState {
                agents: HashMap::new(),
                tasks: HashMap::new(),
                intents: Vec::new(),
                certificates: HashMap::new(),
                memory: MemoryService::new(),
                reasoning: ReasoningService::new(),
                tools: ToolRegistry::new(),
                started_at: Utc::now(),
                completed_tasks_total: 0,
                total_tokens_consumed: 0,
            }),
            max_agents: 1000,
        }
    }
}

impl Default for KernelService {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AgentKernel for KernelService {
    // ── Agent Lifecycle ──

    async fn create_agent(&self, spec: AgentSpec) -> Result<AgentId, KernelError> {
        let mut state = self.state.write().await;
        if state.agents.len() >= self.max_agents as usize {
            return Err(KernelError::AgentLimitReached {
                current: state.agents.len() as u32,
                max: self.max_agents,
            });
        }

        let agent_id = AgentId::new();
        let now = Utc::now();
        let agent = Agent {
            id: agent_id,
            name: spec.name.clone(),
            spec,
            state: AgentState::Created,
            stats: AgentStats::default(),
            created_at: now,
            last_active_at: now,
            terminated_at: None,
        };
        state.agents.insert(agent_id, agent);
        Ok(agent_id)
    }

    async fn destroy_agent(&self, agent_id: AgentId) -> Result<(), KernelError> {
        let mut state = self.state.write().await;
        let agent = state
            .agents
            .get_mut(&agent_id)
            .ok_or(KernelError::AgentNotFound(agent_id))?;
        agent.state = AgentState::Terminated;
        agent.terminated_at = Some(Utc::now());
        Ok(())
    }

    async fn get_agent(&self, agent_id: AgentId) -> Result<Agent, KernelError> {
        let state = self.state.read().await;
        state
            .agents
            .get(&agent_id)
            .cloned()
            .ok_or(KernelError::AgentNotFound(agent_id))
    }

    async fn list_agents(&self) -> Result<Vec<Agent>, KernelError> {
        let state = self.state.read().await;
        Ok(state.agents.values().cloned().collect())
    }

    async fn pause_agent(&self, agent_id: AgentId) -> Result<(), KernelError> {
        let mut state = self.state.write().await;
        let agent = state
            .agents
            .get_mut(&agent_id)
            .ok_or(KernelError::AgentNotFound(agent_id))?;
        agent.state = AgentState::Suspended;
        Ok(())
    }

    async fn resume_agent(&self, agent_id: AgentId) -> Result<(), KernelError> {
        let mut state = self.state.write().await;
        let agent = state
            .agents
            .get_mut(&agent_id)
            .ok_or(KernelError::AgentNotFound(agent_id))?;
        agent.state = AgentState::Idle;
        agent.last_active_at = Utc::now();
        Ok(())
    }

    // ── Cognitive Tasks ──

    async fn submit_task(&self, task: CognitiveTask) -> Result<TaskId, KernelError> {
        let task_id = task.task_id;
        // Verify the referenced agent exists
        {
            let state = self.state.read().await;
            if !state.agents.contains_key(&task.agent_id) {
                return Err(KernelError::AgentNotFound(task.agent_id));
            }
        }
        let mut state = self.state.write().await;
        state.tasks.insert(task_id, task);
        Ok(task_id)
    }

    async fn cancel_task(&self, task_id: TaskId) -> Result<(), KernelError> {
        let mut state = self.state.write().await;
        let task = state
            .tasks
            .get_mut(&task_id)
            .ok_or(KernelError::TaskNotFound(task_id))?;
        task.state = TaskState::Cancelled;
        task.updated_at = Utc::now();
        Ok(())
    }

    async fn get_task(&self, task_id: TaskId) -> Result<CognitiveTask, KernelError> {
        let state = self.state.read().await;
        state
            .tasks
            .get(&task_id)
            .cloned()
            .ok_or(KernelError::TaskNotFound(task_id))
    }

    // ── Intent Routing ──

    async fn send_intent(&self, intent: IntentMessage) -> Result<IntentId, KernelError> {
        let intent_id = intent.intent_id;
        let mut state = self.state.write().await;
        state.intents.push(intent);
        Ok(intent_id)
    }

    async fn await_intent(
        &self,
        _agent_id: AgentId,
        subscription: IntentSubscription,
    ) -> Result<IntentMessage, KernelError> {
        // MVP: simple linear scan of intent queue
        // Phase 2: proper pub/sub with topics and efficient matching
        let state = self.state.read().await;
        for intent in state.intents.iter().rev() {
            // Filter by intent type
            if let Some(ref sub_types) = subscription.intent_types {
                if !sub_types.is_empty() && !sub_types.contains(&intent.intent_type) {
                    continue;
                }
            }
            // Filter by topic pattern
            if let Some(ref pattern) = subscription.topic_pattern {
                let content = intent.content.natural_language.to_lowercase();
                if !content.contains(&pattern.to_lowercase()) {
                    continue;
                }
            }
            return Ok(intent.clone());
        }

        Err(KernelError::Timeout { duration_ms: 0 })
    }

    // ── Semantic Memory ──

    async fn store_memory(
        &self,
        agent_id: AgentId,
        chunk: MemoryChunk,
    ) -> Result<(), KernelError> {
        let mut state = self.state.write().await;
        state.memory.store(agent_id, chunk);
        Ok(())
    }

    async fn query_memory(
        &self,
        agent_id: AgentId,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<MemoryChunk>, KernelError> {
        let state = self.state.read().await;
        Ok(state.memory.query(agent_id, query, top_k))
    }

    async fn forget_memory(
        &self,
        agent_id: AgentId,
        chunk_id: MemoryChunkId,
    ) -> Result<(), KernelError> {
        let mut state = self.state.write().await;
        if state.memory.forget(agent_id, chunk_id) {
            Ok(())
        } else {
            Err(KernelError::NotImplemented(format!(
                "memory chunk {chunk_id} not found"
            )))
        }
    }

    // ── Capability Management ──

    async fn grant_capability(
        &self,
        from_agent: AgentId,
        to_agent: AgentId,
        capability: Capability,
    ) -> Result<CapabilityCertId, KernelError> {
        let mut state = self.state.write().await;
        // Verify both agents exist
        if !state.agents.contains_key(&from_agent) {
            return Err(KernelError::AgentNotFound(from_agent));
        }
        if !state.agents.contains_key(&to_agent) {
            return Err(KernelError::AgentNotFound(to_agent));
        }

        let cert_id = CapabilityCertId::new();
        let cert = CapabilityCertificate {
            cert_id,
            agent_id: to_agent,
            issued_by: from_agent,
            capabilities: vec![capability],
            issued_at: Utc::now(),
            expires_at: None,
            parent_cert_id: None,
        };
        state.certificates.insert(cert_id, cert);
        Ok(cert_id)
    }

    async fn revoke_capability(&self, cert_id: CapabilityCertId) -> Result<(), KernelError> {
        let mut state = self.state.write().await;
        state.certificates.remove(&cert_id).map(|_| ()).ok_or(
            KernelError::DelegationError(format!("certificate not found: {cert_id}")),
        )
    }

    async fn delegate_capability(
        &self,
        from_agent: AgentId,
        to_agent: AgentId,
        requirement: CapabilityRequirement,
        new_scope: Option<String>,
        new_permissions: Option<u32>,
    ) -> Result<CapabilityCertId, KernelError> {
        let mut state = self.state.write().await;
        // Verify both agents exist
        if !state.agents.contains_key(&from_agent) {
            return Err(KernelError::AgentNotFound(from_agent));
        }
        if !state.agents.contains_key(&to_agent) {
            return Err(KernelError::AgentNotFound(to_agent));
        }

        // Find a certificate the from_agent owns that matches the requirement
        let matching_cert = state.certificates.values().find(|cert| {
            cert.agent_id == from_agent
                && cert
                    .capabilities
                    .iter()
                    .any(|cap| cap.satisfies(&requirement))
        });

        let parent_id = matching_cert.map(|c| c.cert_id);

        // Create attenuated capability
        let attenuated_permissions = match new_permissions {
            Some(p) => CapabilityPermission::from_bits_truncate(p) & requirement.min_permissions,
            None => requirement.min_permissions,
        };

        let scope = CapabilityScope {
            pattern: new_scope.unwrap_or_else(|| requirement.scope.clone()),
        };

        let attenuated_cap = Capability {
            cap_type: requirement.cap_type,
            scope,
            permissions: attenuated_permissions,
            constraints: vec![],
        };

        let cert_id = CapabilityCertId::new();
        let cert = CapabilityCertificate {
            cert_id,
            agent_id: to_agent,
            issued_by: from_agent,
            capabilities: vec![attenuated_cap],
            issued_at: Utc::now(),
            expires_at: None,
            parent_cert_id: parent_id,
        };

        state.certificates.insert(cert_id, cert);
        Ok(cert_id)
    }

    async fn get_capabilities(
        &self,
        agent_id: AgentId,
    ) -> Result<CapabilityCertificate, KernelError> {
        let state = self.state.read().await;
        // Collect all capabilities for this agent across all certificates
        let caps: Vec<Capability> = state
            .certificates
            .values()
            .filter(|cert| cert.agent_id == agent_id)
            .flat_map(|cert| cert.capabilities.clone())
            .collect();

        Ok(CapabilityCertificate {
            cert_id: CapabilityCertId::new(),
            agent_id,
            issued_by: AgentId::SYSTEM,
            capabilities: caps,
            issued_at: Utc::now(),
            expires_at: None,
            parent_cert_id: None,
        })
    }

    // ── System ──

    async fn get_system_status(&self) -> Result<SystemStatus, KernelError> {
        let state = self.state.read().await;
        let active_agents = state
            .agents
            .values()
            .filter(|a| a.state.is_active())
            .count() as u32;

        let pending_tasks = state
            .tasks
            .values()
            .filter(|t| t.state == TaskState::Pending)
            .count() as u32;

        let avg_coi = if state.agents.is_empty() {
            0.0
        } else {
            state.agents.values().map(|a| a.stats.coi).sum::<f32>()
                / state.agents.len() as f32
        };

        let uptime = Utc::now()
            .signed_duration_since(state.started_at)
            .num_seconds() as u64;

        Ok(SystemStatus {
            active_agents,
            max_agents: self.max_agents,
            pending_tasks,
            completed_tasks_total: state.completed_tasks_total,
            total_tokens_consumed: state.total_tokens_consumed,
            average_coi: avg_coi,
            scheduler_load: pending_tasks as f32 / self.max_agents.max(1) as f32,
            uptime_seconds: uptime,
        })
    }

    async fn shutdown(&self) -> Result<(), KernelError> {
        tracing::info!("[KernelService] Shutdown requested");
        Ok(())
    }
}
