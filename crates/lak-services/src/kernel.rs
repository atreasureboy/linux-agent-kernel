//! KernelService — concrete implementation of the AgentKernel trait
//!
//! This is the operational core of LAK: it wires together memory,
//! reasoning, scheduler, tool registry, agent processes, and manages
//! agent/task/intent/capability state.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use tokio::sync::RwLock;

use lak_are::AgentProcess;
use lak_core::error::KernelError;
use lak_core::token_budget::TokenBudget;
use lak_core::traits::{AgentKernel, SystemStatus};
use lak_core::types::agent::{Agent, AgentSpec, AgentState, AgentStats};
use lak_core::types::capability::{
    Capability, CapabilityCertificate, CapabilityPermission, CapabilityRequirement, CapabilityScope,
};
use lak_core::types::ids::{AgentId, CapabilityCertId, IntentId, MemoryChunkId, TaskId};
use lak_core::types::intent::{IntentMessage, IntentSubscription};
use lak_core::types::memory::MemoryChunk;
use lak_core::types::task::{CognitiveTask, TaskState};
use lak_tal::tools::{FileReadTool, HttpGetTool, ShellCmdTool, ToolContext, ToolResult};

use super::intent_router::IntentRouter;
use super::journal::{CognitiveJournal, JournalOperation};
use super::memory::service::MemoryService;
use super::reasoning::service::ReasoningService;
use super::scheduler::CognitiveScheduler;
use super::tool_registry::ToolRegistry;

/// Per-agent runtime state
struct AgentRuntime {
    agent: Agent,
    process: AgentProcess,
    /// Token budget for this agent
    token_budget: TokenBudget,
}

/// Internal state protected by a read-write lock
struct KernelState {
    /// Agent runtimes (Agent + AgentProcess + TokenBudget)
    agents: HashMap<AgentId, AgentRuntime>,
    /// Task storage
    tasks: HashMap<TaskId, CognitiveTask>,
    /// Capability certificates
    certificates: HashMap<CapabilityCertId, CapabilityCertificate>,
    /// Intent pub/sub router (delivers to per-agent mailboxes)
    router: IntentRouter,
    /// Per-agent intent mailboxes (delivered, not yet consumed)
    mailboxes: HashMap<AgentId, VecDeque<IntentMessage>>,
    /// Semantic memory
    memory: MemoryService,
    /// Reasoning service (LLM orchestration)
    reasoning: ReasoningService,
    /// Cognitive scheduler (COI-based)
    scheduler: CognitiveScheduler,
    /// Tool registry
    tools: ToolRegistry,
    /// WAL-style journal of task state transitions
    journal: CognitiveJournal,
    /// Kernel start time
    started_at: chrono::DateTime<chrono::Utc>,
    /// Cumulative completed tasks
    completed_tasks_total: u64,
    /// Total tokens consumed across all agents
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
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(FileReadTool));
        tools.register(Arc::new(ShellCmdTool));
        tools.register(Arc::new(HttpGetTool));

        Self {
            state: RwLock::new(KernelState {
                agents: HashMap::new(),
                tasks: HashMap::new(),
                certificates: HashMap::new(),
                router: IntentRouter::new(),
                mailboxes: HashMap::new(),
                memory: MemoryService::new(),
                reasoning: ReasoningService::new(),
                scheduler: CognitiveScheduler::new(10),
                tools,
                journal: CognitiveJournal::new(),
                started_at: Utc::now(),
                completed_tasks_total: 0,
                total_tokens_consumed: 0,
            }),
            max_agents: 1000,
        }
    }

    /// Create with a custom max_agents limit
    pub fn with_max_agents(mut self, max: u32) -> Self {
        self.max_agents = max;
        self
    }

    /// Register an LLM driver with the reasoning service
    pub async fn add_driver(&self, driver: Arc<dyn lak_tal::llm::LLMDriver>) -> usize {
        let mut state = self.state.write().await;
        state.reasoning.add_driver(driver);
        state.reasoning.driver_count()
    }

    /// List the names of all registered tools
    pub async fn list_tools(&self) -> Vec<String> {
        let state = self.state.read().await;
        state
            .tools
            .list()
            .iter()
            .map(|t| t.name().to_string())
            .collect()
    }

    /// Execute a registered tool on behalf of an agent with full
    /// capability enforcement and injection scanning (defense layers 3+4),
    /// followed by an audit record (layer 5).
    pub async fn execute_tool(
        &self,
        agent_id: AgentId,
        tool_name: &str,
        params: serde_json::Value,
        context: ToolContext,
    ) -> Result<(ToolResult, super::injection_defense::AuditEntry), KernelError> {
        let mut state = self.state.write().await;

        let tool = state
            .tools
            .get(tool_name)
            .cloned()
            .ok_or_else(|| KernelError::ToolError {
                tool: tool_name.to_string(),
                message: "tool not registered".into(),
            })?;

        let capabilities = {
            let runtime = state
                .agents
                .get_mut(&agent_id)
                .ok_or(KernelError::AgentNotFound(agent_id))?;
            runtime.process.stats.total_tool_calls += 1;
            // Merged view: initial capabilities + everything granted to
            // this agent — the single source of truth for enforcement.
            Self::merged_certificate(&state, agent_id)
        };

        let result = super::injection_defense::execute_tool_with_enforcement(
            tool.as_ref(),
            agent_id,
            &capabilities,
            params,
            &context,
        )
        .await;

        match result {
            Ok((output, audit)) => Ok((output, audit)),
            Err(e) => {
                if let Some(runtime) = state.agents.get_mut(&agent_id) {
                    runtime.process.stats.total_tool_failures += 1;
                    if matches!(e, lak_tal::tools::ToolError::AccessDenied(_)) {
                        runtime.process.record_capability_violation();
                    }
                }
                Err(KernelError::ToolError {
                    tool: tool_name.to_string(),
                    message: e.to_string(),
                })
            }
        }
    }

    /// Build the merged capability view for an agent: the initial
    /// certificate from its AgentProcess plus every certificate other
    /// agents have granted to it.
    fn merged_certificate(state: &KernelState, agent_id: AgentId) -> CapabilityCertificate {
        let mut caps: Vec<Capability> = state
            .agents
            .get(&agent_id)
            .map(|r| r.process.capabilities.capabilities.clone())
            .unwrap_or_default();

        for cert in state
            .certificates
            .values()
            .filter(|c| c.agent_id == agent_id && c.is_valid())
        {
            caps.extend(cert.capabilities.clone());
        }

        CapabilityCertificate {
            cert_id: CapabilityCertId::new(),
            agent_id,
            issued_by: AgentId::SYSTEM,
            capabilities: caps,
            issued_at: Utc::now(),
            expires_at: None,
            parent_cert_id: None,
        }
    }

    /// Clone the agent record with live runtime stats merged in
    fn agent_snapshot(runtime: &AgentRuntime) -> Agent {
        let mut agent = runtime.agent.clone();
        agent.stats = runtime.process.stats.clone();
        agent
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

        let capabilities = CapabilityCertificate {
            cert_id: CapabilityCertId::new(),
            agent_id,
            issued_by: AgentId::SYSTEM,
            capabilities: spec.initial_capabilities.clone(),
            issued_at: now,
            expires_at: None,
            parent_cert_id: None,
        };

        let process = AgentProcess::new(agent_id, spec.clone(), capabilities);
        let token_budget = TokenBudget::developer_budget();

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

        let runtime = AgentRuntime {
            agent,
            process,
            token_budget,
        };

        tracing::info!(agent_id = %agent_id, name = %runtime.agent.name, "Agent created");
        state.agents.insert(agent_id, runtime);
        Ok(agent_id)
    }

    async fn destroy_agent(&self, agent_id: AgentId) -> Result<(), KernelError> {
        let mut state = self.state.write().await;
        let runtime = state
            .agents
            .get_mut(&agent_id)
            .ok_or(KernelError::AgentNotFound(agent_id))?;
        runtime.agent.state = AgentState::Terminated;
        runtime.agent.terminated_at = Some(Utc::now());
        runtime.process.terminate();

        // Cancel all still-pending tasks belonging to this agent
        let to_cancel: Vec<TaskId> = state
            .tasks
            .values()
            .filter(|t| {
                t.agent_id == agent_id
                    && !matches!(
                        t.state,
                        TaskState::Completed | TaskState::Failed(_) | TaskState::Cancelled
                    )
            })
            .map(|t| t.task_id)
            .collect();
        for tid in to_cancel {
            if let Some(task) = state.tasks.get_mut(&tid) {
                task.state = TaskState::Cancelled;
                task.updated_at = Utc::now();
            }
            state.scheduler.cancel(tid);
        }

        // Drop delivery mailbox (subscriptions stay harmless until GC)
        state.mailboxes.remove(&agent_id);

        tracing::info!(agent_id = %agent_id, "Agent terminated");
        Ok(())
    }

    async fn get_agent(&self, agent_id: AgentId) -> Result<Agent, KernelError> {
        let state = self.state.read().await;
        state
            .agents
            .get(&agent_id)
            .map(Self::agent_snapshot)
            .ok_or(KernelError::AgentNotFound(agent_id))
    }

    async fn list_agents(&self) -> Result<Vec<Agent>, KernelError> {
        let state = self.state.read().await;
        Ok(state.agents.values().map(Self::agent_snapshot).collect())
    }

    async fn pause_agent(&self, agent_id: AgentId) -> Result<(), KernelError> {
        let mut state = self.state.write().await;
        let runtime = state
            .agents
            .get_mut(&agent_id)
            .ok_or(KernelError::AgentNotFound(agent_id))?;
        runtime.agent.state = AgentState::Suspended;
        runtime.process.suspend();
        Ok(())
    }

    async fn resume_agent(&self, agent_id: AgentId) -> Result<(), KernelError> {
        let mut state = self.state.write().await;
        let runtime = state
            .agents
            .get_mut(&agent_id)
            .ok_or(KernelError::AgentNotFound(agent_id))?;
        runtime.agent.state = AgentState::Idle;
        runtime.process.activate();
        runtime.agent.last_active_at = Utc::now();
        Ok(())
    }

    // ── Cognitive Tasks ──

    async fn submit_task(&self, task: CognitiveTask) -> Result<TaskId, KernelError> {
        let task_id = task.task_id;
        let agent_id = task.agent_id;

        let mut state = self.state.write().await;

        // Verify agent exists and check budget (in a block to limit borrow)
        let (coi, should_activate) = {
            let runtime = state
                .agents
                .get_mut(&agent_id)
                .ok_or(KernelError::AgentNotFound(agent_id))?;

            let priority = task.priority.score();
            let estimated_tokens = task.stats.tokens_consumed.max(512);
            match runtime
                .token_budget
                .check_allocation(estimated_tokens, priority)
            {
                lak_core::token_budget::BudgetAllocation::Denied(reason) => {
                    tracing::warn!(
                        agent_id = %agent_id,
                        task_id = %task_id,
                        reason = %reason,
                        "Task rejected by token budget"
                    );
                    return Err(KernelError::TokenBudgetExceeded {
                        used: runtime.token_budget.consumed,
                        limit: runtime.token_budget.hard_limit,
                    });
                }
                _ => {}
            }

            let agent_coi = runtime.process.stats.coi;
            let activate = runtime.agent.state == AgentState::Idle;
            (agent_coi, activate)
        };

        // Journal: record task creation
        state.journal.append(
            task_id,
            agent_id,
            JournalOperation::TaskCreated {
                task_type: format!("{:?}", task.task_type),
                priority_score: task.priority.score(),
            },
        );

        // Submit to scheduler (no conflicting borrow)
        state.scheduler.submit(task.clone(), coi);
        state.tasks.insert(task_id, task);

        // Activate agent if idle
        if should_activate {
            let runtime = state
                .agents
                .get_mut(&agent_id)
                .ok_or(KernelError::AgentNotFound(agent_id))?;
            runtime.agent.state = AgentState::Running;
            runtime.process.activate();
        }

        tracing::debug!(task_id = %task_id, agent_id = %agent_id, "Task submitted");
        Ok(task_id)
    }

    async fn cancel_task(&self, task_id: TaskId) -> Result<(), KernelError> {
        let mut state = self.state.write().await;

        let (agent_id, from) = {
            let task = state
                .tasks
                .get(&task_id)
                .ok_or(KernelError::TaskNotFound(task_id))?;
            (task.agent_id, task.state.clone())
        };

        if let Some(task) = state.tasks.get_mut(&task_id) {
            task.state = TaskState::Cancelled;
            task.updated_at = Utc::now();
        }
        state.scheduler.cancel(task_id);
        state
            .journal
            .record_transition(task_id, agent_id, from, TaskState::Cancelled);
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

        // Publish through the router; delivered copies land in mailboxes,
        // undeliverable intents go to the dead-letter queue for replay.
        let result = state.router.publish(intent.clone());

        state.journal.append(
            TaskId::new(),
            intent.source_agent_id,
            JournalOperation::IntentReceived {
                intent_id: intent_id.to_string(),
                intent_type: format!("{:?}", intent.intent_type),
            },
        );

        for recipient in &result.delivered_to {
            state
                .mailboxes
                .entry(*recipient)
                .or_default()
                .push_back(intent.clone());
        }

        if result.dead_lettered {
            tracing::debug!(intent_id = %intent_id, "Intent dead-lettered (no subscribers)");
        }

        Ok(intent_id)
    }

    async fn await_intent(
        &self,
        agent_id: AgentId,
        subscription: IntentSubscription,
    ) -> Result<IntentMessage, KernelError> {
        let mut state = self.state.write().await;

        // Ensure the subscription is registered so future publishes match.
        // `subscribe_once` keeps repeated await_intent calls idempotent.
        state.router.subscribe_once(IntentSubscription {
            agent_id,
            ..subscription.clone()
        });

        // 1. Drain this agent's mailbox (FIFO order)
        if let Some(mailbox) = state.mailboxes.get_mut(&agent_id) {
            if let Some(pos) = mailbox
                .iter()
                .position(|i| IntentRouter::matches(&subscription, i))
            {
                return Ok(mailbox.remove(pos).expect("position checked above"));
            }
        }

        // 2. Replay dead letters: intents published before this agent
        //    subscribed may still be waiting in the dead-letter queue.
        let dead_ids: Vec<IntentId> = state
            .router
            .dead_letters()
            .iter()
            .filter(|entry| IntentRouter::matches(&subscription, entry.intent()))
            .map(|entry| entry.intent().intent_id)
            .collect();

        for id in dead_ids {
            if let Some(intent) = state.router.requeue_dead_letter(id) {
                tracing::debug!(intent_id = %id, agent_id = %agent_id, "Replayed dead-lettered intent");
                return Ok(intent);
            }
        }

        // 3. Nothing matched yet — caller may retry (poll semantics in MVP)
        Err(KernelError::Timeout { duration_ms: 0 })
    }

    // ── Semantic Memory ──

    async fn store_memory(&self, agent_id: AgentId, chunk: MemoryChunk) -> Result<(), KernelError> {
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
        let mut state = self.state.write().await;
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
            Err(KernelError::MemoryNotFound(chunk_id))
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
        if !state.agents.contains_key(&from_agent) {
            return Err(KernelError::AgentNotFound(from_agent));
        }
        if !state.agents.contains_key(&to_agent) {
            return Err(KernelError::AgentNotFound(to_agent));
        }

        // Security: only agents that actually hold a delegatable capability
        // of this kind may grant it further (no conjuring rights out of
        // thin air). SYSTEM is exempt as the root of trust.
        if from_agent != AgentId::SYSTEM {
            let grantor_caps = Self::merged_certificate(&state, from_agent);
            let requirement = CapabilityRequirement {
                cap_type: capability.cap_type.clone(),
                scope: capability.scope.pattern.clone(),
                min_permissions: capability.permissions | CapabilityPermission::DELEGATE,
            };
            if !grantor_caps.has_capability(&requirement) {
                return Err(KernelError::InsufficientCapability {
                    required: vec![requirement],
                    have: grantor_caps.capabilities,
                });
            }
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
        state
            .certificates
            .remove(&cert_id)
            .map(|_| ())
            .ok_or(KernelError::DelegationError(format!(
                "certificate not found: {cert_id}"
            )))
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
        if !state.agents.contains_key(&from_agent) {
            return Err(KernelError::AgentNotFound(from_agent));
        }
        if !state.agents.contains_key(&to_agent) {
            return Err(KernelError::AgentNotFound(to_agent));
        }

        // Locate the source capability: search granted certificates first,
        // then fall back to the agent's initial certificate. The source
        // must satisfy the requirement AND carry the DELEGATE flag
        // (`find_capability` enforces both).
        let mut source_cap: Option<(CapabilityCertId, Capability)> = None;

        for cert in state
            .certificates
            .values()
            .filter(|cert| cert.agent_id == from_agent && cert.is_valid())
        {
            if let Some(cap) = cert.find_capability(&requirement) {
                source_cap = Some((cert.cert_id, cap.clone()));
                break;
            }
        }

        if source_cap.is_none() {
            if let Some(runtime) = state.agents.get(&from_agent) {
                let initial = &runtime.process.capabilities;
                if initial.is_valid() {
                    if let Some(cap) = initial.find_capability(&requirement) {
                        source_cap = Some((initial.cert_id, cap.clone()));
                    }
                }
            }
        }

        let (parent_id, source_cap) =
            source_cap.ok_or_else(|| KernelError::InsufficientCapability {
                required: vec![requirement.clone()],
                have: Self::merged_certificate(&state, from_agent).capabilities,
            })?;

        // Attenuate: the delegated capability may only shrink scope and
        // permissions, never expand them (enforced by `Capability::attenuate`).
        // Default (least privilege): exactly the requirement's minimum
        // permissions, further intersected with the source's permissions.
        let attenuated_scope = new_scope.map(|pattern| CapabilityScope { pattern });
        let attenuated_permissions = new_permissions
            .map(CapabilityPermission::from_bits_truncate)
            .unwrap_or(requirement.min_permissions)
            & source_cap.permissions;

        let delegated = source_cap
            .attenuate(attenuated_scope, Some(attenuated_permissions), vec![])
            .map_err(|e| KernelError::DelegationError(e.to_string()))?;

        let cert_id = CapabilityCertId::new();
        let cert = CapabilityCertificate {
            cert_id,
            agent_id: to_agent,
            issued_by: from_agent,
            capabilities: vec![delegated],
            issued_at: Utc::now(),
            expires_at: None,
            parent_cert_id: Some(parent_id),
        };
        state.certificates.insert(cert_id, cert);
        Ok(cert_id)
    }

    async fn get_capabilities(
        &self,
        agent_id: AgentId,
    ) -> Result<CapabilityCertificate, KernelError> {
        let state = self.state.read().await;
        Ok(Self::merged_certificate(&state, agent_id))
    }

    // ── System ──

    async fn get_system_status(&self) -> Result<SystemStatus, KernelError> {
        let state = self.state.read().await;
        let active_agents = state
            .agents
            .values()
            .filter(|r| r.agent.state.is_active())
            .count() as u32;

        let avg_coi = if state.agents.is_empty() {
            0.0
        } else {
            state
                .agents
                .values()
                .map(|r| r.process.stats.coi)
                .sum::<f32>()
                / state.agents.len() as f32
        };

        let uptime = Utc::now()
            .signed_duration_since(state.started_at)
            .num_seconds()
            .max(0) as u64;

        Ok(SystemStatus {
            active_agents,
            max_agents: self.max_agents,
            pending_tasks: state.scheduler.pending_count() as u32,
            completed_tasks_total: state.completed_tasks_total,
            total_tokens_consumed: state.total_tokens_consumed,
            average_coi: avg_coi,
            scheduler_load: state.scheduler.load() as f32,
            uptime_seconds: uptime,
        })
    }

    async fn shutdown(&self) -> Result<(), KernelError> {
        tracing::info!("[KernelService] Shutdown requested");
        Ok(())
    }
}
