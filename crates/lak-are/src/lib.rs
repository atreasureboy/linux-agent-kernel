//! Agent Runtime Environment (ARE)
//!
//! Provides the execution environment for individual agents:
//! AgentProcess lifecycle, context management, and intent parsing.

use chrono::Utc;
use lak_core::types::agent::{AgentSpec, AgentState, AgentStats};
use lak_core::types::capability::CapabilityCertificate;
use lak_core::types::context::ContextWindow;
use lak_core::types::ids::AgentId;

/// An executing agent — the runtime representation of a single agent
pub struct AgentProcess {
    pub agent_id: AgentId,
    pub spec: AgentSpec,
    pub state: AgentState,
    pub context: ContextWindow,
    pub capabilities: CapabilityCertificate,
    pub stats: AgentStats,
    pub created_at: chrono::DateTime<Utc>,
    pub last_active_at: chrono::DateTime<Utc>,
}

impl AgentProcess {
    /// Create a new agent process
    pub fn new(spec: AgentSpec, capabilities: CapabilityCertificate) -> Self {
        let max_tokens = spec.max_context_tokens;
        Self {
            agent_id: AgentId::new(),
            spec,
            state: AgentState::Created,
            context: ContextWindow::new(max_tokens),
            capabilities,
            stats: AgentStats::default(),
            created_at: Utc::now(),
            last_active_at: Utc::now(),
        }
    }

    /// Transition to Running state
    pub fn activate(&mut self) {
        self.state = AgentState::Running;
        self.last_active_at = Utc::now();
    }

    /// Transition to Idle state
    pub fn idle(&mut self) {
        self.state = AgentState::Idle;
    }

    /// Suspend the agent
    pub fn suspend(&mut self) {
        self.state = AgentState::Suspended;
    }

    /// Terminate the agent
    pub fn terminate(&mut self) {
        self.state = AgentState::Terminated;
    }

    /// Record a completed task
    pub fn record_task_completion(&mut self, tokens: u64, tool_calls: u32, latency_ms: u64) {
        self.stats.total_tasks_completed += 1;
        self.stats.total_tokens_consumed += tokens;
        self.stats.total_tool_calls += u64::from(tool_calls);
        self.stats.avg_response_latency_seconds = self
            .stats
            .avg_response_latency_seconds
            * 0.9
            + (latency_ms as f64 / 1000.0) * 0.1;
        self.last_active_at = Utc::now();
    }

    /// Add content to the agent's context window
    pub fn append_context(
        &mut self,
        content: impl Into<String>,
        source: lak_core::types::context::TokenSource,
    ) {
        self.context.append(content, source);
    }
}
