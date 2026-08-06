//! Agent Runtime Environment (ARE)
//!
//! Provides the execution environment for individual agents:
//! AgentProcess lifecycle, context management, and cognitive execution.

use chrono::Utc;
use lak_core::types::agent::{AgentSpec, AgentState, AgentStats};
use lak_core::types::capability::CapabilityCertificate;
use lak_core::types::context::{ContextWindow, TokenSource};
use lak_core::types::ids::AgentId;

/// An executing agent — the runtime representation of a single agent.
///
/// Each AgentProcess holds the agent's context window, capabilities,
/// and runtime statistics. It's the unit of execution within the
/// Agent Kernel, analogous to a process in a traditional OS.
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
    /// Create a new agent process from spec and capabilities.
    ///
    /// The `agent_id` must be the kernel-assigned ID so that the process,
    /// the agent record and the capability certificate all agree.
    pub fn new(agent_id: AgentId, spec: AgentSpec, capabilities: CapabilityCertificate) -> Self {
        let max_tokens = spec.max_context_tokens;
        Self {
            agent_id,
            spec,
            state: AgentState::Created,
            context: ContextWindow::new(max_tokens),
            capabilities,
            stats: AgentStats::default(),
            created_at: Utc::now(),
            last_active_at: Utc::now(),
        }
    }

    /// Build the context string for LLM prompting
    pub fn build_context_string(&self) -> String {
        self.context
            .tokens
            .iter()
            .map(|t| format!("[{}] {}", t.source.source_label(), t.content))
            .collect::<Vec<_>>()
            .join("\n")
    }

    // ─── State Management ───

    /// Transition to Running state
    pub fn activate(&mut self) {
        self.state = AgentState::Running;
        self.last_active_at = Utc::now();
    }

    /// Transition to Idle state
    pub fn idle(&mut self) {
        self.state = AgentState::Idle;
    }

    /// Suspend the agent (context preserved)
    pub fn suspend(&mut self) {
        self.state = AgentState::Suspended;
    }

    /// Terminate the agent
    pub fn terminate(&mut self) {
        self.state = AgentState::Terminated;
    }

    // ─── Context Management ───

    /// Add content to the agent's context window
    pub fn append_context(&mut self, content: impl Into<String>, source: TokenSource) {
        self.context.append(content, source);
    }

    /// Get the current context token count
    pub fn context_token_count(&self) -> usize {
        self.context.token_count
    }

    /// Check if context is full
    pub fn is_context_full(&self) -> bool {
        self.context.is_full()
    }

    /// Compress context (keep most recent N%)
    pub fn compress_context(&mut self) -> usize {
        self.context.compress(0.7) // Keep 70%
    }

    // ─── Stats Tracking ───

    /// Record a completed task, updating running statistics
    pub fn record_task_completion(&mut self, tokens: u64, tool_calls: u32, latency_ms: u64) {
        self.stats.total_tasks_completed += 1;
        self.stats.total_tokens_consumed += tokens;
        self.stats.total_tool_calls += u64::from(tool_calls);
        self.stats.avg_response_latency_seconds =
            self.stats.avg_response_latency_seconds * 0.9 + (latency_ms as f64 / 1000.0) * 0.1;
        self.last_active_at = Utc::now();
    }

    /// Record a failed task
    pub fn record_task_failure(&mut self) {
        self.stats.total_tasks_failed += 1;
        self.last_active_at = Utc::now();
    }

    /// Record a capability violation
    pub fn record_capability_violation(&mut self) {
        self.stats.capability_violations += 1;
    }

    /// Update the Cognitive Opportunity Index based on recent activity
    pub fn recalculate_coi(&mut self) {
        let success_rate = if self.stats.total_tasks_completed + self.stats.total_tasks_failed > 0 {
            self.stats.total_tasks_completed as f32
                / (self.stats.total_tasks_completed + self.stats.total_tasks_failed) as f32
        } else {
            0.5
        };

        let activity_factor = (self.stats.total_tokens_consumed as f32 / 1_000_000.0).min(1.0);
        let violation_penalty = (self.stats.capability_violations as f32 * 0.1).min(0.5);

        self.stats.coi = (success_rate * 0.5 + activity_factor * 0.3 + 0.2) - violation_penalty;
        self.stats.coi = self.stats.coi.clamp(0.0, 1.0);
    }
}

impl Default for AgentProcess {
    fn default() -> Self {
        Self {
            agent_id: AgentId::SUPERVISOR,
            spec: AgentSpec::default(),
            state: AgentState::Created,
            context: ContextWindow::new(32768),
            capabilities: CapabilityCertificate {
                cert_id: lak_core::types::ids::CapabilityCertId::new(),
                agent_id: AgentId::SUPERVISOR,
                issued_by: AgentId::SYSTEM,
                capabilities: vec![],
                issued_at: Utc::now(),
                expires_at: None,
                parent_cert_id: None,
            },
            stats: AgentStats::default(),
            created_at: Utc::now(),
            last_active_at: Utc::now(),
        }
    }
}

/// Extension trait to add display labels to TokenSource
pub trait TokenSourceExt {
    fn source_label(&self) -> &'static str;
}

impl TokenSourceExt for TokenSource {
    fn source_label(&self) -> &'static str {
        match self {
            TokenSource::SystemPrompt => "SYSTEM",
            TokenSource::UserInput => "USER",
            TokenSource::AgentThought => "THOUGHT",
            TokenSource::ToolOutput => "TOOL",
            TokenSource::MemoryRetrieval => "MEMORY",
            TokenSource::IntentReceived => "INTENT",
            TokenSource::FileContent => "FILE",
        }
    }
}
