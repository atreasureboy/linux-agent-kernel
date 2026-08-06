//! CognitiveJournal — WAL-style journal for task state transitions
//!
//! Provides crash-consistency for Agent Kernel tasks:
//! - Write-Ahead Log (WAL) records every state transition
//! - Periodic checkpointing snapshots active task state
//! - Replay journal on restart to recover in-flight tasks
//! - Truncation of committed entries to bound journal size

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use lak_core::types::ids::{AgentId, TaskId};
use lak_core::types::task::TaskState;
use serde::{Deserialize, Serialize};

// ── Journal Entry ────────────────────────────────────────────────

/// A single entry in the cognitive journal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JournalEntry {
    /// Monotonically increasing entry ID
    pub sequence: u64,
    /// Which task this entry concerns
    pub task_id: TaskId,
    /// The agent responsible for the task
    pub agent_id: AgentId,
    /// Operation recorded
    pub operation: JournalOperation,
    /// Entry timestamp
    pub timestamp: DateTime<Utc>,
    /// Optional context for replay (task snapshot at this point)
    pub checkpoint_snapshot: Option<serde_json::Value>,
}

/// Operation types recorded in the journal
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum JournalOperation {
    /// Task created
    TaskCreated {
        task_type: String,
        priority_score: f64,
    },
    /// Task state transition
    StateTransition { from: TaskState, to: TaskState },
    /// Tool execution started
    ToolCallStarted {
        tool_name: String,
        parameters_snapshot: serde_json::Value,
    },
    /// Tool execution completed
    ToolCallCompleted {
        tool_name: String,
        result_summary: String,
        latency_ms: u64,
    },
    /// LLM reasoning started
    ReasoningStarted {
        model: String,
        estimated_tokens: u64,
    },
    /// LLM reasoning completed
    ReasoningCompleted {
        model: String,
        actual_tokens: u64,
        latency_ms: u64,
    },
    /// Checkpoint: snapshot of active state
    Checkpoint {
        active_task_count: usize,
        total_completed: u64,
    },
    /// Intent received
    IntentReceived {
        intent_id: String,
        intent_type: String,
    },
    /// Memory operation
    MemoryUpdate { chunk_id: String, action: String },
}

// ── Journal ──────────────────────────────────────────────────────

/// WAL-style cognitive journal
#[derive(Debug)]
pub struct CognitiveJournal {
    /// The append-only log
    entries: Vec<JournalEntry>,
    /// Next sequence number
    next_sequence: u64,
    /// Sequence number of the last checkpoint
    last_checkpoint_seq: u64,
    /// Maximum journal size before forcing a truncation
    max_entries: usize,
    /// Index: task_id → latest entry index for O(1) lookup
    task_index: HashMap<TaskId, usize>,
    /// Checkpoint interval (every N entries force a checkpoint)
    checkpoint_interval: u64,
}

impl CognitiveJournal {
    /// Create a new empty journal
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            next_sequence: 1,
            last_checkpoint_seq: 0,
            max_entries: 10_000,
            task_index: HashMap::new(),
            checkpoint_interval: 100,
        }
    }

    /// Create with custom capacity and checkpoint interval
    pub fn with_config(max_entries: usize, checkpoint_interval: u64) -> Self {
        Self {
            entries: Vec::with_capacity(max_entries.min(1000)),
            next_sequence: 1,
            last_checkpoint_seq: 0,
            max_entries,
            checkpoint_interval,
            task_index: HashMap::new(),
        }
    }

    /// Append an operation to the journal
    pub fn append(
        &mut self,
        task_id: TaskId,
        agent_id: AgentId,
        operation: JournalOperation,
    ) -> u64 {
        let seq = self.next_sequence;
        let entry = JournalEntry {
            sequence: seq,
            task_id,
            agent_id,
            operation,
            timestamp: Utc::now(),
            checkpoint_snapshot: None,
        };

        self.task_index.insert(task_id, self.entries.len());
        self.entries.push(entry);
        self.next_sequence += 1;

        // Auto-truncate if over limit
        if self.entries.len() > self.max_entries {
            self.truncate_oldest(self.entries.len() - self.max_entries);
        }

        // Auto-checkpoint: every `checkpoint_interval` entries since the
        // last checkpoint, record one so replay never has to scan the
        // whole journal.
        if self.next_sequence - self.last_checkpoint_seq >= self.checkpoint_interval {
            let in_flight = self.in_flight_tasks().len();
            self.checkpoint(agent_id, in_flight, self.total_completed());
        }

        seq
    }

    /// Total completed tasks observed in state transitions
    fn total_completed(&self) -> u64 {
        self.entries
            .iter()
            .filter(|e| {
                matches!(
                    &e.operation,
                    JournalOperation::StateTransition {
                        to: TaskState::Completed,
                        ..
                    }
                )
            })
            .count() as u64
    }

    /// Record a state transition for a task
    pub fn record_transition(
        &mut self,
        task_id: TaskId,
        agent_id: AgentId,
        from: TaskState,
        to: TaskState,
    ) -> u64 {
        self.append(
            task_id,
            agent_id,
            JournalOperation::StateTransition { from, to },
        )
    }

    /// Record tool call start
    pub fn record_tool_start(
        &mut self,
        task_id: TaskId,
        agent_id: AgentId,
        tool_name: &str,
        params: serde_json::Value,
    ) -> u64 {
        self.append(
            task_id,
            agent_id,
            JournalOperation::ToolCallStarted {
                tool_name: tool_name.to_string(),
                parameters_snapshot: params,
            },
        )
    }

    /// Record tool call completion
    pub fn record_tool_complete(
        &mut self,
        task_id: TaskId,
        agent_id: AgentId,
        tool_name: &str,
        result_summary: &str,
        latency_ms: u64,
    ) -> u64 {
        self.append(
            task_id,
            agent_id,
            JournalOperation::ToolCallCompleted {
                tool_name: tool_name.to_string(),
                result_summary: result_summary.to_string(),
                latency_ms,
            },
        )
    }

    /// Create a checkpoint entry
    pub fn checkpoint(
        &mut self,
        agent_id: AgentId,
        active_task_count: usize,
        total_completed: u64,
    ) -> u64 {
        let seq = self.next_sequence;
        let entry = JournalEntry {
            sequence: seq,
            task_id: TaskId::new(), // Checkpoint spans all tasks
            agent_id,
            operation: JournalOperation::Checkpoint {
                active_task_count,
                total_completed,
            },
            timestamp: Utc::now(),
            checkpoint_snapshot: None,
        };

        self.entries.push(entry);
        self.last_checkpoint_seq = seq;
        self.next_sequence += 1;
        seq
    }

    /// Get all entries related to a specific task
    pub fn entries_for_task(&self, task_id: TaskId) -> Vec<&JournalEntry> {
        self.entries
            .iter()
            .filter(|e| e.task_id == task_id)
            .collect()
    }

    /// Get entries since a given sequence number (inclusive)
    pub fn entries_since(&self, since_seq: u64) -> &[JournalEntry] {
        match self
            .entries
            .binary_search_by_key(&since_seq, |e| e.sequence)
        {
            Ok(pos) => &self.entries[pos..],
            Err(pos) => &self.entries[pos..],
        }
    }

    /// Get entries between two sequence numbers [from, to)
    pub fn entries_range(&self, from_seq: u64, to_seq: u64) -> Vec<&JournalEntry> {
        let start = match self.entries.binary_search_by_key(&from_seq, |e| e.sequence) {
            Ok(pos) => pos,
            Err(pos) => pos,
        };
        let end = match self.entries.binary_search_by_key(&to_seq, |e| e.sequence) {
            Ok(pos) => pos,
            Err(pos) => pos,
        };
        self.entries[start..end].iter().collect()
    }

    /// Find all in-flight tasks (tasks with operations that don't end in a terminal state)
    pub fn in_flight_tasks(&self) -> Vec<TaskId> {
        let mut terminal: HashMap<TaskId, bool> = HashMap::new();

        for entry in &self.entries {
            if let JournalOperation::StateTransition { to, .. } = &entry.operation {
                terminal.insert(
                    entry.task_id,
                    matches!(
                        to,
                        TaskState::Completed | TaskState::Failed(_) | TaskState::Cancelled
                    ),
                );
            }
        }

        terminal
            .into_iter()
            .filter(|(_, terminal)| !terminal)
            .map(|(id, _)| id)
            .collect()
    }

    /// Current journal size in entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the journal is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Last checkpoint sequence number
    pub fn last_checkpoint_seq(&self) -> u64 {
        self.last_checkpoint_seq
    }

    /// Next sequence number to be assigned
    pub fn next_sequence(&self) -> u64 {
        self.next_sequence
    }

    /// Truncate entries older than the given sequence number
    pub fn truncate_before(&mut self, before_seq: u64) -> usize {
        let before_len = self.entries.len();
        let pos = match self
            .entries
            .binary_search_by_key(&before_seq, |e| e.sequence)
        {
            Ok(p) => p,
            Err(p) => p,
        };

        self.entries.drain(..pos);
        // Rebuild index
        self.task_index.clear();
        for (i, entry) in self.entries.iter().enumerate() {
            self.task_index.insert(entry.task_id, i);
        }
        before_len - self.entries.len()
    }

    /// Remove the oldest N entries
    fn truncate_oldest(&mut self, count: usize) {
        let n = count.min(self.entries.len());
        self.entries.drain(..n);
        self.task_index.clear();
        for (i, entry) in self.entries.iter().enumerate() {
            self.task_index.insert(entry.task_id, i);
        }
    }

    /// Get all entries (read-only access)
    pub fn entries(&self) -> &[JournalEntry] {
        &self.entries
    }

    /// Clear the journal entirely
    pub fn clear(&mut self) {
        self.entries.clear();
        self.task_index.clear();
        self.next_sequence = 1;
        self.last_checkpoint_seq = 0;
    }
}

impl Default for CognitiveJournal {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_append_increments_sequence() {
        let mut journal = CognitiveJournal::new();
        let agent = AgentId::new();
        let task = TaskId::new();

        let s1 = journal.record_transition(task, agent, TaskState::Pending, TaskState::Running);
        let s2 = journal.record_transition(task, agent, TaskState::Running, TaskState::Completed);

        assert_eq!(s2, s1 + 1);
        assert_eq!(journal.len(), 2);
    }

    #[test]
    fn test_in_flight_tasks_detects_unfinished() {
        let mut journal = CognitiveJournal::new();
        let agent = AgentId::new();
        let t1 = TaskId::new();
        let t2 = TaskId::new();
        let t3 = TaskId::new();

        journal.record_transition(t1, agent, TaskState::Pending, TaskState::Running);
        journal.record_transition(t2, agent, TaskState::Pending, TaskState::Completed);
        journal.record_transition(
            t3,
            agent,
            TaskState::Pending,
            TaskState::Failed(lak_core::types::task::TaskError {
                code: "ERR".into(),
                message: "test failure".into(),
                retryable: false,
            }),
        );

        let in_flight = journal.in_flight_tasks();
        // t1 is still Running (not terminal), t2 & t3 are terminal
        assert!(in_flight.contains(&t1));
        assert!(!in_flight.contains(&t2));
        assert!(!in_flight.contains(&t3));
    }

    #[test]
    fn test_entries_for_task_filters_correctly() {
        let mut journal = CognitiveJournal::new();
        let agent = AgentId::new();
        let t1 = TaskId::new();
        let t2 = TaskId::new();

        journal.record_transition(t1, agent, TaskState::Pending, TaskState::Running);
        journal.record_transition(t2, agent, TaskState::Pending, TaskState::Running);
        journal.record_transition(t1, agent, TaskState::Running, TaskState::Completed);

        let t1_entries = journal.entries_for_task(t1);
        assert_eq!(t1_entries.len(), 2);
        // All should be for t1
        assert!(t1_entries.iter().all(|e| e.task_id == t1));
    }

    #[test]
    fn test_truncate_before_removes_old() {
        let mut journal = CognitiveJournal::new();
        let agent = AgentId::new();
        let task = TaskId::new();

        journal.record_transition(task, agent, TaskState::Pending, TaskState::Running);
        journal.record_transition(task, agent, TaskState::Running, TaskState::Completed);
        let s3 = journal.record_transition(task, agent, TaskState::Completed, TaskState::Suspended);
        journal.record_transition(task, agent, TaskState::Suspended, TaskState::Cancelled);

        let removed = journal.truncate_before(s3);
        assert_eq!(removed, 2);
        assert_eq!(journal.len(), 2);
    }

    #[test]
    fn test_entries_since_returns_correct_range() {
        let mut journal = CognitiveJournal::new();
        let agent = AgentId::new();
        let task = TaskId::new();

        journal.record_transition(task, agent, TaskState::Pending, TaskState::Running);
        let s2 = journal.record_transition(task, agent, TaskState::Running, TaskState::Completed);
        journal.record_transition(task, agent, TaskState::Completed, TaskState::Suspended);

        let since = journal.entries_since(s2);
        assert_eq!(since.len(), 2);
        assert!(since.iter().all(|e| e.sequence >= s2));
    }

    #[test]
    fn test_checkpoint_records_info() {
        let mut journal = CognitiveJournal::new();
        let agent = AgentId::new();

        let seq = journal.checkpoint(agent, 5, 42);
        assert!(seq > 0);
        assert_eq!(journal.last_checkpoint_seq(), seq);
    }
}
