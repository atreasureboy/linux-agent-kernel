//! CognitiveScheduler — COI-based task scheduling
//!
//! The scheduler is the "dispatcher" of the Agent Kernel. It selects which
//! cognitive task runs next based on Cognitive Opportunity Index (COI).
//!
//! Unlike a CPU scheduler which gives equal time slices, the cognitive scheduler
//! allocates ThinkingQuanta based on:
//! - Task priority (urgency + importance + context affinity)
//! - Agent COI score
//! - Token budget availability
//! - Resource constraints

use chrono::Utc;
use lak_core::types::ids::TaskId;
use lak_core::types::task::{CognitivePriority, CognitiveTask};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A unit of cognitive execution — the cognitive equivalent of a CPU time quantum.
#[derive(Debug, Clone)]
pub struct ThinkingQuantum {
    /// Maximum tokens the task may consume in this quantum
    pub max_tokens: u64,
    /// Maximum tool calls allowed
    pub max_tool_calls: u32,
    /// Maximum wall-clock time for this quantum
    pub max_wall_clock_ms: u64,
}

impl Default for ThinkingQuantum {
    fn default() -> Self {
        Self {
            max_tokens: 4096,
            max_tool_calls: 5,
            max_wall_clock_ms: 30_000,
        }
    }
}

impl ThinkingQuantum {
    /// Create a quantum scaled by priority
    pub fn for_priority(priority: &CognitivePriority) -> Self {
        let score = priority.score();
        Self {
            max_tokens: (4096.0 * (1.0 + score / 100.0)) as u64,
            max_tool_calls: (5.0 * (1.0 + score / 100.0)) as u32,
            max_wall_clock_ms: (30_000.0 * (1.0 + score / 100.0)) as u64,
        }
    }

    /// Create a minimal quantum for simple tasks
    pub fn minimal() -> Self {
        Self {
            max_tokens: 512,
            max_tool_calls: 1,
            max_wall_clock_ms: 5_000,
        }
    }
}

/// A scheduled task entry with its scheduling metadata
#[derive(Debug, Clone)]
struct ScheduledTask {
    task: CognitiveTask,
    /// Time the task entered the queue
    enqueued_at: chrono::DateTime<Utc>,
    /// Number of times this task has been preempted
    preemptions: u32,
    /// Agent's COI score when this task was submitted
    agent_coi: f32,
}

impl ScheduledTask {
    /// Compute the scheduling score for ordering in the ready queue.
    ///
    /// Higher score = higher priority. Formula:
    ///   score = priority * (1 + aging_factor) * (1 + agent_coi_boost)
    ///
    /// Where aging_factor increases with time spent waiting,
    /// preventing starvation of low-priority tasks.
    fn scheduling_score(&self) -> f64 {
        let priority = self.task.priority.score();
        let wait_secs = (Utc::now() - self.enqueued_at).num_seconds() as f64;
        let aging = 1.0 + (wait_secs / 60.0).min(5.0); // Max 5x aging boost after 5 minutes
        let coi_boost = 1.0 + self.agent_coi as f64 * 0.1;
        priority * aging * coi_boost
    }
}

impl PartialEq for ScheduledTask {
    fn eq(&self, other: &Self) -> bool {
        self.task.task_id == other.task.task_id
    }
}

impl Eq for ScheduledTask {}

impl Ord for ScheduledTask {
    fn cmp(&self, other: &Self) -> Ordering {
        self.scheduling_score()
            .partial_cmp(&other.scheduling_score())
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for ScheduledTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// The Cognitive Scheduler — dispatches tasks to agents based on COI.
///
/// Uses a max-heap (BinaryHeap) ordered by scheduling score.
/// Tasks with higher priority + aging + agent COI run first.
pub struct CognitiveScheduler {
    /// Ready queue (max-heap by scheduling score)
    ready_queue: BinaryHeap<ScheduledTask>,
    /// Blocked tasks waiting for I/O (tool results, intents)
    blocked_queue: Vec<ScheduledTask>,
    /// Maximum concurrent tasks across all agents
    max_concurrent: usize,
    /// Currently running task count
    running_count: usize,
    /// Total tasks ever submitted
    total_submitted: u64,
    /// Total tasks completed
    total_completed: u64,
    /// Default quantum for non-priority tasks
    default_quantum: ThinkingQuantum,
}

impl CognitiveScheduler {
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            ready_queue: BinaryHeap::new(),
            blocked_queue: Vec::new(),
            max_concurrent,
            running_count: 0,
            total_submitted: 0,
            total_completed: 0,
            default_quantum: ThinkingQuantum::default(),
        }
    }

    /// Submit a task to the scheduler
    pub fn submit(&mut self, task: CognitiveTask, agent_coi: f32) {
        let scheduled = ScheduledTask {
            task,
            enqueued_at: Utc::now(),
            preemptions: 0,
            agent_coi,
        };
        self.total_submitted += 1;
        self.ready_queue.push(scheduled);
    }

    /// Get the next task to execute (highest scheduling score)
    pub fn schedule_next(&mut self) -> Option<(CognitiveTask, ThinkingQuantum)> {
        if self.running_count >= self.max_concurrent {
            return None;
        }

        let scheduled = self.ready_queue.pop()?;
        let mut quantum = ThinkingQuantum::for_priority(&scheduled.task.priority);
        // Apply the configured default as a minimum floor so that even
        // the lowest-priority tasks get a reasonable slice.
        let dq = &self.default_quantum;
        quantum.max_tokens = quantum.max_tokens.max(dq.max_tokens);
        quantum.max_tool_calls = quantum.max_tool_calls.max(dq.max_tool_calls);
        quantum.max_wall_clock_ms = quantum.max_wall_clock_ms.max(dq.max_wall_clock_ms);
        self.running_count += 1;

        Some((scheduled.task, quantum))
    }

    /// Mark a task as completed
    pub fn complete(&mut self, _task_id: TaskId) {
        self.running_count = self.running_count.saturating_sub(1);
        self.total_completed += 1;
    }

    /// Move a running task to blocked state (waiting for tool/intent)
    pub fn block(&mut self, task: CognitiveTask, reason: impl Into<String>, agent_coi: f32) {
        self.running_count = self.running_count.saturating_sub(1);
        let mut scheduled = ScheduledTask {
            task,
            enqueued_at: Utc::now(),
            preemptions: 0,
            agent_coi,
        };
        // Keep the original enqueue time? Blocked tasks restart aging once
        // unblocked, which is intentional: they were not starving before.
        scheduled.task.state = lak_core::types::task::TaskState::Blocked(reason.into());
        self.blocked_queue.push(scheduled);
    }

    /// Unblock a task and return it to the ready queue
    pub fn unblock(&mut self, task_id: TaskId, agent_coi: f32) -> Option<CognitiveTask> {
        let pos = self
            .blocked_queue
            .iter()
            .position(|s| s.task.task_id == task_id)?;
        let scheduled = self.blocked_queue.remove(pos);
        let mut task = scheduled.task;
        task.state = lak_core::types::task::TaskState::Ready;
        self.ready_queue.push(ScheduledTask {
            enqueued_at: Utc::now(),
            preemptions: scheduled.preemptions,
            agent_coi,
            task: task.clone(),
        });
        Some(task)
    }

    /// Preempt a running task (cooperative preemption — task yields voluntarily)
    pub fn preempt(&mut self, task: CognitiveTask, agent_coi: f32) {
        self.running_count = self.running_count.saturating_sub(1);
        let scheduled = ScheduledTask {
            task,
            enqueued_at: Utc::now(),
            preemptions: 1,
            agent_coi,
        };
        self.ready_queue.push(scheduled);
    }

    /// Cancel a task from the queue. Returns true if the task was found
    /// in either the ready queue or the blocked queue.
    pub fn cancel(&mut self, task_id: TaskId) -> bool {
        // BinaryHeap has no remove-by-predicate; drain and rebuild.
        let mut kept = Vec::new();
        let mut found_ready = false;
        while let Some(scheduled) = self.ready_queue.pop() {
            if scheduled.task.task_id == task_id {
                found_ready = true;
            } else {
                kept.push(scheduled);
            }
        }
        self.ready_queue = BinaryHeap::from(kept);

        let blocked_before = self.blocked_queue.len();
        self.blocked_queue.retain(|s| s.task.task_id != task_id);
        let found_blocked = self.blocked_queue.len() < blocked_before;

        found_ready || found_blocked
    }

    /// Check if the scheduler has available slots
    pub fn has_capacity(&self) -> bool {
        self.running_count < self.max_concurrent
    }

    /// Number of tasks in the ready queue
    pub fn ready_count(&self) -> usize {
        self.ready_queue.len()
    }

    /// Number of tasks in the blocked queue
    pub fn blocked_count(&self) -> usize {
        self.blocked_queue.len()
    }

    /// Number of currently running tasks
    pub fn running_count(&self) -> usize {
        self.running_count
    }

    /// Scheduler load (0.0 - 1.0) — higher means more contention
    pub fn load(&self) -> f64 {
        if self.max_concurrent == 0 {
            return 1.0;
        }
        (self.ready_queue.len() + self.running_count) as f64 / self.max_concurrent as f64
    }

    /// Set the default thinking quantum
    pub fn set_default_quantum(&mut self, quantum: ThinkingQuantum) {
        self.default_quantum = quantum;
    }

    /// Get pending task count (ready + running)
    pub fn pending_count(&self) -> usize {
        self.ready_queue.len() + self.running_count
    }
}

impl Default for CognitiveScheduler {
    fn default() -> Self {
        Self::new(10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lak_core::types::ids::AgentId;
    use lak_core::types::task::{TaskContent, TaskState, TaskStats, TaskType};

    fn make_task(id: u8, priority: CognitivePriority) -> CognitiveTask {
        CognitiveTask {
            task_id: TaskId::new(),
            agent_id: AgentId::new(),
            task_type: TaskType::Reasoning,
            priority,
            state: TaskState::Pending,
            content: TaskContent {
                natural_language: format!("task {id}"),
                structured_schema: None,
                memory_references: vec![],
            },
            deadline: None,
            dependencies: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            metadata: std::collections::HashMap::new(),
            stats: TaskStats::default(),
        }
    }

    #[test]
    fn test_scheduler_orders_by_priority() {
        let mut sched = CognitiveScheduler::new(5);
        let low = make_task(1, CognitivePriority::low());
        let high = make_task(2, CognitivePriority::high());
        let crit = make_task(3, CognitivePriority::critical());

        sched.submit(low, 0.5);
        sched.submit(high, 0.5);
        sched.submit(crit, 0.5);

        let (first, _) = sched.schedule_next().unwrap();
        let (second, _) = sched.schedule_next().unwrap();
        let (third, _) = sched.schedule_next().unwrap();

        // Higher priority tasks should be scheduled first
        assert!(
            first.priority.score() >= second.priority.score()
                && second.priority.score() >= third.priority.score()
        );
    }

    #[test]
    fn test_scheduler_respects_concurrency_limit() {
        let mut sched = CognitiveScheduler::new(2);
        sched.submit(make_task(1, CognitivePriority::normal()), 0.5);
        sched.submit(make_task(2, CognitivePriority::normal()), 0.5);
        sched.submit(make_task(3, CognitivePriority::normal()), 0.5);

        assert!(sched.schedule_next().is_some());
        assert!(sched.schedule_next().is_some());
        assert!(sched.schedule_next().is_none()); // At capacity
    }

    #[test]
    fn test_complete_frees_slot() {
        let mut sched = CognitiveScheduler::new(1);
        sched.submit(make_task(1, CognitivePriority::normal()), 0.5);
        let (task, _) = sched.schedule_next().unwrap();
        let tid = task.task_id;
        sched.complete(tid);
        assert!(sched.schedule_next().is_none()); // No more tasks
        assert_eq!(sched.running_count(), 0);
    }

    #[test]
    fn test_cancel_removes_task() {
        let mut sched = CognitiveScheduler::new(2);
        let task = make_task(42, CognitivePriority::normal());
        let tid = task.task_id;
        sched.submit(task, 0.5);
        assert!(sched.cancel(tid));
        assert_eq!(sched.ready_count(), 0);
    }
}
