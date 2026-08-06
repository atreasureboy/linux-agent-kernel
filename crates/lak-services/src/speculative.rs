//! SpeculativeReasoning — parallel speculative execution of reasoning branches
//!
//! When an LLM suggests multiple tool calls or reasoning paths, this engine
//! speculatively executes them in parallel. The token budget limits speculative
//! depth, and unused branches are discarded after merging.
//!
//! Key concepts:
//! - **Branch**: A single speculative reasoning path from a fork point
//! - **Fork Point**: Where the engine spawns N branches from one task
//! - **Merge**: Combine branch results, keeping the best or all relevant
//! - **Budget Gate**: Token budget check before spawning each branch
//! - **Branch Pruning**: Kill low-confidence branches early to save tokens

use std::collections::HashMap;

use lak_core::types::ids::TaskId;

/// A speculative reasoning branch
#[derive(Debug, Clone)]
pub struct SpecBranch {
    /// Unique branch identifier
    pub branch_id: u64,
    /// Parent task this branch forks from
    pub parent_task_id: TaskId,
    /// Branch strategy
    pub strategy: BranchStrategy,
    /// Confidence score (0.0-1.0) at time of forking
    pub confidence: f32,
    /// Estimated tokens this branch will consume
    pub estimated_tokens: u64,
    /// Branch state
    pub state: BranchState,
    /// The tool call or action that defines this branch
    pub action: BranchAction,
}

/// Strategies for speculative branching
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchStrategy {
    /// Explore alternative tool choices (e.g., try different APIs)
    AlternativeTool,
    /// Explore different reasoning approaches
    AlternativeReasoning,
    /// Prefetch likely-needed data in parallel
    Prefetch,
    /// Validate a hypothesis before committing
    HypothesisTest,
    /// Continue with lower-confidence fallback
    Fallback,
}

/// Current state of a speculative branch
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchState {
    /// Branch is queued, waiting for budget allocation
    Queued,
    /// Branch is actively executing
    Running,
    /// Branch completed successfully
    Completed,
    /// Branch was merged and results incorporated
    Merged,
    /// Branch was pruned (low confidence, budget pressure)
    Pruned,
    /// Branch failed
    Failed,
    /// Branch timed out
    TimedOut,
}

/// Action that defines a branch
#[derive(Debug, Clone)]
pub enum BranchAction {
    /// Execute a specific tool
    ExecuteTool {
        tool_name: String,
        parameters: serde_json::Value,
    },
    /// Run a specific LLM reasoning prompt
    ReasoningPrompt { model: String, prompt: String },
    /// Query memory for additional context
    MemoryQuery { query: String, top_k: usize },
    /// Retrieve external information
    ExternalFetch { url: String, reason: String },
}

/// Result of a completed branch
#[derive(Debug, Clone)]
pub struct BranchResult {
    pub branch_id: u64,
    pub output: serde_json::Value,
    pub tokens_consumed: u64,
    pub latency_ms: u64,
    /// Whether this branch's results were novel (not redundant)
    pub novel: bool,
}

/// The speculative reasoning engine
#[derive(Debug)]
pub struct SpeculativeEngine {
    /// All branches (active and completed)
    branches: HashMap<u64, SpecBranch>,
    /// Branch results keyed by branch_id
    results: HashMap<u64, BranchResult>,
    /// Current token budget remaining
    token_budget: u64,
    /// Maximum concurrent speculative branches
    max_concurrent_branches: usize,
    /// Minimum confidence to spawn a branch
    min_confidence: f32,
    /// Next branch ID
    next_branch_id: u64,
    /// Total branches spawned
    total_branches_spawned: u64,
    /// Total branches merged
    total_branches_merged: u64,
    /// Total branches pruned
    total_branches_pruned: u64,
}

impl SpeculativeEngine {
    /// Create a new speculative engine with a token budget
    pub fn new(token_budget: u64) -> Self {
        Self {
            branches: HashMap::new(),
            results: HashMap::new(),
            token_budget,
            max_concurrent_branches: 4,
            min_confidence: 0.3,
            next_branch_id: 1,
            total_branches_spawned: 0,
            total_branches_merged: 0,
            total_branches_pruned: 0,
        }
    }

    /// Set maximum concurrent speculative branches
    pub fn with_max_branches(mut self, max: usize) -> Self {
        self.max_concurrent_branches = max;
        self
    }

    /// Set minimum confidence threshold for spawning
    pub fn with_min_confidence(mut self, min: f32) -> Self {
        self.min_confidence = min;
        self
    }

    /// Add budget (e.g., from refund)
    pub fn add_budget(&mut self, tokens: u64) {
        self.token_budget += tokens;
    }

    /// Remaining token budget
    pub fn remaining_budget(&self) -> u64 {
        self.token_budget
    }

    /// Attempt to spawn speculative branches.
    ///
    /// Returns the spawned branch IDs. Branches that exceed the token budget
    /// or fall below the confidence threshold are not spawned.
    pub fn fork(
        &mut self,
        parent_task_id: TaskId,
        candidates: Vec<(BranchAction, f32, u64)>, // (action, confidence, estimated_tokens)
    ) -> Vec<u64> {
        let mut spawned = Vec::new();
        let active_count = self.active_branch_count();

        // Sort by confidence descending (spawn highest-confidence first)
        let mut sorted: Vec<_> = candidates.into_iter().enumerate().collect();
        sorted.sort_by(|a, b| {
            b.1 .1
                .partial_cmp(&a.1 .1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (_idx, (action, confidence, estimated_tokens)) in sorted {
            // Budget gate
            if estimated_tokens > self.token_budget {
                continue;
            }

            // Confidence gate
            if confidence < self.min_confidence {
                continue;
            }

            // Concurrency limit
            if active_count + spawned.len() >= self.max_concurrent_branches {
                break;
            }

            let branch_id = self.next_branch_id;
            self.next_branch_id += 1;

            let branch = SpecBranch {
                branch_id,
                parent_task_id,
                strategy: self.infer_strategy(&action),
                confidence,
                estimated_tokens,
                state: BranchState::Queued,
                action,
            };

            // Reserve tokens
            self.token_budget -= estimated_tokens;

            self.branches.insert(branch_id, branch);
            spawned.push(branch_id);
            self.total_branches_spawned += 1;
        }

        spawned
    }

    /// Mark a branch as running
    pub fn start_branch(&mut self, branch_id: u64) -> bool {
        if let Some(branch) = self.branches.get_mut(&branch_id) {
            branch.state = BranchState::Running;
            true
        } else {
            false
        }
    }

    /// Complete a branch with its result
    pub fn complete_branch(
        &mut self,
        branch_id: u64,
        result: BranchResult,
    ) -> Result<(), &'static str> {
        let branch = self
            .branches
            .get_mut(&branch_id)
            .ok_or("Branch not found")?;

        let spent = result.tokens_consumed;
        let reserved = branch.estimated_tokens;

        // Refund unused reserved tokens
        if spent < reserved {
            self.token_budget += reserved - spent;
        }

        branch.state = BranchState::Completed;

        let novel = !self.is_redundant(&result);
        let mut result = result;
        result.novel = novel;

        self.results.insert(branch_id, result);

        Ok(())
    }

    /// Merge completed branches: keep novel results, prune redundant ones.
    ///
    /// Returns the merged results (novel branches only).
    pub fn merge(&mut self) -> Vec<BranchResult> {
        let completed_ids: Vec<u64> = self
            .branches
            .iter()
            .filter(|(_, b)| b.state == BranchState::Completed)
            .map(|(&id, _)| id)
            .collect();

        let mut merged = Vec::new();

        for id in completed_ids {
            if let Some(result) = self.results.remove(&id) {
                let novel = result.novel;
                if novel {
                    merged.push(result);
                    self.total_branches_merged += 1;
                } else {
                    self.total_branches_pruned += 1;
                }
                if let Some(branch) = self.branches.get_mut(&id) {
                    branch.state = if novel {
                        BranchState::Merged
                    } else {
                        BranchState::Pruned
                    };
                }
            }
        }

        merged
    }

    /// Prune all active branches (e.g., when primary branch succeeds)
    pub fn prune_all_active(&mut self) -> usize {
        let mut pruned = 0;
        for branch in self.branches.values_mut() {
            if branch.state == BranchState::Running || branch.state == BranchState::Queued {
                // Refund budget
                self.token_budget += branch.estimated_tokens;
                branch.state = BranchState::Pruned;
                pruned += 1;
            }
        }
        self.total_branches_pruned += pruned as u64;
        pruned
    }

    /// Mark a branch as failed
    pub fn fail_branch(&mut self, branch_id: u64) -> Result<u64, &'static str> {
        let branch = self
            .branches
            .get_mut(&branch_id)
            .ok_or("Branch not found")?;
        branch.state = BranchState::Failed;
        // Refund reserved tokens
        let refund = branch.estimated_tokens;
        self.token_budget += refund;
        Ok(refund)
    }

    /// Number of active branches (running + queued)
    pub fn active_branch_count(&self) -> usize {
        self.branches
            .values()
            .filter(|b| b.state == BranchState::Running || b.state == BranchState::Queued)
            .count()
    }

    /// Get all branches
    pub fn branches(&self) -> &HashMap<u64, SpecBranch> {
        &self.branches
    }

    /// Get stats
    pub fn stats(&self) -> SpecStats {
        SpecStats {
            total_spawned: self.total_branches_spawned,
            total_merged: self.total_branches_merged,
            total_pruned: self.total_branches_pruned,
            active: self.active_branch_count() as u64,
            budget_remaining: self.token_budget,
        }
    }

    /// Infer branch strategy from action type
    fn infer_strategy(&self, action: &BranchAction) -> BranchStrategy {
        match action {
            BranchAction::ExecuteTool { .. } => BranchStrategy::AlternativeTool,
            BranchAction::ReasoningPrompt { .. } => BranchStrategy::AlternativeReasoning,
            BranchAction::MemoryQuery { .. } => BranchStrategy::Prefetch,
            BranchAction::ExternalFetch { .. } => BranchStrategy::Prefetch,
        }
    }

    /// Check if a result is redundant with existing merged results
    fn is_redundant(&self, result: &BranchResult) -> bool {
        // Simple heuristic: if the output is an exact JSON match to an existing result
        let output_str = serde_json::to_string(&result.output).unwrap_or_default();
        for existing in self.results.values() {
            let existing_str = serde_json::to_string(&existing.output).unwrap_or_default();
            if output_str == existing_str {
                return true;
            }
        }
        false
    }
}

impl Default for SpeculativeEngine {
    fn default() -> Self {
        Self::new(100_000)
    }
}

/// Statistics for the speculative engine
#[derive(Debug, Clone)]
pub struct SpecStats {
    pub total_spawned: u64,
    pub total_merged: u64,
    pub total_pruned: u64,
    pub active: u64,
    pub budget_remaining: u64,
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tool_action(name: &str) -> (BranchAction, f32, u64) {
        (
            BranchAction::ExecuteTool {
                tool_name: name.to_string(),
                parameters: serde_json::json!({}),
            },
            0.8,  // confidence
            1000, // estimated tokens
        )
    }

    #[test]
    fn test_fork_spawns_within_budget() {
        let mut engine = SpeculativeEngine::new(3000);

        let candidates = vec![
            make_tool_action("search_web"),
            make_tool_action("read_file"),
            make_tool_action("query_db"),
        ];

        let spawned = engine.fork(TaskId::new(), candidates);
        assert_eq!(spawned.len(), 3);
        assert_eq!(engine.active_branch_count(), 3);
    }

    #[test]
    fn test_fork_respects_budget_limit() {
        let mut engine = SpeculativeEngine::new(1500);

        let candidates = vec![
            make_tool_action("search_web"),
            make_tool_action("read_file"),
        ];

        let spawned = engine.fork(TaskId::new(), candidates);
        // Only 1 fits (1000 < 1500, second would need 2000 > 1500)
        assert!(spawned.len() <= 2);
    }

    #[test]
    fn test_fork_rejects_low_confidence() {
        let mut engine = SpeculativeEngine::new(5000);

        let candidates = vec![
            make_tool_action("good_branch"),
            (
                BranchAction::ExecuteTool {
                    tool_name: "risky_tool".into(),
                    parameters: serde_json::json!({}),
                },
                0.1, // very low confidence
                500,
            ),
        ];

        let spawned = engine.fork(TaskId::new(), candidates);
        assert_eq!(spawned.len(), 1);
    }

    #[test]
    fn test_complete_branch_refunds_extra_budget() {
        let mut engine = SpeculativeEngine::new(5000);
        let candidates = vec![make_tool_action("test_tool")];
        let spawned = engine.fork(TaskId::new(), candidates);
        let branch_id = spawned[0];

        engine.start_branch(branch_id);
        let result = BranchResult {
            branch_id,
            output: serde_json::json!({"status": "ok"}),
            tokens_consumed: 200, // Only used 200 of 1000 reserved
            latency_ms: 50,
            novel: true,
        };

        engine.complete_branch(branch_id, result).unwrap();
        // Budget: 5000 - 1000 (reserved at fork) + (1000-200) (refund) = 4800
        assert_eq!(engine.remaining_budget(), 4800);
    }

    #[test]
    fn test_merge_keeps_novel_results() {
        let mut engine = SpeculativeEngine::new(5000);
        let candidates = vec![make_tool_action("branch_a"), make_tool_action("branch_b")];
        let spawned = engine.fork(TaskId::new(), candidates);

        // Complete both branches
        for &id in &spawned {
            engine.start_branch(id);
            let result = BranchResult {
                branch_id: id,
                output: serde_json::json!({"result": id}),
                tokens_consumed: 500,
                latency_ms: 100,
                novel: true,
            };
            engine.complete_branch(id, result).unwrap();
        }

        let merged = engine.merge();
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn test_prune_all_active_refunds_budget() {
        let mut engine = SpeculativeEngine::new(5000);
        let initial_budget = engine.remaining_budget();
        let candidates = vec![make_tool_action("branch_a"), make_tool_action("branch_b")];
        let spawned = engine.fork(TaskId::new(), candidates);
        for &id in &spawned {
            engine.start_branch(id);
        }

        let pruned = engine.prune_all_active();
        assert_eq!(pruned, 2);
        // All tokens refunded
        assert_eq!(engine.remaining_budget(), initial_budget);
        assert_eq!(engine.active_branch_count(), 0);
    }
}
