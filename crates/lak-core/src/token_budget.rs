//! TokenBudget — controls LLM token consumption
//!
//! Each agent has a TokenBudget that enforces:
//! - Hard limit: absolute maximum tokens per session (prevents runaway costs)
//! - Soft limit: warning threshold (triggers compression/eviction)
//! - Priority reserve: tokens reserved for high-priority tasks
//! - Rolling window: track recent consumption for rate limiting

use std::time::Duration;

/// Token budget configuration and state
#[derive(Debug, Clone)]
pub struct TokenBudget {
    /// Absolute token limit per billing period
    pub hard_limit: u64,
    /// Soft warning threshold (triggers optimization)
    pub soft_limit: u64,
    /// Tokens reserved exclusively for critical-priority tasks
    pub priority_reserve: u64,
    /// Tokens consumed this period
    pub consumed: u64,
    /// Rolling window tracking for rate limiting
    window: ConsumptionWindow,
}

/// Tracks recent consumption within a sliding time window
#[derive(Debug, Clone)]
struct ConsumptionWindow {
    /// Maximum tokens allowed per window
    max_per_window: u64,
    /// Duration of the window
    window_duration: Duration,
    /// Recent consumption entries (timestamp, tokens)
    entries: Vec<(std::time::Instant, u64)>,
}

/// Result of a budget allocation check
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BudgetAllocation {
    /// Requested tokens granted
    Granted(u64),
    /// Reduced to fit within limits
    Reduced(u64, String), // granted amount, reason
    /// Full denial
    Denied(String),
}

impl TokenBudget {
    /// Create a new token budget
    pub fn new(hard_limit: u64, soft_limit: u64, priority_reserve: u64) -> Self {
        Self {
            hard_limit,
            soft_limit,
            priority_reserve,
            consumed: 0,
            window: ConsumptionWindow {
                // Per-minute rate limit; default to the full budget (rate limit
                // enforced primarily through the total hard_limit rather than
                // per-window caps). Callers can tune this downward later.
                max_per_window: hard_limit,
                window_duration: Duration::from_secs(60),
                entries: Vec::new(),
            },
        }
    }

    /// Default budget for development/experimental agents
    pub fn developer_budget() -> Self {
        Self::new(1_000_000, 800_000, 50_000)
    }

    /// Tight budget for production agents
    pub fn production_budget() -> Self {
        Self::new(100_000, 80_000, 10_000)
    }

    /// Request a token allocation. Priority score determines access to reserve.
    ///
    /// Returns the granted amount (may be zero) and whether access was denied.
    pub fn check_allocation(&mut self, requested: u64, priority_score: f64) -> BudgetAllocation {
        // 1. Expire old window entries
        let now = std::time::Instant::now();
        self.window
            .entries
            .retain(|(t, _)| now.duration_since(*t) < self.window.window_duration);

        // 2. Check window rate limit
        let window_consumed: u64 = self.window.entries.iter().map(|(_, t)| t).sum();
        let window_available = self.window.max_per_window.saturating_sub(window_consumed);
        let effective_request = requested.min(window_available);

        if effective_request == 0 {
            return BudgetAllocation::Denied("Rate limit: window exhausted".into());
        }

        // 3. Check total budget
        let remaining = self.hard_limit.saturating_sub(self.consumed);

        if remaining == 0 {
            return BudgetAllocation::Denied("Hard limit reached".into());
        }

        // 4. Priority-based allocation.
        //    The priority reserve is carved out of the hard limit for critical
        //    tasks only: non-priority tasks see `remaining - reserve`, while
        //    priority tasks may consume the full `remaining` amount.
        let is_priority = priority_score >= 80.0;
        let effective_remaining = if is_priority {
            remaining
        } else {
            remaining.saturating_sub(self.priority_reserve)
        };

        if effective_remaining == 0 {
            return BudgetAllocation::Denied(format!(
                "Budget exhausted (consumed: {}, limit: {})",
                self.consumed, self.hard_limit
            ));
        }

        // 5. Determine actual grant
        let grant = effective_request.min(effective_remaining);

        // 6. Record consumption
        self.consumed += grant;
        self.window.entries.push((now, grant));

        if self.consumed > self.soft_limit {
            BudgetAllocation::Reduced(
                grant,
                format!(
                    "soft limit exceeded ({}/{})",
                    self.consumed, self.soft_limit
                ),
            )
        } else if grant < requested {
            BudgetAllocation::Reduced(grant, "Request reduced to fit remaining budget".into())
        } else {
            BudgetAllocation::Granted(grant)
        }
    }

    /// Record actual token consumption (adjusts from estimated allocation)
    pub fn record_consumption(&mut self, tokens: u64) {
        self.consumed += tokens;
        self.window
            .entries
            .push((std::time::Instant::now(), tokens));
    }

    /// Refund unused tokens (when estimate was too high)
    pub fn refund(&mut self, tokens: u64) {
        self.consumed = self.consumed.saturating_sub(tokens);
    }

    /// Check if budget is near exhaustion
    pub fn is_exhausted(&self) -> bool {
        self.consumed >= self.hard_limit
    }

    /// Check if we're above soft limit
    pub fn is_above_soft_limit(&self) -> bool {
        self.consumed >= self.soft_limit
    }

    /// Available tokens (excluding priority reserve)
    pub fn available(&self) -> u64 {
        self.hard_limit
            .saturating_sub(self.consumed)
            .saturating_sub(self.priority_reserve)
    }

    /// Utilization percentage (0.0 - 1.0)
    pub fn utilization(&self) -> f64 {
        if self.hard_limit == 0 {
            return 1.0;
        }
        self.consumed as f64 / self.hard_limit as f64
    }

    /// Reset the budget period
    pub fn reset(&mut self) {
        self.consumed = 0;
        self.window.entries.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_allocation() {
        let mut budget = TokenBudget::new(1000, 800, 100);
        let result = budget.check_allocation(50, 50.0);
        assert!(matches!(result, BudgetAllocation::Granted(50)));
    }

    #[test]
    fn test_hard_limit_denial() {
        let mut budget = TokenBudget::new(100, 80, 10);
        budget.record_consumption(100);
        let result = budget.check_allocation(10, 50.0);
        assert!(matches!(result, BudgetAllocation::Denied(_)));
    }

    #[test]
    fn test_priority_accesses_reserve() {
        let mut budget = TokenBudget::new(100, 80, 10);
        budget.record_consumption(95); // Only 5 left + 10 reserve with priority
        let result = budget.check_allocation(10, 90.0); // High priority
        assert!(matches!(
            result,
            BudgetAllocation::Granted(_) | BudgetAllocation::Reduced(_, _)
        ));
    }

    #[test]
    fn test_soft_limit_warning() {
        let mut budget = TokenBudget::new(1000, 800, 100);
        budget.record_consumption(850); // Above soft limit
        let result = budget.check_allocation(10, 50.0);
        assert!(matches!(result, BudgetAllocation::Reduced(_, _)));
    }

    #[test]
    fn test_refund() {
        let mut budget = TokenBudget::new(1000, 800, 100);
        budget.record_consumption(100);
        budget.refund(50);
        assert_eq!(budget.consumed, 50);
    }
}
