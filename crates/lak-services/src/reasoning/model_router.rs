//! ModelRouter — selects the best LLM backend for a task

/// Routes cognitive tasks to the appropriate LLM backend
/// based on task complexity, cost, and availability.
pub struct ModelRouter {
    // MVP: simple first-match routing
}

impl ModelRouter {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for ModelRouter {
    fn default() -> Self {
        Self::new()
    }
}
