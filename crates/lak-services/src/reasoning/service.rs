//! ReasoningService — orchestrates LLM calls for cognitive tasks

use std::sync::Arc;

use lak_tal::llm::LLMDriver;

/// Manages LLM invocation for cognitive tasks
pub struct ReasoningService {
    drivers: Vec<Arc<dyn LLMDriver>>,
}

impl ReasoningService {
    pub fn new() -> Self {
        Self { drivers: vec![] }
    }

    pub fn add_driver(&mut self, driver: Arc<dyn LLMDriver>) {
        self.drivers.push(driver);
    }

    pub fn driver_count(&self) -> usize {
        self.drivers.len()
    }
}

impl Default for ReasoningService {
    fn default() -> Self {
        Self::new()
    }
}
