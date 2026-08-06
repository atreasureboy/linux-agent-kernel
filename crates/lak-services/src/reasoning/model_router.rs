//! ModelRouter — selects the best LLM backend for a task
//!
//! Routes cognitive tasks based on:
//! - Task complexity and type (cheap model for simple, expensive for complex)
//! - Cost optimization (prefer local/free models when suitable)
//! - Model availability (health check)
//! - Task priority (reserve best models for critical tasks)

use std::sync::Arc;

use lak_core::types::task::{CognitiveTask, TaskType};
use lak_tal::llm::LLMDriver;

/// Routes cognitive tasks to the appropriate LLM backend.
pub struct ModelRouter;

impl ModelRouter {
    pub fn new() -> Self {
        Self
    }

    /// Select the best driver for a given task.
    ///
    /// Strategy:
    /// 1. Simple tasks (MemoryRetrieval, IdleReflection) → free/cheap models
    /// 2. Complex tasks (Reasoning, ToolExecution) → powerful models
    /// 3. Critical priority → best available model regardless of cost
    /// 4. Fallback → first available driver
    pub fn select_driver<'a>(
        &self,
        drivers: &'a [Arc<dyn LLMDriver>],
        task: &CognitiveTask,
    ) -> Option<&'a Arc<dyn LLMDriver>> {
        if drivers.is_empty() {
            return None;
        }

        // Create weighted candidate list based on task requirements
        let mut candidates: Vec<(&Arc<dyn LLMDriver>, f64)> = drivers
            .iter()
            .map(|d| {
                let score = self.score_driver_for_task(d, task);
                (d, score)
            })
            .collect();

        // Sort by score descending
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top candidate
        candidates.first().map(|(d, _)| *d)
    }

    /// Score a driver's suitability for a task (higher = better fit).
    fn score_driver_for_task(&self, driver: &Arc<dyn LLMDriver>, task: &CognitiveTask) -> f64 {
        let name = driver.name().to_lowercase();
        let mut score = 50.0; // Base score

        // Cost preference: free models get bonus for low-priority tasks
        let cost_per_1k = driver.cost_per_1k_tokens(true);
        if cost_per_1k == 0.0 {
            // Free/local models
            score += 30.0;
            // But penalize for very complex tasks that benefit from cloud models
            if matches!(task.task_type, TaskType::Reasoning)
                && task.content.natural_language.len() > 500
            {
                score -= 15.0;
            }
        } else if cost_per_1k < 0.001 {
            // Very cheap cloud models
            score += 20.0;
        } else if cost_per_1k > 0.01 {
            // Expensive models — only for critical/high tasks
            if task.priority.score() > 60.0 {
                score += 40.0;
            } else {
                score -= 20.0;
            }
        }

        // Model capability scoring
        match name.as_str() {
            "openai" => {
                if matches!(
                    task.task_type,
                    TaskType::Reasoning | TaskType::ToolExecution
                ) {
                    score += 20.0; // OpenAI is good at reasoning
                }
            }
            "anthropic" => {
                if task.content.natural_language.len() > 1000 {
                    score += 25.0; // Anthropic excels at long context
                }
                if matches!(task.task_type, TaskType::Reasoning) {
                    score += 15.0; // Strong reasoning
                }
            }
            "ollama" => {
                // Ollama is good for quick, simple tasks
                if matches!(
                    task.task_type,
                    TaskType::MemoryRetrieval | TaskType::IdleReflection
                ) {
                    score += 35.0;
                }
                // Local model for privacy-sensitive tasks
                if task.content.natural_language.contains("confidential")
                    || task.content.natural_language.contains("private")
                {
                    score += 40.0;
                }
            }
            _ => {}
        }

        // Task type affinity
        match task.task_type {
            TaskType::Reasoning => {
                // Complex reasoning → prefer cloud models with strong capability
                if name == "anthropic" || name == "openai" {
                    score += 10.0;
                }
            }
            TaskType::ToolExecution => {
                // Tool execution needs reliability
                if cost_per_1k > 0.0 {
                    score += 5.0;
                }
            }
            TaskType::MemoryRetrieval | TaskType::IdleReflection => {
                // Simple tasks → prefer cheap/free
                if cost_per_1k == 0.0 {
                    score += 25.0;
                }
            }
            TaskType::SystemTask => {
                // System tasks can use any model
                score += 5.0;
            }
            TaskType::IntentProcessing => {
                // Intent processing needs speed
                score -= cost_per_1k * 1000.0; // Cost penalty
            }
        }

        // Priority adjustment
        let priority = task.priority.score();
        if priority > 80.0 {
            // Critical tasks → best model regardless of cost
            score += (priority - 80.0) * 2.0;
        }

        score
    }

    /// Get the configured model name from a driver for logging purposes.
    pub fn model_info(&self, driver: &dyn LLMDriver) -> String {
        format!(
            "{} (cost: ${:.6}/1k input, ${:.6}/1k output)",
            driver.name(),
            driver.cost_per_1k_tokens(true),
            driver.cost_per_1k_tokens(false)
        )
    }
}

impl Default for ModelRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use futures::stream::BoxStream;
    use lak_core::types::task::CognitivePriority;
    use lak_tal::llm::{LLMDriver, LLMError, LLMRequest, LLMStreamEvent};

    #[derive(Debug)]
    struct FakeDriver {
        name: String,
        cost: f64,
    }

    #[async_trait]
    impl LLMDriver for FakeDriver {
        fn name(&self) -> &str {
            &self.name
        }
        async fn generate_stream(
            &self,
            _req: LLMRequest,
        ) -> Result<BoxStream<'static, Result<LLMStreamEvent, LLMError>>, LLMError> {
            Err(LLMError::UnsupportedModel("fake".into()))
        }
        async fn count_tokens(&self, text: &str) -> Result<usize, LLMError> {
            Ok(text.len() / 4)
        }
        async fn health_check(&self) -> Result<bool, LLMError> {
            Ok(true)
        }
        fn cost_per_1k_tokens(&self, _is_input: bool) -> f64 {
            self.cost
        }
    }

    #[test]
    fn test_router_prefers_free_for_simple_tasks() {
        let free: Arc<dyn LLMDriver> = Arc::new(FakeDriver {
            name: "ollama".into(),
            cost: 0.0,
        });
        let paid: Arc<dyn LLMDriver> = Arc::new(FakeDriver {
            name: "openai".into(),
            cost: 0.01,
        });
        let drivers: Vec<Arc<dyn LLMDriver>> = vec![free, paid];
        let router = ModelRouter::new();

        let task = CognitiveTask {
            task_id: lak_core::types::ids::TaskId::new(),
            agent_id: lak_core::types::ids::AgentId::new(),
            task_type: TaskType::IdleReflection,
            priority: CognitivePriority::low(),
            state: lak_core::types::task::TaskState::Pending,
            content: lak_core::types::task::TaskContent {
                natural_language: "reflect on today".into(),
                structured_schema: None,
                memory_references: vec![],
            },
            deadline: None,
            dependencies: vec![],
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            metadata: std::collections::HashMap::new(),
            stats: lak_core::types::task::TaskStats::default(),
        };

        let selected = router.select_driver(&drivers, &task).unwrap();
        assert_eq!(selected.name(), "ollama"); // Free model preferred
    }

    #[test]
    fn test_router_prefers_cloud_for_complex_reasoning() {
        let free: Arc<dyn LLMDriver> = Arc::new(FakeDriver {
            name: "ollama".into(),
            cost: 0.0,
        });
        let paid: Arc<dyn LLMDriver> = Arc::new(FakeDriver {
            name: "anthropic".into(),
            cost: 0.015,
        });
        let drivers: Vec<Arc<dyn LLMDriver>> = vec![free, paid];
        let router = ModelRouter::new();

        let task = CognitiveTask {
            task_id: lak_core::types::ids::TaskId::new(),
            agent_id: lak_core::types::ids::AgentId::new(),
            task_type: TaskType::Reasoning,
            priority: CognitivePriority::critical(),
            state: lak_core::types::task::TaskState::Pending,
            content: lak_core::types::task::TaskContent {
                natural_language: "Analyze the security implications of this 2000-line audit log"
                    .into(),
                structured_schema: None,
                memory_references: vec![],
            },
            deadline: None,
            dependencies: vec![],
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            metadata: std::collections::HashMap::new(),
            stats: lak_core::types::task::TaskStats::default(),
        };

        let selected = router.select_driver(&drivers, &task).unwrap();
        // Critical + complex reasoning → cloud model preferred
        assert!(selected.cost_per_1k_tokens(true) > 0.0);
    }
}
