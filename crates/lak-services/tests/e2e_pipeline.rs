//! End-to-end test: create agent → submit task → execute → verify completion
//!
//! This test verifies the full cognitive execution pipeline end-to-end
//! by using a fake LLM driver that returns a canned response. It exercises:
//! - Agent lifecycle
//! - Task submission + scheduling
//! - The background execution loop (kernel.start)
//! - Pipeline execution (with fake LLM)
//! - Task state transitions (Pending → Running → Completed)
//! - Agent stats update (tokens, tasks, COI)

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::{self, BoxStream};

use lak_core::traits::AgentKernel;
use lak_core::types::agent::AgentSpec;
use lak_core::types::task::{CognitivePriority, CognitiveTask, TaskContent, TaskState, TaskType};
use lak_services::kernel::KernelService;
use lak_tal::llm::{LLMDriver, LLMError, LLMRequest, LLMResponse, LLMStreamEvent};

/// A fake LLM driver that returns a single canned text response plus Done.
#[derive(Debug)]
struct FakeCannedDriver {
    name: String,
    text: String,
    tokens: u64,
}

#[async_trait]
impl LLMDriver for FakeCannedDriver {
    fn name(&self) -> &str {
        &self.name
    }

    async fn generate_stream(
        &self,
        _request: LLMRequest,
    ) -> Result<BoxStream<'static, Result<LLMStreamEvent, LLMError>>, LLMError> {
        let text = self.text.clone();
        let tokens = self.tokens;
        let events = vec![
            Ok(LLMStreamEvent::Token(text)),
            Ok(LLMStreamEvent::Done(LLMResponse {
                content: String::new(),
                tool_calls: vec![],
                tokens_used: tokens,
                finish_reason: "stop".into(),
            })),
        ];
        Ok(Box::pin(stream::iter(events)))
    }

    async fn count_tokens(&self, text: &str) -> Result<usize, LLMError> {
        Ok(text.len() / 4)
    }

    async fn health_check(&self) -> Result<bool, LLMError> {
        Ok(true)
    }

    fn cost_per_1k_tokens(&self, _is_input: bool) -> f64 {
        0.0
    }
}

fn make_spec(name: &str) -> AgentSpec {
    AgentSpec {
        name: name.into(),
        ..Default::default()
    }
}

fn make_task(agent_id: lak_core::types::ids::AgentId, desc: &str) -> CognitiveTask {
    CognitiveTask {
        task_id: lak_core::types::ids::TaskId::new(),
        agent_id,
        task_type: TaskType::Reasoning,
        priority: CognitivePriority::normal(),
        state: TaskState::Pending,
        content: TaskContent {
            natural_language: desc.to_string(),
            structured_schema: None,
            memory_references: vec![],
        },
        deadline: None,
        dependencies: vec![],
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        metadata: std::collections::HashMap::new(),
        stats: Default::default(),
    }
}

#[tokio::test]
async fn test_e2e_task_completes_with_fake_driver() {
    let kernel = Arc::new(KernelService::new());

    // Register a fake LLM driver that responds instantly
    kernel
        .add_driver(Arc::new(FakeCannedDriver {
            name: "fake-test".into(),
            text: "42 is the answer".into(),
            tokens: 7,
        }))
        .await;

    // Start the execution loop
    kernel.start();

    // Create an agent and resume it (so it's Idle → can accept tasks)
    let agent_id = kernel.create_agent(make_spec("e2e-agent")).await.unwrap();
    kernel.resume_agent(agent_id).await.unwrap();

    // Submit a task
    let task = make_task(agent_id, "What is the answer to everything?");
    let task_id = task.task_id;
    kernel.submit_task(task).await.unwrap();

    // Poll until the task completes (or times out after 5 seconds)
    let mut attempts = 0;
    let result = loop {
        attempts += 1;
        assert!(attempts <= 50, "Task did not complete within 5 seconds");
        tokio::time::sleep(Duration::from_millis(100)).await;
        let t = kernel.get_task(task_id).await.unwrap();
        match t.state {
            TaskState::Completed => break Ok(t),
            TaskState::Failed(e) => break Err(e.message),
            _ => {}
        }
    };

    let task = result.expect("Task should complete successfully");
    assert_eq!(task.stats.tokens_consumed, 7);
    assert_eq!(task.stats.llm_calls, 1);
    assert!(task.stats.started_at.is_some());

    // Agent stats should be updated
    let agent = kernel.get_agent(agent_id).await.unwrap();
    assert_eq!(agent.stats.total_tasks_completed, 1);
    assert_eq!(agent.stats.total_tokens_consumed, 7);
}

#[tokio::test]
async fn test_e2e_task_fails_without_driver() {
    let kernel = Arc::new(KernelService::new());
    kernel.start();

    let agent_id = kernel
        .create_agent(make_spec("no-driver-agent"))
        .await
        .unwrap();
    kernel.resume_agent(agent_id).await.unwrap();

    let task = make_task(agent_id, "This should fail — no LLM driver");
    let task_id = task.task_id;
    kernel.submit_task(task).await.unwrap();

    let mut attempts = 0;
    loop {
        attempts += 1;
        assert!(attempts <= 50, "Task did not fail within 5 seconds");
        tokio::time::sleep(Duration::from_millis(100)).await;
        let t = kernel.get_task(task_id).await.unwrap();
        match t.state {
            TaskState::Failed(e) => {
                assert!(
                    e.message.contains("No available LLM driver"),
                    "expected 'No available LLM driver', got: {e:?}"
                );
                break;
            }
            TaskState::Completed => panic!("Task should not complete without a driver"),
            _ => {}
        }
    }

    let agent = kernel.get_agent(agent_id).await.unwrap();
    assert_eq!(agent.stats.total_tasks_failed, 1);
}
