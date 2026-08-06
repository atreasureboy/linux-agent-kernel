//! Integration test: Agent lifecycle — create → activate → submit task → destroy
//!
//! This test verifies the full AgentKernel path from birth to death of an agent,
//! exercising the scheduler, token budget, reasoning service, and state transitions.

use chrono::Utc;
use lak_core::error::KernelError;
use lak_core::token_budget::{BudgetAllocation, TokenBudget};
use lak_core::traits::AgentKernel;
use lak_core::types::agent::{AgentSpec, AgentState};
use lak_core::types::capability::{
    Capability, CapabilityPermission, CapabilityScope, CapabilityType,
};
use lak_core::types::ids::AgentId;
use lak_core::types::intent::{
    IntentContent, IntentMessage, IntentSubscription, IntentTarget, IntentType,
};
use lak_core::types::memory::{MemoryChunk, MemoryContent, MemoryMetadata, MemoryTier};
use lak_core::types::task::CognitivePriority;
use lak_core::types::task::{CognitiveTask, TaskContent, TaskState, TaskType};

use lak_services::kernel::KernelService;

// ── Utility ──────────────────────────────────────────────────────

fn make_agent_spec(name: &str) -> AgentSpec {
    AgentSpec {
        name: name.to_string(),
        initial_capabilities: vec![Capability {
            cap_type: CapabilityType::FileRead,
            scope: CapabilityScope {
                pattern: "**".into(),
            },
            permissions: CapabilityPermission::READ,
            constraints: vec![],
        }],
        ..Default::default()
    }
}

fn make_task(agent_id: AgentId, description: &str) -> CognitiveTask {
    CognitiveTask {
        task_id: lak_core::types::ids::TaskId::new(),
        agent_id,
        task_type: TaskType::Reasoning,
        priority: CognitivePriority::normal(),
        state: TaskState::Pending,
        content: TaskContent {
            natural_language: description.to_string(),
            structured_schema: None,
            memory_references: vec![],
        },
        deadline: None,
        dependencies: vec![],
        created_at: Utc::now(),
        updated_at: Utc::now(),
        metadata: std::collections::HashMap::new(),
        stats: lak_core::types::task::TaskStats::default(),
    }
}

fn make_memory(text: &str, agent_id: AgentId) -> MemoryChunk {
    MemoryChunk {
        chunk_id: lak_core::types::ids::MemoryChunkId::new(),
        agent_id,
        content: MemoryContent {
            raw_text: text.to_string(),
            structured_data: None,
            embedding: None,
        },
        metadata: MemoryMetadata {
            created_at: Utc::now(),
            last_accessed_at: Utc::now(),
            access_count: 0,
            importance_score: 0.5,
            decay_rate: 0.01,
            source: lak_core::types::memory::MemorySource::UserInput,
            factuality: lak_core::types::memory::Factuality::Belief(0.9),
        },
        relations: vec![],
        tier: MemoryTier::Working,
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[tokio::test]
async fn test_create_and_get_agent() {
    let kernel = KernelService::new();
    let spec = make_agent_spec("test-agent");
    let agent_id = kernel.create_agent(spec).await.unwrap();

    let agent = kernel.get_agent(agent_id).await.unwrap();
    assert_eq!(agent.name, "test-agent");
    assert!(matches!(agent.state, AgentState::Created));
}

#[tokio::test]
async fn test_create_destroy_agent_lifecycle() {
    let kernel = KernelService::new();
    let spec = make_agent_spec("lifecycle-agent");
    let agent_id = kernel.create_agent(spec).await.unwrap();

    // Agent should be created
    let agent = kernel.get_agent(agent_id).await.unwrap();
    assert_eq!(agent.state, AgentState::Created);

    // Destroy the agent
    kernel.destroy_agent(agent_id).await.unwrap();

    // Agent should now be Terminated
    let agent = kernel.get_agent(agent_id).await.unwrap();
    assert_eq!(agent.state, AgentState::Terminated);
    assert!(agent.terminated_at.is_some());
}

#[tokio::test]
async fn test_pause_and_resume_agent() {
    let kernel = KernelService::new();
    let spec = make_agent_spec("pause-agent");
    let agent_id = kernel.create_agent(spec).await.unwrap();

    // Pause agent
    kernel.pause_agent(agent_id).await.unwrap();
    let agent = kernel.get_agent(agent_id).await.unwrap();
    assert_eq!(agent.state, AgentState::Suspended);

    // Resume agent
    kernel.resume_agent(agent_id).await.unwrap();
    let agent = kernel.get_agent(agent_id).await.unwrap();
    assert_eq!(agent.state, AgentState::Idle);
}

#[tokio::test]
async fn test_submit_and_get_task() {
    let kernel = KernelService::new();
    let spec = make_agent_spec("task-agent");
    let agent_id = kernel.create_agent(spec).await.unwrap();

    // Resume so agent is Idle (can accept tasks)
    kernel.resume_agent(agent_id).await.unwrap();

    let task = make_task(agent_id, "Analyze Rust code for memory safety");
    let task_id = task.task_id;

    let returned_id = kernel.submit_task(task).await.unwrap();
    assert_eq!(returned_id, task_id);

    // Task should be retrievable
    let retrieved = kernel.get_task(task_id).await.unwrap();
    assert_eq!(retrieved.task_id, task_id);
    assert_eq!(retrieved.agent_id, agent_id);
}

#[tokio::test]
async fn test_cancel_task() {
    let kernel = KernelService::new();
    let spec = make_agent_spec("cancel-agent");
    let agent_id = kernel.create_agent(spec).await.unwrap();
    kernel.resume_agent(agent_id).await.unwrap();

    let task = make_task(agent_id, "Task to be cancelled");
    let task_id = kernel.submit_task(task).await.unwrap();

    kernel.cancel_task(task_id).await.unwrap();
    let task = kernel.get_task(task_id).await.unwrap();
    assert_eq!(task.state, TaskState::Cancelled);
}

#[tokio::test]
async fn test_list_agents() {
    let kernel = KernelService::new();

    let a1 = make_agent_spec("agent-1");
    let a2 = make_agent_spec("agent-2");

    kernel.create_agent(a1).await.unwrap();
    kernel.create_agent(a2).await.unwrap();

    let agents = kernel.list_agents().await.unwrap();
    assert!(agents.len() >= 2);
}

#[tokio::test]
async fn test_system_status() {
    let kernel = KernelService::new();
    let status = kernel.get_system_status().await.unwrap();

    assert_eq!(status.active_agents, 0);
    // A freshly started kernel reports a small, sane uptime
    assert!(status.uptime_seconds < 60);
    // Default max agents is 1000
    assert_eq!(status.max_agents, 1000);
}

#[tokio::test]
async fn test_store_and_query_memory() {
    let kernel = KernelService::new();
    let spec = make_agent_spec("memory-agent");
    let agent_id = kernel.create_agent(spec).await.unwrap();

    let mem = make_memory("Rust async programming uses tokio runtime", agent_id);
    kernel.store_memory(agent_id, mem).await.unwrap();

    let results = kernel
        .query_memory(agent_id, "tokio async", 5)
        .await
        .unwrap();
    assert!(!results.is_empty());
}

#[tokio::test]
async fn test_send_and_await_intent() {
    let kernel = KernelService::new();
    let spec = make_agent_spec("intent-agent");
    let agent_id = kernel.create_agent(spec).await.unwrap();

    let intent = IntentMessage {
        intent_id: lak_core::types::ids::IntentId::new(),
        source_agent_id: AgentId::new(),
        target: IntentTarget::Broadcast,
        intent_type: IntentType::Inform,
        content: IntentContent {
            natural_language: "security alert detected".to_string(),
            structured_data: None,
            memory_references: vec![],
        },
        priority: CognitivePriority::high(),
        ttl_ms: 30_000,
        correlation_id: None,
        created_at: Utc::now(),
    };

    kernel.send_intent(intent).await.unwrap();

    let sub = IntentSubscription {
        agent_id,
        intent_types: None,
        topic_pattern: Some("security".into()),
        capability_filter: None,
    };

    let received = kernel.await_intent(agent_id, sub).await.unwrap();
    assert!(received.content.natural_language.contains("security"));
}

#[tokio::test]
async fn test_agent_not_found_error() {
    let kernel = KernelService::new();
    let fake_id = AgentId::new();

    let result = kernel.get_agent(fake_id).await;
    assert!(matches!(result, Err(KernelError::AgentNotFound(_))));
}

#[tokio::test]
async fn test_task_not_found_error() {
    let kernel = KernelService::new();
    let fake_id = lak_core::types::ids::TaskId::new();

    let result = kernel.get_task(fake_id).await;
    assert!(matches!(result, Err(KernelError::TaskNotFound(_))));
}

#[tokio::test]
async fn test_token_budget_enforcement() {
    let mut budget = TokenBudget::new(100, 80, 10);

    // Should allow normal allocation
    let result = budget.check_allocation(50, 50.0);
    assert!(matches!(result, BudgetAllocation::Granted(_)));

    // Exhaust the budget
    budget.record_consumption(100);
    let result = budget.check_allocation(10, 50.0);
    assert!(matches!(result, BudgetAllocation::Denied(_)));

    // High priority should access reserve
    let mut budget2 = TokenBudget::new(100, 80, 10);
    budget2.record_consumption(95);
    let result = budget2.check_allocation(10, 90.0); // High priority
    assert!(matches!(
        result,
        BudgetAllocation::Granted(_) | BudgetAllocation::Reduced(_, _)
    ));
}

#[tokio::test]
async fn test_agent_limit_enforcement() {
    let kernel = KernelService::with_max_agents(KernelService::new(), 1);

    let spec1 = make_agent_spec("only-agent");
    kernel.create_agent(spec1).await.unwrap();

    let spec2 = make_agent_spec("second-agent");
    let result = kernel.create_agent(spec2).await;

    assert!(matches!(result, Err(KernelError::AgentLimitReached { .. })));
}
