//! gRPC Server — bridge between proto and KernelService
//!
//! Implements the tonic-generated AgentKernel trait by converting proto
//! messages to/from Rust types and delegating to the KernelService.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use tonic::{Request, Response, Status};
use uuid::Uuid;

use lak_core::error::KernelError;
use lak_core::traits::AgentKernel;
use lak_core::types::agent::{Agent as CoreAgent, AgentConfig, AgentSpec, AgentState};
use lak_core::types::capability::{
    Capability, CapabilityPermission, CapabilityRequirement, CapabilityScope, CapabilityType,
};
use lak_core::types::ids::{AgentId, CapabilityCertId, IntentId, MemoryChunkId, TaskId};
use lak_core::types::intent::{
    IntentContent, IntentMessage, IntentSubscription, IntentTarget, IntentType,
};
use lak_core::types::memory::{
    Factuality, MemoryChunk, MemoryContent, MemoryMetadata, MemorySource, MemoryTier,
};
use lak_core::types::task::{
    CognitivePriority, CognitiveTask, TaskContent, TaskError, TaskState, TaskStats, TaskType,
};

use lak_proto::lak::{
    self as pb,
    agent_kernel_server::{AgentKernel as GrpcAgentKernel, AgentKernelServer},
};

// ════════════════════════════════════════════════════════════════════
// Proto ↔ Core conversions
// ════════════════════════════════════════════════════════════════════

// ── UUID ↔ bytes ──

fn bytes_to_uuid(b: &[u8]) -> Result<Uuid, Status> {
    Uuid::from_slice(b).map_err(|e| Status::invalid_argument(format!("invalid UUID: {e}")))
}

fn bytes_to_agent_id(b: &[u8]) -> Result<AgentId, Status> {
    bytes_to_uuid(b).map(AgentId::from_uuid)
}

fn bytes_to_task_id(b: &[u8]) -> Result<TaskId, Status> {
    bytes_to_uuid(b).map(TaskId::from_uuid)
}

fn bytes_to_intent_id(b: &[u8]) -> Result<IntentId, Status> {
    bytes_to_uuid(b).map(IntentId::from_uuid)
}

fn bytes_to_memory_chunk_id(b: &[u8]) -> Result<MemoryChunkId, Status> {
    bytes_to_uuid(b).map(MemoryChunkId::from_uuid)
}

fn bytes_to_capability_cert_id(b: &[u8]) -> Result<CapabilityCertId, Status> {
    bytes_to_uuid(b).map(CapabilityCertId::from_uuid)
}

fn agent_id_to_bytes(id: AgentId) -> Vec<u8> {
    id.as_uuid().as_bytes().to_vec()
}
fn task_id_to_bytes(id: TaskId) -> Vec<u8> {
    id.as_uuid().as_bytes().to_vec()
}
fn intent_id_to_bytes(id: IntentId) -> Vec<u8> {
    id.as_uuid().as_bytes().to_vec()
}
fn memory_chunk_id_to_bytes(id: MemoryChunkId) -> Vec<u8> {
    id.as_uuid().as_bytes().to_vec()
}
fn capability_cert_id_to_bytes(id: CapabilityCertId) -> Vec<u8> {
    id.as_uuid().as_bytes().to_vec()
}

// ── Timestamp ↔ DateTime ──

fn timestamp_to_datetime(ts: Option<&prost_types::Timestamp>) -> Result<DateTime<Utc>, Status> {
    let ts = ts.ok_or_else(|| Status::invalid_argument("missing timestamp"))?;
    DateTime::from_timestamp(ts.seconds, ts.nanos as u32)
        .ok_or_else(|| Status::invalid_argument("invalid timestamp"))
}

fn datetime_to_timestamp(dt: DateTime<Utc>) -> prost_types::Timestamp {
    prost_types::Timestamp {
        seconds: dt.timestamp(),
        nanos: dt.timestamp_subsec_nanos() as i32,
    }
}

fn opt_datetime_to_timestamp(dt: Option<DateTime<Utc>>) -> Option<prost_types::Timestamp> {
    dt.map(datetime_to_timestamp)
}

// ── Conversions using generated prost enum names ──
// prost strips the common prefix from proto enum values:
//   AGENT_STATE_CREATED → AgentState::Created (strips AGENT_STATE_)

#[allow(dead_code)]
fn agent_state_from_pb(s: i32) -> AgentState {
    match pb::AgentState::try_from(s) {
        Ok(pb::AgentState::Created) => AgentState::Created,
        Ok(pb::AgentState::Initializing) => AgentState::Initializing,
        Ok(pb::AgentState::Running) => AgentState::Running,
        Ok(pb::AgentState::Idle) => AgentState::Idle,
        Ok(pb::AgentState::Blocked) => AgentState::Blocked,
        Ok(pb::AgentState::Suspended) => AgentState::Suspended,
        Ok(pb::AgentState::Sleeping) => AgentState::Sleeping,
        Ok(pb::AgentState::Terminated) => AgentState::Terminated,
        _ => AgentState::Created,
    }
}

fn agent_state_to_pb(s: AgentState) -> i32 {
    match s {
        AgentState::Created => pb::AgentState::Created as i32,
        AgentState::Initializing => pb::AgentState::Initializing as i32,
        AgentState::Running => pb::AgentState::Running as i32,
        AgentState::Idle => pb::AgentState::Idle as i32,
        AgentState::Blocked => pb::AgentState::Blocked as i32,
        AgentState::Suspended => pb::AgentState::Suspended as i32,
        AgentState::Sleeping => pb::AgentState::Sleeping as i32,
        AgentState::Terminated => pb::AgentState::Terminated as i32,
    }
}

fn task_type_from_pb(t: i32) -> TaskType {
    match pb::TaskType::try_from(t) {
        Ok(pb::TaskType::Reasoning) => TaskType::Reasoning,
        Ok(pb::TaskType::ToolExecution) => TaskType::ToolExecution,
        Ok(pb::TaskType::MemoryRetrieval) => TaskType::MemoryRetrieval,
        Ok(pb::TaskType::IntentProcessing) => TaskType::IntentProcessing,
        Ok(pb::TaskType::IdleReflection) => TaskType::IdleReflection,
        Ok(pb::TaskType::SystemTask) => TaskType::SystemTask,
        _ => TaskType::Reasoning,
    }
}

fn task_state_from_pb(s: i32) -> TaskState {
    match pb::TaskState::try_from(s) {
        Ok(pb::TaskState::Pending) => TaskState::Pending,
        Ok(pb::TaskState::Ready) => TaskState::Ready,
        Ok(pb::TaskState::Running) => TaskState::Running,
        Ok(pb::TaskState::Blocked) => TaskState::Blocked(String::new()),
        Ok(pb::TaskState::AwaitingLlm) => TaskState::AwaitingLLM,
        Ok(pb::TaskState::AwaitingTool) => TaskState::AwaitingTool,
        Ok(pb::TaskState::AwaitingIntent) => TaskState::AwaitingIntent,
        Ok(pb::TaskState::Suspended) => TaskState::Suspended,
        Ok(pb::TaskState::Completed) => TaskState::Completed,
        Ok(pb::TaskState::Failed) => TaskState::Failed(TaskError {
            code: "UNKNOWN".into(),
            message: "".into(),
            retryable: false,
        }),
        Ok(pb::TaskState::Cancelled) => TaskState::Cancelled,
        _ => TaskState::Pending,
    }
}

fn task_state_to_pb(s: &TaskState) -> i32 {
    match s {
        TaskState::Pending => pb::TaskState::Pending as i32,
        TaskState::Ready => pb::TaskState::Ready as i32,
        TaskState::Running => pb::TaskState::Running as i32,
        TaskState::Blocked(_) => pb::TaskState::Blocked as i32,
        TaskState::AwaitingLLM => pb::TaskState::AwaitingLlm as i32,
        TaskState::AwaitingTool => pb::TaskState::AwaitingTool as i32,
        TaskState::AwaitingIntent => pb::TaskState::AwaitingIntent as i32,
        TaskState::Suspended => pb::TaskState::Suspended as i32,
        TaskState::Completed => pb::TaskState::Completed as i32,
        TaskState::Failed(_) => pb::TaskState::Failed as i32,
        TaskState::Cancelled => pb::TaskState::Cancelled as i32,
    }
}

fn task_type_to_pb(t: TaskType) -> i32 {
    match t {
        TaskType::Reasoning => pb::TaskType::Reasoning as i32,
        TaskType::ToolExecution => pb::TaskType::ToolExecution as i32,
        TaskType::MemoryRetrieval => pb::TaskType::MemoryRetrieval as i32,
        TaskType::IntentProcessing => pb::TaskType::IntentProcessing as i32,
        TaskType::IdleReflection => pb::TaskType::IdleReflection as i32,
        TaskType::SystemTask => pb::TaskType::SystemTask as i32,
    }
}

fn intent_type_from_pb(t: i32) -> IntentType {
    match pb::IntentType::try_from(t) {
        Ok(pb::IntentType::Query) => IntentType::Query,
        Ok(pb::IntentType::Command) => IntentType::Delegate,
        Ok(pb::IntentType::Notification) => IntentType::Inform,
        Ok(pb::IntentType::Request) => IntentType::RequestApproval,
        Ok(pb::IntentType::Response) => IntentType::Respond,
        Ok(pb::IntentType::Error) => IntentType::Negotiate,
        _ => IntentType::Inform,
    }
}

fn cap_type_from_name(name: &str) -> Option<CapabilityType> {
    let candidates = [
        CapabilityType::FileRead,
        CapabilityType::FileWrite,
        CapabilityType::FileDelete,
        CapabilityType::NetworkHttp,
        CapabilityType::NetworkTcp,
        CapabilityType::ProcessExecute,
        CapabilityType::AgentCreate,
        CapabilityType::AgentDestroy,
        CapabilityType::MemoryShare,
        CapabilityType::IntentSend,
        CapabilityType::LLMCall,
        CapabilityType::ToolExecute,
    ];
    candidates.into_iter().find(|c| format!("{c:?}") == name)
}

fn intent_target_from_pb(t: &Option<pb::IntentTarget>) -> Result<IntentTarget, Status> {
    let t = t
        .as_ref()
        .ok_or_else(|| Status::invalid_argument("missing intent target"))?;
    let tt = pb::IntentTargetType::try_from(t.target_type)
        .unwrap_or(pb::IntentTargetType::IntentTargetBroadcast);
    match tt {
        pb::IntentTargetType::IntentTargetBroadcast => Ok(IntentTarget::Broadcast),
        pb::IntentTargetType::IntentTargetUnicast => {
            let id = t
                .agent_ids
                .first()
                .map(|b| bytes_to_agent_id(b))
                .transpose()?
                .unwrap_or(AgentId::SYSTEM);
            Ok(IntentTarget::Unicast(id))
        }
        pb::IntentTargetType::IntentTargetMulticast => {
            let ids: Result<Vec<AgentId>, Status> =
                t.agent_ids.iter().map(|b| bytes_to_agent_id(b)).collect();
            Ok(IntentTarget::Multicast(ids?))
        }
        pb::IntentTargetType::IntentTargetByCapability => {
            let cap_name = t
                .filters
                .get("capability_type")
                .cloned()
                .unwrap_or_default();
            Ok(IntentTarget::ByCapability {
                cap_type: cap_type_from_name(&cap_name).unwrap_or(CapabilityType::ToolExecute),
                semantic_hint: t.filters.get("semantic_hint").cloned(),
            })
        }
        pb::IntentTargetType::IntentTargetPublishSubscribe => Ok(IntentTarget::PublishSubscribe {
            pattern: t.filters.get("pattern").cloned().unwrap_or_default(),
        }),
        _ => Ok(IntentTarget::Broadcast),
    }
}

fn memory_tier_from_pb(t: i32) -> MemoryTier {
    match pb::MemoryTier::try_from(t) {
        Ok(pb::MemoryTier::Working) => MemoryTier::Working,
        Ok(pb::MemoryTier::Short) => MemoryTier::ShortTerm,
        Ok(pb::MemoryTier::Long) => MemoryTier::LongTerm,
        Ok(pb::MemoryTier::Archival) => MemoryTier::Archival,
        _ => MemoryTier::ShortTerm,
    }
}

fn factuality_from_pb(f: i32) -> Factuality {
    match pb::Factuality::try_from(f) {
        Ok(pb::Factuality::Fact) => Factuality::Fact,
        Ok(pb::Factuality::Belief) => Factuality::Belief(0.5),
        Ok(pb::Factuality::Hearsay) => Factuality::Hearsay,
        _ => Factuality::Belief(0.5),
    }
}

fn memory_source_from_pb(s: i32) -> MemorySource {
    // MemorySourceType values don't share prefix with enum name, so full names retained
    match pb::MemorySourceType::try_from(s) {
        Ok(pb::MemorySourceType::MemorySourceReasoning) => MemorySource::AgentReasoning,
        Ok(pb::MemorySourceType::MemorySourceObservation) => MemorySource::UserInput,
        Ok(pb::MemorySourceType::MemorySourceToolOutput) => MemorySource::ToolOutput,
        _ => MemorySource::ExternalSource,
    }
}

fn cap_type_from_pb(t: i32) -> CapabilityType {
    // CapabilityType values strip CAPABILITY_TYPE_ prefix
    match pb::CapabilityType::try_from(t) {
        Ok(pb::CapabilityType::FileRead) => CapabilityType::FileRead,
        Ok(pb::CapabilityType::FileWrite) => CapabilityType::FileWrite,
        Ok(pb::CapabilityType::FileDelete) => CapabilityType::FileDelete,
        Ok(pb::CapabilityType::ShellExec) => CapabilityType::ProcessExecute,
        Ok(pb::CapabilityType::NetworkHttp) => CapabilityType::NetworkHttp,
        Ok(pb::CapabilityType::NetworkRaw) => CapabilityType::NetworkTcp,
        Ok(pb::CapabilityType::AgentCreate) => CapabilityType::AgentCreate,
        Ok(pb::CapabilityType::AgentDestroy) => CapabilityType::AgentDestroy,
        Ok(pb::CapabilityType::MemoryShare) => CapabilityType::MemoryShare,
        Ok(pb::CapabilityType::IntentSend) => CapabilityType::IntentSend,
        Ok(pb::CapabilityType::LlmCall) => CapabilityType::LLMCall,
        Ok(pb::CapabilityType::LlmStream) => CapabilityType::LLMCall,
        _ => CapabilityType::ToolExecute,
    }
}

fn cap_type_to_pb(t: &CapabilityType) -> i32 {
    match t {
        CapabilityType::FileRead => pb::CapabilityType::FileRead as i32,
        CapabilityType::FileWrite => pb::CapabilityType::FileWrite as i32,
        CapabilityType::FileDelete => pb::CapabilityType::FileDelete as i32,
        CapabilityType::FileWatch => pb::CapabilityType::FileRead as i32,
        CapabilityType::NetworkHttp => pb::CapabilityType::NetworkHttp as i32,
        CapabilityType::NetworkWebSocket | CapabilityType::NetworkTcp => {
            pb::CapabilityType::NetworkRaw as i32
        }
        CapabilityType::ProcessExecute | CapabilityType::ProcessSignal => {
            pb::CapabilityType::ShellExec as i32
        }
        CapabilityType::AgentCreate => pb::CapabilityType::AgentCreate as i32,
        CapabilityType::AgentDestroy => pb::CapabilityType::AgentDestroy as i32,
        CapabilityType::AgentCommunicate => pb::CapabilityType::IntentSend as i32,
        CapabilityType::MemoryRead | CapabilityType::MemoryWrite => {
            pb::CapabilityType::DatabaseRead as i32
        }
        CapabilityType::MemoryShare => pb::CapabilityType::MemoryShare as i32,
        CapabilityType::IntentSend | CapabilityType::IntentBroadcast => {
            pb::CapabilityType::IntentSend as i32
        }
        CapabilityType::LLMCall | CapabilityType::LLMFineTune => pb::CapabilityType::LlmCall as i32,
        CapabilityType::ToolRegister | CapabilityType::ToolExecute => {
            pb::CapabilityType::SubprocessSpawn as i32
        }
        CapabilityType::CognodeRead | CapabilityType::CognodeWrite => {
            pb::CapabilityType::DatabaseRead as i32
        }
        CapabilityType::SystemConfig => pb::CapabilityType::SystemConfig as i32,
        CapabilityType::SystemMetrics => pb::CapabilityType::SystemMonitor as i32,
        CapabilityType::SystemAudit => pb::CapabilityType::SystemLog as i32,
        CapabilityType::SystemShutdown => pb::CapabilityType::SystemConfig as i32,
    }
}

fn intent_type_to_pb(t: IntentType) -> i32 {
    match t {
        IntentType::Query => pb::IntentType::Query as i32,
        IntentType::Delegate => pb::IntentType::Command as i32,
        IntentType::Inform => pb::IntentType::Notification as i32,
        IntentType::RequestApproval => pb::IntentType::Request as i32,
        IntentType::Respond => pb::IntentType::Response as i32,
        IntentType::Negotiate => pb::IntentType::Request as i32,
        IntentType::Monitor => pb::IntentType::Query as i32,
    }
}

fn intent_target_to_pb(target: &IntentTarget) -> pb::IntentTarget {
    let mut filters = std::collections::HashMap::new();
    let (target_type, agent_ids) = match target {
        IntentTarget::Broadcast => (pb::IntentTargetType::IntentTargetBroadcast, vec![]),
        IntentTarget::Unicast(id) => (
            pb::IntentTargetType::IntentTargetUnicast,
            vec![agent_id_to_bytes(*id)],
        ),
        IntentTarget::Multicast(ids) => (
            pb::IntentTargetType::IntentTargetMulticast,
            ids.iter().map(|id| agent_id_to_bytes(*id)).collect(),
        ),
        IntentTarget::ByCapability {
            cap_type,
            semantic_hint,
        } => {
            filters.insert("capability_type".into(), format!("{cap_type:?}"));
            if let Some(hint) = semantic_hint {
                filters.insert("semantic_hint".into(), hint.clone());
            }
            (pb::IntentTargetType::IntentTargetByCapability, vec![])
        }
        IntentTarget::PublishSubscribe { pattern } => {
            filters.insert("pattern".into(), pattern.clone());
            (pb::IntentTargetType::IntentTargetPublishSubscribe, vec![])
        }
    };

    pb::IntentTarget {
        target_type: target_type as i32,
        agent_ids,
        filters,
    }
}

fn memory_source_to_pb(s: MemorySource) -> i32 {
    match s {
        MemorySource::UserInput => pb::MemorySourceType::MemorySourceObservation as i32,
        MemorySource::AgentReasoning => pb::MemorySourceType::MemorySourceReasoning as i32,
        MemorySource::ToolOutput => pb::MemorySourceType::MemorySourceToolOutput as i32,
        MemorySource::OtherAgent(_) | MemorySource::ExternalSource => {
            pb::MemorySourceType::MemorySourceObservation as i32
        }
    }
}

fn factuality_to_pb(f: Factuality) -> (i32, Option<f32>) {
    match f {
        Factuality::Fact => (pb::Factuality::Fact as i32, None),
        Factuality::Belief(conf) => (pb::Factuality::Belief as i32, Some(conf)),
        Factuality::Hearsay => (pb::Factuality::Hearsay as i32, None),
    }
}

// ── AgentSpec proto → core ──

fn agent_spec_from_pb(pb_spec: &pb::AgentSpec) -> AgentSpec {
    let config = pb_spec.config.as_ref();
    AgentSpec {
        name: pb_spec.name.clone(),
        system_prompt: pb_spec.system_prompt.clone(),
        model: pb_spec.model.clone(),
        max_context_tokens: pb_spec.max_context_tokens as usize,
        initial_capabilities: pb_spec
            .initial_capabilities
            .iter()
            .map(|c| Capability {
                cap_type: cap_type_from_pb(c.cap_type),
                scope: CapabilityScope {
                    pattern: c.scope.clone(),
                },
                permissions: CapabilityPermission::from_bits_truncate(c.permissions),
                constraints: vec![],
            })
            .collect(),
        memory_quota_bytes: pb_spec.memory_quota_bytes,
        config: AgentConfig {
            temperature: config.map(|c| c.temperature).unwrap_or(0.7),
            max_tool_calls_per_task: config.map(|c| c.max_tool_calls_per_task).unwrap_or(20),
            reasoning_timeout_seconds: config.map(|c| c.reasoning_timeout_seconds).unwrap_or(120),
            allow_self_reflection: config.map(|c| c.allow_self_reflection).unwrap_or(true),
            require_approval_for_write: config
                .map(|c| c.require_approval_for_write)
                .unwrap_or(false),
            metadata: config.map(|c| c.metadata.clone()).unwrap_or_default(),
            tags: config.map(|c| c.tags.clone()).unwrap_or_default(),
        },
    }
}

// ── Agent core → proto ──

fn agent_to_pb(agent: &CoreAgent) -> pb::Agent {
    pb::Agent {
        agent_id: agent_id_to_bytes(agent.id),
        name: agent.name.clone(),
        spec: Some(pb::AgentSpec {
            name: agent.spec.name.clone(),
            system_prompt: agent.spec.system_prompt.clone(),
            model: agent.spec.model.clone(),
            max_context_tokens: agent.spec.max_context_tokens as u64,
            initial_capabilities: vec![],
            memory_quota_bytes: agent.spec.memory_quota_bytes,
            config: Some(pb::AgentConfig {
                temperature: agent.spec.config.temperature,
                max_tool_calls_per_task: agent.spec.config.max_tool_calls_per_task,
                reasoning_timeout_seconds: agent.spec.config.reasoning_timeout_seconds,
                allow_self_reflection: agent.spec.config.allow_self_reflection,
                require_approval_for_write: agent.spec.config.require_approval_for_write,
                metadata: agent.spec.config.metadata.clone(),
                tags: agent.spec.config.tags.clone(),
            }),
        }),
        state: agent_state_to_pb(agent.state),
        stats: Some(pb::AgentStats {
            total_tasks_completed: agent.stats.total_tasks_completed,
            total_tasks_failed: agent.stats.total_tasks_failed,
            total_tokens_consumed: agent.stats.total_tokens_consumed,
            total_tool_calls: agent.stats.total_tool_calls,
            total_tool_failures: agent.stats.total_tool_failures,
            coi: agent.stats.coi,
            hallucination_rate: agent.stats.hallucination_rate,
            reasoning_loop_count: agent.stats.reasoning_loop_count,
            avg_response_latency_seconds: agent.stats.avg_response_latency_seconds,
            capability_violations: agent.stats.capability_violations,
        }),
        created_at: Some(datetime_to_timestamp(agent.created_at)),
        last_active_at: Some(datetime_to_timestamp(agent.last_active_at)),
        terminated_at: opt_datetime_to_timestamp(agent.terminated_at),
    }
}

// ── CognitiveTask proto ↔ core ──

fn cognitive_task_from_pb(pb_task: &pb::CognitiveTask) -> Result<CognitiveTask, Status> {
    Ok(CognitiveTask {
        task_id: bytes_to_task_id(&pb_task.task_id)?,
        agent_id: bytes_to_agent_id(&pb_task.agent_id)?,
        task_type: task_type_from_pb(pb_task.task_type),
        priority: CognitivePriority {
            urgency: pb_task
                .priority
                .as_ref()
                .map(|p| p.urgency as u8)
                .unwrap_or(40),
            importance: pb_task
                .priority
                .as_ref()
                .map(|p| p.importance as u8)
                .unwrap_or(50),
            context_affinity: pb_task
                .priority
                .as_ref()
                .map(|p| p.context_affinity as u8)
                .unwrap_or(50),
        },
        state: task_state_from_pb(pb_task.state),
        content: TaskContent {
            natural_language: pb_task
                .content
                .as_ref()
                .map(|c| c.natural_language.clone())
                .unwrap_or_default(),
            structured_schema: pb_task
                .content
                .as_ref()
                .and_then(|c| c.structured_schema_json.as_ref())
                .and_then(|s| serde_json::from_str(s).ok()),
            memory_references: pb_task
                .content
                .as_ref()
                .map(|c| {
                    c.memory_references
                        .iter()
                        .filter_map(|b| bytes_to_memory_chunk_id(b).ok())
                        .collect()
                })
                .unwrap_or_default(),
        },
        deadline: pb_task
            .deadline
            .as_ref()
            .and_then(|ts| timestamp_to_datetime(Some(ts)).ok()),
        dependencies: pb_task
            .dependencies
            .iter()
            .filter_map(|b| bytes_to_task_id(b).ok())
            .collect(),
        created_at: pb_task
            .created_at
            .as_ref()
            .and_then(|ts| timestamp_to_datetime(Some(ts)).ok())
            .unwrap_or_else(Utc::now),
        updated_at: pb_task
            .updated_at
            .as_ref()
            .and_then(|ts| timestamp_to_datetime(Some(ts)).ok())
            .unwrap_or_else(Utc::now),
        metadata: pb_task.metadata.clone(),
        stats: TaskStats::default(),
    })
}

fn cognitive_task_to_pb(task: &CognitiveTask) -> pb::CognitiveTask {
    pb::CognitiveTask {
        task_id: task_id_to_bytes(task.task_id),
        agent_id: agent_id_to_bytes(task.agent_id),
        task_type: task_type_to_pb(task.task_type),
        priority: Some(pb::CognitivePriority {
            urgency: task.priority.urgency as u32,
            importance: task.priority.importance as u32,
            context_affinity: task.priority.context_affinity as u32,
        }),
        state: task_state_to_pb(&task.state),
        content: Some(pb::TaskContent {
            natural_language: task.content.natural_language.clone(),
            structured_schema_json: task
                .content
                .structured_schema
                .as_ref()
                .map(|v| v.to_string()),
            memory_references: task
                .content
                .memory_references
                .iter()
                .map(|id| memory_chunk_id_to_bytes(*id))
                .collect(),
        }),
        deadline: opt_datetime_to_timestamp(task.deadline),
        dependencies: task
            .dependencies
            .iter()
            .map(|id| task_id_to_bytes(*id))
            .collect(),
        created_at: Some(datetime_to_timestamp(task.created_at)),
        updated_at: Some(datetime_to_timestamp(task.updated_at)),
        metadata: task.metadata.clone(),
        stats: Some(pb::TaskStats {
            tokens_consumed: task.stats.tokens_consumed,
            tool_calls_made: task.stats.tool_calls_made,
            reasoning_steps: task.stats.reasoning_steps,
            memory_retrievals: task.stats.memory_retrievals,
            started_at: opt_datetime_to_timestamp(task.stats.started_at),
            completed_at: opt_datetime_to_timestamp(task.stats.completed_at),
            llm_calls: task.stats.llm_calls,
            total_wall_time_ms: task.stats.total_wall_time_ms,
        }),
        blocked_reason: match &task.state {
            TaskState::Blocked(reason) => Some(reason.clone()),
            _ => None,
        },
        error: match &task.state {
            TaskState::Failed(e) => Some(pb::TaskErrorProto {
                code: e.code.clone(),
                message: e.message.clone(),
                retryable: e.retryable,
            }),
            _ => None,
        },
    }
}

// ── Intent proto → core ──

fn intent_message_from_pb(pb_intent: &pb::IntentMessage) -> Result<IntentMessage, Status> {
    Ok(IntentMessage {
        intent_id: bytes_to_intent_id(&pb_intent.intent_id)?,
        source_agent_id: bytes_to_agent_id(&pb_intent.sender_id)?,
        target: intent_target_from_pb(&pb_intent.target)?,
        intent_type: intent_type_from_pb(pb_intent.intent_type),
        content: IntentContent {
            natural_language: pb_intent
                .content
                .as_ref()
                .map(|c| c.natural_language.clone())
                .unwrap_or_default(),
            structured_data: pb_intent
                .content
                .as_ref()
                .and_then(|c| c.structured_json.as_ref())
                .and_then(|s| serde_json::from_str(s).ok()),
            memory_references: vec![],
        },
        priority: CognitivePriority::normal(),
        ttl_ms: 30_000,
        correlation_id: pb_intent
            .reply_to_intent_id
            .as_ref()
            .and_then(|b| bytes_to_intent_id(b).ok()),
        created_at: pb_intent
            .created_at
            .as_ref()
            .and_then(|ts| timestamp_to_datetime(Some(ts)).ok())
            .unwrap_or_else(Utc::now),
    })
}

fn intent_message_to_pb(intent: &IntentMessage) -> pb::IntentMessage {
    pb::IntentMessage {
        intent_id: intent_id_to_bytes(intent.intent_id),
        sender_id: agent_id_to_bytes(intent.source_agent_id),
        target: Some(intent_target_to_pb(&intent.target)),
        intent_type: intent_type_to_pb(intent.intent_type),
        content: Some(pb::IntentContent {
            natural_language: intent.content.natural_language.clone(),
            structured_json: intent
                .content
                .structured_data
                .as_ref()
                .map(|v| v.to_string()),
            headers: std::collections::HashMap::new(),
        }),
        priority: Some(pb::CognitivePriority {
            urgency: intent.priority.urgency as u32,
            importance: intent.priority.importance as u32,
            context_affinity: intent.priority.context_affinity as u32,
        }),
        reply_to_intent_id: intent.correlation_id.map(intent_id_to_bytes),
        requires_ack: matches!(intent.intent_type, IntentType::RequestApproval),
        created_at: Some(datetime_to_timestamp(intent.created_at)),
        expires_at: None,
    }
}

// ── IntentSubscription proto → core ──

fn intent_subscription_from_pb(
    pb_sub: &pb::IntentSubscription,
    agent_id: AgentId,
) -> IntentSubscription {
    IntentSubscription {
        agent_id,
        intent_types: if pb_sub.intent_types.is_empty() {
            None
        } else {
            Some(
                pb_sub
                    .intent_types
                    .iter()
                    .map(|t| intent_type_from_pb(*t))
                    .collect(),
            )
        },
        topic_pattern: pb_sub.topic_filters.values().next().cloned(),
        capability_filter: pb_sub
            .topic_filters
            .get("capability_type")
            .and_then(|name| cap_type_from_name(name)),
    }
}

// ── MemoryChunk proto ↔ core ──

fn memory_chunk_from_pb(pb_chunk: &pb::MemoryChunk) -> Result<MemoryChunk, Status> {
    Ok(MemoryChunk {
        chunk_id: bytes_to_memory_chunk_id(&pb_chunk.chunk_id)?,
        agent_id: bytes_to_agent_id(&pb_chunk.agent_id)?,
        content: MemoryContent {
            raw_text: pb_chunk
                .content
                .as_ref()
                .map(|c| c.raw_text.clone())
                .unwrap_or_default(),
            structured_data: None,
            embedding: pb_chunk
                .content
                .as_ref()
                .and_then(|c| c.embedding_json.as_ref())
                .and_then(|s| serde_json::from_str(s).ok()),
        },
        metadata: MemoryMetadata {
            created_at: pb_chunk
                .created_at
                .as_ref()
                .and_then(|ts| timestamp_to_datetime(Some(ts)).ok())
                .unwrap_or_else(Utc::now),
            last_accessed_at: pb_chunk
                .last_accessed_at
                .as_ref()
                .and_then(|ts| timestamp_to_datetime(Some(ts)).ok())
                .unwrap_or_else(Utc::now),
            access_count: pb_chunk.access_count as u64,
            importance_score: pb_chunk
                .metadata
                .as_ref()
                .map(|m| m.importance_score as f32)
                .unwrap_or(0.5),
            decay_rate: 0.01,
            source: pb_chunk
                .metadata
                .as_ref()
                .map(|m| memory_source_from_pb(m.source))
                .unwrap_or(MemorySource::ExternalSource),
            factuality: pb_chunk
                .metadata
                .as_ref()
                .map(|m| factuality_from_pb(m.factuality))
                .unwrap_or(Factuality::Belief(0.5)),
        },
        relations: vec![],
        tier: memory_tier_from_pb(pb_chunk.tier),
    })
}

fn memory_chunk_to_pb(chunk: &MemoryChunk) -> pb::MemoryChunk {
    let (factuality, confidence) = factuality_to_pb(chunk.metadata.factuality);
    pb::MemoryChunk {
        chunk_id: memory_chunk_id_to_bytes(chunk.chunk_id),
        agent_id: agent_id_to_bytes(chunk.agent_id),
        tier: match chunk.tier {
            MemoryTier::Working => pb::MemoryTier::Working as i32,
            MemoryTier::ShortTerm => pb::MemoryTier::Short as i32,
            MemoryTier::LongTerm => pb::MemoryTier::Long as i32,
            MemoryTier::Archival => pb::MemoryTier::Archival as i32,
        },
        content: Some(pb::MemoryContentProto {
            raw_text: chunk.content.raw_text.clone(),
            embedding_json: chunk
                .content
                .embedding
                .as_ref()
                .map(|e| serde_json::to_string(e).unwrap_or_default()),
            tags: std::collections::HashMap::new(),
        }),
        metadata: Some(pb::MemoryMetadataProto {
            source: memory_source_to_pb(chunk.metadata.source),
            factuality,
            confidence,
            importance_score: chunk.metadata.importance_score as f64,
            relation_ids: chunk
                .relations
                .iter()
                .map(|r| r.target_chunk_id.to_string())
                .collect(),
        }),
        created_at: Some(datetime_to_timestamp(chunk.metadata.created_at)),
        last_accessed_at: Some(datetime_to_timestamp(chunk.metadata.last_accessed_at)),
        access_count: chunk.metadata.access_count as u32,
    }
}

// ── KernelError → tonic::Status ──

fn kernel_error_to_status(err: KernelError) -> Status {
    match &err {
        KernelError::AgentNotFound(_) => Status::not_found(err.to_string()),
        KernelError::TaskNotFound(_) => Status::not_found(err.to_string()),
        KernelError::IntentNotFound(_) => Status::not_found(err.to_string()),
        KernelError::MemoryNotFound(_) => Status::not_found(err.to_string()),
        KernelError::PromptInjection(_) => Status::permission_denied(err.to_string()),
        KernelError::DelegationError(_) => Status::permission_denied(err.to_string()),
        KernelError::InsufficientCapability { .. } => Status::permission_denied(err.to_string()),
        KernelError::ContextOverflow { .. } => Status::resource_exhausted(err.to_string()),
        KernelError::ResourceExhausted { .. } => Status::resource_exhausted(err.to_string()),
        KernelError::TokenBudgetExceeded { .. } => Status::resource_exhausted(err.to_string()),
        KernelError::AgentLimitReached { .. } => Status::resource_exhausted(err.to_string()),
        KernelError::Timeout { .. } => Status::deadline_exceeded(err.to_string()),
        KernelError::NotImplemented(_) => Status::unimplemented(err.to_string()),
        _ => Status::internal(err.to_string()),
    }
}

// ════════════════════════════════════════════════════════════════════
// gRPC Service Implementation
// ════════════════════════════════════════════════════════════════════

/// The gRPC service wrapping a KernelService
pub struct LakGrpcServer {
    kernel: Arc<dyn AgentKernel>,
}

impl LakGrpcServer {
    pub fn new(kernel: Arc<dyn AgentKernel>) -> Self {
        Self { kernel }
    }

    pub fn into_service(self) -> AgentKernelServer<Self> {
        AgentKernelServer::new(self)
    }
}

#[tonic::async_trait]
impl GrpcAgentKernel for LakGrpcServer {
    // ── Agent Lifecycle ──

    async fn create_agent(
        &self,
        request: Request<pb::CreateAgentRequest>,
    ) -> Result<Response<pb::CreateAgentResponse>, Status> {
        let req = request.into_inner();
        let spec = req
            .spec
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("missing spec"))?;
        let agent_spec = agent_spec_from_pb(spec);
        let agent_id = self
            .kernel
            .create_agent(agent_spec)
            .await
            .map_err(kernel_error_to_status)?;
        Ok(Response::new(pb::CreateAgentResponse {
            agent_id: agent_id_to_bytes(agent_id),
        }))
    }

    async fn destroy_agent(
        &self,
        request: Request<pb::DestroyAgentRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();
        let agent_id = bytes_to_agent_id(&req.agent_id)?;
        self.kernel
            .destroy_agent(agent_id)
            .await
            .map_err(kernel_error_to_status)?;
        Ok(Response::new(()))
    }

    async fn get_agent(
        &self,
        request: Request<pb::GetAgentRequest>,
    ) -> Result<Response<pb::GetAgentResponse>, Status> {
        let req = request.into_inner();
        let agent_id = bytes_to_agent_id(&req.agent_id)?;
        let agent = self
            .kernel
            .get_agent(agent_id)
            .await
            .map_err(kernel_error_to_status)?;
        Ok(Response::new(pb::GetAgentResponse {
            agent: Some(agent_to_pb(&agent)),
        }))
    }

    async fn list_agents(
        &self,
        _request: Request<()>,
    ) -> Result<Response<pb::ListAgentsResponse>, Status> {
        let agents = self
            .kernel
            .list_agents()
            .await
            .map_err(kernel_error_to_status)?;
        Ok(Response::new(pb::ListAgentsResponse {
            agents: agents.iter().map(agent_to_pb).collect(),
        }))
    }

    async fn pause_agent(
        &self,
        request: Request<pb::PauseAgentRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();
        self.kernel
            .pause_agent(bytes_to_agent_id(&req.agent_id)?)
            .await
            .map_err(kernel_error_to_status)?;
        Ok(Response::new(()))
    }

    async fn resume_agent(
        &self,
        request: Request<pb::ResumeAgentRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();
        self.kernel
            .resume_agent(bytes_to_agent_id(&req.agent_id)?)
            .await
            .map_err(kernel_error_to_status)?;
        Ok(Response::new(()))
    }

    // ── Cognitive Tasks ──

    async fn submit_task(
        &self,
        request: Request<pb::SubmitTaskRequest>,
    ) -> Result<Response<pb::SubmitTaskResponse>, Status> {
        let req = request.into_inner();
        let task_pb = req
            .task
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("missing task"))?;
        let task = cognitive_task_from_pb(task_pb)?;
        let task_id = self
            .kernel
            .submit_task(task)
            .await
            .map_err(kernel_error_to_status)?;
        Ok(Response::new(pb::SubmitTaskResponse {
            task_id: task_id_to_bytes(task_id),
        }))
    }

    async fn cancel_task(
        &self,
        request: Request<pb::CancelTaskRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();
        self.kernel
            .cancel_task(bytes_to_task_id(&req.task_id)?)
            .await
            .map_err(kernel_error_to_status)?;
        Ok(Response::new(()))
    }

    async fn get_task(
        &self,
        request: Request<pb::GetTaskRequest>,
    ) -> Result<Response<pb::GetTaskResponse>, Status> {
        let req = request.into_inner();
        let task = self
            .kernel
            .get_task(bytes_to_task_id(&req.task_id)?)
            .await
            .map_err(kernel_error_to_status)?;
        Ok(Response::new(pb::GetTaskResponse {
            task: Some(cognitive_task_to_pb(&task)),
        }))
    }

    // ── Intent Routing ──

    async fn send_intent(
        &self,
        request: Request<pb::SendIntentRequest>,
    ) -> Result<Response<pb::SendIntentResponse>, Status> {
        let req = request.into_inner();
        let intent_pb = req
            .intent
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("missing intent"))?;
        let intent = intent_message_from_pb(intent_pb)?;
        let intent_id = self
            .kernel
            .send_intent(intent)
            .await
            .map_err(kernel_error_to_status)?;
        Ok(Response::new(pb::SendIntentResponse {
            intent_id: intent_id_to_bytes(intent_id),
        }))
    }

    async fn await_intent(
        &self,
        request: Request<pb::AwaitIntentRequest>,
    ) -> Result<Response<pb::AwaitIntentResponse>, Status> {
        let req = request.into_inner();
        let agent_id = bytes_to_agent_id(&req.agent_id)?;
        let sub_pb = req
            .subscription
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("missing subscription"))?;
        let subscription = intent_subscription_from_pb(sub_pb, agent_id);
        let intent = self
            .kernel
            .await_intent(agent_id, subscription)
            .await
            .map_err(kernel_error_to_status)?;
        Ok(Response::new(pb::AwaitIntentResponse {
            intent: Some(intent_message_to_pb(&intent)),
        }))
    }

    // ── Semantic Memory ──

    async fn store_memory(
        &self,
        request: Request<pb::StoreMemoryRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();
        let agent_id = bytes_to_agent_id(&req.agent_id)?;
        let chunk_pb = req
            .chunk
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("missing chunk"))?;
        let chunk = memory_chunk_from_pb(chunk_pb)?;
        self.kernel
            .store_memory(agent_id, chunk)
            .await
            .map_err(kernel_error_to_status)?;
        Ok(Response::new(()))
    }

    async fn query_memory(
        &self,
        request: Request<pb::QueryMemoryRequest>,
    ) -> Result<Response<pb::QueryMemoryResponse>, Status> {
        let req = request.into_inner();
        let agent_id = bytes_to_agent_id(&req.agent_id)?;
        let chunks = self
            .kernel
            .query_memory(agent_id, &req.query, req.top_k as usize)
            .await
            .map_err(kernel_error_to_status)?;
        Ok(Response::new(pb::QueryMemoryResponse {
            chunks: chunks.iter().map(memory_chunk_to_pb).collect(),
        }))
    }

    async fn forget_memory(
        &self,
        request: Request<pb::ForgetMemoryRequest>,
    ) -> Result<Response<pb::ForgetMemoryResponse>, Status> {
        let req = request.into_inner();
        let agent_id = bytes_to_agent_id(&req.agent_id)?;
        let chunk_id = bytes_to_memory_chunk_id(&req.chunk_id)?;
        let found = self.kernel.forget_memory(agent_id, chunk_id).await.is_ok();
        Ok(Response::new(pb::ForgetMemoryResponse { found }))
    }

    // ── Capability Management ──

    async fn grant_capability(
        &self,
        request: Request<pb::GrantCapabilityRequest>,
    ) -> Result<Response<pb::GrantCapabilityResponse>, Status> {
        let req = request.into_inner();
        let from_agent = bytes_to_agent_id(&req.from_agent)?;
        let to_agent = bytes_to_agent_id(&req.to_agent)?;
        let cap_pb = req
            .capability
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("missing capability"))?;
        let capability = Capability {
            cap_type: cap_type_from_pb(cap_pb.cap_type),
            scope: CapabilityScope {
                pattern: cap_pb.scope.clone(),
            },
            permissions: CapabilityPermission::from_bits_truncate(cap_pb.permissions),
            constraints: vec![],
        };
        let cert_id = self
            .kernel
            .grant_capability(from_agent, to_agent, capability)
            .await
            .map_err(kernel_error_to_status)?;
        Ok(Response::new(pb::GrantCapabilityResponse {
            cert_id: capability_cert_id_to_bytes(cert_id),
        }))
    }

    async fn revoke_capability(
        &self,
        request: Request<pb::RevokeCapabilityRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();
        self.kernel
            .revoke_capability(bytes_to_capability_cert_id(&req.cert_id)?)
            .await
            .map_err(kernel_error_to_status)?;
        Ok(Response::new(()))
    }

    async fn delegate_capability(
        &self,
        request: Request<pb::DelegateCapabilityRequest>,
    ) -> Result<Response<pb::DelegateCapabilityResponse>, Status> {
        let req = request.into_inner();
        let from_agent = bytes_to_agent_id(&req.from_agent)?;
        let to_agent = bytes_to_agent_id(&req.to_agent)?;
        let req_pb = req
            .requirement
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("missing requirement"))?;
        let requirement = CapabilityRequirement {
            cap_type: cap_type_from_pb(req_pb.cap_type),
            scope: req_pb.scope.clone(),
            min_permissions: CapabilityPermission::from_bits_truncate(req_pb.permissions),
        };
        let cert_id = self
            .kernel
            .delegate_capability(
                from_agent,
                to_agent,
                requirement,
                req.new_scope,
                req.new_permissions,
            )
            .await
            .map_err(kernel_error_to_status)?;
        Ok(Response::new(pb::DelegateCapabilityResponse {
            cert_id: capability_cert_id_to_bytes(cert_id),
        }))
    }

    async fn get_capabilities(
        &self,
        request: Request<pb::GetCapabilitiesRequest>,
    ) -> Result<Response<pb::GetCapabilitiesResponse>, Status> {
        let req = request.into_inner();
        let agent_id = bytes_to_agent_id(&req.agent_id)?;
        let cert = self
            .kernel
            .get_capabilities(agent_id)
            .await
            .map_err(kernel_error_to_status)?;
        Ok(Response::new(pb::GetCapabilitiesResponse {
            certificate: Some(pb::CapabilityCertificate {
                cert_id: capability_cert_id_to_bytes(cert.cert_id),
                owner_id: agent_id_to_bytes(cert.agent_id),
                capabilities: cert
                    .capabilities
                    .iter()
                    .map(|c| pb::CapabilityProto {
                        cap_type: cap_type_to_pb(&c.cap_type),
                        scope: c.scope.pattern.clone(),
                        permissions: c.permissions.bits(),
                        constraints_json: serde_json::to_string(&c.constraints).unwrap_or_default(),
                    })
                    .collect(),
                constraints: vec![],
                issued_at: Some(datetime_to_timestamp(cert.issued_at)),
                expires_at: opt_datetime_to_timestamp(cert.expires_at),
                parent_cert_id: cert.parent_cert_id.map(capability_cert_id_to_bytes),
                max_depth: None,
            }),
        }))
    }

    // ── System ──

    async fn get_system_status(
        &self,
        _request: Request<()>,
    ) -> Result<Response<pb::SystemStatus>, Status> {
        let status = self
            .kernel
            .get_system_status()
            .await
            .map_err(kernel_error_to_status)?;
        Ok(Response::new(pb::SystemStatus {
            active_agents: status.active_agents,
            max_agents: status.max_agents,
            pending_tasks: status.pending_tasks,
            completed_tasks_total: status.completed_tasks_total,
            total_tokens_consumed: status.total_tokens_consumed,
            average_coi: status.average_coi,
            scheduler_load: status.scheduler_load,
            uptime_seconds: status.uptime_seconds,
        }))
    }

    async fn shutdown(&self, _request: Request<()>) -> Result<Response<()>, Status> {
        self.kernel
            .shutdown()
            .await
            .map_err(kernel_error_to_status)?;
        Ok(Response::new(()))
    }
}
