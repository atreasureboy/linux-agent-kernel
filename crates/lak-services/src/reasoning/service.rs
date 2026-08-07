//! ReasoningService — orchestrates LLM calls through the cognitive pipeline
//!
//! 5-stage cognitive pipeline:
//!   Attention → Understand → Reason → Retrieve → Integrate

use std::sync::Arc;

use futures::StreamExt;
use lak_core::types::ids::TaskId;
use lak_core::types::memory::{MemoryChunk, MemoryTier};
use lak_core::types::task::{CognitiveTask, TaskContent, TaskType};
use lak_tal::llm::{ChatMessage, ChatRole, LLMDriver, LLMRequest, LLMStreamEvent, ToolDefinition};

use super::model_router::ModelRouter;

/// Result of the cognitive pipeline execution
#[derive(Debug, Clone)]
pub struct PipelineResult {
    pub response: String,
    pub task_id: TaskId,
    pub tokens_used: u64,
    pub tool_calls: Vec<lak_tal::llm::ToolCallRequest>,
    pub memory_retrievals: Vec<MemoryChunk>,
    pub reasoning_steps: u32,
    pub llm_calls: u32,
    pub total_wall_time_ms: u64,
}

/// Result of a single raw LLM call (used in multi-turn tool loops)
#[derive(Debug, Clone)]
pub struct LlmCallResult {
    pub content: String,
    pub tool_calls: Vec<lak_tal::llm::ToolCallRequest>,
    pub tokens_used: u64,
    pub finish_reason: String,
}

/// Orchestrates LLM invocations through the 5-stage cognitive pipeline
pub struct ReasoningService {
    drivers: Vec<Arc<dyn LLMDriver>>,
    router: ModelRouter,
}

impl ReasoningService {
    pub fn new() -> Self {
        Self {
            drivers: vec![],
            router: ModelRouter::new(),
        }
    }

    /// Register an LLM backend
    pub fn add_driver(&mut self, driver: Arc<dyn LLMDriver>) {
        self.drivers.push(driver);
    }

    pub fn driver_count(&self) -> usize {
        self.drivers.len()
    }

    /// Single LLM streaming call with messages and optional tools.
    ///
    /// Used directly by the execution loop for multi-turn tool calling
    /// (where messages accumulate across tool→LLM rounds).
    pub async fn call_llm_with_messages(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<ToolDefinition>>,
        task: &CognitiveTask,
    ) -> Result<LlmCallResult, String> {
        let driver = self
            .router
            .select_driver(&self.drivers, task)
            .ok_or("No available LLM driver")?;

        let llm_request = LLMRequest {
            messages,
            tools,
            max_tokens: Some(4096),
            temperature: Some(0.7),
        };

        let mut stream = driver
            .generate_stream(llm_request)
            .await
            .map_err(|e| format!("LLM error: {e}"))?;

        let mut content = String::new();
        let mut tool_calls = Vec::new();
        let mut tokens = 0u64;
        let mut finish_reason = String::from("stop");

        while let Some(event) = stream.next().await {
            match event.map_err(|e| format!("stream error: {e}"))? {
                LLMStreamEvent::Token(t) => content.push_str(&t),
                LLMStreamEvent::ToolCall(tc) => tool_calls.push(tc),
                LLMStreamEvent::Thinking(t) => {
                    tracing::debug!(thinking = %t, "LLM reasoning");
                }
                LLMStreamEvent::Done(resp) => {
                    tokens = resp.tokens_used;
                    finish_reason = resp.finish_reason;
                    break;
                }
                LLMStreamEvent::Error(e) => {
                    return Err(format!("LLM stream error: {e}"));
                }
            }
        }

        Ok(LlmCallResult {
            content,
            tool_calls,
            tokens_used: tokens,
            finish_reason,
        })
    }

    /// Execute the full 5-stage cognitive pipeline for a task.
    ///
    /// Returns the integrated response with all execution metadata.
    pub async fn execute_pipeline(
        &self,
        task: &CognitiveTask,
        context: &str,
        memories: &[MemoryChunk],
    ) -> Result<PipelineResult, String> {
        let start = std::time::Instant::now();
        let mut stats = PipelineResult {
            response: String::new(),
            task_id: task.task_id,
            tokens_used: 0,
            tool_calls: vec![],
            memory_retrievals: vec![],
            reasoning_steps: 0,
            llm_calls: 0,
            total_wall_time_ms: 0,
        };

        // ── Stage 1: Attention — select relevant context ──
        let attended_context = Self::attention_stage(context, &task.content);
        stats.reasoning_steps += 1;

        // ── Stage 2: Understand — parse task intent, extract requirements ──
        let understanding =
            Self::understand_stage(&task.content.natural_language, &attended_context);
        stats.reasoning_steps += 1;

        // ── Stage 3: Reason — call LLM with tools ──
        let reasoning_prompt = Self::build_reasoning_prompt(
            &understanding,
            &attended_context,
            &task.content,
            task.task_type,
        );

        let driver = self
            .router
            .select_driver(&self.drivers, task)
            .ok_or("No available LLM driver")?;

        let llm_request = LLMRequest {
            messages: reasoning_prompt,
            tools: None, // Will be wired from ToolRegistry in Phase 2
            max_tokens: Some(4096),
            temperature: Some(0.7),
        };

        let mut stream = driver
            .generate_stream(llm_request)
            .await
            .map_err(|e| format!("LLM error: {e}"))?;

        let mut content = String::new();
        let mut tool_calls = Vec::new();
        let mut tokens = 0u64;

        while let Some(event) = stream.next().await {
            match event.map_err(|e| format!("stream error: {e}"))? {
                LLMStreamEvent::Token(t) => content.push_str(&t),
                LLMStreamEvent::ToolCall(tc) => tool_calls.push(tc),
                LLMStreamEvent::Thinking(t) => {
                    tracing::debug!(thinking = %t, "LLM reasoning");
                }
                LLMStreamEvent::Done(resp) => {
                    tokens += resp.tokens_used;
                    break;
                }
                LLMStreamEvent::Error(e) => {
                    return Err(format!("LLM stream error: {e}"));
                }
            }
        }
        stats.tokens_used = tokens;
        stats.tool_calls = tool_calls;
        stats.llm_calls += 1;
        stats.reasoning_steps += 1;

        // ── Stage 4: Retrieve — augment with memory ──
        let retrieved = Self::retrieve_stage(&content, memories);
        stats.memory_retrievals = retrieved.clone();
        stats.reasoning_steps += 1;

        // ── Stage 5: Integrate — combine reasoning + memories → final response ──
        stats.response = Self::integrate_stage(&content, &retrieved, &understanding);

        stats.total_wall_time_ms = start.elapsed().as_millis() as u64;
        Ok(stats)
    }

    // ─── Pipeline Stages ───

    /// Stage 1: Attention — select the most relevant context fragments.
    /// In MVP, simply truncate to keep the freshest context within token limits.
    fn attention_stage(context: &str, task_content: &TaskContent) -> String {
        // Combine task with context, prioritizing recent items
        let max_chars: usize = 8000;
        let task = task_content.natural_language.as_str();

        // A very long task leaves no room for context — keep the task alone.
        let Some(keep) = max_chars.checked_sub(task.len() + 50) else {
            return format!("Task: {task}");
        };

        if context.len() <= keep {
            return format!("Task: {task}\nContext:\n{context}");
        }

        // Keep the most recent `keep` bytes of context, snapping the cut
        // point to a UTF-8 char boundary (arbitrary byte slices panic).
        let mut start = context.len() - keep;
        while start < context.len() && !context.is_char_boundary(start) {
            start += 1;
        }
        format!("Task: {task}\nContext:\n...{}", &context[start..])
    }

    /// Stage 2: Understand — extract structured intent from natural language.
    fn understand_stage(task_text: &str, _context: &str) -> String {
        // MVP: Extract key entities and intent type via keyword analysis
        let mut understanding = String::from("Understanding:\n");
        let lower = task_text.to_lowercase();

        // Intent classification
        let intents = [
            ("find", "SEARCH", "Agent needs to locate information"),
            ("read", "READ", "Agent needs to read a file or resource"),
            ("write", "WRITE", "Agent needs to write or modify data"),
            ("create", "CREATE", "Agent needs to create a new resource"),
            ("delete", "DELETE", "Agent needs to remove a resource"),
            ("run", "EXECUTE", "Agent needs to execute a command"),
            ("analyze", "ANALYZE", "Agent needs to analyze data"),
            (
                "summarize",
                "SUMMARIZE",
                "Agent needs to condense information",
            ),
            ("translate", "TRANSLATE", "Agent needs to translate content"),
            ("compare", "COMPARE", "Agent needs to compare alternatives"),
        ];

        for (keyword, intent_type, description) in &intents {
            if lower.contains(keyword) {
                understanding.push_str(&format!("- {intent_type}: {description}\n"));
            }
        }

        if understanding == "Understanding:\n" {
            understanding.push_str("- GENERAL: General reasoning task\n");
        }

        understanding
    }

    /// Build the LLM prompt from understanding + context + task content.
    fn build_reasoning_prompt(
        understanding: &str,
        context: &str,
        task_content: &TaskContent,
        task_type: TaskType,
    ) -> Vec<ChatMessage> {
        let task_type_label = match task_type {
            TaskType::Reasoning => "reasoning",
            TaskType::ToolExecution => "tool_execution",
            TaskType::MemoryRetrieval => "memory_retrieval",
            TaskType::IntentProcessing => "intent_processing",
            TaskType::IdleReflection => "idle_reflection",
            TaskType::SystemTask => "system_task",
        };

        let system = format!(
            "You are a Linux Agent Kernel (LAK) cognitive agent.\n\
             Task type: {task_type_label}\n\
             You are operating in a capability-based security model.\n\
             Context is provided — use it to inform your reasoning.\n\
             Provide clear, actionable responses.\n\
             {understanding}"
        );

        vec![
            ChatMessage {
                role: ChatRole::System,
                content: system,
            },
            ChatMessage {
                role: ChatRole::User,
                content: format!(
                    "Context:\n{context}\n\nTask:\n{}\n\nPlease reason through this task step by step.",
                    task_content.natural_language
                ),
            },
        ]
    }

    /// Stage 4: Retrieve — augment reasoning with relevant memories.
    ///
    /// In MVP, memories are pre-fetched by the caller and scored here.
    fn retrieve_stage(reasoning: &str, memories: &[MemoryChunk]) -> Vec<MemoryChunk> {
        let lower = reasoning.to_lowercase();
        let query_terms: Vec<&str> = lower.split_whitespace().collect();

        let mut scored: Vec<(f64, MemoryChunk)> = memories
            .iter()
            .map(|m| {
                let content_lower = m.content.raw_text.to_lowercase();
                let mut score = 0f64;

                // Term frequency scoring
                for term in &query_terms {
                    if content_lower.contains(term) {
                        score += 1.0;
                    }
                }

                // Tier bonus: Working > ShortTerm > LongTerm > Archival
                let tier_bonus = match m.tier {
                    MemoryTier::Working => 2.0,
                    MemoryTier::ShortTerm => 1.5,
                    MemoryTier::LongTerm => 1.0,
                    MemoryTier::Archival => 0.5,
                };
                score *= tier_bonus;

                // Importance boost
                score *= 1.0 + m.metadata.importance_score as f64;

                (score, m.clone())
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(5); // Top 5 memories

        scored.into_iter().map(|(_, m)| m).collect()
    }

    /// Stage 5: Integrate — combine reasoning output with retrieved memories.
    fn integrate_stage(reasoning: &str, memories: &[MemoryChunk], understanding: &str) -> String {
        let mut result = format!("## Reasoning Output\n\n{reasoning}\n");

        if !memories.is_empty() {
            result.push_str("\n## Relevant Memories\n\n");
            for (i, mem) in memories.iter().enumerate() {
                let tier = match mem.tier {
                    MemoryTier::Working => "working",
                    MemoryTier::ShortTerm => "short-term",
                    MemoryTier::LongTerm => "long-term",
                    MemoryTier::Archival => "archival",
                };
                result.push_str(&format!(
                    "**Memory {i}** [{tier}, importance={:.2}]:\n{}\n\n",
                    mem.metadata.importance_score, mem.content.raw_text
                ));
            }
        }

        result.push_str("\n## Integration\n\n");
        result.push_str(&format!(
            "Based on the reasoning above and {understanding}, \
             the response combines fresh reasoning with retrieved context."
        ));

        result
    }
}

impl Default for ReasoningService {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lak_core::types::memory::{Factuality, MemoryContent, MemoryMetadata, MemorySource};

    #[test]
    fn test_attention_truncation() {
        let long_context = "x".repeat(10000);
        let task = TaskContent {
            natural_language: "Find the log entry".into(),
            structured_schema: None,
            memory_references: vec![],
        };
        let result = ReasoningService::attention_stage(&long_context, &task);
        assert!(result.contains("Find the log entry"));
        assert!(result.len() <= 8200); // Within limit
    }

    #[test]
    fn test_understand_extracts_intents() {
        let result = ReasoningService::understand_stage(
            "Please read the config file and analyze the settings",
            "",
        );
        assert!(result.contains("READ"));
        assert!(result.contains("ANALYZE"));
    }

    #[test]
    fn test_retrieve_scores_memories() {
        let mem1 = MemoryChunk {
            chunk_id: lak_core::types::ids::MemoryChunkId::new(),
            agent_id: lak_core::types::ids::AgentId::new(),
            content: MemoryContent {
                raw_text: "The config file is at /etc/app/config.toml".into(),
                structured_data: None,
                embedding: None,
            },
            metadata: MemoryMetadata {
                created_at: chrono::Utc::now(),
                last_accessed_at: chrono::Utc::now(),
                access_count: 10,
                importance_score: 0.8,
                decay_rate: 0.01,
                source: MemorySource::AgentReasoning,
                factuality: Factuality::Fact,
            },
            relations: vec![],
            tier: MemoryTier::ShortTerm,
        };

        let mem2 = MemoryChunk {
            chunk_id: lak_core::types::ids::MemoryChunkId::new(),
            agent_id: lak_core::types::ids::AgentId::new(),
            content: MemoryContent {
                raw_text: "Weather forecast: sunny".into(),
                structured_data: None,
                embedding: None,
            },
            metadata: MemoryMetadata {
                created_at: chrono::Utc::now(),
                last_accessed_at: chrono::Utc::now(),
                access_count: 1,
                importance_score: 0.1,
                decay_rate: 0.01,
                source: MemorySource::ToolOutput,
                factuality: Factuality::Belief(0.3),
            },
            relations: vec![],
            tier: MemoryTier::Archival,
        };

        let memories = vec![mem1, mem2];
        let result = ReasoningService::retrieve_stage("where is the config file", &memories);
        assert!(!result.is_empty());
        // Config-related memory should be top
        assert!(result[0].content.raw_text.contains("config"));
    }
}
