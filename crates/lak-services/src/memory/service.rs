//! MemoryService — stores and retrieves semantic memories

use std::collections::HashMap;

use lak_core::types::ids::{AgentId, MemoryChunkId};
use lak_core::types::memory::MemoryChunk;

/// Simple in-memory semantic memory store (MVP)
pub struct MemoryService {
    /// Memories organized by agent
    memories: HashMap<AgentId, Vec<MemoryChunk>>,
}

impl MemoryService {
    pub fn new() -> Self {
        Self {
            memories: HashMap::new(),
        }
    }

    /// Store a memory chunk for an agent
    pub fn store(&mut self, agent_id: AgentId, chunk: MemoryChunk) {
        self.memories
            .entry(agent_id)
            .or_default()
            .push(chunk);
    }

    /// Query memories by simple text matching (MVP)
    /// Phase 2: vector similarity search
    pub fn query(&self, agent_id: AgentId, query: &str, top_k: usize) -> Vec<MemoryChunk> {
        let Some(chunks) = self.memories.get(&agent_id) else {
            return vec![];
        };

        let query_lower = query.to_lowercase();
        let mut scored: Vec<(f64, MemoryChunk)> = chunks
            .iter()
            .map(|c| {
                let content_lower = c.content.raw_text.to_lowercase();
                // Simple keyword overlap score
                let score = query_lower
                    .split_whitespace()
                    .filter(|word| content_lower.contains(*word))
                    .count() as f64;
                (score, c.clone())
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        scored.truncate(top_k);
        scored.into_iter().map(|(_, c)| c).collect()
    }

    /// Forget a specific memory
    pub fn forget(
        &mut self,
        agent_id: AgentId,
        chunk_id: MemoryChunkId,
    ) -> bool {
        if let Some(chunks) = self.memories.get_mut(&agent_id) {
            if let Some(pos) = chunks.iter().position(|c| c.chunk_id == chunk_id) {
                chunks.remove(pos);
                return true;
            }
        }
        false
    }

    pub fn memory_count(&self, agent_id: AgentId) -> usize {
        self.memories
            .get(&agent_id)
            .map_or(0, |c| c.len())
    }
}

impl Default for MemoryService {
    fn default() -> Self {
        Self::new()
    }
}
