//! MemoryStore trait — pluggable memory backends

use async_trait::async_trait;

use lak_core::types::ids::{AgentId, MemoryChunkId};
use lak_core::types::memory::MemoryChunk;

/// Backend-agnostic memory store
#[async_trait]
pub trait MemoryStore: Send + Sync {
    async fn store(&self, agent_id: AgentId, chunk: MemoryChunk) -> Result<(), String>;
    async fn query(
        &self,
        agent_id: AgentId,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<MemoryChunk>, String>;
    async fn forget(&self, agent_id: AgentId, chunk_id: MemoryChunkId) -> Result<(), String>;
}
