//! MemoryService — semantic memory store with TF-IDF ranking and S-Clock eviction
//!
//! Stores and retrieves semantic memories for agents. Upgraded from simple
//! keyword matching to:
//! - TF-IDF vectorization for document-term scoring
//! - Cosine similarity ranking against queries
//! - S-Clock eviction: score = recency*w1 + frequency*w2 + importance*w3 + retrieval_count*w4
//! - Memory tier promotion/demotion (Working→ShortTerm→LongTerm→Archival)

use std::collections::HashMap;

use chrono::Utc;
use lak_core::types::ids::{AgentId, MemoryChunkId};
use lak_core::types::memory::{MemoryChunk, MemoryTier};

// ── S-Clock weights ──────────────────────────────────────────────

/// Weights for the S-Clock eviction scoring formula:
///   score = recency * w1 + frequency * w2 + importance * w3 + retrieval_count * w4
///
/// Lower score → more likely to be evicted (ascending sort → evict lowest first).
#[derive(Debug, Clone)]
pub struct SClockWeights {
    /// Weight for recency (how recently was it accessed)
    pub recency: f64,
    /// Weight for access frequency
    pub frequency: f64,
    /// Weight for importance score
    pub importance: f64,
    /// Weight for retrieval count
    pub retrieval_count: f64,
}

impl Default for SClockWeights {
    fn default() -> Self {
        Self {
            recency: 0.4,
            frequency: 0.2,
            importance: 0.25,
            retrieval_count: 0.15,
        }
    }
}

// ── TF-IDF internals ─────────────────────────────────────────────

/// A simple in-process TF-IDF index.
///
/// Tracks document frequencies across the corpus and supports
/// cosine-similarity ranking against a query.
#[derive(Debug, Default)]
struct TfIdfIndex {
    /// Number of documents in the corpus
    doc_count: usize,
    /// Document frequency: term → how many documents contain it
    doc_freq: HashMap<String, usize>,
    /// Per-document term-frequency vectors (chunk_id → (term → count))
    term_vectors: HashMap<MemoryChunkId, HashMap<String, usize>>,
}

impl TfIdfIndex {
    fn new() -> Self {
        Self::default()
    }

    /// Tokenize text into lowercase alphanumeric terms (min 2 chars)
    fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| t.len() >= 2)
            .map(String::from)
            .collect()
    }

    /// Build a term-frequency map from token list
    fn term_freq(tokens: &[String]) -> HashMap<String, usize> {
        let mut tf = HashMap::new();
        for token in tokens {
            *tf.entry(token.clone()).or_default() += 1;
        }
        tf
    }

    /// Add a document to the index
    fn add(&mut self, chunk_id: MemoryChunkId, text: &str) {
        let tokens = Self::tokenize(text);
        let tf = Self::term_freq(&tokens);

        // Update document frequencies
        for term in tf.keys() {
            *self.doc_freq.entry(term.clone()).or_default() += 1;
        }

        self.term_vectors.insert(chunk_id, tf);
        self.doc_count += 1;
    }

    /// Remove a document from the index
    fn remove(&mut self, chunk_id: MemoryChunkId) {
        if let Some(tf) = self.term_vectors.remove(&chunk_id) {
            self.doc_count = self.doc_count.saturating_sub(1);
            for term in tf.keys() {
                if let Some(count) = self.doc_freq.get_mut(term) {
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        self.doc_freq.remove(term);
                    }
                }
            }
        }
    }

    /// Compute TF-IDF weights for a query against all indexed documents.
    ///
    /// Returns (chunk_id, cosine_similarity) sorted descending.
    fn rank(&self, query: &str, top_k: usize) -> Vec<(MemoryChunkId, f64)> {
        let query_tokens = Self::tokenize(query);
        let query_tf = Self::term_freq(&query_tokens);

        // Compute IDF for each query term
        let query_vec: HashMap<String, f64> = query_tf
            .iter()
            .map(|(term, &tf_count)| {
                let df = self.doc_freq.get(term).copied().unwrap_or(0) as f64;
                // Smooth IDF: log((doc_count + 1) / (df + 1)) + 1
                let idf = ((self.doc_count as f64 + 1.0) / (df + 1.0)).ln() + 1.0;
                (term.clone(), tf_count as f64 * idf)
            })
            .collect();

        let query_norm: f64 = query_vec.values().map(|v| v * v).sum::<f64>().sqrt();

        if query_norm == 0.0 {
            return Vec::new();
        }

        let mut scored: Vec<(MemoryChunkId, f64)> = self
            .term_vectors
            .iter()
            .filter_map(|(&chunk_id, doc_tf)| {
                // Compute dot product
                let dot_product: f64 = doc_tf
                    .iter()
                    .map(|(term, &tf_count)| {
                        let doc_tf = tf_count as f64;
                        let df = self.doc_freq.get(term).copied().unwrap_or(0) as f64;
                        let idf = ((self.doc_count as f64 + 1.0) / (df + 1.0)).ln() + 1.0;
                        let doc_weight = doc_tf * idf;
                        doc_weight * query_vec.get(term).copied().unwrap_or(0.0)
                    })
                    .sum();

                // Compute doc norm
                let doc_norm: f64 = doc_tf
                    .iter()
                    .map(|(term, &tf_count)| {
                        let doc_tf = tf_count as f64;
                        let df = self.doc_freq.get(term).copied().unwrap_or(0) as f64;
                        let idf = ((self.doc_count as f64 + 1.0) / (df + 1.0)).ln() + 1.0;
                        let w = doc_tf * idf;
                        w * w
                    })
                    .sum::<f64>()
                    .sqrt();

                if doc_norm == 0.0 {
                    return None;
                }

                let cosine = dot_product / (query_norm * doc_norm);
                if cosine > 0.0 {
                    Some((chunk_id, cosine))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }
}

// ── S-Clock eviction ─────────────────────────────────────────────

/// Compute the S-Clock retention score for a memory chunk.
///
/// Higher score → more valuable → keep. Lower score → eviction candidate.
/// Formula: recency*w1 + frequency*w2 + importance*w3 + retrieval_count*w4
fn s_clock_score(chunk: &MemoryChunk, weights: &SClockWeights) -> f64 {
    let now = Utc::now();

    // Recency: normalized 0.0–1.0 (1.0 = accessed just now, decays over 7 days)
    let age_secs = (now - chunk.metadata.last_accessed_at).num_seconds().max(0) as f64;
    let seven_days = 7.0 * 24.0 * 3600.0;
    let recency = 1.0 / (1.0 + age_secs / seven_days); // 0→1 after one week

    // Frequency: total accesses normalized (log scale, max ~1000 accesses)
    let frequency = ((chunk.metadata.access_count as f64 + 1.0).ln() / 7.0).min(1.0);

    // Importance: directly from metadata (0.0–1.0)
    let importance = chunk.metadata.importance_score as f64;

    // Retrieval count: normalized same as frequency (capped at 1.0)
    let retrieval_count = ((chunk.metadata.access_count as f64 + 1.0).ln() / 7.0).min(1.0);

    recency * weights.recency
        + frequency * weights.frequency
        + importance * weights.importance
        + retrieval_count * weights.retrieval_count
}

// ── Tier configuration ───────────────────────────────────────────

/// Thresholds for memory tier promotion/demotion
#[derive(Debug, Clone)]
pub struct TierThresholds {
    /// Number of accesses to promote from Working to ShortTerm
    pub working_to_short_term_hits: u64,
    /// Number of accesses to promote from ShortTerm to LongTerm
    pub short_term_to_long_term_hits: u64,
    /// Minimum importance to keep in LongTerm
    pub long_term_min_importance: f32,
    /// Maximum items per tier before eviction
    pub max_working: usize,
    pub max_short_term: usize,
    pub max_long_term: usize,
}

impl Default for TierThresholds {
    fn default() -> Self {
        Self {
            working_to_short_term_hits: 3,
            short_term_to_long_term_hits: 10,
            long_term_min_importance: 0.3,
            max_working: 50,
            max_short_term: 200,
            max_long_term: 1000,
        }
    }
}

// ── Per-agent memory store ───────────────────────────────────────

/// Memory store for a single agent
struct AgentMemoryStore {
    /// All memories indexed by chunk_id
    chunks: HashMap<MemoryChunkId, MemoryChunk>,
    /// TF-IDF index for semantic search
    index: TfIdfIndex,
    /// S-Clock weights for this agent
    clock_weights: SClockWeights,
    /// Tier thresholds
    thresholds: TierThresholds,
}

impl AgentMemoryStore {
    fn new() -> Self {
        Self {
            chunks: HashMap::new(),
            index: TfIdfIndex::new(),
            clock_weights: SClockWeights::default(),
            thresholds: TierThresholds::default(),
        }
    }

    fn store(&mut self, chunk: MemoryChunk) {
        self.index.add(chunk.chunk_id, &chunk.content.raw_text);
        self.chunks.insert(chunk.chunk_id, chunk);
        self.enforce_tier_limits();
    }

    fn query(&mut self, query: &str, top_k: usize) -> Vec<MemoryChunk> {
        let ranked = self.index.rank(query, top_k);

        // First pass: update access metadata, decide promotions, compute
        // the final blended score for each candidate.
        let mut scored: Vec<(f64, MemoryChunkId)> = Vec::new();
        for (chunk_id, tfidf_score) in ranked {
            let Some(chunk) = self.chunks.get_mut(&chunk_id) else {
                continue;
            };

            chunk.metadata.last_accessed_at = Utc::now();
            chunk.metadata.access_count += 1;

            let tier_bonus = match chunk.tier {
                MemoryTier::Working => 0.1,
                MemoryTier::ShortTerm => 0.2,
                MemoryTier::LongTerm => 0.3,
                MemoryTier::Archival => 0.0,
            };
            let importance = chunk.metadata.importance_score;

            // Promote frequently-accessed chunks up the tier ladder
            let new_tier = match chunk.tier {
                MemoryTier::Working
                    if chunk.metadata.access_count
                        >= self.thresholds.working_to_short_term_hits =>
                {
                    Some(MemoryTier::ShortTerm)
                }
                MemoryTier::ShortTerm
                    if chunk.metadata.access_count
                        >= self.thresholds.short_term_to_long_term_hits =>
                {
                    Some(MemoryTier::LongTerm)
                }
                _ => None,
            };
            if let Some(tier) = new_tier {
                chunk.tier = tier;
            }

            let combined = tfidf_score * 0.6 + tier_bonus + importance as f64 * 0.2;
            scored.push((combined, chunk_id));
        }

        // Re-rank by the *final* blended score (TF-IDF alone would ignore
        // the tier/importance bonuses we just applied).
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .filter_map(|(combined, chunk_id)| {
                let mut result = self.chunks.get(&chunk_id)?.clone();
                result.metadata.importance_score = (combined as f32).clamp(0.0, 1.0);
                Some(result)
            })
            .collect()
    }

    fn forget(&mut self, chunk_id: MemoryChunkId) -> bool {
        self.index.remove(chunk_id);
        self.chunks.remove(&chunk_id).is_some()
    }

    fn memory_count(&self) -> usize {
        self.chunks.len()
    }

    /// Enforce tier capacity limits using S-Clock eviction
    fn enforce_tier_limits(&mut self) {
        for (tier, max) in [
            (MemoryTier::Working, self.thresholds.max_working),
            (MemoryTier::ShortTerm, self.thresholds.max_short_term),
            (MemoryTier::LongTerm, self.thresholds.max_long_term),
        ] {
            let tier_chunks: Vec<MemoryChunkId> = self
                .chunks
                .values()
                .filter(|c| c.tier == tier)
                .map(|c| c.chunk_id)
                .collect();

            if tier_chunks.len() <= max {
                continue;
            }

            // Score all chunks in this tier, evict lowest-scoring ones
            let excess = tier_chunks.len() - max;
            let mut scored: Vec<(MemoryChunkId, f64)> = tier_chunks
                .iter()
                .filter_map(|&id| {
                    self.chunks
                        .get(&id)
                        .map(|c| (id, s_clock_score(c, &self.clock_weights)))
                })
                .collect();

            // Sort ascending: lowest score → most evictable
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Evict lowest-scoring (demote to Archival or forget)
            for (id, _score) in scored.iter().take(excess) {
                if let Some(chunk) = self.chunks.get_mut(id) {
                    // Demote to Archival rather than fully deleting
                    chunk.tier = MemoryTier::Archival;
                }
            }
        }
    }
}

// ── MemoryService ────────────────────────────────────────────────

/// Semantic memory store with TF-IDF ranking and S-Clock eviction
pub struct MemoryService {
    /// Memories organized by agent
    stores: HashMap<AgentId, AgentMemoryStore>,
}

impl MemoryService {
    pub fn new() -> Self {
        Self {
            stores: HashMap::new(),
        }
    }

    /// Store a memory chunk for an agent
    pub fn store(&mut self, agent_id: AgentId, chunk: MemoryChunk) {
        self.stores
            .entry(agent_id)
            .or_insert_with(AgentMemoryStore::new)
            .store(chunk);
    }

    /// Query memories using TF-IDF semantic search
    ///
    /// Ranks memories by cosine similarity between the query vector
    /// and stored document vectors, then blends with tier bonus and
    /// importance score.
    pub fn query(&mut self, agent_id: AgentId, query: &str, top_k: usize) -> Vec<MemoryChunk> {
        self.stores
            .get_mut(&agent_id)
            .map(|store| store.query(query, top_k))
            .unwrap_or_default()
    }

    /// Forget a specific memory
    pub fn forget(&mut self, agent_id: AgentId, chunk_id: MemoryChunkId) -> bool {
        self.stores
            .get_mut(&agent_id)
            .map(|store| store.forget(chunk_id))
            .unwrap_or(false)
    }

    /// Total memory count for an agent
    pub fn memory_count(&self, agent_id: AgentId) -> usize {
        self.stores
            .get(&agent_id)
            .map(|s| s.memory_count())
            .unwrap_or(0)
    }

    /// Set S-Clock weights for an agent (tunes eviction behavior)
    pub fn set_sclock_weights(&mut self, agent_id: AgentId, weights: SClockWeights) {
        if let Some(store) = self.stores.get_mut(&agent_id) {
            store.clock_weights = weights;
        }
    }

    /// Set tier thresholds for an agent
    pub fn set_tier_thresholds(&mut self, agent_id: AgentId, thresholds: TierThresholds) {
        if let Some(store) = self.stores.get_mut(&agent_id) {
            store.thresholds = thresholds;
        }
    }
}

impl Default for MemoryService {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use lak_core::types::memory::{MemoryContent, MemoryMetadata};

    fn make_chunk(_id: u64, text: &str, importance: f32) -> MemoryChunk {
        MemoryChunk {
            chunk_id: MemoryChunkId::new(),
            agent_id: AgentId::new(),
            content: MemoryContent {
                raw_text: text.to_string(),
                structured_data: None,
                embedding: None,
            },
            metadata: MemoryMetadata {
                created_at: Utc::now(),
                last_accessed_at: Utc::now(),
                access_count: 0,
                importance_score: importance,
                decay_rate: 0.01,
                source: lak_core::types::memory::MemorySource::UserInput,
                factuality: lak_core::types::memory::Factuality::Belief(0.9),
            },
            relations: vec![],
            tier: MemoryTier::Working,
        }
    }

    #[test]
    fn test_semantic_search_ranks_by_relevance() {
        let mut service = MemoryService::new();
        let agent = AgentId::new();

        service.store(
            agent,
            make_chunk(1, "Rust programming language systems development", 0.5),
        );
        service.store(
            agent,
            make_chunk(2, "Python for machine learning and data science", 0.5),
        );
        service.store(
            agent,
            make_chunk(3, "Rust memory safety and ownership model", 0.5),
        );

        let results = service.query(agent, "Rust systems programming", 3);
        // Python chunk has zero semantic overlap with "Rust systems programming"
        // so it correctly gets excluded by the cosine similarity filter
        assert_eq!(results.len(), 2);
        // The Rust-related chunks should rank higher than Python
        assert!(
            results[0].content.raw_text.contains("Rust"),
            "First result should be Rust-related, got: {}",
            results[0].content.raw_text
        );
    }

    #[test]
    fn test_semantic_search_weights_importance() {
        let mut service = MemoryService::new();
        let agent = AgentId::new();

        service.store(
            agent,
            make_chunk(1, "important security vulnerability found", 0.95),
        );
        service.store(agent, make_chunk(2, "random note about lunch", 0.1));

        let results = service.query(agent, "security issue", 2);
        // High-importance security memory should rank first
        assert!(
            results[0].content.raw_text.contains("security"),
            "Security memory should rank first"
        );
    }

    #[test]
    fn test_tfidf_distinguishes_relevant() {
        let mut service = MemoryService::new();
        let agent = AgentId::new();

        service.store(
            agent,
            make_chunk(1, "database connection pooling postgres", 0.5),
        );
        service.store(
            agent,
            make_chunk(2, "frontend react component styling", 0.5),
        );

        let results = service.query(agent, "postgres db pool", 2);
        // The database chunk should score higher
        assert!(
            results[0].content.raw_text.contains("database"),
            "Database chunk should rank first for db query"
        );
    }

    #[test]
    fn test_tier_promotion() {
        let mut store = AgentMemoryStore::new();

        let mut chunk = make_chunk(1, "important system configuration", 0.5);
        chunk.tier = MemoryTier::Working;
        chunk.metadata.access_count = 0;
        let chunk_id = chunk.chunk_id;
        store.store(chunk);

        // Simulate repeated accesses
        for _ in 0..5 {
            let _ = store.query("system config", 5);
        }

        // After 5 accesses, should be promoted to ShortTerm
        if let Some(c) = store.chunks.get(&chunk_id) {
            assert_eq!(c.tier, MemoryTier::ShortTerm);
        } else {
            panic!("Chunk should exist");
        }
    }

    #[test]
    fn test_sclock_eviction_when_over_limit() {
        let mut store = AgentMemoryStore::new();
        store.thresholds.max_working = 3;

        for i in 0..5 {
            let mut chunk = make_chunk(i, &format!("memory entry {i}"), 0.1 + i as f32 * 0.05);
            chunk.tier = MemoryTier::Working;
            // Stagger access times so recency differs
            chunk.metadata.last_accessed_at =
                Utc::now() - chrono::Duration::seconds(i as i64 * 100);
            store.store(chunk);
        }

        // Should have evicted (demoted to Archival) some working entries
        let archival_count = store
            .chunks
            .values()
            .filter(|c| c.tier == MemoryTier::Archival)
            .count();

        let working_count = store
            .chunks
            .values()
            .filter(|c| c.tier == MemoryTier::Working)
            .count();

        assert!(working_count <= 3, "Working should be capped at 3");
        assert!(archival_count > 0, "Some should be demoted to Archival");
    }

    #[test]
    fn test_forget_removes_from_index() {
        let mut service = MemoryService::new();
        let agent = AgentId::new();

        let chunk = make_chunk(42, "unique term that only appears here", 0.5);
        let chunk_id = chunk.chunk_id;
        service.store(agent, chunk);
        assert_eq!(service.memory_count(agent), 1);

        service.forget(agent, chunk_id);
        assert_eq!(service.memory_count(agent), 0);

        // Querying should return nothing
        let results = service.query(agent, "unique term", 5);
        assert!(results.is_empty());
    }
}
