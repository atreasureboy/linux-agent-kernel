//! IntentRouter — pub/sub routing for agent intents
//!
//! Routes intent messages between agents using topic-pattern-based subscription
//! trees (prefix matching like MQTT) and capability-based semantic routing.
//! Undeliverable messages land in a dead-letter queue for inspection/replay.

use std::collections::HashMap;

use chrono::Utc;
use lak_core::types::capability::CapabilityType;
use lak_core::types::ids::{AgentId, IntentId};
use lak_core::types::intent::{IntentMessage, IntentSubscription, IntentTarget};

#[cfg(test)]
use lak_core::types::intent::IntentType;

// ── Subscription management ──────────────────────────────────────

/// A stored subscription entry
#[derive(Debug, Clone)]
struct SubscriptionEntry {
    subscription: IntentSubscription,
}

/// Result of a publish operation
#[derive(Debug, Clone)]
pub struct PublishResult {
    /// Intent ID that was published
    pub intent_id: IntentId,
    /// Agent IDs that received the message
    pub delivered_to: Vec<AgentId>,
    /// Whether the intent was dead-lettered
    pub dead_lettered: bool,
    /// Reason for dead-lettering (if any)
    pub dead_letter_reason: Option<String>,
}

/// The intent pub/sub router
#[derive(Debug)]
pub struct IntentRouter {
    /// Active subscriptions by agent
    subscriptions: HashMap<AgentId, Vec<SubscriptionEntry>>,
    /// Dead-letter queue for undeliverable intents
    dead_letters: Vec<DeadLetterEntry>,
    /// Maximum dead-letter queue size
    max_dead_letters: usize,
    /// Total messages published
    total_published: u64,
    /// Total messages delivered
    total_delivered: u64,
    /// Total messages dead-lettered
    total_dead_lettered: u64,
}

/// An intent that could not be delivered to any subscriber
#[derive(Debug, Clone)]
pub struct DeadLetterEntry {
    intent: IntentMessage,
    reason: String,
    enqueued_at: chrono::DateTime<Utc>,
}

impl DeadLetterEntry {
    /// The undelivered intent
    pub fn intent(&self) -> &IntentMessage {
        &self.intent
    }

    /// Why delivery failed
    pub fn reason(&self) -> &str {
        &self.reason
    }

    /// When the intent was dead-lettered
    pub fn enqueued_at(&self) -> chrono::DateTime<Utc> {
        self.enqueued_at
    }
}

impl IntentRouter {
    /// Create a new intent router
    pub fn new() -> Self {
        Self {
            subscriptions: HashMap::new(),
            dead_letters: Vec::new(),
            max_dead_letters: 1000,
            total_published: 0,
            total_delivered: 0,
            total_dead_lettered: 0,
        }
    }

    /// Set maximum dead-letter queue size
    pub fn with_max_dead_letters(mut self, max: usize) -> Self {
        self.max_dead_letters = max;
        self
    }

    /// Subscribe an agent to matching intents.
    ///
    /// An agent can have multiple subscriptions. Each subscription can
    /// filter by intent type, topic pattern, or capability requirement.
    pub fn subscribe(&mut self, subscription: IntentSubscription) {
        let agent_id = subscription.agent_id;
        self.subscriptions
            .entry(agent_id)
            .or_default()
            .push(SubscriptionEntry { subscription });
    }

    /// Subscribe unless an identical subscription already exists.
    /// Callers polling `await_intent` must not accumulate duplicates.
    pub fn subscribe_once(&mut self, subscription: IntentSubscription) {
        let agent_id = subscription.agent_id;
        let subs = self.subscriptions.entry(agent_id).or_default();
        let exists = subs.iter().any(|e| {
            e.subscription.topic_pattern == subscription.topic_pattern
                && e.subscription.intent_types == subscription.intent_types
                && e.subscription.capability_filter == subscription.capability_filter
        });
        if !exists {
            subs.push(SubscriptionEntry { subscription });
        }
    }

    /// Check whether a subscription matches an intent (public so that the
    /// kernel can replay dead letters against late-arriving subscriptions).
    pub fn matches(subscription: &IntentSubscription, intent: &IntentMessage) -> bool {
        Self::check_sub_match(subscription, intent)
    }

    /// Unsubscribe an agent from all subscriptions, or a specific pattern
    pub fn unsubscribe(&mut self, agent_id: AgentId, pattern: Option<&str>) -> usize {
        let mut removed = 0;
        if let Some(subs) = self.subscriptions.get_mut(&agent_id) {
            if let Some(pat) = pattern {
                subs.retain(|entry| {
                    let matches = entry.subscription.topic_pattern.as_deref() == Some(pat);
                    if matches {
                        removed += 1;
                    }
                    !matches
                });
            } else {
                removed = subs.len();
                subs.clear();
            }
        }
        removed
    }

    /// Publish an intent, routing to all matching subscribers.
    ///
    /// Resolution order:
    /// 1. Unicast: deliver to exact target if subscribed (or anyone if no filter)
    /// 2. Multicast: deliver to each listed agent
    /// 3. ByCapability: deliver to agents whose capabilities match
    /// 4. PublishSubscribe: deliver to agents with matching topic patterns
    /// 5. Broadcast: deliver to ALL subscribed agents
    ///
    /// If no subscribers match, the intent goes to the dead-letter queue.
    pub fn publish(&mut self, intent: IntentMessage) -> PublishResult {
        self.total_published += 1;

        let delivered = match &intent.target {
            IntentTarget::Unicast(target_id) => self.deliver_to_agent(&intent, &[*target_id]),
            IntentTarget::Multicast(agent_ids) => self.deliver_to_agent(&intent, agent_ids),
            IntentTarget::Broadcast => {
                let all: Vec<AgentId> = self.subscriptions.keys().copied().collect();
                self.deliver_to_agent(&intent, &all)
            }
            IntentTarget::ByCapability {
                cap_type,
                semantic_hint: _,
            } => {
                let matching = self.find_by_capability(cap_type.clone());
                self.deliver_to_agent(&intent, &matching)
            }
            IntentTarget::PublishSubscribe { pattern } => {
                let matching = self.find_by_topic_pattern(pattern);
                self.deliver_to_agent(&intent, &matching)
            }
        };

        let successful = !delivered.is_empty();
        self.total_delivered += delivered.len() as u64;

        if !successful {
            self.total_dead_lettered += 1;
            // Enforce dead-letter limit
            while self.dead_letters.len() >= self.max_dead_letters {
                self.dead_letters.remove(0);
            }
            self.dead_letters.push(DeadLetterEntry {
                intent: intent.clone(),
                reason: format!("No matching subscribers for target: {:?}", intent.target),
                enqueued_at: Utc::now(),
            });
        }

        PublishResult {
            intent_id: intent.intent_id,
            delivered_to: delivered,
            dead_lettered: !successful,
            dead_letter_reason: if successful {
                None
            } else {
                Some("No matching subscribers".into())
            },
        }
    }

    /// Get all dead-lettered intents
    pub fn dead_letters(&self) -> &[DeadLetterEntry] {
        &self.dead_letters
    }

    /// Requeue a dead-lettered intent by ID (remove from dead-letter queue,
    /// caller should re-publish)
    pub fn requeue_dead_letter(&mut self, intent_id: IntentId) -> Option<IntentMessage> {
        if let Some(pos) = self
            .dead_letters
            .iter()
            .position(|e| e.intent.intent_id == intent_id)
        {
            let entry = self.dead_letters.remove(pos);
            Some(entry.intent)
        } else {
            None
        }
    }

    /// Clear all dead letters
    pub fn clear_dead_letters(&mut self) -> usize {
        let count = self.dead_letters.len();
        self.dead_letters.clear();
        count
    }

    /// Number of active subscriptions
    pub fn subscription_count(&self) -> usize {
        self.subscriptions.values().map(|v| v.len()).sum()
    }

    /// Total messages published
    pub fn total_published(&self) -> u64 {
        self.total_published
    }

    /// Total messages delivered
    pub fn total_delivered(&self) -> u64 {
        self.total_delivered
    }

    /// Total messages dead-lettered
    pub fn total_dead_lettered(&self) -> u64 {
        self.total_dead_lettered
    }

    /// Dead-letter queue size
    pub fn dead_letter_count(&self) -> usize {
        self.dead_letters.len()
    }

    // ── Private helpers ──────────────────────────────────────────

    /// Find agents whose subscriptions match the given topic pattern
    fn find_by_topic_pattern(&self, pattern: &str) -> Vec<AgentId> {
        let mut matches = Vec::new();
        for (&agent_id, subs) in &self.subscriptions {
            for entry in subs {
                if let Some(ref sub_pattern) = entry.subscription.topic_pattern {
                    if topic_matches(sub_pattern, pattern) {
                        matches.push(agent_id);
                        break;
                    }
                }
            }
        }
        matches
    }

    /// Find agents whose subscriptions include the given capability type
    fn find_by_capability(&self, cap_type: CapabilityType) -> Vec<AgentId> {
        let mut matches = Vec::new();
        for (&agent_id, subs) in &self.subscriptions {
            for entry in subs {
                if let Some(ref filter) = entry.subscription.capability_filter {
                    if *filter == cap_type {
                        matches.push(agent_id);
                        break;
                    }
                }
            }
        }
        matches
    }

    /// Deliver intent to a list of agents, filtering by subscription criteria
    fn deliver_to_agent(&self, intent: &IntentMessage, candidates: &[AgentId]) -> Vec<AgentId> {
        let mut delivered = Vec::new();

        for &agent_id in candidates {
            // Don't deliver to self
            if agent_id == intent.source_agent_id {
                continue;
            }

            let has_match = self
                .subscriptions
                .get(&agent_id)
                .map(|subs| {
                    subs.iter()
                        .any(|entry| Self::check_sub_match(&entry.subscription, intent))
                })
                .unwrap_or(false);

            if has_match {
                delivered.push(agent_id);
            }
        }

        delivered
    }

    /// Shared matching logic (no self borrow needed)
    fn check_sub_match(sub: &IntentSubscription, intent: &IntentMessage) -> bool {
        // Check intent type filter
        if let Some(ref types) = sub.intent_types {
            if !types.is_empty() && !types.contains(&intent.intent_type) {
                return false;
            }
        }

        // Check topic pattern filter:
        // - For PublishSubscribe intents, match against the published topic
        // - For all others, match against the content's natural language
        if let Some(ref pattern) = sub.topic_pattern {
            let match_text = match &intent.target {
                IntentTarget::PublishSubscribe { pattern: topic } => topic.as_str(),
                _ => intent.content.natural_language.as_str(),
            };
            if !topic_matches(pattern, match_text) {
                return false;
            }
        }

        // Check capability filter
        if let Some(ref cap_filter) = sub.capability_filter {
            if let IntentTarget::ByCapability {
                cap_type,
                semantic_hint: _,
            } = &intent.target
            {
                if *cap_type != *cap_filter {
                    return false;
                }
            } else {
                return false;
            }
        }

        true
    }
}

impl Default for IntentRouter {
    fn default() -> Self {
        Self::new()
    }
}

// ── Topic pattern matching ───────────────────────────────────────

/// Match a subscription pattern against a topic string.
///
/// Supports MQTT-style single-level wildcard `*` (matches exactly one
/// segment) and multi-level wildcard `**` (matches zero or more segments).
/// Segments are delimited by `/`.
///
/// # Examples
/// - `topic_matches("sys/agent/*", "sys/agent/create")` → true
/// - `topic_matches("sys/agent/*", "sys/agent/create/child")` → false
/// - `topic_matches("sys/**", "sys/agent/create/task")` → true
/// - `topic_matches("security/**", "security/intrusion")` → true
fn topic_matches(pattern: &str, topic: &str) -> bool {
    if pattern == topic {
        return true;
    }

    // If it has wildcard chars, do MQTT-style matching
    if pattern.contains('*') {
        return mqtt_wildcard_match(pattern, topic);
    }

    // Simple substring match (case-insensitive containment)
    pattern.to_lowercase().contains(&topic.to_lowercase())
        || topic.to_lowercase().contains(&pattern.to_lowercase())
}

/// MQTT-style wildcard matching with `*` and `**`
fn mqtt_wildcard_match(pattern: &str, topic: &str) -> bool {
    let pattern_segments: Vec<&str> = pattern.split('/').collect();
    let topic_segments: Vec<&str> = topic.split('/').collect();

    mqtt_match_segments(&pattern_segments, &topic_segments, 0, 0)
}

fn mqtt_match_segments(pattern: &[&str], topic: &[&str], pi: usize, ti: usize) -> bool {
    if pi == pattern.len() {
        return ti == topic.len();
    }

    let seg = pattern[pi];

    if seg == "**" {
        // Multi-level wildcard: match zero or more remaining topic segments
        for skip in 0..=(topic.len() - ti) {
            if mqtt_match_segments(pattern, topic, pi + 1, ti + skip) {
                return true;
            }
        }
        false
    } else if seg == "*" {
        // Single-level wildcard: match exactly one topic segment
        if ti >= topic.len() {
            return false;
        }
        mqtt_match_segments(pattern, topic, pi + 1, ti + 1)
    } else {
        // Literal segment: case-insensitive match
        if ti >= topic.len() {
            return false;
        }
        if seg.to_lowercase() != topic[ti].to_lowercase() {
            return false;
        }
        mqtt_match_segments(pattern, topic, pi + 1, ti + 1)
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use lak_core::types::intent::IntentContent;
    use lak_core::types::task::CognitivePriority;

    fn make_intent(
        _id: u64,
        source: AgentId,
        target: IntentTarget,
        content: &str,
    ) -> IntentMessage {
        IntentMessage {
            intent_id: IntentId::new(),
            source_agent_id: source,
            target,
            intent_type: IntentType::Inform,
            content: IntentContent {
                natural_language: content.to_string(),
                structured_data: None,
                memory_references: vec![],
            },
            priority: CognitivePriority::normal(),
            ttl_ms: 30_000,
            correlation_id: None,
            created_at: Utc::now(),
        }
    }

    fn make_sub(
        agent_id: AgentId,
        types: Option<Vec<IntentType>>,
        pattern: Option<&str>,
        cap_filter: Option<CapabilityType>,
    ) -> IntentSubscription {
        IntentSubscription {
            agent_id,
            intent_types: types,
            topic_pattern: pattern.map(String::from),
            capability_filter: cap_filter,
        }
    }

    #[test]
    fn test_topic_wildcard_single_level() {
        assert!(topic_matches("sys/agent/*", "sys/agent/create"));
        assert!(!topic_matches("sys/agent/*", "sys/agent/create/child"));
    }

    #[test]
    fn test_topic_wildcard_multi_level() {
        assert!(topic_matches("sys/**", "sys/agent/create/task"));
        assert!(topic_matches("security/**", "security/intrusion"));
        assert!(topic_matches("**", "anything/goes/here"));
    }

    #[test]
    fn test_topic_simple_substring() {
        assert!(topic_matches("security", "security alert detected"));
        assert!(topic_matches("alert", "security alert detected"));
    }

    #[test]
    fn test_publish_subscribe_matches_topic() {
        let mut router = IntentRouter::new();
        let agent_a = AgentId::new();
        let agent_b = AgentId::new();

        router.subscribe(make_sub(agent_a, None, Some("sys/agent/*"), None));
        router.subscribe(make_sub(agent_b, None, Some("data/**"), None));

        let intent = make_intent(
            1,
            AgentId::new(),
            IntentTarget::PublishSubscribe {
                pattern: "sys/agent/create".into(),
            },
            "create new agent",
        );

        let result = router.publish(intent);
        assert!(result.delivered_to.contains(&agent_a));
        assert!(!result.delivered_to.contains(&agent_b));
        assert!(!result.dead_lettered);
    }

    #[test]
    fn test_broadcast_delivers_to_all() {
        let mut router = IntentRouter::new();
        let a = AgentId::new();
        let b = AgentId::new();
        let source = AgentId::new();

        router.subscribe(make_sub(a, None, None, None));
        router.subscribe(make_sub(b, None, None, None));

        let intent = make_intent(2, source, IntentTarget::Broadcast, "system announcement");

        let result = router.publish(intent);
        assert!(result.delivered_to.contains(&a));
        assert!(result.delivered_to.contains(&b));
        // Source should not receive its own broadcast
        assert!(!result.delivered_to.contains(&source));
    }

    #[test]
    fn test_dead_letter_when_no_match() {
        let mut router = IntentRouter::new();
        let a = AgentId::new();
        router.subscribe(make_sub(a, Some(vec![IntentType::Query]), None, None));

        let intent = make_intent(
            3,
            AgentId::new(),
            IntentTarget::PublishSubscribe {
                pattern: "unknown/topic".into(),
            },
            "nobody listens",
        );

        let result = router.publish(intent);
        assert!(result.dead_lettered);
        assert!(result.delivered_to.is_empty());
        assert_eq!(router.dead_letter_count(), 1);
    }

    #[test]
    fn test_unsubscribe_removes_agent() {
        let mut router = IntentRouter::new();
        let agent = AgentId::new();

        router.subscribe(make_sub(agent, None, Some("test/*"), None));
        assert_eq!(router.subscription_count(), 1);

        router.unsubscribe(agent, None);
        assert_eq!(router.subscription_count(), 0);
    }

    #[test]
    fn test_intent_type_filtering() {
        let mut router = IntentRouter::new();
        let agent = AgentId::new();

        router.subscribe(make_sub(agent, Some(vec![IntentType::Query]), None, None));

        // Send an Inform (doesn't match Query subscription)
        let intent = make_intent(4, AgentId::new(), IntentTarget::Broadcast, "some info");
        let result = router.publish(intent);
        assert!(result.dead_lettered);
    }

    #[test]
    fn test_requeue_dead_letter() {
        let mut router = IntentRouter::new();
        let intent_id = IntentId::new();

        let intent = IntentMessage {
            intent_id,
            source_agent_id: AgentId::new(),
            target: IntentTarget::Broadcast,
            intent_type: IntentType::Inform,
            content: IntentContent {
                natural_language: "orphaned message".into(),
                structured_data: None,
                memory_references: vec![],
            },
            priority: CognitivePriority::normal(),
            ttl_ms: 30_000,
            correlation_id: None,
            created_at: Utc::now(),
        };

        router.publish(intent);
        assert_eq!(router.dead_letter_count(), 1);

        let requeued = router.requeue_dead_letter(intent_id);
        assert!(requeued.is_some());
        assert_eq!(router.dead_letter_count(), 0);
    }
}
