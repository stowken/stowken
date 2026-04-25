//! Metadata index replication for multi-node deployments.
//!
//! In a single-node deployment, the SQLite index is the source of truth.
//! In a multi-node deployment, changes must be replicated so that all nodes
//! share consistent metadata. This module defines a pluggable `ReplicationSink`
//! trait and a no-op implementation suitable for single-node use.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::Mutex;

use stowken::types::SegmentHash;

use crate::types::ConversationMetadata;

type EventLog = Arc<Mutex<Vec<(DateTime<Utc>, ReplicationEvent)>>>;

/// Events that are replicated across nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReplicationEvent {
    /// A new conversation was stored.
    ConversationStored {
        conversation_id: String,
        metadata: ConversationMetadata,
        stored_at: DateTime<Utc>,
    },
    /// A conversation was deleted.
    ConversationDeleted {
        conversation_id: String,
        deleted_at: DateTime<Utc>,
    },
    /// A segment was referenced by a new conversation.
    SegmentReferenced {
        hash: SegmentHash,
        conversation_id: String,
    },
    /// A segment's reference count dropped to zero and it was garbage-collected.
    SegmentCollected {
        hash: SegmentHash,
        collected_at: DateTime<Utc>,
    },
}

impl ReplicationEvent {
    /// The conversation ID associated with this event, if any.
    pub fn conversation_id(&self) -> Option<&str> {
        match self {
            Self::ConversationStored { conversation_id, .. } => Some(conversation_id),
            Self::ConversationDeleted { conversation_id, .. } => Some(conversation_id),
            Self::SegmentReferenced { conversation_id, .. } => Some(conversation_id),
            Self::SegmentCollected { .. } => None,
        }
    }
}

/// Errors from a replication sink.
#[derive(Debug, Error)]
pub enum ReplicationError {
    #[error("replication backend unavailable: {0}")]
    Unavailable(String),
    #[error("serialization error: {0}")]
    Serialization(String),
}

/// A pluggable sink that receives replication events.
///
/// Implementations might publish to Kafka, SQS, Redis Streams, etc.
#[async_trait]
pub trait ReplicationSink: Send + Sync {
    /// Publish a single event to the replication stream.
    async fn publish(&self, event: ReplicationEvent) -> Result<(), ReplicationError>;

    /// Publish a batch of events atomically where possible.
    async fn publish_batch(
        &self,
        events: Vec<ReplicationEvent>,
    ) -> Result<(), ReplicationError> {
        for event in events {
            self.publish(event).await?;
        }
        Ok(())
    }
}

/// A replication source that allows nodes to catch up to the current state.
#[async_trait]
pub trait ReplicationSource: Send + Sync {
    /// Poll for events since `since`, returning up to `limit` events.
    async fn poll(
        &self,
        since: Option<DateTime<Utc>>,
        limit: usize,
    ) -> Result<Vec<ReplicationEvent>, ReplicationError>;
}

/// No-op sink — suitable for single-node deployments.
///
/// Drops all events silently. Replace with a Kafka or SQS implementation
/// for multi-node horizontal scaling.
#[derive(Default, Clone)]
pub struct NoopReplicationSink;

impl NoopReplicationSink {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ReplicationSink for NoopReplicationSink {
    async fn publish(&self, _event: ReplicationEvent) -> Result<(), ReplicationError> {
        Ok(())
    }
}

/// In-memory replication sink/source for testing and local development.
///
/// Stores events in a `Vec` protected by a `Mutex`. Not suitable for production.
#[derive(Default, Clone)]
pub struct MemoryReplicationBus {
    events: EventLog,
}

impl MemoryReplicationBus {
    pub fn new() -> Self {
        Self::default()
    }

    /// Drain all events — useful in tests.
    pub async fn drain(&self) -> Vec<ReplicationEvent> {
        let mut guard = self.events.lock().await;
        guard.drain(..).map(|(_, e)| e).collect()
    }
}

#[async_trait]
impl ReplicationSink for MemoryReplicationBus {
    async fn publish(&self, event: ReplicationEvent) -> Result<(), ReplicationError> {
        self.events.lock().await.push((Utc::now(), event));
        Ok(())
    }
}

#[async_trait]
impl ReplicationSource for MemoryReplicationBus {
    async fn poll(
        &self,
        since: Option<DateTime<Utc>>,
        limit: usize,
    ) -> Result<Vec<ReplicationEvent>, ReplicationError> {
        let guard = self.events.lock().await;
        let events: Vec<ReplicationEvent> = guard
            .iter()
            .filter(|(ts, _)| since.is_none_or(|s| *ts > s))
            .take(limit)
            .map(|(_, e)| e.clone())
            .collect();
        Ok(events)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ConversationMetadata;

    fn sample_metadata() -> ConversationMetadata {
        ConversationMetadata {
            conversation_id: "conv-1".into(),
            model: "gpt-4".into(),
            application: Some("test-app".into()),
            tags: vec![],
            total_tokens: 42,
            segment_count: 2,
            dedup_ratio: 0.0,
            stored_at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn noop_sink_accepts_all_events() {
        let sink = NoopReplicationSink::new();
        sink.publish(ReplicationEvent::ConversationDeleted {
            conversation_id: "conv-1".into(),
            deleted_at: Utc::now(),
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn memory_bus_roundtrip() {
        let bus = MemoryReplicationBus::new();

        bus.publish(ReplicationEvent::ConversationStored {
            conversation_id: "conv-1".into(),
            metadata: sample_metadata(),
            stored_at: Utc::now(),
        })
        .await
        .unwrap();

        bus.publish(ReplicationEvent::SegmentCollected {
            hash: SegmentHash("abc123".into()),
            collected_at: Utc::now(),
        })
        .await
        .unwrap();

        let events = bus.poll(None, 10).await.unwrap();
        assert_eq!(events.len(), 2);
    }

    #[tokio::test]
    async fn memory_bus_since_filter() {
        let bus = MemoryReplicationBus::new();

        bus.publish(ReplicationEvent::ConversationDeleted {
            conversation_id: "conv-old".into(),
            deleted_at: Utc::now(),
        })
        .await
        .unwrap();

        let cutoff = Utc::now();

        // small sleep to ensure timestamps differ
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;

        bus.publish(ReplicationEvent::ConversationDeleted {
            conversation_id: "conv-new".into(),
            deleted_at: Utc::now(),
        })
        .await
        .unwrap();

        let events = bus.poll(Some(cutoff), 10).await.unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].conversation_id(), Some("conv-new"));
    }

    #[tokio::test]
    async fn batch_publish() {
        let bus = MemoryReplicationBus::new();
        let events = vec![
            ReplicationEvent::ConversationDeleted {
                conversation_id: "c1".into(),
                deleted_at: Utc::now(),
            },
            ReplicationEvent::ConversationDeleted {
                conversation_id: "c2".into(),
                deleted_at: Utc::now(),
            },
        ];
        bus.publish_batch(events).await.unwrap();
        assert_eq!(bus.poll(None, 10).await.unwrap().len(), 2);
    }
}
