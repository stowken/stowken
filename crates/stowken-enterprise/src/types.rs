//! Enterprise-specific types not present in the open-source crate.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Summary metadata for a stored conversation, used in analytics and replication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMetadata {
    pub conversation_id: String,
    pub model: String,
    pub application: Option<String>,
    pub tags: Vec<String>,
    pub total_tokens: u64,
    pub segment_count: u32,
    pub dedup_ratio: f64,
    pub stored_at: DateTime<Utc>,
}
