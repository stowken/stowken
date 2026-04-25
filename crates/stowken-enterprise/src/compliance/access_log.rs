//! Structured audit log of all reads and writes, with caller identity.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

/// The type of operation being logged.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditOperation {
    StoreConversation,
    RetrieveConversation,
    RetrieveSegment,
    DeleteConversation,
    GarbageCollect,
    ExportData,
    TrainDictionary,
}

/// A single audit log entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub timestamp: DateTime<Utc>,
    /// The caller's identity (API key hash, user ID, service account, etc.).
    pub principal: String,
    pub tenant_id: Option<String>,
    pub operation: AuditOperation,
    /// The primary resource affected (conversation ID, segment hash, etc.).
    pub resource_id: Option<String>,
    /// Whether the operation succeeded.
    pub success: bool,
    /// Optional detail or error message.
    pub detail: Option<String>,
}

/// In-memory audit log. In production, persist to an append-only store
/// (S3, CloudWatch Logs, BigQuery, etc.).
#[derive(Clone, Default)]
pub struct AuditLog {
    entries: Arc<Mutex<Vec<AuditEntry>>>,
}

impl AuditLog {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an audit entry.
    pub async fn record(&self, entry: AuditEntry) {
        self.entries.lock().await.push(entry);
    }

    /// Convenience method for recording a successful operation.
    pub async fn log_success(
        &self,
        principal: &str,
        tenant_id: Option<&str>,
        operation: AuditOperation,
        resource_id: Option<&str>,
    ) {
        self.record(AuditEntry {
            timestamp: Utc::now(),
            principal: principal.to_owned(),
            tenant_id: tenant_id.map(str::to_owned),
            operation,
            resource_id: resource_id.map(str::to_owned),
            success: true,
            detail: None,
        })
        .await;
    }

    /// Convenience method for recording a failed operation.
    pub async fn log_failure(
        &self,
        principal: &str,
        tenant_id: Option<&str>,
        operation: AuditOperation,
        resource_id: Option<&str>,
        error: &str,
    ) {
        self.record(AuditEntry {
            timestamp: Utc::now(),
            principal: principal.to_owned(),
            tenant_id: tenant_id.map(str::to_owned),
            operation,
            resource_id: resource_id.map(str::to_owned),
            success: false,
            detail: Some(error.to_owned()),
        })
        .await;
    }

    /// Return all entries (for export or inspection).
    pub async fn entries(&self) -> Vec<AuditEntry> {
        self.entries.lock().await.clone()
    }

    /// Filter entries by principal.
    pub async fn entries_for_principal(&self, principal: &str) -> Vec<AuditEntry> {
        self.entries
            .lock()
            .await
            .iter()
            .filter(|e| e.principal == principal)
            .cloned()
            .collect()
    }

    /// Filter entries by tenant.
    pub async fn entries_for_tenant(&self, tenant_id: &str) -> Vec<AuditEntry> {
        self.entries
            .lock()
            .await
            .iter()
            .filter(|e| e.tenant_id.as_deref() == Some(tenant_id))
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn log_and_retrieve() {
        let log = AuditLog::new();
        log.log_success("alice", Some("acme"), AuditOperation::StoreConversation, Some("conv-1"))
            .await;
        log.log_failure("bob", Some("acme"), AuditOperation::RetrieveConversation, Some("conv-2"), "not found")
            .await;

        let all = log.entries().await;
        assert_eq!(all.len(), 2);

        let alice_entries = log.entries_for_principal("alice").await;
        assert_eq!(alice_entries.len(), 1);
        assert!(alice_entries[0].success);

        let acme_entries = log.entries_for_tenant("acme").await;
        assert_eq!(acme_entries.len(), 2);
    }
}
