//! Configurable data retention policies with automatic expiration.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

use stowken::types::AnalyticsQuery;

/// A retention policy applied to conversations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    pub name: String,
    /// How long to retain conversations before marking them for deletion.
    pub retain_for: RetentionPeriod,
    /// When set, the policy applies only to conversations from this application.
    pub application_filter: Option<String>,
    /// When set, the policy applies only to conversations from this model.
    pub model_filter: Option<String>,
}

/// Retention duration specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetentionPeriod {
    Days(u32),
    Months(u32),
    Years(u32),
    /// Keep forever — no automatic expiration.
    Forever,
}

impl RetentionPeriod {
    /// Compute the cutoff timestamp before which conversations should be expired.
    pub fn cutoff(&self) -> Option<DateTime<Utc>> {
        match self {
            Self::Days(n) => Some(Utc::now() - Duration::days(*n as i64)),
            Self::Months(n) => Some(Utc::now() - Duration::days(*n as i64 * 30)),
            Self::Years(n) => Some(Utc::now() - Duration::days(*n as i64 * 365)),
            Self::Forever => None,
        }
    }
}

/// The result of a retention sweep.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionSweepResult {
    pub policy_name: String,
    pub expired_conversations: Vec<String>,
    pub swept_at: DateTime<Utc>,
}

/// Registry of active retention policies.
#[derive(Clone, Default)]
pub struct RetentionPolicyRegistry {
    policies: Arc<RwLock<Vec<RetentionPolicy>>>,
}

impl RetentionPolicyRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a retention policy.
    pub async fn add_policy(&self, policy: RetentionPolicy) {
        self.policies.write().await.push(policy);
    }

    /// List all registered policies.
    pub async fn policies(&self) -> Vec<RetentionPolicy> {
        self.policies.read().await.clone()
    }

    /// Run all retention policies against the given backend and return expired
    /// conversation IDs.
    ///
    /// The caller is responsible for actually deleting the conversations.
    pub async fn sweep<B: stowken::storage::StorageBackend>(
        &self,
        backend: &B,
    ) -> Result<Vec<RetentionSweepResult>, stowken::storage::StorageError> {
        let policies = self.policies.read().await.clone();
        let mut results = Vec::new();

        for policy in &policies {
            let cutoff = match policy.retain_for.cutoff() {
                Some(c) => c,
                None => continue, // Forever — skip
            };

            let query = AnalyticsQuery {
                model: policy.model_filter.clone(),
                application: policy.application_filter.clone(),
                date_to: Some(cutoff),
                ..Default::default()
            };

            let expired = backend.list_conversations(&query, u64::MAX, 0).await?;
            results.push(RetentionSweepResult {
                policy_name: policy.name.clone(),
                expired_conversations: expired,
                swept_at: Utc::now(),
            });
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stowken::storage::MemoryBackend;

    #[tokio::test]
    async fn retention_sweep_on_empty_store() {
        let registry = RetentionPolicyRegistry::new();
        registry
            .add_policy(RetentionPolicy {
                name: "30-day".to_owned(),
                retain_for: RetentionPeriod::Days(30),
                application_filter: None,
                model_filter: None,
            })
            .await;
        let backend = MemoryBackend::new();
        let results = registry.sweep(&backend).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].expired_conversations.is_empty());
    }

    #[test]
    fn retention_period_cutoffs() {
        assert!(RetentionPeriod::Days(30).cutoff().is_some());
        assert!(RetentionPeriod::Forever.cutoff().is_none());
    }
}
