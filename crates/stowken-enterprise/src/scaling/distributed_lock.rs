//! Distributed segment-level locking for concurrent multi-node ingest.
//!
//! In a single-node deployment, Tokio's `Mutex` is sufficient.
//! In a multi-node deployment, use a distributed lock backend such as
//! Redis (SETNX + TTL), DynamoDB conditional writes, or etcd leases.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, OwnedMutexGuard};
use thiserror::Error;

/// Errors from the lock manager.
#[derive(Debug, Error)]
pub enum LockError {
    #[error("failed to acquire lock for '{key}' within the timeout")]
    Timeout { key: String },
    #[error("lock backend error: {0}")]
    Backend(String),
}

/// A lock guard that releases the lock when dropped.
///
/// For the in-process implementation this is an `OwnedMutexGuard`.
/// A distributed implementation would release the remote lock on drop.
pub struct LockGuard {
    _inner: OwnedMutexGuard<()>,
}

/// A pluggable distributed lock manager.
#[async_trait::async_trait]
pub trait LockManager: Send + Sync {
    /// Acquire an exclusive lock for `key`, waiting up to `timeout`.
    async fn lock(&self, key: &str, timeout: Duration) -> Result<LockGuard, LockError>;
}

/// In-process lock manager backed by per-key `Arc<Mutex<()>>`.
///
/// Suitable for single-node deployments. Replace with a Redis or DynamoDB
/// implementation for multi-node horizontal scaling.
#[derive(Default, Clone)]
pub struct LocalLockManager {
    locks: Arc<Mutex<HashMap<String, Arc<Mutex<()>>>>>,
}

impl LocalLockManager {
    pub fn new() -> Self {
        Self::default()
    }

    async fn get_or_create(&self, key: &str) -> Arc<Mutex<()>> {
        let mut map = self.locks.lock().await;
        map.entry(key.to_owned()).or_insert_with(|| Arc::new(Mutex::new(()))).clone()
    }
}

#[async_trait::async_trait]
impl LockManager for LocalLockManager {
    async fn lock(&self, key: &str, timeout: Duration) -> Result<LockGuard, LockError> {
        let mutex = self.get_or_create(key).await;
        match tokio::time::timeout(timeout, mutex.lock_owned()).await {
            Ok(guard) => Ok(LockGuard { _inner: guard }),
            Err(_) => Err(LockError::Timeout { key: key.to_owned() }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn acquire_and_release() {
        let mgr = LocalLockManager::new();
        let guard = mgr.lock("seg-abc", Duration::from_secs(1)).await.unwrap();
        drop(guard);
        // Should be acquirable again after release
        mgr.lock("seg-abc", Duration::from_secs(1)).await.unwrap();
    }

    #[tokio::test]
    async fn independent_keys_do_not_block() {
        let mgr = LocalLockManager::new();
        let _g1 = mgr.lock("key-1", Duration::from_secs(1)).await.unwrap();
        // key-2 is independent — should not block
        let _g2 = mgr.lock("key-2", Duration::from_secs(1)).await.unwrap();
    }
}
