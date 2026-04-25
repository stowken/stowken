//! Multi-tenant isolation: per-tenant segment hashing, ACL enforcement, and
//! per-tenant encryption key management.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use stowken::types::{SegmentHash, Token};

use crate::encryption::{EncryptResult, Encryptor, KeyProvider};

/// Per-tenant configuration.
#[derive(Debug, Clone)]
pub struct TenantConfig {
    pub tenant_id: String,
    /// KMS key reference used for encrypting this tenant's segment data.
    pub encryption_key_ref: Option<String>,
    /// Maximum segments this tenant may store; `None` = unlimited.
    pub max_segments: Option<u64>,
}

/// Registry of known tenants and their configurations.
#[derive(Clone, Default)]
pub struct TenantRegistry {
    tenants: Arc<RwLock<HashMap<String, TenantConfig>>>,
}

impl TenantRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tenant.
    pub async fn register(&self, config: TenantConfig) {
        self.tenants.write().await.insert(config.tenant_id.clone(), config);
    }

    /// Look up a tenant's configuration.
    pub async fn get(&self, tenant_id: &str) -> Option<TenantConfig> {
        self.tenants.read().await.get(tenant_id).cloned()
    }

    /// Returns `true` if the tenant is registered.
    pub async fn is_registered(&self, tenant_id: &str) -> bool {
        self.tenants.read().await.contains_key(tenant_id)
    }
}

/// Compute a tenant-scoped segment hash.
///
/// Two tenants with identical segment content produce different hashes,
/// preventing information leakage through hash probing.
pub fn tenant_scoped_hash(tokens: &[Token], tenant_id: &str) -> SegmentHash {
    stowken::dedup::exact::hash_segment_tenant(tokens, tenant_id)
}

/// Encrypt a segment's compressed data using the tenant's key.
pub async fn encrypt_for_tenant(
    data: &[u8],
    tenant_id: &str,
    key_provider: &dyn KeyProvider,
) -> EncryptResult<Vec<u8>> {
    let key = key_provider.get_key(tenant_id).await?;
    let enc = Encryptor::new(key);
    enc.encrypt(data)
}

/// Decrypt a segment's data using the tenant's key.
pub async fn decrypt_for_tenant(
    data: &[u8],
    tenant_id: &str,
    key_provider: &dyn KeyProvider,
) -> EncryptResult<Vec<u8>> {
    let key = key_provider.get_key(tenant_id).await?;
    let enc = Encryptor::new(key);
    enc.decrypt(data)
}

/// An ACL entry granting a principal access to a tenant's data.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AclEntry {
    /// The principal (user ID, service account, API key hash, etc.).
    pub principal: String,
    pub tenant_id: String,
    pub permissions: Vec<Permission>,
}

/// Permissions that can be granted to a principal.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Permission {
    Read,
    Write,
    Delete,
    Admin,
}

/// In-memory ACL store.
#[derive(Default, Clone)]
pub struct AclStore {
    entries: Arc<RwLock<Vec<AclEntry>>>,
}

impl AclStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn grant(&self, entry: AclEntry) {
        self.entries.write().await.push(entry);
    }

    /// Check whether `principal` has `permission` for `tenant_id`.
    pub async fn is_permitted(&self, principal: &str, tenant_id: &str, permission: &Permission) -> bool {
        self.entries
            .read()
            .await
            .iter()
            .any(|e| e.principal == principal && e.tenant_id == tenant_id && e.permissions.contains(permission))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encryption::RawKeyProvider;

    #[tokio::test]
    async fn tenant_registry_roundtrip() {
        let registry = TenantRegistry::new();
        registry
            .register(TenantConfig {
                tenant_id: "acme".to_owned(),
                encryption_key_ref: None,
                max_segments: Some(1000),
            })
            .await;
        assert!(registry.is_registered("acme").await);
        assert!(!registry.is_registered("other").await);
    }

    #[tokio::test]
    async fn encrypt_decrypt_for_tenant() {
        let provider = RawKeyProvider::random();
        let data = b"compressed segment bytes";
        let enc = encrypt_for_tenant(data, "tenant-1", &provider).await.unwrap();
        let dec = decrypt_for_tenant(&enc, "tenant-1", &provider).await.unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn tenant_scoped_hashes_differ() {
        let tokens = vec![1u32, 2, 3];
        let a = tenant_scoped_hash(&tokens, "tenant-a");
        let b = tenant_scoped_hash(&tokens, "tenant-b");
        assert_ne!(a, b);
    }

    #[tokio::test]
    async fn acl_permissions() {
        let store = AclStore::new();
        store
            .grant(AclEntry {
                principal: "alice".to_owned(),
                tenant_id: "acme".to_owned(),
                permissions: vec![Permission::Read, Permission::Write],
            })
            .await;
        assert!(store.is_permitted("alice", "acme", &Permission::Read).await);
        assert!(!store.is_permitted("alice", "acme", &Permission::Admin).await);
        assert!(!store.is_permitted("bob", "acme", &Permission::Read).await);
    }
}
