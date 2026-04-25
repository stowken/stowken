//! # Stowken Enterprise
//!
//! Enterprise-tier extensions for [Stowken](https://docs.rs/stowken):
//!
//! - **Cloud backends** — AWS S3, Google Cloud Storage, Azure Blob Storage
//! - **Encryption at rest** — AES-256-GCM with pluggable KMS adapters
//! - **Multi-tenant isolation** — per-tenant segment hashing, encryption, and ACL enforcement
//! - **Near-dedup** — MinHash LSH for detecting similar (non-identical) segments
//! - **Advanced analytics** — cost attribution, usage forecasting, anomaly detection
//! - **Compliance** — GDPR deletion verification, configurable retention policies, audit logging
//! - **Horizontal scaling** — distributed lock abstraction, metadata replication sinks
//! - **Parquet export** — Arrow-native Parquet export for analytics and training pipelines
//!
//! ## License validation
//!
//! Enterprise features require a valid license token. Validate one at startup:
//!
//! ```no_run
//! use stowken_enterprise::license::License;
//!
//! let license = License::from_jwt("your.jwt.token").expect("invalid license");
//! license.require_feature("near-dedup").expect("feature not licensed");
//! ```
//!
//! In development, enable the `dev-license` Cargo feature for an offline license stub.

// ── Core enterprise modules ──────────────────────────────────────────────────

pub mod enterprise_vault;
pub mod types;
pub mod advanced_analytics;
pub mod backends;
pub mod compliance;
pub mod encryption;
pub mod export;
pub mod license;
pub mod multi_tenant;
pub mod near_dedup;
pub mod scaling;

// ── Convenience re-exports ───────────────────────────────────────────────────

pub use types::ConversationMetadata;
pub use advanced_analytics::{
    CostAttributionRow, TokenForecast, UsageAnomaly,
};
pub use compliance::{
    access_log::{AuditEntry, AuditLog, AuditOperation},
    deletion_audit::{DeletionReport, VerificationResult},
    retention::{RetentionPeriod, RetentionPolicy, RetentionPolicyRegistry, RetentionSweepResult},
};
pub use encryption::{Encryptor, KeyProvider, RawKeyProvider};
pub use license::{License, LicenseError, LicenseTier};
pub use multi_tenant::{AclEntry, AclStore, Permission, TenantRegistry};
pub use near_dedup::{compute_signature, jaccard_estimate, MinHashSignature};
pub use scaling::{
    distributed_lock::{LockError, LockGuard, LockManager, LocalLockManager},
    replication::{MemoryReplicationBus, NoopReplicationSink, ReplicationEvent, ReplicationSink, ReplicationSource},
};
