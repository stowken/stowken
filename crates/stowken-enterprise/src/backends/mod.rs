//! Cloud storage backends for Stowken Enterprise.
//!
//! Each backend implements [`stowken::storage::StorageBackend`] and stores
//! segment blobs and conversation manifests in the respective cloud object store.
//!
//! Analytics operations (stats, query_analytics, etc.) are intentionally
//! minimal — pair with a [`stowken::index::MetadataIndex`] for full
//! analytics in production.

#[cfg(feature = "s3")]
pub mod s3;

#[cfg(feature = "gcs")]
pub mod gcs;

#[cfg(feature = "azure")]
pub mod azure;

#[cfg(feature = "s3")]
pub use s3::S3Backend;

#[cfg(feature = "gcs")]
pub use gcs::GcsBackend;

#[cfg(feature = "azure")]
pub use azure::AzureBackend;
