//! Storage backend trait and implementations.

pub mod backend;
pub mod filesystem;
pub mod memory;

#[cfg(feature = "s3")]
pub mod s3;

pub use backend::{StorageBackend, StorageError, StorageResult};
pub use filesystem::FilesystemBackend;
pub use memory::MemoryBackend;

#[cfg(feature = "s3")]
pub use s3::S3Backend;
