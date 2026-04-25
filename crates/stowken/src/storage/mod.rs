//! Storage backend trait and implementations.

pub mod backend;
pub mod filesystem;
pub mod memory;

pub use backend::{StorageBackend, StorageError, StorageResult};
pub use filesystem::FilesystemBackend;
pub use memory::MemoryBackend;
