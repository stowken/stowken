//! # Stowken
//!
//! Compressed storage and retrieval of LLM token sequences.
//!
//! Stowken applies segment-level deduplication, variable-width integer
//! encoding, and zstd dictionary compression to achieve 70–85% storage
//! savings over naive `Vec<u32>` storage.
//!
//! Licensed under the Apache License 2.0. Enterprise features (cloud backends,
//! compliance tooling, advanced analytics) are available in `stowken-enterprise`.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use stowken::{
//!     Stowken,
//!     types::{Conversation, Message, MessageContent, StowkenConfig},
//!     storage::MemoryBackend,
//! };
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let vault = Stowken::new(MemoryBackend::new(), StowkenConfig::default()).await?;
//!
//!     let conversation = Conversation {
//!         id: None,
//!         application: Some("my-app".to_string()),
//!         model: "gpt-4".to_string(),
//!         tokenizer: "cl100k_base".to_string(),
//!         messages: vec![
//!             Message {
//!                 role: "system".to_string(),
//!                 content: MessageContent::Tokens(vec![1, 2, 3]),
//!                 name: None,
//!                 tool_call_id: None,
//!             },
//!             Message {
//!                 role: "user".to_string(),
//!                 content: MessageContent::Tokens(vec![4, 5, 6]),
//!                 name: None,
//!                 tool_call_id: None,
//!             },
//!             Message {
//!                 role: "assistant".to_string(),
//!                 content: MessageContent::Tokens(vec![7, 8, 9]),
//!                 name: None,
//!                 tool_call_id: None,
//!             },
//!         ],
//!         metadata: None,
//!     };
//!
//!     let result = vault.store(conversation).await?;
//!     println!("Stored {} — {} segments, {} deduped",
//!         result.id, result.total_segments, result.deduped_segments);
//!
//!     let retrieved = vault.retrieve(&result.id).await?;
//!     println!("Retrieved {} segments", retrieved.segments.len());
//!
//!     Ok(())
//! }
//! ```

pub mod compression;
pub mod dedup;
pub mod dict_registry;
pub mod export;
pub mod index;
pub mod segmenter;
pub mod storage;
pub mod tokenizer;
pub mod types;
pub mod vault;

pub use tokenizer::get_tokenizer;
pub use vault::{Stowken, StowkenError};
