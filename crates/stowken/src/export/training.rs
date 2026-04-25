//! Training data export: user/assistant pairs in JSONL or HuggingFace format.

use std::collections::HashSet;
use std::io::Write;

use thiserror::Error;

use crate::{
    storage::backend::StorageBackend,
    types::{AnalyticsQuery, ExportConfig, ExportFormat, ExportStats, SegmentType},
    vault::Stowken,
};

/// Errors from export operations.
#[derive(Debug, Error)]
pub enum ExportError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serialize(String),
    #[error("vault error: {0}")]
    Vault(String),
}

/// Export conversations as JSONL training pairs.
///
/// Each output line is a JSON object with a `"messages"` array containing
/// role/content pairs suitable for fine-tuning.
pub async fn export_jsonl<B: StorageBackend + 'static>(
    vault: &Stowken<B>,
    config: &ExportConfig,
    writer: &mut dyn Write,
) -> Result<ExportStats, ExportError> {
    let query = AnalyticsQuery {
        model: config.model.clone(),
        application: config.application.clone(),
        ..Default::default()
    };

    let limit = config.max_conversations.unwrap_or(u64::MAX);
    let ids = vault
        .backend()
        .list_conversations(&query, limit, 0)
        .await
        .map_err(|e| ExportError::Vault(e.to_string()))?;

    let mut total_pairs: u64 = 0;
    let mut unique_pairs: u64 = 0;
    let mut tokens_exported: u64 = 0;
    let mut seen_pairs: HashSet<(String, String)> = HashSet::new();

    for id in &ids {
        let conv = vault
            .retrieve(id)
            .await
            .map_err(|e| ExportError::Vault(e.to_string()))?;

        // Collect all segments in order
        let mut sys_tokens: Option<Vec<u32>> = None;
        let mut user_segs: Vec<Vec<u32>> = Vec::new();
        let mut asst_segs: Vec<Vec<u32>> = Vec::new();

        for seg in &conv.segments {
            match seg.segment_type {
                SegmentType::SystemPrompt if config.include_system_prompts => {
                    sys_tokens = Some(seg.tokens.clone());
                }
                SegmentType::UserTurn => user_segs.push(seg.tokens.clone()),
                SegmentType::AssistantTurn => asst_segs.push(seg.tokens.clone()),
                _ => {}
            }
        }

        // Emit paired turns
        let pair_count = user_segs.len().min(asst_segs.len());
        for i in 0..pair_count {
            let user_hash = format!("{:x}", md5_tokens(&user_segs[i]));
            let asst_hash = format!("{:x}", md5_tokens(&asst_segs[i]));
            total_pairs += 1;

            if config.deduplicate_pairs {
                let key = (user_hash, asst_hash);
                if seen_pairs.contains(&key) {
                    continue;
                }
                seen_pairs.insert(key);
            }

            unique_pairs += 1;

            let mut messages: Vec<serde_json::Value> = Vec::new();

            if let Some(sys) = &sys_tokens {
                messages.push(serde_json::json!({
                    "role": "system",
                    "content_tokens": sys,
                }));
                tokens_exported += sys.len() as u64;
            }

            messages.push(serde_json::json!({
                "role": "user",
                "content_tokens": user_segs[i],
            }));
            tokens_exported += user_segs[i].len() as u64;

            messages.push(serde_json::json!({
                "role": "assistant",
                "content_tokens": asst_segs[i],
            }));
            tokens_exported += asst_segs[i].len() as u64;

            let line = match config.format {
                ExportFormat::Jsonl => serde_json::json!({ "messages": messages }),
                ExportFormat::HuggingFace => serde_json::json!({
                    "conversations": messages,
                }),
                ExportFormat::Parquet => {
                    return Err(ExportError::Vault(
                        "use export_parquet for Parquet format".to_owned(),
                    ))
                }
            };

            serde_json::to_writer(&mut *writer, &line)
                .map_err(|e| ExportError::Serialize(e.to_string()))?;
            writeln!(writer)?;
        }
    }

    Ok(ExportStats {
        total_pairs,
        unique_pairs,
        tokens_exported,
    })
}

/// Lightweight 64-bit hash of a token slice used for dedup key generation.
fn md5_tokens(tokens: &[u32]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    tokens.hash(&mut hasher);
    hasher.finish()
}
