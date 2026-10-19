//! SQLite metadata index for structural queryability.
//!
//! The index is intentionally synchronous (rusqlite) and bridged to async
//! callers via `tokio::task::spawn_blocking` in `vault.rs`.

use rusqlite::{params, Connection};
use std::sync::{Arc, Mutex};
use thiserror::Error;

use crate::types::{
    AnalyticsQuery, ConversationManifest, SegmentHash, SegmentType, SegmentTypeStats,
    StoredSegment, SystemPromptInfo, TokenUsageStats,
};

/// Errors from the metadata index.
#[derive(Debug, Error)]
pub enum MetadataError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("parse error: {0}")]
    Parse(String),
}

pub type MetadataResult<T> = Result<T, MetadataError>;

/// A single segment index operation to be applied inside a batch transaction.
pub enum BatchSegmentOp {
    Upsert(StoredSegment),
    IncrementRef(SegmentHash),
}

/// SQLite-backed metadata index.
///
/// Wrapped in `Arc<Mutex<Connection>>` so it can be moved into
/// `spawn_blocking` closures.
#[derive(Clone)]
pub struct MetadataIndex {
    pub(super) conn: Arc<Mutex<Connection>>,
}

impl MetadataIndex {
    /// Open (or create) the metadata database at `path`.
    pub fn open(path: &str) -> MetadataResult<Self> {
        let conn = Connection::open(path)?;
        let idx = Self { conn: Arc::new(Mutex::new(conn)) };
        idx.initialize()?;
        Ok(idx)
    }

    /// Open an in-memory database (for testing / memory backend).
    pub fn open_in_memory() -> MetadataResult<Self> {
        let conn = Connection::open_in_memory()?;
        let idx = Self { conn: Arc::new(Mutex::new(conn)) };
        idx.initialize()?;
        Ok(idx)
    }

    fn initialize(&self) -> MetadataResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(SCHEMA_SQL)?;
        Ok(())
    }

    // ── Indexing ──────────────────────────────────────────────────────────

    /// Index a conversation manifest and its segment references.
    pub fn index_conversation(&self, manifest: &ConversationManifest) -> MetadataResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO conversations \
             (id, application, model, tokenizer, total_tokens, segment_count, created_at, metadata_json) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                manifest.id,
                manifest.application,
                manifest.model,
                manifest.tokenizer,
                manifest.total_tokens as i64,
                manifest.segments.len() as i64,
                manifest.created_at.to_rfc3339(),
                manifest.metadata.as_ref().and_then(|m| serde_json::to_string(m).ok()),
            ],
        )?;

        // Delete stale segment refs for this conversation (re-index)
        conn.execute(
            "DELETE FROM segment_refs WHERE conversation_id = ?1",
            params![manifest.id],
        )?;

        for seg_ref in &manifest.segments {
            conn.execute(
                "INSERT INTO segment_refs \
                 (conversation_id, segment_hash, segment_type, position, token_count) \
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    manifest.id,
                    seg_ref.hash.0,
                    seg_ref.segment_type.to_string(),
                    seg_ref.position as i64,
                    seg_ref.token_count as i64,
                ],
            )?;
        }
        Ok(())
    }

    /// Write all segment index ops and the conversation manifest in one transaction.
    ///
    /// This is the hot path for `store()` — one mutex lock, one BEGIN/COMMIT.
    pub fn store_conversation_batch(
        &self,
        ops: &[BatchSegmentOp],
        manifest: &ConversationManifest,
    ) -> MetadataResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("BEGIN", [])?;

        for op in ops {
            match op {
                BatchSegmentOp::Upsert(seg) => {
                    conn.execute(
                        "INSERT INTO segments_meta \
                         (hash, segment_type, tokenizer, token_count, compressed_size, raw_size, ref_count, created_at) \
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8) \
                         ON CONFLICT(hash) DO UPDATE SET ref_count = ref_count + 1",
                        params![
                            seg.hash.0,
                            seg.segment_type.to_string(),
                            seg.tokenizer,
                            seg.token_count as i64,
                            seg.compressed_size as i64,
                            seg.raw_size as i64,
                            seg.ref_count as i64,
                            seg.created_at.to_rfc3339(),
                        ],
                    )?;
                }
                BatchSegmentOp::IncrementRef(hash) => {
                    conn.execute(
                        "UPDATE segments_meta SET ref_count = ref_count + 1 WHERE hash = ?1",
                        params![hash.0],
                    )?;
                }
            }
        }

        // Conversation row
        conn.execute(
            "INSERT OR REPLACE INTO conversations \
             (id, application, model, tokenizer, total_tokens, segment_count, created_at, metadata_json) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                manifest.id,
                manifest.application,
                manifest.model,
                manifest.tokenizer,
                manifest.total_tokens as i64,
                manifest.segments.len() as i64,
                manifest.created_at.to_rfc3339(),
                manifest.metadata.as_ref().and_then(|m| serde_json::to_string(m).ok()),
            ],
        )?;

        conn.execute("DELETE FROM segment_refs WHERE conversation_id = ?1", params![manifest.id])?;
        for seg_ref in &manifest.segments {
            conn.execute(
                "INSERT INTO segment_refs \
                 (conversation_id, segment_hash, segment_type, position, token_count) \
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    manifest.id,
                    seg_ref.hash.0,
                    seg_ref.segment_type.to_string(),
                    seg_ref.position as i64,
                    seg_ref.token_count as i64,
                ],
            )?;
        }

        conn.execute("COMMIT", [])?;
        Ok(())
    }

    /// Upsert segment metadata (called when a segment is first stored).
    pub fn upsert_segment(&self, seg: &StoredSegment) -> MetadataResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO segments_meta \
             (hash, segment_type, tokenizer, token_count, compressed_size, raw_size, ref_count, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8) \
             ON CONFLICT(hash) DO UPDATE SET ref_count = ref_count + 1",
            params![
                seg.hash.0,
                seg.segment_type.to_string(),
                seg.tokenizer,
                seg.token_count as i64,
                seg.compressed_size as i64,
                seg.raw_size as i64,
                seg.ref_count as i64,
                seg.created_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    /// Every distinct segment hash currently tracked by the index.
    /// Used by compaction and substring GC to walk the corpus.
    pub fn all_segment_hashes(&self) -> MetadataResult<Vec<SegmentHash>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT hash FROM segments_meta")?;
        let rows = stmt
            .query_map([], |r| r.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows.into_iter().map(SegmentHash).collect())
    }

    /// Update the `compressed_size` of an existing segment without
    /// touching ref_count or any other column. Used by compaction to
    /// keep stats accurate after re-encoding.
    pub fn update_segment_compressed_size(
        &self,
        hash: &SegmentHash,
        new_size: u32,
    ) -> MetadataResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE segments_meta SET compressed_size = ?1 WHERE hash = ?2",
            params![new_size as i64, hash.0],
        )?;
        Ok(())
    }

    /// Increment ref_count for an existing segment.
    pub fn increment_ref(&self, hash: &SegmentHash) -> MetadataResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE segments_meta SET ref_count = ref_count + 1 WHERE hash = ?1",
            params![hash.0],
        )?;
        Ok(())
    }

    /// Decrement ref_count. Returns `true` if ref_count reached 0.
    pub fn decrement_ref(&self, hash: &SegmentHash) -> MetadataResult<bool> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE segments_meta SET ref_count = MAX(0, ref_count - 1) WHERE hash = ?1",
            params![hash.0],
        )?;
        let count: i64 = conn.query_row(
            "SELECT ref_count FROM segments_meta WHERE hash = ?1",
            params![hash.0],
            |row| row.get(0),
        )?;
        Ok(count == 0)
    }

    /// Remove a conversation from the index.
    pub fn remove_conversation(&self, id: &str) -> MetadataResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM segment_refs WHERE conversation_id = ?1", params![id])?;
        conn.execute("DELETE FROM conversations WHERE id = ?1", params![id])?;
        Ok(())
    }

    /// Wipe every row from every index table. Used by reindex; manifests on
    /// disk remain the source of truth and are replayed afterwards.
    pub fn clear(&self) -> MetadataResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM segment_refs", [])?;
        conn.execute("DELETE FROM conversations", [])?;
        conn.execute("DELETE FROM segments_meta", [])?;
        Ok(())
    }

    // Near-dedup LSH state lives in the packed append-only logs managed by
    // `crate::near_dedup_index::NearDupIndex`, not SQLite. See that module.

    // ── Analytics ─────────────────────────────────────────────────────────

    /// Aggregate storage statistics.
    pub fn get_stats(&self) -> MetadataResult<TokenUsageStats> {
        let conn = self.conn.lock().unwrap();

        let total_conversations: i64 =
            conn.query_row("SELECT COUNT(*) FROM conversations", [], |r| r.get(0))?;
        let total_tokens: i64 =
            conn.query_row("SELECT COALESCE(SUM(total_tokens), 0) FROM conversations", [], |r| r.get(0))?;
        let unique_segments: i64 =
            conn.query_row("SELECT COUNT(*) FROM segments_meta", [], |r| r.get(0))?;
        let total_segments: i64 =
            conn.query_row("SELECT COALESCE(SUM(ref_count), 0) FROM segments_meta", [], |r| r.get(0))?;
        let storage_bytes: i64 =
            conn.query_row("SELECT COALESCE(SUM(compressed_size), 0) FROM segments_meta", [], |r| r.get(0))?;
        let naive_bytes: i64 =
            conn.query_row("SELECT COALESCE(SUM(CAST(raw_size AS INTEGER) * ref_count), 0) FROM segments_meta", [], |r| r.get(0))?;
        let raw_bytes: i64 =
            conn.query_row("SELECT COALESCE(SUM(raw_size), 0) FROM segments_meta", [], |r| r.get(0))?;

        let dedup_ratio = if total_segments == 0 {
            0.0
        } else {
            1.0 - (unique_segments as f64 / total_segments as f64)
        };
        let compression_ratio = if raw_bytes == 0 {
            1.0
        } else {
            storage_bytes as f64 / raw_bytes as f64
        };
        let savings_percentage = if naive_bytes == 0 {
            0.0
        } else {
            (1.0 - (storage_bytes as f64 / naive_bytes as f64)) * 100.0
        };

        Ok(TokenUsageStats {
            total_tokens: total_tokens as u64,
            total_conversations: total_conversations as u64,
            unique_segments: unique_segments as u64,
            total_segments: total_segments as u64,
            dedup_ratio,
            compression_ratio,
            storage_bytes: storage_bytes as u64,
            naive_bytes: naive_bytes as u64,
            savings_percentage,
        })
    }

    /// Per-segment-type statistics.
    pub fn get_segment_type_stats(&self) -> MetadataResult<Vec<SegmentTypeStats>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT segment_type, \
                    COUNT(*) as unique_count, \
                    SUM(ref_count) as total_refs, \
                    SUM(CAST(token_count AS REAL) * ref_count) / MAX(SUM(ref_count), 1) as avg_tokens, \
                    SUM(CAST(token_count AS INTEGER) * ref_count) as total_tokens, \
                    SUM(compressed_size) as compressed_bytes \
             FROM segments_meta \
             GROUP BY segment_type",
        )?;

        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, f64>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, i64>(5)?,
            ))
        })?;

        let mut stats = Vec::new();
        for row in rows {
            let (type_str, unique, refs, avg_tokens, total_tokens, compressed) = row?;
            let seg_type: SegmentType = type_str
                .parse()
                .map_err(|e: String| MetadataError::Parse(e))?;
            let dedup_ratio = if refs == 0 { 0.0 } else { 1.0 - unique as f64 / refs as f64 };
            stats.push(SegmentTypeStats {
                segment_type: seg_type,
                unique_count: unique as u64,
                total_references: refs as u64,
                dedup_ratio,
                avg_token_count: avg_tokens,
                total_tokens: total_tokens as u64,
                compressed_bytes: compressed as u64,
            });
        }
        Ok(stats)
    }

    /// List conversation IDs matching a query, with pagination.
    pub fn list_conversations(
        &self,
        query: &AnalyticsQuery,
        limit: u64,
        offset: u64,
    ) -> MetadataResult<Vec<String>> {
        let conn = self.conn.lock().unwrap();
        let mut sql = String::from("SELECT id FROM conversations WHERE 1=1");
        let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

        if let Some(model) = &query.model {
            sql.push_str(" AND model = ?");
            params_vec.push(Box::new(model.clone()));
        }
        if let Some(app) = &query.application {
            sql.push_str(" AND application = ?");
            params_vec.push(Box::new(app.clone()));
        }
        if let Some(from) = &query.date_from {
            sql.push_str(" AND created_at >= ?");
            params_vec.push(Box::new(from.to_rfc3339()));
        }
        if let Some(to) = &query.date_to {
            sql.push_str(" AND created_at <= ?");
            params_vec.push(Box::new(to.to_rfc3339()));
        }
        sql.push_str(" ORDER BY created_at DESC LIMIT ? OFFSET ?");
        params_vec.push(Box::new(limit as i64));
        params_vec.push(Box::new(offset as i64));

        let refs: Vec<&dyn rusqlite::ToSql> = params_vec.iter().map(|b| b.as_ref()).collect();
        let mut stmt = conn.prepare(&sql)?;
        let ids = stmt
            .query_map(refs.as_slice(), |r| r.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ids)
    }

    /// Conversations referencing a specific segment hash.
    pub fn find_conversations_by_segment(&self, hash: &SegmentHash) -> MetadataResult<Vec<String>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT DISTINCT conversation_id FROM segment_refs WHERE segment_hash = ?1",
        )?;
        let ids = stmt
            .query_map(params![hash.0], |r| r.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ids)
    }

    /// Sum of `raw_size` across unique segments — i.e., storage if every
    /// unique segment were stored uncompressed but exactly once.
    ///
    /// Used by benchmarks to separate "savings from dedup" from "savings from
    /// compression" in the three-stage breakdown:
    ///   naive_bytes (refs × raw)  →  dedup_only_bytes (uniques × raw)  →  storage_bytes (uniques × compressed)
    pub fn dedup_only_bytes(&self) -> MetadataResult<u64> {
        let conn = self.conn.lock().unwrap();
        let bytes: i64 = conn.query_row(
            "SELECT COALESCE(SUM(raw_size), 0) FROM segments_meta",
            [],
            |r| r.get(0),
        )?;
        Ok(bytes as u64)
    }

    /// Find segments referenced at least `min_refs` times, ordered by ref_count
    /// descending. Each row reports `wasted_bytes = (ref_count - 1) * raw_size`,
    /// which is the storage that dedup is saving on that segment alone.
    pub fn find_duplicates(
        &self,
        min_refs: u64,
        limit: u64,
    ) -> MetadataResult<Vec<crate::types::DuplicateSegment>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT hash, segment_type, token_count, ref_count, raw_size \
             FROM segments_meta \
             WHERE ref_count >= ?1 \
             ORDER BY ref_count DESC \
             LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(params![min_refs as i64, limit as i64], |row| {
                let hash: String = row.get(0)?;
                let type_str: String = row.get(1)?;
                let token_count: i64 = row.get(2)?;
                let ref_count: i64 = row.get(3)?;
                let raw_size: i64 = row.get(4)?;
                Ok((hash, type_str, token_count, ref_count, raw_size))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        let mut out = Vec::with_capacity(rows.len());
        for (hash, type_str, token_count, ref_count, raw_size) in rows {
            let segment_type = type_str
                .parse()
                .map_err(|e: String| MetadataError::Parse(e))?;
            let ref_count = ref_count as u64;
            let wasted = (ref_count.saturating_sub(1)) * (raw_size as u64);
            out.push(crate::types::DuplicateSegment {
                hash: SegmentHash(hash),
                segment_type,
                token_count: token_count as u32,
                ref_count,
                wasted_bytes: wasted,
            });
        }
        Ok(out)
    }

    /// List unique system prompts ordered by ref_count descending.
    pub fn list_unique_system_prompts(&self) -> MetadataResult<Vec<SystemPromptInfo>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT hash, token_count, ref_count FROM segments_meta \
             WHERE segment_type = 'system_prompt' \
             ORDER BY ref_count DESC",
        )?;
        let rows = stmt
            .query_map([], |r| {
                Ok(SystemPromptInfo {
                    hash: SegmentHash(r.get::<_, String>(0)?),
                    token_count: r.get::<_, i64>(1)? as u32,
                    ref_count: r.get::<_, i64>(2)? as u64,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    // ── Semantic search: segment-level embeddings ─────────────────────────

    #[cfg(feature = "semantic-search")]
    /// Insert or replace a normalized embedding for a segment.
    pub fn upsert_segment_embedding(
        &self,
        hash: &SegmentHash,
        embedding_model: &str,
        embedding: &[f32],
    ) -> MetadataResult<()> {
        let conn = self.conn.lock().unwrap();
        let bytes = f32_slice_to_bytes(embedding);
        conn.execute(
            "INSERT OR REPLACE INTO segment_embeddings \
             (segment_hash, embedding_model, dimension, embedding, embedded_at) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                hash.0,
                embedding_model,
                embedding.len() as i64,
                bytes,
                chrono::Utc::now().to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    #[cfg(feature = "semantic-search")]
    /// Fetch one segment's stored embedding for a model, if present. Used by
    /// the embedding-reuse path to copy a canonical's vector onto a
    /// near-duplicate variant without a fresh API call.
    pub fn get_segment_embedding(
        &self,
        hash: &SegmentHash,
        embedding_model: &str,
    ) -> MetadataResult<Option<Vec<f32>>> {
        let conn = self.conn.lock().unwrap();
        let row: Result<Vec<u8>, _> = conn.query_row(
            "SELECT embedding FROM segment_embeddings \
             WHERE segment_hash = ?1 AND embedding_model = ?2",
            params![hash.0, embedding_model],
            |r| r.get(0),
        );
        match row {
            Ok(bytes) => Ok(Some(bytes_to_f32_vec(&bytes))),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    #[cfg(feature = "semantic-search")]
    /// Load every (segment_hash, embedding) pair for a model. Used for the
    /// brute-force cosine scan during semantic search.
    pub fn load_all_segment_embeddings(
        &self,
        embedding_model: &str,
    ) -> MetadataResult<Vec<(SegmentHash, Vec<f32>)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT segment_hash, embedding FROM segment_embeddings \
             WHERE embedding_model = ?1",
        )?;
        let rows = stmt
            .query_map(params![embedding_model], |r| {
                let hash: String = r.get(0)?;
                let bytes: Vec<u8> = r.get(1)?;
                Ok((SegmentHash(hash), bytes_to_f32_vec(&bytes)))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    #[cfg(feature = "semantic-search")]
    /// List segment hashes that have NOT been embedded for the given model.
    pub fn unembedded_segment_hashes(
        &self,
        embedding_model: &str,
        limit: u64,
    ) -> MetadataResult<Vec<SegmentHash>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT sm.hash FROM segments_meta sm \
             LEFT JOIN segment_embeddings se \
               ON sm.hash = se.segment_hash AND se.embedding_model = ?1 \
             WHERE se.segment_hash IS NULL \
             LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(params![embedding_model, limit as i64], |r| {
                Ok(SegmentHash(r.get::<_, String>(0)?))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    #[cfg(feature = "semantic-search")]
    /// Of the given hashes, return the subset that already have embeddings
    /// for the given model. Used to skip work after a crash-restart.
    pub fn embedded_segment_hashes(
        &self,
        hashes: &[SegmentHash],
        embedding_model: &str,
    ) -> MetadataResult<std::collections::HashSet<SegmentHash>> {
        if hashes.is_empty() {
            return Ok(Default::default());
        }
        let conn = self.conn.lock().unwrap();
        let placeholders: Vec<String> = (0..hashes.len()).map(|_| "?".to_owned()).collect();
        let sql = format!(
            "SELECT segment_hash FROM segment_embeddings \
             WHERE embedding_model = ? AND segment_hash IN ({})",
            placeholders.join(",")
        );
        let mut stmt = conn.prepare(&sql)?;
        let mut params_vec: Vec<&dyn rusqlite::ToSql> = Vec::with_capacity(1 + hashes.len());
        params_vec.push(&embedding_model);
        for h in hashes {
            params_vec.push(&h.0);
        }
        let rows: Vec<SegmentHash> = stmt
            .query_map(rusqlite::params_from_iter(params_vec), |r| {
                Ok(SegmentHash(r.get::<_, String>(0)?))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows.into_iter().collect())
    }

    #[cfg(feature = "semantic-search")]
    /// Count segments that don't yet have an embedding for the given model.
    pub fn count_unembedded_segments(&self, embedding_model: &str) -> MetadataResult<u64> {
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM segments_meta sm \
             LEFT JOIN segment_embeddings se \
               ON sm.hash = se.segment_hash AND se.embedding_model = ?1 \
             WHERE se.segment_hash IS NULL",
            params![embedding_model],
            |r| r.get(0),
        )?;
        Ok(count as u64)
    }

    #[cfg(feature = "semantic-search")]
    /// Resolve segment hashes to (conversation, segment) rows with optional
    /// structural filters. Used after the cosine scan identifies top-K
    /// segments to map them back to conversations.
    pub fn conversations_for_segments_filtered(
        &self,
        hashes: &[SegmentHash],
        filter_model: Option<&str>,
        filter_application: Option<&str>,
        filter_segment_type: Option<&SegmentType>,
        filter_date_from: Option<chrono::DateTime<chrono::Utc>>,
        filter_date_to: Option<chrono::DateTime<chrono::Utc>>,
    ) -> MetadataResult<Vec<ConversationSegmentRow>> {
        if hashes.is_empty() {
            return Ok(vec![]);
        }
        let conn = self.conn.lock().unwrap();
        let placeholders: Vec<String> = (0..hashes.len()).map(|_| "?".to_owned()).collect();
        let mut sql = format!(
            "SELECT sr.conversation_id, sr.segment_hash, sr.segment_type, \
                    c.application, c.model, c.created_at \
             FROM segment_refs sr \
             JOIN conversations c ON c.id = sr.conversation_id \
             WHERE sr.segment_hash IN ({})",
            placeholders.join(",")
        );
        let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> =
            hashes.iter().map(|h| Box::new(h.0.clone()) as Box<dyn rusqlite::ToSql>).collect();
        if let Some(m) = filter_model {
            sql.push_str(" AND c.model = ?");
            params_vec.push(Box::new(m.to_owned()));
        }
        if let Some(a) = filter_application {
            sql.push_str(" AND c.application = ?");
            params_vec.push(Box::new(a.to_owned()));
        }
        if let Some(st) = filter_segment_type {
            sql.push_str(" AND sr.segment_type = ?");
            params_vec.push(Box::new(st.to_string()));
        }
        if let Some(from) = filter_date_from {
            sql.push_str(" AND c.created_at >= ?");
            params_vec.push(Box::new(from.to_rfc3339()));
        }
        if let Some(to) = filter_date_to {
            sql.push_str(" AND c.created_at <= ?");
            params_vec.push(Box::new(to.to_rfc3339()));
        }
        let mut stmt = conn.prepare(&sql)?;
        let param_refs: Vec<&dyn rusqlite::ToSql> =
            params_vec.iter().map(|b| b.as_ref()).collect();
        let rows = stmt
            .query_map(rusqlite::params_from_iter(param_refs), |r| {
                let st_str: String = r.get(2)?;
                let created: String = r.get(5)?;
                Ok(ConversationSegmentRow {
                    conversation_id: r.get(0)?,
                    segment_hash: SegmentHash(r.get::<_, String>(1)?),
                    segment_type: st_str.parse().unwrap_or(SegmentType::UserTurn),
                    application: r.get(3)?,
                    model: r.get(4)?,
                    created_at: chrono::DateTime::parse_from_rfc3339(&created)
                        .map(|d| d.with_timezone(&chrono::Utc))
                        .unwrap_or_else(|_| chrono::Utc::now()),
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    // ── Semantic search: conversation-level embeddings ────────────────────

    #[cfg(feature = "semantic-search")]
    /// Insert or replace a normalized embedding for a conversation summary.
    #[allow(clippy::too_many_arguments)]
    pub fn upsert_conversation_embedding(
        &self,
        conversation_id: &str,
        embedding_model: &str,
        summary_strategy: &str,
        embedding: &[f32],
        summary_text: Option<&str>,
    ) -> MetadataResult<()> {
        let conn = self.conn.lock().unwrap();
        let bytes = f32_slice_to_bytes(embedding);
        conn.execute(
            "INSERT OR REPLACE INTO conversation_embeddings \
             (conversation_id, embedding_model, summary_strategy, dimension, \
              embedding, summary_text, embedded_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                conversation_id,
                embedding_model,
                summary_strategy,
                embedding.len() as i64,
                bytes,
                summary_text,
                chrono::Utc::now().to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    #[cfg(feature = "semantic-search")]
    /// Load every (conversation_id, embedding) pair for a (model, strategy).
    pub fn load_all_conversation_embeddings(
        &self,
        embedding_model: &str,
        summary_strategy: &str,
    ) -> MetadataResult<Vec<(String, Vec<f32>)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT conversation_id, embedding FROM conversation_embeddings \
             WHERE embedding_model = ?1 AND summary_strategy = ?2",
        )?;
        let rows = stmt
            .query_map(params![embedding_model, summary_strategy], |r| {
                let id: String = r.get(0)?;
                let bytes: Vec<u8> = r.get(1)?;
                Ok((id, bytes_to_f32_vec(&bytes)))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    #[cfg(feature = "semantic-search")]
    /// List conversation IDs that have NOT been embedded for the given
    /// (model, strategy) pair.
    pub fn unembedded_conversation_ids(
        &self,
        embedding_model: &str,
        summary_strategy: &str,
        limit: u64,
    ) -> MetadataResult<Vec<String>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT c.id FROM conversations c \
             LEFT JOIN conversation_embeddings ce \
               ON c.id = ce.conversation_id \
                  AND ce.embedding_model = ?1 \
                  AND ce.summary_strategy = ?2 \
             WHERE ce.conversation_id IS NULL \
             LIMIT ?3",
        )?;
        let rows = stmt
            .query_map(
                params![embedding_model, summary_strategy, limit as i64],
                |r| r.get::<_, String>(0),
            )?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    #[cfg(feature = "semantic-search")]
    /// Resolve conversation IDs to metadata rows with optional structural filters.
    pub fn conversation_meta_filtered(
        &self,
        ids: &[String],
        filter_model: Option<&str>,
        filter_application: Option<&str>,
        filter_date_from: Option<chrono::DateTime<chrono::Utc>>,
        filter_date_to: Option<chrono::DateTime<chrono::Utc>>,
    ) -> MetadataResult<Vec<ConversationMetaRow>> {
        if ids.is_empty() {
            return Ok(vec![]);
        }
        let conn = self.conn.lock().unwrap();
        let placeholders: Vec<String> = (0..ids.len()).map(|_| "?".to_owned()).collect();
        let mut sql = format!(
            "SELECT id, application, model, created_at FROM conversations \
             WHERE id IN ({})",
            placeholders.join(",")
        );
        let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> =
            ids.iter().map(|i| Box::new(i.clone()) as Box<dyn rusqlite::ToSql>).collect();
        if let Some(m) = filter_model {
            sql.push_str(" AND model = ?");
            params_vec.push(Box::new(m.to_owned()));
        }
        if let Some(a) = filter_application {
            sql.push_str(" AND application = ?");
            params_vec.push(Box::new(a.to_owned()));
        }
        if let Some(from) = filter_date_from {
            sql.push_str(" AND created_at >= ?");
            params_vec.push(Box::new(from.to_rfc3339()));
        }
        if let Some(to) = filter_date_to {
            sql.push_str(" AND created_at <= ?");
            params_vec.push(Box::new(to.to_rfc3339()));
        }
        let mut stmt = conn.prepare(&sql)?;
        let param_refs: Vec<&dyn rusqlite::ToSql> =
            params_vec.iter().map(|b| b.as_ref()).collect();
        let rows = stmt
            .query_map(rusqlite::params_from_iter(param_refs), |r| {
                let created: String = r.get(3)?;
                Ok(ConversationMetaRow {
                    conversation_id: r.get(0)?,
                    application: r.get(1)?,
                    model: r.get(2)?,
                    created_at: chrono::DateTime::parse_from_rfc3339(&created)
                        .map(|d| d.with_timezone(&chrono::Utc))
                        .unwrap_or_else(|_| chrono::Utc::now()),
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }
}

// ── Helpers and row types for semantic search ────────────────────────────────

#[cfg(feature = "semantic-search")]
fn f32_slice_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

#[cfg(feature = "semantic-search")]
fn bytes_to_f32_vec(b: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(b.len() / 4);
    for chunk in b.as_chunks::<4>().0 {
        out.push(f32::from_le_bytes(*chunk));
    }
    out
}

#[cfg(feature = "semantic-search")]
#[derive(Debug, Clone)]
pub struct ConversationSegmentRow {
    pub conversation_id: String,
    pub segment_hash: SegmentHash,
    pub segment_type: SegmentType,
    pub application: Option<String>,
    pub model: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[cfg(feature = "semantic-search")]
#[derive(Debug, Clone)]
pub struct ConversationMetaRow {
    pub conversation_id: String,
    pub application: Option<String>,
    pub model: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

const SCHEMA_SQL: &str = "
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS conversations (
    id              TEXT PRIMARY KEY,
    application     TEXT,
    model           TEXT NOT NULL,
    tokenizer       TEXT NOT NULL,
    total_tokens    INTEGER NOT NULL,
    segment_count   INTEGER NOT NULL,
    created_at      TEXT NOT NULL,
    metadata_json   TEXT
);

CREATE TABLE IF NOT EXISTS segment_refs (
    conversation_id TEXT NOT NULL,
    segment_hash    TEXT NOT NULL,
    segment_type    TEXT NOT NULL,
    position        INTEGER NOT NULL,
    token_count     INTEGER NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);
CREATE INDEX IF NOT EXISTS idx_segref_conv ON segment_refs(conversation_id);
CREATE INDEX IF NOT EXISTS idx_segref_hash ON segment_refs(segment_hash);
CREATE INDEX IF NOT EXISTS idx_segref_type ON segment_refs(segment_type);

CREATE TABLE IF NOT EXISTS segments_meta (
    hash            TEXT PRIMARY KEY,
    segment_type    TEXT NOT NULL,
    tokenizer       TEXT NOT NULL,
    token_count     INTEGER NOT NULL,
    compressed_size INTEGER NOT NULL,
    raw_size        INTEGER NOT NULL,
    ref_count       INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_segmeta_type ON segments_meta(segment_type);
CREATE INDEX IF NOT EXISTS idx_segmeta_refs ON segments_meta(ref_count);

-- Near-dedup MinHash signatures + LSH band buckets are NOT stored here.
-- They live in the packed append-only logs managed by
-- `crate::near_dedup_index::NearDupIndex` (near_dedup/{signatures,aliases,
-- tombstones}.log next to this DB), so near-dedup is decoupled from SQLite.

-- Per-unique-segment embeddings for semantic search.
-- One row per unique segment per embedding model. Embeddings are stored
-- L2-normalized so cosine similarity reduces to a dot product at query time.
CREATE TABLE IF NOT EXISTS segment_embeddings (
    segment_hash    TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    dimension       INTEGER NOT NULL,
    embedding       BLOB NOT NULL,
    embedded_at     TEXT NOT NULL,
    PRIMARY KEY (segment_hash, embedding_model),
    FOREIGN KEY (segment_hash) REFERENCES segments_meta(hash)
);
CREATE INDEX IF NOT EXISTS idx_seg_emb_model ON segment_embeddings(embedding_model);

-- Per-conversation summary embeddings. The summary_strategy column allows
-- multiple summary approaches (concat-truncate, llm-generated) to coexist
-- for the same (conversation, embedding_model) pair.
CREATE TABLE IF NOT EXISTS conversation_embeddings (
    conversation_id  TEXT NOT NULL,
    embedding_model  TEXT NOT NULL,
    summary_strategy TEXT NOT NULL,
    dimension        INTEGER NOT NULL,
    embedding        BLOB NOT NULL,
    summary_text     TEXT,
    embedded_at      TEXT NOT NULL,
    PRIMARY KEY (conversation_id, embedding_model, summary_strategy),
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);
CREATE INDEX IF NOT EXISTS idx_conv_emb_model ON conversation_embeddings(embedding_model);
";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SegmentRef, SegmentType};
    use chrono::Utc;

    fn make_manifest(id: &str) -> ConversationManifest {
        ConversationManifest {
            schema_version: crate::types::MANIFEST_SCHEMA_VERSION,
            id: id.to_owned(),
            application: Some("myapp".to_owned()),
            model: "gpt-4".to_owned(),
            tokenizer: "cl100k_base".to_owned(),
            total_tokens: 100,
            segments: vec![SegmentRef {
                segment_type: SegmentType::SystemPrompt,
                hash: SegmentHash("abc123".to_owned()),
                token_count: 50,
                position: 0,
            }],
            created_at: Utc::now(),
            metadata: None,
        }
    }

    fn make_stored_seg() -> StoredSegment {
        StoredSegment {
            hash: SegmentHash("abc123".to_owned()),
            segment_type: SegmentType::SystemPrompt,
            tokenizer: "cl100k_base".to_owned(),
            token_count: 50,
            compressed_data: vec![0; 10],
            raw_size: 200,
            compressed_size: 10,
            ref_count: 1,
            created_at: Utc::now(),
        }
    }

    #[test]
    fn index_and_stats() {
        let idx = MetadataIndex::open_in_memory().unwrap();
        idx.upsert_segment(&make_stored_seg()).unwrap();
        idx.index_conversation(&make_manifest("conv-1")).unwrap();

        let stats = idx.get_stats().unwrap();
        assert_eq!(stats.total_conversations, 1);
        assert_eq!(stats.unique_segments, 1);
        assert_eq!(stats.total_segments, 1);
        assert_eq!(stats.total_tokens, 100);
    }

    #[test]
    fn dedup_increases_ref_count() {
        let idx = MetadataIndex::open_in_memory().unwrap();
        idx.upsert_segment(&make_stored_seg()).unwrap();
        idx.upsert_segment(&make_stored_seg()).unwrap(); // second call → ref_count = 2

        let stats = idx.get_stats().unwrap();
        assert_eq!(stats.unique_segments, 1);
        assert_eq!(stats.total_segments, 2);
        assert!(stats.dedup_ratio > 0.0);
    }

    #[test]
    fn system_prompt_listing() {
        let idx = MetadataIndex::open_in_memory().unwrap();
        idx.upsert_segment(&make_stored_seg()).unwrap();
        let prompts = idx.list_unique_system_prompts().unwrap();
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0].hash.0, "abc123");
    }

    #[test]
    fn find_conversations_by_segment() {
        let idx = MetadataIndex::open_in_memory().unwrap();
        idx.upsert_segment(&make_stored_seg()).unwrap();
        idx.index_conversation(&make_manifest("conv-1")).unwrap();
        idx.index_conversation(&make_manifest("conv-2")).unwrap();
        let ids = idx.find_conversations_by_segment(&SegmentHash("abc123".to_owned())).unwrap();
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn segment_type_stats() {
        let idx = MetadataIndex::open_in_memory().unwrap();
        idx.upsert_segment(&make_stored_seg()).unwrap();
        let type_stats = idx.get_segment_type_stats().unwrap();
        assert_eq!(type_stats.len(), 1);
        assert_eq!(type_stats[0].segment_type, SegmentType::SystemPrompt);
    }

    #[test]
    fn remove_conversation() {
        let idx = MetadataIndex::open_in_memory().unwrap();
        idx.upsert_segment(&make_stored_seg()).unwrap();
        idx.index_conversation(&make_manifest("conv-1")).unwrap();
        idx.remove_conversation("conv-1").unwrap();

        let stats = idx.get_stats().unwrap();
        assert_eq!(stats.total_conversations, 0);
    }
}
