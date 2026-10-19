//! Pre-built analytics queries over the metadata index.

use std::collections::HashMap;

use crate::types::AnalyticsQuery;

use super::metadata::{MetadataIndex, MetadataResult};

/// A grouped analytics row, keyed by the `group_by` dimensions.
pub type AnalyticsRow = HashMap<String, serde_json::Value>;

impl MetadataIndex {
    /// Run a flexible analytics query with optional grouping and filtering.
    ///
    /// Supported `group_by` values: `"model"`, `"application"`, `"segment_type"`, `"day"`.
    pub fn query_analytics(&self, query: &AnalyticsQuery) -> MetadataResult<Vec<AnalyticsRow>> {
        let conn = self.conn.lock().unwrap();

        let group_by = query.group_by.as_deref().unwrap_or_default();
        let mut select_cols = vec!["SUM(sr.token_count) as total_tokens".to_owned()];
        let mut group_cols: Vec<String> = Vec::new();

        for dim in group_by {
            match dim.as_str() {
                "model" => {
                    select_cols.push("c.model".to_owned());
                    group_cols.push("c.model".to_owned());
                }
                "application" => {
                    select_cols.push("c.application".to_owned());
                    group_cols.push("c.application".to_owned());
                }
                "segment_type" => {
                    select_cols.push("sr.segment_type".to_owned());
                    group_cols.push("sr.segment_type".to_owned());
                }
                "day" => {
                    select_cols.push("DATE(c.created_at) as day".to_owned());
                    group_cols.push("day".to_owned());
                }
                _ => {}
            }
        }

        let mut sql = format!(
            "SELECT {} FROM segment_refs sr \
             JOIN conversations c ON c.id = sr.conversation_id \
             WHERE 1=1",
            select_cols.join(", ")
        );

        let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

        if let Some(model) = &query.model {
            sql.push_str(" AND c.model = ?");
            params_vec.push(Box::new(model.clone()));
        }
        if let Some(app) = &query.application {
            sql.push_str(" AND c.application = ?");
            params_vec.push(Box::new(app.clone()));
        }
        if let Some(seg_type) = &query.segment_type {
            sql.push_str(" AND sr.segment_type = ?");
            params_vec.push(Box::new(seg_type.to_string()));
        }
        if let Some(from) = &query.date_from {
            sql.push_str(" AND c.created_at >= ?");
            params_vec.push(Box::new(from.to_rfc3339()));
        }
        if let Some(to) = &query.date_to {
            sql.push_str(" AND c.created_at <= ?");
            params_vec.push(Box::new(to.to_rfc3339()));
        }

        if !group_cols.is_empty() {
            sql.push_str(&format!(" GROUP BY {}", group_cols.join(", ")));
        }

        let refs: Vec<&dyn rusqlite::ToSql> = params_vec.iter().map(|b| b.as_ref()).collect();
        let mut stmt = conn.prepare(&sql)?;
        let col_count = stmt.column_count();
        let col_names: Vec<String> = (0..col_count)
            .map(|i| stmt.column_name(i).unwrap_or("?").to_owned())
            .collect();

        let rows = stmt
            .query_map(refs.as_slice(), |row| {
                let mut map = HashMap::new();
                for (i, name) in col_names.iter().enumerate() {
                    let val: rusqlite::types::Value = row.get(i)?;
                    let json_val = sqlite_value_to_json(val);
                    // Strip table prefix from column names
                    let short_name = name.split('.').next_back().unwrap_or(name).to_owned();
                    map.insert(short_name, json_val);
                }
                Ok(map)
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(rows)
    }

    /// Tokens by model by day.
    pub fn tokens_by_model_by_day(&self) -> MetadataResult<Vec<AnalyticsRow>> {
        self.query_analytics(&AnalyticsQuery {
            group_by: Some(vec!["model".to_owned(), "day".to_owned()]),
            ..Default::default()
        })
    }

    /// Cost attribution: tokens per application per segment type.
    pub fn cost_attribution(&self) -> MetadataResult<Vec<AnalyticsRow>> {
        self.query_analytics(&AnalyticsQuery {
            group_by: Some(vec!["application".to_owned(), "segment_type".to_owned()]),
            ..Default::default()
        })
    }

    /// Dedup ratio per segment type (unique vs total refs).
    pub fn dedup_ratio_by_type(&self) -> MetadataResult<Vec<AnalyticsRow>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT segment_type, \
                    COUNT(*) as unique_count, \
                    SUM(ref_count) as total_refs, \
                    1.0 - CAST(COUNT(*) AS REAL) / MAX(SUM(ref_count), 1) as dedup_ratio \
             FROM segments_meta \
             GROUP BY segment_type",
        )?;

        let col_names: Vec<String> = (0..stmt.column_count())
            .map(|i| stmt.column_name(i).unwrap_or("?").to_owned())
            .collect();

        let rows = stmt
            .query_map([], |row| {
                let mut map = HashMap::new();
                for (i, name) in col_names.iter().enumerate() {
                    let val: rusqlite::types::Value = row.get(i)?;
                    map.insert(name.clone(), sqlite_value_to_json(val));
                }
                Ok(map)
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Compression stats: actual vs naive storage size.
    pub fn compression_stats(&self) -> MetadataResult<AnalyticsRow> {
        let conn = self.conn.lock().unwrap();
        let (actual, naive): (i64, i64) = conn.query_row(
            "SELECT COALESCE(SUM(compressed_size), 0), COALESCE(SUM(CAST(raw_size AS INTEGER) * ref_count), 0) FROM segments_meta",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        let mut row = HashMap::new();
        row.insert(
            "actual_bytes".to_owned(),
            serde_json::Value::Number(actual.into()),
        );
        row.insert(
            "naive_bytes".to_owned(),
            serde_json::Value::Number(naive.into()),
        );
        let ratio = if naive == 0 {
            1.0
        } else {
            actual as f64 / naive as f64
        };
        row.insert(
            "compression_ratio".to_owned(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(ratio).unwrap_or(serde_json::Number::from(1)),
            ),
        );
        Ok(row)
    }
}

fn sqlite_value_to_json(val: rusqlite::types::Value) -> serde_json::Value {
    match val {
        rusqlite::types::Value::Null => serde_json::Value::Null,
        rusqlite::types::Value::Integer(i) => serde_json::Value::Number(i.into()),
        rusqlite::types::Value::Real(f) => serde_json::Number::from_f64(f)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        rusqlite::types::Value::Text(s) => serde_json::Value::String(s),
        rusqlite::types::Value::Blob(b) => serde_json::Value::String(hex::encode(b)),
    }
}

#[cfg(test)]
mod tests {
    use crate::index::metadata::MetadataIndex;
    use crate::types::{ConversationManifest, SegmentHash, SegmentRef, SegmentType, StoredSegment};
    use chrono::Utc;

    fn setup() -> MetadataIndex {
        let idx = MetadataIndex::open_in_memory().unwrap();

        let seg = StoredSegment {
            hash: SegmentHash("h1".to_owned()),
            segment_type: SegmentType::SystemPrompt,
            tokenizer: "cl100k".to_owned(),
            token_count: 100,
            compressed_data: vec![],
            raw_size: 400,
            compressed_size: 80,
            ref_count: 1,
            created_at: Utc::now(),
        };
        idx.upsert_segment(&seg).unwrap();

        let manifest = ConversationManifest {
            schema_version: crate::types::MANIFEST_SCHEMA_VERSION,
            id: "c1".to_owned(),
            application: Some("app-a".to_owned()),
            model: "gpt-4".to_owned(),
            tokenizer: "cl100k".to_owned(),
            total_tokens: 100,
            segments: vec![SegmentRef {
                segment_type: SegmentType::SystemPrompt,
                hash: SegmentHash("h1".to_owned()),
                token_count: 100,
                position: 0,
            }],
            created_at: Utc::now(),
            metadata: None,
        };
        idx.index_conversation(&manifest).unwrap();
        idx
    }

    #[test]
    fn cost_attribution_returns_rows() {
        let idx = setup();
        let rows = idx.cost_attribution().unwrap();
        assert!(!rows.is_empty());
        let row = &rows[0];
        assert!(row.contains_key("total_tokens"));
    }

    #[test]
    fn compression_stats_returns_actuals() {
        let idx = setup();
        let stats = idx.compression_stats().unwrap();
        let actual = stats["actual_bytes"].as_i64().unwrap();
        assert_eq!(actual, 80);
    }

    #[test]
    fn dedup_ratio_by_type_returns_rows() {
        let idx = setup();
        let rows = idx.dedup_ratio_by_type().unwrap();
        assert!(!rows.is_empty());
    }
}
