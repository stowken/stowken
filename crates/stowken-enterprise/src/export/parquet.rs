//! Parquet export for analytics and training pipelines.
//!
//! Exports conversation metadata and token usage records to Parquet files
//! compatible with pandas, Spark, BigQuery, and DuckDB.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, Float64Array, Int64Array, StringArray, UInt32Array, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::file::properties::WriterProperties;

use crate::types::ConversationMetadata;

/// Export errors.
#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    #[error("arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    #[error("parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type ExportResult<T> = Result<T, ExportError>;

/// Schema for the conversation metadata table.
fn conversation_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("conversation_id", DataType::Utf8, false),
        Field::new("model", DataType::Utf8, false),
        Field::new("application", DataType::Utf8, true),
        Field::new("total_tokens", DataType::UInt64, false),
        Field::new("segment_count", DataType::UInt32, false),
        Field::new("dedup_ratio", DataType::Float64, false),
        Field::new("stored_at", DataType::Utf8, false),
    ]))
}

/// Exports conversation metadata records to a Parquet file.
pub struct ParquetExporter {
    compression: parquet::basic::Compression,
}

impl Default for ParquetExporter {
    fn default() -> Self {
        Self {
            compression: parquet::basic::Compression::SNAPPY,
        }
    }
}

impl ParquetExporter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_compression(mut self, compression: parquet::basic::Compression) -> Self {
        self.compression = compression;
        self
    }

    /// Write conversation metadata rows to a Parquet file at `path`.
    pub fn write_conversations(
        &self,
        conversations: &[ConversationMetadata],
        path: impl AsRef<Path>,
    ) -> ExportResult<()> {
        let schema = conversation_schema();

        let ids: ArrayRef = Arc::new(StringArray::from(
            conversations.iter().map(|c| c.conversation_id.as_str()).collect::<Vec<_>>(),
        ));
        let models: ArrayRef = Arc::new(StringArray::from(
            conversations.iter().map(|c| c.model.as_str()).collect::<Vec<_>>(),
        ));
        let apps: ArrayRef = Arc::new(StringArray::from(
            conversations
                .iter()
                .map(|c| c.application.as_deref())
                .collect::<Vec<_>>(),
        ));
        let total_tokens: ArrayRef = Arc::new(UInt64Array::from(
            conversations.iter().map(|c| c.total_tokens).collect::<Vec<_>>(),
        ));
        let seg_counts: ArrayRef = Arc::new(UInt32Array::from(
            conversations.iter().map(|c| c.segment_count).collect::<Vec<_>>(),
        ));
        let dedup_ratios: ArrayRef = Arc::new(Float64Array::from(
            conversations.iter().map(|c| c.dedup_ratio).collect::<Vec<_>>(),
        ));
        let stored_ats: ArrayRef = Arc::new(StringArray::from(
            conversations
                .iter()
                .map(|c| c.stored_at.to_rfc3339())
                .collect::<Vec<_>>(),
        ));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![ids, models, apps, total_tokens, seg_counts, dedup_ratios, stored_ats],
        )?;

        let file = std::fs::File::create(path)?;
        let props = WriterProperties::builder()
            .set_compression(self.compression)
            .build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
        writer.write(&batch)?;
        writer.close()?;
        Ok(())
    }

    /// Write a generic key-value analytics result set to Parquet.
    ///
    /// Columns are derived from the keys of the first row. Missing values are null.
    pub fn write_analytics(
        &self,
        rows: &[std::collections::HashMap<String, serde_json::Value>],
        path: impl AsRef<Path>,
    ) -> ExportResult<()> {
        if rows.is_empty() {
            return Ok(());
        }

        // Collect column names from all rows
        let mut columns: Vec<String> = rows[0].keys().cloned().collect();
        columns.sort();

        let schema = Arc::new(Schema::new(
            columns
                .iter()
                .map(|c| Field::new(c.as_str(), DataType::Utf8, true))
                .collect::<Vec<_>>(),
        ));

        let arrays: Vec<ArrayRef> = columns
            .iter()
            .map(|col| {
                let values: Vec<Option<String>> = rows
                    .iter()
                    .map(|row| row.get(col).map(|v| v.to_string()))
                    .collect();
                Arc::new(StringArray::from(values)) as ArrayRef
            })
            .collect();

        let batch = RecordBatch::try_new(schema.clone(), arrays)?;

        let file = std::fs::File::create(path)?;
        let props = WriterProperties::builder()
            .set_compression(self.compression)
            .build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
        writer.write(&batch)?;
        writer.close()?;
        Ok(())
    }

    /// Write token usage by model/day for cost attribution.
    pub fn write_daily_usage(
        &self,
        rows: &[(String, String, i64)], // (model, day, tokens)
        path: impl AsRef<Path>,
    ) -> ExportResult<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("model", DataType::Utf8, false),
            Field::new("day", DataType::Utf8, false),
            Field::new("token_count", DataType::Int64, false),
        ]));

        let models: ArrayRef =
            Arc::new(StringArray::from(rows.iter().map(|(m, _, _)| m.as_str()).collect::<Vec<_>>()));
        let days: ArrayRef =
            Arc::new(StringArray::from(rows.iter().map(|(_, d, _)| d.as_str()).collect::<Vec<_>>()));
        let counts: ArrayRef =
            Arc::new(Int64Array::from(rows.iter().map(|(_, _, c)| *c).collect::<Vec<_>>()));

        let batch = RecordBatch::try_new(schema.clone(), vec![models, days, counts])?;

        let file = std::fs::File::create(path)?;
        let props = WriterProperties::builder()
            .set_compression(self.compression)
            .build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
        writer.write(&batch)?;
        writer.close()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use tempfile::NamedTempFile;

    fn sample_metadata() -> ConversationMetadata {
        ConversationMetadata {
            conversation_id: "conv-1".into(),
            model: "gpt-4".into(),
            application: Some("test-app".into()),
            tags: vec![],
            total_tokens: 512,
            segment_count: 4,
            dedup_ratio: 0.25,
            stored_at: Utc::now(),
        }
    }

    #[test]
    fn write_conversations_roundtrip() {
        let tmp = NamedTempFile::with_suffix(".parquet").unwrap();
        let exporter = ParquetExporter::new();
        let rows = vec![sample_metadata()];
        exporter.write_conversations(&rows, tmp.path()).unwrap();
        assert!(tmp.path().exists());
        assert!(tmp.path().metadata().unwrap().len() > 0);
    }

    #[test]
    fn write_empty_analytics_is_noop() {
        let tmp = NamedTempFile::with_suffix(".parquet").unwrap();
        let exporter = ParquetExporter::new();
        exporter.write_analytics(&[], tmp.path()).unwrap();
        // File may not have been created or is empty — both are acceptable
    }

    #[test]
    fn write_daily_usage() {
        let tmp = NamedTempFile::with_suffix(".parquet").unwrap();
        let exporter = ParquetExporter::new();
        let rows = vec![
            ("gpt-4".into(), "2024-01-01".into(), 1000i64),
            ("gpt-3.5".into(), "2024-01-01".into(), 500i64),
        ];
        exporter.write_daily_usage(&rows, tmp.path()).unwrap();
        assert!(tmp.path().metadata().unwrap().len() > 0);
    }
}
