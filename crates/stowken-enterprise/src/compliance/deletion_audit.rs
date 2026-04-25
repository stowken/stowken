//! GDPR deletion verification — prove that a user's data is fully purged.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use stowken::types::SegmentHash;

/// A deletion audit report proving that all data for a user has been removed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeletionReport {
    /// The subject whose data was deleted (user ID, email, etc.).
    pub subject: String,
    pub requested_at: DateTime<Utc>,
    pub completed_at: DateTime<Utc>,
    /// Conversation IDs that were deleted.
    pub deleted_conversations: Vec<String>,
    /// Segment hashes that were garbage-collected as a result.
    pub freed_segments: Vec<SegmentHash>,
    /// Whether all known data for this subject has been removed.
    pub verified_complete: bool,
    /// Any residual items that could not be confirmed as deleted.
    pub unverified_items: Vec<String>,
}

impl DeletionReport {
    /// Create a completed deletion report.
    pub fn completed(
        subject: impl Into<String>,
        requested_at: DateTime<Utc>,
        deleted_conversations: Vec<String>,
        freed_segments: Vec<SegmentHash>,
    ) -> Self {
        Self {
            subject: subject.into(),
            requested_at,
            completed_at: Utc::now(),
            deleted_conversations,
            freed_segments,
            verified_complete: true,
            unverified_items: vec![],
        }
    }

    /// Serialize this report to JSON for storage in a compliance record.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}

/// Verify that none of the given segment hashes remain in the backend.
///
/// Returns the set of hashes that were NOT found (i.e., successfully deleted).
pub async fn verify_segments_deleted<B: stowken::storage::StorageBackend>(
    backend: &B,
    hashes: &[SegmentHash],
) -> Result<VerificationResult, stowken::storage::StorageError> {
    let existing = backend.has_segments(hashes).await?;
    let existing_set: HashSet<&SegmentHash> = existing.iter().collect();
    let still_present: Vec<SegmentHash> = hashes
        .iter()
        .filter(|h| existing_set.contains(h))
        .cloned()
        .collect();
    let deleted: Vec<SegmentHash> = hashes
        .iter()
        .filter(|h| !existing_set.contains(h))
        .cloned()
        .collect();
    Ok(VerificationResult { deleted, still_present })
}

/// Result of a deletion verification sweep.
#[derive(Debug)]
pub struct VerificationResult {
    /// Hashes confirmed deleted from the backend.
    pub deleted: Vec<SegmentHash>,
    /// Hashes still present (deletion incomplete or failed).
    pub still_present: Vec<SegmentHash>,
}

impl VerificationResult {
    pub fn is_complete(&self) -> bool {
        self.still_present.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stowken::{storage::{MemoryBackend, StorageBackend}, types::*};
    use chrono::Utc;

    #[tokio::test]
    async fn verify_deleted_segments() {
        let backend = MemoryBackend::new();
        let seg = StoredSegment {
            hash: SegmentHash("abc123".into()),
            segment_type: SegmentType::UserTurn,
            tokenizer: "test".into(),
            token_count: 3,
            compressed_data: vec![1, 2, 3],
            raw_size: 12,
            compressed_size: 3,
            ref_count: 1,
            created_at: Utc::now(),
        };
        backend.put_segment(&seg).await.unwrap();

        let hashes = vec![SegmentHash("abc123".into()), SegmentHash("missing".into())];
        let result = verify_segments_deleted(&backend, &hashes).await.unwrap();
        // "missing" was never stored, so it's "deleted"
        // "abc123" is still present
        assert_eq!(result.still_present.len(), 1);
        assert_eq!(result.deleted.len(), 1);
        assert!(!result.is_complete());
    }

    #[test]
    fn deletion_report_serialization() {
        let report = DeletionReport::completed(
            "user-123",
            Utc::now(),
            vec!["conv-1".into(), "conv-2".into()],
            vec![SegmentHash("seg-abc".into())],
        );
        let json = report.to_json();
        assert!(json.contains("user-123"));
        assert!(json.contains("verified_complete"));
    }
}
