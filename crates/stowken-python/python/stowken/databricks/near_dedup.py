"""
Near-duplicate detection for the distributed pipeline.

Exact dedup (the ingest path) collapses byte-identical segments. This pass
catches segments that are *almost* identical — interpolated org names, A/B
prompt variants, one tweaked rule — which exact dedup misses. It runs over the
``segments`` table *after* ingest (idempotent, re-runnable), re-encoding each
near-duplicate variant as a Stowken ``0x04`` delta frame against a canonical
segment and recording the link in ``segments.canonical_hash``.

Algorithm (all in Spark, using the stateless ``stowken`` primitives per row):

  1. Candidates = self-contained segments not already deltas, above a token
     floor (short segments don't yield a delta win and bloat candidate gen).
  2. MinHash signature per segment; split into 16 LSH bands.
  3. Within each (band_index, band_hash) bucket, elect the most-referenced
     segment as representative and pair every member with it → candidate pairs.
     (A full self-join here is O(N²) and explodes when many segments are
     near-duplicates of a few templates; the representative scheme is O(N).)
  4. Verify with full-signature Jaccard ≥ ``threshold``.
  5. Orient each pair toward the "more canonical" node (higher ref_count, tie →
     smaller hash). Keep only variant→canonical edges whose canonical is a
     *root* (never itself a variant) so deltas never chain. Each variant takes
     its best-Jaccard canonical.
  6. Build the delta frame; keep it only when it's a storage win (the
     ``make_delta_frame`` cost check). MERGE the winners into ``segments``.

NOTE (PySpark type-hint inference): like ``ingest.py`` this module avoids
``from __future__ import annotations`` so pandas-UDF signature inference sees
real classes, not strings.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .backend import DatabricksBackend


@dataclass
class NearDedupResult:
    candidates: int
    """Self-contained segments considered (above the token floor)."""
    variants_written: int
    """Segments rewritten as deltas this run."""
    bytes_before: int
    """Total stored bytes of those variants' frames before the rewrite."""
    bytes_after: int
    """Total stored bytes of their delta frames after the rewrite."""

    @property
    def bytes_saved(self) -> int:
        return self.bytes_before - self.bytes_after

    @property
    def savings_pct(self) -> float:
        return (
            100.0 * self.bytes_saved / self.bytes_before if self.bytes_before else 0.0
        )


def near_dedup_segments(
    backend: "DatabricksBackend",
    *,
    threshold: float = 0.85,
    min_token_count: int = 64,
) -> NearDedupResult:
    """
    Run one near-dedup pass over ``backend``'s segments table. Idempotent:
    variants written gain a ``canonical_hash`` and are excluded next run.

    ``threshold`` is the Jaccard bar for declaring a near-duplicate (0.80–0.95
    is sensible). ``min_token_count`` skips short segments where a delta can't
    beat the full frame.
    """
    import pandas as pd
    from pyspark.sql import Window
    from pyspark.sql.functions import (
        col,
        length,
        pandas_udf,
        posexplode,
        row_number,
        when,
    )
    from pyspark.sql.functions import sum as sum_
    from pyspark.sql.types import ArrayType, StringType

    spark = backend.spark

    # ── UDFs: per-row calls into the stateless Rust primitives ────────────────
    @pandas_udf("binary")
    def sig_udf(content: pd.Series) -> pd.Series:
        import stowken

        out = []
        for blob in content:
            if blob is None:
                out.append(None)
                continue
            data = bytes(blob)
            # Only self-contained varint(+zstd) frames are decodable here;
            # 0x03/0x04/0x05 need registries/canonicals and aren't produced on
            # the Databricks ingest path.
            if not data or data[0] not in (0x01, 0x02):
                out.append(None)
                continue
            try:
                out.append(stowken.segment_minhash(data))
            except Exception:
                out.append(None)
        return pd.Series(out)

    @pandas_udf(ArrayType(StringType()))
    def bands_udf(sig: pd.Series) -> pd.Series:
        import stowken

        # Band hashes are u64; carry them as decimal strings to dodge Spark's
        # signed-long range on the join key.
        return pd.Series(
            [
                None
                if s is None
                else [str(b) for b in stowken.signature_bands(bytes(s))]
                for s in sig
            ]
        )

    @pandas_udf("double")
    def jaccard_udf(a: pd.Series, b: pd.Series) -> pd.Series:
        import stowken

        return pd.Series(
            [stowken.signature_jaccard(bytes(x), bytes(y)) for x, y in zip(a, b)]
        )

    @pandas_udf("binary")
    def delta_udf(
        canonical_hash: pd.Series,
        canonical_zstd: pd.Series,
        variant_zstd: pd.Series,
    ) -> pd.Series:
        import stowken

        out = []
        for h, c, v in zip(canonical_hash, canonical_zstd, variant_zstd):
            try:
                out.append(stowken.make_delta_frame(h, bytes(c), bytes(v)))
            except Exception:
                out.append(None)
        return pd.Series(out)

    # ── 1-2. Candidates → signatures → bands ──────────────────────────────────
    candidates = (
        spark.table(backend.segments_table)
        .where(col("canonical_hash").isNull() & (col("token_count") >= min_token_count))
        .select("segment_hash", "content_zstd", "ref_count")
    )
    candidate_count = candidates.count()
    if candidate_count == 0:
        return NearDedupResult(0, 0, 0, 0)

    sigged = candidates.withColumn("sig", sig_udf(col("content_zstd"))).where(
        col("sig").isNotNull()
    )
    exploded = sigged.select(
        col("segment_hash"),
        col("ref_count"),
        col("sig"),
        posexplode(bands_udf(col("sig"))).alias("band_idx", "band_hash"),
    )

    # ── 3. Candidate pairs by shared LSH bucket (representative scheme) ────────
    #
    # Within each (band_idx, band_hash) bucket, elect the most-referenced segment
    # as the bucket *representative* and emit only (representative, member) pairs
    # — O(N) per bucket. A full self-join here is O(N²) and explodes precisely on
    # the workload near-dedup is meant to excel at: when many segments are
    # near-duplicates of a few templates they collide into a handful of giant
    # buckets, and the self-join fans out to ~N²/2 Jaccard calls per bucket
    # (measured: a 100k-segment pass ran >90 min with zero progress). The
    # representative scheme keeps candidate generation linear. Recall is
    # preserved in practice because each segment lands in 16 bands — 16 chances
    # to meet its family's representative — and correctness is unaffected: step 4
    # still verifies every surviving pair by full-signature Jaccard, so no merge
    # happens below ``threshold``.
    rep_win = Window.partitionBy("band_idx", "band_hash").orderBy(
        col("ref_count").desc(), col("segment_hash").asc()
    )
    reps = (
        exploded.withColumn("_rk", row_number().over(rep_win))
        .where(col("_rk") == 1)
        .select(
            "band_idx",
            "band_hash",
            col("segment_hash").alias("rep_hash"),
            col("ref_count").alias("rep_ref"),
            col("sig").alias("rep_sig"),
        )
    )
    pairs = (
        exploded.join(reps, on=["band_idx", "band_hash"])
        .where(col("segment_hash") != col("rep_hash"))
        .select(
            col("rep_hash").alias("h1"),
            col("rep_ref").alias("r1"),
            col("rep_sig").alias("s1"),
            col("segment_hash").alias("h2"),
            col("ref_count").alias("r2"),
            col("sig").alias("s2"),
        )
        .distinct()
    )

    # ── 4. Verify by full-signature Jaccard ───────────────────────────────────
    scored = pairs.withColumn("sim", jaccard_udf(col("s1"), col("s2"))).where(
        col("sim") >= threshold
    )

    # ── 5. Orient edges variant→canonical; keep root canonicals only ──────────
    # "more canonical" = higher ref_count, tie broken by smaller hash.
    h1_is_canonical = (col("r1") > col("r2")) | (
        (col("r1") == col("r2")) & (col("h1") < col("h2"))
    )
    edges = scored.select(
        when(h1_is_canonical, col("h2")).otherwise(col("h1")).alias("variant"),
        when(h1_is_canonical, col("h1")).otherwise(col("h2")).alias("canonical"),
        col("sim"),
    )
    variants = edges.select("variant").distinct()
    # Drop edges whose canonical is itself a variant — prevents delta chaining.
    rooted = edges.join(
        variants.withColumnRenamed("variant", "canonical"),
        on="canonical",
        how="left_anti",
    )
    best_win = Window.partitionBy("variant").orderBy(
        col("sim").desc(), col("canonical").asc()
    )
    best = (
        rooted.withColumn("_rn", row_number().over(best_win))
        .where(col("_rn") == 1)
        .select("variant", "canonical", "sim")
    )

    # ── 6. Build delta frames; keep storage wins ──────────────────────────────
    blobs = spark.table(backend.segments_table).select("segment_hash", "content_zstd")
    joined = (
        best.join(
            blobs.select(
                col("segment_hash").alias("variant"),
                col("content_zstd").alias("variant_zstd"),
            ),
            on="variant",
        )
        .join(
            blobs.select(
                col("segment_hash").alias("canonical"),
                col("content_zstd").alias("canonical_zstd"),
            ),
            on="canonical",
        )
        .withColumn(
            "delta",
            delta_udf(col("canonical"), col("canonical_zstd"), col("variant_zstd")),
        )
        .where(col("delta").isNotNull())
        .select(
            col("variant").alias("segment_hash"),
            col("delta").alias("content_zstd"),
            col("canonical").alias("canonical_hash"),
            col("sim").cast("float").alias("canonical_similarity"),
            length(col("variant_zstd")).alias("_before"),
            length(col("delta")).alias("_after"),
        )
    )

    # Metrics before the MERGE consumes the rows (no caching on serverless).
    agg = joined.agg(
        sum_("_before").alias("before"), sum_("_after").alias("after")
    ).first()
    bytes_before = int((agg["before"] or 0) if agg else 0)
    bytes_after = int((agg["after"] or 0) if agg else 0)

    written = backend.put_segment_deltas(
        joined.select(
            "segment_hash", "content_zstd", "canonical_hash", "canonical_similarity"
        )
    )

    return NearDedupResult(
        candidates=candidate_count,
        variants_written=written,
        bytes_before=bytes_before,
        bytes_after=bytes_after,
    )
