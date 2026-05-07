#!/usr/bin/env bash
# Benchmark sweep across all corpora and flag combinations.
# Results are written to bench_results/ as individual .txt files.
set -euo pipefail

BIN="./target/release/stowken"
OUT="bench_results"
mkdir -p "$OUT"

run() {
    local label="$1"; shift
    local outfile="$OUT/${label}.txt"
    echo "=== $label ===" | tee -a "$OUT/sweep.log"
    "$@" 2>&1 | tee "$outfile"
    echo "" >> "$OUT/sweep.log"
}

# ── drift (1 000 convs, synthetic near-dup corpus) ─────────────────────────
run drift_baseline              $BIN bench data/drift.jsonl --retrieve 1000
run drift_near08                $BIN --near-dedup 0.8 bench data/drift.jsonl --retrieve 1000
run drift_near07                $BIN --near-dedup 0.7 bench data/drift.jsonl --retrieve 1000
run drift_split256              $BIN --split-segments 256 bench data/drift.jsonl --retrieve 1000
run drift_split256_near08       $BIN --near-dedup 0.8 --split-segments 256 bench data/drift.jsonl --retrieve 1000
run drift_dict200               $BIN bench data/drift.jsonl --retrieve 1000 --dict-warmup 200
run drift_substrings            $BIN bench data/drift.jsonl --retrieve 1000 --with-substrings

# ── hh_rlhf (10 000 convs, chosen+rejected pairs) ──────────────────────────
run hh_baseline                 $BIN bench data/hh_rlhf.jsonl --retrieve 1000
run hh_near08                   $BIN --near-dedup 0.8 bench data/hh_rlhf.jsonl --retrieve 1000
run hh_split256                 $BIN --split-segments 256 bench data/hh_rlhf.jsonl --retrieve 1000
run hh_split256_near08          $BIN --near-dedup 0.8 --split-segments 256 bench data/hh_rlhf.jsonl --retrieve 1000
run hh_dict1000                 $BIN bench data/hh_rlhf.jsonl --retrieve 1000 --dict-warmup 1000
run hh_substrings               $BIN bench data/hh_rlhf.jsonl --retrieve 1000 --with-substrings

# ── lmsys (10 000 convs cap, diverse real-world) ───────────────────────────
run lmsys_baseline              $BIN bench data/lmsys.jsonl --max 10000 --retrieve 1000
run lmsys_split256              $BIN --split-segments 256 bench data/lmsys.jsonl --max 10000 --retrieve 1000
run lmsys_split128              $BIN --split-segments 128 bench data/lmsys.jsonl --max 10000 --retrieve 1000
run lmsys_dict2000              $BIN bench data/lmsys.jsonl --max 10000 --retrieve 1000 --dict-warmup 2000
run lmsys_substrings            $BIN bench data/lmsys.jsonl --max 10000 --retrieve 1000 --with-substrings

# ── open_assistant (3 482 convs, crowd-sourced Q&A) ────────────────────────
run oa_baseline                 $BIN bench data/open_assistant.jsonl --retrieve 1000
run oa_near08                   $BIN --near-dedup 0.8 bench data/open_assistant.jsonl --retrieve 1000
run oa_split256                 $BIN --split-segments 256 bench data/open_assistant.jsonl --retrieve 1000
run oa_dict500                  $BIN bench data/open_assistant.jsonl --retrieve 1000 --dict-warmup 500
run oa_substrings               $BIN bench data/open_assistant.jsonl --retrieve 1000 --with-substrings

echo "Sweep complete. Results in $OUT/"
