#!/usr/bin/env python3
"""Fetch the Anthropic HH-RLHF dataset from HuggingFace and emit JSONL
conversations in Stowken's input format.

HH-RLHF (Helpful and Harmless Reinforcement Learning from Human Feedback)
contains ~170K preference pairs. Each row has a `chosen` and a `rejected`
field — both are the same conversation up to the final assistant turn, where
they diverge. This makes it an excellent dedup stress-test:

  - Exact dedup:    the shared user turns are byte-identical across the pair.
  - Near-dedup:     chosen and rejected final turns are semantically similar
                    but worded differently — exactly the drift pattern.
  - Substring dedup: recurring boilerplate ("Sure, here's", "I'd be happy to")
                    appears across thousands of assistant turns.

No HuggingFace token required — the dataset is public.

# Usage

    pip install datasets
    python scripts/fetch_hh_rlhf.py --output hh_rlhf.jsonl
    python scripts/fetch_hh_rlhf.py --output hh_rlhf.jsonl --max 5000

    cargo run --release -p stowken-cli -- bench hh_rlhf.jsonl --retrieve 1000

# Format

Each HH-RLHF row emits TWO conversations (chosen + rejected), so --max 5000
produces up to 10,000 conversation lines. Pass --chosen-only to emit one per
row if you want a cleaner corpus without the deliberate duplicates.

Each emitted line:

    {"id": "<row_index>-chosen"|"<row_index>-rejected",
     "model": "claude-1",
     "tokenizer": "cl100k_base",
     "messages": [{"role": "user"|"assistant", "content": "..."}, ...]}
"""

from __future__ import annotations

import argparse
import json
import re
import sys

try:
    from datasets import load_dataset
except ImportError:
    sys.stderr.write("datasets is not installed. Run: pip install datasets\n")
    sys.exit(1)

# HH-RLHF conversations are stored as a single string with this turn pattern.
_TURN_RE = re.compile(r"\n\nHuman: |\n\nAssistant: ")


def _parse_turns(text: str) -> list[dict] | None:
    """Split an HH-RLHF conversation string into role/content message dicts."""
    # The string typically starts with "\n\nHuman: " — split on turn markers.
    parts = _TURN_RE.split(text)
    # Determine roles by scanning the original string for the markers.
    markers = _TURN_RE.findall(text)

    messages = []
    for marker, content in zip(markers, parts[1:]):
        role = "user" if "Human" in marker else "assistant"
        content = content.strip()
        if content:
            messages.append({"role": role, "content": content})

    return messages if messages else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", "-o", required=True, help="JSONL output path")
    ap.add_argument(
        "--max",
        type=int,
        default=None,
        help="Cap on source rows processed. Each row yields 2 conversations "
             "(chosen + rejected) unless --chosen-only. Default: no cap.",
    )
    ap.add_argument(
        "--split",
        default="train",
        choices=["train", "test"],
        help="Dataset split (default: train, ~169K rows).",
    )
    ap.add_argument(
        "--chosen-only",
        action="store_true",
        help="Emit only the chosen conversation per row (no deliberate dupes).",
    )
    args = ap.parse_args()

    print(f"loading Anthropic/hh-rlhf split={args.split}…", file=sys.stderr)
    ds = load_dataset("Anthropic/hh-rlhf", split=args.split)
    print(f"  {len(ds):,} rows loaded", file=sys.stderr)

    emitted = 0
    skipped = 0
    with open(args.output, "w") as fh:
        for idx, row in enumerate(ds):
            if args.max is not None and idx >= args.max:
                print(f"  reached --max {args.max} rows, stopping", file=sys.stderr)
                break

            variants = [("chosen", row.get("chosen", ""))]
            if not args.chosen_only:
                variants.append(("rejected", row.get("rejected", "")))

            for label, text in variants:
                messages = _parse_turns(text)
                if not messages:
                    skipped += 1
                    continue
                conv = {
                    "id": f"{idx}-{label}",
                    "model": "claude-1",
                    "tokenizer": "cl100k_base",
                    "messages": messages,
                }
                fh.write(json.dumps(conv, ensure_ascii=False))
                fh.write("\n")
                emitted += 1

    print(
        f"emitted {emitted:,} conversations to {args.output} "
        f"({skipped} skipped empty)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
