#!/usr/bin/env python3
"""Fetch the LMSys-Chat-1M dataset from HuggingFace and emit JSONL
conversations in Stowken's input format.

LMSys-Chat-1M is 1 million real ChatGPT-class conversations across many
models (gpt-4, claude-1, vicuna-13b, etc.). It's a good "real production
mix" benchmark — not as drift-heavy as a single-tenant chatbot but more
shaped than crowd-sourced uniqueness.

# Access

The dataset is gated. Before running this script:
  1. Visit https://huggingface.co/datasets/lmsys/lmsys-chat-1m
  2. Accept the dataset license
  3. Generate an HF token at https://huggingface.co/settings/tokens
  4. Export it: `export HF_TOKEN=<your-token>`

# Usage

    pip install datasets huggingface_hub
    huggingface-cli login   # or: export HF_TOKEN=...
    python scripts/fetch_lmsys.py --output lmsys.jsonl --max 10000

    stowken bench lmsys.jsonl --retrieve 1000

# Format

Each emitted line is one conversation:

    {"id": "<conversation_id>",
     "model": "<model_name>",
     "tokenizer": "cl100k_base",
     "messages": [{"role": "user"|"assistant", "content": "..."}, ...]}

LMSys conversations don't carry system prompts in the dataset itself —
each row is a `conversation` field with alternating user/assistant
turns. Stowken's exact-dedup will catch repeated prompts naturally;
near-dedup catches prompt drift; substring dedup catches recurring
boilerplate inside otherwise-unique messages.
"""

from __future__ import annotations

import argparse
import json
import sys

try:
    from datasets import load_dataset
except ImportError:
    sys.stderr.write(
        "datasets is not installed. Run: pip install datasets\n"
    )
    sys.exit(1)


ROLE_MAP = {
    "user": "user",
    "assistant": "assistant",
}


def _convert_row(row: dict) -> dict | None:
    """Convert one LMSys row to a Stowken conversation. Returns None if
    the row has no usable messages."""
    messages = []
    for msg in row.get("conversation", []):
        role = ROLE_MAP.get(msg.get("role"))
        if role is None:
            continue
        content = msg.get("content", "")
        if not content:
            continue
        messages.append({"role": role, "content": content})
    if not messages:
        return None
    return {
        "id": row.get("conversation_id", ""),
        "model": row.get("model", "unknown"),
        "tokenizer": "cl100k_base",
        "messages": messages,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", "-o", required=True, help="JSONL output path")
    ap.add_argument(
        "--max",
        type=int,
        default=None,
        help="Cap on conversations emitted. Default: no cap (all 1M).",
    )
    ap.add_argument(
        "--language",
        default="English",
        help="Filter by `language` column. Default: English. "
        "Use empty string for all.",
    )
    args = ap.parse_args()

    print("loading lmsys/lmsys-chat-1m (split=train)…", file=sys.stderr)
    print("  this is gated; ensure HF_TOKEN is set or you've run "
          "`huggingface-cli login`", file=sys.stderr)
    ds = load_dataset("lmsys/lmsys-chat-1m", split="train")
    print(f"  {len(ds):,} rows loaded", file=sys.stderr)

    emitted = 0
    skipped_lang = 0
    skipped_empty = 0
    with open(args.output, "w") as fh:
        for row in ds:
            if args.language and row.get("language") != args.language:
                skipped_lang += 1
                continue
            conv = _convert_row(dict(row))
            if conv is None:
                skipped_empty += 1
                continue
            fh.write(json.dumps(conv, ensure_ascii=False))
            fh.write("\n")
            emitted += 1
            if args.max is not None and emitted >= args.max:
                print(f"  reached --max {args.max}, stopping", file=sys.stderr)
                break

    print(
        f"emitted {emitted:,} conversations to {args.output} "
        f"(skipped {skipped_lang:,} on language filter, "
        f"{skipped_empty:,} empty)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
