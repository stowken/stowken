#!/usr/bin/env python3
"""Synthesize a 'production drift' corpus: a small set of system prompt
templates with interpolated tokens and slight RAG context variations, the
shape of real chatbot/agent traffic.

The structure is what near-dedup is designed to catch: many segments that
are *almost* identical but not byte-identical (so exact dedup misses
them), where storing each variant as a delta against a canonical is a
big win.

Usage:
    python scripts/synth_drift_corpus.py --output drift.jsonl --conversations 1000
    stowken bench drift.jsonl                    # baseline
    stowken --near-dedup 0.85 bench drift.jsonl  # with near-dedup
"""

from __future__ import annotations

import argparse
import json
import random
import sys

# ── Templates: token IDs roughly matching what tiktoken cl100k_base would
#    produce for short English text. We work directly in token space to
#    skip the tokenizer dependency in this script — Stowken accepts
#    `{"role": "user", "tokens": [...]}` directly.

def synth_system_prompt(rng: random.Random, size: int) -> list[int]:
    tokens = list(range(0, size))
    # Interpolate 3 slots with random tokens (the "drift" signal).
    edits = max(3, size // 33)
    for _ in range(edits):
        pos = rng.randint(0, size - 1)
        tokens[pos] = rng.randint(50_000, 60_000)
    return tokens


def synth_rag_context(rng: random.Random, size: int) -> list[int]:
    tokens = list(range(1000, 1000 + size))
    edits = max(4, size // 50)
    for _ in range(edits):
        pos = rng.randint(0, size - 1)
        tokens[pos] = rng.randint(70_000, 80_000)
    return tokens


def synth_user_turn(rng: random.Random, conv_id: int) -> list[int]:
    # Each user turn is unique — exact dedup gets nothing here.
    return [80_000 + (conv_id * 17 + i) % 5000 for i in range(40)]


def synth_assistant_turn(rng: random.Random, conv_id: int) -> list[int]:
    # Each assistant turn is also unique.
    return [85_000 + (conv_id * 23 + i) % 5000 for i in range(80)]


def synth_conversation(
    conv_id: int, rng: random.Random, system_tokens: int, rag_tokens: int
) -> dict:
    return {
        "id": f"drift-{conv_id:06d}",
        "model": "synth-gpt-4",
        "tokenizer": "cl100k_base",
        "application": "chatbot",
        "messages": [
            {"role": "system", "tokens": synth_system_prompt(rng, system_tokens)},
            # Inject the RAG context inside the user turn — close to how
            # production prompts often interleave retrieved chunks with the
            # actual user query.
            {
                "role": "user",
                "tokens": synth_rag_context(rng, rag_tokens) + synth_user_turn(rng, conv_id),
            },
            {"role": "assistant", "tokens": synth_assistant_turn(rng, conv_id)},
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", "-o", required=True, help="JSONL output path")
    ap.add_argument("--conversations", "-n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--system-tokens",
        type=int,
        default=100,
        help="System prompt size in tokens (default 100). Large values make "
        "drift cheaper relative to the canonical and favour near-dedup.",
    )
    ap.add_argument(
        "--rag-tokens",
        type=int,
        default=200,
        help="RAG context block size in tokens (default 200).",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    with open(args.output, "w") as fh:
        for i in range(args.conversations):
            fh.write(
                json.dumps(synth_conversation(i, rng, args.system_tokens, args.rag_tokens))
            )
            fh.write("\n")
    print(
        f"wrote {args.conversations} conversations to {args.output} "
        f"(system={args.system_tokens}t, rag={args.rag_tokens}t)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
