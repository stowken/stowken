#!/usr/bin/env python3
"""Fetch the OpenAssistant Conversations (oasst1) dataset from HuggingFace and
emit JSONL conversations in Stowken's input format.

OpenAssistant is tree-structured: each message has a parent, and any path from
root to leaf is a valid conversation. We emit ONE conversation per message
tree by walking the highest-ranked child at every step (tie-broken by message
creation time).

Each emitted line is one conversation:

    {"id": "<message_tree_id>", "model": "openassistant", "tokenizer": "cl100k_base",
     "messages": [{"role": "user"|"assistant", "content": "..."}, ...]}

Usage:
    pip install datasets
    python scripts/fetch_openassistant.py --output corpus.jsonl
    python scripts/fetch_openassistant.py --output sample.jsonl --max 1000
    python scripts/fetch_openassistant.py --output en.jsonl --lang en

Then benchmark:
    stowken bench corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from typing import Iterable

try:
    from datasets import load_dataset
except ImportError:
    sys.stderr.write(
        "datasets is not installed. Run: pip install datasets\n"
    )
    sys.exit(1)


ROLE_MAP = {
    "prompter": "user",
    "assistant": "assistant",
}


def _best_child(children: list[dict]) -> dict:
    """Pick the highest-ranked child. Lower 'rank' = better in oasst.
    Falls back to creation time on ties or missing rank."""
    def key(msg: dict) -> tuple:
        rank = msg.get("rank")
        # rank can be None for root prompts or unranked messages — treat as worst.
        rank_key = rank if rank is not None else 1_000_000
        # created_date is an ISO string; lex-sort works as a tiebreaker.
        return (rank_key, msg.get("created_date", ""))
    return min(children, key=key)


def _walk_best_path(root: dict, children_by_parent: dict[str, list[dict]]) -> list[dict]:
    path = [root]
    cur = root
    while True:
        kids = children_by_parent.get(cur["message_id"], [])
        if not kids:
            break
        cur = _best_child(kids)
        path.append(cur)
    return path


def _conversation_from_path(
    tree_id: str, path: list[dict]
) -> dict | None:
    """Convert a list of oasst messages into a Stowken conversation dict.
    Returns None if the path produces no valid messages."""
    messages = []
    for msg in path:
        role = ROLE_MAP.get(msg.get("role"))
        if role is None:
            continue  # unknown role — skip
        text = msg.get("text", "")
        if not text:
            continue
        messages.append({"role": role, "content": text})
    if not messages:
        return None
    return {
        "id": tree_id,
        "model": "openassistant",
        "tokenizer": "cl100k_base",
        "messages": messages,
    }


def iter_conversations(
    split: str,
    lang_filter: str | None,
    max_count: int | None,
) -> Iterable[dict]:
    print(f"loading OpenAssistant/oasst1 split={split}…", file=sys.stderr)
    ds = load_dataset("OpenAssistant/oasst1", split=split)
    print(f"  {len(ds):,} messages loaded", file=sys.stderr)

    # Group by tree, build parent → children index per tree.
    trees: dict[str, list[dict]] = defaultdict(list)
    for row in ds:
        if lang_filter and row.get("lang") != lang_filter:
            continue
        trees[row["message_tree_id"]].append(dict(row))
    print(f"  {len(trees):,} message trees (after lang filter)", file=sys.stderr)

    emitted = 0
    for tree_id, msgs in trees.items():
        children_by_parent: dict[str, list[dict]] = defaultdict(list)
        root: dict | None = None
        for msg in msgs:
            parent_id = msg.get("parent_id")
            if parent_id is None:
                if root is None:
                    root = msg
                # If a tree has multiple roots (data anomaly), keep the first.
            else:
                children_by_parent[parent_id].append(msg)

        if root is None:
            continue

        path = _walk_best_path(root, children_by_parent)
        conv = _conversation_from_path(tree_id, path)
        if conv is None:
            continue

        yield conv
        emitted += 1
        if max_count is not None and emitted >= max_count:
            print(f"  reached --max {max_count}, stopping", file=sys.stderr)
            return

    print(f"  emitted {emitted:,} conversations", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", "-o", required=True, help="JSONL output path")
    ap.add_argument(
        "--split",
        default="train",
        choices=["train", "validation"],
        help="Dataset split to use (default: train)",
    )
    ap.add_argument(
        "--lang",
        default="en",
        help="Filter to one language (default: en). Use empty string for all.",
    )
    ap.add_argument(
        "--max",
        type=int,
        default=None,
        help="Cap on conversations emitted. Default: no cap.",
    )
    args = ap.parse_args()
    lang_filter = args.lang if args.lang else None

    with open(args.output, "w") as fh:
        for conv in iter_conversations(args.split, lang_filter, args.max):
            fh.write(json.dumps(conv, ensure_ascii=False))
            fh.write("\n")

    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
