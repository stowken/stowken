"""Tests for the `_async` Python API.

Run with:
    pytest python/tests/test_async.py
"""

import asyncio
import os
import tempfile

import pytest

import stowken


def make_conv(id_: str, system: str = "Async test prompt.", user: str = "Hi", assistant: str = "Hello"):
    return {
        "id": id_,
        "model": "gpt-4",
        "tokenizer": "cl100k_base",
        "messages": [
            {"role": "system",    "content": system},
            {"role": "user",      "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def make_token_conv(id_: str, sys_tokens, user_tokens):
    return {
        "id": id_,
        "model": "gpt-4",
        "tokenizer": "cl100k_base",
        "messages": [
            {"role": "system", "tokens": sys_tokens},
            {"role": "user",   "tokens": user_tokens},
        ],
    }


# ── Basic async roundtrip ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_store_async_returns_StoreResult():
    v = stowken.Stowken()
    r = await v.store_async(make_conv("sa-1"))
    assert isinstance(r, stowken.StoreResult)
    assert r.id == "sa-1"
    assert r.new_segments > 0


@pytest.mark.asyncio
async def test_retrieve_async_returns_tokens():
    v = stowken.Stowken()
    sys_tokens = list(range(50))
    user_tokens = [999, 1000, 1001]
    await v.store_async(make_token_conv("ra-1", sys_tokens, user_tokens))
    result = await v.retrieve_async("ra-1")
    assert result["id"] == "ra-1"
    sys_seg = next(s for s in result["segments"] if s["segment_type"] == "system_prompt")
    assert sys_seg["tokens"] == sys_tokens


@pytest.mark.asyncio
async def test_stats_async_after_store():
    v = stowken.Stowken()
    await v.store_async(make_conv("st-1"))
    s = await v.stats_async()
    assert s.total_conversations == 1
    assert s.total_tokens > 0


@pytest.mark.asyncio
async def test_delete_async_removes_conversation():
    v = stowken.Stowken()
    await v.store_async(make_conv("del-1"))
    await v.delete_async("del-1")
    with pytest.raises(Exception):
        await v.retrieve_async("del-1")


@pytest.mark.asyncio
async def test_segment_stats_async():
    v = stowken.Stowken()
    await v.store_async(make_conv("ss-1"))
    stats = await v.segment_stats_async()
    assert isinstance(stats, list)
    assert len(stats) > 0


@pytest.mark.asyncio
async def test_gc_async_lifecycle():
    v = stowken.Stowken()
    await v.store_async(make_conv("gc-1"))
    await v.delete_async("gc-1")
    candidates = await v.gc_candidates_async()
    assert len(candidates) > 0
    freed = await v.gc_async()
    assert freed == len(candidates)
    after = await v.gc_candidates_async()
    assert after == []


# ── Concurrency: many async stores in parallel ─────────────────────────────────

@pytest.mark.asyncio
async def test_concurrent_async_stores():
    """Run 50 store_async calls concurrently and confirm everything ingests
    correctly. Validates the Arc-wrapped vault is safe to share across
    awaiting tasks."""
    v = stowken.Stowken()
    convs = [
        make_conv(
            f"conc-{i}",
            system="Shared concurrent prompt.",
            user=f"User question {i}",
            assistant=f"Answer {i}",
        )
        for i in range(50)
    ]
    results = await asyncio.gather(*(v.store_async(c) for c in convs))
    assert len(results) == 50
    assert all(r.id.startswith("conc-") for r in results)

    # Shared system prompt → at least 49 dedup hits across the concurrent batch.
    s = await v.stats_async()
    assert s.total_conversations == 50
    # 50 system + 50 user + 50 assistant = 150 refs
    assert s.total_segments == 150
    # 1 system (shared) + 50 user (unique) + 50 assistant (unique) = 101 unique
    assert s.unique_segments == 101


# ── Persistent vault with async ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_filesystem_vault_async_persistence():
    with tempfile.TemporaryDirectory() as data_dir:
        v1 = stowken.Stowken.open(data_dir)
        await v1.store_async(make_token_conv("p-1", [10, 20, 30], [40, 50]))
        del v1

        v2 = stowken.Stowken.open(data_dir)
        result = await v2.retrieve_async("p-1")
        sys_seg = next(s for s in result["segments"] if s["segment_type"] == "system_prompt")
        assert sys_seg["tokens"] == [10, 20, 30]


# ── Reindex via async ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_reindex_async():
    with tempfile.TemporaryDirectory() as data_dir:
        v = stowken.Stowken.open(data_dir)
        for i in range(3):
            await v.store_async(make_token_conv(f"ridx-{i}", [1, 2, 3], [10 + i]))
        before = await v.stats_async()
        del v

        os.remove(os.path.join(data_dir, "metadata.db"))

        v2 = stowken.Stowken.open(data_dir)
        report = await v2.reindex_async()
        assert report["conversations_indexed"] == 3
        after = await v2.stats_async()
        assert after.total_conversations == before.total_conversations
        assert after.unique_segments == before.unique_segments


# ── Mixing sync and async on the same vault ───────────────────────────────────

@pytest.mark.asyncio
async def test_sync_and_async_share_state():
    """Sync and async methods must operate on the same underlying vault state —
    they share the same Arc<VaultKind> internally."""
    v = stowken.Stowken()
    v.store(make_conv("mixed-sync"))
    await v.store_async(make_conv("mixed-async"))

    # Sync stats sees both.
    s_sync = v.stats()
    assert s_sync.total_conversations == 2

    # Async stats sees both.
    s_async = await v.stats_async()
    assert s_async.total_conversations == 2

    # Async retrieve of the sync-stored conversation.
    r = await v.retrieve_async("mixed-sync")
    assert r["id"] == "mixed-sync"

    # Sync retrieve of the async-stored conversation.
    r2 = v.retrieve("mixed-async")
    assert r2["id"] == "mixed-async"
