"""Stowken Python binding tests.

Run with:
    cd crates/stowken-python
    maturin develop          # build + install editable
    pytest python/tests/     # run this file
"""

import pytest
import sys
import os

# Allow running from the crate root after `maturin develop`.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import stowken


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def vault():
    return stowken.Stowken()


def make_conv(id_: str, system: str = "You are helpful.", user: str = "Hello", assistant: str = "Hi!"):
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


def make_token_conv(id_: str, sys_tokens: list, user_tokens: list):
    return {
        "id": id_,
        "model": "gpt-4",
        "tokenizer": "cl100k_base",
        "messages": [
            {"role": "system", "tokens": sys_tokens},
            {"role": "user",   "tokens": user_tokens},
        ],
    }


# ── StoreResult shape ─────────────────────────────────────────────────────────

class TestStoreResult:
    def test_has_expected_fields(self, vault):
        r = vault.store(make_conv("sr-1"))
        assert isinstance(r.id, str)
        assert r.id == "sr-1"
        assert isinstance(r.total_segments, int)
        assert isinstance(r.new_segments, int)
        assert isinstance(r.deduped_segments, int)
        assert isinstance(r.bytes_saved, int)
        assert isinstance(r.compression_ratio, float)

    def test_new_segments_positive_on_first_store(self, vault):
        r = vault.store(make_conv("sr-2"))
        assert r.new_segments > 0

    def test_total_equals_new_plus_deduped(self, vault):
        r = vault.store(make_conv("sr-3"))
        assert r.total_segments == r.new_segments + r.deduped_segments

    def test_repr_includes_id(self, vault):
        r = vault.store(make_conv("sr-repr"))
        assert "sr-repr" in repr(r)


# ── Store / retrieve roundtrip ─────────────────────────────────────────────────

class TestStoreRetrieve:
    def test_retrieve_returns_dict_with_expected_keys(self, vault):
        vault.store(make_conv("rt-1"))
        result = vault.retrieve("rt-1")
        assert result["id"] == "rt-1"
        assert result["model"] == "gpt-4"
        assert "segments" in result
        assert isinstance(result["segments"], list)

    def test_retrieve_segments_have_expected_fields(self, vault):
        vault.store(make_conv("rt-2"))
        result = vault.retrieve("rt-2")
        for seg in result["segments"]:
            assert "segment_type" in seg
            assert "token_count" in seg
            assert "hash" in seg
            assert isinstance(seg["token_count"], int)
            assert seg["token_count"] > 0

    def test_retrieve_with_pre_tokenized_content(self, vault):
        tokens = list(range(50))
        vault.store(make_token_conv("rt-tok", tokens, list(range(10, 20))))
        result = vault.retrieve("rt-tok")
        assert result["total_tokens"] == 60

    def test_retrieve_unknown_id_raises(self, vault):
        with pytest.raises(Exception):
            vault.retrieve("nonexistent-conversation-id")


# ── Deduplication ──────────────────────────────────────────────────────────────

class TestDeduplication:
    def test_exact_dup_reports_deduped_segments(self, vault):
        conv1 = make_conv("dup-1")
        conv2 = {**make_conv("dup-2"), "id": "dup-2"}
        # Both conversations have identical message content.
        vault.store(conv1)
        r2 = vault.store(conv2)
        assert r2.deduped_segments > 0, "identical content should be deduped"

    def test_bytes_saved_positive_on_dedup(self, vault):
        vault.store(make_conv("bs-1"))
        r2 = vault.store({**make_conv("bs-2"), "id": "bs-2"})
        assert r2.bytes_saved > 0

    def test_unique_segments_does_not_grow_on_exact_dup(self, vault):
        vault.store(make_conv("us-1"))
        stats_before = vault.stats()
        vault.store({**make_conv("us-2"), "id": "us-2"})
        stats_after = vault.stats()
        assert stats_after.unique_segments == stats_before.unique_segments


# ── Stats ──────────────────────────────────────────────────────────────────────

class TestStats:
    def test_empty_vault_has_zero_conversations(self):
        v = stowken.Stowken()
        s = v.stats()
        assert s.total_conversations == 0

    def test_stats_after_store(self, vault):
        vault.store(make_conv("st-1"))
        s = vault.stats()
        assert s.total_conversations >= 1
        assert s.total_tokens > 0
        assert s.unique_segments >= 1
        assert isinstance(s.compression_ratio, float)
        assert isinstance(s.savings_percentage, float)

    def test_stats_repr_is_string(self, vault):
        vault.store(make_conv("st-repr"))
        s = vault.stats()
        assert isinstance(repr(s), str)
        assert "conversations=" in repr(s)


# ── Delete ─────────────────────────────────────────────────────────────────────

class TestDelete:
    def test_delete_removes_conversation(self, vault):
        vault.store(make_conv("del-1"))
        vault.delete("del-1")
        with pytest.raises(Exception):
            vault.retrieve("del-1")

    def test_delete_nonexistent_raises(self, vault):
        with pytest.raises(Exception):
            vault.delete("no-such-id")


# ── Segment stats ──────────────────────────────────────────────────────────────

class TestSegmentStats:
    def test_returns_list_of_dicts(self, vault):
        vault.store(make_conv("ss-1"))
        stats = vault.segment_stats()
        assert isinstance(stats, list)
        assert len(stats) > 0
        for s in stats:
            assert "segment_type" in s
            assert "unique_count" in s
            assert "total_references" in s
            assert "dedup_ratio" in s
            assert "total_tokens" in s
            assert "compressed_bytes" in s

    def test_system_prompt_segment_present(self, vault):
        vault.store(make_conv("ss-2"))
        stats = vault.segment_stats()
        types = [s["segment_type"] for s in stats]
        assert any("system" in t.lower() for t in types), f"expected system segment, got {types}"


# ── GC ─────────────────────────────────────────────────────────────────────────

class TestGC:
    def test_gc_returns_int(self, vault):
        result = vault.gc()
        assert isinstance(result, int)

    def test_gc_removes_orphaned_segments(self, vault):
        vault.store(make_conv("gc-1"))
        vault.delete("gc-1")
        freed = vault.gc()
        assert freed > 0, "gc should have freed at least one segment"


# ── Compression flag ───────────────────────────────────────────────────────────

class TestCompressionFlag:
    def test_compression_disabled_vault_still_stores(self):
        v = stowken.Stowken(enable_compression=False)
        r = v.store(make_conv("nc-1"))
        assert r.new_segments > 0

    def test_repr_contains_vault_name(self):
        v = stowken.Stowken()
        assert "Stowken" in repr(v)


# ── Tokens field ───────────────────────────────────────────────────────────────

class TestTokenField:
    def test_explicit_tokens_stored_correctly(self, vault):
        sys_tokens = list(range(100))
        user_tokens = [999, 1000, 1001]
        r = vault.store(make_token_conv("tok-1", sys_tokens, user_tokens))
        assert r.new_segments > 0
        # total_tokens should match input count
        result = vault.retrieve("tok-1")
        assert result["total_tokens"] == len(sys_tokens) + len(user_tokens)
