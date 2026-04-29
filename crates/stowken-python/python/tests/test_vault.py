"""Stowken Python binding tests.

Run with:
    cd crates/stowken-python
    maturin develop          # editable build + install into current venv
    pytest python/tests/     # run this file

Or against an already-installed wheel:
    pip install <path-to-wheel>
    pytest python/tests/
"""

import os
import tempfile

import pytest

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

    def test_retrieve_returns_full_tokens(self, vault):
        """retrieve() must return the actual u32 token IDs, not just counts."""
        sys_tokens = [9125, 25, 1472, 527, 264, 11190]
        user_tokens = [3923, 374, 220, 17, 489, 220, 17, 30]
        vault.store(make_token_conv("full-tok", sys_tokens, user_tokens))
        result = vault.retrieve("full-tok")
        sys_seg = next(s for s in result["segments"] if s["segment_type"] == "system_prompt")
        user_seg = next(s for s in result["segments"] if s["segment_type"] == "user_turn")
        assert sys_seg["tokens"] == sys_tokens
        assert user_seg["tokens"] == user_tokens


# ── Persistent vault ──────────────────────────────────────────────────────────

class TestPersistentVault:
    def test_filesystem_vault_persists_across_instances(self):
        with tempfile.TemporaryDirectory() as data_dir:
            tokens = list(range(50))

            v1 = stowken.Stowken.open(data_dir)
            v1.store(make_token_conv("persist-1", tokens, [1, 2, 3]))
            del v1  # drop the first vault — second vault sees only on-disk state

            v2 = stowken.Stowken.open(data_dir)
            result = v2.retrieve("persist-1")
            sys_seg = next(s for s in result["segments"] if s["segment_type"] == "system_prompt")
            assert sys_seg["tokens"] == tokens

    def test_filesystem_dedup_works_across_reopen(self):
        with tempfile.TemporaryDirectory() as data_dir:
            v1 = stowken.Stowken.open(data_dir)
            v1.store(make_conv("a"))
            del v1

            v2 = stowken.Stowken.open(data_dir)
            r = v2.store({**make_conv("b"), "id": "b"})
            assert r.deduped_segments > 0, "dedup should still work after reopen"


# ── Reindex ────────────────────────────────────────────────────────────────────

class TestReindex:
    def test_reindex_after_db_deletion_restores_stats(self):
        with tempfile.TemporaryDirectory() as data_dir:
            v = stowken.Stowken.open(data_dir)
            for i in range(3):
                v.store(make_token_conv(f"reidx-{i}", list(range(50)), [i, i + 1, i + 2]))
            stats_before = v.stats()
            del v

            # Wipe the index. Manifests + segment files survive.
            os.remove(os.path.join(data_dir, "metadata.db"))

            v2 = stowken.Stowken.open(data_dir)
            assert v2.stats().total_conversations == 0, "fresh DB should be empty"

            report = v2.reindex()
            assert report["conversations_indexed"] == 3
            assert report["segments_missing"] == 0

            stats_after = v2.stats()
            assert stats_after.total_conversations == stats_before.total_conversations
            assert stats_after.unique_segments == stats_before.unique_segments
            assert stats_after.total_segments == stats_before.total_segments

    def test_reindex_on_empty_vault(self):
        with tempfile.TemporaryDirectory() as data_dir:
            v = stowken.Stowken.open(data_dir)
            report = v.reindex()
            assert report["conversations_indexed"] == 0
            assert report["unique_segments_indexed"] == 0
            assert report["segments_missing"] == 0


# ── Dictionaries ──────────────────────────────────────────────────────────────

class TestDictionaries:
    def _seed(self, vault, n: int = 30):
        """Train-friendly seed: distinct content per conversation."""
        for i in range(n):
            sys_tokens = [(t * 13 + i * 7) % 50_000 for t in range(100)]
            user_tokens = [(t * 5 + i * 3) % 50_000 for t in range(30)]
            vault.store(make_token_conv(f"dict-seed-{i}", sys_tokens, user_tokens))

    def test_empty_vault_has_no_dicts(self):
        v = stowken.Stowken()
        assert v.list_dictionaries() == []

    def test_train_without_activate_registers_inactive(self):
        v = stowken.Stowken()
        self._seed(v)
        info = v.train_dictionary(samples=20)
        assert info["is_active"] is False
        assert info["sample_count"] == 20
        assert info["size_bytes"] > 0
        listed = v.list_dictionaries()
        assert len(listed) == 1
        assert listed[0]["id"] == info["id"]
        assert listed[0]["is_active"] is False

    def test_train_with_activate_flips_active(self):
        v = stowken.Stowken()
        self._seed(v)
        info = v.train_dictionary(samples=20, activate=True)
        assert info["is_active"] is True
        listed = v.list_dictionaries()
        active_ids = [d["id"] for d in listed if d["is_active"]]
        assert active_ids == [info["id"]]

    def test_activate_unknown_dict_raises(self):
        v = stowken.Stowken()
        with pytest.raises(Exception):
            v.activate_dictionary(99999)

    def test_too_few_samples_raises(self):
        v = stowken.Stowken()
        # Only one tiny conversation — far below min training samples.
        v.store(make_token_conv("solo", [1, 2, 3], [4, 5]))
        with pytest.raises(Exception) as excinfo:
            v.train_dictionary(samples=50)
        assert "not enough" in str(excinfo.value).lower()

    def test_rotation_preserves_old_frames(self):
        """The v0.1 data-loss bug: confirm old dict frames remain readable
        after a new dictionary is activated."""
        v = stowken.Stowken()
        self._seed(v, n=30)

        # Train and activate dict A.
        info_a = v.train_dictionary(samples=20, activate=True)

        # Store conv X under dict A.
        x_tokens = list(range(120))
        v.store(make_token_conv("X", x_tokens, [1, 2, 3]))

        # Seed more, train + activate dict B.
        for i in range(100, 130):
            sys_tokens = [(t * 17 + i * 11) % 50_000 for t in range(100)]
            v.store(make_token_conv(f"more-{i}", sys_tokens, [i]))
        info_b = v.train_dictionary(samples=20, activate=True)
        assert info_a["id"] != info_b["id"]

        # Store conv Y under dict B.
        y_tokens = list(range(5000, 5120))
        v.store(make_token_conv("Y", y_tokens, [4, 5, 6]))

        # Both retrievals must work.
        x = v.retrieve("X")
        y = v.retrieve("Y")
        x_sys = next(s for s in x["segments"] if s["segment_type"] == "system_prompt")
        y_sys = next(s for s in y["segments"] if s["segment_type"] == "system_prompt")
        assert x_sys["tokens"] == x_tokens, "old dict frame lost!"
        assert y_sys["tokens"] == y_tokens

    def test_dict_persistence_across_reopen(self):
        with tempfile.TemporaryDirectory() as data_dir:
            v = stowken.Stowken.open(data_dir)
            self._seed(v, n=30)
            info = v.train_dictionary(samples=20, activate=True)
            persisted_id = info["id"]
            persisted_tokens = list(range(7000, 7150))
            v.store(make_token_conv("persist", persisted_tokens, [99]))
            del v

            v2 = stowken.Stowken.open(data_dir)
            dicts = v2.list_dictionaries()
            assert len(dicts) == 1
            assert dicts[0]["id"] == persisted_id
            assert dicts[0]["is_active"] is True

            # The dict-compressed frame must round-trip.
            r = v2.retrieve("persist")
            sys_seg = next(s for s in r["segments"] if s["segment_type"] == "system_prompt")
            assert sys_seg["tokens"] == persisted_tokens


# ── Substring dedup (v0.5) ────────────────────────────────────────────────────

def _drift_system(seed: int, length: int = 250) -> list:
    """Match the synth-drift fixture: a deterministic base with a few
    interpolated tokens at seed-derived positions."""
    tokens = list(range(length))
    positions = [
        seed % length,
        (seed * 7) % length,
        (seed * 13) % length,
    ]
    for i, p in enumerate(positions):
        tokens[p] = 90_000 + (seed * 17 + i * 11) % 5_000
    return tokens


class TestSubstrings:
    def test_empty_vault_no_substrings(self):
        v = stowken.Stowken()
        assert v.list_substrings() == []

    def test_train_promotes_high_frequency_windows(self):
        v = stowken.Stowken()
        for i in range(30):
            v.store(make_token_conv(f"seed-{i}", _drift_system(i, 200), [1000 + i]))
        promoted = v.train_substrings(samples=30, min_occurrences=5)
        assert len(promoted) > 0
        for info in promoted:
            assert info["length"] >= 16
            assert info["source_occurrences"] >= 5

    def test_post_train_round_trip(self):
        v = stowken.Stowken()
        for i in range(30):
            v.store(make_token_conv(f"seed-{i}", _drift_system(i, 250), [i]))
        v.train_substrings(samples=30, min_occurrences=5)

        novel = _drift_system(999, 250)
        v.store(make_token_conv("post-train", novel, [88_888]))
        r = v.retrieve("post-train")
        sys = next(s for s in r["segments"] if s["segment_type"] == "system_prompt")
        assert sys["tokens"] == novel

    def test_substring_persistence_across_reopen(self):
        with tempfile.TemporaryDirectory() as data_dir:
            v = stowken.Stowken.open(data_dir)
            for i in range(30):
                v.store(make_token_conv(f"seed-{i}", _drift_system(i, 250), [i]))
            promoted = v.train_substrings(samples=30, min_occurrences=5)
            count = len(promoted)
            assert count > 0
            del v

            v2 = stowken.Stowken.open(data_dir)
            assert len(v2.list_substrings()) == count


# ── Near-duplicate detection ──────────────────────────────────────────────────

class TestNearDedup:
    def test_disabled_by_default(self):
        v = stowken.Stowken()
        # Empty vault, no near-dedup configured.
        assert v.near_dedup_index_size() == 0
        v.store(make_token_conv("a", list(range(50)), [1]))
        # LSH index stays empty when feature is disabled.
        assert v.near_dedup_index_size() == 0

    def test_variant_round_trips_with_near_dedup_enabled(self):
        v = stowken.Stowken(near_dedup=0.80)
        canonical = list(range(200))
        v.store(make_token_conv("canon", canonical, [1, 2, 3]))

        variant = canonical.copy()
        variant[10] = 99_999
        variant[150] = 88_888
        v.store(make_token_conv("variant", variant, [4, 5, 6]))

        # Variant must reconstruct to its original tokens via delta apply.
        r = v.retrieve("variant")
        sys = next(s for s in r["segments"] if s["segment_type"] == "system_prompt")
        assert sys["tokens"] == variant

    def test_lsh_index_grows_per_canonical(self):
        v = stowken.Stowken(near_dedup=0.80)
        # 3 conversations × 2 segments each = 6 LSH entries (none deduped
        # because the segments are all distinct).
        for i in range(3):
            v.store(make_token_conv(f"c{i}", [i, i + 1, i + 2], [i + 10, i + 11]))
        assert v.near_dedup_index_size() == 6

    def test_find_near_duplicates_via_python(self):
        v = stowken.Stowken(near_dedup=0.80)
        canonical = list(range(200))
        # Make canonical heavily-referenced so it shows up as a cluster.
        for i in range(6):
            v.store(make_token_conv(f"canon-{i}", canonical, [i + 100]))
        # Add a near-variant.
        variant = canonical.copy()
        variant[20] = 99_999
        v.store(make_token_conv("variant", variant, [200]))

        clusters = v.find_near_duplicates(threshold=0.80, min_refs=5)
        assert len(clusters) >= 1
        c = clusters[0]
        assert "canonical" in c
        assert c["canonical_ref_count"] >= 5
        assert isinstance(c["variants"], list)


# ── GC candidates ─────────────────────────────────────────────────────────────

class TestGCCandidates:
    def test_gc_candidates_returns_list(self, vault):
        candidates = vault.gc_candidates()
        assert isinstance(candidates, list)

    def test_gc_candidates_lists_orphans_without_deleting(self, vault):
        vault.store(make_conv("orphan-1"))
        vault.delete("orphan-1")
        before = vault.gc_candidates()
        assert len(before) > 0
        # Calling gc_candidates does NOT delete — list should still be non-empty.
        again = vault.gc_candidates()
        assert len(again) == len(before)
        # Now actually GC.
        freed = vault.gc()
        assert freed == len(before)
        assert vault.gc_candidates() == []
