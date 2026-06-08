"""Tests for the stateless near-dedup primitives used by the distributed
(Spark) pipeline.

Run with:
    cd crates/stowken-python
    maturin develop
    pytest python/tests/test_near_dedup_primitives.py

Content is many distinct pseudo-random *words*: real text the tokenizer
round-trips losslessly, yet non-repetitive so zstd can't crush it (a delta is
then a genuine storage win). The near-duplicate appends a few words to the
canonical — a BPE-safe edit that yields a tiny `Copy(all)+Insert(small)` delta.
(Short or highly-repetitive segments compress so well that a raw delta can't beat
the full zstd frame, and an in-place edit re-tokenizes a whole run — both are the
cost model / BPE working as intended, not bugs.)
"""

import stowken

_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def _words(n: int, seed: int) -> str:
    """Deterministic space-joined pseudo-random 6-letter words."""
    x = seed & 0x7FFFFFFF
    out = []
    for _ in range(n):
        w = []
        for _ in range(6):
            x = (x * 1103515245 + 12345) & 0x7FFFFFFF
            w.append(_ALPHABET[x % 26])
        out.append("".join(w))
    return " ".join(out)


def _frame(vault, text: str):
    """Chunk a one-message conversation and return (hash, frame_bytes) for its
    system segment — a real, self-contained Stowken frame."""
    result = vault.chunk_only(
        {
            "model": "gpt-4",
            "tokenizer": "cl100k_base",
            "messages": [{"role": "system", "content": text}],
        }
    )
    seg = next(s for s in result["segments"] if s["segment_type"] == "system_prompt")
    return seg["hash"], bytes(seg["compressed_data"])


CANON = _words(600, seed=42)
# A near-duplicate: the canonical with a few words appended (BPE-safe edit).
VARIANT = CANON + " " + _words(5, seed=777)


def test_minhash_and_jaccard_detect_near_duplicate():
    vault = stowken.Stowken()
    _, frame_a = _frame(vault, CANON)
    _, frame_b = _frame(vault, VARIANT)

    sig_a = stowken.segment_minhash(frame_a)
    sig_b = stowken.segment_minhash(frame_b)
    assert len(sig_a) == 1024

    sim = stowken.signature_jaccard(sig_a, sig_b)
    assert sim > 0.9, f"appending 5 words to 600 should score very high, got {sim}"

    bands_a = stowken.signature_bands(sig_a)
    bands_b = stowken.signature_bands(sig_b)
    assert len(bands_a) == 16
    assert any(a == b for a, b in zip(bands_a, bands_b)), "must collide in a band"


def test_dissimilar_segments_do_not_match():
    vault = stowken.Stowken()
    _, frame_a = _frame(vault, CANON)
    _, frame_other = _frame(vault, _words(600, seed=99999))
    sim = stowken.signature_jaccard(
        stowken.segment_minhash(frame_a), stowken.segment_minhash(frame_other)
    )
    assert sim < 0.5, f"unrelated text should score low, got {sim}"


def test_delta_frame_round_trips_and_saves_bytes():
    vault = stowken.Stowken()
    hash_a, frame_a = _frame(vault, CANON)
    _, frame_b = _frame(vault, VARIANT)

    delta = stowken.make_delta_frame(hash_a, frame_a, frame_b)
    assert delta is not None, "delta should be a storage win for a near-duplicate"
    assert len(delta) < len(frame_b)

    # Decoding the delta against the canonical reconstructs the variant exactly
    # — same text as decoding the variant's own self-contained frame.
    via_delta = stowken.decode_segment_text(delta, "cl100k_base", canonical_zstd=frame_a)
    direct = stowken.decode_segment_text(frame_b, "cl100k_base")
    assert via_delta == direct


def test_decode_self_contained_frame_needs_no_canonical():
    vault = stowken.Stowken()
    _, frame_a = _frame(vault, CANON)
    text = stowken.decode_segment_text(frame_a, "cl100k_base")
    assert isinstance(text, str) and len(text) > 0


def test_decode_delta_without_canonical_errors():
    import pytest

    vault = stowken.Stowken()
    hash_a, frame_a = _frame(vault, CANON)
    _, frame_b = _frame(vault, VARIANT)
    delta = stowken.make_delta_frame(hash_a, frame_a, frame_b)
    assert delta is not None

    with pytest.raises(Exception):
        stowken.decode_segment_text(delta, "cl100k_base")  # missing canonical
