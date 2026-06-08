"""
Stowken — compressed, deduplicated storage for LLM token sequences.

Quick start::

    import stowken

    vault = stowken.Stowken()
    result = vault.store({
        "model": "gpt-4",
        "messages": [
            {"role": "system",    "content": "You are a helpful assistant."},
            {"role": "user",      "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    })
    print(result)         # StoreResult(id='...', ...)
    stats = vault.stats()
    print(stats)          # TokenUsageStats(...)
"""

from .stowken import (  # noqa: F401
    Stowken,
    StoreResult,
    TokenUsageStats,
    # Stateless near-dedup primitives for the distributed (Spark) pipeline.
    decode_segment_text,
    make_delta_frame,
    segment_minhash,
    signature_bands,
    signature_jaccard,
)

__all__ = [
    "Stowken",
    "StoreResult",
    "TokenUsageStats",
    "segment_minhash",
    "signature_bands",
    "signature_jaccard",
    "make_delta_frame",
    "decode_segment_text",
]
