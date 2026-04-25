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

from .stowken import Stowken, StoreResult, TokenUsageStats  # noqa: F401

__all__ = ["Stowken", "StoreResult", "TokenUsageStats"]
