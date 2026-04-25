//! Built-in tokenizer implementations.
//!
//! When the `tiktoken` feature is enabled (default), `get_tokenizer(name)`
//! returns a ready-to-use adapter for the common OpenAI BPE encodings:
//!
//! | Name            | Models                              |
//! |-----------------|-------------------------------------|
//! | `cl100k_base`   | GPT-4, GPT-3.5-turbo, Claude        |
//! | `o200k_base`    | GPT-4o, o1, o3                      |
//! | `p50k_base`     | `text-davinci-003`, Codex           |
//! | `r50k_base`     | GPT-3 (`davinci`, etc.)             |

use crate::types::{Token, TokenizerAdapter};

/// Return a built-in tokenizer adapter for the given encoding name, or `None`
/// if the name is unrecognised or the `tiktoken` feature is not compiled in.
pub fn get_tokenizer(name: &str) -> Option<Box<dyn TokenizerAdapter>> {
    #[cfg(feature = "tiktoken")]
    {
        TiktokenAdapter::new(name).map(|a| Box::new(a) as Box<dyn TokenizerAdapter>)
    }
    #[cfg(not(feature = "tiktoken"))]
    {
        let _ = name;
        None
    }
}

// ── tiktoken adapter ──────────────────────────────────────────────────────────

#[cfg(feature = "tiktoken")]
struct TiktokenAdapter {
    bpe: std::sync::Arc<tiktoken_rs::CoreBPE>,
    encoding_name: &'static str,
    vocab_size: u32,
}

#[cfg(feature = "tiktoken")]
impl TiktokenAdapter {
    fn new(name: &str) -> Option<Self> {
        // Each encoding is loaded once and cached in a static Arc.
        // Subsequent calls just clone the Arc pointer — no vocab loading.
        let (bpe, encoding_name, vocab_size) = match name {
            "cl100k_base" => {
                static CACHE: std::sync::OnceLock<std::sync::Arc<tiktoken_rs::CoreBPE>> = std::sync::OnceLock::new();
                let arc = CACHE.get_or_init(|| std::sync::Arc::new(tiktoken_rs::cl100k_base().expect("cl100k_base")));
                (std::sync::Arc::clone(arc), "cl100k_base", 100_277u32)
            }
            "o200k_base" => {
                static CACHE: std::sync::OnceLock<std::sync::Arc<tiktoken_rs::CoreBPE>> = std::sync::OnceLock::new();
                let arc = CACHE.get_or_init(|| std::sync::Arc::new(tiktoken_rs::o200k_base().expect("o200k_base")));
                (std::sync::Arc::clone(arc), "o200k_base", 200_019u32)
            }
            "p50k_base" => {
                static CACHE: std::sync::OnceLock<std::sync::Arc<tiktoken_rs::CoreBPE>> = std::sync::OnceLock::new();
                let arc = CACHE.get_or_init(|| std::sync::Arc::new(tiktoken_rs::p50k_base().expect("p50k_base")));
                (std::sync::Arc::clone(arc), "p50k_base", 50_281u32)
            }
            "p50k_edit" => {
                static CACHE: std::sync::OnceLock<std::sync::Arc<tiktoken_rs::CoreBPE>> = std::sync::OnceLock::new();
                let arc = CACHE.get_or_init(|| std::sync::Arc::new(tiktoken_rs::p50k_edit().expect("p50k_edit")));
                (std::sync::Arc::clone(arc), "p50k_edit", 50_281u32)
            }
            "r50k_base" => {
                static CACHE: std::sync::OnceLock<std::sync::Arc<tiktoken_rs::CoreBPE>> = std::sync::OnceLock::new();
                let arc = CACHE.get_or_init(|| std::sync::Arc::new(tiktoken_rs::r50k_base().expect("r50k_base")));
                (std::sync::Arc::clone(arc), "r50k_base", 50_257u32)
            }
            _ => return None,
        };
        Some(Self { bpe, encoding_name, vocab_size })
    }
}

#[cfg(feature = "tiktoken")]
impl TokenizerAdapter for TiktokenAdapter {
    fn tokenize(&self, text: &str) -> Vec<Token> {
        self.bpe.encode_ordinary(text)
    }

    fn detokenize(&self, tokens: &[Token]) -> String {
        self.bpe.decode(tokens).unwrap_or_default()
    }

    fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    fn name(&self) -> &str {
        self.encoding_name
    }
}

#[cfg(all(test, feature = "tiktoken"))]
mod tests {
    use super::*;

    #[test]
    fn cl100k_roundtrip() {
        let adapter = get_tokenizer("cl100k_base").expect("cl100k_base should load");
        let text = "Hello, world!";
        let tokens = adapter.tokenize(text);
        assert!(!tokens.is_empty());
        let recovered = adapter.detokenize(&tokens);
        assert_eq!(recovered, text);
    }

    #[test]
    fn o200k_roundtrip() {
        let adapter = get_tokenizer("o200k_base").expect("o200k_base should load");
        let tokens = adapter.tokenize("The quick brown fox");
        assert!(!tokens.is_empty());
    }

    #[test]
    fn unknown_name_returns_none() {
        assert!(get_tokenizer("gpt2_custom").is_none());
    }

    #[test]
    fn vocab_size_nonzero() {
        for name in ["cl100k_base", "o200k_base", "p50k_base", "r50k_base"] {
            let a = get_tokenizer(name).unwrap();
            assert!(a.vocab_size() > 0, "{name} vocab_size is 0");
        }
    }
}
