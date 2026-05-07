//! Built-in OpenAI adapters for embeddings and conversation summarization.
//!
//! Gated behind the `semantic-search` feature. Uses `reqwest::blocking`
//! because the Stowken adapter traits are sync — async embedding calls are
//! bridged via `tokio::task::spawn_blocking` inside the vault.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use stowken::types::{EmbeddingAdapter, SummarizerAdapter};

const OPENAI_EMBEDDINGS_URL: &str = "https://api.openai.com/v1/embeddings";
const OPENAI_CHAT_URL: &str = "https://api.openai.com/v1/chat/completions";

const SUMMARY_SYSTEM_PROMPT: &str = "Summarize the following AI agent \
conversation in one paragraph. Capture the user's intent, the agent's \
behavior (including any tool calls), and the outcome. Do not editorialize.";

// ── Retry helpers ─────────────────────────────────────────────────────────────

enum RetryError {
    /// Hit a rate limit; sleep for this duration then retry.
    RateLimit(std::time::Duration),
    /// Non-retriable error; propagate immediately.
    Fatal(String),
}

/// Call `f` up to `max_attempts` times, sleeping on 429s. Returns the first
/// successful result or the last fatal error.
fn retry_on_rate_limit<T, F>(max_attempts: u32, mut f: F) -> Result<T, String>
where
    F: FnMut() -> Result<T, RetryError>,
{
    let mut last_err = String::new();
    for attempt in 0..max_attempts {
        match f() {
            Ok(v) => return Ok(v),
            Err(RetryError::Fatal(e)) => return Err(e),
            Err(RetryError::RateLimit(wait)) => {
                // Back off at least the server-suggested wait, plus a small
                // exponential component to spread retries.
                let backoff = wait + std::time::Duration::from_millis(200 * (1 << attempt));
                eprintln!(
                    "  rate limited (attempt {}/{}), retrying in {:.1}s…",
                    attempt + 1,
                    max_attempts,
                    backoff.as_secs_f32()
                );
                std::thread::sleep(backoff);
                last_err = format!("rate limited after {max_attempts} attempts");
            }
        }
    }
    Err(last_err)
}

/// Extract the Retry-After duration from a 429 response, falling back to 1s.
fn parse_retry_after(resp: &reqwest::blocking::Response) -> std::time::Duration {
    resp.headers()
        .get("retry-after")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<f64>().ok())
        .map(std::time::Duration::from_secs_f64)
        .unwrap_or(std::time::Duration::from_secs(1))
}

// ── Client construction ───────────────────────────────────────────────────────

/// Build a `reqwest::blocking::Client` on a fresh OS thread to avoid the
/// nested-Tokio-runtime panic that occurs when building inside an async context.
fn build_blocking_client() -> reqwest::blocking::Client {
    std::thread::spawn(|| {
        reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .expect("building reqwest client")
    })
    .join()
    .expect("reqwest client builder thread panicked")
}

// ── Embedding adapter ─────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct OpenAiEmbeddingAdapter {
    api_key: String,
    model: String,
    dimension: usize,
    client: reqwest::blocking::Client,
}

impl OpenAiEmbeddingAdapter {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>, dimension: usize) -> Self {
        Self { api_key: api_key.into(), model: model.into(), dimension, client: build_blocking_client() }
    }

    #[allow(dead_code)]
    pub fn small(api_key: impl Into<String>) -> Self {
        Self::new(api_key, "text-embedding-3-small", 1536)
    }

    #[allow(dead_code)]
    pub fn large(api_key: impl Into<String>) -> Self {
        Self::new(api_key, "text-embedding-3-large", 3072)
    }
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    input: &'a [&'a str],
    model: &'a str,
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedItem>,
}

#[derive(Deserialize)]
struct EmbedItem {
    embedding: Vec<f32>,
}

impl EmbeddingAdapter for OpenAiEmbeddingAdapter {
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        let owned: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let model = self.model.clone();
        std::thread::spawn(move || {
            let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
            let body = EmbedRequest { input: &refs, model: &model };
            retry_on_rate_limit(8, || {
                let resp = client
                    .post(OPENAI_EMBEDDINGS_URL)
                    .bearer_auth(&api_key)
                    .json(&body)
                    .send()
                    .map_err(|e| RetryError::Fatal(format!("openai embeddings request failed: {e}")))?;
                if resp.status().as_u16() == 429 {
                    return Err(RetryError::RateLimit(parse_retry_after(&resp)));
                }
                if !resp.status().is_success() {
                    let status = resp.status();
                    let body = resp.text().unwrap_or_default();
                    return Err(RetryError::Fatal(format!(
                        "openai embeddings returned {status}: {}",
                        body.chars().take(500).collect::<String>()
                    )));
                }
                let parsed: EmbedResponse = resp
                    .json()
                    .map_err(|e| RetryError::Fatal(format!("decoding openai embeddings response: {e}")))?;
                Ok(parsed.data.into_iter().map(|i| i.embedding).collect())
            })
        })
        .join()
        .map_err(|_| "embedding thread panicked".to_string())?
    }

    fn dimension(&self) -> usize { self.dimension }
    fn model_name(&self) -> &str { &self.model }
}

// ── Summarizer adapter ────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct OpenAiSummarizerAdapter {
    api_key: String,
    model: String,
    client: reqwest::blocking::Client,
}

impl OpenAiSummarizerAdapter {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self { api_key: api_key.into(), model: model.into(), client: build_blocking_client() }
    }

    #[allow(dead_code)]
    pub fn default_model(api_key: impl Into<String>) -> Self {
        Self::new(api_key, "gpt-4o-mini")
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    temperature: f32,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessageContent,
}

#[derive(Deserialize)]
struct ChatMessageContent {
    content: String,
}

impl SummarizerAdapter for OpenAiSummarizerAdapter {
    fn summarize(&self, conversation_text: &str) -> Result<String, String> {
        let text = conversation_text.to_string();
        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let model = self.model.clone();
        std::thread::spawn(move || {
            retry_on_rate_limit(8, || {
                let body = ChatRequest {
                    model: &model,
                    messages: vec![
                        ChatMessage { role: "system", content: SUMMARY_SYSTEM_PROMPT },
                        ChatMessage { role: "user",   content: &text },
                    ],
                    temperature: 0.2,
                };
                let resp = client
                    .post(OPENAI_CHAT_URL)
                    .bearer_auth(&api_key)
                    .json(&body)
                    .send()
                    .map_err(|e| RetryError::Fatal(format!("openai chat request failed: {e}")))?;
                if resp.status().as_u16() == 429 {
                    return Err(RetryError::RateLimit(parse_retry_after(&resp)));
                }
                if !resp.status().is_success() {
                    let status = resp.status();
                    let body = resp.text().unwrap_or_default();
                    return Err(RetryError::Fatal(format!(
                        "openai chat returned {status}: {}",
                        body.chars().take(500).collect::<String>()
                    )));
                }
                let parsed: ChatResponse = resp
                    .json()
                    .map_err(|e| RetryError::Fatal(format!("decoding openai chat response: {e}")))?;
                parsed
                    .choices
                    .into_iter()
                    .next()
                    .map(|c| c.message.content)
                    .ok_or_else(|| RetryError::Fatal("openai chat returned no choices".to_owned()))
            })
        })
        .join()
        .map_err(|_| "summarizer thread panicked".to_string())?
    }

    fn model_name(&self) -> &str { &self.model }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

pub fn require_openai_key(opt: Option<String>) -> Result<String> {
    opt.context("OpenAI API key not provided. Pass --openai-key or set OPENAI_API_KEY.")
}

pub fn openai_dimension_for(model: &str) -> Option<usize> {
    match model {
        "text-embedding-3-small" => Some(1536),
        "text-embedding-3-large" => Some(3072),
        "text-embedding-ada-002" => Some(1536),
        _ => None,
    }
}
