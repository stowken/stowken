//! Stowken CLI — store, retrieve, inspect, export, and maintain a vault.
//!
//! Data directory defaults to `~/.local/share/stowken` (XDG) on Linux/Mac
//! and `%APPDATA%\stowken` on Windows.  Override with `--data-dir` or the
//! `STOWKEN_DATA_DIR` environment variable.

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use tabled::{Table, Tabled};

use stowken::{
    storage::{FilesystemBackend, MemoryBackend},
    types::{
        AnalyticsQuery, Conversation, ExportConfig, ExportFormat, Message, MessageContent,
        StowkenConfig,
    },
    Stowken,
};

#[cfg(feature = "semantic-search")]
mod openai_adapter;

// ── CLI structure ─────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "stowken",
    version,
    about = "Compressed, deduplicated storage for LLM token sequences",
    long_about = None,
)]
struct Cli {
    /// Directory where vault data is stored.
    #[arg(long, env = "STOWKEN_DATA_DIR", global = true)]
    data_dir: Option<PathBuf>,

    /// Disable zstd compression.
    #[arg(long, global = true)]
    no_compression: bool,

    /// Enable near-duplicate detection at this Jaccard threshold (typical:
    /// 0.85). When set, segments that are 0x04-stored as deltas against
    /// existing similar segments. Off by default.
    #[arg(long, global = true)]
    near_dedup: Option<f64>,

    /// Reuse a near-duplicate canonical's embedding for the variant instead
    /// of computing a fresh one, when their Jaccard similarity meets this
    /// threshold. Stricter than --near-dedup (typical: 0.97). Off by default.
    #[arg(long, global = true)]
    embed_reuse: Option<f64>,

    /// Split long messages into sub-segments of at most N tokens. Text
    /// messages are split at paragraph boundaries first; pre-tokenized
    /// messages are split at the hard token limit. Subsequent chunks use
    /// the Continuation segment type, increasing exact-dedup hit rate for
    /// messages that share common openings, closings, or boilerplate.
    #[arg(long, global = true, value_name = "N")]
    split_segments: Option<usize>,

    /// Output as JSON (where applicable).
    #[arg(long, short = 'j', global = true)]
    json: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum SubstringsAction {
    /// Discover and register high-frequency token-level substrings.
    Train {
        /// Number of segments to sample for discovery.
        #[arg(long, default_value_t = 500)]
        samples: usize,
        /// Minimum distinct-segment occurrences for a window to be promoted.
        #[arg(long, default_value_t = 5)]
        min_occurrences: u32,
    },
    /// List all registered substrings.
    List,
    /// Retroactively rewrite pre-train segments to use the substring
    /// registry. Walks every segment, decompresses it, tries `0x05`
    /// encoding, and replaces the stored bytes when smaller.
    /// Skips delta (0x04) and already-substring (0x05) frames.
    Compact,
    /// Drop substrings from the registry whose only references in any
    /// 0x05 frame have been deleted. Walks every segment to compute the
    /// reference set, then prunes the registry.
    Gc,
}

#[derive(Subcommand)]
enum DictAction {
    /// Train a new dictionary from existing segments. Without --activate the
    /// dict is registered but not used for new compressions.
    Train {
        /// Number of segments to sample for training.
        #[arg(long, default_value_t = 200)]
        samples: usize,
        /// Activate the dictionary immediately after training.
        #[arg(long)]
        activate: bool,
    },
    /// List all known dictionaries.
    List,
    /// Mark `id` as the dictionary new compressions will use.
    Activate {
        id: u32,
    },
}

#[derive(Subcommand)]
enum Command {
    /// Store a conversation from a JSON file or stdin.
    Store {
        /// Path to a JSON file containing the conversation. Use `-` for stdin.
        #[arg(default_value = "-")]
        file: String,

        /// Override the model field in the conversation.
        #[arg(long)]
        model: Option<String>,

        /// Tag the conversation with an application name.
        #[arg(long)]
        application: Option<String>,

        /// Assign a specific ID instead of auto-generating one.
        #[arg(long)]
        id: Option<String>,
    },

    /// Bulk-store conversations from a JSONL file or a directory of JSON/JSONL
    /// files into the real vault (unlike `bench`, which uses an in-memory vault).
    /// One conversation per line for JSONL; one conversation per file for JSON.
    Ingest {
        /// JSONL file, single JSON file, or directory of JSON/JSONL files.
        input: PathBuf,

        /// Tag every ingested conversation with an application name.
        #[arg(long)]
        application: Option<String>,

        /// Cap on conversations ingested. Default: no cap.
        #[arg(long)]
        max: Option<u64>,

        /// Skip embedding after ingest (useful for bulk loads; run `embed-all` afterward).
        #[cfg(feature = "semantic-search")]
        #[arg(long)]
        no_embed: bool,
    },

    /// Retrieve a conversation by ID and print it as JSON.
    Retrieve {
        /// Conversation ID.
        id: String,

        /// Print raw segment token arrays instead of the manifest.
        #[arg(long)]
        segments: bool,
    },

    /// Retrieve multiple conversations by ID and print them as human-readable
    /// text with role markers. Designed to pair with semantic-search: pipe hit
    /// IDs in here to read the actual conversations.
    RetrieveBatch {
        /// Conversation IDs (space-separated). Alternatively, pass one ID per
        /// line via stdin with `--stdin`.
        ids: Vec<String>,

        /// Read IDs from stdin (one per line) instead of positional args.
        #[arg(long)]
        stdin: bool,
    },

    /// Delete a conversation by ID.
    Delete {
        /// Conversation ID.
        id: String,

        /// Skip confirmation prompt.
        #[arg(long, short = 'y')]
        yes: bool,
    },

    /// Show aggregate storage statistics.
    Stats,

    /// Show per-segment-type breakdown.
    SegmentStats,

    /// Export training data (user/assistant pairs) to stdout or a file.
    Export {
        /// Output file path. Defaults to stdout.
        #[arg(long, short = 'o')]
        output: Option<PathBuf>,

        /// Export format: jsonl, huggingface.
        #[arg(long, default_value = "jsonl")]
        format: String,

        /// Filter by model.
        #[arg(long)]
        model: Option<String>,

        /// Filter by application.
        #[arg(long)]
        application: Option<String>,

        /// Include system prompts in the export.
        #[arg(long)]
        include_system: bool,

        /// Include context segments.
        #[arg(long)]
        include_context: bool,

        /// Skip duplicate user/assistant pairs.
        #[arg(long)]
        deduplicate: bool,

        /// Maximum number of conversations to export.
        #[arg(long)]
        max: Option<u64>,
    },

    /// Run garbage collection (delete zero-reference segments).
    Gc {
        /// Dry-run: report what would be deleted without deleting.
        #[arg(long, short = 'n')]
        dry_run: bool,
    },

    /// Rebuild the SQLite metadata index from on-disk manifests.
    ///
    /// Use after a corrupted or deleted metadata.db, after restoring a vault
    /// from a backup that didn't include the index, or to reconcile drift.
    /// Does not touch the segment files or backend ref-counts.
    Reindex,

    /// Manage zstd compression dictionaries.
    ///
    /// Dictionaries are trained from existing segments and can give
    /// 5–15 percentage points of additional compression on unique-per-call
    /// content. Multiple dictionaries coexist; rotation is non-destructive.
    Dict {
        #[command(subcommand)]
        action: DictAction,
    },

    /// Manage shared token-level substrings for v0.5 substring dedup.
    ///
    /// Train discovers high-frequency token-level substrings across stored
    /// segments and registers them as shared blobs. Future ingests use
    /// frame format 0x05 (substring references) when it beats the
    /// baseline. Old segments are unaffected; `compact` to retroactively
    /// rewrite them lands in v0.5.x.
    Substrings {
        #[command(subcommand)]
        action: SubstringsAction,
    },

    /// Audit unique system prompts by reference count (find prompt drift).
    AuditSystemPrompts {
        /// Show only prompts with at least this many references.
        #[arg(long, default_value_t = 1)]
        min_refs: u64,
        /// Maximum number of prompts to print.
        #[arg(long, default_value_t = 50)]
        limit: u64,
    },

    /// Token attribution by model, application, segment type, or day.
    Cost {
        /// Comma-separated grouping dimensions: model, application, segment_type, day.
        /// Example: --by application,model
        #[arg(long, default_value = "model")]
        by: String,
        /// Filter by model.
        #[arg(long)]
        model: Option<String>,
        /// Filter by application.
        #[arg(long)]
        application: Option<String>,
    },

    /// Find heavily-referenced segments — refactor candidates.
    FindDuplicates {
        /// Minimum reference count to surface.
        #[arg(long, default_value_t = 10)]
        threshold: u64,
        /// Maximum number of segments to print.
        #[arg(long, default_value_t = 25)]
        limit: u64,
    },

    /// Surface clusters of segments that are *almost* identical — prompt
    /// drift detection. Requires the LSH index to have been populated by
    /// running with `--near-dedup <threshold>` during ingest, or by
    /// running `stowken reindex` after enabling near-dedup.
    FindNearDuplicates {
        /// Jaccard similarity threshold to declare a near-duplicate.
        /// Conservative defaults; lower for more aggressive matching.
        #[arg(long, default_value_t = 0.85)]
        threshold: f64,
        /// Minimum ref count for a segment to be considered as a canonical
        /// for clustering. Higher = surface only the high-traffic clusters.
        #[arg(long, default_value_t = 5)]
        min_refs: u64,
        /// Maximum clusters to print.
        #[arg(long, default_value_t = 25)]
        limit: u64,
    },

    /// Benchmark Stowken on a directory of conversations.
    ///
    /// Ingests every `*.json` (one conversation per file) and `*.jsonl` (one
    /// per line) under INPUT into a fresh in-memory vault, then reports the
    /// three-stage savings breakdown plus per-segment-type detail. After
    /// ingest, samples N conversations and times retrieval to report
    /// decompression throughput + latency percentiles.
    ///
    /// Does not touch the configured data directory — bench is read-only
    /// w.r.t. your real vault.
    Bench {
        /// Directory or single file to ingest.
        input: PathBuf,
        /// Cap on conversations ingested. Default: no cap.
        #[arg(long)]
        max: Option<u64>,
        /// Number of conversations to retrieve in the post-ingest
        /// retrieval pass. Default: min(ingested, 1000).
        #[arg(long)]
        retrieve: Option<u64>,
        /// Ingest the first N conversations, train and activate a zstd
        /// dictionary from them, then ingest the remainder with the
        /// dictionary active. Shows dictionary compression benefit on
        /// the tail of the corpus.
        #[arg(long, value_name = "N")]
        dict_warmup: Option<usize>,
        /// After ingesting all conversations, run substring discovery
        /// (train) and compact all segments to use the substring registry.
        /// Reports final storage after the compaction pass.
        #[arg(long)]
        with_substrings: bool,
    },

    /// Print the resolved data directory and exit.
    DataDir,

    /// Embed every stored segment + conversation summary that lacks an
    /// embedding for the configured model. Run once after enabling
    /// semantic search on an existing vault. Subsequent stores embed
    /// incrementally via the embed-on-store hook.
    ///
    /// Summary strategy: "concat" (default) is free — it detokenizes and
    /// concatenates the conversation text without any API call and works
    /// well for the vast majority of queries. Use "llm" only when you need
    /// rich behavioral summaries (e.g. "sessions where the agent gave up").
    #[cfg(feature = "semantic-search")]
    EmbedAll {
        #[arg(long, env = "OPENAI_API_KEY")]
        openai_key: Option<String>,
        #[arg(long, default_value = "text-embedding-3-small")]
        model: String,
        /// Summary strategy: "concat" (default, free, recommended) or "llm"
        /// (one GPT call per conversation — slower and costs tokens, but
        /// captures intent and behavior rather than raw content).
        #[arg(long, default_value = "concat")]
        summary_strategy: String,
        /// LLM model used when --summary-strategy=llm.
        #[arg(long, default_value = "gpt-4o-mini")]
        summary_model: String,
    },

    /// Search conversations by semantic meaning across both segment-level
    /// and conversation-summary embeddings.
    #[cfg(feature = "semantic-search")]
    SemanticSearch {
        /// Natural-language query.
        query: String,
        #[arg(long, env = "OPENAI_API_KEY")]
        openai_key: Option<String>,
        #[arg(long, default_value = "text-embedding-3-small")]
        model: String,
        /// "segment", "conversation", or "both".
        #[arg(long, default_value = "both")]
        granularity: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
        #[arg(long, default_value_t = 0.0)]
        min_score: f32,
        #[arg(long)]
        application: Option<String>,
        #[arg(long)]
        filter_model: Option<String>,
        /// Filter by segment type (system_prompt, user_turn, assistant_turn,
        /// tool_call, tool_result, context). Only meaningful when
        /// --granularity is segment or both.
        #[arg(long)]
        segment_type: Option<String>,
    },

    /// Search across system prompts only.
    #[cfg(feature = "semantic-search")]
    SearchPrompts {
        query: String,
        #[arg(long, env = "OPENAI_API_KEY")]
        openai_key: Option<String>,
        #[arg(long, default_value = "text-embedding-3-small")]
        model: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },

    /// Search across assistant responses only.
    #[cfg(feature = "semantic-search")]
    SearchResponses {
        query: String,
        #[arg(long, env = "OPENAI_API_KEY")]
        openai_key: Option<String>,
        #[arg(long, default_value = "text-embedding-3-small")]
        model: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },

    /// Search across user turns only.
    #[cfg(feature = "semantic-search")]
    SearchUserQuestions {
        query: String,
        #[arg(long, env = "OPENAI_API_KEY")]
        openai_key: Option<String>,
        #[arg(long, default_value = "text-embedding-3-small")]
        model: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },

    /// Search across tool calls only.
    #[cfg(feature = "semantic-search")]
    SearchToolUses {
        query: String,
        #[arg(long, env = "OPENAI_API_KEY")]
        openai_key: Option<String>,
        #[arg(long, default_value = "text-embedding-3-small")]
        model: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },

    /// Search across conversation summaries only.
    #[cfg(feature = "semantic-search")]
    SearchConversations {
        query: String,
        #[arg(long, env = "OPENAI_API_KEY")]
        openai_key: Option<String>,
        #[arg(long, default_value = "text-embedding-3-small")]
        model: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },

    /// Cluster all conversation summaries with k-means++.
    #[cfg(feature = "semantic-search")]
    Cluster {
        #[arg(long, env = "OPENAI_API_KEY")]
        openai_key: Option<String>,
        #[arg(long, default_value = "text-embedding-3-small")]
        model: String,
        /// Number of clusters.
        #[arg(long, default_value_t = 10)]
        k: usize,
        /// Conversations per cluster to surface as representatives.
        #[arg(long, default_value_t = 3)]
        representatives: usize,
    },

    /// Surface conversations that are far from any cluster centroid —
    /// candidates for "this conversation is unusual" review.
    #[cfg(feature = "semantic-search")]
    Outliers {
        #[arg(long, env = "OPENAI_API_KEY")]
        openai_key: Option<String>,
        #[arg(long, default_value = "text-embedding-3-small")]
        model: String,
        #[arg(long, default_value_t = 10)]
        k: usize,
        /// Distance (1 - cosine) above which a conversation is flagged.
        #[arg(long, default_value_t = 0.5)]
        threshold: f32,
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let data_dir = resolve_data_dir(cli.data_dir)?;
    let config = StowkenConfig {
        enable_compression: !cli.no_compression,
        near_dedup_threshold: cli.near_dedup,
        embed_reuse_threshold: cli.embed_reuse,
    };
    let split_segments = cli.split_segments;

    match cli.command {
        Command::DataDir => {
            println!("{}", data_dir.display());
            return Ok(());
        }
        Command::Store { file, model, application, id } => {
            cmd_store(&data_dir, config, file, model, application, id, cli.json).await?;
        }
        Command::Ingest { input, application, max, #[cfg(feature = "semantic-search")] no_embed } => {
            #[cfg(feature = "semantic-search")]
            let embed = !no_embed;
            #[cfg(not(feature = "semantic-search"))]
            let embed = false;
            cmd_ingest(&data_dir, config, input, application, max, embed).await?;
        }
        Command::Retrieve { id, segments } => {
            cmd_retrieve(&data_dir, config, id, segments, cli.json).await?;
        }
        Command::RetrieveBatch { ids, stdin } => {
            cmd_retrieve_batch(&data_dir, config, ids, stdin, cli.json).await?;
        }
        Command::Delete { id, yes } => {
            cmd_delete(&data_dir, config, id, yes).await?;
        }
        Command::Stats => {
            cmd_stats(&data_dir, config, cli.json).await?;
        }
        Command::SegmentStats => {
            cmd_segment_stats(&data_dir, config, cli.json).await?;
        }
        Command::Export {
            output,
            format,
            model,
            application,
            include_system,
            include_context,
            deduplicate,
            max,
        } => {
            let fmt = match format.as_str() {
                "jsonl" => ExportFormat::Jsonl,
                "huggingface" | "hf" => ExportFormat::HuggingFace,
                other => anyhow::bail!("unknown format '{other}'; use 'jsonl' or 'huggingface'"),
            };
            let export_config = ExportConfig {
                format: fmt,
                include_system_prompts: include_system,
                include_context,
                deduplicate_pairs: deduplicate,
                tokenizer: None,
                model,
                application,
                max_conversations: max,
            };
            let vault = open_vault(&data_dir, config).await?;
            let stats = if let Some(path) = output {
                let mut file = std::fs::File::create(&path)
                    .with_context(|| format!("creating {}", path.display()))?;
                vault.export_training_data(export_config, &mut file).await.context("exporting")?
            } else {
                let mut stdout = io::stdout();
                vault.export_training_data(export_config, &mut stdout).await.context("exporting")?
            };
            eprintln!(
                "Exported {} pairs ({} unique, {} tokens)",
                stats.total_pairs, stats.unique_pairs, stats.tokens_exported
            );
        }
        Command::Gc { dry_run } => {
            cmd_gc(&data_dir, config, dry_run).await?;
        }
        Command::Reindex => {
            cmd_reindex(&data_dir, config, cli.json).await?;
        }
        Command::Dict { action } => {
            cmd_dict(&data_dir, config, action, cli.json).await?;
        }
        Command::Substrings { action } => {
            cmd_substrings(&data_dir, config, action, cli.json).await?;
        }
        Command::AuditSystemPrompts { min_refs, limit } => {
            cmd_audit_system_prompts(&data_dir, config, min_refs, limit, cli.json).await?;
        }
        Command::Cost { by, model, application } => {
            cmd_cost(&data_dir, config, &by, model, application, cli.json).await?;
        }
        Command::FindDuplicates { threshold, limit } => {
            cmd_find_duplicates(&data_dir, config, threshold, limit, cli.json).await?;
        }
        Command::FindNearDuplicates { threshold, min_refs, limit } => {
            cmd_find_near_duplicates(&data_dir, config, threshold, min_refs, limit, cli.json).await?;
        }
        Command::Bench { input, max, retrieve, dict_warmup, with_substrings } => {
            cmd_bench(config, split_segments, input, max, retrieve, dict_warmup, with_substrings, cli.json).await?;
        }
        #[cfg(feature = "semantic-search")]
        Command::EmbedAll { openai_key, model, summary_strategy, summary_model } => {
            cmd_embed_all(&data_dir, config, openai_key, model, summary_strategy, summary_model, cli.json).await?;
        }
        #[cfg(feature = "semantic-search")]
        Command::SemanticSearch {
            query, openai_key, model, granularity, limit, min_score,
            application, filter_model, segment_type,
        } => {
            cmd_semantic_search(
                &data_dir, config, query, openai_key, model, granularity, limit,
                min_score, application, filter_model, segment_type, cli.json,
            ).await?;
        }
        #[cfg(feature = "semantic-search")]
        Command::SearchPrompts { query, openai_key, model, limit } => {
            cmd_typed_search(&data_dir, config, query, openai_key, model, limit, "prompts", cli.json).await?;
        }
        #[cfg(feature = "semantic-search")]
        Command::SearchResponses { query, openai_key, model, limit } => {
            cmd_typed_search(&data_dir, config, query, openai_key, model, limit, "responses", cli.json).await?;
        }
        #[cfg(feature = "semantic-search")]
        Command::SearchUserQuestions { query, openai_key, model, limit } => {
            cmd_typed_search(&data_dir, config, query, openai_key, model, limit, "user_questions", cli.json).await?;
        }
        #[cfg(feature = "semantic-search")]
        Command::SearchToolUses { query, openai_key, model, limit } => {
            cmd_typed_search(&data_dir, config, query, openai_key, model, limit, "tool_uses", cli.json).await?;
        }
        #[cfg(feature = "semantic-search")]
        Command::SearchConversations { query, openai_key, model, limit } => {
            cmd_typed_search(&data_dir, config, query, openai_key, model, limit, "conversations", cli.json).await?;
        }
        #[cfg(feature = "semantic-search")]
        Command::Cluster { openai_key, model, k, representatives } => {
            cmd_cluster(&data_dir, config, openai_key, model, k, representatives, cli.json).await?;
        }
        #[cfg(feature = "semantic-search")]
        Command::Outliers { openai_key, model, k, threshold, limit } => {
            cmd_outliers(&data_dir, config, openai_key, model, k, threshold, limit, cli.json).await?;
        }
    }

    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn resolve_data_dir(override_path: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(p) = override_path {
        return Ok(p);
    }
    let base = directories::BaseDirs::new()
        .context("could not determine home directory")?;
    Ok(base.data_local_dir().join("stowken"))
}

async fn open_vault(data_dir: &PathBuf, config: StowkenConfig) -> Result<Stowken<FilesystemBackend>> {
    let db_path = data_dir.join("metadata.db");
    let backend = FilesystemBackend::new(data_dir)
        .await
        .with_context(|| format!("opening filesystem backend at {}", data_dir.display()))?;

    Stowken::open(backend, config, db_path.to_str().context("non-UTF-8 db path")?)
        .await
        .context("opening vault")
}

fn read_input(file: &str) -> Result<String> {
    if file == "-" {
        let stdin = io::stdin();
        let mut buf = String::new();
        for line in stdin.lock().lines() {
            buf.push_str(&line?);
            buf.push('\n');
        }
        Ok(buf)
    } else {
        std::fs::read_to_string(file)
            .with_context(|| format!("reading {file}"))
    }
}

/// Parse a JSON conversation object into the internal `Conversation` type.
///
/// Accepts the OpenAI chat format: `{ model, messages, tokenizer?, application?, id? }`.
/// Each message may include a `tokens: [number]` field; otherwise the content
/// field is encoded with a UTF-8 byte-pair fallback tokenizer.
/// Parse a JSON conversation object into the internal `Conversation` type.
///
/// Accepts the OpenAI chat format: `{ model, messages, tokenizer?, application?, id? }`.
/// Each message may include a `tokens: [number]` field with pre-tokenized content;
/// otherwise the `content` string is passed through as `MessageContent::Text` and
/// tokenized automatically by the vault's built-in tiktoken encoder.
fn parse_conversation_json(raw: &str) -> Result<Conversation> {
    let v: serde_json::Value = serde_json::from_str(raw).context("parsing JSON")?;

    let model = v["model"].as_str().context("missing 'model' field")?.to_owned();
    let tokenizer = v["tokenizer"].as_str().unwrap_or("cl100k_base").to_owned();
    let application = v["application"].as_str().map(str::to_owned);
    let id = v["id"].as_str().map(str::to_owned);

    let messages_raw = v["messages"].as_array().context("missing 'messages' array")?;
    let mut messages = Vec::with_capacity(messages_raw.len());

    for msg in messages_raw {
        let role = msg["role"].as_str().context("message missing 'role'")?.to_owned();

        let content = if let Some(tokens_arr) = msg.get("tokens").and_then(|t| t.as_array()) {
            let tokens: Vec<u32> = tokens_arr.iter().map(|t| t.as_u64().unwrap_or(0) as u32).collect();
            MessageContent::Tokens(tokens)
        } else {
            let text = match msg.get("content") {
                Some(serde_json::Value::String(s)) => s.clone(),
                Some(serde_json::Value::Array(parts)) => parts
                    .iter()
                    .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                    .collect::<Vec<_>>()
                    .join(""),
                _ => String::new(),
            };
            MessageContent::Text(text)
        };

        let name = msg.get("name").and_then(|v| v.as_str()).map(str::to_owned);
        let tool_call_id = msg.get("tool_call_id").and_then(|v| v.as_str()).map(str::to_owned);
        messages.push(Message { role, content, name, tool_call_id });
    }

    Ok(Conversation { id, model, tokenizer, application, messages, metadata: None })
}

// ── Commands ──────────────────────────────────────────────────────────────────

async fn cmd_store(
    data_dir: &PathBuf,
    config: StowkenConfig,
    file: String,
    model: Option<String>,
    application: Option<String>,
    id: Option<String>,
    json: bool,
) -> Result<()> {
    let raw = read_input(&file)?;
    let mut conv = parse_conversation_json(&raw)?;

    if let Some(m) = model { conv.model = m; }
    if application.is_some() { conv.application = application; }
    if id.is_some() { conv.id = id; }

    let vault = open_vault(data_dir, config).await?;
    let result = vault.store(conv).await.context("storing conversation")?;

    if json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        println!("Stored conversation: {}", result.id);
        println!("  Segments : {} total ({} new, {} deduped)", result.total_segments, result.new_segments, result.deduped_segments);
        println!("  Saved    : {} bytes", result.bytes_saved);
        println!("  Ratio    : {:.3}x", result.compression_ratio);
    }
    Ok(())
}

async fn cmd_ingest(
    data_dir: &PathBuf,
    config: StowkenConfig,
    input: PathBuf,
    application: Option<String>,
    max: Option<u64>,
    _embed: bool,
) -> Result<()> {
    let conversations = collect_conversations(&input, max)?;
    let total = conversations.len();
    if total == 0 {
        eprintln!("No conversations found in {}", input.display());
        return Ok(());
    }

    let vault = open_vault(data_dir, config).await?;

    let mut stored = 0u64;
    let mut failed = 0u64;
    let mut deduped_total = 0u64;
    let mut bytes_saved_total = 0u64;

    for mut conv in conversations {
        if application.is_some() && conv.application.is_none() {
            conv.application = application.clone();
        }
        match vault.store(conv).await {
            Ok(r) => {
                stored += 1;
                deduped_total += r.deduped_segments as u64;
                bytes_saved_total += r.bytes_saved;
                if stored % 100 == 0 {
                    eprintln!("  {stored}/{total} stored…");
                }
            }
            Err(e) => {
                eprintln!("  warning: store failed: {e}");
                failed += 1;
            }
        }
    }

    eprintln!(
        "Ingested {stored}/{total} conversations ({failed} failed). \
         {deduped_total} segments deduped, {} saved.",
        format_bytes(bytes_saved_total)
    );
    Ok(())
}

async fn cmd_retrieve(
    data_dir: &PathBuf,
    config: StowkenConfig,
    id: String,
    segments: bool,
    json: bool,
) -> Result<()> {
    let vault = open_vault(data_dir, config).await?;
    let conv = vault.retrieve(&id).await.context("retrieving conversation")?;

    if segments {
        let segs: Vec<serde_json::Value> = conv.segments.iter().map(|s| {
            serde_json::json!({
                "position": s.position,
                "segment_type": s.segment_type.to_string(),
                "hash": s.hash.0,
                "token_count": s.token_count,
                "tokens": s.tokens,
            })
        }).collect();
        println!("{}", serde_json::to_string_pretty(&segs)?);
    } else if json {
        let out = serde_json::json!({
            "id": conv.manifest.id,
            "model": conv.manifest.model,
            "tokenizer": conv.manifest.tokenizer,
            "application": conv.manifest.application,
            "total_tokens": conv.manifest.total_tokens,
            "created_at": conv.manifest.created_at,
            "segments": conv.segments.len(),
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        println!("ID         : {}", conv.manifest.id);
        println!("Model      : {}", conv.manifest.model);
        println!("Tokenizer  : {}", conv.manifest.tokenizer);
        if let Some(app) = &conv.manifest.application {
            println!("Application: {app}");
        }
        println!("Tokens     : {}", conv.manifest.total_tokens);
        println!("Segments   : {}", conv.segments.len());
        println!("Created    : {}", conv.manifest.created_at.format("%Y-%m-%d %H:%M:%S UTC"));
        println!();
        println!("{:<4}  {:<16}  {:<8}  Hash", "Pos", "Type", "Tokens");
        println!("{}", "-".repeat(80));
        for s in &conv.segments {
            println!("{:<4}  {:<16}  {:<8}  {}", s.position, s.segment_type, s.token_count, s.hash);
        }
    }
    Ok(())
}

async fn cmd_retrieve_batch(
    data_dir: &PathBuf,
    config: StowkenConfig,
    mut ids: Vec<String>,
    from_stdin: bool,
    json: bool,
) -> Result<()> {
    if from_stdin {
        use std::io::BufRead;
        for line in std::io::BufReader::new(std::io::stdin()).lines() {
            let line = line?;
            let trimmed = line.trim().to_string();
            if !trimmed.is_empty() {
                ids.push(trimmed);
            }
        }
    }
    if ids.is_empty() {
        anyhow::bail!("no conversation IDs provided");
    }
    let vault = open_vault(data_dir, config).await?;
    let id_refs: Vec<&str> = ids.iter().map(String::as_str).collect();
    let convs = vault.retrieve_batch(&id_refs).await.context("retrieve_batch")?;

    if json {
        println!("{}", serde_json::to_string_pretty(&convs)?);
    } else {
        for conv in &convs {
            println!("━━━ {} ({}) ━━━", conv.conversation_id, conv.model);
            for turn in &conv.turns {
                println!("[{}]", turn.role.to_uppercase());
                println!("{}", turn.text);
                println!();
            }
        }
        eprintln!("Retrieved {} conversation(s).", convs.len());
    }
    Ok(())
}

async fn cmd_delete(
    data_dir: &PathBuf,
    config: StowkenConfig,
    id: String,
    yes: bool,
) -> Result<()> {
    if !yes {
        print!("Delete conversation {id}? [y/N] ");
        io::stdout().flush()?;
        let mut line = String::new();
        io::stdin().lock().read_line(&mut line)?;
        if !line.trim().eq_ignore_ascii_case("y") {
            println!("Aborted.");
            return Ok(());
        }
    }
    let vault = open_vault(data_dir, config).await?;
    vault.delete(&id).await.context("deleting conversation")?;
    println!("Deleted {id}");
    Ok(())
}

async fn cmd_stats(data_dir: &PathBuf, config: StowkenConfig, json: bool) -> Result<()> {
    let vault = open_vault(data_dir, config).await?;
    let s = vault.stats().await.context("fetching stats")?;

    if json {
        println!("{}", serde_json::to_string_pretty(&s)?);
    } else {
        #[derive(Tabled)]
        struct Row { metric: &'static str, value: String }

        let rows = vec![
            Row { metric: "Conversations", value: s.total_conversations.to_string() },
            Row { metric: "Total tokens",  value: s.total_tokens.to_string() },
            Row { metric: "Unique segs",   value: s.unique_segments.to_string() },
            Row { metric: "Total segs",    value: s.total_segments.to_string() },
            Row { metric: "Dedup ratio",   value: format!("{:.1}%", s.dedup_ratio * 100.0) },
            Row { metric: "Compression",   value: format!("{:.3}x", s.compression_ratio) },
            Row { metric: "Storage bytes", value: format_bytes(s.storage_bytes) },
            Row { metric: "Savings",       value: format!("{:.1}%", s.savings_percentage) },
        ];
        println!("{}", Table::new(rows));
    }
    Ok(())
}

async fn cmd_segment_stats(data_dir: &PathBuf, config: StowkenConfig, json: bool) -> Result<()> {
    let vault = open_vault(data_dir, config).await?;
    let stats = vault.segment_stats().await.context("fetching segment stats")?;

    if json {
        let out: Vec<serde_json::Value> = stats.iter().map(|s| serde_json::json!({
            "segment_type":      s.segment_type.to_string(),
            "unique_count":      s.unique_count,
            "total_references":  s.total_references,
            "dedup_ratio":       s.dedup_ratio,
            "avg_token_count":   s.avg_token_count,
            "total_tokens":      s.total_tokens,
            "compressed_bytes":  s.compressed_bytes,
        })).collect();
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        #[derive(Tabled)]
        struct Row {
            #[tabled(rename = "Type")]
            segment_type: String,
            #[tabled(rename = "Unique")]
            unique_count: u64,
            #[tabled(rename = "Refs")]
            total_references: u64,
            #[tabled(rename = "Dedup%")]
            dedup_pct: String,
            #[tabled(rename = "Avg tokens")]
            avg_tokens: String,
            #[tabled(rename = "Compressed")]
            compressed: String,
        }

        let rows: Vec<Row> = stats.iter().map(|s| Row {
            segment_type: s.segment_type.to_string(),
            unique_count: s.unique_count,
            total_references: s.total_references,
            dedup_pct: format!("{:.1}%", s.dedup_ratio * 100.0),
            avg_tokens: format!("{:.0}", s.avg_token_count),
            compressed: format_bytes(s.compressed_bytes),
        }).collect();
        println!("{}", Table::new(rows));
    }
    Ok(())
}

async fn cmd_gc(data_dir: &PathBuf, config: StowkenConfig, dry_run: bool) -> Result<()> {
    let vault = open_vault(data_dir, config).await?;
    if dry_run {
        let candidates = vault.gc_candidates().await.context("listing GC candidates")?;
        if candidates.is_empty() {
            println!("Dry run: 0 segment(s) eligible for GC.");
        } else {
            println!("Dry run: {} segment(s) eligible for GC:", candidates.len());
            for hash in &candidates {
                println!("  {}", hash.0);
            }
            println!("Run without --dry-run to delete them.");
        }
    } else {
        let deleted = vault.gc().await.context("running GC")?;
        println!("GC complete: deleted {deleted} orphaned segment(s)");
    }
    Ok(())
}

async fn cmd_audit_system_prompts(
    data_dir: &PathBuf,
    config: StowkenConfig,
    min_refs: u64,
    limit: u64,
    json: bool,
) -> Result<()> {
    let vault = open_vault(data_dir, config).await?;
    let prompts = vault
        .list_system_prompts()
        .await
        .context("listing system prompts")?;
    let filtered: Vec<_> = prompts
        .into_iter()
        .filter(|p| p.ref_count >= min_refs)
        .take(limit as usize)
        .collect();

    if json {
        let out: Vec<serde_json::Value> = filtered
            .iter()
            .map(|p| serde_json::json!({
                "hash":        p.hash.0,
                "token_count": p.token_count,
                "ref_count":   p.ref_count,
            }))
            .collect();
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else if filtered.is_empty() {
        println!("No system prompts with ref_count >= {min_refs}.");
    } else {
        #[derive(Tabled)]
        struct Row {
            #[tabled(rename = "Refs")]
            ref_count: u64,
            #[tabled(rename = "Tokens")]
            token_count: u32,
            #[tabled(rename = "Hash")]
            hash: String,
        }
        let rows: Vec<Row> = filtered
            .iter()
            .map(|p| Row {
                ref_count: p.ref_count,
                token_count: p.token_count,
                hash: format!("{}…", &p.hash.0[..16]),
            })
            .collect();
        println!("{}", Table::new(rows));
        println!(
            "{} unique system prompt(s). High duplicate counts on different hashes \
             may indicate prompt drift (interpolated values, formatting differences).",
            filtered.len()
        );
    }
    Ok(())
}

async fn cmd_cost(
    data_dir: &PathBuf,
    config: StowkenConfig,
    by: &str,
    model: Option<String>,
    application: Option<String>,
    json: bool,
) -> Result<()> {
    let valid_dims = ["model", "application", "segment_type", "day"];
    let group_by: Vec<String> = by
        .split(',')
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .collect();
    for dim in &group_by {
        if !valid_dims.contains(&dim.as_str()) {
            anyhow::bail!(
                "unknown --by dimension '{dim}'; valid: {}",
                valid_dims.join(", ")
            );
        }
    }

    let vault = open_vault(data_dir, config).await?;
    let query = AnalyticsQuery {
        model,
        application,
        group_by: Some(group_by.clone()),
        ..Default::default()
    };
    let rows = vault.query(query).await.context("running cost query")?;

    if json {
        println!("{}", serde_json::to_string_pretty(&rows)?);
    } else if rows.is_empty() {
        println!("No data matching the query.");
    } else {
        // Render dynamically — group_by columns first, then total_tokens.
        let mut headers: Vec<String> = group_by.clone();
        headers.push("total_tokens".to_owned());
        println!("{}", headers.join("\t"));
        println!("{}", "-".repeat(headers.iter().map(|h| h.len()).sum::<usize>() + headers.len() * 2));
        for row in &rows {
            let cells: Vec<String> = headers
                .iter()
                .map(|h| match row.get(h.as_str()) {
                    Some(serde_json::Value::Null) | None => "—".to_owned(),
                    Some(serde_json::Value::String(s)) => s.clone(),
                    Some(other) => other.to_string(),
                })
                .collect();
            println!("{}", cells.join("\t"));
        }
    }
    Ok(())
}

async fn cmd_find_duplicates(
    data_dir: &PathBuf,
    config: StowkenConfig,
    threshold: u64,
    limit: u64,
    json: bool,
) -> Result<()> {
    let vault = open_vault(data_dir, config).await?;
    let dups = vault
        .find_duplicates(threshold, limit)
        .await
        .context("finding duplicates")?;

    if json {
        let out: Vec<serde_json::Value> = dups
            .iter()
            .map(|d| serde_json::json!({
                "hash":         d.hash.0,
                "segment_type": d.segment_type.to_string(),
                "token_count":  d.token_count,
                "ref_count":    d.ref_count,
                "wasted_bytes": d.wasted_bytes,
            }))
            .collect();
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else if dups.is_empty() {
        println!("No segments with ref_count >= {threshold}.");
    } else {
        #[derive(Tabled)]
        struct Row {
            #[tabled(rename = "Refs")]
            refs: u64,
            #[tabled(rename = "Type")]
            seg_type: String,
            #[tabled(rename = "Tokens")]
            tokens: u32,
            #[tabled(rename = "Saved")]
            saved: String,
            #[tabled(rename = "Hash")]
            hash: String,
        }
        let total_saved: u64 = dups.iter().map(|d| d.wasted_bytes).sum();
        let rows: Vec<Row> = dups
            .iter()
            .map(|d| Row {
                refs: d.ref_count,
                seg_type: d.segment_type.to_string(),
                tokens: d.token_count,
                saved: format_bytes(d.wasted_bytes),
                hash: format!("{}…", &d.hash.0[..16]),
            })
            .collect();
        println!("{}", Table::new(rows));
        println!(
            "{} segment(s) shown — dedup is saving {} on these alone.",
            dups.len(),
            format_bytes(total_saved)
        );
    }
    Ok(())
}

async fn cmd_reindex(data_dir: &PathBuf, config: StowkenConfig, json: bool) -> Result<()> {
    let vault = open_vault(data_dir, config).await?;
    let started = std::time::Instant::now();
    let stats = vault.reindex().await.context("running reindex")?;
    let elapsed = started.elapsed();

    if json {
        let out = serde_json::json!({
            "conversations_indexed":   stats.conversations_indexed,
            "unique_segments_indexed": stats.unique_segments_indexed,
            "segments_missing":        stats.segments_missing,
            "elapsed_ms":              elapsed.as_millis(),
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        println!(
            "Reindex complete in {}ms: {} conversation(s), {} unique segment(s)",
            elapsed.as_millis(),
            stats.conversations_indexed,
            stats.unique_segments_indexed
        );
        if stats.segments_missing > 0 {
            eprintln!(
                "warning: {} segment reference(s) had no backing data — likely corruption. \
                 Reindex completed; consider running `gc` to clean up dangling refs.",
                stats.segments_missing
            );
        }
    }
    Ok(())
}

async fn cmd_find_near_duplicates(
    data_dir: &PathBuf,
    config: StowkenConfig,
    threshold: f64,
    min_refs: u64,
    limit: u64,
    json: bool,
) -> Result<()> {
    let vault = open_vault(data_dir, config).await?;
    let clusters = vault
        .find_near_duplicates(threshold, min_refs, limit)
        .await
        .context("finding near-duplicates")?;

    if json {
        println!("{}", serde_json::to_string_pretty(&clusters)?);
        return Ok(());
    }

    if clusters.is_empty() {
        println!(
            "No near-duplicate clusters found (threshold {:.2}, min_refs {}).",
            threshold, min_refs
        );
        println!("Tip: ensure near-dedup was enabled during ingest with `--near-dedup <threshold>` \
                  or run `stowken reindex` to populate the LSH index.");
        return Ok(());
    }

    for (i, cluster) in clusters.iter().enumerate() {
        if i > 0 {
            println!();
        }
        println!(
            "Cluster {}: canonical {}… ({} tokens, {} refs)",
            i + 1,
            &cluster.canonical.0[..16],
            cluster.canonical_token_count,
            cluster.canonical_ref_count,
        );
        #[derive(Tabled)]
        struct Row {
            #[tabled(rename = "Sim")]
            similarity: String,
            #[tabled(rename = "Tokens")]
            tokens: u32,
            #[tabled(rename = "Refs")]
            refs: u64,
            #[tabled(rename = "Hash")]
            hash: String,
        }
        let rows: Vec<Row> = cluster
            .variants
            .iter()
            .map(|v| Row {
                similarity: format!("{:.0}%", v.similarity * 100.0),
                tokens: v.token_count,
                refs: v.ref_count,
                hash: format!("{}…", &v.hash.0[..16]),
            })
            .collect();
        println!("{}", Table::new(rows));
    }
    println!(
        "\n{} cluster(s) shown. Variants with high similarity but distinct hashes \
         often indicate prompt drift (interpolated values, formatting changes).",
        clusters.len()
    );
    Ok(())
}

// ── Substrings ────────────────────────────────────────────────────────────────

async fn cmd_substrings(
    data_dir: &PathBuf,
    config: StowkenConfig,
    action: SubstringsAction,
    json: bool,
) -> Result<()> {
    let vault = open_vault(data_dir, config).await?;
    match action {
        SubstringsAction::Train { samples, min_occurrences } => {
            let promoted = vault
                .train_substrings(samples, min_occurrences)
                .await
                .context("training substrings")?;
            if json {
                println!("{}", serde_json::to_string_pretty(&promoted)?);
            } else {
                println!(
                    "Promoted {} substring(s) (sampled {} segments, min occurrences {}).",
                    promoted.len(),
                    samples,
                    min_occurrences
                );
                if promoted.is_empty() {
                    println!("Tip: try lowering --min-occurrences or sampling more segments.");
                }
            }
        }
        SubstringsAction::Compact => {
            let report = vault
                .compact_substrings()
                .await
                .context("compacting segments to use substring registry")?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!(
                    "Compaction: examined {} segment(s), rewrote {}, skipped {} \
                     ({} delta or already-substring), saved {}.",
                    report.segments_examined,
                    report.segments_rewritten,
                    report.segments_skipped,
                    report.segments_skipped,
                    format_bytes(report.bytes_saved),
                );
                if report.segments_rewritten == 0 && report.segments_examined > 0 {
                    println!(
                        "No segments benefited from substring encoding. \
                         Tip: train substrings first (`stowken substrings train`) \
                         or check that your corpus has structural repetition."
                    );
                }
            }
        }
        SubstringsAction::Gc => {
            let report = vault
                .gc_substrings()
                .await
                .context("gc'ing unreferenced substrings")?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!(
                    "Substring GC: registry had {} substring(s), {} dropped, {} remaining.",
                    report.registry_size_before,
                    report.substrings_dropped,
                    report.registry_size_after,
                );
            }
        }
        SubstringsAction::List => {
            let listed = vault.list_substrings();
            if json {
                println!("{}", serde_json::to_string_pretty(&listed)?);
            } else if listed.is_empty() {
                println!("No substrings registered for this vault.");
            } else {
                #[derive(Tabled)]
                struct Row {
                    #[tabled(rename = "ID")]
                    id: u32,
                    #[tabled(rename = "Length")]
                    length: u32,
                    #[tabled(rename = "Source occ.")]
                    occurrences: u32,
                    #[tabled(rename = "Created")]
                    created: String,
                }
                let rows: Vec<Row> = listed
                    .iter()
                    .map(|s| Row {
                        id: s.id,
                        length: s.length,
                        occurrences: s.source_occurrences,
                        created: s.created_at.format("%Y-%m-%d %H:%M").to_string(),
                    })
                    .collect();
                println!("{}", Table::new(rows));
                println!("({} substring(s) total)", listed.len());
            }
        }
    }
    Ok(())
}

// ── Dictionaries ──────────────────────────────────────────────────────────────

async fn cmd_dict(
    data_dir: &PathBuf,
    config: StowkenConfig,
    action: DictAction,
    json: bool,
) -> Result<()> {
    let vault = open_vault(data_dir, config).await?;
    match action {
        DictAction::Train { samples, activate } => {
            let info = if activate {
                vault.train_and_activate_dictionary(samples).await
            } else {
                vault.train_dictionary(samples).await
            }
            .context("training dictionary")?;

            if json {
                println!("{}", serde_json::to_string_pretty(&info)?);
            } else {
                println!(
                    "Trained dictionary id={} ({} bytes, sampled {} segments){}",
                    info.id,
                    info.size_bytes,
                    info.sample_count,
                    if info.is_active { " — active" } else { "" }
                );
                if !info.is_active {
                    println!(
                        "Run `stowken dict activate {}` to use it for new compressions.",
                        info.id
                    );
                }
            }
        }
        DictAction::List => {
            let dicts = vault.list_dictionaries();
            if json {
                println!("{}", serde_json::to_string_pretty(&dicts)?);
            } else if dicts.is_empty() {
                println!("No dictionaries trained for this vault.");
            } else {
                #[derive(Tabled)]
                struct Row {
                    #[tabled(rename = "Active")]
                    active: &'static str,
                    #[tabled(rename = "ID")]
                    id: u32,
                    #[tabled(rename = "Size")]
                    size: String,
                    #[tabled(rename = "Samples")]
                    samples: u32,
                    #[tabled(rename = "Created")]
                    created: String,
                }
                let rows: Vec<Row> = dicts
                    .iter()
                    .map(|d| Row {
                        active: if d.is_active { "*" } else { " " },
                        id: d.id,
                        size: format_bytes(d.size_bytes),
                        samples: d.sample_count,
                        created: d.created_at.format("%Y-%m-%d %H:%M").to_string(),
                    })
                    .collect();
                println!("{}", Table::new(rows));
                println!("(* = currently writing new frames)");
            }
        }
        DictAction::Activate { id } => {
            vault
                .activate_dictionary(id)
                .await
                .with_context(|| format!("activating dict {id}"))?;
            if json {
                println!("{}", serde_json::json!({ "activated": id }));
            } else {
                println!("Dictionary {id} is now active for new compressions.");
                println!("Old frames keep their original dict — rotation is non-destructive.");
            }
        }
    }
    Ok(())
}

// ── Benchmark ─────────────────────────────────────────────────────────────────

async fn cmd_bench(
    config: StowkenConfig,
    split_segments: Option<usize>,
    input: PathBuf,
    max: Option<u64>,
    retrieve: Option<u64>,
    dict_warmup: Option<usize>,
    with_substrings: bool,
    json: bool,
) -> Result<()> {
    let conversations = collect_conversations(&input, max)?;
    if conversations.is_empty() {
        anyhow::bail!("no conversations found under {}", input.display());
    }

    let near_dedup_active = config.near_dedup_threshold.is_some();
    let mut vault = Stowken::new(MemoryBackend::new(), config)
        .await
        .context("opening in-memory bench vault")?;

    if let Some(max_tokens) = split_segments {
        vault.set_segmenter_config(stowken::segmenter::SegmenterConfig {
            max_segment_tokens: Some(max_tokens),
            ..Default::default()
        });
    }

    // Dict-warmup phase: ingest the first N conversations, train + activate a
    // zstd dictionary, then ingest the remainder with it active.
    let (warmup_count, warmup_elapsed) = if let Some(n) = dict_warmup {
        let warmup = conversations[..n.min(conversations.len())].to_vec();
        let t0 = std::time::Instant::now();
        for conv in &warmup {
            let _ = vault.store(conv.clone()).await;
        }
        let warmup_elapsed = t0.elapsed();
        vault
            .train_and_activate_dictionary(warmup.len().min(500))
            .await
            .context("training dictionary after warmup")?;
        (warmup.len(), Some(warmup_elapsed))
    } else {
        (0, None)
    };

    let started = std::time::Instant::now();
    let mut ingested = 0u64;
    let mut failures = 0u64;
    let mut ingested_ids: Vec<String> = Vec::new();
    let tail = if dict_warmup.is_some() {
        &conversations[warmup_count..]
    } else {
        &conversations[..]
    };
    for conv in tail {
        match vault.store(conv.clone()).await {
            Ok(result) => {
                ingested += 1;
                ingested_ids.push(result.id);
            }
            Err(_) => failures += 1,
        }
    }
    let elapsed = started.elapsed();

    // Substring phase: train registry + compact all segments.
    let (substring_elapsed, substring_compacted) = if with_substrings {
        let t0 = std::time::Instant::now();
        vault
            .train_substrings(500, 5)
            .await
            .context("training substring registry")?;
        let compact = vault
            .compact_substrings()
            .await
            .context("compacting substrings")?;
        (Some(t0.elapsed()), Some(compact))
    } else {
        (None, None)
    };

    let stats = vault.stats().await.context("fetching stats")?;
    let dedup_only_bytes = vault
        .dedup_only_bytes()
        .await
        .context("fetching dedup-only bytes")?;
    let seg_stats = vault.segment_stats().await.context("fetching segment stats")?;

    // Retrieval pass: sample some ingested IDs and time each retrieve.
    let retrieval = run_retrieval_pass(&vault, &ingested_ids, retrieve).await?;

    if json {
        let mut out = serde_json::json!({
            "source":            input.display().to_string(),
            "ingested":          ingested,
            "failures":          failures,
            "elapsed_ms":        elapsed.as_millis(),
            "naive_bytes":       stats.naive_bytes,
            "dedup_only_bytes":  dedup_only_bytes,
            "storage_bytes":     stats.storage_bytes,
            "total_tokens":      stats.total_tokens,
            "total_segments":    stats.total_segments,
            "unique_segments":   stats.unique_segments,
            "dedup_ratio":       stats.dedup_ratio,
            "compression_ratio": stats.compression_ratio,
            "savings_pct":       stats.savings_percentage,
            "per_segment_type":  seg_stats.iter().map(|s| serde_json::json!({
                "type":             s.segment_type.to_string(),
                "unique_count":     s.unique_count,
                "total_references": s.total_references,
                "dedup_ratio":      s.dedup_ratio,
                "total_tokens":     s.total_tokens,
                "compressed_bytes": s.compressed_bytes,
            })).collect::<Vec<_>>(),
        });
        if let Some(r) = &retrieval {
            out.as_object_mut().unwrap().insert(
                "retrieval".into(),
                serde_json::json!({
                    "sampled":              r.count,
                    "elapsed_ms":           r.elapsed.as_millis(),
                    "tokens_returned":      r.tokens,
                    "tokens_per_sec":       r.tokens_per_sec(),
                    "convs_per_sec":        r.convs_per_sec(),
                    "p50_us":               r.p50_us(),
                    "p95_us":               r.p95_us(),
                    "p99_us":               r.p99_us(),
                }),
            );
        }
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }

    println!("Stowken Benchmark");
    println!("{}", "=".repeat(60));
    println!("Source        : {}", input.display());
    println!();

    println!("Ingest");
    println!("{}", "-".repeat(60));
    let per_conv_us = if ingested > 0 {
        elapsed.as_micros() / ingested as u128
    } else {
        0
    };
    let toks_per_sec = if elapsed.as_secs_f64() > 0.0 {
        (stats.total_tokens as f64 / elapsed.as_secs_f64()) as u64
    } else {
        0
    };
    if let Some(we) = warmup_elapsed {
        println!("  Warmup        : {} convs (dict trained in {}ms)", warmup_count, we.as_millis());
    }
    println!(
        "  Conversations : {}  ({} failed)",
        ingested, failures
    );
    println!("  Tokens        : {}", stats.total_tokens);
    println!(
        "  Time          : {}ms  ({}µs/conv, {} tok/s)",
        elapsed.as_millis(),
        per_conv_us,
        toks_per_sec
    );
    if let Some(compact) = &substring_compacted {
        println!(
            "  Substrings    : {}ms — {} segs rewritten, {} bytes saved",
            substring_elapsed.unwrap_or_default().as_millis(),
            compact.segments_rewritten,
            compact.bytes_saved,
        );
    }
    println!();

    // Two reference points:
    //   * incoming_bytes = what the user actually sent in (constant per
    //     corpus regardless of near-dedup, dict, etc.) — the right
    //     baseline for "how much did Stowken save vs sending raw u32 to
    //     disk".
    //   * naive_bytes (from index) = SUM(raw_size × ref_count) which
    //     includes near-dedup's canonical cross-refs. Useful for showing
    //     dedup effectiveness across the reference graph.
    let incoming_bytes = stats.total_tokens * 4;

    println!("Storage layers");
    println!("{}", "-".repeat(60));
    let dedup_save_pct = pct_saved(incoming_bytes, dedup_only_bytes);
    let zstd_save_pct = pct_saved(dedup_only_bytes, stats.storage_bytes);
    let total_save_pct = pct_saved(incoming_bytes, stats.storage_bytes);
    println!(
        "  Incoming (u32): {:>10}   (ingested tokens \u{00d7} 4)",
        format_bytes(incoming_bytes)
    );
    if near_dedup_active {
        println!(
            "  After dedup   : {:>10}   {:>5.1}% saved by exact dedup (near-dedup delta savings in compression layer)",
            format_bytes(dedup_only_bytes),
            dedup_save_pct
        );
    } else {
        println!(
            "  After dedup   : {:>10}   {:>5.1}% saved by dedup alone (refs \u{00d7} raw \u{2192} unique \u{00d7} raw)",
            format_bytes(dedup_only_bytes),
            dedup_save_pct
        );
    }
    println!(
        "  After zstd    : {:>10}   {:>5.1}% saved on top of dedup",
        format_bytes(stats.storage_bytes),
        zstd_save_pct
    );
    println!();
    println!(
        "  Combined      : {:.1}% smaller than incoming  ({} \u{2192} {})",
        total_save_pct,
        format_bytes(incoming_bytes),
        format_bytes(stats.storage_bytes),
    );
    println!();

    println!("Per-segment-type breakdown");
    println!("{}", "-".repeat(60));
    if near_dedup_active {
        println!("  (Refs/Dedup% include canonical GC-pin refs from near-dedup variants)");
    }
    #[derive(Tabled)]
    struct Row {
        #[tabled(rename = "Type")]
        seg_type: String,
        #[tabled(rename = "Unique")]
        unique: u64,
        #[tabled(rename = "Refs")]
        refs: u64,
        #[tabled(rename = "Dedup%")]
        dedup_pct: String,
        #[tabled(rename = "Tokens")]
        tokens: u64,
        #[tabled(rename = "Compressed")]
        compressed: String,
    }
    let mut sorted = seg_stats.clone();
    sorted.sort_by_key(|s| std::cmp::Reverse(s.total_references));
    let rows: Vec<Row> = sorted
        .iter()
        .map(|s| Row {
            seg_type: s.segment_type.to_string(),
            unique: s.unique_count,
            refs: s.total_references,
            dedup_pct: format!("{:.1}%", s.dedup_ratio * 100.0),
            tokens: s.total_tokens,
            compressed: format_bytes(s.compressed_bytes),
        })
        .collect();
    println!("{}", Table::new(rows));

    if let Some(r) = retrieval {
        println!();
        println!("Retrieval (decompression-only, MemoryBackend)");
        println!("{}", "-".repeat(60));
        println!(
            "  Sampled       : {} conversations",
            r.count
        );
        println!(
            "  Time          : {}ms  ({} tok/s, {} conv/s)",
            r.elapsed.as_millis(),
            r.tokens_per_sec(),
            r.convs_per_sec(),
        );
        println!(
            "  Latency p50/p95/p99: {}µs / {}µs / {}µs per conversation",
            r.p50_us(),
            r.p95_us(),
            r.p99_us(),
        );
    }
    Ok(())
}

/// Per-conversation retrieval timing collected during a bench pass.
struct RetrievalReport {
    count: u64,
    tokens: u64,
    elapsed: std::time::Duration,
    /// Sorted ascending in microseconds — used for percentile lookups.
    sorted_micros: Vec<u128>,
}

impl RetrievalReport {
    fn tokens_per_sec(&self) -> u64 {
        let secs = self.elapsed.as_secs_f64();
        if secs <= 0.0 {
            0
        } else {
            (self.tokens as f64 / secs) as u64
        }
    }
    fn convs_per_sec(&self) -> u64 {
        let secs = self.elapsed.as_secs_f64();
        if secs <= 0.0 {
            0
        } else {
            (self.count as f64 / secs) as u64
        }
    }
    fn percentile_us(&self, p: f64) -> u128 {
        if self.sorted_micros.is_empty() {
            return 0;
        }
        // Nearest-rank percentile.
        let rank = ((p / 100.0) * self.sorted_micros.len() as f64).ceil() as usize;
        let idx = rank.saturating_sub(1).min(self.sorted_micros.len() - 1);
        self.sorted_micros[idx]
    }
    fn p50_us(&self) -> u128 { self.percentile_us(50.0) }
    fn p95_us(&self) -> u128 { self.percentile_us(95.0) }
    fn p99_us(&self) -> u128 { self.percentile_us(99.0) }
}

/// Time `min(retrieve_n, ingested.len())` retrievals against the bench
/// vault. Defaults to `min(ingested, 1000)` when `retrieve_n` is None.
/// Returns `Ok(None)` if there's nothing to retrieve.
async fn run_retrieval_pass<B>(
    vault: &Stowken<B>,
    ingested: &[String],
    retrieve_n: Option<u64>,
) -> Result<Option<RetrievalReport>>
where
    B: stowken::storage::StorageBackend + 'static,
{
    if ingested.is_empty() {
        return Ok(None);
    }
    let n = retrieve_n
        .unwrap_or_else(|| (ingested.len() as u64).min(1000))
        .min(ingested.len() as u64) as usize;

    // Pick `n` IDs spread evenly across the ingest order. Avoids the
    // pathological case where the first/last `n` are all stored under
    // the same dictionary or the same canonical, which would skew the
    // mix of frame types we exercise.
    let stride = ingested.len() / n.max(1);
    let mut sample: Vec<&String> = (0..n)
        .map(|i| &ingested[(i * stride.max(1)).min(ingested.len() - 1)])
        .collect();
    sample.dedup();

    let mut tokens = 0u64;
    let mut micros: Vec<u128> = Vec::with_capacity(sample.len());
    let started = std::time::Instant::now();
    for id in &sample {
        let t = std::time::Instant::now();
        let conv = vault
            .retrieve(id)
            .await
            .with_context(|| format!("bench retrieve {id}"))?;
        let elapsed_micros = t.elapsed().as_micros();
        for seg in &conv.segments {
            tokens += seg.tokens.len() as u64;
        }
        micros.push(elapsed_micros);
    }
    let total = started.elapsed();
    micros.sort_unstable();

    Ok(Some(RetrievalReport {
        count: micros.len() as u64,
        tokens,
        elapsed: total,
        sorted_micros: micros,
    }))
}

fn pct_saved(before: u64, after: u64) -> f64 {
    if before == 0 {
        0.0
    } else {
        (1.0 - (after as f64 / before as f64)) * 100.0
    }
}

fn collect_conversations(input: &PathBuf, max: Option<u64>) -> Result<Vec<Conversation>> {
    let mut out = Vec::new();
    let limit = max.unwrap_or(u64::MAX);

    let metadata = std::fs::metadata(input)
        .with_context(|| format!("stat {}", input.display()))?;

    let mut paths: Vec<PathBuf> = if metadata.is_file() {
        vec![input.clone()]
    } else if metadata.is_dir() {
        let mut v: Vec<PathBuf> = std::fs::read_dir(input)
            .with_context(|| format!("read_dir {}", input.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.is_file()
                    && p.extension()
                        .and_then(|x| x.to_str())
                        .map(|x| x == "json" || x == "jsonl")
                        .unwrap_or(false)
            })
            .collect();
        // Stable, deterministic input order so two runs over the same dir
        // produce the same dedup statistics.
        v.sort();
        v
    } else {
        anyhow::bail!("{} is neither a file nor a directory", input.display());
    };
    paths.retain(|p| p.is_file());

    for path in paths {
        if (out.len() as u64) >= limit {
            break;
        }
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let is_jsonl = path
            .extension()
            .and_then(|x| x.to_str())
            .map(|x| x == "jsonl")
            .unwrap_or(false);

        if is_jsonl {
            for (lineno, line) in raw.lines().enumerate() {
                if (out.len() as u64) >= limit {
                    break;
                }
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let conv = parse_conversation_json(trimmed).with_context(|| {
                    format!("{} line {}", path.display(), lineno + 1)
                })?;
                out.push(conv);
            }
        } else {
            let conv = parse_conversation_json(&raw)
                .with_context(|| format!("parsing {}", path.display()))?;
            out.push(conv);
        }
    }

    Ok(out)
}

// ── Formatting helpers ────────────────────────────────────────────────────────

fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut val = bytes as f64;
    let mut unit = 0;
    while val >= 1024.0 && unit + 1 < UNITS.len() {
        val /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{} B", bytes)
    } else {
        format!("{val:.1} {}", UNITS[unit])
    }
}

// ── Semantic search commands ─────────────────────────────────────────────────

#[cfg(feature = "semantic-search")]
async fn open_vault_with_adapters(
    data_dir: &PathBuf,
    config: StowkenConfig,
    openai_key: Option<String>,
    model: String,
    summary_strategy: Option<(String, String)>, // (strategy_name, summary_model)
) -> Result<Stowken<FilesystemBackend>> {
    use openai_adapter::{
        openai_dimension_for, require_openai_key, OpenAiEmbeddingAdapter, OpenAiSummarizerAdapter,
    };

    let key = require_openai_key(openai_key)?;
    let dim = openai_dimension_for(&model).with_context(|| {
        format!(
            "unknown OpenAI embedding model '{model}'. \
             Pass --model text-embedding-3-small or text-embedding-3-large."
        )
    })?;
    let vault = open_vault(data_dir, config).await?;
    vault.set_embedding_adapter(OpenAiEmbeddingAdapter::new(&key, &model, dim));
    if let Some((strategy, summary_model)) = summary_strategy {
        match strategy.as_str() {
            "concat" => vault.use_concat_summary(24_000),
            "llm" => vault.set_summarizer(OpenAiSummarizerAdapter::new(&key, &summary_model)),
            other => anyhow::bail!(
                "unknown --summary-strategy '{other}'; expected 'concat' or 'llm'"
            ),
        }
    }
    Ok(vault)
}

#[cfg(feature = "semantic-search")]
async fn cmd_embed_all(
    data_dir: &PathBuf,
    config: StowkenConfig,
    openai_key: Option<String>,
    model: String,
    summary_strategy: String,
    summary_model: String,
    json: bool,
) -> Result<()> {
    let vault = open_vault_with_adapters(
        data_dir,
        config,
        openai_key,
        model,
        Some((summary_strategy, summary_model)),
    )
    .await?;
    eprintln!("Counting unembedded items…");
    let pending = vault.embed_all_pending_counts().await.context("counting pending")?;
    eprintln!(
        "  {} unique segments to embed, {} conversations to embed",
        pending.0, pending.1
    );
    if pending.0 == 0 && pending.1 == 0 {
        eprintln!("Nothing to do — all items already embedded.");
        return Ok(());
    }
    if pending.0 > 0 {
        eprintln!("Embedding segments…");
    }
    let stats = vault.embed_all().await.context("embedding all segments")?;
    if json {
        println!("{}", serde_json::to_string_pretty(&stats)?);
    } else {
        println!(
            "Done. Embedded {} segments and {} conversations (model={}, strategy={})",
            stats.segments_embedded,
            stats.conversations_embedded,
            stats.embedding_model,
            stats.summary_strategy
        );
    }
    Ok(())
}

#[cfg(feature = "semantic-search")]
#[allow(clippy::too_many_arguments)]
async fn cmd_semantic_search(
    data_dir: &PathBuf,
    config: StowkenConfig,
    query: String,
    openai_key: Option<String>,
    model: String,
    granularity: String,
    limit: usize,
    min_score: f32,
    application: Option<String>,
    filter_model: Option<String>,
    segment_type: Option<String>,
    json: bool,
) -> Result<()> {
    use stowken::types::{SearchGranularity, SegmentType, SemanticSearchQuery};
    let vault = open_vault_with_adapters(data_dir, config, openai_key, model, None).await?;

    let granularity: SearchGranularity = granularity
        .parse()
        .map_err(|e: String| anyhow::anyhow!(e))?;
    let segment_type = match segment_type {
        Some(s) => Some(s.parse::<SegmentType>().map_err(|e| anyhow::anyhow!(e))?),
        None => None,
    };

    let q = SemanticSearchQuery {
        text: query,
        granularity,
        model: filter_model,
        application,
        segment_type,
        date_from: None,
        date_to: None,
        limit,
        min_score,
    };
    let hits = vault
        .semantic_search(q)
        .await
        .context("running semantic search")?;
    print_hits(&hits, json)?;
    Ok(())
}

#[cfg(feature = "semantic-search")]
async fn cmd_typed_search(
    data_dir: &PathBuf,
    config: StowkenConfig,
    query: String,
    openai_key: Option<String>,
    model: String,
    limit: usize,
    kind: &str,
    json: bool,
) -> Result<()> {
    let vault = open_vault_with_adapters(data_dir, config, openai_key, model, None).await?;
    let hits = match kind {
        "prompts" => vault.search_prompts(&query, limit).await,
        "responses" => vault.search_responses(&query, limit).await,
        "user_questions" => vault.search_user_questions(&query, limit).await,
        "tool_uses" => vault.search_tool_uses(&query, limit).await,
        "conversations" => vault.search_conversations(&query, limit).await,
        other => anyhow::bail!("unknown typed search kind '{other}'"),
    }
    .with_context(|| format!("typed search ({kind})"))?;
    print_hits(&hits, json)?;
    Ok(())
}

#[cfg(feature = "semantic-search")]
fn print_hits(hits: &[stowken::types::SemanticSearchHit], json: bool) -> Result<()> {
    use stowken::types::MatchedVia;
    if json {
        println!("{}", serde_json::to_string_pretty(hits)?);
        return Ok(());
    }
    if hits.is_empty() {
        println!("No matching conversations.");
        return Ok(());
    }
    #[derive(Tabled)]
    struct Row {
        score: String,
        conversation_id: String,
        matched_via: String,
        application: String,
        model: String,
    }
    let rows: Vec<Row> = hits
        .iter()
        .map(|h| Row {
            score: format!("{:.3}", h.score),
            conversation_id: h.conversation_id.clone(),
            matched_via: match &h.matched_via {
                MatchedVia::Segment { segment_type, .. } => format!("segment:{segment_type}"),
                MatchedVia::Conversation => "conversation".to_owned(),
            },
            application: h.application.clone().unwrap_or_else(|| "-".to_owned()),
            model: h.model.clone(),
        })
        .collect();
    println!("{}", Table::new(rows));
    Ok(())
}

#[cfg(feature = "semantic-search")]
async fn cmd_cluster(
    data_dir: &PathBuf,
    config: StowkenConfig,
    openai_key: Option<String>,
    model: String,
    k: usize,
    representatives: usize,
    json: bool,
) -> Result<()> {
    let vault = open_vault_with_adapters(data_dir, config, openai_key, model, None).await?;
    let clusters = vault
        .cluster_conversations(k, representatives)
        .await
        .context("clustering")?;
    if json {
        println!("{}", serde_json::to_string_pretty(&clusters)?);
        return Ok(());
    }
    if clusters.is_empty() {
        println!("No conversation embeddings found. Run `stowken embed-all` first.");
        return Ok(());
    }
    for c in &clusters {
        println!(
            "─── Cluster {} ({} conversations) ───",
            c.cluster_id, c.size
        );
        for id in &c.representative_ids {
            println!("  rep: {id}");
        }
        println!();
    }
    Ok(())
}

#[cfg(feature = "semantic-search")]
async fn cmd_outliers(
    data_dir: &PathBuf,
    config: StowkenConfig,
    openai_key: Option<String>,
    model: String,
    k: usize,
    threshold: f32,
    limit: usize,
    json: bool,
) -> Result<()> {
    let vault = open_vault_with_adapters(data_dir, config, openai_key, model, None).await?;
    let outliers = vault
        .find_outliers(k, threshold, limit)
        .await
        .context("finding outliers")?;
    if json {
        println!("{}", serde_json::to_string_pretty(&outliers)?);
        return Ok(());
    }
    if outliers.is_empty() {
        println!("No outliers above threshold {threshold}.");
        return Ok(());
    }
    #[derive(Tabled)]
    struct Row {
        score: String,
        conversation_id: String,
        application: String,
        model: String,
    }
    let rows: Vec<Row> = outliers
        .iter()
        .map(|o| Row {
            score: format!("{:.3}", o.isolation_score),
            conversation_id: o.conversation_id.clone(),
            application: o.application.clone().unwrap_or_else(|| "-".to_owned()),
            model: o.model.clone(),
        })
        .collect();
    println!("{}", Table::new(rows));
    Ok(())
}
