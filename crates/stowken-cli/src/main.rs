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

    /// Output as JSON (where applicable).
    #[arg(long, short = 'j', global = true)]
    json: bool,

    #[command(subcommand)]
    command: Command,
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

    /// Retrieve a conversation by ID and print it as JSON.
    Retrieve {
        /// Conversation ID.
        id: String,

        /// Print raw segment token arrays instead of the manifest.
        #[arg(long)]
        segments: bool,
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
    /// three-stage savings breakdown plus per-segment-type detail.
    ///
    /// Does not touch the configured data directory — bench is read-only
    /// w.r.t. your real vault.
    Bench {
        /// Directory or single file to ingest.
        input: PathBuf,
        /// Cap on conversations ingested. Default: no cap.
        #[arg(long)]
        max: Option<u64>,
    },

    /// Print the resolved data directory and exit.
    DataDir,
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let data_dir = resolve_data_dir(cli.data_dir)?;
    let config = StowkenConfig {
        enable_compression: !cli.no_compression,
        near_dedup_threshold: cli.near_dedup,
    };

    match cli.command {
        Command::DataDir => {
            println!("{}", data_dir.display());
            return Ok(());
        }
        Command::Store { file, model, application, id } => {
            cmd_store(&data_dir, config, file, model, application, id, cli.json).await?;
        }
        Command::Retrieve { id, segments } => {
            cmd_retrieve(&data_dir, config, id, segments, cli.json).await?;
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
        Command::Bench { input, max } => {
            cmd_bench(config, input, max, cli.json).await?;
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
    input: PathBuf,
    max: Option<u64>,
    json: bool,
) -> Result<()> {
    let conversations = collect_conversations(&input, max)?;
    if conversations.is_empty() {
        anyhow::bail!("no conversations found under {}", input.display());
    }

    let vault = Stowken::new(MemoryBackend::new(), config)
        .await
        .context("opening in-memory bench vault")?;

    let started = std::time::Instant::now();
    let mut ingested = 0u64;
    let mut failures = 0u64;
    for conv in conversations {
        match vault.store(conv).await {
            Ok(_) => ingested += 1,
            Err(_) => failures += 1,
        }
    }
    let elapsed = started.elapsed();

    let stats = vault.stats().await.context("fetching stats")?;
    let dedup_only_bytes = vault
        .dedup_only_bytes()
        .await
        .context("fetching dedup-only bytes")?;
    let seg_stats = vault.segment_stats().await.context("fetching segment stats")?;

    if json {
        let out = serde_json::json!({
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
    println!();

    println!("Storage layers");
    println!("{}", "-".repeat(60));
    let dedup_save_pct = pct_saved(stats.naive_bytes, dedup_only_bytes);
    let zstd_save_pct = pct_saved(dedup_only_bytes, stats.storage_bytes);
    let total_save_pct = pct_saved(stats.naive_bytes, stats.storage_bytes);
    println!(
        "  Naive (u32)   : {:>10}",
        format_bytes(stats.naive_bytes)
    );
    println!(
        "  After dedup   : {:>10}   {:>5.1}% saved by dedup alone",
        format_bytes(dedup_only_bytes),
        dedup_save_pct
    );
    println!(
        "  After zstd    : {:>10}   {:>5.1}% saved on top of dedup",
        format_bytes(stats.storage_bytes),
        zstd_save_pct
    );
    println!();
    println!(
        "  Combined      : {:.1}% smaller than naive  ({} \u{2192} {})",
        total_save_pct,
        format_bytes(stats.naive_bytes),
        format_bytes(stats.storage_bytes),
    );
    println!();

    println!("Per-segment-type breakdown");
    println!("{}", "-".repeat(60));
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
    Ok(())
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
