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
    storage::FilesystemBackend,
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

    /// Output as JSON (where applicable).
    #[arg(long, short = 'j', global = true)]
    json: bool,

    #[command(subcommand)]
    command: Command,
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
        Command::AuditSystemPrompts { min_refs, limit } => {
            cmd_audit_system_prompts(&data_dir, config, min_refs, limit, cli.json).await?;
        }
        Command::Cost { by, model, application } => {
            cmd_cost(&data_dir, config, &by, model, application, cli.json).await?;
        }
        Command::FindDuplicates { threshold, limit } => {
            cmd_find_duplicates(&data_dir, config, threshold, limit, cli.json).await?;
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
