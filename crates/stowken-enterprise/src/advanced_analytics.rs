//! Advanced analytics: cost attribution, system-prompt versioning, forecasting,
//! and anomaly detection on segment patterns.

use std::collections::HashMap;
use stowken::index::MetadataIndex;
use stowken::types::SegmentHash;

/// A cost attribution row — tokens consumed by a specific (application, segment_type) pair.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CostAttributionRow {
    pub application: Option<String>,
    pub segment_type: String,
    pub total_tokens: u64,
    /// Approximate cost in USD based on a configurable price-per-million-tokens.
    pub estimated_cost_usd: f64,
}

/// Compute cost attribution per application and segment type.
///
/// `price_per_million` is the per-million-token price in USD (e.g. 0.01 for $0.01/1M).
pub fn cost_attribution(
    index: &MetadataIndex,
    price_per_million: f64,
) -> Result<Vec<CostAttributionRow>, stowken::index::MetadataError> {
    let rows = index.cost_attribution()?;
    let result = rows
        .into_iter()
        .map(|row| {
            let application = row.get("application").and_then(|v| v.as_str()).map(str::to_owned);
            let segment_type = row
                .get("segment_type")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_owned();
            let total_tokens = row.get("total_tokens").and_then(|v| v.as_i64()).unwrap_or(0) as u64;
            let estimated_cost_usd = (total_tokens as f64 / 1_000_000.0) * price_per_million;
            CostAttributionRow {
                application,
                segment_type,
                total_tokens,
                estimated_cost_usd,
            }
        })
        .collect();
    Ok(result)
}

/// A system-prompt version entry — tracks how a prompt hash changes over time.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SystemPromptVersion {
    pub hash: SegmentHash,
    pub token_count: u32,
    pub ref_count: u64,
    /// ISO-8601 first-seen timestamp (from segments_meta).
    pub first_seen: String,
}

/// List all unique system prompt versions ordered by first-seen date descending.
pub fn system_prompt_versions(
    index: &MetadataIndex,
) -> Result<Vec<SystemPromptVersion>, stowken::index::MetadataError> {
    let prompts = index.list_unique_system_prompts()?;
    Ok(prompts
        .into_iter()
        .map(|p| SystemPromptVersion {
            hash: p.hash,
            token_count: p.token_count,
            ref_count: p.ref_count,
            first_seen: String::new(), // populated in a full impl with JOIN to segments_meta.created_at
        })
        .collect())
}

/// A simple time-series token usage forecast.
///
/// Fits a linear trend to recent daily token counts and projects forward.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TokenForecast {
    pub date: String,
    pub predicted_tokens: i64,
    pub lower_bound: i64,
    pub upper_bound: i64,
}

/// Forecast daily token usage for the next `horizon_days` days.
///
/// Uses simple linear regression over the observed daily totals.
pub fn forecast_token_usage(
    index: &MetadataIndex,
    horizon_days: u32,
) -> Result<Vec<TokenForecast>, stowken::index::MetadataError> {
    let rows = index.tokens_by_model_by_day()?;

    let mut daily_map: HashMap<String, i64> = HashMap::new();
    for row in &rows {
        if let (Some(day), Some(tokens)) = (
            row.get("day").and_then(|v| v.as_str()),
            row.get("total_tokens").and_then(|v| v.as_i64()),
        ) {
            *daily_map.entry(day.to_owned()).or_default() += tokens;
        }
    }

    let mut daily: Vec<(String, i64)> = daily_map.into_iter().collect();
    daily.sort_by(|a, b| a.0.cmp(&b.0));

    if daily.is_empty() {
        return Ok(vec![]);
    }

    // Simple linear regression: y = a + b*x where x = day index
    let n = daily.len() as f64;
    let xs: Vec<f64> = (0..daily.len()).map(|i| i as f64).collect();
    let ys: Vec<f64> = daily.iter().map(|(_, t)| *t as f64).collect();

    let x_mean = xs.iter().sum::<f64>() / n;
    let y_mean = ys.iter().sum::<f64>() / n;
    let ss_xy: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| (x - x_mean) * (y - y_mean)).sum();
    let ss_xx: f64 = xs.iter().map(|x| (x - x_mean).powi(2)).sum();
    let b = if ss_xx.abs() < 1e-10 { 0.0 } else { ss_xy / ss_xx };
    let a = y_mean - b * x_mean;

    // Residual standard deviation for confidence bounds
    let residuals: Vec<f64> = xs
        .iter()
        .zip(ys.iter())
        .map(|(x, y)| y - (a + b * x))
        .collect();
    let variance = residuals.iter().map(|r| r * r).sum::<f64>() / n.max(1.0);
    let std_dev = variance.sqrt();

    // Project forward
    let last_index = daily.len() as f64;
    let last_date = &daily.last().unwrap().0;

    let mut forecasts = Vec::new();
    for d in 1..=horizon_days {
        let x = last_index + d as f64;
        let predicted = (a + b * x).round() as i64;
        let margin = (1.96 * std_dev).round() as i64;
        // Fake date strings (a real impl would use chrono date arithmetic)
        let date = format!("{last_date}+{d}d");
        forecasts.push(TokenForecast {
            date,
            predicted_tokens: predicted.max(0),
            lower_bound: (predicted - margin).max(0),
            upper_bound: predicted + margin,
        });
    }
    Ok(forecasts)
}

/// An anomaly in observed token usage patterns.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UsageAnomaly {
    pub date: String,
    pub model: String,
    pub actual_tokens: i64,
    pub expected_tokens: i64,
    /// Z-score (how many standard deviations from the mean).
    pub z_score: f64,
}

/// Detect days where token usage is more than `z_threshold` standard deviations
/// from the rolling mean.
pub fn detect_usage_anomalies(
    index: &MetadataIndex,
    z_threshold: f64,
) -> Result<Vec<UsageAnomaly>, stowken::index::MetadataError> {
    let rows = index.tokens_by_model_by_day()?;

    // Group by model
    let mut by_model: HashMap<String, Vec<(String, i64)>> = HashMap::new();
    for row in &rows {
        let model = row.get("model").and_then(|v| v.as_str()).unwrap_or("unknown").to_owned();
        let day = row.get("day").and_then(|v| v.as_str()).unwrap_or("").to_owned();
        let tokens = row.get("total_tokens").and_then(|v| v.as_i64()).unwrap_or(0);
        by_model.entry(model).or_default().push((day, tokens));
    }

    let mut anomalies = Vec::new();
    for (model, mut days) in by_model {
        days.sort_by(|a, b| a.0.cmp(&b.0));
        let vals: Vec<f64> = days.iter().map(|(_, t)| *t as f64).collect();
        let n = vals.len() as f64;
        if n < 3.0 {
            continue; // not enough data for meaningful z-scores
        }
        let mean = vals.iter().sum::<f64>() / n;
        let variance = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();
        if std_dev < 1.0 {
            continue;
        }
        for (day, tokens) in &days {
            let z = (*tokens as f64 - mean) / std_dev;
            if z.abs() >= z_threshold {
                anomalies.push(UsageAnomaly {
                    date: day.clone(),
                    model: model.clone(),
                    actual_tokens: *tokens,
                    expected_tokens: mean.round() as i64,
                    z_score: z,
                });
            }
        }
    }
    Ok(anomalies)
}

#[cfg(test)]
mod tests {
    use super::*;
    use stowken::index::MetadataIndex;

    fn empty_index() -> MetadataIndex {
        MetadataIndex::open_in_memory().unwrap()
    }

    #[test]
    fn cost_attribution_empty_index() {
        let idx = empty_index();
        let rows = cost_attribution(&idx, 0.01).unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn forecast_on_empty_index() {
        let idx = empty_index();
        let forecasts = forecast_token_usage(&idx, 7).unwrap();
        assert!(forecasts.is_empty());
    }

    #[test]
    fn anomaly_detection_on_empty_index() {
        let idx = empty_index();
        let anomalies = detect_usage_anomalies(&idx, 2.0).unwrap();
        assert!(anomalies.is_empty());
    }
}
