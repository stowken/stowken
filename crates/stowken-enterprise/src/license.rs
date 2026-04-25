//! License validation for Stowken Enterprise.
//!
//! Licenses are Ed25519-signed JWTs validated either offline (embedded public key)
//! or online against the Stowken license server.

use base64::Engine as _;
use chrono::{DateTime, Utc};
use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors from license validation.
#[derive(Debug, Error)]
pub enum LicenseError {
    #[error("license token is malformed: {0}")]
    Malformed(String),
    #[error("license signature is invalid")]
    InvalidSignature,
    #[error("license has expired (expired {0})")]
    Expired(DateTime<Utc>),
    #[error("feature '{0}' is not included in this license")]
    FeatureNotLicensed(String),
    #[error("license conversation limit exceeded ({used} / {limit})")]
    LimitExceeded { used: u64, limit: u64 },
}

pub type LicenseResult<T> = Result<T, LicenseError>;

/// License tier determining which enterprise features are available.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LicenseTier {
    /// $500–2K/mo — cloud backends, multi-tenant, near-dedup.
    Team,
    /// $2K–10K/mo — all features including compliance, encryption, scaling.
    Enterprise,
}

impl LicenseTier {
    /// Returns the list of features included in this tier.
    pub fn included_features(&self) -> &'static [&'static str] {
        match self {
            Self::Team => &[
                "cloud_backends",
                "multi_tenant",
                "near_dedup",
                "parquet_export",
                "advanced_analytics",
            ],
            Self::Enterprise => &[
                "cloud_backends",
                "multi_tenant",
                "near_dedup",
                "parquet_export",
                "advanced_analytics",
                "compliance",
                "encryption",
                "horizontal_scaling",
                "gdpr_deletion",
                "audit_log",
                "retention_policies",
            ],
        }
    }
}

/// Claims embedded inside a license JWT.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LicenseClaims {
    /// Issued-to (subject) — the licensee's organisation ID.
    pub sub: String,
    /// Issued-at (Unix timestamp).
    pub iat: i64,
    /// Expiry (Unix timestamp).
    pub exp: i64,
    /// License tier.
    pub tier: LicenseTier,
    /// Maximum stored conversations; `None` = unlimited.
    pub max_conversations: Option<u64>,
    /// Maximum tokens processed per calendar month; `None` = unlimited.
    pub max_tokens_per_month: Option<u64>,
    /// Explicit feature overrides (additional grants or removals).
    #[serde(default)]
    pub features: Vec<String>,
}

/// A validated license ready for enforcement checks.
#[derive(Debug, Clone)]
pub struct License {
    pub tenant_id: String,
    pub tier: LicenseTier,
    pub max_conversations: Option<u64>,
    pub max_tokens_per_month: Option<u64>,
    pub features: Vec<String>,
    pub expires_at: DateTime<Utc>,
}

impl License {
    /// Parse and verify a license JWT.
    ///
    /// The JWT is structured as `base64(header).base64(claims).base64(signature)`.
    /// The signature covers `header.claims` (the standard JWT format) using Ed25519.
    pub fn from_jwt(token: &str) -> LicenseResult<Self> {
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() != 3 {
            return Err(LicenseError::Malformed("expected 3 JWT parts".into()));
        }

        // Verify signature
        let message = format!("{}.{}", parts[0], parts[1]);
        let sig_bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(parts[2])
            .map_err(|_| LicenseError::Malformed("bad signature encoding".into()))?;
        let sig = Signature::from_slice(&sig_bytes)
            .map_err(|_| LicenseError::Malformed("signature wrong length".into()))?;

        let pubkey = embedded_public_key();
        pubkey
            .verify_strict(message.as_bytes(), &sig)
            .map_err(|_| LicenseError::InvalidSignature)?;

        // Decode claims
        let claims_json = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(parts[1])
            .map_err(|_| LicenseError::Malformed("bad claims encoding".into()))?;
        let claims: LicenseClaims = serde_json::from_slice(&claims_json)
            .map_err(|e| LicenseError::Malformed(e.to_string()))?;

        // Check expiry
        let expires_at = DateTime::from_timestamp(claims.exp, 0)
            .unwrap_or(DateTime::<Utc>::MIN_UTC);
        if expires_at < Utc::now() {
            return Err(LicenseError::Expired(expires_at));
        }

        // Merge tier features with explicit grants
        let mut features: Vec<String> = claims
            .tier
            .included_features()
            .iter()
            .map(|s| s.to_string())
            .collect();
        for f in &claims.features {
            if !features.contains(f) {
                features.push(f.clone());
            }
        }

        Ok(License {
            tenant_id: claims.sub,
            tier: claims.tier,
            max_conversations: claims.max_conversations,
            max_tokens_per_month: claims.max_tokens_per_month,
            features,
            expires_at,
        })
    }

    /// Validate a feature is licensed.
    pub fn require_feature(&self, feature: &str) -> LicenseResult<()> {
        if self.features.iter().any(|f| f == feature) {
            Ok(())
        } else {
            Err(LicenseError::FeatureNotLicensed(feature.to_owned()))
        }
    }

    /// Validate the conversation count is within limits.
    pub fn check_conversation_limit(&self, current: u64) -> LicenseResult<()> {
        if let Some(limit) = self.max_conversations {
            if current >= limit {
                return Err(LicenseError::LimitExceeded {
                    used: current,
                    limit,
                });
            }
        }
        Ok(())
    }
}

/// The Ed25519 public key embedded in the enterprise crate.
///
/// The matching private key is held exclusively by the Stowken license server.
/// Replace this with the real key before shipping.
fn embedded_public_key() -> VerifyingKey {
    // 32-byte placeholder public key (all zeros — replace in production).
    let bytes = [0u8; 32];
    VerifyingKey::from_bytes(&bytes).expect("valid placeholder key")
}

/// Create a development/test license that never expires and enables all features.
///
/// **For testing only.** Never ship a binary that calls this in production.
#[cfg(any(test, feature = "dev-license"))]
pub fn dev_license(tenant_id: &str) -> License {
    License {
        tenant_id: tenant_id.to_owned(),
        tier: LicenseTier::Enterprise,
        max_conversations: None,
        max_tokens_per_month: None,
        features: LicenseTier::Enterprise
            .included_features()
            .iter()
            .map(|s| s.to_string())
            .collect(),
        expires_at: DateTime::<Utc>::MAX_UTC,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dev_license_has_all_features() {
        let lic = dev_license("test-org");
        assert_eq!(lic.tier, LicenseTier::Enterprise);
        lic.require_feature("compliance").unwrap();
        lic.require_feature("encryption").unwrap();
        lic.require_feature("cloud_backends").unwrap();
    }

    #[test]
    fn feature_not_licensed_returns_error() {
        let lic = License {
            tenant_id: "test".into(),
            tier: LicenseTier::Team,
            max_conversations: None,
            max_tokens_per_month: None,
            features: vec!["cloud_backends".into()],
            expires_at: DateTime::<Utc>::MAX_UTC,
        };
        assert!(lic.require_feature("compliance").is_err());
        assert!(lic.require_feature("cloud_backends").is_ok());
    }

    #[test]
    fn conversation_limit_enforced() {
        let lic = License {
            tenant_id: "test".into(),
            tier: LicenseTier::Team,
            max_conversations: Some(100),
            max_tokens_per_month: None,
            features: vec![],
            expires_at: DateTime::<Utc>::MAX_UTC,
        };
        assert!(lic.check_conversation_limit(99).is_ok());
        assert!(lic.check_conversation_limit(100).is_err());
    }
}
