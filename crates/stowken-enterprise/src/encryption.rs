//! AES-256-GCM encryption at rest with pluggable KMS key sources.

use aes_gcm::{
    aead::{Aead, KeyInit, OsRng},
    Aes256Gcm, Nonce,
};
use rand::RngCore;
use thiserror::Error;

/// Errors from encryption/decryption.
#[derive(Debug, Error)]
pub enum EncryptionError {
    #[error("encryption failed: {0}")]
    Encrypt(String),
    #[error("decryption failed: ciphertext is invalid or key is wrong")]
    Decrypt,
    #[error("key provider error: {0}")]
    KeyProvider(String),
    #[error("invalid key length: expected 32 bytes, got {0}")]
    InvalidKeyLength(usize),
}

pub type EncryptResult<T> = Result<T, EncryptionError>;

/// Trait for pluggable KMS backends.
///
/// Implement this to wrap AWS KMS, GCP KMS, Azure Key Vault, or HashiCorp Vault.
/// Stowken never stores the raw key; only the wrapped/ciphertext key is persisted.
#[async_trait::async_trait]
pub trait KeyProvider: Send + Sync {
    /// Return the 32-byte AES-256 key for a given key reference.
    async fn get_key(&self, key_ref: &str) -> EncryptResult<[u8; 32]>;
    /// Return the primary key reference used for new encryptions.
    fn primary_key_ref(&self) -> &str;
}

/// A raw in-memory key provider (for testing and local deployments).
///
/// In production, wrap `AwsKmsProvider`, `GcpKmsProvider`, etc.
pub struct RawKeyProvider {
    key: [u8; 32],
    key_ref: String,
}

impl RawKeyProvider {
    /// Create a provider from a 32-byte AES key.
    pub fn new(key: [u8; 32]) -> Self {
        Self { key, key_ref: "local".to_owned() }
    }

    /// Generate a random 32-byte key.
    pub fn random() -> Self {
        let mut key = [0u8; 32];
        OsRng.fill_bytes(&mut key);
        Self { key, key_ref: "local-random".to_owned() }
    }
}

#[async_trait::async_trait]
impl KeyProvider for RawKeyProvider {
    async fn get_key(&self, _key_ref: &str) -> EncryptResult<[u8; 32]> {
        Ok(self.key)
    }
    fn primary_key_ref(&self) -> &str {
        &self.key_ref
    }
}

/// AES-256-GCM encryptor/decryptor.
///
/// Each ciphertext is prefixed with a random 12-byte nonce.
pub struct Encryptor {
    key: [u8; 32],
}

impl Encryptor {
    pub fn new(key: [u8; 32]) -> Self {
        Self { key }
    }

    /// Encrypt `plaintext`. Returns `nonce (12 bytes) || ciphertext`.
    pub fn encrypt(&self, plaintext: &[u8]) -> EncryptResult<Vec<u8>> {
        let cipher = Aes256Gcm::new_from_slice(&self.key)
            .map_err(|e| EncryptionError::Encrypt(e.to_string()))?;

        let mut nonce_bytes = [0u8; 12];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = cipher
            .encrypt(nonce, plaintext)
            .map_err(|e| EncryptionError::Encrypt(e.to_string()))?;

        let mut out = Vec::with_capacity(12 + ciphertext.len());
        out.extend_from_slice(&nonce_bytes);
        out.extend_from_slice(&ciphertext);
        Ok(out)
    }

    /// Decrypt `data` (must be `nonce || ciphertext` from `encrypt`).
    pub fn decrypt(&self, data: &[u8]) -> EncryptResult<Vec<u8>> {
        if data.len() < 12 {
            return Err(EncryptionError::Decrypt);
        }
        let cipher = Aes256Gcm::new_from_slice(&self.key)
            .map_err(|_| EncryptionError::Decrypt)?;

        let nonce = Nonce::from_slice(&data[..12]);
        cipher
            .decrypt(nonce, &data[12..])
            .map_err(|_| EncryptionError::Decrypt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encrypt_decrypt_roundtrip() {
        let mut key = [0u8; 32];
        OsRng.fill_bytes(&mut key);
        let enc = Encryptor::new(key);
        let plaintext = b"hello token world";
        let ciphertext = enc.encrypt(plaintext).unwrap();
        let recovered = enc.decrypt(&ciphertext).unwrap();
        assert_eq!(recovered, plaintext);
    }

    #[test]
    fn wrong_key_fails_decryption() {
        let mut key_a = [0u8; 32];
        let mut key_b = [1u8; 32];
        OsRng.fill_bytes(&mut key_a);
        OsRng.fill_bytes(&mut key_b);
        let enc_a = Encryptor::new(key_a);
        let enc_b = Encryptor::new(key_b);
        let ciphertext = enc_a.encrypt(b"secret").unwrap();
        assert!(enc_b.decrypt(&ciphertext).is_err());
    }

    #[tokio::test]
    async fn raw_key_provider() {
        let provider = RawKeyProvider::random();
        let key = provider.get_key("local").await.unwrap();
        assert_eq!(key.len(), 32);
    }
}
