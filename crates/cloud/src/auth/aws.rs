//! AWS S3 authentication for COG access.
//!
//! Provides a simple Bearer-token or query-string auth mechanism for S3.
//! Full SigV4 request signing is **not implemented**: [`AwsAuth`] returns
//! an explicit error instead of sending half-signed requests (audit
//! A2-cloud). Use presigned URLs, a VPC endpoint, or public-bucket HTTPS.
//!
//! This implementation reads credentials from environment variables:
//! - `AWS_ACCESS_KEY_ID`
//! - `AWS_SECRET_ACCESS_KEY`
//! - `AWS_SESSION_TOKEN` (optional)
//! - `AWS_REGION` (optional, defaults to `us-east-1`)

use crate::auth::CloudAuth;
use crate::error::{CloudError, Result};

/// AWS credentials loaded from environment variables.
#[derive(Debug, Clone)]
pub struct AwsCredentials {
    /// AWS access key ID (`AWS_ACCESS_KEY_ID`).
    pub access_key_id: String,
    /// AWS secret access key (`AWS_SECRET_ACCESS_KEY`).
    pub secret_access_key: String,
    /// Optional temporary session token (`AWS_SESSION_TOKEN`).
    pub session_token: Option<String>,
    /// AWS region (`AWS_REGION`, defaults to `us-east-1`).
    pub region: String,
}

impl AwsCredentials {
    /// Load credentials from standard AWS environment variables.
    pub fn from_env() -> Result<Self> {
        let access_key_id = std::env::var("AWS_ACCESS_KEY_ID")
            .map_err(|_| CloudError::Auth("AWS_ACCESS_KEY_ID not set".into()))?;
        let secret_access_key = std::env::var("AWS_SECRET_ACCESS_KEY")
            .map_err(|_| CloudError::Auth("AWS_SECRET_ACCESS_KEY not set".into()))?;
        let session_token = std::env::var("AWS_SESSION_TOKEN").ok();
        let region = std::env::var("AWS_REGION")
            .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
            .unwrap_or_else(|_| "us-east-1".to_string());

        Ok(Self {
            access_key_id,
            secret_access_key,
            session_token,
            region,
        })
    }

    /// Create credentials with explicit values.
    pub fn new(
        access_key_id: impl Into<String>,
        secret_access_key: impl Into<String>,
        region: impl Into<String>,
    ) -> Self {
        Self {
            access_key_id: access_key_id.into(),
            secret_access_key: secret_access_key.into(),
            session_token: None,
            region: region.into(),
        }
    }

    /// Set the session token (for temporary credentials / STS).
    pub fn with_session_token(mut self, token: impl Into<String>) -> Self {
        self.session_token = Some(token.into());
        self
    }
}

/// AWS S3 credential holder. **SigV4 request signing is not implemented.**
///
/// Signing a request with this type returns an explicit
/// [`CloudError::Auth`] error instead of sending a half-signed request that
/// S3 answers with a confusing 403. Until SigV4 lands, private S3 buckets
/// are reachable through presigned URLs, VPC endpoints, or any proxy that
/// injects the signature; public buckets need no auth object at all
/// (`s3://` URLs are rewritten to plain HTTPS).
pub struct AwsAuth {
    credentials: AwsCredentials,
}

impl AwsAuth {
    /// Create from explicit credentials.
    pub fn new(credentials: AwsCredentials) -> Self {
        Self { credentials }
    }

    /// Create from environment variables.
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            credentials: AwsCredentials::from_env()?,
        })
    }
}

impl CloudAuth for AwsAuth {
    fn sign_request(
        &self,
        _url: &str,
        _method: &str,
        _headers: &mut Vec<(String, String)>,
    ) -> Result<()> {
        // The old behavior pushed x-amz-content-sha256 / x-amz-security-token
        // without ever building `Authorization: AWS4-HMAC-SHA256`, so S3
        // rejected the request with a 403 even when the credentials were
        // valid. Fail honestly until real SigV4 is implemented.
        let _ = &self.credentials;
        Err(CloudError::Auth(
            "AWS SigV4 request signing is not implemented; use presigned URLs, \
             a VPC endpoint, or public-bucket HTTPS access instead"
                .to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_request_fails_honestly_until_sigv4_exists() {
        // A2-cloud: the old implementation half-signed the request (no
        // Authorization header) and private buckets answered 403 with valid
        // credentials. The contract until SigV4 lands is an explicit error.
        let auth = AwsAuth::new(AwsCredentials::new("AKIA_TEST", "secret", "us-east-1"));
        let mut headers = Vec::new();
        let err = auth
            .sign_request("https://bucket.s3.amazonaws.com/key", "GET", &mut headers)
            .unwrap_err();
        assert!(matches!(err, CloudError::Auth(_)));
        assert!(err.to_string().contains("not implemented"));
        assert!(headers.is_empty(), "no half-signed headers must be emitted");
    }
}
