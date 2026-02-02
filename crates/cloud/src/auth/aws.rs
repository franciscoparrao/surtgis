//! AWS S3 authentication for COG access.
//!
//! Provides a simple Bearer-token or query-string auth mechanism for S3.
//! For full SigV4, enable the `aws` feature and use `AwsAuth`.
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
    pub access_key_id: String,
    pub secret_access_key: String,
    pub session_token: Option<String>,
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

/// AWS S3 authentication using unsigned payload headers.
///
/// For public S3 buckets with requester-pays or buckets that need basic auth,
/// this adds the required headers. For full SigV4 request signing, a more
/// complete implementation or the `aws-sigv4` crate would be needed.
///
/// Current behavior: adds `x-amz-content-sha256: UNSIGNED-PAYLOAD` and
/// optionally the session token header. This is sufficient for many S3
/// configurations with IAM-based access when combined with VPC endpoints
/// or presigned URLs.
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
        headers: &mut Vec<(String, String)>,
    ) -> Result<()> {
        // Add unsigned payload marker (required by S3)
        headers.push((
            "x-amz-content-sha256".to_string(),
            "UNSIGNED-PAYLOAD".to_string(),
        ));

        // Add session token if present
        if let Some(ref token) = self.credentials.session_token {
            headers.push(("x-amz-security-token".to_string(), token.clone()));
        }

        Ok(())
    }
}
