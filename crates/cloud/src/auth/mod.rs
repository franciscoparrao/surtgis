//! Authentication traits and implementations for cloud storage.

pub mod aws;
mod none;

pub use aws::{AwsAuth, AwsCredentials};
pub use none::NoAuth;

use crate::error::Result;

/// Trait for signing HTTP requests to cloud storage providers.
///
/// Implementations add authentication headers (e.g., AWS SigV4, Bearer tokens)
/// to outgoing requests before they are sent.
pub trait CloudAuth: Send + Sync {
    /// Sign a request by adding authentication headers.
    ///
    /// `url` is the full request URL, `headers` is a mutable map where
    /// auth headers should be inserted.
    fn sign_request(
        &self,
        url: &str,
        method: &str,
        headers: &mut Vec<(String, String)>,
    ) -> Result<()>;
}
