//! No-op authentication for public endpoints.

use crate::auth::CloudAuth;
use crate::error::Result;

/// No authentication — used for public COG endpoints.
pub struct NoAuth;

impl CloudAuth for NoAuth {
    fn sign_request(
        &self,
        _url: &str,
        _method: &str,
        _headers: &mut Vec<(String, String)>,
    ) -> Result<()> {
        Ok(())
    }
}
