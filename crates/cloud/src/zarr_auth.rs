//! Azure Blob auth bridge for Zarr stores.
//!
//! Bridges Planetary Computer SAS token signing to `object_store` Azure
//! configuration for Zarr chunk-based access.

use std::sync::Arc;

use zarrs_object_store::AsyncObjectStore;
use zarrs_storage::AsyncReadableListableStorage;

use crate::error::{CloudError, Result};

/// Build a zarrs-compatible async store from a URL.
///
/// - **Azure Blob URLs** (`https://{account}.blob.core.windows.net/...`):
///   Uses `MicrosoftAzureBuilder` with optional SAS token auth.
/// - **Other HTTPS URLs**: Uses `HttpBuilder` (read-only).
pub async fn build_zarr_store(
    store_url: &str,
    sas_token: Option<&str>,
) -> Result<AsyncReadableListableStorage> {
    // Convert abfs:// to https:// if needed
    let url = abfs_to_https(store_url);

    if let Some((account, container, path)) = parse_azure_blob_url(&url) {
        build_azure_store(&account, &container, &path, sas_token).await
    } else {
        build_http_store(&url).await
    }
}

/// Build an Azure Blob Storage backed store, optionally with a path prefix.
async fn build_azure_store(
    account: &str,
    container: &str,
    path: &str,
    sas_token: Option<&str>,
) -> Result<AsyncReadableListableStorage> {
    use zarrs_object_store::object_store::azure::MicrosoftAzureBuilder;
    use zarrs_object_store::object_store::prefix::PrefixStore;

    let mut builder = MicrosoftAzureBuilder::new()
        .with_account(account)
        .with_container_name(container);

    if let Some(sas_query) = sas_token {
        let pairs = parse_sas_query(sas_query);
        builder = builder.with_sas_authorization(pairs);
    } else {
        // Anonymous access (public containers)
        builder = builder.with_skip_signature(true);
    }

    let store = builder
        .build()
        .map_err(|e| CloudError::Zarr(format!("failed to build Azure store: {e}")))?;

    // If there's a path prefix (e.g. "ERA5/2020/12/variable.zarr"),
    // wrap with PrefixStore so zarrs sees the store root at that path.
    if path.is_empty() {
        Ok(Arc::new(AsyncObjectStore::new(store)))
    } else {
        let prefixed = PrefixStore::new(store, path);
        Ok(Arc::new(AsyncObjectStore::new(prefixed)))
    }
}

/// Build an HTTP-backed store (read-only, for non-Azure URLs).
async fn build_http_store(url: &str) -> Result<AsyncReadableListableStorage> {
    use zarrs_object_store::object_store::http::HttpBuilder;
    use zarrs_object_store::object_store::ClientOptions;

    let client_opts = ClientOptions::new().with_allow_http(true);
    let store = HttpBuilder::new()
        .with_url(url)
        .with_client_options(client_opts)
        .build()
        .map_err(|e| CloudError::Zarr(format!("failed to build HTTP store: {e}")))?;

    Ok(Arc::new(AsyncObjectStore::new(store)))
}

/// Parse an Azure Blob Storage URL into (account, container, path).
///
/// Supports both formats:
/// - `https://{account}.blob.core.windows.net/{container}/{path...}`
/// - `abfs://{container}/{path...}` (account is inferred = container name)
pub fn parse_azure_blob_url(url: &str) -> Option<(String, String, String)> {
    // Handle abfs:// URLs: abfs://container/path
    if let Some(stripped) = url.strip_prefix("abfs://") {
        let (container, path) = stripped.split_once('/').unwrap_or((stripped, ""));
        // For abfs://, account = container (convention on Planetary Computer)
        return Some((container.to_string(), container.to_string(), path.to_string()));
    }

    let stripped = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))?;

    // account.blob.core.windows.net/container/path
    let (host, rest) = stripped.split_once('/')?;
    if !host.contains(".blob.core.windows.net") {
        return None;
    }

    let account = host.split('.').next()?.to_string();
    let (container, path) = rest.split_once('/').unwrap_or((rest, ""));

    Some((account, container.to_string(), path.to_string()))
}

/// Convert an `abfs://` URL to an `https://` Azure Blob URL.
///
/// If `account` is provided, uses it as the storage account.
/// Otherwise, uses the container name as the account (which is sometimes wrong).
///
/// `abfs://container/path` → `https://{account}.blob.core.windows.net/container/path`
pub fn abfs_to_https_with_account(url: &str, account: Option<&str>) -> String {
    if let Some(stripped) = url.strip_prefix("abfs://") {
        let container = stripped.split('/').next().unwrap_or(stripped);
        let acct = account.unwrap_or(container);
        format!("https://{acct}.blob.core.windows.net/{stripped}")
    } else {
        url.to_string()
    }
}

/// Convert an `abfs://` URL to an `https://` URL (uses container as account).
pub fn abfs_to_https(url: &str) -> String {
    abfs_to_https_with_account(url, None)
}

/// Parse a SAS query string into key-value pairs for `object_store`.
///
/// URL-decodes values since PC returns encoded tokens (e.g. `%3A` → `:`).
///
/// Input: `"se=2026-04-07T12%3A00%3A00Z&sp=r&sv=2023-11-03&sr=c&sig=abc123"`
/// Output: `vec![("se", "2026-04-07T12:00:00Z"), ("sp", "r"), ...]`
fn parse_sas_query(query: &str) -> Vec<(String, String)> {
    query
        .split('&')
        .filter_map(|pair| {
            let (k, v) = pair.split_once('=')?;
            Some((k.to_string(), url_decode(v)))
        })
        .collect()
}

/// Simple percent-decoding for SAS token values.
fn url_decode(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '%' {
            let hex: String = chars.by_ref().take(2).collect();
            if let Ok(byte) = u8::from_str_radix(&hex, 16) {
                result.push(byte as char);
            } else {
                result.push('%');
                result.push_str(&hex);
            }
        } else {
            result.push(c);
        }
    }
    result
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_azure_blob_url() {
        let url = "https://era5.blob.core.windows.net/era5-pds/era5-pds-2020.zarr";
        let (account, container, path) = parse_azure_blob_url(url).unwrap();
        assert_eq!(account, "era5");
        assert_eq!(container, "era5-pds");
        assert_eq!(path, "era5-pds-2020.zarr");
    }

    #[test]
    fn test_parse_azure_blob_url_no_path() {
        let url = "https://myaccount.blob.core.windows.net/mycontainer";
        let (account, container, path) = parse_azure_blob_url(url).unwrap();
        assert_eq!(account, "myaccount");
        assert_eq!(container, "mycontainer");
        assert_eq!(path, "");
    }

    #[test]
    fn test_parse_non_azure_url() {
        let url = "https://example.com/path/to/data.zarr";
        assert!(parse_azure_blob_url(url).is_none());
    }

    #[test]
    fn test_parse_sas_query() {
        let query = "se=2026-04-07&sp=r&sv=2023-11-03&sr=c&sig=abc123";
        let pairs = parse_sas_query(query);
        assert_eq!(pairs.len(), 5);
        assert_eq!(pairs[0], ("se".to_string(), "2026-04-07".to_string()));
        assert_eq!(pairs[1], ("sp".to_string(), "r".to_string()));
        assert_eq!(pairs[4], ("sig".to_string(), "abc123".to_string()));
    }
}
