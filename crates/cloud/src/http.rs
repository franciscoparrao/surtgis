//! HTTP client wrapper with Range request support and retry logic.

use crate::auth::CloudAuth;
use crate::error::{CloudError, Result};
use reqwest::Client;
use std::borrow::Cow;
use std::time::Duration;

/// Translate `s3://bucket/key` into `https://bucket.s3.amazonaws.com/key` so
/// anonymous public-bucket reads work through a standard HTTPS client.
///
/// Some STAC catalogs (notably Earth Search for the `cop-dem-glo-30`
/// collection) return raw `s3://` hrefs instead of HTTPS. reqwest can't build
/// a request from the `s3://` scheme directly — but AWS serves every public
/// bucket on virtual-hosted-style HTTPS, and auto-redirects (307) from
/// `s3.amazonaws.com` to the bucket's regional endpoint, so we don't need to
/// know the region in advance. reqwest follows redirects by default.
///
/// Non-`s3://` URLs pass through unchanged.
pub(crate) fn normalize_url(url: &str) -> Cow<'_, str> {
    if let Some(rest) = url.strip_prefix("s3://") {
        if let Some((bucket, key)) = rest.split_once('/') {
            return Cow::Owned(format!(
                "https://{}.s3.amazonaws.com/{}",
                bucket, key
            ));
        }
    }
    Cow::Borrowed(url)
}

/// HTTP client for fetching byte ranges from remote files.
pub struct HttpClient {
    client: Client,
    max_retries: u32,
    request_timeout: Duration,
}

/// Response from a HEAD request.
pub struct HeadInfo {
    /// Total file size in bytes, if reported by the server.
    pub content_length: Option<u64>,
    /// Whether the server supports Range requests.
    pub accept_ranges: bool,
}

impl HttpClient {
    /// Create a new HTTP client.
    pub fn new(request_timeout: Duration, max_retries: u32) -> Result<Self> {
        let client = Client::builder()
            .timeout(request_timeout)
            .build()?;

        Ok(Self {
            client,
            max_retries,
            request_timeout,
        })
    }

    /// Send a HEAD request to discover file size and Range support.
    pub async fn head(
        &self,
        url: &str,
        auth: &dyn CloudAuth,
    ) -> Result<HeadInfo> {
        let url_norm = normalize_url(url);
        let url = url_norm.as_ref();
        let mut auth_headers = Vec::new();
        auth.sign_request(url, "HEAD", &mut auth_headers)?;

        let mut req = self.client.head(url);
        for (key, value) in &auth_headers {
            req = req.header(key.as_str(), value.as_str());
        }

        let resp = self.execute_with_retry(req).await?;

        let accept_ranges = resp
            .headers()
            .get("accept-ranges")
            .and_then(|v| v.to_str().ok())
            .map(|v| v.contains("bytes"))
            .unwrap_or(false);

        let content_length = resp
            .headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok());

        Ok(HeadInfo {
            content_length,
            accept_ranges,
        })
    }

    /// Fetch a byte range from a remote file.
    ///
    /// Returns the raw bytes for `[offset .. offset + length)`.
    pub async fn fetch_range(
        &self,
        url: &str,
        offset: u64,
        length: u64,
        auth: &dyn CloudAuth,
    ) -> Result<Vec<u8>> {
        let url_norm = normalize_url(url);
        let url = url_norm.as_ref();
        let mut auth_headers = Vec::new();
        auth.sign_request(url, "GET", &mut auth_headers)?;

        let range_value = format!("bytes={}-{}", offset, offset + length - 1);

        let mut req = self
            .client
            .get(url)
            .header("Range", &range_value);

        for (key, value) in &auth_headers {
            req = req.header(key.as_str(), value.as_str());
        }

        let resp = self.execute_with_retry(req).await?;

        let status = resp.status();
        if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE
            || (status.is_success() && status != reqwest::StatusCode::PARTIAL_CONTENT)
        {
            return Err(CloudError::RangeNotSupported {
                url: url.to_string(),
            });
        }

        if !status.is_success() {
            return Err(CloudError::Network(format!(
                "HTTP {} fetching {}",
                status, url
            )));
        }

        let bytes = resp.bytes().await?;
        Ok(bytes.to_vec())
    }

    /// Fetch multiple byte ranges concurrently, with automatic coalescing.
    ///
    /// Adjacent or overlapping ranges are merged into a single HTTP request
    /// for efficiency. Each element in `ranges` is `(offset, length)`.
    /// Returns one `Vec<u8>` per original range, in the same order.
    pub async fn fetch_ranges(
        &self,
        url: &str,
        ranges: &[(u64, u64)],
        auth: &dyn CloudAuth,
    ) -> Result<Vec<Vec<u8>>> {
        if ranges.is_empty() {
            return Ok(Vec::new());
        }

        // Coalesce adjacent ranges (within 4KB gap tolerance).
        // COG tiles are often sequential in the file.
        let coalesced = coalesce_ranges(ranges, 4096);

        if coalesced.len() == ranges.len() {
            // No coalescing benefit — fetch all concurrently
            return self.fetch_ranges_parallel(url, ranges, auth).await;
        }

        // Fetch coalesced ranges concurrently
        let coalesced_ranges: Vec<(u64, u64)> = coalesced.iter()
            .map(|g| (g.offset, g.length))
            .collect();
        let fetched = self.fetch_ranges_parallel(url, &coalesced_ranges, auth).await?;

        // Split coalesced data back into original ranges
        let mut results = vec![Vec::new(); ranges.len()];
        for (group_idx, group) in coalesced.iter().enumerate() {
            let group_data = &fetched[group_idx];
            for &(orig_idx, local_offset, local_length) in &group.members {
                let start = local_offset as usize;
                let end = (local_offset + local_length) as usize;
                if end <= group_data.len() {
                    results[orig_idx] = group_data[start..end].to_vec();
                } else {
                    // Fallback: fetch individually if coalesced data is short
                    results[orig_idx] = self.fetch_range(
                        url, ranges[orig_idx].0, ranges[orig_idx].1, auth
                    ).await?;
                }
            }
        }

        Ok(results)
    }

    /// Fetch multiple ranges concurrently without coalescing.
    async fn fetch_ranges_parallel(
        &self,
        url: &str,
        ranges: &[(u64, u64)],
        auth: &dyn CloudAuth,
    ) -> Result<Vec<Vec<u8>>> {
        use futures::stream::{FuturesOrdered, StreamExt};

        let mut futs = FuturesOrdered::new();
        for &(offset, length) in ranges {
            futs.push_back(self.fetch_range(url, offset, length, auth));
        }

        let mut results = Vec::with_capacity(ranges.len());
        while let Some(res) = futs.next().await {
            results.push(res?);
        }

        Ok(results)
    }

    /// Execute a request with exponential backoff retry.
    async fn execute_with_retry(
        &self,
        request: reqwest::RequestBuilder,
    ) -> std::result::Result<reqwest::Response, reqwest::Error> {
        let mut last_err = None;

        for attempt in 0..=self.max_retries {
            if attempt > 0 {
                let backoff_ms = 100u64 * 2u64.pow(attempt - 1);
                // Platform-agnostic sleep: on native use a simple future-based
                // approach that doesn't require tokio::time. On WASM, reqwest
                // handles its own async runtime.
                futures::future::ready(()).await;
                // Simple busy-wait-free backoff using the runtime's timer if
                // available.  Since reqwest already depends on tokio on native
                // and wasm_bindgen on WASM, we can use a platform-conditional
                // sleep.
                #[cfg(not(target_arch = "wasm32"))]
                {
                    // On native, use std::thread::sleep in a spawn_blocking
                    // context or simply rely on the retry itself as backoff.
                    // For simplicity without requiring tokio::time, we use
                    // std::thread::sleep which blocks the current thread
                    // briefly—acceptable for retry backoff.
                    std::thread::sleep(Duration::from_millis(backoff_ms));
                }
                #[cfg(target_arch = "wasm32")]
                {
                    // On WASM, we cannot block. Just yield and proceed.
                    // Real WASM retry backoff would need gloo_timers, but
                    // we keep deps minimal.
                    let _ = backoff_ms;
                }
            }

            match request.try_clone() {
                Some(cloned) => match cloned.send().await {
                    Ok(resp) => return Ok(resp),
                    Err(e) if e.is_timeout() || e.is_connect() => {
                        last_err = Some(e);
                        continue;
                    }
                    Err(e) => return Err(e),
                },
                None => {
                    return request.send().await;
                }
            }
        }

        Err(last_err.unwrap())
    }

    /// Getter for the timeout duration.
    pub fn request_timeout(&self) -> Duration {
        self.request_timeout
    }
}

// ---------------------------------------------------------------------------
// Range coalescing
// ---------------------------------------------------------------------------

/// A group of original ranges merged into a single fetch.
struct CoalescedRange {
    /// Start offset of the merged range.
    offset: u64,
    /// Total length of the merged range.
    length: u64,
    /// Members: (original_index, local_offset_within_group, original_length).
    members: Vec<(usize, u64, u64)>,
}

/// Merge adjacent/overlapping byte ranges to reduce HTTP request count.
///
/// Ranges within `gap_tolerance` bytes of each other are merged.
/// Returns groups that can be fetched as single requests and split afterwards.
fn coalesce_ranges(ranges: &[(u64, u64)], gap_tolerance: u64) -> Vec<CoalescedRange> {
    if ranges.is_empty() {
        return Vec::new();
    }

    // Sort by offset, keeping track of original indices
    let mut indexed: Vec<(usize, u64, u64)> = ranges.iter()
        .enumerate()
        .map(|(i, &(o, l))| (i, o, l))
        .collect();
    indexed.sort_by_key(|&(_, o, _)| o);

    let mut groups: Vec<CoalescedRange> = Vec::new();
    let (first_idx, first_off, first_len) = indexed[0];
    let mut current = CoalescedRange {
        offset: first_off,
        length: first_len,
        members: vec![(first_idx, 0, first_len)],
    };

    for &(orig_idx, offset, length) in &indexed[1..] {
        let current_end = current.offset + current.length;
        if offset <= current_end + gap_tolerance {
            // Merge: extend current group
            let local_offset = offset - current.offset;
            current.members.push((orig_idx, local_offset, length));
            let new_end = offset + length;
            if new_end > current_end {
                current.length = new_end - current.offset;
            }
        } else {
            // Gap too large: start new group
            groups.push(current);
            current = CoalescedRange {
                offset,
                length,
                members: vec![(orig_idx, 0, length)],
            };
        }
    }
    groups.push(current);

    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_s3_scheme_to_https_virtual_hosted() {
        let out = normalize_url("s3://copernicus-dem-30m/Copernicus_DSM_COG_10_S36_00_W071_00_DEM/Copernicus_DSM_COG_10_S36_00_W071_00_DEM.tif");
        assert_eq!(
            out.as_ref(),
            "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_S36_00_W071_00_DEM/Copernicus_DSM_COG_10_S36_00_W071_00_DEM.tif"
        );
    }

    #[test]
    fn normalize_https_passes_through_unchanged() {
        let url = "https://planetarycomputer.microsoft.com/api/stac/v1";
        let out = normalize_url(url);
        assert_eq!(out.as_ref(), url);
        // Borrowed, not allocated
        assert!(matches!(out, std::borrow::Cow::Borrowed(_)));
    }

    #[test]
    fn normalize_malformed_s3_without_key_passes_through() {
        // "s3://bucket" without a trailing key — leave as-is rather than
        // producing a surprising URL. Caller will get a clearer reqwest error.
        let url = "s3://mybucket";
        let out = normalize_url(url);
        assert_eq!(out.as_ref(), url);
    }
}
