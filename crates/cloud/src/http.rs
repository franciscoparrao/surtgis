//! HTTP client wrapper with Range request support and retry logic.

use crate::auth::CloudAuth;
use crate::error::{CloudError, Result};
use reqwest::Client;
use std::time::Duration;

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

    /// Fetch multiple byte ranges concurrently.
    ///
    /// Each element in `ranges` is `(offset, length)`.
    /// Returns one `Vec<u8>` per range, in the same order.
    pub async fn fetch_ranges(
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
