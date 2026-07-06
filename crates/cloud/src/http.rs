//! HTTP client wrapper with Range request support and retry logic.

use crate::auth::CloudAuth;
use crate::error::{CloudError, Result};
use reqwest::Client;
use std::borrow::Cow;
use std::sync::{Arc, OnceLock};
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
            return Cow::Owned(format!("https://{}.s3.amazonaws.com/{}", bucket, key));
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
    /// Create a new HTTP client tuned for high-concurrency COG / STAC fetches.
    ///
    /// Notable settings (native targets only; WASM uses defaults):
    /// - ALPN-negotiated HTTP/2 is enabled by default in reqwest; we add HTTP/2
    ///   PING keepalives so stale connections to S3 / Azure Blob are detected
    ///   before the next range request times out.
    /// - Connection pool size is bumped to 64 idle per host with a 60 s idle
    ///   timeout so concurrent tile fetches reuse warm sockets rather than
    ///   redoing TLS each time. Reqwest's default keeps idle conns only for
    ///   90 s but allows them to expire quickly under churn; the explicit
    ///   `pool_max_idle_per_host` ensures we keep the persistent fleet
    ///   regardless of churn.
    /// - TCP_NODELAY disables Nagle's algorithm — small range-request packets
    ///   should ship immediately rather than wait for ACK coalescing.
    /// - TCP keepalive at 30 s catches half-open connections (NAT timeouts,
    ///   mobile hotspots) without a request-level timeout firing first.
    ///
    /// These changes target the "compute in Chile, data in Azure" deployment
    /// pattern documented in `BUG_RAM_V070_BUDGET_VS_REAL_MAULE_PC.md`, where
    /// network RTT dominates wall-clock and connection reuse matters most.
    pub fn new(request_timeout: Duration, max_retries: u32) -> Result<Self> {
        let builder = Client::builder().timeout(request_timeout);

        #[cfg(not(target_arch = "wasm32"))]
        let builder = builder
            .pool_max_idle_per_host(64)
            .pool_idle_timeout(Some(Duration::from_secs(60)))
            .tcp_nodelay(true)
            .tcp_keepalive(Some(Duration::from_secs(30)))
            .http2_adaptive_window(true)
            .http2_keep_alive_interval(Duration::from_secs(30))
            .http2_keep_alive_timeout(Duration::from_secs(10))
            .http2_keep_alive_while_idle(true);

        let client = builder.build()?;

        Ok(Self {
            client,
            max_retries,
            request_timeout,
        })
    }

    /// Send a HEAD request to discover file size and Range support.
    pub async fn head(&self, url: &str, auth: &dyn CloudAuth) -> Result<HeadInfo> {
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

        let mut req = self.client.get(url).header("Range", &range_value);

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
            return Err(CloudError::HttpStatus {
                status,
                url: url.to_string(),
            });
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
        let coalesced_ranges: Vec<(u64, u64)> =
            coalesced.iter().map(|g| (g.offset, g.length)).collect();
        let fetched = self
            .fetch_ranges_parallel(url, &coalesced_ranges, auth)
            .await?;

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
                    results[orig_idx] = self
                        .fetch_range(url, ranges[orig_idx].0, ranges[orig_idx].1, auth)
                        .await?;
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

    /// Execute a request with retry.
    ///
    /// Two failure classes are retried, up to `max_retries` times:
    ///
    /// - **Transport errors** (timeout / connect), with exponential backoff
    ///   plus jitter ([`backoff_with_jitter`]).
    /// - **Retryable HTTP statuses** (429, 500, 502, 503, 504 — see
    ///   [`is_retryable_status`]), honouring the server's `Retry-After`
    ///   header (delta-seconds or HTTP-date, capped at [`RETRY_AFTER_CAP`])
    ///   and falling back to the same exponential backoff + jitter when the
    ///   header is absent or unparseable.
    ///
    /// The backoff sleep is async ([`backoff_sleep`]) so it never blocks a
    /// runtime worker thread. A retryable status on the final attempt is
    /// returned as a response; callers map non-success statuses to
    /// [`CloudError::HttpStatus`](crate::error::CloudError::HttpStatus).
    async fn execute_with_retry(
        &self,
        request: reqwest::RequestBuilder,
    ) -> std::result::Result<reqwest::Response, reqwest::Error> {
        let mut last_err: Option<reqwest::Error> = None;
        let mut server_delay: Option<Duration> = None;

        for attempt in 0..=self.max_retries {
            if attempt > 0 {
                let delay = server_delay
                    .take()
                    .unwrap_or_else(|| backoff_with_jitter(attempt));
                backoff_sleep(delay).await;
            }

            let cloned = match request.try_clone() {
                Some(c) => c,
                // Non-cloneable request (streaming body): single attempt only.
                None => return request.send().await,
            };

            match cloned.send().await {
                Ok(resp) => {
                    let status = resp.status();
                    if is_retryable_status(status) && attempt < self.max_retries {
                        server_delay = resp
                            .headers()
                            .get("retry-after")
                            .and_then(|v| v.to_str().ok())
                            .and_then(|v| parse_retry_after(v, unix_now_secs(), RETRY_AFTER_CAP));
                        continue;
                    }
                    return Ok(resp);
                }
                Err(e) if e.is_timeout() || e.is_connect() => {
                    last_err = Some(e);
                    continue;
                }
                Err(e) => return Err(e),
            }
        }

        // Only reachable when the final iteration hit the transport-error
        // `continue` above, which always sets `last_err` first.
        Err(last_err.expect("retry loop exhausted without a transport error"))
    }

    /// Getter for the timeout duration.
    pub fn request_timeout(&self) -> Duration {
        self.request_timeout
    }
}

// ---------------------------------------------------------------------------
// Process-wide shared client
// ---------------------------------------------------------------------------

/// Timeout/retry configuration used to build the [`shared_client`] singleton.
///
/// Matches [`crate::cog_reader::CogReaderOptions::default`]'s
/// `request_timeout`/`max_retries` — the common case for every native call
/// site in this workspace today.
const SHARED_CLIENT_TIMEOUT: Duration = Duration::from_secs(30);
const SHARED_CLIENT_MAX_RETRIES: u32 = 3;

static SHARED_CLIENT: OnceLock<Arc<HttpClient>> = OnceLock::new();

/// Return the process-wide shared [`HttpClient`], building it on first use.
///
/// # Why this exists
///
/// [`HttpClient::new`] builds a fresh `reqwest::Client` — and therefore a
/// fresh connection pool and TLS config — every time it's called. Two
/// `HttpClient`s pointed at the same host never share a socket, so opening
/// the same remote COG through N separate `HttpClient::new()` calls costs N
/// TLS handshakes to that host instead of one warm, reused connection. The
/// STAC composite pipeline reopens the same COG many times (once per
/// strip/tile task), so a shared client — and therefore a shared pool — is
/// one of the largest available wall-clock wins on that path.
///
/// This function lazily builds **one** `HttpClient` for the whole process
/// (via [`OnceLock`]) using the same pool tuning as [`HttpClient::new`] (see
/// its docs for the `pool_max_idle_per_host` / keepalive rationale), and
/// hands out cheap `Arc` clones of it thereafter.
///
/// Callers that need a different timeout or retry policy than
/// [`SHARED_CLIENT_TIMEOUT`]/[`SHARED_CLIENT_MAX_RETRIES`] should keep using
/// [`HttpClient::new`] directly to build their own private client — this
/// function does not replace that constructor, it's an additional, opt-in
/// fast path for the common case.
pub fn shared_client() -> Arc<HttpClient> {
    SHARED_CLIENT
        .get_or_init(|| {
            Arc::new(
                HttpClient::new(SHARED_CLIENT_TIMEOUT, SHARED_CLIENT_MAX_RETRIES)
                    .expect("failed to build process-wide shared reqwest client"),
            )
        })
        .clone()
}

// ---------------------------------------------------------------------------
// Retry helpers
// ---------------------------------------------------------------------------

/// Maximum delay honoured from a `Retry-After` header.
///
/// Servers occasionally send very large values (or far-future HTTP-dates);
/// anything above this cap is clamped to the cap rather than stalling the
/// caller for minutes.
const RETRY_AFTER_CAP: Duration = Duration::from_secs(60);

/// HTTP statuses worth retrying: rate limiting (429) and transient
/// server-side failures (500, 502, 503, 504).
fn is_retryable_status(status: reqwest::StatusCode) -> bool {
    matches!(status.as_u16(), 429 | 500 | 502 | 503 | 504)
}

/// Seconds since the Unix epoch, saturating to 0 on pre-epoch clocks.
fn unix_now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Exponential backoff with jitter: base 100ms · 2^(attempt-1), plus up to
/// +50% jitter (derived from the clock's sub-second nanos — no `rand` dep)
/// to de-synchronize concurrent retries against the same host.
fn backoff_with_jitter(attempt: u32) -> Duration {
    let base_ms = 100u64 << attempt.saturating_sub(1).min(10);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| u64::from(d.subsec_nanos()))
        .unwrap_or(0);
    let jitter_ms = nanos % (base_ms / 2).max(1);
    Duration::from_millis(base_ms + jitter_ms)
}

/// Async, runtime-friendly sleep for retry backoff.
///
/// Uses `tokio::time::sleep` so the backoff yields the worker thread instead
/// of blocking it — with a small shared runtime (see `sync_api`), a blocking
/// sleep here would freeze all in-flight I/O for the whole process.
#[cfg(not(target_arch = "wasm32"))]
async fn backoff_sleep(delay: Duration) {
    tokio::time::sleep(delay).await;
}

/// On WASM there is no timer without extra deps (`gloo-timers`); yield to the
/// microtask queue and rely on the retry round-trip itself as backoff.
#[cfg(target_arch = "wasm32")]
async fn backoff_sleep(_delay: Duration) {
    futures::future::ready(()).await;
}

/// Parse a `Retry-After` header value into a wait [`Duration`].
///
/// Supports both forms from RFC 9110 §10.2.3:
/// - delta-seconds, e.g. `"120"`;
/// - IMF-fixdate, e.g. `"Wed, 21 Oct 2015 07:28:00 GMT"` (interpreted
///   relative to `now_unix`, seconds since the Unix epoch).
///
/// The result is clamped to `cap`; dates in the past yield a zero duration.
/// Returns `None` when the value matches neither form.
fn parse_retry_after(value: &str, now_unix: i64, cap: Duration) -> Option<Duration> {
    let value = value.trim();
    if let Ok(secs) = value.parse::<u64>() {
        return Some(Duration::from_secs(secs).min(cap));
    }
    let target = parse_imf_fixdate(value)?;
    let delta = target.saturating_sub(now_unix).max(0);
    Some(Duration::from_secs(delta as u64).min(cap))
}

/// Parse an IMF-fixdate (`"Sun, 06 Nov 1994 08:49:37 GMT"`) into Unix seconds.
///
/// Hand-rolled to avoid pulling `chrono`/`httpdate` into the default feature
/// set; precision needs are modest since the result is capped anyway.
fn parse_imf_fixdate(s: &str) -> Option<i64> {
    // "<day-name>, <day> <month> <year> <HH>:<MM>:<SS> GMT"
    let rest = s.split_once(',')?.1;
    let mut parts = rest.split_whitespace();
    let day: i64 = parts.next()?.parse().ok()?;
    let month: i64 = match parts.next()? {
        "Jan" => 1,
        "Feb" => 2,
        "Mar" => 3,
        "Apr" => 4,
        "May" => 5,
        "Jun" => 6,
        "Jul" => 7,
        "Aug" => 8,
        "Sep" => 9,
        "Oct" => 10,
        "Nov" => 11,
        "Dec" => 12,
        _ => return None,
    };
    let year: i64 = parts.next()?.parse().ok()?;
    let mut time = parts.next()?.split(':');
    let h: i64 = time.next()?.parse().ok()?;
    let m: i64 = time.next()?.parse().ok()?;
    let sec: i64 = time.next()?.parse().ok()?;
    if parts.next()? != "GMT" {
        return None;
    }
    if !(1..=31).contains(&day) || h > 23 || m > 59 || sec > 60 {
        return None;
    }
    Some(days_from_civil(year, month, day) * 86_400 + h * 3600 + m * 60 + sec)
}

/// Days since 1970-01-01 for a proleptic Gregorian date (Howard Hinnant's
/// `days_from_civil` algorithm).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
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
    let mut indexed: Vec<(usize, u64, u64)> = ranges
        .iter()
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
        let out = normalize_url(
            "s3://copernicus-dem-30m/Copernicus_DSM_COG_10_S36_00_W071_00_DEM/Copernicus_DSM_COG_10_S36_00_W071_00_DEM.tif",
        );
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

    // ── Retry helpers ────────────────────────────────────────────────

    #[test]
    fn retryable_statuses_are_429_and_transient_5xx() {
        use reqwest::StatusCode;
        for code in [429u16, 500, 502, 503, 504] {
            assert!(
                is_retryable_status(StatusCode::from_u16(code).unwrap()),
                "{code} should be retryable"
            );
        }
        for code in [200u16, 206, 301, 400, 403, 404, 416, 501] {
            assert!(
                !is_retryable_status(StatusCode::from_u16(code).unwrap()),
                "{code} should NOT be retryable"
            );
        }
    }

    #[test]
    fn retry_after_delta_seconds() {
        let cap = Duration::from_secs(60);
        assert_eq!(parse_retry_after("5", 0, cap), Some(Duration::from_secs(5)));
        assert_eq!(
            parse_retry_after(" 30 ", 0, cap),
            Some(Duration::from_secs(30))
        );
        // Values above the cap are clamped to the cap.
        assert_eq!(parse_retry_after("120", 0, cap), Some(cap));
        assert_eq!(parse_retry_after("0", 0, cap), Some(Duration::ZERO));
    }

    #[test]
    fn retry_after_http_date() {
        let cap = Duration::from_secs(60);
        // 2015-10-21 07:28:00 UTC == 1445412480 (verified against `date -u`).
        let date = "Wed, 21 Oct 2015 07:28:00 GMT";
        let target = 1_445_412_480i64;

        // 30 seconds in the future → wait 30s.
        assert_eq!(
            parse_retry_after(date, target - 30, cap),
            Some(Duration::from_secs(30))
        );
        // Date in the past → zero wait (retry immediately).
        assert_eq!(
            parse_retry_after(date, target + 100, cap),
            Some(Duration::ZERO)
        );
        // Far-future date → clamped to the cap.
        assert_eq!(parse_retry_after(date, target - 3600, cap), Some(cap));
    }

    #[test]
    fn retry_after_garbage_is_none() {
        let cap = Duration::from_secs(60);
        assert_eq!(parse_retry_after("soon", 0, cap), None);
        assert_eq!(parse_retry_after("", 0, cap), None);
        assert_eq!(parse_retry_after("-5", 0, cap), None);
        assert_eq!(parse_retry_after("Wed, 21 Oct 2015", 0, cap), None);
        // Missing the trailing "GMT" token.
        assert_eq!(parse_retry_after("Wed, 21 Oct 2015 07:28:00", 0, cap), None);
        // Unknown month.
        assert_eq!(
            parse_retry_after("Wed, 21 Foo 2015 07:28:00 GMT", 0, cap),
            None
        );
    }

    #[test]
    fn imf_fixdate_epoch_and_leap_years() {
        assert_eq!(parse_imf_fixdate("Thu, 01 Jan 1970 00:00:00 GMT"), Some(0));
        assert_eq!(
            parse_imf_fixdate("Thu, 01 Jan 1970 00:01:40 GMT"),
            Some(100)
        );
        // 2000-03-01 (day after the leap day of a century leap year)
        // == 951868800 (verified against `date -u`).
        assert_eq!(
            parse_imf_fixdate("Wed, 01 Mar 2000 00:00:00 GMT"),
            Some(951_868_800)
        );
    }

    // ── Shared client ────────────────────────────────────────────────

    /// `shared_client()` must hand out clones of the *same* underlying
    /// `HttpClient` (and therefore the same `reqwest::Client` / connection
    /// pool) across repeated calls, rather than constructing a fresh one
    /// each time. This is the whole point of the singleton: reopening the
    /// same COG many times should reuse warm sockets instead of paying a
    /// fresh TLS handshake per open.
    #[test]
    fn shared_client_returns_the_same_instance_every_call() {
        let a = shared_client();
        let b = shared_client();
        assert!(
            Arc::ptr_eq(&a, &b),
            "shared_client() must return clones of one process-wide instance"
        );
    }

    /// Calling `shared_client()` concurrently from many threads must still
    /// converge on a single constructed instance (guards against a naive
    /// non-atomic "check then build" race).
    #[test]
    fn shared_client_is_singleton_under_concurrent_first_use() {
        let handles: Vec<_> = (0..16).map(|_| std::thread::spawn(shared_client)).collect();
        let clients: Vec<Arc<HttpClient>> =
            handles.into_iter().map(|h| h.join().unwrap()).collect();
        let first = &clients[0];
        for c in &clients[1..] {
            assert!(
                Arc::ptr_eq(first, c),
                "all threads must observe the same client"
            );
        }
    }

    #[test]
    fn backoff_grows_exponentially_with_bounded_jitter() {
        for attempt in 1..=5u32 {
            let base = 100u64 << (attempt - 1);
            let d = backoff_with_jitter(attempt).as_millis() as u64;
            assert!(
                d >= base && d < base + base / 2 + 1,
                "attempt {attempt}: {d}ms outside [{base}, {})",
                base + base / 2 + 1
            );
        }
    }
}
