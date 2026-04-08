//! Async STAC client for searching spatio-temporal asset catalogs.
//!
//! Supports Planetary Computer and Earth Search out of the box, plus
//! arbitrary STAC API endpoints via [`StacCatalog::Custom`].

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::error::{CloudError, Result};
use crate::stac_models::{StacItem, StacItemCollection, StacLink, StacSearchParams};

// ---------------------------------------------------------------------------
// SAS Token Cache
// ---------------------------------------------------------------------------

/// Cache for Planetary Computer SAS tokens.
///
/// PC signs URLs at the container level: the SAS query string from one signed
/// URL is valid for any file in the same Azure Blob container. We cache tokens
/// per container and reuse them, eliminating the 200ms throttle per asset.
///
/// Token structure: `?se=<expiry>&sp=r&sv=...&sr=c&sig=<signature>`
/// Typical validity: ~1 hour.
struct SasTokenCache {
    /// Map from container key (account + container) to (sas_query, obtained_at)
    tokens: Mutex<HashMap<String, (String, Instant)>>,
    /// Maximum age before a token is considered stale (re-sign).
    /// PC tokens are valid ~1h, we refresh at 45min.
    max_age: Duration,
}

impl SasTokenCache {
    fn new() -> Self {
        Self {
            tokens: Mutex::new(HashMap::new()),
            max_age: Duration::from_secs(45 * 60), // 45 minutes
        }
    }

    /// Extract the container key from an Azure Blob Storage URL.
    /// `https://account.blob.core.windows.net/container/path/file.tif` → `account/container`
    fn container_key(href: &str) -> Option<String> {
        let url = href.strip_prefix("https://").or_else(|| href.strip_prefix("http://"))?;
        let parts: Vec<&str> = url.splitn(3, '/').collect();
        if parts.len() >= 2 {
            Some(format!("{}/{}", parts[0], parts[1]))
        } else {
            None
        }
    }

    /// Try to apply a cached SAS token to the given href.
    /// Returns Some(signed_url) if a valid cached token exists.
    fn try_sign(&self, href: &str) -> Option<String> {
        let key = Self::container_key(href)?;
        let tokens = self.tokens.lock().ok()?;
        let (sas_query, obtained_at) = tokens.get(&key)?;
        if obtained_at.elapsed() < self.max_age {
            // Apply cached SAS token: strip any existing query and append cached one
            let base = href.split('?').next().unwrap_or(href);
            Some(format!("{}?{}", base, sas_query))
        } else {
            None
        }
    }

    /// Store a SAS token extracted from a signed URL.
    fn cache_from_signed_url(&self, href: &str, signed_url: &str) {
        if let Some(key) = Self::container_key(href) {
            if let Some(query) = signed_url.split_once('?').map(|(_, q)| q.to_string()) {
                if let Ok(mut tokens) = self.tokens.lock() {
                    tokens.insert(key, (query, Instant::now()));
                }
            }
        }
    }

    /// Number of cached containers.
    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.tokens.lock().map(|t| t.len()).unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Catalog enum
// ---------------------------------------------------------------------------

/// Well-known STAC catalogs plus custom endpoints.
#[derive(Debug, Clone)]
pub enum StacCatalog {
    /// Microsoft Planetary Computer STAC API.
    PlanetaryComputer,
    /// AWS Earth Search (Element 84).
    EarthSearch,
    /// Any STAC API endpoint (provide the root URL, e.g.
    /// `"https://my-stac.example.com/api/v1"`).
    Custom(String),
}

impl StacCatalog {
    /// Return the full POST `/search` URL for this catalog.
    pub fn search_url(&self) -> String {
        match self {
            Self::PlanetaryComputer => {
                "https://planetarycomputer.microsoft.com/api/stac/v1/search".to_string()
            }
            Self::EarthSearch => {
                "https://earth-search.aws.element84.com/v1/search".to_string()
            }
            Self::Custom(base) => {
                let base = base.trim_end_matches('/');
                if base.ends_with("/search") {
                    base.to_string()
                } else {
                    format!("{}/search", base)
                }
            }
        }
    }

    /// Parse a shorthand string into a catalog.
    ///
    /// Recognized shorthands: `"pc"`, `"planetary-computer"`, `"es"`,
    /// `"earth-search"`. Anything else is treated as a custom URL.
    pub fn from_str_or_url(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "pc" | "planetary-computer" | "planetarycomputer" => Self::PlanetaryComputer,
            "es" | "earth-search" | "earthsearch" => Self::EarthSearch,
            url => Self::Custom(url.to_string()),
        }
    }

    /// Whether this catalog requires SAS token signing for asset access.
    pub fn needs_signing(&self) -> bool {
        matches!(self, Self::PlanetaryComputer)
    }
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// Configuration for [`StacClient`].
pub struct StacClientOptions {
    /// Per-request timeout (default 30 s).
    pub request_timeout: Duration,
    /// Maximum retries on transient failures (default 3).
    pub max_retries: u32,
    /// Maximum total items to fetch across pages (default 100).
    pub max_items: usize,
}

impl Default for StacClientOptions {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(30),
            max_retries: 3,
            max_items: 100,
        }
    }
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// Async client for STAC Item Search.
pub struct StacClient {
    catalog: StacCatalog,
    client: reqwest::Client,
    options: StacClientOptions,
    sas_cache: SasTokenCache,
}

impl StacClient {
    /// Create a new STAC client.
    pub fn new(catalog: StacCatalog, options: StacClientOptions) -> Result<Self> {
        let builder = reqwest::Client::builder();
        #[cfg(not(target_arch = "wasm32"))]
        let builder = builder.timeout(options.request_timeout);
        let client = builder
            .build()
            .map_err(|e| CloudError::Network(format!("failed to build HTTP client: {e}")))?;

        Ok(Self {
            catalog,
            client,
            options,
            sas_cache: SasTokenCache::new(),
        })
    }

    /// The catalog this client is configured for.
    pub fn catalog(&self) -> &StacCatalog {
        &self.catalog
    }

    // ── Single-page search ──────────────────────────────────────────

    /// Execute a single search request and return one page of results.
    pub async fn search(&self, params: &StacSearchParams) -> Result<StacItemCollection> {
        let url = self.catalog.search_url();
        self.post_search(&url, params).await
    }

    // ── Paginated search ────────────────────────────────────────────

    /// Search with automatic pagination, collecting up to `max_items` items.
    pub async fn search_all(&self, params: &StacSearchParams) -> Result<Vec<StacItem>> {
        let mut all_items: Vec<StacItem> = Vec::new();
        let max = self.options.max_items;

        // First page
        let mut page = self.search(params).await?;

        loop {
            let next = page.next_link().cloned();
            all_items.append(&mut page.features);

            if all_items.len() >= max {
                break;
            }

            match next {
                Some(link) => {
                    page = self.follow_next(&link, params).await?;
                    if page.is_empty() {
                        break;
                    }
                }
                None => break,
            }
        }

        all_items.truncate(max);
        Ok(all_items)
    }

    // ── Planetary Computer SAS token signing ────────────────────────

    /// Sign an asset href for Planetary Computer via the `/sign` endpoint.
    ///
    /// Uses a per-container SAS token cache: the first sign for a storage
    /// container hits the PC API, subsequent signs for the same container
    /// reuse the cached token (valid ~45 min). This eliminates the 200ms
    /// throttle for most requests.
    ///
    /// For non-PC catalogs this is a no-op and returns the href unchanged.
    pub async fn sign_asset_href(
        &self,
        href: &str,
        _collection: &str,
    ) -> Result<String> {
        if !self.catalog.needs_signing() {
            return Ok(href.to_string());
        }

        // Try cache first
        if let Some(signed) = self.sas_cache.try_sign(href) {
            return Ok(signed);
        }

        // Cache miss → sign via API and cache the token
        let signed = self.sign_pc_href(href).await?;
        self.sas_cache.cache_from_signed_url(href, &signed);
        Ok(signed)
    }

    /// Extract a raw SAS query string for a Planetary Computer URL.
    ///
    /// Returns `Some("se=...&sp=r&sv=...&sr=c&sig=...")` for PC catalogs,
    /// `None` for non-PC catalogs. Uses the SAS token cache.
    #[cfg(feature = "zarr")]
    pub async fn get_sas_token(&self, href: &str) -> Result<Option<String>> {
        if !self.catalog.needs_signing() {
            return Ok(None);
        }
        let signed = self.sign_asset_href(href, "").await?;
        Ok(signed.split_once('?').map(|(_, q)| q.to_string()))
    }

    /// Get collection-level SAS token and storage info from Planetary Computer.
    ///
    /// Uses the STAC collection metadata (`msft:storage_account`, `msft:container`)
    /// and the `/api/sas/v1/token/{collection}` endpoint.
    ///
    /// Returns `Some((token, account, container))` for PC catalogs, `None` for others.
    #[cfg(feature = "zarr")]
    pub async fn get_collection_zarr_auth(
        &self,
        collection: &str,
    ) -> Result<Option<(String, String, String)>> {
        if !self.catalog.needs_signing() {
            return Ok(None);
        }

        // Get storage account and container from collection metadata
        let coll_url = format!(
            "https://planetarycomputer.microsoft.com/api/stac/v1/collections/{}",
            collection
        );
        let coll_resp = self
            .client
            .get(&coll_url)
            .send()
            .await
            .map_err(|e| CloudError::Auth(format!("PC collection request failed: {e}")))?;

        let coll_body: serde_json::Value = coll_resp
            .json()
            .await
            .map_err(|e| CloudError::Auth(format!("parsing PC collection: {e}")))?;

        let account = coll_body["msft:storage_account"]
            .as_str()
            .unwrap_or("")
            .to_string();
        let container = coll_body["msft:container"]
            .as_str()
            .unwrap_or("")
            .to_string();

        // Get SAS token
        let token_url = format!(
            "https://planetarycomputer.microsoft.com/api/sas/v1/token/{}",
            collection
        );

        let resp = self
            .client
            .get(&token_url)
            .send()
            .await
            .map_err(|e| CloudError::Auth(format!("PC token request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(CloudError::Auth(format!(
                "PC token returned HTTP {}: {}",
                status,
                body.chars().take(300).collect::<String>()
            )));
        }

        let body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| CloudError::Auth(format!("parsing PC token: {e}")))?;

        let token = body["token"]
            .as_str()
            .ok_or_else(|| CloudError::Auth("PC token response missing 'token'".into()))?
            .to_string();

        Ok(Some((token, account, container)))
    }

    // ── Private helpers ─────────────────────────────────────────────

    async fn post_search(
        &self,
        url: &str,
        params: &StacSearchParams,
    ) -> Result<StacItemCollection> {
        let mut last_err = None;

        for attempt in 0..=self.options.max_retries {
            if attempt > 0 {
                // Exponential backoff: 500ms, 1s, 2s, ...
                #[cfg(feature = "native")]
                {
                    let delay = Duration::from_millis(500 * (1 << (attempt - 1)));
                    tokio::time::sleep(delay).await;
                }
                // On WASM or without native feature we skip the sleep.
            }

            let resp = self
                .client
                .post(url)
                .header("Content-Type", "application/json")
                .json(params)
                .send()
                .await;

            match resp {
                Ok(r) if r.status().is_success() => {
                    let body = r
                        .text()
                        .await
                        .map_err(|e| CloudError::Network(format!("reading response body: {e}")))?;
                    let col: StacItemCollection = serde_json::from_str(&body).map_err(|e| {
                        CloudError::Network(format!("parsing STAC response: {e}"))
                    })?;
                    return Ok(col);
                }
                Ok(r) => {
                    let status = r.status();
                    let body = r.text().await.unwrap_or_default();
                    last_err = Some(CloudError::Network(format!(
                        "STAC search returned HTTP {}: {}",
                        status,
                        body.chars().take(500).collect::<String>()
                    )));
                    // Don't retry client errors (4xx)
                    if status.is_client_error() {
                        break;
                    }
                }
                Err(e) => {
                    last_err = Some(CloudError::Network(format!("STAC search request failed: {e}")));
                }
            }
        }

        Err(last_err.unwrap_or_else(|| CloudError::Network("STAC search failed".into())))
    }

    /// Follow a pagination link. Handles both POST (body/merge) and GET links.
    async fn follow_next(
        &self,
        link: &StacLink,
        original_params: &StacSearchParams,
    ) -> Result<StacItemCollection> {
        let method = link
            .method
            .as_deref()
            .unwrap_or("GET")
            .to_uppercase();

        if method == "POST" {
            // Build the body for the next request
            let body = if link.merge.unwrap_or(false) {
                // Merge: start from original params, overlay link body
                let mut base = serde_json::to_value(original_params)
                    .map_err(|e| CloudError::Network(format!("serializing params: {e}")))?;
                if let Some(ref link_body) = link.body
                    && let (Some(base_obj), Some(link_obj)) =
                        (base.as_object_mut(), link_body.as_object())
                {
                    for (k, v) in link_obj {
                        base_obj.insert(k.clone(), v.clone());
                    }
                }
                base
            } else if let Some(ref link_body) = link.body {
                link_body.clone()
            } else {
                serde_json::to_value(original_params)
                    .map_err(|e| CloudError::Network(format!("serializing params: {e}")))?
            };

            // Parse the merged body back to StacSearchParams for retry logic
            let merged: StacSearchParams = serde_json::from_value(body)
                .map_err(|e| CloudError::Network(format!("parsing merged params: {e}")))?;
            self.post_search(&link.href, &merged).await
        } else {
            // GET-based pagination
            let resp = self
                .client
                .get(&link.href)
                .send()
                .await
                .map_err(|e| CloudError::Network(format!("GET pagination: {e}")))?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                return Err(CloudError::Network(format!(
                    "STAC pagination returned HTTP {}: {}",
                    status,
                    body.chars().take(500).collect::<String>()
                )));
            }

            let body = resp
                .text()
                .await
                .map_err(|e| CloudError::Network(format!("reading pagination body: {e}")))?;
            serde_json::from_str(&body)
                .map_err(|e| CloudError::Network(format!("parsing pagination response: {e}")))
        }
    }

    /// Sign a single href via the Planetary Computer `/api/sas/v1/sign` endpoint.
    ///
    /// Returns the fully-signed URL ready for HTTP Range requests.
    async fn sign_pc_href(&self, href: &str) -> Result<String> {
        let url = format!(
            "https://planetarycomputer.microsoft.com/api/sas/v1/sign?href={}",
            href
        );

        let max_retries = self.options.max_retries.max(5); // more retries for large batches
        let mut last_err = None;

        // Throttle: small delay before each sign request to avoid rate-limiting.
        // PC allows ~100 req/min. With 200+ assets, we need to pace requests.
        #[cfg(feature = "native")]
        {
            tokio::time::sleep(Duration::from_millis(200)).await;
        }

        for attempt in 0..=max_retries {
            if attempt > 0 {
                // Exponential backoff: 2s, 4s, 8s, 16s, 32s
                #[cfg(feature = "native")]
                {
                    let delay = Duration::from_millis(2000 * (1 << (attempt - 1)));
                    tokio::time::sleep(delay).await;
                }
            }

            let resp = match self.client.get(&url).send().await {
                Ok(r) => r,
                Err(e) => {
                    last_err = Some(CloudError::Auth(format!("PC sign request failed: {e}")));
                    continue;
                }
            };

            if resp.status().as_u16() == 429 || resp.status().is_server_error() {
                // Rate limited or server error — retry
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                last_err = Some(CloudError::Auth(format!(
                    "PC sign HTTP {} (attempt {}/{}): {}",
                    status, attempt + 1, max_retries + 1,
                    body.chars().take(200).collect::<String>()
                )));
                continue;
            }

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                return Err(CloudError::Auth(format!(
                    "PC sign returned HTTP {}: {}",
                    status,
                    body.chars().take(300).collect::<String>()
                )));
            }

            let body: serde_json::Value = resp
                .json()
                .await
                .map_err(|e| CloudError::Auth(format!("parsing PC sign response: {e}")))?;

            return body["href"]
                .as_str()
                .map(|s| s.to_string())
                .ok_or_else(|| CloudError::Auth("PC sign response missing 'href'".into()));
        }

        Err(last_err.unwrap_or_else(|| CloudError::Auth("PC sign: all retries exhausted".into())))
    }

}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_search_urls() {
        assert_eq!(
            StacCatalog::PlanetaryComputer.search_url(),
            "https://planetarycomputer.microsoft.com/api/stac/v1/search"
        );
        assert_eq!(
            StacCatalog::EarthSearch.search_url(),
            "https://earth-search.aws.element84.com/v1/search"
        );
        assert_eq!(
            StacCatalog::Custom("https://example.com/stac".into()).search_url(),
            "https://example.com/stac/search"
        );
        // Already has /search
        assert_eq!(
            StacCatalog::Custom("https://example.com/stac/search".into()).search_url(),
            "https://example.com/stac/search"
        );
        // Trailing slash
        assert_eq!(
            StacCatalog::Custom("https://example.com/stac/".into()).search_url(),
            "https://example.com/stac/search"
        );
    }

    #[test]
    fn catalog_from_str_or_url() {
        assert!(matches!(
            StacCatalog::from_str_or_url("pc"),
            StacCatalog::PlanetaryComputer
        ));
        assert!(matches!(
            StacCatalog::from_str_or_url("es"),
            StacCatalog::EarthSearch
        ));
        assert!(matches!(
            StacCatalog::from_str_or_url("https://my-stac.com"),
            StacCatalog::Custom(_)
        ));
    }

    #[test]
    fn needs_signing() {
        assert!(StacCatalog::PlanetaryComputer.needs_signing());
        assert!(!StacCatalog::EarthSearch.needs_signing());
        assert!(!StacCatalog::Custom("https://x.com".into()).needs_signing());
    }
}
