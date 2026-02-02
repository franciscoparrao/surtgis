//! Async STAC client for searching spatio-temporal asset catalogs.
//!
//! Supports Planetary Computer and Earth Search out of the box, plus
//! arbitrary STAC API endpoints via [`StacCatalog::Custom`].

use std::time::Duration;

use crate::error::{CloudError, Result};
use crate::stac_models::{StacItem, StacItemCollection, StacLink, StacSearchParams};

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
            all_items.extend(page.features.drain(..));

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
    /// For non-PC catalogs this is a no-op and returns the href unchanged.
    pub async fn sign_asset_href(
        &self,
        href: &str,
        _collection: &str,
    ) -> Result<String> {
        if !self.catalog.needs_signing() {
            return Ok(href.to_string());
        }
        self.sign_pc_href(href).await
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
                #[cfg(not(target_arch = "wasm32"))]
                {
                    let delay = Duration::from_millis(500 * (1 << (attempt - 1)));
                    tokio::time::sleep(delay).await;
                }
                // On WASM we skip the sleep (no tokio::time available).
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
                if let Some(ref link_body) = link.body {
                    if let (Some(base_obj), Some(link_obj)) =
                        (base.as_object_mut(), link_body.as_object())
                    {
                        for (k, v) in link_obj {
                            base_obj.insert(k.clone(), v.clone());
                        }
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

        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| CloudError::Auth(format!("PC sign request failed: {e}")))?;

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

        body["href"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| CloudError::Auth("PC sign response missing 'href' field".into()))
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
