// ── STAC Client for SurtGIS Web Demo ─────────────────────
// Handles catalog discovery, collection browsing, item search, and COG download.

const PROXY = '/proxy?url=';

// Try direct fetch first (CORS-enabled APIs), fall back to proxy
async function stacFetch(url) {
    try {
        const resp = await fetch(url, { headers: { Accept: 'application/json' } });
        if (resp.ok) return await resp.json();
    } catch (_) { /* CORS blocked, try proxy */ }
    const resp = await fetch(PROXY + encodeURIComponent(url));
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return await resp.json();
}

async function stacPost(url, body) {
    try {
        const resp = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', Accept: 'application/geo+json' },
            body: JSON.stringify(body),
        });
        if (resp.ok) return await resp.json();
    } catch (_) { /* CORS, try proxy */ }
    const resp = await fetch(PROXY + encodeURIComponent(url), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return await resp.json();
}

// ── Catalog discovery from STAC Index ────────────────────
export async function searchCatalogs(keyword) {
    const data = await stacFetch('https://stacindex.org/api/catalogs');
    let catalogs = Array.isArray(data) ? data : (data.catalogs || []);
    if (keyword) {
        const kw = keyword.toLowerCase();
        catalogs = catalogs.filter(c =>
            (c.title || '').toLowerCase().includes(kw) ||
            (c.id || '').toLowerCase().includes(kw) ||
            (c.description || '').toLowerCase().includes(kw)
        );
    }
    return catalogs.slice(0, 50).map(c => ({
        id: c.id || c.slug || 'unknown',
        title: c.title || c.id || 'Untitled',
        description: (c.description || '').slice(0, 120),
        url: c.url || c.link || '',
    }));
}

// ── Fetch collections from a STAC API ────────────────────
export async function fetchCollections(catalogUrl) {
    const url = catalogUrl.replace(/\/+$/, '') + '/collections';
    const data = await stacFetch(url);
    const cols = data.collections || [];
    return cols.map(c => ({
        id: c.id,
        title: c.title || c.id,
        description: (c.description || '').slice(0, 150),
        extent: c.extent,
    }));
}

// ── Search items in a collection ─────────────────────────
export async function searchItems(catalogUrl, collectionId, bbox, datetime, limit = 5) {
    const base = catalogUrl.replace(/\/+$/, '');

    // Ensure RFC3339 format: "2024-06-01" → "2024-06-01T00:00:00Z"
    const rfc3339 = datetime.split('/').map(d =>
        d.includes('T') ? d : d + 'T00:00:00Z'
    ).join('/');

    // Try POST /search first (STAC API spec)
    const body = {
        collections: [collectionId],
        bbox: bbox,
        datetime: rfc3339,
        limit: limit,
    };

    try {
        const data = await stacPost(base + '/search', body);
        if (data.features && data.features.length > 0) return data.features;
    } catch (e) {
        console.warn('POST /search failed, trying GET:', e.message);
    }

    // Fallback: GET /collections/{id}/items — build URL manually to avoid double encoding
    const qs = `bbox=${bbox.join(',')}&datetime=${rfc3339}&limit=${limit}`;
    const data = await stacFetch(`${base}/collections/${collectionId}/items?${qs}`);
    return data.features || [];
}

// ── Extract assets from a STAC item ──────────────────────
export function extractAssets(item) {
    if (!item.assets) return [];
    return Object.entries(item.assets).map(([key, asset]) => {
        // If there's an alternate_href or alternate.s3, prefer HTTP
        let href = asset.href;
        if (asset.alternate && asset.alternate.s3 && asset.alternate.s3.href) {
            href = asset.alternate.s3.href;
        }
        return {
            key,
            title: asset.title || key,
            href: href,
            type: asset.type || '',
            roles: asset.roles || [],
        };
    }).filter(a =>
        a.type.includes('tiff') || a.type.includes('geotiff') || a.type.includes('cog') ||
        a.type.includes('image/') || a.type === '' ||
        a.roles.includes('data') || a.roles.includes('visual') ||
        /^(B\d|SR_B|red|green|blue|nir|swir|scl|qa|elevation|dem|VV|VH|iw-)/i.test(a.key)
    );
}

// ── Check file size before download ───────────────────────
export async function checkFileSize(href) {
    let url = href;
    if (url.startsWith('s3://')) {
        const s3path = url.slice(5);
        url = `https://${s3path.split('/')[0]}.s3.amazonaws.com/${s3path.split('/').slice(1).join('/')}`;
    }
    try {
        const resp = await fetch(url, { method: 'HEAD' });
        const len = resp.headers.get('Content-Length');
        return len ? parseInt(len) : null;
    } catch (_) {
        return null; // Can't check, proceed anyway
    }
}

// ── Download COG preview (overview only, fast) ───────────
export async function downloadCOGPreview(href, maxDim = 512) {
    let url = href;
    if (url.startsWith('s3://')) {
        const s3path = url.slice(5);
        url = `https://${s3path.split('/')[0]}.s3.amazonaws.com/${s3path.split('/').slice(1).join('/')}`;
    }
    // Sign Planetary Computer URLs
    if (url.includes('blob.core.windows.net') && !url.includes('?')) {
        try {
            const signResp = await stacFetch(
                `https://planetarycomputer.microsoft.com/api/sas/v1/sign?href=${encodeURIComponent(href)}`
            );
            if (signResp.href) url = signResp.href;
        } catch (_) {}
    }

    // Try direct fromUrl (CORS-enabled servers)
    let tiff;
    try {
        tiff = await GeoTIFF.fromUrl(url);
    } catch (_) {
        // Fallback: use proxy with range support
        const proxyUrl = window.location.origin + PROXY + encodeURIComponent(url);
        tiff = await GeoTIFF.fromUrl(proxyUrl);
    }

    // Find best overview: pick the smallest image that's >= maxDim
    const count = await tiff.getImageCount();
    let bestIdx = 0;
    let bestImage = await tiff.getImage(0);
    const fullW = bestImage.getWidth(), fullH = bestImage.getHeight();

    if (count > 1) {
        // Overviews are index 1, 2, 3... (progressively smaller)
        for (let i = count - 1; i >= 1; i--) {
            const img = await tiff.getImage(i);
            const w = img.getWidth(), h = img.getHeight();
            if (w >= maxDim || h >= maxDim) {
                bestIdx = i;
                bestImage = img;
                break;
            }
        }
        // If all overviews are too small, use the largest overview
        if (bestIdx === 0 && count > 1) {
            bestImage = await tiff.getImage(1);
            bestIdx = 1;
        }
    }

    const w = bestImage.getWidth(), h = bestImage.getHeight();
    const data = (await bestImage.readRasters())[0];
    let bbox = null;
    try { bbox = bestImage.getBoundingBox(); } catch(_) {}

    let epsg = null;
    try {
        const gk = bestImage.getGeoKeys ? bestImage.getGeoKeys() : {};
        if (gk.ProjectedCSTypeGeoKey && gk.ProjectedCSTypeGeoKey !== 32767) epsg = gk.ProjectedCSTypeGeoKey;
        else if (gk.GeographicTypeGeoKey && gk.GeographicTypeGeoKey !== 32767) epsg = gk.GeographicTypeGeoKey;
    } catch(_) {}

    return {
        data, width: w, height: h, bbox, epsg,
        fullWidth: fullW, fullHeight: fullH,
        overviewIndex: bestIdx, overviewCount: count,
    };
}

// ── Download a COG asset (full resolution) ───────────────
export async function downloadCOG(href) {
    // Convert s3:// URLs to HTTPS
    let url = href;
    if (url.startsWith('s3://')) {
        const s3path = url.slice(5);
        url = `https://${s3path.split('/')[0]}.s3.amazonaws.com/${s3path.split('/').slice(1).join('/')}`;
    }
    if (href.includes('blob.core.windows.net') && !href.includes('?')) {
        try {
            const signResp = await stacFetch(
                `https://planetarycomputer.microsoft.com/api/sas/v1/sign?href=${encodeURIComponent(href)}`
            );
            if (signResp.href) url = signResp.href;
        } catch (_) { /* use unsigned */ }
    }

    // Try direct fetch for CORS-enabled hosts
    try {
        const resp = await fetch(url);
        if (resp.ok) {
            return await resp.arrayBuffer();
        }
    } catch (_) { /* CORS blocked */ }

    // Proxy fallback (works for any host, but limited to ~50MB)
    const resp = await fetch(PROXY + encodeURIComponent(url));
    if (!resp.ok) throw new Error(`Download failed: HTTP ${resp.status}`);
    return await resp.arrayBuffer();
}
