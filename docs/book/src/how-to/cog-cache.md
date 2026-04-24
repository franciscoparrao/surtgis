# Cache COG tiles for fast re-runs

Any STAC or COG command that supports `--cache` will persist downloaded
tiles to disk and serve them from cache on subsequent runs over the same
bbox and asset.

## Enable

```bash
surtgis stac composite --cache --catalog pc --asset red,nir ...
surtgis stac fetch-mosaic --cache --catalog pc --collection cop-dem-glo-30 ...
```

The first run downloads from the origin; subsequent runs over the same
bbox read from local disk. No code changes, no extra flags beyond
`--cache` being present.

## Where the cache lives

Default: `$XDG_CACHE_HOME/surtgis/cog/` (typically `~/.cache/surtgis/cog/`
on Linux, `~/Library/Caches/surtgis/cog/` on macOS). Override with:

```bash
export XDG_CACHE_HOME=/path/to/big/disk
```

Cache keys are derived from the COG base URL (without SAS query parameters)
plus the exact bbox. This means re-signed Azure Blob URLs still hit cache,
but shifting the bbox by even one pixel produces a cache miss.

## When to use it

Always, if you're iterating on the same area of interest. The cost is
disk space (each tile is typically 1–10 MB, a medium study area might
cache 0.5–2 GB); the benefit is that subsequent runs go from minutes to
seconds.

## When not to use it

- One-shot production runs where you don't expect to re-read the tiles.
- Highly variable bboxes that don't share tile coverage (each run caches
  fresh, adds disk pressure, returns nothing usable).
- Sentinel-2 scenes with very short SAS token lifetimes where you might
  cache items whose signed URLs go stale. This is rare in practice;
  SurtGIS re-signs tokens automatically within the same run.

## Cache hygiene

No automatic eviction yet. Manually clear when the cache gets too big:

```bash
rm -rf ~/.cache/surtgis/cog/
```

A TTL-based eviction policy is on the roadmap. Until then, a cron job
removing files older than N days is a reasonable workaround.

## Disk layout

Tiles are sharded into 256 subdirectories by hash prefix so no single
directory grows unbounded:

```
~/.cache/surtgis/cog/
├── 00/
│   ├── 42/
│   │   └── <rest-of-hash>.tif
│   └── ...
├── 01/
└── ...
```

Safe to browse with any file manager. The `.tif` files are regular
GeoTIFFs and can be opened directly.
