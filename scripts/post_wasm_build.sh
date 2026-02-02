#!/usr/bin/env bash
# Post wasm-pack build: patch pkg/package.json with custom fields.
# wasm-pack overwrites package.json on every build, so this script
# re-applies metadata that npm needs.

set -euo pipefail

PKG_DIR="${1:-pkg}"
PKG_JSON="$PKG_DIR/package.json"

if [ ! -f "$PKG_JSON" ]; then
  echo "ERROR: $PKG_JSON not found. Run wasm-pack build first."
  exit 1
fi

# Use a temp file to merge fields via node (available in any CI with npm)
node -e "
const fs = require('fs');
const pkg = JSON.parse(fs.readFileSync('$PKG_JSON', 'utf8'));

// Metadata for npm
pkg.name = 'surtgis';
pkg.description = 'High-performance geospatial analysis in the browser via WebAssembly';
pkg.keywords = ['gis','geospatial','terrain','dem','wasm','webassembly','slope','hillshade','hydrology','ndvi'];
pkg.repository = { type: 'git', url: 'https://github.com/franciscoparrao/surtgis' };
pkg.homepage = 'https://franciscoparrao.github.io/surtgis';
pkg.license = 'MIT OR Apache-2.0';

// Include extra files in the npm package
pkg.files = [
  ...pkg.files,
  'surtgis.js',
  'surtgis.d.ts',
  'worker.js',
  'surtgis-worker.js',
  'surtgis-worker.d.ts',
  'README.md'
].filter((v, i, a) => a.indexOf(v) === i);

fs.writeFileSync('$PKG_JSON', JSON.stringify(pkg, null, 2) + '\n');
console.log('Patched ' + '$PKG_JSON');
"
