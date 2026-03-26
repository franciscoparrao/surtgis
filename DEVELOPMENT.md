# SurtGIS Demo - Development Guide

## Local Testing

### Option 1: Python HTTP Server (Easiest)

```bash
cd surtgis-demo
python3 -m http.server 8765
```

Then open: **http://localhost:8765**

### Option 2: Node.js HTTP Server

```bash
npm install -g http-server
cd surtgis-demo
http-server -p 8765
```

### Option 3: Live Server (VS Code Extension)

Install "Live Server" extension, right-click `index.html` → "Open with Live Server"

---

## Rebuild WASM

If you modify `crates/wasm/src/lib.rs`:

```bash
cd crates/wasm
wasm-pack build --target web --release -- --no-default-features
cp pkg/* ../../surtgis-demo/wasm/
```

**Note**: The `.wasm` binary is >500KB and should NOT be committed to git. It's built fresh during deployment.

---

## Files

| File | Purpose |
|------|---------|
| `index.html` | UI structure + Mapbox GL JS |
| `styles.css` | Tailwind CSS + custom styles |
| `app.js` | Main JavaScript logic, WASM orchestration |
| `wasm/` | Compiled WebAssembly bindings |
| `data/` | Sample GeoTIFF DEMs (optional) |

---

## Debugging

1. **Open browser DevTools**: F12 or Right-click → "Inspect"
2. **Console**: Check for JavaScript errors
3. **Network**: Verify `.wasm` file loads (should be ~600KB)
4. **Performance**: Use DevTools Performance tab to profile computation

## Deployment

To deploy to GitHub Pages:

```bash
# Build WASM (if needed)
cd crates/wasm
wasm-pack build --target web --release -- --no-default-features
cp pkg/* ../../surtgis-demo/wasm/

# Push to gh-pages branch
cd ../../
git add surtgis-demo/
git commit -m "Update demo (rebuild WASM)"
git subtree push --prefix surtgis-demo origin gh-pages
```

Live at: **https://franciscoparrao.github.io/surtgis/demo/**

---

## Adding New Algorithms

1. Add function to `crates/wasm/src/lib.rs` with `#[wasm_bindgen]`
2. Export from WASM via `wasm-pack build`
3. Add to `app.js` switch statement
4. Add to HTML `<select>` dropdown
5. Test locally
6. Rebuild and deploy

Example:

```rust
// src/lib.rs
#[wasm_bindgen]
pub fn new_algorithm(tiff_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    let dem = read_geotiff_from_buffer::<f64>(tiff_bytes, None)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let result = surtgis_algorithms::module::new_algo(&dem)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    write_geotiff_to_buffer(&result, None)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}
```

```javascript
// app.js
case 'new_algo':
    result = new_algorithm(currentDem);
    break;
```

```html
<!-- index.html -->
<option value="new_algo">New Algorithm</option>
```

---

## Browser Compatibility

Tested on:
- ✅ Chrome 120+
- ✅ Firefox 121+
- ✅ Safari 17+
- ✅ Edge 120+

Requirements:
- WebAssembly support
- ES6+ JavaScript
- Fetch API
- GeoTIFF.js library (loaded from CDN)

---

## Performance Optimization

Current bottlenecks:
1. WASM function calls (small overhead)
2. GeoTIFF parsing/writing (in-memory)
3. Computation (single-threaded)

Optimization opportunities:
- [ ] Streaming input/output for large files
- [ ] Worker threads for background computation
- [ ] Progressive visualization (tile-based)
- [ ] Cached intermediate results

---

## Testing

Manual test checklist:

- [ ] Page loads without console errors
- [ ] WASM binary loads successfully
- [ ] File upload works with .tif files
- [ ] Slope computation completes in <2s (512x512)
- [ ] Results visualize on map
- [ ] Download produces valid GeoTIFF
- [ ] All 10 algorithms work
- [ ] Responsive design on mobile
- [ ] Links in README work

---

## Troubleshooting Common Issues

| Issue | Solution |
|-------|----------|
| "WASM not defined" | Check `surtgis_wasm.js` loaded in Network tab |
| CORS errors | Ensure serving via HTTP (not `file://`) |
| Slow computation | Try smaller DEM (512×512) |
| Map not showing | Check Mapbox GL JS CDN link |
| "Invalid GeoTIFF" | Ensure file has numeric data + valid georeferencing |

---

Last updated: 2026-03-26
