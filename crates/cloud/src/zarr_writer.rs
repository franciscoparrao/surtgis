//! Minimal Zarr v2 writer — local backing store for [`Cube`]s.
//!
//! Writes a [`Cube<f64>`] (or a single [`Raster<f64>`]) as a Zarr v2
//! directory store following the xarray convention
//! (`_ARRAY_DIMENSIONS`), so the output opens directly in
//! xarray/zarr-python (2.x and 3.x), GDAL ≥ 3.4 and `zarrs`. This is
//! the *write* half of SPEC P1.3; reading (HTTP/Azure, CF decoding)
//! lives in [`crate::zarr_reader`].
//!
//! Layout for a cube with bands `["red", "nir"]`:
//!
//! ```text
//! store.zarr/
//!   .zgroup .zattrs            # group + CRS/transform attrs
//!   time/  .zarray .zattrs 0   # i64 epoch seconds, CF units
//!   y/     .zarray .zattrs 0   # cell-centre northings
//!   x/     .zarray .zattrs 0   # cell-centre eastings
//!   red/   .zarray .zattrs 0.0.0 1.0.0 …   # [time, y, x] f64
//!   nir/   .zarray .zattrs 0.0.0 1.0.0 …
//! ```
//!
//! Deliberately minimal (v1): uncompressed chunks of one full
//! `(y, x)` slab per time step, f64 data, NaN fill. Compression and
//! tuned chunking can come once datacube-rs profiles real ARD loads.

use std::fs;
use std::path::Path;

use surtgis_core::cube::Cube;
use surtgis_core::raster::Raster;

use crate::error::{CloudError, Result};

/// CRS / georeferencing attributes written to the group `.zattrs`.
const EPSG_ATTR: &str = "surtgis:epsg";
const TRANSFORM_ATTR: &str = "surtgis:geotransform";

fn check_var_name(name: &str) -> Result<()> {
    if name.is_empty()
        || name.starts_with('.')
        || name.contains('/')
        || name.contains('\\')
        || matches!(name, "time" | "y" | "x")
    {
        return Err(CloudError::Zarr(format!(
            "zarr write: invalid variable name '{}'",
            name
        )));
    }
    Ok(())
}

fn write_json(path: &Path, value: &serde_json::Value) -> Result<()> {
    fs::write(path, serde_json::to_string_pretty(value).unwrap())
        .map_err(|e| CloudError::Zarr(format!("zarr write: {}: {}", path.display(), e)))?;
    Ok(())
}

fn zarray_meta(shape: &[usize], chunks: &[usize], dtype: &str) -> serde_json::Value {
    serde_json::json!({
        "zarr_format": 2,
        "shape": shape,
        "chunks": chunks,
        "dtype": dtype,
        "compressor": null,
        "fill_value": if dtype == "<f8" { serde_json::json!("NaN") } else { serde_json::json!(0) },
        "order": "C",
        "filters": null,
    })
}

fn write_array_f64(
    dir: &Path,
    name: &str,
    shape: &[usize],
    chunks: &[usize],
    dims: &[&str],
    extra_attrs: serde_json::Value,
    chunk_writer: impl Fn(usize) -> Vec<f64>,
    n_chunks0: usize,
) -> Result<()> {
    let arr_dir = dir.join(name);
    fs::create_dir_all(&arr_dir)
        .map_err(|e| CloudError::Zarr(format!("zarr write: mkdir {}: {}", name, e)))?;
    write_json(&arr_dir.join(".zarray"), &zarray_meta(shape, chunks, "<f8"))?;

    let mut attrs = serde_json::json!({ "_ARRAY_DIMENSIONS": dims });
    if let serde_json::Value::Object(extra) = extra_attrs {
        for (k, v) in extra {
            attrs[k] = v;
        }
    }
    write_json(&arr_dir.join(".zattrs"), &attrs)?;

    for c in 0..n_chunks0 {
        let values = chunk_writer(c);
        let mut bytes = Vec::with_capacity(values.len() * 8);
        for v in &values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let key = if shape.len() == 1 {
            c.to_string()
        } else {
            // one chunk spans the full trailing dimensions
            let mut k = c.to_string();
            for _ in 1..shape.len() {
                k.push_str(".0");
            }
            k
        };
        fs::write(arr_dir.join(&key), bytes)
            .map_err(|e| CloudError::Zarr(format!("zarr write: chunk {}: {}", key, e)))?;
    }
    Ok(())
}

fn write_time_array(dir: &Path, times: &[i64]) -> Result<()> {
    let arr_dir = dir.join("time");
    fs::create_dir_all(&arr_dir)
        .map_err(|e| CloudError::Zarr(format!("zarr write: mkdir time: {}", e)))?;
    write_json(
        &arr_dir.join(".zarray"),
        &zarray_meta(&[times.len()], &[times.len()], "<i8"),
    )?;
    write_json(
        &arr_dir.join(".zattrs"),
        &serde_json::json!({
            "_ARRAY_DIMENSIONS": ["time"],
            "units": "seconds since 1970-01-01T00:00:00Z",
            "calendar": "proleptic_gregorian",
        }),
    )?;
    let mut bytes = Vec::with_capacity(times.len() * 8);
    for t in times {
        bytes.extend_from_slice(&t.to_le_bytes());
    }
    fs::write(arr_dir.join("0"), bytes)
        .map_err(|e| CloudError::Zarr(format!("zarr write: time chunk: {}", e)))?;
    Ok(())
}

fn group_attrs(epsg: Option<u32>, transform: &surtgis_core::GeoTransform) -> serde_json::Value {
    let mut attrs = serde_json::json!({
        TRANSFORM_ATTR: transform.to_gdal(),
    });
    if let Some(code) = epsg {
        attrs[EPSG_ATTR] = serde_json::json!(code);
    }
    attrs
}

/// Write a [`Cube<f64>`] as a Zarr v2 directory store at `path`.
///
/// One `[time, y, x]` f64 variable per band (chunked one time step
/// per chunk), plus `time` / `y` / `x` coordinate arrays. `path` is
/// created (parents included) and must not already contain a store.
/// CRS and transform are stored as `surtgis:epsg` /
/// `surtgis:geotransform` group attributes.
pub fn write_cube_zarr<P: AsRef<Path>>(cube: &Cube<f64>, path: P) -> Result<()> {
    let dir = path.as_ref();
    if dir.join(".zgroup").exists() {
        return Err(CloudError::Zarr(format!(
            "zarr write: store already exists at {}",
            dir.display()
        )));
    }
    for band in cube.bands() {
        check_var_name(band)?;
    }
    fs::create_dir_all(dir)
        .map_err(|e| CloudError::Zarr(format!("zarr write: mkdir store: {}", e)))?;

    write_json(
        &dir.join(".zgroup"),
        &serde_json::json!({ "zarr_format": 2 }),
    )?;
    let epsg = cube.slice(0, 0).unwrap().crs().and_then(|c| c.epsg());
    write_json(&dir.join(".zattrs"), &group_attrs(epsg, cube.transform()))?;

    let (rows, cols) = cube.shape();
    write_time_array(dir, cube.times())?;
    write_coord_arrays(dir, cube.transform(), rows, cols)?;

    let n_bands = cube.n_bands();
    for (b, band) in cube.bands().iter().enumerate() {
        write_array_f64(
            dir,
            band,
            &[cube.n_times(), rows, cols],
            &[1, rows, cols],
            &["time", "y", "x"],
            serde_json::json!({}),
            |t| {
                cube.slice(t, b)
                    .map(|s| s.data().iter().copied().collect())
                    .unwrap_or_default()
            },
            cube.n_times(),
        )?;
        let _ = n_bands;
    }
    Ok(())
}

/// Write a single [`Raster<f64>`] as a 2-D `[y, x]` Zarr v2 variable.
pub fn write_raster_zarr<P: AsRef<Path>>(
    raster: &Raster<f64>,
    variable: &str,
    path: P,
) -> Result<()> {
    let dir = path.as_ref();
    if dir.join(".zgroup").exists() {
        return Err(CloudError::Zarr(format!(
            "zarr write: store already exists at {}",
            dir.display()
        )));
    }
    check_var_name(variable)?;
    fs::create_dir_all(dir)
        .map_err(|e| CloudError::Zarr(format!("zarr write: mkdir store: {}", e)))?;

    write_json(
        &dir.join(".zgroup"),
        &serde_json::json!({ "zarr_format": 2 }),
    )?;
    let epsg = raster.crs().and_then(|c| c.epsg());
    write_json(&dir.join(".zattrs"), &group_attrs(epsg, raster.transform()))?;

    let (rows, cols) = raster.shape();
    write_coord_arrays(dir, raster.transform(), rows, cols)?;
    write_array_f64(
        dir,
        variable,
        &[rows, cols],
        &[rows, cols],
        &["y", "x"],
        serde_json::json!({}),
        |_| raster.data().iter().copied().collect(),
        1,
    )?;
    Ok(())
}

fn write_coord_arrays(
    dir: &Path,
    transform: &surtgis_core::GeoTransform,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let y: Vec<f64> = (0..rows).map(|r| transform.pixel_to_geo(0, r).1).collect();
    let x: Vec<f64> = (0..cols).map(|c| transform.pixel_to_geo(c, 0).0).collect();
    write_array_f64(
        dir,
        "y",
        &[rows],
        &[rows],
        &["y"],
        serde_json::json!({}),
        |_| y.clone(),
        1,
    )?;
    write_array_f64(
        dir,
        "x",
        &[cols],
        &[cols],
        &["x"],
        serde_json::json!({}),
        |_| x.clone(),
        1,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::{GeoTransform, Raster};
    use zarrs::array::{Array, ArraySubset};

    use zarrs::filesystem::FilesystemStore;

    fn slice(rows: usize, cols: usize, base: f64) -> Raster<f64> {
        let mut r = Raster::new(rows, cols);
        r.set_transform(GeoTransform::new(500_000.0, 6_300_000.0, 30.0, -30.0));
        for row in 0..rows {
            for col in 0..cols {
                r.set(row, col, base + (row * cols + col) as f64).unwrap();
            }
        }
        r
    }

    /// Round-trip through an independent implementation: our v2
    /// writer → zarrs reader.
    #[test]
    fn cube_roundtrip_via_zarrs() {
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("cube.zarr");

        let cube = Cube::from_slices(
            vec![0, 86_400, 172_800],
            vec!["ndvi".into()],
            vec![slice(8, 6, 0.0), slice(8, 6, 100.0), slice(8, 6, 200.0)],
        )
        .unwrap();
        write_cube_zarr(&cube, &store_path).unwrap();

        let store = std::sync::Arc::new(FilesystemStore::new(&store_path).unwrap());
        let array = Array::open(store.clone(), "/ndvi").unwrap();
        assert_eq!(array.shape(), &[3, 8, 6]);

        let data: Vec<f64> = array
            .retrieve_array_subset(&ArraySubset::new_with_shape(vec![3, 8, 6]))
            .unwrap();
        // value(t, r, c) = t*100 + r*cols + c; flat index = t*48 + r*6 + c
        assert_eq!(data[0], 0.0);
        assert_eq!(data[48 + 2 * 6 + 3], 100.0 + (2 * 6 + 3) as f64);
        assert_eq!(data[2 * 48 + 7 * 6 + 5], 200.0 + (7 * 6 + 5) as f64);

        // Coordinates
        let time = Array::open(store.clone(), "/time").unwrap();
        let t: Vec<i64> = time
            .retrieve_array_subset(&ArraySubset::new_with_shape(vec![3]))
            .unwrap();
        assert_eq!(t, vec![0, 86_400, 172_800]);

        let x = Array::open(store, "/x").unwrap();
        let xs: Vec<f64> = x
            .retrieve_array_subset(&ArraySubset::new_with_shape(vec![6]))
            .unwrap();
        assert_eq!(xs[0], 500_015.0); // first cell centre
    }

    #[test]
    fn raster_roundtrip_via_zarrs() {
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("dem.zarr");
        let mut r = slice(5, 4, 0.0);
        r.set(2, 2, f64::NAN).unwrap();
        write_raster_zarr(&r, "dem", &store_path).unwrap();

        let store = std::sync::Arc::new(FilesystemStore::new(&store_path).unwrap());
        let array = Array::open(store, "/dem").unwrap();
        let data: Vec<f64> = array
            .retrieve_array_subset(&ArraySubset::new_with_shape(vec![5, 4]))
            .unwrap();
        assert_eq!(data[1 * 4 + 3], 7.0);
        assert!(data[2 * 4 + 2].is_nan());
    }

    #[test]
    fn rejects_existing_store_and_bad_names() {
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("s.zarr");
        let r = slice(3, 3, 0.0);
        write_raster_zarr(&r, "dem", &store_path).unwrap();
        assert!(write_raster_zarr(&r, "dem", &store_path).is_err());

        let other = dir.path().join("s2.zarr");
        assert!(write_raster_zarr(&r, "time", &other).is_err());
        assert!(write_raster_zarr(&r, "a/b", &other).is_err());
    }
}
