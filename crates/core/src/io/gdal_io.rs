//! GeoTIFF reading and writing using GDAL

use crate::crs::CRS;
use crate::error::{Error, Result};
use crate::raster::{GeoTransform, Raster, RasterElement};
use gdal::raster::{GdalDataType, GdalType, RasterBand};
use gdal::spatial_ref::SpatialRef;
use gdal::{Dataset, DriverManager};
use std::path::Path;

/// Options for writing GeoTIFF files
#[derive(Debug, Clone)]
pub struct GeoTiffOptions {
    /// Compression type: "DEFLATE", "LZW", "ZSTD", "NONE"
    pub compression: String,
    /// Tile size for tiled TIFFs (0 for strips)
    pub tile_size: usize,
    /// Create Cloud-Optimized GeoTIFF
    pub cog: bool,
    /// BigTIFF for files > 4GB
    pub bigtiff: bool,
}

impl Default for GeoTiffOptions {
    fn default() -> Self {
        Self {
            compression: "DEFLATE".to_string(),
            tile_size: 256,
            cog: false,
            bigtiff: false,
        }
    }
}

/// Read a GeoTIFF file into a Raster
///
/// # Arguments
/// * `path` - Path to the GeoTIFF file
/// * `band` - Band number (1-indexed), defaults to 1
///
/// # Example
/// ```ignore
/// let raster: Raster<f32> = read_geotiff("dem.tif", None)?;
/// ```
pub fn read_geotiff<T, P>(path: P, band: Option<usize>) -> Result<Raster<T>>
where
    T: RasterElement + GdalType,
    P: AsRef<Path>,
{
    let dataset = Dataset::open(path.as_ref())?;
    let band_idx = band.unwrap_or(1);
    let rasterband = dataset.rasterband(band_idx)?;

    let (cols, rows) = dataset.raster_size();

    // Read data
    let buffer = rasterband.read_as::<T>((0, 0), (cols, rows), (cols, rows), None)?;

    let mut raster = Raster::from_vec(buffer.data().to_vec(), rows, cols)?;

    // Set geotransform
    if let Ok(gt) = dataset.geo_transform() {
        raster.set_transform(GeoTransform::from_gdal(gt));
    }

    // Set CRS
    if let Ok(srs) = dataset.spatial_ref() {
        if let Ok(wkt) = srs.to_wkt() {
            let mut crs = CRS::from_wkt(wkt);
            // Try to get EPSG code
            if let Ok(code) = srs.auth_code() {
                // Update with EPSG if available
                crs = CRS::from_epsg(code as u32);
            }
            raster.set_crs(Some(crs));
        }
    }

    // Set nodata
    if let Ok(nodata) = rasterband.no_data_value() {
        if let Some(nd) = num_traits::cast(nodata) {
            raster.set_nodata(Some(nd));
        }
    }

    Ok(raster)
}

/// Write a Raster to a GeoTIFF file
///
/// # Arguments
/// * `raster` - The raster to write
/// * `path` - Output file path
/// * `options` - Optional GeoTIFF options
pub fn write_geotiff<T, P>(
    raster: &Raster<T>,
    path: P,
    options: Option<GeoTiffOptions>,
) -> Result<()>
where
    T: RasterElement + GdalType,
    P: AsRef<Path>,
{
    let opts = options.unwrap_or_default();
    let driver = DriverManager::get_driver_by_name("GTiff")?;

    let (rows, cols) = raster.shape();

    // Build creation options
    let mut create_options = vec![
        format!("COMPRESS={}", opts.compression),
    ];

    if opts.tile_size > 0 {
        create_options.push("TILED=YES".to_string());
        create_options.push(format!("BLOCKXSIZE={}", opts.tile_size));
        create_options.push(format!("BLOCKYSIZE={}", opts.tile_size));
    }

    if opts.bigtiff {
        create_options.push("BIGTIFF=YES".to_string());
    }

    let create_options_refs: Vec<&str> = create_options.iter().map(|s| s.as_str()).collect();

    // Determine GDAL data type
    let gdal_type = T::gdal_ordinal();

    let mut dataset = driver.create_with_band_type_with_options::<T, _>(
        path.as_ref(),
        cols as isize,
        rows as isize,
        1,
        &create_options_refs,
    )?;

    // Set geotransform
    dataset.set_geo_transform(&raster.transform().to_gdal())?;

    // Set CRS
    if let Some(crs) = raster.crs() {
        if let Some(epsg) = crs.epsg() {
            let srs = SpatialRef::from_epsg(epsg)?;
            dataset.set_spatial_ref(&srs)?;
        } else if let Some(wkt) = crs.wkt() {
            let srs = SpatialRef::from_wkt(wkt)?;
            dataset.set_spatial_ref(&srs)?;
        }
    }

    // Write data
    let mut band = dataset.rasterband(1)?;

    // Set nodata
    if let Some(nodata) = raster.nodata() {
        if let Some(nd) = num_traits::cast(nodata) {
            band.set_no_data_value(Some(nd))?;
        }
    }

    // Write the raster data
    let data: Vec<T> = raster.data().iter().copied().collect();
    band.write((0, 0), (cols, rows), &data)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_write_read_roundtrip() {
        let mut raster: Raster<f32> = Raster::new(100, 100);
        raster.set_transform(GeoTransform::new(0.0, 100.0, 1.0, -1.0));
        raster.set_crs(Some(CRS::from_epsg(4326)));
        raster.set_nodata(Some(-9999.0));

        // Fill with some data
        for i in 0..100 {
            for j in 0..100 {
                raster.set(i, j, (i * 100 + j) as f32).unwrap();
            }
        }

        let tmp = NamedTempFile::with_suffix(".tif").unwrap();
        write_geotiff(&raster, tmp.path(), None).unwrap();

        let loaded: Raster<f32> = read_geotiff(tmp.path(), None).unwrap();

        assert_eq!(loaded.shape(), raster.shape());
        assert_eq!(loaded.get(50, 50).unwrap(), raster.get(50, 50).unwrap());
    }
}
