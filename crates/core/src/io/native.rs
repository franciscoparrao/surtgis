//! Native GeoTIFF reading/writing (without GDAL dependency)
//!
//! Uses the `tiff` crate for basic TIFF I/O.
//! For full GeoTIFF support (projections, advanced types), enable the `gdal` feature.

use crate::error::{Error, Result};
use crate::raster::{GeoTransform, Raster, RasterElement};
use std::fs::File;
use std::path::Path;
use tiff::decoder::{Decoder, DecodingResult};
use tiff::encoder::colortype::Gray32Float;
use tiff::encoder::TiffEncoder;
use tiff::tags::Tag;

/// Options for writing GeoTIFF files
#[derive(Debug, Clone)]
pub struct GeoTiffOptions {
    /// Compression (not fully supported in native mode)
    pub compression: String,
}

impl Default for GeoTiffOptions {
    fn default() -> Self {
        Self {
            compression: "NONE".to_string(),
        }
    }
}

/// Read a GeoTIFF file into a Raster
///
/// Native reader with limited GeoTIFF metadata support.
/// For full support, enable the `gdal` feature.
pub fn read_geotiff<T, P>(path: P, _band: Option<usize>) -> Result<Raster<T>>
where
    T: RasterElement,
    P: AsRef<Path>,
{
    let file = File::open(path.as_ref())?;
    let mut decoder = Decoder::new(file)
        .map_err(|e| Error::Other(format!("TIFF decode error: {}", e)))?;

    let (width, height) = decoder.dimensions()
        .map_err(|e| Error::Other(format!("Cannot read dimensions: {}", e)))?;

    let rows = height as usize;
    let cols = width as usize;

    // Read image data
    let result = decoder.read_image()
        .map_err(|e| Error::Other(format!("Cannot read image data: {}", e)))?;

    let data: Vec<T> = match result {
        DecodingResult::F32(buf) => buf
            .iter()
            .map(|&v| num_traits::cast(v).unwrap_or(T::default_nodata()))
            .collect(),
        DecodingResult::F64(buf) => buf
            .iter()
            .map(|&v| num_traits::cast(v).unwrap_or(T::default_nodata()))
            .collect(),
        DecodingResult::U8(buf) => buf
            .iter()
            .map(|&v| num_traits::cast(v).unwrap_or(T::default_nodata()))
            .collect(),
        DecodingResult::U16(buf) => buf
            .iter()
            .map(|&v| num_traits::cast(v).unwrap_or(T::default_nodata()))
            .collect(),
        DecodingResult::U32(buf) => buf
            .iter()
            .map(|&v| num_traits::cast(v).unwrap_or(T::default_nodata()))
            .collect(),
        DecodingResult::I8(buf) => buf
            .iter()
            .map(|&v| num_traits::cast(v).unwrap_or(T::default_nodata()))
            .collect(),
        DecodingResult::I16(buf) => buf
            .iter()
            .map(|&v| num_traits::cast(v).unwrap_or(T::default_nodata()))
            .collect(),
        DecodingResult::I32(buf) => buf
            .iter()
            .map(|&v| num_traits::cast(v).unwrap_or(T::default_nodata()))
            .collect(),
        _ => return Err(Error::UnsupportedDataType("Unsupported TIFF pixel format".to_string())),
    };

    if data.len() != rows * cols {
        return Err(Error::InvalidDimensions {
            width: cols,
            height: rows,
        });
    }

    let mut raster = Raster::from_vec(data, rows, cols)?;

    // Try to read GeoTIFF tags (ModelTiepointTag + ModelPixelScaleTag)
    if let Ok(transform) = read_geotransform(&mut decoder) {
        raster.set_transform(transform);
    }

    Ok(raster)
}

/// Attempt to read GeoTransform from TIFF tags
fn read_geotransform<R: std::io::Read + std::io::Seek>(
    decoder: &mut Decoder<R>,
) -> Result<GeoTransform> {
    // ModelPixelScaleTag = 33550
    // ModelTiepointTag = 33922
    // ModelTransformationTag = 34264 (alternative)

    // Try ModelPixelScaleTag + ModelTiepointTag first
    let scale_tag = Tag::Unknown(33550);
    let tiepoint_tag = Tag::Unknown(33922);

    let scale = decoder
        .get_tag_f64_vec(scale_tag)
        .map_err(|_| Error::Other("No pixel scale tag".into()))?;

    let tiepoint = decoder
        .get_tag_f64_vec(tiepoint_tag)
        .map_err(|_| Error::Other("No tiepoint tag".into()))?;

    if scale.len() >= 2 && tiepoint.len() >= 6 {
        // tiepoint: [I, J, K, X, Y, Z]
        // scale: [ScaleX, ScaleY, ScaleZ]
        let origin_x = tiepoint[3] - tiepoint[0] * scale[0];
        let origin_y = tiepoint[4] + tiepoint[1] * scale[1];
        let pixel_width = scale[0];
        let pixel_height = -scale[1]; // Negative for north-up

        return Ok(GeoTransform::new(origin_x, origin_y, pixel_width, pixel_height));
    }

    Err(Error::Other("Cannot determine geotransform".into()))
}

/// Write a Raster to a GeoTIFF file
///
/// Native writer with limited GeoTIFF metadata support.
/// Writes as 32-bit float. For full support, enable the `gdal` feature.
pub fn write_geotiff<T, P>(
    raster: &Raster<T>,
    path: P,
    _options: Option<GeoTiffOptions>,
) -> Result<()>
where
    T: RasterElement,
    P: AsRef<Path>,
{
    let file = File::create(path.as_ref())?;
    let mut encoder = TiffEncoder::new(file)
        .map_err(|e| Error::Other(format!("TIFF encoder error: {}", e)))?;

    let (rows, cols) = raster.shape();

    // Convert data to f32
    let data: Vec<f32> = raster
        .data()
        .iter()
        .map(|&v| num_traits::cast(v).unwrap_or(f32::NAN))
        .collect();

    let mut image = encoder
        .new_image::<Gray32Float>(cols as u32, rows as u32)
        .map_err(|e| Error::Other(format!("Cannot create TIFF image: {}", e)))?;

    // Write GeoTIFF tags
    let gt = raster.transform();

    // ModelPixelScaleTag
    let scale = vec![gt.pixel_width, gt.pixel_height.abs(), 0.0];
    image
        .encoder()
        .write_tag(Tag::Unknown(33550), scale.as_slice())
        .map_err(|e| Error::Other(format!("Cannot write scale tag: {}", e)))?;

    // ModelTiepointTag
    let tiepoint = vec![0.0, 0.0, 0.0, gt.origin_x, gt.origin_y, 0.0];
    image
        .encoder()
        .write_tag(Tag::Unknown(33922), tiepoint.as_slice())
        .map_err(|e| Error::Other(format!("Cannot write tiepoint tag: {}", e)))?;

    image
        .write_data(&data)
        .map_err(|e| Error::Other(format!("Cannot write image data: {}", e)))?;

    Ok(())
}
