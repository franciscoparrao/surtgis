//! Shared latitude/longitude grid helpers for the gridded cube readers
//! (Zarr, NetCDF).
//!
//! A latitude axis lives in two index spaces at once:
//!
//! - **ascending space** — indices into a reader's `lat_coords`, which the
//!   constructors always sort ascending (south → north);
//! - **raw space** — indices into the on-disk array, whose latitude axis
//!   may be stored descending (north → south, e.g. ERA5).
//!
//! Conflating the two spaces mirrors the output georeferencing to the
//! opposite hemisphere (docs/bug-reports/BUG_ZARR_CLIMATE_LAT_MIRROR.md):
//! a range produced by [`lat_range_ascending`] must only ever index
//! `lat_coords`, and must be mapped through [`lat_range_raw`] before
//! indexing the raw array.

use ndarray::Array2;
use surtgis_core::raster::GeoTransform;

/// Half-open latitude index range `(start, end)` in ascending space
/// (indices into `lat_coords`, sorted south → north). Returns `None`
/// when the bbox selects no rows.
pub(crate) fn lat_range_ascending(
    lat_coords: &[f64],
    min_y: f64,
    max_y: f64,
) -> Option<(usize, usize)> {
    let start = find_nearest(lat_coords, min_y);
    let end = (find_nearest(lat_coords, max_y) + 1).min(lat_coords.len());
    if start >= end {
        None
    } else {
        Some((start, end))
    }
}

/// Map an ascending-space range onto the raw (on-disk) latitude axis.
///
/// For a descending axis, ascending index `i` addresses raw row
/// `n - 1 - i`, so the half-open range flips to `(n - end, n - start)`.
/// For an ascending axis the spaces coincide.
pub(crate) fn lat_range_raw(
    (start, end): (usize, usize),
    n: usize,
    lat_descending: bool,
) -> (usize, usize) {
    if lat_descending {
        (n - end, n - start)
    } else {
        (start, end)
    }
}

/// Whether rows read from the raw array need a vertical flip to come out
/// north-up (row 0 = northernmost), matching the negative pixel height
/// that [`build_geotransform`] always produces.
///
/// A descending (N→S) source is already north-up in raw index order; an
/// ascending source arrives south-up and must be flipped.
pub(crate) fn needs_north_up_flip(lat_descending: bool) -> bool {
    !lat_descending
}

/// Build a north-up GeoTransform from **ascending** lat and coordinate
/// arrays of pixel centers (origin at the north-west corner, negative
/// pixel height).
pub(crate) fn build_geotransform(lat: &[f64], lon: &[f64]) -> GeoTransform {
    if lat.len() < 2 || lon.len() < 2 {
        return GeoTransform::new(
            lon.first().copied().unwrap_or(0.0),
            lat.last().copied().unwrap_or(0.0),
            1.0,
            -1.0,
        );
    }

    let pixel_width = (lon[lon.len() - 1] - lon[0]) / (lon.len() - 1) as f64;
    let lat_step = (lat[lat.len() - 1] - lat[0]) / (lat.len() - 1) as f64;

    // Origin: top-left corner (half pixel offset from first center)
    let origin_x = lon[0] - pixel_width / 2.0;
    let origin_y = lat[lat.len() - 1] + lat_step / 2.0;

    GeoTransform::new(origin_x, origin_y, pixel_width, -lat_step)
}

/// Nearest index in a sorted ascending coordinate array.
pub(crate) fn find_nearest(coords: &[f64], value: f64) -> usize {
    match coords.binary_search_by(|c| c.partial_cmp(&value).unwrap()) {
        Ok(i) => i,
        Err(i) => {
            if i == 0 {
                0
            } else if i >= coords.len() {
                coords.len() - 1
            } else if (coords[i] - value).abs() < (coords[i - 1] - value).abs() {
                i
            } else {
                i - 1
            }
        }
    }
}

/// Reverse row order (used to turn south-up data north-up).
pub(crate) fn flip_rows(data: Array2<f64>) -> Array2<f64> {
    let nrows = data.nrows();
    let ncols = data.ncols();
    let mut flipped = Array2::zeros((nrows, ncols));
    for r in 0..nrows {
        flipped.row_mut(r).assign(&data.row(nrows - 1 - r));
    }
    flipped
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ERA5-style latitude axis: 90, 89.75, ..., -90 (descending, 721 rows).
    fn era5_raw_lats() -> Vec<f64> {
        (0..721).map(|i| 90.0 - 0.25 * i as f64).collect()
    }

    fn ascending(raw: &[f64]) -> Vec<f64> {
        raw.iter().rev().copied().collect()
    }

    /// Regression for BUG_ZARR_CLIMATE_LAT_MIRROR: a southern-hemisphere
    /// bbox (Chile, -34.6..-34.2) must produce a GeoTransform in the
    /// southern hemisphere. The pre-fix code sliced `lat_coords` with
    /// raw-space indices and georeferenced the output at +34.125..+34.625.
    #[test]
    fn south_bbox_stays_in_southern_hemisphere() {
        let lat_coords = ascending(&era5_raw_lats());
        let (start, end) = lat_range_ascending(&lat_coords, -34.6, -34.2).unwrap();

        let sub_lat = &lat_coords[start..end];
        assert!(
            sub_lat.iter().all(|&l| l < 0.0),
            "requested southern latitudes, got {sub_lat:?}"
        );
        assert_eq!(sub_lat, &[-34.5, -34.25]);

        let gt = build_geotransform(sub_lat, &[-71.875, -71.625, -71.375]);
        assert!(
            (gt.origin_y - (-34.125)).abs() < 1e-9,
            "north edge must be -34.125, got {}",
            gt.origin_y
        );
        assert!(gt.pixel_height < 0.0, "output must be north-up");
        let south_edge = gt.origin_y + gt.pixel_height * sub_lat.len() as f64;
        assert!((south_edge - (-34.625)).abs() < 1e-9);
    }

    /// The mirrored request (same |lat|, northern hemisphere) must land in
    /// the northern hemisphere — the bug was an exact equator mirror.
    #[test]
    fn north_bbox_stays_in_northern_hemisphere() {
        let lat_coords = ascending(&era5_raw_lats());
        let (start, end) = lat_range_ascending(&lat_coords, 34.2, 34.6).unwrap();
        let sub_lat = &lat_coords[start..end];
        assert_eq!(sub_lat, &[34.25, 34.5]);
    }

    /// The raw-space range must address the *requested* region of the
    /// on-disk (descending) array: raw values == sub_lat reversed.
    #[test]
    fn raw_range_reads_the_requested_latitudes() {
        let raw = era5_raw_lats();
        let lat_coords = ascending(&raw);
        let range = lat_range_ascending(&lat_coords, -34.6, -34.2).unwrap();

        let (rs, re) = lat_range_raw(range, lat_coords.len(), true);
        let raw_lats: Vec<f64> = raw[rs..re].to_vec();
        assert_eq!(
            raw_lats,
            vec![-34.25, -34.5],
            "N→S order of the requested rows"
        );

        // Ascending storage: both spaces coincide.
        let (as_, ae) = lat_range_raw(range, lat_coords.len(), false);
        assert_eq!((as_, ae), range);
    }

    /// Rows must come out north-up (row 0 = northernmost) for BOTH
    /// storage orders, matching the negative pixel height of the
    /// GeoTransform. Simulates read_bbox: each cell's value is the
    /// latitude it was read from.
    #[test]
    fn data_rows_are_north_up_for_both_storage_orders() {
        let raw_desc = era5_raw_lats();
        let lat_coords = ascending(&raw_desc);
        let range = lat_range_ascending(&lat_coords, -34.6, -34.2).unwrap();
        let northernmost = lat_coords[range.1 - 1];

        for (lat_descending, raw_axis) in [(true, raw_desc.clone()), (false, lat_coords.clone())] {
            let (rs, re) = lat_range_raw(range, lat_coords.len(), lat_descending);
            // Raw read: rows in on-disk index order, value = source latitude.
            let rows: Vec<f64> = raw_axis[rs..re].to_vec();
            let data = Array2::from_shape_fn((rows.len(), 2), |(r, _)| rows[r]);
            let data = if needs_north_up_flip(lat_descending) {
                flip_rows(data)
            } else {
                data
            };
            assert_eq!(
                data[[0, 0]],
                northernmost,
                "row 0 must be the northernmost latitude (lat_descending={lat_descending})"
            );
        }
    }

    #[test]
    fn empty_selection_is_none() {
        // Bbox entirely north of the grid on a 2-row axis collapses to a
        // single nearest row; a truly empty range needs start >= end.
        let lat_coords = vec![-1.0, 0.0, 1.0];
        assert!(lat_range_ascending(&lat_coords, 0.9, 1.1).is_some());
        // Degenerate: max below min selects nothing once indices cross.
        assert!(lat_range_ascending(&lat_coords, 1.1, -1.1).is_none());
    }
}
