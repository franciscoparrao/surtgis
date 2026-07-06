//! Fast building blocks for focal (moving-window) statistics.
//!
//! Brute-force focal statistics cost O(k^2) per cell, where `k = 2*radius + 1`
//! is the window side. This module implements the standard incremental
//! algorithms that remove that quadratic dependency on the window size:
//!
//! - [`SummedAreaTable`] — a 2D prefix-sum (integral image) giving O(1)
//!   box queries for Mean/StdDev/Sum/Count on a *square* window.
//! - [`RowPrefixTable`] + [`circular_row_aggregate`] — the same idea
//!   decomposed per-row for a *circular* window (not separable in 2D, but
//!   still O(radius) per cell instead of O(radius^2)).
//! - [`sliding_extreme_1d`] — the van Herk (1992) / Gil-Werman (1993)
//!   sliding min/max filter, O(1) amortized per element. Applied first
//!   along rows then along columns (min/max is separable) it gives an
//!   O(1)-amortized Min/Max/Range for a *square* window.
//! - [`huang_focal`] — a Huang (1981) sliding-window median/percentile
//!   filter, O(radius * log D) per cell (`D` = number of distinct valid
//!   raster values) vs. O(k^2 log k) for a sort-per-cell brute force.
//!   Works for both square and circular windows because it only needs to
//!   know which cells enter/exit the window as it slides one column at a
//!   time. Unlike a classic Huang histogram (which quantizes values into
//!   a small fixed number of bins, trading accuracy for O(1) bin lookup),
//!   this implementation coordinate-compresses the raster's own distinct
//!   values and tracks window membership in a [`Fenwick`] tree, so
//!   results are the exact order statistic — no quantization error.
//!
//! All functions treat raster nodata (via [`Raster::is_nodata_at`]) as
//! "not present" exactly like the brute-force path in `focal.rs`, and
//! reproduce its edge behaviour (only real in-bounds neighbors count,
//! windows are never padded with invented values).

use crate::maybe_rayon::par_map_rows;
use ndarray::Array2;
use surtgis_core::raster::Raster;

// ---------------------------------------------------------------------
// Summed-area table (square window): Mean / StdDev / Sum / Count
// ---------------------------------------------------------------------

/// 2D summed-area table (integral image) over a raster's valid cells.
///
/// Exposes O(1) box queries for the running sum, the running count of
/// valid cells, and the running sum of squared deviations from a fixed
/// reference value (`ref_mean`, typically the raster's global mean).
///
/// Centering the sum-of-squares on `ref_mean` before accumulating avoids
/// catastrophic cancellation in `var = E[(x-ref)^2] - E[x-ref]^2` when
/// values sit far from zero (e.g. elevations around 1000 m) and the true
/// local variance is small.
pub(crate) struct SummedAreaTable {
    rows: usize,
    cols: usize,
    /// Prefix sum of raw values, shape (rows+1) x (cols+1), row-major.
    sum: Vec<f64>,
    /// Prefix sum of `(value - ref_mean)`.
    sum_centered: Vec<f64>,
    /// Prefix sum of `(value - ref_mean)^2`.
    sumsq_centered: Vec<f64>,
    /// Prefix count of valid (non-nodata) cells.
    count: Vec<f64>,
}

impl SummedAreaTable {
    /// Build the table for the whole raster. `ref_mean` should be the
    /// raster's global mean (used only to stabilize the variance
    /// computation; any finite value keeps results correct).
    pub(crate) fn build(raster: &Raster<f64>, ref_mean: f64) -> Self {
        let (rows, cols) = raster.shape();
        let w = cols + 1;
        let mut sum = vec![0.0; (rows + 1) * w];
        let mut sum_centered = vec![0.0; (rows + 1) * w];
        let mut sumsq_centered = vec![0.0; (rows + 1) * w];
        let mut count = vec![0.0; (rows + 1) * w];

        for r in 0..rows {
            let row_off = (r + 1) * w;
            let prev_off = r * w;
            let mut run_sum = 0.0;
            let mut run_sum_c = 0.0;
            let mut run_sumsq_c = 0.0;
            let mut run_count = 0.0;

            for c in 0..cols {
                let valid = !raster.is_nodata_at(r, c).unwrap_or(true);
                if valid {
                    let v = unsafe { raster.get_unchecked(r, c) };
                    let vc = v - ref_mean;
                    run_sum += v;
                    run_sum_c += vc;
                    run_sumsq_c += vc * vc;
                    run_count += 1.0;
                }
                // prefix(r, c) = prefix(r-1, c) + row_r partial sum up to c
                sum[row_off + c + 1] = sum[prev_off + c + 1] + run_sum;
                sum_centered[row_off + c + 1] = sum_centered[prev_off + c + 1] + run_sum_c;
                sumsq_centered[row_off + c + 1] = sumsq_centered[prev_off + c + 1] + run_sumsq_c;
                count[row_off + c + 1] = count[prev_off + c + 1] + run_count;
            }
        }

        Self {
            rows,
            cols,
            sum,
            sum_centered,
            sumsq_centered,
            count,
        }
    }

    #[inline]
    fn box_query(&self, table: &[f64], r0: usize, c0: usize, r1: usize, c1: usize) -> f64 {
        let w = self.cols + 1;
        table[(r1 + 1) * w + (c1 + 1)] - table[r0 * w + (c1 + 1)] - table[(r1 + 1) * w + c0]
            + table[r0 * w + c0]
    }

    /// Inclusive box sum of raw values over `[r0,c0]..[r1,c1]`.
    pub(crate) fn box_sum(&self, r0: usize, c0: usize, r1: usize, c1: usize) -> f64 {
        self.box_query(&self.sum, r0, c0, r1, c1)
    }

    /// Inclusive box sum of `(value - ref_mean)`.
    pub(crate) fn box_sum_centered(&self, r0: usize, c0: usize, r1: usize, c1: usize) -> f64 {
        self.box_query(&self.sum_centered, r0, c0, r1, c1)
    }

    /// Inclusive box sum of `(value - ref_mean)^2`.
    pub(crate) fn box_sumsq_centered(&self, r0: usize, c0: usize, r1: usize, c1: usize) -> f64 {
        self.box_query(&self.sumsq_centered, r0, c0, r1, c1)
    }

    /// Inclusive box count of valid (non-nodata) cells.
    pub(crate) fn box_count(&self, r0: usize, c0: usize, r1: usize, c1: usize) -> f64 {
        self.box_query(&self.count, r0, c0, r1, c1)
    }

    #[inline]
    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    pub(crate) fn cols(&self) -> usize {
        self.cols
    }
}

/// Compute Mean/StdDev/Sum/Count with a square window via [`SummedAreaTable`].
///
/// `stat_kind` selects which of the four to compute: 0=Mean, 1=StdDev,
/// 2=Sum, 3=Count.
pub(crate) fn square_moment_stat(
    raster: &Raster<f64>,
    radius: usize,
    stat_kind: u8,
) -> Array2<f64> {
    let (rows, cols) = raster.shape();
    let ref_mean = raster.statistics().mean.unwrap_or(0.0);
    let sat = SummedAreaTable::build(raster, ref_mean);

    par_map_rows(rows, cols, |row, out_row| {
        let r0 = row.saturating_sub(radius);
        let r1 = (row + radius).min(sat.rows() - 1);
        for (col, out) in out_row.iter_mut().enumerate() {
            let c0 = col.saturating_sub(radius);
            let c1 = (col + radius).min(sat.cols() - 1);

            let count = sat.box_count(r0, c0, r1, c1);
            if count <= 0.0 {
                *out = f64::NAN;
                continue;
            }

            *out = match stat_kind {
                3 => count,
                2 => sat.box_sum(r0, c0, r1, c1),
                0 => {
                    let sum_c = sat.box_sum_centered(r0, c0, r1, c1);
                    ref_mean + sum_c / count
                }
                1 => {
                    let sum_c = sat.box_sum_centered(r0, c0, r1, c1);
                    let sumsq_c = sat.box_sumsq_centered(r0, c0, r1, c1);
                    let mean_c = sum_c / count;
                    let var = (sumsq_c / count - mean_c * mean_c).max(0.0);
                    var.sqrt()
                }
                _ => unreachable!("stat_kind must be 0..=3"),
            };
        }
    })
}

// ---------------------------------------------------------------------
// Row-prefix table (circular window): Mean / StdDev / Sum / Count
// ---------------------------------------------------------------------

/// Per-row 1D prefix-sum table, used to give a circular window O(radius)
/// per cell instead of O(radius^2): a circular window decomposes into one
/// horizontal run per row offset `dr`, and each run is an O(1) range
/// query into this table.
pub(crate) struct RowPrefixTable {
    cols: usize,
    sum: Vec<f64>,
    sum_centered: Vec<f64>,
    sumsq_centered: Vec<f64>,
    count: Vec<f64>,
}

impl RowPrefixTable {
    pub(crate) fn build(raster: &Raster<f64>, ref_mean: f64) -> Self {
        let (rows, cols) = raster.shape();
        let w = cols + 1;
        let mut sum = vec![0.0; rows * w];
        let mut sum_centered = vec![0.0; rows * w];
        let mut sumsq_centered = vec![0.0; rows * w];
        let mut count = vec![0.0; rows * w];

        for r in 0..rows {
            let off = r * w;
            let mut rs = 0.0;
            let mut rsc = 0.0;
            let mut rssq = 0.0;
            let mut rc = 0.0;
            for c in 0..cols {
                let valid = !raster.is_nodata_at(r, c).unwrap_or(true);
                if valid {
                    let v = unsafe { raster.get_unchecked(r, c) };
                    let vc = v - ref_mean;
                    rs += v;
                    rsc += vc;
                    rssq += vc * vc;
                    rc += 1.0;
                }
                sum[off + c + 1] = rs;
                sum_centered[off + c + 1] = rsc;
                sumsq_centered[off + c + 1] = rssq;
                count[off + c + 1] = rc;
            }
        }

        Self {
            cols,
            sum,
            sum_centered,
            sumsq_centered,
            count,
        }
    }

    /// Inclusive range query `[c0, c1]` within a single row.
    #[inline]
    fn range_query(&self, row: usize, c0: usize, c1: usize) -> (f64, f64, f64, f64) {
        let w = self.cols + 1;
        let off = row * w;
        let s = self.sum[off + c1 + 1] - self.sum[off + c0];
        let sc = self.sum_centered[off + c1 + 1] - self.sum_centered[off + c0];
        let ssq = self.sumsq_centered[off + c1 + 1] - self.sumsq_centered[off + c0];
        let cnt = self.count[off + c1 + 1] - self.count[off + c0];
        (s, sc, ssq, cnt)
    }
}

/// Half-width of a circular window's row band at vertical offset `dr`:
/// the largest `dc` such that `dr^2 + dc^2 <= radius^2`. Returns `None`
/// when `dr` is outside the circle entirely.
#[inline]
pub(crate) fn circular_half_width(radius: usize, dr: isize) -> Option<usize> {
    let r_sq = (radius * radius) as isize;
    let rem = r_sq - dr * dr;
    if rem < 0 {
        None
    } else {
        Some((rem as f64).sqrt().floor() as usize)
    }
}

/// Aggregate `(sum, sum_centered, sumsq_centered, count)` over a circular
/// window centered at `(row, col)`, decomposed row-by-row via
/// [`RowPrefixTable`]. O(radius) per cell.
pub(crate) fn circular_row_aggregate(
    table: &RowPrefixTable,
    row: usize,
    col: usize,
    radius: usize,
    rows: usize,
) -> (f64, f64, f64, f64) {
    let r = radius as isize;
    let mut sum = 0.0;
    let mut sum_c = 0.0;
    let mut sumsq_c = 0.0;
    let mut count = 0.0;

    for dr in -r..=r {
        let nr = row as isize + dr;
        if nr < 0 || nr as usize >= rows {
            continue;
        }
        let Some(hw) = circular_half_width(radius, dr) else {
            continue;
        };
        let hw = hw as isize;
        let c0 = (col as isize - hw).max(0);
        let c1 = (col as isize + hw).min(table.cols as isize - 1);
        if c0 > c1 {
            continue;
        }
        let (s, sc, ssq, cnt) = table.range_query(nr as usize, c0 as usize, c1 as usize);
        sum += s;
        sum_c += sc;
        sumsq_c += ssq;
        count += cnt;
    }

    (sum, sum_c, sumsq_c, count)
}

/// Compute Mean/StdDev/Sum/Count with a circular window via
/// [`RowPrefixTable`]. `stat_kind`: 0=Mean, 1=StdDev, 2=Sum, 3=Count.
pub(crate) fn circular_moment_stat(
    raster: &Raster<f64>,
    radius: usize,
    stat_kind: u8,
) -> Array2<f64> {
    let (rows, cols) = raster.shape();
    let ref_mean = raster.statistics().mean.unwrap_or(0.0);
    let table = RowPrefixTable::build(raster, ref_mean);

    par_map_rows(rows, cols, |row, out_row| {
        for (col, out) in out_row.iter_mut().enumerate() {
            let (_sum, sum_c, sumsq_c, count) =
                circular_row_aggregate(&table, row, col, radius, rows);

            if count <= 0.0 {
                *out = f64::NAN;
                continue;
            }

            *out = match stat_kind {
                3 => count,
                2 => ref_mean * count + sum_c, // = sum
                0 => ref_mean + sum_c / count,
                1 => {
                    let mean_c = sum_c / count;
                    let var = (sumsq_c / count - mean_c * mean_c).max(0.0);
                    var.sqrt()
                }
                _ => unreachable!("stat_kind must be 0..=3"),
            };
        }
    })
}

// ---------------------------------------------------------------------
// van Herk (1992) / Gil-Werman (1993) sliding min/max: Min / Max / Range
// ---------------------------------------------------------------------

/// Sliding-window min or max over a 1D slice, O(n) total instead of
/// O(n*radius).
///
/// `pad` conceptually extends `data` by `radius` cells on each side with
/// a value that can never win the reduction (`+inf` for a min-filter,
/// `-inf` for a max-filter). This reproduces two things simultaneously:
/// (a) the "only real in-bounds neighbors count" edge behaviour of the
/// brute-force window (an out-of-range padded value never becomes the
/// min/max unless the *entire* real window is also excluded), and (b)
/// nodata exclusion, *provided* the caller has already substituted `pad`
/// for nodata cells in `data`.
pub(crate) fn sliding_extreme_1d(data: &[f64], radius: usize, take_min: bool) -> Vec<f64> {
    let len = data.len();
    if len == 0 {
        return Vec::new();
    }
    let pad = if take_min {
        f64::INFINITY
    } else {
        f64::NEG_INFINITY
    };
    let reduce: fn(f64, f64) -> f64 = if take_min { f64::min } else { f64::max };

    let w = 2 * radius + 1;
    let padded_len = len + 2 * radius;
    let n_blocks = padded_len.div_ceil(w);
    let ext_len = n_blocks * w;

    // Extended-array accessor: index i maps to original data[i-radius],
    // or `pad` when that position falls outside [0, len).
    let get = |i: usize| -> f64 {
        if i < radius {
            pad
        } else {
            let j = i - radius;
            if j < len { data[j] } else { pad }
        }
    };

    let mut g = vec![0.0; ext_len]; // prefix reduction within each block
    let mut h = vec![0.0; ext_len]; // suffix reduction within each block

    for b in 0..n_blocks {
        let start = b * w;
        let end = start + w - 1; // inclusive

        g[start] = get(start);
        for i in start + 1..=end {
            g[i] = reduce(g[i - 1], get(i));
        }

        h[end] = get(end);
        let mut i = end;
        while i > start {
            i -= 1;
            h[i] = reduce(h[i + 1], get(i));
        }
    }

    // Output position c (original coords) <-> window [c, c+2*radius] in
    // extended-array coords (since original index j maps to i = j+radius).
    let mut out = Vec::with_capacity(len);
    for c in 0..len {
        let lo = c;
        let hi = c + 2 * radius;
        out.push(reduce(h[lo], g[hi]));
    }
    out
}

/// Compute Min or Max with a square window via two separable 1D HGW
/// passes (rows, then columns). O(1) amortized per cell.
pub(crate) fn hgw_square_2d(raster: &Raster<f64>, radius: usize, take_min: bool) -> Array2<f64> {
    let (rows, cols) = raster.shape();
    let pad = if take_min {
        f64::INFINITY
    } else {
        f64::NEG_INFINITY
    };

    // Horizontal pass.
    let row_pass = par_map_rows(rows, cols, |row, out_row| {
        let mut buf = vec![0.0f64; cols];
        for (c, b) in buf.iter_mut().enumerate() {
            let valid = !raster.is_nodata_at(row, c).unwrap_or(true);
            *b = if valid {
                unsafe { raster.get_unchecked(row, c) }
            } else {
                pad
            };
        }
        let filtered = sliding_extreme_1d(&buf, radius, take_min);
        out_row.copy_from_slice(&filtered);
    });

    // Transpose so the vertical pass can reuse the same 1D routine.
    let mut transposed = vec![0.0f64; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            transposed[c * rows + r] = row_pass[[r, c]];
        }
    }

    let col_pass = par_map_rows(cols, rows, |c, out_col| {
        let slice = &transposed[c * rows..(c + 1) * rows];
        let filtered = sliding_extreme_1d(slice, radius, take_min);
        out_col.copy_from_slice(&filtered);
    });

    // Transpose back to (rows, cols) and convert the sentinel pad value
    // (an entirely-nodata window) to NaN.
    let mut out = Array2::from_elem((rows, cols), f64::NAN);
    for c in 0..cols {
        for r in 0..rows {
            let v = col_pass[[c, r]];
            out[[r, c]] = if v.is_infinite() { f64::NAN } else { v };
        }
    }
    out
}

// ---------------------------------------------------------------------
// Huang (1981) sliding window + exact order statistics: Median / Percentile
// (square + circular)
// ---------------------------------------------------------------------

/// A Fenwick tree (binary indexed tree) over a fixed, coordinate-compressed
/// universe of `n` ranks, supporting O(log n) point updates and O(log n)
/// "find the k-th smallest present element" queries.
///
/// This is the exact-value counterpart to a classic Huang histogram: instead
/// of quantizing values into a small fixed number of bins (lossy), the
/// "bins" here are the raster's own distinct valid values (coordinate
/// compression), so order-statistic queries return an exact raster value,
/// never an interpolated/binned approximation.
struct Fenwick {
    tree: Vec<i64>,
    n: usize,
    log: u32,
}

impl Fenwick {
    fn new(n: usize) -> Self {
        let log = if n == 0 { 0 } else { n.ilog2() };
        Self {
            tree: vec![0i64; n + 1],
            n,
            log,
        }
    }

    /// Add `delta` to the count at 0-indexed rank `i`.
    fn add(&mut self, i: usize, delta: i64) {
        let mut idx = i + 1;
        while idx <= self.n {
            self.tree[idx] += delta;
            idx += idx & idx.wrapping_neg();
        }
    }

    /// Return the 0-indexed rank of the `k`-th smallest present element
    /// (0-indexed `k`, i.e. `k=0` is the minimum). Caller must ensure the
    /// total count of present elements is `> k`.
    fn find_kth(&self, k: usize) -> usize {
        let mut pos = 0usize;
        let mut remaining = (k + 1) as i64;
        let mut bit = 1usize << self.log;
        while bit > 0 {
            let next = pos + bit;
            if next <= self.n && self.tree[next] < remaining {
                pos = next;
                remaining -= self.tree[pos];
            }
            bit >>= 1;
        }
        pos // 0-indexed rank in the coordinate-compressed value array
    }
}

/// Compute Median (`percentile = None`) or a Percentile (`Some(p)`,
/// `p` in `0..=100`) with a square or circular window via a sliding
/// window maintained in a [`Fenwick`] tree over the raster's
/// coordinate-compressed distinct valid values.
///
/// The window's exact membership is tracked incrementally exactly like
/// Huang's (1981) sliding-histogram median filter: at the start of each
/// row the window is rebuilt from scratch (O(radius^2) cells for a
/// square window, O(radius^2) for circular too since the full disc must
/// be scanned once), then as the window slides one column at a time only
/// the entering/exiting cells are updated — O(radius) per shift for a
/// square window (one whole column enters/exits) and O(radius) for a
/// circular window too (one cell per row-offset band enters/exits, per
/// [`circular_half_width`]). Each entering/exiting update costs
/// `O(log D)` on the Fenwick tree (`D` = number of distinct valid raster
/// values), and each cell's result is one `O(log D)` order-statistic
/// query — so the total cost is `O(radius * log D)` per cell, vastly
/// better than the `O(radius^2 log radius)` a brute-force sort-per-cell
/// median would pay, and with **no quantization error**: results are
/// exact raster values, matching a brute-force reference bit-for-bit
/// (mod the usual average-of-two-middle-values for an even-sized
/// window, which is exact floating point arithmetic on two real values).
///
/// Rows are processed **sequentially** (not row-parallel like the other
/// fast paths) because the Fenwick tree is reused across the whole raster
/// and reset between rows via an "undo log" of the (rank, delta) updates
/// applied during that row, rather than being reallocated — reallocating
/// and zeroing a tree sized to the raster's full distinct-value count for
/// every row would itself be O(rows * D), which can be worse than the
/// brute-force baseline for wide/tall rasters. This is a deliberate
/// correctness/parallelism trade-off (see module docs: exactness over
/// raw throughput for this statistic).
pub(crate) fn huang_focal(
    raster: &Raster<f64>,
    radius: usize,
    circular: bool,
    percentile: Option<f64>,
) -> Array2<f64> {
    let (rows, cols) = raster.shape();
    let mut out = Array2::from_elem((rows, cols), f64::NAN);

    // Coordinate-compress all valid raster values into a sorted, deduped
    // list; this is the exact-value analogue of Huang's histogram bins.
    let mut sorted_vals: Vec<f64> = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let valid = !raster.is_nodata_at(r, c).unwrap_or(true);
            if valid {
                sorted_vals.push(unsafe { raster.get_unchecked(r, c) });
            }
        }
    }
    if sorted_vals.is_empty() {
        return out; // all-nodata raster: every window is empty -> NaN
    }
    sorted_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted_vals.dedup();
    let d = sorted_vals.len();

    let rank_of = |v: f64| -> usize {
        sorted_vals
            .binary_search_by(|probe| probe.partial_cmp(&v).unwrap())
            .expect("value must be a raster value present in the global sorted list")
    };

    let r = radius as isize;
    let width_at = |dr: isize| -> Option<isize> {
        if circular {
            circular_half_width(radius, dr).map(|hw| hw as isize)
        } else {
            Some(r)
        }
    };

    let mut fenwick = Fenwick::new(d);
    let mut touched: Vec<(usize, i64)> = Vec::new();

    for row in 0..rows {
        let mut n: i64 = 0;

        for col in 0..cols {
            if col == 0 {
                for dr in -r..=r {
                    let nr = row as isize + dr;
                    if nr < 0 || nr as usize >= rows {
                        continue;
                    }
                    let Some(w) = width_at(dr) else { continue };
                    let c_lo = 0isize.max(-w);
                    let c_hi = w.min(cols as isize - 1);
                    if c_lo > c_hi {
                        continue;
                    }
                    for nc in c_lo..=c_hi {
                        let valid = !raster
                            .is_nodata_at(nr as usize, nc as usize)
                            .unwrap_or(true);
                        if valid {
                            let v = unsafe { raster.get_unchecked(nr as usize, nc as usize) };
                            let rank = rank_of(v);
                            fenwick.add(rank, 1);
                            touched.push((rank, 1));
                            n += 1;
                        }
                    }
                }
            } else {
                for dr in -r..=r {
                    let nr = row as isize + dr;
                    if nr < 0 || nr as usize >= rows {
                        continue;
                    }
                    let Some(w) = width_at(dr) else { continue };
                    let exit_c = col as isize - 1 - w;
                    let enter_c = col as isize + w;

                    if exit_c >= 0 && (exit_c as usize) < cols {
                        let nc = exit_c as usize;
                        let valid = !raster.is_nodata_at(nr as usize, nc).unwrap_or(true);
                        if valid {
                            let v = unsafe { raster.get_unchecked(nr as usize, nc) };
                            let rank = rank_of(v);
                            fenwick.add(rank, -1);
                            touched.push((rank, -1));
                            n -= 1;
                        }
                    }
                    if enter_c >= 0 && (enter_c as usize) < cols {
                        let nc = enter_c as usize;
                        let valid = !raster.is_nodata_at(nr as usize, nc).unwrap_or(true);
                        if valid {
                            let v = unsafe { raster.get_unchecked(nr as usize, nc) };
                            let rank = rank_of(v);
                            fenwick.add(rank, 1);
                            touched.push((rank, 1));
                            n += 1;
                        }
                    }
                }
            }

            if n <= 0 {
                out[[row, col]] = f64::NAN;
                continue;
            }
            let nn = n as usize;

            out[[row, col]] = match percentile {
                Some(p) => {
                    let idx = ((p / 100.0) * (nn as f64 - 1.0)).round();
                    let idx = (idx.clamp(0.0, (nn - 1) as f64)) as usize;
                    sorted_vals[fenwick.find_kth(idx)]
                }
                None => {
                    let mid = nn / 2;
                    if nn % 2 == 0 {
                        let lo = sorted_vals[fenwick.find_kth(mid - 1)];
                        let hi = sorted_vals[fenwick.find_kth(mid)];
                        (lo + hi) / 2.0
                    } else {
                        sorted_vals[fenwick.find_kth(mid)]
                    }
                }
            };
        }

        // Reset the Fenwick tree for the next row without reallocating:
        // undo every update applied while processing this row.
        for (rank, delta) in touched.drain(..) {
            fenwick.add(rank, -delta);
        }
    }

    out
}

#[cfg(test)]
mod fenwick_tests {
    use super::Fenwick;

    #[test]
    fn find_kth_matches_naive_sorted_order() {
        // Multiset {2,2,5,7,7,7,9} inserted as ranks into a coordinate
        // space of 5 distinct values [2,5,7,9,11] (rank 4 = 11 unused).
        let mut fw = Fenwick::new(5);
        let inserts = [(0, 2), (1, 1), (2, 3), (3, 1)]; // rank -> count
        for (rank, count) in inserts {
            for _ in 0..count {
                fw.add(rank, 1);
            }
        }
        // Expanded sorted multiset by rank: [0,0,1,2,2,2,3]
        let expanded = [0usize, 0, 1, 2, 2, 2, 3];
        for (k, &expected) in expanded.iter().enumerate() {
            assert_eq!(fw.find_kth(k), expected, "k={k}");
        }
    }

    #[test]
    fn add_and_remove_round_trip() {
        let mut fw = Fenwick::new(8);
        for r in 0..8 {
            fw.add(r, 1);
        }
        for k in 0..8 {
            assert_eq!(fw.find_kth(k), k);
        }
        // Remove rank 3, everything above shifts down by one in k-order.
        fw.add(3, -1);
        assert_eq!(fw.find_kth(0), 0);
        assert_eq!(fw.find_kth(1), 1);
        assert_eq!(fw.find_kth(2), 2);
        assert_eq!(fw.find_kth(3), 4); // rank 3 skipped
        assert_eq!(fw.find_kth(4), 5);
    }
}
