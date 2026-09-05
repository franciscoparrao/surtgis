//! Multiresolution reconstruction: the line-recursive inverse DWT.
//!
//! ECW's decompressor is asymmetric by design: analysis uses an 11-tap
//! filter bank, but synthesis collapses the pyramid with a 3-tap (1-2-1)
//! filter that needs only two sideband lines per level in memory at any
//! moment. Each level keeps a two-line ring buffer per band per sideband;
//! producing one output line of level *L* consumes (every second line) one
//! new sideband line of *L*, whose LL component is produced recursively by
//! level *L − 1*. Memory use is therefore proportional to image *width*,
//! never area — a 1.3-gigapixel orthomosaic reconstructs through a few
//! megabytes of line buffers.
//!
//! The synthesis formulas and the region/reflection bookkeeping follow the
//! format description in `docs/ecw_format.md`; the four even/odd phases
//! come from expanding the 1-2-1 filter over the interleaved sub-bands.

use crate::block::Block;
use crate::header::EcwHeader;
use crate::{EcwError, Result};
use std::io::{Read, Seek, SeekFrom};

const LL: usize = 0;
const LH: usize = 1;
const HL: usize = 2;
const HH: usize = 3;

/// A rectangular read request in file coordinates (inclusive bounds) with
/// an output size; the engine picks the shallowest pyramid level that
/// covers `number_x × number_y` and resamples nearest-neighbour from it.
#[derive(Debug, Clone, Copy)]
pub struct RegionParams {
    /// Left column (inclusive).
    pub start_x: u32,
    /// Top row (inclusive).
    pub start_y: u32,
    /// Right column (inclusive).
    pub end_x: u32,
    /// Bottom row (inclusive).
    pub end_y: u32,
    /// Output width in cells (≤ region width).
    pub number_x: u32,
    /// Output height in cells (≤ region height).
    pub number_y: u32,
}

/// Immutable per-level geometry computed at region setup.
#[derive(Debug, Clone)]
struct LevelPlan {
    x_size: u32,
    y_size: u32,
    level_size_x: u32,
    level_start_y: u32,
    level_size_y: u32,
    reflect_start_x: u32,
    reflect_end_x: u32,
    reflect_start_y: u32,
    output_start_x: u32,
    output_size_x: u32,
    start_x_block: u32,
    x_block_count: u32,
    first_block_skip: u32,
    last_block_skip: u32,
}

/// Mutable per-level reconstruction state.
struct LevelState {
    plan: LevelPlan,
    /// Ring buffers indexed `band * 4 + sideband`, length `size_x + 2`.
    line0: Vec<Vec<f32>>,
    line1: Vec<Vec<f32>>,
    /// Per-band scratch the smaller level's output lands in.
    ll_scratch: Vec<Vec<f32>>,
    current_line: u32,
    read_lines: u32,
    blocks: Option<Vec<Block>>,
    next_block_y_line: u32,
}

/// Streaming region decoder over an open ECW file.
pub struct Region<'a, R: Read + Seek> {
    file: &'a mut R,
    header: &'a EcwHeader,
    table: &'a [u64],
    levels: Vec<LevelState>,
    chosen: usize,
    bands: usize,
    /// Output lines of the chosen level, one buffer per band.
    ll_lines: Vec<Vec<f32>>,
    start_line: f64,
    current_line: f64,
    increment_x: f64,
    increment_y: f64,
    counter: u64,
    read_line: u32,
    params: RegionParams,
}

impl<'a, R: Read + Seek> Region<'a, R> {
    /// Validate the request, choose the pyramid level, and lay out the
    /// per-level geometry chain.
    pub fn new(
        file: &'a mut R,
        header: &'a EcwHeader,
        table: &'a [u64],
        params: RegionParams,
    ) -> Result<Self> {
        let RegionParams {
            start_x,
            start_y,
            end_x,
            end_y,
            number_x,
            number_y,
        } = params;
        if start_x > end_x
            || start_y > end_y
            || end_x >= header.x_size
            || end_y >= header.y_size
            || number_x == 0
            || number_y == 0
            || number_x > 1 + end_x - start_x
            || number_y > 1 + end_y - start_y
        {
            return Err(EcwError::InvalidRegion(format!(
                "region ({start_x},{start_y})..({end_x},{end_y}) at {number_x}x{number_y} \
                 is outside the {}x{} image or over-sampled",
                header.x_size, header.y_size
            )));
        }

        // Descend the pyramid until the level's output resolution is no
        // more than 2x the requested output size in both axes.
        let mut o_start_x = start_x;
        let mut o_end_x = end_x;
        let mut o_start_y = start_y;
        let mut o_end_y = end_y;
        let mut o_size_x = 1 + o_end_x - o_start_x;
        let mut o_size_y = 1 + o_end_y - o_start_y;
        let mut chosen = header.num_levels as usize - 1;
        while o_size_y > number_y * 2 && o_size_x > number_x * 2 {
            if chosen == 0 {
                break;
            }
            o_start_x /= 2;
            o_start_y /= 2;
            o_end_x /= 2;
            o_end_y /= 2;
            o_size_x = 1 + o_end_x - o_start_x;
            o_size_y = 1 + o_end_y - o_start_y;
            chosen -= 1;
        }
        // Single-pixel outputs read one representative (centre) sample.
        if number_x == 1 {
            o_start_x += (o_end_x - o_start_x) / 2;
            o_end_x = o_start_x;
            o_size_x = 1;
        }
        if number_y == 1 {
            o_start_y += (o_end_y - o_start_y) / 2;
            o_end_y = o_start_y;
            o_size_y = 1;
        }

        let bands = header.nr_bands as usize;
        let increment_x = f64::from(o_size_x) / f64::from(number_x);
        let increment_y = f64::from(o_size_y) / f64::from(number_y);
        let start_line = f64::from(o_start_y);
        let out_w = o_size_x as usize;

        // Walk down the chain computing each level's slice of the region.
        let mut levels: Vec<LevelState> = Vec::with_capacity(chosen + 1);
        for level in (0..=chosen).rev() {
            let info = &header.levels[level];
            // Size of the grid this level's output lands on.
            let larger_x_size = if level + 1 < header.num_levels as usize {
                header.levels[level + 1].x_size
            } else {
                header.x_size
            };
            let larger_y_size = if level + 1 < header.num_levels as usize {
                header.levels[level + 1].y_size
            } else {
                header.y_size
            };

            // Output pixel N needs level pixels (N-1)/2 and (N-1)/2 + 1;
            // edges reflect instead of reading outside the level.
            let (level_start_x, reflect_start_x) = if o_start_x > 0 {
                ((o_start_x - 1) / 2, 0)
            } else {
                (0, 1)
            };
            let (level_end_x, reflect_end_x) = if o_end_x < larger_x_size - 1 {
                (
                    if o_end_x > 0 {
                        (o_end_x - 1) / 2 + 1
                    } else {
                        0
                    },
                    0,
                )
            } else {
                (info.x_size - 1, 1)
            };
            let (level_start_y, reflect_start_y) = if o_start_y > 0 {
                ((o_start_y - 1) / 2, 0)
            } else {
                (0, 1)
            };
            let (level_end_y, _reflect_end_y) = if o_end_y < larger_y_size - 1 {
                (
                    if o_end_y > 0 {
                        (o_end_y - 1) / 2 + 1
                    } else {
                        0
                    },
                    0,
                )
            } else {
                (info.y_size - 1, 1)
            };
            if level_end_x >= info.x_size || level_end_y >= info.y_size {
                return Err(EcwError::Malformed(format!(
                    "level {level} smaller than the region requires"
                )));
            }
            let level_size_x = 1 + level_end_x - level_start_x;
            let level_size_y = 1 + level_end_y - level_start_y;

            let bs = u32::from(header.x_block_size);
            let start_x_block = level_start_x / bs;
            let last_x_block = level_end_x / bs;
            let first_block_skip = level_start_x - start_x_block * bs;
            let last_pixel = ((last_x_block + 1) * bs - 1).min(info.x_size - 1);
            let last_block_skip = last_pixel - level_end_x;

            let plan = LevelPlan {
                x_size: info.x_size,
                y_size: info.y_size,
                level_size_x,
                level_start_y,
                level_size_y,
                reflect_start_x,
                reflect_end_x,
                reflect_start_y,
                output_start_x: o_start_x,
                output_size_x: o_size_x,
                start_x_block,
                x_block_count: 1 + last_x_block - start_x_block,
                first_block_skip,
                last_block_skip,
            };

            let buf_len = level_size_x as usize + 2;
            let make_lines = || (0..bands * 4).map(|_| vec![0.0f32; buf_len]).collect();
            levels.push(LevelState {
                current_line: plan.level_start_y,
                read_lines: 2 - plan.reflect_start_y,
                ll_scratch: (0..bands).map(|_| vec![0.0f32; buf_len]).collect(),
                line0: make_lines(),
                line1: make_lines(),
                blocks: None,
                next_block_y_line: 0,
                plan,
            });

            // The next smaller level's output grid is this level's slice.
            o_start_x = level_start_x;
            o_end_x = level_end_x;
            o_start_y = level_start_y;
            o_end_y = level_end_y;
            o_size_x = level_size_x;
        }
        levels.reverse(); // index by level number, smallest first

        Ok(Self {
            file,
            header,
            table,
            levels,
            chosen,
            bands,
            ll_lines: (0..bands).map(|_| vec![0.0f32; out_w]).collect(),
            start_line,
            current_line: start_line,
            increment_x,
            increment_y,
            counter: 0,
            read_line: 1,
            params,
        })
    }

    /// The requested output size, `(number_x, number_y)`.
    pub fn output_size(&self) -> (u32, u32) {
        (self.params.number_x, self.params.number_y)
    }

    /// World-space cell size of the output, as multiples of the file cell.
    pub fn output_cell_scale(&self) -> (f64, f64) {
        let sx = f64::from(1 + self.params.end_x - self.params.start_x)
            / f64::from(self.params.number_x);
        let sy = f64::from(1 + self.params.end_y - self.params.start_y)
            / f64::from(self.params.number_y);
        (sx, sy)
    }

    /// Produce the next output row. `out` gets one f32 slice per band, each
    /// `number_x` long, in the file's native band space (Y,U,V for YUV
    /// files; the caller applies the colour transform).
    pub fn next_line(&mut self, out: &mut [Vec<f32>]) -> Result<()> {
        let n_line_y = self.current_line as u32;
        while self.read_line > 0 {
            let y_line = n_line_y + 1 - self.read_line;
            let mut ll = std::mem::take(&mut self.ll_lines);
            let result = self.produce_line(self.chosen, y_line, &mut ll);
            self.ll_lines = ll;
            result?;
            self.read_line -= 1;
        }

        // Nearest-neighbour horizontal resample with 32.32 fixed point,
        // like the reference reader (1/2^16 increments shifted up 16).
        let increment_x = ((self.increment_x * 65536.0) as u64) << 16;
        for (band, out_row) in out.iter_mut().enumerate().take(self.bands) {
            let src = &self.ll_lines[band];
            let mut x_offset: u64 = 0;
            for slot in out_row.iter_mut().take(self.params.number_x as usize) {
                *slot = src[(x_offset >> 32) as usize];
                x_offset += increment_x;
            }
        }

        // Decide how many source lines the next output row needs.
        self.counter += 1;
        let y_line = self.start_line + self.increment_y * self.counter as f64;
        let next = y_line as u32;
        let cur = self.current_line as u32;
        self.read_line = next.saturating_sub(cur);
        self.current_line = y_line;
        Ok(())
    }

    /// Reconstruct one output line of `level` into `out[band]` (each slice
    /// `plan.output_size_x` long). `y_line` is the absolute row index on
    /// the level's output grid; its parity selects the filter phase.
    fn produce_line(&mut self, level: usize, y_line: u32, out: &mut [Vec<f32>]) -> Result<()> {
        // Pull in as many sideband lines as this position requires.
        while self.levels[level].read_lines > 0 {
            {
                let st = &mut self.levels[level];
                for buf in 0..st.line0.len() {
                    std::mem::swap(&mut st.line0[buf], &mut st.line1[buf]);
                }
            }

            // The LL band comes from the smaller level (except at level 0,
            // where it is stored in the file like the other sidebands).
            if level > 0 {
                let current = self.levels[level].current_line;
                let mut scratch = std::mem::take(&mut self.levels[level].ll_scratch);
                let result = self.produce_line(level - 1, current, &mut scratch);
                let st = &mut self.levels[level];
                if result.is_ok() {
                    let rsx = st.plan.reflect_start_x as usize;
                    let w = st.plan.level_size_x as usize;
                    for (band, src) in scratch.iter().enumerate().take(self.bands) {
                        st.line1[band * 4 + LL][rsx..rsx + w].copy_from_slice(&src[..w]);
                    }
                }
                self.levels[level].ll_scratch = scratch;
                result?;
            }

            let st = &self.levels[level];
            if st.current_line >= st.plan.y_size {
                // Reading past the bottom edge: reflect the previous line.
                let st = &mut self.levels[level];
                for buf in 0..st.line0.len() {
                    let (src, dst) = (st.line0[buf].clone(), &mut st.line1[buf]);
                    dst.copy_from_slice(&src);
                }
            } else {
                self.fetch_sideband_line(level)?;
                let st = &mut self.levels[level];
                if st.plan.reflect_start_y != 0 && st.current_line == 0 {
                    for buf in 0..st.line0.len() {
                        let (src, dst) = (st.line1[buf].clone(), &mut st.line0[buf]);
                        dst.copy_from_slice(&src);
                    }
                }
            }

            // Left/right edge reflection on the fresh line.
            let st = &mut self.levels[level];
            let rsx = st.plan.reflect_start_x as usize;
            let w = st.plan.level_size_x as usize;
            if st.plan.reflect_start_x != 0 {
                for buf in st.line1.iter_mut() {
                    buf[0] = buf[1];
                }
            }
            if st.plan.reflect_end_x != 0 {
                for buf in st.line1.iter_mut() {
                    buf[w + rsx] = buf[w + rsx - 1];
                }
            }

            st.read_lines -= 1;
            st.current_line += 1;
        }

        // 3-tap synthesis: four phases by (row, column) parity.
        let st = &self.levels[level];
        let width = st.plan.output_size_x as usize;
        for (band, out_row) in out.iter_mut().enumerate().take(self.bands) {
            let ll0 = &st.line0[band * 4 + LL];
            let ll1 = &st.line1[band * 4 + LL];
            let lh0 = &st.line0[band * 4 + LH];
            let lh1 = &st.line1[band * 4 + LH];
            let hl0 = &st.line0[band * 4 + HL];
            let hl1 = &st.line1[band * 4 + HL];
            let hh0 = &st.line0[band * 4 + HH];
            let hh1 = &st.line1[band * 4 + HH];

            let mut i = 0usize;
            let mut x = st.plan.output_start_x;
            if y_line & 1 == 0 {
                for slot in out_row.iter_mut().take(width) {
                    if x & 1 == 0 {
                        *slot = ll1[i + 1] - (lh0[i + 1] + lh1[i + 1] + hl1[i] + hl1[i + 1]) * 0.5
                            + (hh0[i] + hh0[i + 1] + hh1[i] + hh1[i + 1]) * 0.25;
                        i += 1;
                    } else {
                        *slot = ((ll1[i] + ll1[i + 1]) - (hh0[i] + hh1[i])) * 0.5
                            - (lh0[i] + lh0[i + 1] + lh1[i] + lh1[i + 1]) * 0.25
                            + hl1[i];
                    }
                    x += 1;
                }
            } else {
                for slot in out_row.iter_mut().take(width) {
                    if x & 1 == 0 {
                        *slot = ((ll0[i + 1] + ll1[i + 1]) - (hh0[i] + hh0[i + 1])) * 0.5
                            + lh0[i + 1]
                            - (hl0[i] + hl0[i + 1] + hl1[i] + hl1[i + 1]) * 0.25;
                        i += 1;
                    } else {
                        *slot = (ll0[i] + ll0[i + 1] + ll1[i] + ll1[i + 1]) * 0.25
                            + (lh0[i] + lh0[i + 1] + hl0[i] + hl1[i]) * 0.5
                            + hh0[i];
                    }
                    x += 1;
                }
            }
        }

        // After an even output row the next (odd) row needs one new line.
        if y_line & 1 == 0 {
            self.levels[level].read_lines = 1;
        }
        Ok(())
    }

    /// Decode one row of every stored sideband of `level` into `line1`,
    /// loading the row of blocks it lives in on demand.
    fn fetch_sideband_line(&mut self, level: usize) -> Result<()> {
        let bands = self.bands;
        let y_block_size = u32::from(self.header.y_block_size);
        let x_block_size = u32::from(self.header.x_block_size);
        let (first_sb, sb_count) = if level == 0 { (LL, 4) } else { (LH, 3) };
        let info = &self.header.levels[level];
        let scale = f32::from(self.header.scale_factor);

        // Load the block row covering current_line if not already loaded.
        if self.levels[level].blocks.is_none() {
            let st_plan = self.levels[level].plan.clone();
            let current = self.levels[level].current_line;
            let y_block = current / y_block_size;
            let lines_to_skip = current - y_block * y_block_size;

            let mut row = Vec::with_capacity(st_plan.x_block_count as usize);
            for i in 0..st_plan.x_block_count {
                let xb = st_plan.start_x_block + i;
                let data = self.read_block_bytes(level, xb, y_block)?;
                let mut block = Block::parse(data, bands, sb_count)?;
                if lines_to_skip > 0 {
                    let stored_w = stored_block_width(info.x_size, x_block_size, xb);
                    for idx in 0..bands * sb_count {
                        block.skip_values(idx, (lines_to_skip * stored_w) as usize)?;
                    }
                }
                row.push(block);
            }
            let st = &mut self.levels[level];
            st.blocks = Some(row);
            st.next_block_y_line = lines_to_skip;
        }

        // Decode one row from each block in the row.
        {
            let st = &mut self.levels[level];
            let plan = &st.plan;
            let rsx = plan.reflect_start_x as usize;
            let blocks = st.blocks.as_mut().expect("block row just ensured");
            let n_blocks = blocks.len();
            let mut valid0 = 0usize;
            for (i, block) in blocks.iter_mut().enumerate() {
                let xb = plan.start_x_block + i as u32;
                let stored_w = stored_block_width(plan.x_size, x_block_size, xb) as usize;
                let pre = if i == 0 {
                    plan.first_block_skip as usize
                } else {
                    0
                };
                let post = if i + 1 == n_blocks {
                    plan.last_block_skip as usize
                } else {
                    0
                };
                let valid = stored_w - pre - post;
                if i == 0 {
                    valid0 = valid;
                }
                let write_off = if i == 0 {
                    rsx
                } else {
                    rsx + valid0 + (i - 1) * x_block_size as usize
                };
                for band in 0..bands {
                    let bin_size = info.bin_sizes[band] as f32 / scale;
                    for sb in first_sb..4 {
                        let idx = band * sb_count + (sb - first_sb);
                        if pre > 0 {
                            block.skip_values(idx, pre)?;
                        }
                        block.read_values(
                            idx,
                            &mut st.line1[band * 4 + sb][write_off..write_off + valid],
                            bin_size,
                        )?;
                        if post > 0 {
                            block.skip_values(idx, post)?;
                        }
                    }
                }
            }
            st.next_block_y_line += 1;
            // Drop the row when it is exhausted or the region is done.
            if st.next_block_y_line >= y_block_size
                || st.current_line - plan.level_start_y >= plan.level_size_y - 1
            {
                st.blocks = None;
            }
        }
        Ok(())
    }

    /// Read the raw bytes of one block via the global offset table.
    fn read_block_bytes(&mut self, level: usize, x_block: u32, y_block: u32) -> Result<Vec<u8>> {
        let info = &self.header.levels[level];
        if x_block >= info.nr_x_blocks || y_block >= info.nr_y_blocks {
            return Err(EcwError::Malformed(format!(
                "block ({x_block},{y_block}) outside level {level}"
            )));
        }
        let id = (info.first_block + y_block * info.nr_x_blocks + x_block) as usize;
        let offset = self.table[id];
        let length = self.table[id + 1] - offset;
        if length == 0 {
            return Ok(Vec::new());
        }
        self.file
            .seek(SeekFrom::Start(self.header.blocks_start + offset))
            .map_err(|e| EcwError::Io(e.to_string()))?;
        let mut buf = vec![0u8; length as usize];
        self.file
            .read_exact(&mut buf)
            .map_err(|e| EcwError::Io(format!("block {id}: {e}")))?;
        Ok(buf)
    }
}

/// Width in cells actually stored for block column `xb` (edge blocks only
/// store the columns that exist).
fn stored_block_width(level_x_size: u32, block_size: u32, xb: u32) -> u32 {
    let full_blocks = level_x_size / block_size;
    if xb < full_blocks {
        block_size
    } else {
        level_x_size - xb * block_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stored_width_of_edge_block() {
        // level 124 wide, 64-cell blocks: block 0 stores 64, block 1 stores 60
        assert_eq!(stored_block_width(124, 64, 0), 64);
        assert_eq!(stored_block_width(124, 64, 1), 60);
        // exact multiple: both full
        assert_eq!(stored_block_width(128, 64, 1), 64);
    }
}
