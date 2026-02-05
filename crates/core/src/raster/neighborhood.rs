//! Neighborhood operations for raster analysis

use super::{Raster, RasterElement};

/// Defines a neighborhood pattern around a cell
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Neighborhood {
    /// 3x3 neighborhood (8 neighbors + center)
    Queen3x3,
    /// 3x3 without corners (4 neighbors + center)
    Rook3x3,
    /// 5x5 neighborhood (24 neighbors + center)
    Queen5x5,
    /// Custom square neighborhood of given radius
    Square(usize),
    /// Circular neighborhood of given radius (in cells)
    Circle(usize),
}

impl Neighborhood {
    /// Get the radius of the neighborhood
    pub fn radius(&self) -> usize {
        match self {
            Neighborhood::Queen3x3 | Neighborhood::Rook3x3 => 1,
            Neighborhood::Queen5x5 => 2,
            Neighborhood::Square(r) | Neighborhood::Circle(r) => *r,
        }
    }

    /// Get the size of the neighborhood (width and height)
    pub fn size(&self) -> usize {
        self.radius() * 2 + 1
    }

    /// Check if a relative position is within this neighborhood
    pub fn contains(&self, dr: isize, dc: isize) -> bool {
        match self {
            Neighborhood::Queen3x3 => dr.abs() <= 1 && dc.abs() <= 1,
            Neighborhood::Rook3x3 => {
                (dr.abs() <= 1 && dc == 0) || (dr == 0 && dc.abs() <= 1)
            }
            Neighborhood::Queen5x5 => dr.abs() <= 2 && dc.abs() <= 2,
            Neighborhood::Square(r) => {
                let r = *r as isize;
                dr.abs() <= r && dc.abs() <= r
            }
            Neighborhood::Circle(r) => {
                let r = *r as f64;
                let dist = ((dr * dr + dc * dc) as f64).sqrt();
                dist <= r
            }
        }
    }

    /// Iterate over relative positions in this neighborhood
    pub fn offsets(&self) -> Vec<(isize, isize)> {
        let r = self.radius() as isize;
        let mut offsets = Vec::new();

        for dr in -r..=r {
            for dc in -r..=r {
                if self.contains(dr, dc) {
                    offsets.push((dr, dc));
                }
            }
        }

        offsets
    }

    /// Get offsets excluding the center cell
    pub fn offsets_no_center(&self) -> Vec<(isize, isize)> {
        self.offsets()
            .into_iter()
            .filter(|&(dr, dc)| dr != 0 || dc != 0)
            .collect()
    }
}

/// D8 flow directions (standard encoding)
#[allow(dead_code)]
pub mod d8 {
    /// Direction offsets: (row_offset, col_offset)
    /// Indexed by direction code (1-8), 0 is unused
    pub const OFFSETS: [(isize, isize); 9] = [
        (0, 0),   // 0: no flow / pit
        (0, 1),   // 1: E
        (-1, 1),  // 2: NE
        (-1, 0),  // 3: N
        (-1, -1), // 4: NW
        (0, -1),  // 5: W
        (1, -1),  // 6: SW
        (1, 0),   // 7: S
        (1, 1),   // 8: SE
    ];

    /// Distance multipliers for each direction
    /// Cardinal directions = 1.0, diagonal = sqrt(2)
    pub const DISTANCES: [f64; 9] = [
        0.0,
        1.0,
        std::f64::consts::SQRT_2,
        1.0,
        std::f64::consts::SQRT_2,
        1.0,
        std::f64::consts::SQRT_2,
        1.0,
        std::f64::consts::SQRT_2,
    ];

    /// Get the opposite direction
    pub fn opposite(dir: u8) -> u8 {
        if dir == 0 {
            0
        } else {
            ((dir - 1 + 4) % 8) + 1
        }
    }
}

/// Iterator over neighborhood values for a specific cell
pub struct NeighborhoodIterator<'a, T: RasterElement> {
    raster: &'a Raster<T>,
    center_row: usize,
    center_col: usize,
    offsets: Vec<(isize, isize)>,
    index: usize,
}

impl<'a, T: RasterElement> NeighborhoodIterator<'a, T> {
    pub fn new(raster: &'a Raster<T>, row: usize, col: usize, neighborhood: Neighborhood) -> Self {
        Self {
            raster,
            center_row: row,
            center_col: col,
            offsets: neighborhood.offsets_no_center(),
            index: 0,
        }
    }
}

impl<'a, T: RasterElement> Iterator for NeighborhoodIterator<'a, T> {
    /// (row, col, value) or None if out of bounds
    type Item = Option<(usize, usize, T)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.offsets.len() {
            return None;
        }

        let (dr, dc) = self.offsets[self.index];
        self.index += 1;

        let new_row = self.center_row as isize + dr;
        let new_col = self.center_col as isize + dc;

        if new_row < 0
            || new_col < 0
            || new_row >= self.raster.rows() as isize
            || new_col >= self.raster.cols() as isize
        {
            Some(None)
        } else {
            let r = new_row as usize;
            let c = new_col as usize;
            // Safe because we just checked bounds
            let value = unsafe { self.raster.get_unchecked(r, c) };
            Some(Some((r, c, value)))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.offsets.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, T: RasterElement> ExactSizeIterator for NeighborhoodIterator<'a, T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neighborhood_offsets() {
        let queen = Neighborhood::Queen3x3;
        let offsets = queen.offsets();
        assert_eq!(offsets.len(), 9); // 3x3

        let rook = Neighborhood::Rook3x3;
        let offsets = rook.offsets();
        assert_eq!(offsets.len(), 5); // center + 4 cardinal

        let queen5 = Neighborhood::Queen5x5;
        let offsets = queen5.offsets();
        assert_eq!(offsets.len(), 25); // 5x5
    }

    #[test]
    fn test_d8_opposite() {
        assert_eq!(d8::opposite(1), 5); // E -> W
        assert_eq!(d8::opposite(3), 7); // N -> S
        assert_eq!(d8::opposite(2), 6); // NE -> SW
    }
}
