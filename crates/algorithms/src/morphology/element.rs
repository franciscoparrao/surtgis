//! Structuring element definitions for morphological operations
//!
//! A structuring element defines the neighborhood shape used in
//! erosion, dilation, and derived transforms.

use surtgis_core::raster::Neighborhood;
use surtgis_core::{Error, Result};

/// Shape of a structuring element for morphological operations
#[derive(Debug, Clone, PartialEq)]
pub enum StructuringElement {
    /// Square element of given radius (side = 2*radius + 1)
    Square(usize),
    /// Cross (plus-shaped) element of given radius
    Cross(usize),
    /// Disk element of given radius
    Disk(usize),
    /// User-provided boolean mask (must be odd-sized and square)
    Custom(Vec<Vec<bool>>),
}

impl Default for StructuringElement {
    fn default() -> Self {
        StructuringElement::Square(1)
    }
}

impl StructuringElement {
    /// Validate the structuring element, returning an error for invalid configurations
    pub fn validate(&self) -> Result<()> {
        match self {
            StructuringElement::Square(r) | StructuringElement::Cross(r) | StructuringElement::Disk(r) => {
                if *r == 0 {
                    return Err(Error::InvalidParameter {
                        name: "radius",
                        value: "0".to_string(),
                        reason: "structuring element radius must be at least 1".to_string(),
                    });
                }
                Ok(())
            }
            StructuringElement::Custom(mask) => {
                if mask.is_empty() {
                    return Err(Error::InvalidParameter {
                        name: "custom_mask",
                        value: "empty".to_string(),
                        reason: "custom mask must not be empty".to_string(),
                    });
                }
                let size = mask.len();
                if size % 2 == 0 {
                    return Err(Error::InvalidParameter {
                        name: "custom_mask",
                        value: format!("{}x{}", size, size),
                        reason: "custom mask size must be odd".to_string(),
                    });
                }
                for row in mask {
                    if row.len() != size {
                        return Err(Error::InvalidParameter {
                            name: "custom_mask",
                            value: format!("row length {}", row.len()),
                            reason: format!("custom mask must be square (expected {})", size),
                        });
                    }
                }
                Ok(())
            }
        }
    }

    /// Get the radius of the structuring element
    pub fn radius(&self) -> usize {
        match self {
            StructuringElement::Square(r)
            | StructuringElement::Cross(r)
            | StructuringElement::Disk(r) => *r,
            StructuringElement::Custom(mask) => mask.len() / 2,
        }
    }

    /// Compute (dr, dc) offsets relative to center for all active cells
    pub fn offsets(&self) -> Vec<(isize, isize)> {
        match self {
            StructuringElement::Square(r) => {
                Neighborhood::Square(*r).offsets()
            }
            StructuringElement::Disk(r) => {
                Neighborhood::Circle(*r).offsets()
            }
            StructuringElement::Cross(r) => {
                let r = *r as isize;
                let mut offsets = Vec::new();
                for d in -r..=r {
                    offsets.push((d, 0)); // vertical arm
                    if d != 0 {
                        offsets.push((0, d)); // horizontal arm (skip center duplicate)
                    }
                }
                offsets
            }
            StructuringElement::Custom(mask) => {
                let center = mask.len() / 2;
                let mut offsets = Vec::new();
                for (r, row) in mask.iter().enumerate() {
                    for (c, &active) in row.iter().enumerate() {
                        if active {
                            offsets.push(
                                (r as isize - center as isize, c as isize - center as isize),
                            );
                        }
                    }
                }
                offsets
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_square_offsets() {
        let se = StructuringElement::Square(1);
        let offsets = se.offsets();
        // 3x3 = 9 offsets
        assert_eq!(offsets.len(), 9);
        assert!(offsets.contains(&(0, 0)));
        assert!(offsets.contains(&(-1, -1)));
        assert!(offsets.contains(&(1, 1)));
    }

    #[test]
    fn test_cross_offsets() {
        let se = StructuringElement::Cross(1);
        let offsets = se.offsets();
        // Plus shape: center + 4 arms = 5
        assert_eq!(offsets.len(), 5);
        assert!(offsets.contains(&(0, 0)));
        assert!(offsets.contains(&(-1, 0)));
        assert!(offsets.contains(&(1, 0)));
        assert!(offsets.contains(&(0, -1)));
        assert!(offsets.contains(&(0, 1)));
        // Corners should NOT be present
        assert!(!offsets.contains(&(-1, -1)));
        assert!(!offsets.contains(&(1, 1)));
    }

    #[test]
    fn test_disk_offsets() {
        let se = StructuringElement::Disk(1);
        let offsets = se.offsets();
        // Disk(1): cells within distance 1.0 of center
        // Center + 4 cardinal = 5 (diagonals are sqrt(2) > 1.0)
        assert_eq!(offsets.len(), 5);
        assert!(offsets.contains(&(0, 0)));
        assert!(offsets.contains(&(-1, 0)));
    }

    #[test]
    fn test_custom_offsets() {
        // L-shaped custom element
        let mask = vec![
            vec![true, false, false],
            vec![true, false, false],
            vec![true, true, true],
        ];
        let se = StructuringElement::Custom(mask);
        let offsets = se.offsets();
        assert_eq!(offsets.len(), 5);
        assert!(offsets.contains(&(-1, -1))); // top-left
        assert!(offsets.contains(&(0, -1)));  // mid-left
        assert!(offsets.contains(&(1, -1)));  // bottom-left
        assert!(offsets.contains(&(1, 0)));   // bottom-center
        assert!(offsets.contains(&(1, 1)));   // bottom-right
    }

    #[test]
    fn test_validate_zero_radius() {
        assert!(StructuringElement::Square(0).validate().is_err());
        assert!(StructuringElement::Cross(0).validate().is_err());
        assert!(StructuringElement::Disk(0).validate().is_err());
    }

    #[test]
    fn test_validate_even_custom() {
        let mask = vec![
            vec![true, false],
            vec![false, true],
        ];
        assert!(StructuringElement::Custom(mask).validate().is_err());
    }

    #[test]
    fn test_default() {
        let se = StructuringElement::default();
        assert_eq!(se, StructuringElement::Square(1));
        assert_eq!(se.radius(), 1);
    }
}
