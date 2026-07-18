//! Flow state in structure-of-arrays layout.

/// Conserved shallow-water state `U = (h, hu, hv)` in strict `SoA` (structure-of-arrays) layout
/// (spec §6): three separate row-major `Vec<f32>`, row 0 = north.
///
/// `h` is the flow thickness in metres; `hu`/`hv` are the depth-integrated
/// momenta in m²/s (`u` positive eastwards, `v` positive northwards).
#[derive(Clone, Debug)]
pub struct FlowState {
    /// Flow thickness per cell, in metres.
    pub h: Vec<f32>,
    /// Depth-integrated x-momentum per cell (`h·u`), in m²/s.
    pub hu: Vec<f32>,
    /// Depth-integrated y-momentum per cell (`h·v`), in m²/s.
    pub hv: Vec<f32>,
}

impl FlowState {
    /// All-zero state for `n` cells.
    pub(crate) fn zeros(n: usize) -> Self {
        Self {
            h: vec![0.0; n],
            hu: vec![0.0; n],
            hv: vec![0.0; n],
        }
    }

    /// Overwrite `self` with a copy of `other` (equal lengths assumed).
    pub(crate) fn copy_from(&mut self, other: &Self) {
        self.h.copy_from_slice(&other.h);
        self.hu.copy_from_slice(&other.hu);
        self.hv.copy_from_slice(&other.hv);
    }

    /// `true` if any component is NaN or infinite.
    pub(crate) fn has_non_finite(&self) -> bool {
        // Chunked any() keeps this a tight scan; the three arrays are checked
        // independently so the common all-finite case reads each once.
        self.h.iter().any(|v| !v.is_finite())
            || self.hu.iter().any(|v| !v.is_finite())
            || self.hv.iter().any(|v| !v.is_finite())
    }
}
