//! Color schemes and multi-stop interpolation engine.
//!
//! Direct port of `web/src/lib/colormap.js` to Rust.

/// RGB color as (r, g, b) with values in 0..=255.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Rgb {
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Transparent black (used for nodata).
    pub const TRANSPARENT: Self = Self { r: 0, g: 0, b: 0 };
}

/// A color stop: position in [0, 1] mapped to an RGB color.
#[derive(Debug, Clone, Copy)]
pub struct ColorStop {
    pub t: f64,
    pub color: Rgb,
}

impl ColorStop {
    pub const fn new(t: f64, r: u8, g: u8, b: u8) -> Self {
        Self {
            t,
            color: Rgb::new(r, g, b),
        }
    }
}

/// Available color schemes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorScheme {
    /// Green -> Yellow -> Brown -> White (elevation)
    Terrain,
    /// Blue -> White -> Red (divergent data)
    Divergent,
    /// Black -> White
    Grayscale,
    /// Brown -> Yellow -> Green (NDVI-specific)
    Ndvi,
    /// Blue -> White -> Red (curvature, TPI, DEV)
    BlueWhiteRed,
    /// 10 distinct landform classes
    Geomorphons,
    /// White -> Cyan -> Blue (water depth/moisture)
    Water,
    /// Yellow -> Orange -> Red -> Purple (flow accumulation)
    Accumulation,
}

impl ColorScheme {
    /// All available schemes, useful for UI combo boxes.
    pub const ALL: &[ColorScheme] = &[
        Self::Terrain,
        Self::Divergent,
        Self::Grayscale,
        Self::Ndvi,
        Self::BlueWhiteRed,
        Self::Geomorphons,
        Self::Water,
        Self::Accumulation,
    ];

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Terrain => "Terrain",
            Self::Divergent => "Divergent",
            Self::Grayscale => "Grayscale",
            Self::Ndvi => "NDVI",
            Self::BlueWhiteRed => "Blue-White-Red",
            Self::Geomorphons => "Geomorphons",
            Self::Water => "Water",
            Self::Accumulation => "Accumulation",
        }
    }
}

// ─── Color stop definitions (from colormap.js) ────────────────────────

const TERRAIN_STOPS: &[ColorStop] = &[
    ColorStop::new(0.00, 34, 139, 34),
    ColorStop::new(0.25, 144, 190, 60),
    ColorStop::new(0.50, 220, 200, 80),
    ColorStop::new(0.75, 180, 120, 60),
    ColorStop::new(1.00, 255, 255, 255),
];

const DIVERGENT_STOPS: &[ColorStop] = &[
    ColorStop::new(0.00, 44, 62, 180),
    ColorStop::new(0.25, 120, 160, 220),
    ColorStop::new(0.50, 240, 240, 240),
    ColorStop::new(0.75, 220, 120, 80),
    ColorStop::new(1.00, 180, 30, 30),
];

const NDVI_STOPS: &[ColorStop] = &[
    ColorStop::new(0.0, 120, 70, 20),
    ColorStop::new(0.3, 200, 170, 60),
    ColorStop::new(0.5, 240, 230, 100),
    ColorStop::new(0.7, 100, 180, 50),
    ColorStop::new(1.0, 10, 100, 20),
];

const BLUE_WHITE_RED_STOPS: &[ColorStop] = &[
    ColorStop::new(0.00, 33, 102, 172),
    ColorStop::new(0.25, 103, 169, 207),
    ColorStop::new(0.50, 247, 247, 247),
    ColorStop::new(0.75, 239, 138, 98),
    ColorStop::new(1.00, 178, 24, 43),
];

const WATER_STOPS: &[ColorStop] = &[
    ColorStop::new(0.00, 240, 249, 255),
    ColorStop::new(0.25, 186, 228, 250),
    ColorStop::new(0.50, 80, 180, 230),
    ColorStop::new(0.75, 30, 120, 200),
    ColorStop::new(1.00, 8, 48, 107),
];

const ACCUMULATION_STOPS: &[ColorStop] = &[
    ColorStop::new(0.00, 255, 255, 212),
    ColorStop::new(0.25, 254, 217, 142),
    ColorStop::new(0.50, 254, 153, 41),
    ColorStop::new(0.75, 204, 76, 2),
    ColorStop::new(1.00, 102, 37, 6),
];

/// Geomorphons palette: 10 discrete landform classes.
const GEOMORPHON_PALETTE: &[Rgb] = &[
    Rgb::new(255, 255, 255), // 0: flat
    Rgb::new(56, 168, 0),    // 1: summit
    Rgb::new(198, 219, 124), // 2: ridge
    Rgb::new(255, 255, 115), // 3: shoulder
    Rgb::new(255, 200, 65),  // 4: spur
    Rgb::new(233, 150, 0),   // 5: slope
    Rgb::new(255, 85, 0),    // 6: hollow
    Rgb::new(196, 0, 0),     // 7: footslope
    Rgb::new(132, 0, 168),   // 8: valley
    Rgb::new(0, 92, 230),    // 9: depression
];

// ─── Interpolation engine ──────────────────────────────────────────────

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

fn lerp_color(c1: Rgb, c2: Rgb, t: f64) -> Rgb {
    Rgb::new(
        lerp(c1.r as f64, c2.r as f64, t).round() as u8,
        lerp(c1.g as f64, c2.g as f64, t).round() as u8,
        lerp(c1.b as f64, c2.b as f64, t).round() as u8,
    )
}

fn multi_stop(stops: &[ColorStop], t: f64) -> Rgb {
    if t <= 0.0 {
        return stops[0].color;
    }
    if t >= 1.0 {
        return stops[stops.len() - 1].color;
    }
    for i in 1..stops.len() {
        if t <= stops[i].t {
            let ratio = (t - stops[i - 1].t) / (stops[i].t - stops[i - 1].t);
            return lerp_color(stops[i - 1].color, stops[i].color, ratio);
        }
    }
    stops[stops.len() - 1].color
}

/// Evaluate a color scheme at normalized position `t` ∈ [0, 1].
///
/// For most schemes this performs multi-stop linear interpolation.
/// For `Geomorphons`, `t` is mapped to one of 10 discrete classes.
/// For `Grayscale`, a simple linear ramp is used.
pub fn evaluate(scheme: ColorScheme, t: f64) -> Rgb {
    match scheme {
        ColorScheme::Terrain => multi_stop(TERRAIN_STOPS, t),
        ColorScheme::Divergent => multi_stop(DIVERGENT_STOPS, t),
        ColorScheme::Grayscale => {
            let v = (t.clamp(0.0, 1.0) * 255.0).round() as u8;
            Rgb::new(v, v, v)
        }
        ColorScheme::Ndvi => multi_stop(NDVI_STOPS, t),
        ColorScheme::BlueWhiteRed => multi_stop(BLUE_WHITE_RED_STOPS, t),
        ColorScheme::Geomorphons => {
            let idx = (t * 10.0).floor().min(9.0).max(0.0) as usize;
            GEOMORPHON_PALETTE[idx]
        }
        ColorScheme::Water => multi_stop(WATER_STOPS, t),
        ColorScheme::Accumulation => multi_stop(ACCUMULATION_STOPS, t),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terrain_endpoints() {
        let c0 = evaluate(ColorScheme::Terrain, 0.0);
        assert_eq!(c0, Rgb::new(34, 139, 34));
        let c1 = evaluate(ColorScheme::Terrain, 1.0);
        assert_eq!(c1, Rgb::new(255, 255, 255));
    }

    #[test]
    fn grayscale_midpoint() {
        let c = evaluate(ColorScheme::Grayscale, 0.5);
        assert_eq!(c, Rgb::new(128, 128, 128));
    }

    #[test]
    fn geomorphons_classes() {
        // Class 0 (flat)
        let c = evaluate(ColorScheme::Geomorphons, 0.0);
        assert_eq!(c, Rgb::new(255, 255, 255));
        // Class 9 (depression) at t=0.95
        let c = evaluate(ColorScheme::Geomorphons, 0.95);
        assert_eq!(c, Rgb::new(0, 92, 230));
    }

    #[test]
    fn ndvi_endpoints() {
        let c0 = evaluate(ColorScheme::Ndvi, 0.0);
        assert_eq!(c0, Rgb::new(120, 70, 20));
        let c1 = evaluate(ColorScheme::Ndvi, 1.0);
        assert_eq!(c1, Rgb::new(10, 100, 20));
    }

    #[test]
    fn clamping_below_zero() {
        let c = evaluate(ColorScheme::Terrain, -0.5);
        assert_eq!(c, Rgb::new(34, 139, 34));
    }

    #[test]
    fn clamping_above_one() {
        let c = evaluate(ColorScheme::Terrain, 1.5);
        assert_eq!(c, Rgb::new(255, 255, 255));
    }

    #[test]
    fn all_schemes_list() {
        assert_eq!(ColorScheme::ALL.len(), 8);
    }

    #[test]
    fn all_schemes_evaluate_midpoint() {
        for &scheme in ColorScheme::ALL {
            let c = evaluate(scheme, 0.5);
            // Just verify it doesn't panic and returns valid RGB
            assert!(c.r <= 255 && c.g <= 255 && c.b <= 255);
        }
    }
}
