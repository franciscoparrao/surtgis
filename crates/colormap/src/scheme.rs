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

    // ── Imhof-style shaded relief palettes (P3) ──
    //
    // Eduard Imhof's "Cartographic Relief Presentation" (1965) is the
    // de-facto reference for shaded-relief colour conventions: green
    // valleys, straw mid-slopes, ochre upper slopes, snow ridges. The
    // four variants below are not bit-equivalent to rayshader's
    // imhof1..imhof4 (rayshader does not publish source stops), but
    // they follow the same intent and produce a comparable read for
    // landscape figures.
    /// Green valleys -> straw -> ochre -> snow ridges. The default
    /// rayshader-style choice for mountain DEMs.
    Imhof1,
    /// Cooler variant — alpine greens, grey-blues at mid, snow tops.
    Imhof2,
    /// Desert-leaning variant — olive valleys, tan slopes, red rock,
    /// bone-white tops.
    Imhof3,
    /// Twilight variant — deep teal valleys, mauve mid, soft blush
    /// at the top. For artistic figures (talks, posters).
    Imhof4,
    /// Smooth black -> white grayscale, mid contrast biased so most of
    /// the action lives in the [0.3, 0.7] band.
    Bw1,
    /// Higher-contrast grayscale with a sharp light mid-ridge — gives
    /// pen-and-ink-style results when paired with strong shadows.
    Bw2,
    /// Dry desert: sand -> tan -> red rock -> bone white.
    DesertDry,
    /// Pastel rainbow (gentle saturation) for non-technical figures.
    Pastel,
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
        Self::Imhof1,
        Self::Imhof2,
        Self::Imhof3,
        Self::Imhof4,
        Self::Bw1,
        Self::Bw2,
        Self::DesertDry,
        Self::Pastel,
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
            Self::Imhof1 => "Imhof 1 (classic)",
            Self::Imhof2 => "Imhof 2 (alpine)",
            Self::Imhof3 => "Imhof 3 (desert)",
            Self::Imhof4 => "Imhof 4 (twilight)",
            Self::Bw1 => "Grayscale (smooth)",
            Self::Bw2 => "Grayscale (high contrast)",
            Self::DesertDry => "Desert dry",
            Self::Pastel => "Pastel rainbow",
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

// ── Imhof-style shaded-relief palettes ──
//
// Stops follow Eduard Imhof's "Cartographic Relief Presentation" (1965)
// conventions: cool greens at valleys, warm earth tones at mid-slope,
// snow-pale at ridges. Each palette was tuned visually against
// `dem_filled.tif` rather than copied bit-for-bit from rayshader (which
// does not publish source values). Hex colours documented inline so
// future taste arguments can audit the curation.

const IMHOF1_STOPS: &[ColorStop] = &[
    ColorStop::new(0.00, 88, 122, 88),   // #587a58 dark valley green
    ColorStop::new(0.20, 142, 165, 110), // #8ea56e meadow green
    ColorStop::new(0.45, 196, 178, 119), // #c4b277 straw mid-slope
    ColorStop::new(0.70, 168, 130, 94),  // #a8825e ochre upper
    ColorStop::new(0.90, 217, 200, 178), // #d9c8b2 warm grey near peak
    ColorStop::new(1.00, 252, 248, 240), // #fcf8f0 snow ridge
];

const IMHOF2_STOPS: &[ColorStop] = &[
    ColorStop::new(0.00, 79, 109, 102),  // #4f6d66 spruce
    ColorStop::new(0.20, 116, 144, 132), // #749084 alpine green
    ColorStop::new(0.45, 154, 158, 153), // #9a9e99 cool stone
    ColorStop::new(0.70, 178, 169, 161), // #b2a9a1 grey-tan
    ColorStop::new(0.90, 224, 216, 213), // #e0d8d5 mineral white
    ColorStop::new(1.00, 250, 250, 252), // #fafafc snow
];

const IMHOF3_STOPS: &[ColorStop] = &[
    ColorStop::new(0.00, 108, 121, 78),  // #6c794e olive valley
    ColorStop::new(0.25, 173, 156, 96),  // #ad9c60 tan
    ColorStop::new(0.50, 192, 137, 84),  // #c08954 dry red ochre
    ColorStop::new(0.75, 158, 96, 64),   // #9e6040 red rock
    ColorStop::new(0.92, 220, 198, 168), // #dcc6a8 bone
    ColorStop::new(1.00, 248, 240, 230), // #f8f0e6 bone white
];

const IMHOF4_STOPS: &[ColorStop] = &[
    ColorStop::new(0.00, 60, 72, 80),    // #3c4850 deep teal
    ColorStop::new(0.30, 110, 100, 130), // #6e6482 mauve
    ColorStop::new(0.55, 174, 142, 162), // #ae8ea2 dusty pink
    ColorStop::new(0.80, 222, 196, 196), // #dec4c4 soft blush
    ColorStop::new(1.00, 250, 240, 235), // #faf0eb cream highlight
];

const BW1_STOPS: &[ColorStop] = &[
    // Compressed contrast so most of the action happens between 0.3 and
    // 0.7 — gives a softer, more "etching"-like read than a linear
    // black-to-white ramp.
    ColorStop::new(0.00, 28, 28, 28),    // #1c1c1c
    ColorStop::new(0.30, 90, 90, 90),    // #5a5a5a
    ColorStop::new(0.50, 150, 150, 150), // #969696
    ColorStop::new(0.70, 210, 210, 210), // #d2d2d2
    ColorStop::new(1.00, 252, 252, 252), // #fcfcfc
];

const BW2_STOPS: &[ColorStop] = &[
    ColorStop::new(0.00, 10, 10, 10),    // near black valley
    ColorStop::new(0.45, 80, 80, 80),    // dark grey
    ColorStop::new(0.55, 230, 230, 230), // sharp light ridge
    ColorStop::new(1.00, 255, 255, 255),
];

const DESERT_DRY_STOPS: &[ColorStop] = &[
    ColorStop::new(0.00, 196, 174, 124), // #c4ae7c sand
    ColorStop::new(0.30, 212, 174, 116), // #d4ae74 dry tan
    ColorStop::new(0.55, 188, 118, 80),  // #bc7650 red rock
    ColorStop::new(0.80, 158, 96, 76),   // #9e604c canyon shadow
    ColorStop::new(1.00, 240, 232, 220), // #f0e8dc bone white
];

const PASTEL_STOPS: &[ColorStop] = &[
    ColorStop::new(0.00, 184, 220, 244), // light blue
    ColorStop::new(0.25, 192, 232, 200), // mint
    ColorStop::new(0.50, 244, 232, 184), // butter
    ColorStop::new(0.75, 244, 196, 184), // peach
    ColorStop::new(1.00, 232, 200, 232), // lilac
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
            let idx = (t * 10.0).floor().clamp(0.0, 9.0) as usize;
            GEOMORPHON_PALETTE[idx]
        }
        ColorScheme::Water => multi_stop(WATER_STOPS, t),
        ColorScheme::Accumulation => multi_stop(ACCUMULATION_STOPS, t),
        ColorScheme::Imhof1 => multi_stop(IMHOF1_STOPS, t),
        ColorScheme::Imhof2 => multi_stop(IMHOF2_STOPS, t),
        ColorScheme::Imhof3 => multi_stop(IMHOF3_STOPS, t),
        ColorScheme::Imhof4 => multi_stop(IMHOF4_STOPS, t),
        ColorScheme::Bw1 => multi_stop(BW1_STOPS, t),
        ColorScheme::Bw2 => multi_stop(BW2_STOPS, t),
        ColorScheme::DesertDry => multi_stop(DESERT_DRY_STOPS, t),
        ColorScheme::Pastel => multi_stop(PASTEL_STOPS, t),
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
        // 8 original + 8 P3 Imhof-style additions.
        assert_eq!(ColorScheme::ALL.len(), 16);
    }

    #[test]
    fn imhof1_endpoints() {
        // Valley green at t=0
        let c = evaluate(ColorScheme::Imhof1, 0.0);
        assert_eq!(c, Rgb::new(88, 122, 88));
        // Snow ridge at t=1
        let c = evaluate(ColorScheme::Imhof1, 1.0);
        assert_eq!(c, Rgb::new(252, 248, 240));
    }

    #[test]
    fn bw2_high_contrast_jumps_at_mid() {
        // BW2 has a sharp 80→230 transition between t=0.45 and t=0.55.
        // The 80 colour is anchored at t=0.45 and the 230 colour at
        // t=0.55, so test just before and just after.
        let dark = evaluate(ColorScheme::Bw2, 0.44);
        let light = evaluate(ColorScheme::Bw2, 0.56);
        assert!(dark.r <= 90, "dark side: r={}", dark.r);
        assert!(light.r >= 220, "light side: r={}", light.r);
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
