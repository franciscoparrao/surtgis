//! Coordinate Reference System handling

use serde::{Deserialize, Serialize};
use std::fmt;

/// Coordinate Reference System representation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CRS {
    /// WKT representation (primary)
    wkt: Option<String>,
    /// EPSG code if known
    epsg: Option<u32>,
    /// PROJ string if available
    proj: Option<String>,
}

impl CRS {
    /// Create a CRS from an EPSG code
    pub fn from_epsg(code: u32) -> Self {
        Self {
            wkt: None,
            epsg: Some(code),
            proj: None,
        }
    }

    /// Create a CRS from a WKT string
    pub fn from_wkt(wkt: impl Into<String>) -> Self {
        Self {
            wkt: Some(wkt.into()),
            epsg: None,
            proj: None,
        }
    }

    /// Create a CRS from a PROJ string
    pub fn from_proj(proj: impl Into<String>) -> Self {
        Self {
            wkt: None,
            epsg: None,
            proj: Some(proj.into()),
        }
    }

    /// WGS84 geographic CRS (EPSG:4326)
    pub fn wgs84() -> Self {
        Self::from_epsg(4326)
    }

    /// Web Mercator (EPSG:3857)
    pub fn web_mercator() -> Self {
        Self::from_epsg(3857)
    }

    /// Get EPSG code if known
    pub fn epsg(&self) -> Option<u32> {
        self.epsg
    }

    /// Get WKT representation
    pub fn wkt(&self) -> Option<&str> {
        self.wkt.as_deref()
    }

    /// Get PROJ string
    pub fn proj(&self) -> Option<&str> {
        self.proj.as_deref()
    }

    /// Whether this CRS is geographic (longitude/latitude in degrees).
    ///
    /// Detection is heuristic but covers the practical cases:
    /// - Known geographic EPSG codes (4326, 4269, 4258, 4267, 4283, 4617,
    ///   4674, 4759, plus the 4000–4999 block, which EPSG reserves for
    ///   geographic 2D CRS).
    /// - WKT starting with `GEOGCS`/`GEOGCRS` (WKT1/WKT2 geographic).
    /// - PROJ strings containing `+proj=longlat`.
    ///
    /// Returns `false` when the CRS carries no usable information — callers
    /// must not silently assume projected in that case if it matters.
    pub fn is_geographic(&self) -> bool {
        if let Some(code) = self.epsg {
            // The EPSG 4000-4999 block is geographic 2D (4326, 4269, ...).
            return (4000..5000).contains(&code);
        }
        if let Some(wkt) = &self.wkt {
            let head = wkt.trim_start().to_ascii_uppercase();
            return head.starts_with("GEOGCS") || head.starts_with("GEOGCRS");
        }
        if let Some(proj) = &self.proj {
            return proj.contains("+proj=longlat");
        }
        false
    }

    /// Check if two CRS are equivalent
    pub fn is_equivalent(&self, other: &CRS) -> bool {
        // Simple check: if both have EPSG codes, compare them
        if let (Some(a), Some(b)) = (self.epsg, other.epsg) {
            return a == b;
        }

        // If both have WKT, compare (this is imperfect)
        if let (Some(a), Some(b)) = (&self.wkt, &other.wkt) {
            return a == b;
        }

        // If both have PROJ, compare
        if let (Some(a), Some(b)) = (&self.proj, &other.proj) {
            return a == b;
        }

        false
    }

    /// Get a string identifier for this CRS
    pub fn identifier(&self) -> String {
        if let Some(code) = self.epsg {
            return format!("EPSG:{}", code);
        }
        if let Some(proj) = &self.proj {
            return proj.clone();
        }
        if let Some(wkt) = &self.wkt {
            // First ~50 chars of WKT, cut at a char boundary (byte slicing
            // panics if byte 50 falls inside a multi-byte character, e.g.
            // accented datum names).
            let cut = wkt
                .char_indices()
                .map(|(i, _)| i)
                .take_while(|&i| i <= 50)
                .last()
                .unwrap_or(0);
            return format!("WKT:{}", &wkt[..cut.min(wkt.len())]);
        }
        "Unknown".to_string()
    }
}

impl fmt::Display for CRS {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.identifier())
    }
}

impl Default for CRS {
    fn default() -> Self {
        Self::wgs84()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crs_epsg() {
        let crs = CRS::from_epsg(4326);
        assert_eq!(crs.epsg(), Some(4326));
        assert_eq!(crs.identifier(), "EPSG:4326");
    }

    #[test]
    fn test_is_geographic() {
        assert!(CRS::from_epsg(4326).is_geographic());
        assert!(CRS::from_epsg(4269).is_geographic()); // NAD83
        assert!(!CRS::from_epsg(32719).is_geographic()); // UTM 19S
        assert!(!CRS::from_epsg(3857).is_geographic());
        assert!(CRS::from_wkt("GEOGCS[\"WGS 84\",DATUM[...]]").is_geographic());
        assert!(!CRS::from_wkt("PROJCS[\"UTM 19S\",GEOGCS[...]]").is_geographic());
        assert!(CRS::from_proj("+proj=longlat +datum=WGS84").is_geographic());
        assert!(!CRS::from_proj("+proj=utm +zone=19 +south").is_geographic());
    }

    #[test]
    fn test_identifier_multibyte_wkt_no_panic() {
        // Byte 50 falls inside a multi-byte char — must not panic
        let wkt = format!("GEOGCS[\"{}\"]", "á".repeat(60));
        let crs = CRS::from_wkt(wkt);
        let _ = crs.identifier();
    }

    #[test]
    fn test_crs_equivalence() {
        let a = CRS::from_epsg(4326);
        let b = CRS::wgs84();
        assert!(a.is_equivalent(&b));
    }
}
