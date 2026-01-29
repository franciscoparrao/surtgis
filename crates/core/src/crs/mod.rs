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
            // Return first 50 chars of WKT
            return format!("WKT:{}", &wkt[..wkt.len().min(50)]);
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
    fn test_crs_equivalence() {
        let a = CRS::from_epsg(4326);
        let b = CRS::wgs84();
        assert!(a.is_equivalent(&b));
    }
}
