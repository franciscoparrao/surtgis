//! The operators a tile can ask for, with the focal support each needs.
//!
//! Only local (per-cell) and focal (fixed window) operators tile on the
//! fly — see the design document §3. Each entry declares its **gutter**:
//! the number of extra cells read around the tile so the kernel has its
//! full support at the tile edges.

use std::collections::HashMap;

use surtgis_algorithms::imagery::index_builder;
use surtgis_algorithms::terrain::{HillshadeParams, SlopeParams, SlopeUnits, hillshade, slope};
use surtgis_core::Raster;

use crate::error::ServeError;

/// Catalogue entry for `/algorithms`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AlgoSpec {
    /// Name used in `?alg=`.
    pub name: &'static str,
    /// `local` or `focal`.
    pub class: &'static str,
    /// Extra cells read around the tile.
    pub gutter: usize,
    /// One-line description.
    pub description: &'static str,
    /// Accepted `?params=` keys with their defaults.
    pub params: Vec<(&'static str, String)>,
    /// Default colormap domain when `?rescale=` is absent.
    pub default_range: Option<[f64; 2]>,
}

/// The M0 catalogue.
pub fn catalog() -> Vec<AlgoSpec> {
    let h = HillshadeParams::default();
    vec![
        AlgoSpec {
            name: "hillshade",
            class: "focal",
            gutter: 1,
            description: "Analytical hillshade (Horn gradient), 0–255",
            params: vec![
                ("azimuth", h.azimuth.to_string()),
                ("altitude", h.altitude.to_string()),
                ("z_factor", h.z_factor.to_string()),
            ],
            default_range: Some([0.0, 255.0]),
        },
        AlgoSpec {
            name: "slope",
            class: "focal",
            gutter: 1,
            description: "Slope (Horn gradient) in degrees, percent or radians",
            params: vec![("units", "degrees".into()), ("z_factor", "1".into())],
            default_range: Some([0.0, 60.0]),
        },
        AlgoSpec {
            name: "formula",
            class: "local",
            gutter: 0,
            description: "Band formula in the Awesome Spectral Indices grammar (`?formula=(N-R)/(N+R)&bands=N:4,R:3`)",
            params: vec![],
            default_range: Some([-1.0, 1.0]),
        },
        AlgoSpec {
            name: "value",
            class: "local",
            gutter: 0,
            description: "The source band itself (`?bands=1` picks a band, default 1)",
            params: vec![],
            default_range: None,
        },
    ]
}

/// A parsed operator ready to run on a window.
#[derive(Debug, Clone)]
pub enum Op {
    /// Raw band.
    Value {
        /// 1-based band index.
        band: usize,
    },
    /// Slope.
    Slope(SlopeParams),
    /// Hillshade.
    Hillshade(HillshadeParams),
    /// Band formula.
    Formula {
        /// Expression in the ASI grammar.
        expr: String,
        /// Formula band names → 1-based band indices.
        bands: Vec<(String, usize)>,
    },
}

fn parse_kv(params: Option<&str>) -> Result<HashMap<String, String>, ServeError> {
    let mut map = HashMap::new();
    for item in params
        .unwrap_or("")
        .split(',')
        .filter(|s| !s.trim().is_empty())
    {
        let (k, v) = item.split_once(':').ok_or_else(|| {
            ServeError::BadRequest(format!("params entry '{item}' is not key:value"))
        })?;
        map.insert(k.trim().to_string(), v.trim().to_string());
    }
    Ok(map)
}

fn take_f64(map: &HashMap<String, String>, key: &str, default: f64) -> Result<f64, ServeError> {
    match map.get(key) {
        None => Ok(default),
        Some(v) => v.parse::<f64>().map_err(|_| {
            ServeError::BadRequest(format!("params: {key} must be a number, got '{v}'"))
        }),
    }
}

/// Parse `bands=N:4,R:3` (name → 1-based index) or `bands=4` (single index).
fn parse_bands(bands: Option<&str>) -> Result<Vec<(String, usize)>, ServeError> {
    let mut out = Vec::new();
    for item in bands
        .unwrap_or("")
        .split(',')
        .filter(|s| !s.trim().is_empty())
    {
        let (name, idx) = match item.split_once(':') {
            Some((n, i)) => (n.trim().to_string(), i.trim()),
            None => (item.trim().to_string(), item.trim()),
        };
        let idx: usize = idx.parse().map_err(|_| {
            ServeError::BadRequest(format!("bands: '{item}' has no 1-based band index"))
        })?;
        if idx == 0 {
            return Err(ServeError::BadRequest("bands: indices are 1-based".into()));
        }
        out.push((name, idx));
    }
    Ok(out)
}

impl Op {
    /// Build the operator from the query parameters.
    pub fn parse(
        alg: Option<&str>,
        formula: Option<&str>,
        bands: Option<&str>,
        params: Option<&str>,
    ) -> Result<Op, ServeError> {
        let map = parse_kv(params)?;
        if let Some(expr) = formula {
            if alg.is_some_and(|a| a != "formula") {
                return Err(ServeError::BadRequest(
                    "formula and alg are mutually exclusive".into(),
                ));
            }
            let bands = parse_bands(bands)?;
            if bands.is_empty() {
                return Err(ServeError::BadRequest(
                    "formula needs bands=NAME:INDEX[,NAME:INDEX...]".into(),
                ));
            }
            return Ok(Op::Formula {
                expr: expr.to_string(),
                bands,
            });
        }
        match alg.unwrap_or("value") {
            "value" => {
                let b = parse_bands(bands)?;
                if b.len() > 1 {
                    return Err(ServeError::BadRequest("value takes a single band".into()));
                }
                Ok(Op::Value {
                    band: b.first().map(|(_, i)| *i).unwrap_or(1),
                })
            }
            "slope" => {
                let units = match map.get("units").map(String::as_str).unwrap_or("degrees") {
                    "degrees" | "deg" => SlopeUnits::Degrees,
                    "percent" | "pct" => SlopeUnits::Percent,
                    "radians" | "rad" => SlopeUnits::Radians,
                    other => {
                        return Err(ServeError::BadRequest(format!(
                            "params: units '{other}' (degrees|percent|radians)"
                        )));
                    }
                };
                let mut p = SlopeParams::default();
                p.units = units;
                p.z_factor = take_f64(&map, "z_factor", p.z_factor)?;
                Ok(Op::Slope(p))
            }
            "hillshade" => {
                let mut p = HillshadeParams::default();
                p.azimuth = take_f64(&map, "azimuth", p.azimuth)?;
                p.altitude = take_f64(&map, "altitude", p.altitude)?;
                p.z_factor = take_f64(&map, "z_factor", p.z_factor)?;
                p.normalized = false;
                Ok(Op::Hillshade(p))
            }
            "formula" => Err(ServeError::BadRequest("formula needs ?formula=...".into())),
            other => Err(ServeError::BadRequest(format!(
                "unknown alg '{other}'; see /algorithms"
            ))),
        }
    }

    /// Extra cells around the tile the operator needs.
    pub fn gutter(&self) -> usize {
        match self {
            Op::Value { .. } | Op::Formula { .. } => 0,
            Op::Slope(_) | Op::Hillshade(_) => 1,
        }
    }

    /// Highest 1-based band index the operator reads.
    pub fn max_band(&self) -> usize {
        match self {
            Op::Value { band } => *band,
            Op::Slope(_) | Op::Hillshade(_) => 1,
            Op::Formula { bands, .. } => bands.iter().map(|(_, i)| *i).max().unwrap_or(1),
        }
    }

    /// Colormap domain used when the request gives none.
    pub fn default_range(&self) -> Option<(f64, f64)> {
        match self {
            Op::Value { .. } => None,
            Op::Slope(p) => Some(match p.units {
                SlopeUnits::Percent => (0.0, 100.0),
                SlopeUnits::Radians => (0.0, 1.0),
                _ => (0.0, 60.0),
            }),
            Op::Hillshade(_) => Some((0.0, 255.0)),
            Op::Formula { .. } => Some((-1.0, 1.0)),
        }
    }

    /// Run on co-registered bands (nodata as NaN) and return one band.
    pub fn run(&self, bands: &[Raster<f64>]) -> Result<Raster<f64>, ServeError> {
        let need = self.max_band();
        if bands.len() < need {
            return Err(ServeError::BadRequest(format!(
                "source has {} band(s), request needs band {need}",
                bands.len()
            )));
        }
        match self {
            Op::Value { band } => Ok(bands[band - 1].clone()),
            Op::Slope(p) => {
                slope(&bands[0], p.clone()).map_err(|e| ServeError::Compute(e.to_string()))
            }
            Op::Hillshade(p) => {
                hillshade(&bands[0], p.clone()).map_err(|e| ServeError::Compute(e.to_string()))
            }
            Op::Formula { expr, bands: map } => {
                let lookup: HashMap<&str, &Raster<f64>> = map
                    .iter()
                    .map(|(name, idx)| (name.as_str(), &bands[idx - 1]))
                    .collect();
                index_builder(expr, &lookup).map_err(|e| ServeError::Compute(e.to_string()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_operators_and_params() {
        match Op::parse(
            Some("hillshade"),
            None,
            None,
            Some("azimuth:135, altitude:30"),
        )
        .unwrap()
        {
            Op::Hillshade(p) => {
                assert_eq!(p.azimuth, 135.0);
                assert_eq!(p.altitude, 30.0);
                assert_eq!(p.z_factor, 1.0);
            }
            other => panic!("{other:?}"),
        }
        match Op::parse(Some("slope"), None, None, Some("units:percent")).unwrap() {
            Op::Slope(p) => assert!(matches!(p.units, SlopeUnits::Percent)),
            other => panic!("{other:?}"),
        }
        match Op::parse(None, Some("(N-R)/(N+R)"), Some("N:4,R:3"), None).unwrap() {
            Op::Formula { expr, bands } => {
                assert_eq!(expr, "(N-R)/(N+R)");
                assert_eq!(bands, vec![("N".to_string(), 4), ("R".to_string(), 3)]);
            }
            other => panic!("{other:?}"),
        }
        assert!(matches!(
            Op::parse(None, None, Some("2"), None).unwrap(),
            Op::Value { band: 2 }
        ));
        assert!(Op::parse(Some("slope"), None, None, Some("z_factor:abc")).is_err());
        assert!(Op::parse(Some("slope"), Some("N"), Some("N:1"), None).is_err());
        assert!(Op::parse(None, Some("N"), None, None).is_err());
        assert!(Op::parse(Some("nope"), None, None, None).is_err());
        assert!(Op::parse(None, None, Some("0"), None).is_err());
    }

    #[test]
    fn runs_formula_and_value() {
        let mut a = Raster::<f64>::new(2, 2);
        let mut b = Raster::<f64>::new(2, 2);
        a.data_mut().fill(3.0);
        b.data_mut().fill(1.0);
        let op = Op::parse(None, Some("(N-R)/(N+R)"), Some("N:1,R:2"), None).unwrap();
        let out = op.run(&[a.clone(), b.clone()]).unwrap();
        assert!((out.data()[[0, 0]] - 0.5).abs() < 1e-12);
        let op = Op::parse(None, None, Some("2"), None).unwrap();
        assert_eq!(op.run(&[a, b]).unwrap().data()[[1, 1]], 1.0);
        let op = Op::parse(Some("slope"), None, None, None).unwrap();
        assert!(op.run(&[]).is_err());
    }

    #[test]
    fn catalogue_matches_gutters() {
        for spec in catalog() {
            let op = match spec.name {
                "formula" => Op::parse(None, Some("N"), Some("N:1"), None).unwrap(),
                n => Op::parse(Some(n), None, None, None).unwrap(),
            };
            assert_eq!(op.gutter(), spec.gutter, "{}", spec.name);
        }
    }
}
