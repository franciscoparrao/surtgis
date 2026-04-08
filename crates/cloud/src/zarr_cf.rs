//! CF Conventions metadata parser for Zarr climate datasets.
//!
//! Extracts dimension roles (time, lat, lon), time decoding info,
//! fill value, scale/offset, and variable metadata from Zarr attributes
//! following the [CF Conventions](https://cfconventions.org/).

use chrono::{DateTime, NaiveDate, NaiveDateTime, TimeDelta, Utc};

use crate::error::{CloudError, Result};

/// Parsed CF conventions metadata for a Zarr variable.
#[derive(Debug, Clone)]
pub struct CfMetadata {
    /// Index of the time dimension in the array shape.
    pub time_dim: Option<usize>,
    /// Index of the latitude dimension.
    pub lat_dim: Option<usize>,
    /// Index of the longitude dimension.
    pub lon_dim: Option<usize>,
    /// Time units string, e.g. `"hours since 1900-01-01 00:00:00.0"`.
    pub time_units: Option<String>,
    /// Calendar type, e.g. `"gregorian"`, `"365_day"`.
    pub time_calendar: Option<String>,
    /// `_FillValue` for masked/missing data.
    pub fill_value: Option<f64>,
    /// `scale_factor` for packed data: `value = raw * scale_factor + add_offset`.
    pub scale_factor: Option<f64>,
    /// `add_offset` for packed data.
    pub add_offset: Option<f64>,
    /// Variable units (e.g. `"K"`, `"mm"`).
    pub units: Option<String>,
    /// Human-readable variable name.
    pub long_name: Option<String>,
}

/// Time unit extracted from a CF `units` attribute.
#[derive(Debug, Clone, Copy)]
enum TimeUnit {
    Seconds,
    Minutes,
    Hours,
    Days,
}

impl CfMetadata {
    /// Parse CF metadata from Zarr array and group attributes.
    ///
    /// `dimension_names` should come from the Zarr array's dimension labels
    /// (Zarr v3 `dimension_names` or xarray encoding).
    pub fn from_zarr_attributes(
        array_attrs: &serde_json::Value,
        _group_attrs: &serde_json::Value,
        dimension_names: &[String],
    ) -> Self {
        let time_dim = Self::find_dim(dimension_names, &["time", "t", "datetime"]);
        let lat_dim = Self::find_dim(dimension_names, &["latitude", "lat", "y"]);
        let lon_dim = Self::find_dim(dimension_names, &["longitude", "lon", "x"]);

        let time_units = Self::get_str(array_attrs, "units")
            .filter(|u| u.contains("since"));
        let time_calendar = Self::get_str(array_attrs, "calendar");

        let fill_value = Self::get_f64(array_attrs, "_FillValue");
        let scale_factor = Self::get_f64(array_attrs, "scale_factor");
        let add_offset = Self::get_f64(array_attrs, "add_offset");

        let units = Self::get_str(array_attrs, "units")
            .filter(|u| !u.contains("since"));
        let long_name = Self::get_str(array_attrs, "long_name");

        Self {
            time_dim,
            lat_dim,
            lon_dim,
            time_units,
            time_calendar,
            fill_value,
            scale_factor,
            add_offset,
            units,
            long_name,
        }
    }

    /// Decode raw time coordinate values to UTC datetimes.
    ///
    /// Parses the `units` attribute format: `"{unit} since {reference_datetime}"`
    /// where unit is `hours`, `days`, `seconds`, or `minutes`.
    pub fn decode_time(&self, raw_values: &[f64]) -> Result<Vec<DateTime<Utc>>> {
        let units_str = self.time_units.as_deref().ok_or_else(|| {
            CloudError::ZarrCfError("no time units attribute found".into())
        })?;

        let (unit, reference) = parse_time_units(units_str)?;

        raw_values
            .iter()
            .map(|&val| {
                let delta = match unit {
                    TimeUnit::Seconds => TimeDelta::milliseconds((val * 1000.0) as i64),
                    TimeUnit::Minutes => TimeDelta::seconds((val * 60.0) as i64),
                    TimeUnit::Hours => TimeDelta::seconds((val * 3600.0) as i64),
                    TimeUnit::Days => TimeDelta::seconds((val * 86400.0) as i64),
                };
                Ok(reference + delta)
            })
            .collect()
    }

    /// Apply scale_factor and add_offset to unpack a raw value.
    ///
    /// CF convention: `value = raw * scale_factor + add_offset`.
    pub fn unpack_value(&self, raw: f64) -> f64 {
        let scaled = match self.scale_factor {
            Some(sf) => raw * sf,
            None => raw,
        };
        match self.add_offset {
            Some(ao) => scaled + ao,
            None => scaled,
        }
    }

    /// Check if a value matches the fill value.
    pub fn is_fill(&self, val: f64) -> bool {
        match self.fill_value {
            Some(fv) => val == fv || (fv.is_nan() && val.is_nan()),
            None => false,
        }
    }

    // ── helpers ──────────────────────────────────────────────────────

    fn find_dim(dimension_names: &[String], candidates: &[&str]) -> Option<usize> {
        dimension_names.iter().position(|name| {
            let lower = name.to_lowercase();
            candidates.iter().any(|c| lower == *c)
        })
    }

    fn get_str(attrs: &serde_json::Value, key: &str) -> Option<String> {
        attrs.get(key).and_then(|v| v.as_str()).map(|s| s.to_string())
    }

    fn get_f64(attrs: &serde_json::Value, key: &str) -> Option<f64> {
        attrs.get(key).and_then(|v| {
            v.as_f64().or_else(|| v.as_i64().map(|i| i as f64))
        })
    }
}

/// Parse a CF time units string like `"hours since 1900-01-01 00:00:00.0"`.
fn parse_time_units(units: &str) -> Result<(TimeUnit, DateTime<Utc>)> {
    let parts: Vec<&str> = units.splitn(3, ' ').collect();
    if parts.len() < 3 || parts[1] != "since" {
        return Err(CloudError::ZarrCfError(format!(
            "invalid time units format: '{units}' (expected '<unit> since <datetime>')"
        )));
    }

    let unit = match parts[0].to_lowercase().as_str() {
        "seconds" | "second" | "s" => TimeUnit::Seconds,
        "minutes" | "minute" | "min" => TimeUnit::Minutes,
        "hours" | "hour" | "h" | "hr" => TimeUnit::Hours,
        "days" | "day" | "d" => TimeUnit::Days,
        other => {
            return Err(CloudError::ZarrCfError(format!(
                "unsupported time unit: '{other}'"
            )));
        }
    };

    let ref_str = parts[2].trim();
    let reference = parse_reference_datetime(ref_str)?;

    Ok((unit, reference))
}

/// Parse a reference datetime string in various CF-compatible formats.
fn parse_reference_datetime(s: &str) -> Result<DateTime<Utc>> {
    // Try common formats
    let formats = [
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ];

    for fmt in &formats {
        if let Ok(ndt) = NaiveDateTime::parse_from_str(s, fmt) {
            return Ok(ndt.and_utc());
        }
    }

    // Try date-only
    if let Ok(nd) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return Ok(nd.and_hms_opt(0, 0, 0).unwrap().and_utc());
    }

    Err(CloudError::ZarrCfError(format!(
        "cannot parse reference datetime: '{s}'"
    )))
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_dimension_detection() {
        let dims = vec![
            "time".to_string(),
            "latitude".to_string(),
            "longitude".to_string(),
        ];
        let cf = CfMetadata::from_zarr_attributes(&json!({}), &json!({}), &dims);
        assert_eq!(cf.time_dim, Some(0));
        assert_eq!(cf.lat_dim, Some(1));
        assert_eq!(cf.lon_dim, Some(2));
    }

    #[test]
    fn test_dimension_detection_short_names() {
        let dims = vec!["t".to_string(), "lat".to_string(), "lon".to_string()];
        let cf = CfMetadata::from_zarr_attributes(&json!({}), &json!({}), &dims);
        assert_eq!(cf.time_dim, Some(0));
        assert_eq!(cf.lat_dim, Some(1));
        assert_eq!(cf.lon_dim, Some(2));
    }

    #[test]
    fn test_dimension_detection_xy() {
        let dims = vec!["time".to_string(), "y".to_string(), "x".to_string()];
        let cf = CfMetadata::from_zarr_attributes(&json!({}), &json!({}), &dims);
        assert_eq!(cf.time_dim, Some(0));
        assert_eq!(cf.lat_dim, Some(1));
        assert_eq!(cf.lon_dim, Some(2));
    }

    #[test]
    fn test_cf_attributes_parsing() {
        let attrs = json!({
            "_FillValue": -9999.0,
            "scale_factor": 0.01,
            "add_offset": 273.15,
            "units": "K",
            "long_name": "Temperature at 2m"
        });
        let cf = CfMetadata::from_zarr_attributes(
            &attrs,
            &json!({}),
            &["time".into(), "lat".into(), "lon".into()],
        );
        assert_eq!(cf.fill_value, Some(-9999.0));
        assert_eq!(cf.scale_factor, Some(0.01));
        assert_eq!(cf.add_offset, Some(273.15));
        assert_eq!(cf.units.as_deref(), Some("K"));
        assert_eq!(cf.long_name.as_deref(), Some("Temperature at 2m"));
    }

    #[test]
    fn test_time_units_parsing_era5() {
        let attrs = json!({
            "units": "hours since 1900-01-01 00:00:00.0",
            "calendar": "gregorian"
        });
        let cf = CfMetadata::from_zarr_attributes(
            &attrs,
            &json!({}),
            &["time".into(), "lat".into(), "lon".into()],
        );
        assert_eq!(
            cf.time_units.as_deref(),
            Some("hours since 1900-01-01 00:00:00.0")
        );
        assert_eq!(cf.time_calendar.as_deref(), Some("gregorian"));
    }

    #[test]
    fn test_decode_time_hours() {
        let cf = CfMetadata {
            time_dim: Some(0),
            lat_dim: Some(1),
            lon_dim: Some(2),
            time_units: Some("hours since 1900-01-01 00:00:00.0".into()),
            time_calendar: Some("gregorian".into()),
            fill_value: None,
            scale_factor: None,
            add_offset: None,
            units: None,
            long_name: None,
        };

        // 2020-01-01 00:00 = 1,052,880 hours since 1900-01-01
        // 120 years * 365.25 * 24 ≈ 1,052,856 -- but let's use exact
        let raw = vec![0.0, 24.0, 48.0];
        let decoded = cf.decode_time(&raw).unwrap();
        assert_eq!(decoded[0], DateTime::parse_from_rfc3339("1900-01-01T00:00:00Z").unwrap());
        assert_eq!(decoded[1], DateTime::parse_from_rfc3339("1900-01-02T00:00:00Z").unwrap());
        assert_eq!(decoded[2], DateTime::parse_from_rfc3339("1900-01-03T00:00:00Z").unwrap());
    }

    #[test]
    fn test_decode_time_days() {
        let cf = CfMetadata {
            time_dim: Some(0),
            lat_dim: Some(1),
            lon_dim: Some(2),
            time_units: Some("days since 1950-01-01".into()),
            time_calendar: None,
            fill_value: None,
            scale_factor: None,
            add_offset: None,
            units: None,
            long_name: None,
        };

        let raw = vec![0.0, 365.0];
        let decoded = cf.decode_time(&raw).unwrap();
        assert_eq!(decoded[0], DateTime::parse_from_rfc3339("1950-01-01T00:00:00Z").unwrap());
        assert_eq!(decoded[1], DateTime::parse_from_rfc3339("1951-01-01T00:00:00Z").unwrap());
    }

    #[test]
    fn test_unpack_value() {
        let cf = CfMetadata {
            time_dim: None,
            lat_dim: None,
            lon_dim: None,
            time_units: None,
            time_calendar: None,
            fill_value: None,
            scale_factor: Some(0.1),
            add_offset: Some(200.0),
            units: None,
            long_name: None,
        };
        // 500 * 0.1 + 200 = 250.0
        assert!((cf.unpack_value(500.0) - 250.0).abs() < 1e-10);
    }

    #[test]
    fn test_unpack_no_transform() {
        let cf = CfMetadata {
            time_dim: None,
            lat_dim: None,
            lon_dim: None,
            time_units: None,
            time_calendar: None,
            fill_value: None,
            scale_factor: None,
            add_offset: None,
            units: None,
            long_name: None,
        };
        assert!((cf.unpack_value(42.0) - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_is_fill() {
        let cf = CfMetadata {
            time_dim: None,
            lat_dim: None,
            lon_dim: None,
            time_units: None,
            time_calendar: None,
            fill_value: Some(-9999.0),
            scale_factor: None,
            add_offset: None,
            units: None,
            long_name: None,
        };
        assert!(cf.is_fill(-9999.0));
        assert!(!cf.is_fill(0.0));
    }

    #[test]
    fn test_is_fill_nan() {
        let cf = CfMetadata {
            time_dim: None,
            lat_dim: None,
            lon_dim: None,
            time_units: None,
            time_calendar: None,
            fill_value: Some(f64::NAN),
            scale_factor: None,
            add_offset: None,
            units: None,
            long_name: None,
        };
        assert!(cf.is_fill(f64::NAN));
        assert!(!cf.is_fill(0.0));
    }

    #[test]
    fn test_parse_reference_datetime_formats() {
        let expected = DateTime::parse_from_rfc3339("1900-01-01T00:00:00Z").unwrap();
        assert_eq!(parse_reference_datetime("1900-01-01 00:00:00.0").unwrap(), expected);
        assert_eq!(parse_reference_datetime("1900-01-01 00:00:00").unwrap(), expected);
        assert_eq!(parse_reference_datetime("1900-01-01T00:00:00").unwrap(), expected);
        assert_eq!(parse_reference_datetime("1900-01-01").unwrap(), expected);
    }

    #[test]
    fn test_invalid_time_units() {
        let cf = CfMetadata {
            time_dim: Some(0),
            lat_dim: None,
            lon_dim: None,
            time_units: Some("invalid format".into()),
            time_calendar: None,
            fill_value: None,
            scale_factor: None,
            add_offset: None,
            units: None,
            long_name: None,
        };
        assert!(cf.decode_time(&[0.0]).is_err());
    }
}
