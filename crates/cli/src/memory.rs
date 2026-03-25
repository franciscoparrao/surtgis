//! Memory estimation and streaming decision logic.

use anyhow::{anyhow, Result};
use std::path::Path;

/// Parse a memory size string into bytes.
/// Supports units: B, KB/K, MB/M, GB/G, TB/T, KIB, MIB, GIB, TIB
/// Examples: "4G", "1024MB", "500MiB", "2.5GB"
pub fn parse_memory_size(s: &str) -> Result<u64> {
    let s = s.trim();
    if s.is_empty() {
        return Err(anyhow!("Invalid memory size format: empty string"));
    }

    // Extract numeric part and unit
    let (num_str, unit) = {
        let mut i = 0;
        let bytes = s.as_bytes();
        while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
            i += 1;
        }
        if i == 0 {
            return Err(anyhow!("Invalid memory size format: no number found in '{}'", s));
        }
        (&s[..i], s[i..].trim().to_lowercase())
    };

    let num: f64 = num_str
        .parse()
        .map_err(|_| anyhow!("Invalid number in memory size: {}", num_str))?;

    if num < 0.0 {
        return Err(anyhow!("Memory size must be positive: {}", s));
    }

    let multiplier = match unit.as_str() {
        "" | "b" => 1u64,
        "kb" | "k" => 1_000u64,
        "mb" | "m" => 1_000_000u64,
        "gb" | "g" => 1_000_000_000u64,
        "tb" | "t" => 1_000_000_000_000u64,
        "kib" => 1024u64,
        "mib" => 1024u64 * 1024u64,
        "gib" => 1024u64 * 1024u64 * 1024u64,
        "tib" => 1024u64 * 1024u64 * 1024u64 * 1024u64,
        _ => return Err(anyhow!("Unknown unit in memory size: {}", unit)),
    };

    let result = (num * multiplier as f64).round() as u64;
    Ok(result)
}

/// Estimate the decompressed size of a GeoTIFF file.
/// Reads metadata (width, height) without decompressing data.
/// Assumes Float64 (8 bytes per pixel).
pub fn estimate_decompressed_size(path: &Path) -> Result<u64> {
    use surtgis_core::io::strip_reader::StripReader;

    let reader = StripReader::open(path)
        .map_err(|e| anyhow!("Failed to read TIFF metadata: {}", e))?;

    let total_pixels = reader.rows() as u64 * reader.cols() as u64;
    let bytes_per_pixel = 8u64; // Float64

    Ok(total_pixels * bytes_per_pixel)
}

/// Decide whether to use streaming based on file size and memory limit.
/// Returns true if:
/// - force_streaming is true, OR
/// - file size > 500MB, OR
/// - estimated decompressed size > max_memory_bytes
pub fn should_stream(
    file_path: &Path,
    max_memory_bytes: Option<u64>,
    force_streaming: bool,
) -> Result<bool> {
    if force_streaming {
        return Ok(true);
    }

    // Check file size (500MB threshold)
    if let Ok(metadata) = std::fs::metadata(file_path) {
        if metadata.len() > 500_000_000 {
            return Ok(true);
        }
    }

    // Check estimated decompressed size against memory limit
    if let Some(limit) = max_memory_bytes {
        let est_size = estimate_decompressed_size(file_path)?;
        if est_size > limit {
            return Ok(true);
        }
    }

    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_memory_size_bytes() {
        assert_eq!(parse_memory_size("100").unwrap(), 100);
        assert_eq!(parse_memory_size("100b").unwrap(), 100);
        assert_eq!(parse_memory_size("100B").unwrap(), 100);
    }

    #[test]
    fn test_parse_memory_size_decimal() {
        assert_eq!(parse_memory_size("1.5KB").unwrap(), 1500);
        assert_eq!(parse_memory_size("2.5MB").unwrap(), 2_500_000);
    }

    #[test]
    fn test_parse_memory_size_case_insensitive() {
        assert_eq!(parse_memory_size("1kb").unwrap(), 1000);
        assert_eq!(parse_memory_size("1KB").unwrap(), 1000);
        assert_eq!(parse_memory_size("1Kb").unwrap(), 1000);
    }

    #[test]
    fn test_parse_memory_size_decimal_units() {
        assert_eq!(parse_memory_size("1MB").unwrap(), 1_000_000);
        assert_eq!(parse_memory_size("2GB").unwrap(), 2_000_000_000);
        assert_eq!(parse_memory_size("1TB").unwrap(), 1_000_000_000_000);
    }

    #[test]
    fn test_parse_memory_size_binary_units() {
        assert_eq!(parse_memory_size("1KiB").unwrap(), 1024);
        assert_eq!(parse_memory_size("1MiB").unwrap(), 1024 * 1024);
        assert_eq!(parse_memory_size("1GiB").unwrap(), 1024 * 1024 * 1024);
        assert_eq!(parse_memory_size("1TiB").unwrap(), 1024 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_parse_memory_size_with_whitespace() {
        assert_eq!(parse_memory_size("  100  ").unwrap(), 100);
        assert_eq!(parse_memory_size("1 MB").unwrap(), 1_000_000);
        assert_eq!(parse_memory_size("  2  GB  ").unwrap(), 2_000_000_000);
    }

    #[test]
    fn test_parse_memory_size_errors() {
        assert!(parse_memory_size("").is_err());
        assert!(parse_memory_size("MB").is_err());
        assert!(parse_memory_size("-100MB").is_err());
        assert!(parse_memory_size("100XB").is_err());
    }
}
