//! Embedded, verifiable provenance for every raster SurtGIS writes.
//!
//! A [`Provenance`] record says *how a file came to be*: which engine
//! version ran, the operation and its parameters (or the exact command
//! line), the inputs with their BLAKE3 digests, the thread count and
//! seed, and — filled in by the writer — a digest of the output pixel
//! array itself. The record travels **inside** the GeoTIFF/COG as the
//! `SURTGIS_PROVENANCE` item of the `GDAL_METADATA` tag (42112), so
//! `gdalinfo`, rasterio and QGIS show it as ordinary metadata and it
//! survives copies, renames and object-store round trips. Nothing about
//! it changes the pixels.
//!
//! Verification is a pure function of the file and its inputs:
//! [`crate::io::read_provenance`] returns the record, and the output
//! digest recomputed over the decoded array must equal
//! [`OutputRecord::data_blake3`]. Every input listed with a digest can be
//! re-hashed to confirm the lineage is intact. `surtgis verify` does
//! exactly that.
//!
//! # How records reach the writers
//!
//! Two paths, both optional:
//!
//! - **Explicit.** [`crate::io::write_geotiff_with_provenance`] and
//!   [`crate::io::write_cog_with_provenance`] take a record; the writer
//!   fills [`Provenance::output`] and embeds it. Use this from long-lived
//!   services (one record per job).
//! - **Process hooks.** A command-line tool installs an *input observer*
//!   ([`set_input_observer`]) that every native reader calls with the
//!   source it opens, and an *output provider* ([`set_output_provider`])
//!   that every native writer consults when no explicit record was
//!   given. The tool's provider assembles the record from its argv and
//!   the observed inputs. Without hooks the readers and writers behave
//!   exactly as before (no tag is written), so library users pay nothing.
//!
//! # Digest definition
//!
//! `data_blake3` is BLAKE3 over `b"surtgis-data/1\0<dtype>\0<rows>\0<cols>\0<bands>\0"`
//! followed by each band's samples in row-major order as their native
//! little-endian bytes, band after band. `<dtype>` is the Rust primitive
//! name (`f32`, `u16`, …). Multi-band stacks hash band-sequentially even
//! though the TIFF stores them interleaved; the streaming writer hashes
//! the `f32` strips it writes. Input digests are BLAKE3 over the raw file
//! bytes.

use std::path::Path;
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Version of the record layout.
pub const SCHEMA: &str = "surtgis-provenance/1";
/// Name of the `GDAL_METADATA` item carrying the JSON record.
pub const METADATA_ITEM: &str = "SURTGIS_PROVENANCE";
/// Version of the output-array digest definition (see module docs).
pub const DATA_DIGEST_DOMAIN: &str = "surtgis-data/1";

/// Which software produced the file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Engine {
    /// Always `"surtgis"` for records written by this crate.
    pub name: String,
    /// Crate version (`CARGO_PKG_VERSION` of the engine).
    pub version: String,
}

/// One input the operation read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputRecord {
    /// Path or URL as the operation saw it.
    pub source: String,
    /// BLAKE3 of the file bytes, hex; `None` for remote or unhashed
    /// sources.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blake3: Option<String>,
    /// File size in bytes when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes: Option<u64>,
    /// Free-form role (`dem`, `release`, `band:red`, …).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
}

/// Digest of the output array, filled in by the writer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputRecord {
    /// BLAKE3 of the pixel array (see module docs), hex.
    pub data_blake3: String,
    /// `[rows, cols]`.
    pub shape: [usize; 2],
    /// Number of bands hashed.
    pub bands: usize,
    /// Rust primitive name of the samples as written (`f32`, `u8`, …).
    pub dtype: String,
}

/// The provenance record. Construct with [`Provenance::new`] and the
/// `with_*` builders; readers get one back from
/// [`crate::io::read_provenance`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Provenance {
    /// [`SCHEMA`].
    pub schema: String,
    /// Producing software.
    pub engine: Engine,
    /// RFC 3339 UTC timestamp of the write.
    pub created: String,
    /// Short operation name (`terrain slope`, `serve job`, …).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operation: Option<String>,
    /// Command line as typed, when the producer is a CLI.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub command: Vec<String>,
    /// Effective parameters as a JSON object; `Null` when not recorded.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub parameters: serde_json::Value,
    /// Inputs read.
    #[serde(default)]
    pub inputs: Vec<InputRecord>,
    /// Output digest, set by the writer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<OutputRecord>,
    /// Worker threads the computation ran with.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threads: Option<usize>,
    /// Random seed when the operation is stochastic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// `<os>-<arch>` of the producing host.
    pub platform: String,
    /// Working directory the command ran in, when recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
}

impl Provenance {
    /// A record for this engine version, timestamped now, with no
    /// operation, inputs or output yet.
    pub fn new() -> Self {
        Self {
            schema: SCHEMA.to_string(),
            engine: Engine {
                name: "surtgis".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
            created: rfc3339_utc_now(),
            operation: None,
            command: Vec::new(),
            parameters: serde_json::Value::Null,
            inputs: Vec::new(),
            output: None,
            threads: None,
            seed: None,
            platform: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
            cwd: None,
        }
    }

    /// Set the operation name.
    pub fn with_operation(mut self, op: impl Into<String>) -> Self {
        self.operation = Some(op.into());
        self
    }

    /// Set the command line.
    pub fn with_command(mut self, argv: impl IntoIterator<Item = String>) -> Self {
        self.command = argv.into_iter().collect();
        self
    }

    /// Set the parameters object.
    pub fn with_parameters(mut self, params: serde_json::Value) -> Self {
        self.parameters = params;
        self
    }

    /// Set the thread count.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = Some(threads);
        self
    }

    /// Set the seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Record the working directory.
    pub fn with_cwd(mut self, cwd: impl Into<String>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }

    /// Append an input described by hand (remote URL, in-memory buffer…).
    pub fn push_input(&mut self, input: InputRecord) {
        self.inputs.push(input);
    }

    /// Append a local file as input, hashing its bytes. Fails only if the
    /// file cannot be read.
    pub fn push_input_file(&mut self, path: &Path, role: Option<&str>) -> Result<()> {
        let (digest, bytes) = hash_file(path)?;
        self.inputs.push(InputRecord {
            source: path.display().to_string(),
            blake3: Some(digest),
            bytes: Some(bytes),
            role: role.map(str::to_string),
        });
        Ok(())
    }

    /// Serialise to compact, pure-ASCII JSON: every non-ASCII character
    /// (a path under `chañaral/`, an argument with an accent) is written
    /// as a `\uXXXX` escape. The record lives in a TIFF ASCII tag, which
    /// cannot hold UTF-8 bytes; any JSON reader turns the escapes back
    /// into the original text.
    pub fn to_json(&self) -> String {
        ascii_json(&serde_json::to_string(self).expect("provenance serialises"))
    }

    /// Serialise to indented JSON.
    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).expect("provenance serialises")
    }

    /// Parse a record (UTF-8 or `\u`-escaped); the schema must be one
    /// this crate understands.
    pub fn from_json(json: &str) -> Result<Self> {
        let p: Provenance = serde_json::from_str(json)
            .map_err(|e| Error::Other(format!("provenance: invalid JSON record: {e}")))?;
        if p.schema != SCHEMA {
            return Err(Error::Other(format!(
                "provenance: unsupported schema '{}' (this build reads '{}')",
                p.schema, SCHEMA
            )));
        }
        Ok(p)
    }
}

impl Default for Provenance {
    fn default() -> Self {
        Self::new()
    }
}

/// BLAKE3 of a file's bytes as hex, plus its length.
pub fn hash_file(path: &Path) -> Result<(String, u64)> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    let bytes = std::io::copy(&mut file, &mut hasher)?;
    Ok((hasher.finalize().to_hex().to_string(), bytes))
}

/// Incremental digest of an output array following the definition in
/// the module docs. Writers feed it the header once, then the sample
/// bytes band by band, row-major.
pub struct DataDigest {
    hasher: blake3::Hasher,
    rows: usize,
    cols: usize,
    bands: usize,
    dtype: &'static str,
}

impl DataDigest {
    /// Start a digest for an array of the given geometry and sample type
    /// (`std::any::type_name` of the primitive, e.g. `"f32"`).
    pub fn new(rows: usize, cols: usize, bands: usize, dtype: &'static str) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(DATA_DIGEST_DOMAIN.as_bytes());
        hasher.update(b"\0");
        hasher.update(dtype.as_bytes());
        hasher.update(b"\0");
        hasher.update(rows.to_string().as_bytes());
        hasher.update(b"\0");
        hasher.update(cols.to_string().as_bytes());
        hasher.update(b"\0");
        hasher.update(bands.to_string().as_bytes());
        hasher.update(b"\0");
        Self {
            hasher,
            rows,
            cols,
            bands,
            dtype,
        }
    }

    /// Feed native little-endian sample bytes.
    pub fn update(&mut self, sample_bytes: &[u8]) {
        self.hasher.update(sample_bytes);
    }

    /// Finish into an [`OutputRecord`].
    pub fn finish(self) -> OutputRecord {
        OutputRecord {
            data_blake3: self.hasher.finalize().to_hex().to_string(),
            shape: [self.rows, self.cols],
            bands: self.bands,
            dtype: self.dtype.to_string(),
        }
    }
}

// ─── Process-level hooks ──────────────────────────────────────────────

type InputObserver = dyn Fn(&str) + Send + Sync;
type OutputProvider = dyn Fn() -> Option<Provenance> + Send + Sync;

static INPUT_OBSERVER: RwLock<Option<Arc<InputObserver>>> = RwLock::new(None);
static OUTPUT_PROVIDER: RwLock<Option<Arc<OutputProvider>>> = RwLock::new(None);

/// Install (or clear) the process-wide input observer. Every native
/// reader that opens a path or URL calls it once with that source.
pub fn set_input_observer(observer: Option<Arc<InputObserver>>) {
    *INPUT_OBSERVER.write().unwrap_or_else(|e| e.into_inner()) = observer;
}

/// Install (or clear) the process-wide output provider. Native writers
/// call it when they were not handed an explicit record; `None` means
/// "write no provenance".
pub fn set_output_provider(provider: Option<Arc<OutputProvider>>) {
    *OUTPUT_PROVIDER.write().unwrap_or_else(|e| e.into_inner()) = provider;
}

/// Called by readers: report a source to the installed observer, if any.
pub fn observe_input(source: &str) {
    let obs = INPUT_OBSERVER
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    if let Some(obs) = obs {
        obs(source);
    }
}

/// Called by writers: the record to embed when none was given
/// explicitly.
pub fn provided_output() -> Option<Provenance> {
    let prov = OUTPUT_PROVIDER
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    prov.and_then(|p| p())
}

// ─── Time ─────────────────────────────────────────────────────────────

/// Current UTC time as RFC 3339 with second precision (no dependency).
pub fn rfc3339_utc_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    rfc3339_utc(secs)
}

/// Unix seconds → RFC 3339 UTC (`2026-09-28T14:03:07Z`).
pub fn rfc3339_utc(unix_secs: i64) -> String {
    let days = unix_secs.div_euclid(86_400);
    let sod = unix_secs.rem_euclid(86_400);
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y,
        m,
        d,
        sod / 3600,
        (sod % 3600) / 60,
        sod % 60
    )
}

/// Rewrite every non-ASCII character of a serialised JSON document as a
/// `\uXXXX` escape (UTF-16 surrogate pairs above the BMP). Non-ASCII can
/// only occur inside JSON strings, where the escape is equivalent.
fn ascii_json(json: &str) -> String {
    if json.is_ascii() {
        return json.to_string();
    }
    let mut out = String::with_capacity(json.len() + 16);
    for c in json.chars() {
        if c.is_ascii() {
            out.push(c);
        } else {
            let mut buf = [0u16; 2];
            for unit in c.encode_utf16(&mut buf) {
                out.push_str(&format!("\\u{unit:04x}"));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc3339_known_instants() {
        assert_eq!(rfc3339_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339_utc(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(rfc3339_utc(1_790_000_000), "2026-09-21T14:13:20Z");
        assert_eq!(rfc3339_utc(-1), "1969-12-31T23:59:59Z");
    }

    #[test]
    fn json_round_trip_and_schema_check() {
        let mut p = Provenance::new()
            .with_operation("terrain slope")
            .with_command(["surtgis", "terrain", "slope", "a.tif", "b.tif"].map(String::from))
            .with_parameters(serde_json::json!({"units": "degrees"}))
            .with_threads(8);
        p.push_input(InputRecord {
            source: "a.tif".into(),
            blake3: Some("00".into()),
            bytes: Some(1),
            role: Some("dem".into()),
        });
        let back = Provenance::from_json(&p.to_json()).unwrap();
        assert_eq!(back, p);
        let bad = p.to_json().replace(SCHEMA, "surtgis-provenance/99");
        assert!(Provenance::from_json(&bad).is_err());
    }

    #[test]
    fn digest_depends_on_header_and_bytes() {
        let mut a = DataDigest::new(2, 2, 1, "u8");
        a.update(&[1, 2, 3, 4]);
        let mut b = DataDigest::new(2, 2, 1, "u8");
        b.update(&[1, 2, 3, 5]);
        let mut c = DataDigest::new(1, 4, 1, "u8");
        c.update(&[1, 2, 3, 4]);
        let (a, b, c) = (a.finish(), b.finish(), c.finish());
        assert_ne!(a.data_blake3, b.data_blake3);
        assert_ne!(a.data_blake3, c.data_blake3);
        assert_eq!(a.shape, [2, 2]);
        assert_eq!(a.dtype, "u8");
    }

    #[test]
    fn hooks_are_optional_and_replaceable() {
        assert!(provided_output().is_none());
        observe_input("nothing installed");
        let seen = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let s2 = seen.clone();
        set_input_observer(Some(Arc::new(move |s: &str| {
            s2.lock().unwrap().push(s.into())
        })));
        set_output_provider(Some(Arc::new(|| {
            Some(Provenance::new().with_operation("t"))
        })));
        observe_input("x.tif");
        assert_eq!(seen.lock().unwrap().as_slice(), ["x.tif"]);
        assert_eq!(provided_output().unwrap().operation.as_deref(), Some("t"));
        set_input_observer(None);
        set_output_provider(None);
        assert!(provided_output().is_none());
    }
    #[test]
    fn json_is_ascii_and_round_trips_non_ascii_paths() {
        let mut p = Provenance::new();
        p.command = vec!["surtgis".into(), "/datos/chañaral/Aysén/dem 𝛼.tif".into()];
        p.cwd = Some("/home/usuario/Ñuble".into());
        let json = p.to_json();
        assert!(json.is_ascii(), "{json}");
        assert!(json.contains("cha\\u00f1aral"), "{json}");
        assert!(json.contains("\\ud835\\udefc"), "surrogate pair: {json}");
        let back = Provenance::from_json(&json).unwrap();
        assert_eq!(back, p);
    }
}
