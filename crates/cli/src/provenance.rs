//! Process-level provenance for the CLI.
//!
//! One `surtgis` invocation is one operation, so the record is assembled
//! from the process: the command line as typed, the working directory,
//! the thread count, and every input the native readers open (hashed with
//! BLAKE3 when it is a local file). The engine's writers pick the record
//! up through the hooks in [`surtgis_core::provenance`] and embed it,
//! with the digest of the array they write, in every GeoTIFF/COG the
//! command produces. `--no-provenance` (or `SURTGIS_NO_PROVENANCE=1`)
//! skips all of it, including the input hashing.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use surtgis_core::provenance::{self, InputRecord, Provenance, hash_file};

struct Context {
    argv: Vec<String>,
    cwd: Option<String>,
    inputs: Mutex<BTreeMap<String, InputRecord>>,
}

static CONTEXT: OnceLock<Arc<Context>> = OnceLock::new();

/// Install the hooks for this process. Idempotent.
pub fn install(argv: Vec<String>) {
    let ctx = CONTEXT
        .get_or_init(|| {
            Arc::new(Context {
                argv,
                cwd: std::env::current_dir()
                    .ok()
                    .map(|p| p.display().to_string()),
                inputs: Mutex::new(BTreeMap::new()),
            })
        })
        .clone();
    let observer = ctx.clone();
    provenance::set_input_observer(Some(Arc::new(move |source: &str| {
        observer.note_input(source)
    })));
    let provider = ctx;
    provenance::set_output_provider(Some(Arc::new(move || Some(provider.record()))));
}

/// Whether the environment asks to skip provenance.
pub fn disabled_by_env() -> bool {
    matches!(
        std::env::var("SURTGIS_NO_PROVENANCE").as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    )
}

impl Context {
    fn note_input(&self, source: &str) {
        let mut inputs = self.inputs.lock().unwrap_or_else(|e| e.into_inner());
        if inputs.contains_key(source) {
            return;
        }
        let path = Path::new(source);
        let record = if source.contains("://") || !path.is_file() {
            InputRecord {
                source: source.to_string(),
                blake3: None,
                bytes: None,
                role: None,
            }
        } else {
            match hash_file(path) {
                Ok((digest, bytes)) => InputRecord {
                    source: source.to_string(),
                    blake3: Some(digest),
                    bytes: Some(bytes),
                    role: None,
                },
                Err(_) => InputRecord {
                    source: source.to_string(),
                    blake3: None,
                    bytes: None,
                    role: None,
                },
            }
        };
        inputs.insert(source.to_string(), record);
    }

    fn record(&self) -> Provenance {
        let mut p = Provenance::new()
            .with_operation(operation_name(&self.argv))
            .with_command(self.argv.iter().cloned())
            .with_threads(rayon::current_num_threads());
        if let Some(cwd) = &self.cwd {
            p = p.with_cwd(cwd.clone());
        }
        let inputs = self.inputs.lock().unwrap_or_else(|e| e.into_inner());
        for rec in inputs.values() {
            p.push_input(rec.clone());
        }
        p
    }
}

/// `surtgis --compress terrain slope a.tif b.tif` → `terrain slope`: the
/// first two positional tokens after the binary.
fn operation_name(argv: &[String]) -> String {
    argv.iter()
        .skip(1)
        .filter(|a| !a.starts_with('-'))
        .take(2)
        .cloned()
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operation_is_the_two_leading_subcommands() {
        let argv: Vec<String> = [
            "surtgis",
            "--compress",
            "terrain",
            "slope",
            "a.tif",
            "b.tif",
        ]
        .map(String::from)
        .to_vec();
        assert_eq!(operation_name(&argv), "terrain slope");
        assert_eq!(operation_name(&["surtgis".to_string()]), "");
    }
}
