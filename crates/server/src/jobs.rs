//! Materialisation jobs (design §3 and §5, M1.5).
//!
//! Global operators — the value of a cell depends on the whole basin
//! upstream — cannot be tiled on the fly. They run here as a job over the
//! whole source: `POST /jobs` with a pipeline, the result is written as a
//! Cloud Optimized GeoTIFF under the jobs folder (inside `--root`, so it is
//! servable through the ordinary tile endpoints), and `GET /jobs/{id}`
//! reports progress. One job runs at a time; the rest queue.
//!
//! Pipeline steps, applied in order on the running state:
//!
//! | step | needs | produces | params |
//! |---|---|---|---|
//! | `fill_sinks` | DEM | filled DEM (becomes the DEM) | `min_slope` (1e-5) |
//! | `flow_direction` | DEM | D8 codes | |
//! | `flow_accumulation` | D8 (derived if missing) | cell counts | |
//! | `stream_network` | accumulation (derived) | 0/1 mask | `threshold` (1000) |
//! | `hand` | DEM, D8, accumulation (derived) | height above drainage | `stream_threshold` (1000) |
//! | `twi` | DEM, accumulation (derived) | wetness index | |
//!
//! The output is the product of the last step.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use surtgis_algorithms::hydrology::{
    FillSinksParams, HandParams, StreamNetworkParams, fill_sinks, flow_accumulation,
    flow_direction, hand, stream_network,
};
use surtgis_algorithms::terrain::{SlopeParams, SlopeUnits, slope, twi};
use surtgis_core::Raster;
use surtgis_core::io::{CogCompression, CogOptions, write_cog};

use crate::AppState;
use crate::error::ServeError;
use crate::source::Source;

/// Body of `POST /jobs`.
#[derive(Debug, Clone, Deserialize)]
pub struct JobRequest {
    /// Source: local GeoTIFF under `--root` or an allow-listed COG URL.
    pub url: String,
    /// Steps in order (see the module docs).
    pub pipeline: Vec<String>,
    /// Output name (letters, digits, `_`, `-`); written as
    /// `<jobs dir>/<output>.tif`.
    pub output: String,
    /// Step parameters by name.
    #[serde(default)]
    pub params: HashMap<String, f64>,
}

/// Timing of one executed step.
#[derive(Debug, Clone, Serialize)]
pub struct StepReport {
    /// Step name.
    pub name: String,
    /// Wall time in seconds.
    pub seconds: f64,
}

/// State of a job as reported by `GET /jobs/{id}`.
#[derive(Debug, Clone, Serialize)]
pub struct JobStatus {
    /// Job id.
    pub id: String,
    /// `queued`, `running`, `done` or `failed`.
    pub status: String,
    /// Source requested.
    pub url: String,
    /// Steps requested.
    pub pipeline: Vec<String>,
    /// `?url=` value that serves the result once `done`.
    pub output_url: String,
    /// Output file.
    pub output_path: String,
    /// Unix seconds.
    pub created: f64,
    /// Unix seconds, once running.
    pub started: Option<f64>,
    /// Unix seconds, once finished.
    pub finished: Option<f64>,
    /// Error text, once failed.
    pub error: Option<String>,
    /// Executed steps with timings.
    pub steps: Vec<StepReport>,
    /// Output size, once done.
    pub shape: Option<[usize; 2]>,
}

/// In-memory registry of jobs, newest last.
#[derive(Default)]
pub struct JobRegistry {
    jobs: Mutex<Vec<JobStatus>>,
}

impl JobRegistry {
    fn insert(&self, job: JobStatus) {
        self.jobs.lock().unwrap().push(job);
    }

    fn update(&self, id: &str, f: impl FnOnce(&mut JobStatus)) {
        if let Some(j) = self.jobs.lock().unwrap().iter_mut().find(|j| j.id == id) {
            f(j);
        }
    }

    /// One job by id.
    pub fn get(&self, id: &str) -> Option<JobStatus> {
        self.jobs
            .lock()
            .unwrap()
            .iter()
            .find(|j| j.id == id)
            .cloned()
    }

    /// Every job, oldest first.
    pub fn list(&self) -> Vec<JobStatus> {
        self.jobs.lock().unwrap().clone()
    }

    /// `(queued, running, done, failed)`.
    pub fn counts(&self) -> (usize, usize, usize, usize) {
        let g = self.jobs.lock().unwrap();
        let n = |s: &str| g.iter().filter(|j| j.status == s).count();
        (n("queued"), n("running"), n("done"), n("failed"))
    }
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Output names are a single path component of safe characters.
pub fn validate_output_name(name: &str) -> Result<(), ServeError> {
    if name.is_empty() || name.len() > 100 {
        return Err(ServeError::BadRequest(
            "output must be 1–100 characters".into(),
        ));
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err(ServeError::BadRequest(
            "output may only contain letters, digits, '_' and '-'".into(),
        ));
    }
    Ok(())
}

/// Known step names.
pub const STEPS: &[&str] = &[
    "fill_sinks",
    "flow_direction",
    "flow_accumulation",
    "stream_network",
    "hand",
    "twi",
];

fn param(params: &HashMap<String, f64>, key: &str, default: f64) -> f64 {
    params.get(key).copied().unwrap_or(default)
}

fn u8_to_f64(r: &Raster<u8>) -> Raster<f64> {
    let mut out = Raster::from_array(r.data().mapv(f64::from));
    out.set_transform(*r.transform());
    out.set_crs(r.crs().cloned());
    out
}

fn compute(e: surtgis_core::Error) -> ServeError {
    ServeError::Compute(e.to_string())
}

/// Run `pipeline` on `dem` and return the last product with per-step
/// timings. Pure: no I/O, so it is unit-testable and reusable.
pub fn run_pipeline(
    dem: Raster<f64>,
    pipeline: &[String],
    params: &HashMap<String, f64>,
) -> Result<(Raster<f64>, Vec<StepReport>), ServeError> {
    if pipeline.is_empty() {
        return Err(ServeError::BadRequest("pipeline is empty".into()));
    }
    if let Some(bad) = pipeline.iter().find(|s| !STEPS.contains(&s.as_str())) {
        return Err(ServeError::BadRequest(format!(
            "unknown step '{bad}'; known: {}",
            STEPS.join(", ")
        )));
    }
    let mut dem = dem;
    let mut dir: Option<Raster<u8>> = None;
    let mut acc: Option<Raster<f64>> = None;
    let mut last: Option<Raster<f64>> = None;
    let mut reports = Vec::new();

    for step in pipeline {
        let t0 = Instant::now();
        match step.as_str() {
            "fill_sinks" => {
                let mut p = FillSinksParams::default();
                p.min_slope = param(params, "min_slope", p.min_slope);
                dem = fill_sinks(&dem, p).map_err(compute)?;
                dir = None;
                acc = None;
                last = Some(dem.clone());
            }
            "flow_direction" => {
                let d = flow_direction(&dem).map_err(compute)?;
                last = Some(u8_to_f64(&d));
                dir = Some(d);
            }
            "flow_accumulation" => {
                let d = match &dir {
                    Some(d) => d.clone(),
                    None => flow_direction(&dem).map_err(compute)?,
                };
                let a = flow_accumulation(&d).map_err(compute)?;
                dir = Some(d);
                last = Some(a.clone());
                acc = Some(a);
            }
            "stream_network" => {
                let a = ensure_acc(&dem, &mut dir, &mut acc)?;
                let mut p = StreamNetworkParams::default();
                p.threshold = param(params, "threshold", p.threshold);
                let s = stream_network(&a, p).map_err(compute)?;
                last = Some(u8_to_f64(&s));
            }
            "hand" => {
                let a = ensure_acc(&dem, &mut dir, &mut acc)?;
                let d = dir.clone().expect("ensure_acc sets dir");
                let mut p = HandParams::default();
                p.stream_threshold = param(params, "stream_threshold", p.stream_threshold);
                last = Some(hand(&dem, &d, &a, p).map_err(compute)?);
            }
            "twi" => {
                let a = ensure_acc(&dem, &mut dir, &mut acc)?;
                let mut sp = SlopeParams::default();
                sp.units = SlopeUnits::Radians;
                let s = slope(&dem, sp).map_err(compute)?;
                last = Some(twi(&a, &s).map_err(compute)?);
            }
            _ => unreachable!("validated above"),
        }
        reports.push(StepReport {
            name: step.clone(),
            seconds: t0.elapsed().as_secs_f64(),
        });
    }
    Ok((last.expect("non-empty pipeline"), reports))
}

fn ensure_acc(
    dem: &Raster<f64>,
    dir: &mut Option<Raster<u8>>,
    acc: &mut Option<Raster<f64>>,
) -> Result<Raster<f64>, ServeError> {
    if let Some(a) = acc {
        return Ok(a.clone());
    }
    let d = match dir {
        Some(d) => d.clone(),
        None => flow_direction(dem).map_err(compute)?,
    };
    let a = flow_accumulation(&d).map_err(compute)?;
    *dir = Some(d);
    *acc = Some(a.clone());
    Ok(a)
}

/// Write the product as a tiled, deflate-compressed COG with overviews.
pub fn write_output(raster: &Raster<f64>, path: &Path) -> Result<(), ServeError> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)
            .map_err(|e| ServeError::Compute(format!("cannot create {}: {e}", dir.display())))?;
    }
    let opts = CogOptions {
        compression: CogCompression::Deflate,
        ..CogOptions::default()
    };
    write_cog(raster, path, &opts).map_err(|e| ServeError::Compute(format!("write COG: {e}")))
}

/// Read the whole source as one f64 band (band 0), nodata as NaN.
async fn read_whole(state: &AppState, source: &Source) -> Result<Raster<f64>, ServeError> {
    match source {
        Source::Local(path) => {
            let path = path.clone();
            tokio::task::spawn_blocking(move || {
                surtgis_core::io::read_geotiff::<f64, _>(&path, Some(0))
                    .map(crate::source::nan_nodata)
                    .map_err(|e| ServeError::Source(format!("read {}: {e}", path.display())))
            })
            .await
            .map_err(|e| ServeError::Source(format!("read task failed: {e}")))?
        }
        Source::Http(url) => {
            let mut reader = state.pool.acquire(url).await?;
            let nodata = reader.metadata().nodata;
            let mut r = reader
                .read_full::<f64>(None)
                .await
                .map_err(|e| ServeError::Source(format!("read {url}: {e}")))?;
            if r.nodata().is_none() {
                r.set_nodata(nodata);
            }
            Ok(crate::source::nan_nodata(r))
        }
        Source::Ecw(_) => Err(ServeError::BadRequest(
            "jobs run on elevation rasters; ECW imagery is not a DEM".into(),
        )),
    }
}

/// Register a job and start it (queued behind the single worker).
pub fn submit(state: Arc<AppState>, req: JobRequest) -> Result<JobStatus, ServeError> {
    let Some(jobs_dir) = state.jobs_dir.clone() else {
        return Err(ServeError::BadRequest(
            "jobs need --root (outputs go to <root>/_jobs) or --jobs-dir under --root".into(),
        ));
    };
    validate_output_name(&req.output)?;
    if req.pipeline.is_empty() {
        return Err(ServeError::BadRequest("pipeline is empty".into()));
    }
    if let Some(bad) = req.pipeline.iter().find(|s| !STEPS.contains(&s.as_str())) {
        return Err(ServeError::BadRequest(format!(
            "unknown step '{bad}'; known: {}",
            STEPS.join(", ")
        )));
    }
    let source = Source::resolve(&req.url, &state.sources)?;
    let output_path = jobs_dir.join(format!("{}.tif", req.output));
    let output_url = match &state.sources.root {
        Some(root) => output_path
            .strip_prefix(root)
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| output_path.to_string_lossy().to_string()),
        None => output_path.to_string_lossy().to_string(),
    };

    let id = {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (&req.url, &req.pipeline, &req.output, now().to_bits()).hash(&mut h);
        format!("{:016x}", h.finish())
    };
    let job = JobStatus {
        id: id.clone(),
        status: "queued".into(),
        url: req.url.clone(),
        pipeline: req.pipeline.clone(),
        output_url,
        output_path: output_path.to_string_lossy().to_string(),
        created: now(),
        started: None,
        finished: None,
        error: None,
        steps: Vec::new(),
        shape: None,
    };
    state.jobs.insert(job.clone());

    let st = state.clone();
    let job_id = id.clone();
    tokio::spawn(async move {
        let _permit = st.job_worker.acquire().await;
        st.jobs.update(&job_id, |j| {
            j.status = "running".into();
            j.started = Some(now());
        });
        let result: Result<(Vec<StepReport>, [usize; 2]), ServeError> = async {
            let dem = read_whole(&st, &source).await?;
            let (pipeline, params, out) = (
                req.pipeline.clone(),
                req.params.clone(),
                output_path.clone(),
            );
            tokio::task::spawn_blocking(move || {
                let (product, reports) = run_pipeline(dem, &pipeline, &params)?;
                write_output(&product, &out)?;
                let (r, c) = product.shape();
                Ok((reports, [r, c]))
            })
            .await
            .map_err(|e| ServeError::Compute(format!("job task failed: {e}")))?
        }
        .await;
        st.jobs.update(&job_id, |j| {
            j.finished = Some(now());
            match &result {
                Ok((reports, shape)) => {
                    j.status = "done".into();
                    j.steps = reports.clone();
                    j.shape = Some(*shape);
                }
                Err(e) => {
                    j.status = "failed".into();
                    j.error = Some(e.to_string());
                }
            }
        });
        match &result {
            Ok(_) => tracing::info!(job = %job_id, "job done"),
            Err(e) => tracing::warn!(job = %job_id, error = %e, "job failed"),
        }
    });
    Ok(job)
}

/// Default jobs folder for a root: `<root>/_jobs`.
pub fn default_jobs_dir(root: &Path) -> PathBuf {
    root.join("_jobs")
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::raster::GeoTransform;

    /// A valley: elevation rises away from the middle column, plus a pit.
    fn valley(rows: usize, cols: usize) -> Raster<f64> {
        let mut r = Raster::<f64>::new(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                let dx = (j as f64 - cols as f64 / 2.0).abs();
                r.data_mut()[[i, j]] = 100.0 + dx * 2.0 + i as f64 * 0.1;
            }
        }
        r.data_mut()[[rows / 2, cols / 4]] = 50.0; // pit
        r.set_transform(GeoTransform::new(500_000.0, 6_300_000.0, 10.0, -10.0));
        r.set_crs(Some(surtgis_core::CRS::from_epsg(32719)));
        r
    }

    #[test]
    fn output_name_rules() {
        assert!(validate_output_name("maule_facc-2020").is_ok());
        for bad in ["", "../x", "a/b", "a b", "ü"] {
            assert!(validate_output_name(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn pipeline_runs_and_derives_missing_stages() {
        let dem = valley(40, 60);
        let params = HashMap::from([("threshold".to_string(), 20.0)]);
        // stream_network without explicit direction/accumulation derives both.
        let (out, reports) = run_pipeline(
            dem.clone(),
            &["fill_sinks".into(), "stream_network".into()],
            &params,
        )
        .unwrap();
        assert_eq!(out.shape(), (40, 60));
        assert_eq!(
            reports.iter().map(|r| r.name.as_str()).collect::<Vec<_>>(),
            ["fill_sinks", "stream_network"]
        );
        let ones = out.data().iter().filter(|v| **v == 1.0).count();
        assert!(ones > 0 && ones < 40 * 60, "streams {ones}");
        // Accumulation: the valley axis collects the most.
        let (acc, _) = run_pipeline(
            dem.clone(),
            &["fill_sinks".into(), "flow_accumulation".into()],
            &params,
        )
        .unwrap();
        let (r, c) = acc.shape();
        let axis: f64 = (0..r).map(|i| acc.data()[[i, c / 2]]).sum();
        let edge: f64 = (0..r).map(|i| acc.data()[[i, 1]]).sum();
        assert!(axis > edge, "axis {axis} vs edge {edge}");
        // twi and hand run to completion with finite values somewhere.
        for step in ["twi", "hand"] {
            let (o, _) =
                run_pipeline(dem.clone(), &["fill_sinks".into(), step.into()], &params).unwrap();
            assert!(o.data().iter().any(|v| v.is_finite()), "{step}");
        }
        assert!(run_pipeline(dem.clone(), &[], &params).is_err());
        assert!(run_pipeline(dem, &["nope".into()], &params).is_err());
    }

    #[test]
    fn output_is_a_cog_with_overviews() {
        let dir = tempfile::tempdir().unwrap();
        let (out, _) =
            run_pipeline(valley(700, 1100), &["fill_sinks".into()], &HashMap::new()).unwrap();
        let path = dir.path().join("_jobs").join("filled.tif");
        write_output(&out, &path).unwrap();
        let info = surtgis_core::io::window::geotiff_info(&path).unwrap();
        assert_eq!((info.width, info.height), (1100, 700));
        assert!(info.levels[0].tiled);
        assert!(info.levels.len() >= 2, "{:?}", info.levels);
        assert_eq!(info.crs.and_then(|c| c.epsg()), Some(32719));
    }
}
