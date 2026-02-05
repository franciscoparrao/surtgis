//! File I/O with native file dialogs (rfd).

use std::path::PathBuf;

use crossbeam_channel::Sender;

use surtgis_core::io::{read_geotiff, write_geotiff, GeoTiffOptions};
use surtgis_core::raster::Raster;

use crate::state::{AppMessage, DatasetId, LogEntry};

/// Open a file dialog and load a GeoTIFF in a background thread.
pub fn open_geotiff(tx: Sender<AppMessage>) {
    std::thread::spawn(move || {
        let path = rfd::FileDialog::new()
            .add_filter("GeoTIFF", &["tif", "tiff"])
            .add_filter("All files", &["*"])
            .set_title("Open GeoTIFF")
            .pick_file();

        if let Some(path) = path {
            load_geotiff(path, tx);
        }
    });
}

/// Load a GeoTIFF from a known path in a background thread.
pub fn load_geotiff(path: PathBuf, tx: Sender<AppMessage>) {
    std::thread::spawn(move || {
        let _ = tx.send(AppMessage::Log(LogEntry::info(format!(
            "Loading {}...",
            path.display()
        ))));

        match read_geotiff::<f64, _>(&path, None) {
            Ok(raster) => {
                let stats = raster.statistics();
                let _ = tx.send(AppMessage::Log(LogEntry::success(format!(
                    "Loaded {} ({}x{}, min={:.2}, max={:.2})",
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    raster.cols(),
                    raster.rows(),
                    stats.min.map(|v| format!("{:.2}", v)).unwrap_or_default(),
                    stats.max.map(|v| format!("{:.2}", v)).unwrap_or_default(),
                ))));
                let _ = tx.send(AppMessage::RasterLoaded { path, raster });
            }
            Err(e) => {
                let _ = tx.send(AppMessage::Error {
                    context: "File open".to_string(),
                    message: format!("Failed to read {}: {}", path.display(), e),
                });
                let _ = tx.send(AppMessage::Log(LogEntry::error(format!(
                    "Failed to read {}: {}",
                    path.display(),
                    e
                ))));
            }
        }
    });
}

/// Open a save dialog and write a raster to GeoTIFF.
pub fn save_geotiff(raster: Raster<f64>, dataset_id: DatasetId, tx: Sender<AppMessage>) {
    std::thread::spawn(move || {
        let path = rfd::FileDialog::new()
            .add_filter("GeoTIFF", &["tif", "tiff"])
            .set_title("Save GeoTIFF")
            .set_file_name("output.tif")
            .save_file();

        if let Some(path) = path {
            let _ = tx.send(AppMessage::Log(LogEntry::info(format!(
                "Saving to {}...",
                path.display()
            ))));

            match write_geotiff(&raster, &path, Some(GeoTiffOptions::default())) {
                Ok(()) => {
                    let _ = tx.send(AppMessage::RasterSaved {
                        dataset_id,
                        path: path.clone(),
                    });
                    let _ = tx.send(AppMessage::Log(LogEntry::success(format!(
                        "Saved to {}",
                        path.display()
                    ))));
                }
                Err(e) => {
                    let _ = tx.send(AppMessage::Error {
                        context: "File save".to_string(),
                        message: format!("Failed to write {}: {}", path.display(), e),
                    });
                }
            }
        }
    });
}
