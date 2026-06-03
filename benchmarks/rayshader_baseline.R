#!/usr/bin/env Rscript
# rayshader_baseline.R — wall-clock baseline for the M2 acceptance criterion of
# the upcoming `surtgis-relief` crate.
#
# Per SPEC_SURTGIS_RELIEF_REVIEW.md §3 Gap 2: before claiming SurtGIS will be
# "faster than rayshader", we record the rayshader wall-clock on the same DEM
# we will use to validate `relief::ray_shade` and `relief::sphere_shade`.
#
# Inputs : dem_filled.tif (637×570 Andes DEM, EPSG:32719, committed in repo root)
# Outputs: benchmarks/results/rayshader_baseline.csv (per-rep timings)
#          stdout summary with median + IQR
#
# Run:   Rscript benchmarks/rayshader_baseline.R [reps] [out_csv]
# Default: 5 reps, writes to benchmarks/results/rayshader_baseline.csv

suppressPackageStartupMessages({
  stopifnot(requireNamespace("rayshader", quietly = TRUE))
  stopifnot(requireNamespace("terra", quietly = TRUE))
})

args <- commandArgs(trailingOnly = TRUE)
n_reps  <- if (length(args) >= 1) as.integer(args[1]) else 5L
out_csv <- if (length(args) >= 2) args[2] else
           "benchmarks/results/rayshader_baseline.csv"

dem_path <- "dem_filled.tif"
if (!file.exists(dem_path)) stop("Missing ", dem_path)

# ----------------------------------------------------------------------------
# Load DEM as a plain numeric matrix (rayshader's native input shape).
# ----------------------------------------------------------------------------

r <- terra::rast(dem_path)
elmat <- matrix(terra::values(r, mat = TRUE),
                nrow = terra::nrow(r), ncol = terra::ncol(r))
# rayshader expects rows = E-W (i.e. transpose of terra row-major)
elmat <- t(elmat)

cat(sprintf("DEM: %s  shape: %d x %d  (%d cells)\n",
            dem_path, nrow(elmat), ncol(elmat), length(elmat)))

# ----------------------------------------------------------------------------
# Benchmark: ray_shade + sphere_shade — the two functions surtgis-relief's M2
# claims to beat. We time them independently and as a composite.
# Sun: azimuth 315, altitude 45 (rayshader and SurtGIS defaults).
# ----------------------------------------------------------------------------

bench_once <- function(elmat) {
  t0 <- Sys.time()
  rs <- rayshader::ray_shade(
    heightmap = elmat,
    anglebreaks = seq(40, 50, 1),   # ~rayshader default soft-shadow window
    sunangle = 315
  )
  t_rs <- as.numeric(Sys.time() - t0, units = "secs")

  t0 <- Sys.time()
  ss <- rayshader::sphere_shade(
    heightmap = elmat,
    sunangle  = 315,
    texture   = "imhof1"
  )
  t_ss <- as.numeric(Sys.time() - t0, units = "secs")

  list(ray_shade_s = t_rs, sphere_shade_s = t_ss,
       total_s = t_rs + t_ss)
}

# Warm-up: one untimed run to load R bytecode + JIT paths.
cat("[warmup]\n"); invisible(bench_once(elmat))

# ----------------------------------------------------------------------------
# Timed reps.
# ----------------------------------------------------------------------------

rows <- vector("list", n_reps)
for (i in seq_len(n_reps)) {
  cat(sprintf("[rep %d/%d] ", i, n_reps))
  r <- bench_once(elmat)
  cat(sprintf("ray=%.2fs  sphere=%.2fs  total=%.2fs\n",
              r$ray_shade_s, r$sphere_shade_s, r$total_s))
  rows[[i]] <- data.frame(rep = i,
                          ray_shade_s = r$ray_shade_s,
                          sphere_shade_s = r$sphere_shade_s,
                          total_s = r$total_s)
}

df <- do.call(rbind, rows)
df$package_version <- as.character(packageVersion("rayshader"))
df$r_version       <- paste(R.version$major, R.version$minor, sep = ".")
df$n_cells         <- length(elmat)
df$nrow            <- nrow(elmat)
df$ncol            <- ncol(elmat)
df$datetime_utc    <- format(Sys.time(), tz = "UTC", "%Y-%m-%dT%H:%M:%SZ")

dir.create(dirname(out_csv), showWarnings = FALSE, recursive = TRUE)
write.csv(df, out_csv, row.names = FALSE)

cat("\n--- summary ---\n")
summarise <- function(x) sprintf(
  "median=%.2fs  IQR=[%.2f, %.2f]  mean=%.2fs  sd=%.2fs",
  median(x), quantile(x, 0.25), quantile(x, 0.75), mean(x), sd(x))
cat(sprintf("ray_shade    : %s\n", summarise(df$ray_shade_s)))
cat(sprintf("sphere_shade : %s\n", summarise(df$sphere_shade_s)))
cat(sprintf("TOTAL        : %s\n", summarise(df$total_s)))
cat(sprintf("\nWrote %s (%d rows)\n", out_csv, nrow(df)))
