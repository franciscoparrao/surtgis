# Paper-grade figure for the GFM-prep benchmark.
#
# Reads benchmarks/results/gfm_prep/timings.csv (produced by
# run_gfm_prep_bench.sh) and writes paper/figures/gfm_prep_throughput.pdf.
#
# Asymmetry note: the current Python reference processes single-timestamp
# input only, while SurtGIS processes all BENCH_TIMESTAMPS in one call.
# When the bench is run with BENCH_TIMESTAMPS > 1, SurtGIS is doing more
# work per run; the figure caption must call this out explicitly.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(scales)
})

args_cli <- commandArgs(trailingOnly = TRUE)
root <- if (length(args_cli) >= 1) args_cli[1] else Sys.getenv("SURTGIS_ROOT",
                                                              unset = getwd())
csv  <- file.path(root, "benchmarks/results/gfm_prep/timings.csv")
out  <- file.path(root, "paper/figures/gfm_prep_throughput.pdf")
dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)

df <- read_csv(csv, show_col_types = FALSE)

# Order implementations and pretty-label
df$implementation <- factor(
  df$implementation,
  levels = c("python", "surtgis"),
  labels = c("Python reference (rasterio + numpy)", "SurtGIS extract-patches")
)

# Summary stats per (implementation, size)
summ <- df %>%
  group_by(implementation, size) %>%
  summarise(
    mean_s = mean(wall_clock_s),
    sd_s   = sd(wall_clock_s),
    n_reps = n(),
    .groups = "drop"
  )

# Headline number for caption: speedup at the default size
size_default <- 224
sp_py <- summ %>% filter(implementation == "Python reference (rasterio + numpy)",
                          size == size_default) %>% pull(mean_s)
sp_su <- summ %>% filter(implementation == "SurtGIS extract-patches",
                          size == size_default) %>% pull(mean_s)
speedup <- if (length(sp_py) == 1 && length(sp_su) == 1) sp_py / sp_su else NA

p <- ggplot(df, aes(x = factor(size), y = wall_clock_s, fill = implementation)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.85, width = 0.55,
               position = position_dodge(width = 0.7)) +
  geom_jitter(aes(colour = implementation),
              position = position_jitterdodge(jitter.width = 0.15,
                                              dodge.width = 0.7),
              size = 1.6, alpha = 0.6) +
  scale_fill_manual(values = c("Python reference (rasterio + numpy)" = "#7B7B7B",
                                "SurtGIS extract-patches"            = "#1B5E78")) +
  scale_colour_manual(values = c("Python reference (rasterio + numpy)" = "#3B3B3B",
                                  "SurtGIS extract-patches"            = "#0A2E3A"),
                       guide = "none") +
  labs(
    x = "Patch tile size (pixels)",
    y = "Wall-clock seconds (lower is better)",
    fill = NULL,
    title = "GFM preprocessing throughput: SurtGIS vs Python reference"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "top",
    legend.text = element_text(size = 10),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    axis.text = element_text(colour = "black"),
    plot.title = element_text(size = 12, face = "plain"),
    plot.title.position = "plot",
  )

# Footnote: configuration of this run
cfg <- df[1, ]
cap <- sprintf(
  paste0(
    "Wall-clock time to extract %d labelled patches of size HxH from a ",
    "%dx%d grid with %d HLS-equivalent bands, repeated %d times. ",
    "SurtGIS additionally processes %d timestamps per chip (output ",
    "[N,C,T,H,W]); the Python reference handles one timestamp. ",
    "Mean speedup at tile %d: %.2fx."
  ),
  cfg$n_points, cfg$grid, cfg$grid, 6, max(df$rep),
  cfg$n_timestamps, size_default, speedup
)
p <- p + labs(caption = strwrap(cap, width = 95) |> paste(collapse = "\n")) +
  theme(plot.caption = element_text(hjust = 0, size = 8.5, colour = "#555555"))

ggsave(out, plot = p, width = 7, height = 4.4, dpi = 320, device = cairo_pdf)
cat(sprintf("Wrote %s\n", out))
cat(sprintf("Speedup at tile %d: %.2fx (SurtGIS does %dx the work)\n",
            size_default, speedup, cfg$n_timestamps))
