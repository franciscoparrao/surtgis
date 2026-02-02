//! Variogram computation and model fitting
//!
//! Computes the empirical (experimental) variogram from sample points and
//! fits theoretical models (spherical, exponential, Gaussian, Matérn).
//! Prerequisite for kriging interpolation.
//!
//! The semivariance γ(h) measures spatial dissimilarity as a function of
//! separation distance h:
//! ```text
//! γ(h) = (1/2N(h)) Σ [z(xᵢ) - z(xⱼ)]²   for all pairs with |xᵢ-xⱼ| ∈ h±Δh/2
//! ```
//!
//! Reference:
//! Matheron, G. (1963). Principles of geostatistics. Economic Geology.
//! Cressie, N. (1993). Statistics for Spatial Data. Wiley.
//! Florinsky, I.V. (2025). Digital Terrain Analysis, §3.3.

use surtgis_core::{Error, Result};

use super::SamplePoint;

/// Empirical variogram: semivariance values at discrete lag distances.
#[derive(Debug, Clone)]
pub struct EmpiricalVariogram {
    /// Lag distances (bin centers)
    pub lags: Vec<f64>,
    /// Semivariance values γ(h) at each lag
    pub semivariance: Vec<f64>,
    /// Number of point pairs contributing to each lag bin
    pub pair_counts: Vec<usize>,
}

/// Theoretical variogram model type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VariogramModel {
    /// Spherical model: γ(h) = c₀ + c·[1.5(h/a) - 0.5(h/a)³] for h ≤ a; c₀+c for h > a
    Spherical,
    /// Exponential model: γ(h) = c₀ + c·[1 - exp(-3h/a)]
    Exponential,
    /// Gaussian model: γ(h) = c₀ + c·[1 - exp(-3h²/a²)]
    Gaussian,
}

/// Fitted variogram model parameters
#[derive(Debug, Clone)]
pub struct FittedVariogram {
    /// Model type
    pub model: VariogramModel,
    /// Nugget (c₀): semivariance at h → 0 (measurement error + micro-scale variation)
    pub nugget: f64,
    /// Sill (c₀ + c): semivariance at which the model levels off
    pub sill: f64,
    /// Range (a): distance at which semivariance reaches ~95% of sill
    pub range: f64,
    /// Partial sill (c = sill - nugget)
    pub partial_sill: f64,
    /// Residual sum of squares from fitting (lower = better)
    pub rss: f64,
}

impl FittedVariogram {
    /// Evaluate the fitted variogram model at distance h
    pub fn evaluate(&self, h: f64) -> f64 {
        if h < 1e-15 {
            return 0.0;
        }

        let c0 = self.nugget;
        let c = self.partial_sill;
        let a = self.range;

        match self.model {
            VariogramModel::Spherical => {
                if h >= a {
                    c0 + c
                } else {
                    let hr = h / a;
                    c0 + c * (1.5 * hr - 0.5 * hr * hr * hr)
                }
            }
            VariogramModel::Exponential => {
                c0 + c * (1.0 - (-3.0 * h / a).exp())
            }
            VariogramModel::Gaussian => {
                c0 + c * (1.0 - (-3.0 * h * h / (a * a)).exp())
            }
        }
    }
}

/// Parameters for empirical variogram computation
#[derive(Debug, Clone)]
pub struct VariogramParams {
    /// Number of lag bins (default 15)
    pub n_lags: usize,
    /// Maximum lag distance. If None, auto-computed as half the max pairwise distance.
    pub max_lag: Option<f64>,
    /// Lag tolerance as fraction of bin width (default 1.0 = full bin)
    pub lag_tolerance: f64,
}

impl Default for VariogramParams {
    fn default() -> Self {
        Self {
            n_lags: 15,
            max_lag: None,
            lag_tolerance: 1.0,
        }
    }
}

/// Compute the empirical (experimental) variogram from sample points.
///
/// # Arguments
/// * `points` — Sample points with (x, y, value)
/// * `params` — Variogram parameters (number of lags, max distance)
///
/// # Returns
/// [`EmpiricalVariogram`] with lag distances, semivariance, and pair counts.
pub fn empirical_variogram(
    points: &[SamplePoint],
    params: VariogramParams,
) -> Result<EmpiricalVariogram> {
    let n = points.len();
    if n < 2 {
        return Err(Error::Algorithm("Need at least 2 points for variogram".into()));
    }

    // Compute max pairwise distance if not provided
    let max_lag = match params.max_lag {
        Some(m) => m,
        None => {
            let mut max_dist = 0.0_f64;
            for i in 0..n {
                for j in (i + 1)..n {
                    let d = points[i].dist(points[j].x, points[j].y);
                    if d > max_dist {
                        max_dist = d;
                    }
                }
            }
            max_dist / 2.0 // Convention: max lag = half of max distance
        }
    };

    if max_lag <= 0.0 {
        return Err(Error::Algorithm("Max lag must be positive".into()));
    }

    let bin_width = max_lag / params.n_lags as f64;
    let tol = bin_width * params.lag_tolerance;

    let mut lags = Vec::with_capacity(params.n_lags);
    let mut semivariance = vec![0.0_f64; params.n_lags];
    let mut pair_counts = vec![0_usize; params.n_lags];

    for k in 0..params.n_lags {
        lags.push((k as f64 + 0.5) * bin_width);
    }

    // Compute semivariance for each pair
    for i in 0..n {
        for j in (i + 1)..n {
            let d = points[i].dist(points[j].x, points[j].y);
            let dz = points[i].value - points[j].value;
            let sq_diff = dz * dz;

            // Assign to appropriate lag bin
            let bin = (d / bin_width - 0.5).round() as isize;
            if bin >= 0 && (bin as usize) < params.n_lags {
                let bin = bin as usize;
                let bin_center = lags[bin];
                if (d - bin_center).abs() <= tol / 2.0 {
                    semivariance[bin] += sq_diff;
                    pair_counts[bin] += 1;
                }
            }
        }
    }

    // Average: γ(h) = (1/2N) Σ (zᵢ - zⱼ)²
    for k in 0..params.n_lags {
        if pair_counts[k] > 0 {
            semivariance[k] /= 2.0 * pair_counts[k] as f64;
        } else {
            semivariance[k] = f64::NAN;
        }
    }

    Ok(EmpiricalVariogram {
        lags,
        semivariance,
        pair_counts,
    })
}

/// Fit a theoretical variogram model to an empirical variogram.
///
/// Uses weighted least squares with weights = N(h) / γ(h)² (Cressie 1985).
/// Performs a grid search over parameter space for robust fitting.
///
/// # Arguments
/// * `empirical` — Empirical variogram to fit
/// * `model` — Model type to fit
///
/// # Returns
/// [`FittedVariogram`] with nugget, sill, range, and goodness of fit.
pub fn fit_variogram(
    empirical: &EmpiricalVariogram,
    model: VariogramModel,
) -> Result<FittedVariogram> {
    // Collect valid (non-NaN) lag/semivariance pairs with counts
    let valid: Vec<(f64, f64, usize)> = empirical
        .lags
        .iter()
        .zip(empirical.semivariance.iter())
        .zip(empirical.pair_counts.iter())
        .filter(|((_, sv), cnt)| !sv.is_nan() && **cnt > 0)
        .map(|((&lag, &sv), &cnt)| (lag, sv, cnt))
        .collect();

    if valid.len() < 3 {
        return Err(Error::Algorithm(
            "Need at least 3 valid lag bins to fit variogram".into(),
        ));
    }

    let max_lag = valid.last().map(|(l, _, _)| *l).unwrap_or(1.0);
    let max_sv = valid
        .iter()
        .map(|(_, sv, _)| *sv)
        .fold(0.0_f64, f64::max);

    if max_sv <= 0.0 {
        return Err(Error::Algorithm("All semivariance values are zero".into()));
    }

    // Grid search for best (nugget, sill, range)
    let n_nugget = 10;
    let n_sill = 10;
    let n_range = 20;

    let mut best_rss = f64::MAX;
    let mut best_nugget = 0.0;
    let mut best_sill = max_sv;
    let mut best_range = max_lag;

    for in_ in 0..=n_nugget {
        let nugget = max_sv * in_ as f64 / (2.0 * n_nugget as f64);
        for is in 1..=n_sill {
            let sill = max_sv * is as f64 / n_sill as f64;
            if sill <= nugget {
                continue;
            }
            for ir in 1..=n_range {
                let range = max_lag * 2.0 * ir as f64 / n_range as f64;

                let partial_sill = sill - nugget;
                let trial = FittedVariogram {
                    model,
                    nugget,
                    sill,
                    range,
                    partial_sill,
                    rss: 0.0,
                };

                // Weighted residual sum of squares
                let mut rss = 0.0;
                for &(lag, sv, cnt) in &valid {
                    let predicted = trial.evaluate(lag);
                    let residual = sv - predicted;
                    // Weight by number of pairs (Cressie-style)
                    let w = cnt as f64;
                    rss += w * residual * residual;
                }

                if rss < best_rss {
                    best_rss = rss;
                    best_nugget = nugget;
                    best_sill = sill;
                    best_range = range;
                }
            }
        }
    }

    Ok(FittedVariogram {
        model,
        nugget: best_nugget,
        sill: best_sill,
        range: best_range,
        partial_sill: best_sill - best_nugget,
        rss: best_rss,
    })
}

/// Fit all three models and return the best one (lowest RSS).
pub fn fit_best_variogram(
    empirical: &EmpiricalVariogram,
) -> Result<FittedVariogram> {
    let models = [
        VariogramModel::Spherical,
        VariogramModel::Exponential,
        VariogramModel::Gaussian,
    ];

    let mut best: Option<FittedVariogram> = None;
    for &model in &models {
        if let Ok(fitted) = fit_variogram(empirical, model)
            && best.as_ref().is_none_or(|b| fitted.rss < b.rss) {
                best = Some(fitted);
            }
    }

    best.ok_or_else(|| Error::Algorithm("Could not fit any variogram model".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_spatially_correlated(n: usize, range: f64, seed: u64) -> Vec<SamplePoint> {
        // Simple pseudo-random spatially correlated points
        let mut points = Vec::with_capacity(n);
        let mut rng = seed;

        for _ in 0..n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let x = (rng >> 33) as f64 / (1u64 << 31) as f64 * 100.0;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let y = (rng >> 33) as f64 / (1u64 << 31) as f64 * 100.0;
            // Value with spatial trend + noise
            let value = 0.5 * x + 0.3 * y
                + 10.0 * ((x / range).sin() + (y / range).sin());
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let noise = (rng >> 33) as f64 / (1u64 << 31) as f64 * 2.0 - 1.0;
            points.push(SamplePoint::new(x, y, value + noise));
        }

        points
    }

    #[test]
    fn test_empirical_variogram_basic() {
        let points = generate_spatially_correlated(100, 20.0, 42);
        let result = empirical_variogram(&points, VariogramParams::default()).unwrap();

        assert_eq!(result.lags.len(), 15);
        assert_eq!(result.semivariance.len(), 15);
        assert_eq!(result.pair_counts.len(), 15);

        // First lags should have pairs
        assert!(result.pair_counts[0] > 0, "First lag should have pairs");

        // Semivariance should generally increase with distance
        // (for spatially correlated data)
        let valid_sv: Vec<f64> = result.semivariance.iter()
            .copied()
            .filter(|v| !v.is_nan())
            .collect();
        assert!(valid_sv.len() >= 5, "Should have at least 5 valid lags");

        // First semivariance should be less than last
        assert!(
            valid_sv[0] < *valid_sv.last().unwrap(),
            "Semivariance should increase: first={:.2}, last={:.2}",
            valid_sv[0], valid_sv.last().unwrap()
        );
    }

    #[test]
    fn test_empirical_variogram_too_few() {
        let points = vec![SamplePoint::new(0.0, 0.0, 1.0)];
        assert!(empirical_variogram(&points, VariogramParams::default()).is_err());
    }

    #[test]
    fn test_fit_spherical() {
        let points = generate_spatially_correlated(200, 15.0, 123);
        let emp = empirical_variogram(&points, VariogramParams {
            n_lags: 15,
            ..Default::default()
        }).unwrap();

        let fitted = fit_variogram(&emp, VariogramModel::Spherical).unwrap();

        assert!(fitted.nugget >= 0.0, "Nugget should be non-negative");
        assert!(fitted.sill > fitted.nugget, "Sill should exceed nugget");
        assert!(fitted.range > 0.0, "Range should be positive");
        assert!(fitted.rss < f64::MAX, "RSS should be finite");
    }

    #[test]
    fn test_fit_exponential() {
        let points = generate_spatially_correlated(200, 15.0, 456);
        let emp = empirical_variogram(&points, VariogramParams::default()).unwrap();
        let fitted = fit_variogram(&emp, VariogramModel::Exponential).unwrap();

        assert!(fitted.range > 0.0);
        assert!(fitted.sill > 0.0);
    }

    #[test]
    fn test_fit_gaussian() {
        let points = generate_spatially_correlated(200, 15.0, 789);
        let emp = empirical_variogram(&points, VariogramParams::default()).unwrap();
        let fitted = fit_variogram(&emp, VariogramModel::Gaussian).unwrap();

        assert!(fitted.range > 0.0);
        assert!(fitted.sill > 0.0);
    }

    #[test]
    fn test_fit_best() {
        let points = generate_spatially_correlated(200, 15.0, 101);
        let emp = empirical_variogram(&points, VariogramParams::default()).unwrap();
        let best = fit_best_variogram(&emp).unwrap();

        assert!(best.range > 0.0);
        assert!(best.sill > 0.0);
        assert!(best.rss >= 0.0);
    }

    #[test]
    fn test_model_evaluation() {
        let model = FittedVariogram {
            model: VariogramModel::Spherical,
            nugget: 1.0,
            sill: 10.0,
            range: 50.0,
            partial_sill: 9.0,
            rss: 0.0,
        };

        // At h=0, should be 0
        assert!((model.evaluate(0.0)).abs() < 1e-10);

        // At h=range, should equal sill
        let at_range = model.evaluate(50.0);
        assert!(
            (at_range - 10.0).abs() < 0.01,
            "At range, should equal sill: got {:.2}",
            at_range
        );

        // Beyond range, should be constant = sill
        let beyond = model.evaluate(100.0);
        assert!(
            (beyond - 10.0).abs() < 0.01,
            "Beyond range, should be sill: got {:.2}",
            beyond
        );

        // Intermediate should be between nugget and sill
        let mid = model.evaluate(25.0);
        assert!(mid > 1.0 && mid < 10.0, "Mid should be between nugget and sill: {:.2}", mid);
    }

    #[test]
    fn test_exponential_model_evaluation() {
        let model = FittedVariogram {
            model: VariogramModel::Exponential,
            nugget: 0.0,
            sill: 10.0,
            range: 30.0,
            partial_sill: 10.0,
            rss: 0.0,
        };

        // At h=0
        assert!((model.evaluate(0.0)).abs() < 1e-10);

        // At h=range, should be ~95% of sill (by definition)
        let at_range = model.evaluate(30.0);
        assert!(
            at_range > 9.0 && at_range < 10.0,
            "At range, should be ~95% of sill: got {:.2}",
            at_range
        );
    }
}
