//! Orbit camera with mouse interaction.
//!
//! The camera orbits a fixed target (the DEM centre) on a sphere
//! parametrised by polar/azimuth angles + distance. Left-drag rotates
//! (yaw + pitch), scroll-wheel zooms in/out, right-drag pans the
//! target laterally.
//!
//! `view_proj()` returns the matrix the vertex shader needs. The view
//! direction is `+Y` up to match the mesh built by [`crate::mesh`].

use glam::{Mat4, Vec3};

/// Orbit camera state. Mutate via [`OrbitCamera::rotate`],
/// [`OrbitCamera::zoom`], and [`OrbitCamera::pan`]; sample the matrix
/// with [`OrbitCamera::view_proj`].
#[derive(Debug, Clone)]
pub struct OrbitCamera {
    pub target: Vec3,
    /// Azimuth in radians (0 = +X, +π/2 = +Z when looking down +Y).
    pub azimuth: f32,
    /// Polar angle in radians from +Y. Clamped to (0, π) so the camera
    /// cannot flip through the pole.
    pub polar: f32,
    /// Distance from `target` in scene units.
    pub distance: f32,
    pub fov_deg: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            azimuth: std::f32::consts::FRAC_PI_4,
            polar: 1.0_f32,
            distance: 3.0,
            fov_deg: 45.0,
            aspect: 16.0 / 9.0,
            near: 0.01,
            far: 100.0,
        }
    }
}

impl OrbitCamera {
    /// Construct a camera that frames a unit-scale DEM (longer side =
    /// 2 scene units) from a comfortable mountain-DEM angle.
    pub fn for_dem() -> Self {
        Self {
            distance: 3.2,
            polar: 1.05,
            azimuth: std::f32::consts::FRAC_PI_4,
            ..Default::default()
        }
    }

    /// World-space camera position derived from spherical params.
    pub fn eye(&self) -> Vec3 {
        let sp = self.polar.sin();
        Vec3::new(
            self.target.x + self.distance * sp * self.azimuth.cos(),
            self.target.y + self.distance * self.polar.cos(),
            self.target.z + self.distance * sp * self.azimuth.sin(),
        )
    }

    pub fn view_proj(&self) -> Mat4 {
        let proj = Mat4::perspective_rh(
            self.fov_deg.to_radians(),
            self.aspect.max(1e-4),
            self.near,
            self.far,
        );
        let view = Mat4::look_at_rh(self.eye(), self.target, Vec3::Y);
        proj * view
    }

    /// Apply a mouse-drag delta in pixels. `(dx, dy)` is the screen
    /// movement; positive `dx` rotates the camera right around the
    /// target, positive `dy` rotates it down toward the pole.
    pub fn rotate(&mut self, dx: f32, dy: f32) {
        const PX_TO_RAD: f32 = 0.005;
        self.azimuth -= dx * PX_TO_RAD;
        self.polar = (self.polar + dy * PX_TO_RAD).clamp(0.05, std::f32::consts::PI - 0.05);
    }

    /// Multiplicative zoom. `factor > 1` zooms out, `< 1` zooms in.
    pub fn zoom(&mut self, factor: f32) {
        self.distance = (self.distance * factor).clamp(0.2, 50.0);
    }

    /// Pan the target in the camera's screen-space X / Y plane.
    /// Pixel deltas are scaled by `distance` so the world-space pan
    /// feels consistent at any zoom level.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        const PX_TO_WORLD: f32 = 0.0015;
        let eye = self.eye();
        let forward = (self.target - eye).normalize_or_zero();
        let right = forward.cross(Vec3::Y).normalize_or_zero();
        let up = right.cross(forward).normalize_or_zero();
        let delta = (-right * dx + up * dy) * PX_TO_WORLD * self.distance;
        self.target += delta;
    }

    pub fn set_aspect(&mut self, aspect: f32) {
        self.aspect = aspect;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_camera_frames_origin() {
        let cam = OrbitCamera::for_dem();
        let eye = cam.eye();
        assert!(eye.length() > 0.0);
        // Looking at origin from a positive direction.
        let view_dir = (cam.target - eye).normalize();
        assert!(view_dir.length() > 0.99);
    }

    #[test]
    fn rotate_changes_azimuth() {
        let mut cam = OrbitCamera::for_dem();
        let a0 = cam.azimuth;
        cam.rotate(100.0, 0.0);
        assert!((cam.azimuth - a0).abs() > 1e-3);
    }

    #[test]
    fn polar_clamped() {
        let mut cam = OrbitCamera::for_dem();
        cam.rotate(0.0, 100_000.0);
        assert!(cam.polar < std::f32::consts::PI);
        assert!(cam.polar > 0.0);
    }

    #[test]
    fn zoom_clamped() {
        let mut cam = OrbitCamera::for_dem();
        for _ in 0..200 {
            cam.zoom(0.5);
        }
        assert!(cam.distance >= 0.2);
        for _ in 0..200 {
            cam.zoom(2.0);
        }
        assert!(cam.distance <= 50.0);
    }
}
