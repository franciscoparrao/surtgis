//! # `surtgis-relief-3d`
//!
//! Native wgpu 3D viewer for SurtGis shaded relief. Renders a DEM as a
//! displaced, textured mesh in a native window (winit) or a browser
//! canvas (WebGPU / WebGL2 fallback).
//!
//! Status: **M1 spike** — skeleton only. Builds a wgpu pipeline and
//! renders a 1024×1024 textured plane (1M vertices) to validate
//! production-workload performance. M2 wires real DEM heights and the
//! `surtgis-relief` texture; M3 adds lighting; M4 ships WASM/browser;
//! M5 adds headless screenshots. See `SPEC_SURTGIS_RELIEF_3D.md`.

use thiserror::Error;

pub mod camera;
pub mod lod;
pub mod mesh;
pub mod pipeline;

#[cfg(not(target_arch = "wasm32"))]
pub mod headless;

#[cfg(not(target_arch = "wasm32"))]
pub mod native;

#[cfg(target_arch = "wasm32")]
pub mod web;

#[derive(Debug, Error)]
pub enum ReliefError {
    #[error("wgpu adapter request failed: no compatible GPU found")]
    NoAdapter,
    #[error("wgpu device request failed: {0}")]
    Device(String),
    #[error("surface creation failed: {0}")]
    Surface(String),
    #[error("window event loop error: {0}")]
    EventLoop(String),
}

pub type Result<T> = std::result::Result<T, ReliefError>;

/// Vertex layout. Position + UV + normal — 8 floats per vertex.
/// Normals are computed at "baseline" vertical exaggeration; the shader
/// re-orients them per-frame when the user changes the runtime
/// `vertical_scale` uniform.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
    pub normal: [f32; 3],
}

impl Vertex {
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x2,
            },
            wgpu::VertexAttribute {
                offset: (std::mem::size_of::<[f32; 3]>() + std::mem::size_of::<[f32; 2]>())
                    as wgpu::BufferAddress,
                shader_location: 2,
                format: wgpu::VertexFormat::Float32x3,
            },
        ],
    };
}

/// Per-frame uniforms — 144 bytes, every field a vec4 slot to keep the
/// std140-equivalent layout obvious on both sides of the FFI:
///   view_proj            : mat4x4<f32>   (offset   0,  64 B)
///   light_dir.xyz        : vec3 in vec4  (offset  64,  16 B)  // direction TOWARDS light
///   light_color.xyz / .w : colour + amb. (offset  80,  16 B)
///   vertical_scale.x     : f32 in vec4   (offset  96,  16 B)
///   fog_color.xyz / .w   : colour + density [0,1] (offset 112, 16 B)
///   fog_range.x / .y     : near / far stops (offset 128, 16 B)
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    pub view_proj: [[f32; 4]; 4],
    pub light_dir: [f32; 4],
    pub light_color: [f32; 4],
    pub vertical_scale: [f32; 4],
    pub fog_color: [f32; 4],
    pub fog_range: [f32; 4],
}

impl Uniforms {
    /// Identity / neutral defaults. Light at azimuth 315°, altitude 45°
    /// (matching the rayshader recipe so the 3D light direction lines up
    /// with whatever was baked into the 2D texture). Fog density is 0
    /// so the M3 output is bit-equivalent to pre-P3 by default.
    pub fn identity() -> Self {
        let dir = sun_dir(315.0, 45.0);
        Self {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            light_dir: [dir.x, dir.y, dir.z, 0.0],
            light_color: [1.0, 1.0, 1.0, 0.4], // .w = ambient
            vertical_scale: [1.0, 0.0, 0.0, 0.0],
            // Neutral fog: light grey-blue colour that reads as sky,
            // density 0 so it does not affect the output unless the
            // viewer / CLI sets it. fog_range is in world-space units
            // — the mesh is normalised to longer-side = 2, so a
            // near/far of (1.5, 6.0) gives haze that ramps up from
            // "near the camera" to "the far horizon".
            fog_color: [0.78, 0.83, 0.88, 0.0],
            fog_range: [1.5, 6.0, 0.0, 0.0],
        }
    }
}

/// Unit vector pointing *from the surface toward the sun*. Convention
/// matches `RayShadeParams::with_soft_shadow_altitude` (azimuth 0=N,
/// clockwise; altitude 0=horizon, 90=zenith).
pub fn sun_dir(azimuth_deg: f32, altitude_deg: f32) -> glam::Vec3 {
    let az = azimuth_deg.to_radians();
    let alt = altitude_deg.to_radians();
    let cos_alt = alt.cos();
    glam::Vec3::new(cos_alt * az.sin(), alt.sin(), -cos_alt * az.cos()).normalize()
}
