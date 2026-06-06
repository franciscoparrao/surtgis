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

/// P4-M3b vertex compression. 16 bytes per vertex (half the f32 layout).
/// Used as the GPU-side storage format by `pipeline::build_pipeline`,
/// which converts `&[Vertex]` → `Vec<VertexC>` once at upload time.
///
/// Memory budget impact on the M2 spike (4 K × 4 K DEM with skirts,
/// 18.81 M vertices):
///
///   uncompressed: 18.81 M × 32 B = **602 MB**
///   compressed  : 18.81 M × 16 B = **301 MB**
///
/// 50 % reduction lets DEMs up to ~3 K side fit the typical 256 MB
/// WebGL2 single-buffer cap. 4 K still needs M3c lazy upload to fit.
///
/// Encoding:
///   - `pos` in `[-1, 1]` as snorm16 — the mesh builders normalise
///     positions to longer-side = 2 scene units so XZ always fit.
///     Y is clamped at ±1 in scene units (typical zex defaults give
///     y in `[0, 0.45]`).
///   - `uv` as unorm16 in `[0, 1]`.
///   - `normal` as snorm8 — ~1° angular precision, well below the
///     Lambertian shading threshold for visible artefacts.
#[repr(C, align(2))]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VertexC {
    pub pos: [i16; 3],
    pub _pad0: u16,
    pub uv: [u16; 2],
    pub normal: [i8; 3],
    pub _pad1: u8,
}

impl VertexC {
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            // pos+pad consumed as a vec4<f32> in [-1, 1]; the shader
            // takes .xyz.
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Snorm16x4,
            },
            wgpu::VertexAttribute {
                offset: 8,
                shader_location: 1,
                format: wgpu::VertexFormat::Unorm16x2,
            },
            // normal+pad consumed as a vec4<f32> in [-1, 1]; shader
            // takes .xyz.
            wgpu::VertexAttribute {
                offset: 12,
                shader_location: 2,
                format: wgpu::VertexFormat::Snorm8x4,
            },
        ],
    };

    /// Convert from a full-precision Vertex. Clamping is conservative
    /// — the encoder won't panic on out-of-range input, just saturates.
    pub fn from_vertex(v: &Vertex) -> Self {
        #[inline]
        fn s16(x: f32) -> i16 {
            (x.clamp(-1.0, 1.0) * 32767.0).round() as i16
        }
        #[inline]
        fn u16f(x: f32) -> u16 {
            (x.clamp(0.0, 1.0) * 65535.0).round() as u16
        }
        #[inline]
        fn s8(x: f32) -> i8 {
            (x.clamp(-1.0, 1.0) * 127.0).round() as i8
        }
        Self {
            pos: [s16(v.position[0]), s16(v.position[1]), s16(v.position[2])],
            _pad0: 0,
            uv: [u16f(v.uv[0]), u16f(v.uv[1])],
            normal: [s8(v.normal[0]), s8(v.normal[1]), s8(v.normal[2])],
            _pad1: 0,
        }
    }
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
