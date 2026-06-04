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

pub mod mesh;
pub mod pipeline;

#[cfg(not(target_arch = "wasm32"))]
pub mod native;

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

/// Single vertex layout used by the M1 spike. M2 will extend this with
/// per-vertex normals computed from the DEM derivative.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
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
        ],
    };
}

/// `(view * projection)` matrix uploaded once per frame. Matches the
/// `Uniforms` struct in `shaders/relief.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    pub view_proj: [[f32; 4]; 4],
}
