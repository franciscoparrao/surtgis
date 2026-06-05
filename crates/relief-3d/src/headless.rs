//! Offscreen render path — no winit window, no canvas, no display.
//!
//! Same wgpu pipeline as native/web, but the colour target is a
//! `wgpu::Texture` instead of a `Surface`. The rendered pixels are
//! copied to a staging buffer, mapped, and returned as RGBA `Vec<u8>`.
//!
//! Use case: CLI screenshots, paper figures generated from a script,
//! CI relief regression checks. Runs anywhere wgpu can find an
//! adapter — desktop, headless servers (Vulkan/llvmpipe), even
//! `force_fallback_adapter: true` to opt into software rendering.

use bytemuck::cast_slice;
use surtgis_core::raster::Raster;
use surtgis_relief::RgbaImage;

use crate::camera::OrbitCamera;
use crate::mesh::from_dem;
use crate::pipeline::{build_pipeline, upload_rgba_texture};
use crate::{ReliefError, Result, Uniforms, sun_dir};

/// Headless render configuration. Wraps the runtime knobs in one
/// struct so the caller (CLI handler) does not have to juggle 8
/// positional arguments.
#[derive(Debug, Clone)]
pub struct HeadlessConfig {
    pub width: u32,
    pub height: u32,
    pub sun_azimuth_deg: f32,
    pub sun_altitude_deg: f32,
    pub ambient: f32,
    pub vertical_scale: f32,
    pub vertical_exaggeration: f32,
    pub camera_azimuth_deg: f32,
    pub camera_polar_deg: f32,
    pub camera_distance: f32,
    pub fov_deg: f32,
    /// Atmospheric haze density in [0, 1]. 0 = off (default).
    pub haze_density: f32,
    /// Haze colour (default light sky-grey-blue).
    pub haze_rgb: [f32; 3],
}

impl Default for HeadlessConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            sun_azimuth_deg: 315.0,
            sun_altitude_deg: 45.0,
            ambient: 0.4,
            vertical_scale: 1.0,
            vertical_exaggeration: 0.45,
            camera_azimuth_deg: 45.0,
            camera_polar_deg: 60.0,
            camera_distance: 3.2,
            fov_deg: 45.0,
            haze_density: 0.0,
            haze_rgb: [0.78, 0.83, 0.88],
        }
    }
}

/// Render the DEM + texture to an RGBA byte buffer. Returns
/// `width * height * 4` bytes in row-major order.
///
/// The PNG encoder lives in `surtgis-colormap` — call
/// `RgbaImage::from_rgba(width, height, bytes)?.save_png(path)` once
/// you have the bytes back.
pub fn render_to_rgba(
    dem: &Raster<f64>,
    texture: &RgbaImage,
    cfg: &HeadlessConfig,
) -> Result<Vec<u8>> {
    let (vertices, indices) = from_dem(dem, cfg.vertical_exaggeration);
    pollster::block_on(render_offscreen(
        &vertices,
        &indices,
        &texture.pixels,
        texture.width as u32,
        texture.height as u32,
        cfg,
    ))
}

async fn render_offscreen(
    vertices: &[crate::Vertex],
    indices: &[u32],
    rgba_pixels: &[u8],
    rgba_width: u32,
    rgba_height: u32,
    cfg: &HeadlessConfig,
) -> Result<Vec<u8>> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            // Allow llvmpipe / Lavapipe so CI / containers without a
            // discrete GPU still work — the perf will be lower but the
            // output is byte-equivalent for our simple shader.
            force_fallback_adapter: false,
        })
        .await
        .ok_or(ReliefError::NoAdapter)?;
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("relief3d.headless.device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults()
                    .using_resolution(adapter.limits()),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .map_err(|e| ReliefError::Device(e.to_string()))?;

    // The offscreen color target uses the *non-sRGB* RGBA format so
    // mapping back to `Vec<u8>` matches what users expect to feed into
    // a PNG encoder. The shader output is linear; if we used
    // Rgba8UnormSrgb here the sampled values would be re-gamma-encoded
    // and the PNG would look washed-out.
    let color_format = wgpu::TextureFormat::Rgba8Unorm;

    let color_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("relief3d.headless.color"),
        size: wgpu::Extent3d {
            width: cfg.width,
            height: cfg.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: color_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let texture_view = upload_rgba_texture(&device, &queue, rgba_pixels, rgba_width, rgba_height);
    let pipeline = build_pipeline(
        device,
        queue,
        color_format,
        vertices,
        indices,
        texture_view,
        (cfg.width, cfg.height),
    );

    // Build a one-shot uniform buffer with the requested camera + sun.
    let mut camera = OrbitCamera::default();
    camera.target = glam::Vec3::ZERO;
    camera.azimuth = cfg.camera_azimuth_deg.to_radians();
    camera.polar = cfg.camera_polar_deg.to_radians();
    camera.distance = cfg.camera_distance;
    camera.fov_deg = cfg.fov_deg;
    camera.aspect = cfg.width as f32 / cfg.height.max(1) as f32;

    let dir = sun_dir(cfg.sun_azimuth_deg, cfg.sun_altitude_deg);
    let uniforms = Uniforms {
        view_proj: camera.view_proj().to_cols_array_2d(),
        light_dir: [dir.x, dir.y, dir.z, 0.0],
        light_color: [1.0, 1.0, 1.0, cfg.ambient],
        vertical_scale: [cfg.vertical_scale, 0.0, 0.0, 0.0],
        fog_color: [
            cfg.haze_rgb[0],
            cfg.haze_rgb[1],
            cfg.haze_rgb[2],
            cfg.haze_density,
        ],
        fog_range: [1.5, 6.0, 0.0, 0.0],
    };
    pipeline
        .queue
        .write_buffer(&pipeline.uniform_buffer, 0, cast_slice(&[uniforms]));

    // Encode the render pass.
    let mut encoder = pipeline
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("relief3d.headless.encoder"),
        });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("relief3d.headless.pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.06,
                        g: 0.07,
                        b: 0.09,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &pipeline.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&pipeline.render_pipeline);
        pass.set_bind_group(0, &pipeline.bind_group, &[]);
        pass.set_vertex_buffer(0, pipeline.vertex_buffer.slice(..));
        pass.set_index_buffer(pipeline.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..pipeline.index_count, 0, 0..1);
    }

    // Copy the color texture into a row-aligned staging buffer.
    // wgpu requires `bytes_per_row` to be a multiple of 256.
    const ALIGN: u32 = 256;
    let bytes_per_pixel = 4u32;
    let unpadded_bpr = cfg.width * bytes_per_pixel;
    let padded_bpr = unpadded_bpr.div_ceil(ALIGN) * ALIGN;
    let buffer_size = (padded_bpr * cfg.height) as u64;

    let staging = pipeline.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("relief3d.headless.staging"),
        size: buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &color_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &staging,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded_bpr),
                rows_per_image: Some(cfg.height),
            },
        },
        wgpu::Extent3d {
            width: cfg.width,
            height: cfg.height,
            depth_or_array_layers: 1,
        },
    );

    pipeline.queue.submit(Some(encoder.finish()));

    // Map the staging buffer and copy into a tightly-packed output.
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    pipeline.device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| ReliefError::Device(format!("map channel: {e}")))?
        .map_err(|e| ReliefError::Device(format!("map_async: {e}")))?;

    let data = slice.get_mapped_range();
    let mut out = vec![0u8; (unpadded_bpr * cfg.height) as usize];
    for row in 0..cfg.height as usize {
        let src_start = row * padded_bpr as usize;
        let src_end = src_start + unpadded_bpr as usize;
        let dst_start = row * unpadded_bpr as usize;
        let dst_end = dst_start + unpadded_bpr as usize;
        out[dst_start..dst_end].copy_from_slice(&data[src_start..src_end]);
    }
    drop(data);
    staging.unmap();

    Ok(out)
}
