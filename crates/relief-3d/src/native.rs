//! Native winit-driven window + event loop.
//!
//! Two public entry points:
//!   - [`run_spike`]   — M1 acceptance: textured mesh + auto-orbit
//!     camera + FPS log. The texture is an in-tree checker pattern.
//!   - [`run_viewer`]  — M2 viewer: textured mesh + interactive
//!     OrbitCamera (left-drag rotate, scroll zoom, right-drag pan).
//!     The texture is supplied as a row-major `Vec<u8>` RGBA buffer
//!     of size `width * height * 4`, typically the output of
//!     `surtgis-relief`.

use std::time::Instant;

use bytemuck::cast_slice;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::camera::OrbitCamera;
use crate::pipeline::{
    ReliefPipeline, build_pipeline, make_checker_texture, make_depth, upload_rgba_texture,
};
use crate::{ReliefError, Result, Uniforms, Vertex, sun_dir};

/// Mutable lighting/displacement state owned by the viewer. Read at
/// every frame to build the per-frame `Uniforms`.
pub struct LightingState {
    pub sun_azimuth_deg: f32,
    pub sun_altitude_deg: f32,
    pub ambient: f32,
    pub light_rgb: [f32; 3],
    pub vertical_scale: f32,
    /// Atmospheric haze density in [0, 1]. 0 disables the effect.
    pub haze_density: f32,
    /// Haze colour (defaults to a light sky-grey-blue).
    pub haze_rgb: [f32; 3],
    /// Haze near-stop in world units (mesh longer side = 2). Cells
    /// closer than this stay fully lit.
    pub haze_near: f32,
    /// Haze far-stop in world units. Cells beyond this are fully fogged.
    pub haze_far: f32,
}

impl Default for LightingState {
    fn default() -> Self {
        Self {
            sun_azimuth_deg: 315.0,
            sun_altitude_deg: 45.0,
            ambient: 0.4,
            light_rgb: [1.0, 1.0, 1.0],
            vertical_scale: 1.0,
            haze_density: 0.0,
            haze_rgb: [0.78, 0.83, 0.88],
            haze_near: 1.5,
            haze_far: 6.0,
        }
    }
}

/// Texture source for the viewer. M1 ships a procedural checker; M2's
/// `render_dem` example wraps a `surtgis-relief` RGBA buffer.
pub enum TextureSource {
    Checker,
    Rgba {
        pixels: Vec<u8>,
        width: u32,
        height: u32,
    },
}

/// Whether the camera orbits automatically (M1 spike) or responds to
/// mouse input (M2 viewer).
pub enum CameraMode {
    AutoOrbit,
    Interactive,
}

/// M1 helper: run the spike with auto-orbit + checker texture.
pub fn run_spike(vertices: Vec<Vertex>, indices: Vec<u32>, label: &str) -> Result<()> {
    run_inner(
        vertices,
        indices,
        TextureSource::Checker,
        CameraMode::AutoOrbit,
        label,
        None,
    )
}

/// M2 entry point: run the interactive viewer with the given texture.
pub fn run_viewer(
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    rgba_pixels: Vec<u8>,
    width: u32,
    height: u32,
    label: &str,
) -> Result<()> {
    run_inner(
        vertices,
        indices,
        TextureSource::Rgba {
            pixels: rgba_pixels,
            width,
            height,
        },
        CameraMode::Interactive,
        label,
        None,
    )
}

/// P4 entry point: run the viewer over a [`crate::lod::QuadtreeMesh`]
/// instead of a flat mesh. Each frame the viewer culls + selects a
/// LOD per chunk; only surviving chunks contribute draw calls.
///
/// Pass a checker pattern (`TextureSource::Checker`) for the spike
/// example; M3+ wires the real `surtgis-relief` texture in.
pub fn run_lod_viewer(
    mesh: crate::lod::QuadtreeMesh,
    lod_params: crate::lod::LodParams,
    texture: TextureSource,
    label: &str,
) -> Result<()> {
    run_lod_viewer_with_mode(mesh, lod_params, texture, label, CameraMode::Interactive)
}

/// Same as [`run_lod_viewer`] but lets the caller force auto-orbit.
/// Useful for the M1 spike where a deterministic camera motion makes
/// the FPS log reproducible.
pub fn run_lod_viewer_with_mode(
    mesh: crate::lod::QuadtreeMesh,
    lod_params: crate::lod::LodParams,
    texture: TextureSource,
    label: &str,
    mode: CameraMode,
) -> Result<()> {
    let vertices = mesh.vertices.clone();
    let indices = mesh.indices.clone();
    run_inner(
        vertices,
        indices,
        texture,
        mode,
        label,
        Some((mesh, lod_params)),
    )
}

fn run_inner(
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    texture: TextureSource,
    mode: CameraMode,
    label: &str,
    lod: Option<(crate::lod::QuadtreeMesh, crate::lod::LodParams)>,
) -> Result<()> {
    let event_loop = EventLoop::new().map_err(|e| ReliefError::EventLoop(e.to_string()))?;
    let mut app = App {
        title: label.to_string(),
        window: None,
        state: None,
        pending: Some((vertices, indices, texture, mode, lod)),
        last_log: Instant::now(),
        frames_since_log: 0,
        mouse: MouseState::default(),
    };
    event_loop
        .run_app(&mut app)
        .map_err(|e| ReliefError::EventLoop(e.to_string()))
}

#[derive(Default)]
struct MouseState {
    left_pressed: bool,
    right_pressed: bool,
    last_pos: Option<(f64, f64)>,
}

struct App {
    title: String,
    window: Option<std::sync::Arc<Window>>,
    state: Option<RenderState>,
    pending: Option<(
        Vec<Vertex>,
        Vec<u32>,
        TextureSource,
        CameraMode,
        Option<(crate::lod::QuadtreeMesh, crate::lod::LodParams)>,
    )>,
    last_log: Instant,
    frames_since_log: u32,
    mouse: MouseState,
}

struct RenderState {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pipeline: ReliefPipeline,
    /// Input-driven camera (mouse handlers write here).
    camera: OrbitCamera,
    /// Frame-time damped copy of `camera` (used to build the view-proj).
    /// Lerps toward `camera` with tau = `DAMP_TAU_S` seconds, giving
    /// the camera a critically-damped feel and a brief inertia tail
    /// after the user releases the mouse.
    camera_smooth: OrbitCamera,
    mode: CameraMode,
    auto_angle: f32,
    lighting: LightingState,
    /// `Instant` of the previous redraw — needed for the damping `dt`.
    last_frame: Instant,
    /// Set true when the user presses `S`; render() consumes it.
    screenshot_pending: bool,
    /// Optional P4 quadtree LOD. When present, render() culls and
    /// draws per chunk; when None, render() does a single full-mesh
    /// draw against `pipeline.index_count`.
    lod: Option<(crate::lod::QuadtreeMesh, crate::lod::LodParams)>,
    /// Last-frame stats — counted by the LOD render path for the FPS
    /// log so a user can tell at a glance whether culling is doing
    /// any work.
    lod_drawn_chunks: u32,
    lod_total_chunks: u32,
}

const DAMP_TAU_S: f32 = 0.08;

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title(&self.title)
            .with_inner_size(LogicalSize::new(1280.0, 800.0));
        let window = std::sync::Arc::new(
            event_loop
                .create_window(attrs)
                .expect("create window failed"),
        );
        let (vertices, indices, texture, mode, lod) = self.pending.take().expect("viewer data");
        let state = pollster::block_on(setup(
            window.clone(),
            &vertices,
            &indices,
            texture,
            mode,
            lod,
        ))
        .expect("wgpu setup failed");
        self.window = Some(window);
        self.state = Some(state);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        let Some(window) = self.window.as_ref() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(new_size) => {
                if new_size.width > 0 && new_size.height > 0 {
                    state.config.width = new_size.width;
                    state.config.height = new_size.height;
                    state
                        .surface
                        .configure(&state.pipeline.device, &state.config);
                    state.pipeline.depth_view =
                        make_depth(&state.pipeline.device, new_size.width, new_size.height);
                    state.pipeline.depth_size = (new_size.width, new_size.height);
                    let aspect = new_size.width as f32 / new_size.height.max(1) as f32;
                    state.camera.set_aspect(aspect);
                    state.camera_smooth.set_aspect(aspect);
                }
            }
            WindowEvent::MouseInput {
                button, state: bs, ..
            } => match button {
                MouseButton::Left => self.mouse.left_pressed = bs == ElementState::Pressed,
                MouseButton::Right => self.mouse.right_pressed = bs == ElementState::Pressed,
                _ => {}
            },
            WindowEvent::CursorMoved { position, .. } => {
                let pos = (position.x, position.y);
                if let Some((px, py)) = self.mouse.last_pos {
                    let dx = (pos.0 - px) as f32;
                    let dy = (pos.1 - py) as f32;
                    if self.mouse.left_pressed {
                        state.camera.rotate(dx, dy);
                    }
                    if self.mouse.right_pressed {
                        state.camera.pan(dx, dy);
                    }
                }
                self.mouse.last_pos = Some(pos);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => handle_key(key, state),
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => (p.y / 32.0) as f32,
                };
                // Negative scroll (towards user) zooms out by convention.
                let factor = (1.0_f32 - 0.1 * scroll).clamp(0.5, 2.0);
                state.camera.zoom(factor);
            }
            WindowEvent::RedrawRequested => {
                if matches!(state.mode, CameraMode::AutoOrbit) {
                    state.auto_angle += 0.005;
                    state.camera.azimuth = state.auto_angle;
                }
                render(state);
                self.frames_since_log += 1;
                let now = Instant::now();
                let elapsed = now.duration_since(self.last_log).as_secs_f32();
                if elapsed >= 1.0 {
                    let fps = self.frames_since_log as f32 / elapsed;
                    if state.lod.is_some() {
                        eprintln!(
                            "relief-3d: {:.1} FPS — chunks {}/{} visible",
                            fps, state.lod_drawn_chunks, state.lod_total_chunks
                        );
                    } else {
                        eprintln!("relief-3d: {:.1} FPS", fps);
                    }
                    self.frames_since_log = 0;
                    self.last_log = now;
                }
                window.request_redraw();
            }
            _ => {}
        }
    }
}

async fn setup(
    window: std::sync::Arc<Window>,
    vertices: &[Vertex],
    indices: &[u32],
    texture: TextureSource,
    mode: CameraMode,
    lod: Option<(crate::lod::QuadtreeMesh, crate::lod::LodParams)>,
) -> Result<RenderState> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let surface: wgpu::Surface<'static> = instance
        .create_surface(window.clone())
        .map_err(|e| ReliefError::Surface(e.to_string()))?;
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .ok_or(ReliefError::NoAdapter)?;
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("relief3d.device"),
                required_features: wgpu::Features::empty(),
                required_limits: {
                    // The downlevel default caps `max_buffer_size` at
                    // 256 MB, which a 4 K × 4 K mesh blows through
                    // (~537 MB vertex buffer). Most desktop GPUs report
                    // multi-GB caps in `adapter.limits().max_buffer_size`,
                    // so we let `using_resolution` widen the limit to
                    // whatever the hardware actually supports. WASM
                    // (web.rs) keeps a tighter ceiling because WebGL2
                    // caps are far lower.
                    let mut limits =
                        wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits());
                    limits.max_buffer_size = adapter.limits().max_buffer_size;
                    limits
                },
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .map_err(|e| ReliefError::Device(e.to_string()))?;

    let size = window.inner_size();
    let caps = surface.get_capabilities(&adapter);
    let format = caps
        .formats
        .iter()
        .copied()
        .find(|f| f.is_srgb())
        .unwrap_or(caps.formats[0]);
    let config = wgpu::SurfaceConfiguration {
        // COPY_SRC lets the M3.3 screenshot key copy the presented frame
        // into a staging buffer. Cost: zero on every backend we target.
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        format,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &config);

    let tex_view = match texture {
        TextureSource::Checker => make_checker_texture(&device, &queue),
        TextureSource::Rgba {
            pixels,
            width,
            height,
        } => upload_rgba_texture(&device, &queue, &pixels, width, height),
    };
    let pipeline = build_pipeline(
        device,
        queue,
        format,
        vertices,
        indices,
        tex_view,
        (config.width, config.height),
    );

    let mut camera = OrbitCamera::for_dem();
    camera.set_aspect(size.width as f32 / size.height.max(1) as f32);
    let camera_smooth = camera.clone();

    Ok(RenderState {
        surface,
        config,
        pipeline,
        camera,
        camera_smooth,
        mode,
        auto_angle: 0.0,
        lighting: LightingState::default(),
        last_frame: Instant::now(),
        screenshot_pending: false,
        lod,
        lod_drawn_chunks: 0,
        lod_total_chunks: 0,
    })
}

fn handle_key(key: KeyCode, state: &mut RenderState) {
    let lighting = &mut state.lighting;
    match key {
        // Vertical exaggeration: + / =  and  - / _
        KeyCode::Equal | KeyCode::NumpadAdd => {
            lighting.vertical_scale = (lighting.vertical_scale + 0.1).min(5.0);
        }
        KeyCode::Minus | KeyCode::NumpadSubtract => {
            lighting.vertical_scale = (lighting.vertical_scale - 0.1).max(0.1);
        }
        // Sun azimuth: [ / ]
        KeyCode::BracketLeft => {
            lighting.sun_azimuth_deg = (lighting.sun_azimuth_deg - 10.0).rem_euclid(360.0)
        }
        KeyCode::BracketRight => {
            lighting.sun_azimuth_deg = (lighting.sun_azimuth_deg + 10.0).rem_euclid(360.0)
        }
        // Sun altitude: ; / '
        KeyCode::Semicolon => {
            lighting.sun_altitude_deg = (lighting.sun_altitude_deg - 5.0).max(5.0)
        }
        KeyCode::Quote => lighting.sun_altitude_deg = (lighting.sun_altitude_deg + 5.0).min(89.0),
        // Ambient up/down: , / .
        KeyCode::Comma => lighting.ambient = (lighting.ambient - 0.05).max(0.0),
        KeyCode::Period => lighting.ambient = (lighting.ambient + 0.05).min(1.0),
        // Haze (P3-M1): H toggles, F decreases density, G increases.
        KeyCode::KeyH => {
            lighting.haze_density = if lighting.haze_density > 0.0 {
                0.0
            } else {
                0.55
            };
        }
        KeyCode::KeyF => lighting.haze_density = (lighting.haze_density - 0.05).max(0.0),
        KeyCode::KeyG => lighting.haze_density = (lighting.haze_density + 0.05).min(1.0),
        // P3-M3.2 help. Slash + shift = `?` on most layouts; print
        // on both Slash and the dedicated `?` key (no separate
        // PhysicalKey for `?`, so Slash handles it).
        KeyCode::Slash => {
            print_help();
            return;
        }
        // P3-M3.3 screenshot.
        KeyCode::KeyS => {
            state.screenshot_pending = true;
            eprintln!("screenshot: queued");
            return;
        }
        _ => return,
    }
    eprintln!(
        "lighting: zex={:.1} sun=({:.0}°, {:.0}°) ambient={:.2} haze={:.2}",
        lighting.vertical_scale,
        lighting.sun_azimuth_deg,
        lighting.sun_altitude_deg,
        lighting.ambient,
        lighting.haze_density
    );
}

/// Keybindings overlay printed to stderr on `?` press. M3.2 acceptance
/// bar is "first 60 s user sees the controls"; the visual on-window
/// overlay is deferred behind a font dep.
fn print_help() {
    eprintln!();
    eprintln!("── surtgis-relief-3d controls ──");
    eprintln!("  mouse drag        rotate (orbit)");
    eprintln!("  right-mouse drag  pan");
    eprintln!("  scroll wheel      zoom");
    eprintln!();
    eprintln!("  =  / -            vertical exaggeration  +0.1 / -0.1");
    eprintln!("  [  / ]            sun azimuth            -10 / +10  deg");
    eprintln!("  ;  / '            sun altitude           -5  / +5   deg");
    eprintln!("  ,  / .            ambient term           -0.05 / +0.05");
    eprintln!("  H                 toggle atmospheric haze (0 ↔ 0.55)");
    eprintln!("  F  / G            haze density           -0.05 / +0.05");
    eprintln!();
    eprintln!("  S                 screenshot → relief3d-<ts>.png");
    eprintln!("  ?                 print this help");
    eprintln!("  close window      exit");
    eprintln!();
}

/// Exponential smoothing toward a target. `factor = 1 - exp(-dt / tau)`.
/// At tau = 80 ms, ~63 % closes in 80 ms, ~95 % in 240 ms — feels like
/// "critical damping with a brief inertia tail" without an explicit
/// velocity term.
fn damp(current: f32, target: f32, factor: f32) -> f32 {
    current + (target - current) * factor
}

fn damp_camera(smooth: &mut OrbitCamera, target: &OrbitCamera, dt: f32) {
    let factor = 1.0 - (-dt / DAMP_TAU_S).exp();
    // Azimuth wraps; lerp through the short arc.
    let mut da = target.azimuth - smooth.azimuth;
    let two_pi = std::f32::consts::TAU;
    if da > std::f32::consts::PI {
        da -= two_pi;
    } else if da < -std::f32::consts::PI {
        da += two_pi;
    }
    smooth.azimuth += da * factor;
    smooth.polar = damp(smooth.polar, target.polar, factor);
    smooth.distance = damp(smooth.distance, target.distance, factor);
    smooth.target.x = damp(smooth.target.x, target.target.x, factor);
    smooth.target.y = damp(smooth.target.y, target.target.y, factor);
    smooth.target.z = damp(smooth.target.z, target.target.z, factor);
    smooth.aspect = target.aspect;
    smooth.fov_deg = target.fov_deg;
    smooth.near = target.near;
    smooth.far = target.far;
}

fn render(state: &mut RenderState) {
    let now = Instant::now();
    let dt = now.duration_since(state.last_frame).as_secs_f32().min(0.1);
    state.last_frame = now;
    damp_camera(&mut state.camera_smooth, &state.camera, dt);

    let lighting = &state.lighting;
    let dir = sun_dir(lighting.sun_azimuth_deg, lighting.sun_altitude_deg);
    let uniforms = Uniforms {
        view_proj: state.camera_smooth.view_proj().to_cols_array_2d(),
        light_dir: [dir.x, dir.y, dir.z, 0.0],
        light_color: [
            lighting.light_rgb[0],
            lighting.light_rgb[1],
            lighting.light_rgb[2],
            lighting.ambient,
        ],
        vertical_scale: [lighting.vertical_scale, 0.0, 0.0, 0.0],
        fog_color: [
            lighting.haze_rgb[0],
            lighting.haze_rgb[1],
            lighting.haze_rgb[2],
            lighting.haze_density,
        ],
        fog_range: [lighting.haze_near, lighting.haze_far, 0.0, 0.0],
    };
    state
        .pipeline
        .queue
        .write_buffer(&state.pipeline.uniform_buffer, 0, cast_slice(&[uniforms]));

    let frame = match state.surface.get_current_texture() {
        Ok(frame) => frame,
        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
            state
                .surface
                .configure(&state.pipeline.device, &state.config);
            return;
        }
        Err(_) => return,
    };
    let view_color = frame
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder =
        state
            .pipeline
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("relief3d.encoder"),
            });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("relief3d.pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view_color,
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
                view: &state.pipeline.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&state.pipeline.render_pipeline);
        pass.set_bind_group(0, &state.pipeline.bind_group, &[]);
        pass.set_vertex_buffer(0, state.pipeline.vertex_buffer.slice(..));
        pass.set_index_buffer(
            state.pipeline.index_buffer.slice(..),
            wgpu::IndexFormat::Uint32,
        );

        if let Some((mesh, params)) = &state.lod {
            // P4 LOD path. Cull + select per chunk, then issue one
            // draw call per surviving chunk. Stats roll up for the
            // FPS log line.
            let view_proj = state.camera_smooth.view_proj();
            let camera_pos = state.camera_smooth.eye();
            let visible = mesh.select(view_proj, camera_pos, params);
            state.lod_drawn_chunks = visible.len() as u32;
            state.lod_total_chunks = mesh.chunks.len() as u32;
            for (chunk_idx, lod_idx) in visible {
                let r = &mesh.chunks[chunk_idx].lod_indices[lod_idx];
                pass.draw_indexed(r.clone(), 0, 0..1);
            }
        } else {
            pass.draw_indexed(0..state.pipeline.index_count, 0, 0..1);
        }
    }

    // P3-M3.3 screenshot: copy the just-rendered frame into a staging
    // buffer before present/submit. Mapping + PNG-write happens after
    // submit so the GPU has finished by then.
    let screenshot_buffer = if state.screenshot_pending {
        Some(queue_screenshot_copy(state, &mut encoder, &frame))
    } else {
        None
    };

    state.pipeline.queue.submit(Some(encoder.finish()));
    frame.present();

    if let Some((staging, width, height, padded_bpr, format)) = screenshot_buffer {
        state.screenshot_pending = false;
        save_screenshot(
            &state.pipeline.device,
            &staging,
            width,
            height,
            padded_bpr,
            format,
        );
    }
}

/// Encode `copy_texture_to_buffer` into the existing render encoder so
/// the screenshot reads the same frame that gets presented. Returns the
/// staging buffer + the metadata `save_screenshot` needs after submit.
fn queue_screenshot_copy(
    state: &RenderState,
    encoder: &mut wgpu::CommandEncoder,
    frame: &wgpu::SurfaceTexture,
) -> (wgpu::Buffer, u32, u32, u32, wgpu::TextureFormat) {
    let width = state.config.width;
    let height = state.config.height;
    let format = state.config.format;
    const ALIGN: u32 = 256;
    let unpadded_bpr = width * 4;
    let padded_bpr = unpadded_bpr.div_ceil(ALIGN) * ALIGN;
    let buffer_size = (padded_bpr * height) as u64;

    let staging = state
        .pipeline
        .device
        .create_buffer(&wgpu::BufferDescriptor {
            label: Some("relief3d.screenshot.staging"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &frame.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &staging,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded_bpr),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    (staging, width, height, padded_bpr, format)
}

/// Block on the staging buffer, strip row padding, swap channels into
/// RGBA if the surface uses BGRA, and write a timestamped PNG in cwd.
fn save_screenshot(
    device: &wgpu::Device,
    staging: &wgpu::Buffer,
    width: u32,
    height: u32,
    padded_bpr: u32,
    format: wgpu::TextureFormat,
) {
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device.poll(wgpu::Maintain::Wait);
    if rx.recv().is_err() || matches!(rx.try_recv(), Ok(Err(_))) {
        eprintln!("screenshot: buffer map failed");
        return;
    }

    let data = slice.get_mapped_range();
    let unpadded_bpr = (width * 4) as usize;
    let mut rgba = vec![0u8; unpadded_bpr * height as usize];
    for row in 0..height as usize {
        let src_start = row * padded_bpr as usize;
        let dst_start = row * unpadded_bpr;
        rgba[dst_start..dst_start + unpadded_bpr]
            .copy_from_slice(&data[src_start..src_start + unpadded_bpr]);
    }
    drop(data);
    staging.unmap();

    // Surface format on most desktops is Bgra8UnormSrgb; PNG encoders
    // want RGBA. Swap the R and B channels in place.
    if matches!(
        format,
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    ) {
        for px in rgba.chunks_exact_mut(4) {
            px.swap(0, 2);
        }
    }

    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let path = format!("relief3d-{secs}.png");
    match surtgis_colormap::rgba_to_png_bytes(width, height, &rgba) {
        Ok(bytes) => {
            if let Err(e) = std::fs::write(&path, bytes) {
                eprintln!("screenshot: write {path}: {e}");
            } else {
                eprintln!("screenshot: wrote {path} ({width}×{height})");
            }
        }
        Err(e) => eprintln!("screenshot: encode {path}: {e}"),
    }
}
