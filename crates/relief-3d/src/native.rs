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
    )
}

fn run_inner(
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    texture: TextureSource,
    mode: CameraMode,
    label: &str,
) -> Result<()> {
    let event_loop = EventLoop::new().map_err(|e| ReliefError::EventLoop(e.to_string()))?;
    let mut app = App {
        title: label.to_string(),
        window: None,
        state: None,
        pending: Some((vertices, indices, texture, mode)),
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
    pending: Option<(Vec<Vertex>, Vec<u32>, TextureSource, CameraMode)>,
    last_log: Instant,
    frames_since_log: u32,
    mouse: MouseState,
}

struct RenderState {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pipeline: ReliefPipeline,
    camera: OrbitCamera,
    mode: CameraMode,
    auto_angle: f32,
    lighting: LightingState,
}

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
        let (vertices, indices, texture, mode) = self.pending.take().expect("viewer data");
        let state = pollster::block_on(setup(window.clone(), &vertices, &indices, texture, mode))
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
                    state
                        .camera
                        .set_aspect(new_size.width as f32 / new_size.height.max(1) as f32);
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
                    eprintln!("relief-3d: {:.1} FPS", fps);
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
                required_limits: wgpu::Limits::downlevel_defaults()
                    .using_resolution(adapter.limits()),
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
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
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

    Ok(RenderState {
        surface,
        config,
        pipeline,
        camera,
        mode,
        auto_angle: 0.0,
        lighting: LightingState::default(),
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

fn render(state: &mut RenderState) {
    let lighting = &state.lighting;
    let dir = sun_dir(lighting.sun_azimuth_deg, lighting.sun_altitude_deg);
    let uniforms = Uniforms {
        view_proj: state.camera.view_proj().to_cols_array_2d(),
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
        pass.draw_indexed(0..state.pipeline.index_count, 0, 0..1);
    }
    state.pipeline.queue.submit(Some(encoder.finish()));
    frame.present();
}
