//! Browser entry point for the 3D viewer.
//!
//! Mirrors `native.rs` but uses winit's `spawn_app` (browsers cannot
//! block on a `run_app` loop) and attaches the surface to a JS
//! `HTMLCanvasElement` instead of a winit-created OS window. Sun /
//! exaggeration controls live on the JS side and call back through
//! a `ReliefHandle` returned to JavaScript.
//!
//! Same wgpu pipeline and render code as the native build — the only
//! cfg-split is the windowing layer, per `SPEC_SURTGIS_RELIEF_3D.md` §3.

use std::cell::RefCell;
use std::rc::Rc;

use bytemuck::cast_slice;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::web::{EventLoopExtWebSys, WindowAttributesExtWebSys};
use winit::window::{Window, WindowId};

use surtgis_algorithms::terrain::HillshadeParams;
use surtgis_core::io::read_geotiff_from_buffer;
use surtgis_relief::{
    ColorScheme, RayShadeParams, ReliefBuilder, ambient_shade, ray_shade, sphere_shade,
};

use crate::camera::OrbitCamera;
use crate::mesh::from_dem;
use crate::pipeline::{ReliefPipeline, build_pipeline, make_depth, upload_rgba_texture};
use crate::{Uniforms, Vertex, sun_dir};

/// Install the panic hook once per page-load.
fn install_panic_hook() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        console_error_panic_hook::set_once();
    });
}

/// JS-facing handle. Wraps the shared lighting state — JS sliders mutate
/// it from outside the event loop without going through wgpu's queue.
#[wasm_bindgen]
pub struct ReliefHandle {
    lighting: Rc<RefCell<LightingShared>>,
}

#[derive(Clone, Copy)]
struct LightingShared {
    sun_azimuth_deg: f32,
    sun_altitude_deg: f32,
    ambient: f32,
    vertical_scale: f32,
}

impl Default for LightingShared {
    fn default() -> Self {
        Self {
            sun_azimuth_deg: 315.0,
            sun_altitude_deg: 45.0,
            ambient: 0.4,
            vertical_scale: 1.0,
        }
    }
}

#[wasm_bindgen]
impl ReliefHandle {
    pub fn set_sun(&self, azimuth_deg: f32, altitude_deg: f32) {
        let mut s = self.lighting.borrow_mut();
        s.sun_azimuth_deg = azimuth_deg.rem_euclid(360.0);
        s.sun_altitude_deg = altitude_deg.clamp(0.5, 89.5);
    }

    pub fn set_vertical_scale(&self, zex: f32) {
        self.lighting.borrow_mut().vertical_scale = zex.clamp(0.05, 10.0);
    }

    pub fn set_ambient(&self, ambient: f32) {
        self.lighting.borrow_mut().ambient = ambient.clamp(0.0, 1.0);
    }
}

/// Compute the rayshader-style relief composite for the given DEM bytes,
/// build the 3D mesh, attach a wgpu surface to `canvas_id`, and start the
/// render loop.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn run_relief3d_canvas(
    canvas_id: &str,
    tiff_bytes: &[u8],
    colormap: &str,
    sun_azimuth: f32,
    sun_altitude: f32,
    shadows: bool,
    ambient: bool,
    vertical_exaggeration: f32,
) -> Result<ReliefHandle, JsValue> {
    install_panic_hook();

    let scheme = match colormap.to_ascii_lowercase().as_str() {
        "divergent" => ColorScheme::Divergent,
        "grayscale" | "greyscale" => ColorScheme::Grayscale,
        "ndvi" => ColorScheme::Ndvi,
        "bwr" | "blue-white-red" => ColorScheme::BlueWhiteRed,
        "geomorphons" => ColorScheme::Geomorphons,
        "water" => ColorScheme::Water,
        "accumulation" => ColorScheme::Accumulation,
        _ => ColorScheme::Terrain,
    };

    let dem = read_geotiff_from_buffer::<f64>(tiff_bytes, None)
        .map_err(|e| JsValue::from_str(&format!("DEM decode: {e}")))?;
    let (rows, cols) = dem.shape();

    let sphere = sphere_shade(
        &dem,
        HillshadeParams {
            azimuth: sun_azimuth as f64,
            altitude: sun_altitude as f64,
            z_factor: 1.0,
            normalized: true,
        },
    )
    .map_err(|e| JsValue::from_str(&format!("sphere_shade: {e}")))?;
    let mut builder = ReliefBuilder::new(&dem)
        .base_colormap(scheme)
        .add_shade(sphere, 0.6);
    if shadows {
        let p = RayShadeParams::with_soft_shadow_altitude(
            sun_azimuth as f64,
            (sun_altitude as f64 - 5.0).max(0.5),
            (sun_altitude as f64 + 5.0).min(89.0),
            11,
        );
        let p = RayShadeParams {
            suns: p.suns,
            radius: rows.max(cols),
        };
        let shadow =
            ray_shade(&dem, &p).map_err(|e| JsValue::from_str(&format!("ray_shade: {e}")))?;
        builder = builder.add_shadow(shadow, 0.7);
    }
    if ambient {
        let ao = ambient_shade(&dem, 20)
            .map_err(|e| JsValue::from_str(&format!("ambient_shade: {e}")))?;
        builder = builder.add_ambient(ao, 0.3);
    }
    let img = builder
        .render()
        .map_err(|e| JsValue::from_str(&format!("compose: {e}")))?;

    let (vertices, indices) = from_dem(&dem, vertical_exaggeration);

    let document = web_sys::window()
        .and_then(|w| w.document())
        .ok_or_else(|| JsValue::from_str("no document"))?;
    let canvas: web_sys::HtmlCanvasElement = document
        .get_element_by_id(canvas_id)
        .ok_or_else(|| JsValue::from_str(&format!("canvas '{canvas_id}' not found")))?
        .dyn_into()
        .map_err(|_| JsValue::from_str("element is not an HTMLCanvasElement"))?;

    let lighting = Rc::new(RefCell::new(LightingShared::default()));
    let state_slot: Rc<RefCell<Option<RenderState>>> = Rc::new(RefCell::new(None));
    let app = App {
        canvas: Some(canvas),
        window: None,
        state_slot: state_slot.clone(),
        pending: Some((
            vertices,
            indices,
            img.pixels,
            img.width as u32,
            img.height as u32,
            lighting.clone(),
        )),
        mouse: MouseState::default(),
    };
    let event_loop = EventLoop::new().map_err(|e| JsValue::from_str(&e.to_string()))?;
    event_loop.spawn_app(app);
    Ok(ReliefHandle { lighting })
}

#[derive(Default)]
struct MouseState {
    left_pressed: bool,
    right_pressed: bool,
    last_pos: Option<(f64, f64)>,
}

struct App {
    canvas: Option<web_sys::HtmlCanvasElement>,
    window: Option<std::sync::Arc<Window>>,
    state_slot: Rc<RefCell<Option<RenderState>>>,
    pending: Option<(
        Vec<Vertex>,
        Vec<u32>,
        Vec<u8>,
        u32,
        u32,
        Rc<RefCell<LightingShared>>,
    )>,
    mouse: MouseState,
}

struct RenderState {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pipeline: ReliefPipeline,
    camera: OrbitCamera,
    lighting: Rc<RefCell<LightingShared>>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let canvas = self.canvas.take();
        let attrs = Window::default_attributes().with_canvas(canvas);
        let window = std::sync::Arc::new(
            event_loop
                .create_window(attrs)
                .expect("create_window failed"),
        );
        let (vertices, indices, pixels, width, height, lighting) =
            self.pending.take().expect("viewer payload");

        // Async wgpu setup: spawn_local fills the shared state_slot,
        // and the next redraw picks it up via state_slot.borrow().
        let window_clone = window.clone();
        let slot = self.state_slot.clone();
        wasm_bindgen_futures::spawn_local(async move {
            match setup(window_clone, &vertices, &indices, &pixels, width, height, lighting).await
            {
                Ok(s) => {
                    *slot.borrow_mut() = Some(s);
                }
                Err(e) => {
                    web_sys::console::error_1(&JsValue::from_str(&format!("setup: {e}")));
                }
            }
        });
        self.window = Some(window);
        self.canvas = None;
        // Kick the render loop — once state_slot fills, redraws produce frames.
        if let Some(w) = self.window.as_ref() {
            w.request_redraw();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let mut slot = self.state_slot.borrow_mut();
        let Some(state) = slot.as_mut() else {
            // Not yet initialised — only keep redraws ticking until setup finishes.
            if let WindowEvent::RedrawRequested = event {
                if let Some(w) = self.window.as_ref() {
                    w.request_redraw();
                }
            }
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
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => (p.y / 32.0) as f32,
                };
                let factor = (1.0_f32 - 0.1 * scroll).clamp(0.5, 2.0);
                state.camera.zoom(factor);
            }
            WindowEvent::RedrawRequested => {
                render(state);
                window.request_redraw();
            }
            _ => {}
        }
    }
}

fn log(msg: &str) {
    web_sys::console::log_1(&JsValue::from_str(msg));
}

async fn setup(
    window: std::sync::Arc<Window>,
    vertices: &[Vertex],
    indices: &[u32],
    rgba_pixels: &[u8],
    rgba_width: u32,
    rgba_height: u32,
    lighting: Rc<RefCell<LightingShared>>,
) -> Result<RenderState, String> {
    // WebGL2-only on the web. Trying WebGPU first sounds good in
    // theory but leaves the canvas in a half-attached state when the
    // browser can't satisfy it (Firefox, current Safari), so the
    // subsequent GL fallback fails with "no adapter". WebGL2 covers
    // every modern browser and matches our perf needs for ≤1M-vertex
    // DEMs comfortably.
    log("relief3d: instance.new (GL backend)");
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::GL,
        ..Default::default()
    });
    let surface: wgpu::Surface<'static> = instance
        .create_surface(window.clone())
        .map_err(|e| format!("create_surface: {e}"))?;
    log("relief3d: surface created, requesting adapter");
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .ok_or_else(|| "no adapter".to_string())?;
    log(&format!("relief3d: adapter {:?}", adapter.get_info()));
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("relief3d.device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits()),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .map_err(|e| format!("request_device: {e}"))?;

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
    log(&format!(
        "relief3d: configuring surface {}x{} format={:?}",
        config.width, config.height, format
    ));
    surface.configure(&device, &config);
    log("relief3d: surface configured, building pipeline");

    let tex_view = upload_rgba_texture(&device, &queue, rgba_pixels, rgba_width, rgba_height);
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

    log("relief3d: setup complete");
    Ok(RenderState {
        surface,
        config,
        pipeline,
        camera,
        lighting,
    })
}

fn render(state: &mut RenderState) {
    let lighting = state.lighting.borrow();
    let dir = sun_dir(lighting.sun_azimuth_deg, lighting.sun_altitude_deg);
    let uniforms = Uniforms {
        view_proj: state.camera.view_proj().to_cols_array_2d(),
        light_dir: [dir.x, dir.y, dir.z, 0.0],
        light_color: [1.0, 1.0, 1.0, lighting.ambient],
        vertical_scale: [lighting.vertical_scale, 0.0, 0.0, 0.0],
    };
    drop(lighting);
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
