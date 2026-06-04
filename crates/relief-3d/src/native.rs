//! Native winit-driven window + event loop.
//!
//! M1 scope: create a window, set up the wgpu surface, render the
//! given mesh + texture at every redraw, log FPS once per second.
//! The FPS readout is what makes the M1 acceptance bar measurable
//! ("≥60 FPS at 1M vertices").

use std::time::Instant;

use bytemuck::cast_slice;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

use crate::pipeline::{ReliefPipeline, build_pipeline, make_checker_texture, make_depth};
use crate::{ReliefError, Result, Uniforms, Vertex};

/// Run a native window that renders the given vertices + indices with
/// the M1 spike pipeline (checker texture). Blocks until the user
/// closes the window. Returns immediately on event-loop error.
pub fn run_spike(vertices: Vec<Vertex>, indices: Vec<u32>, label: &str) -> Result<()> {
    let event_loop = EventLoop::new().map_err(|e| ReliefError::EventLoop(e.to_string()))?;
    let mut app = App {
        title: label.to_string(),
        window: None,
        state: None,
        pending: Some((vertices, indices)),
        last_log: Instant::now(),
        frames_since_log: 0,
    };
    event_loop
        .run_app(&mut app)
        .map_err(|e| ReliefError::EventLoop(e.to_string()))
}

struct App {
    title: String,
    window: Option<std::sync::Arc<Window>>,
    state: Option<RenderState>,
    pending: Option<(Vec<Vertex>, Vec<u32>)>,
    last_log: Instant,
    frames_since_log: u32,
}

struct RenderState {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pipeline: ReliefPipeline,
    angle: f32,
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
        let (vertices, indices) = self.pending.take().expect("spike data");
        let state = pollster::block_on(setup(window.clone(), &vertices, &indices))
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
                }
            }
            WindowEvent::RedrawRequested => {
                state.angle += 0.005;
                render(state);
                self.frames_since_log += 1;
                let now = Instant::now();
                let elapsed = now.duration_since(self.last_log).as_secs_f32();
                if elapsed >= 1.0 {
                    let fps = self.frames_since_log as f32 / elapsed;
                    eprintln!("relief-3d M1: {:.1} FPS", fps);
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

    let tex_view = make_checker_texture(&device, &queue);
    let pipeline = build_pipeline(
        device,
        queue,
        format,
        vertices,
        indices,
        tex_view,
        (config.width, config.height),
    );

    Ok(RenderState {
        surface,
        config,
        pipeline,
        angle: 0.0,
    })
}

fn render(state: &mut RenderState) {
    // Camera: orbit a couple of mesh-widths back, slowly rotating.
    let aspect = state.config.width as f32 / state.config.height.max(1) as f32;
    let proj = glam::Mat4::perspective_rh(45f32.to_radians(), aspect, 0.01, 100.0);
    let eye = glam::Vec3::new(2.5 * state.angle.cos(), 1.8, 2.5 * state.angle.sin());
    let view = glam::Mat4::look_at_rh(eye, glam::Vec3::ZERO, glam::Vec3::Y);
    let uniforms = Uniforms {
        view_proj: (proj * view).to_cols_array_2d(),
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
