//! wgpu device + render pipeline construction.
//!
//! M1 spike scope: single pipeline that renders a textured mesh. M3 will
//! add a uniform for the light direction and extend the shader to do
//! per-fragment Lambertian shading.

use crate::{Uniforms, Vertex};
use bytemuck::cast_slice;
use wgpu::util::DeviceExt;

/// All the GPU state needed to render one textured mesh. Built once,
/// mutated only when the texture changes (M2+) or the viewport
/// resizes (handled by `native::App::resize`).
pub struct ReliefPipeline {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub render_pipeline: wgpu::RenderPipeline,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub depth_view: wgpu::TextureView,
    pub depth_size: (u32, u32),
}

/// Upload an arbitrary RGBA byte buffer as a wgpu texture and return its
/// view. `pixels.len()` must equal `width * height * 4`; the panic on
/// mismatch is the right policy because the buffer comes from
/// `surtgis-relief`, which already guarantees the layout.
pub fn upload_rgba_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pixels: &[u8],
    width: u32,
    height: u32,
) -> wgpu::TextureView {
    assert_eq!(
        pixels.len(),
        (width * height * 4) as usize,
        "rgba upload: pixels.len()={} expected={}",
        pixels.len(),
        width * height * 4
    );

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("relief3d.rgba"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        pixels,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Single-channel 256×256 checkerboard upload for the M1 spike. M2's
/// `render_dem` example uses [`upload_rgba_texture`] with the
/// `surtgis-relief` output instead.
pub fn make_checker_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::TextureView {
    const N: u32 = 256;
    const CHECK: u32 = 32;
    let mut pixels = vec![0u8; (N * N * 4) as usize];
    for y in 0..N {
        for x in 0..N {
            let i = ((y * N + x) * 4) as usize;
            let on = ((x / CHECK) + (y / CHECK)) % 2 == 0;
            let c = if on { 255u8 } else { 64u8 };
            pixels[i] = c;
            pixels[i + 1] = c;
            pixels[i + 2] = (c / 2).saturating_add(64);
            pixels[i + 3] = 255;
        }
    }
    upload_rgba_texture(device, queue, &pixels, N, N)
}

/// Build a depth buffer view for the swapchain. Re-created on resize.
pub fn make_depth(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("relief3d.depth"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth24Plus,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Build a ReliefPipeline given the device/queue plus a vertex/index
/// pair and a texture view to sample. The bind-group layout
/// (uniform + texture + sampler) is internal to this function so
/// callers don't have to thread it around.
#[allow(clippy::too_many_arguments)]
pub fn build_pipeline(
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_format: wgpu::TextureFormat,
    vertices: &[Vertex],
    indices: &[u32],
    texture_view: wgpu::TextureView,
    initial_size: (u32, u32),
) -> ReliefPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("relief3d.shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/relief.wgsl").into()),
    });

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("relief3d.vbo"),
        contents: cast_slice(vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("relief3d.ibo"),
        contents: cast_slice(indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("relief3d.uniforms"),
        contents: cast_slice(&[Uniforms::identity()]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("relief3d.sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("relief3d.bind_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                // The fragment shader reads light_color + ambient as well,
                // not just the vertex shader, so flag both stages.
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("relief3d.bind_group"),
        layout: &bind_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("relief3d.pipeline_layout"),
        bind_group_layouts: &[&bind_layout],
        push_constant_ranges: &[],
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("relief3d.pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[Vertex::LAYOUT],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            front_face: wgpu::FrontFace::Ccw,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24Plus,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    let depth_view = make_depth(&device, initial_size.0, initial_size.1);

    ReliefPipeline {
        device,
        queue,
        render_pipeline,
        vertex_buffer,
        index_buffer,
        index_count: indices.len() as u32,
        uniform_buffer,
        bind_group,
        depth_view,
        depth_size: initial_size,
    }
}
