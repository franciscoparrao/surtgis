// M1 spike shader: textured mesh, no lighting yet (M3 adds it).
//
// Vertex layout:
//   @location(0) position : vec3<f32>   // world-space, displaced
//   @location(1) uv       : vec2<f32>   // texture coordinate
//
// Uniforms (group 0, binding 0): view-projection matrix.
// Texture (group 0, binding 1) + sampler (group 0, binding 2).

struct Uniforms {
  view_proj : mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> u : Uniforms;
@group(0) @binding(1) var tex : texture_2d<f32>;
@group(0) @binding(2) var samp : sampler;

struct VsOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) uv : vec2<f32>,
};

@vertex
fn vs_main(
  @location(0) position : vec3<f32>,
  @location(1) uv       : vec2<f32>,
) -> VsOut {
  var out : VsOut;
  out.clip = u.view_proj * vec4<f32>(position, 1.0);
  out.uv   = uv;
  return out;
}

@fragment
fn fs_main(in : VsOut) -> @location(0) vec4<f32> {
  return textureSample(tex, samp, in.uv);
}
