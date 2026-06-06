// M3 shader + P3-M1 atmospheric haze.
//
// Inputs:
//   @location(0) position : vec3<f32>   // world-space at baseline zex
//   @location(1) uv       : vec2<f32>
//   @location(2) normal   : vec3<f32>   // baked at baseline zex
//
// Uniforms (group 0, binding 0):
//   view_proj      : mat4x4<f32>
//   light_dir      : vec4 (.xyz direction TOWARDS light)
//   light_color    : vec4 (.xyz colour, .w ambient term)
//   vertical_scale : vec4 (.x runtime scale)
//   fog_color      : vec4 (.xyz colour, .w density in [0, 1])
//   fog_range      : vec4 (.x = fog_near, .y = fog_far, depth-fog params)
//
// The fragment shader applies a depth-based haze AFTER lighting:
// `mix(shaded, fog_color, fog_t)` where `fog_t = density * smoothstep(
//   fog_near, fog_far, linear_depth)`. Density 0 disables the effect
// entirely so the M3 output is bit-equivalent to pre-P3.

struct Uniforms {
  view_proj      : mat4x4<f32>,
  light_dir      : vec4<f32>,
  light_color    : vec4<f32>,
  vertical_scale : vec4<f32>,
  fog_color      : vec4<f32>,
  fog_range      : vec4<f32>,
};

@group(0) @binding(0) var<uniform> u    : Uniforms;
@group(0) @binding(1) var          tex  : texture_2d<f32>;
@group(0) @binding(2) var          samp : sampler;

struct VsOut {
  @builtin(position) clip         : vec4<f32>,
  @location(0)       uv           : vec2<f32>,
  @location(1)       normal_ws    : vec3<f32>,
  @location(2)       linear_depth : f32,
};

@vertex
fn vs_main(
  // P4-M3b compressed inputs. snorm16x4 / unorm16x2 / snorm8x4 — the
  // hardware decodes to f32 vectors in [-1, 1] (positions, normals)
  // and [0, 1] (UV). The 4th component of the packed vec4 inputs is
  // padding; ignored via `.xyz`.
  @location(0) pos_packed    : vec4<f32>,
  @location(1) uv            : vec2<f32>,
  @location(2) normal_packed : vec4<f32>,
) -> VsOut {
  let zex = u.vertical_scale.x;

  var pos = pos_packed.xyz;
  pos.y = pos.y * zex;

  var n = normal_packed.xyz;
  let safe_zex = max(abs(zex), 1e-6);
  n.y = n.y / safe_zex;
  n = normalize(n);

  // Negative `w` in clip space is "in front of the camera"; for depth
  // fog we want a positive scalar that grows with distance, so we
  // recompute from the clip coordinate.
  let clip_pos = u.view_proj * vec4<f32>(pos, 1.0);

  var out : VsOut;
  out.clip         = clip_pos;
  out.uv           = uv;
  out.normal_ws    = n;
  // `clip_pos.w` equals -view_z under a standard right-handed
  // projection — directly the distance from the camera in world units.
  out.linear_depth = clip_pos.w;
  return out;
}

@fragment
fn fs_main(in : VsOut) -> @location(0) vec4<f32> {
  let base = textureSample(tex, samp, in.uv);
  let n = normalize(in.normal_ws);
  let l = normalize(u.light_dir.xyz);

  let lambert = max(dot(n, l), 0.0);
  let ambient = u.light_color.w;
  let lit = ambient + (1.0 - ambient) * lambert;
  var shaded = base.rgb * (lit * u.light_color.xyz);

  // Atmospheric haze. fog_color.w is density in [0, 1] — when 0, the
  // mix is the identity. fog_range.x / .y are the linear-depth near
  // and far stops. smoothstep instead of linear so the transition
  // does not have a hard near-edge.
  let density  = u.fog_color.w;
  let fog_near = u.fog_range.x;
  let fog_far  = max(u.fog_range.y, fog_near + 1e-3);
  let fog_t    = density * smoothstep(fog_near, fog_far, in.linear_depth);
  shaded       = mix(shaded, u.fog_color.xyz, fog_t);

  return vec4<f32>(shaded, base.a);
}
