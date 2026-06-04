// M3 shader: textured mesh + Lambertian directional light.
//
// Inputs:
//   @location(0) position : vec3<f32>   // world-space at baseline zex
//   @location(1) uv       : vec2<f32>
//   @location(2) normal   : vec3<f32>   // baked at baseline zex
//
// Uniforms (group 0, binding 0):
//   view_proj      : mat4x4<f32>
//   light_dir      : vec3<f32>  (pad to vec4)        // direction TOWARDS the light
//   light_color    : vec3<f32>
//   ambient        : f32
//   vertical_scale : f32        (pad to vec4)
//
// `light_dir` is interpreted as the direction *from the surface toward
// the sun*, so the Lambertian term is `dot(n, light_dir)`. The host
// computes it from sun azimuth + altitude (see `OrbitCamera`-style
// sun controls) and uploads it normalised.
//
// `vertical_scale` re-scales the Y axis of the baked mesh at render
// time. The normal is re-oriented for the new Y stretch so lighting
// stays correct without rebuilding the vertex buffer.

// Pack everything into vec4 slots so the std140-equivalent layout is
// unambiguous and the Rust side can declare a matching POD without
// having to thread `vec3 + f32` packing rules manually.
struct Uniforms {
  view_proj      : mat4x4<f32>,
  light_dir      : vec4<f32>,  // .xyz = direction toward light, .w unused
  light_color    : vec4<f32>,  // .xyz = colour, .w = ambient term
  vertical_scale : vec4<f32>,  // .x   = scale, .yzw unused
};

@group(0) @binding(0) var<uniform> u    : Uniforms;
@group(0) @binding(1) var          tex  : texture_2d<f32>;
@group(0) @binding(2) var          samp : sampler;

struct VsOut {
  @builtin(position) clip       : vec4<f32>,
  @location(0)       uv         : vec2<f32>,
  @location(1)       normal_ws  : vec3<f32>,
};

@vertex
fn vs_main(
  @location(0) pos_in    : vec3<f32>,
  @location(1) uv        : vec2<f32>,
  @location(2) normal_in : vec3<f32>,
) -> VsOut {
  let zex = u.vertical_scale.x;

  // Scale the displacement axis.
  var pos = pos_in;
  pos.y = pos.y * zex;

  // Re-orient the normal: when y stretches by k, the y component of the
  // normal compresses by 1/k. Then renormalise.
  var n = normal_in;
  let safe_zex = max(abs(zex), 1e-6);
  n.y = n.y / safe_zex;
  n = normalize(n);

  var out : VsOut;
  out.clip      = u.view_proj * vec4<f32>(pos, 1.0);
  out.uv        = uv;
  out.normal_ws = n;
  return out;
}

@fragment
fn fs_main(in : VsOut) -> @location(0) vec4<f32> {
  let base = textureSample(tex, samp, in.uv);
  let n = normalize(in.normal_ws);
  let l = normalize(u.light_dir.xyz);

  // Lambertian + ambient. Clamp dot to [0, 1] so back-faces don't
  // contribute negative light; the ambient term keeps shadowed faces
  // from going pure black even when the texture already bakes the
  // 2D shadows.
  let lambert = max(dot(n, l), 0.0);
  let ambient = u.light_color.w;
  let lit = ambient + (1.0 - ambient) * lambert;
  let shaded = base.rgb * (lit * u.light_color.xyz);
  return vec4<f32>(shaded, base.a);
}
