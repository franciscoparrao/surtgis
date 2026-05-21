pub mod classification;
pub mod clip;
pub mod extract;
pub mod extract_patches;
pub mod gfm_profiles;
pub mod hydrology;
pub mod imagery;
pub mod info;
pub mod interpolation;
pub mod landscape;
pub mod morphology;
pub mod mosaic;
// Pipeline workflows depend on STAC for input data acquisition; without
// the cloud feature there's no useful pipeline operation available, and
// compiling it triggers unresolved imports for super::stac. Gate the
// whole module to match cog/stac.
#[cfg(feature = "cloud")]
pub mod pipeline;
pub mod stac_writer;
pub mod statistics;
pub mod temporal;
pub mod terrain;
pub mod texture;
pub mod vector;
pub mod zarr_writer;

#[cfg(feature = "cloud")]
pub mod cog;
#[cfg(feature = "cloud")]
pub mod stac;

#[cfg(feature = "projections")]
pub mod reproject;

#[cfg(feature = "ml")]
pub mod ml;
