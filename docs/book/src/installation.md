# Installation

Three paths, in order of increasing effort.

## 1. Precompiled binaries (recommended for evaluation)

Download the archive for your platform from the
[GitHub releases page](https://github.com/franciscoparrao/surtgis/releases/latest):

| Platform | Archive |
|---|---|
| Linux x86_64 | `surtgis-v<version>-x86_64-unknown-linux-gnu.tar.gz` |
| macOS arm64 (Apple Silicon) | `surtgis-v<version>-aarch64-apple-darwin.tar.gz` |
| Windows x86_64 | `surtgis-v<version>-x86_64-pc-windows-msvc.zip` |

Unpack and put `surtgis` (or `surtgis.exe`) somewhere on your `PATH`:

```bash
# Linux / macOS
tar xzf surtgis-*.tar.gz
sudo mv surtgis /usr/local/bin/
surtgis --version
```

The precompiled binary has the feature set `cloud,zarr,projections`:
STAC, COG, climate-data Zarr readers, and UTM reprojection. This covers
every tutorial and every how-to in this book.

## 2. `cargo install` (for the full feature set)

If you have the Rust toolchain and want `netcdf` / `grib` support on top,
install from [crates.io](https://crates.io/crates/surtgis):

```bash
cargo install surtgis --all-features
```

System libraries required for `--all-features`:

| Feature | Linux | macOS | Windows |
|---|---|---|---|
| `netcdf` | `libnetcdf-dev` | `brew install netcdf` | not supported |
| `grib` | libgribapi or eccodes | `brew install eccodes` | not supported |

If you only need the same feature set as the precompiled binary:

```bash
cargo install surtgis
```

## 3. From source (for contributors)

```bash
git clone https://github.com/franciscoparrao/surtgis
cd surtgis
cargo build --release -p surtgis --bin surtgis
./target/release/surtgis --version
```

The workspace contains ten crates. You only need the `surtgis` package for
the CLI; the library crates (`surtgis-core`, `surtgis-algorithms`,
`surtgis-cloud`, etc.) are separately publishable for embedding.

## Verifying the install

Any of these should return sensible output:

```bash
surtgis --version                    # surtgis 0.7.0
surtgis --help                       # top-level command list
surtgis terrain --help               # terrain subcommand list
```

If the binary runs but a specific command fails with "feature not enabled",
you're on the precompiled binary and the command needs `netcdf` / `grib` /
`gdal`. Rebuild from source with `cargo install surtgis --all-features`.

## Next step

Head to the [first tutorial](tutorials/first-terrain-analysis.md) for an
end-to-end walk-through using real data from the Copernicus Digital
Elevation Model.
