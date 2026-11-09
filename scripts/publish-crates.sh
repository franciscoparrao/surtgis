#!/usr/bin/env bash
# Publish the workspace to crates.io.
#
# The CLI (`surtgis`) and `surtgis-server` carry an optional `ecw` feature
# backed by `surtgis-ecw`, which is not published (licensing review pending,
# docs/ecw_format.md). cargo refuses to publish a crate whose manifest names
# an unpublishable dependency, even an optional one, so the lines declaring
# that feature end in `# @ecw-only` and are stripped from the two manifests
# for the duration of the publish. The published crates therefore have no
# `ecw` feature; the GitHub Release binaries are built from the tree and
# keep it. The manifests and Cargo.lock are restored on exit.
#
# Usage: scripts/publish-crates.sh [--dry-run] [extra cargo publish flags]
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "publish-crates: the working tree must be clean (the tag must match what is published)" >&2
  exit 1
fi
MANIFESTS=(crates/cli/Cargo.toml crates/server/Cargo.toml)
restore() { git checkout -- "${MANIFESTS[@]}" Cargo.lock; }
trap restore EXIT
sed -i '/# @ecw-only$/d' "${MANIFESTS[@]}"
echo "publish-crates: ecw lines stripped from ${MANIFESTS[*]}"
cargo publish --workspace --allow-dirty "$@"
