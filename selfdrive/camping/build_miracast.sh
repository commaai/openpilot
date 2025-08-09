#!/bin/bash
set -euo pipefail

# Build script for Miraclecast on agnos OS (aarch64)
# Requires (on device/agnos): meson, ninja, pkg-config, glib2, systemd-dev, readline, gstreamer devs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIRACAST_DIR="$SCRIPT_DIR/miraclecast"
BUILD_DIR="$MIRACAST_DIR/build"
OUTPUT_DIR="$SCRIPT_DIR/bin"

# Check if submodule is initialized
if [ ! -d "$MIRACAST_DIR/.git" ]; then
  echo "Initializing miraclecast submodule..."
  git submodule update --init --recursive selfdrive/camping/miraclecast
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Clean previous build
if [ -d "$BUILD_DIR" ]; then
  rm -rf "$BUILD_DIR"
fi

echo "Configuring MiracleCast..."
cd "$MIRACAST_DIR"
# Prefer native build on device; fallback to cross if present
MESON_ARGS=(
  "${BUILD_DIR}"
  "--prefix=/usr"
  "--buildtype=release"
  "-Drely-udev=false"
  "-Dbuild-tests=false"
  "-Dbuild-gstreamer=true"
)
if [ -f /usr/local/share/meson/cross/aarch64-linux-gnu.txt ]; then
  MESON_ARGS+=("--cross-file=/usr/local/share/meson/cross/aarch64-linux-gnu.txt")
fi
meson setup "${MESON_ARGS[@]}"

# Build
echo "Building MiracleCast..."
ninja -C "$BUILD_DIR"

# Copy binaries to output directory
echo "Copying binaries to $OUTPUT_DIR..."
cp "$BUILD_DIR/src/wifi/miracle-wifid" "$OUTPUT_DIR/"
cp "$BUILD_DIR/src/ctl/miracle-wfdctl" "$OUTPUT_DIR/"
cp "$BUILD_DIR/src/ctl/miracle-sinkctl" "$OUTPUT_DIR/"

# Make binaries executable
chmod +x "$OUTPUT_DIR"/*

echo "Build complete! Binaries are in $OUTPUT_DIR"
echo "Run scripts/install_camping_receiver.sh to install on device"