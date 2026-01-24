#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "   PoCL Installer for Jetson"
echo "=========================================="

POCL_PREFIX=/usr/local/pocl
OPENCL_HEADERS_DIR="$PROJECT_DIR/opencl_headers"

# --- 1. Clean up previous installations ---
echo "[1/6] Cleaning up previous installations..."
sudo rm -rf "$POCL_PREFIX"
sudo rm -f /etc/OpenCL/vendors/pocl.icd
rm -rf "$PROJECT_DIR/pocl/build"
rm -rf "$OPENCL_HEADERS_DIR"

# --- 2. Install dependencies ---
echo "[2/6] Installing dependencies..."
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake pkg-config curl git \
    libclang-18-dev libedit-dev \
    libhwloc-dev zlib1g-dev \
    ocl-icd-libopencl1

if ! command -v clang-18 &> /dev/null; then
    curl -fsSL https://apt.llvm.org/llvm.sh | sudo bash -s -- 18
    sudo apt-get update
    sudo apt-get install -y llvm-18 llvm-18-dev llvm-18-tools clang-18
fi

# --- 3. Prepare OpenCL headers ---
# On ARM Ubuntu (Jetson), ocl-icd-opencl-dev package lacks ocl_icd.h or has it in
# a non-standard path. We must fetch headers manually from GitHub.
echo "[3/6] Preparing OpenCL Headers..."
git clone --depth 1 https://github.com/KhronosGroup/OpenCL-Headers.git "$OPENCL_HEADERS_DIR"

HEADER_DIR="$OPENCL_HEADERS_DIR/CL"
if [ ! -f "$HEADER_DIR/cl_icd.h" ]; then
    echo "Downloading missing cl_icd.h..."
    wget -qN -P "$HEADER_DIR" \
        "https://raw.githubusercontent.com/KhronosGroup/OpenCL-ICD-Loader/main/inc/CL/cl_icd.h"
fi

# PoCL expects ocl_icd.h at the include root, not in CL/ subdirectory
cp "$HEADER_DIR/cl_icd.h" "$OPENCL_HEADERS_DIR/ocl_icd.h"

# --- 4. Create fake pkg-config for ocl-icd ---
# This is REQUIRED on Jetson. Without it, PoCL's CMake cannot find the ICD loader
# headers and falls back to building itself as a loader (libOpenCL.so) instead of
# a driver (libpocl.so), causing conflicts with the system's ICD loader.
echo "[4/6] Configuring build environment..."
FAKE_PKG_DIR=$(mktemp -d)
printf '%s\n' \
    "prefix=${OPENCL_HEADERS_DIR}" \
    'exec_prefix=${prefix}' \
    'libdir=/usr/lib/aarch64-linux-gnu' \
    'includedir=${prefix}' \
    '' \
    'Name: ocl-icd' \
    'Description: OpenCL ICD Loader' \
    'Version: 3.0.0' \
    'Libs: -L${libdir} -lOpenCL' \
    'Cflags: -I${includedir}' \
    > "${FAKE_PKG_DIR}/ocl-icd.pc"
export PKG_CONFIG_PATH="${FAKE_PKG_DIR}:${PKG_CONFIG_PATH}"

# --- 5. Build PoCL ---
echo "[5/6] Building PoCL..."
cd "$PROJECT_DIR/pocl"
mkdir -p build && cd build

cmake .. \
    -DCMAKE_INSTALL_PREFIX="$POCL_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_CUDA=ON \
    -DENABLE_ICD=ON \
    -DPOCL_ICD_ABSOLUTE_PATH=ON \
    -DENABLE_TESTS=OFF \
    -DENABLE_HWLOC=ON \
    -DLLVM_DIR=/usr/lib/llvm-18/cmake \
    -DLLC_HOST_CPU=cortex-a78 \
    -DWERROR=OFF

# Use -j4 to prevent OOM on memory-constrained systems
make -j4
sudo make install

# Clean up temp pkg-config
rm -rf "$FAKE_PKG_DIR"

# --- 6. Verify and register ICD ---
echo "[6/6] Verifying and registering..."

POCL_LIB_PATH=$(find "$POCL_PREFIX" -name "libpocl.so" | head -n 1)

if [ -z "$POCL_LIB_PATH" ]; then
    echo "FATAL ERROR: 'libpocl.so' was NOT generated."
    echo "This means PoCL built as a loader instead of a driver."
    echo "Debug Info:"
    ls -lF "$POCL_PREFIX/lib/" 2>/dev/null || echo "lib directory does not exist"
    exit 1
fi

echo "Success! Driver found at: $POCL_LIB_PATH"

# Register PoCL as an ICD vendor
sudo mkdir -p /etc/OpenCL/vendors
echo "$POCL_LIB_PATH" | sudo tee /etc/OpenCL/vendors/pocl.icd >/dev/null

# Remove PoCL's bundled OpenCL library to avoid conflicts with system ICD loader
sudo rm -f "$POCL_PREFIX/lib/libOpenCL.so"*

echo "================================================================"
echo "Installation Complete."
echo "Running clinfo to verify..."
echo "================================================================"
clinfo