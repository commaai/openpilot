#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# llvm 18 is required to build pocl
curl -fsSL https://apt.llvm.org/llvm.sh | sudo bash -s -- 18
sudo apt install -y llvm-18 llvm-18-dev llvm-18-tools clang-18
sudo apt install -y libclang-18-dev libedit-dev
sudo apt install -y ocl-icd-libopencl1 ocl-icd-opencl-dev

POCL_PREFIX=/usr/local/pocl
if [ -f "$POCL_PREFIX/lib/libpocl.so" ]; then
  echo "pocl already installed at $POCL_PREFIX, skipping build"
else
  cd "$PROJECT_DIR/pocl" || exit
  mkdir -p build && cd build || exit
  cmake .. \
    -DCMAKE_INSTALL_PREFIX="$POCL_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_CUDA=ON \
    -DLLVM_DIR=/usr/lib/llvm-18/cmake \
    -DLLC_HOST_CPU=cortex-a78

  make -j"$(nproc)"
  sudo make install
fi

sudo mkdir -p /etc/OpenCL/vendors
POCL_ICD_FILE=/etc/OpenCL/vendors/pocl.icd
POCL_ICD_ENTRY="/usr/local/pocl/lib/libpocl.so"
if ! sudo grep -qxF "$POCL_ICD_ENTRY" "$POCL_ICD_FILE" 2>/dev/null; then
  echo "$POCL_ICD_ENTRY" | sudo tee -a "$POCL_ICD_FILE" >/dev/null
fi

VENV_PATH="$(python3 -c 'import sys; print(sys.prefix)')/lib"
BASHRC="$HOME/.bashrc"
LINE="# Added by install_opencl.sh
export LD_LIBRARY_PATH=${VENV_PATH}:\$LD_LIBRARY_PATH"

if ! grep -Fqx "export LD_LIBRARY_PATH=${VENV_PATH}:\$LD_LIBRARY_PATH" "$BASHRC"; then
  printf '%s\n' "$LINE" >> "$BASHRC"
fi

# check
ldd /usr/local/pocl/lib/libpocl.so | grep opencv
clinfo