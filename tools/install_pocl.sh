#!/bin/bash

# Install Portable CL to allow modeld to run on various additional cpus
# adapted from: https://github.com/pocl/pocl/blob/main/tools/docker/Ubuntu/20_04.64bit

LLVM_VERSION=12

apt update && apt install -y build-essential ocl-icd-libopencl1 cmake git pkg-config libclang-${LLVM_VERSION}-dev clang-${LLVM_VERSION} libclang-cpp${LLVM_VERSION}-dev llvm-${LLVM_VERSION}-dev make ninja-build ocl-icd-libopencl1 ocl-icd-dev ocl-icd-opencl-dev libhwloc-dev zlib1g zlib1g-dev dialog apt-utils

cd /opt
git clone https://github.com/pocl/pocl.git
cd /opt/pocl
git checkout $GIT_COMMIT
test -z "$GH_PR" || (git fetch origin +refs/pull/$GH_PR/merge && git checkout -qf FETCH_HEAD)
mkdir b && cd b
cmake -G Ninja -DWITH_LLVM_CONFIG=/usr/bin/llvm-config-${LLVM_VERSION} -DCMAKE_INSTALL_PREFIX=/usr ..
ninja install
# removing this picks up PoCL from the system install, not the build dir
rm -f CTestCustom.cmake
ctest -j4 --output-on-failure -L internal