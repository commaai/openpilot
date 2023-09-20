#!/bin/bash

# Install Portable CL to allow modeld to run on various additional cpus
# adapted from: https://github.com/pocl/pocl/blob/main/tools/docker/Ubuntu/20_04.64bit

LLVM_VERSION=12
POCL_DIR=/opt/pocl

apt update && apt install -y build-essential ocl-icd-libopencl1 cmake git pkg-config libclang-${LLVM_VERSION}-dev clang-${LLVM_VERSION} libclang-cpp${LLVM_VERSION}-dev llvm-${LLVM_VERSION}-dev make ninja-build ocl-icd-libopencl1 ocl-icd-dev ocl-icd-opencl-dev libhwloc-dev zlib1g zlib1g-dev dialog apt-utils

cd /opt ; git clone https://github.com/pocl/pocl.git ; cd $POCL_DIR ; git checkout $GIT_COMMIT
cd $POCL_DIR ; test -z "$GH_PR" || (git fetch origin +refs/pull/$GH_PR/merge && git checkout -qf FETCH_HEAD) && :
cd $POCL_DIR ; mkdir b ; cd b; cmake -G Ninja -DWITH_LLVM_CONFIG=/usr/bin/llvm-config-${LLVM_VERSION} -DCMAKE_INSTALL_PREFIX=/usr ..
cd $POCL_DIR/b ; ninja install
# removing this picks up PoCL from the system install, not the build dir
cd $POCL_DIR/b ; rm -f CTestCustom.cmake
cd $POCL_DIR/b ; ctest -j4 --output-on-failure -L internal

apt install -y software-properties-common
add-apt-repository -y ppa:ocl-icd/ppa
apt update && apt install -y ocl-icd-dev