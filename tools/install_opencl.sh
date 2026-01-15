#!/usr/bin/env bash

# llvm 18系が pocl の build に必要
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 18
sudo apt install -y llvm-18 llvm-18-dev llvm-18-tools clang-18
sudo apt install -y libclang-18-dev libedit-dev
sudo apt install -y ocl-icd-libopencl1 ocl-icd-opencl-dev


# pocl の build と install
cd ../pocl || exit
mkdir build && cd build || exit
cmake .. \
  -DCMAKE_INSTALL_PREFIX=/usr/local/pocl \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=ON \
  -DLLVM_DIR=/usr/lib/llvm-18/cmake \
  -DLLC_HOST_CPU=cortex-a78

make -j"$(nproc)"
sudo make install

# pocl の icd を登録
sudo mkdir -p /etc/OpenCL/vendors
echo "/usr/local/pocl/lib/libpocl.so" | sudo tee /etc/OpenCL/vendors/pocl.icd

# LD_LIBRARY_PATH の path に opencv の path を追加
VENV_PATH="$(python3 -c 'import sys; print(sys.prefix)')/lib"
export LD_LIBRARY_PATH=$VENV_PATH:$LD_LIBRARY_PATH

# check
ldd /usr/local/pocl/lib/libpocl.so | grep opencv
clinfo