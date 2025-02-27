#!/bin/bash
mkdir -p src
cd src
git clone https://github.com/RadeonOpenCompute/ROCT-Thunk-Interface.git -b rocm-5.5.0
git clone https://github.com/RadeonOpenCompute/ROCm-Device-Libs.git -b rocm-5.5.0
git clone https://github.com/RadeonOpenCompute/llvm-project.git -b rocm-5.5.0 --depth 1
git clone https://github.com/RadeonOpenCompute/ROCR-Runtime.git -b rocm-5.5.0
git clone https://github.com/ROCm-Developer-Tools/ROCclr.git -b rocm-5.5.0
git clone https://github.com/RadeonOpenCompute/ROCm-CompilerSupport.git -b rocm-5.5.0
git clone https://github.com/RadeonOpenCompute/ROCm-OpenCL-Runtime.git -b rocm-5.5.0
cd ../