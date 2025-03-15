# Runtimes

tinygrad supports various runtimes, enabling your code to scale across a wide range of devices. The default runtime can be automatically selected based on the available hardware, or you can force a specific runtime to be default using environment variables (e.g., `CLANG=1`).

| Runtime | Description | Requirements |
|---------|-------------|--------------|
| [NV](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/runtime/ops_nv.py) | Provides acceleration for NVIDIA GPUs | Ampere/Ada series GPUs |
| [AMD](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/runtime/ops_amd.py) | Provides acceleration for AMD GPUs | RDNA2/RDNA3 series GPUs |
| [QCOM](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/runtime/ops_qcom.py) | Provides acceleration for QCOM GPUs | 6xx series GPUs |
| [METAL](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/runtime/ops_metal.py) | Utilizes Metal for acceleration on Apple devices | M1+ Macs; Metal 3.0+ for `bfloat` support |
| [CUDA](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/runtime/ops_cuda.py) | Utilizes CUDA for acceleration on NVIDIA GPUs | NVIDIA GPU with CUDA support |
| [GPU (OpenCL)](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/runtime/ops_gpu.py) | Accelerates computations using OpenCL on GPUs | OpenCL 2.0 compatible device |
| [CLANG (C Code)](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/runtime/ops_clang.py) | Runs on CPU using the clang compiler | `clang` compiler in system `PATH` |
| [LLVM](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/runtime/ops_llvm.py) | Runs on CPU using the LLVM compiler infrastructure | `llvmlite` package installed |
