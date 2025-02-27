# Runtimes

tinygrad supports various runtimes, enabling your code to scale across a wide range of devices. The default runtime can be automatically selected based on the available hardware, or you can force a specific runtime to be default using environment variables (e.g., `CPU=1`).

| Runtime | Description | Requirements |
|---------|-------------|--------------|
| [NV](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/runtime/ops_nv.py) | Provides acceleration for NVIDIA GPUs | Ampere/Ada series GPUs |
| [AMD](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/runtime/ops_amd.py) | Provides acceleration for AMD GPUs | RDNA2/RDNA3 series GPUs |
| [QCOM](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/runtime/ops_qcom.py) | Provides acceleration for QCOM GPUs | 6xx series GPUs |
| [METAL](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/runtime/ops_metal.py) | Utilizes Metal for acceleration on Apple devices | M1+ Macs; Metal 3.0+ for `bfloat` support |
| [CUDA](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/runtime/ops_cuda.py) | Utilizes CUDA for acceleration on NVIDIA GPUs | NVIDIA GPU with CUDA support |
| [GPU (OpenCL)](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/runtime/ops_gpu.py) | Accelerates computations using OpenCL on GPUs | OpenCL 2.0 compatible device |
| [CPU (C Code)](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/runtime/ops_cpu.py) | Runs on CPU using the clang compiler | `clang` compiler in system `PATH` |
| [LLVM (LLVM IR)](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/runtime/ops_llvm.py) | Runs on CPU using the LLVM compiler infrastructure | llvm libraries installed and findable |
| [WEBGPU](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/runtime/ops_webgpu.py) | Runs on GPU using the Dawn WebGPU engine (used in Google Chrome) | Dawn library installed and findable. Download binaries [here](https://github.com/wpmed92/pydawn/releases/tag/v0.1.6). |
