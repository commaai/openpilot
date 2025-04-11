# List of environment variables that control tinygrad behavior.

This is a list of environment variable that control the runtime behavior of tinygrad and its examples.
Most of these are self-explanatory, and are usually used to set an option at runtime.

Example: `GPU=1 DEBUG=4 python3 -m pytest`

However you can also decorate a function to set a value only inside that function.

```python
# in tensor.py (probably only useful if you are a tinygrad developer)
@Context(DEBUG=4)
def numpy(self) -> ...
```

Or use contextmanager to temporarily set a value inside some scope:

```python
with Context(DEBUG=0):
  a = Tensor.ones(10, 10)
  a *= 2
```

## Global Variables
The columns of this list are are: Variable, Possible Value(s) and Description.

- A `#` means that the variable can take any integer value.

These control the behavior of core tinygrad even when used as a library.

Variable | Possible Value(s) | Description
---|---|---
DEBUG               | [1-6]      | enable debugging output, with 4 you get operations, timings, speed, generated code and more
GPU                 | [1]        | enable the GPU (OpenCL) backend
CUDA                | [1]        | enable CUDA backend
AMD                 | [1]        | enable AMD backend
NV                  | [1]        | enable NV backend
METAL               | [1]        | enable Metal backend (for Mac M1 and after)
METAL_XCODE         | [1]        | enable Metal using macOS Xcode SDK
CPU                 | [1]        | enable CPU (Clang) backend
LLVM                | [1]        | enable LLVM backend
BEAM                | [#]        | number of beams in kernel beam search
DEFAULT_FLOAT       | [HALF, ...]| specify the default float dtype (FLOAT32, HALF, BFLOAT16, FLOAT64, ...), default to FLOAT32
IMAGE               | [1-2]      | enable 2d specific optimizations
FLOAT16             | [1]        | use float16 for images instead of float32
PTX                 | [1]        | enable the specialized [PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/) assembler for Nvidia GPUs. If not set, defaults to generic CUDA codegen backend.
PROFILE             | [1]        | enable profiling. This feature is supported in NV, AMD, QCOM and METAL backends.
VISIBLE_DEVICES     | [list[int]]| restricts the NV/AMD devices that are available. The format is a comma-separated list of identifiers (indexing starts with 0).
JIT                 | [0-2]      | 0=disabled, 1=[jit enabled](quickstart.md#jit) (default), 2=jit enabled, but graphs are disabled
VIZ                 | [1]        | 0=disabled, 1=[viz enabled](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/viz)
ALLOW_TF32          | [1]        | enable TensorFloat-32 tensor cores on Ampere or newer GPUs.
WEBGPU_BACKEND      | [WGPUBackendType_Metal, ...]          | Force select a backend for WebGPU (Metal, DirectX, OpenGL, Vulkan...)