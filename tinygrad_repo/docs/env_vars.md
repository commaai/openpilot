# List of environment variables that control tinygrad behavior.

This is a list of environment variable that control the runtime behavior of tinygrad and its examples.
Most of these are self-explanatory, and are usually used to set an option at runtime.

Example: `DEV=CL DEBUG=4 python3 -m pytest`

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
DEBUG               | [1-7]      | enable debugging output (operations, timings, speed, generated code and more)
DEV                 | [AMD, NV, ...] | enable a specific backend, see [below](#dev-variable)
BEAM                | [#]        | number of beams in kernel beam search
DEFAULT_FLOAT       | [HALF, ...]| specify the default float dtype (FLOAT32, HALF, BFLOAT16, FLOAT64, ...), default to FLOAT32
IMAGE               | [1]        | enable 2d specific optimizations
FLOAT16             | [1]        | use float16 for images instead of float32
JIT                 | [0-2]      | 0=disabled, 1=[jit enabled](quickstart.md#jit) (default), 2=jit enabled, but graphs are disabled
VIZ                 | [1]        | 0=disabled, 1=[viz enabled](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/viz)
ALLOW_TF32          | [1]        | enable TensorFloat-32 tensor cores on Ampere or newer GPUs.
WEBGPU_BACKEND      | [WGPUBackendType_Metal, ...]          | Force select a backend for WebGPU (Metal, DirectX, OpenGL, Vulkan...)
CUDA_PATH           | str        | Use `CUDA_PATH/include` for CUDA headers for CUDA and NV backends. If not set, TinyGrad will use `/usr/local/cuda/include`, `/usr/include` and `/opt/cuda/include`.

### DEV variable

The `DEV` variable deserves special note due to its more nuanced syntax.
`DEV` is used to specify the target device, target renderer and target architecture for said device, separated by colons.
Specifying the renderer and architecture is optional, omitting a preference will cause tinygrad to automatically determine a suitable setting.
The `DEV` variable may also be used to specify the interface through which to access the device (eg. `PCI`, `USB`). Interfaces may be specified preceding the target triple,
separated by a plus (eg. `DEV=USB+AMD:LLVM`). Similarly as above, the interface may be omitted. Example usage follows:

`DEV` contents | Interpretation
--- | ---
AMD           | use the AMD device
AMD:LLVM      | use the AMD device with the LLVM renderer
NV:CUDA:sm_70 | use the NV device with the CUDA renderer targetting sm_70
AMD::gfx950   | use the AMD device targetting gfx950
USB+AMD       | use the AMD device over the USB interface
CPU:LLVM      | use the CPU device with the LLVM renderer
CPU:LLVM:x86_64,znver2,avx2,-avx512f | use the CPU device with the LLVM renderer, with [additional arch flags](runtime.md#cpu-arch)

### Debug breakdown

Variable | Value | Description
---|---|---
DEBUG               | >= 1       | Enables debugging and lists devices being used
DEBUG               | >= 2       | Provides performance metrics for operations, including timing, memory usage, bandwidth for each kernel execution
DEBUG               | >= 3       | Outputs the applied optimizations at a kernel level
DEBUG               | >= 4       | Outputs the generated kernel code
DEBUG               | >= 5       | Displays the intermediate representation of the computation UOps
DEBUG               | >= 6       | Displays the intermediate representation of the computation UOps in a linearized manner, detailing the operation sequence
DEBUG               | >= 7       | Outputs the assembly code generated for the target hardware
