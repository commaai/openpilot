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

## Interoperability

tinygrad provides interoperability with OpenCL and PyTorch, allowing efficient tensor data sharing between frameworks through the `Tensor.from_blob` API. This enables zero-copy operations by working directly with external memory pointers.

**Important**: When using external memory pointers with tinygrad tensors, you must ensure these pointers remain valid throughout the entire lifetime of the tinygrad tensor to prevent memory corruption.

### `CUDA`/`METAL` PyTorch Interoperability

You can seamlessly work with CUDA/MPS tensors between PyTorch and tinygrad without data copying:
```python
from tinygrad.dtype import _from_torch_dtype
tensor1 = torch.tensor([1.0, 2.0, 3.0], device=torch.device("cuda"))
tiny_tensor1 = Tensor.from_blob(tensor1.data_ptr(), tensor1.shape, dtype=_from_torch_dtype(tensor1.dtype), device='CUDA')

# Before tinygrad calculations, mps needs to be synchronized to make sure data is valid.
if data.device.type == "mps": torch.mps.synchronize()
else: torch.cuda.synchronize()

x = (tiny_tensor1 + 1).realize()
```

### `QCOM` OpenCL Interoperability

tinygrad supports OpenCL interoperability on `QCOM` backend.

Buffer interop allows direct access to OpenCL memory buffers:
```python
# create raw opencl buffer.
cl_buf = cl.clCreateBuffer(cl_context, cl.CL_MEM_READ_WRITE, 0x100, None, status := ctypes.c_int32())

# extract pointers
cl_buf_desc_ptr = to_mv(ctypes.addressof(cl_buf), 8).cast('Q')[0]
rawbuf_ptr = to_mv(cl_buf_desc_ptr, 0x100).cast('Q')[20] # offset 0xA0 is a raw gpu pointer.

# create tiny tensor
tiny = Tensor.from_blob(rawbuf_ptr, (8, 8), dtype=dtypes.int, device='QCOM')
```

And the same for the images:
```python
# create cl image.
cl_img = cl.clCreateImage2D(cl_context, cl.CL_MEM_READ_WRITE, cl.cl_image_format(cl.CL_RGBA, cl.CL_FLOAT), w, h, 0, None, status := ctypes.c_int32())

# extract pointers
cl_buf_desc_ptr = to_mv(ctypes.addressof(cl_img), 8).cast('Q')[0]
rawbuf_ptr = to_mv(cl_buf_desc_ptr, 0x100).cast('Q')[20] # offset 0xA0 is a raw gpu pointer.

# create tiny tensor
tiny = Tensor.from_blob(rawbuf_ptr, (h*w*4,), dtype=dtypes.imagef((h,w)), device='QCOM')
```
