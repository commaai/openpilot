import numpy as np
import ctypes, functools
import extra.hip_wrapper as hip
from typing import Tuple
from tinygrad.helpers import DEBUG, getenv, diskcache
from tinygrad.ops import Compiled
from tinygrad.runtime.lib import RawBufferCopyInOut, LRUAllocator, RawBufferTransfer
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.renderer.cstyle import uops_to_cstyle, CStyleLanguage

# TODO: if you fork and exit the child process after creating anything with cl on AMD, it hangs on e.wait()
if DEBUG >= 6:
  from extra.helpers import enable_early_exec
  early_exec = enable_early_exec()

# The default HIP stream is used for everything.

class HIPAllocator(LRUAllocator):
  def _do_alloc(self, size, dtype, device, **kwargs):
    hip.hipSetDevice(device)
    return hip.hipMalloc(size * dtype.itemsize)
  def _do_free(self, buf): hip.hipFree(buf)
  def _cached_bufkey(self, size, dtype, device): return (device, size*dtype.itemsize) # Buffers of the same length could be reused, no matter what dtype.

class _HIP:
  def __init__(self, device=None):
    self.default_device = device or getenv("HIP_DEFAULT_DEVICE")
    hip.hipSetDevice(self.default_device)
    self.device_count = hip.hipGetDeviceCount()
    self.allocator = HIPAllocator(hip.hipGetDeviceProperties(self.default_device).totalGlobalMem)
HIP = _HIP()

class RawHIPBuffer(RawBufferCopyInOut, RawBufferTransfer):
  def __init__(self, size, dtype, device=HIP.default_device, buf=None, allocator=HIP.allocator): super().__init__(size, dtype, buf=buf, allocator=allocator, **{'device': int(device)})
  def _copyin(self, x:np.ndarray):
    hip.hipSetDevice(self._device)
    hip.hipMemcpyAsync(self._buf, np.require(x, requirements='C').ctypes.data_as(ctypes.c_void_p), self.size * self.dtype.itemsize, hip.hipMemcpyHostToDevice, 0)
  def _copyout(self, x:np.ndarray):
    hip.hipSetDevice(self._device)
    hip.hipMemcpy(x.ctypes.data, self._buf, self.size * self.dtype.itemsize, hip.hipMemcpyDeviceToHost)
  def _transfer(self, x):
    hip.hipSetDevice(x._device)
    hip.hipMemcpy(self._buf, x._buf, self.size * self.dtype.itemsize, hip.hipMemcpyDeviceToDevice)

@diskcache
def compile_hip(prg) -> bytes:
  prog = hip.hiprtcCreateProgram(prg, "<null>", [], [])
  hip.hiprtcCompileProgram(prog, [f'--offload-arch={hip.hipGetDeviceProperties(HIP.default_device).gcnArchName}'])
  return hip.hiprtcGetCode(prog)

class HIPProgram:
  def __init__(self, name:str, prg:bytes):
    self.modules, self.prgs = [], []

    if DEBUG >= 6:
      asm = early_exec((["/opt/rocm/llvm/bin/llvm-objdump", '-d', '-'], prg))
      print('\n'.join([x for x in asm.decode('utf-8').split("\n") if 's_code_end' not in x]))

    for i in range(HIP.device_count):
      hip.hipSetDevice(i)
      self.modules.append(hip.hipModuleLoadData(prg))
      self.prgs.append(hip.hipModuleGetFunction(self.modules[-1], name))

  def __call__(self, *args, global_size:Tuple[int,int,int], local_size:Tuple[int,int,int], wait=False):
    hip.hipSetDevice(args[0]._device)
    if wait:
      start, end = hip.hipEventCreate(), hip.hipEventCreate()
      hip.hipEventRecord(start)
    class PackageStruct(ctypes.Structure):
      _fields_ = [(f'field{idx}', ctypes.c_void_p if not isinstance(args[idx], int) else ctypes.c_int) for idx in range(len(args))]
    struct = PackageStruct(*[data._buf if not isinstance(data, int) else np.int32(data) for data in args])
    hip.hipModuleLaunchKernel(self.prgs[args[0]._device], global_size[0], global_size[1], global_size[2], local_size[0], local_size[1], local_size[2], 0, 0, struct)
    if wait:
      hip.hipEventRecord(end)
      hip.hipEventSynchronize(end)
      return hip.hipEventElapsedTime(start, end)*1e-3

  def __del__(self):
    for module in self.modules: hip.hipModuleUnload(module)

renderer = functools.partial(uops_to_cstyle, CStyleLanguage(
  kernel_prefix = "#include <hip/hip_common.h>\n#define INFINITY (__builtin_inff())\n#define NAN (__builtin_nanf(\"\"))" + """
__device__ float4 max(float4 x, float4 y) { return float4(max(x.x, y.x), max(x.y, y.y), max(x.z, y.z), max(x.w, y.w)); }
__device__ float4 pow(float x, float4 y) { return float4(pow(x, y.x), pow(x, y.y), pow(x, y.z), pow(x, y.w)); }
__device__ float4 pow(float4 x, float4 y) { return float4(pow(x.x, y.x), pow(x.y, y.y), pow(x.z, y.z), pow(x.w, y.w)); }
__device__ float4 log2(float4 x) { return float4(log2(x.x), log2(x.y), log2(x.z), log2(x.w)); }
__device__ float4 exp2(float4 x) { return float4(exp2(x.x), exp2(x.y), exp2(x.z), exp2(x.w)); }
__device__ float4 sin(float4 x) { return float4(sin(x.x), sin(x.y), sin(x.z), sin(x.w)); }
typedef float float8 __attribute__((ext_vector_type(8)));
typedef _Float16 half16 __attribute__((ext_vector_type(16)));
extern "C" __global__
  """, launch_bounds=True,
  smem_prefix = "__shared__ ", smem_prefix_for_cast=False, barrier = "__syncthreads();", float4 = "make_float4", uses_vload=True, uses_ptr_arithmetic=True, arg_int_prefix = "const int",
  half_prekernel = "#include <hip/hip_fp16.h>\nusing half4 = HIP_vector_type<half, 4>;" + """
__device__ float vload_half(size_t offset, const half *p) { return (float)*(p + offset); }
__device__ float2 vload_half2(size_t offset, const half *p) { return make_float2((float)*(p + offset*2), (float)*(p + offset*2 + 1)); }
__device__ float4 vload_half4(size_t offset, const half *p) { return make_float4((float)*(p + offset*4), (float)*(p + offset*4 + 1), (float)*(p + offset*4 + 2), (float)*(p + offset*4 + 3)); }
__device__ void vstore_half(float data, size_t offset, half *p) { *(p + offset) = (half)data; }
__device__ void vstore_half2(float2 data, size_t offset, half *p) { *(p + offset*2) = (half)data.x; *(p + offset*2 + 1) = (half)data.y; }
__device__ void vstore_half4(float4 data, size_t offset, half *p) { *(p + offset*4) = (half)data.x; *(p + offset*4 + 1) = (half)data.y; *(p + offset*4 + 2) = (half)data.z; *(p + offset*4 + 3) = (half)data.w; }
  """,
  gid = [f'blockIdx.{chr(120+i)}' for i in range(3)],
  lid = [f'threadIdx.{chr(120+i)}' for i in range(3)]))
HIPBuffer = Compiled(RawHIPBuffer, LinearizerOptions(device="HIP"), renderer, compile_hip, HIPProgram, hip.hipDeviceSynchronize)
