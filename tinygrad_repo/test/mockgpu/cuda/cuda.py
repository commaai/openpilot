from __future__ import annotations
from typing import Any
import ctypes, time
from tinygrad.runtime.autogen import cuda as orig_cuda
from tinygrad.helpers import mv_address

for attr in dir(orig_cuda):
  if not attr.startswith('__'):
    globals()[attr] = getattr(orig_cuda, attr)

try:
  gpuocelot_lib = ctypes.CDLL(ctypes.util.find_library("gpuocelot"))
  gpuocelot_lib.ptx_run.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]  # noqa: E501
except Exception: pass

# Global state
class CUDAState:
  def __init__(self):
    self.memory: dict[int, memoryview] = {}
    self.events: dict[int, float] = {} # Event ID -> timestamp
    self.modules: dict[int, memoryview] = {} # Module ID -> code
    self.current_context: int|None = None
    self.contexts: dict[int, dict] = {} # Context ID -> context data
    self.devices: dict[int, dict] = {}  # Device ID -> device data
    self.next_ptr = 1000  # For memory allocation
    self.next_event_id = 1
    self.next_module_id = 1
    self.next_context_id = 1

cuda_state = CUDAState()

# Helper functions
def check_context():
  if cuda_state.current_context is None:
    return orig_cuda.CUDA_ERROR_INVALID_VALUE
  return orig_cuda.CUDA_SUCCESS

# CUDA API simulation
def cuInit(flags: int) -> int:
  return orig_cuda.CUDA_SUCCESS

def cuDeviceGet(device, ordinal: int) -> int:
  if ordinal < 0:
    return orig_cuda.CUDA_ERROR_INVALID_VALUE
  device._obj.value = ordinal
  cuda_state.devices[ordinal] = {"compute_capability": (3, 5)}
  return orig_cuda.CUDA_SUCCESS

def cuCtxCreate_v2(pctx, flags: int, dev: int) -> int:
  ctx_id = cuda_state.next_context_id
  cuda_state.next_context_id += 1
  cuda_state.contexts[ctx_id] = {"device": dev, "flags": flags}
  pctx._obj.value = ctx_id
  return orig_cuda.CUDA_SUCCESS

def cuCtxSetCurrent(context) -> int:
  if context.value not in cuda_state.contexts:
    return orig_cuda.CUDA_ERROR_INVALID_VALUE
  cuda_state.current_context = context.value
  return orig_cuda.CUDA_SUCCESS

def cuMemAlloc_v2(dptr, bytesize: int) -> int:
  x = memoryview(bytearray(bytesize))
  dptr._obj.value = mv_address(x)
  cuda_state.memory[dptr._obj.value] = x
  return orig_cuda.CUDA_SUCCESS

def cuMemFree_v2(dptr) -> int:
  if dptr.value in cuda_state.memory:
    del cuda_state.memory[dptr.value]
    return orig_cuda.CUDA_SUCCESS
  return orig_cuda.CUDA_ERROR_INVALID_VALUE

def cuMemcpyHtoDAsync_v2(dst, src: ctypes.c_void_p, bytesize: int, stream: Any) -> int:
  ctypes.memmove(dst.value, src, bytesize)
  return orig_cuda.CUDA_SUCCESS

def cuMemcpyDtoH_v2(dst: ctypes.c_void_p, src, bytesize: int) -> int:
  ctypes.memmove(dst, src.value, bytesize)
  return orig_cuda.CUDA_SUCCESS

def cuEventCreate(phEvent, flags: int) -> int:
  event_id = cuda_state.next_event_id
  cuda_state.next_event_id += 1
  cuda_state.events[event_id] = 0.0
  phEvent._obj.value = event_id
  return orig_cuda.CUDA_SUCCESS

def cuEventRecord(hEvent, hStream: Any) -> int:
  if hEvent.value not in cuda_state.events:
    return orig_cuda.CUDA_ERROR_INVALID_VALUE
  cuda_state.events[hEvent.value] = time.perf_counter_ns()
  return orig_cuda.CUDA_SUCCESS

def cuEventSynchronize(hEvent) -> int:
  if hEvent.value not in cuda_state.events:
    return orig_cuda.CUDA_ERROR_INVALID_VALUE
  return orig_cuda.CUDA_SUCCESS

def cuEventElapsedTime(pMilliseconds, hStart, hEnd) -> int:
  if hStart.value not in cuda_state.events or hEnd.value not in cuda_state.events:
    return orig_cuda.CUDA_ERROR_INVALID_VALUE
  elapsed = (cuda_state.events[hEnd.value] - cuda_state.events[hStart.value]) * 1e-6
  pMilliseconds._obj.value = elapsed
  return orig_cuda.CUDA_SUCCESS

def cuEventDestroy_v2(hEvent) -> int:
  if hEvent.value in cuda_state.events:
    del cuda_state.events[hEvent.value]
  return orig_cuda.CUDA_SUCCESS

def cuModuleLoadData(module, image: bytes) -> int:
  module_id = cuda_state.next_module_id
  cuda_state.next_module_id += 1
  cuda_state.modules[module_id] = memoryview(bytearray(image))
  module._obj.value = module_id
  return orig_cuda.CUDA_SUCCESS

def cuModuleGetFunction(hfunc, hmod, name: bytes) -> int:
  if hmod.value not in cuda_state.modules:
    return orig_cuda.CUDA_ERROR_INVALID_VALUE
  hfunc._obj.value = mv_address(cuda_state.modules[hmod.value])
  return orig_cuda.CUDA_SUCCESS

def cuModuleUnload(hmod) -> int:
  if hmod.value in cuda_state.modules:
    del cuda_state.modules[hmod.value]
  return orig_cuda.CUDA_SUCCESS

def cuLaunchKernel(f, gx: int, gy: int, gz: int, lx: int, ly: int, lz: int, sharedMemBytes: int,
                   hStream: Any, kernelParams: Any, extra: Any) -> int:
  cargs = [ctypes.cast(getattr(extra, field[0]), ctypes.c_void_p) for field in extra._fields_]
  gpuocelot_lib.ptx_run(ctypes.cast(f.value, ctypes.c_char_p), len(cargs), (ctypes.c_void_p*len(cargs))(*cargs), lx, ly, lz, gx, gy, gz, 0)
  return orig_cuda.CUDA_SUCCESS

def cuDeviceComputeCapability(major, minor, dev: int) -> int:
  if dev not in cuda_state.devices:
    return orig_cuda.CUDA_ERROR_INVALID_VALUE
  major._obj.value = 3
  minor._obj.value = 5
  return orig_cuda.CUDA_SUCCESS

def cuDeviceCanAccessPeer(canAccessPeer, dev: int, peerDev: int) -> int:
  canAccessPeer._obj.value = 1  # Always allow peer access in simulation
  return orig_cuda.CUDA_SUCCESS

def cuCtxEnablePeerAccess(peerContext, flags: int) -> int:
  return orig_cuda.CUDA_SUCCESS

def cuMemHostAlloc(pp, bytesize: int, flags: int) -> int:
  return cuMemAlloc_v2(pp, bytesize)

def cuMemFreeHost(p: ctypes.c_void_p) -> int: return cuMemFree_v2(p)

def cuMemcpyDtoDAsync_v2(dst, src, bytesize: int, stream: Any) -> int:
  ctypes.memmove(dst.value, src.value, bytesize)
  return orig_cuda.CUDA_SUCCESS

def cuFuncSetAttribute(hfunc, attrib: int, value: int) -> int:
  return orig_cuda.CUDA_SUCCESS

def cuStreamWaitEvent(stream: Any, event, flags: int) -> int: return orig_cuda.CUDA_SUCCESS
def cuCtxSynchronize() -> int: return orig_cuda.CUDA_SUCCESS

def cuGetErrorString(error: int, pStr) -> int:
  error_str = orig_cuda.cudaError_enum__enumvalues.get(error, "Unknown CUDA error").encode()
  buf = ctypes.create_string_buffer(error_str)
  # Set the pointer to point to our error string buffer
  pStr._obj.value = ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))
  return orig_cuda.CUDA_SUCCESS
