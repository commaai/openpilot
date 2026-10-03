# the cuda driver api in python: the runtime reaches it through function pointers, kernels run on gpuocelot
import ctypes, re
from typing import Any, Callable
from tinygrad.runtime.autogen import cuda
from tinygrad.helpers import mv_address, round_up
from test.mockgpu.helpers import ptx_run

memory:dict[int, bytearray] = {}
modules:list[ctypes.Array] = [] # a function handle is the address of its module's ptx
streams:dict[int, list[Callable[[], Any]]] = {} # pending work per stream, in order: an entry returns False while it has to wait

def out(p, value, t=ctypes.c_uint64) -> int:
  ctypes.cast(p, ctypes.POINTER(t))[0] = value
  return cuda.CUDA_SUCCESS
def handle(p) -> int: return ctypes.cast(p, ctypes.c_void_p).value or 0

def enqueue(stream, work:Callable[[], Any]) -> int:
  streams[handle(stream)].append(work)
  progress = True
  while progress: # run whatever can proceed, a stream waits at its head
    progress = False
    for q in streams.values():
      while q and q[0]():
        q.pop(0)
        progress = True
  return cuda.CUDA_SUCCESS

def alloc(p, size:int) -> int:
  mem = bytearray(size)
  memory[address:=mv_address(memoryview(mem))] = mem
  return out(p, address)
def free(p) -> int:
  del memory[p]
  return cuda.CUDA_SUCCESS

def run_kernel(func:int, args:int, size:int, gx:int, gy:int, gz:int, lx:int, ly:int, lz:int, smem:int) -> bool:
  src = ctypes.string_at(func).decode()
  params = re.search(r"\.entry\s+\w+\s*\(([^)]*)\)", src).group(1) # type: ignore[union-attr]
  vals, off = [], 0
  for bits in re.findall(r"\.param\s+\.\w(\d+)", params): # the arguments at their natural alignment
    off = round_up(off, n:=int(bits) // 8)
    vals.append(int.from_bytes(ctypes.string_at(args + off, n), "little"))
    off += n
  assert off <= size, f"a {size} byte parameter buffer holds no {off} bytes of parameters"
  ptx_run(src.encode(), len(vals), (ctypes.c_void_p * len(vals))(*vals), lx, ly, lz, gx, gy, gz, smem)
  return True

def cuInit(flags): return cuda.CUDA_SUCCESS
def cuDeviceGetCount(count): return out(count, 1, ctypes.c_int32)
def cuDeviceGet(device, ordinal): return out(device, ordinal, ctypes.c_int32)
def cuDeviceComputeCapability(major, minor, dev): return out(major, 3, ctypes.c_int32) or out(minor, 5, ctypes.c_int32)
def cuCtxCreate_v2(pctx, flags, dev): return out(pctx, dev + 1)
def cuCtxSetCurrent(ctx): return cuda.CUDA_SUCCESS
def cuCtxSynchronize(): return cuda.CUDA_SUCCESS
def cuStreamCreate(pstream, flags):
  streams[sid:=len(streams) + 1] = []
  return out(pstream, sid)

def cuMemAlloc_v2(dptr, size): return alloc(dptr, size)
def cuMemHostAlloc(pp, size, flags): return alloc(pp, size)
def cuMemFree_v2(dptr): return free(dptr)
def cuMemFreeHost(p): return free(p)
def cuMemHostRegister_v2(p, size, flags): return cuda.CUDA_SUCCESS
def cuMemHostUnregister(p): return cuda.CUDA_SUCCESS

def cuModuleLoadData(module, image):
  modules.append(ctypes.create_string_buffer(ctypes.string_at(image)))
  return out(module, ctypes.addressof(modules[-1]))
def cuModuleGetFunction(hfunc, hmod, name): return out(hfunc, handle(hmod))

def cuLaunchKernel(f, gx, gy, gz, lx, ly, lz, smem, stream, params, extra):
  args, size = extra[1], ctypes.c_size_t.from_address(extra[3]).value # CU_LAUNCH_PARAM_BUFFER_POINTER, CU_LAUNCH_PARAM_BUFFER_SIZE
  return enqueue(stream, lambda: run_kernel(handle(f), args, size, gx, gy, gz, lx, ly, lz, smem))
def cuLaunchHostFunc(stream, fn, data): return enqueue(stream, lambda: fn(data) or True)
def cuMemcpyAsync(dst, src, size, stream): return enqueue(stream, lambda: ctypes.memmove(dst, src, size) or True)
def cuStreamWriteValue64_v2(stream, address, value, flags): return enqueue(stream, lambda: out(address, value) == 0)
def cuStreamWaitValue64_v2(stream, address, value, flags): return enqueue(stream, lambda: ctypes.c_uint64.from_address(address).value >= value)

for name, fn in list(globals().items()):
  if name.startswith("cu") and callable(fn): setattr(cuda.dll, name, ctypes.CFUNCTYPE(getattr(cuda, name).restype, *getattr(cuda, name).argtypes)(fn))
