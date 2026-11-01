from __future__ import annotations
import ctypes, functools, mmap, struct, time
from typing import Any, Callable
from tinygrad.helpers import DEBUG, DEV, getenv, unwrap, to_name
from tinygrad.device import Device, Buffer, BufferStorage, BufferSpec, Allocator, Compiled, MMIOInterface, HCQ_RUNTIME_DEV
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher, KernelInfo
from tinygrad.engine.realize import get_call_kernel_args
from tinygrad.renderer.cstyle import CUDARenderer, NVCCRenderer
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.runtime.autogen import cuda
from tinygrad.runtime.support.compiler_cuda import pretty_ptx
from tinygrad.runtime.support.c import init_c_var
from tinygrad.runtime.support.hcq2 import ccall, layout_args, patch, make_program
if getenv("IOCTL"): import extra.nv_gpu_driver.nv_ioctl  # noqa: F401  # pylint: disable=unused-import
if DEV.target("CUDA").interface == "MOCK": import test.mockgpu.cuda.cuda  # noqa: F401  # pylint: disable=unused-import

def check(status:int):
  if status != 0: raise RuntimeError(f"CUDA Error {status}, {cuda.enum_cudaError_enum.get(status, 'unknown')}")

# *****************
# runtime

def cu_launch_kernel(ctx:CUDADevice, call:UOp, prg:UOp) -> tuple:
  rows = layout_args(get_call_kernel_args(call, prg, (ctx.device,)), 48)
  buf = UOp.alloc((max([o + w.dtype.itemsize for o, w in rows], default=48),), dtypes.uint8, device=HCQ_RUNTIME_DEV.device, name="kernargs")
  addr = buf.getaddr(HCQ_RUNTIME_DEV.device) # [extra: buffer pointer, &args, buffer size, &size, end][size][args]
  hdr = layout_args([UOp.const(v, dtypes.uint64) if isinstance(v, int) else v for v in (1, addr+48, 2, addr+40, ctx.launch_end, buf.max_numel()-48)])
  func = (p:=make_program(prg, 8, ctx.device)).after(p.store(prg)).bitcast(dtypes.u64).index(0).load()
  return (cuda.cuLaunchKernel, func, *prg.arg.global_size, *prg.arg.local_size, 0, None, 0, patch(buf, hdr + rows).index(0))

pm_cuda = PatternMatcher([ # None is the stream
  (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, name="prg"),), name="call", allow_any_len=True), cu_launch_kernel),
  (UPat(Ops.CALL, src=(UPat(Ops.STORE), UPat(name="dst"), UPat(name="src")), allow_any_len=True),
   lambda ctx, dst, src: (cuda.cuMemcpyAsync, dst.getaddr(ctx.device), src.getaddr(ctx.device), src.max_numel() * src.dtype.itemsize, None)),
  (UPat(Ops.CALL, src=(UPat.custom_function("copy"), UPat(name="dst"), UPat(name="src"), UPat(name="n"))),
   lambda ctx, dst, src, n: (cuda.cuMemcpyAsync, dst.getaddr(ctx.device), src.getaddr(ctx.device), n.val, None)),
  (UPat(Ops.CALL, src=(UPat.custom_function("wait"), UPat(name="sig"), UPat(name="val"))),
   lambda ctx, sig, val: (cuda.cuStreamWaitValue64_v2, None, sig.getaddr(ctx.device), val, cuda.CU_STREAM_WAIT_VALUE_GEQ)),
  (UPat(Ops.CALL, src=(UPat.custom_function("store"), UPat(name="sig"), UPat(name="val"))),
   lambda ctx, sig, val: (cuda.cuStreamWriteValue64_v2, None, sig.getaddr(ctx.device), val, cuda.CU_STREAM_WRITE_VALUE_DEFAULT)),
  (UPat(Ops.CALL, src=(UPat.custom_function("timestamp"), UPat(name="sig"))), # [sig][stamp]
   lambda ctx, sig: (cuda.cuLaunchHostFunc, None, UOp.alloc((1,), dtypes.uint64, 0, device=ctx.device, name="stamp").getaddr(ctx.host),
                     sig[1:2].getaddr(ctx.device))),
])

def cuda_submit(s:UOp, ccallfn:Callable[[Any, *tuple[UOp, ...]], UOp]=ccall) -> UOp:
  (call, *deps), q = s.src, s.src[0].src[1]
  rt_vars = UOp.param(0, dtypes.uint64, 3, q.device, name="rt_vars") # [ctx, compute stream, copy stream]
  h = ccallfn(cuda.cuCtxSetCurrent, rt_vars.index(0).load())
  for fn, *args in [x for c in call.body.src if (x:=pm_cuda.rewrite(c, ctx=Device[q.device])) is not None]:
    stream = rt_vars.after(h).index(2 if q.arg.name.startswith("COPY") else 1).load() # the stream after the last call
    h = ccallfn(fn, *[stream if a is None else UOp.const(a, dtypes.uint64) if isinstance(a, int) else a for a in args])
  return h.sink().call(UOp.alloc((3,), dtypes.uint64, 0, device=q.device, name="handles").after(*deps), name=call.body.arg.name)

Compiled.pm_encode += PatternMatcher([
  (UPat.sink(arg=KernelInfo("cuda_compute_submit")).call(UPat()).after(allow_any_len=True, name="s"), cuda_submit),
  (UPat.sink(arg=KernelInfo("cuda_copy_submit")).call(UPat()).after(allow_any_len=True, name="s"), cuda_submit),
])

# *****************
# device

class CUDAAllocator(Allocator['CUDADevice']):
  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage:
    check((api:=self.dev.api).cuCtxSetCurrent(self.dev.context))
    if options.external_ptr: return BufferStorage(options.external_ptr)
    if options.host or options.cpu_access:
      ptr = init_c_var(ctypes.c_void_p, lambda x: check(api.cuMemHostAlloc(ctypes.byref(x), size, 0))).value
      return BufferStorage(ptr, None, MMIOInterface(ptr, size))
    return BufferStorage(init_c_var(api.CUdeviceptr, lambda x: check(api.cuMemAlloc_v2(ctypes.byref(x), size))).value)

  def _free(self, storage:BufferStorage, options:BufferSpec):
    self.dev.synchronize()
    check((self.dev.api.cuMemFreeHost if options.host or options.cpu_access else self.dev.api.cuMemFree_v2)(storage.buf))

  def _map(self, buf:Buffer) -> BufferStorage:
    if buf.device.split(":")[0] == self.dev.device.split(":")[0]: return BufferStorage(buf._buf)
    if (host:=buf.get_storage().host) is None or host.addr % mmap.PAGESIZE: raise RuntimeError(f"{buf.device} memory is not page aligned host memory")
    check((api:=self.dev.api).cuCtxSetCurrent(self.dev.context))
    if (status:=api.cuMemHostRegister_v2(host.addr, buf.nbytes, 0)) != cuda.CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
      check(status) # another device pinned it already
    addr = init_c_var(api.CUdeviceptr, lambda x: check(api.cuMemHostGetDevicePointer_v2(ctypes.byref(x), host.addr, 0))).value
    return BufferStorage(addr, status == cuda.CUDA_SUCCESS)
  def _unmap(self, mapping:BufferStorage):
    if mapping.meta: check(self.dev.api.cuMemHostUnregister(unwrap(mapping.host).addr))
  def _offset(self, buf:int, size:int, offset:int) -> int: return buf + offset

class CUDADevice(Compiled):
  api:Any = cuda
  launch_end = 0 # CU_LAUNCH_PARAM_END
  renderers = [CUDARenderer, PTXRenderer, NVCCRenderer]

  def __init__(self, device:str=""):
    device_id = int(device.split(":")[1]) if ":" in device else 0
    check((api:=self.api).cuInit(0))
    self.cu_device = init_c_var(api.CUdevice, lambda x: check(api.cuDeviceGet(ctypes.byref(x), device_id)))
    self.context = init_c_var(api.CUcontext, lambda x: check(api.cuCtxCreate_v2(ctypes.byref(x), 0, self.cu_device)))
    self.streams = [init_c_var(api.CUstream, lambda x: check(api.cuStreamCreate(ctypes.byref(x), cuda.CU_STREAM_NON_BLOCKING))) for _ in range(2)]
    super().__init__(device, CUDAAllocator(self), self.renderers, None, arch=self.get_arch(device_id))

    # [ctx, compute stream, copy stream], and the host function that writes a timestamp
    handles = Buffer(HCQ_RUNTIME_DEV.device, 24, initial_value=b"".join(map(bytes, (self.context, *self.streams))))
    fn = cuda.CUhostFn(lambda slot: ctypes.memmove(slot, struct.pack("Q", time.perf_counter_ns()), 8))
    stamp = Buffer(HCQ_RUNTIME_DEV.device, 8, opaque=BufferStorage(int.from_bytes(bytes(fn), "little"), fn))
    Compiled.link_bufs |= {to_name(self.device, "handles"): handles, to_name(self.device, "stamp"): stamp}
    Compiled.pm_link += PatternMatcher([(UPat(Ops.ALLOC, name="p").store(UPat(Ops.PROGRAM, name="prg")), lambda p, prg, d=self:
      p.store(UOp(Ops.BINARY, arg=d.function(prg.src[3].arg, prg.src[0].arg.function_name))) if p.device == d.device else None)])

  def get_arch(self, device_id:int) -> str:
    check(cuda.cuDeviceComputeCapability(ctypes.byref(major:=ctypes.c_int()), ctypes.byref(minor:=ctypes.c_int()), device_id))
    return f"sm_{major.value}{minor.value}"

  @functools.cache
  def function(self, lib:bytes, name:str) -> bytes:
    if DEBUG >= 5 and self.api is cuda: print("\n".join([f"{i+1:>3} {line}" for i, line in enumerate(pretty_ptx(lib.decode()).split("\n"))]))
    check(self.api.cuCtxSetCurrent(self.context))
    check(self.api.cuModuleLoadData(ctypes.byref(module:=self.api.CUmodule()), lib))
    check(self.api.cuModuleGetFunction(ctypes.byref(func:=self.api.CUfunction()), module, name.encode()))
    return bytes(func)

  def count(self) -> int: return init_c_var(ctypes.c_int, lambda x: check(self.api.cuDeviceGetCount(ctypes.byref(x)))).value

  def _wait_signal(self, sig:MMIOInterface|memoryview, value:int, timeout:int|None=None): # a fault raises here
    check(self.api.cuCtxSetCurrent(self.context))
    check(self.api.cuCtxSynchronize())
