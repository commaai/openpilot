from __future__ import annotations
import ctypes, functools, mmap, struct, time
from tinygrad.helpers import DEBUG, DEV, getenv, unwrap
from tinygrad.device import Buffer, BufferStorage, BufferSpec, Allocator, Compiled, MMIOInterface, HCQ_RUNTIME_DEV
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher
from tinygrad.engine.realize import get_call_arg_uops, get_call_var_uops
from tinygrad.renderer.cstyle import CUDARenderer, NVCCRenderer
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.runtime.autogen import cuda
from tinygrad.runtime.support.compiler_cuda import pretty_ptx
from tinygrad.runtime.support.c import init_c_var
from tinygrad.runtime.support.hcq2 import HWQueue, EncodeCtx, encode_submit, ccall, rt_addr, layout_args, pack_args
if getenv("IOCTL"): import extra.nv_gpu_driver.nv_ioctl  # noqa: F401  # pylint: disable=unused-import
if DEV.target("CUDA").interface == "MOCK": import test.mockgpu.cuda.cuda  # noqa: F401  # pylint: disable=unused-import

def check(status:int):
  if status != 0: raise RuntimeError(f"CUDA Error {status}, {cuda.enum_cudaError_enum.get(status, 'unknown')}")

def as_int(handle) -> int: return unwrap(ctypes.cast(handle, ctypes.c_void_p).value)

def host_stamp(slot:int): ctypes.c_uint64.from_address(slot).value = time.perf_counter_ns()
def extern(ptr:int, meta=None) -> Buffer: return Buffer(HCQ_RUNTIME_DEV.value, 1, dtypes.uint64, opaque=BufferStorage(ptr, meta))

# *****************
# queue

class CUDAQueue(HWQueue):
  dev:CUDADevice
  def __init__(self, ctx:EncodeCtx, submit:UOp):
    super().__init__(ctx, submit)
    self.rt_vars = UOp.placeholder((4,), dtypes.uint64, 0, device=self.devs, tag="cuda") # [context, compute stream, copy stream, status]
    self.kernargs = UOp.placeholder((8,), dtypes.uint8, device=self.devs)
    self.h = ccall(cuda.cuCtxSetCurrent, self.rt_vars.index(0).load())

  @property
  def stream(self) -> UOp: return self.rt_vars.after(self.h).index(2 if self.queue.startswith("COPY") else 1).load() # read after the last call
  def extern(self, tag) -> UOp: return rt_addr(UOp.placeholder((1,), dtypes.uint64, 0, device=self.devs, tag=tag), self.dev.host)

  def launch(self, func:UOp, global_size, local_size, args:list[UOp]):
    rows = layout_args(args, 8)
    size = max([o + w.dtype.itemsize for o, w in rows], default=8) - 8
    addr = UOp(Ops.LINEAR, src=tuple(pack_args([(0, UOp.const(size, dtypes.uint64))] + rows, 8 + size)), arg="kernargs").getaddr(self.devs)

    # use .q() to stack kernargs descs
    extra = self.q(UOp.const(1, dtypes.uint64), addr + 8, UOp.const(2, dtypes.uint64), addr, UOp.const(0, dtypes.uint64)) - 40
    self.h = ccall(cuda.cuLaunchKernel, func, *global_size, *local_size, 0, self.stream, UOp.const(0, dtypes.uint64), self.kernargs.index(extra))

  def exec(self, call:UOp, prg:UOp):
    obj, bufs, vals = prg.to_elf(), get_call_arg_uops(call), get_call_var_uops(call, prg)
    self.launch(self.extern(("function", obj.lib, obj.name)), prg.arg.global_size, prg.arg.local_size,
                [bufs[i].getaddr(self.devs) for i in prg.arg.globals] + [v.ccast(var.dtype) for v, var in zip(vals, prg.arg.vars)])

  def copy(self, call:UOp):
    self.h = ccall(cuda.cuMemcpyAsync, *[rt_addr(a, self.devs) for a in call.src[1:3]], UOp.const(call.src[2].nbytes(), dtypes.uint64), self.stream)

  def wait(self, signal:UOp, value:UOp):
    self.h = ccall(cuda.cuStreamWaitValue64_v2, self.stream, rt_addr(signal, self.devs), value, cuda.CU_STREAM_WAIT_VALUE_GEQ)

  def signal(self, signal:UOp, value:UOp):
    self.h = ccall(cuda.cuStreamWriteValue64_v2, self.stream, rt_addr(signal, self.devs), value, cuda.CU_STREAM_WRITE_VALUE_DEFAULT)

  def timestamp(self, signal:UOp): # a slot is [signal][timestamp]
    self.h = ccall(cuda.cuLaunchHostFunc, self.stream, self.extern("stamp"), rt_addr(signal[1:2], self.devs))

  def submit(self, ka:UOp) -> UOp: return self.rt_vars.after(self.h).index(3).store(self.h.cast(dtypes.uint64)).substitute({self.kernargs: ka})

# *****************
# device

class CUDAAllocator(Allocator['CUDADevice']):
  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage:
    check(cuda.cuCtxSetCurrent(self.dev.context))
    if options.external_ptr: return BufferStorage(options.external_ptr)
    if options.host or options.cpu_access:
      ptr = init_c_var(ctypes.c_void_p, lambda x: check(cuda.cuMemHostAlloc(ctypes.byref(x), size, 0))).value
      return BufferStorage(ptr, None, MMIOInterface(ptr, size))
    return BufferStorage(init_c_var(cuda.CUdeviceptr, lambda x: check(cuda.cuMemAlloc_v2(ctypes.byref(x), size))).value)

  def _free(self, storage:BufferStorage, options:BufferSpec):
    self.dev.synchronize()
    check((cuda.cuMemFreeHost if options.host or options.cpu_access else cuda.cuMemFree_v2)(storage.buf))

  def _map(self, buf:Buffer) -> BufferStorage:
    if buf.device.startswith("CUDA"):
      if buf.get_storage().host is None: raise RuntimeError(f"{buf.device} device memory is only reachable through the host")
      return BufferStorage(buf._buf)
    if (host:=buf.get_storage().host) is None or host.addr % mmap.PAGESIZE: raise RuntimeError(f"{buf.device} memory is not page aligned host memory")
    check(cuda.cuCtxSetCurrent(self.dev.context))
    if (status:=cuda.cuMemHostRegister_v2(host.addr, buf.nbytes, 0)) != cuda.CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
      check(status) # another device pinned it already
    return BufferStorage(host.addr, status == cuda.CUDA_SUCCESS)
  def _unmap(self, mapping:BufferStorage):
    if mapping.meta: check(cuda.cuMemHostUnregister(mapping.buf))
  def _offset(self, buf:int, size:int, offset:int) -> int: return buf + offset

class CUDADevice(Compiled):
  pm_encode = PatternMatcher([
    (UPat(Ops.CUSTOM_FUNCTION, arg="submit_cuda_compute", name="submit"), lambda ctx, submit: encode_submit(CUDAQueue(ctx, submit))),
    (UPat(Ops.CUSTOM_FUNCTION, arg="submit_cuda_copy", name="submit"), lambda ctx, submit: encode_submit(CUDAQueue(ctx, submit))),
  ])

  def __init__(self, device:str=""):
    device_id = int(device.split(":")[1]) if ":" in device else 0
    check(cuda.cuInit(0))
    self.cu_device = init_c_var(cuda.CUdevice, lambda x: check(cuda.cuDeviceGet(ctypes.byref(x), device_id)))
    self.context = init_c_var(cuda.CUcontext, lambda x: check(cuda.cuCtxCreate_v2(ctypes.byref(x), 0, self.cu_device)))
    check(cuda.cuDeviceComputeCapability(ctypes.byref(major:=ctypes.c_int()), ctypes.byref(minor:=ctypes.c_int()), device_id))
    self.streams = [init_c_var(cuda.CUstream, lambda x: check(cuda.cuStreamCreate(ctypes.byref(x), cuda.CU_STREAM_NON_BLOCKING))) for _ in range(2)]
    super().__init__(device, CUDAAllocator(self), [CUDARenderer, PTXRenderer, NVCCRenderer], None, arch=f"sm_{major.value}{minor.value}")
    self.pm_bufferize = PatternMatcher([
      (UPat(Ops.PARAM, tag="cuda"), lambda ctx: ctx.handles),
      (UPat(Ops.PARAM, tag="stamp"), lambda ctx: ctx.stamp),
      (UPat(Ops.PARAM, name="b"), lambda ctx, b: ctx.function(*b.tag[1:]) if isinstance(b.tag, tuple) and b.tag[0] == "function" else None),
    ]) + self.pm_bufferize

  @functools.cached_property
  def handles(self) -> Buffer:
    return Buffer(HCQ_RUNTIME_DEV.value, 4, dtypes.uint64, initial_value=struct.pack("4Q", *[as_int(h) for h in (self.context, *self.streams)], 0))

  @functools.cached_property
  def stamp(self) -> Buffer: return extern(as_int(fn:=cuda.CUhostFn(host_stamp)), fn)

  @functools.cache
  def function(self, lib:bytes, name:str) -> Buffer:
    if DEBUG >= 5: print("\n".join([f"{i+1:>3} {line}" for i, line in enumerate(pretty_ptx(lib.decode()).split("\n"))]))
    check(cuda.cuCtxSetCurrent(self.context))
    check(cuda.cuModuleLoadData(ctypes.byref(module:=cuda.CUmodule()), lib))
    check(cuda.cuModuleGetFunction(ctypes.byref(func:=cuda.CUfunction()), module, name.encode()))
    return extern(as_int(func))

  def count(self) -> int: return init_c_var(ctypes.c_int, lambda x: check(cuda.cuDeviceGetCount(ctypes.byref(x)))).value

  def _wait_signal(self, sig:MMIOInterface|memoryview, value:int, timeout:int|None=None): # a fault raises here
    check(cuda.cuCtxSetCurrent(self.context))
    check(cuda.cuCtxSynchronize())
