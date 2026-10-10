from __future__ import annotations
import ctypes, functools, mmap, struct, time
from tinygrad.helpers import DEBUG, DEV, getenv, unwrap
from tinygrad.device import Buffer, BufferStorage, BufferSpec, Allocator, Compiled, MMIOInterface, HCQ_RUNTIME_DEV
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher, uopfunc
from tinygrad.engine.realize import get_call_arg_uops, get_call_var_uops
from tinygrad.renderer.cstyle import CUDARenderer, NVCCRenderer
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.runtime.autogen import cuda
from tinygrad.runtime.support.compiler_cuda import pretty_ptx
from tinygrad.runtime.support.c import init_c_var
from tinygrad.runtime.support.hcq2 import HWQueue, ccall, layout_args, pack_args, encode_cmdbuf
if getenv("IOCTL"): import extra.nv_gpu_driver.nv_ioctl  # noqa: F401  # pylint: disable=unused-import
if DEV.target("CUDA").interface == "MOCK": import test.mockgpu.cuda.cuda  # noqa: F401  # pylint: disable=unused-import

def check(status:int):
  if status != 0: raise RuntimeError(f"CUDA Error {status}, {cuda.enum_cudaError_enum.get(status, 'unknown')}")

def as_int(handle) -> int: return unwrap(ctypes.cast(handle, ctypes.c_void_p).value)

def host_stamp(slot:int): ctypes.c_uint64.from_address(slot).value = time.perf_counter_ns()
def extern(ptr:int, meta=None) -> Buffer: return Buffer(HCQ_RUNTIME_DEV.value, 8, opaque=BufferStorage(ptr, meta))

# *****************
# queue

@uopfunc
def cuda_run(cmdbuf:UOp, rt_vars:UOp, cmds:tuple, copy:bool) -> UOp: # rt_vars: [ctx, compute stream, copy stream, status]
  words, h = cmdbuf.bitcast(dtypes.uint64), ccall(cuda.cuCtxSetCurrent, rt_vars.index(0).load())
  for fn, *args in cmds:
    s = rt_vars.after(h).index(2 if copy else 1).load() # the stream after the last call
    h = ccall(fn, *[s if a is None else cmdbuf.index(a[0]) if isinstance(a, tuple) else words.index(a).load() for a in args])
  return rt_vars.after(h).index(3).store(h.cast(dtypes.uint64)).sink()

class CUDAQueue(HWQueue):
  dev:CUDADevice
  def call(self, fn, *args): # the values go in the cmdbuf as words
    vals = [UOp.const(a, dtypes.uint64) if isinstance(a, int) else a for a in args]
    self.cmds.append((fn, *[a if a is None or isinstance(a, tuple) else self.q(a.cast(dtypes.uint64)) // 8 - 1 for a in vals]))
  def extern(self, tag) -> UOp: return UOp.alloc((1,), dtypes.uint64, 0, device=self.devs[0]).rtag(tag).getaddr(self.dev.host)

  def exec(self, call:UOp, prg:UOp):
    obj, bufs, vals = prg.to_elf(), get_call_arg_uops(call), get_call_var_uops(call, prg)
    rows = layout_args([bufs[i].getaddr(self.devs) for i in prg.arg.globals] + [v.ccast(var.dtype) for v, var in zip(vals, prg.arg.vars)], 8)
    size = max([o + w.dtype.itemsize for o, w in rows], default=8) - 8
    addr = UOp(Ops.LINEAR, src=tuple(pack_args([(0, UOp.const(size, dtypes.uint64))] + rows, 8 + size)), arg="kernargs").getaddr(self.devs)
    # extra: [buffer pointer, &args, buffer size, &size, end]
    self.call(cuda.cuLaunchKernel, self.extern((self.dev.tag("function"), obj.lib, obj.name)), *prg.arg.global_size, *prg.arg.local_size, 0, None, 0,
              (self.q(*[UOp.const(v, dtypes.uint64) if isinstance(v, int) else v for v in (1, addr + 8, 2, addr, 0)]) - 40,))

  def copy(self, dst:UOp, src:UOp, sz:int): self.call(cuda.cuMemcpyAsync, dst.getaddr(self.devs), src.getaddr(self.devs), sz, None)
  def wait(self, sig:UOp, val:UOp): self.call(cuda.cuStreamWaitValue64_v2, None, sig.getaddr(self.devs), val, cuda.CU_STREAM_WAIT_VALUE_GEQ)
  def signal(self, sig:UOp, val:UOp): self.call(cuda.cuStreamWriteValue64_v2, None, sig.getaddr(self.devs), val, cuda.CU_STREAM_WRITE_VALUE_DEFAULT)
  def timestamp(self, sig:UOp): self.call(cuda.cuLaunchHostFunc, None, self.extern(self.dev.tag("stamp")), sig[1:2].getaddr(self.devs)) # [sig][stamp]

  def encode(self) -> UOp:
    self.cmds:list[tuple] = []
    rt_vars = UOp.alloc((4,), dtypes.uint64, 0, device=self.devs[0]).rtag(self.dev.tag("cuda"))
    return cuda_run(encode_cmdbuf(self, self.lin), rt_vars, tuple(self.cmds), self.queue.startswith("COPY"))

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
    if buf.device.startswith("CUDA"): return BufferStorage(buf._buf)
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
    (UPat(Ops.CALL, src=(UPat.custom_function("submit_cuda_compute"), UPat()), name="s"), lambda s: CUDAQueue(s).encode()),
    (UPat(Ops.CALL, src=(UPat.custom_function("submit_cuda_copy"), UPat()), name="s"), lambda s: CUDAQueue(s).encode()),
  ])

  def __init__(self, device:str=""):
    device_id = int(device.split(":")[1]) if ":" in device else 0
    check(cuda.cuInit(0))
    self.cu_device = init_c_var(cuda.CUdevice, lambda x: check(cuda.cuDeviceGet(ctypes.byref(x), device_id)))
    self.context = init_c_var(cuda.CUcontext, lambda x: check(cuda.cuCtxCreate_v2(ctypes.byref(x), 0, self.cu_device)))
    check(cuda.cuDeviceComputeCapability(ctypes.byref(major:=ctypes.c_int()), ctypes.byref(minor:=ctypes.c_int()), device_id))
    self.streams = [init_c_var(cuda.CUstream, lambda x: check(cuda.cuStreamCreate(ctypes.byref(x), cuda.CU_STREAM_NON_BLOCKING))) for _ in range(2)]
    super().__init__(device, CUDAAllocator(self), [CUDARenderer, PTXRenderer, NVCCRenderer], None, arch=f"sm_{major.value}{minor.value}")
    Compiled.pm_bufferize += PatternMatcher([
      (UPat(Ops.ALLOC, tag=self.tag("cuda")), lambda d=self: d.handles),
      (UPat(Ops.ALLOC, tag=self.tag("stamp")), lambda d=self: d.stamp),
      (UPat(Ops.ALLOC, name="b"), lambda b, d=self: d.function(*b.tag[1:]) if isinstance(b.tag, tuple) and b.tag[0] == d.tag("function") else None)])

  @functools.cached_property
  def handles(self) -> Buffer:
    return Buffer(HCQ_RUNTIME_DEV.value, 32, initial_value=struct.pack("4Q", *[as_int(h) for h in (self.context, *self.streams)], 0))

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
