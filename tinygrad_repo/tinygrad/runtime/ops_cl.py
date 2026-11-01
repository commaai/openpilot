from __future__ import annotations
import ctypes, hashlib, functools
from tinygrad.runtime.autogen import opencl as cl, libc
from tinygrad.runtime.support import c
from tinygrad.helpers import to_char_p_p, from_mv, cpu_profile, OSX, DEBUG, suppress_finalizing, unwrap, round_up, is_image_shape
from tinygrad.renderer.cstyle import OpenCLRenderer
from tinygrad.device import BufferStorage, BufferSpec, HostAllocator, Compiled, Compiler, CompileError, MMIOInterface
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher, KernelInfo, uopfunc
from tinygrad.engine.realize import get_call_kernel_args
from tinygrad.runtime.support.hcq2 import HWQueue, ccall, encode_cmdbuf, make_program

CC_CB = c.CFUNCTYPE[None, [c.POINTER[ctypes.c_char], c.POINTER[None], cl.size_t, c.POINTER[None]]]
BP_CB = c.CFUNCTYPE[None, [cl.cl_program, c.POINTER[None]]]

cl_errors = {attr: k for k in dir(cl) if k.startswith("CL_") and isinstance(attr:=getattr(cl, k), int) and attr <= 0}
def check(status):
  if status != 0: raise RuntimeError(f"OpenCL Error {status}: {cl_errors.get(status, 'Unknown error')}")
def checked(fn, *args): return (fn(*args, status:=ctypes.c_int32()), check(status.value))[0]
def info(fn, *args) -> str:
  check(fn(*args, 0, None, ctypes.byref(size:=ctypes.c_size_t())))
  return (check(fn(*args, size.value, buf:=ctypes.create_string_buffer(size.value), None)), buf.value.decode())[1]

class CLCompiler(Compiler):
  def __init__(self, dev:CLDevice):
    self.dev = dev
    super().__init__(f"compile_cl_{hashlib.md5(dev.device_name.encode() + dev.driver_version.encode()).hexdigest()}")
  def compile(self, src:str) -> bytes:
    program = checked(cl.clCreateProgramWithSource, self.dev.context, 1, to_char_p_p([src.encode()]), None)
    if cl.clBuildProgram(program, 1, self.dev.cl_dev, None, BP_CB(), None) != 0:
      raise CompileError(f"OpenCL Compile Error\n\n{info(cl.clGetProgramBuildInfo, program, self.dev.cl_dev, cl.CL_PROGRAM_BUILD_LOG)}")
    check(cl.clGetProgramInfo(program, cl.CL_PROGRAM_BINARY_SIZES, 8, ctypes.byref(size:=ctypes.c_size_t()), None))
    check(cl.clGetProgramInfo(program, cl.CL_PROGRAM_BINARIES, 8, ctypes.byref(ctypes.pointer(lib:=ctypes.create_string_buffer(size.value))), None))
    return (check(cl.clReleaseProgram(program)), bytes(lib))[1]

# *****************
# queue

@uopfunc
def cl_run(buf:UOp, cmds:tuple) -> UOp: # one in-order queue, every call after the last
  cb = buf
  for fn, *args in cmds:
    h = ccall(fn, *[cb.index(a[0]) if isinstance(a, tuple) else a if isinstance(a, UOp) else cb.bitcast(dtypes.uint64).index(a).load() for a in args])
    if fn is cl.clCreateImage: h = buf.bitcast(dtypes.uint64).index(args[-1][0] // 8).store(h) # the image replaces its errcode, not chained
    cb = cb.after(h)
  return cb.sink()

class CLQueue(HWQueue):
  dev:CLDevice
  def q64(self, *vals) -> int: return self.q(*[UOp.const(v, dtypes.uint64) if isinstance(v, int) else v.cast(dtypes.uint64) for v in vals])
  def call(self, fn, *args): self.cmds.append((fn, *[UOp.const(a, dtypes.uint64) if isinstance(a, int) else a for a in args]))

  def exec(self, call:UOp, prg:UOp):
    args, images = get_call_kernel_args(call, prg, self.devs), []
    kernel = (p:=make_program(prg, 8, self.devs[0])).after(p.store(prg)).bitcast(dtypes.u64).index(0).load()
    for i, (_, _, dt, shape) in enumerate(prg.to_elf().signature):
      if is_image_shape(shape): # [format][desc][errcode]
        desc = (cl.CL_MEM_OBJECT_IMAGE2D, shape[1], shape[0], 0, 0, round_up(shape[1], 256 if OSX else 1) * 4 * dt.itemsize, 0, 0, args[i])
        end = self.q64(cl.CL_RGBA | {2:cl.CL_HALF_FLOAT, 4:cl.CL_FLOAT}[dt.itemsize] << 32, *desc, 0)
        self.call(cl.clCreateImage, self.dev.handles[0], cl.CL_MEM_READ_WRITE, (end - 88,), (end - 80,), 0, (end - 8,))
        images.append((end - 8) // 8)
      else: end = self.q64(args[i])
      self.call(cl.clSetKernelArg, kernel, i, args[i].dtype.itemsize, (end - 8,))
    end = self.q64(*[g * l for g, l in zip(prg.arg.global_size, prg.arg.local_size)], *prg.arg.local_size) # [global][local]
    self.call(cl.clEnqueueNDRangeKernel, self.dev.handles[1], kernel, 3, 0, (end - 48,), (end - 24,), 0, 0, 0)
    self.cmds += [(cl.clReleaseMemObject, w) for w in images]

  def wait(self, sig:UOp, val:UOp, eq=False): pass # the queue is in order
  def signal(self, sig:UOp, val:UOp): self.call(libc.memcpy, sig.getaddr(self.devs), (self.q64(val) - 8,), 8) # synchronize finishes the queue
  def timestamp(self, sig:UOp): self.call(cl.clEnqueueMarkerWithWaitList, self.dev.handles[1], 0, 0, sig.getaddr(self.devs)) # [event][time]

  def encode(self) -> UOp:
    self.cmds:list[tuple] = []
    return cl_run(encode_cmdbuf(self, self.lin), tuple(self.cmds))

Compiled.pm_encode += PatternMatcher([(UPat.sink(arg=KernelInfo("cl_compute_submit")).call(UPat()).after(allow_any_len=True, name="s"),
                                       lambda s: CLQueue(s).encode())])

# *****************
# device

class CLAllocator(HostAllocator):
  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage:
    if options.host or options.cpu_access: return super()._alloc(size, options)
    return BufferStorage(int.from_bytes(bytes(mem:=checked(cl.clCreateBuffer, self.dev.context, cl.CL_MEM_READ_WRITE, size, None)), "little"), mem)

  @suppress_finalizing
  def _free(self, storage:BufferStorage, options:BufferSpec):
    if storage.host is None: check(cl.clReleaseMemObject(storage.meta))
  def _copy(self, fn, name:str, mem:int, host, size:int): # blocking, so the host memory can go
    with cpu_profile(name, f"{self.dev.device}:COPY"): check(fn(self.dev.queue, ctypes.cast(mem, cl.cl_mem), True, 0, size, host, 0, None, None))
  def _copyin(self, dest:int, src:memoryview): self._copy(cl.clEnqueueWriteBuffer, f"TINY -> {self.dev.device}", dest, bytes(src), src.nbytes)
  def _copyout(self, dest:memoryview, src:int): self._copy(cl.clEnqueueReadBuffer, f"{self.dev.device} -> TINY", src, from_mv(dest), dest.nbytes)

class CLDevice(Compiled):
  device_ids = None                 # this is global and only initted once
  timestamp_divider = 3000 / 125 if OSX else 1000.0

  def __init__(self, device:str=""):
    if CLDevice.device_ids is None:
      check(cl.clGetPlatformIDs(1, platform:=cl.cl_platform_id(), num_platforms:=ctypes.c_uint32()))
      for device_type in [cl.CL_DEVICE_TYPE_GPU, cl.CL_DEVICE_TYPE_DEFAULT]:
        if cl.clGetDeviceIDs(platform, device_type, 0, None, num_devices:=ctypes.c_uint32()) == 0 and num_devices.value != 0: break
      if DEBUG >= 1: print(f"CLDevice: got {num_platforms.value} platforms and {num_devices.value} devices")
      CLDevice.device_ids = (check(cl.clGetDeviceIDs(platform, device_type, num_devices, ids:=(cl.cl_device_id * num_devices.value)(), None)), ids)[1]

    self.cl_dev = CLDevice.device_ids[0 if ":" not in device else int(device.split(":")[1])]
    self.device_name, self.driver_version = [info(cl.clGetDeviceInfo, self.cl_dev, p) for p in (cl.CL_DEVICE_NAME, cl.CL_DRIVER_VERSION)]
    if DEBUG >= 1: print(f"CLDevice: opening {self.device_name} with version {self.driver_version}")
    self.context = checked(cl.clCreateContext, None, 1, self.cl_dev, CC_CB(), None)
    self.queue = checked(cl.clCreateCommandQueue, self.context, self.cl_dev, cl.CL_QUEUE_PROFILING_ENABLE)
    self.handles = [int.from_bytes(bytes(h), "little") for h in (self.context, self.queue)]

    arch = ",".join(exts:=info(cl.clGetDeviceInfo, self.cl_dev, cl.CL_DEVICE_EXTENSIONS).split())
    if "cl_khr_image2d_from_buffer" in exts:
      check(cl.clGetDeviceInfo(self.cl_dev, cl.CL_DEVICE_IMAGE_PITCH_ALIGNMENT, 4, ctypes.byref(ipa := ctypes.c_uint32()), None))
      arch += f",IMAGE_PITCH_ALIGNMENT={ipa.value}"
    super().__init__(device, CLAllocator(self), [OpenCLRenderer], None, arch=arch)
    Compiled.pm_link += PatternMatcher([(UPat(Ops.ALLOC, name="p").store(UPat(Ops.PROGRAM, name="prg")), lambda p, prg, d=self:
      p.store(UOp(Ops.BINARY, arg=d.function(prg.src[3].arg, prg.src[0].arg.function_name))) if p.device == d.device else None)])

  @functools.cache
  def function(self, src:bytes, name:str) -> bytes:
    lib = CLCompiler(self).compile_cached(src.decode())
    program = checked(cl.clCreateProgramWithBinary, self.context, 1, self.cl_dev, ctypes.c_size_t(len(lib)), to_char_p_p([lib], ctypes.c_ubyte), None)
    check(cl.clBuildProgram(program, 1, self.cl_dev, None, BP_CB(), None)) # NOTE: OSX requires this
    return bytes(checked(cl.clCreateKernel, program, name.encode()))

  def count(self) -> int: return len(unwrap(self.device_ids))
  def _wait_signal(self, sig:MMIOInterface|memoryview, value:int, timeout:int|None=None):
    check(cl.clFinish(self.queue))
    for s, i in [(s, i) for b, st in self.prof_ents for s in [b.host.view(fmt='Q')] for i in (st - 1, st + 1) if s[i]]: # [event, 0]
      check(cl.clGetEventProfilingInfo(ev:=ctypes.cast(s[i], cl.cl_event), cl.CL_PROFILING_COMMAND_END, 8, s.addr + i * 8 + 8, None))
      s[i] = (check(cl.clReleaseEvent(ev)), 0)[1]
