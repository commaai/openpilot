from __future__ import annotations
import os
os.environ['PYOPENCL_NO_CACHE'] = '1'
import pathlib
import numpy as np
import pyopencl as cl  # type: ignore
from typing import Optional, List, Tuple
from tinygrad.helpers import DEBUG, getenv, prod, ImageDType, OSX, fromimport, diskcache
from tinygrad.ops import Compiled
from tinygrad.renderer.opencl import OpenCLRenderer
from tinygrad.runtime.lib import RawBufferCopyInOut, LRUAllocator, RawBufferTransfer
from tinygrad.codegen.kernel import LinearizerOptions

OSX_TIMING_RATIO = (125/3) if OSX else 1.0   # see test/external/external_osx_profiling.py to determine this ratio. it's in like GPU clocks or something

# TODO: if you fork and exit the child process after creating anything with cl on AMD, it hangs on e.wait()
ROCM_LLVM_PATH = pathlib.Path("/opt/rocm/llvm/bin")
#ROCM_LLVM_PATH = pathlib.Path(__file__).parents[3] / "extra/rocm/build/llvm-project/bin"
if DEBUG >= 5:
  early_exec = fromimport("extra.helpers", "enable_early_exec")()

class CLAllocator(LRUAllocator):
  def _do_alloc(self, size, dtype, device, **kwargs):
    if isinstance(dtype, ImageDType):
      # NOTE: the memory is a bit off here due to padding, it's buf.row_pitch * buf.height * 4 * dtype.itemsize
      assert size == prod(dtype.shape), f"image size mismatch {size} != {dtype.shape}"
      fmt = cl.ImageFormat(cl.channel_order.RGBA, {2: cl.channel_type.HALF_FLOAT, 4: cl.channel_type.FLOAT}[dtype.itemsize])
      buf = cl.Image(CL.cl_ctxs[int(device)], cl.mem_flags.READ_WRITE, fmt, shape=(dtype.shape[1], dtype.shape[0]))
    else:
      buf = cl.Buffer(CL.cl_ctxs[int(device)], cl.mem_flags.READ_WRITE, size * dtype.itemsize)
    setattr(buf, 'device', int(device)) # device is tracked on the underlying buffer
    return buf

class _CL:
  def __init__(self):
    cl_platforms = cl.get_platforms()
    platform_devices: List[List[cl.Device]] = [y for y in ([x.get_devices(device_type=cl.device_type.GPU) for x in cl_platforms] + [x.get_devices(device_type=cl.device_type.CPU) for x in cl_platforms]) if y]
    self.devices = [device for device in platform_devices[getenv('CL_PLATFORM', 0)] if device.name not in getenv('CL_EXCLUDE', "").split(",")]
    self.cl_platform = self.devices[0].platform
  def post_init(self, device=None):
    self.cl_ctxs: List[cl.Context] = [cl.Context(devices=[x]) for x in self.devices] if device is None else [cl.Context(devices=[self.devices[device]])]
    if DEBUG >= 1: print(f"using devices: {[ctx.devices[0].hashable_model_and_version_identifier for ctx in self.cl_ctxs]}")
    self.cl_queue: List[cl.CommandQueue] = [cl.CommandQueue(ctx, device=ctx.devices[0], properties=cl.command_queue_properties.PROFILING_ENABLE) for ctx in self.cl_ctxs]
    self.cl_allocator = CLAllocator(CL.cl_ctxs[0].devices[0].get_info(cl.device_info.GLOBAL_MEM_SIZE))
  def synchronize(self):
    for q in self.cl_queue: q.finish()
CL = _CL()
if not getenv("DELAYED_RUNTIME_INIT", False): CL.post_init()

class CLBuffer(RawBufferCopyInOut, RawBufferTransfer):
  def __init__(self, size, dtype, device='0'): super().__init__(size, dtype, allocator=CL.cl_allocator, **{'device': device})
  def _copyin(self, x:np.ndarray):
    assert not self.dtype.name.startswith("image"), f"can't copyin images {self.dtype}"
    self.event = cl.enqueue_copy(CL.cl_queue[self._buf.device], self._buf, np.require(x, requirements=['C', 'A']), is_blocking=False)
  def _copyout(self, x:np.ndarray):
    assert not self.dtype.name.startswith("image"), f"can't copyout images {self.dtype}"
    CL.cl_allocator.ensure_has_free_space(self.size, self.dtype, self._device)
    buf = cl.Buffer(CL.cl_ctxs[self._buf.device], cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR, 0, hostbuf=x.data)
    mapped, event = cl.enqueue_map_buffer(CL.cl_queue[self._buf.device], buf, cl.map_flags.WRITE, 0, self.size, dtype=self.dtype.np, is_blocking=False)
    with mapped.base: cl.enqueue_copy(CL.cl_queue[self._buf.device], mapped, self._buf, is_blocking=True, wait_for=[event] + ([self.event] if hasattr(self, "event") else []))
  def _transfer(self, x):
    if "gfx" in CL.cl_ctxs[x._buf.device].devices[0].name:
      cl.enqueue_copy_buffer_p2p_amd(CL.cl_platform, CL.cl_queue[x._buf.device], x._buf, self._buf, x.size * x.dtype.itemsize).wait()
    else: raise NotImplementedError("p2p transfer between devices not implemented on non-amd")

@diskcache
def compile_gpu(prg:str) -> bytes:
  clprg = cl.Program(CL.cl_ctxs[0], prg)
  clprg.build()
  return clprg.get_info(cl.program_info.BINARIES)[0]

class CLProgram:
  def __init__(self, name:str, prg:bytes, argdtypes=None, options=None):
    self.name, self.clprograms = name, [cl.Program(ctx, ctx.devices, [prg]*len(ctx.devices)) for ctx in CL.cl_ctxs]  # type: ignore
    self._clprgs = [clprogram.build(options=options) for clprogram in self.clprograms]
    self.clprgs = [clprg.__getattr__(name) for clprg in self._clprgs]
    if DEBUG >= 5 and not OSX:
      if 'Adreno' in CL.cl_ctxs[0].devices[0].name:
        fromimport('disassemblers.adreno', 'disasm')(prg)
      elif CL.cl_ctxs[0].devices[0].name.startswith('gfx'):
        asm = early_exec(([ROCM_LLVM_PATH / "llvm-objdump", '-d', '-'], prg))
        print('\n'.join([x for x in asm.decode('utf-8').split("\n") if 's_code_end' not in x]))
      else:
        # print the PTX for NVIDIA. TODO: probably broken for everything else
        print(prg.decode('utf-8'))
    if argdtypes is not None: self.set_argdtypes(argdtypes)

  def set_argdtypes(self, argdtypes): self.argdtypes, _ = argdtypes, [clprg.set_scalar_arg_dtypes(argdtypes) for clprg in self.clprgs]

  @staticmethod
  def max_work_group_size(): return CL.cl_ctxs[0].devices[0].max_work_group_size

  def __call__(self, *bufs, global_size:Tuple[int,int,int], local_size:Optional[Tuple[int,int,int]]=None, wait=False) -> Optional[float]:
    if not hasattr(self, 'argdtypes'): self.set_argdtypes(tuple(None if x.__class__ is CLBuffer else np.int32 for x in bufs))
    cl_bufs, wait_for = [], []
    for x in bufs:
      if x.__class__ is CLBuffer:
        cl_bufs.append(x._buf)
        if hasattr(x, "event"): wait_for.append(x.event)
      else: cl_bufs.append(x)
    e = self.clprgs[cl_bufs[0].device](CL.cl_queue[cl_bufs[0].device], [int(g*l) for g,l in zip(global_size, local_size)] if local_size is not None else global_size, local_size, *cl_bufs, wait_for=wait_for)
    if wait:
      e.wait()
      try:
        return ((e.profile.end - e.profile.start) * OSX_TIMING_RATIO) * 1e-9
      except cl.RuntimeError:   # no profiling info available
        return None
    return None

GPUBuffer = Compiled(CLBuffer, LinearizerOptions(), OpenCLRenderer, compile_gpu, CLProgram, CL.synchronize)
