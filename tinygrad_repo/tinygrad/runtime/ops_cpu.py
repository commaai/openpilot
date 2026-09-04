from __future__ import annotations
import platform, sys, ctypes, functools, mmap, array, struct, time
from dataclasses import replace
from typing import cast, Callable
from tinygrad.helpers import to_mv, from_mv, OSX, WIN, Context, mv_address, suppress_finalizing, unwrap, data64_le
from tinygrad.device import Buffer, BufferSpec, TinyELF, Program, Device
from tinygrad.runtime.support.hcq import HCQBuffer, MMIOInterface
from tinygrad.runtime.support.hcq2 import HCQ2Compiled, HCQAllocator, make_buf, hcq_size_var
from tinygrad.runtime.support.c import DLL
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.renderer.llvmir import CPULLVMRenderer
from tinygrad.renderer.nir import LVPRenderer
from tinygrad.renderer.isa.x86 import X86Renderer
from tinygrad.runtime.support.elf import jit_loader
from tinygrad.runtime.autogen import libc
from tinygrad.codegen import do_to_program
from tinygrad.engine.realize import get_call_arg_uops, get_call_var_uops, get_runtime
from tinygrad import UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import KernelInfo, Ops, UPat, PatternMatcher

MAX_ARGS, CMD_SIZE, FUNCS = 63, 64, (() if WIN else ('clock_gettime',))

# *****************
# 1. signal programs

def signal_prog():
  val = UOp.param(1, dtypes.int, (), vmin_vmax=(0, dtypes.int.max), name="value", addrspace=AddrSpace.ALU)
  return UOp.param(0, dtypes.uint32, 1)[0].store(val.cast(dtypes.uint32))

def wait_prog():
  val = UOp.param(1, dtypes.int, (), vmin_vmax=(0, dtypes.int.max), name="value", addrspace=AddrSpace.ALU)
  return (v:=UOp.param(0, dtypes.uint32, 1, volatile=True).after(l:=UOp.loop(0))[0].load()).end(l, v < val.cast(dtypes.uint32))

def timestamp_prog():
  if WIN: val = UOp.const(0, dtypes.uint64)
  else:
    fn, ts = UOp.param(1, dtypes.uint64, 1), UOp.placeholder((2,), dtypes.uint64, slot=0, addrspace=AddrSpace.REG)
    call = fn[0].load().call(UOp.const(6 if OSX else 1, dtypes.int), ts[0], ret_dtype=dtypes.void) # clock_gettime(CLOCK_MONOTONIC, &ts)
    val = ts.after(call)[0].load() * 1_000_000_000 + ts.after(call)[1].load()
  return UOp.param(0, dtypes.uint64, 1)[0].store(val)

# *****************
# 2. queue encoders

def cpu_cmd(devs:tuple[str, ...], prog, *args:UOp) -> UOp:
  progs = [get_runtime(d, prog) if isinstance(prog, UOp) else cast(CPUDevice, Device[d]).prgs[prog] for d in devs]
  addrs = tuple(UOp.const(p.addr, dtypes.uint64) for p in progs)
  words = ((addrs[0] if len(addrs) == 1 else UOp(Ops.STACK, src=addrs)),) + args
  return UOp(Ops.LINEAR, src=words + (UOp.const(0, dtypes.uint64),) * (CMD_SIZE - len(words)))

def cpu_exec(ctx, call:UOp, prg:UOp) -> UOp:
  devs = ctx.devs
  args = [get_call_arg_uops(call)[i].getaddr(devs) for i in prg.arg.globals] + [v.cast(dtypes.uint64) for v in get_call_var_uops(call, prg)]
  return cpu_cmd(devs, prg, *args)

pm_cpu_opsel = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, name="prg"),), name="call", allow_any_len=True), cpu_exec),

  (UPat(Ops.INS, arg=("barrier", dtypes.void)), lambda: UOp(Ops.LINEAR)),
  (UPat(Ops.INS, arg=("wait", dtypes.void), src=(UPat(name="dst"), UPat(name="val"))),
   lambda ctx, dst, val: cpu_cmd(ctx.devs, wait_prog, dst.getaddr(ctx.devs), val.cast(dtypes.uint64))),
  (UPat(Ops.INS, arg=("store", dtypes.void), src=(UPat((Ops.BUFFER, Ops.PARAM), name="dst"), UPat(name="val"))),
   lambda ctx, dst, val: cpu_cmd(ctx.devs, signal_prog, dst.getaddr(ctx.devs), val.cast(dtypes.uint64))),
  (UPat(Ops.INS, arg=("timestamp", dtypes.void), src=(UPat(name="dst"),)), lambda ctx, dst: cpu_cmd(ctx.devs, timestamp_prog, dst.getaddr(ctx.devs),
   *(() if WIN else (make_buf(ctx.devs, tag="func:clock_gettime").getaddr(ctx.devs),)))),
])

def cpu_submit(ctx, cmdbuf:UOp) -> UOp:
  # run the cmd entries inline on the submitting thread, the cpu has no worker threads
  cb, cnt = cmdbuf.bitcast(dtypes.uint64), hcq_size_var(cmdbuf) // (CMD_SIZE * 8)
  e = UOp.range(cnt, 10, dtype=dtypes.int, src=(cmdbuf,))
  entry = [cb.index(e*CMD_SIZE + i).load() for i in range(CMD_SIZE)]
  return entry[0].call(*entry[1:], ret_dtype=dtypes.void).end(e)

pm_cpu_submit = PatternMatcher([(UPat(Ops.CUSTOM_FUNCTION, arg="submit_cmdbuf", src=(UPat(name="cmdbuf"),)), cpu_submit)])

# *****************

# NOTE: MAP_JIT is added to mmap module in python 3.13
MAP_JIT = 0x0800

class CPUProgram(Program['CPUDevice']):
  rt_lib, libm = DLL('rt', 'System' if OSX else 'kernel' if WIN else 'gcc_s'), DLL('m', 'm')

  def _load(self, lib, base=0): return lib if lib[:4] != libc.ELFMAG.encode() else jit_loader(lib, base=base, link_libs=[self.libm, self.rt_lib])

  def __init__(self, dev:CPUDevice, obj:TinyELF):
    self.dev, self.name, self.signature = dev, obj.name, obj.signature
    self.lvp = obj.target.renderer == "LVP"

    if sys.platform == "win32": # mypy doesn't understand when WIN is used here
      PAGE_EXECUTE_READWRITE, MEM_COMMIT, MEM_RESERVE = 0x40, 0x1000, 0x2000
      ctypes.windll.kernel32.VirtualAlloc.restype = ctypes.c_void_p
      self.addr = ctypes.windll.kernel32.VirtualAlloc(ctypes.c_void_p(0), ctypes.c_size_t(len(obj.lib)), MEM_COMMIT | MEM_RESERVE,
                                                      PAGE_EXECUTE_READWRITE)
      ctypes.memmove(self.addr, (loaded:=self._load(obj.lib, self.addr)), len(loaded))
      ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
      proc = ctypes.windll.kernel32.GetCurrentProcess()
      ctypes.windll.kernel32.FlushInstructionCache(ctypes.c_void_p(proc), ctypes.c_void_p(self.addr), ctypes.c_size_t(len(loaded)))
      self.fxn = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(self.addr) if self.lvp else ctypes.CFUNCTYPE(None)(self.addr)
    else:
      # On apple silicon with SPRR enabled (it always is in macos) RWX pages are unrepresentable: https://blog.svenpeter.dev/posts/m1_sprr_gxf/
      # MAP_JIT allows us to easily flip pages from RW- to R-X and vice versa. It is a noop on intel cpus. (man pthread_jit_write_protect_np)
      self.mem = mmap.mmap(-1, len(obj.lib), mmap.MAP_ANON|mmap.MAP_PRIVATE|(MAP_JIT if OSX else 0), mmap.PROT_READ|mmap.PROT_WRITE|mmap.PROT_EXEC)
      self.addr = mv_address(self.mem)

      if OSX: unwrap(CPUProgram.rt_lib).pthread_jit_write_protect_np(False)
      self.mem.write(loaded:=self._load(obj.lib, mv_address(self.mem)))
      if OSX: unwrap(CPUProgram.rt_lib).pthread_jit_write_protect_np(True)

      # __clear_cache isn't a normal libc function, but a compiler support routine found in libgcc_s for gcc and compiler-rt for clang.
      # libgcc_s comes as shared library but compiler-rt is only a bunch of static library archives which we can't directly load, but fortunately
      # it somehow found its way into libSystem on macos (likely because it used __builtin_clear_cache) and libgcc_s is ~always present on linux
      # Using ["name"] instead of .name because otherwise name is getting mangled: https://docs.python.org/3.12/reference/expressions.html#index-5
      if 'rt' in DLL._loaded_: CPUProgram.rt_lib["__clear_cache"](ctypes.c_void_p(self.addr), ctypes.c_void_p(self.addr + len(loaded)))
      else:
        # msync should be a universal POSIX way to do this
        libc.msync(ctypes.c_void_p(self.addr), len(loaded), libc.MS_SYNC | libc.MS_INVALIDATE)

      self.fxn = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(self.addr) if self.lvp else ctypes.CFUNCTYPE(None)(self.addr)

  def __call__(self, *bufs:HCQBuffer, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1),
               vals:tuple[int|None, ...]=(), wait:bool=False, timeout:int|None=None) -> float|None:
    st = time.perf_counter()
    if self.lvp:
      lvp_args = bytearray(12 + (len(bufs) + len(vals)) * 8)
      addr = mv_address(lvp_args)
      struct.pack_into(f'<3I{len(bufs)}Q', lvp_args, 0, *data64_le(addr+12), (len(bufs)+len(vals))*2, *[b.va_addr for b in bufs])
      for v,(off,dt) in zip(vals, TinyELF.iter_sig(self.signature[-len(vals):], len(bufs)*8)): struct.pack_into(f'<{dt.fmt}', lvp_args, 12+off, v)
      self.fxn(addr)
    else:
      args = [*[cast(int, b.va_addr) for b in bufs], *cast(tuple[int, ...], vals)]
      assert len(args) <= MAX_ARGS, f"CPU programs support at most {MAX_ARGS} arguments, got {len(args)}"
      self.fxn(*[ctypes.c_uint64(x) for x in args])
    return time.perf_counter() - st if wait else None

  @suppress_finalizing
  def __del__(self):
    if sys.platform == 'win32': ctypes.windll.kernel32.VirtualFree(ctypes.c_void_p(self.addr), ctypes.c_size_t(0), 0x8000) #0x8000 - MEM_RELEASE

class CPUAllocator(HCQAllocator['CPUDevice']):
  def __init__(self, dev:CPUDevice): super().__init__(dev, supports_copy_from_disk=False, supports_transfer=False)
  def _alloc(self, size:int, options:BufferSpec) -> HCQBuffer:
    if options.external_ptr is not None: addr, buf = options.external_ptr, None
    elif WIN: addr = mv_address(buf:=mmap.mmap(-1, size, access=mmap.ACCESS_WRITE))
    else: addr = mv_address(buf:=mmap.mmap(-1, size, mmap.MAP_ANON | mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE))
    return HCQBuffer(va:=addr, sz:=size, meta=buf, view=MMIOInterface(va, sz, fmt='B'), owner=self.dev)
  def _as_buffer(self, src) -> memoryview: return to_mv(src.va_addr, src.size)
  def _copyin(self, dest:HCQBuffer, src:memoryview):
    self.dev.synchronize()
    ctypes.memmove(int(dest.va_addr), from_mv(src), len(src))
  def _copyout(self, dest:memoryview, src:HCQBuffer):
    self.dev.synchronize()
    ctypes.memmove(from_mv(dest), int(src.va_addr), len(dest))
  def _do_map(self, buf:HCQBuffer):
    if buf.view is None or not isinstance(buf.view, MMIOInterface): raise RuntimeError("Cannot map buffer without view to cpu")
    return HCQBuffer(buf.view.addr, buf.size, view=buf.view, owner=buf.owner)
  def _unmap(self, mb): pass  # CPU _do_map returns a view wrapper, nothing to release

class CPUDevice(HCQ2Compiled):
  wait_timeout_ms, has_copy_queue = 30000, False
  pm_encode, pm_lower = {"COMPUTE": pm_cpu_opsel, "SUBMIT": pm_cpu_opsel}, {"COMPUTE": pm_cpu_submit, "SUBMIT": pm_cpu_submit}

  def __init__(self, device:str=""):
    super().__init__(device, CPUAllocator(self), [ClangRenderer, CPULLVMRenderer, LVPRenderer, X86Renderer], CPUProgram,
      arch={'amd64':'x86_64', 'aarch64':'arm64'}.get(m:=platform.machine().lower(), m)+",native")

    self.pm_bufferize = PatternMatcher(
      [(UPat(Ops.PARAM, tag=f"func:{f}"), lambda ctx, f=f: ctx[0].func_ptr(f)) for f in FUNCS]) + self.pm_bufferize

    with Context(EMULATED_DTYPES="", TRACK_MATCH_STATS=0):
      clang = ClangRenderer(replace(self.renderer.target, renderer="CLANG"))
      self.prgs:dict[Callable, CPUProgram] = {f: CPUProgram(self, do_to_program(f().sink(arg=KernelInfo(f.__name__), tag=1), clang).to_elf())
                                              for f in (signal_prog, wait_prog, timestamp_prog)}

  def func_ptr(self, name:str) -> Buffer: return self.func_table.view(1, dtypes.uint64, FUNCS.index(name)*8).ensure_allocated()

  @functools.cached_property
  def func_table(self) -> Buffer:
    lib = ctypes.windll.kernel32 if sys.platform == "win32" else libc.dll # type: ignore[attr-defined]
    (ft:=Buffer(self.device, len(FUNCS), dtypes.uint64, preallocate=True))._buf.cpu_view().view(fmt='Q')[:] = \
      array.array('Q', [unwrap(ctypes.cast(getattr(lib, f), ctypes.c_void_p).value) for f in FUNCS])
    return ft
