from __future__ import annotations
import platform, sys, ctypes, functools, time, mmap, threading, queue
from tinygrad.helpers import to_mv, OSX, WIN, mv_address, wait_cond, suppress_finalizing, unwrap, data64_le
from tinygrad.helpers import CPU_CC, CPU_LVP, CPU_LLVM
from tinygrad.device import BufferSpec, DMACPURef, CompilerSet, CompilerPair
from tinygrad.runtime.support.hcq import HCQCompiled, HCQAllocator, HCQBuffer, HWQueue, HCQArgsState, HCQSignal, HCQProgram, MMIOInterface
from tinygrad.runtime.support.hcq import CLikeArgsState
from tinygrad.renderer.cstyle import ClangJITRenderer
from tinygrad.renderer.llvmir import CPULLVMRenderer
from tinygrad.renderer.nir import LVPRenderer
from tinygrad.runtime.support.compiler_cpu import CPULLVMCompiler
from tinygrad.runtime.support.elf import jit_loader
from tinygrad.uop.ops import sint

class CPUSignal(HCQSignal):
  def _sleep(self, time_spent_waiting_ms:int) -> bool:
    if self.is_timeline and self.owner is not None: self.owner.tasks.join()
    return False

class CPUWorker(threading.Thread):
  def __init__(self, dev, tasks, thread_id):
    super().__init__()
    self.dev, self.tasks, self.thread_id, self.pool, self.daemon = dev, tasks, thread_id, [], True

  def push_task(self, tid, cmd, args):
    if len(self.pool) <= tid:
      self.pool.append(queue.Queue())
      CPUWorker(self, self.pool[tid], thread_id=tid+1).start()
    self.pool[tid].put([cmd, 1, len(args)] + args)

  def run(self):
    while True:
      cmd_iter = iter(self.tasks.get())
      for cmd in cmd_iter:
        threads, args_cnt = next(cmd_iter), next(cmd_iter)
        args = [next(cmd_iter) for _ in range(args_cnt)]
        for th in range(threads - 1): self.push_task(th, cmd, args)
        cmd(self.thread_id, *args)
        for th in range(threads - 1): self.pool[th].join()
      self.tasks.task_done()

class CPUComputeQueue(HWQueue):
  def _exec(self, tid, prg, bufs, *args):
    vals = list(args[bufs:])
    if 'core_id' in prg.runtimevars: vals[prg.runtimevars['core_id']] = tid
    prg.fxn(*map(ctypes.c_uint64, args[:bufs]), *map(ctypes.c_int64 if platform.machine() == "arm64" else ctypes.c_int32, vals))
  def _signal(self, tid, signal_addr, value): to_mv(signal_addr, 4).cast('I')[0] = value
  def _wait(self, tid, signal_addr, value): wait_cond(lambda: to_mv(signal_addr, 4).cast('I')[0] >= value, timeout_ms=60000)
  def _timestamp(self, tid, timestamp_addr): to_mv(timestamp_addr, 8).cast('Q')[0] = time.perf_counter_ns()
  def cmd(self, cmd, *args, threads=1):
    self.q(cmd, threads, len(args), *args)
    return self

  def memory_barrier(self): return self
  def exec(self, prg:CPUProgram, args_state:HCQArgsState, global_size, local_size):
    if isinstance(args_state, LVPArgsState):
      self.bind_args_state(args_state)
      return self.cmd(self._exec, prg, 1, args_state.buf.va_addr)
    return self.cmd(self._exec, prg, len(args_state.bufs), *[x.va_addr for x in args_state.bufs], *args_state.vals, threads=(global_size or (1,))[0])
  def wait(self, signal, value=0): return self.cmd(self._wait, signal.value_addr, value)
  def timestamp(self, signal): return self.cmd(self._timestamp, signal.timestamp_addr)
  def signal(self, signal, value:sint=0): return self.cmd(self._signal, signal.value_addr, value)
  def _submit(self, dev): dev.tasks.put(self._q[:])

class LVPArgsState(CLikeArgsState):
  def __init__(self, buf, prg, bufs, vals=()): super().__init__(buf, prg, bufs, vals, [*data64_le(buf.va_addr + 12), (len(bufs) + len(vals)) * 2])

# NOTE: MAP_JIT is added to mmap module in python 3.13
MAP_JIT = 0x0800

class CPUProgram(HCQProgram):
  rt_lib = None
  try: rt_lib = ctypes.CDLL(ctypes.util.find_library('System' if OSX else 'kernel32') if OSX or WIN else 'libgcc_s.so.1')
  except OSError: pass

  def __init__(self, dev, name:str, lib:bytes, runtimevars:dict[str, int]|None=None, **kwargs):
    self.runtimevars = runtimevars or {}

    LVP = isinstance(dev.renderer, LVPRenderer)
    if sys.platform == "win32": # mypy doesn't understand when WIN is used here
      PAGE_EXECUTE_READWRITE, MEM_COMMIT, MEM_RESERVE = 0x40, 0x1000, 0x2000
      ctypes.windll.kernel32.VirtualAlloc.restype = ctypes.c_void_p
      self.mem = ctypes.windll.kernel32.VirtualAlloc(ctypes.c_void_p(0), ctypes.c_size_t(len(lib)), MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE)
      ctypes.memmove(self.mem, lib, len(lib))
      ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
      proc = ctypes.windll.kernel32.GetCurrentProcess()
      ctypes.windll.kernel32.FlushInstructionCache(ctypes.c_void_p(proc), ctypes.c_void_p(self.mem), ctypes.c_size_t(len(lib)))
      self.fxn = ctypes.CFUNCTYPE(None)(self.mem)
    else:
      # On apple silicon with SPRR enabled (it always is in macos) RWX pages are unrepresentable: https://blog.svenpeter.dev/posts/m1_sprr_gxf/
      # MAP_JIT allows us to easily flip pages from RW- to R-X and vice versa. It is a noop on intel cpus. (man pthread_jit_write_protect_np)
      self.mem = mmap.mmap(-1, len(lib), mmap.MAP_ANON|mmap.MAP_PRIVATE|(MAP_JIT if OSX else 0), mmap.PROT_READ|mmap.PROT_WRITE|mmap.PROT_EXEC)

      if OSX: unwrap(CPUProgram.rt_lib).pthread_jit_write_protect_np(False)
      if LVP: lib = jit_loader(lib, base=ctypes.addressof(ctypes.c_void_p.from_buffer(self.mem)), link_libs=['m'])
      self.mem.write(lib)
      if OSX: unwrap(CPUProgram.rt_lib).pthread_jit_write_protect_np(True)

      # __clear_cache isn't a normal libc function, but a compiler support routine found in libgcc_s for gcc and compiler-rt for clang.
      # libgcc_s comes as shared library but compiler-rt is only a bunch of static library archives which we can't directly load, but fortunately
      # it somehow found its way into libSystem on macos (likely because it used __builtin_clear_cache) and libgcc_s is ~always present on linux
      # Using ["name"] instead of .name because otherwise name is getting mangled: https://docs.python.org/3.12/reference/expressions.html#index-5
      if CPUProgram.rt_lib is not None:
        CPUProgram.rt_lib["__clear_cache"](ctypes.c_void_p(mv_address(self.mem)), ctypes.c_void_p(mv_address(self.mem) + len(lib)))
      else:
        # msync should be a universal POSIX way to do this
        from tinygrad.runtime.autogen import libc
        libc.msync(ctypes.c_void_p(mv_address(self.mem)), len(lib), libc.MS_SYNC | libc.MS_INVALIDATE)

      self.fxn = ctypes.CFUNCTYPE(None)(mv_address(self.mem))

    super().__init__(LVPArgsState if LVP else HCQArgsState, dev, name, kernargs_alloc_size=12+256 if LVP else 0)

  @suppress_finalizing
  def __del__(self):
    if sys.platform == 'win32': ctypes.windll.kernel32.VirtualFree(ctypes.c_void_p(self.mem), ctypes.c_size_t(0), 0x8000) #0x8000 - MEM_RELEASE

class CPUAllocator(HCQAllocator):
  def __init__(self, dev:CPUDevice): super().__init__(dev, supports_copy_from_disk=False, supports_transfer=False)
  def _alloc(self, size:int, options:BufferSpec) -> HCQBuffer:
    if options.external_ptr is not None: addr, buf = options.external_ptr, None
    elif WIN: addr = mv_address(buf:=mmap.mmap(-1, size, access=mmap.ACCESS_WRITE))
    else: addr = mv_address(buf:=mmap.mmap(-1, size, mmap.MAP_ANON | mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE))
    return HCQBuffer(va:=addr, sz:=size, meta=buf, view=MMIOInterface(va, sz, fmt='B'), owner=self.dev)
  def _as_buffer(self, src) -> memoryview:
    self.dev.synchronize()
    return to_mv(src.va_addr, src.size)
  def _as_dmaref(self, buf):
    self.dev.synchronize()
    return DMACPURef(buf.va_addr, buf.size)
  def _map(self, buf:HCQBuffer):
    if buf.view is None or not isinstance(buf.view, MMIOInterface): raise RuntimeError("Cannot map buffer without view to cpu")

class CPUDevice(HCQCompiled):
  def __init__(self, device:str=""):
    self.tasks:queue.Queue = queue.Queue()
    CPUWorker(self, self.tasks, thread_id=0).start()
    compilers = CompilerSet([CompilerPair(ClangJITRenderer, None), CompilerPair(CPULLVMRenderer, CPULLVMCompiler, ctrl_var=CPU_LLVM),
                             CompilerPair(LVPRenderer, None, ctrl_var=CPU_LVP)], ctrl_var=CPU_CC)
    super().__init__(device, CPUAllocator(self), compilers, functools.partial(CPUProgram, self), CPUSignal, CPUComputeQueue)
