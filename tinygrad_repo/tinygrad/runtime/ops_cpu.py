from __future__ import annotations
import platform, sys, os, ctypes, functools, mmap, threading, array, itertools
from dataclasses import replace
from typing import cast
from tinygrad.helpers import to_mv, OSX, WIN, Context, mv_address, suppress_finalizing, unwrap, data64_le, partition
from tinygrad.device import Buffer, BufferSpec, TinyELF
from tinygrad.runtime.support.hcq import HCQCompiled, HCQAllocator, HCQBuffer, HWQueue, HCQArgsState, HCQSignal, HCQProgram, MMIOInterface
from tinygrad.runtime.support.hcq import CLikeArgsState
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.renderer.llvmir import CPULLVMRenderer
from tinygrad.renderer.nir import LVPRenderer
from tinygrad.renderer.isa.x86 import X86Renderer
from tinygrad.runtime.support.elf import jit_loader
from tinygrad.runtime.autogen import libc
from tinygrad.codegen import do_to_program
from tinygrad import UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import sint, KernelInfo, Ops, UPat, PatternMatcher, graph_rewrite

MAX_ARGS, CMD_SIZE, RING_SLOTS = 63, 64, (16 << 10)

def signal_prog():
  val = UOp.param(1, dtypes.int, (), vmin_vmax=(0, dtypes.int.max), name="value", addrspace=AddrSpace.ALU)
  return UOp.param(0, dtypes.uint32, (1,))[0].store(val.cast(dtypes.uint32))

def wait_prog():
  val = UOp.param(1, dtypes.int, (), vmin_vmax=(0, dtypes.int.max), name="value", addrspace=AddrSpace.ALU)
  return (v:=UOp.param(0, dtypes.uint32, (1,), volatile=True).after(l:=UOp.loop(0))[0].load()).end(l, v < val.cast(dtypes.uint32))

def timestamp_prog():
  if WIN: val = UOp.const(0, dtypes.uint64)
  else:
    fn, ts = UOp.param(1, dtypes.uint64, (1,)), UOp.placeholder((2,), dtypes.uint64, slot=0, addrspace=AddrSpace.REG)
    call = fn[0].load().call(UOp.const(6 if OSX else 1, dtypes.int), ts[0], ret_dtype=dtypes.void) # clock_gettime(CLOCK_MONOTONIC, &ts)
    val = ts.after(call)[0].load() * 1_000_000_000 + ts.after(call)[1].load()
  return UOp.param(0, dtypes.uint64, (1,))[0].store(val)

def quit_prog():
  fn = UOp.param(0, dtypes.uint64, (1 if WIN else 3,))
  if WIN: return fn[0].load().call(UOp.const(0, dtypes.uint64), ret_dtype=dtypes.void) # ExitThread(0)
  sem = UOp.param(1, dtypes.uint64, (1,))

  close = fn[2].load().call(sem[0], ret_dtype=dtypes.void) # sem_close(sem)
  return fn.after(close)[0].load().call(UOp.const(0, dtypes.uint64), ret_dtype=dtypes.void) # pthread_exit(0)

def worker_prog():
  ring = UOp.param(0, dtypes.uint64, (RING_SLOTS * CMD_SIZE,), volatile=True)
  wait, sem = UOp.param(1, dtypes.uint64, (1,), volatile=True), UOp.param(2, dtypes.uint64, (1,))
  cur = UOp.range(2**64-1, 0, dtype=dtypes.uint64)

  # spin on windows, sem_wait to sleep on posix
  if WIN: ready = (v:=wait.after(lw:=UOp.loop(1), cur)[0].load()).end(lw, v <= cur)
  else: ready = (rv:=wait.after(lw:=UOp.loop(1), cur)[0].load().call(sem.after(cur)[0], ret_dtype=dtypes.int)).end(lw, rv != 0)

  entry = [ring.after(ready).index((cur % RING_SLOTS) * CMD_SIZE + i).load() for i in range(CMD_SIZE)]
  return entry[0].call(*entry[1:], ret_dtype=dtypes.void).end(cur)

def host_wait(ctx, dst:UOp, val:UOp) -> UOp:
  return (cur:=dst.after(loop:=UOp.loop(next(ctx))).index(UOp.const(0, dtypes.int)).load()).end(loop, cur < val)

pm_host_opsel = PatternMatcher([(UPat(Ops.INS, arg="wait", src=(UPat(name="dst"), UPat(name="val"))), host_wait)])

def encode_host_queue(q:UOp) -> UOp:
  # TODO: subset of hcq2 for now
  spins, (store,) = partition(graph_rewrite(q, pm_host_opsel, ctx=itertools.count(), walk=True, name="host opsel").src, lambda u: u.op is Ops.END)
  assert store.op is Ops.INS and store.arg == "store", f"host queue cannot encode {store.op} {store.arg}"
  return store.src[0].after(*spins).index(UOp.const(0, dtypes.int)).store(store.src[1])

class CPUComputeQueue(HWQueue):
  def __init__(self, dev):
    super().__init__()
    self.dev = dev
  def _cmd(self, prog, args=(), vals=()): return self.exec(prg:=self.dev.prgs[prog], prg.fill_kernargs(args, vals), None, None)
  def memory_barrier(self): return self
  def exec(self, prg:CPUProgram, args_state:HCQArgsState, global_size, local_size):
    if (lvp:=isinstance(args_state, LVPArgsState)): self.bind_args_state(args_state)
    args:list[sint|None] = [args_state.buf.va_addr] if lvp else [*[x.va_addr for x in args_state.bufs], *args_state.vals]
    assert len(args) <= MAX_ARGS, f"CPU programs support at most {MAX_ARGS} arguments, got {len(args)}"
    for tid in range(1 if lvp else (global_size or (1,))[0]):
      if not lvp and 'core_id' in prg.runtimevars: args[prg.runtimevars['core_id']] = tid
      self.q(prg, *[unwrap(x) for x in args], *([0] * (MAX_ARGS - len(args))))
    return self
  def wait(self, signal, value=0): return self._cmd(wait_prog, (signal.base_buf,), (value,))
  def timestamp(self, signal): return self._cmd(timestamp_prog, (signal.base_buf.offset(8, 8), self.dev.func_table._buf.offset(0, 8)))
  def signal(self, signal, value:sint=0): return self._cmd(signal_prog, (signal.base_buf,), (value,))
  def _submit(self, dev):
    dev.ensure_worker()
    ring_view = dev.ring.as_memoryview(force_zero_copy=True, no_sync=True).cast('Q')
    for off in range(0, len(self._q), CMD_SIZE):
      entry = [self._q[off].addr, *self._q[off+1:off+CMD_SIZE]]
      ring_view[(base:=(dev.ring_pos % RING_SLOTS) * CMD_SIZE):base+CMD_SIZE] = array.array('Q', (int(x) & ((1<<64)-1) for x in entry))
      dev.ring_pos += 1
      if WIN: dev.sys.as_memoryview(force_zero_copy=True, no_sync=True).cast('Q')[0] = dev.ring_pos
      else: assert libc.sem_post(dev.sem) == 0

class LVPArgsState(CLikeArgsState):
  def __init__(self, buf, prg, bufs, vals=()): super().__init__(buf, prg, bufs, vals, [*data64_le(buf.va_addr + 12), (len(bufs) + len(vals)) * 2])

# NOTE: MAP_JIT is added to mmap module in python 3.13
MAP_JIT = 0x0800

class CPUProgram(HCQProgram['CPUDevice']):
  rt_lib = None
  try: rt_lib = ctypes.CDLL(ctypes.util.find_library('System' if OSX else 'kernel32') if OSX or WIN else 'libgcc_s.so.1')
  except OSError: pass

  def __init__(self, dev:CPUDevice, obj:TinyELF):
    self.signature, self.runtimevars = obj.signature, {name:slot for name,slot,*_ in obj.signature if name == 'core_id'}

    LVP = obj.target.renderer == "LVP"
    if sys.platform == "win32": # mypy doesn't understand when WIN is used here
      PAGE_EXECUTE_READWRITE, MEM_COMMIT, MEM_RESERVE = 0x40, 0x1000, 0x2000
      ctypes.windll.kernel32.VirtualAlloc.restype = ctypes.c_void_p
      self.addr = ctypes.windll.kernel32.VirtualAlloc(ctypes.c_void_p(0), ctypes.c_size_t(len(obj.lib)), MEM_COMMIT | MEM_RESERVE,
                                                      PAGE_EXECUTE_READWRITE)
      ctypes.memmove(self.addr, obj.lib, len(obj.lib))
      ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
      proc = ctypes.windll.kernel32.GetCurrentProcess()
      ctypes.windll.kernel32.FlushInstructionCache(ctypes.c_void_p(proc), ctypes.c_void_p(self.addr), ctypes.c_size_t(len(obj.lib)))
      self.fxn = ctypes.CFUNCTYPE(None)(self.addr)
    else:
      # On apple silicon with SPRR enabled (it always is in macos) RWX pages are unrepresentable: https://blog.svenpeter.dev/posts/m1_sprr_gxf/
      # MAP_JIT allows us to easily flip pages from RW- to R-X and vice versa. It is a noop on intel cpus. (man pthread_jit_write_protect_np)
      self.mem = mmap.mmap(-1, len(obj.lib), mmap.MAP_ANON|mmap.MAP_PRIVATE|(MAP_JIT if OSX else 0), mmap.PROT_READ|mmap.PROT_WRITE|mmap.PROT_EXEC)
      self.addr = mv_address(self.mem)

      if OSX: unwrap(CPUProgram.rt_lib).pthread_jit_write_protect_np(False)
      lib = jit_loader(obj.lib, base=ctypes.addressof(ctypes.c_void_p.from_buffer(self.mem)), link_libs=['m']) if LVP else obj.lib
      self.mem.write(lib)
      if OSX: unwrap(CPUProgram.rt_lib).pthread_jit_write_protect_np(True)

      # __clear_cache isn't a normal libc function, but a compiler support routine found in libgcc_s for gcc and compiler-rt for clang.
      # libgcc_s comes as shared library but compiler-rt is only a bunch of static library archives which we can't directly load, but fortunately
      # it somehow found its way into libSystem on macos (likely because it used __builtin_clear_cache) and libgcc_s is ~always present on linux
      # Using ["name"] instead of .name because otherwise name is getting mangled: https://docs.python.org/3.12/reference/expressions.html#index-5
      if CPUProgram.rt_lib is not None: CPUProgram.rt_lib["__clear_cache"](ctypes.c_void_p(self.addr), ctypes.c_void_p(self.addr + len(lib)))
      else:
        # msync should be a universal POSIX way to do this
        libc.msync(ctypes.c_void_p(self.addr), len(lib), libc.MS_SYNC | libc.MS_INVALIDATE)

      self.fxn = ctypes.CFUNCTYPE(None)(self.addr)

    super().__init__(LVPArgsState if LVP else HCQArgsState, dev, obj, kernargs_alloc_size=12+256 if LVP else 0)

  @suppress_finalizing
  def __del__(self):
    if sys.platform == 'win32': ctypes.windll.kernel32.VirtualFree(ctypes.c_void_p(self.addr), ctypes.c_size_t(0), 0x8000) #0x8000 - MEM_RELEASE

class CPUAllocator(HCQAllocator):
  def __init__(self, dev:CPUDevice): super().__init__(dev, supports_copy_from_disk=False, supports_transfer=False)
  def _alloc(self, size:int, options:BufferSpec) -> HCQBuffer:
    if options.external_ptr is not None: addr, buf = options.external_ptr, None
    elif WIN: addr = mv_address(buf:=mmap.mmap(-1, size, access=mmap.ACCESS_WRITE))
    else: addr = mv_address(buf:=mmap.mmap(-1, size, mmap.MAP_ANON | mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE))
    return HCQBuffer(va:=addr, sz:=size, meta=buf, view=MMIOInterface(va, sz, fmt='B'), owner=self.dev)
  def _as_buffer(self, src) -> memoryview: return to_mv(src.va_addr, src.size)
  def _do_map(self, buf:HCQBuffer):
    if buf.view is None or not isinstance(buf.view, MMIOInterface): raise RuntimeError("Cannot map buffer without view to cpu")
    return HCQBuffer(buf.view.addr, buf.size, view=buf.view, owner=buf.owner)
  def _unmap(self, mb): pass  # CPU _do_map returns a view wrapper, nothing to release

class CPUDevice(HCQCompiled):
  pm_lower = PatternMatcher([
    (UPat(Ops.CUSTOM_FUNCTION, arg="submit_cmdbuf", src=(UPat(Ops.LINEAR, name="q"),)), encode_host_queue)])

  pm_bufferize = PatternMatcher([
    (UPat(Ops.PARAM, tag="sentinel_signal"), lambda ctx: ctx[0].signal("sentinel", (1 << 64) - 1)),
    (UPat(Ops.PARAM, tag="timeline_signal"), lambda ctx: ctx[0].signal("timeline")),
    (UPat(Ops.PARAM, tag="timeline_value"), lambda ctx: ctx[0].signal("value", 1)),
    (UPat(Ops.PARAM, tag="signal", name="b"), lambda ctx, b: ctx[0].signal(b.arg.slot)),
  ])

  @functools.cache
  def signal(self, name:str|int, init_value:int=0) -> Buffer:
    (buf:=Buffer(self.device, 1, dtypes.uint64, preallocate=True)).as_memoryview(force_zero_copy=True, no_sync=True).cast('Q')[0] = init_value
    return buf

  def __init__(self, device:str=""):
    super().__init__(device, CPUAllocator(self), [ClangRenderer, CPULLVMRenderer, LVPRenderer, X86Renderer], CPUProgram, HCQSignal,
      functools.partial(CPUComputeQueue, self), arch={'amd64':'x86_64', 'aarch64':'arm64'}.get(m:=platform.machine().lower(), m)+",native")

    self.ring_pos = 0

    # posix uses sem to put cpus into sleep
    self.sem_addr = 0
    if not WIN:
      self.sem = libc.sem_open(sem_name:=f"/tinygrad-{os.getpid()}-{id(self):x}".encode(), os.O_CREAT|os.O_EXCL, 0o600, 0) # type: ignore[call-arg]
      self.sem_addr = unwrap(ctypes.cast(self.sem, ctypes.c_void_p).value)
      if self.sem_addr == ctypes.c_void_p(-1).value or libc.sem_unlink(sem_name): raise OSError(ctypes.get_errno(), "semaphore")

    # TODO: move to hcq2
    with Context(EMULATED_DTYPES="", TRACK_MATCH_STATS=0):
      prgs = {f: f().sink(arg=KernelInfo(f.__name__), tag=1) for f in (signal_prog, wait_prog, timestamp_prog, quit_prog, worker_prog)}
      self.prgs = {f: self.runtime(do_to_program(v, ClangRenderer(replace(self.renderer.target, renderer="CLANG"))).to_elf()) for f,v in prgs.items()}

  @functools.cached_property
  def ring(self) -> Buffer: return Buffer(self.device, RING_SLOTS * CMD_SIZE, dtypes.uint64, preallocate=True)
  @functools.cached_property
  def sys(self) -> Buffer: return Buffer(self.device, 1, dtypes.uint64, preallocate=True)
  @functools.cached_property
  def sem_buf(self) -> Buffer: return Buffer(self.device, 1, dtypes.uint8, options=BufferSpec(external_ptr=self.sem_addr), preallocate=True)

  # TODO: move to hcq2 infra
  @functools.cached_property
  def func_table(self) -> Buffer:
    fns = ([0, ctypes.windll.kernel32.ExitThread, 0, 0] if WIN else  # type: ignore[attr-defined]
           [libc.dll.clock_gettime, libc.dll.pthread_exit, libc.dll.sem_wait, libc.dll.sem_close])
    addrs = array.array('Q', [unwrap(ctypes.cast(f, ctypes.c_void_p).value) if f else 0 for f in fns])
    (ft:=Buffer(self.device, len(fns), dtypes.uint64, preallocate=True)).as_memoryview(force_zero_copy=True, no_sync=True).cast('Q')[:] = addrs
    return ft

  @functools.cache
  def ensure_worker(self):
    threading.Thread(target=cast(CPUProgram, self.prgs[worker_prog]).fxn, daemon=True, args=[ctypes.c_uint64(x) for x in
      [self.ring._buf.va_addr, self.sys._buf.va_addr if WIN else self.func_table._buf.va_addr+16, self.sem_addr]]).start()

  def finalize(self):
    if self.ring_pos == 0: return # the worker starts with the first submit
    ft = self.func_table._buf
    CPUComputeQueue(self)._cmd(quit_prog, (ft.offset(8, 8),) if WIN else (ft.offset(8, 24), self.sem_buf._buf)).submit(self)
    self.ring_pos = 0
