from __future__ import annotations
import platform, sys, os, ctypes, functools, mmap, threading, array, struct, time
from dataclasses import dataclass, replace
from typing import cast, Callable
from tinygrad.helpers import to_mv, from_mv, OSX, WIN, Context, mv_address, suppress_finalizing, unwrap, data64_le, to_tuple
from tinygrad.device import Buffer, BufferSpec, TinyELF, Program, Device
from tinygrad.runtime.support.hcq import HCQBuffer, MMIOInterface
from tinygrad.runtime.support.hcq2 import HCQ2Compiled, HCQAllocator, make_cmdbuf, make_signal
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.renderer.llvmir import CPULLVMRenderer
from tinygrad.renderer.nir import LVPRenderer
from tinygrad.renderer.isa.x86 import X86Renderer
from tinygrad.runtime.support.elf import jit_loader
from tinygrad.runtime.autogen import libc
from tinygrad.codegen import do_to_program
from tinygrad.engine.realize import pm_flatten_linear, get_call_arg_uops, get_runtime
from tinygrad import UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import KernelInfo, Ops, UPat, PatternMatcher, graph_rewrite

MAX_ARGS, CMD_SIZE, RING_SLOTS, FUNCS = 63, 64, (16 << 10), (() if WIN else ('clock_gettime', 'sem_wait', 'sem_post'))

# *****************
# 1. workers

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

def worker_prog():
  ring = UOp.param(0, dtypes.uint64, (RING_SLOTS * CMD_SIZE,), volatile=True)
  wait, done = UOp.param(1, dtypes.uint64, (1,), volatile=True), UOp.param(2, dtypes.uint64, (1,), volatile=True)
  sem, cur = UOp.param(3, dtypes.uint64, (1,)), UOp.range(2**64-1, 0, dtype=dtypes.uint64) # sem is unused on windows, it has to come last

  # spin on windows, sem_wait to sleep on posix
  if WIN: ready = (v:=wait.after(lw:=UOp.loop(1), cur)[0].load()).end(lw, v <= cur)
  else: ready = (rv:=wait.after(lw:=UOp.loop(1), cur)[0].load().call(sem.after(cur)[0], ret_dtype=dtypes.int)).end(lw, rv != 0)

  entry = [ring.after(ready).index((cur % RING_SLOTS) * CMD_SIZE + i).load() for i in range(CMD_SIZE)]
  return done.after(entry[0].call(*entry[1:], ret_dtype=dtypes.void)).index(0).store(cur + 1).end(cur)

@dataclass
class CPUWorker: ring:Buffer; put:Buffer; sem:Buffer; sys:Buffer; done:Buffer; thread:threading.Thread  # noqa: E702

# *****************
# 2. queue encoders

def cpu_cmd(devs:tuple[str, ...], prog, *args:UOp) -> UOp:
  progs = [get_runtime(d, prog) if isinstance(prog, UOp) else cast(CPUDevice, Device[d]).prgs[prog] for d in devs]
  addrs = tuple(UOp.const(p.addr, dtypes.uint64) for p in progs)
  words = ((addrs[0] if len(addrs) == 1 else UOp(Ops.STACK, dtypes.uint64, addrs)),) + args
  return UOp(Ops.INS, dtypes.void, words + (UOp.const(0, dtypes.uint64),) * (CMD_SIZE - len(words)), arg="cmd")

def cpu_exec(ctx:tuple[str, ...], call:UOp, prg:UOp) -> UOp:
  args = [get_call_arg_uops(call)[i].getaddr(ctx) for i in prg.arg.globals] + [v.cast(dtypes.uint64) for v in prg.arg.vars]
  if (core:=prg.arg.runtimevars.get('core_id')) is None: return cpu_cmd(ctx, prg, *args)

  la = [cpu_cmd(ctx,prg,*args[:(cid:=(len(prg.arg.globals)+core))],UOp.const(t, dtypes.uint64),*args[cid+1:]) for t in range(prg.arg.global_size[0])]
  return UOp(Ops.LINEAR, dtypes.void, tuple(la))

pm_cpu_opsel = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, name="prg"),), name="call", allow_any_len=True), cpu_exec),

  (UPat(Ops.INS, arg="barrier"), lambda: UOp(Ops.NOOP, dtypes.void, ())),
  (UPat(Ops.INS, arg="wait", src=(UPat(name="dst"), UPat(name="val"))),
   lambda ctx, dst, val: cpu_cmd(ctx, wait_prog, dst.getaddr(ctx), val.cast(dtypes.uint64))),
  (UPat(Ops.INS, arg="store", src=(UPat((Ops.BUFFER, Ops.PARAM), name="dst"), UPat(name="val"))),
   lambda ctx, dst, val: cpu_cmd(ctx, signal_prog, dst.getaddr(ctx), val.cast(dtypes.uint64))),
  (UPat(Ops.INS, arg="timestamp", src=(UPat(name="dst"),)),
   lambda ctx, dst: cpu_cmd(ctx, timestamp_prog, dst.getaddr(ctx), *(() if WIN else (make_signal(ctx, tag="func:clock_gettime").getaddr(ctx),)))),
])

def encode_queue(q:UOp) -> UOp:
  devs, queue = to_tuple(q.arg[0]), q.arg[1]
  lin = graph_rewrite(q, pm_cpu_opsel+pm_flatten_linear, ctx=devs, walk=True, name=f"{queue} opsel")

  cnt = sum(len(ins.src) for ins in lin.src) // CMD_SIZE
  assert cnt < RING_SLOTS, f"submit of {cnt} entries doesn't fit the ring"
  cmdbuf = make_cmdbuf(lin, devs, buf=UOp.placeholder((cnt*CMD_SIZE,), dtypes.uint64, next(UOp.unique_num), device=devs).rtag("cmdbuf"))
  ring = UOp.placeholder((ring_words:=RING_SLOTS*CMD_SIZE,), dtypes.uint64, 0, device=devs, volatile=True).rtag(f"{queue}_ring")
  put, done, sem, sysbuf = (make_signal(devs, tag=f"{queue}_{name}") for name in ("put", "done", "sem", "sys"))

  # submits are serialized on the submitter, so they can bump put without atomics
  ran = done.after(l:=UOp.loop(next(UOp.unique_num))).index(0).load()
  room = ran.end(l, put.index(0).load() - ran > RING_SLOTS - cnt) # wait until cnt entries fit in the ring
  base = ((put.after(room).index(0).load() % RING_SLOTS) * CMD_SIZE).cast(dtypes.int)
  e = UOp.range(cnt, next(UOp.unique_num), dtype=dtypes.int, src=(cmdbuf, ring))
  copy = UOp.group(*[ring.index((base + e*CMD_SIZE + w) % ring_words).store(cmdbuf.index(e*CMD_SIZE + w).load()) for w in range(CMD_SIZE)])

  # wake the worker after each entry, keeping the post with the stores stops it from hoisting out of the loop
  wake = copy.end(e) if WIN else make_signal(devs, tag="func:sem_post").after(copy).index(0).load().call(sem.index(0), ret_dtype=dtypes.void).end(e)
  bumped = put.after(wake).index(0).store(put.index(0).load() + cnt)
  return sysbuf.after(bumped).index(0).store(put.index(0).load() + cnt) if WIN else bumped

# *****************

# NOTE: MAP_JIT is added to mmap module in python 3.13
MAP_JIT = 0x0800

class CPUProgram(Program['CPUDevice']):
  rt_lib = None
  try: rt_lib = ctypes.CDLL(ctypes.util.find_library('System' if OSX else 'kernel32') if OSX or WIN else 'libgcc_s.so.1')
  except OSError: pass

  def __init__(self, dev:CPUDevice, obj:TinyELF):
    self.dev, self.name, self.signature = dev, obj.name, obj.signature
    self.runtimevars = {name:slot for name,slot,*_ in obj.signature if name == 'core_id'}
    self.lvp = obj.target.renderer == "LVP"

    if sys.platform == "win32": # mypy doesn't understand when WIN is used here
      PAGE_EXECUTE_READWRITE, MEM_COMMIT, MEM_RESERVE = 0x40, 0x1000, 0x2000
      ctypes.windll.kernel32.VirtualAlloc.restype = ctypes.c_void_p
      self.addr = ctypes.windll.kernel32.VirtualAlloc(ctypes.c_void_p(0), ctypes.c_size_t(len(obj.lib)), MEM_COMMIT | MEM_RESERVE,
                                                      PAGE_EXECUTE_READWRITE)
      ctypes.memmove(self.addr, obj.lib, len(obj.lib))
      ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
      proc = ctypes.windll.kernel32.GetCurrentProcess()
      ctypes.windll.kernel32.FlushInstructionCache(ctypes.c_void_p(proc), ctypes.c_void_p(self.addr), ctypes.c_size_t(len(obj.lib)))
      self.fxn = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(self.addr) if self.lvp else ctypes.CFUNCTYPE(None)(self.addr)
    else:
      # On apple silicon with SPRR enabled (it always is in macos) RWX pages are unrepresentable: https://blog.svenpeter.dev/posts/m1_sprr_gxf/
      # MAP_JIT allows us to easily flip pages from RW- to R-X and vice versa. It is a noop on intel cpus. (man pthread_jit_write_protect_np)
      self.mem = mmap.mmap(-1, len(obj.lib), mmap.MAP_ANON|mmap.MAP_PRIVATE|(MAP_JIT if OSX else 0), mmap.PROT_READ|mmap.PROT_WRITE|mmap.PROT_EXEC)
      self.addr = mv_address(self.mem)

      if OSX: unwrap(CPUProgram.rt_lib).pthread_jit_write_protect_np(False)
      lib = jit_loader(obj.lib, base=ctypes.addressof(ctypes.c_void_p.from_buffer(self.mem)), link_libs=['m']) if self.lvp else obj.lib
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
      for tid in range(global_size[0]):
        if 'core_id' in self.runtimevars: args[self.runtimevars['core_id']] = tid
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
  pm_lower = PatternMatcher([(UPat(Ops.CUSTOM_FUNCTION, arg="submit_cmdbuf", src=(UPat(Ops.LINEAR, name="q"),)), encode_queue)])

  def __init__(self, device:str=""):
    super().__init__(device, CPUAllocator(self), [ClangRenderer, CPULLVMRenderer, LVPRenderer, X86Renderer], CPUProgram,
      arch={'amd64':'x86_64', 'aarch64':'arm64'}.get(m:=platform.machine().lower(), m)+",native")

    self.pm_bufferize = PatternMatcher(
      [(UPat(Ops.PARAM, tag=f"COMPUTE:0_{n}"), lambda ctx, n=n: getattr(ctx[0].worker, n)) for n in ("ring", "put", "sem", "sys", "done")] +
      [(UPat(Ops.PARAM, tag=f"func:{f}"), lambda ctx, f=f: ctx[0].func_ptr(f)) for f in FUNCS]) + self.pm_bufferize

    with Context(EMULATED_DTYPES="", TRACK_MATCH_STATS=0):
      clang = ClangRenderer(replace(self.renderer.target, renderer="CLANG"))
      self.prgs:dict[Callable, CPUProgram] = {f: CPUProgram(self, do_to_program(f().sink(arg=KernelInfo(f.__name__), tag=1), clang).to_elf())
                                              for f in (signal_prog, wait_prog, timestamp_prog, worker_prog)}

  def func_ptr(self, name:str) -> Buffer: return self.func_table.view(1, dtypes.uint64, FUNCS.index(name)*8).ensure_allocated()

  @functools.cached_property
  def func_table(self) -> Buffer:
    lib = ctypes.windll.kernel32 if sys.platform == "win32" else libc.dll # type: ignore[attr-defined]
    (ft:=Buffer(self.device, len(FUNCS), dtypes.uint64, preallocate=True))._buf.cpu_view().view(fmt='Q')[:] = \
      array.array('Q', [unwrap(ctypes.cast(getattr(lib, f), ctypes.c_void_p).value) for f in FUNCS])
    return ft

  @functools.cached_property
  def worker(self) -> CPUWorker:
    ring, put, sysbuf, done = (Buffer(self.device, sz, dtypes.uint64, preallocate=True) for sz in (RING_SLOTS*CMD_SIZE, 1, 1, 1))
    addr, hsem = 0, None

    # sem are posix-only
    if not WIN:
      hsem = libc.sem_open(nm:=f"/tinygrad-{os.getpid()}-{id(ring):x}".encode(), os.O_CREAT|os.O_EXCL, 0o600, 0) # type: ignore[call-arg]
      if (addr:=unwrap(ctypes.cast(hsem, ctypes.c_void_p).value)) == ctypes.c_void_p(-1).value or libc.sem_unlink(nm):
        raise OSError(ctypes.get_errno(), "semaphore")
    sem = Buffer(self.device, 1, dtypes.uint64, options=BufferSpec(external_ptr=addr), preallocate=True)

    worker_args = [ring._buf.va_addr, sysbuf._buf.va_addr if WIN else self.func_ptr('sem_wait')._buf.va_addr, done._buf.va_addr, addr]
    (worker:=threading.Thread(target=self.prgs[worker_prog].fxn, daemon=True, args=[ctypes.c_uint64(x) for x in worker_args])).start()
    return CPUWorker(ring, put, sem, sysbuf, done, worker)
