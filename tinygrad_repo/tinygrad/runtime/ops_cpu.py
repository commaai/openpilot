from __future__ import annotations
import platform, sys, ctypes, mmap, struct
from typing import cast, Any
from tinygrad.helpers import OSX, WIN, mv_address, suppress_finalizing, unwrap, data64_le, cpu_profile
from tinygrad.device import Compiled, TinyELF, Program, HostAllocator
from tinygrad.runtime.support.c import DLL
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.renderer.llvmir import CPULLVMRenderer
from tinygrad.renderer.nir import LVPRenderer
from tinygrad.renderer.isa.x86 import X86Renderer
from tinygrad.runtime.support.elf import jit_loader
from tinygrad.runtime.support.system import RemotePCIDevice, RemoteCmd
from tinygrad.runtime.autogen import libc

# NOTE: MAP_JIT is added to mmap module in python 3.13
MAP_JIT = 0x0800

class CPUProgram(Program['CPUDevice']):
  rt_lib, libm = DLL('rt', 'System' if OSX else 'kernel' if WIN else 'gcc_s'), DLL('m', 'm')

  def _load(self, lib, base=0): return lib if lib[:4] != libc.ELFMAG.encode() else jit_loader(lib, base=base, link_libs=[self.libm, self.rt_lib])

  def __init__(self, dev:CPUDevice, obj:TinyELF):
    self.dev, self.name, self.signature, self.profile_key = dev, obj.name, obj.signature, obj.profile_key
    self.lvp = obj.target.renderer == "LVP"
    if dev.remote is not None: self.fxn:Any = dev.remote.rpc(RemoteCmd.LOAD_PROG, len(obj.lib), payload=obj.lib)[0]
    elif sys.platform == "win32": # mypy doesn't understand when WIN is used here
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

  def __call__(self, *bufs:int, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1),
               vals:tuple[int|None, ...]=(), wait:bool=False, timeout:int|None=None) -> float|None:
    args = [*bufs, *cast(tuple[int, ...], vals)]
    if (remote:=self.dev.remote) is not None:
      data = struct.pack(f'<{len(args)}Q', *(a & 0xffffffffffffffff for a in args))
      ret = (remote._rpc if wait else remote._post)(remote.sock, RemoteCmd.EXEC_PROG, self.fxn, len(args), int(wait), payload=data)
      return ret[0] / 1e9 if ret is not None else None
    with cpu_profile(self.name, self.dev.device, profile_key=self.profile_key) as prof:
      if self.lvp:
        lvp_args = bytearray(12 + (len(bufs) + len(vals)) * 8)
        addr = mv_address(lvp_args)
        struct.pack_into(f'<3I{len(bufs)}Q', lvp_args, 0, *data64_le(addr+12), (len(bufs)+len(vals))*2, *bufs)
        for v,(off,dt) in zip(vals, TinyELF.iter_sig(self.signature[-len(vals):], len(bufs)*8)): struct.pack_into(f'<{dt.fmt}', lvp_args, 12+off, v)
        self.fxn(addr)
      else: self.fxn(*[ctypes.c_uint64(x) for x in args])
    return float(unwrap(prof.en) - prof.st) * 1e-6 if wait else None

  @suppress_finalizing
  def __del__(self):
    if self.dev.remote is None and sys.platform == 'win32':
      ctypes.windll.kernel32.VirtualFree(ctypes.c_void_p(self.addr), ctypes.c_size_t(0), 0x8000) #0x8000 - MEM_RELEASE

class CPUDevice(Compiled):
  wait_timeout_ms = 30000

  @property
  def has_copy_queue(self) -> bool: return False

  def __init__(self, device:str=""):
    self.remote = None
    if len(parts:=device.split(':')) == 3:
      self.remote = RemotePCIDevice("CPU", f"remote:{device[4:]}:0", RemotePCIDevice.connect(parts[1], int(parts[2])))
    super().__init__(device, HostAllocator(self), [ClangRenderer, CPULLVMRenderer, LVPRenderer, X86Renderer], CPUProgram,
      arch={'amd64':'x86_64', 'aarch64':'arm64'}.get(m:=platform.machine().lower(), m)+",native")
    if self.remote is not None: self.peer_group = self.remote.peer_group

  def synchronize(self, timeout:int|None=None):
    if self.remote is not None: self.remote.rpc(RemoteCmd.PING)
    super().synchronize(timeout)
