from __future__ import annotations
import platform, ctypes, struct, json
from typing import Any
from tinygrad.helpers import OSX, unwrap, data64_le, cpu_profile
from tinygrad.device import Compiled, TinyELF, Program, HostAllocator, BufferSpec
from tinygrad.runtime.support.c import DLL
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.renderer.llvmir import CPULLVMRenderer
from tinygrad.renderer.nir import LVPRenderer
from tinygrad.renderer.x86 import X86Renderer
from tinygrad.runtime.support.elf import jit_loader
from tinygrad.runtime.support.system import RemotePCIDevice, RemoteCmd
from tinygrad.runtime.autogen import libc

class CPUProgram(Program['CPUDevice']):
  rt_lib, libm = DLL('rt', 'System' if OSX else 'gcc_s'), DLL('m', 'm')

  def _load(self, lib, base=0): return lib if lib[:4] != libc.ELFMAG.encode() else jit_loader(lib, base=base, link_libs=list(DLL._loaded_.values()),
                                                                                              link_syms={n:p.buf.buf for n,p in self.links.items()})

  def __init__(self, dev:CPUDevice, obj:TinyELF, links:dict[str,CPUProgram]):
    self.dev, self.name, self.signature, self.profile_key, self.links = dev, obj.name, obj.signature, obj.profile_key, links
    self.lvp = obj.target.renderer == "LVP"
    self.packer = struct.Struct(''.join(unwrap(dt.fmt) for _, dt in TinyELF.iter_sig(self.signature)))
    if dev.remote is not None:
      link_ids = json.dumps({n:p.fxn for n,p in links.items()}).encode()
      self.fxn:Any = dev.remote.rpc(RemoteCmd.LOAD_PROG, len(obj.lib), len(link_ids), payload=obj.lib + link_ids)[0]
    else:
      self.buf = dev.allocator.alloc(len(obj.lib), BufferSpec(nolru=True))
      loaded = self._load(obj.lib, self.buf.buf)
      unwrap(self.buf.host)[:len(loaded)] = loaded

      # RWX pages are unrepresentable on apple silicon (https://blog.svenpeter.dev/posts/m1_sprr_gxf/), so flip the written pages to R-X
      libc.mprotect(ctypes.c_void_p(self.buf.buf), len(obj.lib), libc.PROT_READ | libc.PROT_EXEC)

      # __clear_cache isn't a normal libc function, but a compiler support routine found in libgcc_s for gcc and compiler-rt for clang.
      # libgcc_s comes as shared library but compiler-rt is only a bunch of static library archives which we can't directly load, but fortunately
      # it somehow found its way into libSystem on macos (likely because it used __builtin_clear_cache) and libgcc_s is ~always present on linux
      # Using ["name"] instead of .name because otherwise name is getting mangled: https://docs.python.org/3.12/reference/expressions.html#index-5
      if 'rt' in DLL._loaded_: CPUProgram.rt_lib["__clear_cache"](ctypes.c_void_p(self.buf.buf), ctypes.c_void_p(self.buf.buf + len(loaded)))
      else:
        # msync should be a universal POSIX way to do this
        libc.msync(ctypes.c_void_p(self.buf.buf), len(loaded), libc.MS_SYNC | libc.MS_INVALIDATE)

      self.fxn = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(self.buf.buf)

  def __call__(self, *args:int, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1),
               wait:bool=False, timeout:int|None=None) -> float|None:
    hdr = 12 if self.lvp else 0
    kernargs = ctypes.create_string_buffer(hdr + len(args) * 8)
    if self.lvp: struct.pack_into('<3I', kernargs, 0, *data64_le(ctypes.addressof(kernargs) + hdr), len(args) * 2)
    self.packer.pack_into(kernargs, hdr, *args)
    if (remote:=self.dev.remote) is not None:
      ret = (remote._rpc if wait else remote._post)(remote.sock, RemoteCmd.EXEC_PROG, self.fxn, len(kernargs), int(wait), payload=bytes(kernargs))
      return ret[0] / 1e9 if ret is not None else None
    with cpu_profile(self.name, self.dev.device, profile_key=self.profile_key) as prof: self.fxn(kernargs)
    return float(unwrap(prof.en) - prof.st) * 1e-6 if wait else None

class CPUDevice(Compiled):
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
