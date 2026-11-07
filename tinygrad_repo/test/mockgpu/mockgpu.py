import ctypes, time, os, builtins, fcntl, typing
from tinygrad.helpers import DEV, dedup, to_tuple, round_up
from tinygrad.device import Compiled, Device
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp, UPat, Ops, PatternMatcher
from tinygrad.runtime.support.system import FileIOInterface, System
from tinygrad.runtime.support.hcq2 import HWQueue
from tinygrad.runtime.autogen import libc
from test.mockgpu.nv.nvdriver import NVDriver
from test.mockgpu.amd.amddriver import AMDDriver
from test.mockgpu.am.amdriver import AMDriver, AMUSBDriver
start = time.perf_counter()

drivers = [cls() for t in DEV.value if (cls:={"MOCKPCI+AMD": AMDriver, "MOCKKFD+AMD": AMDDriver, "MOCK+AMD": AMDDriver, "MOCKUSB+AMD": AMUSBDriver,
                                              "MOCK+NV": NVDriver}.get(f"{t.interface}+{t.device}"))]
tracked_fds: dict[int, typing.Any] = {}

@ctypes.CFUNCTYPE(None, ctypes.c_uint64)
def mockgpu_doorbell(addr:int): # compiled stores aren't tracked, so the submits report their doorbells
  for st,en,_,wcb in [x for d in drivers for x in d.tracked_addresses]:
    if st <= addr < en: wcb(None, addr - st)
setattr(libc.dll, "mockgpu_doorbell", mockgpu_doorbell)

mocked = {t.device for t in DEV.value if t.interface.startswith("MOCK")}
def hook_doorbells(s:UOp) -> UOp|None:
  hooks = tuple(UOp.custom_function("mockgpu_doorbell").call(b.after(c).getaddr(Device[d].host)) for c in s.toposort() if c.op is Ops.CALL
                for b in c.src[1:] if b.op is Ops.ALLOC and "doorbell" in str(b.arg.name) and (d:=to_tuple(b.device)[0]).split(":")[0] in mocked)
  return None if all(h in s.src for h in hooks) else s.replace(src=tuple(dedup(s.src + hooks)))
Compiled.pm_lower += PatternMatcher([(UPat(Ops.SINK, name="s"), hook_doorbells)])

@ctypes.CFUNCTYPE(None, ctypes.c_uint64)
def mockgpu_progress(_:int): # compiled spin loops wait on the gpus, which run only when called
  for d in drivers: d.progress()
setattr(libc.dll, "mockgpu_progress", mockgpu_progress)

def kick_gpus(body:UOp, loop:UOp, cond:UOp) -> UOp|None: # each iteration of a spin loop runs the gpus
  kick = UOp.custom_function("mockgpu_progress").call(UOp.const(0, dtypes.uint64).after(body))
  return None if kick in cond.src else body.backedge(loop, cond.after(kick))
Compiled.pm_lower += PatternMatcher([(UPat(Ops.BACKEDGE, src=(UPat(name="body"), UPat(name="loop"), UPat(name="cond"))), kick_gpus)])

# this links AMD emulator's kernel back to the original kernel
def emit_kernel_info(ctx:HWQueue, prg:UOp) -> None:
  from tinygrad.runtime.ops_amd import AMDComputeQueue, _amd_program_image
  if isinstance(ctx, AMDComputeQueue):
    from tinygrad.viz.serve import get_elf_section
    name = prg.src[0].arg.name.encode() + b"\x00"
    text = get_elf_section(prg.src[3].arg, ".text").header
    text_size = text.sh_addr + text.sh_size - _amd_program_image(ctx.dev, prg.src[3].arg)[0].entry_point_offset
    ctx.pkt3(ctx.pm4.PACKET3_NOP, *memoryview(name.ljust(round_up(len(name), 4), b"\x00")).cast("I"), text_size)

HWQueue.q_rewrite = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, name="prg"),), allow_any_len=True), emit_kernel_info),
]) + HWQueue.q_rewrite

original_memoryview = builtins.memoryview
class TrackedMemoryView:
  def __init__(self, data, rcb, wcb):
    self.mv = original_memoryview(data)
    self.rcb, self.wcb = rcb, wcb

  def __getitem__(self, index):
    self.rcb(self.mv, index)
    return self.mv[index]

  def __setitem__(self, index, value):
    self.mv[index] = value
    self.wcb(self.mv, index)

  def cast(self, new_type, **kwargs):
    self.mv = self.mv.cast('B').cast(new_type, **kwargs)
    return self

  def __buffer__(self, flags): return self.mv.__buffer__(flags)
  @property
  def nbytes(self): return self.mv.nbytes
  def __len__(self): return len(self.mv)
  def __repr__(self): return repr(self.mv)

def _memoryview(cls, mem):
  if isinstance(mem, int) or isinstance(mem, ctypes.Array):
    addr = ctypes.addressof(mem) if isinstance(mem, ctypes.Array) else mem
    for d in drivers:
      for st,en,rcb,wcb in d.tracked_addresses:
        if st <= addr < en: return TrackedMemoryView(mem, rcb, wcb)
  return original_memoryview(mem)
class _MockMemoryviewMeta(type):
  def __instancecheck__(cls, instance): return isinstance(instance, (original_memoryview, TrackedMemoryView))
builtins.memoryview = _MockMemoryviewMeta("memoryview", (), {'__new__': _memoryview}) # type: ignore

def _open(path, flags):
  for d in drivers:
    for x in d.tracked_files:
      if path == x.path:
        virtfd = d.open(path, flags, 0o777, x)
        tracked_fds[virtfd.fd] = virtfd
        return virtfd.fd
  return os.open(path, flags, 0o777) if os.path.exists(path) else None

class MockFileIOInterface(FileIOInterface):
  def __init__(self, path:str="", flags:int=os.O_RDONLY, fd:int|None=None):
    self.path = path
    self.fd = fd or _open(path, flags)

  def __del__(self):
    if self.fd in tracked_fds:
      tracked_fds[self.fd].close(self.fd)
      tracked_fds.pop(self.fd)
    elif self.fd is not None: os.close(self.fd)

  def ioctl(self, request, arg):
    if self.fd in tracked_fds:
      return tracked_fds[self.fd].ioctl(self.fd, request, ctypes.addressof(arg))
    return fcntl.ioctl(self.fd, request, arg)

  def mmap(self, start, sz, prot, flags, offset):
    if self.fd in tracked_fds:
      return tracked_fds[self.fd].mmap(start, sz, prot, flags, self.fd, offset)
    return libc.mmap(start, sz, prot, flags, self.fd, offset)

  def read(self, size=None, binary=False, offset=None):
    if self.fd in tracked_fds:
      if offset is not None: tracked_fds[self.fd].seek(offset)
      return tracked_fds[self.fd].read_contents(size)
    if binary: raise NotImplementedError()
    with open(self.fd, "rb" if binary else "r", closefd=False) as file:
      if file.tell() >= os.fstat(self.fd).st_size: file.seek(0)
      return file.read(size)

  def listdir(self):
    if self.fd in tracked_fds:
      return tracked_fds[self.fd].list_contents()
    return os.listdir(self.path)

  def write(self, content, binary=False, offset=None):
    if self.fd in tracked_fds:
      if offset is not None: tracked_fds[self.fd].seek(offset)
      return tracked_fds[self.fd].write_contents(content)
    raise NotImplementedError()
  def seek(self, offset):
    if self.fd in tracked_fds:
      tracked_fds[self.fd].seek(offset)
    else:
      os.lseek(self.fd, offset, os.SEEK_CUR)
  @staticmethod
  def anon_mmap(start, sz, prot, flags, offset):
    return FileIOInterface._mmap(start, sz, prot, flags & ~0x4a000, -1, offset)  # strip MAP_LOCKED|MAP_POPULATE|MAP_HUGETLB
  @staticmethod
  def exists(path): return _open(path, os.O_RDONLY) is not None
  @staticmethod
  def readlink(path): raise NotImplementedError()
  @staticmethod
  def eventfd(initval, flags=None): NotImplementedError()

System.lock_memory = lambda addr, size: None # type: ignore[method-assign,has-type] # nothing dmas from host memory, like MAP_LOCKED in anon_mmap
