from __future__ import annotations
from dataclasses import dataclass, replace
from collections import defaultdict
from typing import Optional, Any, Iterator, Generator
import multiprocessing, importlib, inspect, functools, pathlib, os, ctypes, ctypes.util, platform, contextlib, sys, re, atexit, pickle, decimal, time
from tinygrad.helpers import CI, OSX, LRU, getenv, diskcache_get, diskcache_put, DEBUG, GlobalCounters, flat_mv, from_mv, PROFILE, temp, mv_address, \
                             cpu_time_execution, colored, Context, round_up
from tinygrad.dtype import DType, ImageDType, PtrDType, dtypes
from tinygrad.renderer import Renderer

# **************** Device ****************

ALL_DEVICES = ["METAL", "AMD", "NV", "CUDA", "QCOM", "GPU", "CPU", "LLVM", "DSP", "WEBGPU"]
class _Device:
  def __init__(self) -> None:
    self._devices = [x.stem[len("ops_"):].upper() for x in (pathlib.Path(__file__).parent/"runtime").iterdir() if x.stem.startswith("ops_")]
    self._opened_devices:set[str] = set()
  @functools.lru_cache(maxsize=None)  # this class is a singleton, pylint: disable=method-cache-max-size-none
  def _canonicalize(self, device:str) -> str: return re.sub(r":0$", "", (d:=device.split(":", 1)[0].upper()) + device[len(d):])
  # NOTE: you can't cache canonicalize in case Device.DEFAULT changes
  def canonicalize(self, device:Optional[str]) -> str: return self._canonicalize(device) if device is not None else Device.DEFAULT
  def __getitem__(self, ix:str) -> Compiled: return self.__get_canonicalized_item(self.canonicalize(ix))
  @functools.lru_cache(maxsize=None)  # this class is a singleton, pylint: disable=method-cache-max-size-none
  def __get_canonicalized_item(self, ix:str) -> Compiled:
    cpn = multiprocessing.current_process().name
    assert (cpn == "MainProcess") or ix.split(":")[0] in ["DISK", "NPY", "PYTHON"], f"can only open device {ix} from parent, not {cpn}"
    x = ix.split(":")[0].upper()
    ret = [cls for cname, cls in inspect.getmembers(importlib.import_module(f'tinygrad.runtime.ops_{x.lower()}')) \
           if (cname.lower() == x.lower() + "device")][0](ix)
    if DEBUG >= 1: print(f"opened device {ix} from pid:{os.getpid()}")
    self._opened_devices.add(ix)
    return ret
  @property
  def default(self) -> Compiled: return self[self.DEFAULT]
  def get_available_devices(self) -> Iterator[str]:
    for device in ALL_DEVICES:
      with contextlib.suppress(Exception): yield self[device].device
  @functools.cached_property
  def DEFAULT(self) -> str:
    if (from_env:=next((d for d in self._devices if d not in ["DISK", "NPY"] and getenv(d) == 1), None)): return from_env
    try:
      device = next(self.get_available_devices())
      os.environ[device] = "1"   # we set this in environment for spawned children
      return device
    except StopIteration as exc: raise RuntimeError("no usable devices") from exc
Device = _Device()
atexit.register(lambda: [Device[dn].finalize() for dn in Device._opened_devices])

# **************** Profile ****************

class ProfileEvent: pass

@dataclass(frozen=True)
class ProfileDeviceEvent(ProfileEvent):
  device:str; comp_tdiff:decimal.Decimal=decimal.Decimal(0); copy_tdiff:decimal.Decimal=decimal.Decimal(0) # noqa: E702

@dataclass(frozen=True)
class ProfileRangeEvent(ProfileEvent): device:str; name:str; st:decimal.Decimal; en:decimal.Decimal; is_copy:bool # noqa: E702

@dataclass(frozen=True)
class ProfileGraphEntry: device:str; name:str; st_id:int; en_id:int; is_copy:bool # noqa: E702

@dataclass(frozen=True)
class ProfileGraphEvent(ProfileEvent): ents:list[ProfileGraphEntry]; deps:list[list[int]]; sigs:list[decimal.Decimal] # noqa: E702

@dataclass
class ProfileResult: st:Optional[int]=None; en:Optional[int]=None # noqa: E702

@contextlib.contextmanager
def cpu_profile(name, device="CPU", is_copy=False, display=True) -> Generator[ProfileResult, None, None]:
  yield (res:=ProfileResult(st:=time.perf_counter_ns()))
  res.en = en = time.perf_counter_ns()
  if PROFILE and display:
    Compiled.profile_events += [ProfileRangeEvent(device, name, decimal.Decimal(st) / 1000, decimal.Decimal(en) / 1000, is_copy=is_copy)]

# **************** Buffer + Allocators ****************


@dataclass(frozen=True, eq=True)
class BufferSpec:
  # TODO: move device, size, dtype here?
  image: Optional[ImageDType] = None
  uncached: bool = False
  cpu_access: bool = False
  host: bool = False
  nolru: bool = False
  external_ptr: Optional[int] = None

class Buffer:
  def __init__(self, device:str, size:int, dtype:DType, opaque:Any=None, options:Optional[BufferSpec]=None, initial_value:Optional[bytes]=None,
               lb_refcount=0, base:Optional[Buffer]=None, offset:int=0, preallocate=False):
    if isinstance(dtype, ImageDType): options = BufferSpec(image=dtype) # TODO: image hack shouldn't be here. where should it be?
    else: assert isinstance(dtype, DType) and not isinstance(dtype, PtrDType)
    self.device, self.size, self.dtype, self.options, self.offset = device, size, dtype, options, offset
    if base is None:
      assert offset == 0, "base buffers can't have offset"
      self._base = None
      self._lb_refcount = lb_refcount
      if opaque is not None: self.allocate(opaque)
      if initial_value is not None:
        self.allocate()
        self.copyin(memoryview(initial_value))
    else:
      assert base._base is None, "base can't have a base"
      assert device == base.device, "base must have the same device"
      self._base = base
    if preallocate: self.allocate()
  @property
  def base(self) -> Buffer: return self._base if self._base is not None else self
  @property
  def lb_refcount(self): return self.base._lb_refcount
  def ref(self, cnt): self.base._lb_refcount += cnt
  def is_allocated(self) -> bool: return hasattr(self, '_buf')
  def ensure_allocated(self) -> Buffer: return self.allocate() if not self.is_allocated() else self
  def allocate(self, opaque=None, external_ptr=None) -> Buffer:
    assert not self.is_allocated(), "can't allocate already allocated buffer"
    self.allocator:Allocator = Device[self.device].allocator
    if external_ptr is not None:
      self.options = replace(self.options, external_ptr=external_ptr) if self.options else BufferSpec(external_ptr=external_ptr)
    if self._base is not None:
      self._base.ensure_allocated()
      assert hasattr(self.allocator, "_offset"), "offset function required for view"
      self._buf: Any = self.allocator._offset(self.base._buf, self.nbytes, self.offset)
    else:
      self._buf = opaque if opaque is not None else self.allocator.alloc(self.nbytes, self.options)
      if not self.device.startswith("DISK"): GlobalCounters.mem_used += self.nbytes
    return self
  def deallocate(self):
    assert self.is_allocated(), "buffer must be allocated to deallocate"
    if self._base is None and (self.options is None or self.options.external_ptr is None):
      if not self.device.startswith("DISK"): GlobalCounters.mem_used -= self.nbytes
      self.allocator.free(self._buf, self.nbytes, self.options)
    del self._buf
  def __reduce__(self):
    buf = None
    if self._base is not None:
      return self.__class__, (self.device, self.size, self.dtype, None, None, None, 0, self.base, self.offset, self.is_allocated())
    if self.device == "NPY": return self.__class__, (self.device, self.size, self.dtype, self._buf, self.options, None, self.lb_refcount)
    if self.is_allocated():
      buf = bytearray(self.nbytes)
      self.copyout(memoryview(buf))
    return self.__class__, (self.device, self.size, self.dtype, None, self.options, buf, self.lb_refcount)
  @property
  def nbytes(self): return self.size*self.dtype.itemsize
  def __del__(self): (not self.is_allocated()) or self.deallocate()
  def __repr__(self):
    return f"<buf real:{self.is_allocated()} device:{self.device} size:{self.size} dtype:{self.dtype}" + \
           (f" offset:{self.offset}" if hasattr(self, "base") else "") + (f" {self.options=}" if self.options is not None else "") + ">"
  def as_buffer(self, allow_zero_copy=False, force_zero_copy=False) -> memoryview:
    # zero copy with as_buffer (disabled by default due to use after free)
    if (force_zero_copy or allow_zero_copy) and hasattr(self.allocator, '_as_buffer') and (self.options is None or self.options.image is None):
      return self.allocator._as_buffer(self._buf)
    assert not force_zero_copy, "force zero copy was passed, but copy is required"
    return self.copyout(memoryview(bytearray(self.nbytes)))
  def copyin(self, mv:memoryview):
    mv = flat_mv(mv)
    assert len(mv) == self.nbytes, f"size mismatch, {len(mv)=} != {self.dtype=} {self.size=}"
    assert self.is_allocated(), "can't copyin to unallocated buffer"
    self.allocator._copyin(self._buf, mv)
    return self
  def copyout(self, mv:memoryview) -> memoryview:
    mv = flat_mv(mv)
    assert len(mv) == self.nbytes, f"size mismatch, {len(mv)=} != {self.dtype=} {self.size=}"
    assert self.is_allocated(), "can't copyout unallocated buffer"
    self.allocator._copyout(mv, self._buf)
    return mv
  def view(self, size:int, dtype:DType, offset:int) -> Buffer:
    assert offset < self.nbytes, "offset must be less than nbytes"
    if self._base is not None: return Buffer(self.device, size, dtype, base=self._base, offset=self.offset+offset)
    return Buffer(self.device, size, dtype, base=self, offset=offset)

# TODO: size, dest, src are the same type. can we enforce this?
class Allocator:
  # overridden in LRUAllocator
  def alloc(self, size:int, options:Optional[BufferSpec]=None):
    assert size > 0, f"alloc size must be positive, getting {size}"
    return self._alloc(size, options if options is not None else BufferSpec())
  def free(self, opaque, size:int, options:Optional[BufferSpec]=None): self._free(opaque, options if options is not None else BufferSpec())

  # implemented by the runtime
  def _alloc(self, size:int, options:BufferSpec): raise NotImplementedError("need alloc")
  def _free(self, opaque, options:BufferSpec): pass  # if opaque is a Python object, you don't need a free
  def _copyin(self, dest, src:memoryview): raise NotImplementedError("need copyin")
  def _copyout(self, dest:memoryview, src): raise NotImplementedError("need copyout")
  # def _as_buffer(self, src) -> memoryview:
  # def _offset(self, buf, size:int, offset:int):
  # def _transfer(self, dest, src, sz:int, src_dev, dest_dev):

class LRUAllocator(Allocator):
  """
  The LRU Allocator is responsible for caching buffers.
  It ensures that buffers are not freed until it is absolutely necessary, optimizing performance.
  """
  def __init__(self): self.cache: dict[tuple[int, Optional[BufferSpec]], Any] = defaultdict(list)
  def alloc(self, size:int, options:Optional[BufferSpec]=None):
    if len(c := self.cache[(size, options)]): return c.pop()
    try: return super().alloc(size, options)
    except (RuntimeError, MemoryError):
      self.free_cache()
      return super().alloc(size, options)
  def free_cache(self):
    for (sz,options),opaques in self.cache.items():
      for opaque in opaques: super().free(opaque, sz, options)
      opaques.clear()
  def free(self, opaque:Any, size:int, options:Optional[BufferSpec]=None):
    if LRU and (options is None or not options.nolru): self.cache[(size, options)].append(opaque)
    else: super().free(opaque, size, options)

class _MallocAllocator(LRUAllocator):
  def _alloc(self, size:int, options:BufferSpec):
    # must be aligned to 0x20 for 256-bit ymm registers
    # TODO: investigate if this is the cause of nondeterminism in speed
    alignment = 0x1000 if size >= 0x1000 else 0x20
    return (ctypes.c_uint8 * size).from_address(options.external_ptr) if options.external_ptr else self._alloc_aligned(size, alignment)
  def _alloc_aligned(self, size:int, alignment:int):
    buffer = (ctypes.c_uint8 * (size + alignment))()
    offset = round_up(ctypes.addressof(buffer), alignment) - ctypes.addressof(buffer)
    return (ctypes.c_uint8 * size).from_buffer(buffer, offset)
  def _as_buffer(self, src) -> memoryview: return flat_mv(memoryview(src))
  def _copyin(self, dest, src:memoryview): ctypes.memmove(dest, from_mv(src), len(src))
  def _copyout(self, dest:memoryview, src): ctypes.memmove(from_mv(dest), src, len(dest))
  def _offset(self, buf, size:int, offset:int): return from_mv(self._as_buffer(buf)[offset:offset+size])

MallocAllocator = _MallocAllocator()

# NOTE: MAP_JIT is added to mmap module in python 3.13
MAP_JIT = 0x0800

# CPUProgram is a jit/shellcode program that can be just mmapped and jumped to
class CPUProgram:
  rt_lib = ctypes.CDLL(ctypes.util.find_library('System' if OSX else 'kernel32') if OSX or sys.platform == "win32" else 'libgcc_s.so.1')
  atomic_lib = ctypes.CDLL(ctypes.util.find_library('atomic')) if sys.platform == "linux" else None

  def __init__(self, name:str, lib:bytes):
    if sys.platform == "win32":
      PAGE_EXECUTE_READWRITE = 0x40
      MEM_COMMIT =  0x1000
      MEM_RESERVE = 0x2000
      ctypes.windll.kernel32.VirtualAlloc.restype = ctypes.c_void_p
      self.mem = ctypes.windll.kernel32.VirtualAlloc(ctypes.c_void_p(0), ctypes.c_size_t(len(lib)), MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE)
      ctypes.memmove(self.mem, lib, len(lib))
      ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
      proc = ctypes.windll.kernel32.GetCurrentProcess()
      ctypes.windll.kernel32.FlushInstructionCache(ctypes.c_void_p(proc), ctypes.c_void_p(self.mem), ctypes.c_size_t(len(lib)))
      self.fxn = ctypes.CFUNCTYPE(None)(self.mem)
    else:
      from mmap import mmap, PROT_READ, PROT_WRITE, PROT_EXEC, MAP_ANON, MAP_PRIVATE
      # On apple silicon with SPRR enabled (it always is in macos) RWX pages are unrepresentable: https://blog.svenpeter.dev/posts/m1_sprr_gxf/
      # MAP_JIT allows us to easily flip pages from RW- to R-X and vice versa. It is a noop on intel cpus. (man pthread_jit_write_protect_np)
      self.mem = mmap(-1, len(lib), MAP_ANON | MAP_PRIVATE | (MAP_JIT if OSX else 0), PROT_READ | PROT_WRITE | PROT_EXEC)

      if OSX: CPUProgram.rt_lib.pthread_jit_write_protect_np(False)
      self.mem.write(lib)
      if OSX: CPUProgram.rt_lib.pthread_jit_write_protect_np(True)

      # __clear_cache isn't a normal libc function, but a compiler support routine found in libgcc_s for gcc and compiler-rt for clang.
      # libgcc_s comes as shared library but compiler-rt is only a bunch of static library archives which we can't directly load, but fortunately
      # it somehow found its way into libSystem on macos (likely because it used __builtin_clear_cache) and libgcc_s is ~always present on linux
      # Using ["name"] instead of .name because otherwise name is getting mangled: https://docs.python.org/3.12/reference/expressions.html#index-5
      CPUProgram.rt_lib["__clear_cache"](ctypes.c_void_p(mv_address(self.mem)), ctypes.c_void_p(mv_address(self.mem) + len(lib)))

      self.fxn = ctypes.CFUNCTYPE(None)(mv_address(self.mem))

  def __call__(self, *bufs, vals=(), wait=False):
    args = list(bufs) + list(vals)
    # NOTE: replace this by --target={host's triple}-elf in clang args once we only support macos sequoia and later.
    # Apple relaxes abi requirement for stack arguments to always be at least 8 byte aligned on arm64
    # https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms
    # This hack is required because clang/llvm bug doesn't allow us to just use {host's triple}+'-elf' (relocation failures)
    # The bug was fixed in https://github.com/llvm/llvm-project/commit/454cc36630296262cdb6360b60f90a64a97f7f1a but was only backported to xcode 16+
    if platform.machine() == "arm64" and OSX: args = args[:8] + [ctypes.c_int64(a) if isinstance(a, int) else a for a in args[8:]]
    return cpu_time_execution(lambda: self.fxn(*args), enable=wait)

  def __del__(self):
    if sys.platform == 'win32': ctypes.windll.kernel32.VirtualFree(ctypes.c_void_p(self.mem), ctypes.c_size_t(0), 0x8000) #0x8000 - MEM_RELEASE

# **************** for Compiled Devices ****************

class CompileError(Exception): pass

class Compiler:
  def __init__(self, cachekey:Optional[str]=None): self.cachekey = None if getenv("DISABLE_COMPILER_CACHE") else cachekey
  def compile(self, src:str) -> bytes: return src.encode()   # NOTE: empty compiler is the default
  def compile_cached(self, src:str) -> bytes:
    if self.cachekey is None or (lib := diskcache_get(self.cachekey, src)) is None:
      assert not getenv("ASSERT_COMPILE"), f"tried to compile with ASSERT_COMPILE set\n{src}"
      lib = self.compile(src)
      if self.cachekey is not None: diskcache_put(self.cachekey, src, lib)
    return lib
  def disassemble(self, lib:bytes): pass

class Compiled:
  profile_events:list[ProfileEvent] = [ProfileDeviceEvent("CPU")] # NOTE: CPU is the default device.

  def __init__(self, device:str, allocator:Allocator, renderer:Optional[Renderer], compiler:Optional[Compiler], runtime, graph=None):
    self.device, self.allocator, self.compiler, self.runtime, self.graph = device, allocator, compiler or Compiler(), runtime, graph
    self.renderer = renderer or Renderer()
  def synchronize(self):
    """
    Synchronize all pending operations on the device.

    This method ensures that all previously queued operations on the device have been completed before proceeding.
    """
    # override this in your device implementation
  def _at_profile_finalize(self):
    """
    Called at the end of profiling to allow the device to finalize any profiling.
    """
    # override this in your device implementation
  def finalize(self):
    """
    Called at the end of process lifetime to allow the device to finalize.
    """
    # override this in your device implementation

# TODO: move this to each Device
def is_dtype_supported(dtype:DType, device:Optional[str]=None) -> bool:
  if device is None: device = Device.DEFAULT
  if dtype == dtypes.bfloat16:
    # NOTE: this requires bf16 buffer support
    return device in {"AMD"} or (device in {"CUDA", "NV"} and not CI and not getenv("PTX"))
  if device == "WEBGPU": return dtype in [dtypes.bool, dtypes.char, dtypes.uchar, dtypes.short,
                                          dtypes.ushort, dtypes.float, dtypes.int32, dtypes.uint32, dtypes.half]
  # for CI GPU and OSX, cl_khr_fp16 isn't supported
  # for CI LLVM, it segfaults because it can't link to the casting function
  # CI CUDA architecture is sm_35 but we need at least sm_70 to run fp16 ALUs
  # PYTHON supports half memoryview in 3.12+ https://github.com/python/cpython/issues/90751
  if dtype == dtypes.half:
    if device == "GPU": return not CI and not OSX
    if device in ["CUDA", "NV"]: return not CI
    if device == "LLVM": return OSX
    if device == "PYTHON": return sys.version_info >= (3, 12)
  if dtype == dtypes.float64: return device != "METAL" and not (OSX and device == "GPU")
  return True

if PROFILE:
  @atexit.register
  def finalize_profile():
    devs = [Device[d] for d in Device._opened_devices]
    for dev in devs: dev.synchronize()
    for dev in devs: dev._at_profile_finalize()

    with open(fn:=temp("profile.pkl", append_user=True), "wb") as f: pickle.dump(Compiled.profile_events, f)

    from tinygrad.ops import launch_viz
    launch_viz("PROFILE", fn)

if __name__ == "__main__":
  for device in ALL_DEVICES:
    try:
      _ = Device[device].device
      try:
        from tinygrad import Tensor
        with Context(CACHELEVEL=0): test = (Tensor([1,2,3], device=device) * 2).tolist()
        if test != [2,4,6]: raise ValueError(f"got {test} instead of [2, 4, 6]")
        result = colored("PASS", "green")
      except Exception as e:
        result = f"{colored('FAIL', 'yellow')} {e}"
    except Exception as e:
      result = f"{colored('FAIL', 'red')} {e}"
    print(f"{'*' if device == Device.DEFAULT else ' '} {device:10s}: {result}")
