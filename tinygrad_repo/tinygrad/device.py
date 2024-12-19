from __future__ import annotations
from dataclasses import dataclass, replace
from collections import defaultdict
from typing import Optional, Dict, Tuple, Any, Iterator
import multiprocessing, importlib, inspect, functools, pathlib, os, ctypes, contextlib, sys, re
from tinygrad.helpers import CI, OSX, getenv, diskcache_get, diskcache_put, DEBUG, GlobalCounters, flat_mv, from_mv
from tinygrad.dtype import DType, ImageDType, PtrDType, dtypes
from tinygrad.renderer import Renderer
from tinygrad.ops import UOp, buffers

# **************** Device ****************

class _Device:
  def __init__(self) -> None:
    self._devices = [x.stem[len("ops_"):].upper() for x in (pathlib.Path(__file__).parent/"runtime").iterdir() if x.stem.startswith("ops_")]
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
    ret = [cls for cname, cls in inspect.getmembers(importlib.import_module(f'{__name__.split(".")[0]}.runtime.ops_{x.lower()}')) \
           if (cname.lower() == x.lower() + "device")][0](ix)
    if DEBUG >= 1: print(f"opened device {ix} from pid:{os.getpid()}")
    return ret
  @property
  def default(self) -> Compiled: return self[self.DEFAULT]
  def get_available_devices(self) -> Iterator[str]:
    for device in ["METAL", "AMD", "NV", "CUDA", "QCOM", "GPU", "CLANG", "LLVM"]:
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
               lb_refcount=0, uop_ref:Optional[UOp]=None, base:Optional[Buffer]=None, offset:int=0, preallocate=False):
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
    if uop_ref is not None: buffers[uop_ref] = self
  @property
  def base(self) -> Buffer: return self._base if self._base is not None else self
  @property
  def lb_refcount(self): return self.base._lb_refcount
  def ref(self, cnt): self.base._lb_refcount += cnt
  def is_allocated(self) -> bool: return hasattr(self, '_buf')
  def ensure_allocated(self) -> Buffer: return self.allocate() if not self.is_allocated() else self
  def allocate(self, opaque=None, external_ptr=None) -> Buffer:
    assert not self.is_allocated(), "can't allocate already allocated buffer"
    self.allocator = Device[self.device].allocator
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
  def __reduce__(self):
    buf = None
    if len(uop_refs:=[u for u,v in buffers.items() if self is v]) > 1: raise RuntimeError(f"double ref to buffer? {len(uop_refs)}")
    uop_ref = None if len(uop_refs) == 0 else uop_refs[0]
    if self._base is not None:
      return self.__class__, (self.device, self.size, self.dtype, None, None, None, 0, uop_ref, self.base, self.offset, self.is_allocated())
    if self.device == "NPY": return self.__class__, (self.device, self.size, self.dtype, self._buf, self.options, None, self.lb_refcount, uop_ref)
    if self.is_allocated():
      buf = bytearray(self.nbytes)
      self.copyout(memoryview(buf))
    return self.__class__, (self.device, self.size, self.dtype, None, self.options, buf, self.lb_refcount, uop_ref)
  @property
  def nbytes(self): return self.size*self.dtype.itemsize
  def __del__(self):
    if not self.is_allocated(): return
    if self._base is None and (self.options is None or self.options.external_ptr is None):
      if not self.device.startswith("DISK"): GlobalCounters.mem_used -= self.nbytes
      self.allocator.free(self._buf, self.nbytes, self.options)
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
  # overriden in LRUAllocator
  def alloc(self, size:int, options:Optional[BufferSpec]=None):
    assert size > 0, f"alloc size must be positve, getting {size}"
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
  def __init__(self): self.cache: Dict[Tuple[int, Optional[BufferSpec]], Any] = defaultdict(list)
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
    if getenv("LRU", 1) and (options is None or not options.nolru): self.cache[(size, options)].append(opaque)
    else: super().free(opaque, size, options)

class _MallocAllocator(LRUAllocator):
  def _alloc(self, size:int, options:BufferSpec):
    return (ctypes.c_uint8 * size).from_address(options.external_ptr) if options.external_ptr else (ctypes.c_uint8 * size)()
  def _as_buffer(self, src) -> memoryview: return flat_mv(memoryview(src))
  def _copyin(self, dest, src:memoryview): ctypes.memmove(dest, from_mv(src), len(src))
  def _copyout(self, dest:memoryview, src): ctypes.memmove(from_mv(dest), src, len(dest))
  def _offset(self, buf, size:int, offset:int): return from_mv(self._as_buffer(buf)[offset:offset+size])

MallocAllocator = _MallocAllocator()

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
  def __init__(self, device:str, allocator:Allocator, renderer:Optional[Renderer], compiler:Optional[Compiler], runtime, graph=None):
    self.device, self.allocator, self.compiler, self.runtime, self.graph = device, allocator, compiler or Compiler(), runtime, graph
    self.renderer = renderer or Renderer()
  def synchronize(self):
    """
    Synchronize all pending operations on the device.

    This method ensures that all previously queued operations on the device have been completed before proceeding.
    """
    # override this in your device implementation

# TODO: move this to each Device
def is_dtype_supported(dtype:DType, device:Optional[str]=None) -> bool:
  if device is None: device = Device.DEFAULT
  if dtype == dtypes.bfloat16:
    # NOTE: this requires bf16 buffer support
    return device in {"AMD"} or (device in {"CUDA", "NV"} and not CI and not getenv("PTX"))
  if device == "WEBGPU": return dtype in [dtypes.bool, dtypes.char, dtypes.uchar, dtypes.short,
                                          dtypes.ushort, dtypes.float, dtypes.int32, dtypes.uint32]
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
