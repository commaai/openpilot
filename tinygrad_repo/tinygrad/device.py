from __future__ import annotations
from dataclasses import dataclass, replace
from collections import defaultdict
from typing import Any, Generic, TypeVar, Iterator, Generator, TYPE_CHECKING
import importlib, inspect, functools, pathlib, os, contextlib, re, atexit, pickle, decimal
from tinygrad.helpers import LRU, getenv, diskcache_get, diskcache_put, DEBUG, GlobalCounters, flat_mv, PROFILE, temp, colored
from tinygrad.helpers import Context, CCACHE, ALLOW_DEVICE_USAGE, MAX_BUFFER_SIZE, cpu_events, ProfileEvent, ProfilePointEvent, suppress_finalizing
from tinygrad.helpers import select_by_name, select_first_inited, DEV, TracingKey, size_to_str, pluralize
from tinygrad.dtype import DType, PtrDType, _to_np_dtype
if TYPE_CHECKING: from tinygrad.renderer import Renderer

# **************** Device ****************

ALL_DEVICES = ["METAL", "AMD", "NV", "CUDA", "QCOM", "CL", "CPU", "DSP", "WEBGPU"]
class _Device:
  def __init__(self) -> None:
    self._devices = [x.stem[len("ops_"):].upper() for x in (pathlib.Path(__file__).parent/"runtime").iterdir() if x.stem.startswith("ops_")]
    self._opened_devices:set[str] = set()
  @functools.cache  # this class is a singleton, pylint: disable=method-cache-max-size-none
  def _canonicalize(self, device:str) -> str: return re.sub(r":0$", "", (d:=device.split(":", 1)[0].upper()) + device[len(d):])
  # NOTE: you can't cache canonicalize in case Device.DEFAULT changes
  def canonicalize(self, device:str|None) -> str: return self._canonicalize(device if device is not None else Device.DEFAULT)
  def __getitem__(self, ix:str) -> Compiled:
    ix = self.canonicalize(ix)
    assert ALLOW_DEVICE_USAGE or ix.split(":")[0] in ["DISK", "TINYFS", "NPY", "PYTHON"], f"usage of device {ix} disallowed"
    return self.__get_canonicalized_item(ix)
  @functools.cache  # this class is a singleton, pylint: disable=method-cache-max-size-none
  def __get_canonicalized_item(self, ix:str) -> Compiled:
    base = (__package__ or __name__).split('.')[0]  # tinygrad
    x = ix.split(":")[0].lower()
    ret = [cls for cname, cls in inspect.getmembers(importlib.import_module(f'{base}.runtime.ops_{x}')) \
           if (cname.lower() == x + "device")][0](ix)
    if DEBUG >= 1: print(f"opened device {ix} from pid:{os.getpid()}")
    self._opened_devices.add(ix)
    return ret
  @property
  def default(self) -> Compiled: return self[self.DEFAULT]
  def get_available_devices(self) -> Iterator[str]:
    for device in ALL_DEVICES:
      with contextlib.suppress(Exception): yield self[device].device
  @property
  def DEFAULT(self) -> str: return DEV.device or self._select_device
  @DEFAULT.setter
  def DEFAULT(self, v): raise AttributeError(f'setting Device.DEFAULT is deprecated, use "with Context(DEV={v!r})" or "DEV.value = {v!r}"')
  @functools.cached_property
  def _select_device(self) -> str:
    assert (dev:=next((d for d in self._devices if d not in ["DISK", "TINYFS", "NPY"] and getenv(d) == 1), None)) is None, \
      f"{dev}=1 is deprecated, use DEV={dev} instead"
    try:
      device = next(self.get_available_devices())
      os.environ["DEV"] = device   # we set this in environment for spawned children
      return device
    except StopIteration as exc: raise RuntimeError("no usable devices") from exc
Device: _Device = _Device()
atexit.register(lambda: [Device[dn].finalize() for dn in Device._opened_devices])

def canonicalize_device(device:str|tuple|list|None) -> str|tuple[str, ...]:
  if not isinstance(device, (tuple, list)): return Device.canonicalize(device)
  return canonical[0] if len(canonical:=tuple(Device.canonicalize(d) for d in device)) == 1 else canonical

# **************** Profile ****************

@dataclass(frozen=True)
class ProfileDeviceEvent(ProfileEvent): device:str; tdiff:decimal.Decimal=decimal.Decimal(0); props:dict[str,Any]|None=None # noqa: E702

@dataclass(frozen=True)
class ProfileProgramEvent(ProfileEvent): device:str; name:str; lib:bytes|None; base:int|None; tag:int|None=None # noqa: E702

@dataclass(frozen=True)
class ProfileGraphEntry: device:str; name:str|TracingKey; st_id:int; en_id:int # noqa: E702

@dataclass(frozen=True)
class ProfileGraphEvent(ProfileEvent): ents:list[ProfileGraphEntry]; deps:list[list[int]]; sigs:list[decimal.Decimal] # noqa: E702

# **************** Buffer + Allocators ****************

@dataclass(frozen=True, eq=True)
class BufferSpec:
  # TODO: move device, size, dtype here?
  uncached: bool = False
  cpu_access: bool = False
  host: bool = False
  nolru: bool = False
  external_ptr: int|None = None

class MultiBuffer:
  def __init__(self, device:tuple[str, ...], size:int, dtype:DType):
    self.bufs = [Buffer(d, size, dtype) for d in device]
  @property
  def size(self): return self.bufs[0].size
  @property
  def dtype(self): return self.bufs[0].dtype
  def ref(self, cnt):
    for b in self.bufs: b.ref(cnt)
    return self
  def is_allocated(self): return all(x.is_allocated() for x in self.bufs)
  def __repr__(self): return f"<multibuf real:{self.is_allocated()} device:{tuple(x.device for x in self.bufs)} size:{self.size} dtype:{self.dtype}>"

class Buffer:
  profile_events:list[ProfileEvent] = []
  def __init__(self, device:str, size:int, dtype:DType, opaque:Any=None, options:BufferSpec|None=None, initial_value:bytes|None=None,
               uop_refcount=0, base:Buffer|None=None, offset:int=0, preallocate=False):
    assert isinstance(dtype, DType) and not isinstance(dtype, PtrDType)
    self.device, self.size, self.dtype, self.options, self.offset, self.allocated_views = device, size, dtype, options, offset, 0
    self._bufs: dict[str, Any] = {}
    if base is None:
      assert offset == 0, "base buffers can't have offset"
      self._base = None
      self._uop_refcount = uop_refcount
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
  def uop_refcount(self): return self.base._uop_refcount
  @property
  def _buf(self) -> Any: return self._bufs[self.device]
  def ref(self, cnt):
    self.base._uop_refcount += cnt
    return self
  # check if the underlying buffer is allocated and the current buffer/view is initialized
  def is_initialized(self) -> bool: return self.is_allocated() and self.device in self._bufs
  # check if the underlying buffer is allocated, possibly from the base object
  def is_allocated(self) -> bool: return self.base.is_allocated() if self._base is not None else self.device in self._bufs
  def get_buf(self, device: str) -> Any:
    if device not in self._bufs:
      allocator = Device[device].allocator
      if device == self.device: self.ensure_allocated()
      elif self._base is not None:
        assert hasattr(allocator, "_offset"), "offset function required for view"
        self._bufs[device] = allocator._offset(self._base.get_buf(device), self.nbytes, self.offset)
      else: self._bufs[device] = allocator._map(self.ensure_allocated()._buf)
    return self._bufs[device]
  def ensure_allocated(self) -> Buffer: return self.allocate() if not self.is_initialized() else self
  def allocate(self, opaque=None, external_ptr=None) -> Buffer:
    assert not self.is_initialized(), "can't allocate already allocated buffer"
    if DEBUG >= 7: print(f"buffer: allocate {self.nbytes} bytes on {self.device}")
    if not self.device.startswith("NULL") and self.size > MAX_BUFFER_SIZE > 0 and (self.options is None or self.options.external_ptr is None):
      raise RuntimeError(f"buffer of size {self.size/1e6:.2f}M is too large")
    self.allocator:Allocator = Device[self.device].allocator
    if external_ptr is not None:
      self.options = replace(self.options, external_ptr=external_ptr) if self.options else BufferSpec(external_ptr=external_ptr)
    if self._base is not None:
      self._base.ensure_allocated()
      self._base.allocated_views += 1
      assert hasattr(self.allocator, "_offset"), "offset function required for view"
      self._bufs[self.device] = self.allocator._offset(self.base._buf, self.nbytes, self.offset)
    else:
      self._bufs[self.device] = opaque if opaque is not None else self.allocator.alloc(self.nbytes, self.options)
      if not self.device.startswith("DISK") and (self.options is None or self.options.external_ptr is None):
        GlobalCounters.mem_used += self.nbytes
        GlobalCounters.mem_used_per_device[self.device] += self.nbytes
      if PROFILE: Buffer.profile_events.append(ProfilePointEvent(self.device, "alloc", self.trace_num, {"dtype":self.dtype, "sz":self.size}))
    return self
  def deallocate(self):
    assert self.device in self._bufs, "buffer must be allocated to deallocate"
    if DEBUG is not None and DEBUG >= 7: print(f"buffer: deallocate {self.nbytes} bytes on {self.device}")
    if self._base is None:
      if GlobalCounters is not None and not self.device.startswith("DISK") and (self.options is None or self.options.external_ptr is None):
        GlobalCounters.mem_used -= self.nbytes
        GlobalCounters.mem_used_per_device[self.device] -= self.nbytes
      if PROFILE: Buffer.profile_events.append(ProfilePointEvent(self.device, "free", self.trace_num))
      for dev, mb in self._bufs.items():
        if dev != self.device: Device[dev].allocator._unmap(mb)
      self.allocator.free(self._buf, self.nbytes, self.options)
    elif self._base is not None: self._base.allocated_views -= 1
    self._bufs.clear()
  def __reduce__(self):
    buf = None
    if self._base is not None:
      return self.__class__, (self.device, self.size, self.dtype, None, None, None, 0, self.base, self.offset, self.is_allocated())
    if self.device == "NPY": return self.__class__, (self.device, self.size, self.dtype, self._buf, self.options, None, self.uop_refcount)
    if self.is_allocated():
      buf = bytearray(self.nbytes)
      self.copyout(memoryview(buf))
    return self.__class__, (self.device, self.size, self.dtype, None, self.options, buf, self.uop_refcount)
  @property
  def trace_num(self) -> int:
    if not hasattr(self, '_trace_num'): self._trace_num = len(Buffer.profile_events)
    return self._trace_num
  @property
  def nbytes(self): return self.size*self.dtype.itemsize
  @suppress_finalizing
  def __del__(self): (self.device not in self._bufs) or self.deallocate()
  def __repr__(self):
    return f"<buf real:{self.is_allocated()} device:{self.device} size:{self.size} dtype:{self.dtype}" + \
           (f" offset:{self.offset}" if self._base is not None else "") + (f" {self.options=}" if self.options is not None else "") + ">"
  def as_memoryview(self, allow_zero_copy=False, force_zero_copy=False) -> memoryview:
    # zero copy with as_memoryview (disabled by default due to use after free)
    if (force_zero_copy or allow_zero_copy) and hasattr(self.allocator, '_as_buffer') and self.options is None:
      return self.allocator._as_buffer(self._buf)
    assert not force_zero_copy, "force zero copy was passed, but copy is required"
    return self.copyout(memoryview(bytearray(self.nbytes)))
  def numpy(self) -> 'np.ndarray': # type: ignore [name-defined] # noqa: F821
    import numpy as np
    assert _to_np_dtype(self.dtype.base) is not None, f"no np dtype for {self.dtype.base}"
    return np.frombuffer(self.as_memoryview(), dtype=_to_np_dtype(self.dtype.base))
  def copyin(self, mv:memoryview):
    mv = flat_mv(mv)
    assert len(mv) == self.nbytes, f"size mismatch, {len(mv)=} != {self.dtype=} {self.size=}"
    assert self.is_initialized(), "can't copyin to unallocated buffer"
    self.allocator._copyin(self._buf, mv)
    return self
  def copyout(self, mv:memoryview) -> memoryview:
    mv = flat_mv(mv)
    assert len(mv) == self.nbytes, f"size mismatch, {len(mv)=} != {self.dtype=} {self.size=}"
    assert self.is_initialized(), "can't copyout unallocated buffer"
    self.allocator._copyout(mv, self._buf)
    return mv
  def view(self, size:int, dtype:DType, offset:int) -> Buffer:
    assert offset < self.nbytes, "offset must be less than nbytes"
    return Buffer(self.device, size, dtype, base=self.base, offset=self.offset+offset)

DeviceType = TypeVar('DeviceType', bound='Compiled')

# TODO: size, dest, src are the same type. can we enforce this?
class Allocator(Generic[DeviceType]):
  def __init__(self, dev:DeviceType, supports_copy_from_disk:bool=True, supports_transfer:bool=True):
    self.dev: DeviceType = dev
    self.default_buffer_spec: BufferSpec = BufferSpec()
    self.supports_copy_from_disk, self.supports_transfer = supports_copy_from_disk, supports_transfer
  # overridden in LRUAllocator
  def alloc(self, size:int, options:BufferSpec|None=None):
    assert size > 0, f"alloc size must be positive, getting {size}"
    try: return self._alloc(size, options if options is not None else self.default_buffer_spec)
    except (RuntimeError, MemoryError) as e: raise MemoryError(f"Allocation of {size_to_str(size)} failed on {self.dev.device}. "
                                                               f"Used: {size_to_str(GlobalCounters.mem_used_per_device[self.dev.device])}") from e
  def free(self, opaque, size:int, options:BufferSpec|None=None):
    self._free(opaque, options if options is not None else self.default_buffer_spec)

  # implemented by the runtime
  def _alloc(self, size:int, options:BufferSpec): raise NotImplementedError("need alloc")
  def _free(self, opaque, options:BufferSpec): pass  # if opaque is a Python object, you don't need a free
  def _copyin(self, dest, src:memoryview): raise NotImplementedError("need copyin")
  def _copyout(self, dest:memoryview, src): raise NotImplementedError("need copyout")
  def _map(self, buf): raise NotImplementedError("need map")
  def _unmap(self, mb): pass  # default no-op; override if _map allocates iface-side state
  # def _as_buffer(self, src) -> memoryview:
  # def _offset(self, buf, size:int, offset:int):
  # def _transfer(self, dest, src, sz:int, src_dev, dest_dev):
  def _encode_decode(self, bufout, bufin, desc, hist:list, shape:tuple[int,...], frame_pos:int): raise NotImplementedError("need encdec") # optional

class LRUAllocator(Allocator, Generic[DeviceType]):
  """
  The LRU Allocator is responsible for caching buffers.
  It ensures that buffers are not freed until it is absolutely necessary, optimizing performance.
  """
  def __init__(self, dev:DeviceType, **kwargs):
    self.cache: dict[tuple[int, BufferSpec|None], Any] = defaultdict(list)
    super().__init__(dev, **kwargs)
  def alloc(self, size:int, options:BufferSpec|None=None):
    if len(c := self.cache[(size, options)]): return c.pop()
    try: return super().alloc(size, options)
    except (RuntimeError, MemoryError):
      self.free_cache()
      return super().alloc(size, options)
  def free_cache(self):
    for (sz,options),opaques in self.cache.items():
      for opaque in opaques: super().free(opaque, sz, options)
      opaques.clear()
  def free(self, opaque:Any, size:int, options:BufferSpec|None=None):
    if LRU and (options is None or (not options.nolru and options.external_ptr is None)): self.cache[(size, options)].append(opaque)
    else: super().free(opaque, size, options)

# **************** for Compiled Devices ****************

class CompileError(Exception): pass

class Compiler:
  def __init__(self, cachekey:str|None=None): self.cachekey = cachekey if CCACHE else None
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

  def __init__(self, device:str, allocator:Allocator, renderers:list[type[Renderer]], runtime, graph=None, arch=None):
    from tinygrad.renderer import Renderer
    self.device, self.allocator, self.runtime, self.graph, self.renderers = device, allocator, runtime, graph, renderers or [Renderer]
    self.arch = arch
    self.cached_renderer:dict[Any, Renderer] = {}

  @property
  def renderer(self) -> Renderer: return self._select_renderer()

  @property
  def compiler(self) -> Compiler:
    if (ret:=self.renderer.compiler) is None: raise RuntimeError(f"no compiler for {self.device}")
    return ret

  def _renderer_name(self, r:type[Renderer]) -> str:
    return r.__name__.upper().removesuffix("RENDERER").removeprefix(devname:=self.device.split(':')[0].upper()) or devname

  def _select_renderer(self) -> Renderer:
    assert (rn:=next((self._renderer_name(r) for r in self.renderers if getenv(f"{self.device}_{self._renderer_name(r)}")), None)) is None, \
      f"{self.device}_{rn}=1 is deprecated, use DEV={self.device}:{rn} or {self.device}_CC={rn} instead"
    t = DEV.target(self.device.split(':')[0], **({"arch":self.arch} if self.arch else {}))
    return select_first_inited(select_by_name(self.renderers, self._renderer_name, t.renderer, f"{self.device} has no renderer {t.renderer!r}"),
                               f"No renderer for {self.device} is available", self.cached_renderer, t)

  def count(self) -> int:
    """
    Returns the number of physical accelerators available to the runtime.
    """
    return 1

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

if PROFILE:
  @atexit.register
  def finalize_profile():
    devs = [Device[d] for d in Device._opened_devices]
    for dev in devs: dev.synchronize()
    for dev in devs: dev._at_profile_finalize()

    with open(fn:=temp("profile.pkl", append_user=True), "wb") as f: pickle.dump(cpu_events+Compiled.profile_events+Buffer.profile_events, f)

    PROFILE.value = 0
    from tinygrad.uop.ops import launch_viz
    launch_viz("PROFILE", fn)

def enumerate_devices_str() -> Generator[str, None, None]:
  from tinygrad import Tensor, Device

  for device in ALL_DEVICES:
    ren_results, iface_results = [], []
    try:
      d = Device[device]
      for iface in [i for i in getattr(d, 'ifaces', []) if not i.__name__.startswith("MOCK")]:
        try:
          name = iface.__name__[:-5]
          default_text, count = ("(default)", d.count()) if type(d.iface) is iface else (f"(DEV={name}+{device} to make default)", iface(d, 0).count) # type: ignore
          iface_results.append(f"{colored('+', 'green')} {name}: {pluralize('device', count)} {default_text}")
        except Exception as e: iface_results.append(f"{colored('-', 'red')} {iface.__name__[:-5]}: {e}")
      for r in d.renderers:
        try:
          with Context(CACHELEVEL=0, DEV=f"{device}:{d._renderer_name(r)}"): test = (Tensor([1,2,3], device=device) * 2).tolist()
          if test != [2,4,6]: raise ValueError(f"got {test} instead of [2, 4, 6]")
          default_text = '(default)' if type(d.renderer) is r else f'(DEV={device}:{d._renderer_name(r)} to make default)'
          ren_results.append(f"{colored('+', 'green')} {d._renderer_name(r)} {default_text}")
        except Exception as e: ren_results.append(f"{colored('-', 'red')} {d._renderer_name(r)}: {e}")
      result = (colored('PASS', 'green') + ("\n"+" "*12+"interfaces:\n" if iface_results else "") + '\n'.join([" "*13+x for x in iface_results]) +
                (("\n"+" "*12+"renderers:\n") + '\n'.join([" "*13+x for x in ren_results]) if len(ren_results) > 1 else ""))
    except Exception as e: result = f"{colored('FAIL', 'red')} {e}"
    yield f"{'*' if device == Device.DEFAULT else ' '} {device:8s}: {result}"

if __name__ == "__main__":
  for s in enumerate_devices_str(): print(s)
