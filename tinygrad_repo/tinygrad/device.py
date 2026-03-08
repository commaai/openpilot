from __future__ import annotations
from dataclasses import dataclass, replace
from collections import defaultdict
from typing import Any, Generic, TypeVar, Iterator, Generator
import importlib, inspect, functools, pathlib, os, platform, contextlib, sys, re, atexit, pickle, decimal
from tinygrad.helpers import CI, OSX, LRU, getenv, diskcache_get, diskcache_put, DEBUG, GlobalCounters, flat_mv, PROFILE, temp, colored
from tinygrad.helpers import Context, CCACHE, ALLOW_DEVICE_USAGE, MAX_BUFFER_SIZE, cpu_events, ProfileEvent, ProfilePointEvent, dedup, ContextVar
from tinygrad.helpers import unwrap_class_type, suppress_finalizing, select_first_inited, VIZ, CPU_LLVM, CPU_LVP, NV_PTX, CUDA_PTX, NV_NAK
from tinygrad.helpers import EMULATED_DTYPES
from tinygrad.dtype import DType, ImageDType, PtrDType, dtypes, _to_np_dtype
from tinygrad.renderer import Renderer

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
  @functools.cached_property
  def DEFAULT(self) -> str:
    dev = [dev] if (dev:=getenv("DEV", "").upper()) else []
    from_env = dedup(dev + [d for d in self._devices if d not in ["DISK", "TINYFS", "NPY"] and getenv(d) == 1])
    assert len(from_env) < 2, f"multiple devices set in env: {from_env}"
    if len(from_env) == 1: return from_env[0]
    try:
      device = next(self.get_available_devices())
      os.environ[device] = "1"   # we set this in environment for spawned children
      return device
    except StopIteration as exc: raise RuntimeError("no usable devices") from exc
Device: _Device = _Device()
atexit.register(lambda: [Device[dn].finalize() for dn in Device._opened_devices])

# **************** Profile ****************

@dataclass(frozen=True)
class ProfileDeviceEvent(ProfileEvent):
  device:str; comp_tdiff:decimal.Decimal=decimal.Decimal(0); copy_tdiff:decimal.Decimal=decimal.Decimal(0); props:dict[str,Any]|None=None # noqa: E702

@dataclass(frozen=True)
class ProfileProgramEvent(ProfileEvent): device:str; name:str; lib:bytes|None; base:int|None; tag:int|None=None # noqa: E702

@dataclass(frozen=True)
class ProfileGraphEntry: device:str; name:str; st_id:int; en_id:int; is_copy:bool # noqa: E702

@dataclass(frozen=True)
class ProfileGraphEvent(ProfileEvent): ents:list[ProfileGraphEntry]; deps:list[list[int]]; sigs:list[decimal.Decimal] # noqa: E702

# **************** Buffer + Allocators ****************

@dataclass(frozen=True, eq=True)
class BufferSpec:
  # TODO: move device, size, dtype here?
  image: ImageDType|None = None
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
    if isinstance(dtype, ImageDType): options = BufferSpec(image=dtype) # TODO: image hack shouldn't be here. where should it be?
    else: assert isinstance(dtype, DType) and not isinstance(dtype, PtrDType)
    self.device, self.size, self.dtype, self.options, self.offset, self.allocated_views = device, size, dtype, options, offset, 0
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
  def ref(self, cnt):
    self.base._uop_refcount += cnt
    return self
  # check if the underlying buffer is allocated and the current buffer/view is initialized
  def is_initialized(self) -> bool: return self.is_allocated() and hasattr(self, '_buf')
  # check if the underlying buffer is allocated, possibly from the base object
  def is_allocated(self) -> bool: return self.base.is_allocated() if self._base is not None else hasattr(self, '_buf')
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
      self._buf: Any = self.allocator._offset(self.base._buf, self.nbytes, self.offset)
    else:
      self._buf = opaque if opaque is not None else self.allocator.alloc(self.nbytes, self.options)
      if not self.device.startswith("DISK") and (self.options is None or self.options.external_ptr is None):
        GlobalCounters.mem_used += self.nbytes
      if PROFILE: Buffer.profile_events.append(ProfilePointEvent(self.device, "alloc", self.trace_num, {"dtype":self.dtype, "sz":self.size}))
    return self
  def deallocate(self):
    assert hasattr(self, '_buf'), "buffer must be allocated to deallocate"
    if DEBUG is not None and DEBUG >= 7: print(f"buffer: deallocate {self.nbytes} bytes on {self.device}")
    if self._base is None:
      if GlobalCounters is not None and not self.device.startswith("DISK") and (self.options is None or self.options.external_ptr is None):
        GlobalCounters.mem_used -= self.nbytes
      if PROFILE: Buffer.profile_events.append(ProfilePointEvent(self.device, "free", self.trace_num))
      self.allocator.free(self._buf, self.nbytes, self.options)
    elif self._base is not None: self._base.allocated_views -= 1
    del self._buf
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
  def __del__(self): (not hasattr(self, '_buf')) or self.deallocate()
  def __repr__(self):
    return f"<buf real:{self.is_allocated()} device:{self.device} size:{self.size} dtype:{self.dtype}" + \
           (f" offset:{self.offset}" if self._base is not None else "") + (f" {self.options=}" if self.options is not None else "") + ">"
  def as_dmaref(self) -> DMARef:
    assert hasattr(self.allocator, "_as_dmaref"), f"Device {self.device} doesn't support DMA"
    return self.allocator._as_dmaref(self._buf)
  def as_buffer(self, allow_zero_copy=False, force_zero_copy=False) -> memoryview:
    # zero copy with as_buffer (disabled by default due to use after free)
    if (force_zero_copy or allow_zero_copy) and hasattr(self.allocator, '_as_buffer') and (self.options is None or self.options.image is None):
      return self.allocator._as_buffer(self._buf)
    assert not force_zero_copy, "force zero copy was passed, but copy is required"
    return self.copyout(memoryview(bytearray(self.nbytes)))
  def as_typed_buffer(self, shape=None, allow_zero_copy=False, force_zero_copy=False) -> memoryview:
    assert self.dtype.base.fmt is not None, f"no fmt dtype for {self.dtype.base}"
    assert self.dtype.base.fmt != "e" or sys.version_info >= (3, 12)
    return self.as_buffer(allow_zero_copy, force_zero_copy).cast(self.dtype.base.fmt, shape if shape is not None else (self.size,))
  def numpy(self) -> 'np.ndarray': # type: ignore [name-defined] # noqa: F821
    import numpy as np
    assert _to_np_dtype(self.dtype.base) is not None, f"no np dtype for {self.dtype.base}"
    return np.frombuffer(self.as_buffer(), dtype=_to_np_dtype(self.dtype.base))
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

@dataclass(frozen=True)
class DMACPURef:
  addr: int
  size: int

@dataclass(frozen=True)
class DMAFdRef:
  fd: int
  offset: int
  size: int

DMARef = DMACPURef|DMAFdRef

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
    return self._alloc(size, options if options is not None else self.default_buffer_spec)
  def free(self, opaque, size:int, options:BufferSpec|None=None):
    self._free(opaque, options if options is not None else self.default_buffer_spec)

  # implemented by the runtime
  def _alloc(self, size:int, options:BufferSpec): raise NotImplementedError("need alloc")
  def _free(self, opaque, options:BufferSpec): pass  # if opaque is a Python object, you don't need a free
  def _copyin(self, dest, src:memoryview): raise NotImplementedError("need copyin")
  def _copyout(self, dest:memoryview, src): raise NotImplementedError("need copyout")
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

@dataclass(frozen=True)
class CompilerPair: renderer:type[Renderer]|functools.partial; compiler:type[Compiler]|functools.partial|None; ctrl_var:ContextVar|None = None # noqa: E702

@dataclass(frozen=True)
class CompilerSet: cset:list[CompilerPair]; ctrl_var:ContextVar|None = None # noqa: E702

class Compiled:
  profile_events:list[ProfileEvent] = [ProfileDeviceEvent("CPU")] # NOTE: CPU is the default device.

  def __init__(self, device:str, allocator:Allocator, compilers:CompilerSet|None, runtime, graph=None, group_id=None):
    self.device, self.allocator, self.runtime, self.graph, self.group_id = device, allocator, runtime, graph, group_id

    self.comps_ctrl_var = compilers.ctrl_var if compilers is not None else None
    self.comp_sets:dict[Any, tuple[ContextVar|None, tuple[type[Renderer]|functools.partial, type[Compiler]|functools.partial|None]]] = {}
    self.cached_pair:dict[Any, tuple[Renderer, Compiler|None]] = {}
    for cpair in (compilers.cset if compilers is not None else [CompilerPair(Renderer, Compiler)]):
      self.comp_sets[self._compiler_name(cpair.renderer, cpair.compiler)] = (cpair.ctrl_var, (cpair.renderer, cpair.compiler))

  @property
  def renderer(self) -> Renderer: return self._select_compiler_pair()[0]

  @property
  def compiler(self) -> Compiler:
    if (ret:=self.renderer.compiler or self._select_compiler_pair()[1]) is None: raise RuntimeError(f"no compiler for {self.device}")
    return ret

  def _compiler_name(self, r:type[Renderer]|functools.partial, c:type[Compiler]|functools.partial|None) -> str:
    devname = self.device.split(':')[0].upper()
    if c is None: return unwrap_class_type(r).__name__.upper().removesuffix("RENDERER").removeprefix(devname) or devname
    return unwrap_class_type(c).__name__.upper().removesuffix("COMPILER").removeprefix(devname) or devname

  def _select_compiler_pair(self) -> tuple[Renderer, Compiler|None]:
    # select forced compiler from global env var.
    forced_comps = set([self.comp_sets[val][1]] if self.comps_ctrl_var is not None and (val:=self.comps_ctrl_var.value) else [])

    # add forced compilers from individual env vars (only if global env var is not set, as it takes precedence).
    if not forced_comps: forced_comps |= set(rc for en, rc in self.comp_sets.values() if en is not None and en.value == 1)
    if len(forced_comps) > 1: raise RuntimeError(f"{self.device}: multiple compilers set in env {forced_comps}")

    # select remaining compilers (all or forced only)
    comps = list(rc for en, rc in self.comp_sets.values())

    # remove disabled compilers
    for en, rc in self.comp_sets.values():
      if en is not None and en.value == 0 and rc in comps: comps.remove(rc)

    return select_first_inited(list(forced_comps) if len(forced_comps)>0 else comps, f"No compiler for {self.device} is available", self.cached_pair)

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
def is_dtype_supported(dtype:DType, device:str|None=None) -> bool:
  if dtype == dtypes.index: return False
  if device is None: device = Device.DEFAULT
  if dtype == dtypes.bfloat16:
    if device == "METAL": return not CI
    if device == "CUDA": return not CI and not CUDA_PTX
    if device == "NV": return not CI and not NV_PTX and not NV_NAK
    if device in {"CPU"}: return not CI and platform.machine() in {"arm", "arm64", "aarch64", "x86_64", "amd64"} and not CPU_LVP
    return device in {"AMD", "CL", "PYTHON", "NULL"}
  if dtype in dtypes.fp8s:
    if device == "CUDA": return not CI and not CUDA_PTX
    if device == "NV": return not CI and not NV_PTX and not NV_NAK
    if device == "AMD": return not CI and getattr(Device["AMD"], "target") in {(9,4,2), (9,5,0)}
    return device in {"PYTHON", "NULL"}
  if device == "WEBGPU": return dtype in [dtypes.bool, dtypes.char, dtypes.uchar, dtypes.short,
                                          dtypes.ushort, dtypes.float, dtypes.int32, dtypes.uint32, dtypes.half]
  # for CI GPU and OSX, cl_khr_fp16 isn't supported
  # for CI LLVM, it segfaults because it can't link to the casting function
  # CI CUDA architecture is sm_35 but we need at least sm_70 to run fp16 ALUs
  # PYTHON supports half memoryview in 3.12+ https://github.com/python/cpython/issues/90751
  # double can't be bitcast to anything without long support
  if dtype == dtypes.half:
    if device == "CL": return not CI and not OSX
    if device == "QCOM": return False # QCOM compiler is flaky with half
    if device in ["CUDA", "NV"]: return not CI
    if device == "CPU" and CPU_LLVM: return OSX
    if device == "PYTHON": return sys.version_info >= (3, 12)
  if dtype == dtypes.float64: return (device not in {"METAL", "QCOM"} and not (OSX and device == "CL") and not getenv("NULL_IR3")
                                      and dtypes.long not in EMULATED_DTYPES.tolist(dtypes))
  return True

if PROFILE:
  @atexit.register
  def finalize_profile():
    devs = [Device[d] for d in Device._opened_devices]
    for dev in devs: dev.synchronize()
    for dev in devs: dev._at_profile_finalize()

    with open(fn:=temp("profile.pkl", append_user=True), "wb") as f: pickle.dump(cpu_events+Compiled.profile_events+Buffer.profile_events, f)

    if VIZ:
      from tinygrad.uop.ops import launch_viz
      launch_viz("PROFILE", fn)

def enumerate_devices_str() -> Generator[str, None, None]:
  from tinygrad import Tensor, Device

  for device in ALL_DEVICES:
    compilers_results, any_works = [], False
    try:
      d = Device[device]
      default_comp_pairs, default_compiler, cc_ctrl_var = d.comp_sets, d.compiler, d.comps_ctrl_var
      try:
        for k,(en,(r,c)) in default_comp_pairs.items():
          d.comp_sets = {k:(None,(r,c))} # env var set to None, so it doesn't interfere
          d.comps_ctrl_var = None
          try:
            # d.renderer, d.compiler = r(), c()
            with Context(CACHELEVEL=0): test = (Tensor([1,2,3], device=device) * 2).tolist()
            if test != [2,4,6]: raise ValueError(f"got {test} instead of [2, 4, 6]")
            set_text = f'({cc_ctrl_var.key}={d._compiler_name(r, c)} to make default)' if cc_ctrl_var is not None else ''
            default_text = '(default)' if type(default_compiler) is type(d.compiler) else set_text
            compilers_results.append(f"{colored('+', 'green')} {unwrap_class_type(c).__name__} {default_text}")
            any_works = True
          except Exception as e: compilers_results.append(f"{colored('-', 'yellow')} {unwrap_class_type(c).__name__}: {e}")
      finally:
        # put the defaults back!
        d.comp_sets, d.comps_ctrl_var = default_comp_pairs, cc_ctrl_var
      result = (colored('PASS', 'green') if any_works else f"{colored('FAIL', 'yellow')}") + ''.join([f'\n{" "*16} {x}' for x in compilers_results])
    except Exception as e:
      result = f"{colored('FAIL', 'red')} {e}"
    yield f"{'*' if device == Device.DEFAULT else ' '} {device:10s}: {result}"

if __name__ == "__main__":
  for s in enumerate_devices_str(): print(s)
