from __future__ import annotations
from dataclasses import dataclass, replace, field
from collections import defaultdict
from typing import Any, Callable, Generic, TypeVar, Iterator, Generator, Self, TYPE_CHECKING
import importlib, inspect, functools, pathlib, os, contextlib, re, atexit, pickle, decimal, subprocess, struct, mmap, time, statistics
from tinygrad.helpers import mv_address, LRU, getenv, diskcache_get, diskcache_put, DEBUG, GlobalCounters, PROFILE, temp, colored
from tinygrad.helpers import Context, CCACHE, ALLOW_DEVICE_USAGE, MAX_BUFFER_SIZE, cpu_events, ProfileEvent, ProfilePointEvent, suppress_finalizing
from tinygrad.helpers import select_by_name, select_first_inited, DEV, TracingKey, size_to_str, pluralize, Target, unwrap, round_up, is_numpy_ndarray
from tinygrad.helpers import cpu_profile, perf_counter_us, ContextVar
from tinygrad.dtype import dtypes, DType, _to_np_dtype
from tinygrad.runtime.support.memory import BumpAllocator, MMIOInterface
if TYPE_CHECKING:
  from tinygrad.renderer import Renderer
  from tinygrad.uop.ops import UOp

# **************** Device ****************

HCQ_RUNTIME_DEV = ContextVar("HCQ_RUNTIME_DEV", "PYTHON" if DEV.interface.startswith("MOCK") else "CPU")

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
    assert ALLOW_DEVICE_USAGE or ix.split(":")[0] in ["DISK", "NPY", "PYTHON"], f"usage of device {ix} disallowed"
    return self.__get_canonicalized_item(ix)
  @functools.cache  # this class is a singleton, pylint: disable=method-cache-max-size-none
  def get_class(self, ix:str):
    base = (__package__ or __name__).split('.')[0]  # tinygrad
    x = ix.split(":")[0].lower()
    return [cls for cname, cls in inspect.getmembers(importlib.import_module(f'{base}.runtime.ops_{x}')) if (cname.lower() == x + "device")][0]
  @functools.cache  # this class is a singleton, pylint: disable=method-cache-max-size-none
  def __get_canonicalized_item(self, ix:str) -> Compiled:
    ret = self.get_class(ix)(ix)
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
    assert (dev:=next((d for d in self._devices if d not in ["DISK", "NPY"] and getenv(d) == 1), None)) is None, \
      f"{dev}=1 is deprecated, use DEV={dev} instead"
    try:
      device = next(self.get_available_devices())
      os.environ["DEV"] = device   # we set this in environment for spawned children
      return device
    except StopIteration as exc: raise RuntimeError("no usable devices") from exc
Device: _Device = _Device()
atexit.register(lambda: [Device[dn].finalize() for dn in tuple(Device._opened_devices)])

def canonicalize_device(device:str|tuple|list|None) -> str|tuple[str, ...]:
  if not isinstance(device, (tuple, list)): return Device.canonicalize(device)
  return canonical[0] if len(canonical:=tuple(Device.canonicalize(d) for d in device)) == 1 else canonical

# **************** Profile ****************

@dataclass(frozen=True)
class ProfileDeviceEvent(ProfileEvent): device:str; tdiff:decimal.Decimal=decimal.Decimal(0); props:dict[str,Any]|None=None # noqa: E702

@dataclass(frozen=True)
class ProfileProgramEvent(ProfileEvent): device:str; name:str; lib:bytes|None; base:int|None; tag:int|None=None; profile_key:bytes|None=None # noqa: E702

@dataclass(frozen=True)
class ProfileGraphEntry: device:str; name:str|TracingKey; st_id:int; en_id:int; profile_key:bytes|None=None # noqa: E702

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
  zero: bool = False
  external_ptr: int|None = None

class MultiBuffer:
  def __init__(self, device:tuple[str, ...], size:int, dtype:DType):
    self.bufs = [Buffer(d, size, dtype) for d in device]
  @property
  def size(self): return self.bufs[0].size
  @property
  def dtype(self): return self.bufs[0].dtype
  def is_allocated(self): return all(x.is_allocated() for x in self.bufs)
  def __repr__(self): return f"<multibuf real:{self.is_allocated()} device:{tuple(x.device for x in self.bufs)} size:{self.size} dtype:{self.dtype}>"

@dataclass(frozen=True)
class BufferStorage: buf:Any; meta:Any=None; host:MMIOInterface|None=None; maps:dict[Compiled, BufferStorage]=field(default_factory=dict) # noqa: E702

class Buffer:
  profile_events:list[ProfileEvent] = []
  def __init__(self, device:str, size:int, dtype:DType, opaque:Any=None, options:BufferSpec|None=None,
               initial_value:bytes|pickle.PickleBuffer|None=None, base:Buffer|None=None, offset:int=0, preallocate=False,
               allocator:Allocator|None=None):
    assert isinstance(dtype, DType)
    self.device, self.size, self.dtype, self.offset, self.allocated_views, self._base = Device.canonicalize(device), size, dtype, offset, 0, base
    if allocator is not None: self.allocator = allocator
    self.options = options if options is not None else BufferSpec()
    self._storage:BufferStorage|None = None
    if base is None:
      assert offset == 0, "base buffers can't have offset"
      if opaque is not None: self.allocate(opaque)
      if initial_value is not None:
        self.allocate()
        if (host:=self.get_storage().host) is not None: host[:] = memoryview(initial_value).cast('B')
        else: self.copy_from(Buffer("PYTHON", self.size, self.dtype, opaque=memoryview(bytearray(initial_value))))
        if isinstance(initial_value, pickle.PickleBuffer): initial_value.release()
    else:
      assert base._base is None, "base can't have a base"
      assert self.device == base.device, "base must have the same device"
    if preallocate: self.allocate()

  @suppress_finalizing
  def __del__(self): self._storage is None or self.deallocate()

  def __repr__(self):
    return f"<buf real:{self.is_allocated()} device:{self.device} size:{self.size} dtype:{self.dtype}" + \
           (f" offset:{self.offset}" if self._base is not None else "") + (f" {self.options=}" if self.options != BufferSpec() else "") + ">"

  @property
  def base(self) -> Buffer: return self._base if self._base is not None else self
  @functools.cached_property
  def allocator(self) -> Allocator: return self.base.allocator if self._base is not None else Device[self.device].allocator
  @property
  def _buf(self) -> Any: return self.get_storage().buf
  @property
  def host(self) -> MMIOInterface: return unwrap(self.get_storage().host)
  @property
  def meta(self) -> Any: return self.get_storage().meta
  @property
  def nbytes(self): return self.size * self.dtype.itemsize

  def get_storage(self, device:str|None=None) -> BufferStorage:
    storage = unwrap(self.ensure_allocated()._storage)
    device = Device.canonicalize(device) if device is not None else self.device
    if device == self.device: return storage
    if (dev:=Device[device]) not in storage.maps:
      alloc = dev.allocator
      storage.maps[dev] = BufferStorage(alloc._offset(self.base.get_buf(device), self.nbytes, self.offset)) if self._base else alloc.map(self)
    if storage.maps[dev].host is not storage.host: storage.maps[dev] = replace(storage.maps[dev], host=storage.host)
    return storage.maps[dev]

  def get_buf(self, device:str) -> Any: return self.get_storage(device).buf

  def is_allocated(self) -> bool: return self._storage is not None and (self._base is None or self._base_storage is self.base._storage)
  def ensure_allocated(self) -> Buffer: return self.allocate() if not self.is_allocated() else self
  def allocate(self, opaque=None, external_ptr=None) -> Buffer:
    assert not self.is_allocated(), "can't allocate already allocated buffer"
    if DEBUG >= 7: print(f"buffer: allocate {self.nbytes} bytes on {self.device}")
    if not self.device.startswith("NULL") and self.size > MAX_BUFFER_SIZE > 0 and self.options.external_ptr is None:
      raise RuntimeError(f"buffer of size {self.size/1e6:.2f}M is too large")
    if external_ptr is not None: self.options = replace(self.options, external_ptr=external_ptr)
    if self._base is not None:
      storage = replace(self.base.get_storage(), buf=self.allocator._offset(self.base._buf, self.nbytes, self.offset), maps={})
    elif opaque is not None:
      self.options = replace(self.options, nolru=True)
      if is_numpy_ndarray(opaque):
        if not opaque.flags.c_contiguous: opaque = opaque.copy(order='C')
        opaque = BufferStorage(addr:=opaque.ctypes.data, memoryview(opaque), MMIOInterface(addr, self.nbytes))
      elif isinstance(opaque, memoryview):
        opaque = BufferStorage(addr:=mv_address(opaque) if self.nbytes else 0, opaque, MMIOInterface(addr, self.nbytes))
      storage = opaque if isinstance(opaque, BufferStorage) else BufferStorage(opaque)
    else: storage = self.allocator.alloc(self.nbytes, self.options)
    storage = replace(storage, host=storage.host.view(self.offset, self.nbytes, fmt='B') if storage.host is not None else None)
    if self._base is None:
      if not self.device.startswith("DISK") and self.options.external_ptr is None:
        GlobalCounters.mem_used += self.nbytes
        GlobalCounters.mem_used_per_device[self.device] += self.nbytes
      if PROFILE: Buffer.profile_events.append(ProfilePointEvent(self.device, "alloc", self.trace_num, {"dtype":self.dtype, "sz":self.size}))
    elif self._storage is None: self.base.allocated_views += 1
    self._storage, self._base_storage = storage, self.base._storage if self._base else None
    return self

  def deallocate(self):
    assert self._storage is not None, "buffer must be allocated to deallocate"
    if DEBUG is not None and DEBUG >= 7: print(f"buffer: deallocate {self.nbytes} bytes on {self.device}")
    if self._base is None:
      if GlobalCounters is not None and not self.device.startswith("DISK") and self.options.external_ptr is None:
        GlobalCounters.mem_used -= self.nbytes
        GlobalCounters.mem_used_per_device[self.device] -= self.nbytes
      if PROFILE: Buffer.profile_events.append(ProfilePointEvent(self.device, "free", self.trace_num))
      self.allocator.free(self._storage, self.nbytes, self.options)
    else: self.base.allocated_views -= 1
    self._storage, self._base_storage = None, None

  def __reduce_ex__(self, protocol):
    buf:bytearray|pickle.PickleBuffer|None = None
    if self._base is not None:
      return self.__class__, (self.device, self.size, self.dtype, None, None, None, self.base, self.offset, self.is_allocated())
    if self.device == "NPY":
      import numpy as np
      arr = np.frombuffer(self.meta, _to_np_dtype(self.dtype)) # over the storage itself, so an out-of-band pickle buffer keeps it alive
      return self.__class__, (self.device, self.size, self.dtype, arr, self.options, None)
    if self.is_allocated():
      buf = pickle.PickleBuffer(self.as_memoryview()) if protocol >= 5 else bytearray(self.as_memoryview())
    return self.__class__, (self.device, self.size, self.dtype, None, self.options, buf)

  @property
  def trace_num(self) -> int:
    if not hasattr(self, '_trace_num'): self._trace_num = len(Buffer.profile_events)
    return self._trace_num

  def _host_mv(self) -> memoryview|None:
    if self.is_allocated() and hasattr(host:=self.get_storage().host, 'mv'): return unwrap(host).view(fmt='B').mv
    if self.is_allocated() and hasattr(self.allocator, '_as_buffer'): return self.allocator._as_buffer(self._buf)
    return None

  def as_memoryview(self, allow_zero_copy=False) -> memoryview:
    if (mv:=self._host_mv()) is not None:
      self.allocator.dev.synchronize()
      if allow_zero_copy: return mv
      with cpu_profile(f"{self.device} -> TINY", f"{self.device}:COPY"): return memoryview(bytearray(mv))
    Buffer("PYTHON", self.size, self.dtype, opaque=(mv:=memoryview(bytearray(self.nbytes)))).copy_from(self)
    return mv

  def numpy(self) -> 'np.ndarray': # type: ignore [name-defined] # noqa: F821
    import numpy as np
    assert _to_np_dtype(self.dtype) is not None, f"no np dtype for {self.dtype}"
    return np.frombuffer(self.as_memoryview(), dtype=_to_np_dtype(self.dtype))

  def copy_from(self, src:Buffer) -> Buffer:
    assert self.nbytes == src.nbytes, f"copy size mismatch, {self.nbytes} != {src.nbytes}"
    assert self.is_allocated() and src.is_allocated(), "copy requires allocated buffers"
    from tinygrad.engine.realize import run_linear
    from tinygrad.uop.ops import UOp, Ops
    du, su = UOp.from_buffer(self), UOp.from_buffer(src)
    run_linear(UOp(Ops.LINEAR, src=(su.param_like(1).copy_to_device(self.device).call(du, su),)), update_stats=False)
    return self

  def view(self, size:int, dtype:DType, offset:int) -> Buffer:
    assert offset < self.nbytes, "offset must be less than nbytes"
    return Buffer(self.device, size, dtype, base=self.base, offset=self.offset+offset)

DeviceType = TypeVar('DeviceType', bound='Compiled')

# TODO: size, dest, src are the same type. can we enforce this?
class Allocator(Generic[DeviceType]):
  lru = True

  def __init__(self, dev:DeviceType, supports_copy_from_disk:bool=True, supports_transfer:bool=True):
    self.dev: DeviceType = dev
    self.default_buffer_spec: BufferSpec = BufferSpec()
    self.cache:dict[tuple[int, BufferSpec|None], list[BufferStorage]] = defaultdict(list)
    self.supports_copy_from_disk, self.supports_transfer = supports_copy_from_disk, supports_transfer

  def alloc(self, size:int, options:BufferSpec|None=None) -> BufferStorage:
    assert size > 0, f"alloc size must be positive, getting {size}"
    if len(c:=self.cache[(size, options)]): return c.pop()
    spec = options if options is not None else self.default_buffer_spec
    try: return self._alloc(size, spec)
    except (RuntimeError, MemoryError): self.free_cache()
    try: return self._alloc(size, spec)
    except (RuntimeError, MemoryError) as e: raise MemoryError(f"Allocation of {size_to_str(size)} failed on {self.dev.device}. "
                                                            f"Used: {size_to_str(GlobalCounters.mem_used_per_device[self.dev.device])}") from e

  def free(self, storage:BufferStorage, size:int, options:BufferSpec|None=None):
    spec = options if options is not None else self.default_buffer_spec
    if LRU and self.lru and not (spec.nolru or spec.zero) and spec.external_ptr is None: self.cache[(size, options)].append(storage)
    else: self.do_free(storage, spec)

  def free_cache(self):
    for (_, options), storages in self.cache.items():
      for storage in storages: self.do_free(storage, options if options is not None else self.default_buffer_spec)
      storages.clear()

  def do_free(self, storage:BufferStorage, options:BufferSpec):
    for dev in storage.maps: dev.synchronize()
    for dev, mb in storage.maps.items(): dev.allocator._unmap(mb)
    if options.external_ptr is None: self._free(storage, options)

  def map(self, buf:Buffer) -> BufferStorage: return self._map(buf.ensure_allocated())

  # implemented by the runtime
  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage: raise NotImplementedError("need alloc")
  def _free(self, storage:BufferStorage, options:BufferSpec): pass  # if opaque is a Python object, you don't need a free
  def _copyin(self, dest, src:memoryview): raise NotImplementedError("need copyin")
  def _copyout(self, dest:memoryview, src): raise NotImplementedError("need copyout")
  def _map(self, buf) -> BufferStorage: raise NotImplementedError("need map")
  def _unmap(self, mb): pass  # default no-op; override if _map allocates iface-side state
  def _offset(self, buf, size:int, offset:int): raise NotImplementedError("need offset")
  # def _transfer(self, dest, src, sz:int, src_dev, dest_dev):
  def _encode_decode(self, bufout, bufin, desc, hist:list, shape:tuple[int,...], frame_pos:int): raise NotImplementedError("need encdec") # optional

class HostAllocator(Allocator):
  def __init__(self, dev): super().__init__(dev, supports_copy_from_disk=False, supports_transfer=False)
  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage:
    if options.external_ptr is not None: view, meta = self._view(options.external_ptr, size), None
    elif (remote:=getattr(self.dev, "remote", None)) is not None: view, meta = remote.alloc_sysmem(round_up(size, mmap.PAGESIZE))
    else: view = self._view(mv_address(meta:=mmap.mmap(-1, size, access=mmap.ACCESS_WRITE)), size)
    return BufferStorage(view.addr, meta, view)

  def _free(self, storage:BufferStorage, options:BufferSpec):
    if (remote:=getattr(self.dev, "remote", None)) is not None:
      self.dev.synchronize()
      remote.free_sysmem(storage.host)

  def _view(self, addr:int, size:int) -> MMIOInterface:
    return remote.cpu_view(addr, size) if (remote:=getattr(self.dev, "remote", None)) is not None else MMIOInterface(addr, size, fmt='B')

  def _copyin(self, dest:int, src:memoryview):
    self.dev.synchronize()
    with cpu_profile(f"TINY -> {self.dev.device}", f"{self.dev.device}:COPY"): self._view(dest, src.nbytes)[:] = src.cast('B')
  def _copyout(self, dest:memoryview, src:int):
    self.dev.synchronize()
    with cpu_profile(f"{self.dev.device} -> TINY", f"{self.dev.device}:COPY"): dest[:] = self._view(src, dest.nbytes)[:]
  def _map(self, buf:Buffer) -> BufferStorage:
    if Device[buf.device].host != self.dev.host: raise RuntimeError(f"host memory is not on the node of {self.dev.device}")
    return BufferStorage(buf.host.addr)
  def _offset(self, buf:int, size:int, offset:int) -> int: return buf + offset

class DepsTracker:
  def __init__(self):
    # tracks (offset, end, dep) ranges per base buffer id to handle suballocated buffers correctly.
    self.w_dependency_map: dict[int, list[tuple[int, int, Any]]] = defaultdict(list)
    self.r_dependency_map: dict[int, list[tuple[int, int, Any]]] = defaultdict(list)

  @staticmethod
  def _key(buf:Any) -> tuple[Any, int, int]: return id(buf.base), buf.offset, buf.offset + buf.nbytes

  def access_resources(self, bufs:list[Any], write:list[int], new_dependency:Any):
    wait_nodes = []
    for i,buf in enumerate(bufs):
      key, s, e = self._key(buf)
      wait_nodes += [dep for st,en,dep in self.w_dependency_map[key] if st < e and s < en]
      if i in write: wait_nodes += [dep for st,en,dep in self.r_dependency_map[key] if st < e and s < en]
    for i,buf in enumerate(bufs):
      key, s, e = self._key(buf)
      if i in write:
        for dmap in [self.w_dependency_map, self.r_dependency_map]:
          kept = []
          for entry in dmap[key]:
            st, en, dep = entry
            if st == en: continue
            if en <= s or e <= st: kept.append(entry)
            else:
              if st < s: kept.append((st, s, dep))
              if e < en: kept.append((e, en, dep))
          dmap[key] = kept
        self.w_dependency_map[key].append((s, e, new_dependency))
      else: self.r_dependency_map[key].append((s, e, new_dependency))
    return list({id(x):x for x in wait_nodes}.values())

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
  def server(self, cmd:str, arch:str, *args) -> subprocess.Popen:
    argv = f"{cmd} {pathlib.Path(__file__).parent}/runtime/support/compileserver.py {type(self).__module__}:{type(self).__name__} {arch}"
    return subprocess.Popen(argv.split() + [str(a) for a in args], stdout=subprocess.PIPE, stdin=subprocess.PIPE, bufsize=0)
  def compile_server(self, src:str, proc:subprocess.Popen) -> bytes:
    unwrap(proc.stdin).write(struct.pack("I", len(src.encode())) + src.encode())
    if (lib:=unwrap(proc.stdout).read(struct.unpack("I", unwrap(proc.stdout).read(4))[0])): return lib
    raise CompileError("Compilation Error")


@dataclass
class TinyELF:
  lib: bytes
  name: str
  target: Target
  # tuple of (name, slot, dtype, shape)
  signature: tuple[tuple[str|None, int, DType, tuple], ...]
  profile_key: bytes|None = None

  @staticmethod
  def iter_sig(signature:tuple[tuple[str|None, int, DType, tuple], ...], offset:int=0) -> Generator[tuple[int, DType], None, None]:
    for _,_,dt,_ in signature:
      yield (offset:=round_up(offset, dt.itemsize)), dt
      offset += dt.itemsize

class Program(Generic[DeviceType]):
  def __init__(self, dev:DeviceType, obj:TinyELF): pass
  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(),
               wait=False) -> float|None: pass

class Compiled:
  ifaces:list[Callable] = []
  profile_events:list[ProfileEvent] = [ProfileDeviceEvent("CPU")] # NOTE: CPU is the default device.

  timestamp_divider: float = 1000.0
  wait_timeout_ms: float = 30000.0
  sleep_timeout_ms: int|None = None
  can_recover:bool = False
  rtalloc_size:int = 64<<20 # the pool every per-linear buffer is carved out of
  var_vals: dict[str, int] = {}

  # hcq2
  pm_batch:Any = None
  pm_encode:Any = None
  pm_lower:Any = None

  def __init__(self, device:str, allocator:Allocator, renderers:list[type[Renderer]], runtime:type[Program[Self]]|None, graph=None, arch=None):
    from tinygrad.renderer import Renderer
    from tinygrad.uop.ops import Ops, UPat, PatternMatcher
    from tinygrad.runtime.support.hcq2 import cfunc_buf

    self.device, self.allocator, self.runtime_t, self.graph, self.renderers = device, allocator, runtime, graph, renderers or [Renderer]
    self.device_id, self.arch = (int(idx) if ":" in device and (idx:=device.split(":")[1]).isdigit() else 0), arch
    self.peer_group = getattr(getattr(self, 'iface', None), 'peer_group', device.split(":")[0])
    self.cached_renderer:dict[Any, Renderer] = {}
    self.pending:dict[Compiled, int] = {} # timeline values of the devices that touched our memory

    # hcq2
    self.pm_bufferize = PatternMatcher([
      (UPat(Ops.PARAM, tag="timeline"), lambda ctx: ctx.timeline),
      (UPat(Ops.PARAM, tag="program", name="b"),
       lambda ctx, b: ctx.prog_bufs.setdefault(b, Buffer(ctx.device, b.max_numel(), b.dtype, options=BufferSpec(cpu_access=True, nolru=True)))),
      (UPat(Ops.PARAM, name="b"), lambda b, cfunc_buf=cfunc_buf: cfunc_buf(*b.tag[1:]) if isinstance(b.tag, tuple) and b.tag[0] == "cfunc" else None),
    ])

    # profiling
    self.prog_bufs:dict[UOp, Buffer] = {} # cache bufferized for programs
    self.prof_ents:dict[tuple[Buffer, int], ProfileGraphEntry] = {} # (a batch's timestamps, start slot) -> entry, read at synchronize

  @property
  def has_copy_queue(self) -> bool: return True

  @property
  def host(self) -> str: return f"CPU:{self.peer_group[7:]}" if self.peer_group.startswith("remote:") else HCQ_RUNTIME_DEV.value

  @property
  def renderer(self) -> Renderer: return self._select_renderer()

  @property
  def compiler(self) -> Compiler:
    if (ret:=self.renderer.compiler) is None: raise RuntimeError(f"no compiler for {self.device}")
    return ret

  def runtime(self, obj:TinyELF) -> Program[Self]: return unwrap(self.runtime_t)(self, obj)

  @functools.cache
  def rt_allocator(self, uncached:bool=True, host:bool=False) -> BumpAllocator: return BumpAllocator(self.rtalloc_size)

  @functools.cache
  def rt_buffer(self, uncached:bool=True, host:bool=False) -> Buffer:
    spec = BufferSpec(host=host, uncached=uncached, cpu_access=True)
    return Buffer(self.device, self.rt_allocator(uncached, host).size, dtypes.uint8, options=spec, preallocate=True)

  @functools.cached_property
  def timeline(self) -> Buffer: # [the signal, the value the last submitted batch signals]
    return Buffer(self.device, 2, dtypes.uint64, options=BufferSpec(host=True, uncached=True, cpu_access=True), initial_value=bytes(16))

  def _wait_signal(self, sig:MMIOInterface|memoryview, value:int, timeout:int|None=None):
    timeout = timeout if timeout is not None and self.can_recover else None
    st, done = time.perf_counter(), sig[0]
    while done < value:
      if done != (done:=sig[0]): st = time.perf_counter()
      elif (elapsed:=time.perf_counter() - st) > (timeout or self.wait_timeout_ms) / 1000: raise RuntimeError(f"{self.device} signal wait timed out")
      elif self.sleep_timeout_ms is not None and elapsed > self.sleep_timeout_ms / 1000: self.on_sleep()

  def synchronize(self, timeout:int|None=None):
    try:
      self._wait_signal(tl:=self.timeline.host.view(fmt='Q'), tl[1], timeout)
      for d, v in self.pending.items(): d._wait_signal(d.timeline.host.view(fmt='Q'), v, timeout)
    except RuntimeError:
      self.on_device_hang()
      raise
    if self.prof_ents: self.collect_prof()

  def count(self) -> int:
    """
    Returns the number of physical accelerators available to the runtime.
    """
    return self.iface.count if hasattr(self, 'iface') else 1

  def on_device_hang(self): raise RuntimeError(f"{self.device} hang detected")

  def on_sleep(self):
    if (iface:=getattr(self, "iface", None)) is not None and hasattr(iface, "sleep"): iface.sleep(self.sleep_timeout_ms)

  def device_props(self) -> dict[str,Any]: return {} # to be overridden if needed. dict keys are backend dependent.

  def finalize(self):
    """
    Called at the end of process lifetime to allow the device to finalize.
    """
    try: self.synchronize() # try to finalize the device in any case
    except RuntimeError as e: print(f"{self.device} synchronization failed before finalizing: {e}")
    if hasattr(self, 'iface') and hasattr(self.iface, 'device_fini'): self.iface.device_fini()

  # helpers

  def _renderer_name(self, r:type[Renderer]) -> str:
    return r.__name__.upper().removesuffix("RENDERER").removeprefix(devname:=self.device.split(':')[0].upper()) or devname

  def _select_renderer(self) -> Renderer:
    assert (rn:=next((self._renderer_name(r) for r in self.renderers if getenv(f"{self.device}_{self._renderer_name(r)}")), None)) is None, \
      f"{self.device}_{rn}=1 is deprecated, use DEV={self.device}:{rn} instead"
    t = DEV.target(self.device.split(':')[0], **({"arch":self.arch} if self.arch else {}))
    return select_first_inited(select_by_name(self.renderers, self._renderer_name, t.renderer, f"{self.device} has no renderer {t.renderer!r}"),
                               f"No renderer for {self.device} is available", self.cached_renderer, t)

  def _select_iface(self, device:str):
    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    assert (v:=getenv(k:=f'{type(self).__name__[:-6].upper()}_IFACE', "")) == "",  \
      f"{k}={v} is deprecated, use DEV={replace(DEV.target(type(self).__name__[:-6]), interface=v)} instead"
    t = DEV.target(dev:=type(self).__name__[:-6])
    filtered = select_by_name(self.ifaces, lambda i: i.__name__[:-5], t.interface, f"{dev} has no interface {t.interface!r}")
    filtered = [i for i in filtered if t.interface.startswith("MOCK") or not i.__name__[:-5].startswith("MOCK")] # never fallback to mock ifaces
    return select_first_inited([functools.partial(iface, self, self.device_id) for iface in filtered],
                               f"No interface for {dev}:{self.device_id} is available")

  # profiling

  def collect_prof(self):
    if PROFILE:
      es = list(self.prof_ents.items())
      sigs = [buf.host.view(fmt='Q')[i]/decimal.Decimal(self.timestamp_divider) for (buf, _), e in es for i in (e.st_id, e.en_id)]
      Compiled.profile_events.append(ProfileGraphEvent([replace(e, st_id=2*i, en_id=2*i+1) for i,(_, e) in enumerate(es)], [], sigs))
    self.prof_ents.clear()

  def _at_profile_finalize(self):
    if self.pm_encode is None: return
    from tinygrad.tensor import Tensor
    tdiffs = []
    for _ in range(5):
      with Context(DEBUG=0, BEAM=0, TRACK_MATCH_STATS=0): Tensor.ones(1, device=self.device).contiguous().realize()
      if not (ents:=list(self.prof_ents.items())): return
      self.prof_ents.clear()
      st = perf_counter_us()
      self.synchronize()
      gpu = max(buf.host.view(fmt='Q')[e.en_id] for (buf, _), e in ents)/decimal.Decimal(self.timestamp_divider)
      tdiffs.append((st+perf_counter_us())/2 - gpu)
    Compiled.profile_events.append(ProfileDeviceEvent(self.device, statistics.median(tdiffs), self.device_props()))

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
      for iface in [i for i in d.ifaces if not i.__name__.startswith("MOCK")]:
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
