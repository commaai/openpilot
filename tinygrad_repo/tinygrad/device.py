from __future__ import annotations
from dataclasses import dataclass, replace, field
from collections import defaultdict
from typing import Any, Callable, Generic, TypeVar, Iterator, Generator, Self
import importlib, inspect, functools, itertools, pathlib, os, contextlib, atexit, pickle, decimal, mmap, time, statistics
from tinygrad.helpers import mv_address, LRU, getenv, DEBUG, GlobalCounters, PROFILE, temp, colored
from tinygrad.helpers import Context, ALLOW_DEVICE_USAGE, MAX_BUFFER_SIZE, cpu_events, ProfileEvent, ProfilePointEvent, suppress_finalizing
from tinygrad.helpers import select_by_name, select_first_inited, DEV, TracingKey, size_to_str, pluralize, unwrap, round_up, is_numpy_ndarray
from tinygrad.helpers import cpu_profile, perf_counter_us, HCQ_RUNTIME_DEV, is_disk_device, canonicalize_device_name, Target, to_tuple, NODE_NDEVS
from tinygrad.dtype import dtypes
from tinygrad.runtime.support.memory import MMIOInterface
from tinygrad.renderer import Renderer, Compiler, TinyELF
from tinygrad.uop.ops import PatternMatcher

# **************** Device ****************

ALL_DEVICES = ["METAL", "AMD", "NV", "QCOM", "CUDA", "HIP", "CL", "CPU", "DSP", "WEBGPU"]
HCQ_DEVS = frozenset(("AMD", "NV", "QCOM", "CUDA", "HIP", "NULL", "METAL", "CL"))
HOST_DEVS = frozenset(("CPU", "PYTHON"))
UVM_DEVS = frozenset(("QCOM", "METAL", "CL")) # cl uses host memcpys

class _Device:
  def __init__(self) -> None:
    self._devices = [x.stem[len("ops_"):].upper() for x in (pathlib.Path(__file__).parent/"runtime").iterdir() if x.stem.startswith("ops_")]
    self._opened_devices:dict[str, Compiled] = {}
  # NOTE: you can't cache canonicalize in case Device.DEFAULT changes
  def canonicalize(self, device:str|None) -> str: return canonicalize_device_name(device if device is not None else Device.DEFAULT)
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
    self._opened_devices[ix] = ret = self.get_class(ix)(ix)
    if DEBUG >= 1: print(f"opened device {ix} from pid:{os.getpid()} as DEV={self.target_for(ix)}")
    return ret
  def target_for(self, device:str, cv=DEV) -> Target:
    if (t:=cv.target(device.split(':')[0])).arch: return t
    try: d = self._opened_devices.get(device:=self.canonicalize(device)) or self[device]
    except Exception as e: raise RuntimeError(f"no target for {device}: set its arch in DEV, or open it") from e
    iface = type(d.iface).__name__[:-5] if hasattr(d, "iface") else ""
    return replace(t, arch=d.arch or "", interface=t.interface or iface, props=",".join(f"{k}={v}" for k, v in (d.props | t.opts).items()))
  def renderer_for(self, device:str, cv=DEV) -> Renderer: return select_renderer(cv.target((t:=self.target_for(device, cv)).device, arch=t.arch))
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
def node_of(dev:str) -> str:
  if dev.startswith("CPU:") and dev.count(":") == 2: return dev[4:]
  nodes = [n if n == "local" or ":" in n else f"{n}:6667" for n in getenv("REMOTE", "").split(",") if n]
  if dev.split(":")[0] not in HCQ_DEVS or not nodes: return "local"
  assert len(nodes) == 1 or NODE_NDEVS, f"REMOTE={getenv('REMOTE', '')} has several nodes, set NODE_NDEVS to the gpus of one"
  return nodes[int(dev.split(":")[1]) // NODE_NDEVS.value if ":" in dev and NODE_NDEVS else 0]
def host_of(dev:str) -> str:
  if dev.split(":")[0] == "NULL": return dev
  return HCQ_RUNTIME_DEV.device if (node:=node_of(dev)) == "local" else f"CPU:{node}"
def peer_group(dev:str) -> str:
  if (node:=node_of(dev:=Device.canonicalize(dev))) != "local": return node
  return dev if (iface:=Device.target_for(dev).interface).endswith("USB") else "PCI" if iface.endswith("PCI") else dev.split(":")[0]
def is_iface_usb(dev:str|tuple[str, ...]) -> bool: return Device.target_for(Device.canonicalize(to_tuple(dev)[0])).interface.endswith("USB")

# **************** Profile ****************

@dataclass(frozen=True)
class ProfileDeviceEvent(ProfileEvent): device:str; tdiff:decimal.Decimal=decimal.Decimal(0); target:Target|None=None # noqa: E702

@dataclass(frozen=True)
class ProfileProgramEvent(ProfileEvent): device:str; name:str; lib:bytes|None; base:int|None; tag:int|None=None; profile_key:bytes|None=None # noqa: E702

@dataclass(frozen=True)
class ProfileGraphEntry: device:str; name:str|TracingKey; st_id:int; en_id:int; profile_key:bytes|None=None # noqa: E702

@dataclass(frozen=True)
class ProfileGraphEvent(ProfileEvent): ents:list[ProfileGraphEntry]; sigs:list[decimal.Decimal] # noqa: E702

# **************** Buffer + Allocators ****************

@dataclass(frozen=True, eq=True)
class BufferSpec:
  uncached: bool = False
  cpu_access: bool = False
  host: bool = False
  nolru: bool = False
  external_ptr: int|None = None

class MultiBuffer:
  def __init__(self, device:tuple[str, ...], nbytes:int):
    self.bufs = [Buffer(d, nbytes) for d in device]
  @property
  def device(self): return tuple(x.device for x in self.bufs)
  @property
  def nbytes(self): return self.bufs[0].nbytes
  def is_allocated(self): return all(x.is_allocated() for x in self.bufs)
  def __repr__(self): return f"<multibuf real:{self.is_allocated()} device:{tuple(x.device for x in self.bufs)} nbytes:{self.nbytes}>"

@dataclass(frozen=True)
class BufferStorage: buf:Any; meta:Any=None; host:MMIOInterface|None=None; maps:dict[Compiled, BufferStorage]=field(default_factory=dict) # noqa: E702

class Buffer:
  profile_events:list[ProfileEvent] = []
  def __init__(self, device:str, nbytes:int, opaque:Any=None, options:BufferSpec|None=None,
               initial_value:bytes|pickle.PickleBuffer|None=None, base:Buffer|None=None, offset:int=0, preallocate=False,
               allocator:Allocator|None=None):
    self.device, self.nbytes, self.offset, self._base = Device.canonicalize(device), nbytes, offset, base
    if allocator is not None: self.allocator = allocator
    self.options = options if options is not None else BufferSpec()
    self._storage:BufferStorage|None = None
    if base is None:
      assert offset == 0, "base buffers can't have offset"
      if opaque is not None: self.allocate(opaque)
      if initial_value is not None:
        self.allocate()
        if (host:=self.get_storage().host) is not None: host[:] = memoryview(initial_value).cast('B')
        else: self.copy_from(Buffer("PYTHON", self.nbytes, initial_value=initial_value))
        if isinstance(initial_value, pickle.PickleBuffer): initial_value.release()
    else:
      assert base._base is None, "base can't have a base"
      assert self.device == base.device, "base must have the same device"
    if preallocate: self.allocate()

  @suppress_finalizing
  def __del__(self): self._storage is None or self.deallocate()

  def __repr__(self):
    return f"<buf real:{self.is_allocated()} device:{self.device} nbytes:{self.nbytes}" + \
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
    # MAX_BUFFER_SIZE limits raw storage bytes (not logical elements).
    if not self.device.startswith("NULL") and self.nbytes > MAX_BUFFER_SIZE > 0 and self.options.external_ptr is None:
      raise RuntimeError(f"buffer of size {self.nbytes/1e6:.2f}MB is too large")
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
      if not is_disk_device(self.device) and self.options.external_ptr is None:
        GlobalCounters.mem_used += self.nbytes
        GlobalCounters.mem_used_per_device[self.device] += self.nbytes
      if PROFILE: Buffer.profile_events.append(ProfilePointEvent(self.device, "alloc", self.trace_num, {"nbytes":self.nbytes}))
    self._storage, self._base_storage = storage, self.base._storage if self._base else None
    return self

  def deallocate(self):
    assert self._storage is not None, "buffer must be allocated to deallocate"
    if DEBUG is not None and DEBUG >= 7: print(f"buffer: deallocate {self.nbytes} bytes on {self.device}")
    if self._base is None:
      if GlobalCounters is not None and not is_disk_device(self.device) and self.options.external_ptr is None:
        GlobalCounters.mem_used -= self.nbytes
        GlobalCounters.mem_used_per_device[self.device] -= self.nbytes
      if PROFILE: Buffer.profile_events.append(ProfilePointEvent(self.device, "free", self.trace_num))
      self.allocator.free(self._storage, self.nbytes, self.options)
    self._storage, self._base_storage = None, None

  def __reduce_ex__(self, protocol):
    buf:bytearray|pickle.PickleBuffer|None = None
    if self._base is not None:
      return self.__class__, (self.device, self.nbytes, None, None, None, self.base, self.offset, self.is_allocated())
    if self.device == "NPY" and self.is_allocated():
      import numpy as np
      arr = np.frombuffer(self.meta, np.uint8, count=self.nbytes) # an out-of-band pickle buffer keeps the storage alive
      return self.__class__, (self.device, self.nbytes, arr, self.options, None)
    if self.is_allocated():
      buf = pickle.PickleBuffer(self.as_memoryview()) if protocol >= 5 else bytearray(self.as_memoryview())
    return self.__class__, (self.device, self.nbytes, None, self.options, buf)

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
    Buffer("NPY", self.nbytes, opaque=(mv:=memoryview(bytearray(self.nbytes)))).copy_from(self)
    return mv

  def copy_from(self, src:Buffer) -> Buffer:
    assert self.nbytes == src.nbytes, f"copy size mismatch, {self.nbytes} != {src.nbytes}"
    assert self.is_allocated() and src.is_allocated(), "copy requires allocated buffers"
    from tinygrad.engine.realize import link_linear, run_linear
    from tinygrad.codegen.compile import compile_linear
    from tinygrad.uop.ops import UOp, Ops
    du, su = UOp.from_buffer(self, dtypes.uint8), UOp.from_buffer(src, dtypes.uint8)
    linear = compile_linear(UOp(Ops.LINEAR, src=(du.as_extern(0).store_call(su.as_extern(1)),)), cache=True)
    run_linear(link_linear(linear, inputs:=[du, su]), input_uops=inputs, update_stats=False)
    return self

  def view(self, nbytes:int, offset:int) -> Buffer:
    assert offset < self.nbytes, "offset must be less than nbytes"
    return Buffer(self.device, nbytes, base=self.base, offset=self.offset+offset)

DeviceType = TypeVar('DeviceType', bound='Compiled')

# TODO: size, dest, src are the same type. can we enforce this?
class Allocator(Generic[DeviceType]):
  lru = True

  def __init__(self, dev:DeviceType):
    self.dev: DeviceType = dev
    self.cache:dict[tuple[int, BufferSpec|None], list[tuple[BufferStorage, dict[Compiled, int]]]] = defaultdict(list)

  def alloc(self, size:int, options:BufferSpec|None=None) -> BufferStorage:
    assert size > 0, f"alloc size must be positive, getting {size}"

    # check if old most is free
    if (c:=self.cache[(size, options)]) and all(dev.timeline.host.view(fmt='Q')[0] >= freed_at for dev, freed_at in c[0][1].items()):
      return c.pop(0)[0]

    spec = options if options is not None else BufferSpec()
    try: return self._alloc(size, spec)
    except (RuntimeError, MemoryError): self.free_cache()
    try: return self._alloc(size, spec)
    except (RuntimeError, MemoryError) as e: raise MemoryError(f"Allocation of {size_to_str(size)} failed on {self.dev.device}. "
                                                               f"Used: {size_to_str(GlobalCounters.mem_used_per_device[self.dev.device])}") from e

  def free(self, storage:BufferStorage, size:int, options:BufferSpec|None=None):
    spec = options if options is not None else BufferSpec()
    if not LRU or not self.lru or spec.nolru or spec.external_ptr is not None: return self.do_free(storage, spec)

    # add that buf to cache
    users:list[Compiled] = [self.dev, *storage.maps] if storage.host is not None else []
    freed_at = {dev:dev.timeline.host.view(fmt='Q')[1] for dev in users if 'timeline' in vars(dev)}
    self.cache[(size, options)].append((storage, freed_at))

  def free_cache(self):
    for (_, options), storages in self.cache.items():
      for storage, _ in storages: self.do_free(storage, options if options is not None else BufferSpec())
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

class HostAllocator(Allocator):
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

# **************** for Compiled Devices ****************

class Program(Generic[DeviceType]):
  def __init__(self, dev:DeviceType, obj:TinyELF, links:dict[str,Program[DeviceType]]): pass
  def __call__(self, *args, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), wait=False) -> float|None: pass

# a renderer is built from a fully specified Target alone, the device isn't opened
def renderer_name(r:type[Renderer], device:str) -> str: return r.__name__.upper().removesuffix("RENDERER").removeprefix(device) or device
@functools.cache
def select_renderer(t:Target) -> Renderer:
  if not (renderers:=Device.get_class(t.device).renderers): raise RuntimeError(f"{t.device} has no renderers, it can't run kernels")
  assert (rn:=next((renderer_name(r, t.device) for r in renderers if getenv(f"{t.device}_{renderer_name(r, t.device)}")), None)) is None, \
    f"{t.device}_{rn}=1 is deprecated, use DEV={t.device}:{rn} instead"
  candidates = select_by_name(renderers, lambda r: renderer_name(r, t.device), t.renderer, f"{t.device} has no renderer {t.renderer!r}")
  return select_first_inited(candidates, f"No renderer for {t.device} is available", None, t)

class Compiled:
  ifaces:list[Callable] = []
  renderers:list[type[Renderer]] = []
  profile_events:list[ProfileEvent] = [ProfileDeviceEvent("CPU")] # NOTE: CPU is the default device.

  timestamp_divider: float = 1000.0
  wait_timeout_ms: float = 30000.0
  sleep_timeout_ms: int|None = None
  can_recover:bool = False
  var_vals: dict[str, int] = {}

  # hcq2
  pm_encode:PatternMatcher
  pm_lower:PatternMatcher
  pm_link:PatternMatcher
  link_bufs:dict[str, Buffer|Callable[..., Buffer]] = {}

  def __init__(self, device:str, allocator:Allocator, runtime:type[Program[Self]]|None, arch=None, props:dict[str, Any]|None=None):
    self.device, self.allocator, self.runtime_t = device, allocator, runtime
    self.device_id, self.arch, self.props = (int(idx) if ":" in device and (idx:=device.split(":")[1]).isdigit() else 0), arch, props or {}
    self.peer_group = getattr(getattr(self, 'iface', None), 'peer_group', device.split(":")[0])

    # profiling
    self.prof_ents:dict[tuple[Buffer, int], ProfileGraphEntry] = {} # (a batch's timestamps, start slot) -> entry, read at synchronize

  @property
  def host(self) -> str: return host_of(self.device)

  @property
  def renderer(self) -> Renderer: return Device.renderer_for(self.device)

  @property
  def compiler(self) -> Compiler: return self.renderer.compiler

  def runtime(self, obj:TinyELF, links:dict[str,Program[Self]]={}) -> Program[Self]:
    return unwrap(self.runtime_t)(self, obj, links=links)

  @functools.cached_property
  def timeline(self) -> Buffer: # [the signal, the value the last submitted batch signals]
    return Buffer(self.device, 16, options=BufferSpec(host=True, uncached=True, cpu_access=True), initial_value=bytes(16))

  @functools.cached_property
  def error_state(self) -> Buffer: return Buffer(self.host, 8, options=BufferSpec(nolru=True), preallocate=True)

  def _wait_signal(self, sig:MMIOInterface|memoryview, value:int, timeout:int|None=None):
    timeout = timeout if timeout is not None and self.can_recover else None
    st, done, err = time.perf_counter(), sig[0], self.error_state.host.view(fmt='q')
    while done < value and not err[0]:
      if done != (done:=sig[0]): st = time.perf_counter()
      elif (elapsed:=time.perf_counter() - st) > (timeout or self.wait_timeout_ms) / 1000: raise RuntimeError(f"{self.device} signal wait timed out")
      elif self.sleep_timeout_ms is not None and elapsed > self.sleep_timeout_ms / 1000: self.on_sleep()
    if err[0]: raise RuntimeError(f"{self.device} failed with {err[0]}")

  def synchronize(self, timeout:int|None=None):
    try: self._wait_signal(tl:=self.timeline.host.view(fmt='Q'), tl[1], timeout)
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

  def finalize(self):
    """
    Called at the end of process lifetime to allow the device to finalize.
    """
    try: self.synchronize() # try to finalize the device in any case
    except RuntimeError as e: print(f"{self.device} synchronization failed before finalizing: {e}")
    if hasattr(self, 'iface') and hasattr(self.iface, 'device_fini'): self.iface.device_fini()

  # helpers

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
    div = decimal.Decimal(self.timestamp_divider)
    for buf, g in itertools.groupby(self.prof_ents.items(), key=lambda x: x[0][0]) if PROFILE else []:
      Compiled.profile_events.append(ProfileGraphEvent([e for _, e in g], [t/div for t in buf.host.view(fmt='Q')[:]]))
    self.prof_ents.clear()

  def _at_profile_finalize(self):
    if self.device.split(":")[0] not in HCQ_DEVS: return
    from tinygrad.tensor import Tensor
    tdiffs, n = [], len(Compiled.profile_events)
    for _ in range(5):
      with Context(DEBUG=0, BEAM=0, TRACK_MATCH_STATS=0): Tensor.ones(1, device=self.device).contiguous().realize()
      if not (ents:=list(self.prof_ents.items())): return
      st = perf_counter_us()
      self.synchronize()
      gpu = max(buf.host.view(fmt='Q')[e.en_id] for (buf, _), e in ents)/decimal.Decimal(self.timestamp_divider)
      tdiffs.append((st+perf_counter_us())/2 - gpu)
    del Compiled.profile_events[n:] # cut calibration kernels
    Compiled.profile_events.append(ProfileDeviceEvent(self.device, statistics.median(tdiffs), Device.target_for(self.device)))

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
        name = renderer_name(r, device.split(':')[0])
        try:
          with Context(CACHELEVEL=0, DEV=f"{device}:{name}"): test = (Tensor([1,2,3], device=device) * 2).tolist()
          if test != [2,4,6]: raise ValueError(f"got {test} instead of [2, 4, 6]")
          default_text = '(default)' if type(d.renderer) is r else f'(DEV={device}:{name} to make default)'
          ren_results.append(f"{colored('+', 'green')} {name} {default_text}")
        except Exception as e: ren_results.append(f"{colored('-', 'red')} {name}: {e}")
      result = (colored('PASS', 'green') + ("\n"+" "*12+"interfaces:\n" if iface_results else "") + '\n'.join([" "*13+x for x in iface_results]) +
                (("\n"+" "*12+"renderers:\n") + '\n'.join([" "*13+x for x in ren_results]) if len(ren_results) > 1 else ""))
    except Exception as e: result = f"{colored('FAIL', 'red')} {e}"
    yield f"{'*' if device == Device.DEFAULT else ' '} {device:8s}: {result}"

if __name__ == "__main__":
  for s in enumerate_devices_str(): print(s)
