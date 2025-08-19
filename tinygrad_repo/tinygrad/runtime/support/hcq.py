from __future__ import annotations
from typing import cast, Callable, Type, TypeVar, Generic, Any
import contextlib, decimal, statistics, time, ctypes, array, os, struct, traceback, collections
try: import fcntl # windows misses that
except ImportError: fcntl = None #type:ignore[assignment]
from tinygrad.helpers import PROFILE, getenv, to_mv, round_up, ProfileRangeEvent
from tinygrad.renderer import Renderer
from tinygrad.device import BufferSpec, Compiler, Compiled, LRUAllocator, ProfileDeviceEvent, ProfileProgramEvent
from tinygrad.uop.ops import sym_infer, sint, Variable, UOp
from tinygrad.runtime.autogen import libc

class MMIOInterface:
  def __init__(self, addr:int, nbytes:int, fmt='B'): self.mv, self.addr, self.nbytes, self.fmt = to_mv(addr, nbytes).cast(fmt), addr, nbytes, fmt
  def __len__(self): return self.nbytes // struct.calcsize(self.fmt)
  def __getitem__(self, k): return (bytes(self.mv[k]) if self.fmt == 'B' else self.mv[k].tolist()) if isinstance(k, slice) else self.mv[k]
  def __setitem__(self, k, v): self.mv[k] = v
  def view(self, offset:int=0, size:int|None=None, fmt=None) -> MMIOInterface:
    return MMIOInterface(self.addr+offset, size or (self.nbytes - offset), fmt=fmt or self.fmt)

class FileIOInterface:
  """
  Hardware Abstraction Layer for HCQ devices. The class provides a unified interface for interacting with hardware devices.
  """

  def __init__(self, path:str="", flags:int=os.O_RDONLY, fd:int|None=None):
    self.path:str = path
    self.fd:int = fd or os.open(path, flags)
  def __del__(self):
    if hasattr(self, 'fd'): os.close(self.fd)
  def ioctl(self, request, arg): return fcntl.ioctl(self.fd, request, arg)
  def mmap(self, start, sz, prot, flags, offset):
    x = libc.mmap(start, sz, prot, flags, self.fd, offset)
    if x == 0xffffffffffffffff: raise OSError(f"Failed to mmap {sz} bytes at {hex(start)}: {os.strerror(ctypes.get_errno())}")
    return x
  def read(self, size=None, binary=False, offset=None):
    if offset is not None: self.seek(offset)
    with open(self.fd, "rb" if binary else "r", closefd=False) as file: return file.read(size)
  def write(self, content, binary=False, offset=None):
    if offset is not None: self.seek(offset)
    with open(self.fd, "wb" if binary else "w", closefd=False) as file: file.write(content)
  def listdir(self): return os.listdir(self.path)
  def seek(self, offset): os.lseek(self.fd, offset, os.SEEK_SET)
  @staticmethod
  def anon_mmap(start, sz, prot, flags, offset):
    x = libc.mmap(start, sz, prot, flags, -1, offset)
    if x == 0xffffffffffffffff: raise OSError(f"Failed to mmap {sz} bytes at {hex(start)}: {os.strerror(ctypes.get_errno())}")
    return x
  @staticmethod
  def munmap(buf, sz): return libc.munmap(buf, sz)
  @staticmethod
  def exists(path): return os.path.exists(path)
  @staticmethod
  def readlink(path): return os.readlink(path)
  @staticmethod
  def eventfd(initval, flags=None): return FileIOInterface(fd=os.eventfd(initval, flags))  # type: ignore[attr-defined]

if MOCKGPU:=getenv("MOCKGPU"): from test.mockgpu.mockgpu import MockFileIOInterface as FileIOInterface  # noqa: F401 # pylint: disable=unused-import

# **************** for HCQ Compatible Devices ****************

SignalType = TypeVar('SignalType', bound='HCQSignal')
HCQDeviceType = TypeVar('HCQDeviceType', bound='HCQCompiled')
ProgramType = TypeVar('ProgramType', bound='HCQProgram')
ArgsStateType = TypeVar('ArgsStateType', bound='HCQArgsState')
QueueType = TypeVar('QueueType', bound='HWQueue')

class BumpAllocator:
  def __init__(self, size:int, base:int=0, wrap:bool=True): self.size, self.ptr, self.base, self.wrap = size, 0, base, wrap
  def alloc(self, size:int, alignment:int=1) -> int:
    if round_up(self.ptr, alignment) + size > self.size:
      if not self.wrap: raise RuntimeError("Out of memory")
      self.ptr = 0
    self.ptr = (res:=round_up(self.ptr, alignment)) + size
    return res + self.base

class HWQueue(Generic[SignalType, HCQDeviceType, ProgramType, ArgsStateType]):
  """
  A base class for hardware command queues in the HCQ (Hardware Command Queue) API.
  """

  def __init__(self):
    self._q:Any = []
    self.binded_device:HCQDeviceType|None = None
    self.q_sints:list[tuple[int, int]] = []
    self.mv_sints:list[tuple[MMIOInterface, int, int, int|None]] = []
    self.syms:list[sint] = []
    self._prev_resolved_syms:list[int|None] = []

  def _new_sym(self, sym:sint) -> int:
    if sym not in self.syms:
      self.syms.append(sym)
      self._prev_resolved_syms.append(None)
    return self.syms.index(sym)

  def q(self, *values):
    """
    Enqueues values in the queue.

    Args:
      values: The values to enqueue in the queue.
    """

    for v in values:
      if isinstance(v, UOp):
        self.q_sints.append((len(self._q), self._new_sym(v)))
        self._q.append(0xbadc0ded)
      else: self._q.append(v)

  # *** common commands  ***

  def timestamp(self, signal:SignalType):
    """
    Enqueues a timestamp command which records the current time in a signal after all previously enqueued commands are completed.

    Args:
      signal: The signal to store the timestamp
    """

  def signal(self, signal:SignalType, value:sint):
    """
    Enqueues a signal command which sets the signal to the given value, ensuring all previous operations are completed.

    Args:
      signal: The signal to set
      value: The value to set the signal to
    """

  def wait(self, signal:SignalType, value:sint):
    """
    Enqueues a wait command which halts execution until the signal is greater than or equal to a specific value.

    Args:
      signal: The signal to wait on
      value: The value to wait for
    """

  # *** commands for compute queues ***

  def memory_barrier(self):
    """
    Enqueues a memory barrier command to ensure memory coherence between agents. Only on compute queues.
    """

  def exec(self, prg:ProgramType, args_state:ArgsStateType, global_size:tuple[sint, ...], local_size:tuple[sint, ...]):
    """
    Enqueues an execution command for a kernel program. Only on compute queues.

    Args:
      prg: The program to execute
      args_state: The args state to execute program with
      global_size: The global work size
      local_size: The local work size
    """

  # *** commands for copy queues ***

  def copy(self, dest:sint, src:sint, copy_size:int):
    """
    Enqueues a copy command to transfer data. Only on copy queues.

    Args:
      dest: The destination of the copy
      src: The source of the copy
      copy_size: The size of data to copy
    """

  # *** submit and bind commands  ***

  def bind(self, dev:HCQDeviceType):
    """
    Associates the queue with a specific device for optimized execution.

    This optional method allows backend implementations to tailor the queue for efficient use on the given device. When implemented, it can eliminate
    the need to copy queues into the device, thereby enhancing performance.

    Args:
      dev: The target device for queue optimization.

    Note:
      Implementing this method is optional but recommended for performance gains.
    """

  def bind_args_state(self, args_state:ArgsStateType):
    for vals, mem, fmt in args_state.bind_data: self.bind_sints_to_mem(*vals, mem=mem, fmt=fmt)

  def bind_sints(self, *vals:sint, mem:MMIOInterface, struct_t:Type[ctypes.Structure], start_field:str, fmt, mask:int|None=None):
    self.bind_sints_to_mem(*vals, mem=mem, fmt=fmt, mask=mask, offset=getattr(struct_t, start_field).offset)

  def bind_sints_to_mem(self, *vals:sint, mem:MMIOInterface, fmt, mask:int|None=None, offset:int=0):
    mv = mem.view(offset=offset, size=len(vals)*8, fmt=fmt)
    for i, val in enumerate(vals):
      if isinstance(val, int): mv[i] = val if mask is None else ((mv[i] & ~mask) | val)
      else: self.mv_sints.append((mv, i, self._new_sym(val), mask))

  def _apply_var_vals(self, var_vals:dict[Variable, int]):
    resolved_syms = [sym_infer(sym, var_vals) for sym in self.syms]

    for off, sym_idx in self.q_sints:
      if self._prev_resolved_syms[sym_idx] == resolved_syms[sym_idx]: continue
      self._q[off] = resolved_syms[sym_idx]

    for mv, off, sym_idx, mask in self.mv_sints:
      if self._prev_resolved_syms[sym_idx] == resolved_syms[sym_idx]: continue
      mv[off] = resolved_syms[sym_idx] if mask is None else ((mv[off] & ~mask) | resolved_syms[sym_idx])

    self._prev_resolved_syms = cast(list[int|None], resolved_syms)

  def submit(self, dev:HCQDeviceType, var_vals:dict[Variable, int]|None=None):
    """
    Submits the command queue to a specific device for execution.

    Args:
      dev: The device to submit the queue to
    """

    if var_vals is not None: self._apply_var_vals(var_vals)
    self._submit(dev)
    return self
  def _submit(self, dev:HCQDeviceType): raise NotImplementedError("need _submit")

class HCQSignal(Generic[HCQDeviceType]):
  def __init__(self, base_buf:HCQBuffer, value:int=0, owner:HCQDeviceType|None=None, is_timeline:bool=False, timestamp_divider=1000):
    self.base_buf, self.value_addr, self.timestamp_addr, self.owner = base_buf, base_buf.va_addr+0, base_buf.va_addr+8, owner
    self.is_timeline = is_timeline
    self.timestamp_divider:decimal.Decimal = decimal.Decimal(timestamp_divider)

    if isinstance(self.base_buf.va_addr, int):
      self.value_mv, self.timestamp_mv = self.base_buf.cpu_view().view(0, 8, 'Q'), self.base_buf.cpu_view().view(8, 8, 'Q')
      self.value_mv[0] = value

  def __del__(self):
    if isinstance(self.base_buf.va_addr, int) and self.owner is not None: HCQCompiled.signal_pool[self.owner.peer_group].append(self.base_buf)

  @property
  def value(self) -> int: return self.value_mv[0]

  @value.setter
  def value(self, new_value:int): self.value_mv[0] = new_value

  @property
  def timestamp(self) -> decimal.Decimal:
    """
    Get the timestamp field of the signal.

    This property provides read-only access to the signal's timestamp.

    Returns:
      The timestamp in microseconds.
    """
    return self.timestamp_mv[0] / self.timestamp_divider

  def _sleep(self, time_spent_waiting_ms:int):
    """
    Optional function which can implement sleep functionality for the signal.
    """

  def wait(self, value:int, timeout:int=getenv("HCQDEV_WAIT_TIMEOUT_MS", 30000)):
    """
    Waits the signal is greater than or equal to a specific value.

    Args:
      value: The value to wait for.
      timeout: Maximum time to wait in milliseconds. Defaults to 30s.
    """
    start_time = int(time.perf_counter() * 1000)
    while (not_passed:=(prev_value:=self.value) < value) and (time_spent:=int(time.perf_counter() * 1000) - start_time) < timeout:
      self._sleep(time_spent)
      if self.value != prev_value: start_time = int(time.perf_counter() * 1000) # progress was made, reset timer
    if not_passed and self.value < value: raise RuntimeError(f"Wait timeout: {timeout} ms! (the signal is not set to {value}, but {self.value})")

@contextlib.contextmanager
def hcq_profile(dev:HCQCompiled, enabled, desc, queue_type:Callable[[], HWQueue]|None=None, queue:HWQueue|None=None):
  st, en = (dev.new_signal(), dev.new_signal()) if enabled else (None, None)

  if enabled and queue is not None: queue.timestamp(st)
  elif enabled:
    assert queue_type is not None
    queue_type().wait(dev.timeline_signal, dev.timeline_value - 1).timestamp(st).signal(dev.timeline_signal, dev.next_timeline()).submit(dev)

  try: yield (st, en)
  finally:
    if enabled and queue is not None: queue.timestamp(en)
    elif enabled:
      assert queue_type is not None
      queue_type().wait(dev.timeline_signal, dev.timeline_value - 1).timestamp(en).signal(dev.timeline_signal, dev.next_timeline()).submit(dev)

    if enabled and PROFILE: dev.sig_prof_records.append((cast(HCQSignal, st), cast(HCQSignal, en), desc, queue_type is dev.hw_copy_queue_t))

class HCQArgsState(Generic[ProgramType]):
  def __init__(self, buf:HCQBuffer, prg:ProgramType, bufs:tuple[HCQBuffer, ...], vals:tuple[sint, ...]=()):
    self.buf, self.prg, self.bufs, self.vals = buf, prg, bufs, vals
    self.bind_data:list[tuple[tuple[sint, ...], MMIOInterface, str]] = []

  def bind_sints_to_buf(self, *vals:sint, buf:HCQBuffer, fmt, offset=0): self.bind_data.append((vals, buf.cpu_view().view(offset=offset), fmt))

class CLikeArgsState(HCQArgsState[ProgramType]):
  def __init__(self, buf:HCQBuffer, prg:ProgramType, bufs:tuple[HCQBuffer, ...], vals:tuple[sint, ...]=(), prefix:list[int]|None=None):
    super().__init__(buf, prg, bufs, vals=vals)

    if prefix is not None: self.buf.cpu_view().view(size=len(prefix) * 4, fmt='I')[:] = array.array('I', prefix)

    self.bind_sints_to_buf(*[b.va_addr for b in bufs], buf=self.buf, fmt='Q', offset=len(prefix or []) * 4)
    self.bind_sints_to_buf(*vals, buf=self.buf, fmt='I', offset=len(prefix or []) * 4 + len(bufs) * 8)

class HCQProgram(Generic[HCQDeviceType]):
  def __init__(self, args_state_t:Type[HCQArgsState], dev:HCQDeviceType, name:str, kernargs_alloc_size:int, lib:bytes|None=None, base:int|None=None):
    self.args_state_t, self.dev, self.name, self.kernargs_alloc_size = args_state_t, dev, name, kernargs_alloc_size
    if PROFILE: Compiled.profile_events += [ProfileProgramEvent(dev.device, name, lib, base)]

  @staticmethod
  def _fini(dev, buf, spec): dev.allocator.free(buf, buf.size, spec)

  def fill_kernargs(self, bufs:tuple[HCQBuffer, ...], vals:tuple[int, ...]=(), kernargs:HCQBuffer|None=None) -> HCQArgsState:
    """
    Fills arguments for the kernel, optionally allocating space from the device if `kernargs_ptr` is not provided.
    Args:
      bufs: Buffers to be written to kernel arguments.
      vals: Values to be written to kernel arguments.
      kernargs_ptr: Optional pointer to pre-allocated kernel arguments memory.
    Returns:
      Arguments state with the given buffers and values set for the program.
    """
    argsbuf = kernargs or self.dev.kernargs_buf.offset(offset=self.dev.kernargs_offset_allocator.alloc(self.kernargs_alloc_size),
                                                       size=self.kernargs_alloc_size)
    return self.args_state_t(argsbuf, self, bufs, vals=vals)

  def __call__(self, *bufs:HCQBuffer, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1),
               vals:tuple[int, ...]=(), wait:bool=False) -> float|None:
    """
    Enqueues the program for execution with the given arguments and dimensions.

    Args:
      bufs: Buffer arguments to execute the kernel with.
      global_size: Specifies the global work size for kernel execution (equivalent to CUDA's grid size).
      local_size: Specifies the local work size for kernel execution (equivalent to CUDA's block size).
      vals: Value arguments to execute the kernel with.
      wait: If True, waits for the kernel to complete execution.

    Returns:
      Execution time of the kernel if 'wait' is True, otherwise None.
    """

    kernargs = self.fill_kernargs(bufs, vals)
    q = self.dev.hw_compute_queue_t().wait(self.dev.timeline_signal, self.dev.timeline_value - 1).memory_barrier()

    with hcq_profile(self.dev, queue=q, desc=self.name, enabled=wait or PROFILE) as (sig_st, sig_en):
      q.exec(self, kernargs, global_size, local_size)

    q.signal(self.dev.timeline_signal, self.dev.next_timeline()).submit(self.dev)

    if wait: self.dev.synchronize()
    return (float(sig_en.timestamp - sig_st.timestamp) / 1e6) if wait else None

class HCQCompiled(Compiled, Generic[SignalType]):
  """
  A base class for devices compatible with the HCQ (Hardware Command Queue) API.
  """
  peer_groups: dict[str, list[HCQCompiled]] = collections.defaultdict(list)
  signal_pages: dict[str, list[HCQBuffer]] = collections.defaultdict(list) # per peer group
  signal_pool: dict[str, list[HCQBuffer]] = collections.defaultdict(list) # per peer group
  cpu_devices: list[HCQCompiled] = []

  def __init__(self, device:str, allocator:HCQAllocatorBase, renderer:Renderer, compiler:Compiler, runtime, signal_t:Type[SignalType],
               comp_queue_t:Callable[[], HWQueue], copy_queue_t:Callable[[], HWQueue]|None=None, kernargs_size=(16 << 20), sigalloc_size=0x1000):
    self.device_id:int = int(device.split(":")[1]) if ":" in device else 0

    from tinygrad.runtime.graph.hcq import HCQGraph
    super().__init__(device, allocator, renderer, compiler, runtime, HCQGraph)

    # TODO: peer logic is determined based on device name.
    self.peer_group = device.split(":")[0]
    HCQCompiled.peer_groups[self.peer_group].append(self)

    # Map signals if any
    for sig_page in HCQCompiled.signal_pages[self.peer_group]: cast(HCQAllocator, self.allocator).map(sig_page)

    self.sigalloc_size = sigalloc_size
    self.signal_t, self.hw_compute_queue_t, self.hw_copy_queue_t = signal_t, comp_queue_t, copy_queue_t
    self.timeline_value:int = 1
    self.timeline_signal, self._shadow_timeline_signal = self.new_signal(value=0, is_timeline=True), self.new_signal(value=0, is_timeline=True)
    self.sig_prof_records:list[tuple[HCQSignal, HCQSignal, str, bool]] = []

    self.kernargs_buf:HCQBuffer = self.allocator.alloc(kernargs_size, BufferSpec(cpu_access=True))
    self.kernargs_offset_allocator:BumpAllocator = BumpAllocator(self.kernargs_buf.size, wrap=True)

    if self._is_cpu(): HCQCompiled.cpu_devices.append(self)

  def synchronize(self):
    # If we have any work on CPU devices, need to synchronize them. This is just an optimization to release GIL allowing to finish faster.
    if not self._is_cpu():
      for dev in HCQCompiled.cpu_devices: dev.synchronize()

    try: self.timeline_signal.wait(self.timeline_value - 1)
    except RuntimeError as e:
      if hasattr(self, 'on_device_hang'): self.on_device_hang()
      else: raise e

    if self.timeline_value > (1 << 31): self._wrap_timeline_signal()
    if PROFILE:
      Compiled.profile_events += [ProfileRangeEvent(self.device, name, st.timestamp, en.timestamp, cp) for st,en,name,cp in self.sig_prof_records]
      self.sig_prof_records = []

  def next_timeline(self):
    self.timeline_value += 1
    return self.timeline_value - 1

  def new_signal(self, **kwargs) -> SignalType:
    if not HCQCompiled.signal_pool[pg:=self.peer_group]:
      HCQCompiled.signal_pages[pg].append(alc:=self.allocator.alloc(self.sigalloc_size, BufferSpec(host=True, uncached=True, cpu_access=True)))
      HCQCompiled.signal_pool[pg] += [alc.offset(offset=off, size=16) for off in range(0, alc.size, 16)]
      for dev in HCQCompiled.peer_groups[pg]: cast(HCQAllocator, dev.allocator).map(alc)
    return self.signal_t(base_buf=HCQCompiled.signal_pool[pg].pop(), owner=self, **kwargs)

  def _at_profile_finalize(self):
    self.synchronize() # Expect device to be synchronizes

    def _sync(d:HCQCompiled, q_t:Callable[[], HWQueue]):
      q_t().timestamp(d.timeline_signal).signal(d.timeline_signal, d.next_timeline()).submit(d)
      st = time.perf_counter_ns()
      d.timeline_signal.wait(d.timeline_value - 1)  # average of the two
      et = time.perf_counter_ns()
      return (decimal.Decimal(et+st) / 2000) - d.timeline_signal.timestamp

    gpu2cpu_compute_time_diff = statistics.median([_sync(self, self.hw_compute_queue_t) for _ in range(40)])
    if self.hw_copy_queue_t is None: gpu2cpu_copy_time_diff = decimal.Decimal(0)
    else: gpu2cpu_copy_time_diff = statistics.median([_sync(self, self.hw_copy_queue_t) for _ in range(40)])
    Compiled.profile_events += [ProfileDeviceEvent(self.device, gpu2cpu_compute_time_diff, gpu2cpu_copy_time_diff)]

  def _wrap_timeline_signal(self):
    self.timeline_signal, self._shadow_timeline_signal, self.timeline_value = self._shadow_timeline_signal, self.timeline_signal, 1
    self.timeline_signal.value = 0
    cast(HCQAllocatorBase, self.allocator).b_timeline = [0] * len(cast(HCQAllocatorBase, self.allocator).b)

  def _realloc(self, oldbuf:HCQBuffer|None, new_size:int, options:BufferSpec|None=None) -> tuple[HCQBuffer, bool]:
    if oldbuf is not None: self.allocator.free(oldbuf, oldbuf.size, options=options)
    try: buf, realloced = self.allocator.alloc(new_size, options=options), True
    except MemoryError: buf, realloced = self.allocator.alloc(oldbuf.size if oldbuf is not None else new_size, options=options), False
    return buf, realloced

  def _select_iface(self, *ifaces:Type):
    errs:str = ""
    if val:=getenv(f'{type(self).__name__[:-6].upper()}_IFACE', ""): ifaces = tuple(x for x in ifaces if x.__name__.startswith(val.upper()))
    for iface_t in ifaces:
      try: return iface_t(self, self.device_id)
      except Exception: errs += f"\n{iface_t.__name__}: {traceback.format_exc()}"
    raise RuntimeError(f"Cannot find a usable interface for {type(self).__name__[:-6]}:{self.device_id}:\n{errs}")

  def _is_cpu(self) -> bool: return hasattr(self, 'device') and self.device.split(":")[0] in ("CPU", "LLVM")

  def finalize(self):
    try: self.synchronize() # Try to finalize device in any case.
    except RuntimeError as e: print(f"{self.device} synchronization failed before finalizing: {e}")

    # If the device has an interface, call its device_fini method to clean up resources.
    if hasattr(self, 'iface') and hasattr(self.iface, 'device_fini'): self.iface.device_fini()

class HCQBuffer:
  def __init__(self, va_addr:sint, size:int, texture_info:Any=None, meta:Any=None, _base:HCQBuffer|None=None, view:MMIOInterface|None=None,
               owner:HCQCompiled|None=None):
    self.va_addr, self.size, self.texture_info, self.meta, self._base, self.view = va_addr, size, texture_info, meta, _base, view
    self._devs, self.owner = ([owner] if owner is not None else []), owner
    self._mappings:dict[HCQCompiled, HCQBuffer] = {} # mapping to the other devices

  def offset(self, offset:int=0, size:int|None=None) -> HCQBuffer:
    return HCQBuffer(self.va_addr+offset, size or (self.size - offset), owner=self.owner, texture_info=self.texture_info, meta=self.meta,
      _base=self._base or self, view=(self.view.view(offset=offset, size=size) if self.view is not None else None))

  def cpu_view(self) -> MMIOInterface:
    assert self.view is not None, "buffer has no cpu_view"
    return self.view

  @property
  def mappings(self): return self._mappings if self._base is None else self._base._mappings

  @property
  def mapped_devs(self): return self._devs if self._base is None else self._base._devs

class HCQAllocatorBase(LRUAllocator[HCQDeviceType], Generic[HCQDeviceType]):
  """
  A base allocator class compatible with the HCQ (Hardware Command Queue) API.

  This class implements basic copy operations following the HCQ API, utilizing both types of `HWQueue`.
  """

  def __init__(self, dev:HCQDeviceType, batch_size:int=(2 << 20), batch_cnt:int=32, copy_bufs=None, max_copyout_size:int|None=None):
    super().__init__(dev)
    self.b = copy_bufs or [self._alloc(batch_size, BufferSpec(host=True)) for _ in range(batch_cnt)]
    self.b_timeline, self.b_next, self.max_copyout_size = [0] * len(self.b), 0, max_copyout_size

  def map(self, buf:HCQBuffer):
    if self.dev in buf.mapped_devs: return
    if buf.owner is None: raise RuntimeError(f"map failed: buffer {buf.va_addr} has no owner, it's a virtual buffer")
    if not hasattr(self, '_map'): raise NotImplementedError("map failed: no method implemented")

    # Since it's unified memory space, any buffer mapping is valid for all devices after successful map.
    # Devices can save mappings and internal metadata as a new buffer.
    if (mb:=self._map(buf)) is not None: buf.mappings[self.dev] = mb
    buf.mapped_devs.append(self.dev)

  def _offset(self, buf, size:int, offset:int) -> HCQBuffer: return buf.offset(offset=offset, size=size)

class HCQAllocator(HCQAllocatorBase, Generic[HCQDeviceType]):
  def _copyin(self, dest:HCQBuffer, src:memoryview):
    assert self.dev.hw_copy_queue_t is not None
    with hcq_profile(self.dev, queue_type=self.dev.hw_copy_queue_t, desc=f"TINY -> {self.dev.device}", enabled=PROFILE):
      for i in range(0, src.nbytes, self.b[0].size):
        self.b_next = (self.b_next + 1) % len(self.b)
        self.dev.timeline_signal.wait(self.b_timeline[self.b_next])

        lsize = min(self.b[self.b_next].size, src.nbytes - i)
        self.b[self.b_next].cpu_view().view(size=lsize, fmt='B')[:] = src[i:i+lsize]
        self.dev.hw_copy_queue_t().wait(self.dev.timeline_signal, self.dev.timeline_value - 1) \
                                  .copy(dest.va_addr+i, self.b[self.b_next].va_addr, lsize) \
                                  .signal(self.dev.timeline_signal, self.dev.next_timeline()).submit(self.dev)
        self.b_timeline[self.b_next] = self.dev.timeline_value - 1

  def copy_from_disk(self, dest:HCQBuffer, src, size):
    def _get_temp_buf():
      # Check if the next buffer is safe to be used (its signal has passed) and reserve it.
      if self.b_timeline[(self.b_next + 1) % len(self.b)] <= self.dev.timeline_signal.value:
        self.b_timeline[(self.b_next + 1) % len(self.b)], self.b_next = (1 << 64), (self.b_next + 1) % len(self.b)
        return (self.b[self.b_next].va_addr, self.b_next)
      return None

    assert self.dev.hw_copy_queue_t is not None
    with hcq_profile(self.dev, queue_type=self.dev.hw_copy_queue_t, desc=f"DISK -> {self.dev.device}", enabled=PROFILE):
      for (batch_info, dst_off, src_off, copy_size) in src.device.allocator._copyout_sharded(src, size, _get_temp_buf, seg_len=self.b[0].size):
        self.dev.hw_copy_queue_t().wait(self.dev.timeline_signal, self.dev.timeline_value - 1) \
                                  .copy(dest.va_addr + dst_off, batch_info[0] + src_off, copy_size) \
                                  .signal(self.dev.timeline_signal, self.dev.next_timeline()).submit(self.dev)
        self.b_timeline[batch_info[1]] = self.dev.timeline_value - 1

  def _copyout(self, dest:memoryview, src:HCQBuffer):
    self.dev.synchronize()

    assert self.dev.hw_copy_queue_t is not None
    with hcq_profile(self.dev, queue_type=self.dev.hw_copy_queue_t, desc=f"{self.dev.device} -> TINY", enabled=PROFILE):
      for i in range(0, dest.nbytes, cp_size:=(self.max_copyout_size or self.b[0].size)):
        self.dev.hw_copy_queue_t().wait(self.dev.timeline_signal, self.dev.timeline_value - 1) \
                                  .copy(self.b[0].va_addr, src.va_addr+i, lsize:=min(cp_size, dest.nbytes-i)) \
                                  .signal(self.dev.timeline_signal, self.dev.next_timeline()).submit(self.dev)
        self.dev.timeline_signal.wait(self.dev.timeline_value - 1)
        dest[i:i+lsize] = self.b[0].cpu_view().view(size=lsize, fmt='B')[:]

  def _transfer(self, dest:HCQBuffer, src:HCQBuffer, sz:int, src_dev:HCQDeviceType, dest_dev:HCQDeviceType):
    cast(HCQAllocator, src_dev.allocator).map(dest)

    assert src_dev.hw_copy_queue_t is not None
    with hcq_profile(src_dev, queue_type=src_dev.hw_copy_queue_t, desc=f"{src_dev.device} -> {dest_dev.device}", enabled=PROFILE):
      src_dev.hw_copy_queue_t().wait(src_dev.timeline_signal, src_dev.timeline_value - 1) \
                               .wait(dest_dev.timeline_signal, dest_dev.timeline_value - 1) \
                               .copy(dest.va_addr, src.va_addr, sz) \
                               .signal(src_dev.timeline_signal, src_dev.next_timeline()).submit(src_dev)

    if src_dev != dest_dev:
      dest_dev.hw_compute_queue_t().wait(src_dev.timeline_signal, src_dev.timeline_value - 1) \
                                   .wait(dest_dev.timeline_signal, dest_dev.timeline_value - 1) \
                                   .signal(dest_dev.timeline_signal, dest_dev.next_timeline()).submit(dest_dev)
