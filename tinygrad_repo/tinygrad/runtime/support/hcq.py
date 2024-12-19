from __future__ import annotations
from typing import List, Optional, Dict, Tuple, cast, Type, Union, TypeVar, Generic, Any
import contextlib, decimal, statistics, random, json, atexit, time, ctypes, array
from tinygrad.helpers import PROFILEPATH, PROFILE, from_mv, getenv, to_mv, round_up
from tinygrad.renderer import Renderer
from tinygrad.device import BufferSpec, Compiler, Compiled, LRUAllocator
from tinygrad.ops import sym_infer, sint, Variable

# **************** for HCQ Compatible Devices ****************

SignalType = TypeVar('SignalType', bound='HCQSignal')
DeviceType = TypeVar('DeviceType', bound='HCQCompiled')
ProgramType = TypeVar('ProgramType', bound='HCQProgram')
ArgsStateType = TypeVar('ArgsStateType', bound='HCQArgsState')
QueueType = TypeVar('QueueType', bound='HWQueue')

class BumpAllocator:
  def __init__(self, size:int, start:int=0, wrap:bool=True): self.size, self.ptr, self.start_off, self.wrap = size, 0, start, wrap
  def alloc(self, size:int, alignment:int=1) -> int:
    if round_up(self.ptr, alignment) + size > self.size:
      if not self.wrap: raise RuntimeError("Out of memory")
      self.ptr = 0
    self.ptr = (res:=round_up(self.ptr, alignment)) + size
    return res + self.start_off

class HWQueue(Generic[SignalType, DeviceType, ProgramType, ArgsStateType]):
  """
  A base class for hardware command queues in the HCQ (Hardware Command Queue) API.
  """

  def __init__(self):
    self._q:Any = []
    self.binded_device:Optional[DeviceType] = None
    self.q_sints:List[Tuple[int, int]] = []
    self.mv_sints:List[Tuple[memoryview, int, int, Optional[int]]] = []
    self.syms:List[sint] = []
    self._prev_resolved_syms:List[Optional[int]] = []

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
      if isinstance(v, int): self._q.append(v)
      else:
        self.q_sints.append((len(self._q), self._new_sym(v)))
        self._q.append(0xbadc0ded)

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

  def exec(self, prg:ProgramType, args_state:ArgsStateType, global_size:Tuple[sint, ...], local_size:Tuple[sint, ...]):
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

  def bind(self, dev:DeviceType):
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
    for vals, ptr, fmt in args_state.bind_data: self.bind_sints_to_ptr(*vals, ptr=ptr, fmt=fmt)

  def bind_sints(self, *vals:sint, struct:ctypes.Structure, start_field:str, fmt, mask:Optional[int]=None):
    self.bind_sints_to_ptr(*vals, ptr=ctypes.addressof(struct) + getattr(type(struct), start_field).offset, fmt=fmt, mask=mask)

  def bind_sints_to_ptr(self, *vals:sint, ptr:int, fmt, mask:Optional[int]=None):
    mv = to_mv(ptr, 8*len(vals)).cast(fmt)
    for i, val in enumerate(vals):
      if isinstance(val, int): mv[i] = val if mask is None else ((mv[i] & ~mask) | val)
      else: self.mv_sints.append((mv, i, self._new_sym(val), mask))

  def _apply_var_vals(self, var_vals:Dict[Variable, int]):
    resolved_syms = [sym_infer(sym, var_vals) for sym in self.syms]

    for off, sym_idx in self.q_sints:
      if self._prev_resolved_syms[sym_idx] == resolved_syms[sym_idx]: continue
      self._q[off] = resolved_syms[sym_idx]

    for mv, off, sym_idx, mask in self.mv_sints:
      if self._prev_resolved_syms[sym_idx] == resolved_syms[sym_idx]: continue
      mv[off] = resolved_syms[sym_idx] if mask is None else ((mv[off] & ~mask) | resolved_syms[sym_idx])

    self._prev_resolved_syms = cast(List[Optional[int]], resolved_syms)

  def submit(self, dev:DeviceType, var_vals:Optional[Dict[Variable, int]]=None):
    """
    Submits the command queue to a specific device for execution.

    Args:
      dev: The device to submit the queue to
    """

    if var_vals is not None: self._apply_var_vals(var_vals)
    self._submit(dev)
    return self
  def _submit(self, dev:DeviceType): raise NotImplementedError("need _submit")

class HCQSignal(Generic[DeviceType]):
  def __init__(self, base_addr:sint=0, value:int=0, timeline_for_device:Optional[DeviceType]=None, timestamp_divider=1, value_off=0, timestamp_off=8):
    self.base_addr, self.value_addr, self.timestamp_addr = base_addr, base_addr+value_off, base_addr+timestamp_off
    self.timestamp_divider:decimal.Decimal = decimal.Decimal(timestamp_divider)
    self.timeline_for_device:Optional[DeviceType] = timeline_for_device

    if isinstance(base_addr, int):
      self.value_mv, self.timestamp_mv = to_mv(self.value_addr, 8).cast('Q'), to_mv(self.timestamp_addr, 8).cast('Q')
      self.value_mv[0] = value

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
      timeout: Maximum time to wait in milliseconds. Defaults to 10s.
    """
    start_time = int(time.perf_counter() * 1000)
    while self.value < value and (time_spent:=int(time.perf_counter() * 1000) - start_time) < timeout:
      self._sleep(time_spent)
    if self.value < value: raise RuntimeError(f"Wait timeout: {timeout} ms! (the signal is not set to {value}, but {self.value})")

@contextlib.contextmanager
def hcq_profile(dev:HCQCompiled, enabled, desc, queue_type:Optional[Type[HWQueue]]=None, queue:Optional[HWQueue]=None):
  st, en = (dev.signal_t(), dev.signal_t()) if enabled else (None, None)

  if enabled and queue is not None: queue.timestamp(st)
  elif enabled:
    assert queue_type is not None
    queue_type().wait(dev.timeline_signal, dev.timeline_value - 1).timestamp(st).signal(dev.timeline_signal, dev.timeline_value).submit(dev)
    dev.timeline_value += 1

  try: yield (st, en)
  finally:
    if enabled and queue is not None: queue.timestamp(en)
    elif enabled:
      assert queue_type is not None
      queue_type().wait(dev.timeline_signal, dev.timeline_value - 1).timestamp(en).signal(dev.timeline_signal, dev.timeline_value).submit(dev)
      dev.timeline_value += 1

    if enabled and PROFILE: dev.sig_prof_records.append((cast(HCQSignal, st), cast(HCQSignal, en), desc, queue_type is dev.hw_copy_queue_t))

class HCQArgsState(Generic[ProgramType]):
  def __init__(self, ptr:int, prg:ProgramType, bufs:Tuple[HCQBuffer, ...], vals:Tuple[sint, ...]=()):
    self.ptr, self.prg = ptr, prg
    self.bind_data:List[Tuple[Tuple[sint, ...], int, str]] = []

  def bind_sints_to_ptr(self, *vals:sint, ptr:int, fmt): self.bind_data.append((vals, ptr, fmt))

class CLikeArgsState(HCQArgsState[ProgramType]):
  def __init__(self, ptr:int, prg:ProgramType, bufs:Tuple[HCQBuffer, ...], vals:Tuple[sint, ...]=(), prefix:Optional[List[int]]=None):
    super().__init__(ptr, prg, bufs, vals=vals)

    if prefix is not None: to_mv(self.ptr, len(prefix) * 4).cast('I')[:] = array.array('I', prefix)

    self.bind_sints_to_ptr(*[b.va_addr for b in bufs], ptr=self.ptr + len(prefix or []) * 4, fmt='Q')
    self.bind_sints_to_ptr(*vals, ptr=self.ptr + len(prefix or []) * 4 + len(bufs) * 8, fmt='I')

class HCQProgram(Generic[DeviceType]):
  def __init__(self, args_state_t:Type[HCQArgsState], dev:DeviceType, name:str, kernargs_alloc_size:int):
    self.args_state_t, self.dev, self.name, self.kernargs_alloc_size = args_state_t, dev, name, kernargs_alloc_size

  def fill_kernargs(self, bufs:Tuple[HCQBuffer, ...], vals:Tuple[int, ...]=(), kernargs_ptr:Optional[int]=None) -> HCQArgsState:
    """
    Fills arguments for the kernel, optionally allocating space from the device if `kernargs_ptr` is not provided.
    Args:
      bufs: Buffers to be written to kernel arguments.
      vals: Values to be written to kernel arguments.
      kernargs_ptr: Optional pointer to pre-allocated kernel arguments memory.
    Returns:
      Arguments state with the given buffers and values set for the program.
    """
    return self.args_state_t(kernargs_ptr or self.dev.kernargs_alloctor.alloc(self.kernargs_alloc_size), self, bufs, vals=vals)

  def __call__(self, *bufs:HCQBuffer, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1),
               vals:Tuple[int, ...]=(), wait:bool=False) -> Optional[float]:
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

    q.signal(self.dev.timeline_signal, self.dev.timeline_value).submit(self.dev)
    self.dev.timeline_value += 1

    if wait: self.dev.synchronize()
    return (float(sig_en.timestamp - sig_st.timestamp) / 1e6) if wait else None

class ProfileLogger:
  writers: int = 0
  mjson: List[Dict] = []
  actors: Dict[Union[str, Tuple[str, str]], int] = {}

  def __init__(self): self.events, self.deps, ProfileLogger.writers = [], [], ProfileLogger.writers + 1

  def add_event(self, ev_name, ev_start, ev_end, actor, subactor=None, args=None): self.events += [(ev_name, ev_start, ev_end, actor, subactor, args)]

  def _ensure_actor(self, actor_name, subactor_name):
    if actor_name not in self.actors:
      self.actors[actor_name] = (pid:=len(self.actors))
      self.mjson.append({"name": "process_name", "ph": "M", "pid": pid, "args": {"name": actor_name}})

    if (subactor_key:=(actor_name,subactor_name)) not in self.actors:
      self.actors[subactor_key] = (tid:=len(self.actors))
      self.mjson.append({"name": "thread_name", "ph": "M", "pid": self.actors[actor_name], "tid":tid, "args": {"name": subactor_name}})

    return self.actors[actor_name], self.actors.get(subactor_key, -1)

  def __del__(self):
    # perfetto json docs: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
    for name, st, et, actor_name, subactor_name, args in self.events:
      pid, tid = self._ensure_actor(actor_name,subactor_name)
      args = {k: (v if v.__class__ is str else v(et-st)) for k, v in args.items()} if args is not None else None
      self.mjson.append({"name": name, "ph": "X", "pid": pid, "tid": tid, "ts": st, "dur": et-st, "args": args})

    for en,st,dep_actor_name,dep_subactor_name,actor_name,subactor_name in self.deps:
      dep_pid, dep_tid = self._ensure_actor(dep_actor_name,dep_subactor_name)
      pid, tid = self._ensure_actor(actor_name,subactor_name)
      self.mjson.append({"ph": "s", "pid": dep_pid, "tid": dep_tid, "id": len(self.mjson), "ts": en, "bp": "e"})
      self.mjson.append({"ph": "f", "pid": pid, "tid": tid, "id": len(self.mjson)-1, "ts": st, "bp": "e"})

    ProfileLogger.writers -= 1
    if ProfileLogger.writers == 0 and len(self.mjson) > 0:
      with open(PROFILEPATH.value, "w") as f: f.write(json.dumps({"traceEvents": self.mjson}))
      print(f"Saved profile to {PROFILEPATH.value}. Use https://ui.perfetto.dev/ to open it.")

class HCQCompiled(Compiled, Generic[SignalType]):
  """
  A base class for devices compatible with the HCQ (Hardware Command Queue) API.
  """
  devices: List[HCQCompiled] = []
  gpu2cpu_copy_time_diff: decimal.Decimal = decimal.Decimal('nan')
  gpu2cpu_compute_time_diff: decimal.Decimal = decimal.Decimal('nan')

  def __init__(self, device:str, allocator:HCQAllocatorBase, renderer:Renderer, compiler:Compiler, runtime, signal_t:Type[SignalType],
               comp_queue_t:Type[HWQueue], copy_queue_t:Optional[Type[HWQueue]]):
    self.device_id:int = int(device.split(":")[1]) if ":" in device else 0
    self.signal_t, self.hw_compute_queue_t, self.hw_copy_queue_t = signal_t, comp_queue_t, copy_queue_t
    self.timeline_value:int = 1
    self.timeline_signal:SignalType = self.signal_t(value=0, timeline_for_device=self)
    self._shadow_timeline_signal:SignalType = self.signal_t(value=0, timeline_for_device=self)
    self.sig_prof_records:List[Tuple[HCQSignal, HCQSignal, str, bool]] = []
    self.raw_prof_records:List[Tuple[decimal.Decimal, decimal.Decimal, str, bool, Optional[Dict]]] = []
    self.dep_prof_records:List[Tuple[decimal.Decimal, decimal.Decimal, HCQCompiled, bool, decimal.Decimal, decimal.Decimal, HCQCompiled, bool]] = []
    if PROFILE: self._prof_setup()

    from tinygrad.runtime.graph.hcq import HCQGraph
    super().__init__(device, allocator, renderer, compiler, runtime, HCQGraph)

    self.kernargs_page:HCQBuffer = self.allocator.alloc(16 << 20, BufferSpec(cpu_access=True))
    self.kernargs_alloctor:BumpAllocator = BumpAllocator(self.kernargs_page.size, start=cast(int, self.kernargs_page.va_addr), wrap=True)
    self.devices.append(self)

  def synchronize(self):
    try: self.timeline_signal.wait(self.timeline_value - 1)
    except RuntimeError as e:
      if hasattr(self, 'on_device_hang'): self.on_device_hang()
      else: raise e

    if self.timeline_value > (1 << 31): self._wrap_timeline_signal()
    if PROFILE:
      self.raw_prof_records += [(st.timestamp, en.timestamp, name, is_cp, None) for st, en, name, is_cp in self.sig_prof_records]
      self.sig_prof_records = []

  def _ensure_shared_time_base(self):
    if not self.gpu2cpu_compute_time_diff.is_nan(): return

    def _sync_cpu_queue(d:HCQCompiled, q_t:Type[HWQueue]):
      q_t().timestamp(d.timeline_signal).signal(d.timeline_signal, d.timeline_value).submit(d)
      d.timeline_value += 1
      st = time.perf_counter_ns()
      d.timeline_signal.wait(d.timeline_value - 1)  # average of the two
      et = time.perf_counter_ns()
      return (decimal.Decimal(et+st) / 2000) - d.timeline_signal.timestamp

    # randomly sample the timing from GPU to CPU
    choices: List = [(d, d.hw_compute_queue_t, []) for d in self.devices]
    choices += [(d, d.hw_copy_queue_t, []) for d in self.devices if d.hw_copy_queue_t is not None]
    for _ in range(100*len(self.devices)):
      d,q,l = random.choice(choices)
      l.append(_sync_cpu_queue(d,q))
    for d,q,l in choices:
      if q == d.hw_compute_queue_t: d.gpu2cpu_compute_time_diff = statistics.median(l)
      if q == d.hw_copy_queue_t: d.gpu2cpu_copy_time_diff = statistics.median(l)

    def _sync_gpu_to_gpu_queue(d1:HCQCompiled, d2:HCQCompiled, q1_t:Type[HWQueue], q2_t:Type[HWQueue]):
      q1_t().signal(d1.timeline_signal, d1.timeline_value).wait(d2.timeline_signal, d2.timeline_value) \
            .timestamp(d1.timeline_signal).signal(d1.timeline_signal, d1.timeline_value+1).submit(d1)
      q2_t().signal(d2.timeline_signal, d2.timeline_value).wait(d1.timeline_signal, d1.timeline_value) \
            .timestamp(d2.timeline_signal).signal(d2.timeline_signal, d2.timeline_value+1).submit(d2)
      d1.timeline_value += 2
      d2.timeline_value += 2
      d1.timeline_signal.wait(d1.timeline_value - 1)
      d2.timeline_signal.wait(d2.timeline_value - 1)
      return d2.timeline_signal.timestamp - d1.timeline_signal.timestamp

    # then test it by timing the GPU to GPU times
    jitter_matrix = [[float('nan')]*len(self.devices) for _ in range(len(self.devices))]
    for i1, d1 in enumerate(self.devices):
      for i2, d2 in enumerate(self.devices):
        if d1 == d2: continue
        d1_to_d2 = statistics.median(_sync_gpu_to_gpu_queue(d1, d2, d1.hw_compute_queue_t, d2.hw_compute_queue_t) - \
                                     _sync_gpu_to_gpu_queue(d2, d1, d2.hw_compute_queue_t, d1.hw_compute_queue_t) for _ in range(20)) / 2
        jitter_matrix[i1][i2] = d1_to_d2 - (d1.gpu2cpu_compute_time_diff - d2.gpu2cpu_compute_time_diff)
    print("pairwise clock jitter matrix (us):\n" + '\n'.join([''.join([f'{float(item):8.3f}' for item in row]) for row in jitter_matrix]))

  def _gpu2cpu_time(self, gpu_time:decimal.Decimal, is_copy:bool) -> float:
    """
    Translates local gpu time (timestamp) into global cpu time.
    """
    self._ensure_shared_time_base()
    return float(gpu_time + (self.gpu2cpu_copy_time_diff if is_copy else self.gpu2cpu_compute_time_diff))

  def _prof_setup(self):
    if hasattr(self, 'profile_logger'): return
    atexit.register(self._prof_finalize)
    self.profile_logger = ProfileLogger()

  def _prof_finalize(self):
    qname = ["COMPUTE", "DMA"]

    # Sync to be sure all events on the device are recorded.
    self.synchronize()

    for st, en, name, is_cp, args in self.raw_prof_records:
      self.profile_logger.events += [(name, self._gpu2cpu_time(st, is_cp), self._gpu2cpu_time(en, is_cp), self.device, qname[is_cp], args)]
    for a_st, a_en, a_dev, a_is_copy, b_st, b_en, b_dev, b_is_copy in self.dep_prof_records:
      # Perfetto connects nodes based on timing data, ensuring every choice is valid by averaging times to a midpoint.
      a_tm, b_tm = a_dev._gpu2cpu_time((a_st+a_en)/decimal.Decimal(2), a_is_copy), b_dev._gpu2cpu_time((b_st+b_en)/decimal.Decimal(2), b_is_copy)
      self.profile_logger.deps += [(a_tm, b_tm, a_dev.device, qname[a_is_copy], b_dev.device, qname[b_is_copy])]
    self.raw_prof_records, self.dep_prof_records = [], []

    # Remove the logger, this flushes all data written by the device.
    del self.profile_logger

  def _wrap_timeline_signal(self):
    self.timeline_signal, self._shadow_timeline_signal, self.timeline_value = self._shadow_timeline_signal, self.timeline_signal, 1
    self.timeline_signal.value = 0
    cast(HCQAllocatorBase, self.allocator).b_timeline = [0] * len(cast(HCQAllocatorBase, self.allocator).b)

class HCQBuffer:
  def __init__(self, va_addr:sint, size:int, texture_info:Any=None, meta:Any=None, _base:Optional[HCQBuffer]=None):
    self.va_addr, self.size, self.texture_info, self.meta, self._base = va_addr, size, texture_info, meta, _base

class HCQAllocatorBase(LRUAllocator, Generic[DeviceType]):
  """
  A base allocator class compatible with the HCQ (Hardware Command Queue) API.

  This class implements basic copy operations following the HCQ API, utilizing both types of `HWQueue`.
  """

  def __init__(self, dev:DeviceType, batch_size:int=(2 << 20), batch_cnt:int=32):
    self.dev:DeviceType = dev
    self.b = [self._alloc(batch_size, BufferSpec(host=True)) for _ in range(batch_cnt)]
    self.b_timeline, self.b_next = [0] * len(self.b), 0
    super().__init__()

  def map(self, buf:HCQBuffer): pass

  def _offset(self, buf, size:int, offset:int) -> HCQBuffer:
    return HCQBuffer(va_addr=buf.va_addr + offset, size=size, texture_info=buf.texture_info, meta=buf.meta, _base=buf._base or buf)

class HCQAllocator(HCQAllocatorBase, Generic[DeviceType]):
  def _copyin(self, dest:HCQBuffer, src:memoryview):
    assert self.dev.hw_copy_queue_t is not None
    with hcq_profile(self.dev, queue_type=self.dev.hw_copy_queue_t, desc=f"CPU -> {self.dev.device}", enabled=PROFILE):
      for i in range(0, src.nbytes, self.b[0].size):
        self.b_next = (self.b_next + 1) % len(self.b)
        self.dev.timeline_signal.wait(self.b_timeline[self.b_next])
        ctypes.memmove(self.b[self.b_next].va_addr, from_mv(src[i:]), lsize:=min(self.b[self.b_next].size, src.nbytes-i))
        self.dev.hw_copy_queue_t().wait(self.dev.timeline_signal, self.dev.timeline_value - 1) \
                                  .copy(dest.va_addr+i, self.b[self.b_next].va_addr, lsize) \
                                  .signal(self.dev.timeline_signal, self.dev.timeline_value).submit(self.dev)
        self.b_timeline[self.b_next] = self.dev.timeline_value
        self.dev.timeline_value += 1

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
                                  .signal(self.dev.timeline_signal, self.dev.timeline_value).submit(self.dev)
        self.b_timeline[batch_info[1]] = self.dev.timeline_value
        self.dev.timeline_value += 1

  def _copyout(self, dest:memoryview, src:HCQBuffer):
    self.dev.synchronize()

    assert self.dev.hw_copy_queue_t is not None
    with hcq_profile(self.dev, queue_type=self.dev.hw_copy_queue_t, desc=f"{self.dev.device} -> CPU", enabled=PROFILE):
      for i in range(0, dest.nbytes, self.b[0].size):
        self.dev.hw_copy_queue_t().wait(self.dev.timeline_signal, self.dev.timeline_value - 1) \
                                  .copy(self.b[0].va_addr, src.va_addr+i, lsize:=min(self.b[0].size, dest.nbytes-i)) \
                                  .signal(self.dev.timeline_signal, self.dev.timeline_value).submit(self.dev)
        self.dev.timeline_signal.wait(self.dev.timeline_value)
        self.dev.timeline_value += 1

        ctypes.memmove(from_mv(dest[i:]), self.b[0].va_addr, lsize)

  def _transfer(self, dest:HCQBuffer, src:HCQBuffer, sz:int, src_dev:DeviceType, dest_dev:DeviceType):
    cast(HCQAllocator, src_dev.allocator).map(dest)

    assert src_dev.hw_copy_queue_t is not None
    with hcq_profile(src_dev, queue_type=src_dev.hw_copy_queue_t, desc=f"{src_dev.device} -> {dest_dev.device}", enabled=PROFILE):
      src_dev.hw_copy_queue_t().wait(src_dev.timeline_signal, src_dev.timeline_value - 1) \
                               .wait(dest_dev.timeline_signal, dest_dev.timeline_value - 1) \
                               .copy(dest.va_addr, src.va_addr, sz) \
                               .signal(src_dev.timeline_signal, src_dev.timeline_value).submit(src_dev)
      src_dev.timeline_value += 1

    if src_dev != dest_dev:
      dest_dev.hw_compute_queue_t().wait(src_dev.timeline_signal, src_dev.timeline_value - 1) \
                                   .wait(dest_dev.timeline_signal, dest_dev.timeline_value - 1) \
                                   .signal(dest_dev.timeline_signal, dest_dev.timeline_value).submit(dest_dev)
      dest_dev.timeline_value += 1
