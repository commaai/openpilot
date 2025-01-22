from __future__ import annotations
import ctypes, functools, subprocess, io, atexit, collections, json
from typing import Tuple, TypeVar, List, Dict, Any
import tinygrad.runtime.autogen.hsa as hsa
from tinygrad.helpers import DEBUG, init_c_var, from_mv, round_up, to_mv, init_c_struct_t, getenv, PROFILE
from tinygrad.device import Compiled, Compiler, CompileError, BufferSpec, LRUAllocator
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.runtime.support.hsa import check, scan_agents, find_memory_pool, AQLQueue
from tinygrad.runtime.support.hip_comgr import compile_hip
if getenv("IOCTL"): import extra.hip_gpu_driver.hip_ioctl  # noqa: F401

class HSAProfiler:
  def __init__(self):
    self.tracked_signals = collections.defaultdict(list)
    self.collected_events: List[Tuple[Any, ...]] = []
    self.copy_timings = hsa.hsa_amd_profiling_async_copy_time_t()
    self.disp_timings = hsa.hsa_amd_profiling_dispatch_time_t()

  def track(self, signal, device, name, is_copy=False): self.tracked_signals[device].append((signal, name, is_copy))
  def process(self, device):
    # Process all tracked signals, should be called before any of tracked signals are reused.
    for sig,name,is_copy in self.tracked_signals[device]:
      if is_copy: check(hsa.hsa_amd_profiling_get_async_copy_time(sig, ctypes.byref(timings :=  self.copy_timings)))
      else: check(hsa.hsa_amd_profiling_get_dispatch_time(device.agent, sig, ctypes.byref(timings := self.disp_timings))) #type:ignore
      self.collected_events.append((device.device_id, 1 if is_copy else 0, name, timings.start, timings.end))
    self.tracked_signals.pop(device)

  def save(self, path):
    mjson = []
    for i in range(len(HSADevice.devices)):
      mjson.append({"name": "process_name", "ph": "M", "pid": i, "args": {"name": "HSA"}})
      mjson.append({"name": "thread_name", "ph": "M", "pid": i, "tid": 0, "args": {"name": "AQL"}})
      mjson.append({"name": "thread_name", "ph": "M", "pid": i, "tid": 1, "args": {"name": "SDMA"}})

    for dev_id,queue_id,name,st,et in self.collected_events:
      mjson.append({"name": name, "ph": "B", "pid": dev_id, "tid": queue_id, "ts": st*1e-3})
      mjson.append({"name": name, "ph": "E", "pid": dev_id, "tid": queue_id, "ts": et*1e-3})
    with open(path, "w") as f: f.write(json.dumps({"traceEvents": mjson}))
    print(f"Saved HSA profile to {path}")
Profiler = HSAProfiler()

class HSACompiler(Compiler):
  def __init__(self, arch:str):
    self.arch = arch
    super().__init__(f"compile_hip_{self.arch}")
  def compile(self, src:str) -> bytes:
    try: return compile_hip(src, self.arch)
    except RuntimeError as e: raise CompileError(e)

class HSAProgram:
  def __init__(self, device:HSADevice, name:str, lib:bytes):
    self.device, self.name, self.lib = device, name, lib

    if DEBUG >= 6:
      asm = subprocess.check_output(["/opt/rocm/llvm/bin/llvm-objdump", '-d', '-'], input=lib)
      print('\n'.join([x for x in asm.decode('utf-8').split("\n") if 's_code_end' not in x]))

    self.exec = init_c_var(hsa.hsa_executable_t(), lambda x: check(hsa.hsa_executable_create_alt(hsa.HSA_PROFILE_FULL, hsa.HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, None, ctypes.byref(x)))) # noqa: E501
    self.code_reader = init_c_var(hsa.hsa_code_object_reader_t(),
                                  lambda x: check(hsa.hsa_code_object_reader_create_from_memory(lib, len(lib), ctypes.byref(x))))
    check(hsa.hsa_executable_load_agent_code_object(self.exec, self.device.agent, self.code_reader, None, None))
    check(hsa.hsa_executable_freeze(self.exec, None))

    self.kernel = init_c_var(hsa.hsa_executable_symbol_t(), lambda x: check(hsa.hsa_executable_get_symbol_by_name(self.exec, (name+".kd").encode("utf-8"), ctypes.byref(self.device.agent), ctypes.byref(x)))) # noqa: E501
    self.handle = init_c_var(ctypes.c_uint64(), lambda x: check(hsa.hsa_executable_symbol_get_info(self.kernel, hsa.HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, ctypes.byref(x)))) # noqa: E501
    self.kernargs_segment_size = init_c_var(ctypes.c_uint32(), lambda x: check(hsa.hsa_executable_symbol_get_info(self.kernel, hsa.HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, ctypes.byref(x)))).value # noqa: E501
    self.group_segment_size = init_c_var(ctypes.c_uint32(), lambda x: check(hsa.hsa_executable_symbol_get_info(self.kernel, hsa.HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, ctypes.byref(x)))).value # noqa: E501
    self.private_segment_size = init_c_var(ctypes.c_uint32(), lambda x: check(hsa.hsa_executable_symbol_get_info(self.kernel, hsa.HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, ctypes.byref(x)))).value # noqa: E501

  def __del__(self):
    self.device.synchronize()
    if hasattr(self, 'code_reader'): check(hsa.hsa_code_object_reader_destroy(self.code_reader))
    if hasattr(self, 'exec'): check(hsa.hsa_executable_destroy(self.exec))

  def __call__(self, *args, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    if not hasattr(self, "args_struct_t"):
      self.args_struct_t = init_c_struct_t(tuple([(f'f{i}', ctypes.c_void_p) for i in range(len(args))] +
                                                 [(f'v{i}', ctypes.c_int) for i in range(len(vals))]))
      if ctypes.sizeof(self.args_struct_t) != self.kernargs_segment_size:
        raise RuntimeError(f"HSAProgram.__call__: incorrect args struct size {ctypes.sizeof(self.args_struct_t)} != {self.kernargs_segment_size}")

    kernargs = None
    if self.kernargs_segment_size > 0:
      kernargs = self.device.alloc_kernargs(self.kernargs_segment_size)
      args_st = self.args_struct_t.from_address(kernargs)
      for i in range(len(args)): args_st.__setattr__(f'f{i}', args[i])
      for i in range(len(vals)): args_st.__setattr__(f'v{i}', vals[i])
      self.device.flush_hdp()

    signal = self.device.alloc_signal(reusable=True) if wait or PROFILE else None
    self.device.hw_queue.submit_kernel(self, global_size, local_size, kernargs, completion_signal=signal)
    if PROFILE: Profiler.track(signal, self.device, self.name)
    if wait:
      hsa.hsa_signal_wait_scacquire(signal, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_ACTIVE)
      check(hsa.hsa_amd_profiling_get_dispatch_time(self.device.agent, signal, ctypes.byref(timings := hsa.hsa_amd_profiling_dispatch_time_t())))
      return (timings.end - timings.start) * self.device.clocks_to_time

T = TypeVar("T")
CHUNK_SIZE, PAGE_SIZE = 256*1024*1024, 0x1000
class HSAAllocator(LRUAllocator):
  def __init__(self, device:HSADevice):
    self.device = device
    super().__init__()

  def _alloc(self, size:int, options:BufferSpec):
    if options.host:
      check(hsa.hsa_amd_memory_pool_allocate(HSADevice.cpu_mempool, size, 0, ctypes.byref(mem := ctypes.c_void_p())))
      check(hsa.hsa_amd_agents_allow_access(2, (hsa.hsa_agent_t*2)(HSADevice.cpu_agent, self.device.agent), None, mem))
      return mem.value
    c_agents = (hsa.hsa_agent_t * len(HSADevice.agents[hsa.HSA_DEVICE_TYPE_GPU]))(*HSADevice.agents[hsa.HSA_DEVICE_TYPE_GPU])
    check(hsa.hsa_amd_memory_pool_allocate(self.device.gpu_mempool, size, 0, ctypes.byref(buf := ctypes.c_void_p())))
    check(hsa.hsa_amd_agents_allow_access(len(HSADevice.agents[hsa.HSA_DEVICE_TYPE_GPU]), c_agents, None, buf))
    return buf.value

  def _free(self, opaque:T, options:BufferSpec):
    HSADevice.synchronize_system()
    check(hsa.hsa_amd_memory_pool_free(opaque))

  def _copyin(self, dest:T, src: memoryview):
    # Async copyin sync model uses barriers on the main hw queue, since barriers are guaranteed to execute in order with all other packets.
    self.device.hw_queue.submit_barrier([], sync_signal := self.device.alloc_signal(reusable=True))
    mem = self._alloc(src.nbytes, BufferSpec(host=True))
    ctypes.memmove(mem, from_mv(src), src.nbytes)
    check(hsa.hsa_amd_memory_async_copy_on_engine(dest, self.device.agent, mem, HSADevice.cpu_agent, src.nbytes, 1, ctypes.byref(sync_signal),
                                                  copy_signal := self.device.alloc_signal(reusable=True), hsa.HSA_AMD_SDMA_ENGINE_0, True))
    self.device.hw_queue.submit_barrier([copy_signal])
    self.device.delayed_free.append(mem)
    if PROFILE: Profiler.track(copy_signal, self.device, f"copyin: CPU -> HSA:{self.device.device_id}", is_copy=True)

  def copy_from_fd(self, dest, fd, offset, size):
    self.device.hw_queue.submit_barrier([], sync_signal := self.device.alloc_signal(reusable=True))

    if not hasattr(self, 'hb'):
      self.hb = [self._alloc(CHUNK_SIZE, BufferSpec(host=True)) for _ in range(2)]
      self.hb_signals = [self.device.alloc_signal(reusable=False) for _ in range(2)]
      self.hb_polarity = 0
      self.sdma = [hsa.HSA_AMD_SDMA_ENGINE_0, hsa.HSA_AMD_SDMA_ENGINE_1]
      for sig in self.hb_signals: hsa.hsa_signal_store_relaxed(sig, 0)

    fo = io.FileIO(fd, "a+b", closefd=False)
    fo.seek(offset - (minor_offset:=offset % PAGE_SIZE))

    copies_called = 0
    copied_in = 0
    for local_offset in range(0, size+minor_offset, CHUNK_SIZE):
      local_size = min(round_up(size+minor_offset, PAGE_SIZE)-local_offset, CHUNK_SIZE)
      copy_size = min(local_size-minor_offset, size-copied_in)
      if copy_size == 0: break

      hsa.hsa_signal_wait_scacquire(self.hb_signals[self.hb_polarity], hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_ACTIVE)
      self.device.reusable_signals.append(self.hb_signals[self.hb_polarity]) # it's free now and can be reused
      self.hb_signals[self.hb_polarity] = self.device.alloc_signal(reusable=False)

      fo.readinto(to_mv(self.hb[self.hb_polarity], local_size))
      check(hsa.hsa_amd_memory_async_copy_on_engine(dest+copied_in, self.device.agent, self.hb[self.hb_polarity]+minor_offset, HSADevice.cpu_agent,
                                                    copy_size, 1, ctypes.byref(sync_signal), self.hb_signals[self.hb_polarity],
                                                    self.sdma[self.hb_polarity], True))
      copied_in += copy_size
      self.hb_polarity = (self.hb_polarity + 1) % len(self.hb)
      minor_offset = 0 # only on the first
      copies_called += 1

    wait_signals = [self.hb_signals[self.hb_polarity - 1]]
    if copies_called > 1: wait_signals.append(self.hb_signals[self.hb_polarity])
    self.device.hw_queue.submit_barrier(wait_signals)

  def _copyout(self, dest:memoryview, src:T):
    HSADevice.synchronize_system()
    copy_signal = self.device.alloc_signal(reusable=True)
    c_agents = (hsa.hsa_agent_t*2)(self.device.agent, HSADevice.cpu_agent)
    check(hsa.hsa_amd_memory_lock_to_pool(from_mv(dest), dest.nbytes, c_agents, 2, HSADevice.cpu_mempool, 0, ctypes.byref(addr:=ctypes.c_void_p())))
    check(hsa.hsa_amd_memory_async_copy(addr, HSADevice.cpu_agent, src, self.device.agent, dest.nbytes, 0, None, copy_signal))
    hsa.hsa_signal_wait_scacquire(copy_signal, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_ACTIVE)
    check(hsa.hsa_amd_memory_unlock(from_mv(dest)))
    if PROFILE: Profiler.track(copy_signal, self.device, f"copyout: HSA:{self.device.device_id} -> CPU", is_copy=True)

  def transfer(self, dest:T, src:T, sz:int, src_dev=None, dest_dev=None):
    src_dev.hw_queue.submit_barrier([], sync_signal_1 := src_dev.alloc_signal(reusable=True))
    dest_dev.hw_queue.submit_barrier([], sync_signal_2 := dest_dev.alloc_signal(reusable=True))
    c_wait_signal = (hsa.hsa_signal_t*2)(sync_signal_1, sync_signal_2)
    check(hsa.hsa_amd_memory_async_copy_on_engine(dest, dest_dev.agent, src, src_dev.agent, sz, 2, c_wait_signal,
                                                  copy_signal := dest_dev.alloc_signal(reusable=False), hsa.HSA_AMD_SDMA_ENGINE_0, True))
    src_dev.hw_queue.submit_barrier([copy_signal])
    dest_dev.hw_queue.submit_barrier([copy_signal])
    if PROFILE: Profiler.track(copy_signal, src_dev, f"transfer: HSA:{src_dev.device_id} -> HSA:{dest_dev.device_id}", is_copy=True)

class HSADevice(Compiled):
  devices: List[HSADevice] = []
  agents: Dict[int, List[hsa.hsa_agent_t]] = {}
  cpu_agent: hsa.hsa_agent_t
  cpu_mempool: hsa.hsa_amd_memory_pool_t
  def __init__(self, device:str=""):
    if not HSADevice.agents:
      check(hsa.hsa_init())
      atexit.register(hsa_terminate)
      HSADevice.agents = scan_agents()
      HSADevice.cpu_agent = HSADevice.agents[hsa.HSA_DEVICE_TYPE_CPU][0]
      HSADevice.cpu_mempool = find_memory_pool(HSADevice.cpu_agent, segtyp=hsa.HSA_AMD_SEGMENT_GLOBAL, location=hsa.HSA_AMD_MEMORY_POOL_LOCATION_CPU)
      if PROFILE: check(hsa.hsa_amd_profiling_async_copy_enable(1))

    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    self.agent = HSADevice.agents[hsa.HSA_DEVICE_TYPE_GPU][self.device_id]
    self.gpu_mempool = find_memory_pool(self.agent, segtyp=hsa.HSA_AMD_SEGMENT_GLOBAL, location=hsa.HSA_AMD_MEMORY_POOL_LOCATION_GPU)
    self.hw_queue = AQLQueue(self)
    HSADevice.devices.append(self)

    check(hsa.hsa_agent_get_info(self.agent, hsa.HSA_AGENT_INFO_NAME, ctypes.byref(agent_name_buf := ctypes.create_string_buffer(256))))
    self.arch = ctypes.string_at(agent_name_buf).decode()

    check(hsa.hsa_system_get_info(hsa.HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, ctypes.byref(gpu_freq := ctypes.c_uint64())))
    self.clocks_to_time: float = 1 / gpu_freq.value

    check(hsa.hsa_agent_get_info(self.agent, hsa.HSA_AMD_AGENT_INFO_HDP_FLUSH, ctypes.byref(hdp_flush := hsa.hsa_amd_hdp_flush_t())))
    self.hdp_flush = hdp_flush

    self.delayed_free: List[int] = []
    self.reusable_signals: List[hsa.hsa_signal_t] = []

    from tinygrad.runtime.graph.hsa import HSAGraph
    super().__init__(device, HSAAllocator(self), HIPRenderer(), HSACompiler(self.arch), functools.partial(HSAProgram, self), HSAGraph)

    # Finish init: preallocate some signals + space for kernargs
    self.signal_pool = [init_c_var(hsa.hsa_signal_t(), lambda x: check(hsa.hsa_signal_create(1, 0, None, ctypes.byref(x)))) for _ in range(4096)]
    self._new_kernargs_region(16 << 20) # initial region size is 16mb

  def synchronize(self):
    self.hw_queue.wait()

    for sig in self.reusable_signals: hsa.hsa_signal_silent_store_relaxed(sig, 1)
    self.signal_pool.extend(self.reusable_signals)
    self.reusable_signals.clear()

    for opaque_to_free in self.delayed_free: check(hsa.hsa_amd_memory_pool_free(opaque_to_free))
    self.delayed_free.clear()

    self.kernarg_next_addr = self.kernarg_start_addr
    Profiler.process(self)

  @staticmethod
  def synchronize_system():
    for d in HSADevice.devices: d.synchronize()

  def alloc_signal(self, reusable=False):
    if len(self.signal_pool): signal = self.signal_pool.pop()
    else: check(hsa.hsa_amd_signal_create(1, 0, None, 0, ctypes.byref(signal := hsa.hsa_signal_t())))

    # reusable means a signal could be reused after synchronize for the device it's allocated from is called.
    if reusable: self.reusable_signals.append(signal)
    return signal

  def alloc_kernargs(self, sz):
    if self.kernarg_next_addr + sz >= self.kernarg_start_addr + self.kernarg_pool_sz: self._new_kernargs_region(int(self.kernarg_pool_sz * 2))
    result = self.kernarg_next_addr
    self.kernarg_next_addr = round_up(self.kernarg_next_addr + sz, 16)
    return result

  def _new_kernargs_region(self, sz:int):
    if hasattr(self, 'kernarg_start_addr'): self.delayed_free.append(self.kernarg_start_addr)
    self.kernarg_start_addr: int = self.allocator._alloc(sz, BufferSpec())
    self.kernarg_next_addr = self.kernarg_start_addr
    self.kernarg_pool_sz: int = sz

  def flush_hdp(self): self.hdp_flush.HDP_MEM_FLUSH_CNTL[0] = 1

def hsa_terminate():
  # Need to stop/delete aql queue before hsa shut down, this leads to gpu hangs.
  for dev in HSADevice.devices:
    Profiler.process(dev)
    del dev.hw_queue

  # hsa_shut_down cleans up all hsa-related resources.
  hsa.hsa_shut_down()
  HSADevice.synchronize = lambda: None #type:ignore
  HSAProgram.__del__ = lambda _: None #type:ignore
  if Profiler.collected_events: Profiler.save("/tmp/profile.json")
