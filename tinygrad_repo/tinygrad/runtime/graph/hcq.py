import collections, time
from typing import Any, cast
from tinygrad.helpers import round_up, PROFILE
from tinygrad.runtime.support.hcq import HCQCompiled, HCQAllocator, HCQSignal, HCQBuffer, HWQueue, HCQArgsState, BumpAllocator
from tinygrad.device import Buffer, BufferSpec, Compiled, Device, ProfileGraphEntry, ProfileGraphEvent
from tinygrad.dtype import dtypes
from tinygrad.ops import UOp, Variable
from tinygrad.engine.realize import ExecItem, BufferXfer, CompiledRunner
from tinygrad.engine.jit import MultiGraphRunner

class HCQGraph(MultiGraphRunner):
  def __init__(self, jit_cache: list[ExecItem], input_rawbuffers: list[Buffer], var_vals: dict[Variable, int]):
    super().__init__(jit_cache, input_rawbuffers, var_vals)
    self.devices = list(set(cast(HCQCompiled, d) for ji in jit_cache for d in [Device[cast(Buffer, x).device] for x in ji.bufs]))

    # Replace input buffers with variables.
    self.hcq_bufs = [[cast(Buffer, x)._buf for x in ji.bufs] for ji in jit_cache]
    self.input_replace_to_var: dict[tuple[int, int], Variable] = {}

    for (j,i), input_idx in self.input_replace.items():
      x = self.input_replace_to_var.setdefault((j,i), UOp.variable(f"input_{input_idx}", 0, 0xffffffffffffffff, dtype=dtypes.uint64))
      self.hcq_bufs[j][i] = HCQBuffer(x, self.hcq_bufs[j][i].size, texture_info=self.hcq_bufs[j][i].texture_info) # Create fake buffer with variable

    # Allocate kernel args.
    kernargs_size: dict[Compiled, int] = collections.defaultdict(int)
    for ji in jit_cache:
      if not isinstance(ji.prg, CompiledRunner): continue
      kernargs_size[ji.prg.dev] += round_up(ji.prg._prg.kernargs_alloc_size, 16)
    self.kernargs_bufs: dict[Compiled, HCQBuffer] = {dev:dev.allocator._alloc(sz, BufferSpec(cpu_access=True)) for dev,sz in kernargs_size.items()}

    # Fill initial arguments.
    self.ji_args: dict[int, HCQArgsState] = {}

    kargs_alloc: dict[Compiled, BumpAllocator] = {dev:BumpAllocator(buf.size) for dev,buf in self.kernargs_bufs.items()}
    for j,ji in enumerate(jit_cache):
      if not isinstance(ji.prg, CompiledRunner): continue

      argsbuf = self.kernargs_bufs[ji.prg.dev].offset(kargs_alloc[ji.prg.dev].alloc(ji.prg._prg.kernargs_alloc_size, 16))
      self.ji_args[j] = ji.prg._prg.fill_kernargs(self.hcq_bufs[j], ji.prg.p.vars, argsbuf)

    # Schedule Dependencies.
    # There are two types of queues on each device: copy and compute. Both must synchronize with all external operations before launching any
    # graph-related tasks. This synchronization uses a global timeline signal per device. Within the graph, the compute queue coordinates with
    # global operations and sets a kickoff signal. Any queue accessing a buffer from another device waits for this signal from the device’s
    # compute queue to ensure exclusive access. The compute queue signals the completion of the graph, synchronizing with the device's copy queue.
    self.ji_schedule: dict[int, tuple[HCQCompiled, HWQueue, list, list, HCQSignal, int|None]] = {}

    self.comp_queues: dict[HCQCompiled, HWQueue] = {dev: dev.hw_compute_queue_t() for dev in self.devices}
    self.copy_queues: dict[HCQCompiled, HWQueue] = {} # lazy allocation

    self.signals: dict[Any, HCQSignal] = {**{dev: dev.signal_t(value=0) for dev in self.devices}, **{"CPU": self.devices[0].signal_t(value=0)}}
    self.kickoff_value: int = 0
    self.kickoff_var = UOp.variable("kickoff_var", 0, 0xffffffff, dtype=dtypes.uint32)

    # When profiling allocate 2 signals for each jit item to measure speed. The jth jit item have signals at 2*j and 2*j+1.
    # TODO: This logic might allocate a few extra signals...
    self.prof_signals: list[HCQSignal] = [self.devices[0].signal_t() for i in range(len(jit_cache) * 2)] if PROFILE else []
    self.prog_graph_deps: list[list[int]] = []
    self.prof_graph_entries: list[ProfileGraphEntry] = []

    last_j: dict[HWQueue, int|None] = collections.defaultdict(lambda: None)
    queue_access: dict[HWQueue, dict[HWQueue, int|None]] = collections.defaultdict(lambda: collections.defaultdict(lambda: None))
    dev_access: dict[HWQueue, set[HCQCompiled]] = collections.defaultdict(set)

    for dev, queue in self.comp_queues.items(): dev_access[queue].add(dev)

    for j,ji in enumerate(jit_cache):
      enqueue_dev: HCQCompiled = ji.prg.dev if (is_exec_prg:=isinstance(ji.prg, CompiledRunner)) else Device[ji.bufs[1].device] #type:ignore

      if is_exec_prg:
        enqueue_queue = self.comp_queues[enqueue_dev]
      else:
        assert (enqueue_dev.hw_copy_queue_t is not None), "device must implement a copy queue"
        enqueue_queue = self.copy_queues.setdefault(enqueue_dev, enqueue_dev.hw_copy_queue_t())

      out_signal = self.signals.setdefault(enqueue_queue, enqueue_dev.signal_t(value=0))

      # Get dependencies based on input and output buffers.
      rdeps = self._access_resources(ji.bufs, ji.prg.p.outs if is_exec_prg else [0], (enqueue_queue, j + 1)) #type:ignore

      # Update dependencies to include previous kernel in queue. This is required for timeline signals.
      opt_deps, deps = [], rdeps + ([(enqueue_queue, prev_ji + 1)] if (prev_ji:=last_j[enqueue_queue]) is not None else [])

      # Optimize dependencies by removing redundant ones. Remove waiting for the value of the queue which is known to be already
      # synced with the current queue.
      for dep_queue, dep_val in sorted(deps, key=lambda x: x[1], reverse=True):
        if (qa:=queue_access[enqueue_queue][dep_queue]) is None or qa < dep_val:
          opt_deps.append((self.signals[dep_queue], dep_val))
          queue_access[enqueue_queue][dep_queue] = dep_val

      # Ensure device is ready for use in current context: the graph has initialized the device and it's safe to operate on it within this graph.
      for dep_queue, _ in opt_deps: dev_access[enqueue_queue].update(dev_access[dep_queue])
      sync_signals = [(self.signals[d], self.kickoff_var) for b in ji.bufs if (d:=Device[cast(Buffer, b).device]) not in dev_access[enqueue_queue]]
      dev_access[enqueue_queue].update(cast(HCQCompiled, Device[cast(Buffer, b).device]) for b in ji.bufs)

      # Remove self-dependency for compute and copy queues.
      # For compute, in case of NV, optimize when only 1 same-queue dependency exists, since NV chains 2+ executions in this case,
      # eliminating dependency need.
      dname = enqueue_dev.device.split(":", 1)[0]
      can_opt = dname in {"AMD", "QCOM"} or (dname == "NV" and len(sync_signals) == 0 and len(opt_deps) == 1 and id(opt_deps[0][0]) == id(out_signal))
      if can_opt or isinstance(ji.prg, BufferXfer): opt_deps = [x for x in opt_deps if id(x[0]) != id(out_signal)]

      # Enable necessary signals in the schedule by setting the signal value.
      for sig, val in opt_deps: self.ji_schedule[val - 1] = self.ji_schedule[val - 1][:5] + (val,)
      self.ji_schedule[j] = (enqueue_dev, enqueue_queue, sync_signals, opt_deps[::-1], out_signal, None if is_exec_prg else (j + 1))

      # Collect profile information if profiling is enabled.
      if PROFILE:
        # When execution are chained, we can reuse the end timestamp from the previous command as the start timestamp for the current command.
        sig_st = prev_ji * 2 + 1 if len(opt_deps) == 0 and (prev_ji:=last_j[enqueue_queue]) is not None else j * 2

        # Description based on the command.
        prof_ji_desc = ji.prg._prg.name if is_exec_prg else f"{ji.bufs[1].device} -> {ji.bufs[0].device}" # type: ignore

        self.prof_graph_entries.append(ProfileGraphEntry(enqueue_dev.device, prof_ji_desc, sig_st, j * 2 + 1, is_copy=not is_exec_prg))
        self.prog_graph_deps.append([d - 1 for _, d in rdeps])

      last_j[enqueue_queue] = j

    # Check which signals are used in the profile graph.
    self.prof_signal_is_used = [any(ent.st_id == j or ent.en_id == j for ent in self.prof_graph_entries) for j in range(len(self.prof_signals))]

    # Build hardware queues.
    self.copy_to_devs: dict[HCQCompiled, set[HCQCompiled]] = {dev: set() for dev in self.devices}

    # Create variable timeline signals for each device.
    timeline_sigaddrs = {dev: UOp.variable(f"timeline_sig_{dev.device_id}", 0, 0xffffffffffffffff, dtype=dtypes.uint64) for dev in self.devices}
    self.virt_timeline_vals = {dev: UOp.variable(f"timeline_var_{dev.device_id}", 0, 0xffffffff, dtype=dtypes.uint32) for dev in self.devices}
    self.virt_timeline_signals = {dev: dev.signal_t(base_buf=HCQBuffer(timeline_sigaddrs[dev], 16), timeline_for_device=dev) for dev in self.devices}

    for dev in self.devices:
      self.comp_queues[dev].memory_barrier().wait(self.virt_timeline_signals[dev], self.virt_timeline_vals[dev]) \
                           .wait(self.signals['CPU'], self.kickoff_var).signal(self.signals[dev], self.kickoff_var)

    for j,ji in enumerate(jit_cache):
      enqueue_dev, enqueue_queue, sync_signals, deps, signal, signal_val = self.ji_schedule[j]

      for sig, val in sync_signals + deps: enqueue_queue.wait(sig, val)

      # Encode waits and start profile timestamp (if needed).
      if PROFILE and self.prof_signal_is_used[j * 2]: enqueue_queue.timestamp(self.prof_signals[j * 2])

      # Encode main commands based on ji type.
      if isinstance(ji.prg, CompiledRunner):
        enqueue_queue.exec(ji.prg._prg, self.ji_args[j], tuple(ji.prg.p.global_size or (1,1,1)), tuple(ji.prg.p.local_size or (1,1,1)))
      elif isinstance(ji.prg, BufferXfer):
        dest, src = [cast(Buffer, x) for x in ji.bufs[0:2]]
        cast(HCQAllocator, Device[src.device].allocator).map(dest._buf)

        enqueue_queue.copy(self.hcq_bufs[j][0].va_addr, self.hcq_bufs[j][1].va_addr, dest.nbytes)
        self.copy_to_devs[cast(HCQCompiled, Device[dest.device])].add(cast(HCQCompiled, Device[src.device]))

      # Encode finish profile timestamp (if needed).
      if PROFILE and self.prof_signal_is_used[j * 2 + 1]: enqueue_queue.timestamp(self.prof_signals[j * 2 + 1])

      if signal_val is not None: enqueue_queue.signal(signal, signal_val)

    for dev in self.devices:
      for dep_dev in list(self.copy_to_devs[dev]) + [dev]:
        if dep_dev in self.copy_queues: self.comp_queues[dev].wait(self.signals[(copy_q:=self.copy_queues[dep_dev])], cast(int, last_j[copy_q]) + 1)

      self.comp_queues[dev].signal(self.virt_timeline_signals[dev], self.virt_timeline_vals[dev] + 1).bind(dev)
      if dev in self.copy_queues: self.copy_queues[dev].bind(dev)

    self.last_timeline: dict[HCQCompiled, tuple[HCQSignal, int]] = {dev: (dev.timeline_signal, 0) for dev in self.devices}
    self.queue_signals_to_reset = [self.signals[q] for q in list(self.comp_queues.values()) + list(self.copy_queues.values()) if q in self.signals]

  def __call__(self, input_rawbuffers: list[Buffer], var_vals: dict[Variable, int], wait=False) -> float|None:
    # Wait and restore signals
    self.kickoff_value += 1
    for dev in self.devices: self.last_timeline[dev][0].wait(self.last_timeline[dev][1])
    for sig in self.queue_signals_to_reset: sig.value = 0
    self.signals['CPU'].value = self.kickoff_value

    if PROFILE and self.kickoff_value > 1: self.collect_timestamps()

    hcq_var_vals = {self.kickoff_var: self.kickoff_value, **var_vals,
                    **{var: dev.timeline_value - 1 for dev, var in self.virt_timeline_vals.items()},
                    **{sig.base_buf.va_addr: dev.timeline_signal.base_buf.va_addr for dev, sig in self.virt_timeline_signals.items()}}

    # Update rawbuffers
    for (j,i),input_idx in self.input_replace.items(): hcq_var_vals[self.input_replace_to_var.get((j,i))] = input_rawbuffers[input_idx]._buf.va_addr

    for dev in self.devices:
      self.comp_queues[dev].submit(dev, hcq_var_vals)
      if (copy_queue:=self.copy_queues.get(dev, None)) is not None: copy_queue.submit(dev, hcq_var_vals)

      self.last_timeline[dev] = (dev.timeline_signal, dev.next_timeline())

    if wait:
      st = time.perf_counter()
      for dev in self.devices: self.last_timeline[dev][0].wait(self.last_timeline[dev][1])
      return time.perf_counter() - st
    return None

  def collect_timestamps(self):
    # NOTE: Append to any device is fine...
    self.devices[0].profile_events += [ProfileGraphEvent(self.prof_graph_entries, self.prog_graph_deps, [s.timestamp for s in self.prof_signals])]

  def __del__(self):
    for dev in self.devices: self.last_timeline[dev][0].wait(self.last_timeline[dev][1])

    if PROFILE and self.kickoff_value >= 1: self.collect_timestamps()

    for fdev, buf in self.kernargs_bufs.items(): fdev.allocator._free(buf, BufferSpec(cpu_access=True))
