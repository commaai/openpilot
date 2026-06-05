import collections, time
from typing import Any, cast
from tinygrad.helpers import round_up, PROFILE, ALL2ALL, merge_dicts, getenv, suppress_finalizing, TracingKey, unwrap
from tinygrad.runtime.support.hcq import HCQCompiled, HCQAllocator, HCQSignal, HCQBuffer, HWQueue, HCQArgsState, BumpAllocator, MMIOInterface
from tinygrad.device import Buffer, BufferSpec, Compiled, Device, MultiBuffer, ProfileGraphEntry, ProfileGraphEvent
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp, Ops, Variable
from tinygrad.engine.jit import GraphRunner, MultiGraphRunner
from tinygrad.runtime.ops_rdma import RDMACopyQueue

class HCQGraph(MultiGraphRunner):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.devices = list({cast(HCQCompiled, Device[b.device]) for (_,_,bufs,_) in self.calls for b in bufs})

    # CPU Device is always last
    self.devices = sorted(self.devices, key=lambda x: 1 if x._is_cpu() else 0)

    # Replace input buffers with variables.
    self.hcq_bufs = [[b._buf for b in bufs] for (_,_,bufs,_) in self.calls]
    self.input_replace_to_var: dict[tuple[int, int], Variable] = {}

    for j, replace in enumerate(self.uop_replace):
      for pos, iidx in replace:
        x = self.input_replace_to_var.setdefault((j,pos), UOp.variable(f"inp_{iidx}_{self.calls[j][0]}", 0, 0xffffffffffffffff, dtype=dtypes.uint64))
        self.hcq_bufs[j][pos] = HCQBuffer(x, self.hcq_bufs[j][pos].size) # Create fake buffer with variable

    # Allocate kernel args.
    kernargs_size: dict[Compiled, int] = collections.defaultdict(int)
    for runtime in self.runtimes:
      if runtime is None: continue
      kernargs_size[runtime.dev] += round_up(runtime.kernargs_alloc_size, 16)
    self.kernargs_bufs: dict[Compiled, HCQBuffer] = {d:d.allocator._alloc(max(sz, 1), BufferSpec(cpu_access=True)) for d,sz in kernargs_size.items()}

    # Fill initial arguments.
    self.ji_args: dict[int, HCQArgsState] = {}

    kargs_alloc: dict[Compiled, BumpAllocator] = {dev:BumpAllocator(buf.size) for dev,buf in self.kernargs_bufs.items()}
    for j, runtime in enumerate(self.runtimes):
      if runtime is None: continue
      argsbuf = self.kernargs_bufs[runtime.dev].offset(kargs_alloc[runtime.dev].alloc(runtime.kernargs_alloc_size, 16))
      self.ji_args[j] = runtime.fill_kernargs(self.hcq_bufs[j], self.calls[j][1].arg.vars, argsbuf)

    # Schedule Dependencies.
    # There are two types of queues on each device: copy and compute. Both must synchronize with all external operations before launching any
    # graph-related tasks. This synchronization uses a global timeline signal per device. Within the graph, the compute queue coordinates with
    # global operations and sets a kickoff signal. Any queue accessing a buffer from another device waits for this signal from the device’s
    # compute queue to ensure exclusive access. The compute queue signals the completion of the graph, synchronizing with the device's copy queue.
    self.ji_schedule: dict[int, tuple[HCQCompiled, HWQueue, list, list, HCQSignal, int|None]] = {}

    self.comp_queues: dict[HCQCompiled, HWQueue] = {dev: unwrap(dev.hw_compute_queue_t)() for dev in self.devices}
    self.copy_queues: dict[tuple[HCQCompiled, int], HWQueue] = {} # lazy allocation, keyed by (device, queue_idx)
    self.rdma_queues: dict[tuple[HCQCompiled, HCQCompiled], RDMACopyQueue] = {} # lazy allocation, keyed by device pair
    self.num_copy_queues: int = getenv("HCQ_NUM_SDMA", min(len(self.devices), 8) if ALL2ALL >= 1 else 1)
    self.num_rdma_ops: dict[tuple[HCQCompiled, HCQCompiled], int] = collections.defaultdict(int)

    self.rdma_vars: dict[tuple[HCQCompiled, HCQCompiled], tuple[Variable, Any]] = {} # value is variable and src_qp
    self.rdma_deps: dict[int, tuple[HWQueue, list[tuple[HCQSignal, int]], HCQSignal, int]] = {}
    self.rdma_last_dest: dict[int, tuple[HWQueue, int]] = {} # per QP id: last (queue, signal_value) for dbell ordering

    # Per-peer-group representative device for signal allocation. For cpu, use devices[0].
    self.pg_dev: dict[Any, HCQCompiled] = {dev.peer_group: self.devices[0] for dev in self.devices if dev._is_cpu()} \
                                        |  {dev.peer_group: dev for dev in self.devices if not dev._is_cpu()}

    self.kick_signals: dict[Any, HCQSignal] = {pg: pg_dev.new_signal(value=0) for pg, pg_dev in self.pg_dev.items()}
    self.signals: dict[Any, HCQSignal] = {**{dev: dev.new_signal(value=0) for dev in self.devices if not dev._is_cpu()},
      **{dev: self.pg_dev[dev.peer_group].new_signal(value=0) for dev in self.devices if dev._is_cpu()}}
    self.kickoff_value: int = 0
    self.kickoff_var = UOp.variable("kickoff_var", 0, 0xffffffff, dtype=dtypes.uint32)

    # When profiling allocate 2 signals for each jit item to measure speed. The jth jit item have signals at 2*j and 2*j+1.
    # TODO: This logic might allocate a few extra signals...
    self.prof_signals: list[HCQSignal] = []
    self.prof_graph_deps: list[list[int]] = []
    self.prof_graph_entries: list[ProfileGraphEntry] = []

    self.last_j: dict[HWQueue, int|None] = collections.defaultdict(lambda: None)
    self.queue_access: dict[HWQueue, dict[HWQueue, int|None]] = collections.defaultdict(lambda: collections.defaultdict(lambda: None))
    self.dev_access: dict[HWQueue, set[HCQCompiled]] = collections.defaultdict(set)

    for dev, queue in self.comp_queues.items(): self.dev_access[queue].add(dev)

    self.input_replace_map: dict[HCQCompiled, set[tuple[int, int]]] = collections.defaultdict(set)
    self.device_vars: dict[HCQCompiled, dict[str, int]] = {}

    for j, ((_, ast, bufs, device_vars), runtime) in enumerate(zip(self.calls, self.runtimes)):
      is_xfer = ast.op is Ops.COPY and hasattr(alc:=Device[bufs[0].device].allocator, '_transfer') and alc.supports_transfer \
                and bufs[0].device.split(":")[0] == bufs[1].device.split(":")[0]
      ji_devs = [cast(HCQCompiled, Device[b.device]) for b in bufs] if is_xfer else []
      is_rdma = len(ji_devs) > 0 and not any(d._is_cpu() for d in ji_devs) and len(set(d.peer_group for d in ji_devs)) > 1

      if runtime is not None: enqueue_dev: HCQCompiled = runtime.dev
      else:
        # For copy ops prioritize enqeueuing on the src device, so reverse the buffers.
        for b in bufs[::-1]:
          if (enqueue_dev:=cast(HCQCompiled, Device[b.device])).hw_copy_queue_t is not None: break

      # set any fixedvars on the device
      self.device_vars[enqueue_dev] = merge_dicts([self.device_vars.get(enqueue_dev, {}), device_vars])
      if runtime is not None: self.device_vars[enqueue_dev] = merge_dicts([self.device_vars[enqueue_dev], {k: 0 for k in ast.arg.runtimevars}])

      if runtime is not None:
        enqueue_queue = self.comp_queues[enqueue_dev]
      elif is_rdma:
        enqueue_queue = self.comp_queues[enqueue_dev]
        rdma_key = (cast(HCQCompiled, Device[bufs[0].device]).rdma_dev(), enqueue_dev.rdma_dev())
        self.rdma_queues.setdefault(rdma_key, RDMACopyQueue(enqueue_dev.rdma_dev()))
      else:
        assert (enqueue_dev.hw_copy_queue_t is not None), "device must implement a copy queue"
        queue_idx = self.devices.index(cast(HCQCompiled, Device[bufs[0].device])) % self.num_copy_queues
        enqueue_queue = self.copy_queues.setdefault((enqueue_dev, queue_idx),
          enqueue_dev.hw_copy_queue_t(queue_idx=queue_idx).wait(self.kick_signals[enqueue_dev.peer_group], self.kickoff_var))

      out_signal = self.signals.setdefault(enqueue_queue, self.pg_dev[enqueue_dev.peer_group].new_signal(value=0))

      # Get dependencies based on input and output buffers.
      if is_rdma:
        src_qp, dest_qp = rdma_key[1].iface.connect(rdma_key[0])[:2]
        sync_signals, opt_deps, rdeps = self._resolve_deps(bufs[1:], [], enqueue_queue, enqueue_dev, out_signal, j,
                                                           is_copy=is_xfer, rdma_qp=src_qp)
        peer_queue = self.comp_queues[peer_dev:=cast(HCQCompiled, Device[bufs[0].device])]
        peer_out_signal = self.signals.setdefault(peer_queue, self.pg_dev[peer_dev.peer_group].new_signal(value=0))
        peer_sync_signals, peer_opt_deps, peer_rdeps = self._resolve_deps(bufs[:1], [0], peer_queue, peer_dev, peer_out_signal, j,
                                                                          is_copy=is_xfer, rdma_qp=dest_qp)
        self.rdma_deps[j] = (peer_queue, peer_sync_signals + peer_opt_deps, peer_out_signal, j + 1)
        self.last_j[peer_queue] = j
      else:
        sync_signals, opt_deps, rdeps = self._resolve_deps(bufs, ast.arg.outs if runtime is not None else [0], enqueue_queue,
          enqueue_dev, out_signal, j, is_copy=is_xfer)

      self.ji_schedule[j] = (enqueue_dev, enqueue_queue, sync_signals, opt_deps[::-1], out_signal, None if runtime is not None else (j + 1))

      # Collect profile information if profiling is enabled.
      if PROFILE:
        # When execution are chained, we can reuse the end timestamp from the previous command as the start timestamp for the current command.
        sig_st = prev_ji * 2 + 1 if len(opt_deps) == 0 and (prev_ji:=self.last_j[enqueue_queue]) is not None else j * 2

        # Description based on the command.
        prof_ji_desc = runtime.name if runtime is not None else TracingKey(f"{bufs[1].device} -> {bufs[0].device}", ret=bufs[0].nbytes) # type: ignore

        prof_name = enqueue_dev.device if runtime is not None else f"{enqueue_dev.device}:SDMA:{queue_idx}"
        self.prof_graph_entries.append(ProfileGraphEntry(prof_name, prof_ji_desc, sig_st, j * 2 + 1))
        self.prof_graph_deps.append([d - 1 for _, d in rdeps])

      self.last_j[enqueue_queue] = j

    # Check which signals are used in the profile graph.
    self.prof_signal_is_used: set[int] = {sid for ent in self.prof_graph_entries for sid in (ent.st_id, ent.en_id)}

    # Build hardware queues.
    self.copy_to_devs: dict[HCQCompiled, set[HCQCompiled]] = {dev: set() for dev in self.devices}

    # Create variable timeline signals for each device.
    timeline_sigaddrs = {dev: UOp.variable(f"timeline_sig_{self.dev_name(dev)}", 0, 0xffffffffffffffff, dtype=dtypes.uint64) for dev in self.devices}
    self.virt_timeline_vals = {dev: UOp.variable(f"timeline_var_{self.dev_name(dev)}", 0, 0xffffffff, dtype=dtypes.uint32) for dev in self.devices}
    self.virt_timeline_signals = {dev: unwrap(dev.signal_t)(HCQBuffer(timeline_sigaddrs[dev], 16),owner=dev,is_timeline=True) for dev in self.devices}

    for dev in self.devices:
      self.comp_queues[dev].memory_barrier().wait(self.virt_timeline_signals[dev], self.virt_timeline_vals[dev]) \
                           .wait(self.kick_signals[dev.peer_group], self.kickoff_var).signal(self.signals[dev], self.kickoff_var)

    for j, ((dev_idx, ast, bufs, _), runtime) in enumerate(zip(self.calls, self.runtimes)):
      enqueue_dev, enqueue_queue, sync_signals, deps, signal, signal_val = self.ji_schedule[j]

      # Lazy allocate signals
      if PROFILE: self.prof_signals += [enqueue_dev.new_signal(value=0) for _ in range(2)]

      for sig, val in sync_signals + deps: enqueue_queue.wait(sig, val)

      # Encode waits and start profile timestamp (if needed).
      if PROFILE and j * 2 in self.prof_signal_is_used: enqueue_queue.timestamp(self.prof_signals[j * 2])

      # Encode main commands based on ji type.
      if runtime is not None:
        enqueue_queue.exec(runtime, self.ji_args[j], ast.arg.global_size or (1,1,1), ast.arg.local_size or (1,1,1))  # type: ignore[arg-type]
      elif j in self.rdma_deps:
        dest_queue, dest_deps, dest_out_signal, dest_out_val = self.rdma_deps[j]
        for sig, val in dest_deps: dest_queue.wait(sig, val)

        dest, src = bufs[0], bufs[1]
        dest_dev, src_dev = cast(HCQCompiled, Device[dest.device]), cast(HCQCompiled, Device[src.device])
        dest_rdma, src_rdma = dest_dev.rdma_dev(), src_dev.rdma_dev()

        # get qp info
        src_qp, dest_qp, src_cq_buf, dest_cq_buf = src_rdma.iface.connect(dest_rdma)

        # use var for head
        head_var = self.rdma_vars.setdefault((dest_rdma, src_rdma), (UOp.variable(f"rdma_var_{j}", 0, 0xffffffff, dtype=dtypes.uint32), src_qp))[0]
        next_head = self.num_rdma_ops[(dest_rdma, src_rdma)]

        rdma_queue = self.rdma_queues[(dest_rdma, src_rdma)]
        rdma_queue.copy(self.hcq_bufs[j][0], self.hcq_bufs[j][1], dest.nbytes) \
                  .encode_ring(enqueue_queue, src_dev, src_rdma.iface, src_qp, src_cq_buf, head_var + next_head, ring_uar=True) \
                  .encode_ring(self.comp_queues[dest_dev], dest_dev, dest_rdma.iface, dest_qp, dest_cq_buf, head_var + next_head)

        dest_queue.signal(dest_out_signal, dest_out_val)
        self.num_rdma_ops[(dest_rdma, src_rdma)] += 1
      elif ast.op is Ops.COPY:
        dest, src = bufs[0], bufs[1]
        uop_replace_j = dict(self.uop_replace[j])
        for bufid in range(len(bufs)):
          if (replace_iidx:=uop_replace_j.get(bufid)) is not None: self.input_replace_map[enqueue_dev].add((replace_iidx, dev_idx))
          else: cast(HCQAllocator, enqueue_dev.allocator).map(self.hcq_bufs[j][bufid])
        enqueue_queue.copy(self.hcq_bufs[j][0], self.hcq_bufs[j][1], dest.nbytes)
        self.copy_to_devs[cast(HCQCompiled, Device[dest.device])].add(cast(HCQCompiled, Device[src.device]))

      # Encode finish profile timestamp (if needed).
      if PROFILE and j * 2 + 1 in self.prof_signal_is_used: enqueue_queue.timestamp(self.prof_signals[j * 2 + 1])

      if signal_val is not None: enqueue_queue.signal(signal, signal_val)

    for dev in self.devices:
      for dep_dev in list(self.copy_to_devs[dev]) + [dev]:
        for copy_q in self._dev_copy_queues(dep_dev):
          if copy_q in self.signals: self.comp_queues[dev].wait(self.signals[copy_q], cast(int, self.last_j[copy_q]) + 1)

      self.comp_queues[dev].signal(self.virt_timeline_signals[dev], self.virt_timeline_vals[dev] + 1).bind(dev)
      for copy_q in self._dev_copy_queues(dev): copy_q.bind(dev)

    self.last_timeline: dict[HCQCompiled, tuple[HCQSignal, int]] = {dev: (dev.timeline_signal, 0) for dev in self.devices}
    self.queue_signals_to_reset = [self.signals[q] for q in list(self.comp_queues.values()) + list(self.copy_queues.values()) if q in self.signals]

  def _resolve_deps(self, bufs, outs, enqueue_queue, enqueue_dev, out_signal, j, is_copy, rdma_qp=None):
    rdeps = self._access_resources(bufs, outs, (enqueue_queue, j + 1)) #type:ignore

    # Order shared QP doorbell record writes across different compute queues (head+1 must complete before head+2).
    if rdma_qp is not None and (prev:=self.rdma_last_dest.get(id(rdma_qp))) is not None and prev[0] is not enqueue_queue:
      rdeps = rdeps + [(prev[0], prev[1])]
    if rdma_qp is not None: self.rdma_last_dest[id(rdma_qp)] = (enqueue_queue, j + 1)

    # Update dependencies to include previous kernel in queue. This is required for timeline signals.
    opt_deps, deps = [], rdeps + ([(enqueue_queue, prev_ji + 1)] if (prev_ji:=self.last_j[enqueue_queue]) is not None else [])

    # Optimize dependencies by removing redundant ones. Remove waiting for the value of the queue which is known to be already
    # synced with the current queue.
    for dep_queue, dep_val in sorted(deps, key=lambda x: x[1], reverse=True):
      if (qa:=self.queue_access[enqueue_queue][dep_queue]) is None or qa < dep_val:
        opt_deps.append((self.signals[dep_queue], dep_val))
        self.queue_access[enqueue_queue][dep_queue] = dep_val
        self.dev_access[enqueue_queue].update(self.dev_access[dep_queue])

    # Ensure device is ready for use in current context: the graph has initialized the device and it's safe to operate on it within this graph.
    # Only sync with same-peer-group devices; cross-peer-group sync is handled by RDMA.
    sync_signals = [(self.signals[d], self.kickoff_var) for b in bufs
      if (d:=cast(HCQCompiled, Device[cast(Buffer, b).device])) not in self.dev_access[enqueue_queue]
      and (d.peer_group == enqueue_dev.peer_group or rdma_qp is None)]
    self.dev_access[enqueue_queue].update(cast(HCQCompiled, Device[cast(Buffer, b).device]) for b in bufs)

    # Remove self-dependency for compute and copy queues.
    # For compute, in case of NV, optimize when only 1 same-queue dependency exists, since NV chains 2+ executions in this case,
    # eliminating dependency need. For RDMA, keep self-dependency to flush cache.
    dname = enqueue_dev.device.split(":", 1)[0]
    can_opt = dname in {"AMD", "QCOM"} or (dname == "NV" and len(sync_signals) == 0 and len(opt_deps) == 1 and id(opt_deps[0][0]) == id(out_signal))
    if (can_opt or is_copy) and rdma_qp is None: opt_deps = [x for x in opt_deps if id(x[0]) != id(out_signal)]

    # Enable necessary signals in the schedule by setting the signal value.
    for sig, val in opt_deps: self.ji_schedule[val - 1] = self.ji_schedule[val - 1][:5] + (val,)

    return sync_signals, opt_deps, rdeps

  def _dev_copy_queues(self, dev): return [q for (d, _), q in self.copy_queues.items() if d == dev]

  def __call__(self, input_uops:tuple[UOp, ...], var_vals:dict[str, int], wait=False) -> float|None:
    # Map input buffers
    for dev in self.devices:
      for iidx, dev_idx in self.input_replace_map[dev]:
        buf = b.bufs[dev_idx] if isinstance(b:=input_uops[iidx].buffer, MultiBuffer) else b
        cast(HCQAllocator, dev.allocator).map(buf._buf)

    # Wait and restore signals
    self.kickoff_value += 1
    for dev in self.devices: self.last_timeline[dev][0].wait(self.last_timeline[dev][1])
    if PROFILE and self.kickoff_value > 1: self.collect_timestamps()

    hcq_var_vals = {self.kickoff_var.expr: self.kickoff_value, **var_vals,
                    **{var.expr: dev.timeline_value - 1 for dev, var in self.virt_timeline_vals.items()},
                    **{sig.base_buf.va_addr.expr: dev.timeline_signal.base_buf.va_addr for dev, sig in self.virt_timeline_signals.items()}}

    # Update buffers
    for j, replace in enumerate(self.uop_replace):
      dev_idx = self.calls[j][0]
      for pos, iidx in replace:
        buf = b.bufs[dev_idx] if isinstance(b:=input_uops[iidx].buffer, MultiBuffer) else b
        hcq_var_vals[self.input_replace_to_var[(j,pos)].expr] = buf._buf.va_addr

    for (var, qp) in self.rdma_vars.values(): hcq_var_vals[var.expr] = qp.head
    for q in self.rdma_queues.values(): q.submit(q.dev, hcq_var_vals)

    for dev in self.devices:
      self.comp_queues[dev].submit(dev, hcq_var_vals_local:=hcq_var_vals|self.device_vars.get(dev, {}))
      for copy_queue in self._dev_copy_queues(dev): copy_queue.submit(dev, hcq_var_vals_local)
      self.last_timeline[dev] = (dev.timeline_signal, dev.next_timeline())

    # Launch graph
    for sig in self.queue_signals_to_reset: sig.value = 0
    for sig in self.kick_signals.values(): sig.value = self.kickoff_value

    if wait:
      st = time.perf_counter()
      for dev in self.devices: self.last_timeline[dev][0].wait(self.last_timeline[dev][1])
      return time.perf_counter() - st
    return None

  def collect_timestamps(self):
    # NOTE: Append to any device is fine...
    self.devices[0].profile_events += [ProfileGraphEvent(self.prof_graph_entries, self.prof_graph_deps, [s.timestamp for s in self.prof_signals])]

  def dev_name(self, dev) -> str: return dev.device.replace(":", "_")

  @suppress_finalizing
  def __del__(self):
    for dev in self.devices: self.last_timeline[dev][0].wait(self.last_timeline[dev][1])

    if PROFILE and self.kickoff_value >= 1: self.collect_timestamps()

    for fdev, buf in self.kernargs_bufs.items(): fdev.allocator._free(buf, BufferSpec(cpu_access=True))

  @staticmethod
  def supports_uop(batch_devs:list[Compiled], new_call:UOp) -> bool:
    # Check if all devices are HCQ
    all_devs = cast(list[HCQCompiled], GraphRunner._all_devs(batch_devs, new_call))
    if not all(issubclass(type(d), HCQCompiled) for d in all_devs): return False

    # If all of devices are mapped into CPU address space, can use CPU inside the peer group.
    cpu_support = all(type(d.timeline_signal.base_buf.view) is MMIOInterface for d in all_devs)

    # Check if all devices are within the same peer group. Allow cross-peer-group if all peer groups have RDMA devices.
    if len(set(d.peer_group for d in all_devs if not (cpu_support and d._is_cpu()))) > 1:
      try: [d.rdma_dev() for d in all_devs if not d._is_cpu()]
      except RuntimeError: return False

    if new_call.src[0].op is Ops.COPY:
      # MOCKGPU is not supported, since it can't execute commands in parallel
      is_xfer = len(set(type(d) for d in all_devs)) == 1 and hasattr(alc:=all_devs[0].allocator, '_transfer') and alc.supports_transfer
      return is_xfer or (all_devs[0].hw_copy_queue_t is not None and not getattr(all_devs[0], 'iface', None).__class__.__name__.startswith("MOCK"))
    return new_call.src[0].op is Ops.PROGRAM
