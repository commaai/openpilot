from __future__ import annotations
from typing import cast, Any, Sequence
import itertools, ctypes, struct, time
from dataclasses import replace, dataclass, field
from collections import defaultdict, Counter
from tinygrad.helpers import dedup, pluralize, unwrap, to_tuple, Context, panic, partition, to_name, DEBUG, VIZ, ALL2ALL, PROFILE
from tinygrad.device import Device, Buffer, BufferSpec, Compiled, TinyELF, HCQ_RUNTIME_DEV, HCQ_DEVS, HOST_DEVS, UVM_DEVS, ProfileProgramEvent
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher, KernelInfo, GroupOp, graph_rewrite, rewrite_group, exec_alu, uopfunc
from tinygrad.uop.ops import pm_renumber_slots, CountCtx, sint_to_uop
from tinygrad.dtype import dtypes, AddrSpace, truncate
from tinygrad.renderer import Estimates
from tinygrad.runtime.support.c import Struct
from tinygrad.engine.realize import get_call_arg_uops, get_call_name, get_call_outs_ins, estimate_uop, pm_flatten_linear, lower_and_compile

# **** calls

@dataclass(frozen=True)
class HCQInfo:
  device:tuple[str, ...]

  kernels:tuple[tuple, ...] = () # (device, name, estimates, timestamp slots, profile key, input slots of the buffers, (outs, ins))
  estimates:Estimates = Estimates()

  slots:tuple[tuple[str, int], ...] = () # per device, the position of its batch slots in the args

  skip_wait:bool = False # TODO: remove. an rdma copy between nodes is two batches, so waiting on the first alone deadlocks

def all_devices_in(d:Any, c:frozenset[str]) -> bool: return {x.split(":")[0] for x in to_tuple(d)} <= c

def get_enqueue_devs(call:UOp) -> tuple[str, ...]:
  if call.op is not Ops.CALL: return () # entries can be AFTER-wrapped calls
  if call.body.op not in (Ops.PROGRAM, Ops.STORE) and not (call.body.op is Ops.CUSTOM_FUNCTION and call.body.arg.name == "encdec"): return ()
  runs_on = call.device if call.body.op is Ops.STORE and all_devices_in(call.device, HCQ_DEVS - UVM_DEVS) else get_call_arg_uops(call)[0].device
  devs = tuple(Device.canonicalize(d) for d in to_tuple(runs_on))
  if not all_devices_in(devs, HCQ_DEVS) or (call.body.op is Ops.STORE and all_devices_in(devs, UVM_DEVS)): return ()
  return devs

def unwrap_view(v:UOp) -> tuple[UOp, int]: # look through views to (base, byte offset)
  if v.op in (Ops.BITCAST, Ops.AFTER): return unwrap_view(v.src[0])
  if v.op is not Ops.SHRINK: return v, 0
  base, off = unwrap_view(v.src[0])
  return base, off + v.src[1].val * v.dtype.itemsize

def unwrap_lane(v:UOp) -> tuple[UOp, int|None, int]: # look through views and a lane select to (base, lane, byte offset)
  sel, off = unwrap_view(v)
  if sel.op is not Ops.MSELECT: return sel, None, off
  return (inner:=unwrap_view(sel.src[0]))[0], sel.arg, off + inner[1]

# **** commands

def timeline(dev:str) -> UOp: return UOp.alloc((2,), dtypes.u64, 0, device=dev, name="timeline")
def timeline_value(dev:str) -> UOp: return timeline(dev).index(1).load()

def make_program(prg:UOp, size:int, device:str) -> UOp:
  slot = int.from_bytes(prg.key[:8], "little")
  if PROFILE: Compiled.profile_events.append(ProfileProgramEvent(device, prg.src[0].arg.function_name, prg.src[3].arg, None, slot, prg.key))
  return UOp.alloc((size,), dtypes.u8, slot, device=device, name="program")

def make_submit(*cmds:UOp, dev:str, queue:str, deps:tuple[UOp, ...]) -> UOp: # a submit calls its commands on a queue, after deps
  kind = to_name(dev.split(":")[0], queue.split(":")[0], "submit") # nv_compute_submit: the backend matches the kind
  q = UOp.param(0, dtypes.u8, 1, dev, name=queue) # the queue instance: its device and name
  return UOp.sink(*cmds, arg=KernelInfo(kind)).call(q).after(*deps)

@uopfunc
def do_get_time_ms(ms:UOp) -> UOp:
  ts = UOp.placeholder((2,), dtypes.u64, addrspace=AddrSpace.REG)
  ts = ts.after(UOp.custom_function("clock_gettime", dtype=dtypes.i32).call(UOp.const(time.CLOCK_MONOTONIC, dtypes.i32), ts.index(0)))
  return ms.index(0).store(ts[0] * 1000 + ts[1] // 1000000).sink()
def get_time_ms(dep:UOp) -> UOp: return (ms:=UOp.placeholder((1,), dtypes.u64, addrspace=AddrSpace.REG)).after(do_get_time_ms(ms.after(dep)))[0]

@uopfunc
def hcq_fence(slots:UOp, tl:UOp, last:int) -> UOp: # wait for the previous run of this schedule
  tl = tl.replace(arg=replace(tl.arg, volatile=True)) # make it volatile, since it's polled

  loop = UOp.range(UOp(Ops.NOOP).after(start:=get_time_ms(target:=slots.index(last).load())), next(UOp.unique_num), dtype=dtypes.void)
  done = tl.after(target, loop).index(0).load()
  return done.backedge(loop, (done < target) & (get_time_ms(done) - start < int(Compiled.wait_timeout_ms))).sink()

def make_fence(slots:dict[str, UOp], num_queues:dict[str, int]) -> UOp:
  fences = [hcq_fence(patch(b, [], bytes(b.nbytes())), timeline(dev), 2 * num_queues[dev], name=to_name("fence", dev)) for dev, b in slots.items()]
  rearmed = [slots[dev].after(*fences).index(2 * i).store(0) for dev, n in num_queues.items() for i in range(n)]
  return UOp.group(*rearmed).barrier()

def announce(slots:dict[str, UOp], num_queues:dict[str, int], dep:UOp) -> list[UOp]: # bump the timeline and record it for the next run's fence
  bumped = [timeline(dev).after(dep).index(1).store(timeline_value(dev) + 1) for dev in slots]
  return bumped + [buf.after(dep).index(2 * num_queues[dev]).store(timeline_value(dev) + 1) for dev, buf in slots.items()]

def ins(name:str, *src:UOp|int) -> UOp: return UOp.custom_function(name).call(*[UOp.const(s, dtypes.u32) if isinstance(s, int) else s for s in src])

# **** patchers

def _is_link_patch(w:UOp) -> bool:
  if w.op is Ops.GETADDR: return not ((base:=unwrap_lane(w.src[0])[0]).op is Ops.PARAM and base.arg.name != "lt_input") # an input's is runtime
  if w.op is Ops.BUFFER: return w.addrspace is AddrSpace.GLOBAL # a register is written at runtime
  if w.op in {Ops.LOAD, Ops.CALL, Ops.PARAM}: return False
  if w.op is Ops.PROGRAM: return True
  return all(_is_link_patch(s) for s in w.src)

def patch(buf:UOp, rows:Sequence[tuple[int|UOp, UOp]], blob:bytes|None=None) -> UOp:
  # group into stacks based on dtype, alignment, is_link (rt/lt can't share a store) and ranges (a ranged offset is a loop)
  keys = [(w.dtype, (u:=sint_to_uop(o)).vmin % w.dtype.itemsize, _is_link_patch(w), tuple(u.ranges)) for o, w in rows]
  groups = [(key, [row for row, k in zip(rows, keys) if k == key]) for key in dedup(keys)]

  # a link patch writes the bare buffer after the blob, a runtime patch writes it after its deps
  dep = [buf.without_after.store(UOp(Ops.BINARY, arg=blob).bitcast(buf.dtype))] if blob is not None else []
  base, stores = buf.after(*dep), []
  for (dt, phase, link, rngs), grp in groups:
    view = (buf.without_after.after(*dep) if link else base)[phase:phase + (buf.max_numel() - phase) // dt.itemsize * dt.itemsize].bitcast(dt)
    offs = [UOp.const(i) if isinstance(i:=(o - phase) // dt.itemsize, int) else i for o, _ in grp]
    stores.append(view.index(UOp.stack(*offs)).store(UOp.stack(*[w for _, w in grp])).end(*rngs))
  return buf.after(*dep, *stores)

def layout_args(args:Sequence[UOp|int], offset:int=0) -> list[tuple[int, UOp]]:
  words = [a if isinstance(a, UOp) else UOp.const(a, dtypes.u32) for a in args]
  return [(offset + o, w) for (o, _), w in zip(TinyELF.iter_sig(tuple((None, AddrSpace.ALU, w.dtype, ()) for w in words)), words)]

def pack_args(args:list[tuple[int, UOp]], size:int) -> list[UOp]:
  words, end = [], 0
  for offset, arg in sorted(args, key=lambda x: x[0]):
    words += [UOp(Ops.BINARY, arg=bytes(offset - end)), arg] if offset != end else [arg]
    end = offset + arg.dtype.itemsize
  return words + [UOp(Ops.BINARY, arg=bytes(size - end))]

def ccall(fn:Any, *args:UOp|int) -> UOp:
  ret = dtypes.void if fn.restype is None else getattr(dtypes, fn.restype.__name__[2:], dtypes.u64) # void, c_int -> int, pointers -> u64
  return UOp.custom_function(fn.__name__, dtype=ret).call(*[UOp.const(a, dtypes.i32) if isinstance(a, int) else a for a in args])

def cstruct(struct_t:type[Struct], **fields:UOp|int) -> UOp:
  flds = {n: (o, {d.itemsize: d for d in dtypes.uints}[ctypes.sizeof(t)]) for n, t, o, *_ in struct_t._real_fields_ if ctypes.sizeof(t)}
  rows = [(flds[n][0], UOp.const(v, flds[n][1])) for n, v in fields.items()]
  buf = UOp.alloc((ctypes.sizeof(struct_t),), dtypes.u8, device=HCQ_RUNTIME_DEV.device, name=struct_t.__name__)
  return patch(buf, rows, bytes(ctypes.sizeof(struct_t)))

# **** prep: params, unwrap multi, staging copies

def replace_buffer(ctx:tuple[list[UOp], dict[UOp, int]], b:UOp) -> UOp:
  bufs, slots = ctx
  if slots.setdefault(b, len(bufs)) == len(bufs): bufs.append(b)
  return UOp.param(slots[b], b.dtype, b.max_numel(), b.device, name="lt_input")
pm_replace_buffers = PatternMatcher([(UPat(Ops.BUFFER, name="b"), lambda ctx, b: replace_buffer(ctx, b))])

def unwrap_call(call:UOp) -> UOp|None:
  if not get_enqueue_devs(call) or (n:=max(len(to_tuple(a.device)) for a in get_call_arg_uops(call))) == 1: return None
  dnum = UOp.variable("_device_num", 0, n - 1, dtypes.i32)
  return UOp(Ops.LINEAR, src=tuple(call.replace(src=(call.body, *[a.src[i] if a.op is Ops.MSTACK else a.mselect(i) if len(to_tuple(a.device)) > 1
                                                                   else a for a in call.src[1:]], dnum.bind(i))) for i in range(n)))
pm_unwrap_multi = PatternMatcher([(UPat(Ops.CALL, name="call"), unwrap_call)])

def split_rdma(call:UOp, dst:UOp, src:UOp) -> UOp|None:
  devs = [to_tuple(b.device)[0] for b in (dst, src)]
  if not all(hasattr(Device[d], "iface") for d in devs) or Device[devs[0]].peer_group == Device[devs[1]].peer_group: return None # not 2 nodes

  from tinygrad.runtime.ops_rdma import rdma_nic_for
  if None in (nics:=[rdma_nic_for(Device[d], Device[min(devs)]) for d in devs]): return None

  # wires: a placeholder per nic in place of the far gpu, named by it
  wires = [UOp.alloc(src.max_shape, src.dtype, 0, device=unwrap(nic).device, name=peer) for nic, peer in zip(nics, devs[::-1])]
  send = call.replace(src=wires[1].store_call(src).src)
  return UOp(Ops.LINEAR, src=(send, call.replace(src=dst.store_call(wires[0]).src)))

def stage_copy(call:UOp, dst:UOp, src:UOp) -> UOp|None:
  devs = [to_tuple(b.device)[0] for b in (dst, src)]
  if any(d.startswith("RDMA") for d in devs): return None # over the nic
  if not (enqueued:=get_enqueue_devs(call)): return None
  device = enqueued[0]

  from tinygrad.runtime.ops_disk import is_disk_read
  dev, host, usb_memcpys = Device[device], Device[device].host, getattr(Device[device], "is_usb", False)
  mappable = {"CPU", "PYTHON"} | ({"NPY", "DISK"} if usb_memcpys else {"DISK"} if is_disk_read(call) else set()) # ops_disk stages it in the batch
  if device != host and not all(Device[d].peer_group == dev.peer_group or (d.split(":")[0] in mappable and Device[d].host == host) for d in devs):
    stage = UOp.alloc(src.max_shape, src.dtype, int.from_bytes(call.key[:8], "little"), device=host)
    return UOp(Ops.LINEAR, src=(stage.store_call(src), dst.store_call(stage)))

  if Device[device].has_copy_queue: return None
  out, inp = (UOp.param(i, dtypes.u8, b.nbytes(), device=device) for i, b in enumerate((dst, src)))
  ast = out.index(r:=UOp.range(src.nbytes(), 0)).store(inp.index(r).load()).end(r).sink(arg=KernelInfo())
  return lower_and_compile(call.replace(src=(ast, *call.src[1:])))

pm_insert_copy_staging = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.STORE), UPat(name="dst"), UPat(name="src")), name="call", allow_any_len=True), split_rdma),
  (UPat(Ops.CALL, src=(UPat(Ops.STORE), UPat(name="dst"), UPat(name="src")), name="call", allow_any_len=True), stage_copy),
])

# **** batch: calls become queue programs

class DepsTracker:
  def __init__(self, queue_of:Sequence[Any]):
    self.queue_of = queue_of
    self.ranges:dict[tuple[UOp, int|None], list[tuple[int, int, bool, int]]] = defaultdict(list) # per (base, lane): (start, end, is_write, tag)

  def access_resources(self, accesses:Sequence[tuple[UOp, bool]], tag:int) -> list[int]: # (buffer, is_write) accesses -> earlier tags to wait on
    lanes = [(*unwrap_lane(b), b, is_write) for b, is_write in accesses]
    spans = [((base, lane), off, off + b.max_numel() * b.dtype.itemsize, is_write) for base, lane, off, b, is_write in lanes]

    # a write waits for the overlapping reads and writes, a read for the overlapping writes
    waits = [t for key, start, end, is_write in spans for lo, hi, was_write, t in self.ranges[key]
             if lo < end and start < hi and (is_write or was_write)]

    # a write owns its span: the earlier accesses keep only the parts outside it.
    for key, start, end, is_write in spans:
      if is_write:
        self.ranges[key] = [(lo, hi, w, t) for a, b, w, t in self.ranges[key] for lo, hi in ((a, min(b, start)), (max(a, end), b)) if lo < hi]
      else: self.ranges[key] = [r for r in self.ranges[key] if r[:3] != (start, end, False) or self.queue_of[r[3]] != self.queue_of[tag]]
      self.ranges[key].append((start, end, is_write, tag))
    return dedup(waits)

@dataclass
class BatchCtx:
  batch:list[tuple[UOp, str, str]] # (call, device, queue) per enqueued call
  profile:bool
  queues:dict[str, list[str]] = field(init=False) # per device, its queues in first-use order: none for a peer without calls of its own
  calls:dict[tuple[str, str], list[int]] = field(init=False) # per queue, the tags of its calls in order
  peers:dict[tuple[str, str], set[str]] = field(init=False) # the other devices whose memory a queue touches
  slots:dict[str, UOp] = field(init=False)

  def __post_init__(self):
    # the other devices whose memory a call touches: a usb gpu can't reach host memory
    reach = [HCQ_DEVS if getattr(Device[d], "is_usb", False) else HCQ_DEVS | HOST_DEVS for _, d, _ in self.batch]
    arg_devs = [{Device.canonicalize(x) for b in get_call_arg_uops(c) for x in to_tuple(b.device)} for c, _, _ in self.batch]
    touched = [{x for x in xs if all_devices_in(x, r)} - {d} for xs, r, (_, d, _) in zip(arg_devs, reach, self.batch)]

    # queues and devices in first-use order, a call's own device then the peers it touches
    used = dedup(keys:=[(d, q) for _, d, q in self.batch])
    devs = dedup([x for (d, _), t in zip(keys, touched) for x in [d, *sorted(t)]])
    self.queues = {dev: [q for d, q in used if d == dev] for dev in devs}

    # the tags of a queue in order, and the devices its calls touch
    self.calls = {k: [tag for _, tag in grp] for k, grp in itertools.groupby(sorted((key, tag) for tag, key in enumerate(keys)), key=lambda e: e[0])}
    self.peers = {k: {d for tag in tags for d in touched[tag]} for k, tags in self.calls.items()}

    # a slot is [signal][timestamp], 16 bytes: the queue signals, the timeline, then two per call if profiling
    self.slots = {dev: UOp.alloc((2 * (len(qs) + 1 + (2 * len(self.batch) if self.profile else 0)),), dtypes.u64, device=dev,
                                 spec=BufferSpec(host=True, uncached=True, cpu_access=True), name="slots") for dev, qs in self.queues.items()}

  def slot(self, dev:str, i:int) -> UOp: return self.slots[dev].shrink(((2 * i, 2 * i + 2),)) # not a slice: 10x the cost
  def queue_signal(self, dev:str, queue:str) -> UOp: return self.slot(dev, self.queues[dev].index(queue))
  def stamps(self, dev:str, tag:int) -> tuple[int, ...]: return (st:=len(self.queues[dev])+1+2*tag, st + 1) if self.profile else ()

def _wait_ins(ctx:BatchCtx, waits:dict[tuple[str, str], int]) -> list[UOp]:
  return [ins("wait", ctx.queue_signal(d, q), UOp.const(t + 1, dtypes.u64)) for (d, q), t in waits.items()]

def _start_ins(ctx:BatchCtx, dev:str, queue:str) -> list[UOp]:
  return [ins("wait", timeline(d), timeline_value(d)) for d in [dev, *sorted(ctx.peers.get((dev, queue), ()))]] + [ins("memory_barrier")]

def _build_queues(ctx:BatchCtx) -> dict[tuple[str, str], list[UOp]]:
  queue_of = [(dev, queue) for _, dev, queue in ctx.batch] # the (device, queue) of each call by tag

  # calc deps for each tag
  tracker = DepsTracker(queue_of)
  accs = [[(s, k in get_call_outs_ins(c)[0]) for k, s in enumerate(c.src[1:]) if not s.is_variable] for c, _, _ in ctx.batch]
  dep_tags:list[list[int]] = [tracker.access_resources(acc, tag) for tag, acc in enumerate(accs)]

  # add barriers when a call deps on another call of its queue
  barriers, last_barrier = set[int](), dict[tuple[str, str], int]()
  for tag, (deps, queue) in enumerate(zip(dep_tags, queue_of)):
    if any(queue_of[t] == queue and t >= last_barrier.get(queue, 0) for t in deps):
      barriers.add(tag)
      last_barrier[queue] = tag

  # keep only latest tags to wait for each queue
  call_waits = [{queue_of[t]: t for t in sorted(deps) if queue_of[t] != queue_of[tag]} for tag, deps in enumerate(dep_tags)]

  # NV waits break QMD chaining, so a call on an NV compute queue that waits also waits for the previous launch there
  # TODO: can we handle that better for nv?
  nv_compute = [queue for queue in ctx.calls if queue[0].split(":")[0] == "NV" and queue[1].startswith("COMPUTE")]
  prev = {tag: p for queue in nv_compute for p, tag in itertools.pairwise(ctx.calls[queue])}
  call_waits = [waits | {queue_of[tag]: prev[tag]} if waits and tag in prev else waits for tag, waits in enumerate(call_waits)]

  # for each device collect what queues it depends on (i.e. which queues touch its memory)
  dev_users = {dev: sorted(queue for queue, peers in ctx.peers.items() if dev == queue[0] or dev in peers) for dev in ctx.queues}

  # to finalize device, closer waits for other users to complete
  own_closers = {dev: (dev, queues[0] if len(queues) == 1 else "COMPUTE:0") for dev, queues in ctx.queues.items()}
  closers = {dev: dev_users[dev][0] if dev.split(":")[0] in HOST_DEVS else own_closers[dev] for dev in ctx.queues}
  close_waits = {dev: {queue: ctx.calls[queue][-1] for queue in dev_users[dev] if queue != closers[dev]} for dev in ctx.queues}

  # all tags that should signal
  signal_tags = {tag for waits in [*call_waits, *close_waits.values()] for tag in waits.values()}

  def call_cmds(tag:int, call:UOp, dev:str, queue:str) -> list[UOp]:
    barrier = [ins("exec_barrier")] if tag in barriers else []
    stamps = [ins("timestamp", ctx.slot(dev, i)) for i in ctx.stamps(dev, tag)]
    signal = [ins("store", ctx.queue_signal(dev, queue), UOp.const(tag + 1, dtypes.u64))] if tag in signal_tags else []
    return barrier + _wait_ins(ctx, call_waits[tag]) + [ins("memory_barrier")] * bool(call_waits[tag]) + stamps[:1] + [call] + stamps[1:] + signal

  # closers advance device timelines
  close_cmds = {dev: _wait_ins(ctx, close_waits[dev]) + [ins("store", timeline(dev), timeline_value(dev) + 1)] for dev in ctx.queues}

  # wait for timeline accessing devices for the first time
  order = dedup(queue_of + list(closers.values()))
  open_cmds = {queue: _start_ins(ctx, *queue) if queue in ctx.calls or not ctx.queues[queue[0]] else [] for queue in order}

  programs = {queue: [cmd for tag in ctx.calls.get(queue, []) for cmd in call_cmds(tag, *ctx.batch[tag])] for queue in order}
  closings = {queue: [cmd for dev in ctx.queues if closers[dev] == queue for cmd in close_cmds[dev]] for queue in order}
  return {queue: open_cmds[queue] + programs[queue] + closings[queue] for queue in order}

def _finalize_batch(ctx:BatchCtx, skip_wait:bool=False) -> UOp:
  queues = _build_queues(ctx)

  # wait for the previous (this) schedule run
  fence = make_fence(ctx.slots, num_queues:={dev: len(qs) for dev, qs in ctx.queues.items()})
  submits:list[UOp] = []
  for (dev, queue), cmds in queues.items(): submits.append(make_submit(*cmds, dev=dev, queue=queue, deps=(fence, *submits[-1:])))
  sink = UOp.sink(submits[-1], *announce(ctx.slots, num_queues, submits[-1]), arg=KernelInfo("hcq_submit", estimates=Estimates(), optimized=True))

  # per call metadata
  names = [get_call_name(c, get_call_arg_uops(c)) for c, _, _ in ctx.batch]
  estimates = [estimate_uop(c) for c, _, _ in ctx.batch]
  stamps = [tuple(2 * s + 1 for s in ctx.stamps(d, tag)) for tag, (_, d, _) in enumerate(ctx.batch)]
  profile_keys = [c.body.key if c.body.op is Ops.PROGRAM else None for c, _, _ in ctx.batch]
  args = [[unwrap_lane(c.src[1 + g])[:2] for g in getattr(c.body.arg, "globals", range(len(c.src) - 1))] for c, _, _ in ctx.batch]
  bufs = [tuple(b.arg.slot for b, _ in a) if all(b.op is Ops.PARAM and lane is None for b, lane in a) else () for a in args]
  kerns = tuple(zip([d for _, d, _ in ctx.batch], names, estimates, stamps, profile_keys, bufs, [get_call_outs_ins(c) for c, _, _ in ctx.batch]))
  slots = ctx.slots if ctx.profile else {} # profiling reads them
  info = HCQInfo(tuple(ctx.queues), skip_wait=skip_wait, kernels=kerns, slots=tuple((d, i) for i, d in enumerate(slots)),
                 estimates=Estimates(*[sum(x*n for x, n in Counter(t).items()) for t in zip(*[vars(e).values() for e in estimates])]).simplify())
  return sink.call(*slots.values(), aux=info)

@rewrite_group(new_ctx=False)
def sched_batches(l:UOp, profile:bool) -> UOp:
  copies = [c for c in l.src if c.op is Ops.CALL and c.body.op is Ops.STORE]
  ring = sorted({Device.canonicalize(d) for c in copies for b in get_call_arg_uops(c) for d in to_tuple(b.device) if all_devices_in(d, HCQ_DEVS)})
  # TODO: handle max queue amt better
  num_copy = max(1, min(len(ring), 8 if {d.split(":")[0] for d in ring} == {"AMD"} else 1) if ALL2ALL >= 1 else 1)
  copy_of = {(dst, src): (i - j - 1) % len(ring) % num_copy for i, dst in enumerate(ring) for j, src in enumerate(ring)}

  def queue_for(c:UOp) -> str:
    if c.body.op is Ops.PROGRAM: return "COMPUTE:0"
    if c.body.op is Ops.CUSTOM_FUNCTION: return "ENCDEC:0"
    return f"COPY:{copy_of.get((c.src[1].device, c.src[2].device), 0)}"

  devs = [get_enqueue_devs(c) for c in l.src]
  srcs:list[UOp] = []
  for hcq, grp in itertools.groupby(zip(l.src, devs), key=lambda e: bool(e[1])):
    run = list(grp)
    peer_groups = dedup([Device[d[0]].peer_group for _, d in run if d])
    batches = [[(c, d[0], queue_for(c)) for c, d in run if Device[d[0]].peer_group == pg] for pg in peer_groups]
    if hcq: srcs += [_finalize_batch(BatchCtx(batch, profile), batch is not batches[-1]) for batch in batches]
    else: srcs += [c for c, _ in run]
  return l.replace(src=tuple(srcs))

# **** encode: queue programs become cmdbufs

class HWQueue:
  q_rewrite = PatternMatcher([
    # rewrites from calls
    (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, name="prg"),), name="call", allow_any_len=True), lambda ctx, call, prg: ctx.exec(call, prg)),
    (UPat(Ops.CALL, src=(UPat(Ops.STORE), UPat(name="dst"), UPat(name="src")), allow_any_len=True),
     lambda ctx, dst, src: ctx.copy(dst, src, src.max_numel() * src.dtype.itemsize)),
    (UPat(Ops.CALL, src=(UPat.custom_function("encdec", name="s"),), name="c", allow_any_len=True), lambda ctx, c, s: ctx.encdec(c, s)),

    # ins
    (UPat(Ops.CALL, src=(UPat.custom_function("copy"), UPat(name="dst"), UPat(name="src"), UPat(name="n"))),
     lambda ctx, dst, src, n: ctx.copy(dst, src, n.val)),
    (UPat(Ops.CALL, src=(UPat.custom_function("memory_barrier"),)), lambda ctx: ctx.memory_barrier()),
    (UPat(Ops.CALL, src=(UPat.custom_function("exec_barrier"),)), lambda ctx: ctx.exec_barrier()),
    (UPat(Ops.CALL, src=(UPat.custom_function("wait"), UPat(name="dst"), UPat(name="val"))), lambda ctx, dst, val: ctx.wait(dst, val)),
    (UPat(Ops.CALL, src=(UPat.custom_function("wait_eq"), UPat(name="dst"), UPat(name="val"))), lambda ctx, dst, val: ctx.wait(dst, val, eq=True)),
    (UPat(Ops.CALL, src=(UPat.custom_function("timestamp"), UPat(name="dst"))), lambda ctx, dst: ctx.timestamp(dst)),
    (UPat(Ops.CALL, src=(UPat.custom_function("store"), UPat(name="dst"), UPat(name="val"))), lambda ctx, dst, val: ctx.signal(dst, val)),
    (UPat(Ops.CALL, src=(UPat.custom_function("write"),), allow_any_len=True, name="u"), lambda ctx, u: ctx.write(*u.src[1:])),

    # loop in cmdbufs
    (UPat(Ops.END, src=(UPat(Ops.LINEAR, name="body"), UPat(Ops.RANGE, name="r"))), lambda ctx, body, r: ctx.loop(body, r)),
  ])

  def __init__(self, submit:UOp): # submit: sink(commands).call(queue).after(deps)
    call, self.deps = submit.src[0], submit.src[1:]
    self.lin, q = call.body, call.src[1]
    self.devs, self.queue = (q.device,), q.arg.name
    self.dev = Device[self.devs[0]]
    self.blob, self.patches = bytearray(), list[tuple[int|UOp, UOp]]()

  def q(self, *words:UOp|int) -> int:
    for w in words:
      if isinstance(w, int): self.blob += struct.pack("I", truncate[dtypes.u32](w))
      else:
        c = w
        while c.op is Ops.CAST: c = c.src[0]
        if c.op is Ops.BINARY: self.blob += c.arg
        elif c.op is not Ops.CONST:
          self.patches.append((len(self.blob), w))
          self.blob += bytes(w.dtype.itemsize)
        else: self.blob += struct.pack(unwrap(w.dtype.fmt), truncate[w.dtype](c.val))
    return len(self.blob)

  def loop(self, body:UOp, r:UOp):
    start, first = len(self.blob), len(self.patches)
    for u in body.src: self.q_rewrite.rewrite(u, ctx=self)
    trip, words = len(self.blob) - start, self.patches[first:]
    # each trip gets its words as dwords: a trip need not be 8 bytes aligned
    self.patches[first:] = [(o + 4 * k + r * trip, (w >> (32 * k)).cast(dtypes.u32)) for o, w in words for k in range(w.dtype.itemsize // 4)]
    self.blob += self.blob[start:] * int(r.vmax)

  def memory_barrier(self): pass # a copy queue has nothing to flush
  def exec_barrier(self): pass
  def submit(self, cmdbuf:UOp) -> UOp: raise NotImplementedError("queues need a submit")
  def encode(self) -> UOp: return self.submit(encode_cmdbuf(self, self.lin)) # submit(linear) becomes submit(cmdbuf)

def encode_cmdbuf(hq:HWQueue, lin:UOp|None=None, name:str="cmdbuf", device:str|tuple[str, ...]|None=None) -> UOp:
  for u in lin.src if lin is not None else (): hq.q_rewrite.rewrite(u, ctx=hq) # the commands of the linear go to the stream
  stream = bytes(hq.blob)

  # loop over pathes with the same value
  rt = [(o, w) for o, w in hq.patches if isinstance(o, int) and o % 4 == 0 and w.dtype.itemsize in (4, 8) and not _is_link_patch(w)]
  uses = {w: [o for o, _ in grp] for w, grp in itertools.groupby(sorted(rt, key=lambda p: p[1].key), key=lambda p: p[1])}
  looped = {w: (at, UOp.range(len(at), next(UOp.unique_num))) for w, at in uses.items() if len(at) > 1}
  dwords = [(UOp(Ops.BINARY, arg=struct.pack(f"<{len(at)}I", *at)).bitcast(dtypes.u32).index(r).load() + 4 * k, (w >> 32 * k).cast(dtypes.u32))
            for w, (at, r) in looped.items() for k in range(w.dtype.itemsize // 4)]
  patches = [(o, w) for o, w in hq.patches if w not in looped] + dwords
  nested = dedup([g.src[0] for g in UOp.sink(*[w for _, w in patches]).toposort() if g.op is Ops.GETADDR and g.src[0].op is Ops.LINEAR])

  # nested linears (like kernargs) merge into a buffer per name, patched before the stream
  bufs = []
  for lname, ls in itertools.groupby(sorted(nested, key=lambda l: l.arg), key=lambda l: l.arg):
    hq.blob, hq.patches = bytearray(), []
    offs = {l: (hq.q(UOp(Ops.BINARY, arg=bytes(-len(hq.blob) % 128))), hq.q(*l.src)) for l in ls}
    bufs.append((offs, encode_cmdbuf(hq, name=lname)))
  views = {l: buf.without_after[o:e] for offs, buf in bufs for l, (o, e) in offs.items()}

  buf = UOp.alloc((len(stream),), dtypes.u8, device=device or hq.devs[0], name=to_name(name, hq.queue)).after(*hq.deps)
  words = UOp.sink(*[w for _, w in patches]).substitute(views).src
  return patch(buf, list(zip([o for o, _ in patches], words)), stream).after(*[b for _, b in bufs])

# **** lower: cmdbufs become a host program

def hoist_links(ctx:list[UOp], a:UOp) -> UOp|None:
  links, rest = partition(a.src[1:], lambda s: s.op is Ops.STORE and _is_link_patch(s))
  if not links: return None
  ctx.extend(links)
  return a.src[0].after(*rest)
pm_hoist_links = PatternMatcher([(UPat(Ops.AFTER, name="a"), hoist_links)])

pm_lift_deps = PatternMatcher([
  # f(x.after(dep)) -> f(x).after(dep)
  (UPat((Ops.SHRINK, Ops.BITCAST, Ops.MSELECT, Ops.GETADDR), src=(UPat(Ops.AFTER, name="a"),), allow_any_len=True, name="u"),
   lambda a, u: u.replace(src=(a.src[0], *u.src[1:])).after(*a.src[1:])),

  # x.after(a).after(b) -> x.after(a, b)
  (UPat(Ops.AFTER, src=(UPat(Ops.AFTER, name="a"),), allow_any_len=True, name="u"), lambda a, u: a.src[0].after(*dedup([*a.src[1:], *u.src[1:]]))),
])

def _needs_arg(u:UOp, root:bool) -> bool:
  if u.op in (Ops.BUFFER, Ops.ALLOC): return u.addrspace is AddrSpace.GLOBAL
  return u.op is Ops.PARAM and (u.is_variable != root or u.arg.name == "lt_input")

def _param_for(u:UOp, slot:int) -> UOp:
  if u.op is Ops.GETADDR or u.is_variable:
    return UOp.param(slot, u.commit_dtype(dtypes.i32), name=u.arg.name if u.is_variable else None, addrspace=AddrSpace.ALU).cast(u.dtype)
  return UOp.param(slot, u.dtype, u.max_numel(), HCQ_RUNTIME_DEV.device, name=f"{u.arg.name}_{slot}" if u.op is Ops.ALLOC and u.arg.name else None)

def lift(call:UOp, root:bool=False) -> UOp: # callees are lifted already
  body, args = graph_rewrite(call.body, pm_lift_deps + pm_renumber_slots, init_ctx=CountCtx, walk=True, name="lift deps"), list(call.src[1:])
  nodes = body.toposort(gate=lambda u: u.op is not Ops.GETADDR)
  leaves = dedup([u for u in nodes if _needs_arg(u, root)] + [g for u in nodes for g in u.src if g.op is Ops.GETADDR])
  slots = args + (new:=[u for u in leaves if u not in args])
  body = body.substitute({u: _param_for(u, slots.index(u)) for u in leaves}, walk=True)

  # new args in the caller
  own = {p: args[p.arg.slot].without_after for u in new for p in u.toposort() if p.op is Ops.PARAM and p.arg.name and not _needs_arg(p, root)}
  return call.replace(src=(body, *args, *UOp.sink(*new).substitute(own, walk=True).src))

pm_lift = PatternMatcher([
  # a param of a lowered function, an "lt_input" keeps its device
  (UPat(Ops.PARAM, name="u"), lambda u: u.replace(arg=replace(u.arg, device=None)) if u.arg.name not in (None, "lt_input") else None),
  (UPat(Ops.CALL, src=(UPat(Ops.SINK),), allow_any_len=True, name="call"), lift), # lift allocs and getaddrs
])

Compiled.pm_encode = PatternMatcher([])
Compiled.pm_lower = PatternMatcher([])

def lower_call(call:UOp) -> UOp|None:
  if not isinstance(call.arg.aux, HCQInfo): return None

  # encode bodies
  body = graph_rewrite(call.body, Compiled.pm_encode, ctx=(lt_patches:=list[UOp]()), bpm=pm_hoist_links, name="encode")
  body = graph_rewrite(body, Compiled.pm_lower, ctx=lt_patches, bpm=pm_hoist_links, enter_calls=True, name="lower")
  body = graph_rewrite(body, pm_lift, walk=True, enter_calls=True, name="lift")

  if VIZ: graph_rewrite(UOp.sink(*dedup(lt_patches)), PatternMatcher([]), name="View Link-Time Patches")

  # batch args, link patches wait after the call
  return lift(call.replace(src=(body, *call.src[1:])), root=True).after(*dedup(lt_patches))
pm_lower_call = PatternMatcher([(UPat(Ops.CALL, src=(UPat(Ops.SINK),), name="call", allow_any_len=True), lower_call)])

# **** link: allocs to buffers, addresses to consts

Compiled.link_bufs |= {"timeline": lambda b: Device[b.device].timeline}
Compiled.pm_link = PatternMatcher([])

@dataclass
class LinkCtx: inputs:list[UOp]; refs:list[UOp] = field(default_factory=list) # noqa: E702

def bufferize_buf(ctx:LinkCtx, b:UOp) -> UOp:
  if b.arg.name is None: return UOp.new_buffer(b.device, b.max_numel(), b.dtype)
  dev, spec = Device[b.device], b.arg.spec or BufferSpec(cpu_access=True)

  owned = Compiled.link_bufs.get(b.arg.name, Compiled.link_bufs.get(to_name(b.device, b.arg.name)))
  if owned is not None: r = owned if isinstance(owned, Buffer) else owned(b)
  else: r = Buffer(dev.device, max(b.max_numel(), 1) * b.dtype.itemsize, options=spec, preallocate=True)

  return UOp.from_buffer(r, b.dtype, HCQ_RUNTIME_DEV.device)

def resolve_getaddr(ctx:LinkCtx, g:UOp) -> UOp|None:
  if unwrap_lane(buf:=(v:=unwrap_view(g.src[0]))[0])[0].op is not Ops.BUFFER: return None # input address, resolved per run
  ctx.refs.append(buf) # add to refs
  return UOp.const(cast(Buffer, buf.buffer).view(g.src[0].nbytes(), v[1]).get_buf(to_tuple(g.arg)[0]), dtypes.u64)

def _write_host(buf:UOp, writes:list[tuple[int, bytes]]) -> UOp:
  base, off = unwrap_view(buf)
  mv = cast(Buffer, base.buffer).ensure_allocated().host.view(fmt='B')
  for o, data in writes: mv[off + o:off + o + len(data)] = data
  return UOp(Ops.NOOP)

def fold_words(buf:UOp, offs:UOp, ws:UOp) -> UOp:
  return _write_host(buf, [(o.val * w.dtype.itemsize, struct.pack(w.dtype.fmt, truncate[w.dtype](w.val))) for o, w in zip(offs.src, ws.src)])

pm_resolve_link = PatternMatcher([
  # collapse committed const conversions
  (UPat(Ops.CAST, src=(UPat(Ops.CAST, src=(UPat.cvar(),), name="inner"),), name="c"), lambda c, inner: inner.src[0].cast(c.dtype)),
  # an alloc and a link time input become buffers
  (UPat(Ops.ALLOC, name="b"), bufferize_buf),
  (UPat(Ops.PARAM, name="b"), lambda ctx, b: ctx.inputs[b.arg.slot] if b.arg.name == "lt_input" else None),
  # the address of a buffer is a const
  (UPat(Ops.GETADDR, name="g"), resolve_getaddr),
  # math on consts is a const
  (UPat(GroupOp.ALU, src=UPat.cvar().or_casted(), name="a"), lambda a: UOp.const(exec_alu(a.op, a.dtype, [s.val for s in a.src]), a.dtype)),
  # fold rules
  (UPat(name="buf").store(UPat(Ops.BINARY, name="blob").or_bitcasted()), lambda buf, blob: _write_host(buf, [(0, blob.arg)])),
  (UPat(name="buf").index(UPat(Ops.STACK, src=UPat.cvar(), name="offs")).store(UPat(Ops.STACK, src=UPat.cvar().or_casted(), name="ws")), fold_words),
  (UPat(name="buf").index(UPat.cvar("off")).store(UPat.cvar().or_casted("w")), lambda buf, off, w: fold_words(buf, UOp.stack(off), UOp.stack(w))),
  # a call keeps the deps that are not written yet
  (UPat(Ops.AFTER, src=(UPat(Ops.CALL),), allow_any_len=True, name="a"), lambda a: a.src[0].after(*(s for s in a.src[1:] if s.op is not Ops.NOOP))),
  (UPat(Ops.AFTER, name="a"), lambda a: None if a.without_after.op is Ops.CALL else
   a.src[0] if all(s.op is Ops.NOOP for s in a.src[1:]) else panic(RuntimeError, f"unresolved link words on {a.src[0].op}")),
])

# **** compile and link

hcq_compile_cache:dict[tuple[UOp, bool, bool], UOp] = {} # eager templates: a buffer-free linear (uops are hash-consed) to its compiled form

@rewrite_group(lambda linear,input_uops,profile,cache=False,ret=None: f"HCQ Compile {pluralize('Kernel', len(ret.src))}")
def hcq_compile(linear:UOp, input_uops:list[UOp]|None, profile:bool, cache:bool=False) -> UOp:
  if any(isinstance(getattr(c.without_after.arg, "aux", None), HCQInfo) for c in linear.src): return linear # compiled already

  if cache and input_uops is not None:
    slots = {u:i for i,u in reversed(tuple(enumerate(input_uops)))}
    linear = graph_rewrite(linear, pm_replace_buffers, ctx=(input_uops, slots), walk=True, name="replace buffers")
  linear = graph_rewrite(linear, pm_unwrap_multi+pm_insert_copy_staging+pm_flatten_linear, name="prep calls")
  if cache and input_uops is not None and (cached:=hcq_compile_cache.get(key:=(linear, profile, ALL2ALL >= 1))) is not None: return cached
  lin = graph_rewrite(sched_batches(linear, profile), pm_lower_call, walk=True, name="lower call")
  with Context(EMULATED_DTYPES=""): final_linear = lower_and_compile(lin, verbose=DEBUG>=3)
  if cache and input_uops is not None and final_linear is not linear: hcq_compile_cache[key] = final_linear
  return final_linear

@rewrite_group(lambda _,input_uops=None,ret=None: f"HCQ Link {pluralize('Kernel', len(ret.src))}")
def hcq_link(linear:UOp, input_uops:list[UOp]|None=None) -> UOp:
  linked = graph_rewrite(linear, Compiled.pm_link, walk=True, name="link programs") if Compiled.pm_link.patterns else linear
  ctx = LinkCtx(input_uops or [])
  linked = graph_rewrite(linked, pm_resolve_link+Compiled.pm_link, ctx=ctx, walk=True, name="link")
  if ctx.refs: linked = linked.replace(src=(linked.src[0].after(*dedup(ctx.refs)), *linked.src[1:])) # attach refs to linear
  return linked
