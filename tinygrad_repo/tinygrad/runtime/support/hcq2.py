from __future__ import annotations
from typing import cast, Any, Sequence
import functools, itertools, weakref, ctypes, struct, time
from dataclasses import replace, dataclass, field
from collections import defaultdict
from tinygrad.helpers import dedup, pluralize, unwrap, to_tuple, ContextVar, Context, panic, partition, getenv, to_name
from tinygrad.helpers import DEBUG, VIZ, DEV, ALL2ALL, PROFILE
from tinygrad.device import Device, Buffer, BufferSpec, Compiled, TinyELF, HCQ_RUNTIME_DEV, ProfileProgramEvent
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher, KernelInfo, GroupOp, graph_rewrite, rewrite_group, exec_alu, uopfunc, sym_infer
from tinygrad.uop.ops import pm_renumber_slots
from tinygrad.dtype import dtypes, DType, DTYPES_DICT, AddrSpace
from tinygrad.renderer import Estimates
from tinygrad.engine.realize import get_call_arg_uops, get_call_name, get_call_outs_ins
from tinygrad.engine.realize import estimate_uop, pm_flatten_linear, lower_and_compile

# *****************
# 0. helpers

HCQ_CACHE_THRESH = ContextVar("HCQ_CACHE_THRESH", 64)
HCQ_DEVS = frozenset(("AMD", "NV", "QCOM", "CUDA", "NULL", "METAL"))
HOST_DEVS = frozenset(("CPU", "PYTHON"))

@dataclass(frozen=True)
class HCQInfo:
  device:tuple[str, ...]

  kernels:tuple[tuple, ...] = () # (devices, name, estimates, timestamp slots, profile key, input slots of the buffers, (outs, ins))
  estimates:Estimates = Estimates()

  slots:tuple[tuple[str, int], ...] = () # per device, the position of its batch slots in the args

  skip_wait:bool = False # TODO: remove. an rdma copy between nodes is two batches, so waiting on the first alone deadlocks

def all_devices_in(d:Any, c:frozenset[str]) -> bool: return {x.split(":")[0] for x in to_tuple(d)} <= c

def get_enqueue_devs(call:UOp) -> Any|None:
  if call.op is not Ops.CALL: return None # entries can be AFTER-wrapped calls
  if call.body.op not in (Ops.PROGRAM, Ops.STORE) and not (call.body.op is Ops.CUSTOM_FUNCTION and call.body.arg.name == "encdec"): return None
  if not (bufs:=get_call_arg_uops(call)): return None
  if call.body.op is Ops.STORE: bufs = bufs[::-1] # copies push from the src device: p2p writes are faster than reads
  devs = min(bufs, key=lambda b: not all_devices_in(b.device, HCQ_DEVS)).device
  if not all_devices_in(devs, HCQ_DEVS): return None
  if call.body.op is Ops.STORE and to_tuple(devs)[0].startswith(("QCOM", "METAL")): return None # unified memory uses host copies
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

def select_lane(u:UOp, lane:int) -> UOp: return u.src[lane] if u.op is Ops.MSTACK else u.mselect(lane) if len(to_tuple(u.device)) > 1 else u


def timeline(devs:tuple[str, ...]) -> UOp: return UOp.alloc((2,), dtypes.uint64, 0, device=devs[0]).rtag("timeline")
def timeline_value(devs:tuple[str, ...]) -> UOp: return timeline(devs).index(1).load()

def make_program(prg:UOp, size:int, device:str) -> UOp:
  slot = int.from_bytes(prg.key[:8], "little")
  if PROFILE: Compiled.profile_events.append(ProfileProgramEvent(device, prg.src[0].arg.function_name, prg.src[3].arg, None, slot, prg.key))
  return UOp.alloc((size,), dtypes.uint8, slot, device=device).rtag("program")

def make_submit(*cmds, devs:str|tuple[str, ...], queue:str, fn:str|None=None, deps:tuple[UOp, ...]=()) -> UOp: # the order is on the arg
  lin = UOp(Ops.LINEAR, src=tuple(cmds), arg=(to_tuple(devs), queue)).after(*deps)
  return UOp.custom_function(fn or to_name("submit", to_tuple(devs)[0].split(":")[0], queue.split(":")[0])).call(lin)

def ins(name, *src) -> UOp: return UOp(Ops.INS, tuple(UOp.const(s, dtypes.u32) if isinstance(s, int) else s for s in src), (name, dtypes.void))

def chunks(nbytes:int, chunk_sz:int, dtype:DType=dtypes.int) -> list[tuple[UOp|int, int]]: # (index, bytes): full chunks as one range, then the tail
  full, tail = divmod(nbytes, chunk_sz) # no one-trip loops
  return ([(UOp.range(full, next(UOp.unique_num), dtype=dtype) if full > 1 else 0, chunk_sz)] if full else []) + ([(full, tail)] if tail else [])

# C FFI

def layout_args(args:Sequence[UOp|int], offset:int=0) -> list[tuple[int, UOp]]:
  words = [a if isinstance(a, UOp) else UOp.const(a, dtypes.uint32) for a in args]
  return [(offset + o, w) for (o, _), w in zip(TinyELF.iter_sig(tuple((None, i, w.dtype, ()) for i, w in enumerate(words))), words)]

def pack_args(args:list[tuple[int, UOp]], size:int) -> list[UOp]:
  words, end = [], 0
  for offset, arg in sorted(args, key=lambda x: x[0]):
    words += [UOp(Ops.BINARY, arg=bytes(offset - end)), arg] if offset != end else [arg]
    end = offset + arg.dtype.itemsize
  return words + [UOp(Ops.BINARY, arg=bytes(size - end))]

def ccall(fn:Any, *args:UOp|int) -> UOp:
  ret = dtypes.void if fn.restype is None else dtypes.uint64 if fn.restype is ctypes.c_void_p else \
    next(d for d in DTYPES_DICT.values() if d.fmt == fn.restype._type_)
  return UOp.custom_function(fn.__name__, dtype=ret).call(*[UOp.const(a, dtypes.int) if isinstance(a, int) else a for a in args])

@uopfunc
def do_get_time_ms(ms:UOp) -> UOp:
  ts = UOp.placeholder((2,), dtypes.uint64, addrspace=AddrSpace.REG)
  ts = ts.after(UOp.custom_function("clock_gettime", dtype=dtypes.int).call(UOp.const(time.CLOCK_MONOTONIC, dtypes.int), ts.index(0)))
  return ms.index(0).store(ts[0] * 1000 + ts[1] // 1000000).sink()
def get_time_ms(dep:UOp) -> UOp: return (ms:=UOp.placeholder((1,), dtypes.uint64, addrspace=AddrSpace.REG)).after(do_get_time_ms(ms.after(dep)))[0]

CDTYPE = {1: dtypes.uchar, 2: dtypes.ushort, 4: dtypes.uint, 8: dtypes.ulong} # a C field as the unsigned int of its size

def cstruct(struct_t, **fields:UOp|int) -> UOp:
  flds = {n: (o, CDTYPE[ctypes.sizeof(t)]) for n, t, o, *_ in struct_t._real_fields_ if ctypes.sizeof(t)} # skips zero length arrays
  rows = [(flds[n][0], v.cast(flds[n][1]) if isinstance(v, UOp) else UOp.const(v, flds[n][1])) for n, v in fields.items()]
  buf = UOp.alloc((ctypes.sizeof(struct_t),), dtypes.uint8, device=HCQ_RUNTIME_DEV.value).rtag(struct_t.__name__)
  return patch(buf, rows, bytes(ctypes.sizeof(struct_t)))

def cfield(buf:UOp, struct_t, name:str) -> UOp: return buf[(f:=getattr(struct_t, name)).offset:f.offset + f.size].bitcast(CDTYPE[f.size]).index(0)

# *****************
# 0.1. prep: eager buffers become tagged params

def replace_buffer(ctx:tuple[bool, list[UOp], dict[UOp, int]], b:UOp) -> UOp:
  use_rt, bufs, slots = ctx
  if slots.setdefault(b, len(bufs)) == len(bufs): bufs.append(b)
  param = UOp.param(slots[b], b.dtype, b.max_numel(), b.device)
  return param if use_rt else param.replace(tag="lt_input")
pm_replace_buffers = PatternMatcher([(UPat(Ops.BUFFER, name="b"), lambda ctx, b: replace_buffer(ctx, b))])

# *****************
# 1.1. prep: unwrap multi

def unwrap_call(call:UOp) -> UOp|None:
  if get_enqueue_devs(call) is None or (n:=max(len(to_tuple(a.device)) for a in get_call_arg_uops(call))) == 1: return None
  dnum = UOp.variable("_device_num", 0, n - 1, dtypes.int)
  return UOp(Ops.LINEAR, src=tuple(call.replace(src=(call.body, *[a if a.is_bound_var else select_lane(a, i) for a in call.src[1:]], dnum.bind(i)))
                                   for i in range(n)))
pm_unwrap_multi = PatternMatcher([(UPat(Ops.CALL, name="call"), unwrap_call)])

# *****************
# 1.2. prep: staging copies

STAGING_SIZE, STAGING_SLOTS = (4 if DEV.interface.startswith("MOCK") else 128) << 20, 2

@functools.cache
def _staging(device:str) -> Buffer: return Buffer(device, STAGING_SIZE, preallocate=True)

def split_rdma(call:UOp, dst:UOp, src:UOp) -> UOp|None:
  devs = [to_tuple(b.device)[0] for b in (dst, src)]
  if not all(hasattr(Device[d], "iface") for d in devs) or Device[devs[0]].peer_group == Device[devs[1]].peer_group: return None # not 2 nodes

  from tinygrad.runtime.ops_rdma import rdma_nic_for
  if None in (nics:=[rdma_nic_for(Device[d], Device[min(devs)]) for d in devs]): return None

  # wires: a placeholder per nic in place of the far gpu, tagged by it
  wires = [UOp.alloc(src.max_shape, src.dtype, 0, device=unwrap(nic).device).rtag(peer) for nic, peer in zip(nics, devs[::-1])]
  send = call.replace(src=wires[1].store_call(src).src)
  return UOp(Ops.LINEAR, src=(send, call.replace(src=dst.store_call(wires[0]).src)))

def stage_copy(call:UOp, dst:UOp, src:UOp) -> UOp|None:
  devs = [to_tuple(b.device)[0] for b in (dst, src)]
  if any(d.startswith("RDMA") for d in devs): return None # over the nic

  if (device:=get_enqueue_devs(call)) is None: return None
  dev, host, usb_memcpys = Device[device], Device[device].host, getattr(Device[device], "is_usb", False)
  mappable = {"CPU", "PYTHON"} | ({"NPY", "DISK"} if usb_memcpys else set())
  if device != host and not all(Device[d].peer_group == dev.peer_group or (d.split(":")[0] in mappable and Device[d].host == host) for d in devs):
    (staging:=_staging(host)).get_buf(device)
    base, it, copies = UOp.from_buffer(staging, dtypes.uint8), src.dtype.itemsize, []
    chunk = (STAGING_SIZE // STAGING_SLOTS) // it
    for i, off in enumerate(range(0, src.max_numel(), chunk)):
      stage, part = base[(so:=(i % STAGING_SLOTS) * chunk * it):so + (n:=min(chunk, src.max_numel() - off)) * it], src[off:off+n]
      copies += [stage.store_call(part), dst[off:off+n].store_call(stage)]
    return UOp(Ops.LINEAR, src=tuple(copies))

  if Device[device].has_copy_queue: return None
  out, inp = (UOp.param(i, dtypes.uint8, b.nbytes(), device=device) for i, b in enumerate((dst, src)))
  ast = out.index(r:=UOp.range(src.nbytes(), 0)).store(inp.index(r).load()).end(r).sink(arg=KernelInfo())
  return lower_and_compile(call.replace(src=(ast, *call.src[1:])))

pm_insert_copy_staging = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.STORE), UPat(name="dst"), UPat(name="src")), name="call", allow_any_len=True), split_rdma),
  (UPat(Ops.CALL, src=(UPat(Ops.STORE), UPat(name="dst"), UPat(name="src")), name="call", allow_any_len=True), stage_copy),
])

# *****************
# 2. deps

class DepsTracker:
  def __init__(self):
    # tracks (offset, end, dep) ranges per base buffer/lane to handle suballocated buffers correctly.
    self.w_dependency_map: dict[Any, list[tuple[int, int, Any]]] = defaultdict(list)
    self.r_dependency_map: dict[Any, list[tuple[int, int, Any]]] = defaultdict(list)

  def access_resources(self, bufs:Sequence[UOp|Buffer], write:list[int], new_dependency:Any):
    ranges:list[tuple[Any, int, int]] = []
    for buf in bufs:
      if isinstance(buf, Buffer): ranges.append((id(buf.base), buf.offset, buf.offset + buf.nbytes))
      else:
        base, lane, off = unwrap_lane(buf)
        ranges.append(((base, lane), off, off + buf.max_numel() * buf.dtype.itemsize))
    wait_nodes = []
    for i, (key, s, e) in enumerate(ranges):
      wait_nodes += [dep for st,en,dep in self.w_dependency_map[key] if st < e and s < en]
      if i in write: wait_nodes += [dep for st,en,dep in self.r_dependency_map[key] if st < e and s < en]
    for i, (key, s, e) in enumerate(ranges):
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

@dataclass
class BatchCtx:
  batch:list[tuple[UOp, tuple[str, ...], str]] # (call, devices, queue) per enqueued call
  profile:bool
  tracker:DepsTracker = field(default_factory=DepsTracker)
  queues:dict[str, list[str]] = field(init=False)
  last:dict[tuple[str, str], int] = field(init=False)
  prev:list[int|None] = field(init=False)
  signal_tags:set[int] = field(init=False)
  slots:dict[str, UOp] = field(init=False)
  peers:dict[tuple[str, str], set[str]] = field(init=False) # the other devices whose memory a queue touches

  def __post_init__(self):
    self.queues, self.last, self.prev, self.peers = {}, {}, [], {}
    for tag, (c, devs, q) in enumerate(self.batch):
      if q not in self.queues.setdefault(devs[0], []): self.queues[devs[0]].append(q)
      self.prev.append(self.last.get((devs[0], q)))
      self.last[(devs[0], q)] = tag
      peers = HCQ_DEVS if getattr(Device[devs[0]], "is_usb", False) else HCQ_DEVS | HOST_DEVS # usb gpu can't reach host memory
      for d in {Device.canonicalize(x) for b in get_call_arg_uops(c) for x in to_tuple(b.device) if all_devices_in(x, peers)} - {devs[0]}:
        self.peers.setdefault((devs[0], q), set()).add(d)
        self.queues.setdefault(d, [])
    self.signal_tags = {tag for (dev, q), tag in self.last.items() if q != self.epilogue_queue(dev) or (dev, q) in self.peers}
    # a slot is [signal][timestamp], 16 bytes: the queue signals, the timeline, then two per call if profiling
    self.slots = {dev: UOp.alloc((2 * (len(qs) + 1 + (2 * len(self.batch) if self.profile else 0)),), dtypes.uint64, device=dev,
                                 spec=BufferSpec(host=True, uncached=True, cpu_access=True)).rtag("slots") for dev, qs in self.queues.items()}

  def epilogue_queue(self, dev:str) -> str: return "COMPUTE:0" if len(self.queues[dev]) != 1 else self.queues[dev][0] # closes the device

  def slot(self, devs:tuple[str, ...], i:int) -> UOp: return self.slots[devs[0]].shrink(((2 * i, 2 * i + 2),)) # not a slice: 10x the cost
  def queue_signal(self, devs:tuple[str, ...], queue:str) -> UOp: return self.slot(devs, self.queues[devs[0]].index(queue))
  def sched_timeline(self, devs:tuple[str, ...]) -> UOp: return self.slot(devs, len(self.queues[devs[0]]))
  def stamps(self, devs:tuple[str, ...], tag:int) -> tuple[int, ...]: return (st:=len(self.queues[devs[0]])+1+2*tag, st + 1) if self.profile else ()

def _wait_ins(ctx:BatchCtx, call:UOp, device:str, queue:str, tag:int) -> list[UOp]:
  bufs, write = list(get_call_arg_uops(call)), get_call_outs_ins(call)[0]
  latest:dict[tuple[str, str], int] = {} # (producer device, queue) -> the latest submit tag to wait on, same-queue submits are fifo
  for d, q, t in ctx.tracker.access_resources(bufs, list(range(len(bufs)) if write is None else write), (device, queue, tag)):
    if t < tag and (d, q) != (device, queue): latest[(d, q)] = max(latest.get((d, q), 0), t)

  # NV waits break QMD chaining, so also wait for the previous launch
  if latest and device.split(":")[0] == "NV" and queue.startswith("COMPUTE") and (p:=ctx.prev[tag]) is not None: latest[(device, queue)] = p

  ctx.signal_tags |= set(latest.values())
  return [UOp(Ops.INS, arg=("wait", dtypes.void), src=(ctx.queue_signal((d,), q), UOp.const(t + 1, dtypes.uint64))) for (d, q), t in latest.items()]

def _start_ins(ctx:BatchCtx, dev:str, queue:str) -> list[UOp]: # a queue first waits for prior work of its device and of the peers it touches
  return [UOp(Ops.INS, arg=("barrier", dtypes.void), src=())] + \
    [UOp(Ops.INS, arg=("wait", dtypes.void), src=(timeline((d,)), timeline_value((d,)))) for d in [dev, *sorted(ctx.peers.get((dev, queue), ()))]]

def _build_queues(ctx:BatchCtx) -> dict[tuple[tuple[str, ...], str], list[UOp]]:
  # find all waits first to mark calls that must signal
  call_waits = [_wait_ins(ctx, c, d[0], q, tag) for tag, (c, d, q) in enumerate(ctx.batch)]
  queues:dict[tuple[tuple[str, ...], str], list[UOp]] = {}
  for tag, ((call, devices, queue), waits) in enumerate(zip(ctx.batch, call_waits)):
    if not (q:=queues.setdefault((devices, queue), [])): q += _start_ins(ctx, devices[0], queue) # first use of a queue

    # dependency waits, then the call between its timestamps
    ts_ins = [UOp(Ops.INS, arg=("timestamp", dtypes.void), src=(ctx.slot(devices, i),)) for i in ctx.stamps(devices, tag)]
    q += waits + ts_ins[:1] + [call] + ts_ins[1:]

    # signal the queue if someone waits for us
    if tag in ctx.signal_tags:
      q += [UOp(Ops.INS, arg=("store", dtypes.void), src=(ctx.queue_signal(devices, queue), UOp.const(tag + 1, dtypes.uint64)))]

  # one queue advances the device timeline after all other queues finish, and after the queues of the peers that touched the device
  for dev in ctx.queues:
    touched = sorted(k for k, ds in ctx.peers.items() if dev in ds)
    qdev, queue = touched[0] if dev.split(":")[0] in HOST_DEVS else (dev, ctx.epilogue_queue(dev)) # the host has no queues, a toucher closes it
    waits = [UOp(Ops.INS, arg=("wait", dtypes.void), src=(ctx.queue_signal((d,), q), UOp.const(ctx.last[(d, q)] + 1, dtypes.uint64)))
             for d, q in [(dev, q) for q in ctx.queues[dev]] + touched if (d, q) != (qdev, queue)]
    bump = UOp(Ops.INS, arg=("store", dtypes.void), src=(timeline((dev,)), timeline_value((dev,)) + UOp.const(1, dtypes.uint64)))

    # multiple copy queues may need a new compute stream. a peer without calls of its own starts like any queue
    if not (q:=queues.setdefault(((qdev,), queue), [])) and qdev not in {d for d, _ in ctx.last}: q += _start_ins(ctx, qdev, queue)
    q.extend([*waits, bump])
  return queues

def _finalize_batch(ctx:BatchCtx, skip_wait:bool=False) -> UOp:
  queues = _build_queues(ctx)

  # re-arm the batch signals before submitting queues in first-use order
  submits:list[UOp] = []
  timelines = [ctx.sched_timeline((dev,)) for dev in ctx.queues]
  signals = [ctx.queue_signal((dev,), q) for dev, qs in ctx.queues.items() for q in qs]
  fence = UOp.custom_function("hcq_fence").call(*timelines, *signals)
  for (devs, queue), cmds in queues.items(): submits.append(make_submit(*cmds, devs=devs, queue=queue, deps=(fence, *submits[-1:])))
  sink = UOp.sink(*submits, arg=KernelInfo("hcq_submit", estimates=Estimates()), tag=1)
  for pm in [Device[d].pm_batch for d in ctx.queues]: # a device adds its own work to the batch
    if (r:=pm.rewrite(sink)) is not None: sink = r

  # per call metadata
  names = [get_call_name(c, get_call_arg_uops(c)) for c, _, _ in ctx.batch]
  estimates = [estimate_uop(c) for c, _, _ in ctx.batch]
  stamps = [tuple(2 * s + 1 for s in ctx.stamps(d, tag)) for tag, (_, d, _) in enumerate(ctx.batch)]
  profile_keys = [c.body.key if c.body.op is Ops.PROGRAM else None for c, _, _ in ctx.batch]
  args = [[unwrap_lane(bs[g])[:2] for g in getattr(c.body.arg, "globals", range(len(bs)))] for c, _, _ in ctx.batch for bs in [get_call_arg_uops(c)]]
  bufs = [tuple(b.arg.slot for b, _ in a) if all(b.op is Ops.PARAM and lane is None for b, lane in a) else () for a in args]
  kerns = tuple(zip([d for _, d, _ in ctx.batch], names, estimates, stamps, profile_keys, bufs, [get_call_outs_ins(c) for c, _, _ in ctx.batch]))
  slots = ctx.slots if ctx.profile else {} # profiling reads them
  info = HCQInfo(tuple(ctx.queues), skip_wait=skip_wait, kernels=kerns, estimates=sum(estimates, start=Estimates()).simplify(),
                 slots=tuple((d, i) for i, d in enumerate(slots)))
  return sink.call(*slots.values(), aux=info)

@rewrite_group(new_ctx=False)
def sched_batches(l:UOp, profile:bool) -> UOp:
  devs = [() if (d:=get_enqueue_devs(c)) is None else tuple(Device.canonicalize(x) for x in to_tuple(d)) for c in l.src]

  # assign to queues
  peers = sorted({Device.canonicalize(d) for c in l.src if c.op is Ops.CALL and c.body.op is Ops.STORE
                  for b in get_call_arg_uops(c) for d in to_tuple(b.device) if d.split(":")[0] == "AMD"})
  num_queues = max(1, getenv("HCQ_NUM_SDMA", min(len(peers), 8) if ALL2ALL >= 1 else 1))
  queues = ["COMPUTE:0" if c.op is Ops.CALL and c.body.op is Ops.PROGRAM else "COPY:0" for c in l.src]
  for i, c in enumerate(l.src):
    if c.op is Ops.CALL and c.body.op is Ops.CUSTOM_FUNCTION and c.body.arg.name == "encdec": queues[i] = "ENCDEC:0"
    if c.op is Ops.CALL and c.body.op is Ops.STORE and all(b.device in peers for b in get_call_arg_uops(c)):
      queues[i] = f"COPY:{(peers.index(c.src[1].device) - peers.index(c.src[2].device) - 1) % len(peers) % num_queues}"

  srcs:list[UOp] = []
  for hcq, grp in itertools.groupby(zip(l.src, devs, queues), key=lambda e: bool(e[1])):
    nodes:dict[str, list] = {}
    for e in grp: nodes.setdefault(Device[e[1][0]].peer_group if hcq else "", []).append(e)
    batches = list(nodes.values())
    for batch in batches: srcs += [_finalize_batch(BatchCtx(batch, profile), batch is not batches[-1])] if hcq else [c for c, _, _ in batch]
  return l.replace(src=tuple(srcs))

# *****************
# 3. encode

class HWQueue:
  q_rewrite = PatternMatcher([
    # rewrites from calls
    (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, name="prg"),), name="call", allow_any_len=True), lambda ctx, call, prg: ctx.exec(call, prg)),
    (UPat(Ops.CALL, src=(UPat(Ops.STORE), UPat(name="dst"), UPat(name="src")), allow_any_len=True),
     lambda ctx, dst, src: ctx.copy(dst, src, src.max_numel() * src.dtype.itemsize)),
    (UPat(Ops.CALL, src=(UPat.custom_function("encdec", name="s"),), name="c", allow_any_len=True), lambda ctx, c, s: ctx.encdec(c, s)),

    # ins
    (UPat(Ops.INS, arg=("copy", dtypes.void), src=(UPat(name="dst"), UPat(name="src"), UPat(name="n"))),
     lambda ctx, dst, src, n: ctx.copy(dst, src, n.val)),
    (UPat(Ops.INS, arg=("barrier", dtypes.void)), lambda ctx: ctx.memory_barrier()),
    (UPat(Ops.INS, arg=("wait", dtypes.void), src=(UPat(name="dst"), UPat(name="val"))), lambda ctx, dst, val: ctx.wait(dst, val)),
    (UPat(Ops.INS, arg=("wait_eq", dtypes.void), src=(UPat(name="dst"), UPat(name="val"))), lambda ctx, dst, val: ctx.wait(dst, val, eq=True)),
    (UPat(Ops.INS, arg=("timestamp", dtypes.void), src=(UPat(name="dst"),)), lambda ctx, dst: ctx.timestamp(dst)),
    (UPat(Ops.INS, arg=("store", dtypes.void), src=(UPat(name="dst"), UPat(name="val"))), lambda ctx, dst, val: ctx.signal(dst, val)),
    (UPat(Ops.INS, arg=("write", dtypes.void), name="u"), lambda ctx, u: ctx.write(*u.src)),

    # loop in cmdbufs
    (UPat(Ops.END, src=(UPat(Ops.LINEAR, name="body"), UPat(Ops.RANGE, name="r"))), lambda ctx, body, r: ctx.loop(body, r)),
  ])

  def __init__(self, submit:UOp):
    self.lin, self.deps = (lin:=submit.src[1]).without_after, lin.src[1:] if lin.op is Ops.AFTER else ()
    self.devs, self.queue = self.lin.arg
    self.dev = Device[self.devs[0]]
    self.blob, self.patches = bytearray(), list[tuple[int|UOp, UOp]]()

  def q(self, *words) -> int:
    for w in words:
      c = w
      while isinstance(c, UOp) and c.op is Ops.CAST: c = c.src[0]
      if isinstance(c, UOp) and c.op is Ops.BINARY: self.blob += c.arg
      elif isinstance(c, UOp) and c.op is not Ops.CONST:
        self.patches.append((len(self.blob), w))
        self.blob += bytes(w.dtype.itemsize)
      else:
        v, n = (c.val, w.dtype.itemsize) if isinstance(w, UOp) else (c, 4)
        self.blob += (v & (1 << 8 * n) - 1).to_bytes(n, 'little')
    return len(self.blob)

  def loop(self, body:UOp, r:UOp):
    start, first = len(self.blob), len(self.patches)
    for u in body.src: self.q_rewrite.rewrite(u, ctx=self)
    trip, words = len(self.blob) - start, self.patches[first:]
    # each trip gets its words as dwords: a trip need not be 8 bytes aligned
    self.patches[first:] = [(o + 4 * k + r * trip, (w >> (32 * k)).cast(dtypes.uint32)) for o, w in words for k in range(w.dtype.itemsize // 4)]
    self.blob += self.blob[start:] * int(r.vmax)

  def memory_barrier(self): pass # a copy queue has nothing to flush
  def submit(self, cmdbuf:UOp) -> UOp: raise NotImplementedError("queues need a submit")
  def encode(self) -> UOp: return self.submit(encode_cmdbuf(self, self.lin)) # submit(linear) becomes submit(cmdbuf)

# *****************
# 3.1. hcq special functions

def _is_link_patch(w:UOp) -> bool:
  if w.op is Ops.GETADDR: return not ((base:=unwrap_lane(w.src[0])[0]).op is Ops.PARAM and base.tag is None) # an input's is runtime
  if w.op is Ops.PARAM: return w.tag is not None
  if w.op is Ops.BUFFER: return w.addrspace is AddrSpace.GLOBAL # a register is written at runtime
  if w.op in {Ops.LOAD, Ops.AFTER}: return False
  return all(_is_link_patch(s) for s in w.src)

def patch(buf:UOp, rows:Sequence[tuple[int|UOp, UOp]], blob:bytes|None=None) -> UOp:
  # group into stacks based on dtype, alignment, is_link (rt/lt can't share a store) and ranges (a ranged offset is a loop)
  keys = [(w.dtype, getattr(o, "vmin", o) % w.dtype.itemsize, _is_link_patch(w), tuple(getattr(o, "ranges", ()))) for o, w in rows]
  groups = [(key, [row for row, k in zip(rows, keys) if k == key]) for key in dedup(keys)]

  # a link patch writes the bare buffer after the blob, a runtime patch writes it after its deps
  dep = [buf.without_after.store(UOp(Ops.BINARY, arg=blob).bitcast(buf.dtype))] if blob is not None else []
  base, stores = buf.after(*dep), []
  for (dt, phase, link, rngs), grp in groups:
    view = (buf.without_after.after(*dep) if link else base)[phase:phase + (buf.max_numel() - phase) // dt.itemsize * dt.itemsize].bitcast(dt)
    offs = [UOp.const(i) if isinstance(i:=(o - phase) // dt.itemsize, int) else i for o, _ in grp]
    stores.append(view.index(UOp.stack(*offs)).store(UOp.stack(*[w for _, w in grp])).end(*rngs))
  return buf.after(*dep, *stores)

@uopfunc
def hcq_fence(slots:UOp, tl:UOp, tv:UOp, last:int) -> UOp: # wait for the previous run of this schedule, then announce and record this one
  tl = tl.replace(arg=replace(tl.arg, volatile=True)) # make it volatile, since it's polled

  loop = UOp.range(UOp(Ops.NOOP).after(start:=get_time_ms(target:=slots.index(last).load())), next(UOp.unique_num), dtype=dtypes.void)
  done = tl.after(target, loop).index(0).load()
  bumped = tl.after(done.backedge(loop, (done < target) & (get_time_ms(done) - start < 30000))).index(1).store(nxt:=tv + UOp.const(1, dtypes.uint64))
  return slots.after(bumped).index(last).store(nxt).sink()

def encode_fence(f:UOp) -> UOp:
  devs = dedup(to_tuple(s.device)[0] for s in f.src[1:])
  lasts, sigs = f.src[1:1 + len(devs)], f.src[1 + len(devs):]
  last:tuple[UOp, ...] = ()

  # wait for prev schedule to not collide, the slots are zeroed at link
  for dev, (slots, off) in zip(devs, map(unwrap_view, lasts)):
    slots = patch(slots, [], bytes(slots.nbytes())).after(*last)
    last = (hcq_fence(slots, timeline((dev,)), timeline_value((dev,)), off // slots.dtype.itemsize),)

  # re-arm the signals
  for slots, off in map(unwrap_view, sigs): last = (slots.after(*last).index(off // slots.dtype.itemsize).store(0),)
  return last[0].barrier()
pm_hcq_encode = PatternMatcher([(UPat(Ops.CALL, src=(UPat.custom_function("hcq_fence"),), allow_any_len=True, name="f"), encode_fence)])

# *****************
# 3.2. encode

def encode_cmdbuf(hq:HWQueue, lin:UOp|None=None, name:str="cmdbuf", device:str|tuple[str, ...]|None=None) -> UOp:
  for u in lin.src if lin is not None else (): hq.q_rewrite.rewrite(u, ctx=hq) # the commands of the linear go to the stream
  stream, patches = bytes(hq.blob), hq.patches

  # loop over pathes with the same value
  rt = [(o, w) for o, w in patches if isinstance(o, int) and o % 4 == 0 and w.dtype.itemsize in (4, 8) and not _is_link_patch(w)]
  uses = {w: [o for o, _ in grp] for w, grp in itertools.groupby(sorted(rt, key=lambda p: p[1].key), key=lambda p: p[1])}
  looped = {w: (at, [*w.ranges][0] if w.ranges else UOp.range(len(at), next(UOp.unique_num))) for w, at in uses.items() if len(at) > 1 or w.ranges}
  dwords = [(UOp(Ops.BINARY, arg=struct.pack(f"<{len(at)}I", *at)).bitcast(dtypes.uint32).index(r).load() + 4 * k, (w >> 32 * k).cast(dtypes.uint32))
            for w, (at, r) in looped.items() for k in range(w.dtype.itemsize // 4)]
  patches = [(o, w) for o, w in patches if w not in looped] + dwords
  nested = dedup([g.src[0] for _, w in patches for g in w.toposort() if g.op is Ops.GETADDR and g.src[0].op is Ops.LINEAR])

  # nested linears (like kernargs) merge into a buffer per name, patched before the stream
  bufs = []
  for lname, ls in itertools.groupby(sorted(nested, key=lambda l: l.arg), key=lambda l: l.arg):
    hq.blob, hq.patches = bytearray(), []
    offs = {l: (hq.q(UOp(Ops.BINARY, arg=bytes(-len(hq.blob) % 128))), hq.q(*l.src)) for l in ls}
    bufs.append((offs, encode_cmdbuf(hq, name=lname)))
  views = {l: buf.without_after[o:e] for offs, buf in bufs for l, (o, e) in offs.items()}

  buf = UOp.alloc((len(stream),), dtypes.uint8, device=device or hq.devs[0]).rtag(to_name(name, hq.queue)).after(*hq.deps)
  words = UOp.sink(*[w for _, w in patches]).substitute(views).src
  return patch(buf, list(zip([o for o, _ in patches], words)), stream).after(*[b for _, b in bufs])

# *****************
# 3.3. lift

def hoist_links(ctx:list[UOp], a:UOp) -> UOp|None:
  links, rest = partition(a.src[1:], lambda s: (s.src[0] if s.op is Ops.END else s).op is Ops.STORE and _is_link_patch(s)) # a store or its loop
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
  return u.addrspace is AddrSpace.GLOBAL if u.op in (Ops.BUFFER, Ops.ALLOC) else u.op is Ops.PARAM and u.is_variable != root

def _param_for(u:UOp, slot:int) -> UOp:
  if u.op is Ops.GETADDR or u.is_variable:
    return UOp.param(slot, u.commit_dtype(dtypes.int), name=u.arg.name if u.is_variable else None, addrspace=AddrSpace.ALU).cast(u.dtype)
  return UOp.param(slot, u.dtype, u.max_numel(), HCQ_RUNTIME_DEV.value, name=f"{u.tag}_{slot}" if isinstance(u.tag, str) else None)

def lift(call:UOp, root:bool=False) -> UOp: # callees are lifted already
  body, args = graph_rewrite(call.body, pm_lift_deps + pm_renumber_slots, ctx=itertools.count(), walk=True, name="lift deps"), list(call.src[1:])
  nodes = body.toposort(gate=lambda u: u.op is not Ops.GETADDR, enter_calls=False)
  leaves = dedup([u for u in nodes if _needs_arg(u, root)] + [g for u in nodes for g in u.src if g.op is Ops.GETADDR])
  slots = args + (new:=[u for u in leaves if u not in args])
  body = body.substitute({u: _param_for(u, slots.index(u)) for u in leaves}, walk=True)

  # new args in the caller
  own = {p: args[p.arg.slot].without_after for u in new for p in u.toposort() if p.op is Ops.PARAM and not _needs_arg(p, root)}
  return call.replace(src=(body, *args, *UOp.sink(*new).substitute(own, walk=True).src))

pm_lift = PatternMatcher([
  (UPat(Ops.PARAM, name="u"), lambda u: u.replace(arg=replace(u.arg, device=None)) if u.arg.name else None), # of a lowered function
  (UPat(Ops.CALL, src=(UPat(Ops.SINK),), allow_any_len=True, name="call"), lift), # lift allocs and getaddrs
])

def lower_call(call:UOp) -> UOp|None:
  if not isinstance(call.arg.aux, HCQInfo): return None

  # encode bodies
  from tinygrad.runtime.ops_rdma import pm_rdma_encode
  devs = [Device[d] for d in dedup([d.split(":")[0] for d in call.arg.aux.device])]
  body = graph_rewrite(call.body, pm_rdma_encode + sum([d.pm_encode for d in devs], PatternMatcher([])) + pm_hcq_encode,
                       ctx=(lt_patches:=list[UOp]()), bpm=pm_hoist_links, name="encode")
  body = graph_rewrite(body, sum([d.pm_lower for d in devs], PatternMatcher([])),
                       ctx=lt_patches, bpm=pm_hoist_links, enter_calls=True, name="lower")
  body = graph_rewrite(body, pm_lift, walk=True, enter_calls=True, name="lift")

  if VIZ: graph_rewrite(UOp.sink(*dedup(lt_patches)), PatternMatcher([]), name="View Link-Time Patches")

  # batch args, link patches wait after the call
  return lift(call.replace(src=(body, *call.src[1:])), root=True).after(*dedup(lt_patches))

pm_encode = PatternMatcher([(UPat(Ops.CALL, src=(UPat(Ops.SINK),), name="call", allow_any_len=True), lower_call)])

# *****************
# 4. compile

hcq_compile_cache:dict[tuple[UOp, bool, bool], UOp] = {} # eager templates: a buffer-free linear (uops are hash-consed) to its compiled form

@rewrite_group(lambda linear,input_uops,profile,cache=False,ret=None: f"HCQ Compile {pluralize('Kernel', len(ret.src))}")
def hcq_compile(linear:UOp, input_uops:list[UOp]|None, profile:bool, cache=False) -> UOp:
  if any(isinstance(getattr(c.without_after.arg, "aux", None), HCQInfo) for c in linear.src): return linear # compiled already

  if cache and input_uops is not None:
    use_rt = len(linear.src) < HCQ_CACHE_THRESH # small schedules use runtime address patches so linked schedules can be cached without input buffers
    slots = {u:i for i,u in reversed(tuple(enumerate(input_uops)))}
    linear = graph_rewrite(linear, pm_replace_buffers, ctx=(use_rt, input_uops, slots), walk=True, name="replace buffers")
  linear = graph_rewrite(linear, pm_unwrap_multi+pm_insert_copy_staging+pm_flatten_linear, name="prep calls")
  if cache and input_uops is not None and (cached:=hcq_compile_cache.get(key:=(linear, profile, ALL2ALL >= 1))) is not None: return cached
  lin = graph_rewrite(sched_batches(linear, profile), pm_encode, walk=True, name="encode")
  with Context(EMULATED_DTYPES=""): final_linear = lower_and_compile(lin, verbose=DEBUG>=3)
  if cache and input_uops is not None and final_linear is not linear: hcq_compile_cache[key] = final_linear
  return final_linear

# *****************
# 5. link

Compiled.pm_batch = Compiled.pm_encode = Compiled.pm_lower = PatternMatcher([]) # a device adds its own rules
Compiled.pm_bufferize = PatternMatcher([(UPat(Ops.ALLOC, tag="timeline", name="b"), lambda b: Device[b.device].timeline)])

@dataclass
class LinkCtx: inputs:dict[UOp, UOp]; use_rt:bool; refs:list[UOp] = field(default_factory=list) # noqa: E702

def bufferize_buf(ctx:LinkCtx, b:UOp) -> UOp: # ctx: a kept link (the jit's) owns the linear's buffers, a one-shot borrows ring slots
  dev, spec = Device[b.device], b.arg.spec or BufferSpec(cpu_access=True) # data the device reads, unless the alloc says otherwise

  # a device owns the placeholders it names, the rest are allocated where they live
  if (r:=cast(Buffer|None, Compiled.pm_bufferize.rewrite(b))) is not None: pass
  elif not ctx.use_rt: r = Buffer(dev.device, max(b.max_numel(), 1) * b.dtype.itemsize, options=spec, preallocate=True)
  else: r = dev.rt_buffer(spec).view(b.nbytes(), dev.rt_allocator(spec).alloc(max(b.nbytes(), 1), alignment=256)).ensure_allocated()

  return UOp.from_buffer(r, b.dtype, HCQ_RUNTIME_DEV.value)

def resolve_getaddr(ctx:LinkCtx, g:UOp) -> UOp|None:
  if unwrap_lane(buf:=unwrap_view(g.src[0])[0])[0].op is not Ops.BUFFER: return None # input address, resolved per run
  ctx.refs.append(buf) # add to refs
  return UOp.const(g.val, dtypes.uint64)

def fold_binary(buf:UOp, blob:UOp) -> UOp:
  base, off = unwrap_view(buf)
  cast(Buffer, base.buffer).ensure_allocated().host.view(fmt='B')[off:off + len(blob.arg)] = blob.arg
  return UOp(Ops.NOOP)

def write_words(buf:UOp, writes:list[tuple[int, int, int]]) -> UOp: # (word index, size, value)
  base, off = unwrap_view(buf)
  mv = cast(Buffer, base.buffer).ensure_allocated().host.view(fmt='B')
  for o, n, v in writes: mv[off + o * n:off + (o + 1) * n] = (v & (1 << 8 * n) - 1).to_bytes(n, 'little')
  return UOp(Ops.NOOP)

def words(offs:UOp, ws:UOp): return zip(*[s.src if s.op is Ops.STACK else (s,) for s in (offs, ws)]) # a stack or one word

def fold_words(buf:UOp, offs:UOp, ws:UOp) -> UOp: return write_words(buf, [(o.val, w.dtype.itemsize, w.val) for o, w in words(offs, ws)])

def fold_ranged(buf:UOp, offs:UOp, ws:UOp, e:UOp) -> UOp: # the words of every trip
  vs = {r: UOp.variable(f"r{i}", 0, r.vmax) for i, r in enumerate(e.src[1:])}
  ows = list(words(offs.substitute(vs), ws.substitute(vs)))
  trips = [dict(zip([v.expr for v in vs.values()], t)) for t in itertools.product(*[range(int(r.vmax) + 1) for r in vs])]
  return write_words(buf, [(sym_infer(o, t), w.dtype.itemsize, sym_infer(w, t)) for t in trips for o, w in ows])

pm_link = PatternMatcher([
  # collapse committed const conversions
  (UPat(Ops.CAST, src=(UPat(Ops.CAST, src=(UPat.cvar(),), name="inner"),), name="c"), lambda c, inner: inner.src[0].cast(c.dtype)),
  # an alloc and a link time input become buffers
  (UPat(Ops.ALLOC, name="b"), bufferize_buf),
  (UPat(Ops.PARAM, name="b"), lambda ctx, b: ctx.inputs.get(b)),
  # the address of a buffer is a const
  (UPat(Ops.GETADDR, name="g"), resolve_getaddr),
  # math on consts is a const
  (UPat(GroupOp.ALU, src=UPat.cvar().or_casted(), name="a"), lambda a: UOp.const(exec_alu(a.op, a.dtype, [s.val for s in a.src], False), a.dtype)),
  # fold rules
  (UPat(name="buf").store(UPat.any(UPat(Ops.BINARY, name="blob"), UPat(Ops.BINARY, name="blob").bitcast())), fold_binary),
  (UPat(name="buf").index(UPat(Ops.STACK, src=UPat.cvar(), name="offs")).store(UPat(Ops.STACK, src=UPat.cvar().or_casted(), name="ws")), fold_words),
  (UPat(name="buf").index(UPat.cvar("offs")).store(UPat.cvar().or_casted("ws")), fold_words),
  (UPat(name="buf").index(UPat(name="offs")).store(UPat(name="ws")).end(allow_any_len=True, name="e"), fold_ranged),
  # a call keeps the deps that are not written yet
  (UPat(Ops.AFTER, src=(UPat(Ops.CALL),), allow_any_len=True, name="a"), lambda a: a.src[0].after(*(s for s in a.src[1:] if s.op is not Ops.NOOP))),
  (UPat(Ops.AFTER, name="a"), lambda a: None if a.without_after.op is Ops.CALL else
   a.src[0] if all(s.op is Ops.NOOP for s in a.src[1:]) else panic(RuntimeError, f"unresolved link words on {a.src[0].op}")),
])

link_linear_cache:weakref.WeakKeyDictionary[UOp, UOp] = weakref.WeakKeyDictionary() # a baked link lives as long as its bound linear

@rewrite_group(lambda _,input_uops=None,allow_cache=True,ret=None: f"HCQ Link {pluralize('Kernel', len(ret.src))}")
def hcq_link(linear:UOp, input_uops:list[UOp]|None=None, allow_cache=True) -> UOp:
  if allow_cache and (linked:=link_linear_cache.get(linear)) is not None: return linked

  # if we have any link time buffers, do not cache this linear
  cache = allow_cache and not any(u.tag == "lt_input" for u in linear.toposort(enter_calls=False) if u.op is Ops.PARAM)

  inputs = {UOp.param(i, b.dtype, b.max_numel(), b.device).replace(tag="lt_input"): b for i, b in enumerate(input_uops or ())}
  linked = graph_rewrite(linear, pm_link, ctx=(ctx:=LinkCtx(inputs, use_rt=allow_cache and not cache)), walk=True, name="link")
  if ctx.refs: linked = linked.replace(src=(linked.src[0].after(*dedup(ctx.refs)), *linked.src[1:])) # attach refs to linear
  if cache and linked is not linear: link_linear_cache[linear] = linked
  return linked
