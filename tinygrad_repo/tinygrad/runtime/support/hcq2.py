from __future__ import annotations
from typing import cast, Any, Sequence
import functools, itertools, weakref, ctypes, importlib
from dataclasses import replace, dataclass, field
from tinygrad.helpers import dedup, pluralize, unwrap, VIZ, HCQ2, to_tuple, ContextVar, Context, panic, partition, DEV, ALL2ALL, getenv, round_up
from tinygrad.device import Device, Buffer, BufferSpec, DepsTracker, TinyELF, HCQ_RUNTIME_DEV
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher, KernelInfo, GroupOp, graph_rewrite, rewrite_group, exec_alu
from tinygrad.dtype import dtypes, DType, DTYPES_DICT, AddrSpace
from tinygrad.renderer import Estimates
from tinygrad.schedule.prepare import pm_mops
from tinygrad.engine.realize import get_call_arg_uops, get_call_name, get_call_outs_ins, get_call_written_bufs
from tinygrad.engine.realize import estimate_uop, pm_flatten_linear, lower_and_compile, _resolve

# *****************
# 0. helpers

HCQ_CACHE_THRESH = ContextVar("HCQ_CACHE_THRESH", 64)
HCQ_DEVS = frozenset(("NV", "QCOM", "CUDA")) | (frozenset(("AMD",)) if HCQ2 else frozenset())

@dataclass(frozen=True)
class HCQInfo:
  device:tuple[str, ...]

  kernels:tuple[tuple[tuple[str, ...], str, Estimates, tuple[int, ...], bytes], ...] = () # (devices, name, estimates, timestamp slots, profile key)
  estimates:Estimates = Estimates()

  nargs:int = 0
  table:int = -1
  inputs:tuple[tuple[UOp, str, int], ...] = ()
  slots:tuple[tuple[str, int], ...] = () # per device, the position of its batch slots in the args
  host_deps:tuple[tuple[str, str], ...] = () # (memory owner, accessing device)
  written_bufs:tuple[UOp, ...] = () # write args

  skip_wait:bool = False # TODO: remove. an rdma copy between nodes is two batches, so waiting on the first alone deadlocks

def all_devices_in(d:Any, c:frozenset[str]) -> bool: return {x.split(":")[0] for x in to_tuple(d)} <= c

def get_enqueue_devs(call:UOp) -> Any|None:
  if call.op is not Ops.CALL: return None # entries can be AFTER-wrapped calls
  if call.body.op not in (Ops.PROGRAM, Ops.COPY): return None # only these bodies can be enqueued
  if not (bufs:=get_call_arg_uops(call)): return None
  if call.body.op is Ops.COPY: bufs = bufs[::-1] # copies push from the src device: p2p writes are faster than reads
  devs = min(bufs, key=lambda b: not all_devices_in(b.device, HCQ_DEVS)).device
  if not all_devices_in(devs, HCQ_DEVS): return None
  if call.body.op is Ops.COPY and to_tuple(devs)[0].startswith("QCOM"): return None # QCOM is unified memory and uses host copies
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

def to_name(*parts:str) -> str: return "_".join(parts).replace(":", "_").lower()

def timeline(devs:tuple[str, ...]) -> UOp: return UOp.placeholder((2,), dtypes.uint64, 0, device=devs, volatile=True, tag="timeline")
def timeline_value(devs:tuple[str, ...]) -> UOp: return timeline(devs).index(1).load()

def rt_addr(b:UOp, dev="CPU", *deps:UOp) -> UOp:
  base, off = unwrap_view(b)
  word = UOp.placeholder((1,), dtypes.uint64, device=Device[to_tuple(dev)[0]].host, tag="addr")
  return patch(word, [(0, base.bitcast(dtypes.uint8)[off:off + b.nbytes()].getaddr(dev))]).after(*deps).index(0).load()

def make_submit(*cmds, devs:str|tuple[str, ...], queue:str) -> UOp:
  fn = to_name("submit", (devs:=to_tuple(devs))[0].split(":")[0], queue.split(":")[0])
  return UOp.custom_function(fn, UOp(Ops.LINEAR, src=tuple(cmds), arg=(devs, queue)))

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

@functools.cache
def cfunc_buf(lib:str, name:str) -> Buffer:
  fn = getattr(importlib.import_module(f"tinygrad.runtime.autogen.{lib}").dll, name)
  (b:=Buffer(HCQ_RUNTIME_DEV.value, 1, dtypes.uint64, preallocate=True)).host.view(fmt='Q')[0] = unwrap(ctypes.cast(fn, ctypes.c_void_p).value)
  return b

def ccall(fn:Any, *args:UOp|int) -> UOp:
  ptr = UOp.placeholder((1,), dtypes.uint64, 0, device=HCQ_RUNTIME_DEV.value, tag=("cfunc", fn.__module__.split(".")[-1], fn.__name__))
  ret = dtypes.void if fn.restype is None else dtypes.uint64 if fn.restype is ctypes.c_void_p else \
    next(d for d in DTYPES_DICT.values() if d.fmt == fn.restype._type_)
  cargs = [UOp.const(a, dtypes.int) if isinstance(a, int) else a for a in args]
  return UOp.custom_function(fn.__name__, ptr.index(0).load()).call(*cargs, ret_dtype=ret)

CDTYPE = {1: dtypes.uchar, 2: dtypes.ushort, 4: dtypes.uint, 8: dtypes.ulong} # a C field as the unsigned int of its size

def cstruct(struct_t, **fields:UOp|int) -> UOp:
  flds = {n: (o, CDTYPE[ctypes.sizeof(t)]) for n, t, o, *_ in struct_t._real_fields_ if ctypes.sizeof(t)} # skips zero length arrays
  rows = [(flds[n][0], v.cast(flds[n][1]) if isinstance(v, UOp) else UOp.const(v, flds[n][1])) for n, v in fields.items()]
  buf = UOp.placeholder((ctypes.sizeof(struct_t),), dtypes.uint8, device=HCQ_RUNTIME_DEV.value, volatile=True, tag=struct_t.__name__)
  return patch(buf, rows, bytes(ctypes.sizeof(struct_t)))

def cfield(buf:UOp, struct_t, name:str) -> UOp: return buf[(f:=getattr(struct_t, name)).offset:f.offset + f.size].bitcast(CDTYPE[f.size]).index(0)

# *****************
# 0.1. prep: eager buffers become tagged params

def replace_buffer(ctx:tuple[bool, list[UOp], dict[UOp, int]], b:UOp) -> UOp:
  use_rt, bufs, slots = ctx
  if slots.setdefault(b, len(bufs)) == len(bufs): bufs.append(b)
  param = UOp.param(slots[b], b.dtype, b.max_numel(), b.device)
  return param if use_rt else param.replace(tag="lt_input")
pm_replace_buffers = PatternMatcher([(UPat(Ops.BUFFER, name="b"), replace_buffer)])

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
def _staging(device:str) -> Buffer: return Buffer(device, STAGING_SIZE, dtypes.uint8, preallocate=True)

def split_rdma(call:UOp, dst:UOp, src:UOp) -> UOp|None:
  devs = [to_tuple(b.device)[0] for b in (dst, src)]
  if not all(hasattr(Device[d], "iface") for d in devs) or Device[devs[0]].peer_group == Device[devs[1]].peer_group: return None # not 2 nodes

  from tinygrad.runtime.ops_rdma import rdma_nic_for
  if None in (nics:=[rdma_nic_for(Device[d]) for d in devs]): return None

  # wires: a placeholder per nic in place of the far gpu, tagged by it
  wires = [UOp.placeholder(src.max_shape, src.dtype, 0, device=unwrap(nic).device, tag=peer) for nic, peer in zip(nics, devs[::-1])]
  send = call.replace(src=(call.src[0].replace(arg=wires[1].device), wires[1], src))
  return UOp(Ops.LINEAR, src=(send, call.replace(src=(call.src[0], dst, wires[0]))))

def stage_copy(ctx:tuple[UOp, ...], call:UOp, dst:UOp, src:UOp) -> UOp|None:
  if any(to_tuple(b.device)[0].startswith("RDMA") for b in (dst, src)): return None # over the nic

  if (device:=get_enqueue_devs(call)) is None: return None
  try:
    for b in (dst, src): cast(Buffer, _resolve(b, ctx).buffer).get_buf(device)
  except (RuntimeError, OSError):
    (staging:=_staging(Device[device].host)).get_buf(device)
    base, it, copies = UOp.from_buffer(staging), src.dtype.itemsize, []
    chunk = (STAGING_SIZE // STAGING_SLOTS) // it
    for i, off in enumerate(range(0, src.max_numel(), chunk)):
      stage, part = base[(so:=(i % STAGING_SLOTS) * chunk * it):so + (n:=min(chunk, src.max_numel() - off)) * it], src[off:off+n]
      copies += [part.copy_to_device(staging.device).call(stage, part), stage.copy_to_device(dst.device).call(dst[off:off+n], stage)]
    return UOp(Ops.LINEAR, src=tuple(copies))

  if Device[device].has_copy_queue: return None
  out, inp = (UOp.param(i, dtypes.uint8, b.nbytes(), device=device) for i, b in enumerate((dst, src)))
  ast = out.index(r:=UOp.range(src.nbytes(), 0)).store(inp.index(r).load()).end(r).sink(arg=KernelInfo())
  return lower_and_compile(call.replace(src=(ast, *call.src[1:])))

pm_insert_copy_staging = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.COPY), UPat(name="dst"), UPat(name="src")), name="call", allow_any_len=True), split_rdma),
  (UPat(Ops.CALL, src=(UPat(Ops.COPY), UPat(name="dst"), UPat(name="src")), name="call", allow_any_len=True), stage_copy),
])

# *****************
# 2. deps

class HCQDepsTracker(DepsTracker):
  @staticmethod
  def _key(a:UOp) -> tuple[Any, int, int]: # (base, lane) and the byte range: overlapping views of one base depend
    base, lane, off = unwrap_lane(a)
    return (base, lane), off, off + a.max_numel() * a.dtype.itemsize

@dataclass
class BatchCtx:
  batch:list[tuple[UOp, tuple[str, ...], str]] # (call, devices, queue) per enqueued call
  profile:bool
  tracker:HCQDepsTracker = field(default_factory=HCQDepsTracker)
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
      for d in {Device.canonicalize(x) for b in get_call_arg_uops(c) for x in to_tuple(b.device) if all_devices_in(x, HCQ_DEVS)} - {devs[0]}:
        self.peers.setdefault((devs[0], q), set()).add(d)
        self.queues.setdefault(d, [])
    self.signal_tags = {tag for (dev, q), tag in self.last.items() if q != self.epilogue_queue(dev) or (dev, q) in self.peers}
    # a slot is [signal][timestamp], 16 bytes: the queue signals, the timeline, then two per call if profiling
    self.slots = {dev: UOp.placeholder((2 * (len(qs) + 1 + (2 * len(self.batch) if self.profile else 0)),), dtypes.uint64, device=(dev,),
                                       volatile=True, tag="slots") for dev, qs in self.queues.items()}

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
    queue = ctx.epilogue_queue(dev)
    waits = [UOp(Ops.INS, arg=("wait", dtypes.void), src=(ctx.queue_signal((d,), q), UOp.const(ctx.last[(d, q)] + 1, dtypes.uint64)))
             for d, q in [(dev, q) for q in ctx.queues[dev] if q != queue] + sorted(k for k, ds in ctx.peers.items() if dev in ds)]
    bump = UOp(Ops.INS, arg=("store", dtypes.void), src=(timeline((dev,)), timeline_value((dev,)) + UOp.const(1, dtypes.uint64)))

    # multiple copy queues may need a new compute stream. a peer without calls of its own starts like any queue
    if not (q:=queues.setdefault(((dev,), queue), [])) and dev not in {d for d, _ in ctx.last}: q += _start_ins(ctx, dev, queue)
    q.extend([*waits, bump])
  return queues

def _finalize_batch(ctx:BatchCtx, skip_wait:bool=False) -> UOp:
  queues = _build_queues(ctx)

  # re-arm the batch signals before submitting queues in first-use order
  submits:list[UOp] = []
  timelines = [ctx.sched_timeline((dev,)) for dev in ctx.queues]
  signals = [ctx.queue_signal((dev,), q) for dev, qs in ctx.queues.items() for q in qs]
  fence = UOp.custom_function("hcq_fence", *timelines, *signals)
  for (devs, queue), cmds in queues.items(): submits.append(make_submit(*cmds, devs=devs, queue=queue).after(fence, *submits[-1:]))
  sink = UOp.sink(*submits, arg=KernelInfo("hcq_submit", estimates=Estimates()), tag=1)
  for pm in [Device[d].pm_batch for d in ctx.queues if Device[d].pm_batch is not None]: # a device adds its own work to the batch
    if (r:=pm.rewrite(sink)) is not None: sink = r

  # per call metadata
  names = [get_call_name(c, get_call_arg_uops(c)) for c, _, _ in ctx.batch]
  estimates = [estimate_uop(c) for c, _, _ in ctx.batch]
  stamps = [tuple(2 * s + 1 for s in ctx.stamps(d, tag)) for tag, (_, d, _) in enumerate(ctx.batch)]
  profile_keys = [getattr(c.body.arg, "profile_key", None) for c, _, _ in ctx.batch]
  kerns:tuple[tuple, ...] = tuple(zip([d for _, d, _ in ctx.batch], names, estimates, stamps, profile_keys))
  written_bufs = tuple(dedup(b for c, _, _ in ctx.batch for b in get_call_written_bufs(c)))
  host_deps = tuple(dedup((host, devs[0]) for call, devs, _ in ctx.batch for buf in get_call_arg_uops(call)
                         for host in to_tuple(buf.device) if host not in ctx.queues))
  info = HCQInfo(tuple(ctx.queues), skip_wait=skip_wait, kernels=kerns, written_bufs=written_bufs,
                 estimates=sum(estimates, start=Estimates()).simplify(), host_deps=host_deps)
  return sink.call(*(ctx.slots.values() if ctx.profile else ()), aux=info)

@rewrite_group(new_ctx=False)
def sched_batches(l:UOp, profile:bool) -> UOp:
  devs = [() if (d:=get_enqueue_devs(c)) is None else tuple(Device.canonicalize(x) for x in to_tuple(d)) for c in l.src]

  # assign to queues
  peers = sorted({Device.canonicalize(d) for c in l.src if c.op is Ops.CALL and c.body.op is Ops.COPY
                  for b in get_call_arg_uops(c) for d in to_tuple(b.device) if d.split(":")[0] == "AMD"})
  num_queues = max(1, getenv("HCQ_NUM_SDMA", min(len(peers), 8) if ALL2ALL >= 1 else 1))
  queues = ["COMPUTE:0" if c.op is Ops.CALL and c.body.op is Ops.PROGRAM else "COPY:0" for c in l.src]
  for i, c in enumerate(l.src):
    if c.op is Ops.CALL and c.body.op is Ops.COPY and all(b.device in peers for b in get_call_arg_uops(c)):
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

@dataclass
class EncodeCtx:
  devs:tuple[str, ...]
  inputs:dict[tuple[UOp, str, int], int] = field(default_factory=dict)
  lt_patches:list[UOp] = field(default_factory=list)

  def __post_init__(self): self.table = UOp.placeholder((1,), dtypes.uint64, device=Device[self.devs[0]].host, tag="inputs")

class HWQueue:
  q_rewrite = PatternMatcher([ # the ops of a queue: a queue defines the methods it supports
    (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, name="prg"),), name="call", allow_any_len=True), lambda ctx, call, prg: ctx.exec(call, prg)),
    (UPat(Ops.CALL, src=(UPat(Ops.COPY),), name="call", allow_any_len=True), lambda ctx, call: ctx.copy(call)),
    (UPat(Ops.INS, arg=("barrier", dtypes.void)), lambda ctx: ctx.memory_barrier()),
    (UPat(Ops.INS, arg=("wait", dtypes.void), src=(UPat(name="dst"), UPat(name="val"))), lambda ctx, dst, val: ctx.wait(dst, val)),
    (UPat(Ops.INS, arg=("wait_eq", dtypes.void), src=(UPat(name="dst"), UPat(name="val"))), lambda ctx, dst, val: ctx.wait(dst, val, eq=True)),
    (UPat(Ops.INS, arg=("timestamp", dtypes.void), src=(UPat(name="dst"),)), lambda ctx, dst: ctx.timestamp(dst)),
    (UPat(Ops.INS, arg=("store", dtypes.void), src=(UPat(name="dst"), UPat(name="val"))), lambda ctx, dst, val: ctx.signal(dst, val)),
    (UPat(Ops.INS, arg=("write", dtypes.void), name="u"), lambda ctx, u: ctx.write(*u.src)),
  ])

  def __init__(self, ctx:EncodeCtx, submit:UOp):
    self.ctx, self.lin = ctx, submit.src[0]
    self.devs, self.queue = self.lin.arg
    self.dev = Device[self.devs[0]]
    self.blob, self.patches = bytearray(), list[tuple[int, UOp]]()

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

  def memory_barrier(self): pass # a copy queue has nothing to flush
  def submit(self, cmdbuf:UOp) -> UOp: raise NotImplementedError("queues need a submit")

# *****************
# 3.1. hcq special functions

def hcq_fence(ctx:EncodeCtx, f:UOp) -> UOp:
  lasts, sigs = f.src[:len(ctx.devs)], f.src[len(ctx.devs):]
  last:tuple[UOp, ...] = ()

  # wait for prev schedule to not collide
  # TODO: timeout?
  for i, dev in enumerate(ctx.devs):
    slots, off = unwrap_view(lasts[i])
    slots = patch(slots, [], bytes(slots.max_numel() * slots.dtype.itemsize)) # zeroed at link
    target = slots.after(*last, tv:=timeline_value((dev,))).index(off // slots.dtype.itemsize).load()
    done = timeline((dev,)).after(target, loop:=UOp.loop(i)).index(0).load()
    bumped = timeline((dev,)).after(done.end(loop, done < target)).index(1).store(nxt:=tv + UOp.const(1, dtypes.uint64))
    last = (slots.after(bumped).index(off // slots.dtype.itemsize).store(nxt),)

  # re-arm the signals
  for sig in sigs:
    base, off = unwrap_view(sig)
    last = (base.after(*last).index(off // sig.dtype.itemsize).store(0),)
  return last[0].barrier(*last[1:])

pm_hcq_encode = PatternMatcher([
  (UPat(Ops.CUSTOM_FUNCTION, arg="hcq_fence", name="f"), hcq_fence),

  # after blocks are lowered, rechain stores saving original order
  (UPat(Ops.AFTER, src=(UPat(dtype=dtypes.void, name="root"),), allow_any_len=True, name="a"),
    lambda root, a: root.substitute({s.buf_uop: s.buf_uop.after(*a.src[1:]) for s in root.toposort() if s.op is Ops.STORE}, walk=True)),
])

# *****************
# 3.2. split

def _is_input_addr(g:UOp) -> bool: return (base:=unwrap_lane(g.src[0])[0]).op is Ops.PARAM and base.tag is None

def addrs_to_table(ctx:EncodeCtx, g:UOp) -> UOp|None:
  if not _is_input_addr(g): return None
  base, off = unwrap_view(g.src[0])
  slot = ctx.inputs.setdefault((base, to_tuple(g.arg)[0], off), len(ctx.inputs))
  return ctx.table.index(slot).load()

def _is_link_patch(w:UOp) -> bool:
  if w.op is Ops.GETADDR: return not _is_input_addr(w)
  if w.op is Ops.PARAM: return w.tag is not None
  if w.op is Ops.BUFFER: return w.addrspace is AddrSpace.GLOBAL # a register is written at runtime
  if w.op in {Ops.LOAD, Ops.AFTER} or w.is_variable: return False
  return all(_is_link_patch(s) for s in w.src)

def hoist_links(ctx:EncodeCtx, a:UOp) -> UOp|None:
  links, rest = partition(a.src[1:], lambda s: s.op is Ops.STORE and _is_link_patch(s))
  if not links: return None
  ctx.lt_patches.extend(links)
  return a.src[0].after(*rest)

pm_patches = PatternMatcher([(UPat(Ops.GETADDR, name="g"), addrs_to_table), (UPat(Ops.AFTER, name="a"), hoist_links)])

def patch(buf:UOp, rows:list[tuple[int, UOp]], blob:bytes|None=None) -> UOp:
  groups:dict[tuple[DType, int, bool], list[tuple[int, UOp]]] = {} # split by: dtype, alignment, is_link (rt/lt can't share a store)
  for offb, w in rows: groups.setdefault((w.dtype, offb % w.dtype.itemsize, _is_link_patch(w)), []).append((offb, w))

  dep = [buf.store(UOp(Ops.BINARY, arg=blob).bitcast(buf.dtype))] if blob is not None else []
  base, stores = buf.after(*dep), [] # keep buf.after to be sure that link applies patches after the blob
  for (dt, phase, _), grp in groups.items():
    view = base[phase:phase + (buf.max_numel() - phase) // dt.itemsize * dt.itemsize].bitcast(dt)
    stores.append(view.index(UOp.stack(*[UOp.const((o - phase) // dt.itemsize) for o, _ in grp])).store(UOp.stack(*[w for _, w in grp])))
  return buf.after(*dep, *stores)

def bufferize_linear(hq:HWQueue, name:str, device:str|tuple[str, ...]) -> UOp:
  stream, patches = bytes(hq.blob), hq.patches
  nested = dedup([g.src[0] for _, w in patches for g in w.toposort() if g.op is Ops.GETADDR and g.src[0].op is Ops.LINEAR])

  # nested linears (like kernargs) merge into a buffer per name, patched before the stream
  bufs = []
  for lname, ls in itertools.groupby(sorted(nested, key=lambda l: l.arg), key=lambda l: l.arg):
    hq.blob, hq.patches = bytearray(), []
    offs = {l: (hq.q(UOp(Ops.BINARY, arg=bytes(-len(hq.blob) % 128))), hq.q(*l.src)) for l in ls}
    bufs.append((offs, bufferize_linear(hq, lname, hq.devs)))
  views = {l: buf.without_after[o:e] for offs, buf in bufs for l, (o, e) in offs.items()}

  buf = UOp.placeholder((len(stream),), dtypes.uint8, device=device, tag=to_name(name, hq.queue))
  words = UOp.sink(*[w for _, w in patches]).substitute(views).src
  return patch(buf, list(zip([o for o, _ in patches], words)), stream).after(*[b for _, b in bufs])

def encode_submit(hq:HWQueue) -> UOp:
  for u in hq.lin.src: hq.q_rewrite.rewrite(u, ctx=hq)
  return hq.submit(bufferize_linear(hq, "cmdbuf", hq.devs))

# *****************
# 4. lower call

def bitcast_view(x:UOp, v:UOp, b:UOp) -> UOp|None:
  (o, n), k, m = v.marg[0], x.dtype.itemsize, b.dtype.itemsize
  return x.bitcast(b.dtype)[o*k//m:(o+n)*k//m] if len(v.shape) == 1 and not ((o*k) % m or (n*k) % m or (x.max_numel()*k) % m) else None

pm_views = PatternMatcher([
  # a shrink of a shrink is one shrink
  (UPat(Ops.SHRINK, name="x").f(Ops.SHRINK, allow_any_len=True, name="s"),
   lambda s,x: x.src[0].shrink(tuple((o+p, o+p+n) for (o,_),(p,n) in zip(x.marg, s.marg)))),
  # a bitcast of a 1-d view of storage is a view of the bitcast, so pm_mops folds the view into the index
  (UPat((Ops.PARAM, Ops.BUFFER)).or_after("x").f(Ops.SHRINK, allow_any_len=True, name="v").bitcast().named("b"), bitcast_view),
])

pm_renumber = PatternMatcher([
  (UPat(Ops.RANGE, name="u"), lambda ctx, u: u.replace(arg=(next(ctx),)+u.arg[1:])),
  (UPat(Ops.BUFFER, name="u"), lambda ctx, u: u.replace(arg=replace(u.arg, slot=next(ctx))) if u.addrspace is AddrSpace.REG else None),
])

def lower_call(call:UOp) -> UOp|None:
  if not isinstance(call.arg.aux, HCQInfo) or call.arg.aux.nargs: return None # not an hcq call, or lowered already

  # encode bodies
  from tinygrad.runtime.ops_rdma import pm_rdma_encode
  ctx = EncodeCtx(call.arg.aux.device)
  devs = [Device[d] for d in dedup([d.split(":")[0] for d in ctx.devs])]
  body = graph_rewrite(call.body, pm_rdma_encode + sum([d.pm_encode for d in devs if d.pm_encode is not None], PatternMatcher([])) + pm_hcq_encode,
                       ctx=ctx, bpm=pm_patches, name="encode")
  body = graph_rewrite(body, sum([d.pm_lower for d in devs if d.pm_lower is not None], PatternMatcher([])), ctx=ctx, bpm=pm_patches, name="lower")

  # resize table
  body = body.substitute({ctx.table: (table:=ctx.table.replace(arg=replace(ctx.table.arg, size=len(ctx.inputs))))})

  # combine placeholders into one and replace with views
  words = [u for u in body.toposort() if u.op is Ops.PARAM and u.tag not in (None, "program") and u.arg.slot]
  keys = {u: (u.tag, u.device, u.dtype, u.arg.volatile) for u in words}
  groups = [[u for u in words if keys[u] == k] for k in dedup(list(keys.values()))]
  # each becomes a 128-byte aligned view
  offs = {g[0]: list(itertools.accumulate([round_up(u.nbytes(), 128) // u.dtype.itemsize for u in g], initial=0)) for g in groups}
  merged = {g[0]: g[0].replace(arg=replace(g[0].arg, size=offs[g[0]][-1])) for g in groups if len(g) > 1}
  views = {u: merged[g[0]][o:o + u.max_numel()] for g in groups if len(g) > 1 for u, o in zip(g, offs[g[0]])}
  body = body.substitute(views, extra_pm=pm_mops+pm_views, enter_calls=True)
  patches = UOp.sink(*dedup(ctx.lt_patches)).substitute(views).src

  # the placeholders become the body's params in visit order, variables bind by name after them, the ranges renumber
  bufs, alus = partition([u for u in body.toposort() if u.op is Ops.PARAM], lambda u: u.tag is not None)
  bufs = dedup([*call.src[1:], *bufs])
  names = dedup([a.arg.name for a in alus])
  # bufs to params
  params = {b: UOp.param(i, b.dtype, b.shape, HCQ_RUNTIME_DEV.value, volatile=b.arg.volatile, name=f"{b.arg.name}_{i}") for i, b in enumerate(bufs)}
  # new slots for vars
  vals = {a: a.replace(arg=replace(a.arg, slot=len(bufs) + names.index(a.arg.name))) for a in alus}
  sink = graph_rewrite(body.substitute(params | vals, enter_calls=True), pm_renumber, ctx=itertools.count(), walk=True, enter_calls=True)

  if VIZ: graph_rewrite(UOp.sink(*patches), PatternMatcher([]), name="View Link-Time Patches")
  if VIZ: graph_rewrite(sink, PatternMatcher([]), name="View Body")

  info = replace(call.arg.aux, nargs=len(bufs), table=bufs.index(table) if table in bufs else -1, inputs=tuple(ctx.inputs),
                 slots=tuple((to_tuple(b.device)[0], i) for i, b in enumerate(bufs) if b.tag == "slots"))
  return call.replace(src=(sink, *bufs), arg=replace(call.arg, aux=info)).after(*patches)
pm_encode = PatternMatcher([(UPat(Ops.CALL, src=(UPat(Ops.SINK),), name="call", allow_any_len=True), lower_call)])

hcq_compile_cache:dict[tuple[UOp, bool, bool], UOp] = {} # eager templates: a buffer-free linear (uops are hash-consed) to its compiled form

@rewrite_group(lambda linear,input_uops,profile,cache=False,ret=None: f"HCQ Compile {pluralize('Kernel', len(ret.src))}")
def hcq_compile(linear:UOp, input_uops:list[UOp]|None, profile:bool, cache=False) -> UOp:
  if any(isinstance(getattr(c.without_after.arg, "aux", None), HCQInfo) for c in linear.src): return linear # compiled already

  if cache and input_uops is not None:
    use_rt = len(linear.src) < HCQ_CACHE_THRESH # small schedules use runtime address patches so linked schedules can be cached without input buffers
    slots = {u:i for i,u in reversed(tuple(enumerate(input_uops)))}
    linear = graph_rewrite(linear, pm_replace_buffers, ctx=(use_rt, input_uops, slots), walk=True, name="replace buffers")
  linear = graph_rewrite(linear, pm_unwrap_multi+pm_insert_copy_staging+pm_flatten_linear, ctx=tuple(input_uops or ()), name="prep calls")
  if cache and input_uops is not None and (cached:=hcq_compile_cache.get(key:=(linear, profile, ALL2ALL >= 1))) is not None: return cached
  lin = graph_rewrite(sched_batches(linear, profile), pm_encode, walk=True, name="encode")
  with Context(EMULATED_DTYPES=""): final_linear = lower_and_compile(lin)
  if cache and input_uops is not None and final_linear is not linear: hcq_compile_cache[key] = final_linear
  return final_linear

# *****************
# 5. link

@dataclass
class LinkCtx: inputs:dict[UOp, UOp]; use_rt:bool; refs:list[UOp] = field(default_factory=list) # noqa: E702

def bufferize_buf(ctx:LinkCtx, b:UOp) -> UOp|None: # ctx: a kept link (the jit's) owns the linear's buffers, a one-shot borrows ring slots
  if b.tag is None: return None # a param, not a placeholder

  dev = Device[to_tuple(b.device)[0]]

  # device owns the placeholders it names
  if (r:=cast(Buffer|None, dev.pm_bufferize.rewrite(b, ctx=dev))) is not None: pass
  elif not ctx.use_rt:
    spec = BufferSpec(host=b.arg.volatile, uncached=b.arg.volatile or b.tag.startswith("cmdbuf"), cpu_access=True)
    r = Buffer(dev.device, b.max_numel(), b.dtype, options=spec, preallocate=True)
  else:
    off = dev.rt_allocator(True, b.arg.volatile).alloc(max(b.max_numel() * b.dtype.itemsize, 1), alignment=256)
    r = dev.rt_buffer(True, b.arg.volatile).view(b.max_numel(), b.dtype, off).ensure_allocated()

  return UOp.from_buffer(r, HCQ_RUNTIME_DEV.value)

def resolve_getaddr(ctx:LinkCtx, g:UOp) -> UOp|None:
  buf, off = unwrap_view(g.src[0])
  if buf.op not in {Ops.BUFFER, Ops.MSELECT}: return None
  ctx.refs.append(buf) # add to refs
  return UOp.const(cast(Buffer, buf.buffer).get_buf(to_tuple(g.arg)[0]) + off, dtypes.uint64)

def fold_binary(buf:UOp, blob:UOp) -> UOp:
  base, off = unwrap_view(buf)
  if getattr(b:=cast(Buffer, base.buffer), '_hcq_written', {}).get(off) is not blob.arg: # TODO: remove me
    cast(Any, b.ensure_allocated())._hcq_written = getattr(b, '_hcq_written', {}) | {off: blob.arg}
    b.host.view(fmt='B')[off:off + len(blob.arg)] = blob.arg
  return UOp(Ops.NOOP)

def fold_words(buf:UOp, offs:UOp, ws:UOp) -> UOp:
  base, off = unwrap_view(buf)
  mv = cast(Buffer, base.buffer).ensure_allocated().host.view(fmt='B')
  for o, w in zip(offs.src, ws.src):
    n, at = w.dtype.itemsize, off + o.val * w.dtype.itemsize
    mv[at:at + n] = (w.val & (1 << 8 * n) - 1).to_bytes(n, 'little')
  return UOp(Ops.NOOP)

pm_link = PatternMatcher([
  (UPat(Ops.CAST, src=(UPat(Ops.CAST, src=(UPat.cvar(),), name="inner"),), name="c"), lambda c, inner: inner.src[0].cast(c.dtype)),
  (UPat(Ops.PARAM, name="b"), lambda ctx, b: ctx.inputs[b] if b in ctx.inputs else bufferize_buf(ctx, b)),
  (UPat(Ops.GETADDR, name="g"), resolve_getaddr),
  (UPat(GroupOp.ALU, src=UPat.cvar().or_casted(), name="a"),
    lambda a: UOp.const(exec_alu(a.op, a.dtype, [s.val for s in a.src], False), a.dtype)),
  (UPat(name="buf").store(UPat.any(UPat(Ops.BINARY, name="blob"), UPat(Ops.BINARY, name="blob").bitcast())), fold_binary),
  (UPat(name="buf").index(UPat(Ops.STACK, src=UPat.cvar().or_casted(), name="offs")).store(UPat(Ops.STACK, src=UPat.cvar().or_casted(), name="ws")),
    fold_words),
  (UPat(Ops.AFTER, src=(UPat(Ops.CALL),), allow_any_len=True, name="a"),
    lambda a: a.src[0].after(*(s for s in a.src[1:] if s.op is not Ops.NOOP))),
  (UPat(Ops.AFTER, name="a"), lambda a: None if a.is_bound_var or a.src[0].op is Ops.CALL else
   a.src[0] if all(s.op is Ops.NOOP for s in a.src[1:]) else panic(RuntimeError, f"unresolved link words on {a.src[0].op}")),
])

link_linear_cache:weakref.WeakKeyDictionary[UOp, UOp] = weakref.WeakKeyDictionary() # a baked link lives as long as its bound linear

@rewrite_group(lambda _,input_uops=None,allow_cache=True,ret=None: f"HCQ Link {pluralize('Kernel', len(ret.src))}")
def hcq_link(linear:UOp, input_uops:list[UOp]|None=None, allow_cache=True) -> UOp:
  if allow_cache and (linked:=link_linear_cache.get(linear)) is not None: return linked

  # if we have any link time buffers, do not cache this linear
  cache = allow_cache and not any(u.tag == "lt_input" for u in linear.toposort() if u.op is Ops.PARAM)

  inputs = {UOp.param(i, b.dtype, b.max_numel(), b.device).replace(tag="lt_input"): b for i, b in enumerate(input_uops or ())}
  linked = graph_rewrite(linear, pm_link, ctx=(ctx:=LinkCtx(inputs, use_rt=allow_cache and not cache)), walk=True, name="link")
  if ctx.refs: linked = linked.replace(src=(linked.src[0].after(*dedup(ctx.refs)), *linked.src[1:])) # attach refs to linear
  if cache and linked is not linear: link_linear_cache[linear] = linked
  return linked
