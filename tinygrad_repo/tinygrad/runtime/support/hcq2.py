from __future__ import annotations
from typing import cast, TypeVar, Generic, Any, Sequence, TYPE_CHECKING
import struct, functools, time, collections, itertools, decimal, statistics
from dataclasses import replace, dataclass, field
from tinygrad.helpers import suppress_finalizing, dedup, pluralize, JIT_BATCH_SIZE, unwrap, PROFILE, all_same, all_int
from tinygrad.helpers import to_tuple, ContextVar, perf_counter_us, Context, panic, partition, round_up, flatten, next_power2
from tinygrad.device import Device, Buffer, BufferSpec, Compiled, LRUAllocator, MultiBuffer, DepsTracker
from tinygrad.device import ProfileDeviceEvent, ProfileGraphEntry, ProfileGraphEvent
from tinygrad.uop.ops import Ops, sint, UOp, UPat, PatternMatcher, KernelInfo, graph_rewrite, rewrite_group, GroupOp
from tinygrad.uop.symbolic import symbolic
from tinygrad.dtype import dtypes, DType, AddrSpace
from tinygrad.runtime.support.memory import BumpAllocator, MMIOInterface
from tinygrad.renderer import Renderer, Estimates
from tinygrad.engine.realize import to_program, get_call_arg_uops, get_call_name, get_call_outs_ins, estimate_uop, pm_flatten_linear
from tinygrad.engine.realize import lower_and_compile

if TYPE_CHECKING: from tinygrad.runtime.support.hcq import HCQBuffer # TODO: remove that

# *****************
# 0. helpers

HCQDeviceType = TypeVar('HCQDeviceType', bound='HCQ2Compiled')
HCQ_RUNTIME_DEV = ContextVar("HCQ_RUNTIME_DEV", "CPU")
HCQ_DEVS = frozenset(("AMD", "CPU"))

@dataclass(frozen=True)
class HCQInfo:
  device:tuple[str, ...]
  estimates:Estimates = Estimates()

  kernels:tuple[tuple[tuple[str, ...], str, Estimates, tuple[int, ...], bytes], ...] = ()

  args:tuple[tuple[UOp, UOp], ...] = () # placeholder -> the canonical body param it becomes, in call src order
  table:Any = None # the inputs table placeholder (its src position after lower)
  inputs:tuple[tuple[UOp, int, str], ...] = () # per table slot: (src, lane, device) the exec resolves to an address
  vals:tuple[tuple[str, int], ...] = () # bound values of the body variables (the queue byte size, table slots)

def all_devices_in(d:Any, c:frozenset[str]) -> bool: return {x.split(":")[0] for x in to_tuple(d)} <= c

def unwrap_mstack(u:UOp) -> tuple[UOp, ...]:
  if u.op is Ops.MSTACK: return tuple(x for s in u.src for x in unwrap_mstack(s))
  return unwrap_mstack(u.src[0]) if u.op is Ops.MSELECT else (u,)

def unwrap_view(v:UOp) -> tuple[UOp, int]: # look through views to (base, element offset)
  return unwrap_view(v.src[0]) if v.op in (Ops.BITCAST, Ops.AFTER) else (v.src[0], v.src[1].val) if v.op is Ops.SHRINK else (v, 0)

def _lane(u:UOp, lane:int) -> UOp: return u.src[lane] if u.op is Ops.MSTACK else u.mselect(lane) if len(to_tuple(u.device)) > 1 else u

def make_submit(*cmds, devs:str|tuple[str, ...], queue:str) -> UOp:
  return UOp.custom_function("submit_cmdbuf", UOp(Ops.LINEAR, src=tuple(cmds), arg=(to_tuple(devs), queue)))
def get_submit(ast:UOp) -> UOp|None:
  return next((u for u in ast.toposort() if u.op is Ops.CUSTOM_FUNCTION and u.arg == "submit_cmdbuf"), None)

def make_call(name:str, body:UOp, info:HCQInfo) -> UOp: return UOp.custom_function("hcq", body).call(name=name, aux=info)

def hcq_size_var(cmdbuf:UOp) -> UOp: # the sealed byte count, bounded by the cmdbuf it walks so the submit copy loops stay in bounds
  return UOp.variable("hcq_size", 0, cmdbuf.max_numel() * cmdbuf.dtype.itemsize, dtypes.uint32, param=True)

def make_buf(devs, slot:int=0, tag:str="signal") -> UOp: return UOp.placeholder((1,), dtypes.uint64, slot, device=devs, volatile=True, tag=tag)

# *****************
# 0.1. prep: replace buffers with params

def replace_call_buffers(ctx:tuple[list[UOp], dict[UOp, int]], call:UOp) -> UOp|None:
  bufs, slots = ctx
  for s in call.src[1:]:
    if s.op is not Ops.PARAM and not s.is_bound_var and slots.setdefault(s, len(bufs)) == len(bufs): bufs.append(s)
  return call.replace(src=call.src[:1] + tuple(s if s.op is Ops.PARAM or s.is_bound_var else s.param_like(slots[s]) for s in call.src[1:]))
pm_replace_buffers = PatternMatcher([(UPat(Ops.CALL, name="call"), replace_call_buffers)])

# *****************
# 1.1. prep: staging copies

STAGING_SIZE, STAGING_SLOTS = 128 << 20, 2

@functools.cache
def _staging() -> Buffer: return Buffer("CPU", STAGING_SIZE, dtypes.uint8, preallocate=True)

def _need_staging(a, b): return all_devices_in(a.device, HCQ_DEVS - {"CPU"}) and not all_devices_in(b.device, HCQ_DEVS)

def stage_copy_ext(call:UOp) -> UOp|None:
  if (d:=next((d for b in call.src[1:] for d in to_tuple(b.device) if not d.startswith("CPU")), None)) is None: return None
  return pm.rewrite(call) if (pm:=getattr(Device[d], "pm_stage_copy", None)) is not None else None

def stage_copy(dst:UOp, src:UOp) -> UOp|None:
  if not (_need_staging(src, dst) or _need_staging(dst, src)): return None

  assert src.dtype.itemsize == dst.dtype.itemsize, "staged copies must be dtype-size matched"
  base, it, copies = UOp.from_buffer(_staging()), src.dtype.itemsize, []
  chunk = (STAGING_SIZE // STAGING_SLOTS) // it
  for i, off in enumerate(range(0, src.max_numel(), chunk)):
    stage = base[(so:=(i % STAGING_SLOTS) * chunk * it):so + (n:=min(chunk, src.max_numel() - off)) * it]
    copies += [src[off:off+n].copy_to_device("CPU").call(stage, src[off:off+n]), stage.copy_to_device(dst.device).call(dst[off:off+n], stage)]
  return UOp(Ops.LINEAR, src=tuple(copies))

# *****************
# 1.2. prep: kernel copies

def _get_enqueue_devs(call:UOp) -> Any|None:
  if call.src[0].op not in (Ops.PROGRAM, Ops.COPY): return None # only these bodies can be enqueued
  if not (bufs:=call.src[1:]) or not all(all_devices_in(b.device, HCQ_DEVS) for b in bufs): return None
  if call.src[0].op is Ops.COPY: bufs = bufs[::-1] # copies push from the src device: p2p writes are faster than reads
  devs = min(bufs, key=lambda b: to_tuple(b.device)[0].startswith("CPU")).device # prio to enqueue on not CPU device
  return devs if all_devices_in(devs, HCQ_DEVS) else None

def copy_with_kernel(call:UOp, dst:UOp, src:UOp) -> UOp|None:
  if (devs:=_get_enqueue_devs(call)) is None or Device[(dev:=to_tuple(devs)[0])].has_copy_queue: return None
  d, s = (UOp.param(i, dst.dtype, n:=dst.max_numel(), device=devs) for i in range(2))
  ast = d.index(r:=UOp.range(n, 0)).store(s.index(r).load()).end(r).sink(arg=KernelInfo(name="copy"), tag=1)
  return call.replace(src=(to_program(ast, Device[dev].renderer), dst, src))

pm_insert_copy_staging = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.COPY),), name="call", allow_any_len=True), stage_copy_ext),
  (UPat(Ops.CALL, src=(UPat(Ops.COPY), UPat(name="dst"), UPat(name="src"))), stage_copy),
  (UPat(Ops.CALL, src=(UPat(Ops.COPY), UPat(name="dst"), UPat(name="src")), name="call"), copy_with_kernel)
])

# *****************
# 2. deps

class HCQDepsTracker(DepsTracker):
  @staticmethod
  def _key(buf:Any) -> tuple[Any, int, int]:
    if isinstance(buf, UOp) and buf.op is Ops.MSELECT: buf = buf.src[0]
    return (buf.arg.slot, 0, buf.max_numel() * buf.dtype.itemsize) if isinstance(buf, UOp) else DepsTracker._key(buf)

@dataclass(frozen=True)
class Dep: dev:str; queue:str; tag:int; lane:int # producer submit (dev, queue, tag) awaited by consumer lane # noqa: E702

@dataclass
class BatchCtx:
  batch:list[tuple[UOp, tuple[str, ...]]]; batch_info:list[tuple[tuple[str, ...], str]]; profile:bool # noqa: E702
  tracker:HCQDepsTracker = field(default_factory=HCQDepsTracker); signal_tags:set[int] = field(default_factory=set) # noqa: E702
  slots:dict[str, int] = field(default_factory=lambda: collections.defaultdict(lambda: next(UOp.unique_num)))

def _get_call_bufs_by_lane(call:UOp, devices:tuple[str, ...]) -> list[list[Any]]:
  def dep_buf(b:UOp) -> Any: return base if (base:=(b.src[0] if b.op is Ops.MSELECT else b).base).op is Ops.PARAM else b.buffer
  return [[dep_buf(_lane(a, lane)) for a in get_call_arg_uops(call)] for lane in range(len(devices))]

def _wait_ins(ctx:BatchCtx, bufs_by_lane:list[list[Any]], write, devices:tuple[str, ...], queue:str, tag:int) -> list[UOp]:
  deps:list[Dep] = []
  for lane, bufs in enumerate(bufs_by_lane):
    written = write if write is not None else list(range(len(bufs)))
    deps += [Dep(d, q, t, lane) for d, q, t in ctx.tracker.access_resources(bufs, written, (devices[lane], queue, tag)) if t < tag]

  # same-queue submits are fifo-ordered, no wait needed
  if devices[0].split(":")[0] in {"AMD", "QCOM", "CPU"} or queue.startswith("COPY"):
    deps = [d for d in deps if (d.dev, d.queue) != (devices[d.lane], queue)]
  latest = {(d.dev, d.queue, d.lane): d for d in sorted(deps, key=lambda d: d.tag)}

  # keep only the latest signal
  rows:dict[tuple[str, int], dict[int, list[str]]] = collections.defaultdict(lambda: collections.defaultdict(list))
  for d in latest.values(): rows[(d.queue, d.tag)][d.lane].append(d.dev)
  waits = []
  for (dqueue, dtag), by_lane in rows.items():
    for ds in itertools.zip_longest(*(by_lane[lane] for lane in range(len(devices)))):
      sig = UOp.mstack(*[make_buf(d, tag="sentinel_signal") if dd is None else make_buf(dd, ctx.slots[dqueue]) for dd, d in zip(ds, devices)])
      waits.append(UOp(Ops.INS, arg=("wait", dtypes.void), src=(sig, UOp.const(dtag + 1, dtypes.uint64))))
  ctx.signal_tags |= {t for _, t in rows}
  return waits

def _merge_submits(calls:list[UOp]) -> UOp:
  if len(calls) == 1: return calls[0]
  devs, queue = unwrap(get_submit(calls[0])).src[0].arg
  body = make_submit(*[cmd for c in calls for cmd in unwrap(get_submit(c)).src[0].src], devs=devs, queue=queue).sink()
  return make_call(f"submit {queue} ({len(calls)})", body, replace(calls[0].arg.aux,
    estimates=sum((c.arg.aux.estimates for c in calls), start=Estimates()).simplify()))

def _merge_queues(submits:list[UOp]) -> list[UOp]:
  # CPU submits run inline and can block on another queue. Keep multi-queue CPU work in schedule order so every
  # producer queue is submitted before a CPU wait; merging by queue can otherwise deadlock alternating dependencies.
  keys = [unwrap(get_submit(call)).src[0].arg for call in submits]
  if len(set(keys)) > 1 and any(any(d.split(":")[0] == "CPU" for d in devs) for devs, _ in keys): return submits

  merged:list[UOp] = []
  opened:dict[tuple[tuple[str, ...], str], list[UOp]] = {} # (devs, queue) -> hcq calls in submit order
  limits:dict[tuple[tuple[str, ...], str], int] = collections.defaultdict(lambda: JIT_BATCH_SIZE.value)
  for call in submits:
    devs, queue = key = unwrap(get_submit(call)).src[0].arg
    if (group:=opened.pop(key, None)) is None:
      # first submit on this queue: close open groups on the same queue with shared devices, so submit order is kept
      for k in [k for k in opened if k[1] == queue and set(k[0]) & set(devs)]: merged.append(_merge_submits(opened.pop(k)))
      group = []
    elif limits[key] and len(group) >= limits[key]: merged, group, limits[key] = merged + [_merge_submits(group)], [], limits[key] * 2
    opened[key] = group + [call]
  return merged + [_merge_submits(g) for g in opened.values()]

def _make_finalizers(ctx:BatchCtx) -> tuple[list[UOp], list[UOp], list[UOp]]:
  # collect all buffers which belong to devices
  dev_bufs:dict[str, dict[int, Any]] = collections.defaultdict(dict)
  for call, devices in ctx.batch:
    for b in itertools.chain.from_iterable(_get_call_bufs_by_lane(call, devices)):
      for bd in to_tuple(b.device): dev_bufs[bd][id(b)] = b

  n, fences, resets, fins = len(ctx.batch_info), [], [], []
  for _, devgroup in itertools.groupby(sorted(dev_bufs), key=lambda d: d.split(":")[0]):
    sched_epoch = make_buf(devs:=tuple(devgroup), next(UOp.unique_num), tag="epoch")
    tl_signal, tl_value = make_buf(devs, tag="timeline_signal"), make_buf(devs, tag="timeline_value")

    # fence: spin until the device timeline reaches this schedule's previous epoch
    done = tl_signal.after(loop:=UOp.loop(0)).index(0).load()
    fences.append(make_call("hcq_fence", UOp.sink(done.end(loop, done < sched_epoch.index(0).load())), HCQInfo(devs)))

    # reset: queues of other groups wait on this group's signals, zero them only after every group reached its epoch
    qs = dedup([qn for bdevs, qn in ctx.batch_info if set(bdevs) & set(devs)])
    rst:tuple[UOp, ...] = ()
    for q in qs: rst += (make_buf(devs, ctx.slots[q]).after(*rst[-1:]).index(0).store(0),)
    if rst: resets.append(make_call("hcq_reset", UOp.sink(*rst), HCQInfo(devs)))

    # finalizer: the submit stores the host timeline into the device timeline signal, then the host bumps the
    # timeline and remembers this schedule's epoch for the next fence
    waits = _wait_ins(ctx, [list(dev_bufs[d].values()) for d in devs], None, devs, "COMPUTE:0", n)
    fin_submit = make_submit(*waits, UOp(Ops.INS, arg=("store", dtypes.void), src=(tl_signal, tl_value.index(0))), devs=devs, queue="COMPUTE:0")
    epoch = (epoch_slot:=tl_value.after(fin_submit).index(0)).load()
    fins.append(make_call("hcq_finalizer", UOp.sink(epoch_slot.store(epoch + 1), sched_epoch.after(fin_submit).index(0).store(epoch)), HCQInfo(devs)))
  return fences, resets, fins

def _emit_submits(ctx:BatchCtx, call_waits:list[list[UOp]]) -> tuple[list[UOp], list[tuple]]:
  # one submit per call: timeline sync on first queue use, timestamps, the call, and a signal if someone waits on it
  src, kerns, seen_queues = [], [], set()
  for tag, ((call, _), (devices, queue), q) in enumerate(zip(ctx.batch, ctx.batch_info, call_waits)):
    # first queue use, sync prior device work with the device timeline
    if (devices, queue) not in seen_queues:
      seen_queues.add((devices, queue))
      epoch = make_buf(devices, tag="timeline_value").index(0) - 1
      q = [UOp(Ops.INS, arg=("barrier", dtypes.void), src=()),
           UOp(Ops.INS, arg=("wait", dtypes.void), src=(make_buf(devices, tag="timeline_signal"), epoch))] + q

    # and make hcq call
    name, info = get_call_name(call, get_call_arg_uops(call)), HCQInfo(devices, estimate_uop(call))
    ts_ids = [next(UOp.unique_num) for _ in range(2)] if ctx.profile else []
    kerns.append((devices, name, info.estimates, tuple(ts_ids), make_call(name, call.src[0], info).key))

    ts_ins = [UOp(Ops.INS, arg=("timestamp", dtypes.void), src=(make_buf(devices, s),)) for s in ts_ids]
    q += ts_ins[:1] + [call.replace(arg=replace(call.arg, aux=info))] + ts_ins[1:]

    # signal the queue if someone waits for us
    if tag in ctx.signal_tags:
      q += [UOp(Ops.INS, arg=("store", dtypes.void), src=(make_buf(devices, ctx.slots[queue]), UOp.const(tag + 1, dtypes.uint64)))]
    src.append(make_call(f"submit {name}", make_submit(*q, devs=devices, queue=queue).sink(), info))
  return src, kerns

def _finalize_batch(batch:list[tuple[UOp, tuple[str, ...]]], profile:bool) -> list[UOp]:
  ctx = BatchCtx(batch, [(devices, "COMPUTE:0" if call.src[0].op is Ops.PROGRAM else "COPY:0") for call, devices in batch], profile)

  call_waits = [_wait_ins(ctx, _get_call_bufs_by_lane(call, devices), get_call_outs_ins(call)[0], devices, queue, tag)
                for tag, ((call, _), (devices, queue)) in enumerate(zip(ctx.batch, ctx.batch_info))]
  fences, resets, fins = _make_finalizers(ctx)
  submits, kerns = _emit_submits(ctx, call_waits)

  # append batch kernels to the finalizers, their exec collects the profiles after everything is in flight
  fins = [f.replace(arg=replace(f.arg, aux=replace(a:=f.arg.aux, kernels=tuple(x for x in kerns if set(x[0]) & set(a.device))))) for f in fins]
  return fences + resets + _merge_queues(submits) + fins

@rewrite_group(new_ctx=False)
def sched_batches(l:UOp, profile:bool) -> UOp:
  srcs:list[UOp] = []
  batch:list[tuple[UOp, tuple[str, ...]]] = []
  for call in l.src:
    if (devs:=_get_enqueue_devs(call)) is not None: batch.append((call, to_tuple(devs)))
    else: srcs, batch = srcs + _finalize_batch(batch, profile) + [call], []
  return l.replace(src=tuple(srcs + _finalize_batch(batch, profile)))

# *****************
# 3. encode: the backend rewrites the ops of every submit into flat command words. a word is a const, a uop the
#    link or the exec resolves to a value, or a getaddr of a nested LINEAR (an indirect blob like kernargs)

class EncodeCtx: # devs/queue (and for the submit lowering, the sealed byte count) plus everything on the device
  def __init__(self, dev, devs:tuple[str, ...], queue:str, nbytes:int=0): self.dev, self.devs, self.queue, self.nbytes = dev, devs, queue, nbytes
  def __getattr__(self, name): return getattr(self.dev, name)

def encode_call(call:UOp) -> UOp|None:
  if (submit:=get_submit(call.src[0])) is None or (lin:=submit.src[0]).op is not Ops.LINEAR: return None
  if not any(w.op in {Ops.INS, Ops.CALL} for w in lin.src): return None # already flat words
  devs, queue = lin.arg
  ctx = EncodeCtx(dev:=Device[devs[0]], devs, queue)
  body = graph_rewrite(call.src[0], dev.pm_encode[queue.split(":")[0]] + pm_flatten_linear, ctx=ctx, name=f"encode {queue}")
  return call.replace(src=(body, *call.src[1:])) if body is not call.src[0] else None
pm_encode = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="hcq"),), name="call", allow_any_len=True), encode_call)])

# *****************
# 4. seal: pack the words into one blob per linear tag. consts bake into the blobs, link values fold in at link time,
#    and the rest the body stores each call from the exec-filled inputs table and the link-filled offset table.
#    the submit keeps only its buffer: custom_function(submit) -> placeholder

def is_link_value(w:UOp) -> bool: # resolvable when the linear links: no variables or memory reads, no input-param or program addresses
  if w.op is Ops.GETADDR: return all(b.op is not Ops.PARAM or b.tag is not None for b in unwrap_mstack(w.buf_uop))
  if w.op in {Ops.LOAD, Ops.INDEX} or w.is_variable or (w.op is Ops.PARAM and w.arg.addrspace is AddrSpace.ALU): return False
  return all(is_link_value(s) for s in w.src)

def blobify(ph:UOp, data:bytes, stores:Sequence[tuple[int, UOp]]=()) -> UOp:
  binary = ph.store(UOp(Ops.BINARY, src=(), arg=data).bitcast(ph.dtype))
  return ph.after(binary, *[ph.shrink(((o, o + w.dtype.itemsize),)).bitcast(w.dtype).index(0).store(w) for o, w in stores])

def seal_call(call:UOp) -> UOp|None:
  if (submit:=get_submit(call.src[0])) is None or (lin:=submit.src[0]).op is not Ops.LINEAR: return None
  devs, queue = lin.arg

  # pack every linear into its own blob
  patches:list[tuple[UOp, int, UOp]] = [] # (linear, local byte offset, word)
  blobs:dict[UOp, bytearray] = {l: bytearray() for l in [lin] + [u for u in lin.toposort() if u.op is Ops.LINEAR and u is not lin]}

  for l, blob in blobs.items():
    for w in l.src:
      if is_uop:=isinstance(c:=w.ssimplify(), UOp): patches.append((l, len(blob), w)) # the original w: simplify can rebuild nested linears
      blob += (b"\xbe" * w.dtype.itemsize) if is_uop else cast(int, c).to_bytes(w.dtype.itemsize, 'little')

  # merge blobs based on tags
  tags, offs, datas = {l: l.tag or ("cmdbuf", queue) for l in blobs}, {}, collections.defaultdict[Any, bytearray](bytearray)
  for l, b in blobs.items():
    offs[l] = len(datas[tags[l]])
    datas[tags[l]] += b.ljust(round_up(len(b), 128), b"\xbf")

  bufs = {t: UOp.placeholder((next_power2(len(d)),), dtypes.uint8, next(UOp.unique_num), device=devs).rtag(t) for t, d in datas.items()}
  views = {l: bufs[tags[l]][offs[l]:offs[l] + len(blobs[l])] for l in blobs}

  # place the words in the merged blobs, then split: link words fold at link time, runtime words the body stores every call
  placed = UOp.sink(*[w for _, _, w in patches]).substitute(views).src
  links, runtime = partition([(bufs[tags[l]], offs[l] + o, w) for (l, o, _), w in zip(patches, placed)], lambda p: is_link_value(p[2]))

  rt_sink = UOp.sink(*[w for _, _, w in runtime])
  rt_vars = {u: u.src[0] for u in rt_sink.toposort() if u.is_bound_var}

  # all getaddrs are one input table. the body walks it on the host, so it's a CPU buffer: an emulated runtime device has no memory of its own
  gaddrs = dedup([g for g in rt_sink.toposort() if g.op is Ops.GETADDR])
  table_srcs = dedup([g.src[0].without_after for g in gaddrs])
  slots = {src: i * len(devs) for i, src in enumerate(table_srcs)}
  table = UOp.placeholder((tsz:=next_power2(len(slots)*len(devs)),), dtypes.uint64, next(UOp.unique_num), device="CPU").rtag("inputs")
  dvar = UOp.variable("_device_num", 0, len(devs) - 1, dtypes.int, param=True) if len(devs) > 1 else UOp.const(0, dtypes.int)
  reads = {g: table.index(slots[g.src[0].without_after] + dvar).load() for g in gaddrs}

  # group rt-patches by target and uop
  groups:dict[tuple[UOp, UOp], list[tuple[int, int]]] = collections.defaultdict(list)
  for (buf, off, w), v in zip(runtime, rt_sink.substitute(reads | rt_vars).src):
    if w.op is Ops.GETADDR: groups[(buf, table)].append((off, slots[w.src[0].without_after]))
    else: groups[(buf, v)].append((off, 0))

  stores, vals, base = [], [], UOp.const(0, dtypes.int)
  offtbl = UOp.placeholder((osz:=next_power2(2 * len(runtime)),), dtypes.uint32, next(UOp.unique_num), device="CPU").rtag("offtbl")
  for j, ((buf, v), grp) in enumerate(groups.items()):
    n = UOp.variable(f"hcq_off_len{j}", 0, osz // 2, dtypes.uint32, param=True)
    vals.append((n.arg.name, len(grp)))

    # the lens sum to the entry count, so both masks are no-ops. they just keep the table indices provably in bounds
    r = UOp.range(n, 20 + j, dtype=dtypes.int, src=(buf,))
    ent = 2 * ((base + r) & (osz // 2 - 1))
    off = offtbl.index(ent).load().cast(dtypes.int)
    val = table.index((offtbl.index(ent + 1).load().cast(dtypes.int) + dvar) & (tsz - 1)).load() if v is table else v # reindex table
    stores.append(buf.shrink(((off, off + val.dtype.itemsize),)).bitcast(val.dtype).index(0).store(val).end(r))
    base = base + n.cast(dtypes.int)

  patched = bufs[tags[lin]].after(*stores)
  body = call.src[0].substitute({submit: submit.replace(src=(patched,))})

  # link-time patches are just stores
  link_stores:dict[UOp, list[tuple[int, UOp]]] = collections.defaultdict(list)
  for b, off, w in links: link_stores[b].append((off, w))

  # blobs, blobs
  data_blobs = [blobify(offtbl, struct.pack(f"<{2*len(runtime)}I", *flatten(flatten(groups.values()))))] if runtime else []
  link_blobs = [blobify(bufs[t], bytes(d), link_stores[bufs[t]]) for t, d in datas.items()]
  prog_blobs = dedup([u for u in UOp.sink(*[w for _, _, w in links]).toposort() if u.op is Ops.AFTER])
  info = replace(call.arg.aux, table=table if table_srcs else call.arg.aux.table, vals=(("hcq_size", len(blobs[lin])), *vals),
                 inputs=call.arg.aux.inputs + tuple((src, lane, dev) for src in table_srcs for lane, dev in enumerate(devs)))
  return call.replace(src=(body, *dedup([*call.src[1:], *link_blobs, *data_blobs, *prog_blobs])), arg=replace(call.arg, aux=info))

pm_seal = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="hcq", src=(UPat(Ops.SINK),)),), name="call", allow_any_len=True), seal_call)])

# *****************
# 5. lower submits: the backend's pm_lower turns custom_function(submit, cmdbuf) into the queue push. per-call values
#    it needs (like the cmdbuf address) are written as plain uops: the body lowering routes them through the input table

def lower_submit_call(call:UOp) -> UOp|None:
  if (submit:=get_submit(call.src[0])) is None: return None

  devs, queue = to_tuple((cmdbuf:=submit.src[0]).device), cmdbuf.without_after.tag[1]
  ctx = EncodeCtx(Device[devs[0]], devs, queue, dict(call.arg.aux.vals)["hcq_size"])
  return call.replace(src=(call.src[0].substitute({submit: unwrap(ctx.dev.pm_lower[queue.split(":")[0]].rewrite(submit, ctx=ctx))}), *call.src[1:]))

pm_lower_submit = PatternMatcher([(UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="hcq"),), name="call", allow_any_len=True), lower_submit_call)])

# *****************
# 7. lower the hcq bodies (submits, fences, finalizers) to plain programs: per-call addresses go through the input
#    table, placeholders become canonically-sized params so one body program is shared across batches

def lower_hcq_call(call:UOp) -> UOp|None:
  if call.arg.aux.args or get_submit(call.src[0]) is not None: return None # lowered already, or the submit isn't lowered yet

  variables = (body:=call.src[0]).variables()
  tops = body.toposort(gate=lambda u: u.op not in {Ops.PARAM, Ops.MSTACK})
  placeholders = dedup([s for u in tops for s in u.src if s.op in {Ops.PARAM, Ops.MSTACK} and s not in variables])

  # args -> params
  args = {b: UOp.param(i, b.dtype, shape=b.shape, device=HCQ_RUNTIME_DEV.value, volatile=any(x.arg.volatile for x in unwrap_mstack(b)))
          for i, b in enumerate(placeholders)}

  # vars slots aft args
  base = max([len(args)] + [v.arg.slot + 1 for v in variables]) # above every existing slot, or a renumber can cycle
  vrs = {v: v.replace(arg=replace(v.arg, slot=base + i)) for i, v in enumerate(sorted(variables, key=lambda v: (v.arg.name, v.arg.slot)))}

  # reenum ranges
  rngs = {r: r.replace(arg=(i,)+r.arg[1:]) for i,r in enumerate(sorted([u for u in tops if u.op is Ops.RANGE], key=lambda r: r.arg))}

  sink = body.src[0].substitute(cast(dict[UOp, UOp], args) | vrs | rngs).replace(arg=KernelInfo("hcq_submit"), tag=1)

  # args the link writes into keep their seal after
  patched = {s.without_after: s for s in call.src[1:] if s.op is Ops.AFTER}
  arg_src = [patched.get(b, b) for b in placeholders]
  src = (body.replace(src=(sink,)), *arg_src, *[s for s in call.src[1:] if s not in arg_src])

  table = None if (t:=call.arg.aux.table) is None else next(i for i, x in enumerate(src) if x.without_after is t)
  return call.replace(src=src, arg=replace(call.arg, aux=replace(call.arg.aux, table=table, args=tuple(args.items()))))

pm_lower_hcq = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="hcq", src=(UPat(Ops.SINK),)),), name="call", allow_any_len=True), lower_hcq_call)])

# *****************
# 6. batch: adjacent hcq calls fold into one submitter on the host SUBMIT:0 queue: a submit whose cmds call the
#    compiled piece programs, so the batch runs in fifo order and the python exec is one submitter call

def _lane_arg(a:UOp, lane:int) -> UOp: return a.mselect(lane) if len(to_tuple(a.device)) > 1 else a

def _batch_hcq_calls(calls:list[UOp]) -> UOp:
  # flatten tables + views for each cmd
  table = UOp.placeholder((next_power2(sum(len(c.arg.aux.inputs) for c in calls)),), dtypes.uint64, next(UOp.unique_num),
                          device=HCQ_RUNTIME_DEV.value).rtag("inputs")
  offs = itertools.accumulate((len(c.arg.aux.inputs) for c in calls), initial=0)
  views = {c: table[off:off + len(c.arg.aux.inputs)] for c, off in zip(calls, offs)}

  # build submitter
  def cmd(c:UOp, j:int) -> UOp:
    args = [views[c] if i == c.arg.aux.table else _lane_arg(c.src[i].without_after, j) for i in range(1, 1 + len(c.arg.aux.args))]
    vals = [UOp.variable(n, 0, 0xffffffff).bind(v) for n, v in c.arg.aux.vals]
    return c.src[0].src[0].call(*args, *vals, UOp.variable("_device_num", 0, 1 << 30).bind(j))
  cmds = [cmd(c, j) for c in calls for j in range(len(to_tuple(c.arg.aux.device)))]
  body = UOp.custom_function("hcq", make_submit(*cmds, devs=HCQ_RUNTIME_DEV.value, queue="SUBMIT:0").sink())

  # update info
  estimates = sum((c.arg.aux.estimates for c in calls), start=Estimates()).simplify()
  kernels = sum((c.arg.aux.kernels for c in calls), start=())
  inputs = sum((c.arg.aux.inputs for c in calls), start=())
  cargo = dedup([table] + [s for c in calls for s in c.src[1:] if s.without_after.tag != "inputs"])
  info = HCQInfo((HCQ_RUNTIME_DEV.value,), estimates, kernels=kernels, table=table, inputs=inputs)
  return body.call(*cargo, name=f"hcq_submitter ({len(calls)})", aux=info)

def batch_hcq_linear(linear:UOp) -> UOp:
  def _key(c:UOp): return c.op is Ops.CALL and c.src[0].op is Ops.CUSTOM_FUNCTION and c.src[0].arg == "hcq"
  return linear.replace(src=tuple(flatten([_batch_hcq_calls(list(g))] if k else g for k, g in itertools.groupby(linear.src, key=_key))))

# *****************
# 8. hcq compile

def hcq_lower(linear:UOp, name:str="lower") -> UOp:
  linear = graph_rewrite(linear, pm_encode + pm_seal + pm_lower_submit + pm_lower_hcq, bottom_up=True, name=name)
  with Context(EMULATED_DTYPES=""): return lower_and_compile(linear)

hcq_compile_cache:dict[tuple[bytes, bool], UOp] = {}

@rewrite_group(lambda linear,input_uops,profile,ret: f"HCQ Compile {pluralize('Kernel', len(ret.src))}")
def hcq_compile(linear:UOp, input_uops:list[UOp]|None, profile:bool) -> UOp:
  if input_uops is not None:
    slots = {u:i for i,u in reversed(tuple(enumerate(input_uops)))}
    linear = graph_rewrite(linear, pm_replace_buffers, ctx=(input_uops, slots), walk=True, name="replace buffer")

  if (final_linear:=(hcq_compile_cache.get(cache_key:=(linear.key, profile)))) is None:
    # prep
    linear = linear.substitute(back_map:={s.param_like(i): s for i,s in enumerate(input_uops)} if input_uops is not None else {}, walk=True)
    linear = graph_rewrite(linear, pm_insert_copy_staging+pm_flatten_linear, name="insert copy staging")

    # schedule on real buffers
    linear = sched_batches(linear, profile).substitute({s:p for p,s in back_map.items()}, walk=True, enter_calls=True)

    # lower
    linear = hcq_lower(linear)
    final_linear = hcq_compile_cache[cache_key] = \
      hcq_lower(batch_hcq_linear(linear), name="lower c submitter") if HCQ_RUNTIME_DEV.value == "CPU" else linear

  return final_linear

# *****************
# 9. bufferize placeholders: replace placeholders with real buffers

def bufferize_buf(ctx:tuple[bool, list[UOp]], buf:UOp) -> UOp|None:
  if buf.tag is None: return None
  return UOp.mstack(*(UOp.from_buffer((dv:=Device[dev]).pm_bufferize.rewrite(buf, ctx=(dv, ctx[0])), HCQ_RUNTIME_DEV.value)
                      for dev in to_tuple(buf.device)))
pm_bufferize = PatternMatcher([(UPat(Ops.PARAM, name="buf"), bufferize_buf)])

# *****************
# 10. link: bufferize the placeholders, then the patch stores fold into plain memory writes

def push_stack(op:UOp) -> UOp|None:
  if not (ns:=[s for s in op.src if s.op is Ops.STACK]) or not all_same([len(s.src) for s in ns]): return None
  return UOp(Ops.STACK, src=tuple(op.replace(src=tuple(s.src[i] if s.op is Ops.STACK else s for s in op.src)) for i in range(len(ns[0].src))))

def _bufs(buf:UOp) -> list[Buffer]: # one Buffer per lane
  if buf.op is Ops.MSTACK: return [cast(Buffer, x.buffer) for x in buf.src]
  return list(m.bufs) if isinstance(m:=buf.buffer, MultiBuffer) else [m]

def fold_binary(buf:UOp, blob:UOp) -> UOp:
  for b in _bufs(buf):
    if getattr(b, '_hcq_written', None) is not blob.arg: # programs are shared across linears, write them once
      cast(Any, b.ensure_allocated())._hcq_written = blob.arg
      b._buf.cpu_view().view(fmt='B')[:len(blob.arg)] = blob.arg
  return UOp(Ops.NOOP)

def fold_word_store(view:UOp, idx:UOp, val:UOp) -> UOp|None:
  vals = [v.ssimplify() for v in (val.src if val.op is Ops.STACK else (val,))]
  if not all_int(vals): return None
  buf, start = unwrap_view(view)
  width, bo = view.dtype.itemsize, start * buf.dtype.itemsize + idx.val * view.dtype.itemsize
  for b, c in zip(_bufs(buf), itertools.cycle(vals)): # a single value broadcasts over the lanes
    b.ensure_allocated()._buf.cpu_view().view(fmt='B')[bo:bo+width] = (c & (1 << 8 * width) - 1).to_bytes(width, 'little')
  return UOp(Ops.NOOP)

def resolve_getaddr(ctx:tuple[bool, list[UOp]], buf:UOp, g:UOp) -> UOp:
  ctx[1].append(buf) # the address bakes into the blob, the linked linear refholds the buffer (amd scratch outlives its realloc)
  devs, bufs = to_tuple(g.arg), _bufs(buf)
  if len(bufs) == 1: bufs = bufs * len(devs) # one buffer shared by every lane
  assert len(bufs) == len(devs), f"can't resolve {len(bufs)} buffers on {len(devs)} devices"
  addrs = tuple(UOp.const(x.get_buf(d).va_addr, dtypes.uint64) for x, d in zip(bufs, devs))
  return addrs[0] if len(addrs) == 1 else UOp(Ops.STACK, src=addrs)

def resolve_getaddr_view(bv:UOp, g:UOp) -> UOp:
  addr = UOp(Ops.GETADDR, src=(bv.src[0],), arg=g.arg)
  return addr if bv.op is Ops.BITCAST else addr + UOp.const(bv.src[1].val * bv.dtype.itemsize, dtypes.uint64)

pm_resolve_patches = PatternMatcher([
  # multi
  (UPat(GroupOp.ALU | {Ops.CAST}, name="op"), push_stack),

  # getaddr
  (UPat(Ops.GETADDR, src=(UPat(Ops.AFTER, name="a"),), name="g"), lambda a, g: g.replace(src=(a.src[0],))),
  (UPat(Ops.GETADDR, src=(UPat((Ops.SHRINK, Ops.BITCAST), name="bv"),), name="g"), resolve_getaddr_view),
  (UPat(Ops.GETADDR, src=(UPat((Ops.BUFFER, Ops.MSTACK, Ops.MSELECT), name="buf"),), name="g"), resolve_getaddr),

  # folders
  (UPat(name="buf").store(UPat.any(UPat(Ops.BINARY, name="blob"), UPat(Ops.BINARY, name="blob").bitcast())), fold_binary),
  (UPat((Ops.BITCAST, Ops.SHRINK, Ops.BUFFER, Ops.MSTACK), name="view")
    .index(UPat(Ops.CONST, name="idx")).store(UPat(name="val")), fold_word_store),
])

pm_assert_no_afters = PatternMatcher([(UPat(Ops.AFTER, name="a"), lambda a: panic(RuntimeError, f"AFTER left at hcq_link: {a.src[0].op}"))])

link_linear_cache:dict[bytes, UOp] = {}

@rewrite_group(lambda _,cache,ret: f"HCQ Link {pluralize('Kernel', len(ret.src))}")
def hcq_link(linear:UOp, cache=True) -> UOp:
  if (linked:=link_linear_cache.get(linear_key:=linear.key)) is not None: return linked
  refs:list[UOp] = []
  linear = graph_rewrite(linear, pm_resolve_patches+symbolic+pm_assert_no_afters, bpm=pm_bufferize, ctx=(cache, refs), bottom_up=False,
                         name="resolve patches")
  if refs: linear = linear.replace(src=(linear.src[0].replace(src=linear.src[0].src + tuple(dedup(refs))), *linear.src[1:]))
  if cache: link_linear_cache[linear_key] = linear
  return linear

# *****************
# Device classes

class HCQ2Compiled(Compiled):
  timestamp_divider: float = 1000.0
  wait_timeout_ms: float = 30000.0
  rt_nbytes: int = 64 << 20 # the pool every per-linear buffer is carved out of

  def __init__(self, device:str, allocator:HCQAllocator, compilers:list[type[Renderer]], runtime, can_recover:bool=False, arch=None):
    self.can_recover = can_recover

    self.pm_bufferize = PatternMatcher([
      (UPat(Ops.PARAM, tag="sentinel_signal"), lambda ctx: ctx[0].signal("sentinel", (1 << 64) - 1)),
      (UPat(Ops.PARAM, tag="timeline_signal"), lambda ctx: ctx[0].signal("timeline")),
      (UPat(Ops.PARAM, tag="timeline_value"), lambda ctx: ctx[0].signal("value", 1, device="CPU")),
      (UPat(Ops.PARAM, tag="epoch", name="b"), lambda ctx, b: ctx[0].signal(b.arg.slot, device="CPU")),
      (UPat(Ops.PARAM, tag="signal", name="b"), lambda ctx, b: ctx[0].signal(b.arg.slot)),
      (UPat(Ops.PARAM, name="b"), lambda ctx, b: None if b.tag is None else ctx[0].new_buffer(b, cache=ctx[1]))
    ])

    super().__init__(device, allocator, compilers, runtime, None, arch=arch)

    self.rt_allocator = BumpAllocator(self.rt_nbytes)
    self.prog_bufs:dict[UOp, Buffer] = {}
    self.prof_ents:dict[int, ProfileGraphEntry] = {}

  def collect_prof(self):
    if PROFILE:
      es = list(self.prof_ents.values())
      sigs = [self.signal(i)._buf.cpu_view().view(fmt='Q')[0]/decimal.Decimal(self.timestamp_divider) for e in es for i in (e.st_id, e.en_id)]
      Compiled.profile_events.append(ProfileGraphEvent([replace(e, st_id=2*i, en_id=2*i+1) for i,e in enumerate(es)], [], sigs))
    self.prof_ents.clear()

  def _at_profile_finalize(self):
    from tinygrad.tensor import Tensor
    tdiffs = []
    for _ in range(5):
      with Context(DEBUG=0, BEAM=0, TRACK_MATCH_STATS=0): Tensor.ones(1, device=self.device).contiguous().realize()
      if not (ents:=list(self.prof_ents.values())): return
      self.prof_ents.clear()
      st = perf_counter_us()
      self.synchronize()
      gpu = max(self.signal(e.en_id)._buf.cpu_view().view(fmt='Q')[0] for e in ents)/decimal.Decimal(self.timestamp_divider)
      tdiffs.append((st+perf_counter_us())/2 - gpu)
    Compiled.profile_events.append(ProfileDeviceEvent(self.device, statistics.median(tdiffs), self.device_props()))

  @functools.cache
  def rt_buffer(self, uncached:bool=True) -> Buffer:
    return Buffer(self.device, self.rt_allocator.size, dtypes.uint8, options=BufferSpec(uncached=uncached, cpu_access=True), preallocate=True)

  def rt_view(self, nbytes:int, dtype:DType=dtypes.uint8, uncached:bool=True) -> Buffer:
    return self.rt_buffer(uncached).view(nbytes // dtype.itemsize, dtype, self.rt_allocator.alloc(max(nbytes, 1), alignment=128)).ensure_allocated()

  def new_buffer(self, b:UOp, cache:bool) -> Buffer:
    if b.tag == "program": # program buffers are shared across linears, keyed on the placeholder
      if (buf:=self.prog_bufs.get(b)) is None:
        buf = self.prog_bufs[b] = Buffer(self.device, b.max_numel(), b.dtype, options=BufferSpec(cpu_access=True, nolru=True)).ensure_allocated()
      return buf
    return self.rt_view(b.max_numel() * b.dtype.itemsize, b.dtype)

  @functools.cache
  def signal(self, name:str|int, init_value:int=0, device:str|None=None) -> Buffer:
    buf = Buffer(device or self.device, 1, dtypes.uint64, options=BufferSpec(host=True, uncached=True, cpu_access=True), preallocate=True)
    buf._buf.cpu_view().view(fmt='Q')[0] = init_value
    return buf

  def _wait_signal(self, sig:MMIOInterface|memoryview, value:int, timeout:int|None=None):
    timeout = timeout if timeout is not None and self.can_recover else None
    st, done = time.perf_counter(), sig[0]
    while done < value:
      if done != (done:=sig[0]): st = time.perf_counter()
      elif time.perf_counter() - st > (timeout or self.wait_timeout_ms) / 1000: self.on_device_hang()

  def synchronize(self, timeout:int|None=None):
    if HCQ_RUNTIME_DEV.value != self.device: Device[HCQ_RUNTIME_DEV.value].synchronize()

    sig = self.signal("timeline")._buf.cpu_view().view(fmt='Q')
    tl = self.signal("value", 1, device="CPU")._buf.cpu_view().view(fmt='Q')
    self._wait_signal(sig, tl[0] - 1, timeout)
    if self.prof_ents: self.collect_prof()

  def on_device_hang(self): raise RuntimeError(f"{self.device} hang detected")

  def device_props(self) -> dict[str,Any]: return {} # to be overridden if needed. dict keys are backend dependent.

  def _is_cpu(self) -> bool: return hasattr(self, 'device') and self.device.split(":")[0] == "CPU"

  def finalize(self):
    try: self.synchronize() # try to finalize the device in any case
    except RuntimeError as e: print(f"{self.device} synchronization failed before finalizing: {e}")
    super().finalize()

@dataclass
class HCQ2Buffer:
  va_addr:sint
  meta:Any=None
  view:MMIOInterface|None=None

  def offset(self, offset:int, size:int) -> HCQ2Buffer:
    return HCQ2Buffer(self.va_addr+offset, meta=self.meta, view=(self.view.view(offset=offset, size=size) if self.view is not None else None))

class HCQAllocator(LRUAllocator[HCQDeviceType], Generic[HCQDeviceType]):
  def _as_buffer(self, buf:HCQBuffer) -> memoryview:
    return unwrap(buf.view).mv

  def _map(self, buf:HCQBuffer) -> HCQBuffer:
    if not hasattr(self, '_do_map'): raise NotImplementedError("map failed: no method implemented")
    return self._do_map(buf)

  def _do_unmap(self, mb): self.dev.iface.free(mb)

  @suppress_finalizing
  def _free(self, buf:HCQBuffer, options:BufferSpec|None=None):
    if options is not None and options.external_ptr is not None: return
    self.dev.synchronize()
    if hasattr(self, '_do_free'): self._do_free(buf, options)

  def _unmap(self, mb):
    self.dev.synchronize()
    self._do_unmap(mb)

  def _offset(self, buf, size:int, offset:int) -> HCQBuffer: return buf.offset(offset=offset, size=size)
