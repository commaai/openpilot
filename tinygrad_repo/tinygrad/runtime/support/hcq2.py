from __future__ import annotations
from typing import cast, Callable, TypeVar, Generic, Any, Sequence
import struct, functools, time, collections, itertools, decimal, statistics
from dataclasses import replace, dataclass
from tinygrad.helpers import DEV, getenv, select_first_inited, select_by_name, suppress_finalizing, dedup, pluralize, JIT_BATCH_SIZE, unwrap, PROFILE
from tinygrad.helpers import to_tuple, round_up, partition, data64_le, panic, ContextVar, perf_counter_us, Context
from tinygrad.device import Device, Buffer, BufferSpec, Compiled, LRUAllocator, MultiBuffer, DepsTracker
from tinygrad.device import ProfileDeviceEvent, ProfileGraphEntry, ProfileGraphEvent
from tinygrad.uop.ops import Ops, sint, UOp, UPat, PatternMatcher, KernelInfo, graph_rewrite, rewrite_group, GroupOp
from tinygrad.uop.symbolic import symbolic, pm_fold_cast_const
from tinygrad.dtype import dtypes, truncate
from tinygrad.runtime.support.hcq import MMIOInterface
from tinygrad.runtime.support.memory import BumpAllocator
from tinygrad.renderer import Renderer, Estimates
from tinygrad.engine.realize import to_program, get_call_arg_uops, get_call_name, get_call_outs_ins, estimate_uop
from tinygrad.engine.realize import pm_flatten_linear

# *****************
# 0. helpers

HCQDeviceType = TypeVar('HCQDeviceType', bound='HCQ2Compiled')

HCQ_RUNTIME_DEV = ContextVar("HCQ_RUNTIME_DEV", "CPU")

HCQ_DEVS = frozenset(("AMD",))
HCQ_P2P_DEVS = HCQ_DEVS | frozenset(("CPU",))
HCQ_CACHE_TAGS = frozenset(("program", "systems", "template"))

@dataclass(frozen=True)
class HCQInfo:
  device:tuple[str, ...]
  estimates:Estimates = Estimates()

  input_idxs:tuple[int, ...] = () # indexes into input_uops used by this call
  inputs:int|None = None
  prof:tuple[ProfileGraphEntry, ...] = () # st_id/en_id are timestamp signal slots until collect

def all_devices_in(d:Any, c:frozenset[str]) -> bool: return {x.split(":")[0] for x in to_tuple(d)} <= c

def unwrap_mstack(u):
  if u.op is Ops.MSTACK: return tuple(x for s in u.src for x in unwrap_mstack(s))
  return unwrap_mstack(u.src[0]) if u.op in {Ops.MSELECT, Ops.SLICE} else (u,)

def is_value_known_at_link(val:UOp) -> bool:
  runtime_reads = [u for u in val.toposort() if u.op in (Ops.LOAD, Ops.INDEX)]
  addressed_bufs = [b for g in val.toposort() if g.op is Ops.GETADDR for b in unwrap_mstack(g.buf_uop)]

  # addr of input params is not known at link time
  return not val.variables() and not runtime_reads and all(b.op is not Ops.PARAM or b.tag is not None for b in addressed_bufs)

def make_patches(buf:UOp, patches:Sequence[tuple[sint, UOp]]) -> tuple[UOp, ...]:
  return tuple(buf.index(UOp(Ops.STACK, dtypes.int, tuple(UOp.const(off // buf.dtype.itemsize, dtypes.int) for off,_ in ps)))
               .store(UOp(Ops.STACK, buf.dtype, tuple(val.cast(buf.dtype) for _,val in ps))).rtag(tag)
               for ps, tag in zip(partition(patches, lambda p: is_value_known_at_link(p[1])), ("link", None)) if ps)

def make_binary_patch(buf:UOp, blob:bytes) -> UOp:
  data = UOp(Ops.BINARY, src=(), arg=blob).bitcast(buf.dtype)
  r = UOp.range(len(blob) // buf.dtype.itemsize, 0, dtype=dtypes.int, src=(buf, data))
  return buf.index(r).store(data.index(r).load()).end(r).rtag("link")

def make_cmdbuf(lin, devs, buf:UOp|None=None):
  blob, patches = bytearray(), []
  for s in (s for ins in lin.src for s in ins.src):
    if s.op is not Ops.CONST: patches.append((len(blob), s))
    blob.extend(struct.pack(f'<{s.dtype.fmt}', s.val if s.op is Ops.CONST else 0x0))
  cmdbuf = buf if buf is not None else UOp.placeholder((len(blob) // 4,), dtypes.uint32, next(UOp.unique_num), device=devs).rtag("cmdbuf")
  return cmdbuf.after(make_binary_patch(cmdbuf, bytes(blob)), *make_patches(cmdbuf, patches))

def make_signal(devs, slot:int=0, tag:str="signal") -> UOp:
  return UOp.placeholder((1,), dtypes.uint64, slot, device=devs, volatile=True).rtag(tag)

def make_submit(*cmds, devs:str|tuple[str, ...], queue:str) -> UOp:
  return UOp.custom_function("submit_cmdbuf", UOp(Ops.LINEAR, src=tuple(cmds), arg=(to_tuple(devs), queue)))
def get_submit(ast:UOp) -> UOp: return next(u for u in ast.toposort() if u.op is Ops.CUSTOM_FUNCTION and u.arg == "submit_cmdbuf")

def make_call(name:str, body:UOp, info:HCQInfo) -> UOp: return UOp.custom_function("hcq", body).call(name=name, aux=info)

def encode_kernargs_clike(call:UOp, prg:UOp, devs:str|tuple[str, ...]) -> UOp:
  data, info = prg.arg
  buf = UOp.placeholder((data.kernargs_alloc_size // 4,), dtypes.uint32, next(UOp.unique_num), device=devs).rtag("kernargs")
  words = [w for gi in info.globals for w in data64_le(get_call_arg_uops(call)[gi].getaddr(devs))] + list(info.vars)
  return buf.after(*make_patches(buf, [(i * 4, w) for i, w in enumerate(words)]))

# *****************
# 0.1. prep: replace buffers with params

def replace_call_buffers(ctx:tuple[list[UOp], dict[UOp, int]], call:UOp) -> UOp|None:
  bufs, slots = ctx
  for s in call.src[1:]:
    if s.op not in (Ops.PARAM, Ops.BIND) and slots.setdefault(s, len(bufs)) == len(bufs): bufs.append(s)
  return call.replace(src=call.src[:1] + tuple(s if s.op in (Ops.PARAM, Ops.BIND) else s.param_like(slots[s]) for s in call.src[1:]))
pm_replace_buffers = PatternMatcher([(UPat(Ops.CALL, name="call"), replace_call_buffers)])

# *****************
# 1.1. prep: staging copies

def _need_staging(a, b): return all_devices_in(a.device, HCQ_DEVS) and not all_devices_in(b.device, HCQ_P2P_DEVS)

def stage_copy(dst:UOp, src:UOp) -> UOp|None:
  if not (_need_staging(src, dst) or _need_staging(dst, src)): return None

  stage = UOp.new_buffer("CPU", src.max_numel() * src.dtype.itemsize, dtypes.uint8)
  return UOp(Ops.LINEAR, src=(src.copy_to_device("CPU").call(stage, src), stage.copy_to_device(dst.device).call(dst, stage)))
pm_insert_copy_staging = PatternMatcher([(UPat(Ops.CALL, src=(UPat(Ops.COPY), UPat(name="dst"), UPat(name="src"))), stage_copy)])

# *****************
# 2. deps

class HCQDepsTracker(DepsTracker):
  @staticmethod
  def _key(buf:Any) -> tuple[Any, int, int]:
    return (buf.arg.slot, 0, buf.max_numel() * buf.dtype.itemsize) if isinstance(buf, UOp) else DepsTracker._key(buf)

def _get_call_bufs_by_lane(call:UOp, devices:tuple[str, ...]) -> list[list[Any]]:
  refs = get_call_arg_uops(call)
  return [[b if b.op is Ops.PARAM else mb.bufs[lane] if isinstance(mb:=b.buffer, MultiBuffer) else mb for b in refs] for lane in range(len(devices))]

def _get_deps(ctx:DepsTracker, bufs_by_lane:list[list[Any]], write, key:tuple[tuple[str, ...], str, int]) -> list[tuple[tuple, int, int]]:
  dep_lanes:list[tuple[tuple, int, int]] = []
  for lane, bufs in enumerate(bufs_by_lane):
    written = write if write is not None else list(range(len(bufs)))
    dep_lanes += [(dep, dlane, lane) for dep, dlane in ctx.access_resources(bufs, written, (key, lane))]
  return dep_lanes

def _build_wait_cmds(slots:dict[str, int], dep_lanes:list[tuple[tuple, int, int]], devices:tuple[str, ...], queue:str) -> tuple[list[UOp], set[int]]:
  # opt1: same-queue ops are fifo-ordered
  if devices[0].split(":")[0] in {"AMD", "QCOM"} or queue.startswith("COPY"):
    dep_lanes = [(dep, dlane, lane) for dep, dlane, lane in dep_lanes if (dep[0][dlane], dep[1]) != (devices[lane], queue)]

  # opt2: keep latest dep per (dep device, queue, cur lane)
  latest = {((dep[0][dlane], dep[1]), lane): (dep, dlane) for dep, dlane, lane in sorted(dep_lanes, key=lambda x: x[0][2])}
  deps:dict[tuple, list[int|None]] = collections.defaultdict(lambda: [None]*len(devices))
  for (_, lane), (dep, dlane) in latest.items(): deps[dep][lane] = dlane

  waits = []
  for (ddevs, dqueue, dtag), lanes in deps.items():
    sig = UOp.mstack(*[make_signal(d, tag="sentinel_signal") if dl is None else make_signal(ddevs[dl], slots[dqueue])
                       for dl, d in zip(lanes, devices)])
    waits.append(UOp(Ops.INS, arg="wait", src=(sig, UOp.const(dtag + 1, dtypes.uint64))))
  return waits, {dtag for _, _, dtag in deps}

def _build_finalizers(batch:list[tuple[UOp, tuple[str, ...]]], batch_info:list[tuple[tuple[str, ...], str]],
                      tracker:HCQDepsTracker, slots:dict[str, int]) -> tuple[list[UOp], list[UOp], set[int]]:
  # collect all buffers which belong to devices
  dev_bufs:dict[str, dict[int, Any]] = collections.defaultdict(dict)
  for call, devices in batch:
    for b in itertools.chain.from_iterable(_get_call_bufs_by_lane(call, devices)):
      for bd in to_tuple(b.device): dev_bufs[bd][id(b)] = b

  n, fences, fins, signal_tags = len(batch_info), [], [], set()
  for _, devgroup in itertools.groupby(sorted(dev_bufs), key=lambda d: d.split(":")[0]):
    devs = tuple(devgroup)

    # to finalize the batch, sync all accesses from other devices to buffers that belong to this device
    fin_deps = [dl for dl in _get_deps(tracker, [list(dev_bufs[d].values()) for d in devs], None, key=(devs, "COMPUTE:0", n)) if dl[0][2] < n]
    waits, cur_signal_tags = _build_wait_cmds(slots, fin_deps, devs, "COMPUTE:0")
    signal_tags |= cur_signal_tags

    # wait the syncs and signal the device epoch, then bump the timeline on the host
    tl_signal, tl_value = make_signal(devs, tag="timeline_signal"), make_signal(devs, tag="timeline_value")
    fin_submit = make_submit(*waits, UOp(Ops.INS, arg="store", src=(tl_signal, tl_value.index(0))), devs=devs, queue="COMPUTE:0")
    epoch = (epoch_slot:=tl_value.after(fin_submit).index(0)).load()

    # fence once per device group on this schedule's previous epoch, then reset any queue signals used by the group
    qs = dedup([qn for bdevs, qn in batch_info if set(bdevs) & set(devs)])
    sched_epoch = make_signal(devs, next(UOp.unique_num))

    wait_device_epoch = (done:=tl_signal.after(loop:=UOp.loop(0)).index(0).load()).end(loop, done < sched_epoch.index(0).load())
    resets = [make_signal(devs, slots[q]).after(wait_device_epoch).index(0).store(0) for q in qs]

    fences.append(make_call("hcq_fence", UOp.sink(*(resets or [wait_device_epoch])), HCQInfo(devs)))
    fins.append(make_call("hcq_finalizer", UOp.sink(epoch_slot.store(epoch + 1), sched_epoch.after(fin_submit).index(0).store(epoch)), HCQInfo(devs)))
  return fences, fins, signal_tags

def _finalize_batch(batch:list[tuple[UOp, tuple[str, ...]]], profile:bool) -> list[UOp]:
  batch_info = [(devices, "COMPUTE:0" if call.src[0].op is Ops.PROGRAM else "COPY:0") for call, devices in batch]

  # schedule deps
  signal_tags:set[int] = set()
  slots:dict[str, int] = collections.defaultdict(lambda: next(UOp.unique_num))
  deps_tracker = HCQDepsTracker()
  call_waits:list[list[UOp]] = []
  for tag, ((call, _), (devices, queue)) in enumerate(zip(batch, batch_info)):
    deps = _get_deps(deps_tracker, _get_call_bufs_by_lane(call, devices), get_call_outs_ins(call)[0], key=(devices, queue, tag))
    cmds, cur_signal_tags = _build_wait_cmds(slots, deps, devices, queue)
    call_waits.append(cmds)
    signal_tags |= cur_signal_tags

  # build fences and finalizers
  fences, finalizers, finalizer_signal_tags = _build_finalizers(batch, batch_info, deps_tracker, slots)
  signal_tags |= finalizer_signal_tags

  src, prof = [], []
  for tag, ((call, _), (devices, queue), q) in enumerate(zip(batch, batch_info, call_waits)):
    # first queue use, sync prior device work with the device timeline
    if batch_info.index((devices, queue)) == tag:
      epoch = make_signal(devices, tag="timeline_value").index(0) - 1
      q = [UOp(Ops.INS, arg="barrier", src=()), UOp(Ops.INS, arg="wait", src=(make_signal(devices, tag="timeline_signal"), epoch))] + q

    # and make hcq call
    name, info = get_call_name(call, get_call_arg_uops(call)), HCQInfo(devices, estimate_uop(call))
    ts_ids = [next(UOp.unique_num) for _ in range(2)] if profile else []
    prof += [ProfileGraphEntry(d, name, *ts_ids) for d in devices if ts_ids]

    ts_ins = [UOp(Ops.INS, arg="timestamp", src=(make_signal(devices, s),)) for s in ts_ids]
    q += ts_ins[:1] + [call.replace(arg=replace(call.arg, aux=info))] + ts_ins[1:]

    # signal the queue if someone waits for us
    if tag in signal_tags: q += [UOp(Ops.INS, arg="store", src=(make_signal(devices, slots[queue]), UOp.const(tag + 1, dtypes.uint64)))]
    src.append(make_call(name, make_submit(*q, devs=devices, queue=queue).sink(), info))

  # append batch timestamps to finalizers
  finalizers = [f.replace(arg=replace(f.arg, aux=replace(a:=f.arg.aux, prof=tuple(e for e in prof if e.device in a.device)))) for f in finalizers]
  return fences + src + finalizers

def sched_hcq_batches(l:UOp, profile:bool) -> UOp:
  srcs:list[UOp] = []
  batch:list[tuple[UOp, tuple[str, ...]]] = []
  for call in l.src:
    if (devs:=next((b.device for b in call.src[1:] if all_devices_in(b.device, HCQ_DEVS)), None)) is not None: batch.append((call, to_tuple(devs)))
    else: srcs, batch = srcs + _finalize_batch(batch, profile) + [call], []
  return l.replace(src=tuple(srcs + _finalize_batch(batch, profile)))

# *****************
# 3. merge into queues

def _merged_hcq_call(calls:list[UOp]) -> UOp: # TODO: simplify?
  if len(calls) == 1: return calls[0]
  devs, queue = get_submit(calls[0]).src[0].arg
  body = make_submit(*[cmd for c in calls for cmd in get_submit(c).src[0].src], devs=devs, queue=queue).sink()
  return make_call(f"submit {queue} ({len(calls)})", body,
    replace(calls[0].arg.aux, estimates=sum((c.arg.aux.estimates for c in calls), start=Estimates())))

def merge_queues(linear:UOp) -> UOp:
  new_src:list[UOp] = []
  opened_qs:dict[tuple[tuple[str, ...], str], list[UOp]] = {} # (devs, queue) -> list of hcq calls, kept in submit order
  limits:dict[tuple[tuple[str, ...], str], int] = collections.defaultdict(lambda: JIT_BATCH_SIZE.value)

  for call in linear.src:
    # non-hcq call, fence or finalizer: close all open queues
    if not isinstance(call.arg.aux, HCQInfo) or (call.arg.name or "").startswith("hcq_"):
      new_src += [_merged_hcq_call(opened_qs.pop(k)) for k in list(opened_qs)] + [call]
      continue

    devs, queue = get_submit(call).src[0].arg
    if (old:=opened_qs.pop(key:=(devs, queue), None)) is not None:
      if limits[key] and len(old) >= limits[key]: new_src, old, limits[key] = new_src + [_merged_hcq_call(old)], [], limits[key] * 2
      new_rec = old + [call]
    else:
      # no such queue opened: close every open submit on this queue that shares a device, so submit order is kept
      closing = [k for k in opened_qs if k[1] == queue and set(k[0]) & set(devs)]
      new_src += [_merged_hcq_call(opened_qs.pop(k)) for k in closing]
      new_rec = [call]
    opened_qs[(devs, queue)] = new_rec
  return linear.replace(src=tuple(new_src + [_merged_hcq_call(c) for c in opened_qs.values()]))

pm_schedule_and_merge = PatternMatcher([(UPat(Ops.LINEAR, name="l"),
  lambda ctx, l: merge_queues(sched_hcq_batches(l, ctx[1]).substitute(ctx[0], walk=True, enter_calls=True)))])

# *****************
# 4.2. hcq lowering: ops to ir

def encode_cmdbuf(submit:UOp, lin:UOp) -> UOp|None:
  if (pm:=Device.get_class(lin.arg[0][0]).pm_lower) is None: return None
  return graph_rewrite(submit, pm, name=f"encode {lin.arg[0]}", enter_calls=True)
pm_encode_cmdbufs = PatternMatcher([
  (UPat(Ops.CUSTOM_FUNCTION, arg="submit_cmdbuf", src=(UPat(Ops.LINEAR, name="lin"),), name="submit"), encode_cmdbuf)])

# *****************

def get_getaddrs(p:UOp) -> list[UOp]: return [u for u in p.toposort(gate=lambda u: u.op is not Ops.AFTER) if u.op is Ops.GETADDR]

def trim_link_patches(ctx:tuple[list[UOp], list[UOp]], a:UOp) -> UOp|None:
  links, kept = partition(a.src[1:], lambda p: p.tag == "link")
  ctx[0].extend(kept)

  # keep all patches from the link-time patches' subtrees in the C code
  afters = [u for u in UOp.sink(*links).toposort() if u.op is Ops.AFTER]
  ctx[1].extend(UOp.sink(*links).substitute({p: p.src[0] for p in afters}).src)
  return a.src[0].after(*kept, *[d for p in afters for d in p.src[1:]]) if links else None
pm_trim_link_patches = PatternMatcher([(UPat(Ops.AFTER, src=(UPat((Ops.PARAM, Ops.MSTACK)),), allow_any_len=True, name="a"), trim_link_patches)])

def make_addr_table(call:UOp, gaddrs:list[UOp], name:str) -> tuple[UOp, dict[UOp, UOp], tuple[UOp, ...], dict[UOp, int]]:
  bare = {g: g.replace(src=(g.src[0].without_after,)) for g in gaddrs}

  order = sorted(dedup(bare.values()), key=lambda g: ((b:=unwrap_mstack(g.buf_uop)[0]).arg.slot, repr(b.tag)))
  slots = {g:i for i,g in enumerate(order)}
  table = UOp.placeholder((len(order),), dtypes.uint64, next(UOp.unique_num), device=call.arg.aux.device).rtag(name)

  reads = {g: table.after(*g.src[0].src[1:] if g.src[0].op is Ops.AFTER else ()).index(UOp.const(slots[bare[g]], dtypes.int)).load() for g in gaddrs}
  fills = (table.after(*make_patches(table, [(i*table.dtype.itemsize, addr) for addr, i in slots.items()])),) if slots else ()
  return table, reads, fills, {g:slots[bare[g]] for g in gaddrs}

def is_bare_addr(val:UOp) -> bool: return val.op is Ops.CAST and val.src[0].op in (Ops.AND, Ops.SHR) and val.src[0].src[0].op is Ops.GETADDR

def make_scatter_loops(patches:list[UOp], inputs_table:tuple, lt_patches:list[UOp]) -> dict[UOp, UOp]:
  table, _, _, slots = inputs_table
  subs, by_dst = {}, collections.defaultdict(list)
  for p in patches: by_dst[p.buf_uop].append(p)
  for dst, patches in by_dst.items():
    data = []
    for p in patches:
      words = [(off, val, get_getaddrs(val)) for off,val in zip(p.src[0].src[1].src, p.src[1].src)]
      data += [(off.val, slots[gaddrs[0]]) for off,_,gaddrs in words if gaddrs][::2]
      scalars = [(off.val*dst.dtype.itemsize, val) for off,val,gaddrs in words if not gaddrs]
      subs[p] = UOp.group(*make_patches(dst, scalars)) if scalars else UOp(Ops.NOOP)

    word_table, slot_table = (UOp.placeholder((len(data),), dtypes.uint32, next(UOp.unique_num), device=dst.device).rtag("systems") for _ in range(2))
    ridx = UOp.range(len(data), next(UOp.unique_num), dtype=dtypes.int, src=(word_table, slot_table, dst))
    widx, slot = ((p.index(ridx).load() % bound).cast(dtypes.int) for p,bound in ((word_table, dst.max_numel()-1), (slot_table, table.max_numel())))
    loop = UOp.group(*[dst.index(widx+i).store((table.index(slot).load() >> 32*i).cast(dtypes.uint32)) for i in range(2)]).end(ridx)
    lt_patches += [make_binary_patch(buf, struct.pack(f'<{len(data)}I', *vals)) for buf,vals in zip((word_table, slot_table), zip(*data))]
    subs[patches[0]] = UOp.group(loop, subs[patches[0]])
  return subs

def is_input_addr(g:UOp) -> bool: return all(x.op is Ops.PARAM and x.tag is None for x in unwrap_mstack(g.buf_uop))

def split_patches(call:UOp) -> UOp|None:
  rt_patches:list[UOp] = []
  lt_patches:list[UOp] = []
  body = graph_rewrite(call.src[0], pm_trim_link_patches, ctx=(rt_patches, lt_patches), name=f"trim link-time patches ({call.arg.name})")

  # split patches
  inputs, internals = partition(dedup(g for p in rt_patches for g in get_getaddrs(p)), is_input_addr)
  runtimes, systems = partition(internals, lambda g: any(x.tag in {"program", "kernargs", "cmdbuf"} for x in unwrap_mstack(g.buf_uop)))
  tables = [make_addr_table(call, gs, n) for gs,n in ((inputs, "inputs"), (runtimes, "runtime"), (systems, "systems"))]
  reads, fills = {k:v for _,r,_,_ in tables for k,v in r.items()}, [f for t in tables[1:] for f in t[2]] # inputs table is filled by exec
  input_patches = [p for p in rt_patches if (gs:=get_getaddrs(p)) and all(map(is_input_addr, gs))
    and all(is_bare_addr(v) for v in p.src[1].src if get_getaddrs(v))]
  scatter = make_scatter_loops(input_patches, tables[0], lt_patches)
  body = body.substitute({p:p.substitute(scatter | reads) for p in rt_patches})

  if inputs: # fence inputs
    fills.append((t:=tables[0][0]).after(make_binary_patch(t, bytes(t.max_numel() * 8)))) # zeroed at link, slot 0 is the host fence
    body = body.replace(src=(UOp.sink(*body.src[0].src, t.after(*body.src[0].src).index(0).store(0)),)) # open it once consumed

  lt_srcs = collections.defaultdict(list)
  for p in lt_patches: lt_srcs[p.buf_uop].append(p)
  return call.replace(src=(body, *call.src[1:], *[b.after(*ps) for b,ps in lt_srcs.items()], *fills),
    arg=replace(call.arg, aux=replace(call.arg.aux, input_idxs=tuple(sorted(dedup(b.arg.slot for g in inputs for b in unwrap_mstack(g.buf_uop)))))))
pm_split_patches = PatternMatcher([(UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="hcq"),), name="call", allow_any_len=True), split_patches)])

# *****************

def replace_params(call:UOp) -> UOp|None:
  body, variables, param_ops = call.src[0], call.src[0].variables(), {Ops.PARAM, Ops.MSTACK}
  args = dedup([s for u in body.toposort(gate=lambda u: u.op not in param_ops) for s in u.src if s.op in param_ops and s not in variables])

  patched, refhold = partition(call.src[1:], lambda x: x.src[0] in args)
  by_root = {p.src[0]: p for p in patched}
  c_args = [by_root.get(a, a) for a in args]

  # keep buffers whose addresses become link-time constants alive and mapped
  held = args + [r.without_after for r in refhold]
  addrs = dedup([g.src[0].without_after for x in call.src for g in x.toposort() if g.op is Ops.GETADDR])
  refhold += [a for a in addrs if a not in held and all(b.op is not Ops.PARAM or b.tag is not None for b in unwrap_mstack(a))]

  sub = {(b:=u.without_after): UOp.param(i, u.dtype, shape=b.shape, device=HCQ_RUNTIME_DEV.value, volatile=b.op is Ops.PARAM and b.arg.volatile)
         for i,u in enumerate(c_args)} | {v: v.replace(arg=replace(v.arg, slot=-1)) for v in variables if v.op is Ops.PARAM}
  info = replace(call.arg.aux, inputs=next((i for i,u in enumerate(c_args) if u.without_after.tag == "inputs"), None))
  return call.replace(src=(body.substitute(sub).replace(arg="hcq_args"), *c_args, *refhold),
                      arg=replace(call.arg, aux=info)) # TODO: call.after(*refhold)?
pm_replace_params = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="hcq"),), name="call", allow_any_len=True), replace_params)])

# *****************

def resolve_getaddr_slice(bv:UOp, g:UOp) -> UOp:
  base = bv.src[0].after(*g.src[0].src[1:] if g.src[0].op is Ops.AFTER else ())
  itemsize = bv.src[0].dtype.itemsize if bv.src[0].without_after.op in (Ops.BUFFER, Ops.SLICE, Ops.MSTACK, Ops.MSELECT) else bv.dtype.itemsize
  return UOp(Ops.GETADDR, src=(base,), arg=g.arg) + UOp.const(bv.src[1].val * itemsize, dtypes.uint64)

pm_early_simplify = PatternMatcher([
  (UPat(Ops.GETADDR, src=(UPat.any(sl:=UPat(Ops.SLICE, name="bv"), sl.after(allow_any_len=True)),), name="g"), resolve_getaddr_slice),
  (UPat(Ops.INDEX, src=(UPat(Ops.SLICE, name="bv"),), allow_any_len=True, name="x"),
   lambda bv,x: x.replace(src=(bv.src[0], x.src[1] + bv.src[1].cast(x.src[1].dtype), *x.src[2:]))),
])

# *****************
# 5.3. pack placeholders buffers

def pack_hcq_placeholders(call:UOp) -> UOp|None:
  bufs = [b for b in call.src[0].toposort() if b.op is Ops.PARAM and b.tag in {"scratch", "kernargs"}]
  offs:dict[UOp, int] = {}
  sizes:dict[Any, int] = {}
  for b in bufs:
    if b.tag == "scratch": sizes[b.tag] = max(sizes.get(b.tag, 0), b.max_numel())
    else:
      offs[b] = round_up(sizes.get(b.tag, 0), 128 // b.dtype.itemsize)
      sizes[b.tag] = offs[b] + b.max_numel()
  counts = collections.Counter(b.tag for b in bufs)
  bases = {b.tag:UOp.placeholder((sizes[b.tag],), b.dtype, next(UOp.unique_num), device=b.device).rtag(b.tag) for b in bufs if counts[b.tag] > 1}
  subs = {b:UOp(Ops.SLICE, b.dtype, (bases[b.tag], UOp.const(offs.get(b, 0))), b.max_numel()) for b in bufs if b.tag in bases}
  return call.replace(src=(call.src[0].substitute(subs, walk=True), *call.src[1:])) if subs else None
pm_pack_placeholders = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="hcq"),), name="call", allow_any_len=True), pack_hcq_placeholders)])

# *****************
# 8. callify hcq programs

def callify_hcq(call:UOp, cf:UOp) -> UOp:
  prg = to_program(cf.src[0].replace(arg=KernelInfo("hcq_submit"), tag=1), Device[HCQ_RUNTIME_DEV.value].renderer)
  return call.replace(src=(cf.replace(src=(prg,), arg="hcq"), *call.src[1:]))
pm_callify_hcq = PatternMatcher([(UPat(Ops.CALL, src=(
  UPat(Ops.CUSTOM_FUNCTION, arg="hcq_args", src=(UPat(Ops.SINK),), name="cf"),), name="call", allow_any_len=True), callify_hcq)])

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

    # schedule
    linear = graph_rewrite(linear, pm_schedule_and_merge, ctx=({s:p for p,s in back_map.items()}, profile), walk=True, name="schedule and merge hcq")

    # lowering to hcq ir
    linear = graph_rewrite(linear, pm_encode_cmdbufs+pm_pack_placeholders, walk=True, name="encode and pack", enter_calls=True)

    # patches and runtime uops
    linear = graph_rewrite(linear, pm_early_simplify+symbolic+pm_fold_cast_const, bottom_up=False, name="simplify patches", enter_calls=True)
    linear = graph_rewrite(linear, pm_split_patches, walk=True, name="split patches")

    # and compile it
    linear = graph_rewrite(linear, pm_replace_params, name="replace params")
    final_linear = hcq_compile_cache[cache_key] = graph_rewrite(linear, pm_callify_hcq, name="callify hcq", enter_calls=True)

  return final_linear

# *****************
# 6. bufferize placeholders: replace placeholders with real buffers.

def bufferize_buf(ctx:bool, buf:UOp) -> UOp|None:
  if buf.tag is None: return None
  return UOp.mstack(*(UOp.from_buffer((dv:=Device[dev]).pm_bufferize.rewrite(buf, ctx=(dv, ctx)), HCQ_RUNTIME_DEV.value)
                      for dev in to_tuple(buf.device)))
pm_bufferize = PatternMatcher([(UPat(Ops.PARAM, name="buf"), bufferize_buf)])

# *****************
# 7. resolve patches

def push_stack(op, s): return UOp(Ops.STACK,
  src=tuple(op.replace(dtype=op.dtype.scalar(), src=tuple(x if y is s else y for y in op.src)) for x in s.src))

def fold_binary(buf:UOp, blob:UOp) -> UOp:
  for b in (m.bufs if isinstance(m:=buf.buffer, MultiBuffer) else (m,)):
    b.ensure_allocated().as_memoryview(force_zero_copy=True, no_sync=True).cast('B')[:len(blob.arg)] = blob.arg
  return UOp(Ops.NOOP)

def fold_const_store(buf:UOp, off:UOp, val:UOp) -> UOp:
  for off,val in zip(off.src, val.src):
    for b,v in zip((bs:=mb.bufs if isinstance((mb:=buf.buffer), MultiBuffer) else (mb,)), val.src if val.op is Ops.STACK else (val,)*len(bs)):
      data = struct.pack(f'<{v.dtype.fmt}', truncate[v.dtype](v.val))
      b.ensure_allocated().as_memoryview(force_zero_copy=True, no_sync=True).cast('B')[(bo:=off.val*buf.dtype.itemsize):bo+len(data)] = data
  return UOp(Ops.NOOP)

def resolve_getaddr(buf:UOp, g:UOp) -> UOp:
  assert buf.op in (Ops.BUFFER, Ops.MSTACK, Ops.MSELECT), f"{buf.op}"

  devs, b = g.arg, buf.buffer
  bufs = tuple(cast(Buffer, x.buffer) for x in buf.src) if buf.op is Ops.MSTACK else tuple(b.bufs if isinstance(b, MultiBuffer) else (b,)*len(devs))
  assert len(bufs) == len(devs), f"can't resolve {len(bufs)} buffers on {len(devs)} devices"
  addrs = tuple(UOp.const(x.get_buf(d).va_addr, dtypes.uint64) for x, d in zip(bufs, devs))
  return addrs[0] if len(addrs) == 1 else UOp(Ops.STACK, src=addrs)

pm_resolve_patches = PatternMatcher([
  # multi
  (UPat(GroupOp.ALU, src=[UPat(Ops.STACK, name="s"), UPat(Ops.CONST)], name="op"), push_stack),
  (UPat(Ops.CAST, src=(UPat(Ops.STACK, name="s"),), name="op"), push_stack),

  # getaddr
  (UPat(Ops.GETADDR, src=(UPat(name="buf"),), name="g"), resolve_getaddr),

  # folders
  (UPat(name="buf").index(UPat(Ops.RANGE), allow_any_len=True)
    .store(UPat.any(UPat(Ops.BINARY, name="blob"), UPat(Ops.BINARY, name="blob").bitcast()).index(UPat(Ops.RANGE), allow_any_len=True).load())
    .end(UPat(Ops.RANGE)), fold_binary),
  (UPat({Ops.BUFFER, Ops.SLICE, Ops.MSTACK}, name="buf").index(UPat(Ops.STACK, name="off")).store(UPat(Ops.STACK, name="val")), fold_const_store),
])

pm_assert_no_afters = PatternMatcher([(UPat(Ops.AFTER, name="a"), lambda a: panic(RuntimeError, f"AFTER left at hcq_link: {a.src[0].op}"))])

def link_buf_key(a:UOp): return a.key, to_tuple(a.device)
link_buf_cache:dict[tuple[bytes, tuple[str, ...]], UOp] = {}
link_linear_cache:dict[bytes, UOp] = {}

@rewrite_group(lambda _,cache,ret: f"HCQ Link {pluralize('Kernel', len(ret.src))}")
def hcq_link(linear:UOp, cache=True) -> UOp:
  if (linked:=link_linear_cache.get(linear_key:=linear.key)) is not None: return linked

  bufs = {(j,i):a for j,c in enumerate(linear.src) for i,a in enumerate(c.src[1:], 1)
          if a.op is Ops.AFTER and unwrap_mstack(a.src[0])[0].tag in HCQ_CACHE_TAGS}
  linear = linear.substitute({x:link_buf_cache[k] for a in bufs.values() if (k:=link_buf_key(a)) in link_buf_cache for x in (a, a.src[0])}, walk=True)
  linear = graph_rewrite(linear, pm_resolve_patches+symbolic+pm_fold_cast_const+pm_assert_no_afters, bpm=pm_bufferize, ctx=cache, bottom_up=False,
                         name="resolve patches")
  for (j,i),a in bufs.items(): link_buf_cache.setdefault(link_buf_key(a), linear.src[j].src[i])
  if cache: link_linear_cache[linear_key] = linear
  return linear

# *****************
# Device classes

class HCQ2Compiled(Compiled):
  timestamp_divider: float = 1000.0

  def __init__(self, device:str, allocator:HCQAllocator, compilers:list[type[Renderer]], runtime, can_recover:bool=False, arch=None):
    self.device_id:int = int(device.split(":")[1]) if ":" in device else 0
    self.can_recover = can_recover

    self.pm_bufferize = PatternMatcher([
      (UPat(Ops.PARAM, tag="sentinel_signal"), lambda ctx: ctx[0].signal("sentinel", (1 << 64) - 1)),
      (UPat(Ops.PARAM, tag="timeline_signal"), lambda ctx: ctx[0].signal("timeline")),
      (UPat(Ops.PARAM, tag="timeline_value"), lambda ctx: ctx[0].signal("value", 1)),
      (UPat(Ops.PARAM, tag="signal", name="b"), lambda ctx, b: ctx[0].signal(b.arg.slot)),
      (UPat(Ops.PARAM, name="b"), lambda ctx, b: None if b.tag is None else ctx[0].new_buffer(b, cache=ctx[1]))
    ])

    super().__init__(device, allocator, compilers, runtime, None, arch=arch)

    self.rt_buffer = Buffer(self.device, 64 << 20, dtypes.uint8, options=BufferSpec(uncached=True, cpu_access=True))
    self.rt_allocator = BumpAllocator(64 << 20)
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

  def new_buffer(self, b:UOp, cache:bool) -> Buffer:
    if cache or b.tag in HCQ_CACHE_TAGS:
      return Buffer(self.device, b.max_numel(), b.dtype, options=BufferSpec(uncached=True, cpu_access=True, nolru=True))
    return self.rt_buffer.view(b.max_numel(), b.dtype, self.rt_allocator.alloc(b.max_numel() * b.dtype.itemsize, alignment=128))

  @functools.cache
  def signal(self, name:str|int, init_value:int=0) -> Buffer:
    buf = Buffer(self.device, 1, dtypes.uint64, options=BufferSpec(host=True, uncached=True, cpu_access=True), preallocate=True)
    buf.as_memoryview(force_zero_copy=True, no_sync=True).cast('Q')[0] = init_value
    return buf

  def synchronize(self, timeout:int|None=None):
    if HCQ_RUNTIME_DEV.value != self.device: Device[HCQ_RUNTIME_DEV.value].synchronize()

    sig = self.signal("timeline").as_memoryview(force_zero_copy=True, no_sync=True).cast('Q')
    tl = self.signal("value", 1).as_memoryview(force_zero_copy=True, no_sync=True).cast('Q')
    timeout = timeout if timeout is not None and self.can_recover else None
    st = time.perf_counter()
    while sig[0] < tl[0] - 1:
      if time.perf_counter() - st > (timeout or 3000) / 1000: self.on_device_hang()
    if self.prof_ents: self.collect_prof()

  def on_device_hang(self): raise RuntimeError(f"{self.device} hang detected")

  def device_props(self) -> dict[str,Any]: return {} # to be overridden if needed. dict keys are backend dependent.

  def count(self) -> int: return self.iface.count if hasattr(self, 'iface') else 1

  def _select_iface(self):
    assert (v:=getenv(k:=f'{type(self).__name__[:-6].upper()}_IFACE', "")) == "",  \
      f"{k}={v} is deprecated, use DEV={replace(DEV.target(type(self).__name__[:-6]), interface=v)} instead"
    assert hasattr(self, "ifaces"), "must have ifaces to select an iface"
    t = DEV.target(dev:=type(self).__name__[:-6])
    filtered = select_by_name(self.ifaces, lambda i: i.__name__[:-5], t.interface, f"{dev} has no interface {t.interface!r}")
    filtered = [i for i in filtered if t.interface.startswith("MOCK") or not i.__name__[:-5].startswith("MOCK")] # never fall back to mock ifaces
    return select_first_inited([functools.partial(cast(Callable, iface), self, self.device_id) for iface in filtered],
                               f"No interface for {dev}:{self.device_id} is available")

  def _is_cpu(self) -> bool: return hasattr(self, 'device') and self.device.split(":")[0] == "CPU"

  def finalize(self):
    try: self.synchronize() # try to finalize the device in any case
    except RuntimeError as e: print(f"{self.device} synchronization failed before finalizing: {e}")

    # if the device has an interface, call device_fini to clean up resources
    if hasattr(self, 'iface') and hasattr(self.iface, 'device_fini'): self.iface.device_fini()

@dataclass
class HCQ2Buffer:
  va_addr:sint
  meta:Any=None
  view:MMIOInterface|None=None

  def offset(self, offset:int, size:int) -> HCQ2Buffer:
    return HCQ2Buffer(self.va_addr+offset, meta=self.meta, view=(self.view.view(offset=offset, size=size) if self.view is not None else None))

class HCQAllocator(LRUAllocator[HCQDeviceType], Generic[HCQDeviceType]):
  def _as_buffer(self, buf:HCQ2Buffer) -> memoryview:
    return unwrap(buf.view).mv

  def _map(self, buf:HCQ2Buffer) -> HCQ2Buffer:
    if not hasattr(self, '_do_map'): raise NotImplementedError("map failed: no method implemented")
    return self._do_map(buf)

  @suppress_finalizing
  def _free(self, buf:HCQ2Buffer, options:BufferSpec|None=None):
    if options is not None and options.external_ptr is not None: return
    self.dev.synchronize()
    if hasattr(self, '_do_free'): self._do_free(buf, options)

  def _unmap(self, mb):
    self.dev.synchronize()
    self.dev.iface.free(mb)

  def _offset(self, buf, size:int, offset:int) -> HCQ2Buffer: return buf.offset(offset=offset, size=size)
