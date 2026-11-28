from dataclasses import dataclass, field
from typing import cast
import itertools
from tinygrad.dtype import dtypes, AddrSpace, Invalid
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, resolve, GroupOp, KernelInfo, graph_rewrite, sint, AxisType, BottomUpGate, rewrite_group
from tinygrad.uop.ops import StageOpts
from tinygrad.uop.symbolic import symbolic
from tinygrad.helpers import prod, dedup, VIZ, MAX_KERNEL_BUFFERS, SPEC, get_single_element
from tinygrad.codegen.simplify import pm_flatten_range, pm_reduce_simplify
from tinygrad.codegen.indexing import run_rangeify, apply_movement_op
from tinygrad.codegen.prepare import pm_mops

# creation can recurse a lot
import sys
sys.setrecursionlimit(10000)

# *****************
# 3.5 cleanups

# you don't know in the first pass if axes are going to die, this happens if there's an EXPAND to the left
def cleanup_dead_axes(b:UOp):
  if not b.arg.removable: return None
  # don't optimize AFTER (AFTER is a buffer identity — ranges define consumer access, not computation)
  if b.src[0].op is Ops.AFTER: return None

  new_rng = []
  reshape: list[sint] = []
  for s,rng in zip(b.shape, b.src[1:]):
    # skip for symbolic. TODO: fix this
    if rng.op is Ops.RANGE and rng.src[0].op is not Ops.CONST: return None
    # CONSTs are already dead axes
    if rng.op is Ops.CONST or (rng.op is Ops.RANGE and rng not in b.src[0].ranges):
      reshape.append(1)
    else:
      reshape.append(s)
      new_rng.append(rng)
  if len(new_rng) != len(reshape):
    return b.replace(src=b.src[0:1]+tuple(new_rng)).reshape(tuple(reshape)).expand(b.shape)

def gate_substitute(ctx, b:UOp) -> None:
  if not any(r in b.ranges for r in ctx.keys()): raise BottomUpGate()
pm_gate_substitute = PatternMatcher([(UPat(GroupOp.All, name="b"), gate_substitute)], compiled=False)
# if a buffer is being stored just for permutes or something, remove it
# we want to reexpress the indexes of idx2 in terms of the implied b1
def remove_stage(src:UOp, buf:UOp, idx:UOp):
  # see if we can't do it, should this ever hit?
  assert len(buf.src) == len(idx.src), f"index on wrong STAGE, {len(buf.src)} != {len(idx.src)}"
  assert all(x.op in {Ops.RANGE, Ops.CONST} for x in buf.src[1:])

  # if it's user contiguous, we never remove it
  if not buf.arg.removable: return None

  # *** here is where we compute the cost ***
  # if we return None, the STAGE is kept

  accessed_buffers: list[UOp] = []
  reduces: list[UOp] = []
  def red_gate(x:UOp):
    if x.op is Ops.AFTER:
      accessed_buffers.append(x.buf_uop)
      return False
    if (x.op is Ops.STAGE and x.arg.addrspace == AddrSpace.GLOBAL) or x.op is Ops.MSTACK:
      accessed_buffers.append(x)
      return False
    if x.op is Ops.STORE:
      # don't look inside stores, this doesn't count toward buffer accesses
      return False
    if x.op in {Ops.PARAM, Ops.EXTERN}:
      accessed_buffers.append(x)
    if x.op is Ops.REDUCE: reduces.append(x)
    return True
  src.toposort(gate=red_gate)
  del red_gate
  accessed_buffers = dedup(accessed_buffers)

  # if this is generated from multiple buffers, don't remove this buffer
  if len(accessed_buffers) > 3: return None

  # if any reduces access a buffer, don't remove this buffer
  buffer_in_reduce = False
  def buf_gate(x:UOp):
    nonlocal buffer_in_reduce
    if x.op in {Ops.PARAM, Ops.EXTERN, Ops.STAGE, Ops.AFTER}: buffer_in_reduce = True
    return not buffer_in_reduce
  UOp.sink(*[x.src[0] for x in reduces]).toposort(gate=buf_gate)
  del buf_gate
  if buffer_in_reduce:
    return None

  # if it makes it here, the STAGE is removed
  # this is the ranges replaced
  # NOTE: if buf src is a const, we don't replace it. if idx is Invalid (dead load), don't replace it either
  replaced = {k:v for k,v in zip(buf.src[1:], idx.src[1:]) if k.op is not Ops.CONST and not (v.op is Ops.CONST and v.val is Invalid)}
  return src.substitute(replaced, extra_pm=pm_gate_substitute)

def remove_noop_stage(idx,b2):
  if idx.src[1:] != b2.src[1:]: return None
  return idx.src[0].shrink(tuple((0, s) for s in b2.shape))

def after_all_invalid(after:UOp):
  buf = after.src[0].buf_uop
  # check all ranges are used (no expand), and same size (no pad and shrink)
  return all(s.op is Ops.END and (st:=s.src[0]).op is Ops.STORE and st.src[1].base.is_invalid and st.src[0].buf_uop is buf
    and all(r in st.src[0].ranges for r in s.ended_ranges)
    and resolve(cast(UOp, prod(r.src[0] for r in s.ended_ranges)).eq(buf.numel()), False) for s in after.src[1:])

pm_const_buffer_folding = pm_mops+PatternMatcher([
  (UPat(Ops.STAGE, name="b"), cleanup_dead_axes),
  # remove noop buffers. if we look at the next index we can remove even more of these
  (UPat(Ops.INDEX, name="idx").f(Ops.STAGE, allow_any_len=True, name="b2"), remove_noop_stage),
  # no buffers for a const, in either spelling
  (UPat.cvar('c').or_casted().f(Ops.STAGE, allow_any_len=True, name="b"), lambda c,b: b.const_like(c.val)),
  # indexing a const is the const
  (UPat(Ops.INDEX, src=(UPat.cvar().or_casted("c"),),), lambda c: c),
  # indexing an after with all fully invalid stores is invalid
  (UPat(Ops.INDEX, src=(UPat(Ops.AFTER, name="after"),), allow_any_len=True, name="idx"),
   lambda idx,after: idx.const_like(Invalid) if after_all_invalid(after) else None),
])

pm_remove_stage = PatternMatcher([
  # remove reindexing with cost function
  (UPat.var("src").f(Ops.STAGE, allow_any_len=True, name="buf").f(Ops.INDEX, allow_any_len=True, name="idx"), remove_stage),
  # STORE to self is NOOP
  (UPat.var("x").store(UPat.var("x")), lambda x: UOp(Ops.NOOP)),
])

def strip_zero_offset_shrink(x:UOp) -> UOp:
  return x.src[0] if x.op is Ops.SHRINK and all(resolve(start == 0, False) for start,_ in x.marg) else x

def no_indexing_calls(u:UOp):
  def strip(x:UOp) -> UOp:
    # sometimes if call srcs have children the call will get an INDEX. we remove it here.
    # TODO: we should add safety checks here for contiguous
    if x.op is Ops.INDEX: return x.src[0]
    # SHRINK with offset 0 is fine, also inside an MSTACK
    return x.replace(src=tuple(map(strip_zero_offset_shrink, x.src))) if x.op is Ops.MSTACK else strip_zero_offset_shrink(x)
  return u.replace(src=tuple(map(strip, u.src)))

pm_no_indexing_calls = PatternMatcher([
  (UPat(Ops.CALL, name="u"), no_indexing_calls),
])

# the kernel graph is what gets executed: no shape views left in it, the storage of a value is just the storage
pm_no_views = PatternMatcher([
  (UPat((Ops.RESHAPE, Ops.SHRINK), name="v",
        src=(UPat(GroupOp.Defines|{Ops.AFTER, Ops.UNSHARD, Ops.MSTACK}),), allow_any_len=True), lambda v: v.src[0]),
])

DEVICE_MAX_BUFS = {"WEBGPU": 8} # TODO: get from device?
@dataclass
class LimitBufsContext:
  buf_cache: dict[UOp, frozenset[UOp]] = field(default_factory=dict)
  range_idx: itertools.count = field(default_factory=itertools.count)

def insert_stage(ctx:LimitBufsContext, s:UOp) -> UOp:
  # Insert STAGE: use fresh WEAK ranges, while the DEVICE range stays a launched axis
  end_ranges = [x.replace(arg=(AxisType.WEAK, next(ctx.range_idx))) if x.axis_type is not AxisType.DEVICE else x for x in s.ranges]
  return s.substitute(dict(zip(s.ranges, end_ranges))).stage(*end_ranges, arg=StageOpts(device=s.device)).index(*s.ranges)

def _limit_bufs(ctx:LimitBufsContext, root:UOp):
  if (device:=root.device) is None: return None # no device, index related calculations
  device = device if isinstance(device, str) else device[0].split(":")[0]
  if not (MAX_BUFS:=MAX_KERNEL_BUFFERS.value or DEVICE_MAX_BUFS.get(device, 0)): return None

  def visitor(u:UOp) -> frozenset[UOp]:
    if u.op in {Ops.STAGE, Ops.AFTER, Ops.PARAM, Ops.EXTERN, Ops.MSELECT, Ops.MSTACK}: return frozenset((u,))
    if len(u.src) == 1: return ctx.buf_cache[u.src[0]]
    return frozenset().union(*[ctx.buf_cache[s] for s in u.src])
  bufs = root.topovisit(visitor, ctx.buf_cache)

  if len(bufs) > MAX_BUFS - 1: # NOTE: this -1 is for the output buffer
    return root.replace(src=tuple(insert_stage(ctx, s) if s.op in GroupOp.Elementwise and s.device is not None else s for s in root.src))

def fix_store_hazard(ctx:LimitBufsContext, dst:UOp, src:UOp):
  loads = [u for u in src.toposort(gate=lambda u: u.op not in {Ops.STAGE, Ops.AFTER}) if u.op is Ops.INDEX and u.buf_uop is dst.buf_uop]
  if any(u.src[1].get_idx() is not dst.src[1].get_idx() for u in loads): return dst.store(insert_stage(ctx, src))

pm_limit_bufs = PatternMatcher([(UPat(set.union(GroupOp.Binary, GroupOp.Ternary), name="root"), _limit_bufs),
  (UPat(Ops.INDEX, name="dst").store(UPat.var("src")), fix_store_hazard)])

# *****************
# 4. put in buffers for STAGE
# TODO: should STAGE look a lot more like STORE
# STAGE has device in arg
# STAGE doesn't have indexing, that's implied by the ranges it closes
# STAGE returns the BUFFER ready for INDEXing (doing this will make splitting a lot easier)
# NOTE: this has been fixed up a bit

def stage_to_store(ctx:itertools.count, x:UOp, idx:UOp):
  size = prod(x.shape)
  dtype = x.commit_dtype()  # a BUFFER is never weak: store at the committed dtype, the .cast(x.dtype) on the result keeps readers unchanged
  # the store ends the loops it is in, the DEVICE range is launched per device
  rngs = sorted([r for r in idx.ranges if r.axis_type is not AxisType.DEVICE], key=lambda x: x.arg)
  assert size > 0 and isinstance(size, int), f"no zero sized or symbolic sized buffers {size}"

  # AFTER: add END to the existing STORE, return buffer with kernel dependency
  if (after:=x.src[0]).op is Ops.AFTER:
    buf = after.src[0].buf_uop.base
    if not (stores := [s for s in after.src[1:] if s.op is Ops.STORE and s.src[0].op is Ops.INDEX]): return buf
    # STAGE(INDEX(...)); store through the underlying global index instead.
    ended_stores = []
    for store in stores:
      store_target = store.src[0]
      if store_target.src[0].op is Ops.STAGE and store_target.src[0].src[0].op is Ops.INDEX:
        store_target = store_target.src[0].src[0]
      if store.src[1] is store_target: continue  # skip self-assign
      end_rngs = sorted(dedup([r for r in store_target.ranges if r.axis_type is not AxisType.DEVICE] + rngs), key=lambda x: x.arg)
      ended_stores.append(store_target.store(store.src[1]).end(*end_rngs))
    return buf.after(*ended_stores)

  if x.arg.addrspace == AddrSpace.GLOBAL:
    buf = UOp.alloc((size,), dtype, slot=next(ctx), device=x.arg.device)
    do_store = buf.index(idx).store(x.src[0].cast(dtype)).end(*rngs)
    return buf.after(do_store).cast(x.dtype)

# collapse any STAGE to single input STAGE
def flatten_stage(x:UOp):
  if len(x.src) == 2: return None
  ret = x.replace(src=(x.src[0], get_single_element(apply_movement_op(Ops.RESHAPE, (prod(x.shape),), x.shape, x.src[1:]))))
  return ret.reshape(x.shape).view_as(tuple(r.src[0] if r.op is not Ops.CONST else 1 for r in x.src[1:]))
pm_flatten_stage = PatternMatcher([(UPat(Ops.STAGE, name="x"), flatten_stage)])

def is_noop_after_dep(x:UOp) -> bool:
  return (x.op is Ops.NOOP and len(x.src) == 0) or (x.op in (Ops.END, Ops.AFTER) and is_noop_after_dep(x.src[0]) and
    all(s.op is Ops.RANGE for s in x.src[1:]))

def remove_noop_afters(x:UOp) -> UOp|None:
  src = (x.src[0],) + tuple(s for s in x.src[1:] if not is_noop_after_dep(s))
  if len(src) != len(x.src): return src[0] if len(src) == 1 else x.replace(src=src)
  return None

pm_add_buffers = pm_mops+pm_flatten_stage+PatternMatcher([
  (UPat(Ops.STAGE, src=(UPat(), UPat(name="idx")), name="x"), stage_to_store),

  # INDEX of a buffer through the weak cast added above: index the buffer directly and cast the loaded value instead.
  # this must run in the same rewrite that adds the cast, or the expander expands the whole casted buffer into one big VECTORIZE
  (UPat(Ops.INDEX, src=(UPat(Ops.CAST, dtype=dtypes.weaks, src=(UPat.var("buf"),)),), allow_any_len=True, name="u"),
   lambda u,buf: u.replace(src=(buf,)+u.src[1:]).cast(u.dtype)),

  # move RESHAPEs through MSELECT/MSTACK
  (UPat((Ops.MSELECT, Ops.MSTACK), src=UPat(Ops.RESHAPE), name="m"),
   lambda m: m.replace(src=tuple([x.src[0].base for x in m.src])).reshape(m.shape)),

  # remove invalid writes
  (UPat(Ops.STORE, src=(UPat(), UPat(Ops.CONST, arg=Invalid))), lambda: UOp(Ops.NOOP)),
  (UPat(Ops.AFTER, name="x"), remove_noop_afters),
])

# *****************
# 5. split into kernels

@dataclass
class LocalAddBufferContext:
  map:dict = field(default_factory=dict)
  ranges:dict = field(default_factory=dict)

def add_buf(ctx:LocalAddBufferContext, buf:UOp):
  if buf not in ctx.map: ctx.map[buf] = buf
  raise BottomUpGate

def debuf(slot:int, buf:UOp):
  param = UOp.param(slot, buf.dtype, buf.max_numel(), addrspace=buf.addrspace, device=buf.device)
  return param.reshape(buf.max_shape).view_as(buf.shape)

def handle_after(ctx:LocalAddBufferContext, after:UOp):
  if after.addrspace == AddrSpace.LOCAL: return None
  if after.addrspace is AddrSpace.ALU: raise RuntimeError("kernel boundary requires storage")
  buf = after.buf_uop
  # NOTE: this is bottom up, so we only add it once
  if buf not in ctx.map: ctx.map[buf] = after
  return buf

def check_buf_states(x:UOp):
  idxs = [s for s in x.toposort(gate=lambda x: x.op is not Ops.AFTER) if s.op is Ops.INDEX]
  read_from: dict[UOp, UOp] = {}
  if any((buf:=idx.buf_uop).op in GroupOp.Defines and read_from.setdefault(buf, state:=idx.src[0]) is not state for idx in idxs):
    raise RuntimeError(f"cycle detected while indexing {buf}")

to_define_global = PatternMatcher([
  (UPat(Ops.STORE, name="x"), check_buf_states),
  (UPat((Ops.BUFFER, Ops.ALLOC, Ops.MSTACK, Ops.MSELECT), name="buf"), add_buf),
  # storage params get kernel-local slots here, the scalars are lifted once it's a call
  (UPat((Ops.PARAM, Ops.EXTERN), name="buf"), lambda ctx, buf: None if buf.addrspace is AddrSpace.ALU else add_buf(ctx, buf)),

  # ALU params are scalar symbolic values, not buffers.
  (UPat(Ops.INDEX, src=(UPat((Ops.PARAM, Ops.EXTERN), name="v"),)), lambda v: v if v.addrspace == AddrSpace.ALU else None),

  (UPat(Ops.AFTER, name="after"), handle_after),

  (UPat(Ops.RANGE, name="r"), lambda ctx, r: ctx.ranges.setdefault(r.arg)),
])

# renumber the ranges starting with 0 so that kernel deduping works
pm_renumber_range = PatternMatcher([(UPat(Ops.RANGE, name="r"), lambda ctx, r: r.replace(arg=(r.axis_type, ctx[r.arg])+r.axis_id[1:]))])

def split_store(x:UOp) -> UOp|None:
  # if we have any open ranges here, we don't split. open DEVICE ranges are fine, they are bound per device at launch
  if any(r.axis_type is not AxisType.DEVICE for r in x.ranges): return None

  # local kernel rewrite
  lctx = LocalAddBufferContext()
  ret = graph_rewrite(x, to_define_global+pm_flatten_range, ctx=lctx, name="kernel split", bottom_up=True)
  ret = ret.substitute({b:debuf(i, b) for i,b in enumerate(lctx.map)}, walk=True)
  ret = graph_rewrite(ret, pm_renumber_range, ctx={a:i for i,a in enumerate(lctx.ranges)}, walk=True)

  # create the Kernel. NOTE: buffers can be on different devices here now, they are compiled to SDMA copies later by schedule
  kernel = ret.sink(arg=KernelInfo()).call(*lctx.map.values())
  # the scalars it reads (also in the symbolic shapes of its storage) and its device number are args after the storage
  return kernel.lift_args([u for u in kernel.body.toposort() if (u.op in {Ops.PARAM, Ops.EXTERN} and u.addrspace is AddrSpace.ALU) or
                           (u.op is Ops.RANGE and u.axis_type is AxisType.DEVICE)])

split_kernels = PatternMatcher([
  (UPat((Ops.STORE, Ops.END), name="x"), split_store),
])

@rewrite_group(new_ctx=False)
def get_kernel_graph(tsink:UOp) -> UOp:
  # convert movement ops to ranges
  tsink = run_rangeify(tsink)

  # cleanups for speed and runability
  tsink = graph_rewrite(tsink,
                        symbolic+pm_reduce_simplify+pm_const_buffer_folding+pm_remove_stage,
                        name="symbolic+reduce_collapse+remove_stage")
  next_range = max((x.axis_id[0] for x in tsink.toposort() if x.op is Ops.RANGE), default=-1) + 1
  tsink = graph_rewrite(tsink, pm_limit_bufs, ctx=LimitBufsContext(range_idx=itertools.count(next_range)), name="limit buffers")
  if VIZ: graph_rewrite(tsink, PatternMatcher([]), name="View Rangeify")

  # STAGE -> STORE
  slots = [x.arg.slot for x in tsink.toposort() if x.op is Ops.ALLOC]
  paramarg_start: int = max([-1]+slots) + 1
  tsink = graph_rewrite(tsink, pm_add_buffers, ctx=itertools.count(paramarg_start), bottom_up=True, name="stage to store")
  tsink = graph_rewrite(tsink, pm_no_indexing_calls, name="remove indexing from call args")
  tsink = graph_rewrite(tsink, split_kernels, bottom_up=True, name="split kernels")
  tsink = graph_rewrite(tsink, pm_no_views, name="remove views from the kernel graph")

  if VIZ: graph_rewrite(tsink, PatternMatcher([]), name="View Kernel Graph")
  if SPEC:
    # validate the kernel graph
    from tinygrad.uop.spec import type_verify, spec_kernel_graph
    type_verify(tsink, spec_kernel_graph)
  return tsink
