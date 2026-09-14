from dataclasses import dataclass, field, replace
from typing import cast
import itertools
from tinygrad.dtype import dtypes, AddrSpace, Invalid
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, resolve, GroupOp, KernelInfo, ParamArg
from tinygrad.uop.ops import graph_rewrite, sint, AxisType, BottomUpGate, rewrite_group
from tinygrad.uop.symbolic import symbolic
from tinygrad.helpers import prod, dedup, DEBUG_RANGEIFY, VIZ, MAX_KERNEL_BUFFERS, SPEC
from tinygrad.helpers import get_single_element
from tinygrad.codegen.simplify import pm_flatten_range, pm_reduce_simplify
from tinygrad.codegen.opt import Opt
from tinygrad.schedule.indexing import run_rangeify, BufferizeOpts, apply_movement_op
from tinygrad.schedule.prepare import pm_mops

# creation can recurse a lot
import sys
sys.setrecursionlimit(10000)

# *****************
# 3.5 cleanups

ALWAYS_RUN_OPS = {Ops.CONTIGUOUS, Ops.NOOP}

# you don't know in the first pass if axes are going to die, this happens if there's an EXPAND to the left
def cleanup_dead_axes(b:UOp):
  if not b.arg.removable: return None
  # don't optimize ALWAYS_RUN_OPS or AFTER (AFTER is a buffer identity — ranges define consumer access, not computation)
  if b.src[0].op in ALWAYS_RUN_OPS or b.src[0].op is Ops.AFTER: return None

  new_rng = []
  hit = False
  reshape: list[sint] = []
  for s,rng in zip(b.shape, b.src[1:]):
    # skip for symbolic. TODO: fix this
    if rng.op is Ops.RANGE and rng.src[0].op is not Ops.CONST: return None
    # CONSTs are already dead axes
    if rng.op is Ops.CONST or (rng.op is Ops.RANGE and rng not in b.src[0].ranges):
      reshape.append(1)
      hit = True
    else:
      reshape.append(s)
      new_rng.append(rng)
  if hit:
    return b.replace(src=b.src[0:1]+tuple(new_rng)).reshape(tuple(reshape)).expand(b.shape)

def gate_substitute(ctx, b:UOp) -> None:
  if not any(r in b.ranges for r in ctx.keys()): raise BottomUpGate()
pm_gate_substitute = PatternMatcher([(UPat(GroupOp.All, name="b"), gate_substitute)], compiled=False)
# if a buffer is being stored just for permutes or something, remove it
# we want to reexpress the indexes of idx2 in terms of the implied b1
def remove_bufferize(src:UOp, buf:UOp, idx:UOp):
  # see if we can't do it, should this ever hit?
  assert len(buf.src) == len(idx.src), f"index on wrong bufferize, {len(buf.src)} != {len(idx.src)}"
  assert all(x.op in {Ops.RANGE, Ops.CONST} for x in buf.src[1:])

  # if it's user contiguous, we never remove it
  if src.op in ALWAYS_RUN_OPS or not buf.arg.removable: return None

  # *** here is where we compute the cost ***
  # if we return None, the bufferize is kept

  accessed_buffers: list[UOp] = []
  indexes: list[UOp] = []
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
    if x.op is Ops.PARAM:
      accessed_buffers.append(x)
    if x.op is Ops.INDEX:
      indexes.append(x)
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
    if x.op in {Ops.PARAM, Ops.STAGE, Ops.AFTER}: buffer_in_reduce = True
    return not buffer_in_reduce
  UOp.sink(*[x.src[0] for x in reduces]).toposort(gate=buf_gate)
  del buf_gate
  if buffer_in_reduce:
    return None

  # if it makes it here, the bufferize is removed
  # this is the ranges replaced
  # NOTE: if buf src is a const, we don't replace it. if idx is Invalid (dead load), don't replace it either
  replaced = {k:v for k,v in zip(buf.src[1:], idx.src[1:]) if k.op is not Ops.CONST and not (v.op is Ops.CONST and v.val is Invalid)}
  return src.substitute(replaced, extra_pm=pm_gate_substitute)

def remove_noop_bufferize(idx,b2):
  if idx.src[1:] != b2.src[1:]: return None
  return idx.src[0].shrink(tuple((0, s) for s in b2.shape)) if b2.shape else idx.src[0]

def after_all_invalid(after:UOp):
  buf = after.src[0].buf_uop
  # check all ranges are used (no expand), and same size (no pad and shrink)
  return all(s.op is Ops.END and (st:=s.src[0]).op is Ops.STORE and st.src[1].base.is_invalid and st.src[0].buf_uop is buf
    and all(r in st.src[0].ranges for r in s.ended_ranges)
    and resolve(cast(UOp, prod(r.src[0] for r in s.ended_ranges)).eq(buf.numel()), False) for s in after.src[1:])

pm_const_buffer_folding = pm_mops+PatternMatcher([
  (UPat(Ops.STAGE, name="b"), cleanup_dead_axes),
  # remove noop buffers. if we look at the next index we can remove even more of these
  (UPat(Ops.INDEX, name="idx").f(Ops.STAGE, allow_any_len=True, name="b2"), remove_noop_bufferize),
  # no buffers for a const, in either spelling
  (UPat.cvar('c').or_casted().f(Ops.STAGE, allow_any_len=True, name="b"), lambda c,b: b.const_like(c.val)),
  # indexing a const is the const
  (UPat(Ops.INDEX, src=(UPat.cvar().or_casted("c"),),), lambda c: c),
  # indexing an after with all fully invalid stores is invalid
  (UPat(Ops.INDEX, src=(UPat(Ops.AFTER, name="after"),), allow_any_len=True, name="idx"),
   lambda idx,after: idx.const_like(Invalid) if after_all_invalid(after) else None),
  # a deviceless MSTACK src is the same value on every device, so indexing the stack is just indexing that value
  (UPat(Ops.MSTACK, src=(UPat.var("s"),), allow_any_len=True).f(Ops.INDEX, allow_any_len=True, name="idx"),
   lambda s,idx: idx.replace(src=(s,)+idx.src[1:]) if s.device is None else None),
])

pm_remove_bufferize = PatternMatcher([
  # remove reindexing with cost function
  (UPat.var("src").f(Ops.STAGE, allow_any_len=True, name="buf").f(Ops.INDEX, allow_any_len=True, name="idx"), remove_bufferize),
  # STORE to self is NOOP
  (UPat.var("x").store(UPat.var("x")), lambda x: UOp(Ops.NOOP)),
  # END on NOOP is NOOP
  (UPat(Ops.END, src=(UPat(Ops.NOOP, name="x"),), allow_any_len=True), lambda x: x),
])

def strip_zero_offset_shrink(x:UOp) -> UOp:
  return x.src[0] if x.op is Ops.SHRINK and all(resolve(start == 0, False) for start,_ in x.marg) else x

def no_indexing_calls(u:UOp):
  new_srcs = []
  for x in u.src:
    if x.op is Ops.INDEX:
      # sometimes if call srcs have children the call will get an INDEX. we remove it here.
      # TODO: we should add safety checks here for contiguous
      new_srcs.append(x.src[0])
    elif x.op is Ops.SHRINK:
      # SHRINK with offset 0 is fine
      new_srcs.append(strip_zero_offset_shrink(x))
    elif x.op is Ops.MSTACK:
      new_srcs.append(x.replace(src=tuple(strip_zero_offset_shrink(s) for s in x.src)))
    else:
      # everything else we pass through
      new_srcs.append(x)
  return u.replace(src=tuple(new_srcs))

pm_no_indexing_calls = PatternMatcher([
  (UPat(Ops.CALL, name="u"), no_indexing_calls),
])

# the kernel graph is what gets executed: no shape views left in it, the storage of a value is just the storage
pm_no_views = PatternMatcher([
  (UPat((Ops.RESHAPE, Ops.SHRINK), name="v", src=(UPat((Ops.AFTER, Ops.PARAM, Ops.UNSHARD, Ops.MSTACK, Ops.BUFFER)),), allow_any_len=True), lambda v:
   v.src[0]),
])

DEVICE_MAX_BUFS = {"METAL": 31, "WEBGPU": 8, "CPU": 31} # TODO: get from device?
@dataclass
class LimitBufsContext:
  buf_cache: dict[UOp, frozenset[UOp]] = field(default_factory=dict)
  range_idx: itertools.count = field(default_factory=itertools.count)

def _limit_bufs(ctx:LimitBufsContext, root:UOp):
  if (device:=root.device) is None: return None # no device, index related calculations
  device = device if isinstance(device, str) else device[0].split(":")[0]
  if not (MAX_BUFS:=MAX_KERNEL_BUFFERS.value or DEVICE_MAX_BUFS.get(device, 0)): return None

  def visitor(u:UOp) -> frozenset[UOp]:
    if u.op in {Ops.STAGE, Ops.AFTER, Ops.PARAM, Ops.MSELECT, Ops.MSTACK}: return frozenset((u,))
    if len(u.src) == 1: return ctx.buf_cache[u.src[0]]
    return frozenset().union(*[ctx.buf_cache[s] for s in u.src])
  bufs = root.topovisit(visitor, ctx.buf_cache)

  if len(bufs) > MAX_BUFS - 1: # NOTE: this -1 is for the output buffer
    srcs = []
    for s in root.src:
      if s.op in GroupOp.Elementwise and s.device is not None:
        # Insert bufferize: all AxisType.REDUCE before bufferize are AxisType.WEAK, the DEVICE range stays a launched axis
        orig_ranges = s.ranges
        end_ranges = [x.replace(arg=(next(ctx.range_idx), AxisType.WEAK)) if x.op is Ops.RANGE and x.arg[-1] is not AxisType.DEVICE else x
                      for x in s.ranges]
        s = s.substitute(dict(zip(orig_ranges, end_ranges))).bufferize(*end_ranges, arg=BufferizeOpts(device=s.device)).index(*orig_ranges)
      srcs.append(s)
    return root.replace(src=tuple(srcs))
pm_limit_bufs = PatternMatcher([(UPat(set.union(GroupOp.Binary, GroupOp.Ternary), name="root"), _limit_bufs)])

# *****************
# 4. put in buffers for bufferize
# TODO: should BUFFERIZE look a lot more like STORE
# BUFFERIZE has device in arg
# BUFFERIZE doesn't have indexing, that's implied by the ranges it closes
# BUFFERIZE returns the BUFFER ready for INDEXing (doing this will make splitting a lot easier)
# NOTE: this has been fixed up a bit

def bufferize_to_store(ctx:itertools.count, x:UOp, idx:UOp, allow_locals=True):
  size = prod(x.shape)
  dtype = x.commit_dtype()  # a BUFFER is never weak: store at the committed dtype, the .cast(x.dtype) on the result keeps readers unchanged
  rngs = sorted(idx.ranges, key=lambda x: x.arg)
  assert size > 0 and isinstance(size, int), f"no zero sized or symbolic sized buffers {size}"

  # AFTER: add END to the existing STORE, return buffer with kernel dependency
  if (after:=x.src[0]).op is Ops.AFTER:
    buf = after.src[0].buf_uop.base
    if not (stores := [s for s in after.src[1:] if s.op is Ops.STORE and s.src[0].op is Ops.INDEX]): return buf
    # BUFFERIZE(INDEX(...)); store through the underlying global index instead.
    ended_stores = []
    for store in stores:
      store_target = store.src[0]
      if store_target.src[0].op is Ops.STAGE and store_target.src[0].src[0].op is Ops.INDEX:
        store_target = store_target.src[0].src[0]
      if store.src[1] is store_target: continue  # skip self-assign
      end_rngs = sorted(dedup(tuple(store_target.ranges) + tuple(rngs)), key=lambda x: x.arg)
      ended_stores.append(store_target.store(store.src[1]).end(*end_rngs))
    return buf.after(*ended_stores)

  # NOTE: the local BUFFER needs to be disambiguated here
  if x.arg.addrspace == AddrSpace.GLOBAL:
    buf = UOp(Ops.BUFFER, arg=ParamArg(next(ctx), dtype, size=size, device=x.arg.device, addrspace=AddrSpace.GLOBAL))
    do_store = buf.index(idx).store(x.src[0].cast(dtype)).end(*rngs)
    return buf.after(do_store).cast(x.dtype)

  if allow_locals:
    # handle locals
    buf = UOp.placeholder((size,), dtype, next(ctx), AddrSpace.LOCAL)
    do_store = buf.index(idx).store(x.src[0].cast(dtype)).end(*rngs)
    return buf.after(do_store).cast(x.dtype)

# collapse any BUFFERIZE to single input BUFFERIZE
def flatten_bufferize(x:UOp):
  if len(x.src) == 2: return None
  ret = x.replace(src=(x.src[0], get_single_element(apply_movement_op(Ops.RESHAPE, (prod(x.shape),), x.shape, x.src[1:]))))
  rngs = x.src[1:]
  ret = ret.reshape(x.shape)
  if any(r.op is Ops.RANGE and r.src[0].op is not Ops.CONST for r in rngs):
    sym_shape = tuple([r.src[0] if r.op is not Ops.CONST else 1 for r in rngs])
    ret = ret.shrink(tuple([(0,x) for x in sym_shape]))
  return ret
pm_flatten_bufferize = PatternMatcher([(UPat(Ops.STAGE, name="x"), flatten_bufferize)])

def is_noop_after_dep(x:UOp) -> bool:
  return (x.op is Ops.NOOP and len(x.src) == 0) or (x.op is Ops.END and is_noop_after_dep(x.src[0]))

def remove_noop_afters(x:UOp) -> UOp|None:
  src = (x.src[0],) + tuple(s for s in x.src[1:] if not is_noop_after_dep(s))
  if len(src) != len(x.src): return src[0] if len(src) == 1 else x.replace(src=src)
  return None

pm_add_buffers = pm_mops+pm_flatten_bufferize+PatternMatcher([
  (UPat(Ops.STAGE, src=(UPat(), UPat(name="idx")), name="x"), lambda ctx,x,idx: bufferize_to_store(ctx, x, idx, allow_locals=False)),

  # INDEX of a buffer through the weak cast added above: index the buffer directly and cast the loaded value instead.
  # this must run in the same rewrite that adds the cast, or the expander expands the whole casted buffer into one big VECTORIZE
  (UPat(Ops.INDEX, src=(UPat(Ops.CAST, dtype=dtypes.weaks, src=(UPat.var("buf"),)),), allow_any_len=True, name="u"),
   lambda u,buf: u.replace(src=(buf,)+u.src[1:]).cast(u.dtype)),

  # move RESHAPEs through MSELECT/MSTACK
  (UPat((Ops.MSELECT, Ops.MSTACK), src=UPat(Ops.RESHAPE), name="m"),
   lambda m: m.replace(src=tuple([x.src[0].base for x in m.src])).reshape(m.shape)),

  # remove any RESHAPEs on KERNEL
  (UPat(Ops.CALL, name="k"), lambda k: k.replace(src=tuple(x.src[0] if x.op is Ops.RESHAPE else x for x in k.src))),

  # remove invalid writes
  (UPat(Ops.STORE, src=(UPat(), UPat(Ops.CONTIGUOUS, src=(UPat(Ops.CONST, arg=Invalid),)))), lambda: UOp(Ops.NOOP)),
  (UPat(Ops.STORE, src=(UPat(), UPat(Ops.CONST, arg=Invalid))), lambda: UOp(Ops.NOOP)),
  (UPat(Ops.AFTER, name="x"), remove_noop_afters),
])

# *****************
# 5. split into kernels

@dataclass
class LocalAddBufferContext:
  dg:int = 0
  map:dict = field(default_factory=dict)
  range:int = 0
  opts:tuple|None = None

def debuf(ctx:LocalAddBufferContext, buf:UOp):
  # Variables (ALU buffers with a value range) are scalar symbolic values, not real buffers: they become ALU params with no slot
  if buf.is_variable: return buf.replace(op=Ops.PARAM)
  param = UOp(Ops.PARAM, arg=ParamArg(ctx.dg, buf.dtype, prod(buf.max_shape), addrspace=buf.addrspace, device=buf.device))
  ret = param.reshape(buf.max_shape)
  # if the buffer has symbolic shape, shrink the max-sized view to the actual shape
  if buf.max_shape != buf.shape: ret = ret.shrink(tuple((0, s) for s in buf.shape))
  if buf not in ctx.map: ctx.map[buf] = buf
  ctx.dg += 1
  return ret

def handle_after(ctx:LocalAddBufferContext, after:UOp):
  if after.addrspace == AddrSpace.LOCAL: return None
  buf = after.buf_uop
  # NOTE: this is bottom up, so we only add it once
  if buf not in ctx.map: ctx.map[buf] = after
  return buf

def renumber_range(ctx:LocalAddBufferContext, r:UOp):
  if r.tag != (): return None
  ret = r.replace(arg=(ctx.range,)+r.arg[1:], tag=None)
  ctx.range += 1
  return ret

def find_bufs(x:UOp):
  idxs = [s for s in x.toposort(gate=lambda x: x.op is not Ops.AFTER) if s.op is Ops.INDEX]
  read_from: dict[UOp, Ops] = {}
  if any((buf:=idx.buf_uop).op in {Ops.BUFFER, Ops.PARAM} and read_from.setdefault(buf, op:=idx.src[0].op) is not op for idx in idxs):
    raise RuntimeError(f"cycle detected while indexing {buf}")

to_define_global = PatternMatcher([
  (UPat(Ops.STORE, name="x"), find_bufs),
  (UPat((Ops.BUFFER, Ops.MSTACK, Ops.MSELECT), name="buf"), debuf),
  (UPat(Ops.PARAM, name="v"), lambda v:
   v.replace(arg=replace(v.arg, slot=-1)) if v.arg.name is not None and v.arg.vmin_vmax is not None and v.arg.slot != -1 else None),

  # this renumbers the params
  (UPat(Ops.PARAM, name="buf"), lambda ctx, buf:
   None if buf.tag != () or buf.arg.name is not None or buf._shape is None else debuf(ctx, buf)),

  # ALU params are scalar symbolic values, not buffers.
  (UPat(Ops.INDEX, src=(UPat(Ops.PARAM, name="v"),)), lambda v: v if v.addrspace == AddrSpace.ALU else None),

  # bound Variables are stores into Variable buffers: strip the store, the buffer becomes an ALU param via debuf
  (UPat(Ops.AFTER, name="b"), lambda b: b.src[0] if b.is_bound_var else None),
  (UPat(Ops.AFTER, name="after"), handle_after),

  # remove device from local BUFFERIZE
  (UPat(Ops.STAGE, name="b"), lambda b: b.replace(arg=replace(b.arg, device=None))),

  # renumber the ranges starting with 0 so that kernel deduping works
  (UPat(Ops.RANGE, name="r"), renumber_range),
])

def get_contiguous(ctx:LocalAddBufferContext, x:UOp):
  if isinstance(x.arg, tuple) and all(isinstance(y, Opt) for y in x.arg): ctx.opts = x.arg
  return x.src[0]

rangeify_codegen = PatternMatcher([
  (UPat(Ops.CONTIGUOUS, name="x"), get_contiguous),
])

pm_add_param_range_tags = PatternMatcher([
  (UPat((Ops.PARAM, Ops.RANGE), name="x"), lambda x: x.rtag(())),
])

def split_store(x:UOp) -> UOp|None:
  # if we have any open ranges here, we don't split. open DEVICE ranges are fine, they are bound per device at launch
  if any(r.arg[-1] is not AxisType.DEVICE for r in x.ranges): return None
  # the store of a bound Variable is an input value, not a kernel
  st = x.src[0] if x.op is Ops.END else x
  if st.op is Ops.STORE and st.src[0].is_variable: return None

  # local kernel rewrite
  lctx = LocalAddBufferContext()
  ret = graph_rewrite(x, to_define_global+pm_flatten_range+rangeify_codegen, ctx=lctx, name="kernel split", bottom_up=True)

  # create the Kernel. NOTE: buffers can be on different devices here now, they are compiled to SDMA copies later by schedule
  return ret.sink(arg=KernelInfo(opts_to_apply=lctx.opts)).call(*lctx.map.values())

split_kernels = PatternMatcher([
  (UPat((Ops.STORE, Ops.END), name="x"), split_store),
])

@rewrite_group(new_ctx=False)
def get_kernel_graph(tsink:UOp) -> UOp:
  # convert movement ops to ranges
  tsink = run_rangeify(tsink, bool(DEBUG_RANGEIFY))

  # cleanups for speed and runability
  tsink = graph_rewrite(tsink,
                        symbolic+pm_reduce_simplify+pm_const_buffer_folding+pm_remove_bufferize,
                        name="symbolic+reduce_collapse+debuf")
  next_range = max((x.arg[0] for x in tsink.toposort() if x.op is Ops.RANGE), default=-1) + 1
  tsink = graph_rewrite(tsink, pm_limit_bufs, ctx=LimitBufsContext(range_idx=itertools.count(next_range)), name="limit buffers")
  if VIZ: graph_rewrite(tsink, PatternMatcher([]), name="View Rangeify")

  # bufferize -> store
  slots = [x.arg.slot for x in tsink.toposort() if x.op is Ops.BUFFER and isinstance(x.arg, ParamArg) and x.addrspace is AddrSpace.GLOBAL]
  paramarg_start: int = max([-1]+slots) + 1
  tsink = graph_rewrite(tsink, pm_add_buffers+pm_add_param_range_tags, ctx=itertools.count(paramarg_start), bottom_up=True, name="stage to store")
  tsink = graph_rewrite(tsink, split_kernels, bottom_up=True, name="split kernels")
  tsink = graph_rewrite(tsink, pm_no_indexing_calls, name="remove indexing from call args")
  tsink = graph_rewrite(tsink, pm_no_views, name="remove views from the kernel graph")

  if VIZ: graph_rewrite(tsink, PatternMatcher([]), name="View Kernel Graph")
  if SPEC:
    # validate the kernel graph
    from tinygrad.uop.spec import type_verify, spec_kernel_graph
    type_verify(tsink, spec_kernel_graph, enter_calls=False)
  return tsink
