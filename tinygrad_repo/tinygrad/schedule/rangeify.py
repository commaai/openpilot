from dataclasses import dataclass, field, replace
import itertools
from tinygrad.dtype import dtypes, PtrDType, AddrSpace, Invalid
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, resolve, GroupOp, _substitute, KernelInfo
from tinygrad.uop.ops import graph_rewrite, sint, AxisType, BottomUpGate, profile_matches, identity_element
from tinygrad.uop.symbolic import symbolic
from tinygrad.helpers import prod, all_same, getenv, dedup, all_int, DEBUG, SPLIT_REDUCEOP, DEBUG_RANGEIFY, VIZ, MAX_KERNEL_BUFFERS
from tinygrad.helpers import PCONTIG, FLOAT16, OPENPILOT_HACKS, argsort, partition, get_single_element
from tinygrad.codegen.simplify import pm_flatten_range, pm_reduce_simplify
from tinygrad.codegen.opt import Opt
from tinygrad.schedule.indexing import run_rangeify, BufferizeOpts, IndexingContext, apply_movement_op
from tinygrad.schedule.multi import multi_pm
from tinygrad.schedule.allreduce import create_allreduce_function

# creation can recurse a lot
import sys
sys.setrecursionlimit(10000)

def add_ranges_to_store(ctx, x):
  if x.src[0]._shape is None or x.src[1]._shape is None or x.src[0].shape == (): return None
  assert x.src[0].shape == x.src[1].shape, "bad store shape"
  idxs = [UOp.range(r, next(ctx), AxisType.LOOP) for r in x.src[0].shape]
  return UOp.store(x.src[0].index(*idxs), x.src[1].index(*idxs)).end(*idxs)

def lower_shaped_wmma(ctx, x):
  dims, device, threads = x.arg
  dtype_in, dtype_out = x.src[0].dtype.base, x.dtype
  upcasts = [(s, UOp.range(s.shape[-1], next(ctx), axis_type=AxisType.UPCAST)) for s in x.src]
  tc_upcast_axes = tuple(((u.arg[0], s.shape[-1]),) for s, u in upcasts)
  name = f"WMMA_{'_'.join(map(str, dims))}_{dtype_in.name}_{dtype_out.name}"
  wmma_arg = (name, dims, dtype_in, dtype_out, device, threads, tc_upcast_axes, ())
  wmma = UOp(Ops.WMMA, dtype_out.vec(x.src[2].shape[-1]), tuple(s[u].contract(u) for s, u in upcasts), arg=wmma_arg)
  tmp = UOp.placeholder((x.src[2].shape[-1],), dtype_out, slot=next(ctx), addrspace=AddrSpace.REG)
  return tmp.after(UOp.group(*[tmp[e].store(wmma.gep(e)) for e in range(x.src[2].shape[-1])]))

pm_store_ranges = PatternMatcher([
  (UPat(Ops.STORE, name="x"), add_ranges_to_store),
])

pm_syntactic_sugar = PatternMatcher([
  # INDEX on ptr INDEX concats them
  (UPat(Ops.INDEX, name="i1").f(Ops.INDEX, name="i2", allow_any_len=True),
   lambda i1,i2: i2.replace(src=i1.src+i2.src[1:]) if isinstance(i1.dtype, PtrDType) and not isinstance(i2.dtype, PtrDType) else None),
  # early rangeify
  (UPat(Ops.INDEX, src=(UPat(GroupOp.Elementwise | {Ops.CONST}, name="x"),), allow_any_len=True, name="idx"),
   lambda idx,x: x.replace(src=tuple([s.index(*idx.src[1:]) for s in x.src]))),
])

def found_after(ctx:dict[UOp, UOp], after:UOp, src:UOp):
  if (x:=src).op is Ops.CAST and x.dtype == dtypes.half and FLOAT16: x, after = x.src[0], after.cast(dtypes.float)
  while True:
    if x.op is Ops.PERMUTE: x, after = x.src[0], after.permute(argsort(x.marg))
    elif x.op is Ops.RESHAPE: x, after = x.src[0], after.reshape(x.src[0].shape)
    elif x.op is Ops.WHERE and x.src[2].base.arg == Invalid and x.src[1].op is Ops.PAD:
      x, after = x.src[1].src[0], after.shrink(tuple((l, s-r) for (l,r),s in zip(x.src[1].marg, x.shape)))
    else: break
  ctx[x] = after

# *** fold moved AFTERs (hack for openpilot) ***
pm_fold_moved_after = PatternMatcher([
  (UPat(Ops.AFTER, src=(UPat(), UPat(Ops.STORE, src=(UPat(), UPat((*GroupOp.Movement,Ops.CAST,Ops.WHERE), name="src")))), name="after"), found_after),
  # replace ALU sources with AFTER versions found above
  (UPat(GroupOp.ALU, name="alu"), lambda ctx,alu: alu.replace(src=new_src) if (new_src:=tuple(ctx.get(s, s) for s in alu.src)) != alu.src else None),
])

# movement op on INDEX as a PatternMatcher
def _mop_index(r:UOp, idx:UOp):
  idxs = idx.src[1:]
  if len(idxs) == len(r.shape):
    return r.src[0].index(*apply_movement_op(r.op, r.src[0].shape, r.marg, idxs), dtype=idx.dtype, arg=idx.arg)
  if r.op is Ops.RESHAPE:
    src_prefix = len(r.src[0].shape) - len(r.shape[len(idxs):])
    if src_prefix >= 0 and r.src[0].shape[src_prefix:] == r.shape[len(idxs):]:
      if src_prefix == 0: return r.src[0] if r.src[0].dtype == idx.dtype else None
      ret = r.src[0].index(*apply_movement_op(r.op, r.src[0].shape[:src_prefix], r.shape[:len(idxs)], idxs), dtype=idx.dtype, arg=idx.arg)
      return ret if ret.shape == idx.shape else None

pm_mops = PatternMatcher([
  (UPat(GroupOp.Movement, name="r").f(Ops.INDEX, allow_any_len=True, name="idx"), _mop_index),
  # move movement ops and INDEX after AFTER (but not when AFTER has a raw STORE with shaped children — from replace_contig_with_store_after)
  (UPat(GroupOp.Movement|{Ops.INDEX}, name="r").after(name="a", allow_any_len=True),
   lambda r,a: UOp(r.op, r.dtype, (a.replace(src=(r.src[0],)+a.src[1:]),)+r.src[1:], r.arg)),
  (UPat(GroupOp.Movement, name="r").end(name="a", allow_any_len=True), lambda r,a: a.replace(src=(r.src[0],)+a.src[1:])),
  # lower SHAPED_WMMA to WMMA with CONTRACT/UNROLL
  (UPat(Ops.SHAPED_WMMA, name="x"), lower_shaped_wmma),
])

# *****************
# 0. do some cleanup rewrites, mostly copied from the old stuff

def fix_store_hazard(target:UOp, src:UOp):
  # PERMUTE and FLIP reorder indices, SHRINK can have overlapping regions when dest is also shrunk
  unsafe = {Ops.PERMUTE, Ops.FLIP} | ({Ops.SHRINK} if target.op_in_backward_slice_with_self(Ops.SHRINK) else set())
  base = target.base
  reaches_base: dict[UOp, bool] = {}
  for s in src.toposort(gate=lambda s: s.op is not Ops.CONTIGUOUS):
    reaches_base[s] = s is base or any(reaches_base.get(c) for c in s.src)
    if reaches_base[s] and s.op in unsafe: return target.store(src.contiguous())

def split_reduceop(reduce:UOp, x:UOp):
  if prod(reduce.shape) == 0: return None
  if not SPLIT_REDUCEOP or not all_int(x.shape) or (prod(x.shape)//prod(reduce.shape))<getenv("REDUCEOP_SPLIT_THRESHOLD", 32768): return None
  # if there are few globals, make some reduces into globals by splitting into two kernels
  # cap output buffer to 2**22: heuristic number of global outputs to achieve max occupancy with enough locals+upcasts for gemm
  #   ~2**10 should be enough if GROUP is used
  # 256 split maximum should be "negligible reduce" for low prod(reduce.shape), 8 split minimum.
  # split is moved to the end to provide maximum locality for the second phase reduce.

  # get expanded by rangeifying the UOp x
  indexed = x.index(*[UOp.range(s, i) if resolve(s>1) else UOp.const(dtypes.weakint, 0) for i,s in enumerate(x.shape)])
  range_nums = [y.arg[0] for y in indexed.substitute({x.base:UOp(Ops.NOOP)}, extra_pm=pm_mops).ranges]
  is_expanded = [i not in range_nums for i in range(len(x.shape))]

  if not (split_candidates:=[(i,d) for i in reduce.arg[1] for d in range(min(256,2**getenv("REDUCEOP_SPLIT_SIZE",22)//prod(reduce.shape)),8-1,-1)
                             if x.shape[i]%d==0 and not is_expanded[i]]): return None
  dim_to_split, divisor = split_candidates[0]
  splitted_shape = x.shape[:dim_to_split]+(divisor,)+(x.shape[dim_to_split]//divisor,)+x.shape[dim_to_split+1:]
  splitted = x.reshape(splitted_shape).permute(tuple([d for d in range(len(splitted_shape)) if d!=dim_to_split]+[dim_to_split]))
  if DEBUG >= 3: print(f"split {divisor}: {x.shape} -> {splitted.shape} -> {reduce.shape}")
  # reduce original axes, then split
  return splitted._rop(*reduce.arg).contiguous()._rop(reduce.arg[0], (len(reduce.shape),)).reshape(reduce.shape)

mop_cleanup = PatternMatcher([
  # merge adjacent RESHAPES
  (UPat(Ops.RESHAPE, src=(UPat(Ops.RESHAPE, name="x2"), UPat()), name="x"), lambda x,x2: x.replace(src=(x2.src[0], x.src[1]))),
])

pm_gather_params = PatternMatcher([ (UPat(Ops.PARAM, name="p"), lambda ctx, p: ctx.append(p)), ])
def resolve_function(c:UOp, allow_param_mismatch=True) -> UOp|None:
  if c.arg.precompile: return None
  params: list[UOp] = []
  graph_rewrite(c.src[0], pm_gather_params, bottom_up=True, ctx=params, name="gather params")
  params = sorted(params, key=lambda x: x.arg)
  args = c.src[1:]

  # NOTE: this isn't really needed. it's okay if there's unused args in the function
  if not allow_param_mismatch:
    if [x.arg for x in params] != list(range(len(params))): raise RuntimeError(f"params not in order: {[x.arg for x in params]}")
    if len(params) != len(args): raise TypeError(f"expected {len(params)} args, got {len(args)}")

  dict_map = {x:args[x.arg] for x in params}
  for i, (p, a) in enumerate(dict_map.items()):
    if p.axis != a.axis: raise TypeError(f"arg {i} axis mismatch: expected {p.axis}, got {a.axis}")
    if p.max_shape != a.max_shape: raise TypeError(f"arg {i} shape mismatch: expected {p.shape}, got {a.shape}")
    if p.dtype != a.dtype: raise TypeError(f"arg {i} dtype mismatch: expected {p.dtype}, got {a.dtype}")
  return c.src[0].substitute(dict_map, walk=True)

earliest_rewrites = mop_cleanup+PatternMatcher([
  # early fixup const copy
  (UPat(Ops.COPY, src=(UPat.var("s"), UPat.var("d"))),
   lambda s,d: s.substitute({UOp(Ops.DEVICE, arg=s.device):d}) if s.base.op is Ops.CONST else None),

  # resolve FUNCTION calls (inline the body)
  (UPat(Ops.FUNCTION, name="c"), resolve_function),

  # resolve TUPLE+GETTUPLE
  (UPat(Ops.GETTUPLE, src=(UPat(Ops.TUPLE, name="t"),), name="g"), lambda g,t: t.src[g.arg]),

  # resolve allreduce (must be bottom up)
  (UPat(Ops.ALLREDUCE, src=(UPat.var("buf"), UPat()), name="red"), create_allreduce_function),

  # split_reduceop
  (UPat(Ops.REDUCE, name="reduce", src=(UPat.var("x"),)), split_reduceop),

  # remove DETACH/CONTIGUOUS_BACKWARD (TODO: this is copied in allocations)
  (UPat((Ops.DETACH, Ops.CONTIGUOUS_BACKWARD), name="x"), lambda x: x.src[0]),

  # remove contiguous on movement ops before a copy on disk
  (UPat(GroupOp.Movement-{Ops.SHRINK, Ops.RESHAPE}, name="x").f(Ops.CONTIGUOUS).f(Ops.COPY, allow_any_len=True, name="copy"),
   lambda x,copy: copy.replace(src=(x,)+copy.src[1:]) if isinstance(x.device, str) and x.device.startswith("DISK") else None),
  # push copy past movement ops to disk
  (UPat(GroupOp.Movement-{Ops.SHRINK, Ops.RESHAPE}, name="x").f(Ops.COPY, allow_any_len=True, name="copy"),
   lambda x,copy: x.replace(src=(copy.replace(src=(x.src[0],)+copy.src[1:]),)+x.src[1:]) \
      if isinstance(x.device, str) and x.device.startswith("DISK") else None),

  # SINK only ever references the base
  (UPat(Ops.SINK, name="x"), lambda x: x.replace(src=tuple(y.base for y in x.src))),

  # ** copy rules **

  # COPY and source size need to match
  (UPat(Ops.COPY, src=(UPat(GroupOp.Movement, name="r"), UPat(name="d")), name="c"),
   lambda c,r,d: c.replace(src=(r.contiguous(), d)) if resolve(r.numel() != r.base.numel(), False) else None),

  # copy only to different device
  (UPat(Ops.COPY, src=(UPat.var("x"), UPat()), name="copy"), lambda x,copy: x.f(Ops.NOOP) if x.device == copy.device else None),

  # ** store rules **

  # fix store hazard (dest is in used in src) by adding contiguous: TestAssign.test_post_flipped_assignment
  (UPat(Ops.STORE, src=(UPat(name="target"), UPat(name="src"))), fix_store_hazard),

  # remove two STOREs that store the same thing to the same place: TestSchedule.test_dedup_assign
  (UPat.var("buf").after(UPat.var("buf").store(UPat.var("src")), name="a1").after(UPat.var("a1").store(UPat.var("src"))), lambda buf,src,a1:a1),

  # store a buffer's own current contents back into itself: TestAssign.test_nested_after_contiguous_store_no_init
  (UPat.var("buf").after(UPat.var("buf").store(UPat.var("buf").after(UPat.var("buf").store(UPat.var("src")), name="a1"))), lambda buf,src,a1:a1),

  # move bitcast from store dest to source: TestAssign.test_assign_bitcast
  (UPat(Ops.STORE, src=(UPat(Ops.BITCAST, src=(UPat(name="target"),)), UPat(name="src"))),
   lambda target, src: target.store(src.bitcast(target.dtype))),

  # ** size 0 **

  # reduce of size 0 is the identity element
  (UPat(Ops.REDUCE, name="reduce", src=(UPat.var("x"),)),
   lambda reduce,x: reduce.const_like(identity_element(reduce.arg[0], reduce.dtype)) if 0 in x.shape and 0 not in reduce.shape else None),
  # handle size 0
  (UPat(GroupOp.All-{Ops.SINK}, name="x"), lambda x: x.const_like(0).rtag(x.tag) if x._shape is not None and 0 in x.shape else None),
])

# *****************
# 3.5 cleanups

ALWAYS_RUN_OPS = {Ops.CONTIGUOUS, Ops.COPY, Ops.NOOP}

# you don't know in the first pass if axes are going to die, this happens if there's an EXPAND to the left
def cleanup_dead_axes(b:UOp):
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
  if len(accessed_buffers) > 3 and not (PCONTIG > 2): return None

  # if any reduces access a buffer, don't remove this buffer
  buffer_in_reduce = False
  def buf_gate(x:UOp):
    nonlocal buffer_in_reduce
    if x.op in {Ops.PARAM, Ops.STAGE}: buffer_in_reduce = True
    return not buffer_in_reduce
  UOp.sink(*[x.src[0] for x in reduces]).toposort(gate=buf_gate)
  del buf_gate
  if buffer_in_reduce:
    if PCONTIG > 2:
      out_in_ratio = (prod(buf.shape)+1) / (sum([x.numel() for x in accessed_buffers])+1)
      if out_in_ratio < 10: return None
      # here we have to check the indexes, we might do a partial contig here
      local_indexes = [x for x in indexes if x.src[0].op is Ops.STAGE and x.src[0].arg.addrspace == AddrSpace.LOCAL]
      exclude_ranges = UOp.group(*[UOp.group(*x.src[1:]) for x in local_indexes]).ranges
      subs = [(k,v) for k,v in zip(buf.src[1:], idx.src[1:]) if k.op is not Ops.CONST]
      # if it's bufferized or a reduce, it's pcontig
      is_pcontig, is_subs = partition(subs, lambda x: x[0] in exclude_ranges or any([r.arg[-1] == AxisType.REDUCE for r in x[1].ranges]))
      if not len(is_subs):
        return None
      if len(is_pcontig):
        ret = src.substitute(dict(is_subs), extra_pm=pm_gate_substitute)
        return ret.bufferize(*[x[0] for x in is_pcontig], arg=BufferizeOpts(None, AddrSpace.LOCAL)).index(*[x[1] for x in is_pcontig])
    else:
      return None

  # if it makes it here, the bufferize is removed
  # this is the ranges replaced
  # NOTE: if buf src is a const, we don't replace it. if idx is Invalid (dead load), don't replace it either
  replaced = {k:v for k,v in zip(buf.src[1:], idx.src[1:]) if k.op is not Ops.CONST and not (v.op is Ops.CONST and v.arg is Invalid)}
  return src.substitute(replaced, extra_pm=pm_gate_substitute)

def remove_noop_bufferize(idx,b2):
  if idx.src[1:] != b2.src[1:] or idx.src[0].op is Ops.BUFFER_VIEW: return None
  return idx.src[0].shrink(tuple((0, s) for s in b2.shape)) if b2.shape else idx.src[0]

pm_const_buffer_folding = pm_mops+PatternMatcher([
  (UPat(Ops.STAGE, name="b"), cleanup_dead_axes),
  # remove noop buffers. if we look at the next index we can remove even more of these
  (UPat(Ops.INDEX, name="idx").f(Ops.STAGE, allow_any_len=True, name="b2"), remove_noop_bufferize),
  # no buffers for const (ranges don't matter for const - it's the same value everywhere)
  (UPat(Ops.CONST, name='c').f(Ops.STAGE, allow_any_len=True, name="b"), lambda c,b: b.const_like(c.arg)),
  # indexing a const is a const
  (UPat(Ops.INDEX, src=(UPat(Ops.CONST, name="c"),),), lambda c: c),
  # copy on CONST is CONST
  (UPat(Ops.COPY, src=(UPat.cvar("x"), UPat()), name="copy"), lambda copy,x: copy.const_like(x.arg)),
  # hack if a noop turned to a const
  (UPat(Ops.NOOP, src=(UPat.cvar("c"),)), lambda c: c),
  # mstack on CONST is CONST
  (UPat(Ops.MSTACK, src=(UPat.var("s"),), allow_any_len=True).f(Ops.INDEX, allow_any_len=True),
   lambda s: UOp.const(c.dtype, c.arg) if (c:=s.base).op is Ops.CONST else None),
])

pm_remove_bufferize = PatternMatcher([
  # remove reindexing with cost function
  (UPat.var("src").f(Ops.STAGE, allow_any_len=True, name="buf").f(Ops.INDEX, allow_any_len=True, name="idx"), remove_bufferize),
  # STORE to self is NOOP
  (UPat.var("x").store(UPat.var("x")), lambda x: UOp(Ops.NOOP)),
  # END on NOOP is NOOP
  (UPat(Ops.END, src=(UPat(Ops.NOOP, name="x"),), allow_any_len=True), lambda x: x),
])

def late_buffer_view(t:UOp, b:UOp):
  if not (isinstance(b.device, str) and b.device.startswith(("DISK", "TINYFS"))): return b
  shape = b.shape
  size = prod(shape)

  # walk up for the INDEX
  x = t
  while not any(u.op is Ops.INDEX for u in x.src):
    assert x.op not in GroupOp.Elementwise, "can't buffer view elementwise"
    x = x.src[0]
  x = next(u for u in x.src if u.op is Ops.INDEX)

  if len(shape) == 0: offset = x.src[1].arg
  else: offset = max(sum(idx.vmin for idx in x.src[1:]), 0)

  return b.replace(src=(UOp(Ops.BUFFER_VIEW, t.dtype, (x.base,), (size, offset)), b.src[1]))

to_bufferview = PatternMatcher([
  (UPat(Ops.STAGE, src=(UPat((Ops.BITCAST, Ops.CONTIGUOUS), name="t"), UPat()), name="b"), late_buffer_view),
])

DEVICE_MAX_BUFS = {"METAL": 31, "WEBGPU": 8} # TODO: get from device?
def limit_bufs(ctx:IndexingContext, root:UOp):
  if (device:=root.device) is None: return None # no device, index related calculations
  device = device if isinstance(device, str) else device[0].split(":")[0]
  if not (MAX_BUFS:=MAX_KERNEL_BUFFERS.value or DEVICE_MAX_BUFS.get(device, 0)): return None

  bufs: set[UOp] = set()
  def gate_input(u:UOp):
    # TODO: add cache to fix n^2
    if is_load:=(u.op in {Ops.STAGE, Ops.AFTER, Ops.PARAM, Ops.MSELECT, Ops.MSTACK, Ops.DEFINE_VAR}): bufs.add(u)
    return not is_load
  root.toposort(gate=gate_input)

  if len(bufs) > MAX_BUFS - 1: # NOTE: this -1 is for the output buffer
    srcs = []
    for s in root.src:
      if s.op in GroupOp.Elementwise and s.device is not None:
        # Insert bufferize: all AxisType.REDUCE before bufferize are AxisType.LOOP
        orig_ranges, end_ranges = s.ranges, [x.replace(arg=(next(ctx.range_idx), AxisType.LOOP)) if x.op is Ops.RANGE else x for x in s.ranges]
        s = s.substitute(dict(zip(orig_ranges, end_ranges))).bufferize(*end_ranges, arg=BufferizeOpts(device=s.device)).index(*orig_ranges)
      srcs.append(s)
    return root.replace(src=tuple(srcs))
pm_limit_bufs = PatternMatcher([(UPat(set.union(GroupOp.Binary, GroupOp.Ternary), name="root"), limit_bufs)])

# *****************
# 4. put in buffers for bufferize
# TODO: should BUFFERIZE look a lot more like STORE
# BUFFERIZE has device in arg
# BUFFERIZE doesn't have indexing, that's implied by the ranges it closes
# BUFFERIZE returns the BUFFER ready for INDEXing (doing this will make splitting a lot easier)
# NOTE: this has been fixed up a bit

def bufferize_to_store(ctx:itertools.count, x:UOp, idx:UOp, allow_locals=True):
  size = prod(x.shape) // x.dtype.count
  rngs = sorted(idx.ranges, key=lambda x: x.arg)
  assert size > 0 and isinstance(size, int), f"no zero sized or symbolic sized buffers {size}"

  sdtype = x.dtype.ptr(size=size, addrspace=x.arg.addrspace)
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
      ended_stores.append(store_target.replace(dtype=sdtype).store(store.src[1]).end(*end_rngs))
    return buf.after(*ended_stores)

  # NOTE: the DEFINE_LOCAL needs to be disambiguated here
  if sdtype.addrspace == AddrSpace.GLOBAL:
    buf = UOp(Ops.BUFFER, x.dtype, (UOp(Ops.LUNIQUE, arg=next(ctx)), UOp(Ops.DEVICE, arg=x.arg.device)), size)
    do_store = buf.index(idx, dtype=sdtype).store(x.src[0]).end(*rngs)
    return buf.after(do_store)

  if allow_locals:
    # handle locals
    buf = UOp.placeholder((size,), x.dtype, next(ctx), AddrSpace.LOCAL)
    do_store = buf.broadcast(x.src[1].dtype.count).index(idx, dtype=sdtype).store(x.src[0]).end(*rngs)
    return buf.after(do_store.barrier())

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

pm_add_buffers = pm_mops+pm_flatten_bufferize+to_bufferview+PatternMatcher([
  (UPat(Ops.STAGE, src=(UPat(), UPat(name="idx")), name="x"), lambda ctx,x,idx: bufferize_to_store(ctx, x, idx, allow_locals=False)),

  # move RESHAPEs through MSELECT/MSTACK
  (UPat((Ops.MSELECT, Ops.MSTACK), src=UPat(Ops.RESHAPE), name="m"),
   lambda m: m.replace(src=tuple([x.src[0].base for x in m.src])).reshape(m.shape)),

  # remove any RESHAPEs on KERNEL
  (UPat(Ops.CALL, name="k"), lambda k: k.replace(src=tuple(x.src[0] if x.op is Ops.RESHAPE else x for x in k.src))),

  # remove MOP on AFTER
  (UPat(Ops.AFTER, src=(UPat.var("x"), UPat(GroupOp.Movement, name="y"))), lambda x,y: x.after(y.src[0])),
  # remove double AFTER
  (UPat(Ops.AFTER, src=(UPat.var("x"), UPat(Ops.AFTER, name="y"))), lambda x,y: x.after(*y.src[1:])),

  # remove invalid writes
  (UPat(Ops.STORE, src=(UPat(), UPat(Ops.CONTIGUOUS, src=(UPat(Ops.CONST, arg=Invalid),)))), lambda: UOp(Ops.NOOP)),
  (UPat(Ops.STORE, src=(UPat(), UPat(Ops.CONST, arg=Invalid))), lambda: UOp(Ops.NOOP)),
  (UPat(Ops.AFTER, src=(UPat.var("x"), UPat(Ops.NOOP, src=()))), lambda x: x),
  (UPat(Ops.AFTER, src=(UPat.var("x"), UPat(Ops.END, src=(UPat(Ops.NOOP, src=()),), allow_any_len=True))), lambda x: x),
])

pm_add_buffers_local = pm_mops+pm_flatten_bufferize+to_bufferview+PatternMatcher([
  (UPat(Ops.STAGE, src=(UPat(), UPat(name="idx")), name="x"), bufferize_to_store),
])

# *****************
# 5. split into kernels

@dataclass
class LocalAddBufferContext:
  dg:int = 0
  map:dict = field(default_factory=dict)
  vars:dict = field(default_factory=dict)
  range:int = 0
  opts:tuple|None = None

def debuf(ctx:LocalAddBufferContext, buf:UOp):
  ret = UOp(Ops.PARAM, buf.dtype.ptr(prod(buf.max_shape)), arg=ctx.dg).reshape(buf.max_shape)
  # if the buffer has symbolic shape, shrink the max-sized view to the actual shape
  if buf.max_shape != buf.shape: ret = ret.shrink(tuple((0, s) for s in buf.shape))
  if buf not in ctx.map: ctx.map[buf] = buf
  ctx.dg += 1
  return ret

def unbind_kernel(ctx:LocalAddBufferContext, b:UOp):
  ctx.vars[b] = None
  return b.src[0]

def handle_after(ctx:LocalAddBufferContext, after:UOp):
  if isinstance(after.dtype, PtrDType) and after.ptrdtype.addrspace == AddrSpace.LOCAL: return None
  buf = after.buf_uop
  # HACK to put the buffer in the MAP instead of MSTACK/MSELECT
  if buf.op in {Ops.MSTACK, Ops.MSELECT}: buf = buf.src[0]
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
  (UPat(Ops.BUFFER, name="buf"), debuf),
  (UPat(Ops.PARAM, src=(UPat(), UPat(Ops.DEVICE)), name="buf"), debuf),
  (UPat(Ops.PARAM, src=(UPat(), UPat(), UPat.cvar('vmin'), UPat.cvar('vmax'), UPat.var("nm")), name="v"),
   lambda v, vmin, vmax, nm: UOp.variable(nm.arg, vmin.arg, vmax.arg, v.dtype)),
  (UPat(Ops.INDEX, src=(UPat(Ops.DEFINE_VAR, name="v"),)), lambda v: v),

  (UPat(Ops.BIND, name="b"), unbind_kernel),
  (UPat((Ops.MSTACK, Ops.MSELECT, Ops.AFTER), name="after"), handle_after),

  # remove device from local BUFFERIZE
  (UPat(Ops.STAGE, name="b"), lambda b: b.replace(arg=replace(b.arg, device=None))),

  # remove UNIQUE/DEVICE to dedup CONST
  (UPat(Ops.CONST, name="c"), lambda c: c.replace(src=()) if len(c.src) else None),

  # renumber the ranges starting with 0 so that kernel deduping works
  (UPat(Ops.RANGE, name="r"), renumber_range),
])

def get_contiguous(ctx:LocalAddBufferContext, x:UOp):
  if isinstance(x.arg, tuple) and all(isinstance(y, Opt) for y in x.arg): ctx.opts = x.arg
  return x.src[0]

rangeify_codegen = PatternMatcher([
  (UPat(Ops.CONTIGUOUS, name="x"), get_contiguous),

  # no NOOP in the kernel graph
  # TODO: this can be moved into codegen?
  (UPat(Ops.NOOP, name="x"), lambda x: x.src[0] if len(x.src) else None),

  # fix broadcast dtype
  (UPat(Ops.AFTER, name="a").broadcast(name="b"), lambda a,b: a.broadcast(len(b.src))),
  (UPat(Ops.DEFINE_LOCAL).f(Ops.AFTER, allow_any_len=True).broadcast(name="dg").f(Ops.INDEX, name="idx", allow_any_len=True),
    lambda dg,idx: None if isinstance(idx.dtype, PtrDType) else
      idx.replace(dtype=dg.dtype, arg=None).load(dtype=dg.dtype.base.scalar().vec(dg.dtype.vcount))),
  (UPat(Ops.AFTER, name="a").gep(name="b"), lambda a,b: a.gep(b.arg)),
  (UPat(Ops.DEFINE_LOCAL).f(Ops.AFTER, allow_any_len=True).gep(name="dg").f(Ops.INDEX, name="idx", allow_any_len=True),
    lambda dg,idx: None if isinstance(idx.dtype, PtrDType) else
      idx.replace(dtype=dg.dtype, arg=None).load(dtype=dg.dtype.base.scalar().vec(dg.dtype.vcount))),
])

pm_add_range_tags = PatternMatcher([
  (UPat(Ops.RANGE, name="x"), lambda x: x.rtag(())),
])

def split_store(x:UOp) -> UOp|None:
  # if we have any open ranges here, we don't split
  if x.ranges: return None

  # local kernel rewrite
  lctx = LocalAddBufferContext()
  ret = graph_rewrite(x, to_define_global+pm_flatten_range+rangeify_codegen, ctx=lctx, name="kernel split", bottom_up=True)

  # SINK requires all buffers on the same device, but COPY/BUFFER_VIEW are cross-device or special hardware ops
  if ret.op is Ops.STORE: stored = ret.src[1]
  elif ret.op is Ops.END and ret.src[0].op is Ops.STORE: stored = ret.src[0].src[1]
  else: raise RuntimeError(f"unknown kernel type {ret.op}")
  if stored.op in {Ops.COPY, Ops.BUFFER_VIEW}: ret = stored.replace(src=stored.src + ret.ended_ranges)
  else: ret = ret.sink(arg=KernelInfo(opts_to_apply=lctx.opts))

  kernel = ret.call(*lctx.map.values(), *lctx.vars.keys())
  if ret.op is Ops.SINK and not all_same([x.device for x in kernel.src[1:] if x.op is not Ops.BIND]):
    raise RuntimeError(f"all buffers must be on the same device: {tuple(b.buf_uop for b in kernel.src[1:])}")
  return kernel

split_kernels = PatternMatcher([
  (UPat((Ops.STORE, Ops.END), name="x"), split_store),
])

@profile_matches
def get_kernel_graph(sink:UOp) -> UOp:
  tsink = graph_rewrite(sink, multi_pm, name="multi_pm")
  if OPENPILOT_HACKS: tsink = graph_rewrite(tsink, pm_fold_moved_after, ctx={}, name="fold moved afters")
  tsink = graph_rewrite(tsink, pm_syntactic_sugar+pm_mops+earliest_rewrites, bottom_up=True, name="earliest rewrites")

  # convert movement ops to ranges
  tsink, rctx = run_rangeify(tsink, bool(DEBUG_RANGEIFY))

  tsink = graph_rewrite(tsink, symbolic+pm_reduce_simplify+pm_const_buffer_folding+pm_remove_bufferize, name="symbolic+reduce_collapse+debuf")
  tsink = graph_rewrite(tsink, pm_limit_bufs, ctx=rctx, name="limit buffers")

  if VIZ: graph_rewrite(tsink, PatternMatcher([]), name="View Rangeify")

  # bufferize -> store
  lunique_start: int = max([-1]+[x.arg for x in tsink.toposort() if x.op is Ops.LUNIQUE]) + 1
  tsink = graph_rewrite(tsink, pm_add_buffers+pm_add_range_tags, ctx=itertools.count(lunique_start), bottom_up=True, name="stage to store")
  tsink = graph_rewrite(tsink, split_kernels, bottom_up=True, name="split kernels")

  # WAR deps: if kernel U reads buffer S, and S is also written by another kernel, S's write must wait for U to finish
  afters = [u for u in tsink.toposort() if u.op is Ops.AFTER]
  kernel_assign: dict[UOp, UOp] = {u.buf_uop:u for u in afters}
  assign_rep: dict[UOp, UOp] = {}
  for u in afters:
    for s in u.src[1].src:
      # TODO: this is probably broken for MSELECT/MSTACK
      if s.op not in {Ops.BUFFER, Ops.PARAM} or s is u.buf_uop or (a:=kernel_assign.get(s)) is None: continue
      if a.src[1] is u.src[1]: continue  # same kernel (multi-output custom kernels)
      if any(x.op is Ops.AFTER and x.buf_uop is s for x in kernel_assign[u.buf_uop].backward_slice):
        raise RuntimeError(f"cycle detected in assign graph, buffers {s} and {u.buf_uop} have circular dependency")
      assign_rep[a] = kernel_assign[s] = a.replace(src=a.src+(u,))
  if assign_rep: tsink = graph_rewrite(tsink, _substitute, ctx=assign_rep, bottom_up=True, name="fix_assign")
  if VIZ: graph_rewrite(tsink, PatternMatcher([]), name="View Kernel Graph")
  return tsink
