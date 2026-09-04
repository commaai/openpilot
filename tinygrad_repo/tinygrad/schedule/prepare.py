import itertools
from tinygrad.dtype import dtypes, to_dtype
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp, resolve, GroupOp
from tinygrad.uop.ops import graph_rewrite, rewrite_group, ParamArg, identity_element, resolve_returned_after
from tinygrad.uop.movement import mop_cleanup
from tinygrad.helpers import prod, getenv, all_int, DEBUG, SPLIT_REDUCEOP, OPENPILOT_HACKS, FLOAT16, argsort
from tinygrad.schedule.indexing import apply_movement_op
from tinygrad.schedule.allreduce import create_allreduce_function
from tinygrad.schedule.multi import multi_pm

def walk_mop(u:UOp):
  if u.op in GroupOp.Movement or u.op in {Ops.INDEX, Ops.UNSHARD, Ops.BITCAST}: return walk_mop(u.src[0])
  if u.op is Ops.AFTER and (b:=walk_mop(u.src[0])) is not u.src[0]: return b.after(*u.src[1:])
  return u

def found_after(ctx:dict[UOp, UOp], after:UOp, src:UOp):
  if (x:=src).op is Ops.CAST and x.dtype == dtypes.half and FLOAT16: x, after = x.src[0], after.cast(dtypes.float)
  while True:
    if x.op is Ops.PERMUTE: x, after = x.src[0], after.permute(argsort(x.marg))
    elif x.op is Ops.RESHAPE: x, after = x.src[0], after.reshape(x.src[0].shape)
    elif x.op is Ops.WHERE and x.src[2].base.is_invalid and x.src[1].op is Ops.PAD:
      x, after = x.src[1].src[0], after.shrink(tuple((o, s+o) for (o,_),s in zip(x.src[1].marg, x.src[1].src[0].shape)))
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
    return r.src[0].index(*apply_movement_op(r.op, r.src[0].shape, r.marg, idxs), arg=idx.arg)
  if r.op is Ops.RESHAPE:
    src_prefix = len(r.src[0].shape) - len(r.shape[len(idxs):])
    if src_prefix >= 0 and r.src[0].shape[src_prefix:] == r.shape[len(idxs):]:
      if src_prefix == 0: return r.src[0]
      ret = r.src[0].index(*apply_movement_op(r.op, r.src[0].shape[:src_prefix], r.shape[:len(idxs)], idxs), arg=idx.arg)
      return ret if ret.shape == idx.shape else None

pm_mops = PatternMatcher([
  # handle movement ops on INDEX
  (UPat(GroupOp.Movement, name="r").f(Ops.INDEX, allow_any_len=True, name="idx"), _mop_index),
  # move movement ops and INDEX after AFTER
  (UPat(GroupOp.Movement|{Ops.INDEX}, name="r").after(name="a", allow_any_len=True),
   lambda r,a: UOp(r.op, src=(a.replace(src=(r.src[0],)+a.src[1:]),)+r.src[1:], arg=r.arg)),
  (UPat(GroupOp.Movement, name="r").end(name="a", allow_any_len=True), lambda r,a: a.replace(src=(r.src[0],)+a.src[1:])),
])

# *****************
# 0. do some cleanup rewrites, mostly copied from the old stuff

def fix_store_hazard(target:UOp, src:UOp):
  if (base:=target.base) not in src.toposort(enter_calls=False): return None
  # PERMUTE and FLIP reorder indices, SHRINK can have overlapping regions when dest is also shrunk
  unsafe = {Ops.PERMUTE, Ops.FLIP} | ({Ops.SHRINK} if target.op_in_backward_slice_with_self(Ops.SHRINK) else set())
  reaches_base: dict[UOp, bool] = {}
  for s in src.toposort(gate=lambda s: s.op is not Ops.CONTIGUOUS):
    reaches_base[s] = s is base or any(reaches_base.get(c) for c in s.src)
    if reaches_base[s] and s.op in unsafe and not (s is target and s.op is Ops.SHRINK): return target.store(src.contiguous())

def split_reduceop(reduce:UOp, x:UOp):
  if prod(reduce.shape) == 0: return None
  if not SPLIT_REDUCEOP or not all_int(x.shape) or (prod(x.shape)//prod(reduce.shape))<getenv("REDUCEOP_SPLIT_THRESHOLD", 32768): return None
  # if there are few globals, make some reduces into globals by splitting into two kernels
  # cap output buffer to 2**22: heuristic number of global outputs to achieve max occupancy with enough locals+upcasts for gemm
  #   ~2**10 should be enough if GROUP is used
  # 256 split maximum should be "negligible reduce" for low prod(reduce.shape), 8 split minimum.
  # split is moved to the end to provide maximum locality for the second phase reduce.

  # get expanded by rangeifying the UOp x
  indexed = x.index(*[UOp.range(s, i) if resolve(s>1) else 0 for i,s in enumerate(x.shape)])
  range_nums = [y.arg[0] for y in indexed.substitute({x.base:UOp(Ops.NOOP)}, extra_pm=pm_mops).ranges]
  is_expanded = [i not in range_nums for i in range(len(x.shape))]

  if not (split_candidates:=[(i,d) for i in range(reduce.arg[1])
                             for d in range(min(256,2**getenv("REDUCEOP_SPLIT_SIZE",22)//prod(reduce.shape)),8-1,-1)
                             if x.shape[i]%d==0 and not is_expanded[i]]): return None
  dim_to_split, divisor = split_candidates[0]
  splitted_shape = x.shape[:dim_to_split]+(divisor,)+(x.shape[dim_to_split]//divisor,)+x.shape[dim_to_split+1:]
  splitted = x.reshape(splitted_shape).permute(tuple([d for d in range(len(splitted_shape)) if d!=dim_to_split]+[dim_to_split]))
  if DEBUG >= 3: print(f"split {divisor}: {x.shape} -> {splitted.shape} -> {reduce.shape}")
  # reduce original axes, then split
  return splitted._rop(reduce.arg[0], tuple(range(reduce.arg[1]))).contiguous()._rop(reduce.arg[0], (len(reduce.shape),)).reshape(reduce.shape)

pm_gather_params = PatternMatcher([ (UPat(Ops.PARAM, name="p"), lambda ctx, p: ctx.append(p) if p.arg.slot >= 0 else None), ])
def resolve_function(c:UOp, allow_param_mismatch=True) -> UOp|None:
  if c.arg.precompile: return None
  params: list[UOp] = []
  graph_rewrite(c.src[0], pm_gather_params, bottom_up=True, ctx=params, name="gather params")
  params = sorted(params, key=lambda x: x.arg.slot)
  # the RETURNED inputs bind positionally to the output PARAMs, just like the args bind to the input PARAMs
  args = c.src[1:]

  # NOTE: this isn't really needed. it's okay if there's unused args in the function
  if not allow_param_mismatch:
    if [x.arg.slot for x in params] != list(range(len(params))): raise RuntimeError(f"params not in order: {[x.arg.slot for x in params]}")
    if len(params) != len(args): raise TypeError(f"expected {len(params)} args, got {len(args)}")

  # params have a flat storage size in the arg, the logical shape is a view (RESHAPE/SHRINK/UNSHARD) on top of it.
  # substitute args by their flat max-shaped storage view so the movement views on the params stay valid
  def flat_storage(a:UOp) -> tuple[int, UOp]:  # returns (size, view of a as flat max-shaped storage)
    shp = a.max_shard_shape if a.axis is not None and isinstance(a.device, tuple) else a.max_shape
    return (n:=prod(shp)), a if a.shape == (n,) else a.pad_to(shp).reshape((n,))
  dict_map = {x:args[x.arg.slot] for x in params}
  for i, (p, a) in enumerate(dict_map.items()):
    if p.arg.size is not None:
      n, flat = flat_storage(a)
      if p.arg.size != n: raise TypeError(f"arg {i} shape mismatch: expected size {p.arg.size}, got {a.shape}")
      dict_map[p] = flat
    elif a.shape != ():
      raise TypeError(f"arg {i} shape mismatch: expected scalar, got {a.shape}")
    if p.dtype != a.dtype: raise TypeError(f"arg {i} dtype mismatch: expected {p.dtype}, got {a.dtype}")
  return c.src[0].substitute(dict_map, walk=True)

# shape-changing bitcast
def expand_bitcast(bc:UOp) -> UOp|None:
  x = bc.src[0]
  if (ns:=bc.dtype.itemsize) == (os:=x.dtype.itemsize) or (isinstance(x.device, str) and x.device.startswith("DISK")): return None
  new_uint, tmp = to_dtype(f"uint{8*ns}"), x.bitcast(to_dtype(f"uint{8*os}"))
  if ns > os:
    tmp = tmp.reshape(x.shape[:-1] + (x.shape[-1]//(rate := ns//os), rate))
    parts = [tmp.shrink((None,)*(len(tmp.shape)-1) + ((i, i+1),)).cast(new_uint)<<8*i*os for i in range(rate)]
    return parts[0].usum(*parts[1:]).squeeze(-1).bitcast(bc.dtype)
  parts = [tmp>>8*i*ns for i in range(os//ns)]
  return parts[0].stack(*parts[1:], dim=-1).flatten(-2).cast(new_uint).bitcast(bc.dtype)

earliest_rewrites = mop_cleanup+PatternMatcher([
  # resolve calls with RETURNED inputs (inline the body)
  (UPat(Ops.CALL, name="c"), lambda c: resolve_function(c) if c.num_returned else None),

  # resolve AFTER on RETURNED (call outputs)
  (UPat(Ops.AFTER, src=(UPat(name="r"), UPat(Ops.SINK, name="t")), allow_any_len=True), resolve_returned_after),

  # resolve allreduce (must be bottom up)
  (UPat(Ops.ALLREDUCE, src=(UPat.var("buf"),), name="red"), create_allreduce_function),

  # split_reduceop
  (UPat(Ops.REDUCE, name="reduce", src=(UPat.var("x"),)), split_reduceop),

  # remove DETACH/CONTIGUOUS_BACKWARD (TODO: this is copied in allocations)
  (UPat((Ops.DETACH, Ops.CONTIGUOUS_BACKWARD), name="x"), lambda x: x.src[0]),

  # SINK only ever references the base
  (UPat(Ops.SINK, name="x"), lambda x: x.replace(src=tuple(y.unsharded_base for y in x.src))),

  # ** copy rules **

  # copy to same device is a no-op
  (UPat(Ops.COPY, src=(UPat.var("x"),), name="copy"), lambda x,copy: x if x.device == copy.device else None),

  # copy on reshape is reshape on copy
  (UPat(Ops.COPY, src=(UPat(Ops.RESHAPE, name="shp"),), name="cpy"), lambda shp,cpy: shp.src[0].copy_to_device(cpy.device).reshape(shp.shape)),

  # reshaping on STORE can be a NOOP
  (UPat(Ops.STORE, src=(UPat(Ops.RESHAPE, src=(UPat.var("dst",),), allow_any_len=True),
                        UPat(Ops.RESHAPE, src=(UPat.var("src",),), allow_any_len=True))),
   lambda dst,src: dst.store(src) if dst.shape == src.shape else None),

  # ** store rules **

  # fix store hazard (dest is in used in src) by adding contiguous: TestAssign.test_post_flipped_assignment
  (UPat(Ops.STORE, src=(UPat(name="target"), UPat(name="src"))), fix_store_hazard),

  # remove two STOREs that store the same thing to the same place: TestSchedule.test_dedup_Assign
  (UPat.var("buf").after(UPat.var("buf").store(UPat.var("src")), name="a1").after(UPat.var("a1").store(UPat.var("src"))), lambda buf,src,a1:a1),

  # store a buffer's own current contents back into itself: TestAssign.test_nested_after_contiguous_store_no_init
  (UPat.var("buf").after(UPat.var("buf").store(UPat.var("buf").after(UPat.var("buf").store(UPat.var("src")), name="a1"))), lambda buf,src,a1:a1),

  # move bitcast from store dest to source: TestAssign.test_assign_bitcast
  (UPat(Ops.STORE, src=(UPat(Ops.BITCAST, src=(UPat(name="target"),)), UPat(name="src"))),
   lambda target, src: target.store(src.bitcast(target.dtype))),

  (UPat(Ops.BITCAST, name="bc"), expand_bitcast),

  # ** size 0 **

  # reduce of size 0 is the identity element
  (UPat(Ops.REDUCE, name="reduce", src=(UPat.var("x"),)),
   lambda reduce,x: reduce.const_like(identity_element(reduce.arg[0], reduce.dtype)) if 0 in x.shape and 0 not in reduce.shape else None),
  # handle size 0
  (UPat(GroupOp.All-{Ops.SINK}, name="x"), lambda x: x.const_like(0).rtag(x.tag) if x._shape is not None and 0 in x.shape else None),

  # remove movement ops from SINK/AFTER. TODO: should be generic
  (UPat(Ops.SINK, name="s"), lambda s: s.replace(src=tuple(walk_mop(u) for u in s.src if u.op is not Ops.NOOP))),
  (UPat(Ops.AFTER, name="s"), lambda s: s.replace(src=(s.src[0],)+tuple(walk_mop(u) for u in s.src[1:] if u.op is not Ops.NOOP))),
])

def convert_copy_to_store(ctx, copy:UOp, existing_buf:UOp|None=None):
  input_src = copy.src[0]
  # if it's a COPY, we need to give the input buffer identity
  if not input_src.has_buffer_identity(after_ok=True) and copy.op is Ops.COPY: input_src = input_src.contiguous()
  input_src = input_src.flatten()
  if existing_buf is not None:
    # if the existing buffer is not a full buffer, we can't use it
    if not existing_buf.has_buffer_identity(after_ok=True): return None
    # if there's already a buffer, we just use it
    return existing_buf.flatten().store(input_src)
  # create the output buffer
  buf = UOp(Ops.BUFFER, arg=ParamArg(next(ctx), copy.dtype, size=prod(input_src.max_shape), device=copy.device))
  # reshape back to input
  return buf.reshape(input_src.max_shape).after(buf.store(input_src)).reshape(copy.shape)

pm_copy_to_store = PatternMatcher([
  (UPat(name="existing_buf").store(UPat(Ops.COPY, name="copy")), convert_copy_to_store),
  (UPat(Ops.COPY, name="copy"), convert_copy_to_store),
])

@rewrite_group(new_ctx=False)
def prepare_rangeify(sink:UOp) -> UOp:
  # prepare for rangeify
  tsink = graph_rewrite(sink, multi_pm, name="multi_pm")
  if OPENPILOT_HACKS: tsink = graph_rewrite(tsink, pm_fold_moved_after, ctx={}, name="fold moved afters")
  tsink = graph_rewrite(tsink, pm_mops+earliest_rewrites, bottom_up=True, name="earliest rewrites")
  tsink = graph_rewrite(tsink, pm_copy_to_store, ctx=itertools.count(0), bottom_up=True, name="convert copy to store")
  return tsink
