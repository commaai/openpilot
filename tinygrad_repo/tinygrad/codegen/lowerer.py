# the job of the lowerer is to do indexing
import functools, operator
from typing import cast
from dataclasses import dataclass
from tinygrad.dtype import dtypes, AddrSpace, PtrDType
from tinygrad.uop.ops import KernelInfo, UOp, Ops, PatternMatcher, UPat, sint_to_uop, AxisType, graph_rewrite
from tinygrad.helpers import prod, partition, flatten

# ***** indexing *****

@dataclass
class IndexContext:
  axis_types: tuple[AxisType, ...]
  idxs: list[UOp]
  start: int = 0

def shape_to_idx(s, axis_types, start=0):
  # indexes
  idxs = []
  for i, (s, at) in enumerate(zip(s, axis_types)):
    if at in (AxisType.UPCAST, AxisType.UNROLL):
      assert isinstance(s, int), "needs to be int to upcast/unroll"
      idxs.append(UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(s), tuple(range(s))),), ((i,s),), tag=1))
    else:
      # all others are RANGES
      idxs.append(UOp(Ops.RANGE, dtypes.int, (sint_to_uop(s),), start+i))
  return idxs

def get_index(ast:UOp) -> IndexContext:
  axis_types = ast.arg.axis_types if isinstance(ast.arg, KernelInfo) else ()
  if len(ast.full_shape) != len(axis_types): axis_types = (AxisType.LOOP,)*len(ast.full_shape)
  return IndexContext(axis_types, [], 0)

# ***** lowering (given index) *****

def subblock(ctx: IndexContext, full_new_idx: list[UOp], src: UOp):
  lc = IndexContext(ctx.axis_types, full_new_idx, ctx.start+1000)
  ctx.start = lc.start
  return graph_rewrite(src, pm_lowerer, lc, name="subblock", bottom_up=True)

def lower_reduce_axis(ctx: IndexContext, x: UOp):
  new_idxs = shape_to_idx(x.src[0].shape, ctx.axis_types, ctx.start)
  full_new_idx = list(ctx.idxs)
  for a in x.axis_arg: full_new_idx[a] = new_idxs[a]

  ret = subblock(ctx, full_new_idx, x.src[0])

  # NOTE: always using ridxs is fine here
  reduce_range, reduce_expand = partition([full_new_idx[i] for i in x.axis_arg], lambda y: y.op is Ops.RANGE)
  assert all(x.op is Ops.UNROLL for x in reduce_expand), f"not all UNROLLS in {reduce_expand} for {x.axis_arg}"
  if len(contract_axis:=flatten(x.arg for x in reduce_expand)):
    ret = UOp(Ops.CONTRACT, x.dtype.vec(prod(x[1] for x in contract_axis)), (ret,), tuple(contract_axis), tag=1)
  # REDUCE supports both "horizontal" reduction and range reduction. the horizontal elements are taken in the nearest group
  return UOp(Ops.REDUCE, x.dtype, (ret,)+tuple(reduce_range), x.arg[0])

def lower_store(ctx: IndexContext, x: UOp, buf: UOp):
  # TODO: reenable after REDUCE_AXIS is fixed
  #assert x.src[1].shape == x.src[0].shape, f"shape mismatch on store {x.src[1].shape} != {x.src[0].shape}"

  new_idxs = shape_to_idx(x.src[0].shape, ctx.axis_types, ctx.start)
  idx, valid = x.st_arg.to_indexed_uops(new_idxs)
  used_idxs = [x for x in UOp.sink(idx, valid).toposort() if x in new_idxs]
  real_new_idxs = []
  for i in range(len(x.src[0].shape)):
    if new_idxs[i] in used_idxs or len(ctx.idxs) <= i: real_new_idxs.append(new_idxs[i])
    else: real_new_idxs.append(ctx.idxs[i])

  stored = subblock(ctx, real_new_idxs, x.src[1])
  used_ranges = [x for x in used_idxs if x.op is Ops.RANGE]
  ret = buf.index(idx, valid).store(stored, *used_ranges)

  # insert BARRIER if we are ending a LOCAL, IF if we are ending a GROUP_REDUCE
  if cast(PtrDType, buf.dtype).addrspace == AddrSpace.LOCAL and \
      any(ctx.axis_types[x.arg%1000] in {AxisType.GROUP_REDUCE, AxisType.LOCAL} for x in used_ranges):
    ret = ret.barrier()
    range_gates = [x.eq(0) for x in used_ranges if ctx.axis_types[x.arg%1000] == AxisType.GROUP_REDUCE]
    if len(range_gates): ret = UOp(Ops.IF, src=(functools.reduce(operator.and_, range_gates), ret))
  return ret

def fixup_wmma(ctx:IndexContext, x:UOp):
  if x.tag is not None: return None
  new_idxs = shape_to_idx(x.src[0].shape, ctx.axis_types, ctx.start)
  full_new_idx = list(ctx.idxs)
  for a in x.arg[-1]: full_new_idx[a] = new_idxs[a]

  srcs = subblock(ctx, full_new_idx, UOp.sink(*x.src)).src

  # NOTE: this assumes these are expanded. which now shouldn't change anything
  new_x_arg_m2 = tuple([tuple([(full_new_idx[a].arg[0][0], sz) for a,sz in v]) for v in x.arg[-2]])
  new_x_arg_m1 = tuple([full_new_idx[a].arg[0][0] for a in x.arg[-1]])
  return x.replace(src=srcs, arg=x.arg[:-2]+(new_x_arg_m2, new_x_arg_m1), tag=1)

pm_lowerer = PatternMatcher([
  # TODO: remove these hacks
  # hack for old style CONST(VIEW) (now it's just VIEW(CONST))
  (UPat((Ops.DEFINE_VAR, Ops.CONST), src=(UPat(Ops.VIEW, name="v"),), name="c"), lambda c,v: c.replace(src=()).view(v.arg)),
  # hack for old style VALID (now it's just VIEW(CONST))
  (UPat(Ops.VALID, src=(UPat(Ops.VIEW, name="v"),)).where(UPat.cvar("c"), UPat(Ops.CONST, arg=0)), lambda c,v: c.replace(src=()).view(v.arg)),

  # consts and loads
  (UPat(Ops.VIEW, src=(UPat((Ops.CONST, Ops.DEFINE_VAR), name="c"),), name="view"),
   lambda ctx,view,c: c if all(x.mask is None for x in view.arg.views) else view.arg.to_indexed_uops(ctx.idxs)[1].where(c, c.const_like(0))),
  (UPat(Ops.LOAD, src=(UPat.var("buf").view(),), allow_any_len=True, name="x"),
   lambda ctx,buf,x: UOp(Ops.LOAD, x.dtype, (buf.index(*x.st_arg.to_indexed_uops(ctx.idxs)),)+x.src[1:])),

  # reduce/view_const
  (UPat(Ops.REDUCE_AXIS, name="x"), lower_reduce_axis),
  (UPat(Ops.STORE, src=(UPat.var("buf").view(),), allow_any_len=True, name="x"), lower_store),
  (UPat(Ops.WMMA, name="x"), fixup_wmma),

  # axis fixups for WMMA
  (UPat((Ops.CONTRACT, Ops.UNROLL), name="x"),
   lambda ctx,x: x.replace(tag=1, arg=tuple([(ctx.idxs[a].arg[0][0], sz) for a,sz in x.arg])) if x.tag is None else None),
])
