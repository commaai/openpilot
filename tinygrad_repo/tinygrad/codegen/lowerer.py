# the job of the lowerer is to do indexing
from dataclasses import dataclass
from typing import cast
from tinygrad.dtype import dtypes, PtrDType
from tinygrad.uop.ops import KernelInfo, UOp, Ops, PatternMatcher, UPat, sint_to_uop
from tinygrad.helpers import prod, partition, flatten

# ***** indexing *****

@dataclass
class IndexContext:
  idxs: list[UOp]
  ridxs: list[UOp]

def get_index(ast:UOp) -> IndexContext:
  ki = ast.arg if isinstance(ast.arg, KernelInfo) else KernelInfo()
  # NOTE: assumes the shape is <global dims> <local dims> <group_for_reduces> <reduces> <upcasts/unrolls>
  full_shape = ast.full_shape
  first_upcasted = len(full_shape)-ki.upcasted

  # all loops are RANGES
  idxs = [UOp(Ops.RANGE, dtypes.int, (sint_to_uop(g),), i) for i,g in enumerate(full_shape[:first_upcasted])]

  # upcast loops
  for i,g in enumerate(full_shape[first_upcasted:], start=first_upcasted):
    assert isinstance(g, int), "needs to be int to upcast/unroll"
    idxs.append(UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(g), tuple(range(g))),), ((i,g),)))

  # late indexes (group for reduce)
  # if there's no reduce, this is first_upcasted. assumes reduces are at the end
  first_reduce = min([first_upcasted]+flatten(x.axis_arg for x in ast.toposort() if x.op is Ops.REDUCE_AXIS))
  local_loads = [x for x in ast.toposort() if x.op is Ops.LOAD and x.src[0].base.op is Ops.DEFINE_LOCAL]
  # NOTE: sum up the reduced axes looking across all local loads, yields the number of grouped reduces
  group_for_reduces = sum([any(l.st_arg.shape[i]!=ast.src[0].st_arg.shape[i] for l in local_loads) for i in range(first_reduce,first_upcasted)])
  ridxs = idxs[:]
  for a in range(first_reduce, first_reduce+group_for_reduces):
    ridxs[a] = UOp(Ops.RANGE, dtypes.int, (sint_to_uop(full_shape[a]),), 1000+a)

  return IndexContext(idxs, ridxs)

# ***** lowering (given index) *****

def lower_reduce_axis(ctx: IndexContext, x: UOp):
  # NOTE: always using ridxs is fine here
  reduce_range, reduce_expand = partition([ctx.ridxs[i] for i in x.axis_arg], lambda y: y.op is Ops.RANGE)
  assert all(x.op is Ops.UNROLL for x in reduce_expand), f"not all UNROLLS in {reduce_expand} for {x.axis_arg}"
  ret = x.src[0]
  if len(contract_axis:=flatten(x.arg for x in reduce_expand)):
    ret = UOp(Ops.CONTRACT, x.dtype.vec(prod(x[1] for x in contract_axis)), (ret,), tuple(contract_axis))
  # REDUCE supports both "horizontal" reduction and range reduction. the horizontal elements are taken in the nearest group
  return UOp(Ops.REDUCE, x.dtype, (ret,)+tuple(reduce_range), x.arg[0])

def lower_load(ctx: IndexContext, x: UOp, buf: UOp):
  idx, valid = x.st_arg.to_indexed_uops(ctx.ridxs if buf.op is Ops.DEFINE_LOCAL else ctx.idxs)
  barrier = (UOp(Ops.BARRIER, dtypes.void, (x.src[1],)),) if buf.op is Ops.DEFINE_LOCAL else ()
  return UOp(Ops.LOAD, x.dtype, (buf.index(idx, valid),) + barrier)

def lower_store(ctx: IndexContext, x: UOp, buf: UOp):
  idx, valid = x.st_arg.to_indexed_uops(ctx.idxs)
  if not cast(PtrDType, buf.dtype).local:
    # NOTE: only store the local reduceop in the threads that are actually doing the reduce
    for oidx, ridx in zip(ctx.idxs, ctx.ridxs):
      if oidx is not ridx: valid = valid * oidx.eq(0)
  return UOp(Ops.STORE, dtypes.void, (buf.index(idx, valid), x.src[1]))

def lower_const(ctx:IndexContext, view:UOp, c:UOp):
  if all(x.mask is None for x in view.arg.views): return c
  _, valid = view.arg.to_indexed_uops(ctx.idxs)
  return valid.where(c, c.const_like(0))

pm_lowerer = PatternMatcher([
  # TODO: remove these hacks
  # hack for old style CONST(VIEW) (now it's just VIEW(CONST))
  (UPat((Ops.DEFINE_VAR, Ops.CONST), src=(UPat(Ops.VIEW, name="v"),), name="c"), lambda c,v: c.replace(src=()).view(v.arg)),
  # hack for old style VALID (now it's just VIEW(CONST))
  (UPat(Ops.VALID, src=(UPat(Ops.VIEW, name="v"),)).where(UPat.cvar("c"), UPat(Ops.CONST, arg=0)), lambda c,v: c.replace(src=()).view(v.arg)),

  # reduce/view_const
  (UPat(Ops.REDUCE_AXIS, name="x"), lower_reduce_axis),
  (UPat(Ops.VIEW, src=(UPat((Ops.CONST, Ops.DEFINE_VAR), name="c"),), name="view"), lower_const),
  # rewrite LOAD/STORE VIEW to LOAD/STORE with indexed
  (UPat(Ops.LOAD, src=(UPat.var("buf").view(),), allow_any_len=True, name="x"), lower_load),
  (UPat(Ops.STORE, src=(UPat.var("buf").view(),), allow_any_len=True, name="x"), lower_store),
])
