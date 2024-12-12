# the job of the lowerer is to do indexing
from __future__ import annotations
import functools, itertools, operator
from dataclasses import dataclass
from typing import List, Tuple, cast, Optional
from tinygrad.dtype import dtypes, PtrDType
from tinygrad.ops import KernelInfo, UOp, Ops, graph_rewrite, PatternMatcher, UPat, sint, identity_element, sint_to_uop
from tinygrad.renderer import Renderer
from tinygrad.helpers import all_int, prod, partition, flatten

# returns the axes to create new_shape if new_shape can be created by combining axis from old_shape
def get_contraction(old_shape:Tuple[sint, ...], new_shape:Tuple[sint, ...]) -> Optional[List[List[int]]]:
  acc_old, acc_new = list(itertools.accumulate(old_shape, operator.mul)), list(itertools.accumulate(new_shape, operator.mul))
  try: split = [acc_old.index(acc)+1 if acc != 1 else 0 for acc in acc_new]
  except ValueError: return None
  return [list(range(st,ed)) for st,ed in zip([0]+split[:-1], split[:-1]+[len(old_shape)])]

# ***** indexing *****

def _limit_dims(dims:Tuple[sint, ...], max_sizes:Tuple[int, ...]):
  # TODO: symbolic shape
  if not all_int(dims): return dims
  while len(dims) > len(max_sizes) or any(d > m for d,m in zip(dims, max_sizes)):
    for i,m in enumerate(max_sizes):
      if dims[i] * dims[i+1] <= m:
        dims = dims[:i] + (dims[i]*dims[i+1],) + dims[i+2:]
        break
    else: raise RuntimeError(f"cannot limit dim {dims=}, {max_sizes=}")
  return dims

def get_grouped_dims(prefix, dims:Tuple[sint, ...], max_sizes:Optional[Tuple[int, ...]], reverse=False) -> List[UOp]:
  if reverse: dims = dims[::-1]
  limited = _limit_dims(dims, max_sizes) if max_sizes is not None else dims
  ret = raw_idxs = [UOp(Ops.SPECIAL, dtypes.int, (), (f"{prefix}{i}", s)) for i,s in enumerate(limited)]
  if limited != dims:
    ret = []
    if (contraction:=get_contraction(dims, limited)) is None: raise AssertionError(f"get_contraction should not be None {dims=} {limited=}")
    for idx, contraction_group in zip(raw_idxs, contraction):
      for c in contraction_group[:-1]:
        ret.append(idx % dims[c])
        idx //= dims[c]
      ret.append(idx)
  return ret[::-1] if reverse else ret

@dataclass
class IndexContext:
  idxs: List[UOp]
  ridxs: List[UOp]
  acc_num: int = 0

def get_index(ast:UOp, opts:Renderer) -> IndexContext:
  ki = ast.arg if isinstance(ast.arg, KernelInfo) else KernelInfo()
  # NOTE: assumes the shape is <global dims> <local dims> <group_for_reduces> <reduces> <upcasts/unrolls>
  full_shape = ast.full_shape
  first_upcasted = len(full_shape)-ki.upcasted
  # if there's no reduce, this is first_upcasted. assumes reduces are at the end
  first_reduce = min([first_upcasted]+flatten(x.axis_arg for x in ast.toposort if x.op is Ops.REDUCE_AXIS))
  local_loads = [x for x in ast.toposort if x.op is Ops.LOAD and x.src[0].op is Ops.DEFINE_LOCAL]
  # NOTE: sum up the reduced axes looking across all local loads, yields the number of grouped reduces
  group_for_reduces = sum([any(l.st_arg.shape[i]!=ast.src[0].st_arg.shape[i] for l in local_loads) for i in range(first_reduce,first_upcasted)])
  global_dims = first_reduce-ki.local_dims

  if opts.has_local:
    if ki.dont_use_locals:
      assert ki.local_dims == 0, "can't use locals if there's no local dims"
      idxs = get_grouped_dims("idx", full_shape[:global_dims], opts.global_max, reverse=True)
    else:
      # define indexes for GPU-like execution
      idxs = get_grouped_dims("gidx", full_shape[:global_dims], opts.global_max, reverse=True) + \
             get_grouped_dims("lidx", full_shape[global_dims:first_reduce+group_for_reduces], opts.local_max)
  else:
    # all loops are RANGES
    idxs = [UOp(Ops.RANGE, dtypes.int, (sint_to_uop(0), sint_to_uop(g)), i) for i,g in enumerate(full_shape[:first_reduce])]

  # reduce loops
  idxs += [UOp(Ops.RANGE, dtypes.int, (sint_to_uop(0), sint_to_uop(g)), i)
    for i,g in enumerate(full_shape[first_reduce+group_for_reduces:first_upcasted], start=first_reduce+group_for_reduces)]

  # upcast loops
  for i,g in enumerate(full_shape[first_upcasted:], start=first_upcasted):
    assert isinstance(g, int), "needs to be int to upcast/unroll"
    idxs.append(UOp(Ops.EXPAND, dtypes.int, (UOp.const(dtypes.int.vec(g), tuple(range(g))),), ((i,g),)))

  # late indexes (group for reduce)
  ridxs = idxs[:]
  for a in range(first_reduce, first_reduce+group_for_reduces):
    ridxs[a] = UOp(Ops.RANGE, dtypes.int, (sint_to_uop(0), sint_to_uop(full_shape[a])), 1000+a)

  return IndexContext(idxs, ridxs)

# ***** lowering (given index) *****

def lower_reduce_axis(ctx: IndexContext, x: UOp):
  # NOTE: always using ridxs is fine here
  reduce_range, reduce_expand = partition([ctx.ridxs[i] for i in x.axis_arg], lambda y: y.op is Ops.RANGE)
  assert all(x.op is Ops.EXPAND for x in reduce_expand), f"not all EXPANDS in {reduce_expand} for {x.axis_arg}"
  alu_op: Ops = x.arg[0]
  ret = x.src[0]
  if len(contract_axis:=flatten(x.arg for x in reduce_expand)):
    ret = UOp(Ops.CONTRACT, x.dtype.vec(prod(x[1] for x in contract_axis)), (ret,), tuple(contract_axis))
    ret = functools.reduce(lambda x,y: x.alu(alu_op, y), [ret.gep(i) for i in range(ret.dtype.count)])
  if not len(reduce_range): return ret
  # create ACC and assign
  acc = UOp(Ops.DEFINE_ACC, x.dtype, (x.const_like(identity_element(alu_op, x.dtype.scalar())),) + tuple(reduce_range), (ctx.acc_num,))
  ctx.acc_num += 1
  return acc.assign(acc.alu(alu_op, ret))

def lower_load_store(ctx: IndexContext, x: UOp):
  idx, valid = x.st_arg.to_indexed_uops(ctx.ridxs if x.op is Ops.LOAD and x.src[0].op is Ops.DEFINE_LOCAL else ctx.idxs)
  # TODO: check has_valid in UPat, not here
  has_valid = valid.op is not Ops.CONST or valid.arg is not True
  buf = x.src[0]
  if x.op is Ops.LOAD:
    barrier = (UOp(Ops.BARRIER, dtypes.void, (x.src[2],)),) if x.src[0].op is Ops.DEFINE_LOCAL else ()
    return UOp(Ops.LOAD, x.dtype, (buf.index(idx, valid if has_valid else None),) + barrier)
  # NOTE: only store the local reduceop in the threads that are actually doing the reduce
  if cast(PtrDType, x.src[0].dtype).local and x.src[2].op is Ops.ASSIGN:
    reduce_input = x.src[2].src[1].src[1] if x.src[2].src[1].src[1] is not x.src[2].src[0] else x.src[2].src[1].src[0]
    store_back = reduce_input.op is Ops.LOAD and cast(PtrDType, reduce_input.src[0].dtype).local
  else: store_back = False
  # NOTE: If we're storing the reduced value back into each thread, need to zero-out the reduced axes
  if store_back: idx, _ = x.st_arg.to_indexed_uops([u.const_like(0) if u in x.src[2].src else u for u in ctx.idxs])
  if (not cast(PtrDType, x.src[0].dtype).local) or store_back:
    for oidx, ridx in zip(ctx.idxs, ctx.ridxs):
      if oidx is not ridx: valid = valid * oidx.eq(0)
    has_valid = valid.op is not Ops.CONST or valid.arg is not True
  return UOp(Ops.STORE, dtypes.void, (buf.index(idx, valid if has_valid else None), x.src[2]))

pm_lowerer = PatternMatcher([
  (UPat(Ops.REDUCE_AXIS, name="x"), lower_reduce_axis),
  (UPat(Ops.VALID, src=(UPat(Ops.VIEW),), name="x"), lambda ctx,x: x.st_arg.to_indexed_uops(ctx.idxs)[1]),
  # rewrite LOAD/STORE VIEW to LOAD/STORE with indexed
  (UPat((Ops.LOAD, Ops.STORE), src=(UPat(), UPat(Ops.VIEW)), allow_any_len=True, name="x"), lower_load_store),
])

def rewrite_shapetracker_with_index(ast:UOp, opts:Renderer) -> UOp: return graph_rewrite(ast, pm_lowerer, ctx=get_index(ast, opts))
