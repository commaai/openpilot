import functools, itertools, operator
from tinygrad.helpers import all_same, all_int, dedup, prod, DEBUG, RING, getenv
from tinygrad.ops import Ops, UOp, sint

def all_reduce(bop: Ops, lbs: list[UOp]) -> list[UOp]:
  assert all_int(lbs[0].shape), f"does not support symbolic shape {lbs[0].shape}"
  assert all_same([lb.shape[0] for lb in lbs]), "allreduce with uneven shards is undefined"
  n_lbs, shape, numel = len(lbs), lbs[0].shape, prod(lbs[0].shape)
  # ring allreduce doesn't provide a benefit with only 2 nodes or where number of elements is less than 256k (empirically)
  # fallback to naive allreduce to save on kernel dispatch, chunking and reassembling chunks.
  use_ring = (RING >= 2 or (n_lbs > 2 and numel > getenv("RING_ALLREDUCE_THRESHOLD", 256_000) and RING >= 1))
  if DEBUG >= 2: print(f"{'RING ALLREDUCE' if use_ring else 'NAIVE ALLREDUCE'} {n_lbs}x{numel} | {lbs[0].dtype}")
  if not use_ring: return [functools.reduce(lambda x,y: x.alu(bop, y), [x.copy_to_device(lb.device) for x in lbs]) for lb in lbs]

  factor = next((f for f in [32, 16, 8, 4, 2] if numel % f == 0), 1)
  base, left = (numel // factor) // n_lbs, (numel // factor) % n_lbs
  chunk_sizes = [(base + 1) * factor] * left + [base * factor] * (n_lbs - left)
  chunks = list(itertools.pairwise(itertools.accumulate(chunk_sizes, initial=0)))
  chunked = [[lb.reshape((numel,)).shrink(((s,e),)) for s,e in chunks] for lb in lbs]

  # scatter-reduce
  for step in range(n_lbs-1):
    for i in range(len(chunks)):
      src, dest = (i+step)%n_lbs, (i+step+1)%n_lbs
      chunked[dest][i] = chunked[dest][i].alu(bop, chunked[src][i].copy_to_device(chunked[dest][i].device))

  # allgather
  for step in range(n_lbs-1):
    for i in range(len(chunks)):
      src, dest = (i+step-1)%n_lbs, (i+step)%n_lbs
      chunked[dest][i] = chunked[src][i].copy_to_device(chunked[dest][i].device)

  # assemble chunks back
  pads = [((s,numel-e),) for s,e in chunks]
  return [functools.reduce(operator.add, [c.pad(pad) for pad,c in zip(pads,lb_c)]).reshape(shape) for lb_c in chunked]

def to_sharded(lbs:list[UOp], axis:int, bounds: tuple[tuple[int, int], ...]) -> list[UOp]:
  if lbs[0].shape[axis] % len(lbs) != 0: raise RuntimeError(f"multi axis uneven: {lbs[0].shape=} {axis=} {len(lbs)=}, bounds={bounds}")
  return [lb.shrink(tuple((0,s) if a != axis else bound for a,s in enumerate(lb.shape))) for i, (bound, lb) in enumerate(zip(bounds, lbs))]

# ***** multi functions *****

from tinygrad.ops import PatternMatcher, UPat, GroupOp, graph_rewrite_map, track_rewrites

def alu_multi(root:UOp):
  msrcs = root.src
  assert all(x.op is Ops.MULTI for x in msrcs), f"all buffers must be MultiLazyBuffer {[x.op for x in msrcs]}"
  assert all_same([x.device for x in msrcs]), f"all buffers must have the same device {[x.device for x in msrcs]}"

  axis = root.axis
  bounds = dedup([x.bounds for x in root.src if x.axis == axis])[-1] if axis is not None else None
  srcs:list[list[UOp]] = []
  not_all_real = not all(all(mlb.real) for mlb in msrcs)
  new_real = tuple(all(transposed) for transposed in zip(*[mlb.real for mlb in msrcs])) if not_all_real else msrcs[0].real
  for mlb in msrcs:
    if (mlb.axis == axis and (mlb.axis is None or mlb.bounds == bounds)) or not_all_real: srcs.append(list(mlb.src))
    else:
      assert axis is not None and bounds is not None
      if mlb.axis is None: srcs.append(to_sharded(list(mlb.src), axis, bounds))
      else: srcs.append(to_sharded([mlb.copy_to_device(lb.device) for lb in mlb.src], axis, bounds))
  new_lbs = [lsrcs[0].alu(root.op, *lsrcs[1:]) for lsrcs in zip(*srcs)]
  new_lbs = [x if r else x.const_like(0) for r,x in zip(new_real, new_lbs)] # TODO: is this needed?
  return UOp.multi(*new_lbs, axis=axis, real=new_real)

def reduce_multi(root:UOp, multi:UOp):
  op, axis = root.arg
  if multi.axis is not None and multi.axis in axis:
    # all-reduce on sharded axes
    reduced_parts = [(x if r else x.const_like(0)).r(op, axis) for x,r in zip(multi.src, multi.real)]
    # if all partitions are real, do all_reduce
    if all(multi.real): return UOp.multi(*all_reduce(op, reduced_parts), axis=root.axis)
    # only one partition is real, keep it
    return UOp.multi(*reduced_parts, axis=root.axis, real=multi.real)
  # reduce on non sharded axes, piecewise is fine. if axis is None this is also correct
  return UOp.multi(*[x.r(op, axis) for x in multi.src], axis=root.axis, real=multi.real)

def _shape_to_single_shard(axis, shape:tuple[sint, ...], lb:UOp) -> tuple[sint, ...]:
  return tuple(lb.shape[axis] if a == axis else s for a,s in enumerate(shape))

def reshape_multi(root:UOp, multi:UOp):
  arg = root.arg
  if (new_axis:=root.axis) is None: return UOp.multi(*[x.reshape(arg) for x in multi.src], axis=new_axis, real=multi.real)
  assert prod(multi.shape) == prod(arg), "reshape must maintain prod(shape)"
  assert all(prod(lb.shape[multi.axis:])%prod(arg[new_axis+1:])==0 for lb in multi.src), \
    f"reshape cannot move items between shards {multi.shape} -> {root.arg=}"
  lbs = [x.reshape(tuple(s if a!=new_axis else prod(x.shape[multi.axis:])//prod(arg[new_axis+1:]) for a,s in enumerate(arg))) for x in multi.src]
  return UOp.multi(*lbs, axis=new_axis, real=multi.real)

def expand_multi(root:UOp, multi:UOp):
  # NOTE: this assert isn't needed, sharded axis can have dim 1
  assert multi.axis is None or root.arg[multi.axis] == multi.shape[multi.axis], f"expand not supported on sharded axis {root.arg=}"
  return UOp.multi(*[x.expand(_shape_to_single_shard(multi.axis, root.arg, x)) for x in multi.src], axis=multi.axis, real=multi.real)

def pad_multi(root:UOp, multi:UOp):
  assert multi.axis is None or root.arg[multi.axis] == (0,0) or not all(multi.real), f"padding not supported for {root.arg=}"
  # pad on shard axis -> fill others with zeros and set real to all True
  if multi.axis is not None and root.arg[multi.axis] != (0,0):
    # pad back to whole axis, remove real mask
    assert all(root.arg[i] == (0, 0) for i in range(len(multi.shape)) if i != multi.axis), "cannot pad sharded and non-sharded axis at the same time"
    dim, bound = sum(lb.shape[multi.axis] for lb in multi.src), multi.bounds[multi.real.index(True)]
    assert root.arg[multi.axis] == (bound[0], dim-bound[1]), "can only pad to whole axis"
    return UOp.multi(*[x if r else x.const_like(0) for x,r in zip(multi.src, multi.real)], axis=multi.axis)
  return UOp.multi(*[x.pad(root.arg) for x in multi.src], axis=multi.axis, real=multi.real)

def permute_multi(root:UOp, multi:UOp):
  # all permutes supported!
  return UOp.multi(*[x.permute(root.arg) for x in multi.src], axis=root.axis, real=multi.real)

def shrink_multi(root:UOp, multi:UOp):
  assert multi.axis is None or root.arg[multi.axis] == (0, multi.shape[multi.axis]) or root.arg[multi.axis] in multi.bounds, \
    f"shrinking not supported for {root.arg=}"
  if multi.axis is not None and root.arg[multi.axis] in multi.bounds and root.arg[multi.axis] != (0, multi.shape[multi.axis]):
    assert all(root.arg[i] == (0, s) or i == multi.axis for i,s in enumerate(multi.shape)), \
      "cannot shrink sharded and non-sharded axis at the same time"
    # NOTE: shrink on the shard axis is only allowed when result is a single partition, denoted by the new real
    idx = multi.bounds.index(root.arg[multi.axis])
    # zero out other lbs to not create lb reference
    return UOp.multi(*[lb if i==idx else lb.const_like(0) for i,lb in enumerate(multi.src)],
                      axis=multi.axis, real=tuple(i==idx for i in range(len(multi.src))))
  return UOp.multi(*[x.shrink(tuple((0, x.shape[multi.axis]) if a == multi.axis else s for a,s in enumerate(root.arg))) for x in multi.src],
                   axis=multi.axis, real=multi.real)

def flip_multi(root:UOp, multi:UOp):
  assert multi.axis is None or not root.arg[multi.axis], "flipping not supported on sharded axis"
  return UOp.multi(*[x.flip(root.arg) for x in multi.src], axis=multi.axis, real=multi.real)

def copy_multi(multi:UOp, device:UOp):
  # if we already have a copy on the device, return that
  if multi.axis is None: return next((lb for lb in multi.real_lbs if lb.device == device.arg), multi.real_lbs[0].copy_to_device(device))
  # copy lbs to device, pad to final shape, and sum
  llbs:list[UOp] = []
  for lb,real,(start,end) in zip(multi.src, multi.real, multi.bounds):
    if not real: continue
    pad_arg = tuple((0,0) if a != multi.axis else (start, multi.bounds[-1][1]-end) for a in range(len(lb.shape)))
    llbs.append(lb.copy_to_device(device).pad(pad_arg))
  return functools.reduce(operator.add, llbs)

def assign_multi(dest:UOp, src:UOp):
  assert dest.axis == src.axis, f"axis must match in assign {dest.axis} != {src.axis}"
  return UOp.multi(*[x.assign(y) for x,y in zip(dest.src, src.src)], axis=src.axis, real=src.real)

def passthrough_multi(root:UOp, multi:UOp): return UOp.multi(*[root.replace(src=(m,)) for m in multi.src], axis=multi.axis, real=multi.real)

# NOTE: this is the same pattern as Ops.UNROLL
multi_pm = PatternMatcher([
  (UPat(GroupOp.ALU, name="root", custom_early_reject=set([Ops.MULTI])), alu_multi),
  (UPat(Ops.REDUCE_AXIS, src=(UPat(Ops.MULTI, name="multi"), ), name="root"), reduce_multi),
  (UPat(Ops.RESHAPE, src=(UPat(Ops.MULTI, name="multi"), ), name="root"), reshape_multi),
  (UPat(Ops.EXPAND, src=(UPat(Ops.MULTI, name="multi"), ), name="root"), expand_multi),
  (UPat(Ops.PAD, src=(UPat(Ops.MULTI, name="multi"), ), name="root"), pad_multi),
  (UPat(Ops.PERMUTE, src=(UPat(Ops.MULTI, name="multi"), ), name="root"), permute_multi),
  (UPat(Ops.SHRINK, src=(UPat(Ops.MULTI, name="multi"), ), name="root"), shrink_multi),
  (UPat(Ops.FLIP, src=(UPat(Ops.MULTI, name="multi"), ), name="root"), flip_multi),
  (UPat(Ops.ASSIGN, src=(UPat(Ops.MULTI, name="dest"), UPat(Ops.MULTI, name="src"))), assign_multi),
  (UPat(Ops.COPY, src=(UPat(Ops.MULTI, name="multi"), UPat(Ops.DEVICE, name="device"))), copy_multi),
  (UPat((Ops.CAST, Ops.BITCAST, Ops.CONTIGUOUS, Ops.DETACH, Ops.CONTIGUOUS_BACKWARD, Ops.FUSE),
        src=(UPat(Ops.MULTI, name="multi"), ), name="root"), passthrough_multi),
])

@track_rewrites(named=True)
def get_multi_map(big_sink:UOp) -> dict[UOp, UOp]: return {k:v for k,v in graph_rewrite_map(big_sink, multi_pm).items() if k is not v}
