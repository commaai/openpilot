from typing import cast
import functools, itertools, operator
from tinygrad.helpers import all_same, all_int, prod, DEBUG, RING, getenv, unwrap
from tinygrad.uop.ops import Ops, UOp, sint, PatternMatcher, UPat, GroupOp, resolve
from tinygrad.device import Device

# *** allreduce implementation ***
def handle_allreduce_multirank(buf:UOp, red:UOp) -> UOp|None:
  if not isinstance(buf.device, tuple): return None

  # Group buffers
  groups: dict[int|None, list[UOp]] = {}
  for i,dev in enumerate(buf.device):
    groups.setdefault(Device[dev].group_id, []).append(buf.mselect(i))

  # Skip if only one group or if every group has only one buffer
  if len(groups) <= 1 or not any(len(g) > 1 for g in groups.values()): return None

  # Reduce inside each group
  inner = [UOp(Ops.MSTACK, buf.dtype, tuple(bufs)).allreduce(red.arg, (cast(str, bufs[0].device),)).mselect(0) for bufs in groups.values()]

  # Allreduce across groups
  outer = UOp(Ops.MSTACK, buf.dtype, tuple(inner)).allreduce(red.arg, tuple(buf.device for buf in inner))

  # Broadcast back to all devices in the group
  gid2bid = {Device[device].group_id: i for i,device in enumerate(outer.device)}
  return outer.mselect(gid2bid[Device[red.device].group_id]).copy_to_device(red.device) if not isinstance(red.device, tuple) else \
         UOp(Ops.MSTACK, buf.dtype, tuple(outer.mselect(gid2bid[Device[device].group_id]).copy_to_device(device) for device in red.device))

def handle_allreduce(buf:UOp, red:UOp) -> UOp|None:
  if not isinstance(buf.device, tuple): return None
  assert all_int(buf.shape), f"does not support symbolic shape {buf.shape}"
  n_lbs, shape, numel = len(buf.device), buf.shape, prod(buf.shape)
  # ring allreduce doesn't provide a benefit with only 2 nodes or where number of elements is less than 256k (empirically)
  # fallback to naive allreduce to save on kernel dispatch, chunking and reassembling chunks.
  use_ring = (RING >= 2 or (n_lbs > 2 and numel > getenv("RING_ALLREDUCE_THRESHOLD", 256_000) and RING >= 1))
  if DEBUG >= 2: print(f"{'RING ALLREDUCE' if use_ring else 'NAIVE ALLREDUCE'} {n_lbs}x{numel} | {buf.dtype}")

  # contiguous before we copy it
  buf = buf.contiguous()

  # copy to all devices. if you shrink later, that'll be handled
  if not use_ring: return functools.reduce(lambda x,y: x.alu(red.arg, y),
                                           [UOp(Ops.COPY, buf.dtype, (buf.mselect(i), red.src[1])) for i in range(len(buf.device))])

  # new ring reduce
  factor = next((f for f in [32, 16, 8, 4, 2] if numel % f == 0), 1)
  base, left = (numel // factor) // n_lbs, (numel // factor) % n_lbs
  chunk_sizes = [(base + 1) * factor] * left + [base * factor] * (n_lbs - left)
  chunks = list(itertools.pairwise(itertools.accumulate(chunk_sizes, initial=0)))

  # extract chunks and scatter-reduce
  reduced_chunks = []
  for i,(s,e) in enumerate(chunks):
    chunk = buf.reshape((numel,)).shrink(((s,e),))
    reduced_chunk = chunk
    for step in range(n_lbs-1):
      src, dest = (i+step)%n_lbs, (i+step+1)%n_lbs
      # copy the chunk from the src device to the dest (operating device), and select the chunk on the dest device
      reduced_chunk = reduced_chunk.copy_to_device(buf.device[dest], src if isinstance(reduced_chunk.device, tuple) else None) \
        .alu(red.arg, chunk.copy_to_device(buf.device[dest], dest))
    reduced_chunks.append(reduced_chunk)

  # allgather
  copied_chunks = []
  for i,c in enumerate(reduced_chunks):
    this_chunk = [None] * len(buf.device)
    this_chunk[(i+len(buf.device)-1)%n_lbs] = c
    for step in range(n_lbs-1):
      dest = (i+step)%n_lbs
      this_chunk[dest] = c = c.copy_to_device(buf.device[dest])
    copied_chunks.append(UOp(Ops.MSTACK, buf.dtype, tuple(cast(list[UOp], this_chunk))))

  # reassemble
  pads = [((s,numel-e),) for s,e in chunks]
  return functools.reduce(operator.add, [c.pad(pad) for pad,c in zip(pads, copied_chunks)]).reshape(shape)

# ***** multi rewrite MSELECT/MSTACK *****

def _replace_dnum(st, val):
  # replace dnum in ShapeTracker with literal const for this mselect
  if (dnums:=[x for x in st.vars() if x.op is Ops.DEFINE_VAR and x.arg[0] == '_device_num']):
    assert len(dnums) == 1, f"view must have exactly 0 or 1 dnum, got {dnums}"
    st = st.substitute({dnums[0]:dnums[0].const_like(val)})
  return st

def mstack_reorder_view(ms:UOp):
  args = [x.arg for x in ms.src]
  if not all_same(args) or len([x for x in args[0].vars() if x.arg[0] == '_device_num']) != 0: return None
  return UOp(Ops.MSTACK, ms.dtype, tuple(x.src[0] for x in ms.src)).view(args[0])

def mstack_early_shrink(view:UOp, ms:UOp):
  if resolve(prod(view.shape) >= prod(ms.shape)) or _replace_dnum(view.st, 0) == view.st: return None
  ret = []
  for i, x in enumerate(ms.src):
    new_view = _replace_dnum(view.st, i)
    if x.op is Ops.COPY:
      # if src device doesn't have a renderer, we have to view after the copy
      # TODO: a way to understand this
      if x.src[0].device in {"DISK", "NPY"}:
        ret.append(x.view(new_view))
      else:
        ret.append(x.src[0].view(new_view).copy_to_device(x.device))
    else:
      ret.append(x.view(new_view).contiguous())
  return ms.replace(src=tuple(ret))

replace_allreduce = PatternMatcher([
  (UPat(Ops.ALLREDUCE, src=(UPat.var("buf"), UPat()), name="red"), handle_allreduce_multirank),
  (UPat(Ops.ALLREDUCE, src=(UPat.var("buf"), UPat()), name="red"), handle_allreduce),
  # BROADCAST: explicitly expand broadcast copies and combine with MSTACK
  (UPat(Ops.COPY, name="c", src=(UPat(GroupOp.All-{Ops.CONST}, name="x"), UPat(Ops.DEVICE))), lambda c,x:
    UOp(Ops.MSTACK, c.dtype, tuple(x.copy_to_device(d) for d in c.device)) if isinstance(c.device, tuple) and isinstance(x.device, str) else None),
  # COPY_TO_ONE: if copying from multidevice to one, MSELECT the first (TODO: a little from each?)
  (UPat(Ops.COPY, name="c", src=(UPat(GroupOp.All-{Ops.CONST}, name="x"), UPat(Ops.DEVICE))), lambda c,x:
    x.mselect(0).copy_to_device(c.device) if isinstance(c.device, str) and isinstance(x.device, tuple) else None),
  # MSELECT on MSTACK is replaced with nothing
  (UPat(Ops.MSELECT, src=(UPat(Ops.MSTACK, name="mstack"),), name="ms"), lambda mstack, ms: mstack.src[ms.arg]),
  # MSELECT must select a base, if there are views apply them after selecting the base
  (UPat(Ops.MSELECT, src=(UPat(Ops.VIEW, src=(UPat.var("base"),), name="view"),), name="ms"), lambda ms, view, base:
    base.mselect(ms.arg).view(_replace_dnum(unwrap(view.st), ms.arg))),
  # move view through MSTACK
  (UPat(Ops.MSTACK, src=UPat(Ops.VIEW), name="ms"), mstack_reorder_view),
  # move shrink before MSTACK
  (UPat(Ops.VIEW, src=(UPat(Ops.MSTACK, name="ms"),), name="view"), mstack_early_shrink),
])

# ***** multi functions *****

def alu_multi(root:UOp):
  msrcs = root.src
  assert all_same([x.device for x in msrcs]), f"all buffers must have the same device {[x.device for x in msrcs]}"
  axis = root.axis
  assert axis is not None

  srcs = []
  for mlb in msrcs:
    if mlb.axis == axis:
      # same axis, just copy through
      assert mlb.op is Ops.MULTI
      srcs.append(mlb.src[0])
    elif mlb.axis is None:
      # no axis, shard it
      assert mlb.op is not Ops.MULTI
      srcs.append(mlb._shard(axis))
    else:
      # axis mismatch, unshard it, send it to all devices, and shard it correctly
      assert mlb.op is Ops.MULTI
      srcs.append(mlb.src[0]._unshard(mlb.axis).allreduce(Ops.ADD, mlb.device)._shard(axis))
  return srcs[0].alu(root.op, *srcs[1:]).multi(axis)

def reduce_multi(root:UOp, multi:UOp):
  op, axis = root.arg
  if multi.axis is not None and multi.axis in axis:
    # all-reduce on sharded axes
    return multi.src[0].r(op, axis).allreduce(op, multi.device)
  # reduce on non sharded axes, piecewise is fine. if axis is None this is also correct
  return multi.src[0].r(op, axis).multi(axis=multi.axis)

def _shape_to_single_shard(axis, shape:tuple[sint, ...], lb:UOp) -> tuple[sint, ...]:
  return tuple(lb.shape[axis] if a == axis else s for a,s in enumerate(shape))

def reshape_multi(root:UOp, multi:UOp):
  arg = root.arg
  if (new_axis:=root.axis) is None: return multi.src[0].reshape(arg).multi(new_axis)
  assert prod(multi.shape) == prod(arg), "reshape must maintain prod(shape)"
  assert prod(multi.src[0].shape[multi.axis:])%prod(arg[new_axis+1:]) == 0, f"reshape cannot move items between shards {multi.shape} -> {root.arg=}"
  new_shape_axis = prod(multi.src[0].shape[multi.axis:]) // prod(arg[new_axis+1:])
  return multi.src[0].reshape(tuple(s if a!=new_axis else new_shape_axis for a,s in enumerate(arg))).multi(new_axis)

def expand_multi(root:UOp, multi:UOp):
  # NOTE: this assert isn't needed, sharded axis can have dim 1
  assert multi.axis is None or root.arg[multi.axis] == multi.shape[multi.axis], f"expand not supported on sharded axis {root.arg=}"
  return multi.src[0].expand(_shape_to_single_shard(multi.axis, root.arg, multi.src[0])).multi(multi.axis)

def pad_multi(root:UOp, multi:UOp):
  assert multi.axis is None or root.arg[multi.axis] == (0,0), f"padding not supported for {root.arg=}"
  return multi.src[0].pad(root.arg).multi(multi.axis)

def permute_multi(root:UOp, multi:UOp):
  # all permutes supported!
  return multi.src[0].permute(root.arg).multi(root.axis)

def shrink_multi(root:UOp, multi:UOp):
  assert multi.axis is None or root.arg[multi.axis] == (0, multi.shape[multi.axis]) or root.arg[multi.axis] in multi.bounds, \
    f"shrinking not supported for {root.arg=}"
  if multi.axis is not None and root.arg[multi.axis] in multi.bounds and root.arg[multi.axis] != (0, multi.shape[multi.axis]):
    assert all(root.arg[i] == (0, s) or i == multi.axis for i,s in enumerate(multi.shape)), \
      "cannot shrink sharded and non-sharded axis at the same time"
    # NOTE: shrink on the shard axis is only allowed when result is a single partition, denoted by the new real
    # we just copy it to all the devices, no real. this will be optimized out later
    return multi.src[0].copy_to_device(multi.device, arg=multi.bounds.index(root.arg[multi.axis]))
  return multi.src[0].shrink(tuple((0, multi.src[0].shape[multi.axis]) if a == multi.axis else s for a,s in enumerate(root.arg))).multi(multi.axis)

def flip_multi(root:UOp, multi:UOp):
  assert multi.axis is None or not root.arg[multi.axis], "flipping not supported on sharded axis"
  return multi.src[0].flip(root.arg).multi(multi.axis)

# from multiple devices -> one
def copy_multi(multi:UOp, device:UOp):
  assert multi.axis is not None, "all multi ops have axis"
  return multi.src[0]._unshard(multi.axis).allreduce(Ops.ADD, device)

def assign_multi(dest:UOp, src:UOp):
  if dest.axis != src.axis: raise RuntimeError(f"axis must match in assign {dest.axis} != {src.axis}")
  return dest.src[0].assign(src.src[0]).multi(src.axis)

def passthrough_multi(root:UOp, multi:UOp):
  return root.replace(src=(multi.src[0],)).multi(multi.axis)

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
  (UPat(Ops.ALLREDUCE, src=(UPat(Ops.MULTI, name="multi"), UPat(Ops.DEVICE, name="device")), name="red"),
    lambda multi,device,red: multi.src[0].allreduce(red.arg, device).multi(axis=multi.axis)),
  (UPat((Ops.CAST, Ops.BITCAST, Ops.CONTIGUOUS, Ops.DETACH, Ops.CONTIGUOUS_BACKWARD, Ops.FUSE),
        src=(UPat(Ops.MULTI, name="multi"), ), name="root"), passthrough_multi),
])+replace_allreduce
