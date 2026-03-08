from typing import cast
import functools, itertools, operator
from tinygrad.helpers import all_same, all_int, prod, DEBUG, RING, ALL2ALL, getenv
from tinygrad.uop.ops import Ops, UOp, sint, PatternMatcher, UPat, GroupOp, graph_rewrite_map, graph_rewrite
from tinygrad.device import Device

# *** allreduce implementation ***
def handle_allreduce_multirank(buf:UOp, red:UOp) -> UOp|None:
  if not isinstance(buf.device, tuple): return None

  # Group buffers
  groups: dict[int|None, list[UOp]] = {}
  for i,dev in enumerate(buf.device):
    groups.setdefault(Device[dev].group_id, []).append(buf.mselect(i))

  # Put reduce leader of each group first
  reduce_leaders = set(getenv("REDUCE_LEADERS", "").split(","))
  groups = {gid: sorted(bufs, key=lambda x: (x.device not in reduce_leaders, x.device)) for gid,bufs in groups.items()}

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
  use_all2all = (ALL2ALL >= 2 or (n_lbs > 2 and numel > getenv("RING_ALLREDUCE_THRESHOLD", 256_000) and ALL2ALL >= 1))
  use_ring = not use_all2all and (RING >= 2 or (n_lbs > 2 and numel > getenv("RING_ALLREDUCE_THRESHOLD", 256_000) and RING >= 1))
  if DEBUG >= 2: print(f"{'ALL2ALL' if use_all2all else 'RING' if use_ring else 'NAIVE'} ALLREDUCE {n_lbs}x{numel} | {buf.dtype}")

  # contiguous before we copy it
  buf = buf.contiguous()

  # naive: copy to all devices. if you shrink later, that'll be handled
  if not use_ring and not use_all2all:
    return functools.reduce(lambda x,y: x.alu(red.arg, y), [UOp(Ops.COPY, buf.dtype, (buf.mselect(i), red.src[1])) for i in range(n_lbs)])

  # chunk data into n_lbs pieces
  factor = next((f for f in [32, 16, 8, 4, 2] if numel % f == 0), 1)
  base, left = (numel // factor) // n_lbs, (numel // factor) % n_lbs
  chunks = list(itertools.pairwise(itertools.accumulate([(base + 1) * factor] * left + [base * factor] * (n_lbs - left), initial=0)))

  # reduce-scatter
  reduced_chunks = []
  for i,(s,e) in enumerate(chunks):
    if use_all2all:
      chunks_on_i = [buf.mselect(j).reshape((numel,)).shrink(((s,e),)).copy_to_device(buf.device[i]) for j in range(n_lbs)]
      reduced_chunks.append(functools.reduce(lambda x,y: x.alu(red.arg, y), chunks_on_i))
    else:
      chunk, reduced = buf.reshape((numel,)).shrink(((s,e),)), buf.reshape((numel,)).shrink(((s,e),))
      for step in range(n_lbs-1):
        src, dest = (i+step)%n_lbs, (i+step+1)%n_lbs
        cp = reduced.copy_to_device(buf.device[dest], src if isinstance(reduced.device, tuple) else None)
        reduced = cp.alu(red.arg, chunk.copy_to_device(buf.device[dest], dest))
      reduced_chunks.append(reduced)

  # allgather
  copied_chunks = []
  for i,rc in enumerate(reduced_chunks):
    if isinstance(red.src[1].arg, str): copied_chunks.append(rc.copy_to_device(red.src[1].arg))
    elif use_all2all: copied_chunks.append(UOp(Ops.MSTACK, buf.dtype, tuple(rc.copy_to_device(buf.device[j]) for j in range(n_lbs))))
    else:
      this_chunk: list[UOp|None] = [None] * n_lbs
      this_chunk[(i+n_lbs-1)%n_lbs] = rc
      for step in range(n_lbs-1):
        this_chunk[(i+step)%n_lbs] = rc = rc.copy_to_device(buf.device[(i+step)%n_lbs])
      copied_chunks.append(UOp(Ops.MSTACK, buf.dtype, tuple(cast(list[UOp], this_chunk))))

  # reassemble
  pads = [((s,numel-e),) for s,e in chunks]
  return functools.reduce(operator.add, [c.pad(pad) for pad,c in zip(pads, copied_chunks)]).reshape(shape)

# ***** multi rewrite MSELECT/MSTACK *****

def mstack_early_shrink(ms:UOp, shrink:UOp):
  ret:list[UOp] = []
  def apply_shrink(s:UOp, i:int) -> UOp:
    new_arg = [tuple([x.substitute({dvar[0]:dvar[0].const_like(i)}) if isinstance(x, UOp) and
                      (dvar:=[v for v in x.vars() if v.op is Ops.DEFINE_VAR and v.arg[0]=='_device_num']) else x for x in ss]) for ss in shrink.marg]
    return s.shrink(tuple(new_arg))
  for i, x in enumerate(ms.src):
    if x.op is Ops.COPY:
      ret.append(apply_shrink(x.src[0], i).copy_to_device(x.device))
    else:
      ret.append(apply_shrink(x, i).contiguous())
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
  # move shrink before MSTACK
  (UPat(Ops.SHRINK, src=(UPat(Ops.MSTACK, name="ms"),), allow_any_len=True, name="shrink"), mstack_early_shrink),
  # move MSELECT before movement ops
  (UPat(Ops.MSELECT, src=(UPat(GroupOp.Movement, src=(UPat.var("s"),), allow_any_len=True, name="v"),), name="ms"),
   lambda s,v,ms: v.replace(src=(s.mselect(ms.arg),)+v.src[1:])),
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
  arg = root.marg
  if (new_axis:=root.axis) is None: return multi.src[0].reshape(arg).multi(new_axis)
  assert prod(multi.shape) == prod(arg), "reshape must maintain prod(shape)"
  assert prod(multi.src[0].shape[multi.axis:])%prod(arg[new_axis+1:]) == 0, f"reshape cannot move items between shards {multi.shape} -> {arg=}"
  new_shape_axis = prod(multi.src[0].shape[multi.axis:]) // prod(arg[new_axis+1:])
  return multi.src[0].reshape(tuple(s if a!=new_axis else new_shape_axis for a,s in enumerate(arg))).multi(new_axis)

def expand_multi(root:UOp, multi:UOp):
  # NOTE: this assert isn't needed, sharded axis can have dim 1
  assert multi.axis is None or root.marg[multi.axis] == multi.shape[multi.axis], f"expand not supported on sharded axis {root.marg=}"
  return multi.src[0].expand(_shape_to_single_shard(multi.axis, root.marg, multi.src[0])).multi(multi.axis)

def pad_multi(root:UOp, multi:UOp):
  assert multi.axis is None or root.marg[multi.axis] == (0,0), f"padding not supported for {root.marg=}"
  return multi.src[0].pad(root.marg).multi(multi.axis)

def permute_multi(root:UOp, multi:UOp):
  # all permutes supported!
  return multi.src[0].permute(root.marg).multi(root.axis)

def shrink_multi(root:UOp, multi:UOp):
  assert multi.axis is None or root.marg[multi.axis] == (0, multi.shape[multi.axis]) or root.marg[multi.axis] in multi.bounds, \
    f"shrinking not supported for {root.marg=}"
  if multi.axis is not None and root.marg[multi.axis] in multi.bounds and root.marg[multi.axis] != (0, multi.shape[multi.axis]):
    assert all(root.marg[i] == (0, s) or i == multi.axis for i,s in enumerate(multi.shape)), \
      "cannot shrink sharded and non-sharded axis at the same time"
    # NOTE: shrink on the shard axis is only allowed when result is a single partition, denoted by the new real
    # we just copy it to all the devices, no real. this will be optimized out later
    return multi.src[0].copy_to_device(multi.device, arg=multi.bounds.index(root.marg[multi.axis]))
  return multi.src[0].shrink(tuple((0, multi.src[0].shape[multi.axis]) if a == multi.axis else s for a,s in enumerate(root.marg))).multi(multi.axis)

def flip_multi(root:UOp, multi:UOp):
  assert multi.axis is None or not root.marg[multi.axis], "flipping not supported on sharded axis"
  return multi.src[0].flip([i for i,x in enumerate(root.marg) if x]).multi(multi.axis)

# from multiple devices -> one
def copy_multi(multi:UOp, device:UOp):
  assert multi.axis is not None, "all multi ops have axis"
  return multi.src[0]._unshard(multi.axis).allreduce(Ops.ADD, device)

def assign_multi(dest:UOp, src:UOp):
  if dest.axis != src.axis: raise RuntimeError(f"axis must match in assign {dest.axis} != {src.axis}")
  return dest.src[0].assign(src.src[0]).multi(src.axis)

def passthrough_multi(root:UOp, multi:UOp):
  return UOp(root.op, root.dtype, (multi.src[0],)+tuple(x.src[0] if x.op is Ops.MULTI else x for x in root.src[1:]), root.arg).multi(multi.axis)

# NOTE: this is the same pattern as Ops.UNROLL
multi_pm = PatternMatcher([
  (UPat(GroupOp.ALU, name="root", custom_early_reject=set([Ops.MULTI])), alu_multi),
  (UPat(Ops.REDUCE_AXIS, src=(UPat(Ops.MULTI, name="multi"), ), name="root"), reduce_multi),
  (UPat(Ops.RESHAPE, src=(UPat(Ops.MULTI, name="multi"), UPat()), name="root"), reshape_multi),
  (UPat(Ops.EXPAND, src=(UPat(Ops.MULTI, name="multi"), UPat()), name="root"), expand_multi),
  (UPat(Ops.PAD, src=(UPat(Ops.MULTI, name="multi"), UPat(), UPat()), name="root"), pad_multi),
  (UPat(Ops.SHRINK, src=(UPat(Ops.MULTI, name="multi"), UPat(), UPat()), name="root"), shrink_multi),
  (UPat(Ops.PERMUTE, src=(UPat(Ops.MULTI, name="multi"), ), name="root"), permute_multi),
  (UPat(Ops.FLIP, src=(UPat(Ops.MULTI, name="multi"), ), name="root"), flip_multi),
  (UPat(Ops.ASSIGN, src=(UPat(Ops.MULTI, name="dest"), UPat(Ops.MULTI, name="src"))), assign_multi),
  (UPat(Ops.COPY, src=(UPat(Ops.MULTI, name="multi"), UPat(Ops.DEVICE, name="device"))), copy_multi),
  (UPat(Ops.ALLREDUCE, src=(UPat(Ops.MULTI, name="multi"), UPat(Ops.DEVICE, name="device")), name="red"),
    lambda multi,device,red: multi.src[0].allreduce(red.arg, device).multi(axis=multi.axis)),
  (UPat(Ops.CALL, src=(UPat(Ops.MULTI, name="multi"), ), name="root", allow_any_len=True), passthrough_multi),
  (UPat((Ops.CAST, Ops.BITCAST, Ops.CONTIGUOUS, Ops.DETACH, Ops.CONTIGUOUS_BACKWARD),
        src=(UPat(Ops.MULTI, name="multi"), ), name="root"), passthrough_multi),
  # multi supports custom kernels with CUSTOM_KERNEL + AFTER
  (UPat(Ops.CUSTOM_KERNEL, src=UPat((Ops.MULTI, Ops.CONTIGUOUS)), name="ck"),
    lambda ck: ck.replace(src=tuple(m.src[0] if m.op is Ops.MULTI else m for m in ck.src))),
  (UPat(Ops.AFTER, src=(UPat(Ops.MULTI, name="multi"), UPat(Ops.CUSTOM_KERNEL)), name="a"),
    lambda multi,a: a.replace(src=(multi.src[0],)+a.src[1:]).multi(multi.axis))
])+replace_allreduce

def get_multi_map(big_sink:UOp) -> dict[UOp, UOp]:
  if getenv("VIZ"): graph_rewrite(big_sink, PatternMatcher([]), name="View Multi AST")
  ret = graph_rewrite_map(big_sink, multi_pm, name="multi_pm")
  if getenv("VIZ"): graph_rewrite(ret[big_sink], PatternMatcher([]), name="View Post Multi AST")
  return ret
