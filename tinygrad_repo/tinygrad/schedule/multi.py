from tinygrad.helpers import all_same, prod, getenv, ALLREDUCE_CAST
from tinygrad.uop.ops import Ops, UOp, PatternMatcher, UPat, GroupOp, AxisType, graph_rewrite, broadcast_axes, _broadcast_shape, sint_to_uop
from tinygrad.uop.ops import sint, ssimplify
from tinygrad.dtype import dtypes
from tinygrad.schedule.allreduce import handle_allreduce

# ***** multi rewrite MSELECT/MSTACK *****

def _apply_shrink(marg, s:UOp, i:int) -> UOp:
  new_arg = [tuple([x.substitute({drng[0]:drng[0].const_like(i)}) if isinstance(x, UOp) and
                    (drng:=[r for r in x.ranges if r.arg[-1] is AxisType.DEVICE]) else x for x in ss]) for ss in marg]
  return s._mop(Ops.SHRINK, tuple(new_arg))

def mstack_early_shrink(ms:UOp, shrink:UOp):
  ret:list[UOp] = []
  for i, x in enumerate(ms.src):
    if x.op is Ops.COPY:
      ret.append(_apply_shrink(shrink.marg, x.src[0], i).copy_to_device(x.device))
    else:
      ret.append(_apply_shrink(shrink.marg, x, i).contiguous())
  return ms.replace(src=tuple(ret))

def lower_broadcast_copy(c:UOp, x:UOp):
  if not (isinstance(c.device, tuple) and isinstance(x.device, str)): return None
  if (sx:=x.simplify()).device is None and sx.base.op is Ops.CONST: return UOp(Ops.MSTACK, src=(sx,)*len(c.device))
  return UOp(Ops.MSTACK, src=tuple(x.copy_to_device(d) for d in c.device))

replace_allreduce = PatternMatcher([
  # BROADCAST: explicitly expand broadcast copies and combine with MSTACK
  (UPat(Ops.COPY, name="c", src=(UPat(GroupOp.All-{Ops.CONST}, name="x"),)), lower_broadcast_copy),
  # COPY_TO_ONE: if copying from multidevice to one, MSELECT the first (TODO: a little from each?)
  (UPat(Ops.COPY, name="c", src=(UPat(GroupOp.All-{Ops.CONST}, name="x"),)), lambda c,x:
    x.mselect(0).copy_to_device(c.device) if isinstance(c.device, str) and isinstance(x.device, tuple) else None),
  # MSELECT on MSTACK is replaced with nothing
  (UPat(Ops.MSELECT, src=(UPat(Ops.MSTACK, name="mstack"),), name="ms"), lambda mstack, ms: mstack.src[ms.arg]),
  # move shrink before MSTACK
  (UPat(Ops.SHRINK, src=(UPat(Ops.MSTACK, name="ms"),), allow_any_len=True, name="shrink"), mstack_early_shrink),
  # move MSELECT before movement ops
  (UPat(Ops.MSELECT, src=(UPat(GroupOp.Movement, src=(UPat.var("s"),), allow_any_len=True, name="v"),), name="ms"),
   lambda s,v,ms: v.replace(src=(s.mselect(ms.arg),)+v.src[1:])),
])

_early_allreduce = PatternMatcher([
  (UPat(Ops.ALLREDUCE, src=(UPat.var("buf"),), name="red"), handle_allreduce),
])
if not getenv("LATE_ALLREDUCE", 1): replace_allreduce = _early_allreduce + replace_allreduce

# ***** multi functions *****

def shard_srcs(msrcs:tuple[UOp, ...], axis:int) -> list[UOp]:
  # normalize srcs to local shards on axis
  devices = [x.device for x in msrcs if x.device is not None]
  assert all_same(devices), f"all buffers must have the same device {devices}"
  # without devices the sharding range comes from the UNSHARD itself (e.g. a LOCAL thread range);
  # device shards range over the devices instead
  if len(devices): sharding_rng = UOp.range(len(devices[0]), -1, AxisType.DEVICE)
  else:
    sharding_rng = next((m.src[1] for m in msrcs if m.op is Ops.UNSHARD), None)
    assert sharding_rng is not None, "shard_srcs requires a device or a sharding range"

  out_shape = _broadcast_shape(*[x.shape for x in msrcs])
  srcs:list[UOp] = []
  for mlb in msrcs:
    src_axis = axis - (len(out_shape)-len(mlb.shape))
    if mlb.axis == src_axis:
      # same axis, just copy through
      srcs.append(mlb.src[0])
    else:
      # otherwise every shard gets the full copy, sharded iff this src has the axis (broadcast srcs stay whole)
      full = mlb if mlb.axis is None else copy_multi(mlb, mlb.device)
      srcs.append(full if axis in broadcast_axes(mlb.shape, out_shape) else full._shard(src_axis, sharding_rng))
  return srcs

def shard_subview(full:UOp, multi:UOp) -> UOp:
  """the sub-view of an unsharded full-shape value (shape == multi.shape) that belongs to this shard:
  _shard along every sharded axis (contiguous blocks, like the device path)."""
  assert tuple(full.shape) == tuple(multi.shape), f"shard sub-view shape mismatch {full.shape} != {multi.shape}"
  # an EXPAND of a scalar over the full shape is the same broadcast on every shard: re-expand over the shard shape
  if full.op is Ops.EXPAND and full.src[0].shape == (): return full.src[0].expand(multi.src[0].shape)
  for ax, rng in multi.sharding: full = full._shard(ax, rng)
  return full

def alu_multi(root:UOp):
  multis = [m for m in root.src if m.op is Ops.UNSHARD]
  if not multis: return None
  sharding = multis[0].sharding
  target = multis[0]
  def can_handle(m:UOp) -> bool:
    # same sharding (peel the UNSHARD), or a whole unsharded value of the full tile shape (takes its per-shard
    # sub-view), or a broadcast scalar
    if m.sharding: return m.sharding == sharding
    return m.shape == () or tuple(m.shape) == tuple(target.shape)
  if all(can_handle(m) for m in root.src):
    # every src either has the target sharding or is whole on every shard: run the alu per-shard
    srcs = [m.src[0] if m.op is Ops.UNSHARD else m if m.shape == () else shard_subview(m, target) for m in root.src]
    return srcs[0].alu(root.op, *srcs[1:]).unshard(target.arg, target.src[1:])
  # resharding: single-axis fallback via shard_srcs
  axis = root.axis
  assert axis is not None
  srcs = shard_srcs(root.src, axis)
  return srcs[0].alu(root.op, *srcs[1:]).unshard(axis, next(m.src[1] for m in root.src if m.op is Ops.UNSHARD))

def reduce_multi(root:UOp, multi:UOp):
  op, num_axes = root.arg
  sharding = multi.sharding
  reduced = [(ax, rng) for ax, rng in sharding if ax < num_axes]
  remaining = [(ax, rng) for ax, rng in sharding if ax >= num_axes]
  local = multi.src[0]._rop(op, tuple(range(num_axes)))
  if reduced:
    assert not remaining, f"partial allreduce not supported for multi-axis sharding {sharding}"
    # all sharded axes are reduced: full allreduce
    if ALLREDUCE_CAST and multi.src[0].op is Ops.CAST and multi.src[0].src[0].dtype in (dtypes.bfloat16, dtypes.half):
      orig_dtype = multi.src[0].src[0].dtype
      return local.cast(orig_dtype).allreduce(op, multi.device).cast(local.dtype)
    return local.allreduce(op, multi.device)
  # no sharded axes reduced: piecewise, keep all remaining sharding
  new_axes = tuple(ax - num_axes for ax, _ in remaining)
  new_rngs = tuple(rng for _, rng in remaining)
  return local.unshard(new_axes, new_rngs)

def reshape_multi(root:UOp, multi:UOp):
  if prod(multi.shape) != prod(new_shape:=root.marg): raise RuntimeError("reshape must maintain prod(shape)")
  # map every sharded axis through the reshape: the axis boundary must survive intact and stay divisible by its shard count
  arg_acc:list[sint] = [1]
  for s in new_shape: arg_acc.append(ssimplify(arg_acc[-1]*s))
  new_shardings = []
  for ax, rng in multi.sharding:
    count = int(rng.vmax)+1
    target = ssimplify(prod(multi.shape[:ax]))
    if target not in arg_acc: raise RuntimeError(f"reshape {multi.shape} -> {new_shape} moved items between shards")
    new_ax = len(arg_acc) - arg_acc[::-1].index(target) - 1
    if new_shape[new_ax] % count != 0: raise RuntimeError(f"reshape {multi.shape} -> {new_shape} moved items between shards")
    new_shardings.append((new_ax, rng))
  new_axs = {a for a, _ in new_shardings}
  new_shape = tuple(s//(int(rng.vmax)+1) if a in new_axs else s for a,s in enumerate(new_shape))
  return multi.src[0].reshape(new_shape).unshard(tuple(a for a,_ in new_shardings), tuple(r for _,r in new_shardings))

def expand_multi(root:UOp, multi:UOp):
  shift = len(root.marg)
  return multi.src[0]._mop(Ops.EXPAND, arg=root.marg) \
    .unshard(tuple(ax+shift for ax,_ in multi.sharding), tuple(r for _,r in multi.sharding))

def pad_multi(root:UOp, multi:UOp):
  for ax, _ in multi.sharding:
    assert root.marg[ax] == (0, multi.shape[ax]), f"padding not supported for {root.marg=}"
  counts = {a for a,_ in multi.sharding}
  local_pad = tuple((0, multi.src[0].shape[a]) if a in counts else s for a,s in enumerate(root.marg))
  return multi.src[0]._mop(Ops.PAD, local_pad).unshard(multi.arg, multi.src[1:])

def permute_multi(root:UOp, multi:UOp):
  # all permutes supported!
  return multi.src[0].permute(root.marg) \
    .unshard(tuple(root.marg.index(ax) for ax,_ in multi.sharding), tuple(r for _,r in multi.sharding))

def shrink_multi(root:UOp, multi:UOp):
  # resolve each sharded axis independently: a shrink to exactly this range's own shard resolves the UNSHARD along
  # that axis (e.g. a fragment indexed by its LOCAL thread range becomes that thread's REG shard, no copy needed)
  local_marg = list(root.marg)
  remaining = list(multi.sharding)
  for ax, rng in multi.sharding:
    shard_sz = multi.src[0].shape[ax]
    s, l = root.marg[ax]  # SHRINK marg is (start, length)
    if sint_to_uop(l).ssimplify() == shard_sz and (sint_to_uop(s)-rng*shard_sz).ssimplify() == 0:
      local_marg[ax] = (0, shard_sz)
      remaining.remove((ax, rng))
      continue
    part_bounds = tuple((i*shard_sz, shard_sz) for i in range(int(rng.vmax)+1))
    if (s, l) == (0, multi.shape[ax]): local_marg[ax] = (0, shard_sz)  # full axis stays sharded, shrink the other axes locally
    else:
      # NOTE: otherwise a shrink on the shard axis is only allowed on the legacy device path, selecting a single
      # partition (which is copied to all the devices and optimized out later)
      if len(multi.sharding) != 1 or not isinstance(multi.device, tuple) or (s, l) not in part_bounds:
        raise RuntimeError(f"shrinking not supported for {root.marg=}")
      non_shard_shrink = tuple((0, shard_sz) if i == ax else t for i, t in enumerate(root.marg))
      return multi.src[0].copy_to_device(multi.device, arg=part_bounds.index((s, l)))._mop(Ops.SHRINK, non_shard_shrink)
  val = multi.src[0]._mop(Ops.SHRINK, tuple(local_marg))
  return val if not remaining else val.unshard(tuple(a for a,_ in remaining), tuple(r for _,r in remaining))

def flip_multi(root:UOp, multi:UOp):
  for ax, _ in multi.sharding:
    if root.marg[ax]: raise RuntimeError(f"flipping not supported on sharded axis {ax}")
  return multi.src[0].flip([i for i,x in enumerate(root.marg) if x]).unshard(multi.arg, multi.src[1:])

def stack_multi(root:UOp):
  # STACK adds a leading axis: srcs are sharded one axis below the output
  multis = [m for m in root.src if m.op is Ops.UNSHARD]
  if not multis: return None
  sharding = multis[0].sharding
  if all(m.sharding == sharding for m in multis):
    srcs = [m.src[0] if m.op is Ops.UNSHARD else m for m in root.src]
    new_sharding = tuple((ax+1, rng) for ax, rng in sharding)
    return UOp(Ops.STACK, src=tuple(srcs)).unshard(tuple(a for a,_ in new_sharding), tuple(r for _,r in new_sharding))
  # resharding: single-axis fallback
  axis = root.axis
  assert axis is not None
  return UOp(Ops.STACK, src=tuple(shard_srcs(root.src, axis-1))).unshard(axis, next(m.src[1] for m in root.src if m.op is Ops.UNSHARD))

def index_multi(root:UOp, multi:UOp):
  # INDEX on UNSHARD: resolve each sharded axis into this range's own shard.
  # Two ownership patterns are supported:
  #   contiguous: idx = rng*shard_sz + local   (thread rng owns [rng*shard_sz, ...))
  #   strided:    idx = rng + ir*shard_sz      (thread rng owns {rng, rng+shard_sz, ...})
  idxs = list(root.src[1:])
  for ax, rng in multi.sharding:
    shard_sz = multi.src[0].shape[ax]
    local = (idxs[ax] - rng*shard_sz).simplify()
    if local.vmin >= 0 and local.vmax < shard_sz:
      idxs[ax] = local
      continue
    # strided ownership: idx ≡ rng (mod shard_sz), intra-shard position is (idx - rng) // shard_sz
    diff = (idxs[ax] - rng).simplify()
    if (mod:=(diff % shard_sz).simplify()).op is Ops.CONST and mod.val == 0:
      local = (diff // shard_sz).simplify()
      if local.vmin >= 0 and local.vmax < shard_sz:
        idxs[ax] = local
        continue
    raise RuntimeError(f"index_multi: cannot shard index {idxs[ax]} for UNSHARD axis {ax} with shard size {shard_sz}")
  return multi.src[0].index(*idxs)

def _shard_idx(rng:UOp, dev_idx:int) -> int:
  drngs = [r for r in rng.ranges if r.arg[-1] is AxisType.DEVICE]
  return 0 if not drngs else int(rng.substitute({drngs[0]: drngs[0].const_like(dev_idx)}).ssimplify())

def copy_multi(multi:UOp, device:str | tuple[str, ...]):
  sharding = multi.sharding
  if isinstance(device, str):
    # reconstruct by concatenating along each axis from last to first
    piece_info: list[tuple[tuple, UOp]] = []
    for i in range(len(multi.device)):
      idxs = tuple(_shard_idx(r, i) for _, r in sharding)
      piece_info.append((idxs, multi.src[0].mselect(i).copy_to_device(device)))
    for j in range(len(sharding) - 1, -1, -1):
      ax, rng = sharding[j]
      groups: dict[tuple, list[tuple[int, UOp]]] = {}
      for idxs, p in piece_info:
        key = idxs[:j] + idxs[j+1:]
        groups.setdefault(key, []).append((idxs[j], p))
      piece_info = []
      for key in sorted(groups):
        grp = sorted(groups[key], key=lambda x: x[0])
        piece_info.append((key, grp[0][1].cat(*[x[1] for x in grp[1:]], dim=ax)))
    return piece_info[0][1]
  # multi-device target: unshard all axes and allreduce
  val = multi.src[0]
  for ax, rng in sharding:
    bsz = val.shape[ax]
    val = val.pad(tuple((0,0) if a != ax else (bsz*rng, bsz*int(rng.vmax) - bsz*rng) for a in range(len(val.shape))))
  return val.allreduce(Ops.ADD, device)

def store_after_multi(dest:UOp, src:UOp): return dest.after(dest.store(src.src[0])).unshard(src.arg, src.src[1:])

def store_value_multi(dest:UOp, multi:UOp):
  # storing a sharded value into an unsharded dest: every shard stores into its own sub-view of the dest
  return shard_subview(dest, multi).store(multi.src[0])

def store_dest_multi(root:UOp, multi:UOp):
  # STORE with a sharded dest: every shard stores into its own shard of the dest.
  # the value is handled like in alu_multi: UNSHARD srcs peel, full-shape values take their per-shard sub-view
  # (scalars arrive EXPANDed to the full shape by UOp.store's const_like, so they sub-view like everything else)
  srcs = [multi.src[0]] + [x.src[0] if x.op is Ops.UNSHARD else shard_subview(x, multi) if tuple(x.shape) == tuple(multi.shape) else x
                           for x in root.src[1:]]
  return UOp(root.op, root.dtype, tuple(srcs), root.arg)

def passthrough_multi(root:UOp, multi:UOp):
  new_src = (multi.src[0],)+tuple(x.src[0] if x.op is Ops.UNSHARD else x for x in root.src[1:])
  return UOp(root.op, root.dtype, src=new_src, arg=root.arg).unshard(multi.arg, multi.src[1:])

def rewrite_into_function(call:UOp):
  if call.arg.precompile: return None
  new_body = graph_rewrite(call.src[0], multi_pm, name="subcall")
  new_args = tuple(a.src[0] if a.op is Ops.UNSHARD else a for a in call.src[1:])
  # after multi resolution, TUPLE elements may be UNSHARD — strip UNSHARD from body, create per-shard FUNCTION, wrap each GETTUPLE in its own UNSHARD
  assert new_body.op is Ops.TUPLE
  if any(s.op is Ops.UNSHARD for s in new_body.src):
    shard_call = call.replace(src=(UOp.maketuple(*[s.src[0] if s.op is Ops.UNSHARD else s for s in new_body.src]),)+new_args)
    return UOp.maketuple(*[shard_call.gettuple(i).unshard(s.arg, s.src[1:]) if s.op is Ops.UNSHARD else shard_call.gettuple(i)
                           for i, s in enumerate(new_body.src)])
  return call.replace(src=(new_body,)+new_args)

def param_to_multi(p:UOp):
  if p.axis is None: return None
  return UOp.param(p.arg.slot, p.dtype, p.shard_shape, p.device, p.arg.vmin_vmax, p.arg.multiple_of, p.arg.name, p.arg.addrspace).unshard(p.axis)

# NOTE: this is the same pattern as unrolled ranges
multi_pm = PatternMatcher([
  (UPat(Ops.PARAM, name="p"), param_to_multi),
  (UPat(GroupOp.ALU, name="root", custom_early_reject=set([Ops.UNSHARD])), alu_multi),
  (UPat(Ops.REDUCE, src=(UPat(Ops.UNSHARD, name="multi"), ), name="root"), reduce_multi),
  (UPat(Ops.RESHAPE, src=(UPat(Ops.UNSHARD, name="multi"), UPat()), name="root"), reshape_multi),
  (UPat(Ops.EXPAND, src=(UPat(Ops.UNSHARD, name="multi"), UPat()), name="root"), expand_multi),
  (UPat(Ops.PAD, src=(UPat(Ops.UNSHARD, name="multi"), UPat(), UPat()), name="root"), pad_multi),
  (UPat(Ops.SHRINK, src=(UPat(Ops.UNSHARD, name="multi"), UPat(), UPat()), name="root"), shrink_multi),
  (UPat(Ops.PERMUTE, src=(UPat(Ops.UNSHARD, name="multi"), ), name="root"), permute_multi),
  (UPat(Ops.FLIP, src=(UPat(Ops.UNSHARD, name="multi"), ), name="root"), flip_multi),
  (UPat(Ops.STACK, name="root", custom_early_reject=set([Ops.UNSHARD])), stack_multi),
  (UPat(Ops.INDEX, src=(UPat(Ops.UNSHARD, name="multi"),), name="root", allow_any_len=True), index_multi),
  (UPat(Ops.AFTER, src=(UPat(Ops.UNSHARD), UPat(Ops.STORE, src=(UPat(Ops.UNSHARD, name="dest"), UPat(Ops.UNSHARD, name="src"))))), store_after_multi),
  (UPat(Ops.COPY, src=(UPat(Ops.UNSHARD, name="multi"),), name="copy"), lambda multi,copy: copy_multi(multi, copy.arg)),
  (UPat(Ops.ALLREDUCE, src=(UPat(Ops.UNSHARD, name="multi"),), name="red"),
    lambda multi,red: multi.src[0].allreduce(*red.arg).unshard(multi.arg, multi.src[1:])),

  # resolve TUPLE+GETTUPLE (needed in multi)
  (UPat(Ops.GETTUPLE, src=(UPat(Ops.TUPLE, name="t"),), name="g"), lambda g,t: t.src[g.arg]),
  # GETTUPLE on UNSHARD: passthrough UNSHARD (e.g. when FUNCTION was replaced by UNSHARD(GETTUPLE(...)))
  (UPat(Ops.GETTUPLE, src=(UPat(Ops.UNSHARD, name="multi"),), name="g"),
    lambda g, multi: multi.src[0].gettuple(g.arg).unshard(multi.arg, multi.src[1:]) if multi.src[0].op in {Ops.FUNCTION, Ops.TUPLE} else multi),
  # rewrite into FUNCTION calls explicitly for UNSHARD (value-producing)
  (UPat(Ops.FUNCTION, name="call"), rewrite_into_function),
  (UPat((Ops.CALL, Ops.FUNCTION, Ops.AFTER), src=(UPat(Ops.UNSHARD, name="multi"), ), name="root", allow_any_len=True), passthrough_multi),
  # just strip the UNSHARD from non-value-producing CALLs (custom kernels, etc.) — FUNCTION is handled by rewrite_into_function
  (UPat(Ops.CALL, dtype=dtypes.void, name="root", custom_early_reject=set([Ops.UNSHARD])), lambda root:
    UOp(root.op, root.dtype, tuple(x.src[0] if x.op is Ops.UNSHARD else x for x in root.src), root.arg)),
  (UPat((Ops.CAST, Ops.BITCAST, Ops.CONTIGUOUS, Ops.DETACH, Ops.CONTIGUOUS_BACKWARD),
        src=(UPat(Ops.UNSHARD, name="multi"), ), name="root"), passthrough_multi),
  # STORE of a sharded value into an unsharded dest (e.g. a fragment into a full output tile)
  (UPat(Ops.STORE, src=(UPat.var("dest"), UPat(Ops.UNSHARD, name="multi"))), store_value_multi),
  # STORE into a sharded dest (e.g. the fragment init): every shard stores into its own shard
  (UPat(Ops.STORE, src=(UPat(Ops.UNSHARD, name="multi"), ), name="root", allow_any_len=True), store_dest_multi),
])+replace_allreduce
