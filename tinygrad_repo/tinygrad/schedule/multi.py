from tinygrad.helpers import all_same, prod, getenv, ALLREDUCE_CAST
from tinygrad.uop.ops import Ops, UOp, PatternMatcher, UPat, GroupOp, graph_rewrite
from tinygrad.dtype import dtypes
from tinygrad.schedule.allreduce import handle_allreduce

# ***** multi rewrite MSELECT/MSTACK *****

def mstack_early_shrink(ms:UOp, shrink:UOp):
  ret:list[UOp] = []
  def apply_shrink(s:UOp, i:int) -> UOp:
    new_arg = [tuple([x.substitute({dvar[0]:dvar[0].const_like(i)}) if isinstance(x, UOp) and
                      (dvar:=[v for v in x.variables() if v.expr=='_device_num']) else x for x in ss]) for ss in shrink.marg]
    return s._mop(Ops.SHRINK, tuple(new_arg))
  for i, x in enumerate(ms.src):
    if x.op is Ops.COPY:
      ret.append(apply_shrink(x.src[0], i).copy_to_device(x.device))
    else:
      ret.append(apply_shrink(x, i).contiguous())
  return ms.replace(src=tuple(ret))

replace_allreduce = PatternMatcher([
  # BROADCAST: explicitly expand broadcast copies and combine with MSTACK
  (UPat(Ops.COPY, name="c", src=(UPat(GroupOp.All-{Ops.CONST}, name="x"),)), lambda c,x:
    UOp(Ops.MSTACK, src=tuple(x.copy_to_device(d) for d in c.device)) if isinstance(c.device, tuple) and isinstance(x.device, str) else None),
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
  dcount = len(devices[0])

  srcs:list[UOp] = []
  for mlb in msrcs:
    if mlb.axis is None:
      # no axis, shard it
      assert mlb.op is not Ops.MULTI
      srcs.append(mlb._shard(axis, dcount))
    else:
      assert mlb.op is Ops.MULTI
      if mlb.axis == axis:
        # same axis, just copy through
        srcs.append(mlb.src[0])
      else:
        # axis mismatch, copy to all devices, and shard it correctly
        srcs.append(copy_multi(mlb, mlb.device)._shard(axis, dcount))
  return srcs

def alu_multi(root:UOp):
  axis = root.axis
  assert axis is not None
  srcs = shard_srcs(root.src, axis)
  return srcs[0].alu(root.op, *srcs[1:]).multi(axis)

def reduce_multi(root:UOp, multi:UOp):
  op, num_axes = root.arg
  if multi.axis is not None and multi.axis < num_axes:
    local = multi.src[0]._rop(op, tuple(range(num_axes)))
    # allreduce in pre-cast dtype when sum_acc_dtype promoted from bf16/half
    if ALLREDUCE_CAST and multi.src[0].op is Ops.CAST and multi.src[0].src[0].dtype in (dtypes.bfloat16, dtypes.half):
      orig_dtype = multi.src[0].src[0].dtype
      return local.cast(orig_dtype).allreduce(op, multi.device).cast(local.dtype)
    return local.allreduce(op, multi.device)
  # reduce on non sharded axes, piecewise is fine. if axis is None this is also correct
  new_axis = multi.axis - num_axes if multi.axis is not None else None
  return multi.src[0]._rop(op, tuple(range(num_axes))).multi(axis=new_axis)

def reshape_multi(root:UOp, multi:UOp):
  if prod(multi.shape) != prod(new_shape:=root.marg): raise RuntimeError("reshape must maintain prod(shape)")
  if (new_axis:=root.axis) is not None: new_shape = tuple(s//len(multi.device) if a==new_axis else s for a,s in enumerate(new_shape))
  return multi.src[0].reshape(new_shape).multi(new_axis)

def expand_multi(root:UOp, multi:UOp):
  new_axis = None if multi.axis is None else multi.axis + len(root.marg)
  return multi.src[0]._mop(Ops.EXPAND, arg=root.marg).multi(new_axis)

def pad_multi(root:UOp, multi:UOp):
  assert multi.axis is None or root.marg[multi.axis] == (0, multi.shape[multi.axis]), f"padding not supported for {root.marg=}"
  local_pad = tuple((0, multi.src[0].shape[multi.axis]) if a == multi.axis else s for a,s in enumerate(root.marg))
  return multi.src[0]._mop(Ops.PAD, local_pad).multi(multi.axis)

def permute_multi(root:UOp, multi:UOp):
  # all permutes supported!
  return multi.src[0].permute(root.marg).multi(root.axis)

def shrink_multi(root:UOp, multi:UOp):
  shard_bounds = tuple((s,e-s) for s,e in multi.bounds) if multi.axis is not None else ()
  assert multi.axis is None or root.marg[multi.axis] == (0, multi.shape[multi.axis]) or root.marg[multi.axis] in shard_bounds, \
    f"shrinking not supported for {root.marg=}"
  if multi.axis is not None and root.marg[multi.axis] in shard_bounds and root.marg[multi.axis] != (0, multi.shape[multi.axis]):
    # NOTE: shrink on the shard axis is only allowed when result is a single partition, denoted by the new real
    # we just copy it to all the devices, no real. this will be optimized out later
    non_shard_shrink = tuple((0, multi.src[0].shape[i]) if i == multi.axis else s for i, s in enumerate(root.marg))
    return multi.src[0].copy_to_device(multi.device, arg=shard_bounds.index(root.marg[multi.axis]))._mop(Ops.SHRINK, non_shard_shrink)
  local_shrink = tuple((0, multi.src[0].shape[multi.axis]) if a == multi.axis else s for a,s in enumerate(root.marg))
  return multi.src[0]._mop(Ops.SHRINK, local_shrink).multi(multi.axis)

def flip_multi(root:UOp, multi:UOp):
  assert multi.axis is None or not root.marg[multi.axis], "flipping not supported on sharded axis"
  return multi.src[0].flip([i for i,x in enumerate(root.marg) if x]).multi(multi.axis)

def stack_multi(root:UOp):
  # STACK adds a leading axis: srcs are sharded one axis below the output
  axis = root.axis
  assert axis is not None
  return UOp(Ops.STACK, src=tuple(shard_srcs(root.src, axis-1))).multi(axis)

def copy_multi(multi:UOp, device:str | tuple[str, ...]):
  assert multi.axis is not None, "all multi ops have axis"
  if isinstance(device, str):
    pieces = [multi.src[0].mselect(i).copy_to_device(device) for i in range(len(multi.device))]
    return pieces[0].cat(*pieces[1:], dim=multi.axis)
  return multi.src[0]._unshard(multi.axis).allreduce(Ops.ADD, device)

def store_after_multi(dest:UOp, src:UOp): return dest.after(dest.store(src.src[0])).multi(src.axis)

def passthrough_multi(root:UOp, multi:UOp):
  new_src = (multi.src[0],)+tuple(x.src[0] if x.op is Ops.MULTI else x for x in root.src[1:])
  return UOp(root.op, root.dtype, src=new_src, arg=root.arg).multi(multi.axis)

def rewrite_into_function(call:UOp):
  if call.arg.precompile: return None
  new_body = graph_rewrite(call.src[0], multi_pm, name="subcall")
  new_args = tuple(a.src[0] if a.op is Ops.MULTI else a for a in call.src[1:])
  # after multi resolution, TUPLE elements may be MULTI — strip MULTI from body, create per-shard FUNCTION, wrap each GETTUPLE in its own MULTI
  assert new_body.op is Ops.TUPLE
  if any(s.op is Ops.MULTI for s in new_body.src):
    shard_call = call.replace(src=(UOp.maketuple(*[s.src[0] if s.op is Ops.MULTI else s for s in new_body.src]),)+new_args)
    return UOp.maketuple(*[shard_call.gettuple(i).multi(s.axis) if s.op is Ops.MULTI else shard_call.gettuple(i) for i, s in enumerate(new_body.src)])
  return call.replace(src=(new_body,)+new_args)

def param_to_multi(p:UOp):
  if p.axis is None: return None
  return UOp.param(p.arg.slot, p.dtype, p.shard_shape, p.device, p.arg.vmin_vmax, p.arg.multiple_of, p.arg.name, p.arg.addrspace).multi(p.axis)

# NOTE: this is the same pattern as unrolled ranges
multi_pm = PatternMatcher([
  (UPat(Ops.PARAM, name="p"), param_to_multi),
  (UPat(GroupOp.ALU, name="root", custom_early_reject=set([Ops.MULTI])), alu_multi),
  (UPat(Ops.REDUCE, src=(UPat(Ops.MULTI, name="multi"), ), name="root"), reduce_multi),
  (UPat(Ops.RESHAPE, src=(UPat(Ops.MULTI, name="multi"), UPat()), name="root"), reshape_multi),
  (UPat(Ops.EXPAND, src=(UPat(Ops.MULTI, name="multi"), UPat()), name="root"), expand_multi),
  (UPat(Ops.PAD, src=(UPat(Ops.MULTI, name="multi"), UPat(), UPat()), name="root"), pad_multi),
  (UPat(Ops.SHRINK, src=(UPat(Ops.MULTI, name="multi"), UPat(), UPat()), name="root"), shrink_multi),
  (UPat(Ops.PERMUTE, src=(UPat(Ops.MULTI, name="multi"), ), name="root"), permute_multi),
  (UPat(Ops.FLIP, src=(UPat(Ops.MULTI, name="multi"), ), name="root"), flip_multi),
  (UPat(Ops.STACK, name="root", custom_early_reject=set([Ops.MULTI])), stack_multi),
  (UPat(Ops.AFTER, src=(UPat(Ops.MULTI), UPat(Ops.STORE, src=(UPat(Ops.MULTI, name="dest"), UPat(Ops.MULTI, name="src"))))), store_after_multi),
  (UPat(Ops.COPY, src=(UPat(Ops.MULTI, name="multi"),), name="copy"), lambda multi,copy: copy_multi(multi, copy.arg)),
  (UPat(Ops.ALLREDUCE, src=(UPat(Ops.MULTI, name="multi"),), name="red"),
    lambda multi,red: multi.src[0].allreduce(*red.arg).multi(axis=multi.axis)),

  # resolve TUPLE+GETTUPLE (needed in multi)
  (UPat(Ops.GETTUPLE, src=(UPat(Ops.TUPLE, name="t"),), name="g"), lambda g,t: t.src[g.arg]),
  # GETTUPLE on MULTI: passthrough MULTI (e.g. when FUNCTION was replaced by MULTI(GETTUPLE(...)))
  (UPat(Ops.GETTUPLE, src=(UPat(Ops.MULTI, name="multi"),), name="g"),
    lambda g, multi: multi.src[0].gettuple(g.arg).multi(multi.axis) if multi.src[0].op in {Ops.FUNCTION, Ops.TUPLE}
    else multi),
  # rewrite into FUNCTION calls explicitly for MULTI (value-producing)
  (UPat(Ops.FUNCTION, name="call"), rewrite_into_function),
  (UPat((Ops.CALL, Ops.FUNCTION, Ops.AFTER), src=(UPat(Ops.MULTI, name="multi"), ), name="root", allow_any_len=True), passthrough_multi),
  # just strip the MULTI from non-value-producing CALLs (custom kernels, etc.) — FUNCTION is handled by rewrite_into_function
  (UPat(Ops.CALL, dtype=dtypes.void, name="root", custom_early_reject=set([Ops.MULTI])), lambda root:
    UOp(root.op, root.dtype, tuple(x.src[0] if x.op is Ops.MULTI else x for x in root.src), root.arg)),
  (UPat((Ops.CAST, Ops.BITCAST, Ops.CONTIGUOUS, Ops.DETACH, Ops.CONTIGUOUS_BACKWARD),
        src=(UPat(Ops.MULTI, name="multi"), ), name="root"), passthrough_multi),
  # remove MULTI from STORE
  (UPat(Ops.STORE, src=(UPat(Ops.MULTI, name="multi"), ), name="root", allow_any_len=True),
    lambda root,multi: UOp(root.op, root.dtype, (multi.src[0],)+tuple(x.src[0] if x.op is Ops.MULTI else x for x in root.src[1:]), root.arg)),
])+replace_allreduce
