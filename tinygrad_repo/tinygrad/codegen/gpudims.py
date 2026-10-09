import math
from tinygrad.uop.ops import UOp, Ops, sint, PatternMatcher, UPat, ssimplify, AxisType
from tinygrad.codegen.late.linearizer import pm_split_ends
from tinygrad.dtype import AddrSpace
from tinygrad.renderer import Renderer

def _dim_max(d:sint) -> int: return d if isinstance(d, int) else int(d.vmax)

def _group_dims(dims:tuple[sint, ...], max_sizes:tuple[int, ...]):
  while len(dims) > len(max_sizes) or any(_dim_max(d) > m for d,m in zip(dims, max_sizes)):
    for i,m in enumerate(max_sizes):
      if i < (len(dims)-1) and _dim_max(dims[i]) * _dim_max(dims[i+1]) <= m:
        dims = dims[:i] + (dims[i]*dims[i+1],) + dims[i+2:]
        break
    else: return None
  return dims

def _split_dims(dims, max_sizes):
  if all(d <= m for d,m in zip(dims, max_sizes)): return dims
  _dims = list(dims) + [1]*(3-len(dims))
  for i in range(len(_dims)):
    while _dims[i] > max_sizes[i]:
      div = next((d for d in range(2, math.ceil(math.sqrt(_dims[i])) + 1) if (_dims[i] % d) == 0), 1)
      if div == 1: raise RuntimeError(f"cannot limit dim {dims=}, {max_sizes=}")
      _dims[i], _dims[(i+1)%len(_dims)] = _dims[i]//div, _dims[(i+1)%len(_dims)]*div
  return tuple(_dims[:2] if _dims[2] == 1 else _dims)

def get_grouped_dims(prefix, dims:tuple[sint, ...], max_sizes:tuple[int, ...]|None, reverse=False) -> list[UOp]:
  if reverse: return get_grouped_dims(prefix, dims[::-1], max_sizes)[::-1]
  if max_sizes is None: limited = dims
  else:
    # try to group first: (a, b, c, d) -> (ab, c, d)
    limited = grouped if (grouped := _group_dims(dims, max_sizes)) else dims
    # check if grouping failed
    if len(limited) > len(max_sizes): raise RuntimeError(f"cannot limit dim {dims=}, {max_sizes=}")
    # try to split up dims: (a,) -> (b, c)
    if limited == dims: limited = _split_dims(dims, max_sizes)
  # Keep hardware axes as ranges through index lowering. The axis id is the hardware dimension.
  axis_type = AxisType.GLOBAL if prefix == "gidx" else AxisType.LOCAL
  raw_idxs = [UOp.range(s, i, axis_type) for i,s in enumerate(limited)]
  flat = sum(idx * math.prod(limited[i+1:]) for i,idx in enumerate(raw_idxs))
  return [ssimplify(flat // math.prod(dims[i+1:])) if i == 0 else ssimplify((flat // math.prod(dims[i+1:])) % dims[i]) for i in range(len(dims))]

def group_gpudims(ctx:Renderer, s:UOp):
  if s.arg is None: return None
  s_topo = list(s.toposort())
  if any(x.op is Ops.SPECIAL for x in s_topo): return None

  # get ranges
  all_ranges = {x.arg:x for x in s_topo if x.op is Ops.RANGE}

  # extract global/local dims
  global_dims = sorted([x.arg for x in all_ranges.values() if x.axis_type is AxisType.GLOBAL])
  # WARP maps to hardware dimension zero, independently of nesting order.
  local_dims = [x.arg for x in sorted((x for x in all_ranges.values() if x.axis_type in (AxisType.WARP, AxisType.LOCAL)),
                                    key=lambda x: (x.axis_type is not AxisType.WARP, x.axis_id))]
  if not global_dims and not local_dims: return None

  # get global and local shape
  global_shape = tuple(ssimplify(all_ranges[r].src[0]) for r in global_dims)
  local_shape = tuple(ssimplify(all_ranges[r].src[0]) for r in local_dims)

  # define indexes for GPU-like execution
  # if we got a WARP, set the local_max to it so it does not fold with other dims
  local_max = (local_shape[0],)+ctx.local_max[1:] if ctx.local_max is not None and local_dims and \
    all_ranges[local_dims[0]].axis_type is AxisType.WARP else ctx.local_max
  local_idxs = get_grouped_dims("lidx", local_shape, local_max)
  hw_local = [_dim_max(u.src[0]) for u in local_idxs if u.op is Ops.RANGE]
  global_max = ctx.global_max if ctx.global_prod_max is None else \
    tuple(min(gm, pm//l) for gm,pm,l in zip(ctx.global_max or ctx.global_prod_max, ctx.global_prod_max, hw_local+[1]*3))
  idxs = get_grouped_dims("gidx", global_shape, global_max, reverse=True) + local_idxs

  # apply to multiple ranges
  subs, masks = {}, {}
  for r in s_topo:
    # look for local INDEXes that are not used in the GLOBAL store, then add them as an INVALID
    if r.op is Ops.STORE and len((idx := r.src[0]).src) and idx.src[0].addrspace == AddrSpace.GLOBAL:
      missing_locals = [all_ranges[rng] for rng in local_dims if all_ranges[rng] not in idx.ranges]
      if len(missing_locals):
        assert len(idx.src) == 2, "index has 2 sources"
        mask: UOp = UOp.uprod(*[x.eq(0) for x in missing_locals])
        masks[idx] = idx.replace(src=(idx.src[0], idx.src[1].valid(mask)))
    if r.op is not Ops.RANGE: continue
    try:
      ii = (global_dims+local_dims).index(r.arg)
      subs[r] = idxs[ii]
    except ValueError: continue
  # Hardware ids may coincide with logical ids: replace ranges simultaneously, including those in the new masks.
  return s.substitute(masks).substitute(subs, walk=True)

pm_device_to_var = PatternMatcher([
  # the DEVICE axis is not a program axis, it's bound per device at launch. lower it to the _device_num variable (like SPECIAL for devices)
  (UPat(Ops.RANGE, name="r"),
   lambda r: UOp.variable("_device_num", 0, r.vmax, dtype=r.dtype) if r.axis_type is AxisType.DEVICE else None),
  # ENDs that closed a DEVICE range no longer close it
  (UPat(Ops.END, name="e"), lambda e: e.replace(src=(e.src[0],)+tuple(s for s in e.src[1:] if s.op is not Ops.PARAM))
   if any(s.op is Ops.PARAM and s.arg.name == '_device_num' for s in e.src[1:]) else None),
])

# Run once: grouping creates new GLOBAL/LOCAL ranges, which must not be grouped again.
pm_group_gpudims = PatternMatcher([(UPat(Ops.SINK, name="s"), group_gpudims)])+pm_device_to_var

pm_range_to_special = PatternMatcher([
  (UPat(Ops.RANGE, name="r"), lambda r: r.replace(op=Ops.SPECIAL, arg=f"{'g' if r.axis_type is AxisType.GLOBAL else 'l'}idx{r.axis_id[-1]}")
   if r.axis_type in (AxisType.GLOBAL, AxisType.LOCAL) else None),
])+pm_split_ends
