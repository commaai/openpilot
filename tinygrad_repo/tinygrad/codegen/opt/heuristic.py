import itertools
from tinygrad.codegen.opt import Opt, OptOps, KernelOptError
from tinygrad.helpers import getenv, DEBUG, prod, TC_OPT, TC_SELECT, TC_MIN_GLOBALS, USE_TC, IMAGE
from tinygrad.uop.ops import Ops, resolve, AxisType
from tinygrad.codegen.late.coalesce import image_valid_dims
from tinygrad.codegen.opt.postrange import Scheduler

def hand_coded_optimizations(k:Scheduler) -> Scheduler:
  # first try the tensor cores
  """ Attempts to apply a tensor core optimization to the kernel. If one exists and applies properly, return true, otherwise return false.
  Tensor cores are optimized instructions that matrix multiply-accumulate across a wave of threads: D(M, N) = A(M, K) * B(K, N) + C(M, N).

  ContextVars:
  USE_TC -- controls how tensor cores are applied (default 1)
    0: will disable any tensor core matching
    1: enable tensor cores
    2: apply tensor core shape but don't use UOp.WMMA
  TC_SELECT -- specifies which tensor core(s) to use for optimization (default -1)
    -1: iterates through all available tensor cores in order and uses the first one that matches the requirements (dims and dtypes)
    [0-N]: uses only the n'th tensor core available; useful for search
  TC_OPT -- controls which kinds of kernels may be eligible for tensor cores application (default 2 during BEAM, 0 otherwise)
    0: applies to only kernels with a single reduce axis and direct Ops.LOAD into Ops.MUL
    1: allows kernels with multiple reduce axes and also multiplication of Ops.CAST'd buffers
    2: allows kernels with M, N, K axes that are not multiples of the tensor core dimensions by applying padding those axes as needed
  TC_MIN_GLOBALS -- do not upcast N when it would drop the specified global count
  """
  # NOTE: unless TC_OPT is > 0, we only trigger tensor cores if there's only one reduce axis
  if USE_TC > 0 and (len(k.axes_of(AxisType.GROUP_REDUCE, AxisType.REDUCE)) == 1 or (TC_OPT.value >= 1)):
    for axis in range(3):
      tk = k.copy()
      # check TC first and apply hand-coded opts if successful
      try: rngs = tk.apply_opt(Opt(OptOps.TC, axis, (TC_SELECT.value, TC_OPT.value, USE_TC.value)))
      except KernelOptError: continue
      def split(idx, size, atype): rngs[idx] = tk.apply_opt(Opt(OptOps.SPLIT, tk.rngs.index(rngs[idx]), (size, atype)))[0]
      if TC_MIN_GLOBALS: # attempt to upcast M, local N, upcast N, skipping upcast N if we'd end up with too few globals
        if (size:=next(filter(lambda sz: rngs[1].src[0].divides(sz) is not None, [5,4,3,2]), None)) is not None: split(1, size, AxisType.UPCAST)
        if (size:=next(filter(lambda sz: rngs[0].src[0].divides(sz) is not None, [4,2]), None)) is not None: split(0, size, AxisType.LOCAL)
        if ((size:=next(filter(lambda sz: rngs[0].src[0].divides(sz) is not None, [5,4,3,2]), None)) is not None and
            resolve(prod(tk.full_shape[i] for i in tk.axes_of(AxisType.GLOBAL)) >= size*TC_MIN_GLOBALS.value, False)): split(0, size, AxisType.UPCAST)
      else: # attempt to upcast M, N, local N
        for i in [1,0]:
          if (size:=next(filter(lambda sz: rngs[i].src[0].divides(sz) is not None, [5,4,3,2]), None)) is not None: split(i, size, AxisType.UPCAST)
        if (size:=next(filter(lambda sz: rngs[0].src[0].divides(sz) is not None, [4,2]), None)) is not None: split(0, size, AxisType.LOCAL)
      return tk

  # make a copy so it does not mutate the input
  k = k.copy()

  # upcast float4 images, this must be early so we don't accidentally add locals before the upcast
  if IMAGE:
    for buf_index,buf in enumerate(k.bufs):
      if image_valid_dims(buf.src[0].dtype, buf.src[0].max_numel(), k.ren.target.arch):
        idx = k.bufs[buf_index].src[1]
        # IMAGE upcasts require one validity shared by all four unit-stride lanes so memory_coalescing can combine them into one vector read.
        unit_stride_axes_mul_4 = [k.rngs.index(c) for c in idx.get_idx().split_uop(Ops.ADD) if
          c.op is Ops.RANGE and (c.vmax+1)%4 == 0 and c not in idx.get_valid().backward_slice]
        if len(unit_stride_axes_mul_4):
          if (axis:=unit_stride_axes_mul_4[0]) in (upd:=k.upcastable_dims)+k.unrollable_dims:
            k.apply_opt(Opt(OptOps.SPLIT, axis, (4, AxisType.UPCAST if axis in upd else AxisType.UNROLL)))

  # should use matvec - TODO: adjust/tune based on the wide vs tall/large vs small mat
  MV_BLOCKSIZE, MV_THREADS_PER_ROW, MV_ROWS_PER_THREAD = getenv("MV_BLOCKSIZE", 4), getenv("MV_THREADS_PER_ROW", 8), getenv("MV_ROWS_PER_THREAD", 4)
  if k.ren.has_local and getenv("MV",1) != 0 and (MV_BLOCKSIZE > 1 or MV_THREADS_PER_ROW > 1 or MV_ROWS_PER_THREAD > 1) and  \
    k.reduceop is not None and k.reduceop.arg[0] is Ops.ADD and len(k.full_shape) >= 2 and k.ren.has_shared and \
    (mulop:=k.reduceop.src[0]).op is Ops.MUL and mulop.src[0].op is Ops.INDEX and mulop.src[1].op is Ops.INDEX:
    idx0, idx1 = mulop.src[0].src[1].get_idx(), mulop.src[1].src[1].get_idx()
    if k.ranges_of(AxisType.REDUCE):
      first_reduce_rng = k.ranges_of(AxisType.REDUCE)[0]
      if any(u is first_reduce_rng for u in idx0.split_uop(Ops.ADD)) and all(r in idx1.ranges for r in idx0.ranges):
        for global_idx in k.axes_of(AxisType.GLOBAL):
          if first_reduce_rng.src[0].divides(MV_THREADS_PER_ROW) is not None and k.full_shape[global_idx]%(MV_BLOCKSIZE*MV_ROWS_PER_THREAD) == 0:
            if DEBUG >= 3:
              print(f"MATVEC: {k.full_shape=} {first_reduce_rng.render()} {MV_BLOCKSIZE=} {MV_THREADS_PER_ROW=} {MV_ROWS_PER_THREAD=}")
            try:
              if MV_THREADS_PER_ROW > 1: k.apply_opt(Opt(OptOps.SPLIT, k.axes_of(AxisType.REDUCE)[0], (MV_THREADS_PER_ROW, AxisType.GROUP_REDUCE)))
            except KernelOptError: pass
            if MV_BLOCKSIZE > 1: k.apply_opt(Opt(OptOps.SPLIT, global_idx, (MV_BLOCKSIZE, AxisType.LOCAL)))
            if MV_ROWS_PER_THREAD > 1: k.apply_opt(Opt(OptOps.SPLIT, global_idx, (MV_ROWS_PER_THREAD, AxisType.UPCAST)))
            return k

  # are we grouping? (requires local shape support)
  if resolve(prod(k.output_shape[i] for i in k.upcastable_dims) <= (240 if k.ren.target.device == "QCOM" else 2048), False):
    for axis, sz in itertools.product(k.axes_of(AxisType.REDUCE)[:3], (16,)):
      try:
        k.apply_opt(Opt(OptOps.SPLIT, axis, (sz, AxisType.GROUP_REDUCE, True)))
        break
      except KernelOptError: pass

  # no more opt if we are grouping
  if k.group_for_reduces: return k

  # **** below this line need to be optional and benchmarked ****

  # if there are small dims with lots of valid masks, upcast them (they might be from Tensor.stack)
  to_upcast: list[int] = []
  where_gate_rngs = {r for u in k.ast.backward_slice if u.op is Ops.WHERE for r in u.src[0].ranges}
  # upcast leading axes first (hack-ish for winograd; we actually want to upcast masked axes with low stride first)
  for axis in k.upcastable_dims:
    # for Schedule, we check if the range is used in INDEX gates or WHERE gates
    is_masked = k.rngs[axis] in where_gate_rngs
    if k.full_shape[axis] <= 7 and is_masked and prod(k.full_shape[j] for j in to_upcast) * k.full_shape[axis] <= 7 * 7:
      # upcasting a masked global axis moves that range out of the launch grid into each work-item
      # under IMAGE, skip the upcast unless enough global work-items remain after it to hide memory latency
      if IMAGE and k.axis_types[axis] is AxisType.GLOBAL:
        global_upcast = prod(k.full_shape[i] for i in to_upcast if k.axis_types[i] is AxisType.GLOBAL) * k.full_shape[axis]
        global_items_after = prod(k.full_shape[i] for i in k.axes_of(AxisType.GLOBAL)) // global_upcast
        if resolve(global_items_after < getenv("OCCUPANCY_FLOOR", 4096), False): continue
      if DEBUG >= 4: print(f"upcasting masked axis : {axis}")
      to_upcast.append(axis)
  for axis in to_upcast[::-1]: k.apply_opt(Opt(OptOps.SPLIT, axis, (0, AxisType.UPCAST)))

  # potentially do more upcasts of non reduce axes based on a heuristic
  is_dsp = k.ren is not None and k.ren.target.device == "DSP"
  upcasted_axis: set[int] = set()
  while resolve(prod(k.output_shape[i] for i in k.upcastable_dims) >= 1024) and (k.upcast_size() < 32):
    xb_choices = []
    # consider all upcastable axes with 3 or 4 upcast (128 on the DSP)
    for axis, upcast_amount in itertools.product(k.upcastable_dims, ([128] if not len(upcasted_axis) else []) if is_dsp else [3,4]):
      # if we haven't upcasted it, it mods, and buffer has stride 0 on axis while having no stride 0 in the upcasted axis already
      if axis in upcasted_axis or k.full_shape[axis]%upcast_amount != 0: continue
      rng = k.rngs[axis]
      if any(rng not in b.src[1].get_idx().backward_slice and all(r2 in b.src[1].get_idx().backward_slice
          for r2 in k.ranges_of(AxisType.UPCAST, AxisType.UNROLL)) for b in k.bufs):
        num_strides, sum_strides = 0, 0
        for b in k.bufs:
          idx = b.src[1].get_idx()
          if rng in idx.backward_slice: num_strides += 1
          for c in idx.split_uop(Ops.ADD):
            if c is rng: sum_strides += 1
            if c.op is Ops.MUL and c.src[0] is rng and c.src[1].op is Ops.CONST: sum_strides += c.src[1].val
            if c.op is Ops.MUL and c.src[1] is rng and c.src[0].op is Ops.CONST: sum_strides += c.src[0].val
        xb_choices.append((num_strides, sum_strides, axis, upcast_amount))
    if xb_choices:
      xb_choices = sorted(xb_choices)
      if DEBUG >= 4: print(f"more upcast axis : {xb_choices}")
      k.apply_opt(Opt(OptOps.SPLIT, xb_choices[0][2], (xb_choices[0][3], AxisType.UPCAST)))
      upcasted_axis.add(xb_choices[0][2])
    else: break

  # if last reduce dim is small(ish), loop unroll the reduce
  # NOTE: this can fail on multireduce with mismatching dimensions, this is okay
  try:
    if k.unrollable_dims and (k.upcast_size() <= 4 or not k.axes_of(AxisType.UNROLL)) and (k.upcast_size() < 64):
      if (s:=k.full_shape[k.unrollable_dims[-1]]) <= 32:
        k.apply_opt(Opt(OptOps.SPLIT, k.unrollable_dims[-1], (0, AxisType.UNROLL)))
        # if it's small, upcast a second reduce dimension too
        if k.unrollable_dims and s <= 3 and k.full_shape[k.unrollable_dims[-1]] <= 3:
          k.apply_opt(Opt(OptOps.SPLIT, k.unrollable_dims[-1], (0, AxisType.UNROLL)))
      else:
        for splits in [4]:
          if k.full_shape[axis:=k.unrollable_dims[-1]]%splits == 0:
            k.apply_opt(Opt(OptOps.SPLIT, axis, (splits, AxisType.UNROLL)))
            break
  except KernelOptError: pass

  # if nothing at all is upcasted and it's easy to, do an upcast
  for splits in [4]:
    if not k.upcasted and k.upcastable_dims and k.full_shape[k.upcastable_dims[-1]] % splits == 0:
      k.apply_opt(Opt(OptOps.SPLIT, k.upcastable_dims[-1], (splits, AxisType.UPCAST)))

  # **** local groups ****

  if k.ren.has_local:
    if k.ren.target.device == "QCOM":
      # for openpilot: use 32..128 threads per workgroup, at most 8 on the innermost axis
      # apply innermost global axes first so the leading hardware local dims hold the trailing global axes, like gidx
      workgroup = 1
      opts: list[tuple[int, int]] = []
      for axis in [a for a in k.axes_of(AxisType.GLOBAL, AxisType.WEAK) if k.rngs[a].src[0].op is Ops.CONST][-3:][::-1]:
        if (sz:=max(x for x in range(1, min(int(k.full_shape[axis]), 128 // workgroup if opts else 8) + 1) if int(k.full_shape[axis]) % x == 0)) > 1:
          opts.append((axis, sz))
          workgroup *= sz
      if opts and workgroup < 32:  # fill at least one wave: grow the innermost local as much as possible
        axis, sz = opts[0]
        opts[0] = axis, max(x for x in range(1, min(int(k.full_shape[axis]), 128 * sz // workgroup) + 1) if int(k.full_shape[axis]) % x == 0)
      for axis, sz in opts: k.apply_opt(Opt(OptOps.SPLIT, axis, (sz, AxisType.LOCAL)))
    else:
      # prioritize making expand axes local
      local_axis_ranking = [(any(k.rngs[axis] not in b.src[1].get_idx().backward_slice for b in k.bufs), axis) \
                              for axis in k.axes_of(AxisType.GLOBAL, AxisType.WEAK) if k.rngs[axis].src[0].op is Ops.CONST]
      to_local: list[tuple[int, int]] = []
      for _, axis in sorted(local_axis_ranking, key=lambda x: (-x[0], -x[1])):
        local_size = prod(sz for _, sz in to_local)
        local_sz: int|None = next((x for x in ([32] * (axis == 0) + [16,8,4,3,2]) if k.full_shape[axis] % x == 0 and local_size * x <= 128), None)
        if local_sz is not None: to_local.append((axis, local_sz))
      deleted_shape = 0
      for axis, local_sz in sorted(to_local[:3]):
        axis = axis - deleted_shape
        will_delete_shape = local_sz == k.full_shape[axis]
        k.apply_opt(Opt(OptOps.SPLIT, axis, (local_sz, AxisType.LOCAL)))
        if will_delete_shape: deleted_shape += 1

  return k
