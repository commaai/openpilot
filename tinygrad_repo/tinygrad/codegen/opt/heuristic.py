import itertools
from tinygrad.codegen.opt import Opt, OptOps, KernelOptError
from tinygrad.helpers import getenv, DEBUG, prod, NOLOCALS, TC_OPT, TC_SELECT, USE_TC, AMX
from tinygrad.dtype import ImageDType
from tinygrad.uop.ops import Ops, resolve, AxisType
from tinygrad.codegen.opt.postrange import Scheduler

def hand_coded_optimizations(k:Scheduler) -> Scheduler:
  # first try the tensor cores
  """ Attempts to apply a tensor core optimization to the kernel. If one exists and applies properly, return true, otherwise return false.
  Tensor cores are optimized instructions that matrix multiply-accumulate across a wave of threads: D(M, N) = A(M, K) * B(K, N) + C(M, N).

  Keyword arguments:
  use_tensor_cores -- controls how tensor cores are applied (default 1)
    0: will disable any tensor core matching
    1: enable tensor cores
    2: apply tensor core shape but don't use UOp.WMMA
  extra_opts -- additional Opt's to apply after the tensor core instead of the hand-coded additional Opt's (default None)
  tc_select -- specifies which tensor core(s) to use for optimization (default -1)
    -1: iterates through all available tensor cores in order and uses the first one that matches the requirements (dims and dtypes)
    [0-N]: uses only the n'th tensor core available; useful for search
  tc_opt -- controls which kinds of kernels may be eligible for tensor cores application (default 2 during BEAM, 0 otherwise)
    0: applies to only kernels with a single reduce axis and direct Ops.LOAD into Ops.MUL
    1: allows kernels with multiple reduce axes and also multiplication of Ops.CAST'd buffers
    2: allows kernels with M, N, K axes that are not multiples of the tensor core dimensions by applying padding those axes as needed
  """
  # NOTE: unless TC_OPT is > 0, we only trigger tensor cores if there's only one reduce axis
  if USE_TC > 0 and (len(k.axes_of(AxisType.GROUP_REDUCE, AxisType.REDUCE)) == 1 or (TC_OPT.value >= 1)):
    good_tc_opt = False
    try: # check TC first and apply hand-coded opts if successful
      tk = k.copy()
      rngs = tk.apply_opt(Opt(OptOps.TC, 0, (TC_SELECT.value, TC_OPT.value, USE_TC.value)))
      good_tc_opt = True
    except KernelOptError:
      pass
    if good_tc_opt:
      # skip hand-coded TC opts if AMX, upcasting will make kernel slower
      if rngs is not None and not AMX:
        for tc_dim in [1,0]: # attempt to upcast M and N
          szs = [sz for sz in [5,4,3,2] if rngs[tc_dim].src[0].divides(sz) is not None]
          if szs:
            # set it to the replaced range
            rngs[tc_dim] = tk.apply_opt(Opt(OptOps.UPCAST, tk.rngs.index(rngs[tc_dim]), szs[0]))[0]
        if (szs := [sz for sz in [4,2] if rngs[0].src[0].divides(sz) is not None]): # attempt to local N
          tk.apply_opt(Opt(OptOps.LOCAL, tk.rngs.index(rngs[0]), szs[0]))
      return tk

  # make a copy so it does not mutate the input
  k = k.copy()

  # should use matvec - TODO: adjust/tune based on the wide vs tall/large vs small mat
  MV_BLOCKSIZE, MV_THREADS_PER_ROW, MV_ROWS_PER_THREAD = getenv("MV_BLOCKSIZE", 4), getenv("MV_THREADS_PER_ROW", 8), getenv("MV_ROWS_PER_THREAD", 4)
  if k.opts.has_local and getenv("MV",1) != 0 and (MV_BLOCKSIZE > 1 or MV_THREADS_PER_ROW > 1 or MV_ROWS_PER_THREAD > 1) and  \
    k.reduceop is not None and k.reduceop.arg[0] is Ops.ADD and len(k.full_shape) >= 2 and k.opts.has_shared and \
    (mulop:=k.reduceop.src[0]).op is Ops.MUL and mulop.src[0].op is Ops.LOAD and mulop.src[1].op is Ops.LOAD:
    idx0, idx1 = mulop.src[0].src[0].src[1].get_idx(), mulop.src[1].src[0].src[1].get_idx()
    first_reduce_rng = k.ranges_of(AxisType.REDUCE)[0]
    if any(u is first_reduce_rng for u in idx0.split_uop(Ops.ADD)) and all(r in idx1.ranges for r in idx0.ranges):
      for global_idx in k.axes_of(AxisType.GLOBAL):
        if first_reduce_rng.src[0].divides(MV_THREADS_PER_ROW) is not None and k.full_shape[global_idx]%(MV_BLOCKSIZE*MV_ROWS_PER_THREAD) == 0:
          if DEBUG >= 3:
            print(f"MATVEC: {k.full_shape=} {first_reduce_rng.render()} {MV_BLOCKSIZE=} {MV_THREADS_PER_ROW=} {MV_ROWS_PER_THREAD=}")
          if MV_THREADS_PER_ROW > 1: k.apply_opt(Opt(OptOps.GROUP, 0, MV_THREADS_PER_ROW))
          if MV_BLOCKSIZE > 1: k.apply_opt(Opt(OptOps.LOCAL, global_idx, MV_BLOCKSIZE))
          if MV_ROWS_PER_THREAD > 1: k.apply_opt(Opt(OptOps.UPCAST, global_idx, MV_ROWS_PER_THREAD))
          return k

  # are we grouping? (requires local shape support)
  if resolve(prod(k.output_shape[i] for i in k.upcastable_dims) <= 2048, False):
    for sz in [16]:
      try:
        k.apply_opt(Opt(OptOps.GROUPTOP, 0, sz))
        break
      except KernelOptError: pass

  # upcast float4 images
  for buf_index,buf in enumerate(k.bufs):
    if isinstance(buf.src[0].dtype, ImageDType):
      # part of real_strides
      unit_stride_axes_mul_4 = [k.rngs.index(c) for c in k.bufs[buf_index].src[1].get_idx().split_uop(Ops.ADD) if
        c.op is Ops.RANGE and (c.vmax+1)%4 == 0]
      if len(unit_stride_axes_mul_4):
        if (axis:=unit_stride_axes_mul_4[0]) in k.upcastable_dims:
          k.apply_opt(Opt(OptOps.UPCAST, axis, 4))
        elif axis in k.unrollable_dims:
          k.apply_opt(Opt(OptOps.UNROLL, k.unrollable_dims.index(axis), 4))

  # no more opt if we are grouping
  if k.group_for_reduces: return k

  # **** below this line need to be optional and benchmarked ****

  # if there are small dims with lots of valid masks, upcast them (they might be from Tensor.stack)
  to_upcast: list[int] = []
  # upcast leading axes first (hack-ish for winograd; we actually want to upcast masked axes with low stride first)
  for axis in k.upcastable_dims:
    # for Schedule, we check if the range is used in INDEX gates or WHERE gates
    is_masked = any(any(o is k.rngs[axis] for o in u.src[0].parents) for u in k.ast.parents if u.op is Ops.WHERE)
    if k.full_shape[axis] <= 7 and is_masked and prod(k.full_shape[j] for j in to_upcast) * k.full_shape[axis] <= 7 * 7:
      if DEBUG >= 4: print(f"upcasting masked axis : {axis}")
      to_upcast.append(axis)
  for axis in to_upcast[::-1]: k.apply_opt(Opt(OptOps.UPCAST, axis, 0))

  # potentially do more upcasts of non reduce axes based on a heuristic
  is_dsp = k.opts is not None and k.opts.device == "DSP"
  upcasted_axis: set[int] = set()
  while resolve(prod(k.output_shape[i] for i in k.upcastable_dims) >= 1024):
    xb_choices = []
    # consider all upcastable axes with 3 or 4 upcast (128 on the DSP)
    for axis, upcast_amount in itertools.product(k.upcastable_dims, ([128] if not len(upcasted_axis) else []) if is_dsp else [3,4]):
      # if we haven't upcasted it, it mods, and buffer has stride 0 on axis while having no stride 0 in the upcasted axis already
      if axis in upcasted_axis or k.full_shape[axis]%upcast_amount != 0: continue
      rng = k.rngs[axis]
      if any(rng not in b.src[1].get_idx().parents and all(r2 in b.src[1].get_idx().parents
          for r2 in k.ranges_of(AxisType.UPCAST, AxisType.UNROLL)) for b in k.bufs):
        num_strides, sum_strides = 0, 0
        for b in k.bufs:
          idx = b.src[1].get_idx()
          if rng in idx.parents: num_strides += 1
          for c in idx.split_uop(Ops.ADD):
            if c is rng: sum_strides += 1
            if c.op is Ops.MUL and c.src[0] is rng and c.src[1].op is Ops.CONST: sum_strides += c.src[1].arg
            if c.op is Ops.MUL and c.src[1] is rng and c.src[0].op is Ops.CONST: sum_strides += c.src[0].arg
        xb_choices.append((num_strides, sum_strides, axis, upcast_amount))
    if xb_choices:
      xb_choices = sorted(xb_choices)
      if DEBUG >= 4: print(f"more upcast axis : {xb_choices}")
      k.apply_opt(Opt(OptOps.UPCAST, xb_choices[0][2], xb_choices[0][3]))
      upcasted_axis.add(xb_choices[0][2])
    else: break

  # if last reduce dim is small(ish), loop unroll the reduce
  # NOTE: this can fail on multireduce with mismatching dimensions, this is okay
  try:
    upcast_size = prod(k.full_shape[a] for a in k.axes_of(AxisType.UPCAST, AxisType.UNROLL))
    if k.unrollable_dims and (upcast_size <= 4 or not k.axes_of(AxisType.UNROLL)) and (upcast_size < 64):
      if (s:=k.full_shape[k.unrollable_dims[-1]]) <= 32:
        k.apply_opt(Opt(OptOps.UNROLL, len(k.unrollable_dims)-1, 0))
        # if it's small, upcast a second reduce dimension too
        if k.unrollable_dims and s <= 3 and k.full_shape[k.unrollable_dims[-1]] <= 3:
          k.apply_opt(Opt(OptOps.UNROLL, len(k.unrollable_dims)-1, 0))
      else:
        for splits in [4]:
          if k.full_shape[axis:=k.unrollable_dims[-1]]%splits == 0:
            k.apply_opt(Opt(OptOps.UNROLL, len(k.unrollable_dims)-1, splits))
            break
  except KernelOptError: pass

  # if nothing at all is upcasted and it's easy to, do an upcast
  for splits in [4]:
    # TODO: somehow this never hits a reduce
    if not k.upcasted and k.upcastable_dims and k.full_shape[k.upcastable_dims[-1]] % splits == 0:
      k.apply_opt(Opt(OptOps.UPCAST, k.upcastable_dims[-1], splits))

  # **** local groups ****

  if k.opts.has_local:
    if NOLOCALS:
      k.apply_opt(Opt(OptOps.NOLOCALS))
    else:
      # prioritize making expand axes local
      local_axis_ranking = [(any(k.rngs[axis] not in b.src[1].get_idx().parents for b in k.bufs), axis) \
                              for axis in k.axes_of(AxisType.GLOBAL, AxisType.LOOP) if k.rngs[axis].src[0].op is Ops.CONST]
      to_local: list[tuple[int, int]] = []
      for _, axis in sorted(local_axis_ranking, key=lambda x: (-x[0], -x[1])):
        local_size = prod(sz for _, sz in to_local)
        local_sz: int|None = next((x for x in ([32] * (axis == 0) + [16,8,4,3,2]) if k.full_shape[axis] % x == 0 and local_size * x <= 128), None)
        if local_sz is not None: to_local.append((axis, local_sz))
      deleted_shape = 0
      for axis, local_sz in sorted(to_local[:3]):
        axis = axis - deleted_shape
        will_delete_shape = local_sz == k.full_shape[axis]
        k.apply_opt(Opt(OptOps.LOCAL, axis, local_sz))
        if will_delete_shape: deleted_shape += 1

  # **** threading ****

  if k.opts.has_threads and k.opts.global_max is not None:
    for threads in [32,16,12,8,6,5,4,3,2]:
      # Skip is too many threads. Heuristic: use about 128K ops per thread
      if threads > k.opts.global_max[0] or resolve(prod(k.full_shape) // (128 << 10) < threads): continue
      for axis in k.axes_of(AxisType.LOOP):
        if k.full_shape[axis] % threads == 0:
          k.apply_opt(Opt(OptOps.THREAD, axis, threads))
          break
      if k.applied_opts and k.applied_opts[-1].op is OptOps.THREAD: break

  return k
