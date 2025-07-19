import itertools
from tinygrad.opt.kernel import Kernel, Opt, OptOps, KernelOptError, AxisType
from tinygrad.helpers import getenv, DEBUG, prod, NOLOCALS
from tinygrad.dtype import ImageDType
from tinygrad.uop.ops import Ops, resolve

def hand_coded_optimizations(k:Kernel) -> list[Opt]:
  # make a copy so it does not mutate the input
  k = k.copy()

  # should use matvec - TODO: adjust/tune based on the wide vs tall/large vs small mat
  MV_BLOCKSIZE, MV_THREADS_PER_ROW, MV_ROWS_PER_THREAD = getenv("MV_BLOCKSIZE", 4), getenv("MV_THREADS_PER_ROW", 8), getenv("MV_ROWS_PER_THREAD", 4)
  if k.opts.has_local and getenv("MV",1) != 0 and (MV_BLOCKSIZE > 1 or MV_THREADS_PER_ROW > 1 or MV_ROWS_PER_THREAD > 1) and  \
    k.reduceop is not None and k.reduceop.arg[0] is Ops.ADD and len(k.full_shape) >= 2 and k.opts.has_shared and \
    (mulop:=k.reduceop.src[0]).op is Ops.MUL and mulop.src[0].op is Ops.LOAD and mulop.src[1].op is Ops.LOAD:
    st0, st1 = k.sts[k.bufs.index(mulop.src[0])], k.sts[k.bufs.index(mulop.src[1])]
    strides0, strides1 = st0.real_strides(), st1.real_strides()
    def has_expanded_axis(shape, strides): return any(resolve(s > 1) and not resolve(st != 0) for s,st in zip(shape,strides))
    if strides0[k.first_reduce] == 1 and not (has_expanded_axis(st0.shape, strides0) and has_expanded_axis(st1.shape, strides1)):
      for global_idx in range(k.global_dims):
        if k.full_shape[k.first_reduce]%MV_THREADS_PER_ROW == 0 and k.full_shape[global_idx]%(MV_BLOCKSIZE*MV_ROWS_PER_THREAD) == 0:
          if DEBUG >= 3:
            print(f"MATVEC: {k.full_shape=} {k.first_reduce=} {strides0=} {MV_BLOCKSIZE=} {MV_THREADS_PER_ROW=} {MV_ROWS_PER_THREAD=}")
          if MV_THREADS_PER_ROW > 1: k.apply_opt(Opt(OptOps.GROUP, 0, MV_THREADS_PER_ROW))
          if MV_BLOCKSIZE > 1: k.apply_opt(Opt(OptOps.LOCAL, global_idx, MV_BLOCKSIZE))
          if MV_ROWS_PER_THREAD > 1: k.apply_opt(Opt(OptOps.UPCAST, global_idx, MV_ROWS_PER_THREAD))
          return k.applied_opts

  # are we grouping? (requires local shape support)
  if resolve(prod(k.sts[0].shape[:k.first_reduce]) <= 2048, False):
    for sz in [16]:
      try: # may fail due to excessive smem usage
        k.apply_opt(Opt(OptOps.GROUPTOP, 0, sz))
        break
      except KernelOptError: pass

  # upcast float4 images
  for buf_index,buf in enumerate(k.bufs):
    if isinstance(buf.src[0].dtype, ImageDType):
      if (unit_stride_axes_mul_4 := [i for i in k.sts[buf_index].unit_stride_axes(ignore_valid=True) if k.sts[buf_index].shape[i]%4 == 0]):
        if k.axis_types[axis:=unit_stride_axes_mul_4[0]] in (AxisType.GLOBAL, AxisType.LOCAL):
          k.apply_opt(Opt(OptOps.UPCAST, axis, 4))
        elif k.axis_types[axis:=unit_stride_axes_mul_4[0]] in (AxisType.GROUP_REDUCE, AxisType.REDUCE):
          k.apply_opt(Opt(OptOps.UNROLL, axis-k.first_reduce, 4))

  # no more opt if we are grouping
  if k.group_for_reduces: return k.applied_opts

  # **** below this line need to be optional and benchmarked ****

  # if there are small dims with lots of valid masks, upcast them (they might be from Tensor.stack)
  to_upcast: list[int] = []
  # upcast leading axes first (hack-ish for winograd; we actually want to upcast masked axes with low stride first)
  for axis in range(k.first_reduce):
    # for now skip upcasting here if there is a symbolic axis
    if isinstance(k.full_shape[axis], int) and k.full_shape[axis] <= 7 and any(st.axis_is_masked(axis) for st in k.sts) and \
      prod(k.full_shape[j] for j in to_upcast) * k.full_shape[axis] <= 7 * 7:
      if DEBUG >= 4: print(f"upcasting masked axis : {axis}")
      to_upcast.append(axis)
  for axis in to_upcast[::-1]: k.apply_opt(Opt(OptOps.UPCAST, axis, 0))

  # potentially do more upcasts of non reduce axes based on a heuristic
  is_dsp = k.opts is not None and k.opts.device == "DSP"
  upcasted_axis: set[int] = set()
  while resolve(prod(k.sts[0].shape[:k.first_reduce]) >= 1024):
    xb_choices = []
    # consider all the non reduce axes, and a 3 or 4 reduce. (128 on the DSP)
    for axis, upcast_amount in itertools.product(range(k.first_reduce), ([128] if not len(upcasted_axis) else []) if is_dsp else [3,4]):
      # if we haven't upcasted it, it's not symbolic, it mods, and buffer has stride 0 on axis while having no stride 0 in the upcasted axis already
      if axis not in upcasted_axis and isinstance(k.full_shape[axis], int) and k.full_shape[axis]%upcast_amount == 0 and \
        any(st.views[-1].strides[axis] == 0 and not any(x == 0 for x in st.real_strides()[k.first_upcast:]) \
        for st in k.sts):
        xb_choices.append((sum(st.views[-1].strides[axis]>0 for st in k.sts),
                           sum(st.views[-1].strides[axis] for st in k.sts), axis, upcast_amount))
    if xb_choices:
      xb_choices = sorted(xb_choices)
      if DEBUG >= 4: print(f"float4 merging axis : {xb_choices}")
      k.apply_opt(Opt(OptOps.UPCAST, xb_choices[0][2], xb_choices[0][3]))
      upcasted_axis.add(xb_choices[0][2])
    else: break

  # if last dim is small(ish) and it's a reduce dim, loop unroll the reduce
  if k.first_reduce < k.first_upcast and \
      (prod(k.full_shape[k.first_upcast:]) <= 4 or (AxisType.UNROLL not in k.axis_types)) and (prod(k.full_shape[k.first_upcast:]) < 64):
    if isinstance(s:=k.full_shape[k.first_upcast-1], int) and s <= 32:  # NOTE: cannot loop unroll symbolic axis
      k.apply_opt(Opt(OptOps.UNROLL, k.first_upcast-1-k.first_reduce, 0))
      # if it's small, upcast a second reduce dimension too
      if k.first_reduce < k.first_upcast and s <= 3 and isinstance(s2:=k.full_shape[k.first_upcast-1], int) and s2 <= 3:
        k.apply_opt(Opt(OptOps.UNROLL, k.first_upcast-1-k.first_reduce, 0))
    else:
      for splits in [4]:
        if k.full_shape[k.first_upcast-1]%splits == 0:
          k.apply_opt(Opt(OptOps.UNROLL, k.first_upcast-1-k.first_reduce, splits))
          break

  # if nothing at all is upcasted and it's easy to, do an upcast
  for splits in [4]:
    # TODO: somehow this never hits a reduce
    if k.upcasted == 0 and k.shape_len > 0 and k.full_shape[k.first_upcast-1] % splits == 0:
      k.apply_opt(Opt(OptOps.UPCAST, k.first_upcast-1, splits))

  # **** local groups ****

  if k.opts.has_local:
    if NOLOCALS:
      k.apply_opt(Opt(OptOps.NOLOCALS))
    else:
      # prioritize making expand axes local
      local_axis_ranking = [(any(st.views[-1].strides[axis] == 0 for st in k.sts), axis) for axis in range(k.first_reduce)]
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

  return k.applied_opts
