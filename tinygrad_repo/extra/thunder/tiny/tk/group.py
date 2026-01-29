import math
from typing import cast, Callable
from tinygrad import dtypes
from tinygrad.uop.ops import AxisType, UOp, Ops
from tinygrad.dtype import AddrSpace, PtrDType
from tinygrad.helpers import prod

from extra.thunder.tiny.tk import WARP_THREADS
from extra.thunder.tiny.tk.tiles import ALL_TILES, ST, RT, RV, TileLayout, VecLayout

class Group:
  def __init__(self, warps:int, ker):
    self.warps = warps
    self.group_threads = warps * WARP_THREADS
    self.ker = ker

  # helpers
  @property
  def laneid(self): return self.ker.threadIdx_x % self.group_threads
  @property
  def warpid(self): return self.laneid // WARP_THREADS
  @property
  def groupid(self): return self.ker.threadIdx_x // self.group_threads

  # ops that only work on a single warp

  def clear(self, reg:ALL_TILES, value:float=0):
    reg = cast(UOp, reg)
    assert self.warps == 1

    rngs_for_shape = tuple(self.ker.raw_range(dim) for dim in reg.shape)

    reg_store = reg[*rngs_for_shape].store(value).end(*rngs_for_shape)

    self.ker.push_store(reg_store, reg)
    return reg.after(reg_store).reshape(reg.shape)

  def zero(self, reg:ALL_TILES): return self.clear(reg, 0)
  def ones(self, reg:ALL_TILES): return self.clear(reg, 1)
  def neg_inf(self, reg:ALL_TILES): return self.clear(reg, -math.inf)

  def copy(self, dst:ALL_TILES, src:ALL_TILES):
    dst, src = cast(UOp, dst), cast(UOp, src)
    assert self.warps == 1
    assert dst.shape == src.shape

    rngs_for_shape = tuple(self.ker.raw_range(dim) for dim in dst.shape)

    src_load = src[*rngs_for_shape]
    if src.dtype.base != dst.dtype.base:
      src_load = src_load.cast(dst.dtype.base)
    dst_store = dst[*rngs_for_shape].store(src_load).end(*rngs_for_shape)

    self.ker.push_store(dst_store, dst)
    return dst.after(dst_store).reshape(dst.shape)

  def transpose(self, dst:UOp|RT, src:UOp|RT):
    dst, src = cast(UOp, dst), cast(UOp, src)
    assert self.warps == 1

    for height in self.ker.range(src.shape[-3], track=False):
      for width in self.ker.range(src.shape[-2], track=False):
        for inner in self.ker.range(src.shape[-1], track=False):
          src_load = src[height, width, inner]
          if src.dtype.base != dst.dtype.base:
            src_load = src_load.cast(dst.dtype.base)
          dst_store = dst[width, height, inner].store(src_load).end(height, width, inner)

    self.ker.push_store(dst_store, dst)
    return dst.after(dst_store).reshape(dst.shape)

  def mma_AB(self, c:UOp|RT, a:UOp|RT, b:UOp|RT):
    c, a, b = cast(UOp, c), cast(UOp, a), cast(UOp, b)
    assert self.warps == 1

    a_base_shape = cast(RT, a).base_shape
    if a_base_shape.cols == 16:
      wmma_arg = ('WMMA_16_16_16___bf16_float', (16, 16, 16), dtypes.bfloat16, dtypes.float, 'AMD', 64, (((4, 2), (3, 2)), ((4, 2), (3, 2)), ((4, 2), (3, 2))), ()) # type: ignore
    elif a_base_shape.cols == 32:
      wmma_arg = ('WMMA_16_16_32___bf16_float', (16, 16, 32), dtypes.bfloat16, dtypes.float, 'AMD', 64, (((4, 2), (3, 2), (9, 2)), ((4, 2), (3, 2), (9, 2)), ((4, 2), (3, 2))), ()) # type: ignore
    else: raise NotImplementedError(f"mma_AB not implemented for {a_base_shape.cols=}")

    for height in self.ker.range(c.shape[-3], track=False):
      for width in self.ker.range(c.shape[-2], track=False):
        for inner in self.ker.range(a.shape[-2], axis_type=AxisType.REDUCE, track=False):
          if a_base_shape.cols == 16:
            a_in = UOp.vectorize(*[a[height, inner, i] for i in range(4)])
            b_in = UOp.vectorize(*[b[inner, width, i] for i in range(4)])
          elif a_base_shape.cols == 32:
            a_in = UOp.vectorize(*[a[height, inner, i] for i in range(8)])
            b_in = UOp.vectorize(*[b[inner, width, i] for i in range(8)])
          else: raise NotImplementedError(f"mma_AB not implemented for {a_base_shape.cols=}")
          d_in = UOp.vectorize(*[c[height, width, i] for i in range(4)])

          out = UOp(Ops.WMMA, dtypes.float32.vec(4), (a_in, b_in, d_in), arg=wmma_arg)
          c_i = [c[height, width, i].store(out.gep(i)) for i in range(4)]
          c_store = UOp.group(*c_i).end(height, width, inner)

    self.ker.push_store(c_store, c)
    return c.after(c_store).reshape(c.shape)

  def mma_ABt(self, c:UOp|RT, a:UOp|RT, b:UOp|RT):
    c, a, b = cast(UOp, c), cast(UOp, a), cast(UOp, b)
    assert self.warps == 1

    a_base_shape = cast(RT, a).base_shape
    if a_base_shape.cols == 16:
      wmma_arg = ('WMMA_16_16_16___bf16_float', (16, 16, 16), dtypes.bfloat16, dtypes.float, 'AMD', 64, (((4, 2), (3, 2)), ((4, 2), (3, 2)), ((4, 2), (3, 2))), ()) # type: ignore
    elif a_base_shape.cols == 32:
      wmma_arg = ('WMMA_16_16_32___bf16_float', (16, 16, 32), dtypes.bfloat16, dtypes.float, 'AMD', 64, (((4, 2), (3, 2), (9, 2)), ((4, 2), (3, 2), (9, 2)), ((4, 2), (3, 2))), ()) # type: ignore
    else: raise NotImplementedError(f"mma_ABt not implemented for {a_base_shape.cols=}")

    for height in self.ker.range(c.shape[-3], track=False):
      for width in self.ker.range(c.shape[-2], track=False):
        for inner in self.ker.range(a.shape[-2], axis_type=AxisType.REDUCE, track=False):
          if a_base_shape.cols == 16:
            a_in = UOp.vectorize(*[a[height, inner, i] for i in range(4)])
            b_in = UOp.vectorize(*[b[width, inner, i] for i in range(4)])
          elif a_base_shape.cols == 32:
            a_in = UOp.vectorize(*[a[height, inner, i] for i in range(8)])
            b_in = UOp.vectorize(*[b[width, inner, i] for i in range(8)])
          else: raise NotImplementedError(f"mma_ABt not implemented for {a_base_shape.cols=}")
          d_in = UOp.vectorize(*[c[height, width, i] for i in range(4)])

          out = UOp(Ops.WMMA, dtypes.float32.vec(4), (a_in, b_in, d_in), arg=wmma_arg)
          c_i = [c[height, width, i].store(out.gep(i)) for i in range(4)]
          c_store = UOp.group(*c_i).end(height, width, inner)

    self.ker.push_store(c_store, c)
    return c.after(c_store).reshape(c.shape)

  def mma_AtB(self, c:UOp|RT, a:UOp|RT, b:UOp|RT):
    c, a, b = cast(UOp, c), cast(UOp, a), cast(UOp, b)
    assert self.warps == 1

    a_base_shape = cast(RT, a).base_shape
    if a_base_shape.cols == 16:
      wmma_arg = ('WMMA_16_16_16___bf16_float', (16, 16, 16), dtypes.bfloat16, dtypes.float, 'AMD', 64, (((4, 2), (3, 2)), ((4, 2), (3, 2)), ((4, 2), (3, 2))), ()) # type: ignore
    elif a_base_shape.cols == 32:
      wmma_arg = ('WMMA_16_16_32___bf16_float', (16, 16, 32), dtypes.bfloat16, dtypes.float, 'AMD', 64, (((4, 2), (3, 2), (9, 2)), ((4, 2), (3, 2), (9, 2)), ((4, 2), (3, 2))), ()) # type: ignore
    else: raise NotImplementedError(f"mma_AtB not implemented for {a_base_shape.cols=}")

    for height in self.ker.range(c.shape[-3], track=False):
      for width in self.ker.range(c.shape[-2], track=False):
        for inner in self.ker.range(a.shape[-3], axis_type=AxisType.REDUCE, track=False):
          if a_base_shape.cols == 16:
            a_in = UOp.vectorize(*[a[inner, height, i] for i in range(4)])
            b_in = UOp.vectorize(*[b[inner, width, i] for i in range(4)])
          elif a_base_shape.cols == 32:
            a_in = UOp.vectorize(*[a[inner, height, i] for i in range(8)])
            b_in = UOp.vectorize(*[b[inner, width, i] for i in range(8)])
          else: raise NotImplementedError(f"mma_AtB not implemented for {a_base_shape.cols=}")
          d_in = UOp.vectorize(*[c[height, width, i] for i in range(4)])

          out = UOp(Ops.WMMA, dtypes.float32.vec(4), (a_in, b_in, d_in), arg=wmma_arg)
          c_i = [c[height, width, i].store(out.gep(i)) for i in range(4)]
          c_store = UOp.group(*c_i).end(height, width, inner)

    self.ker.push_store(c_store, c)
    return c.after(c_store).reshape(c.shape)

  def mma_AtBt(self, c:UOp|RT, a:UOp|RT, b:UOp|RT):
    c, a, b = cast(UOp, c), cast(UOp, a), cast(UOp, b)
    assert self.warps == 1

    a_base_shape = cast(RT, a).base_shape
    if a_base_shape.cols == 16:
      wmma_arg = ('WMMA_16_16_16___bf16_float', (16, 16, 16), dtypes.bfloat16, dtypes.float, 'AMD', 64, (((4, 2), (3, 2)), ((4, 2), (3, 2)), ((4, 2), (3, 2))), ()) # type: ignore
    elif a_base_shape.cols == 32:
      wmma_arg = ('WMMA_16_16_32___bf16_float', (16, 16, 32), dtypes.bfloat16, dtypes.float, 'AMD', 64, (((4, 2), (3, 2), (9, 2)), ((4, 2), (3, 2), (9, 2)), ((4, 2), (3, 2))), ()) # type: ignore
    else: raise NotImplementedError(f"mma_AtBt not implemented for {a_base_shape.cols=}")

    for height in self.ker.range(c.shape[-3], track=False):
      for width in self.ker.range(c.shape[-2], track=False):
        for inner in self.ker.range(a.shape[-3], axis_type=AxisType.REDUCE, track=False):
          if a_base_shape.cols == 16:
            a_in = UOp.vectorize(*[a[inner, height, i] for i in range(4)])
            b_in = UOp.vectorize(*[b[width, inner, i] for i in range(4)])
          elif a_base_shape.cols == 32:
            a_in = UOp.vectorize(*[a[inner, height, i] for i in range(8)])
            b_in = UOp.vectorize(*[b[width, inner, i] for i in range(8)])
          else: raise NotImplementedError(f"mma_AtBt not implemented for {a_base_shape.cols=}")
          d_in = UOp.vectorize(*[c[height, width, i] for i in range(4)])

          out = UOp(Ops.WMMA, dtypes.float32.vec(4), (a_in, b_in, d_in), arg=wmma_arg)
          c_i = [c[height, width, i].store(out.gep(i)) for i in range(4)]
          c_store = UOp.group(*c_i).end(height, width, inner)

    self.ker.push_store(c_store, c)
    return c.after(c_store).reshape(c.shape)

  def map(self, a:ALL_TILES, op:Callable[[UOp], UOp]|Callable[[UOp, tuple], UOp]):
    a = cast(UOp, a)
    assert self.warps == 1

    rngs_for_shape = tuple(self.ker.raw_range(dim) for dim in a.shape)

    if op.__code__.co_argcount == 1:
      to_store = op(a[*rngs_for_shape]) # type: ignore
    else:
      to_store = op(a[*rngs_for_shape], rngs_for_shape) # type: ignore

    a_store = a[*rngs_for_shape].store(to_store).end(*rngs_for_shape)

    self.ker.push_store(a_store, a)
    return a.after(a_store).reshape(a.shape)

  def row_reduce(self, vec:UOp|RV, src:UOp|RT, op:Callable[[UOp, UOp], UOp], init_value:float=0.0):
    vec, src = cast(UOp, vec), cast(UOp, src)
    assert self.warps == 1

    red_local = self.ker.alloc((self.group_threads,), src.dtype.base, AddrSpace.LOCAL)
    red_reg = self.ker.alloc((1,), src.dtype.base, AddrSpace.REG)

    for height in self.ker.range(src.shape[-3], track=False):
      i = self.ker.raw_range(red_reg.size)
      red_reg = red_reg.after(height, *[tkr._rng for tkr in self.ker.range_stack])
      reg_store = red_reg.flatten()[i].store(init_value).end(i)
      red_reg = red_reg.after(reg_store).reshape(red_reg.shape)

      for width in self.ker.range(src.shape[-2], axis_type=AxisType.REDUCE, track=False):
        for inner in self.ker.range(4, axis_type=AxisType.REDUCE, track=False):
          reg_store = red_reg[0].store(op(red_reg[0], src[height, width, inner])).end(width, inner)
          red_reg = red_reg.after(reg_store).reshape(red_reg.shape)

      # store to shared memory
      red_local_store = red_local[self.laneid].store(red_reg[0])
      red_local = red_local.after(red_local_store.barrier()).reshape(red_local.shape)

      # reduce from shared memory
      for inner in self.ker.range(3, axis_type=AxisType.REDUCE, track=False):
        offset = (self.laneid + (1 + inner) * 16) % self.group_threads
        reg_store = red_reg[0].store(op(red_reg[0], red_local[offset])).end(inner)
        red_reg = red_reg.after(reg_store).reshape(red_reg.shape)

      # reduce with vec
      vec_store = vec[height, 0].store(op(vec[height, 0], red_reg[0])).end(height)

    self.ker.push_store(vec_store, vec)
    return vec.after(vec_store).reshape(vec.shape)

  def col_reduce(self, vec:UOp|RV, src:UOp|RT, op:Callable[[UOp, UOp], UOp], init_value:float=0.0):
    vec, src = cast(UOp, vec), cast(UOp, src)
    assert self.warps == 1

    red_local = self.ker.alloc((self.group_threads,), src.dtype.base, AddrSpace.LOCAL)
    red_reg = self.ker.alloc((1,), src.dtype.base, AddrSpace.REG)

    for width in self.ker.range(src.shape[-2], track=False):
      i = self.ker.raw_range(red_reg.size)
      red_reg = red_reg.after(width, *[tkr._rng for tkr in self.ker.range_stack])
      reg_store = red_reg.flatten()[i].store(init_value).end(i)
      red_reg = red_reg.after(reg_store).reshape(red_reg.shape)

      for height in self.ker.range(src.shape[-3], axis_type=AxisType.REDUCE, track=False):
        for inner in self.ker.range(4, axis_type=AxisType.REDUCE, track=False):
          reg_store = red_reg[0].store(op(red_reg[0], src[height, width, inner])).end(height, inner)
          red_reg = red_reg.after(reg_store).reshape(red_reg.shape)

      # store to shared memory
      red_local_store = red_local[self.laneid].store(red_reg[0])
      red_local = red_local.after(red_local_store.barrier()).reshape(red_local.shape)

      # reduce from shared memory
      for inner in self.ker.range(3, axis_type=AxisType.REDUCE, track=False):
        offset = (self.laneid + (1 + inner) * 16) % self.group_threads
        reg_store = red_reg[0].store(op(red_reg[0], red_local[offset])).end(inner)
        red_reg = red_reg.after(reg_store).reshape(red_reg.shape)

      # reduce with vec
      vec_store = vec[width, 0].store(op(vec[width, 0], red_reg[0])).end(width)

    self.ker.push_store(vec_store, vec)
    return vec.after(vec_store).reshape(vec.shape)

  # ops that can work across multiple warps

  def load(self, dst:ALL_TILES, src:ALL_TILES, dst_idxs:tuple[UOp|int,...]=(), idxs:tuple[UOp|int,...]=(), axis:int=0):
    dst, src = cast(UOp, dst), cast(UOp, src)
    assert isinstance(dst.dtype, PtrDType) and isinstance(src.dtype, PtrDType)
    dst_dtype, src_dtype = dst.dtype, src.dtype
    if dst_dtype.addrspace == AddrSpace.REG and src_dtype.addrspace == AddrSpace.LOCAL:
      laneid = self.ker.laneid
      rt, st = cast(RT, dst), cast(ST, src)
      elements_per_thread = rt.base_shape.elements_per_thread

      for height in self.ker.range(dst.shape[-3], track=False):
        for width in self.ker.range(dst.shape[-2], track=False):
          for inner in self.ker.range(elements_per_thread, track=False):
            if rt.layout != st.layout:
              row = rt.base_shape.stride * (laneid // rt.base_shape.cols) + inner
              col = laneid % rt.base_shape.cols
            else:
              row = laneid % rt.base_shape.rows
              col = rt.base_shape.stride * (laneid // rt.base_shape.rows) + inner

            sheight = height
            swidth = width
            if len(idxs) == 2:
              row_idx = idxs[0] * dst.shape[-3] * rt.base_shape.rows
              col_idx = idxs[1] * dst.shape[-2] * rt.base_shape.cols

              row += row_idx % st.base_shape.rows
              col += col_idx % st.base_shape.cols
              sheight += row_idx // st.base_shape.rows
              swidth += col_idx // st.base_shape.cols

            srow, scol = cast(ST, src).swizzle(row, col)

            src_load = src[*idxs[:-2], sheight, swidth, srow, scol]
            if src.dtype.base != dst.dtype.base:
              src_load = src_load.cast(dst.dtype.base)
            dst_store = dst[*dst_idxs, height, width, inner].store(src_load)
            dst_store = dst_store.end(height, width, inner)
    elif dst_dtype.addrspace == AddrSpace.LOCAL and src_dtype.addrspace == AddrSpace.GLOBAL:
      srcf = src.flatten()
      row_stride = prod(src.shape[axis+1:])

      st = cast(ST, dst)
      idxs = tuple(idx * st.rows if i == axis else idx for i, idx in enumerate(idxs))
      idxs = tuple(idx * st.cols if i == 3 else idx for i, idx in enumerate(idxs))
      src_i = ((idxs[0] * src.shape[-3] + idxs[1]) * src.shape[-2] + idxs[2]) * src.shape[-1] + idxs[3]

      elements_per_thread = st.base_shape.elements_per_thread
      memcpy_per_row = st.cols // elements_per_thread
      total_calls = (dst.shape[-4] * dst.shape[-3] * st.base_shape.num_elements) // (self.group_threads * elements_per_thread)

      for outer in self.ker.range(total_calls, track=False):
        for inner in self.ker.range(elements_per_thread, axis_type=AxisType.UPCAST, track=False):
          load_idx = outer * self.group_threads + self.laneid
          row = load_idx // memcpy_per_row
          col = (load_idx * elements_per_thread) % st.cols + inner
          height = row // st.base_shape.rows
          width = col // st.base_shape.cols

          row = row % st.base_shape.rows
          col = col % st.base_shape.cols

          srow, scol = cast(ST, dst).swizzle(row, col)

          src_i += height * st.base_shape.rows * row_stride + width * st.base_shape.cols
          src_i += row * row_stride + col

          src_load = srcf[src_i]
          if src.dtype.base != dst.dtype.base:
            src_load = src_load.cast(dst.dtype.base)
          dst_store = dst[*dst_idxs, height, width, srow, scol].store(src_load)
          dst_store = dst_store.end(height, width, outer, inner).barrier()
    elif dst_dtype.addrspace == AddrSpace.REG and src_dtype.addrspace == AddrSpace.GLOBAL and isinstance(dst, RT):
      srcf = src.flatten()
      row_stride = prod(src.shape[axis+1:])

      laneid = self.ker.laneid
      rt = cast(RT, dst)
      elements_per_thread = rt.base_shape.elements_per_thread

      idxs = tuple(idx * dst.shape[-3] * rt.base_shape.rows if i == axis else idx for i, idx in enumerate(idxs))
      idxs = tuple(idx * dst.shape[-2] * rt.base_shape.cols if i == 3 else idx for i, idx in enumerate(idxs))
      src_i = ((idxs[0] * src.shape[-3] + idxs[1]) * src.shape[-2] + idxs[2]) * src.shape[-1] + idxs[3]

      for height in self.ker.range(dst.shape[-3], track=False):
        for width in self.ker.range(dst.shape[-2], track=False):
          for inner in self.ker.range(elements_per_thread, track=False):
            base_row = height * rt.base_shape.rows
            base_col = width * rt.base_shape.cols

            if rt.layout == TileLayout.COL:
              row = rt.base_shape.stride * (laneid // rt.base_shape.cols) + inner
              col = laneid % rt.base_shape.cols
            else:
              row = laneid % rt.base_shape.rows
              col = rt.base_shape.stride * (laneid // rt.base_shape.rows) + inner

            srow, scol = base_row + row, base_col + col

            src_i += srow * row_stride + scol

            src_load = srcf[src_i]
            if src.dtype.base != dst.dtype.base:
              src_load = src_load.cast(dst.dtype.base)
            dst_store = dst[*dst_idxs, height, width, inner].store(src_load).end(height, width, inner)
    elif dst_dtype.addrspace == AddrSpace.REG and src_dtype.addrspace == AddrSpace.GLOBAL and isinstance(dst, RV):
      srcf = src.flatten()
      row_stride = prod(src.shape[axis+1:])

      laneid = self.ker.laneid
      rv = cast(RV, dst)
      reductions = rv.base_shape.rows

      assert rv.layout == VecLayout.ORTHO, "only ortho layout supported"

      idxs = tuple(idx * rv.length if i == 3 else idx for i, idx in enumerate(idxs))
      src_i = ((idxs[0] * src.shape[-3] + idxs[1]) * src.shape[-2] + idxs[2]) * src.shape[-1] + idxs[3]

      for outer in self.ker.range(dst.shape[-2], track=False):
        src_i += outer * reductions + (laneid % reductions)

        src_load = srcf[src_i]
        if src.dtype.base != dst.dtype.base:
          src_load = src_load.cast(dst.dtype.base)
        dst_store = dst[outer, 0].store(src_load).end(outer)
    else:
      raise NotImplementedError(f"load from {src_dtype.addrspace} to {dst_dtype.addrspace} not implemented for {type(dst)=}")

    self.ker.push_store(dst_store, dst)
    return dst.after(dst_store).reshape(dst.shape)

  def store(self, dst:ALL_TILES, src:ALL_TILES, idxs:tuple[UOp|int,...]=(), src_idxs:tuple[UOp|int,...]=(), axis:int=0):
    dst, src = cast(UOp, dst), cast(UOp, src)
    assert isinstance(dst.dtype, PtrDType) and isinstance(src.dtype, PtrDType)
    dst_dtype, src_dtype = dst.dtype, src.dtype
    if src_dtype.addrspace == AddrSpace.REG and dst_dtype.addrspace == AddrSpace.LOCAL:
      laneid = self.ker.laneid
      st, rt = cast(ST, dst), cast(RT, src)
      elements_per_thread = rt.base_shape.elements_per_thread

      for height in self.ker.range(src.shape[-3], track=False):
        for width in self.ker.range(src.shape[-2], track=False):
          for inner in self.ker.range(elements_per_thread, track=False):
            if rt.layout != st.layout:
              row = rt.base_shape.stride * (laneid // rt.base_shape.cols) + inner
              col = laneid % rt.base_shape.cols
            else:
              row = laneid % rt.base_shape.rows
              col = rt.base_shape.stride * (laneid // rt.base_shape.rows) + inner

            srow, scol = cast(ST, dst).swizzle(row, col)

            src_load = src[*src_idxs, height, width, inner]
            if src.dtype.base != dst.dtype.base:
              src_load = src_load.cast(dst.dtype.base)
            dst_store = dst[*idxs[:-2], height, width, srow, scol].store(src_load)
            dst_store = dst_store.end(height, width, inner)
    elif src_dtype.addrspace == AddrSpace.REG and dst_dtype.addrspace == AddrSpace.GLOBAL and isinstance(src, RT):
      dstf = dst.flatten()
      row_stride = prod(dst.shape[axis+1:])

      laneid = self.ker.laneid
      rt = cast(RT, src)
      elements_per_thread = rt.base_shape.elements_per_thread

      idxs = tuple(idx * src.shape[-3] * rt.base_shape.rows if i == axis else idx for i, idx in enumerate(idxs))
      idxs = tuple(idx * src.shape[-2] * rt.base_shape.cols if i == 3 else idx for i, idx in enumerate(idxs))
      dst_i = ((idxs[0] * dst.shape[-3] + idxs[1]) * dst.shape[-2] + idxs[2]) * dst.shape[-1] + idxs[3]

      for height in self.ker.range(src.shape[-3], track=False):
        for width in self.ker.range(src.shape[-2], track=False):
          for inner in self.ker.range(elements_per_thread, track=False):
            base_row = height * rt.base_shape.rows
            base_col = width * rt.base_shape.cols

            if rt.layout == TileLayout.COL:
              row = rt.base_shape.stride * (laneid // rt.base_shape.cols) + inner
              col = laneid % rt.base_shape.cols
            else:
              row = laneid % rt.base_shape.rows
              col = rt.base_shape.stride * (laneid // rt.base_shape.rows) + inner

            srow, scol = base_row + row, base_col + col

            dst_i += srow * row_stride + scol

            src_load = src[*src_idxs, height, width, inner]
            if src.dtype.base != dst.dtype.base:
              src_load = src_load.cast(dst.dtype.base)
            dst_store = dstf[dst_i].store(src_load).end(height, width, inner)
    elif src_dtype.addrspace == AddrSpace.REG and dst_dtype.addrspace == AddrSpace.GLOBAL and isinstance(src, RV):
      dstf = dst.flatten()
      row_stride = prod(dst.shape[axis+1:])

      laneid = self.ker.laneid
      rv = cast(RV, src)
      reductions = rv.base_shape.rows

      assert rv.layout == VecLayout.ORTHO, "only ortho layout supported"

      idxs = tuple(idx * rv.length if i == 3 else idx for i, idx in enumerate(idxs))
      dst_i = ((idxs[0] * dst.shape[-3] + idxs[1]) * dst.shape[-2] + idxs[2]) * dst.shape[-1] + idxs[3]

      for outer in self.ker.range(src.shape[-2], track=False):
        dst_i += outer * reductions + (laneid % reductions)

        src_load = src[outer, 0]
        if src.dtype.base != dst.dtype.base:
          src_load = src_load.cast(dst.dtype.base)
        dst_store = dstf[dst_i].store(src_load).end(outer)
    else:
      raise NotImplementedError(f"store from {src_dtype.addrspace} to {dst_dtype.addrspace} not implemented for {type(src)=}")

    self.ker.push_store(dst_store, dst)
    return dst.after(dst_store).reshape(dst.shape)
