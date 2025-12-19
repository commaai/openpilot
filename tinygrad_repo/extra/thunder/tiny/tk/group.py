import math, functools
from typing import cast, Callable
from tinygrad import Tensor, Device, Context, GlobalCounters, dtypes
from tinygrad.uop.ops import AxisType, UOp, KernelInfo, Ops
from tinygrad.engine.realize import ExecItem, get_runner
from tinygrad.dtype import AddrSpace, PtrDType
from tinygrad.helpers import getenv, prod

from extra.thunder.tiny.tk import WARP_THREADS
from extra.thunder.tiny.tk.tiles import RT

class Group:
  def __init__(self, warps:int, ker):
    self.warps = warps
    self.group_threads = warps * WARP_THREADS
    self.threadIdx_x = ker.threadIdx_x
    self.ker = ker

  # helpers
  @property
  def laneid(self): return self.threadIdx_x % self.group_threads
  @property
  def warpid(self): return self.laneid // WARP_THREADS
  @property
  def groupid(self): return self.threadIdx_x // self.group_threads

  # ops that only work on a single warp

  clear_rid = 1000
  def clear(self, reg:UOp, value:float=0):
    assert self.warps == 1

    rngs_for_shape = tuple(UOp.range(dim, Group.clear_rid + i) for i, dim in enumerate(reg.shape))
    Group.clear_rid += len(reg.shape)

    reg_store = reg[*rngs_for_shape].store(value).end(*rngs_for_shape)

    self.ker.push_store(reg_store, reg)
    return reg.after(reg_store).reshape(reg.shape)

  def zero(self, reg:UOp): return self.clear(reg, 0)
  def neg_inf(self, reg:UOp): return self.clear(reg, -math.inf)

  copy_rid = 300
  def copy(self, dst:UOp, src:UOp):
    assert self.warps == 1
    assert dst.shape == src.shape

    rngs_for_shape = tuple(UOp.range(dim, Group.copy_rid + i) for i, dim in enumerate(dst.shape))
    Group.copy_rid += len(dst.shape)

    dst_store = dst[*rngs_for_shape].store(src[*rngs_for_shape].cast(dst.dtype.base)).end(*rngs_for_shape)

    self.ker.push_store(dst_store, dst)
    return dst.after(dst_store).reshape(dst.shape)

  def mma_AB(self, c:UOp, a:UOp, b:UOp, after=True):
    assert self.warps == 1

    for height in self.ker.range(c.shape[-3], track=False):
      for width in self.ker.range(c.shape[-2], track=False):
        for inner in self.ker.range(a.shape[-2], AxisType.REDUCE, track=False):
          wmma_arg = ("WMMA_8_16_16_bfloat16_float", (8, 16, 16), dtypes.bfloat16, dtypes.float, "CUDA", 32, (((4, 2), (3, 2), (8, 2)), ((4, 2), (3, 2)), ((4, 2), (3, 2))), ())

          a_in = UOp.vectorize(*[a[height, inner, i] for i in range(8)])
          b_in1 = UOp.vectorize(*([b[inner, width, i] for i in range(2)] + [b[inner, width, 4+i] for i in range(2)]))
          c_out1 = UOp.vectorize(*[c[height, width, i] for i in range(4)])
          b_in2 = UOp.vectorize(*([b[inner, width, 2+i] for i in range(2)] + [b[inner, width, 6+i] for i in range(2)]))
          c_out2 = UOp.vectorize(*[c[height, width, 4+i] for i in range(4)])

          out1 = UOp(Ops.WMMA, dtypes.float32.vec(4), (a_in, b_in1, c_out1), arg=wmma_arg)
          out2 = UOp(Ops.WMMA, dtypes.float32.vec(4), (a_in, b_in2, c_out2), arg=wmma_arg)
          c_i = [c[height, width, i].store(out1.gep(i)) for i in range(4)] + [c[height, width, 4+i].store(out2.gep(i)) for i in range(4)]
          c_store = UOp.group(*c_i).end(height, width, inner)

    self.ker.push_store(c_store, c)
    return c.after(c_store).reshape(c.shape) if after else c_store

  def mma_ABt(self, c:UOp, a:UOp, b:UOp, after=True):
    assert self.warps == 1

    for height in self.ker.range(c.shape[-3], track=False):
      for width in self.ker.range(c.shape[-2], track=False):
        for inner in self.ker.range(a.shape[-2], AxisType.REDUCE, track=False):
          wmma_arg = ("WMMA_8_16_16_bfloat16_float", (8, 16, 16), dtypes.bfloat16, dtypes.float, "CUDA", 32, (((4, 2), (3, 2), (8, 2)), ((4, 2), (3, 2)), ((4, 2), (3, 2))), ())

          a_in = UOp.vectorize(*[a[height, inner, i] for i in range(8)])
          b_in1 = UOp.vectorize(*([b[width, inner, i] for i in range(2)] + [b[width, inner, 4+i] for i in range(2)]))
          c_out1 = UOp.vectorize(*[c[height, width, i] for i in range(4)])
          b_in2 = UOp.vectorize(*([b[width, inner, 2+i] for i in range(2)] + [b[width, inner, 6+i] for i in range(2)]))
          c_out2 = UOp.vectorize(*[c[height, width, 4+i] for i in range(4)])

          out1 = UOp(Ops.WMMA, dtypes.float32.vec(4), (a_in, b_in1, c_out1), arg=wmma_arg)
          out2 = UOp(Ops.WMMA, dtypes.float32.vec(4), (a_in, b_in2, c_out2), arg=wmma_arg)
          c_i = [c[height, width, i].store(out1.gep(i)) for i in range(4)] + [c[height, width, 4+i].store(out2.gep(i)) for i in range(4)]
          c_store = UOp.group(*c_i).end(height, width, inner)

    self.ker.push_store(c_store, c)
    return c.after(c_store).reshape(c.shape) if after else c_store

  map_rid = 400
  def map(self, a:UOp, op:Callable[[UOp], UOp]|Callable[[UOp, tuple], UOp]):
    assert self.warps == 1

    rngs_for_shape = tuple(UOp.range(dim, Group.map_rid + i) for i, dim in enumerate(a.shape))
    Group.map_rid += len(a.shape)

    if op.__code__.co_argcount == 1:
      to_store = op(a[*rngs_for_shape])
    else:
      to_store = op(a[*rngs_for_shape], rngs_for_shape)

    a_store = a[*rngs_for_shape].store(to_store).end(*rngs_for_shape)

    self.ker.push_store(a_store, a)
    return a.after(a_store).reshape(a.shape)

  def row_reduce(self, vec:UOp, src:UOp, op:Callable[[UOp, UOp], UOp]):
    assert self.warps == 1

    red_local = self.ker.alloc((self.group_threads, 2), src.dtype.base, AddrSpace.LOCAL)
    red_reg = self.ker.alloc((2,), src.dtype.base, AddrSpace.REG)

    for height in self.ker.range(src.shape[-3], track=False):
      i = UOp.range(red_reg.size, Group.clear_rid)
      Group.clear_rid += 1
      red_reg = red_reg.after(height, *[tkr._rng for tkr in self.ker.range_stack])
      reg_store = red_reg.flatten()[i].store(0.).end(i)
      red_reg = red_reg.after(reg_store).reshape(red_reg.shape)

      for outer in self.ker.range(2, track=False):
        for width in self.ker.range(src.shape[-2], AxisType.REDUCE, track=False):
          for inner in self.ker.range(4, AxisType.REDUCE, track=False):
            elem_index = inner + 2 * (inner // 2) + outer * 2
            reg_store = red_reg[outer].store(op(red_reg[outer], src[height, width, elem_index])).end(inner, width, outer)
            red_reg = red_reg.after(reg_store).reshape(red_reg.shape)

      # store to shared memory
      for outer in self.ker.range(2, track=False):
        red_local_store = red_local[self.laneid, outer].store(red_reg[outer]).end(outer)
        red_local = red_local.after(red_local_store.barrier()).reshape(red_local.shape)

      # reduce from shared memory
      for outer in self.ker.range(2, track=False):
        for inner in self.ker.range(3, AxisType.REDUCE, track=False):
          offset = (self.laneid // 4) * 4 + ((self.laneid + inner + 1) % 4)
          reg_store = red_reg[outer].store(op(red_reg[outer], red_local[offset, outer])).end(inner, outer)
          red_reg = red_reg.after(reg_store).reshape(red_reg.shape)

      # reduce with vec
      for outer in self.ker.range(2, track=False):
        vec_store = vec[height, 0, outer].store(op(vec[height, 0, outer], red_reg[outer])).end(outer, height)

    self.ker.push_store(vec_store, vec)
    return vec.after(vec_store).reshape(vec.shape)

  # ops that can work across multiple warps

  LOAD_INNER = 8
  def load(self, dst:UOp, src:UOp, dst_idxs:tuple[UOp|int,...]=(), idxs:tuple[UOp|int,...]=(), axis:int=0, transpose:bool=False):
    assert isinstance(dst.dtype, PtrDType) and isinstance(src.dtype, PtrDType)
    dst_dtype, src_dtype = cast(PtrDType, dst.dtype), cast(PtrDType, src.dtype)
    if dst_dtype.addrspace == AddrSpace.REG and src_dtype.addrspace == AddrSpace.LOCAL:
      srcf = src.flatten(-2)

      if self.warps % 4 == 0: local_warpid = (self.warpid // 4) + (self.warpid % 4) * (self.warps // 4)
      else: local_warpid = self.warpid
      warp_laneid = self.threadIdx_x % WARP_THREADS

      for height in self.ker.range(dst.shape[-3], track=False):
        for width in self.ker.range(dst.shape[-2], track=False):
          for inner in self.ker.range(RT.BASE_TILE_NEPT, track=False):
            if not transpose:
              row = (local_warpid * dst.shape[-3] + height) * RT.TILE_ROW_DIM + (warp_laneid // 4)
              col = width * RT.TILE_COL_DIM + 2 * (warp_laneid % 4)

              row_offset = ((inner % 4) // 2) * 8
              col_offset = (inner % 2) + (inner // 4) * 8
            else:
              row = (local_warpid * dst.shape[-3] + height) * RT.TILE_ROW_DIM + 2 * (warp_laneid % 4)
              col = width * RT.TILE_COL_DIM + (warp_laneid // 4)

              row_offset = (inner % 2) + (inner // 4) * 8
              col_offset = ((inner % 4) // 2) * 8

            src_i_last = (row + row_offset) * src.shape[-1] + col + col_offset

            dst_store = dst[*dst_idxs, height, width, inner].store(srcf[*idxs[:-2], src_i_last])
            dst_store = dst_store.end(height, width, inner)
    elif dst_dtype.addrspace == AddrSpace.LOCAL and src_dtype.addrspace == AddrSpace.GLOBAL:
      dstf = dst.flatten(-2)

      srcf = src.flatten()
      row_stride = prod(src.shape[axis+1:])

      idxs = tuple(idx * dst.shape[-2] if i == axis else idx for i, idx in enumerate(idxs))
      idxs = tuple(idx * dst.shape[-1] if i == 3 else idx for i, idx in enumerate(idxs))
      src_i = ((idxs[0] * src.shape[-3] + idxs[1]) * src.shape[-2] + idxs[2]) * src.shape[-1] + idxs[3]

      memcpy_per_row = dst.shape[-1] // Group.LOAD_INNER
      total_calls = prod(dst.shape[-2:]) // (self.group_threads * Group.LOAD_INNER)

      for outer in self.ker.range(total_calls, track=False):
        for inner in self.ker.range(Group.LOAD_INNER, track=False):
          load_idx = outer * self.group_threads + self.laneid
          row = load_idx // memcpy_per_row
          col = (load_idx * Group.LOAD_INNER) % dst.shape[-1]

          dst_i = row * dst.shape[-1] + col + inner
          src_i += row * row_stride + col + inner

          dst_store = dstf[*dst_idxs, dst_i].store(srcf[src_i]).end(outer, inner)
    else:
      raise NotImplementedError(f"load from {src_dtype.addrspace} to {dst_dtype.addrspace} not implemented")

    return dst.after(dst_store.barrier()).reshape(dst.shape)

  STORE_INNER = 8
  def store(self, dst:UOp, src:UOp, idxs:tuple[UOp|int,...]=(), src_idxs:tuple[UOp|int,...]=(), axis=0, after=True):
    assert isinstance(dst.dtype, PtrDType) and isinstance(src.dtype, PtrDType)
    dst_dtype, src_dtype = cast(PtrDType, dst.dtype), cast(PtrDType, src.dtype)
    if src_dtype.addrspace == AddrSpace.REG and dst_dtype.addrspace == AddrSpace.LOCAL:
      dstf = dst.flatten(-2)

      if self.warps % 4 == 0: local_warpid = (self.warpid // 4) + (self.warpid % 4) * (self.warps // 4)
      else: local_warpid = self.warpid
      warp_laneid = self.threadIdx_x % WARP_THREADS

      for height in self.ker.range(src.shape[-3], track=False):
        for width in self.ker.range(src.shape[-2], track=False):
          for inner in self.ker.range(RT.BASE_TILE_NEPT, track=False):
            row = (local_warpid * src.shape[-3] + height) * RT.TILE_ROW_DIM + (warp_laneid // 4)
            col = width * RT.TILE_COL_DIM + 2 * (warp_laneid % 4)

            row_offset = ((inner % 4) // 2) * 8
            col_offset = (inner % 2) + (inner // 4) * 8

            dst_i_last = (row + row_offset) * dst.shape[-1] + col + col_offset

            dst_store = dstf[*idxs[:-2], dst_i_last].store(src[*src_idxs, height, width, inner])
            dst_store = dst_store.end(height, width, inner)
    elif src_dtype.addrspace == AddrSpace.LOCAL and dst_dtype.addrspace == AddrSpace.GLOBAL:
      dstf = dst.flatten()
      row_stride = prod(dst.shape[axis+1:])

      idxs = tuple(idx * src.shape[-2] if i == axis else idx for i, idx in enumerate(idxs))
      idxs = tuple(idx * src.shape[-1] if i == 3 else idx for i, idx in enumerate(idxs))
      dst_i = ((idxs[0] * dst.shape[-3] + idxs[1]) * dst.shape[-2] + idxs[2]) * dst.shape[-1] + idxs[3]

      srcf = src.flatten(-2)

      memcpy_per_row = src.shape[-1] // Group.STORE_INNER
      total_calls = prod(src.shape[-2:]) // (self.group_threads * Group.STORE_INNER)

      for outer in self.ker.range(total_calls, track=False):
        for inner in self.ker.range(Group.STORE_INNER, track=False):
          load_idx = outer * self.group_threads + self.laneid
          row = load_idx // memcpy_per_row
          col = (load_idx * Group.STORE_INNER) % src.shape[-1]

          src_i = row * src.shape[-1] + col + inner
          dst_i += row * row_stride + col + inner

          dst_store = dstf[dst_i].store(srcf[*src_idxs, src_i]).end(outer, inner)
    else:
      raise NotImplementedError(f"store from {src_dtype.addrspace} to {dst_dtype.addrspace} not implemented")

    self.ker.push_store(dst_store, dst)
    return dst.after(dst_store.barrier()).reshape(dst.shape) if after else dst_store
