import unittest, math, time

from tinygrad import Tensor, Device, dtypes, Context
from tinygrad.uop.ops import UOp, Ops
from tinygrad.engine.realize import get_runner
from tinygrad.engine.schedule import ExecItem
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import CI
import numpy as np

from extra.thunder.tiny.tk import WARP_THREADS
from extra.thunder.tiny.tk.kernel import Kernel
from extra.thunder.tiny.tk.tiles import ST_16X32, RT_16X32, RT_16X16, TileLayout

@unittest.skipIf(CI or Device.DEFAULT not in ["AMD"], "only amd")
class TestTK(unittest.TestCase):
  def setUp(self):
    arch = Device["AMD"].arch
    if not arch.startswith("gfx9"):
      self.skipTest(f"arch {arch} not supported")

  @unittest.skipIf(CI, "no wmma in ci")
  def test_simple_matmul(self):
    N = 8192
    BLOCK_SIZE = 64
    with Kernel("simple_matmul", (N // BLOCK_SIZE, N // BLOCK_SIZE, 1), WARP_THREADS) as ker:
      warp = ker.warp

      c = ker.gl((1, 1, N, N), dtypes.float32)
      a = ker.gl((1, 1, N, N), dtypes.bfloat16)
      b = ker.gl((1, 1, N, N), dtypes.bfloat16)

      a_smem = ker.st((BLOCK_SIZE, BLOCK_SIZE), dtypes.bfloat16)
      b_smem = ker.st((BLOCK_SIZE, BLOCK_SIZE), dtypes.bfloat16)
      c_smem = ker.st((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      a_reg = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.bfloat16)
      b_reg = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.bfloat16, TileLayout.COL)
      c_reg_col = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32, TileLayout.COL)
      c_reg = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      col, row = ker.blockIdx_x, ker.blockIdx_y

      c_reg_col = warp.zero(c_reg_col)
      for tile in ker.range(N // BLOCK_SIZE):
        a_smem = warp.load(a_smem, a, (), (0, 0, row, tile), axis=2)
        b_smem = warp.load(b_smem, b, (), (0, 0, tile, col), axis=2)

        a_reg = warp.load(a_reg, a_smem)
        b_reg = warp.load(b_reg, b_smem)

        c_reg_col = warp.mma_AB(c_reg_col, a_reg, b_reg)
      c_reg_col = ker.endrange()

      c_smem = warp.store(c_smem, c_reg_col)
      c_reg = warp.load(c_reg, c_smem)

      c = warp.store(c, c_reg, (0, 0, row, col), (), axis=2)

      sink = ker.finish()

    with Context(DEBUG=0):
      a = Tensor.rand(1, 1, N, N, dtype="bfloat16").contiguous()
      b = Tensor.rand(1, 1, N, N, dtype="bfloat16").contiguous()
      c = Tensor.empty(1, 1, N, N, dtype="float32")
      Tensor.realize(a, b, c)

    ei = ExecItem(sink, [t.uop.buffer for t in (c, a, b)], prg=get_runner(Device.DEFAULT, sink))
    for _ in range(5): ei.run(wait=True)
    c = c.float()

    ref = a.matmul(b, dtype=dtypes.float32).float()

    np.testing.assert_allclose(c.numpy(), ref.numpy())

  @unittest.skipIf(CI, "no wmma in ci")
  def test_simple_matmul_transposed(self):
    N = 8192
    BLOCK_N, BLOCK_M, BLOCK_K = 64, 64, 128
    with Kernel("simple_matmul_transposed", (N // BLOCK_N, N // BLOCK_M, 1), WARP_THREADS) as ker:
      warp = ker.warp

      c = ker.gl((1, 1, N, N), dtypes.float32)
      a = ker.gl((1, 1, N, N), dtypes.bfloat16)
      b = ker.gl((1, 1, N, N), dtypes.bfloat16)

      a_smem = ker.st((BLOCK_N, BLOCK_K), dtypes.bfloat16, base_shape=ST_16X32)
      b_smem = ker.st((BLOCK_M, BLOCK_K), dtypes.bfloat16, base_shape=ST_16X32)

      a_reg = ker.rt((BLOCK_N, BLOCK_K), dtypes.bfloat16, base_shape=RT_16X32)
      b_reg = ker.rt((BLOCK_M, BLOCK_K), dtypes.bfloat16, base_shape=RT_16X32)
      c_reg = ker.rt((BLOCK_N, BLOCK_M), dtypes.float32, TileLayout.COL, base_shape=RT_16X16)

      col, row = ker.blockIdx_x, ker.blockIdx_y

      c_reg = warp.zero(c_reg)
      for tile in ker.range(N // BLOCK_K):
        a_smem = warp.load(a_smem, a, (), (0, 0, row, tile), axis=2)
        b_smem = warp.load(b_smem, b, (), (0, 0, col, tile), axis=2)

        a_reg = warp.load(a_reg, a_smem)
        b_reg = warp.load(b_reg, b_smem)

        c_reg = warp.mma_ABt(c_reg, a_reg, b_reg)
      c_reg = ker.endrange()

      c = warp.store(c, c_reg, (0, 0, row, col), (), axis=2)

      sink = ker.finish()

    with Context(DEBUG=0):
      a = Tensor.rand(1, 1, N, N, dtype="bfloat16").contiguous()
      b = Tensor.rand(1, 1, N, N, dtype="bfloat16").contiguous()
      c = Tensor.empty(1, 1, N, N, dtype="float32")
      Tensor.realize(a, b, c)

    ei = ExecItem(sink, [t.uop.buffer for t in (c, a, b)], prg=get_runner(Device.DEFAULT, sink))
    for _ in range(5): ei.run(wait=True)
    c = c.float()

    ref = a.matmul(b.transpose(2, 3), dtype=dtypes.float32).float()

    np.testing.assert_allclose(c.numpy(), ref.numpy())

  def test_load_store(self):
    N = 64
    BLOCK_SIZE = 32
    with Kernel("load_store", (N // BLOCK_SIZE, N // BLOCK_SIZE, 1), WARP_THREADS) as ker:
      warp = ker.warp

      b = ker.gl((1, 1, N, N), dtypes.float32)
      a = ker.gl((1, 1, N, N), dtypes.float32)

      a_smem = ker.st((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      a_reg = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)
      b_reg = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      col, row = ker.blockIdx_x, ker.blockIdx_y

      a_smem = warp.load(a_smem, a, (), (0, 0, row, col), axis=2)
      a_reg = warp.load(a_reg, a_smem)
      b_reg = warp.copy(b_reg, a_reg)
      b = warp.store(b, b_reg, (0, 0, row, col), (), axis=2)

      sink = ker.finish()

    with Context(DEBUG=0):
      a = Tensor.rand(1, 1, N, N, dtype="float32").contiguous()
      b = Tensor.empty(1, 1, N, N, dtype="float32")
      Tensor.realize(a, b)

    ei = ExecItem(sink, [t.uop.buffer for t in (b, a)], prg=get_runner(Device.DEFAULT, sink))
    for _ in range(5): ei.run(wait=True)
    b = b.float()

    ref = a.float()

    np.testing.assert_allclose(b.numpy(), ref.numpy())

  def test_load_store_local_hop(self):
    N = 64
    BLOCK_SIZE = 32
    with Kernel("load_store_local_hop", (N // BLOCK_SIZE, N // BLOCK_SIZE, 1), WARP_THREADS) as ker:
      warp = ker.warp

      b = ker.gl((1, 1, N, N), dtypes.float32)
      a = ker.gl((1, 1, N, N), dtypes.float32)

      a_smem = ker.st((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)
      b_smem = ker.st((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      a_reg = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)
      b_reg = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      col, row = ker.blockIdx_x, ker.blockIdx_y

      a_smem = warp.load(a_smem, a, (), (0, 0, row, col), axis=2)
      a_reg = warp.load(a_reg, a_smem)
      b_reg = warp.copy(b_reg, a_reg)
      b_smem = warp.store(b_smem, b_reg)
      b_reg = warp.load(b_reg, b_smem)
      b = warp.store(b, b_reg, (0, 0, row, col), (), axis=2)

      sink = ker.finish()

    with Context(DEBUG=0):
      a = Tensor.rand(1, 1, N, N, dtype="float32").contiguous()
      b = Tensor.empty(1, 1, N, N, dtype="float32")
      Tensor.realize(a, b)

    ei = ExecItem(sink, [t.uop.buffer for t in (b, a)], prg=get_runner(Device.DEFAULT, sink))
    for _ in range(5): ei.run(wait=True)
    b = b.float()

    ref = a.float()

    np.testing.assert_allclose(b.numpy(), ref.numpy())

  def test_load_store_multioutput(self):
    N = 64
    BLOCK_SIZE = 32
    with Kernel("load_store_multioutput", (N // BLOCK_SIZE, N // BLOCK_SIZE, 1), WARP_THREADS) as ker:
      warp = ker.warp

      b = ker.gl((1, 1, N, N), dtypes.float32)
      c = ker.gl((1, 1, N, N), dtypes.float32)
      a = ker.gl((1, 1, N, N), dtypes.float32)

      a_smem = ker.st((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)
      b_smem = ker.st((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      a_reg = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)
      b_reg = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      col, row = ker.blockIdx_x, ker.blockIdx_y

      a_smem = warp.load(a_smem, a, (), (0, 0, row, col), axis=2)
      a_reg = warp.load(a_reg, a_smem)
      b_reg = warp.copy(b_reg, a_reg)
      b_smem = warp.store(b_smem, b_reg)
      b_reg = warp.load(b_reg, b_smem)
      b = warp.store(b, b_reg, (0, 0, row, col), (), axis=2)
      c = warp.store(c, b_reg, (0, 0, row, col), (), axis=2)

      sink = ker.finish(2)

    with Context(DEBUG=0):
      a = Tensor.rand(1, 1, N, N, dtype="float32").contiguous()
      b = Tensor.empty(1, 1, N, N, dtype="float32")
      c = Tensor.empty(1, 1, N, N, dtype="float32")
      Tensor.realize(a, b, c)

    ei = ExecItem(sink, [t.uop.buffer for t in (b, c, a)], prg=get_runner(Device.DEFAULT, sink))
    for _ in range(5): ei.run(wait=True)
    b = b.float()
    c = c.float()

    ref = a.float()

    np.testing.assert_allclose(b.numpy(), ref.numpy())
    np.testing.assert_allclose(c.numpy(), ref.numpy())

  def test_load_store_group(self):
    N = 1024
    BLOCK_SIZE = 64
    NUM_WORKERS = 4
    with Kernel("load_store_group", (N // (BLOCK_SIZE * NUM_WORKERS), N // BLOCK_SIZE, 1), WARP_THREADS * NUM_WORKERS) as ker:
      warp = ker.warp
      group = ker.group(NUM_WORKERS)

      b = ker.gl((1, 1, N, N), dtypes.float32)
      a = ker.gl((1, 1, N, N), dtypes.float32)

      a_smem = ker.st((BLOCK_SIZE, BLOCK_SIZE * NUM_WORKERS), dtypes.float32)

      a_reg = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)
      b_reg = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      col, row = ker.blockIdx_x, ker.blockIdx_y

      a_smem = group.load(a_smem, a, (), (0, 0, row, col), axis=2)
      a_reg = warp.load(a_reg, a_smem, (), (0, ker.warpid,))
      b_reg = warp.copy(b_reg, a_reg)
      b = warp.store(b, b_reg, (0, 0, row, col * NUM_WORKERS + ker.warpid), (), axis=2)

      sink = ker.finish()

    with Context(DEBUG=0):
      a = Tensor.rand(1, 1, N, N, dtype="float32").contiguous()
      b = Tensor.empty(1, 1, N, N, dtype="float32")
      Tensor.realize(a, b)

    ei = ExecItem(sink, [t.uop.buffer for t in (b, a)], prg=get_runner(Device.DEFAULT, sink))
    for _ in range(5): ei.run(wait=True)
    b = b.float()

    ref = a.float()

    np.testing.assert_allclose(b.numpy(), ref.numpy())

  def test_add(self):
    N = 64
    BLOCK_SIZE = 32
    with Kernel("add", (1, 1, 1), WARP_THREADS) as ker:
      warp = ker.warp

      b = ker.gl((1, 1, N, N), dtypes.float32)
      a = ker.gl((1, 1, N, N), dtypes.float32)

      a_smem = ker.st((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      a_reg = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      for tile_row in ker.range(N // BLOCK_SIZE):
        for tile_col in ker.range(N // BLOCK_SIZE):
          a_smem = warp.load(a_smem, a, (), (0, 0, tile_row, tile_col), axis=2)
          a_reg = warp.load(a_reg, a_smem)

          a_reg += 1

          b = warp.store(b, a_reg, (0, 0, tile_row, tile_col), (), axis=2)

      sink = ker.finish()

      with Context(DEBUG=0):
        a = Tensor.rand(1, 1, N, N, dtype="float32").contiguous()
        b = Tensor.empty(1, 1, N, N, dtype="float32")
        Tensor.realize(a, b)

      ei = ExecItem(sink, [t.uop.buffer for t in (b, a)], prg=get_runner(Device.DEFAULT, sink))
      for _ in range(5): ei.run(wait=True)
      b = b.float()

      ref = a.float() + 1

      np.testing.assert_allclose(b.numpy(), ref.numpy())

  def test_max(self):
    N = 64
    BLOCK_SIZE = 32
    with Kernel("max", (1, 1, 1), WARP_THREADS) as ker:
      warp = ker.warp

      b = ker.gl((1, 1, N, N), dtypes.float32)
      a = ker.gl((1, 1, N, N), dtypes.float32)

      a_smem = ker.st((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      a_reg = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32, TileLayout.COL)
      b_reg = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32, TileLayout.COL)

      max_reg = ker.rv(BLOCK_SIZE, dtypes.float32)

      for tile_col in ker.range(N // BLOCK_SIZE):
        max_reg = warp.neg_inf(max_reg.after(tile_col))

        for tile_row in ker.range(N // BLOCK_SIZE):
          a_smem = warp.load(a_smem, a, (), (0, 0, tile_row, tile_col), axis=2)
          a_reg = warp.load(a_reg, a_smem)
          max_reg = warp.col_reduce(max_reg, a_reg, lambda a, b: a.maximum(b), init_value=-math.inf)
        max_reg = ker.endrange()

        b_reg = warp.map(b_reg, lambda _, idx: max_reg[idx[1], 0])

        for tile_row in ker.range(N // BLOCK_SIZE):
          b = warp.store(b, b_reg, (0, 0, tile_row, tile_col), (), axis=2)

      sink = ker.finish()

    with Context(DEBUG=0):
      a = Tensor.rand(1, 1, N, N, dtype="float32").contiguous()
      b = Tensor.empty(1, 1, N, N, dtype="float32")
      Tensor.realize(a, b)

    ei = ExecItem(sink, [t.uop.buffer for t in (b, a)], prg=get_runner(Device.DEFAULT, sink))
    for _ in range(5): ei.run(wait=True)
    b = b.float()

    ref = a.float().max(axis=2, keepdim=True).expand(a.shape)

    np.testing.assert_allclose(b.numpy(), ref.numpy())

  def test_max_nonsquare(self):
    N, M = 32, 128
    BLOCK_N, BLOCK_M = 16, 64
    with Kernel("max_nonsquare", (1, 1, 1), WARP_THREADS) as ker:
      warp = ker.warp

      b = ker.gl((1, 1, N, M), dtypes.float32)
      a = ker.gl((1, 1, N, M), dtypes.float32)

      a_smem = ker.st((BLOCK_N, BLOCK_M), dtypes.float32)

      a_reg = ker.rt((BLOCK_N, BLOCK_M), dtypes.float32, TileLayout.COL)
      b_reg = ker.rt((BLOCK_N, BLOCK_M), dtypes.float32, TileLayout.COL)

      max_reg = ker.rv(BLOCK_M, dtypes.float32)

      for tile_col in ker.range(M // BLOCK_M):
        max_reg = warp.neg_inf(max_reg.after(tile_col))

        for tile_row in ker.range(N // BLOCK_N):
          a_smem = warp.load(a_smem, a, (), (0, 0, tile_row, tile_col), axis=2)
          a_reg = warp.load(a_reg, a_smem)
          max_reg = warp.col_reduce(max_reg, a_reg, lambda a, b: a.maximum(b), init_value=-math.inf)
        max_reg = ker.endrange()

        b_reg = warp.map(b_reg, lambda _, idx: max_reg[idx[1], 0])

        for tile_row in ker.range(N // BLOCK_N):
          b = warp.store(b, b_reg, (0, 0, tile_row, tile_col), (), axis=2)

      sink = ker.finish()

    with Context(DEBUG=0):
      a = Tensor.rand(1, 1, N, M, dtype="float32").contiguous()
      b = Tensor.empty(1, 1, N, M, dtype="float32")
      Tensor.realize(a, b)

    ei = ExecItem(sink, [t.uop.buffer for t in (b, a)], prg=get_runner(Device.DEFAULT, sink))
    for _ in range(5): ei.run(wait=True)
    b = b.float()

    ref = a.float().max(axis=2, keepdim=True).expand(a.shape)

    np.testing.assert_allclose(b.numpy(), ref.numpy())

  def test_sum(self):
    N = 64
    BLOCK_SIZE = 32
    with Kernel("sum", (1, 1, 1), WARP_THREADS) as ker:
      warp = ker.warp

      b = ker.gl((1, 1, N, N), dtypes.float32)
      a = ker.gl((1, 1, N, N), dtypes.float32)

      a_smem = ker.st((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      a_reg = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32, TileLayout.COL)
      b_reg = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32, TileLayout.COL)

      sum_reg = ker.rv(BLOCK_SIZE, dtypes.float32)

      for tile_col in ker.range(N // BLOCK_SIZE):
        sum_reg = warp.zero(sum_reg.after(tile_col))

        for tile_row in ker.range(N // BLOCK_SIZE):
          a_smem = warp.load(a_smem, a, (), (0, 0, tile_row, tile_col), axis=2)
          a_reg = warp.load(a_reg, a_smem)
          sum_reg = warp.col_reduce(sum_reg, a_reg, lambda a, b: a + b)
        sum_reg = ker.endrange()

        b_reg = warp.map(b_reg, lambda _, idx: sum_reg[idx[1], 0])

        for tile_row in ker.range(N // BLOCK_SIZE):
          b = warp.store(b, b_reg, (0, 0, tile_row, tile_col), (), axis=2)

      sink = ker.finish()

    with Context(DEBUG=0):
      a = Tensor.rand(1, 1, N, N, dtype="float32").contiguous()
      b = Tensor.empty(1, 1, N, N, dtype="float32")
      Tensor.realize(a, b)

    ei = ExecItem(sink, [t.uop.buffer for t in (b, a)], prg=get_runner(Device.DEFAULT, sink))
    for _ in range(5): ei.run(wait=True)
    b = b.float()

    ref = a.float().sum(axis=2, keepdim=True).expand(a.shape)

    np.testing.assert_allclose(b.numpy(), ref.numpy(), atol=1e-5, rtol=1e-5)

  def test_sum_nonsquare(self):
    N, M = 32, 128
    BLOCK_N, BLOCK_M = 16, 64
    with Kernel("sum_nonsquare", (1, 1, 1), WARP_THREADS) as ker:
      warp = ker.warp

      b = ker.gl((1, 1, N, M), dtypes.float32)
      a = ker.gl((1, 1, N, M), dtypes.float32)

      a_smem = ker.st((BLOCK_N, BLOCK_M), dtypes.float32)

      a_reg = ker.rt((BLOCK_N, BLOCK_M), dtypes.float32, TileLayout.COL)
      b_reg = ker.rt((BLOCK_N, BLOCK_M), dtypes.float32, TileLayout.COL)

      sum_reg = ker.rv(BLOCK_M, dtypes.float32)

      for tile_col in ker.range(M // BLOCK_M):
        sum_reg = warp.zero(sum_reg.after(tile_col))

        for tile_row in ker.range(N // BLOCK_N):
          a_smem = warp.load(a_smem, a, (), (0, 0, tile_row, tile_col), axis=2)
          a_reg = warp.load(a_reg, a_smem)
          sum_reg = warp.col_reduce(sum_reg, a_reg, lambda a, b: a + b)
        sum_reg = ker.endrange()

        b_reg = warp.map(b_reg, lambda _, idx: sum_reg[idx[1], 0])

        for tile_row in ker.range(N // BLOCK_N):
          b = warp.store(b, b_reg, (0, 0, tile_row, tile_col), (), axis=2)

      sink = ker.finish()

    with Context(DEBUG=0):
      a = Tensor.rand(1, 1, N, M, dtype="float32").contiguous()
      b = Tensor.empty(1, 1, N, M, dtype="float32")
      Tensor.realize(a, b)

    ei = ExecItem(sink, [t.uop.buffer for t in (b, a)], prg=get_runner(Device.DEFAULT, sink))
    for _ in range(5): ei.run(wait=True)
    b = b.float()

    ref = a.float().sum(axis=2, keepdim=True).expand(a.shape)

    np.testing.assert_allclose(b.numpy(), ref.numpy(), atol=1e-5, rtol=1e-5)

  def test_softmax(self):
    N = 64
    BLOCK_SIZE = 32
    with Kernel("softmax", (1, 1, 1), WARP_THREADS) as ker:
      warp = ker.warp

      b = ker.gl((1, 1, BLOCK_SIZE, N), dtypes.float32)
      a = ker.gl((1, 1, BLOCK_SIZE, N), dtypes.float32)

      a_smem = ker.st((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      a_reg = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      max_vec_last = ker.rv(BLOCK_SIZE, dtypes.float32)
      max_vec = ker.rv(BLOCK_SIZE, dtypes.float32)
      norm_vec = ker.rv(BLOCK_SIZE, dtypes.float32)

      max_vec = warp.neg_inf(max_vec)
      norm_vec = warp.zero(norm_vec)

      for tile_col in ker.range(N // BLOCK_SIZE):
        a_smem_ = warp.load(a_smem, a, (), (0, 0, 0, tile_col), axis=2)
        a_reg_ = warp.load(a_reg, a_smem_)

        a_reg_ *= 1.0 / math.log(2)

        max_vec_last = warp.copy(max_vec_last.after(tile_col), max_vec)
        max_vec = warp.row_reduce(max_vec.after(max_vec_last), a_reg_, lambda a, b: a.maximum(b), init_value=-math.inf)
        a_reg_ = (a_reg_ - max_vec).exp2()
        max_vec_last = (max_vec_last - max_vec).exp2()
        norm_vec *= max_vec_last
        norm_vec = warp.row_reduce(norm_vec, a_reg_, lambda a, b: a + b)
      norm_vec = ker.endrange()
      max_vec = max_vec.after(norm_vec)

      for tile_col in ker.range(N // BLOCK_SIZE):
        a_smem_ = warp.load(a_smem, a, (), (0, 0, 0, tile_col), axis=2)
        a_reg_ = warp.load(a_reg, a_smem_)

        a_reg_ *= 1.0 / math.log(2)
        a_reg_ = (a_reg_ - max_vec).exp2()
        a_reg_ /= norm_vec

        b = warp.store(b, a_reg_, (0, 0, 0, tile_col), (), axis=2)

      sink = ker.finish()

    with Context(DEBUG=0):
      a = Tensor.rand(1, 1, BLOCK_SIZE, N, dtype="float32")
      b = Tensor.empty(1, 1, BLOCK_SIZE, N, dtype="float32")
      Tensor.realize(a, b)

    ei = ExecItem(sink, [t.uop.buffer for t in (b, a)], prg=get_runner(Device.DEFAULT, sink))
    for _ in range(5): ei.run(wait=True)
    b = b.float()

    ref = a.float().softmax(axis=3)

    np.testing.assert_allclose(b.numpy(), ref.numpy(), atol=1e-5, rtol=1e-5)

  def test_softmax_col(self):
    N = 64
    BLOCK_SIZE = 32
    with Kernel("softmax_col", (1, 1, 1), WARP_THREADS) as ker:
      warp = ker.warp

      b = ker.gl((1, 1, N, BLOCK_SIZE), dtypes.float32)
      a = ker.gl((1, 1, N, BLOCK_SIZE), dtypes.float32)

      a_smem = ker.st((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32)

      a_reg = ker.rt((BLOCK_SIZE, BLOCK_SIZE), dtypes.float32, TileLayout.COL)

      max_vec_last = ker.rv(BLOCK_SIZE, dtypes.float32)
      max_vec = ker.rv(BLOCK_SIZE, dtypes.float32)
      norm_vec = ker.rv(BLOCK_SIZE, dtypes.float32)

      max_vec = warp.neg_inf(max_vec)
      norm_vec = warp.zero(norm_vec)

      for tile_row in ker.range(N // BLOCK_SIZE):
        a_smem_ = warp.load(a_smem, a, (), (0, 0, tile_row, 0), axis=2)
        a_reg_ = warp.load(a_reg, a_smem_)

        a_reg_ *= 1.0 / math.log(2)

        max_vec_last = warp.copy(max_vec_last.after(tile_row), max_vec)
        max_vec = warp.col_reduce(max_vec.after(max_vec_last), a_reg_, lambda a, b: a.maximum(b), init_value=-math.inf)
        a_reg_ = (a_reg_ - max_vec).exp2()
        max_vec_last = (max_vec_last - max_vec).exp2()
        norm_vec *= max_vec_last
        norm_vec = warp.col_reduce(norm_vec, a_reg_, lambda a, b: a + b)
      norm_vec = ker.endrange()
      max_vec = max_vec.after(norm_vec)

      for tile_row in ker.range(N // BLOCK_SIZE):
        a_smem_ = warp.load(a_smem, a, (), (0, 0, tile_row, 0), axis=2)
        a_reg_ = warp.load(a_reg.after(norm_vec), a_smem_)

        a_reg_ *= 1.0 / math.log(2)
        a_reg_ = (a_reg_ - max_vec).exp2()
        a_reg_ /= norm_vec

        b = warp.store(b, a_reg_, (0, 0, tile_row, 0), (), axis=2)

      sink = ker.finish()

    with Context(DEBUG=0):
      a = Tensor.rand(1, 1, N, BLOCK_SIZE, dtype="float32")
      b = Tensor.empty(1, 1, N, BLOCK_SIZE, dtype="float32")
      Tensor.realize(a, b)

    ei = ExecItem(sink, [t.uop.buffer for t in (b, a)], prg=get_runner(Device.DEFAULT, sink))
    for _ in range(5): ei.run(wait=True)
    b = b.float()

    ref = a.float().softmax(axis=2)

    np.testing.assert_allclose(b.numpy(), ref.numpy(), atol=1e-5, rtol=1e-5)

  def test_fa(self):
    NUM_WORKERS = 1
    B, N, H, H_KV, D = 2, 8192, 32, 8, 128
    Q_BLOCK_SIZE = 16
    KV_BLOCK_SIZE = 16
    GROUP_SIZE = H // H_KV
    with Kernel("fa", (H, N // (Q_BLOCK_SIZE*NUM_WORKERS), B), NUM_WORKERS * WARP_THREADS) as ker:
      warp = ker.warp

      # kernel
      o = ker.gl((B, N, H, D), dtypes.bfloat16)
      q = ker.gl((B, N, H, D), dtypes.bfloat16)
      k = ker.gl((B, N, H_KV, D), dtypes.bfloat16)
      v = ker.gl((B, N, H_KV, D), dtypes.bfloat16)

      head = ker.blockIdx_x
      head_kv = head // GROUP_SIZE
      batch = ker.blockIdx_z
      q_seq = ker.blockIdx_y * NUM_WORKERS + ker.warpid

      k_smem = ker.st((KV_BLOCK_SIZE, D), dtypes.bfloat16)
      v_smem = ker.st((KV_BLOCK_SIZE, D), dtypes.bfloat16)

      q_reg_fl = ker.rt((Q_BLOCK_SIZE, D), dtypes.float32)
      q_reg = ker.rt((Q_BLOCK_SIZE, D), dtypes.bfloat16)
      q_reg_transposed = ker.rt((D, Q_BLOCK_SIZE), dtypes.bfloat16, TileLayout.COL)
      k_reg = ker.rt((KV_BLOCK_SIZE, D), dtypes.bfloat16)
      k_reg_transposed = ker.rt((D, KV_BLOCK_SIZE), dtypes.bfloat16, TileLayout.COL)
      v_reg = ker.rt((KV_BLOCK_SIZE, D), dtypes.bfloat16, TileLayout.COL)
      o_reg = ker.rt((D, Q_BLOCK_SIZE), dtypes.float32, TileLayout.COL)
      o_reg_transposed = ker.rt((Q_BLOCK_SIZE, D), dtypes.float32)
      att_block = ker.rt((KV_BLOCK_SIZE, Q_BLOCK_SIZE), dtypes.float32, TileLayout.COL)
      att_block_mma = ker.rt((KV_BLOCK_SIZE, Q_BLOCK_SIZE), dtypes.bfloat16, TileLayout.COL)
      max_vec_last = ker.rv(KV_BLOCK_SIZE, dtypes.float32)
      max_vec = ker.rv(KV_BLOCK_SIZE, dtypes.float32)
      norm_vec = ker.rv(KV_BLOCK_SIZE, dtypes.float32)
      scale_vec = ker.rv(KV_BLOCK_SIZE, dtypes.float32)

      max_vec = warp.neg_inf(max_vec)
      norm_vec = warp.zero(norm_vec)
      o_reg = warp.zero(o_reg)
      scale_vec = warp.ones(scale_vec)

      # load q tile
      q_reg_fl = warp.load(q_reg_fl, q, (), (batch, q_seq, head, 0), axis=1)
      q_reg_fl *= (1.0 / math.sqrt(D)) * (1.0 / math.log(2))
      q_reg = warp.copy(q_reg, q_reg_fl)
      q_reg_transposed = warp.transpose(q_reg_transposed, q_reg)

      for kv_idx in ker.range(N // KV_BLOCK_SIZE):
        k_smem = warp.load(k_smem, k, (), (batch, kv_idx, head_kv, 0), axis=1)
        v_smem = warp.load(v_smem, v, (), (batch, kv_idx, head_kv, 0), axis=1)

        k_reg = warp.load(k_reg, k_smem)
        v_reg = warp.load(v_reg, v_smem)

        # mma qk^t
        att_block = warp.zero(att_block.after(kv_idx))
        k_reg_transposed = warp.transpose(k_reg_transposed, k_reg)
        att_block = warp.mma_AtB(att_block, k_reg_transposed, q_reg_transposed)

        # mask for causal
        q_base = q_seq * Q_BLOCK_SIZE + (warp.laneid % 16)
        kv_base = kv_idx * KV_BLOCK_SIZE + (warp.laneid // 16) * 4
        att_block = warp.map(att_block,
                             lambda x, idx: ((kv_base + idx[0]*16 + idx[2]) > (q_base + idx[1]*16)).alu(Ops.WHERE, UOp.ufix(x._uop, -math.inf), x))

        # softmax
        max_vec_last = warp.copy(max_vec_last.after(kv_idx), max_vec)
        max_vec = warp.row_reduce(max_vec.after(max_vec_last), att_block, lambda a, b: a.maximum(b), init_value=-math.inf)

        scale_vec = warp.map(scale_vec.after(max_vec_last, max_vec), lambda _, idx: max_vec_last[*idx] - max_vec[*idx])
        scale_vec = scale_vec.exp2()

        o_reg *= scale_vec
        norm_vec *= scale_vec

        att_block -= max_vec
        att_block = att_block.exp2()

        norm_vec = warp.row_reduce(norm_vec.after(scale_vec), att_block, lambda a, b: a + b)

        # mma av
        att_block_mma = warp.copy(att_block_mma.after(kv_idx, norm_vec), att_block)
        o_reg = warp.mma_AtB(o_reg, v_reg, att_block_mma)
      o_reg = ker.endrange()
      norm_vec = norm_vec.after(o_reg)

      o_reg /= norm_vec

      o_reg_transposed = warp.transpose(o_reg_transposed, o_reg)
      o = warp.store(o, o_reg_transposed, (batch, q_seq, head, 0), (), axis=1)

      sink = ker.finish()

    with Context(DEBUG=0):
      q = Tensor.randn(B, N, H, D, dtype=dtypes.bfloat16).contiguous()
      k = Tensor.randn(B, N, H_KV, D, dtype=dtypes.bfloat16).contiguous()
      v = Tensor.randn(B, N, H_KV, D, dtype=dtypes.bfloat16).contiguous()
      out = Tensor.empty(B, N, H, D, dtype=dtypes.bfloat16)
      Tensor.realize(q, k, v, out)

    ei = ExecItem(sink, [t.uop.buffer for t in (out, q, k, v)], prg=get_runner(Device.DEFAULT, sink))
    for _ in range(5):
      et = ei.run(wait=True)
      attn_flops = 2 * B * H * N * N * D + \
                   4 * B * H * N * N + \
                   2 * B * H * N * N * D
      print(f"{attn_flops/(et*1e9):2f} GFLOPS")
    out = out.float()

    q_permuted = q.permute(0, 2, 1, 3)
    k_permuted = k.permute(0, 2, 1, 3)
    v_permuted = v.permute(0, 2, 1, 3)
    ref = q_permuted.scaled_dot_product_attention(k_permuted, v_permuted, is_causal=True, enable_gqa=True).float()
    ref = ref.permute(0, 2, 1, 3)

    np.testing.assert_allclose(out.numpy(), ref.numpy(), atol=2e-2, rtol=2e-2)

  def test_fast_fa(self):
    from extra.thunder.tiny.fa import flash_attention

    B, N, H, H_KV, D = 2, 8192, 32, 8, 128

    with Context(DEBUG=0):
      q = Tensor.randn(B, N, H, D, dtype=dtypes.bfloat16).contiguous()
      k = Tensor.randn(B, N, H_KV, D, dtype=dtypes.bfloat16).contiguous()
      v = Tensor.randn(B, N, H_KV, D, dtype=dtypes.bfloat16).contiguous()
      Tensor.realize(q, k, v)

    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    fa_jitted = TinyJit(flash_attention)

    for _ in range(10):
      st = time.perf_counter()
      out = fa_jitted(q, k, v, is_causal=False)
      et = time.perf_counter() - st
      attn_flops = 2 * B * H * N * N * D + \
                   4 * B * H * N * N + \
                   2 * B * H * N * N * D
      print(f"{attn_flops/(et*1e9):2f} GFLOPS")
    out = out.float().transpose(1, 2)

    ref = q.scaled_dot_product_attention(k, v, is_causal=False, enable_gqa=True).float().transpose(1, 2)

    np.testing.assert_allclose(out.numpy(), ref.numpy(), atol=2e-2, rtol=2e-2)

  def test_fast_fa_causal(self):
    from extra.thunder.tiny.fa import flash_attention

    B, N, H, H_KV, D = 2, 8192, 32, 8, 128

    with Context(DEBUG=0):
      q = Tensor.randn(B, N, H, D, dtype=dtypes.bfloat16).contiguous()
      k = Tensor.randn(B, N, H_KV, D, dtype=dtypes.bfloat16).contiguous()
      v = Tensor.randn(B, N, H_KV, D, dtype=dtypes.bfloat16).contiguous()
      Tensor.realize(q, k, v)

    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    fa_jitted = TinyJit(flash_attention)

    for _ in range(10):
      st = time.perf_counter()
      out = fa_jitted(q, k, v, is_causal=True)
      et = time.perf_counter() - st
      attn_flops = 2 * B * H * N * N * D + \
                   4 * B * H * N * N + \
                   2 * B * H * N * N * D
      print(f"{attn_flops/(et*1e9):2f} GFLOPS")
    out = out.float().transpose(1, 2)

    ref = q.scaled_dot_product_attention(k, v, is_causal=True, enable_gqa=True).float().transpose(1, 2)

    np.testing.assert_allclose(out.numpy(), ref.numpy(), atol=2e-2, rtol=2e-2)

  def test_fast_fa_bwd(self):
    from extra.thunder.tiny.fa import flash_attention

    Tensor.manual_seed(42)

    B, N, H, H_KV, D = 1, 32, 2, 1, 32

    with Context(DEBUG=0):
      q = Tensor.randn(B, N, H, D, dtype=dtypes.bfloat16, requires_grad=True).contiguous()
      k = Tensor.randn(B, N, H_KV, D, dtype=dtypes.bfloat16, requires_grad=True).contiguous()
      v = Tensor.randn(B, N, H_KV, D, dtype=dtypes.bfloat16, requires_grad=True).contiguous()
      Tensor.realize(q, k, v)

      do = Tensor.ones(B, N, H, D, dtype=dtypes.float32).contiguous()
      Tensor.realize(do)

    q_, k_, v_ = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    out = flash_attention(q_, k_, v_)
    out = out.float().transpose(1, 2)
    out.backward(do)
    Tensor.realize(q.grad, k.grad, v.grad)

    with Context(DEBUG=0):
      q_ref = q.detach().clone().requires_grad_(True)
      k_ref = k.detach().clone().requires_grad_(True)
      v_ref = v.detach().clone().requires_grad_(True)
      Tensor.realize(q_ref, k_ref, v_ref)

    q_ref_, k_ref_, v_ref_ = q_ref.transpose(1, 2), k_ref.transpose(1, 2), v_ref.transpose(1, 2)
    ref = q_ref_.scaled_dot_product_attention(k_ref_, v_ref_)
    ref = ref.float().transpose(1, 2)
    ref.backward(do)
    Tensor.realize(q_ref.grad, k_ref.grad, v_ref.grad)

    np.testing.assert_allclose(q.grad.numpy(), q_ref.grad.numpy(), atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(v.grad.numpy(), v_ref.grad.numpy(), atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(k.grad.numpy(), k_ref.grad.numpy(), atol=5e-2, rtol=2e-2)

  def test_fast_fa_bwd_causal(self):
    from extra.thunder.tiny.fa import flash_attention

    Tensor.manual_seed(42)

    B, N, H, H_KV, D = 1, 8192, 32, 32, 128

    with Context(DEBUG=0):
      q = Tensor.randn(B, N, H, D, dtype=dtypes.bfloat16, requires_grad=True).contiguous()
      k = Tensor.randn(B, N, H_KV, D, dtype=dtypes.bfloat16, requires_grad=True).contiguous()
      v = Tensor.randn(B, N, H_KV, D, dtype=dtypes.bfloat16, requires_grad=True).contiguous()
      Tensor.realize(q, k, v)

      do = Tensor.ones(B, N, H, D, dtype=dtypes.float32).contiguous()
      Tensor.realize(do)

    q_, k_, v_ = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    out = flash_attention(q_, k_, v_, is_causal=True)
    out = out.float().transpose(1, 2)
    out.backward(do)
    Tensor.realize(q.grad, k.grad, v.grad)

    with Context(DEBUG=0):
      q_ref = q.detach().clone().requires_grad_(True)
      k_ref = k.detach().clone().requires_grad_(True)
      v_ref = v.detach().clone().requires_grad_(True)
      Tensor.realize(q_ref, k_ref, v_ref)

    q_ref_, k_ref_, v_ref_ = q_ref.transpose(1, 2), k_ref.transpose(1, 2), v_ref.transpose(1, 2)
    ref = q_ref_.scaled_dot_product_attention(k_ref_, v_ref_, is_causal=True, enable_gqa=True)
    ref = ref.float().transpose(1, 2)
    ref.backward(do)
    Tensor.realize(q_ref.grad, k_ref.grad, v_ref.grad)

    np.testing.assert_allclose(q.grad.numpy(), q_ref.grad.numpy(), atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(v.grad.numpy(), v_ref.grad.numpy(), atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(k.grad.numpy(), k_ref.grad.numpy(), atol=6e-2, rtol=2e-2)

  def test_fast_fa_bwd_causal_jitted(self):
    from extra.thunder.tiny.fa import flash_attention

    Tensor.manual_seed(42)

    B, N, H, H_KV, D = 1, 8192, 32, 32, 128

    with Context(DEBUG=0):
      q = Tensor.randn(B, N, H, D, dtype=dtypes.bfloat16, requires_grad=True).contiguous()
      k = Tensor.randn(B, N, H_KV, D, dtype=dtypes.bfloat16, requires_grad=True).contiguous()
      v = Tensor.randn(B, N, H_KV, D, dtype=dtypes.bfloat16, requires_grad=True).contiguous()
      Tensor.realize(q, k, v)

      do = Tensor.ones(B, N, H, D, dtype=dtypes.float32).contiguous()
      Tensor.realize(do)

    def fn(q, k, v, do):
      q_, k_, v_ = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
      out = flash_attention(q_, k_, v_, is_causal=True)
      out = out.float().transpose(1, 2)
      out.backward(do)
      Tensor.realize(out, q.grad, k.grad, v.grad)
      return q.grad, k.grad, v.grad

    fn_jitted = TinyJit(fn)

    for _ in range(10):
      q = Tensor.randn(B, N, H, D, dtype=dtypes.bfloat16, requires_grad=True).contiguous()
      k = Tensor.randn(B, N, H_KV, D, dtype=dtypes.bfloat16, requires_grad=True).contiguous()
      v = Tensor.randn(B, N, H_KV, D, dtype=dtypes.bfloat16, requires_grad=True).contiguous()
      Tensor.realize(q, k, v)
      do = Tensor.ones(B, N, H, D, dtype=dtypes.float32).contiguous()
      Tensor.realize(do)
      q.grad, k.grad, v.grad = fn_jitted(q, k, v, do)

    with Context(DEBUG=0):
      q_ref = q.detach().clone().requires_grad_(True)
      k_ref = k.detach().clone().requires_grad_(True)
      v_ref = v.detach().clone().requires_grad_(True)
      Tensor.realize(q_ref, k_ref, v_ref)

    q_ref_, k_ref_, v_ref_ = q_ref.transpose(1, 2), k_ref.transpose(1, 2), v_ref.transpose(1, 2)
    ref = flash_attention(q_ref_, k_ref_, v_ref_, is_causal=True)
    ref = ref.float().transpose(1, 2)
    ref.backward(do)
    Tensor.realize(q_ref.grad, k_ref.grad, v_ref.grad)

    np.testing.assert_allclose(q.grad.numpy(), q_ref.grad.numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(k.grad.numpy(), k_ref.grad.numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(v.grad.numpy(), v_ref.grad.numpy(), atol=1e-5, rtol=1e-5)

  def test_fast_fa_bwd_multidevice(self):
    from extra.thunder.tiny.fa import flash_attention

    Tensor.manual_seed(42)

    B, N, H, H_KV, D = 2, 1024, 32, 32, 128
    GPUS = tuple(f"AMD:{i}" for i in range(B))

    with Context(DEBUG=0):
      base_q = Tensor.randn(B, N, H, D, dtype=dtypes.bfloat16, requires_grad=True).contiguous()
      base_k = Tensor.randn(B, N, H_KV, D, dtype=dtypes.bfloat16, requires_grad=True).contiguous()
      base_v = Tensor.randn(B, N, H_KV, D, dtype=dtypes.bfloat16, requires_grad=True).contiguous()

      base_do = Tensor.ones(B, N, H, D, dtype=dtypes.float32).contiguous()

    with Context(DEBUG=0):
      q = base_q.clone().requires_grad_(True).shard(GPUS, axis=0)
      k = base_k.clone().requires_grad_(True).shard(GPUS, axis=0)
      v = base_v.clone().requires_grad_(True).shard(GPUS, axis=0)
      Tensor.realize(q, k, v)

      do = base_do.clone().shard(GPUS, axis=0)
      Tensor.realize(do)

    q_, k_, v_ = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    out = flash_attention(q_, k_, v_, is_causal=True)
    out = out.float().transpose(1, 2)
    out.backward(do)
    Tensor.realize(q.grad, k.grad, v.grad)

    with Context(DEBUG=0):
      q_ref = base_q.clone().requires_grad_(True)
      k_ref = base_k.clone().requires_grad_(True)
      v_ref = base_v.clone().requires_grad_(True)
      Tensor.realize(q_ref, k_ref, v_ref)

      do_ref = base_do.clone()
      Tensor.realize(do_ref)

    q_ref_, k_ref_, v_ref_ = q_ref.transpose(1, 2), k_ref.transpose(1, 2), v_ref.transpose(1, 2)
    ref = flash_attention(q_ref_, k_ref_, v_ref_, is_causal=True)
    ref = ref.float().transpose(1, 2)
    ref.backward(do_ref)
    Tensor.realize(q_ref.grad, k_ref.grad, v_ref.grad)

    np.testing.assert_allclose(q.grad.numpy(), q_ref.grad.numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(v.grad.numpy(), v_ref.grad.numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(k.grad.numpy(), k_ref.grad.numpy(), atol=1e-5, rtol=1e-5)

if __name__ == "__main__":
  unittest.main()
