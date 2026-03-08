# tensor tests that pass on NULL backend (no copyout needed)
import numpy as np
import unittest
from tinygrad import Tensor, Device, dtypes
from tinygrad.device import is_dtype_supported
from tinygrad.uop.ops import Ops, UOp
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.renderer.nir import NIRRenderer
from tinygrad.engine.realize import get_program
from tinygrad.dtype import DType

x_init = np.random.randn(1,3).astype(np.float32)
W_init = np.random.randn(3,3).astype(np.float32)
m_init = np.random.randn(1,3).astype(np.float32)

class TestTrainMode(unittest.TestCase):
  def test_train_mode(self):
    assert not Tensor.training
    @Tensor.train()
    def f():
      assert Tensor.training
    f()
    assert not Tensor.training

class TestInferenceMode(unittest.TestCase):
  def test_inference(self):
    x = Tensor(x_init, requires_grad=True)
    m = Tensor(m_init, requires_grad=True)
    W = Tensor(W_init, requires_grad=True)
    tmp = x.mul(m)
    mm = tmp.matmul(W)
    out = mm.relu()
    out = out.sum()
    #out.backward()
    assert x.grad is None
    assert m.grad is None
    assert tmp.grad is None
    assert mm.grad is None
    assert W.grad is None
    assert W.requires_grad

  def test_no_grad_mode_context_manager(self):
    x = Tensor(x_init, requires_grad=True)
    m = Tensor(m_init, requires_grad=True)
    W = Tensor(W_init, requires_grad=True)
    def f(x, m, W):
      tmp = x.mul(m)
      mm = tmp.matmul(W)
      out = mm.relu()
      out = out.sum()
      #out.backward()
      assert x.grad is None
      assert m.grad is None
      assert tmp.grad is None
      assert mm.grad is None
      assert W.grad is None
    f(x, m, W)

class TestIdxUpcast(unittest.TestCase):
  def _find_op(self, ast: UOp, op: Ops):
    if ast.op is op: return ast
    for src in ast.src:
      if (ret:=self._find_op(src, op)) is not None: return ret
  def _schedule_render(self, a: Tensor):
    schedule, _ = a.schedule_with_vars()
    for s in schedule:
      if s.ast.op is Ops.SINK:
        renderer = Device[s.bufs[0].device].renderer
        prg = get_program(s.ast, renderer)
        return prg.uops

  def _assert(self, dtype: DType, a: Tensor):
    uops = self._schedule_render(a)
    # Assert the dtype of the INDEX value, This will need be updated if UOp spec changes
    store = next(uop for uop in uops if uop.op is Ops.STORE)
    assert store.op is Ops.STORE
    idx = self._find_op(store, Ops.INDEX)
    # PTX and NIR turn Ops.INDEX into pointer arithmetic earlier than cstyle, plus it's already cast to int64
    if not isinstance(Device[Device.DEFAULT].renderer, (PTXRenderer, NIRRenderer)):
      assert idx.op is Ops.INDEX
      idx_val = idx.src[1]
      assert idx_val.dtype is dtype

  # use expand to generate kernel that uses large idx
  def do_op_then_assert(self, dtype: DType, dim1, dim2, dim3):
    self._assert(dtype, Tensor.empty(dim1, dim2, 1).expand(-1, -1, dim3).contiguous())

  @unittest.skipUnless(is_dtype_supported(dtypes.long), "int64 is supported")
  def test_overflow(self):
    # 2**11, 2**11, 2**11 -> 2**33 will overflow when indexed
    self.do_op_then_assert(dtypes.long, 2048, 2048, 2048)

  @unittest.skipUnless(is_dtype_supported(dtypes.long), "int64 is supported")
  def test_overflow_sym(self):
    self.do_op_then_assert(dtypes.long, 2048, 2048, UOp.variable("dim3", 1, 2048).bind(32))

  def test_regular(self):
    self.do_op_then_assert(dtypes.int, 64, 64, 64)

  def test_regular_sym(self):
    self.do_op_then_assert(dtypes.int, 2048, 2048, UOp.variable("dim3", 1, 64).bind(32))

  @unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, (PTXRenderer, NIRRenderer)), "PTX and NIR always converts Ops.INDEX to int64")
  def test_symfold(self):
    # This would cause an overflow, but after sym fold it's within int32
    a = Tensor.arange(65535)
    uops = self._schedule_render(a)
    assert all(uop.dtype is not dtypes.long for uop in uops)

  def test_arange_raise_overflow(self):
    with self.assertRaises(ValueError):
      self._schedule_render(Tensor.arange(2**33, dtype=dtypes.int))

  @unittest.skipIf(is_dtype_supported(dtypes.long), "int64 is supported")
  def test_int64_unsupported_overflow_sym(self):
    with self.assertRaises(KeyError):
      self.do_op_then_assert(dtypes.long, 2048, 2048, UOp.variable("dim3", 1, 2048).bind(32))

  @unittest.skipIf(is_dtype_supported(dtypes.long), "int64 is supported")
  @unittest.expectedFailure  # bug in gpu dims limiting
  def test_int64_unsupported_overflow(self):
    with self.assertRaises(KeyError):
      self.do_op_then_assert(dtypes.long, 2048, 2048, 2048)

  @unittest.skip("This is kept for reference, it requires large memory to run")
  def test_overflow_kernel_run(self):
    # This creates a total of 2**31+10 elements, requiring at least 2147 MB memory to run
    # Modified example from issue 3271
    a = Tensor.empty(2**11, 2**11, 1, dtype=dtypes.int8).permute((2, 0, 1)).expand((2**9+10, -1, -1)).contiguous()
    a.realize()

class TestTensorUnique(unittest.TestCase):
  def test_empty_bufs_unique(self):
    a = Tensor.empty(10, 10).contiguous()
    b = Tensor.empty(10, 10).contiguous()
    Tensor.realize(a,b)
    self.assertIsNot(a.uop.buffer, b.uop.buffer)

  def test_zeros_bufs_unique_sep(self):
    a = Tensor.zeros(10, 10).contiguous()
    Tensor.realize(a)
    b = Tensor.zeros(10, 10).contiguous()
    Tensor.realize(b)
    self.assertIsNot(a.uop.buffer, b.uop.buffer)

  def test_zeros_bufs_unique(self):
    a = Tensor.zeros(10, 10).contiguous()
    b = Tensor.zeros(10, 10).contiguous()
    Tensor.realize(a,b)
    self.assertIsNot(a.uop.buffer, b.uop.buffer)

  def test_eye_bufs_unique(self):
    a = Tensor.eye(10).contiguous()
    b = Tensor.eye(10).contiguous()
    Tensor.realize(a,b)
    self.assertIsNot(a.uop.buffer, b.uop.buffer)

  def test_times_2_not_unique(self):
    a = Tensor.zeros(10, 10).contiguous()
    b = a * 2
    c = a * 2
    Tensor.realize(b,c)
    self.assertIs(b.uop.buffer, c.uop.buffer)

if __name__ == '__main__':
  unittest.main()
