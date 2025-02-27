#!/usr/bin/env python
import numpy as np
import unittest
from tinygrad import Tensor, Device, dtypes
from tinygrad.engine.realize import run_schedule
from tinygrad.ops import Ops, UOp, UPat

class TestTensorUOp(unittest.TestCase):
  def test_fromcpu_shape_tracker(self):
    def helper(a: np.ndarray):
      print(a.shape, a.strides, a.flags.c_contiguous)
      b = Tensor(a).lazydata
      #assert b.st.contiguous == a.flags.c_contiguous
      assert b.st.shape == a.shape
      np.testing.assert_equal(a, Tensor(b).numpy())

    for ndims in range(1, 4):
      a = np.random.randn(*(4,)*ndims).astype(np.float32)
      for stride in [-2, 1, 2]:
        for start in [0, 1]:
          helper(a[(slice(start, None, stride),)*ndims])

  def test_shuffle_pad_ops_cmpeq(self):
    y = Tensor([1]).cat(Tensor([1]) == 0).numpy()
    z = Tensor([1, 0]).numpy()
    np.testing.assert_allclose(y, z)

  def test_shuffle_pad_ops_div(self):
    y = Tensor([1]).cat(Tensor([1]).div(Tensor([2.0]))).numpy()
    z = Tensor([1, 0.5]).numpy()
    np.testing.assert_allclose(y, z)

  def test_shuffle_pad_ops_log(self):
    y = Tensor([1]).cat(Tensor([1]).log()).numpy()
    z = Tensor([1, 0]).numpy()
    np.testing.assert_allclose(y, z)

  def test_shuffle_pad_ops_exp(self):
    y = Tensor([1]).cat(Tensor([1]).exp()).numpy()
    z = Tensor([1, np.e]).numpy()
    np.testing.assert_allclose(y, z)

  def test_device_0_is_the_same_device(self):
    a = Tensor([1, 2, 3], f"{Device.DEFAULT}")
    b = Tensor([1, 2, 3], f"{Device.DEFAULT}:0")
    assert a.device == b.device

  def test_shrink_const_into_zero(self):
    # regression test to make sure the shapetracker is preserved
    a = Tensor.zeros(4,4,4).shrink((None, (0,0), None))
    b = Tensor.zeros(4,1,4)
    c = a.cat(b, dim=1)
    np.testing.assert_allclose(c.numpy(), np.concatenate((a.numpy(), b.numpy()), axis=1))

  def test_shrink_const_then_cast(self):
    # regression test to make sure the shapetracker is preserved
    a = Tensor.zeros(4,4,4).shrink((None, (0,0), None)).cast(dtypes.int32)
    b = Tensor.zeros(4,1,4)
    c = a.cat(b, dim=1)
    np.testing.assert_allclose(c.numpy(), np.concatenate((a.numpy(), b.numpy()), axis=1))

  def test_const_dtype(self):
    lb: UOp = Tensor([1], dtype=dtypes.int).lazydata
    assert lb.const_like(1).base.arg == 1
    assert type(lb.const_like(1).base.arg) is int

    lb: UOp = Tensor([1], dtype=dtypes.float).lazydata
    assert lb.const_like(1).base.arg == 1.0
    assert type(lb.const_like(1).base.arg) is float

  def test_contiguous_alu(self):
    a = Tensor.randn(2, 2).realize()
    b = Tensor.randn(2, 2).realize()
    add = (a+b).contiguous()
    out = add+2
    sched = out.schedule()
    self.assertEqual(len(sched), 2)
    run_schedule(sched)
    np.testing.assert_allclose(out.numpy(), a.numpy()+b.numpy()+2)

  # NOTE: contiguous on a buffer collapses
  def test_contiguous_empty(self):
    empty = Tensor.empty(1).contiguous()
    sched = empty.schedule()
    self.assertEqual(len(sched), 0)

  def test_contiguous_folded_alu(self):
    a = Tensor.empty(8, 8)
    # NOTE: the buffer for mul_0 late folds to just a CONST
    mul_0 = a*0
    out = mul_0.shrink(((4, 8), (0, 8))).contiguous()
    out.realize()
    self.assertEqual(out.tolist(), Tensor.zeros(4, 8).tolist())

reduce_kernel = UPat(Ops.SINK, src=(UPat(Ops.STORE, src=(UPat(), UPat(), UPat(Ops.REDUCE_AXIS)))))
class TestReduceOp(unittest.TestCase):
  def test_no_split_reduce_kernel(self):
    a = Tensor.rand(4, 4).realize()
    a = a.sum()
    sched = a.schedule()
    assert len(sched) == 1
    assert reduce_kernel.match(sched[0].ast, {})

  def test_split_reduce_kernel_dim0(self):
    a = Tensor.rand(256, 255).realize()
    a = a.sum()
    sched = a.schedule()
    assert len(sched) == 2
    for s in sched:
      assert reduce_kernel.match(s.ast, {})

  def test_split_reduce_kernel_dim1(self):
    a = Tensor.rand(255, 256).realize()
    a = a.sum()
    sched = a.schedule()
    assert len(sched) == 2
    for s in sched:
      assert reduce_kernel.match(s.ast, {})

if __name__ == "__main__":
  unittest.main()
