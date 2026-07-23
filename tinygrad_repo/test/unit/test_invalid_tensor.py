import unittest
from tinygrad import Tensor
from tinygrad.device import Buffer
from tinygrad.dtype import Invalid, dtypes
from tinygrad.engine.realize import run_linear
from tinygrad.uop.ops import Ops, UOp

class TestInvalidTensor(unittest.TestCase):
  def _invalid_test_helper(self, out, expected):
    linear, var_vals = out.linear_with_vars()
    buf = out.uop.buffer
    buf.allocate()
    sentinel = memoryview(bytearray(b'\x42' * buf.nbytes))
    buf.copy_from(Buffer("PYTHON", buf.size, buf.dtype, opaque=sentinel))
    before = buf.as_memoryview().cast(out.dtype.fmt).tolist()
    run_linear(linear, var_vals)
    ret = buf.as_memoryview().cast(out.dtype.fmt).tolist()

    for i,v in enumerate(expected): self.assertEqual(ret[i], before[i] if v is None else v)

  def test_where_x_invalid(self):
    mask = Tensor.arange(4) < 2
    out = mask.where(Tensor([1.0, 2.0, 3.0, 4.0]), Invalid)
    self._invalid_test_helper(out, [1.0, 2.0, None, None])

  def test_where_invalid_x(self):
    mask = Tensor.arange(4) < 2
    out = mask.where(Invalid, Tensor([1.0, 2.0, 3.0, 4.0]))
    self._invalid_test_helper(out, [None, None, 3.0, 4.0])

  def test_where_invalid_2d(self):
    mask = Tensor.arange(6).reshape(2, 3) < 3
    vals = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = mask.where(vals, Invalid)
    self._invalid_test_helper(out, [1.0, 2.0, 3.0, None, None, None])

  def test_where_invalid_int(self):
    mask = Tensor.arange(3) < 2
    out = mask.where(Tensor([10, 20, 30]), Invalid)
    self._invalid_test_helper(out, [10, 20, None])

  def test_where_invalid_add(self):
    mask = Tensor.arange(3) < 2
    mixed = mask.where(Tensor([10.0, 20.0, 30.0]), Invalid)
    out = mixed + Tensor([1.0, 2.0, 3.0])
    self._invalid_test_helper(out, [11.0, 22.0, None])

  def test_where_invalid_add_left(self):
    mask = Tensor.arange(3) < 2
    mixed = mask.where(Tensor([10.0, 20.0, 30.0]), Invalid)
    out = Tensor([1.0, 2.0, 3.0]) + mixed
    self._invalid_test_helper(out, [11.0, 22.0, None])

  def test_where_always_true(self):
    mask = Tensor.arange(3) < 10
    out = mask.where(Tensor([10.0, 20.0, 30.0]), Invalid)
    self._invalid_test_helper(out, [10.0, 20.0, 30.0])

  def test_where_cast(self):
    mask = Tensor.arange(4) < 2
    out = mask.where(Tensor([1.0, 2.0, 3.0, 4.0]), Invalid).cast(dtypes.int)
    self._invalid_test_helper(out, [1, 2, None, None])

  def test_where_compare(self):
    mask = Tensor.arange(4) < 2
    out = mask.where(Tensor([1.0, 2.0, 3.0, 4.0]), Invalid) > 1
    self._invalid_test_helper(out, [False, True, None, None])

  def test_where_invalid_condition(self):
    a, x = Tensor.arange(4), Tensor([0, 1, 2, 3])
    bad = (a < 2).where(a, Invalid)
    out = (bad < 1).logical_not().where(x + 10, x + 20)
    self._invalid_test_helper(out, [20, 11, None, None])

  def test_where_invalid_condition_bare(self):
    cond = Tensor.full((4,), Invalid, dtype=dtypes.bool, buffer=False)
    out = cond.where(Tensor([1.0, 2.0, 3.0, 4.0]), Tensor([10.0, 20.0, 30.0, 40.0]))
    self._invalid_test_helper(out, [None, None, None, None])

  def test_where_unary(self):
    mask = Tensor.arange(4) < 2
    out = mask.where(Tensor([1.0, 4.0, 9.0, 16.0]), Invalid).sqrt()
    self._invalid_test_helper(out, [1.0, 2.0, None, None])

  def test_where_where(self):
    mask1 = Tensor.arange(4) < 2
    mask2 = Tensor.arange(4) > 0
    out = mask2.where(mask1.where(Tensor([1.0, 2.0, 3.0, 4.0]), Invalid), Invalid)
    self._invalid_test_helper(out, [None, 2.0, None, None])

  def test_where_reduce_always_true(self):
    mask = Tensor.arange(4) < 9
    out = mask.where(Tensor([1.0, 2.0, 3.0, 4.0]), Invalid).sum()
    self._invalid_test_helper(out, [10.0])

  def test_invalid_unary(self):
    mask = Tensor.arange(4) < 2
    out = mask.where(Tensor([1.0, 2.0, 3.0, 4.0]), Tensor.full((4,), Invalid, dtype=dtypes.float, buffer=False).sqrt())
    self._invalid_test_helper(out, [1.0, 2.0, None, None])

  def test_invalid_binary(self):
    mask = Tensor.arange(4) < 2
    out = mask.where(Tensor([1.0, 2.0, 3.0, 4.0]), Tensor.full((4,), Invalid, dtype=dtypes.float, buffer=False) + 2)
    self._invalid_test_helper(out, [1.0, 2.0, None, None])

  def test_invalid_binary_left(self):
    mask = Tensor.arange(4) < 2
    out = mask.where(Tensor([1.0, 2.0, 3.0, 4.0]), 2 + Tensor.full((4,), Invalid, dtype=dtypes.float, buffer=False))
    self._invalid_test_helper(out, [1.0, 2.0, None, None])

  def test_invalid_reshape(self):
    mask = Tensor.arange(4) < 2
    out = mask.where(Tensor([1.0, 2.0, 3.0, 4.0]), Invalid).reshape(2,2)
    self._invalid_test_helper(out, [1.0, 2.0, None, None])

  def test_invalid_cast(self):
    mask = Tensor.arange(4) < 2
    out = mask.where(Tensor([1.0, 2.0, 3.0, 4.0]), Tensor.full((4,), Invalid, dtype=dtypes.int, buffer=False).cast(dtypes.float))
    self._invalid_test_helper(out, [1.0, 2.0, None, None])

  def test_invalid_bitcast(self):
    mask = Tensor.arange(4) < 2
    out = mask.where(Tensor([1.0, 2.0, 3.0, 4.0]), Tensor.full((4,), Invalid, dtype=dtypes.int, buffer=False).bitcast(dtypes.float))
    self._invalid_test_helper(out, [1.0, 2.0, None, None])

  def test_where_bitcast(self):
    mask = Tensor.arange(4) < 2
    out = mask.where(Tensor([1.0, 2.0, 3.0, 4.0]), Tensor.full((4,), Invalid, dtype=dtypes.int, buffer=False)).bitcast(dtypes.int)
    self._invalid_test_helper(out, [0x3f800000, 0x40000000, None, None])

  def test_tensor_index(self):
    idx = (Tensor.arange(4) < 2).where(Tensor([0, 1, 2, 3]), Invalid)
    out = Tensor([1.0, 2.0, 3.0, 4.0])[idx]
    self._invalid_test_helper(out, [1.0, 2.0, None, None])

  def test_uop_where_keeps_invalid_bare(self):
    cond = UOp.const(dtypes.weakint, 0) < UOp.const(dtypes.weakint, 1)
    idx = UOp(Ops.STACK, src=tuple(UOp.const(dtypes.weakint, x) for x in range(3)))
    out = cond.where(idx, UOp.invalid())
    self.assertIs(cond.op, Ops.CMPLT)
    self.assertIs(idx.op, Ops.STACK)
    self.assertIs(out.op, Ops.WHERE)
    self.assertIs(out.src[2].op, Ops.CONST)
    self.assertIs(out.src[2].arg, Invalid)

if __name__ == '__main__':
  unittest.main()
