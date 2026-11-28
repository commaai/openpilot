import unittest, math
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import Ops
import numpy as np
from test.null.test_const_folding import _check_ast_count

class TestMovedConstFolding(unittest.TestCase):
  def test_copy_padded_const(self):
    schedule = Tensor.ones(4, buffer=False).pad(((1, 1),)).to("CPU:1").schedule_linear()
    assert not any(si.src[0].op is Ops.STORE for si in schedule.src), "const copy should be folded"
    np.testing.assert_equal(Tensor.ones(4, buffer=False).pad(((1, 1),)).to("CPU:1").numpy(), [0, 1, 1, 1, 1, 0])

  def test_cast_padded(self):
    # NOTE: it's always 1 kernel when calling .numpy, limitation of _check_ast_count
    _check_ast_count(1, Tensor.ones(4).pad(((1, 1),)).cast(dtypes.int16))
    np.testing.assert_equal(Tensor.ones(4).pad(((1, 1),)).cast(dtypes.int16).numpy(), [0, 1, 1, 1, 1, 0])
    _check_ast_count(1, Tensor.full(4, fill_value=-1).pad(((1, 1),)).cast(dtypes.uint16))
    np.testing.assert_equal(Tensor.full(4, fill_value=-1).pad(((1, 1),)).cast(dtypes.uint16).numpy(), [0, 65535, 65535, 65535, 65535, 0])
    # folded
    _check_ast_count(1, Tensor.ones(4).pad(((1, 1),)).cast(dtypes.int64))
    np.testing.assert_equal(Tensor.ones(4).pad(((1, 1),)).cast(dtypes.int64).numpy(), [0, 1, 1, 1, 1, 0])

class TestReduceOpsConstFolding(unittest.TestCase):
  def test_const_sum(self):
    _check_ast_count(0, Tensor.ones(4, 5, 6).sum())
    np.testing.assert_equal(Tensor.ones(4, 5, 6).sum().numpy(), 4 * 5 * 6)
    _check_ast_count(0, Tensor.ones(4, 5, 6).sum(axis=0))
    np.testing.assert_equal(Tensor.ones(4, 5, 6).sum(axis=0).numpy(), np.full((5, 6), 4))
    _check_ast_count(0, Tensor(4).sum())
    np.testing.assert_equal(Tensor(4).sum().numpy(), 4)

  def test_padded_const_sum(self):
    _check_ast_count(0, Tensor.ones(4).pad(((1, 1),)).sum())
    np.testing.assert_equal(Tensor.ones(4).pad(((1, 1),)).sum().numpy(), 4)

    # NOTE: cannot just count the non-padded area because some Ops f do not have f(0) = 0.
    _check_ast_count(1, Tensor.ones(4).pad(((1, 1),)).exp().sum())
    np.testing.assert_allclose(Tensor.ones(4).pad(((1, 1),)).exp().sum().numpy(), 4 * math.e + 2)

  def test_bool_zero_max(self):
    _check_ast_count(0, Tensor.full((1, 2), True).shrink(((0, 1), (0, 0))).max((1, 0)))
    np.testing.assert_equal(Tensor.full((1, 2), True).shrink(((0, 1), (0, 0))).max((1, 0)).numpy(), False)

  def test_zero_size_ops(self):
    for reduceop in [lambda x:x.prod(), lambda x:x.sum()]: # lambda x:x.max() NOTE: numpy gives "reduction operation maximum which has no identity"
      _check_ast_count(0, reduceop(Tensor.empty(1, 0)))
      np.testing.assert_equal(reduceop(Tensor.empty(shape:=(1, 0))).numpy(), reduceop(np.empty(shape)))

  def test_zero_size_ops_view(self):
    for reduceop in [lambda x:x.prod(), lambda x:x.sum()]:
      _check_ast_count(0, reduceop(Tensor.empty(1, 0, 4).permute((1, 2, 0)).contiguous()))
      np.testing.assert_equal(reduceop(Tensor.empty(shape:=(1, 0))).numpy(), reduceop(np.empty((shape))))

  def test_zero_size_ops_realized(self):
    for reduceop in [lambda x:x.prod(), lambda x:x.sum()]:
      _check_ast_count(0, reduceop((Tensor.randn(0, 1)+1).realize()))
      np.testing.assert_equal(reduceop((Tensor.randn(shape:=(0, 1))+1).realize()).numpy(), reduceop(np.empty(shape)))

  def test_zero_size_realize_folded(self):
    # folded output doesn't realize on its own
    _check_ast_count(0, Tensor.empty(1, 0).sum())
    # explicit storage of the folded const still schedules, and the value is usable
    a = Tensor.empty(1, 0).sum().clone()
    _check_ast_count(2, a+2)
    self.assertIs(a.uop.base.op, Ops.BUFFER)
    np.testing.assert_equal((Tensor.empty(1, 0).sum().contiguous()+2).numpy(), 2)
    # otherwise we just fuse it
    _check_ast_count(1, (Tensor.empty(1, 0).sum()+2).contiguous())
    np.testing.assert_equal((Tensor.empty(1, 0).sum()+2).numpy(), 2)

  def test_const_prod(self):
    _check_ast_count(0, Tensor.full((2, 3), fill_value=2).prod())
    np.testing.assert_equal(Tensor.full((2, 3), fill_value=2).prod().numpy(), 2**(2*3))
    _check_ast_count(0, Tensor.full((4, 5, 6), fill_value=2).prod(axis=0))
    np.testing.assert_equal(Tensor.full((4, 5, 6), fill_value=2).prod(axis=0).numpy(), np.full((5, 6), 2**4))
    _check_ast_count(0, Tensor(4).prod())
    np.testing.assert_equal(Tensor(4).prod().numpy(), 4)

  def test_const_max(self):
    _check_ast_count(0, Tensor.ones(4, 5, 6).max())
    np.testing.assert_equal(Tensor.ones(4, 5, 6).max().numpy(), 1)
    _check_ast_count(0, Tensor(4).max())
    np.testing.assert_equal(Tensor(4).max().numpy(), 4)

class TestTautologicalCompare(unittest.TestCase):
  # without const folding, these would have triggered -Wtautological-compare in clang
  def test_lt_false(self):
    # bool < False is always false
    np.testing.assert_equal((Tensor([True, False]) < False).numpy(), [False, False])

  def test_true_lt(self):
    # True < bool is always false
    np.testing.assert_equal((True < Tensor([True, False])).numpy(), [False, False])

  def test_truth_table(self):
    np.testing.assert_equal((Tensor(False) < Tensor(False)).numpy(), False)
    np.testing.assert_equal((Tensor(False) < Tensor(True)).numpy(), True)
    np.testing.assert_equal((Tensor(True) < Tensor(False)).numpy(), False)
    np.testing.assert_equal((Tensor(True) < Tensor(True)).numpy(), False)

  def test_a_ne_a(self):
    # self not eq is always false for int or bool
    a = Tensor([1, 2, 3])
    np.testing.assert_equal((a != a).numpy(), [False, False, False])

    # not true for nan
    a = Tensor([math.nan, 1.0, 2.0])
    np.testing.assert_equal((a != a).numpy(), [True, False, False])

if __name__ == '__main__':
  unittest.main()
