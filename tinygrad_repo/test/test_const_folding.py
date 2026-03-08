import unittest, math
from tinygrad import Tensor, Device, dtypes
from tinygrad.dtype import DTYPES_DICT
from tinygrad.uop.ops import Ops
from tinygrad.device import is_dtype_supported
import numpy as np
from test.helpers import not_support_multi_device

def _check_ast_count(desired_count:int, t:Tensor):
  # NOTE: this has side effect because everything can be scheduled only once
  schedule = t.schedule()
  asts = [s for s in schedule if s.ast.op is Ops.SINK]
  assert len(asts) == desired_count, f"{len(asts)} != {desired_count}"

class TestMovedConstFolding(unittest.TestCase):
  def test_add_shrunk_zero(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) + Tensor.zeros(6).shrink(((1, 5),)))

  def test_add_padded_zero(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) + Tensor.zeros(2).pad(((1, 1),)))

  def test_mul_shrunk_one(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) * Tensor.ones(6).shrink(((1, 5),)))

  def test_add_padded_one(self):
    _check_ast_count(1, Tensor([1.0, 2, 3, 4]) * Tensor.ones(2).pad(((1, 1),)))

  def test_cast_padded(self):
    # NOTE: it's always 1 kernel when calling .numpy, limitation of _check_ast_count
    if is_dtype_supported(dtypes.int16):
      _check_ast_count(1, Tensor.ones(4).pad(((1, 1),)).cast(dtypes.int16))
      np.testing.assert_equal(Tensor.ones(4).pad(((1, 1),)).cast(dtypes.int16).numpy(), [0, 1, 1, 1, 1, 0])
    if is_dtype_supported(dtypes.uint16):
      _check_ast_count(1, Tensor.full(4, fill_value=-1).pad(((1, 1),)).cast(dtypes.uint16))
      np.testing.assert_equal(Tensor.full(4, fill_value=-1).pad(((1, 1),)).cast(dtypes.uint16).numpy(), [0, 65535, 65535, 65535, 65535, 0])
    # folded
    if is_dtype_supported(dtypes.int64):
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
    # non contiguous folded output doesn't realize
    _check_ast_count(0, Tensor.empty(1, 0).sum())
    # contiguous folded const can still schedule
    a = Tensor.empty(1, 0).sum().contiguous()
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

  def test_sum_output_dtype(self):
    # sum output dtype can be different from input
    for dt in DTYPES_DICT.values():
      if is_dtype_supported(dt):
        t = Tensor.ones(16, dtype=dt).reshape(4, 4)
        assert t.sum().dtype == t.contiguous().sum().dtype

@unittest.skipIf(not_support_multi_device() or True, "no multi, RANGEIFY doesn't support multi const folding")
class TestMultiConstFolding(unittest.TestCase):
  def test_multi_const_folding_literal(self):
    ds = tuple(f"{Device.DEFAULT}:{i}" for i in range(4))
    t = Tensor.arange(16).float().to(ds).realize()

    # non const folding case creates one ast on each shard
    _check_ast_count(4, t + 1)
    _check_ast_count(4, 1 + t)
    _check_ast_count(4, t * 2)
    _check_ast_count(4, 2 * t)

    # const folded
    _check_ast_count(0, t + 0)
    _check_ast_count(0, 0 + t)
    _check_ast_count(0, t * 0)
    _check_ast_count(0, 0 * t)
    _check_ast_count(0, t * 1)
    _check_ast_count(0, 1 * t)
    np.testing.assert_equal((t + 0).numpy(), np.arange(16))
    np.testing.assert_equal((t * 0).numpy(), [0] * 16)
    np.testing.assert_equal((t * 1).numpy(), np.arange(16))

    _check_ast_count(0, t ** 0)
    _check_ast_count(0, t ** 1)
    _check_ast_count(0, 1 ** t)

  def test_multi_const_folding_tensor(self):
    ds = tuple(f"{Device.DEFAULT}:{i}" for i in range(4))
    t = Tensor.arange(16).float().to(ds).realize()
    zero = Tensor.zeros(16).to(ds).realize()
    one = Tensor.ones(16).to(ds).realize()

    # const folded
    _check_ast_count(0, t + zero)
    _check_ast_count(0, zero + t)
    _check_ast_count(0, t * zero)
    _check_ast_count(0, zero * t)
    _check_ast_count(0, t * one)
    _check_ast_count(0, one * t)
    np.testing.assert_equal((t + zero).numpy(), np.arange(16))
    np.testing.assert_equal((t * zero).numpy(), [0] * 16)
    np.testing.assert_equal((t * one).numpy(), np.arange(16))
    _check_ast_count(0, t ** zero)
    _check_ast_count(0, t ** one)
    _check_ast_count(0, one ** t)
    np.testing.assert_equal((t ** zero).numpy(), [1] * 16)
    np.testing.assert_equal((t ** one).numpy(), np.arange(16))
    np.testing.assert_equal((one ** t).numpy(), [1] * 16)

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

  @unittest.skipIf(Device.DEFAULT == "WEBGPU", "WEBGPU doesn't support NaN comparison correctly")
  def test_a_eq_a(self):
    # self eq is always true for int or bool
    a = Tensor([1, 2, 3])
    np.testing.assert_equal((a == a).numpy(), [True, True, True])

    # not true for nan
    a = Tensor([math.nan, 1.0, 2.0])
    np.testing.assert_equal((a == a).numpy(), [False, True, True])

  def test_a_ne_a(self):
    # self not eq is always false for int or bool
    a = Tensor([1, 2, 3])
    np.testing.assert_equal((a != a).numpy(), [False, False, False])

    # not true for nan
    a = Tensor([math.nan, 1.0, 2.0])
    np.testing.assert_equal((a != a).numpy(), [True, False, False])

if __name__ == '__main__':
  unittest.main()
