import unittest, itertools, math
from typing import Any
from tinygrad import Tensor, Device, dtypes
from tinygrad.dtype import DType
from tinygrad.ops import Ops, UOp
from tinygrad.codegen import full_rewrite_to_sink
import numpy as np
from tinygrad.device import is_dtype_supported
from test.helpers import not_support_multi_device

def _check_ast_count(desired_count:int, t:Tensor):
  # NOTE: this has side effect because everything can be scheduled only once
  schedule = t.schedule()
  asts = [s for s in schedule if s.ast.op is Ops.SINK]
  assert len(asts) == desired_count, f"{len(asts)} != {desired_count}"

class TestUnaryOpsConstFolding(unittest.TestCase):
  def test_all_consts_ops(self):
    _check_ast_count(0, Tensor.ones(4).exp())
    _check_ast_count(0, Tensor.ones(4).sqrt())
    _check_ast_count(0, Tensor.ones(4) + Tensor.ones(4))
    _check_ast_count(0, Tensor.ones(4) / Tensor.ones(4))

  def test_cast(self):
    _check_ast_count(0, Tensor.ones(4).cast(dtypes.int16))
    _check_ast_count(0, Tensor.full(4, fill_value=-1).cast(dtypes.uint16))

  @unittest.expectedFailure  # no two level fold at lazybuffer
  def test_neg_folding(self):
    _check_ast_count(0, Tensor([1, 2, 3]).mul(-1).neg())
    _check_ast_count(0, Tensor([1, 2, 3]).neg().mul(-1))
    _check_ast_count(0, Tensor([1, 2, 3]).neg().neg())

  def test_neg_realized_no_fold(self):
    x = Tensor.randn(32, 32)
    x = x.clip(0, 1).realize()
    _check_ast_count(1, x.neg())

class TestBinaryOpsConstFolding(unittest.TestCase):
  def test_add_literal_zero(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) + 0)
  def test_add_tensor_zero(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) + Tensor.zeros(4))
  def test_literal_zero_add(self):
    _check_ast_count(0, 0 + Tensor([1.0, 2, 3, 4]))
  def test_tensor_zero_add(self):
    _check_ast_count(0, Tensor.zeros(4) + Tensor([1.0, 2, 3, 4]))

  def test_sub_literal_zero(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) - 0)
  def test_sub_tensor_zero(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) - Tensor.zeros(4))

  def test_mul_literal_zero(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) * 0)
  def test_mul_tensor_zero(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) * Tensor.zeros(4))
  def test_literal_zero_mul(self):
    _check_ast_count(0, 0 * Tensor([1.0, 2, 3, 4]) * 0)
  def test_tensor_zero_mul(self):
    _check_ast_count(0, Tensor.zeros(4) * Tensor([1.0, 2, 3, 4]))

  def test_mul_literal_one(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) * 1)
  def test_mul_tensor_one(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) * Tensor.ones(4))
  def test_literal_one_mul(self):
    _check_ast_count(0, 1 * Tensor([1.0, 2, 3, 4]))
  def test_tensor_one_mul(self):
    _check_ast_count(0, Tensor.ones(4) * Tensor([1.0, 2, 3, 4]))

  def test_bool_tensor_mul_bool(self):
    _check_ast_count(0, Tensor([True, False]) * True)
    _check_ast_count(0, Tensor([True, False]) * False)
  def test_bool_mul_bool_tensor(self):
    _check_ast_count(0, True * Tensor([True, False]))
    _check_ast_count(0, False * Tensor([True, False]))

  def test_div_literal_one(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) / 1)
  def test_div_tensor_one(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) / Tensor.ones(4))

  def test_idiv_literal_one(self):
    _check_ast_count(0, Tensor([1, 2, 3, 4]) // 1)
  def test_idiv_tensor_one(self):
    _check_ast_count(0, Tensor([1, 2, 3, 4]) // Tensor.ones(4, dtype=dtypes.int32))

  def test_pow_literal_zero(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) ** 0)
  def test_pow_tensor_zero(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) ** Tensor.zeros(4))

  def test_pow_literal_one(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) ** 1)
  def test_pow_tensor_one(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) ** Tensor.ones(4))
  def test_literal_one_pow(self):
    _check_ast_count(0, 1 ** Tensor([1.0, 2, 3, 4]))
  def test_tensor_one_pow(self):
    _check_ast_count(0, Tensor.ones(4) ** Tensor([1.0, 2, 3, 4]))

class TestBitcastConstFolding(unittest.TestCase):
  def test_scalar_bitcast(self):
    def t(cases: dict[DType, Any]):
      for (from_dt, from_v), (to_dt, to_v) in itertools.product(cases.items(), cases.items()):
        if not math.isnan(from_v):
          r = full_rewrite_to_sink(UOp.const(from_dt, from_v).bitcast(to_dt).sink()).src[0]
          self.assertEqual(r.op, Ops.CONST, msg:=f"{from_dt} -> {to_dt} ({from_v} -> {to_v})")
          self.assertEqual(r.dtype, to_dt, msg)
          np.testing.assert_equal(r.arg, to_v, msg)

    t({dtypes.int8: 0, dtypes.uint8: 0, dtypes.bool: False})
    t({dtypes.int8: 1, dtypes.uint8: 1, dtypes.bool: True})

    t({dtypes.int8:  -1, dtypes.uint8:  2**8-1})
    t({dtypes.int16: -1, dtypes.uint16: 2**16-1, dtypes.float16: float('nan')})
    t({dtypes.int32: -1, dtypes.uint32: 2**32-1, dtypes.float32: float('nan')})
    t({dtypes.int64: -1, dtypes.uint64: 2**64-1, dtypes.float64: float('nan')})

    t({dtypes.int8:  -2**7,  dtypes.uint8:  2**7})
    t({dtypes.int16: -2**15, dtypes.uint16: 2**15})
    t({dtypes.int32: -2**31, dtypes.uint32: 2**31})
    t({dtypes.int64: -2**63, dtypes.uint64: 2**63})

    t({dtypes.int16: 13496, dtypes.uint16: 13496, dtypes.float16: 0.294921875})
    t({dtypes.int32: 1050081145, dtypes.uint32: 1050081145, dtypes.float32: 0.29485681653022766})
    t({dtypes.int64: 4598983288165178391, dtypes.uint64: 4598983288165178391, dtypes.float64: 0.29485681936461233})

  def test_vec_bitcast(self):
    r = full_rewrite_to_sink(UOp.const(dtypes.int32.vec(3), (-1, -2**31, 75)).bitcast(dtypes.uint32.vec(3)).sink()).src[0]
    self.assertEqual(r.op, Ops.VECTORIZE)
    self.assertEqual(r.dtype, dtypes.uint32.vec(3))
    self.assertEqual(tuple(x.arg for x in r.src), (2**32-1, 2**31, 75))

# folds advance indexing into basic indexing
class TestIndexingConstFolding(unittest.TestCase):
  def test_scalar_index(self):
    t = Tensor.arange(16).float().reshape(1,1,4,4).realize()
    # TODO: fold these
    _check_ast_count(2, t[:,:,Tensor(1),:])
    _check_ast_count(2, t[:,:,Tensor(1)+2,:])
    _check_ast_count(2, t[:,:,Tensor(1),Tensor(0)])

  @unittest.expectedFailure
  def test_const_tensor_index(self):
    # TODO: implement const tensor folded indexing
    t = Tensor.arange(16).float().reshape(1,1,4,4).realize()
    _check_ast_count(0, t[:,:,Tensor.ones(2,1),:])
    _check_ast_count(0, t[:,:,Tensor.ones(1,2)+2,:])
    _check_ast_count(0, t[:,:,Tensor.ones(1,1),Tensor.zeros(2,1,2)])

class TestMovedConstFolding(unittest.TestCase):
  def test_add_shrunk_zero(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) + Tensor.zeros(6).shrink(((1, 5),)))

  def test_add_padded_zero(self):
    # TODO: it's 1 now, this might be possible to fold
    _check_ast_count(1, Tensor([1.0, 2, 3, 4]) + Tensor.zeros(2).pad(((1, 1),)))

  def test_mul_shrunk_one(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) * Tensor.ones(6).shrink(((1, 5),)))

  def test_add_padded_one(self):
    _check_ast_count(1, Tensor([1.0, 2, 3, 4]) * Tensor.ones(2).pad(((1, 1),)))

  def test_cast_padded(self):
    # NOTE: this is folded due to CAST_BEFORE_VIEW
    if is_dtype_supported(dtypes.int16):
      _check_ast_count(0, Tensor.ones(4).pad(((1, 1),)).cast(dtypes.int16))
      np.testing.assert_equal(Tensor.ones(4).pad(((1, 1),)).cast(dtypes.int16).numpy(), [0, 1, 1, 1, 1, 0])
    if is_dtype_supported(dtypes.uint16):
      _check_ast_count(0, Tensor.full(4, fill_value=-1).pad(((1, 1),)).cast(dtypes.uint16))
      np.testing.assert_equal(Tensor.full(4, fill_value=-1).pad(((1, 1),)).cast(dtypes.uint16).numpy(), [0, 65535, 65535, 65535, 65535, 0])
    # not folded
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
    _check_ast_count(1, Tensor.ones(4).pad(((1, 1),)).sum())
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
    self.assertIs(a.lazydata.base.op, Ops.BUFFER)
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
    for dt in dtypes.fields().values():
      if is_dtype_supported(dt):
        t = Tensor.ones(16, dtype=dt).reshape(4, 4)
        assert t.sum().dtype == t.contiguous().sum().dtype

@unittest.skipIf(not_support_multi_device(), "no multi")
class TestMultiConstFolding(unittest.TestCase):
  def test_multi_const_folding_literal(self):
    ds = tuple(f"{Device.DEFAULT}:{i}" for i in range(4))
    t = Tensor.arange(16).float().realize().to(ds)

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
    t = Tensor.arange(16).float().realize().to(ds)
    zero = Tensor.zeros(16).realize().to(ds)
    one = Tensor.ones(16).realize().to(ds)

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

  def test_multi_todo_pow(self):
    ds = tuple(f"{Device.DEFAULT}:{i}" for i in range(4))
    t = Tensor.arange(16).float().realize().to(ds)
    zero = Tensor.zeros(16).realize().to(ds)
    one = Tensor.ones(16).realize().to(ds)

    # TODO: fix pow folding
    _check_ast_count(0, t ** zero)
    _check_ast_count(0, t ** one)
    _check_ast_count(0, one ** t)

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
