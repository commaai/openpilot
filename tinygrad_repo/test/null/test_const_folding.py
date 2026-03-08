import unittest, itertools, math
from tinygrad import Tensor, dtypes, Context
from tinygrad.dtype import DType, ConstType
from tinygrad.uop.ops import Ops, UOp
from tinygrad.codegen import full_rewrite_to_sink
import numpy as np

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
    def t(cases: dict[DType, ConstType]):
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
    with Context(SPEC=0):
      r = full_rewrite_to_sink(UOp.const(dtypes.int32.vec(3), (-1, -2**31, 75)).bitcast(dtypes.uint32.vec(3)).sink()).src[0]
    self.assertEqual(r.op, Ops.VECTORIZE)
    self.assertEqual(r.dtype, dtypes.uint32.vec(3))
    self.assertEqual(tuple(x.arg for x in r.src), (2**32-1, 2**31, 75))

# folds advance indexing into basic indexing
class TestIndexingConstFolding(unittest.TestCase):
  def test_scalar_index(self):
    t = Tensor.arange(16).float().reshape(1,1,4,4).realize()
    _check_ast_count(1, t[:,:,Tensor(1),:])
    _check_ast_count(1, t[:,:,Tensor(1)+2,:])
    _check_ast_count(1, t[:,:,Tensor(1),Tensor(0)])

  def test_const_tensor_index(self):
    # TODO: these can be 0, implement const tensor folded indexing
    t = Tensor.arange(16).float().reshape(1,1,4,4).realize()
    _check_ast_count(1, t[:,:,Tensor.ones(2,1,dtype=dtypes.int),:])
    _check_ast_count(1, t[:,:,Tensor.ones(1,2,dtype=dtypes.int)+2,:])
    _check_ast_count(1, t[:,:,Tensor.ones(1,1,dtype=dtypes.int),Tensor.zeros(2,1,2,dtype=dtypes.int)])

if __name__ == '__main__':
  unittest.main()
