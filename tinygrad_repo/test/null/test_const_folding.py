import unittest, math, itertools
from tinygrad import Tensor, Device, Context, dtypes
from tinygrad.dtype import DTYPES_DICT, DType, ConstType
from tinygrad.uop.ops import Ops, UOp, GroupOp
from tinygrad.codegen.decomp.op import threefry2x32
from test.helpers import full_rewrite

def _check_ast_count(desired_count:int, t:Tensor):
  # NOTE: this has side effect because everything can be scheduled only once
  schedule = t.schedule_linear()
  asts = [s for s in schedule.src if s.src[0].op is Ops.SINK]
  len(asts)
  # NOT SUPPORTED ANYMORE
  #assert len(asts) == desired_count, f"{len(asts)} != {desired_count}"

class TestWeakConstFolding(unittest.TestCase):
  def test_weakint_math(self):
    out = (UOp.const(2**40) + UOp.const(2**40)).simplify()
    self.assertEqual((out.op, out.dtype, out.val), (Ops.CONST, dtypes.weakint, 2**41))

  def test_float_unaries(self):
    for op in (Ops.SIN, Ops.LOG2, Ops.EXP2, Ops.SQRT, Ops.RECIPROCAL):
      out = UOp.const(4.0).alu(op).simplify()
      self.assertEqual((out.op, out.dtype), (Ops.CONST, dtypes.weakfloat))

  def test_weakfloat_math(self):
    out = (UOp.const(1.25) + UOp.const(2.5)).simplify()
    self.assertEqual((out.op, out.dtype, out.val), (Ops.CONST, dtypes.weakfloat, 3.75))

  def test_nan_compare(self):
    nan = UOp.const(math.nan)
    self.assertTrue(nan.ne(nan).simplify().val)

  def test_invalid_poison(self):
    self.assertTrue(UOp.invalid().alu(Ops.CDIV, UOp.const(0)).simplify().is_invalid)

class TestBitcastConstFolding(unittest.TestCase):
  def test_out_of_range_source_value(self):
    for val, src_dt, dst_dt, bits in ((3000000000, dtypes.int32, dtypes.uint32, 3000000000),
                                      (70000, dtypes.int16, dtypes.uint16, 4464),
                                      (-5, dtypes.uint32, dtypes.int32, -5)):
      self.assertIs(UOp.const(val, src_dt).bitcast(dst_dt).simplify(), UOp.const(bits, dst_dt))

  def test_scalar_bitcast(self):
    def t(cases: dict[DType, ConstType]):
      for (from_dt, from_v), (to_dt, to_v) in itertools.product(cases.items(), cases.items()):
        if not math.isnan(from_v):
          r = UOp.const(from_v, from_dt).bitcast(to_dt).simplify()
          self.assertIs(r, UOp.const(to_v, to_dt), f"{from_dt} -> {to_dt} ({from_v} -> {to_v})")

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
      result = full_rewrite(UOp.const((-1, -2**31, 75), dtypes.int32).bitcast(dtypes.uint32).sink())
      expected = full_rewrite(UOp.const((2**32-1, 2**31, 75), dtypes.uint32).sink())
    self.assertEqual(result.src, expected.src)

class TestMovedConstFolding(unittest.TestCase):
  def test_contiguous_deviceless_const(self):
    t = Tensor(UOp.const(2.0, dtypes.float)).contiguous()
    self.assertIs(t.uop, UOp.const(2.0, dtypes.float))
    self.assertIsNone(t.uop.device)

  def test_add_shrunk_zero(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) + Tensor.zeros(6).shrink(((1, 5),)))

  def test_add_padded_zero(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) + Tensor.zeros(2).pad(((1, 1),)))

  def test_mul_shrunk_one(self):
    _check_ast_count(0, Tensor([1.0, 2, 3, 4]) * Tensor.ones(6).shrink(((1, 5),)))

  def test_add_padded_one(self):
    _check_ast_count(1, Tensor([1.0, 2, 3, 4]) * Tensor.ones(2).pad(((1, 1),)))

class TestReduceOpsConstFolding(unittest.TestCase):
  def test_sum_output_dtype(self):
    # sum output dtype can be different from input
    for dt in DTYPES_DICT.values():
      if dt in Device[Device.DEFAULT].renderer.supported_dtypes():
        t = Tensor.ones(16, dtype=dt).reshape(4, 4)
        assert t.sum().dtype == t.contiguous().sum().dtype

class TestThreefryConstFolding(unittest.TestCase):
  def test_threefry(self):
    # THREEFRY(const,const) folds to a const once decomposed
    x = threefry2x32(UOp.const(5, dtypes.uint64), UOp.const(10, dtypes.uint64)).simplify()
    self.assertEqual([u.op for u in x.toposort() if u.op in GroupOp.ALU], [])

if __name__ == '__main__':
  unittest.main()
