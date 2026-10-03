import unittest, itertools, math
from tinygrad import dtypes, Context
from tinygrad.dtype import DType, ConstType
from tinygrad.uop.ops import Ops, UOp
from test.helpers import full_rewrite

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

if __name__ == '__main__':
  unittest.main()
