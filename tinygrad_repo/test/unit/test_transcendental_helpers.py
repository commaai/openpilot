import unittest, math
import numpy as np
from tinygrad import dtypes
from tinygrad.uop.ops import UOp, Ops
from tinygrad.uop.transcendental import TRANSCENDENTAL_SUPPORTED_DTYPES, payne_hanek_reduction, cody_waite_reduction
from tinygrad.uop.transcendental import frexp, rintk, xpow, xexp2, xlog2, trig_poly, pow2if
from test.helpers import eval_uop

class TestTranscendentalFunctions(unittest.TestCase):
  def test_payne_hanek_reduction(self):
    # TODO: Test constant input when constant folding is fixed (or maybe test both variants)
    # Load input value from a buffer to prevent constant folding
    input_buf = UOp(Ops.DEFINE_GLOBAL, dtypes.double.ptr(), arg=1, src=())
    loaded_value = UOp.load(input_buf.index(UOp.const(dtypes.int, 0)), dtype=dtypes.double)
    def eval_payne_hanek_reduction(v:float) -> tuple[float, int]:
      return tuple(eval_uop(u, [(dtypes.float64, [v])]) for u in payne_hanek_reduction(loaded_value))

    r, q = eval_payne_hanek_reduction(12 * math.pi + 0.1)
    np.testing.assert_allclose(r, 0.1 - math.pi / 2)
    np.testing.assert_equal(q, 1)

    r, q = eval_payne_hanek_reduction(12 * math.pi)
    np.testing.assert_allclose(r, 0.0, atol=1e-8)
    np.testing.assert_equal(q, 4)

    r, q = eval_payne_hanek_reduction(12 * math.pi - 0.1)
    np.testing.assert_allclose(r, -0.1)
    np.testing.assert_equal(q, 4)

  def test_cody_waite_reduction(self):
    r, q = (eval_uop(u) for u in cody_waite_reduction(UOp.const(dtypes.float64, 12 * math.pi + 0.1)))
    np.testing.assert_allclose(r, 0.1)
    np.testing.assert_equal(q, 12)

  def test_frexp(self):
    for x in (1, -1):
      mantissa, exponent = (eval_uop(u) for u in frexp(UOp.const(dtypes.float64, x)))
      np.testing.assert_equal(mantissa, 0.5)
      np.testing.assert_equal(exponent, 1)

    for x in (2, -2):
      mantissa, exponent = (eval_uop(u) for u in frexp(UOp.const(dtypes.float64, 2.0)))
      np.testing.assert_equal(mantissa, 0.5)
      np.testing.assert_equal(exponent, 2)

    mantissa, exponent = (eval_uop(u) for u in frexp(UOp.const(dtypes.float64, 5.0)))
    np.testing.assert_equal(mantissa, 0.625)
    np.testing.assert_equal(exponent, 3)

    mantissa, exponent = (eval_uop(u) for u in frexp(UOp.const(dtypes.float64, 1000.0)))
    np.testing.assert_allclose(mantissa, 0.9765625)
    np.testing.assert_equal(exponent, 10)

  def test_rintk(self):
    np.testing.assert_allclose(eval_uop(rintk(UOp.const(dtypes.float, 0.0))), 0)
    np.testing.assert_allclose(eval_uop(rintk(UOp.const(dtypes.float, 5.0))), 5)
    np.testing.assert_allclose(eval_uop(rintk(UOp.const(dtypes.float, 5.5))), 6)
    np.testing.assert_allclose(eval_uop(rintk(UOp.const(dtypes.float, 5.999))), 6)
    np.testing.assert_allclose(eval_uop(rintk(UOp.const(dtypes.float, -5.0))), -5)
    np.testing.assert_allclose(eval_uop(rintk(UOp.const(dtypes.float, -5.5))), -6)
    np.testing.assert_allclose(eval_uop(rintk(UOp.const(dtypes.float, -5.999))), -6)

  def test_pow2if(self):
    np.testing.assert_allclose(eval_uop(pow2if(UOp.const(dtypes.int, 0), dtypes.float)), 1.0)
    np.testing.assert_allclose(eval_uop(pow2if(UOp.const(dtypes.int, 1), dtypes.float)), 2.0)
    np.testing.assert_allclose(eval_uop(pow2if(UOp.const(dtypes.int, 2), dtypes.float)), 4.0)
    np.testing.assert_allclose(eval_uop(pow2if(UOp.const(dtypes.int, 10), dtypes.float)), 1024.0)
    np.testing.assert_allclose(eval_uop(pow2if(UOp.const(dtypes.int, 63), dtypes.float)), 2**63)
    np.testing.assert_allclose(eval_uop(pow2if(UOp.const(dtypes.int, -1), dtypes.float)), 0.5)
    np.testing.assert_allclose(eval_uop(pow2if(UOp.const(dtypes.int, -2), dtypes.float)), 0.25)
    np.testing.assert_allclose(eval_uop(pow2if(UOp.const(dtypes.int, -10), dtypes.float)), 2**-10)
    np.testing.assert_allclose(eval_uop(pow2if(UOp.const(dtypes.int, -63), dtypes.float)), 2**-63)

class TestTranscendentalVectorizedFunctions(unittest.TestCase):
  # given a scalar and vectorized input, check that the fxn outputs have the same
  # scalar_dtypes, args, ops, and vcount (only for vectorized input)

  def _check_uop_vcount(self, u:tuple|UOp, vcount:int):
    # check all UOps in u are vectorized with vcount
    if isinstance(u, UOp):
      assert u.dtype.vcount == vcount, f'expected {vcount=} but got {u.dtype.vcount=} for UOp\n{u=}'
    [self._check_uop_vcount(x, vcount) for x in (u if isinstance(u, tuple) else u.src)]

  def _check_uops_match(self, u1:tuple|UOp, u2:tuple|UOp):
    # check all UOps in u1, u2 have the same scalar_dtype, args, ops
    if isinstance(u1, UOp) and isinstance(u2, UOp):
      assert u1.dtype.scalar() == u2.dtype.scalar(), f'expected {u1.dtype.scalar()=} but got {u2.dtype.scalar()=} for UOps\n{u1=}\n{u2}'
      assert u1.arg == u2.arg or (math.isnan(u1.arg) and math.isnan(u2.arg)), f'expected {u1.arg=} but got {u2.arg=} for UOps\n{u1=}\n{u2}'
      assert u1.op == u2.op, f'expected {u1.op=} but got {u2.op=} for UOps\n{u1=}\n{u2}'
    [self._check_uops_match(x1, x2) for x1, x2 in zip((u1 if isinstance(u1, tuple) else u1.src), (u2 if isinstance(u2, tuple) else u2.src))]

  def _test_vectorized(self, fxn, scalar_dtypes=TRANSCENDENTAL_SUPPORTED_DTYPES, vals=[-2,1.3,194], vcounts=[1,4,19]):
    for scalar_dtype in scalar_dtypes:
      for val in vals:
        for vcount in vcounts:
          in_scalar, in_vec = UOp.const(scalar_dtype, val), UOp.const(scalar_dtype.vec(vcount), val)
          out_scalar, out_vec = fxn(in_scalar), fxn(in_vec)
          self._check_uops_match(out_scalar, out_vec)
          self._check_uop_vcount(out_vec, vcount)

  def test_xpow(self): return self._test_vectorized(lambda x: xpow(x, x))
  def test_xexp2(self): return self._test_vectorized(xexp2)
  def test_xlog2(self): return self._test_vectorized(xlog2)
  def test_payne_hanek_reduction(self): return self._test_vectorized(payne_hanek_reduction)
  def test_cody_waite_reduction(self): return self._test_vectorized(cody_waite_reduction)
  def test_trig_poly(self): return self._test_vectorized(lambda x: trig_poly(x, [0.0], [1.0]))

if __name__ == '__main__':
  unittest.main()
