import numpy as np
import unittest
from tinygrad import Tensor
from tinygrad.helpers import get_single_element
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.engine.realize import run_linear
from tinygrad.uop.ops import Ops, UOp
from test.helpers import replace_opts

class TestOptGemm(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    N = 64
    cls.a = Tensor.randn(N, N).contiguous().realize()
    cls.b = Tensor.randn(N, N).contiguous().realize()
    with np.errstate(all='ignore'):
      cls.res = cls.a.T.numpy() @ cls.b.T.numpy()

  def _test_gemm_unrolled_permute_l(self, opts=[]):
    t = self.a.T @ self.b.T
    # TODO: this should be a generic test helper
    call = get_single_element(t.schedule_linear().src)
    new_call = call.replace(src=(replace_opts(call.src[0], opts), *call.src[1:]))
    run_linear(UOp(Ops.LINEAR, src=(new_call,)))
    test = call.src[1].buffer.numpy().reshape(self.res.shape)
    np.testing.assert_allclose(self.res, test, atol=1e-4)

  def test_gemm_unrolled_permute_l_44(self):
    opts = [Opt(op=OptOps.UPCAST, axis=0, arg=4), Opt(op=OptOps.UPCAST, axis=1, arg=4)]
    self._test_gemm_unrolled_permute_l(opts)

  def test_gemm_unrolled_permute_l_424(self):
    # was failing with LLVM
    opts = [Opt(op=OptOps.UPCAST, axis=0, arg=4), Opt(op=OptOps.UPCAST, axis=1, arg=2), Opt(op=OptOps.UPCAST, axis=0, arg=4)]
    self._test_gemm_unrolled_permute_l(opts)

  def test_gemm_unrolled_permute_l_42(self):
    opts = [Opt(op=OptOps.UPCAST, axis=0, arg=4), Opt(op=OptOps.UPCAST, axis=1, arg=2)]
    self._test_gemm_unrolled_permute_l(opts)

  def test_gemm_unrolled_permute_l_22(self):
    opts = [Opt(op=OptOps.UPCAST, axis=0, arg=2), Opt(op=OptOps.UPCAST, axis=1, arg=2)]
    self._test_gemm_unrolled_permute_l(opts)

if __name__ == '__main__':
  unittest.main()
