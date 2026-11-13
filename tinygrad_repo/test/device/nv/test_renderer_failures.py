# PTX is shared by the NV and CUDA backends; both CI jobs run this suite.
import unittest
import numpy as np
from tinygrad import Tensor, Device, dtypes, UOp
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.uop.ops import Ops, KernelInfo
from test.runtime.test_renderer_failures import _test_uop_result

@unittest.skipIf(not isinstance(Device[Device.DEFAULT].renderer, PTXRenderer), "tests for ptx renderer")
class TestPTXFailures(unittest.TestCase):
  @unittest.skip("INDEX can only have a gate ALU parent, not an IF")
  def test_gated_store_with_if(self):
    a = UOp.param(0, dtypes.int, 4)
    gate_alu = (lidx0:=UOp.special(4, 'lidx0')).ne(0)
    val = UOp.const(1).cast(dtypes.int)
    if_uop = UOp(Ops.IF, src=(gate_alu,))
    gated_alu_store = UOp(Ops.STORE, src=(a.index(lidx0, if_uop), val))
    sink = UOp(Ops.SINK, src=(gated_alu_store,), arg=KernelInfo())
    ret = _test_uop_result([], sink, local_size=[4, 1, 1])[0]
    np.testing.assert_equal(ret, [0, 1, 1, 1])

  @unittest.skipUnless(dtypes.half in Device[Device.DEFAULT].renderer.supported_dtypes(), "need half")
  def test_gated_define_acc_with_half_dtype(self):
    a = Tensor.randn(32, 32, dtype=dtypes.half).realize()
    b = Tensor.randn(34, 32, dtype=dtypes.half).realize()
    result = a.pad((1,1)).matmul(b, dtype=dtypes.half).numpy()
    reference = a.pad((1,1)).matmul(b, dtype=dtypes.float).numpy()
    np.testing.assert_allclose(result, reference, atol=1e-2, rtol=1e-2)

if __name__ == "__main__": unittest.main()
