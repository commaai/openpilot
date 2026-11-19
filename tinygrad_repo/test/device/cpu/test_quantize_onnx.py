import unittest
import numpy as np
from tinygrad import Tensor, Context, Device, dtypes
from tinygrad.uop.ops import Ops
from tinygrad.dtype import AddrSpace
from tinygrad.codegen import to_program
from test.device.dsp.test_quantize_onnx import get_quantized_model

@unittest.skip("this is broken")
@unittest.skipIf(Device.DEFAULT != "CPU", "only tests for CPU")
class TestQuantizeOnnxCPU(unittest.TestCase):
  def test_quant_128(self, sz=128):
    try:
      import onnx # noqa: F401 # pylint: disable=unused-import
    except ImportError:
      raise unittest.SkipTest()
    from tinygrad.nn.onnx import OnnxRunner
    out_file = get_quantized_model(sz)
    run_onnx = OnnxRunner(out_file)
    inp = Tensor(np.random.uniform(size=(sz, sz)).astype(np.float32))
    with Context(QUANTIZE=1):
      linear = run_onnx({"input":inp})["output"].schedule_linear()
      prg = to_program(linear.src[-2].src[0], renderer=Device[Device.DEFAULT].renderer)
      daccs = [u for u in tuple(prg.src[1].src) if u.op is Ops.BUFFER and u.addrspace is AddrSpace.REG]
      assert all(u.dtype is dtypes.int for u in daccs)

if __name__ == "__main__": unittest.main()
