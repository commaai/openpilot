import unittest
from tinygrad import Tensor, Device
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp, KernelInfo
from tinygrad.renderer.cstyle import CStyleLanguage

# an external call is a CALL on a CUSTOM_FUNCTION body naming the symbol, the program links against the loaded libraries
def call_out_kernel(C:UOp) -> UOp: # frexp writes the exponent through its pointer arg
  call = UOp.custom_function("frexp", dtype=dtypes.float64).call(UOp.const(8.0, dtypes.float64), C[0])
  return C.after(call)[1].store(C.after(call)[0].load() + 1).sink(arg=KernelInfo(name="call_out"))

def call_ret_kernel(C:UOp) -> UOp:
  val = UOp.custom_function("sqrt", dtype=dtypes.float64).call(UOp.const(16.0, dtypes.float64))
  return C[0].store(val.cast(dtypes.int) * 2).sink(arg=KernelInfo(name="call_ret"))

@unittest.skipUnless(isinstance(Device["CPU"].renderer, CStyleLanguage), "TODO: CALL is rendered in C style only")
class TestExternalCall(unittest.TestCase):
  def test_call_out_param(self):
    c = Tensor.custom_kernel(Tensor.empty(2, dtype=dtypes.int, device="CPU"), fxn=call_out_kernel)[0]
    self.assertEqual(c.tolist(), [4, 5])

  def test_call_ret(self):
    c = Tensor.custom_kernel(Tensor.empty(1, dtype=dtypes.int, device="CPU"), fxn=call_ret_kernel)[0]
    self.assertEqual(c.item(), 8)

if __name__ == "__main__": unittest.main()
