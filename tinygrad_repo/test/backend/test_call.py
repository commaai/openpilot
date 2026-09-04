import unittest, ctypes
from tinygrad import Tensor, UOp
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.uop.ops import KernelInfo

def call_out_kernel(F:UOp, C:UOp) -> UOp:
  call = F[0].load().call(UOp.const(3).cast(dtypes.int), C[0], ret_dtype=dtypes.void)
  return C.after(call)[1].store(C.after(call)[0].load() + 1).sink(arg=KernelInfo(name="call_out"))

def call_ret_kernel(F:UOp, C:UOp) -> UOp:
  val = F[0].load().call(UOp.const(21).cast(dtypes.int), ret_dtype=dtypes.int)
  return C[0].store(val * 2).sink(arg=KernelInfo(name="call_ret"))

@unittest.skipUnless(isinstance(Device["CPU"].renderer, CStyleLanguage), "TODO: CALL is rendered in C style only")
class TestCall(unittest.TestCase):
  def test_call_out_param(self):
    called = []
    @ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.POINTER(ctypes.c_int))
    def fxn(n, out):
      called.append(n)
      out[0] = n * 2
    f = Tensor([ctypes.cast(fxn, ctypes.c_void_p).value], dtype=dtypes.uint64, device="CPU")
    c = Tensor.empty(2, dtype=dtypes.int, device="CPU")
    c = Tensor.custom_kernel(f, c, fxn=call_out_kernel)[1]
    self.assertEqual(c.tolist(), [6, 7])
    self.assertEqual(called, [3])

  def test_call_ret(self):
    @ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)
    def fxn(n): return n + 1
    f = Tensor([ctypes.cast(fxn, ctypes.c_void_p).value], dtype=dtypes.uint64, device="CPU")
    c = Tensor.empty(1, dtype=dtypes.int, device="CPU")
    c = Tensor.custom_kernel(f, c, fxn=call_ret_kernel)[1]
    c.realize()
    self.assertEqual(c.item(), 44)

if __name__ == "__main__": unittest.main()
