import unittest
from tinygrad import Tensor, Device
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp, KernelInfo, Ops
from tinygrad.renderer.cstyle import CStyleLanguage

# an external call is a CALL on a NATIVE body naming the symbol, the program links against the loaded libraries
def call_out_kernel(C:UOp) -> UOp: # frexp writes the exponent through its pointer arg
  call = UOp.native("frexp", dtype=dtypes.float64).call(UOp.const(8.0, dtypes.float64), C[0])
  return C.after(call)[1].store(C.after(call)[0].load() + 1).sink(arg=KernelInfo(name="call_out"))

def call_shrink_kernel(C:UOp, D:UOp) -> UOp: # frexp writes the exponent of r every iteration, D gates r to r<2
  r = UOp.range(4, 0)
  call = UOp.native("frexp", dtype=dtypes.float64).call(r.cast(dtypes.float64), C[0])
  return D[r.valid(r<2)].store(call.cast(dtypes.int)).end(r).sink(arg=KernelInfo(name="call_shrink", opts_to_apply=()))

def call_collapse_kernel(C:UOp, O:UOp) -> UOp: # frexp writes the exponent of r+1 to C[r] every iteration, the sum only picks r==2
  r = UOp.range(4, 0)
  call = UOp.native("frexp", dtype=dtypes.float64).call((r+1).cast(dtypes.float64), C.index(r))
  return O[0].store(r.eq(2).where(call.cast(dtypes.int)+1, 0).reduce(r, arg=Ops.ADD)).sink(arg=KernelInfo(name="call_collapse", opts_to_apply=()))

def call_ret_kernel(C:UOp) -> UOp:
  val = UOp.native("sqrt", dtype=dtypes.float64).call(UOp.const(16.0, dtypes.float64))
  return C[0].store(val.cast(dtypes.int) * 2).sink(arg=KernelInfo(name="call_ret"))

@unittest.skipUnless(isinstance(Device["CPU"].renderer, CStyleLanguage), "TODO: CALL is rendered in C style only")
class TestExternalCall(unittest.TestCase):
  def test_call_out_param(self):
    c = Tensor.custom_kernel(Tensor.empty(2, dtype=dtypes.int, device="CPU"), fxn=call_out_kernel)[0]
    self.assertEqual(c.tolist(), [4, 5])

  def test_call_keeps_range(self):
    c = Tensor.custom_kernel(Tensor.empty(1, dtype=dtypes.int, device="CPU"), Tensor.empty(4, dtype=dtypes.int, device="CPU"),
                             fxn=call_shrink_kernel)[0]
    self.assertEqual(c.tolist(), [2])

  def test_call_keeps_reduce(self):
    c, o = Tensor.custom_kernel(Tensor.empty(4, dtype=dtypes.int, device="CPU"), Tensor.empty(1, dtype=dtypes.int, device="CPU"),
                                fxn=call_collapse_kernel)[:2]
    Tensor.realize(c, o)
    self.assertEqual((c.tolist(), o.tolist()), ([1, 2, 2, 3], [1]))

  def test_call_ret(self):
    c = Tensor.custom_kernel(Tensor.empty(1, dtype=dtypes.int, device="CPU"), fxn=call_ret_kernel)[0]
    self.assertEqual(c.item(), 8)

if __name__ == "__main__": unittest.main()
