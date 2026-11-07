import unittest
from tinygrad import Device, dtypes
from tinygrad.uop.ops import Ops
from test.runtime.test_uops import TestUOps

class TestFloatUOps(TestUOps):
  @unittest.skipUnless(Device.DEFAULT == "PYTHON", "only python supports MULACC")
  def test_mulacc(self):
    self._test_top_fxn(Ops.MULACC, lambda a,b,c: a*b+c, (dtypes.float, dtypes.float, dtypes.float))

if __name__ == "__main__": unittest.main()
