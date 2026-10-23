import unittest
from tinygrad import Device, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import Ops, UOp
from test.runtime.test_uops import uop, _test_uops_result

class TestLocalAccess(unittest.TestCase):
  # NOTE: webgpu specific, since only webgpu performs bitpacking
  @unittest.skipUnless(Device.DEFAULT == "WEBGPU", "Test local access with packed data type")
  def test_local_packed(self):
    uops = []
    smem = UOp.placeholder((16,), dtypes.uint8, slot=0, addrspace=AddrSpace.LOCAL)
    uops.append(smem)
    st = uop(uops, Ops.STORE, dtypes.void, (smem.index(uop(uops, Ops.CONST, dtypes.int32, (), 0)), uop(uops, Ops.CONST, dtypes.uint8, (), 42)))
    barr = uop(uops, Ops.BARRIER, dtypes.void, (st,))
    sres = smem.after(barr).index(uop(uops, Ops.CONST, dtypes.int32, (), 0))
    self.assertEqual(_test_uops_result(dtypes.uint8, uops, sres), 42)

if __name__ == "__main__": unittest.main()
