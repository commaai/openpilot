import unittest
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import DEV
from tinygrad.uop.ops import UOp

supported_dtypes = Device[Device.DEFAULT].renderer.supported_dtypes()

@unittest.skipUnless(dtypes.uint8 in supported_dtypes and dtypes.uint64 in supported_dtypes, "Device must support uint8 and uint64")
@unittest.skipIf(DEV.interface.startswith("MOCK") and Device.DEFAULT == "NV", "crashes in NV CI")
class TestKeccak(unittest.TestCase):
  def test_variable_bs(self):
    data = Tensor([b"abc", b"abc", b"def"], dtype=dtypes.uint8).repeat(2048, 1)
    bs = UOp.variable("bs", 1, 4096).bind(3)
    out = data.shrink_to(bs, data.shape[-1]).keccak().shrink_to(3, 32).realize()
    self.assertEqual(bytes(out[0].tolist()), bytearray.fromhex("3a985da74fe225b2 045c172d6bd390bd 855f086e3e9d525b 46bfe24511431532"))
    self.assertEqual(bytes(out[1].tolist()), bytearray.fromhex("3a985da74fe225b2 045c172d6bd390bd 855f086e3e9d525b 46bfe24511431532"))
    self.assertEqual(bytes(out[2].tolist()), bytearray.fromhex("8e0d8f672252acb0 ffc5093db8653b18 1513bf9a2097e737 b4f73533dcaf46df"))

if __name__ == "__main__":
  unittest.main()
