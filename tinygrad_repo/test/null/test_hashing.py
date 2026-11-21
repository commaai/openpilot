import random, unittest
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import DEV
supported_dtypes = Device[Device.DEFAULT].renderer.supported_dtypes()

@unittest.skipUnless(dtypes.uint8 in supported_dtypes and dtypes.uint64 in supported_dtypes, "Device must support uint8 and uint64")
@unittest.skipIf(DEV.interface.startswith("MOCK") and Device.DEFAULT == "NV", "crashes in NV CI")
class TestKeccak(unittest.TestCase):
  def setUp(self) -> None: random.seed(1337)

  def test_shape_keeping(self):
    s = (1, 2, 3, 4)
    for i in range(len(s)):
      out_shape = Tensor.randint(*s[i:], high=255, dtype=dtypes.uint8).keccak().shape
      self.assertTupleEqual(s[i:-1], out_shape[:-1])

if __name__ == '__main__':
  unittest.main()
