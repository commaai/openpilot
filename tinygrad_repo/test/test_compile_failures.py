import unittest
from tinygrad import Tensor, dtypes, Device
from tinygrad.engine.realize import lower_schedule
from tinygrad.device import is_dtype_supported

class TestCompileFailures(unittest.TestCase):
  def compile(self, out:Tensor):
    for _ in lower_schedule(out.schedule()): pass

  @unittest.skipUnless(is_dtype_supported(dtypes.uchar, Device.DEFAULT), f"no uint8 on {Device.DEFAULT}")
  def test_interpolate_atari(self):
    self.compile(Tensor.empty(210, 160, dtype='uint8').interpolate((64, 64)))

  def test_add_max_uchar(self):
    self.compile((Tensor.empty(1024, dtype='uint8') + Tensor.empty(1024, dtype='uint8')).max())

if __name__ == '__main__':
  unittest.main()
