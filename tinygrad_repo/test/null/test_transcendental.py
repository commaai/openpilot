import unittest
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import Context
from test.helpers import check_schedule
supported_dtypes = Device[Device.DEFAULT].renderer.supported_dtypes()

class TestTranscendentalSchedule(unittest.TestCase):
  @unittest.skipUnless(dtypes.ulong in supported_dtypes, "Needs ulong")
  def test_transcendental_sin_fusion(self):
    with Context(TRANSCENDENTAL=2):
      a = Tensor.empty(10)
      b = Tensor.empty(10)
      c = a.sin() + b.sin()
      c = c.sin()
      check_schedule(c, 1)

  def test_transcendental_log2_fusion(self):
    with Context(TRANSCENDENTAL=2):
      a = Tensor.empty(10)
      b = Tensor.empty(10)
      c = a.log2() + b.log2()
      c = c.log2()
      check_schedule(c, 1)

  def test_transcendental_exp2_fusion(self):
    with Context(TRANSCENDENTAL=2):
      a = Tensor.empty(10)
      b = Tensor.empty(10)
      c = a.exp2() + b.exp2()
      c = c.exp2()
      check_schedule(c, 1)

if __name__ == '__main__':
  unittest.main()
