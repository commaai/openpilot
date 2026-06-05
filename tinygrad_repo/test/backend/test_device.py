import unittest
from tinygrad import Device

class TestDeviceCount(unittest.TestCase):
  def test_count(self):
    self.assertGreaterEqual(Device[Device.DEFAULT].count(), 1)

if __name__ == "__main__":
  unittest.main()
