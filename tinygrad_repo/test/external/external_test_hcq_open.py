import os
if "DEV" not in os.environ: os.environ["DEV"] = "AMD"

import unittest, time
from tinygrad import Device

class TestOpen(unittest.TestCase):
  def generate_test_open(n):
    def test(self):
      dev = Device[Device.DEFAULT]
      for i in range(10):
        dev.allocator.alloc(10 << 20)
        time.sleep(0.5)
    test.__name__ = f'test_open_{n}'
    return test

  for i in range(64): locals()[f'test_open_{i}'] = generate_test_open(i)

if __name__ == '__main__':
  unittest.main()
