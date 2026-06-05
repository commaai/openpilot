import unittest
from tinygrad import Device
from tinygrad.helpers import Timing, Profiling

class TestDeviceSpeed(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.dev = Device[Device.DEFAULT]
    cls.empty = Device[Device.DEFAULT].renderer.render([])

  def test_empty_compile(self):
    with Timing("compiler "):
      self.dev.compiler.compile(self.empty)

  def test_empty_compile_twice(self):
    self.dev.compiler.compile(self.empty)
    with Timing("compiler "):
      self.dev.compiler.compile(self.empty)

  def test_launch_speed(self):
    prg_bin = self.dev.compiler.compile(self.empty)
    prg = self.dev.runtime("test", prg_bin)
    for _ in range(10): prg() # ignore first launches
    with Timing("launch 1000x "):
      for _ in range(1000): prg()
    with Timing("launch 1000x with wait "):
      for _ in range(1000): prg(wait=True)

  def test_profile_launch_speed(self):
    prg_bin = self.dev.compiler.compile(self.empty)
    prg = self.dev.runtime("test", prg_bin)
    for _ in range(10): prg() # ignore first launches
    with Profiling():
      for _ in range(1000): prg()

if __name__ == '__main__':
  unittest.main()
