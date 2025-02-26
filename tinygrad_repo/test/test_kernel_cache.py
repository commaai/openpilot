#!/usr/bin/env python
import unittest
from tinygrad.tensor import Tensor
from tinygrad import Device

class TestKernelCache(unittest.TestCase):
  def test_kernel_cache_in_action(self):
    if Device.DEFAULT not in ["CPU"]:
      self.skipTest("No custom kernel cache is implemented")

    unique_const = 0.6765677269
    a = Tensor.rand(4,4).realize()
    b = Tensor.rand(4,4).realize()
    x = a + b + unique_const
    x.realize()

    a1 = Tensor.rand(4,4).realize()
    b1 = Tensor.rand(4,4).realize()
    orig_compile_func = Device['CPU'].compiler
    Device['CPU'].compiler = None # making it not callable

    try:
      x1 = a1 + b1 + unique_const
      x1.realize() # Same kernel should be from cache.
    finally:
      Device['CPU'].compiler = orig_compile_func

if __name__ == "__main__":
  unittest.main()
