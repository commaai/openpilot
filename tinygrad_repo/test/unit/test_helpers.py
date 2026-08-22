import unittest, gc
import numpy as np
from tinygrad.helpers import polyN, disable_gc
from tinygrad.tensor import Tensor, is_numpy_ndarray

class TestPolyN(unittest.TestCase):
  def test_tensor(self):
    np.testing.assert_allclose(polyN(Tensor([1.0, 2.0, 3.0, 4.0]), [1.0, -2.0, 1.0]).numpy(), [0.0, 1.0, 4.0, 9.0])

class TestIsNumpyNdarray(unittest.TestCase):
  def test_tensor_numpy(self):
    self.assertTrue(is_numpy_ndarray(Tensor([1, 2, 3]).numpy()))

class TestDisableGC(unittest.TestCase):
  def test_recursive_decorator(self):
    was_enabled = gc.isenabled()
    @disable_gc()
    def recurse(depth:int):
      self.assertFalse(gc.isenabled())
      if depth: recurse(depth-1)
      self.assertFalse(gc.isenabled())
    try:
      recurse(2)
      self.assertEqual(gc.isenabled(), was_enabled)
    finally:
      (gc.enable if was_enabled else gc.disable)()

if __name__ == '__main__':
  unittest.main()
