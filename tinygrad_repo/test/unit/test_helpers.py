import unittest
import numpy as np
from tinygrad.helpers import polyN, is_numpy_ndarray
from tinygrad.tensor import Tensor

class TestPolyN(unittest.TestCase):
  def test_tensor(self):
    np.testing.assert_allclose(polyN(Tensor([1.0, 2.0, 3.0, 4.0]), [1.0, -2.0, 1.0]).numpy(), [0.0, 1.0, 4.0, 9.0])

class TestIsNumpyNdarray(unittest.TestCase):
  def test_tensor_numpy(self):
    self.assertTrue(is_numpy_ndarray(Tensor([1, 2, 3]).numpy()))

if __name__ == '__main__':
  unittest.main()
