from tinygrad.tensor import Tensor
import numpy as np
import pickle
import unittest

class TestToNumpy(unittest.TestCase):
  def test_numpy_is_numpy(self):
    output = Tensor.ones((1, 3, 4096)).realize().numpy()
    new = np.copy(output)
    print(type(new))
    serialized = pickle.dumps(new)
    out = pickle.loads(serialized)
    assert out.shape == (1,3,4096)
    assert (out==1).all()

if __name__ == '__main__':
  unittest.main()