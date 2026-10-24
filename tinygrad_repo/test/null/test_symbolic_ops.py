import unittest
from tinygrad import Tensor, Variable

class TestSymbolicOps(unittest.TestCase):
  def test_invalid_symbolic_reshape(self):
    a = Tensor.rand(30)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      # Cannot reshape into symbolic from non-symbolic
      with self.assertRaises(ValueError): a.reshape((3, vi))

if __name__ == '__main__':
  unittest.main()
