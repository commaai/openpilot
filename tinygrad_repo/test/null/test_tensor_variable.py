import unittest
from tinygrad import Tensor, Variable

class TestTensorVariable(unittest.TestCase):
  def test_symbolic_chunk_error_on_symbolic_dim(self):
    # chunk should fail when trying to split along a symbolic dimension
    vv = Variable("a", 1, 10).bind(4)
    t = Tensor.ones(10, 8).contiguous()[:vv, :]  # shape (vv, 8)
    with self.assertRaises(AssertionError):
      t.chunk(2, dim=0)  # can't split along symbolic dim

if __name__ == '__main__':
  unittest.main()
