import unittest
from tinygrad import Variable
from tinygrad.tensor import Tensor

class TestSymbolicPad(unittest.TestCase):
  def test_pad(self):
    v = Variable("v", 1, 100).bind(5)
    t = Tensor.ones(100)[:v].pad(((4, 0),))
    t = t[:9]
    assert t.tolist() == [0,0,0,0,1,1,1,1,1]

if __name__ == '__main__':
  unittest.main()
