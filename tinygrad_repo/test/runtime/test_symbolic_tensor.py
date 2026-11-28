import unittest
from tinygrad import Variable, Context
from tinygrad.tensor import Tensor

class TestSymbolicPad(unittest.TestCase):
  def test_pad(self):
    v = Variable("v", 1, 100).bind(5)
    t = Tensor.ones(100)[:v].pad(((4, 0),))
    t = t[:9]
    assert t.tolist() == [0,0,0,0,1,1,1,1,1]

  def test_pad_image(self):
    v = Variable("v", 1, 4).bind(2)
    t = Tensor(list(range(1, 17)), dtype="float", device="PYTHON").reshape(4,4).realize()
    with Context(IMAGE=1):
      assert t[:v].pad(((0, 5-v), None)).tolist() == [[1,2,3,4],[5,6,7,8]] + [[0]*4]*3

if __name__ == '__main__':
  unittest.main()
