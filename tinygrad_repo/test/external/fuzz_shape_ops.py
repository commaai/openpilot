import unittest
from tinygrad import Tensor

class TestShapeOps(unittest.TestCase):
  def test_chunk(self):
    self.assertEqual([t.tolist() for t in Tensor([]).chunk(3)], [[], [], []])
    self.assertEqual([t.tolist() for t in Tensor([1, 2]).chunk(3, -1)], [[1], [2]])
    with self.assertRaises(AssertionError): Tensor([1]).chunk(0)

if __name__ == '__main__':
  unittest.main()
