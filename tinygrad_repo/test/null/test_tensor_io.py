import unittest
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import TensorIO

class TestTensorIO(unittest.TestCase):
  def test_create(self):
    with self.assertRaises(ValueError):
      TensorIO(Tensor(b"Hello World").reshape(1, -1))
    with self.assertRaises(ValueError):
      TensorIO(Tensor([], dtype=dtypes.int64).reshape(1, -1))

  def test_seek(self):
    t = Tensor(b"Hello World!")
    fobj = TensorIO(t)
    self.assertEqual(fobj.tell(), 0)
    self.assertEqual(fobj.seek(1), 1)
    self.assertEqual(fobj.seek(-2, 2), len(t) - 2)
    self.assertEqual(fobj.seek(1, 1), len(t) - 1)
    self.assertEqual(fobj.seek(10, 1), len(t))
    self.assertEqual(fobj.seek(10, 2), len(t))
    self.assertEqual(fobj.seek(-10, 0), 0)

if __name__ == '__main__':
  unittest.main()
