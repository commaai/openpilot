import unittest
from tinygrad import Tensor
from tinygrad.nn.state import TensorIO

class TestTensorIO(unittest.TestCase):
  def test_read(self):
    data = b"Hello World!"
    fobj = TensorIO(Tensor(data))
    self.assertEqual(fobj.read(1), data[:1])
    self.assertEqual(fobj.read(5), data[1:6])
    self.assertEqual(fobj.read(100), data[6:])
    self.assertEqual(fobj.read(100), b"")

  def test_read_nolen(self):
    data = b"Hello World!"
    fobj = TensorIO(Tensor(data))
    fobj.seek(2)
    self.assertEqual(fobj.read(), data[2:])

if __name__ == '__main__':
  unittest.main()
