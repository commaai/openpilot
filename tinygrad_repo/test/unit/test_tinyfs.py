import unittest
from tinygrad import Tensor

class TestLoadStore(unittest.TestCase):
  def test_load_shape(self):
    t = Tensor(bytes(16)).load(1024).kernelize()
    assert t.shape == (1024,), t.shape

  def test_store_shape(self):
    t = Tensor.zeros(1024).store().kernelize()
    assert t.shape == (16,), t.shape

  def test_load_large_shape(self):
    t = Tensor(bytes(16)).load(10_000_000).kernelize()
    assert t.shape == (10_000_000,), t.shape

  def test_store_large_shape(self):
    t = Tensor.zeros(10_000_000).store().kernelize()
    assert t.shape == (16,), t.shape

if __name__ == "__main__":
  unittest.main()
