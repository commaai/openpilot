import unittest
from tinygrad import Tensor

class TestLoadStore(unittest.TestCase):
  def test_load_shape(self):
    t = Tensor(bytes(16)).fs_load(1024)
    assert t.shape == (1024,), t.shape

  def test_store_shape(self):
    t = Tensor.zeros(1024).fs_store()
    assert t.shape == (16,), t.shape

  def test_load_large_shape(self):
    t = Tensor(bytes(16)).fs_load(10_000_000)
    assert t.shape == (10_000_000,), t.shape

  def test_store_large_shape(self):
    t = Tensor.zeros(10_000_000).fs_store()
    assert t.shape == (16,), t.shape

if __name__ == "__main__":
  unittest.main()
