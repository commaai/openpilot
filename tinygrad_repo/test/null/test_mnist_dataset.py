import unittest
from tinygrad.helpers import GlobalCounters
from tinygrad.nn.datasets import mnist

class TestDataset(unittest.TestCase):
  def test_dataset_is_realized(self):
    X_train, _, _, _ = mnist()
    X_train[0].contiguous().realize()
    GlobalCounters.reset()
    X_train[0].contiguous().realize()
    self.assertLessEqual(GlobalCounters.kernel_count, 1)  # 0 if BUFFER_VIEW (zero-copy), 1 otherwise

if __name__ == '__main__':
  unittest.main()
