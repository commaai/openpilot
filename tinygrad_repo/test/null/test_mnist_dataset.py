import unittest
from tinygrad.helpers import GlobalCounters
from tinygrad.nn.datasets import mnist
from test.helpers import KernelCountException

class TestDataset(unittest.TestCase):
  def test_dataset_is_realized(self):
    X_train, _, _, _ = mnist()
    X_train[0].contiguous().realize()
    GlobalCounters.reset()
    X_train[0].contiguous().realize()
    if GlobalCounters.kernel_count > 1: raise KernelCountException(1, GlobalCounters.kernel_count)  # 0 if SLICE (zero-copy), 1 otherwise

if __name__ == '__main__':
  unittest.main()
