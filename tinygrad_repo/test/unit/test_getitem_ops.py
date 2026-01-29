import unittest
import numpy as np
from tinygrad import Tensor, GlobalCounters

class TestGetitemOps(unittest.TestCase):
  def test_two_tensor_indices(self):
    # linear indexing is O(idx_size), one-hot masks is O(idx_size * src_size)
    src_np = np.random.rand(10, 100, 200).astype(np.float32)
    idx1_np, idx2_np = np.random.randint(0, 100, (50, 60), dtype=np.int32), np.random.randint(0, 200, (50, 60), dtype=np.int32)
    src, idx1, idx2 = Tensor(src_np), Tensor(idx1_np), Tensor(idx2_np)
    # O(50*60) = 3K vs O(50*60*100*200) = 60M
    GlobalCounters.reset()
    np.testing.assert_allclose(src_np[0, idx1_np, idx2_np], src[0, idx1, idx2].numpy())
    self.assertLess(GlobalCounters.global_ops, 50_000)
    # consecutive indices not starting from dim 0: O(10*50*60) = 30K vs O(10*50*60*100*200) = 600M
    GlobalCounters.reset()
    np.testing.assert_allclose(src_np[:, idx1_np, idx2_np], src[:, idx1, idx2].numpy())
    self.assertLess(GlobalCounters.global_ops, 500_000)

if __name__ == '__main__':
  unittest.main()
