import unittest
import multiprocessing.shared_memory as shared_memory
from tinygrad.helpers import CI
from tinygrad.tensor import Tensor, Device
import numpy as np

class TestRawShmBuffer(unittest.TestCase):
  def test_e2e(self):
    t = Tensor.randn(2, 2, 2).realize()

    # copy to shm
    shm_name = (s := shared_memory.SharedMemory(create=True, size=t.nbytes())).name
    s.close()
    t_shm = t.to(f"disk:shm:{shm_name}").realize()

    # copy from shm
    t2 = t_shm.to(Device.DEFAULT).realize()

    assert np.allclose(t.numpy(), t2.numpy())
    s.unlink()

  @unittest.skipIf(CI, "CI doesn't like big shared memory")
  def test_e2e_big(self):
    t = Tensor.randn(2048, 2048, 8).realize()

    # copy to shm
    shm_name = (s := shared_memory.SharedMemory(create=True, size=t.nbytes())).name
    s.close()
    t_shm = t.to(f"disk:shm:{shm_name}").realize()

    # copy from shm
    t2 = t_shm.to(Device.DEFAULT).realize()

    assert np.allclose(t.numpy(), t2.numpy())
    s.unlink()

if __name__ == "__main__":
  unittest.main()
