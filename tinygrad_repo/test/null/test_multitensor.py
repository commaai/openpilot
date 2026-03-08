import unittest
from tinygrad import Tensor, GlobalCounters, dtypes

class TestMultiRamUsage(unittest.TestCase):
  def setUp(self):
    self.baseline = GlobalCounters.mem_used
    self.N = 100
  def assertUsed(self, amt, strict=True):
    used = GlobalCounters.mem_used - self.baseline
    print(f"used {used} bytes")
    if strict: self.assertEqual(used, amt)
    else: self.assertLessEqual(used, amt)

  def test_zeros(self):
    _ = Tensor.zeros(self.N, self.N).contiguous().realize()
    self.assertUsed(self.N*self.N*4)

  def test_zeros_del(self):
    _ = Tensor.zeros(self.N, self.N).contiguous().realize()
    del _
    self.assertUsed(0)

  @unittest.skip("flaky")
  def test_zeros_copy(self):
    devices_2 = ("NULL:1", "NULL:2")
    _ = Tensor.zeros(self.N, self.N).contiguous().to(devices_2).realize()
    # NOTE: the first one on the DEFAULT device should be freed
    self.assertUsed(self.N*self.N*4*2)

  @unittest.skip("flaky")
  def test_zeros_shard(self, devices=("NULL:1", "NULL:2")):
    _ = Tensor.zeros(self.N, self.N).contiguous().shard(devices, axis=0).realize()
    self.assertUsed(self.N*self.N*4) # sharding should not increase total ram usage
  def test_zeros_shard_self(self): self.test_zeros_shard(("NULL:0", "NULL:1"))

  @unittest.skip("flaky")
  def test_zeros_contiguous_shard(self):
    devices_2 = ("NULL:1", "NULL:2")
    _ = Tensor.zeros(self.N, self.N).contiguous().shard(devices_2, axis=0).contiguous().realize()
    self.assertUsed(self.N*self.N*4) # sharding should not increase total ram usage

  def _test_matmul_half(self, dev_count:int):
    N = 32
    total_mem = {}
    devs = tuple(f"NULL:{i}" for i in range(dev_count))
    for dtype in {dtypes.float, dtypes.half}:
      GlobalCounters.reset()
      a = Tensor.empty((N, N), dtype=dtype, device=devs[0]).shard(devs, axis=0)
      b = Tensor.empty((N, N), dtype=dtype, device=devs[0]).shard(devs, axis=None)
      (a @ b).realize()
      total_mem[dtype] = GlobalCounters.global_mem
    self.assertEqual(total_mem[dtypes.half], total_mem[dtypes.float] // 2)

  def test_matmul_half(self): self._test_matmul_half(dev_count=2)
  def test_matmul_half_alt(self): self._test_matmul_half(dev_count=4)

if __name__ == '__main__':
  unittest.main()
