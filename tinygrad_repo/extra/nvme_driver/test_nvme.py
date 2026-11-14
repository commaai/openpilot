#!/usr/bin/env python3
# NVME=0 GPU=AMD python3 extra/nvme_driver/test_nvme.py: reads of the last 256 MiB of a drive (which must hold data) into gpu memory. never writes
import unittest
import numpy as np
from tinygrad import Tensor, Device, TinyJit, dtypes
from tinygrad.helpers import getenv

NVME, GPU, N = f"NVME:{getenv('NVME', 0)}", getenv("GPU", "AMD"), 256 << 20

class TestNVME(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.region = Tensor.empty(size:=Device[NVME].nvme.size, dtype=dtypes.uint8, device=NVME)[size - N:]
    cls.ref = cls.region.to(GPU).numpy()
    assert np.count_nonzero(np.bincount(cls.ref)) > 64, "the last 256 MiB of the drive hold no data to compare"
  def check(self, reads:list[tuple[int, int]]):
    Tensor.realize(*(ts:=[self.region[off:off + n].to(GPU) for off, n in reads]))
    for t, (off, n) in zip(ts, reads): np.testing.assert_equal(t.numpy(), self.ref[off:off + n], err_msg=f"{n} bytes at {off}")

  def test_reads(self):
    for read in [(0, 4096), (4096, 8192), (0, 3 * 4096), (0, 1 << 20), (12288, (1 << 20) + 4096), (4096, N - 8192)]: self.check([read])
  def test_reads_one_submit(self): self.check([(0, 1 << 20), (8 << 20, 3 << 20)])
  def test_jit(self):
    @TinyJit
    def f(t:Tensor) -> Tensor: return (t.to(GPU) + 1).realize()
    for _ in range(5): np.testing.assert_equal(f(self.region[1 << 20:3 << 20]).numpy(), self.ref[1 << 20:3 << 20] + 1)

if __name__ == "__main__": unittest.main()
