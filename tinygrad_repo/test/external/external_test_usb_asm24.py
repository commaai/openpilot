import unittest
from tinygrad.helpers import Timing, getenv
from tinygrad import Tensor, Device, TinyJit
from tinygrad.runtime.support.usb import HALF, CHUNK, SLOT
import numpy as np

class USBTestCase(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.sz = getenv("SIZE", 2000000)
    cls.dev = Device["AMD"]
    if not cls.dev.is_usb: raise unittest.SkipTest("only test this on USB devices")
    cls.rng = np.random.default_rng(0)

  def roundtrip(self, a:np.ndarray): # a copy in, a kernel, a copy out: the queue must order them
    np.testing.assert_array_equal(a, Tensor(a, device="NPY").to(Device.DEFAULT).numpy())
    np.testing.assert_array_equal(a + 1, (Tensor(a, device="NPY").to(Device.DEFAULT) + 1).numpy())

class TestDevCopySpeeds(USBTestCase):
  def testCopyCPUtoDefault(self):
    for _ in range(10):
      t = Tensor.ones(self.sz, device="CPU", dtype='uchar').contiguous().realize()
      with Timing(f"copyin of {t.nbytes()/1e6:.2f} MB:  ", on_exit=lambda ns: f" @ {t.nbytes()/ns * 1e3:.2f} MB/s"): # noqa: F821
        t.to(Device.DEFAULT).realize()
        Device[Device.DEFAULT].synchronize()
      del t

  def testCopyDefaulttoCPU(self):
    t = Tensor.ones(self.sz, dtype='uchar').contiguous().realize()
    for _ in range(10):
      with Timing(f"copyout of {t.nbytes()/1e6:.2f} MB:  ", on_exit=lambda ns: f" @ {t.nbytes()/ns * 1e3:.2f} MB/s"):
        t.to('CPU').realize()

class TestUSBIntegrity(USBTestCase):
  def testValidateCopies(self):
    t = Tensor.randn(self.sz, device="CPU", dtype='uchar').contiguous().realize()
    x = t.to(Device.DEFAULT).realize()
    Device[Device.DEFAULT].synchronize()
    y = x.to('CPU').realize()
    np.testing.assert_equal(t.numpy(), y.numpy())

  def testBoundaries(self): # around the slot, the chunk and the read window
    for size in (1, 3, 508, 509, SLOT - 513, SLOT - 512, SLOT - 511, CHUNK - 1, CHUNK, CHUNK + 1, 2 * CHUNK - 1, 2 * CHUNK, 2 * CHUNK + 31, HALF,
                 2 * HALF, 1 << 20):
      with self.subTest(size=size): self.roundtrip(self.rng.integers(0, 256, size, dtype=np.uint8))

  def testManyCopiesInABatch(self):
    for n in (2, 7, 64, 300): # 300 chunks: the fence byte wraps
      with self.subTest(n=n):
        arrs = [self.rng.integers(0, 256, int(s), dtype=np.uint8) for s in self.rng.integers(1, 5000, n)]
        ts = [Tensor(a, device="NPY").to(Device.DEFAULT) for a in arrs]
        Tensor.realize(*ts)
        for t, a in zip(ts, arrs): np.testing.assert_array_equal(a, t.numpy())

  def testMixedBatch(self): # copies out and in, in one batch: runs of both directions
    arrs = [self.rng.integers(0, 256, s, dtype=np.uint8) for s in (5, CHUNK + 7, 9, 2 * CHUNK + 3, 11)]
    ts = [Tensor(a, device="NPY").to(Device.DEFAULT).realize() for a in arrs]
    more = [self.rng.integers(0, 256, s, dtype=np.uint8) for s in (5, CHUNK + 7, 9, 2 * CHUNK + 3, 11)]
    outs = [t.to("NPY") for t in ts] + [Tensor(a, device="NPY").to(Device.DEFAULT) for a in more]
    Tensor.realize(*outs)
    for o, a in zip(outs, arrs + more): np.testing.assert_array_equal(a, o.numpy())

  def testRepeatedBatches(self): # a batch numbers its chunks from 0: the same batch again must not see what the last one left behind
    a = self.rng.integers(0, 256, 2 * CHUNK + 31, dtype=np.uint8)
    for _ in range(5): self.roundtrip(a)
    @TinyJit
    def step(x:Tensor) -> Tensor: return (x + 1).realize()
    src = Tensor(a, device="NPY")
    for i in range(5):
      x = src.to(Device.DEFAULT)
      np.testing.assert_array_equal(a + 1, step(x).numpy())

  def testStaleSentinel(self): # payloads full of the tags the queue waits for, in both directions, before and around the real chunks
    tags = np.array([0x51000000 | k for k in range(8)], dtype=np.uint32)
    for tag in tags: # every dword of every chunk is the tag of some chunk of the copy
      with self.subTest(payload=hex(tag)):
        a = np.full((2 * CHUNK + 31) // 4, tag, dtype=np.uint32).view(np.uint8)
        self.roundtrip(a)
    with self.subTest(case="copyout residue"): # a read fills the sram with tags, then small chunks land in both halves
      a = np.tile(tags, 2 * CHUNK // 32).view(np.uint8)
      np.testing.assert_array_equal(a, (Tensor(a, device="NPY").to(Device.DEFAULT) * 1).numpy())
      for size in (31, CHUNK + 31, 2 * CHUNK + 31): self.roundtrip(np.tile(tags, size // 32 + 1).view(np.uint8)[:size])

  def testRingWrap(self): # 64MB of chunks: the sdma ring (1MB on usb) wraps within the copy
    a = self.rng.integers(0, 256, 64 << 20, dtype=np.uint8)
    t = Tensor(a, device="NPY").to(Device.DEFAULT).realize()
    np.testing.assert_array_equal(a, t.numpy())

if __name__ == "__main__":
  unittest.main()
