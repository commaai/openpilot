import unittest, time
from tinygrad.runtime.support.usb import ASM24Controller
from tinygrad.helpers import Timing
from tinygrad import Tensor, Device
import numpy as np

class TestASMController(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.ctrl = ASM24Controller()

  def test_write_and_read(self):
    base = 0xF000
    data = b"hello!"
    self.ctrl.write(base, data)
    out = self.ctrl.read(base, len(data))
    self.assertEqual(out, data)

  def test_scsi_write_and_read_from_f000(self):
    payload = bytes([0x5B]) * 4096
    self.ctrl.scsi_write(payload, lba=0)
    back = self.ctrl.read(0xF000, len(payload))
    self.assertEqual(back, payload)

  def test_scsi_write_speed_4k(self):
    payload = bytes([0x5A]) * 4096
    start = time.perf_counter()
    self.ctrl.scsi_write(payload, lba=0)
    dur_ms = (time.perf_counter() - start) * 1000
    print(f"scsi_write 4K took {dur_ms:.3f} ms")

  def test_read_speed_4k(self):
    payload = bytes([0xA5]) * 4096
    self.ctrl.write(0xF000, payload)
    start = time.perf_counter()
    out = self.ctrl.read(0xF000, 4096)
    dur_ms = (time.perf_counter() - start) * 1000
    print(f"read 4K took {dur_ms:.3f} ms")
    self.assertEqual(out, payload)

class TestDevCopySpeeds(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.sz = 512
    cls.dev = Device["AMD"]
    if not cls.dev.is_usb(): raise unittest.SkipTest("only test this on USB devices")

  def testCopyCPUtoDefault(self):
    for _ in range(10):
      t = Tensor.ones(self.sz, self.sz, device="CPU").contiguous().realize()
      with Timing(f"copyin of {t.nbytes()/1e6:.2f} MB:  ", on_exit=lambda ns: f" @ {t.nbytes()/ns * 1e3:.2f} MB/s"): # noqa: F821
        t.to(Device.DEFAULT).realize()
        Device[Device.DEFAULT].synchronize()
      del t

  def testCopyDefaulttoCPU(self):
    t = Tensor.ones(self.sz, self.sz).contiguous().realize()
    for _ in range(10):
      with Timing(f"copyout of {t.nbytes()/1e6:.2f} MB:  ", on_exit=lambda ns: f" @ {t.nbytes()/ns * 1e3:.2f} MB/s"):
        t.to('CPU').realize()

  def testValidateCopies(self):
    t = Tensor.randn(self.sz, self.sz, device="CPU").contiguous().realize()
    x = t.to(Device.DEFAULT).realize()
    Device[Device.DEFAULT].synchronize()

    y = x.to('CPU').realize()

    np.testing.assert_equal(t.numpy(), y.numpy())
    del x, y, t

if __name__ == "__main__":
  unittest.main()