import unittest
from tinygrad import Tensor
from tinygrad import Device
from tinygrad.helpers import Timing, CI, OSX
import multiprocessing.shared_memory as shared_memory

N = 4096
class TestCopySpeed(unittest.TestCase):
  @classmethod
  def setUpClass(cls): Device[Device.DEFAULT].synchronize()

  def testCopySHMtoDefault(self):
    s = shared_memory.SharedMemory(name="test_X", create=True, size=N*N*4)
    s.close()
    if CI and not OSX:
      t = Tensor.empty(N, N, device="disk:/dev/shm/test_X").realize()
    else:
      t = Tensor.empty(N, N, device="disk:shm:test_X").realize()
    for _ in range(3):
      with Timing("sync:  ", on_exit=lambda ns: f" @ {t.nbytes()/ns:.2f} GB/s"):
        with Timing("queue: "):
          t.to(Device.DEFAULT).realize()
        Device[Device.DEFAULT].synchronize()
    s.unlink()

  def testCopyCPUtoDefault(self):
    t = Tensor.ones(N, N, device="CPU").contiguous().realize()
    print(f"buffer: {t.nbytes()*1e-9:.2f} GB")
    for _ in range(3):
      with Timing("sync:  ", on_exit=lambda ns: f" @ {t.nbytes()/ns:.2f} GB/s"):
        with Timing("queue: "):
          t.to(Device.DEFAULT).realize()
        Device[Device.DEFAULT].synchronize()

  def testCopyCPUtoDefaultFresh(self):
    print("fresh copy")
    for _ in range(3):
      t = Tensor.ones(N, N, device="CPU").contiguous().realize()
      with Timing("sync:  ", on_exit=lambda ns: f" @ {t.nbytes()/ns:.2f} GB/s"): # noqa: F821
        with Timing("queue: "):
          t.to(Device.DEFAULT).realize()
        Device[Device.DEFAULT].synchronize()
      del t

  def testCopyDefaulttoCPU(self):
    t = Tensor.ones(N, N).contiguous().realize()
    print(f"buffer: {t.nbytes()*1e-9:.2f} GB")
    for _ in range(3):
      with Timing("sync:  ", on_exit=lambda ns: f" @ {t.nbytes()/ns:.2f} GB/s"):
        t.to('CPU').realize()

  @unittest.skipIf(CI, "CI doesn't have 6 GPUs")
  @unittest.skipIf(Device.DEFAULT != "GPU", "only test this on GPU")
  def testCopyCPUto6GPUs(self):
    from tinygrad.runtime.ops_gpu import CLDevice
    if len(CLDevice.device_ids) != 6: raise unittest.SkipTest("computer doesn't have 6 GPUs")
    t = Tensor.ones(N, N, device="CPU").contiguous().realize()
    print(f"buffer: {t.nbytes()*1e-9:.2f} GB")
    for _ in range(3):
      with Timing("sync:  ", on_exit=lambda ns: f" @ {t.nbytes()/ns:.2f} GB/s ({t.nbytes()*6/ns:.2f} GB/s total)"):
        with Timing("queue: "):
          for g in range(6):
            t.to(f"gpu:{g}").realize()
        Device["gpu"].synchronize()

if __name__ == '__main__':
  unittest.main()
