import unittest, numpy as np
from unittest.mock import patch
from tinygrad import Device, Tensor
from tinygrad.helpers import getenv
from tinygrad.runtime.support.hcq2 import HCQ_DEVS, all_devices_in

@unittest.skipUnless(getenv("HCQ2") and all_devices_in(Device.DEFAULT, HCQ_DEVS), "hcq2 device required")
class TestHCQ2(unittest.TestCase):
  def test_copy_without_copy_queue(self):
    with patch.object(Device[Device.DEFAULT], "has_copy_queue", False):
      np.testing.assert_equal(Tensor(np.arange(61, dtype=np.float32)).to(Device.DEFAULT).contiguous().realize().numpy(), np.arange(61))

  def test_overlapping_device_tuples(self):
    # an op on a wide device tuple followed by an op on an overlapping smaller tuple used to MMU-fault the smaller one
    d4, d2 = tuple(f"{Device.DEFAULT}:{i}" for i in range(4)), tuple(f"{Device.DEFAULT}:{i}" for i in range(2))
    ref = Tensor.arange(16).contiguous().realize()
    Tensor(ref.uop.copy_to_device(d4)).realize()
    out = Tensor.ones(8).shard(d2, axis=0).contiguous().realize()
    np.testing.assert_equal(out.numpy(), np.ones(8))

if __name__ == "__main__":
  unittest.main()
