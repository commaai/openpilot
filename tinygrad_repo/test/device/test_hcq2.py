import unittest, numpy as np
from unittest.mock import patch
from tinygrad import Device, Tensor
from tinygrad.device import Buffer
from tinygrad.dtype import dtypes
from tinygrad.helpers import HCQ2
from tinygrad.runtime.support.hcq2 import HCQ_DEVS, all_devices_in, hcq_compile_cache, link_linear_cache

@unittest.skipUnless(HCQ2 and all_devices_in(Device.DEFAULT, HCQ_DEVS), "hcq2 device required")
class TestHCQ2(unittest.TestCase):
  def test_copy_without_copy_queue(self):
    with patch.object(Device[Device.DEFAULT], "has_copy_queue", False):
      np.testing.assert_equal(Tensor(np.arange(61, dtype=np.float32)).to(Device.DEFAULT).contiguous().realize().numpy(), np.arange(61))

  @unittest.skipIf(Device.DEFAULT == "CPU", "ping-pong needs a non-CPU hcq2 device")
  def test_cpu_device_ping_pong(self):
    # CPU submits run inline, so alternating dependencies must be submitted in schedule order to avoid blocking the host submitter.
    x = Tensor.ones(16, device="CPU").contiguous().realize()
    a = (x + 1).contiguous()
    b = (a.to(Device.DEFAULT).contiguous() + 1).contiguous()
    c = (b.to("CPU").contiguous() + 1).contiguous()
    out = (c.to(Device.DEFAULT).contiguous() + 1).contiguous().realize()
    np.testing.assert_equal(out.numpy(), np.full(16, 5))

  @unittest.skipIf(Device.DEFAULT == "CPU", "staged copies need a non-CPU hcq2 device")
  def test_staged_copy_slot_reuse(self):
    # chunks of a staged copy rotate through the staging buffer slots, many rotations must stay bit-exact in both directions
    import tinygrad.runtime.support.hcq2 as hcq2
    buf = Buffer("CPU", 1 << 20, dtypes.uint8, preallocate=True)
    data = np.random.default_rng(42).integers(0, 256, (5 << 20) + 123, dtype=np.uint8)
    with patch.object(hcq2, "STAGING_SIZE", 1 << 20), patch.object(hcq2, "STAGING_SLOTS", 4), patch.object(hcq2, "_staging", lambda: buf):
      np.testing.assert_equal(Tensor(data).to(Device.DEFAULT).realize().numpy(), data)

  def test_overlapping_device_tuples(self):
    # an op on a wide device tuple followed by an op on an overlapping smaller tuple used to MMU-fault the smaller one
    d4, d2 = tuple(f"{Device.DEFAULT}:{i}" for i in range(4)), tuple(f"{Device.DEFAULT}:{i}" for i in range(2))
    ref = Tensor.arange(16).contiguous().realize()
    Tensor(ref.uop.copy_to_device(d4)).realize()
    out = Tensor.ones(8).shard(d2, axis=0).contiguous().realize()
    np.testing.assert_equal(out.numpy(), np.ones(8))

  def relowers(self, t:Tensor) -> tuple[int, int]:
    # a miss in either cache relowers the whole submit, a hit costs ~0.1ms
    before = (len(hcq_compile_cache), len(link_linear_cache))
    t.realize()
    return (len(hcq_compile_cache) - before[0], len(link_linear_cache) - before[1])

  def test_relower_only_on_new_kernel(self):
    a, b = (Tensor.empty(64, 64).contiguous().realize() for _ in range(2))
    self.relowers(a.sin())
    self.assertEqual(self.relowers(a.sin()), (0, 0))  # nothing changed
    self.assertEqual(self.relowers(b.sin()), (0, 0))  # new buffers, patched in at link time
    self.assertEqual(self.relowers(a.cos()), (1, 1))  # new kernel, though only the code address moved
    self.assertEqual(self.relowers(a.cos()), (0, 0))
    self.assertEqual(self.relowers(Tensor.empty(32, 32).contiguous().realize().sin()), (1, 1))  # new shape

  def test_dtype_sweep_relowers_every_dtype(self):
    # test_dtype sweeps dtypes at one shape, so nearly every kernel is new: this is where hcq2 ci time goes
    src = Tensor.empty(64, 64).contiguous().realize()
    dts = (dtypes.int8, dtypes.uint8, dtypes.int16, dtypes.uint16, dtypes.int32)
    self.assertEqual([self.relowers(src.cast(dt).contiguous())[0] for dt in dts], [1] * len(dts))

if __name__ == "__main__":
  unittest.main()
