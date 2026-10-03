import unittest
from tinygrad import Device
from tinygrad.uop.ops import Ops, UOp
from tinygrad.dtype import dtypes

@unittest.skipUnless(Device.DEFAULT == "METAL", "Metal device required to run")
class TestMetalGraph(unittest.TestCase):
  def setUp(self):
    from tinygrad.runtime.graph.metal import MetalGraph
    self.MetalGraph = MetalGraph
    self.dev = Device[Device.DEFAULT]

  def metal_buf(self, offset, bitcast=False):
    size = 4 if bitcast else 1
    buf = UOp.new_buffer(Device.DEFAULT, offset+size, dtypes.uint8)
    if offset: buf = buf[offset:offset+size]
    return buf.bitcast(dtypes.float32) if bitcast else buf

  def supports_uop(self, *bufs):
    return self.MetalGraph.supports_uop([self.dev], UOp(Ops.PROGRAM, src=(UOp.sink(),)).call(*bufs))

  def test_supports_uop_normal_offset(self):
    assert self.supports_uop(self.metal_buf(0), self.metal_buf(100), self.metal_buf(0xFFFFFFFF)) is True

  def test_supports_uop_overflow_offset(self):
    assert self.supports_uop(self.metal_buf(0), self.metal_buf(0x100000000)) is False

  def test_supports_uop_non_view_buf(self):
    assert self.supports_uop(self.metal_buf(0)) is True

  def test_supports_uop_bitcast(self):
    assert self.supports_uop(self.metal_buf(0xFFFFFFFF, bitcast=True)) is True
    assert self.supports_uop(self.metal_buf(0x100000000, bitcast=True)) is False

if __name__ == "__main__":
  unittest.main()
