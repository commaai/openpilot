import unittest
from unittest.mock import MagicMock
from tinygrad import Device
from tinygrad.uop.ops import Ops
from tinygrad.dtype import dtypes

@unittest.skipUnless(Device.DEFAULT == "METAL", "Metal device required to run")
class TestMetalGraph(unittest.TestCase):
  def setUp(self):
    from tinygrad.runtime.graph.metal import MetalGraph
    self.MetalGraph = MetalGraph
    self.dev = Device[Device.DEFAULT]

  def metal_buf(self, offset):
    buf = MagicMock()
    if offset > 0:
      buf.op = Ops.BUFFER_VIEW
      buf.arg = (None, offset)
      buf.dtype = dtypes.uint8
    else:
      buf.op = Ops.BUFFER
    buf.device = Device.DEFAULT
    return buf

  def call(self, *bufs):
    c = MagicMock()
    c.src = (MagicMock(op=Ops.PROGRAM),) + tuple(bufs)
    return c

  def test_supports_uop_normal_offset(self):
    assert self.MetalGraph.supports_uop([self.dev], self.call(self.metal_buf(0), self.metal_buf(100), self.metal_buf(0xFFFFFFFF))) is True

  def test_supports_uop_overflow_offset(self):
    assert self.MetalGraph.supports_uop([self.dev], self.call(self.metal_buf(0), self.metal_buf(0x100000000))) is False

  def test_supports_uop_nonmetal_buf(self):
    # non-BUFFER_VIEW ops should not be checked for offset
    buf = MagicMock()
    buf.op = Ops.BUFFER
    buf.device = Device.DEFAULT
    self.MetalGraph.supports_uop([self.dev], self.call(buf))

if __name__ == "__main__":
  unittest.main()
