import unittest
from tinygrad import Device, dtypes, Tensor
from tinygrad.device import Buffer
from tinygrad.helpers import Context

@unittest.skipUnless(hasattr(Device[Device.DEFAULT].allocator, "_offset"), "subbuffer not supported")
class TestSubBuffer(unittest.TestCase):
  def setUp(self):
    self.buf = Buffer(Device.DEFAULT, 10, dtypes.uint8).ensure_allocated()
    self.buf.copyin(memoryview(bytearray(range(10))))

  def test_subbuffer(self):
    vbuf = self.buf.view(2, dtypes.uint8, offset=3).ensure_allocated()
    tst = vbuf.as_buffer().tolist()
    assert tst == [3, 4]

  def test_subbuffer_cast(self):
    # NOTE: bitcast depends on endianness
    vbuf = self.buf.view(2, dtypes.uint16, offset=3).ensure_allocated()
    tst = vbuf.as_buffer().cast("H").tolist()
    assert tst == [3|(4<<8), 5|(6<<8)]

  def test_subbuffer_double(self):
    vbuf = self.buf.view(4, dtypes.uint8, offset=3).ensure_allocated()
    vvbuf = vbuf.view(2, dtypes.uint8, offset=1).ensure_allocated()
    tst = vvbuf.as_buffer().tolist()
    assert tst == [4, 5]

  def test_subbuffer_len(self):
    vbuf = self.buf.view(5, dtypes.uint8, 2).ensure_allocated()
    mv = vbuf.as_buffer()
    assert len(mv) == 5
    mv = vbuf.as_buffer(allow_zero_copy=True)
    assert len(mv) == 5

  def test_subbuffer_used(self):
    t = Tensor.arange(0, 10, dtype=dtypes.uint8).realize()
    vt = t[2:4].realize()
    out = (vt + 100).tolist()
    assert out == [102, 103]

  @unittest.skipIf(Device.DEFAULT not in {"CUDA", "NV", "AMD"}, "only NV, AMD, CUDA")
  def test_subbuffer_transfer(self):
    t = Tensor.arange(0, 10, dtype=dtypes.uint8).realize()
    vt = t[2:5].contiguous().realize()
    out = vt.to(f"{Device.DEFAULT}:1").realize().tolist()
    assert out == [2, 3, 4]

  def test_subbuffer_deallocate(self):
    with Context(LRU=0):
      vbuf = self.buf.view(2, dtypes.uint8, offset=3).ensure_allocated()
      self.buf.deallocate()
      vbuf.deallocate()

      # Allocate a fake one on the same place
      _ = Buffer(Device.DEFAULT, 10, dtypes.uint8).ensure_allocated()

      self.buf.ensure_allocated()
      self.buf.copyin(memoryview(bytearray(range(10, 20))))

      vbuf.ensure_allocated()

      tst = vbuf.as_buffer().tolist()
      assert tst == [13, 14]

if __name__ == '__main__':
  unittest.main()
