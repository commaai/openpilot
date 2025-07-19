import unittest
from tinygrad import Device, dtypes, Tensor
from tinygrad.device import Buffer
from tinygrad.helpers import Context
from test.helpers import REAL_DEV

@unittest.skipUnless(hasattr(Device[Device.DEFAULT].allocator, "_offset"), "subbuffer not supported")
class TestSubBuffer(unittest.TestCase):
  def setUp(self):
    self.buf = Buffer(Device.DEFAULT, 10, dtypes.uint8).ensure_allocated()
    self.buf.copyin(memoryview(bytearray(range(10))))
    self.buf_unalloc = Buffer(Device.DEFAULT, 10, dtypes.uint8)

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

  @unittest.skipIf(REAL_DEV not in {"CUDA", "NV", "AMD"}, "only NV, AMD, CUDA")
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

  def test_subbuffer_is_allocated(self):
    buf = self.buf_unalloc
    sub_buf = buf.view(3, dtypes.uint8, offset=4)
    self.assertFalse(buf.is_allocated())
    self.assertFalse(buf.is_initialized())
    self.assertFalse(sub_buf.is_allocated())
    self.assertFalse(sub_buf.is_initialized())

    # base buffer alloc
    buf.allocate()
    self.assertTrue(buf.is_allocated())
    self.assertTrue(buf.is_initialized())
    self.assertTrue(sub_buf.is_allocated())
    self.assertFalse(sub_buf.is_initialized())

    # sub buffer alloc
    sub_buf.allocate()
    self.assertTrue(sub_buf.is_initialized())

    # sub buffer dealloc
    sub_buf.deallocate()
    self.assertTrue(buf.is_allocated())
    self.assertTrue(buf.is_initialized())
    self.assertTrue(sub_buf.is_allocated())
    self.assertFalse(sub_buf.is_initialized())

    # base buffer dealloc
    buf.deallocate()
    self.assertFalse(buf.is_allocated())
    self.assertFalse(buf.is_initialized())
    self.assertFalse(sub_buf.is_allocated())
    self.assertFalse(sub_buf.is_initialized())

    # sub buffer alloc
    sub_buf.ensure_allocated()
    self.assertTrue(buf.is_allocated())
    self.assertTrue(buf.is_initialized())
    self.assertTrue(sub_buf.is_allocated())
    self.assertTrue(sub_buf.is_initialized())

  def test_subbuffer_copy_in_out(self):
    sub_buf = self.buf.view(3, dtypes.uint8, offset=3).ensure_allocated() # [3:6]
    data_out_sub = bytearray([0]*3)
    sub_buf.copyout(memoryview(data_out_sub))
    assert data_out_sub == bytearray(range(3, 6))
    sub_buf.copyin(memoryview(bytearray(range(3))))
    assert sub_buf.as_buffer().tolist() == list(range(3))
    assert self.buf.as_buffer().tolist()[3:6] == list(range(3))
    sub_buf.copyout(memoryview(data_out_sub))
    assert data_out_sub == bytearray(range(3))
    data_out_base = bytearray([0]*10)
    self.buf.copyout(memoryview(data_out_base))
    assert data_out_base[0:3] == bytearray(range(0, 3))
    assert data_out_base[3:6] == data_out_sub
    assert data_out_base[6:10] == bytearray(range(6, 10))

  def test_subbuffer_copy_in_out_view_of_view(self):
    view1 = self.buf.view(7, dtypes.uint8, offset=2).ensure_allocated() # [2:9]
    view2 = view1.view(3, dtypes.uint8, offset=2).ensure_allocated()   # [4:7]
    self.assertTrue(view1.is_allocated())
    self.assertTrue(view2.is_allocated())

    data_in = bytearray([7, 8, 9])
    view2.copyin(memoryview(data_in))
    data_out_v2 = bytearray([0]*3)
    view2.copyout(memoryview(data_out_v2))
    assert data_in == data_out_v2

    expected_base_data = memoryview(bytearray(range(10)))
    expected_base_data[4:7] = data_in

    data_out_base = bytearray([0]*10)
    self.buf.copyout(memoryview(data_out_base))
    assert expected_base_data == data_out_base

  def test_subbuffer_alloc(self):
    sub_buf = self.buf.view(4, dtypes.int8, offset=3)
    sub_buf.allocate()
    sub_buf.copyin(memoryview(bytearray(range(10, 14))))
    assert self.buf.as_buffer().tolist()[3:7] == sub_buf.as_buffer().tolist()

    sub_buf = self.buf_unalloc.view(4, dtypes.int8, offset=3)
    sub_buf.allocate()
    sub_buf.copyin(memoryview(bytearray(range(10, 14))))
    assert self.buf_unalloc.as_buffer().tolist()[3:7] == sub_buf.as_buffer().tolist()

  def test_subbuffer_dealloc(self):
    sub_buf = self.buf.view(4, dtypes.int8, offset=3).ensure_allocated()
    sub_buf.deallocate()
    assert self.buf.as_buffer().tolist() == list(range(10))

  def test_subbuffer_double_dealloc(self):
    sub_buf = self.buf.view(3, dtypes.uint8, offset=4).ensure_allocated()
    self.buf.deallocate()
    with self.assertRaises(AssertionError):
      self.buf.deallocate()
    sub_buf.deallocate()
    with self.assertRaises(AssertionError):
      sub_buf.deallocate()

  def test_subbuffer_uaf(self):
    sub_buf = self.buf.view(4, dtypes.int8, offset=3).ensure_allocated()
    assert self.buf.as_buffer().tolist(), list(range(10))
    sub_buf.deallocate()
    with self.assertRaises(AssertionError):
      sub_buf.as_buffer().tolist()
    assert self.buf.as_buffer().tolist(), list(range(10))

    sub_buf = self.buf.view(4, dtypes.int8, offset=3).ensure_allocated()
    assert sub_buf.as_buffer().tolist(), list(range(3, 7))
    self.buf.deallocate()
    with self.assertRaises(AssertionError):
      sub_buf.as_buffer().tolist()

if __name__ == '__main__':
  unittest.main()
