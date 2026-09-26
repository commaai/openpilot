import unittest
from tinygrad.device import Buffer
from tinygrad.dtype import dtypes
from tinygrad.helpers import Context

class TestBuffer(unittest.TestCase):
  def test_host_view(self):
    b = Buffer("CPU", 4, dtypes.uint32)
    v = b.view(2, dtypes.uint16, 4)
    host = v.host
    host.view(fmt='H')[0] = 0x1234
    self.assertEqual(b.host.view(fmt='H')[2], 0x1234)
    self.assertEqual(v._buf, b._buf + 4)
    self.assertIs(v.host, host)
    self.assertIs(v.meta, b.meta)

  def test_mapping(self):
    b = Buffer("CPU", 8, dtypes.uint8, initial_value=b"abcdefgh")
    self.assertEqual(b.get_buf("PYTHON"), b._buf)
    v = b.view(4, dtypes.uint8, 2)
    mapped = v.get_storage("PYTHON")
    self.assertEqual(mapped.buf, b._buf + 2)
    self.assertEqual(bytes(mapped.host.mv), b"cdef")
    self.assertIs(mapped.host, v.host)
    self.assertIsNone(mapped.meta)
    self.assertIs(v.get_storage("PYTHON"), mapped)

  def test_view_reallocation(self):
    b = Buffer("CPU", 8, dtypes.uint8)
    v = b.view(4, dtypes.uint8, 2)
    old = v.get_storage("PYTHON")
    b.deallocate()
    b.allocate()
    self.assertFalse(v.is_allocated())
    v.host[:] = b"test"
    self.assertIsNot(v.get_storage("PYTHON"), old)
    self.assertEqual(bytes(v.get_storage("PYTHON").host.mv), b"test")

  def test_cache_owned_storage_only(self):
    for opaque in (None, memoryview(bytearray(8))):
      with self.subTest(imported=opaque is not None), Context(LRU=1):
        b = Buffer("PYTHON", 8, dtypes.uint8, opaque=opaque)
        buf = b._buf
        b.deallocate()
        self.assertEqual(b._buf is buf, opaque is None)

if __name__ == "__main__": unittest.main()
