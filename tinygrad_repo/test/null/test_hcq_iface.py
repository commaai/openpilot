import unittest, array, time
from tinygrad.helpers import mv_address
from tinygrad.runtime.support.hcq import MMIOInterface
from tinygrad.runtime.support.usb import USBMMIOInterface
from test.mockgpu.usb import MockUSB

class TestHCQIface(unittest.TestCase):
  def setUp(self):
    self.size = 4 << 10
    self.buffer = bytearray(self.size)
    self.mv = memoryview(self.buffer).cast('I')
    self.mmio = MMIOInterface(mv_address(self.mv), self.size, fmt='I')

  def test_getitem_setitem(self):
    self.mmio[1] = 0xdeadbeef
    self.assertEqual(self.mmio[1], 0xdeadbeef)
    values = array.array('I', [10, 20, 30, 40])
    self.mmio[2:6] = values
    read_slice = self.mmio[2:6]
    # self.assertIsInstance(read_slice, array.array)
    self.assertEqual(read_slice, values.tolist())
    self.assertEqual(self.mv[2:6].tolist(), values.tolist())

  def test_view(self):
    full = self.mmio.view()
    self.assertEqual(len(full), len(self.mmio))
    self.mmio[0] = 0x12345678
    self.assertEqual(full[0], 0x12345678)

    # offset-only view
    self.mmio[1] = 0xdeadbeef
    off = self.mmio.view(offset=4)
    self.assertEqual(off[0], 0xdeadbeef)

    # offset + size view: write into sub-view and confirm underlying buffer
    values = array.array('I', [11, 22, 33])
    sub = self.mmio.view(offset=8, size=12)
    sub[:] = values
    self.assertEqual(sub[:], values.tolist())
    self.assertEqual(self.mv[2:5].tolist(), values.tolist())

  def test_speed(self):
    start = time.perf_counter()
    for i in range(10000):
      self.mmio[3:100] = array.array('I', [i] * 97)
      _ = self.mmio[3:100]
    end = time.perf_counter()

    mvstart = time.perf_counter()
    for i in range(10000):
      self.mv[3:100] = array.array('I', [i] * 97)
      _ = self.mv[3:100].tolist()
    mvend = time.perf_counter()
    print(f"speed: hcq {end - start:.6f}s vs plain mv {mvend - mvstart:.6f}s")

class TestUSBMMIOInterface(unittest.TestCase):
  def setUp(self):
    self.size = 256
    self.buffer = bytearray(self.size)
    self.usb = MockUSB(self.buffer)
    self.mmio = USBMMIOInterface(self.usb, 0, self.size, fmt='B', pcimem=False)

  def test_getitem_setitem_byte(self):
    self.mmio[1] = 0xAB
    self.assertEqual(self.mmio[1], 0xAB)
    self.assertEqual(self.usb.mem[1], 0xAB)

  def test_slice_getitem_setitem(self):
    values = [1, 2, 3, 4]
    self.mmio[10:14] = values
    raw = self.mmio[10:14]
    self.assertIsInstance(raw, bytes)
    self.assertEqual(list(raw), values)
    self.assertEqual(list(self.usb.mem[10:14]), values)

  def test_view(self):
    self.mmio[0] = 5
    view = self.mmio.view(offset=1, size=3)
    self.assertEqual(view[0], self.usb.mem[1])
    view[:] = [7, 8, 9]
    self.assertEqual(list(self.usb.mem[1:4]), [7, 8, 9])
    full_view = self.mmio.view()
    self.assertEqual(len(full_view), len(self.mmio))
    self.mmio[2] = 0xFE
    self.assertEqual(full_view[2], 0xFE)

  def test_pcimem_byte(self):
    usb2 = MockUSB(bytearray(self.size))
    mmio_pci = USBMMIOInterface(usb2, 0, self.size, fmt='B', pcimem=True)
    mmio_pci[3] = 0x11
    self.assertEqual(mmio_pci[3], 0x11)
    self.assertEqual(usb2.mem[3], 0x11)

  def test_pcimem_slice(self):
    usb3 = MockUSB(bytearray(self.size))
    mmio_pci = USBMMIOInterface(usb3, 0, self.size, fmt='B', pcimem=True)
    values = [2, 3, 4]
    mmio_pci[4:7] = values
    raw = mmio_pci[4:7]
    self.assertIsInstance(raw, bytes)
    self.assertEqual(list(raw), values)
    self.assertEqual([mmio_pci[i] for i in range(4, 7)], values)

if __name__ == "__main__":
  unittest.main()
