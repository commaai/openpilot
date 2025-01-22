import unittest
from tinygrad.runtime.support.allocator import TLSFAllocator

class TestTLSFAllocator(unittest.TestCase):
  def setUp(self):
    self.allocator = TLSFAllocator(1024, block_size=16)

  def test_basic_alloc_free(self):
    addr1 = self.allocator.alloc(32)
    self.assertEqual(addr1, 0)

    addr2 = self.allocator.alloc(64)
    self.assertEqual(addr2, 32)

    self.allocator.free(addr1)
    addr3 = self.allocator.alloc(32)
    self.assertEqual(addr3, 0)

  def test_block_size_alignment(self):
    addr1 = self.allocator.alloc(20)
    addr2 = self.allocator.alloc(35)

    self.assertEqual(addr1 % 16, 0)
    self.assertEqual(addr2 % 16, 0)

  def test_merge_blocks(self):
    addr1 = self.allocator.alloc(32)
    addr2 = self.allocator.alloc(32)
    self.allocator.alloc(32)

    self.allocator.free(addr1)
    self.allocator.free(addr2)
    addr4 = self.allocator.alloc(64)
    self.assertEqual(addr4, addr1)

  def test_split_blocks(self):
    addr1 = self.allocator.alloc(128)
    self.allocator.free(addr1)

    addr2 = self.allocator.alloc(32)
    self.assertEqual(addr2, addr1)

    addr3 = self.allocator.alloc(32)
    self.assertEqual(addr3, addr1 + 32)

  def test_out_of_memory(self):
    with self.assertRaises(MemoryError):
      self.allocator.alloc(2048)

  def test_fragmentation_handling(self):
    addrs = []
    for _ in range(5):
      addrs.append(self.allocator.alloc(32))

    # Free alternate blocks
    for i in range(0, len(addrs), 2):
      self.allocator.free(addrs[i])

  def test_custom_start_address(self):
    allocator = TLSFAllocator(1024, start_addr=1000)
    addr1 = allocator.alloc(32)
    self.assertEqual(addr1, 1000)

    addr2 = allocator.alloc(64)
    self.assertEqual(addr2, 1032)

  def test_block_tracking(self):
    addr1 = self.allocator.alloc(32)
    addr2 = self.allocator.alloc(64)

    self.assertTrue(addr1 in [addr - self.allocator.start_addr for addr in self.allocator.blocks])
    self.assertTrue(addr2 in [addr - self.allocator.start_addr for addr in self.allocator.blocks])

    self.allocator.free(addr1)
    self.assertTrue(addr1 in [addr - self.allocator.start_addr for addr in self.allocator.blocks])

if __name__ == '__main__':
  unittest.main()
