import unittest
from tinygrad.runtime.support.am.amdev import AMMemoryManager

class FakeGMC:
  def flush_tlb(self, *args, **kwargs): pass

class FakePCIRegion:
  def __init__(self): self.base_addr = 0xc12300000000

class FakePCIDev:
  def __init__(self): self.regions = [FakePCIRegion()]

class FakeAM:
  def __init__(self):
    self.pcidev = FakePCIDev()
    self.vram = memoryview(bytearray(4 << 30))
    self.gmc = FakeGMC()
    self.mm = AMMemoryManager(self, vram_size=4 << 30)

#  * PTE format:
#  * 63:59 reserved
#  * 58:57 reserved
#  * 56 F
#  * 55 L
#  * 54 reserved
#  * 53:52 SW
#  * 51 T
#  * 50:48 mtype
#  * 47:12 4k physical page base address
#  * 11:7 fragment
#  * 6 write
#  * 5 read
#  * 4 exe
#  * 3 Z
#  * 2 snooped
#  * 1 system
#  * 0 valid
def helper_read_entry_components(entry_val):
  return {"paddr": entry_val & 0x0000FFFFFFFFF000, "fragment":(entry_val >> 7) & 0x1f, "valid": entry_val & 0x1,
          "read": (entry_val >> 5) & 0x1, "write": (entry_val >> 6) & 0x1, "exec": (entry_val >> 4) & 0x1,
          "mtype": (entry_val >> 48) & 0x7, "T": (entry_val >> 51) & 0x1, "L": (entry_val >> 55) & 0x1, "F": (entry_val >> 56) & 0x1}

class TestAMPageTable(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.d = [FakeAM() for _ in range(2)]

  def test_page_table_walkers(self):
    mm = self.d[0].mm

    for va,sz in [(0x10000, 0x3000), (0x11000, 0x300000), (0x10000, 0x2000), (0x11000, 0x5000),
                  (0x2000000, 0x2000), (0x4000000, 0x4000000), (0x38000, 0x303000), (0x8000, 0x1000)]:
      exteranl_va = va + AMMemoryManager.va_allocator.base
      mm.map_range(vaddr=exteranl_va, size=sz, paddr=va)

      results = list(mm.page_table_walker(mm.root_page_table, vaddr=va, size=sz))

      total_covered = 0
      for tup in results:
        _vaddr, _offset, _pte_idx, _n_ptes, _pte_covers, _pt = tup
        total_covered += _n_ptes * _pte_covers

      assert total_covered == sz, f"Expected total coverage {total_covered} to be {sz}"

      for tup in results:
        _vaddr, _offset, pte_idx, n_ptes, pte_covers, pt = tup
        for i in range(n_ptes):
          pte = helper_read_entry_components(pt.get_entry(pte_idx + i))
          assert pte['paddr'] == va + _offset + i * pte_covers, f"Expected paddr {pte['paddr']:#x} to be {va + _offset + i * pte_covers:#x}"
          assert pte['valid'] == 1

      mm.unmap_range(va, sz, free_paddrs=False)

      for tup in results:
        _vaddr, _offset, pte_idx, n_ptes, pte_covers, pt = tup
        for i in range(n_ptes):
          pte = helper_read_entry_components(pt.get_entry(pte_idx + i))
          assert pte['paddr'] == 0
          assert pte['valid'] == 0

  def test_double_map(self):
    mm0 = self.d[0].mm

    for va,sz in [(0x10000, 0x3000), (0x1000000, 0x1000000), (0x12000, 0x4000)]:
      exteranl_va = va + AMMemoryManager.va_allocator.base
      mm0.map_range(vaddr=exteranl_va, size=sz, paddr=va)

      with self.assertRaises(AssertionError):
        mm0.map_range(vaddr=exteranl_va, size=0x1000, paddr=va)

      with self.assertRaises(AssertionError):
        mm0.map_range(vaddr=exteranl_va, size=0x100000, paddr=va)

      with self.assertRaises(AssertionError):
        mm0.map_range(vaddr=exteranl_va + 0x1000, size=0x1000, paddr=va)

      with self.assertRaises(AssertionError):
        mm0.map_range(vaddr=exteranl_va + 0x2000, size=0x100000, paddr=va)

      mm0.unmap_range(vaddr=exteranl_va, size=sz, free_paddrs=False)

      # Finally can map and check paddrs
      mm0.map_range(vaddr=exteranl_va + 0x2000, size=0x100000, paddr=0xdead0000)
      for tup in mm0.page_table_walker(mm0.root_page_table, vaddr=va + 0x2000, size=0x100000):
        _vaddr, _offset, pte_idx, n_ptes, pte_covers, pt = tup
        for i in range(n_ptes):
          pte = helper_read_entry_components(pt.get_entry(pte_idx + i))
          assert pte['paddr'] == 0xdead0000 + _offset + i * pte_covers, f"paddr {pte['paddr']:#x} not {0xdead0000 + _offset + i * pte_covers:#x}"
          assert pte['valid'] == 1

      mm0.unmap_range(vaddr=exteranl_va + 0x2000, size=0x100000, free_paddrs=False)

  def test_map_from(self):
    mm0 = self.d[0].mm
    mm1 = self.d[1].mm

    for va,sz in [(0x10000, 0x3000), (0x11000, 0x300000), (0x10000, 0x2000), (0x11000, 0x5000),
                  (0x2000000, 0x2000), (0x4000000, 0x4000000), (0x38000, 0x303000), (0x8000, 0x1000)]:
      exteranl_va = va + AMMemoryManager.va_allocator.base
      mm0.map_range(vaddr=exteranl_va, size=sz, paddr=va)
      mm1.map_range(vaddr=exteranl_va, size=sz, paddr=va)

      with self.assertRaises(AssertionError):
        mm0.map_from(vaddr=exteranl_va, size=sz, from_adev=mm0.adev) # self mapping -- bad

      with self.assertRaises(AssertionError):
        mm0.map_from(vaddr=exteranl_va, size=sz, from_adev=mm1.adev) # mapping from mm1 to same addrs -- bad

      mm0.unmap_range(vaddr=exteranl_va, size=sz, free_paddrs=False) # unmap from mm0
      mm0.map_from(vaddr=exteranl_va, size=sz, from_adev=mm1.adev) # mapping from mm1 to same addrs -- ok

      d1_pci_base = self.d[1].pcidev.regions[0].base_addr
      for tup in mm0.page_table_walker(mm0.root_page_table, vaddr=va, size=sz):
        _vaddr, _offset, pte_idx, n_ptes, pte_covers, pt = tup
        for i in range(n_ptes):
          pte = helper_read_entry_components(pt.get_entry(pte_idx + i))
          assert pte['paddr'] == d1_pci_base + va + _offset + i * pte_covers, f"paddr {pte['paddr']:#x} not {d1_pci_base+va+_offset+i*pte_covers:#x}"
          assert pte['valid'] == 1

      mm0.unmap_range(vaddr=exteranl_va, size=sz, free_paddrs=False)
      mm1.unmap_range(vaddr=exteranl_va, size=sz, free_paddrs=False)

  def test_try_bad_unmap(self):
    mm0 = self.d[0].mm

    with self.assertRaises(AssertionError):
      mm0.unmap_range(0x10000, 0x3000, free_paddrs=False)

    mm0.map_range(0x10000, 0x3000, 0x10000)
    mm0.unmap_range(0x10000, 0x3000, free_paddrs=False)

    with self.assertRaises(AssertionError):
      mm0.unmap_range(0x10000, 0x3000, free_paddrs=False)

    mm0.map_range(0x10000, 0x3000, 0x10000)
    mm0.unmap_range(0x10000, 0x3000, free_paddrs=False)

    with self.assertRaises(AssertionError):
      mm0.unmap_range(0x10000, 0x3000, free_paddrs=False)

if __name__ == "__main__":
  unittest.main()
