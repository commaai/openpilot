import unittest
from tinygrad.runtime.support.am.amdev import AMMemoryManager, AMPageTableEntry
from tinygrad.runtime.support.am.ip import AM_GMC
from tinygrad.runtime.support.hcq import MMIOInterface
from tinygrad.runtime.support.memory import PageTableTraverseContext
from tinygrad.runtime.autogen.am import am
from tinygrad.helpers import mv_address

class FakeGMC(AM_GMC):
  def __init__(self, adev):
    self.adev = adev
    self.vm_base = 0x0
    self.address_space_mask = (1 << 44) - 1
  def init_hw(self): pass
  def flush_tlb(self, *args, **kwargs): pass

class FakePCIDev:
  def __init__(self): self.regions = [(0xc12300000000, 0xc12400000000, 0x0)]

class FakeAM:
  def __init__(self):
    self.is_booting, self.smi_dev = True, False
    self.pcidev = FakePCIDev()
    self.vram_size = (512 << 20)
    self.vram_mv = memoryview(bytearray(self.vram_size))
    self.vram = MMIOInterface(mv_address(self.vram_mv), self.vram_mv.nbytes)
    self.gmc = FakeGMC(self)
    self.mm = AMMemoryManager(self, self.vram_size, boot_size=(32 << 20), pt_t=AMPageTableEntry, va_shifts=[12, 21, 30, 39], va_bits=48,
      first_lv=am.AMDGPU_VM_PDB2, va_base=AMMemoryManager.va_allocator.base,
      palloc_ranges=[(1 << (i + 12), 0x1000) for i in range(9 * (3 - am.AMDGPU_VM_PDB2), -1, -1)])
    self.is_booting = False
    self.ip_ver = {am.GC_HWIP: (11, 0, 0)}
  def paddr2cpu(self, paddr:int) -> int: return paddr + mv_address(self.vram)
  def paddr2mc(self, paddr:int) -> int: return paddr

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

def helper_va(va:int): return va + AMMemoryManager.va_allocator.base

class TestAMPageTable(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.d = [FakeAM() for _ in range(2)]

  def test_page_table_walkers(self):
    mm = self.d[0].mm

    for va,sz in [(0x10000, 0x3000), (0x11000, 0x300000), (0x10000, 0x2000), (0x11000, 0x5000),
                  (0x2000000, 0x2000), (0x4000000, 0x4000000), (0x38000, 0x303000), (0x8000, 0x1000)]:
      mm.map_range(vaddr=helper_va(va), size=sz, paddrs=[(va, sz)])

      ctx = PageTableTraverseContext(self.d[0], mm.root_page_table, helper_va(va))
      results = list(ctx.next(sz))

      total_covered = 0
      for tup in results:
        _offset, _pt, _pte_idx, _n_ptes, _pte_covers = tup
        total_covered += _n_ptes * _pte_covers

      assert total_covered == sz, f"Expected total coverage {total_covered} to be {sz}"

      for tup in results:
        _offset, _pt, _pte_idx, _n_ptes, _pte_covers = tup
        for i in range(_n_ptes):
          pte = helper_read_entry_components(_pt.entries[_pte_idx + i])
          assert pte['paddr'] == va + _offset + i * _pte_covers, f"Expected paddr {pte['paddr']:#x} to be {va + _offset + i * _pte_covers:#x}"
          assert pte['valid'] == 1

      mm.unmap_range(helper_va(va), sz)

      for tup in results:
        _offset, _pt, _pte_idx, _n_ptes, _pte_covers = tup
        for i in range(_n_ptes):
          pte = helper_read_entry_components(_pt.entries[_pte_idx + i])
          assert pte['paddr'] == 0
          assert pte['valid'] == 0

  def test_map_notaligned(self):
    mm0 = self.d[0].mm

    for (va1,sz1),(va2,sz2) in [((0x10000, (0x1000)), (0x11000, (2 << 20)))]:
      mm0.map_range(vaddr=helper_va(va1), size=sz1, paddrs=[(va1, sz1)])
      mm0.map_range(vaddr=helper_va(va2), size=sz2, paddrs=[(va2, sz2)])
      mm0.unmap_range(helper_va(va2), sz2)
      mm0.unmap_range(helper_va(va1), sz1)

  def test_double_map(self):
    mm0 = self.d[0].mm

    for va,sz in [(0x10000, 0x3000), (0x1000000, 0x1000000), (0x12000, 0x4000)]:
      exteranl_va = helper_va(va)
      mm0.map_range(vaddr=exteranl_va, size=sz, paddrs=[(va, sz)])

      with self.assertRaises(AssertionError):
        mm0.map_range(vaddr=exteranl_va, size=0x1000, paddrs=[(va, sz)])

      with self.assertRaises(AssertionError):
        mm0.map_range(vaddr=exteranl_va, size=0x100000, paddrs=[(va, sz)])

      with self.assertRaises(AssertionError):
        mm0.map_range(vaddr=exteranl_va + 0x1000, size=0x1000, paddrs=[(va, sz)])

      with self.assertRaises(AssertionError):
        mm0.map_range(vaddr=exteranl_va + 0x2000, size=0x100000, paddrs=[(va, sz)])

      mm0.unmap_range(vaddr=exteranl_va, size=sz)

      # Finally can map and check paddrs
      mm0.map_range(vaddr=exteranl_va + 0x2000, size=0x100000, paddrs=[(0xdead0000, 0x1000), (0xdead1000, 0xff000)])

      ctx = PageTableTraverseContext(self.d[0], mm0.root_page_table, exteranl_va + 0x2000)
      for tup in ctx.next(0x100000):
        _offset, _pt, _pte_idx, _n_ptes, _pte_covers = tup
        for i in range(_n_ptes):
          pte = helper_read_entry_components(_pt.entries[_pte_idx + i])
          assert pte['paddr'] == 0xdead0000 + _offset + i * _pte_covers, f"paddr {pte['paddr']:#x} not {0xdead0000 + _offset + i * _pte_covers:#x}"
          assert pte['valid'] == 1

      mm0.unmap_range(vaddr=exteranl_va + 0x2000, size=0x100000)

  def test_try_bad_unmap(self):
    mm0 = self.d[0].mm

    with self.assertRaises(AssertionError):
      mm0.unmap_range(helper_va(0x10000), 0x3000)

    mm0.map_range(helper_va(0x10000), 0x3000, paddrs=[(0x10000, 0x3000)])
    mm0.unmap_range(helper_va(0x10000), 0x3000)

    with self.assertRaises(AssertionError):
      mm0.unmap_range(helper_va(0x10000), 0x3000)

    mm0.map_range(helper_va(0x10000), 0x3000, paddrs=[(0x10000, 0x3000)])
    mm0.unmap_range(helper_va(0x10000), 0x3000)

    with self.assertRaises(AssertionError):
      mm0.unmap_range(helper_va(0x10000), 0x3000)

  def test_free_pt(self):
    mm0 = self.d[0].mm

    # offset from start
    for off in [0, 0x3000, 0x10000]:
      mm0.map_range(helper_va(0x1000000) + off, (2 << 20)  - off, paddrs=[(0x10000, 0x1000)] * (512 - off // 0x1000))
      mm0.unmap_range(helper_va(0x1000000) + off, (2 << 20) - off)
      mm0.map_range(helper_va(0x1000000), 2 << 20, paddrs=[(0x10000, 2 << 20)])
      mm0.unmap_range(helper_va(0x1000000), 2 << 20)

    # offset from end
    for off in [0x1000, 0x20000]:
      mm0.map_range(helper_va(0x1000000), (2 << 20) - off, paddrs=[(0x10000, 0x1000)] * (512 - off // 0x1000))
      mm0.unmap_range(helper_va(0x1000000), (2 << 20) - off)
      mm0.map_range(helper_va(0x1000000), 2 << 20, paddrs=[(0x10000, 2 << 20)])
      mm0.unmap_range(helper_va(0x1000000), 2 << 20)

  def test_frag_size(self):
    mm0 = self.d[0].mm

    def must_cover_checker(va, sz):
      ans = (1 << (mm0._frag_size(va=va, sz=sz, must_cover=True) + 12))
      assert va % ans == 0 and sz % ans == 0 and (va % (2 * ans) != 0 or sz % (2 * ans) != 0), f"va {va:#x} sz {sz:#x} ans {ans:#x}"

    def not_cover_checker(va, sz):
      ans = (1 << (mm0._frag_size(va=va, sz=sz, must_cover=False) + 12))
      assert va % ans == 0 and ans <= sz and (va % (2 * ans) != 0 or (2 * ans) > sz), f"va {va:#x} sz {sz:#x} ans {ans:#x}"

    for va, sz in [(0x0, 0x1000), (0x1000, 0x2000), (0x1000, 0x3000), (0x2000, 0x2000), (0x4000, 0x8000), (0x8000, 0x4000), (0x10000, 0x4000),
                   (0x0, 0x4000), (0x10000, 0x4000), (0x10000, 0x40000), (0x10001000, 0x40000), (0x100001000, 0x3000)]:
      must_cover_checker(va, sz)
      not_cover_checker(va, sz)

if __name__ == "__main__":
  unittest.main()
