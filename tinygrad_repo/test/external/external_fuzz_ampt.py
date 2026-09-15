import random
from typing import Optional
from tinygrad.helpers import round_up
from tinygrad.runtime.support.am.amdev import AMPageTableTraverseContext
from test.external.external_test_am import helper_read_entry_components, FakeAM

class AMPTFuzzer:
  def __init__(self, total_size):
    self.total_size = total_size
    self.alloc_payload = 0
    self.d = FakeAM()

    self.allocations: dict[int, tuple[int, int]] = {} # ptr -> (size, pattern)

    self.min_alloc_size = 0x1000
    self.max_alloc_size = int(total_size * 0.1)
    self.alloc_probability = 0.7

  def generate_pattern(self, ptr: int, size: int) -> int: return random.randint(0, 0xff)

  def fill_memory(self, va, size: int, pattern: int):
    ctx = AMPageTableTraverseContext(self.d, self.d.mm.root_page_table, va.va_addr)
    pages = list(ctx.next(size))

    for _offset, _pt, _pte_idx, _n_ptes, _pte_covers in pages:
      _vaddr = va.va_addr + _offset

      for i in range(_n_ptes):
        pte = helper_read_entry_components(_pt.entries[_pte_idx + i])
        self.d.vram[pte['paddr']] = pattern # Mark this page
        assert pte['valid'] == 1

        # If page has contiguous fragment, all range should be this valid memory
        frags_cnt = pte['fragment']
        contig_range = (1 << (frags_cnt + 12))
        start_vaddr = _vaddr & ~(contig_range - 1)
        start_paddr = pte['paddr'] - (_vaddr - start_vaddr)
        contig_ptes = contig_range // _pte_covers
        assert contig_ptes > 0

        ctx = AMPageTableTraverseContext(self.d, self.d.mm.root_page_table, start_vaddr)
        frags_l = list(ctx.next(contig_range))
        for f_offset, f_pt, f_pte_idx, f_n_ptes, f_pte_covers in frags_l:
          for j in range(f_n_ptes):
            f_pte = helper_read_entry_components(f_pt.entries[f_pte_idx + j])
            assert f_pte['valid'] == 1
            assert f_pte['paddr'] == start_paddr+f_offset+j*f_pte_covers, f"paddr {f_pte['paddr']:#x} not {start_paddr+f_offset+j*f_pte_covers:#x}"

        _vaddr += _pte_covers
        _offset += _pte_covers

    return pages

  def verify_memory(self, pages, pattern: int) -> bool:
    for _offset, _pt, _pte_idx, _n_ptes, _pte_covers in pages:
      for i in range(_n_ptes):
        pte = helper_read_entry_components(_pt.entries[_pte_idx + i])
        if self.d.vram[pte['paddr']] != pattern: return False
        if pte['valid'] == 0: return False

    return True

  def random_alloc(self) -> Optional[int]:
    if self.total_size - self.alloc_payload < self.min_alloc_size: return None

    size = random.randint(self.min_alloc_size, min(self.max_alloc_size, self.total_size - self.alloc_payload))
    size = round_up(size, (2 << 20) if size > (4 << 20) else (4 << 10))

    try: ptr = self.d.mm.valloc(size)
    except MemoryError:
      print(f"Failed to allocate {size} bytes. Payload size is {self.alloc_payload}, so fragmenation is {(size / self.total_size)*100.0:.2f}%")
      return None

    pattern = self.generate_pattern(ptr, size)
    pages = self.fill_memory(ptr, size, pattern)
    self.allocations[ptr.va_addr] = (size, pattern, pages, ptr)
    self.alloc_payload += size
    print(f"Allocated {size} bytes at {ptr.va_addr:x}, pattern: {pattern:02x}")
    return ptr

  def random_free(self) -> bool:
    if not self.allocations: return False

    ptr = random.choice(list(self.allocations.keys()))
    size, pattern, pages, vm = self.allocations[ptr]

    # Verify pattern before freeing
    if not self.verify_memory(pages, pattern):
      raise RuntimeError(f"Memory corruption detected at {vm.va_addr:x}!")

    print(f"Freeing {size} bytes at {vm.va_addr:x}, pattern verified: {pattern:02x}")
    self.alloc_payload -= size
    self.d.mm.vfree(vm)
    del self.allocations[ptr]
    return True

  def run(self):
    for i in range(10000000):
      if (random.random() < self.alloc_probability or not self.allocations): self.random_alloc()
      else: self.random_free()

    print("\nCleaning up remaining allocations...")
    while self.allocations: self.random_free()

    print("Fuzzing completed successfully!")

if __name__ == "__main__":
  fuzzer = AMPTFuzzer(1 << 30)
  fuzzer.run()
